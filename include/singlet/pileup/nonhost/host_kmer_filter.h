// host_kmer_filter.h — Host genome minimizer Bloom filter for false-positive suppression
//
// Part of NONHOST-HOST-SUBTRACT (DAG task). Builds a Bloom filter from all
// minimizers extracted from a host genome FASTA using the same (k, w) parameters
// as the NonHostScreener's MinSketchIndex.  When provided to
// NonHostScreener::classify_multi_filtered(), host-shared minimizers are excluded
// before querying the viral/microbial index.
//
// Design:
//   - Canonical minimizer hashes (murmurhash64A(min(fwd,rc))) are inserted,
//     exactly matching the hash space of MinSketchEntry::kmer_hash.
//   - Double hashing: g_i(x) = (h1(x) + i*h2(x)) % n_bits, with h2 forced odd.
//   - n_hash = 7 hash probes, ~10 bits/element → FPR ≈ 1%.
//   - Human genome (3.1B bases, k=21, w=11): ~250M unique minimizers → ~313 MB filter.
//
// Binary format ("HSKBLOOM"):
//   [8B magic][4B version=1][4B k][4B w][4B n_hash][8B n_bits][bit data...]
//
// Usage:
//   // Build (one-time):
//   auto flt = HostKmerFilter::build_from_fasta("/path/genome.fa", k, w);
//   flt.save("/path/nonhost/host_21mer.bloom");
//
//   // Runtime (in run_nonhost_em_screening):
//   auto flt = HostKmerFilter::load("/path/nonhost/host_21mer.bloom");
//   auto hits = screener.classify_multi_batch_filtered(reads, flt);

#pragma once

#include "min_sketch.h"  // murmurhash64A, extract_minimizers, canonical_hash

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace singlet {
namespace nonhost {

class HostKmerFilter {
public:
    // ── Construction ────────────────────────────────────────────────────────

    HostKmerFilter() = default;

    // Construct an empty filter sized for n_elements with target FPR.
    // Uses 10 bits/element and 7 hash probes (gives FPR ≈ 1% at capacity).
    explicit HostKmerFilter(uint64_t n_elements, int k, int w)
        : k_(k), w_(w)
    {
        // m = ceil(n_bits), where n_bits = n_elements * bits_per_element
        constexpr double kBitsPerElem = 10.0;
        n_bits_ = static_cast<uint64_t>(
            static_cast<double>(n_elements) * kBitsPerElem) | 1ULL;  // keep odd
        // Round up to next multiple of 64 for aligned word access
        uint64_t n_words = (n_bits_ + 63ULL) / 64ULL;
        n_bits_ = n_words * 64ULL;
        bits_.assign(static_cast<size_t>(n_words), 0ULL);
    }

    // ── Insert / query ───────────────────────────────────────────────────────

    // Insert a canonical minimizer hash into the filter.
    void insert(uint64_t min_hash) {
        uint64_t h1 = hash1_(min_hash);
        uint64_t h2 = hash2_(min_hash);
        for (int i = 0; i < kNHash; ++i) {
            uint64_t pos = (h1 + static_cast<uint64_t>(i) * h2) % n_bits_;
            bits_[pos >> 6] |= (1ULL << (pos & 63ULL));
        }
    }

    // Returns true if min_hash was (probably) inserted.  False → definitely absent.
    // FPR ≈ 1% when loaded to design capacity.
    bool contains(uint64_t min_hash) const {
        if (bits_.empty()) return false;
        uint64_t h1 = hash1_(min_hash);
        uint64_t h2 = hash2_(min_hash);
        for (int i = 0; i < kNHash; ++i) {
            uint64_t pos = (h1 + static_cast<uint64_t>(i) * h2) % n_bits_;
            if (!(bits_[pos >> 6] & (1ULL << (pos & 63ULL)))) return false;
        }
        return true;
    }

    // Returns true if the filter was loaded (non-empty).
    bool loaded() const { return !bits_.empty(); }

    int      k()      const { return k_; }
    int      w()      const { return w_; }
    uint64_t n_bits() const { return n_bits_; }
    uint64_t n_words() const { return bits_.size(); }

    // ── Build from FASTA ─────────────────────────────────────────────────────

    // Build a HostKmerFilter from a genome FASTA (plain or gzipped via popen).
    //
    // Parameters:
    //   fasta_path   — path to genome.fa or genome.fa.gz
    //   k, w         — must match the MinSketchIndex used for viral screening
    //   n_elements   — expected number of unique minimizers (0 = auto-estimate from file size)
    //   log_progress — if true, print progress to stderr
    //   nthreads     — number of OMP threads for minimizer extraction (>1 = parallel per-contig)
    //
    // For human GRCh38 (3.1B bp, k=21, w=11):
    //   auto-estimated n_elements ≈ 282M → filter ≈ 353 MB; build time < 2 min (1 thread).
    static HostKmerFilter build_from_fasta(
            const std::string& fasta_path,
            int k = 21, int w = 11,
            uint64_t n_elements = 0,
            bool log_progress   = true,
            int  nthreads       = 1) {

        // Auto-estimate number of minimizers from file size when not provided.
        // FASTA has ~1 byte/base + small header overhead.
        if (n_elements == 0) {
            struct stat st{};
            if (::stat(fasta_path.c_str(), &st) == 0 && st.st_size > 0) {
                // ~1 minimizer per w bases × 0.9 to account for headers and newlines
                n_elements = static_cast<uint64_t>(
                    static_cast<double>(st.st_size) * 0.9 / w * 1.2);  // 20% margin
            } else {
                // Fallback: assume human genome
                n_elements = 350'000'000ULL;
            }
        }

        HostKmerFilter flt(n_elements, k, w);

        bool is_gz = fasta_path.size() >= 3 &&
                     fasta_path.substr(fasta_path.size() - 3) == ".gz";

        uint64_t seq_count   = 0;
        uint64_t total_bases = 0;

        auto process_fasta = [&](FILE* fp) {
            char line_buf[65536];
            std::string seq;
            seq.reserve(300'000'000ULL);  // pre-allocate for large chromosomes

            auto flush_seq = [&]() {
                if (seq.empty()) return;
                total_bases += seq.size();
                ++seq_count;

                if (nthreads > 1) {
                    // Parallel minimizer extraction within a long contig.
                    // Split into overlapping chunks so window boundaries don't lose k-mers.
                    const int  chunk_size    = 4'000'000;
                    const int  overlap       = k + w - 1;   // max window span
                    const int  n_chunks      = static_cast<int>(
                        (seq.size() + chunk_size - 1) / chunk_size);

                    std::vector<std::vector<uint64_t>> chunk_mins(n_chunks);

                    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
                    for (int ci = 0; ci < n_chunks; ++ci) {
                        size_t start = static_cast<size_t>(ci) * chunk_size;
                        size_t len   = std::min(
                            static_cast<size_t>(chunk_size + overlap), seq.size() - start);
                        std::string chunk = seq.substr(start, len);
                        chunk_mins[ci]    = extract_minimizers(chunk, k, w);
                    }

                    // Merge results and insert into filter (serial — bitwise OR is not atomic).
                    for (const auto& mv : chunk_mins)
                        for (uint64_t h : mv)
                            flt.insert(h);
                } else {
                    // Sequential path (default): no allocation overhead.
                    for (uint64_t h : extract_minimizers(seq, k, w))
                        flt.insert(h);
                }

                seq.clear();
                if (log_progress && seq_count % 10 == 0)
                    std::cerr << "[host-filter] " << seq_count
                              << " seqs, " << total_bases / 1'000'000 << " Mbp\n";
            };

            while (std::fgets(line_buf, sizeof(line_buf), fp)) {
                size_t len = std::strlen(line_buf);
                while (len > 0 && (line_buf[len-1] == '\n' || line_buf[len-1] == '\r')) --len;
                if (len == 0) continue;
                if (line_buf[0] == '>') {
                    flush_seq();
                } else {
                    seq.append(line_buf, len);
                }
            }
            flush_seq();
        };

        if (is_gz) {
            std::string cmd = "zcat '" + fasta_path + "'";
            FILE* fp = ::popen(cmd.c_str(), "r");
            if (!fp)
                throw std::runtime_error("HostKmerFilter: cannot run zcat for: " + fasta_path);
            process_fasta(fp);
            ::pclose(fp);
        } else {
            FILE* fp = std::fopen(fasta_path.c_str(), "r");
            if (!fp)
                throw std::runtime_error("HostKmerFilter: cannot open FASTA: " + fasta_path);
            process_fasta(fp);
            std::fclose(fp);
        }

        if (log_progress)
            std::cerr << "[host-filter] Build complete: " << seq_count << " seqs, "
                      << total_bases / 1'000'000 << " Mbp, filter="
                      << (flt.n_words() * 8) / 1'000'000 << " MB\n";

        return flt;
    }

    // ── Save / load ──────────────────────────────────────────────────────────

    // Write the filter to a binary file.
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("HostKmerFilter::save: cannot open: " + path);

        BloomHeader hdr{};
        magic_write(hdr.magic);
        hdr.version = 1;
        hdr.k       = static_cast<uint32_t>(k_);
        hdr.w       = static_cast<uint32_t>(w_);
        hdr.n_hash  = static_cast<uint32_t>(kNHash);
        hdr.n_bits  = n_bits_;

        f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        f.write(reinterpret_cast<const char*>(bits_.data()),
                static_cast<std::streamsize>(bits_.size() * sizeof(uint64_t)));

        if (!f)
            throw std::runtime_error("HostKmerFilter::save: write error: " + path);
    }

    // Load a filter from a binary file produced by save().
    // Throws on magic/version mismatch.
    static HostKmerFilter load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("HostKmerFilter::load: cannot open: " + path);

        BloomHeader hdr{};
        f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (!f) throw std::runtime_error("HostKmerFilter::load: truncated header: " + path);

        if (!magic_match(hdr.magic))
            throw std::runtime_error("HostKmerFilter::load: bad magic in: " + path);
        if (hdr.version != 1)
            throw std::runtime_error("HostKmerFilter::load: unsupported version: " + path);

        HostKmerFilter flt;
        flt.k_      = static_cast<int>(hdr.k);
        flt.w_      = static_cast<int>(hdr.w);
        flt.n_bits_ = hdr.n_bits;

        uint64_t n_words = (hdr.n_bits + 63ULL) / 64ULL;
        flt.bits_.resize(static_cast<size_t>(n_words));
        f.read(reinterpret_cast<char*>(flt.bits_.data()),
               static_cast<std::streamsize>(n_words * sizeof(uint64_t)));

        if (!f)
            throw std::runtime_error("HostKmerFilter::load: truncated bit data: " + path);

        return flt;
    }

    // Returns true if path exists and has the correct magic bytes (non-throwing).
    static bool exists(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        char magic[8] = {};
        f.read(magic, 8);
        return f && magic_match(magic);
    }

    // ── Utilities ────────────────────────────────────────────────────────────

    // Count how many minimizers from a read are in the host filter.
    // Returns {n_host, n_total}.
    std::pair<uint32_t, uint32_t>
    count_host_minimizers(const std::vector<uint64_t>& minimizers) const {
        uint32_t n_host = 0;
        for (uint64_t h : minimizers)
            if (contains(h)) ++n_host;
        return {n_host, static_cast<uint32_t>(minimizers.size())};
    }

    // Filter a minimizer list: keep only minimizers NOT in the host filter.
    std::vector<uint64_t>
    remove_host_minimizers(const std::vector<uint64_t>& minimizers) const {
        std::vector<uint64_t> out;
        out.reserve(minimizers.size());
        for (uint64_t h : minimizers)
            if (!contains(h)) out.push_back(h);
        return out;
    }

private:
    // ── Binary file layout ───────────────────────────────────────────────────

    static constexpr int  kNHash    = 7;   // hash probe count; 7 × 10bpe → ~1% FPR

    // Magic bytes — defined inline in the struct to avoid ODR issues in headers.
    static bool magic_match(const char* buf) {
        return buf[0]=='H' && buf[1]=='S' && buf[2]=='K' && buf[3]=='B' &&
               buf[4]=='L' && buf[5]=='O' && buf[6]=='O' && buf[7]=='M';
    }
    static void magic_write(char* buf) {
        buf[0]='H'; buf[1]='S'; buf[2]='K'; buf[3]='B';
        buf[4]='L'; buf[5]='O'; buf[6]='O'; buf[7]='M';
    }

    struct BloomHeader {
        char     magic[8];
        uint32_t version;
        uint32_t k;
        uint32_t w;
        uint32_t n_hash;
        uint64_t n_bits;
    };
    static_assert(sizeof(BloomHeader) == 32, "BloomHeader must be 32 bytes");

    // ── Data members ─────────────────────────────────────────────────────────

    std::vector<uint64_t> bits_;
    uint64_t              n_bits_ = 0;
    int                   k_      = 21;
    int                   w_      = 11;

    // ── Hash functions ───────────────────────────────────────────────────────

    // h1: double-mix of the minimizer hash (different from the original canonical_hash).
    static uint64_t hash1_(uint64_t x) {
        return murmurhash64A(x ^ 0x9e3779b97f4a7c15ULL);
    }
    // h2: uses a different constant; forced odd so that (h1 + i*h2) visits all positions.
    static uint64_t hash2_(uint64_t x) {
        return (murmurhash64A(x ^ 0x6c62272e07bb0142ULL) | 1ULL);
    }
};

} // namespace nonhost
} // namespace singlet
