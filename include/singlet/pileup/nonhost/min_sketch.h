// SPDX-License-Identifier: MIT
// min_sketch.h — MinSketch: canonical k-mer minimizer index for non-host screening
//
// Part of NONHOST-SYLPH-PORT (DAG task). Implements a minimizer-based sketch
// index for rapid species-level classification of unmapped reads before
// expensive re-mapping with minimap2.
//
// Design:
//   k=21 canonical k-mer minimizers over a sliding window of w=11 k-mers.
//   Entries stored in a flat sorted vector (binary search — no hashmap).
//   Thread-safe: MinSketchIndex is read-only after finalize()/load().
//
// Binary format ("SNHSKIDX"):
//   v1: [8B magic][4B version=1][4B k][4B w][4B pad][8B n_entries]
//       + n_entries × 16B MinSketchEntry (sorted ascending by kmer_hash)
//   v2: same as v1, plus species name manifest appended after entries:
//       [4B n_species][for each: 2B name_len, name_bytes (no null terminator)]
//
// Wiring (NONHOST-UNMAPPED-CAPTURE, separate DAG task):
//   After STAR alignment, reads with SAM flag 0x4 (unmapped) are collected.
//   Their sequences are passed to NonHostScreener::classify_batch().
//   Cell barcodes are recovered from the CB tag in the STARsolo-tagged BAM.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace singlet {
namespace nonhost {

// ── Hash ────────────────────────────────────────────────────────────────────

// MurmurHash64A finalizer (fmix64). Used to hash canonical k-mer bit-vectors.
inline uint64_t murmurhash64A(uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
}

// ── Binary structures ────────────────────────────────────────────────────────

struct MinSketchEntry {       // 16 bytes per entry in the flat sorted vector
    uint64_t kmer_hash;       // canonical k-mer hash (murmurhash64A of min(fwd, rc))
    uint32_t species_id;      // which species this minimizer came from
    uint32_t pad = 0;         // reserved, always 0
};
static_assert(sizeof(MinSketchEntry) == 16, "MinSketchEntry must be 16 bytes");

struct MinSketchHeader {      // 32 bytes, written at file offset 0
    char     magic[8];        // "SNHSKIDX"
    uint32_t version;         // 1
    uint32_t k;               // k-mer length
    uint32_t w;               // window size (number of k-mers per window)
    uint32_t reserved;        // 0
    uint64_t n_entries;       // number of MinSketchEntry records following
};
static_assert(sizeof(MinSketchHeader) == 32, "MinSketchHeader must be 32 bytes");

// ── k-mer helpers ────────────────────────────────────────────────────────────

static constexpr uint64_t INVALID_KMER = UINT64_MAX;

// Encode base to 2-bit: A=0, C=1, G=2, T=3. Returns 255 for non-ACGT.
inline uint8_t base2bit(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:            return 255;
    }
}

// 2-bit packed k-mer (MSB = first base). Returns INVALID_KMER if any non-ACGT.
inline uint64_t encode_kmer(const char* seq, int k) {
    uint64_t v = 0;
    for (int i = 0; i < k; ++i) {
        uint8_t b = base2bit(seq[i]);
        if (b == 255) return INVALID_KMER;
        v = (v << 2) | b;
    }
    return v;
}

// Reverse complement of a 2-bit packed k-mer (MSB = first base, length k).
inline uint64_t rc_kmer(uint64_t fwd, int k) {
    uint64_t rc = 0;
    for (int i = 0; i < k; ++i) {
        rc = (rc << 2) | ((fwd & 3) ^ 3);   // complement of LSB 2 bits
        fwd >>= 2;
    }
    return rc;
}

// Canonical hash: murmurhash64A(min(fwd_bits, rc_bits)).
// Returns INVALID_KMER if the sequence contains non-ACGT characters.
inline uint64_t canonical_hash(const char* seq, int k) {
    uint64_t fwd = encode_kmer(seq, k);
    if (fwd == INVALID_KMER) return INVALID_KMER;
    uint64_t rc = rc_kmer(fwd, k);
    return murmurhash64A(std::min(fwd, rc));
}

// Extract minimizers from a sequence using a sliding window of w k-mers.
// Within each window, the minimizer is the k-mer with the smallest canonical hash.
// Only emits a new minimizer when it differs from the previous one (deduplicated).
// Returns empty vector for sequences shorter than k+w-1.
inline std::vector<uint64_t> extract_minimizers(const std::string& seq, int k, int w) {
    const int n_kmers = static_cast<int>(seq.size()) - k + 1;
    if (n_kmers < w) return {};

    // Pre-compute canonical hashes for all k-mers
    std::vector<uint64_t> h(n_kmers);
    for (int i = 0; i < n_kmers; ++i)
        h[i] = canonical_hash(seq.data() + i, k);

    // Slide window of size w, emit minimizer when it changes
    std::vector<uint64_t> mins;
    uint64_t prev = INVALID_KMER;
    for (int i = 0; i + w <= n_kmers; ++i) {
        uint64_t win_min = INVALID_KMER;
        for (int j = i; j < i + w; ++j)
            if (h[j] < win_min) win_min = h[j];
        if (win_min == INVALID_KMER) continue;  // window all-N
        if (win_min != prev) {
            mins.push_back(win_min);
            prev = win_min;
        }
    }
    return mins;
}

// ── MinSketchIndex ───────────────────────────────────────────────────────────

class MinSketchIndex {
public:
    MinSketchIndex() = default;
    MinSketchIndex(int k, int w) : k_(k), w_(w) {}

    // ── Construction API ────────────────────────────────────────────────────

    // Add one FASTA sequence to the index for the given species.
    // Call finalize() once after all sequences have been added.
    void add_sequence(const std::string& seq, uint32_t species_id,
                      uint32_t max_entries_per_species = 50000) {
        auto& cnt = species_count_[species_id];
        if (cnt >= max_entries_per_species) return;
        for (uint64_t h : extract_minimizers(seq, k_, w_)) {
            if (cnt >= max_entries_per_species) break;
            entries_.push_back({h, species_id, 0u});
            ++cnt;
        }
    }

    // Sort entries by kmer_hash. Must be called after all add_sequence() calls,
    // before any query() or save() calls.
    void finalize() {
        std::sort(entries_.begin(), entries_.end(),
                  [](const MinSketchEntry& a, const MinSketchEntry& b) {
                      return a.kmer_hash < b.kmer_hash;
                  });
    }

    // Build index from a FASTA file (all sequences → one species).
    static MinSketchIndex build_from_fasta(const std::string& path,
                                            uint32_t species_id,
                                            int ks = 21, int ws = 11,
                                            uint32_t max_entries_per_species = 50000) {
        MinSketchIndex idx(ks, ws);
        std::ifstream f(path);
        if (!f) throw std::runtime_error("MinSketchIndex: cannot open FASTA: " + path);
        std::string line, seq;
        bool name_set = false;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (line[0] == '>') {
                if (!seq.empty()) {
                    idx.add_sequence(seq, species_id, max_entries_per_species);
                    seq.clear();
                }
                // Capture the first FASTA header as the species name
                if (!name_set) {
                    std::string name = line.substr(1);
                    while (!name.empty() && (name.back() == '\r' || name.back() == ' '))
                        name.pop_back();
                    idx.set_species_name(species_id, name);
                    name_set = true;
                }
            } else {
                seq += line;
            }
        }
        if (!seq.empty()) idx.add_sequence(seq, species_id, max_entries_per_species);
        idx.finalize();
        return idx;
    }

    // Build index from in-memory (sequence, species_id) pairs (useful for testing).
    static MinSketchIndex build_from_sequences(
            const std::vector<std::pair<std::string, uint32_t>>& seqs,
            int ks = 21, int ws = 11,
            uint32_t max_entries_per_species = 50000) {
        MinSketchIndex idx(ks, ws);
        for (const auto& [seq, sid] : seqs)
            idx.add_sequence(seq, sid, max_entries_per_species);
        idx.finalize();
        return idx;
    }

    // Union of two finalized indexes (requires same k/w). The result does not copy
    // the entries: it holds both indexes' sorted runs (sharing their memory maps),
    // and every query adds up the runs, which gives the counts a merged index would.
    static MinSketchIndex merge(const MinSketchIndex& a, const MinSketchIndex& b) {
        if (a.k_ != b.k_ || a.w_ != b.w_)
            throw std::runtime_error("MinSketchIndex::merge: k/w mismatch");
        MinSketchIndex out(a.k_, a.w_);
        for (const MinSketchIndex* src : {&a, &b}) {
            out.blocks_.insert(out.blocks_.end(), src->blocks_.begin(), src->blocks_.end());
            if (!src->entries_.empty()) {   // owned run: share a copy
                auto blk = std::make_shared<Block>();
                blk->owned = src->entries_;
                blk->p = blk->owned.data();
                blk->n = blk->owned.size();
                out.blocks_.push_back(std::move(blk));
            }
        }
        // Merge species names: a's names take precedence; b fills in gaps
        out.species_names_ = a.species_names_;
        for (uint32_t sid = 0; sid < static_cast<uint32_t>(b.species_names_.size()); ++sid) {
            if (b.species_names_[sid].empty()) continue;
            if (sid >= static_cast<uint32_t>(out.species_names_.size()))
                out.species_names_.resize(sid + 1);
            if (out.species_names_[sid].empty())
                out.species_names_[sid] = b.species_names_[sid];
        }
        return out;
    }

    // ── Query API ────────────────────────────────────────────────────────────

    // Query pre-extracted minimizers (avoids re-extraction when screener already
    // has the minimizer list). Returns {species_id → hit_count}.
    std::unordered_map<uint32_t, uint32_t> query_minimizers(
            const std::vector<uint64_t>& mins) const {
        std::unordered_map<uint32_t, uint32_t> hits;
        for (uint64_t h : mins) {
            for_each_run([&](const MinSketchEntry* p, size_t n) {
                const MinSketchEntry* end = p + n;
                const MinSketchEntry* lo = std::lower_bound(p, end, h,
                    [](const MinSketchEntry& e, uint64_t val) {
                        return e.kmer_hash < val;
                    });
                for (auto it = lo; it != end && it->kmer_hash == h; ++it)
                    hits[it->species_id]++;
            });
        }
        return hits;
    }

    // Query a raw sequence. Returns {species_id → hit_count} for all species
    // with ≥1 minimizer hit. Thread-safe (read-only after finalize).
    std::unordered_map<uint32_t, uint32_t> query(const std::string& seq) const {
        return query_minimizers(extract_minimizers(seq, k_, w_));
    }

    // ── Batch sort-merge query (FAST path for large batches) ─────────────────

    // Batch query using a pre-sorted flat minimizer list (sort-merge linear scan).
    //
    // Performance: O(total_mins + DB_size) sequential DRAM reads, vs
    //   O(total_mins × log(DB_size)) random DRAM reads for N × query_minimizers().
    //   For 9.1M reads × 30 mins vs 1.06B-entry fungal DB: ~1000× faster.
    //
    // Contract:
    //   flat_mins: (kmer_hash, read_idx) pairs, SORTED ascending by kmer_hash.
    //   results:   pre-allocated vector of size ≥ max(read_idx)+1.
    //              On return, results[read_idx][species_id] = hit count.
    //              The caller must NOT access results[i] for out-of-range i.
    //   Thread-safe: entries_ is read-only; results must not be shared between threads.
    void batch_query_presorted(
            const std::vector<std::pair<uint64_t, uint32_t>>& flat_mins,
            std::vector<std::unordered_map<uint32_t, uint32_t>>& results) const {

        if (flat_mins.empty()) return;
        for_each_run([&](const MinSketchEntry* run, size_t n_run) {
            batch_query_run(run, n_run, flat_mins, results);
        });
    }

private:
    static void batch_query_run(
            const MinSketchEntry* entries_, size_t n_run,
            const std::vector<std::pair<uint64_t, uint32_t>>& flat_mins,
            std::vector<std::unordered_map<uint32_t, uint32_t>>& results) {
        if (n_run == 0) return;
        size_t qi = 0;
        const size_t n_q = flat_mins.size();
        size_t di = 0;
        const size_t n_d = n_run;

        while (qi < n_q && di < n_d) {
            const uint64_t qh = flat_mins[qi].first;
            const uint64_t dh = entries_[di].kmer_hash;

            if      (qh < dh) { ++qi; continue; }
            else if (qh > dh) { ++di; continue; }

            // Hash match: advance both runs to find the extent
            const size_t qi_start = qi;
            while (qi < n_q && flat_mins[qi].first == qh) ++qi;
            const size_t di_start = di;
            while (di < n_d && entries_[di].kmer_hash == qh) ++di;

            // Record all (read, species) cross-pairs for this hash value
            for (size_t qj = qi_start; qj < qi; ++qj)
                for (size_t dj = di_start; dj < di; ++dj)
                    results[flat_mins[qj].second][entries_[dj].species_id]++;
        }
    }

public:

    // ── Serialization ────────────────────────────────────────────────────────

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("MinSketchIndex::save: cannot write: " + path);
        MinSketchHeader hdr{};
        std::memcpy(hdr.magic, "SNHSKIDX", 8);
        hdr.version   = 2;  // v2: species name manifest appended after entries
        hdr.k         = static_cast<uint32_t>(k_);
        hdr.w         = static_cast<uint32_t>(w_);
        hdr.reserved  = 0;
        hdr.n_entries = static_cast<uint64_t>(size());
        f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        if (blocks_.empty()) {
            f.write(reinterpret_cast<const char*>(entries_.data()),
                    static_cast<std::streamsize>(entries_.size() * sizeof(MinSketchEntry)));
        } else {   // union of runs: write them merged, sorted by hash
            std::vector<MinSketchEntry> all(entries_);
            all.reserve(size());
            for (const auto& blk : blocks_) {
                const size_t mid = all.size();
                all.insert(all.end(), blk->p, blk->p + blk->n);
                std::inplace_merge(all.begin(), all.begin() + mid, all.end(),
                    [](const MinSketchEntry& x, const MinSketchEntry& y) { return x.kmer_hash < y.kmer_hash; });
            }
            f.write(reinterpret_cast<const char*>(all.data()),
                    static_cast<std::streamsize>(all.size() * sizeof(MinSketchEntry)));
        }
        // v2: species name manifest
        uint32_t n_sp = static_cast<uint32_t>(species_names_.size());
        f.write(reinterpret_cast<const char*>(&n_sp), sizeof(n_sp));
        for (const auto& name : species_names_) {
            uint16_t nlen = static_cast<uint16_t>(
                std::min(name.size(), static_cast<size_t>(65535)));
            f.write(reinterpret_cast<const char*>(&nlen), sizeof(nlen));
            if (nlen > 0) f.write(name.data(), nlen);
        }
        if (!f) throw std::runtime_error("MinSketchIndex::save: write error: " + path);
    }

    // Load an index. On POSIX the entries are memory-mapped read-only (MAP_SHARED), so
    // every process that loads the same file shares one copy in the page cache
    // instead of holding a private one. SINGLET_NONHOST_NO_MMAP=1 reads the file into
    // the process instead.
    static MinSketchIndex load(const std::string& path) {
#ifndef _WIN32
        const char* no_mmap = std::getenv("SINGLET_NONHOST_NO_MMAP");
        if (!(no_mmap && no_mmap[0] == '1')) {
            MinSketchIndex idx;
            if (idx.load_mapped(path)) return idx;
        }
#endif
        return load_private(path);
    }

    static MinSketchIndex load_private(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("MinSketchIndex::load: cannot open: " + path);
        MinSketchHeader hdr{};
        f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (!f || std::memcmp(hdr.magic, "SNHSKIDX", 8) != 0)
            throw std::runtime_error("MinSketchIndex::load: invalid magic");
        if (hdr.version != 1 && hdr.version != 2)
            throw std::runtime_error("MinSketchIndex::load: unsupported version " +
                                     std::to_string(hdr.version));
        MinSketchIndex idx(static_cast<int>(hdr.k), static_cast<int>(hdr.w));
        idx.entries_.resize(static_cast<size_t>(hdr.n_entries));
        f.read(reinterpret_cast<char*>(idx.entries_.data()),
               static_cast<std::streamsize>(hdr.n_entries * sizeof(MinSketchEntry)));
        if (!f) throw std::runtime_error("MinSketchIndex::load: truncated file: " + path);
        // v2: read species name manifest (v1 files have no manifest — names stay empty)
        if (hdr.version >= 2) {
            uint32_t n_sp = 0;
            f.read(reinterpret_cast<char*>(&n_sp), sizeof(n_sp));
            if (f && n_sp > 0) {
                idx.species_names_.resize(n_sp);
                for (uint32_t i = 0; i < n_sp; ++i) {
                    uint16_t nlen = 0;
                    f.read(reinterpret_cast<char*>(&nlen), sizeof(nlen));
                    if (f && nlen > 0) {
                        idx.species_names_[i].resize(nlen);
                        f.read(&idx.species_names_[i][0], nlen);
                    }
                }
            }
        }
        return idx;
    }

    // The header (k, w) and the species-name manifest of an index file, without its entries (the GPU engine holds the entries in
    // device memory; the host needs only the names). The result is empty() like an index with no entries.
    static MinSketchIndex load_names(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("MinSketchIndex::load_names: cannot open: " + path);
        MinSketchHeader hdr{};
        f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (!f || std::memcmp(hdr.magic, "SNHSKIDX", 8) != 0)
            throw std::runtime_error("MinSketchIndex::load_names: invalid magic");
        if (hdr.version != 1 && hdr.version != 2)
            throw std::runtime_error("MinSketchIndex::load_names: unsupported version " +
                                     std::to_string(hdr.version));
        MinSketchIndex idx(static_cast<int>(hdr.k), static_cast<int>(hdr.w));
        f.seekg(static_cast<std::streamoff>(sizeof(hdr) + hdr.n_entries * sizeof(MinSketchEntry)));
        if (!f) throw std::runtime_error("MinSketchIndex::load_names: truncated file: " + path);
        if (hdr.version >= 2) {
            uint32_t n_sp = 0;
            f.read(reinterpret_cast<char*>(&n_sp), sizeof(n_sp));
            if (f && n_sp > 0) {
                idx.species_names_.resize(n_sp);
                for (uint32_t i = 0; i < n_sp; ++i) {
                    uint16_t nlen = 0;
                    f.read(reinterpret_cast<char*>(&nlen), sizeof(nlen));
                    if (f && nlen > 0) {
                        idx.species_names_[i].resize(nlen);
                        f.read(&idx.species_names_[i][0], nlen);
                    }
                }
            }
        }
        return idx;
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    size_t size()  const {
        size_t n = entries_.size();
        for (const auto& blk : blocks_) n += blk->n;
        return n;
    }
    int    k()     const { return k_; }
    int    w()     const { return w_; }
    bool   empty() const { return size() == 0; }

    // Species name manifest (populated from FASTA headers during build)
    void set_species_name(uint32_t species_id, const std::string& name) {
        if (species_id >= static_cast<uint32_t>(species_names_.size()))
            species_names_.resize(species_id + 1);
        species_names_[species_id] = name;
    }
    const std::string& species_name(uint32_t species_id) const {
        static const std::string empty_name;
        if (species_id >= static_cast<uint32_t>(species_names_.size())) return empty_name;
        return species_names_[species_id];
    }
    size_t n_species_names() const { return species_names_.size(); }

private:
    // A sorted run of entries that is not in entries_: a read-only memory map of an
    // index file, or an owned copy. Shared between a union index and its sources.
    struct Block {
        const MinSketchEntry* p = nullptr;
        size_t n = 0;
        std::vector<MinSketchEntry> owned;
        void* map_base = nullptr;
        size_t map_len = 0;
        Block() = default;
        Block(const Block&) = delete;
        Block& operator=(const Block&) = delete;
        ~Block() {
#ifndef _WIN32
            if (map_base) munmap(map_base, map_len);
#endif
        }
    };

    template <class Fn>
    void for_each_run(Fn&& fn) const {
        if (!entries_.empty()) fn(entries_.data(), entries_.size());
        for (const auto& blk : blocks_) fn(blk->p, blk->n);
    }

#ifndef _WIN32
    // Map `path` read-only. Returns false (caller reads the file instead) if it cannot
    // be mapped; throws for a file that is not an index.
    bool load_mapped(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat sb{};
        if (::fstat(fd, &sb) != 0 || static_cast<size_t>(sb.st_size) < sizeof(MinSketchHeader)) {
            ::close(fd);
            return false;
        }
        const size_t len = static_cast<size_t>(sb.st_size);
        void* base = ::mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (base == MAP_FAILED) return false;
        auto blk = std::make_shared<Block>();
        blk->map_base = base;
        blk->map_len = len;   // unmapped by ~Block, also on the throws below
        MinSketchHeader hdr{};
        std::memcpy(&hdr, base, sizeof(hdr));
        if (std::memcmp(hdr.magic, "SNHSKIDX", 8) != 0)
            throw std::runtime_error("MinSketchIndex::load: invalid magic");
        if (hdr.version != 1 && hdr.version != 2)
            throw std::runtime_error("MinSketchIndex::load: unsupported version " +
                                     std::to_string(hdr.version));
        const size_t body = static_cast<size_t>(hdr.n_entries) * sizeof(MinSketchEntry);
        if (sizeof(MinSketchHeader) + body > len)
            throw std::runtime_error("MinSketchIndex::load: truncated file: " + path);
        k_ = static_cast<int>(hdr.k);
        w_ = static_cast<int>(hdr.w);
        blk->p = reinterpret_cast<const MinSketchEntry*>(static_cast<const char*>(base) + sizeof(MinSketchHeader));
        blk->n = static_cast<size_t>(hdr.n_entries);
        blocks_.push_back(std::move(blk));
        if (hdr.version >= 2) {   // species name manifest follows the entries
            const char* q = static_cast<const char*>(base) + sizeof(MinSketchHeader) + body;
            const char* end = static_cast<const char*>(base) + len;
            uint32_t n_sp = 0;
            if (q + sizeof(n_sp) <= end) {
                std::memcpy(&n_sp, q, sizeof(n_sp));
                q += sizeof(n_sp);
                if (n_sp > 0) species_names_.resize(n_sp);
                for (uint32_t i = 0; i < n_sp && q + sizeof(uint16_t) <= end; ++i) {
                    uint16_t nlen = 0;
                    std::memcpy(&nlen, q, sizeof(nlen));
                    q += sizeof(nlen);
                    if (q + nlen > end) break;
                    species_names_[i].assign(q, nlen);
                    q += nlen;
                }
            }
        }
        return true;
    }
#endif

    std::vector<std::shared_ptr<Block>>   blocks_;        // loaded / union runs (each sorted)
    std::vector<MinSketchEntry>           entries_;       // sorted by kmer_hash
    std::unordered_map<uint32_t,uint32_t> species_count_; // per-species entry counter
    std::vector<std::string>              species_names_; // species_id → name from FASTA header
    int k_ = 21;
    int w_ = 11;
};

} // namespace nonhost
} // namespace singlet
