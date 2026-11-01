// SPDX-License-Identifier: MIT
#pragma once
// Constant 5' prefix detection in cDNA (R2) reads.
//
// The pipeline clips a constant prefix (adapter / primer read-through, e.g. the 50 bp prefix of SRR34789664) from the
// cDNA read before STAR. The probe used to look at the first 25 reads of the decoded file. A .1fq is sorted by barcode,
// so those 25 reads are PCR duplicates of a few molecules and look "constant": PBMC 1k v3 lost 11 bp from every read and
// 14% of its splices. The probe now takes reads spread through the whole file (sample_fastq_spread, or the block
// sampler in singlet.cpp for FIFO decode) and calls a prefix only when
//   * at least kMinProbeReads reads were sampled,
//   * each prefix position is dominated by one base in >= 80% of reads (random cDNA: ~30-40%),
//   * the prefix holds in both halves of the sample (first and second half of the file) separately,
//   * the prefix is longer than 10 bp and is not a poly-A / poly-T run (3' inserts with a short tail).
// A 2-3 bp TSO "GGG" is normal and is never clipped (length rule).

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace singlet {
namespace fq {

namespace detail {
inline int fseek64(FILE* f, long long off) {
#ifdef _WIN32
    return _fseeki64(f, off, SEEK_SET);
#else
    return fseeko(f, static_cast<off_t>(off), SEEK_SET);
#endif
}
inline long long ftell64(FILE* f) {
#ifdef _WIN32
    return _ftelli64(f);
#else
    return static_cast<long long>(ftello(f));
#endif
}
}  // namespace detail

constexpr size_t kMinProbeReads = 1000;

struct PrefixProbeResult {
    int prefix_len = 0;        ///< bases to clip (0 = none)
    int raw_prefix_len = 0;    ///< dominated positions from 0, before the length / poly-A / halves rules
    size_t n_reads = 0;        ///< reads in the sample
    size_t max_len = 0;        ///< longest read in the sample
    std::string reason;        ///< why no clip (empty when clipping)
};

namespace detail {
inline int dominated_prefix(const std::vector<std::string>& seqs, size_t begin, size_t end, size_t max_len,
                            double threshold, int* n_polyx) {
    int len = 0, polyx = 0;
    for (size_t p = 0; p < max_len && p < 200; ++p) {
        size_t counts[256] = {};
        size_t total = 0;
        for (size_t i = begin; i < end; ++i)
            if (p < seqs[i].size()) { counts[static_cast<unsigned char>(seqs[i][p])]++; ++total; }
        if (total < 5 || total * 2 < (end - begin)) break;
        size_t best = 0; unsigned char bb = 0;
        for (int c = 0; c < 256; ++c) if (counts[c] > best) { best = counts[c]; bb = static_cast<unsigned char>(c); }
        if (static_cast<double>(best) / static_cast<double>(total) < threshold || bb == 'N') break;
        len = static_cast<int>(p) + 1;
        if (bb == 'A' || bb == 'T') ++polyx;
    }
    if (n_polyx) *n_polyx = polyx;
    return len;
}
}  // namespace detail

/// Decide on a constant 5' prefix from reads sampled across the file, in file order.
inline PrefixProbeResult detect_constant_prefix(const std::vector<std::string>& seqs, double threshold = 0.80,
                                                int min_len = 11, size_t min_reads = kMinProbeReads) {
    PrefixProbeResult r;
    r.n_reads = seqs.size();
    for (const auto& s : seqs) r.max_len = std::max(r.max_len, s.size());
    if (seqs.size() < min_reads) { r.reason = "too_few_reads"; return r; }
    if (r.max_len < 30) { r.reason = "short_reads"; return r; }
    int polyx = 0;
    const size_t half = seqs.size() / 2;
    int all = detail::dominated_prefix(seqs, 0, seqs.size(), r.max_len, threshold, &polyx);
    int a = detail::dominated_prefix(seqs, 0, half, r.max_len, threshold, nullptr);
    int b = detail::dominated_prefix(seqs, half, seqs.size(), r.max_len, threshold, nullptr);
    r.raw_prefix_len = all;
    int len = std::min(all, std::min(a, b));
    if (len < min_len) { r.reason = all >= min_len ? "not_in_both_halves" : "no_constant_prefix"; return r; }
    if (polyx * 5 > all * 4) { r.reason = "poly_a_or_t"; return r; }
    if (static_cast<int>(r.max_len) - len < 15) { r.reason = "leaves_under_15bp"; return r; }
    r.prefix_len = len;
    return r;
}

/// The probe's reads, chosen by content: the kProbeReads distinct (R1, R2) records with the smallest 64-bit hash. The same
/// records are selected whatever the input route (.1fq / .1fg2, FIFO / chunk / file decode), thread count or block order, so
/// the clip decision and n_reads are identical for the same sample. Offer every read with both mates non-empty (codes 0-4:
/// ACGTN); sequences() returns the R2 texts ordered by hash (the "halves" of detect_constant_prefix are hash halves).
constexpr size_t kProbeReads = 12000;

class ContentProbeSampler {
public:
    static uint64_t hash(const uint8_t* r1, size_t n1, const uint8_t* r2, size_t n2) {
        uint64_t h = 0x9E3779B97F4A7C15ULL ^ (n1 * 0xff51afd7ed558ccdULL) ^ (n2 << 17);
        auto mix = [&](const uint8_t* p, size_t n) {
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                uint64_t w; std::memcpy(&w, p + i, 8);
                h = (h ^ w) * 0x9FB21C651E98DF25ULL; h ^= h >> 29;
            }
            for (; i < n; ++i) { h = (h ^ p[i]) * 0x100000001B3ULL; }
            h ^= h >> 32;
        };
        mix(r1, n1); mix(r2, n2);
        h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33;
        return h;
    }
    /// thread-safe; the common case (hash above the current cut) costs one relaxed atomic load
    void offer(const uint8_t* r1, size_t n1, const uint8_t* r2, size_t n2) {
        const uint64_t h = hash(r1, n1, r2, n2);
        if (h > cut_.load(std::memory_order_relaxed)) return;
        std::lock_guard<std::mutex> g(mu_);
        static constexpr char N2A[5] = {'A', 'C', 'G', 'T', 'N'};
        std::string seq(n2, ' ');
        for (size_t j = 0; j < n2; ++j) seq[j] = N2A[std::min<uint8_t>(r2[j], 4)];
        kept_.emplace_back(h, std::move(seq));
        if (kept_.size() > 2 * kProbeReads) trim();
    }
    std::vector<std::string> sequences() {
        std::lock_guard<std::mutex> g(mu_);
        trim();
        std::vector<std::string> out;
        out.reserve(kept_.size());
        for (auto& e : kept_) out.push_back(e.second);
        return out;
    }
private:
    void trim() {
        std::sort(kept_.begin(), kept_.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        kept_.erase(std::unique(kept_.begin(), kept_.end(), [](const auto& a, const auto& b) { return a.first == b.first; }), kept_.end());   // duplicate records
        if (kept_.size() > kProbeReads) kept_.resize(kProbeReads);
        if (kept_.size() == kProbeReads) cut_.store(kept_.back().first, std::memory_order_relaxed);
    }
    std::mutex mu_;
    std::vector<std::pair<uint64_t, std::string>> kept_;
    std::atomic<uint64_t> cut_{UINT64_MAX};
};

/// Sample reads from a FASTQ file at n_pos evenly spaced byte offsets, per_pos consecutive records each.
/// Re-synchronises on a record boundary after each seek ('@' line followed two lines later by a '+' line).
inline std::vector<std::string> sample_fastq_spread(const std::string& path, size_t n_pos = 1000, size_t per_pos = 12) {
    std::vector<std::string> out;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    std::fseek(f, 0, SEEK_END);
    long long size = detail::ftell64(f);
    if (size <= 0) { std::fclose(f); return out; }
    std::vector<char> l1(1 << 16), l2(1 << 16), l3(1 << 16);
    auto getl = [&](std::vector<char>& b) -> bool {
        if (!std::fgets(b.data(), static_cast<int>(b.size()), f)) return false;
        return true;
    };
    long long last_end = -1;
    for (size_t k = 0; k < n_pos; ++k) {
        long long off = static_cast<long long>((static_cast<double>(k) / static_cast<double>(n_pos)) * size);
        if (off <= last_end) continue;  // small file: positions overlap; keep file order, no duplicates
        detail::fseek64(f, off);
        if (off > 0 && !getl(l1)) break;  // skip the partial line
        // find a header line whose third line starts with '+'
        bool synced = false;
        for (int tries = 0; tries < 8 && !synced; ++tries) {
            long long pos = detail::ftell64(f);
            if (!getl(l1)) break;
            if (l1[0] != '@') continue;
            if (!getl(l2) || !getl(l3)) break;
            if (l3[0] == '+') {
                detail::fseek64(f, pos);
                synced = true;
            } else {
                detail::fseek64(f, pos);
                getl(l1);
            }
        }
        if (!synced) continue;
        for (size_t r = 0; r < per_pos; ++r) {
            if (!getl(l1) || !getl(l2) || !getl(l3)) break;
            if (!getl(l3)) break;
            size_t len = std::strlen(l2.data());
            while (len > 0 && (l2[len - 1] == '\n' || l2[len - 1] == '\r')) --len;
            if (len > 0) out.emplace_back(l2.data(), len);
        }
        last_end = detail::ftell64(f);
    }
    std::fclose(f);
    return out;
}

}  // namespace fq
}  // namespace singlet
