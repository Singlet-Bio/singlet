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
#include <cstdint>
#include <cstdio>
#include <cstring>
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
