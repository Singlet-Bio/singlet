// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: barcode_collapse.h
// Error-barcode collapse for whitelist-free droplet protocols (Drop-seq, Seq-Well).
//
// A bead's barcode is random, so there is no list to correct against, and two kinds of barcodes sit one substitution from a real
// cell's barcode: sequencing errors (about 0.17% of the cell's reads per neighbor) and beads of the same synthesis lot with one
// wrong base (Drop-seq's "bead synthesis errors", 7-18% of the parent's UMIs in the Macosko mixture). Each shows up as a barcode
// of its own: at 268 M reads the Macosko HEK/3T3 library had 936 of its 4,443 called "cells" one substitution from a larger
// barcode. This collapses them: a barcode is merged into its largest Hamming-1 neighbor that has at least `ratio` times its reads.
// Counting the merged barcodes together is what Drop-seq tools' barcode merging (DetectBeadSubstitutionErrors / bead synthesis
// error repair, MIT) and UMI-tools' whitelist edit-distance merging (MIT) do; neither is ported here, this is a short
// re-implementation of the same rule. Small neighbors of a kept barcode (below the discovery threshold) are aliased as well, so
// their reads join the parent instead of being dropped.
#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace singlet {

struct BarcodeCollapseResult {
    std::vector<std::pair<uint64_t, std::string>> kept;               // (reads, barcode), descending reads: the barcode universe
    std::vector<std::pair<std::string, std::string>> alias;           // (alias barcode, kept barcode)
    uint64_t n_children_in_universe = 0;                              // discovered barcodes merged into a larger neighbor
    uint64_t n_small_aliases = 0;                                     // barcodes below the discovery threshold merged into a kept barcode
};

/// `sorted`: discovered barcodes (reads, barcode), descending by reads; all the same length, A/C/G/T/N.
/// `all_counts`: reads of every distinct barcode seen (barcode -> reads), used to alias small neighbors; may be empty.
/// `min_alias_reads`: smallest barcode (in reads) that is aliased from `all_counts`.
inline BarcodeCollapseResult collapse_barcodes(const std::vector<std::pair<uint64_t, std::string>>& sorted, double ratio,
                                               const std::unordered_map<std::string, uint64_t>* all_counts, uint64_t min_alias_reads) {
    BarcodeCollapseResult R;
    static const char kB[4] = {'A', 'C', 'G', 'T'};
    std::unordered_map<std::string, size_t> idx;
    idx.reserve(sorted.size() * 2);
    for (size_t i = 0; i < sorted.size(); ++i) idx.emplace(sorted[i].second, i);
    std::vector<int64_t> parent(sorted.size(), -1);
    std::string t;
    for (size_t i = 0; i < sorted.size(); ++i) {
        const std::string& b = sorted[i].second;
        t = b;
        int64_t best = -1;
        for (size_t p = 0; p < b.size(); ++p) {
            const char orig = b[p];
            for (char c : kB) {
                if (c == orig) continue;
                t[p] = c;
                auto it = idx.find(t);
                if (it != idx.end() && it->second < i &&   // earlier in the order = at least as many reads
                    static_cast<double>(sorted[it->second].first) >= ratio * static_cast<double>(sorted[i].first) &&
                    (best < 0 || sorted[it->second].first > sorted[static_cast<size_t>(best)].first))
                    best = static_cast<int64_t>(it->second);
            }
            t[p] = orig;
        }
        parent[i] = best;
    }
    // root of each barcode (parents come earlier in the order, so one forward pass resolves chains)
    std::vector<int64_t> root(sorted.size());
    for (size_t i = 0; i < sorted.size(); ++i) root[i] = parent[i] < 0 ? static_cast<int64_t>(i) : root[static_cast<size_t>(parent[i])];
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (root[i] == static_cast<int64_t>(i)) {
            R.kept.push_back(sorted[i]);
        } else {
            R.alias.emplace_back(sorted[i].second, sorted[static_cast<size_t>(root[i])].second);
            ++R.n_children_in_universe;
        }
    }
    // small neighbors of a kept barcode
    if (all_counts) {
        std::unordered_map<std::string, bool> in_universe;
        in_universe.reserve(sorted.size() * 2);
        for (const auto& s : sorted) in_universe.emplace(s.second, true);
        for (const auto& kp : R.kept) {
            t = kp.second;
            for (size_t p = 0; p < t.size(); ++p) {
                const char orig = kp.second[p];
                for (char c : kB) {
                    if (c == orig) continue;
                    t[p] = c;
                    if (in_universe.count(t)) continue;
                    auto it = all_counts->find(t);
                    if (it != all_counts->end() && it->second >= min_alias_reads) {
                        R.alias.emplace_back(t, kp.second);
                        ++R.n_small_aliases;
                    }
                }
                t[p] = orig;
            }
        }
    }
    return R;
}

}  // namespace singlet
