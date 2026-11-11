// SPDX-License-Identifier: MIT
// include/singlet/pileup/storage_safeguards.h
//
// Private outputs that keep a later revision of the cell calls from needing the reads (owner decisions D-137..D-141: calls are metadata,
// never removals). Everything here is derived from the counts both engines already hold, so the CPU and GPU engines write identical bytes.
//
//  * mt_allele_counts_uncalled.tsv.gz: the chrM allele counts (both strands) of every barcode that is not a called cell, pooled. Empty
//    droplets carry much of a library's mitochondrial RNA; without their allele profile an ambient contribution to a heteroplasmy call
//    can never be estimated.
//  * The barcode set of the two private members that used to hold called cells only (private/mt_allele_counts*, private/allsite/*) is the
//    kept universe (every matrix column, barcodes with >= 5 reads), or, in mode 2, the uncalled part goes to files of its own and the
//    called-cell members stay as they were (uncalled_mode()).
//  * private/donor/donor<k>_callable.rle.gz: where each donor's depth supports a confident genotype, so that the donor's variant file can
//    say "matches the reference" instead of "not seen" (genotype::donor_gt_index is the rule of the VCF's GT column).

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <zlib.h>

#include "allsite_store.h"
#include "donor_demux.h"
#include "genotype_export.h"

namespace singlet { namespace safeguards {

// Barcode set of the private chrM allele-count and all-site members.
//   0  called cells only (the layout before this change)
//   1  the kept universe in the same members (uncalled columns after the called ones in the all-site store)
//   2  called cells in the same members, as before; the uncalled barcodes of the all-site member in separate files
//      (allsite/allele_delta_uncalled.bin, allsite/cells_uncalled.tsv.gz); the uncalled mt counts only pooled
//   3  as 2, and the per-barcode uncalled mt counts too (mt_allele_counts_uncalled*.1pz)
// SINGLET_UNCALLED_MODE overrides the default (measurement runs).
constexpr int kDefaultUncalledMode = 2;   // measured on the gate samples: mode 1 grew the two members by 63-207% (mt) and 65-81% (all-site), far over 15%
inline int uncalled_mode() {
    if (const char* e = std::getenv("SINGLET_UNCALLED_MODE")) {
        const int m = std::atoi(e);
        if (m >= 0 && m <= 3) return m;
    }
    return kDefaultUncalledMode;
}

// Columns of `a` whose flag in `mask` equals `want` keep their entries; the others become empty columns (every barcode stays a column).
template <class CSC>
inline CSC select_columns(const CSC& a, const std::vector<uint8_t>& mask, uint8_t want) {
    CSC o;
    o.nrows = a.nrows; o.ncols = a.ncols; o.indptr.assign(static_cast<size_t>(a.ncols) + 1, 0);
    for (uint32_t j = 0; j < a.ncols; ++j) {
        const bool keep = (j < mask.size() ? mask[j] : 0) == want;
        if (keep)
            for (int32_t k = a.indptr[j]; k < a.indptr[j + 1]; ++k) {
                o.indices.push_back(a.indices[k]);
                o.data.push_back(a.data[k]);
            }
        o.indptr[j + 1] = static_cast<int32_t>(o.indices.size());
    }
    return o;
}

// The pooled chrM allele counts of the barcodes with is_cell == 0, as one row per covered position:
//   pos  A  C  G  T  revA  revC  revG  revT        (1-based position; all-reads counts, then the reverse-strand subset; forward = all - rev)
// rows ascend by position, positions without a count are left out. Layout of the rows: feature = pos0 * 4 + base (mt::mt_feature_names).
// Returns the number of barcodes pooled, or -1 on a write failure.
template <class CSC>
inline long long write_mt_uncalled_pooled(const std::string& path, const CSC& tot, const CSC* rev, const std::vector<uint8_t>& is_cell,
                                          uint64_t n_called) {
    std::vector<uint64_t> T(tot.nrows, 0), R(tot.nrows, 0);
    long long pooled = 0;
    uint64_t entries = 0;
    for (uint32_t j = 0; j < tot.ncols; ++j) {
        if (j < is_cell.size() && is_cell[j]) continue;
        ++pooled;
        for (int32_t k = tot.indptr[j]; k < tot.indptr[j + 1]; ++k) { T[static_cast<size_t>(tot.indices[k])] += tot.data[k]; ++entries; }
        if (rev && j < rev->ncols)
            for (int32_t k = rev->indptr[j]; k < rev->indptr[j + 1]; ++k) R[static_cast<size_t>(rev->indices[k])] += rev->data[k];
    }
    const std::string tmp = path + ".tmp";
    gzFile f = gzopen(tmp.c_str(), "wb6");
    if (!f) return -1;
    std::string out;
    out += "#singlet mt_allele_counts_uncalled v1: chrM allele counts pooled over the barcodes that are not called cells\n";
    out += "#barcodes_pooled=" + std::to_string(pooled) + " called_cells=" + std::to_string(n_called) + " entries_pooled=" + std::to_string(entries) + "\n";
    out += "#forward-strand count = all - rev; positions without a count are omitted\n";
    out += "pos\tA\tC\tG\tT\trevA\trevC\trevG\trevT\n";
    bool ok = gzwrite(f, out.data(), static_cast<unsigned>(out.size())) > 0;
    const size_t npos = tot.nrows / 4;
    for (size_t p = 0; ok && p < npos; ++p) {
        uint64_t s = 0;
        for (int b = 0; b < 4; ++b) s += T[p * 4 + b];
        if (!s) continue;
        out = std::to_string(p + 1);
        for (int b = 0; b < 4; ++b) { out += '\t'; out += std::to_string(T[p * 4 + b]); }
        for (int b = 0; b < 4; ++b) { out += '\t'; out += std::to_string(R[p * 4 + b]); }
        out += '\n';
        ok = gzwrite(f, out.data(), static_cast<unsigned>(out.size())) > 0;
    }
    if (gzclose(f) != Z_OK) ok = false;
    if (ok) ok = std::rename(tmp.c_str(), path.c_str()) == 0;
    if (!ok) { std::remove(tmp.c_str()); return -1; }
    return pooled;
}

// One donor's covered position: contig index into the writer's table, 0-based position, donor depths.
struct Covered { uint32_t ci; int32_t pos; uint32_t ad, dp; };

// private/donor/donor<k>_callable.rle.gz for every donor: run-length coded intervals (per contig a ">name" line, then one "gap<TAB>length" line per
// interval: start = previous end + gap, end = start + length, 0-based half open) of the positions where the donor's summed depth
// gives a confident genotype (genotype::donor_gt_index >= 0), from the panel sites (aggregated depths of the donor's cells) and from the
// all-site member's sites (r, calls; null without it). A position in both takes the deeper record. Adjacent positions are merged.
// Returns the intervals written over all donors, or -1 on a write failure.
inline long long write_callable_masks(const std::string& dir, const DemuxResult& demux, const DonorDepths& depths,
                                      const std::vector<std::string>& snp_names, const std::vector<std::string>& contig_names,
                                      const allsite::Result* r, const allsite::DonorCalls* calls, std::vector<uint64_t>* per_donor_intervals = nullptr) {
    const int K = demux.n_donors_k;
    if (K <= 0) return 0;
    std::filesystem::create_directories(dir);
    std::vector<std::string> names = contig_names;                 // the allsite tids index contig_names
    std::unordered_map<std::string, uint32_t> idx;
    for (uint32_t i = 0; i < names.size(); ++i) idx.emplace(names[i], i);
    std::vector<std::vector<Covered>> per(static_cast<size_t>(K));
    for (int j = 0; j < depths.n_covered; ++j) {
        const int32_t orig = demux.covered_to_original[static_cast<size_t>(j)];
        if (orig < 0 || orig >= static_cast<int32_t>(snp_names.size())) continue;
        std::string chrom; int32_t pos = 0; char ref = 'N', alt = 'N';
        parse_snp_name(snp_names[static_cast<size_t>(orig)], chrom, pos, ref, alt);
        if (pos <= 0) continue;
        auto it = idx.find(chrom);
        if (it == idx.end()) { it = idx.emplace(chrom, static_cast<uint32_t>(names.size())).first; names.push_back(chrom); }
        for (int k = 0; k < K; ++k) {
            const uint32_t dp = depths.dp[static_cast<size_t>(j)][static_cast<size_t>(k)];
            if (dp) per[static_cast<size_t>(k)].push_back({it->second, pos - 1, depths.ad[static_cast<size_t>(j)][static_cast<size_t>(k)], dp});
        }
    }
    if (r && calls && calls->K == K)
        for (uint32_t s = 0; s < calls->sites; ++s)
            for (int k = 0; k < K; ++k) {
                const size_t i = static_cast<size_t>(s) * K + static_cast<size_t>(k);
                if (calls->sdp[i]) per[static_cast<size_t>(k)].push_back({static_cast<uint32_t>(r->sites[s].tid), r->sites[s].pos, calls->sad[i], calls->sdp[i]});
            }
    long long total = 0;
    if (per_donor_intervals) per_donor_intervals->assign(static_cast<size_t>(K), 0);
    for (int k = 0; k < K; ++k) {
        auto& v = per[static_cast<size_t>(k)];
        std::sort(v.begin(), v.end(), [](const Covered& a, const Covered& b) {
            if (a.ci != b.ci) return a.ci < b.ci;
            if (a.pos != b.pos) return a.pos < b.pos;
            if (a.dp != b.dp) return a.dp > b.dp;
            return a.ad < b.ad;
        });
        const std::string path = dir + "/donor" + std::to_string(k) + "_callable.rle.gz";
        const std::string tmp = path + ".tmp";
        gzFile f = gzopen(tmp.c_str(), "wb6");
        if (!f) return -1;
        std::string out = "#singlet donor_callable v2 (run-length): positions where the depth summed over the donor's cells gives a confident genotype (GT rule of the donor VCFs: posterior >= 0.99)\n"
                          "#sources: the SNP panel and the sites of the all-site member; positions outside both are not recorded\n"
                          "#per contig a >name line, then gap<TAB>length lines: start = previous end + gap (0 at the contig start), end = start + length; 0-based half-open, adjacent positions merged\n";
        bool ok = gzwrite(f, out.data(), static_cast<unsigned>(out.size())) > 0;
        uint64_t n = 0;
        uint32_t cur_ci = 0, last_ci = 0; int32_t s0 = 0, e0 = 0, prev_end = 0; bool open = false, have_ci = false;
        auto flush = [&]() {
            if (!open) return;
            out.clear();
            if (!have_ci || cur_ci != last_ci) { out = ">" + names[cur_ci] + "\n"; last_ci = cur_ci; have_ci = true; prev_end = 0; }
            out += std::to_string(s0 - prev_end) + "\t" + std::to_string(e0 - s0) + "\n";
            prev_end = e0;
            if (ok) ok = gzwrite(f, out.data(), static_cast<unsigned>(out.size())) > 0;
            ++n; open = false;
        };
        for (size_t i = 0; ok && i < v.size(); ++i) {
            if (i && v[i].ci == v[i - 1].ci && v[i].pos == v[i - 1].pos) continue;   // the deeper record came first
            if (genotype::donor_gt_index(v[i].ad, v[i].dp) < 0) continue;
            if (open && v[i].ci == cur_ci && v[i].pos == e0) { ++e0; continue; }
            flush();
            cur_ci = v[i].ci; s0 = v[i].pos; e0 = v[i].pos + 1; open = true;
        }
        flush();
        if (gzclose(f) != Z_OK) ok = false;
        if (ok) ok = std::rename(tmp.c_str(), path.c_str()) == 0;
        if (!ok) { std::remove(tmp.c_str()); return -1; }
        if (per_donor_intervals) (*per_donor_intervals)[static_cast<size_t>(k)] = n;
        total += static_cast<long long>(n);
        std::vector<Covered>().swap(v);
    }
    return total;
}

}}  // namespace singlet::safeguards
