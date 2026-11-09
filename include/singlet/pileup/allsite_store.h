// SPDX-License-Identifier: MIT
// include/singlet/pileup/allsite_store.h
//
// D-101 storage rule for the all-site member (private/allsite/, owner-only). The engine computes the member over every barcode and every
// site with alternate-allele evidence (allsite.h); what it stores is smaller:
//  * every sample: the called cells only (the other barcodes' columns are empty);
//  * one donor (demux ran with one donor): per-cell entries only at the sites with ALT evidence in the called cells (summed ALT molecules
//    >= 2), the sites allele-specific expression and allelic copy-number methods use. A genotype call would drop monoallelic or strongly
//    imbalanced expression (imprinting, X-inactivation clones), which is the ASE signal, so the rule counts ALT molecules instead. The rows
//    of the other sites stay, without per-cell entries, so row i is still line i of sites.tsv.gz, whose summed DP and AD cover every site;
//  * two or more donors, or no demux: every site, since donor assignment needs each cell's reference evidence. The column pointers of the
//    matrices are int32, so when the called cells' merged DP entries over every site reach INT32_MAX, the ALT-evidence rule of the single
//    donor applies instead ("multi-donor-alt-sites", "no-demux-alt-sites"); when even those do not fit, no member is stored and
//    provenance.json says why.
// The rule is applied while the matrices are built (build_member), before any entry is indexed, so the int32 limit is checked on the
// matrix that would be stored. The sites kept always include those a donor genotype call needs (variant_rows), so the VCFs, which are
// written from the matrices in memory, never change.
// The VCFs (donors_allsite.vcf.gz, cells.vcf.gz) are written from the full member in memory, so the rule changes no genotype.
// store.json beside the matrices records the rule and the donor count, so a reader knows which form a sample has.

#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "allsite.h"
#include "genotype_export.h"

namespace singlet { namespace allsite {

constexpr uint32_t kStoreMinAlt = 2;   // one donor: ALT molecules over the called cells for a site to keep per-cell entries

struct StorePlan {
    std::string rule;                 // "single-donor-alt-sites", "multi-donor-all-sites", "no-demux-all-sites", or the "-alt-sites" fallbacks
    int n_donors = 0;                 // demux's donor count (0: demux did not run)
    uint32_t cells = 0;               // called cells kept
    uint32_t sites = 0, sites_per_cell = 0;
    uint32_t sites_genotype_variant = 0;   // one donor: the sites a 0/1 or 1/1 call would keep (reported beside the ALT rule)
    std::vector<uint8_t> keep_col;    // per barcode: a called cell
    std::vector<uint8_t> keep_row;    // per site: per-cell entries stored
};

template <class CSC>
inline StorePlan plan_store(const CSC& ad, const CSC& dp, const std::vector<uint32_t>& called, int n_donors, bool alt_rule = false) {
    StorePlan p;
    p.n_donors = n_donors;
    p.sites = dp.nrows;
    p.keep_col.assign(dp.ncols, 0);
    for (uint32_t c : called) if (c < dp.ncols && !p.keep_col[c]) { p.keep_col[c] = 1; ++p.cells; }
    p.keep_row.assign(dp.nrows, 1);
    if (n_donors == 1 || alt_rule) {
        p.rule = n_donors == 1 ? "single-donor-alt-sites" : n_donors >= 2 ? "multi-donor-alt-sites" : "no-demux-alt-sites";
        std::vector<uint32_t> sad(dp.nrows, 0), sdp(dp.nrows, 0);
        for (uint32_t c = 0; c < dp.ncols; ++c) {
            if (!p.keep_col[c]) continue;
            for (int32_t i = dp.indptr[c]; i < dp.indptr[c + 1]; ++i) sdp[static_cast<size_t>(dp.indices[i])] += dp.data[i];
            if (c < ad.ncols) for (int32_t i = ad.indptr[c]; i < ad.indptr[c + 1]; ++i) sad[static_cast<size_t>(ad.indices[i])] += ad.data[i];
        }
        for (uint32_t s = 0; s < dp.nrows; ++s) {
            p.keep_row[s] = sad[s] >= kStoreMinAlt ? 1 : 0;
            if (genotype::donor_gt_index(sad[s], sdp[s]) >= 1) ++p.sites_genotype_variant;
        }
    } else {
        p.rule = n_donors >= 2 ? "multi-donor-all-sites" : "no-demux-all-sites";
    }
    for (uint8_t k : p.keep_row) p.sites_per_cell += k;
    return p;
}

// The rows the ALT-evidence rule keeps, from the per-site sums over the called cells: summed ALT molecules >= kStoreMinAlt, or a donor
// genotype call (0/1 or 1/1) from those sums, which is what the VCFs read.
inline std::vector<uint8_t> variant_rows(const std::vector<uint64_t>& kept_ad, const std::vector<uint64_t>& kept_dp) {
    std::vector<uint8_t> rows(kept_dp.size(), 0);
    for (size_t s = 0; s < rows.size(); ++s) {
        const uint64_t a = s < kept_ad.size() ? kept_ad[s] : 0;
        const uint32_t ad = static_cast<uint32_t>(std::min<uint64_t>(a, UINT32_MAX)), dp = static_cast<uint32_t>(std::min<uint64_t>(kept_dp[s], UINT32_MAX));
        rows[s] = (ad >= kStoreMinAlt || genotype::donor_gt_index(ad, dp) >= 1) ? 1 : 0;
    }
    return rows;
}

struct BuildReport {
    bool built = false;
    bool alt_rule = false;            // the ALT-evidence rows were applied (always with one donor; a fallback otherwise)
    std::string reason;               // when not built: why
    uint64_t ad_entries = 0, dp_entries = 0;   // merged entries of the matrices built, or of the last attempt
};

// Builds the member's matrices from a Params::defer_build run for the barcodes marked in keep_cols (null: all) under the storage rule
// above, given demux's donor count. False, with the reason in rep, when the entries do not fit the int32 column pointers.
inline bool build_member(Result& r, const std::vector<uint8_t>* keep_cols, int n_donors, BuildReport& rep) {
    rep = BuildReport();
    r.prepare(keep_cols);
    const std::vector<uint8_t> alt = variant_rows(r.kept_ad, r.kept_dp);
    auto drop = [&](const std::string& why) { r.release(); r.ad = {}; r.dp = {}; rep.reason = why; return false; };
    if (!r.emit(true, nullptr, rep.ad_entries)) return drop("AD entries exceed the int32 column pointers");
    rep.alt_rule = n_donors == 1;
    if (!r.emit(false, rep.alt_rule ? &alt : nullptr, rep.dp_entries)) {
        if (rep.alt_rule) return drop("DP entries at the ALT-evidence sites exceed the int32 column pointers");
        rep.alt_rule = true;
        if (!r.emit(false, &alt, rep.dp_entries)) return drop("DP entries at the ALT-evidence sites exceed the int32 column pointers");
    }
    r.release();
    rep.built = true;
    return true;
}

// The entries of m in kept columns and kept rows; same shape, so barcodes and sites keep their indices.
template <class CSC>
inline CSC restrict_store(const CSC& m, const StorePlan& p) {
    CSC o;
    o.nrows = m.nrows; o.ncols = m.ncols;
    o.indptr.assign(static_cast<size_t>(m.ncols) + 1, 0);
    for (uint32_t c = 0; c < m.ncols; ++c) {
        if (c < p.keep_col.size() && p.keep_col[c])
            for (int32_t i = m.indptr[c]; i < m.indptr[c + 1]; ++i) {
                const int32_t r = m.indices[i];
                if (static_cast<size_t>(r) < p.keep_row.size() && p.keep_row[r]) { o.indices.push_back(r); o.data.push_back(m.data[i]); }
            }
        o.indptr[c + 1] = static_cast<int32_t>(o.indices.size());
    }
    return o;
}

inline bool write_store_json(const std::string& path, const StorePlan& p, size_t ad_entries, size_t dp_entries) {
    std::ofstream f(path);
    if (!f) return false;
    f << "{\n  \"rule\": \"" << p.rule << "\",\n  \"n_donors\": " << p.n_donors << ",\n  \"called_cells\": " << p.cells
      << ",\n  \"sites\": " << p.sites << ",\n  \"sites_with_per_cell_entries\": " << p.sites_per_cell
      << ",\n  \"sites_genotype_variant\": " << p.sites_genotype_variant
      << ",\n  \"ad_entries\": " << ad_entries << ",\n  \"dp_entries\": " << dp_entries
      << ",\n  \"note\": \"sites.tsv.gz column per_cell marks the rows with per-cell entries; its dp and ad sum every barcode\"\n}\n";
    return static_cast<bool>(f);
}

}}  // namespace singlet::allsite
