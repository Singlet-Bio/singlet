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
//  * two or more donors, or no demux: every site, since donor assignment needs each cell's reference evidence.
// The VCFs (donors_allsite.vcf.gz, cells.vcf.gz) are written from the full member in memory, so the rule changes no genotype.
// store.json beside the matrices records the rule and the donor count, so a reader knows which form a sample has.

#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "genotype_export.h"

namespace singlet { namespace allsite {

constexpr uint32_t kStoreMinAlt = 2;   // one donor: ALT molecules over the called cells for a site to keep per-cell entries

struct StorePlan {
    std::string rule;                 // "single-donor-alt-sites", "multi-donor-all-sites" or "no-demux-all-sites"
    int n_donors = 0;                 // demux's donor count (0: demux did not run)
    uint32_t cells = 0;               // called cells kept
    uint32_t sites = 0, sites_per_cell = 0;
    uint32_t sites_genotype_variant = 0;   // one donor: the sites a 0/1 or 1/1 call would keep (reported beside the ALT rule)
    std::vector<uint8_t> keep_col;    // per barcode: a called cell
    std::vector<uint8_t> keep_row;    // per site: per-cell entries stored
};

template <class CSC>
inline StorePlan plan_store(const CSC& ad, const CSC& dp, const std::vector<uint32_t>& called, int n_donors) {
    StorePlan p;
    p.n_donors = n_donors;
    p.sites = dp.nrows;
    p.keep_col.assign(dp.ncols, 0);
    for (uint32_t c : called) if (c < dp.ncols && !p.keep_col[c]) { p.keep_col[c] = 1; ++p.cells; }
    p.keep_row.assign(dp.nrows, 1);
    if (n_donors == 1) {
        p.rule = "single-donor-alt-sites";
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
