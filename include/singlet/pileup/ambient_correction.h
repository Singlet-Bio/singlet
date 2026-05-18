// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: ambient_correction.h
// N11: SoupX-style ambient RNA contamination estimation and correction.
//
// Algorithm (SoupX-lite v4 — depth-based ambient estimator):
//   1. Estimate ambient profile from empty droplets (barcodes <= lower_umi).
//      Empty droplet counts are summed across all genes and normalised to
//      a probability simplex (with per-gene pseudocount 0.5).
//   2. Estimate per-cell contamination fraction from the physical droplet
//      model: every droplet captures approximately the same volume of ambient
//      soup, contributing S = median_empty_UMI molecules.  For a called cell c
//      with Nc total UMIs: rho_c = S / Nc.  High-UMI cells have low rho
//      (dominated by genuine RNA); low-UMI cells have high rho.
//   3. Clip rho_c to [0.001, 0.50].
//   4. Correct counts: corrected[g,c] = max(0, obs[g,c] - round(rho_c * N_c * ambient[g]))
//
// Motivation for switching to depth-based estimator (v4):
//   The Poisson MLE maximises LL over rho using marker genes selected by
//   soup-enrichment ratio (ambient/cell_frac).  However, for PBMC and other
//   datasets, the top ambient genes (MALAT1, MT-CO1, ribosomal) are ALSO
//   the top genuinely-expressed genes.  This makes cell_frac ≈ ambient for
//   these genes, so marker_delta ≈ 0 and the MLE has no discriminating power
//   — it converges to the rho_max cap for 90%+ of cells.  Decontaminating
//   cell_frac with a global rho doesn't help because global rho is tiny
//   (0.5%).  Without per-cluster genuine profiles (SoupX requires clustering),
//   gene-based estimators are fundamentally circular.
//
//   The depth-based estimator uses the physical model: each droplet captures
//   the same volume of ambient soup (S ≈ median_empty_UMI molecules).  Per-cell
//   rho = S / Nc, inversely proportional to cell UMI depth.  This is correct,
//   variable, and independent of gene-level circularity.
//
// Reference: Young & Bhatt (2020)  "SoupX removes ambient RNA contamination
// from droplet-based single-cell RNA sequencing data", GigaScience 9(12).
//
// Integration: call after call_cells_emptydrops() in export.h (Phase 1.75).
// Outputs: ambient_profile.tsv, ambient_contamination.tsv,
//          (optional) exon_counts_corrected.1pz/.mtx.

#include "sparse_accumulator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace singlet {

using CSCu16 = SparseAccumulator<uint16_t>::CSCMatrix;

// ── Data structures ──────────────────────────────────────────────────────────

struct AmbientProfile {
    std::vector<double> gene_fractions;    ///< per-gene ambient fraction (sums to 1)
    double   total_ambient_umi    = 0.0;   ///< total UMI across ambient barcodes
    uint32_t n_ambient_barcodes   = 0;     ///< number of empty droplets used
};

struct CellContamination {
    double   rho          = 0.0;  ///< estimated contamination fraction [0, 0.95]
    double   log_likelihood = 0.0; ///< reserved (not computed in v1)
    uint32_t n_genes_used = 0;    ///< top-ambient genes that had non-zero count in cell
};

// ── Ambient profile estimation ───────────────────────────────────────────────

/// Estimate ambient RNA profile from empty droplets (barcodes <= lower_umi).
///
/// @param counts        Full CSC gene×barcode matrix (all barcodes).
/// @param bc_totals     Per-barcode total UMI (size = counts.ncols).
/// @param lower_umi     Threshold: barcodes with count <= this define ambient pool.
/// @param n_ambient_max Maximum empty droplets to use (default 50000).
/// @return              AmbientProfile with per-gene fractions summing to 1.
inline AmbientProfile estimate_ambient(
    const CSCu16&                counts,
    const std::vector<uint64_t>& bc_totals,
    uint64_t                     lower_umi,
    int                          n_ambient_max = 50000)
{
    AmbientProfile profile;
    const uint32_t n_genes    = counts.nrows;
    const uint32_t n_barcodes = counts.ncols;
    if (n_barcodes == 0 || n_genes == 0) return profile;

    profile.gene_fractions.assign(n_genes, 0.0);

    // Collect non-zero empty barcodes; sort descending by count to take the most
    // informative (highest-count) empty droplets up to n_ambient_max.
    std::vector<uint32_t> empty_bcs;
    empty_bcs.reserve(n_barcodes);
    for (uint32_t b = 0; b < n_barcodes; ++b) {
        if (bc_totals[b] > 0 && bc_totals[b] <= lower_umi)
            empty_bcs.push_back(b);
    }
    std::sort(empty_bcs.begin(), empty_bcs.end(),
              [&](uint32_t a, uint32_t b2) { return bc_totals[a] > bc_totals[b2]; });
    if (static_cast<int>(empty_bcs.size()) > n_ambient_max)
        empty_bcs.resize(static_cast<size_t>(n_ambient_max));

    for (uint32_t b : empty_bcs) {
        profile.total_ambient_umi += static_cast<double>(bc_totals[b]);
        const int32_t start = counts.indptr[b];
        const int32_t end   = counts.indptr[b + 1];
        for (int32_t k = start; k < end; ++k)
            profile.gene_fractions[counts.indices[k]] +=
                static_cast<double>(counts.data[k]);
    }
    profile.n_ambient_barcodes = static_cast<uint32_t>(empty_bcs.size());
    if (profile.total_ambient_umi < 1.0) return profile;

    // Normalize with pseudocount 0.5 per gene to avoid zero fractions.
    double sum = 0.0;
    for (uint32_t g = 0; g < n_genes; ++g) {
        profile.gene_fractions[g] += 0.5;
        sum += profile.gene_fractions[g];
    }
    const double inv_sum = 1.0 / sum;
    for (uint32_t g = 0; g < n_genes; ++g)
        profile.gene_fractions[g] *= inv_sum;

    return profile;
}

// ── Per-cell contamination estimation ────────────────────────────────────────

/// Estimate contamination fraction ρ per cell using the depth-based model.
///
/// Physical model: every droplet (empty or cell-containing) captures the same
/// volume of ambient soup, contributing S = total_ambient_UMI / n_empty_barcodes
/// molecules on average.  For a called cell c with Nc total UMIs:
///   rho_c = S / Nc
/// High-UMI cells are dominated by genuine RNA (low rho); low-UMI cells near
/// the EmptyDrops knee have proportionally more ambient (high rho).
///
/// This avoids the circularity of marker-gene-based estimators where the
/// average cell profile ≈ ambient profile for soup-enriched genes (MALAT1,
/// MT, ribosomal), causing MLE and ratio approaches to converge to rho_max.
///
/// @param counts            Full CSC gene×barcode matrix.
/// @param ambient           Ambient profile from estimate_ambient().
/// @param cell_indices      Barcode indices of called cells.
/// @param bc_totals         Per-barcode total UMI.
/// @param n_top_genes       (unused, kept for API compat)
/// @param rho_min           Minimum contamination fraction (default 0.001).
/// @param rho_max           Maximum contamination fraction (default 0.50).
/// @param min_soup_ratio    (unused, kept for API compat)
/// @param min_ambient_frac  (unused, kept for API compat)
/// @param min_markers       (unused, kept for API compat)
/// @param pct_estimator     (unused, kept for API compat)
/// @return                  Per-cell contamination (size == cell_indices.size()).
inline std::vector<CellContamination> estimate_contamination(
    const CSCu16&                counts,
    const AmbientProfile&        ambient,
    const std::vector<uint32_t>& cell_indices,
    const std::vector<uint64_t>& bc_totals,
    uint32_t                     n_top_genes      = 200,
    double                       rho_min          = 0.001,
    double                       rho_max          = 0.50,
    double                       min_soup_ratio   = 2.0,
    double                       min_ambient_frac = 1e-5,
    uint32_t                     min_markers      = 20,
    double                       pct_estimator    = 0.05)
{
    (void)counts; (void)n_top_genes; (void)min_soup_ratio;
    (void)min_ambient_frac; (void)min_markers; (void)pct_estimator;

    std::vector<CellContamination> result(cell_indices.size());

    if (ambient.total_ambient_umi < 1.0 || ambient.n_ambient_barcodes == 0 ||
        cell_indices.empty())
        return result;

    // S = average UMI per empty droplet (ambient soup contribution per droplet)
    const double S = ambient.total_ambient_umi /
                     static_cast<double>(ambient.n_ambient_barcodes);

    for (size_t ci = 0; ci < cell_indices.size(); ++ci) {
        const uint32_t bc = cell_indices[ci];
        const double Nc = static_cast<double>(bc_totals[bc]);
        if (Nc <= 0.0) {
            result[ci].rho = rho_max;
            continue;
        }
        double rho = S / Nc;
        rho = std::max(rho_min, std::min(rho, rho_max));
        result[ci].rho          = rho;
        result[ci].n_genes_used = 0;  // depth-based, no marker genes
    }

    return result;
}

// ── Count correction ─────────────────────────────────────────────────────────

/// Subtract estimated ambient contamination from each called cell's counts.
///
/// For each cell c and gene g:
///   corrected[g, c] = max(0, obs[g, c] - round(ρ_c × N_c × ambient_fraction[g]))
///
/// Non-cell barcodes (not in cell_indices) are left unchanged.
///
/// @param counts       Full CSC gene×barcode matrix (uint16).
/// @param ambient      Estimated ambient profile.
/// @param contam       Per-cell ρ (size == cell_indices.size()).
/// @param cell_indices Barcode indices of called cells.
/// @param bc_totals    Per-barcode total UMI.
/// @return             New CSCu16 with corrected counts (same dimensions).
inline CSCu16 correct_counts(
    const CSCu16&                        counts,
    const AmbientProfile&                ambient,
    const std::vector<CellContamination>& contam,
    const std::vector<uint32_t>&         cell_indices,
    const std::vector<uint64_t>&         bc_totals)
{
    CSCu16 out;
    out.nrows   = counts.nrows;
    out.ncols   = counts.ncols;
    out.indptr.resize(static_cast<size_t>(counts.ncols) + 1, 0);

    if (ambient.total_ambient_umi < 1.0 || cell_indices.empty()) {
        out = counts;
        return out;
    }

    // Build per-barcode ρ lookup
    std::vector<double> bc_rho(counts.ncols, 0.0);
    for (size_t i = 0; i < cell_indices.size() && i < contam.size(); ++i)
        bc_rho[cell_indices[i]] = contam[i].rho;

    out.indices.reserve(counts.indices.size());
    out.data.reserve(counts.data.size());

    for (uint32_t c = 0; c < counts.ncols; ++c) {
        const double rho = bc_rho[c];
        const double Nc  = static_cast<double>(bc_totals[c]);
        const int32_t st = counts.indptr[c];
        const int32_t en = counts.indptr[c + 1];

        for (int32_t kk = st; kk < en; ++kk) {
            const uint32_t g   = static_cast<uint32_t>(counts.indices[kk]);
            const int32_t  obs = static_cast<int32_t>(counts.data[kk]);
            int32_t corrected  = obs;
            if (rho > 0.0 && Nc > 0.0) {
                const int32_t sub = static_cast<int32_t>(
                    std::round(rho * Nc * ambient.gene_fractions[g]));
                corrected = obs - sub;
            }
            if (corrected <= 0) continue;  // drop zeros and negatives
            out.indices.push_back(static_cast<int32_t>(g));
            out.data.push_back(static_cast<uint16_t>(
                std::min(corrected, static_cast<int32_t>(UINT16_MAX))));
        }
        out.indptr[c + 1] = static_cast<int32_t>(out.indices.size());
    }

    return out;
}

// ── I/O helpers ──────────────────────────────────────────────────────────────

/// Write per-gene ambient fraction profile to TSV.
/// Columns: feature, ambient_fraction
inline void write_ambient_profile(
    const std::string&              outpath,
    const AmbientProfile&           profile,
    const std::vector<std::string>& feature_names)
{
    std::ofstream out(outpath);
    if (!out.is_open()) {
        std::cerr << "[ambient] ERROR: cannot write " << outpath << "\n";
        return;
    }
    out << "feature\tambient_fraction\n";
    const uint32_t n = static_cast<uint32_t>(
        std::min(profile.gene_fractions.size(), feature_names.size()));
    for (uint32_t g = 0; g < n; ++g)
        out << feature_names[g] << '\t' << profile.gene_fractions[g] << '\n';
}

/// Write per-cell contamination estimates to TSV.
/// Columns: barcode, rho, n_genes_used
inline void write_contamination_tsv(
    const std::string&                    outpath,
    const std::vector<CellContamination>& contam,
    const std::vector<uint32_t>&          cell_indices,
    const std::vector<std::string>&       barcodes)
{
    std::ofstream out(outpath);
    if (!out.is_open()) {
        std::cerr << "[ambient] ERROR: cannot write " << outpath << "\n";
        return;
    }
    out << "barcode\trho\tn_genes_used\n";
    const size_t n = std::min(contam.size(), cell_indices.size());
    for (size_t i = 0; i < n; ++i) {
        const uint32_t bc = cell_indices[i];
        const std::string& bc_name = (bc < barcodes.size()) ? barcodes[bc] : "?";
        out << bc_name     << '\t'
            << contam[i].rho         << '\t'
            << contam[i].n_genes_used << '\n';
    }
}

} // namespace singlet
