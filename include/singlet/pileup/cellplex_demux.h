// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: cellplex_demux.h
// G-CELLPLEX: 10x CellPlex / Cell Multiplexing Oligo (CMO) demultiplexing.
//
// Algorithm:
//   1. CLR normalisation: CLR(x_i) = log(x_i+1) - mean(log(x_j+1)) per cell.
//      Identical to HtoDemux CLR.
//   2. Per-CMO threshold: for each CMO, find threshold using Otsu's method on
//      the 1D CLR distribution across all cells.  Otsu maximises between-class
//      variance and correctly separates the positive/negative populations even
//      when the global distribution is heavily skewed.
//   3. Classification:
//        Count of CMOs above threshold == 1  →  SINGLET (assigned to that CMO)
//        Count == 0                          →  UNASSIGNED
//        Count >= 2                          →  MULTIPLET
//
// Input:  SparseAccumulator<uint32_t> in (cmo × cell) layout (from AdtCounter).
// Output: vector<CellPlexResult>, one entry per cell.
//
// Reuses: same CLR transform as hto_demux.h.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "sparse_accumulator.h"

namespace singlet {

// ── Result type ───────────────────────────────────────────────────────────────

struct CellPlexResult {
    std::string call;     ///< "SINGLET", "MULTIPLET", or "UNASSIGNED"
    std::string sample;   ///< CMO name for singlets; empty otherwise
    int32_t sample_idx;   ///< 0-based CMO index for singlets; -1 otherwise
    float top_clr;        ///< highest CLR value across all CMOs for this cell
    std::string top_cmo;  ///< CMO name with highest CLR
};

// ── CellPlexDemux ─────────────────────────────────────────────────────────────

class CellPlexDemux {
   public:
    CellPlexDemux() = default;

    // ── Demux ──────────────────────────────────────────────────────────────

    /// Demultiplex cells using the CMO count matrix.
    ///
    /// @param cmo_counts  SparseAccumulator<uint32_t> (cmos × cells).
    ///                    cmo_counts.n_features() == cmo_names.size().
    /// @param cmo_names   CMO feature names (length == number of rows).
    /// @param n_cells     Number of cells (columns in matrix).
    /// @returns           One CellPlexResult per cell (length == n_cells).
    std::vector<CellPlexResult> demux(
        const SparseAccumulator<uint32_t>& cmo_counts,
        const std::vector<std::string>& cmo_names,
        size_t n_cells) const {
        const size_t n_cmos = cmo_names.size();
        std::vector<CellPlexResult> results(n_cells,
                                            {"UNASSIGNED", "", -1, -1e9f, ""});

        if (n_cmos == 0 || n_cells == 0) return results;

        // ── Build dense (cmos × cells) float matrix ───────────────────────
        // mat[cmo * n_cells + cell] = raw count
        std::vector<float> mat(n_cmos * n_cells, 0.0f);
        {
            auto csc = cmo_counts.to_csc();
            for (uint32_t col = 0; col < static_cast<uint32_t>(n_cells); ++col) {
                for (int32_t j = csc.indptr[col]; j < csc.indptr[col + 1]; ++j) {
                    uint32_t row = static_cast<uint32_t>(csc.indices[j]);
                    if (row < n_cmos)
                        mat[row * n_cells + col] =
                            static_cast<float>(csc.data[j]);
                }
            }
        }

        // ── CLR normalisation (per cell, over CMOs) ───────────────────────
        // CLR(x_i) = log(x_i + 1) - mean(log(x_j + 1))
        std::vector<float> clr(n_cmos * n_cells, 0.0f);
        for (size_t c = 0; c < n_cells; ++c) {
            double log_sum = 0.0;
            for (size_t k = 0; k < n_cmos; ++k)
                log_sum += std::log(mat[k * n_cells + c] + 1.0);
            double log_gm = log_sum / static_cast<double>(n_cmos);
            for (size_t k = 0; k < n_cmos; ++k)
                clr[k * n_cells + c] = static_cast<float>(
                    std::log(mat[k * n_cells + c] + 1.0) - log_gm);
        }

        // ── Per-CMO threshold via Otsu's method ───────────────────────────
        // For each CMO, find threshold t that maximises between-class variance
        // over all cells:  w0*w1*(mu1 - mu0)^2  (1D Otsu)
        std::vector<float> threshold(n_cmos, 0.0f);
        for (size_t k = 0; k < n_cmos; ++k) {
            // Collect CLR values for this CMO across all cells
            std::vector<float> vals(n_cells);
            for (size_t c = 0; c < n_cells; ++c)
                vals[c] = clr[k * n_cells + c];

            std::vector<float> sorted_vals = vals;
            std::sort(sorted_vals.begin(), sorted_vals.end());

            // Unique candidate thresholds = midpoints between successive values
            float best_bcv = -1.0f;
            float best_t = 0.0f;
            const size_t N = n_cells;

            for (size_t i = 0; i + 1 < N; ++i) {
                if (sorted_vals[i] == sorted_vals[i + 1]) continue;
                float t = 0.5f * (sorted_vals[i] + sorted_vals[i + 1]);

                // Compute class means efficiently
                float sum_lo = 0.0f, sum_hi = 0.0f;
                size_t n_lo = 0, n_hi = 0;
                for (size_t c = 0; c < N; ++c) {
                    if (vals[c] <= t) {
                        sum_lo += vals[c];
                        ++n_lo;
                    } else {
                        sum_hi += vals[c];
                        ++n_hi;
                    }
                }
                if (n_lo == 0 || n_hi == 0) continue;

                float mu0 = sum_lo / static_cast<float>(n_lo);
                float mu1 = sum_hi / static_cast<float>(n_hi);
                float w0 = static_cast<float>(n_lo) / static_cast<float>(N);
                float w1 = static_cast<float>(n_hi) / static_cast<float>(N);
                float bcv = w0 * w1 * (mu1 - mu0) * (mu1 - mu0);

                if (bcv > best_bcv) {
                    best_bcv = bcv;
                    best_t = t;
                }
            }
            threshold[k] = best_t;
        }

        // ── Classification ────────────────────────────────────────────────
        for (size_t c = 0; c < n_cells; ++c) {
            int n_above = 0;
            int best_cmo = -1;
            float best_v = -1e9f;

            // find top-CLR CMO unconditionally for reporting
            for (size_t k = 0; k < n_cmos; ++k) {
                float v = clr[k * n_cells + c];
                if (v > best_v) {
                    best_v = v;
                    best_cmo = static_cast<int>(k);
                }
            }

            for (size_t k = 0; k < n_cmos; ++k) {
                if (clr[k * n_cells + c] > threshold[k]) ++n_above;
            }

            std::string top_name = (best_cmo >= 0)
                                       ? cmo_names[static_cast<size_t>(best_cmo)]
                                       : "";

            if (n_above == 1) {
                // Find which CMO is positive
                int pos_cmo = -1;
                for (size_t k = 0; k < n_cmos; ++k)
                    if (clr[k * n_cells + c] > threshold[k]) pos_cmo = static_cast<int>(k);
                results[c] = {"SINGLET",
                              cmo_names[static_cast<size_t>(pos_cmo)],
                              pos_cmo,
                              best_v, top_name};
            } else if (n_above >= 2) {
                results[c] = {"MULTIPLET", "", -1, best_v, top_name};
            } else {
                results[c] = {"UNASSIGNED", "", -1, best_v, top_name};
            }
        }
        return results;
    }

    // ── Output ────────────────────────────────────────────────────────────

    /// Write a TSV: barcode, call, sample, top_cmo, top_clr
    void write_tsv(const std::string& path,
                   const std::vector<std::string>& barcodes,
                   const std::vector<CellPlexResult>& results) const {
        std::ofstream ofs(path);
        ofs << "barcode\tcall\tsample\ttop_cmo\ttop_clr\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            std::string bc = (i < barcodes.size()) ? barcodes[i]
                                                   : std::to_string(i);
            ofs << bc << '\t'
                << r.call << '\t'
                << r.sample << '\t'
                << r.top_cmo << '\t'
                << r.top_clr << '\n';
        }
    }
};

}  // namespace singlet
