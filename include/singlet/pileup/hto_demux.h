#pragma once
// singlet-pileup: hto_demux.h
// T3: HTO (hashtag oligonucleotide) demultiplexing for CITE-seq experiments.
//
// Algorithm:
//   1. CLR normalisation: CLR(x_i) = log(x_i / geometric_mean(x)) per cell.
//      Cells with all-zero HTO counts receive classification NEGATIVE.
//   2. Per-HTO threshold: for each HTO, compute threshold = mean + 1.5 * MAD
//      of the *negative mode* (cells whose CLR for that HTO is < 0).
//      This robustly separates low (negative) and high (positive) HTO signal
//      without assuming a particular distribution.
//   3. Classification:
//        Count of HTOs above threshold == 1  →  SINGLET_<HTO_name>
//        Count == 0                          →  NEGATIVE
//        Count >= 2                          →  DOUBLET
//
// Input:  SparseAccumulator<uint32_t> in (tag × cell) layout (from AdtCounter).
// Output: vector<HtoDemuxResult>, one entry per cell (same order as barcodes).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "sparse_accumulator.h"

namespace singlet {

// ── Result type ───────────────────────────────────────────────────────────────

struct HtoDemuxResult {
    std::string classification;  ///< "SINGLET", "DOUBLET", or "NEGATIVE"
    std::string assignment;      ///< HTO name for singlets; empty otherwise
    float max_clr;               ///< highest CLR value across all HTOs for this cell
};

// ── HtoDemux ──────────────────────────────────────────────────────────────────

class HtoDemux {
   public:
    HtoDemux() = default;

    // ── Demux ──────────────────────────────────────────────────────────────

    /// Demultiplex cells using the HTO count matrix.
    ///
    /// @param hto_counts  SparseAccumulator<uint32_t> (tags × cells).
    ///                    hto_counts.n_features() == tag_names.size().
    /// @param tag_names   HTO tag names (length == number of rows).
    /// @param n_cells     Number of cells (number of columns in matrix).
    /// @returns           One HtoDemuxResult per cell (length == n_cells).
    std::vector<HtoDemuxResult> demux(
        const SparseAccumulator<uint32_t>& hto_counts,
        const std::vector<std::string>& tag_names,
        size_t n_cells) const {
        const size_t n_tags = tag_names.size();
        std::vector<HtoDemuxResult> results(n_cells, {"NEGATIVE", "", 0.0f});

        // ── Build dense (tags × cells) float matrix from sparse accumulator ──
        // mat[tag * n_cells + cell] = raw count
        std::vector<float> mat(n_tags * n_cells, 0.0f);
        {
            auto csc = hto_counts.to_csc();
            for (uint32_t col = 0; col < static_cast<uint32_t>(n_cells); ++col) {
                for (int32_t j = csc.indptr[col]; j < csc.indptr[col + 1]; ++j) {
                    uint32_t row = static_cast<uint32_t>(csc.indices[j]);
                    if (row < n_tags)
                        mat[row * n_cells + col] = static_cast<float>(csc.data[j]);
                }
            }
        }

        // ── CLR normalisation (per cell, over tags) ───────────────────────
        // CLR(x_i) = log(x_i + 1) - mean(log(x_j + 1)) across j
        // (pseudocount 1 avoids log(0); geometric mean with pseudocount)
        std::vector<float> clr(n_tags * n_cells, 0.0f);
        for (size_t c = 0; c < n_cells; ++c) {
            double log_sum = 0.0;
            for (size_t t = 0; t < n_tags; ++t)
                log_sum += std::log(mat[t * n_cells + c] + 1.0);
            double log_gm = log_sum / static_cast<double>(n_tags);
            for (size_t t = 0; t < n_tags; ++t)
                clr[t * n_cells + c] = static_cast<float>(
                    std::log(mat[t * n_cells + c] + 1.0) - log_gm);
        }

        // ── Per-HTO threshold: mean + 1.5 * MAD on negative-mode cells ───
        std::vector<float> threshold(n_tags, 0.0f);
        for (size_t t = 0; t < n_tags; ++t) {
            // Collect all CLR values for this HTO
            std::vector<float> neg_mode;
            neg_mode.reserve(n_cells);
            for (size_t c = 0; c < n_cells; ++c) {
                float v = clr[t * n_cells + c];
                if (v < 0.0f) neg_mode.push_back(v);
            }
            if (neg_mode.empty()) {
                // All cells positive — use global mean as fallback threshold
                float s = 0.0f;
                for (size_t c = 0; c < n_cells; ++c) s += clr[t * n_cells + c];
                threshold[t] = s / static_cast<float>(n_cells);
                continue;
            }
            // Mean of negative mode
            float mean = 0.0f;
            for (float v : neg_mode) mean += v;
            mean /= static_cast<float>(neg_mode.size());
            // MAD
            std::vector<float> devs;
            devs.reserve(neg_mode.size());
            for (float v : neg_mode) devs.push_back(std::abs(v - mean));
            std::sort(devs.begin(), devs.end());
            float mad = devs[devs.size() / 2];
            threshold[t] = mean + 1.5f * mad;
        }

        // ── Classification ────────────────────────────────────────────────
        for (size_t c = 0; c < n_cells; ++c) {
            int n_above = 0;
            int best_tag = -1;
            float best_clr = -1e9f;
            for (size_t t = 0; t < n_tags; ++t) {
                float v = clr[t * n_cells + c];
                if (v > threshold[t]) {
                    ++n_above;
                    if (v > best_clr) {
                        best_clr = v;
                        best_tag = static_cast<int>(t);
                    }
                }
            }
            if (n_above == 1) {
                results[c] = {"SINGLET", tag_names[static_cast<size_t>(best_tag)], best_clr};
            } else if (n_above >= 2) {
                // Record best CLR even for doublets
                float mx = *std::max_element(clr.data() + c,
                                             clr.data() + n_tags * n_cells,
                                             [&](float a, float b) {
                                                 // stride n_cells access
                                                 (void)a;
                                                 (void)b;
                                                 return false;  // unused
                                             });
                // Proper max over all tags for this cell
                mx = -1e9f;
                for (size_t t = 0; t < n_tags; ++t)
                    mx = std::max(mx, clr[t * n_cells + c]);
                results[c] = {"DOUBLET", "", mx};
            } else {
                float mx = -1e9f;
                for (size_t t = 0; t < n_tags; ++t)
                    mx = std::max(mx, clr[t * n_cells + c]);
                results[c] = {"NEGATIVE", "", mx};
            }
        }
        return results;
    }

    // ── Output ────────────────────────────────────────────────────────────

    /// Write a TSV with columns: barcode, classification, assignment, max_clr
    void write_tsv(const std::string& path,
                   const std::vector<std::string>& barcodes,
                   const std::vector<HtoDemuxResult>& results) const {
        std::ofstream ofs(path);
        ofs << "barcode\tclassification\tassignment\tmax_clr\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            std::string bc = (i < barcodes.size()) ? barcodes[i] : std::to_string(i);
            ofs << bc << '\t'
                << r.classification << '\t'
                << r.assignment << '\t'
                << r.max_clr << '\n';
        }
    }
};

}  // namespace singlet
