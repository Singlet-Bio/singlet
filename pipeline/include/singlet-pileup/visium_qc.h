#pragma once
// singlet-pileup: visium_qc.h
// V3: Visium spatial QC metrics.
//
// Computes spatial-specific QC:
//   - Per-spot: total_umis, genes_detected, mt_pct, in_tissue flag,
//               array_row, array_col
//   - Summary: tissue_coverage (fraction of in_tissue spots ≥ min_umi),
//              median_umis_in_tissue, mean_umis_in_tissue,
//              std_umis_in_tissue, n_in_tissue, n_detected_spots
//
// V3 builds on cell_qc_metrics.h (per-cell) and visium_spatial.h (coords).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "visium_spatial.h"

namespace singlet {

// ── Per-spot QC record ────────────────────────────────────────────────────────

struct SpotQCRecord {
    std::string barcode;
    uint32_t total_umis = 0;
    uint32_t genes_detected = 0;
    float mt_pct = 0.f;
    int32_t array_row = 0;
    int32_t array_col = 0;
    float pixel_row = 0.f;
    float pixel_col = 0.f;
    bool in_tissue = false;
    bool in_positions = false;  ///< true if barcode found in tissue_positions.csv
};

// ── Summary QC ────────────────────────────────────────────────────────────────

struct VisiumQCSummary {
    size_t n_barcodes = 0;        ///< total barcodes in pileup output
    size_t n_in_tissue = 0;       ///< barcodes found in positions AND in_tissue==1
    size_t n_detected_spots = 0;  ///< in_tissue spots with ≥ min_umi UMIs
    double tissue_coverage = 0.;  ///< n_detected / n_in_tissue (0–1.0)
    double median_umis = 0.;      ///< median total_umis over in_tissue spots
    double mean_umis = 0.;        ///< mean over in_tissue spots
    double std_umis = 0.;         ///< std-dev over in_tissue spots
    uint32_t min_umi_threshold = 100;
};

// ── Core computation ──────────────────────────────────────────────────────────

/// Build per-spot QC records by joining pileup metrics with spatial positions.
///
/// @param barcodes       Ordered barcode list from pileup output (n_cells)
/// @param total_umis     Per-cell total UMI counts (n_cells)
/// @param genes_detected Per-cell gene counts (n_cells)
/// @param mt_pct         Per-cell mt fraction 0–100 (n_cells; may be empty)
/// @param parser         Loaded VisiumSpatialParser (may be empty/unloaded)
/// @return               One SpotQCRecord per barcode
inline std::vector<SpotQCRecord> build_spot_qc(
    const std::vector<std::string>& barcodes,
    const std::vector<uint32_t>& total_umis,
    const std::vector<uint32_t>& genes_detected,
    const std::vector<float>& mt_pct,
    const VisiumSpatialParser& parser) {
    const size_t n = barcodes.size();
    std::vector<SpotQCRecord> records(n);

    for (size_t i = 0; i < n; ++i) {
        auto& rec = records[i];
        rec.barcode = barcodes[i];
        rec.total_umis = (i < total_umis.size()) ? total_umis[i] : 0;
        rec.genes_detected = (i < genes_detected.size()) ? genes_detected[i] : 0;
        rec.mt_pct = (i < mt_pct.size()) ? mt_pct[i] : 0.f;

        const SpotCoordinate* c = parser.get_coordinate(barcodes[i]);
        if (c) {
            rec.in_positions = true;
            rec.in_tissue = c->in_tissue;
            rec.array_row = c->array_row;
            rec.array_col = c->array_col;
            rec.pixel_row = c->pixel_row;
            rec.pixel_col = c->pixel_col;
        }
    }
    return records;
}

/// Compute summary statistics over in_tissue spots.
///
/// @param records        Output of build_spot_qc()
/// @param min_umi        UMI threshold for tissue_coverage (default 100)
/// @return               VisiumQCSummary
inline VisiumQCSummary compute_visium_summary(
    const std::vector<SpotQCRecord>& records,
    uint32_t min_umi = 100) {
    VisiumQCSummary s;
    s.n_barcodes = records.size();
    s.min_umi_threshold = min_umi;

    // Collect UMIs for in_tissue spots
    std::vector<double> tissue_umis;
    tissue_umis.reserve(records.size());

    for (const auto& rec : records) {
        if (rec.in_tissue) {
            ++s.n_in_tissue;
            tissue_umis.push_back(static_cast<double>(rec.total_umis));
            if (rec.total_umis >= min_umi)
                ++s.n_detected_spots;
        }
    }

    if (s.n_in_tissue == 0) return s;

    // tissue_coverage
    s.tissue_coverage = static_cast<double>(s.n_detected_spots) /
                        static_cast<double>(s.n_in_tissue);

    // mean
    double sum = 0.;
    for (double v : tissue_umis) sum += v;
    s.mean_umis = sum / static_cast<double>(tissue_umis.size());

    // std-dev
    double ss = 0.;
    for (double v : tissue_umis) {
        double d = v - s.mean_umis;
        ss += d * d;
    }
    s.std_umis = (tissue_umis.size() > 1)
                     ? std::sqrt(ss / static_cast<double>(tissue_umis.size() - 1))
                     : 0.;

    // median
    std::vector<double> sorted = tissue_umis;
    std::sort(sorted.begin(), sorted.end());
    const size_t m = sorted.size();
    s.median_umis = (m % 2 == 1)
                        ? sorted[m / 2]
                        : (sorted[m / 2 - 1] + sorted[m / 2]) / 2.0;

    return s;
}

// ── Output ────────────────────────────────────────────────────────────────────

/// Write per-spot QC TSV.
/// Columns: barcode  total_umis  genes_detected  mt_pct
///          array_row  array_col  pixel_row  pixel_col  in_tissue  in_positions
inline bool write_visium_qc_tsv(
    const std::string& path,
    const std::vector<SpotQCRecord>& records) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;

    ofs << "barcode\ttotal_umis\tgenes_detected\tmt_pct"
           "\tarray_row\tarray_col\tpixel_row\tpixel_col"
           "\tin_tissue\tin_positions\n";

    for (const auto& r : records) {
        ofs << r.barcode << '\t'
            << r.total_umis << '\t'
            << r.genes_detected << '\t'
            << r.mt_pct << '\t'
            << r.array_row << '\t'
            << r.array_col << '\t'
            << r.pixel_row << '\t'
            << r.pixel_col << '\t'
            << (r.in_tissue ? 1 : 0) << '\t'
            << (r.in_positions ? 1 : 0) << '\n';
    }
    return true;
}

/// Print summary QC to stderr in the singlify log style.
inline void log_visium_summary(const VisiumQCSummary& s) {
    std::cerr << "[visium_qc] Spots: " << s.n_barcodes << " barcodes, "
              << s.n_in_tissue << " in-tissue\n";
    if (s.n_in_tissue > 0) {
        std::cerr << "[visium_qc] Tissue coverage (≥" << s.min_umi_threshold
                  << " UMIs): " << s.n_detected_spots << "/" << s.n_in_tissue
                  << " = " << static_cast<int>(s.tissue_coverage * 100) << "%\n";
        std::cerr << "[visium_qc] UMIs/spot (in-tissue): median="
                  << static_cast<int>(s.median_umis)
                  << " mean=" << static_cast<int>(s.mean_umis)
                  << " std=" << static_cast<int>(s.std_umis) << "\n";
    }
}

}  // namespace singlet
