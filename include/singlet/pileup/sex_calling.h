// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: sex_calling.h
// N14: Sex & karyotype calling from gene expression data.
//
// Infers biological sex from expression of sex-chromosome marker genes:
//   Female: high XIST expression, low/absent chrY markers
//   Male:   absent XIST, detectable chrY marker genes (UTY, DDX3Y, KDM5D, EIF1AY, ZFY)
//
// Algorithm:
//   1. Walk GeneModel to find XIST exon rows and chrY marker exon rows
//   2. Sum expression across all cells (sample-level aggregate)
//   3. Normalize to CPM using total exon UMIs
//   4. Classify based on XIST_cpm vs Y_cpm thresholds
//
// Usage:
//   auto result = singlet::call_sex(exon_csc, engine.gene_model());
//   singlet::write_sex_call_json(out_prefix + "/sex_call.json", result);

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gene_model.h"

namespace singlet {

/// Result of sample-level sex inference from expression.
struct SexCallResult {
    std::string sex;          ///< "male", "female", or "unknown"
    double      confidence;   ///< 0–1 score: how clear the signal is
    double      xist_cpm;     ///< XIST expression in CPM
    double      y_cpm;        ///< Combined chrY marker expression in CPM
    uint64_t    xist_umis;    ///< Raw XIST UMI count across all cells
    uint64_t    y_umis;       ///< Raw chrY marker UMI count across all cells
    int         n_y_genes_detected; ///< Number of distinct chrY marker genes with >0 UMIs
    uint64_t    total_umis;   ///< Total UMIs in the exon count matrix
};

/// chrY marker gene names used for male calling.
/// These are robustly expressed only in XY individuals.
static const std::unordered_set<std::string> kYMarkerGenes = {
    "UTY", "DDX3Y", "KDM5D", "EIF1AY", "ZFY", "RPS4Y1", "USP9Y"
};

/// Returns true if chrom represents chrX (handles "X", "chrX").
inline bool is_chrom_x(const std::string& chrom) {
    return chrom == "X" || chrom == "chrX";
}

/// Returns true if chrom represents chrY (handles "Y", "chrY").
inline bool is_chrom_y(const std::string& chrom) {
    return chrom == "Y" || chrom == "chrY";
}

/// Infer biological sex from an exon count matrix.
///
/// @tparam T        Value type of the CSC matrix (typically uint16_t)
/// @tparam CSCMat   CSC matrix type with indptr, indices, data, nrows, ncols
/// @param  exon_csc Exon count matrix (features × barcodes) in CSC format
/// @param  model    GeneModel loaded from GTF
/// @return          SexCallResult with sex call and confidence
template <typename CSCMat>
SexCallResult call_sex(const CSCMat& exon_csc, const GeneModel& model) {
    SexCallResult result;
    result.sex = "unknown";
    result.confidence = 0.0;
    result.xist_cpm = 0.0;
    result.y_cpm = 0.0;
    result.xist_umis = 0;
    result.y_umis = 0;
    result.n_y_genes_detected = 0;
    result.total_umis = 0;

    if (!model.has_gene_hierarchy() || exon_csc.nrows == 0 || exon_csc.ncols == 0) {
        return result;
    }

    const auto& genes = model.genes();

    // Build sets of exon row indices for XIST and chrY markers
    std::unordered_set<uint32_t> xist_exon_rows;
    // Y marker: exon_row → gene_name (to count distinct genes detected)
    std::unordered_map<uint32_t, std::string> y_exon_rows;

    for (const auto& gene : genes) {
        if (is_chrom_x(gene.chrom) && gene.gene_name == "XIST") {
            for (uint32_t ei : gene.exon_indices) {
                if (ei < exon_csc.nrows) xist_exon_rows.insert(ei);
            }
        } else if (is_chrom_y(gene.chrom) && kYMarkerGenes.count(gene.gene_name)) {
            for (uint32_t ei : gene.exon_indices) {
                if (ei < exon_csc.nrows) y_exon_rows[ei] = gene.gene_name;
            }
        }
    }

    // Single pass over all nonzero entries: accumulate total, XIST, and Y marker UMIs
    uint64_t xist_umis = 0, y_umis = 0, total_umis = 0;
    std::unordered_map<std::string, uint64_t> y_gene_umis;  // gene_name → UMIs

    const uint32_t ncols = exon_csc.ncols;
    for (uint32_t j = 0; j < ncols; ++j) {
        for (int32_t k = exon_csc.indptr[j]; k < exon_csc.indptr[j + 1]; ++k) {
            uint32_t row = static_cast<uint32_t>(exon_csc.indices[k]);
            uint64_t  val = static_cast<uint64_t>(exon_csc.data[k]);
            total_umis += val;
            if (xist_exon_rows.count(row)) {
                xist_umis += val;
            } else {
                auto yt = y_exon_rows.find(row);
                if (yt != y_exon_rows.end()) {
                    y_umis += val;
                    y_gene_umis[yt->second] += val;
                }
            }
        }
    }

    result.xist_umis  = xist_umis;
    result.y_umis     = y_umis;
    result.total_umis = total_umis;

    // Count distinct chrY marker genes with any expression
    int n_y_genes = 0;
    for (auto& [name, cnt] : y_gene_umis) {
        if (cnt > 0) n_y_genes++;
    }
    result.n_y_genes_detected = n_y_genes;

    if (total_umis == 0) return result;

    // Normalize to CPM
    const double cpm_scale = 1e6 / static_cast<double>(total_umis);
    result.xist_cpm = xist_umis * cpm_scale;
    result.y_cpm    = y_umis    * cpm_scale;

    // Classification thresholds (CPM-based, validated empirically)
    //   FEMALE:  XIST_cpm > 1.0  AND  Y_cpm < 2.0
    //   MALE:    Y_cpm > 2.0     AND  n_y_genes >= 2
    //   UNKNOWN: neither criterion met (low coverage, monosomy, etc.)
    const double XIST_FEMALE_THRESH = 1.0;
    const double Y_MALE_THRESH      = 2.0;
    const int    Y_GENE_MIN         = 2;   // require ≥2 distinct chrY genes

    double xist_signal = result.xist_cpm;
    double y_signal    = result.y_cpm;

    bool female_xist = (xist_signal > XIST_FEMALE_THRESH);
    bool male_y      = (y_signal > Y_MALE_THRESH) && (n_y_genes >= Y_GENE_MIN);

    if (female_xist && !male_y) {
        result.sex = "female";
        // Confidence: how far above threshold, capped at 1
        result.confidence = std::min(1.0, xist_signal / (XIST_FEMALE_THRESH * 10.0));
    } else if (male_y && !female_xist) {
        result.sex = "male";
        result.confidence = std::min(1.0, y_signal / (Y_MALE_THRESH * 10.0));
    } else if (female_xist && male_y) {
        // Both signals present — ratio test to resolve ambient noise
        if (xist_signal > 0 && y_signal / xist_signal > 10.0) {
            result.sex = "male";
            result.confidence = 1.0 - (xist_signal / y_signal);
        } else if (y_signal > 0 && xist_signal / y_signal > 10.0) {
            result.sex = "female";
            result.confidence = 1.0 - (y_signal / xist_signal);
        } else {
            result.sex = "unknown";
            result.confidence = 0.0;
        }
    } else {
        // Neither clear signal
        result.sex = "unknown";
        // Low but measurable confidence for partial signals
        double partial = std::max(xist_signal / XIST_FEMALE_THRESH,
                                  y_signal / Y_MALE_THRESH);
        result.confidence = std::min(0.5, partial * 0.3);
    }

    return result;
}

/// Write sex call result to a JSON file.
///
/// @param path   Output file path (e.g. "outdir/sex_call.json")
/// @param result SexCallResult from call_sex()
inline void write_sex_call_json(const std::string& path,
                                 const SexCallResult& result) {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "[sex_calling] Cannot write: " << path << "\n";
        return;
    }
    auto fmt_dbl = [](double v) -> std::string {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.4f", v);
        return buf;
    };
    f << "{\n"
      << "  \"sex\": \""             << result.sex             << "\",\n"
      << "  \"confidence\": "        << fmt_dbl(result.confidence) << ",\n"
      << "  \"xist_cpm\": "          << fmt_dbl(result.xist_cpm) << ",\n"
      << "  \"y_marker_cpm\": "      << fmt_dbl(result.y_cpm)    << ",\n"
      << "  \"n_y_genes_detected\": "<< result.n_y_genes_detected << ",\n"
      << "  \"xist_umis\": "         << result.xist_umis          << ",\n"
      << "  \"y_marker_umis\": "     << result.y_umis             << ",\n"
      << "  \"total_umis\": "        << result.total_umis         << "\n"
      << "}\n";
}

} // namespace singlet
