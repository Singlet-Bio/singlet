// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: validate_output.h
// Post-export output validation for production-hardened pipeline runs.
// Checks that all expected files exist and are non-empty after export_results().

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace singlet {

/// Result of post-export output validation.
struct ValidationResult {
    bool        passed = false;
    uint32_t    n_expected = 0;
    uint32_t    n_found    = 0;
    uint32_t    n_missing  = 0;
    uint32_t    n_empty    = 0;
    std::vector<std::string> missing_files;
    std::vector<std::string> empty_files;
    std::vector<std::string> warnings;
};

/// Canonical list of files that MUST exist after every successful droplet export.
/// Files are always written; empty/stub content when the underlying data is absent.
inline std::vector<std::string> required_output_files() {
    return {
        // Sparse matrices (.1pz)
        "exon_counts.1pz",
        "intron_counts.1pz",
        "sj_counts.1pz",
        "mt_heteroplasmy.1pz",
        "splice_psi.1pz",
        "vdj_gene_usage.1pz",
        // Per-cell annotations (TSV)
        "cell_qc_metrics.tsv",
        "cell_calls.tsv",
        "cell_cycle_scores.tsv",
        "doublet_scores.tsv",
        "ambient_contamination.tsv",
        "ambient_profile.tsv",
        "read_stats.tsv",
        "donor_assignments.tsv",
        "auto_barcodes.tsv",
        "saturation_curve.tsv",
        "ase_counts.tsv",
        "mt_variants.tsv",
        "gene_expression.tsv",
        "splice_events.tsv",
        "metrics_summary.csv",
        // Metadata (JSON)
        "summary.json",
        "provenance.json",
        "pileup_stats.json",
        "sex_call.json",
        "ancestry_call.json",
        "rrna_report.json",
        // MT reference
        "mt_reference.fa",
        // STAR artifacts
        "star_Log.final.out",
        "star_Log.out",
    };
}

/// Validate that all required output files exist and are non-empty.
///
/// Call this at the end of export_results() after all write threads have joined.
/// Sets summary.json status to "export_incomplete" if validation fails.
///
/// @param out_prefix  Output directory path
/// @return ValidationResult with details of any missing or empty files
inline ValidationResult validate_output(const std::string& out_prefix) {
    ValidationResult result;
    auto required = required_output_files();
    result.n_expected = static_cast<uint32_t>(required.size());

    for (const auto& filename : required) {
        std::string path = out_prefix + "/" + filename;
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            result.missing_files.push_back(filename);
            ++result.n_missing;
        } else {
            auto sz = std::filesystem::file_size(path, ec);
            if (ec || sz == 0) {
                result.empty_files.push_back(filename);
                ++result.n_empty;
            } else {
                ++result.n_found;
            }
        }
    }

    result.passed = (result.n_missing == 0 && result.n_empty == 0);
    return result;
}

/// Write validation result to a JSON file for auditing.
inline void write_validation_json(const std::string& filepath,
                                  const ValidationResult& vr) {
    std::ofstream f(filepath);
    if (!f) return;
    f << "{\n"
      << "  \"passed\": " << (vr.passed ? "true" : "false") << ",\n"
      << "  \"n_expected\": " << vr.n_expected << ",\n"
      << "  \"n_found\": " << vr.n_found << ",\n"
      << "  \"n_missing\": " << vr.n_missing << ",\n"
      << "  \"n_empty\": " << vr.n_empty << ",\n"
      << "  \"missing\": [";
    for (size_t i = 0; i < vr.missing_files.size(); ++i) {
        if (i > 0) f << ", ";
        f << "\"" << vr.missing_files[i] << "\"";
    }
    f << "],\n"
      << "  \"empty\": [";
    for (size_t i = 0; i < vr.empty_files.size(); ++i) {
        if (i > 0) f << ", ";
        f << "\"" << vr.empty_files[i] << "\"";
    }
    f << "]\n}\n";
}

}  // namespace singlet
