#pragma once
// singlet-pileup: summary_json.h
// Comprehensive structured pipeline summary JSON.
// VAL2-compatible single output capturing all key metrics from a pipeline run.
//
// Design: standalone header — no dependency on pileup_engine.h or gene_model.h.
// Callers fill PipelineSummary from external data and call write_summary_json().

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace singlet {

/// Comprehensive pipeline run metrics.
struct PipelineSummary {
    // Sample info
    std::string sample_id;
    std::string protocol;
    std::string organism;
    std::string singlify_version;

    // Read metrics
    uint64_t total_reads = 0;
    uint64_t mapped_reads = 0;
    double   mapping_rate = 0.0;
    double   exonic_fraction = 0.0;
    double   intronic_fraction = 0.0;
    double   intergenic_fraction = 0.0;

    // Cell metrics
    uint64_t estimated_cells = 0;
    double   median_genes_per_cell = 0.0;
    double   median_umis_per_cell = 0.0;
    double   mean_reads_per_cell = 0.0;
    uint64_t total_genes_detected = 0;

    // QC metrics
    double sequencing_saturation = 0.0;
    double median_mito_fraction = 0.0;
    double fraction_reads_in_cells = 0.0;

    // Timing (seconds)
    double wall_seconds   = 0.0;
    double star_seconds   = 0.0;
    double pileup_seconds = 0.0;
    double export_seconds = 0.0;

    // Status
    int         exit_code = 0;
    std::string status;   ///< "success" | "low_mapping" | "no_cells" | "low_cells" | "low_genes"
    std::vector<std::string> warnings;

    // GEO / pipeline metadata passthrough
    std::map<std::string, std::string> user_meta;
};

/// Classify pipeline outcome from filled-in metrics.
///
/// Checks are applied in priority order; first match wins.
/// @param s         Filled PipelineSummary (exit_code not consulted here — check separately).
/// @param assay_type  "scrna" (default), "atac", "cite", "multiome", "visium"
/// @return  "success" | "low_mapping" | "no_cells" | "low_cells" | "low_genes"
inline std::string classify_outcome(const PipelineSummary& s,
                                    const std::string& assay_type = "scrna") {
    // No reads at all — treat as empty/no-data success rather than failure
    if (s.total_reads == 0)
        return "success";

    // Low mapping: common threshold 50% for scRNA/CITE/Multiome, 30% for ATAC
    const double map_floor = (assay_type == "atac") ? 0.30 : 0.50;
    if (s.mapping_rate > 0.0 && s.mapping_rate < map_floor)
        return "low_mapping";

    // ATAC: skip cell / gene checks
    if (assay_type == "atac" || assay_type == "visium")
        return "success";

    // Cell count checks (scrna, cite, multiome)
    if (s.estimated_cells == 0)
        return "no_cells";
    if (s.estimated_cells < 10)
        return "low_cells";

    // Gene content check: only when cell calling ran and median > 0
    if (s.median_genes_per_cell > 0.0 && s.median_genes_per_cell < 200.0)
        return "low_genes";

    return "success";
}

/// Escape a string value for embedding inside a JSON double-quoted string.
inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    // Encode low control chars as \uXXXX
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
                break;
        }
    }
    return out;
}

/// Write pipeline summary as a JSON file to `filepath`.
/// JSON is hand-serialized (no external deps).
/// @return true on success, false if file could not be opened or write failed.
inline bool write_summary_json(const PipelineSummary& s, const std::string& filepath) {
    std::ofstream f(filepath);
    if (!f) return false;

    f << "{\n";
    // Sample info
    f << "  \"sample_id\": \""          << json_escape(s.sample_id)          << "\",\n";
    f << "  \"protocol\": \""           << json_escape(s.protocol)           << "\",\n";
    f << "  \"organism\": \""           << json_escape(s.organism)           << "\",\n";
    f << "  \"singlify_version\": \""   << json_escape(s.singlify_version)   << "\",\n";
    // Read metrics
    f << "  \"total_reads\": "          << s.total_reads         << ",\n";
    f << "  \"mapped_reads\": "         << s.mapped_reads        << ",\n";
    f << "  \"mapping_rate\": "         << s.mapping_rate        << ",\n";
    f << "  \"exonic_fraction\": "      << s.exonic_fraction     << ",\n";
    f << "  \"intronic_fraction\": "    << s.intronic_fraction   << ",\n";
    f << "  \"intergenic_fraction\": "  << s.intergenic_fraction << ",\n";
    // Cell metrics
    f << "  \"estimated_cells\": "      << s.estimated_cells        << ",\n";
    f << "  \"median_genes_per_cell\": "<< s.median_genes_per_cell  << ",\n";
    f << "  \"median_umis_per_cell\": " << s.median_umis_per_cell   << ",\n";
    f << "  \"mean_reads_per_cell\": "  << s.mean_reads_per_cell    << ",\n";
    f << "  \"total_genes_detected\": " << s.total_genes_detected   << ",\n";
    // QC metrics
    f << "  \"sequencing_saturation\": "  << s.sequencing_saturation   << ",\n";
    f << "  \"median_mito_fraction\": "   << s.median_mito_fraction    << ",\n";
    f << "  \"fraction_reads_in_cells\": "<< s.fraction_reads_in_cells << ",\n";
    // Timing
    f << "  \"wall_seconds\": "   << s.wall_seconds   << ",\n";
    f << "  \"star_seconds\": "   << s.star_seconds   << ",\n";
    f << "  \"pileup_seconds\": " << s.pileup_seconds << ",\n";
    f << "  \"export_seconds\": " << s.export_seconds << ",\n";
    // Status
    f << "  \"exit_code\": " << s.exit_code << ",\n";
    f << "  \"status\": \""  << json_escape(s.status) << "\",\n";
    // Warnings array
    f << "  \"warnings\": [";
    for (size_t i = 0; i < s.warnings.size(); ++i) {
        if (i > 0) f << ", ";
        f << "\"" << json_escape(s.warnings[i]) << "\"";
    }
    f << "],\n";
    // User metadata passthrough
    f << "  \"user_meta\": {";
    bool first_meta = true;
    for (const auto& kv : s.user_meta) {
        if (!first_meta) f << ", ";
        f << "\"" << json_escape(kv.first) << "\": \""
          << json_escape(kv.second) << "\"";
        first_meta = false;
    }
    f << "}\n";
    f << "}\n";
    return f.good();
}

} // namespace singlet
