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
    // Schema required keys (§3.6)
    std::string schema_version  = "1.0";
    std::string track           = "A";  ///< "A" (STAR-only) or "B" (cascade)

    // Sample info
    std::string sample_id;
    std::string protocol;           ///< legacy/internal; kept for backward compat
    std::string organism;           ///< legacy/internal; kept for backward compat
    std::string singlify_version;
    // §3.6 canonical names
    int         protocol_id   = 0;
    std::string protocol_name;
    std::string species;
    std::string reference_build;

    // Read metrics (legacy field names — also written as canonical names)
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

    // Resource usage
    double      peak_rss_gb   = 0.0;
    std::string memory_tier   = "unknown"; ///< "Small"|"Medium"|"Large"|"XLarge"|"unknown"

    // Timing (seconds)
    double wall_seconds   = 0.0;
    double star_seconds   = 0.0;
    double pileup_seconds = 0.0;
    double export_seconds = 0.0;

    // Status
    int         exit_code = 0;
    std::string status;   ///< "success" | "align_low_map" | "align_zero_cells" | "align_low_cells" | "align_low_genes"
    std::vector<std::string> warnings;

    // Donor block (§3.6)
    int         donor_n_donors_inferred = 0;
    std::string donor_demux_method;    ///< "vb_binomial" when VB demux ran, else ""

    // mt block (§3.6)
    int      mt_donors_with_consensus = 0;
    uint64_t mt_n_events_total        = 0;

    // nonhost block (§3.6)
    bool nonhost_screened        = false;
    int  nonhost_species_above_em = 0;

    // Track B cascade block (T-L5-4)
    bool     cascade_used           = false;
    uint64_t cascade_l1_resolved    = 0;
    uint64_t cascade_l2_resolved    = 0;
    uint64_t cascade_l3_star        = 0;
    uint64_t cascade_l4_nonhost     = 0;

    // GEO / pipeline metadata passthrough
    std::map<std::string, std::string> user_meta;
};

/// Classify pipeline outcome from filled-in metrics.
///
/// Checks are applied in priority order; first match wins.
/// @param s         Filled PipelineSummary (exit_code not consulted here — check separately).
/// @param assay_type  "scrna" (default), "atac", "cite", "multiome", "visium"
/// @return  "success" | "align_low_map" | "align_zero_cells" | "align_low_cells" | "align_low_genes"
inline std::string classify_outcome(const PipelineSummary& s,
                                    const std::string& assay_type = "scrna") {
    // No reads at all — treat as empty/no-data success rather than failure
    if (s.total_reads == 0)
        return "success";

    // Low mapping: common threshold 50% for scRNA/CITE/Multiome, 30% for ATAC
    const double map_floor = (assay_type == "atac") ? 0.30 : 0.50;
    if (s.mapping_rate > 0.0 && s.mapping_rate < map_floor)
        return "align_low_map";

    // ATAC: skip cell / gene checks
    if (assay_type == "atac" || assay_type == "visium")
        return "success";

    // Cell count checks (scrna, cite, multiome)
    if (s.estimated_cells == 0)
        return "align_zero_cells";
    if (s.estimated_cells < 10)
        return "align_low_cells";

    // Gene content check: only when cell calling ran and median > 0
    if (s.median_genes_per_cell > 0.0 && s.median_genes_per_cell < 200.0)
        return "align_low_genes";

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

    // Helper: write a double with limited precision, avoiding -0.0
    auto jd = [](double v) -> double { return (v == 0.0) ? 0.0 : v; };

    f << "{\n";
    // §3.6 required schema keys
    f << "  \"schema_version\": \""       << json_escape(s.schema_version)   << "\",\n";
    f << "  \"sample_id\": \""            << json_escape(s.sample_id)        << "\",\n";
    f << "  \"status\": \""               << json_escape(s.status)           << "\",\n";
    f << "  \"protocol_id\": "            << s.protocol_id                   << ",\n";
    f << "  \"protocol_name\": \""        << json_escape(s.protocol_name)    << "\",\n";
    f << "  \"species\": \""              << json_escape(s.species)          << "\",\n";
    f << "  \"reference_build\": \""      << json_escape(s.reference_build)  << "\",\n";
    f << "  \"n_input_reads\": "          << s.total_reads                   << ",\n";
    f << "  \"n_uniquely_mapped\": "      << s.mapped_reads                  << ",\n";
    f << "  \"uniquely_mapped_pct\": "    << jd(s.mapping_rate * 100.0)      << ",\n";
    f << "  \"n_cells_called\": "         << s.estimated_cells               << ",\n";
    f << "  \"median_umi_per_cell\": "    << jd(s.median_umis_per_cell)      << ",\n";
    f << "  \"median_genes_per_cell\": "  << jd(s.median_genes_per_cell)     << ",\n";
    f << "  \"memory_tier\": \""          << json_escape(s.memory_tier)      << "\",\n";
    f << "  \"peak_rss_gb\": "            << jd(s.peak_rss_gb)               << ",\n";
    f << "  \"wall_seconds\": "           << jd(s.wall_seconds)              << ",\n";
    f << "  \"track\": \""               << json_escape(s.track)            << "\",\n";
    // donor block
    f << "  \"donor\": { \"n_donors_inferred\": " << s.donor_n_donors_inferred
      << ", \"demux_method\": \"" << json_escape(s.donor_demux_method) << "\" },\n";
    // mt block
    f << "  \"mt\": { \"donors_with_consensus\": " << s.mt_donors_with_consensus
      << ", \"n_mt_events_total\": " << s.mt_n_events_total << " },\n";
    // nonhost block
    f << "  \"nonhost\": { \"screened\": "
      << (s.nonhost_screened ? "true" : "false")
      << ", \"species_above_em\": " << s.nonhost_species_above_em << " },\n";
    // cascade block (T-L5-4)
    f << "  \"cascade\": { \"used\": " << (s.cascade_used ? "true" : "false")
      << ", \"layers_resolved\": { \"l1_txome\": " << s.cascade_l1_resolved
      << ", \"l2_te\": "    << s.cascade_l2_resolved
      << ", \"l3_star\": "  << s.cascade_l3_star
      << ", \"l4_nonhost\": " << s.cascade_l4_nonhost
      << " } },\n";
    // Legacy / extra metrics (for downstream tools that read old keys)
    f << "  \"singlify_version\": \""   << json_escape(s.singlify_version)   << "\",\n";
    f << "  \"total_reads\": "          << s.total_reads         << ",\n";
    f << "  \"mapped_reads\": "         << s.mapped_reads        << ",\n";
    f << "  \"mapping_rate\": "         << jd(s.mapping_rate)    << ",\n";
    f << "  \"exonic_fraction\": "      << jd(s.exonic_fraction)     << ",\n";
    f << "  \"intronic_fraction\": "    << jd(s.intronic_fraction)   << ",\n";
    f << "  \"intergenic_fraction\": "  << jd(s.intergenic_fraction) << ",\n";
    f << "  \"estimated_cells\": "      << s.estimated_cells        << ",\n";
    f << "  \"median_umis_per_cell\": " << jd(s.median_umis_per_cell)   << ",\n";
    f << "  \"mean_reads_per_cell\": "  << jd(s.mean_reads_per_cell)    << ",\n";
    f << "  \"total_genes_detected\": " << s.total_genes_detected   << ",\n";
    f << "  \"sequencing_saturation\": "  << jd(s.sequencing_saturation)   << ",\n";
    f << "  \"median_mito_fraction\": "   << jd(s.median_mito_fraction)    << ",\n";
    f << "  \"fraction_reads_in_cells\": "<< jd(s.fraction_reads_in_cells) << ",\n";
    f << "  \"star_seconds\": "   << jd(s.star_seconds)   << ",\n";
    f << "  \"pileup_seconds\": " << jd(s.pileup_seconds) << ",\n";
    f << "  \"export_seconds\": " << jd(s.export_seconds) << ",\n";
    f << "  \"exit_code\": " << s.exit_code << ",\n";
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
