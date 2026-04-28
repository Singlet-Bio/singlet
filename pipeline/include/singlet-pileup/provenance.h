#pragma once
// singlet-pileup: provenance.h
// Pipeline provenance manifest writer (N8).
// Writes provenance.json at end of each pipeline run for reproducibility tracking.
//
// Integration: populate ProvenanceConfig and add to ExportConfig.provenance;
// export_results() calls write_provenance_json() automatically.
// See INTEGRATION_NOTES.md for singlify.cpp wiring instructions.

#include <chrono>
#include <ctime>
#include <fstream>
#include <string>

namespace singlet {

/// Configuration for the provenance manifest.
/// If input_file is empty, no provenance.json is written.
struct ProvenanceConfig {
    std::string singlify_version    = "0.3.0";
    std::string input_file;          ///< .1fq path (empty = skip provenance)
    uint64_t    input_reads         = 0;
    std::string genome_dir;
    std::string gtf_path;
    int         threads             = 0;
    bool        umi_dedup           = true;
    bool        umi_dedup_directional = false;
    bool        pipeline            = false;
    double      wall_seconds        = 0.0;
    double      star_seconds        = 0.0;
    double      pileup_seconds      = 0.0;
    /// Cell caller method used: "emptydrops", "knee_fallback", "top_n_fallback",
    /// "forced_cells", or "user_barcodes".  Empty = cell calling not run.
    std::string cell_caller;
};

/// Write {out_prefix}/provenance.json.
/// No-op if prov.input_file is empty.
///
/// @param out_prefix    Output directory (without trailing slash)
/// @param prov          Provenance configuration
/// @param n_exon_features  Number of rows in exon count matrix
/// @param n_cells          Number of cells (columns in exon count matrix)
/// @param total_umis       Sum of all exon count matrix entries
inline void write_provenance_json(
    const std::string& out_prefix,
    const ProvenanceConfig& prov,
    uint32_t n_exon_features,
    uint32_t n_cells,
    uint64_t total_umis)
{
    if (prov.input_file.empty()) return;

    // ISO 8601 UTC timestamp
    auto now   = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char ts[32] = {};
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));

    // Extract basename from path
    auto basename = [](const std::string& p) -> std::string {
        auto pos = p.rfind('/');
        return (pos == std::string::npos) ? p : p.substr(pos + 1);
    };

    // For genome_dir, return parent directory name (e.g. "GRCh38-2024-A")
    auto genome_label = [](const std::string& p) -> std::string {
        std::string s = p;
        // strip trailing slash
        while (!s.empty() && s.back() == '/') s.pop_back();
        auto pos1 = s.rfind('/');
        if (pos1 == std::string::npos) return s;
        std::string leaf = s.substr(pos1 + 1);
        // If leaf looks like a STAR index dir (e.g., "star_2.7.11b"), go up one more level
        if (leaf.find("star") != std::string::npos || leaf.find("STAR") != std::string::npos) {
            std::string parent = s.substr(0, pos1);
            auto pos2 = parent.rfind('/');
            if (pos2 != std::string::npos) return parent.substr(pos2 + 1);
            return parent;
        }
        return leaf;
    };

    // Minimal JSON string escaping
    auto esc = [](const std::string& s) -> std::string {
        std::string r;
        r.reserve(s.size());
        for (unsigned char c : s) {
            if (c == '"')       r += "\\\"";
            else if (c == '\\') r += "\\\\";
            else if (c < 0x20)  { r += "\\u00"; char h[3]; snprintf(h,3,"%02x",c); r+=h; }
            else                r += c;
        }
        return r;
    };

    std::ofstream f(out_prefix + "/provenance.json");
    if (!f) return;

    f << "{\n"
      << "  \"singlify_version\": \""  << esc(prov.singlify_version) << "\",\n"
      << "  \"timestamp\": \""         << ts << "\",\n"
      << "  \"input\": {\n"
      << "    \"file\": \""            << esc(basename(prov.input_file)) << "\",\n"
      << "    \"reads\": "             << prov.input_reads << "\n"
      << "  },\n"
      << "  \"reference\": {\n"
      << "    \"genome\": \""          << esc(genome_label(prov.genome_dir)) << "\",\n"
      << "    \"gtf\": \""             << esc(basename(prov.gtf_path)) << "\"\n"
      << "  },\n"
      << "  \"parameters\": {\n"
      << "    \"threads\": "           << prov.threads << ",\n"
      << "    \"umi_dedup\": "         << (prov.umi_dedup ? "true" : "false") << ",\n"
      << "    \"umi_dedup_directional\": " << (prov.umi_dedup_directional ? "true" : "false") << ",\n"
      << "    \"pipeline\": "          << (prov.pipeline ? "true" : "false") << "\n"
      << "  },\n"
      << "  \"cell_caller\": \""       << esc(prov.cell_caller) << "\",\n"
      << "  \"output\": {\n"
      << "    \"exon_features\": "     << n_exon_features << ",\n"
      << "    \"cells\": "             << n_cells << ",\n"
      << "    \"total_umis\": "        << total_umis << "\n"
      << "  },\n"
      << "  \"timings\": {\n"
      << "    \"wall_seconds\": "      << prov.wall_seconds << ",\n"
      << "    \"star_seconds\": "      << prov.star_seconds << ",\n"
      << "    \"pileup_seconds\": "    << prov.pileup_seconds << "\n"
      << "  }\n"
      << "}\n";
}

} // namespace singlet
