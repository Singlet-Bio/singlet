#pragma once
// singlet-pileup: provenance.h
// Pipeline provenance manifest writer (N8).
// Writes provenance.json at end of each pipeline run for reproducibility tracking.
//
// Integration: populate ProvenanceConfig and add to ExportConfig.provenance;
// export_results() calls write_provenance_json() automatically.
// See INTEGRATION_NOTES.md for singlet.cpp wiring instructions.

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <string>
#include <unistd.h>
#include <vector>

namespace singlet {

/// Configuration for the provenance manifest.
/// If input_file is empty, no provenance.json is written.
struct ProvenanceConfig {
    std::string singlet_version    = "0.3.0";
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
    std::string cell_caller;

    // §3.7 required fields
    std::string singlet_git_sha    = "unknown";  ///< set via CMake -DGIT_SHA=...
    std::vector<std::string> command_line;         ///< argv[0..argc-1]
    std::string snp_vcf_path;                      ///< --snps arg (for references block)
    std::string whitelist_name;                    ///< whitelist file basename

    // Track B cascade state (I-4, B-X-3)
    bool        cascade_enabled     = false;
    std::string cascade_mode        = "off";       ///< off, on, auto
    std::string te_classify_mode    = "off";       ///< off, on
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

    // Derive hostname and kernel string
    char hostname[256] = "unknown";
    gethostname(hostname, sizeof(hostname));
    hostname[sizeof(hostname)-1] = '\0';

    // Peak RSS from /proc/self/status (Linux)
    double peak_rss_gb = 0.0;
    {
        std::ifstream proc("/proc/self/status");
        std::string ln;
        while (std::getline(proc, ln)) {
            if (ln.rfind("VmPeak:", 0) == 0 || ln.rfind("VmRSS:", 0) == 0) {
                // format: "VmPeak:   123456 kB"
                const char* p = ln.c_str() + 7;
                while (*p == ' ' || *p == '\t') ++p;
                double kb = std::strtod(p, nullptr);
                if (kb > 0) { peak_rss_gb = kb / (1024.0 * 1024.0); break; }
            }
        }
    }

    std::ofstream f(out_prefix + "/provenance.json");
    if (!f) return;

    f << "{\n"
      << "  \"schema_version\": \"1.0\",\n"
      << "  \"singlet_git_sha\": \""  << esc(prov.singlet_git_sha) << "\",\n"
      << "  \"singlet_version\": \""  << esc(prov.singlet_version) << "\",\n"
      << "  \"timestamp\": \""         << ts << "\",\n"
      << "  \"build_flags\": [\"-O3\", \"-DNDEBUG\"],\n";

    // command_line array
    f << "  \"command_line\": [";
    for (size_t i = 0; i < prov.command_line.size(); ++i) {
        if (i > 0) f << ", ";
        f << "\"" << esc(prov.command_line[i]) << "\"";
    }
    f << "],\n";

    // env object
    auto env_get = [](const char* name) -> std::string {
        const char* v = std::getenv(name);
        return v ? std::string(v) : std::string("");
    };
    f << "  \"env\": {"
      << "\"OMP_NUM_THREADS\": \"" << esc(env_get("OMP_NUM_THREADS")) << "\""
      << ", \"TMPDIR\": \""        << esc(env_get("TMPDIR")) << "\""
      << "},\n";

    // host object
    f << "  \"host\": {"
      << "\"node\": \"" << esc(hostname) << "\""
      << ", \"ram_gb\": " << std::fixed << std::setprecision(1) << peak_rss_gb
      << "},\n";

    // input object
    f << "  \"input\": {\n"
      << "    \"file\": \""            << esc(basename(prov.input_file)) << "\",\n"
      << "    \"reads\": "             << prov.input_reads << "\n"
      << "  },\n";

    // references object
    f << "  \"references\": {\n"
      << "    \"genome\": {\"build\": \"" << esc(genome_label(prov.genome_dir)) << "\""
      << ", \"gtf\": \"" << esc(basename(prov.gtf_path)) << "\"},\n"
      << "    \"whitelist\": {\"name\": \"" << esc(prov.whitelist_name) << "\"},\n"
      << "    \"snp_vcf\": {\"name\": \"" << esc(basename(prov.snp_vcf_path)) << "\"}\n"
      << "  },\n";

    // parameters
    f << "  \"parameters\": {\n"
      << "    \"threads\": "           << prov.threads << ",\n"
      << "    \"umi_dedup\": "         << (prov.umi_dedup ? "true" : "false") << ",\n"
      << "    \"umi_dedup_directional\": " << (prov.umi_dedup_directional ? "true" : "false") << ",\n"
      << "    \"pipeline\": "          << (prov.pipeline ? "true" : "false") << "\n"
      << "  },\n";

    // Track B cascade state (I-4)
    f << "  \"cascade\": {\n"
      << "    \"enabled\": "           << (prov.cascade_enabled ? "true" : "false") << ",\n"
      << "    \"mode\": \""            << esc(prov.cascade_mode) << "\",\n"
      << "    \"te_classify\": \""     << esc(prov.te_classify_mode) << "\"\n"
      << "  },\n";

    f << "  \"cell_caller\": \""       << esc(prov.cell_caller) << "\",\n";

    // output block
    f << "  \"output\": {\n"
      << "    \"exon_features\": "     << n_exon_features << ",\n"
      << "    \"cells\": "             << n_cells << ",\n"
      << "    \"total_umis\": "        << total_umis << "\n"
      << "  },\n";

    // timings
    f << "  \"timings\": {\n"
      << "    \"wall_seconds\": "      << prov.wall_seconds << ",\n"
      << "    \"star_seconds\": "      << prov.star_seconds << ",\n"
      << "    \"pileup_seconds\": "    << prov.pileup_seconds << "\n"
      << "  },\n";

    f << "  \"output_schema_version\": \"1.0\"\n"
      << "}\n";
}

} // namespace singlet
