#pragma once
// singlet-pileup: cascade_stats_writer.h  — T-L2-8
// Writes cascade_stats.json alongside summary.json when cascade is active.
// Schema: DROPLET_OUTPUT_SCHEMA.md §6.6 (v1.1).
//
// Integration point: call write_cascade_stats() in the singlet success-path
// write block, after pileup_stats.json is written, when cascade is enabled.
// See INTEGRATION_NOTES.md §Feature: CascadeStatsWriter (T-L2-8).

#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>

namespace singlet {

// Per-layer resolution counters.  Fill these as reads are routed.
struct CascadeLayerStats {
    uint64_t reads_in    = 0;
    uint64_t resolved    = 0;  // L1: unique txome hit; L2: TE family hit
    uint64_t passthrough = 0;  // reads forwarded to the next layer
    uint64_t mapped      = 0;  // L3 STAR mapped
    uint64_t unmapped    = 0;  // L3 STAR unmapped → L4
    uint64_t unmappable  = 0;  // L4 final unassigned
    double   wall_seconds = 0.0;
    double   peak_rss_gb  = 0.0;
};

struct CascadeStats {
    CascadeLayerStats L1_txome;
    CascadeLayerStats L2_te;
    CascadeLayerStats L3_star;
    CascadeLayerStats L4_nonhost;
    uint64_t em_seed = 0xC0FFEE;
    bool     cascade_enabled = true;
    bool     deterministic   = true;
};

/// Write cascade_stats.json to `out_prefix/cascade_stats.json`.
/// Returns true on success, false if the file cannot be written.
inline bool write_cascade_stats(const std::string& out_prefix,
                                 const CascadeStats& cs)
{
    std::string path = out_prefix + "/cascade_stats.json";
    std::ofstream f(path);
    if (!f) return false;

    auto hex_u64 = [](uint64_t v) -> std::string {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "0x%llX",
                      static_cast<unsigned long long>(v));
        return buf;
    };

    f << "{\n";
    f << "  \"schema_version\": \"1.1\",\n";
    f << "  \"track\": \"B\",\n";
    f << "  \"cascade_enabled\": " << (cs.cascade_enabled ? "true" : "false") << ",\n";
    f << "  \"deterministic\": "   << (cs.deterministic   ? "true" : "false") << ",\n";
    f << "  \"em_seed\": \""       << hex_u64(cs.em_seed)                     << "\",\n";

    // ── Layers ──
    f << "  \"layers\": {\n";

    // L1
    f << "    \"L1_txome\": {\n";
    f << "      \"reads_in\": "    << cs.L1_txome.reads_in    << ",\n";
    f << "      \"resolved\": "    << cs.L1_txome.resolved    << ",\n";
    f << "      \"passthrough\": " << cs.L1_txome.passthrough << "\n";
    f << "    },\n";

    // L2
    f << "    \"L2_te\": {\n";
    f << "      \"reads_in\": "    << cs.L2_te.reads_in    << ",\n";
    f << "      \"resolved\": "    << cs.L2_te.resolved    << ",\n";
    f << "      \"passthrough\": " << cs.L2_te.passthrough << "\n";
    f << "    },\n";

    // L3
    f << "    \"L3_star\": {\n";
    f << "      \"reads_in\": "  << cs.L3_star.reads_in  << ",\n";
    f << "      \"mapped\": "    << cs.L3_star.mapped    << ",\n";
    f << "      \"unmapped\": "  << cs.L3_star.unmapped  << "\n";
    f << "    },\n";

    // L4
    f << "    \"L4_nonhost\": {\n";
    f << "      \"reads_in\": "    << cs.L4_nonhost.reads_in    << ",\n";
    f << "      \"resolved\": "    << cs.L4_nonhost.resolved    << ",\n";
    f << "      \"unmappable\": "  << cs.L4_nonhost.unmappable  << "\n";
    f << "    }\n";

    f << "  },\n";

    // ── Timing ──
    f << "  \"timing_seconds\": {\n";
    f << "    \"L1\": " << cs.L1_txome.wall_seconds  << ",\n";
    f << "    \"L2\": " << cs.L2_te.wall_seconds     << ",\n";
    f << "    \"L3\": " << cs.L3_star.wall_seconds   << ",\n";
    f << "    \"L4\": " << cs.L4_nonhost.wall_seconds << "\n";
    f << "  },\n";

    // ── Peak RSS ──
    f << "  \"peak_rss_gb_per_layer\": {\n";
    f << "    \"L1\": " << cs.L1_txome.peak_rss_gb   << ",\n";
    f << "    \"L2\": " << cs.L2_te.peak_rss_gb      << ",\n";
    f << "    \"L3\": " << cs.L3_star.peak_rss_gb    << ",\n";
    f << "    \"L4\": " << cs.L4_nonhost.peak_rss_gb  << "\n";
    f << "  }\n";

    f << "}\n";
    return f.good();
}

}  // namespace singlet
