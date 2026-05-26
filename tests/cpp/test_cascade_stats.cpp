// test_cascade_stats.cpp — Unit tests for cascade_stats_writer.h (T-L2-8)
// Tests: write_cascade_stats JSON structure, field values, schema version.

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "singlet/pileup/cascade_stats_writer.h"

using namespace singlet;

static int g_pass = 0, g_fail = 0;

static void CHECK(bool cond, const char* name) {
    if (cond) { ++g_pass; }
    else { ++g_fail; std::fprintf(stderr, "FAIL: %s\n", name); }
}

static std::string read_file_str(const std::string& path) {
    std::ifstream f(path);
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

// ── Test 1: basic write succeeds ─────────────────────────────────────────────

static void test_basic_write() {
    CascadeStats cs;
    cs.L1_txome.reads_in = 1000000;
    cs.L1_txome.resolved = 300000;
    cs.L1_txome.passthrough = 700000;
    cs.L2_te.reads_in = 700000;
    cs.L2_te.resolved = 50000;
    cs.L2_te.passthrough = 650000;
    cs.L3_star.reads_in = 650000;
    cs.L3_star.mapped = 600000;
    cs.L3_star.unmapped = 50000;
    cs.L4_nonhost.reads_in = 50000;
    cs.L4_nonhost.resolved = 10000;
    cs.L4_nonhost.unmappable = 40000;

    std::string dir = "/tmp/test_cascade_stats_out";
    system(("mkdir -p " + dir).c_str());
    bool ok = write_cascade_stats(dir, cs);
    CHECK(ok, "write_succeeds");

    std::string json = read_file_str(dir + "/cascade_stats.json");
    CHECK(!json.empty(), "file_not_empty");

    // Check key fields
    CHECK(json.find("\"schema_version\": \"1.1\"") != std::string::npos, "schema_version");
    CHECK(json.find("\"track\": \"B\"") != std::string::npos, "track_B");
    CHECK(json.find("\"cascade_enabled\": true") != std::string::npos, "cascade_enabled");
    CHECK(json.find("\"deterministic\": true") != std::string::npos, "deterministic");
    CHECK(json.find("\"em_seed\": \"0xC0FFEE\"") != std::string::npos, "em_seed");

    // Layer values
    CHECK(json.find("\"reads_in\": 1000000") != std::string::npos, "L1_reads_in");
    CHECK(json.find("\"resolved\": 300000") != std::string::npos, "L1_resolved");
    CHECK(json.find("\"mapped\": 600000") != std::string::npos, "L3_mapped");
    CHECK(json.find("\"unmappable\": 40000") != std::string::npos, "L4_unmappable");

    // Clean up
    system(("rm -rf " + dir).c_str());
}

// ── Test 2: default values ───────────────────────────────────────────────────

static void test_defaults() {
    CascadeStats cs;  // all defaults
    CHECK(cs.em_seed == 0xC0FFEE, "default_seed");
    CHECK(cs.cascade_enabled == true, "default_enabled");
    CHECK(cs.deterministic == true, "default_deterministic");
    CHECK(cs.L1_txome.reads_in == 0, "default_L1_reads=0");
    CHECK(cs.L1_txome.wall_seconds == 0.0, "default_L1_time=0");

    CascadeLayerStats layer;
    CHECK(layer.reads_in == 0, "layer_default_reads=0");
    CHECK(layer.resolved == 0, "layer_default_resolved=0");
    CHECK(layer.peak_rss_gb == 0.0, "layer_default_rss=0");
}

// ── Test 3: JSON structure valid ─────────────────────────────────────────────

static void test_json_structure() {
    CascadeStats cs;
    cs.L1_txome.reads_in = 100;
    cs.L1_txome.wall_seconds = 1.5;
    cs.L3_star.peak_rss_gb = 28.3;

    std::string dir = "/tmp/test_cascade_stats_struct";
    system(("mkdir -p " + dir).c_str());
    write_cascade_stats(dir, cs);
    std::string json = read_file_str(dir + "/cascade_stats.json");

    // Check balanced braces
    int brace_count = 0;
    for (char c : json) {
        if (c == '{') ++brace_count;
        if (c == '}') --brace_count;
    }
    CHECK(brace_count == 0, "balanced_braces");

    // Check no trailing comma before }
    // Simple check: no ",\n  }" pattern
    CHECK(json.find(",\n}") == std::string::npos, "no_trailing_comma_top");

    // Check layers section exists
    CHECK(json.find("\"layers\"") != std::string::npos, "layers_section");
    CHECK(json.find("\"L1_txome\"") != std::string::npos, "L1_section");
    CHECK(json.find("\"L2_te\"") != std::string::npos, "L2_section");
    CHECK(json.find("\"L3_star\"") != std::string::npos, "L3_section");
    CHECK(json.find("\"L4_nonhost\"") != std::string::npos, "L4_section");

    // Check timing section
    CHECK(json.find("\"timing_seconds\"") != std::string::npos, "timing_section");
    CHECK(json.find("\"peak_rss_gb_per_layer\"") != std::string::npos, "rss_section");

    system(("rm -rf " + dir).c_str());
}

// ── Test 4: disabled cascade ─────────────────────────────────────────────────

static void test_disabled_cascade() {
    CascadeStats cs;
    cs.cascade_enabled = false;
    cs.deterministic = false;
    cs.em_seed = 42;

    std::string dir = "/tmp/test_cascade_stats_disabled";
    system(("mkdir -p " + dir).c_str());
    write_cascade_stats(dir, cs);
    std::string json = read_file_str(dir + "/cascade_stats.json");

    CHECK(json.find("\"cascade_enabled\": false") != std::string::npos, "disabled");
    CHECK(json.find("\"deterministic\": false") != std::string::npos, "non_deterministic");
    CHECK(json.find("\"em_seed\": \"0x2A\"") != std::string::npos, "seed_0x2A");

    system(("rm -rf " + dir).c_str());
}

// ── Test 5: write to nonexistent directory fails ─────────────────────────────

static void test_write_failure() {
    CascadeStats cs;
    bool ok = write_cascade_stats("/tmp/test_cascade_stats_NOPE/subdir/deep", cs);
    CHECK(!ok, "write_to_nonexistent_dir_fails");
}

// ── Test 6: large values don't overflow ──────────────────────────────────────

static void test_large_values() {
    CascadeStats cs;
    cs.L1_txome.reads_in = 5000000000ULL;  // 5 billion
    cs.L1_txome.resolved = 3000000000ULL;
    cs.L3_star.mapped = 1500000000ULL;

    std::string dir = "/tmp/test_cascade_stats_large";
    system(("mkdir -p " + dir).c_str());
    write_cascade_stats(dir, cs);
    std::string json = read_file_str(dir + "/cascade_stats.json");

    CHECK(json.find("5000000000") != std::string::npos, "large_reads_in");
    CHECK(json.find("3000000000") != std::string::npos, "large_resolved");

    system(("rm -rf " + dir).c_str());
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    test_basic_write();
    test_defaults();
    test_json_structure();
    test_disabled_cascade();
    test_write_failure();
    test_large_values();

    std::printf("cascade_stats: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
