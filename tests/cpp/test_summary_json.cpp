// test_summary_json.cpp
// Unit tests for summary_json.h
// Tests: PipelineSummary serialization, classify_outcome, json_escape,
//        edge cases (zero/empty metrics), warnings, user_meta passthrough.
//
// Run standalone:
//   g++ -std=c++17 -O2 -I../include -o /tmp/test_summary_json \
//       test/test_summary_json.cpp && /tmp/test_summary_json

#include <algorithm>
#include <iostream>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/summary_json.h"

using namespace singlet;

// ── Helpers ──────────────────────────────────────────────────────────────────

static int s_passed = 0;
static int s_failed = 0;

static void check(const char* name, bool cond, const char* msg = "") {
    if (cond) {
        ++s_passed;
        std::cerr << "PASS: " << name << "\n";
    } else {
        ++s_failed;
        std::cerr << "FAIL: " << name;
        if (msg[0]) std::cerr << " -- " << msg;
        std::cerr << "\n";
    }
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

/// Count occurrences of `needle` in `haystack`
static int count_substr(const std::string& haystack, const std::string& needle) {
    int n = 0;
    std::string::size_type pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        ++n;
        pos += needle.size();
    }
    return n;
}

/// Number of unescaped '{' minus '}' in JSON (should be 0 for balanced)
static int brace_balance(const std::string& json) {
    int balance = 0;
    bool in_string = false;
    bool escaped = false;
    for (char c : json) {
        if (escaped) { escaped = false; continue; }
        if (c == '\\' && in_string) { escaped = true; continue; }
        if (c == '"') { in_string = !in_string; continue; }
        if (!in_string) {
            if (c == '{') ++balance;
            else if (c == '}') --balance;
        }
    }
    return balance;
}

// Build a fully-populated PipelineSummary for testing
static PipelineSummary make_full_summary() {
    PipelineSummary s;
    s.sample_id            = "GSM1234567";
    s.protocol             = "10xv3";
    s.organism             = "Homo sapiens";
    s.singlify_version     = "0.3.0";
    s.total_reads          = 50000000;
    s.mapped_reads         = 46000000;
    s.mapping_rate         = 0.92;
    s.exonic_fraction      = 0.62;
    s.intronic_fraction    = 0.17;
    s.intergenic_fraction  = 0.05;
    s.estimated_cells      = 4500;
    s.median_genes_per_cell = 2100.5;
    s.median_umis_per_cell  = 6800.0;
    s.mean_reads_per_cell   = 11111.0;
    s.total_genes_detected  = 19500;
    s.sequencing_saturation = 0.713;
    s.median_mito_fraction  = 0.035;
    s.fraction_reads_in_cells = 0.82;
    s.wall_seconds   = 340.5;
    s.star_seconds   = 280.0;
    s.pileup_seconds = 55.0;
    s.export_seconds = 5.5;
    s.exit_code      = 0;
    s.status         = "success";
    s.warnings.push_back("test_warning_A");
    s.user_meta["gsm_id"]   = "GSM1234567";
    s.user_meta["gse_id"]   = "GSE123456";
    s.user_meta["organism"] = "Homo sapiens";
    return s;
}

// ── Test 1: write_summary_json writes a file ──────────────────────────────────
static void test_write_creates_file() {
    const std::string path = "/tmp/test_summary_1.json";
    std::remove(path.c_str());
    auto s = make_full_summary();
    bool ok = write_summary_json(s, path);
    check("write_creates_file_returns_true", ok);
    std::ifstream f(path);
    check("write_creates_file_exists", f.good());
    std::remove(path.c_str());
}

// ── Test 2: Output contains all expected top-level keys ───────────────────────
static void test_all_keys_present() {
    const std::string path = "/tmp/test_summary_2.json";
    auto s = make_full_summary();
    write_summary_json(s, path);
    std::string json = read_file(path);

    check("key_sample_id",              contains(json, "\"sample_id\""));
    check("key_protocol",               contains(json, "\"protocol\""));
    check("key_organism",               contains(json, "\"organism\""));
    check("key_singlify_version",       contains(json, "\"singlify_version\""));
    check("key_total_reads",            contains(json, "\"total_reads\""));
    check("key_mapped_reads",           contains(json, "\"mapped_reads\""));
    check("key_mapping_rate",           contains(json, "\"mapping_rate\""));
    check("key_exonic_fraction",        contains(json, "\"exonic_fraction\""));
    check("key_intronic_fraction",      contains(json, "\"intronic_fraction\""));
    check("key_intergenic_fraction",    contains(json, "\"intergenic_fraction\""));
    check("key_estimated_cells",        contains(json, "\"estimated_cells\""));
    check("key_median_genes_per_cell",  contains(json, "\"median_genes_per_cell\""));
    check("key_median_umis_per_cell",   contains(json, "\"median_umis_per_cell\""));
    check("key_mean_reads_per_cell",    contains(json, "\"mean_reads_per_cell\""));
    check("key_total_genes_detected",   contains(json, "\"total_genes_detected\""));
    check("key_sequencing_saturation",  contains(json, "\"sequencing_saturation\""));
    check("key_median_mito_fraction",   contains(json, "\"median_mito_fraction\""));
    check("key_fraction_reads_in_cells",contains(json, "\"fraction_reads_in_cells\""));
    check("key_wall_seconds",           contains(json, "\"wall_seconds\""));
    check("key_star_seconds",           contains(json, "\"star_seconds\""));
    check("key_pileup_seconds",         contains(json, "\"pileup_seconds\""));
    check("key_export_seconds",         contains(json, "\"export_seconds\""));
    check("key_exit_code",              contains(json, "\"exit_code\""));
    check("key_status",                 contains(json, "\"status\""));
    check("key_warnings",               contains(json, "\"warnings\""));
    check("key_user_meta",              contains(json, "\"user_meta\""));
    std::remove(path.c_str());
}

// ── Test 3: JSON has balanced braces ─────────────────────────────────────────
static void test_json_balanced_braces() {
    const std::string path = "/tmp/test_summary_3.json";
    auto s = make_full_summary();
    write_summary_json(s, path);
    std::string json = read_file(path);
    check("json_balanced_braces", brace_balance(json) == 0, "unbalanced { }");
    check("json_starts_with_brace", json.size() > 0 && json[0] == '{');
    std::remove(path.c_str());
}

// ── Test 4: classify_outcome — success path ───────────────────────────────────
static void test_classify_outcome_success() {
    PipelineSummary s;
    s.total_reads          = 50000000;
    s.mapping_rate         = 0.92;
    s.estimated_cells      = 4500;
    s.median_genes_per_cell = 2100.0;
    check("classify_success", classify_outcome(s) == "success");
}

// ── Test 5: classify_outcome — low_mapping ────────────────────────────────────
static void test_classify_outcome_low_mapping() {
    PipelineSummary s;
    s.total_reads     = 10000000;
    s.mapping_rate    = 0.35;   // < 50% threshold
    s.estimated_cells = 500;
    check("classify_low_mapping", classify_outcome(s) == "low_mapping");
}

// ── Test 6: classify_outcome — no_cells ──────────────────────────────────────
static void test_classify_outcome_no_cells() {
    PipelineSummary s;
    s.total_reads     = 5000000;
    s.mapping_rate    = 0.85;
    s.estimated_cells = 0;
    check("classify_no_cells", classify_outcome(s) == "no_cells");
}

// ── Test 7: classify_outcome — low_cells ─────────────────────────────────────
static void test_classify_outcome_low_cells() {
    PipelineSummary s;
    s.total_reads      = 1000000;
    s.mapping_rate     = 0.80;
    s.estimated_cells  = 5;
    s.median_genes_per_cell = 1800.0;
    check("classify_low_cells", classify_outcome(s) == "low_cells");
}

// ── Test 8: classify_outcome — low_genes ─────────────────────────────────────
static void test_classify_outcome_low_genes() {
    PipelineSummary s;
    s.total_reads           = 2000000;
    s.mapping_rate          = 0.80;
    s.estimated_cells       = 150;
    s.median_genes_per_cell = 85.0;  // < 200 threshold
    check("classify_low_genes", classify_outcome(s) == "low_genes");
}

// ── Test 9: classify_outcome — zero total_reads → success (no-data) ──────────
static void test_classify_outcome_zero_reads() {
    PipelineSummary s;
    // all zeroes
    check("classify_zero_reads_is_success", classify_outcome(s) == "success");
}

// ── Test 10: classify_outcome — ATAC skips cell/gene checks ──────────────────
static void test_classify_outcome_atac() {
    PipelineSummary s;
    s.total_reads     = 10000000;
    s.mapping_rate    = 0.25;  // below atac floor (30%)
    s.estimated_cells = 0;
    check("classify_atac_low_mapping", classify_outcome(s, "atac") == "low_mapping");

    PipelineSummary s2;
    s2.total_reads     = 10000000;
    s2.mapping_rate    = 0.60;
    s2.estimated_cells = 0;  // no cells — but atac skips cell check
    check("classify_atac_no_cells_is_success", classify_outcome(s2, "atac") == "success");
}

// ── Test 11: Empty/zero PipelineSummary — must not crash ─────────────────────
static void test_empty_summary_no_crash() {
    const std::string path = "/tmp/test_summary_11.json";
    PipelineSummary s;
    bool ok = write_summary_json(s, path);
    check("empty_summary_no_crash", ok);
    std::string json = read_file(path);
    check("empty_summary_balanced", brace_balance(json) == 0);
    std::remove(path.c_str());
}

// ── Test 12: Warnings array — empty ──────────────────────────────────────────
static void test_warnings_empty() {
    const std::string path = "/tmp/test_summary_12.json";
    PipelineSummary s = make_full_summary();
    s.warnings.clear();
    write_summary_json(s, path);
    std::string json = read_file(path);
    check("warnings_empty_array", contains(json, "\"warnings\": []"));
    std::remove(path.c_str());
}

// ── Test 13: Warnings array — multiple entries ────────────────────────────────
static void test_warnings_multiple() {
    const std::string path = "/tmp/test_summary_13.json";
    PipelineSummary s = make_full_summary();
    s.warnings = {"low_mapping_rate", "few_cells_detected", "high_mito"};
    write_summary_json(s, path);
    std::string json = read_file(path);
    check("warnings_has_low_mapping",   contains(json, "\"low_mapping_rate\""));
    check("warnings_has_few_cells",     contains(json, "\"few_cells_detected\""));
    check("warnings_has_high_mito",     contains(json, "\"high_mito\""));
    std::remove(path.c_str());
}

// ── Test 14: user_meta passthrough ────────────────────────────────────────────
static void test_user_meta_passthrough() {
    const std::string path = "/tmp/test_summary_14.json";
    PipelineSummary s;
    s.user_meta["gsm_id"]   = "GSM9999999";
    s.user_meta["protocol"] = "10xv3";
    s.user_meta["taxon_id"] = "9606";
    write_summary_json(s, path);
    std::string json = read_file(path);
    check("meta_gsm_id_present",   contains(json, "\"gsm_id\""));
    check("meta_gsm_id_value",     contains(json, "\"GSM9999999\""));
    check("meta_protocol_present", contains(json, "\"taxon_id\""));
    check("meta_taxon_value",      contains(json, "\"9606\""));
    std::remove(path.c_str());
}

// ── Test 15: Wall/timing fields serialized ────────────────────────────────────
static void test_timing_fields() {
    const std::string path = "/tmp/test_summary_15.json";
    PipelineSummary s;
    s.wall_seconds   = 340.5;
    s.star_seconds   = 280.1;
    s.pileup_seconds = 55.2;
    s.export_seconds = 5.2;
    write_summary_json(s, path);
    std::string json = read_file(path);
    check("has_wall_seconds",   contains(json, "\"wall_seconds\""));
    check("has_star_seconds",   contains(json, "\"star_seconds\""));
    check("has_pileup_seconds", contains(json, "\"pileup_seconds\""));
    check("has_export_seconds", contains(json, "\"export_seconds\""));
    // Verify non-zero values appear in output
    check("wall_nonzero_written",   contains(json, "340.5"));
    check("pileup_nonzero_written", contains(json, "55.2"));
    std::remove(path.c_str());
}

// ── Test 16: json_escape special characters ───────────────────────────────────
static void test_json_escape() {
    check("escape_backslash",      json_escape("a\\b")     == "a\\\\b");
    check("escape_double_quote",   json_escape("say \"hi\"") == "say \\\"hi\\\"");
    check("escape_newline",        json_escape("line\nbreak") == "line\\nbreak");
    check("escape_carriage_return",json_escape("a\rb")     == "a\\rb");
    check("escape_tab",            json_escape("a\tb")     == "a\\tb");
    check("escape_plain",          json_escape("hello")    == "hello");
    check("escape_empty",          json_escape("")         == "");
}

// ── Test 17: Large uint64 values (overflow check) ─────────────────────────────
static void test_large_values() {
    const std::string path = "/tmp/test_summary_17.json";
    PipelineSummary s;
    s.total_reads  = UINT64_C(2000000000000);  // 2 trillion
    s.mapped_reads = UINT64_C(1900000000000);
    s.mapping_rate = 0.95;
    bool ok = write_summary_json(s, path);
    check("large_values_write_ok", ok);
    std::string json = read_file(path);
    check("large_total_reads_present",  contains(json, "2000000000000"));
    check("large_mapped_reads_present", contains(json, "1900000000000"));
    std::remove(path.c_str());
}

// ── Test 18: exit_code field ──────────────────────────────────────────────────
static void test_exit_code_field() {
    const std::string path = "/tmp/test_summary_18.json";
    PipelineSummary s;
    s.exit_code = 137;  // OOM
    s.status    = "crash";
    write_summary_json(s, path);
    std::string json = read_file(path);
    check("exit_code_written", contains(json, "\"exit_code\": 137"));
    check("status_crash",      contains(json, "\"crash\""));
    std::remove(path.c_str());
}

// ── Test 19: write to nonexistent directory → returns false ──────────────────
static void test_write_bad_path() {
    PipelineSummary s;
    bool ok = write_summary_json(s, "/nonexistent/dir/summary.json");
    check("write_bad_path_returns_false", !ok);
}

// ── Test 20: classify_outcome — mapping_rate exactly at threshold ──────────────
static void test_classify_boundary() {
    PipelineSummary s;
    s.total_reads     = 1000000;
    s.mapping_rate    = 0.50;   // exactly at threshold — should NOT be low_mapping
    s.estimated_cells = 100;
    s.median_genes_per_cell = 500.0;
    check("classify_exact_threshold_ok", classify_outcome(s) != "low_mapping");

    PipelineSummary s2;
    s2.total_reads     = 1000000;
    s2.mapping_rate    = 0.499;  // just below threshold
    s2.estimated_cells = 100;
    check("classify_below_threshold_low", classify_outcome(s2) == "low_mapping");
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    test_write_creates_file();
    test_all_keys_present();
    test_json_balanced_braces();
    test_classify_outcome_success();
    test_classify_outcome_low_mapping();
    test_classify_outcome_no_cells();
    test_classify_outcome_low_cells();
    test_classify_outcome_low_genes();
    test_classify_outcome_zero_reads();
    test_classify_outcome_atac();
    test_empty_summary_no_crash();
    test_warnings_empty();
    test_warnings_multiple();
    test_user_meta_passthrough();
    test_timing_fields();
    test_json_escape();
    test_large_values();
    test_exit_code_field();
    test_write_bad_path();
    test_classify_boundary();

    std::cerr << "\n=== Summary: " << s_passed << " passed, " << s_failed << " failed ===\n";
    return (s_failed > 0) ? 1 : 0;
}
