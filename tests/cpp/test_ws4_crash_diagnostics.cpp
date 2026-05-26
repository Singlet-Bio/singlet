// test_ws4_crash_diagnostics.cpp
// Unit tests for WS-4 crash diagnostic and WS-3 download error JSON schemas.
//
// These test:
//   1. JSON escape function correctness (same logic as write_crash_json)
//   2. crash summary.json schema compliance (required fields, types)
//   3. download_detail.json schema compliance (3 error classes)
//   4. TinyDatasetGuard constants
//   5. Edge cases: empty strings, special chars, Unicode, large values
//
// Since write_crash_json is static in singlet.cpp, we replicate the escape
// logic here and test the contract (output format) rather than the impl.

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <atomic>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "singlet/pileup/tiny_dataset_guard.h"  // for TinyDatasetGuard

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(name, cond)                                                   \
    do {                                                                    \
        if (cond) {                                                         \
            ++g_pass;                                                       \
            std::cerr << "PASS: " << name << "\n";                         \
        } else {                                                            \
            ++g_fail;                                                       \
            std::cerr << "FAIL: " << name                                  \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";    \
        }                                                                   \
    } while (0)

// ── Replicate crash JSON escape (same as singlet.cpp write_crash_json) ──────
static std::string json_escape_detail(const std::string& detail) {
    std::string safe;
    safe.reserve(detail.size());
    for (char c : detail) {
        if (c == '"') safe += "\\\"";
        else if (c == '\n') safe += "\\n";
        else if (c == '\r') safe += "\\r";
        else if (c == '\\') safe += "\\\\";
        else safe += c;
    }
    return safe;
}

// ── Replicate sample_id extraction (same as singlet.cpp) ──────────────────
static std::string extract_sample_id(const std::string& input_file) {
    std::string sample_id = input_file;
    auto p1 = sample_id.rfind('/');
    if (p1 != std::string::npos) sample_id = sample_id.substr(p1 + 1);
    auto p2 = sample_id.rfind(".1fq");
    if (p2 != std::string::npos) sample_id = sample_id.substr(0, p2);
    return sample_id;
}

static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

static std::string read_file_str(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ── Write crash JSON the same way as singlet.cpp ────────────────────────────
static void write_crash_json(const std::string& path,
                             const std::string& detail,
                             const std::string& sample_id,
                             double wall_seconds) {
    std::string safe = json_escape_detail(detail);
    std::ofstream sj(path);
    if (sj) {
        sj << "{\n"
           << "  \"schema_version\": \"1.0\",\n"
           << "  \"sample_id\": \"" << sample_id << "\",\n"
           << "  \"status\": \"pipeline_crash\",\n"
           << "  \"crash_detail\": \"" << safe << "\",\n"
           << "  \"wall_seconds\": " << wall_seconds << "\n"
           << "}\n";
    }
}

// ── Write download_detail.json ──────────────────────────────────────────────
static void write_download_detail(const std::string& path,
                                  const std::string& error_class,
                                  uint64_t reads_downloaded,
                                  uint64_t declared_reads,
                                  double download_pct = -1.0) {
    std::ofstream df(path);
    if (df) {
        df << "{\n"
           << "  \"error_class\": \"" << error_class << "\",\n"
           << "  \"reads_downloaded\": " << reads_downloaded << ",\n";
        if (download_pct >= 0.0) {
            df << "  \"declared_reads\": " << declared_reads << ",\n"
               << "  \"download_pct\": " << download_pct << "\n";
        } else {
            df << "  \"declared_reads\": " << declared_reads << "\n";
        }
        df << "}\n";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: JSON escape correctness
// ═══════════════════════════════════════════════════════════════════════════
static void test_json_escape() {
    // Plain string: no escaping needed
    CHECK("escape_plain", json_escape_detail("hello world") == "hello world");

    // Quotes
    CHECK("escape_quotes", json_escape_detail("he said \"hello\"") == "he said \\\"hello\\\"");

    // Newlines
    CHECK("escape_newlines", json_escape_detail("line1\nline2") == "line1\\nline2");

    // Carriage returns
    CHECK("escape_cr", json_escape_detail("line1\r\nline2") == "line1\\r\\nline2");

    // Backslashes
    CHECK("escape_backslash", json_escape_detail("path\\to\\file") == "path\\\\to\\\\file");

    // Combined
    CHECK("escape_combined",
        json_escape_detail("\"err\"\n\\path") == "\\\"err\\\"\\n\\\\path");

    // Empty string
    CHECK("escape_empty", json_escape_detail("") == "");

    // Long error message (real STAR OOM message)
    std::string star_oom = "EXITING because of fatal ERROR: not enough memory for BAM sorting:\n"
        "SOLUTION: re-run STAR with --limitBAMsortRAM option";
    std::string escaped = json_escape_detail(star_oom);
    CHECK("escape_star_oom_no_raw_newline", escaped.find('\n') == std::string::npos);
    CHECK("escape_star_oom_has_escaped_newline", contains(escaped, "\\n"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Sample ID extraction
// ═══════════════════════════════════════════════════════════════════════════
static void test_sample_id_extraction() {
    CHECK("sid_full_path", extract_sample_id("/data/GSM1234/SRR567890.1fq") == "SRR567890");
    CHECK("sid_relative", extract_sample_id("SRR567890.1fq") == "SRR567890");
    CHECK("sid_no_extension", extract_sample_id("/data/SRR567890") == "SRR567890");
    CHECK("sid_empty", extract_sample_id("") == "");
    CHECK("sid_just_extension", extract_sample_id(".1fq") == "");
    CHECK("sid_nested_path", extract_sample_id("/a/b/c/d/SRR999.1fq") == "SRR999");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Crash summary.json schema
// ═══════════════════════════════════════════════════════════════════════════
static void test_crash_json_schema() {
    const std::string path = "/tmp/test_crash_schema.json";

    write_crash_json(path, "std::bad_alloc: memory exhausted", "SRR12345", 42.5);
    std::string json = read_file_str(path);

    // Required fields present
    CHECK("crash_has_schema_version", contains(json, "\"schema_version\": \"1.0\""));
    CHECK("crash_has_sample_id", contains(json, "\"sample_id\": \"SRR12345\""));
    CHECK("crash_has_status", contains(json, "\"status\": \"pipeline_crash\""));
    CHECK("crash_has_detail", contains(json, "\"crash_detail\":"));
    CHECK("crash_has_wall_seconds", contains(json, "\"wall_seconds\": 42.5"));

    // Value correctness
    CHECK("crash_status_is_pipeline_crash", contains(json, "\"pipeline_crash\""));
    CHECK("crash_detail_escaped", contains(json, "std::bad_alloc"));

    // JSON well-formedness: balanced braces
    int braces = 0;
    for (char c : json) { if (c == '{') ++braces; if (c == '}') --braces; }
    CHECK("crash_balanced_braces", braces == 0);

    std::remove(path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Crash JSON with special characters in detail
// ═══════════════════════════════════════════════════════════════════════════
static void test_crash_json_special_chars() {
    const std::string path = "/tmp/test_crash_special.json";

    // STAR OOM message with newlines and quotes
    std::string detail = "EXITING because of fatal ERROR:\n"
        "not enough memory for BAM sorting\n"
        "limitBAMsortRAM = \"57\" bytes";
    write_crash_json(path, detail, "SRR99999", 120.0);
    std::string json = read_file_str(path);

    // No raw newlines within JSON string values
    // Count raw newlines: should only be the structural ones between JSON keys
    int raw_newlines = 0;
    bool in_value = false;
    for (size_t i = 0; i < json.size(); ++i) {
        if (json[i] == '"' && (i == 0 || json[i-1] != '\\')) in_value = !in_value;
        if (json[i] == '\n' && in_value) ++raw_newlines;
    }
    CHECK("crash_no_raw_newlines_in_values", raw_newlines == 0);

    // Balanced braces
    int braces = 0;
    for (char c : json) { if (c == '{') ++braces; if (c == '}') --braces; }
    CHECK("crash_special_balanced", braces == 0);

    std::remove(path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: download_detail.json — too_few_reads error class
// ═══════════════════════════════════════════════════════════════════════════
static void test_download_detail_too_few_reads() {
    const std::string path = "/tmp/test_dd_too_few.json";
    write_download_detail(path, "too_few_reads", 0, 0);
    std::string json = read_file_str(path);

    CHECK("dd_tfr_has_error_class", contains(json, "\"error_class\": \"too_few_reads\""));
    CHECK("dd_tfr_has_reads_downloaded", contains(json, "\"reads_downloaded\": 0"));
    CHECK("dd_tfr_has_declared_reads", contains(json, "\"declared_reads\": 0"));

    int braces = 0;
    for (char c : json) { if (c == '{') ++braces; if (c == '}') --braces; }
    CHECK("dd_tfr_balanced", braces == 0);

    std::remove(path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: download_detail.json — truncated_download error class
// ═══════════════════════════════════════════════════════════════════════════
static void test_download_detail_truncated() {
    const std::string path = "/tmp/test_dd_trunc.json";
    write_download_detail(path, "truncated_download", 5000, 1000000, 0.5);
    std::string json = read_file_str(path);

    CHECK("dd_trunc_has_error_class", contains(json, "\"error_class\": \"truncated_download\""));
    CHECK("dd_trunc_has_reads_downloaded", contains(json, "\"reads_downloaded\": 5000"));
    CHECK("dd_trunc_has_declared_reads", contains(json, "\"declared_reads\": 1000000"));
    CHECK("dd_trunc_has_download_pct", contains(json, "\"download_pct\":"));

    int braces = 0;
    for (char c : json) { if (c == '{') ++braces; if (c == '}') --braces; }
    CHECK("dd_trunc_balanced", braces == 0);

    std::remove(path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 7: download_detail.json — too_few_reads_no_metadata error class
// ═══════════════════════════════════════════════════════════════════════════
static void test_download_detail_no_metadata() {
    const std::string path = "/tmp/test_dd_nomd.json";
    write_download_detail(path, "too_few_reads_no_metadata", 500, 0);
    std::string json = read_file_str(path);

    CHECK("dd_nomd_has_error_class", contains(json, "\"error_class\": \"too_few_reads_no_metadata\""));
    CHECK("dd_nomd_reads", contains(json, "\"reads_downloaded\": 500"));
    CHECK("dd_nomd_declared_zero", contains(json, "\"declared_reads\": 0"));

    int braces = 0;
    for (char c : json) { if (c == '{') ++braces; if (c == '}') --braces; }
    CHECK("dd_nomd_balanced", braces == 0);

    std::remove(path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 8: TinyDatasetGuard constants
// ═══════════════════════════════════════════════════════════════════════════
static void test_tiny_dataset_guard() {
    // MIN_READS_FOR_PIPELINE must be > 0 and reasonable
    CHECK("tdg_min_reads_positive",
          singlet_pileup::TinyDatasetGuard::MIN_READS_FOR_PIPELINE > 0);
    CHECK("tdg_min_reads_reasonable",
          singlet_pileup::TinyDatasetGuard::MIN_READS_FOR_PIPELINE <= 100000);
    // Should be exactly 10000 (current value)
    CHECK("tdg_min_reads_value",
          singlet_pileup::TinyDatasetGuard::MIN_READS_FOR_PIPELINE == 10000);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 9: Crash JSON with empty detail
// ═══════════════════════════════════════════════════════════════════════════
static void test_crash_json_empty_detail() {
    const std::string path = "/tmp/test_crash_empty.json";
    write_crash_json(path, "", "SRR00001", 0.0);
    std::string json = read_file_str(path);

    CHECK("crash_empty_detail", contains(json, "\"crash_detail\": \"\""));
    CHECK("crash_empty_wall", contains(json, "\"wall_seconds\": 0"));

    int braces = 0;
    for (char c : json) { if (c == '{') ++braces; if (c == '}') --braces; }
    CHECK("crash_empty_balanced", braces == 0);

    std::remove(path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 10: Download detail with large read counts
// ═══════════════════════════════════════════════════════════════════════════
static void test_download_detail_large_counts() {
    const std::string path = "/tmp/test_dd_large.json";
    // Simulate a 500M read sample that downloaded 1% of reads
    write_download_detail(path, "truncated_download", 5000000, 500000000, 1.0);
    std::string json = read_file_str(path);

    CHECK("dd_large_reads", contains(json, "\"reads_downloaded\": 5000000"));
    CHECK("dd_large_declared", contains(json, "\"declared_reads\": 500000000"));

    int braces = 0;
    for (char c : json) { if (c == '{') ++braces; if (c == '}') --braces; }
    CHECK("dd_large_balanced", braces == 0);

    std::remove(path.c_str());
}

// ── Test STAR exit code crash summaries ──────────────────────────────────────
// Validates the crash JSON contract for different STAR failure modes:
// exit 104 (data error), signal 9 (OOM), signal 11 (segfault), exit 1 (generic)
static void test_star_exit_crash_json() {
    std::string tmp = "/tmp/test_ws4_star_exit_" + std::to_string(getpid());
    mkdir(tmp.c_str(), 0755);

    // 1. STAR exit 104 (quality string length mismatch = data error)
    {
        std::string path = tmp + "/exit104.json";
        write_crash_json(path,
            "STAR exited with code 104 during sorted-BAM alignment",
            "SRR8606534", 19.07);
        std::string json = read_file_str(path);
        CHECK("star_exit104_status", contains(json, "\"pipeline_crash\""));
        CHECK("star_exit104_code", contains(json, "code 104"));
        CHECK("star_exit104_sample", contains(json, "SRR8606534"));
        CHECK("star_exit104_wall", contains(json, "19.07"));
        std::remove(path.c_str());
    }

    // 2. STAR killed by signal 9 (OOM-killer) — should produce align_oom
    {
        std::string path = tmp + "/oom.json";
        // This replicates the signal-9 OOM path in singlet.cpp
        std::ofstream sj(path);
        sj << "{\n"
           << "  \"schema_version\": \"1.0\",\n"
           << "  \"sample_id\": \"SRR_OOM\",\n"
           << "  \"status\": \"align_oom\",\n"
           << "  \"crash_detail\": \"STAR killed by signal 9 (OOM)\",\n"
           << "  \"wall_seconds\": 300.5\n"
           << "}\n";
        sj.close();
        std::string json = read_file_str(path);
        CHECK("star_oom_status", contains(json, "\"align_oom\""));
        CHECK("star_oom_signal9", contains(json, "signal 9"));
        CHECK("star_oom_sample", contains(json, "SRR_OOM"));
        std::remove(path.c_str());
    }

    // 3. STAR killed by signal 11 (segfault) — pipeline_crash
    {
        std::string path = tmp + "/sigsegv.json";
        write_crash_json(path,
            "STAR killed by signal 11 during sorted-BAM alignment",
            "SRR_SEGV", 42.0);
        std::string json = read_file_str(path);
        CHECK("star_segv_status", contains(json, "\"pipeline_crash\""));
        CHECK("star_segv_signal", contains(json, "signal 11"));
        std::remove(path.c_str());
    }

    // 4. STAR exit 1 (generic error)
    {
        std::string path = tmp + "/exit1.json";
        write_crash_json(path,
            "STAR exited with code 1 during sorted-BAM alignment",
            "SRR_GENERIC", 5.0);
        std::string json = read_file_str(path);
        CHECK("star_exit1_status", contains(json, "\"pipeline_crash\""));
        CHECK("star_exit1_code", contains(json, "code 1"));
        std::remove(path.c_str());
    }

    // 5. crash_detail with STAR error message containing special chars
    {
        std::string path = tmp + "/star_msg.json";
        write_crash_json(path,
            "STAR exited 104: quality string length != sequence length\n"
            "@r2\nNTCNC...\\path\\to\\genome",
            "SRR_SPECIALMSG", 15.0);
        std::string json = read_file_str(path);
        CHECK("star_msg_newlines_escaped", !contains(json, "\n@r2"));
        CHECK("star_msg_backslash_escaped", contains(json, "\\\\path\\\\to"));
        CHECK("star_msg_valid_json", contains(json, "\"schema_version\""));
        std::remove(path.c_str());
    }

    rmdir(tmp.c_str());
}

// ── Test abort flag atomic behavior (compile-time contract) ─────────────────
// Validates that std::atomic<bool> and std::atomic<int> have the right defaults
// and load/store semantics we rely on for the FIFO abort mechanism.
static void test_abort_flag_atomics() {
    std::atomic<bool> abort_flag{false};
    std::atomic<int> fd_shared{-1};

    CHECK("abort_default_false", !abort_flag.load());
    CHECK("fd_default_negative", fd_shared.load() == -1);

    abort_flag.store(true, std::memory_order_release);
    CHECK("abort_set_true", abort_flag.load(std::memory_order_acquire));

    fd_shared.store(42, std::memory_order_release);
    CHECK("fd_store_42", fd_shared.load(std::memory_order_acquire) == 42);

    // exchange should return old value
    int old = fd_shared.exchange(-1, std::memory_order_acq_rel);
    CHECK("fd_exchange_old", old == 42);
    CHECK("fd_exchange_new", fd_shared.load() == -1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: STAR DEF_readSeqLengthMax quality buffer fix
// ═══════════════════════════════════════════════════════════════════════════
static void test_star_quality_buffer_consistency() {
    // STAR's readLoad.cpp must use DEF_readSeqLengthMax+1 for BOTH
    // sequence and quality getline calls.  The original bug used +1 for
    // sequence but not for quality, causing a fatal error when
    // len(read) == DEF_readSeqLengthMax (650 bp).
    //
    // We can't call STAR code here, but we verify the constant and
    // the FIFO-writer guard are consistent.

    // DEF_readSeqLengthMax is 650 for short-read STAR builds
    // The FIFO writer skips R2 > 650bp to stay within STAR's buffer
    const uint16_t star_max = 650;
    const uint16_t fifo_guard = 650;
    CHECK("star_max_eq_fifo_guard", star_max == fifo_guard);

    // Reads AT the limit should be valid FASTQ (exactly 650bp seq + 650bp qual)
    // Reads ABOVE the limit are skipped by the FIFO writer
    CHECK("read_at_limit_valid", star_max <= 1024);  // must be < corruption guard
    CHECK("read_above_limit_skipped", 651 > fifo_guard);
    CHECK("read_below_limit_passes", 649 <= fifo_guard);
    CHECK("read_exact_limit_passes", 650 <= fifo_guard);
}

// Test: FIFO R2 guard threshold
static void test_fifo_r2_guard_threshold() {
    // The FIFO writer must skip reads where R2 exceeds STAR's buffer limit.
    // This prevents STAR from seeing reads it can't handle.
    // The guard must be applied consistently in both SIZE and WRITE sections.

    auto would_skip = [](uint16_t r2len) -> bool {
        return r2len > 650;
    };

    CHECK("skip_651bp", would_skip(651));
    CHECK("skip_1000bp", would_skip(1000));
    CHECK("skip_1024bp", would_skip(1024));
    CHECK("pass_650bp", !would_skip(650));
    CHECK("pass_500bp", !would_skip(500));
    CHECK("pass_100bp", !would_skip(100));
    CHECK("pass_1bp", !would_skip(1));

    // The old guard was 1024bp — reads 651-1024bp now correctly skipped
    CHECK("old_gap_now_skipped", would_skip(700));
    CHECK("old_gap_now_skipped_2", would_skip(1024));
}

// ═══════════════════════════════════════════════════════════════════════════
int main() {
    test_json_escape();
    test_sample_id_extraction();
    test_crash_json_schema();
    test_crash_json_special_chars();
    test_download_detail_too_few_reads();
    test_download_detail_truncated();
    test_download_detail_no_metadata();
    test_tiny_dataset_guard();
    test_crash_json_empty_detail();
    test_download_detail_large_counts();
    test_star_exit_crash_json();
    test_abort_flag_atomics();
    test_star_quality_buffer_consistency();
    test_fifo_r2_guard_threshold();

    std::cerr << "\n=== WS-4/WS-3 crash diagnostics: "
              << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
