// test_logging.cpp — unit tests for singlify_log.h (G-LOGGING)
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "singlify_log.h"

using namespace singlet;

// ── helpers ────────────────────────────────────────────────────────────────

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

static int count_lines(const std::string& s) {
    int n = 0;
    for (char c : s)
        if (c == '\n') ++n;
    return n;
}

[[maybe_unused]] static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// Reset singleton to known state between tests
static void reset_logger() {
    auto& L = SinglifyLogger::instance();
    L.close();
    L.set_level(LogLevel::NORMAL);
    L.set_stderr(false);  // suppress stderr in tests
}

// ── Test 1: level filtering ────────────────────────────────────────────────

static void test_level_filtering() {
    reset_logger();
    auto& L = SinglifyLogger::instance();

    // Only NORMAL messages should pass at NORMAL level
    // (compile-time check: verbose/debug methods exist and accept format strings)
    L.set_level(LogLevel::VERBOSE);
    assert(L.level() == LogLevel::VERBOSE);

    L.set_level(LogLevel::DEBUG);
    assert(L.level() == LogLevel::DEBUG);

    L.set_level(LogLevel::QUIET);
    assert(L.level() == LogLevel::QUIET);

    L.set_level(LogLevel::NORMAL);
    assert(L.level() == LogLevel::NORMAL);

    printf("  [PASS] test_level_filtering\n");
}

// ── Test 2: JSONL output format ────────────────────────────────────────────

static void test_jsonl_format() {
    reset_logger();
    auto& L = SinglifyLogger::instance();
    L.set_level(LogLevel::DEBUG);

    const char* path = "/tmp/test_singlify_log.jsonl";
    L.set_log_file(path);

    // Write several event types from the required 11
    SLOG_EVENT("pipeline_start", "Starting pipeline for SRR12345678");
    SLOG_EVENT("param", "threads=20 genome=GRCh38");
    SLOG_EVENT("star_start", "STAR alignment starting, 40.4M reads");
    SLOG_EVENT("star_complete", "STAR complete: 82.91%% unique mapping");
    SLOG_EVENT("pileup_start", "Streaming pileup starting");
    SLOG_EVENT("pileup_complete", "Pileup complete: 12089 cells, 78298 genes");
    L.stat("wall_seconds", 140.47);
    L.stat("protocol", "10x-3p-v3");
    SLOG_EVENT("summary", "Pipeline complete: 12089 cells, 78298 genes, 140.47s wall");
    SLOG_EVENT("error", "test error event");
    SLOG_EVENT("warning", "test warning event");

    L.close();

    std::string content = read_file(path);

    // Every line must contain the required JSON fields
    assert(contains(content, "\"type\":\"pipeline_start\""));
    assert(contains(content, "\"type\":\"param\""));
    assert(contains(content, "\"type\":\"star_start\""));
    assert(contains(content, "\"type\":\"star_complete\""));
    assert(contains(content, "\"type\":\"pileup_start\""));
    assert(contains(content, "\"type\":\"pileup_complete\""));
    assert(contains(content, "\"type\":\"stat\""));
    assert(contains(content, "\"type\":\"summary\""));
    assert(contains(content, "\"type\":\"error\""));
    assert(contains(content, "\"type\":\"warning\""));
    assert(contains(content, "\"ts\":\"20"));  // timestamp present

    // stat key=value in msg
    assert(contains(content, "wall_seconds="));
    assert(contains(content, "protocol=10x-3p-v3"));

    // line count: 11 events → 11 newlines
    assert(count_lines(content) == 11);

    printf("  [PASS] test_jsonl_format\n");
    remove(path);
}

// ── Test 3: JSON escaping ──────────────────────────────────────────────────

static void test_json_escaping() {
    reset_logger();
    auto& L = SinglifyLogger::instance();
    L.set_level(LogLevel::NORMAL);

    const char* path = "/tmp/test_singlify_escape.jsonl";
    L.set_log_file(path);

    SLOG_EVENT("info", "path=\"/tmp/my dir\\file\" note:\ttab");

    L.close();
    std::string content = read_file(path);

    // Check that special chars are escaped
    assert(contains(content, "\\\""));  // escaped quote
    assert(contains(content, "\\\\"));  // escaped backslash
    assert(contains(content, "\\t"));   // escaped tab

    printf("  [PASS] test_json_escaping\n");
    remove(path);
}

// ── Test 4: macros expand correctly ───────────────────────────────────────

static void test_macros() {
    reset_logger();
    auto& L = SinglifyLogger::instance();
    L.set_level(LogLevel::DEBUG);

    const char* path = "/tmp/test_singlify_macros.jsonl";
    L.set_log_file(path);

    SLOG("info message %d", 1);
    SLOG_VERBOSE("verbose message");
    SLOG_DEBUG("debug message");
    SLOG_ERROR("error message");
    SLOG_WARN("warn message");
    SLOG_EVENT("pipeline_start", "start %s", "test");

    L.close();
    std::string content = read_file(path);

    assert(contains(content, "\"type\":\"info\""));
    assert(contains(content, "\"type\":\"verbose\""));
    assert(contains(content, "\"type\":\"debug\""));
    assert(contains(content, "\"type\":\"error\""));
    assert(contains(content, "\"type\":\"warning\""));
    assert(contains(content, "\"type\":\"pipeline_start\""));
    assert(count_lines(content) == 6);

    printf("  [PASS] test_macros\n");
    remove(path);
}

// ── Test 5: level filtering suppresses JSONL ──────────────────────────────

static void test_level_suppresses_jsonl() {
    reset_logger();
    auto& L = SinglifyLogger::instance();
    L.set_level(LogLevel::NORMAL);  // verbose/debug should not write

    const char* path = "/tmp/test_singlify_suppress.jsonl";
    L.set_log_file(path);

    SLOG("normal message");   // written
    SLOG_VERBOSE("verbose");  // suppressed
    SLOG_DEBUG("debug");      // suppressed
    SLOG_ERROR("error msg");  // written (always)

    L.close();
    std::string content = read_file(path);

    // Only info + error should be in file → 2 lines
    assert(!contains(content, "\"type\":\"verbose\""));
    assert(!contains(content, "\"type\":\"debug\""));
    assert(contains(content, "\"type\":\"info\""));
    assert(contains(content, "\"type\":\"error\""));
    assert(count_lines(content) == 2);

    printf("  [PASS] test_level_suppresses_jsonl\n");
    remove(path);
}

// ── Test 6: QUIET level suppresses everything except file ─────────────────

static void test_quiet_level() {
    reset_logger();
    auto& L = SinglifyLogger::instance();
    L.set_level(LogLevel::QUIET);

    const char* path = "/tmp/test_singlify_quiet.jsonl";
    L.set_log_file(path);

    SLOG("normal message");     // suppressed (QUIET)
    SLOG_ERROR("error");        // suppressed (QUIET)
    SLOG_EVENT("stat", "x=1");  // event: always written to file

    L.close();
    std::string content = read_file(path);

    // info and error are suppressed at QUIET, but event() always writes to file
    assert(!contains(content, "\"type\":\"info\""));
    assert(!contains(content, "\"type\":\"error\""));
    assert(contains(content, "\"type\":\"stat\""));

    printf("  [PASS] test_quiet_level\n");
    remove(path);
}

// ── Test 7: progress throttling ───────────────────────────────────────────

static void test_progress_throttle() {
    reset_logger();
    auto& L = SinglifyLogger::instance();
    L.set_level(LogLevel::NORMAL);

    const char* path = "/tmp/test_singlify_progress.jsonl";
    L.set_log_file(path);

    // Fire 100 rapid progress events — only the first should go through
    // (throttle defaults to 2s)
    for (int i = 0; i < 100; ++i) {
        L.progress(float(i) / 100.f, "processing read %d", i * 1000);
    }
    // fraction=1.0 always passes (end-of-progress signal)
    L.progress(1.0f, "done");

    L.close();
    std::string content = read_file(path);

    // Should have ≤2 progress lines: one from first call + one from fraction=1.0
    int lines = count_lines(content);
    assert(lines <= 2);
    assert(lines >= 1);

    printf("  [PASS] test_progress_throttle (lines=%d)\n", lines);
    remove(path);
}

// ── Test 8: line count budget for typical run ─────────────────────────────

static void test_line_budget() {
    reset_logger();
    auto& L = SinglifyLogger::instance();
    L.set_level(LogLevel::VERBOSE);

    const char* path = "/tmp/test_singlify_budget.jsonl";
    L.set_log_file(path);

    // Simulate what a typical pipeline would log
    SLOG_EVENT("pipeline_start", "Starting pipeline for SRR12345678");
    SLOG_EVENT("param", "threads=20 genome=GRCh38 protocol=10x-3p-v3");
    SLOG_EVENT("star_start", "STAR alignment starting, 40.4M reads");
    for (int i = 0; i < 5; ++i)
        L.verbose("STAR progress update %d", i);
    SLOG_EVENT("star_complete", "STAR complete: 82.91%% unique mapping");
    SLOG_EVENT("pileup_start", "Streaming pileup starting");
    for (int i = 0; i < 10; ++i)
        L.verbose("Pileup worker %d complete", i);
    SLOG_EVENT("pileup_complete", "Pileup complete: 12089 cells, 78298 genes");
    for (int i = 0; i < 5; ++i)
        L.stat("metric", double(i));
    SLOG_EVENT("summary", "Pipeline complete: 12089 cells, 78298 genes, 140.47s wall");

    L.close();
    std::string content = read_file(path);
    int lines = count_lines(content);

    // 1+1+1+5+1+1+10+1+5+1 = 27 lines — well under 200
    assert(lines <= 200);
    printf("  [PASS] test_line_budget (lines=%d)\n", lines);
    remove(path);
}

// ── Test 9: thread safety ─────────────────────────────────────────────────

static void test_thread_safety() {
    reset_logger();
    auto& L = SinglifyLogger::instance();
    L.set_level(LogLevel::DEBUG);

    const char* path = "/tmp/test_singlify_threads.jsonl";
    L.set_log_file(path);

    const int N_THREADS = 8;
    const int N_PER_THREAD = 20;
    std::vector<std::thread> threads;
    threads.reserve(N_THREADS);

    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&L, t]() {
            for (int i = 0; i < N_PER_THREAD; ++i) {
                L.info("thread %d event %d", t, i);
            }
        });
    }
    for (auto& th : threads) th.join();

    L.close();
    std::string content = read_file(path);
    int lines = count_lines(content);

    // Each write is atomic per-line; total lines = N_THREADS * N_PER_THREAD
    assert(lines == N_THREADS * N_PER_THREAD);

    // No partial lines (every line starts with {"ts":)
    size_t pos = 0;
    while (pos < content.size()) {
        assert(content.substr(pos, 6) == "{\"ts\":");
        size_t nl = content.find('\n', pos);
        if (nl == std::string::npos) break;
        pos = nl + 1;
    }

    printf("  [PASS] test_thread_safety (lines=%d)\n", lines);
    remove(path);
}

// ── main ───────────────────────────────────────────────────────────────────

int main() {
    printf("=== test_logging ===\n");
    test_level_filtering();
    test_jsonl_format();
    test_json_escaping();
    test_macros();
    test_level_suppresses_jsonl();
    test_quiet_level();
    test_progress_throttle();
    test_line_budget();
    test_thread_safety();
    printf("=== ALL PASS ===\n");
    return 0;
}
