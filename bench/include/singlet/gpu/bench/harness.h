// SPDX-License-Identifier: MIT
// singlet/gpu/bench/include/singlet/gpu/bench/harness.h
//
// Header-only benchmark harness for singlet-gpu performance drivers.
//
// CANONICAL API lives in  namespace singlet::gpu::bench  (cycle-53b+ drivers):
//   BenchTimer          — single-shot cuEvent wall-time (driver accumulates samples).
//   PeakMemTracker      — cudaMemGetInfo delta, sample_before/after.
//   BenchRow            — scalar wall_ms/mem_mb registry row.
//   RefResult           — Python/R reference subprocess result.
//   throughput()        — cells / (wall_ms / 1000).
//   log_row()           — stdout + append to benchmark-registry.md.
//   today_iso()         — YYYY-MM-DD UTC.
//   git_short_sha()     — 7-char git HEAD SHA.
//   skip()              — structured skip message.
//   run_python_reference() — run a shell cmd, parse the timing JSON.
//   write_csc_bin()     — write a 0x43535343-magic CSC binary for Python refs.
//   WorkspacePlan + fits_in_device / fits_in_host_pinned / fits_in_host_pageable
//                       — OOM pre-flight guard: check memory BEFORE reserve/cudaMalloc
//                         so a too-large scale emits a clean skip instead of crashing.
//
// MULTI-SAMPLE API (legacy drivers that accumulate min/med/max inside the timer):
//   MultiSampleTimer    — cuEvent timer with internal sample vector + min/med/max.
//   PeakMemoryTracker   — snapshot_before/after peak tracker.
//   BenchRowMulti       — registry row carrying wall_ms_min/med/max + mem_mb_peak.
//   make_synthetic_matrix() / SyntheticMatrix — synthetic CSC fallback.
//   gpu_available(), skip_no_gpu(), skip_no_sample(), current_date_str().
//
// For backward compatibility the multi-sample types and the shared scalar
// helpers are also re-exported at file scope via using-declarations at the
// bottom of this header, so legacy drivers that reference the bare names keep
// compiling. Each entity is DEFINED exactly once (in the namespace).
//
// WHY cuEvents over std::chrono: cuEvents measure GPU-side execution time
// inclusive of kernel launches queued on the stream, correctly accounting for
// async execution and PCIe overlap. std::chrono measures host wall clock,
// which will be shorter than actual GPU work when kernels queue asynchronously.

#pragma once

#include <cuda_runtime.h>
#include <sys/sysinfo.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------
static constexpr int BENCH_WARMUP_ITERS = 3;
static constexpr int BENCH_TIMED_ITERS  = 5;

// Canonical test sample path — populated from singlet pipeline output.
static constexpr const char* CANONICAL_SAMPLE_DIR =
    "/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna"
    "/GSE127/GSE127918/GSM4037629";

// Benchmark registry path — each run appends a markdown table row here.
static constexpr const char* BENCH_REGISTRY_PATH =
    "/mnt/home/debruinz/Singlet-AI/singlet/gpu/state/benchmark-registry.md";

// BENCH_REFS_DIR: compile-time default for bench/refs/ directory.
// Drivers expect Python reference scripts here.
#ifndef BENCH_REFS_DIR
#  define BENCH_REFS_DIR "/mnt/home/debruinz/Singlet-AI/singlet/gpu/bench/refs"
#endif

namespace singlet::gpu {
namespace bench {

// ===========================================================================
// GPU availability check (compile-time: CUDA header present; runtime: device 0)
// ===========================================================================
inline bool gpu_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

// ===========================================================================
// BenchTimer — single-shot cuEvent per-iteration wall time (CANONICAL).
//
// Semantics: start()/stop() bracket GPU work; elapsed_ms() synchronizes and
// returns the interval in milliseconds.  Designed for one iteration at a time
// so each iter's time is stored by the caller (typically in a std::vector).
// ===========================================================================
class BenchTimer {
public:
    BenchTimer() {
        cudaEventCreate(&ev_start_);
        cudaEventCreate(&ev_stop_);
    }
    ~BenchTimer() {
        cudaEventDestroy(ev_start_);
        cudaEventDestroy(ev_stop_);
    }
    // Record start event on the given stream.
    void start(cudaStream_t stream = nullptr) {
        cudaEventRecord(ev_start_, stream);
        stream_ = stream;
    }
    // Record stop event on the same stream.
    void stop(cudaStream_t stream = nullptr) {
        cudaEventRecord(ev_stop_, stream ? stream : stream_);
    }
    // Synchronize stop event, compute elapsed, return ms.
    double elapsed_ms() {
        cudaEventSynchronize(ev_stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start_, ev_stop_);
        return static_cast<double>(ms);
    }

private:
    cudaEvent_t  ev_start_{};
    cudaEvent_t  ev_stop_{};
    cudaStream_t stream_  = nullptr;
};

// ===========================================================================
// MultiSampleTimer — cuEvent timer with an internal sample vector.
//
// Legacy idiom: the timer itself accumulates per-iteration samples (via
// record()) and exposes min/median/max.  All cuEvent calls are on a
// caller-supplied stream (default: 0).
//
// Usage:
//   MultiSampleTimer timer(stream);
//   for (...) { timer.start(); work(); timer.stop(); if (timed) timer.record(); }
//   float med = timer.median_ms();
// ===========================================================================
class MultiSampleTimer {
public:
    explicit MultiSampleTimer(cudaStream_t stream = nullptr) : stream_(stream) {
        cudaEventCreate(&ev_start_);
        cudaEventCreate(&ev_stop_);
    }

    ~MultiSampleTimer() {
        cudaEventDestroy(ev_start_);
        cudaEventDestroy(ev_stop_);
    }

    // Call immediately before the work on the stream.
    void start() { cudaEventRecord(ev_start_, stream_); }

    // Call immediately after the last kernel on the stream.
    void stop() { cudaEventRecord(ev_stop_, stream_); }

    // Synchronize, read elapsed, store in samples_.
    // Call once per timed iteration (NOT during warmup).
    void record() {
        cudaEventSynchronize(ev_stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start_, ev_stop_);
        samples_.push_back(ms);
    }

    float min_ms() const {
        if (samples_.empty()) return 0.0f;
        return *std::min_element(samples_.begin(), samples_.end());
    }

    float median_ms() const {
        if (samples_.empty()) return 0.0f;
        std::vector<float> s = samples_;
        std::sort(s.begin(), s.end());
        const int n = static_cast<int>(s.size());
        if (n % 2 == 1) return s[n / 2];
        return 0.5f * (s[n / 2 - 1] + s[n / 2]);
    }

    float max_ms() const {
        if (samples_.empty()) return 0.0f;
        return *std::max_element(samples_.begin(), samples_.end());
    }

    void reset() { samples_.clear(); }

private:
    cudaStream_t stream_;
    cudaEvent_t  ev_start_{};
    cudaEvent_t  ev_stop_{};
    std::vector<float> samples_;
};

// ===========================================================================
// PeakMemTracker — cudaMemGetInfo delta for device peak usage (CANONICAL).
//
// sample_before() / sample_after() bracket a kernel invocation.
// peak_delta_mb() returns the maximum (free_before - free_after) in MB across
// all sample pairs since the last reset.
// ===========================================================================
class PeakMemTracker {
public:
    void sample_before() {
        size_t free_b = 0, total = 0;
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free_b, &total);
        free_before_ = free_b;
    }
    void sample_after() {
        size_t free_a = 0, total = 0;
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free_a, &total);
        double delta = (static_cast<double>(free_before_) -
                        static_cast<double>(free_a)) / (1024.0 * 1024.0);
        if (delta > peak_mb_) peak_mb_ = delta;
    }
    double peak_delta_mb() const { return peak_mb_; }
    void   reset()               { peak_mb_ = 0.0; free_before_ = 0; }

private:
    double peak_mb_     = 0.0;
    size_t free_before_ = 0;
};

// ===========================================================================
// PeakMemoryTracker — snapshot_before/after peak tracker (legacy idiom).
//
// Identical measurement to PeakMemTracker; kept under the legacy method names
// (snapshot_before / snapshot_after / peak_mb) used by older drivers.
// ===========================================================================
class PeakMemoryTracker {
public:
    void snapshot_before() {
        size_t free_b = 0, total = 0;
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free_b, &total);
        free_before_bytes_ = free_b;
    }

    void snapshot_after() {
        size_t free_a = 0, total = 0;
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free_a, &total);
        int64_t delta = static_cast<int64_t>(free_before_bytes_) -
                        static_cast<int64_t>(free_a);
        float mb = static_cast<float>(delta) / (1024.0f * 1024.0f);
        if (mb > peak_mb_) peak_mb_ = mb;
    }

    float peak_mb() const { return peak_mb_; }
    void  reset()         { peak_mb_ = 0.0f; free_before_bytes_ = 0; }

private:
    float  peak_mb_           = 0.0f;
    size_t free_before_bytes_ = 0;
};

// ===========================================================================
// RefResult — timing/memory result from a Python/R reference subprocess.
// wall_ms < 0  → reference script absent or failed (skip comparison).
// mem_mb  < 0  → not reported by reference.
// ===========================================================================
struct RefResult {
    double wall_ms = -1.0;
    double mem_mb  = -1.0;
    std::string impl;   // library name from the JSON "impl" field
};

// ===========================================================================
// BenchRow — scalar registry row, aligned with benchmark-registry.md
// schema (cycle-53b format).  CANONICAL.
// ===========================================================================
struct BenchRow {
    std::string date;
    std::string feature;
    std::string scale;
    std::string impl;
    double wall_ms       = 0.0;
    double mem_mb        = 0.0;
    double cells_per_sec = 0.0;
    double sota_wall     = -1.0;   // ms (< 0 = not measured)
    double sota_mem      = -1.0;   // MB
    double ratio_wall    = -1.0;   // sota_wall / wall_ms (>1 means we're faster)
    double ratio_mem     = -1.0;
    std::string commit   = "n/a";
};

// ===========================================================================
// BenchRowMulti — registry row carrying min/median/max wall time + peak mem
// (legacy idiom, paired with MultiSampleTimer / PeakMemoryTracker).
//
// Schema columns (in order):
//   date | feature | scale | impl | wall_ms | mem_mb | cells_per_sec |
//   pcie_gb | nsys_link | sm_occ | commit
// ===========================================================================
struct BenchRowMulti {
    std::string date;           // YYYY-MM-DD
    std::string feature;        // e.g. "io/pz_loader", "preprocess/lognorm"
    std::string scale;          // "tiny" | "10k" | "100k" | "1m"
    std::string impl;           // "singlet-gpu" or SOTA name

    // GPU timing (singlet-gpu rows) — all three required.
    float wall_ms_min    = 0.0f;
    float wall_ms_med    = 0.0f;
    float wall_ms_max    = 0.0f;
    float mem_mb_peak    = 0.0f;
    float cells_per_sec  = 0.0f;

    // SOTA reference (filled in by the reference subprocess runner).
    float sota_wall_sec  = -1.0f;   // -1 = not measured
    float sota_mem_mb    = -1.0f;

    // Derived ratios (set by log_row).
    float ratio_wall     = -1.0f;   // sota_wall_sec * 1000 / wall_ms_med
    float ratio_mem      = -1.0f;   // sota_mem_mb / mem_mb_peak

    // Provenance.
    std::string pcie_gb  = "—";
    std::string nsys_link= "—";
    std::string sm_occ   = "—";
    std::string commit   = "n/a";
};

// ===========================================================================
// throughput() — cells per second from wall time (ms).
// ===========================================================================
inline double throughput(int64_t n_cells, double wall_ms) {
    if (wall_ms <= 0.0) return 0.0;
    return static_cast<double>(n_cells) / (wall_ms / 1000.0);
}

// ===========================================================================
// Date helpers — YYYY-MM-DD UTC date string.
// today_iso() is the canonical name; current_date_str() is the legacy alias.
// ===========================================================================
inline std::string today_iso() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[16]{};
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", std::gmtime(&t));
    return buf;
}

inline std::string current_date_str() { return today_iso(); }

// ===========================================================================
// git_short_sha() — 7-char git HEAD SHA or "no-git" when unavailable.
// WHY: the registry row needs a commit pointer; if git is absent we record
// "no-git" so the row is still valid and sortable.
// ===========================================================================
inline std::string git_short_sha() {
    FILE* fp = popen("git -C /mnt/home/debruinz/Singlet-AI rev-parse --short HEAD 2>/dev/null",
                     "r");
    if (!fp) return "no-git";
    char buf[16]{};
    if (!std::fgets(buf, sizeof(buf), fp)) { pclose(fp); return "no-git"; }
    pclose(fp);
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
        s.pop_back();
    return s.empty() ? "no-git" : s;
}

// ===========================================================================
// skip() — print a structured skip message and return.
// The caller is responsible for actually returning from main after calling this.
// ===========================================================================
inline void skip(const std::string& driver, const std::string& reason) {
    std::printf("SKIP %s: %s\n", driver.c_str(), reason.c_str());
    std::fflush(stdout);
}

// ===========================================================================
// Benchmark skip helpers — print message (caller exits cleanly).
// ===========================================================================
inline void skip_no_gpu() {
    std::puts("NO GPU — skipping benchmark (no CUDA device found).");
    std::fflush(stdout);
}

inline void skip_no_sample(const std::string& path) {
    std::printf("NO SAMPLE at %s — using synthetic fallback.\n", path.c_str());
    std::fflush(stdout);
}

// ===========================================================================
// run_python_reference() — run a shell command and parse the timing JSON.
//
// Expects the script to write a JSON file at timing_json_path with fields:
//   { "wall_ms": float, "mem_mb": float, "impl": string }
//
// Returns RefResult with wall_ms=-1 / mem_mb=-1 on any failure.
// WHY: centralize JSON parsing so each bench driver doesn't repeat it.
// ===========================================================================
inline RefResult run_python_reference(const std::string& cmd,
                                      const std::string& timing_json_path) {
    RefResult r;
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "[harness] ref cmd failed (rc=%d): %s\n", rc, cmd.c_str());
        return r;
    }
    std::ifstream f(timing_json_path);
    if (!f.is_open()) {
        std::fprintf(stderr, "[harness] timing JSON not found: %s\n",
                     timing_json_path.c_str());
        return r;
    }

    // Minimal JSON parser — extract "wall_ms", "mem_mb", "impl" from a flat object.
    // WHY no external JSON lib: keep bench dependencies minimal.
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    auto extract_dbl = [&](const std::string& key) -> double {
        auto pos = content.find('"' + key + '"');
        if (pos == std::string::npos) return -1.0;
        auto colon = content.find(':', pos);
        if (colon == std::string::npos) return -1.0;
        try { return std::stod(content.substr(colon + 1)); } catch (...) { return -1.0; }
    };
    auto extract_str = [&](const std::string& key) -> std::string {
        auto pos = content.find('"' + key + '"');
        if (pos == std::string::npos) return "";
        auto colon = content.find(':', pos);
        if (colon == std::string::npos) return "";
        auto q1 = content.find('"', colon + 1);
        if (q1 == std::string::npos) return "";
        auto q2 = content.find('"', q1 + 1);
        if (q2 == std::string::npos) return "";
        return content.substr(q1 + 1, q2 - q1 - 1);
    };

    r.wall_ms = extract_dbl("wall_ms");
    r.mem_mb  = extract_dbl("mem_mb");
    r.impl    = extract_str("impl");
    return r;
}

// ===========================================================================
// write_csc_bin() — write a CSC matrix as a flat binary for Python ref scripts.
//
// Format (little-endian):
//   magic  uint32 = 0x43535343 ("CSSC")
//   n_rows uint32
//   n_cols uint32
//   nnz    uint64
//   values   float[nnz]
//   indptr   int[n_cols + 1]
//   rowids   int[nnz]
//
// WHY in the harness: this exact writer was copy-pasted verbatim across ~6 DE /
// perturbation / spatial drivers — consolidate so the format lives in one place.
// ===========================================================================
inline void write_csc_bin(const std::string& path,
                          const float* h_vals, const int* h_indptr,
                          const int* h_rowids,
                          int n_rows, int n_cols, int64_t nnz) {
    std::ofstream f(path, std::ios::binary);
    const uint32_t magic = 0x43535343u;
    const uint32_t nr    = static_cast<uint32_t>(n_rows);
    const uint32_t nc    = static_cast<uint32_t>(n_cols);
    const uint64_t nz    = static_cast<uint64_t>(nnz);
    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&nr),    4);
    f.write(reinterpret_cast<const char*>(&nc),    4);
    f.write(reinterpret_cast<const char*>(&nz),    8);
    f.write(reinterpret_cast<const char*>(h_vals),   nnz * sizeof(float));
    f.write(reinterpret_cast<const char*>(h_indptr), (n_cols + 1) * sizeof(int));
    f.write(reinterpret_cast<const char*>(h_rowids), nnz * sizeof(int));
}

// ===========================================================================
// log_row() — stdout + append to benchmark-registry.md.
//
// Two overloads: the scalar BenchRow (canonical) and the legacy BenchRowMulti
// which encodes min/med/max into the wall_ms cell.
//
// The registry file is append-only. We open in append mode so concurrent
// benchmark processes do not race (each write is a single fwrite; POSIX
// guarantees atomic appends ≤PIPE_BUF bytes to regular files).
// ===========================================================================
inline void log_row(BenchRow& row) {
    // Compute ratios (sota_wall is in ms here for the scalar API).
    if (row.sota_wall > 0.0 && row.wall_ms > 0.0)
        row.ratio_wall = row.sota_wall / row.wall_ms;
    if (row.sota_mem > 0.0 && row.mem_mb > 0.0)
        row.ratio_mem  = row.sota_mem / row.mem_mb;

    std::printf(
        "BENCH %-35s %-12s %-18s wall_ms=%.1f mem_mb=%.1f cells/s=%.0f "
        "sota_wall=%.1fms sota_mem=%.1fMB ratio_wall=%.2fx\n",
        row.feature.c_str(), row.scale.c_str(), row.impl.c_str(),
        row.wall_ms, row.mem_mb, row.cells_per_sec,
        row.sota_wall, row.sota_mem, row.ratio_wall);
    std::fflush(stdout);

    std::ofstream ofs(BENCH_REGISTRY_PATH, std::ios::app);
    if (!ofs.is_open()) return;
    ofs << "| " << row.date
        << " | " << row.feature
        << " | " << row.scale
        << " | " << row.impl
        << " | " << row.wall_ms
        << " | " << row.mem_mb
        << " | " << row.cells_per_sec
        << " | — "
        << " | — "
        << " | — "
        << " | " << row.commit
        << " |\n";
}

inline void log_row(BenchRowMulti& row) {
    // Derive ratios if SOTA data is present.
    if (row.sota_wall_sec > 0.0f && row.wall_ms_med > 0.0f)
        row.ratio_wall = (row.sota_wall_sec * 1000.0f) / row.wall_ms_med;
    if (row.sota_mem_mb > 0.0f && row.mem_mb_peak > 0.0f)
        row.ratio_mem = row.sota_mem_mb / row.mem_mb_peak;

    // Print to stdout.
    std::printf(
        "BENCH %-30s %-7s %-18s wall_ms=[%.1f/%.1f/%.1f] "
        "mem=%.1fMB cells/s=%.0f sota_wall=%.2fs sota_mem=%.1fMB "
        "ratio_wall=%.2fx\n",
        row.feature.c_str(), row.scale.c_str(), row.impl.c_str(),
        row.wall_ms_min, row.wall_ms_med, row.wall_ms_max,
        row.mem_mb_peak, row.cells_per_sec,
        row.sota_wall_sec, row.sota_mem_mb, row.ratio_wall);
    std::fflush(stdout);

    // Append to benchmark-registry.md.
    // We encode min/med/max into the wall_ms cell as "med (min–max)".
    char wall_cell[64]{};
    std::snprintf(wall_cell, sizeof(wall_cell), "%.1f (%.1f–%.1f)",
                  row.wall_ms_med, row.wall_ms_min, row.wall_ms_max);

    char mem_cell[32]{};
    if (row.mem_mb_peak >= 0.0f)
        std::snprintf(mem_cell, sizeof(mem_cell), "%.1f", row.mem_mb_peak);
    else
        std::strcpy(mem_cell, "—");

    char cells_cell[32]{};
    std::snprintf(cells_cell, sizeof(cells_cell), "%.0f", row.cells_per_sec);

    std::ofstream ofs(BENCH_REGISTRY_PATH, std::ios::app);
    if (ofs.is_open()) {
        ofs << "| " << row.date
            << " | " << row.feature
            << " | " << row.scale
            << " | " << row.impl
            << " | " << wall_cell
            << " | " << mem_cell
            << " | " << cells_cell
            << " | " << row.pcie_gb
            << " | " << row.nsys_link
            << " | " << row.sm_occ
            << " | " << row.commit
            << " |\n";
    }
}

// ===========================================================================
// Synthetic fallback matrix — used when no real .1pz sample is found.
// Generates a small genes×cells CSC-order random count matrix with ~5%
// density, returned as host vectors for bench drivers to use.
// ===========================================================================
struct SyntheticMatrix {
    int rows = 0;       // genes
    int cols = 0;       // cells
    int nnz  = 0;
    std::vector<int>   indptr;   // col-major indptr (CSC): size cols+1
    std::vector<int>   indices;  // row indices (size nnz)
    std::vector<float> values;   // float counts (size nnz)
};

inline SyntheticMatrix make_synthetic_matrix(int rows = 200,
                                             int cols = 500,
                                             float density = 0.05f,
                                             uint32_t seed = 42) {
    SyntheticMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.indptr.resize(cols + 1, 0);

    // Simple LCG for reproducibility without <random>.
    uint64_t rng = seed ^ 0xDEADBEEFCAFEBABEull;
    auto next_rng = [&]() -> uint64_t {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return rng;
    };

    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            if ((next_rng() & 0xFFFF) < static_cast<uint64_t>(density * 0x10000)) {
                m.indices.push_back(r);
                m.values.push_back(static_cast<float>(1 + (next_rng() % 15)));
                ++m.nnz;
            }
        }
        m.indptr[c + 1] = m.nnz;
    }
    return m;
}

// ===========================================================================
// Workspace plan + OOM-guard helpers.
//
// Check whether a planned allocation fits BEFORE vector::reserve / cudaMalloc,
// so a too-large scale emits a clean skip instead of crashing with bad_alloc
// or an out-of-memory CUDA error.
//
//   WorkspacePlan plan{ dev_bytes, pinned_bytes, pageable_bytes };
//   if (!fits_in_device(plan))        { skip("driver", "device OOM");   return 0; }
//   if (!fits_in_host_pinned(plan))   { skip("driver", "pinned OOM");   return 0; }
//   if (!fits_in_host_pageable(plan)) { skip("driver", "pageable OOM"); return 0; }
// ===========================================================================
struct WorkspacePlan {
    size_t device_bytes        = 0;
    size_t host_pinned_bytes   = 0;
    size_t host_pageable_bytes = 0;
};

// Device fit: planned device bytes must be under 85% of currently-free VRAM.
inline bool fits_in_device(const WorkspacePlan& p) {
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) return false;
    return static_cast<double>(p.device_bytes) <
           static_cast<double>(free_bytes) * 0.85;
}

// Pinned-host fit: planned pinned bytes must be under a budget cap (default
// 16 GiB; override via SINGLET_BENCH_PINNED_BUDGET_BYTES).  Pinned memory is a
// scarce resource, so we cap it explicitly rather than trusting freeram.
inline bool fits_in_host_pinned(const WorkspacePlan& p) {
    size_t budget = 16ULL << 30;  // 16 GiB default cap
    if (const char* env = std::getenv("SINGLET_BENCH_PINNED_BUDGET_BYTES")) {
        unsigned long long parsed = 0;
        const char* first = env;
        const char* last  = env + std::char_traits<char>::length(env);
        auto [ptr, ec] = std::from_chars(first, last, parsed);
        if (ec == std::errc{} && ptr == last && parsed > 0) {
            budget = static_cast<size_t>(parsed);
        }
    }
    return p.host_pinned_bytes < budget;
}

// Pageable-host fit: planned pageable bytes must be under 80% of free RAM.
inline bool fits_in_host_pageable(const WorkspacePlan& p) {
    struct sysinfo si{};
    if (sysinfo(&si) != 0) return false;
    const double freeram = static_cast<double>(si.freeram) *
                           static_cast<double>(si.mem_unit);
    return static_cast<double>(p.host_pageable_bytes) < freeram * 0.8;
}

}  // namespace bench
}  // namespace singlet::gpu

// ===========================================================================
// File-scope re-exports for legacy drivers.
//
// Older bench drivers reference the multi-sample types and shared helpers by
// their bare (unqualified) names.  These using-declarations make those names
// resolve to the single canonical definitions in singlet::gpu::bench, so each
// entity is defined exactly once while legacy drivers keep compiling.
//
// Newer drivers use the namespaced API directly (namespace bench =
// singlet::gpu::bench; or using namespace singlet::gpu;) and are unaffected.
// ===========================================================================
using BenchTimer        = singlet::gpu::bench::MultiSampleTimer;
using PeakMemoryTracker = singlet::gpu::bench::PeakMemoryTracker;
using BenchRow          = singlet::gpu::bench::BenchRowMulti;
using SyntheticMatrix   = singlet::gpu::bench::SyntheticMatrix;

using singlet::gpu::bench::gpu_available;
using singlet::gpu::bench::skip_no_gpu;
using singlet::gpu::bench::skip_no_sample;
using singlet::gpu::bench::current_date_str;
using singlet::gpu::bench::make_synthetic_matrix;
using singlet::gpu::bench::throughput;
using singlet::gpu::bench::log_row;
