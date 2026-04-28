// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/include/singlet_gpu/bench/harness.h
//
// Header-only benchmark harness for singlet-gpu performance drivers.
//
// Provides:
//   BenchTimer         — cuEvent-based wall-time measurement for GPU kernels.
//   PeakMemoryTracker  — cudaMemGetInfo delta for device memory peak usage.
//   BenchRow           — schema-aligned result struct (benchmark-registry.md).
//   log_row()          — writes a BenchRow to stdout + appends to benchmark-registry.md.
//   throughput()       — helper: cells / (wall_ms / 1000.0).
//
// WHY cuEvents over std::chrono: cuEvents measure GPU-side execution time
// inclusive of kernel launches queued on the stream, correctly accounting for
// async execution and PCIe overlap. std::chrono measures host wall clock,
// which will be shorter than actual GPU work when kernels queue asynchronously.
//
// Usage pattern:
//   BenchTimer timer;
//   PeakMemoryTracker mem;
//   for (int i = 0; i < WARMUP + TIMED; ++i) {
//       mem.snapshot_before();   // only active for timed iters
//       timer.start();
//       run_kernel(...);
//       timer.stop();
//       if (i >= WARMUP) timer.record();
//       mem.snapshot_after();
//   }
//   float min_ms = timer.min_ms();
//   BenchRow row = { ... };

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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

// Canonical test sample path — populated from singlify pipeline output.
static constexpr const char* CANONICAL_SAMPLE_DIR =
    "/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna"
    "/GSE127/GSE127918/GSM4037629";

// Benchmark registry path — each run appends a markdown table row here.
static constexpr const char* BENCH_REGISTRY_PATH =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/benchmark-registry.md";

// ---------------------------------------------------------------------------
// GPU availability check (compile-time: CUDA header present; runtime: device 0)
// ---------------------------------------------------------------------------
inline bool gpu_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

// ---------------------------------------------------------------------------
// BenchTimer — cuEvent-based wall-time accumulator.
//
// All cuEvent calls are on a caller-supplied stream (default: 0).
// Timing is: cudaEventElapsedTime between start/stop events, which the CUDA
// runtime guarantees is GPU-side inclusive wall time on that stream.
// ---------------------------------------------------------------------------
class BenchTimer {
public:
    explicit BenchTimer(cudaStream_t stream = nullptr) : stream_(stream) {
        cudaEventCreate(&ev_start_);
        cudaEventCreate(&ev_stop_);
    }

    ~BenchTimer() {
        cudaEventDestroy(ev_start_);
        cudaEventDestroy(ev_stop_);
    }

    // Call immediately before the work on the stream.
    void start() {
        cudaEventRecord(ev_start_, stream_);
    }

    // Call immediately after the last kernel on the stream.
    void stop() {
        cudaEventRecord(ev_stop_, stream_);
    }

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

// ---------------------------------------------------------------------------
// PeakMemoryTracker — cudaMemGetInfo delta for device peak usage.
//
// WHY cudaMemGetInfo: it queries the driver for free/total memory on device 0
// at the moment of the call, giving a reliable host-observable snapshot of
// device memory consumption. It is accurate to within ~2 MB (page granularity).
// The delta (free_before - free_after) is the net allocation by the kernel.
// ---------------------------------------------------------------------------
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
        // free_before - free_after = bytes newly allocated (positive = used more).
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

// ---------------------------------------------------------------------------
// BenchRow — schema matching state/benchmark-registry.md
//
// Schema columns (in order):
//   date | feature | scale | impl | wall_ms | mem_mb | cells_per_sec |
//   pcie_gb | nsys_link | sm_occ | commit
//
// For a singlet-gpu row: impl="singlet-gpu", commit=short SHA or "n/a".
// For a SOTA baseline row: impl=<library name>, commit="baseline".
// ---------------------------------------------------------------------------
struct BenchRow {
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

// ---------------------------------------------------------------------------
// throughput() — cells / second from median wall time.
// ---------------------------------------------------------------------------
inline float throughput(int64_t n_cells, float wall_ms) {
    if (wall_ms <= 0.0f) return 0.0f;
    return static_cast<float>(n_cells) / (wall_ms / 1000.0f);
}

// ---------------------------------------------------------------------------
// current_date_str() — YYYY-MM-DD for the BenchRow date field.
// ---------------------------------------------------------------------------
inline std::string current_date_str() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[16]{};
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", std::gmtime(&t));
    return buf;
}

// ---------------------------------------------------------------------------
// log_row() — prints to stdout AND appends a markdown table row to
// state/benchmark-registry.md.
//
// The registry file is append-only. We open in append mode so concurrent
// benchmark processes do not race (each write is a single fwrite; POSIX
// guarantees atomic appends ≤PIPE_BUF bytes to regular files, which a single
// row is easily within).
// ---------------------------------------------------------------------------
inline void log_row(BenchRow& row) {
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
    // Format: | date | feature | scale | impl | wall_ms | mem_mb | cells_per_sec |
    //         pcie_gb | nsys_link | sm_occ | commit |
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

// ---------------------------------------------------------------------------
// Synthetic fallback matrix — used when no real .1pz sample is found.
// Generates a tiny 200×500 (genes × cells) CSR-order random count matrix
// with ~5% density, returned as host vectors for bench drivers to use.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Benchmark skip helpers — print message and exit(0) cleanly.
// ---------------------------------------------------------------------------
inline void skip_no_gpu() {
    std::puts("NO GPU — skipping benchmark (no CUDA device found).");
    std::fflush(stdout);
}

inline void skip_no_sample(const std::string& path) {
    std::printf("NO SAMPLE at %s — using synthetic fallback.\n", path.c_str());
    std::fflush(stdout);
}

// ===========================================================================
// singlet_gpu::bench — namespaced harness API (used by cycle-53b+ drivers).
//
// Provides a thin, consistent surface over the file-scope helpers above, plus
// subprocess reference runner and git SHA helper.
//
// WHY a separate namespace: allows bench drivers to write
//   namespace bench = singlet_gpu::bench;
//   bench::BenchTimer t; bench::skip("driver", "reason");
// without polluting the file scope or creating name collisions with cub/cuda.
// ===========================================================================

// BENCH_REFS_DIR: compile-time default for bench/refs/ directory.
// Drivers expect Python reference scripts here.
#ifndef BENCH_REFS_DIR
#  define BENCH_REFS_DIR "/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/refs"
#endif

namespace singlet_gpu {
namespace bench {

// ---------------------------------------------------------------------------
// BenchTimer — cuEvent-based per-iteration wall time.
//
// Semantics: start()/stop() bracket GPU work; elapsed_ms() synchronizes and
// returns the interval in milliseconds (float).  Designed for one iteration
// at a time so each iter's time is stored by the caller.
// ---------------------------------------------------------------------------
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
    // Synchronize stop event, compute elapsed, reset for next call.
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

// ---------------------------------------------------------------------------
// PeakMemTracker — cudaMemGetInfo delta for device peak usage.
//
// sample_before() / sample_after() bracket a kernel invocation.
// peak_delta_mb() returns the maximum (free_before - free_after) in MB across
// all sample pairs since the last reset.
// ---------------------------------------------------------------------------
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
    double peak_mb_    = 0.0;
    size_t free_before_ = 0;
};

// ---------------------------------------------------------------------------
// RefResult — timing/memory result from a Python/R reference subprocess.
// wall_ms < 0  → reference script absent or failed (skip comparison).
// mem_mb  < 0  → not reported by reference.
// ---------------------------------------------------------------------------
struct RefResult {
    double wall_ms = -1.0;
    double mem_mb  = -1.0;
    std::string impl;   // library name from the JSON "impl" field
};

// ---------------------------------------------------------------------------
// BenchRow — aligned with benchmark-registry.md schema (cycle-53b format).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// today_iso() — YYYY-MM-DD UTC date string.
// ---------------------------------------------------------------------------
inline std::string today_iso() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[16]{};
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", std::gmtime(&t));
    return buf;
}

// ---------------------------------------------------------------------------
// git_short_sha() — 7-char git HEAD SHA or "no-git" when unavailable.
// WHY: the registry row needs a commit pointer; if git is absent we record
// "no-git" so the row is still valid and sortable.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// throughput() — cells per second from median wall time (ms).
// ---------------------------------------------------------------------------
inline double throughput(int64_t n_cells, double wall_ms) {
    if (wall_ms <= 0.0) return 0.0;
    return static_cast<double>(n_cells) / (wall_ms / 1000.0);
}

// ---------------------------------------------------------------------------
// skip() — print a structured skip message and return.
// The caller is responsible for actually returning from main after calling this.
// ---------------------------------------------------------------------------
inline void skip(const std::string& driver, const std::string& reason) {
    std::printf("SKIP %s: %s\n", driver.c_str(), reason.c_str());
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// run_python_reference() — run a shell command and parse the timing JSON.
//
// Expects the script to write a JSON file at timing_json_path with fields:
//   { "wall_ms": float, "mem_mb": float, "impl": string }
//
// Returns RefResult with wall_ms=-1 / mem_mb=-1 on any failure.
// WHY: centralize JSON parsing so each bench driver doesn't repeat it.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// log_row() — stdout + append to benchmark-registry.md.
// ---------------------------------------------------------------------------
inline void log_row(BenchRow& row) {
    // Compute ratios (sota_wall is in ms here for the namespace API).
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

}  // namespace bench
}  // namespace singlet_gpu
