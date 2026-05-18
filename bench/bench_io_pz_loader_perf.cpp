// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_io_pz_loader_perf.cpp
//
// Benchmark: io/pz_device_loader  vs  scanpy.read_10x (SOTA reference)
//
// What is measured:
//   - load_pz() time: file-open + zstd decompress + pinned alloc + 3× async H2D.
//   - Peak device memory delta: indptr + indices + values on device.
//   - Throughput in cells/s.
//
// SOTA reference: scanpy.read_10x_mtx (reads Market Exchange format from 10x
// output dir). Timed via refs/io_pz_loader_ref.py subprocess.
//
// Scale: 10k cells — GSM4037629 (11,560 cells, ~30 M nnz).
// Fallback: if sample not present, uses the synthetic matrix from harness.h
//           and skips the SOTA subprocess (no .mtx to compare).

#include <singlet/gpu/bench/harness.h>
#include <singlet/gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
using namespace singlet::gpu;

// Path to the driver's own directory so we can find the ref script.
static constexpr const char* REF_SCRIPT =
    "/mnt/home/debruinz/Singlet-AI/singlet/gpu/bench/refs/io_pz_loader_ref.py";

int main() {
    if (!gpu_available()) {
        skip_no_gpu();
        return 0;
    }

    // -----------------------------------------------------------------------
    // 1. Locate the .1pz file.
    // -----------------------------------------------------------------------
    const std::string sample_dir = CANONICAL_SAMPLE_DIR;
    const std::string pz_path    = sample_dir + "/counts.1pz";
    bool have_real_sample = fs::exists(pz_path);

    if (!have_real_sample) {
        skip_no_sample(pz_path);
        // Synthetic path: allocate a small fake CSC on device directly.
        // We time cudaMalloc + cudaMemcpy as a proxy for the loader.
        SyntheticMatrix syn = make_synthetic_matrix(200, 500);
        int64_t n_cells = syn.cols;

        // Allocate device buffers.
        int   *d_indptr = nullptr, *d_indices = nullptr;
        float *d_values = nullptr;
        cudaMalloc(&d_indptr,  (syn.cols + 1) * sizeof(int));
        cudaMalloc(&d_indices, syn.nnz        * sizeof(int));
        cudaMalloc(&d_values,  syn.nnz        * sizeof(float));

        BenchTimer       timer;
        PeakMemoryTracker mem;

        for (int iter = 0; iter < BENCH_WARMUP_ITERS + BENCH_TIMED_ITERS; ++iter) {
            bool timed = (iter >= BENCH_WARMUP_ITERS);
            if (timed) mem.snapshot_before();
            timer.start();
            cudaMemcpy(d_indptr,  syn.indptr.data(),  (syn.cols + 1) * sizeof(int),   cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, syn.indices.data(), syn.nnz        * sizeof(int),   cudaMemcpyHostToDevice);
            cudaMemcpy(d_values,  syn.values.data(),  syn.nnz        * sizeof(float), cudaMemcpyHostToDevice);
            timer.stop();
            if (timed) { timer.record(); mem.snapshot_after(); }
        }

        cudaFree(d_indptr); cudaFree(d_indices); cudaFree(d_values);

        BenchRow row;
        row.date          = current_date_str();
        row.feature       = "io/pz_loader";
        row.scale         = "tiny";
        row.impl          = "singlet-gpu";
        row.wall_ms_min   = timer.min_ms();
        row.wall_ms_med   = timer.median_ms();
        row.wall_ms_max   = timer.max_ms();
        row.mem_mb_peak   = mem.peak_mb();
        row.cells_per_sec = throughput(n_cells, row.wall_ms_med);
        log_row(row);
        return 0;
    }

    // -----------------------------------------------------------------------
    // 2. Real sample: time load_pz.
    // -----------------------------------------------------------------------
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    int64_t n_cells = 0;

    BenchTimer        timer(stream);
    PeakMemoryTracker mem;

    for (int iter = 0; iter < BENCH_WARMUP_ITERS + BENCH_TIMED_ITERS; ++iter) {
        bool timed = (iter >= BENCH_WARMUP_ITERS);
        if (timed) mem.snapshot_before();
        timer.start();
        // load_pz synchronizes internally via cudaStreamSynchronize before
        // returning, so this is a complete synchronous load (wall time = GPU wall).
        io::PzDeviceMatrix mat = io::load_pz(pz_path, stream);
        timer.stop();
        if (timed) {
            timer.record();
            mem.snapshot_after();
            n_cells = mat.mat.cols;
        }
        // mat goes out of scope: RAII frees device buffers.
    }

    cudaStreamDestroy(stream);

    // -----------------------------------------------------------------------
    // 3. SOTA reference subprocess.
    // -----------------------------------------------------------------------
    float sota_wall = -1.0f, sota_mem = -1.0f;
    if (fs::exists(REF_SCRIPT)) {
        auto [w, m] = [&]() -> std::pair<float, float> {
            // Run via Python; parse SOTA_WALL_SEC= and SOTA_MEM_MB= from stdout.
            char cmd[1024]{};
            std::snprintf(cmd, sizeof(cmd),
                "python3 %s %s 2>/dev/null", REF_SCRIPT, sample_dir.c_str());
            FILE* fp = popen(cmd, "r");
            if (!fp) return {-1.0f, -1.0f};
            float w = -1.0f, m = -1.0f;
            char line[256]{};
            while (fgets(line, sizeof(line), fp)) {
                if (strncmp(line, "SOTA_WALL_SEC=", 14) == 0)
                    w = std::atof(line + 14);
                if (strncmp(line, "SOTA_MEM_MB=", 12) == 0)
                    m = std::atof(line + 12);
            }
            pclose(fp);
            return {w, m};
        }();
        sota_wall = w; sota_mem = m;
    }

    // -----------------------------------------------------------------------
    // 4. Log result.
    // -----------------------------------------------------------------------
    BenchRow row;
    row.date          = current_date_str();
    row.feature       = "io/pz_loader";
    row.scale         = "10k";
    row.impl          = "singlet-gpu";
    row.wall_ms_min   = timer.min_ms();
    row.wall_ms_med   = timer.median_ms();
    row.wall_ms_max   = timer.max_ms();
    row.mem_mb_peak   = mem.peak_mb();
    row.cells_per_sec = throughput(n_cells, row.wall_ms_med);
    row.sota_wall_sec = sota_wall;
    row.sota_mem_mb   = sota_mem;
    log_row(row);

    return 0;
}
