// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_preprocess_lognorm_perf.cpp
//
// Benchmark: preprocess/lognorm  vs  scanpy.pp.normalize_total + log1p
//
// What is measured:
//   - log_normalize() time: column-sum pass + median + per-cell scale + log1p.
//   - Excludes load_pz time (matrix pre-loaded before timing loop).
//   - Peak device memory delta: size_factors + qc_mask buffers.
//   - Throughput in cells/s.
//
// SOTA reference: scanpy.pp.normalize_total + log1p via lognorm_ref.py.
//
// Scale: 10k — GSM4037629 (11,560 cells).
// Fallback: synthetic 200×500 matrix uploaded to device.

#include <singlet/gpu/bench/harness.h>
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/preprocess/lognorm.h>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
using namespace singlet::gpu;

static constexpr const char* REF_SCRIPT =
    "/mnt/home/debruinz/Singlet-AI/singlet/gpu/bench/refs/lognorm_ref.py";

int main() {
    if (!gpu_available()) { skip_no_gpu(); return 0; }

    const std::string pz_path = std::string(CANONICAL_SAMPLE_DIR) + "/counts.1pz";
    bool have_real = fs::exists(pz_path);

    // -----------------------------------------------------------------------
    // Load data ONCE outside the timing loop.
    // -----------------------------------------------------------------------
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    int64_t n_cells = 0;

    // We need a PzDeviceMatrix that outlives the loop.
    // Re-load each iter would measure IO; we want compute only.
    // Strategy: load once, run log_normalize in-place on the VALUES each iter.
    // (log_normalize operates on a DeviceCSC whose values it modifies; after
    // each iter the values are normalized — that's intentional for the bench.)

    if (have_real) {
        io::PzDeviceMatrix mat = io::load_pz(pz_path, stream);
        cudaStreamSynchronize(stream);
        n_cells = mat.mat.cols;

        preprocess::LogNormConfig cfg;
        cfg.target_count = 1e4f;
        // method defaults to LogNormMethod::TotalCount — no NormFlavor enum needed

        BenchTimer        timer(stream);
        PeakMemoryTracker mem;

        for (int iter = 0; iter < BENCH_WARMUP_ITERS + BENCH_TIMED_ITERS; ++iter) {
            bool timed = (iter >= BENCH_WARMUP_ITERS);
            if (timed) mem.snapshot_before();
            timer.start();
            auto result = preprocess::log_normalize(mat.mat, cfg, stream);
            cudaStreamSynchronize(stream);
            timer.stop();
            if (timed) { timer.record(); mem.snapshot_after(); }
        }

        float sota_wall = -1.0f, sota_mem = -1.0f;
        if (fs::exists(REF_SCRIPT)) {
            char cmd[1024]{};
            std::snprintf(cmd, sizeof(cmd),
                "python3 %s %s 2>/dev/null", REF_SCRIPT, CANONICAL_SAMPLE_DIR);
            FILE* fp = popen(cmd, "r");
            if (fp) {
                char line[256]{};
                while (fgets(line, sizeof(line), fp)) {
                    if (strncmp(line, "SOTA_WALL_SEC=", 14) == 0) sota_wall = std::atof(line + 14);
                    if (strncmp(line, "SOTA_MEM_MB=",   12) == 0) sota_mem  = std::atof(line + 12);
                }
                pclose(fp);
            }
        }

        BenchRow row;
        row.date          = current_date_str();
        row.feature       = "preprocess/lognorm";
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

    } else {
        // Synthetic fallback: upload to device, time the normalize call.
        skip_no_sample(pz_path);
        SyntheticMatrix syn = make_synthetic_matrix(200, 500);
        n_cells = syn.cols;

        // Build a minimal DeviceCSC from the synthetic data.
        // Use factornet::gpu::SparseMatrixGPU directly (= core::DeviceCSC).
        int   *d_indptr = nullptr, *d_indices = nullptr;
        float *d_values = nullptr;
        cudaMalloc(&d_indptr,  (syn.cols + 1) * sizeof(int));
        cudaMalloc(&d_indices, syn.nnz        * sizeof(int));
        cudaMalloc(&d_values,  syn.nnz        * sizeof(float));
        cudaMemcpy(d_indptr,  syn.indptr.data(),  (syn.cols + 1) * sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, syn.indices.data(), syn.nnz        * sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_values,  syn.values.data(),  syn.nnz        * sizeof(float), cudaMemcpyHostToDevice);

        // Wrap into io::PzDeviceMatrix (mat field only; meta is empty).
        io::PzDeviceMatrix mat;
        mat.mat.rows   = syn.rows;
        mat.mat.cols   = syn.cols;
        mat.mat.nnz    = syn.nnz;
        // Wrap raw device pointers as non-owning DeviceMemory (we cudaFree below).
        mat.mat.col_ptr     = factornet::gpu::DeviceMemory<int>::wrap(d_indptr,  syn.cols + 1);
        mat.mat.row_indices = factornet::gpu::DeviceMemory<int>::wrap(d_indices, syn.nnz);
        mat.mat.values      = factornet::gpu::DeviceMemory<float>::wrap(d_values, syn.nnz);
        mat.producer_stream = stream;

        preprocess::LogNormConfig cfg;
        BenchTimer        timer(stream);
        PeakMemoryTracker mem;

        for (int iter = 0; iter < BENCH_WARMUP_ITERS + BENCH_TIMED_ITERS; ++iter) {
            bool timed = (iter >= BENCH_WARMUP_ITERS);
            if (timed) mem.snapshot_before();
            timer.start();
            auto result = preprocess::log_normalize(mat.mat, cfg, stream);
            cudaStreamSynchronize(stream);
            timer.stop();
            if (timed) { timer.record(); mem.snapshot_after(); }
        }

        cudaFree(d_indptr); cudaFree(d_indices); cudaFree(d_values);

        BenchRow row;
        row.date          = current_date_str();
        row.feature       = "preprocess/lognorm";
        row.scale         = "tiny";
        row.impl          = "singlet-gpu";
        row.wall_ms_min   = timer.min_ms();
        row.wall_ms_med   = timer.median_ms();
        row.wall_ms_max   = timer.max_ms();
        row.mem_mb_peak   = mem.peak_mb();
        row.cells_per_sec = throughput(n_cells, row.wall_ms_med);
        log_row(row);
    }

    cudaStreamDestroy(stream);
    return 0;
}
