// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_reduce_nmf_perf.cpp
//
// Benchmark: reduce/nmf (factornet GPU NMF)  vs  sklearn.decomposition.NMF
//
// What is measured:
//   - nmf::fit() time for k=10 on the raw count matrix (NMF operates on
//     non-negative counts; do NOT log-normalize before NMF).
//   - Excludes load_pz time.
//   - Peak device memory delta.
//   - Throughput in cells/s.
//
// WHY k=10: standard NMF rank for exploratory scRNA analysis (matches cNMF).
//
// SOTA reference: sklearn NMF(n_components=10) via nmf_ref.py subprocess.
//   (Rapids cuml NMF is GPU but rare in practice; sklearn is the universal ref.)
//
// Scale: 10k — GSM4037629 (11,560 cells).
// Fallback: synthetic 200×500.
//
// NOTE: nmf::fit requires keep_host_pinned=true on the PzDeviceMatrix
// (factornet NMF reads from pinned host CSC — same constraint as SVD).

#include <singlet_gpu/bench/harness.h>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/reduce/nmf/fit.h>
#include <singlet-gpu/reduce/nmf/types.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
using namespace singlet_gpu;

static constexpr const char* REF_SCRIPT =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/refs/nmf_ref.py";
static constexpr int NMF_K = 10;

int main() {
    if (!gpu_available()) { skip_no_gpu(); return 0; }

    const std::string pz_path = std::string(CANONICAL_SAMPLE_DIR) + "/counts.1pz";
    bool have_real = fs::exists(pz_path);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // SOTA reference (sklearn NMF is slow — run once before GPU loop).
    float sota_wall = -1.0f, sota_mem = -1.0f;
    if (have_real && fs::exists(REF_SCRIPT)) {
        char cmd[1024]{};
        std::snprintf(cmd, sizeof(cmd),
            "python3 %s %s %d 2>/dev/null", REF_SCRIPT, CANONICAL_SAMPLE_DIR, NMF_K);
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

    if (have_real) {
        // keep_host_pinned=true: factornet NMF needs pinned host CSC.
        io::PzDeviceMatrix mat = io::load_pz(pz_path, stream, /*keep_host_pinned=*/true);
        cudaStreamSynchronize(stream);
        int64_t n_cells = mat.mat.cols;

        reduce::nmf::NmfConfig cfg;
        cfg.rank     = NMF_K;
        cfg.max_iter = 100;    // standard convergence budget for timing

        BenchTimer        timer(stream);
        PeakMemoryTracker mem;

        for (int iter = 0; iter < BENCH_WARMUP_ITERS + BENCH_TIMED_ITERS; ++iter) {
            bool timed = (iter >= BENCH_WARMUP_ITERS);
            if (timed) mem.snapshot_before();
            timer.start();
            auto result = reduce::nmf::fit(mat, cfg);
            // NMF is CPU-side factornet — synchronize device for any GPU work done.
            cudaDeviceSynchronize();
            timer.stop();
            if (timed) { timer.record(); mem.snapshot_after(); }
        }

        BenchRow row;
        row.date          = current_date_str();
        row.feature       = "reduce/nmf";
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
        skip_no_sample(pz_path);

        SyntheticMatrix syn = make_synthetic_matrix(200, 500);
        int64_t n_cells = syn.cols;

        // Build pinned-host PzDeviceMatrix for synthetic data.
        int   *d_indptr = nullptr, *d_indices = nullptr;
        float *d_values = nullptr;
        cudaMalloc(&d_indptr,  (syn.cols + 1) * sizeof(int));
        cudaMalloc(&d_indices, syn.nnz        * sizeof(int));
        cudaMalloc(&d_values,  syn.nnz        * sizeof(float));
        cudaMemcpy(d_indptr,  syn.indptr.data(),  (syn.cols + 1) * sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, syn.indices.data(), syn.nnz        * sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_values,  syn.values.data(),  syn.nnz        * sizeof(float), cudaMemcpyHostToDevice);

        int   *h_indptr = nullptr, *h_indices = nullptr;
        float *h_values = nullptr;
        cudaMallocHost(&h_indptr,  (syn.cols + 1) * sizeof(int));
        cudaMallocHost(&h_indices, syn.nnz        * sizeof(int));
        cudaMallocHost(&h_values,  syn.nnz        * sizeof(float));
        memcpy(h_indptr,  syn.indptr.data(),  (syn.cols + 1) * sizeof(int));
        memcpy(h_indices, syn.indices.data(), syn.nnz        * sizeof(int));
        memcpy(h_values,  syn.values.data(),  syn.nnz        * sizeof(float));

        io::PzDeviceMatrix mat;
        mat.mat.rows    = syn.rows;
        mat.mat.cols    = syn.cols;
        mat.mat.nnz     = syn.nnz;
        mat.mat.col_ptr     = factornet::gpu::DeviceMemory<int>::wrap(d_indptr,  syn.cols + 1);
        mat.mat.row_indices = factornet::gpu::DeviceMemory<int>::wrap(d_indices, syn.nnz);
        mat.mat.values      = factornet::gpu::DeviceMemory<float>::wrap(d_values, syn.nnz);
        mat.host_retained = true;
        mat.host_indptr.reset(h_indptr,  [](int*   p){ cudaFreeHost(p); });
        mat.host_indices.reset(h_indices,[](int*   p){ cudaFreeHost(p); });
        mat.host_values.reset(h_values,  [](float* p){ cudaFreeHost(p); });
        mat.producer_stream = stream;

        reduce::nmf::NmfConfig cfg;
        cfg.rank     = std::min(NMF_K, std::min(syn.rows, syn.cols) - 1);
        cfg.max_iter = 50;

        BenchTimer        timer(stream);
        PeakMemoryTracker mem;

        for (int iter = 0; iter < BENCH_WARMUP_ITERS + BENCH_TIMED_ITERS; ++iter) {
            bool timed = (iter >= BENCH_WARMUP_ITERS);
            if (timed) mem.snapshot_before();
            timer.start();
            auto result = reduce::nmf::fit(mat, cfg);
            cudaDeviceSynchronize();
            timer.stop();
            if (timed) { timer.record(); mem.snapshot_after(); }
        }

        cudaFree(d_indptr); cudaFree(d_indices); cudaFree(d_values);

        BenchRow row;
        row.date          = current_date_str();
        row.feature       = "reduce/nmf";
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
