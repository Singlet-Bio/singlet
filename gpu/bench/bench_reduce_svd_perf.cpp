// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_reduce_svd_perf.cpp
//
// Benchmark: reduce/svd (factornet auto-select PCA)  vs  cuml TruncatedSVD / PCA
//
// What is measured:
//   - svd::auto_select() time for k=30 components on the log-normalized matrix.
//   - Excludes load_pz time and log-normalize time.
//   - Peak device memory delta.
//   - Throughput in cells/s.
//
// WHY k=30: standard scRNA-seq PCA rank; matches cuml default in rapids-singlecell.
//
// SOTA reference: cuml.decomposition.TruncatedSVD(n_components=30) via svd_ref.py.
// (Falls back to sklearn if cuml absent.)
//
// Scale: 10k — GSM4037629 (11,560 cells).
// Fallback: synthetic 200×500.
//
// NOTE: svd::auto_select and all factornet SVD adapters require the PzDeviceMatrix
// to have been loaded with keep_host_pinned=true (see integration-notes.md finding 0a).

#include <singlet_gpu/bench/harness.h>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/preprocess/lognorm.h>
#include <singlet-gpu/reduce/svd/auto_select.h>
#include <singlet-gpu/reduce/svd/types.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
using namespace singlet_gpu;

static constexpr const char* REF_SCRIPT =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/refs/svd_ref.py";
static constexpr int SVD_K = 30;

int main() {
    if (!gpu_available()) { skip_no_gpu(); return 0; }

    const std::string pz_path = std::string(CANONICAL_SAMPLE_DIR) + "/counts.1pz";
    bool have_real = fs::exists(pz_path);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // SOTA reference subprocess.
    float sota_wall = -1.0f, sota_mem = -1.0f;
    if (have_real && fs::exists(REF_SCRIPT)) {
        char cmd[1024]{};
        std::snprintf(cmd, sizeof(cmd),
            "python3 %s %s %d 2>/dev/null", REF_SCRIPT, CANONICAL_SAMPLE_DIR, SVD_K);
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
        // keep_host_pinned=true: factornet SVD adapters read host pinned CSC.
        io::PzDeviceMatrix mat = io::load_pz(pz_path, stream, /*keep_host_pinned=*/true);
        cudaStreamSynchronize(stream);
        int64_t n_cells = mat.mat.cols;

        // Pre-normalize (exclude from timing).
        {
            preprocess::LogNormConfig lnc;
            auto lnr = preprocess::log_normalize(mat.mat, lnc, stream);
            cudaStreamSynchronize(stream);
        }

        reduce::svd::SvdConfig cfg;
        // k_max is set inside auto_select; pass k via the second argument.

        BenchTimer        timer(stream);
        PeakMemoryTracker mem;

        for (int iter = 0; iter < BENCH_WARMUP_ITERS + BENCH_TIMED_ITERS; ++iter) {
            bool timed = (iter >= BENCH_WARMUP_ITERS);
            if (timed) mem.snapshot_before();
            timer.start();
            auto result = reduce::svd::auto_select(mat, SVD_K, cfg);
            cudaStreamSynchronize(stream);
            timer.stop();
            if (timed) { timer.record(); mem.snapshot_after(); }
        }

        BenchRow row;
        row.date          = current_date_str();
        row.feature       = "reduce/svd-auto";
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

        int   *d_indptr = nullptr, *d_indices = nullptr;
        float *d_values = nullptr;
        cudaMalloc(&d_indptr,  (syn.cols + 1) * sizeof(int));
        cudaMalloc(&d_indices, syn.nnz        * sizeof(int));
        cudaMalloc(&d_values,  syn.nnz        * sizeof(float));
        cudaMemcpy(d_indptr,  syn.indptr.data(),  (syn.cols + 1) * sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, syn.indices.data(), syn.nnz        * sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_values,  syn.values.data(),  syn.nnz        * sizeof(float), cudaMemcpyHostToDevice);

        // For the synthetic path, build a pinned-host PzDeviceMatrix manually.
        // Allocate pinned copies so keep_host_pinned semantics are satisfied.
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

        reduce::svd::SvdConfig cfg;
        int k = std::min(SVD_K, std::min(syn.rows, syn.cols) - 1);

        BenchTimer        timer(stream);
        PeakMemoryTracker mem;

        for (int iter = 0; iter < BENCH_WARMUP_ITERS + BENCH_TIMED_ITERS; ++iter) {
            bool timed = (iter >= BENCH_WARMUP_ITERS);
            if (timed) mem.snapshot_before();
            timer.start();
            auto result = reduce::svd::auto_select(mat, k, cfg);
            cudaStreamSynchronize(stream);
            timer.stop();
            if (timed) { timer.record(); mem.snapshot_after(); }
        }

        cudaFree(d_indptr); cudaFree(d_indices); cudaFree(d_values);

        BenchRow row;
        row.date          = current_date_str();
        row.feature       = "reduce/svd-auto";
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
