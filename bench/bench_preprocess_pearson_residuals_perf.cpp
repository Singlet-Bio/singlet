// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_preprocess_pearson_residuals_perf.cpp
//
// CYCLE-157 Phase E — preprocess/pearson_residuals standalone kernel benchmark.
//
// Benchmarks pearson_residual_variance() (include/singlet/gpu/preprocess/pearson_residuals.h)
// against synthetic CSC count matrices at two medium scales:
//
//   Scale "10k":  n_cells=10000, n_genes=5000, density=5%
//   Scale "30k":  n_cells=30000, n_genes=5000, density=5%
//
// Protocol: 2 warmup + 5 timed iterations; wall time via cudaEventElapsedTime.
// Memory via cudaMemGetInfo delta (PeakMemTracker).
// Does NOT materialize the residual matrix — just calls the variance kernel.
//
// Output (stdout): CSV header + rows:
//   scale,n_cells,n_genes,density,wall_ms,mem_mb
//
// Node: g003 (V100S sm_70-compatible, CUDA arch 70).

#include <singlet/gpu/bench/harness.h>
#include <singlet/gpu/preprocess/pearson_residuals.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

using namespace singlet::gpu;

// ---------------------------------------------------------------------------
// upload_synthetic_csc — build a CSC sparse matrix on device from host vecs.
// Returns a PzDeviceMatrix whose mat.{rows,cols,nnz,col_ptr,row_indices,values}
// are populated from the SyntheticMatrix.  Uses core::DeviceMemory<T> throughout
// (no raw cudaMalloc — Rule 5).
// ---------------------------------------------------------------------------
static io::PzDeviceMatrix upload_synthetic_csc(const SyntheticMatrix& syn,
                                               cudaStream_t stream)
{
    // Allocate device buffers via DeviceMemory (RAII, Rule 5).
    core::DeviceMemory<int>   d_col_ptr(syn.cols + 1);
    core::DeviceMemory<int>   d_row_idx(syn.nnz);
    core::DeviceMemory<float> d_values(syn.nnz);

    cudaMemcpyAsync(d_col_ptr.get(), syn.indptr.data(),
                    (syn.cols + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_row_idx.get(), syn.indices.data(),
                    syn.nnz * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_values.get(), syn.values.data(),
                    syn.nnz * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    io::PzDeviceMatrix pzmat;
    pzmat.mat.rows        = syn.rows;
    pzmat.mat.cols        = syn.cols;
    pzmat.mat.nnz         = syn.nnz;
    pzmat.mat.col_ptr     = std::move(d_col_ptr);
    pzmat.mat.row_indices = std::move(d_row_idx);
    pzmat.mat.values      = std::move(d_values);
    return pzmat;
}

// ---------------------------------------------------------------------------
// bench_scale — run 2 warmup + 5 timed iters of pearson_residual_variance.
// Returns median wall_ms and peak mem_mb.
// ---------------------------------------------------------------------------
static void bench_scale(
    const io::PzDeviceMatrix& pzmat,
    const preprocess::PearsonResidualsConfig& cfg,
    cudaStream_t stream,
    double& out_wall_ms,
    double& out_mem_mb)
{
    constexpr int WARMUP = 2;
    constexpr int TIMED  = 5;
    double samples[TIMED]{};
    double mem_samples[TIMED]{};

    bench::BenchTimer    timer;
    bench::PeakMemTracker mem;

    for (int iter = 0; iter < WARMUP + TIMED; ++iter) {
        const bool is_timed = (iter >= WARMUP);
        if (is_timed) mem.sample_before();

        timer.start(stream);
        // Call the CYCLE-118 standalone kernel — result discarded (wall-time only).
        auto var_out = preprocess::pearson_residual_variance(pzmat, cfg, stream);
        // pearson_residual_variance syncs internally before returning (Rule 9).
        timer.stop(stream);

        const double ms = timer.elapsed_ms();
        if (is_timed) {
            mem.sample_after();
            samples[iter - WARMUP]     = ms;
            mem_samples[iter - WARMUP] = mem.peak_delta_mb();
            mem.reset();
        }
    }

    std::sort(samples, samples + TIMED);
    out_wall_ms = (TIMED % 2 == 1)
        ? samples[TIMED / 2]
        : 0.5 * (samples[TIMED / 2 - 1] + samples[TIMED / 2]);
    out_mem_mb = *std::max_element(mem_samples, mem_samples + TIMED);
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------
int main() {
    if (!gpu_available()) {
        std::puts("NO GPU — skipping Cycle 157 pearson_residuals bench.");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // CSV header.
    std::printf("scale,n_cells,n_genes,density,wall_ms,mem_mb\n");
    std::fflush(stdout);

    // Config: theta=100 (Lause default), no clip (pure variance bench).
    preprocess::PearsonResidualsConfig cfg;
    cfg.theta = 100.f;

    // Scales: 10k and 30k cells, 5000 genes, density 5%.
    struct Scale { const char* name; int n_cells; int n_genes; float density; };
    const Scale scales[] = {
        { "10k",  10000, 5000, 0.05f },
        { "30k",  30000, 5000, 0.05f },
    };

    for (const auto& s : scales) {
        std::printf("[bench] Synthesizing %s: %d cells × %d genes density=%.0f%%...\n",
                    s.name, s.n_cells, s.n_genes, s.density * 100.f);
        std::fflush(stdout);

        SyntheticMatrix syn = make_synthetic_matrix(
            s.n_genes, s.n_cells, s.density, /*seed=*/42);

        io::PzDeviceMatrix pzmat = upload_synthetic_csc(syn, stream);

        double wall_ms = 0.0, mem_mb = 0.0;
        bench_scale(pzmat, cfg, stream, wall_ms, mem_mb);

        // CSV row.
        std::printf("%s,%d,%d,%.2f,%.3f,%.1f\n",
                    s.name, s.n_cells, s.n_genes, s.density, wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "preprocess/pearson_residuals";
        row.scale         = s.name;
        row.impl          = "singlet-gpu";
        row.wall_ms       = wall_ms;
        row.mem_mb        = mem_mb;
        row.cells_per_sec = bench::throughput(static_cast<int64_t>(s.n_cells), wall_ms);
        row.commit        = bench::git_short_sha();
        bench::log_row(row);
    }

    cudaStreamDestroy(stream);
    std::printf("[bench] Cycle 157 pearson_residuals perf bench complete.\n");
    return 0;
}
