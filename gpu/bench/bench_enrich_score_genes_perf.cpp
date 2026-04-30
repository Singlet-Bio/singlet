// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_enrich_score_genes_perf.cpp
//
// CYCLE-158 Phase E — enrich/score_genes standalone kernel benchmark.
//
// Benchmarks score_genes() (include/singlet-gpu/enrich/score_genes.h)
// against synthetic CSC expression matrices at two medium scales:
//
//   Scale "10k":  n_cells=10000, n_genes=5000, density=5%, 5 sets × 50 genes
//   Scale "30k":  n_cells=30000, n_genes=5000, density=5%, 5 sets × 50 genes
//
// Protocol: 2 warmup + 5 timed iterations; wall time via cudaEventElapsedTime.
// Memory via cudaMemGetInfo delta (PeakMemTracker).
// Gene sets pre-built once; reused across iterations (not rebuilt per timed call).
//
// Output (stdout): CSV header + rows:
//   scale,n_cells,n_genes,density,n_sets,wall_ms,mem_mb
//
// Node: g050 (H100 sm_90-compatible).

#include <singlet_gpu/bench/harness.h>
#include <singlet-gpu/enrich/score_genes.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

using namespace singlet_gpu;

// ---------------------------------------------------------------------------
// upload_synthetic_csc — build a CSC sparse matrix on device from host vecs.
// Returns a PzDeviceMatrix whose mat.{rows,cols,nnz,col_ptr,row_indices,values}
// are populated from the SyntheticMatrix.  Uses core::DeviceMemory<T> throughout
// (no raw cudaMalloc — Rule 5).
// ---------------------------------------------------------------------------
static io::PzDeviceMatrix upload_synthetic_csc(const SyntheticMatrix& syn,
                                               cudaStream_t stream)
{
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
// make_gene_sets — build n_sets random gene index lists of size set_size each.
// Indices drawn uniformly from [0, n_genes) with a simple LCG (seed=99).
// Gene sets are pre-computed once and reused across all timed iterations.
// ---------------------------------------------------------------------------
static std::vector<std::vector<int>> make_gene_sets(int n_genes,
                                                     int n_sets,
                                                     int set_size,
                                                     uint32_t seed = 99)
{
    std::vector<std::vector<int>> gene_sets(n_sets);
    uint64_t rng = static_cast<uint64_t>(seed) ^ 0xCAFEBABEDEADBEEFull;
    auto next_rng = [&]() -> uint64_t {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return rng;
    };
    for (int s = 0; s < n_sets; ++s) {
        gene_sets[s].reserve(set_size);
        for (int i = 0; i < set_size; ++i) {
            gene_sets[s].push_back(static_cast<int>(next_rng() % static_cast<uint64_t>(n_genes)));
        }
    }
    return gene_sets;
}

// ---------------------------------------------------------------------------
// bench_scale — run 2 warmup + 5 timed iters of score_genes.
// Returns median wall_ms and peak mem_mb.
// Gene sets must be pre-built before calling (not rebuilt per iter).
// ---------------------------------------------------------------------------
static void bench_scale(
    const io::PzDeviceMatrix&             pzmat,
    const std::vector<std::vector<int>>& gene_sets,
    const enrich::ScoreGenesConfig&       cfg,
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
        // Call the CYCLE-129 score_genes kernel — result discarded (wall-time only).
        // score_genes() calls cudaStreamSynchronize internally before returning.
        auto result = enrich::score_genes(pzmat, gene_sets, cfg, stream);
        // Ensure GPU work is complete before stopping the timer.
        cudaStreamSynchronize(stream);
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
        std::puts("NO GPU — skipping Cycle 158 score_genes bench.");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // CSV header.
    std::printf("scale,n_cells,n_genes,density,n_sets,wall_ms,mem_mb\n");
    std::fflush(stdout);

    // Config: scanpy defaults — ctrl_size=50, n_bins=25.
    enrich::ScoreGenesConfig cfg;
    cfg.ctrl_size             = 50;
    cfg.n_bins                = 25;
    cfg.seed                  = 42;
    cfg.use_set_size_for_ctrl = true;

    // Gene set parameters: 5 sets × 50 genes each, indices in [0, n_genes).
    constexpr int N_SETS   = 5;
    constexpr int SET_SIZE = 50;

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

        // Build synthetic CSC matrix (genes = rows, cells = cols in PzDeviceMatrix).
        SyntheticMatrix syn = make_synthetic_matrix(
            s.n_genes, s.n_cells, s.density, /*seed=*/42);

        io::PzDeviceMatrix pzmat = upload_synthetic_csc(syn, stream);

        // Pre-build gene sets once — indices in [0, n_genes).
        std::vector<std::vector<int>> gene_sets =
            make_gene_sets(s.n_genes, N_SETS, SET_SIZE, /*seed=*/99);

        std::printf("[bench] Running %s: %d sets × %d genes each...\n",
                    s.name, N_SETS, SET_SIZE);
        std::fflush(stdout);

        double wall_ms = 0.0, mem_mb = 0.0;
        bench_scale(pzmat, gene_sets, cfg, stream, wall_ms, mem_mb);

        // CSV row.
        std::printf("%s,%d,%d,%.2f,%d,%.3f,%.1f\n",
                    s.name, s.n_cells, s.n_genes, s.density,
                    N_SETS, wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "enrich/score_genes";
        row.scale         = s.name;
        row.impl          = "singlet-gpu";
        row.wall_ms       = wall_ms;
        row.mem_mb        = mem_mb;
        row.cells_per_sec = bench::throughput(static_cast<int64_t>(s.n_cells), wall_ms);
        row.commit        = bench::git_short_sha();
        bench::log_row(row);
    }

    cudaStreamDestroy(stream);
    std::printf("[bench] Cycle 158 score_genes perf bench complete.\n");
    return 0;
}
