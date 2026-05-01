// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_integrate_combat_perf.cpp
//
// CYCLE-175 Phase E — integrate/combat standalone kernel benchmark.
//
// Benchmarks integrate::combat() (include/singlet-gpu/integrate/combat.h)
// against scanpy.pp.combat CPU baseline at two medium scales:
//
//   Scale "10k":  n_cells=10000, n_genes=5000, dense (all cells), 4 batches, max_iter=2.
//   Scale "30k":  n_cells=30000, n_genes=5000, dense (all cells), 4 batches, max_iter=2.
//
// NOTE: ComBat requires dense-friendly data (not just sparse nnz hits) because
//   it materialises a full m×n dense Z-buffer in Pass 3.  For a fair comparison
//   with scanpy.pp.combat (which also works on dense X), we use a 100% dense
//   synthetic CSC (density=1.0) — every gene expressed in every cell.
//
// GPU kernel: 7 device passes, in-place Z-buffer reuse, EB shrinkage at max_iter=2.
//   (a) combat_gene_scatter_kernel     — α_g, sum_x2 via atomic-scatter by gene.
//   (b) combat_alpha_kernel            — α_g = sum_x[g] / n.
//   (c) combat_gb_scatter_kernel       — per-(g,b) sufficient stats.
//   (d) combat_pooled_var_kernel       — pooled σ²_g.
//   (e) combat_z_fill_implicit_kernel  — Z dense fill (-α/σ).
//   (f) combat_z_scatter_stored_kernel — Z overwrite stored entries.
//   (g) combat_gamma_scatter_kernel    — γ_g,b / δ²_g,b scatter.
//   (h) combat_gamma_finalize_kernel   — finalize γ and δ².
//   (i) combat_eb_hyperparams_kernel   — EB hyperparameters per batch.
//   (j) combat_eb_shrink_kernel × max_iter=2 — EB shrinkage.
//   (k) combat_adjust_kernel           — X_adj = (Z - γ*) · σ / sqrt(δ²*) + α.
// Reference: Johnson WE, Li C, Rabinovic A (2007) Biostatistics 8:118-127.
//
// Protocol:
//   Expression matrix: synthetic dense CSC (5000 genes × n_cells, density=100%),
//     LCG seed=42, integer counts 0–20 (log-normalised range).
//   Batch labels: round-robin 0..3 across n_cells columns.
//   combat() — 2 warmup + 5 timed (cudaEvent).
//   Memory — bench::PeakMemTracker (cudaMemGetInfo delta).
//
// Output (stdout): CSV header + rows:
//   scale,n_cells,n_genes,n_batches,max_iter,wall_ms,mem_mb
//
// §J.7 prediction: scanpy.pp.combat is numpy-vectorised (mostly).  The EB step
//   involves a per-gene Python loop in the original R but scanpy implements it
//   analytically (closed-form EB, parametric prior, no Python loop).  So SOTA
//   structure is "vectorized scipy/numpy" — class 2-3 (50-300×) expected.
//   Could be class 1 if the dense m*n pass dominates on CPU.
//
// Node: any non-excluded GPU node (--exclude=g001,g002,g005 per §J.2).

#include <singlet_gpu/bench/harness.h>
#include <singlet-gpu/integrate/combat.h>
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

using namespace singlet_gpu;

// ---------------------------------------------------------------------------
// make_dense_csc_pzmat — synthetic 100%-dense CSC expression matrix
// (m_genes × n_cells). LCG seed=42, integer counts 0–20.
// ---------------------------------------------------------------------------
static io::PzDeviceMatrix make_dense_csc_pzmat(int m_genes, int n_cells,
                                                cudaStream_t stream)
{
    const size_t nnz = static_cast<size_t>(m_genes) * static_cast<size_t>(n_cells);

    std::vector<int>   h_col_ptr(static_cast<size_t>(n_cells) + 1);
    std::vector<int>   h_row_idx(nnz);
    std::vector<float> h_values(nnz);

    // Dense CSC: column c has rows 0..m_genes-1 in order.
    for (int c = 0; c <= n_cells; ++c)
        h_col_ptr[static_cast<size_t>(c)] = c * m_genes;

    uint64_t rng = 42ULL ^ 0xDEADBEEFCAFEBABEull;
    auto lcg_next = [&]() -> uint64_t {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return rng;
    };

    for (int c = 0; c < n_cells; ++c) {
        for (int g = 0; g < m_genes; ++g) {
            const size_t e = static_cast<size_t>(c) * m_genes + g;
            h_row_idx[e] = g;
            h_values[e]  = static_cast<float>(lcg_next() % 21u);  // counts 0–20
        }
    }

    const int actual_nnz = static_cast<int>(nnz);

    core::DeviceMemory<int32_t> d_col_ptr(static_cast<size_t>(n_cells) + 1);
    core::DeviceMemory<int32_t> d_row_idx(static_cast<size_t>(actual_nnz));
    core::DeviceMemory<float>   d_values (static_cast<size_t>(actual_nnz));

    cudaMemcpyAsync(d_col_ptr.get(), h_col_ptr.data(),
                    (static_cast<size_t>(n_cells) + 1) * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_row_idx.get(), h_row_idx.data(),
                    static_cast<size_t>(actual_nnz) * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_values.get(), h_values.data(),
                    static_cast<size_t>(actual_nnz) * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    io::PzDeviceMatrix pzmat;
    pzmat.mat.rows        = m_genes;
    pzmat.mat.cols        = n_cells;
    pzmat.mat.nnz         = actual_nnz;
    pzmat.mat.col_ptr     = std::move(d_col_ptr);
    pzmat.mat.row_indices = std::move(d_row_idx);
    pzmat.mat.values      = std::move(d_values);
    return pzmat;
}

// ---------------------------------------------------------------------------
// make_batch_labels — round-robin batch labels 0..n_batches-1 across n_cells.
// ---------------------------------------------------------------------------
static core::DeviceMemory<int> make_batch_labels(int n_cells, int n_batches,
                                                  cudaStream_t stream)
{
    std::vector<int> h_batch(n_cells);
    for (int c = 0; c < n_cells; ++c)
        h_batch[static_cast<size_t>(c)] = c % n_batches;

    core::DeviceMemory<int> d_batch(n_cells);
    cudaMemcpyAsync(d_batch.get(), h_batch.data(),
                    static_cast<size_t>(n_cells) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return d_batch;
}

// ---------------------------------------------------------------------------
// bench_combat — 2 warmup + 5 timed iterations of integrate::combat().
// Returns median wall_ms and peak mem_mb via out params.
// ---------------------------------------------------------------------------
static void bench_combat(
    const io::PzDeviceMatrix&  X,
    const core::DeviceMemory<int>& d_batch,
    int                        n_batches,
    int                        max_iter,
    cudaStream_t               stream,
    double&                    out_wall_ms,
    double&                    out_mem_mb)
{
    constexpr int WARMUP = 2;
    constexpr int TIMED  = 5;
    double samples[TIMED]{};
    double mem_samples[TIMED]{};

    bench::BenchTimer     timer;
    bench::PeakMemTracker mem;

    integrate::CombatConfig cfg{};
    cfg.max_iter = max_iter;

    for (int iter = 0; iter < WARMUP + TIMED; ++iter) {
        const bool is_timed = (iter >= WARMUP);
        if (is_timed) mem.sample_before();

        timer.start(stream);

        {
            auto result = integrate::combat(X, d_batch.get(), n_batches, cfg, stream);
            (void)result;
        }

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
        std::puts("NO GPU — skipping Cycle 175 integrate/combat bench.");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // CSV header.
    std::printf("scale,n_cells,n_genes,n_batches,max_iter,wall_ms,mem_mb\n");
    std::fflush(stdout);

    // Scales: 10k / 30k cells, 5000 genes, dense, 4 batches, max_iter=2.
    struct Scale { const char* name; int n_cells; int n_genes; int n_batches; int max_iter; };
    const Scale scales[] = {
        { "10k", 10000, 5000, 4, 2 },
        { "30k", 30000, 5000, 4, 2 },
    };

    for (const auto& s : scales) {
        std::printf("[bench] Synthesizing %s: %d genes × %d cells "
                    "(dense, n_batches=%d, max_iter=%d)...\n",
                    s.name, s.n_genes, s.n_cells, s.n_batches, s.max_iter);
        std::fflush(stdout);

        // Build dense CSC expression matrix (untimed).
        io::PzDeviceMatrix X = make_dense_csc_pzmat(s.n_genes, s.n_cells, stream);

        std::printf("[bench] %s: nnz=%d (dense 100%%)\n", s.name, X.mat.nnz);
        std::fflush(stdout);

        // Build batch labels (untimed).
        core::DeviceMemory<int> d_batch = make_batch_labels(s.n_cells, s.n_batches, stream);

        std::printf("[bench] %s: batch labels 0..%d round-robin. "
                    "Timing integrate::combat (max_iter=%d)...\n",
                    s.name, s.n_batches - 1, s.max_iter);
        std::fflush(stdout);

        double wall_ms = 0.0, mem_mb = 0.0;
        bench_combat(X, d_batch, s.n_batches, s.max_iter, stream, wall_ms, mem_mb);

        // CSV row.
        std::printf("%s,%d,%d,%d,%d,%.3f,%.1f\n",
                    s.name, s.n_cells, s.n_genes,
                    s.n_batches, s.max_iter,
                    wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "integrate/combat";
        row.scale         = std::string(s.name);
        row.impl          = "singlet-gpu";
        row.wall_ms       = wall_ms;
        row.mem_mb        = mem_mb;
        row.cells_per_sec = bench::throughput(static_cast<int64_t>(s.n_cells),
                                               wall_ms);
        row.commit        = bench::git_short_sha();
        bench::log_row(row);
    }

    cudaStreamDestroy(stream);
    std::printf("[bench] Cycle 175 integrate/combat perf bench complete.\n");
    return 0;
}
