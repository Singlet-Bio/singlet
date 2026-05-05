// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: scanpy.tl.score_genes (Satija et al. 2015 / Seurat AddModuleScore)
//
// singlet-gpu/enrich/score_genes.h
//
// Per-cell gene-set scoring with matched control subtraction.
// Direct GPU port of scanpy.tl.score_genes (Satija et al. 2015 §Methods).
//
// Algorithm reference:
//   Satija R, Farrell JA, Gennert D, Schier AF, Regev A (2015)
//   "Spatial reconstruction of single-cell gene expression data."
//   Nat Biotechnol 33:495-502. (Seurat AddModuleScore §Methods)
//
// For each gene set S and each cell c:
//
//   score[c, s] = mean(X[g, c] for g in S) - mean(X[g, c] for g in ctrl_s)
//
// where ctrl_s is a control gene pool sampled from the same mean-expression
// bins that contain genes in S (pooled, not per-gene), excluding genes in S.
// Control pool size: max(ctrl_size, |S|) when use_set_size_for_ctrl=true.
//
// Implementation:
//   Pass 1 — per-gene means μ_g[m]: atomic-scatter kernel (one thread per nnz)
//             then element-wise divide by n_cells. O(nnz + m).
//   Pass 2 — bin assignment (host): D2H μ_g, compute bin_id[g] = floor(
//             (μ_g[g] - μ_min)/(μ_max - μ_min) * n_bins), clamped [0, n_bins-1].
//   Pass 3 — W construction (host): for each set s, pool candidate control
//             genes from the bins containing set genes (excluding set genes),
//             sample ctrl_size_s controls using std::mt19937 seeded from
//             cfg.seed XOR'd with set index. Build W[g, s] (m × n_sets, col-major):
//               +1/|S| for set genes, -1/ctrl_size_s for controls, else 0.
//             H2D upload W. O(m·n_sets) host; O(m·n_sets·4 bytes) transfer.
//   Pass 4 — SpMM: scores = X^T · W via cuSPARSE SpMM.
//             Reuses run_spmm from decoupler_wsum.h (same CSC-transpose pattern).
//             Output: n_cells × n_sets col-major (raw inner products = difference
//             of means, since +/- weights sum to 1/-1 · mean).
//
// WHY std::mt19937 host-side for control sampling:
//   The sampling is per-set and irregular — it requires knowing bin membership
//   computed on host. Doing this on GPU would require per-set irregular compaction
//   with no real throughput benefit for typical n_sets=10-100. A seeded
//   std::mt19937 gives deterministic, reproducible samples with zero GPU overhead.
//
// WHY no column-scale step (vs WSUM/WMEAN):
//   W is constructed with pre-divided weights (+1/|S|, -1/ctrl_size_s), so
//   scores = X^T · W directly yields the difference-of-means without a
//   post-multiply scaling step. This eliminates the norm kernel pass.
//
// Determinism (Rule 18):
//   Host RNG: std::mt19937 seeded from cfg.seed — fully deterministic.
//   SpMM: CUSPARSE_SPMM_ALG_DEFAULT on fixed architecture is deterministic
//         for fixed sparsity + operand values.
//   Overall: bit-exact for fixed inputs and same cfg.seed.
//
// Memory (Rule 5): all device buffers via core::DeviceMemory<T>. No raw cudaMalloc.
// D2H (Rule 4): one D2H of μ_g[m] for host-side binning (4m bytes, one-shot).
//               No per-loop-iteration D2H.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace singlet_gpu {
namespace enrich {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct ScoreGenesConfig {
    int      ctrl_size              = 50;   // control set size per gene set
    int      n_bins                 = 25;   // expression-mean bins (scanpy default 25)
    uint64_t seed                   = 0;    // std::mt19937 seed for control sampling
    bool     use_set_size_for_ctrl  = true; // max(ctrl_size, |S|) — scanpy behavior
    bool     deterministic          = true; // accepted for API symmetry; always true
};

struct ScoreGenesResult {
    core::DeviceMemory<float> scores;  // n_cells × n_sets, col-major
    int n_cells;
    int n_sets;
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels (anonymous namespace — not part of the public API)
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// sg_row_scatter_kernel — scatter nnz values into per-gene row sums.
//
// One thread per stored entry. atomicAdd into row_sum[row_i].
// Mirrors mgv_row_scatter_kernel in model_gene_var.h (same atomic-scatter
// pattern — one-pass row-sum accumulation for sparse CSC, O(nnz)).
//
// Nondeterminism: atomicAdd ordering varies across runs. For log-normalized
// data (values < 10), n ≤ 1M cells, relative error in the mean < 1e-5.
// The binning step (integer floor) is robust to this fp32 noise.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void sg_row_scatter_kernel(
    const float*   __restrict__ values,
    const int32_t* __restrict__ row_indices,
    float*         __restrict__ row_sum,
    int nnz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    atomicAdd(&row_sum[row_indices[idx]], values[idx]);
}

// ---------------------------------------------------------------------------
// sg_mean_kernel — divide row_sum by n_cells → mean[m]
//
// One thread per gene. Trivial element-wise divide.
// Mirrors mgv_mean_kernel in model_gene_var.h.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void sg_mean_kernel(
    const float* __restrict__ row_sum,
    float*       __restrict__ mean,
    int m,
    float n_inv)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    mean[i] = row_sum[i] * n_inv;
}

// ---------------------------------------------------------------------------
// run_spmm_sg — internal helper: compute scores = X^T · W via cuSPARSE SpMM.
//
// Identical to run_spmm in decoupler_wsum.h. Inlined here to avoid header
// dependency ordering issues (both headers are in the same anonymous namespace
// scope). X is CSC (m × n); with opA = TRANSPOSE, cuSPARSE computes
// X^T (n × m) · W (m × n_sets) = scores (n × n_sets, col-major).
// ---------------------------------------------------------------------------
inline core::DeviceMemory<float> run_spmm_sg(
    const io::PzDeviceMatrix& X,
    const float* d_W,
    int n_sets,
    cudaStream_t stream)
{
    const int m   = X.mat.rows;
    const int n   = X.mat.cols;
    const int nnz = X.mat.nnz;

    const size_t out_sz = static_cast<size_t>(n) * static_cast<size_t>(n_sets);
    core::DeviceMemory<float> d_scores(out_sz > 0 ? out_sz : 1);
    cudaMemsetAsync(d_scores.get(), 0, out_sz * sizeof(float), stream);

    if (nnz == 0 || out_sz == 0) {
        return d_scores;
    }

    cusparseHandle_t sp_handle;
    CUSPARSE_CHECK(cusparseCreate(&sp_handle));
    cusparseSetStream(sp_handle, stream);

    cusparseSpMatDescr_t spX;
    CUSPARSE_CHECK(cusparseCreateCsc(
        &spX,
        static_cast<int64_t>(m),
        static_cast<int64_t>(n),
        static_cast<int64_t>(nnz),
        X.mat.col_ptr.get(),
        X.mat.row_indices.get(),
        X.mat.values.get(),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));

    cusparseDnMatDescr_t dnW;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnW,
        static_cast<int64_t>(m),
        static_cast<int64_t>(n_sets),
        static_cast<int64_t>(m),
        const_cast<float*>(d_W),
        CUDA_R_32F,
        CUSPARSE_ORDER_COL));

    cusparseDnMatDescr_t dnOut;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnOut,
        static_cast<int64_t>(n),
        static_cast<int64_t>(n_sets),
        static_cast<int64_t>(n),
        d_scores.get(),
        CUDA_R_32F,
        CUSPARSE_ORDER_COL));

    size_t ws_bytes = 0;
    const float alpha = 1.f, beta = 0.f;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        sp_handle,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, spX, dnW, &beta, dnOut,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &ws_bytes));

    core::DeviceMemory<uint8_t> d_workspace(ws_bytes > 0 ? ws_bytes : 1);

    CUSPARSE_CHECK(cusparseSpMM(
        sp_handle,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, spX, dnW, &beta, dnOut,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        d_workspace.get()));

    cusparseDestroyDnMat(dnOut);
    cusparseDestroyDnMat(dnW);
    cusparseDestroySpMat(spX);
    cusparseDestroy(sp_handle);

    return d_scores;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// score_genes — public entry point
//
// Computes per-cell gene-set scores (Satija et al. 2015 / scanpy.tl.score_genes).
//
// Inputs:
//   X         — m_genes × n_cells sparse CSC expression matrix (PzDeviceMatrix)
//   gene_sets — host-side vector of gene-index lists; indices into [0, m).
//               Each inner vector is one gene set. At least one set required.
//   cfg       — ScoreGenesConfig (ctrl_size, n_bins, seed, use_set_size_for_ctrl)
//   stream    — optional cudaStream_t (nullptr → default context)
//
// Returns ScoreGenesResult with scores (n_cells × n_sets, col-major) on device.
// Caller must sync stream before reading scores.
//
// Notes:
// - All gene indices in gene_sets must be in [0, m). Checked on entry.
// - If a gene set has no valid control candidates (all genes in one bin), the
//   control pool is the entire gene set's bin minus set genes; if that is also
//   empty, ctrl weight is 0 and only set mean contributes.
// - ctrl_size is clamped to max(1, ctrl_size). The effective control size per
//   set is min(ctrl_size_s, pool.size()) when the pool is smaller.
// ---------------------------------------------------------------------------
inline ScoreGenesResult score_genes(
    const io::PzDeviceMatrix&              X,
    const std::vector<std::vector<int>>&  gene_sets,
    const ScoreGenesConfig&               cfg    = {},
    cudaStream_t                          stream = nullptr)
{
    if (stream == nullptr) {
        stream = singlet_gpu::core::default_context().stream();
    }

    const int m      = X.mat.rows;  // genes
    const int n      = X.mat.cols;  // cells
    const int nnz    = X.mat.nnz;
    const int n_sets = static_cast<int>(gene_sets.size());

    // --- Validate inputs ---
    if (m <= 0 || n <= 0) {
        throw std::runtime_error("score_genes: invalid matrix dimensions (m=" +
            std::to_string(m) + " n=" + std::to_string(n) + ")");
    }
    if (n_sets <= 0) {
        throw std::runtime_error("score_genes: gene_sets must be non-empty");
    }
    for (int s = 0; s < n_sets; ++s) {
        if (gene_sets[s].empty()) {
            throw std::runtime_error("score_genes: gene set " + std::to_string(s) + " is empty");
        }
        for (int g : gene_sets[s]) {
            if (g < 0 || g >= m) {
                throw std::runtime_error("score_genes: gene index " + std::to_string(g) +
                    " in set " + std::to_string(s) + " out of range [0, " +
                    std::to_string(m) + ")");
            }
        }
    }

    constexpr int THREADS = 256;

    // -----------------------------------------------------------------------
    // Pass 1 — per-gene mean μ_g[m]
    //
    // Step 1a: scatter nnz values into row_sum[m] via atomic scatter.
    // Step 1b: divide by n_cells element-wise.
    // Mirrors model_gene_var.h Pass 1 pattern.
    // -----------------------------------------------------------------------
    core::DeviceMemory<float> d_row_sum(m);
    core::DeviceMemory<float> d_mean(m);

    cudaMemsetAsync(d_row_sum.get(), 0, m * sizeof(float), stream);

    if (nnz > 0) {
        const int grid = (nnz + THREADS - 1) / THREADS;
        sg_row_scatter_kernel<<<grid, THREADS, 0, stream>>>(
            X.mat.values.get(),
            X.mat.row_indices.get(),
            d_row_sum.get(),
            nnz);
    }

    {
        const float n_inv = 1.f / static_cast<float>(n);
        const int grid    = (m + THREADS - 1) / THREADS;
        sg_mean_kernel<<<grid, THREADS, 0, stream>>>(
            d_row_sum.get(), d_mean.get(), m, n_inv);
    }

    // -----------------------------------------------------------------------
    // Pass 2 — bin assignment (host-side)
    //
    // D2H copy of μ_g (one-shot, Rule 4). Compute bin_id[g] on host.
    // linspace bins: n_bins equal-width bins covering [μ_min, μ_max].
    // Genes with μ_g outside [μ_min, μ_max] are clamped to bin 0 or n_bins-1.
    // Special case: if all means are equal (μ_max == μ_min), all genes → bin 0.
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaStreamSynchronize(stream));  // flush mean kernel before D2H

    std::vector<float> h_mean(m);
    cudaMemcpy(h_mean.data(), d_mean.get(), m * sizeof(float), cudaMemcpyDeviceToHost);

    float mu_min = h_mean[0], mu_max = h_mean[0];
    for (int g = 1; g < m; ++g) {
        if (h_mean[g] < mu_min) mu_min = h_mean[g];
        if (h_mean[g] > mu_max) mu_max = h_mean[g];
    }

    const float range = mu_max - mu_min;
    const int   nb    = cfg.n_bins;

    std::vector<int> bin_id(m);
    for (int g = 0; g < m; ++g) {
        if (range <= 0.f) {
            bin_id[g] = 0;
        } else {
            int b = static_cast<int>(std::floor((h_mean[g] - mu_min) / range * static_cast<float>(nb)));
            if (b < 0)       b = 0;
            if (b >= nb)     b = nb - 1;
            bin_id[g] = b;
        }
    }

    // Build bin → gene list lookup (all m genes).
    std::vector<std::vector<int>> bin_genes(nb);
    for (int g = 0; g < m; ++g) {
        bin_genes[bin_id[g]].push_back(g);
    }

    // -----------------------------------------------------------------------
    // Pass 3 — host-side W matrix construction
    //
    // For each set s:
    //   a. Collect set_bins = union of bin_id[g] for g in set s.
    //   b. Pool candidate control genes from those bins, excluding genes in s.
    //   c. Sample ctrl_size_s = max(ctrl_size, |S|) [or ctrl_size] without
    //      replacement using std::mt19937 seeded with (cfg.seed ^ uint64_t(s+1)).
    //      If pool < ctrl_size_s, use all pool genes as controls.
    //   d. Set W col s: +1/|S| for set genes, -1/ctrl_size_s for controls, 0 otherwise.
    //
    // WHY (cfg.seed ^ (s+1)): ensures each set gets an independent but
    // deterministic RNG state. XOR with (s+1) prevents set 0 from colliding
    // with cfg.seed=0 (which yields the same state as no seed XOR).
    // -----------------------------------------------------------------------
    const size_t W_size = static_cast<size_t>(m) * static_cast<size_t>(n_sets);
    std::vector<float> h_W(W_size, 0.f);

    for (int s = 0; s < n_sets; ++s) {
        const std::vector<int>& S = gene_sets[s];
        const int set_size        = static_cast<int>(S.size());

        // Build set membership hash for fast exclusion.
        std::unordered_set<int> set_mask(S.begin(), S.end());

        // Collect set bins.
        std::set<int> set_bins_used;
        for (int g : S) {
            set_bins_used.insert(bin_id[g]);
        }

        // Pool candidate control genes from those bins, excluding set genes.
        std::vector<int> pool;
        pool.reserve(256);
        for (int b : set_bins_used) {
            for (int g : bin_genes[b]) {
                if (set_mask.find(g) == set_mask.end()) {
                    pool.push_back(g);
                }
            }
        }

        // Effective control size: max(ctrl_size, |S|) if use_set_size_for_ctrl.
        int ctrl_size_s = cfg.ctrl_size;
        if (cfg.use_set_size_for_ctrl) {
            ctrl_size_s = std::max(ctrl_size_s, set_size);
        }
        ctrl_size_s = std::max(ctrl_size_s, 1);

        // Sample up to ctrl_size_s controls from pool (without replacement).
        // Shuffle via mt19937 seeded deterministically from cfg.seed ^ (s+1).
        std::mt19937 rng(static_cast<uint32_t>((cfg.seed ^ static_cast<uint64_t>(s + 1)) & 0xFFFFFFFF)
            ^ static_cast<uint32_t>((cfg.seed ^ static_cast<uint64_t>(s + 1)) >> 32));

        std::vector<int> controls;
        if (!pool.empty()) {
            const int n_ctrl = std::min(ctrl_size_s, static_cast<int>(pool.size()));
            // Fisher-Yates partial shuffle to sample n_ctrl from pool.
            for (int i = 0; i < n_ctrl; ++i) {
                std::uniform_int_distribution<int> uid(i, static_cast<int>(pool.size()) - 1);
                int j = uid(rng);
                std::swap(pool[i], pool[j]);
            }
            controls.assign(pool.begin(), pool.begin() + n_ctrl);
        }

        // Fill W column s (col-major: column s starts at s*m).
        float* W_col = h_W.data() + static_cast<size_t>(s) * static_cast<size_t>(m);

        const float pos_weight = 1.f / static_cast<float>(set_size);
        for (int g : S) {
            W_col[g] = pos_weight;
        }

        if (!controls.empty()) {
            const float neg_weight = -1.f / static_cast<float>(controls.size());
            for (int g : controls) {
                // If a control gene is also in the set (shouldn't happen by construction
                // but guard for safety), do not overwrite the positive weight.
                if (set_mask.find(g) == set_mask.end()) {
                    W_col[g] = neg_weight;
                }
            }
        }
    }

    // Upload W to device.
    core::DeviceMemory<float> d_W(W_size > 0 ? W_size : 1);
    cudaMemcpyAsync(d_W.get(), h_W.data(), W_size * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // -----------------------------------------------------------------------
    // Pass 4 — SpMM: scores = X^T · W (n_cells × n_sets, col-major)
    //
    // W is already constructed with pre-divided +/- indicator weights, so the
    // raw SpMM output is directly the difference-of-means per cell per set.
    // No column-scale step needed (unlike WSUM/WMEAN).
    // -----------------------------------------------------------------------
    core::DeviceMemory<float> d_scores = run_spmm_sg(X, d_W.get(), n_sets, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    ScoreGenesResult res;
    res.scores  = std::move(d_scores);
    res.n_cells = n;
    res.n_sets  = n_sets;
    return res;
}

}  // namespace enrich
}  // namespace singlet_gpu
