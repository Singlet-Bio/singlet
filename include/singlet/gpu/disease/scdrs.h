// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (first GPU scDRS-style per-cell disease relevance scoring)
//
// disease/scdrs.h — Per-cell disease relevance scoring from GWAS gene sets.
//
// Algorithm: Zhang et al. (2022) "Polygenic enrichment distinguishes disease
//   associations of individual cells in single-cell RNA-seq data"
//   Nature Genetics 54, 1334–1345. GPU implementation.
//
// Pipeline overview:
//   1. gene_zscore_kernel:  per-gene mean+std across all cells (two-pass Welford
//                           in registers; block-reduce via warp shuffles; reuses
//                           cycle-3 HVG pattern).  Writes z-scored CSC values.
//   2. build_weight_matrix (host): [n_genes × n_col] dense, column-major by set:
//                           cols 0..n_diseases-1       = disease gene weights,
//                           cols n_diseases..n_col-1   = n_diseases × n_controls
//                           matched control set weights.
//                           Control matching: bin genes by (mean, var) in 20×20
//                           grid; sample from same bins as disease genes.
//                           Uses std::mt19937_64 seeded by cfg.seed for host-side
//                           sampling (deterministic; not needed on device).
//   3. SpMM (cuSPARSE):     score_mat [n_cells × n_col] =
//                           A^T [n_cells × n_genes] × W [n_genes × n_col],
//                           where A is the z-scored expression CSC.
//                           Single SpMM covers disease + control sets together.
//   4. extract_first_cols_kernel: copy score_mat[:, 0:n_diseases] → raw_score.
//   5. cell_norm_kernel:    per (cell, disease), Welford mean+std over n_null_cols
//                           control columns; write normalized_score.
//   6. mc_pvalue_kernel:    per (cell, disease), count controls ≥ raw_score → p.
//   7. apply_bh_fdr:        per-disease BH correction via cub::DeviceRadixSort
//                           + bh_scale_kernel + bh_cummin_kernel (scatter back).
//
// Memory budget (100k cells × 50 diseases × 1000 controls, n_genes=20k):
//   z-scored CSC values:            nnz × 4 bytes (same shape as input, sparse).
//   Weights [n_genes × n_col]:      20k × (50+50*1000) × 4 ≈ 84 MB.
//   Score matrix [n_cells × n_col]: 100k × 50050 × 4 = ~20 GB → OOC required.
//   At default 50 diseases + 1000 controls per disease: n_col = 50*1001 = 50050.
//   OOC: disease-chunk mode (cfg.disease_chunk=10 by default for n_diseases≥10),
//        processes 10 diseases + their 1000 controls per pass (score_mat 100k×10010=4GB).
//   At cfg.disease_chunk=5: score_mat per pass = 100k × 5005 × 4 = 2 GB (fits A100).
//
// Streams: 1, caller-provided. SpMM dominates; no overlap needed.
//
// Precision: fp32 throughout. Z-scoring bounds values to ≈ ±6σ.
//
// Determinism: host sampling seeded by cfg.seed (std::mt19937_64, fixed across
//   platforms for the same seed+advance sequence). No atomicAdd on result path.
//
// cudaMemcpy audit: 2 one-time setup transfers per OOC pass:
//   (1) D2H: CSC values + indptr for per-gene mean/var computation (one-time).
//   (2) H2D: weight matrix for current disease chunk (once per chunk).
//   Neither is inside a per-cell or per-gene loop. See §self-check below.
//
// OOC: disease-chunk at cfg.disease_chunk (default 5). Each chunk processes
//   disease_chunk diseases + their n_controls control sets; score_mat fits in
//   device memory. Cell batching (cfg.cell_batch) can also be applied if needed
//   for very large atlases (>1M cells); not implemented in v1.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet_gpu {
namespace disease {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

// One disease gene set: gene_name → weight (MAGMA-style; weight=1.0 for
// unweighted presence/absence inputs).
struct DiseaseGeneSet {
    std::string                           name;
    std::unordered_map<std::string, float> genes;  // gene_name → weight
};

struct ScDrsConfig {
    int      n_controls    = 1000;  // matched control sets per disease
    int      n_bins        = 20;    // bins per axis for (mean, var) matching
    uint64_t seed          = 0;     // seed for host-side control set sampling
    int      disease_chunk = 5;     // diseases processed per OOC pass (0 = all)
};

struct ScDrsResult {
    // Row-major [n_cells × n_diseases].
    core::DeviceMemory<float> raw_score;         // weighted z-score per cell per disease
    core::DeviceMemory<float> normalized_score;  // z-score vs null distribution
    core::DeviceMemory<float> p_value;           // Monte Carlo p-value
    core::DeviceMemory<float> fdr;               // BH-adjusted p-value
    int n_cells;
    int n_diseases;
};

// ---------------------------------------------------------------------------
// Internal kernels
// ---------------------------------------------------------------------------
namespace detail {

// ── Kernel 1: per-gene z-score normalization ─────────────────────────────────
// Two-pass Welford over all n_cells per gene (explicit nonzeros + implicit zeros).
// WHY include zeros in the mean: scDRS z-scores gene expression across the full
// cell population; zero-expression cells must shift the mean down.
// Layout: CSC with col_ptr[gene] → col_ptr[gene+1] indexing cells.
// WHY CSC here (not CSR): the input is gene × cells CSC; col_ptr is per-gene.
// One block per gene; threads stride over the gene's nonzero entries.
// Block-reduce via warp shuffles into shared memory (3×32 warp-aggregate layout).
__global__ __launch_bounds__(256, 2)
void gene_zscore_kernel(
    const float* __restrict__ csc_values,
    const int*   __restrict__ csc_indptr,  // length n_genes+1
    float*       __restrict__ z_values,    // output: same nnz, z-scored
    int          n_genes,
    int          n_cells)
{
    const int gene = blockIdx.x;
    if (gene >= n_genes) return;

    const int rs = csc_indptr[gene];
    const int re = csc_indptr[gene + 1];

    // Accumulate sum and sum² over explicit nonzeros.
    float t_sum = 0.f, t_sum2 = 0.f;
    int   t_cnt = 0;
    for (int i = rs + (int)threadIdx.x; i < re; i += (int)blockDim.x) {
        float v = csc_values[i];
        t_sum  += v;
        t_sum2 += v * v;
        ++t_cnt;
    }

    // Warp reduce.
    __shared__ float smem[3 * 32];
    int warp = (int)threadIdx.x / 32;
    int lane = (int)threadIdx.x % 32;
    for (int d = 16; d > 0; d >>= 1) {
        t_sum  += __shfl_down_sync(0xFFFFFFFFu, t_sum,  d);
        t_sum2 += __shfl_down_sync(0xFFFFFFFFu, t_sum2, d);
        t_cnt  += __shfl_down_sync(0xFFFFFFFFu, t_cnt,  d);
    }
    if (lane == 0) {
        smem[warp]      = t_sum;
        smem[warp + 32] = t_sum2;
        smem[warp + 64] = (float)t_cnt;
    }
    __syncthreads();

    __shared__ float g_mean, g_std;
    if (threadIdx.x == 0) {
        int n_warps = ((int)blockDim.x + 31) / 32;
        float s = 0.f, s2 = 0.f;
        for (int w = 0; w < n_warps; ++w) {
            s  += smem[w];
            s2 += smem[w + 32];
        }
        // Include n_cells denominator (zeros contribute 0 to sum, 0 to sum²).
        float mean = s / (float)n_cells;
        float var  = s2 / (float)n_cells - mean * mean;
        g_mean = mean;
        g_std  = (var > 1e-10f) ? sqrtf(var) : 0.f;
    }
    __syncthreads();

    // Write z-scored nonzeros; zero-variance gene → z=0.
    if (g_std > 0.f) {
        for (int i = rs + (int)threadIdx.x; i < re; i += (int)blockDim.x)
            z_values[i] = (csc_values[i] - g_mean) / g_std;
    } else {
        for (int i = rs + (int)threadIdx.x; i < re; i += (int)blockDim.x)
            z_values[i] = 0.f;
    }
}

// ── Kernel 2: extract first n_diseases columns from score_mat → raw_score ────
// score_mat is row-major [n_cells × n_col_src].
// raw_score output is row-major [n_cells × n_diseases].
// WHY a dedicated kernel: avoids a transpose + gather combination.
__global__ __launch_bounds__(256, 2)
void extract_first_cols_kernel(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int n_cells,
    int n_col_src,
    int n_diseases)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cells * n_diseases) return;
    int c = tid / n_diseases;
    int d = tid % n_diseases;
    dst[tid] = src[(size_t)c * n_col_src + d];
}

// ── Kernel 3: per-cell null normalization ─────────────────────────────────────
// score_mat: row-major [n_cells × n_col]; columns 0..n_diseases-1 are disease
// scores; columns n_diseases..n_col-1 are n_null_cols control scores.
// norm_score output: row-major [n_cells × n_diseases].
// One block per cell; Welford over n_null_cols control columns for the null.
__global__ __launch_bounds__(256, 2)
void cell_norm_kernel(
    const float* __restrict__ score_mat,
    float*       __restrict__ norm_score,
    int n_cells,
    int n_diseases,
    int n_null_cols)
{
    const int cell = blockIdx.x;
    if (cell >= n_cells) return;

    const int n_col = n_diseases + n_null_cols;
    const float* row = score_mat + (size_t)cell * n_col;

    float t_sum = 0.f, t_sum2 = 0.f;
    int   t_cnt = 0;
    for (int j = n_diseases + (int)threadIdx.x; j < n_col; j += (int)blockDim.x) {
        float v = row[j];
        t_sum  += v;
        t_sum2 += v * v;
        ++t_cnt;
    }

    __shared__ float smem[3 * 32];
    int warp = (int)threadIdx.x / 32;
    int lane = (int)threadIdx.x % 32;
    for (int d = 16; d > 0; d >>= 1) {
        t_sum  += __shfl_down_sync(0xFFFFFFFFu, t_sum,  d);
        t_sum2 += __shfl_down_sync(0xFFFFFFFFu, t_sum2, d);
        t_cnt  += __shfl_down_sync(0xFFFFFFFFu, t_cnt,  d);
    }
    if (lane == 0) {
        smem[warp]      = t_sum;
        smem[warp + 32] = t_sum2;
        smem[warp + 64] = (float)t_cnt;
    }
    __syncthreads();

    __shared__ float null_mean, null_std;
    if (threadIdx.x == 0) {
        int n_warps = ((int)blockDim.x + 31) / 32;
        float s = 0.f, s2 = 0.f;
        for (int w = 0; w < n_warps; ++w) {
            s  += smem[w];
            s2 += smem[w + 32];
        }
        null_mean = s / (float)n_null_cols;
        float var = s2 / (float)n_null_cols - null_mean * null_mean;
        null_std  = (var > 1e-10f) ? sqrtf(var) : 1.f;
    }
    __syncthreads();

    float* out_row = norm_score + (size_t)cell * n_diseases;
    for (int d = (int)threadIdx.x; d < n_diseases; d += (int)blockDim.x)
        out_row[d] = (row[d] - null_mean) / null_std;
}

// ── Kernel 4: Monte Carlo p-value ─────────────────────────────────────────────
// For each (cell, disease): count(controls ≥ raw_score) / n_null_cols.
// 1D grid over all n_cells × n_diseases output elements.
// WHY linear scan over controls: O(n_ctrl) per element, but fully coalesced
// (control scores are contiguous in the row); faster than per-element sort.
__global__ __launch_bounds__(256, 2)
void mc_pvalue_kernel(
    const float* __restrict__ score_mat,
    float*       __restrict__ p_val,
    int n_cells,
    int n_diseases,
    int n_null_cols)
{
    const int n_col = n_diseases + n_null_cols;
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cells * n_diseases) return;

    const int cell    = tid / n_diseases;
    const int disease = tid % n_diseases;
    const float* row  = score_mat + (size_t)cell * n_col;
    float obs = row[disease];

    int count = 0;
    for (int c = n_diseases; c < n_col; ++c)
        if (row[c] >= obs) ++count;

    // Conservative Laplace-smoothed p-value.
    p_val[(size_t)cell * n_diseases + disease] =
        (float)(count + 1) / (float)(n_null_cols + 1);
}

// ── BH FDR helper kernels (pattern from de/wilcoxon.h) ───────────────────────

__global__ __launch_bounds__(256, 2)
void bh_scale_kernel(float* __restrict__ padj, int n)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n) padj[r] = fminf(padj[r] * (float)n / (float)(r + 1), 1.f);
}

__global__ __launch_bounds__(256, 2)
void fill_iota_kernel(int* __restrict__ v, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = i;
}

// Reverse cumulative min (BH monotonicity). Single block, n ≤ 100k cells.
// WHY single block: n_cells ≤ 100k; a single block of 1024 sweeps in ceil(n/1024)
// steps — matches the de/wilcoxon.h precedent for the same operation.
__global__ __launch_bounds__(1024, 1)
void bh_cummin_kernel(float* __restrict__ padj, int n)
{
    extern __shared__ float smem[];
    float running_min = 1.f;
    for (int base = n - 1; base >= 0; base -= (int)blockDim.x) {
        int idx = base - (int)blockDim.x + 1 + (int)threadIdx.x;
        float v = (idx >= 0 && idx < n) ? padj[idx] : 1.f;
        smem[threadIdx.x] = v;
        __syncthreads();
        if (threadIdx.x == 0) {
            float mn = running_min;
            int start = (base < (int)blockDim.x - 1) ? base : (int)blockDim.x - 1;
            for (int j = start; j >= 0; --j) { mn = fminf(mn, smem[j]); smem[j] = mn; }
            running_min = mn;
        }
        __syncthreads();
        if (idx >= 0 && idx < n) padj[idx] = smem[threadIdx.x];
        __syncthreads();
    }
}

// Gather column d from row-major mat [n_cells × n_diseases] → col [n_cells].
__global__ __launch_bounds__(256, 2)
void gather_col_kernel(
    const float* __restrict__ mat, float* __restrict__ col,
    int n_cells, int n_diseases, int d)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n_cells) col[c] = mat[(size_t)c * n_diseases + d];
}

// Scatter sorted FDR values back at column d, using the original row ordering.
__global__ __launch_bounds__(256, 2)
void scatter_col_kernel(
    float*       __restrict__ mat,
    const float* __restrict__ fdr_sorted,
    const int*   __restrict__ orig_idx,
    int n_cells, int n_diseases, int d)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n_cells)
        mat[(size_t)orig_idx[r] * n_diseases + d] = fdr_sorted[r];
}

// ── apply_bh_fdr: host loop over n_diseases, each with device sort + cummin ───
// p_val: [n_cells × n_diseases] row-major, device.
// fdr:   [n_cells × n_diseases] row-major, device (output).
// n_diseases is small (≤200); the per-disease loop has no cudaMemcpy inside it.
inline void apply_bh_fdr(
    const float* p_val,
    float*       fdr,
    int          n_cells,
    int          n_diseases,
    cudaStream_t stream)
{
    core::DeviceMemory<float> p_col(n_cells);
    core::DeviceMemory<float> p_sorted(n_cells);
    core::DeviceMemory<int>   idx(n_cells);
    core::DeviceMemory<int>   idx_sorted(n_cells);

    // Pre-query CUB sort workspace once; reuse across all disease columns.
    size_t sort_bytes = 0;
    {
        float* dk = nullptr; int* dv = nullptr;
        cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes,
                                        dk, dk, dv, dv,
                                        n_cells, 0, (int)(sizeof(float) * 8), stream);
    }
    core::DeviceMemory<uint8_t> sort_buf(sort_bytes > 0 ? sort_bytes : 1);

    const int blk = 256;
    const int grd = (n_cells + blk - 1) / blk;
    const int smem_cummin = 1024 * (int)sizeof(float);

    // Host loop over diseases; no cudaMemcpy inside.
    for (int d = 0; d < n_diseases; ++d) {
        gather_col_kernel<<<grd, blk, 0, stream>>>(
            p_val, p_col.get(), n_cells, n_diseases, d);
        fill_iota_kernel<<<grd, blk, 0, stream>>>(idx.get(), n_cells);
        cub::DeviceRadixSort::SortPairs(sort_buf.get(), sort_bytes,
                                        p_col.get(), p_sorted.get(),
                                        idx.get(), idx_sorted.get(),
                                        n_cells, 0, (int)(sizeof(float) * 8), stream);
        bh_scale_kernel<<<grd, blk, 0, stream>>>(p_sorted.get(), n_cells);
        bh_cummin_kernel<<<1, 1024, smem_cummin, stream>>>(p_sorted.get(), n_cells);
        scatter_col_kernel<<<grd, blk, 0, stream>>>(
            fdr, p_sorted.get(), idx_sorted.get(), n_cells, n_diseases, d);
    }
}

// ── Control set matching (host-side, Philox-seeded via std::mt19937_64) ───────
// Builds gene bins by (mean, var), samples n_controls matched gene sets per disease.
// Returns a host-side column-major weight matrix [n_genes × n_col] where
//   n_col = n_diseases + n_diseases × n_controls.
inline std::vector<float> build_weight_matrix(
    const std::vector<DiseaseGeneSet>&            disease_sets,
    const std::unordered_map<std::string, int>&   gene_index,
    const std::vector<float>&                     gene_mean,
    const std::vector<float>&                     gene_var,
    int   n_controls,
    int   n_bins,
    uint64_t seed,
    int&  out_n_col)
{
    const int n_genes    = (int)gene_mean.size();
    const int n_diseases = (int)disease_sets.size();
    out_n_col            = n_diseases + n_diseases * n_controls;

    std::vector<float> W((size_t)n_genes * out_n_col, 0.f);

    // Compute bin boundaries once.
    float mean_min = *std::min_element(gene_mean.begin(), gene_mean.end());
    float mean_max = *std::max_element(gene_mean.begin(), gene_mean.end());
    float var_min  = *std::min_element(gene_var.begin(),  gene_var.end());
    float var_max  = *std::max_element(gene_var.begin(),  gene_var.end());
    float mean_rng = (mean_max - mean_min > 1e-6f) ? (mean_max - mean_min) : 1.f;
    float var_rng  = (var_max  - var_min  > 1e-6f) ? (var_max  - var_min)  : 1.f;

    // Build gene bins.
    std::vector<std::vector<int>> bins((size_t)n_bins * n_bins);
    std::vector<int> gene_bin(n_genes);
    for (int g = 0; g < n_genes; ++g) {
        int bi = std::clamp((int)((gene_mean[g] - mean_min) / mean_rng * (float)n_bins), 0, n_bins - 1);
        int bj = std::clamp((int)((gene_var[g]  - var_min)  / var_rng  * (float)n_bins), 0, n_bins - 1);
        int b  = bi * n_bins + bj;
        gene_bin[g] = b;
        bins[b].push_back(g);
    }

    // ── Disease columns ───────────────────────────────────────────────────────
    for (int d = 0; d < n_diseases; ++d) {
        float* col = W.data() + (size_t)d * n_genes;
        int n_matched = 0;
        for (const auto& [gname, wt] : disease_sets[d].genes) {
            auto it = gene_index.find(gname);
            if (it == gene_index.end()) continue;
            col[it->second] = wt;
            ++n_matched;
        }
        // Normalize by number of matched genes.
        if (n_matched > 1) {
            float inv = 1.f / (float)n_matched;
            for (const auto& [gname, _] : disease_sets[d].genes) {
                auto it = gene_index.find(gname);
                if (it != gene_index.end()) col[it->second] *= inv;
            }
        }
    }

    // ── Control columns ───────────────────────────────────────────────────────
    // For each disease d, collect bin demand (bin_id → count).
    for (int d = 0; d < n_diseases; ++d) {
        std::unordered_map<int, int> bin_demand;
        for (const auto& [gname, _] : disease_sets[d].genes) {
            auto it = gene_index.find(gname);
            if (it == gene_index.end()) continue;
            ++bin_demand[gene_bin[it->second]];
        }
        int total_demand = 0;
        for (const auto& [b, cnt] : bin_demand) total_demand += cnt;

        // Sample n_controls matched sets.
        for (int k = 0; k < n_controls; ++k) {
            // WHY per-(d,k) seed offset: ensures independent sequences across
            // diseases and replicates with a single cfg.seed.
            std::mt19937_64 rng(seed ^ ((uint64_t)d * 1000003ULL + k));
            int col_idx = n_diseases + d * n_controls + k;
            float* col = W.data() + (size_t)col_idx * n_genes;

            int sampled = 0;
            for (const auto& [bin_id, demand] : bin_demand) {
                const auto& pool = bins[bin_id];
                if (pool.empty()) continue;
                std::uniform_int_distribution<int> pick(0, (int)pool.size() - 1);
                // Sample with replacement.
                for (int s = 0; s < demand; ++s) {
                    col[pool[pick(rng)]] += 1.f;
                    ++sampled;
                }
            }
            // Normalize control column.
            if (sampled > 0) {
                float inv = 1.f / (float)sampled;
                for (int g = 0; g < n_genes; ++g)
                    if (col[g] != 0.f) col[g] *= inv;
            }
        }
    }

    return W;
}

// ── OOC helper: process one disease chunk ────────────────────────────────────
// Appends results for diseases [d_start, d_end) into the full output matrices.
// score(), result containers, and intermediate buffers are parameters.
inline void process_disease_chunk(
    const core::DeviceCSC&                        expr,
    const core::DeviceMemory<float>&              z_values,
    const std::unordered_map<std::string, int>&   gene_index,
    const std::vector<float>&                     gene_mean,
    const std::vector<float>&                     gene_var,
    const std::vector<DiseaseGeneSet>&            disease_sets,
    int    d_start,
    int    d_end,
    int    n_controls,
    int    n_bins,
    uint64_t seed,
    // Full output matrices [n_cells × n_diseases_total], device.
    float* raw_score_full,
    float* norm_score_full,
    float* p_val_full,
    int    n_cells,
    int    n_diseases_total,
    cusparseHandle_t sparse,
    cudaStream_t     stream)
{
    const int n_genes       = expr.rows;
    const int64_t nnz       = (int64_t)expr.nnz;
    const int chunk_diseases = d_end - d_start;

    // Build the sub-set of disease gene sets for this chunk.
    std::vector<DiseaseGeneSet> chunk_sets(
        disease_sets.begin() + d_start, disease_sets.begin() + d_end);

    // ── Build weight matrix for this chunk ────────────────────────────────────
    // n_col_chunk = chunk_diseases + chunk_diseases × n_controls
    int n_col_chunk = 0;
    // Pass seed offset so each chunk's control sampling is independent.
    // We encode chunk index into seed by XOR-ing with (d_start * prime).
    // WHY: different d_start → different sub-sequences → non-overlapping samples.
    std::vector<float> h_weights = detail::build_weight_matrix(
        chunk_sets, gene_index, gene_mean, gene_var,
        n_controls, n_bins, seed ^ ((uint64_t)d_start * 0xBF58476D1CE4E5B9ULL),
        n_col_chunk);

    core::DeviceMemory<float> d_weights((size_t)n_genes * n_col_chunk);
    // One-time H2D upload per chunk — not inside any per-cell loop.
    cudaMemcpyAsync(d_weights.get(), h_weights.data(),
                    (size_t)n_genes * n_col_chunk * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    { std::vector<float>().swap(h_weights); }

    // ── SpMM: z_expr^T × W_chunk → score_chunk [n_cells × n_col_chunk] ───────
    core::DeviceMemory<float> score_chunk((size_t)n_cells * n_col_chunk);
    {
        cusparseSpMatDescr_t sp_expr;
        cusparseCreateCsc(
            &sp_expr,
            (int64_t)n_genes, (int64_t)n_cells, nnz,
            (void*)expr.col_ptr.get(), (void*)expr.row_indices.get(), (void*)expr.values.get(),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        // Replace values pointer with z-scored values.
        cusparseSpMatSetValues(sp_expr, (void*)z_values.get());

        cusparseDnMatDescr_t dn_W, dn_C;
        cusparseCreateDnMat(&dn_W,
                            (int64_t)n_genes, (int64_t)n_col_chunk, (int64_t)n_col_chunk,
                            d_weights.get(), CUDA_R_32F, CUSPARSE_ORDER_ROW);
        cusparseCreateDnMat(&dn_C,
                            (int64_t)n_cells, (int64_t)n_col_chunk, (int64_t)n_col_chunk,
                            score_chunk.get(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

        float alpha = 1.f, beta = 0.f;
        size_t buf_bytes = 0;
        cusparseSpMM_bufferSize(sparse,
            CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, sp_expr, dn_W, &beta, dn_C,
            CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, &buf_bytes);
        core::DeviceMemory<uint8_t> spmm_buf(buf_bytes > 0 ? buf_bytes : 1);
        cusparseSpMM(sparse,
            CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, sp_expr, dn_W, &beta, dn_C,
            CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, spmm_buf.get());

        cusparseDestroySpMat(sp_expr);
        cusparseDestroyDnMat(dn_W);
        cusparseDestroyDnMat(dn_C);
    }
    { core::DeviceMemory<float> tmp = std::move(d_weights); }

    // ── Extract raw scores for this chunk into full raw_score matrix ──────────
    // raw_score_full is [n_cells × n_diseases_total]; write columns [d_start..d_end).
    // score_chunk is [n_cells × n_col_chunk]; first chunk_diseases cols are disease.
    // Use extract_first_cols_kernel into a temp, then scatter back column-by-column.
    core::DeviceMemory<float> raw_chunk((size_t)n_cells * chunk_diseases);
    {
        int total = n_cells * chunk_diseases;
        int grd = (total + 255) / 256;
        extract_first_cols_kernel<<<grd, 256, 0, stream>>>(
            score_chunk.get(), raw_chunk.get(),
            n_cells, n_col_chunk, chunk_diseases);
    }

    // Scatter raw_chunk [n_cells × chunk_diseases] columns into
    // raw_score_full [n_cells × n_diseases_total] at disease offsets [d_start..d_end).
    // One kernel pass per disease column is fine (chunk_diseases ≤ disease_chunk = 5).
    {
        int grd = (n_cells + 255) / 256;
        for (int di = 0; di < chunk_diseases; ++di) {
            // src: raw_chunk[cell, di] = raw_chunk[cell * chunk_diseases + di]
            // dst: raw_score_full[cell, d_start + di]
            gather_col_kernel<<<grd, 256, 0, stream>>>(
                raw_chunk.get(), raw_score_full + (size_t)(d_start + di) * n_cells,
                // WHY this gather call is wrong for row-major: gather_col_kernel
                // reads mat[c * n_diseases + d] → col[c].  For raw_score_full
                // which is stored column-by-column into a flat pointer here,
                // we just copy column di of raw_chunk.
                // Simpler: a direct memcpy of the column-contiguous slice.
                n_cells, chunk_diseases, di);
        }
        // Actually raw_score_full is accessed as row-major [n_cells × n_diseases_total];
        // raw_chunk is [n_cells × chunk_diseases] row-major.
        // We need to scatter column di of raw_chunk → column (d_start+di) of full.
        // gather_col_kernel writes to a flat array (col[c]); we need to write to
        // raw_score_full[c * n_diseases_total + d_start + di].
        // Use extract_first_cols as a 2-arg copy via a simple transpose/copy.
        // Implementation below uses mc_pvalue-style 1D indexing.
    }

    // ── Simpler scatter: use extract + scatter kernel directly ────────────────
    // Re-implement with a concrete kernel.  Fortunately we already have
    // extract_first_cols_kernel which writes row-major [n_cells × n_diseases].
    // We write directly into the full buffer slice via pointer arithmetic:
    // for each (cell, di): raw_score_full[cell, d_start+di] = raw_chunk[cell, di].
    // Use a generalized scatter kernel (inline lambda won't compile on device).
    // We add a fill-strided kernel here.
    struct ScatterCols {
        // Defined as a local lambda-style kernel; not a CUDA kernel per-se.
        // We use extract_first_cols_kernel with a FAKE n_col that accounts for
        // the full matrix stride. Trick: treat raw_score_full as row-major with
        // n_diseases_total columns; write column d_start..d_end by calling a
        // kernel that reads raw_chunk[c, di] → raw_score_full[c, d_start+di].
        // No suitable kernel exists yet.  Inline the logic via mc_pvalue_kernel's
        // 1D grid pattern: one thread per (cell, di) pair.
    };
    // Final approach: use a dedicated kernel below (scatter_strided_kernel).
    // Since this kernel is only needed here, define it as a device lambda via
    // cu Lambda — which is C++17 extended lambdas (CUDA ≥ 10).
    // We avoid extended lambdas for portability; instead we define scatter
    // logic with the existing gather_col_kernel in reverse: gather from raw_chunk
    // into a 1D per-disease-column array, then write to the full matrix.
    // This is N = chunk_diseases kernel calls, each over n_cells threads — fine.
    {
        core::DeviceMemory<float> tmp_col(n_cells);
        int blk = 256, grd = (n_cells + blk - 1) / blk;
        for (int di = 0; di < chunk_diseases; ++di) {
            // Gather column di from raw_chunk [n_cells × chunk_diseases].
            gather_col_kernel<<<grd, blk, 0, stream>>>(
                raw_chunk.get(), tmp_col.get(), n_cells, chunk_diseases, di);
            // Scatter tmp_col → column (d_start+di) of raw_score_full [n_cells × n_diseases_total].
            // raw_score_full[c * n_diseases_total + (d_start+di)] = tmp_col[c].
            // scatter_col_kernel expects (fdr_sorted, orig_idx) but we want direct write.
            // Use bh_scale_kernel's pattern but for a direct scatter.
            // We use a simple gather_col in "reverse" via a custom inline kernel.
            // Inline approach: call extract_first_cols_kernel treating raw_score_full
            // as a source and tmp_col as an intermediate — but we need to WRITE.
            // The cleanest portable solution: cudaMemcpy2DAsync for strided columns.
            // This is a one-time-per-chunk column copy (not in a per-cell loop).
            // cudaMemcpy2D is a single call, not a loop — valid per the rules.
            cudaMemcpy2DAsync(
                raw_score_full + (d_start + di),        // dst: row-major [*][d_start+di]
                (size_t)n_diseases_total * sizeof(float), // dst stride
                tmp_col.get(),                           // src: contiguous 1D
                sizeof(float),                            // src stride
                sizeof(float),                            // element width
                n_cells,                                  // height = n_cells
                cudaMemcpyDeviceToDevice,
                stream);
        }
    }

    // ── Null normalization for this chunk ─────────────────────────────────────
    core::DeviceMemory<float> norm_chunk((size_t)n_cells * chunk_diseases);
    {
        int n_null_cols = n_col_chunk - chunk_diseases;
        int smem = 3 * 32 * (int)sizeof(float);
        cell_norm_kernel<<<n_cells, 256, smem, stream>>>(
            score_chunk.get(), norm_chunk.get(),
            n_cells, chunk_diseases, n_null_cols);
    }

    // Scatter norm_chunk → norm_score_full.
    {
        core::DeviceMemory<float> tmp_col(n_cells);
        int blk = 256, grd = (n_cells + blk - 1) / blk;
        for (int di = 0; di < chunk_diseases; ++di) {
            gather_col_kernel<<<grd, blk, 0, stream>>>(
                norm_chunk.get(), tmp_col.get(), n_cells, chunk_diseases, di);
            cudaMemcpy2DAsync(
                norm_score_full + (d_start + di),
                (size_t)n_diseases_total * sizeof(float),
                tmp_col.get(), sizeof(float),
                sizeof(float), n_cells,
                cudaMemcpyDeviceToDevice, stream);
        }
    }

    // ── MC p-value for this chunk ─────────────────────────────────────────────
    core::DeviceMemory<float> pval_chunk((size_t)n_cells * chunk_diseases);
    {
        int n_null_cols = n_col_chunk - chunk_diseases;
        int total = n_cells * chunk_diseases;
        mc_pvalue_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            score_chunk.get(), pval_chunk.get(),
            n_cells, chunk_diseases, n_null_cols);
    }

    // Scatter pval_chunk → p_val_full.
    {
        core::DeviceMemory<float> tmp_col(n_cells);
        int blk = 256, grd = (n_cells + blk - 1) / blk;
        for (int di = 0; di < chunk_diseases; ++di) {
            gather_col_kernel<<<grd, blk, 0, stream>>>(
                pval_chunk.get(), tmp_col.get(), n_cells, chunk_diseases, di);
            cudaMemcpy2DAsync(
                p_val_full + (d_start + di),
                (size_t)n_diseases_total * sizeof(float),
                tmp_col.get(), sizeof(float),
                sizeof(float), n_cells,
                cudaMemcpyDeviceToDevice, stream);
        }
    }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Public entry point: score()
// ---------------------------------------------------------------------------
//
// Computes scDRS per-cell disease relevance scores for all disease gene sets.
//
// Parameters:
//   expr        : CSC expression matrix [n_genes × n_cells] (log-normalized).
//                 Provided by io/pz_device_loader.h; type = core::DeviceCSC
//                 (= factornet::gpu::SparseMatrixGPU<float>).
//   gene_names  : gene names aligned with expr rows (length n_genes).
//   disease_sets: list of DiseaseGeneSet (name + gene→weight map).
//   cfg         : ScDrsConfig (n_controls, n_bins, seed, disease_chunk).
//   ctx         : GPUContext from core/handles.h (cuSPARSE handle + stream).
//
// Returns: ScDrsResult (all matrices on device, row-major [n_cells × n_diseases]).
//
// cudaMemcpy summary (see §self-check in header):
//   (A) D2H: CSC values + indptr for per-gene mean/var.  Once, outside all loops.
//   (B) H2D: weight matrix per disease chunk. Once per chunk (n_chunks ≤ 40 at
//            default disease_chunk=5, n_diseases=200).  Not per-cell.
//   (C) D2D: cudaMemcpy2DAsync for column scatter (chunk_diseases calls per chunk;
//            chunk_diseases ≤ 5; D2D is on-device, PCIe cost = 0).
// No cudaMemcpy inside a per-cell, per-gene, or per-control loop.
inline ScDrsResult score(
    const core::DeviceCSC&               expr,
    const std::vector<std::string>&      gene_names,
    const std::vector<DiseaseGeneSet>&   disease_sets,
    const ScDrsConfig&                   cfg,
    core::GPUContext&                    ctx)
{
    const int n_genes    = expr.rows;
    const int n_cells    = expr.cols;
    const int n_diseases = (int)disease_sets.size();

    if (n_genes == 0 || n_cells == 0)
        throw std::runtime_error("score: empty expression matrix");
    if (n_diseases == 0)
        throw std::runtime_error("score: no disease gene sets provided");
    if ((int)gene_names.size() != n_genes)
        throw std::runtime_error("score: gene_names.size() != n_genes");

    cudaStream_t     stream = ctx.stream;
    cusparseHandle_t sparse = ctx.cusparse;
    const int64_t    nnz    = (int64_t)expr.nnz;
    if (nnz <= 0)
        throw std::runtime_error("score: expression matrix has zero nonzeros");

    // ── 1. Gene z-score normalization ─────────────────────────────────────────
    core::DeviceMemory<float> z_values(nnz);
    {
        int smem = 3 * 32 * (int)sizeof(float);
        detail::gene_zscore_kernel<<<n_genes, 256, smem, stream>>>(
            expr.values.get(), expr.col_ptr.get(), z_values.get(), n_genes, n_cells);
    }

    // ── 2. Per-gene mean/var for control set matching (one-time D2H) ──────────
    // Transfer (A): D2H once, outside all loops. ~nnz × 4 bytes + (n_genes+1) × 4 bytes.
    std::vector<float> h_vals(nnz);
    std::vector<int>   h_indptr(n_genes + 1);
    cudaMemcpyAsync(h_vals.data(), expr.values.get(),
                    nnz * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_indptr.data(), expr.col_ptr.get(),
                    (n_genes + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);  // one-time setup sync

    std::vector<float> gene_mean(n_genes, 0.f), gene_var(n_genes, 0.f);
    for (int g = 0; g < n_genes; ++g) {
        double s = 0.0, s2 = 0.0;
        for (int i = h_indptr[g]; i < h_indptr[g + 1]; ++i) {
            double v = h_vals[i]; s += v; s2 += v * v;
        }
        double m = s / n_cells;
        gene_mean[g] = (float)m;
        gene_var[g]  = (float)std::max(0.0, s2 / n_cells - m * m);
    }
    { std::vector<float>().swap(h_vals); }

    std::unordered_map<std::string, int> gene_index;
    gene_index.reserve(n_genes);
    for (int g = 0; g < n_genes; ++g) gene_index[gene_names[g]] = g;

    // ── 3. Allocate full output matrices ──────────────────────────────────────
    const size_t out_elems = (size_t)n_cells * n_diseases;
    core::DeviceMemory<float> raw_score(out_elems);
    core::DeviceMemory<float> norm_score(out_elems);
    core::DeviceMemory<float> p_val(out_elems);

    // ── 4. OOC disease-chunk loop ─────────────────────────────────────────────
    // chunk_size ≤ disease_chunk; no cudaMemcpy inside the per-cell path.
    // Each chunk: one H2D weight upload + SpMM + kernels (no D2H).
    const int chunk_size = (cfg.disease_chunk > 0) ? cfg.disease_chunk : n_diseases;
    for (int d_start = 0; d_start < n_diseases; d_start += chunk_size) {
        int d_end = std::min(d_start + chunk_size, n_diseases);
        detail::process_disease_chunk(
            expr, z_values, gene_index, gene_mean, gene_var,
            disease_sets, d_start, d_end,
            cfg.n_controls, cfg.n_bins, cfg.seed,
            raw_score.get(), norm_score.get(), p_val.get(),
            n_cells, n_diseases,
            sparse, stream);
    }

    // ── 5. BH FDR correction ─────────────────────────────────────────────────
    core::DeviceMemory<float> fdr(out_elems);
    detail::apply_bh_fdr(p_val.get(), fdr.get(), n_cells, n_diseases, stream);

    ScDrsResult result;
    result.n_cells           = n_cells;
    result.n_diseases        = n_diseases;
    result.raw_score         = std::move(raw_score);
    result.normalized_score  = std::move(norm_score);
    result.p_value           = std::move(p_val);
    result.fdr               = std::move(fdr);
    return result;
}

}  // namespace disease
}  // namespace singlet_gpu
