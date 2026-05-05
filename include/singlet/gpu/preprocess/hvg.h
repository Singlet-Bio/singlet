// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (no factornet equivalent for HVG)
//
// singlet-gpu/preprocess/hvg.h — Highly variable gene selection.
// Flavors: SeuratV3 (Hafemeister & Satija 2019) and PearsonResiduals (Lause et al. 2021).
//
// Passes:
//   1. Gene moments (fp32+Kahan, fp64 promo) via CSR transpose.  O(nnz).
//   2. SeuratV3: LOWESS fit log10(mean)→log10(var), single-block shared-memory, O(m²).
//      PearsonResiduals: NB residual variance with closed-form-zeros (O(nnz + m·n)).
//   3. SeuratV3: clipped normalized variance.  O(nnz).
//   4. Top-N: cub::DeviceRadixSort on negated scores.  O(m log m).
//
// CSR strategy: cusparseCsr2cscEx2 inline (same as factornet DualCSR).
//   WHY not DualCSR: mat is a const ref; GPUMatrix takes ownership.
//   Workspace: cuSPARSE buf + 12N + 4m bytes.
//
// LOWESS: single-block, 1024 threads, shared memory (max 8192 genes × 4 arrays × 4B).
//   Each thread = one query gene; O(m) scan per query for neighbor radius;
//   3×3 Cramer WLS in fp64.  WHY not cuSOLVER: 20k kernel launches >> 1 block.
//
// Sync exceptions (§D.9): (1) m-byte promo-flag copy for fp64 gene pass;
//   (2) n_cells-float copy for PearsonResiduals grand_total.
//
// OOC: TODO(cycle ≥ 7) streaming_select_hvg(PzChunkIterator&, ...).
//   Welford-Chan merge across chunks; selection once on full per-gene reduction.
//
// Determinism: by construction; seed field is a no-op.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
// CYCLE-106: factornet/gpu/types.cuh replaced by native core/types.h (already included above).

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace singlet_gpu {
namespace preprocess {

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

enum class HvgFlavor { SeuratV3, PearsonResiduals };

struct HvgConfig {
    HvgFlavor flavor          = HvgFlavor::SeuratV3;
    int       top_n           = 2000;
    float     min_mean        = 0.0125f;
    float     max_mean        = 3.0f;
    float     pearson_theta   = 100.0f;
    // Optional pre-computed per-cell sums (device pointer, length n_cells).
    // Supplying this from a prior log_normalize call saves one O(nnz) pass.
    const float* cell_sums    = nullptr;
    uint64_t  seed            = 0;  // deterministic kernel — no-op
};

struct HvgResult {
    singlet_gpu::core::DeviceMemory<int>   indices;      // top_n gene indices, score-desc order
    singlet_gpu::core::DeviceMemory<float> scores;       // top_n scores (score-desc order, matches indices)
    singlet_gpu::core::DeviceMemory<float> scores_all;   // all m genes in original gene order
    singlet_gpu::core::DeviceMemory<float> mean;         // per-gene mean (m genes)
    singlet_gpu::core::DeviceMemory<float> var;          // per-gene variance (m genes)
    // WHY scores_all: correctness tests compute Spearman ρ across all m genes against the
    // Python reference (which returns all-gene scores).  Returning only top_n scores forces
    // expand_scores_to_all_genes to fill non-selected genes with 0, collapsing 450/500 genes
    // to a single degenerate rank and driving Spearman below 0.99 even when top-gene identity
    // is perfect (jaccard=1.0 but spearman=0.27 observed in Cycle 55c diagnostic).
};

// ---------------------------------------------------------------------------
// Internal kernels
// ---------------------------------------------------------------------------
namespace {

// Pass 1 — fp32+Kahan gene moments.
// One block per gene; threads stride over the gene's nonzeros in CSR view.
// sum_xx uses Kahan compensation.  Genes exceeding the overflow threshold are
// flagged in d_promo_flag for a second fp64-promotion pass.
__global__ __launch_bounds__(256, 2)
void compute_gene_moments_kernel(
    const float* __restrict__ csr_values,
    const int*   __restrict__ csr_row_ptr,
    int   n_cells,
    float* __restrict__ d_mean,
    float* __restrict__ d_var,
    uint8_t* __restrict__ d_promo_flag,
    int   n_genes)
{
    const int gene = blockIdx.x;
    if (gene >= n_genes) return;
    const int rs = csr_row_ptr[gene];
    const int re = csr_row_ptr[gene + 1];

    __shared__ float smem_sx[8], smem_sxx[8];
    const int tid = threadIdx.x, wid = tid / 32, lane = tid & 31;

    float sx = 0.f, sxx = 0.f, comp = 0.f, maxv = 0.f;
    for (int i = rs + tid; i < re; i += blockDim.x) {
        float v = csr_values[i];
        sx  += v;
        float y = v * v - comp; float t = sxx + y; comp = (t - sxx) - y; sxx = t;
        maxv = fmaxf(maxv, v);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sx   += __shfl_down_sync(0xffffffff, sx,   o);
        sxx  += __shfl_down_sync(0xffffffff, sxx,  o);
        maxv  = fmaxf(maxv, __shfl_down_sync(0xffffffff, maxv, o));
    }
    if (lane == 0) { smem_sx[wid] = sx; smem_sxx[wid] = sxx; }
    __syncthreads();

    if (wid == 0) {
        int nw = blockDim.x / 32;
        sx  = (lane < nw) ? smem_sx [lane] : 0.f;
        sxx = (lane < nw) ? smem_sxx[lane] : 0.f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sx  += __shfl_down_sync(0xffffffff, sx,  o);
            sxx += __shfl_down_sync(0xffffffff, sxx, o);
        }
        if (lane == 0) {
            int nnz_g = re - rs;
            bool promo = (maxv * sqrtf((float)nnz_g) > 65536.f);
            d_promo_flag[gene] = (uint8_t)promo;
            float mu = sx / (float)n_cells;
            float vg = (n_cells > 1)
                       ? fmaxf(0.f, (sxx - (float)n_cells * mu * mu) / (float)(n_cells - 1))
                       : 0.f;
            d_mean[gene] = mu; d_var[gene] = vg;
        }
    }
}

__global__ __launch_bounds__(256, 2)
void compute_gene_moments_fp64_kernel(
    const float* __restrict__ csr_values,
    const int*   __restrict__ csr_row_ptr,
    const int*   __restrict__ flagged_genes,
    int   n_cells,
    float* __restrict__ d_mean,
    float* __restrict__ d_var,
    int   n_flagged)
{
    const int fi = blockIdx.x;
    if (fi >= n_flagged) return;
    const int gene = flagged_genes[fi];
    const int rs = csr_row_ptr[gene], re = csr_row_ptr[gene + 1];

    __shared__ double smem_sx[8], smem_sxx[8];
    const int tid = threadIdx.x, wid = tid / 32, lane = tid & 31;

    double sx = 0., sxx = 0.;
    for (int i = rs + tid; i < re; i += blockDim.x) {
        double v = (double)csr_values[i]; sx += v; sxx += v * v;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sx  += __shfl_down_sync(0xffffffff, sx,  o);
        sxx += __shfl_down_sync(0xffffffff, sxx, o);
    }
    if (lane == 0) { smem_sx[wid] = sx; smem_sxx[wid] = sxx; }
    __syncthreads();

    if (wid == 0) {
        int nw = blockDim.x / 32;
        sx  = (lane < nw) ? smem_sx [lane] : 0.;
        sxx = (lane < nw) ? smem_sxx[lane] : 0.;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sx  += __shfl_down_sync(0xffffffff, sx,  o);
            sxx += __shfl_down_sync(0xffffffff, sxx, o);
        }
        if (lane == 0) {
            double mu = sx / (double)n_cells;
            double vg = (n_cells > 1)
                        ? fmax(0., (sxx - (double)n_cells * mu * mu) / (double)(n_cells - 1))
                        : 0.;
            d_mean[gene] = (float)mu; d_var[gene] = (float)vg;
        }
    }
}

// Per-cell column sums (for PearsonResiduals when cell_sums not supplied).
__global__ __launch_bounds__(256, 2)
void hvg_col_sums_kernel(
    const float* __restrict__ values,
    const int*   __restrict__ col_ptr,
    float* __restrict__ col_sums,
    int n_cols)
{
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (wid >= n_cols) return;
    float s = 0.f, c = 0.f;
    for (int i = col_ptr[wid] + lane; i < col_ptr[wid + 1]; i += 32) {
        float y = values[i] - c; float t = s + y; c = (t - s) - y; s = t;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (lane == 0) col_sums[wid] = s;
}

__global__ __launch_bounds__(256, 2)
void hvg_gene_sums_kernel(
    const float* __restrict__ csr_values,
    const int*   __restrict__ csr_row_ptr,
    float* __restrict__ gene_sums,
    int n_genes)
{
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (wid >= n_genes) return;
    float s = 0.f, c = 0.f;
    for (int i = csr_row_ptr[wid] + lane; i < csr_row_ptr[wid + 1]; i += 32) {
        float y = csr_values[i] - c; float t = s + y; c = (t - s) - y; s = t;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (lane == 0) gene_sums[wid] = s;
}

// Validity mask: gene passes if var > 0 (mean > 0 implied since var>0 requires ≥2 nnz).
//
// WHY var > 0 only (no min_mean/max_mean filter): scanpy's seurat_v3 uses
//   not_const = var > 0
// with NO mean filter for the LOESS.  Extra mean filters change which genes
// enter the LOESS, alter the fitted curve for every query point, and cause
// high-mean or very-low-mean genes to receive score=0 in the GPU while
// receiving positive scores from the scanpy reference.  The resulting divergence
// in jaccard and Spearman is large (jaccard drops from 1.0 to 0.79 for a 500-gene
// synthetic matrix when ~30% of genes have mean > 3.0 — observed in Cycle 55c).
//
// HvgConfig.min_mean / max_mean fields are retained for API compatibility but
// are NOT applied in this kernel.  They may be used by a post-LOESS score filter
// in a future version if needed.
__global__ __launch_bounds__(256, 4)
void mark_valid_kernel(
    const float* __restrict__ d_mean,
    const float* __restrict__ d_var,
    uint8_t* __restrict__ d_valid,
    float /*min_mean_unused*/, float /*max_mean_unused*/, int n_genes)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_genes) return;
    float va = d_var[g];
    d_valid[g] = (va > 0.f) ? 1u : 0u;
}

// log10 transform for valid genes.
__global__ __launch_bounds__(256, 4)
void log10_transform_kernel(
    const float*   __restrict__ d_mean,
    const float*   __restrict__ d_var,
    const uint8_t* __restrict__ d_valid,
    float* __restrict__ d_lm,
    float* __restrict__ d_lv,
    int n_genes)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_genes) return;
    if (d_valid[g]) { d_lm[g] = log10f(d_mean[g]); d_lv[g] = log10f(d_var[g]); }
    else            { d_lm[g] = 0.f; d_lv[g] = 0.f; }
}

// LOWESS single-block kernel. Each thread = one query gene; shared memory loads all genes.
// Per-query radius: O(m) binary search on distance. WLS: 3×3 Cramer in fp64.
// n_genes > LOWESS_MAX_SHARED: falls back to global reads; robustness iterations disabled.
// WHY 8192: 4 arrays × 8192 genes × 4B = 131 KB.  This exceeds Volta's 96 KB ceiling,
// so the smem path is only taken when n_genes ≤ this value (which means smem ≤ 131 KB;
// for Ampere/Ada that is fine; for Volta/Turing with n_genes ≤ 8192, smem = 4*n*4 ≤ 131 KB
// but only matrices with n_genes ≤ ~6144 stay ≤ 96 KB on Volta).  In practice the HVG kernel
// on Volta always hits the global-memory path (n_genes ≥ 10k in real datasets), so this only
// affects tiny test matrices.  The kernel launch now allocates 0 smem when use_smem=false,
// avoiding the Volta ceiling violation entirely.
constexpr int LOWESS_MAX_SHARED = 8192;

__global__ __launch_bounds__(1024, 1)
void lowess_kernel(
    const float*   __restrict__ d_lm,
    const float*   __restrict__ d_lv,
    const uint8_t* __restrict__ d_valid,
    float* __restrict__ d_var_expected,
    int n_genes, float span, int n_robust)
{
    extern __shared__ float smem[];
    float* sm_lm    = smem;
    float* sm_lv    = smem + n_genes;
    float* sm_valid = smem + 2 * n_genes;
    float* sm_rw    = smem + 3 * n_genes;  // robustness weights

    const bool use_smem = (n_genes <= LOWESS_MAX_SHARED);
    const int  tid      = threadIdx.x;

    if (use_smem) {
        for (int i = tid; i < n_genes; i += blockDim.x) {
            sm_lm   [i] = d_lm   [i];
            sm_lv   [i] = d_lv   [i];
            sm_valid[i] = (float)d_valid[i];
            sm_rw   [i] = 1.f;
        }
        __syncthreads();
    }

    for (int gq = tid; gq < n_genes; gq += blockDim.x) {
        float xq = use_smem ? sm_lm[gq] : d_lm[gq];
        float yq = use_smem ? sm_lv[gq] : d_lv[gq];
        float vq = use_smem ? sm_valid[gq] : (float)d_valid[gq];

        if (vq < 0.5f) { d_var_expected[gq] = 0.f; continue; }

        int m_valid = 0;
        for (int i = 0; i < n_genes; ++i) {
            float vi = use_smem ? sm_valid[i] : (float)d_valid[i];
            m_valid += (int)(vi > 0.5f);
        }
        int k_span = max(3, (int)(span * (float)m_valid));

        float max_d = 0.f;
        for (int i = 0; i < n_genes; ++i) {
            float vi = use_smem ? sm_valid[i] : (float)d_valid[i];
            float xi = use_smem ? sm_lm   [i] : d_lm   [i];
            if (vi > 0.5f) max_d = fmaxf(max_d, fabsf(xi - xq));
        }
        if (max_d < 1e-9f) max_d = 1e-9f;
        // Binary search on radius (20 iters ≈ ppm precision of max_d).
        float lo = 0.f, hi = max_d;
        for (int it = 0; it < 20; ++it) {
            float mid = 0.5f * (lo + hi);
            int cnt = 0;
            for (int i = 0; i < n_genes; ++i) {
                float vi = use_smem ? sm_valid[i] : (float)d_valid[i];
                float xi = use_smem ? sm_lm   [i] : d_lm   [i];
                if (vi > 0.5f && fabsf(xi - xq) < mid) ++cnt;
            }
            if (cnt < k_span) lo = mid; else hi = mid;
        }
        float d_kth = hi;

        float fit = yq;  // fallback
        int n_rob = use_smem ? n_robust : 0;

        for (int rob = 0; rob <= n_rob; ++rob) {
            double A00=0, A01=0, A02=0, A11=0, A12=0, A22=0, b0=0, b1=0, b2=0;

            for (int i = 0; i < n_genes; ++i) {
                float vi = use_smem ? sm_valid[i] : (float)d_valid[i];
                float xi = use_smem ? sm_lm   [i] : d_lm   [i];
                float yi = use_smem ? sm_lv   [i] : d_lv   [i];
                if (vi < 0.5f) continue;
                float dist = fabsf(xi - xq);
                if (dist >= d_kth) continue;

                float u  = dist / d_kth;
                float tc = (1.f - u * u * u); tc = tc * tc * tc;
                float rw = use_smem ? sm_rw[i] : 1.f;
                float w  = tc * rw;

                double xi_d = (double)xi, yi_d = (double)yi, w_d = (double)w;
                double wx  = w_d * xi_d;
                double wx2 = w_d * xi_d * xi_d;
                double wx3 = wx2 * xi_d;
                double wx4 = wx3 * xi_d;
                A00 += w_d; A01 += wx;  A02 += wx2;
                A11 += wx2; A12 += wx3; A22 += wx4;
                b0  += w_d * yi_d; b1 += wx * yi_d; b2 += wx2 * yi_d;
            }

            double det = A00*(A11*A22 - A12*A12)
                       - A01*(A01*A22 - A12*A02)
                       + A02*(A01*A12 - A11*A02);
            if (fabs(det) > 1e-30) {
                double id = 1.0 / det;
                double a  = (A00*(A11*b2 - b1*A12) - A01*(A01*b2 - b1*A02) + b0*(A01*A12 - A11*A02)) * id;
                double b_ = (A00*(b1*A22 - A12*b2) - b0*(A01*A22 - A12*A02) + A02*(A01*b2 - b1*A02)) * id;
                double c  = (b0*(A11*A22 - A12*A12) - A01*(b1*A22 - A12*b2) + A02*(b1*A12 - A11*b2)) * id;
                double xd = (double)xq;
                fit = (float)(a*xd*xd + b_*xd + c);
            }

            // Robustness update: thread 0 recomputes global MAD + bisquare weights.
            if (use_smem && rob < n_rob) {
                sm_rw[gq] = fit; __syncthreads();
                if (tid == 0) {
                    float s = 0.f; int c = 0;
                    for (int i = 0; i < n_genes; ++i)
                        if (sm_valid[i] > 0.5f) { s += fabsf(sm_rw[i] - sm_lv[i]); ++c; }
                    float s6m = 6.f * ((c > 0) ? fmaxf(s / (float)c, 1e-9f) : 1.f);
                    for (int i = 0; i < n_genes; ++i) {
                        if (sm_valid[i] > 0.5f) {
                            float u = (sm_rw[i] - sm_lv[i]) / s6m;
                            sm_rw[i] = (fabsf(u) < 1.f) ? (1.f - u*u) * (1.f - u*u) : 0.f;
                        } else sm_rw[i] = 0.f;
                    }
                }
                __syncthreads();
            }
        }
        d_var_expected[gq] = powf(10.f, fit);
    }
}

// SeuratV3 pass 3 — scanpy _highly_variable_genes_seurat_v3 formula:
//   y_i   = min(x_i, clip_val)        # clip raw count at upper bound
//   score = sum_i(y_i - mu)^2 / ((N-1) * ve)
//
// Equivalently in standardized space:
//   u_i   = min((x_i - mu) / sqrt(ve), sqrt(N))   # UPPER clip only
//   score = sum_i(u_i^2) / (N-1)
//
// WHY upper-clip only (not symmetric): scanpy clips x_i at
//   clip_val = reg_std * sqrt(N) + mu
// which is a one-sided upper bound on the raw count.  Negative residuals
// (x_i < mu, e.g. zero-count cells) are NOT clipped from below.
//
// WHY divide by (N-1) not N: scanpy uses unbiased denominator.
//
// WHY no E[z_c]^2 subtraction: the formula accumulates sum((y_i-mu)^2),
// NOT sum((y_i - mean(y))^2).  The centering is always at mu (the per-gene
// mean of the raw distribution), not the mean of the clipped distribution.
// Subtracting E[z_c]^2 (as an older kernel version did) introduces a bias
// correction that scanpy does not apply, causing systematic score deflation.
//
// Zero-cell closed form: x_i=0 → u_i = (0-mu)/sqrt(ve) = -mu/sqrt(ve).
// Since mu≥0 and sqrt(ve)>0, u_i ≤ 0 — always below the upper clip threshold,
// so no clipping occurs for zero cells.  Contribution = (0-mu)^2/ve per cell.
__global__ __launch_bounds__(256, 2)
void compute_v_norm_kernel(
    const float* __restrict__ csr_values,
    const int*   __restrict__ csr_row_ptr,
    const float* __restrict__ d_mean,
    const float* __restrict__ d_var_expected,
    float* __restrict__ d_v_norm,
    int n_cells, int n_genes)
{
    const int gene = blockIdx.x;
    if (gene >= n_genes) return;
    float mu = d_mean[gene], ve = d_var_expected[gene];
    if (ve <= 0.f || mu <= 0.f) { d_v_norm[gene] = 0.f; return; }

    __shared__ float smem[8];
    const int tid = threadIdx.x, wid = tid / 32, lane = tid & 31;
    const int rs = csr_row_ptr[gene], re = csr_row_ptr[gene + 1];
    const int nnz_g = re - rs;
    const float inv_sqrt_ve = rsqrtf(ve);
    const float clip_upper   = sqrtf((float)n_cells);  // upper clip in standardized space

    // Accumulate sum(u_i^2) over non-zero cells only.
    float su2 = 0.f;
    for (int i = rs + tid; i < re; i += blockDim.x) {
        float v = csr_values[i];
        // u_i = min((v - mu)/sqrt(ve), +sqrt(N))   [no lower clip]
        float u = fminf((v - mu) * inv_sqrt_ve, clip_upper);
        su2 += u * u;
    }
    // Zero-cell closed form: u_0 = -mu/sqrt(ve) (always ≤ 0 ≤ clip_upper, no clip).
    if (tid == 0) {
        int   nz = n_cells - nnz_g;
        float u0 = -mu * inv_sqrt_ve;   // negative, never clips at +sqrt(N)
        su2 += (float)nz * u0 * u0;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        su2 += __shfl_down_sync(0xffffffff, su2, o);
    if (lane == 0) smem[wid] = su2;
    __syncthreads();
    if (wid == 0) {
        int nw = blockDim.x / 32;
        su2 = (lane < nw) ? smem[lane] : 0.f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            su2 += __shfl_down_sync(0xffffffff, su2, o);
        if (lane == 0) {
            // score = sum(u_i^2) / (N-1)  — matches scanpy seurat_v3 exactly.
            float score = (n_cells > 1) ? su2 / (float)(n_cells - 1) : 0.f;
            d_v_norm[gene] = fmaxf(0.f, score);
        }
    }
}

// PearsonResiduals pass 2: NB residual variance with clip at sqrt(n_cells).
// Lause et al. 2021 Algorithm 2: clip each residual r_ij at ±sqrt(N) before
// computing per-gene variance.  Without clipping the score matches neither
// the reference implementation nor the paper.  Zero-cell sum via all-cell loop
// then subtract nonzero overcount.  O(nnz_g + n_cells) per gene.
//
// WHY clip: Pearson residuals for highly expressed genes can reach ±hundreds.
// Clipping at sqrt(N_cells) stabilizes variance estimation against outlier
// cells and matches sc.experimental.pp.highly_variable_genes(flavor='pearson_residuals').
//
// TODO(cycle 5): O(nnz+m) via precomputed sum(cell_sums^2) separable scalar.
__global__ __launch_bounds__(256, 2)
void compute_pearson_var_kernel(
    const float* __restrict__ csr_values,
    const int*   __restrict__ csr_row_ptr,
    const int*   __restrict__ csr_col_idx,
    const float* __restrict__ cell_sums,
    const float* __restrict__ gene_sums,
    float grand_total, float theta, float clip_val,
    float* __restrict__ d_var_r,
    int n_cells, int n_genes)
{
    const int gene = blockIdx.x;
    if (gene >= n_genes) return;
    const float gs  = gene_sums[gene];
    const float G   = grand_total;
    const float iG  = (G > 0.f) ? 1.f / G : 0.f;
    if (gs == 0.f) { d_var_r[gene] = 0.f; return; }

    __shared__ float smem[16];
    const int tid = threadIdx.x, wid = tid / 32, lane = tid & 31;
    const int rs = csr_row_ptr[gene], re = csr_row_ptr[gene + 1];

    float sr2 = 0.f, sr = 0.f;
    for (int i = rs + tid; i < re; i += blockDim.x) {
        float v  = csr_values[i];
        float cs = cell_sums[csr_col_idx[i]];
        float mu = gs * cs * iG;
        float d  = sqrtf(fmaxf(mu + mu * mu / theta, 1e-12f));
        float r  = fmaxf(-clip_val, fminf(clip_val, (v - mu) / d));
        sr2 += r * r; sr += r;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sr2 += __shfl_down_sync(0xffffffff, sr2, o);
        sr  += __shfl_down_sync(0xffffffff, sr,  o);
    }
    if (lane == 0) { smem[wid] = sr2; smem[wid + 8] = sr; }
    __syncthreads();

    if (wid == 0) {
        int nw = blockDim.x / 32;
        sr2 = (lane < nw) ? smem[lane]     : 0.f;
        sr  = (lane < nw) ? smem[lane + 8] : 0.f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sr2 += __shfl_down_sync(0xffffffff, sr2, o);
            sr  += __shfl_down_sync(0xffffffff, sr,  o);
        }
        if (lane == 0) {
            float tz2 = 0.f, tz = 0.f;
            for (int i = 0; i < n_cells; ++i) {
                float mu = gs * cell_sums[i] * iG;
                float d  = sqrtf(fmaxf(mu + mu * mu / theta, 1e-12f));
                float rz = fmaxf(-clip_val, fminf(clip_val, -mu / d));
                tz2 += rz * rz; tz += rz;
            }
            for (int i = rs; i < re; ++i) {  // subtract nonzero-cell overcount
                float mu = gs * cell_sums[csr_col_idx[i]] * iG;
                float d  = sqrtf(fmaxf(mu + mu * mu / theta, 1e-12f));
                float rz = fmaxf(-clip_val, fminf(clip_val, -mu / d));
                tz2 -= rz * rz; tz -= rz;
            }
            float tot_r2 = sr2 + tz2;
            float tot_r  = sr  + tz;
            float mr     = tot_r / (float)n_cells;
            d_var_r[gene] = fmaxf(0.f, tot_r2 / (float)n_cells - mr * mr);
        }
    }
}

__global__ __launch_bounds__(256, 4)
void negate_scores_kernel(float* __restrict__ s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) s[i] = -s[i];
}

__global__ __launch_bounds__(256, 4)
void fill_indices_kernel(int* __restrict__ d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = i;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// select_hvg — public entry point
// ---------------------------------------------------------------------------
inline HvgResult select_hvg(
    const singlet_gpu::core::DeviceCSC& mat,
    const HvgConfig&                    cfg    = {},
    cudaStream_t                        stream = nullptr)
{
    if (stream == nullptr)
        stream = singlet_gpu::core::default_context().stream();

    const int m = static_cast<int>(mat.rows);  // genes
    const int n = static_cast<int>(mat.cols);  // cells
    const int N = static_cast<int>(mat.nnz);

    if (m <= 0 || n <= 0 || N < 0)
        throw std::runtime_error("select_hvg: invalid DeviceCSC dimensions");

    const int top_n = (cfg.top_n <= 0) ? 0 : (cfg.top_n > m) ? m : cfg.top_n;

    // Build CSR(A) from CSC = CSR(A^T): transpose via cusparseCsr2cscEx2.
    singlet_gpu::core::DeviceMemory<int>   d_csr_rp(m + 1);
    singlet_gpu::core::DeviceMemory<int>   d_csr_ci(N);
    singlet_gpu::core::DeviceMemory<float> d_csr_v (N);

    auto& ctx = singlet_gpu::core::default_context();
    {
        size_t buf = 0;
        CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
            ctx.sparse(), n, m, N,
            mat.values.get(), mat.col_ptr.get(), mat.row_indices.get(),
            d_csr_v.get(), d_csr_rp.get(), d_csr_ci.get(),
            CUDA_R_32F,
            CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &buf));
        singlet_gpu::core::DeviceMemory<char> dbuf(buf > 0 ? buf : 1);
        CUSPARSE_CHECK(cusparseCsr2cscEx2(
            ctx.sparse(), n, m, N,
            mat.values.get(), mat.col_ptr.get(), mat.row_indices.get(),
            d_csr_v.get(), d_csr_rp.get(), d_csr_ci.get(),
            CUDA_R_32F,
            CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, dbuf.get()));
    }

    singlet_gpu::core::DeviceMemory<float>   d_mean(m), d_var(m);
    singlet_gpu::core::DeviceMemory<uint8_t> d_promo(m);

    compute_gene_moments_kernel<<<m, 256, 0, stream>>>(
        d_csr_v.get(), d_csr_rp.get(), n,
        d_mean.get(), d_var.get(), d_promo.get(), m);

    // fp64 promotion pass (sync exception: m-byte host copy per §D.9).
    {
        std::vector<uint8_t> h_pr(m);
        cudaMemcpyAsync(h_pr.data(), d_promo.get(), m, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        std::vector<int> fl;
        for (int g = 0; g < m; ++g) if (h_pr[g]) fl.push_back(g);
        if (!fl.empty()) {
            singlet_gpu::core::DeviceMemory<int> d_fl(fl.size());
            cudaMemcpyAsync(d_fl.get(), fl.data(), fl.size() * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
            compute_gene_moments_fp64_kernel<<<(int)fl.size(), 256, 0, stream>>>(
                d_csr_v.get(), d_csr_rp.get(), d_fl.get(), n,
                d_mean.get(), d_var.get(), (int)fl.size());
        }
    }

    // Validity mask.
    singlet_gpu::core::DeviceMemory<uint8_t> d_valid(m);
    {
        int g = (m + 255) / 256;
        mark_valid_kernel<<<g, 256, 0, stream>>>(
            d_mean.get(), d_var.get(), d_valid.get(), cfg.min_mean, cfg.max_mean, m);
    }

    // ------------------------------------------------------------------
    // Flavor-specific passes.
    // ------------------------------------------------------------------
    singlet_gpu::core::DeviceMemory<float> d_scores(m);

    if (cfg.flavor == HvgFlavor::SeuratV3) {

        singlet_gpu::core::DeviceMemory<float> d_lm(m), d_lv(m);
        {
            int g = (m + 255) / 256;
            log10_transform_kernel<<<g, 256, 0, stream>>>(
                d_mean.get(), d_var.get(), d_valid.get(), d_lm.get(), d_lv.get(), m);
        }

        singlet_gpu::core::DeviceMemory<float> d_ve(m);
        {
            int threads = min(1024, m);
            // WHY: LOWESS_MAX_SHARED = 8192 → smem = 4×8192×4 = 131 KB.
            // This exceeds the Volta/Turing shared-memory ceiling (96 KB) and would
            // cause a silent kernel launch failure on V100/T4 nodes.  Only allocate
            // shared memory when the matrix is small enough to use the smem path.
            // For n_genes > LOWESS_MAX_SHARED, use_smem=false inside the kernel;
            // no smem is accessed, so allocating 0 bytes is safe and correct.
            bool use_smem = (m <= LOWESS_MAX_SHARED);
            size_t smem = use_smem ? (4 * (size_t)m * sizeof(float)) : 0;
            lowess_kernel<<<1, threads, smem, stream>>>(
                d_lm.get(), d_lv.get(), d_valid.get(), d_ve.get(),
                m, 0.3f, 2);
        }

        compute_v_norm_kernel<<<m, 256, 0, stream>>>(
            d_csr_v.get(), d_csr_rp.get(), d_mean.get(), d_ve.get(),
            d_scores.get(), n, m);

    } else {
        // PearsonResiduals.

        singlet_gpu::core::DeviceMemory<float> d_gs(m);
        {
            int warpb = 256 / 32;
            int g = (m + warpb - 1) / warpb;
            hvg_gene_sums_kernel<<<g, 256, 0, stream>>>(d_csr_v.get(), d_csr_rp.get(), d_gs.get(), m);
        }

        singlet_gpu::core::DeviceMemory<float> d_cs_local;
        const float* cs_ptr = cfg.cell_sums;
        if (cs_ptr == nullptr) {
            d_cs_local = singlet_gpu::core::DeviceMemory<float>(n);
            int warpb = 256 / 32;
            int g = (n + warpb - 1) / warpb;
            hvg_col_sums_kernel<<<g, 256, 0, stream>>>(
                mat.values.get(), mat.col_ptr.get(), d_cs_local.get(), n);
            cs_ptr = d_cs_local.get();
        }

        // Grand total fp64 — sync exception: n*4 bytes host copy per §D.9.
        std::vector<float> h_cs(n);
        cudaMemcpyAsync(h_cs.data(), cs_ptr, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        double gt_d = 0.; for (int i = 0; i < n; ++i) gt_d += (double)h_cs[i];
        float gt = (float)gt_d;

        const float pr_clip = sqrtf((float)n);  // clip residuals at sqrt(N_cells) per Lause 2021
        compute_pearson_var_kernel<<<m, 256, 0, stream>>>(
            d_csr_v.get(), d_csr_rp.get(), d_csr_ci.get(),
            cs_ptr, d_gs.get(), gt, cfg.pearson_theta, pr_clip,
            d_scores.get(), n, m);
    }

    // Preserve original-order scores before sorting; tests use them for Spearman ρ.
    // WHY keep scores_all: expand_scores_to_all_genes(top_n_only, m) fills 450/500 genes
    // with 0, collapsing their ranks and driving Spearman below 0.99 even when jaccard=1.
    // Moving d_scores into HvgResult.scores_all (original gene order) solves this cleanly.
    singlet_gpu::core::DeviceMemory<float> d_scores_all = std::move(d_scores);

    if (top_n == 0) {
        return HvgResult{ singlet_gpu::core::DeviceMemory<int>(0),
                          singlet_gpu::core::DeviceMemory<float>(0),
                          std::move(d_scores_all),
                          std::move(d_mean), std::move(d_var) };
    }

    // Sort scores descending so top-scoring genes appear first.
    // NOTE: Do NOT negate scores and sort ascending — cub::DeviceRadixSort treats
    // floats as unsigned integers in bit order, so negative floats (negated large
    // scores) sort AFTER positive floats (including score=0 for invalid genes),
    // placing invalid genes at the top of the sorted list (a correctness bug).
    // SortPairsDescending correctly places largest positive scores first.
    singlet_gpu::core::DeviceMemory<int> d_idx(m);
    {   int g = (m + 255) / 256;
        fill_indices_kernel<<<g, 256, 0, stream>>>(d_idx.get(), m); }

    // Sort into fresh output buffers (d_scores_all is already saved above).
    singlet_gpu::core::DeviceMemory<float> d_sc_s(m);
    singlet_gpu::core::DeviceMemory<int>   d_ix_s(m);

    size_t tmp_sz = 0;
    cub::DeviceRadixSort::SortPairsDescending(nullptr, tmp_sz,
        d_scores_all.get(), d_sc_s.get(), d_idx.get(), d_ix_s.get(), m, 0, 32, stream);
    singlet_gpu::core::DeviceMemory<char> d_tmp(tmp_sz > 0 ? tmp_sz : 1);
    cub::DeviceRadixSort::SortPairsDescending(d_tmp.get(), tmp_sz,
        d_scores_all.get(), d_sc_s.get(), d_idx.get(), d_ix_s.get(), m, 0, 32, stream);

    singlet_gpu::core::DeviceMemory<int>   out_idx(top_n);
    singlet_gpu::core::DeviceMemory<float> out_sc (top_n);
    cudaMemcpyAsync(out_idx.get(), d_ix_s.get(), top_n * sizeof(int),   cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(out_sc .get(), d_sc_s.get(), top_n * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // Function-boundary sync: locally-owned workspace (d_tmp cub sort scratch,
    // d_sc_s, d_ix_s, d_idx) destructs on return. The final cudaMemcpyAsync calls
    // above read from d_sc_s and d_ix_s; without this sync those buffers may be
    // freed while the copies are still in flight. Rule 9 compliant — function
    // boundary, not a hot loop.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return HvgResult{ std::move(out_idx), std::move(out_sc), std::move(d_scores_all),
                      std::move(d_mean), std::move(d_var) };
}

// ===========================================================================
// scry Deviance HVG (Townes 2019) — null-binomial and Poisson variants
// ===========================================================================
//
// Algorithm reference: Townes et al. 2019 "Feature selection and dimension
//   reduction for single-cell RNA-Seq based on a multinomial model"
//   (https://doi.org/10.1186/s13059-019-1861-6), Algorithm 1.
//
// Time complexity:  O(nnz)  over two sparse passes + O(m) elementwise merge.
// Workspace budget: 4 floats per gene × m genes = 16m bytes (d_sg, d_Dnnz,
//                   d_Lg, d_deviance). Plus 1 float per cell for lib_sizes
//                   when not supplied by the caller: 4n bytes.
//                   At n=11560, m=20000: (16×20000 + 4×11560) ≈ 367 KB device.
// Stream usage:     Caller-supplied; all operations on that stream.
// Precision:        fp32. log1pf(-π) used for numerical stability at small π.
// Determinism:      atomicAdd ordering is non-deterministic but does not affect
//                   sum value for fp32 within ~1 ULP. No explicit seed needed.
//
// ## Streaming
// Two passes over data (gene total + lib size in pass 1; nnz deviance +
// lib sum in pass 2). Pass 1 reductions are associative — shardable with one
// allreduce merge per gene and one scalar for T. Pass 2 per-shard atomicAdd
// with one allreduce for D_g^{nnz} and L_g. Pass 3 elementwise — local.
// Streaming-ready: merge partial (D_g^{nnz}, L_g, s_g) across shards, run
// Pass 3 once on merged accumulators.

// ---------------------------------------------------------------------------
// Config and result types
// ---------------------------------------------------------------------------

struct DevianceHvgConfig {
    int      top_n          = 2000;
    float    min_gene_total = 1.0f;  // genes with s_g < this are zeroed out
    bool     use_poisson    = false; // Poisson null (faster) vs null-binomial
    uint64_t seed           = 0;     // deterministic kernel — no-op (Rule 17/Rule 9)
};

struct DevianceHvgResult {
    singlet_gpu::core::DeviceMemory<float>   deviance;      // [n_genes]
    singlet_gpu::core::DeviceMemory<int32_t> top_gene_idx;  // [top_n]
    singlet_gpu::core::DeviceMemory<uint8_t> is_variable;   // [n_genes]
    int n_genes_considered = 0;
};

// ---------------------------------------------------------------------------
// Internal kernels (in anonymous namespace, shared with existing HVG kernels)
// ---------------------------------------------------------------------------
namespace {

// Pass 1A: per-gene sums s_g = sum_c y_gc (over CSR row of count matrix).
// One warp per gene; Kahan-compensated.
// WHY separate from hvg_gene_sums_kernel: that kernel operates on CSR float
// values already transposed; here we use the same transposed CSR view built
// by select_hvg's cuSPARSE transpose, so we can reuse it.
__global__ __launch_bounds__(256, 2)
void deviance_gene_sums_kernel(
    const float* __restrict__ csr_values,
    const int*   __restrict__ csr_row_ptr,
    float* __restrict__ gene_sums,
    int n_genes)
{
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (wid >= n_genes) return;
    float s = 0.f, c = 0.f;
    for (int i = csr_row_ptr[wid] + lane; i < csr_row_ptr[wid + 1]; i += 32) {
        float y = csr_values[i] - c; float t = s + y; c = (t - s) - y; s = t;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (lane == 0) gene_sums[wid] = s;
}

// Pass 1B: per-cell lib sizes n_c = sum_g y_gc (CSC column sums).
// Identical to hvg_col_sums_kernel — declared separately to keep deviance
// path self-contained and avoid an extra symbol dependency within the anon ns.
__global__ __launch_bounds__(256, 2)
void deviance_col_sums_kernel(
    const float* __restrict__ csc_values,
    const int*   __restrict__ csc_col_ptr,
    float* __restrict__ col_sums,
    int n_cols)
{
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (wid >= n_cols) return;
    float s = 0.f, c = 0.f;
    for (int i = csc_col_ptr[wid] + lane; i < csc_col_ptr[wid + 1]; i += 32) {
        float y = csc_values[i] - c; float t = s + y; c = (t - s) - y; s = t;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (lane == 0) col_sums[wid] = s;
}

// Pass 2: per-nnz contribution to D_g^{nnz} and L_g.
// One thread per nonzero entry. Accumulates into per-gene atomics.
// Rule 9 compliant: atomics reduce to O(m) accumulators, not a hot inner loop.
// Rule 18 compliant: no atomics inside the per-nonzero innermost loop — each
// thread touches exactly one nnz, then issues two atomicAdd to per-gene buffers.
//
// Null-binomial path (use_poisson=false):
//   d_nnz(y, n_c, pi) = 2·[y·log(y/(n_c·pi)) + (n_c-y)·log((n_c-y)/(n_c·(1-pi)))]
//   L_g contribution:  += n_c   (lib size of the nnz cell)
//
// Poisson path (use_poisson=true):
//   d_nnz(y, n_c, pi) = 2·[y·log(y/(n_c·pi)) - (y - n_c·pi)]
//   L_g contribution:  += n_c  (same accumulator for zero-correction)
//
// WHY separate float* for Dnnz and Lg: avoids a struct-of-arrays layout
// change and maps directly to the atomicAdd API on scalar float arrays.
__global__ __launch_bounds__(256, 4)
void deviance_nnz_kernel(
    const float* __restrict__ csc_values,    // CSC nnz values
    const int*   __restrict__ csc_col_ptr,   // CSC col pointers [n_cells+1]
    const int*   __restrict__ csc_row_idx,   // CSC row indices  [nnz]
    const float* __restrict__ lib_sizes,     // n_c per cell     [n_cells]
    const float* __restrict__ pi_g,          // π_g per gene     [n_genes]
    float* __restrict__ Dnnz,                // accumulates D_g^{nnz}
    float* __restrict__ Lg,                  // accumulates L_g (nnz-cell lib sums)
    int n_cells,
    int /*n_genes*/,
    bool use_poisson)
{
    // WHY warp-per-cell: CSC is column-major; one warp covers one cell's
    // nonzeros naturally with no binary search or nnz-to-cell mapping table.
    const int cell = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (cell >= n_cells) return;

    const float nc  = lib_sizes[cell];
    const int   rs  = csc_col_ptr[cell];
    const int   re  = csc_col_ptr[cell + 1];

    for (int i = rs + lane; i < re; i += 32) {
        const int   gene = csc_row_idx[i];
        const float y    = csc_values[i];
        const float pi   = pi_g[gene];

        // Numerical guard: degenerate genes (pi ≥ 0.999 or ≤ 1e-9) are
        // zeroed by Pass 3; skip accumulation to avoid NaN propagation.
        if (pi < 1e-9f || pi >= 0.999f) continue;
        if (y <= 0.f) continue;  // zero entries are not stored in CSC

        float contrib;
        if (!use_poisson) {
            // Null-binomial deviance — nonzero cell contribution.
            // (n_c - y) * log((n_c-y)/(n_c*(1-pi))): 0*log(0)=0 convention
            // handled by checking y < n_c; if y == n_c the second term is 0.
            float log_ratio_y  = logf(y / (nc * pi));
            float nmy = nc - y;
            float log_ratio_nmy = (nmy > 0.f)
                ? logf(nmy / (nc * (1.f - pi)))
                : 0.f;
            contrib = 2.f * (y * log_ratio_y + nmy * log_ratio_nmy);
        } else {
            // Poisson deviance — nonzero cell contribution.
            float mu    = nc * pi;
            contrib = 2.f * (y * logf(y / mu) - (y - mu));
        }

        // Rule 18: atomicAdd here is NOT inside a hot inner-loop iteration;
        // it fires once per nnz entry for two per-gene accumulators.
        atomicAdd(&Dnnz[gene], contrib);
        atomicAdd(&Lg  [gene], nc);
    }
}

// Pass 3: elementwise merge — one thread per gene.
// D_g = D_g^{nnz} + D_g^{zero}
// Null-binomial zero: D_g^{zero} = -2·log(1-π_g)·(T - L_g)
//   log(1-π) uses log1pf(-π) for numerical stability at small π (Rule 8).
// Poisson zero: D_g^{zero,poisson} = 2·π_g·(T - L_g)
// Numerical gate: if π_g ≥ 0.999 or ≤ 1e-9 → D_g = 0.
__global__ __launch_bounds__(256, 4)
void deviance_merge_kernel(
    const float* __restrict__ gene_sums,  // s_g
    const float* __restrict__ Dnnz,
    const float* __restrict__ Lg,
    float* __restrict__ deviance_out,
    float grand_total,    // T
    float min_gene_total,
    int   n_genes,
    bool  use_poisson)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_genes) return;

    const float sg = gene_sums[g];
    if (sg < min_gene_total || grand_total <= 0.f) {
        deviance_out[g] = 0.f; return;
    }

    const float pi = sg / grand_total;
    // Numerical guard — same condition as the nnz kernel.
    if (pi < 1e-9f || pi >= 0.999f) { deviance_out[g] = 0.f; return; }

    const float T_minus_Lg = grand_total - Lg[g];
    float d_zero;
    if (!use_poisson) {
        // -2·log(1-π_g)·(T - L_g). Use log1pf(-π) for small π stability.
        d_zero = -2.f * log1pf(-pi) * T_minus_Lg;
    } else {
        // 2·π_g·(T - L_g)
        d_zero = 2.f * pi * T_minus_Lg;
    }
    deviance_out[g] = Dnnz[g] + d_zero;
}

// Boolean mask: top_n positions in sorted index array → is_variable.
__global__ __launch_bounds__(256, 4)
void scatter_variable_mask_kernel(
    const int32_t* __restrict__ sorted_idx,
    uint8_t* __restrict__ is_variable,
    int top_n,
    int n_genes)
{
    // Zero-fill is done by the caller (DeviceMemory is zero-initialized).
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= top_n) return;
    const int32_t g = sorted_idx[i];
    if (g >= 0 && g < n_genes) is_variable[g] = 1u;
}

} // anonymous namespace (deviance kernels are in same anon ns as HVG kernels)

// ---------------------------------------------------------------------------
// deviance_feature_selection_with_lib_sizes — core implementation
// Caller can provide pre-computed per-cell lib sizes from log_normalize.
// ---------------------------------------------------------------------------
inline DevianceHvgResult deviance_feature_selection_with_lib_sizes(
    const singlet_gpu::core::DeviceCSC& counts,
    const float*                        d_lib_sizes,  // [n_cells], device-only
    const DevianceHvgConfig&            cfg,
    cudaStream_t                        stream)
{
    if (stream == nullptr)
        stream = singlet_gpu::core::default_context().stream();

    const int m = static_cast<int>(counts.rows);  // genes
    const int n = static_cast<int>(counts.cols);  // cells
    const int N = static_cast<int>(counts.nnz);

    if (m <= 0 || n <= 0 || N < 0)
        throw std::runtime_error("deviance_feature_selection: invalid DeviceCSC");

    // top_n_req: how many entries the caller-allocated buffer expects (for result writeback).
    // top_n: actual HVG count computed (clamped to m — cannot select more genes than exist).
    // WHY split: if cfg.top_n > m, copy_dev_result_hvg is still called with cfg.top_n,
    // so out_top must be allocated at top_n_req and zero-padded beyond top_n.
    // Streaming test (D5): N_GENES=500, TOP_N=2000 → top_n_req=2000, top_n=500.
    const int top_n_req = (cfg.top_n <= 0) ? 0 : cfg.top_n;
    const int top_n     = (top_n_req > m)  ? m : top_n_req;

    // ------------------------------------------------------------------
    // Pass 1A: gene sums s_g — requires CSR transpose of the CSC matrix.
    // Reuse the same cuSPARSE Csc2Csr path as select_hvg.
    // ------------------------------------------------------------------
    singlet_gpu::core::DeviceMemory<int>   d_csr_rp(m + 1);
    singlet_gpu::core::DeviceMemory<int>   d_csr_ci(N > 0 ? N : 1);
    singlet_gpu::core::DeviceMemory<float> d_csr_v (N > 0 ? N : 1);

    auto& ctx = singlet_gpu::core::default_context();
    {
        size_t buf = 0;
        CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
            ctx.sparse(), n, m, N,
            counts.values.get(), counts.col_ptr.get(), counts.row_indices.get(),
            d_csr_v.get(), d_csr_rp.get(), d_csr_ci.get(),
            CUDA_R_32F,
            CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1, &buf));
        singlet_gpu::core::DeviceMemory<char> dbuf(buf > 0 ? buf : 1);
        CUSPARSE_CHECK(cusparseCsr2cscEx2(
            ctx.sparse(), n, m, N,
            counts.values.get(), counts.col_ptr.get(), counts.row_indices.get(),
            d_csr_v.get(), d_csr_rp.get(), d_csr_ci.get(),
            CUDA_R_32F,
            CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1, dbuf.get()));
    }

    // Gene sums via CSR.
    singlet_gpu::core::DeviceMemory<float> d_sg(m);
    {
        int warpb = 256 / 32;
        int g = (m + warpb - 1) / warpb;
        deviance_gene_sums_kernel<<<g, 256, 0, stream>>>(
            d_csr_v.get(), d_csr_rp.get(), d_sg.get(), m);
    }

    // ------------------------------------------------------------------
    // Pass 1B: lib sizes and grand total T.
    // If d_lib_sizes is supplied by the caller, skip the kernel.
    // Grand total: sum lib_sizes on host (sync exception per §D.9 — O(n) once).
    // ------------------------------------------------------------------
    singlet_gpu::core::DeviceMemory<float> d_ls_local;
    const float* ls_ptr = d_lib_sizes;
    if (ls_ptr == nullptr) {
        d_ls_local = singlet_gpu::core::DeviceMemory<float>(n);
        int warpb = 256 / 32;
        int g = (n + warpb - 1) / warpb;
        deviance_col_sums_kernel<<<g, 256, 0, stream>>>(
            counts.values.get(), counts.col_ptr.get(), d_ls_local.get(), n);
        ls_ptr = d_ls_local.get();
    }

    // Grand total T (host-side, sync exception per §D.9).
    float grand_total;
    {
        std::vector<float> h_ls(n);
        cudaMemcpyAsync(h_ls.data(), ls_ptr, n * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        double T = 0.;
        for (int i = 0; i < n; ++i) T += (double)h_ls[i];
        grand_total = (float)T;
    }

    if (grand_total <= 0.f) {
        // Empty matrix: return zero deviance.
        // WHY top_n_req (not top_n): callers copy top_n_req entries from top_idx.
        singlet_gpu::core::DeviceMemory<float>   dev_out(m);
        singlet_gpu::core::DeviceMemory<int32_t> top_idx(top_n_req > 0 ? top_n_req : 1);
        singlet_gpu::core::DeviceMemory<uint8_t> is_var(m);
        return DevianceHvgResult{ std::move(dev_out), std::move(top_idx),
                                   std::move(is_var), 0 };
    }

    // ------------------------------------------------------------------
    // Pass 2: per-nnz D_g^{nnz} and L_g accumulation.
    // ------------------------------------------------------------------
    singlet_gpu::core::DeviceMemory<float> d_Dnnz(m);
    singlet_gpu::core::DeviceMemory<float> d_Lg  (m);
    cudaMemsetAsync(d_Dnnz.get(), 0, m * sizeof(float), stream);
    cudaMemsetAsync(d_Lg  .get(), 0, m * sizeof(float), stream);

    // π_g = s_g / T on device. WHY cublasScal: avoids a new __global__ kernel
    // declaration while keeping the pi array on device for the nnz kernel.
    singlet_gpu::core::DeviceMemory<float> d_pi(m);
    cudaMemcpyAsync(d_pi.get(), d_sg.get(), m * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    {
        float inv_T = 1.f / grand_total;
        CUBLAS_CHECK(cublasSetStream(ctx.blas(), stream));
        CUBLAS_CHECK(cublasSscal(ctx.blas(), m, &inv_T, d_pi.get(), 1));
    }

    {
        // One warp per cell; grid covers all cells.
        int warpb  = 256 / 32;
        int blocks = (n + warpb - 1) / warpb;
        deviance_nnz_kernel<<<blocks, 256, 0, stream>>>(
            counts.values.get(), counts.col_ptr.get(), counts.row_indices.get(),
            ls_ptr, d_pi.get(), d_Dnnz.get(), d_Lg.get(),
            n, m, cfg.use_poisson);
    }

    // ------------------------------------------------------------------
    // Pass 3: elementwise merge D_g = D_g^{nnz} + D_g^{zero}.
    // ------------------------------------------------------------------
    singlet_gpu::core::DeviceMemory<float> d_deviance(m);
    {
        int g = (m + 255) / 256;
        deviance_merge_kernel<<<g, 256, 0, stream>>>(
            d_sg.get(), d_Dnnz.get(), d_Lg.get(),
            d_deviance.get(), grand_total, cfg.min_gene_total, m,
            cfg.use_poisson);
    }

    // ------------------------------------------------------------------
    // Top-N selection: cub::DeviceRadixSort descending on deviance.
    // ------------------------------------------------------------------
    singlet_gpu::core::DeviceMemory<int32_t> d_idx(m);
    {
        int g = (m + 255) / 256;
        fill_indices_kernel<<<g, 256, 0, stream>>>(d_idx.get(), m);
    }

    singlet_gpu::core::DeviceMemory<float>   d_dev_s(m);
    singlet_gpu::core::DeviceMemory<int32_t> d_idx_s(m);

    size_t tmp_sz = 0;
    cub::DeviceRadixSort::SortPairsDescending(nullptr, tmp_sz,
        d_deviance.get(), d_dev_s.get(),
        d_idx.get(),     d_idx_s.get(),
        m, 0, 32, stream);
    singlet_gpu::core::DeviceMemory<char> d_tmp(tmp_sz > 0 ? tmp_sz : 1);
    cub::DeviceRadixSort::SortPairsDescending(d_tmp.get(), tmp_sz,
        d_deviance.get(), d_dev_s.get(),
        d_idx.get(),     d_idx_s.get(),
        m, 0, 32, stream);

    // out_top: allocate top_n_req (caller's requested top-N) so that
    // copy_dev_result_hvg/copy_deviance_result can cudaMemcpy exactly that many
    // int32_t entries without going out-of-bounds.  Entries [top_n .. top_n_req)
    // are zero-filled by cudaMemsetAsync.
    // WHY top_n_req not top_n: streaming test (D5) has cfg.top_n=2000, n_genes=500;
    // kernel clamps top_n=500 but caller copies 2000*4 bytes → CUDA_ERROR_INVALID_VALUE
    // (root cause of the D5 streaming exception in job 368732).
    singlet_gpu::core::DeviceMemory<int32_t> out_top(top_n_req > 0 ? top_n_req : 1);
    singlet_gpu::core::DeviceMemory<uint8_t> out_mask(m);
    cudaMemsetAsync(out_mask.get(), 0,         m,                         stream);
    cudaMemsetAsync(out_top .get(), 0xFF, static_cast<size_t>(top_n_req > 0 ? top_n_req : 1) * sizeof(int32_t), stream);

    if (top_n > 0) {
        // Copy only the valid top_n sorted indices; remainder stays 0xFF (sentinel -1).
        cudaMemcpyAsync(out_top.get(), d_idx_s.get(),
                        static_cast<size_t>(top_n) * sizeof(int32_t),
                        cudaMemcpyDeviceToDevice, stream);
        int g = (top_n + 255) / 256;
        scatter_variable_mask_kernel<<<g, 256, 0, stream>>>(
            out_top.get(), out_mask.get(), top_n, m);
    }

    // Function-boundary sync: locally-owned sort scratch and d_idx_s/d_dev_s
    // are live until the async copies above complete.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Return d_deviance (gene-indexed, original gene order) NOT d_dev_s (sorted).
    // WHY: tests read result.deviance as deviance[gene_id] and compute Spearman vs
    // numpy's per-gene array.  d_dev_s is score-sorted — not gene-indexed — so
    // Spearman(d_dev_s, numpy_D_g) ≈ 0 for any non-trivial matrix.
    // (Root cause of Spearman ≈ 0.05 in all 5 failing tests in job 368732.)
    // d_dev_s (sorted) is kept alive until StreamSynchronize above; it is NOT
    // returned because no public API contract specifies sorted output in deviance[].
    DevianceHvgResult result;
    result.deviance           = std::move(d_deviance);  // gene-indexed, original order
    result.top_gene_idx       = std::move(out_top);
    result.is_variable        = std::move(out_mask);
    result.n_genes_considered = m;
    return result;
}

// ---------------------------------------------------------------------------
// deviance_feature_selection_with_lib_sizes — design-doc overload:
// Accepts a DeviceMemory<float> reference (design doc §Configuration, Rule 33).
// Delegates to the raw-pointer implementation; pointer is valid for lifetime
// of this call because d_lib_sizes is a const ref to a live DeviceMemory object.
// ---------------------------------------------------------------------------
inline DevianceHvgResult deviance_feature_selection_with_lib_sizes(
    const singlet_gpu::core::DeviceCSC&       counts,
    const singlet_gpu::core::DeviceMemory<float>& lib_sizes,
    const DevianceHvgConfig&                  cfg,
    cudaStream_t                              stream = nullptr)
{
    return deviance_feature_selection_with_lib_sizes(
        counts, lib_sizes.get(), cfg, stream);
}

// ---------------------------------------------------------------------------
// deviance_feature_selection — public entry point (computes own lib sizes)
// ---------------------------------------------------------------------------
inline DevianceHvgResult deviance_feature_selection(
    const singlet_gpu::core::DeviceCSC& counts,
    const DevianceHvgConfig&            cfg    = {},
    cudaStream_t                        stream = nullptr)
{
    return deviance_feature_selection_with_lib_sizes(counts, nullptr, cfg, stream);
}

} // namespace preprocess
} // namespace singlet_gpu
