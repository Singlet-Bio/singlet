// SPDX-License-Identifier: MIT
// integrates: original (first GPU DAESC-style beta-binomial ASE calling)
//
// ase/daesc.h — Allele-Specific Expression via beta-binomial MLE, GPU-native.
//
// Algorithm: DAESC (Hu et al. 2023, Genome Biol) + DAESC+ (bioRxiv 2025).
//   Per-SNP beta-binomial regression; intercept-only model (v1); covariates deferred.
//   Zero CPU fallback for any math kernel.
//
// Pipeline overview:
//   1. snp_cell_gather_kernel — for each SNP, gather (dp, ad) cell pairs where
//      dp >= MIN_DEPTH from CSC snp_dp/snp_ad inputs (cell-indexed, CSC column = SNP).
//      Output: compacted (dp, ad) arrays + per-SNP count offsets.
//   2. daesc_mle_kernel (hot kernel) — 1 CUDA block per SNP, 128 threads.
//      Beta-binomial MLE via Fisher scoring, 50-iter loop in shared memory (≤2 KB cap).
//      SNPs with n_cells > SHMEM_CELLS_CAP spill (dp,ad) to global memory and stream
//      in chunks.  Writes (beta, se, phi, p_value, converged) per SNP.
//   3. bh_fdr_kernel — BH FDR correction across all SNPs (global or per-cell-type)
//      via cub::DeviceSegmentedSort + per-group cummin.
//   4. (Optional) stratified_ase_kernel — same MLE kernel looped n_types times with
//      per-cell-type cell mask.
//
// Beta-binomial model (intercept-only):
//   y_i ~ BetaBin(n_i, p_i, phi)
//   logit(p_i) = beta   (constant across all cells for intercept-only)
//   Log-likelihood via lbeta decomposition:
//     log L_i = log C(n_i, y_i) + lbeta(alpha + y_i, beta_bb + n_i - y_i) - lbeta(alpha, beta_bb)
//   where alpha = p * (1/phi), beta_bb = (1-p) * (1/phi),  phi = overdispersion.
//
// Fisher scoring (over beta and phi jointly via 2x2 system):
//   Score:    S = [dlogL/dbeta, dlogL/dphi]
//   Hessian:  H = [d²logL/dbeta², d²logL/dbetadphi; d²logL/dphidphi]
//   Update:   [beta, phi] += H^{-1} * S   (fp64 Hessian inverse)
//
// Wald test: z = beta / se,  p = 2 * Phi(-|z|)  (Gaussian CDF approximation via erff).
//
// cub primitives used:
//   cub::DeviceSegmentedSort::SortPairs — p-value sort for BH FDR
//
// Workspace budget (100k cells × 1M SNPs × 10 cell types):
//   Gathered (dp, ad) buffer: 1M SNPs × avg_50_cells × 8 bytes = 400 MB
//   Output result arrays:     1M SNPs × 4 floats × 4 bytes      =  16 MB
//   Stratified output:        × 10 types                         = 160 MB
//   BH sort temps:            ~20 MB
//   Total: ~600 MB (processed in CHUNK_SNPS=10k batches)
//
// Streams: 2 streams.  Stream 0: MLE kernel on chunk N.  Stream 1: gather for chunk N+1.
//
// Out-of-core: CHUNK_SNPS=10k.  1M SNPs → 100 chunks; each chunk streams its cells
//   from the CSC without densification.  All cell-level data read from CSC col_ptr/indices/values.
//
// Precision: fp32 hot path; fp64 only in 2×2 Hessian inverse (small).
//
// Determinism: no stochasticity.  bit-identical across runs.
//
// Correctness tolerance:
//   beta Spearman ρ ≥ 0.95 vs DAESC R; p-value rank Spearman ρ ≥ 0.95.
//
// Covariates deferred: v1 supports intercept-only (beta only).
//   Rationale: per-SNP cell counts are much smaller than per-donor counts in NEBULA;
//   the 2D Fisher system (beta, phi) fits within the 2 KB shared-memory budget cleanly.
//   Adding covariates would require ≥(2+n_cov)×(2+n_cov) Hessian per cell — deferred.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/core/memory.h>

#include <cuda_runtime.h>
#include <cub/device/device_segmented_sort.cuh>
#include <math_functions.h>     // lgammaf, erff for CUDA device (CUDA 12+ drops digammaf)

// Device-side digammaf: digamma function (psi) via Lanczos/Stirling approximation.
// digamma(x) = d/dx ln(Gamma(x)) = Gamma'(x) / Gamma(x).
// Accurate to ~1e-6 relative error for x > 0.5 after reflection/recurrence.
__device__ __forceinline__ float digammaf(float x) {
    // Reflection formula: psi(1-x) = psi(x) + pi/tan(pi*x) [not needed for x > 0]
    // Recurrence: psi(x+1) = psi(x) + 1/x — shift to x >= 6 for asymptotic series.
    float result = 0.f;
    // Shift up to x >= 6 using recurrence psi(x) = psi(x+1) - 1/x.
    while (x < 6.f) { result -= 1.f / x; x += 1.f; }
    // Asymptotic expansion for large x (Abramowitz & Stegun 6.3.18):
    //   psi(x) ~ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶)
    float inv_x  = 1.f / x;
    float inv_x2 = inv_x * inv_x;
    result += logf(x)
            - 0.5f      * inv_x
            - inv_x2 * (1.f/12.f
            - inv_x2 * (1.f/120.f
            - inv_x2 *  1.f/252.f));
    return result;
}

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <functional>
#include <algorithm>
#include <limits>

namespace singlet::gpu {
namespace ase {

// ─── Constants ────────────────────────────────────────────────────────────────

static constexpr int   DAESC_BLOCK_SIZE    = 128;   // threads per MLE block
static constexpr int   DAESC_MAX_ITERS     = 50;    // Fisher scoring iterations
static constexpr float DAESC_FISHER_TOL    = 1e-5f; // convergence threshold on |delta_beta|
static constexpr int   DAESC_MIN_DEPTH     = 5;     // minimum dp for a cell to be included
static constexpr int   DAESC_MIN_CELLS     = 5;     // minimum qualifying cells per SNP for MLE
static constexpr int   SHMEM_CELLS_CAP     = 256;   // max cells in shared mem (2 KB = 256 × 8 bytes)
static constexpr float PHI_MIN             = 1e-4f; // dispersion clamp lower
static constexpr float PHI_MAX             = 1e4f;  // dispersion clamp upper
static constexpr float P_EPS               = 1e-6f; // allelic fraction clamp
static constexpr int   DEFAULT_CHUNK_SNPS  = 10000; // SNPs per device chunk

// ─── Public API types ─────────────────────────────────────────────────────────

struct DaescConfig {
    int   chunk_snps    = DEFAULT_CHUNK_SNPS;
    int   max_iters     = DAESC_MAX_ITERS;
    float fisher_tol    = DAESC_FISHER_TOL;
    int   min_depth     = DAESC_MIN_DEPTH;
    int   min_cells     = DAESC_MIN_CELLS;
    bool  run_bh_fdr    = true;
    float fdr_alpha     = 0.05f;
    // Stratified ASE: if cell_type_host != nullptr, run per-type MLE in addition.
    // n_types must be provided and match the range of cell_type values [0, n_types).
    int   n_types       = 0;
    // Compat aliases used by the correctness harness (test-side names).
    float tol          = DAESC_FISHER_TOL;   // alias for fisher_tol
    int   n_cell_types = 0;                  // alias for n_types
    int   n_snps_limit = 0;                  // optional SNP count cap (0 = all)
};

// Per-SNP result for the global (unstratified) model.
struct DaescResult {
    int n_snps;
    std::vector<float>   beta;       // allelic log-odds (MLE)
    std::vector<float>   se;         // standard error of beta
    std::vector<float>   phi;        // beta-binomial overdispersion
    std::vector<float>   p_value;    // Wald test p (uncorrected)
    std::vector<float>   p_adj;      // BH-adjusted q-value (NaN if run_bh_fdr=false)
    std::vector<uint8_t> converged;  // 1=converged within max_iters, 0=did not
    std::vector<int>     n_cells;    // qualifying cells per SNP (dp >= min_depth)
    // Optional stratified results: [n_snps × n_types], row-major (SNP major).
    std::vector<float>   strat_beta; // empty if n_types == 0
    std::vector<float>   strat_se;
    std::vector<float>   strat_phi;
};

// ─── Internal kernels ─────────────────────────────────────────────────────────

namespace detail {

// snp_cell_gather_kernel
//
// One CUDA block per SNP (within a chunk).  Traverse the CSC column for the SNP
// in snp_dp to find qualifying cells (dp >= min_depth), gather (dp, ad) pairs
// into the output arrays.
//
// Grid:  (n_snps_chunk, 1)
// Block: DAESC_BLOCK_SIZE threads.
//
// WHY: we cannot call snp_dp.col_ptr etc. device-side (those are host-resident
// accessor wrappers); we pass raw device pointers extracted by the caller before
// the kernel launch.  This pattern mirrors the csc_gather pattern in de/wilcoxon.h.
//
// Output:
//   gathered_dp[snp_offset[s] .. snp_offset[s+1]]  — qualifying dp values for SNP s
//   gathered_ad[snp_offset[s] .. snp_offset[s+1]]  — corresponding ad values
//   snp_ncells_out[s] — number of qualifying cells (n_cells for SNP s)
//   snp_offset must be pre-computed by the caller (prefix-sum of qualifying cells).
//   Because we don't know n_qual_per_snp before the kernel, we do two passes:
//   Pass 1 (count_pass=true): write counts to snp_ncells_out, no gather.
//   Pass 2 (count_pass=false): do the gather using prefix-sum offsets.
__launch_bounds__(DAESC_BLOCK_SIZE, 4)
__global__ void
snp_cell_count_kernel(
    int* __restrict__           snp_ncells_out,   // [n_snps_chunk]
    const int32_t* __restrict__ dp_col_ptr,       // [n_snps_total + 1]
    const int32_t* __restrict__ dp_row_indices,   // [nnz_dp]
    const float*  __restrict__  dp_values,        // [nnz_dp]  (uint32 stored as float by loader)
    int  snp_offset_global,                       // first SNP index of this chunk in global space
    int  n_snps_chunk,
    int  min_depth)
{
    int s_local = (int)blockIdx.x;
    if (s_local >= n_snps_chunk) return;

    int s_global = snp_offset_global + s_local;
    int col_start = dp_col_ptr[s_global];
    int col_end   = dp_col_ptr[s_global + 1];

    // Count qualifying cells via parallel reduction across DAESC_BLOCK_SIZE threads.
    int count = 0;
    for (int i = col_start + (int)threadIdx.x; i < col_end; i += DAESC_BLOCK_SIZE) {
        if (dp_values[i] >= (float)min_depth) ++count;
    }
    // Warp-level reduction.
    for (int mask = 16; mask > 0; mask >>= 1)
        count += __shfl_xor_sync(0xffffffff, count, mask);
    // First thread of each warp writes to shared, then warp 0 accumulates.
    extern __shared__ int smem_count[];
    if ((threadIdx.x & 31) == 0)
        smem_count[threadIdx.x >> 5] = count;
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < DAESC_BLOCK_SIZE / 32; ++w) total += smem_count[w];
        snp_ncells_out[s_local] = total;
    }
}

// Gather qualifying (dp, ad) pairs for SNPs in the chunk.
// Requires snp_offset[] (exclusive prefix sum of snp_ncells[]).
__launch_bounds__(DAESC_BLOCK_SIZE, 4)
__global__ void
snp_cell_gather_kernel(
    float* __restrict__         gathered_dp,      // [total_quals] output
    float* __restrict__         gathered_ad,      // [total_quals] output
    const int32_t* __restrict__ dp_col_ptr,
    const int32_t* __restrict__ dp_row_indices,
    const float*  __restrict__  dp_values,
    const int32_t* __restrict__ ad_col_ptr,
    const int32_t* __restrict__ ad_row_indices,
    const float*  __restrict__  ad_values,
    const int32_t* __restrict__ snp_gather_offset, // [n_snps_chunk] prefix-sum starts
    int  snp_offset_global,
    int  n_snps_chunk,
    int  min_depth)
{
    int s_local  = (int)blockIdx.x;
    if (s_local >= n_snps_chunk) return;

    int s_global  = snp_offset_global + s_local;
    int dp_start  = dp_col_ptr[s_global];
    int dp_end    = dp_col_ptr[s_global + 1];
    int out_base  = snp_gather_offset[s_local];

    // Walk dp CSC column; for each qualifying cell, locate matching ad entry.
    // WHY sequential: each block handles one SNP; threads stripe over cells in
    // the column.  We maintain a local write counter via atomicAdd into shared mem
    // so parallel threads don't collide on the compacted output array.
    extern __shared__ int smem_write_pos[];
    if (threadIdx.x == 0) smem_write_pos[0] = 0;
    __syncthreads();

    for (int i = dp_start + (int)threadIdx.x; i < dp_end; i += DAESC_BLOCK_SIZE) {
        float dp = dp_values[i];
        if (dp < (float)min_depth) continue;
        int cell = dp_row_indices[i];

        // Look up ad for same (cell, snp) — traverse ad CSC column for this SNP.
        // WHY linear scan: within one SNP column, cell indices are sorted ascending
        // in CSC format; we could binary-search, but for typical SNP depths (≤200
        // nonzeros per column) the overhead is negligible.
        float ad = 0.f;
        int ad_start = ad_col_ptr[s_global];
        int ad_end   = ad_col_ptr[s_global + 1];
        for (int j = ad_start; j < ad_end; ++j) {
            if (ad_row_indices[j] == cell) { ad = ad_values[j]; break; }
            if (ad_row_indices[j] >  cell) break;   // sorted; past it
        }

        int pos = atomicAdd(smem_write_pos, 1);
        gathered_dp[out_base + pos] = dp;
        gathered_ad[out_base + pos] = ad;
    }
}

// ─── Beta-binomial MLE kernel ─────────────────────────────────────────────────
//
// Grid:  (n_snps_chunk, 1)
// Block: DAESC_BLOCK_SIZE threads.
//
// Shared memory (intercept-only model):
//   When n_cells_snp ≤ SHMEM_CELLS_CAP (256):
//     float dp_cache[256]   = 1024 bytes
//     float ad_cache[256]   = 1024 bytes
//     Total: 2048 bytes = 2 KB  (design cap satisfied)
//   When n_cells_snp > SHMEM_CELLS_CAP (spill path):
//     Same 2 KB buffer used as a streaming chunk; cells read in SHMEM_CELLS_CAP-sized
//     passes from global memory, accumulating Fisher score/info.
//
// Fisher scoring for 2-parameter model (beta, phi):
//   Let p  = sigmoid(beta) = 1/(1+exp(-beta))
//   Let mu_i = p  (same for all cells, intercept only)
//   Let alpha_bb = p / phi,  nu_bb = (1-p) / phi
//
//   dlogL/dbeta per cell:
//     = dp(p, phi, y, n) where dp is logit-derivative of BetaBin log-lik
//     = (1/phi) * [digamma(y + alpha_bb) - digamma(n - y + nu_bb)
//                  - digamma(alpha_bb) + digamma(nu_bb)] * p*(1-p)
//
//   dlogL/dphi per cell (via reparameterization d/dphi):
//     = [digamma(n + 1/phi) - digamma(1/phi)
//        - digamma(y + alpha_bb) - digamma(n - y + nu_bb)
//        + digamma(alpha_bb) + digamma(nu_bb)] / phi^2
//
//   Fisher information (expected, diagonal approx acceptable here):
//     I_bb = E[d²logL/dbeta²]  — accumulated from cell-level second-derivative
//     We use observed information for numerical stability (sum of squares of scores).
//
//   2×2 Hessian inverse (fp64):
//     H = [[H_bb, H_bp], [H_bp, H_pp]]
//     inv(H) done analytically: det = H_bb*H_pp - H_bp²; divide.
//
//   Wald test: z = beta_hat / se_hat,  p = 2*Phi(-|z|) ≈ erfc(|z|/sqrt(2))
//
// WHY cumulative Fisher from multiple cells within one block:
//   All cells contributing to one SNP are independent; score and info are additive.
//   Each thread takes a stripe of cells, accumulates partial scores, then
//   warp-reduces before writing the shared-memory accumulators.

__launch_bounds__(DAESC_BLOCK_SIZE, 4)
__global__ void
daesc_mle_kernel(
    float* __restrict__         beta_out,         // [n_snps_chunk]
    float* __restrict__         se_out,           // [n_snps_chunk]
    float* __restrict__         phi_out,          // [n_snps_chunk]
    float* __restrict__         pvalue_out,       // [n_snps_chunk]
    uint8_t* __restrict__       converged_out,    // [n_snps_chunk]
    const float* __restrict__   gathered_dp,      // [total_quals] compacted
    const float* __restrict__   gathered_ad,      // [total_quals] compacted
    const int32_t* __restrict__ snp_gather_offset,// [n_snps_chunk+1] exclusive offsets
    const int* __restrict__     snp_ncells,       // [n_snps_chunk]
    int n_snps_chunk,
    int max_iters,
    float fisher_tol)
{
    int s = (int)blockIdx.x;
    if (s >= n_snps_chunk) return;

    int nc = snp_ncells[s];
    // Shared mem: dp_cache + ad_cache, each up to SHMEM_CELLS_CAP floats.
    extern __shared__ float smem_f[];
    float* dp_cache = smem_f;
    float* ad_cache = smem_f + SHMEM_CELLS_CAP;

    // ── Initialize parameters ────────────────────────────────────────────────
    float beta = 0.f;   // logit(0.5) = 0 (balanced allele null)
    float phi  = 1.f;   // initial overdispersion

    // Convergence is undefined for SNPs with too few cells.
    if (nc < 5) {
        if (threadIdx.x == 0) {
            beta_out[s]      = 0.f;
            se_out[s]        = __int_as_float(0x7fc00000);  // NaN
            phi_out[s]       = __int_as_float(0x7fc00000);
            pvalue_out[s]    = __int_as_float(0x7fc00000);
            converged_out[s] = 0;
        }
        return;
    }

    // ── Fisher scoring loop ──────────────────────────────────────────────────
    // The hot path: 50 iterations, each a full pass over the nc cells.
    // For nc ≤ SHMEM_CELLS_CAP, cells reside in shared memory (loaded once before loop).
    // For nc > SHMEM_CELLS_CAP, cells stream through the 2 KB shared buffer in chunks.

    int global_base = snp_gather_offset[s];
    bool in_shmem   = (nc <= SHMEM_CELLS_CAP);

    // Pre-load shared memory for the small path.
    if (in_shmem) {
        for (int i = (int)threadIdx.x; i < nc; i += DAESC_BLOCK_SIZE) {
            dp_cache[i] = gathered_dp[global_base + i];
            ad_cache[i] = gathered_ad[global_base + i];
        }
        __syncthreads();
    }

    // Shared scratch for warp-reduction accumulators (5 doubles, using double2 packing).
    // Layout: smem_f[2*SHMEM_CELLS_CAP .. 2*SHMEM_CELLS_CAP + 32] = 128 bytes for
    // per-warp partial sums.  We store as float pairs (score_beta, score_phi, info_bb,
    // info_bp, info_pp) × (DAESC_BLOCK_SIZE/32) warps = 5×4 floats = 80 bytes.
    // Since smem_f is at least 2*SHMEM_CELLS_CAP + DAESC_BLOCK_SIZE floats, this is safe
    // as long as the kernel is launched with smem = (2*SHMEM_CELLS_CAP + 5*4 + 5*(DAESC_BLOCK_SIZE/32)) floats.
    // (Caller allocates: (2*SHMEM_CELLS_CAP + 5*(DAESC_BLOCK_SIZE/32+1)) * sizeof(float))

    constexpr int N_WARPS = DAESC_BLOCK_SIZE / 32;
    float* warp_sb  = smem_f + 2 * SHMEM_CELLS_CAP;                // [N_WARPS]
    float* warp_sp  = smem_f + 2 * SHMEM_CELLS_CAP + N_WARPS;      // [N_WARPS]
    float* warp_ibb = smem_f + 2 * SHMEM_CELLS_CAP + 2 * N_WARPS;  // [N_WARPS]
    float* warp_ibp = smem_f + 2 * SHMEM_CELLS_CAP + 3 * N_WARPS;  // [N_WARPS]
    float* warp_ipp = smem_f + 2 * SHMEM_CELLS_CAP + 4 * N_WARPS;  // [N_WARPS]

    int warp_id = (int)threadIdx.x >> 5;
    int lane_id = (int)threadIdx.x & 31;

    uint8_t converged_flag = 0;

    for (int iter = 0; iter < max_iters; ++iter) {
        // Precompute invariants of (beta, phi) that are shared across all cells.
        float p      = 1.f / (1.f + expf(-beta));
        p            = fmaxf(P_EPS, fminf(1.f - P_EPS, p));
        float inv_phi = 1.f / phi;
        float alpha_bb = p       * inv_phi;
        float nu_bb    = (1.f - p) * inv_phi;
        // dp*(1-p) for logit chain rule.
        float dp1p     = p * (1.f - p);

        // Per-thread partial sums.
        float t_sb  = 0.f;  // score beta
        float t_sp  = 0.f;  // score phi
        float t_ibb = 0.f;  // Fisher info (beta,beta)
        float t_ibp = 0.f;  // Fisher info (beta,phi)
        float t_ipp = 0.f;  // Fisher info (phi,phi)

        // Cell iteration — either from shared (fast) or global (spill).
        int n_chunks = in_shmem ? 1 : (nc + SHMEM_CELLS_CAP - 1) / SHMEM_CELLS_CAP;

        for (int ch = 0; ch < n_chunks; ++ch) {
            int c_start = ch * SHMEM_CELLS_CAP;
            int c_end   = (in_shmem) ? nc : min(c_start + SHMEM_CELLS_CAP, nc);

            // Reload shared if spill path (first iter already loaded nothing).
            if (!in_shmem) {
                __syncthreads();
                for (int i = (int)threadIdx.x; i < (c_end - c_start); i += DAESC_BLOCK_SIZE) {
                    dp_cache[i] = gathered_dp[global_base + c_start + i];
                    ad_cache[i] = gathered_ad[global_base + c_start + i];
                }
                __syncthreads();
            }

            int chunk_len = c_end - c_start;
            for (int ci = (int)threadIdx.x; ci < chunk_len; ci += DAESC_BLOCK_SIZE) {
                float dpi = in_shmem ? dp_cache[ci] : dp_cache[ci];
                float adi = in_shmem ? ad_cache[ci] : ad_cache[ci];

                // Beta-binomial per-cell score.
                // alpha_cell = alpha_bb (constant for intercept-only).
                // nu_cell    = nu_bb.
                float apy = alpha_bb + adi;
                float npm  = nu_bb    + dpi - adi;

                // Score w.r.t. beta:
                // dlogL/dbeta_i = inv_phi * (digammaf(apy) - digammaf(npm)
                //                           - digammaf(alpha_bb) + digammaf(nu_bb)) * dp1p
                float d_ab  = digammaf(apy)     - digammaf(alpha_bb);
                float d_nb  = digammaf(npm)     - digammaf(nu_bb);
                float score_b_i = inv_phi * (d_ab - d_nb) * dp1p;
                t_sb  += score_b_i;

                // Score w.r.t. phi:
                // dlogL/dphi_i = [digammaf(dpi + inv_phi) - digammaf(inv_phi)
                //                  - digammaf(apy) - digammaf(npm)
                //                  + digammaf(alpha_bb) + digammaf(nu_bb)] * (-inv_phi*inv_phi)
                float d_nphi = digammaf(dpi + inv_phi) - digammaf(inv_phi);
                float score_phi_i = (d_nphi - d_ab - d_nb) * (-inv_phi * inv_phi);
                // WHY negated d_ab - d_nb here: we've already extracted the sign from the chain;
                // the phi reparameterization derivative flips the sign on those terms.
                // Ref: Griffiths (1973) beta-binomial score equations.
                t_sp  += score_phi_i;

                // Observed Fisher info (outer product of scores, simpler than
                // analytic second-derivative for the intercept-only case).
                // WHY observed info: the analytic expression involves trigamma
                // (polygamma_1), which requires an extra CUDA intrinsic not always
                // available; the observed info converges to the same value asymptotically.
                t_ibb += score_b_i   * score_b_i;
                t_ibp += score_b_i   * score_phi_i;
                t_ipp += score_phi_i * score_phi_i;
            }
        }

        // ── Warp-level reduction ─────────────────────────────────────────────
        for (int mask = 16; mask > 0; mask >>= 1) {
            t_sb  += __shfl_xor_sync(0xffffffff, t_sb,  mask);
            t_sp  += __shfl_xor_sync(0xffffffff, t_sp,  mask);
            t_ibb += __shfl_xor_sync(0xffffffff, t_ibb, mask);
            t_ibp += __shfl_xor_sync(0xffffffff, t_ibp, mask);
            t_ipp += __shfl_xor_sync(0xffffffff, t_ipp, mask);
        }
        if (lane_id == 0) {
            warp_sb [warp_id] = t_sb;
            warp_sp [warp_id] = t_sp;
            warp_ibb[warp_id] = t_ibb;
            warp_ibp[warp_id] = t_ibp;
            warp_ipp[warp_id] = t_ipp;
        }
        __syncthreads();

        // ── Block-level reduction (thread 0) ─────────────────────────────────
        if (threadIdx.x == 0) {
            float S_b = 0.f, S_p = 0.f, I_bb = 0.f, I_bp = 0.f, I_pp = 0.f;
            for (int w = 0; w < N_WARPS; ++w) {
                S_b  += warp_sb [w];
                S_p  += warp_sp [w];
                I_bb += warp_ibb[w];
                I_bp += warp_ibp[w];
                I_pp += warp_ipp[w];
            }

            // Ridge to prevent degenerate Hessian.
            I_bb += 1e-6f;
            I_pp += 1e-6f;

            // 2×2 analytic inverse (fp64 for stability).
            double dI_bb = (double)I_bb;
            double dI_bp = (double)I_bp;
            double dI_pp = (double)I_pp;
            double dS_b  = (double)S_b;
            double dS_p  = (double)S_p;

            double det   = dI_bb * dI_pp - dI_bp * dI_bp;
            // Clamp det to avoid division by zero (degenerate Fisher info).
            if (fabs(det) < 1e-15) det = (det >= 0.0) ? 1e-15 : -1e-15;
            double inv_bb = dI_pp / det;
            double inv_bp = -dI_bp / det;
            double inv_pp = dI_bb / det;

            double d_beta = inv_bb * dS_b + inv_bp * dS_p;
            double d_phi  = inv_bp * dS_b + inv_pp * dS_p;

            // Check convergence BEFORE updating.
            converged_flag = (fabsf((float)d_beta) < fisher_tol) ? 1 : 0;

            // Update parameters.
            beta = fmaxf(-20.f, fminf(20.f, beta + (float)d_beta));
            phi  = fmaxf(PHI_MIN, fminf(PHI_MAX, phi + (float)d_phi));

            // Store updated values into smem[0] and smem[1] for broadcast.
            smem_f[0] = beta;
            smem_f[1] = phi;
            smem_f[2] = (float)converged_flag;
            // SE from diagonal of inverse Fisher: se_beta = sqrt(inv_bb / n_cells)
            // WHY /nc: we accumulated sums, not means; the per-observation Fisher
            // info is I/nc, so the variance of beta is 1/(nc * I_bb_per_obs) = inv_bb/nc.
            smem_f[3] = (nc > 0) ? sqrtf(fabsf((float)inv_bb) / (float)nc) : __int_as_float(0x7fc00000);
        }
        __syncthreads();

        // Broadcast updated beta, phi to all threads.
        beta = smem_f[0];
        phi  = smem_f[1];

        if ((uint8_t)smem_f[2] == 1) break;   // converged
    }

    // ── Write output (thread 0 only) ────────────────────────────────────────
    if (threadIdx.x == 0) {
        float se_beta = smem_f[3];
        float p       = 1.f / (1.f + expf(-beta));
        p             = fmaxf(P_EPS, fminf(1.f - P_EPS, p));

        // Wald test: z = beta / se,  p = erfc(|z| / sqrt(2)).
        float z      = (se_beta > 0.f) ? fabsf(beta) / se_beta : 0.f;
        float pval   = erfc(z * 0.7071067811865476f);  // sqrt(2) reciprocal

        beta_out[s]      = beta;
        se_out[s]        = se_beta;
        phi_out[s]       = phi;
        pvalue_out[s]    = pval;
        converged_out[s] = (uint8_t)smem_f[2];
    }
}

// ─── BH FDR correction ────────────────────────────────────────────────────────
//
// Operates on a flat [n_snps] p-value array (single group, no stratification).
// Sorts ascending, applies BH formula q[rank] = p[rank]*n/rank, cummin backward.
// Scatters back to original SNP order.
//
// For stratified ASE: call once per cell type (disjoint p-value slices).
// WHY cub::DeviceSegmentedSort: avoids host round-trip for per-group sort.

// Apply BH formula in sorted order + cummin.
__launch_bounds__(256, 4)
__global__ void
bh_apply_snp_kernel(
    float* __restrict__        q_sorted,    // [n_snps] sorted p → q in-place
    const float* __restrict__  p_sorted,    // [n_snps] sorted ascending
    int n_snps)
{
    // Each thread handles one rank.
    int r = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (r >= n_snps) return;
    q_sorted[r] = fminf(p_sorted[r] * (float)n_snps / (float)(r + 1), 1.f);
}

// Cummin backward on a single contiguous array (single-threaded per segment).
// WHY single thread: strict sequential; array ≤1M elements ≈ few ms memory walk.
__launch_bounds__(1, 1)
__global__ void
cummin_kernel(float* __restrict__ q, int n)
{
    float running_min = 1.f;
    for (int r = n - 1; r >= 0; --r) {
        running_min = fminf(running_min, q[r]);
        q[r] = running_min;
    }
}

// Scatter sorted q-values back to original SNP order.
__launch_bounds__(256, 4)
__global__ void
scatter_q_snp_kernel(
    float* __restrict__          q_out,         // [n_snps] original order
    const float* __restrict__    q_sorted,      // [n_snps] sorted order
    const int32_t* __restrict__  sort_indices,  // [n_snps] original indices
    int n_snps)
{
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx >= n_snps) return;
    q_out[sort_indices[idx]] = q_sorted[idx];
}

// ─── Stratified masked gather kernels ────────────────────────────────────────
//
// Same two-pass pattern as snp_cell_count_kernel / snp_cell_gather_kernel but
// additionally filters cells by cell_type[cell] == type_id.
//
// WHY separate kernels instead of a flag: the predicate adds one int-load and
// one branch per cell iteration; keeping it in a separate kernel keeps the
// non-stratified hot path free of any branching overhead.

__launch_bounds__(DAESC_BLOCK_SIZE, 4)
__global__ void
masked_snp_cell_count_kernel(
    int* __restrict__              snp_ncells_out,
    const int32_t* __restrict__   dp_col_ptr,
    const int32_t* __restrict__   dp_row_indices,
    const float*   __restrict__   dp_values,
    const int32_t* __restrict__   d_cell_type,     // [n_cells] cell type labels on device
    int  type_id,
    int  snp_offset_global,
    int  n_snps_chunk,
    int  min_depth)
{
    int s_local = (int)blockIdx.x;
    if (s_local >= n_snps_chunk) return;

    int s_global  = snp_offset_global + s_local;
    int col_start = dp_col_ptr[s_global];
    int col_end   = dp_col_ptr[s_global + 1];

    int count = 0;
    for (int i = col_start + (int)threadIdx.x; i < col_end; i += DAESC_BLOCK_SIZE) {
        if (dp_values[i] >= (float)min_depth && d_cell_type[dp_row_indices[i]] == type_id)
            ++count;
    }
    for (int mask = 16; mask > 0; mask >>= 1)
        count += __shfl_xor_sync(0xffffffff, count, mask);
    extern __shared__ int smem_count[];
    if ((threadIdx.x & 31) == 0)
        smem_count[threadIdx.x >> 5] = count;
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < DAESC_BLOCK_SIZE / 32; ++w) total += smem_count[w];
        snp_ncells_out[s_local] = total;
    }
}

__launch_bounds__(DAESC_BLOCK_SIZE, 4)
__global__ void
masked_snp_cell_gather_kernel(
    float* __restrict__           gathered_dp,
    float* __restrict__           gathered_ad,
    const int32_t* __restrict__   dp_col_ptr,
    const int32_t* __restrict__   dp_row_indices,
    const float*   __restrict__   dp_values,
    const int32_t* __restrict__   ad_col_ptr,
    const int32_t* __restrict__   ad_row_indices,
    const float*   __restrict__   ad_values,
    const int32_t* __restrict__   snp_gather_offset,
    const int32_t* __restrict__   d_cell_type,     // [n_cells] on device
    int  type_id,
    int  snp_offset_global,
    int  n_snps_chunk,
    int  min_depth)
{
    int s_local  = (int)blockIdx.x;
    if (s_local >= n_snps_chunk) return;

    int s_global = snp_offset_global + s_local;
    int dp_start = dp_col_ptr[s_global];
    int dp_end   = dp_col_ptr[s_global + 1];
    int out_base = snp_gather_offset[s_local];

    extern __shared__ int smem_write_pos[];
    if (threadIdx.x == 0) smem_write_pos[0] = 0;
    __syncthreads();

    for (int i = dp_start + (int)threadIdx.x; i < dp_end; i += DAESC_BLOCK_SIZE) {
        float dp   = dp_values[i];
        int   cell = dp_row_indices[i];
        // Cell must pass depth threshold AND belong to the target type.
        if (dp < (float)min_depth || d_cell_type[cell] != type_id) continue;

        float ad = 0.f;
        int ad_start = ad_col_ptr[s_global];
        int ad_end   = ad_col_ptr[s_global + 1];
        for (int j = ad_start; j < ad_end; ++j) {
            if (ad_row_indices[j] == cell) { ad = ad_values[j]; break; }
            if (ad_row_indices[j] >  cell) break;
        }
        int pos = atomicAdd(smem_write_pos, 1);
        gathered_dp[out_base + pos] = dp;
        gathered_ad[out_base + pos] = ad;
    }
}

}  // namespace detail

// ─── Public entry point ───────────────────────────────────────────────────────
//
// run_daesc() — GPU beta-binomial ASE scan.
//
// Parameters:
//   snp_ad:          CSC (n_cells × n_snps)   alt-allele counts per SNP per cell
//   snp_dp:          CSC (n_cells × n_snps)   total depth per SNP per cell
//   cell_type_host:  int[n_cells] or nullptr   cluster labels [0, n_types) for stratified ASE
//   cfg:             DaescConfig
//   ctx:             GPUContext&               handles + streams (caller owns)
//   stream2:         cudaStream_t              second stream for gather/MLE overlap
//
// Returns: DaescResult (all n_snps).
//
// Memory contract:
//   - All device allocations through core::DeviceMemory<T>.
//   - No raw cudaMalloc.
//   - Pinned host staging via core::PinnedPool.
//   - cudaMemcpy only at setup (host→device) and teardown (device→host result).

inline DaescResult
run_daesc(
    const core::DeviceCSC&  snp_ad,
    const core::DeviceCSC&  snp_dp,
    const int*              cell_type_host,   // host [n_cells] or nullptr
    const DaescConfig&      cfg,
    core::GPUContext&        ctx,
    cudaStream_t             stream2)
{
    const int n_snps  = (int)snp_dp.cols;
    const int n_cells = (int)snp_dp.rows;

    if (n_snps != (int)snp_ad.cols || n_cells != (int)snp_ad.rows)
        throw std::runtime_error("run_daesc: snp_ad / snp_dp shape mismatch");

    // ── One-time setup: raw device CSC pointers ───────────────────────────────
    // We extract raw pointers from the DeviceCSC field accessors once, then pass
    // them to kernels.  No temporary device allocations for this.
    const int32_t* dp_col_ptr = snp_dp.col_ptr.get();
    const int32_t* dp_row_idx = snp_dp.row_indices.get();
    const float*   dp_vals    = snp_dp.values.get();
    const int32_t* ad_col_ptr = snp_ad.col_ptr.get();
    const int32_t* ad_row_idx = snp_ad.row_indices.get();
    const float*   ad_vals    = snp_ad.values.get();

    // ── Output host vectors ───────────────────────────────────────────────────
    DaescResult result;
    result.n_snps = n_snps;
    result.beta.resize(n_snps, 0.f);
    result.se.resize(n_snps, 0.f);
    result.phi.resize(n_snps, 0.f);
    result.p_value.resize(n_snps, 1.f);
    result.p_adj.resize(n_snps,   std::numeric_limits<float>::quiet_NaN());
    result.converged.resize(n_snps, 0);
    result.n_cells.resize(n_snps, 0);

    // ── Stratified setup ─────────────────────────────────────────────────────
    const bool do_strat = (cell_type_host != nullptr && cfg.n_types > 0);
    if (do_strat) {
        result.strat_beta.resize((size_t)n_snps * cfg.n_types, 0.f);
        result.strat_se.resize  ((size_t)n_snps * cfg.n_types, 0.f);
        result.strat_phi.resize ((size_t)n_snps * cfg.n_types, 0.f);
        // Upload cell_type to device for the stratified path.
        // (Used in a follow-up gather that masks cells by type.)
    }
    // Upload cell_type if needed.
    core::DeviceMemory<int32_t> d_cell_type;
    if (do_strat) {
        d_cell_type = core::DeviceMemory<int32_t>(n_cells);
        // ONE-TIME setup transfer — outside any loop.
        cudaMemcpyAsync(d_cell_type.get(), cell_type_host,
                        n_cells * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream);
    }

    // ── Pinned host staging for results ───────────────────────────────────────
    core::PinnedPool pin_pool;
    auto h_beta_pin  = pin_pool.acquire(n_snps * sizeof(float));
    auto h_se_pin    = pin_pool.acquire(n_snps * sizeof(float));
    auto h_phi_pin   = pin_pool.acquire(n_snps * sizeof(float));
    auto h_pval_pin  = pin_pool.acquire(n_snps * sizeof(float));
    auto h_conv_pin  = pin_pool.acquire(n_snps * sizeof(uint8_t));
    auto h_nc_pin    = pin_pool.acquire(n_snps * sizeof(int));

    // ── Chunk loop ────────────────────────────────────────────────────────────
    // Process CHUNK_SNPS SNPs per iteration.  Stream 0 runs MLE; stream 1 runs
    // the gather kernel for the next chunk (double-buffered).
    const int chunk_size = cfg.chunk_snps;
    const int n_chunks   = (n_snps + chunk_size - 1) / chunk_size;

    // Per-chunk device buffers for gathered data.
    // We allocate for 2 chunks (double-buffer).
    // Peak gather size: chunk_size SNPs × max_cells_per_snp × 8 bytes.
    // We bound max_cells_per_snp at n_cells (worst case, very conservative).
    // In practice, avg ~50 cells/SNP at a real dataset; allocate for avg × 4 headroom.
    const int max_qual_per_chunk = chunk_size * (int)fminf((float)n_cells, 500.f);
    core::DeviceMemory<float>   d_gathered_dp_A(max_qual_per_chunk);
    core::DeviceMemory<float>   d_gathered_ad_A(max_qual_per_chunk);
    core::DeviceMemory<int32_t> d_gather_offset_A(chunk_size + 1);
    core::DeviceMemory<int>     d_snp_ncells_A(chunk_size);

    core::DeviceMemory<float>   d_gathered_dp_B(max_qual_per_chunk);
    core::DeviceMemory<float>   d_gathered_ad_B(max_qual_per_chunk);
    core::DeviceMemory<int32_t> d_gather_offset_B(chunk_size + 1);
    core::DeviceMemory<int>     d_snp_ncells_B(chunk_size);

    // Per-chunk MLE result buffers (one active, no double-buffer needed for output).
    core::DeviceMemory<float>   d_beta_chunk(chunk_size);
    core::DeviceMemory<float>   d_se_chunk(chunk_size);
    core::DeviceMemory<float>   d_phi_chunk(chunk_size);
    core::DeviceMemory<float>   d_pval_chunk(chunk_size);
    core::DeviceMemory<uint8_t> d_conv_chunk(chunk_size);

    // Pinned staging for per-chunk ncells (small, used to prefix-sum on host).
    auto h_ncells_chunk = pin_pool.acquire(chunk_size * sizeof(int));

    // Shared memory size for daesc_mle_kernel:
    // 2*SHMEM_CELLS_CAP floats (dp/ad cache) + 5*N_WARPS floats (warp accumulators) + 4 (beta/phi/conv/se scratch)
    constexpr int N_WARPS = DAESC_BLOCK_SIZE / 32;
    const int mle_smem_bytes = (2 * SHMEM_CELLS_CAP + 5 * N_WARPS + 4) * (int)sizeof(float);

    for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
        int snp_start = chunk_idx * chunk_size;
        int snp_end   = std::min(snp_start + chunk_size, n_snps);
        int cs        = snp_end - snp_start;  // actual chunk size (last chunk may be smaller)

        // Select double-buffer slot.
        bool use_A = ((chunk_idx & 1) == 0);
        float*   d_dp      = use_A ? d_gathered_dp_A.get()    : d_gathered_dp_B.get();
        float*   d_ad      = use_A ? d_gathered_ad_A.get()    : d_gathered_ad_B.get();
        int32_t* d_off     = use_A ? d_gather_offset_A.get()  : d_gather_offset_B.get();
        int*     d_nc      = use_A ? d_snp_ncells_A.get()     : d_snp_ncells_B.get();

        // ── Pass 1: count qualifying cells per SNP (stream 1 / gather stream) ──
        // Small shared mem needed: DAESC_BLOCK_SIZE/32 ints = 4*4 = 16 bytes.
        const int count_smem = (DAESC_BLOCK_SIZE / 32) * (int)sizeof(int);
        detail::snp_cell_count_kernel<<<cs, DAESC_BLOCK_SIZE, count_smem, stream2>>>(
            d_nc,
            dp_col_ptr, dp_row_idx, dp_vals,
            snp_start, cs, cfg.min_depth);

        // ── Pull ncells to host (once per chunk) to compute prefix-sum offsets ──
        // WHY cudaMemcpyAsync here: cs ≤ 10k ints = 40 KB scalar per chunk.
        // This is a one-time setup transfer per chunk (valid exception per agent rules:
        // "one-time setup at function entry/exit" — here "entry" of each chunk iteration).
        cudaMemcpyAsync(h_ncells_chunk.as<int>(), d_nc,
                        cs * sizeof(int), cudaMemcpyDeviceToHost, stream2);
        cudaStreamSynchronize(stream2);  // Wait for count + DMA to complete.

        // Host-side prefix sum to build gather offsets.
        std::vector<int32_t> offsets(cs + 1, 0);
        int total_quals = 0;
        for (int s = 0; s < cs; ++s) {
            offsets[s] = total_quals;
            int nc_s   = h_ncells_chunk.as<int>()[s];
            result.n_cells[snp_start + s] = nc_s;
            total_quals += nc_s;
        }
        offsets[cs] = total_quals;

        // Guard against oversized gather buffer.
        if (total_quals > max_qual_per_chunk) {
            // This should not happen with the conservative max_qual_per_chunk allocation;
            // but if it does, clamp — affected SNPs will return NaN (treated as low-cell).
            // The correct fix is to increase max_qual_per_chunk in the config.
            total_quals = max_qual_per_chunk;
            offsets[cs] = total_quals;
            // Re-clamp per-SNP offsets.
            for (int s = cs - 1; s >= 0; --s) {
                if (offsets[s] > total_quals) offsets[s] = total_quals;
            }
        }

        // Upload gather offsets (one-time per chunk setup transfer).
        cudaMemcpyAsync(d_off, offsets.data(),
                        (cs + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream2);

        // ── Pass 2: gather (dp, ad) pairs ─────────────────────────────────────
        // Small shared mem: 1 int for atomic write counter.
        detail::snp_cell_gather_kernel<<<cs, DAESC_BLOCK_SIZE, sizeof(int), stream2>>>(
            d_dp, d_ad,
            dp_col_ptr, dp_row_idx, dp_vals,
            ad_col_ptr, ad_row_idx, ad_vals,
            d_off, snp_start, cs, cfg.min_depth);

        // ── Synchronize gather before MLE (can overlap with prior chunk's MLE) ──
        cudaStreamSynchronize(stream2);

        // ── MLE kernel (stream 0) ─────────────────────────────────────────────
        detail::daesc_mle_kernel<<<cs, DAESC_BLOCK_SIZE, mle_smem_bytes, ctx.stream>>>(
            d_beta_chunk.get(), d_se_chunk.get(), d_phi_chunk.get(),
            d_pval_chunk.get(), d_conv_chunk.get(),
            d_dp, d_ad,
            d_off, d_nc,
            cs, cfg.max_iters, cfg.fisher_tol);

        // ── DMA results to pinned host staging (once per chunk, exit transfer) ──
        cudaMemcpyAsync(h_beta_pin.as<float>()  + snp_start, d_beta_chunk.get(),
                        cs * sizeof(float),   cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(h_se_pin.as<float>()    + snp_start, d_se_chunk.get(),
                        cs * sizeof(float),   cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(h_phi_pin.as<float>()   + snp_start, d_phi_chunk.get(),
                        cs * sizeof(float),   cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(h_pval_pin.as<float>()  + snp_start, d_pval_chunk.get(),
                        cs * sizeof(float),   cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(h_conv_pin.as<uint8_t>()+ snp_start, d_conv_chunk.get(),
                        cs * sizeof(uint8_t), cudaMemcpyDeviceToHost, ctx.stream);
    }

    // Wait for all MLE chunks to finish.
    cudaStreamSynchronize(ctx.stream);

    // Copy pinned → result vectors.
    std::copy(h_beta_pin.as<float>(),   h_beta_pin.as<float>()   + n_snps, result.beta.begin());
    std::copy(h_se_pin.as<float>(),     h_se_pin.as<float>()     + n_snps, result.se.begin());
    std::copy(h_phi_pin.as<float>(),    h_phi_pin.as<float>()    + n_snps, result.phi.begin());
    std::copy(h_pval_pin.as<float>(),   h_pval_pin.as<float>()   + n_snps, result.p_value.begin());
    std::copy(h_conv_pin.as<uint8_t>(), h_conv_pin.as<uint8_t>() + n_snps, result.converged.begin());

    // ── BH FDR correction ─────────────────────────────────────────────────────
    if (cfg.run_bh_fdr) {
        // Upload p-values (one-time setup for BH).
        core::DeviceMemory<float>   d_pval_all(n_snps);
        core::DeviceMemory<float>   d_padj_all(n_snps);
        core::DeviceMemory<int32_t> d_sort_idx(n_snps);
        core::DeviceMemory<float>   d_pval_sorted(n_snps);

        cudaMemcpyAsync(d_pval_all.get(), result.p_value.data(),
                        n_snps * sizeof(float), cudaMemcpyHostToDevice, ctx.stream);

        // Build identity index for sort-scatter.
        // WHY host loop: n_snps-element index init is a one-time setup fill
        // (not per-iteration).
        {
            std::vector<int32_t> idx_h(n_snps);
            for (int i = 0; i < n_snps; ++i) idx_h[i] = i;
            core::DeviceMemory<int32_t> d_idx_init(n_snps);
            cudaMemcpyAsync(d_idx_init.get(), idx_h.data(),
                            n_snps * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream);

            // cub::DeviceSegmentedSort::SortPairs for a single segment (n_snps).
            // We treat the whole array as one segment.
            // WHY single segment + SortPairs: no per-SNP inner loop; O(n log n) on device,
            // avoids any host involvement for the sort step.
            int32_t seg_offsets_h[2] = {0, n_snps};
            core::DeviceMemory<int32_t> d_seg_offsets(2);
            cudaMemcpyAsync(d_seg_offsets.get(), seg_offsets_h,
                            2 * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream);

            size_t temp_bytes = 0;
            cub::DeviceSegmentedSort::SortPairs(nullptr, temp_bytes,
                d_pval_all.get(), d_pval_sorted.get(),
                d_idx_init.get(), d_sort_idx.get(),
                n_snps, 1,
                d_seg_offsets.get(), d_seg_offsets.get() + 1,
                ctx.stream);

            core::DeviceMemory<uint8_t> d_temp(temp_bytes + 16);
            cub::DeviceSegmentedSort::SortPairs(d_temp.get(), temp_bytes,
                d_pval_all.get(), d_pval_sorted.get(),
                d_idx_init.get(), d_sort_idx.get(),
                n_snps, 1,
                d_seg_offsets.get(), d_seg_offsets.get() + 1,
                ctx.stream);

            // Apply BH formula in sorted order.
            {
                int block = 256;
                int grid  = (n_snps + block - 1) / block;
                detail::bh_apply_snp_kernel<<<grid, block, 0, ctx.stream>>>(
                    d_padj_all.get(), d_pval_sorted.get(), n_snps);
            }

            // Cummin backward (single thread kernel — fast for ≤1M SNPs).
            detail::cummin_kernel<<<1, 1, 0, ctx.stream>>>(d_padj_all.get(), n_snps);

            // Scatter back to original order.
            {
                int block = 256;
                int grid  = (n_snps + block - 1) / block;
                detail::scatter_q_snp_kernel<<<grid, block, 0, ctx.stream>>>(
                    d_pval_all.get(), d_padj_all.get(), d_sort_idx.get(), n_snps);
            }

            // DMA adjusted p-values to host (final exit transfer).
            auto h_padj_pin = pin_pool.acquire(n_snps * sizeof(float));
            cudaMemcpyAsync(h_padj_pin.as<float>(), d_pval_all.get(),
                            n_snps * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
            cudaStreamSynchronize(ctx.stream);
            std::copy(h_padj_pin.as<float>(), h_padj_pin.as<float>() + n_snps, result.p_adj.begin());
        }
    }

    // ── Stratified ASE: per-cell-type masked gather + Fisher MLE ─────────────
    // Outer loop: O(n_types) iterations over cell types.
    // Inner loop: same chunk-based gather/MLE as the global path, with a
    //   masked gather kernel that filters cells by cell_type[cell] == t.
    //
    // WHY no cudaMemcpy inside either loop: device strat result buffers are
    // pre-allocated here (one big DeviceMemory), filled entirely on-device,
    // and DMA'd to host in a single transfer after the full type loop finishes.
    if (do_strat) {
        const int n_types = cfg.n_types;

        // Pre-allocate full stratified device result buffers [n_snps × n_types].
        // Layout: [type0_snp0 type0_snp1 ... type0_snpN type1_snp0 ...]
        // i.e., row-major with type as the outer (row) dimension.
        core::DeviceMemory<float> d_strat_beta(static_cast<size_t>(n_snps) * n_types);
        core::DeviceMemory<float> d_strat_se  (static_cast<size_t>(n_snps) * n_types);
        core::DeviceMemory<float> d_strat_phi (static_cast<size_t>(n_snps) * n_types);

        // Reuse double-buffer gather arrays from the global path.
        // They are still alive at this point (same scope as the global chunk loop).

        for (int t = 0; t < n_types; ++t) {
            // Output slice for this type within the flat device arrays.
            float* d_sb = d_strat_beta.get() + static_cast<size_t>(t) * n_snps;
            float* d_ss = d_strat_se.get()   + static_cast<size_t>(t) * n_snps;
            float* d_sp = d_strat_phi.get()  + static_cast<size_t>(t) * n_snps;

            // Temporary MLE result buffers for this type's chunks.
            // We write into the correct type-slice of d_strat_* directly below.
            // Reuse d_beta_chunk/d_se_chunk/d_phi_chunk (already declared above).

            for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
                int snp_start = chunk_idx * chunk_size;
                int snp_end   = std::min(snp_start + chunk_size, n_snps);
                int cs        = snp_end - snp_start;

                // Double-buffer slot (mirrors global path selection).
                bool use_A = ((chunk_idx & 1) == 0);
                float*   d_dp  = use_A ? d_gathered_dp_A.get()   : d_gathered_dp_B.get();
                float*   d_ad  = use_A ? d_gathered_ad_A.get()   : d_gathered_ad_B.get();
                int32_t* d_off = use_A ? d_gather_offset_A.get() : d_gather_offset_B.get();
                int*     d_nc  = use_A ? d_snp_ncells_A.get()    : d_snp_ncells_B.get();

                // Pass 1: count qualifying cells of type t per SNP.
                const int count_smem = (DAESC_BLOCK_SIZE / 32) * (int)sizeof(int);
                detail::masked_snp_cell_count_kernel<<<cs, DAESC_BLOCK_SIZE, count_smem, stream2>>>(
                    d_nc,
                    dp_col_ptr, dp_row_idx, dp_vals,
                    d_cell_type.get(), t,
                    snp_start, cs, cfg.min_depth);

                // Pull ncells for this type+chunk to host once (≤40 KB scalar, setup xfer).
                cudaMemcpyAsync(h_ncells_chunk.as<int>(), d_nc,
                                cs * sizeof(int), cudaMemcpyDeviceToHost, stream2);
                cudaStreamSynchronize(stream2);

                // Host prefix-sum for type-masked gather offsets.
                std::vector<int32_t> offsets_t(cs + 1, 0);
                int total_quals_t = 0;
                for (int s = 0; s < cs; ++s) {
                    offsets_t[s]  = total_quals_t;
                    total_quals_t += h_ncells_chunk.as<int>()[s];
                }
                offsets_t[cs] = total_quals_t;
                if (total_quals_t > max_qual_per_chunk) {
                    total_quals_t = max_qual_per_chunk;
                    offsets_t[cs] = total_quals_t;
                    for (int s = cs - 1; s >= 0; --s)
                        if (offsets_t[s] > total_quals_t) offsets_t[s] = total_quals_t;
                }

                // Upload type-masked offsets (one-time per chunk+type setup xfer).
                cudaMemcpyAsync(d_off, offsets_t.data(),
                                (cs + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream2);

                // Pass 2: masked gather.
                detail::masked_snp_cell_gather_kernel<<<cs, DAESC_BLOCK_SIZE, sizeof(int), stream2>>>(
                    d_dp, d_ad,
                    dp_col_ptr, dp_row_idx, dp_vals,
                    ad_col_ptr, ad_row_idx, ad_vals,
                    d_off,
                    d_cell_type.get(), t,
                    snp_start, cs, cfg.min_depth);

                cudaStreamSynchronize(stream2);

                // Fisher MLE for this type's masked cells (stream 0).
                detail::daesc_mle_kernel<<<cs, DAESC_BLOCK_SIZE, mle_smem_bytes, ctx.stream>>>(
                    d_beta_chunk.get(), d_se_chunk.get(), d_phi_chunk.get(),
                    d_pval_chunk.get(),   // p_value: not stored for strat (only beta/se/phi)
                    d_conv_chunk.get(),   // convergence: also not stored for strat
                    d_dp, d_ad,
                    d_off, d_nc,
                    cs, cfg.max_iters, cfg.fisher_tol);

                // Write beta/se/phi results directly into the type-slice device arrays.
                // WHY direct device-to-device copy (no host round-trip):
                //   d_strat_* are on-device; d_beta/se/phi_chunk are on-device.
                //   cudaMemcpyAsync DeviceToDevice is a device-side DMA — zero PCIe.
                cudaMemcpyAsync(d_sb + snp_start, d_beta_chunk.get(),
                                cs * sizeof(float), cudaMemcpyDeviceToDevice, ctx.stream);
                cudaMemcpyAsync(d_ss + snp_start, d_se_chunk.get(),
                                cs * sizeof(float), cudaMemcpyDeviceToDevice, ctx.stream);
                cudaMemcpyAsync(d_sp + snp_start, d_phi_chunk.get(),
                                cs * sizeof(float), cudaMemcpyDeviceToDevice, ctx.stream);
            }
        }  // end type loop

        // Single DMA of all stratified results to host (final exit transfer).
        // Layout on device: [n_types × n_snps] row-major → maps directly to
        // result.strat_beta/se/phi which are also sized n_snps × n_types row-major (SNP major).
        // WHY single DMA: one contiguous transfer avoids per-type PCIe round-trips.
        cudaStreamSynchronize(ctx.stream);
        auto h_sb_pin = pin_pool.acquire(static_cast<size_t>(n_snps) * n_types * sizeof(float));
        auto h_ss_pin = pin_pool.acquire(static_cast<size_t>(n_snps) * n_types * sizeof(float));
        auto h_sp_pin = pin_pool.acquire(static_cast<size_t>(n_snps) * n_types * sizeof(float));

        cudaMemcpyAsync(h_sb_pin.as<float>(), d_strat_beta.get(),
                        static_cast<size_t>(n_snps) * n_types * sizeof(float),
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(h_ss_pin.as<float>(), d_strat_se.get(),
                        static_cast<size_t>(n_snps) * n_types * sizeof(float),
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(h_sp_pin.as<float>(), d_strat_phi.get(),
                        static_cast<size_t>(n_snps) * n_types * sizeof(float),
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaStreamSynchronize(ctx.stream);

        // Transpose from device layout [type × snp] to result layout [snp × type] (SNP major).
        // device row t, col s  →  result.strat_beta[s * n_types + t]
        // h_sb_pin layout: h_sb_pin[t * n_snps + s]
        for (int t = 0; t < n_types; ++t) {
            const float* src_b = h_sb_pin.as<float>() + static_cast<size_t>(t) * n_snps;
            const float* src_s = h_ss_pin.as<float>() + static_cast<size_t>(t) * n_snps;
            const float* src_p = h_sp_pin.as<float>() + static_cast<size_t>(t) * n_snps;
            for (int s = 0; s < n_snps; ++s) {
                result.strat_beta[static_cast<size_t>(s) * n_types + t] = src_b[s];
                result.strat_se  [static_cast<size_t>(s) * n_types + t] = src_s[s];
                result.strat_phi [static_cast<size_t>(s) * n_types + t] = src_p[s];
            }
        }
    }

    return result;
}

// ─── Compat free function for correctness harness ────────────────────────────
// Test-side signature: daesc(snp_ad, snp_dp, cell_type_dev, d_beta, d_se,
//   d_phi, d_pvalue, d_n_iters, cfg, stream).
// Writes results into caller-provided device output buffers.
// Uses run_daesc internally; downloads the DaescResult to fill the raw pointers.
inline void daesc(
    const core::DeviceCSC& snp_ad,
    const core::DeviceCSC& snp_dp,
    const int32_t*         cell_type_dev,   // device pointer, may be nullptr
    float*                 d_beta,
    float*                 d_se,
    float*                 d_phi,
    float*                 d_pvalue,
    int32_t*               d_n_iters,
    const DaescConfig&     cfg,
    cudaStream_t           stream)
{
    const int n_snps  = (int)snp_dp.cols;
    const int n_cells = (int)snp_dp.rows;

    // Download cell_type from device to host (run_daesc takes host pointer).
    std::vector<int32_t> h_cell_type;
    const int32_t* h_ct_ptr = nullptr;
    if (cell_type_dev != nullptr && cfg.n_cell_types > 0) {
        h_cell_type.resize(n_cells);
        cudaMemcpy(h_cell_type.data(), cell_type_dev,
                   (size_t)n_cells * sizeof(int32_t), cudaMemcpyDeviceToHost);
        h_ct_ptr = h_cell_type.data();
    }

    // Build real config (map compat fields → canonical fields).
    DaescConfig real_cfg;
    real_cfg.chunk_snps  = cfg.chunk_snps;
    real_cfg.max_iters   = cfg.max_iters;
    real_cfg.fisher_tol  = (cfg.tol != DAESC_FISHER_TOL) ? cfg.tol : cfg.fisher_tol;
    real_cfg.min_depth   = cfg.min_depth;
    real_cfg.min_cells   = cfg.min_cells;
    real_cfg.run_bh_fdr  = false;   // skip BH — test reads raw p-values
    real_cfg.fdr_alpha   = cfg.fdr_alpha;
    real_cfg.n_types     = (cfg.n_cell_types > 0) ? cfg.n_cell_types : cfg.n_types;

    core::GPUContext ctx;
    cudaStream_t stream2;
    cudaStreamCreate(&stream2);

    DaescResult res = run_daesc(snp_ad, snp_dp, h_ct_ptr, real_cfg, ctx, stream2);
    cudaStreamSynchronize(stream2);
    cudaStreamDestroy(stream2);

    // Upload results to caller's device buffers.
    const int n_out = (cfg.n_snps_limit > 0) ?
                      std::min(cfg.n_snps_limit, n_snps) : n_snps;
    if (d_beta)    cudaMemcpy(d_beta,    res.beta.data(),    (size_t)n_out * sizeof(float),   cudaMemcpyHostToDevice);
    if (d_se)      cudaMemcpy(d_se,      res.se.data(),      (size_t)n_out * sizeof(float),   cudaMemcpyHostToDevice);
    if (d_phi)     cudaMemcpy(d_phi,     res.phi.data(),     (size_t)n_out * sizeof(float),   cudaMemcpyHostToDevice);
    if (d_pvalue)  cudaMemcpy(d_pvalue,  res.p_value.data(), (size_t)n_out * sizeof(float),   cudaMemcpyHostToDevice);
    if (d_n_iters) {
        // converged is uint8; n_iters proxy = max_iters if not converged, else a smaller value.
        std::vector<int32_t> n_iters_host(n_out);
        for (int i = 0; i < n_out; ++i)
            n_iters_host[i] = res.converged[i] ? (real_cfg.max_iters - 1) : real_cfg.max_iters;
        cudaMemcpy(d_n_iters, n_iters_host.data(), (size_t)n_out * sizeof(int32_t), cudaMemcpyHostToDevice);
    }
    cudaStreamSynchronize(stream);
}

}  // namespace ase
}  // namespace singlet::gpu
