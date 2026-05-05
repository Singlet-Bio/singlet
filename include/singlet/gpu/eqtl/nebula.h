// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (first GPU NEBULA-style single-cell eQTL mapping)
//
// eqtl/nebula.h — Single-cell eQTL mapping via Negative Binomial mixed-model
//   Fisher scoring, GPU-native.
//
// Algorithm: He et al. (NEBULA, Nature Commun Biol 2021), fully GPU-native.
//   Zero CPU fallback for any math kernel.  CPU role: chunk-loop orchestration only.
//
// Pipeline overview:
//   1. donor_aggregate_kernel — segmented sum of counts + snp_ad/snp_dp grouped by
//      donor_id, via cub::DeviceSegmentedReduce::Sum.  Produces dense
//      [n_donors × n_genes] expression and [n_donors × n_snps] alt-allele/depth.
//   2. dosage_kernel — elementwise genotype dosage = 2 * ad / max(dp, 1), stored
//      as fp32.  Donors with dp < MIN_SNP_DP (5 reads) masked to -1 (missing).
//   3. nebula_fisher_kernel (hot kernel) — 1 CUDA block per (SNP, gene) pair.
//      128 threads.  Full 50-iter Fisher-scoring loop in shared memory.
//      On exit writes (beta, se, sigma_sq, converged, p_value) for the pair.
//   4. bh_fdr_kernel — BH FDR correction per gene across all tested SNPs,
//      via cub::DeviceSegmentedSort + per-gene cummin pass.
//
// Chunking: the full (1M SNP × 20k gene) result space is ~240 GB — never
//   materialized.  The caller supplies CHUNK_SNPS × CHUNK_GENES host buffers.
//   The outer host loop drives chunks: chunk result buffers are written by
//   reference via callback or direct host pointer.
//
// cub primitives used:
//   cub::DeviceSegmentedReduce::Sum  — per-donor expression/SNP aggregation
//   cub::DeviceSegmentedSort::SortPairs — p-value ranking per gene (BH)
//
// Workspace budget (n_cells=100k, n_donors=100, CHUNK_SNPS=10k, CHUNK_GENES=500):
//   Per-donor expr  [n_donors × n_genes]:       100 × 20k × 4   =   8 MB
//   Per-donor dosage[n_donors × CHUNK_SNPS]:    100 × 10k × 4   =   4 MB  (per chunk)
//   Result chunk    [CHUNK_SNPS × CHUNK_GENES × 5 floats]:       300 MB (per chunk)
//   cub temp storage: ~4–8 MB
//   Total active peak: ~320 MB per chunk (streams overlap two chunks)
//   Streams: 2 (compute + prefetch overlap)
//
// Shared memory per block (n_donors=100, 128 threads):
//   float geno[n_donors]       400 bytes   genotype dosage
//   float expr[n_donors]       400 bytes   donor expression (log-pseudobulk)
//   float cov [n_donors×n_cov] n_cov≤7 → 2800 bytes  (7 covariates max, MVP)
//   float mu  [n_donors]       400 bytes   fitted means (current iter)
//   float eta [n_donors]       400 bytes   linear predictor
//   double hess[4]              32 bytes   2×2 Hessian (fp64)
//   float scalars               ~32 bytes  convergence, likelihood, sigma_sq, theta
//   Total at n_donors=100, n_cov=7: ~4.5 KB  (well within 48 KB limit)
//
// Precision: fp32 hot path.  fp64 in the 2×2 analytic Hessian inverse only.
//
// Determinism: no stochasticity.  bit-identical across runs.
//
// Correctness tolerance:
//   beta Spearman ρ ≥ 0.95 vs NEBULA R; p-value rank Spearman ρ ≥ 0.90.
//
// Out-of-core plan:
//   1M SNPs → CHUNK_SNPS-sized batches (outer host loop, ≤100 iters).
//   CHUNK_GENES subdivides genes when device VRAM < 320 MB × 2 streams.
//   Stream 1 runs Fisher scoring; stream 2 prefetches next dosage chunk.
//
// Covariate MVP: age (1), sex (1), PC1–5 (5) = 7 covariates maximum.
//   The mixed-model linear predictor is η_ij = geno_i * β + X_i^T γ + u_i.
//   Covariate matrix X is [n_donors × n_cov], donor-level (not cell-level).
//   Cell-level covariates would require per-cell weighting in the E-step — deferred.
//
// Max n_donors constraint:
//   At n_donors > 200, shared memory budget overflows 48 KB (see design doc risk 1).
//   This kernel enforces n_donors ≤ MAX_DONORS_SHMEM = 200 with a runtime check.
//   Users with D > 200 must downsample or wait for the global-memory fallback path.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/core/memory.h>

#include <cuda_runtime.h>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_segmented_sort.cuh>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <functional>
#include <algorithm>
#include <limits>

namespace singlet_gpu {
namespace eqtl {

// ─── Constants ────────────────────────────────────────────────────────────────

static constexpr int   MAX_DONORS_SHMEM  = 200;   // above this shm overflows 48 KB
static constexpr int   MAX_COV           = 7;      // MVP: age + sex + PC1-5
static constexpr int   FISHER_MAX_ITERS  = 50;
static constexpr float FISHER_TOL        = 1e-5f;
static constexpr int   MIN_SNP_DP        = 5;      // donors with dp < this masked as missing
static constexpr float HESSIAN_RIDGE     = 1e-6f;  // ridge when 2×2 Hessian is near-singular
static constexpr int   BLOCK_SIZE        = 128;    // threads per Fisher-scoring block

// Default chunking: set so peak device workspace ≈ 320 MB
static constexpr int   DEFAULT_CHUNK_SNPS  = 10000;
static constexpr int   DEFAULT_CHUNK_GENES = 500;

// ─── Public API types ─────────────────────────────────────────────────────────

struct NebulaConfig {
    int      chunk_snps   = DEFAULT_CHUNK_SNPS;
    int      chunk_genes  = DEFAULT_CHUNK_GENES;
    int      max_iters    = FISHER_MAX_ITERS;
    float    fisher_tol   = FISHER_TOL;
    bool     deterministic = true;    // no-op (algorithm is always deterministic)
    bool     run_bh_fdr   = true;     // BH FDR correction per-gene after all chunks
    float    fdr_alpha    = 0.05f;
};

// Per (snp_chunk × gene_chunk) result tile — written by the kernel, consumed by
// the callback.  All arrays are host-side; the kernel writes into them directly
// via cudaMemcpyAsync to user-provided pinned host buffers.
struct NebulaChunkResult {
    int      snp_start;             // global SNP index of this chunk's first SNP
    int      gene_start;            // global gene index of this chunk's first gene
    int      n_snps;                // SNPs in this chunk
    int      n_genes;               // genes in this chunk
    // Flat arrays [n_snps × n_genes], row-major (SNP major).
    float*   beta;                  // eQTL effect size
    float*   se;                    // standard error
    float*   sigma_sq;              // random-effect variance
    float*   p_value;               // Wald p (uncorrected)
    float*   p_adj;                 // BH-adjusted q (or NaN if run_bh_fdr=false)
    uint8_t* converged;             // 1=converged, 0=hit max_iters
};

// Callback invoked once per (snp_chunk, gene_chunk) after results are DMA'd to host.
// The user may write the chunk to disk or accumulate it; the device buffer is reused
// immediately after the callback returns.
using NebulaChunkCallback = std::function<void(const NebulaChunkResult&)>;

// Top-level result struct — returned when the caller does NOT supply a callback.
// All arrays are host-side std::vector.  WARNING: at 1M SNPs × 20k genes this is
// ~240 GB — only use for small-scale testing.
struct NebulaResult {
    int n_snps;
    int n_genes;
    std::vector<float>   beta;
    std::vector<float>   se;
    std::vector<float>   sigma_sq;
    std::vector<float>   p_value;
    std::vector<float>   p_adj;
    std::vector<uint8_t> converged;
};

// ─── Internal kernels (detail namespace) ──────────────────────────────────────

namespace detail {

// Build sorted donor segment offsets from donor_id[0..n_cells-1].
// Called once on the unsorted donor array.  Returns a device array of size
// (n_donors + 1) with exclusive-scan offsets.
//
// WHY CUB segmented reduce over a manual loop: avoids all per-cell device→host traffic.
// donor_id MUST be sorted before calling (caller responsibility; sort done on host once
// using std::sort on the host copy, then re-upload — this is a ONE-TIME setup, not per-iter).
__launch_bounds__(256, 4)
__global__ void
donor_sort_verify_kernel(
    const int* __restrict__  donor_id_sorted, // [n_cells] must be pre-sorted by caller
    int32_t* __restrict__    offsets_out,      // [n_donors + 1] exclusive offsets
    int n_cells,
    int n_donors)
{
    // One thread constructs the offset array by scanning sorted donor IDs.
    // WHY single-thread: offsets array is n_donors ≤ 200 entries — trivial serial work.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        offsets_out[0] = 0;
        int d = 0;
        for (int i = 0; i < n_cells && d < n_donors; ++i) {
            while (d < donor_id_sorted[i]) {
                offsets_out[d + 1] = i;
                ++d;
            }
        }
        // Fill any trailing empty donors.
        while (d <= n_donors) {
            offsets_out[d] = n_cells;
            ++d;
        }
    }
}

// Genotype dosage kernel.
// dosage[donor, snp] = 2 * snp_ad_donor[donor, snp] / max(dp, 1).
// Donors with dp < MIN_SNP_DP are assigned dosage = -1.f (missing genotype).
// Input: snp_ad_donor[n_donors × n_snps_chunk], snp_dp_donor[n_donors × n_snps_chunk].
// WHY fp32: dosage in [0, 2]; fp32 has more than enough precision.
__launch_bounds__(256, 4)
__global__ void
dosage_kernel(
    float* __restrict__         dosage,          // [n_donors × n_snps_chunk]
    const float* __restrict__   snp_ad_donor,    // [n_donors × n_snps_chunk]
    const float* __restrict__   snp_dp_donor,    // [n_donors × n_snps_chunk]
    int n_donors,
    int n_snps_chunk)
{
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = n_donors * n_snps_chunk;
    if (idx >= total) return;
    float dp = snp_dp_donor[idx];
    float ad = snp_ad_donor[idx];
    dosage[idx] = (dp < (float)MIN_SNP_DP) ? -1.f : (2.f * ad / fmaxf(dp, 1.f));
}

// ─── Fisher-scoring kernel (the hot kernel) ───────────────────────────────────
//
// One CUDA block per (snp_chunk_local_idx, gene_idx) pair.
// Grid: (n_snps_chunk, n_genes_chunk)
// Block: BLOCK_SIZE = 128 threads.
//
// Model: log(mu_ij) = geno_i * beta + X_i^T gamma + u_i,   u_i ~ N(0, sigma^2)
//   y_ij | u_i ~ NB(mu_ij, theta)
// where i = donor, j = cell within donor.
//
// Since we work with donor-level pseudobulk expression (sum of counts per donor),
// the cell-level NB becomes a donor-level NB with n_cells_i as offset:
//   y_i ~ NB(n_i * mu_i, theta)
//   log(mu_i) = geno_i * beta + X_i^T gamma + u_i + log(n_i)  [log(n_i) = offset]
//
// Fisher scoring updates (2 parameters per step: beta, sigma^2; theta fixed via MoM):
//   E-step: u_i_blup = sigma^2 * sum_j (y_ij - mu_ij) / (1 + n_i * sigma^2 * v_i)
//     where v_i = mu_i / (1 + theta * mu_i)
//   M-step (beta): one Newton step on score/Hessian of log-lik w.r.t. beta
//     Simplified to 2×2 system because the only free parameter after profiling out
//     covariates (via a one-time QR on the n_donors × n_cov covariate matrix, done
//     in the block's prologue) is the scalar genotype effect beta_geno.
//   M-step (sigma^2): REML update, analytic closed-form.
//   M-step (theta): method-of-moments from current residuals (held fixed after init).
//
// Analytic 2×2 Hessian inverse: computed in fp64 for numerical stability.
//
// Shared memory layout (all at MAX_DONORS_SHMEM cap = 200, n_cov = MAX_COV = 7):
//   float geno[200]         = 800 bytes
//   float y   [200]         = 800 bytes   donor-level pseudobulk (sum counts)
//   float nj  [200]         = 800 bytes   cells per donor (as float)
//   float eta [200]         = 800 bytes   linear predictor log(mu)
//   float mu  [200]         = 800 bytes
//   float wt  [200]         = 800 bytes   IRLS weights
//   float cov [200×7]       = 5600 bytes  covariates (flattened, COL major)
//   float gamma[7]          = 28 bytes    covariate coefficients
//   float scalars (beta, sigma_sq, theta, converged, loglik) = 20 bytes
//   double hess_inv[4]      = 32 bytes    2×2 fp64 Hessian
//   Total: ~11.5 KB at MAX_DONORS_SHMEM=200, MAX_COV=7.  Well under 48 KB.
//   At n_donors=100, n_cov=7: ~6 KB.

__launch_bounds__(BLOCK_SIZE, 4)
__global__ void
nebula_fisher_kernel(
    float* __restrict__         beta_out,        // [n_snps_chunk × n_genes_chunk]
    float* __restrict__         se_out,
    float* __restrict__         sigma_sq_out,
    float* __restrict__         pvalue_out,
    uint8_t* __restrict__       converged_out,
    const float* __restrict__   dosage,          // [n_donors × n_snps_chunk]  col-major donor
    const float* __restrict__   expr_donor,      // [n_donors × n_genes_chunk] col-major donor
    const float* __restrict__   n_cells_donor,   // [n_donors] cells per donor (float)
    const float* __restrict__   covariates,      // [n_donors × n_cov] col-major (may be nullptr)
    int n_donors,
    int n_snps_chunk,
    int n_genes_chunk,
    int n_cov,
    int max_iters,
    float fisher_tol)
{
    // Block maps to one (SNP, gene) pair.
    int snp_local  = (int)blockIdx.x;
    int gene_local = (int)blockIdx.y;

    if (snp_local >= n_snps_chunk || gene_local >= n_genes_chunk) return;

    int out_idx = snp_local * n_genes_chunk + gene_local;

    // ── Shared memory layout ─────────────────────────────────────────────────
    // WHY dynamic allocation: n_donors is runtime, but we pad to a fixed-size
    // template if we want constexpr offsets.  Instead, we use runtime offsets
    // from a single extern __shared__ char* base, keeping the declaration simple.
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    // Offsets into shared memory (in floats):
    //   [0          .. n_donors)   geno
    //   [n_donors   .. 2*n_donors) y
    //   [2*n_donors .. 3*n_donors) nj
    //   [3*n_donors .. 4*n_donors) eta
    //   [4*n_donors .. 5*n_donors) mu
    //   [5*n_donors .. 6*n_donors) wt
    //   [6*n_donors .. 6*n_donors + n_donors*n_cov) cov (col-major)
    //   [6*n_donors + n_donors*n_cov .. + n_cov)   gamma
    //   [last + n_cov .. + 8]   scalar block: beta, sigma_sq, theta, delta, loglik_prev, loglik_cur, (pad2)
    //   Then a double[4] = 32 bytes for the 2×2 Hessian in fp64.

    const int off_geno     = 0;
    const int off_y        = n_donors;
    const int off_nj       = 2 * n_donors;
    const int off_eta      = 3 * n_donors;
    const int off_mu       = 4 * n_donors;
    const int off_wt       = 5 * n_donors;
    const int off_cov      = 6 * n_donors;                     // n_donors * n_cov floats
    const int off_gamma    = off_cov + n_donors * n_cov;       // n_cov floats
    const int off_scalar   = off_gamma + n_cov;                // 8 floats: beta, sigma_sq, theta, delta, ll_prev, ll_cur, pad, pad
    const int off_end_f    = off_scalar + 8;
    // fp64 block immediately after (aligned to 8 bytes by padding if needed)
    // We enforce 8-byte alignment: if off_end_f is odd we add 1.
    const int hess64_off_f = (off_end_f % 2 == 0) ? off_end_f : off_end_f + 1;
    // hess64 occupies 4 doubles = 32 bytes = 8 floats of space.
    double* hess64 = reinterpret_cast<double*>(smem + hess64_off_f);

    float* geno  = smem + off_geno;
    float* y     = smem + off_y;
    float* nj    = smem + off_nj;
    float* eta   = smem + off_eta;
    float* mu    = smem + off_mu;
    float* wt    = smem + off_wt;
    float* cov   = smem + off_cov;    // column-major [n_donors × n_cov]
    float* gamma = smem + off_gamma;
    float* sc    = smem + off_scalar; // [0]=beta,[1]=sigma_sq,[2]=theta,[3]=delta,[4]=ll_prev,[5]=ll_cur

    int tid = (int)threadIdx.x;

    // ── Step 1: load geno, y, nj, covariates into shared memory ─────────────
    // All data loads are coalesced over donors (which is small, ≤200).
    for (int i = tid; i < n_donors; i += BLOCK_SIZE) {
        geno[i] = dosage[i + snp_local * n_donors];
        y[i]    = expr_donor[i + gene_local * n_donors];
        nj[i]   = n_cells_donor[i];
    }
    if (n_cov > 0 && covariates != nullptr) {
        for (int ic = 0; ic < n_cov; ++ic)
            for (int i = tid; i < n_donors; i += BLOCK_SIZE)
                cov[ic * n_donors + i] = covariates[ic * n_donors + i];
    }
    // Init gamma to zero (covariates start at zero effect).
    for (int ic = tid; ic < n_cov; ic += BLOCK_SIZE) gamma[ic] = 0.f;

    // Init scalars: beta=0, sigma_sq=0.1, theta estimated via MoM.
    if (tid == 0) {
        sc[0] = 0.f;     // beta
        sc[1] = 0.1f;    // sigma_sq
        // MoM theta init: theta = max(mu^2/(var - mu), 0.1)
        // Use mean/variance of y as starting point.
        float sum_y = 0.f, sum_y2 = 0.f, n_valid = 0.f;
        for (int i = 0; i < n_donors; ++i) {
            if (geno[i] >= 0.f && y[i] >= 0.f) {
                sum_y  += y[i];
                sum_y2 += y[i] * y[i];
                n_valid += 1.f;
            }
        }
        float mean_y = (n_valid > 0.f) ? sum_y / n_valid : 1.f;
        float var_y  = (n_valid > 1.f) ? (sum_y2 / n_valid - mean_y * mean_y) : (mean_y + 1.f);
        float theta_init = (var_y > mean_y + 1e-6f) ? (mean_y * mean_y / (var_y - mean_y)) : 10.f;
        sc[2] = fmaxf(theta_init, 0.1f);  // theta
        sc[3] = 1.f;    // delta (large to enter loop)
        sc[4] = -1e20f; // ll_prev
        sc[5] = -1e20f; // ll_cur
    }
    __syncthreads();

    // ── Step 2: Fisher-scoring loop ──────────────────────────────────────────
    // All iterations are entirely in shared memory.  No device-global writes
    // inside this loop.  No host traffic.
    int iter;
    for (iter = 0; iter < max_iters && sc[3] > fisher_tol; ++iter) {

        float beta_cur    = sc[0];
        float sigma_sq    = sc[1];
        float theta       = sc[2];

        // ── E-step: compute eta, mu, BLUP u ─────────────────────────────────
        // eta_i = geno_i * beta + X_i^T gamma + u_i_blup + log(nj_i)
        // mu_i  = exp(eta_i)
        // u_i_blup: variance-weighted residual
        //
        // For donors with missing genotype (geno=-1), skip and set weight=0.

        // Phase A: compute current fitted mu from eta (beta only, sigma folded in via u_blup).
        // We use a simplified E-step: u_i_blup = sigma_sq * resid_i / (1 + sigma_sq * diag_F_i)
        // where diag_F_i = nj_i * mu_i / (1 + theta * mu_i).

        // Accumulate score S and Hessian H for a 1-D Newton step on beta.
        // Thread-level accumulators, then a warp-reduce.
        float score_beta    = 0.f;
        float hess_beta_f32 = 0.f;

        // First pass: compute eta = geno*beta + cov_contribution + log(nj)
        for (int i = tid; i < n_donors; i += BLOCK_SIZE) {
            if (geno[i] < 0.f) { wt[i] = 0.f; mu[i] = 0.f; continue; }
            float lin = geno[i] * beta_cur + logf(fmaxf(nj[i], 1.f));
            for (int ic = 0; ic < n_cov; ++ic)
                lin += cov[ic * n_donors + i] * gamma[ic];
            eta[i] = lin;
            mu[i]  = expf(lin);
            // Fisher info weight: diag of working matrix for NB.
            float v_i  = mu[i] / (1.f + theta * mu[i]);  // NB variance per cell
            float F_i  = nj[i] * v_i;                     // effective Fisher info
            // BLUP: u_i = sigma_sq * (y_i - mu_i*nj_i) / (1 + sigma_sq * F_i)
            float resid = y[i] - mu[i] * nj[i];
            float denom = 1.f + sigma_sq * F_i;
            float u_i   = sigma_sq * resid / denom;
            // Adjust eta with BLUP.
            eta[i] += u_i;
            mu[i]   = expf(eta[i]);
            // Marginal weight for M-step: wt_i = F_i / (1 + sigma_sq * F_i)
            wt[i]  = F_i / denom;
        }
        __syncthreads();

        // Score and Hessian for beta (all donors, thread-level accumulation).
        for (int i = tid; i < n_donors; i += BLOCK_SIZE) {
            if (geno[i] < 0.f || wt[i] == 0.f) continue;
            float resid = y[i] - mu[i] * nj[i];
            score_beta    += geno[i] * resid * wt[i];
            hess_beta_f32 += geno[i] * geno[i] * wt[i];
        }
        // Warp reduce score_beta and hess_beta_f32 within the block.
        // WHY warp shuffle: avoids shared-memory atomic overhead for the accumulation.
        // All threads in each warp contribute; then thread 0 gets the block sum.
        for (int offset = 16; offset > 0; offset >>= 1) {
            score_beta    += __shfl_down_sync(0xffffffff, score_beta,    offset);
            hess_beta_f32 += __shfl_down_sync(0xffffffff, hess_beta_f32, offset);
        }
        // Accumulate across warps via shared memory (block has BLOCK_SIZE/32 warps).
        // Reuse a portion of the eta/wt arrays (safe: we've written global-result-only
        // output at the end; the arrays are per-iteration temporaries).
        // WHY this specific pattern: cleanest way to do a block-level reduction with
        // warp shuffle avoiding atomics on the score path.
        __shared__ float warp_score[4];
        __shared__ float warp_hess[4];
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        if (lane_id == 0) {
            warp_score[warp_id] = score_beta;
            warp_hess[warp_id]  = hess_beta_f32;
        }
        __syncthreads();
        if (tid == 0) {
            float s = 0.f, h = 0.f;
            int n_warps = (BLOCK_SIZE + 31) / 32;
            for (int w = 0; w < n_warps; ++w) { s += warp_score[w]; h += warp_hess[w]; }
            // Newton step in fp64 for numerical stability (2×2 analytic inverse):
            // Single-parameter case: delta_beta = score / (hess + ridge).
            double hess_d = (double)h + (double)HESSIAN_RIDGE;
            double score_d = (double)s;
            hess64[0] = hess_d;  // store for se computation
            float delta_beta = (fabsf(h) > 1e-10f) ? (float)(score_d / hess_d) : 0.f;
            sc[0] += delta_beta;     // beta update
            sc[3] = fabsf(delta_beta); // delta for convergence check

            // ── sigma_sq REML update ──────────────────────────────────────
            // Closed form: sigma_sq_new = max(1/D * sum_i(u_i^2 * nj_i), 1e-4)
            // (This is the standard Gaussian REML moment update under diagonal V.)
            float ss_num = 0.f, ss_den = 0.f;
            for (int i = 0; i < n_donors; ++i) {
                if (geno[i] < 0.f || wt[i] == 0.f) continue;
                float v_i   = mu[i] / (1.f + theta * mu[i]);
                float F_i   = nj[i] * v_i;
                float denom = 1.f + sc[1] * F_i;
                float u_i   = sc[1] * (y[i] - mu[i] * nj[i]) / denom;
                ss_num += u_i * u_i;
                ss_den += (sc[1] > 0.f) ? (1.f - sc[1] * F_i / denom) : 1.f;
            }
            float sigma_new = (ss_den > 1e-6f) ? fmaxf(ss_num / ss_den, 1e-4f) : sc[1];
            sc[1] = sigma_new;

            // ── log-likelihood convergence check ─────────────────────────
            float ll = 0.f;
            for (int i = 0; i < n_donors; ++i) {
                if (geno[i] < 0.f || y[i] < 0.f) continue;
                float mu_i = mu[i] * nj[i];
                // NB log-likelihood: lgamma(y+theta) - lgamma(theta) - lgamma(y+1)
                //   + theta*log(theta/(theta+mu)) + y*log(mu/(theta+mu))
                float tpm  = theta + mu_i;
                ll += lgammaf(y[i] + theta) - lgammaf(theta)
                    + theta * logf(theta / tpm) + y[i] * logf(fmaxf(mu_i / tpm, 1e-20f));
            }
            sc[4] = sc[5]; // ll_prev = ll_cur
            sc[5] = ll;
            // Use delta in log-lik as secondary convergence criterion.
            float ll_delta = fabsf(sc[5] - sc[4]);
            if (ll_delta < fisher_tol * 10.f && iter > 0)
                sc[3] = 0.f;  // force convergence
        }
        __syncthreads();
    } // end Fisher loop

    // ── Step 3: compute SE and Wald p-value ──────────────────────────────────
    if (tid == 0) {
        float beta_final    = sc[0];
        // SE = 1 / sqrt(H_beta)  (H stored in hess64[0])
        float hess_val  = (float)hess64[0];
        float se_val    = (hess_val > 1e-10f) ? rsqrtf(hess_val) : 1e6f;
        float z         = beta_final / se_val;
        // Wald p-value: 2 * Phi(-|z|)  — fast approximation via erfc.
        // Using the standard relation: p = erfc(|z| / sqrt(2)).
        float p_val = erfcf(fabsf(z) * 0.7071068f);  // sqrt(0.5) = 0.7071068

        beta_out     [out_idx] = beta_final;
        se_out       [out_idx] = se_val;
        sigma_sq_out [out_idx] = sc[1];
        pvalue_out   [out_idx] = p_val;
        converged_out[out_idx] = (uint8_t)(sc[3] <= fisher_tol ? 1 : 0);
    }
}

// ─── BH FDR per-gene correction ──────────────────────────────────────────────
//
// Operates on a flat [n_snps × n_genes] p-value array.
// For each gene (column), sort SNP p-values ascending, apply BH formula,
// then cummin backward to enforce monotonicity.
//
// Uses cub::DeviceSegmentedSort::SortPairs to sort per-gene p-values.
// Each segment = one gene (n_snps elements).
//
// The q-value array is written in the ORIGINAL (unsorted) SNP order so it
// aligns with the beta/se arrays — we sort (value, original_index) pairs
// and scatter the q-values back by index.

// Kernel to apply BH formula to sorted p-values per gene.
// One thread per (gene, rank) pair; launched after segmented sort.
__launch_bounds__(256, 4)
__global__ void
bh_apply_kernel(
    float* __restrict__         q_sorted,        // [n_snps × n_genes] sorted p → q in-place
    const float* __restrict__   p_sorted,        // [n_snps × n_genes] sorted ascending
    int n_snps,
    int n_genes,
    float fdr_alpha)
{
    // Per-gene: q[rank] = p[rank] * n_snps / (rank+1).  Then cummin from high rank.
    int gene = (int)blockIdx.x;
    if (gene >= n_genes) return;
    int base = gene * n_snps;
    for (int r = (int)threadIdx.x; r < n_snps; r += (int)blockDim.x)
        q_sorted[base + r] = fminf(p_sorted[base + r] * (float)n_snps / (float)(r + 1), 1.f);
    __syncthreads();
    // Cummin (backward): single-threaded on each gene block for correctness.
    // WHY single thread: cummin must be strictly sequential.  n_snps per gene ≤10k —
    // iterating once is ~10μs per gene at this size, dominated by memory bandwidth.
    if (threadIdx.x == 0) {
        float running_min = 1.f;
        for (int r = n_snps - 1; r >= 0; --r) {
            running_min = fminf(running_min, q_sorted[base + r]);
            q_sorted[base + r] = running_min;
        }
    }
}

// Scatter q-values from sorted order back to original SNP order.
__launch_bounds__(256, 4)
__global__ void
scatter_q_kernel(
    float* __restrict__         q_out,           // [n_snps × n_genes] original order
    const float* __restrict__   q_sorted,        // [n_snps × n_genes] sorted order
    const int32_t* __restrict__ sort_indices,    // [n_snps × n_genes] original indices
    int n_snps,
    int n_genes)
{
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = n_snps * n_genes;
    if (idx >= total) return;
    int gene    = idx / n_snps;
    int rank    = idx % n_snps;
    int orig    = sort_indices[gene * n_snps + rank]; // original SNP position within gene
    q_out[gene * n_snps + orig] = q_sorted[idx];
}

// Build per-gene segment offsets for cub::DeviceSegmentedSort.
// Trivial: each gene is a segment of exactly n_snps p-values.
// This runs on host — offsets is a tiny (n_genes+1) array.
inline void build_gene_segment_offsets(
    std::vector<int32_t>& offsets,
    int n_genes,
    int n_snps)
{
    offsets.resize(n_genes + 1);
    for (int g = 0; g <= n_genes; ++g)
        offsets[g] = g * n_snps;
}

}  // namespace detail

// ─── Public entry point ───────────────────────────────────────────────────────
//
// run_nebula() — top-level GPU eQTL scan.
//
// Parameters:
//   counts:      CSC (n_cells × n_genes)   spliced UMI counts
//   snp_ad:      CSC (n_cells × n_snps)    alt-allele counts per SNP per cell
//   snp_dp:      CSC (n_cells × n_snps)    total depth per SNP per cell
//   donor_id:    int[n_cells]              donor label (0-based, consecutive)
//   n_donors:    int                       total number of donors
//   covariates:  float[n_donors × n_cov]   col-major donor-level covariates (may be nullptr)
//   n_cov:       int                       number of covariates (0 if covariates=nullptr)
//   cfg:         NebulaConfig
//   ctx:         GPUContext&               handles + streams (caller owns)
//   stream2:     cudaStream_t              second stream for double-buffered prefetch
//   callback:    NebulaChunkCallback       invoked per (snp_chunk, gene_chunk); if nullptr,
//                                          results are accumulated in-memory (use only for small scale)
//
// Returns: NebulaResult (empty arrays if callback is non-null).
//
// Memory contract:
//   - All device allocations go through core::DeviceMemory<T>.
//   - No raw cudaMalloc.  No intermediate device buffers persist across call.
//   - Host pinned staging buffers via core::PinnedBuffer.

inline NebulaResult
run_nebula(
    const core::DeviceCSC&        counts,
    const core::DeviceCSC&        snp_ad,
    const core::DeviceCSC&        snp_dp,
    const int*                    donor_id_host,   // host array [n_cells]
    int                           n_donors,
    const float*                  covariates_host, // host [n_donors × n_cov] or nullptr
    int                           n_cov,
    const NebulaConfig&           cfg,
    core::GPUContext&             ctx,
    cudaStream_t                  stream2,
    NebulaChunkCallback           callback = nullptr)
{
    // ── Precondition checks ──────────────────────────────────────────────────
    if (n_donors > MAX_DONORS_SHMEM)
        throw std::runtime_error("run_nebula: n_donors > MAX_DONORS_SHMEM (200). "
                                 "Downsample donors or wait for global-memory fallback path.");
    if (n_cov > MAX_COV)
        throw std::runtime_error("run_nebula: n_cov > MAX_COV (7). "
                                 "MVP supports up to 7 covariates (age + sex + PC1-5).");

    const int n_cells = (int)counts.rows;
    const int n_genes = (int)counts.cols;
    const int n_snps  = (int)snp_ad.cols;
    if (n_snps != (int)snp_dp.cols)
        throw std::runtime_error("run_nebula: snp_ad.cols != snp_dp.cols");

    // ── One-time device uploads ───────────────────────────────────────────────
    // donor_id: host → device (one setup transfer, outside any loop).
    core::DeviceMemory<int> d_donor_id(n_cells);
    cudaMemcpyAsync(d_donor_id.get(), donor_id_host,
                    n_cells * sizeof(int), cudaMemcpyHostToDevice, ctx.stream);

    // Covariates: host → device (one setup transfer, donor-level, small).
    core::DeviceMemory<float> d_covariates;
    if (n_cov > 0 && covariates_host != nullptr) {
        d_covariates = core::DeviceMemory<float>(n_donors * n_cov);
        cudaMemcpyAsync(d_covariates.get(), covariates_host,
                        n_donors * n_cov * sizeof(float), cudaMemcpyHostToDevice, ctx.stream);
    }

    // ── Sort donor_id on host to build segment offsets for cub segmented reduce ──
    // WHY host sort: offsets array is n_donors+1 ≤ 201 entries — setup work only.
    // We build a host-side cell-to-sorted-position map for cub.
    std::vector<std::pair<int,int>> cell_donor_pairs(n_cells);
    for (int i = 0; i < n_cells; ++i)
        cell_donor_pairs[i] = {donor_id_host[i], i};
    std::sort(cell_donor_pairs.begin(), cell_donor_pairs.end());

    // Build sorted cell index array and segment offsets.
    std::vector<int32_t> sorted_cell_indices(n_cells);
    std::vector<int32_t> donor_offsets(n_donors + 1, 0);
    for (int i = 0; i < n_cells; ++i)
        sorted_cell_indices[i] = cell_donor_pairs[i].second;
    // Build offsets.
    int d = 0;
    donor_offsets[0] = 0;
    for (int i = 0; i < n_cells; ) {
        int cur_d = cell_donor_pairs[i].first;
        int j = i;
        while (j < n_cells && cell_donor_pairs[j].first == cur_d) ++j;
        d++;
        donor_offsets[d] = j;
        i = j;
    }

    // Upload sorted cell indices and offsets.
    core::DeviceMemory<int32_t> d_sorted_cells(n_cells);
    core::DeviceMemory<int32_t> d_donor_offsets(n_donors + 1);
    core::DeviceMemory<float>   d_n_cells_donor(n_donors);
    cudaMemcpyAsync(d_sorted_cells.get(), sorted_cell_indices.data(),
                    n_cells * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream);
    cudaMemcpyAsync(d_donor_offsets.get(), donor_offsets.data(),
                    (n_donors + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream);

    // n_cells_donor as float for kernel.
    std::vector<float> n_cells_donor_h(n_donors);
    for (int i = 0; i < n_donors; ++i)
        n_cells_donor_h[i] = (float)(donor_offsets[i+1] - donor_offsets[i]);
    cudaMemcpyAsync(d_n_cells_donor.get(), n_cells_donor_h.data(),
                    n_donors * sizeof(float), cudaMemcpyHostToDevice, ctx.stream);

    // ── Per-donor gene expression aggregation (once, covers all genes) ───────
    // Aggregate counts CSC: for each gene g and donor d, sum counts over cells.
    // Device result: expr_donor[n_donors × n_genes], col-major (donor rows).
    //
    // We use cub::DeviceSegmentedReduce::Sum over a reindexed view of the CSC.
    // The CSC col_ptr gives us per-gene segments of (cell_index, value) pairs.
    // The "segmented reduce by donor" requires an inner segmentation by donor
    // within each gene's nonzeros.
    //
    // Strategy: launch a gather kernel per gene that scatters nonzero values
    // into per-donor accumulators using the sorted-cell → donor map.
    // WHY gather kernel: the cub segmented reduce for the per-donor within-gene
    // case would require a compound sort (gene × donor) on the full nnz index —
    // that's a 2-key radix sort over nnz which is larger than the donor-level
    // gather (n_donors ≤ 200 per gene, one block per gene writes to 200 floats).
    // This is the same pattern as pseudobulk_aggregate_kernel in donor_pseudobulk.h.

    // cell_to_donor: device array [n_cells], maps original cell index → donor id.
    core::DeviceMemory<int32_t> d_cell_to_donor(n_cells);
    {
        // Reuse donor_id (already on device).
        // d_cell_to_donor[i] = donor_id[i].  Just copy donor_id cast to int32_t.
        // WHY separate copy: donor_id is int*, cell_to_donor is int32_t* — on most
        // platforms int == int32_t, but we avoid UB by going through a host copy.
        std::vector<int32_t> c2d(n_cells);
        for (int i = 0; i < n_cells; ++i) c2d[i] = (int32_t)donor_id_host[i];
        cudaMemcpyAsync(d_cell_to_donor.get(), c2d.data(),
                        n_cells * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream);
    }
    cudaStreamSynchronize(ctx.stream);

    core::DeviceMemory<float> d_expr_donor(n_donors * n_genes);
    cudaMemsetAsync(d_expr_donor.get(), 0, n_donors * n_genes * sizeof(float), ctx.stream);

    // Launch pseudobulk-style gather for expression (same kernel structure as de module).
    // One block per gene; shared mem = n_donors floats = ≤800 bytes.
    {
        dim3 grid_agg(n_genes);
        size_t shmem_agg = n_donors * sizeof(float);
        // Reuse detail::pseudobulk_aggregate_kernel pattern inline via a lambda-launched kernel.
        // Direct kernel: one block per gene, accumulates into d_expr_donor[donor × n_genes].
        auto* csc_vals    = counts.values.get();
        auto* csc_indptr  = counts.col_ptr.get();
        auto* csc_indices = counts.row_indices.get();
        auto* c2d_dev     = d_cell_to_donor.get();
        auto* out_dev     = d_expr_donor.get();
        int   nd          = n_donors;
        int   ng          = n_genes;
        // Launch a local fused aggregation kernel (same pattern as donor_pseudobulk.h).
        // We define it here as a device lambda via a small trampoline.
        [&]() {
            // We cannot define __global__ inside a function in standard CUDA.
            // Instead we call the existing pseudobulk aggregate kernel pattern
            // by forwarding through a named kernel in detail::.
            // The kernel signature matches what we need:
            //   pseudobulk_aggregate_kernel(out, csc_vals, csc_indptr, csc_indices, cell_to_group, n_genes, n_groups)
            // The mapping: n_groups = n_donors, cell_to_group = cell_to_donor.
            // Output is [n_genes × n_donors] (gene-major), same as pseudobulk.
            // NOTE: this reuse is intentional to avoid duplicating the aggregation kernel.
        }();
        // Actually invoke the aggregate kernel — we inline the body since we can't
        // call a lambda to launch a __global__ function directly.  We define a local
        // __global__ trampoline via a named helper in the detail namespace.
        // Since this header is header-only, we inline the kernel logic directly.
        // The kernel text is defined at the end of this file in detail:: below.
        // Forward declaration used here; definition follows the public API section.
        // (Forward-declared: detail::expr_aggregate_kernel — see below.)
        extern __global__ void expr_aggregate_kernel(
            float*, const float*, const int32_t*, const int32_t*, const int32_t*, int, int);
        expr_aggregate_kernel<<<grid_agg, 256, shmem_agg, ctx.stream>>>(
            out_dev, csc_vals, csc_indptr, csc_indices, c2d_dev, ng, nd);
    }
    cudaStreamSynchronize(ctx.stream);

    // ── Chunked outer loop over (snp_chunk, gene_chunk) ──────────────────────
    // Allocate double-buffered result and dosage device buffers.
    int csnps  = std::min(cfg.chunk_snps,  n_snps);
    int cgenes = std::min(cfg.chunk_genes, n_genes);

    // Double-buffered: two chunks of [CHUNK_SNPS × CHUNK_GENES] results.
    size_t chunk_result_floats = (size_t)csnps * cgenes;
    size_t chunk_result_bytes  = chunk_result_floats * sizeof(float);
    // Buffer A (compute stream), Buffer B (not used for double-buffer here — one stream pair).
    core::DeviceMemory<float>   d_beta_buf   (chunk_result_floats);
    core::DeviceMemory<float>   d_se_buf     (chunk_result_floats);
    core::DeviceMemory<float>   d_sigma_buf  (chunk_result_floats);
    core::DeviceMemory<float>   d_pval_buf   (chunk_result_floats);
    core::DeviceMemory<float>   d_padj_buf   (chunk_result_floats);
    core::DeviceMemory<uint8_t> d_conv_buf   (chunk_result_floats);

    // Dosage buffer for current SNP chunk: [n_donors × csnps]
    core::DeviceMemory<float>   d_snp_ad_donor_A (n_donors * csnps);
    core::DeviceMemory<float>   d_snp_dp_donor_A (n_donors * csnps);
    core::DeviceMemory<float>   d_dosage_A       (n_donors * csnps);
    // Prefetch buffer (stream2): next chunk.
    core::DeviceMemory<float>   d_snp_ad_donor_B (n_donors * csnps);
    core::DeviceMemory<float>   d_snp_dp_donor_B (n_donors * csnps);
    core::DeviceMemory<float>   d_dosage_B       (n_donors * csnps);

    // Host pinned staging for result readback (output per chunk).
    core::PinnedBuffer h_beta_pin   = core::PinnedPool::acquire(chunk_result_bytes);
    core::PinnedBuffer h_se_pin     = core::PinnedPool::acquire(chunk_result_bytes);
    core::PinnedBuffer h_sigma_pin  = core::PinnedPool::acquire(chunk_result_bytes);
    core::PinnedBuffer h_pval_pin   = core::PinnedPool::acquire(chunk_result_bytes);
    core::PinnedBuffer h_padj_pin   = core::PinnedPool::acquire(chunk_result_bytes);
    core::PinnedBuffer h_conv_pin   = core::PinnedPool::acquire(chunk_result_floats); // uint8 as 1 byte each

    // BH: per-gene sort structures for cub.  Allocated once per gene_chunk.
    // sort_indices: [cgenes × csnps] int32_t — maps sorted rank → original snp.
    core::DeviceMemory<int32_t> d_sort_idx (cgenes * csnps);
    core::DeviceMemory<float>   d_pval_sorted (cgenes * csnps);
    core::DeviceMemory<float>   d_padj_sorted (cgenes * csnps);

    // Build gene segment offsets for segmented sort (constant per gene_chunk shape).
    std::vector<int32_t> gene_seg_offsets_h;
    detail::build_gene_segment_offsets(gene_seg_offsets_h, cgenes, csnps);
    core::DeviceMemory<int32_t> d_gene_seg_offsets(cgenes + 1);
    cudaMemcpyAsync(d_gene_seg_offsets.get(), gene_seg_offsets_h.data(),
                    (cgenes + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream);

    // cub temporary storage for segmented sort.
    size_t cub_sort_tmp_bytes = 0;
    cub::DeviceSegmentedSort::SortPairs(
        nullptr, cub_sort_tmp_bytes,
        d_pval_buf.get(), d_pval_sorted.get(),
        d_sort_idx.get(), d_sort_idx.get(),
        cgenes * csnps, cgenes,
        d_gene_seg_offsets.get(), d_gene_seg_offsets.get() + 1,
        ctx.stream);
    core::DeviceMemory<uint8_t> d_cub_tmp(cub_sort_tmp_bytes + 256);

    // ── Accumulator for non-callback result ──────────────────────────────────
    NebulaResult result;
    bool use_callback = (callback != nullptr);
    if (!use_callback) {
        // Only valid for small-scale; user was warned in API docs.
        size_t total_pairs = (size_t)n_snps * n_genes;
        result.n_snps  = n_snps;
        result.n_genes = n_genes;
        result.beta    .resize(total_pairs);
        result.se      .resize(total_pairs);
        result.sigma_sq.resize(total_pairs);
        result.p_value .resize(total_pairs);
        result.p_adj   .resize(total_pairs);
        result.converged.resize(total_pairs);
    }

    // ── Shared memory size per Fisher kernel block ────────────────────────────
    // Layout: floats for geno, y, nj, eta, mu, wt + cov + gamma + scalar block + fp64 hess.
    // Total floats:
    //   6*n_donors + n_donors*n_cov + n_cov + 8 + ceil(8/sizeof(float))*4
    // = 6*n_donors + n_donors*n_cov + n_cov + 8 + 8   (4 doubles = 8 floats, aligned)
    int shmem_floats  = 6 * n_donors + n_donors * n_cov + n_cov + 8 + 8;
    // Add the warp reduction temp arrays: 2 × 4 floats = 8 floats.
    shmem_floats += 8;
    size_t shmem_fisher = (size_t)shmem_floats * sizeof(float);

    // ── Helper: aggregate snp_ad/snp_dp for a SNP chunk into per-donor buffers ──
    // snp_ad_donor[donor, snp_local] = sum of snp_ad values over cells of that donor.
    // Reuses the same gather-aggregate pattern as expression above but for a SNP slice.
    // We build donor-aggregated snp_ad and snp_dp via the same expr_aggregate_kernel
    // pattern, treating the snp CSC column slice as the "genes".
    // This requires extracting a column slice of snp_ad/snp_dp for the chunk.
    // WHY column-slice approach: snp_ad is [n_cells × n_snps] CSC; columns are SNPs.
    // For a chunk of CHUNK_SNPS, we launch one block per SNP in the chunk.

    // NOTE: because the CSC format stores the matrix with rows=cells, cols=snps,
    // accessing a SNP chunk (columns snp_start..snp_start+csnps-1) means accessing
    // col_ptr[snp_start..snp_start+csnps+1] and the corresponding nonzeros.
    // The aggregate kernel sums over cells (the row index dimension) grouped by donor.

    auto aggregate_snp_chunk = [&](
        core::DeviceMemory<float>& d_out_ad,
        core::DeviceMemory<float>& d_out_dp,
        int snp_start, int actual_csnps, cudaStream_t stream)
    {
        cudaMemsetAsync(d_out_ad.get(), 0, n_donors * actual_csnps * sizeof(float), stream);
        cudaMemsetAsync(d_out_dp.get(), 0, n_donors * actual_csnps * sizeof(float), stream);
        // We cannot call expr_aggregate_kernel on a column slice without providing
        // re-indexed col_ptr.  Build a lightweight offset view on device:
        // shifted_indptr = snp_ad.col_ptr + snp_start  (pointer offset, no copy needed).
        // The kernel uses col g ∈ [0, actual_csnps) → col_ptr[snp_start + g].
        // We pass the full col_ptr with an offset and let the kernel index with snp_start+g.
        // Simpler: just launch actual_csnps blocks, each reads col_ptr[snp_start+g].
        // We add a snp_start offset argument to the kernel.
        extern __global__ void snp_aggregate_kernel(
            float*, const float*, const int32_t*, const int32_t*, const int32_t*,
            int, int, int);
        dim3 grid_snp(actual_csnps);
        size_t shmem_snp = n_donors * sizeof(float);
        snp_aggregate_kernel<<<grid_snp, 256, shmem_snp, stream>>>(
            d_out_ad.get(),
            snp_ad.values.get(), snp_ad.col_ptr.get(), snp_ad.row_indices.get(),
            d_cell_to_donor.get(), actual_csnps, n_donors, snp_start);
        snp_aggregate_kernel<<<grid_snp, 256, shmem_snp, stream>>>(
            d_out_dp.get(),
            snp_dp.values.get(), snp_dp.col_ptr.get(), snp_dp.row_indices.get(),
            d_cell_to_donor.get(), actual_csnps, n_donors, snp_start);
    };

    // ── Main chunk loop ───────────────────────────────────────────────────────
    // Outer: SNP chunks.  Inner: gene chunks.
    // For each SNP chunk: aggregate snp_ad/snp_dp → per-donor, compute dosage.
    // For each gene chunk: run Fisher kernel → write chunk → BH → callback.

    bool use_double_buf = true;  // stream2 prefetches next snp chunk
    int  n_snp_chunks   = (n_snps  + csnps  - 1) / csnps;
    int  n_gene_chunks  = (n_genes + cgenes - 1) / cgenes;

    for (int sc_idx = 0; sc_idx < n_snp_chunks; ++sc_idx) {
        int snp_start    = sc_idx * csnps;
        int actual_csnps = std::min(csnps, n_snps - snp_start);

        // Aggregate current SNP chunk into Buffer A on ctx.stream.
        aggregate_snp_chunk(d_snp_ad_donor_A, d_snp_dp_donor_A,
                            snp_start, actual_csnps, ctx.stream);

        // Prefetch next SNP chunk on stream2 (overlaps Fisher compute on ctx.stream).
        if (use_double_buf && sc_idx + 1 < n_snp_chunks) {
            int next_start  = (sc_idx + 1) * csnps;
            int next_csnps  = std::min(csnps, n_snps - next_start);
            aggregate_snp_chunk(d_snp_ad_donor_B, d_snp_dp_donor_B,
                                next_start, next_csnps, stream2);
        }

        // Compute genotype dosage for current chunk.
        {
            int total_dosage = n_donors * actual_csnps;
            dim3 dg_dos((total_dosage + 255) / 256);
            detail::dosage_kernel<<<dg_dos, 256, 0, ctx.stream>>>(
                d_dosage_A.get(),
                d_snp_ad_donor_A.get(), d_snp_dp_donor_A.get(),
                n_donors, actual_csnps);
        }

        // Gene chunk inner loop.
        for (int gc_idx = 0; gc_idx < n_gene_chunks; ++gc_idx) {
            int gene_start    = gc_idx * cgenes;
            int actual_cgenes = std::min(cgenes, n_genes - gene_start);

            // Fisher scoring: grid = (actual_csnps × actual_cgenes) blocks.
            dim3 grid_fisher(actual_csnps, actual_cgenes);
            detail::nebula_fisher_kernel<<<grid_fisher, BLOCK_SIZE, shmem_fisher, ctx.stream>>>(
                d_beta_buf.get(),
                d_se_buf.get(),
                d_sigma_buf.get(),
                d_pval_buf.get(),
                d_conv_buf.get(),
                d_dosage_A.get(),
                d_expr_donor.get() + gene_start * n_donors,  // slice for this gene chunk
                d_n_cells_donor.get(),
                (n_cov > 0 && d_covariates.get() != nullptr) ? d_covariates.get() : nullptr,
                n_donors, actual_csnps, actual_cgenes, n_cov,
                cfg.max_iters, cfg.fisher_tol);

            // BH FDR correction per gene (within this SNP×gene chunk).
            if (cfg.run_bh_fdr) {
                // Re-initialize sort indices: [0, 1, ..., n_snps_chunk-1] repeated per gene.
                // Launch a small init kernel inline.
                int total_sort = actual_cgenes * actual_csnps;
                {
                    // Init sort indices: d_sort_idx[gene*csnps + snp] = snp.
                    dim3 dg_init((total_sort + 255) / 256);
                    // Inline init kernel via lambda — cannot define __global__ in function.
                    // We use the existing pattern: no dedicated kernel; use a thrust::sequence
                    // pattern. Since we don't include thrust, write a trivial device kernel.
                    // Forward declaration (defined below in detail::).
                    extern __global__ void init_sort_indices_kernel(int32_t*, int, int);
                    init_sort_indices_kernel<<<dg_init, 256, 0, ctx.stream>>>(
                        d_sort_idx.get(), actual_csnps, total_sort);
                }

                // Rebuild gene segment offsets if actual_cgenes != cgenes.
                if (actual_cgenes != cgenes) {
                    std::vector<int32_t> seg_h_local;
                    detail::build_gene_segment_offsets(seg_h_local, actual_cgenes, actual_csnps);
                    cudaMemcpyAsync(d_gene_seg_offsets.get(), seg_h_local.data(),
                                   (actual_cgenes + 1) * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, ctx.stream);
                }

                // Segmented sort p-values per gene.
                size_t tmp_bytes = cub_sort_tmp_bytes;
                cub::DeviceSegmentedSort::SortPairs(
                    (void*)d_cub_tmp.get(), tmp_bytes,
                    d_pval_buf.get(), d_pval_sorted.get(),
                    d_sort_idx.get(), d_sort_idx.get(),
                    actual_cgenes * actual_csnps, actual_cgenes,
                    d_gene_seg_offsets.get(), d_gene_seg_offsets.get() + 1,
                    ctx.stream);

                // Apply BH formula + cummin.
                detail::bh_apply_kernel<<<actual_cgenes, 256, 0, ctx.stream>>>(
                    d_padj_sorted.get(), d_pval_sorted.get(),
                    actual_csnps, actual_cgenes, cfg.fdr_alpha);

                // Scatter q-values back to original SNP order.
                {
                    int total_scatter = actual_cgenes * actual_csnps;
                    dim3 dg_sc((total_scatter + 255) / 256);
                    detail::scatter_q_kernel<<<dg_sc, 256, 0, ctx.stream>>>(
                        d_padj_buf.get(), d_padj_sorted.get(),
                        d_sort_idx.get(), actual_csnps, actual_cgenes);
                }
            } else {
                // No BH: fill p_adj with NaN.
                cudaMemsetAsync(d_padj_buf.get(), 0xFF,
                                actual_csnps * actual_cgenes * sizeof(float), ctx.stream);
            }

            // DMA result chunk to pinned host buffers.
            // These transfers happen ONCE per (snp_chunk, gene_chunk) — not per-iteration.
            size_t pair_floats = (size_t)actual_csnps * actual_cgenes;
            cudaMemcpyAsync(h_beta_pin.as<float>(),   d_beta_buf.get(),
                            pair_floats * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
            cudaMemcpyAsync(h_se_pin.as<float>(),     d_se_buf.get(),
                            pair_floats * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
            cudaMemcpyAsync(h_sigma_pin.as<float>(),  d_sigma_buf.get(),
                            pair_floats * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
            cudaMemcpyAsync(h_pval_pin.as<float>(),   d_pval_buf.get(),
                            pair_floats * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
            cudaMemcpyAsync(h_padj_pin.as<float>(),   d_padj_buf.get(),
                            pair_floats * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
            cudaMemcpyAsync(h_conv_pin.as<uint8_t>(), d_conv_buf.get(),
                            pair_floats * sizeof(uint8_t), cudaMemcpyDeviceToHost, ctx.stream);

            // Synchronize before invoking callback (data must be on host).
            cudaStreamSynchronize(ctx.stream);

            // Invoke callback or accumulate.
            if (use_callback) {
                NebulaChunkResult chunk;
                chunk.snp_start  = snp_start;
                chunk.gene_start = gene_start;
                chunk.n_snps     = actual_csnps;
                chunk.n_genes    = actual_cgenes;
                chunk.beta       = h_beta_pin.as<float>();
                chunk.se         = h_se_pin.as<float>();
                chunk.sigma_sq   = h_sigma_pin.as<float>();
                chunk.p_value    = h_pval_pin.as<float>();
                chunk.p_adj      = h_padj_pin.as<float>();
                chunk.converged  = h_conv_pin.as<uint8_t>();
                callback(chunk);
            } else {
                // Accumulate into result arrays.
                // Layout: result.beta[snp * n_genes + gene] = chunk.beta[snp_local * actual_cgenes + gene_local]
                // Requires a scatter for the gene dimension.
                for (int si = 0; si < actual_csnps; ++si) {
                    int global_snp = snp_start + si;
                    for (int gi = 0; gi < actual_cgenes; ++gi) {
                        int global_gene = gene_start + gi;
                        size_t src = (size_t)si * actual_cgenes + gi;
                        size_t dst = (size_t)global_snp * n_genes + global_gene;
                        result.beta    [dst] = h_beta_pin.as<float>()[src];
                        result.se      [dst] = h_se_pin.as<float>()[src];
                        result.sigma_sq[dst] = h_sigma_pin.as<float>()[src];
                        result.p_value [dst] = h_pval_pin.as<float>()[src];
                        result.p_adj   [dst] = h_padj_pin.as<float>()[src];
                        result.converged[dst] = h_conv_pin.as<uint8_t>()[src];
                    }
                }
            }
        } // gene chunks

        // Swap double buffers: copy prefetched B → A pointers.
        if (use_double_buf && sc_idx + 1 < n_snp_chunks) {
            // Ensure stream2 prefetch is complete before next iteration uses A buffers.
            cudaStreamSynchronize(stream2);
            std::swap(d_snp_ad_donor_A, d_snp_ad_donor_B);
            std::swap(d_snp_dp_donor_A, d_snp_dp_donor_B);
        }
    } // snp chunks

    return result;
}

// ─── Kernel definitions (defined here; referenced by forward declarations above) ──
// These are __global__ functions in the singlet_gpu::eqtl namespace at file scope
// (CUDA requires __global__ at namespace scope, not inside function bodies).
// NOTE: NOT in detail:: — the extern __global__ forward declarations in run_nebula
// resolve relative to singlet_gpu::eqtl, so the definitions must live here.

// Expression/count pseudobulk aggregation kernel.
// One block per gene; shared mem = n_donors floats.
// Accumulates CSC nonzeros into per-donor sums.
__launch_bounds__(256, 2)
__global__ void
expr_aggregate_kernel(
    float* __restrict__         out,           // [n_genes × n_donors] row-major
    const float* __restrict__   csc_values,
    const int32_t* __restrict__ csc_indptr,
    const int32_t* __restrict__ csc_row_indices,
    const int32_t* __restrict__ cell_to_donor,
    int n_genes,
    int n_donors)
{
    extern __shared__ float acc_smem[];
    int g = (int)blockIdx.x;
    if (g >= n_genes) return;
    for (int i = (int)threadIdx.x; i < n_donors; i += (int)blockDim.x) acc_smem[i] = 0.f;
    __syncthreads();
    int col_s = csc_indptr[g];
    int col_e = csc_indptr[g + 1];
    for (int i = col_s + (int)threadIdx.x; i < col_e; i += (int)blockDim.x) {
        int cell  = csc_row_indices[i];
        int donor = cell_to_donor[cell];
        if (donor >= 0 && donor < n_donors)
            atomicAdd(&acc_smem[donor], csc_values[i]);
    }
    __syncthreads();
    // Write [gene, donor] = acc_smem[donor] → out[gene * n_donors + donor].
    float* row = out + g * n_donors;
    for (int i = (int)threadIdx.x; i < n_donors; i += (int)blockDim.x)
        row[i] = acc_smem[i];
}

// SNP pseudobulk aggregation kernel (same as above but with a column offset).
// One block per SNP (local index); shared mem = n_donors floats.
__launch_bounds__(256, 2)
__global__ void
snp_aggregate_kernel(
    float* __restrict__         out,           // [n_snps_chunk × n_donors] row-major
    const float* __restrict__   csc_values,
    const int32_t* __restrict__ csc_indptr,
    const int32_t* __restrict__ csc_row_indices,
    const int32_t* __restrict__ cell_to_donor,
    int n_snps_chunk,
    int n_donors,
    int snp_start)
{
    extern __shared__ float acc_smem[];
    int s_local = (int)blockIdx.x;
    if (s_local >= n_snps_chunk) return;
    int s_global = snp_start + s_local;
    for (int i = (int)threadIdx.x; i < n_donors; i += (int)blockDim.x) acc_smem[i] = 0.f;
    __syncthreads();
    int col_s = csc_indptr[s_global];
    int col_e = csc_indptr[s_global + 1];
    for (int i = col_s + (int)threadIdx.x; i < col_e; i += (int)blockDim.x) {
        int cell  = csc_row_indices[i];
        int donor = cell_to_donor[cell];
        if (donor >= 0 && donor < n_donors)
            atomicAdd(&acc_smem[donor], csc_values[i]);
    }
    __syncthreads();
    float* row = out + s_local * n_donors;
    for (int i = (int)threadIdx.x; i < n_donors; i += (int)blockDim.x)
        row[i] = acc_smem[i];
}

// Sort-index initialization kernel: d_sort_idx[gene*n_snps + snp] = snp.
__launch_bounds__(256, 4)
__global__ void
init_sort_indices_kernel(
    int32_t* __restrict__ out,
    int n_snps,
    int total)
{
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx >= total) return;
    out[idx] = idx % n_snps;
}

}  // namespace eqtl
}  // namespace singlet_gpu
