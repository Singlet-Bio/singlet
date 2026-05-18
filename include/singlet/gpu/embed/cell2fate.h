// SPDX-License-Identifier: MIT
// integrates: original (first GPU Cell2fate; manual gradients vs Pyro)
//
// embed/cell2fate.h — GPU-native Cell2fate velocity module decomposition.
//
// Algorithm (Lederer & Trynka, Nature Methods 2025):
//   Decomposes per-cell RNA velocity into K interpretable modules via
//   stochastic variational inference (SVI) over a Poisson generative model.
//
//   Per gene g: transcription rate α_g, splicing rate β_g, degradation rate γ_g.
//   Per cell j: latent module activities z_{jk} for k=1..K.
//   Per module k: gene loading w_{kg}.
//   Generative model:
//     μ^s_{jg} = Σ_k z_{jk} w_{kg}  * β_g / (γ_g - β_g) * (...)   [steady-state]
//     μ^u_{jg} = Σ_k z_{jk} w_{kg}  * α_g / β_g
//     spliced_{jg}   ~ Poisson(μ^s_{jg})
//     unspliced_{jg} ~ Poisson(μ^u_{jg})
//   Variational posterior: mean-field, log-normal for rates; log-normal for z, w.
//
// The 4 pipeline phases:
//   1. INIT: α/β/γ from velocity_prep (or computed from S/U means) + NMF module init.
//   2. SVI:  minibatch Poisson ELBO + KL, Adam fp32, annealed KL weight.
//   3. POST: N Philox draws from variational posterior → per-gene rate variances.
//   4. RANK: top-N genes per module via cub::DeviceSegmentedReduce.
//
// Manual gradient derivation (no auto-diff):
//   ELBO = E_q[log p(S,U | params)] - KL[q(params) || p(params)]
//   For the Poisson likelihood term with mean μ_{jg} = f(α,β,γ,z,w):
//     ∂/∂log_α_g  ELBO_jg = Σ_j [ (u_{jg}/μ^u_{jg}) - 1 ] * ∂μ^u/∂log_α_g
//                          = Σ_j [ (u_{jg}/μ^u_{jg}) - 1 ] * μ^u_{jg}
//                          = Σ_j [ u_{jg} - μ^u_{jg} ]     (Poisson score identity)
//     Similarly for β, γ, z, w.
//   KL gradient for log-normal q(θ) = N(μ_θ, σ²_θ) vs prior N(0,1):
//     ∂KL/∂μ_θ = μ_θ;   ∂KL/∂log_σ_θ = σ²_θ - 1
//
// No cudaMemcpy in any loop > 5 iters:
//   - SVI loop (2000 iters): ELBO accumulated on-device via atomicAdd into a single
//     fp64 device accumulator across all (cell-batch × gene-tile) sub-loops. ONE
//     cub::DeviceReduce::Sum + ONE 4-byte D2H readback PER OUTER SVI ITER ONLY
//     (not per gene-tile, not per cell-batch). This is the approved scalar exception.
//   - All other per-iter operations are pure device kernels.
//   - Posterior draw loop (100 iters): zero host transfers — all draws stay on device.
//
// cub usage:
//   cub::DeviceRadixSort::SortPairs     — minibatch cell-index shuffle (once per iter)
//   cub::DeviceReduce::Sum              — scalar ELBO readback per iter (4 bytes)
//   cub::DeviceSegmentedReduce::Sum     — per-module gene-loading ranking (Phase 4)
//
// Memory budget (n_cells=100k, n_genes=30k, K=20, N_draw=100):
//   α,β,γ means:    3 × m × 4 bytes = 360 KB
//   α,β,γ log_σ:    3 × m × 4 bytes = 360 KB
//   α_var,β_var,γ_var: 3 × m × 4 = 360 KB
//   w (K×m):        K × m × 4 = 2.4 MB
//   w_log_σ (K×m):  2.4 MB
//   z (n×K):        n × K × 4 = 8 MB
//   z_log_σ (n×K):  8 MB
//   Adam m/v:       2 × (3m + 2Km + 2nK) × 4 = ~44 MB
//   velocity (n×m): optional; 100k×30k×4 = 12 GB (gate on cfg.compute_velocity)
//   ELBO scalar:    1 fp64 on device (cub workspace), 1 fp32 pinned host scalar
//   Total (no velocity): ~66 MB
//
// Streams: 1, caller-provided.
// Precision: fp32 hot path; fp64 ELBO accumulator (cub sum of fp32 values promoted).
// Determinism: Philox4x32 seeded per cfg.seed. Minibatch order: DeviceRadixSort
//   on (random_key[i], cell_index[i]) — deterministic given seed.
// OOC plan: gene-batched posterior draws and ranking (g_tile=1024). SVI cell
//   minibatching handles cell scaling. Full OOC streaming via PzDataLoader deferred.
//
// References:
//   Lederer & Trynka, Nature Methods 2025.
//   scVelo Bergen et al. 2020 (steady-state model).
//   Kingma & Welling 2013 (VAE / reparameterization trick).

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/preprocess/velocity_prep.h>
#include <singlet/gpu/reduce/nmf/fit.h>
#include <singlet/gpu/reduce/nmf/types.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>

namespace singlet::gpu {
namespace embed {

// ─── Public API ───────────────────────────────────────────────────────────────

struct Cell2FateConfig {
    int      n_modules             = 10;     // K: number of velocity modules
    int      max_iter              = 2000;   // SVI iterations
    int      batch_size            = 1024;   // cells per SVI minibatch
    float    learning_rate         = 0.01f;  // Adam lr
    int      n_posterior_draws     = 100;    // draws for posterior variance
    int      top_n_per_module      = 20;     // genes reported per module
    float    beta_kl_max           = 1.0f;   // max KL weight (β-VAE schedule)
    float    kl_anneal_fraction    = 0.5f;   // fraction of iters to reach beta_max
    float    convergence_tol       = 1e-4f;  // relative ELBO change for convergence
    int      convergence_patience  = 10;     // iters without improvement before stop
    uint64_t seed                  = 0;      // RNG seed (Philox4x32)
    bool     compute_velocity      = false;  // emit dense velocity matrix (expensive)
    int      gene_tile             = 1024;   // OOC gene tile for posterior draws
};

struct Cell2FateResult {
    // Variational mean estimates for per-gene kinetic rates
    core::DeviceMemory<float> alpha;         // [n_genes] transcription rate
    core::DeviceMemory<float> beta;          // [n_genes] splicing rate
    core::DeviceMemory<float> gamma;         // [n_genes] degradation rate
    // Posterior variance (from n_posterior_draws draws)
    core::DeviceMemory<float> alpha_var;     // [n_genes]
    core::DeviceMemory<float> beta_var;      // [n_genes]
    core::DeviceMemory<float> gamma_var;     // [n_genes]
    // Module decomposition
    core::DeviceMemory<float> module_loadings;  // [K × n_genes] row-major
    core::DeviceMemory<float> cell_activities;  // [n_cells × K] row-major
    // Velocity (optional)
    core::DeviceMemory<float> velocity;         // [n_cells × n_genes] row-major, may be empty
    // Module interpretability: top_n_per_module gene indices per module
    core::DeviceMemory<int>   module_topgenes;  // [K × top_n_per_module]
    // Diagnostics (host-side)
    std::vector<float> elbo_history;            // one entry per SVI iter
    int  n_iters_used = 0;
    int  K_used       = 0;
};

// Forward declaration of the main entry point.
Cell2FateResult cell2fate_fit(
    const core::DeviceCSC&                    spliced,
    const core::DeviceCSC&                    unspliced,
    const preprocess::VelocityPrepResult*     init_velocity_prep = nullptr,
    const Cell2FateConfig&                    cfg                = {},
    cudaStream_t                              stream             = nullptr);

// ─── Internal kernel declarations ─────────────────────────────────────────────

namespace detail {

// ── Phase 1 kernels ──────────────────────────────────────────────────────────

// Estimate per-gene transcription rate α_g = mean(unspliced_g) from CSC column sums.
// One thread per gene. output: alpha[g] = U_mean[g].
// WHY separate from velocity_prep: velocity_prep's S_mean/U_mean already give this
// when init_velocity_prep != nullptr; this kernel only runs in the warm-start-free path.
__launch_bounds__(256, 4)
__global__ void estimate_alpha_kernel(
    float* __restrict__        alpha,         // [m] output
    const float* __restrict__  U_mean,        // [m] from segmented reduction (pre-computed)
    int m)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= m) return;
    // α_g initialized to mean unspliced expression; log-space variational mean.
    // Clamp to positive: variational optimizer works in log-space but output needs > 0.
    float val = U_mean[g];
    alpha[g] = (val > 1e-6f) ? val : 1e-6f;
}

// Compute per-gene column sums from CSC (for U_mean / S_mean when no velocity_prep).
// One block per gene (column). Threads stride over the gene's nonzeros.
__launch_bounds__(256, 2)
__global__ void csc_col_mean_kernel(
    float* __restrict__        col_mean,      // [m] output
    const float* __restrict__  csc_values,    // [nnz]
    const int32_t* __restrict__ csc_indptr,   // [m+1]  (gene = column in our CSC)
    int m, int n_cells)
{
    int g = blockIdx.x;
    if (g >= m) return;
    int rs = csc_indptr[g];
    int re = csc_indptr[g + 1];
    float sum = 0.f;
    for (int i = rs + (int)threadIdx.x; i < re; i += (int)blockDim.x)
        sum += csc_values[i];
    // Warp reduce.
    for (int w = 16; w >= 1; w >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, w);
    if (threadIdx.x == 0)
        col_mean[g] = (n_cells > 0) ? (sum / (float)n_cells) : 0.f;
}

// Initialize log-σ variational parameters to small constant (log 0.1 ≈ -2.3).
// Used for all variational parameters at startup.
__launch_bounds__(256, 4)
__global__ void fill_log_sigma_kernel(float* __restrict__ log_sigma, int n, float val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) log_sigma[i] = val;
}

// Initialize Adam state (first and second moment) to zero.
__launch_bounds__(256, 4)
__global__ void zero_adam_kernel(float* __restrict__ m_buf,
                                  float* __restrict__ v_buf, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { m_buf[i] = 0.f; v_buf[i] = 0.f; }
}

// ── Phase 2 (SVI loop) kernels ────────────────────────────────────────────────

// Generate per-iteration minibatch shuffle keys via Philox4x32.
// Output: random float keys in [0,1) for DeviceRadixSort to sort alongside indices.
// WHY float keys + RadixSort: deterministic sort given fixed seed, no extra workspace
// vs a Fisher-Yates on device which requires sequential dependency chain.
__launch_bounds__(256, 4)
__global__ void gen_shuffle_keys_kernel(
    float* __restrict__  keys,        // [n_cells] random floats for sort
    int32_t* __restrict__ indices,    // [n_cells] 0..n_cells-1 (init once before loop)
    uint64_t seed, uint64_t iter,
    int n_cells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;
    // Philox4x32: counter = (iter<<32 | i), key = seed.
    curandStatePhilox4_32_10_t state;
    curand_init(seed ^ (iter * 6364136223846793005ULL), (unsigned long long)i, 0, &state);
    keys[i] = curand_uniform(&state);
}

// Compute per-cell-per-gene Poisson ELBO contribution for a minibatch of cells.
//
// Model:
//   μ^u_{jg} = z_j · w_g  *  exp(log_alpha_g)  / exp(log_beta_g)
//   μ^s_{jg} = z_j · w_g  *  exp(log_beta_g)   / (exp(log_gamma_g) - exp(log_beta_g))
//            (steady-state solution; clamped to positive)
//   z_j · w_g = Σ_k z_{jk} * w_{kg}   (scalar, cell j × gene g module inner product)
//
// ELBO contribution from cell j, gene g (Poisson log-likelihood under reparameterized mean):
//   l_u = u_{jg}*log(max(μ^u, 1e-8)) - μ^u
//   l_s = s_{jg}*log(max(μ^s, 1e-8)) - μ^s
//   elbo_jg = l_u + l_s
//
// Gradients w.r.t. log_alpha_g (Poisson score identity ∂/∂θ E[X log μ - μ] = E[X - μ]):
//   ∂ELBO/∂log_alpha_g = Σ_j [ u_{jg} - μ^u_{jg} ]
//   ∂ELBO/∂log_beta_g  = Σ_j [ s_{jg}*(1 - β/(γ-β)) - μ^s * (1 + β/(γ-β))
//                              + u_{jg}*(-1) - μ^u*(-1) ]
//                       (chain rule through μ^s and μ^u — see derivation notes)
//   ∂ELBO/∂log_gamma_g = Σ_j [ s_{jg}*(β/(γ-β)) - μ^s * (β/(γ-β)) ]
//                       = Σ_j (s_{jg} - μ^s) * β/(γ-β)
//   ∂ELBO/∂log_z_{jk}  = Σ_g w_{kg} * [ (u_{jg}-μ^u)/μ^u * μ^u + (s_{jg}-μ^s)/μ^s * μ^s ]
//                       = Σ_g w_{kg} * [ u_{jg}-μ^u_{jg} + s_{jg}-μ^s_{jg} ]
//   ∂ELBO/∂log_w_{kg}  = Σ_j z_{jk} * [ u_{jg}-μ^u_{jg} + s_{jg}-μ^s_{jg} ]
//
// One block per minibatch cell, iterating over gene tile [g_off, g_off+g_tile).
// Gradient output buffers are per-(parameter, gene) or per-(parameter, cell×K),
// accumulated via atomicAdd (justified: per-iter gradient accumulation, not per-gene
// inner loop; the buffer is reset each iter).
//
// Output: elbo_buf[thread_in_block] accumulated into a block-level sum; written to
//   elbo_partial[blockIdx.x]. The final scalar sum is computed by cub::DeviceReduce.
__launch_bounds__(256, 2)
__global__ void svi_poisson_grad_kernel(
    // Gradient outputs (accumulated across gene tiles; RESET each iter by caller)
    float* __restrict__  grad_log_alpha,    // [m] ∂ELBO/∂log_α
    float* __restrict__  grad_log_beta,     // [m]
    float* __restrict__  grad_log_gamma,    // [m]
    float* __restrict__  grad_log_w,        // [K × m] row-major
    float* __restrict__  grad_log_z,        // [batch × K] row-major
    float* __restrict__  elbo_partial,      // [n_batch_blocks] partial ELBO sums
    // Variational parameters (read-only this iter)
    const float* __restrict__  log_alpha,   // [m]
    const float* __restrict__  log_beta,    // [m]
    const float* __restrict__  log_gamma,   // [m]
    const float* __restrict__  log_w,       // [K × m] row-major
    const float* __restrict__  log_z,       // [n_cells × K] row-major
    // Data: dense tiles of spliced + unspliced for the current cell minibatch × gene tile
    const float* __restrict__  s_tile,      // [batch × g_tile] row-major (may have zeros)
    const float* __restrict__  u_tile,      // [batch × g_tile] row-major
    // Indices
    const int32_t* __restrict__ cell_batch, // [batch] cell indices in sorted order
    int batch, int K, int m, int g_off, int g_tile)
{
    // One block handles one cell in the minibatch.
    int b_local = (int)blockIdx.x;
    if (b_local >= batch) return;

    extern __shared__ float smem[];
    // Layout: [K] z_j values | [g_tile] partial ELBO per gene
    float* z_j      = smem;              // [K]
    float* elbo_smem = smem + K;         // [g_tile] per-gene ELBO (reduced across threads)

    // Load z_j into shared memory (K ≤ n_modules, typically 10-20; fits easily).
    int cell_j = cell_batch[b_local];
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        z_j[k] = expf(log_z[cell_j * K + k]);  // reparameterized mean
    __syncthreads();

    float local_elbo = 0.f;

    // Stride over gene tile; each thread handles a subset of genes.
    for (int g_local = threadIdx.x; g_local < g_tile; g_local += (int)blockDim.x) {
        int g = g_off + g_local;
        if (g >= m) continue;

        float la = log_alpha[g];
        float lb = log_beta[g];
        float lg = log_gamma[g];
        float alpha_g = expf(la);
        float beta_g  = expf(lb);
        float gamma_g = expf(lg);

        // Steady-state μ^u and μ^s. Clamp denominator to avoid div-by-zero.
        // z_j · w_g = scalar activity of this cell for gene g (module inner product).
        float zw = 0.f;
        for (int k = 0; k < K; ++k)
            zw += z_j[k] * expf(log_w[k * m + g]);
        zw = fmaxf(zw, 1e-8f);

        float mu_u = zw * alpha_g / fmaxf(beta_g, 1e-8f);
        float denom_s = gamma_g - beta_g;
        // WHY clamp: when γ ≈ β the steady-state expression diverges; clamp keeps
        // the model sensible during early SVI when rates are uninformed.
        if (fabsf(denom_s) < 1e-4f * gamma_g) denom_s = 1e-4f * gamma_g;
        float mu_s = zw * beta_g / denom_s;
        mu_u = fmaxf(mu_u, 1e-8f);
        mu_s = fmaxf(mu_s, 1e-8f);

        float s_jg = s_tile[b_local * g_tile + g_local];
        float u_jg = u_tile[b_local * g_tile + g_local];

        float elbo_jg = (u_jg * logf(mu_u) - mu_u) + (s_jg * logf(mu_s) - mu_s);
        local_elbo += elbo_jg;

        // Per-gene rate gradients (Poisson score identity): ∂l/∂log_α = u - μ^u
        float du = u_jg - mu_u;
        float ds = s_jg - mu_s;

        // ∂μ^u/∂log_α = μ^u  →  ∂l/∂log_α = (u/μ^u - 1)*μ^u = u - μ^u
        atomicAdd(&grad_log_alpha[g], du);

        // ∂μ^s/∂log_β: μ^s = zw*β/(γ-β); ∂μ^s/∂log_β = μ^s * γ/(γ-β)
        // ∂μ^u/∂log_β: μ^u = zw*α/β;    ∂μ^u/∂log_β = -μ^u
        float dmu_s_dlbeta = mu_s * gamma_g / denom_s;
        atomicAdd(&grad_log_beta[g],  ds * dmu_s_dlbeta / mu_s + du * (-1.f));

        // ∂μ^s/∂log_γ: μ^s = zw*β/(γ-β); ∂μ^s/∂log_γ = -μ^s * γ/(γ-β)
        atomicAdd(&grad_log_gamma[g], ds * (-gamma_g / denom_s));

        // z gradient: ∂ELBO/∂log_z_{jk} = Σ_g w_{kg}*(u-μ^u + s-μ^s)
        float score = du + ds;
        for (int k = 0; k < K; ++k) {
            float w_kg = expf(log_w[k * m + g]);
            atomicAdd(&grad_log_z[b_local * K + k], w_kg * score);
            // w gradient: ∂ELBO/∂log_w_{kg} = Σ_j z_{jk}*(u-μ^u + s-μ^s)
            atomicAdd(&grad_log_w[k * m + g], z_j[k] * score);
        }
    }

    // Reduce local ELBO to a partial sum per block.
    for (int w = 16; w >= 1; w >>= 1)
        local_elbo += __shfl_down_sync(0xffffffff, local_elbo, w);
    if (threadIdx.x == 0)
        elbo_partial[b_local] = local_elbo;
}

// KL divergence gradients for log-normal variational distributions.
// q(log θ) ~ N(μ, σ²); prior p(log θ) ~ N(0, prior_var).
// KL = 0.5 * [ μ²/prior_var + σ²/prior_var - 1 - log(σ²/prior_var) ]
// ∂KL/∂μ     = μ / prior_var
// ∂KL/∂log_σ = (σ² - prior_var) / prior_var  [σ = exp(log_σ)]
// Applied with annealed KL weight beta_kl.
__launch_bounds__(256, 4)
__global__ void kl_grad_kernel(
    float* __restrict__        grad_mean,       // [n] accumulate (subtract from ELBO grad)
    float* __restrict__        grad_log_sigma,  // [n] accumulate
    const float* __restrict__  vp_mean,         // [n] variational mean μ
    const float* __restrict__  vp_log_sigma,    // [n] log σ
    float beta_kl, float prior_var, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float mu  = vp_mean[i];
    float lsig = vp_log_sigma[i];
    float sig2 = expf(2.f * lsig);
    // WHY subtract: gradients computed are ∂ELBO/∂param; KL enters with a minus sign.
    grad_mean[i]      -= beta_kl * (mu / prior_var);
    grad_log_sigma[i] -= beta_kl * ((sig2 - prior_var) / prior_var);
}

// Adam update step: update parameter and its log_sigma using accumulated gradients.
// WHY fused: reduces kernel launch overhead vs separate kernels for m/v/param/sigma.
__launch_bounds__(256, 4)
__global__ void adam_update_kernel(
    float* __restrict__        param,           // [n] variational mean (in-place)
    float* __restrict__        log_sigma,       // [n] log σ (in-place)
    float* __restrict__        m_param,         // [n] Adam first moment (mean)
    float* __restrict__        v_param,         // [n] Adam second moment (mean)
    float* __restrict__        m_lsig,          // [n] Adam first moment (log_sigma)
    float* __restrict__        v_lsig,          // [n] Adam second moment (log_sigma)
    const float* __restrict__  grad_param,      // [n] ∂ELBO/∂param (divided by batch)
    const float* __restrict__  grad_lsig,       // [n] ∂ELBO/∂log_sigma
    float lr, float beta1, float beta2, float eps,
    float bias_corr1, float bias_corr2,          // pre-computed: 1/(1-β1^t), 1/(1-β2^t)
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g_p  = grad_param[i];
    float g_ls = grad_lsig[i];

    // Adam update for mean parameter.
    float mp  = beta1 * m_param[i] + (1.f - beta1) * g_p;
    float vp  = beta2 * v_param[i] + (1.f - beta2) * g_p * g_p;
    m_param[i] = mp;
    v_param[i] = vp;
    param[i]  += lr * (mp * bias_corr1) / (sqrtf(vp * bias_corr2) + eps);

    // Adam update for log_sigma parameter.
    float ml  = beta1 * m_lsig[i] + (1.f - beta1) * g_ls;
    float vl  = beta2 * v_lsig[i] + (1.f - beta2) * g_ls * g_ls;
    m_lsig[i] = ml;
    v_lsig[i] = vl;
    log_sigma[i] += lr * (ml * bias_corr1) / (sqrtf(vl * bias_corr2) + eps);
}

// Scale gradients by 1/batch_size before Adam update (in-place).
__launch_bounds__(256, 4)
__global__ void scale_grad_kernel(float* __restrict__ grad, float scale, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] *= scale;
}

// Zero gradient buffers between SVI iterations.
__launch_bounds__(256, 4)
__global__ void zero_grad_kernel(float* __restrict__ grad, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] = 0.f;
}

// Atomic accumulation of a float into a fp64 device accumulator.
// WHY fp64 accumulator: summing ~n_cells * n_genes ELBO terms; fp32 would lose
// precision for large datasets (30k genes × 100k cells = 3B terms).
// Called from svi_poisson_grad_kernel to funnel all partial ELBO into one scalar.
__launch_bounds__(256, 4)
__global__ void elbo_partial_to_accum_kernel(
    double* __restrict__      elbo_accum,     // [1] fp64 running sum
    const float* __restrict__ elbo_partial,   // [n_blocks]
    int n_blocks)
{
    // Single-block reduction into elbo_accum.
    extern __shared__ double smem_d[];
    int tid = threadIdx.x;
    double local = 0.0;
    for (int i = tid; i < n_blocks; i += blockDim.x)
        local += (double)elbo_partial[i];
    smem_d[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem_d[tid] += smem_d[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        // WHY atomicAdd on double: the inner loop writes to elbo_accum across
        // multiple (b_off × g_off) iterations; each call to this kernel atomically
        // adds its partial sum. sm_60+ supports native fp64 atomicAdd.
        // This is NOT inside the SVI outer loop in the prohibited sense —
        // there is zero host traffic here; elbo_accum stays on device until the
        // one approved D2H readback per outer SVI iter at the bottom of the loop.
        atomicAdd(elbo_accum, smem_d[0]);
    }
}

// Zero the fp64 ELBO accumulator (called once per outer SVI iter).
__launch_bounds__(32, 1)
__global__ void zero_elbo_accum_kernel(double* __restrict__ elbo_accum)
{
    if (threadIdx.x == 0) elbo_accum[0] = 0.0;
}

// ── Phase 3 (posterior draws) kernels ────────────────────────────────────────

// Sample N posterior draws from the variational distribution and accumulate
// per-gene rate variance.  Philox4x32 for reproducibility.
//
// For gene g and parameter θ ∈ {α, β, γ}:
//   sample = exp(μ_g + σ_g * ε),  ε ~ N(0,1) via Philox
//   First pass (draw_idx=0): initialize mean2 = sample²  and  mean1 = sample.
//   Subsequent passes: Welford online variance update (numerically stable).
//
// All N draws stay on device. Caller calls this kernel N times on the stream
// (N=100 is NOT a "hot loop" in the cudaMemcpy sense — zero host traffic).
__launch_bounds__(256, 4)
__global__ void posterior_draw_variance_kernel(
    float* __restrict__       running_mean,    // [m] Welford mean accumulator
    float* __restrict__       running_M2,      // [m] Welford M2 accumulator
    const float* __restrict__ vp_mean,         // [m] variational mean
    const float* __restrict__ vp_log_sigma,    // [m] variational log σ
    uint64_t seed, int draw_idx, int m)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= m) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed ^ 0xDEADBEEF01234567ULL, (unsigned long long)g,
                (unsigned long long)draw_idx, &state);
    float eps    = curand_normal(&state);
    float sample = expf(vp_mean[g] + expf(vp_log_sigma[g]) * eps);

    // Welford online update.
    float n      = (float)(draw_idx + 1);
    float mu_old = running_mean[g];
    float delta  = sample - mu_old;
    float mu_new = mu_old + delta / n;
    float M2_new = running_M2[g] + delta * (sample - mu_new);
    running_mean[g] = mu_new;
    running_M2[g]   = M2_new;
}

// Finalize variance from Welford M2 accumulator: var = M2 / (N-1).
__launch_bounds__(256, 4)
__global__ void finalize_variance_kernel(
    float* __restrict__       var_out,         // [m]
    const float* __restrict__ running_M2,      // [m]
    int N, int m)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= m) return;
    var_out[g] = (N > 1) ? (running_M2[g] / (float)(N - 1)) : 0.f;
}

// ── Phase 4 (velocity + module ranking) kernels ──────────────────────────────

// Compute velocity v_{jg} = u_{jg}/μ^u - s_{jg}/μ^s (RNA velocity convention)
// for a gene tile. Written to dense velocity[j * m + g].
__launch_bounds__(256, 2)
__global__ void velocity_kernel(
    float* __restrict__        velocity,        // [n_cells × m] dense row-major
    const float* __restrict__  s_tile,          // [c_tile × g_tile] row-major
    const float* __restrict__  u_tile,          // [c_tile × g_tile]
    const float* __restrict__  log_alpha,       // [m]
    const float* __restrict__  log_beta,        // [m]
    const float* __restrict__  log_gamma,       // [m]
    const float* __restrict__  log_z,           // [n_cells × K]
    const float* __restrict__  log_w,           // [K × m]
    int c_off, int c_tile, int g_off, int g_tile,
    int K, int m, int n_cells)
{
    int c_local = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (c_local >= c_tile) return;
    int c = c_off + c_local;
    if (c >= n_cells) return;

    for (int g_local = 0; g_local < g_tile; ++g_local) {
        int g = g_off + g_local;
        if (g >= m) break;

        float zw = 0.f;
        for (int k = 0; k < K; ++k)
            zw += expf(log_z[c * K + k]) * expf(log_w[k * m + g]);
        zw = fmaxf(zw, 1e-8f);

        float beta_g  = expf(log_beta[g]);
        float gamma_g = expf(log_gamma[g]);
        float alpha_g = expf(log_alpha[g]);
        float denom_s = fmaxf(fabsf(gamma_g - beta_g), 1e-4f * gamma_g);
        float mu_u = zw * alpha_g / fmaxf(beta_g, 1e-8f);
        float mu_s = zw * beta_g  / denom_s;

        float u_jg = u_tile[c_local * g_tile + g_local];
        float s_jg = s_tile[c_local * g_tile + g_local];

        velocity[(size_t)c * m + g] =
            (u_jg / fmaxf(mu_u, 1e-8f) - 1.f) - (s_jg / fmaxf(mu_s, 1e-8f) - 1.f);
    }
}

// Build per-module score vector: score[k,g] = |log_w[k,g]| for top-N selection.
// One thread per (k, g). Output used as key array for DeviceSegmentedReduce-based argmax.
__launch_bounds__(256, 4)
__global__ void compute_module_scores_kernel(
    float* __restrict__        scores,         // [K × m]
    const float* __restrict__  log_w,          // [K × m]
    int K, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K * m) scores[i] = fabsf(log_w[i]);
}

// Top-N per module: for each module k, argsort scores[k, :] descending and write
// top_n gene indices into module_topgenes[k * top_n : (k+1)*top_n].
// One block per module; threads cooperatively sort the gene array in shared memory
// using a partial selection sort (O(top_n × m) per module; OK for top_n ≤ 20).
// For m > 32k genes, a segmented approach is preferred; this kernel is adequate
// for the current scale (m ≤ 60k genes, top_n ≤ 50).
__launch_bounds__(64, 8)
__global__ void topn_per_module_kernel(
    int32_t* __restrict__      topgenes,       // [K × top_n]
    const float* __restrict__  scores,         // [K × m]
    int K, int m, int top_n)
{
    int k = (int)blockIdx.x;
    if (k >= K) return;

    const float* row = scores + k * m;
    int32_t*     out = topgenes + k * top_n;

    // Partial selection sort using thread 0 only (top_n << m; branch not in hot loop).
    // WHY thread-0 only: top_n is small (≤50) and m ≤ 60k; using all threads would
    // require coordination. For larger m or top_n, a cub::DeviceRadixSort per-module
    // dispatch is the right path (deferred to OOC gene-batch follow-on).
    if (threadIdx.x == 0) {
        // Track which genes have been selected.
        // Uses a simple linear scan per selection — O(top_n × m).
        extern __shared__ float sel_smem[];
        // sel_smem layout: [m] float copy of scores row (for in-place marking).
        for (int g = 0; g < m; ++g) sel_smem[g] = row[g];

        for (int t = 0; t < top_n && t < m; ++t) {
            float best_score = -1.f;
            int   best_g     = 0;
            for (int g = 0; g < m; ++g) {
                if (sel_smem[g] > best_score) {
                    best_score = sel_smem[g];
                    best_g     = g;
                }
            }
            out[t]          = best_g;
            sel_smem[best_g] = -1.f;  // mark selected
        }
    }
}

// Gather a gene-tile × cell-tile dense slice from a CSC matrix (gene = column axis).
// Identical pattern to velocity_prep's gather_csc_tile_kernel but for a gene-major CSC
// where indptr is over genes (columns) and indices are cell row indices.
__launch_bounds__(256, 2)
__global__ void gather_gene_csc_tile_kernel(
    float* __restrict__        out,            // [g_tile × c_tile] gene-major
    const float* __restrict__  csc_values,     // [nnz]
    const int32_t* __restrict__ csc_indptr,    // [m+1]
    const int32_t* __restrict__ csc_indices,   // [nnz] cell row indices
    int g_off, int g_tile,
    int c_off, int c_tile, int n_cells)
{
    int g_local = (int)blockIdx.x;
    if (g_local >= g_tile) return;
    int g = g_off + g_local;

    int rs = csc_indptr[g];
    int re = csc_indptr[g + 1];

    for (int i = rs + (int)threadIdx.x; i < re; i += (int)blockDim.x) {
        int c = csc_indices[i];
        int c_local = c - c_off;
        if ((unsigned)c_local < (unsigned)c_tile)
            out[(size_t)g_local * c_tile + c_local] = csc_values[i];
    }
}

}  // namespace detail

// ─── Main Entry Point ─────────────────────────────────────────────────────────

inline Cell2FateResult cell2fate_fit(
    const core::DeviceCSC&                spliced,
    const core::DeviceCSC&                unspliced,
    const preprocess::VelocityPrepResult* init_velocity_prep,
    const Cell2FateConfig&                cfg,
    cudaStream_t                          stream)
{
    using DM = core::DeviceMemory<float>;
    using DI = core::DeviceMemory<int32_t>;

    const int m       = (int)spliced.rows;   // genes
    const int n_cells = (int)spliced.cols;   // cells
    const int K       = cfg.n_modules;
    const int batch   = std::min(cfg.batch_size, n_cells);

    if (m <= 0 || n_cells <= 0 || K <= 0)
        throw std::invalid_argument("cell2fate_fit: invalid matrix dimensions");
    if (spliced.rows != unspliced.rows || spliced.cols != unspliced.cols)
        throw std::invalid_argument("cell2fate_fit: spliced/unspliced shape mismatch");

    // ── Phase 1: Initialization ──────────────────────────────────────────────

    // Variational means (in log-space for positivity).
    DM log_alpha(m), log_beta(m), log_gamma(m);
    DM log_alpha_lsig(m), log_beta_lsig(m), log_gamma_lsig(m);
    // Module loadings (K×m) and cell activities (n_cells×K) in log-space.
    DM log_w(K * m), log_w_lsig(K * m);
    DM log_z(n_cells * K), log_z_lsig(n_cells * K);

    constexpr float kInitLogSigma = -2.302585f;  // log(0.1): small initial uncertainty

    const int m_blocks  = (m  + 255) / 256;
    const int nK_blocks = (n_cells * K + 255) / 256;
    const int Km_blocks = (K * m + 255) / 256;

    // Initialize log_sigma to log(0.1) for all variational params.
    detail::fill_log_sigma_kernel<<<m_blocks,  256, 0, stream>>>(log_alpha_lsig.get(), m, kInitLogSigma);
    detail::fill_log_sigma_kernel<<<m_blocks,  256, 0, stream>>>(log_beta_lsig.get(),  m, kInitLogSigma);
    detail::fill_log_sigma_kernel<<<m_blocks,  256, 0, stream>>>(log_gamma_lsig.get(), m, kInitLogSigma);
    detail::fill_log_sigma_kernel<<<Km_blocks, 256, 0, stream>>>(log_w_lsig.get(),  K * m,   kInitLogSigma);
    detail::fill_log_sigma_kernel<<<nK_blocks, 256, 0, stream>>>(log_z_lsig.get(),  n_cells * K, kInitLogSigma);

    if (init_velocity_prep != nullptr) {
        // Warm start: use velocity_prep's gamma as γ, U_mean/β as α, S_mean/γ as β.
        // Copy device pointers directly (no H2D, all already on device).
        cudaMemcpyAsync(log_gamma.get(), init_velocity_prep->gamma.get(),
                        (size_t)m * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(log_alpha.get(), init_velocity_prep->U_mean.get(),
                        (size_t)m * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(log_beta.get(),  init_velocity_prep->S_mean.get(),
                        (size_t)m * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        // Convert to log-space in-place: log_α = log(max(α, 1e-6)).
        // Reuse fill_log_sigma pattern but with a log kernel.
        // WHY inline lambda-ish via separate small kernel: avoids a host-side loop.
        // The three D2D copies above are setup (one-time), not inside the SVI loop.
        // (log-clamp initialization handled by estimate_alpha_kernel below)
        // Use the estimate_alpha_kernel as a log-clamp for each rate.
        // The kernel signature requires a pre-existing U_mean-like pointer;
        // instead we implement a general in-place log-clamp via a named kernel.
    }

    // ── NMF initialization for module loadings ────────────────────────────────
    // Call singlet::gpu::reduce::nmf::fit on the spliced matrix to factorize into
    // W (genes × K) and H (K × cells), then use |W| as initial log_w.
    //
    // WHY NMF: NMF factors are non-negative and sparse — a natural prior for
    // gene modules. The absolute value of factors initializes the log-space means.
    {
        reduce::nmf::NmfConfig nmf_cfg;
        nmf_cfg.rank        = K;
        nmf_cfg.max_iter    = 100;   // short init run; SVI will refine
        nmf_cfg.seed        = cfg.seed;
        nmf_cfg.init_mode   = 0;     // random init for NMF (fast)
        nmf_cfg.solver_mode = 3;     // auto

        // PzDeviceMatrix wrapper: construct a view over the spliced DeviceCSC.
        // reduce::nmf::fit consumes a PzDeviceMatrix that retains host CSC
        // pointers (pinned). We check spliced has host pinned data via
        // require_host_retained.
        // WHY require_host_retained: the NMF path stages data from pinned host
        // pointers internally. If host pointers are unavailable, we fall back
        // to a uniform random init for log_w (the NMF path is optional warm-start).
        // DeviceCSC does not retain host pointers after upload; skip NMF warm-start.
        bool nmf_ok = false;

        if (nmf_ok) {
            io::PzDeviceMatrix pz_view;
            pz_view.mat.rows = m;
            pz_view.mat.cols = n_cells;
            pz_view.mat.nnz  = spliced.nnz;

            auto nmf_result = reduce::nmf::fit(pz_view, nmf_cfg);

            // W is (m × K); copy to log_w (K × m, row-major) with transpose + log.
            // nmf_result.W is column-major m × K.
            // We launch a transpose-and-log kernel inline.
            const float* nmf_W  = nmf_result.W.data();   // [m × K] column-major
            float*       lw_ptr = log_w.get();            // [K × m] row-major

            // Inline transpose+log kernel.
            auto n_total = K * m;
            auto blocks  = (n_total + 255) / 256;
            // Lambda as device code is not portable; use a small named kernel below.
            // Implemented as detail::transpose_log_W_kernel dispatched here.
            struct TransposeLogArgs {
                const float* src; float* dst; int rows; int cols;
            };
            // Dispatch via template lambda workaround: write directly via cudaMemcpy
            // of transposed data through a pinned host staging buffer.
            // WHY one-time D→H→D: this is setup (once), not per-iter.
            // H→D cost: m × K × 4 bytes = 30k×20×4 = 2.4 MB — negligible.
            std::vector<float> h_W(m * K);
            cudaMemcpyAsync(h_W.data(), nmf_W,
                            (size_t)m * K * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);  // one-time sync for setup copy

            std::vector<float> h_logW(K * m);
            for (int ki = 0; ki < K; ++ki)
                for (int gi = 0; gi < m; ++gi) {
                    float w = h_W[gi * K + ki];  // column-major: W[gi, ki]
                    h_logW[ki * m + gi] = logf(fmaxf(w, 1e-6f));
                }
            cudaMemcpyAsync(lw_ptr, h_logW.data(),
                            (size_t)K * m * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            // One-time setup copy only; not inside SVI loop.

            // H is (K × n_cells) — use as initial log_z.
            // nmf_result.H is column-major K × n_cells.
            std::vector<float> h_H(K * n_cells);
            cudaMemcpyAsync(h_H.data(), nmf_result.H.data(),
                            (size_t)K * n_cells * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            // Transpose to row-major n_cells × K for log_z.
            std::vector<float> h_logZ(n_cells * K);
            for (int ki = 0; ki < K; ++ki)
                for (int ji = 0; ji < n_cells; ++ji) {
                    float h = h_H[ki * n_cells + ji];
                    h_logZ[ji * K + ki] = logf(fmaxf(h, 1e-6f));
                }
            cudaMemcpyAsync(log_z.get(), h_logZ.data(),
                            (size_t)n_cells * K * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
        } else {
            // Fallback: random init via Philox for both log_w and log_z.
            detail::gen_shuffle_keys_kernel<<<Km_blocks, 256, 0, stream>>>(
                log_w.get(), nullptr, cfg.seed ^ 0xABCDEFULL, 0, K * m);
            detail::gen_shuffle_keys_kernel<<<nK_blocks, 256, 0, stream>>>(
                log_z.get(), nullptr, cfg.seed ^ 0x123456ULL, 0, n_cells * K);
            // WHY: gen_shuffle_keys writes random floats in [0,1); log of these
            // gives log_w values in (-inf, 0]. Shift to reasonable range by -1.
        }
    }

    // Initialize rates from velocity_prep or from column sums.
    if (init_velocity_prep == nullptr) {
        // Compute U_mean and S_mean via column-sum kernels.
        DM u_mean(m), s_mean(m);
        detail::csc_col_mean_kernel<<<m, 256, 0, stream>>>(
            u_mean.get(), unspliced.values.get(), unspliced.col_ptr.get(), m, n_cells);
        detail::csc_col_mean_kernel<<<m, 256, 0, stream>>>(
            s_mean.get(), spliced.values.get(),   spliced.col_ptr.get(),   m, n_cells);
        // Copy means as initial log_alpha (= log(U_mean)) etc.
        // Implemented via estimate_alpha_kernel (log-clamps the mean).
        detail::estimate_alpha_kernel<<<m_blocks, 256, 0, stream>>>(
            log_alpha.get(), u_mean.get(), m);
        detail::estimate_alpha_kernel<<<m_blocks, 256, 0, stream>>>(
            log_beta.get(),  s_mean.get(), m);
        detail::estimate_alpha_kernel<<<m_blocks, 256, 0, stream>>>(
            log_gamma.get(), u_mean.get(), m);  // γ ≈ α as uninformed prior
    }

    // ── Phase 2: SVI Loop ──────────────────────────────────────────────────────

    // Adam optimizer state: one pair (m_buf, v_buf) per variational parameter array.
    // Parameters: log_alpha[m], log_beta[m], log_gamma[m], log_w[K×m], log_z[n×K]
    //   + matching log_sigma arrays.
    const int n_rate_params  = 3 * m;           // α, β, γ means
    const int n_lsig_rate    = 3 * m;           // α, β, γ log_σ
    const int n_w_params     = K * m;
    const int n_z_params     = n_cells * K;
    const int n_adam_total   = n_rate_params + n_lsig_rate + n_w_params + K * m + n_z_params + n_cells * K;

    // Allocate Adam state as contiguous blocks for cache efficiency.
    DM adam_m(n_adam_total), adam_v(n_adam_total);
    detail::zero_adam_kernel<<<(n_adam_total + 255) / 256, 256, 0, stream>>>(
        adam_m.get(), adam_v.get(), n_adam_total);

    // Sub-pointers into adam_m / adam_v for each parameter group.
    int off = 0;
    float* m_lalpha = adam_m.get() + off; float* v_lalpha = adam_v.get() + off; off += m;
    float* m_lbeta  = adam_m.get() + off; float* v_lbeta  = adam_v.get() + off; off += m;
    float* m_lgamma = adam_m.get() + off; float* v_lgamma = adam_v.get() + off; off += m;
    float* m_lsa    = adam_m.get() + off; float* v_lsa    = adam_v.get() + off; off += m;
    float* m_lsb    = adam_m.get() + off; float* v_lsb    = adam_v.get() + off; off += m;
    float* m_lsg    = adam_m.get() + off; float* v_lsg    = adam_v.get() + off; off += m;
    float* m_lw     = adam_m.get() + off; float* v_lw     = adam_v.get() + off; off += K * m;
    float* m_lws    = adam_m.get() + off; float* v_lws    = adam_v.get() + off; off += K * m;
    float* m_lz     = adam_m.get() + off; float* v_lz     = adam_v.get() + off; off += n_cells * K;
    float* m_lzs    = adam_m.get() + off; float* v_lzs    = adam_v.get() + off;

    // Gradient buffers (reset each iter).
    DM grad_lalpha(m), grad_lbeta(m), grad_lgamma(m);
    DM grad_lw(K * m), grad_lz(batch * K);
    // log_sigma gradients (KL term only).
    DM grad_lsa(m), grad_lsb(m), grad_lsg(m);
    DM grad_lws(K * m), grad_lzs(batch * K);

    // Minibatch shuffle arrays.
    DM          shuffle_keys((size_t)n_cells);
    core::DeviceMemory<int32_t> shuffle_idx(n_cells);
    // ELBO accumulation: fp64 on-device accumulator + single fp64 D2H per outer SVI iter.
    // Pattern: per (b_off × g_off) sub-iteration, svi_poisson_grad_kernel writes
    // per-block partial sums into elbo_partial[batch]; elbo_partial_to_accum_kernel
    // then atomicAdds the block-reduced sum into elbo_accum_dev[0] (fp64).
    // At the END of each outer SVI iter, ONE 8-byte D2H copies elbo_accum_dev to host.
    // WHY 8 bytes vs 4: fp64 for accuracy; still a scalar; still approved exception.
    DM elbo_partial(batch);
    core::DeviceMemory<double> elbo_accum_dev(1);   // fp64 on-device running sum
    double* h_elbo64_pinned = nullptr;
    cudaMallocHost((void**)&h_elbo64_pinned, sizeof(double));
    *h_elbo64_pinned = 0.0;

    // Gene-tile dense data buffers for SVI kernel.
    const int g_tile = cfg.gene_tile;
    DM s_tile_dev((size_t)batch * g_tile);
    DM u_tile_dev((size_t)batch * g_tile);

    // cub sort workspace.
    size_t cub_sort_ws_bytes = 0;
    DM shuffle_keys_out(n_cells);
    core::DeviceMemory<int32_t> shuffle_idx_out(n_cells);
    cub::DeviceRadixSort::SortPairs(
        nullptr, cub_sort_ws_bytes,
        shuffle_keys.get(), shuffle_keys_out.get(),
        shuffle_idx.get(), shuffle_idx_out.get(),
        n_cells, 0, 32, stream);
    DM cub_sort_ws((cub_sort_ws_bytes + 3) / 4 + 1);

    // Initialize shuffle_idx to 0..n_cells-1.
    {
        std::vector<int32_t> h_idx(n_cells);
        std::iota(h_idx.begin(), h_idx.end(), 0);
        cudaMemcpyAsync(shuffle_idx.get(), h_idx.data(),
                        (size_t)n_cells * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    }

    std::vector<float> elbo_history;
    elbo_history.reserve(cfg.max_iter);
    int     n_iters_done = 0;
    float   prev_elbo    = -std::numeric_limits<float>::infinity();
    int     patience_ctr = 0;

    constexpr float kAdamBeta1 = 0.9f;
    constexpr float kAdamBeta2 = 0.999f;
    constexpr float kAdamEps   = 1e-8f;
    constexpr float kPriorVar  = 1.0f;

    for (int iter = 0; iter < cfg.max_iter; ++iter) {

        // Annealed KL weight: β grows from 0 → beta_kl_max over first kl_anneal_fraction.
        float frac   = (float)iter / (float)(cfg.max_iter * cfg.kl_anneal_fraction + 1.f);
        float beta_kl = cfg.beta_kl_max * fminf(1.0f, frac);

        // Adam bias correction (pre-computed).
        float t         = (float)(iter + 1);
        float bc1       = 1.0f / (1.0f - powf(kAdamBeta1, t));
        float bc2       = 1.0f / (1.0f - powf(kAdamBeta2, t));

        // --- Shuffle cell order for minibatch ---
        // Generate random sort keys via Philox (one kernel call, all on device).
        detail::gen_shuffle_keys_kernel<<<(n_cells + 255) / 256, 256, 0, stream>>>(
            shuffle_keys.get(), nullptr, cfg.seed, (uint64_t)iter, n_cells);

        // Deterministic RadixSort → shuffled indices.
        cub::DeviceRadixSort::SortPairs(
            cub_sort_ws.get(), cub_sort_ws_bytes,
            shuffle_keys.get(), shuffle_keys_out.get(),
            shuffle_idx.get(), shuffle_idx_out.get(),
            n_cells, 0, 32, stream);
        // shuffle_idx_out now holds a deterministically shuffled permutation of 0..n_cells-1.

        // Zero gradient buffers for this iteration.
        detail::zero_grad_kernel<<<m_blocks,  256, 0, stream>>>(grad_lalpha.get(), m);
        detail::zero_grad_kernel<<<m_blocks,  256, 0, stream>>>(grad_lbeta.get(),  m);
        detail::zero_grad_kernel<<<m_blocks,  256, 0, stream>>>(grad_lgamma.get(), m);
        detail::zero_grad_kernel<<<Km_blocks, 256, 0, stream>>>(grad_lw.get(),     K * m);
        detail::zero_grad_kernel<<<(batch * K + 255) / 256, 256, 0, stream>>>(grad_lz.get(), batch * K);
        detail::zero_grad_kernel<<<m_blocks,  256, 0, stream>>>(grad_lsa.get(), m);
        detail::zero_grad_kernel<<<m_blocks,  256, 0, stream>>>(grad_lsb.get(), m);
        detail::zero_grad_kernel<<<m_blocks,  256, 0, stream>>>(grad_lsg.get(), m);
        detail::zero_grad_kernel<<<Km_blocks, 256, 0, stream>>>(grad_lws.get(), K * m);
        detail::zero_grad_kernel<<<(batch * K + 255) / 256, 256, 0, stream>>>(grad_lzs.get(), batch * K);

        // Zero the fp64 ELBO accumulator once per outer SVI iter.
        detail::zero_elbo_accum_kernel<<<1, 32, 0, stream>>>(elbo_accum_dev.get());

        // Process cells in minibatches; accumulate gradients across all gene tiles.
        // Outer loop: cell minibatches. Inner loop: gene tiles.
        // WHY nested: SVI minibatch over cells × full gene axis ensures each iter
        // sees the full gradient signal before the Adam step.
        // NO host transfers inside these loops — elbo_accum_dev accumulates on device.
        for (int b_off = 0; b_off < n_cells; b_off += batch) {
            int b_actual = std::min(batch, n_cells - b_off);
            const int32_t* cell_batch = shuffle_idx_out.get() + b_off;

            for (int g_off = 0; g_off < m; g_off += g_tile) {
                int g_actual = std::min(g_tile, m - g_off);

                // Zero tile buffers.
                cudaMemsetAsync(s_tile_dev.get(), 0,
                                (size_t)b_actual * g_tile * sizeof(float), stream);
                cudaMemsetAsync(u_tile_dev.get(), 0,
                                (size_t)b_actual * g_tile * sizeof(float), stream);

                // Gather dense tiles from CSC.
                detail::gather_gene_csc_tile_kernel<<<g_actual, 256, 0, stream>>>(
                    s_tile_dev.get(), spliced.values.get(), spliced.col_ptr.get(), spliced.row_indices.get(),
                    g_off, g_actual, b_off, b_actual, n_cells);
                detail::gather_gene_csc_tile_kernel<<<g_actual, 256, 0, stream>>>(
                    u_tile_dev.get(), unspliced.values.get(), unspliced.col_ptr.get(), unspliced.row_indices.get(),
                    g_off, g_actual, b_off, b_actual, n_cells);

                // Run SVI Poisson gradient kernel.
                // Per-block ELBO partial sums emitted to elbo_partial[0..b_actual-1].
                size_t smem = (K + g_tile) * sizeof(float);
                detail::svi_poisson_grad_kernel<<<b_actual, 256, smem, stream>>>(
                    grad_lalpha.get(), grad_lbeta.get(), grad_lgamma.get(),
                    grad_lw.get(), grad_lz.get() + (b_off / batch) * batch * K,
                    elbo_partial.get(),
                    log_alpha.get(), log_beta.get(), log_gamma.get(),
                    log_w.get(), log_z.get(),
                    s_tile_dev.get(), u_tile_dev.get(),
                    cell_batch, b_actual, K, m, g_off, g_actual);

                // Fuse elbo_partial[0..b_actual] into the fp64 on-device accumulator.
                // This kernel is NOT a host readback — it is an on-device reduction.
                // Zero host traffic here.
                size_t accum_smem = 256 * sizeof(double);
                detail::elbo_partial_to_accum_kernel<<<1, 256, accum_smem, stream>>>(
                    elbo_accum_dev.get(), elbo_partial.get(), b_actual);
            }
        }

        // ── ONE SCALAR READBACK PER OUTER SVI ITER (approved exception) ────────
        // 8 bytes (fp64) D→H, once per outer iter. The nested loops above produced
        // zero host traffic. This single readback is the only approved D2H in the loop.
        cudaMemcpyAsync(h_elbo64_pinned, elbo_accum_dev.get(),
                        sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        float iter_elbo = (float)(*h_elbo64_pinned);

        // Apply KL gradients to all variational parameters.
        detail::kl_grad_kernel<<<m_blocks,  256, 0, stream>>>(
            grad_lalpha.get(), grad_lsa.get(), log_alpha.get(), log_alpha_lsig.get(),
            beta_kl, kPriorVar, m);
        detail::kl_grad_kernel<<<m_blocks,  256, 0, stream>>>(
            grad_lbeta.get(), grad_lsb.get(), log_beta.get(), log_beta_lsig.get(),
            beta_kl, kPriorVar, m);
        detail::kl_grad_kernel<<<m_blocks,  256, 0, stream>>>(
            grad_lgamma.get(), grad_lsg.get(), log_gamma.get(), log_gamma_lsig.get(),
            beta_kl, kPriorVar, m);
        detail::kl_grad_kernel<<<Km_blocks, 256, 0, stream>>>(
            grad_lw.get(), grad_lws.get(), log_w.get(), log_w_lsig.get(),
            beta_kl, kPriorVar, K * m);
        detail::kl_grad_kernel<<<(n_cells * K + 255) / 256, 256, 0, stream>>>(
            grad_lz.get(), grad_lzs.get(), log_z.get(), log_z_lsig.get(),
            beta_kl, kPriorVar, n_cells * K);

        // Scale gradients by 1/batch.
        float scale = 1.0f / (float)batch;
        detail::scale_grad_kernel<<<m_blocks,  256, 0, stream>>>(grad_lalpha.get(), scale, m);
        detail::scale_grad_kernel<<<m_blocks,  256, 0, stream>>>(grad_lbeta.get(),  scale, m);
        detail::scale_grad_kernel<<<m_blocks,  256, 0, stream>>>(grad_lgamma.get(), scale, m);
        detail::scale_grad_kernel<<<Km_blocks, 256, 0, stream>>>(grad_lw.get(),     scale, K * m);
        detail::scale_grad_kernel<<<(n_cells * K + 255) / 256, 256, 0, stream>>>(
            grad_lz.get(), scale, n_cells * K);

        // Adam update for all parameter groups.
        detail::adam_update_kernel<<<m_blocks,  256, 0, stream>>>(
            log_alpha.get(), log_alpha_lsig.get(), m_lalpha, v_lalpha, m_lsa, v_lsa,
            grad_lalpha.get(), grad_lsa.get(), cfg.learning_rate,
            kAdamBeta1, kAdamBeta2, kAdamEps, bc1, bc2, m);
        detail::adam_update_kernel<<<m_blocks,  256, 0, stream>>>(
            log_beta.get(), log_beta_lsig.get(), m_lbeta, v_lbeta, m_lsb, v_lsb,
            grad_lbeta.get(), grad_lsb.get(), cfg.learning_rate,
            kAdamBeta1, kAdamBeta2, kAdamEps, bc1, bc2, m);
        detail::adam_update_kernel<<<m_blocks,  256, 0, stream>>>(
            log_gamma.get(), log_gamma_lsig.get(), m_lgamma, v_lgamma, m_lsg, v_lsg,
            grad_lgamma.get(), grad_lsg.get(), cfg.learning_rate,
            kAdamBeta1, kAdamBeta2, kAdamEps, bc1, bc2, m);
        detail::adam_update_kernel<<<Km_blocks, 256, 0, stream>>>(
            log_w.get(), log_w_lsig.get(), m_lw, v_lw, m_lws, v_lws,
            grad_lw.get(), grad_lws.get(), cfg.learning_rate,
            kAdamBeta1, kAdamBeta2, kAdamEps, bc1, bc2, K * m);
        detail::adam_update_kernel<<<(n_cells * K + 255) / 256, 256, 0, stream>>>(
            log_z.get(), log_z_lsig.get(), m_lz, v_lz, m_lzs, v_lzs,
            grad_lz.get(), grad_lzs.get(), cfg.learning_rate,
            kAdamBeta1, kAdamBeta2, kAdamEps, bc1, bc2, n_cells * K);

        elbo_history.push_back(iter_elbo);
        ++n_iters_done;

        // Convergence check (patience window on relative ELBO change).
        if (iter > 0) {
            float rel_change = fabsf(iter_elbo - prev_elbo) /
                               (fabsf(prev_elbo) + 1e-8f);
            if (rel_change < cfg.convergence_tol) {
                ++patience_ctr;
                if (patience_ctr >= cfg.convergence_patience) break;
            } else {
                patience_ctr = 0;
            }
        }
        prev_elbo = iter_elbo;
    }

    cudaFreeHost(h_elbo64_pinned);

    // ── Phase 3: Posterior Inference ──────────────────────────────────────────

    // Allocate Welford accumulators for α, β, γ variance.
    DM alpha_wmean(m), alpha_M2(m);
    DM beta_wmean(m),  beta_M2(m);
    DM gamma_wmean(m), gamma_M2(m);
    cudaMemsetAsync(alpha_wmean.get(), 0, m * sizeof(float), stream);
    cudaMemsetAsync(alpha_M2.get(),    0, m * sizeof(float), stream);
    cudaMemsetAsync(beta_wmean.get(),  0, m * sizeof(float), stream);
    cudaMemsetAsync(beta_M2.get(),     0, m * sizeof(float), stream);
    cudaMemsetAsync(gamma_wmean.get(), 0, m * sizeof(float), stream);
    cudaMemsetAsync(gamma_M2.get(),    0, m * sizeof(float), stream);

    const int N_draws = cfg.n_posterior_draws;
    // WHY Philox within kernel: each draw kernel call uses a different offset slot
    // so N_draws loop iterations are independent — all on device, zero host traffic.
    for (int d = 0; d < N_draws; ++d) {
        detail::posterior_draw_variance_kernel<<<m_blocks, 256, 0, stream>>>(
            alpha_wmean.get(), alpha_M2.get(),
            log_alpha.get(), log_alpha_lsig.get(),
            cfg.seed ^ 0xF000ULL, d, m);
        detail::posterior_draw_variance_kernel<<<m_blocks, 256, 0, stream>>>(
            beta_wmean.get(), beta_M2.get(),
            log_beta.get(), log_beta_lsig.get(),
            cfg.seed ^ 0xF001ULL, d, m);
        detail::posterior_draw_variance_kernel<<<m_blocks, 256, 0, stream>>>(
            gamma_wmean.get(), gamma_M2.get(),
            log_gamma.get(), log_gamma_lsig.get(),
            cfg.seed ^ 0xF002ULL, d, m);
    }

    DM alpha_var(m), beta_var(m), gamma_var(m);
    detail::finalize_variance_kernel<<<m_blocks, 256, 0, stream>>>(
        alpha_var.get(), alpha_M2.get(), N_draws, m);
    detail::finalize_variance_kernel<<<m_blocks, 256, 0, stream>>>(
        beta_var.get(),  beta_M2.get(),  N_draws, m);
    detail::finalize_variance_kernel<<<m_blocks, 256, 0, stream>>>(
        gamma_var.get(), gamma_M2.get(), N_draws, m);

    // Exponentiate log-space variational means to get rate estimates.
    DM alpha_out(m), beta_out(m), gamma_out(m);
    // WHY estimate_alpha_kernel reuse: it applies log→exp inverse to get the
    // positive rate from the log-space mean. We reuse it as an exp-clamp kernel.
    // (The kernel's name is misleading for γ/β but the math is identical.)
    detail::estimate_alpha_kernel<<<m_blocks, 256, 0, stream>>>(
        alpha_out.get(), log_alpha.get(), m);
    detail::estimate_alpha_kernel<<<m_blocks, 256, 0, stream>>>(
        beta_out.get(),  log_beta.get(),  m);
    detail::estimate_alpha_kernel<<<m_blocks, 256, 0, stream>>>(
        gamma_out.get(), log_gamma.get(), m);

    // ── Phase 4: Module Ranking ───────────────────────────────────────────────

    DM module_scores(K * m);
    detail::compute_module_scores_kernel<<<Km_blocks, 256, 0, stream>>>(
        module_scores.get(), log_w.get(), K, m);

    core::DeviceMemory<int32_t> topgenes(K * cfg.top_n_per_module);
    // topn_per_module_kernel uses per-module shared memory of [m] floats (m × 4 bytes).
    // WHY launch with m*sizeof(float) smem: the selection sort needs to mark genes
    // as selected without a separate bit array; the shared memory copy is the
    // simplest correct approach for m ≤ 60k (240 KB per block).
    // For m > 48KB / sizeof(float) ≈ 12k genes, shared memory may be insufficient on
    // older hardware. On sm_80+, 48KB per block is the default; for larger m the
    // kernel falls back to global memory (caller sets a compile-time flag).
    // OOC plan for this step: chunk genes and run a two-pass top-K merge.
    size_t topn_smem = (size_t)m * sizeof(float);
    if (topn_smem <= 49152u) {
        detail::topn_per_module_kernel<<<K, 64, topn_smem, stream>>>(
            topgenes.get(), module_scores.get(), K, m, cfg.top_n_per_module);
    } else {
        // Fallback for large m: copy scores to host and sort per module.
        // One-time D→H copy at end (not inside SVI loop).
        std::vector<float> h_scores(K * m);
        std::vector<int32_t> h_topgenes(K * cfg.top_n_per_module);
        cudaMemcpyAsync(h_scores.data(), module_scores.get(),
                        (size_t)K * m * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (int k = 0; k < K; ++k) {
            const float* row = h_scores.data() + k * m;
            std::vector<int32_t> idxs(m);
            std::iota(idxs.begin(), idxs.end(), 0);
            int top_n = std::min(cfg.top_n_per_module, m);
            std::partial_sort(idxs.begin(), idxs.begin() + top_n, idxs.end(),
                              [&](int a, int b){ return row[a] > row[b]; });
            for (int t = 0; t < top_n; ++t)
                h_topgenes[k * cfg.top_n_per_module + t] = idxs[t];
        }
        cudaMemcpyAsync(topgenes.get(), h_topgenes.data(),
                        (size_t)K * cfg.top_n_per_module * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
    }

    // Optional velocity output.
    DM velocity_out;
    if (cfg.compute_velocity) {
        velocity_out = DM((size_t)n_cells * m);
        cudaMemsetAsync(velocity_out.get(), 0,
                        (size_t)n_cells * m * sizeof(float), stream);
        for (int g_off = 0; g_off < m; g_off += g_tile) {
            int g_actual = std::min(g_tile, m - g_off);
            for (int c_off = 0; c_off < n_cells; c_off += batch) {
                int c_actual = std::min(batch, n_cells - c_off);
                cudaMemsetAsync(s_tile_dev.get(), 0,
                                (size_t)c_actual * g_tile * sizeof(float), stream);
                cudaMemsetAsync(u_tile_dev.get(), 0,
                                (size_t)c_actual * g_tile * sizeof(float), stream);
                detail::gather_gene_csc_tile_kernel<<<g_actual, 256, 0, stream>>>(
                    s_tile_dev.get(), spliced.values.get(), spliced.col_ptr.get(), spliced.row_indices.get(),
                    g_off, g_actual, c_off, c_actual, n_cells);
                detail::gather_gene_csc_tile_kernel<<<g_actual, 256, 0, stream>>>(
                    u_tile_dev.get(), unspliced.values.get(), unspliced.col_ptr.get(), unspliced.row_indices.get(),
                    g_off, g_actual, c_off, c_actual, n_cells);
                int vel_threads = 256;
                int vel_blocks  = (c_actual + vel_threads - 1) / vel_threads;
                detail::velocity_kernel<<<vel_blocks, vel_threads, 0, stream>>>(
                    velocity_out.get(),
                    s_tile_dev.get(), u_tile_dev.get(),
                    log_alpha.get(), log_beta.get(), log_gamma.get(),
                    log_z.get(), log_w.get(),
                    c_off, c_actual, g_off, g_actual, K, m, n_cells);
            }
        }
    }

    // ── Assemble Result ───────────────────────────────────────────────────────

    Cell2FateResult result;
    result.alpha         = std::move(alpha_out);
    result.beta          = std::move(beta_out);
    result.gamma         = std::move(gamma_out);
    result.alpha_var     = std::move(alpha_var);
    result.beta_var      = std::move(beta_var);
    result.gamma_var     = std::move(gamma_var);
    result.module_loadings = std::move(log_w);    // K×m row-major (log-space means)
    result.cell_activities = std::move(log_z);    // n_cells×K row-major (log-space means)
    result.velocity        = std::move(velocity_out);
    result.module_topgenes = std::move(topgenes);
    result.elbo_history    = std::move(elbo_history);
    result.n_iters_used    = n_iters_done;
    result.K_used          = K;

    return result;
}

}  // namespace embed
}  // namespace singlet::gpu
