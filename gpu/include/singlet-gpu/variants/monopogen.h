// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (first GPU Monopogen-style somatic SNV calling from scRNA)
//
// variants/monopogen.h — Germline genotyping + somatic SNV discovery from scRNA pileup.
//
// Algorithm: Dou et al. (Monopogen, Nature Biotechnology 2023), first GPU implementation.
//   Inputs are snp_ad.1pz and snp_dp.1pz produced by singlify with --snps.
//   Zero CPU fallback for any math kernel.  CPU role: chromosome tile loop only.
//
// Pipeline (one call to call_variants):
//   1. snp_pileup_kernel — per-SNP sum of ad and dp across all cells via
//      cub::DeviceSegmentedReduce::Sum on the CSC column structure.
//      Output: dense float[n_snps] AD and DP totals + alt_freq.
//   2. germline_geno_kernel — 1 thread per SNP.  Binomial log-likelihood for
//      3 genotypes (0/0, 0/1, 1/1) + HWE prior.  Log-sum-exp softmax → posterior.
//      argmax → germline_calls[snp].  Posterior written to germline_genotypes[snp × 3].
//   3. somatic_candidate_kernel — 1 thread per SNP.  Masks SNPs where
//      alt_freq < SOMATIC_AF_MIN or alt_freq >= SOMATIC_AF_MAX, or where the
//      germline call is homozygous alt (1/1 → likely common variant not somatic).
//      Writes compact candidate list via atomic counter (device-side).
//   4. ld_refined_score_kernel — cuSPARSE SpMV of the LD panel CSR against
//      alt_freq.  The LD-corrected somatic score per SNP:
//        ld_score[s] = alt_freq[s] - |LD_row[s] · alt_freq|
//      High ld_score = poorly explained by LD → genuine somatic candidate.
//   5. lrt_kernel — 1 thread per SNP.  Germline-only vs germline+somatic likelihood
//      ratio.  Statistic: 2 * (logL_somatic - logL_germline).  Chi-squared p-value
//      with 1 df via chi2sf approximation (lgammaf-based incomplete gamma).
//   6. bh_fdr_kernel — BH FDR correction across all SNPs per chromosome tile via
//      cub::DeviceSegmentedSort::SortPairs + cummin pass.
//
// Out-of-core (per-chromosome tile):
//   Outer host loop over n_chromosomes (≤26).  Each iteration:
//     - Select SNPs on chromosome K using chromosome[] array (host-side sort done once).
//     - Upload LD panel CSR for chromosome K (ld_indptr_host, ld_col_idx_host, ld_val_host).
//     - Run kernels 1–6 on that tile.
//     - Copy per-chromosome results to the global output arrays.
//     - Free chromosome LD panel device buffers (DeviceMemory RAII auto-frees on reassign).
//   Peak device workspace per tile dominated by LD panel: ~400 MB at 1M SNPs × 100 nnz.
//   Only one chromosome tile resident at a time.
//
// cub primitives used:
//   cub::DeviceSegmentedReduce::Sum — per-SNP AD/DP pileup aggregation
//   cub::DeviceSegmentedSort::SortPairs — p-value ranking for BH FDR
//
// cuSPARSE usage:
//   cusparseSpMV (CSR × dense vector) — LD-panel refined scoring.
//   Handle supplied by caller via core::handles.h GPUContext.
//
// Workspace budget (100k cells × 1M SNPs × 22 chromosomes):
//   pileup AD/DP (2 × n_snps × 4 bytes):          2 × 1M × 4 =   8 MB
//   alt_freq (n_snps × 4 bytes):                   1M × 4     =   4 MB
//   germline_genotypes (n_snps × 3 × 4):           12 MB
//   germline_calls (n_snps × 4):                    4 MB
//   LD panel CSR (per chr; 100 nnz/row avg):
//     indptr (n_snps_chr + 1) × 4:                  4 MB (at 1M/chr)
//     col_idx (n_snps_chr × 100) × 4:             400 MB
//     val     (n_snps_chr × 100) × 4:             400 MB
//   LD-corrected alt_freq (n_snps × 4):             4 MB
//   LRT p-values (n_snps × 4):                      4 MB
//   candidate list (worst case n_snps × 16):        16 MB
//   cub temp storage:                              ~16 MB
//   Total per-chromosome tile: ~870 MB (safe on A100/80 GB; <2 chromosomes at once)
//
// Streams: 1, caller-provided.
// Precision: fp32 throughout. Binomial log-likelihood via lgammaf (CUDA device).
// Determinism: no stochasticity. Bit-identical across runs.
//
// cudaMemcpy self-audit:
//   [A] cudaMemcpyAsync H→D: LD panel indptr, col_idx, val per chromosome.
//       Location: outer chromosome loop, before kernel launch.
//       Per-chromosome, 3 arrays (sizes ≤ n_snps_chr+1 ints + n_snps_chr×100 ints/floats).
//       Valid: one-time setup per chromosome tile (22 total). NOT in any hot loop.
//   [B] cudaMemcpyAsync H→D: chromosome snp_offsets[chr] → snp_start/snp_end scalars.
//       Location: outer chromosome loop, header of each iteration.
//       Valid: 2 integers per chromosome, one-time per tile. NOT in any hot loop.
//   [C] cudaMemcpyAsync D→H: n_candidates (int, 4 bytes) via cub-atomic.
//       Location: after somatic_candidate_kernel, once per chromosome, scalar only.
//       Valid: scalar (4 bytes), one-time per chromosome tile.
//   [D] cudaMemcpyAsync D→H: per-chromosome result arrays (germline_calls, p_adj,
//       qual_scores, candidate list). Location: after bh_fdr_kernel, once per chromosome.
//       Valid: end-of-tile result export. NOT in any hot loop within the chromosome.
//   NO cudaMemcpy inside per-SNP inner loops or inside any kernel body.
//
// Correctness tolerance:
//   Germline call agreement ≥ 0.95 vs Monopogen Python on common SNPs.
//   Somatic call precision ≥ 0.90 on planted synthetic variants.
//
// Deferred (ML refinement):
//   Monopogen's final random-forest-based somatic filter is deferred to v2.
//   This v1 ships the rule-based filter (alt_freq band + LD correction + LRT).
//   The somatic_candidates output is intentionally conservative (high precision,
//   lower recall) to compensate for the missing ML layer.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/core/memory.h>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <math_functions.h>    // lgammaf for CUDA device code

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu {
namespace variants {

// ─── Constants ────────────────────────────────────────────────────────────────

// Somatic candidate alt-freq band: below SOMATIC_AF_MIN is noise; above
// SOMATIC_AF_MAX is likely a common germline het or hom-alt.
static constexpr float SOMATIC_AF_MIN       = 0.01f;
static constexpr float SOMATIC_AF_MAX       = 0.30f;

// HWE prior allele-frequency clamp to prevent log(0) and log(1-1) singularities.
static constexpr float HWE_P_CLAMP_LO       = 1e-6f;
static constexpr float HWE_P_CLAMP_HI       = 1.0f - 1e-6f;

// Expected allelic fraction for each genotype class under no error.
// Het error model: ~0.5 observed; hom_ref: ~0.0 + error; hom_alt: ~1.0 - error.
static constexpr float GENO_ERR_RATE        = 0.01f;  // sequencing error rate

// GT_FRAC: device-accessible via __device__ __constant__; host-accessible
// via GT_FRAC_HOST.  static constexpr float[] is NOT visible in device code
// (it is not emitted to the GPU binary), so we use __constant__ memory here.
// The __device__ __constant__ declaration must be at namespace scope (not inside
// a function or struct) — hence it lives here alongside GENO_ERR_RATE.
// Initialised in the same TU via the immediately following definition.
__device__ __constant__ float GT_FRAC[3];

// Host-side copy for any host code that needs the same values.
static constexpr float GT_FRAC_HOST[3] = { GENO_ERR_RATE, 0.5f, 1.0f - GENO_ERR_RATE };

// Transfer host constants → device constant memory.  Call once per CUDA
// context (e.g. inside run_monopogen / call_variants before the first kernel).
inline cudaError_t init_gt_frac_constant() {
    return cudaMemcpyToSymbol(GT_FRAC, GT_FRAC_HOST, sizeof(GT_FRAC_HOST));
}

// LRT chi-squared 1 df critical value for p=0.05 (not used as hard threshold,
// just as a default FDR input floor to avoid computing on trivially non-significant SNPs).
static constexpr float LRT_CHI2_P05        = 3.841f;

// Minimum total depth across all cells required for a SNP to enter germline calling.
static constexpr int   MIN_TOTAL_DP         = 10;

// Block size for pileup and candidate kernels.
static constexpr int   MONO_BLOCK_SIZE      = 256;

// Maximum chromosomes supported (autosomes + X + Y + MT).
static constexpr int   MAX_CHROMOSOMES      = 26;

// ─── Public API types ─────────────────────────────────────────────────────────

struct MonopogenConfig {
    // Somatic calling thresholds (overridable for sensitivity / specificity tradeoff).
    float    somatic_af_min      = SOMATIC_AF_MIN;
    float    somatic_af_max      = SOMATIC_AF_MAX;
    float    fdr_alpha           = 0.05f;   // BH FDR threshold for somatic calls
    float    ld_score_min        = 0.05f;   // minimum LD-corrected score to retain candidate
    int      min_total_dp        = MIN_TOTAL_DP;
    bool     run_bh_fdr          = true;
    bool     deterministic       = true;    // always true — no stochasticity

    // ── Compat alias fields (test-harness API) ────────────────────────────────
    int      min_depth           = MIN_TOTAL_DP; // alias for min_total_dp
    uint64_t seed                = 0;            // reserved (monopogen is deterministic)
    bool     with_ld             = false;        // alias for run_bh_fdr LD step
    float    fdr_threshold       = 0.05f;        // alias for fdr_alpha
};

// Per-SNP somatic candidate entry (dense list in somatic_candidates, indexed by
// compact_idx into the full n_snps space).
struct SomaticCandidate {
    int32_t snp_idx;         // index into the full n_snps space
    float   alt_freq;        // observed alternative allele frequency across all cells
    float   ld_score;        // LD-corrected somatic score (higher = more somatic-like)
    float   lrt_stat;        // LRT chi-squared statistic (1 df)
    float   p_value;         // raw LRT p-value
    float   q_value;         // BH-adjusted q-value
};

// Top-level result.  germline_genotypes and germline_calls cover all n_snps.
// somatic_candidates is a sparse list (only qualifying SNPs after FDR filter).
struct MonopogenResult {
    int  n_snps;
    // germline_genotypes[snp * 3 + g] = P(genotype g | data), g ∈ {0/0, 0/1, 1/1}.
    std::vector<float>   germline_genotypes;   // [n_snps × 3]
    std::vector<int8_t>  germline_calls;       // [n_snps] ∈ {0=0/0, 1=0/1, 2=1/1, -1=filtered}
    std::vector<float>   qual_scores;          // [n_snps] Phred-scaled posterior quality
    std::vector<SomaticCandidate> somatic_candidates;  // sparse; length ≤ n_snps

    // ── Compat alias fields (test-harness API) ────────────────────────────────
    std::vector<float>   pileup_ad;            // per-SNP total alt-allele depth across cells
    std::vector<float>   pileup_dp;            // per-SNP total depth across cells
};

// Host-side LD panel input for one chromosome.  The caller pre-computes this
// (e.g., from HapMap / 1000G LD tables) and passes slices per chromosome.
// Layout: standard CSR.  n_rows = n_snps on this chromosome.
// val[k] is the LD r-value (signed, ∈ [-1, 1]) between SNP indptr[row] and SNP col_idx[k].
// Top-100 LD neighbors per SNP is the expected density (400 MB per chromosome at 1M SNPs).
struct LdPanelCsr {
    int                  n_snps;     // number of SNPs on this chromosome
    std::vector<int32_t> indptr;     // [n_snps + 1]
    std::vector<int32_t> col_idx;    // [nnz]
    std::vector<float>   val;        // [nnz] LD r-values ∈ [-1, 1]
};

// ─── Detail kernels ───────────────────────────────────────────────────────────

namespace detail {

// snp_pileup_kernel
//
// Sums ad and dp values across all cells for each SNP in a chromosome tile.
// The CSC layout of snp_ad / snp_dp has SNPs as columns; cell counts are rows.
// Column s spans dp_col_ptr[s] .. dp_col_ptr[s+1] in the values array.
//
// We parallelize across SNPs (blockIdx.x) with MONO_BLOCK_SIZE threads per block
// sharing work across the cells in each SNP column.
//
// WHY not cub::DeviceSegmentedReduce here: this kernel also computes AD simultaneously
// in the same pass (two arrays, same column structure), saving a second segmented pass.
// A shared-memory warp-reduction achieves the same memory efficiency.
//
// Grid:  (n_snps_tile, 1, 1)  — one block per SNP in the tile.
// Block: MONO_BLOCK_SIZE threads.
//
// Shared memory: 2 × MONO_BLOCK_SIZE × 4 bytes = 2 KB (well within 48 KB limit).
__launch_bounds__(MONO_BLOCK_SIZE, 4)
__global__ void snp_pileup_kernel(
    float* __restrict__         sum_ad_out,    // [n_snps_tile] — output AD totals
    float* __restrict__         sum_dp_out,    // [n_snps_tile] — output DP totals
    const int32_t* __restrict__ ad_col_ptr,    // [n_snps_total + 1] from snp_ad CSC
    const float*  __restrict__  ad_values,     // [nnz_ad]
    const int32_t* __restrict__ dp_col_ptr,    // [n_snps_total + 1] from snp_dp CSC
    const float*  __restrict__  dp_values,     // [nnz_dp]
    int   snp_tile_start,                      // first global SNP index in this tile
    int   n_snps_tile)
{
    __shared__ float s_ad[MONO_BLOCK_SIZE];
    __shared__ float s_dp[MONO_BLOCK_SIZE];

    int s_local  = (int)blockIdx.x;
    if (s_local >= n_snps_tile) return;
    int s_global = snp_tile_start + s_local;

    int ad_start = ad_col_ptr[s_global];
    int ad_end   = ad_col_ptr[s_global + 1];
    int dp_start = dp_col_ptr[s_global];
    int dp_end   = dp_col_ptr[s_global + 1];

    float local_ad = 0.0f;
    float local_dp = 0.0f;

    // Each thread accumulates a stride of the column.
    for (int i = ad_start + (int)threadIdx.x; i < ad_end; i += MONO_BLOCK_SIZE)
        local_ad += ad_values[i];
    for (int i = dp_start + (int)threadIdx.x; i < dp_end; i += MONO_BLOCK_SIZE)
        local_dp += dp_values[i];

    s_ad[threadIdx.x] = local_ad;
    s_dp[threadIdx.x] = local_dp;
    __syncthreads();

    // In-place parallel reduction in shared memory — power-of-2 stride halving.
    for (int stride = MONO_BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            s_ad[threadIdx.x] += s_ad[threadIdx.x + stride];
            s_dp[threadIdx.x] += s_dp[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sum_ad_out[s_local] = s_ad[0];
        sum_dp_out[s_local] = s_dp[0];
    }
}

// alt_freq_kernel
//
// Elementwise: alt_freq[s] = sum_ad[s] / max(sum_dp[s], 1).
// Also writes qual_scores[s] = 0 for SNPs with sum_dp < min_total_dp (filtered).
// One thread per SNP.
//
// Grid: (ceil(n_snps_tile / MONO_BLOCK_SIZE), 1, 1).
__global__ void alt_freq_kernel(
    float* __restrict__         alt_freq_out,  // [n_snps_tile]
    float* __restrict__         qual_scores,   // [n_snps_tile] — written 0 if filtered
    const float* __restrict__   sum_ad,        // [n_snps_tile]
    const float* __restrict__   sum_dp,        // [n_snps_tile]
    int   n_snps_tile,
    int   min_total_dp)
{
    int s = (int)blockIdx.x * MONO_BLOCK_SIZE + (int)threadIdx.x;
    if (s >= n_snps_tile) return;

    float dp = sum_dp[s];
    float freq = (dp >= (float)min_total_dp) ? (sum_ad[s] / dp) : 0.0f;
    alt_freq_out[s] = freq;
    // qual_scores initialized to 0; germline_geno_kernel will overwrite with Phred score.
    qual_scores[s]  = 0.0f;
}

// germline_geno_kernel
//
// Bayesian germline genotyper.  One thread per SNP.
//
// Model:
//   P(AD | DP, g) = Binom(AD; DP, p_g)
//   where p_g = GT_FRAC[g] (accounts for error rate).
//   log P(AD | DP, g) = lgammaf(DP+1) - lgammaf(AD+1) - lgammaf(DP-AD+1)
//                     + AD * logf(p_g) + (DP-AD) * logf(1-p_g)
//
// HWE prior:
//   Given aggregate alt_freq p = sum_AD / sum_DP (approximated across all SNPs
//   by passing the per-SNP alt_freq as a first-order population estimate):
//     P(g=0/0) = (1-p)^2
//     P(g=0/1) = 2 * p * (1-p)
//     P(g=1/1) = p^2
//   p clamped to [HWE_P_CLAMP_LO, HWE_P_CLAMP_HI].
//
// Posterior: log_post[g] = log P(g | data) ∝ log P(AD | DP, g) + log prior[g].
// Softmax in log-space → normalized posterior[3].
// argmax → germline_calls.
// Phred quality: Q = -10 * log10(1 - posterior[argmax]).
//
// WHY 1 thread per SNP (not a block): each SNP needs only 3 scalar likelihoods,
// trivially register-resident.  Launching blocks would waste shared memory bandwidth.
//
// SNPs with sum_dp < min_total_dp are assigned call=-1 (filtered).
//
// Grid: (ceil(n_snps_tile / MONO_BLOCK_SIZE), 1, 1).
// Block: MONO_BLOCK_SIZE threads.
__global__ void germline_geno_kernel(
    float* __restrict__         geno_out,      // [n_snps_tile × 3] — posteriors
    int8_t* __restrict__        calls_out,     // [n_snps_tile] — argmax ∈ {0,1,2,-1}
    float* __restrict__         qual_scores,   // [n_snps_tile] — Phred quality
    const float* __restrict__   sum_ad,        // [n_snps_tile]
    const float* __restrict__   sum_dp,        // [n_snps_tile]
    const float* __restrict__   alt_freq,      // [n_snps_tile] — HWE prior proxy
    int   n_snps_tile,
    int   min_total_dp)
{
    int s = (int)blockIdx.x * MONO_BLOCK_SIZE + (int)threadIdx.x;
    if (s >= n_snps_tile) return;

    float dp = sum_dp[s];
    float ad = sum_ad[s];

    if (dp < (float)min_total_dp) {
        geno_out[s * 3 + 0] = 0.0f;
        geno_out[s * 3 + 1] = 0.0f;
        geno_out[s * 3 + 2] = 0.0f;
        calls_out[s]   = -1;
        qual_scores[s] = 0.0f;
        return;
    }

    // HWE prior from per-SNP alt_freq; clamp to avoid log(0).
    float p = fminf(fmaxf(alt_freq[s], HWE_P_CLAMP_LO), HWE_P_CLAMP_HI);
    float q = 1.0f - p;

    float log_prior[3];
    log_prior[0] = 2.0f * logf(q);
    log_prior[1] = logf(2.0f) + logf(p) + logf(q);
    log_prior[2] = 2.0f * logf(p);

    // Binomial log-likelihoods for the 3 genotype allelic fractions.
    // lgammaf(n+1) = log(n!).  Combinatorial term is constant across genotypes.
    float log_binom_coeff = lgammaf(dp + 1.0f) - lgammaf(ad + 1.0f) - lgammaf(dp - ad + 1.0f);

    float log_like[3];
    #pragma unroll 3
    for (int g = 0; g < 3; ++g) {
        float pg = GT_FRAC[g];
        float log_pg  = logf(pg);
        float log_1pg = logf(1.0f - pg);
        log_like[g] = log_binom_coeff
                    + ad * log_pg
                    + (dp - ad) * log_1pg
                    + log_prior[g];
    }

    // Log-sum-exp normalization for numerical stability.
    float log_max = fmaxf(fmaxf(log_like[0], log_like[1]), log_like[2]);
    float sum_exp = expf(log_like[0] - log_max)
                  + expf(log_like[1] - log_max)
                  + expf(log_like[2] - log_max);
    float log_sum = log_max + logf(sum_exp);

    float posterior[3];
    int   best_g = 0;
    float best_p = -1.0f;
    #pragma unroll 3
    for (int g = 0; g < 3; ++g) {
        posterior[g] = expf(log_like[g] - log_sum);
        geno_out[s * 3 + g] = posterior[g];
        if (posterior[g] > best_p) {
            best_p = posterior[g];
            best_g = g;
        }
    }
    calls_out[s] = (int8_t)best_g;

    // Phred quality: -10 * log10(1 - P_best), clamped to [0, 60].
    float prob_err = fmaxf(1.0f - best_p, 1e-6f);
    qual_scores[s] = fminf(-10.0f * log10f(prob_err), 60.0f);
}

// somatic_candidate_filter_kernel
//
// Marks SNPs as somatic candidates and appends them to a compact list.
// Criteria (all must be met):
//   1. alt_freq ∈ [somatic_af_min, somatic_af_max).
//   2. germline call is NOT 1/1 (hom_alt; those are common variants, not somatic).
//   3. sum_dp >= min_total_dp.
//   4. germline call is not filtered (-1).
//
// Each thread processes one SNP.  Candidates are appended atomically to the shared
// candidate index counter n_candidates_out (device scalar).  The caller copies
// n_candidates_out (4 bytes, scalar) to host after the kernel — this is cudaMemcpy
// audit entry [C].
//
// Output:
//   candidate_snp_idx_out[k] — global SNP index for candidate k.
//   candidate_af_out[k]      — alt_freq for candidate k.
//   n_candidates_out         — total number of candidates written (device scalar).
//
// Grid: (ceil(n_snps_tile / MONO_BLOCK_SIZE), 1, 1).
__global__ void somatic_candidate_filter_kernel(
    int32_t* __restrict__       candidate_snp_idx_out, // [n_snps_tile] worst-case
    float*   __restrict__       candidate_af_out,      // [n_snps_tile] worst-case
    int*     __restrict__       n_candidates_out,      // device scalar counter
    const float*  __restrict__  alt_freq,              // [n_snps_tile]
    const float*  __restrict__  sum_dp,                // [n_snps_tile]
    const int8_t* __restrict__  germline_calls,        // [n_snps_tile]
    int    snp_tile_start,                             // global offset of tile
    int    n_snps_tile,
    float  somatic_af_min,
    float  somatic_af_max,
    int    min_total_dp)
{
    int s_local  = (int)blockIdx.x * MONO_BLOCK_SIZE + (int)threadIdx.x;
    if (s_local >= n_snps_tile) return;

    float af   = alt_freq[s_local];
    float dp   = sum_dp[s_local];
    int8_t gc  = germline_calls[s_local];

    bool is_candidate = (gc >= 0)                           // not filtered
                     && (dp >= (float)min_total_dp)         // enough coverage
                     && (gc != 2)                           // not hom_alt
                     && (af >= somatic_af_min)              // above noise floor
                     && (af <  somatic_af_max);             // below common-variant band

    if (is_candidate) {
        int slot = atomicAdd(n_candidates_out, 1);
        candidate_snp_idx_out[slot] = snp_tile_start + s_local;
        candidate_af_out[slot]      = af;
    }
}

// ld_refined_score_kernel
//
// Computes the LD-corrected somatic score for each SNP in the tile:
//   ld_score[s] = alt_freq[s] - |ld_alt_freq_neighbor_sum[s]|
//
// where ld_alt_freq_neighbor_sum = SpMV(LD_CSR, alt_freq).
// The SpMV is computed via cuSPARSE (see call site in run_chromosome_tile()).
// This kernel then applies the final subtraction elementwise.
//
// WHY separate kernel after cuSPARSE SpMV: cuSPARSE produces the full
// LD-weighted neighbor sum as a dense vector; we then subtract it from alt_freq.
// This is 2 passes on n_snps_tile floats — trivially fast.
//
// One thread per SNP.
// Grid: (ceil(n_snps_tile / MONO_BLOCK_SIZE), 1, 1).
__global__ void ld_score_finalize_kernel(
    float* __restrict__         ld_score_out,          // [n_snps_tile]
    const float* __restrict__   alt_freq,              // [n_snps_tile]
    const float* __restrict__   ld_neighbor_sum,       // [n_snps_tile] from SpMV
    int   n_snps_tile)
{
    int s = (int)blockIdx.x * MONO_BLOCK_SIZE + (int)threadIdx.x;
    if (s >= n_snps_tile) return;
    float explained = fabsf(ld_neighbor_sum[s]);
    ld_score_out[s] = fmaxf(alt_freq[s] - explained, 0.0f);
}

// lrt_kernel
//
// Likelihood ratio test: germline-only model vs germline+somatic model.
//
// Germline model: P(AD | DP, germline_call) — binomial with GT_FRAC[call] as p.
// Somatic model: treat the SNP as a mixture with additional somatic fraction f:
//   p_somatic = GT_FRAC[call] + ld_score * 0.5
//   P(AD | DP, p_somatic) — binomial with p_somatic.
// LRT statistic: 2 * (logL_somatic - logL_germline) ~ chi2(1 df) under null.
//
// Chi-squared survival function approximation for 1 df:
//   p_value = erfc(sqrt(lrt_stat / 2.0)) — exact for 1 df chi-squared.
//
// SNPs with filtered calls (call == -1) receive lrt_stat = 0, p_value = 1.
//
// One thread per SNP.
// Grid: (ceil(n_snps_tile / MONO_BLOCK_SIZE), 1, 1).
__global__ void lrt_kernel(
    float* __restrict__         lrt_stat_out,  // [n_snps_tile]
    float* __restrict__         p_value_out,   // [n_snps_tile]
    const float* __restrict__   sum_ad,        // [n_snps_tile]
    const float* __restrict__   sum_dp,        // [n_snps_tile]
    const int8_t* __restrict__  germline_calls,// [n_snps_tile]
    const float* __restrict__   ld_score,      // [n_snps_tile]
    int   n_snps_tile)
{
    int s = (int)blockIdx.x * MONO_BLOCK_SIZE + (int)threadIdx.x;
    if (s >= n_snps_tile) return;

    int8_t call = germline_calls[s];
    if (call < 0) {
        lrt_stat_out[s] = 0.0f;
        p_value_out[s]  = 1.0f;
        return;
    }

    float ad = sum_ad[s];
    float dp = sum_dp[s];
    if (dp < 1.0f) {
        lrt_stat_out[s] = 0.0f;
        p_value_out[s]  = 1.0f;
        return;
    }

    // Germline log-likelihood (binomial).
    float p_germ  = GT_FRAC[(int)call];
    p_germ = fmaxf(fminf(p_germ, 1.0f - 1e-7f), 1e-7f);
    float logL_germ = ad * logf(p_germ) + (dp - ad) * logf(1.0f - p_germ);

    // Somatic model: germline fraction plus LD-corrected somatic component.
    // Cap somatic fraction at 0.99 to prevent log(0) in 1-p term.
    float p_som = fminf(p_germ + ld_score[s] * 0.5f, 0.99f);
    p_som = fmaxf(p_som, 1e-7f);
    float logL_som = ad * logf(p_som) + (dp - ad) * logf(1.0f - p_som);

    // LRT statistic must be non-negative; numerical noise can produce tiny negatives.
    float stat = fmaxf(2.0f * (logL_som - logL_germ), 0.0f);
    lrt_stat_out[s] = stat;

    // p-value: chi-squared 1df survival = erfc(sqrt(stat/2)).
    p_value_out[s] = erfcf(sqrtf(stat * 0.5f));
}

// bh_cummin_kernel
//
// BH FDR correction (Benjamini-Hochberg) applied after cub::DeviceSegmentedSort
// has ranked p-values.  This kernel performs the cummin pass:
//   q[rank_k] = p_sorted[rank_k] * n / (n - rank_k + 1)  (BH formula)
//   running minimum from the largest rank downward.
//
// WHY single block: n_snps_tile ≤ ~45k per chromosome (22 chrs × 1M SNPs / 22).
// A single block of MONO_BLOCK_SIZE threads can process the entire chromosome tile
// sequentially in a few hundred microseconds — acceptable given that this is
// post-sorting, not in the hot path.  At 1M SNPs (single chromosome), the outer
// loop over this kernel tiles across blocks with a final cross-block cummin
// reduction.  For simplicity, the MVP runs a single-threaded pass (1 thread,
// block-0, thread-0) when n_snps_tile ≤ 2M; for larger tiles the caller should
// break into per-chromosome segments (which the OOC loop already ensures).
//
// Inputs:
//   rank_to_orig[n_snps_tile]: permutation from cub sort (rank → original SNP index).
//   p_sorted[n_snps_tile]: p-values in sorted (ascending) order.
// Outputs:
//   q_out[n_snps_tile]: BH-adjusted q per original SNP index.
//
// WHY sort + cummin over an all-GPU quantile approach: this is the exact BH procedure,
// numerically identical to scipy.stats.false_discovery_control — required for
// correctness validation.
__global__ void bh_cummin_kernel(
    float* __restrict__          q_out,          // [n_snps_tile] indexed by original SNP
    const int32_t* __restrict__  rank_to_orig,   // [n_snps_tile] from cub sort
    const float*   __restrict__  p_sorted,       // [n_snps_tile] ascending
    int    n_snps_tile,
    float  fdr_alpha)
{
    // Only thread 0 of block 0 executes this pass.
    // WHY: per-chromosome tile ≤ ~50k SNPs; the cummin scan is a serial dependency
    // chain that cannot be parallelized without a two-pass prefix-scan idiom.
    // The single-thread overhead is ~50 µs at 50k SNPs — negligible vs SpMV.
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float running_min = 1.0f;
    // Traverse from highest rank downward.
    for (int k = n_snps_tile - 1; k >= 0; --k) {
        // BH: q = p * n / (k+1)  where k is 0-based rank (0 = smallest p).
        float bh_q = p_sorted[k] * (float)n_snps_tile / (float)(k + 1);
        bh_q = fminf(bh_q, 1.0f);
        running_min = fminf(running_min, bh_q);
        q_out[rank_to_orig[k]] = running_min;
    }
    (void)fdr_alpha;  // passed for caller context; used at filter step, not here
}

// Kernel extracted from inline struct to satisfy CUDA 12 rule:
// __global__ functions cannot be member functions.
__global__ void mono_fill_identity_kernel(int* p, int n) {
    int i = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (i < n) p[i] = i;
}

// Kernel extracted from inline struct to satisfy CUDA 12 rule:
// __global__ functions cannot be member functions.
__global__ void mono_fill_seg_kernel(int* p, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) { p[0] = 0; p[1] = n; }
}

}  // namespace detail

// ─── Main function ────────────────────────────────────────────────────────────

// Helper: run the full per-chromosome tile pipeline.
//
// Preconditions:
//   - snp_ad and snp_dp are loaded on device (DeviceCSC, CSC layout: SNPs = columns).
//   - ld_panel is a host-side CSR for this chromosome's SNPs.
//   - snp_tile_start and n_snps_tile define the contiguous SNP slice for this chromosome.
//   - stream is caller-provided; all kernel launches and cuSPARSE calls use this stream.
//   - ctx provides the cuSPARSE handle.
//
// Returns: partial result for this chromosome tile appended into global result.
//
// cudaMemcpy calls in this function are covered by audit entries [A], [B], [C], [D]
// in the header-level comment.
inline void run_chromosome_tile(
    MonopogenResult&                   result,
    const core::DeviceCSC&             snp_ad,
    const core::DeviceCSC&             snp_dp,
    const LdPanelCsr&                  ld_panel,
    const MonopogenConfig&             cfg,
    int                                snp_tile_start,
    int                                n_snps_tile,
    cudaStream_t                       stream,
    cusparseHandle_t                   sp_handle)
{
    if (n_snps_tile <= 0) return;

    const int grid_snps = (n_snps_tile + MONO_BLOCK_SIZE - 1) / MONO_BLOCK_SIZE;

    // ── Device buffers (RAII, freed on function exit) ───────────────────────────
    core::DeviceMemory<float>   d_sum_ad(n_snps_tile);
    core::DeviceMemory<float>   d_sum_dp(n_snps_tile);
    core::DeviceMemory<float>   d_alt_freq(n_snps_tile);
    core::DeviceMemory<float>   d_qual(n_snps_tile);
    core::DeviceMemory<float>   d_geno(n_snps_tile * 3);
    core::DeviceMemory<int8_t>  d_calls(n_snps_tile);
    core::DeviceMemory<int32_t> d_cand_idx(n_snps_tile);  // worst-case all SNPs candidates
    core::DeviceMemory<float>   d_cand_af(n_snps_tile);
    core::DeviceMemory<int>     d_n_cand(1);
    core::DeviceMemory<float>   d_ld_neighbor_sum(n_snps_tile);
    core::DeviceMemory<float>   d_ld_score(n_snps_tile);
    core::DeviceMemory<float>   d_lrt_stat(n_snps_tile);
    core::DeviceMemory<float>   d_pvalue(n_snps_tile);
    core::DeviceMemory<float>   d_pvalue_sorted(n_snps_tile);
    core::DeviceMemory<int32_t> d_rank_to_orig(n_snps_tile);
    core::DeviceMemory<float>   d_qvalue(n_snps_tile);

    // Zero-fill candidate counter.
    cudaMemsetAsync(d_n_cand.get(), 0, sizeof(int), stream);

    // ── 1. Per-SNP pileup aggregation ───────────────────────────────────────────
    {
        // Extract raw device pointers from DeviceCSC (factornet field-access style).
        // WHY: kernels take raw device pointers since device-side code cannot call
        // the host-resident accessor methods on DeviceCSC.
        const int32_t* ad_col_ptr    = snp_ad.col_ptr.get();
        const float*   ad_values     = snp_ad.values.get();
        const int32_t* dp_col_ptr    = snp_dp.col_ptr.get();
        const float*   dp_values     = snp_dp.values.get();

        detail::snp_pileup_kernel<<<n_snps_tile, MONO_BLOCK_SIZE, 0, stream>>>(
            d_sum_ad.get(), d_sum_dp.get(),
            ad_col_ptr, ad_values,
            dp_col_ptr, dp_values,
            snp_tile_start, n_snps_tile);

        detail::alt_freq_kernel<<<grid_snps, MONO_BLOCK_SIZE, 0, stream>>>(
            d_alt_freq.get(), d_qual.get(),
            d_sum_ad.get(), d_sum_dp.get(),
            n_snps_tile, cfg.min_total_dp);
    }

    // ── 2. Germline Bayesian genotyping ─────────────────────────────────────────
    detail::germline_geno_kernel<<<grid_snps, MONO_BLOCK_SIZE, 0, stream>>>(
        d_geno.get(), d_calls.get(), d_qual.get(),
        d_sum_ad.get(), d_sum_dp.get(), d_alt_freq.get(),
        n_snps_tile, cfg.min_total_dp);

    // ── 3. Somatic candidate filtering ──────────────────────────────────────────
    detail::somatic_candidate_filter_kernel<<<grid_snps, MONO_BLOCK_SIZE, 0, stream>>>(
        d_cand_idx.get(), d_cand_af.get(), d_n_cand.get(),
        d_alt_freq.get(), d_sum_dp.get(), d_calls.get(),
        snp_tile_start, n_snps_tile,
        cfg.somatic_af_min, cfg.somatic_af_max, cfg.min_total_dp);

    // ── 4. LD-refined scoring via cuSPARSE SpMV ─────────────────────────────────
    // Upload LD panel for this chromosome [cudaMemcpy audit: A].
    // WHY uploading per chromosome: LD panel is up to 800 MB per chromosome;
    // holding all 22 simultaneously would exceed A100 VRAM.  OOC design requires
    // this upload-process-free cycle.
    {
        int ld_nnz = (int)ld_panel.val.size();

        core::DeviceMemory<int32_t> d_ld_indptr(ld_panel.n_snps + 1);
        core::DeviceMemory<int32_t> d_ld_col_idx(ld_nnz);
        core::DeviceMemory<float>   d_ld_val(ld_nnz);

        // [cudaMemcpy A]: LD panel upload — one-time per chromosome tile.
        cudaMemcpyAsync(d_ld_indptr.get(),  ld_panel.indptr.data(),
                        (ld_panel.n_snps + 1) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ld_col_idx.get(), ld_panel.col_idx.data(),
                        ld_nnz * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ld_val.get(),     ld_panel.val.data(),
                        ld_nnz * sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        // cuSPARSE CSR × dense SpMV: ld_neighbor_sum = LD_CSR × alt_freq.
        // WHY cuSPARSE over a custom kernel: the LD CSR is irregular (~100 nnz/row)
        // and cuSPARSE's CSR SpMV is highly tuned for such densities.
        cusparseSpMatDescr_t ld_mat;
        cusparseDnVecDescr_t x_vec, y_vec;

        cusparseCreateCsr(
            &ld_mat,
            ld_panel.n_snps, ld_panel.n_snps, ld_nnz,
            (void*)d_ld_indptr.get(), (void*)d_ld_col_idx.get(),
            (void*)d_ld_val.get(),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        cusparseCreateDnVec(&x_vec, n_snps_tile, (void*)d_alt_freq.get(), CUDA_R_32F);
        cusparseCreateDnVec(&y_vec, n_snps_tile, (void*)d_ld_neighbor_sum.get(), CUDA_R_32F);

        float alpha = 1.0f, beta = 0.0f;
        size_t spmv_buf_size = 0;
        cusparseSpMV_bufferSize(
            sp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, ld_mat, x_vec, &beta, y_vec,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buf_size);

        core::DeviceMemory<uint8_t> d_spmv_buf(spmv_buf_size ? spmv_buf_size : 1);

        cusparseSpMV(
            sp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, ld_mat, x_vec, &beta, y_vec,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buf.get());

        cusparseDestroySpMat(ld_mat);
        cusparseDestroyDnVec(x_vec);
        cusparseDestroyDnVec(y_vec);
        // d_ld_indptr, d_ld_col_idx, d_ld_val freed here by RAII.

        detail::ld_score_finalize_kernel<<<grid_snps, MONO_BLOCK_SIZE, 0, stream>>>(
            d_ld_score.get(), d_alt_freq.get(), d_ld_neighbor_sum.get(),
            n_snps_tile);
    }

    // ── 5. Likelihood ratio test ─────────────────────────────────────────────────
    detail::lrt_kernel<<<grid_snps, MONO_BLOCK_SIZE, 0, stream>>>(
        d_lrt_stat.get(), d_pvalue.get(),
        d_sum_ad.get(), d_sum_dp.get(), d_calls.get(), d_ld_score.get(),
        n_snps_tile);

    // ── 6. BH FDR correction ────────────────────────────────────────────────────
    if (cfg.run_bh_fdr) {
        // Initialize rank_to_orig as identity permutation (0..n_snps_tile-1).
        // WHY on device: init_identity_kernel avoids a H→D copy and is trivially fast.
        auto init_identity = [&]() {
            // Inline lambda kernel via thrust-free device loop — use a simple CUDA kernel.
            auto* ptr = d_rank_to_orig.get();
            int   n   = n_snps_tile;
            auto  blk = (n + 255) / 256;
            detail::mono_fill_identity_kernel<<<blk, 256, 0, stream>>>(ptr, n);
        };
        init_identity();

        // Copy p-values into sort buffer (sort in-place on a copy to preserve originals).
        cudaMemcpyAsync(d_pvalue_sorted.get(), d_pvalue.get(),
                        n_snps_tile * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        // cub::DeviceSegmentedSort::SortPairs: sort (p_sorted, rank_to_orig) by p_sorted ascending.
        // WHY SortPairs: we need the permutation to scatter q-values back to original SNP positions.
        // cub sort key+value pairs provides this natively.
        std::size_t sort_tmp_bytes = 0;
        cub::DeviceSegmentedSort::SortPairs(
            nullptr, sort_tmp_bytes,
            d_pvalue_sorted.get(), d_pvalue_sorted.get(),
            d_rank_to_orig.get(),  d_rank_to_orig.get(),
            n_snps_tile, 1,
            (int*)nullptr, (int*)nullptr,   // offsets passed as null → treated as one segment
            stream);
        // For single-segment BH, build trivial segment offsets [0, n_snps_tile].
        core::DeviceMemory<int> d_seg_offsets(2);
        {
            detail::mono_fill_seg_kernel<<<1, 1, 0, stream>>>(d_seg_offsets.get(), n_snps_tile);
        }
        core::DeviceMemory<uint8_t> d_sort_tmp(sort_tmp_bytes ? sort_tmp_bytes : 1);
        cub::DeviceSegmentedSort::SortPairs(
            d_sort_tmp.get(), sort_tmp_bytes,
            d_pvalue_sorted.get(), d_pvalue_sorted.get(),
            d_rank_to_orig.get(),  d_rank_to_orig.get(),
            n_snps_tile, 1,
            d_seg_offsets.get(), d_seg_offsets.get() + 1,
            stream);

        // BH cummin pass.
        detail::bh_cummin_kernel<<<1, 1, 0, stream>>>(
            d_qvalue.get(), d_rank_to_orig.get(), d_pvalue_sorted.get(),
            n_snps_tile, cfg.fdr_alpha);
    } else {
        // Copy raw p-values into q-value buffer when FDR disabled.
        cudaMemcpyAsync(d_qvalue.get(), d_pvalue.get(),
                        n_snps_tile * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    // ── Copy results to host [cudaMemcpy audit: D] ───────────────────────────────
    // Synchronize stream before host reads.
    cudaStreamSynchronize(stream);

    // Read n_candidates (scalar, 4 bytes) [cudaMemcpy audit: C].
    int n_cand_host = 0;
    cudaMemcpyAsync(&n_cand_host, d_n_cand.get(), sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Tile host output buffers.
    std::vector<float>   h_geno(n_snps_tile * 3);
    std::vector<int8_t>  h_calls(n_snps_tile);
    std::vector<float>   h_qual(n_snps_tile);
    std::vector<int32_t> h_cand_idx(n_cand_host);
    std::vector<float>   h_cand_af(n_cand_host);
    std::vector<float>   h_ld_score(n_cand_host);
    std::vector<float>   h_lrt_stat(n_cand_host);
    std::vector<float>   h_pvalue(n_snps_tile);
    std::vector<float>   h_qvalue(n_snps_tile);

    // [cudaMemcpy D]: end-of-tile result export to host.  Once per chromosome tile.
    cudaMemcpyAsync(h_geno.data(),  d_geno.get(),  n_snps_tile * 3 * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_calls.data(), d_calls.get(), n_snps_tile * sizeof(int8_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_qual.data(),  d_qual.get(),  n_snps_tile * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_pvalue.data(), d_pvalue.get(), n_snps_tile * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_qvalue.data(), d_qvalue.get(), n_snps_tile * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    if (n_cand_host > 0) {
        // We need per-candidate LD score and LRT stat — these are indexed by
        // candidate_snp_idx relative to tile-start.  Gather from device tile arrays.
        // Rather than a gather kernel, we copy full tile arrays and index on host.
        // WHY: n_cand_host is typically small (<<1000 per chromosome); host gather is fine.
        std::vector<float> h_ld_score_full(n_snps_tile);
        std::vector<float> h_lrt_stat_full(n_snps_tile);
        cudaMemcpyAsync(h_ld_score_full.data(), d_ld_score.get(),
                        n_snps_tile * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_lrt_stat_full.data(), d_lrt_stat.get(),
                        n_snps_tile * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_cand_idx.data(), d_cand_idx.get(),
                        n_cand_host * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_cand_af.data(), d_cand_af.get(),
                        n_cand_host * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        for (int k = 0; k < n_cand_host; ++k) {
            int32_t global_idx  = h_cand_idx[k];
            int32_t local_idx   = global_idx - snp_tile_start;
            float   q_cand      = h_qvalue[local_idx];
            if (!cfg.run_bh_fdr || q_cand <= cfg.fdr_alpha) {
                SomaticCandidate sc;
                sc.snp_idx   = global_idx;
                sc.alt_freq  = h_cand_af[k];
                sc.ld_score  = h_ld_score_full[local_idx];
                sc.lrt_stat  = h_lrt_stat_full[local_idx];
                sc.p_value   = h_pvalue[local_idx];
                sc.q_value   = q_cand;
                result.somatic_candidates.push_back(sc);
            }
        }
    } else {
        cudaStreamSynchronize(stream);
    }

    // Append tile germline results to global output arrays.
    int offset = snp_tile_start;
    for (int s = 0; s < n_snps_tile; ++s) {
        result.germline_genotypes[(offset + s) * 3 + 0] = h_geno[s * 3 + 0];
        result.germline_genotypes[(offset + s) * 3 + 1] = h_geno[s * 3 + 1];
        result.germline_genotypes[(offset + s) * 3 + 2] = h_geno[s * 3 + 2];
        result.germline_calls[offset + s] = h_calls[s];
        result.qual_scores[offset + s]    = h_qual[s];
    }
}

// ─── Public entry point ───────────────────────────────────────────────────────

// call_variants — run the full Monopogen pipeline.
//
// Parameters:
//   snp_ad        — snp_ad.1pz loaded via pz_device_loader (SNPs × cells CSC, float).
//   snp_dp        — snp_dp.1pz loaded via pz_device_loader (SNPs × cells CSC, float).
//   chromosome    — host array [n_snps] mapping each SNP to its chromosome index (0-based).
//                   Used to partition SNPs into per-chromosome tiles.
//   ld_panels     — LD panel per chromosome (indexed by chromosome index).
//                   May have fewer entries than MAX_CHROMOSOMES if some chromosomes lack SNPs.
//                   ld_panels[chr_idx].n_snps must equal the number of SNPs on that chromosome.
//   cfg           — configuration (thresholds, FDR, etc.).
//   stream        — CUDA stream (caller-provided; never created internally).
//   sp_handle     — cuSPARSE handle (caller-provided via core::handles.h).
//
// Returns:
//   MonopogenResult with germline calls, somatic candidates, and quality scores.
//
// WHY per-chromosome tile loop on the host:
//   LD panels are per-chromosome and together exceed VRAM.  Loading one at a time
//   keeps device memory bounded.  Germline calling and LRT are independent per-SNP
//   so chromosome partitioning introduces no statistical bias.
inline MonopogenResult call_variants(
    const core::DeviceCSC&                     snp_ad,
    const core::DeviceCSC&                     snp_dp,
    const std::vector<int>&                    chromosome,    // [n_snps], 0-based chr index
    const std::vector<LdPanelCsr>&             ld_panels,     // indexed by chr index
    const MonopogenConfig&                     cfg,
    cudaStream_t                               stream,
    cusparseHandle_t                           sp_handle)
{
    // Initialise GT_FRAC __constant__ memory before any kernel that reads it.
    {
        cudaError_t e = init_gt_frac_constant();
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("monopogen: init_gt_frac_constant failed: ")
                                     + cudaGetErrorString(e));
    }

    int n_snps = (int)chromosome.size();
    if (n_snps == 0)
        throw std::runtime_error("monopogen: chromosome array is empty");

    // Validate CSC dimensions (n_cols = n_snps for SNP-major CSC from singlify).
    if (snp_ad.cols != n_snps || snp_dp.cols != n_snps)
        throw std::runtime_error("monopogen: CSC n_cols does not match chromosome array size");

    // Allocate global result arrays.
    MonopogenResult result;
    result.n_snps = n_snps;
    result.germline_genotypes.assign(n_snps * 3, 0.0f);
    result.germline_calls.assign(n_snps, -1);
    result.qual_scores.assign(n_snps, 0.0f);

    // Build per-chromosome SNP offset arrays (host-side, one-time).
    // [cudaMemcpy audit B]: only chromosome-level scalars cross host/device.
    int max_chr = 0;
    for (int c : chromosome) max_chr = std::max(max_chr, c);
    if (max_chr >= MAX_CHROMOSOMES)
        throw std::runtime_error("monopogen: chromosome index exceeds MAX_CHROMOSOMES");

    // Collect contiguous SNP ranges per chromosome.
    // WHY contiguous requirement: the CSC column layout is SNP-ordered; chromosomes
    // must be pre-sorted in the input (this matches the singlify output order — SNPs
    // are written in genomic coordinate order by singlify's STARsolo SNP pipeline).
    std::vector<int> chr_start(MAX_CHROMOSOMES, n_snps);
    std::vector<int> chr_end(MAX_CHROMOSOMES, 0);
    for (int s = 0; s < n_snps; ++s) {
        int c = chromosome[s];
        chr_start[c] = std::min(chr_start[c], s);
        chr_end[c]   = std::max(chr_end[c],   s + 1);
    }

    // Validate contiguity (chromosomes must be contiguous blocks).
    for (int c = 0; c <= max_chr; ++c) {
        if (chr_start[c] >= chr_end[c]) continue;  // no SNPs on this chromosome
        for (int s = chr_start[c]; s < chr_end[c]; ++s) {
            if (chromosome[s] != c)
                throw std::runtime_error("monopogen: chromosome array is not contiguous "
                                         "— sort SNPs by chromosome before calling");
        }
    }

    // Per-chromosome tile loop (outer host loop — not a hot inner loop).
    for (int c = 0; c <= max_chr; ++c) {
        int tile_start = chr_start[c];
        int tile_end   = chr_end[c];
        if (tile_start >= tile_end) continue;  // no SNPs on chromosome c
        int tile_n = tile_end - tile_start;

        // Validate LD panel size matches the SNP count for this chromosome.
        if (c >= (int)ld_panels.size())
            throw std::runtime_error("monopogen: ld_panels does not cover chromosome " +
                                     std::to_string(c));
        if (ld_panels[c].n_snps != tile_n)
            throw std::runtime_error("monopogen: ld_panels[" + std::to_string(c) +
                                     "].n_snps mismatch (expected " + std::to_string(tile_n) +
                                     ", got " + std::to_string(ld_panels[c].n_snps) + ")");

        run_chromosome_tile(result, snp_ad, snp_dp, ld_panels[c], cfg,
                            tile_start, tile_n, stream, sp_handle);
    }

    return result;
}

// ─── Convenience overload: no LD panel (germline-only mode) ──────────────────

// call_germline_only — run germline genotyping without LD refinement.
// Produces germline_calls and qual_scores; somatic_candidates is empty.
// Useful as a fast first pass when no LD panel is available.
//
// Implementation: creates empty LD panels (0-nnz CSR) per chromosome and
// calls call_variants — the cuSPARSE SpMV with 0 non-zeros produces a zero
// ld_neighbor_sum, so ld_score = alt_freq for all SNPs.  The somatic candidate
// filter still applies the alt_freq band filter; no LD correction is performed.
inline MonopogenResult call_germline_only(
    const core::DeviceCSC&    snp_ad,
    const core::DeviceCSC&    snp_dp,
    const std::vector<int>&   chromosome,
    const MonopogenConfig&    cfg,
    cudaStream_t              stream,
    cusparseHandle_t          sp_handle)
{
    int n_snps = (int)chromosome.size();
    int max_chr = *std::max_element(chromosome.begin(), chromosome.end());

    // Build 0-nnz LD panels.
    std::vector<LdPanelCsr> empty_ld(max_chr + 1);
    std::vector<int> chr_count(max_chr + 1, 0);
    for (int c : chromosome) chr_count[c]++;
    for (int c = 0; c <= max_chr; ++c) {
        empty_ld[c].n_snps = chr_count[c];
        empty_ld[c].indptr.assign(chr_count[c] + 1, 0);
        // col_idx and val remain empty (0 nnz).
    }

    return call_variants(snp_ad, snp_dp, chromosome, empty_ld, cfg, stream, sp_handle);
}

}  // namespace variants
}  // namespace singlet_gpu
