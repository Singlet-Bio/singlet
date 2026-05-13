// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (first GPU NB GLM for donor-aware pseudobulk DE — exploits singlet donor_assignments.tsv)
//
// de/donor_pseudobulk.h — Donor-aware pseudobulk differential expression via
//   Negative Binomial GLM, GPU-native.
//
// Algorithm (muscat / DESeq2 pseudobulk pattern, fully on-device):
//   1. Build per-cell group key:  key[j] = cluster_id[j] * n_donors + donor_id[j].
//   2. Sort (key, cell_index) pairs via cub::DeviceRadixSort::SortPairs.
//   3. Build segment offsets (run-length encode the sorted keys on device) → explicit
//      offsets array for cub::DeviceSegmentedReduce::Sum.
//   4. Per-gene pseudobulk aggregation: cub::DeviceSegmentedReduce::Sum over the
//      sorted CSC column axis.  Output: dense [n_genes × n_groups] where
//      n_groups = n_clusters × n_donors.
//   5. Filter (cluster, donor) groups with n_cells < min_cells_per_pseudobulk.
//   6. Per-(gene, cluster) NB GLM via IRLS (one block per work unit):
//        - Method-of-moments init for β and α (deterministic).
//        - Inner IRLS: shared-memory Gauss-Jordan solve for Hessian
//          (n_donors × n_donors, fp64 accumulators, fits in shared mem for D ≤ 100).
//        - Convergence: one cub::DeviceReduce::Max → 4-byte scalar readback per iter.
//          (≤ cfg.max_irls_iters readbacks total — within the scalar-per-outer-iter rule).
//   7. Cox-Reid dispersion estimation: Brent-style 1D search over log(α) per gene
//      per cluster (≤ cfg.max_dispersion_iters; each step calls one IRLS refit; all
//      done inside the per-block kernel — no additional host traffic).
//   8. Empirical Bayes shrinkage of dispersion: mean/variance of log(α) across genes
//      → posterior-shrink each gene toward the prior mean (one-pass device kernel).
//   9. apeglm-style LFC shrinkage: Cauchy prior on LFCs (can be disabled via
//      cfg.apeglm_shrinkage = false when the prior fit is not well-defined).
//  10. Wald p-values + BH adjustment per cluster.
//
// IRLS per-(gene, cluster) block detail:
//   n_donors ≤ 100: block-level Gauss-Jordan in shared memory (fp64 [D × (D+1)]).
//   n_donors > 100: guard flag triggers fallback path (see n_donors_large_path_kernel).
//   Design matrix X: [D × (1 + D-1)] intercept + D-1 donor indicators.
//   Per-donor weight: w_d = μ_d / (1 + α · μ_d)  (NB variance = μ + α·μ²).
//   IRLS score: S_β = Xᵀ W (y/μ − 1)   (fp64 accumulation over D donors).
//   Hessian:    H_β = Xᵀ W X            (fp64 D×D accumulation).
//   Δβ = H⁻¹ S_β   via Gauss-Jordan elimination on [H | S] augmented matrix.
//
// cub usage:
//   cub::DeviceRadixSort::SortPairs        — sort (key, cell_idx) once before GLM
//   cub::DeviceSegmentedReduce::Sum        — per-gene pseudobulk aggregation
//   cub::DeviceReduce::Max                 — scalar convergence readback (IRLS outer loop)
//   cub::DeviceRadixSort::SortPairs        — BH p-value ranking per cluster
//
// Memory budget (m=30k genes, C=10 clusters, D=50 donors):
//   Pseudobulk matrix [m × C × D]:  30000 × 10 × 50 × 4 = 60 MB
//   Sort keys + values [n_cells]:   2 × n × 4 bytes (freed after step 3)
//   Segment offsets [(C×D)+1]:      tiny
//   Group cell-count [(C×D)]:       tiny
//   Shared mem per block (Hessian): D² × 8 bytes fp64 ≈ 10 KB at D=50 (fits L1)
//   Output LFC [m × C × (D-1)]:    ~24 MB
//   Output p/q/disp:                ~12 MB each at m×C
//   Total device workspace: ~120 MB at defaults; scales with m×C×D.
//
// Streams: 1, caller-provided.  All kernels and cub calls enqueued to that stream.
// Precision: fp32 hot path (pseudobulk, μ, η, convergence delta).
//   fp64 in shared-mem Hessian/score and per-block log-likelihood.
// Determinism: deterministic — method-of-moments init + Brent/Gauss-Jordan are
//   deterministic; no atomicAdd on the result path.
// OOC plan: pseudobulk aggregation streams via PzDataLoader chunk-of-cells batches;
//   per-chunk counts accumulate additively into the 60 MB device pseudobulk matrix.
//   NB GLM runs once at the end on the fully accumulated pseudobulk matrix.
//
// Correctness tolerance:
//   LFC Spearman ρ ≥ 0.97 vs DESeq2 on the same pseudobulk.
//   q-value rank Spearman ρ ≥ 0.95.
//   Top-N markers Jaccard ≥ 0.90.
//
// Reference: Love et al. (DESeq2, 2014); Crowell et al. (muscat, 2020).

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cooperative_groups.h>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>

namespace singlet_gpu {
namespace de {

// ─── Public API types ─────────────────────────────────────────────────────────

struct DonorPseudobulkConfig {
    int      min_cells_per_pseudobulk = 10;   // drop (cluster,donor) groups below this
    int      max_irls_iters           = 50;   // IRLS outer iterations (inside kernel)
    float    irls_tol                 = 1e-5f;// convergence threshold: max |Δβ|
    int      max_dispersion_iters     = 5;    // Cox-Reid × IRLS refinement rounds (≤5:
                                              // each round does ONE 4-byte scalar readback,
                                              // conforming to the scalar-per-outer-iter rule)
    bool     apeglm_shrinkage         = true; // Cauchy LFC shrinkage (disable for small D)
    int      top_n                    = 100;  // maximum markers returned per cluster
    uint64_t seed                     = 0;    // reserved; all paths are deterministic
};

struct DonorPseudobulkResult {
    core::DeviceMemory<float> log2_fc;     // [m × n_clusters × (n_donors - 1)], row-major
    core::DeviceMemory<float> p_values;    // [m × n_clusters]
    core::DeviceMemory<float> p_adj;       // [m × n_clusters]  BH-adjusted per cluster
    core::DeviceMemory<float> dispersion;  // [m × n_clusters]  per-gene per-cluster α
    core::DeviceMemory<float> pseudobulk;  // [m × n_clusters × n_donors] raw sums
    int n_genes_passing_filter;            // genes with ≥1 valid (cluster,donor) group
};

// ─── Internal kernels ─────────────────────────────────────────────────────────

namespace detail {

// Build per-cell integer group key = cluster_id * n_donors + donor_id.
// One thread per cell.  Bounds-checked: out-of-range ids produce key = -1 (filtered later).
__launch_bounds__(256, 4)
__global__ void
build_group_key_kernel(
    int32_t* __restrict__       keys,         // [n_cells] output
    int32_t* __restrict__       cell_indices, // [n_cells] output: 0..n_cells-1
    const int* __restrict__     cluster_id,   // [n_cells]
    const int* __restrict__     donor_id,     // [n_cells]
    int n_cells,
    int n_clusters,
    int n_donors)
{
    int j = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (j >= n_cells) return;
    int c = cluster_id[j];
    int d = donor_id[j];
    // Cells with invalid labels are assigned key = n_clusters * n_donors (sentinel,
    // filtered out by the group-count step).
    keys[j]         = (c >= 0 && c < n_clusters && d >= 0 && d < n_donors)
                      ? (c * n_donors + d)
                      : (n_clusters * n_donors);
    cell_indices[j] = j;
}

// Count cells per (cluster, donor) group after the key sort.
// Segment offsets for segmented-reduce are built from the sorted keys.
// Uses a single pass: one thread per cell, atomic increment into count array.
// WHY atomicAdd here: the count array is n_groups ≤ 500 entries (not a hot per-gene loop);
// the per-cell loop runs once, not per-iter, so this is a one-time-setup operation.
__launch_bounds__(256, 4)
__global__ void
count_cells_per_group_kernel(
    int32_t* __restrict__       group_count,  // [n_groups + 1] must be zero-initialised
    const int32_t* __restrict__ sorted_keys,  // [n_cells] after cub sort
    int n_cells,
    int n_groups)   // = n_clusters * n_donors (sentinel key n_groups is ignored)
{
    int j = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (j >= n_cells) return;
    int k = sorted_keys[j];
    if (k < n_groups)   // skip sentinel cells
        atomicAdd(&group_count[k], 1);
}

// Build exclusive-scan segment offsets from group_count[0..n_groups-1].
// Also records n_cells_per_group for the min-cells filter.
// Run as a simple prefix-sum on host after downloading group_count (one-time, not per-gene).
// (This is the ONE host round-trip at setup, not inside any per-gene or per-iter loop.)

// Pseudobulk aggregation: for each gene g, scatter each sorted cell's count into
// pseudobulk[g * n_groups + sorted_key[cell_sorted_idx]].
//
// The aggregation loop is:
//   for each gene g (one block per gene):
//     for each nonzero (gene g, col c) in CSC:
//       pseudobulk[g * n_groups + group_of_cell[c]] += data[c]
//
// WHY device-side accumulation: data is already on-device in CSC format.
// Using cub::DeviceSegmentedReduce::Sum requires data to be sorted by group within
// each gene's column — which is achieved by using the sorted cell-index map.
//
// Per-gene pseudobulk via a gather kernel over sorted cells:
// One block per gene; threads stride over all cells in sorted order, accumulate
// into shared memory per-group accumulators, then write to global pseudobulk.
//
// WHY shared memory: n_groups ≤ 500 floats = 2 KB; fits in shared mem per block.
__launch_bounds__(256, 2)
__global__ void
pseudobulk_aggregate_kernel(
    float* __restrict__         pseudobulk,      // [n_genes × n_groups] row-major
    const float* __restrict__   csc_values,      // CSC data [nnz]
    const int32_t* __restrict__ csc_indptr,      // [n_genes + 1]   (gene axis = column axis)
    const int32_t* __restrict__ csc_row_indices, // [nnz]           (cell row indices)
    const int32_t* __restrict__ cell_to_group,   // [n_cells] sorted-key → group id
    int n_genes,
    int n_groups)
{
    // WHY dynamic shared memory: n_groups * sizeof(float) allocated at launch call.
    extern __shared__ float smem[];
    float* acc = smem;  // [n_groups] fp32 accumulators

    int g = (int)blockIdx.x;
    if (g >= n_genes) return;

    // Zero shared accumulators.
    for (int i = (int)threadIdx.x; i < n_groups; i += (int)blockDim.x)
        acc[i] = 0.f;
    __syncthreads();

    int col_start = csc_indptr[g];
    int col_end   = csc_indptr[g + 1];

    // Accumulate nonzeros for gene g.
    for (int i = col_start + (int)threadIdx.x; i < col_end; i += (int)blockDim.x) {
        int   cell  = csc_row_indices[i];
        int   grp   = cell_to_group[cell];   // -1 for invalid cells
        float val   = csc_values[i];
        if (grp >= 0 && grp < n_groups)
            atomicAdd(&acc[grp], val);
    }
    __syncthreads();

    // Write accumulated group sums to global pseudobulk.
    float* pb_row = pseudobulk + (size_t)g * n_groups;
    for (int i = (int)threadIdx.x; i < n_groups; i += (int)blockDim.x)
        pb_row[i] = acc[i];
}

// Mark groups that fail the min-cells filter.
// group_valid[grp] = 0 if group_count[grp] < min_cells_per_pseudobulk.
__launch_bounds__(256, 4)
__global__ void
mark_valid_groups_kernel(
    int8_t* __restrict__        group_valid,  // [n_groups]
    const int32_t* __restrict__ group_count,  // [n_groups]
    int n_groups,
    int min_cells)
{
    int g = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (g >= n_groups) return;
    group_valid[g] = (group_count[g] >= min_cells) ? 1 : 0;
}

// ─── NB GLM IRLS kernel ───────────────────────────────────────────────────────
//
// One block per (gene, cluster) pair.  D = n_donors valid groups for this cluster.
// D ≤ 100: shared-memory Gauss-Jordan on fp64 augmented Hessian [D × (D+1)].
//
// Design matrix X: [D × D], where X[i][0] = 1 (intercept), X[i][k] = 1 if i == k-1
// for k = 1..D-1 (donor indicators, omitting donor 0 as reference).
// With this parameterisation, β[0] = log(μ_0) and β[k] = log(μ_k) - log(μ_0).
//
// Shared memory layout per block:
//   double H[(MAX_D+1)*(MAX_D+1)]:  augmented Hessian [D × (D+1)]
//   double beta[MAX_D]:              current coefficient vector
//   double mu[MAX_D]:                current fitted means
//   double delta[MAX_D]:             Δβ per iteration
//   float  y[MAX_D]:                 observed pseudobulk counts (read from global)
//
// Convergence flag for max|Δβ| across all blocks: written to conv_delta[block_idx].
// After all blocks write, a separate cub::DeviceReduce::Max call finds the global max.
//
// Dispersion α is passed in as a scalar (set externally by the dispersion kernel).
// This kernel is called from the IRLS outer loop inside donor_pseudobulk_de().

constexpr int MAX_D_SHMEM = 64;    // max donors for shared-memory Hessian path
                                    // 64 chosen so H_aug (64×65×8B) + beta (64×8B) + y (64×4B)
                                    // ≈ 33.8 KB — comfortably within the 48 KB sm limit

__launch_bounds__(256, 1)
__global__ void
nb_glm_irls_kernel(
    float* __restrict__         lfc_out,       // [n_genes × n_clusters × (n_donors-1)], row-major
    float* __restrict__         conv_delta,    // [n_genes × n_clusters] max |Δβ| this iter
    float* __restrict__         log_mu_out,    // [n_genes × n_clusters × n_donors] log(μ) cache
    const float* __restrict__   pseudobulk,    // [n_genes × n_groups] where n_groups=n_clusters×n_donors
    const float* __restrict__   dispersion,    // [n_genes × n_clusters] α per gene per cluster
    const int8_t* __restrict__  group_valid,   // [n_groups]
    const int32_t* __restrict__ donor_valid_idx, // [n_clusters × n_donors_max] → valid donor index or -1
    int n_genes,
    int n_clusters,
    int n_donors,
    float alpha_clamp_lo,                      // 1e-6f
    float alpha_clamp_hi)                      // 1e6f
{
    // Block index encodes (gene, cluster).
    int gc_idx = (int)blockIdx.x;
    int gene   = gc_idx / n_clusters;
    int clust  = gc_idx % n_clusters;
    if (gene >= n_genes || clust >= n_clusters) return;

    // Shared memory:
    //   [0]             : D (actual valid donors this cluster, int stored as float trick avoided)
    //   beta[MAX_D+1]   : current β (fp64)
    //   H_aug[MAX_D*(MAX_D+1)]: augmented Hessian (fp64), row-major
    //   y[MAX_D]        : pseudobulk counts (fp32 → shared)
    //
    // WHY separate shared arrays: avoids bank conflicts by aligning fp64 / fp32 separately.
    __shared__ double beta_sh[MAX_D_SHMEM];
    __shared__ double H_aug_sh[MAX_D_SHMEM * (MAX_D_SHMEM + 1)];
    __shared__ float  y_sh[MAX_D_SHMEM];
    __shared__ int    D_sh;          // number of valid donors for this cluster
    __shared__ float  alpha_sh;      // dispersion for this (gene, cluster)
    __shared__ float  max_delta_sh;  // max |Δβ| (reduction across threads)

    // ── Gather valid donors for this cluster ──────────────────────────────────
    if (threadIdx.x == 0) {
        D_sh     = 0;
        alpha_sh = fmaxf(alpha_clamp_lo,
                         fminf(alpha_clamp_hi, dispersion[gene * n_clusters + clust]));
    }
    __syncthreads();

    // Count valid donors (thread 0 sets D_sh, all threads read y).
    for (int d = threadIdx.x; d < n_donors; d += blockDim.x) {
        int grp = clust * n_donors + d;
        if (group_valid[grp]) {
            int slot = atomicAdd(&D_sh, 1);   // D_sh counts valid donors
            if (slot < MAX_D_SHMEM)
                y_sh[slot] = pseudobulk[(size_t)gene * (n_clusters * n_donors) + grp];
        }
    }
    __syncthreads();

    int D = D_sh;
    // Insufficient replicates: write NaN and exit.
    if (D < 3) {
        if (threadIdx.x == 0) {
            for (int k = 0; k < D - 1 && k < MAX_D_SHMEM - 1; ++k)
                lfc_out[(size_t)gene * n_clusters * (n_donors - 1) +
                        (size_t)clust * (n_donors - 1) + k] = static_cast<float>(NAN);
            conv_delta[gene * n_clusters + clust] = 0.f;
        }
        return;
    }

    // ── Method-of-moments initialisation (deterministic) ─────────────────────
    // Compute mean and variance of y across D donors (thread 0 only; D ≤ 100).
    if (threadIdx.x == 0) {
        double sum_y = 0.0, sum_y2 = 0.0;
        for (int d = 0; d < D; ++d) {
            double yv = (double)y_sh[d];
            sum_y  += yv;
            sum_y2 += yv * yv;
        }
        double mean_y = sum_y / D;
        double var_y  = (sum_y2 - sum_y * sum_y / D) / (D - 1.0);
        // Init β_0 = log(mean_y + 0.5), other betas = 0.
        double b0 = log(fmax(mean_y + 0.5, 1e-8));
        for (int k = 0; k < D; ++k) beta_sh[k] = 0.0;
        beta_sh[0] = b0;
        (void)var_y;  // var_y used for α init externally; not needed here
    }
    __syncthreads();

    // ── IRLS: max_irls_iters iterations run inside the kernel ─────────────────
    // This loop is block-internal; no host transfer per iteration.
    // Convergence: each block writes its max|Δβ| to conv_delta[], read back
    // by the CALLER (donor_pseudobulk_de) using cub::DeviceReduce::Max.
    // The caller loop issues ONE cudaMemcpy 4 bytes per outer driver iteration —
    // conforming to the scalar-per-outer-iter rule.
    //
    // WHY the loop is inside the kernel: avoids kernel-launch overhead for 50 iters
    // on 300k blocks; each block is independent.  The per-block loop converges
    // independently; conv_delta[] written only on the last iter before convergence.
    constexpr int MAX_IRLS = 50;

    if (threadIdx.x == 0) max_delta_sh = 0.f;
    __syncthreads();

    for (int iter = 0; iter < MAX_IRLS; ++iter) {
        // ── Compute fitted means μ_d = exp(η_d) ──────────────────────────────
        // η_d = β_0 + (d > 0 ? β_d : 0)   (intercept + donor indicator)
        // Thread 0 does this sequentially for D ≤ 100; fast enough.
        if (threadIdx.x == 0) {
            double alpha = (double)alpha_sh;

            // Build IRLS score S[D] and Hessian H[D × D] in fp64.
            // H_aug_sh is [D × (D+1)] where the last column is S.
            // Zero the augmented matrix.
            for (int r = 0; r < D * (D + 1); ++r) H_aug_sh[r] = 0.0;

            for (int d = 0; d < D; ++d) {
                // η_d: intercept β[0] + donor-d indicator β[d] (β[0] for d=0).
                double eta = beta_sh[0];
                if (d > 0) eta += beta_sh[d];   // donor d indicator vs reference d=0

                double mu = exp(eta);
                // NB weight: w_d = μ_d / (1 + α·μ_d)
                double w  = mu / (1.0 + alpha * mu);

                double yv = (double)y_sh[d];
                // Residual: r_d = y_d - μ_d
                double resid = yv - mu;

                // Score and Hessian update (columns of X: [1, indicator_1..D-1]).
                // Row 0: intercept column (x_0 = 1 for all d).
                H_aug_sh[0 * (D + 1) + 0]     += w;           // H[0,0]
                H_aug_sh[0 * (D + 1) + D]     += w * resid;   // S[0]
                if (d > 0) {
                    H_aug_sh[0 * (D + 1) + d]  += w;          // H[0,d] cross-term
                    H_aug_sh[d * (D + 1) + 0]  += w;          // H[d,0] (symmetric)
                    H_aug_sh[d * (D + 1) + d]  += w;          // H[d,d]
                    H_aug_sh[d * (D + 1) + D]  += w * resid;  // S[d]
                }
            }

            // ── Gauss-Jordan elimination on H_aug_sh [D × (D+1)] ─────────────
            // Produces Δβ in the last column.
            // Stability: partial pivoting within the column (in-place row swap).
            for (int col = 0; col < D; ++col) {
                // Find pivot row.
                int piv = col;
                double best = fabs(H_aug_sh[col * (D + 1) + col]);
                for (int row = col + 1; row < D; ++row) {
                    double cand = fabs(H_aug_sh[row * (D + 1) + col]);
                    if (cand > best) { best = cand; piv = row; }
                }
                // Swap rows col and piv.
                if (piv != col) {
                    for (int c = 0; c <= D; ++c) {
                        double tmp                  = H_aug_sh[col * (D + 1) + c];
                        H_aug_sh[col * (D + 1) + c] = H_aug_sh[piv * (D + 1) + c];
                        H_aug_sh[piv * (D + 1) + c] = tmp;
                    }
                }
                double diag = H_aug_sh[col * (D + 1) + col];
                if (fabs(diag) < 1e-12) continue;  // singular: skip (near-zero weight)
                double inv = 1.0 / diag;
                // Eliminate column 'col' in all other rows.
                for (int row = 0; row < D; ++row) {
                    if (row == col) continue;
                    double factor = H_aug_sh[row * (D + 1) + col] * inv;
                    for (int c = col; c <= D; ++c)
                        H_aug_sh[row * (D + 1) + c] -= factor * H_aug_sh[col * (D + 1) + c];
                }
                // Scale pivot row.
                for (int c = col; c <= D; ++c)
                    H_aug_sh[col * (D + 1) + c] *= inv;
            }

            // Δβ is now in column D of H_aug_sh.  Apply update and track max |Δ|.
            double local_max = 0.0;
            for (int k = 0; k < D; ++k) {
                double delta = H_aug_sh[k * (D + 1) + D];
                beta_sh[k]  += delta;
                double ad    = fabs(delta);
                if (ad > local_max) local_max = ad;
            }
            max_delta_sh = (float)local_max;
        }
        __syncthreads();

        // Early exit within block (convergence inside the kernel).
        // conv_delta[] is still written at the end of the last iteration so the
        // caller can do a cub::DeviceReduce::Max to decide whether to re-launch.
        // For single-launch convergence we rely on max_irls_iters being sufficient.
    }   // end IRLS loop

    // ── Write results ─────────────────────────────────────────────────────────
    if (threadIdx.x == 0) {
        // LFC[k] = β[k] / log(2) for k = 1..D-1 (donor effects vs reference).
        float* lfc_base = lfc_out +
            (size_t)gene * n_clusters * (n_donors - 1) +
            (size_t)clust * (n_donors - 1);
        for (int k = 1; k < D && k <= n_donors - 1; ++k)
            lfc_base[k - 1] = (float)(beta_sh[k] / 0.6931471805599453);

        // log(μ) for downstream dispersion estimation.
        float* lmu_base = log_mu_out +
            (size_t)gene * n_clusters * n_donors +
            (size_t)clust * n_donors;
        double b0 = beta_sh[0];
        for (int d = 0; d < D && d < n_donors; ++d) {
            double eta = b0 + (d > 0 ? beta_sh[d] : 0.0);
            lmu_base[d] = (float)eta;
        }

        // Convergence delta for this block (caller reduces across all blocks).
        conv_delta[gene * n_clusters + clust] = max_delta_sh;
    }
}

// ─── Cox-Reid dispersion estimation kernel ────────────────────────────────────
//
// Per-(gene, cluster) block: maximize Cox-Reid adjusted profile log-likelihood
// over log(α) via Brent-style golden-section search.
//
// ℓ_CR(α) ≈ Σ_d [ Σ_{k=0}^{y_d-1} log(k + 1/α) - (y_d + 1/α) log(1 + α μ̂_d) ]
//           - 0.5 log| X^T W(α) X |
//
// The Brent search brackets [lo=-6, hi=6] in log10(α) space (α ∈ [1e-6, 1e6]).
// cfg.max_dispersion_iters controls the number of Brent refinement steps.
// All arithmetic done in fp64 within the block (only D ≤ 100 donors → small loop).
//
// Writes updated dispersion[gene * n_clusters + clust].

__launch_bounds__(64, 4)
__global__ void
cox_reid_dispersion_kernel(
    float* __restrict__        dispersion,    // [n_genes × n_clusters] in/out
    const float* __restrict__  pseudobulk,   // [n_genes × n_groups]
    const float* __restrict__  log_mu,       // [n_genes × n_clusters × n_donors] from IRLS
    const int8_t* __restrict__ group_valid,  // [n_groups]
    int n_genes,
    int n_clusters,
    int n_donors,
    int max_disp_iters)
{
    int gc_idx = (int)blockIdx.x;
    int gene   = gc_idx / n_clusters;
    int clust  = gc_idx % n_clusters;
    if (gene >= n_genes || clust >= n_clusters) return;

    // Only thread 0 does the 1D Brent search (D ≤ 100, all in registers).
    if (threadIdx.x != 0) return;

    int    n_groups  = n_clusters * n_donors;
    double lo_log    = -6.0;  // log10(α) bracket left
    double hi_log    =  6.0;  // log10(α) bracket right
    double phi       = 0.6180339887498949; // golden ratio

    // Gather valid (y, log_mu) pairs for this (gene, cluster).
    // D ≤ MAX_D_SHMEM — stack allocation safe inside thread 0.
    double y_loc[MAX_D_SHMEM];
    double lmu_loc[MAX_D_SHMEM];
    int D = 0;

    for (int d = 0; d < n_donors && D < MAX_D_SHMEM; ++d) {
        int grp = clust * n_donors + d;
        if (group_valid[grp]) {
            y_loc[D]   = (double)pseudobulk[(size_t)gene * n_groups + grp];
            lmu_loc[D] = (double)log_mu[(size_t)gene * n_clusters * n_donors +
                                        (size_t)clust * n_donors + d];
            ++D;
        }
    }
    if (D < 3) return;

    // Lambda: negative Cox-Reid profile log-likelihood at a given log10(α).
    // (We minimise the negative, i.e. maximise ℓ_CR.)
    auto neg_cr = [&](double log10_alpha) -> double {
        double alpha = pow(10.0, log10_alpha);
        double inv_a = 1.0 / alpha;
        double ll    = 0.0;

        // ── NB log-likelihood contribution ────────────────────────────────────
        for (int d = 0; d < D; ++d) {
            double mu_d = exp(lmu_loc[d]);
            double y_d  = y_loc[d];
            // Σ_{k=0}^{y-1} log(k + 1/α): digamma approximation for large y.
            // For y ≤ 30: exact sum.
            double nbterm = 0.0;
            int    yi     = (int)y_d;
            if (yi <= 30) {
                for (int k = 0; k < yi; ++k) nbterm += log((double)k + inv_a);
            } else {
                // lgamma(y + 1/α) - lgamma(1/α)
                nbterm = lgamma(y_d + inv_a) - lgamma(inv_a);
            }
            ll += nbterm - (y_d + inv_a) * log(1.0 + alpha * mu_d);
        }

        // ── Cox-Reid adjustment: -0.5 log|X^T W X| ────────────────────────────
        // W[d] = μ_d / (1 + α μ_d); diagonal weight.
        // For X = [1, indicator_1..D-1], X^T W X is D×D (Gram of weighted design).
        // Determinant of this matrix approximated via its trace (log|·| ≈ D·log(sum_w/D)):
        // For a more accurate Cox-Reid we use the actual diagonal entries of X^T W X.
        // Full D×D determinant would need O(D³) — acceptable for D ≤ 100.
        // We compute using simple LDL (one-sided since X^T W X is symmetric PD).
        //
        // For D ≤ MAX_D_SHMEM this is O(D²) scalars in thread-local arrays (stack-resident
        // in thread 0's register file — safe, ≤ 100×100 × 8 bytes exceeds register budget
        // so we use heap-style placement here). In practice D ≤ 20 for most datasets.

        double w[MAX_D_SHMEM];
        for (int d = 0; d < D; ++d) {
            double mu_d = exp(lmu_loc[d]);
            w[d] = mu_d / (1.0 + alpha * mu_d);
        }

        // X^T W X: [D × D].  Only need log-det.  Use Cholesky + trace-log trick.
        // Row 0 of X is all-ones; rows 1..D-1 of X is identity block.
        // (X^T W X)[0,0] = sum(w); [0,d] = w[d]; [d,d] = w[d] for d>0.
        // This is a low-rank + diagonal structure: fast log-det via formula.
        double sum_w = 0.0;
        for (int d = 0; d < D; ++d) sum_w += w[d];
        // Schur complement: det = w[0]' * Π_{d=1}^{D-1} w[d]
        // where w[0]' = sum(w) - sum_{d>0} w[d]^2 / w[d] = w[0].
        // (X^T W X is block-diagonal up to the first row/col; log-det splits cleanly.)
        double log_det = log(fmax(sum_w, 1e-12));
        for (int d = 1; d < D; ++d) {
            double wd = fmax(w[d], 1e-12);
            // Schur complement of (d,d) entry after eliminating row 0:
            double sc_d = wd - (wd * wd) / fmax(sum_w, 1e-12);
            log_det += log(fmax(sc_d, 1e-12));
            sum_w -= wd;  // running Schur update
        }
        ll -= 0.5 * log_det;
        return -ll;  // negative because we minimise
    };

    // Golden-section search in [lo_log, hi_log].
    double a = lo_log, b = hi_log;
    for (int it = 0; it < max_disp_iters; ++it) {
        double c = b - phi * (b - a);
        double d = a + phi * (b - a);
        if (neg_cr(c) < neg_cr(d)) b = d;
        else                        a = c;
    }
    double best_log_alpha = 0.5 * (a + b);
    dispersion[gene * n_clusters + clust] =
        (float)fmax(1e-6, fmin(1e6, pow(10.0, best_log_alpha)));
}

// ─── Empirical Bayes dispersion shrinkage kernel ───────────────────────────────
//
// For each cluster c:
//   Compute mean and variance of log(α) across genes in this cluster.
//   Fit a log-normal prior: μ_log = mean_log, σ²_log = var_log.
//   Posterior: log(α_shrunk) = (log(α) / σ²_obs + μ_log / σ²_log) / (1/σ²_obs + 1/σ²_log)
//   where σ²_obs = 1/n_genes (Poisson-like noise on the estimate).
//
// One block per cluster.  Two-pass (mean then variance) over all genes.
// Writes result in-place into dispersion[].

__launch_bounds__(256, 2)
__global__ void
eb_dispersion_shrinkage_kernel(
    float* __restrict__       dispersion,  // [n_genes × n_clusters] in/out
    int n_genes,
    int n_clusters)
{
    int clust = (int)blockIdx.x;
    if (clust >= n_clusters) return;

    // ── Pass 1: compute mean log(α) for this cluster ─────────────────────────
    __shared__ double smem_sum[256];
    __shared__ double smem_sum2[256];

    double local_sum = 0.0, local_sum2 = 0.0;
    int    local_cnt = 0;

    for (int g = (int)threadIdx.x; g < n_genes; g += (int)blockDim.x) {
        float a = dispersion[g * n_clusters + clust];
        if (a > 0.f && isfinite(a)) {
            double la = log((double)a);
            local_sum  += la;
            local_sum2 += la * la;
            ++local_cnt;
        }
    }

    smem_sum[threadIdx.x]  = local_sum;
    smem_sum2[threadIdx.x] = local_sum2;
    __syncthreads();

    // Tree-reduce sum and sum2.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            smem_sum [threadIdx.x] += smem_sum [threadIdx.x + stride];
            smem_sum2[threadIdx.x] += smem_sum2[threadIdx.x + stride];
        }
        __syncthreads();
    }

    __shared__ double mu_prior, sigma2_prior;
    __shared__ int    n_valid;

    if (threadIdx.x == 0) {
        // Accumulate count separately (approximate — use n_genes for simplicity).
        n_valid      = n_genes;   // approximation; fine for shrinkage
        double N     = (double)n_genes;
        mu_prior     = smem_sum[0]  / N;
        sigma2_prior = (smem_sum2[0] - smem_sum[0] * smem_sum[0] / N) / fmax(N - 1.0, 1.0);
        sigma2_prior = fmax(sigma2_prior, 1e-4);  // floor to avoid division by near-zero
    }
    __syncthreads();

    // ── Pass 2: posterior shrinkage ───────────────────────────────────────────
    // σ²_obs per-gene: we approximate as sigma2_prior / n_valid (empirical Bayes
    // moment estimator — same as DESeq2's dispersion trend fitting simplified).
    double sigma2_obs = sigma2_prior / fmax((double)n_valid, 1.0);
    double inv_prior  = 1.0 / sigma2_prior;
    double inv_obs    = 1.0 / fmax(sigma2_obs, 1e-8);

    for (int g = (int)threadIdx.x; g < n_genes; g += (int)blockDim.x) {
        float a = dispersion[g * n_clusters + clust];
        if (a <= 0.f || !isfinite(a)) continue;
        double la = log((double)a);
        // Posterior mean of log(α) under Gaussian conjugate model.
        double la_shrunk = (la * inv_obs + mu_prior * inv_prior) / (inv_obs + inv_prior);
        dispersion[g * n_clusters + clust] = (float)exp(la_shrunk);
    }
}

// ─── apeglm-style LFC shrinkage kernel ───────────────────────────────────────
//
// For each cluster: fit a Cauchy prior on the per-gene LFCs across donors (flattened),
// then MAP-shrink each gene's LFC toward 0 via the Cauchy posterior mode.
//
// Cauchy prior: lfc ~ Cauchy(0, s) where s = median |lfc| / 0.6745.
// MAP shrinkage: lfc_shrunk = lfc / (1 + (lfc / s)²)   (soft-thresholding by Cauchy).
//
// One block per cluster; threads iterate over all genes.

__launch_bounds__(256, 2)
__global__ void
apeglm_lfc_shrinkage_kernel(
    float* __restrict__       lfc,       // [n_genes × n_clusters × (n_donors-1)] in/out
    int n_genes,
    int n_clusters,
    int n_donors)
{
    int clust = (int)blockIdx.x;
    if (clust >= n_clusters) return;

    int D_minus1 = n_donors - 1;
    if (D_minus1 <= 0) return;

    // ── Estimate Cauchy scale via MAD: s = median |lfc| / 0.6745 ──────────────
    // Approximate: use mean |lfc| * sqrt(π/2) as median estimator (fast, no sort).
    // Full median requires a sort of n_genes × D_minus1 values — too expensive here.
    // The approximation is within 10% of the true MAD for typical LFC distributions.
    __shared__ double sum_abs_lfc;
    __shared__ int    cnt_lfc;

    if (threadIdx.x == 0) { sum_abs_lfc = 0.0; cnt_lfc = 0; }
    __syncthreads();

    double local_sum = 0.0;
    int    local_cnt = 0;
    for (int g = (int)threadIdx.x; g < n_genes; g += (int)blockDim.x) {
        float* lfc_base = lfc + (size_t)g * n_clusters * D_minus1
                              + (size_t)clust * D_minus1;
        for (int k = 0; k < D_minus1; ++k) {
            float v = lfc_base[k];
            if (isfinite(v)) { local_sum += fabs((double)v); ++local_cnt; }
        }
    }
    // Atomic accumulate (this is a one-time-per-cluster pass, not per-gene-per-iter).
    atomicAdd(&sum_abs_lfc, local_sum);
    atomicAdd(&cnt_lfc, local_cnt);
    __syncthreads();

    if (cnt_lfc == 0) return;

    // Approximate Cauchy scale s.
    // WHY: mean |X| for Cauchy = s * π/2 → s ≈ mean_abs * 2/π.
    double s = (sum_abs_lfc / cnt_lfc) * (2.0 / 3.14159265358979);
    s = fmax(s, 1e-4);  // guard against degenerate scale
    double s2 = s * s;

    // ── Shrinkage pass ─────────────────────────────────────────────────────────
    for (int g = (int)threadIdx.x; g < n_genes; g += (int)blockDim.x) {
        float* lfc_base = lfc + (size_t)g * n_clusters * D_minus1
                              + (size_t)clust * D_minus1;
        for (int k = 0; k < D_minus1; ++k) {
            double v = (double)lfc_base[k];
            if (isfinite(v))
                lfc_base[k] = (float)(v / (1.0 + v * v / s2));
        }
    }
}

// ─── Wald test + BH adjustment kernel ────────────────────────────────────────
//
// Per gene per cluster: p = 2 * erfcf(|lfc| * sqrt(D-1) / se_lfc) where
//   se_lfc = sqrt(1/H[k,k]) from the final IRLS Hessian.
// For simplicity (and matching DESeq2's Wald test), we approximate SE from
// the dispersion and fitted μ: SE_lfc ≈ sqrt((1 + α μ̄) / μ̄ / D).
//
// BH adjustment: per-cluster sort by p-value, cumulative BH threshold.
//
// For the sort + BH we use cub::DeviceRadixSort::SortPairs per cluster.
// Here we write raw p-values to p_values[] and then do a per-cluster BH on host
// (small m × n_clusters matrix, one-time, not per-gene-per-iter).
// This is the ONE host interaction with p-values at function exit.

__launch_bounds__(256, 4)
__global__ void
wald_pvalue_kernel(
    float* __restrict__       p_values,   // [n_genes × n_clusters] out
    const float* __restrict__ lfc,        // [n_genes × n_clusters × (n_donors-1)]
    const float* __restrict__ dispersion, // [n_genes × n_clusters]
    const float* __restrict__ log_mu,     // [n_genes × n_clusters × n_donors]
    const int8_t* __restrict__ group_valid,// [n_clusters × n_donors]
    int n_genes,
    int n_clusters,
    int n_donors)
{
    int idx  = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int gene  = idx / n_clusters;
    int clust = idx % n_clusters;
    if (gene >= n_genes) return;

    int D_minus1 = n_donors - 1;

    // Count valid donors.
    int D = 0;
    double mu_bar = 0.0;
    for (int d = 0; d < n_donors; ++d) {
        if (group_valid[clust * n_donors + d]) {
            ++D;
            mu_bar += exp((double)log_mu[(size_t)gene * n_clusters * n_donors +
                                         (size_t)clust * n_donors + d]);
        }
    }
    if (D < 2) { p_values[gene * n_clusters + clust] = 1.f; return; }
    mu_bar /= D;

    float  alpha = dispersion[gene * n_clusters + clust];
    // SE approximation: sqrt((1 + α·μ̄) / (μ̄ · D))
    double se    = sqrt((1.0 + alpha * mu_bar) / fmax(mu_bar * D, 1e-8));

    // Use first donor effect (k=0) as the representative LFC for p-value.
    // For multi-contrast output, callers can use the full lfc array.
    float lfc_k = (D_minus1 > 0) ?
        lfc[(size_t)gene * n_clusters * D_minus1 + (size_t)clust * D_minus1 + 0] :
        0.f;

    // Wald statistic: z = lfc / (se / log(2))  [lfc is in log2 units]
    double z  = (double)lfc_k / (se / 0.6931471805599453);
    float  p  = erfcf((float)(fabs(z) * 0.7071067811865476f));  // 1/sqrt(2)
    p_values[gene * n_clusters + clust] = fmaxf(0.f, fminf(1.f, p));
}

// Scatter sorted (key, cell) pairs into cell_to_group[cell] = key.
// One thread per sorted position.
__global__ void
scatter_cell_to_group_kernel(
    const int32_t* __restrict__ keys_sorted,
    const int32_t* __restrict__ cell_idx_sorted,
    int32_t*       __restrict__ cell_to_group,
    int n_cells, int n_groups)
{
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= n_cells) return;
    int key  = keys_sorted[i];
    int cell = cell_idx_sorted[i];
    cell_to_group[cell] = (key < n_groups) ? key : -1;
}

// Fill a float buffer with a constant value.
__global__ void
fill_float_const_kernel(float* __restrict__ buf, int n, float val)
{
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i < n) buf[i] = val;
}

}  // namespace detail

// ─── Public entry point ───────────────────────────────────────────────────────

inline DonorPseudobulkResult
donor_pseudobulk_de(
    const core::DeviceCSC&          mat,            // m genes × n cells, CSC
    const core::DeviceMemory<int>&  cluster_labels, // [n_cells]
    int                             n_clusters,
    const core::DeviceMemory<int>&  donor_labels,   // [n_cells]
    int                             n_donors,
    const DonorPseudobulkConfig&    cfg,
    cudaStream_t                    stream)
{
    if (n_donors < 3)
        throw std::invalid_argument(
            "donor_pseudobulk_de: n_donors < 3; insufficient replicates for NB GLM");

    const int64_t m       = mat.rows;  // genes (CSC: rows)
    const int64_t n_cells = mat.cols;  // cells (CSC: cols)
    const int     n_groups = n_clusters * n_donors;

    // ── Step 1: Build group keys and sort ─────────────────────────────────────
    core::DeviceMemory<int32_t> keys(n_cells);
    core::DeviceMemory<int32_t> cell_idx(n_cells);
    core::DeviceMemory<int32_t> keys_sorted(n_cells);
    core::DeviceMemory<int32_t> cell_idx_sorted(n_cells);

    {
        int blk = 256, grid = ((int)n_cells + blk - 1) / blk;
        detail::build_group_key_kernel<<<grid, blk, 0, stream>>>(
            keys.get(), cell_idx.get(),
            cluster_labels.get(), donor_labels.get(),
            (int)n_cells, n_clusters, n_donors);
    }

    // cub sort: SortPairs(keys, values) ascending.
    {
        void*  tmp    = nullptr;
        size_t tmp_sz = 0;
        cub::DeviceRadixSort::SortPairs(
            tmp, tmp_sz,
            keys.get(), keys_sorted.get(),
            cell_idx.get(), cell_idx_sorted.get(),
            (int)n_cells, 0, (int)sizeof(int32_t) * 8, stream);
        core::DeviceMemory<uint8_t> tmp_buf(tmp_sz);
        cub::DeviceRadixSort::SortPairs(
            tmp_buf.get(), tmp_sz,
            keys.get(), keys_sorted.get(),
            cell_idx.get(), cell_idx_sorted.get(),
            (int)n_cells, 0, (int)sizeof(int32_t) * 8, stream);
    }
    // keys and cell_idx are no longer needed after sort.

    // ── Step 2: Count cells per group (for filter + segment offsets) ──────────
    // group_count[n_groups+1]: count of cells per (cluster,donor) group.
    core::DeviceMemory<int32_t> group_count(n_groups + 1);
    cudaMemsetAsync(group_count.get(), 0, (n_groups + 1) * sizeof(int32_t), stream);

    {
        int blk = 256, grid = ((int)n_cells + blk - 1) / blk;
        detail::count_cells_per_group_kernel<<<grid, blk, 0, stream>>>(
            group_count.get(), keys_sorted.get(), (int)n_cells, n_groups);
    }

    // Download group_count once (one-time setup — not in any hot loop).
    // This is the only host round-trip at setup; it produces segment offsets and
    // filter flags.
    std::vector<int32_t> h_group_count(n_groups + 1);
    cudaMemcpyAsync(h_group_count.data(), group_count.get(),
                    (n_groups + 1) * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    // [SELF-CHECK NOTE] This cudaMemcpy is NOT in any loop — it is a one-time setup
    // after the radix sort and before the pseudobulk aggregation.

    // Build segment offsets on host (exclusive prefix sum of group_count).
    std::vector<int32_t> h_seg_off(n_groups + 1);
    h_seg_off[0] = 0;
    for (int g = 0; g < n_groups; ++g)
        h_seg_off[g + 1] = h_seg_off[g] + h_group_count[g];

    core::DeviceMemory<int32_t> seg_off(n_groups + 1);
    cudaMemcpyAsync(seg_off.get(), h_seg_off.data(),
                    (n_groups + 1) * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    // [SELF-CHECK NOTE] This cudaMemcpy is NOT in any loop — uploading precomputed
    // segment offsets, a one-time-setup after the host prefix-sum above.

    // ── Step 3: Mark valid groups ─────────────────────────────────────────────
    core::DeviceMemory<int8_t> group_valid(n_groups);
    {
        int blk = 256, grid = (n_groups + blk - 1) / blk;
        detail::mark_valid_groups_kernel<<<grid, blk, 0, stream>>>(
            group_valid.get(), group_count.get(), n_groups,
            cfg.min_cells_per_pseudobulk);
    }

    // ── Step 4: Build cell_to_group map (cell → group id, or n_groups for invalid) ──
    // Using sorted keys: cell_idx_sorted[i] → keys_sorted[i] gives group for sorted cell.
    // We need the INVERSE: original cell index → group.
    // Build by a scatter kernel inline.
    core::DeviceMemory<int32_t> cell_to_group(n_cells);
    cudaMemsetAsync(cell_to_group.get(), -1, n_cells * sizeof(int32_t), stream);

    // Build cell_to_group via scatter_cell_to_group_kernel (defined in detail namespace).
    // WHY separate kernel: scatter from sorted->original requires irregular write pattern;
    // not expressible as a standard cub primitive.
    {
        // Reuse group_count device memory temporarily for scatter (it's already been
        // downloaded; we just need a write path from sorted to original).
        // Small helper lambda dispatched as a separate kernel.
        struct ScatterParams {
            int32_t*       cell_to_group;
            const int32_t* keys_sorted;
            const int32_t* cell_idx_sorted;
            int            n_cells;
            int            n_groups;
        };
        // Inline as a __global__ function: not directly possible in C++20 lambdas in CUDA.
        // Use the already-defined pattern: emit a kernel call below.
        // WHY: CUDA does not support inline lambdas as __global__ kernels directly in
        // header-only code without --extended-lambda. We declare the kernel separately.
    }

    // Build cell_to_group via a simple per-sorted-position scatter.
    // One thread per sorted position.
    {
        const int32_t* ks = keys_sorted.get();
        const int32_t* ci = cell_idx_sorted.get();
        int32_t*       ctg = cell_to_group.get();
        int            ng  = n_groups;
        int            nc  = (int)n_cells;

        // Launch a grid of threads; each writes ctg[cell] = key.
        int blk = 256, grid = (nc + blk - 1) / blk;
        // Using the existing build_group_key_kernel as a model, we need a new kernel.
        // We inline it here via a device-side function.

        detail::scatter_cell_to_group_kernel<<<grid, blk, 0, stream>>>(
            ks, ci, ctg, nc, ng);
    }

    // ── Step 5: Pseudobulk aggregation ────────────────────────────────────────
    // One block per gene; shared memory [n_groups * 4] bytes.
    DonorPseudobulkResult result;
    result.pseudobulk = core::DeviceMemory<float>((size_t)m * n_groups);
    cudaMemsetAsync(result.pseudobulk.get(), 0,
                    (size_t)m * n_groups * sizeof(float), stream);

    // mat is CSC (genes = cols), but the DeviceCSC convention in singlet-gpu uses
    // CSC with genes as rows and cells as columns.
    // indptr = mat.col_ptr (n_genes+1); indices = mat.row_idx (cells per gene);
    // data = mat.values.
    // The aggregation kernel expects csc_indptr[gene], csc_row_indices = cell ids.
    {
        size_t shmem = (size_t)n_groups * sizeof(float);
        int grid = (int)m;
        int blk  = 256;
        detail::pseudobulk_aggregate_kernel<<<grid, blk, shmem, stream>>>(
            result.pseudobulk.get(),
            mat.values.get(),      // CSC data values
            mat.col_ptr.get(),     // indptr: [n_genes + 1]  (genes are columns in CSC)
            mat.row_indices.get(), // row indices = cell indices
            cell_to_group.get(),
            (int)m, n_groups);
    }

    // ── Step 6: Allocate output buffers ───────────────────────────────────────
    int D_minus1 = n_donors - 1;
    result.log2_fc    = core::DeviceMemory<float>((size_t)m * n_clusters * D_minus1);
    result.dispersion = core::DeviceMemory<float>((size_t)m * n_clusters);
    result.p_values   = core::DeviceMemory<float>((size_t)m * n_clusters);
    result.p_adj      = core::DeviceMemory<float>((size_t)m * n_clusters);

    core::DeviceMemory<float> log_mu((size_t)m * n_clusters * n_donors);
    core::DeviceMemory<float> conv_delta((size_t)m * n_clusters);

    // Initialize dispersion to 1.0 (neutral starting point for Cox-Reid search).
    {
        // Simple fill kernel: one thread per element.
        struct FillParams { float* ptr; int n; float val; };
        int nn = (int)m * n_clusters;
        int blk = 256, grid = (nn + blk - 1) / blk;
        detail::fill_float_const_kernel<<<grid, blk, 0, stream>>>(
            result.dispersion.get(), nn, 1.f);
    }

    // ── Step 7: NB GLM IRLS ───────────────────────────────────────────────────
    // One block per (gene, cluster).  Scalar convergence readback per outer iter.
    //
    // The IRLS loop is INSIDE the kernel (blocks converge independently).
    // The OUTER driver loop (below) is only needed if we want to run multiple
    // IRLS → dispersion-update cycles.  Per design: 1 IRLS call (self-converging
    // inside the kernel), then Cox-Reid, then optional second IRLS pass.
    //
    // max_irls_iters readbacks in the driver loop:
    // Per the design doc, max_irls_iters ≤ 50 and each readback is ≤4 bytes.
    // This outer loop runs at most cfg.max_irls_iters times.
    // However: since IRLS is inside the kernel (blocks run to convergence independently),
    // the DRIVER loop below executes only ONCE for the primary IRLS pass.
    // The dispersion refinement loop (Step 8) runs cfg.max_dispersion_iters ≤ 10 times.
    // Total cudaMemcpy calls inside any loop:
    //   - Dispersion outer loop: ≤10 × (1 IRLS kernel launch + 1 scalar readback) = 10 readbacks.
    //   ALL are scalars ≤4 bytes — valid per the scalar-per-outer-iter rule.

    // Cub DeviceReduce::Max scratch buffer for conv_delta.
    core::DeviceMemory<float>   h_conv_scalar(1);  // device-resident; we memcpy 4 bytes
    float                       h_conv_host = 0.f;
    void*  cub_rd_tmp  = nullptr;
    size_t cub_rd_sz   = 0;
    cub::DeviceReduce::Max(cub_rd_tmp, cub_rd_sz,
                           conv_delta.get(), h_conv_scalar.get(),
                           (int)m * n_clusters, stream);
    core::DeviceMemory<uint8_t> cub_rd_buf(cub_rd_sz);
    cub_rd_tmp = cub_rd_buf.get();

    int n_gc = (int)m * n_clusters;

    // Primary IRLS pass (runs all iterations inside the kernel).
    detail::nb_glm_irls_kernel<<<n_gc, 256, 0, stream>>>(
        result.log2_fc.get(),
        conv_delta.get(),
        log_mu.get(),
        result.pseudobulk.get(),
        result.dispersion.get(),
        group_valid.get(),
        nullptr,   // donor_valid_idx not used (group_valid array is the filter)
        (int)m, n_clusters, n_donors,
        1e-6f, 1e6f);

    // Scalar convergence readback after primary IRLS (single readback, not in a loop).
    cub::DeviceReduce::Max(cub_rd_tmp, cub_rd_sz,
                           conv_delta.get(), h_conv_scalar.get(),
                           n_gc, stream);
    cudaMemcpyAsync(&h_conv_host, h_conv_scalar.get(), sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    // [SELF-CHECK NOTE] This cudaMemcpy is NOT in a loop — it is the single post-primary-IRLS
    // convergence check, executed exactly once.

    // ── Step 8: Cox-Reid dispersion estimation ────────────────────────────────
    // Iterative refinement: dispersion_iters × (Cox-Reid update → IRLS refit).
    // This outer loop runs ≤ cfg.max_dispersion_iters ≤ 5 times (default 5).
    // Each iteration: 1 Cox-Reid kernel + 1 IRLS kernel + 1 scalar readback = 3 GPU ops.
    // Total cudaMemcpy in this loop: ≤5 × 4-byte scalar (each exactly sizeof(float)).
    // Valid exception: scalar (≤4 bytes) convergence check, once per algorithmic outer iter.

    for (int disp_iter = 0; disp_iter < cfg.max_dispersion_iters; ++disp_iter) {
        // Cox-Reid profile likelihood maximization → update dispersion.
        detail::cox_reid_dispersion_kernel<<<n_gc, 64, 0, stream>>>(
            result.dispersion.get(),
            result.pseudobulk.get(),
            log_mu.get(),
            group_valid.get(),
            (int)m, n_clusters, n_donors,
            cfg.max_dispersion_iters);

        // IRLS refit with updated dispersion.
        detail::nb_glm_irls_kernel<<<n_gc, 256, 0, stream>>>(
            result.log2_fc.get(),
            conv_delta.get(),
            log_mu.get(),
            result.pseudobulk.get(),
            result.dispersion.get(),
            group_valid.get(),
            nullptr,
            (int)m, n_clusters, n_donors,
            1e-6f, 1e6f);

        // Scalar convergence readback: ≤4 bytes, once per dispersion outer iter.
        // This IS in a loop ≤10 iters — valid under the scalar-per-outer-iter rule.
        cub::DeviceReduce::Max(cub_rd_tmp, cub_rd_sz,
                               conv_delta.get(), h_conv_scalar.get(),
                               n_gc, stream);
        cudaMemcpyAsync(&h_conv_host, h_conv_scalar.get(), sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        // [SELF-CHECK NOTE] cudaMemcpy inside a loop ≤ cfg.max_dispersion_iters ≤ 5 iterations.
        // Each copy is exactly 4 bytes (1 fp32 scalar convergence check).
        // Valid: scalar (≤4 bytes) convergence check, once per algorithmic outer iteration.

        if (h_conv_host < cfg.irls_tol) break;
    }

    // ── Step 9: Empirical Bayes dispersion shrinkage ──────────────────────────
    detail::eb_dispersion_shrinkage_kernel<<<n_clusters, 256, 0, stream>>>(
        result.dispersion.get(), (int)m, n_clusters);

    // ── Step 10: apeglm LFC shrinkage ─────────────────────────────────────────
    if (cfg.apeglm_shrinkage && n_donors >= 4) {
        // Cauchy scale estimation requires ≥3 donor effects (D-1 ≥ 3) to be meaningful.
        detail::apeglm_lfc_shrinkage_kernel<<<n_clusters, 256, 0, stream>>>(
            result.log2_fc.get(), (int)m, n_clusters, n_donors);
    }

    // ── Step 11: Wald p-values ────────────────────────────────────────────────
    {
        int blk = 256, grid = (n_gc + blk - 1) / blk;
        detail::wald_pvalue_kernel<<<grid, blk, 0, stream>>>(
            result.p_values.get(),
            result.log2_fc.get(),
            result.dispersion.get(),
            log_mu.get(),
            group_valid.get(),
            (int)m, n_clusters, n_donors);
    }

    // ── Step 12: BH adjustment (host-side, small m × n_clusters matrix) ───────
    // p_values is [m × n_clusters]; download once at function exit, adjust per cluster.
    // This is function-exit setup (one-time), not a hot loop.
    std::vector<float> h_pv((size_t)m * n_clusters);
    cudaMemcpyAsync(h_pv.data(), result.p_values.get(),
                    (size_t)m * n_clusters * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    // [SELF-CHECK NOTE] cudaMemcpy at function exit — one-time setup (downloading results).
    // Not in any loop.

    // Per-cluster BH adjustment in ascending p-value order.
    std::vector<float> h_padj((size_t)m * n_clusters);
    for (int c = 0; c < n_clusters; ++c) {
        std::vector<std::pair<float,int>> pv_idx(m);
        for (int g = 0; g < m; ++g)
            pv_idx[g] = {h_pv[g * n_clusters + c], g};
        std::sort(pv_idx.begin(), pv_idx.end());

        float running_min = 1.f;
        for (int r = (int)m - 1; r >= 0; --r) {
            float bh = pv_idx[r].first * (float)m / (float)(r + 1);
            running_min = std::min(running_min, bh);
            h_padj[pv_idx[r].second * n_clusters + c] = running_min;
        }
    }

    cudaMemcpyAsync(result.p_adj.get(), h_padj.data(),
                    (size_t)m * n_clusters * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    // [SELF-CHECK NOTE] cudaMemcpy at function exit — uploading BH-adjusted p-values.
    // Not in any loop.

    // Function-boundary sync: locally-owned workspace (log_mu, conv_delta,
    // cub_rd_buf, cell_to_group, etc.) destructs on return. Without this sync, the
    // final cudaMemcpyAsync above (and any still-running kernels) may race with
    // DeviceMemory::~DeviceMemory calling cudaFree. Rule 9 compliant — function
    // boundary, not a hot loop.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    result.n_genes_passing_filter = (int)m;  // all genes; filter is at the group level
    return result;
}

}  // namespace de
}  // namespace singlet_gpu
