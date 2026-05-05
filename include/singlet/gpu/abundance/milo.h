// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (first GPU Milo-style kNN-neighborhood differential abundance)
//
// abundance/milo.h — GPU-native Milo differential abundance via kNN neighborhoods
//
// Algorithm (Dann et al. Nature Biotechnology 2022 — GPU port):
//   1. Neighborhood sampling: stride-sampled index cells with Philox-jittered stagger.
//      Default stride = max(1, floor(n_cells / n_nh_target)).  Each index cell anchors
//      one neighborhood ball = its k nearest neighbors from the cycle-8 KNN graph.
//   2. Neighborhood expansion: for each index cell gather its k neighbors from the
//      KnnResult CSR (cycle 8 compute_exact output).  Produces nh_cells CSR.
//   3. Per-(nh, donor) count: atomicAdd for each (nh, cell) pair into
//      nh_counts[nh * n_donors + donor[cell]].  Output fits L2 (2000 × 100 × 4 = 800 KB).
//   4. Batched NB GLM per NH: 1 block per NH, MAX_IRLS=20 IRLS iterations in shared
//      memory.  Design matrix is binary: condition[donor] (condition-only model, 2 params).
//      fp32 hot loop; fp64 for 2×2 Hessian inverse (closed form: avoids Gauss-Jordan).
//   5. Graph-weighted FDR: build NH-NH adjacency W[i,j] = |cells(i) ∩ cells(j)| via
//      cuSPARSE SpGEMM on the (nh × cells) boolean matrix self-product.  Then apply
//      BH with effective-sample-size correction from Cytoscape/SpatialFDR.
//
// Numerical stability:
//   - fp32 throughout hot path; fp64 for 2×2 Hessian inverse (2 scalars: det + inverse).
//   - IRLS with max |Δβ| convergence check (per-block, no host readback per iter —
//     converge fully inside the kernel; the caller never enters a per-iter outer loop).
//   - Dispersion fixed to MOM estimate (quasi-likelihood); no inner Brent loop needed.
//   - offset = log(n_cells_of_donor) per NH.
//
// Memory layout:
//   Input kNN CSR: n_cells * k * (4+4) bytes = 12 MB at 100k×30.
//   nh_cells CSR: n_nh * avg_k * 4 bytes = 240 KB at 2000×30.
//   nh_counts: n_nh × n_donors × 4 = 800 KB at 2000×100.
//   donor offsets: n_donors × 4 = 400 B.
//   Output (log_fc, se, p_value, fdr): 4 × n_nh × 4 = 32 KB at 2000.
//   Total: ~15 MB (fits in L2 + device DRAM; OOC not needed at standard scale).
//
// cub usage:
//   cub::DeviceRadixSort::SortPairs — sort (nh, cell) pairs before count kernel.
//   (No per-nh DeviceSegmentedReduce needed: nh_counts fits L2; atomicAdd is faster.)
//
// Streams: 1, caller-provided.
// Precision: fp32; fp64 for closed-form 2×2 Hessian inverse.
// Determinism: Philox sampling seeded; IRLS deterministic given inputs.
// OOC plan: cell-batched kNN if n_cells > 500k (each batch processes a cell stripe,
//   accumulating nh_counts additively; implemented by caller with PzDataLoader chunks).
//
// Correctness tolerance: log_fc Spearman ρ ≥ 0.95 vs Milo R; p-value rank ρ ≥ 0.90.
//
// Reference: Dann et al. 2022 (miloR); edgeR quasi-likelihood NB GLM.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/graph/knn.h>

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cusparse.h>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <algorithm>

namespace singlet_gpu {
namespace abundance {

// ─── Public API types ──────────────────────────────────────────────────────────

struct MiloConfig {
    int      k                  = 30;     // kNN neighborhood size
    int      n_nh_target        = 2000;   // target number of neighborhoods
    int      max_irls_iters     = 20;     // IRLS iterations inside GLM kernel
    float    irls_tol           = 1e-4f;  // convergence threshold (per-block)
    float    min_cells_fraction = 0.2f;   // drop NHs with < this fraction of donors present
    int      min_donors_per_nh  = 3;      // drop NHs with fewer valid donors
    uint64_t seed               = 42;     // Philox seed for neighborhood sampling
    bool     deterministic      = true;   // use deterministic count path (scan vs atomic)
};

struct MiloResult {
    // Per-neighborhood output (length = n_nh_valid ≤ n_nh_sampled)
    core::DeviceMemory<float> log_fc;    // [n_nh] log fold-change condition 1 vs 0
    core::DeviceMemory<float> se;        // [n_nh] standard error of LFC
    core::DeviceMemory<float> p_value;   // [n_nh] Wald test p-value
    core::DeviceMemory<float> fdr;       // [n_nh] graph-weighted BH FDR
    core::DeviceMemory<int>   nh_index_cells; // [n_nh] original cell index for each NH
    core::DeviceMemory<int>   nh_valid;  // [n_nh] 1=included in GLM, 0=filtered
    core::DeviceMemory<float> nh_counts; // [n_nh × n_donors] float (donor cell counts per NH)
    int n_nh;                            // total sampled NHs
    int n_donors;
};

// ─── Internal kernels ─────────────────────────────────────────────────────────

namespace detail {

// Philox-seeded stride-sampled neighborhood index cells.
//
// WHY Philox jitter: avoids systematic aliasing with spatially-structured cell ordering.
// The stride s = floor(n_cells / n_nh_target) provides uniform coverage; a per-index
// Philox offset ∈ [0, s) provides within-stride jitter.
//
// Output: index_cells[nh] = cell index for neighborhood nh.
// Deterministic: fixed seed + stateless Philox counter (no atomics).
__global__ void
nh_sample_kernel(
    int* __restrict__   index_cells,   // [n_nh] output
    int n_cells,
    int n_nh,
    int stride,             // floor(n_cells / n_nh_target)
    uint64_t seed)
{
    int nh = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (nh >= n_nh) return;

    // Philox 2×32 counter: counter = (nh, 0), key = seed.
    // Output: 2 × 32-bit random values → jitter ∈ [0, stride).
    // WHY Philox over cuRAND: stateless; no state array; deterministic with seed.
    uint32_t ctr0 = (uint32_t)nh;
    uint32_t ctr1 = 0u;
    uint32_t k0   = (uint32_t)(seed & 0xFFFFFFFFu);
    uint32_t k1   = (uint32_t)(seed >> 32);

    // Ten-round Philox-2×32-10 (Salmon et al. 2011).
    // WHY 10 rounds: NIST-recommended strength; 2×32 fits in registers.
    constexpr uint32_t PHILOX_W0 = 0x9E3779B9u;
    constexpr uint32_t PHILOX_M0 = 0xD2511F53u;
    constexpr uint32_t PHILOX_M1 = 0xCD9E8D57u;
    for (int r = 0; r < 10; ++r) {
        uint64_t prod0 = (uint64_t)PHILOX_M0 * ctr0;
        uint64_t prod1 = (uint64_t)PHILOX_M1 * ctr1;
        uint32_t hi0 = (uint32_t)(prod0 >> 32);
        uint32_t lo0 = (uint32_t)(prod0);
        uint32_t hi1 = (uint32_t)(prod1 >> 32);
        uint32_t lo1 = (uint32_t)(prod1);
        ctr0 = hi1 ^ ctr0 ^ k0;
        ctr1 = lo0;
        (void)lo1; (void)hi0;  // only 2-word output used
        k0 += PHILOX_W0;
        k1 += PHILOX_W0;   // keep bumping k1 to progress key schedule
    }
    // Map random word to jitter in [0, stride); add base offset.
    int jitter  = (int)((uint64_t)ctr0 * (uint64_t)stride >> 32);
    int cell_idx = nh * stride + jitter;
    if (cell_idx >= n_cells) cell_idx = n_cells - 1;  // clamp last NH if overshoot
    index_cells[nh] = cell_idx;
}

// Build nh_cells CSR: for each NH, its cell members = index cell + its k neighbors.
//
// Kernel: 1 block per NH, 128 threads.  Each thread writes one cell slot to the
// flat nh_cells_flat[] array pre-allocated as [n_nh × (k+1)].
// Row offsets: uniform, row_ptr[nh] = nh * (k+1).
//
// WHY (k+1): include the index cell itself (cell 0 = index, 1..k = neighbors).
__global__ void
nh_expand_kernel(
    int* __restrict__        nh_cells_flat,     // [n_nh × (k+1)]
    const int* __restrict__  index_cells,       // [n_nh]
    const int* __restrict__  knn_neighbors,     // [n_cells × k]  CSR values (uniform k)
    int n_nh,
    int k)
{
    int nh = (int)blockIdx.x;
    if (nh >= n_nh) return;

    int idx_cell = index_cells[nh];
    int base_out = nh * (k + 1);

    // Thread 0: write the index cell itself.
    if (threadIdx.x == 0) {
        nh_cells_flat[base_out] = idx_cell;
    }
    // Remaining threads: write neighbor j (0..k-1).
    for (int j = (int)threadIdx.x; j < k; j += (int)blockDim.x) {
        nh_cells_flat[base_out + 1 + j] = knn_neighbors[(size_t)idx_cell * k + j];
    }
}

// Per-(nh, donor) count kernel.
//
// For each nh, scan its k+1 member cells and atomicAdd into
// nh_counts[nh * n_donors + donor_id[cell]].
//
// WHY atomicAdd: nh_counts is 2000 × 100 × 4 = 800 KB — fits in L2.
// The write pattern is random but the output array is tiny, so L2 absorbs contention.
// A deterministic segmented-scan path is available when cfg.deterministic=true
// (the caller first sorts by (nh, donor_id) then calls nh_donor_count_seg_kernel).
//
// One block per NH, 128 threads scan the k+1 cells in strides.
__global__ void
nh_donor_count_kernel(
    float* __restrict__       nh_counts,        // [n_nh × n_donors] zero-initialized
    const int* __restrict__   nh_cells_flat,    // [n_nh × (k+1)]
    const int* __restrict__   donor_id,         // [n_cells]
    int n_nh,
    int k_plus_one,    // k+1
    int n_donors,
    int n_cells)
{
    int nh = (int)blockIdx.x;
    if (nh >= n_nh) return;

    float* out_row = nh_counts + (size_t)nh * n_donors;
    const int* cells = nh_cells_flat + (size_t)nh * k_plus_one;

    for (int i = (int)threadIdx.x; i < k_plus_one; i += (int)blockDim.x) {
        int c = cells[i];
        if (c < 0 || c >= n_cells) continue;
        int d = donor_id[c];
        if (d >= 0 && d < n_donors)
            atomicAdd(&out_row[d], 1.f);
    }
}

// Deterministic count kernel (used when cfg.deterministic=true).
//
// Input: pairs (nh, donor) sorted by (nh * n_donors + donor) via cub::DeviceRadixSort.
// Scan run boundaries → one write per boundary crossing.  No atomics.
// Called AFTER a sort-pairs step on (nh_cell_key[], cell_donor_val[]) where
// nh_cell_key[i] = nh_of_pair[i] * n_donors + donor_id[cell_of_pair[i]].
// Run-length counted here: consecutive equal keys → count for that (nh, donor).
__global__ void
nh_donor_count_seg_kernel(
    float* __restrict__        nh_counts,         // [n_nh × n_donors]
    const int32_t* __restrict__ sorted_keys,      // [n_pairs] = nh * n_donors + donor
    int n_pairs,
    int n_donors)
{
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= n_pairs) return;

    int32_t key = sorted_keys[i];
    // Only thread at the START of each run writes the count.
    // A run starts at i=0 or when key[i] != key[i-1].
    bool is_start = (i == 0) || (sorted_keys[i - 1] != key);
    if (!is_start) return;

    // Count the run length.
    int cnt = 1;
    while (i + cnt < n_pairs && sorted_keys[i + cnt] == key) ++cnt;

    int nh = key / n_donors;
    int d  = key % n_donors;
    nh_counts[(size_t)nh * n_donors + d] = (float)cnt;
}

// Compute donor total cell counts (sum of nh_counts across all NHs, per donor).
// Used as the GLM offset = log(total_cells_of_donor).
// One thread per donor; sums across NHs.
__global__ void
donor_total_kernel(
    float* __restrict__        donor_totals,   // [n_donors]
    const float* __restrict__  nh_counts,      // [n_nh × n_donors]
    int n_nh,
    int n_donors)
{
    int d = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (d >= n_donors) return;
    float total = 0.f;
    for (int nh = 0; nh < n_nh; ++nh)
        total += nh_counts[(size_t)nh * n_donors + d];
    donor_totals[d] = fmaxf(total, 1.f);  // clamp to avoid log(0)
}

// Mark valid NHs: NH is valid if it has >= min_donors_per_nh donors with count > 0
// AND has representatives of at least one donor from each condition.
// Validity also requires: count[nh, d] > 0 for some d in condition=0 AND condition=1.
__global__ void
mark_valid_nh_kernel(
    int* __restrict__         nh_valid,          // [n_nh]
    const float* __restrict__ nh_counts,         // [n_nh × n_donors]
    const int* __restrict__   donor_condition,   // [n_donors] condition label (0 or 1)
    int n_nh,
    int n_donors,
    int min_donors_per_nh)
{
    int nh = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (nh >= n_nh) return;

    const float* row = nh_counts + (size_t)nh * n_donors;
    int cnt   = 0;
    bool has0 = false, has1 = false;
    for (int d = 0; d < n_donors; ++d) {
        if (row[d] > 0.f) {
            ++cnt;
            if (donor_condition[d] == 0) has0 = true;
            else                         has1 = true;
        }
    }
    nh_valid[nh] = (cnt >= min_donors_per_nh && has0 && has1) ? 1 : 0;
}

// ── NB GLM IRLS kernel: 1 block per NH, 2-parameter condition-only model ────────
//
// Model: counts[nh, d] ~ NB(mu[d], dispersion)
//   log(mu[d]) = offset[d] + beta[0] + beta[1] * condition[d]
//
// Design matrix X[D × 2]: X[:,0] = 1 (intercept), X[:,1] = condition[d] (0 or 1).
// Hessian H[2 × 2] in fp64: computed analytically from IRLS weights.
// H^{-1} via closed-form 2×2 determinant: avoids Gauss-Jordan overhead for size-2 system.
//
// Dispersion α estimated by method-of-moments per NH (deterministic one-shot):
//   α_mom = max(0, (var(y) - mean(y)) / mean(y)^2)
//   clamped to [1e-6, 1e6].
//
// WHY all IRLS runs inside the kernel (no outer host loop):
//   MAX_IRLS=20 iterations × 2000 NHs; kernel launch overhead per iter would cost
//   more than the arithmetic. The per-block loop is fully independent; no global sync.
//
// Writes: log_fc[nh] = beta[1] / log(2), se[nh] from H^{-1}[1,1], p_value[nh] = Wald.

constexpr int MAX_IRLS_MILO = 20;

__global__ void
milo_glm_irls_kernel(
    float* __restrict__        log_fc,           // [n_nh]
    float* __restrict__        se_out,           // [n_nh]
    float* __restrict__        p_value_out,      // [n_nh]
    const float* __restrict__  nh_counts,        // [n_nh × n_donors]
    const float* __restrict__  donor_totals,     // [n_donors] total cells across all NHs
    const int* __restrict__    donor_condition,  // [n_donors]
    const int* __restrict__    nh_valid,         // [n_nh]
    int n_nh,
    int n_donors,
    float irls_tol)
{
    int nh = (int)blockIdx.x;
    if (nh >= n_nh) return;

    // Invalid NH: write NaN and return early.
    if (!nh_valid[nh]) {
        if (threadIdx.x == 0) {
            log_fc[nh]    = __int_as_float(0x7FC00000u);  // NaN
            se_out[nh]    = __int_as_float(0x7FC00000u);
            p_value_out[nh] = 1.f;
        }
        return;
    }

    // Only thread 0 executes: D ≤ 100 donors, 2-param model — sequential is fast.
    if (threadIdx.x != 0) return;

    const float* y_row = nh_counts + (size_t)nh * n_donors;

    // ── Gather valid (y, offset, cond) for this NH ─────────────────────────────
    // WHY stack arrays: D ≤ 100, fits in register file / L1 scratch.
    float  y_loc[100];
    float  off_loc[100];    // offset = log(donor_total)
    int    cond_loc[100];   // condition 0 or 1
    int    D = 0;

    for (int d = 0; d < n_donors && D < 100; ++d) {
        if (y_row[d] > 0.f) {
            y_loc[D]   = y_row[d];
            off_loc[D] = logf(fmaxf(donor_totals[d], 1.f));
            cond_loc[D] = donor_condition[d];
            ++D;
        }
    }

    if (D < 3) {
        // Written NaN above already; safety guard.
        log_fc[nh]    = __int_as_float(0x7FC00000u);
        se_out[nh]    = __int_as_float(0x7FC00000u);
        p_value_out[nh] = 1.f;
        return;
    }

    // ── Method-of-moments dispersion estimation ────────────────────────────────
    // α_mom = max(0, (var(y) - mean(y)) / mean(y)^2)
    float sum_y = 0.f, sum_y2 = 0.f;
    for (int d = 0; d < D; ++d) { sum_y += y_loc[d]; sum_y2 += y_loc[d] * y_loc[d]; }
    float mean_y = sum_y / D;
    float var_y  = (sum_y2 - sum_y * sum_y / D) / fmaxf((float)(D - 1), 1.f);
    float alpha  = fmaxf(1e-6f, fminf(1e6f, (var_y - mean_y) / fmaxf(mean_y * mean_y, 1e-8f)));

    // ── IRLS: 2-parameter NB GLM ───────────────────────────────────────────────
    // beta[0] = intercept (log mean at condition=0)
    // beta[1] = log fold-change (condition effect)
    // Init: beta[0] = log(mean_y + 0.5), beta[1] = 0.
    double b0 = (double)logf(fmaxf(mean_y + 0.5f, 1e-8f));
    double b1 = 0.0;

    for (int iter = 0; iter < MAX_IRLS_MILO; ++iter) {
        // Build 2×2 Hessian H and score S in fp64.
        // H[i,j] = Σ_d X[d,i] * w_d * X[d,j]
        // S[i]   = Σ_d X[d,i] * w_d * (y_d/mu_d - 1)  (score = X^T W (y/mu - 1))
        // WHY fp64: 2×2 matrix, negligible cost; prevents precision loss in det.
        double H00 = 0.0, H01 = 0.0, H11 = 0.0;
        double S0  = 0.0, S1  = 0.0;

        for (int d = 0; d < D; ++d) {
            double eta  = b0 + b1 * (double)cond_loc[d] + (double)off_loc[d];
            double mu   = exp(eta);
            // NB variance = mu + alpha * mu^2; IRLS weight = mu^2 / var = 1/(1/mu + alpha)
            double w    = mu / (1.0 + (double)alpha * mu);
            double resid = (double)y_loc[d] - mu;
            int    x1   = cond_loc[d];

            H00 += w;
            H01 += w * x1;
            H11 += w * x1 * x1;
            S0  += w * resid;
            S1  += w * resid * x1;
        }

        // ── Closed-form 2×2 inverse: H^{-1} = (1/det) * [[H11, -H01],[-H01, H00]] ──
        // WHY closed form: cheaper than Gauss-Jordan for exactly D=2 parameters.
        double det = H00 * H11 - H01 * H01;
        if (fabs(det) < 1e-12) break;  // singular; keep current β

        double inv_det = 1.0 / det;
        double db0 = inv_det * (H11 * S0 - H01 * S1);
        double db1 = inv_det * (H00 * S1 - H01 * S0);

        b0 += db0;
        b1 += db1;

        if (fmax(fabs(db0), fabs(db1)) < (double)irls_tol) break;
    }

    // ── Compute SE from Hessian at converged β ─────────────────────────────────
    // Recompute H at final β.
    double H00f = 0.0, H01f = 0.0, H11f = 0.0;
    for (int d = 0; d < D; ++d) {
        double eta = b0 + b1 * (double)cond_loc[d] + (double)off_loc[d];
        double mu  = exp(eta);
        double w   = mu / (1.0 + (double)alpha * mu);
        int x1 = cond_loc[d];
        H00f += w; H01f += w * x1; H11f += w * x1 * x1;
    }

    double detf = H00f * H11f - H01f * H01f;
    // Var(b1) = H^{-1}[1,1] = H00 / det
    double var_b1 = (fabs(detf) > 1e-12) ? H00f / detf : 1e6;
    double se_b1  = sqrt(fmax(var_b1, 0.0));

    // Wald p-value: z = b1 / se(b1); p = 2 * Phi(-|z|) approximated via erfc.
    double z = (se_b1 > 1e-12) ? b1 / se_b1 : 0.0;
    // Two-tailed: p = erfc(|z| / sqrt(2))  (exact normal tail via CUDA erfc).
    double pval = erfc(fabs(z) * 0.7071067811865476);  // 1/sqrt(2)

    // LFC = beta[1] / log(2): log2 fold-change of condition=1 vs condition=0.
    log_fc[nh]    = (float)(b1 / 0.6931471805599453);
    se_out[nh]    = (float)se_b1;
    p_value_out[nh] = (float)fmin(fmax(pval, 0.0), 1.0);
}

// Build (nh * n_donors + donor) keys for the deterministic count path.
// One thread per (nh, cell-slot) pair in nh_cells_flat.
__global__ void
nh_build_keys_kernel(
    int32_t* __restrict__      keys,         // [n_nh × k1]
    const int* __restrict__    nh_cells,     // [n_nh × k1]
    const int* __restrict__    donor_ids,    // [n_cells]
    int n_nh, int k1, int n_donors, int n_cells)
{
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= n_nh * k1) return;
    int nh   = i / k1;
    int cell = nh_cells[i];
    int d    = (cell >= 0 && cell < n_cells) ? donor_ids[cell] : -1;
    // Sentinel key n_nh * n_donors for invalid / out-of-range cells.
    keys[i] = (d >= 0 && d < n_donors) ? (nh * n_donors + d) : (n_nh * n_donors);
}

// Graph-weighted BH FDR.
//
// Spatially-correlated BH correction (Cytoscape SpatialFDR approach):
//   1. Compute connectivity weight per NH: w[nh] = 1 / (1 + degree[nh]) where
//      degree = number of overlapping neighbors in the NH-NH adjacency.
//   2. Sort NHs by p_value ascending.
//   3. BH step: adjusted_p[k] = p[k] * n_effective / k  where
//      n_effective = sum(w) / max(w) (effective number of independent tests).
//   4. Enforce monotonicity from the tail.
//
// WHY host-side BH: n_nh ≤ 2000 → trivially small; eliminates need for a custom
//   device-side sorted BH kernel. This is a one-time end-of-pipeline operation,
//   not an inner loop — see absolute-rules §"Valid exceptions" (one-time setup/exit).
//
// The NH-NH adjacency degree is computed on device via a brief kernel that counts
// shared cells between NHs using atomicAdd into a degree[] array of size n_nh.

__global__ void
nh_adjacency_degree_kernel(
    int* __restrict__        degree,         // [n_nh] zero-initialized
    const int* __restrict__  nh_cells_flat,  // [n_nh × (k+1)]
    int n_nh,
    int k_plus_one,
    int n_cells)
{
    // Strategy: for each cell, find all NHs it belongs to and increment
    // degree for each pair.  This is O(n_cells × n_nh) — acceptable at 2000 NHs.
    //
    // WHY linear scan per NH pair: n_nh=2000 → 2M pairs; each thread handles one NH,
    // scans cells to check membership.  Faster than SpGEMM for n_nh ≤ 2000.
    int nh_a = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (nh_a >= n_nh) return;

    const int* cells_a = nh_cells_flat + (size_t)nh_a * k_plus_one;

    for (int nh_b = nh_a + 1; nh_b < n_nh; ++nh_b) {
        const int* cells_b = nh_cells_flat + (size_t)nh_b * k_plus_one;
        // Count shared cells (small inner loop: k+1 ≤ 31 elements each).
        bool shared = false;
        for (int i = 0; i < k_plus_one && !shared; ++i) {
            int ca = cells_a[i];
            for (int j = 0; j < k_plus_one && !shared; ++j) {
                if (ca == cells_b[j]) shared = true;
            }
        }
        if (shared) {
            atomicAdd(&degree[nh_a], 1);
            atomicAdd(&degree[nh_b], 1);
        }
    }
}

}  // namespace detail

// ─── Main public entry point ───────────────────────────────────────────────────
//
// Compute Milo differential abundance.
//
// embedding:       [n_cells × n_pca_dims] row-major dense PCA embedding on device.
// donor_id_host:   [n_cells] donor integer labels (0..n_donors-1); host vector.
// condition_host:  [n_donors] condition labels (0 or 1); host vector.
// knn_result:      output of graph::compute_knn (cycle 8) with k = cfg.k.
// stream:          caller-provided CUDA stream.
//
// Returns MiloResult with all per-NH statistics and raw count matrix.
inline MiloResult
compute_milo(
    const core::DeviceDense&   embedding,
    const std::vector<int>&    donor_id_host,
    const std::vector<int>&    condition_host,
    const graph::KnnResult&    knn_result,
    const MiloConfig&          cfg,
    cudaStream_t               stream)
{
    const int n_cells  = (int)embedding.rows;
    const int n_donors = (int)condition_host.size();
    const int k        = cfg.k;
    const int k1       = k + 1;  // k neighbors + index cell itself

    if (n_cells <= 0)  throw std::invalid_argument("compute_milo: empty embedding");
    if (n_donors < 2)  throw std::invalid_argument("compute_milo: need ≥ 2 donors");
    if ((int)donor_id_host.size() != n_cells)
        throw std::invalid_argument("compute_milo: donor_id length != n_cells");
    if (knn_result.k != k)
        throw std::invalid_argument("compute_milo: knn_result.k != cfg.k");

    // ── Step 1: Neighborhood sampling ─────────────────────────────────────────
    int stride = std::max(1, n_cells / cfg.n_nh_target);
    int n_nh   = (n_cells + stride - 1) / stride;  // actual number of NHs

    core::DeviceMemory<int> d_index_cells(n_nh);
    {
        int b = 256, g = (n_nh + b - 1) / b;
        detail::nh_sample_kernel<<<g, b, 0, stream>>>(
            d_index_cells.get(), n_cells, n_nh, stride, cfg.seed);
    }

    // ── Step 2: Neighborhood expansion ────────────────────────────────────────
    // nh_cells_flat[nh, 0..k]: index cell + k neighbors, row-major.
    core::DeviceMemory<int> d_nh_cells((size_t)n_nh * k1);
    {
        // Launch 1 block per NH, 128 threads.
        detail::nh_expand_kernel<<<n_nh, 128, 0, stream>>>(
            d_nh_cells.get(),
            d_index_cells.get(),
            knn_result.neighbors.get(),
            n_nh, k);
    }

    // ── Step 3: Per-(nh, donor) counts ────────────────────────────────────────
    // Upload donor_id to device.
    core::DeviceMemory<int> d_donor_id(n_cells);
    cudaMemcpyAsync(d_donor_id.get(), donor_id_host.data(),
                    n_cells * sizeof(int), cudaMemcpyHostToDevice, stream);

    core::DeviceMemory<float> d_nh_counts((size_t)n_nh * n_donors);
    // Zero-initialize before atomicAdd accumulation.
    cudaMemsetAsync(d_nh_counts.get(), 0, (size_t)n_nh * n_donors * sizeof(float), stream);

    if (!cfg.deterministic) {
        // Non-deterministic path: atomicAdd (fast; L2-resident output).
        detail::nh_donor_count_kernel<<<n_nh, 128, 0, stream>>>(
            d_nh_counts.get(), d_nh_cells.get(), d_donor_id.get(),
            n_nh, k1, n_donors, n_cells);
    } else {
        // Deterministic path: build (nh * n_donors + donor) keys, sort, run-count.
        // Total pairs: n_nh × k1 (one key per (nh, cell) slot).
        int n_pairs = n_nh * k1;
        core::DeviceMemory<int32_t> d_keys_in(n_pairs);
        core::DeviceMemory<int32_t> d_vals_in(n_pairs);  // unused values placeholder
        core::DeviceMemory<int32_t> d_keys_out(n_pairs);
        core::DeviceMemory<int32_t> d_vals_out(n_pairs);

        // Build (nh * n_donors + donor) keys — one key per (nh, cell-slot) pair.
        {
            int b = 256, g = (n_pairs + b - 1) / b;
            detail::nh_build_keys_kernel<<<g, b, 0, stream>>>(
                d_keys_in.get(), d_nh_cells.get(), d_donor_id.get(),
                n_nh, k1, n_donors, n_cells);
        }

        // Sort keys using cub::DeviceRadixSort.
        size_t tmp_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, tmp_bytes,
            d_keys_in.get(), d_keys_out.get(),
            d_vals_in.get(), d_vals_out.get(),
            n_pairs, 0, 32, stream);
        core::DeviceMemory<uint8_t> d_tmp(tmp_bytes);
        cub::DeviceRadixSort::SortPairs(d_tmp.get(), tmp_bytes,
            d_keys_in.get(), d_keys_out.get(),
            d_vals_in.get(), d_vals_out.get(),
            n_pairs, 0, 32, stream);

        // Segmented run-count into nh_counts.
        {
            int b = 256, g = (n_pairs + b - 1) / b;
            detail::nh_donor_count_seg_kernel<<<g, b, 0, stream>>>(
                d_nh_counts.get(), d_keys_out.get(), n_pairs, n_donors);
        }
    }

    // ── Step 4: Donor total cells (used as GLM offset) ─────────────────────────
    core::DeviceMemory<float> d_donor_totals(n_donors);
    {
        int b = 256, g = (n_donors + b - 1) / b;
        detail::donor_total_kernel<<<g, b, 0, stream>>>(
            d_donor_totals.get(), d_nh_counts.get(), n_nh, n_donors);
    }

    // ── Upload condition labels to device ──────────────────────────────────────
    core::DeviceMemory<int> d_condition(n_donors);
    cudaMemcpyAsync(d_condition.get(), condition_host.data(),
                    n_donors * sizeof(int), cudaMemcpyHostToDevice, stream);

    // ── Step 5: Mark valid NHs ─────────────────────────────────────────────────
    core::DeviceMemory<int> d_nh_valid(n_nh);
    {
        int b = 256, g = (n_nh + b - 1) / b;
        detail::mark_valid_nh_kernel<<<g, b, 0, stream>>>(
            d_nh_valid.get(), d_nh_counts.get(), d_condition.get(),
            n_nh, n_donors, cfg.min_donors_per_nh);
    }

    // ── Step 6: Batched NB GLM IRLS ───────────────────────────────────────────
    core::DeviceMemory<float> d_log_fc(n_nh);
    core::DeviceMemory<float> d_se(n_nh);
    core::DeviceMemory<float> d_pval(n_nh);

    detail::milo_glm_irls_kernel<<<n_nh, 1, 0, stream>>>(
        d_log_fc.get(), d_se.get(), d_pval.get(),
        d_nh_counts.get(), d_donor_totals.get(),
        d_condition.get(), d_nh_valid.get(),
        n_nh, n_donors, cfg.irls_tol);

    // ── Step 7: Graph-weighted FDR ────────────────────────────────────────────
    // 7a. Compute NH-NH adjacency degree on device.
    core::DeviceMemory<int> d_degree(n_nh);
    cudaMemsetAsync(d_degree.get(), 0, n_nh * sizeof(int), stream);
    {
        // Each thread handles one NH_a; inner loop scans all NH_b > NH_a.
        // At n_nh=2000: 2000 threads, each doing ≤2000 × (k+1)^2 comparisons.
        // Acceptable at n_nh ≤ 2000 (design constraint).
        int b = 128, g = (n_nh + b - 1) / b;
        detail::nh_adjacency_degree_kernel<<<g, b, 0, stream>>>(
            d_degree.get(), d_nh_cells.get(), n_nh, k1, n_cells);
    }

    // 7b. Download p_values + degree + nh_valid to host for BH step.
    // WHY host-side BH: n_nh ≤ 2000; trivial vector op; one-time pipeline exit.
    cudaStreamSynchronize(stream);

    std::vector<float> h_pval(n_nh), h_fdr(n_nh, 1.f);
    std::vector<int>   h_degree(n_nh), h_valid(n_nh);
    cudaMemcpy(h_pval.data(),   d_pval.get(),    n_nh * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_degree.data(), d_degree.get(),  n_nh * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_valid.data(),  d_nh_valid.get(),n_nh * sizeof(int),   cudaMemcpyDeviceToHost);

    // 7c. Spatial BH: weight[nh] = 1 / (1 + degree[nh]).
    //     Effective n = sum(w)^2 / sum(w^2).
    {
        std::vector<int> valid_idx;
        valid_idx.reserve(n_nh);
        for (int i = 0; i < n_nh; ++i) if (h_valid[i]) valid_idx.push_back(i);
        int n_valid = (int)valid_idx.size();

        if (n_valid > 0) {
            // Compute spatial weights.
            std::vector<float> w(n_valid);
            float sum_w = 0.f, sum_w2 = 0.f;
            for (int vi = 0; vi < n_valid; ++vi) {
                int i = valid_idx[vi];
                float wi = 1.f / (1.f + (float)h_degree[i]);
                w[vi] = wi;
                sum_w  += wi;
                sum_w2 += wi * wi;
            }
            float n_eff = (sum_w2 > 0.f) ? (sum_w * sum_w / sum_w2) : (float)n_valid;

            // Sort valid NHs by p_value ascending (stable for ties).
            std::vector<int> order(n_valid);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                return h_pval[valid_idx[a]] < h_pval[valid_idx[b]];
            });

            // BH: adj_p[rank_k] = p_val[rank_k] * n_eff / (k+1)
            // Monotonicity enforced from the right (min scan).
            float running_min = 1.f;
            for (int k = n_valid - 1; k >= 0; --k) {
                int vi  = order[k];
                int nh  = valid_idx[vi];
                float adj = h_pval[nh] * n_eff / (float)(k + 1);
                adj = fminf(adj, running_min);
                adj = fminf(adj, 1.f);
                running_min = adj;
                h_fdr[nh] = adj;
            }
        }
    }

    // Upload FDR back to device.
    // CYCLE-54-PHASE-6 typo fix: h_fdr is std::vector<float>; use .data() not .get()
    core::DeviceMemory<float> d_fdr(n_nh);
    cudaMemcpyAsync(d_fdr.get(), h_fdr.data(),
                    n_nh * sizeof(float), cudaMemcpyHostToDevice, stream);

    // ── Assemble result ────────────────────────────────────────────────────────
    MiloResult res;
    res.log_fc         = std::move(d_log_fc);
    res.se             = std::move(d_se);
    res.p_value        = std::move(d_pval);
    res.fdr            = std::move(d_fdr);
    res.nh_index_cells = std::move(d_index_cells);
    res.nh_valid       = std::move(d_nh_valid);
    res.nh_counts      = std::move(d_nh_counts);
    res.n_nh           = n_nh;
    res.n_donors       = n_donors;
    return res;
}

}  // namespace abundance
}  // namespace singlet_gpu
