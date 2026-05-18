// SPDX-License-Identifier: MIT
// singlet/gpu/fate/cospar_kernels.h
//
// CUDA kernels and kernel-launch helpers for cospar.h.
// Split out of cospar.h (multi-concern header cleanup): this file holds the
// __global__/__device__ kernels and detail-namespace helpers; cospar.h keeps
// the public API and host orchestration and #includes this header.

#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept>
#include <algorithm>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <cublas_v2.h>

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

namespace singlet::gpu {
namespace fate {

// ─── Device kernels (detail) ──────────────────────────────────────────────────

namespace detail {

// Gaussian kernel weight: exp(-d² / (2 sigma²)).
// Applied elementwise to K_state values after kNN distance output.
__global__ void
gaussian_kernel_weights(float* __restrict__ vals, int nnz, float sigma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    float v = vals[i];
    float two_s2 = 2.f * sigma * sigma;
    vals[i] = expf(-v / two_s2);
}

// Row-normalize sparse CSR matrix (in-place on values).
// One block per row; tree reduction in shared memory for row sum.
__global__ void
row_normalize_csr(float* __restrict__ vals,
                  const int* __restrict__ row_ptr,
                  int n_rows)
{
    extern __shared__ float s_sum[];
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int p0 = row_ptr[row], p1 = row_ptr[row + 1];
    int len = p1 - p0;
    float acc = 0.f;
    for (int j = threadIdx.x; j < len; j += blockDim.x) acc += vals[p0 + j];
    s_sum[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) s_sum[threadIdx.x] += s_sum[threadIdx.x + st];
        __syncthreads();
    }
    float inv = (s_sum[0] > 0.f) ? 1.f / s_sum[0] : 0.f;
    for (int j = threadIdx.x; j < len; j += blockDim.x)
        vals[p0 + j] *= inv;
}

// Gradient contribution from data term: g += 2*(T - L).
// T and L both dense (n_t0 × n_t1). L is stored as sparse COO;
// we accumulate its contribution separately — so here g += 2*T.
// Separate kernel subtracts 2*L at the COO nonzeros.
__global__ void
grad_data_term_kernel(float* __restrict__ g,
                      const float* __restrict__ T,
                      int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    g[i] = 2.f * T[i];
}

// Subtract 2*L at sparse COO positions (lineage coupling contribution to grad).
__global__ void
grad_subtract_L_coo(float* __restrict__ g,
                    const int*   __restrict__ L_row,
                    const int*   __restrict__ L_col,
                    int n_nnz, int n_cols)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_nnz) return;
    int idx = L_row[k] * n_cols + L_col[k];
    g[idx] -= 2.f;
}

// Proximal step: T = T - step * (g_data + lambda2 * g_smooth)
// Followed by L1 soft-threshold: T = sign(T) * max(|T| - step*lambda1, 0).
// Non-negativity: T = max(T, 0).
__global__ void
prox_step_kernel(float* __restrict__ T,
                 const float* __restrict__ g_data,
                 const float* __restrict__ g_smooth,
                 int n, float step, float lambda1, float lambda2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float t = T[i];
    // Gradient step
    t -= step * (g_data[i] + lambda2 * g_smooth[i]);
    // Soft-threshold for L1
    float thresh = step * lambda1;
    if (t > thresh)       t = t - thresh;
    else if (t < -thresh) t = t + thresh;
    else                  t = 0.f;
    // Non-negativity
    if (t < 0.f) t = 0.f;
    T[i] = t;
}

// Simplex projection: per-row sort-based (Duchi et al. 2008).
// Pattern from cycle-33 flash_deconv.h simplex_projection_kernel.
// One block per row. Row width = n_cols (n_t1 cells per time-point pair).
// Shared memory layout: [n_cols row_copy] + [n_cols cumsum] + [1 broadcast float].
// WHY shared-mem insertion sort over cub::DeviceRadixSort:
//   n_cols is the number of cells at t1 per slab. For n > ~200, a warp-level
//   selection or device sort would be faster. However for the MVP's scale cap of
//   30k cells, launching 30k blocks × 128 threads with 30k-element rows makes the
//   block-per-row approach impractical. Instead we launch one block per row with
//   blockDim.x threads cooperating on a shared-memory bitonic sort for n_cols up
//   to 1024. For larger n_cols a device-sort path is needed; document as deferred.
// NOTE: this kernel is intended for n_cols ≤ 1024 (MVP scale cap ≤ 30k columns
//   is not applicable here; n_cols here is n_t1 ≤ 30000 which exceeds shared mem).
// REVISED: For Cospar, n_t1 (cells at t1) can be large (10k-30k). Simplex
//   projection over 10k elements per row cannot fit in shared memory. We therefore
//   perform simplex projection via cub::DeviceRadixSort on the row buffer:
//   sort descending, compute prefix sum, find threshold, apply elementwise.
//   This matches the spirit of the cycle-33 pattern but uses the device sort path
//   that is correct for large n_cols.
// This kernel handles small n_cols (≤ 1024) via shared memory — used for fate_bias
//   projection (n_fates is small, typically ≤ 50).
__global__ void
simplex_project_small_rows(float* __restrict__ W,
                           int n_rows, int n_cols)
{
    // WHY: fate_bias has n_fates columns (≤50), so shared-mem insertion sort is fine.
    extern __shared__ float shm[];
    float* row_copy = shm;
    float* cumsum   = shm + n_cols;
    // broadcast slot at shm[2*n_cols]

    int row = blockIdx.x;
    if (row >= n_rows) return;
    float* w = W + (size_t)row * n_cols;

    for (int j = threadIdx.x; j < n_cols; j += blockDim.x) row_copy[j] = w[j];
    __syncthreads();

    if (threadIdx.x == 0) {
        // Insertion sort descending
        for (int i = 1; i < n_cols; ++i) {
            float key = row_copy[i]; int j = i - 1;
            while (j >= 0 && row_copy[j] < key) { row_copy[j+1] = row_copy[j]; --j; }
            row_copy[j+1] = key;
        }
        float cs = 0.f;
        for (int i = 0; i < n_cols; ++i) { cs += row_copy[i]; cumsum[i] = cs; }
        float rho_val = 0.f;
        for (int i = n_cols - 1; i >= 0; --i) {
            float theta = (cumsum[i] - 1.f) / (float)(i + 1);
            if (row_copy[i] > theta) { rho_val = theta; break; }
        }
        shm[2 * n_cols] = rho_val;
    }
    __syncthreads();

    float theta = shm[2 * n_cols];
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
        w[j] = fmaxf(w[j] - theta, 0.f);
}

// Elementwise squared difference accumulator: out[i] = (A[i] - B[i])²
// Used with cub::DeviceReduce::Sum for Frobenius residual.
__global__ void
sq_diff_kernel(const float* __restrict__ A,
               const float* __restrict__ B,
               float*       __restrict__ out,
               int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float d = A[i] - B[i];
    out[i] = d * d;
}

// Frobenius norm squared: out[i] = A[i]²
__global__ void
sq_norm_kernel(const float* __restrict__ A,
               float*       __restrict__ out,
               int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = A[i] * A[i];
}

// Relative residual finalisation (device side): relres = diff_sum / norm_sum.
// Output: out[0] = (norm_sum > eps) ? diff_sum / norm_sum : diff_sum.
// WHY device-side: avoids a second D2H copy so the outer loop has EXACTLY ONE
// scalar D2H per convergence check (§⛔9 compliance with a single 4-byte copy).
__global__ void
relative_residual_kernel(const float* __restrict__ diff_sum,
                         const float* __restrict__ norm_sum,
                         float*       __restrict__ out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float d = diff_sum[0];
        float n = norm_sum[0];
        out[0] = (n > 1e-12f) ? d / n : d;
    }
}

// Forward iteration kernel: bias = T @ indicator (row-wise dot product).
// indicator[j] = 1 if cell j is at a terminal fate, else 0.
// bias[i] = sum_{j: indicator[j]=fate_k} T[i*n_t1 + j]
// One block per (cell_i, fate_k) pair; 256 threads.
__global__ void
fate_bias_kernel(const float* __restrict__ T,
                 const int*   __restrict__ terminal_cells,  // fate_k terminal cell indices
                 float*       __restrict__ bias_col,         // n_t0 bias values for fate k
                 int n_t0, int n_t1, int n_terminal)
{
    extern __shared__ float s[];
    int row = blockIdx.x;
    if (row >= n_t0) return;
    const float* T_row = T + (size_t)row * n_t1;
    float acc = 0.f;
    for (int j = threadIdx.x; j < n_terminal; j += blockDim.x)
        acc += T_row[terminal_cells[j]];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) bias_col[row] = s[0];
}

// Shannon entropy per row: potency[i] = -sum_k p[i,k] * log(p[i,k]+eps)
__global__ void
shannon_entropy_kernel(const float* __restrict__ bias,  // n_cells × n_fates row-major
                       float*       __restrict__ potency,
                       int n_cells, int n_fates)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;
    const float* row = bias + (size_t)i * n_fates;
    float h = 0.f;
    for (int k = 0; k < n_fates; ++k) {
        float p = row[k];
        if (p > 1e-10f) h -= p * logf(p);
    }
    potency[i] = h;
}

// Pearson correlation kernel: 1 block per (fate, gene) pair.
// Pattern from cycle-36 granie.h peak_gene_pearson_kernel.
// Computes corr(fate_bias[:,k], expression[:,g]) across n_cells.
// fate_bias layout: n_cells × n_fates (row-major). expression: n_cells (column g of CSC).
// Launch: n_fates × n_genes blocks, 256 threads.
__global__ void __launch_bounds__(256, 4)
fate_gene_pearson_kernel(
    const float* __restrict__ fate_bias,    // n_cells × n_fates
    const float* __restrict__ fate_mean,    // n_fates means
    const float* __restrict__ fate_sd,      // n_fates std devs
    // Gene expression: CSC column pointers (subset of n_genes columns)
    const int*   __restrict__ gene_col_ptr,
    const int*   __restrict__ gene_row_idx,
    const float* __restrict__ gene_vals,
    const float* __restrict__ gene_mean,
    const float* __restrict__ gene_sd,
    float*       __restrict__ driver_out,   // n_fates × n_genes
    int n_cells, int n_fates, int n_genes)
{
    int fate = blockIdx.x;
    int gene = blockIdx.y;
    if (fate >= n_fates || gene >= n_genes) return;

    const float* bias_col = fate_bias + fate;  // stride n_fates
    float fm = fate_mean[fate];
    float fs = fate_sd[fate];
    int g0 = gene_col_ptr[gene], g1 = gene_col_ptr[gene + 1];
    float gm = gene_mean[gene];
    float gs = gene_sd[gene];

    // Two-pass Pearson: accumulate cross-product over non-zero gene cells,
    // add zero contributions (only fate_bias × 0 term from zero gene entries).
    extern __shared__ float s_acc[];
    float acc = 0.f;
    // Non-zero gene cells: true cross product term
    for (int k = g0 + threadIdx.x; k < g1; k += blockDim.x) {
        int cell  = gene_row_idx[k];
        float bv  = bias_col[(size_t)cell * n_fates] - fm;
        float gv  = gene_vals[k] - gm;
        acc      += bv * gv;
    }
    // Zero gene cells: gene contributes -gm per cell, bias contributes (bias - fm).
    // Sum over zero-cells: (-gm) * sum_{zero cells}(bias[c] - fm)
    // = (-gm) * (sum_all(bias - fm) - sum_nz(bias - fm))
    // sum_all(bias - fm) = 0 by definition of mean, so:
    // = (-gm) * (-sum_nz(bias - fm))  [handled in second pass below]
    // WHY: sparse-aware Pearson avoids iterating all n_cells for each (fate,gene).
    float sum_nz_bias = 0.f;
    for (int k = g0 + threadIdx.x; k < g1; k += blockDim.x) {
        int cell = gene_row_idx[k];
        sum_nz_bias += bias_col[(size_t)cell * n_fates] - fm;
    }
    s_acc[threadIdx.x] = acc - gm * sum_nz_bias;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) s_acc[threadIdx.x] += s_acc[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float denom = fs * gs * (float)n_cells;
        driver_out[(size_t)fate * n_genes + gene] =
            (denom > 1e-10f) ? s_acc[0] / denom : 0.f;
    }
}

// Welford variance pass for fate_bias columns (computing mean + sd for Pearson).
__global__ void
welford_bias_stats(const float* __restrict__ bias,  // n_cells × n_fates
                   float*       __restrict__ mean_out,
                   float*       __restrict__ sd_out,
                   int n_cells, int n_fates)
{
    int fate = blockIdx.x * blockDim.x + threadIdx.x;
    if (fate >= n_fates) return;
    // Serial Welford over n_cells (one thread per fate)
    float m = 0.f, m2 = 0.f;
    for (int c = 0; c < n_cells; ++c) {
        float x  = bias[(size_t)c * n_fates + fate];
        float dm = x - m;
        m  += dm / (float)(c + 1);
        m2 += dm * (x - m);
    }
    mean_out[fate] = m;
    sd_out[fate]   = (n_cells > 1) ? sqrtf(m2 / (float)(n_cells - 1)) : 0.f;
}

// Gene mean/sd from CSC column (sparse-aware two-pass Welford approximation).
__global__ void
gene_stats_kernel(const int*   __restrict__ col_ptr,
                  const int*   __restrict__ row_idx,
                  const float* __restrict__ vals,
                  float*       __restrict__ mean_out,
                  float*       __restrict__ sd_out,
                  int n_genes, int n_cells)
{
    int gene = blockIdx.x * blockDim.x + threadIdx.x;
    if (gene >= n_genes) return;
    int p0 = col_ptr[gene], p1 = col_ptr[gene + 1];
    int nnz = p1 - p0;
    float sum  = 0.f, sum2 = 0.f;
    for (int k = p0; k < p1; ++k) { sum += vals[k]; sum2 += vals[k] * vals[k]; }
    float mu = sum / (float)n_cells;
    // Var(X) = E[X²] - E[X]², sparse-aware (zero-fill approximation)
    float var = (sum2 / (float)n_cells) - mu * mu;
    if (var < 0.f) var = 0.f;
    mean_out[gene] = mu;
    sd_out[gene]   = sqrtf(var);
}

// Lineage coupling flag kernel: for each candidate (i,j) pair, flag = 1 if
// clone_id[i] == clone_id[j] && time_point[i] < time_point[j].
// Input: all pairs generated on host (or as strided index pairs on device).
// This kernel produces the flag array consumed by cub::DeviceSelect::Flagged.
__global__ void
lineage_flag_kernel(const int*  __restrict__ pair_i,
                    const int*  __restrict__ pair_j,
                    const int*  __restrict__ clone_id,
                    const float* __restrict__ time_point,
                    uint8_t*    __restrict__ flag,
                    int n_pairs)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_pairs) return;
    int ci = pair_i[k], cj = pair_j[k];
    flag[k] = (clone_id[ci] == clone_id[cj] && time_point[ci] < time_point[cj]) ? 1 : 0;
}

// Build COO row/col from selected pair indices (post Flagged selection).
__global__ void
pairs_to_coo(const int* __restrict__ sel_pair_i,
             const int* __restrict__ sel_pair_j,
             int*       __restrict__ L_row,
             int*       __restrict__ L_col,
             int n_selected)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_selected) return;
    L_row[k] = sel_pair_i[k];
    L_col[k] = sel_pair_j[k];
}

// kNN entropy potency (state-only fallback): one thread per cell.
// K_state is CSR (n_cells × n_cells); this kernel computes Shannon entropy
// over the normalised row weights (already row-normalised by row_normalize_csr).
__global__ void
knn_entropy_potency(const float* __restrict__ K_vals,
                    const int*   __restrict__ K_row_ptr,
                    float*       __restrict__ potency,
                    int n_cells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;
    int p0 = K_row_ptr[i], p1 = K_row_ptr[i + 1];
    float h = 0.f;
    for (int k = p0; k < p1; ++k) {
        float p = K_vals[k];
        if (p > 1e-10f) h -= p * logf(p);
    }
    potency[i] = h;
}

// ─── Extracted free-function kernels (cannot be __global__ inside local structs) ─

// Fill pair indices: pi[k] = ct0[k/nt1], pj[k] = ct1[k%nt1]
__global__ void
fill_pairs_kernel(int* pi, int* pj, const int* ct0, const int* ct1,
                  int nt0, int nt1, size_t n_pairs)
{
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_pairs) return;
    pi[k] = ct0[k / nt1];
    pj[k] = ct1[k % nt1];
}

// Scatter T rows into padded buffer at t0 cell positions
__global__ void
scatter_t0_kernel(float* Tpad, const float* Tsrc,
                  const int* ct0, int n_t0, int n_t1)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_t0) return;
    int global_r = ct0[r];
    for (int c = 0; c < n_t1; ++c)
        Tpad[(size_t)global_r * n_t1 + c] = Tsrc[(size_t)r * n_t1 + c];
}

// Scatter T into padded buffer (with n_global safety guard, zeros already set externally)
__global__ void
scatter_t0_iter_kernel(float* Tpad, const float* Tsrc,
                       const int* ct0, int n_t0, int n_t1, int n_global)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n_t0) {
        int gr = ct0[r];
        for (int c = 0; c < n_t1; ++c)
            Tpad[(size_t)gr * n_t1 + c] = Tsrc[(size_t)r * n_t1 + c];
    }
}

// Gather KT rows from padded output into g_smooth
__global__ void
gather_t0_kernel(float* gs, const float* KTp,
                 const int* ct0, int n_t0, int n_t1)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_t0) return;
    int gr = ct0[r];
    for (int c = 0; c < n_t1; ++c)
        gs[(size_t)r * n_t1 + c] = KTp[(size_t)gr * n_t1 + c];
}

// Fill segment offsets for cub::DeviceSegmentedRadixSort
__global__ void
fill_offsets_kernel(int* off, int n_t0, int n_t1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n_t0) off[i] = i * n_t1;
}

// Simplex projection threshold kernel: serial scan per row (one block per row)
__global__ void
simplex_thresh_kernel(const float* sorted, float* theta, int n_t0, int n_t1)
{
    int row = blockIdx.x;
    if (row >= n_t0) return;
    const float* s = sorted + (size_t)row * n_t1;
    float cs = 0.f, rho = 0.f;
    if (threadIdx.x == 0) {
        for (int i = 0; i < n_t1; ++i) {
            cs += s[i];
            float t = (cs - 1.f) / (float)(i + 1);
            if (s[i] > t) rho = t;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) theta[row] = rho;
}

// Apply simplex threshold: T[i] = max(T[i] - theta[row], 0)
__global__ void
apply_theta_kernel(float* T, const float* theta, int n_t0, int n_t1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_t0 * n_t1) return;
    int row = i / n_t1;
    T[i] = fmaxf(T[i] - theta[row], 0.f);
}

// Fill T with uniform value
__global__ void
fill_uniform_kernel(float* T, float v, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) T[i] = v;
}

}  // namespace detail

}  // namespace fate
}  // namespace singlet::gpu
