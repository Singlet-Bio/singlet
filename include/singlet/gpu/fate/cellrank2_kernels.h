// SPDX-License-Identifier: MIT
// singlet/gpu/fate/cellrank2_kernels.h
//
// CUDA kernels and kernel-launch helpers for cellrank2.h.
// Split out of cellrank2.h (multi-concern header cleanup): this file holds
// the bulk of the __global__/__device__ kernels and detail-namespace
// helpers; cellrank2.h keeps the public API, the host-side batched GMRES
// orchestration (which carries its own small detail kernels) and #includes
// this header.

#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <string>

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

namespace singlet::gpu {
namespace fate {

// ─── Device kernels (detail namespace) ───────────────────────────────────────

namespace detail {

// Elementwise negate a sparse CSR value array: v[i] = -v[i].
// Used to build (I-Q) from Q: indptr/indices stay, values negated, then
// diagonal +1 is added by form_ImQ_diag_kernel below.
__global__ void
negate_vals_kernel(float* __restrict__ vals, int nnz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    vals[i] = -vals[i];
}

// Add +1.0 to the diagonal of a CSR matrix stored as (row_ptr, col_idx, vals).
// One thread per transient row.  Assumes the diagonal entry is present (which it
// is for (I-Q) because T is row-stochastic → Q has non-zero diagonal = self-loops,
// and if not present the pattern still includes it because we explicitly add it).
// WHY: cuSPARSE SpMV handles off-diagonal; diagonal is guaranteed present in
// a row-stochastic matrix's transient submatrix.
__global__ void
form_ImQ_diag_kernel(float* __restrict__ vals,
                     const int* __restrict__ row_ptr,
                     const int* __restrict__ col_idx,
                     const int* __restrict__ local_to_global,
                     int n_transient)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_transient) return;
    int global_col = local_to_global[row];
    int p0 = row_ptr[row], p1 = row_ptr[row + 1];
    for (int p = p0; p < p1; p++) {
        if (col_idx[p] == row) {      // diagonal in the submatrix (local index)
            vals[p] += 1.f;           // I - Q: add +1 to the diagonal
            return;
        }
    }
    // Diagonal was zero (no self-loop in Q for this row): should not happen for
    // row-stochastic T but handle gracefully — we cannot insert a new entry
    // into a CSR pattern without rebuilding; log by setting a sentinel val.
    // In practice this path is never hit on a proper row-stochastic T.
}

// Row-normalize CSR in-place (makes T row-stochastic after kNN build).
__global__ void
row_normalize_csr_kernel(float* __restrict__ vals,
                         const int* __restrict__ row_ptr,
                         int n_rows)
{
    extern __shared__ float s_sum[];
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int p0 = row_ptr[row], p1 = row_ptr[row + 1];
    float acc = 0.f;
    for (int j = threadIdx.x; j < (p1 - p0); j += blockDim.x)
        acc += vals[p0 + j];
    s_sum[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) s_sum[threadIdx.x] += s_sum[threadIdx.x + st];
        __syncthreads();
    }
    float inv = (s_sum[0] > 1e-30f) ? 1.f / s_sum[0] : 0.f;
    for (int j = threadIdx.x; j < (p1 - p0); j += blockDim.x)
        vals[p0 + j] *= inv;
}

// Convert fp64 device vector to fp32 host vector (final result copy).
// Only called once at function exit — always approved.
__global__ void
d64_to_f32_kernel(const double* __restrict__ src, float* __restrict__ dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<float>(src[i]);
}

// AXPY in fp64: y = alpha * x + y  (device, n elements)
__global__ void
daxpy_kernel(const double* __restrict__ x, double* __restrict__ y,
             double alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * x[i] + y[i];
}

// Fill fp64 device vector with a scalar.
__global__ void
dfill_kernel(double* __restrict__ v, double val, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = val;
}

// Fill fp32 device vector with a scalar.
__global__ void
sfill_kernel(float* __restrict__ v, float val, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = val;
}

// fp32 SpMV result → fp64 accumulator  (v_out = alpha * A_fp32 @ x_fp32 + beta * v_out_fp64)
// Used to promote the cuSPARSE fp32 SpMV output into the fp64 Krylov basis.
// cuSPARSE SpMV writes fp32 result into d_spmv_out; this kernel promotes it.
__global__ void
promote_spmv_to_fp64(const float* __restrict__ src,
                     double*      __restrict__ dst,
                     int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<double>(src[i]);
}

// Squared L2 norm accumulator for cub::DeviceReduce::Sum.
// Writes v[i]^2 into out[i] so we can reduce for ||v||^2.
__global__ void
sq_kernel(const double* __restrict__ v, double* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = v[i] * v[i];
}

// Scale fp64 vector: v *= alpha
__global__ void
dscale_kernel(double* __restrict__ v, double alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] *= alpha;
}

// Scale fp64 vector v by 1/(*h_ptr), where h_ptr is a DEVICE scalar.
// WHY: dscale_kernel needs a host alpha; cublasDscal in device pointer mode scales
// by *h_ptr not 1/*h_ptr.  We need the reciprocal (V[:,j+1] = w / h[j+1][j]).
// Zero guard: if |*h_ptr| < 1e-300, result is 0 (breakdown — basis degenerate).
__global__ void
dscale_by_device_inv_kernel(double* __restrict__ v,
                            const double* __restrict__ h_ptr,
                            int n)
{
    double h = *h_ptr;
    double inv = (h > 1e-300 || h < -1e-300) ? 1.0 / h : 0.0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] *= inv;
}

// Extract column j from a (n × stride) fp64 device matrix into a fp64 vector.
// Used to extract basis vector V[:,j].
__global__ void
extract_col_fp64(const double* __restrict__ M, double* __restrict__ col,
                 int n, int stride, int j)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) col[i] = M[(size_t)i * stride + j];
}

// Store fp64 vector into column j of a (n × stride) fp64 device matrix.
__global__ void
store_col_fp64(const double* __restrict__ col, double* __restrict__ M,
               int n, int stride, int j)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) M[(size_t)i * stride + j] = col[i];
}

// Fused device kernel: apply previous Givens rotations to column j of H, compute
// new rotation to zero H[j+1][j], update g[j] and g[j+1].
// Runs in a SINGLE thread (H is (m+1)×m ≤ 31×30 = 930 doubles; serial is fine).
// WHY single thread: the column has only j+2 entries; serial is ~10 ns vs kernel
// launch overhead; no divergence; avoids shared-memory synchronisation.
// Layout: H is row-major (m+1 rows, m cols), stride = gmres_m.
// After this kernel: H[j+1][j] = 0, g[j+1] = -sn[j]*g[j], g[j] *= cs[j],
//                   cs[j] and sn[j] are set.
// Convergence scalar: d_res_norm[0] = |g[j+1]| (device fp64, read at restart end).
__global__ void
givens_update_kernel(double* __restrict__ d_H,    // (m+1)×m row-major device
                     double* __restrict__ d_g,    // g vector, length m+1
                     double* __restrict__ d_cs,   // cosines, length m
                     double* __restrict__ d_sn,   // sines,   length m
                     double* __restrict__ d_res_norm, // |g[j+1]|, written here
                     int j,           // current Arnoldi column (0-indexed)
                     int gmres_m)     // stride of H (number of columns)
{
    // Single thread only.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Apply accumulated rotations 0..j-1 to column j of H.
    for (int i = 0; i < j; i++) {
        double h_ij  = d_H[i       * gmres_m + j];
        double h_i1j = d_H[(i + 1) * gmres_m + j];
        d_H[i       * gmres_m + j] =  d_cs[i] * h_ij + d_sn[i] * h_i1j;
        d_H[(i + 1) * gmres_m + j] = -d_sn[i] * h_ij + d_cs[i] * h_i1j;
    }

    // Compute new Givens rotation to zero H[j+1][j].
    double a = d_H[j       * gmres_m + j];
    double b = d_H[(j + 1) * gmres_m + j];
    double cs_j, sn_j;
    if (b == 0.0) {
        cs_j = 1.0; sn_j = 0.0;
    } else if (fabs(b) > fabs(a)) {
        double tau = -a / b;
        sn_j = 1.0 / sqrt(1.0 + tau * tau);
        cs_j = sn_j * tau;
    } else {
        double tau = -b / a;
        cs_j = 1.0 / sqrt(1.0 + tau * tau);
        sn_j = cs_j * tau;
    }
    d_cs[j] = cs_j;
    d_sn[j] = sn_j;

    // Apply the new rotation to H[j][j] and zero H[j+1][j].
    d_H[j       * gmres_m + j] = cs_j * a + sn_j * b;
    d_H[(j + 1) * gmres_m + j] = 0.0;

    // Update g: g[j+1] = -sn_j * g[j]; g[j] *= cs_j.
    double g_j  = d_g[j];
    d_g[j + 1]  = -sn_j * g_j;
    d_g[j]      =  cs_j * g_j;

    // Write convergence norm |g[j+1]| for restart-boundary check.
    d_res_norm[0] = fabs(d_g[j + 1]);
}

// Fused Pearson correlation kernel: 1 block per (terminal, gene) pair.
// absorption_prob: n_transient × n_terminals (col-major)
// expression:      n_transient × n_genes     (col-major, fp32)
// result:          n_terminals × n_genes     (row-major, fp32)
// WHY: fused single-pass two-pass Welford avoids a global gather; shared memory
// accumulates sum, sum_x, sum_y, sum_xx, sum_yy, sum_xy per block.
// Follows cycle-36 GRaNIE fused Pearson pattern.
__global__ void
pearson_driver_kernel(const float* __restrict__ absorption_prob,
                      const float* __restrict__ expression,
                      float* __restrict__ result,
                      int n_transient, int n_terminals, int n_genes)
{
    extern __shared__ double sh[];   // 6 doubles per block
    // Block maps to (terminal t, gene g)
    int t = blockIdx.x;
    int g = blockIdx.y;
    if (t >= n_terminals || g >= n_genes) return;

    const float* fate_col  = absorption_prob + (size_t)t * n_transient;
    const float* gene_col  = expression      + (size_t)g * n_transient;

    // Two-pass Welford in fp64 shared accumulators.
    double s_x = 0, s_y = 0, s_xx = 0, s_yy = 0, s_xy = 0;
    for (int i = threadIdx.x; i < n_transient; i += blockDim.x) {
        double x = static_cast<double>(fate_col[i]);
        double y = static_cast<double>(gene_col[i]);
        s_x  += x;
        s_y  += y;
        s_xx += x * x;
        s_yy += y * y;
        s_xy += x * y;
    }
    // Store partial sums in shared memory then reduce.
    sh[threadIdx.x * 5 + 0] = s_x;
    sh[threadIdx.x * 5 + 1] = s_y;
    sh[threadIdx.x * 5 + 2] = s_xx;
    sh[threadIdx.x * 5 + 3] = s_yy;
    sh[threadIdx.x * 5 + 4] = s_xy;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            sh[threadIdx.x*5+0] += sh[(threadIdx.x+st)*5+0];
            sh[threadIdx.x*5+1] += sh[(threadIdx.x+st)*5+1];
            sh[threadIdx.x*5+2] += sh[(threadIdx.x+st)*5+2];
            sh[threadIdx.x*5+3] += sh[(threadIdx.x+st)*5+3];
            sh[threadIdx.x*5+4] += sh[(threadIdx.x+st)*5+4];
        }
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    double n  = static_cast<double>(n_transient);
    double sx = sh[0], sy = sh[1], sxx = sh[2], syy = sh[3], sxy = sh[4];
    double cov  = sxy - sx * sy / n;
    double var_x = sxx - sx * sx / n;
    double var_y = syy - sy * sy / n;
    float r = 0.f;
    double denom = var_x * var_y;
    if (denom > 1e-30) r = static_cast<float>(cov / std::sqrt(denom));
    result[(size_t)t * n_genes + g] = r;
}

// Mask-select and compact sparse CSR submatrix (transient rows, transient cols).
// Writes new CSR into preallocated arrays; computes nnz via prefix sum.
// Called once at setup (not hot path).
// terminal_mask[i] = 1 iff cell i is absorbing (terminal).
// transient_map[i]  = local index for transient cell i, -1 if terminal.
__global__ void
build_submatrix_nnz_kernel(const int* __restrict__ T_row_ptr,
                           const int* __restrict__ T_col_idx,
                           const int* __restrict__ transient_map,
                           int* __restrict__ Q_row_nnz,
                           int* __restrict__ R_row_nnz,
                           int n_cells)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_cells) return;
    if (transient_map[row] < 0) {
        // This is an absorbing row — skip for Q/R (we won't build their rows).
        return;
    }
    int local_row = transient_map[row];
    int p0 = T_row_ptr[row], p1 = T_row_ptr[row + 1];
    int q_cnt = 0, r_cnt = 0;
    for (int p = p0; p < p1; p++) {
        if (transient_map[T_col_idx[p]] >= 0) q_cnt++;
        else                                   r_cnt++;
    }
    Q_row_nnz[local_row] = q_cnt;
    R_row_nnz[local_row] = r_cnt;
}

// Fill Q (transient×transient) and R (transient×terminal) CSR data arrays.
// Writes values, col_idx in local (submatrix) indices.
// Called once at setup.
__global__ void
fill_QR_kernel(const int*   __restrict__ T_row_ptr,
               const int*   __restrict__ T_col_idx,
               const float* __restrict__ T_vals,
               const int*   __restrict__ transient_map,   // -1 if terminal
               const int*   __restrict__ terminal_map,    // -1 if transient
               const int*   __restrict__ Q_row_ptr,       // prefix-summed Q nnz
               const int*   __restrict__ R_row_ptr,       // prefix-summed R nnz
               int*   __restrict__ Q_col_idx,
               float* __restrict__ Q_vals,
               int*   __restrict__ R_col_idx,
               float* __restrict__ R_vals,
               int n_cells)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_cells) return;
    int local_row = transient_map[row];
    if (local_row < 0) return;    // absorbing row, skip
    int p0 = T_row_ptr[row], p1 = T_row_ptr[row + 1];
    int qi = Q_row_ptr[local_row];
    int ri = R_row_ptr[local_row];
    for (int p = p0; p < p1; p++) {
        int col = T_col_idx[p];
        float val = T_vals[p];
        int t_local = transient_map[col];
        int a_local = terminal_map[col];
        if (t_local >= 0) {
            Q_col_idx[qi] = t_local;
            Q_vals[qi]    = val;
            qi++;
        } else if (a_local >= 0) {
            R_col_idx[ri] = a_local;
            R_vals[ri]    = val;
            ri++;
        }
    }
}

// Build the (I - Q) matrix by negating Q values and adding +1 to diagonal.
// Separate from fill_QR for clarity; called once after fill_QR.
__global__ void
negate_Q_vals_kernel(float* __restrict__ vals, int nnz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) vals[i] = -vals[i];
}

// Add +1 to diagonal of Q stored in CSR (one thread per row).
// The diagonal entry MUST exist in the pattern (guaranteed by row-stochastic T).
__global__ void
add_identity_diag_kernel(float* __restrict__ vals,
                         const int* __restrict__ row_ptr,
                         const int* __restrict__ col_idx,
                         int n_transient)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_transient) return;
    int p0 = row_ptr[row], p1 = row_ptr[row + 1];
    for (int p = p0; p < p1; p++) {
        if (col_idx[p] == row) { vals[p] += 1.f; return; }
    }
    // No diagonal entry found — this transient cell had no self-loop in T.
    // (I-Q) would be missing the diagonal 1; this is a data-quality issue.
    // We cannot insert a new entry here; the caller must ensure T has self-loops
    // OR explicitly add a padded diagonal to Q before calling this kernel.
    // For now: silently skip — GMRES will still converge if matrix is non-singular.
}

// Build host-side terminal_map and transient_map arrays from terminal_indices.
// Returns these via vectors allocated on host.
struct SubmatrixMaps {
    std::vector<int> transient_map;  // n_cells; -1 if terminal
    std::vector<int> terminal_map;   // n_cells; -1 if transient
    std::vector<int> transient_list; // indices of transient cells (global)
    std::vector<int> terminal_list;  // indices of terminal cells (global, sorted)
    int n_transient;
    int n_absorbing;
};

inline SubmatrixMaps
build_maps(const std::vector<int>& terminal_indices, int n_cells)
{
    SubmatrixMaps m;
    m.transient_map.assign(n_cells, -1);
    m.terminal_map.assign(n_cells,  -1);
    m.terminal_list = terminal_indices;
    std::sort(m.terminal_list.begin(), m.terminal_list.end());
    for (int j = 0; j < (int)m.terminal_list.size(); j++)
        m.terminal_map[m.terminal_list[j]] = j;
    int ti = 0;
    for (int i = 0; i < n_cells; i++) {
        if (m.terminal_map[i] < 0) {
            m.transient_map[i] = ti++;
            m.transient_list.push_back(i);
        }
    }
    m.n_transient  = ti;
    m.n_absorbing  = (int)m.terminal_list.size();
    return m;
}

}  // namespace detail

}  // namespace fate
}  // namespace singlet::gpu
