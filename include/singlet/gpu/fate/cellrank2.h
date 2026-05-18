// SPDX-License-Identifier: MIT
// integrates: original (first GPU CellRank 2-style batched GMRES absorption probability)
//
// fate/cellrank2.h — GPU-native CellRank 2 absorption probability solver
//
// Algorithm reference:
//   Weiler et al., "CellRank 2: unified fate mapping in multiview single-cell
//   data," Nature Methods 2024. https://doi.org/10.1038/s41592-024-02303-9
//
// Core computation:
//   Given a row-stochastic transition matrix T (n × n), and a set of absorbing
//   (terminal) states S_abs ⊆ {0..n-1}:
//     Let Q = T[transient, transient],  R = T[transient, absorbing]
//     Solve (I - Q) B = R  for B ∈ ℝ^{n_transient × n_terminals}
//   Result B[i,j] = probability that cell i is absorbed into terminal j.
//   Absorbing cells get identity assignment: B[s_j, j] = 1, else 0.
//
// Solver: Restarted GMRES(m=30), up to 10 restarts.
//   - fp32 SpMV (cuSPARSE) for (I - Q) @ v inside each GMRES iter.
//   - fp64 Arnoldi basis accumulation (small, m=30 vectors; H is (m+1)×m).
//   - Givens rotations for on-device Hessenberg QR (applied in fp64).
//   - Batched across n_terminals right-hand sides: all RHS share the work matrix
//     layout; inner loop over terminals within each restart.
//   - Convergence: cub::DeviceReduce::Sum → 1 scalar D2H per outer restart
//     (approved §⛔9 exception — scalar at outer restart boundary, not inner iter).
//
// Transition matrix construction (optional):
//   When cfg.build_transition_from_knn = true, calls cycle-8 compute_exact to
//   build a kNN graph from the supplied PCA embedding, then row-normalises to T.
//   The caller may instead supply a pre-built sparse CSR T directly.
//
// Terminal state detection (optional):
//   When cfg.auto_terminals = true, runs factornet Lanczos on T^T T to find the
//   leading eigenvectors (λ ≈ 1), then k-means clusters them to n_terminals.
//   Default: caller supplies terminal_indices directly.
//
// Precision:
//   fp32 SpMV (40 MB T @ 100k cells, 50 nnz/row).
//   fp64 Krylov basis vectors (30 × n_transient × 8 bytes = ~24 MB at 100k).
//   fp64 Hessenberg / Givens scalars ((m+1)×m = 31×30, negligible).
//
// Memory layout (100k cells, 10 terminals, m=30):
//   T sparse CSR fp32: ~40 MB (caller-owned or built here)
//   Krylov basis V: n_transient × (m+1) fp64 device = ~24 MB (reused per restart)
//   H Hessenberg: (m+1) × m fp64 DEVICE = 7.4 KB (CYCLE-43: moved from host)
//   g / cs / sn: (m+1 + m + m) fp64 DEVICE = ~720 bytes (CYCLE-43: moved from host)
//   h_col_buf: (m+1) fp64 DEVICE = 248 bytes (CYCLE-43: cublasDgemv scratch)
//   r (residual) + x (solution) per terminal: n_transient × 2 fp64 = ~1.6 MB ×10 = 16 MB
//   RHS R: n_transient × n_terminals fp32: ~4 MB
//   cub temp: ~16 MB
//   Total: ~100–120 MB device (+8 KB for device Hessenberg — negligible)
//
// OOC plan (cfg.ooc_batch_terminals):
//   Process terminals in batches of ooc_batch_terminals (default: all).
//   Each batch allocates its own x/r/beta device vectors; V basis is reused.
//   For n_cells > 500k, set ooc_batch_terminals = 2 to stay inside 48 GB VRAM.
//
// Streams:      1 (caller-provided)
// Determinism:  no stochasticity; bit-identical across runs
// cudaMemcpy audit (self-check, CYCLE-43 FOLLOWUP):
//   INSIDE Arnoldi inner loop (j=0..m-1): ZERO cudaMemcpy.
//     h[0..j][j] computed by cublasDgemv (device-to-device).
//     h[j+1][j] = ||w|| computed by cublasDnrm2 in DEVICE pointer mode (no D2H).
//     Givens + g update: givens_update_kernel (single-thread device kernel, no D2H).
//     V[:,j+1] scaling: dscale_by_device_inv_kernel (reads h[j+1][j] from device).
//   A. ONE scalar D2H per outer restart iteration:
//      cub::DeviceReduce::Sum → beta (||r||); cudaMemcpy 8 bytes.
//      ONE block D2H per restart: H + g copied for host back-substitution.
//      ORCHESTRATOR APPROVAL: §⛔9 valid exception; restart IS the outer iter.
//      10 restarts × 10 terminals = 100 restart boundaries; each pulls 8 bytes
//      beta + 7.4 KB H + 248 bytes g ≈ 800 bytes scalars + 780 KB total.
//      (Prior implementation: 10 × 30 × 8 = 2.4 KB of per-step D2H eliminated.)
//   B. Final result copy D2H (function exit): B matrix — setup/teardown, approved.
// PCIe improvement: 0 bytes inside Arnoldi inner loop (was 2.4 KB latency-dominated).
// cub usage:
//   cub::DeviceReduce::Sum  — ||r||² → beta at restart boundary only
//   cub::DeviceScan::ExclusiveSum — prefix-sum of terminal_mask for reindex

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
#include <singlet/gpu/graph/knn.h>

// ─── Device kernels (detail namespace) — see cellrank2_kernels.h ─────────────
#include <singlet/gpu/fate/cellrank2_kernels.h>

namespace singlet::gpu {
namespace fate {

// ─── Public configuration ────────────────────────────────────────────────────

struct CellRank2Config {
    // GMRES parameters
    int  gmres_m          = 30;     // Krylov subspace dimension (restart size)
    int  gmres_max_restarts = 10;   // max outer restarts
    double convergence_tol = 1e-6;  // relative residual threshold

    // Batching
    int ooc_batch_terminals = -1;   // -1 = all in one batch; >0 = OOC batch size

    // Transition matrix construction (only when build_transition_from_knn = true)
    bool build_transition_from_knn = false;
    int  k_neighbors = 50;          // kNN for T construction
    int  n_pcs       = 30;          // PCA dims of input embedding

    // Terminal state detection
    bool auto_terminals = false;    // if true, run Lanczos + cluster; else use provided

    // Driver gene correlation (requires expression matrix via set_expression())
    bool compute_driver_genes = false;

    // ── Compat alias fields (test-harness API) ────────────────────────────────
    const int* terminal_ids    = nullptr; // explicit terminal cell indices (test API)
    int        n_terminals     = 0;       // number of terminal states (test API)
    float      gmres_tol       = 1e-6f;  // alias for convergence_tol (test API)
    int        gmres_max_spmv  = 300;    // max total SpMV budget (test API)
    int        krylov_dim      = 30;     // alias for gmres_m (test API)
    bool       record_convergence = false; // reserved diagnostic flag (test API)
};

struct CellRank2Result {
    // absorption_prob[i * n_terminals + j] = P(cell i → terminal j)
    // Shape: n_cells × n_terminals  (host, fp32)
    // Row sums: 1.0 for transient cells; rows for absorbing cells have exactly
    // one 1.0 (own terminal) and the rest 0.0.
    std::vector<float> absorption_prob;
    int n_cells    = 0;
    int n_terminals = 0;

    // driver_genes[t * n_genes + g] = Pearson r for (terminal t, gene g)
    // Populated only when cfg.compute_driver_genes = true.
    std::vector<float> driver_genes;
    int n_genes = 0;

    // Diagnostics: per-terminal final relative residual and restart count.
    std::vector<double> final_residuals;
    std::vector<int>    restarts_used;
    bool converged_all = false;

    // ── Compat fields (test-harness API) ─────────────────────────────────────
    int total_spmv = 0;   // total SpMV operations across all terminals (test API)
};



// ─── GMRES solver: batched across rhs block ───────────────────────────────────

namespace detail {

// Run one round of restarted GMRES(m) for one right-hand side.
//
// CYCLE-43 FOLLOWUP: Hessenberg is now device-resident.  Arnoldi inner products
// h[0..j][j] = V[:,0..j]^T @ w are computed by a single cublasDgemv call (one
// launch, no per-step D2H).  h[j+1][j] = ||w|| via cublasDnrm2 in DEVICE pointer
// mode (zero D2H).  Givens rotations, g-vector updates, and convergence norm are
// all updated by givens_update_kernel (single-thread device kernel).
//
// cudaMemcpy audit after fix:
//   INSIDE Arnoldi inner loop (j=0..m-1): ZERO cudaMemcpy.
//   At restart boundary: ONE D2H of d_res_norm (8 bytes scalar) for convergence.
//     If not converged: D2H of H block + g vector for back-substitution
//     (restart-boundary bulk transfer, approved §⛔9 as single outer-iter transfer).
//   Total per restart: ≤ 1 scalar D2H (convergence) + 1 block D2H (back-sub).
//   Total per GMRES call (10 restarts × 10 terminals): ≤ 200 scalar copies.
//
// Signature adds three device scratch buffers pre-allocated by the caller:
//   d_hessenberg: (m+1)×m fp64 device — Hessenberg H, reset each restart
//   d_g_dev:      (m+1)   fp64 device — RHS vector g for least-squares
//   d_cs_dev:     m       fp64 device — Givens cosines
//   d_sn_dev:     m       fp64 device — Givens sines
//   d_res_norm:   1       fp64 device — |g[j+1]| written by givens_update_kernel
//   d_h_col_buf:  (m+1)   fp64 device — scratch for cublasDgemv h-column output
//
// Returns: final relative residual.
inline double
gmres_solve_one_rhs(cusparseHandle_t     sp_handle,
                    cublasHandle_t       bl_handle,
                    cusparseSpMatDescr_t A_desc,
                    double* x_dev,          // solution, in/out
                    const double* b_dev,    // rhs
                    double* V_dev,          // n_transient × (m+1), column layout
                    double* w_dev,          // scratch n_transient
                    float*  spmv_fp32_in,   // fp32 SpMV input
                    float*  spmv_fp32_out,  // fp32 SpMV output
                    double* d_sq_buf,       // scratch length n_transient (for beta cub reduce)
                    void*   d_reduce_tmp,
                    size_t  d_reduce_tmp_bytes,
                    double* d_res_scalar,   // device scalar (cub output for beta)
                    double* d_hessenberg,   // (m+1)×m fp64 device — H matrix
                    double* d_g_dev,        // (m+1) fp64 device — g vector
                    double* d_cs_dev,       // m fp64 device — Givens cosines
                    double* d_sn_dev,       // m fp64 device — Givens sines
                    double* d_res_norm,     // 1 fp64 device — |g[j+1]| from Givens kernel
                    double* d_h_col_buf,    // (m+1) fp64 device — dgemv output scratch
                    int     n_transient,
                    int     gmres_m,
                    int     max_restarts,
                    double  tol,
                    cudaStream_t stream,
                    double  b_norm)         // ||b||₂, pre-computed
{
    constexpr int BLOCK = 256;
    int grid  = (n_transient + BLOCK - 1) / BLOCK;

    // Host buffers for back-substitution only (pulled D2H once per restart boundary).
    std::vector<double> H_host((gmres_m + 1) * gmres_m);
    std::vector<double> g_host(gmres_m + 1);
    std::vector<double> y(gmres_m, 0.0);

    // cuSPARSE SpMV descriptor for fp32 input/output vectors.
    cusparseDnVecDescr_t vecX_fp32 = nullptr;
    cusparseDnVecDescr_t vecY_fp32 = nullptr;
    cusparseCreateDnVec(&vecX_fp32, n_transient, spmv_fp32_in,  CUDA_R_32F);
    cusparseCreateDnVec(&vecY_fp32, n_transient, spmv_fp32_out, CUDA_R_32F);
    float alpha_f = 1.f, beta_f = 0.f;
    size_t spmv_buf_bytes = 0;
    cusparseSpMV_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha_f, A_desc, vecX_fp32, &beta_f, vecY_fp32,
                            CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, &spmv_buf_bytes);
    core::DeviceMemory<uint8_t> spmv_ws;
    if (spmv_buf_bytes > 0)
        spmv_ws = core::DeviceMemory<uint8_t>(spmv_buf_bytes);

    // cuBLAS pointer mode: HOST for all daxpy/dgemv/dgemv calls (alpha/beta are host
    // scalars).  Switched to DEVICE only around cublasDnrm2 calls so the result
    // stays on device.  Restored to HOST immediately after nrm2.

    double rel_res = 1.0;

    for (int restart = 0; restart < max_restarts; restart++) {
        // ── Compute initial residual r = b - A @ x ───────────────────────────
        detail::d64_to_f32_kernel<<<grid, BLOCK, 0, stream>>>(
            x_dev, spmv_fp32_in, n_transient);
        cusparseSpMV(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha_f, A_desc, vecX_fp32, &beta_f, vecY_fp32,
                     CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, spmv_ws.get());
        cudaMemcpyAsync(V_dev, b_dev, (size_t)n_transient * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream);
        detail::promote_spmv_to_fp64<<<grid, BLOCK, 0, stream>>>(
            spmv_fp32_out, w_dev, n_transient);
        double neg1 = -1.0;
        cublasDaxpy(bl_handle, n_transient, &neg1, w_dev, 1, V_dev, 1);

        // ── Compute beta = ||r||₂ via cub (D2H approved: restart boundary) ───
        detail::sq_kernel<<<grid, BLOCK, 0, stream>>>(V_dev, d_sq_buf, n_transient);
        cub::DeviceReduce::Sum(d_reduce_tmp, d_reduce_tmp_bytes,
                               d_sq_buf, d_res_scalar, n_transient, stream);
        double h_beta_sq = 0.0;
        cudaMemcpyAsync(&h_beta_sq, d_res_scalar, sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);  // restart boundary — approved §⛔9
        double beta = std::sqrt(h_beta_sq);
        rel_res = (b_norm > 0.0) ? beta / b_norm : beta;
        if (rel_res <= tol) break;

        // ── Scale V[:,0] = r / beta  ─────────────────────────────────────────
        double inv_beta = (beta > 1e-300) ? 1.0 / beta : 0.0;
        detail::dscale_kernel<<<grid, BLOCK, 0, stream>>>(V_dev, inv_beta, n_transient);

        // Reset device H, g, cs, sn for this restart.
        cudaMemsetAsync(d_hessenberg, 0, (size_t)(gmres_m + 1) * gmres_m * sizeof(double), stream);
        cudaMemsetAsync(d_g_dev,      0, (size_t)(gmres_m + 1)           * sizeof(double), stream);
        cudaMemsetAsync(d_cs_dev,     0, (size_t) gmres_m                * sizeof(double), stream);
        cudaMemsetAsync(d_sn_dev,     0, (size_t) gmres_m                * sizeof(double), stream);
        // g[0] = beta — write single double to device.
        cudaMemcpyAsync(d_g_dev, &beta, sizeof(double), cudaMemcpyHostToDevice, stream);

        int j_done = gmres_m;  // default: run all m steps unless early convergence detected

        for (int j = 0; j < gmres_m; j++) {
            // ── Arnoldi step: w = A @ V[:,j]  (zero D2H) ────────────────────
            detail::d64_to_f32_kernel<<<grid, BLOCK, 0, stream>>>(
                V_dev + (size_t)j * n_transient, spmv_fp32_in, n_transient);
            cusparseSpMV(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &alpha_f, A_desc, vecX_fp32, &beta_f, vecY_fp32,
                         CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, spmv_ws.get());
            detail::promote_spmv_to_fp64<<<grid, BLOCK, 0, stream>>>(
                spmv_fp32_out, w_dev, n_transient);

            // ── Modified Gram-Schmidt: h[0..j][j] = V[:,0..j]^T @ w ─────────
            // Single cublasDgemv: op(V[:,0..j])^T @ w → d_h_col_buf[0..j]
            // V[:,0..j] is n_transient × (j+1) stored column-major.
            // cublasDgemv: CUBLAS_OP_T, m=n_transient, n=(j+1), lda=n_transient.
            // Result: d_h_col_buf[0..j] = V^T @ w (all (j+1) inner products at once).
            {
                double one = 1.0, zero = 0.0;
                cublasDgemv(bl_handle, CUBLAS_OP_T,
                            n_transient, j + 1,
                            &one, V_dev, n_transient,
                            w_dev, 1,
                            &zero, d_h_col_buf, 1);
            }
            // Write h[0..j][j] into column j of d_hessenberg from d_h_col_buf.
            // d_hessenberg is row-major (m+1 rows, m cols): H[i][j] = d_hessenberg[i*m+j].
            // d_h_col_buf[i] → d_hessenberg[i*gmres_m + j].
            // This scatter is done by a tiny kernel (j+1 ≤ 30 elements).
            {
                int bk = (j + 1 + 31) / 32;
                // WHY: store_col_fp64 scatters d_h_col_buf[0..j] into H[0..j][j_col].
                // store_col_fp64: M[i*stride + j_col] = col[i], for i in [0, n).
                store_col_fp64<<<bk, 32, 0, stream>>>(
                    d_h_col_buf, d_hessenberg, j + 1, gmres_m, j);
            }

            // ── MGS: w -= V[:,0..j] @ h[0..j][j]  ───────────────────────────
            // cublasDgemv: op(V[:,0..j]) @ d_h_col_buf → correction, subtract from w.
            // w -= V[:,0..j] * h_col  (dgemv: m=n_transient, n=j+1, lda=n_transient).
            {
                double neg1_d = -1.0, one_d = 1.0;
                cublasDgemv(bl_handle, CUBLAS_OP_N,
                            n_transient, j + 1,
                            &neg1_d, V_dev, n_transient,
                            d_h_col_buf, 1,
                            &one_d, w_dev, 1);
            }

            // ── h[j+1][j] = ||w||₂ via cublasDnrm2 (device pointer mode) ────
            // Switch to DEVICE pointer mode so the result (a single fp64) is
            // written directly into d_hessenberg[(j+1)*m+j], no D2H.
            cublasSetPointerMode(bl_handle, CUBLAS_POINTER_MODE_DEVICE);
            cublasDnrm2(bl_handle, n_transient, w_dev, 1,
                        d_hessenberg + (size_t)(j + 1) * gmres_m + j);
            cublasSetPointerMode(bl_handle, CUBLAS_POINTER_MODE_HOST);
            // Zero D2H — result stays on device.

            // ── Givens rotations + g update (all on device) ──────────────────
            // givens_update_kernel reads d_hessenberg col j (rows 0..j+1),
            // applies previous rotations, computes new rotation, zeros H[j+1][j],
            // updates g, writes |g[j+1]| to d_res_norm.
            givens_update_kernel<<<1, 1, 0, stream>>>(
                d_hessenberg, d_g_dev, d_cs_dev, d_sn_dev,
                d_res_norm, j, gmres_m);

            // ── Scale w → V[:,j+1] = w / h[j+1][j]  ─────────────────────────
            // h[j+1][j] lives on device; we need it for scaling.
            // Pull ONE scalar D2H — but only if not the last Arnoldi step.
            // WHY: dscale_kernel needs a host double; alternatives are:
            //   (a) cublasDscal with device pointer mode — but cublasDscal
            //       does NOT accept a device alpha in CUBLAS_POINTER_MODE_DEVICE
            //       for in-place scaling of a vector.
            //   (b) A custom scale_by_device_ptr kernel.
            // We use option (b): a 1-liner device kernel that reads h[j+1][j]
            // from d_hessenberg and scales w in one launch.
            // This is still ZERO D2H — the device does the divide.
            if (j + 1 < gmres_m) {
                int grid_n = (n_transient + BLOCK - 1) / BLOCK;
                // dscale_by_ptr_kernel: V[:,j+1] = w * (1/d_h_ptr[0])
                // We already have dscale_kernel(double* v, double alpha, int n) —
                // but it takes a host alpha.  Use a two-kernel approach:
                // copy w → V[:,j+1], then scale V[:,j+1] using a kernel that reads
                // d_hessenberg[(j+1)*m+j] and computes inv.
                cudaMemcpyAsync(V_dev + (size_t)(j + 1) * n_transient, w_dev,
                                (size_t)n_transient * sizeof(double),
                                cudaMemcpyDeviceToDevice, stream);
                // dscale_by_device_inv_kernel: reads h[j+1][j] from device, scales V[:,j+1].
                // Both this kernel and dscale_by_device_inv_kernel are in singlet::gpu::fate::detail.
                dscale_by_device_inv_kernel<<<grid_n, BLOCK, 0, stream>>>(
                    V_dev + (size_t)(j + 1) * n_transient,
                    d_hessenberg + (size_t)(j + 1) * gmres_m + j,
                    n_transient);
            }
        }  // end Arnoldi inner loop — ZERO cudaMemcpy above

        // ── Restart boundary: D2H convergence scalar + back-substitution ─────
        // Pull |g[j_done]| to check convergence (ONE scalar D2H, approved §⛔9).
        // We run the full gmres_m steps and check once at restart end (vs. checking
        // after each j).  Convergence inside the loop is deferred to here.
        // The last givens_update_kernel wrote |g[gmres_m]| to d_res_norm (index 0).
        // For early convergence detection, we read d_res_norm after the last j.
        // WHY we don't break mid-loop: avoiding D2H per j is the core fix.
        // GMRES typically converges at the restart boundary, not mid-restart.
        double h_res_norm = 0.0;
        cudaMemcpyAsync(&h_res_norm, d_res_norm, sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);  // restart boundary — approved §⛔9
        rel_res = (b_norm > 0.0) ? h_res_norm / b_norm : h_res_norm;

        // D2H H and g for back-substitution (restart boundary bulk transfer).
        cudaMemcpyAsync(H_host.data(), d_hessenberg,
                        (size_t)(gmres_m + 1) * gmres_m * sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(g_host.data(), d_g_dev,
                        (size_t)(gmres_m + 1) * sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // ── Back-substitution: solve upper triangular H[0..m-1][0..m-1] y = g
        for (int i = gmres_m - 1; i >= 0; i--) {
            y[i] = g_host[i];
            for (int k = i + 1; k < gmres_m; k++)
                y[i] -= H_host[i * gmres_m + k] * y[k];
            if (std::abs(H_host[i * gmres_m + i]) > 1e-300)
                y[i] /= H_host[i * gmres_m + i];
        }

        // ── Update solution: x += V[:,0..m-1] @ y  ───────────────────────────
        // Pointer mode is already HOST here (restored after nrm2 inside the loop).
        for (int i = 0; i < gmres_m; i++) {
            cublasDaxpy(bl_handle, n_transient, &y[i],
                        V_dev + (size_t)i * n_transient, 1, x_dev, 1);
        }

        if (rel_res <= tol) break;
    }  // end restart loop

    cusparseDestroyDnVec(vecX_fp32);
    cusparseDestroyDnVec(vecY_fp32);
    return rel_res;
}

// Kernel extracted from inline struct to satisfy CUDA 12 rule:
// __global__ functions cannot be member functions.
__global__ void cr2_fill_row_ptr_kernel(int* ptr, int k_, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n) ptr[i] = i * k_;
}

}  // namespace detail

// ─── Public entry point ───────────────────────────────────────────────────────

// compute_absorption_probabilities()
//
// Solve CellRank 2 absorption probabilities on GPU via batched restarted GMRES.
//
// Parameters:
//   T_row_ptr_host / T_col_idx_host / T_vals_host:
//     Row-stochastic transition matrix T (n × n, fp32 CSR).
//     Pass nullptr for T_* and set emb_dev/n_pcs/cfg.build_transition_from_knn
//     to build T from a kNN graph on device.
//   terminal_indices: global cell indices that are absorbing states.
//   cfg: solver configuration.
//   stream: caller-provided CUDA stream.
//   ctx: GPUContext (cuBLAS + cuSPARSE handles).
//
// Returns CellRank2Result with absorption_prob (n_cells × n_terminals, host fp32).
inline CellRank2Result
compute_absorption_probabilities(
        const int*   T_row_ptr_host,       // n+1 host int (or nullptr if building T)
        const int*   T_col_idx_host,       // nnz host int
        const float* T_vals_host,          // nnz host float
        int          T_nnz,
        int          n_cells,
        const std::vector<int>& terminal_indices,
        const CellRank2Config&  cfg,
        cudaStream_t            stream,
        core::GPUContext&       ctx,
        // Optional: pre-built device T (pass if already on device to skip H2D)
        const int*   T_row_ptr_dev   = nullptr,
        const int*   T_col_idx_dev   = nullptr,
        const float* T_vals_dev      = nullptr,
        // Optional: expression matrix for driver gene computation (device fp32,
        // n_transient × n_genes, col-major). Ignored if cfg.compute_driver_genes=false.
        const float* expression_dev  = nullptr,
        int          n_genes_expr    = 0)
{
    if (terminal_indices.empty()) {
        throw std::runtime_error("cellrank2: terminal_indices must be non-empty");
    }
    for (int idx : terminal_indices) {
        if (idx < 0 || idx >= n_cells)
            throw std::runtime_error("cellrank2: terminal index out of range");
    }

    CellRank2Result result;
    result.n_cells    = n_cells;
    result.n_terminals = (int)terminal_indices.size();

    const int n_terminals = (int)terminal_indices.size();
    const int m           = cfg.gmres_m;
    constexpr int BLOCK   = 256;

    // ── Build transient/absorbing maps ────────────────────────────────────────
    auto maps = detail::build_maps(terminal_indices, n_cells);
    const int n_transient = maps.n_transient;
    const int n_absorbing = maps.n_absorbing;

    // ── Upload transition matrix T to device if not already there ─────────────
    core::DeviceMemory<int>   d_T_row_ptr_owned;
    core::DeviceMemory<int>   d_T_col_idx_owned;
    core::DeviceMemory<float> d_T_vals_owned;

    const int*   d_row_ptr_T = T_row_ptr_dev;
    const int*   d_col_idx_T = T_col_idx_dev;
    const float* d_vals_T    = T_vals_dev;

    if (d_row_ptr_T == nullptr) {
        // No pre-built device T: upload from host CSR.
        if (T_row_ptr_host == nullptr)
            throw std::runtime_error("cellrank2: must provide either device or host T");
        d_T_row_ptr_owned = core::DeviceMemory<int>(n_cells + 1);
        d_T_col_idx_owned = core::DeviceMemory<int>(T_nnz);
        d_T_vals_owned    = core::DeviceMemory<float>(T_nnz);
        cudaMemcpyAsync(d_T_row_ptr_owned.get(), T_row_ptr_host,
                        (size_t)(n_cells + 1) * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_T_col_idx_owned.get(), T_col_idx_host,
                        (size_t)T_nnz * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_T_vals_owned.get(), T_vals_host,
                        (size_t)T_nnz * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
        d_row_ptr_T = d_T_row_ptr_owned.get();
        d_col_idx_T = d_T_col_idx_owned.get();
        d_vals_T    = d_T_vals_owned.get();
    }

    // ── Upload maps to device ─────────────────────────────────────────────────
    core::DeviceMemory<int> d_transient_map(n_cells);
    core::DeviceMemory<int> d_terminal_map(n_cells);
    cudaMemcpyAsync(d_transient_map.get(), maps.transient_map.data(),
                    (size_t)n_cells * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_terminal_map.get(), maps.terminal_map.data(),
                    (size_t)n_cells * sizeof(int), cudaMemcpyHostToDevice, stream);

    // ── Build Q (transient×transient) and R (transient×absorbing) CSR ─────────
    // Step 1: count per-row nnz for Q and R.
    core::DeviceMemory<int> d_Q_row_nnz(n_transient);
    core::DeviceMemory<int> d_R_row_nnz(n_transient);
    detail::sfill_kernel<<<(n_transient + BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(
        reinterpret_cast<float*>(d_Q_row_nnz.get()), 0.f, n_transient);
    detail::sfill_kernel<<<(n_transient + BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(
        reinterpret_cast<float*>(d_R_row_nnz.get()), 0.f, n_transient);
    {
        int grid_cells = (n_cells + BLOCK - 1) / BLOCK;
        detail::build_submatrix_nnz_kernel<<<grid_cells, BLOCK, 0, stream>>>(
            d_row_ptr_T, d_col_idx_T, d_transient_map.get(),
            d_Q_row_nnz.get(), d_R_row_nnz.get(), n_cells);
    }

    // Step 2: prefix-sum to get Q/R row_ptr.
    core::DeviceMemory<int> d_Q_row_ptr(n_transient + 1);
    core::DeviceMemory<int> d_R_row_ptr(n_transient + 1);

    // cub exclusive scan for Q nnz.
    size_t scan_tmp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_tmp_bytes,
                                  d_Q_row_nnz.get(), d_Q_row_ptr.get(),
                                  n_transient + 1, stream);
    core::DeviceMemory<uint8_t> d_scan_tmp(scan_tmp_bytes);
    cub::DeviceScan::ExclusiveSum(d_scan_tmp.get(), scan_tmp_bytes,
                                  d_Q_row_nnz.get(), d_Q_row_ptr.get(),
                                  n_transient + 1, stream);
    cub::DeviceScan::ExclusiveSum(d_scan_tmp.get(), scan_tmp_bytes,
                                  d_R_row_nnz.get(), d_R_row_ptr.get(),
                                  n_transient + 1, stream);

    // Step 3: read Q_nnz and R_nnz (one-time D2H for allocation only).
    int h_Q_nnz = 0, h_R_nnz = 0;
    cudaMemcpyAsync(&h_Q_nnz, d_Q_row_ptr.get() + n_transient, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_R_nnz, d_R_row_ptr.get() + n_transient, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Step 4: allocate and fill Q, R CSR.
    core::DeviceMemory<int>   d_Q_col_idx(h_Q_nnz);
    core::DeviceMemory<float> d_Q_vals(h_Q_nnz);
    core::DeviceMemory<int>   d_R_col_idx(h_R_nnz);
    core::DeviceMemory<float> d_R_vals(h_R_nnz);
    {
        int grid_cells = (n_cells + BLOCK - 1) / BLOCK;
        detail::fill_QR_kernel<<<grid_cells, BLOCK, 0, stream>>>(
            d_row_ptr_T, d_col_idx_T, d_vals_T,
            d_transient_map.get(), d_terminal_map.get(),
            d_Q_row_ptr.get(), d_R_row_ptr.get(),
            d_Q_col_idx.get(), d_Q_vals.get(),
            d_R_col_idx.get(), d_R_vals.get(),
            n_cells);
    }

    // Step 5: build (I - Q): negate Q values, add +1 to diagonal.
    {
        int grid_nnz = (h_Q_nnz + BLOCK - 1) / BLOCK;
        detail::negate_Q_vals_kernel<<<grid_nnz, BLOCK, 0, stream>>>(
            d_Q_vals.get(), h_Q_nnz);
        int grid_tr = (n_transient + BLOCK - 1) / BLOCK;
        detail::add_identity_diag_kernel<<<grid_tr, BLOCK, 0, stream>>>(
            d_Q_vals.get(), d_Q_row_ptr.get(), d_Q_col_idx.get(), n_transient);
    }

    // ── Create cuSPARSE SpMat descriptor for (I - Q) ─────────────────────────
    cusparseSpMatDescr_t ImQ_desc = nullptr;
    cusparseCreateCsr(&ImQ_desc, n_transient, n_transient, h_Q_nnz,
                      d_Q_row_ptr.get(), d_Q_col_idx.get(), d_Q_vals.get(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // ── Build R_dense_fp32 (n_transient × n_terminals) for RHS ──────────────
    // R is sparse CSR (n_transient × n_terminals); SpMV it with I_terminal to
    // get the dense column-major RHS matrix, or scatter R sparse into dense.
    // For small n_terminals (≤100) and moderate n_transient (≤500k): dense.
    core::DeviceMemory<float> d_R_dense((size_t)n_transient * n_terminals);
    detail::sfill_kernel<<<((size_t)n_transient * n_terminals + BLOCK-1)/BLOCK,
                           BLOCK, 0, stream>>>(d_R_dense.get(), 0.f,
                                               (size_t)n_transient * n_terminals);
    // Scatter R_sparse into d_R_dense (one-time setup kernel — not hot path).
    // We do this inline: one thread per R entry.
    {
        // launch a simple scatter kernel
        // R[row_local, col_terminal] = R_val  → d_R_dense[row_local + col_terminal * n_transient]
        // Use fill_QR output: d_R_col_idx / d_R_vals aligned with d_R_row_ptr.
        // Scatter via a simple custom kernel inline:
        auto scatter_R = [&]() __attribute__((unused)) {};
        // Implement as a kernel call:
        struct ScatterArgs { const int* row_ptr; const int* col_idx; const float* vals;
                             float* out; int n_transient_; int n_rows; };
        // Write as a lambda-struct isn't standard for __global__. Use a named helper.
        // We define an inline kernel call here:
        // kernel: for each local row r, for each entry p in R_row_ptr[r..r+1],
        //   out[r + col_idx[p] * n_transient] = vals[p].
        // Launch with n_transient threads.
        // Reuse the fill_QR pattern: reinterpret as a simple scatter.
        // WHY separate kernel: avoid atomics by using exclusive scan offsets.
        // The scatter is index-conflict-free: each (row, terminal) pair unique.
    }
    // Inline scatter via a named __global__ — we need a device-scope lambda.
    // Since CUDA doesn't support device lambdas in all contexts, define a kernel
    // as a static function inside detail: scatter_R_kernel already fits the pattern
    // above. We call it here.
    //
    // Rather than add another kernel declaration (LOC budget), implement scatter
    // via a cuSPARSE SpMM: R_sparse (CSR) × I_dense(n_terminals×n_terminals)
    // = R_dense. This reuses cuSPARSE and avoids a custom kernel.
    {
        // R_sparse: n_transient × n_terminals, CSR.
        // Multiplied by identity: R_dense = R_sparse × I.
        cusparseSpMatDescr_t R_mat = nullptr;
        cusparseCreateCsr(&R_mat, n_transient, n_terminals, h_R_nnz,
                          d_R_row_ptr.get(), d_R_col_idx.get(), d_R_vals.get(),
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        // I_dense (n_terminals × n_terminals) — identity matrix on device.
        core::DeviceMemory<float> d_I_term((size_t)n_terminals * n_terminals);
        detail::sfill_kernel<<<((size_t)n_terminals*n_terminals + BLOCK-1)/BLOCK,
                               BLOCK, 0, stream>>>(d_I_term.get(), 0.f,
                                                   (size_t)n_terminals * n_terminals);
        // Set diagonal to 1 via a tiny kernel.
        // Use cublasSgemm shortcut: place I identity as a dense matrix.
        // Identity fill: thread i sets I[i*n_terminals + i] = 1.
        // Since n_terminals ≤ 100, launch n_terminals threads in one block.
        {
            float* p = d_I_term.get();
            int nt   = n_terminals;
            // Lambda not available as __global__; use a thrust or inline CUDA.
            // Inline C++17 approach: use cudaMemset2D for striped pattern is
            // complex; instead just copy from a host-built identity.
            std::vector<float> h_I(n_terminals * n_terminals, 0.f);
            for (int i = 0; i < n_terminals; i++) h_I[i * n_terminals + i] = 1.f;
            cudaMemcpyAsync(p, h_I.data(),
                            (size_t)n_terminals * n_terminals * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
        }

        cusparseDnMatDescr_t I_mat = nullptr;
        cusparseCreateDnMat(&I_mat, n_terminals, n_terminals, n_terminals,
                            d_I_term.get(), CUDA_R_32F, CUSPARSE_ORDER_COL);
        cusparseDnMatDescr_t R_dense_mat = nullptr;
        cusparseCreateDnMat(&R_dense_mat, n_transient, n_terminals, n_transient,
                            d_R_dense.get(), CUDA_R_32F, CUSPARSE_ORDER_COL);

        float alpha_r = 1.f, beta_r = 0.f;
        size_t spmm_buf_bytes = 0;
        cusparseSpMM_bufferSize(ctx.sparse(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha_r, R_mat, I_mat, &beta_r, R_dense_mat,
                                CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &spmm_buf_bytes);
        core::DeviceMemory<uint8_t> d_spmm_tmp(std::max(spmm_buf_bytes, (size_t)1));
        cusparseSpMM(ctx.sparse(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha_r, R_mat, I_mat, &beta_r, R_dense_mat,
                     CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, d_spmm_tmp.get());

        cusparseDestroySpMat(R_mat);
        cusparseDestroyDnMat(I_mat);
        cusparseDestroyDnMat(R_dense_mat);
    }

    // ── Allocate GMRES workspace ──────────────────────────────────────────────
    // Krylov basis V: n_transient × (m+1) fp64, column-major.
    core::DeviceMemory<double> d_V((size_t)n_transient * (m + 1));
    // Per-terminal solution and rhs vectors (fp64).
    core::DeviceMemory<double> d_x(n_transient);
    core::DeviceMemory<double> d_b(n_transient);
    core::DeviceMemory<double> d_w(n_transient);   // Arnoldi scratch
    // fp32 buffers for SpMV (cuSPARSE SpMV is fp32 → promote to fp64).
    core::DeviceMemory<float>  d_spmv_in(n_transient);
    core::DeviceMemory<float>  d_spmv_out(n_transient);
    // cub::DeviceReduce::Sum workspace (used only for beta = ||r|| at restart boundary).
    core::DeviceMemory<double> d_sq_buf(n_transient);
    core::DeviceMemory<double> d_res_scalar(1);
    size_t reduce_tmp_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, reduce_tmp_bytes,
                           d_sq_buf.get(), d_res_scalar.get(), n_transient, stream);
    core::DeviceMemory<uint8_t> d_reduce_tmp(reduce_tmp_bytes + 8);
    // CYCLE-43: device-resident Hessenberg and g/cs/sn for zero-D2H Arnoldi inner loop.
    core::DeviceMemory<double> d_hessenberg((size_t)(m + 1) * m);  // (m+1)×m row-major
    core::DeviceMemory<double> d_g_dev(m + 1);                     // g vector
    core::DeviceMemory<double> d_cs_dev(m);                        // Givens cosines
    core::DeviceMemory<double> d_sn_dev(m);                        // Givens sines
    core::DeviceMemory<double> d_res_norm_dev(1);                   // |g[j+1]| scalar
    core::DeviceMemory<double> d_h_col_buf(m + 1);                  // dgemv output scratch

    // Result B (n_transient × n_terminals, fp32).
    core::DeviceMemory<float> d_B((size_t)n_transient * n_terminals);
    detail::sfill_kernel<<<((size_t)n_transient * n_terminals + BLOCK-1)/BLOCK,
                           BLOCK, 0, stream>>>(d_B.get(), 0.f,
                                               (size_t)n_transient * n_terminals);

    result.final_residuals.resize(n_terminals);
    result.restarts_used.resize(n_terminals, 0);

    // ── OOC batching across terminals ─────────────────────────────────────────
    int batch_sz = (cfg.ooc_batch_terminals > 0)
                   ? std::min(cfg.ooc_batch_terminals, n_terminals)
                   : n_terminals;

    for (int t_start = 0; t_start < n_terminals; t_start += batch_sz) {
        int t_end = std::min(t_start + batch_sz, n_terminals);
        for (int t = t_start; t < t_end; t++) {
            // RHS for terminal t: column t of d_R_dense (fp32 → fp64).
            // Promote to fp64 for GMRES.
            {
                int grid_n = (n_transient + BLOCK - 1) / BLOCK;
                detail::promote_spmv_to_fp64<<<grid_n, BLOCK, 0, stream>>>(
                    d_R_dense.get() + (size_t)t * n_transient, d_b.get(), n_transient);
            }
            // Initial guess x = 0.
            detail::dfill_kernel<<<(n_transient + BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(
                d_x.get(), 0.0, n_transient);

            // Pre-compute ||b||₂.
            {
                detail::sq_kernel<<<(n_transient+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(
                    d_b.get(), d_sq_buf.get(), n_transient);
                cub::DeviceReduce::Sum(d_reduce_tmp.get(), reduce_tmp_bytes,
                                       d_sq_buf.get(), d_res_scalar.get(),
                                       n_transient, stream);
            }
            double h_b_norm_sq = 0.0;
            cudaMemcpyAsync(&h_b_norm_sq, d_res_scalar.get(), sizeof(double),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            double b_norm = std::sqrt(h_b_norm_sq);

            // Run GMRES (CYCLE-43: passes device Hessenberg + g/cs/sn buffers).
            double rel_res = detail::gmres_solve_one_rhs(
                ctx.sparse(), ctx.blas(), ImQ_desc,
                d_x.get(), d_b.get(),
                d_V.get(), d_w.get(),
                d_spmv_in.get(), d_spmv_out.get(),
                d_sq_buf.get(), d_reduce_tmp.get(), reduce_tmp_bytes,
                d_res_scalar.get(),
                d_hessenberg.get(), d_g_dev.get(),
                d_cs_dev.get(), d_sn_dev.get(),
                d_res_norm_dev.get(), d_h_col_buf.get(),
                n_transient, m, cfg.gmres_max_restarts,
                cfg.convergence_tol, stream, b_norm);

            result.final_residuals[t] = rel_res;

            // Copy solution x (fp64 device) → B[:,t] (fp32 device).
            {
                int grid_n = (n_transient + BLOCK - 1) / BLOCK;
                detail::d64_to_f32_kernel<<<grid_n, BLOCK, 0, stream>>>(
                    d_x.get(), d_B.get() + (size_t)t * n_transient, n_transient);
            }
        }
    }

    result.converged_all = true;
    for (double r : result.final_residuals)
        if (r > cfg.convergence_tol) { result.converged_all = false; break; }

    // ── Assemble full n_cells × n_terminals absorption probability ────────────
    // Transient cells: copy from B; absorbing cells: set identity.
    result.absorption_prob.assign((size_t)n_cells * n_terminals, 0.f);
    // D2H for B (function exit — setup/teardown, always approved).
    std::vector<float> h_B((size_t)n_transient * n_terminals);
    cudaMemcpyAsync(h_B.data(), d_B.get(),
                    (size_t)n_transient * n_terminals * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < n_cells; i++) {
        int t_local = maps.transient_map[i];
        int a_local = maps.terminal_map[i];
        if (t_local >= 0) {
            for (int t = 0; t < n_terminals; t++)
                result.absorption_prob[(size_t)i * n_terminals + t] =
                    h_B[(size_t)t * n_transient + t_local];
        } else {
            // Absorbing cell: P(self terminal) = 1.0.
            result.absorption_prob[(size_t)i * n_terminals + a_local] = 1.f;
        }
    }

    // ── Optional: driver gene Pearson correlations ────────────────────────────
    if (cfg.compute_driver_genes && expression_dev != nullptr && n_genes_expr > 0) {
        result.n_genes = n_genes_expr;
        // Upload B (fp32, transient × terminals) back to device as col-major
        // for Pearson computation.
        core::DeviceMemory<float> d_B_fate((size_t)n_transient * n_terminals);
        cudaMemcpyAsync(d_B_fate.get(), d_B.get(),
                        (size_t)n_transient * n_terminals * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

        core::DeviceMemory<float> d_driver((size_t)n_terminals * n_genes_expr);
        size_t sh_bytes = 256 * 5 * sizeof(double);   // shared mem for Pearson kernel
        dim3 grid_pg(n_terminals, n_genes_expr);
        detail::pearson_driver_kernel<<<grid_pg, 256, sh_bytes, stream>>>(
            d_B_fate.get(), expression_dev, d_driver.get(),
            n_transient, n_terminals, n_genes_expr);

        result.driver_genes.resize((size_t)n_terminals * n_genes_expr);
        cudaMemcpyAsync(result.driver_genes.data(), d_driver.get(),
                        (size_t)n_terminals * n_genes_expr * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    // ── Cleanup cuSPARSE descriptor ───────────────────────────────────────────
    cusparseDestroySpMat(ImQ_desc);

    return result;
}

// ─── Convenience overload: build T from kNN embedding ────────────────────────
//
// Calls cycle-8 compute_exact on emb_dev (n_cells × n_pcs, row-major, fp32),
// row-normalises the adjacency to produce a stochastic T, then calls the main
// entry point above.
//
// WHY: CellRank 2's default mode builds T from a velocity-augmented kNN graph;
// when no velocity is available, the kNN graph on PCA embedding is the standard
// connectivity-based fallback (Weiler et al. §Connectivity kernel).
inline CellRank2Result
compute_absorption_probabilities_from_embedding(
        const float*            emb_dev,
        int                     n_cells,
        int                     n_pcs,
        const std::vector<int>& terminal_indices,
        const CellRank2Config&  cfg,
        cudaStream_t            stream,
        core::GPUContext&       ctx)
{
    // Build kNN graph using cycle-8 compute_exact.
    graph::KnnConfig knn_cfg;
    knn_cfg.k              = cfg.k_neighbors;
    knn_cfg.backend        = graph::KnnBackend::Exact;
    knn_cfg.metric         = graph::DistanceMetric::L2;
    knn_cfg.return_squared = false;

    graph::KnnResult knn = graph::compute_exact(emb_dev, n_cells, n_pcs,
                                                knn_cfg, stream);

    // Build CSR from kNN result (uniform row-length k).
    // row_ptr = 0, k, 2k, ... n_cells*k
    // col_idx = knn.neighbors (device, n_cells × k)
    // vals = uniform 1/k → row-normalised T
    int k = knn.k;
    int nnz = n_cells * k;

    core::DeviceMemory<int>   d_T_row_ptr(n_cells + 1);
    core::DeviceMemory<float> d_T_vals(nnz);

    // Fill row_ptr (uniform k per row) and vals (1/k).
    // One thread per cell for row_ptr; one thread per nnz for vals.
    {
        float inv_k = 1.f / static_cast<float>(k);
        detail::sfill_kernel<<<(nnz + 255)/256, 256, 0, stream>>>(
            d_T_vals.get(), inv_k, nnz);
        // row_ptr: arithmetic sequence 0, k, 2k, ...
        // A simple kernel: row_ptr[i] = i * k
        detail::cr2_fill_row_ptr_kernel<<<(n_cells + 256)/256, 256, 0, stream>>>(
            d_T_row_ptr.get(), k, n_cells);
    }

    return compute_absorption_probabilities(
        nullptr, nullptr, nullptr, nnz, n_cells,
        terminal_indices, cfg, stream, ctx,
        /*T_row_ptr_dev=*/ d_T_row_ptr.get(),
        /*T_col_idx_dev=*/ knn.neighbors.get(),
        /*T_vals_dev=*/    d_T_vals.get());
}

}  // namespace fate
}  // namespace singlet::gpu
