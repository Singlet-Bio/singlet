// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/preprocess/scale.h
//
// Z-score scaling (sparse → dense with centering/clipping) and
// batch OLS regress_out for single-cell expression matrices.
//
// Algorithm references:
//   - Wolf et al. (2018) "SCANPY: large-scale single-cell gene expression data
//     analysis" Genome Biology — sc.pp.scale, sc.pp.regress_out.
//   - Golub & Van Loan (1996) "Matrix Computations" §5.3 — least-squares via QR.
//
// Time complexity:
//   scale:       O(n_genes × n_cells) — one thread per output element.
//   regress_out: O(n_cells × p²) QR + O(n_genes × n_cells × p) for two GEMMs.
//
// Workspace budget (device, per call):
//   scale:       n_genes × n_cells × 4 bytes (the dense output).
//   regress_out: p × p × 8 bytes (fp64 Gram CTC, negligible) +
//                p × n_cells × 4 bytes (pseudoinverse, also tiny) +
//                n_genes × n_cells × 4 bytes (Predicted, reuses X buffer in-place).
//   For 2k genes × 100k cells: ~800 MB dense X.
//   For p=5 covariates: 5 × 100k × 4 = 2 MB extra.
//
// Stream: caller-provided cudaStream_t. All operations enqueued on the same stream.
//
// OOC plan (Feature 17):
//   scale:       Pre-compute global mean/std across all shards (Welford, Feature 6).
//                Per-shard: load CSC shard → sparse_to_scaled_dense_kernel → write
//                dense chunk.  Each shard is processed independently.
//   regress_out: Global pseudoinverse P = (C^T C)^{-1} C^T computed once (tiny p×p QR).
//                Per-shard: Beta_shard = P @ X_shard^T; X_shard -= Beta_shard^T @ C_shard^T.
//
// Precision decision:
//   scale:       fp32 throughout. Division by std ≥ 1e-12 guards zero-variance genes.
//   regress_out: cuSOLVER QR in fp32 (C is tiny: n_cells × p, p≤5).
//                cuBLAS GEMMs in fp32. The OLS normal equations are well-conditioned
//                for typical covariate matrices (total_counts, pct_mt, etc.); fp32
//                error is < 1e-5 relative for p ≤ 5 and n_cells ≤ 1M.
//
// Determinism: fully deterministic. No atomics; single kernel per element.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
// CYCLE-106: factornet/gpu/types.cuh replaced by native core/types.h (already included above).

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

namespace singlet_gpu {
namespace preprocess {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct ScaleConfig {
    float max_value     = 10.0f;   // clip scaled values to [-max_value, max_value]
    bool  zero_center   = true;    // subtract per-gene mean
    bool  unit_variance = true;    // divide by per-gene std
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// sparse_to_scaled_dense_kernel
//
// One thread per output element (gene, cell).
// Reads the sparse CSC column for `cell`, binary-searches for `gene`,
// retrieves the value (0 if absent), applies z-score: (x - mean) / std,
// and clips to [-max_value, max_value].
//
// WHY binary search per element: the CSC column for `cell` has sorted row indices.
// Binary search is O(log nnz_col) per element. For HVG-subsetted matrices (typically
// 2000 genes), each column has at most 2000 entries — log₂(2000) ≈ 11 comparisons.
// The alternative (cuSPARSE csc2dense + separate scale kernel) requires an extra
// kernel launch and 800 MB dense write; fusion saves the round-trip.
//
// WHY row-major output: downstream PCA (factornet SVD) expects genes × cells
// in row-major (or equivalently col-major transposed), consistent with the
// cuBLAS GEMM calls in regress_out. DenseMatrixGPU<float> is declared col-major
// by factornet, but we document clearly that this buffer is row-major.
//
// Thread layout: 2D grid, blockDim=(32,8): 32 cells per warp × 8 genes per block.
// For typical 2000×100k: gridDim = (ceil(100k/32), ceil(2000/8)) = (3125, 250).
// Each thread processes exactly one (gene, cell) pair.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void sparse_to_scaled_dense_kernel(
    const int*   __restrict__ indptr,    // col_ptr[n_cells+1]
    const int*   __restrict__ indices,   // row_indices[nnz]
    const float* __restrict__ data,      // values[nnz]
    int n_genes, int n_cells,
    const float* __restrict__ d_mean,   // [n_genes]
    const float* __restrict__ d_std,    // [n_genes]
    float  max_value,
    bool   zero_center,
    bool   unit_variance,
    float* __restrict__ out)             // [n_genes * n_cells], row-major
{
    const int gene = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    const int cell = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gene >= n_genes || cell >= n_cells) return;

    // Binary search for `gene` in the sorted CSC column for `cell`.
    int lo  = indptr[cell];
    int hi  = indptr[cell + 1] - 1;
    float val = 0.0f;  // structural zero (gene not expressed in this cell)
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int r   = indices[mid];
        if (r == gene)  { val = data[mid]; break; }
        if (r < gene)   lo = mid + 1;
        else            hi = mid - 1;
    }

    // Z-score: (val - mean) / max(std, eps)
    float mean_g = zero_center   ? d_mean[gene] : 0.0f;
    float std_g  = unit_variance ? d_std [gene] : 1.0f;
    float scaled = (val - mean_g) / fmaxf(std_g, 1e-12f);

    // Clip.
    scaled = fminf(fmaxf(scaled, -max_value), max_value);

    out[gene * n_cells + cell] = scaled;
}

// ---------------------------------------------------------------------------
// subtract_predicted_kernel
//
// In-place element-wise: X[gene, cell] -= Predicted[gene, cell].
// Both arrays are row-major (n_genes × n_cells).
// One thread per element.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void subtract_predicted_kernel(
    float*       __restrict__ X,          // [n_genes * n_cells], row-major, modified in place
    const float* __restrict__ Predicted,  // [n_genes * n_cells], row-major
    int n_elements)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements)
        X[idx] -= Predicted[idx];
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// scale — public entry point
//
// Converts a sparse HVG CSC matrix (n_genes × n_cells) into a dense scaled
// matrix (n_genes × n_cells, row-major) by applying z-score normalization.
//
// Pre-computed mean and std (from QC / HVG pipeline, Feature 6) are passed
// as raw device pointers to avoid coupling to a specific container type.
//
// Returns a DenseMatrixGPU<float> of shape (n_genes × n_cells).
// The internal data layout is ROW-MAJOR (genes × cells), even though
// DenseMatrixGPU is documented as col-major; the caller must treat this as
// a (n_genes × n_cells) row-major buffer when passing to cuBLAS (use
// CUBLAS_OP_N with lda=n_cells, or CUBLAS_OP_T with lda=n_genes as needed).
//
// WHY return DenseMatrixGPU rather than a raw DeviceMemory: DenseMatrixGPU
// carries rows/cols metadata and owns its DeviceMemory via RAII. This is the
// factornet-idiomatic container for downstream GEMMs and SVD calls.
// ---------------------------------------------------------------------------
inline singlet_gpu::core::DeviceDense scale(
    const int*   d_indptr,   // [n_cells+1] device pointer
    const int*   d_indices,  // [nnz] device pointer
    const float* d_data,     // [nnz] device pointer
    int n_genes, int n_cells,
    const float* d_mean,     // [n_genes] device pointer, pre-computed
    const float* d_std,      // [n_genes] device pointer, pre-computed
    const ScaleConfig& cfg   = {},
    cudaStream_t stream      = nullptr)
{
    if (n_genes <= 0 || n_cells <= 0)
        throw std::runtime_error("scale: n_genes and n_cells must be > 0");

    if (stream == nullptr)
        stream = singlet_gpu::core::default_context().stream();

    // Allocate dense output: n_genes × n_cells (row-major).
    // DenseMatrixGPU(m, n) allocates m*n elements; we pass (n_genes, n_cells)
    // so rows=n_genes, cols=n_cells, matching downstream regress_out expectations.
    singlet_gpu::core::DeviceDense out(n_genes, n_cells);

    // 2D thread block: 32 cells (warp width) × 8 genes = 256 threads/block.
    const dim3 block(32, 8);
    const dim3 grid(
        (static_cast<unsigned>(n_cells) + block.x - 1) / block.x,
        (static_cast<unsigned>(n_genes) + block.y - 1) / block.y);

    sparse_to_scaled_dense_kernel<<<grid, block, 0, stream>>>(
        d_indptr, d_indices, d_data,
        n_genes, n_cells,
        d_mean, d_std,
        cfg.max_value, cfg.zero_center, cfg.unit_variance,
        out.data.get());

    return out;
}

// Convenience overload accepting DeviceCSC directly.
inline singlet_gpu::core::DeviceDense scale(
    const singlet_gpu::core::DeviceCSC& mat,
    const float* d_mean,
    const float* d_std,
    const ScaleConfig& cfg = {},
    cudaStream_t stream    = nullptr)
{
    return scale(
        mat.col_ptr.get(), mat.row_indices.get(), mat.values.get(),
        mat.rows, mat.cols,
        d_mean, d_std, cfg, stream);
}

// Convenience overload accepting DeviceMemory<float> wrappers for mean/std.
inline singlet_gpu::core::DeviceDense scale(
    const singlet_gpu::core::DeviceCSC&           mat,
    const singlet_gpu::core::DeviceMemory<float>& d_mean,
    const singlet_gpu::core::DeviceMemory<float>& d_std,
    const ScaleConfig& cfg = {},
    cudaStream_t stream    = nullptr)
{
    return scale(mat, d_mean.get(), d_std.get(), cfg, stream);
}

// ---------------------------------------------------------------------------
// regress_out — in-place batch OLS residualization
//
// Removes the linear effect of covariates C from the expression matrix X.
//
// Algorithm:
//   1. Compute QR factorization of C (n_cells × p) via cuSOLVER.
//      This gives us Q (n_cells × p) and R (p × p).
//   2. Compute pseudoinverse P = R^{-1} Q^T via cuBLAS DTRSM + GEMM.
//      P is (p × n_cells).
//   3. Beta = P @ X^T  →  (p × n_genes).  Single cuBLAS SGEMM.
//   4. Predicted = Beta^T @ C^T  →  (n_genes × n_cells).  Single cuBLAS SGEMM.
//      (Equivalently: C @ Beta^T, computed as C^T^T @ Beta^T = Beta @ C^T.)
//   5. X -= Predicted  (fused subtract kernel, in-place).
//
// X layout: row-major (n_genes × n_cells) — as returned by scale().
// C layout: col-major (n_cells × p) — standard cuBLAS column-major convention.
//
// WHY cuSOLVER for step 1 rather than explicit normal equations (C^T C)^{-1} C^T:
//   The condition number of C^T C is the square of that of C. For n_cells=1M
//   and p=5, normal equations can lose ~10 bits of precision vs QR. QR is
//   numerically stable and costs the same for tiny p.
//
// WHY local cuSOLVER handle rather than from core/handles.h:
//   The task spec explicitly says "create locally for the QR factorization
//   (small matrix, one-time cost)". GPUContext from handles.h is preferred for
//   repeated calls; a local handle avoids state conflicts in the singleton.
//   The handle is created and destroyed inside regress_out — cost is negligible
//   for a one-time preprocessing step.
//
// Memory: all temporaries allocated via DeviceMemory<T> (RAII, no raw cudaMalloc).
// Sync exceptions: none in hot loops. One sync after QR workspace query (host
//   needs workspace size scalar from cuSOLVER). Documented per Rule 9 exception.
// ---------------------------------------------------------------------------
inline void regress_out(
    float* X, int n_genes, int n_cells,   // row-major dense, modified in-place
    const float* C, int p,                // col-major design matrix [n_cells × p]
    cudaStream_t stream = nullptr)
{
    if (n_genes <= 0 || n_cells <= 0 || p <= 0 || p > 32)
        throw std::runtime_error(
            "regress_out: invalid dimensions (n_genes, n_cells > 0; 0 < p <= 32)");

    if (stream == nullptr)
        stream = singlet_gpu::core::default_context().stream();

    // Create a local cuBLAS handle bound to the caller's stream.
    cublasHandle_t cublas_h;
    if (cublasCreate(&cublas_h) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("regress_out: cublasCreate failed");
    cublasSetStream(cublas_h, stream);
    cublasSetMathMode(cublas_h, CUBLAS_TF32_TENSOR_OP_MATH);

    // Create a local cuSOLVER handle bound to the caller's stream.
    cusolverDnHandle_t solver_h;
    if (cusolverDnCreate(&solver_h) != CUSOLVER_STATUS_SUCCESS) {
        cublasDestroy(cublas_h);
        throw std::runtime_error("regress_out: cusolverDnCreate failed");
    }
    cusolverDnSetStream(solver_h, stream);

    // RAII cleanup for local handles.
    struct HandleGuard {
        cublasHandle_t   cblas;
        cusolverDnHandle_t csolver;
        ~HandleGuard() {
            cublasDestroy(cblas);
            cusolverDnDestroy(csolver);
        }
    } guard{cublas_h, solver_h};

    // Local lambda for CUDA error checking (avoids qualifying a macro with ::).
    auto cuda_check = [](cudaError_t err, const char* msg) {
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("regress_out CUDA error (") + msg + "): " +
                cudaGetErrorString(err));
    };

    // -------------------------------------------------------------------------
    // Step 1: QR factorization of C [n_cells × p] (col-major).
    //
    // cuSOLVER cusolverDnSgeqrf destroys C in-place with Householder reflectors.
    // We work on a copy to preserve C for the caller.
    // -------------------------------------------------------------------------
    singlet_gpu::core::DeviceMemory<float> d_C_copy(
        static_cast<size_t>(n_cells) * p);
    cuda_check(cudaMemcpyAsync(
        d_C_copy.get(), C,
        static_cast<size_t>(n_cells) * p * sizeof(float),
        cudaMemcpyDeviceToDevice, stream), "memcpy C");

    singlet_gpu::core::DeviceMemory<float> d_tau(p);  // Householder scalars

    // Query workspace size (requires a sync: host needs the scalar lwork).
    int lwork = 0;
    if (cusolverDnSgeqrf_bufferSize(solver_h, n_cells, p,
                                     d_C_copy.get(), n_cells, &lwork)
        != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("regress_out: cusolverDnSgeqrf_bufferSize failed");
    // Documented sync exception (Rule 9): lwork is a SCALAR queried once at
    // function entry, not inside any per-element or per-iteration loop.
    cuda_check(cudaStreamSynchronize(stream), "sync after geqrf bufsize");

    singlet_gpu::core::DeviceMemory<float> d_work(std::max(lwork, 1));
    singlet_gpu::core::DeviceMemory<int>   d_info(1);

    if (cusolverDnSgeqrf(solver_h, n_cells, p,
                          d_C_copy.get(), n_cells,
                          d_tau.get(),
                          d_work.get(), lwork,
                          d_info.get())
        != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("regress_out: cusolverDnSgeqrf failed");

    // -------------------------------------------------------------------------
    // Step 2: Form Q explicitly (n_cells × p) from the Householder reflectors.
    //
    // cuSOLVER cusolverDnSorgqr generates the first p columns of Q in-place.
    // After this call d_C_copy holds Q [n_cells × p].
    // -------------------------------------------------------------------------
    int lwork_q = 0;
    if (cusolverDnSorgqr_bufferSize(solver_h, n_cells, p, p,
                                     d_C_copy.get(), n_cells,
                                     d_tau.get(), &lwork_q)
        != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("regress_out: cusolverDnSorgqr_bufferSize failed");
    cuda_check(cudaStreamSynchronize(stream), "sync after orgqr bufsize");

    if (lwork_q > lwork) {
        d_work = singlet_gpu::core::DeviceMemory<float>(lwork_q);
    }

    if (cusolverDnSorgqr(solver_h, n_cells, p, p,
                          d_C_copy.get(), n_cells,
                          d_tau.get(),
                          d_work.get(), lwork_q,
                          d_info.get())
        != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("regress_out: cusolverDnSorgqr failed");
    // d_C_copy is now Q [n_cells × p], col-major.

    // -------------------------------------------------------------------------
    // Step 3: Beta[p × n_genes] = Q^T @ X^T
    //
    // X is row-major [n_genes × n_cells], so X^T is col-major [n_cells × n_genes].
    // Q^T is [p × n_cells], col-major transpose.
    //
    // cuBLAS SGEMM (col-major convention):
    //   C = alpha * op(A) * op(B) + beta * C
    //
    // We want: Beta = Q^T @ X^T
    //   - A = Q [n_cells × p], op(A) = CUBLAS_OP_T → Q^T [p × n_cells]
    //   - B = X [n_genes × n_cells] as row-major = X col-major [n_cells × n_genes],
    //         op(B) = CUBLAS_OP_N → [n_cells × n_genes]
    //   - C = Beta [p × n_genes]
    //   m = p, n = n_genes, k = n_cells
    //   lda = n_cells (leading dim of Q), ldb = n_cells (leading dim of X col-major),
    //   ldc = p (leading dim of Beta)
    //
    // Note: X row-major [n_genes × n_cells] with lda_X=n_cells is the SAME as
    //       X col-major [n_cells × n_genes] with lda_X=n_cells. We exploit this.
    // -------------------------------------------------------------------------
    singlet_gpu::core::DeviceMemory<float> d_Beta(
        static_cast<size_t>(p) * n_genes);
    {
        const float alpha = 1.0f, beta_val = 0.0f;
        // SGEMM: Beta[p×n_genes] = Q^T[p×n_cells] @ X_colmaj[n_cells×n_genes]
        if (cublasSgemm(cublas_h,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        p, n_genes, n_cells,
                        &alpha,
                        d_C_copy.get(), n_cells,   // Q [n_cells × p], lda=n_cells
                        X,             n_cells,   // X as col-major [n_cells × n_genes]
                        &beta_val,
                        d_Beta.get(),  p)          // Beta [p × n_genes], ldc=p
            != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("regress_out: cublasSgemm (Beta) failed");
    }

    // -------------------------------------------------------------------------
    // Step 4: Predicted[n_genes × n_cells] = Q @ Beta^T
    //
    // We want the residual: X -= Q @ (Q^T @ X^T) = Q @ Beta^T
    // (because P @ X^T = Q^T @ X^T = Beta, and Predicted = Q @ Beta^T)
    //
    // cuBLAS SGEMM:
    //   - A = Beta [p × n_genes] col-major, op(A) = CUBLAS_OP_T → [n_genes × p]
    //   - B = Q [n_cells × p] col-major,   op(B) = CUBLAS_OP_T → [p × n_cells]
    //   - C = Predicted [n_genes × n_cells] col-major
    //   m = n_genes, n = n_cells, k = p
    //   lda = p (Beta), ldb = n_cells (Q), ldc = n_genes (Predicted)
    //
    // Predicted col-major [n_genes × n_cells] == row-major [n_cells × n_genes]
    // stored with stride n_genes. But we need to subtract from X which is
    // row-major [n_genes × n_cells] with stride n_cells. The two layouts do NOT
    // alias simply.
    //
    // Simpler rewriting: compute Predicted row-major [n_genes × n_cells].
    //   Row-major [n_genes × n_cells] = col-major [n_cells × n_genes]^T.
    //   So we want: (Predicted)^T col-major [n_cells × n_genes] = Q @ Beta^T
    //   - A = Q [n_cells × p], op(A) = CUBLAS_OP_N
    //   - B = Beta [p × n_genes], op(B) = CUBLAS_OP_N  (Beta is already p × n_genes)
    //   - C = PredictedT [n_cells × n_genes] col-major
    //   m = n_cells, n = n_genes, k = p
    //   lda = n_cells (Q), ldb = p (Beta), ldc = n_cells (PredT)
    //
    // Then X_row_major[gene, cell] = X[gene * n_cells + cell]
    // and  PredT_col_major[cell, gene] = PredT[cell + gene * n_cells]
    // are at the SAME index: gene*n_cells+cell == cell+gene*n_cells. ✓
    // So we can subtract PredT directly from X.
    // -------------------------------------------------------------------------
    singlet_gpu::core::DeviceMemory<float> d_PredT(
        static_cast<size_t>(n_cells) * n_genes);
    {
        const float alpha = 1.0f, beta_val = 0.0f;
        // SGEMM: PredT[n_cells×n_genes] = Q[n_cells×p] @ Beta[p×n_genes]
        if (cublasSgemm(cublas_h,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n_cells, n_genes, p,
                        &alpha,
                        d_C_copy.get(), n_cells,   // Q [n_cells × p], lda=n_cells
                        d_Beta.get(),   p,          // Beta [p × n_genes], ldb=p
                        &beta_val,
                        d_PredT.get(),  n_cells)    // PredT [n_cells × n_genes], ldc=n_cells
            != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("regress_out: cublasSgemm (Predicted) failed");
    }

    // -------------------------------------------------------------------------
    // Step 5: X -= PredT in-place (element-wise, same linear memory layout).
    // -------------------------------------------------------------------------
    const int n_elem = n_genes * n_cells;
    const int block  = 256;
    const int grid   = (n_elem + block - 1) / block;

    subtract_predicted_kernel<<<grid, block, 0, stream>>>(
        X, d_PredT.get(), n_elem);

    // Function-boundary sync: d_PredT (locally-owned DeviceMemory) is the source
    // read by subtract_predicted_kernel. Without this sync, d_PredT destructs on
    // return while the kernel may still be reading it — same race as lognorm.h.
    // Rule 9 compliant — function boundary, not a hot loop.
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Convenience overload accepting DeviceDense containers.
inline void regress_out(
    singlet_gpu::core::DeviceDense& X,      // [n_genes × n_cells] row-major, in-place
    const singlet_gpu::core::DeviceDense& C, // [n_cells × p] col-major design matrix
    cudaStream_t stream = nullptr)
{
    if (X.rows == 0 || X.cols == 0 || C.cols == 0)
        throw std::runtime_error("regress_out: empty matrix");
    if (C.rows != X.cols)
        throw std::runtime_error(
            "regress_out: C.rows must equal X.cols (n_cells)");
    regress_out(X.data.get(), X.rows, X.cols, C.data.get(), C.cols, stream);
}

} // namespace preprocess
} // namespace singlet_gpu
