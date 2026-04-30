// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (no GPU implementation exists in the literature)
//
// singlet-gpu/preprocess/magic.h
//
// MAGIC imputation — Markov Affinity-based Graph Imputation of Cells.
//
// Algorithm reference:
//   van Dijk D, Sharma R, Nainys J, et al. (2018) "Recovering Gene Interactions
//   from Single-Cell Data Using Data Diffusion." Cell 174:716-729.
//   https://doi.org/10.1016/j.cell.2018.05.061
//
// Overview:
//   Given a log-normalised sparse expression matrix X (m genes × n cells, CSC)
//   and a cell-cell affinity graph W (n × n, CSR with Jaccard weights from
//   singlet_gpu::graph::compute_snn), this kernel imputes missing expression by
//   diffusing signals through the graph for t steps.
//
//   Step 1. Build Markov transition matrix M = D⁻¹ W (row-stochastic).
//           D = diag(rowSum(W)).
//   Step 2. Initialise Y₀ = Xᵀ (dense n × m, col-major):
//           Y₀[cell, gene] = X[gene, cell].
//   Step 3. Iterate t times: Y_{s} = M · Y_{s-1}  via cuSPARSE SpMM.
//   Step 4. Return Y_t (dense n × m, col-major).
//
// V0 limitation — α-decay kernel not applied:
//   van Dijk et al. optionally replace W_ij by exp(-(d_ij/σ_i)^α) using the
//   k-th NN distance σ_i and α=15.  In v0 the input graph W is used directly
//   (Jaccard weights from SnnResult are already similarity-based and bounded
//   in [0,1], which provides reasonable smoothing without decay).  Full α-decay
//   is deferred to v1 (MagicConfig::use_alpha_decay; set to false here to
//   document the intent without misrepresenting the implementation).
//
// Time complexity:
//   Row-normalisation: O(nnz_graph)
//   Each SpMM step: O(nnz_graph × m_genes) — dominant term.
//   For 100k cells, k=15 SNN graph (≈1.5M edges), 20k genes, t=3: ~90G FLOPs.
//   At 10 TFLOP/s (A100 fp32 SpMM): ≈9 ms.
//
// Memory:
//   Two ping-pong dense buffers, each n × m × 4 bytes.
//   For n=100k, m=20k: 2 × 8 GB = 16 GB.  A memory guard at entry rejects
//   calls where n×m×4 > 50% of free device memory with an actionable error.
//   Recommended usage: subset to HVGs (m ≤ 3000) before calling.
//
// Workspace budget above the two ping-pong buffers:
//   M_values[nnz_graph]  fp32 Markov weights  4 × nnz bytes
//   row_sum[n]           fp32 row sums         4n bytes
//   cuSPARSE SpMM workspace: queried at runtime (≤few MB for typical graphs)
//
// Determinism (Rule 18):
//   Row-sum kernel: warp-shuffle reduction per row — no atomics → deterministic.
//   SpMM (cuSPARSE CSRMM): deterministic at fixed sm architecture and occupancy
//   for a fixed CSR sparsity pattern.  The deterministic flag is kept for API
//   symmetry with other kernels but is a no-op here.
//
// Stream: caller-provided cudaStream_t; nullptr resolves to default context stream.
//   All kernels and cuSPARSE calls run on the same stream in order.
//
// Precision: fp32 throughout. SpMM accumulation is fp32 with no mixed precision.
//   For very long diffusion chains (t > 10) numerical drift can accumulate;
//   the default t=3 is well within fp32 dynamic range.
//
// CYCLE-124: initial implementation.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/graph/snn.h>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu {
namespace preprocess {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct MagicConfig {
    int   t               = 3;      // diffusion steps (van Dijk 2018 default)
    bool  use_alpha_decay = false;  // V0: assumes input W already decayed;
                                    // full α-decay (α=15, σ_i=k-th NN dist)
                                    // deferred to v1.  Ignored here.
    float epsilon         = 1e-9f;  // guard for D⁻¹ when row_sum = 0
    bool  deterministic   = false;  // SpMM is deterministic; flag for API symmetry
};

// Imputed expression matrix: cells × genes, dense, col-major.
struct MagicResult {
    core::DeviceMemory<float> imputed;  // n × m, col-major
    int n_cells;
    int m_genes;
    int t_used;
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels (anonymous namespace — not part of the public API)
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// row_sum_graph_kernel — per-row sums of the CSR graph weights.
//
// One block per row. Threads stride over the nnz in the row and warp-shuffle
// reduce into a single sum written to row_sum[row].
//
// WHY one block per row: rows are variable-length (up to k entries for a kNN-
// derived SNN graph).  A single block avoids inter-block synchronization.
// WHY warp-shuffle, no atomics: deterministic per Rule 18; the extra shared-
// memory bookkeeping of multi-warp reduction is acceptable because we launch
// at most n blocks (≤1M for practical inputs).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void row_sum_graph_kernel(
    const float* __restrict__ weights,      // nnz graph weights
    const int*   __restrict__ row_offsets,  // CSR row_offsets [n+1]
    float*       __restrict__ row_sum,      // output: row sums [n]
    int n)
{
    const int row   = blockIdx.x;
    const int lane  = threadIdx.x & 31;
    const int wid   = threadIdx.x / 32;
    const int nwarp = blockDim.x / 32;

    if (row >= n) return;

    const int start = row_offsets[row];
    const int end   = row_offsets[row + 1];

    float acc = 0.f;
    for (int e = start + threadIdx.x; e < end; e += blockDim.x) {
        acc += weights[e];
    }

    // Warp-level reduction.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }

    // Cross-warp reduction via shared memory.
    __shared__ float smem[8];  // up to 256/32=8 warps
    if (lane == 0) smem[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        float val = (lane < nwarp) ? smem[lane] : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, off);
        }
        if (lane == 0) row_sum[row] = val;
    }
}

// ---------------------------------------------------------------------------
// normalize_weights_kernel — compute M_values = W_ij / row_sum[i].
//
// One thread per nnz entry.  The row of entry e is found by binary-searching
// row_offsets[0..n].  This is O(log n) per entry — acceptable since n ≤ 1M.
//
// WHY binary search: avoids a second n-array allocation for "row of e" and
// keeps the kernel simple.  For typical SNN graphs (nnz ≈ n*k, k≤30) the
// log-n overhead is negligible versus the fp32 divide.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void normalize_weights_kernel(
    const float* __restrict__ weights,      // input W edge weights [nnz]
    const int*   __restrict__ row_offsets,  // CSR row_offsets [n+1]
    const float* __restrict__ row_sum,      // row sums [n]
    float*       __restrict__ M_values,     // output Markov weights [nnz]
    float        epsilon,
    int          nnz,
    int          n)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nnz) return;

    // Binary-search row_offsets to find which row contains entry e.
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (row_offsets[mid + 1] <= e) lo = mid + 1;
        else                           hi = mid;
    }
    const int row = lo;
    const float rs = row_sum[row];
    M_values[e] = weights[e] / (rs > epsilon ? rs : epsilon);
}

// ---------------------------------------------------------------------------
// csc_transpose_to_dense_kernel — scatter CSC X (m × n) to dense Y (n × m).
//
// Y is col-major (n × m): Y[cell + gene*n] = X_value.
// For CSC input stored as (col_ptr, row_indices, values):
//   - col = j (cell index in X = column in CSC)
//   - row = row_indices[e] (gene index)
//   - Y[j + row*n] = values[e]
//
// One thread per stored entry — simple scatter, no conflicts (each (j, row)
// pair in CSC is unique by definition).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void csc_transpose_to_dense_kernel(
    const float*   __restrict__ values,      // CSC values [nnz]
    const int32_t* __restrict__ row_indices, // CSC row_indices [nnz]
    const int32_t* __restrict__ col_ptr,     // CSC col_ptr [n+1]
    float*         __restrict__ Y,           // dense n × m col-major [n*m]
    int m,   // n_genes  (= CSC rows)
    int n,   // n_cells  (= CSC cols)
    int nnz)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nnz) return;
    // Find column j by binary-searching col_ptr.
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (col_ptr[mid + 1] <= e) lo = mid + 1;
        else                       hi = mid;
    }
    const int j    = lo;       // cell
    const int gene = row_indices[e];
    // Y is col-major n×m: gene is the "column", cell is the "row".
    Y[j + (size_t)gene * n] = values[e];
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// magic_impute — public entry point
//
// Inputs:
//   X      — sparse expression matrix m_genes × n_cells (CSC, log-normalised)
//   graph  — SNN cell-cell graph (n × n CSR, from singlet_gpu::graph::compute_snn)
//   cfg    — MagicConfig (defaults: t=3, epsilon=1e-9)
//   stream — optional cudaStream_t (nullptr → default context)
//
// Returns MagicResult with device-resident dense imputed matrix (n × m, col-major).
// Caller must synchronize the stream before reading imputed.
// ---------------------------------------------------------------------------
inline MagicResult magic_impute(
    const io::PzDeviceMatrix& X,
    const graph::SnnResult&   graph,
    const MagicConfig&        cfg    = {},
    cudaStream_t              stream = nullptr)
{
    if (stream == nullptr) {
        stream = singlet_gpu::core::default_context().stream();
    }

    const int m   = X.mat.rows;   // genes
    const int n   = X.mat.cols;   // cells
    const int nnz = X.mat.nnz;

    // --- Validate dimensions ---
    if (m <= 0 || n <= 0) {
        throw std::runtime_error("magic_impute: invalid expression matrix dimensions");
    }
    if (graph.n != n) {
        throw std::runtime_error(
            "magic_impute: graph.n (" + std::to_string(graph.n) +
            ") != X.mat.cols (" + std::to_string(n) + ") — cell count mismatch");
    }

    // --- Memory guard ---
    // Dense output = n × m × 4 bytes (two ping-pong buffers).
    // Reject if 2 × n × m × 4 > 50% of free device memory.
    {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        const size_t needed = static_cast<size_t>(n) * static_cast<size_t>(m) * 4ULL * 2ULL;
        if (needed > free_mem / 2) {
            const double needed_gb = static_cast<double>(needed) / (1024.0 * 1024.0 * 1024.0);
            const double free_gb   = static_cast<double>(free_mem) / (1024.0 * 1024.0 * 1024.0);
            throw std::runtime_error(
                "magic_impute: dense output would require " +
                std::to_string(needed_gb).substr(0, 5) + " GB (two ping-pong buffers, n=" +
                std::to_string(n) + " cells × m=" + std::to_string(m) + " genes) but only " +
                std::to_string(free_gb).substr(0, 5) + " GB free. "
                "Subset to HVGs (e.g. top 2000–3000 genes) before calling magic_impute.");
        }
    }

    constexpr int THREADS = 256;

    // -------------------------------------------------------------------------
    // Step 1: Build Markov transition matrix M from graph.
    //   row_sum[i] = Σ_j W_ij   (over all edges in row i)
    //   M_values[e] = W[e] / row_sum[row(e)]
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_row_sum(n);
    cudaMemsetAsync(d_row_sum.get(), 0, n * sizeof(float), stream);

    // One block per row; 256 threads/block → 8 warps.
    {
        row_sum_graph_kernel<<<n, THREADS, 0, stream>>>(
            graph.weights.get(),
            graph.row_offsets.get(),
            d_row_sum.get(),
            n);
    }

    core::DeviceMemory<float> d_M_values(graph.nnz > 0 ? graph.nnz : 1);
    if (graph.nnz > 0) {
        const int grid_nnz = (graph.nnz + THREADS - 1) / THREADS;
        normalize_weights_kernel<<<grid_nnz, THREADS, 0, stream>>>(
            graph.weights.get(),
            graph.row_offsets.get(),
            d_row_sum.get(),
            d_M_values.get(),
            cfg.epsilon,
            graph.nnz,
            n);
    }

    // -------------------------------------------------------------------------
    // Step 2: Initialise Y₀ = Xᵀ (dense n × m, col-major).
    //   Y₀[cell j, gene i] = X[gene i, cell j]
    //   In col-major storage: Y₀[j + i*n]
    //   X is CSC (m × n): each stored entry (row_idx=gene, col=cell, val).
    // -------------------------------------------------------------------------
    const size_t dense_sz = static_cast<size_t>(n) * static_cast<size_t>(m);

    // Ping-pong buffers; Y_a holds the current diffusion state.
    core::DeviceMemory<float> Y_a(dense_sz);
    core::DeviceMemory<float> Y_b(dense_sz);

    // Zero-initialize Y_a (covers implicit zeros from sparse X).
    cudaMemsetAsync(Y_a.get(), 0, dense_sz * sizeof(float), stream);

    if (nnz > 0) {
        const int grid_x = (nnz + THREADS - 1) / THREADS;
        csc_transpose_to_dense_kernel<<<grid_x, THREADS, 0, stream>>>(
            X.mat.values.get(),
            X.mat.row_indices.get(),
            X.mat.col_ptr.get(),
            Y_a.get(),
            m, n, nnz);
    }

    // Special case: t=0 → return transposed input unchanged.
    if (cfg.t == 0) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MagicResult res;
        res.imputed = std::move(Y_a);
        res.n_cells = n;
        res.m_genes = m;
        res.t_used  = 0;
        return res;
    }

    // -------------------------------------------------------------------------
    // Step 3: cuSPARSE SpMM — iterate t times Y_{s} = M · Y_{s-1}
    //   M is n × n CSR; Y is n × m dense col-major.
    //   cusparseSpMM computes: Y_out = α·M·Y_in + β·Y_out
    //   We use α=1, β=0.
    // -------------------------------------------------------------------------
    cusparseHandle_t sp_handle;
    CUSPARSE_CHECK(cusparseCreate(&sp_handle));
    cusparseSetStream(sp_handle, stream);

    // Build cuSPARSE descriptor for M (CSR n × n).
    cusparseSpMatDescr_t spM;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &spM,
        static_cast<int64_t>(n),
        static_cast<int64_t>(n),
        static_cast<int64_t>(graph.nnz),
        graph.row_offsets.get(),
        graph.col_indices.get(),
        d_M_values.get(),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));

    // Dense matrix descriptors for ping-pong.
    // Y is n × m col-major: leading dimension = n.
    cusparseDnMatDescr_t dnA, dnB;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnA,
        static_cast<int64_t>(n),
        static_cast<int64_t>(m),
        static_cast<int64_t>(n),  // leading dim (col-major → ld = n_rows)
        Y_a.get(),
        CUDA_R_32F,
        CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnB,
        static_cast<int64_t>(n),
        static_cast<int64_t>(m),
        static_cast<int64_t>(n),
        Y_b.get(),
        CUDA_R_32F,
        CUSPARSE_ORDER_COL));

    // Query workspace size (using Y_a as input, Y_b as output for the query).
    size_t ws_bytes = 0;
    const float alpha = 1.f, beta = 0.f;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        sp_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, spM, dnA, &beta, dnB,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &ws_bytes));

    core::DeviceMemory<uint8_t> d_workspace(ws_bytes > 0 ? ws_bytes : 1);

    // Diffusion loop: t steps.
    // Ping-pong: even steps write to Y_b; odd steps write to Y_a.
    // We alternate descriptors to avoid re-creating them per step.
    bool a_is_input = true;
    for (int step = 0; step < cfg.t; ++step) {
        cusparseDnMatDescr_t& in_desc  = a_is_input ? dnA : dnB;
        cusparseDnMatDescr_t& out_desc = a_is_input ? dnB : dnA;

        CUSPARSE_CHECK(cusparseSpMM(
            sp_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, spM, in_desc, &beta, out_desc,
            CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            d_workspace.get()));

        a_is_input = !a_is_input;
    }

    // Destroy cuSPARSE objects.
    cusparseDestroyDnMat(dnA);
    cusparseDestroyDnMat(dnB);
    cusparseDestroySpMat(spM);
    cusparseDestroy(sp_handle);

    // Function-boundary sync (Rule 9): guarantees all kernels complete before
    // workspace and descriptor resources go out of scope.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // The final result lives in Y_b if t is odd (a_is_input=true → last write to Y_b),
    // or Y_a if t is even. a_is_input tracks which buffer was used as input last,
    // meaning the output is in the OTHER buffer.
    // After the loop: a_is_input flipped after the last step.
    //   If last step wrote to Y_b: a_is_input = true → output in Y_b.
    //   If last step wrote to Y_a: a_is_input = false → output in Y_a.
    core::DeviceMemory<float> result_buf = a_is_input
        ? std::move(Y_b)
        : std::move(Y_a);

    MagicResult res;
    res.imputed = std::move(result_buf);
    res.n_cells = n;
    res.m_genes = m;
    res.t_used  = cfg.t;
    return res;
}

}  // namespace preprocess
}  // namespace singlet_gpu
