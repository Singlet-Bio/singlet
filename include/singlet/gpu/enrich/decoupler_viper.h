// SPDX-License-Identifier: MIT
// integrates: decoupleR VIPER / aREA (Alvarez et al. 2016)
//
// singlet/gpu/enrich/decoupler_viper.h
//
// VIPER (Virtual Inference of Protein-activity by Enriched Regulon analysis)
// via aREA (analytical Rank-based Enrichment Analysis) on GPU.
//
// Algorithm reference:
//   Alvarez MJ, Shen Y, Giorgi FM, et al. (2016) "Functional characterization
//   of somatic mutations in cancer using network-based inference of protein
//   activity." Nat Genet 48:838-847. https://doi.org/10.1038/ng.3593
//
// Given expression matrix X (m_genes × n_cells, sparse CSC, log1p-normalized)
// and a dense signed weight matrix W (m_genes × n_regulons, col-major fp32):
//
//   Step 1 — Per-cell rank transform to T1:
//     Materialize a dense (m × n) matrix, fill from X (zeros for missing entries).
//     Sort each column ascending with CUB DeviceSegmentedRadixSort → permutation.
//     Assign T1[permutation[r], c] = qnorm((r+1)/(m+1)) for r = 0..m-1.
//     qnorm implemented via CUDA built-in normcdfinvf(p) (fp32 inverse-normal CDF).
//
//   Step 2 — cuBLAS Sgemm: NES = T1^T · W  (n × n_reg, col-major).
//     T1 is m × n col-major; W is m × n_reg col-major.
//     cuBLAS call: C = alpha·A^T·B + beta·C with A=T1 (m×n), B=W (m×n_reg).
//     Produces C = T1^T · W  (n × n_reg).
//
//   Step 3 — Column-scale NES by 1/Σ|W[:,r]|:
//     Per-regulon L1 norm computed by warp-shuffle reduction kernel.
//     One-pass element-wise divide (shared with WSUM pattern).
//
// V0 simplification: NES = ES1 = (T1^T · W) / L1(W).
//   Full two-tail ES2 and correlation blending deferred to v1 (see below).
//
// V1 TODO (not in this file):
//   T2[g, c] = qnorm(0.5 + 0.5·|rank[g,c] - median_rank| / (m+1))
//   ES2[c, r] = Σ_g w[g,r]·T2[g,c] / Σ_g |w[g,r]|
//   NES[c, r] = ES1·sqrt(1 - corr²) + ES2·corr  (corr = corr(T1, T2) per cell)
//   For typical scRNA corr ≈ 0, so NES ≈ ES1 anyway.
//
// Memory guard:
//   T1 is a dense m × n fp32 buffer — m=20k, n=100k → 8 GB.
//   ViperConfig::max_dense_gb (default 32 GB) rejects calls that would exceed it.
//   Recommended usage: subset to HVGs (m ≤ 5000) for large n.
//
// Time complexity:
//   Dense fill:     O(m × n)
//   CUB sort:       O(m × n × log m) — per-column sort via segmented radix sort
//   qnorm assign:   O(m × n)
//   cuBLAS Sgemm:   O(m × n × n_reg)  — dominant at large n_reg
//   Norm + scale:   O(m × n_reg) + O(n × n_reg)
//
// Determinism (Rule 18):
//   Sort-based ranking: deterministic.
//   cuBLAS Sgemm: deterministic at fixed architecture for fixed inputs.
//   Norm kernel: warp-shuffle only — no atomics → bit-exact.
//   Overall: bit-exact deterministic. cfg.deterministic is a no-op (API symmetry).
//
// Memory (Rule 5): all device buffers via core::DeviceMemory<T>. No raw cudaMalloc.
// D2H (Rule 4): no per-inner-loop D2H. Three sequential passes; compliant.
//
// CYCLE-137: initial implementation.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>

namespace singlet::gpu {
namespace enrich {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct ViperConfig {
    bool  deterministic  = true;   // sort-based ranking is deterministic; no-op flag
    float epsilon        = 1e-9f;  // guard for L1 norm near zero
    float max_dense_gb   = 32.f;   // memory guard: reject if T1 buffer > this many GB
};

struct ViperResult {
    core::DeviceMemory<float> nes;  // n_cells × n_regulons, col-major; ~N(0,1)
    int n_cells;
    int n_regulons;
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels (anonymous namespace — not part of the public API)
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// viper_fill_dense_kernel — expand CSC X into dense col-major D (m × n).
//
// One thread per CSC non-zero entry. atomicAdd-free: each (row, col) pair is
// unique in a valid CSC matrix, so no write conflicts occur.
//
// Missing entries remain zero (cudaMemset before launch).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void viper_fill_dense_kernel(
    const float*   __restrict__ values,      // CSC values [nnz]
    const int32_t* __restrict__ row_indices, // CSC row indices [nnz]
    const int32_t* __restrict__ col_ptr,     // CSC col offsets [n+1]
    float*         __restrict__ D,           // output: m × n col-major
    int m,                                   // n_genes (rows)
    int n)                                   // n_cells (cols)
{
    // One thread per column entry via flat launch over all nnz.
    // Recover column from col_ptr via a warp-per-column strategy would be more
    // cache-friendly, but for correctness a flat nnz launch is simpler and safe.
    //
    // Strategy: each block handles one column; threads stride over nnz in col.
    const int col = blockIdx.x;
    if (col >= n) return;

    const int start = col_ptr[col];
    const int end   = col_ptr[col + 1];
    float* D_col    = D + (size_t)col * m;

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        D_col[row_indices[i]] = values[i];
    }
}

// ---------------------------------------------------------------------------
// viper_assign_t1_kernel — convert sort permutation to T1 qnorm values.
//
// After CUB DeviceSegmentedRadixSort produces perm[seg_start..seg_end-1]:
//   perm[seg_start + r] = gene index that has rank r (0-based) within this cell.
//
// T1[gene, cell] = normcdfinvf((r+1) / (m+1))  for r = 0..m-1.
//
// normcdfinvf is the CUDA math library fp32 inverse-normal CDF (erfinvf-based,
// far more accurate than the Acklam polynomial and available on all CUDA arches
// from 5.0+). Equivalent to scipy.stats.norm.ppf.
//
// Launch: one block per cell (blockIdx.x = cell index).
//         threads stride over ranks 0..m-1 within the block.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void viper_assign_t1_kernel(
    const int32_t* __restrict__ perm,  // sort permutation [m × n], col-major
    float*         __restrict__ T1,    // output: m × n col-major
    int m,                             // n_genes
    int n,                             // n_cells
    float m_plus1_inv)                 // 1.0f / (m + 1)
{
    const int cell = blockIdx.x;
    if (cell >= n) return;

    const int32_t* perm_col = perm + (size_t)cell * m;
    float*         T1_col   = T1  + (size_t)cell * m;

    for (int r = threadIdx.x; r < m; r += blockDim.x) {
        // rank r (0-based) → quantile (r+1)/(m+1)
        float p    = static_cast<float>(r + 1) * m_plus1_inv;
        T1_col[perm_col[r]] = normcdfinvf(p);
    }
}

// ---------------------------------------------------------------------------
// viper_l1_norm_kernel — per-regulon L1 norm of W.
//
// norm[r] = Σ_{g=0}^{m-1} |W[g + r*m]|
//
// One block per regulon; warp-shuffle reduction; no atomics → deterministic.
// Identical pattern to wsum_l1_norm_kernel in decoupler_wsum.h.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void viper_l1_norm_kernel(
    const float* __restrict__ W,     // m × n_reg col-major weight matrix
    float*       __restrict__ norm,  // output: L1 norm per regulon [n_reg]
    int m)                           // n_genes
{
    const int reg = blockIdx.x;
    const float* W_col = W + (size_t)reg * m;

    float acc = 0.f;
    for (int g = threadIdx.x; g < m; g += blockDim.x) {
        acc += fabsf(W_col[g]);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    __shared__ float smem[8];
    const int wid   = threadIdx.x / 32;
    const int lane  = threadIdx.x & 31;
    const int nwarp = (blockDim.x + 31) / 32;

    if (lane == 0) smem[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        float val = (lane < nwarp) ? smem[lane] : 0.f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) norm[reg] = val;
    }
}

// ---------------------------------------------------------------------------
// viper_scale_kernel — divide each column r of NES by norm[r].
//
// NES is n × n_reg col-major. One thread per element.
// denom = max(norm[r], epsilon) to guard division by zero.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void viper_scale_kernel(
    float*       __restrict__ nes,   // n × n_reg col-major [n*n_reg]
    const float* __restrict__ norm,  // per-regulon L1 norm [n_reg]
    int n,                           // n_cells
    int n_reg,                       // n_regulons
    float epsilon)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n_reg) return;

    const int reg   = idx / n;
    const float den = (norm[reg] < epsilon) ? epsilon : norm[reg];
    nes[idx] /= den;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// viper — public entry point
//
// Computes VIPER/aREA NES scores (Alvarez et al. 2016 Nat Genet).
//
// Inputs:
//   X          — m_genes × n_cells sparse CSC expression matrix (log1p-normalized)
//   d_W        — device pointer to dense signed weight matrix (m × n_reg, col-major fp32)
//                Positive weights = activating, negative = repressing.
//   n_regulons — number of TF regulons (W's column count)
//   cfg        — ViperConfig (memory guard, epsilon, deterministic flag)
//   stream     — optional cudaStream_t (nullptr → default context)
//
// Returns ViperResult with nes (n × n_reg, col-major) on device.
// NES values are approximately N(0,1) under the null.
// Caller must sync stream before reading nes.
//
// V0 note: implements simplified NES = ES1 = (T1^T · W) / L1(W).
// Full two-tail ES2 blending deferred to v1 (cfg.full_two_tail not yet exposed).
// ---------------------------------------------------------------------------
inline ViperResult viper(
    const io::PzDeviceMatrix& X,
    const float*              d_W,
    int                       n_regulons,
    const ViperConfig&        cfg    = {},
    cudaStream_t              stream = nullptr)
{
    if (stream == nullptr) {
        stream = singlet::gpu::core::default_context().stream();
    }

    const int m   = X.mat.rows;   // genes
    const int n   = X.mat.cols;   // cells
    const int nnz = X.mat.nnz;

    if (n_regulons <= 0) {
        throw std::runtime_error("viper: n_regulons must be > 0");
    }
    if (m <= 0 || n <= 0) {
        throw std::runtime_error(
            "viper: X has invalid dimensions (m=" + std::to_string(m) +
            ", n=" + std::to_string(n) + ")");
    }

    // -----------------------------------------------------------------------
    // Memory guard: T1 dense buffer = m × n × 4 bytes.
    // Reject before allocating if it would exceed cfg.max_dense_gb.
    // -----------------------------------------------------------------------
    const double t1_bytes = static_cast<double>(m) * static_cast<double>(n) * 4.0;
    const double t1_gb    = t1_bytes / (1024.0 * 1024.0 * 1024.0);
    if (t1_gb > static_cast<double>(cfg.max_dense_gb)) {
        throw std::runtime_error(
            "viper: T1 dense buffer would require " +
            std::to_string(t1_gb) + " GB (m=" + std::to_string(m) +
            ", n=" + std::to_string(n) + "), exceeding max_dense_gb=" +
            std::to_string(cfg.max_dense_gb) +
            ". Subset to HVGs (m <= 5000) before calling viper.");
    }

    constexpr int THREADS = 256;

    // -----------------------------------------------------------------------
    // Step 1a: Materialize dense D (m × n, col-major) from sparse CSC X.
    // -----------------------------------------------------------------------
    const size_t dense_elems = static_cast<size_t>(m) * n;
    core::DeviceMemory<float> d_dense(dense_elems);
    cudaMemsetAsync(d_dense.get(), 0, dense_elems * sizeof(float), stream);

    if (nnz > 0) {
        // One block per cell column; threads stride over nnz in that column.
        viper_fill_dense_kernel<<<n, THREADS, 0, stream>>>(
            X.mat.values.get(),
            X.mat.row_indices.get(),
            X.mat.col_ptr.get(),
            d_dense.get(),
            m, n);
    }

    // -----------------------------------------------------------------------
    // Step 1b: Sort each column of d_dense to get a permutation.
    //
    // CUB DeviceSegmentedRadixSort sorts n segments of length m.
    // Input keys:   d_dense (float, m × n col-major — segment i at offset i*m).
    // Input values: gene indices 0..m-1, repeated for each cell (the permutation).
    // Output keys:  sorted expression (scratch, discarded).
    // Output values: permutation perm[c*m + r] = gene index with rank r in cell c.
    //
    // Segment offsets: offsets[c] = c*m, offsets[n] = n*m.
    // -----------------------------------------------------------------------

    // Build segment offsets [0, m, 2m, ..., n*m] on host, then upload.
    core::DeviceMemory<int>   d_seg_offsets(n + 1);
    {
        std::vector<int> h_offsets(n + 1);
        for (int c = 0; c <= n; ++c) h_offsets[c] = c * m;
        cudaMemcpyAsync(d_seg_offsets.get(), h_offsets.data(),
                        (n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    }

    // Build initial value array [0,1,...,m-1, 0,1,...,m-1, ...] (n copies).
    core::DeviceMemory<int>   d_vals_in(dense_elems);
    {
        std::vector<int> h_vals(dense_elems);
        for (int c = 0; c < n; ++c)
            for (int g = 0; g < m; ++g)
                h_vals[static_cast<size_t>(c) * m + g] = g;
        cudaMemcpyAsync(d_vals_in.get(), h_vals.data(),
                        dense_elems * sizeof(int), cudaMemcpyHostToDevice, stream);
    }

    // Scratch buffers for CUB sort output (keys_out discarded; vals_out = perm).
    core::DeviceMemory<float> d_keys_out(dense_elems);
    core::DeviceMemory<int>   d_perm(dense_elems);      // perm[c*m + r] = gene at rank r

    {
        size_t temp_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, temp_bytes,
            d_dense.get(), d_keys_out.get(),
            d_vals_in.get(), d_perm.get(),
            static_cast<int>(dense_elems),
            n,
            d_seg_offsets.get(), d_seg_offsets.get() + 1,
            0, sizeof(float) * 8,
            stream);
        core::DeviceMemory<uint8_t> d_tmp(temp_bytes > 0 ? temp_bytes : 1);
        cub::DeviceSegmentedRadixSort::SortPairs(
            d_tmp.get(), temp_bytes,
            d_dense.get(), d_keys_out.get(),
            d_vals_in.get(), d_perm.get(),
            static_cast<int>(dense_elems),
            n,
            d_seg_offsets.get(), d_seg_offsets.get() + 1,
            0, sizeof(float) * 8,
            stream);
        // d_tmp goes out of scope here; freed automatically.
    }

    // -----------------------------------------------------------------------
    // Step 1c: Assign T1 values from permutation.
    //   T1[perm[c*m + r], c] = normcdfinvf((r+1)/(m+1))
    // -----------------------------------------------------------------------
    // Reuse d_dense as T1 — zero-fill first to overwrite old expression values.
    cudaMemsetAsync(d_dense.get(), 0, dense_elems * sizeof(float), stream);

    {
        const float m_plus1_inv = 1.f / static_cast<float>(m + 1);
        viper_assign_t1_kernel<<<n, THREADS, 0, stream>>>(
            d_perm.get(),
            d_dense.get(),    // repurposed as T1
            m, n,
            m_plus1_inv);
    }

    // Release scratch buffers no longer needed (keys_out, vals_in, perm, offsets).
    // DeviceMemory destructs on scope exit — do explicit reset to free early
    // and reduce peak memory before the GEMM allocation.
    d_keys_out    = core::DeviceMemory<float>();
    d_vals_in     = core::DeviceMemory<int>();
    d_perm        = core::DeviceMemory<int>();
    d_seg_offsets = core::DeviceMemory<int>();

    // -----------------------------------------------------------------------
    // Step 2: cuBLAS Sgemm — NES = T1^T · W  (n × n_reg, col-major).
    //
    // T1 is m × n col-major.  W is m × n_reg col-major.
    // cuBLAS: C = alpha · A^T · B + beta · C
    //   A = T1  (m × n),  opA = CUBLAS_OP_T  → A^T = n × m
    //   B = W   (m × n_reg), opB = CUBLAS_OP_N → m × n_reg
    //   C = NES (n × n_reg, col-major), ldc = n
    // -----------------------------------------------------------------------
    const size_t nes_elems = static_cast<size_t>(n) * n_regulons;
    core::DeviceMemory<float> d_nes(nes_elems > 0 ? nes_elems : 1);
    cudaMemsetAsync(d_nes.get(), 0, nes_elems * sizeof(float), stream);

    if (nes_elems > 0) {
        cublasHandle_t blas_handle;
        CUBLAS_CHECK(cublasCreate(&blas_handle));
        cublasSetStream(blas_handle, stream);

        const float alpha = 1.f, beta = 0.f;

        // Sgemm: C (n × n_reg) = T1^T (n × m) · W (m × n_reg)
        // cuBLAS col-major convention:
        //   m_blas = n (rows of C)
        //   n_blas = n_regulons (cols of C)
        //   k_blas = m (shared dimension)
        //   lda = m (leading dim of T1 stored as m × n col-major, transposed)
        //   ldb = m (leading dim of W, stored as m × n_reg col-major)
        //   ldc = n (leading dim of C stored as n × n_reg col-major)
        CUBLAS_CHECK(cublasSgemm(
            blas_handle,
            CUBLAS_OP_T,          // T1^T
            CUBLAS_OP_N,          // W unchanged
            n,                    // rows of C
            n_regulons,           // cols of C
            m,                    // shared dim
            &alpha,
            d_dense.get(), m,     // T1: m × n, lda = m
            d_W,           m,     // W:  m × n_reg, ldb = m
            &beta,
            d_nes.get(),   n));   // C:  n × n_reg, ldc = n

        cublasDestroy(blas_handle);
    }

    // T1 (d_dense) no longer needed — free before norm step.
    d_dense = core::DeviceMemory<float>();

    // -----------------------------------------------------------------------
    // Step 3a: Compute per-regulon L1 norm of W.
    // -----------------------------------------------------------------------
    core::DeviceMemory<float> d_norm(n_regulons);
    {
        viper_l1_norm_kernel<<<n_regulons, THREADS, 0, stream>>>(
            d_W, d_norm.get(), m);
    }

    // -----------------------------------------------------------------------
    // Step 3b: Divide NES by L1 norm — one thread per element.
    // -----------------------------------------------------------------------
    {
        const int total  = static_cast<int>(nes_elems);
        const int grid   = (total + THREADS - 1) / THREADS;
        viper_scale_kernel<<<grid, THREADS, 0, stream>>>(
            d_nes.get(),
            d_norm.get(),
            n, n_regulons,
            cfg.epsilon);
    }

    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    ViperResult res;
    res.nes        = std::move(d_nes);
    res.n_cells    = n;
    res.n_regulons = n_regulons;
    return res;
}

}  // namespace enrich
}  // namespace singlet::gpu
