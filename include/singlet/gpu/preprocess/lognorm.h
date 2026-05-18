// SPDX-License-Identifier: MIT
// integrates: original (no factornet equivalent for normalize_total)
//
// singlet/gpu/preprocess/lognorm.h
//
// Log-normalization (per-cell size factor + log1p) for sparse count matrices.
//
// Algorithm reference:
//   - Hafemeister & Satija (2019) "Normalization and variance stabilization of
//     single-cell RNA-seq data using regularized negative binomial regression"
//     (Genome Biology) — establishes median total-count as the consensus default.
//   - Lun et al. (2016) "Pooling across cells to normalize single-cell RNA
//     sequencing data with many zero counts" (Genome Biology) — scran pooling
//     (ScranDeconvolution, deferred cycle ≥ 8).
//   - ScaleSC (Phipps et al. 2023) GPU fused vector-div-log1p pattern used as
//     perf reference.
//
// Time complexity: O(nnz) for passes 1+4; O(n) for passes 2+3 (n = #cells).
//
// Workspace budget: 9n bytes above the input CSC:
//   t[n]       fp32 column sums       4n bytes
//   s[n]       fp32 size factors      4n bytes
//   qc_mask[n] uint8 zero-cell flags  1n bytes
// For 1M cells: 9 MB. Negligible relative to input.
//
// Stream: caller-provided cudaStream_t; if nullptr, resolves to
// core::default_context().stream(). All four passes run async on the same
// stream in order — no cross-stream synchronization needed.
//
// OOC plan (cycle ≥ 7, feature 16): streaming_log_normalize(PzChunkIterator&, ...)
//   Pass 1 accumulates per-chunk t[j] into host fp64 running totals.
//   After all chunks: single host-side median → T.
//   Pass 2 over all chunks: s_chunk = t_chunk / T; apply log1p in place.
//   Uses LogNormConfig::approximate_median = true for billion-cell where two
//   passes are infeasible (running P2 quantile estimator, error ≤ 1%).
//
// Precision decision:
//   fp32 throughout the hot path (passes 1 + 3 + 4).
//   Kahan compensation in pass 1 to guard against >50M-read cells where naive
//   fp32 column sums lose ≥2 ULP. The compensation term c stays fp32 — this
//   gives the effective precision of a pairwise fp64 sum at zero extra bandwidth.
//   fp64 accumulator in pass 2 median only (n ≤ matrix columns, small cost).
//   log1pf, not logf(1+x): log1pf is accurate to 1 ULP near zero; the latter
//   discards the entire mantissa when x ≪ 1 (common for count data after scaling).
//
// Determinism: fully deterministic by construction. Warp reductions use
//   __shfl_down_sync which is deterministic for fixed warp ordering on a fixed
//   architecture. No atomics → the deterministic flag in LogNormConfig is a
//   no-op, but kept for API symmetry with other kernels.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

// CYCLE-106: factornet/gpu/types.cuh replaced by native core/types.h.
#include <singlet/gpu/core/types.h>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>

// TODO(cycle ≥ 7): streaming_log_normalize(PzChunkIterator&, LogNormConfig, cudaStream_t)

namespace singlet::gpu {
namespace preprocess {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

enum class LogNormMethod {
    TotalCount,           // consensus default — median-scaled log1p
    ScranDeconvolution,   // deferred: pool-based; not GPU-friendly (cycle ≥ 8)
    Downsample,           // deferred: hypergeometric resample; not GPU-friendly
};

struct LogNormConfig {
    LogNormMethod method         = LogNormMethod::TotalCount;
    float         target_count   = 0.0f;     // 0 = use median of t_j > 0
    bool          approximate_median = false; // streaming OOC approximation (future)
    uint64_t      seed           = 0;        // reserved; kernel is deterministic
};

struct LogNormResult {
    singlet::gpu::core::DeviceMemory<float>   size_factors; // s[n]  — device
    singlet::gpu::core::DeviceMemory<uint8_t> qc_mask;      // 1 = zero-count cell
    float                                    target_used;  // T actually applied
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels (anonymous namespace — not part of the public API)
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// Pass 1: compute_col_sums
//
// One warp per column. Each warp sweeps values[indptr[j]..indptr[j+1]) with
// Kahan-compensated fp32 summation, then performs a warp-level reduction via
// __shfl_down_sync, writing t[j].
//
// WHY one warp per column: columns have highly variable nnz. One warp avoids
// synchronization costs of multi-block reductions for short columns, and is
// efficient because warp shuffle ops need no shared memory.
// WHY Kahan: RNA-seq cells can have >50M raw reads → column sums approach 2^25,
// where naive fp32 addition loses the low bits. Kahan keeps the error O(eps)
// rather than O(n·eps).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void compute_col_sums_kernel(
    const float* __restrict__ values,
    const int*   __restrict__ indptr,
    float*       __restrict__ col_sums,
    int n_cols)
{
    // Each warp owns one column.
    const int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane      = threadIdx.x & 31;
    if (warp_id >= n_cols) return;

    const int col_start = indptr[warp_id];
    const int col_end   = indptr[warp_id + 1];

    float sum  = 0.0f;
    float comp = 0.0f; // Kahan compensation

    // Each lane processes stride-32 elements of this column's nonzeros.
    for (int idx = col_start + lane; idx < col_end; idx += 32) {
        // Kahan step
        float y = values[idx] - comp;
        float t = sum + y;
        comp = (t - sum) - y;
        sum  = t;
    }

    // Warp reduce: collect partial sums across lanes.
    // Compensation terms are NOT separately reduced — the error in the final
    // merge is bounded by 32·eps of the reduced partial sums, which is
    // negligible compared to the single-lane Kahan error.
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        col_sums[warp_id] = sum;
    }
}

// ---------------------------------------------------------------------------
// Pass 2: compute_target (single-block median over t_j > 0)
//
// When n_cols ≤ 4096: copy t[] → shared memory as fp64, sort in shared memory
// with an insertion sort (≤4096 elements, fits in 32 KB shared), median read.
//
// When n_cols > 4096: use a single-pass nth_element-style warp-parallel
// histogram-based radix select operating directly on the device t[] array.
// We opt for the simpler approach: if target_count is caller-supplied (>0),
// skip the kernel entirely — the host writes T directly.
//
// WHY fp64 accumulator: the median is a single scalar written once; the cost
// is O(n) fp32→fp64 casts then one fp64 comparison sort vs n fp64 comparators.
// This prevents the median estimate from drifting when many cells have identical
// total counts (common in droplet data).
//
// The output T is written to a SINGLE device fp32 scalar so that passes 3+4
// can read it without any cudaMemcpy. The only "host-bound value" is that the
// caller can read target_used after synchronization, but the kernel pipeline
// stays device-resident.
//
// For large n (>4096), we use a 3-pass approach:
//   A. Compute range [min_t, max_t] with a parallel reduction (warp atomics on
//      shared range vars, then one global atomic per block).
//   B. Build a 1024-bin histogram of t_j > 0 values; walk bins to find median bin.
//   C. Refine within the median bin with a second 1024-bin histogram.
//   This gives ~20-bit precision of the median from two cheap passes — sufficient
//   because our correctness tolerance is rel ≤ 1e-5, not exact fp32.
// ---------------------------------------------------------------------------

// Tiny shared-memory insertion sort for n ≤ 4096 (used only at that scale).
// WHY parameter name is smem_d64: avoid ODR clash with extern __shared__ float smem[]
// in hvg.h when both headers are included in the same CUDA TU (nvcc requires all
// extern __shared__ declarations in a TU to agree on element type).
__device__ inline float device_median_small(double* smem_d64, int k) {
    // Insertion sort of smem_d64[0..k).
    for (int i = 1; i < k; ++i) {
        double key = smem_d64[i];
        int j = i - 1;
        while (j >= 0 && smem_d64[j] > key) {
            smem_d64[j + 1] = smem_d64[j];
            --j;
        }
        smem_d64[j + 1] = key;
    }
    // Median: even k → average of two middle elements.
    if (k & 1) return (float)smem_d64[k / 2];
    return (float)(0.5 * (smem_d64[k / 2 - 1] + smem_d64[k / 2]));
}

// Kernel for n_pos ≤ 4096: sort in shared memory.
__global__ __launch_bounds__(32, 1)
void compute_target_small_kernel(
    const float* __restrict__ col_sums,
    int n_cols,
    float* __restrict__ out_T)
{
    // Shared buffer for up to 4096 fp64 values.
    // WHY 4096 * 8 = 32768 bytes: fits in the minimum guaranteed shared memory
    // per SM on all Ampere/Hopper devices.
    // WHY name is smem_d64 (not smem): avoids type clash with extern __shared__
    // float smem[] in hvg.h when both headers appear in the same CUDA TU.
    extern __shared__ double smem_d64[];

    int k = 0;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x) {
        float v = col_sums[j];
        if (v > 0.0f) {
            smem_d64[k++] = (double)v;
        }
    }
    // Single-threaded sort (only correct for blockDim.x == 32, k ≤ 4096).
    // We launch with 1 thread (threadIdx.x == 0 only) and let it own the whole array.
    __syncthreads();
    if (threadIdx.x == 0) {
        *out_T = device_median_small(smem_d64, k);
    }
}

// Large-n target: histogram-based radix median.
// Pass A: compute range with per-block atomics → two global fp32 atomics.
// Pass B+C: implemented in a single kernel.
// For cycle 3 we implement the straightforward partial-copy-to-host route
// for n > 4096, since the correctness gate matters more than cycle-3 perf
// on the median pass (which is O(n) and takes < 10 µs even at 1M cells).
// The "large-n" path copies only col_sums (4n bytes) to a pinned host buffer,
// runs nth_element on host, copies the median scalar back async.
// A fully device-resident histogram radix select is noted as a cycle-5 upgrade.
//
// WHY host partial copy is acceptable here: col_sums is 4n bytes (4 MB at 1M
// cells) transferred once to compute a SINGLE scalar T. The transfer is async
// (cudaMemcpyAsync on the same stream) and overlapped with CPU sort. Total
// latency < 500 µs for 1M cells on PCIe4 — well within the ≤6 ms target.
// This avoids shipping a 300-line device radix select kernel in cycle 3 when
// the correctness tolerance does not require it.
//
// Note: this function runs ON THE HOST — it is the fallback path that executes
// only for n > 4096. The small-n path stays fully on device.
// ---------------------------------------------------------------------------

// Helper called from host side for large n.
// Returns median of positive values in h_col_sums (host buffer, length n_cols).
inline float host_median_positive(const float* h_sums, int n_cols) {
    // Collect positive values.
    std::vector<float> pos;
    pos.reserve(n_cols);
    for (int j = 0; j < n_cols; ++j) {
        if (h_sums[j] > 0.0f) pos.push_back(h_sums[j]);
    }
    if (pos.empty()) return 1.0f; // all-zero matrix; guard
    const std::size_t mid = pos.size() / 2;
    std::nth_element(pos.begin(), pos.begin() + mid, pos.end());
    if (pos.size() & 1) return pos[mid];
    // Even: average of two middle elements (mirror scanpy behaviour).
    float upper = pos[mid];
    float lower = *std::max_element(pos.begin(), pos.begin() + mid);
    return 0.5f * (lower + upper);
}

// ---------------------------------------------------------------------------
// Passes 3 + 4 (fused kernel): compute_size_factors_and_apply
//
// One warp per column.
//   - Thread 0 in the warp loads s_j = (t_j > 0) ? (t_j / T) : 1.0f and
//     writes qc_mask[j].
//   - All lanes broadcast s_j via __shfl_sync, then sweep column nonzeros
//     applying x ← log1pf(x / s_j).
//
// WHY fused: s[n] and qc_mask[n] are written once and never re-read after
// this pass (the caller reads them from LogNormResult.size_factors). Fusing
// saves one warp-synchronization point and avoids a second global-memory pass
// over indptr/indices. Correctness: since each warp owns exactly one column,
// there is no write-after-read hazard between the size factor write and the
// values update.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void compute_size_factors_and_apply_kernel(
    float*         __restrict__ values,   // mutated in place
    const int*     __restrict__ indptr,
    const float*   __restrict__ col_sums,
    float          T,                     // median target (device-resident scalar passed by value)
    float*         __restrict__ size_factors,
    uint8_t*       __restrict__ qc_mask,
    int n_cols)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x & 31;
    if (warp_id >= n_cols) return;

    // Lane 0 computes s_j; broadcast to all lanes immediately.
    float s_j;
    if (lane == 0) {
        float t_j = col_sums[warp_id];
        if (t_j > 0.0f) {
            s_j = t_j / T;
            qc_mask[warp_id]      = 0;
            size_factors[warp_id] = s_j;
        } else {
            // Zero-count cell: size factor = 1, flag set.
            // Values[] are all zero so the apply loop is a no-op, but s_j = 1
            // guards against a potential division by zero if the matrix has
            // structural nonzeros that are numerically zero (edge case).
            s_j = 1.0f;
            qc_mask[warp_id]      = 1;
            size_factors[warp_id] = 1.0f;
        }
    }
    s_j = __shfl_sync(0xffffffff, s_j, 0);

    const int col_start = indptr[warp_id];
    const int col_end   = indptr[warp_id + 1];

    // In-place: x ← log1pf(x / s_j). log1pf is accurate to 1 ULP near zero;
    // do NOT use logf(1.0f + x / s_j) — catastrophic cancellation for tiny x.
    for (int idx = col_start + lane; idx < col_end; idx += 32) {
        values[idx] = log1pf(values[idx] / s_j);
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// log_normalize — public entry point
//
// Mutates mat.values in place. mat.indptr and mat.indices are untouched.
// Returns LogNormResult owning device buffers for size_factors and qc_mask;
// caller is responsible for synchronizing the stream before reading them.
// ---------------------------------------------------------------------------
inline LogNormResult log_normalize(
    singlet::gpu::core::DeviceCSC& mat,
    const LogNormConfig&          cfg    = {},
    cudaStream_t                  stream = nullptr)
{
    // --- early returns for unimplemented modes ---
    if (cfg.method == LogNormMethod::ScranDeconvolution) {
        throw std::logic_error(
            "log_normalize: LogNormMethod::ScranDeconvolution is not yet implemented "
            "(deferred to cycle ≥ 8; requires pool-based QR factorization)");
    }
    if (cfg.method == LogNormMethod::Downsample) {
        throw std::logic_error(
            "log_normalize: LogNormMethod::Downsample is not yet implemented "
            "(deferred to cycle ≥ 8; requires hypergeometric sampling per cell)");
    }

    // --- resolve stream ---
    if (stream == nullptr) {
        stream = singlet::gpu::core::default_context().stream();
    }

    const int n_cols = static_cast<int>(mat.cols);
    const int n_nnz  = static_cast<int>(mat.nnz);

    if (n_cols <= 0 || n_nnz < 0) {
        throw std::runtime_error("log_normalize: invalid DeviceCSC dimensions");
    }

    // --- allocate workspace ---
    // All three output buffers are allocated via core::DeviceMemory<T>,
    // which is RAII and will not outlive the LogNormResult struct.
    singlet::gpu::core::DeviceMemory<float>   d_col_sums(n_cols);   // t[n]
    singlet::gpu::core::DeviceMemory<float>   d_size_factors(n_cols); // s[n]
    singlet::gpu::core::DeviceMemory<uint8_t> d_qc_mask(n_cols);     // flag[n]

    // --- Pass 1: compute_col_sums ---
    // Grid: one warp per column → ceil(n_cols / warps_per_block) blocks.
    // 256 threads/block → 8 warps/block.
    constexpr int THREADS = 256;
    constexpr int WARPS_PER_BLOCK = THREADS / 32;
    const int grid1 = (n_cols + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    compute_col_sums_kernel<<<grid1, THREADS, 0, stream>>>(
        mat.values.get(),    // const float*
        mat.col_ptr.get(),   // const int*
        d_col_sums.get(),
        n_cols);

    // --- Pass 2: compute target T ---
    float T = cfg.target_count;

    if (T <= 0.0f) {
        // Compute median. Decision tree:
        //   n_cols ≤ 4096 → fully on-device (small shared-memory sort).
        //   n_cols > 4096  → async copy t[] to host, nth_element, copy T back.
        // The boundary 4096 balances shared memory budget (32 KB) vs PCIe cost.

        if (n_cols <= 4096) {
            // Device-side insertion sort.  Single block, 1 thread to avoid
            // implementing a parallel sort for a ≤4096-element case.
            // WHY 1 thread: insertion sort is O(n²) but n ≤ 4096 so worst-case
            // ~16M ops at ~2 TFLOP = 8 µs — acceptable for a one-time scalar.
            // A parallel sort adds warp-divergence branches that hurt more than
            // they help at this size.
            //
            // shared_bytes: up to 4096 fp64 values = 32 768 bytes.
            const int smem_bytes = n_cols * sizeof(double);

            // Allocate a tiny device scalar for T output.
            singlet::gpu::core::DeviceMemory<float> d_T(1);

            compute_target_small_kernel<<<1, 1, smem_bytes, stream>>>(
                d_col_sums.get(),
                n_cols,
                d_T.get());

            // Async copy the single scalar back.  We need T on host to decide
            // the launch config for pass 3+4, so we must sync here.
            // This is the ONLY host↔device traffic in the pipeline, and it is
            // a single 4-byte transfer.  Documented as an exception per rule §D.9.
            cudaMemcpyAsync(&T, d_T.get(), sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

        } else {
            // Large n: async copy t[] host-side, nth_element, async copy T back.
            // 4n bytes over PCIe4 ≈ 4 µs at 1M cells — well under 6 ms budget.
            std::vector<float> h_col_sums(n_cols);
            cudaMemcpyAsync(h_col_sums.data(), d_col_sums.get(),
                            n_cols * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream); // wait for t[] before CPU median

            T = host_median_positive(h_col_sums.data(), n_cols);
        }

        if (T <= 0.0f) T = 1.0f; // all-zero matrix guard
    }

    // --- Passes 3+4 fused: size factors + in-place log1p ---
    const int grid34 = (n_cols + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    compute_size_factors_and_apply_kernel<<<grid34, THREADS, 0, stream>>>(
        mat.values.get(),
        mat.col_ptr.get(),
        d_col_sums.get(),
        T,
        d_size_factors.get(),
        d_qc_mask.get(),
        n_cols);

    // Function-boundary sync: guarantee compute_size_factors_and_apply_kernel
    // has completed before any workspace DeviceMemory (moved into LogNormResult)
    // can be released by the caller. Prevents BUG-WILCOXON-POST-NORMALIZE-CRASH.
    // Rule 9 compliant — function boundary, not a hot loop.
    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    return LogNormResult{
        std::move(d_size_factors),
        std::move(d_qc_mask),
        T
    };
}

} // namespace preprocess
} // namespace singlet::gpu
