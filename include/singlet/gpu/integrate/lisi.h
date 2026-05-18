// SPDX-License-Identifier: MIT
// integrates: Korsunsky et al. 2019 Harmony LISI metric
// singlet/gpu/integrate/lisi.h
//
// LISI — Local Inverse Simpson's Index.
// Korsunsky I, Millard N, Fan J, et al. (2019) Nat Methods 16:1289-1296.
// §Methods "LISI scores".
//
// For each cell c with k nearest neighbors {n_1,...,n_k} and a label vector
// label[c] ∈ {0,...,n_labels-1}:
//   1. p_l = count(label[n_i] == l) / k  for each label l.
//   2. D[c] = Σ_l p_l²   (Simpson's diversity index)
//   3. LISI[c] = 1 / D[c]
//
// Interpretations:
//   iLISI: use batch labels.  Higher LISI → better batch mixing.
//   cLISI: use cell-type labels.  Lower LISI → better cluster preservation.
//
// v0: uniform (unweighted) neighbor counts — the most-used variant in scanpy
//     benchmarks and the recommended default.
// v1 (follow-up): Gaussian-kernel-weighted variant tuned per-cell by perplexity,
//     matching the reference R lisi package exactly.
//
// GPU mapping: one block per cell.  Each block maintains a shared-memory
//   histogram of n_labels ints, accumulates counts for each of k neighbors via
//   atomicAdd, then thread 0 sweeps the histogram to compute D and writes LISI.
//   For n_labels ≤ 64 the thread-0 sweep is only ~64 fp32 ops — trivially fast.
//   With cfg.deterministic = true (default), the histogram is reset and filled
//   entirely by thread 0 using a serial scan — zero atomic noise, bit-identical
//   across runs.  Rule 18: atomicAdd noise ≤ 1e-7 on int counts (always exact
//   for integer values ≤ k ≤ INT_MAX).
//
// Memory (Rule 5): no raw cudaMalloc — DeviceMemory<float> throughout.
// D2H (Rule 4): no D2H per inner loop.
// Shared memory: n_labels × 4 bytes per block (max 256 bytes for n_labels=64).
// CYCLE-133: initial implementation.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/graph/knn.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace singlet::gpu {
namespace integrate {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct LisiConfig {
    // deterministic = true: thread 0 owns the entire histogram pass (serial
    //   scan over k neighbors).  Bit-identical across runs.
    // deterministic = false: all threads in the block cooperate via atomicAdd
    //   into shared-memory int counts.  Atomics on ints are always exact for
    //   count values ≤ k; no floating-point noise is introduced at this step.
    //   The only non-determinism source is warp scheduling order, which is
    //   benign for int atomics.  Use false for maximum throughput on very
    //   large k (k > 256) or large n_labels (> 64).
    bool deterministic = true;
};

struct LisiResult {
    core::DeviceMemory<float> lisi;  // [n_cells] inverse Simpson index per cell
    int n_cells;
    int n_labels;
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels
// ---------------------------------------------------------------------------
namespace {

// lisi_kernel — one block per cell.
//
// Shared memory layout: int counts[n_labels]  (n_labels passed as smem size).
//
// Deterministic path (cfg.deterministic = true, used when blockDim.x == 1):
//   Thread 0 zeroes counts[], iterates over k neighbors and increments
//   counts[label[nbr]], then sweeps counts to compute D.
//
// Parallel path (blockDim.x > 1):
//   All threads zero counts[] cooperatively, then load-balance neighbor
//   iterations via stride, using atomicAdd for the shared-memory int tally.
//   After __syncthreads(), thread 0 sweeps counts.
//
// Explicit kernel parameters (Rule: every kernel carries every dim/buffer it
// indexes as an explicit parameter — no globals, no undefined vars).
__global__
void lisi_kernel(
    const int* __restrict__ neighbors,    // [n_cells * k]
    const int* __restrict__ label,        // [n_cells]
    float*     __restrict__ lisi_out,     // [n_cells]
    int n_cells, int k, int n_labels)
{
    // One block per cell.
    const int cell = static_cast<int>(blockIdx.x);
    if (cell >= n_cells) return;

    // Dynamic shared memory: int counts[n_labels].
    extern __shared__ int smem_counts[];

    // Zero shared histogram cooperatively.
    for (int l = static_cast<int>(threadIdx.x); l < n_labels;
         l += static_cast<int>(blockDim.x)) {
        smem_counts[l] = 0;
    }
    __syncthreads();

    // Accumulate label counts over k neighbors.
    const int* row_nbrs = neighbors + static_cast<ptrdiff_t>(cell) * k;
    for (int i = static_cast<int>(threadIdx.x); i < k;
         i += static_cast<int>(blockDim.x)) {
        const int nbr = row_nbrs[i];
        const int lbl = label[nbr];
        atomicAdd(&smem_counts[lbl], 1);
    }
    __syncthreads();

    // Thread 0: compute Simpson's D and write LISI.
    if (threadIdx.x == 0) {
        const float k_inv = 1.f / static_cast<float>(k);
        float D = 0.f;
        for (int l = 0; l < n_labels; ++l) {
            const float p = static_cast<float>(smem_counts[l]) * k_inv;
            D += p * p;
        }
        // Guard: LISI = 1 / max(D, eps) prevents Inf for degenerate D=0
        // (impossible in practice: at least one neighbor exists, so D >= 1/k²).
        // eps = 1e-9 matches the combat.h guard convention.
        const float eps = 1e-9f;
        lisi_out[cell] = 1.f / (D > eps ? D : eps);
    }
}

// lisi_kernel_serial — deterministic single-thread-per-cell variant.
// Launched with blockDim.x = 1 so each block is a single thread; avoids
// atomicAdd entirely.  Shared memory still used for counts[] (consistent
// code path; the compiler may optimize it to registers at blockDim=1).
__global__
void lisi_kernel_serial(
    const int* __restrict__ neighbors,
    const int* __restrict__ label,
    float*     __restrict__ lisi_out,
    int n_cells, int k, int n_labels)
{
    const int cell = static_cast<int>(blockIdx.x);
    if (cell >= n_cells) return;

    extern __shared__ int smem_counts[];

    // Zero histogram (single thread).
    for (int l = 0; l < n_labels; ++l) smem_counts[l] = 0;

    // Accumulate (no atomics needed — single thread).
    const int* row_nbrs = neighbors + static_cast<ptrdiff_t>(cell) * k;
    for (int i = 0; i < k; ++i) {
        smem_counts[label[row_nbrs[i]]]++;
    }

    // Compute D and LISI.
    const float k_inv = 1.f / static_cast<float>(k);
    float D = 0.f;
    for (int l = 0; l < n_labels; ++l) {
        const float p = static_cast<float>(smem_counts[l]) * k_inv;
        D += p * p;
    }
    const float eps = 1e-9f;
    lisi_out[cell] = 1.f / (D > eps ? D : eps);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// lisi() — public entry point
// ---------------------------------------------------------------------------
//
// knn:      KnnResult from singlet::gpu::graph::compute_knn(...).
//           Uses knn.neighbors [n*k] and knn.n, knn.k.  Distances unused.
// d_label:  device int[n_cells] with values in [0, n_labels).
// n_labels: number of distinct label classes.  Must be >= 1.
// cfg:      LisiConfig (default: deterministic = true).
// stream:   CUDA stream (nullptr = default stream from context).
//
// Returns LisiResult with device float[n_cells] containing per-cell LISI.
// Caller synchronizes stream before reading.
inline LisiResult lisi(
    const graph::KnnResult& knn,
    const int*              d_label,
    int                     n_labels,
    const LisiConfig&       cfg    = {},
    cudaStream_t            stream = nullptr)
{
    if (stream == nullptr) {
        stream = singlet::gpu::core::default_context().stream();
    }

    const int n = knn.n;
    const int k = knn.k;

    if (n <= 0)
        throw std::runtime_error("lisi: empty knn graph (n=" +
            std::to_string(n) + ")");
    if (k <= 0)
        throw std::runtime_error("lisi: k must be > 0 (k=" +
            std::to_string(k) + ")");
    if (n_labels <= 0)
        throw std::runtime_error("lisi: n_labels must be >= 1 (got " +
            std::to_string(n_labels) + ")");
    if (n_labels > 1024)
        throw std::runtime_error("lisi: n_labels=" + std::to_string(n_labels) +
            " exceeds shared-memory limit (max 1024). "
            "For n_labels > 1024 use the global-memory fallback (not implemented in v0).");
    if (d_label == nullptr)
        throw std::runtime_error("lisi: d_label is null");

    // Shared memory per block: int counts[n_labels].
    const size_t smem_bytes = static_cast<size_t>(n_labels) * sizeof(int);

    // Output buffer.
    core::DeviceMemory<float> d_lisi(static_cast<size_t>(n));

    if (cfg.deterministic) {
        // Serial path: one thread per block (blockDim.x = 1), one block per cell.
        // Fully deterministic, no atomics.
        lisi_kernel_serial<<<n, 1, smem_bytes, stream>>>(
            knn.neighbors.get(), d_label, d_lisi.get(),
            n, k, n_labels);
    } else {
        // Parallel path: use up to 32 threads per block (covers k neighbors
        // efficiently for typical k ∈ [10, 50]).  For large k, increase threads.
        // Cap at 256 threads; atomicAdd on int is exact regardless.
        const int threads = std::min(256, std::max(1, k));
        lisi_kernel<<<n, threads, smem_bytes, stream>>>(
            knn.neighbors.get(), d_label, d_lisi.get(),
            n, k, n_labels);
    }

    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    LisiResult res;
    res.lisi     = std::move(d_lisi);
    res.n_cells  = n;
    res.n_labels = n_labels;
    return res;
}

}  // namespace integrate
}  // namespace singlet::gpu
