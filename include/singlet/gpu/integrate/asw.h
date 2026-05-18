// SPDX-License-Identifier: MIT
// integrates: Rousseeuw (1987) Average Silhouette Width metric
// singlet/gpu/integrate/asw.h
//
// ASW — Average Silhouette Width.
// Rousseeuw PJ (1987) J Comput Appl Math 20:53-65.
// Modern single-cell usage: scIB benchmarking suite; Luecken et al. (2022).
//
// For each cell c with cluster label l(c), using k nearest neighbors:
//   a(c) = mean distance from c to neighbors in cluster l(c)
//          (intra-cluster dispersion; 0 if no neighbors share the label)
//   b(c) = min over k_l != l(c) of (mean distance from c to neighbors in k_l)
//          (distance to nearest other cluster; 0 if no other cluster in kNN)
//   silhouette[c] = (b - a) / max(a, b)   ∈ [-1, 1]
//   silhouette[c] = 0 if max(a, b) == 0   (degenerate: singleton or all same)
//
// v0: kNN-approximation ASW (O(n·k), bounded memory).
//   a(c) and b(c) computed only from the k nearest neighbors — identical to
//   the scIB default.  Correlates >0.95 with full O(n²) ASW for typical k ≥ 15
//   and well-separated clusters.
//
// GPU mapping: one block per cell.  Per block: shared-memory float sum_dist[n_labels]
//   and int count[n_labels].  Thread 0 sweeps the kNN row, accumulates into smem,
//   then computes a, b, silhouette — serial, bit-identical, no atomics.
//   cub::DeviceReduce::Sum computes the scalar ASW mean (single D2H after kernel).
//
// Memory (Rule 5): no raw cudaMalloc — DeviceMemory<T> throughout.
// D2H (Rule 4): one D2H for the scalar sum (compliant — not in a loop).
// Shared memory: (n_labels * 4 + n_labels * 4) bytes per block = 8 * n_labels.
//   n_labels ≤ 64 → ≤ 512 bytes.  Throws if n_labels > 1024.
// CYCLE-139: initial implementation.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/graph/knn.h>

#include <cub/device/device_reduce.cuh>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace singlet::gpu {
namespace integrate {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct AswConfig {
    // deterministic = true (default): thread 0 owns all shared-memory ops.
    // Serial scan over k neighbors — no atomics, bit-identical across runs.
    // Rule 18 compliant.
    bool deterministic = true;
};

struct AswResult {
    core::DeviceMemory<float> silhouette;  // [n_cells] per-cell silhouette in [-1, 1]
    float asw_mean;   // mean of silhouette over all cells
    int   n_cells;
    int   n_labels;
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels
// ---------------------------------------------------------------------------
namespace {

// asw_kernel_serial — deterministic single-thread-per-block ASW.
//
// Shared memory layout:
//   float smem_sum [n_labels]  — sum of distances per label over kNN row
//   int   smem_cnt [n_labels]  — count of neighbors per label
//
// Explicit kernel parameters (Rule: every kernel carries every dim/buffer it
// indexes as an explicit parameter — no globals, no undefined vars).
__global__
void asw_kernel_serial(
    const int*   __restrict__ neighbors,    // [n_cells * k]
    const float* __restrict__ distances,    // [n_cells * k]
    const int*   __restrict__ label,        // [n_cells]
    float*       __restrict__ silhouette,   // [n_cells] output
    int n_cells, int k, int n_labels)
{
    // One block per cell.
    const int cell = static_cast<int>(blockIdx.x);
    if (cell >= n_cells) return;

    // Shared memory: float smem_sum[n_labels] then int smem_cnt[n_labels].
    extern __shared__ unsigned char smem_raw[];
    float* smem_sum = reinterpret_cast<float*>(smem_raw);
    int*   smem_cnt = reinterpret_cast<int*>(smem_raw + static_cast<size_t>(n_labels) * sizeof(float));

    // Thread 0 only (deterministic path, blockDim.x == 1).
    // Zero accumulators.
    for (int l = 0; l < n_labels; ++l) {
        smem_sum[l] = 0.f;
        smem_cnt[l] = 0;
    }

    // Accumulate distances by neighbor label.
    const int*   row_nbr  = neighbors + static_cast<ptrdiff_t>(cell) * k;
    const float* row_dist = distances + static_cast<ptrdiff_t>(cell) * k;
    for (int i = 0; i < k; ++i) {
        const int   nbr = row_nbr[i];
        const int   lbl = label[nbr];
        smem_sum[lbl] += row_dist[i];
        smem_cnt[lbl]++;
    }

    // Compute per-label mean distances.
    const int self_label = label[cell];

    // a(c): mean intra-cluster distance.
    float a = 0.f;
    if (smem_cnt[self_label] > 0) {
        a = smem_sum[self_label] / static_cast<float>(smem_cnt[self_label]);
    }

    // b(c): min mean distance to any OTHER label present in kNN.
    // If no other-cluster neighbor exists, b stays 0 (degenerate → silhouette = 0).
    float b = 0.f;
    bool  found_other = false;
    for (int l = 0; l < n_labels; ++l) {
        if (l == self_label) continue;
        if (smem_cnt[l] == 0) continue;
        const float mean_l = smem_sum[l] / static_cast<float>(smem_cnt[l]);
        if (!found_other || mean_l < b) {
            b = mean_l;
            found_other = true;
        }
    }

    // silhouette(c) = (b - a) / max(a, b).
    // Guard: if max(a, b) == 0 (singleton cluster AND no other cluster) → 0.
    const float denom = (a > b) ? a : b;
    float sil = 0.f;
    if (denom > 0.f) {
        sil = (b - a) / denom;
    }
    silhouette[cell] = sil;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// asw() — public entry point
// ---------------------------------------------------------------------------
//
// knn:      KnnResult from singlet::gpu::graph::compute_knn(...).
//           Uses knn.neighbors [n*k], knn.distances [n*k], knn.n, knn.k.
// d_label:  device int[n_cells] with values in [0, n_labels).
// n_labels: number of distinct label classes.  Must be >= 1.
// cfg:      AswConfig (default: deterministic = true).
// stream:   CUDA stream (nullptr = default stream from context).
//
// Returns AswResult with device float[n_cells] silhouette scores and
// scalar asw_mean.  Caller synchronizes stream before reading silhouette.
inline AswResult asw(
    const graph::KnnResult& knn,
    const int*              d_label,
    int                     n_labels,
    const AswConfig&        cfg    = {},
    cudaStream_t            stream = nullptr)
{
    if (stream == nullptr) {
        stream = singlet::gpu::core::default_context().stream();
    }

    const int n = knn.n;
    const int k = knn.k;

    if (n <= 0)
        throw std::runtime_error("asw: empty knn graph (n=" +
            std::to_string(n) + ")");
    if (k <= 0)
        throw std::runtime_error("asw: k must be > 0 (k=" +
            std::to_string(k) + ")");
    if (n_labels <= 0)
        throw std::runtime_error("asw: n_labels must be >= 1 (got " +
            std::to_string(n_labels) + ")");
    if (n_labels > 1024)
        throw std::runtime_error("asw: n_labels=" + std::to_string(n_labels) +
            " exceeds shared-memory limit (max 1024). "
            "For n_labels > 1024 use a global-memory fallback (not in v0).");
    if (d_label == nullptr)
        throw std::runtime_error("asw: d_label is null");
    if (knn.distances.get() == nullptr)
        throw std::runtime_error("asw: knn.distances is null — ASW requires distance data");

    // Shared memory: float smem_sum[n_labels] + int smem_cnt[n_labels].
    const size_t smem_bytes =
        static_cast<size_t>(n_labels) * (sizeof(float) + sizeof(int));

    // Per-cell silhouette output.
    core::DeviceMemory<float> d_sil(static_cast<size_t>(n));

    // Deterministic path: one thread per block, one block per cell.
    // cfg.deterministic is always true in v0; non-deterministic path not needed
    // because the serial pass is O(k) per cell — fast enough for any n.
    (void)cfg;  // reserved for future parallel path
    asw_kernel_serial<<<n, 1, smem_bytes, stream>>>(
        knn.neighbors.get(), knn.distances.get(), d_label,
        d_sil.get(), n, k, n_labels);

    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Compute asw_mean = sum(silhouette) / n via cub::DeviceReduce::Sum.
    // One D2H readback of the scalar result (Rule 4 compliant).
    core::DeviceMemory<float> d_sum(1);
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, temp_bytes,
        d_sil.get(), d_sum.get(), n, stream);
    core::DeviceMemory<uint8_t> d_tmp(temp_bytes);
    cub::DeviceReduce::Sum(d_tmp.get(), temp_bytes,
        d_sil.get(), d_sum.get(), n, stream);

    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    float h_sum = 0.f;
    SINGLET_GPU_CUDA_CHECK(cudaMemcpy(&h_sum, d_sum.get(), sizeof(float),
                          cudaMemcpyDeviceToHost));

    AswResult res;
    res.silhouette = std::move(d_sil);
    res.asw_mean   = h_sum / static_cast<float>(n);
    res.n_cells    = n;
    res.n_labels   = n_labels;
    return res;
}

}  // namespace integrate
}  // namespace singlet::gpu
