// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (Seurat FindNeighbors Jaccard SNN)
//
// graph/snn.h — Shared Nearest Neighbour graph construction via Jaccard overlap pruning.
//
// Algorithm (Seurat FindNeighbors, GPU-native):
//   For each directed edge (i, j) where j ∈ kNN(i):
//     jaccard(i,j) = |kNN(i) ∩ kNN(j)| / |kNN(i) ∪ kNN(j)|
//                  = intersection / (2k - intersection)
//   Drop edge if jaccard < prune_threshold (Seurat default: 1/15).
//   Output: CSR adjacency matrix with Jaccard weights.
//
// Implementation:
//   One CUDA thread per directed kNN edge (i,j).  Both kNN lists are already sorted
//   ascending by distance (from the radix sort in knn.h).  Sorted merge counts
//   intersection in O(k) per pair, all on device.  No host transfer between kNN and SNN.
//
// Input: KnnResult from compute_knn (device-resident, neighbor lists sorted by distance).
//        Neighbors are stored as int (int32), one row of k per cell.
//
// Output: SnnResult — CSR (row_offsets[n+1], col_indices[nnz], weights[nnz]).
//   The CSR is symmetric: both (i,j) and (j,i) are emitted if both survive pruning.
//   nnz is bounded by n*k (full kNN graph) and in practice much smaller.
//
// Memory budget (n=1M, k=15): input kNN 120 MB + output CSR up to 120 MB.
//   Atomic counter for nnz: 4 bytes (device-resident, one cudaMemcpy scalar at the end).
// Streams: 1, caller-provided.
// Precision: fp32 Jaccard weights.
// Determinism: deterministic — intersection counting is order-independent for sorted lists.
// OOC: SNN operates row-by-row via thread blocks; works for any n that fits in device memory.
//   For streaming, run SNN per shard and merge CSRs on host (O(nnz) merge, not implemented here).
//
// CYCLE-62: initial implementation (adopted alongside CAGRA kNN).

#pragma once

#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/graph/knn.h>

namespace singlet_gpu {
namespace graph {

// ─── Public types ─────────────────────────────────────────────────────────────

struct SnnConfig {
    // Seurat default: 1/15 ≈ 0.0667.  Only edges with Jaccard ≥ threshold are kept.
    float prune_threshold = 1.0f / 15.0f;
};

struct SnnResult {
    core::DeviceMemory<int>   row_offsets;  // CSR indptr, size n+1
    core::DeviceMemory<int>   col_indices;  // CSR indices, size nnz
    core::DeviceMemory<float> weights;      // Jaccard similarity, size nnz
    int n;
    int nnz;  // actual edge count after pruning
};

// ─── Device kernels ───────────────────────────────────────────────────────────

namespace detail {

// Phase 1: count how many kNN(i) edges survive Jaccard pruning.
//   out_count[i] = number of j ∈ kNN(i) with jaccard(i,j) ≥ threshold.
//   One thread per cell i; iterates over its k neighbors j and counts intersection
//   via sorted merge on kNN(i) and kNN(j).
//
// Both neighbor rows are sorted ascending (guaranteed by cub::DeviceSegmentedRadixSort
// in knn.h Exact backend; CAGRA also returns sorted lists per its spec).
__global__ void
snn_count_kernel(const int* __restrict__ neighbors,   // n*k row-major
                 int* __restrict__ out_count,          // n output: edges kept per row
                 int n, int k, float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int* row_i = neighbors + (size_t)i * k;
    int kept = 0;

    for (int jj = 0; jj < k; ++jj) {
        int j = row_i[jj];
        if (j < 0 || j >= n) continue;  // padding guard (k > n-1 edge case)

        const int* row_j = neighbors + (size_t)j * k;

        // Sorted merge to count intersection of kNN(i) and kNN(j).
        int pi = 0, pj = 0, inter = 0;
        while (pi < k && pj < k) {
            int vi = row_i[pi];
            int vj = row_j[pj];
            if (vi < 0) { ++pi; continue; }
            if (vj < 0) { ++pj; continue; }
            if (vi == vj) { ++inter; ++pi; ++pj; }
            else if (vi < vj) { ++pi; }
            else              { ++pj; }
        }
        // |union| = 2k - intersection  (each set has exactly k valid entries;
        // for k > n-1 edge case, count valid entries explicitly).
        float jac = (float)inter / (float)(2 * k - inter);
        if (jac >= threshold) ++kept;
    }
    out_count[i] = kept;
}

// Phase 2: write surviving edges into CSR col_indices and weights arrays.
//   row_offsets[i] must already be filled (exclusive prefix sum of out_count).
//   Uses an atomic per-row counter stored in a temporary array to scatter within
//   each row without row-level serialization across threads.
//
// Because each thread i writes to row_offsets[i]..row_offsets[i+1], there is
// zero contention between threads for different i — no global atomics needed.
__global__ void
snn_write_kernel(const int* __restrict__ neighbors,    // n*k
                 const int* __restrict__ row_offsets,  // n+1, from prefix scan
                 int* __restrict__ col_indices,         // nnz output
                 float* __restrict__ weights,           // nnz output
                 int n, int k, float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int* row_i = neighbors + (size_t)i * k;
    int pos = row_offsets[i];  // write cursor for row i (no atomic needed)

    for (int jj = 0; jj < k; ++jj) {
        int j = row_i[jj];
        if (j < 0 || j >= n) continue;

        const int* row_j = neighbors + (size_t)j * k;

        // Sorted merge — identical to snn_count_kernel but writes output.
        int pi = 0, pj = 0, inter = 0;
        while (pi < k && pj < k) {
            int vi = row_i[pi];
            int vj = row_j[pj];
            if (vi < 0) { ++pi; continue; }
            if (vj < 0) { ++pj; continue; }
            if (vi == vj) { ++inter; ++pi; ++pj; }
            else if (vi < vj) { ++pi; }
            else              { ++pj; }
        }
        float jac = (float)inter / (float)(2 * k - inter);
        if (jac >= threshold) {
            col_indices[pos] = j;
            weights[pos]     = jac;
            ++pos;
        }
    }
}

}  // namespace detail

// ─── Public entry point ───────────────────────────────────────────────────────

// Compute a Shared Nearest Neighbour graph from an existing KnnResult.
//
// knn:    KnnResult from compute_knn — device-resident, neighbors sorted ascending.
// cfg:    SnnConfig (default prune_threshold = 1/15 per Seurat).
// stream: caller-provided CUDA stream (nullptr = default stream).
//
// Returns SnnResult with device-resident CSR: row_offsets[n+1], col_indices[nnz],
//   weights[nnz] (Jaccard similarity).  The graph is directed (both (i,j) and (j,i)
//   are emitted independently when both survive pruning, consistent with Seurat output).
//
// No host↔device transfers between kNN output and SNN construction — all three
// phases (count, prefix scan, write) are device-resident.  The only scalar that
// hits the host is nnz (row_offsets[n]), copied once at the end to size the result.
inline SnnResult
compute_snn(const KnnResult& knn, const SnnConfig& cfg = {}, cudaStream_t stream = nullptr)
{
    const int n = knn.n;
    const int k = knn.k;
    if (n <= 0 || k <= 0)
        throw std::invalid_argument("compute_snn: empty KnnResult");

    // Phase 1: count surviving edges per row.
    core::DeviceMemory<int> d_count(n);
    {
        int b = 256, g = (n + b - 1) / b;
        detail::snn_count_kernel<<<g, b, 0, stream>>>(
            knn.neighbors.get(), d_count.get(), n, k, cfg.prune_threshold);
    }

    // Phase 2: exclusive prefix sum → row_offsets[0..n].
    core::DeviceMemory<int> d_offsets(n + 1);
    {
        // cub::DeviceScan::ExclusiveSum requires a temp buffer; query its size first.
        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes,
                                      d_count.get(), d_offsets.get(),
                                      n + 1, stream);
        core::DeviceMemory<uint8_t> tmp(tmp_bytes ? tmp_bytes : 1);
        cub::DeviceScan::ExclusiveSum(tmp.get(), tmp_bytes,
                                      d_count.get(), d_offsets.get(),
                                      n + 1, stream);
    }

    // Phase 3: copy nnz (= row_offsets[n]) from device to host — one scalar copy.
    int nnz = 0;
    cudaMemcpyAsync(&nnz, d_offsets.get() + n, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);  // wait for nnz before allocating output buffers

    // Allocate output CSR arrays.
    core::DeviceMemory<int>   d_col(nnz > 0 ? nnz : 1);  // avoid zero-size alloc
    core::DeviceMemory<float> d_wts(nnz > 0 ? nnz : 1);

    // Phase 4: write surviving edges.
    if (nnz > 0) {
        int b = 256, g = (n + b - 1) / b;
        detail::snn_write_kernel<<<g, b, 0, stream>>>(
            knn.neighbors.get(), d_offsets.get(),
            d_col.get(), d_wts.get(),
            n, k, cfg.prune_threshold);
    }

    SnnResult res;
    res.row_offsets = std::move(d_offsets);
    res.col_indices = std::move(d_col);
    res.weights     = std::move(d_wts);
    res.n           = n;
    res.nnz         = nnz;
    return res;
}

}  // namespace graph
}  // namespace singlet_gpu
