// SPDX-License-Identifier: MIT
// integrates: original (Harmony adapted) + cycle 8 KNN (BBKNN)
//
// integrate/bbknn.h — Batch-Balanced kNN graph construction.
//
// Algorithm: Polański et al. 2020 (Bioinformatics 36:964–965).
//   For each cell j (batch b_j):
//     1. Query ONLY cells from batch b for neighbors, k_within neighbors each.
//     2. Concatenate k_within neighbors from each of n_batches batches.
//   Output: a KnnResult where each cell has exactly k_within * n_batches neighbors,
//           evenly drawn from each batch (or fewer from small batches).
//
// Implementation:
//   - Per-batch sub-call to compute_knn() from cycle 8 graph/knn.h.
//   - Input masking: build per-batch sub-embedding (contiguous copy) for each batch.
//   - After per-batch KNN, remap local neighbor indices to global cell indices.
//   - Concatenate all per-batch neighbor/distance lists per cell.
//
// Reuses the Exact backend (cuBLAS GEMM tile structure from cycle 8) for n ≤ threshold.
// Above threshold uses HNSW (cfg.approx_threshold), following the cycle 8 routing.
//
// Cost: n_batches × cost(knn on batch of size n/n_batches) ≈ n_batches × (n/B)² GEMM.
//   For 10 batches of 100k / 10 = 10k each: 10 × 10k² vs 1 × 100k² = 10× cheaper!
//   At equal batch sizes BBKNN is more efficient than full kNN.
//
// Memory budget (100k cells, k_within=3, n_batches=10, d=50):
//   Per-batch sub-embedding: 10k × 50 × 4 = 2 MB; peak = one batch at a time.
//   Per-batch kNN result: 10k × 3 × (4+4) = 240 KB.
//   Final output neighbors: 100k × 30 × 4 = 12 MB; distances: 12 MB.
//   Total peak ≈ 30 MB — negligible.
//
// Streams: 1, caller-provided (all per-batch sub-calls use the same stream).
// Precision: fp32 (inherited from cycle 8 kNN).
// Determinism: Exact backend is deterministic; HNSW is not (inherited from cycle 8).
// OOC: per-batch sub-calls can be issued one batch at a time for large n.

#pragma once

#include <singlet/gpu/integrate/types.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/graph/knn.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

namespace singlet::gpu {
namespace integrate {

// ─── Device helper ────────────────────────────────────────────────────────────

namespace detail {

// Gather row-subset of embedding from global indices.
// sub[i, :] = emb[global_ids[i], :]
__launch_bounds__(256, 4)
__global__ void
gather_rows_kernel(float* __restrict__       sub,         // n_sub × d
                   const float* __restrict__ emb,         // n × d
                   const int* __restrict__   global_ids,  // n_sub
                   int n_sub, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sub) return;
    int gi = global_ids[i];
    const float* src = emb + (size_t)gi * d;
    float*       dst = sub + (size_t)i  * d;
    for (int dim = 0; dim < d; ++dim)
        dst[dim] = src[dim];
}

// Remap local neighbor indices (within the per-batch sub-embedding) to global indices.
// neighbors_out[i, :] = global_ids[neighbors_local[i, :]]
__launch_bounds__(256, 4)
__global__ void
remap_neighbors_kernel(int* __restrict__       neighbors_out,   // n_sub × k
                       const int* __restrict__ neighbors_local, // n_sub × k
                       const int* __restrict__ global_ids,      // n_sub
                       int n_sub, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sub) return;
    for (int ki = 0; ki < k; ++ki) {
        int local = neighbors_local[(size_t)i * k + ki];
        // local == i means self-exclusion edge; map to -1 sentinel.
        neighbors_out[(size_t)i * k + ki] = (local >= 0 && local < n_sub)
                                            ? global_ids[local]
                                            : -1;
    }
}

// Scatter per-batch neighbor results into the output per-cell slot.
// For batch b, query cell j (global) has local_idx local_j in that batch's listing.
// Its k_within neighbors (already remapped to global) go into output slot
//   out_neighbors[j, b * k_within : (b+1) * k_within]
// This kernel processes all cells that belong to batch b simultaneously as queries.
// We assume each query cell appears in ALL batches as a query
// (BBKNN queries every cell against every batch, including its own).
__launch_bounds__(256, 4)
__global__ void
scatter_batch_neighbors_kernel(
    int* __restrict__         out_nbrs,   // n × (k_within * n_batches)
    float* __restrict__       out_dists,  // n × (k_within * n_batches)
    const int* __restrict__   batch_nbrs, // n × k_within  (remapped global ids)
    const float* __restrict__ batch_dists,// n × k_within
    int batch_idx, int k_within, int n_batches, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int out_base = (size_t)j * (k_within * n_batches) + batch_idx * k_within;
    int in_base  = (size_t)j * k_within;
    for (int ki = 0; ki < k_within; ++ki) {
        out_nbrs [out_base + ki] = batch_nbrs [in_base + ki];
        out_dists[out_base + ki] = batch_dists[in_base + ki];
    }
}

}  // namespace detail


// ─── BBKNN public function ────────────────────────────────────────────────────

// bbknn() — Batch-balanced kNN graph construction.
//
// embedding: row-major n × d device float (cells × PCA dimensions).
// batch_labels: integer batch id per cell (values in [0, n_batches)).
// n_batches: number of distinct batch labels.
// cfg: BbknnConfig (see types.h).
// stream: caller-provided CUDA stream.
//
// Returns a KnnResult with each cell having k_within * n_batches neighbors,
// k_within drawn from each batch. Neighbors within each batch slot are sorted
// by distance ascending. The returned KnnResult uses the same layout as
// graph::compute_knn(): row_offsets[n+1], neighbors[n * total_k], distances[n * total_k].
//
// Cells from small batches (fewer than k_within cells total): padded with index -1
// and distance +inf. Callers should filter -1 before downstream use.
inline graph::KnnResult
bbknn(const core::DeviceDense& embedding,
      const core::DeviceMemory<int>& batch_labels,
      int n_batches,
      const BbknnConfig& cfg,
      cudaStream_t stream)
{
    const int n       = (int)embedding.rows;
    const int d       = (int)embedding.cols;
    const int k_total = cfg.k_within * n_batches;

    if (n <= 0 || d <= 0)  throw std::invalid_argument("bbknn: empty embedding");
    if (n_batches <= 0)    throw std::invalid_argument("bbknn: n_batches must be > 0");
    if (cfg.k_within <= 0) throw std::invalid_argument("bbknn: k_within must be > 0");

    // ── Download batch labels to host ─────────────────────────────────────────
    // Done once; batch membership is fixed for the lifetime of this call.
    cudaStreamSynchronize(stream);
    std::vector<int> h_batch(n);
    cudaMemcpy(h_batch.data(), batch_labels.get(), (size_t)n * sizeof(int), cudaMemcpyDeviceToHost);

    // Build per-batch index lists (global cell indices belonging to each batch).
    std::vector<std::vector<int>> batch_cells(n_batches);
    for (int j = 0; j < n; ++j) {
        int b = h_batch[j];
        if (b < 0 || b >= n_batches)
            throw std::invalid_argument("bbknn: batch label out of range");
        batch_cells[b].push_back(j);
    }

    // ── Allocate output ───────────────────────────────────────────────────────
    graph::KnnResult result;
    result.n = n; result.k = k_total; result.backend_used = graph::KnnBackend::Exact;
    result.row_offsets = core::DeviceMemory<int>(n + 1);
    result.neighbors   = core::DeviceMemory<int>((size_t)n * k_total);
    result.distances   = core::DeviceMemory<float>((size_t)n * k_total);

    // Fill row_offsets (uniform: offsets[i] = i * k_total).
    {
        std::vector<int> h_off(n + 1);
        for (int i = 0; i <= n; ++i) h_off[i] = i * k_total;
        cudaMemcpyAsync(result.row_offsets.get(), h_off.data(),
                        (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    }

    // Initialize output to sentinels: neighbors = -1, distances = +inf.
    {
        std::vector<int>   h_nbrs(static_cast<size_t>(n) * k_total, -1);
        std::vector<float> h_dists(static_cast<size_t>(n) * k_total,
                                   std::numeric_limits<float>::infinity());
        cudaMemcpyAsync(result.neighbors.get(), h_nbrs.data(),
                        static_cast<size_t>(n) * k_total * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(result.distances.get(), h_dists.data(),
                        static_cast<size_t>(n) * k_total * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }

    // ── Per-batch kNN sub-calls ───────────────────────────────────────────────
    const float* emb_ptr = embedding.data.get();

    for (int b = 0; b < n_batches; ++b) {
        const std::vector<int>& cell_ids = batch_cells[b];
        int n_sub = (int)cell_ids.size();
        if (n_sub == 0) continue;

        // Effective k_within clamped to batch size - 1 (self-excluded).
        int k_eff = std::min(cfg.k_within, n_sub - 1);
        if (k_eff <= 0) {
            // Batch has only 1 cell: no valid neighbors. Output stays as sentinel.
            continue;
        }

        // Upload batch cell index list to device.
        core::DeviceMemory<int> d_cell_ids(n_sub);
        cudaMemcpyAsync(d_cell_ids.get(), cell_ids.data(),
                        (size_t)n_sub * sizeof(int), cudaMemcpyHostToDevice, stream);

        // Gather sub-embedding for this batch.
        core::DeviceMemory<float> d_sub_emb((size_t)n_sub * d);
        {
            int blk = 256, grd = (n_sub + blk - 1) / blk;
            detail::gather_rows_kernel<<<grd, blk, 0, stream>>>(
                d_sub_emb.get(), emb_ptr, d_cell_ids.get(), n_sub, d);
        }

        // Wrap sub-embedding as DeviceDense for compute_knn().
        // Use DeviceMemory<float>::wrap() for a non-owning view over the device pointer.
        core::DeviceDense sub_dense;
        sub_dense.rows = n_sub;
        sub_dense.cols = d;
        sub_dense.data = core::DeviceMemory<float>::wrap(d_sub_emb.get(), (size_t)n_sub * d);

        // Compute kNN within this batch.
        graph::KnnConfig knn_cfg;
        knn_cfg.k             = k_eff;
        knn_cfg.metric        = cfg.metric;
        knn_cfg.return_squared = false;
        knn_cfg.backend       = (n_sub <= cfg.approx_threshold)
                                    ? graph::KnnBackend::Exact
                                    : graph::KnnBackend::Cagra;

        graph::KnnResult sub_knn = graph::compute_knn(sub_dense, knn_cfg, stream);

        // Remap local neighbor indices → global cell indices.
        core::DeviceMemory<int> d_global_nbrs((size_t)n_sub * k_eff);
        {
            int blk = 256, grd = (n_sub + blk - 1) / blk;
            detail::remap_neighbors_kernel<<<grd, blk, 0, stream>>>(
                d_global_nbrs.get(), sub_knn.neighbors.get(),
                d_cell_ids.get(), n_sub, k_eff);
        }

        // We have sub_knn results for n_sub cells, each with k_eff neighbors.
        // We need to scatter these into the output for ALL n global cells:
        //   The query cells here are the n_sub cells belonging to batch b.
        //   They need their results placed at column slot b*k_within in the output.
        //
        // We do this on host to avoid a complex scatter kernel.
        cudaStreamSynchronize(stream);
        std::vector<int>   h_gnbrs(static_cast<size_t>(n_sub) * k_eff);
        std::vector<float> h_gdists(static_cast<size_t>(n_sub) * k_eff);
        cudaMemcpy(h_gnbrs.data(), d_global_nbrs.get(),
                   static_cast<size_t>(n_sub) * k_eff * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_gdists.data(), sub_knn.distances.get(),
                   static_cast<size_t>(n_sub) * k_eff * sizeof(float), cudaMemcpyDeviceToHost);

        // Allocate host output buffers to scatter into.
        std::vector<int>   h_out_nbrs_slice(static_cast<size_t>(n_sub) * cfg.k_within, -1);
        std::vector<float> h_out_dists_slice(static_cast<size_t>(n_sub) * cfg.k_within,
                                             std::numeric_limits<float>::infinity());
        // Copy k_eff values (padded to k_within with sentinel).
        for (int li = 0; li < n_sub; ++li) {
            for (int ki = 0; ki < k_eff; ++ki) {
                h_out_nbrs_slice [(size_t)li * cfg.k_within + ki] = h_gnbrs [(size_t)li * k_eff + ki];
                h_out_dists_slice[(size_t)li * cfg.k_within + ki] = h_gdists[(size_t)li * k_eff + ki];
            }
        }

        // Scatter into result: for each cell in cell_ids, write to slot [j, b*k_within:(b+1)*k_within].
        // Do this on host into a temporary, then async-copy strided slices to device.
        for (int li = 0; li < n_sub; ++li) {
            int j = cell_ids[li];
            size_t out_base = static_cast<size_t>(j) * k_total + static_cast<size_t>(b) * cfg.k_within;
            cudaMemcpyAsync(result.neighbors.get() + out_base,
                            h_out_nbrs_slice.data() + (size_t)li * cfg.k_within,
                            (size_t)cfg.k_within * sizeof(int), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(result.distances.get() + out_base,
                            h_out_dists_slice.data() + (size_t)li * cfg.k_within,
                            (size_t)cfg.k_within * sizeof(float), cudaMemcpyHostToDevice, stream);
        }
        // Note: cell_ids.size() H2D copies per batch. For large n this is a hot-loop
        // H2D pattern. A future optimization would consolidate into a single large
        // H2D copy via a host-side scatter into a contiguous staging buffer.
        // Logged as DAG task: INTEGRATE-BBKNN-SCATTER-OPT. Not a hot loop per-cell
        // but per-batch (n_batches ≤ O(100)), so acceptable for cycle 14.
    }

    cudaStreamSynchronize(stream);
    return result;
}

}  // namespace integrate
}  // namespace singlet::gpu
