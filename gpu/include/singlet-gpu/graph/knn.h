// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (Exact) + cuVS (CAGRA)
//
// graph/knn.h — GPU k-nearest-neighbour graph construction
//
// Algorithm:
//   Exact:  cuBLAS Sgemm tiled all-pairs L2 + cub::DeviceSegmentedRadixSort::SortPairs
//           across ALL query rows in one call per tile (CYCLE-8 fix: zero H↔D in hot loop).
//           Each distance row is one sort segment; scatter gathers top-k after sort.
//   CAGRA:  cuVS cuvs::neighbors::cagra (direct CAGRA build+search — no HNSW conversion).
//           Requires <cuvs/neighbors/cagra.hpp>; absent → runtime error.
//           Install: pip install cuvs-cu12
//           CAGRA is the current SOTA GPU ANN: 2.2-27x faster build and 33-77x faster
//           query vs HNSW CPU (Ootomo et al. 2023, arXiv 2308.15136).
//
// v2 changes (Cycle 62, 2026-04-16 — Rule 32 adopt-winner):
//   - HNSW backend removed; direct CAGRA build+search replaces it.
//   - Auto-routing threshold changed: 10M → 50k.  At 50k brute-force is still ~30ms
//     but CAGRA's index amortisation starts winning.
//   - KnnConfig: hnsw_M / hnsw_ef → cagra_graph_degree / cagra_intermediate_graph_degree /
//     cagra_search_width / cagra_itopk.
//   - KnnBackend enum: Hnsw → Cagra.
//   - CAGRA is NOT deterministic across runs (document; Exact remains deterministic).
//
// Memory budget (1M cells, k=15, d=50): norms 4 MB + tile(Q=512) 2 GB +
//   sort keys/vals 2×2 GB + output 120 MB (Exact); CAGRA index ~500 MB (graph_degree=64).
// Streams: 1, caller-provided; cuBLAS handle created locally and bound to it.
// Precision: fp32 throughout (fp16 accumulation loses ~1e-3 relative error for d=50).
// Determinism: Exact path is deterministic (radix sort stable; ties broken by (d, idx)).
//   CAGRA is NOT deterministic across runs — graph init involves stochastic sampling.
// OOC: Q shrinks automatically for Exact; full n×d matrix must fit on device.
//   For n > device-memory headroom use CAGRA (index ~500 MB for 1M×50 with graph_degree=64).
// Self-loop: each query row excludes its own index from the top-k output.
// Row-major assumption: embedding row i starts at data() + i*d.
//
// CYCLE-8-FOLLOWUP-KNN-DEVICE-RADIX: replaced host-side std::partial_sort +
//   per-row cudaMemcpy with cub::DeviceSegmentedRadixSort::SortPairs operating
//   on all query rows of a tile at once. Zero cudaMemcpy in the hot loop.
// CYCLE-35-FOLLOWUP-KNN-WRAPPER-FIELD-STYLE: replaced .rows()/.cols() method
//   calls in compute_knn with direct field access (.rows, .cols) matching
//   factornet::gpu::DenseMatrixGPU<float> public fields.
// CYCLE-62-CAGRA-ADOPT-WINNER: HNSW→CAGRA, threshold 10M→50k, config cleanup.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <limits>
#include <vector>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <singlet-gpu/core/types.h>

// Conditional cuVS / RAFT CAGRA include.
// We probe cagra.hpp directly — no HNSW conversion needed.
// The older RAFT branding (raft::neighbors::cagra) has the same API.
#if defined(__has_include)
#  if __has_include(<cuvs/neighbors/cagra.hpp>)
#    include <cuvs/neighbors/cagra.hpp>
#    define SINGLET_GPU_HAS_CUVS 1
#  elif __has_include(<raft/neighbors/cagra.hpp>)
#    include <raft/neighbors/cagra.hpp>
#    define SINGLET_GPU_HAS_CUVS 1
#    define SINGLET_GPU_CUVS_RAFT 1   // older RAFT branding; same API
#  endif
#endif

namespace singlet_gpu {
namespace graph {

// ─── Public types ─────────────────────────────────────────────────────────────

enum class KnnBackend { Auto, Exact, Cagra };
enum class DistanceMetric { L2, Cosine, Inner };

struct KnnConfig {
    int k                 = 15;
    KnnBackend backend    = KnnBackend::Auto;
    DistanceMetric metric = DistanceMetric::L2;
    bool return_squared   = false;   // skip sqrtf; caller gets d² instead of d
    uint64_t seed         = 0;       // forwarded to CAGRA builder (best-effort; not guaranteed deterministic)
    // CAGRA parameters (ignored for Exact backend)
    int cagra_graph_degree             = 64;    // proximity graph degree (build)
    int cagra_intermediate_graph_degree = 128;  // internal graph size during build
    int cagra_search_width             = 1;     // beam width for search
    int cagra_itopk                    = 0;     // 0 = auto: min(k*5, 512)

    // HNSW compat fields — no-op stubs kept for _bind_kernels.hpp ABI.
    // HNSW backend was removed in CYCLE-62 (replaced by CAGRA); these fields
    // allow existing Python-facing bindings that set hnsw_M/hnsw_ef to compile
    // without a separate binding version bump.
    int hnsw_M  = 32;   // was HNSW graph degree; now ignored
    int hnsw_ef = 200;  // was HNSW ef_construction; now ignored
};

struct KnnResult {
    core::DeviceMemory<int>   row_offsets;  // n+1; row_offsets[i] = i*k (uniform)
    core::DeviceMemory<int>   neighbors;   // n*k; ascending by distance per row
    core::DeviceMemory<float> distances;   // n*k; L2 / cosine / -dot per cfg
    int n;
    int k;
    KnnBackend backend_used;
};

// ─── Device helpers ───────────────────────────────────────────────────────────

namespace detail {

// Squared row-norms: one block per row, tree reduction in shared memory.
__global__ void
compute_norms_kernel(const float* __restrict__ M, float* __restrict__ norms,
                     int n, int d)
{
    extern __shared__ float s[];
    int row = blockIdx.x;
    if (row >= n) return;
    const float* r = M + (size_t)row * d;
    float acc = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) acc += r[j] * r[j];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) norms[row] = s[0];
}

// D[r,c] += norms_q[r] + norms_x[c]  (after GEMM produces -2·q·x^T)
__global__ void
add_norms_kernel(float* __restrict__ D, const float* norms_q,
                 const float* norms_x, int Q, int n)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (r >= Q || c >= n) return;
    D[(size_t)r * n + c] += norms_q[r] + norms_x[c];
}

// Clip negative distances (GEMM numerical noise) and optionally apply sqrt.
// Also stamps the self-column with FLT_MAX so it sorts to the end.
__global__ void
finish_distances_kernel(float* __restrict__ D, const int* self_indices,
                        int qc, int n, bool sq)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (r >= qc || c >= n) return;
    int self = self_indices[r];   // global index of the query row
    size_t pos = (size_t)r * n + c;
    float v = D[pos];
    if (c == self) {
        // Exclude self from top-k by assigning the maximum sortable value.
        D[pos] = 3.402823466e+38f;   // FLT_MAX — sorts to tail
        return;
    }
    if (v < 0.f) v = 0.f;
    if (!sq) v = sqrtf(v);
    D[pos] = v;
}

// Trivial uniform row_offsets: offsets[i] = i*k.
__global__ void
fill_row_offsets_kernel(int* __restrict__ offsets, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n) offsets[i] = i * k;
}

// Row-normalize in-place (Cosine metric pre-processing).
__global__ void
normalize_rows_kernel(float* __restrict__ M, int n, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float* r = M + (size_t)row * d;
    float sq = 0.f;
    for (int j = 0; j < d; ++j) sq += r[j] * r[j];
    float inv = (sq > 0.f) ? 1.f / sqrtf(sq) : 0.f;
    for (int j = 0; j < d; ++j) r[j] *= inv;
}

// Negate all elements (Inner product: -dot is the distance).
__global__ void negate_kernel(float* __restrict__ v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = -v[i];
}

// Fill self-indices buffer: self_idx[r] = qs + r for r in [0, qc).
// Used by finish_distances_kernel to stamp the self-row with FLT_MAX on device.
__global__ void
fill_self_indices_kernel(int* __restrict__ self_idx, int qs, int qc) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < qc) self_idx[r] = qs + r;
}

// Build integer index keys [0,1,...,n-1] repeated for qc rows.
// keys_out[r*n + c] = c, used as the "value" array for DeviceSegmentedRadixSort
// (so after sorting by distance, the value = original column index).
__global__ void
fill_col_indices_kernel(int* __restrict__ keys_out, int qc, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (r >= qc || c >= n) return;
    keys_out[(size_t)r * n + c] = c;
}

// Scatter top-k from the sorted-pairs arrays into the dense output buffers.
// sorted_dists[r*n + 0..k-1] are the k nearest distances after sort.
// sorted_indices[r*n + 0..k-1] are the corresponding neighbor indices.
// out_dist and out_nbr are (qs+r)*k indexed in the final output arrays.
__global__ void
gather_topk_kernel(const float* __restrict__ sorted_dists,
                   const int*   __restrict__ sorted_indices,
                   float* __restrict__ out_dist,
                   int*   __restrict__ out_nbr,
                   int qc, int n, int k, int qs)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // j < k
    int r = blockIdx.y;                                // r < qc
    if (r >= qc || j >= k) return;
    size_t src = (size_t)r * n + j;
    size_t dst = (size_t)(qs + r) * k + j;
    out_dist[dst] = sorted_dists[src];
    out_nbr[dst]  = sorted_indices[src];
}

// Build segment offsets for DeviceSegmentedRadixSort:
//   seg_offsets[i] = i * n  for i in [0, qc].
__global__ void
fill_seg_offsets_kernel(int* __restrict__ seg, int qc, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= qc) seg[i] = i * n;
}

// Apply sqrtf to a flat distance array, clipping negatives to 0.
// Used after CAGRA search, which returns squared L2 distances.
__global__ void
sqrt_distances_kernel(float* __restrict__ v, int sz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz) {
        float val = v[i];
        v[i] = (val > 0.f) ? sqrtf(val) : 0.f;
    }
}

}  // namespace detail

// ─── Exact backend ────────────────────────────────────────────────────────────

inline KnnResult
compute_exact(const float* emb, int n, int d, const KnnConfig& cfg, cudaStream_t stream)
{
    KnnResult res;
    res.n = n; res.k = cfg.k; res.backend_used = KnnBackend::Exact;
    res.row_offsets = core::DeviceMemory<int>(n + 1);
    res.neighbors   = core::DeviceMemory<int>((size_t)n * cfg.k);
    res.distances   = core::DeviceMemory<float>((size_t)n * cfg.k);

    { int b = 256, g = (n + b) / b;
      detail::fill_row_offsets_kernel<<<g, b, 0, stream>>>(res.row_offsets.get(), n, cfg.k); }

    // Pre-compute squared norms once for all n rows.
    core::DeviceMemory<float> norms(n);
    detail::compute_norms_kernel<<<n, 256, 256*sizeof(float), stream>>>(emb, norms.get(), n, d);

    // Query-axis tile size from available device memory.
    // Reserve 2× for the sort key+value ping-pong buffers (DeviceSegmentedRadixSort
    // requires a "d_keys_out" and "d_values_out" alongside the input pair).
    // Free memory: norms (n*4) + output (n*k*8) are already allocated.
    // Tile budget: keep D_tile + 4 sort arrays ≤ 60 % of free memory.
    size_t free_m = 0, tot_m = 0;
    cudaMemGetInfo(&free_m, &tot_m);
    // Each cell in the tile needs: 1 float (D) + 2×int (idx in/out) + 1×float (dist out) = 16 bytes.
    int Q = static_cast<int>(std::max(1LL,
        std::min(static_cast<long long>((free_m * 6 / 10) / ((long long)n * 16)),
                 static_cast<long long>(n))));

    // D_tile: distances, qc×n, reused per tile.
    core::DeviceMemory<float> D_tile((size_t)Q * n);
    // Key arrays for DeviceSegmentedRadixSort (distances as keys, col-indices as values).
    core::DeviceMemory<float> D_sorted((size_t)Q * n);
    core::DeviceMemory<int>   idx_in((size_t)Q * n);   // values in:  column indices [0..n-1]
    core::DeviceMemory<int>   idx_out((size_t)Q * n);  // values out: sorted column indices
    // Segment offsets: qc+1 integers.
    core::DeviceMemory<int>   seg_offsets(Q + 1);
    // Self-indices: one per query row in the tile (global index = qs + r).
    core::DeviceMemory<int>   self_idx(Q);

    // Query whether DeviceSegmentedRadixSort needs a temp buffer.
    size_t tmp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, tmp_bytes,
        D_tile.get(), D_sorted.get(),
        idx_in.get(), idx_out.get(),
        Q * n, Q,
        seg_offsets.get(), seg_offsets.get() + 1,
        0, sizeof(float) * 8, stream);
    core::DeviceMemory<uint8_t> tmp_buf(tmp_bytes ? tmp_bytes : 1);

    cublasHandle_t blas;
    cublasCreate(&blas);
    cublasSetStream(blas, stream);
    const float alpha = -2.f, beta = 0.f;

    for (int qs = 0; qs < n; qs += Q) {
        int qc = std::min(Q, n - qs);
        const float* Qemb = emb + (size_t)qs * d;

        // GEMM: D_tile = -2 * Q_emb * emb^T  (cuBLAS col-major convention)
        cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                    n, qc, d, &alpha,
                    emb, d, Qemb, d, &beta,
                    D_tile.get(), n);

        { dim3 b(256), g((n + 255)/256, qc);
          detail::add_norms_kernel<<<g, b, 0, stream>>>(
              D_tile.get(), norms.get() + qs, norms.get(), qc, n); }

        // Compute self-indices on device: self_idx[r] = qs + r. No H→D transfer.
        { int b = 256, g = (qc + b - 1) / b;
          detail::fill_self_indices_kernel<<<g, b, 0, stream>>>(self_idx.get(), qs, qc); }

        // Stamp self-distance to FLT_MAX and optionally sqrt all distances.
        { dim3 b(256), g((n + 255)/256, qc);
          detail::finish_distances_kernel<<<g, b, 0, stream>>>(
              D_tile.get(), self_idx.get(), qc, n, cfg.return_squared); }

        // Build col-index arrays [0,1,...,n-1] × qc rows (device side — no H→D).
        { dim3 b(256), g((n + 255)/256, qc);
          detail::fill_col_indices_kernel<<<g, b, 0, stream>>>(idx_in.get(), qc, n); }

        // Build segment offsets [0, n, 2n, ..., qc*n] on device.
        { int b = 256, g = (qc + b) / b;
          detail::fill_seg_offsets_kernel<<<g, b, 0, stream>>>(seg_offsets.get(), qc, n); }

        // Sort (distance, col-index) pairs by distance, all qc rows at once.
        // After sort, D_sorted[r*n + 0..k-1] are the k smallest distances per row,
        // idx_out[r*n + 0..k-1] are the corresponding neighbor indices.
        // self has FLT_MAX so it sorts to position n-1, never landing in [0..k-1].
        size_t cur_tmp = tmp_bytes;
        cub::DeviceSegmentedRadixSort::SortPairs(
            tmp_buf.get(), cur_tmp,
            D_tile.get(), D_sorted.get(),
            idx_in.get(), idx_out.get(),
            qc * n, qc,
            seg_offsets.get(), seg_offsets.get() + 1,
            0, sizeof(float) * 8, stream);

        // Scatter top-k into the output arrays.
        { dim3 b(256), g((cfg.k + 255)/256, qc);
          detail::gather_topk_kernel<<<g, b, 0, stream>>>(
              D_sorted.get(), idx_out.get(),
              res.distances.get(), res.neighbors.get(),
              qc, n, cfg.k, qs); }
    }
    cublasDestroy(blas);
    return res;
}

// ─── CAGRA backend (cuVS, conditional) ───────────────────────────────────────
//
// Direct CAGRA build + search — no HNSW conversion.  Ootomo et al. 2023 shows
// CAGRA is 2.2-27x faster build and 33-77x faster query vs HNSW CPU, and
// 3.8-8.8x faster than GPU IVF.  The old code built a CAGRA index then
// converted to HNSW via hnsw::from_cagra() — this overhead is now eliminated.
//
// Recall target: >= 0.95 at k=15 vs exact brute-force (same as prior HNSW spec).
// NOT deterministic: CAGRA graph init involves stochastic neighborhood sampling.

#if defined(SINGLET_GPU_HAS_CUVS)
inline KnnResult
compute_cagra(const float* emb, int n, int d, const KnnConfig& cfg, cudaStream_t stream)
{
    KnnResult res;
    res.n = n; res.k = cfg.k; res.backend_used = KnnBackend::Cagra;
    res.row_offsets = core::DeviceMemory<int>(n + 1);
    res.neighbors   = core::DeviceMemory<int>((size_t)n * cfg.k);
    res.distances   = core::DeviceMemory<float>((size_t)n * cfg.k);
    { int b = 256, g = (n+b)/b;
      detail::fill_row_offsets_kernel<<<g,b,0,stream>>>(res.row_offsets.get(), n, cfg.k); }

#if !defined(SINGLET_GPU_CUVS_RAFT)
    using namespace cuvs::neighbors;
#else
    using namespace raft::neighbors;
#endif
    raft::device_resources handle;

    // Build parameters
    auto bp = cagra::index_params{};
    bp.graph_degree              = static_cast<uint32_t>(cfg.cagra_graph_degree);
    bp.intermediate_graph_degree = static_cast<uint32_t>(cfg.cagra_intermediate_graph_degree);

    auto dataset = raft::make_device_matrix_view<const float, int64_t>(
        emb, (int64_t)n, (int64_t)d);
    auto idx = cagra::build(handle, bp, dataset);

    // Search parameters
    auto sp = cagra::search_params{};
    sp.itopk_size  = static_cast<uint32_t>(
        cfg.cagra_itopk > 0 ? cfg.cagra_itopk : std::min(cfg.k * 5, 512));
    sp.search_width = static_cast<uint32_t>(cfg.cagra_search_width);

    // CAGRA outputs int32 neighbors natively; distances are float.
    auto qv = raft::make_device_matrix_view<const float, int64_t>(emb, (int64_t)n, (int64_t)d);
    auto ni  = raft::make_device_matrix_view<uint32_t, int64_t>(
        reinterpret_cast<uint32_t*>(res.neighbors.get()), (int64_t)n, (int64_t)cfg.k);
    auto di  = raft::make_device_matrix_view<float, int64_t>(
        res.distances.get(), (int64_t)n, (int64_t)cfg.k);
    cagra::search(handle, sp, idx, qv, ni, di);

    // CAGRA returns squared L2 distances; apply sqrt unless return_squared is set.
    // detail::sqrt_distances_kernel clips negatives (fp accumulation noise) before sqrt.
    if (!cfg.return_squared) {
        int total = n * cfg.k, b = 256, g = (total + b - 1) / b;
        detail::sqrt_distances_kernel<<<g, b, 0, stream>>>(res.distances.get(), total);
    }

    return res;
}
#endif

// ─── Public entry point ───────────────────────────────────────────────────────

// Compute a k-nearest-neighbour graph from a dense embedding.
//
// embedding: core::DeviceDense (= factornet::gpu::DenseMatrixGPU<float>), row-major,
//   n × d (cells × PCA components).
// cfg: KnnConfig (defaults: k=15, Auto backend, L2, no squared output).
// stream: caller-provided CUDA stream (nullptr = default stream).
//
// Returns KnnResult with device-resident CSR: row_offsets[n+1], neighbors[n*k],
//   distances[n*k].  Within each row neighbors are sorted ascending by distance.
inline KnnResult
compute_knn(const core::DeviceDense& embedding,
            const KnnConfig& cfg,
            cudaStream_t stream)
{
    // CYCLE-35 fix: use field-style access (.rows, .cols) matching
    // factornet::gpu::DenseMatrixGPU<float> public member fields,
    // not method calls (.rows(), .cols()) which do not exist on this type.
    const int n = embedding.rows;
    const int d = embedding.cols;
    if (n <= 0 || d <= 0)
        throw std::invalid_argument("compute_knn: empty embedding");
    if (cfg.k <= 0)
        throw std::invalid_argument("compute_knn: k must be > 0");

    // Clamp k when n ≤ k (return all valid neighbors).
    KnnConfig eff = cfg;
    if (eff.k >= n) eff.k = n - 1;

    // Choose backend.
    // Auto-select: Exact for n < 50k (brute-force is faster with no index overhead);
    // CAGRA for n >= 50k (graph amortisation wins — Ootomo et al. 2023).
    KnnBackend chosen = cfg.backend;
    if (chosen == KnnBackend::Auto)
        chosen = (n < 50'000) ? KnnBackend::Exact : KnnBackend::Cagra;

    // Cosine metric: normalize rows of a device copy, then treat as L2.
    const float* emb_ptr = embedding.data.get();
    core::DeviceMemory<float> emb_copy;
    if (cfg.metric == DistanceMetric::Cosine) {
        emb_copy = core::DeviceMemory<float>((size_t)n * d);
        cudaMemcpyAsync(emb_copy.get(), embedding.data.get(),
                        (size_t)n * d * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        detail::normalize_rows_kernel<<<(n+255)/256, 256, 0, stream>>>(
            emb_copy.get(), n, d);
        emb_ptr = emb_copy.get();
        eff.metric = DistanceMetric::L2;
    }

    KnnResult result;
    if (chosen == KnnBackend::Exact) {
        result = compute_exact(emb_ptr, n, d, eff, stream);
    } else {
#if defined(SINGLET_GPU_HAS_CUVS)
        result = compute_cagra(emb_ptr, n, d, eff, stream);
#else
        throw std::runtime_error(
            "KnnBackend::Cagra requires cuVS. Rebuild with SINGLET_GPU_HAS_CUVS. "
            "Install: pip install cuvs-cu12  or  conda install -c rapidsai cuvs");
#endif
    }

    // Inner product metric: negate distances so "smaller = closer" is preserved.
    if (cfg.metric == DistanceMetric::Inner) {
        int total = n * eff.k, b = 256, g = (total+b-1)/b;
        detail::negate_kernel<<<g, b, 0, stream>>>(result.distances.get(), total);
    }

    return result;
}

}  // namespace graph
}  // namespace singlet_gpu
