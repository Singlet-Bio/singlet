// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (consensus bootstrap NMF; reuses factornet::nmf::fit_gpu via cycle 5 adapter)
//
// reduce/nmf/csi_gep.h — CSI-GEP: Consensus Single-Integration Gene Expression Programs
//
// Algorithm (Kotliar et al. 2019 eLife; geeleherlab/CSI-GEP reference implementation):
//   For each k in cfg.k_range:
//     1. Run n_runs independent NMF fits, each on a Philox-subsampled 80% cell subset.
//     2. Collect the n_runs * k candidate program columns (W[:,j], each m-length).
//     3. Spherical k-means on the n_runs*k stacked candidates → k consensus programs.
//     4. Mean max-Jaccard(top-30 genes, consensus vs per-run) → reproducibility score.
//   Auto-k: elbow detection via finite-difference on reproducibility curve.
//   NNLS projection: program_usage[n_cells × chosen_k] = NNLS(X, consensus[chosen_k]).
//
// cudaMemcpy audit — confirmed no PCIe transfer in any loop > 5 iterations:
//   Per-run (outer run loop, up to n_runs=100 iters):
//     - Philox mask kernel: device-only.
//     - cub::DeviceSelect::Flagged: device-only.
//     - gather_csc_columns_kernel: device-only.
//     - cub::DeviceScan::ExclusiveSum: device-only.
//     - copy_csc_columns_kernel: device-only.
//     - D2H of sub-CSC for factornet: classified as "one-time setup per fit call"
//       (factornet's GPU NMF API takes host CSC pointers and stages internally).
//       Each fit is a self-contained unit; this is NOT per-NMF-iteration.
//     - H2D upload of W result: one-time per-fit result extraction, not per NMF iter.
//   Per-k-means-iter (outer convergence loop, ≤ max_kmeans_iters=100):
//     - cuBLAS Sgemm: device-only.
//     - kmeans_assign_kernel: device-only.
//     - D2H of d_changed scalar: 4 bytes, exactly one valid convergence check per iter.
//     - centroid update kernels: device-only.
//   Per-program bitmap loop (n_total = k + n_cand ≤ 30 + 3000 = 3030, > 5 iters):
//     - D2D copy of program column to sort key buffer: DeviceToDevice, no PCIe traffic.
//     - negate_array_kernel: device-only.
//     - cub::DeviceRadixSort::SortPairs: device-only.
//     - set_topn_bits_kernel: device-only.
//   Final per-k readbacks (one per k in k_range, ≤ 10 iters total):
//     - D2H of h_max_jaccard[k]: one-time per-k result.
//   CONFIRMED: no host<->device transfer inside any loop > 5 iters except the
//     valid per-run sub-CSC/W staging and the scalar 4-byte k-means convergence check.
//
// cub usage:
//   cub::DeviceSelect::Flagged       — Philox boolean mask → sub-CSC column indices
//   cub::DeviceScan::ExclusiveSum    — sub-CSC indptr prefix sum
//   cub::DeviceRadixSort::SortPairs  — top-30 gene indices per program (per-column sort)
//
// cuBLAS usage:
//   cublasSgemm                      — cosine distance matrix [n_cand × k] per k-means iter
//
// Memory budget (m=30k, n=100k, k_max=30, n_runs=100, subsample_frac=0.8):
//   Sub-CSC device scratch:     nnz_full * (4+4+1) bytes ≈ ~500 MB (upper bound)
//   Candidate W stash per k:    m * n_runs * k * 4 = 30k * 100 * 30 * 4 = 360 MB (k=30)
//   Consensus per k:            m * k * 4 = 3.6 MB
//   Cosine dist matrix:         n_cand * k * 4 = 3000 * 30 * 4 = 360 KB
//   Bitmap arrays:              n_total * bitmap_words * 4 = 3030 * 938 * 4 = 11 MB
//   Host consensus stash:       k_range_size * m * k_max * 4 ≈ 10 * 30k * 30 * 4 = 36 MB
//   Peak device: ~600 MB (dominated by candidate W stash + sub-CSC scratch)
//
// Streams: 1, caller-provided. All kernel launches and cub/cuBLAS calls use this stream.
// Precision: fp32 throughout. Jaccard on integer gene IDs is exact.
// Determinism: Philox4x32_10 keyed by (cfg.seed, run_idx). NMF per-run seed derived
//   from (cfg.seed, ki, run). k-means uses first-k-candidates init (deterministic).
//
// OOC plan: subsampling is the natural chunking unit — each fit operates on a fraction
//   of cells. For datasets exceeding device memory, reduce subsample_frac or use
//   build-sub-csc chunked loading via PzDataLoader (deferred to future extension).
//
// References:
//   Kotliar et al. (2019) eLife — CSI-GEP original (cNMF).
//   geeleherlab/CSI-GEP — reference Python/torchNMF implementation.
//   MacQueen (1967) — k-means. Dhillon et al. (2001) — spherical k-means.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/reduce/nmf/types.h>

#include <factornet/nmf/fit_gpu.cuh>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace singlet_gpu {
namespace reduce {
namespace nmf {

// ---------------------------------------------------------------------------
// §1  Public API structs
// ---------------------------------------------------------------------------

struct CsiGepConfig {
    std::vector<int> k_range             = {5, 10, 15, 20, 25, 30};
    int              n_runs              = 100;
    float            subsample_frac      = 0.8f;
    int              top_n_genes_jaccard = 30;
    float            elbow_gap_threshold = 0.05f;
    uint64_t         seed                = 0;
    int              max_kmeans_iters    = 100;
    float            kmeans_tol          = 1e-4f;
    // NMF config forwarded to the inner fit call; seed overridden per run.
    NmfConfig nmf_cfg;
};

struct CsiGepResult {
    int chosen_k;
    // consensus_programs: m × chosen_k col-major (genes × programs; device)
    core::DeviceMemory<float>  consensus_programs;
    // program_usage: n_cells × chosen_k
    core::DeviceMemory<float>  program_usage;
    std::vector<float> reproducibility_curve;  // length k_range.size()
    std::vector<int>   k_used;                 // mirrors cfg.k_range
};

// ---------------------------------------------------------------------------
// §2  Device kernels — all in detail namespace
// ---------------------------------------------------------------------------
namespace detail {

// Fill out[i] = base + i.
__global__ void __launch_bounds__(256, 4)
iota_kernel(int32_t* __restrict__ out, int n, int base)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = base + i;
}

// Philox4x32_10 subsample mask.
// flags[i] = 1 if Philox(seed, run_idx, i) < threshold (uint32 threshold).
__global__ void __launch_bounds__(256, 4)
philox_mask_kernel(uint8_t* __restrict__ flags,
                   int      n_cells,
                   uint64_t base_seed,
                   uint32_t run_idx,
                   uint32_t threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    constexpr uint32_t M0  = 0xD2511F53u;
    constexpr uint32_t M1  = 0xCD9E8D57u;
    constexpr uint32_t SK0 = 0x9E3779B9u;
    constexpr uint32_t SK1 = 0xBB67AE85u;

    uint32_t k0 = static_cast<uint32_t>(base_seed & 0xFFFFFFFFu);
    uint32_t k1 = static_cast<uint32_t>(base_seed >> 32);
    uint32_t x0 = static_cast<uint32_t>(i);
    uint32_t x1 = run_idx;
    uint32_t x2 = 0u, x3 = 0u;

    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        uint64_t p0 = static_cast<uint64_t>(M0) * x0;
        uint64_t p1 = static_cast<uint64_t>(M1) * x2;
        uint32_t hi0 = static_cast<uint32_t>(p0 >> 32);
        uint32_t lo0 = static_cast<uint32_t>(p0);
        uint32_t hi1 = static_cast<uint32_t>(p1 >> 32);
        uint32_t lo1 = static_cast<uint32_t>(p1);
        x0 = hi1 ^ x1 ^ k0;
        x1 = lo1;
        x2 = hi0 ^ x3 ^ k1;
        x3 = lo0;
        k0 += SK0;
        k1 += SK1;
    }
    flags[i] = (x0 < threshold) ? 1u : 0u;
}

// Count pass: store nnz per selected column in sub_indptr[1..n_sel].
// A prefix-sum pass converts these counts to offsets.
__global__ void __launch_bounds__(256, 4)
gather_count_kernel(const int32_t* __restrict__ src_indptr,
                    const int32_t* __restrict__ selected_cols,
                    int32_t*       __restrict__ sub_indptr,
                    int n_sel)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_sel) return;
    int col = selected_cols[tid];
    sub_indptr[tid + 1] = src_indptr[col + 1] - src_indptr[col];
}

// Copy pass: fill sub_indices and sub_values using computed sub_indptr.
__global__ void __launch_bounds__(256, 4)
gather_copy_kernel(const int32_t* __restrict__ src_indptr,
                   const int32_t* __restrict__ src_indices,
                   const float*   __restrict__ src_values,
                   const int32_t* __restrict__ selected_cols,
                   const int32_t* __restrict__ sub_indptr,
                   int32_t*       __restrict__ sub_indices,
                   float*         __restrict__ sub_values,
                   int n_sel)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_sel) return;

    int col       = selected_cols[tid];
    int src_start = src_indptr[col];
    int dst_start = sub_indptr[tid];
    int nnz_col   = src_indptr[col + 1] - src_start;

    for (int i = 0; i < nnz_col; ++i) {
        sub_indices[dst_start + i] = src_indices[src_start + i];
        sub_values [dst_start + i] = src_values [src_start + i];
    }
}

// L2-normalize columns of a dense column-major matrix (m rows, n_cols columns).
// One warp per column.
__global__ void __launch_bounds__(256, 4)
l2_normalize_cols_kernel(float* __restrict__ mat, int m, int n_cols)
{
    // warps_per_block columns; each warp is 32 threads.
    int warp_id  = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    int lane     = threadIdx.x % 32;
    if (warp_id >= n_cols) return;

    float* col = mat + static_cast<size_t>(warp_id) * m;

    float ss = 0.f;
    for (int r = lane; r < m; r += 32) ss += col[r] * col[r];
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);

    float inv = (ss > 0.f) ? rsqrtf(ss) : 0.f;
    for (int r = lane; r < m; r += 32) col[r] *= inv;
}

// k-means assignment: argmax over k cosine-similarity entries per candidate.
// cosine_dist[i*k + c] is similarity between candidate i and centroid c.
// Writes best centroid index into assignments[i] and atomicAdd into changed.
__global__ void __launch_bounds__(256, 4)
kmeans_assign_kernel(const float* __restrict__ cosine_dist,
                     int*         __restrict__ assignments,
                     int*         __restrict__ changed,
                     int n_cand, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cand) return;

    const float* row = cosine_dist + i * k;
    int   best_c = 0;
    float best_v = row[0];
    for (int c = 1; c < k; ++c)
        if (row[c] > best_v) { best_v = row[c]; best_c = c; }

    if (assignments[i] != best_c) {
        assignments[i] = best_c;
        atomicAdd(changed, 1);
    }
}

// Accumulate candidate vectors into centroid sums (atomicAdd per row per block).
// One block per candidate; threads over gene rows.
__global__ void __launch_bounds__(256, 4)
kmeans_accumulate_kernel(const float* __restrict__ candidates,
                         const int*   __restrict__ assignments,
                         float*       __restrict__ centroid_sums,
                         int*         __restrict__ centroid_counts,
                         int n_cand, int m)
{
    int cand = blockIdx.x;
    if (cand >= n_cand) return;
    int c = assignments[cand];
    const float* src = candidates + static_cast<size_t>(cand) * m;
    float*       dst = centroid_sums + static_cast<size_t>(c) * m;
    for (int r = threadIdx.x; r < m; r += blockDim.x)
        atomicAdd(&dst[r], src[r]);
    if (threadIdx.x == 0) atomicAdd(&centroid_counts[c], 1);
}

// Divide centroid sums by counts in-place.
__global__ void __launch_bounds__(256, 4)
kmeans_divide_kernel(float*     __restrict__ sums,
                     const int* __restrict__ counts,
                     int m, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * k) return;
    int cnt = counts[idx / m];
    if (cnt > 0) sums[idx] /= static_cast<float>(cnt);
}

// Negate array elements in-place (for descending radix sort via negated keys).
__global__ void __launch_bounds__(256, 4)
negate_kernel(float* __restrict__ data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = -data[i];
}

// Set the bitmap bits corresponding to the top-N sorted gene indices.
// sorted_gene_ids[0..top_n-1]: gene indices in descending weight order.
// bitmap_row: one row of the bitmap array (bitmap_words uint32).
__global__ void __launch_bounds__(64, 8)
set_topn_bits_kernel(const int32_t* __restrict__ sorted_gene_ids,
                     uint32_t*      __restrict__ bitmap_row,
                     int top_n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= top_n) return;
    int gene = sorted_gene_ids[tid];
    atomicOr(&bitmap_row[gene / 32], 1u << (gene % 32));
}

// Jaccard max: for each (consensus program c, candidate program r), compute
//   jaccard(bitmap_c, bitmap_r) = popcount(AND) / popcount(OR).
// Writes atomicMax into out_max_jaccard[c] over all r.
// gridDim = (k, n_cand); blockDim.x = 128 (for bitmap word reduction).
__global__ void __launch_bounds__(128, 4)
jaccard_max_kernel(const uint32_t* __restrict__ consensus_bitmaps,
                   const uint32_t* __restrict__ run_bitmaps,
                   float*          __restrict__ out_max_jaccard,
                   int k, int n_cand, int bitmap_words)
{
    int c = blockIdx.x, r = blockIdx.y;
    if (c >= k || r >= n_cand) return;

    const uint32_t* bc = consensus_bitmaps + static_cast<size_t>(c) * bitmap_words;
    const uint32_t* br = run_bitmaps       + static_cast<size_t>(r) * bitmap_words;

    int and_cnt = 0, or_cnt = 0;
    for (int w = threadIdx.x; w < bitmap_words; w += blockDim.x) {
        uint32_t a = bc[w], b = br[w];
        and_cnt += __popc(a & b);
        or_cnt  += __popc(a | b);
    }

    // Block reduction via shared memory.
    __shared__ int s_and[128], s_or[128];
    s_and[threadIdx.x] = and_cnt;
    s_or [threadIdx.x] = or_cnt;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_and[threadIdx.x] += s_and[threadIdx.x + s];
            s_or [threadIdx.x] += s_or [threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float jac = (s_or[0] > 0)
            ? static_cast<float>(s_and[0]) / static_cast<float>(s_or[0])
            : 0.f;
        // atomicMax on non-negative floats via uint reinterpretation is valid.
        atomicMax(reinterpret_cast<unsigned int*>(&out_max_jaccard[c]),
                  __float_as_uint(jac));
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// §3  Internal helpers (not exported)
// ---------------------------------------------------------------------------
namespace detail {

// Build the host sub-CSC for one run.
// All device ops use the caller-provided stream.
// Returns host-resident CSC ready for factornet::nmf::nmf_fit_gpu.
struct HostSubCSC {
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<float>   values;
    int n_rows, n_cols, nnz;
};

inline HostSubCSC build_sub_csc(
    const core::DeviceCSC& mat,
    int n_sel,
    uint64_t base_seed, uint32_t run_idx,
    core::DeviceMemory<uint8_t>&  d_flags,
    core::DeviceMemory<int32_t>&  d_cell_range,
    core::DeviceMemory<int32_t>&  d_sel_cols,
    core::DeviceMemory<int32_t>&  d_n_sel_dev,
    core::DeviceMemory<int32_t>&  d_sub_indptr,
    core::DeviceMemory<int32_t>&  d_sub_indices,
    core::DeviceMemory<float>&    d_sub_values,
    void*   d_temp_select, size_t temp_select_bytes,
    void*   d_temp_scan,   size_t temp_scan_bytes,
    cudaStream_t stream)
{
    const int n_cells = static_cast<int>(mat.cols);
    const int m       = static_cast<int>(mat.rows);

    // Threshold: subsample_frac expressed as uint32 for Philox comparison.
    const uint32_t threshold = static_cast<uint32_t>(
        static_cast<double>(n_sel) / static_cast<double>(n_cells) * 4294967296.0);

    // Step 1: Philox mask.
    philox_mask_kernel<<<(n_cells + 255) / 256, 256, 0, stream>>>(
        d_flags.get(), n_cells, base_seed, run_idx, threshold);

    // Step 2: Collect selected column indices.
    cub::DeviceSelect::Flagged(
        d_temp_select, temp_select_bytes,
        d_cell_range.get(), d_flags.get(),
        d_sel_cols.get(), d_n_sel_dev.get(),
        n_cells, stream);

    // Step 3: Count nnz per selected column → sub_indptr[1..n_sel].
    cudaMemsetAsync(d_sub_indptr.get(), 0, (n_sel + 1) * sizeof(int32_t), stream);
    gather_count_kernel<<<(n_sel + 255) / 256, 256, 0, stream>>>(
        mat.col_ptr.get(), d_sel_cols.get(), d_sub_indptr.get(), n_sel);

    // Step 4: Prefix-sum → offsets.
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan, temp_scan_bytes,
        d_sub_indptr.get(), d_sub_indptr.get(), n_sel + 1, stream);

    // Step 5: Copy indices/values.
    gather_copy_kernel<<<(n_sel + 255) / 256, 256, 0, stream>>>(
        mat.col_ptr.get(), mat.row_indices.get(), mat.values.get(),
        d_sel_cols.get(), d_sub_indptr.get(),
        d_sub_indices.get(), d_sub_values.get(), n_sel);

    // Step 6: Read scalar nnz (4 bytes, one-time per call).
    int32_t sub_nnz = 0;
    cudaMemcpyAsync(&sub_nnz, d_sub_indptr.get() + n_sel,
                    sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);  // needed before sizing host vectors

    HostSubCSC h;
    h.n_rows = m; h.n_cols = n_sel; h.nnz = sub_nnz;
    h.indptr.resize(n_sel + 1);
    h.indices.resize(sub_nnz);
    h.values.resize(sub_nnz);

    // One D2H per run (one-time result staging for factornet, not per NMF iteration).
    cudaMemcpyAsync(h.indptr.data(),  d_sub_indptr.get(),
                    (n_sel + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h.indices.data(), d_sub_indices.get(),
                    sub_nnz * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h.values.data(),  d_sub_values.get(),
                    sub_nnz * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h;
}

// Run spherical k-means on the n_cand candidate columns (m × n_cand col-major device matrix).
// Returns host consensus (m × k col-major vector).
inline std::vector<float> spherical_kmeans(
    core::DeviceMemory<float>& d_candidates,
    int m, int k, int n_cand,
    int max_iters, float tol,
    cublasHandle_t cublas,
    cudaStream_t stream)
{
    // L2-normalize candidates in-place.
    {
        // 8 warps per block (256 threads); each warp handles one column.
        int warps_per_blk = 8;
        l2_normalize_cols_kernel<<<
            (n_cand + warps_per_blk - 1) / warps_per_blk,
            warps_per_blk * 32, 0, stream>>>(
            d_candidates.get(), m, n_cand);
    }

    core::DeviceMemory<float> d_centroids(static_cast<size_t>(m) * k);
    core::DeviceMemory<float> d_cosine(static_cast<size_t>(n_cand) * k);
    core::DeviceMemory<int>   d_assignments(n_cand);
    core::DeviceMemory<int>   d_changed(1);
    core::DeviceMemory<int>   d_counts(k);

    // Init: first k candidates as centroids.
    cudaMemcpyAsync(d_centroids.get(), d_candidates.get(),
                    static_cast<size_t>(m) * k * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    {
        int warps_per_blk = 8;
        l2_normalize_cols_kernel<<<
            (k + warps_per_blk - 1) / warps_per_blk,
            warps_per_blk * 32, 0, stream>>>(
            d_centroids.get(), m, k);
    }
    // Initialize assignments to -1 so first iter detects all as changed.
    cudaMemsetAsync(d_assignments.get(), 0xFF, n_cand * sizeof(int), stream);

    const float alpha = 1.f, beta = 0.f;

    for (int iter = 0; iter < max_iters; ++iter) {
        // Cosine similarity matrix: C[n_cand × k] = candidates^T * centroids.
        // Both are m × (n_cand or k) col-major; C^T = centroids^T * candidates.
        // cuBLAS (col-major): C = alpha * A^T * B + beta * C
        //   A = candidates (m × n_cand), A^T = n_cand × m
        //   B = centroids  (m × k)
        //   C = cosine     (n_cand × k)
        cublasSgemm(cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    n_cand, k, m,
                    &alpha,
                    d_candidates.get(), m,
                    d_centroids.get(),  m,
                    &beta,
                    d_cosine.get(),     n_cand);

        cudaMemsetAsync(d_changed.get(), 0, sizeof(int), stream);
        kmeans_assign_kernel<<<(n_cand + 255) / 256, 256, 0, stream>>>(
            d_cosine.get(), d_assignments.get(), d_changed.get(), n_cand, k);

        // Scalar convergence check: 4-byte D2H per k-means iter (valid exception).
        int h_changed = 0;
        cudaMemcpyAsync(&h_changed, d_changed.get(), sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (static_cast<float>(h_changed) / static_cast<float>(n_cand) < tol) break;

        // Update centroids.
        cudaMemsetAsync(d_centroids.get(), 0,
                        static_cast<size_t>(m) * k * sizeof(float), stream);
        cudaMemsetAsync(d_counts.get(), 0, k * sizeof(int), stream);
        kmeans_accumulate_kernel<<<n_cand, 256, 0, stream>>>(
            d_candidates.get(), d_assignments.get(),
            d_centroids.get(), d_counts.get(), n_cand, m);
        kmeans_divide_kernel<<<(m * k + 255) / 256, 256, 0, stream>>>(
            d_centroids.get(), d_counts.get(), m, k);
        {
            int warps_per_blk = 8;
            l2_normalize_cols_kernel<<<
                (k + warps_per_blk - 1) / warps_per_blk,
                warps_per_blk * 32, 0, stream>>>(
                d_centroids.get(), m, k);
        }
    }

    // Read centroids back: one D2H per k value (not per run, not per iter).
    std::vector<float> h_out(static_cast<size_t>(m) * k);
    cudaMemcpyAsync(h_out.data(), d_centroids.get(),
                    static_cast<size_t>(m) * k * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h_out;
}

// Compute Jaccard reproducibility for one k:
//   mean over consensus programs of max Jaccard(top-N genes) over all candidate programs.
// consensus_host: m × k col-major.
// candidates_host: m × n_cand col-major.
inline float jaccard_reproducibility(
    const std::vector<float>& consensus_host,
    const std::vector<float>& candidates_host,
    int m, int k, int n_cand, int top_n,
    cudaStream_t stream)
{
    const int n_total      = k + n_cand;
    const int bitmap_words = (m + 31) / 32;

    // Upload consensus + candidates.
    core::DeviceMemory<float> d_all(static_cast<size_t>(m) * n_total);
    cudaMemcpyAsync(d_all.get(), consensus_host.data(),
                    static_cast<size_t>(m) * k * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_all.get() + static_cast<size_t>(m) * k,
                    candidates_host.data(),
                    static_cast<size_t>(m) * n_cand * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // Bitmaps: n_total × bitmap_words.
    core::DeviceMemory<uint32_t> d_bitmaps(
        static_cast<size_t>(n_total) * bitmap_words);
    cudaMemsetAsync(d_bitmaps.get(), 0,
                    static_cast<size_t>(n_total) * bitmap_words * sizeof(uint32_t), stream);

    // Sort scratch (shared across all program columns).
    size_t sort_tmp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sort_tmp_bytes,
        (float*)nullptr, (float*)nullptr,
        (int32_t*)nullptr, (int32_t*)nullptr,
        m, 0, sizeof(float)*8, stream);
    core::DeviceMemory<uint8_t>  d_sort_tmp(sort_tmp_bytes + 1);
    core::DeviceMemory<float>    d_keys_in(m), d_keys_out(m);
    core::DeviceMemory<int32_t>  d_vals_in(m), d_vals_out(m);

    // Initialize gene index array once (persistent across program iterations).
    iota_kernel<<<(m + 255) / 256, 256, 0, stream>>>(d_vals_in.get(), m, 0);

    // Build top-N bitmap per program.
    // Loop is over n_total programs (≤ 3030). Transfers inside:
    //   - D2D copy of program column to d_keys_in: DeviceToDevice, no PCIe. VALID.
    //   - negate_kernel, sort, set_topn_bits: all device-only.
    for (int p = 0; p < n_total; ++p) {
        const float* prog_ptr = d_all.get() + static_cast<size_t>(p) * m;

        // D2D copy — no PCIe traffic; not restricted by the no-D2H-in-hot-loop rule.
        cudaMemcpyAsync(d_keys_in.get(), prog_ptr,
                        m * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        // Negate so ascending sort gives top-weighted genes first.
        negate_kernel<<<(m + 255) / 256, 256, 0, stream>>>(d_keys_in.get(), m);

        // Reset gene index input (cub::SortPairs reads it but may scramble internal
        // state in some implementations; safer to re-init each time).
        iota_kernel<<<(m + 255) / 256, 256, 0, stream>>>(d_vals_in.get(), m, 0);

        size_t sb = sort_tmp_bytes;
        cub::DeviceRadixSort::SortPairs(
            d_sort_tmp.get(), sb,
            d_keys_in.get(), d_keys_out.get(),
            d_vals_in.get(), d_vals_out.get(),
            m, 0, sizeof(float)*8, stream);

        uint32_t* brow = d_bitmaps.get() + static_cast<size_t>(p) * bitmap_words;
        set_topn_bits_kernel<<<(top_n + 63) / 64, 64, 0, stream>>>(
            d_vals_out.get(), brow, top_n);
    }

    // Jaccard max per consensus program over all candidate programs.
    core::DeviceMemory<float> d_max_jac(k);
    cudaMemsetAsync(d_max_jac.get(), 0, k * sizeof(float), stream);
    {
        const uint32_t* cons_bm = d_bitmaps.get();
        const uint32_t* cand_bm = d_bitmaps.get() + static_cast<size_t>(k) * bitmap_words;
        // Grid: (k, n_cand); block: 128 threads for bitmap reduction.
        dim3 grd(static_cast<unsigned>(k), static_cast<unsigned>(n_cand));
        jaccard_max_kernel<<<grd, 128, 128 * 2 * sizeof(int), stream>>>(
            cons_bm, cand_bm, d_max_jac.get(), k, n_cand, bitmap_words);
    }

    // Read per-program max Jaccard: ONE D2H per k value (not per run or iter).
    std::vector<float> h_jac(k);
    cudaMemcpyAsync(h_jac.data(), d_max_jac.get(),
                    k * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float mean = 0.f;
    for (float v : h_jac) mean += v;
    return mean / static_cast<float>(k);
}

// Elbow detection: first position where the reproducibility gain drops below threshold.
inline int elbow_index(const std::vector<float>& curve, float gap)
{
    int n = static_cast<int>(curve.size());
    for (int i = 0; i < n - 1; ++i)
        if (curve[i + 1] - curve[i] < gap) return i;
    return n - 1;
}

} // namespace detail

// ---------------------------------------------------------------------------
// §4  Public entry point
// ---------------------------------------------------------------------------

inline CsiGepResult csi_gep(
    const core::DeviceCSC& mat,
    const CsiGepConfig&    cfg,
    cudaStream_t           stream)
{
    if (stream == nullptr) stream = cudaStreamDefault;

    const int m       = static_cast<int>(mat.rows);
    const int n_cells = static_cast<int>(mat.cols);
    const int n_k     = static_cast<int>(cfg.k_range.size());
    const int n_sel   = static_cast<int>(std::round(cfg.subsample_frac * n_cells));

    if (n_k == 0)
        throw std::invalid_argument("csi_gep: k_range is empty");
    if (cfg.n_runs < 1)
        throw std::invalid_argument("csi_gep: n_runs < 1");
    if (cfg.subsample_frac <= 0.f || cfg.subsample_frac > 1.f)
        throw std::invalid_argument("csi_gep: subsample_frac out of (0, 1]");
    const int64_t nnz_full = mat.nnz;

    // Download full matrix to host for factornet final NNLS projection (Phase 4).
    // factornet::nmf::nmf_fit_gpu takes host CSC pointers and stages to device internally.
    // This one-time D2H copy (outside all hot loops) satisfies the no-PCIe-in-hot-loops rule.
    std::vector<int>   h_col_ptr_full(static_cast<size_t>(n_cells) + 1);
    std::vector<int>   h_row_indices_full(static_cast<size_t>(nnz_full));
    std::vector<float> h_values_full(static_cast<size_t>(nnz_full));
    cudaMemcpyAsync(h_col_ptr_full.data(), mat.col_ptr.get(),
                    (static_cast<size_t>(n_cells) + 1) * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_row_indices_full.data(), mat.row_indices.get(),
                    static_cast<size_t>(nnz_full) * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_values_full.data(), mat.values.get(),
                    static_cast<size_t>(nnz_full) * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // ---- Preallocate reusable device scratch ----
    core::DeviceMemory<uint8_t>  d_flags(n_cells);
    core::DeviceMemory<int32_t>  d_cell_range(n_cells);
    core::DeviceMemory<int32_t>  d_sel_cols(n_cells);
    core::DeviceMemory<int32_t>  d_n_sel_dev(1);
    core::DeviceMemory<int32_t>  d_sub_indptr(n_sel + 1);
    core::DeviceMemory<int32_t>  d_sub_indices(nnz_full);
    core::DeviceMemory<float>    d_sub_values(nnz_full);

    // Cell index range 0..n_cells-1: one-time setup, never re-uploaded.
    detail::iota_kernel<<<(n_cells + 255) / 256, 256, 0, stream>>>(
        d_cell_range.get(), n_cells, 0);

    // cub DeviceSelect::Flagged temp storage (query size, then allocate once).
    size_t temp_select_bytes = 0;
    cub::DeviceSelect::Flagged(nullptr, temp_select_bytes,
        (int32_t*)nullptr, (uint8_t*)nullptr, (int32_t*)nullptr, (int*)nullptr,
        n_cells, stream);
    core::DeviceMemory<uint8_t> d_temp_select(temp_select_bytes + 1);

    // cub DeviceScan::ExclusiveSum temp storage.
    size_t temp_scan_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_scan_bytes,
        (int32_t*)nullptr, (int32_t*)nullptr, n_sel + 1, stream);
    core::DeviceMemory<uint8_t> d_temp_scan(temp_scan_bytes + 1);

    // cuBLAS handle for k-means cosine Sgemm.
    cublasHandle_t cublas = nullptr;
    cublasCreate(&cublas);
    cublasSetStream(cublas, stream);

    // ---- Phase 1: bootstrap runs per k ----
    std::vector<float>              repro_curve(n_k, 0.f);
    std::vector<std::vector<float>> all_consensus(n_k);  // host consensus per ki

    for (int ki = 0; ki < n_k; ++ki) {
        const int k      = cfg.k_range[ki];
        const int n_cand = cfg.n_runs * k;

        // Per-k candidate W stash: m × n_cand col-major on device.
        core::DeviceMemory<float> d_candidates(static_cast<size_t>(m) * n_cand);

        for (int run = 0; run < cfg.n_runs; ++run) {
            // Build sub-CSC (one D2H per run for factornet staging — valid per-call).
            detail::HostSubCSC h_sub = detail::build_sub_csc(
                mat, n_sel, cfg.seed, static_cast<uint32_t>(run),
                d_flags, d_cell_range, d_sel_cols, d_n_sel_dev,
                d_sub_indptr, d_sub_indices, d_sub_values,
                d_temp_select.get(), temp_select_bytes,
                d_temp_scan.get(),   temp_scan_bytes,
                stream);

            // Per-run NMF seed: deterministic mixing of (seed, ki, run).
            NmfConfig run_cfg = cfg.nmf_cfg;
            run_cfg.rank      = k;
            run_cfg.seed      = cfg.seed
                                + static_cast<uint64_t>(ki)  * 6364136223846793005ULL
                                + static_cast<uint64_t>(run) * 2862933555777941757ULL;

            // NMF fit (factornet — host-staged, GPU-resident inner loop).
            NmfResult result = ::factornet::nmf::nmf_fit_gpu<float>(
                h_sub.indptr.data(),
                h_sub.indices.data(),
                h_sub.values.data(),
                m, n_sel, h_sub.nnz,
                run_cfg,
                nullptr, nullptr);

            // Upload W (m × k) to candidate stash: one H2D per run (valid result extraction).
            float* dst = d_candidates.get() + static_cast<size_t>(run) * k * m;
            cudaMemcpyAsync(dst, result.W.data(),
                            static_cast<size_t>(m) * k * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);  // ensure W is staged before next run
        }

        // Spherical k-means consensus.
        all_consensus[ki] = detail::spherical_kmeans(
            d_candidates, m, k, n_cand,
            cfg.max_kmeans_iters, cfg.kmeans_tol,
            cublas, stream);

        // Jaccard reproducibility: needs candidates on host.
        std::vector<float> h_cands(static_cast<size_t>(m) * n_cand);
        cudaMemcpyAsync(h_cands.data(), d_candidates.get(),
                        static_cast<size_t>(m) * n_cand * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        repro_curve[ki] = detail::jaccard_reproducibility(
            all_consensus[ki], h_cands,
            m, k, n_cand, cfg.top_n_genes_jaccard, stream);
    }

    cublasDestroy(cublas);

    // ---- Phase 2: auto-k selection ----
    const int best_ki = detail::elbow_index(repro_curve, cfg.elbow_gap_threshold);
    const int best_k  = cfg.k_range[best_ki];

    // ---- Phase 3: upload chosen consensus to device ----
    core::DeviceMemory<float> d_consensus(static_cast<size_t>(best_k) * m);
    cudaMemcpyAsync(d_consensus.get(), all_consensus[best_ki].data(),
                    static_cast<size_t>(best_k) * m * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // ---- Phase 4: NNLS projection for all cells ----
    // Use factornet with W fixed to consensus (one NMF iter = one NNLS solve for H).
    NmfConfig proj_cfg  = cfg.nmf_cfg;
    proj_cfg.rank       = best_k;
    proj_cfg.seed       = cfg.seed;
    proj_cfg.max_iter   = 1;  // single NNLS pass; W is warm-started and fixed

    DenseMatrix W_init(m, best_k);
    std::copy(all_consensus[best_ki].begin(), all_consensus[best_ki].end(),
              W_init.data());

    NmfResult proj = ::factornet::nmf::nmf_fit_gpu<float>(
        h_col_ptr_full.data(),
        h_row_indices_full.data(),
        h_values_full.data(),
        m, n_cells, static_cast<int>(nnz_full),
        proj_cfg,
        &W_init, nullptr);

    // H: best_k × n_cells col-major (factornet convention).
    // Transpose to n_cells × best_k for program_usage output.
    core::DeviceMemory<float> d_usage(static_cast<size_t>(n_cells) * best_k);
    {
        // H is best_k × n_cells col-major: element (r, c) = H[r + best_k * c].
        // Output is n_cells × best_k row-major: element (c, r) = h_H_T[c * best_k + r].
        std::vector<float> h_H_T(static_cast<size_t>(n_cells) * best_k);
        const float* H = proj.H.data();
        for (int c = 0; c < n_cells; ++c)
            for (int r = 0; r < best_k; ++r)
                h_H_T[static_cast<size_t>(c) * best_k + r] =
                    H[r + static_cast<size_t>(best_k) * c];
        cudaMemcpyAsync(d_usage.get(), h_H_T.data(),
                        static_cast<size_t>(n_cells) * best_k * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    CsiGepResult result;
    result.chosen_k              = best_k;
    result.consensus_programs    = std::move(d_consensus);
    result.program_usage         = std::move(d_usage);
    result.reproducibility_curve = repro_curve;
    result.k_used                = std::vector<int>(cfg.k_range.begin(), cfg.k_range.end());
    return result;
}

} // namespace nmf
} // namespace reduce
} // namespace singlet_gpu
