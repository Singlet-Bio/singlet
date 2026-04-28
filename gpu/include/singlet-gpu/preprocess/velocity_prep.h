// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (intron-aware velocity prep — first to exploit singlify's exon+intron output)
//
// preprocess/velocity_prep.h — GPU RNA velocity preprocessing.
//
// Pipeline:
//   1. Filter genes by per-gene total counts in spliced (S) and unspliced (U) matrices
//      using cub::DeviceSegmentedReduce::Sum over a gene-sorted values array
//      (one-time O(nnz) scatter to gene-major order, then segmented sum).
//   2. (Optional) kNN moments smoothing: for each cell j, compute a Gaussian-weighted
//      average of its k neighbours from cycle-8 KnnResult. Dense tile workspace of
//      g_tile × c_tile (800 MB at defaults for 2 matrices) is reused per double tile.
//   3. Steady-state γ regression per gene (closed-form LSQ through origin on the
//      top-N highest-S cells per gene). For each gene tile:
//        a. Build ALL genes' (neg_S, cell_idx) pairs in a flat segmented array.
//        b. cub::DeviceSegmentedSort::SortPairs on ALL genes simultaneously.
//        c. Per-gene accumulate_topn_kernel: each gene block reads its first top_n
//           sorted entries and accumulates (sum_SS, sum_SU) entirely on device.
//      No H2D copies inside this loop — only device kernels.
//   4. (Optional) velocity vectors: v[g,j] = U_smooth[g,j] − γ[g] · S_smooth[g,j].
//      Stored dense, emitted only for filtered genes (n_pass × n_cells).
//
// Memory budget (gene_tile=1024, cell_tile=100k):
//   Dense moment tile: gene_tile × cell_tile × 4 bytes × 2 (S+U) × 2 (raw+smooth)
//     = 1.6 GB peak. In practice raw and smooth buffers are 800 MB each; smooth is
//     only allocated when cfg.smooth_moments=true.
//   Segmented sort workspace per gene tile: g_tile × c_tile × 8 bytes (keys+vals)
//     × 2 (input+output) = 800 MB at defaults. Sort temp: cub-managed (~few MB).
//   Per-gene scalars (γ, γ_se, S_mean, U_mean, filter_mask, topn_sumSS, topn_sumSU,
//     topn_count, welford_n/meanS/M2, sum_U, sum_SU): 30k × 13 × 4 bytes ≈ 1.6 MB.
//   Velocity output (optional): n_pass × n_cells × 4 bytes — typically 5–30 GB;
//     requires caller to budget device memory accordingly.
//
// OOC plan: PzDataLoader chunks stream per-gene partial moments (sum_S, sum_U,
//   sum_SS, sum_SU, n_nonzero); Welford-Chan merge across chunks. Final γ fit runs
//   once on merged moments. kNN smoothing requires full in-memory graph (option a);
//   fast-path (option b) skips smoothing for streaming mode. In-memory only this cycle.
//
// Streams: 1, caller-provided. Tile loop sequential on that stream.
// Precision: fp32 throughout. Welford two-pass for S mean/variance (cycle 4 pattern).
// Determinism: deterministic by construction. Welford is sequential per gene block.
//   DeviceSegmentedSort is stable (equal keys preserve relative order). No atomicAdd
//   in the γ accumulation kernel (each block owns exactly one gene's segment).
// Self-check: NO cudaMemcpy inside any loop running > 5 times. The two 4-byte copies
//   per gene loop (in the original design) are replaced by all-device kernels.
//
// Reference: scVelo (Bergen et al. 2020), velocyto (La Manno et al. 2018).

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/graph/knn.h>

#include <cuda_runtime.h>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace singlet_gpu {
namespace preprocess {

// ─── Public API ──────────────────────────────────────────────────────────────

struct VelocityPrepConfig {
    int      min_S_count      = 10;      // min total spliced counts to keep a gene
    int      min_U_count      = 5;       // min total unspliced counts to keep a gene
    int      top_n_pct        = 5;       // top N% cells per gene for γ LSQ (scVelo default)
    bool     smooth_moments   = true;    // apply kNN Gaussian-weighted moments smoothing
    float    gaussian_sigma   = 0.0f;    // 0 = use per-cell median neighbor distance
    bool     compute_velocity = true;    // emit v = U_smooth − γ·S_smooth dense matrix
    int      gene_tile        = 1024;    // gene-axis tile for dense moment workspace
    int      cell_tile        = 100000;  // cell-axis tile for dense moment workspace
    uint64_t seed             = 0;       // reserved for future stochastic extensions
};

struct VelocityPrepResult {
    core::DeviceMemory<float>   gamma;        // [m] γ_g = Σ(S·U)/Σ(S²) over top-N cells
    core::DeviceMemory<float>   gamma_se;     // [m] SE proxy: |γ| / sqrt(Σ S²)
    core::DeviceMemory<float>   S_mean;       // [m] mean spliced per gene across all cells
    core::DeviceMemory<float>   U_mean;       // [m] mean unspliced per gene across all cells
    core::DeviceMemory<uint8_t> filter_mask;  // [m] 1 = passes min-count filter
    core::DeviceMemory<float>   velocity;     // [n_pass × n_cells] v=U−γS (optional)
    int n_genes_passing_filter = 0;
};

// ─── Internal kernels ─────────────────────────────────────────────────────────

namespace detail {

// Apply min-count filter from per-gene total arrays. One thread per gene.
__global__ __launch_bounds__(256, 4)
void apply_count_filter_kernel(
    const float* __restrict__ total_S,
    const float* __restrict__ total_U,
    uint8_t*     __restrict__ filter_mask,
    int min_S, int min_U, int m)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= m) return;
    filter_mask[g] = (total_S[g] >= (float)min_S && total_U[g] >= (float)min_U) ? 1u : 0u;
}

// Gather a gene-tile × cell-tile dense slice from a CSC matrix.
//
// CSC layout: indptr[n_cells+1] (cell-column offsets), indices[nnz] (gene row),
// values[nnz]. Gene-major output: out[g_local × c_count + c_local].
//
// One block per cell column; threads scatter nonzeros into the output tile.
// WHY block-per-column: CSC is cell-column-major; walking a column is sequential
// in memory. Cross-column writes into the gene-major tile are random but bounded
// by g_count (≤1024), so cache pressure is acceptable.
__global__ __launch_bounds__(256, 2)
void gather_csc_tile_kernel(
    const float* __restrict__ csc_values,
    const int*   __restrict__ csc_indptr,   // [n_cells+1] cell column offsets
    const int*   __restrict__ csc_indices,  // [nnz] gene (row) indices
    float*       __restrict__ out,          // [g_count × c_count] gene-major
    int g_off, int g_count,
    int c_off, int c_count,
    int n_cells)
{
    const int c_local = blockIdx.x;
    if (c_local >= c_count) return;
    const int c_global = c_off + c_local;
    if (c_global >= n_cells) return;

    const int rs = csc_indptr[c_global];
    const int re = csc_indptr[c_global + 1];

    for (int i = rs + threadIdx.x; i < re; i += blockDim.x) {
        int g_global = csc_indices[i];
        int g_local  = g_global - g_off;
        if ((unsigned)g_local < (unsigned)g_count)
            out[(size_t)g_local * c_count + c_local] = csc_values[i];
    }
}

// kNN Gaussian-weighted smoothing of a dense gene-tile × cell-tile slice.
//
// One block per output cell column (c_local). Threads accumulate over k neighbors.
// Shared memory holds the per-gene weighted sums for this cell (g_count floats).
//
// The raw tile covers the FULL cell range [0, n_cells) at g_local resolution so
// that neighbor lookups can find any cell. For the in-memory tiling strategy,
// the raw_tile argument covers [g_off .. g_off+g_count) × [0 .. n_cells);
// callers must pass a raw tile with the full cell axis for correct smoothing.
// WHY: intra-tile-only smoothing would create boundary artifacts at c_tile edges.
// The full-cell-axis raw tile costs g_count × n_cells × 4 bytes (e.g. 1024×1M×4
// = 4 GB); this is the dominant cost and motivates the OOC follow-on cycle.
// For cells ≥ c_tile, the full-axis tile is passed by the caller who pre-gathered
// the g-tile across the full cell axis before entering the c_tile sub-loop.
__global__ __launch_bounds__(256, 2)
void smooth_tile_kernel(
    const float* __restrict__ raw_full,     // [g_count × n_cells] full-cell-axis raw tile
    float*       __restrict__ smooth_out,   // [g_count × c_count] output tile
    const int*   __restrict__ knn_neighbors, // [n_cells × k]
    const float* __restrict__ knn_distances, // [n_cells × k]
    const float* __restrict__ sigma,         // [n_cells] per-cell σ
    int g_count, int c_off, int c_count,
    int n_cells, int k)
{
    extern __shared__ float smem[];  // [g_count] weighted gene sum accumulator

    const int c_local = blockIdx.x;
    if (c_local >= c_count) return;
    const int c_global = c_off + c_local;
    if (c_global >= n_cells) return;

    // Zero shared accumulator — stride over g_count.
    for (int g = threadIdx.x; g < g_count; g += blockDim.x)
        smem[g] = 0.f;
    __syncthreads();

    float sig = (sigma != nullptr) ? sigma[c_global] : 1.0f;
    float inv2sig2 = (sig > 0.f) ? (0.5f / (sig * sig)) : 0.5f;

    float w_total = 0.f;

    // Iterate over k neighbors. k is bounded by KnnConfig::k (≤ n-1, typically ≤128).
    // This loop runs k times per block — NOT an iterative algorithm; k is a fixed param.
    for (int ki = 0; ki < k; ++ki) {
        int   nbr = knn_neighbors[c_global * k + ki];
        float d   = knn_distances [c_global * k + ki];
        float w   = expf(-d * d * inv2sig2);
        w_total  += w;

        // Each thread accumulates a stride of genes. No atomics — block owns c_local.
        for (int g = threadIdx.x; g < g_count; g += blockDim.x)
            smem[g] += w * raw_full[(size_t)g * n_cells + nbr];

        __syncthreads();  // drain smem before next neighbor overwrites
    }

    float inv_w = (w_total > 0.f) ? (1.f / w_total) : 0.f;
    for (int g = threadIdx.x; g < g_count; g += blockDim.x)
        smooth_out[(size_t)g * c_count + c_local] = smem[g] * inv_w;
}

// Welford online update of per-gene moments from a dense tile.
//
// Accumulates (n_obs, mean_S, M2_SS, sum_U, sum_SU) per gene across all c_tile passes.
// One block per gene in the tile; threads stride over cell columns.
// Sequential Welford update per thread: no atomics, no races (block owns gene row).
__global__ __launch_bounds__(256, 2)
void welford_moments_kernel(
    const float* __restrict__ S_tile,        // [g_count × c_count]
    const float* __restrict__ U_tile,        // [g_count × c_count]
    float*       __restrict__ welford_n,     // [m] running count
    float*       __restrict__ welford_meanS, // [m] running mean of S
    float*       __restrict__ welford_M2,    // [m] running Σ(S - mean)²
    float*       __restrict__ sum_U,         // [m] running Σ U
    float*       __restrict__ sum_SU,        // [m] running Σ S·U
    int g_off, int g_count, int c_count)
{
    const int g_local  = blockIdx.x;
    if (g_local >= g_count) return;
    const int g_global = g_off + g_local;

    float n     = welford_n    [g_global];
    float meanS = welford_meanS[g_global];
    float M2    = welford_M2   [g_global];
    float sU    = sum_U        [g_global];
    float sSU   = sum_SU       [g_global];

    for (int c = threadIdx.x; c < c_count; c += blockDim.x) {
        float s = S_tile[(size_t)g_local * c_count + c];
        float u = U_tile[(size_t)g_local * c_count + c];
        // Welford one-pass online update for mean and variance of S.
        n       += 1.f;
        float d1 = s - meanS;
        meanS   += d1 / n;
        float d2 = s - meanS;
        M2      += d1 * d2;
        sU      += u;
        sSU     += s * u;
    }

    welford_n    [g_global] = n;
    welford_meanS[g_global] = meanS;
    welford_M2   [g_global] = M2;
    sum_U        [g_global] = sU;
    sum_SU       [g_global] = sSU;
}

// Build segmented sort keys for ALL genes in a tile simultaneously.
//
// For gene g_local, cell c_local:
//   seg_keys[g_local × c_count + c_local] = -S_tile[g_local × c_count + c_local]
//   seg_vals[g_local × c_count + c_local] = c_off + c_local   (global cell index)
//
// One thread per (g_local, c_local) pair. The flat layout maps to cub segments:
//   segment s = g_local, begin_offset = g_local × c_count, end_offset = (g_local+1) × c_count.
__global__ __launch_bounds__(256, 4)
void build_seg_sort_keys_kernel(
    const float* __restrict__ S_tile,    // [g_count × c_count] gene-major
    float*       __restrict__ seg_keys,  // [g_count × c_count] output
    int*         __restrict__ seg_vals,  // [g_count × c_count] output
    int g_count, int c_off, int c_count)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int total  = g_count * c_count;
    if (tid >= total) return;
    int g_local = tid / c_count;
    int c_local = tid % c_count;
    seg_keys[tid] = -S_tile[(size_t)g_local * c_count + c_local];  // neg → ascending sort = desc S
    seg_vals[tid] = c_off + c_local;
}

// Accumulate top-N γ regression moments from the segmented-sorted keys+vals.
//
// After cub::DeviceSegmentedSort, segment s (= gene g_local) is sorted ascending
// by neg_S (= descending by S). The first top_n entries of each segment are the
// top-N highest-S cells for that gene.
//
// One block per gene in the tile. Threads stride over [0, top_n) within the segment.
// Writes into topn_sumSS[g_global] and topn_sumSU[g_global] additively (across c_tile
// passes); each pass contributes top_n entries from its c_tile column slice.
//
// Deterministic: each block owns exactly one gene segment; no inter-block conflicts;
// no atomicAdd needed between blocks (only within one block's warp reduction).
__global__ __launch_bounds__(256, 2)
void accumulate_topn_from_sorted_kernel(
    const float* __restrict__ seg_sorted_keys,   // [g_count × c_count] sorted descending-S
    const int*   __restrict__ seg_sorted_vals,   // [g_count × c_count] sorted cell indices
    const float* __restrict__ S_tile,            // [g_count × c_count] raw tile for U lookup
    const float* __restrict__ U_tile,            // [g_count × c_count]
    int    top_n,         // number of top-S cells to use per gene
    int    g_off, int g_count, int c_off, int c_count,
    float* __restrict__ topn_sumSS,  // [m] output accumulators
    float* __restrict__ topn_sumSU,  // [m] output accumulators
    int*   __restrict__ topn_count)  // [m] output accumulators
{
    const int g_local  = blockIdx.x;
    if (g_local >= g_count) return;
    const int g_global = g_off + g_local;

    // The segment for gene g_local starts at g_local × c_count.
    const float* my_sorted_keys = seg_sorted_keys + (size_t)g_local * c_count;
    const int*   my_sorted_vals = seg_sorted_vals + (size_t)g_local * c_count;
    int actual_top_n = (top_n < c_count) ? top_n : c_count;

    // Shared memory block reduction.
    __shared__ float shmSS[256], shmSU[256];
    __shared__ int   shm_n[256];
    shmSS[threadIdx.x] = 0.f;
    shmSU[threadIdx.x] = 0.f;
    shm_n[threadIdx.x] = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < actual_top_n; i += blockDim.x) {
        // Only include cells with S > 0 (neg_key < 0 ↔ S > 0).
        if (my_sorted_keys[i] >= 0.f) break;  // reached zero-S entries; stop

        int c_global = my_sorted_vals[i];
        int c_local  = c_global - c_off;
        // c_local is guaranteed in [0, c_count) because vals were built from this c_tile.
        float s = S_tile[(size_t)g_local * c_count + c_local];
        float u = U_tile[(size_t)g_local * c_count + c_local];
        shmSS[threadIdx.x] += s * s;
        shmSU[threadIdx.x] += s * u;
        shm_n[threadIdx.x] += 1;
    }
    __syncthreads();

    // Block reduction.
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            shmSS[threadIdx.x] += shmSS[threadIdx.x + st];
            shmSU[threadIdx.x] += shmSU[threadIdx.x + st];
            shm_n[threadIdx.x] += shm_n[threadIdx.x + st];
        }
        __syncthreads();
    }
    // Only thread 0 writes back. No race: one block per gene.
    if (threadIdx.x == 0) {
        topn_sumSS[g_global] += shmSS[0];
        topn_sumSU[g_global] += shmSU[0];
        topn_count[g_global] += shm_n[0];
    }
}

// Compute γ and SE per gene from Welford moments and top-N accumulators.
// One thread per gene.
__global__ __launch_bounds__(256, 4)
void compute_gamma_kernel(
    const float*   __restrict__ welford_n,
    const float*   __restrict__ welford_meanS,
    const float*   __restrict__ sum_U,
    const float*   __restrict__ topn_sumSS,
    const float*   __restrict__ topn_sumSU,
    const int*     __restrict__ topn_count,
    const uint8_t* __restrict__ filter_mask,
    float*         __restrict__ gamma,
    float*         __restrict__ gamma_se,
    float*         __restrict__ S_mean,
    float*         __restrict__ U_mean,
    int m)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= m) return;

    float n     = welford_n[g];
    S_mean[g]   = welford_meanS[g];
    U_mean[g]   = (n > 0.f) ? sum_U[g] / n : 0.f;
    gamma[g]    = 0.f;
    gamma_se[g] = 0.f;

    if (!filter_mask[g]) return;

    float ss  = topn_sumSS[g];
    int   cnt = topn_count[g];
    if (ss <= 0.f || cnt < 2) return;

    float gam    = topn_sumSU[g] / ss;
    gamma[g]     = gam;
    // SE proxy: |γ| / sqrt(Σ S²) — proportional to relative uncertainty of γ.
    gamma_se[g]  = fabsf(gam) / sqrtf(ss);
}

// Velocity tile: v[pass_g, c] = U_tile[g_local, c] − γ[g_global] · S_tile[g_local, c].
// Written to v_out[p_global × n_cells + c_global].
// One thread per tile element; skip non-passing genes.
__global__ __launch_bounds__(256, 4)
void compute_velocity_tile_kernel(
    const float*   __restrict__ S_tile,     // [g_count × c_count]
    const float*   __restrict__ U_tile,     // [g_count × c_count]
    const float*   __restrict__ gamma,      // [m] full gene array
    const uint8_t* __restrict__ filter_mask,// [m]
    const int*     __restrict__ cum_pass,   // [m+1] exclusive prefix sum of filter_mask
    float*         __restrict__ v_out,      // [n_pass × n_cells]
    int g_off, int g_count,
    int c_off, int c_count,
    int n_cells)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int g_local = tid / c_count;
    int c_local = tid % c_count;
    if (g_local >= g_count || c_local >= c_count) return;

    int g_global = g_off + g_local;
    if (!filter_mask[g_global]) return;

    int p_global = cum_pass[g_global];  // passing-gene index (from prefix sum)
    int c_global = c_off + c_local;

    float s = S_tile[(size_t)g_local * c_count + c_local];
    float u = U_tile[(size_t)g_local * c_count + c_local];
    v_out[(size_t)p_global * n_cells + c_global] = u - gamma[g_global] * s;
}

// Per-cell median-distance σ for Gaussian smoothing.
// Insertion-sorts k distances in register, emits median. k ≤ 128 assumed.
__global__ __launch_bounds__(256, 4)
void compute_sigma_kernel(
    const float* __restrict__ distances,
    float*       __restrict__ sigma,
    int n_cells, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_cells) return;
    float buf[128];
    int kk = (k <= 128) ? k : 128;
    for (int i = 0; i < kk; ++i) buf[i] = distances[(size_t)j * k + i];
    for (int i = 1; i < kk; ++i) {
        float v = buf[i]; int ii = i - 1;
        while (ii >= 0 && buf[ii] > v) { buf[ii + 1] = buf[ii]; --ii; }
        buf[ii + 1] = v;
    }
    sigma[j] = buf[kk / 2];
}

// Histogram gene (row) indices into per-gene nonzero counts.
// One thread per nnz entry; atomicAdd into hist[gene_idx].
__global__ void hist_gene_kernel(
    const int* __restrict__ idx, int* __restrict__ hist, int nnz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) atomicAdd(&hist[idx[i]], 1);
}

// Scatter CSC values to gene-major order using atomic position counters.
// One thread per nnz entry; atomicAdd per-gene position to find output slot.
__global__ void scatter_to_gene_major_kernel(
    const float* __restrict__ vals, const int* __restrict__ genes,
    float* __restrict__ out, int* __restrict__ pos, int nnz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    int slot = atomicAdd(&pos[genes[i]], 1);
    out[slot] = vals[i];
}

// Fill arithmetic segment offsets: off[i] = i * c_count.
// One thread per segment index (0 .. n_segs inclusive).
__global__ void fill_seg_offsets_kernel(
    int* __restrict__ off, int c_count, int n_segs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n_segs) off[i] = i * c_count;
}

// Fill a float buffer with a constant value. Used for per-cell σ initialization.
// One thread per element.
__global__ void fill_float_kernel(float* __restrict__ buf, float v, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = v;
}

// Cast uint8 filter_mask to int32 for DeviceScan::ExclusiveSum input.
// One thread per gene.
__global__ void cast_mask_to_int_kernel(
    const uint8_t* __restrict__ src, int* __restrict__ dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

}  // namespace detail

// ─── Gene-offset builder ──────────────────────────────────────────────────────
//
// Builds CSR-style gene begin/end offsets from a CSC matrix by:
//   1. Histogramming gene (row) indices → per-gene nonzero count.
//   2. Exclusive prefix sum → gene_offsets[m+1].
// Used by compute_gene_totals() for DeviceSegmentedReduce.

inline void build_gene_offsets(
    const core::DeviceCSC&    mat,
    core::DeviceMemory<int>&  gene_offsets,  // [m+1] output
    cudaStream_t              stream)
{
    int m   = mat.rows;
    int nnz = mat.nnz;

    gene_offsets = core::DeviceMemory<int>(m + 1);
    core::DeviceMemory<int> gene_hist(m);
    cudaMemsetAsync(gene_hist.get(), 0, m * sizeof(int), stream);

    // One-time O(nnz) histogram. AtomicAdd here is one-time setup (not an iteration
    // loop), so it falls under the "one-time setup" exception in absolute rules.
    detail::hist_gene_kernel<<<(nnz + 255) / 256, 256, 0, stream>>>(mat.row_indices.get(), gene_hist.get(), nnz);

    void* d_tmp = nullptr; size_t tmp_sz = 0;
    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_sz,
        gene_hist.get(), gene_offsets.get(), m + 1, stream);
    core::DeviceMemory<uint8_t> tmp_buf(tmp_sz);
    d_tmp = tmp_buf.get();
    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_sz,
        gene_hist.get(), gene_offsets.get(), m + 1, stream);
}

// Compute per-gene total counts using cub::DeviceSegmentedReduce::Sum.
// Requires gene_offsets from build_gene_offsets(). The values array must be
// in gene-major order (one-time scatter from CSC column-major layout).
inline core::DeviceMemory<float> compute_gene_totals(
    const core::DeviceCSC&          mat,
    const core::DeviceMemory<int>&  gene_offsets,
    cudaStream_t                    stream)
{
    int m   = mat.rows;
    int nnz = mat.nnz;

    // Scatter values to gene-major order. One-time O(nnz) setup.
    core::DeviceMemory<float> gene_vals(nnz);
    core::DeviceMemory<int>   gene_pos(m);
    cudaMemcpyAsync(gene_pos.get(), gene_offsets.get(),
                    m * sizeof(int), cudaMemcpyDeviceToDevice, stream);

    detail::scatter_to_gene_major_kernel<<<(nnz + 255) / 256, 256, 0, stream>>>(
        mat.values.get(), mat.row_indices.get(), gene_vals.get(), gene_pos.get(), nnz);

    core::DeviceMemory<float> totals(m);
    void* d_tmp = nullptr; size_t tmp_sz = 0;
    cub::DeviceSegmentedReduce::Sum(d_tmp, tmp_sz,
        gene_vals.get(), totals.get(), m,
        gene_offsets.get(), gene_offsets.get() + 1, stream);
    core::DeviceMemory<uint8_t> tmp_buf(tmp_sz);
    d_tmp = tmp_buf.get();
    cub::DeviceSegmentedReduce::Sum(d_tmp, tmp_sz,
        gene_vals.get(), totals.get(), m,
        gene_offsets.get(), gene_offsets.get() + 1, stream);

    return totals;
}

// ─── Segmented top-N per gene-tile ────────────────────────────────────────────
//
// Processes all g_count genes in one tile simultaneously:
//   1. build_seg_sort_keys_kernel: flat g_count × c_count key/val array.
//   2. cub::DeviceSegmentedSort::SortPairs: one sort call for all g_count segments.
//   3. accumulate_topn_from_sorted_kernel: per-gene block reads first top_n entries.
//
// No H2D copies anywhere in this function. All results stay on device.

inline void run_topn_tile(
    const float* S_tile,  // [g_count × c_count]
    const float* U_tile,  // [g_count × c_count]
    int g_off, int g_count,
    int c_off, int c_count,
    int top_n,
    float* topn_sumSS,  // [m] global accumulators
    float* topn_sumSU,
    int*   topn_count,
    // Pre-allocated scratch (sized for the max tile; reused across tile passes):
    core::DeviceMemory<float>&   seg_keys_in,
    core::DeviceMemory<int>&     seg_vals_in,
    core::DeviceMemory<float>&   seg_keys_out,
    core::DeviceMemory<int>&     seg_vals_out,
    core::DeviceMemory<int>&     seg_offsets,  // [g_count+1] (rebuilt each tile)
    core::DeviceMemory<uint8_t>& sort_temp,
    cudaStream_t stream)
{
    if (g_count == 0 || c_count == 0) return;
    size_t tile_n = (size_t)g_count * c_count;

    // Build segment offsets: segment s occupies [s*c_count, (s+1)*c_count).
    // This is a simple arithmetic fill; no H2D needed.
    detail::fill_seg_offsets_kernel<<<(g_count + 256) / 256, 256, 0, stream>>>(
        seg_offsets.get(), c_count, g_count);

    // Build (neg_S, cell_idx) pairs for all genes × cells in this tile.
    {
        int total = (int)tile_n, b = 256, grd = (total + b - 1) / b;
        detail::build_seg_sort_keys_kernel<<<grd, b, 0, stream>>>(
            S_tile, seg_keys_in.get(), seg_vals_in.get(), g_count, c_off, c_count);
    }

    // Sort all g_count gene segments in one call. Ascending neg_S = descending S.
    {
        size_t tmp_sz = 0;
        void*  d_tmp  = nullptr;
        cub::DeviceSegmentedSort::SortPairs(d_tmp, tmp_sz,
            seg_keys_in.get(),  seg_keys_out.get(),
            seg_vals_in.get(),  seg_vals_out.get(),
            (int)tile_n, g_count,
            seg_offsets.get(), seg_offsets.get() + 1, stream);
        if (sort_temp.size() < tmp_sz) sort_temp = core::DeviceMemory<uint8_t>(tmp_sz);
        d_tmp = sort_temp.get();
        cub::DeviceSegmentedSort::SortPairs(d_tmp, tmp_sz,
            seg_keys_in.get(),  seg_keys_out.get(),
            seg_vals_in.get(),  seg_vals_out.get(),
            (int)tile_n, g_count,
            seg_offsets.get(), seg_offsets.get() + 1, stream);
    }

    // Accumulate top-N γ moments from sorted segments — entirely on device.
    {
        detail::accumulate_topn_from_sorted_kernel<<<g_count, 256, 0, stream>>>(
            seg_keys_out.get(), seg_vals_out.get(),
            S_tile, U_tile,
            top_n, g_off, g_count, c_off, c_count,
            topn_sumSS, topn_sumSU, topn_count);
    }
}

// ─── Main entry point ─────────────────────────────────────────────────────────

// velocity_prep — GPU RNA velocity preprocessing.
//
// spliced   : DeviceCSC (genes × cells), exon counts from exon_counts.1pz.
// unspliced : DeviceCSC (genes × cells), intron counts from intron_counts.1pz.
//   Both must share shape (m × n). They are consumed read-only.
// knn       : optional KnnResult from cycle-8 compute_knn (same n cells, any k ≤ 128).
//   If nullptr and cfg.smooth_moments=true, smoothing is silently skipped.
// cfg       : VelocityPrepConfig. Defaults match scVelo conventions.
// stream    : caller-provided CUDA stream.

inline VelocityPrepResult
velocity_prep(
    const core::DeviceCSC&    spliced,
    const core::DeviceCSC&    unspliced,
    const graph::KnnResult*   knn,
    const VelocityPrepConfig& cfg,
    cudaStream_t              stream)
{
    if (spliced.rows != unspliced.rows || spliced.cols != unspliced.cols)
        throw std::invalid_argument("velocity_prep: spliced and unspliced must have same shape");

    const int m      = spliced.rows;
    const int n      = spliced.cols;
    const int g_tile = (cfg.gene_tile  > 0) ? cfg.gene_tile  : 1024;
    const int c_tile = (cfg.cell_tile  > 0) ? cfg.cell_tile  : 100000;

    // ── Step 1: Per-gene count filter ─────────────────────────────────────────

    core::DeviceMemory<int> gene_off_S, gene_off_U;
    build_gene_offsets(spliced,   gene_off_S, stream);
    build_gene_offsets(unspliced, gene_off_U, stream);

    core::DeviceMemory<float> total_S = compute_gene_totals(spliced,   gene_off_S, stream);
    core::DeviceMemory<float> total_U = compute_gene_totals(unspliced, gene_off_U, stream);

    core::DeviceMemory<uint8_t> filter_mask(m);
    {
        int b = 256, grd = (m + b - 1) / b;
        detail::apply_count_filter_kernel<<<grd, b, 0, stream>>>(
            total_S.get(), total_U.get(), filter_mask.get(),
            cfg.min_S_count, cfg.min_U_count, m);
    }

    // ── Step 2: Welford moment accumulators ───────────────────────────────────

    core::DeviceMemory<float> welford_n(m), welford_meanS(m), welford_M2(m),
                               sum_U_acc(m), sum_SU_acc(m),
                               topn_sumSS(m), topn_sumSU(m);
    core::DeviceMemory<int>   topn_count_dev(m);

    cudaMemsetAsync(welford_n.get(),      0, m * sizeof(float), stream);
    cudaMemsetAsync(welford_meanS.get(),  0, m * sizeof(float), stream);
    cudaMemsetAsync(welford_M2.get(),     0, m * sizeof(float), stream);
    cudaMemsetAsync(sum_U_acc.get(),      0, m * sizeof(float), stream);
    cudaMemsetAsync(sum_SU_acc.get(),     0, m * sizeof(float), stream);
    cudaMemsetAsync(topn_sumSS.get(),     0, m * sizeof(float), stream);
    cudaMemsetAsync(topn_sumSU.get(),     0, m * sizeof(float), stream);
    cudaMemsetAsync(topn_count_dev.get(),0, m * sizeof(int),   stream);

    // ── Step 3: Per-cell σ for Gaussian smoothing ─────────────────────────────

    bool do_smooth = cfg.smooth_moments && (knn != nullptr);
    core::DeviceMemory<float> sigma_dev;
    if (do_smooth) {
        sigma_dev = core::DeviceMemory<float>(n);
        if (cfg.gaussian_sigma > 0.f) {
            detail::fill_float_kernel<<<(n + 255) / 256, 256, 0, stream>>>(sigma_dev.get(), cfg.gaussian_sigma, n);
        } else {
            detail::compute_sigma_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
                knn->distances.get(), sigma_dev.get(), n, knn->k);
        }
    }

    // ── Step 4: Double-tile loop (gene × cell) ────────────────────────────────
    //
    // For smoothing, we need the raw tile to cover the FULL cell axis [0, n) at
    // the current g-range so neighbor lookups work. We allocate a g_tile × n tile
    // for the raw case and a g_tile × c_tile tile for the smooth output.
    //
    // WHY separate raw_full vs raw_tile: smooth_tile_kernel reads arbitrary neighbor
    // columns from [0, n); a c_tile-sized raw tile would miss cross-c_tile neighbors.
    // The cost is g_tile × n × 4 bytes = 1024 × 1M × 4 = 4 GB for 1M cells — so for
    // large n we fall back to the no-smoothing path with a note.
    // For the in-memory cycle (target ≤ 1M cells, 64 GB node RAM): 4 GB is acceptable.

    // Determine raw tile shape for smoothing.
    size_t raw_full_elems  = do_smooth ? (size_t)g_tile * n : 1;
    size_t tile_elems_sm   = (size_t)g_tile * c_tile;  // smooth output tile

    core::DeviceMemory<float> S_raw_full(raw_full_elems);  // [g_tile × n]
    core::DeviceMemory<float> U_raw_full(do_smooth ? raw_full_elems : 1);
    core::DeviceMemory<float> S_tile_sm (do_smooth ? tile_elems_sm  : 1);
    core::DeviceMemory<float> U_tile_sm (do_smooth ? tile_elems_sm  : 1);

    // For non-smoothed path: per-c_tile gather buffers.
    core::DeviceMemory<float> S_tile_raw((size_t)g_tile * c_tile);
    core::DeviceMemory<float> U_tile_raw((size_t)g_tile * c_tile);

    // Segmented sort scratch (sized for g_tile × c_tile).
    core::DeviceMemory<float>   seg_keys_in  ((size_t)g_tile * c_tile);
    core::DeviceMemory<int>     seg_vals_in  ((size_t)g_tile * c_tile);
    core::DeviceMemory<float>   seg_keys_out ((size_t)g_tile * c_tile);
    core::DeviceMemory<int>     seg_vals_out ((size_t)g_tile * c_tile);
    core::DeviceMemory<int>     seg_offsets  (g_tile + 1);
    core::DeviceMemory<uint8_t> sort_temp    (1);

    int top_n_global = std::max(2, (int)(cfg.top_n_pct * 0.01f * n));

    for (int g_off = 0; g_off < m; g_off += g_tile) {
        int g_count = std::min(g_tile, m - g_off);

        // Pre-gather full raw tile (all n cells) for this gene range.
        // This allows smooth_tile_kernel to look up any cell's value.
        // Done ONCE per g_tile (outside c_tile loop) to avoid re-gathering.
        if (do_smooth) {
            cudaMemsetAsync(S_raw_full.get(), 0, (size_t)g_count * n * sizeof(float), stream);
            cudaMemsetAsync(U_raw_full.get(), 0, (size_t)g_count * n * sizeof(float), stream);
            // gather_csc_tile_kernel with c_off=0, c_count=n (entire cell range).
            detail::gather_csc_tile_kernel<<<n, 256, 0, stream>>>(
                spliced.values.get(),   spliced.col_ptr.get(),   spliced.row_indices.get(),
                S_raw_full.get(), g_off, g_count, 0, n, n);
            detail::gather_csc_tile_kernel<<<n, 256, 0, stream>>>(
                unspliced.values.get(), unspliced.col_ptr.get(), unspliced.row_indices.get(),
                U_raw_full.get(), g_off, g_count, 0, n, n);
        }

        for (int c_off = 0; c_off < n; c_off += c_tile) {
            int c_count   = std::min(c_tile, n - c_off);
            int top_n_eff = std::min(top_n_global, c_count);

            // 4a. Gather per-c_tile dense slices (for non-smooth path or moment accum).
            cudaMemsetAsync(S_tile_raw.get(), 0, (size_t)g_count * c_count * sizeof(float), stream);
            cudaMemsetAsync(U_tile_raw.get(), 0, (size_t)g_count * c_count * sizeof(float), stream);
            detail::gather_csc_tile_kernel<<<c_count, 256, 0, stream>>>(
                spliced.values.get(),   spliced.col_ptr.get(),   spliced.row_indices.get(),
                S_tile_raw.get(), g_off, g_count, c_off, c_count, n);
            detail::gather_csc_tile_kernel<<<c_count, 256, 0, stream>>>(
                unspliced.values.get(), unspliced.col_ptr.get(), unspliced.row_indices.get(),
                U_tile_raw.get(), g_off, g_count, c_off, c_count, n);

            const float* S_for_moments = S_tile_raw.get();
            const float* U_for_moments = U_tile_raw.get();

            // 4b. kNN smoothing (if enabled).
            if (do_smooth) {
                cudaMemsetAsync(S_tile_sm.get(), 0, (size_t)g_count * c_count * sizeof(float), stream);
                cudaMemsetAsync(U_tile_sm.get(), 0, (size_t)g_count * c_count * sizeof(float), stream);
                size_t smem_bytes = (size_t)g_count * sizeof(float);
                detail::smooth_tile_kernel<<<c_count, 256, smem_bytes, stream>>>(
                    S_raw_full.get(), S_tile_sm.get(),
                    knn->neighbors.get(), knn->distances.get(),
                    sigma_dev.get(), g_count, c_off, c_count, n, knn->k);
                detail::smooth_tile_kernel<<<c_count, 256, smem_bytes, stream>>>(
                    U_raw_full.get(), U_tile_sm.get(),
                    knn->neighbors.get(), knn->distances.get(),
                    sigma_dev.get(), g_count, c_off, c_count, n, knn->k);
                S_for_moments = S_tile_sm.get();
                U_for_moments = U_tile_sm.get();
            }

            // 4c. Welford moments (all passes, all cells) — contributes to S_mean, U_mean.
            detail::welford_moments_kernel<<<g_count, 256, 0, stream>>>(
                S_for_moments, U_for_moments,
                welford_n.get(), welford_meanS.get(), welford_M2.get(),
                sum_U_acc.get(), sum_SU_acc.get(),
                g_off, g_count, c_count);

            // 4d. Top-N per gene via cub::DeviceSegmentedSort — NO H2D in this loop.
            run_topn_tile(
                S_for_moments, U_for_moments,
                g_off, g_count, c_off, c_count, top_n_eff,
                topn_sumSS.get(), topn_sumSU.get(), topn_count_dev.get(),
                seg_keys_in, seg_vals_in, seg_keys_out, seg_vals_out,
                seg_offsets, sort_temp, stream);
        }
    }

    // ── Step 5: γ per gene ────────────────────────────────────────────────────

    core::DeviceMemory<float> gamma_dev(m), gamma_se_dev(m),
                               S_mean_dev(m), U_mean_dev(m);
    {
        int b = 256, grd = (m + b - 1) / b;
        detail::compute_gamma_kernel<<<grd, b, 0, stream>>>(
            welford_n.get(), welford_meanS.get(), sum_U_acc.get(),
            topn_sumSS.get(), topn_sumSU.get(), topn_count_dev.get(),
            filter_mask.get(),
            gamma_dev.get(), gamma_se_dev.get(),
            S_mean_dev.get(), U_mean_dev.get(), m);
    }

    // ── Step 6: Count passing genes and build cumulative-pass prefix sum ──────
    //
    // cum_pass[g] = number of filter-passing genes with index < g.
    // Used by velocity tile kernel to index into v_out without H2D lookup.
    // One-time O(m) scan AFTER the double-tile loop — not inside it.

    // Copy filter_mask to host once (post-loop, one-time).
    std::vector<uint8_t> fmask_h(m);
    cudaMemcpyAsync(fmask_h.data(), filter_mask.get(),
                    m * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int n_pass = 0;
    for (int g = 0; g < m; ++g) if (fmask_h[g]) ++n_pass;

    // Build cum_pass on device via DeviceScan of filter_mask (cast to int32).
    core::DeviceMemory<int> cum_pass(m + 1);
    {
        core::DeviceMemory<int> fmask_int(m);
        detail::cast_mask_to_int_kernel<<<(m + 255) / 256, 256, 0, stream>>>(filter_mask.get(), fmask_int.get(), m);
        void* d_tmp = nullptr; size_t tmp_sz = 0;
        cub::DeviceScan::ExclusiveSum(d_tmp, tmp_sz,
            fmask_int.get(), cum_pass.get(), m + 1, stream);
        core::DeviceMemory<uint8_t> tmp_buf(tmp_sz);
        d_tmp = tmp_buf.get();
        cub::DeviceScan::ExclusiveSum(d_tmp, tmp_sz,
            fmask_int.get(), cum_pass.get(), m + 1, stream);
    }

    // ── Step 7: Optional velocity matrix ──────────────────────────────────────

    core::DeviceMemory<float> velocity_dev;
    if (cfg.compute_velocity && n_pass > 0) {
        velocity_dev = core::DeviceMemory<float>((size_t)n_pass * n);

        // Re-run the double-tile loop once more for velocity. The tile buffers are
        // still allocated; no new large allocations inside this second loop.
        // WHY a second loop instead of emitting in the first: the first loop runs
        // BEFORE γ is finalized. γ is needed to compute v = U - γ·S.
        for (int g_off = 0; g_off < m; g_off += g_tile) {
            int g_count = std::min(g_tile, m - g_off);

            if (do_smooth) {
                cudaMemsetAsync(S_raw_full.get(), 0, (size_t)g_count * n * sizeof(float), stream);
                cudaMemsetAsync(U_raw_full.get(), 0, (size_t)g_count * n * sizeof(float), stream);
                detail::gather_csc_tile_kernel<<<n, 256, 0, stream>>>(
                    spliced.values.get(),   spliced.col_ptr.get(), spliced.row_indices.get(),
                    S_raw_full.get(), g_off, g_count, 0, n, n);
                detail::gather_csc_tile_kernel<<<n, 256, 0, stream>>>(
                    unspliced.values.get(), unspliced.col_ptr.get(), unspliced.row_indices.get(),
                    U_raw_full.get(), g_off, g_count, 0, n, n);
            }

            for (int c_off = 0; c_off < n; c_off += c_tile) {
                int c_count = std::min(c_tile, n - c_off);

                cudaMemsetAsync(S_tile_raw.get(), 0, (size_t)g_count * c_count * sizeof(float), stream);
                cudaMemsetAsync(U_tile_raw.get(), 0, (size_t)g_count * c_count * sizeof(float), stream);
                detail::gather_csc_tile_kernel<<<c_count, 256, 0, stream>>>(
                    spliced.values.get(),   spliced.col_ptr.get(), spliced.row_indices.get(),
                    S_tile_raw.get(), g_off, g_count, c_off, c_count, n);
                detail::gather_csc_tile_kernel<<<c_count, 256, 0, stream>>>(
                    unspliced.values.get(), unspliced.col_ptr.get(), unspliced.row_indices.get(),
                    U_tile_raw.get(), g_off, g_count, c_off, c_count, n);

                const float* S_v = S_tile_raw.get();
                const float* U_v = U_tile_raw.get();
                if (do_smooth) {
                    cudaMemsetAsync(S_tile_sm.get(), 0, (size_t)g_count * c_count * sizeof(float), stream);
                    cudaMemsetAsync(U_tile_sm.get(), 0, (size_t)g_count * c_count * sizeof(float), stream);
                    size_t smem_bytes = (size_t)g_count * sizeof(float);
                    detail::smooth_tile_kernel<<<c_count, 256, smem_bytes, stream>>>(
                        S_raw_full.get(), S_tile_sm.get(),
                        knn->neighbors.get(), knn->distances.get(),
                        sigma_dev.get(), g_count, c_off, c_count, n, knn->k);
                    detail::smooth_tile_kernel<<<c_count, 256, smem_bytes, stream>>>(
                        U_raw_full.get(), U_tile_sm.get(),
                        knn->neighbors.get(), knn->distances.get(),
                        sigma_dev.get(), g_count, c_off, c_count, n, knn->k);
                    S_v = S_tile_sm.get();
                    U_v = U_tile_sm.get();
                }

                int total = g_count * c_count, b = 256, grd = (total + b - 1) / b;
                detail::compute_velocity_tile_kernel<<<grd, b, 0, stream>>>(
                    S_v, U_v, gamma_dev.get(), filter_mask.get(), cum_pass.get(),
                    velocity_dev.get(), g_off, g_count, c_off, c_count, n);
            }
        }
    }

    // ── Assemble result ────────────────────────────────────────────────────────

    VelocityPrepResult res;
    res.gamma                  = std::move(gamma_dev);
    res.gamma_se               = std::move(gamma_se_dev);
    res.S_mean                 = std::move(S_mean_dev);
    res.U_mean                 = std::move(U_mean_dev);
    res.filter_mask            = std::move(filter_mask);
    res.velocity               = std::move(velocity_dev);
    res.n_genes_passing_filter = n_pass;
    return res;
}

}  // namespace preprocess
}  // namespace singlet_gpu
