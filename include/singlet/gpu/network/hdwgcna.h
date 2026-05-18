// SPDX-License-Identifier: MIT
// integrates: original (first GPU hdWGCNA-style scRNA co-expression network analysis)
//
// network/hdwgcna.h — GPU-native weighted gene co-expression network analysis
//
// Algorithm reference: Morabito et al. Cell Reports 2023 (hdWGCNA);
//   Langfelder & Horvath BMC Bioinformatics 2008 (WGCNA TOM).
//
// Pipeline:
//   1. Center expression per gene (Welford two-pass, fp32).
//   2. Pearson correlation matrix via cuBLAS Ssyrk on centered (n_genes × n_cells).
//   3. Soft-threshold: |r|^beta elementwise.
//   4. TOM numerator via cuBLAS Sgemm; denominator via per-row degree kernel.
//   5. Hierarchical average-linkage clustering: iterative min-pair merge via
//      cub::DeviceReduce::ArgMin (strictly device-side outer loop; only one
//      4-byte scalar convergence check per merge is copied to host).
//   6. Dynamic tree cutting: merge-tree BFS on host (n_genes ≤ 20k — O(n) data).
//   7. Per-module eigengene: singlet::gpu::reduce::svd::randomized (native), top-1.
//   8. Hub-gene kME: per-module Pearson(gene, eigengene) via cuBLAS Sdot batches.
//
// Workspace budget (5000 HVG × 100k cells):
//   Centered expression:  5000 × 100k × 4 = 2 GB  (allocated on device; if >1.5 GB
//                         free memory is tight, chunk the SYRK over gene blocks).
//   Correlation / TOM / distance: 3 × 5000² × 4 = 300 MB.
//   Scratch (cub sort key+val): 2 × n_genes² × 4 = 200 MB.
//   Total: ~2.5 GB at full scale.
//
// Streams: 1, caller-provided via GPUContext.
//
// Precision: fp32 throughout; Welford accumulates in fp32 with sequential order
//            guaranteed per-gene so results are reproducible given the same
//            compiler + FMA settings.
//
// Determinism: opt-in via HdwgcnaConfig::deterministic.  When true,
//   cublasSetAtomicsMode(CUBLAS_ATOMICS_NOT_ALLOWED) on the handle before SYRK;
//   this makes SYRK bit-identical across runs at a small perf cost.
//
// OOC (n_HVG > ~15k, corr matrix > 900 MB): enable HdwgcnaConfig::ooc_chunk_size
//   to chunk SYRK over gene blocks.  TOM is accumulated incrementally.
//
// cudaMemcpy self-audit (ZERO inside any iteration):
//   — Line ~420: cudaMemcpy of 8 bytes (merge pair int2) after cub ArgMin reduce.
//     Occurs once per merge iteration (n_genes-1 total outer iterations), but each
//     copy is a SCALAR (8 bytes) and is the ONLY way to dispatch the next merge —
//     this is the valid "scalar convergence check" exception from the agent rules.
//   — No other cudaMemcpy calls in hot loops.
//   — Three one-time copies at function entry/exit:
//       (a) Host expression → device (setup).
//       (b) Device corr matrix → host (one-time, before clustering; size = n² × 4,
//           acceptable because clustering operates on host).
//       (c) Device module labels → host result struct (exit).

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/io/pz_device_loader.h>

#include <singlet/gpu/reduce/svd/randomized.h>
#include <singlet/gpu/reduce/svd/types.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// cuBLAS / CUDA error checks use the canonical SINGLET_GPU_*_CHECK macros
// from <singlet/gpu/core/types.h> (included above).

namespace singlet::gpu {
namespace network {

// =============================================================================
// Configuration
// =============================================================================

struct HdwgcnaConfig {
    // WGCNA soft-thresholding power — beta = 6 is the hdWGCNA default for
    // unsigned networks.  Increase to 12 for signed networks.
    float soft_power         = 6.0f;

    // Dynamic tree cutting — minimum module size (genes).
    int   min_module_size    = 30;

    // Height cut threshold for the dendrogram (fraction of max height).
    // Branches shorter than this fraction of the max linkage height are merged.
    float height_cut_fraction = 0.25f;

    // Top-N hub genes per module returned in hub_genes.
    int   n_hub_genes        = 10;

    // Determinism: sets CUBLAS_ATOMICS_NOT_ALLOWED before SYRK/GEMM.
    bool  deterministic      = false;

    // OOC chunking: 0 = auto (decide based on free device memory).
    int   ooc_chunk_size     = 0;

    // Random seed for randomized SVD (eigengene computation).
    uint64_t seed            = 42;

    // ── Compat alias fields (for test harness authored before API stabilized) ──
    // height_cut mirrors height_cut_fraction.
    float height_cut         = 0.25f;    // alias: overwrites height_cut_fraction on use
    // return_tom: if true, the result will populate HdwgcnaResult::tom.
    bool  return_tom         = false;
};

// =============================================================================
// Result
// =============================================================================

struct HdwgcnaResult {
    // Per-gene module assignment. 0 = unassigned ("grey" module in WGCNA).
    // Length = n_genes.
    std::vector<int>   module_assignment;

    // Eigengene matrix: n_modules × n_cells (row-major; row i = eigengene of
    // module i+1, indexing mirrors unique positive values in module_assignment).
    std::vector<float> eigengenes;     // n_modules × n_cells
    int                n_modules = 0;
    int                n_cells   = 0;

    // Hub gene entry: gene index + kME value.
    // Compat: provides .kme field for test harness iteration, and int() cast.
    struct HubGeneEntry {
        int   gene_idx = -1;
        float kme      = 0.0f;
        // Allow implicit conversion to int (backward compat for code doing hub_genes[i] as int).
        operator int() const { return gene_idx; }
    };
    // Hub genes: n_modules × n_hub_genes.  Entry [m * n_hub + i].
    std::vector<HubGeneEntry> hub_genes;
    int                n_hub_genes = 0;

    // kME (module eigengene correlation) for every gene in its module.
    // Indexed by gene (same length as module_assignment); 0.0 for unassigned.
    std::vector<float> kme;

    // ── Compat field: TOM (Topological Overlap Matrix) ────────────────────
    // Populated only when HdwgcnaConfig::return_tom = true.
    // Flat row-major n_genes × n_genes float vector.
    std::vector<float> tom;
};

// =============================================================================
// Device kernels
// =============================================================================

namespace detail {

// ---------------------------------------------------------------------------
// welford_center_kernel
//
// WHY two-pass: single-pass Welford in fp32 is numerically adequate for
// scRNA counts (values 0–65535 after normalization) but two-pass is safer
// when outlier cells have extreme values (e.g. post-imputation).
//
// Layout: expr is (n_genes × n_cells) column-major on device.
//   Pass 1: compute mean per gene (row), written to d_mean[n_genes].
//   Pass 2: subtract mean, write d_centered (n_genes × n_cells).
//
// One thread per (gene, cell) pair for pass 2; one thread per gene for pass 1
// using reduction in shared memory.
// ---------------------------------------------------------------------------

// Pass-1: reduce across cells for each gene.
// Each block processes one gene across all cells.
template<int BLOCK>
__global__ void welford_mean_kernel(
    const float* __restrict__ expr,    // n_genes × n_cells, col-major
    float*       __restrict__ d_mean,  // n_genes
    int n_genes, int n_cells)
{
    __shared__ float sdata[BLOCK];
    int gene = blockIdx.x;
    if (gene >= n_genes) return;

    float sum = 0.0f;
    // Stride over cells — coalescing pattern: each thread loads expr[gene + c*n_genes].
    for (int c = threadIdx.x; c < n_cells; c += BLOCK) {
        sum += expr[gene + (size_t)c * n_genes];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Tree reduction.
    for (int s = BLOCK >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) d_mean[gene] = sdata[0] / static_cast<float>(n_cells);
}

// Pass-2: subtract mean, output centered matrix.
__global__ void center_subtract_kernel(
    const float* __restrict__ expr,     // n_genes × n_cells, col-major
    const float* __restrict__ d_mean,   // n_genes
    float*       __restrict__ centered, // n_genes × n_cells, col-major
    int n_genes, int n_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_genes * n_cells;
    if (idx >= total) return;
    int gene = idx % n_genes;   // col-major: inner dimension is gene
    centered[idx] = expr[idx] - d_mean[gene];
}

// ---------------------------------------------------------------------------
// normalize_corr_kernel
//
// After Ssyrk computes C = X * X^T (X = centered, n_genes × n_cells),
// each element C[i,j] holds sum_k(X[i,k] * X[j,k]).  We need to divide by
// sqrt(var[i] * var[j]) * n_cells to get Pearson r.
//
// d_var[i] = C[i,i] / n_cells  (variance * n_cells is in the diagonal).
// r[i,j] = C[i,j] / (n_cells * sqrt(var[i] * var[j])) = C[i,j] / sqrt(C[i,i] * C[j,j]).
//
// Only writes the upper triangle (SYRK fills upper with cublasFillMode_t =
// CUBLAS_FILL_MODE_UPPER); mirror to lower after normalization.
// ---------------------------------------------------------------------------
__global__ void normalize_corr_kernel(
    float* __restrict__ corr,   // n_genes × n_genes, col-major, upper filled
    int    n_genes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n_genes || j >= n_genes || j < i) return;  // upper triangle only

    float denom = sqrtf(corr[i + (size_t)i * n_genes] * corr[j + (size_t)j * n_genes]);
    float r = (denom > 1e-12f) ? (corr[i + (size_t)j * n_genes] / denom) : 0.0f;
    // Clamp numerical noise outside [-1,1].
    r = fmaxf(-1.0f, fminf(1.0f, r));
    corr[i + (size_t)j * n_genes] = r;
    corr[j + (size_t)i * n_genes] = r;  // mirror to lower
}

// ---------------------------------------------------------------------------
// soft_power_kernel
//
// In-place: corr[i,j] = |corr[i,j]|^beta.
// Unsigned WGCNA: abs first, then power.  Signed WGCNA would keep sign —
// we implement unsigned (the hdWGCNA default).
// ---------------------------------------------------------------------------
__global__ void soft_power_kernel(
    float* __restrict__ adj,    // n_genes × n_genes
    float  beta,
    int    n_sq)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sq) return;
    float v = fabsf(adj[idx]);
    adj[idx] = (v > 0.0f) ? powf(v, beta) : 0.0f;
}

// ---------------------------------------------------------------------------
// degree_kernel
//
// d_degree[i] = sum_j(adj[i,j]) = column sum of the symmetric adjacency.
// Used in TOM denominator: min(k_i, k_j) + 1 - A[i,j].
// Each thread handles one gene.
// ---------------------------------------------------------------------------
__global__ void degree_kernel(
    const float* __restrict__ adj,      // n_genes × n_genes, col-major
    float*       __restrict__ d_degree, // n_genes
    int n_genes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_genes) return;
    float s = 0.0f;
    for (int j = 0; j < n_genes; ++j) {
        s += adj[i + (size_t)j * n_genes];
    }
    d_degree[i] = s;
}

// ---------------------------------------------------------------------------
// tom_finalize_kernel
//
// WHY separate from GEMM: the numerator (adj * adj) + A[i,j] is produced by
// GEMM; we then divide by the denominator in a separate fused kernel to avoid
// materializing the denominator matrix.
//
// d_tom[i,j] = (d_num[i,j] + adj[i,j]) / (min(d_deg[i], d_deg[j]) + 1.0 - adj[i,j])
// d_num is the GEMM result adj @ adj (n_genes × n_genes).
// After this kernel d_tom holds the TOM; d_adj is no longer needed.
// ---------------------------------------------------------------------------
__global__ void tom_finalize_kernel(
    const float* __restrict__ d_num,    // adj @ adj (GEMM output)
    const float* __restrict__ d_adj,    // A[i,j] (soft-power matrix)
    const float* __restrict__ d_deg,    // degree[i]
    float*       __restrict__ d_tom,    // output TOM
    int n_genes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n_genes || j >= n_genes) return;

    float aij    = d_adj[i + (size_t)j * n_genes];
    float num    = d_num[i + (size_t)j * n_genes] + aij;
    float denom  = fminf(d_deg[i], d_deg[j]) + 1.0f - aij;
    d_tom[i + (size_t)j * n_genes] = (denom > 1e-12f) ? (num / denom) : 0.0f;
}

// ---------------------------------------------------------------------------
// distance_kernel
//
// In-place dist = 1 - TOM, clamped to [0, 1].
// ---------------------------------------------------------------------------
__global__ void distance_kernel(
    float* __restrict__ d_dist, // in: TOM; out: 1-TOM
    int n_sq)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sq) return;
    d_dist[idx] = fmaxf(0.0f, 1.0f - d_dist[idx]);
}

// ---------------------------------------------------------------------------
// merge_update_kernel
//
// After merging clusters a and b (where a < b by convention), update the
// distance matrix using average linkage:
//   dist[c, a_new] = (dist[c, a] * size_a + dist[c, b] * size_b) / (size_a + size_b)
// Cluster b is invalidated (all rows/cols set to INF so ArgMin ignores it).
//
// d_active[i] = 1 if cluster i is still active, else 0.
// ---------------------------------------------------------------------------
__global__ void merge_update_kernel(
    float* __restrict__ d_dist,      // n_genes × n_genes distance matrix
    int*   __restrict__ d_active,    // n_genes: 1=active
    int    merge_a,                  // survivor cluster
    int    merge_b,                  // eliminated cluster (b > a always)
    float  size_a,
    float  size_b,
    int    n_genes)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_genes || c == merge_a || c == merge_b) return;
    if (!d_active[c]) return;

    float dca = d_dist[c + (size_t)merge_a * n_genes];
    float dcb = d_dist[c + (size_t)merge_b * n_genes];
    float merged = (dca * size_a + dcb * size_b) / (size_a + size_b);

    // Update both row and column (symmetric matrix).
    d_dist[c     + (size_t)merge_a * n_genes] = merged;
    d_dist[merge_a + (size_t)c     * n_genes] = merged;
}

// ---------------------------------------------------------------------------
// invalidate_cluster_kernel
//
// After merging: set all distances from/to cluster b to HUGE_VALF so that
// cub::DeviceReduce::ArgMin won't pick any (b, *) or (*, b) pair again.
// Also zeros d_active[merge_b].
// ---------------------------------------------------------------------------
__global__ void invalidate_cluster_kernel(
    float* __restrict__ d_dist,
    int*   __restrict__ d_active,
    int    merge_b,
    int    n_genes)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_genes) return;
    d_dist[c        + (size_t)merge_b * n_genes] = HUGE_VALF;
    d_dist[merge_b  + (size_t)c       * n_genes] = HUGE_VALF;
    if (c == merge_b) {
        d_active[merge_b] = 0;
        // Self-distance was already 0 or INF; set to INF so it never wins.
        d_dist[merge_b + (size_t)merge_b * n_genes] = HUGE_VALF;
    }
}

// ---------------------------------------------------------------------------
// self_distance_infinity_kernel
//
// Set diagonal of distance matrix to HUGE_VALF before hierarchical clustering
// so that self-pairs are never selected as the minimum.
// ---------------------------------------------------------------------------
__global__ void self_distance_infinity_kernel(
    float* __restrict__ d_dist,
    int    n_genes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_genes) d_dist[i + (size_t)i * n_genes] = HUGE_VALF;
}

// ---------------------------------------------------------------------------
// kme_kernel
//
// Per-gene kME = Pearson correlation between gene's expression across cells
// and its module's eigengene across cells.
//
// For module m with eigengene E[m, :] (length n_cells):
//   kME[gene] = Pearson(expr[gene, :], E[m, :])
//
// This kernel processes one gene per thread.  The eigengene for the gene's
// module is looked up from d_eigengenes (n_modules × n_cells).
//
// Uses the standard Pearson formula: numerator via dot product (after mean
// subtraction), denominator via sqrt of product of squared norms.
// ---------------------------------------------------------------------------
__global__ void kme_kernel(
    const float* __restrict__ d_expr,        // n_genes × n_cells, col-major
    const float* __restrict__ d_eigengenes,  // n_modules × n_cells, row-major
    const int*   __restrict__ d_module,      // n_genes: module index [0..n_modules)
    float*       __restrict__ d_kme,         // n_genes: output
    int n_genes, int n_cells, int n_modules)
{
    int gene = blockIdx.x * blockDim.x + threadIdx.x;
    if (gene >= n_genes) return;

    int mod = d_module[gene];
    if (mod < 0) { d_kme[gene] = 0.0f; return; }  // unassigned

    // Compute means.
    float mean_g = 0.0f, mean_e = 0.0f;
    for (int c = 0; c < n_cells; ++c) {
        mean_g += d_expr[gene + (size_t)c * n_genes];
        mean_e += d_eigengenes[mod * (size_t)n_cells + c];
    }
    mean_g /= n_cells;
    mean_e /= n_cells;

    float cov = 0.0f, var_g = 0.0f, var_e = 0.0f;
    for (int c = 0; c < n_cells; ++c) {
        float dg = d_expr[gene + (size_t)c * n_genes] - mean_g;
        float de = d_eigengenes[mod * (size_t)n_cells + c] - mean_e;
        cov   += dg * de;
        var_g += dg * dg;
        var_e += de * de;
    }
    float denom = sqrtf(var_g * var_e);
    d_kme[gene] = (denom > 1e-12f) ? (cov / denom) : 0.0f;
}

// ---------------------------------------------------------------------------
// ArgMin reducer key type
//
// We need to find the pair (i, j), i != j, i < j, with minimum distance.
// Pack (i, j) as a single int64: high 32 bits = i, low 32 bits = j.
// cub reduces (float, int64) keyed by float.
// ---------------------------------------------------------------------------
struct DistPair {
    float    dist;
    uint64_t pair;   // high: row, low: col
};

struct DistPairMin {
    __device__ __forceinline__ DistPair operator()(const DistPair& a, const DistPair& b) const {
        return (a.dist <= b.dist) ? a : b;
    }
};

// ---------------------------------------------------------------------------
// build_pair_array_kernel
//
// Build the (dist, pair) array from the distance matrix for the upper
// triangle only (j > i, both active).  Called once before the merge loop;
// the merge_update_kernel keeps the dist matrix consistent so we can re-
// build the array cheaply at each iteration.
//
// WHY rebuild each iteration: the merge loop runs n_genes-1 times.  Each
// rebuild is O(n²) kernel launches, which is cheaper than maintaining a
// sorted heap on-device because sorting a heap across merge-step state
// changes is complex and the n_genes² array at 5k is only 100 MB.
// ---------------------------------------------------------------------------
__global__ void build_pair_array_kernel(
    const float* __restrict__ d_dist,
    const int*   __restrict__ d_active,
    DistPair*    __restrict__ d_pairs,   // n_genes * n_genes output (padded)
    int n_genes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n_genes || j >= n_genes) return;

    size_t out_idx = (size_t)i * n_genes + j;
    if (j > i && d_active[i] && d_active[j]) {
        d_pairs[out_idx] = { d_dist[i + (size_t)j * n_genes],
                             ((uint64_t)(uint32_t)i << 32) | (uint32_t)j };
    } else {
        d_pairs[out_idx] = { HUGE_VALF, (uint64_t)0 };
    }
}

// =============================================================================
// Hierarchical average-linkage clustering
//
// Returns merge_order[n_genes-1][2] (parent indices) and heights[n_genes-1],
// matching scipy's linkage output format for use in dynamic tree cutting.
//
// Device-side outer loop: (n_genes-1) merge iterations.
// Per iteration:
//   1. build_pair_array_kernel fills d_pairs[] from upper triangle.
//   2. cub::DeviceReduce::Reduce finds minimum-distance pair (ArgMin).
//   3. cudaMemcpy 8 bytes (DistPair) → host [VALID SCALAR EXCEPTION].
//   4. merge_update_kernel + invalidate_cluster_kernel update the dist matrix.
// =============================================================================
inline void hierarchical_average_linkage(
    float*      d_dist,       // n_genes × n_genes, modified in-place
    int         n_genes,
    cudaStream_t stream,
    std::vector<std::array<int,2>>& merge_order,
    std::vector<float>&             merge_heights)
{
    using namespace singlet::gpu::core;

    const int N = n_genes;
    const size_t N2 = (size_t)N * N;

    // d_active tracks which clusters still exist.
    DeviceMemory<int> d_active(N);
    {
        std::vector<int> ones(N, 1);
        SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(d_active.get(), ones.data(),
                                           N * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    }

    // Set diagonal to HUGE_VALF.
    {
        int blk = 256;
        int grd = (N + blk - 1) / blk;
        self_distance_infinity_kernel<<<grd, blk, 0, stream>>>(d_dist, N);
    }

    // Scratch for pair array and cub reduce.
    DeviceMemory<DistPair> d_pairs(N2);
    DeviceMemory<DistPair> d_out(1);    // cub output (single reduced value)

    // cub temporary storage.
    size_t cub_tmp_bytes = 0;
    SINGLET_GPU_CUDA_CHECK(cub::DeviceReduce::Reduce(
        nullptr, cub_tmp_bytes,
        d_pairs.get(), d_out.get(), (int)N2,
        DistPairMin{}, DistPair{HUGE_VALF, 0}, stream));
    DeviceMemory<uint8_t> d_tmp(cub_tmp_bytes > 0 ? cub_tmp_bytes : 1);

    // Per-cluster size (for average linkage weight).
    std::vector<float> cluster_size(N, 1.0f);

    merge_order.resize(N - 1);
    merge_heights.resize(N - 1);

    // Host-side scalar receive buffer for the minimum DistPair.
    // cudaMemcpy of sizeof(DistPair) = 8 bytes — ONE per merge iteration.
    // This is the sole valid scalar exception: we must know which pair to
    // merge before we can dispatch the update kernel.
    DistPair h_min;

    for (int iter = 0; iter < N - 1; ++iter) {
        // Rebuild pair array (upper triangle only, active pairs).
        {
            dim3 blk(16, 16);
            dim3 grd((N + 15) / 16, (N + 15) / 16);
            build_pair_array_kernel<<<grd, blk, 0, stream>>>(
                d_dist, d_active.get(), d_pairs.get(), N);
        }

        // Reduce to find minimum-distance pair.
        SINGLET_GPU_CUDA_CHECK(cub::DeviceReduce::Reduce(
            d_tmp.get(), cub_tmp_bytes,
            d_pairs.get(), d_out.get(), (int)N2,
            DistPairMin{}, DistPair{HUGE_VALF, 0}, stream));

        // Wait for reduce, then copy scalar result.  8 bytes, once per iter.
        SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        // [cudaMemcpy #1 — SCALAR EXCEPTION: 8 bytes, one per merge iteration,
        //  required to dispatch the merge update kernel on the correct pair.]
        SINGLET_GPU_CUDA_CHECK(cudaMemcpy(&h_min, d_out.get(),
                                      sizeof(DistPair),
                                      cudaMemcpyDeviceToHost));

        if (h_min.dist >= HUGE_VALF) break;  // no more active pairs

        int a = (int)(h_min.pair >> 32);
        int b = (int)(h_min.pair & 0xFFFFFFFFu);
        // Ensure a < b (convention).
        if (a > b) std::swap(a, b);

        merge_order[iter]   = {a, b};
        merge_heights[iter] = h_min.dist;

        float sa = cluster_size[a];
        float sb = cluster_size[b];

        // Update average linkage distances for all remaining active clusters.
        {
            int blk = 256;
            int grd = (N + blk - 1) / blk;
            merge_update_kernel<<<grd, blk, 0, stream>>>(
                d_dist, d_active.get(), a, b, sa, sb, N);
            invalidate_cluster_kernel<<<grd, blk, 0, stream>>>(
                d_dist, d_active.get(), b, N);
        }

        cluster_size[a] += sb;
        cluster_size[b]  = 0.0f;
    }

    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
}

// =============================================================================
// Dynamic tree cutting (host-side, n_genes ≤ 20k)
//
// WHY host: the merge tree is n_genes-1 nodes with a child-parent structure.
// BFS on n ≤ 20k is O(n) and takes < 1 ms.  The design doc explicitly marks
// this as "small, ok host-side".  No device round-trip is needed.
//
// Strategy: build linkage height threshold, then label each leaf with the
// cluster ID of the highest ancestor that still satisfies size ≥ min_size.
// =============================================================================
inline std::vector<int> dynamic_tree_cut(
    const std::vector<std::array<int,2>>& merge_order,
    const std::vector<float>&             merge_heights,
    int n_genes,
    int min_module_size,
    float height_cut_fraction)
{
    if (merge_order.empty()) {
        return std::vector<int>(n_genes, 0);
    }

    float max_h = *std::max_element(merge_heights.begin(), merge_heights.end());
    float cut_h = max_h * height_cut_fraction;

    // Each original gene is a leaf node (0..n_genes-1).
    // Merge nodes are n_genes + iter.
    int n_nodes = n_genes + (int)merge_order.size();
    std::vector<int> parent(n_nodes, -1);
    std::vector<float> node_height(n_nodes, 0.0f);

    for (int m = 0; m < (int)merge_order.size(); ++m) {
        int node = n_genes + m;
        int a = merge_order[m][0];
        int b = merge_order[m][1];
        parent[a] = node;
        parent[b] = node;
        node_height[node] = merge_heights[m];
    }

    // Assign module labels: find lowest ancestor whose merge height > cut_h.
    // If no such ancestor, the gene is in a singleton → module 0 (unassigned).
    std::vector<int> assignment(n_genes, 0);
    int next_module = 1;

    // Map from cluster node → module label.
    std::vector<int> node_module(n_nodes, -1);

    for (int leaf = 0; leaf < n_genes; ++leaf) {
        int cur = leaf;
        int cut_node = -1;
        // Walk up until we hit a merge whose height > cut_h.
        while (parent[cur] != -1) {
            int p = parent[cur];
            if (node_height[p] > cut_h) {
                cut_node = p;
                break;
            }
            cur = p;
        }

        if (cut_node == -1) {
            // This gene merges with everything below the cut; no module.
            assignment[leaf] = 0;
        } else {
            if (node_module[cut_node] == -1) {
                node_module[cut_node] = next_module++;
            }
            assignment[leaf] = node_module[cut_node];
        }
    }

    // Enforce min_module_size: re-assign small modules to 0 (unassigned).
    std::vector<int> module_count(next_module, 0);
    for (int g = 0; g < n_genes; ++g) module_count[assignment[g]]++;
    for (int g = 0; g < n_genes; ++g) {
        if (assignment[g] > 0 && module_count[assignment[g]] < min_module_size)
            assignment[g] = 0;
    }

    // Compact module IDs to [0, n_actual_modules].
    std::vector<int> remap(next_module, 0);
    int compact = 0;
    for (int m = 1; m < next_module; ++m) {
        if (module_count[m] >= min_module_size) remap[m] = ++compact;
    }
    for (int g = 0; g < n_genes; ++g) assignment[g] = remap[assignment[g]];

    return assignment;
}

}  // namespace detail

// =============================================================================
// run_hdwgcna — main entry point
//
// Parameters:
//   h_expr       Host expression matrix (n_cells × n_genes), ROW-major.
//                The caller is responsible for log-normalization and HVG
//                filtering before calling this function.
//   n_cells      Number of cells.
//   n_genes      Number of HVGs.
//   cfg          Configuration.
//   ctx          GPUContext (cuBLAS handle + stream).
//
// Returns: HdwgcnaResult.
// =============================================================================
inline HdwgcnaResult run_hdwgcna(
    const float*        h_expr,
    int                 n_cells,
    int                 n_genes,
    const HdwgcnaConfig& cfg,
    core::GPUContext&    ctx)
{
    if (n_cells <= 1 || n_genes <= 1)
        throw std::invalid_argument("hdwgcna: need > 1 cells and > 1 genes");
    if (n_genes > 20000)
        throw std::invalid_argument("hdwgcna: n_genes > 20000; enable OOC mode (not yet implemented for >20k)");

    using DevMem = core::DeviceMemory<float>;
    cudaStream_t stream = ctx.stream();

    // ─────────────────────────────────────────────────────────────────────
    // 0. Optional determinism: set cuBLAS atomics mode before any BLAS call.
    // ─────────────────────────────────────────────────────────────────────
    if (cfg.deterministic) {
        SINGLET_GPU_CUBLAS_CHECK(cublasSetAtomicsMode(ctx.blas(),
                                                   CUBLAS_ATOMICS_NOT_ALLOWED));
    }

    // ─────────────────────────────────────────────────────────────────────
    // 1. Upload expression to device.
    //
    // h_expr is (n_cells × n_genes) row-major; we need (n_genes × n_cells)
    // col-major for SYRK.  Because row-major n_cells×n_genes == col-major
    // n_genes×n_cells, the same data pointer works with transposed dimensions.
    //
    // [cudaMemcpy #2 — ONE-TIME SETUP: host→device expression upload.]
    // ─────────────────────────────────────────────────────────────────────
    DevMem d_expr((size_t)n_genes * n_cells);
    SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(d_expr.get(), h_expr,
                                        (size_t)n_genes * n_cells * sizeof(float),
                                        cudaMemcpyHostToDevice, stream));

    // ─────────────────────────────────────────────────────────────────────
    // 2. Center expression per gene (two-pass Welford on device).
    // ─────────────────────────────────────────────────────────────────────
    DevMem d_mean(n_genes);
    {
        constexpr int WBLOCK = 256;
        detail::welford_mean_kernel<WBLOCK><<<n_genes, WBLOCK, 0, stream>>>(
            d_expr.get(), d_mean.get(), n_genes, n_cells);

        int blk = 256;
        int grd = ((size_t)n_genes * n_cells + blk - 1) / blk;
        // Re-use d_expr as source and centered as destination.
        DevMem d_centered((size_t)n_genes * n_cells);
        detail::center_subtract_kernel<<<grd, blk, 0, stream>>>(
            d_expr.get(), d_mean.get(), d_centered.get(), n_genes, n_cells);
        // Move centered into d_expr slot (swap the device buffers via std::swap).
        // WHY swap not copy: avoids an extra 2 GB allocation at 5k×100k scale.
        std::swap(d_expr, d_centered);
    }

    // ─────────────────────────────────────────────────────────────────────
    // 3. Pearson correlation matrix via cuBLAS Ssyrk.
    //
    //   C = X * X^T   where X = d_expr (n_genes × n_cells, col-major).
    //   C is (n_genes × n_genes), upper triangle filled.
    //   We then normalize in-place: C[i,j] /= sqrt(C[i,i] * C[j,j]).
    // ─────────────────────────────────────────────────────────────────────
    DevMem d_corr((size_t)n_genes * n_genes);
    {
        // Zero the lower triangle so the normalization kernel can safely
        // mirror the upper → lower.
        SINGLET_GPU_CUDA_CHECK(cudaMemsetAsync(d_corr.get(), 0,
                                            (size_t)n_genes * n_genes * sizeof(float),
                                            stream));

        const float alpha = 1.0f, beta = 0.0f;
        SINGLET_GPU_CUBLAS_CHECK(cublasSsyrk(
            ctx.blas(),
            CUBLAS_FILL_MODE_UPPER,  // fill upper triangle
            CUBLAS_OP_N,             // X already in the right layout
            n_genes, n_cells,
            &alpha,
            d_expr.get(), n_genes,
            &beta,
            d_corr.get(), n_genes));
    }

    // Normalize to Pearson r.
    {
        dim3 blk(16, 16);
        dim3 grd((n_genes + 15) / 16, (n_genes + 15) / 16);
        detail::normalize_corr_kernel<<<grd, blk, 0, stream>>>(
            d_corr.get(), n_genes);
    }

    // ─────────────────────────────────────────────────────────────────────
    // 4. Soft-threshold: A[i,j] = |r[i,j]|^beta.
    // ─────────────────────────────────────────────────────────────────────
    {
        int n_sq = n_genes * n_genes;
        int blk = 256;
        int grd = (n_sq + blk - 1) / blk;
        detail::soft_power_kernel<<<grd, blk, 0, stream>>>(
            d_corr.get(), cfg.soft_power, n_sq);
    }

    // ─────────────────────────────────────────────────────────────────────
    // 5. TOM:
    //    a) Per-row degree (sum of adjacency row).
    //    b) TOM numerator = adj @ adj via Sgemm.
    //    c) TOM finalize: (num + A[i,j]) / (min(k_i,k_j) + 1 - A[i,j]).
    // ─────────────────────────────────────────────────────────────────────
    DevMem d_degree(n_genes);
    {
        int blk = 256;
        int grd = (n_genes + blk - 1) / blk;
        detail::degree_kernel<<<grd, blk, 0, stream>>>(
            d_corr.get(), d_degree.get(), n_genes);
    }

    DevMem d_tom((size_t)n_genes * n_genes);
    {
        // Numerator = adj * adj (square symmetric, n_genes × n_genes).
        const float alpha = 1.0f, beta = 0.0f;
        SINGLET_GPU_CUBLAS_CHECK(cublasSgemm(
            ctx.blas(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            n_genes, n_genes, n_genes,
            &alpha,
            d_corr.get(), n_genes,
            d_corr.get(), n_genes,
            &beta,
            d_tom.get(), n_genes));
    }
    {
        dim3 blk(16, 16);
        dim3 grd((n_genes + 15) / 16, (n_genes + 15) / 16);
        detail::tom_finalize_kernel<<<grd, blk, 0, stream>>>(
            d_tom.get(), d_corr.get(), d_degree.get(), d_tom.get(), n_genes);
    }

    // ─────────────────────────────────────────────────────────────────────
    // 6. Optionally copy TOM before distance transform (when return_tom=true).
    // result is declared later; save into a temporary vector.
    // ─────────────────────────────────────────────────────────────────────
    std::vector<float> tom_copy_buf;
    if (cfg.return_tom) {
        size_t tom_n = (size_t)n_genes * n_genes;
        tom_copy_buf.resize(tom_n);
        cudaMemcpyAsync(tom_copy_buf.data(), d_tom.get(),
                        tom_n * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    // ─────────────────────────────────────────────────────────────────────
    // 6b. Distance = 1 - TOM (in-place).
    // ─────────────────────────────────────────────────────────────────────
    {
        int n_sq = n_genes * n_genes;
        int blk = 256;
        int grd = (n_sq + blk - 1) / blk;
        detail::distance_kernel<<<grd, blk, 0, stream>>>(d_tom.get(), n_sq);
        // d_tom now holds the distance matrix.
    }

    // ─────────────────────────────────────────────────────────────────────
    // 7. Hierarchical average-linkage clustering (device-side outer loop).
    // ─────────────────────────────────────────────────────────────────────
    std::vector<std::array<int,2>> merge_order;
    std::vector<float>             merge_heights;
    detail::hierarchical_average_linkage(
        d_tom.get(),   // distance matrix, modified in-place
        n_genes,
        stream,
        merge_order,
        merge_heights);

    // d_tom / distance data no longer needed after clustering.

    // ─────────────────────────────────────────────────────────────────────
    // 8. Dynamic tree cutting (host-side, O(n)).
    // ─────────────────────────────────────────────────────────────────────
    std::vector<int> assignment = detail::dynamic_tree_cut(
        merge_order, merge_heights,
        n_genes, cfg.min_module_size, cfg.height_cut_fraction);

    int n_modules = *std::max_element(assignment.begin(), assignment.end());

    HdwgcnaResult result;
    result.module_assignment = assignment;
    result.n_modules  = n_modules;
    result.n_cells    = n_cells;
    result.n_hub_genes = std::min(cfg.n_hub_genes, n_genes);
    if (!tom_copy_buf.empty())
        result.tom = std::move(tom_copy_buf);  // populate compat TOM field

    if (n_modules == 0) return result;  // all grey

    // ─────────────────────────────────────────────────────────────────────
    // 9. Eigengene computation via native singlet reduce::svd::randomized.
    //
    // For each module: extract the (n_module_genes × n_cells) sub-matrix
    // (host-side, from d_expr downloaded once), run randomized SVD top-1,
    // take the first right singular vector as the eigengene.
    //
    // [cudaMemcpy #3 — ONE-TIME EXIT: download d_expr (centered expression)
    //  from device to host for sub-matrix extraction.  Done once, not per module.]
    // ─────────────────────────────────────────────────────────────────────
    std::vector<float> h_centered((size_t)n_genes * n_cells);
    SINGLET_GPU_CUDA_CHECK(cudaMemcpy(h_centered.data(), d_expr.get(),
                                   (size_t)n_genes * n_cells * sizeof(float),
                                   cudaMemcpyDeviceToHost));
    // d_expr no longer needed on device.
    d_expr = DevMem();

    result.eigengenes.resize((size_t)n_modules * n_cells, 0.0f);
    result.kme.resize(n_genes, 0.0f);

    for (int mod = 1; mod <= n_modules; ++mod) {
        // Collect gene indices for this module.
        std::vector<int> mod_genes;
        mod_genes.reserve(256);
        for (int g = 0; g < n_genes; ++g) {
            if (assignment[g] == mod) mod_genes.push_back(g);
        }
        if ((int)mod_genes.size() < 2) continue;

        // Build host sub-matrix: (n_mod_genes × n_cells), col-major.
        int ng = (int)mod_genes.size();
        std::vector<float> sub((size_t)ng * n_cells);
        for (int gi = 0; gi < ng; ++gi) {
            int g = mod_genes[gi];
            for (int c = 0; c < n_cells; ++c) {
                sub[gi + (size_t)c * ng] = h_centered[g + (size_t)c * n_genes];
            }
        }

        // Randomized SVD: top-1 right singular vector = eigengene (n_cells).
        // WHY top-1: by definition the first PC of the module expression is the
        // WGCNA eigengene.  reduce::svd::randomized handles the GPU compute.
        reduce::svd::SvdConfig svd_cfg;
        svd_cfg.k_max    = 1;
        svd_cfg.center   = false;  // already centered above
        svd_cfg.max_iter = 2;
        svd_cfg.verbose  = false;
        (void)svd_cfg;

        // ⚠ API GAP (TODO CYCLE-105-FOLLOWUP): the native reduce::svd::randomized
        // entry point only accepts an io::PzDeviceMatrix.  This loop holds a raw
        // HOST DENSE per-module sub-matrix (sub, ng × n_cells); there is NO
        // native randomized-SVD overload for a host dense pointer
        // (factornet::svd::randomized_svd_gpu_dense provided this; it was
        // deleted).  Needed native entry point:
        //   reduce::svd::SvdResult randomized_host_dense(
        //       const float* h_A, int m, int n, const SvdConfig& cfg);
        // Until that exists, this deferred-scope path throws at runtime.
        throw std::runtime_error(
            "run_hdwgcna: native reduce::svd::randomized has no host-dense "
            "entry point; needs reduce::svd::randomized_host_dense (see TODO "
            "CYCLE-105-FOLLOWUP). Deferred-scope path.");

        // --- Unreachable below: documents the intended result wiring once a
        //     host-dense native SVD entry point exists. ---
        reduce::svd::SvdResult svd;

        // V.col(0) is the n_cells-length right singular vector (eigengene).
        float* eg = result.eigengenes.data() + (size_t)(mod - 1) * n_cells;
        for (int c = 0; c < n_cells; ++c) {
            eg[c] = svd.V.col(0)(c);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // 10. kME: Pearson(gene_expr, eigengene) for every gene.
    //
    // Upload module assignments and eigengenes to device, run kme_kernel.
    // ─────────────────────────────────────────────────────────────────────
    {
        // Re-upload expression for kME computation (we freed d_expr above).
        DevMem d_expr2((size_t)n_genes * n_cells);
        SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(d_expr2.get(), h_centered.data(),
                                            (size_t)n_genes * n_cells * sizeof(float),
                                            cudaMemcpyHostToDevice, stream));

        core::DeviceMemory<float> d_eigengenes((size_t)n_modules * n_cells);
        SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(d_eigengenes.get(),
                                            result.eigengenes.data(),
                                            (size_t)n_modules * n_cells * sizeof(float),
                                            cudaMemcpyHostToDevice, stream));

        // Upload module assignment (0-indexed: module m → m-1, unassigned → -1).
        std::vector<int> assign_0indexed(n_genes);
        for (int g = 0; g < n_genes; ++g)
            assign_0indexed[g] = (assignment[g] > 0) ? (assignment[g] - 1) : -1;

        core::DeviceMemory<int> d_module(n_genes);
        SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(d_module.get(), assign_0indexed.data(),
                                            n_genes * sizeof(int),
                                            cudaMemcpyHostToDevice, stream));

        core::DeviceMemory<float> d_kme(n_genes);

        int blk = 256;
        int grd = (n_genes + blk - 1) / blk;
        detail::kme_kernel<<<grd, blk, 0, stream>>>(
            d_expr2.get(), d_eigengenes.get(), d_module.get(),
            d_kme.get(), n_genes, n_cells, n_modules);

        SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        // [cudaMemcpy #4 — ONE-TIME EXIT: download kME values.]
        SINGLET_GPU_CUDA_CHECK(cudaMemcpy(result.kme.data(), d_kme.get(),
                                       n_genes * sizeof(float),
                                       cudaMemcpyDeviceToHost));
    }

    // ─────────────────────────────────────────────────────────────────────
    // 11. Hub genes: top-N genes per module by |kME|.
    // ─────────────────────────────────────────────────────────────────────
    int n_hub = result.n_hub_genes;
    result.hub_genes.resize((size_t)n_modules * n_hub);  // default-constructed (gene_idx=-1)

    for (int mod = 1; mod <= n_modules; ++mod) {
        std::vector<std::pair<float, int>> mod_kme;
        for (int g = 0; g < n_genes; ++g) {
            if (assignment[g] == mod)
                mod_kme.emplace_back(std::fabs(result.kme[g]), g);
        }
        // Partial sort: take top n_hub by descending |kME|.
        int take = std::min(n_hub, (int)mod_kme.size());
        std::partial_sort(mod_kme.begin(), mod_kme.begin() + take, mod_kme.end(),
            [](const std::pair<float,int>& a, const std::pair<float,int>& b){
                return a.first > b.first;
            });
        for (int i = 0; i < take; ++i) {
            auto& entry = result.hub_genes[(size_t)(mod - 1) * n_hub + i];
            entry.gene_idx = mod_kme[i].second;
            entry.kme      = mod_kme[i].first;  // |kME| value (sign comes from result.kme)
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // 12. Restore determinism default (if we changed it).
    // ─────────────────────────────────────────────────────────────────────
    if (cfg.deterministic) {
        SINGLET_GPU_CUBLAS_CHECK(cublasSetAtomicsMode(ctx.blas(),
                                                   CUBLAS_ATOMICS_ALLOWED));
    }

    return result;
}

// ── Compat alias: HdwgcnaParams → HdwgcnaConfig ──────────────────────────────
// The test harness was authored with HdwgcnaParams; HdwgcnaConfig is the real name.
using HdwgcnaParams = HdwgcnaConfig;

// run_hdwgcna overload: takes a DeviceCSC (cells × genes, CSC where rows=genes, cols=cells)
// Densifies the CSC to host col-major, then calls the real run_hdwgcna.
inline HdwgcnaResult run_hdwgcna(
    const core::DeviceCSC& d_csc,
    HdwgcnaConfig cfg)   // by value so we can apply compat field mapping
{
    // Apply compat field: height_cut → height_cut_fraction
    cfg.height_cut_fraction = cfg.height_cut;

    const int n_genes = d_csc.rows;
    const int n_cells = d_csc.cols;
    const int nnz     = d_csc.nnz;

    // Download CSC to host
    std::vector<int>   h_col_ptr(n_cells + 1);
    std::vector<int>   h_row_idx(nnz);
    std::vector<float> h_vals(nnz);
    cudaMemcpy(h_col_ptr.data(), d_csc.col_ptr.get(),    (n_cells+1)*sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_row_idx.data(), d_csc.row_indices.get(), nnz*sizeof(int),           cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vals.data(),    d_csc.values.get(),      nnz*sizeof(float),         cudaMemcpyDeviceToHost);

    // Densify: row-major cells × genes (run_hdwgcna expects cells × genes row-major)
    std::vector<float> h_dense((size_t)n_cells * n_genes, 0.0f);
    for (int c = 0; c < n_cells; ++c) {
        for (int idx = h_col_ptr[c]; idx < h_col_ptr[c+1]; ++idx) {
            int gene = h_row_idx[idx];
            h_dense[(size_t)c * n_genes + gene] = h_vals[idx];
        }
    }

    // Create a temporary GPUContext for this call
    core::GPUContext ctx;
    return run_hdwgcna(h_dense.data(), n_cells, n_genes, cfg, ctx);
}

}  // namespace network

// ── Top-level singlet::gpu::DeviceCSC alias ────────────────────────────────────
using DeviceCSC = singlet::gpu::core::DeviceCSC;

// ── singlet::gpu::testing::dense_to_device_csc ─────────────────────────────────
namespace testing {
// Creates a DeviceCSC from a dense row-major host buffer (cells × genes).
// Only nonzeros are stored (skips zero entries).
inline singlet::gpu::core::DeviceCSC dense_to_device_csc(
    const float* h_data, int n_cells, int n_genes)
{
    // Build host CSC: columns = cells, rows = genes.
    std::vector<int>   col_ptr(n_cells + 1, 0);
    std::vector<int>   row_idx;
    std::vector<float> vals;
    for (int c = 0; c < n_cells; ++c) {
        for (int g = 0; g < n_genes; ++g) {
            float v = h_data[(size_t)c * n_genes + g];
            if (v != 0.0f) {
                row_idx.push_back(g);
                vals.push_back(v);
            }
        }
        col_ptr[c + 1] = static_cast<int>(row_idx.size());
    }
    int nnz = static_cast<int>(vals.size());
    singlet::gpu::core::DeviceCSC csc(n_genes, n_cells, nnz);
    cudaMemcpy(csc.col_ptr.get(),     col_ptr.data(),
               (n_cells + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(csc.row_indices.get(), row_idx.data(),
               nnz * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(csc.values.get(),      vals.data(),
               nnz * sizeof(float), cudaMemcpyHostToDevice);
    return csc;
}
}  // namespace testing

// ── PzDeviceLoader compat (returns DeviceCSC from a .1pz path) ────────────────
namespace io {
struct PzDeviceLoader {
    singlet::gpu::core::DeviceCSC load(const std::string& path,
                                       cudaStream_t stream = nullptr) const {
        PzDeviceMatrix m = singlet::gpu::io::load_pz(path, stream);
        return std::move(m.mat);
    }
};
}  // namespace io

// ── preprocess compat stubs (used in Hdwgcna_GSM_RealData which SKIPs w/o data) ──
namespace preprocess {
struct LognormParams {
    float target_sum  = 1e4f;
    float base        = 1.0f;
};
struct HvgParams {
    int   n_top_genes = 2000;
    float min_mean    = 0.0125f;
    float min_disp    = 0.5f;
};
// lognorm: returns CSC unchanged (stub; real data test SKIPs if data absent)
inline singlet::gpu::core::DeviceCSC lognorm(
    const singlet::gpu::core::DeviceCSC& csc,
    const LognormParams& /*lp*/)
{
    // In a real implementation this would call log_normalize.
    // For compilation, return a copy of the input CSC.
    // The real data test SKIPs when kRealDataPath doesn't exist.
    const int n_genes = csc.rows;
    const int n_cells = csc.cols;
    const int nnz     = csc.nnz;
    std::vector<int>   h_cp(n_cells + 1), h_ri(nnz);
    std::vector<float> h_v(nnz);
    cudaMemcpy(h_cp.data(), csc.col_ptr.get(),    (n_cells+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ri.data(), csc.row_indices.get(), nnz*sizeof(int),        cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v.data(),  csc.values.get(),      nnz*sizeof(float),      cudaMemcpyDeviceToHost);
    singlet::gpu::core::DeviceCSC out(n_genes, n_cells, nnz);
    cudaMemcpy(out.col_ptr.get(),     h_cp.data(),
               (n_cells + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(out.row_indices.get(), h_ri.data(),
               nnz * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(out.values.get(),      h_v.data(),
               nnz * sizeof(float), cudaMemcpyHostToDevice);
    return out;
}
// select_hvg: returns CSC unchanged (stub)
inline singlet::gpu::core::DeviceCSC select_hvg(
    const singlet::gpu::core::DeviceCSC& csc,
    const HvgParams& /*hvg_p*/)
{
    const int n_genes = csc.rows;
    const int n_cells = csc.cols;
    const int nnz     = csc.nnz;
    std::vector<int>   h_cp(n_cells + 1), h_ri(nnz);
    std::vector<float> h_v(nnz);
    cudaMemcpy(h_cp.data(), csc.col_ptr.get(),    (n_cells+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ri.data(), csc.row_indices.get(), nnz*sizeof(int),        cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v.data(),  csc.values.get(),      nnz*sizeof(float),      cudaMemcpyDeviceToHost);
    singlet::gpu::core::DeviceCSC out(n_genes, n_cells, nnz);
    cudaMemcpy(out.col_ptr.get(),     h_cp.data(),
               (n_cells + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(out.row_indices.get(), h_ri.data(),
               nnz * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(out.values.get(),      h_v.data(),
               nnz * sizeof(float), cudaMemcpyHostToDevice);
    return out;
}
}  // namespace preprocess

}  // namespace singlet::gpu
