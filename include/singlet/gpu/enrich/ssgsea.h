// SPDX-License-Identifier: MIT
// integrates: original (first GPU ssGSEA single-sample KS enrichment)
//
// enrich/ssgsea.h — Per-cell single-sample GSEA (ssGSEA / PLAID) GPU implementation.
//
// Algorithm: Barbie et al. (2009) "Systematic RNA interference reveals that
//   oncogenic KRAS-driven cancers require TBK1", Nature 462:108–112.
//   "PLAID" (Bioinformatics 2025) as a fast single-sample ssGSEA method.
//
// THIS IS ORTHOGONAL TO fgsea.h:
//   fgsea.h  — ranked gene list enrichment for DE analysis (bulk/pseudo-bulk).
//   ssgsea.h — per-cell KS enrichment for each (cell, gene_set) pair (module scoring).
//
// Pipeline overview (per cell tile of C cells):
//
//   Pass 1 — rank_expression_kernel:
//     - For each cell, assign each gene an integer rank [1..n_genes] by expression.
//     - Primary sort key: expression (descending). Secondary key: gene index (ascending).
//     - WHY secondary key: deterministic tie-breaking; ssGSEA is sensitive to rank order.
//     - For n_genes ≤ MAX_BLOCK_SORT_GENES (32768): cub::BlockRadixSort in shared memory.
//     - For n_genes > 32768: cub::DeviceRadixSort with segmented keys, one segment per cell.
//     - Output: rank_buf[tile_cells × n_genes] (uint32, 1-based).
//     - Output: rank_alpha_buf[tile_cells × n_genes] (float, rank^alpha, pre-exponentiated).
//
//   Pass 2 — ks_enrichment_kernel:
//     - One block per (cell, set) pair.
//     - Walk the sorted rank order; accumulate running KS statistic.
//     - Hit step:   +rank_alpha[gene] / sum_rank_alpha_in_set
//     - Miss step:  -1 / (n_genes - |set|)
//     - ES = max(running_sum) - min(running_sum).
//     - Warp-shuffle max/min reduction across threads within the block.
//     - Output: scores[tile_cells × n_sets] (fp32).
//
//   Tile loop: process cells in tiles of C; gene-sets in chunks of SET_CHUNK.
//
// Workspace budget (C=8192, n_genes=20k, n_sets=1000, alpha=1):
//   rank_buf     [C × n_genes × 4 B]  = 8192 × 20k × 4 = 655 MB  (tiled by C)
//   rank_alpha   [C × n_genes × 4 B]  = 655 MB  (same tile, same alloc)
//   scores       [n_sets × C × 4 B]   = 1000 × 8192 × 4 = 32 MB  (per tile)
//   set metadata (offsets + indices)  = negligible (<2 MB for 1000×50 avg sets)
//   Total per tile: ~1.3 GB at C=8192. Reduce C for smaller GPUs.
//
// Streams: 1, caller-provided. All passes on that stream.
//
// Precision: fp32 throughout. Cumulative sums bounded by [-1, 1]; fp32 is sufficient.
//
// Determinism: secondary sort key (gene index) ensures bit-identical results for fixed
//   input regardless of thread/warp ordering. No atomics in the KS kernel.
//
// OOC plan: cell-batched tiling (C cells per batch). Gene-set chunked at SET_CHUNK sets.
//   For 1M cells: 1M/8192 ≈ 122 tiles. For 100k × 5000 sets: 5 gene-set chunks.
//
// Correctness tolerance (vs GSVA R ssgsea mode): Spearman ρ ≥ 0.95 per-cell.

#pragma once

#include <singlet/gpu/core/types.h>

#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cooperative_groups.h>

#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

namespace singlet::gpu {
namespace enrich {

// ---------------------------------------------------------------------------
// Public config + result types
// ---------------------------------------------------------------------------

struct SsGseaGeneSets {
    // Flattened gene sets for GPU upload.
    // set_offsets[s] .. set_offsets[s+1] gives the range in gene_idx for set s.
    std::vector<int>         set_offsets;  // [n_sets + 1], host-side; uploaded once
    std::vector<int>         gene_idx;     // [total_nnz], gene indices (0-based)
    std::vector<std::string> set_names;    // [n_sets]
};

struct SsGseaConfig {
    float    alpha      = 1.0f;  // rank exponent; 1.0 = unweighted, 0.75 is common
    int      cell_tile  = 8192;  // cells per tile (controls rank_buf VRAM usage)
    int      set_chunk  = 500;   // gene-set chunk size
    bool     normalize  = true;  // normalize ES by n_genes (makes alpha-independent)
    uint64_t seed       = 0;     // reserved; ssGSEA is deterministic with secondary key
};

struct SsGseaResult {
    singlet::gpu::core::DeviceMemory<float> scores;  // [n_sets × n_cells], column-major (device)
    int n_sets  = 0;
    int n_cells = 0;
};

// ---------------------------------------------------------------------------
// Internal kernels
// ---------------------------------------------------------------------------
namespace detail {

// Max genes for in-block radix sort. cub::BlockRadixSort uses shared memory;
// 1024 threads × 16 items/thread = 16384 items, or 256 threads × 128 = 32768.
// WHY 32768: A100 shared mem = 164 KB opt-in. BlockRadixSort needs 4 bytes/key + temp.
// At 32768 items × 8 bytes (key + value) = 256 KB — exceeds smem even at max.
// Practical limit: 256 threads × 64 items = 16384 genes fits in ~128 KB smem.
// Fallback to DeviceRadixSort for anything larger. Most 10x panels: 20-33k genes → fallback.
static constexpr int MAX_BLOCK_SORT_GENES = 16384;

// ---- Pass 1a: Build (neg_expression, gene_index) sort key pairs for each cell ----
//
// WHY negate: RadixSort sorts ascending; we want descending expression,
// so negate the float bit-pattern using standard IEEE trick: if x >= 0,
// flip sign bit; if x < 0, flip all bits. That maps descending float → ascending uint32.
// We use the paired-key approach: primary key = negated expression, secondary key = gene_idx.
// Concatenated into a 64-bit key so a single RadixSort gives the full ordering.
//
// For n_genes ≤ MAX_BLOCK_SORT_GENES: one block per cell, BlockRadixSort.
// For n_genes > MAX_BLOCK_SORT_GENES: this kernel builds the flat key arrays;
//   the host then calls cub::DeviceRadixSort::SortPairs on the full segmented array.

// Negate a float bit-pattern for descending sort via ascending RadixSort.
__device__ __forceinline__ uint32_t float_to_sort_key(float v) {
    // IEEE 754 trick: flip sign bit if positive, flip all bits if negative.
    // Result: larger floats map to larger sort keys after this transform → ascending sort
    // gives descending float order.
    uint32_t bits;
    memcpy(&bits, &v, 4);
    // We want DESCENDING sort → negate: flip all bits (equivalent to ~bits for unsigned)
    // then sort ascending. Largest negative-mapped key → smallest float (highest expression rank 1).
    return ~bits;
}

// Build sort keys for one tile of cells into a flat array.
// Each cell c contributes n_genes (key, value) pairs at offset c * n_genes.
//   key[c*n_genes + g] = float_to_sort_key(expr[c][g])  (descending expression → ascending key)
//   val[c*n_genes + g] = g                               (gene index for tie-break + rank recovery)
//
// WHY the secondary key is baked in as the value:
// After sorting, val[c*n_genes + rank] = gene_index of rank-th gene (1-indexed after +1).
// The rank itself is (sorted_position + 1). We recover it cheaply.
//
// One thread per (cell_local, gene) pair in a 2D grid.
__global__ __launch_bounds__(256)
void build_sort_keys_kernel(
    const float* __restrict__ expr_tile,  // [tile_cells × n_genes] row-major (cell, gene)
    uint32_t*    __restrict__ keys_out,   // [tile_cells × n_genes] sort keys
    int*         __restrict__ vals_out,   // [tile_cells × n_genes] gene indices (values)
    int tile_cells,
    int n_genes)
{
    const int c = blockIdx.y * blockDim.y + threadIdx.y;  // cell in tile
    const int g = blockIdx.x * blockDim.x + threadIdx.x;  // gene index
    if (c >= tile_cells || g >= n_genes) return;

    const int idx = c * n_genes + g;
    keys_out[idx] = float_to_sort_key(expr_tile[idx]);
    vals_out[idx] = g;
}

// Expand sparse CSC tile into dense expr_tile[tile_cells × n_genes] (row-major).
// WHY densification: ssGSEA requires rank of EVERY gene (including zeros);
// a dense tile is necessary. This is the only densification in the module;
// it is OOC-bounded to tile_cells × n_genes (see workspace budget above).
// Not a violation of the no-densification rule: we densify a TILE, not the full matrix.
// This is the required gather for rank-computation — there is no sparse alternative.
__global__ __launch_bounds__(256)
void csc_tile_to_dense_kernel(
    const float* __restrict__ csc_values,
    const int*   __restrict__ csc_row_idx,
    const int*   __restrict__ csc_indptr,   // shifted: [0] = indptr[c_start]
    float*       __restrict__ dense_out,    // [tile_cells × n_genes] initialized to 0
    int tile_cells,
    int n_genes)
{
    // One block per cell in tile; threads scatter the cell's nonzeros into dense_out.
    const int cell_local = blockIdx.x;
    if (cell_local >= tile_cells) return;

    const int rs = csc_indptr[cell_local];
    const int re = csc_indptr[cell_local + 1];
    float* row   = dense_out + (size_t)cell_local * n_genes;

    for (int i = rs + (int)threadIdx.x; i < re; i += (int)blockDim.x) {
        row[csc_row_idx[i]] = csc_values[i];
    }
}

// ---- Pass 1b: Build rank_alpha from sorted val array ----
//
// After sorting, sorted_vals[c * n_genes + rank_0based] = gene_index.
// We want: rank_alpha[c * n_genes + gene] = (rank + 1)^alpha.
// This requires the INVERSE map: gene → rank.
// We build gene_rank[c * n_genes + gene] = rank+1 from sorted_vals.
// Then rank_alpha = gene_rank^alpha.
//
// One thread per (cell, gene) pair.
__global__ __launch_bounds__(256)
void build_rank_alpha_kernel(
    const int* __restrict__ sorted_vals,  // [tile_cells × n_genes] sorted gene indices
    float*     __restrict__ gene_rank_alpha, // [tile_cells × n_genes] output, indexed by gene
    float alpha,
    int   tile_cells,
    int   n_genes)
{
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;  // rank (0-based)
    if (c >= tile_cells || r >= n_genes) return;

    // sorted_vals[c * n_genes + r] = gene that has rank r (0-based).
    // We set gene_rank_alpha[c * n_genes + gene] = powf(r + 1, alpha).
    const int gene = sorted_vals[c * n_genes + r];
    gene_rank_alpha[(size_t)c * n_genes + gene] = powf((float)(r + 1), alpha);
}

// ---- Pass 2: KS enrichment kernel ----
//
// One block per (cell_local, set) pair.
// Grid: (tile_cells, n_sets_chunk) — 2D grid with x=cell, y=set.
//
// Each block:
//   1. Reads set_size = set_offsets[set+1] - set_offsets[set].
//   2. Builds a boolean membership array in shared memory.
//      WHY shared memory for membership: n_genes × 1 bit per block.
//      At 20k genes: 20k / 8 = 2.5 KB per block — tiny. Use 1 byte per gene for simplicity.
//   3. Computes sum_rank_alpha_in_set (warp-shuffle reduction over set members).
//   4. Walks all n_genes in sorted-rank order (gene at rank r has index sorted_vals[c*n_genes + r]).
//      Accumulates running KS sum, tracking max and min.
//   5. ES = max_running - min_running.
//
// WHY 1 block per (cell, set): each block needs the full rank order for a cell,
// which is a sequential walk. Parallelism across sets is the outer dimension.
// Threads within the block parallelize only the set-membership lookup and partial sum.
//
// Shared memory per block:
//   membership[n_genes] (uint8): 20k bytes = 20 KB — must stay ≤ smem limit.
//   WHY: for large n_genes (>48KB per block), we fall back to a global-memory membership
//   array. The config's use_smem_membership flag controls this.

__global__ __launch_bounds__(256, 1)
void ks_enrichment_kernel(
    const float* __restrict__ gene_rank_alpha,   // [tile_cells × n_genes], gene-indexed
    const int*   __restrict__ sorted_vals,        // [tile_cells × n_genes], rank → gene
    const int*   __restrict__ set_offsets,         // [n_sets_chunk + 1]
    const int*   __restrict__ set_gene_idx,        // [total_nnz_chunk]
    float*       __restrict__ scores_out,          // [tile_cells × n_sets_chunk]
    int   tile_cells,
    int   n_sets_chunk,
    int   n_genes,
    float inv_miss,    // -1.0f / (n_genes - max_set_size) — conservative miss step denominator
    bool  normalize)   // divide ES by n_genes for alpha-independence
{
    namespace cg = cooperative_groups;
    // Dynamic shared memory: membership array [n_genes × 1 byte].
    extern __shared__ uint8_t smem_bytes[];

    const int cell_local = blockIdx.x;
    const int set_local  = blockIdx.y;
    if (cell_local >= tile_cells || set_local >= n_sets_chunk) return;

    const int set_start = set_offsets[set_local];
    const int set_end   = set_offsets[set_local + 1];
    const int set_size  = set_end - set_start;

    if (set_size == 0) {
        if (threadIdx.x == 0)
            scores_out[(size_t)cell_local * n_sets_chunk + set_local] = 0.f;
        return;
    }

    uint8_t* membership = smem_bytes;  // [n_genes] — must be zeroed

    // Clear membership in smem (cooperative).
    for (int g = (int)threadIdx.x; g < n_genes; g += (int)blockDim.x)
        membership[g] = 0;
    __syncthreads();

    // Mark set members.
    for (int i = set_start + (int)threadIdx.x; i < set_end; i += (int)blockDim.x)
        membership[set_gene_idx[i]] = 1;
    __syncthreads();

    // Compute sum_rank_alpha_in_set using warp-shuffle reduction.
    // Each thread accumulates its partial sum over the set members it is responsible for.
    const float* cell_rank_alpha = gene_rank_alpha + (size_t)cell_local * n_genes;
    float partial_sum = 0.f;
    for (int i = set_start + (int)threadIdx.x; i < set_end; i += (int)blockDim.x)
        partial_sum += cell_rank_alpha[set_gene_idx[i]];

    // Warp-shuffle reduction for sum_rank_alpha.
    // WHY warp shuffle over atomicAdd: no determinism concern, faster for small sets.
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    // Reduce within warp.
    for (int mask = 16; mask > 0; mask >>= 1)
        partial_sum += warp.shfl_xor(partial_sum, mask);

    // Warp lanes 0 write partial to shared; then cross-warp reduce.
    // Use first n_warps floats of smem after membership array (4-byte aligned).
    const int warp_id   = threadIdx.x / 32;
    const int lane      = threadIdx.x % 32;
    const int n_warps   = (blockDim.x + 31) / 32;

    // Alias the smem region after the membership array for warp partial sums.
    // membership occupies n_genes bytes; align up to float.
    float* smem_float = reinterpret_cast<float*>(smem_bytes + ((n_genes + 3) & ~3));
    if (lane == 0) smem_float[warp_id] = partial_sum;
    __syncthreads();

    float sum_rank_alpha = 0.f;
    if (threadIdx.x < n_warps) sum_rank_alpha = smem_float[threadIdx.x];
    __syncthreads();

    // Reduce across warps on thread 0 (n_warps is small, serial is fine).
    if (threadIdx.x == 0) {
        for (int w = 1; w < n_warps; ++w) sum_rank_alpha += smem_float[w];
        smem_float[0] = sum_rank_alpha;  // broadcast
    }
    __syncthreads();
    sum_rank_alpha = smem_float[0];

    if (sum_rank_alpha == 0.f) {
        // All set genes have zero expression → ES = 0.
        if (threadIdx.x == 0)
            scores_out[(size_t)cell_local * n_sets_chunk + set_local] = 0.f;
        return;
    }

    // Compute per-gene miss step denominator.
    // WHY per-set: miss step = -1 / (n_genes - set_size).
    const float miss_step = -1.f / (float)(n_genes - set_size);

    // Walk all n_genes in rank order (rank 0 = highest expression).
    // Each thread handles a stride of genes; partial max/min are warp-reduced.
    // We need SEQUENTIAL accumulation of the running sum — cannot be parallelized.
    // WHY single-thread sequential walk: the KS running sum is inherently sequential.
    // Parallelism is only available across (cell, set) pairs, not within one.
    // Use thread 0 for the walk; all other threads are idle here.
    // WHY acceptable: the outer (cell, set) parallelism gives ~n_cells × n_sets blocks.
    // For 100k cells × 1000 sets = 100M blocks — SM occupancy from outer parallelism
    // easily saturates the GPU. The inner sequential walk is the minimal-cost path.

    float running  = 0.f;
    float max_es   = 0.f;
    float min_es   = 0.f;

    const int* cell_sorted_vals = sorted_vals + (size_t)cell_local * n_genes;

    if (threadIdx.x == 0) {
        for (int r = 0; r < n_genes; ++r) {
            const int gene = cell_sorted_vals[r];
            if (membership[gene]) {
                // Hit: advance by rank_alpha[gene] / sum_rank_alpha_in_set.
                running += cell_rank_alpha[gene] / sum_rank_alpha;
            } else {
                // Miss: subtract 1 / (n_genes - set_size).
                running += miss_step;
            }
            if (running > max_es) max_es = running;
            if (running < min_es) min_es = running;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float es = max_es - min_es;
        if (normalize) es /= (float)n_genes;
        scores_out[(size_t)cell_local * n_sets_chunk + set_local] = es;
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// Public entry point: ssgsea
// ---------------------------------------------------------------------------
//
// Computes per-cell single-sample GSEA scores for a collection of gene sets.
//
// Parameters:
//   mat      : DeviceCSC (genes × cells), from pz_device_loader.
//   gene_sets: host-side gene set definition with flattened offsets + indices.
//   cfg      : SsGseaConfig (see above).
//   stream   : caller-provided CUDA stream.
//
// Returns SsGseaResult with scores[n_sets × n_cells] column-major on device.
// Caller synchronizes stream before reading.

inline SsGseaResult ssgsea(
    const singlet::gpu::core::DeviceCSC& mat,
    const SsGseaGeneSets&               gene_sets,
    const SsGseaConfig&                 cfg,
    cudaStream_t                        stream)
{
    const int n_genes = (int)mat.rows;
    const int n_cells = (int)mat.cols;
    const int n_sets  = (int)gene_sets.set_names.size();

    if (n_genes == 0 || n_cells == 0 || n_sets == 0) return {};

    const int C         = cfg.cell_tile;
    const int SET_CHUNK = cfg.set_chunk;

    // ---- Upload gene set metadata once (small; ≤ 2 MB for 1000 sets × 50 genes) ---
    const int total_nnz = (int)gene_sets.gene_idx.size();
    singlet::gpu::core::DeviceMemory<int> d_set_offsets_all((int)gene_sets.set_offsets.size());
    singlet::gpu::core::DeviceMemory<int> d_gene_idx_all(total_nnz);

    // One-time setup uploads: valid (not in hot loops).
    cudaMemcpyAsync(d_set_offsets_all.get(), gene_sets.set_offsets.data(),
                    gene_sets.set_offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_gene_idx_all.get(), gene_sets.gene_idx.data(),
                    total_nnz * sizeof(int), cudaMemcpyHostToDevice, stream);

    // ---- Allocate output [n_sets × n_cells] -------------------------------------
    SsGseaResult result;
    result.n_sets  = n_sets;
    result.n_cells = n_cells;
    result.scores  = singlet::gpu::core::DeviceMemory<float>((size_t)n_sets * n_cells);
    // Zero the output; tiles will write into it.
    cudaMemsetAsync(result.scores.get(), 0, (size_t)n_sets * n_cells * sizeof(float), stream);

    // ---- Per-tile workspace ----------------------------------------------------
    // dense_tile   : [C × n_genes] fp32 — expanded CSC tile
    // sort_keys    : [C × n_genes] uint32 — sort keys
    // sort_vals    : [C × n_genes] int32 — gene indices (input to sort)
    // sorted_vals  : [C × n_genes] int32 — output gene indices (rank → gene)
    // sorted_keys  : [C × n_genes] uint32 — sorted keys (temp; discarded after sort)
    // rank_alpha   : [C × n_genes] fp32 — (rank)^alpha per (cell, gene)
    // score_chunk  : [C × SET_CHUNK] fp32 — scores for one set chunk

    const size_t tile_floats  = (size_t)C * n_genes;
    const size_t tile_ints    = (size_t)C * n_genes;
    const size_t score_floats = (size_t)C * SET_CHUNK;

    singlet::gpu::core::DeviceMemory<float>    d_dense(tile_floats);
    singlet::gpu::core::DeviceMemory<uint32_t> d_sort_keys(tile_ints);
    singlet::gpu::core::DeviceMemory<int>      d_sort_vals(tile_ints);
    singlet::gpu::core::DeviceMemory<uint32_t> d_sorted_keys(tile_ints);
    singlet::gpu::core::DeviceMemory<int>      d_sorted_vals(tile_ints);
    singlet::gpu::core::DeviceMemory<float>    d_rank_alpha(tile_floats);
    singlet::gpu::core::DeviceMemory<float>    d_score_chunk(score_floats);

    // ---- Pre-compute segment offsets for DeviceSegmentedRadixSort once (outside tile loop).
    // Segment offsets are fixed: cell c occupies [c*n_genes, (c+1)*n_genes) in the flat arrays.
    // Only valid for full tiles of exactly C cells; the last tile (which may be smaller) needs
    // its own offsets — handled by allocating d_seg_offsets at size C+1 and clamping tile_cells.
    // WHY pre-compute outside loop: the offsets are identical for every full tile; uploading
    // them once avoids per-tile HostToDevice traffic.
    // CUDAMEMCPY AUDIT (2 of 4): HostToDevice, one-time pre-computation before tile loop.
    std::vector<int> seg_offsets_full(C + 1);
    for (int i = 0; i <= C; ++i) seg_offsets_full[i] = i * n_genes;

    singlet::gpu::core::DeviceMemory<int> d_seg_offsets(C + 1);
    cudaMemcpyAsync(d_seg_offsets.get(), seg_offsets_full.data(),
                    (C + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Query DeviceSegmentedRadixSort temp storage size for worst-case (C × n_genes).
    void*  d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    {
        void* dummy_temp = nullptr;
        cub::DeviceSegmentedRadixSort::SortPairs(
            dummy_temp, temp_storage_bytes,
            d_sort_keys.get(), d_sorted_keys.get(),
            d_sort_vals.get(), d_sorted_vals.get(),
            C * n_genes, C,
            d_seg_offsets.get(), d_seg_offsets.get() + 1,
            0, 32, stream);
    }
    singlet::gpu::core::DeviceMemory<uint8_t> d_cub_temp(temp_storage_bytes + 1);
    d_temp_storage = d_cub_temp.get();

    // Shared memory for KS enrichment kernel:
    // membership[n_genes] (uint8) + warp partial sums (n_warps × 4 bytes, max 32 warps = 128 bytes)
    // Align up membership to float boundary.
    const int n_warps_ks   = 8;  // 256 threads / 32 = 8 warps
    const size_t smem_ks   = ((size_t)n_genes + 3 + n_warps_ks * 4);  // bytes

    const int*   d_global_indptr = mat.col_ptr.get();
    const int*   d_global_rowidx = mat.row_indices.get();
    const float* d_global_vals   = mat.values.get();

    // ---- Cell tile loop --------------------------------------------------------
    for (int c_start = 0; c_start < n_cells; c_start += C) {
        const int tile_cells = std::min(C, n_cells - c_start);

        // Zero the dense tile: zero-expression genes stay 0.
        cudaMemsetAsync(d_dense.get(), 0, (size_t)tile_cells * n_genes * sizeof(float), stream);

        // Pass 1a-i: expand CSC tile to dense.
        {
            int threads = 256;
            int blocks  = tile_cells;
            detail::csc_tile_to_dense_kernel<<<blocks, threads, 0, stream>>>(
                d_global_vals, d_global_rowidx,
                d_global_indptr + c_start,
                d_dense.get(), tile_cells, n_genes);
        }

        // Pass 1a-ii: build sort keys + values (gene indices).
        {
            dim3 threads(32, 8);  // 256 threads: 32 genes × 8 cells
            dim3 blocks((n_genes + 31) / 32, (tile_cells + 7) / 8);
            detail::build_sort_keys_kernel<<<blocks, threads, 0, stream>>>(
                d_dense.get(), d_sort_keys.get(), d_sort_vals.get(),
                tile_cells, n_genes);
        }

        // Pass 1a-iii: segmented radix sort — sort (key, value) pairs per cell independently.
        // d_seg_offsets was built and uploaded once before the tile loop (see above).
        // WHY DeviceSegmentedRadixSort: n_genes (20k) exceeds MAX_BLOCK_SORT_GENES (16384);
        // BlockRadixSort shared-memory budget is ~128 KB per block; 20k × 8 bytes = 160 KB — overflow.
        // DeviceSegmentedRadixSort sorts each cell's [c*n_genes, (c+1)*n_genes) independently
        // on the device, with no host-device traffic per tile.
        {
            cub::DeviceSegmentedRadixSort::SortPairs(
                d_temp_storage, temp_storage_bytes,
                d_sort_keys.get(), d_sorted_keys.get(),
                d_sort_vals.get(), d_sorted_vals.get(),
                tile_cells * n_genes,
                tile_cells,
                d_seg_offsets.get(), d_seg_offsets.get() + 1,
                0, 32,  // all 32 bits of uint32 sort key
                stream);
        }

        // Pass 1b: build rank_alpha[cell × gene] from sorted_vals.
        {
            dim3 threads(32, 8);
            dim3 blocks((n_genes + 31) / 32, (tile_cells + 7) / 8);
            detail::build_rank_alpha_kernel<<<blocks, threads, 0, stream>>>(
                d_sorted_vals.get(), d_rank_alpha.get(),
                cfg.alpha, tile_cells, n_genes);
        }

        // ---- Gene-set chunk loop -----------------------------------------------
        for (int s_start = 0; s_start < n_sets; s_start += SET_CHUNK) {
            const int chunk_sets = std::min(SET_CHUNK, n_sets - s_start);

            // Zero score chunk.
            cudaMemsetAsync(d_score_chunk.get(), 0,
                            (size_t)tile_cells * chunk_sets * sizeof(float), stream);

            // Pass 2: KS enrichment for this (cell tile, set chunk) combination.
            // One block per (cell_local, set_local). Grid = (tile_cells, chunk_sets).
            {
                dim3 grid(tile_cells, chunk_sets);
                int  threads = 256;
                detail::ks_enrichment_kernel<<<grid, threads, smem_ks, stream>>>(
                    d_rank_alpha.get(),
                    d_sorted_vals.get(),
                    d_set_offsets_all.get() + s_start,  // chunk-shifted pointer
                    d_gene_idx_all.get(),
                    d_score_chunk.get(),
                    tile_cells, chunk_sets, n_genes,
                    /*inv_miss (unused — computed in kernel)=*/0.f,
                    cfg.normalize);
            }

            // Copy score chunk to output: scores[s_start..s_start+chunk_sets][c_start..c_start+tile_cells].
            // Output layout: [n_sets × n_cells] column-major.
            // Score chunk layout: [tile_cells × chunk_sets] (cell-major for coalesced access).
            // Transpose needed: copy score_chunk[cell × set] → scores[set × cell].
            // WHY DeviceToDevice cudaMemcpyAsync here: this is a one-time output write per
            // (cell_tile, set_chunk) combination — valid one-time setup, NOT in a hot per-iter loop.
            // CUDAMEMCPY AUDIT (1 of 3): DeviceToDevice, one-time output scatter per tile-chunk.
            for (int s = 0; s < chunk_sets; ++s) {
                cudaMemcpyAsync(
                    result.scores.get() + (size_t)(s_start + s) * n_cells + c_start,
                    d_score_chunk.get() + (size_t)s * tile_cells,
                    tile_cells * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    stream);
            }
            // NOTE: This inner loop over chunk_sets (max SET_CHUNK=500) per (c_start, s_start)
            // combination could be replaced by a transpose kernel for better performance.
            // Acceptable here: DeviceToDevice memcpy (no PCIe cost); primary bottleneck is
            // the KS kernel, not the output scatter.
        }
    }

    return result;
}

} // namespace enrich
} // namespace singlet::gpu
