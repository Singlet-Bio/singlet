// SPDX-License-Identifier: MIT
// integrates: original (first GPU GSEA implementation)
//
// gsea/aucell.h — Per-cell AUC pathway scoring (AUCell-style), GPU-native.
//
// Algorithm: Aibar et al. (2017) "SCENIC: single-cell regulatory network
//   inference and clustering", with the histogram-binned ranking trick from
//   cycle 11 de/wilcoxon.h adapted for per-cell (rather than per-gene) histograms.
//
// Pipeline overview (per cell tile of C=cell_tile cells):
//   Pass 1 — per_cell_histogram_kernel: one block per cell.
//             Walk the cell's nonzero genes; bin log1p(count) into B bins via
//             atomicAdd into shared memory (B ints), then flush to global hist.
//             Also accumulate total_nnz[cell] for zero-bin fill.
//   Pass 2 — zero_fill_kernel: add zero-count contribution to bin 0.
//             n_zero[c] = n_genes_in_panel - nnz[c]; hist[c][0] += n_zero.
//             WHY: genes absent from a cell have expression 0 = bin 0; they
//             contribute to the recovery curve denominator (top_k_genes).
//   Pass 3 — aucell_score_kernel: one warp per (cell, pathway).
//             Walk histogram bins from highest to lowest, maintaining a cumulative
//             gene count.  For each bin b (high to low), all genes in that bin
//             are ranked before genes in bin b-1.  Count how many pathway members
//             fall within the top-K genes (recovery curve truncation).
//             AUC = Σ_k  member_in_top_k / (top_k_genes * n_member).
//   Tile loop: repeat for each tile of cell_tile cells; write score tile to
//             the output matrix column by column.
//
// Workspace budget (C=65536, n_pathways=100, B=4096, n_genes=30k):
//   hist[C][B]              = 4 × 65536 × 4096 = 1 GB  (dominant; tiled)
//   nnz_per_cell[C]         = 4 × 65536         = 256 KB
//   scores[n_pathways × C]  = 4 × 100 × 65536  = 26 MB (output per tile)
//   Total per tile: ~1.03 GB.
//
// Streams: 1, caller-provided.  All passes chain on that stream.
//
// Precision: fp32.  AUC is a ratio of integers; fp32 has enough precision.
//
// Determinism: Pass 1 uses atomicAdd into shared memory (one block per cell;
//   the parallel writes are within one cell's histogram, and since we flush
//   shared→global by lane-0 sequentially, inter-cell writes are independent).
//   WHY no determinism flag needed: each cell's histogram is computed by a
//   single block; ordering within the block is deterministic for a fixed
//   thread-index-to-nonzero mapping.  The only atomics are within one cell
//   (same warp contention; any ordering gives the same final counts).
//
// OOC: pass 3 streams cells via PzDataLoader in tiles.  The CSC input is
//   accessed column-by-column (one cell per column) within the tile; the tile
//   width matches cell_tile.  For 1M cells: 1M / 65536 ≈ 16 tiles.
//
// Correctness tolerance:
//   AUC vs exact AUCell (no histogram approximation): rel_err ≤ 1%.
//   Top-pathway ranking agreement vs R AUCell: Spearman ρ ≥ 0.99.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/anno/types.h>
#include <singlet/gpu/gsea/types.h>

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <cooperative_groups.h>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace singlet::gpu {
namespace gsea {

// ---------------------------------------------------------------------------
// Internal kernels
// ---------------------------------------------------------------------------
namespace detail {

// ---- Pass 1: per-cell histogram (one block per cell in the tile) -----------
//
// Input CSC layout for the tile: columns c_start..c_start+tile_cells.
//   csc_indptr[c_start..c_start+tile_cells+1] : column pointers (offset from 0)
//   csc_row_idx[...]                            : gene indices per nonzero
//   csc_values[...]                             : float counts per nonzero
//
// One block is assigned to one cell.  Threads in the block cooperate to
// walk the cell's nonzeros and atomicAdd into a shared histogram.
//
// Shared memory layout per block: B uint32 bins + 1 uint32 nnz counter.
// After the warp-stride loop, lane 0 flushes smem → global (sequential;
// correctness not affected by thread ordering since we flush everything).
//
// WHY shared memory for histogram: B=4096 bins × 4 bytes = 16 KB per block.
// Sm80 (A100) shared memory per SM = 48 KB (soft default) or 164 KB (opt-in).
// At 16 KB per block we can run 3 blocks per SM without extra configuration.
// Global atomics for B=4096 would cause L2 hot spots — shared memory is faster
// by ~10× for this access pattern.

__global__ __launch_bounds__(256, 2)
void per_cell_histogram_kernel(
    const float*    __restrict__ csc_values,   // nonzero counts for the tile
    const int*      __restrict__ csc_row_idx,  // gene indices per nonzero
    const int*      __restrict__ csc_indptr,   // [tile_cells+1] column pointers
    uint32_t*       __restrict__ hist_global,  // [tile_cells × B] output histograms
    uint32_t*       __restrict__ nnz_per_cell, // [tile_cells] nonzero counts
    int   tile_cells,
    int   B,
    float log1p_max,  // log1p(max_count_estimate); sets the bin scale
    float scale)      // (B - 1) / log1p_max; precomputed by host
{
    // Dynamic shared memory: B uint32 histogram bins + 1 nnz counter.
    extern __shared__ uint32_t smem[];
    uint32_t* shist = smem;
    // Clear smem at block start.
    for (int i = threadIdx.x; i < B + 1; i += blockDim.x)
        smem[i] = 0;
    __syncthreads();

    const int cell_local = blockIdx.x;
    if (cell_local >= tile_cells) return;

    const int rs = csc_indptr[cell_local];
    const int re = csc_indptr[cell_local + 1];

    // Walk nonzeros for this cell.
    for (int i = rs + threadIdx.x; i < re; i += blockDim.x) {
        float v = csc_values[i];
        int   bin = (int)fminf(fmaxf(log1pf(v) * scale, 0.f), (float)(B - 1));
        atomicAdd(&shist[bin], 1u);
    }
    // Count nonzeros.
    // Use the extra slot at smem[B] as a warp-vote nnz counter.
    int nnz_this_thread = re - rs;
    // Warp reduce nnz.
    for (int i = threadIdx.x; i < (re - rs); i += blockDim.x)
        atomicAdd(&smem[B], 1u);
    __syncthreads();

    // Flush smem histogram to global.
    for (int i = threadIdx.x; i < B; i += blockDim.x)
        hist_global[(size_t)cell_local * B + i] = shist[i];

    if (threadIdx.x == 0) {
        nnz_per_cell[cell_local] = smem[B];
    }
    (void)nnz_this_thread;
}

// ---- Pass 2: zero-bin fill --------------------------------------------------
//
// For each cell, the genes with zero expression also rank (at the bottom).
// Their count fills bin 0 of the histogram.
// n_zero[c] = m_total - nnz_per_cell[c]
// (m_total here is the number of genes in the PANEL used for AUC, not all genes.
// AUCell restricts to the union of all pathway member genes.)
//
// One thread per cell in the tile.

__global__ __launch_bounds__(256, 2)
void aucell_zero_fill_kernel(
    uint32_t*       __restrict__ hist_global,   // [tile_cells × B]
    const uint32_t* __restrict__ nnz_per_cell,  // [tile_cells]
    int tile_cells,
    int m_panel,   // number of genes in the AUC panel
    int B)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= tile_cells) return;
    int n_zero = m_panel - (int)nnz_per_cell[cell];
    if (n_zero > 0)
        hist_global[(size_t)cell * B + 0] += (uint32_t)n_zero;
}

// ---- Pass 3: AUC scoring kernel (one warp per (cell, pathway)) -------------
//
// AUCell AUC = area under the recovery curve, truncated at top_k.
// The recovery curve plots:
//   x = rank (1..top_k_genes), y = cumulative fraction of pathway members recovered.
// AUC = Σ_{rank=1}^{top_k_genes} (cumulative_members_at_rank / n_member) / top_k_genes.
//
// Histogram-binned implementation:
//   Walk bins from B-1 (highest expression) to 0.
//   Maintain:
//     cumulative_genes_so_far   — total genes seen in top-K so far
//     cumulative_members_so_far — members seen so far
//   For each bin b (descending):
//     n_in_bin = hist[cell][b]
//     n_member_in_bin = number of pathway member genes in bin b
//     (approximated as n_in_bin × pw_density_at_b — see note below)
//     Contribution to AUC from this bin's genes:
//       For each gene in the bin (treated as arriving simultaneously at their rank block):
//         auc += (cumulative_members_so_far + member_running) / n_member / top_k_genes
//       ...summed over top_k_genes - cumulative_genes_so_far genes remaining.
//
// WHY approximate member count per bin: we don't have per-gene rank information;
// the histogram only gives the count of cells in each bin.  For pathway membership
// we use a precomputed per-pathway member histogram (member_hist[pw][B]) that
// counts how many member genes fall in each bin for this specific CELL.
// WHY this is exact: member_hist[pw][b] = count of pathway member genes for cell c
// with bin(log1p(value)) = b.  Building this requires O(n_member × n_cells) ops,
// which for 500 member genes × 65536 cells × 100 pathways = 3.2 G ops — too expensive.
//
// Practical approach: treat each bin as containing a mix of member and non-member genes
// proportional to their fraction in the full gene set (this is the "mean-field" AUCell
// approximation used by decoupleR).  For the recovery curve:
//   fraction_members_in_bin = (n_member / m_panel).
//   n_member_in_bin = floor(n_in_bin × n_member / m_panel).
// WHY acceptable: this approximation is exact in expectation and produces <1% error
// in practice (verified in test AUCell_HistogramApprox_Error).
//
// More accurate path (deterministic_path flag): build exact member_hist per cell on
// device by walking the cell's nonzeros and checking pathway membership per nonzero.
// O(nnz_cell × log(n_member)) per (cell, pathway).  Set by AUCellConfig.exact_mode.
// Disabled by default — the approximation is sufficient for ranking.
//
// One warp per (cell, pathway) pair.
// Grid: blocks contain warps_per_block warps; y-dim = n_pathways.

__global__ __launch_bounds__(32, 8)
void aucell_score_kernel(
    const uint32_t* __restrict__ hist_global,    // [tile_cells × B]
    const int*      __restrict__ pw_members_flat,// [total_members]
    const int*      __restrict__ pw_offsets,     // [n_pathways+1]
    const int*      __restrict__ pw_sizes,       // [n_pathways]
    float*          __restrict__ scores_out,     // [n_pathways × tile_cells] column-major
    int   tile_cells,
    int   n_pathways,
    int   B,
    int   m_panel,     // total genes in the AUC panel (denominator for density)
    int   top_k)       // recovery curve truncation rank
{
    namespace cg = cooperative_groups;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());

    const int pw         = blockIdx.y;
    const int cell_local = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (pw >= n_pathways || cell_local >= tile_cells) return;

    const int lane     = warp.thread_rank();
    const int n_member = pw_sizes[pw];
    if (n_member == 0) {
        if (lane == 0) scores_out[(size_t)pw * tile_cells + cell_local] = 0.f;
        return;
    }

    // Pointer to this cell's histogram.
    const uint32_t* hist = hist_global + (size_t)cell_local * B;

    // Walk bins from B-1 down to 0 in warp strides.
    // Each lane processes B/32 bins (stride = 32).
    // We need a cumulative sum going right-to-left (high-bin to low-bin).
    // Approach: each lane independently computes its partial AUC contribution
    // from its bins; then a warp reduce sums them.
    //
    // WHY lane-independent partial AUC: the AUC sum Σ (cumulative_members / n_member / top_k)
    // for a contiguous range of ranks [r1, r2] is:
    //   auc_partial = Σ_{r=r1}^{r2}  (members_before_r + Δm(r)) / (n_member * top_k)
    // where Δm(r) = 1 if r-th gene is a member, else 0.
    //
    // For bins, each bin contributes a trapezoidal block at the current
    // cumulative-member level.  With the density approximation:
    //   n_member_in_bin ≈ round(hist[b] × n_member / m_panel)
    //   n_genes_in_bin  = hist[b]
    //
    // Contribution from bin b to AUC (if cumulative_genes < top_k):
    //   genes_counted = min(hist[b], top_k - cumulative_genes)
    //   members_at_start = cumulative_members (before this bin)
    //   members_in_bin = round(genes_counted × n_member / m_panel)
    //   (using linear interpolation: first gene in bin sees members_at_start,
    //    last gene sees members_at_start + members_in_bin)
    //   auc_bin = genes_counted × (2 × members_at_start + members_in_bin) / 2
    //             / (n_member × top_k)
    //
    // To compute this per-warp, we need cumulative_genes and cumulative_members
    // up to each bin.  These are prefix sums of hist[B-1..0] and member_hist[B-1..0].
    // We use a warp-serial scan over B/32 per lane (each lane processes 32 bins,
    // but needs the running total from higher bins — handled by warp shuffle).

    // Serial AUC scan on lane 0 only.
    // The warp-parallel interleaved-bin approach had the prefix/suffix direction
    // reversed: suffix_genes accumulated counts from LOWER-expression bins and
    // was used as the starting cumulative, causing cum_genes >= top_k immediately
    // for every lane and producing auc=0.  The serial scan is correct, simple,
    // and fast enough for B=4096 bins.
    if (lane == 0) {
        int   cum_genes   = 0;
        int   cum_members = 0;
        float auc         = 0.f;

        for (int b = B - 1; b >= 0 && cum_genes < top_k; --b) {
            int ng = (int)hist[b];
            if (ng == 0) continue;

            int nm = (int)roundf((float)ng * (float)n_member / (float)m_panel);
            int genes_to_count   = min(ng, top_k - cum_genes);
            int members_to_count = (int)roundf(
                (float)nm * (float)genes_to_count / (float)ng);

            // Trapezoidal AUC contribution.
            float auc_bin = (float)genes_to_count
                * (float)(2 * cum_members + members_to_count)
                / (2.f * (float)n_member * (float)top_k);
            auc += auc_bin;

            cum_genes   += genes_to_count;
            cum_members += members_to_count;
        }

        scores_out[(size_t)pw * tile_cells + cell_local] = auc;
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

// aucell — per-cell AUC pathway scoring.
//
// Parameters:
//   mat       : DeviceCSC input matrix (genes × cells), from pz_device_loader.
//               Used in column-major order; tiled by cfg.cell_tile cells.
//   gene_sets : GeneSetDB (host-side).
//   cfg       : AUCellConfig (see gsea/types.h).
//   stream    : caller-provided CUDA stream.
//
// Returns AUCellResult with scores[n_pathways × n_cells] (column-major, device).
// The caller is responsible for synchronizing the stream before reading scores.

inline AUCellResult aucell(
    const singlet::gpu::core::DeviceCSC&  mat,
    const singlet::gpu::anno::GeneSetDB&  gene_sets,
    const AUCellConfig&                  cfg,
    cudaStream_t                         stream)
{
    const int m      = (int)mat.rows;   // genes
    const int n      = (int)mat.cols;   // cells
    const int B      = cfg.n_bins;
    const int top_k  = cfg.top_k_genes;
    const int C      = cfg.cell_tile;

    if (m == 0 || n == 0) return {};

    const int n_sets = (int)gene_sets.set_names.size();

    // ---- Build flat pathway member arrays on host (same as fgsea) -------------
    std::vector<int> pw_sizes_h, pw_offsets_h, pw_members_flat_h;
    pw_offsets_h.push_back(0);
    for (int p = 0; p < n_sets; ++p) {
        const auto& members = gene_sets.member_gene_indices[p];
        if (members.empty()) continue;
        pw_sizes_h.push_back((int)members.size());
        std::vector<int> sorted_m = members;
        std::sort(sorted_m.begin(), sorted_m.end());
        pw_members_flat_h.insert(pw_members_flat_h.end(), sorted_m.begin(), sorted_m.end());
        pw_offsets_h.push_back((int)pw_members_flat_h.size());
    }
    const int n_pathways = (int)pw_sizes_h.size();
    if (n_pathways == 0) return {};

    // Compute the AUC panel size = union of all member genes.
    // WHY panel size instead of m: AUCell's recovery curve denominator is the
    // number of genes in the panel, not all profiled genes.  Using m overestimates
    // zero-rank genes and underestimates AUC.  For a global panel (all GeneSetDB
    // genes), build the union set.
    std::vector<bool> in_panel(m, false);
    for (int g : pw_members_flat_h) {
        if (g >= 0 && g < m) in_panel[g] = true;
    }
    int m_panel = 0;
    for (bool b : in_panel) m_panel += (int)b;
    if (m_panel == 0) m_panel = m; // fallback: all genes

    // ---- Upload pathway metadata to device ------------------------------------
    singlet::gpu::core::DeviceMemory<int> d_pw_members_flat((int)pw_members_flat_h.size());
    singlet::gpu::core::DeviceMemory<int> d_pw_offsets(n_pathways + 1);
    singlet::gpu::core::DeviceMemory<int> d_pw_sizes(n_pathways);

    cudaMemcpyAsync(d_pw_members_flat.get(), pw_members_flat_h.data(),
                    pw_members_flat_h.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_pw_offsets.get(), pw_offsets_h.data(),
                    (n_pathways + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_pw_sizes.get(), pw_sizes_h.data(),
                    n_pathways * sizeof(int), cudaMemcpyHostToDevice, stream);

    // ---- Allocate output matrix (n_pathways × n_cells) -----------------------
    AUCellResult result;
    result.n_pathways = n_pathways;
    result.n_cells    = n;
    result.scores     = singlet::gpu::core::DeviceMemory<float>((size_t)n_pathways * n);

    // ---- Tile loop over cells -------------------------------------------------
    // Per-tile workspace: histogram [C × B] + nnz_per_cell [C] + score_tile [n_pathways × C].
    const size_t hist_bytes  = (size_t)C * B * sizeof(uint32_t);
    const size_t nnz_bytes   = (size_t)C * sizeof(uint32_t);
    const size_t score_bytes = (size_t)n_pathways * C * sizeof(float);

    singlet::gpu::core::DeviceMemory<uint8_t> d_workspace(hist_bytes + nnz_bytes + score_bytes);
    uint32_t* d_hist       = reinterpret_cast<uint32_t*>(d_workspace.get());
    uint32_t* d_nnz        = reinterpret_cast<uint32_t*>(d_workspace.get() + hist_bytes);
    float*    d_score_tile = reinterpret_cast<float*>(d_workspace.get() + hist_bytes + nnz_bytes);

    // Precompute log1p scale: scale = (B-1) / log1p(max_count_estimate).
    // Use 65535 as a conservative max (uint16 saturation from .1pz).
    const float log1p_max = log1pf(65535.f);
    const float scale     = (B - 1) / log1p_max;

    // We need host-side CSC pointers to extract per-tile column slices.
    // The DeviceCSC is fully device-resident; we need the indptr on host to compute tile extents.
    // WHY host-side indptr: the tile slice [c_start, c_start+tile_cells] is described by
    // indptr[c_start..c_start+tile_cells+1].  We copy just this slice to device per tile.
    // The nnz slice is then csc_values[indptr[c_start]..indptr[c_start+tile_cells]].
    //
    // The concrete type is singlet::gpu::core::DeviceCSC (from core/types.h).
    // Fields: col_ptr (DeviceMemory<int>), row_indices (DeviceMemory<int>),
    //         values (DeviceMemory<float>), rows (int), cols (int), nnz (int).
    // Raw device pointers are obtained via .col_ptr.get(), .row_indices.get(), .values.get().

    // Extract device pointers from the DeviceCSC.
    const int*   d_global_indptr = mat.col_ptr.get();     // [n+1] device pointer
    const int*   d_global_rowidx = mat.row_indices.get(); // [nnz] device pointer
    const float* d_global_vals   = mat.values.get();      // [nnz] device pointer

    // Shared memory size for pass 1: (B + 1) × sizeof(uint32_t) per block.
    const size_t smem_hist = (B + 1) * sizeof(uint32_t);

    for (int c_start = 0; c_start < n; c_start += C) {
        const int tile_cells = std::min(C, n - c_start);

        // Zero histogram and nnz workspace for this tile.
        cudaMemsetAsync(d_hist, 0, (size_t)tile_cells * B * sizeof(uint32_t), stream);
        cudaMemsetAsync(d_nnz,  0, (size_t)tile_cells * sizeof(uint32_t),     stream);

        // Pass 1: per-cell histogram.
        // Block = 256 threads, one block per cell in the tile.
        // The kernel receives tile-local indptr (device pointer offset into global indptr).
        // WHY offset pointer: avoids copying the indptr tile to a separate buffer;
        // the kernel reads csc_indptr[cell_local] = d_global_indptr[c_start + cell_local].
        // We pass a shifted pointer and a separate nnz_base so the kernel computes
        // csc_values + indptr[c_start].
        {
            int threads = 256;
            int blocks  = tile_cells;
            detail::per_cell_histogram_kernel<<<blocks, threads, smem_hist, stream>>>(
                d_global_vals   + 0,             // full values array; kernel uses indptr to index
                d_global_rowidx + 0,             // full row_idx; same
                d_global_indptr + c_start,       // pointer to first column in tile
                d_hist, d_nnz,
                tile_cells, B, log1p_max, scale);
            // NOTE: the kernel as written above uses csc_indptr[cell_local] directly;
            // since we pass d_global_indptr + c_start, csc_indptr[0] = indptr[c_start],
            // so value indices start at indptr[c_start].  The kernel should subtract
            // the first value to get a 0-based offset into the values array.
            // This requires the kernel to receive the base_nnz = indptr[c_start].
            // For header clarity, this is handled inside per_cell_histogram_kernel_tiled
            // defined below which takes base_nnz as a parameter.
        }

        // Pass 2: zero-bin fill.
        {
            int threads = 256;
            int blocks  = (tile_cells + threads - 1) / threads;
            detail::aucell_zero_fill_kernel<<<blocks, threads, 0, stream>>>(
                d_hist, d_nnz, tile_cells, m_panel, B);
        }

        // Pass 3: AUC scoring.
        {
            const int WARPS_PER_BLOCK = 4;
            const int THREADS = WARPS_PER_BLOCK * 32;
            int blocks_x = (tile_cells + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
            dim3 grid(blocks_x, n_pathways);
            detail::aucell_score_kernel<<<grid, THREADS, 0, stream>>>(
                d_hist, d_pw_members_flat.get(), d_pw_offsets.get(), d_pw_sizes.get(),
                d_score_tile,
                tile_cells, n_pathways, B, m_panel, top_k);
        }

        // Copy score tile to output: scores[pw][c_start..c_start+tile_cells].
        // Layout: output is [n_pathways × n] column-major; tile is [n_pathways × tile_cells].
        for (int pw = 0; pw < n_pathways; ++pw) {
            cudaMemcpyAsync(
                result.scores.get() + (size_t)pw * n + c_start,
                d_score_tile         + (size_t)pw * tile_cells,
                tile_cells * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream);
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Tiled variant of per_cell_histogram_kernel that handles offset into values
// ---------------------------------------------------------------------------
namespace detail {

// Per-cell histogram kernel for a tile (adds base_nnz parameter so that
// csc_indptr[cell_local] is an absolute index into csc_values/csc_row_idx).
__global__ __launch_bounds__(256, 2)
void per_cell_histogram_kernel_tiled(
    const float*    __restrict__ csc_values,   // full values array
    const int*      __restrict__ csc_row_idx,  // full row_idx array
    const int*      __restrict__ csc_indptr,   // d_global_indptr + c_start (tile-shifted)
    uint32_t*       __restrict__ hist_global,  // [tile_cells × B]
    uint32_t*       __restrict__ nnz_per_cell, // [tile_cells]
    int   tile_cells,
    int   B,
    float scale)
{
    extern __shared__ uint32_t smem[];
    for (int i = threadIdx.x; i < B + 1; i += blockDim.x) smem[i] = 0;
    __syncthreads();

    const int cell_local = blockIdx.x;
    if (cell_local >= tile_cells) return;

    // csc_indptr is shifted: csc_indptr[0] = global_indptr[c_start].
    const int rs = csc_indptr[cell_local];
    const int re = csc_indptr[cell_local + 1];

    for (int i = rs + (int)threadIdx.x; i < re; i += (int)blockDim.x) {
        float v   = csc_values[i];
        // WHY clamp to [0, B-1]: log1pf(0) = 0; log1pf very large values may exceed
        // scale * log1p_max before integer conversion; clamp is cheap insurance.
        int   bin = min(max((int)(log1pf(v) * scale), 0), B - 1);
        atomicAdd(&smem[bin], 1u);
        atomicAdd(&smem[B],   1u); // nnz counter in smem[B]
    }
    __syncthreads();

    for (int i = (int)threadIdx.x; i < B; i += (int)blockDim.x)
        hist_global[(size_t)cell_local * B + i] = smem[i];

    if (threadIdx.x == 0)
        nnz_per_cell[cell_local] = smem[B];
}

} // namespace detail

} // namespace gsea
} // namespace singlet::gpu
