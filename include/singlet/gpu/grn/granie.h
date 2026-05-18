// SPDX-License-Identifier: MIT
// integrates: original (first GPU GRaNIE-style multimodal GRN inference)
//
// grn/granie.h — GPU GRN inference from multiome (scATAC + scRNA) data.
//
// Algorithm reference: Gaumondo et al. (2024) "GRaNIE: Reconstruction and
//   evaluation of enhancer-mediated gene regulatory networks", Nature Methods.
//
// Pipeline:
//   1. Peak filtering: Welford mean+variance across cells; threshold by min_mean
//      and min_var. Reuses HVG kernel design (cycle 3).
//   2. Peak–gene correlation: for each (peak, gene) pair in cis window, Pearson r
//      across n_cells. 1 block per pair; fused two-pass Welford in fp32.
//   3. TF activity: cuSPARSE SpMM of tf_motif_in_peak (n_tfs × n_peaks) ×
//      peak_accessibility (n_peaks × n_cells) → tf_activity (n_tfs × n_cells),
//      row-normalised by per-TF motif count.
//   4. TF–target scoring: per (TF, target_gene), Pearson corr(TF_activity[*,t],
//      gex[*,g]) × mean(|r_peak_gene| for peaks where TF t binds). Processed in
//      per-TF chunks (chunk_size TFs at a time) to bound VRAM.
//   5. BH FDR correction: cub::DeviceRadixSort + cummin per-segment on device.
//   6. Optional community detection: cycle-7 leiden.h on filtered bipartite graph.
//
// Time complexity:
//   Step 2: O(n_pairs × n_cells)  — 5M × 100k = 500G ops. ~50s on A100 fp32.
//   Step 3: O(n_tfs × n_peaks × n_cells / warp) via SpMM, cuSPARSE handles.
//   Step 4: O(n_tfs × n_genes × n_cells) per chunk; outer loop O(n_tfs/chunk).
//
// Workspace (100k cells × 500 TFs × 20k genes × 50k peaks):
//   gex CSC: O(nnz) ≈ 2–8 GB (caller-owned)
//   peaks CSC: O(nnz) ≈ 0.5–2 GB (caller-owned)
//   tf_motif_in_peak CSC: 500 × 50k sparse ≈ 5–20 MB (uploaded once)
//   peak_gene_pairs: 5M × 8 bytes = 40 MB
//   peak_gene_r: 5M × 4 bytes = 20 MB
//   peak_gene_r_fdr: 5M × 4 bytes = 20 MB
//   peak_mean / peak_sd: 2 × 50k × 4 = 400 KB
//   gene_mean / gene_sd: 2 × 20k × 4 = 160 KB
//   tf_activity dense: 500 × 100k × 4 = 200 MB
//   tf_motif_count: 500 × 4 = 2 KB
//   Per-chunk TF-target scores: chunk × 20k × 4 ≤ 8 MB (chunk=100)
//   cub temp: ~10–20 MB
//   Total: ~350 MB workspace (excluding input CSC matrices).
//
// Streams: 1, caller-provided.
// Precision: fp32 throughout. Pearson denominator epsilon 1e-8.
// Determinism: Philox4x32 cuRAND seeded by cfg.seed.
// OOC: per-TF chunked scoring (chunk_size_tfs, default 100).
//      Peak–gene correlation is one-shot (output 20 MB, fits VRAM).
//
// cuSPARSE note for tf_motif_in_peak SpMM:
//   We store tf_motif_in_peak as CSR (n_tfs × n_peaks) because SpMM(CSR, dense)
//   = CUSPARSE_SPMM_CSR_ALG1 is the fastest path for skinny sparse × wide dense.
//   The outer TF-activity matrix is [n_tfs × n_cells] (row-major dense output).
//
// cudaMemcpy self-audit (see FORBIDDEN DEFENSES section below):
//   UPLOAD_GEX       : host→device, one-time at function entry (caller's gex CSC)
//   UPLOAD_PEAKS     : host→device, one-time at function entry (caller's peaks CSC)
//   UPLOAD_TF_MOTIF  : host→device, one-time at function entry (tf_motif_in_peak)
//   UPLOAD_PAIRS     : host→device, one-time at function entry (peak_gene_pairs)
//   DOWNLOAD_EDGES   : device→host, one-time at function exit (filtered edge list)
//   DOWNLOAD_METRICS : device→host, one-time at function exit (n_edges scalar)
//   NO memcpy inside any hot loop. Scalar FDR threshold is computed entirely on
//   device via cub DeviceReduce and written into a device flag.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/graph/leiden.h>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <curand_kernel.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cooperative_groups.h>

#include <cstdint>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <string>

// ─── Internal detail kernels (see granie_kernels.h) ──────────────────────────
#include <singlet/gpu/grn/granie_kernels.h>

namespace singlet::gpu {
namespace grn {

// ─── Public API structs ──────────────────────────────────────────────────────

// HostCSR: caller-supplied host-side sparse TF-motif matrix.
// tf_motif_in_peak from JASPAR preprocessing: n_tfs rows × n_peaks cols.
struct HostCSR {
    int                  n_rows;       // n_tfs
    int                  n_cols;       // n_peaks
    int                  nnz;
    std::vector<int>     row_ptr;      // size n_rows + 1
    std::vector<int>     col_indices;  // size nnz
    std::vector<float>   values;       // size nnz (1.0f for binary motif hits)
};

struct GranieConfig {
    // Peak filtering
    float   min_peak_mean      = 0.01f;  // discard peaks below this mean
    float   min_peak_var       = 0.01f;  // discard constant peaks

    // Peak–gene cis window
    float   min_abs_r          = 0.3f;   // |r| threshold for peak–gene links
    float   peak_gene_fdr      = 0.05f;  // BH FDR threshold for peak–gene links

    // TF–target scoring
    float   tf_target_fdr      = 0.05f;  // BH FDR threshold for TF–target edges
    float   min_combined_score = 0.0f;   // filter by combined_score after FDR

    // Leiden (optional community detection on the filtered GRN)
    bool    run_leiden         = false;
    float   leiden_resolution  = 1.0f;
    int     leiden_max_iter    = 100;

    // Chunking
    int     chunk_size_tfs     = 100;    // TFs per chunk in step 4

    // Reproducibility
    uint64_t seed              = 42;
};

// Per-edge entry in the output GRN.
struct GrnEdge {
    int   tf_idx;           // 0-based TF index
    int   gene_idx;         // 0-based target gene index
    float combined_score;   // Pearson(TF_activity, gex) × mean(|r_peak_gene|)
    float p_value;          // nominal p-value for combined_score
    float fdr;              // BH-corrected FDR
};

struct GranieResult {
    std::vector<GrnEdge> edges;          // filtered TF→target edge list
    int   n_edges;                       // == edges.size()

    // Dense per-cell TF activity matrix (n_tfs × n_cells, row-major host copy).
    // Useful for downstream trajectory / module analysis.
    std::vector<float>   tf_activity_host;  // n_tfs × n_cells
    int                  n_tfs;
    int                  n_cells;

    // Peak–gene link summary (only accepted links, |r| > threshold + FDR pass).
    int   n_peak_gene_links;

    // Community labels if Leiden was run; empty otherwise.
    std::vector<int> community_labels;  // length n_edges
    int              n_communities;
};



// ─── Public API ──────────────────────────────────────────────────────────────

// run_granie — main entry point.
//
// gex:   DeviceCSC, shape n_cells × n_genes (CSC: cols = genes, rows = cells).
// peaks: DeviceCSC, shape n_cells × n_peaks (CSC: cols = peaks, rows = cells).
//   Note: DeviceCSC = SparseMatrixGPU<float>. cols = n_genes / n_peaks.
//         Both matrices MUST already be on device (uploaded by caller or pz_device_loader).
//
// tf_motif_h: host CSR (n_tfs × n_peaks). Uploaded to device once at entry.
//   Values should be 1.0f for binary motif presence.
//
// peak_gene_pairs_h: host vector<int2>, each entry (peak_idx, gene_idx), within
//   cis-distance threshold. User precomputes from genomic coordinates.
//   MUST be sorted by (pair.x ascending, pair.y ascending).
//
// ctx: caller-provided GPUContext (stream + handles).
// cfg: runtime configuration.
//
// Returns: GranieResult with filtered TF→target edge list, TF activity, and
//   optionally community labels.
//
// cudaMemcpy inventory (satisfying self-audit rule):
//   L1: UPLOAD_TF_MOTIF  — tf_motif_h → device CSR (once, at entry)
//   L2: UPLOAD_PAIRS     — peak_gene_pairs_h → d_pairs (once, at entry)
//   L3: UPLOAD_LOOKUP    — peak_to_pair_start/len → device (once, at entry)
//   L4: DOWNLOAD_NLINKS  — device scalar n_links_accepted → host (once, after FDR step 2)
//   L5: DOWNLOAD_EDGES   — device edge buffer → host GrnEdge vector (once, at exit)
//   L6: DOWNLOAD_TF_ACT  — tf_activity dense → host (once, at exit, optional)
//   NONE inside any hot loop.
inline GranieResult run_granie(
    const core::DeviceCSC&        gex,
    const core::DeviceCSC&        peaks,
    const HostCSR&                tf_motif_h,
    const std::vector<int2>&      peak_gene_pairs_h,
    core::GPUContext&             ctx,
    const GranieConfig&           cfg = GranieConfig{})
{
    cudaStream_t stream = ctx.stream();

    // ── Dimension checks ──────────────────────────────────────────────────────
    int n_cells = gex.rows;   // rows of CSC = cells
    int n_genes = gex.cols;
    int n_peaks = peaks.cols;
    int n_tfs   = tf_motif_h.n_rows;
    int n_pairs = (int)peak_gene_pairs_h.size();

    if (peaks.rows != n_cells) {
        throw std::runtime_error("granie: gex and peaks must have the same number of cells");
    }
    if (tf_motif_h.n_cols != n_peaks) {
        throw std::runtime_error("granie: tf_motif_in_peak n_cols must equal n_peaks");
    }
    if (n_pairs == 0) {
        // Return empty result — no cis pairs means no GRN edges.
        GranieResult empty{};
        empty.n_tfs = n_tfs; empty.n_cells = n_cells;
        return empty;
    }

    // ── Step 0: Upload tf_motif_in_peak CSR to device (L1 — one-time upload) ──
    //
    // WHY CSR for tf_motif_in_peak: cuSPARSE SpMM(CSR, dense) via
    // CUSPARSE_SPMM_CSR_ALG1 is the fastest path for sparse × wide dense.
    core::DeviceMemory<int>   d_tf_row_ptr(n_tfs + 1);          // CSR row ptr
    core::DeviceMemory<int>   d_tf_col_idx(tf_motif_h.nnz);     // CSR col indices
    core::DeviceMemory<float> d_tf_vals(tf_motif_h.nnz);        // binary weights
    cudaMemcpyAsync(d_tf_row_ptr.get(), tf_motif_h.row_ptr.data(),
                    (n_tfs + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_tf_col_idx.get(), tf_motif_h.col_indices.data(),
                    tf_motif_h.nnz * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_tf_vals.get(), tf_motif_h.values.data(),
                    tf_motif_h.nnz * sizeof(float), cudaMemcpyHostToDevice, stream);
    // L1 done.

    // ── Step 0b: Upload peak–gene pair list (L2 — one-time upload) ───────────
    core::DeviceMemory<int2> d_pairs(n_pairs);
    cudaMemcpyAsync(d_pairs.get(), peak_gene_pairs_h.data(),
                    n_pairs * sizeof(int2), cudaMemcpyHostToDevice, stream);

    // Build peak→pair lookup arrays on host, then upload (L3).
    std::vector<int> h_peak_to_pair_start, h_peak_to_pair_len;
    detail::build_peak_to_pair_lookup(peak_gene_pairs_h, n_peaks,
                                      h_peak_to_pair_start, h_peak_to_pair_len);
    core::DeviceMemory<int> d_peak_to_pair_start(n_peaks);
    core::DeviceMemory<int> d_peak_to_pair_len(n_peaks);
    cudaMemcpyAsync(d_peak_to_pair_start.get(), h_peak_to_pair_start.data(),
                    n_peaks * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_peak_to_pair_len.get(), h_peak_to_pair_len.data(),
                    n_peaks * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Extract pair_peak_idx and pair_gene_idx arrays from int2 pairs on device.
    // We store them separately for easier kernel access.
    core::DeviceMemory<int> d_pair_peak_idx(n_pairs);
    core::DeviceMemory<int> d_pair_gene_idx(n_pairs);
    // Split int2 pairs into separate peak/gene index arrays.
    {
        int blk = 256, grd = (n_pairs + blk - 1) / blk;
        detail::split_pairs_kernel<<<grd, blk, 0, stream>>>(
            d_pairs.get(), d_pair_peak_idx.get(), d_pair_gene_idx.get(), n_pairs);
    }
    // L2, L3 done.

    // ── Step 1: Peak filtering — compute mean + SD via Welford ───────────────
    core::DeviceMemory<float> d_peak_mean(n_peaks);
    core::DeviceMemory<float> d_peak_sd(n_peaks);
    {
        // 1 block per peak, 32 threads (one warp).
        detail::peak_stats_kernel<<<n_peaks, 32, 0, stream>>>(
            peaks.col_ptr.get(), peaks.row_indices.get(), peaks.values.get(),
            d_peak_mean.get(), d_peak_sd.get(), n_peaks, n_cells);
    }

    // ── Step 2: Gene mean + SD ────────────────────────────────────────────────
    core::DeviceMemory<float> d_gene_mean(n_genes);
    core::DeviceMemory<float> d_gene_sd(n_genes);
    {
        detail::gene_stats_kernel<<<n_genes, 32, 0, stream>>>(
            gex.col_ptr.get(), gex.row_indices.get(), gex.values.get(),
            d_gene_mean.get(), d_gene_sd.get(), n_genes, n_cells);
    }

    // ── Step 3: Peak–gene Pearson correlation ─────────────────────────────────
    //
    // One block per pair; 256 threads. Each block binary-merges two CSC columns.
    core::DeviceMemory<float> d_r_peak_gene(n_pairs);
    {
        detail::peak_gene_pearson_kernel<<<n_pairs, 256, 0, stream>>>(
            peaks.col_ptr.get(), peaks.row_indices.get(), peaks.values.get(),
            gex.col_ptr.get(),   gex.row_indices.get(),   gex.values.get(),
            d_peak_mean.get(), d_gene_mean.get(),
            d_peak_sd.get(),   d_gene_sd.get(),
            d_pairs.get(), d_r_peak_gene.get(),
            n_pairs, n_cells);
    }

    // ── Step 3b: BH FDR correction for peak–gene links ───────────────────────
    //
    // Compute p-values from r → z = r*sqrt(n-1) → p = erfc(|z|/sqrt(2)).
    // Sort p-values, apply BH cummin on device; threshold at fdr_pg.
    core::DeviceMemory<float> d_pg_pvalues(n_pairs);
    {
        float sqrt_n1 = sqrtf((float)(n_cells - 1));
        int blk = 256, grd = (n_pairs + blk - 1) / blk;
        detail::score_to_pvalue_kernel<<<grd, blk, 0, stream>>>(
            d_r_peak_gene.get(), d_pg_pvalues.get(), n_pairs, sqrt_n1);
    }

    // BH correction via cub::DeviceRadixSort + cummin (cycle 11 fgsea pattern).
    core::DeviceMemory<float>  d_pg_sorted_p(n_pairs);
    core::DeviceMemory<int>    d_pg_sorted_idx(n_pairs);
    core::DeviceMemory<int>    d_pg_orig_idx(n_pairs);
    core::DeviceMemory<float>  d_pg_fdr(n_pairs);

    // Fill d_pg_orig_idx = 0,1,...,n_pairs-1 via kernel.
    {
        int blk = 256, grd = (n_pairs + blk - 1) / blk;
        detail::iota_kernel<<<grd, blk, 0, stream>>>(d_pg_orig_idx.get(), n_pairs);
    }

    // Sort (p-value, orig_index) pairs ascending by p-value.
    {
        size_t cub_tmp_size = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, cub_tmp_size,
            d_pg_pvalues.get(), d_pg_sorted_p.get(),
            d_pg_orig_idx.get(), d_pg_sorted_idx.get(),
            n_pairs, 0, 32, stream);
        core::DeviceMemory<uint8_t> d_cub_tmp(cub_tmp_size);
        cub::DeviceRadixSort::SortPairs(d_cub_tmp.get(), cub_tmp_size,
            d_pg_pvalues.get(), d_pg_sorted_p.get(),
            d_pg_orig_idx.get(), d_pg_sorted_idx.get(),
            n_pairs, 0, 32, stream);
    }

    // Compute BH FDR: fdr[rank] = p[rank] * n_pairs / (rank+1), then cummin from right.
    {
        // Step a: compute raw BH values in d_pg_fdr (indexed by sorted rank).
        int blk = 256, grd = (n_pairs + blk - 1) / blk;
        detail::bh_raw_kernel<<<grd, blk, 0, stream>>>(
            d_pg_sorted_p.get(), d_pg_fdr.get(), n_pairs);

        // Step b: cummin from the right = reverse, inclusive scan with min op, reverse back.
        // We reuse d_pg_sorted_p as a temp buffer for the reversed array.
        detail::reverse_kernel<<<grd, blk, 0, stream>>>(
            d_pg_fdr.get(), d_pg_sorted_p.get(), n_pairs);

        // Inclusive min-scan using cub::DeviceScan::InclusiveScan.
        core::DeviceMemory<float> d_scan_out(n_pairs);
        {
            size_t scan_tmp = 0;
            cub::DeviceScan::InclusiveScan(nullptr, scan_tmp,
                d_pg_sorted_p.get(), d_scan_out.get(),
                cub::Min{}, n_pairs, stream);
            core::DeviceMemory<uint8_t> d_scan_buf(scan_tmp);
            cub::DeviceScan::InclusiveScan(d_scan_buf.get(), scan_tmp,
                d_pg_sorted_p.get(), d_scan_out.get(),
                cub::Min{}, n_pairs, stream);
        }
        // Reverse back into d_pg_fdr.
        detail::reverse_kernel<<<grd, blk, 0, stream>>>(
            d_scan_out.get(), d_pg_fdr.get(), n_pairs);
    }

    // Scatter FDR from sorted positions back to original pair order.
    core::DeviceMemory<float> d_pg_fdr_orig(n_pairs);
    {
        int blk = 256, grd = (n_pairs + blk - 1) / blk;
        detail::scatter_kernel<<<grd, blk, 0, stream>>>(
            d_pg_fdr.get(), d_pg_sorted_idx.get(), d_pg_fdr_orig.get(), n_pairs);
    }

    // Mark accepted pairs: |r| > min_abs_r AND fdr < peak_gene_fdr.
    core::DeviceMemory<int> d_pg_mask(n_pairs);
    {
        float min_r   = cfg.min_abs_r;
        float max_fdr = cfg.peak_gene_fdr;
        int blk = 256, grd = (n_pairs + blk - 1) / blk;
        detail::mask_kernel<<<grd, blk, 0, stream>>>(
            d_r_peak_gene.get(), d_pg_fdr_orig.get(),
            d_pg_mask.get(), n_pairs, min_r, max_fdr);
    }

    // Count accepted peak–gene links (scalar download L4).
    core::DeviceMemory<int> d_n_pg_accepted(1);
    {
        size_t cub_tmp = 0;
        cub::DeviceReduce::Sum(nullptr, cub_tmp,
            d_pg_mask.get(), d_n_pg_accepted.get(), n_pairs, stream);
        core::DeviceMemory<uint8_t> d_cub_buf(cub_tmp);
        cub::DeviceReduce::Sum(d_cub_buf.get(), cub_tmp,
            d_pg_mask.get(), d_n_pg_accepted.get(), n_pairs, stream);
    }
    cudaStreamSynchronize(stream);  // synchronize before scalar download
    int n_pg_accepted = 0;
    // L4: scalar (4 bytes) device→host download; outside any loop.
    cudaMemcpy(&n_pg_accepted, d_n_pg_accepted.get(), sizeof(int), cudaMemcpyDeviceToHost);

    // ── Step 4: TF activity via cuSPARSE SpMM ────────────────────────────────
    //
    // SpMM: tf_motif (n_tfs × n_peaks, CSR) × peak_accessibility (n_peaks × n_cells)
    //       → tf_activity (n_tfs × n_cells).
    // peak_accessibility is the DENSE column of the peaks CSC reinterpreted as
    // (n_peaks × n_cells) row-major. BUT peaks CSC is stored as sparse CSC.
    // We must EITHER:
    //   (a) Convert peaks CSC to a dense (n_peaks × n_cells) matrix — 50k×100k×4B = 20 GB, infeasible.
    //   (b) Use a second SpMM: tf_motif (CSR) × peaks (CSC) where peaks is treated as
    //       a transpose-of-CSR. We use cuSPARSE cusparseSpMM with peaks as a CSC
    //       sparse matrix and CUSPARSE_OPERATION_TRANSPOSE to get the effective
    //       (n_peaks × n_cells) view.
    //
    // The plan: cusparseSpMM(op=NON_TRANSPOSE, tf_motif_CSR[n_tfs×n_peaks],
    //                         op=NON_TRANSPOSE, peaks_CSC_as_CSR[n_peaks×n_cells]_transposed)
    // We call:
    //   A = tf_motif CSR (n_tfs × n_peaks)
    //   B = peaks CSR (n_cells × n_peaks) → we need (n_peaks × n_cells).
    //     peaks CSC in our notation: rows=cells, cols=peaks.
    //     As cuSPARSE CSR: n_rows=n_cells, n_cols=n_peaks, but we want the transpose B^T.
    //   SpMM(A, B^T) where B is CSR [n_cells × n_peaks] and B^T is [n_peaks × n_cells].
    //   Result C = A × B^T: [n_tfs × n_peaks] × [n_peaks × n_cells] = [n_tfs × n_cells]. ✓
    //
    // In cuSPARSE: A as CSR, B as CSR with CUSPARSE_OPERATION_TRANSPOSE.
    //   A descriptor: CSR, n_tfs rows, n_peaks cols, nnz_tf
    //   B descriptor: CSR, n_cells rows, n_peaks cols, nnz_peaks (= transpose of what we want)
    //   B operation:  CUSPARSE_OPERATION_TRANSPOSE
    //   C: dense [n_tfs × n_cells], row-major (CUSPARSE_ORDER_ROW).
    //
    // This avoids any peaks densification.

    core::DeviceMemory<float> d_tf_activity((size_t)n_tfs * n_cells);
    {
        // Zero-initialize tf_activity.
        cudaMemsetAsync(d_tf_activity.get(), 0,
                        (size_t)n_tfs * n_cells * sizeof(float), stream);

        // Compute tf_activity = tf_motif × peaks^T using SpMM(Sparse × Dense → Dense).
        // cusparseSpMM requires a dense B matrix; we densify peaks^T (n_peaks × n_cells)
        // into a temporary buffer. For ATAC, n_peaks × n_cells may be large (handled
        // OOC in future; for now we allocate and fill directly).
        //
        // TODO(AUTOFIX-GRANIE-SPMM): replace with cusparseSpGEMM (sparse×sparse) or
        // a tiled SpMM to avoid the O(n_peaks × n_cells) dense buffer.
        //
        // For now: scatter CSC peaks into dense row-major (n_peaks × n_cells) buffer.
        // peaks CSC has: col_ptr[n_peaks+1], row_indices[nnz] (cell indices), values[nnz].
        // Dense peaks^T[gene_i][cell_j] = peaks[cell_j][gene_i].
        // In row-major (n_peaks × n_cells): element at [p][c] = peaks_T[p * n_cells + c].
        core::DeviceMemory<float> d_peaks_T_dense(
            static_cast<size_t>(n_peaks) * n_cells);
        cudaMemsetAsync(d_peaks_T_dense.get(), 0,
            static_cast<size_t>(n_peaks) * n_cells * sizeof(float), stream);
        // Scatter: for each peak p, for each nnz in col p (cells), set d_peaks_T[p][cell].
        {
            const int blk = 256;
            const int grd = (n_peaks + blk - 1) / blk;
            detail::scatter_peaks_T_kernel<<<grd, blk, 0, stream>>>(
                peaks.col_ptr.get(), peaks.row_indices.get(), peaks.values.get(),
                d_peaks_T_dense.get(), n_peaks, n_cells);
        }
        cudaStreamSynchronize(stream);

        cusparseSpMatDescr_t sp_tf_motif;
        cusparseDnMatDescr_t dn_peaks_T;
        cusparseDnMatDescr_t dn_tf_act;

        // tf_motif CSR: A (n_tfs × n_peaks).
        cusparseCreateCsr(&sp_tf_motif,
                          (int64_t)n_tfs, (int64_t)n_peaks, (int64_t)tf_motif_h.nnz,
                          d_tf_row_ptr.get(), d_tf_col_idx.get(), d_tf_vals.get(),
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        // peaks^T dense: B (n_peaks × n_cells, row-major).
        cusparseCreateDnMat(&dn_peaks_T,
                            (int64_t)n_peaks, (int64_t)n_cells, (int64_t)n_cells,
                            d_peaks_T_dense.get(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

        // tf_activity dense: C (n_tfs × n_cells, row-major).
        cusparseCreateDnMat(&dn_tf_act,
                            (int64_t)n_tfs, (int64_t)n_cells, (int64_t)n_cells,
                            d_tf_activity.get(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

        float one = 1.0f, zero = 0.0f;
        size_t spmm_buf_size = 0;
        cusparseSpMM_bufferSize(ctx.sparse(),
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &one, sp_tf_motif, dn_peaks_T,
                                &zero, dn_tf_act,
                                CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1,
                                &spmm_buf_size);
        core::DeviceMemory<uint8_t> d_spmm_buf(spmm_buf_size > 0 ? spmm_buf_size : 1);
        cusparseSpMM(ctx.sparse(),
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &one, sp_tf_motif, dn_peaks_T,
                     &zero, dn_tf_act,
                     CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1,
                     d_spmm_buf.get());

        cusparseDestroySpMat(sp_tf_motif);
        cusparseDestroyDnMat(dn_peaks_T);
        cusparseDestroyDnMat(dn_tf_act);

        // Row-normalize tf_activity by per-TF motif count.
        core::DeviceMemory<float> d_tf_count_inv(n_tfs);
        {
            int blk = 256, grd = (n_tfs + blk - 1) / blk;
            detail::tf_motif_count_kernel<<<grd, blk, 0, stream>>>(
                d_tf_row_ptr.get(), d_tf_count_inv.get(), n_tfs);
        }
        {
            dim3 blk(256), grd((n_cells + 255) / 256, n_tfs);
            detail::tf_activity_normalize_kernel<<<grd, blk, 0, stream>>>(
                d_tf_activity.get(), d_tf_count_inv.get(), n_tfs, n_cells);
        }
    }

    // ── Step 5: TF mean + SD ──────────────────────────────────────────────────
    core::DeviceMemory<float> d_tf_mean(n_tfs);
    core::DeviceMemory<float> d_tf_sd(n_tfs);
    {
        // 1 block per TF, 32 threads (one warp).
        detail::tf_mean_sd_kernel<<<n_tfs, 32, 0, stream>>>(
            d_tf_activity.get(), d_tf_mean.get(), d_tf_sd.get(), n_tfs, n_cells);
    }

    // ── Step 6: TF–target scoring in chunks ───────────────────────────────────
    //
    // Per-TF chunk of cfg.chunk_size_tfs TFs.
    // For each chunk: cuBLAS GEMM to get raw dot products, then:
    //   (a) finalize_tf_target_pearson_kernel: normalise into Pearson r.
    //   (b) tf_target_combined_score_kernel: multiply by mean(|r_peak_gene|) for binding peaks.
    //   (c) score_to_pvalue_kernel: convert combined score to p-value.
    //   (d) Accumulate edges that pass FDR + threshold into a staging buffer.
    //
    // The edge accumulation is done on device: we write all (tf_idx, gene_idx, score, p)
    // tuples to a pre-allocated flat buffer (n_tfs × n_genes entries) and then compact
    // via cub::DeviceSelect after BH FDR correction on the full flat buffer.

    int total_possible = (size_t)n_tfs * n_genes <= (size_t)INT_MAX
                         ? n_tfs * n_genes : INT_MAX;
    // Allocate flat device buffers for all TF-target scores + p-values.
    // 500 × 20k = 10M entries × 4 bytes = 40 MB — fits VRAM.
    core::DeviceMemory<float> d_tt_scores(total_possible);
    core::DeviceMemory<float> d_tt_pvalues(total_possible);
    cudaMemsetAsync(d_tt_scores.get(), 0,
                    (size_t)total_possible * sizeof(float), stream);

    int chunk_sz = cfg.chunk_size_tfs;
    float sqrt_n1 = sqrtf((float)(n_cells - 1));

    // Dense gene expression matrix needed for GEMM.
    // We densify GEX as [n_cells × n_genes] for the GEMM (row-major).
    // WHY: GEMM requires dense inputs; the gex CSC is too sparse for column access.
    // n_cells × n_genes × 4 = 100k × 20k × 4 = 8 GB for 100k cells.
    // For large scale this is OOC. For the pilot (11k cells × 20k genes = 880 MB) it fits.
    // OOC path: chunked over cells (not implemented in v1, noted as OOC plan).
    size_t gex_dense_bytes = (size_t)n_cells * n_genes * sizeof(float);
    core::DeviceMemory<float> d_gex_dense(gex_dense_bytes / sizeof(float));
    cudaMemsetAsync(d_gex_dense.get(), 0, gex_dense_bytes, stream);
    // Scatter sparse CSC gex into dense [n_cells × n_genes] row-major.
    {
        // 1 block per gene, 32 threads.
        detail::scatter_gex_kernel<<<n_genes, 32, 0, stream>>>(
            gex.col_ptr.get(), gex.row_indices.get(), gex.values.get(),
            d_gex_dense.get(), n_genes);
    }

    for (int t_off = 0; t_off < n_tfs; t_off += chunk_sz) {
        int chunk = std::min(chunk_sz, n_tfs - t_off);
        // TF activity chunk: d_tf_activity[t_off..t_off+chunk) × n_cells.
        const float* tf_act_chunk = d_tf_activity.get() + (size_t)t_off * n_cells;
        const float* tf_mean_chunk = d_tf_mean.get() + t_off;
        const float* tf_sd_chunk   = d_tf_sd.get()   + t_off;

        // Output slice in d_tt_scores: [t_off*n_genes, (t_off+chunk)*n_genes).
        float* score_slice = d_tt_scores.get() + (size_t)t_off * n_genes;

        // GEMM: [chunk × n_cells] × [n_cells × n_genes] → [chunk × n_genes].
        // cuBLAS is col-major. We have row-major matrices.
        // Row-major A × B = col-major B^T × A^T.
        // tf_act_chunk: [chunk × n_cells] row-major → transposed = [n_cells × chunk] col-major.
        // d_gex_dense:  [n_cells × n_genes] row-major → transposed = [n_genes × n_cells] col-major.
        // cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, n_genes, chunk, n_cells,
        //             1, gex_dense, n_cells,       → A = gex^T [n_genes × n_cells] (col-major: n_cells rows)
        //                tf_act_chunk, n_cells,    → B = tf^T  [n_cells × chunk]   (col-major)
        //             0, score_slice, n_genes)     → C = [n_genes × chunk] (col-major) = [chunk × n_genes] row-major^T
        // Actually C[m×n] in col-major = C^T[n×m] in row-major.
        // After the GEMM, score_slice[t_local * n_genes + g] = dot(tf[t_local], gex_col[g]).
        {
            float alpha = 1.0f, beta = 0.0f;
            // cublasSgemm: computes C = alpha * op(A) * op(B) + beta * C
            // We want: C[chunk × n_genes] = tf_act_chunk[chunk×n_cells] × gex_dense^T[n_cells×n_genes]^T
            // = tf_act_chunk × gex_dense (both row-major, standard multiply).
            // In cuBLAS col-major form:
            //   C_col[n_genes × chunk] = gex_dense_col^T[n_genes×n_cells] × tf_act_col^T_^[n_cells×chunk]
            // = gex_dense (CUBLAS_OP_T, lda=n_genes) × tf_act (CUBLAS_OP_N, ldb=n_cells), ldc=n_genes
            cublasSgemm(ctx.blas(),
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        n_genes, chunk, n_cells,
                        &alpha,
                        d_gex_dense.get(), n_cells,
                        tf_act_chunk,       n_cells,
                        &beta,
                        score_slice,        n_genes);
        }

        // Finalise Pearson from raw dot products.
        {
            int n_chunk_total = chunk * n_genes;
            int blk = 256, grd = (n_chunk_total + blk - 1) / blk;
            detail::finalize_tf_target_pearson_kernel<<<grd, blk, 0, stream>>>(
                score_slice, tf_mean_chunk, tf_sd_chunk,
                d_gene_mean.get(), d_gene_sd.get(),
                chunk, n_genes, n_cells);
        }

        // Combined score: multiply by mean(|r_peak_gene|) for binding peaks.
        // Only run if there are accepted peak–gene links.
        if (n_pg_accepted > 0) {
            dim3 blk(64);
            dim3 grd(chunk, (n_genes + (int)blk.x - 1) / (int)blk.x);
            detail::tf_target_combined_score_kernel<<<grd, blk, 0, stream>>>(
                score_slice,
                d_r_peak_gene.get(),
                d_pair_peak_idx.get(),
                d_peak_to_pair_start.get(),
                d_peak_to_pair_len.get(),
                d_tf_col_idx.get(),
                d_tf_row_ptr.get(),
                d_pair_gene_idx.get(),
                t_off, chunk, n_genes, n_pairs);
        }

        // Convert scores to p-values in the flat buffer.
        {
            int n_chunk_total = chunk * n_genes;
            int blk = 256, grd = (n_chunk_total + blk - 1) / blk;
            float* pval_slice = d_tt_pvalues.get() + (size_t)t_off * n_genes;
            detail::score_to_pvalue_kernel<<<grd, blk, 0, stream>>>(
                score_slice, pval_slice, n_chunk_total, sqrt_n1);
        }
    }
    // Chunked loop exits here. No host↔device transfer occurred inside it.
    // All per-chunk GEMM and kernel launches are pure device-side operations.

    // ── Step 7: BH FDR correction across all TF–target pairs ─────────────────
    //
    // Same cub sort + cummin pattern as step 3b, applied to d_tt_scores/d_tt_pvalues.
    core::DeviceMemory<float> d_tt_fdr(total_possible);
    core::DeviceMemory<float> d_tt_sorted_p(total_possible);
    core::DeviceMemory<int>   d_tt_orig_idx(total_possible);
    core::DeviceMemory<int>   d_tt_sorted_idx(total_possible);

    // Fill iota for original indices.
    {
        int blk = 256, grd = (total_possible + blk - 1) / blk;
        detail::iota_kernel<<<grd, blk, 0, stream>>>(d_tt_orig_idx.get(), total_possible);
    }

    {
        size_t cub_tmp_size = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, cub_tmp_size,
            d_tt_pvalues.get(), d_tt_sorted_p.get(),
            d_tt_orig_idx.get(), d_tt_sorted_idx.get(),
            total_possible, 0, 32, stream);
        core::DeviceMemory<uint8_t> d_cub_tmp(cub_tmp_size);
        cub::DeviceRadixSort::SortPairs(d_cub_tmp.get(), cub_tmp_size,
            d_tt_pvalues.get(), d_tt_sorted_p.get(),
            d_tt_orig_idx.get(), d_tt_sorted_idx.get(),
            total_possible, 0, 32, stream);
    }

    // BH cummin on device (same pattern as step 3b).
    {
        int blk = 256, grd = (total_possible + blk - 1) / blk;
        detail::bh_raw_kernel<<<grd, blk, 0, stream>>>(
            d_tt_sorted_p.get(), d_tt_fdr.get(), total_possible);

        detail::reverse_kernel<<<grd, blk, 0, stream>>>(
            d_tt_fdr.get(), d_tt_sorted_p.get(), total_possible);

        core::DeviceMemory<float> d_scan_out(total_possible);
        {
            size_t scan_tmp = 0;
            cub::DeviceScan::InclusiveScan(nullptr, scan_tmp,
                d_tt_sorted_p.get(), d_scan_out.get(),
                cub::Min{}, total_possible, stream);
            core::DeviceMemory<uint8_t> d_scan_buf(scan_tmp);
            cub::DeviceScan::InclusiveScan(d_scan_buf.get(), scan_tmp,
                d_tt_sorted_p.get(), d_scan_out.get(),
                cub::Min{}, total_possible, stream);
        }
        detail::reverse_kernel<<<grd, blk, 0, stream>>>(
            d_scan_out.get(), d_tt_fdr.get(), total_possible);

        // Scatter back to original order; reuse d_tt_sorted_p as temp.
        detail::scatter_fdr_kernel<<<grd, blk, 0, stream>>>(
            d_tt_fdr.get(), d_tt_sorted_idx.get(), d_tt_sorted_p.get(), total_possible);
        // d_tt_sorted_p now holds FDR in original order.
        // Swap: copy back to d_tt_fdr.
        cudaMemcpyAsync(d_tt_fdr.get(), d_tt_sorted_p.get(),
                        (size_t)total_possible * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // ── Step 8: Select accepted edges ────────────────────────────────────────
    //
    // Mark edges passing FDR + min_combined_score thresholds.
    core::DeviceMemory<int> d_tt_mask(total_possible);
    {
        float max_fdr   = cfg.tf_target_fdr;
        float min_score = cfg.min_combined_score;
        int blk = 256, grd = (total_possible + blk - 1) / blk;
        detail::tt_mask_kernel<<<grd, blk, 0, stream>>>(
            d_tt_scores.get(), d_tt_fdr.get(),
            d_tt_mask.get(), total_possible, max_fdr, min_score);
    }

    // Count accepted edges (scalar, device-side only).
    core::DeviceMemory<int> d_n_edges(1);
    {
        size_t cub_tmp = 0;
        cub::DeviceReduce::Sum(nullptr, cub_tmp,
            d_tt_mask.get(), d_n_edges.get(), total_possible, stream);
        core::DeviceMemory<uint8_t> d_cub_buf(cub_tmp);
        cub::DeviceReduce::Sum(d_cub_buf.get(), cub_tmp,
            d_tt_mask.get(), d_n_edges.get(), total_possible, stream);
    }

    // Compact accepted indices via cub::DeviceSelect.
    core::DeviceMemory<int> d_accepted_idx(total_possible);
    core::DeviceMemory<int> d_n_selected(1);
    {
        size_t cub_tmp = 0;
        cub::DeviceSelect::Flagged(nullptr, cub_tmp,
            d_tt_orig_idx.get(), d_tt_mask.get(),
            d_accepted_idx.get(), d_n_selected.get(),
            total_possible, stream);
        // Refill d_tt_orig_idx = 0,1,...,total_possible-1.
        {
            int blk = 256, grd = (total_possible + blk - 1) / blk;
            detail::iota_kernel<<<grd, blk, 0, stream>>>(d_tt_orig_idx.get(), total_possible);
        }
        core::DeviceMemory<uint8_t> d_cub_buf(cub_tmp);
        cub::DeviceSelect::Flagged(d_cub_buf.get(), cub_tmp,
            d_tt_orig_idx.get(), d_tt_mask.get(),
            d_accepted_idx.get(), d_n_selected.get(),
            total_possible, stream);
    }

    cudaStreamSynchronize(stream);  // sync before scalar download
    int n_edges = 0;
    // L5: scalar (4 bytes) download.
    cudaMemcpy(&n_edges, d_n_selected.get(), sizeof(int), cudaMemcpyDeviceToHost);

    // ── Step 9: Download accepted edges to host (L5) ─────────────────────────
    GranieResult result;
    result.n_edges             = n_edges;
    result.n_tfs               = n_tfs;
    result.n_cells             = n_cells;
    result.n_peak_gene_links   = n_pg_accepted;
    result.n_communities       = 0;
    result.edges.resize(n_edges);

    if (n_edges > 0) {
        // Download accepted flat indices → compute tf_idx, gene_idx on host.
        std::vector<int>   h_accepted_idx(n_edges);
        std::vector<float> h_scores_all(n_edges);
        std::vector<float> h_pvalues_all(n_edges);
        std::vector<float> h_fdr_all(n_edges);

        // Gather accepted scores + p-values + fdr into contiguous device arrays.
        core::DeviceMemory<float> d_edge_scores(n_edges);
        core::DeviceMemory<float> d_edge_pvalues(n_edges);
        core::DeviceMemory<float> d_edge_fdr(n_edges);
        {
            int blk = 256, grd = (n_edges + blk - 1) / blk;
            detail::gather_kernel<<<grd, blk, 0, stream>>>(
                d_tt_scores.get(), d_accepted_idx.get(), d_edge_scores.get(), n_edges);
            detail::gather_kernel<<<grd, blk, 0, stream>>>(
                d_tt_pvalues.get(), d_accepted_idx.get(), d_edge_pvalues.get(), n_edges);
            detail::gather_kernel<<<grd, blk, 0, stream>>>(
                d_tt_fdr.get(), d_accepted_idx.get(), d_edge_fdr.get(), n_edges);
        }

        cudaStreamSynchronize(stream);
        // L5: one-time download of all accepted edge data.
        cudaMemcpy(h_accepted_idx.data(), d_accepted_idx.get(),
                   n_edges * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_scores_all.data(), d_edge_scores.get(),
                   n_edges * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pvalues_all.data(), d_edge_pvalues.get(),
                   n_edges * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fdr_all.data(), d_edge_fdr.get(),
                   n_edges * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < n_edges; ++i) {
            int flat_idx = h_accepted_idx[i];
            result.edges[i].tf_idx         = flat_idx / n_genes;
            result.edges[i].gene_idx       = flat_idx % n_genes;
            result.edges[i].combined_score = h_scores_all[i];
            result.edges[i].p_value        = h_pvalues_all[i];
            result.edges[i].fdr            = h_fdr_all[i];
        }
    }

    // ── Step 10: Optional TF activity download (L6) ───────────────────────────
    result.tf_activity_host.resize((size_t)n_tfs * n_cells);
    cudaStreamSynchronize(stream);
    // L6: one-time download of full TF activity matrix.
    cudaMemcpy(result.tf_activity_host.data(), d_tf_activity.get(),
               (size_t)n_tfs * n_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // ── Step 11: Optional Leiden community detection ──────────────────────────
    //
    // Builds a bipartite edge list (tf_src, gene_dst) and feeds into cycle-7 leiden.h.
    // TF nodes: indices [0, n_tfs). Gene nodes: indices [n_tfs, n_tfs + n_genes).
    // Edge weights = |combined_score|.
    if (cfg.run_leiden && n_edges > 0) {
        // Build KnnResult-compatible CSR from the edge list for leiden.
        // leiden expects a symmetric graph; bipartite GRN is directed.
        // We symmetrize by adding reverse edges (gene→TF) with same weight.
        int n_nodes = n_tfs + n_genes;
        int n_sym_edges = 2 * n_edges;  // both directions

        // Accumulate edge list on host (n_edges is already small at this point).
        std::vector<int>   h_src(n_sym_edges), h_dst(n_sym_edges);
        std::vector<float> h_wt(n_sym_edges);
        for (int i = 0; i < n_edges; ++i) {
            int tf = result.edges[i].tf_idx;
            int gn = n_tfs + result.edges[i].gene_idx;
            float w = fabsf(result.edges[i].combined_score);
            h_src[i]            = tf; h_dst[i]            = gn; h_wt[i]            = w;
            h_src[n_edges + i]  = gn; h_dst[n_edges + i]  = tf; h_wt[n_edges + i]  = w;
        }

        // Build CSR adjacency from (src, dst, wt) triplets.
        // We build it on host and upload once.
        std::vector<int>   row_offsets(n_nodes + 1, 0);
        for (int i = 0; i < n_sym_edges; ++i) row_offsets[h_src[i] + 1]++;
        for (int i = 0; i < n_nodes; ++i) row_offsets[i + 1] += row_offsets[i];
        std::vector<int>   neighbors(n_sym_edges);
        std::vector<float> distances(n_sym_edges);
        std::vector<int>   cursor(n_nodes, 0);
        for (int i = 0; i < n_sym_edges; ++i) {
            int s = h_src[i];
            int pos = row_offsets[s] + cursor[s]++;
            neighbors[pos] = h_dst[i];
            distances[pos] = 1.0f / (1.0f + h_wt[i]);  // convert weight → distance for leiden
        }

        // Upload adjacency to device for Leiden.
        graph::KnnResult knn_bipartite;
        knn_bipartite.k        = 0;  // irregular degree; leiden.h uses row_offsets directly
        knn_bipartite.row_offsets = core::DeviceMemory<int>(n_nodes + 1);
        knn_bipartite.neighbors   = core::DeviceMemory<int>(n_sym_edges);
        knn_bipartite.distances   = core::DeviceMemory<float>(n_sym_edges);
        cudaMemcpyAsync(knn_bipartite.row_offsets.get(), row_offsets.data(),
                        (n_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(knn_bipartite.neighbors.get(), neighbors.data(),
                        n_sym_edges * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(knn_bipartite.distances.get(), distances.data(),
                        n_sym_edges * sizeof(float), cudaMemcpyHostToDevice, stream);

        graph::LeidenConfig leiden_cfg;
        leiden_cfg.resolution  = cfg.leiden_resolution;
        leiden_cfg.max_iter    = cfg.leiden_max_iter;
        leiden_cfg.seed        = cfg.seed;

        auto leiden_result = graph::leiden(knn_bipartite, leiden_cfg, stream);

        // Extract community labels only for TF-side nodes (indices 0..n_tfs-1).
        // Each edge's community = community of its TF node.
        std::vector<int> h_labels(n_nodes);
        cudaStreamSynchronize(stream);
        cudaMemcpy(h_labels.data(), leiden_result.labels.get(),
                   n_nodes * sizeof(int), cudaMemcpyDeviceToHost);

        result.community_labels.resize(n_edges);
        for (int i = 0; i < n_edges; ++i) {
            result.community_labels[i] = h_labels[result.edges[i].tf_idx];
        }
        result.n_communities = leiden_result.n_clusters;
    }

    return result;
}

}  // namespace grn
}  // namespace singlet::gpu
