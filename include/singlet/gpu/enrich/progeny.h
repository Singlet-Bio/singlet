// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (first GPU PROGENy pathway activity scoring)
//
// enrich/progeny.h — PROGENy pathway activity scoring, GPU-native.
//
// Algorithm: Schubert et al. (2018) "Perturbation-response genes reveal signaling
//   footprints in cancer gene expression", Nature Communications 9:20.
//   Holland et al. (2020) "Robustness and applicability of transcription factor
//   and pathway analysis tools on single-cell RNA-seq data", Genome Biology 21:36.
//
// THIS IS ORTHOGONAL TO AUCell/fgsea:
//   AUCell   — top-N AUC (ranks only; gene-set membership).
//   fgsea    — ranked list enrichment (ordered gene list).
//   PROGENy  — weighted sum of expression × pre-trained pathway weights (no ranking).
//
// Pipeline overview:
//
//   Step 1 — SpMM (sparse × dense):
//     activity[n_cells × n_pathways] = expression[n_cells × n_genes]ᵀ · weights[n_genes × n_pathways]
//     = cuSPARSE SpMM (CSC expression as CSR transpose, dense progeny_weights).
//     WHY SpMM not GEMM: expression is sparse (10x data: ~5% nonzero); SpMM is 10–20× faster.
//     WHY transpose: expression in singlify is genes × cells (CSC); SpMM wants cells × genes,
//     so we treat the CSC as CSR (transposed), which cuSPARSE supports natively.
//
//   Step 2 — two-pass Welford normalization (optional):
//     For each pathway p:
//       mean_p = mean(activity[:, p])   (over cells)
//       std_p  = std(activity[:, p])
//       activity[:, p] = (activity[:, p] - mean_p) / (std_p + eps)
//     Implemented as a single kernel pass per pathway using Welford's online algorithm.
//     WHY two-pass via Welford: numerically stable for fp32; avoids catastrophic cancellation.
//
// Memory layout:
//   expression (input): DeviceCSC (genes × cells), CSC from pz_device_loader.
//   progeny_weights (input): dense [n_genes × n_pathways] fp32, column-major.
//     Shipped pre-trained weights (14 pathways × top-100 genes): loaded from TSV at
//     program start, uploaded to device once.
//   activity (output): dense [n_cells × n_pathways] fp32, row-major.
//
// Workspace budget (100k cells × 20k genes × 14 pathways):
//   progeny_weights : 20k × 14 × 4 = 1.1 MB  (tiny)
//   activity        : 100k × 14 × 4 = 5.6 MB
//   cuSPARSE temp   : typically < 64 MB (estimated; queried at runtime)
//   Total: ~70 MB.
//
// Streams: 1, caller-provided.
//
// Precision: fp32 throughout. Pathway activity = weighted sum of log-normalized
//   counts; fp32 is sufficient for the ~2 decimal places of precision needed.
//
// Determinism: cuBLAS/cuSPARSE SGEMM/SpMM is deterministic for fixed input +
//   fixed GPU architecture. Normalization uses a deterministic two-pass kernel.
//
// OOC plan: cell-batched. Process C_OOC cells per batch (default: all cells if
//   VRAM allows; reduce if activity output exceeds available memory).
//   For 1M cells × 14 pathways: 1M × 14 × 4 = 56 MB — fits in any A100.
//
// Correctness tolerance (vs decoupleR Python): Spearman ρ ≥ 0.95 per pathway.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cooperative_groups.h>

#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <tuple>

namespace singlet_gpu {
namespace enrich {

// ---------------------------------------------------------------------------
// Pre-trained PROGENy weights
// ---------------------------------------------------------------------------

// PROGENyWeights holds the pre-trained weight matrix (n_genes × n_pathways).
// Standard human model: 14 pathways × top-100 genes per pathway.
// Source: Holland et al. (2020), DoRothEA/PROGENy R package (Bioconductor).
//
// Structure: sparse representation (only top-100 genes per pathway are non-zero).
// Internally stored as dense [n_genes × n_pathways] after gene index remapping.
// The zero columns (non-top-100 genes) contribute nothing to the SpMM output.

struct PROGENyWeights {
    int n_genes;                               // total genes in expression matrix
    int n_pathways;                            // number of pathways (14 for human)
    std::vector<std::string> pathway_names;    // [n_pathways]
    std::vector<std::string> gene_names;       // [n_genes] (mapped from expression matrix)
    std::vector<float>       weights_dense;    // [n_genes × n_pathways], column-major (n_genes rows)

    // GPU copy of weights — uploaded once at program start.
    singlet_gpu::core::DeviceMemory<float> d_weights;

    // Whether d_weights has been uploaded.
    bool gpu_ready = false;
};

// PROGENy config
struct PROGENyConfig {
    bool  normalize   = true;   // per-pathway z-score normalization across cells
    float norm_eps    = 1e-6f;  // stability epsilon for std normalization
    int   cell_batch  = 0;      // 0 = process all cells at once (preferred for PROGENy)
};

// PROGENy result
struct PROGENyResult {
    singlet_gpu::core::DeviceMemory<float> activity;  // [n_cells × n_pathways], row-major (device)
    int n_cells    = 0;
    int n_pathways = 0;
    std::vector<std::string> pathway_names;
};

// ---------------------------------------------------------------------------
// TSV weight-file loader (host, one-time setup)
// ---------------------------------------------------------------------------

// load_progeny_weights_tsv:
//   Reads a TSV file with columns: gene_name, pathway_name, weight.
//   Maps gene names to indices using the gene_name_to_idx map.
//   Creates a dense [n_genes × n_pathways] weight matrix.
//
// File format (progeny_human_top100.tsv):
//   gene  pathway  weight
//   EGFR  EGFR     1.234
//   ...

inline PROGENyWeights load_progeny_weights_tsv(
    const std::string&                              tsv_path,
    const std::vector<std::string>&                 gene_names,   // from expression matrix
    const std::unordered_map<std::string, int>&     gene_name_to_idx)
{
    PROGENyWeights pw;
    pw.gene_names = gene_names;
    pw.n_genes    = (int)gene_names.size();

    std::ifstream f(tsv_path);
    if (!f.is_open())
        throw std::runtime_error("PROGENy: cannot open weight file: " + tsv_path);

    // First pass: collect pathway names.
    std::string line, gene, pathway;
    float weight;
    bool header_skipped = false;

    std::vector<std::tuple<int,int,float>> entries;
    std::vector<std::string> pw_order;
    std::unordered_map<std::string, int> pw_idx;

    while (std::getline(f, line)) {
        if (!header_skipped) { header_skipped = true; continue; }
        if (line.empty()) continue;
        std::istringstream ss(line);
        if (!(ss >> gene >> pathway >> weight)) continue;

        // Map pathway name.
        if (pw_idx.find(pathway) == pw_idx.end()) {
            pw_idx[pathway] = (int)pw_order.size();
            pw_order.push_back(pathway);
        }
        // Map gene name.
        auto it = gene_name_to_idx.find(gene);
        if (it == gene_name_to_idx.end()) continue;  // gene not in expression matrix
        entries.emplace_back(it->second, pw_idx[pathway], weight);
    }

    pw.pathway_names = pw_order;
    pw.n_pathways    = (int)pw_order.size();

    // Allocate dense weight matrix (n_genes × n_pathways), column-major.
    // WHY column-major: cuSPARSE SpMM output is row-major (n_cells × n_pathways);
    // weights are accessed gene-by-gene in the SpMM — column-major over pathways
    // gives coalesced access per pathway in the GEMM.
    pw.weights_dense.assign((size_t)pw.n_genes * pw.n_pathways, 0.f);
    for (auto& [gi, pi, w] : entries)
        pw.weights_dense[(size_t)pi * pw.n_genes + gi] = w;

    return pw;
}

// upload_progeny_weights: copy dense weight matrix to GPU (one-time setup).
// CUDAMEMCPY AUDIT (1 of 2): HostToDevice, one-time setup at program start.
inline void upload_progeny_weights(PROGENyWeights& pw, cudaStream_t stream) {
    if (pw.gpu_ready) return;
    pw.d_weights = singlet_gpu::core::DeviceMemory<float>((size_t)pw.n_genes * pw.n_pathways);
    cudaMemcpyAsync(pw.d_weights.get(), pw.weights_dense.data(),
                    pw.weights_dense.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    pw.gpu_ready = true;
}

// ---------------------------------------------------------------------------
// Internal kernels
// ---------------------------------------------------------------------------
namespace detail {

// Welford two-pass normalization kernel.
//
// For each pathway p (column of activity), compute:
//   mean = sum(activity[:, p]) / n_cells
//   var  = sum((activity[:, p] - mean)^2) / n_cells
//   activity[:, p] = (activity[:, p] - mean) / sqrt(var + eps)
//
// Grid: one block per pathway. Threads cooperate to reduce over n_cells.
// WHY one block per pathway: n_pathways is small (14); each block handles all n_cells.
// Welford's online algorithm: single pass over the column, stable variance estimate.
//
// Layout: activity is [n_cells × n_pathways] row-major.
// Column p: activity[cell * n_pathways + p] for cell = 0..n_cells-1.

__global__ __launch_bounds__(256, 2)
void pathway_normalize_kernel(
    float* __restrict__ activity,   // [n_cells × n_pathways] row-major; modified in-place
    int    n_cells,
    int    n_pathways,
    float  eps)
{
    namespace cg = cooperative_groups;
    const int p = blockIdx.x;  // pathway index
    if (p >= n_pathways) return;

    // Shared memory: Welford mean + M2 accumulators for each warp, then cross-warp reduce.
    extern __shared__ float smem_wf[];
    float* s_mean = smem_wf;
    float* s_m2   = smem_wf + (blockDim.x / 32 + 1);

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int n_warps = (blockDim.x + 31) / 32;

    // Each thread accumulates a partial Welford (count, mean, M2) over its strided cells.
    float w_count = 0.f;
    float w_mean  = 0.f;
    float w_m2    = 0.f;

    for (int c = (int)threadIdx.x; c < n_cells; c += (int)blockDim.x) {
        float x = activity[(size_t)c * n_pathways + p];
        w_count += 1.f;
        float delta  = x - w_mean;
        w_mean      += delta / w_count;
        float delta2 = x - w_mean;
        w_m2        += delta * delta2;
    }

    // Combine partial Welford accumulators across warp via shuffle.
    // Welford parallel merge: see Chan et al. (1979).
    for (int mask = 16; mask > 0; mask >>= 1) {
        float b_cnt  = __shfl_xor_sync(0xFFFFFFFF, w_count, mask);
        float b_mean = __shfl_xor_sync(0xFFFFFFFF, w_mean,  mask);
        float b_m2   = __shfl_xor_sync(0xFFFFFFFF, w_m2,    mask);

        // Merge (w_count, w_mean, w_m2) with (b_cnt, b_mean, b_m2).
        float combined_count = w_count + b_cnt;
        if (combined_count > 0.f) {
            float delta = b_mean - w_mean;
            w_mean  = (w_count * w_mean + b_cnt * b_mean) / combined_count;
            w_m2   += b_m2 + delta * delta * w_count * b_cnt / combined_count;
        }
        w_count = combined_count;
    }

    // Write warp results to shared memory (lane 0 per warp).
    if (lane == 0) {
        s_mean[warp_id] = w_mean;
        s_m2[warp_id]   = w_m2;
        smem_wf[2 * (n_warps + 1) + warp_id] = w_count;  // store counts separately
    }
    __syncthreads();

    // Thread 0 merges across warps serially (n_warps ≤ 8 for 256 threads).
    if (threadIdx.x == 0) {
        float* s_cnt = smem_wf + 2 * (n_warps + 1);
        float g_cnt  = s_cnt[0];
        float g_mean = s_mean[0];
        float g_m2   = s_m2[0];
        for (int w = 1; w < n_warps; ++w) {
            float b_cnt  = s_cnt[w];
            float b_mean = s_mean[w];
            float b_m2   = s_m2[w];
            float combined = g_cnt + b_cnt;
            if (combined > 0.f) {
                float delta = b_mean - g_mean;
                g_mean = (g_cnt * g_mean + b_cnt * b_mean) / combined;
                g_m2  += b_m2 + delta * delta * g_cnt * b_cnt / combined;
            }
            g_cnt = combined;
        }
        s_mean[0] = g_mean;
        // Store std in s_m2[0].
        s_m2[0] = (g_cnt > 1.f) ? sqrtf(g_m2 / (g_cnt - 1.f) + eps) : eps;
    }
    __syncthreads();

    const float g_mean = s_mean[0];
    const float g_std  = s_m2[0];

    // Normalize each cell's activity for pathway p.
    for (int c = (int)threadIdx.x; c < n_cells; c += (int)blockDim.x) {
        float x = activity[(size_t)c * n_pathways + p];
        activity[(size_t)c * n_pathways + p] = (x - g_mean) / g_std;
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// Public entry point: progeny
// ---------------------------------------------------------------------------
//
// Computes per-cell pathway activity scores using the PROGENy model.
//
// Parameters:
//   mat      : DeviceCSC (genes × cells), from pz_device_loader.
//              Treated as CSR (transposed) for cuSPARSE SpMM.
//   pw       : PROGENyWeights with uploaded GPU weights (call upload_progeny_weights first).
//   cfg      : PROGENyConfig.
//   stream   : caller-provided CUDA stream.
//   cusparse : cuSPARSE handle (from core/handles.h).
//
// Returns PROGENyResult with activity[n_cells × n_pathways] row-major on device.
// Caller synchronizes stream before reading.

inline PROGENyResult progeny(
    const singlet_gpu::core::DeviceCSC& mat,
    PROGENyWeights&                     pw,
    const PROGENyConfig&                cfg,
    cudaStream_t                        stream,
    cusparseHandle_t                    cusparse)
{
    const int n_genes    = (int)mat.rows;
    const int n_cells    = (int)mat.cols;
    const int n_pathways = pw.n_pathways;

    if (!pw.gpu_ready)
        throw std::runtime_error("PROGENy: call upload_progeny_weights() before progeny().");
    if (n_genes != pw.n_genes)
        throw std::runtime_error("PROGENy: expression n_genes != weight matrix n_genes.");

    PROGENyResult result;
    result.n_cells      = n_cells;
    result.n_pathways   = n_pathways;
    result.pathway_names = pw.pathway_names;

    // Allocate activity output: [n_cells × n_pathways] row-major.
    result.activity = singlet_gpu::core::DeviceMemory<float>((size_t)n_cells * n_pathways);
    cudaMemsetAsync(result.activity.get(), 0, (size_t)n_cells * n_pathways * sizeof(float), stream);

    // ------------------------------------------------------------------
    // Step 1: SpMM — activity = expressionᵀ × weights
    //
    // expression: genes × cells (CSC stored as mat.col_ptrs[n_cells+1], mat.row_idx[nnz], mat.values[nnz]).
    // weights:    genes × pathways (dense, column-major, stored as pw.d_weights).
    // output:     cells × pathways (dense, row-major, stored as result.activity).
    //
    // In BLAS notation:
    //   C = alpha × op(A) × B + beta × C
    //   where op(A) = Aᵀ (transpose) since A is genes×cells, we want cells×genes as left operand.
    //
    // cuSPARSE SpMM with:
    //   A = expression CSC (genes × cells), operation = CUSPARSE_OPERATION_TRANSPOSE (→ cells × genes)
    //   B = weights (genes × pathways), operation = CUSPARSE_OPERATION_NON_TRANSPOSE
    //   C = activity (cells × pathways), dense row-major
    //
    // The expression CSC (col_ptrs[n_cells+1], row_idx[nnz], values[nnz]) has:
    //   rows = n_genes, cols = n_cells.
    // When we apply TRANSPOSE, cuSPARSE treats it as (n_cells × n_genes) × (n_genes × n_pathways)
    // = (n_cells × n_pathways). This is what we want.
    //
    // weights is dense [n_genes × n_pathways] column-major. cuSPARSE SpMM accepts a
    // cusparseDnMatDescr_t for the dense B matrix.
    //
    // WHY SpMM over GEMM: expression is sparse (~5% nonzero for 10x scRNA); SpMM skips
    // zero operations. For n_cells=100k, n_genes=20k, nnz=1e9: GEMM = 2.8T ops vs SpMM ≈ 140G ops.

    // Set cuSPARSE stream.
    cusparseSetStream(cusparse, stream);

    // Create sparse matrix descriptor for expression CSC (genes × cells).
    cusparseSpMatDescr_t sp_expr;
    cusparseCreateCsc(
        &sp_expr,
        (int64_t)n_genes, (int64_t)n_cells, (int64_t)mat.nnz,
        mat.col_ptr.get(),     // column pointers [n_cells + 1]
        mat.row_indices.get(), // row indices [nnz]
        mat.values.get(),      // values [nnz]
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // Create dense matrix descriptor for weights (n_genes × n_pathways), column-major.
    cusparseDnMatDescr_t dn_weights;
    cusparseCreateDnMat(
        &dn_weights,
        (int64_t)n_genes, (int64_t)n_pathways, (int64_t)n_genes,  // rows, cols, leading dim
        pw.d_weights.get(),
        CUDA_R_32F, CUSPARSE_ORDER_COL);  // column-major

    // Create dense matrix descriptor for activity output (n_cells × n_pathways), row-major.
    cusparseDnMatDescr_t dn_activity;
    cusparseCreateDnMat(
        &dn_activity,
        (int64_t)n_cells, (int64_t)n_pathways, (int64_t)n_pathways,  // rows, cols, leading dim (row-major ld = n_pathways)
        result.activity.get(),
        CUDA_R_32F, CUSPARSE_ORDER_ROW);  // row-major

    // Query SpMM buffer size.
    size_t spmm_bytes = 0;
    float alpha_f = 1.f, beta_f = 0.f;
    cusparseSpMM_bufferSize(
        cusparse,
        CUSPARSE_OPERATION_TRANSPOSE,       // op(A) = Aᵀ: genes×cells → cells×genes
        CUSPARSE_OPERATION_NON_TRANSPOSE,   // op(B) = B:  genes×pathways
        &alpha_f, sp_expr, dn_weights,
        &beta_f, dn_activity,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        &spmm_bytes);

    singlet_gpu::core::DeviceMemory<uint8_t> d_spmm_buf(spmm_bytes + 1);

    // Execute SpMM: activity = expressionᵀ × weights.
    cusparseSpMM(
        cusparse,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_f, sp_expr, dn_weights,
        &beta_f, dn_activity,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        d_spmm_buf.get());

    // Destroy cuSPARSE descriptors (lightweight; no GPU memory owned by them).
    cusparseDestroySpMat(sp_expr);
    cusparseDestroyDnMat(dn_weights);
    cusparseDestroyDnMat(dn_activity);

    // ------------------------------------------------------------------
    // Step 2: per-pathway Welford normalization (optional)
    //
    // WHY Welford: single-pass, numerically stable, no catastrophic cancellation.
    // Two-pass (mean then variance) would require two full column traversals —
    // Welford gives equivalent numerics in one pass, lower memory bandwidth.

    if (cfg.normalize && n_cells > 1) {
        // One block per pathway (n_pathways = 14; trivially parallel).
        // 256 threads; dynamic smem = 3 × (n_warps + 1) floats = 3 × 9 × 4 = 108 bytes.
        const int threads   = 256;
        const int n_warps   = threads / 32;
        const size_t smem_bytes = 3 * (n_warps + 1) * sizeof(float);

        detail::pathway_normalize_kernel<<<n_pathways, threads, smem_bytes, stream>>>(
            result.activity.get(), n_cells, n_pathways, cfg.norm_eps);
    }

    return result;
}

// ---------------------------------------------------------------------------
// Convenience: load + upload PROGENy human top-100 weights
// ---------------------------------------------------------------------------
//
// progeny_load_human:
//   Loads the pre-trained human PROGENy weights (14 pathways × top-100 genes).
//   tsv_path: path to tests/refs/progeny_human_top100.tsv (shipped with singlet-gpu).
//   gene_names: ordered gene names from the expression matrix's row names.
//   stream: CUDA stream for the upload.
//
// Usage:
//   auto pw = progeny_load_human("tests/refs/progeny_human_top100.tsv", gene_names, stream);
//   auto result = progeny(mat, pw, cfg, stream, cusparse_handle);

inline PROGENyWeights progeny_load_human(
    const std::string&              tsv_path,
    const std::vector<std::string>& gene_names,
    cudaStream_t                    stream)
{
    std::unordered_map<std::string, int> gene_name_to_idx;
    for (int i = 0; i < (int)gene_names.size(); ++i)
        gene_name_to_idx[gene_names[i]] = i;

    PROGENyWeights pw = load_progeny_weights_tsv(tsv_path, gene_names, gene_name_to_idx);
    upload_progeny_weights(pw, stream);
    return pw;
}

} // namespace enrich
} // namespace singlet_gpu
