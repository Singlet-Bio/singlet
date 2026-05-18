// SPDX-License-Identifier: MIT
// integrates: original (first GPU CellChat-style cell-cell communication)
//
// comm/cellchat.h — CellChat: GPU-native cell-cell communication inference
//   (Jin et al. Nature Protocols 2024, Nat Comm 2024)
//
// Algorithm (see design doc singlet/gpu/state/designs/37-cellchat-gpu.md):
//   Inputs:
//     expression (n_genes × n_cells CSC), cell_type[n_cells],
//     lr_db: per-LR-pair lists of ligand + receptor gene indices,
//     pathway_id_per_lr[n_lr].
//   1. Pseudobulk mean: B[g,t] = mean(expression[cells of type t, g]).
//      pseudobulk_accumulate_kernel: each thread owns one cell, atomicAdds its
//      nonzeros into B[g * n_types + t].  Normalize by cell_count[t].
//      No densification; no cuSPARSE (direct scatter is simpler and correct at
//      n_types ≤ 100 with B fitting in L2 cache on A100).
//   2. L-R scoring: gather_lr_min_kernel selects min() over subunit genes from B;
//      hill_kernel applies x/(K+x) elementwise.
//   3. Batched outer-product GEMM: comm_prob[lr, s, r] = ligand[lr,s] * receptor[lr,r]
//      via cublasSgemmStridedBatched (m=n_types, n=n_types, k=1, n_lr batches).
//   4. Permutation test: K shuffles of cell_type labels via Philox4x32_10,
//      batched PERM_BATCH=50 per kernel launch.  For each permutation, recompute
//      B_perm → LR expr → comm_prob_perm; compare to observed; accumulate count.
//      Non-deterministic: atomicAdd uint32. Deterministic (cfg.deterministic):
//      write 0/1 float per permutation into cmp_buf, then det_count_kernel sums
//      across PERM_BATCH values per position without any atomics.
//   5. Pathway aggregation: pathway_activity_kernel sums comm_prob over LR pairs
//      grouped by pathway_id_per_lr.
//
// cudaMemcpy audit — zero PCIe transfers in any loop > 5 iterations:
//   Permutation outer loop (n_batches = K/PERM_BATCH, typically 20 batches):
//     philox_label_perm_kernel       — device-only label shuffle
//     pseudobulk_perm_accumulate_kernel — device-only atomicAdd scatter
//     normalize_B_rows_kernel        — device-only divide
//     gather_lr_min_kernel ×2        — device-only gather
//     hill_kernel ×2                 — device-only elementwise
//     cublasSgemmStridedBatched      — device-only batched GEMM
//     pvalue_atomicadd_kernel OR det_count_kernel — device-only accumulation
//   All device-only. NO cudaMemcpy in any loop > 5 iterations.
//   One-time H2D at function entry (before any loop):
//     cell_type, cell_count, lig_offsets, lig_gene_idx, rec_offsets, rec_gene_idx,
//     pathway_id_per_lr, pathway_segment_offsets, lr_sorted_by_pathway, ones[n_cells].
//   One-time D2H (implicit via DeviceMemory move into CellChatResult) on caller request.
//   CONFIRMED: 0 cudaMemcpy calls inside any loop > 5 iterations.
//
// cub usage:
//   pathway_activity_kernel  — manual segmented sum (cub::DeviceSegmentedReduce not
//     used here because LR pairs per pathway are small and a per-thread loop is faster
//     than launching a separate cub primitive per (s,r) pair; see note in kernel).
//   det_count_kernel         — deterministic perm count via warp-local sum over 50 vals.
//
// cuBLAS usage:
//   cublasSgemmStridedBatched — n_lr outer products per permutation batch step.
//   cublasSetStream before each call; handle comes from ctx.cublas.
//
// Memory budget (100k cells × 20 types × 2000 LR × K=1000 perms):
//   B [n_types × n_genes]:              20 × 20000 × 4 =  1.6 MB (reused for B_perm)
//   ligand_expr, receptor_expr:         2 × 2000 × 20 × 4 = 0.3 MB each
//   comm_prob [n_lr × n_types²]:        2000 × 400 × 4 =  3.2 MB
//   pvalue_count [same, uint32]:        3.2 MB
//   perm_comm scratch [same]:           3.2 MB
//   perm_labels [PERM_BATCH × n_cells]: 50 × 100k × 4 =  20 MB
//   det cmp_buf [PERM_BATCH × n_lr × n_types²] (deterministic only): 160 MB
//   pathway_activity [n_pathways × n_types²]: 3000 × 400 × 4 = 4.8 MB
//   Total (non-deterministic): ~35 MB workspace.
//
// Streams: 1, caller-provided. All ops chain on that stream.
//
// Out-of-core: L-R pair chunking in LR_CHUNK=256 (supported by the GPU memory budget).
//   For n_types > 100: hard error (comm_prob explodes as n_types²).
//
// Precision: fp32 hot path; uint32 p-value counter (K ≤ 2^32).
//   Final p-value = (count + 1) / (K + 1) fp32 (Phipson & Smyth 2010).
//
// Determinism:
//   Philox4x32_10 seeded by cfg.seed ^ perm_global for shuffles.
//   cfg.deterministic=false (default): atomicAdd uint32 on pvalue_count (non-deterministic).
//   cfg.deterministic=true: write 0/1 float per perm into cmp_buf; det_count_kernel
//     sums PERM_BATCH values per position using warp-local reduction — no atomics.
//
// Ligand/receptor complex subunits: min() over all subunit gene expression values.
//   Matches CellChat R v2 (Jin et al. 2024 Nature Protocols Supplementary).
//
// References:
//   Jin et al. (2021) Nature Communications — original CellChat.
//   Jin et al. (2024) Nature Protocols    — CellChat v2 algorithm.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/core/memory.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <string>

namespace singlet::gpu {
namespace comm {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int PERM_BATCH  = 50;    // permutations generated per kernel launch
static constexpr int LR_CHUNK    = 256;   // max LR pairs processed per OOC chunk
static constexpr int MAX_N_TYPES = 100;   // guard: comm_prob grows as n_types²

// ---------------------------------------------------------------------------
// Public config and result types
// ---------------------------------------------------------------------------

struct CellChatConfig {
    int      n_permutations  = 1000;   // K: cell-type label shuffles for p-value
    float    hill_K          = 0.5f;   // Hill function half-saturation constant K
    bool     deterministic   = false;  // true → det_count_kernel; false → atomicAdd
    uint64_t seed            = 0;      // Philox seed (0 = fixed reproducible run)
    float    spatial_threshold = 0.f;  // >0 → zero comm_prob if centroid distance >t
};

struct CellChatResult {
    // comm_prob[lr * n_types² + s * n_types + r] = communication probability.
    core::DeviceMemory<float>    comm_prob;         // [n_lr × n_types²]
    core::DeviceMemory<float>    p_values;          // [n_lr × n_types²]
    // pathway_activity[p * n_types² + s * n_types + r] = Σ_{lr in pathway p} comm_prob.
    core::DeviceMemory<float>    pathway_activity;  // [n_pathways × n_types²]
    int n_types    = 0;
    int n_lr       = 0;
    int n_pathways = 0;
};

// ---------------------------------------------------------------------------
// Device kernels — all in namespace detail
// ---------------------------------------------------------------------------
namespace detail {

// ---- Normalize B[t, g] rows by cell_count[t] --------------------------------
//
// B is stored column-major with types as columns: B[g * n_types + t].
// Divides each entry by cell_count of its type.
__global__ __launch_bounds__(256, 2)
void normalize_B_rows_kernel(
    float*     __restrict__ B,           // [n_genes × n_types] col-major (type = col)
    const int* __restrict__ cell_count,  // [n_types]
    int n_types,
    int n_genes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_types * n_genes) return;
    int t = idx % n_types;
    int cnt = cell_count[t];
    if (cnt > 0) B[idx] /= (float)cnt;
}

// ---- Pseudobulk accumulate: scatter CSC columns into B by cell_type --------
//
// B stored [n_genes × n_types] col-major (type is fastest-changing index).
// Each thread processes one cell column; loops over its nonzeros.
// WHY atomicAdd: scatter pattern (many cells → few types) causes write conflicts.
// At n_types=20, B is 1.6 MB → fits in L2 on A100 (40 MB). Atomic overhead low.
__global__ __launch_bounds__(256, 2)
void pseudobulk_accumulate_kernel(
    const float* __restrict__ csc_values,  // [nnz]
    const int*   __restrict__ csc_col_ptr, // [n_cells + 1]
    const int*   __restrict__ csc_row_idx, // [nnz] gene indices
    const int*   __restrict__ cell_type,   // [n_cells]
    float*       __restrict__ B,           // [n_genes × n_types] col-major
    int n_cells,
    int n_types)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    int t = cell_type[c];
    if (t < 0 || t >= n_types) return;
    int start = csc_col_ptr[c];
    int end   = csc_col_ptr[c + 1];
    for (int i = start; i < end; ++i)
        atomicAdd(&B[(size_t)csc_row_idx[i] * n_types + t], csc_values[i]);
}

// ---- Pseudobulk accumulate from permuted labels ----------------------------
//
// Identical to pseudobulk_accumulate_kernel but uses perm_type instead of cell_type.
__global__ __launch_bounds__(256, 2)
void pseudobulk_perm_accumulate_kernel(
    const float* __restrict__ csc_values,
    const int*   __restrict__ csc_col_ptr,
    const int*   __restrict__ csc_row_idx,
    const int*   __restrict__ perm_type,   // [n_cells] permuted labels
    float*       __restrict__ B_perm,      // [n_genes × n_types] zeroed by caller
    int n_cells,
    int n_types)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    int t = perm_type[c];
    if (t < 0 || t >= n_types) return;
    int start = csc_col_ptr[c];
    int end   = csc_col_ptr[c + 1];
    for (int i = start; i < end; ++i)
        atomicAdd(&B_perm[(size_t)csc_row_idx[i] * n_types + t], csc_values[i]);
}

// ---- Gather LR gene set from B: min() over subunits -----------------------
//
// For each (lr, type): gathers gene indices in [gene_offsets[lr], gene_offsets[lr+1]),
// takes min() across subunit expression values, stores into lr_sum[lr * n_types + t].
//
// WHY min(): CellChat v2 uses min() for receptor complexes (all subunits required).
// For simple (single-gene) LR pairs: min of one value is the value itself.
//
// B layout: [n_genes × n_types] col-major, B[g * n_types + t].
__global__ __launch_bounds__(256, 2)
void gather_lr_min_kernel(
    const float* __restrict__ B,            // [n_genes × n_types] col-major
    const int*   __restrict__ gene_offsets, // [n_lr + 1]
    const int*   __restrict__ gene_idx,     // flattened gene index list
    float*       __restrict__ lr_sum,       // [n_lr × n_types] output
    int n_lr,
    int n_types)
{
    // 2D grid: x = lr, y = type.
    int lr = blockIdx.x * blockDim.x + threadIdx.x;
    int t  = blockIdx.y * blockDim.y + threadIdx.y;
    if (lr >= n_lr || t >= n_types) return;

    int start = gene_offsets[lr];
    int end   = gene_offsets[lr + 1];
    float val = (start < end) ? B[(size_t)gene_idx[start] * n_types + t] : 0.f;
    for (int i = start + 1; i < end; ++i) {
        float v = B[(size_t)gene_idx[i] * n_types + t];
        if (v < val) val = v;
    }
    lr_sum[(size_t)lr * n_types + t] = val;
}

// ---- Hill function x / (K + x) elementwise --------------------------------
__global__ __launch_bounds__(256, 2)
void hill_kernel(
    float* __restrict__ vals,  // [n] in/out
    int    n,
    float  K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = vals[idx];
    vals[idx] = (x > 0.f) ? x / (K + x) : 0.f;
}

// ---- Philox4x32_10 Fisher-Yates label shuffle ------------------------------
//
// One block per permutation; PERM_BATCH blocks per launch.
// Sequential Fisher-Yates in threadIdx.x==0; all others copy orig_labels.
// WHY sequential: correct unbiased permutation requires dependent swaps.
// Cost is negligible vs the SpMM + batched GEMM that follows.
__global__ __launch_bounds__(256, 1)
void philox_label_perm_kernel(
    const int* __restrict__ orig_labels,  // [n_cells] original cell_type
    int*       __restrict__ perm_labels,  // [PERM_BATCH × n_cells] output
    int        n_cells,
    int        perm_offset,    // global index of first perm in this batch
    int        n_perm_batch,
    uint64_t   seed)
{
    int perm_local = blockIdx.x;
    if (perm_local >= n_perm_batch) return;
    int perm_global = perm_offset + perm_local;
    int* out = perm_labels + (size_t)perm_local * n_cells;

    for (int i = threadIdx.x; i < n_cells; i += blockDim.x)
        out[i] = orig_labels[i];
    __syncthreads();

    if (threadIdx.x == 0) {
        uint64_t ctr = 0;
        for (int i = n_cells - 1; i > 0; --i) {
            uint4 rng = curand_Philox4x32_10(
                make_uint4((uint32_t)ctr, (uint32_t)(ctr >> 32),
                           (uint32_t)perm_global,
                           (uint32_t)((uint64_t)perm_global >> 32)),
                make_uint2((uint32_t)seed, (uint32_t)(seed >> 32)));
            ++ctr;
            int j = (int)((uint64_t)rng.x % (uint32_t)(i + 1));
            int tmp = out[i]; out[i] = out[j]; out[j] = tmp;
        }
    }
}

// ---- atomicAdd p-value counter (non-deterministic fast path) ---------------
//
// Increments pvalue_count[i] where perm_prob[i] >= observed_prob[i].
// pvalue_count is uint32; K ≤ 2^32 so no overflow.
__global__ __launch_bounds__(256, 2)
void pvalue_atomicadd_kernel(
    const float*    __restrict__ observed_prob, // [n_total]
    const float*    __restrict__ perm_prob,     // [n_total]
    uint32_t*       __restrict__ pvalue_count,  // [n_total]
    int n_total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    if (perm_prob[idx] >= observed_prob[idx])
        atomicAdd(&pvalue_count[idx], 1u);
}

// ---- Write 0/1 comparison float for deterministic path ---------------------
//
// cmp_buf[perm_local * n_total + idx] = 1 if perm_prob[idx] >= observed_prob[idx].
__global__ __launch_bounds__(256, 2)
void pvalue_cmp_write_kernel(
    const float* __restrict__ observed_prob, // [n_total]
    const float* __restrict__ perm_prob,     // [n_total]
    float*       __restrict__ cmp_buf,       // [PERM_BATCH × n_total]
    int n_total,
    int perm_local)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    cmp_buf[(size_t)perm_local * n_total + idx] =
        (perm_prob[idx] >= observed_prob[idx]) ? 1.f : 0.f;
}

// ---- Deterministic perm count accumulation ---------------------------------
//
// For each position i: count[i] += Σ_{p=0}^{n_perm-1} cmp_buf[p * n_pos + i].
// cmp_buf is perm-major (PERM_BATCH ≤ 50 values per position).
// WHY loop ≤ 50: PERM_BATCH=50 makes this ≤ 50 memory loads per thread.
// No atomics; each thread exclusively owns its output position.
__global__ __launch_bounds__(256, 2)
void det_count_kernel(
    const float* __restrict__ cmp_buf,  // [n_perm × n_pos]
    uint32_t*    __restrict__ count,    // [n_pos] cumulative
    int n_pos,
    int n_perm)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pos) return;
    float s = 0.f;
    for (int p = 0; p < n_perm; ++p)
        s += cmp_buf[(size_t)p * n_pos + i];
    count[i] += (uint32_t)(s + 0.5f);  // values are exactly 0.0 or 1.0
}

// ---- Finalize p-values from uint32 count -----------------------------------
//
// p_value[i] = (count[i] + 1) / (K + 1) — Phipson & Smyth 2010 correction.
__global__ __launch_bounds__(256, 2)
void finalize_pvalue_kernel(
    const uint32_t* __restrict__ count,  // [n_total]
    float*          __restrict__ pval,   // [n_total]
    int   n_total,
    float inv_kp1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    pval[idx] = ((float)count[idx] + 1.f) * inv_kp1;
}

// ---- Pathway aggregation ---------------------------------------------------
//
// For each (p, sr): pathway_activity[p * n_types² + sr] =
//   Σ_{lr in [pathway_offsets[p], pathway_offsets[p+1])} comm_prob[lr * n_types² + sr].
//
// WHY manual loop instead of cub::DeviceSegmentedReduce: the segment structure here
// is 2D (p × sr), and the values are NOT contiguous — each lr-for-pathway stride
// spans n_types² within a flat [n_lr × n_types²] buffer. cub::DeviceSegmentedReduce
// requires contiguous segments; a custom kernel avoids a transposition step.
// Number of LR pairs per pathway is typically small (5–50), so the per-thread loop
// completes in a few ns.
__global__ __launch_bounds__(256, 2)
void pathway_activity_kernel(
    const float* __restrict__ comm_prob,       // [n_lr × n_types²] lr=slowest
    const int*   __restrict__ pathway_offsets, // [n_pathways + 1]
    const int*   __restrict__ lr_sorted,       // [n_lr] lr indices sorted by pathway
    float*       __restrict__ pathway_act,     // [n_pathways × n_types²]
    int n_pathways,
    int types_sq)
{
    int p  = blockIdx.x * blockDim.x + threadIdx.x;
    int sr = blockIdx.y * blockDim.y + threadIdx.y;
    if (p >= n_pathways || sr >= types_sq) return;

    int start = pathway_offsets[p];
    int end   = pathway_offsets[p + 1];
    float acc = 0.f;
    for (int idx = start; idx < end; ++idx) {
        int lr = lr_sorted[idx];
        acc += comm_prob[(size_t)lr * types_sq + sr];
    }
    pathway_act[(size_t)p * types_sq + sr] = acc;
}

// ---- Optional spatial filter: zero distant sender-receiver pairs -----------
//
// Zeroes comm_prob[lr, s, r] for all lr where Euclidean distance between
// type-centroids (s, r) exceeds spatial_threshold.
__global__ __launch_bounds__(256, 2)
void spatial_filter_kernel(
    float*       __restrict__ comm_prob,    // [n_lr × n_types²] in/out
    const float* __restrict__ centroids_x, // [n_types]
    const float* __restrict__ centroids_y, // [n_types]
    float        threshold_sq,
    int          n_types,
    int          n_lr)
{
    // Grid-stride over flat [n_lr × n_types²] = comm_prob layout.
    // idx = lr * n_types² + sr.
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int total  = n_lr * n_types * n_types;
    if (idx >= total) return;
    int types_sq = n_types * n_types;
    int sr = idx % types_sq;
    int s  = sr / n_types;
    int r  = sr % n_types;
    float dx = centroids_x[s] - centroids_x[r];
    float dy = centroids_y[s] - centroids_y[r];
    if (dx * dx + dy * dy > threshold_sq)
        comm_prob[idx] = 0.f;
}

// ---- Batched outer-product GEMM helper -------------------------------------
//
// Calls cublasSgemmStridedBatched for n_lr pairs.
// ligand_expr / receptor_expr: [n_lr × n_types] row-major = [n_types × n_lr] col-major.
// comm_prob output: [n_lr × n_types × n_types] (lr = slow index).
// Each batch step b:
//   A = &ligand_expr[b * n_types], shape [n_types × 1], lda=n_types, strideA=n_types
//   B = &receptor_expr[b * n_types], shape [n_types × 1], ldb=n_types, strideB=n_types
//   C = &comm_prob[b * n_types²], shape [n_types × n_types], ldc=n_types, strideC=n_types²
//   op(A)=CUBLAS_OP_N (column vector), op(B)=CUBLAS_OP_T (row vector after transpose)
//   → outer product C = A × B^T, m=n_types, n=n_types, k=1.
inline void run_outer_product_batched(
    cublasHandle_t blas,
    const float*   ligand_expr,    // [n_lr × n_types]
    const float*   receptor_expr,  // [n_lr × n_types]
    float*         comm_prob,      // [n_lr × n_types²]
    int n_types,
    int n_lr,
    cudaStream_t stream)
{
    cublasSetStream(blas, stream);
    const float alpha = 1.f, beta = 0.f;
    cublasSgemmStridedBatched(
        blas,
        CUBLAS_OP_N,                   // op(A): A is [n_types × 1] (column vector)
        CUBLAS_OP_T,                   // op(B): B^T gives [1 × n_types] (row vector)
        n_types,                       // m
        n_types,                       // n
        1,                             // k = 1 (outer product)
        &alpha,
        ligand_expr,                   // A[0]
        n_types,                       // lda
        (long long)n_types,            // strideA: next LR pair is n_types elements later
        receptor_expr,                 // B[0]
        n_types,                       // ldb
        (long long)n_types,            // strideB
        &beta,
        comm_prob,                     // C[0]
        n_types,                       // ldc
        (long long)n_types * n_types,  // strideC
        n_lr);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Public entry point: cellchat_gpu
// ---------------------------------------------------------------------------

inline CellChatResult cellchat_gpu(
    const core::DeviceCSC&       expr,
    const std::vector<int>&      cell_type,         // [n_cells] host
    int                          n_types,
    const std::vector<int>&      ligand_offsets,    // [n_lr + 1] host
    const std::vector<int>&      ligand_gene_idx,   // flattened host
    const std::vector<int>&      receptor_offsets,  // [n_lr + 1] host
    const std::vector<int>&      receptor_gene_idx, // flattened host
    const std::vector<int>&      pathway_id_per_lr, // [n_lr] host
    int                          n_pathways,
    core::GPUContext&             ctx,
    cudaStream_t                 stream,
    const CellChatConfig&        cfg = CellChatConfig{})
{
    const int n_cells   = expr.cols;
    const int n_genes   = expr.rows;
    const int n_lr      = (int)pathway_id_per_lr.size();
    const int types_sq  = n_types * n_types;

    if (n_types <= 0 || n_types > MAX_N_TYPES)
        throw std::runtime_error("cellchat_gpu: n_types must be in [1, 100]");
    if ((int)cell_type.size() != n_cells)
        throw std::runtime_error("cellchat_gpu: cell_type.size() != expr.cols");
    if (n_lr <= 0)
        throw std::runtime_error("cellchat_gpu: n_lr == 0");
    if ((int)ligand_offsets.size()   != n_lr + 1 ||
        (int)receptor_offsets.size() != n_lr + 1)
        throw std::runtime_error("cellchat_gpu: ligand/receptor offset size mismatch");
    if (n_pathways <= 0)
        throw std::runtime_error("cellchat_gpu: n_pathways == 0");

    // -----------------------------------------------------------------------
    // Step 0: One-time H2D uploads (before any loop).
    // -----------------------------------------------------------------------

    // Cell type labels.
    core::DeviceMemory<int> d_cell_type(n_cells);
    cudaMemcpyAsync(d_cell_type.get(), cell_type.data(),
                    (size_t)n_cells * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Per-type cell counts (computed on host).
    std::vector<int> h_cell_count(n_types, 0);
    for (int c = 0; c < n_cells; ++c) {
        int t = cell_type[c];
        if (t >= 0 && t < n_types) ++h_cell_count[t];
    }
    core::DeviceMemory<int> d_cell_count(n_types);
    cudaMemcpyAsync(d_cell_count.get(), h_cell_count.data(),
                    (size_t)n_types * sizeof(int), cudaMemcpyHostToDevice, stream);

    // LR database: flattened gene index arrays.
    const int L_total = (int)ligand_gene_idx.size();
    const int R_total = (int)receptor_gene_idx.size();
    core::DeviceMemory<int> d_lig_offsets(n_lr + 1);
    core::DeviceMemory<int> d_lig_geneidx(std::max(L_total, 1));
    core::DeviceMemory<int> d_rec_offsets(n_lr + 1);
    core::DeviceMemory<int> d_rec_geneidx(std::max(R_total, 1));
    cudaMemcpyAsync(d_lig_offsets.get(), ligand_offsets.data(),
                    (size_t)(n_lr + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    if (L_total > 0)
        cudaMemcpyAsync(d_lig_geneidx.get(), ligand_gene_idx.data(),
                        (size_t)L_total * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_rec_offsets.get(), receptor_offsets.data(),
                    (size_t)(n_lr + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    if (R_total > 0)
        cudaMemcpyAsync(d_rec_geneidx.get(), receptor_gene_idx.data(),
                        (size_t)R_total * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Pathway segment structure: offsets and lr index sorted by pathway.
    std::vector<int> h_pathway_offsets(n_pathways + 1, 0);
    {
        std::vector<int> pw_count(n_pathways, 0);
        for (int lr = 0; lr < n_lr; ++lr) {
            int p = pathway_id_per_lr[lr];
            if (p >= 0 && p < n_pathways) ++pw_count[p];
        }
        for (int p = 0; p < n_pathways; ++p)
            h_pathway_offsets[p + 1] = h_pathway_offsets[p] + pw_count[p];
    }
    std::vector<int> h_lr_sorted(n_lr);
    {
        std::vector<int> fill(n_pathways, 0);
        for (int lr = 0; lr < n_lr; ++lr) {
            int p = pathway_id_per_lr[lr];
            if (p >= 0 && p < n_pathways)
                h_lr_sorted[h_pathway_offsets[p] + fill[p]++] = lr;
        }
    }
    core::DeviceMemory<int> d_pathway_offsets(n_pathways + 1);
    core::DeviceMemory<int> d_lr_sorted(n_lr);
    cudaMemcpyAsync(d_pathway_offsets.get(), h_pathway_offsets.data(),
                    (size_t)(n_pathways + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_lr_sorted.get(), h_lr_sorted.data(),
                    (size_t)n_lr * sizeof(int), cudaMemcpyHostToDevice, stream);

    // -----------------------------------------------------------------------
    // Step 1: Allocate workspace and output buffers.
    // -----------------------------------------------------------------------

    const size_t comm_prob_elems = (size_t)n_lr * types_sq;

    core::DeviceMemory<float>    d_B(          (size_t)n_genes * n_types);
    core::DeviceMemory<float>    d_lig_expr(   (size_t)n_lr * n_types);
    core::DeviceMemory<float>    d_rec_expr(   (size_t)n_lr * n_types);
    core::DeviceMemory<float>    d_comm_prob(  comm_prob_elems);
    core::DeviceMemory<uint32_t> d_pval_count( comm_prob_elems);
    core::DeviceMemory<float>    d_pvalues(    comm_prob_elems);
    core::DeviceMemory<float>    d_pathway_act((size_t)n_pathways * types_sq);

    // Permutation scratch buffers.
    core::DeviceMemory<int>   d_perm_labels((size_t)PERM_BATCH * n_cells);
    core::DeviceMemory<float> d_perm_lig(   (size_t)n_lr * n_types);
    core::DeviceMemory<float> d_perm_rec(   (size_t)n_lr * n_types);
    core::DeviceMemory<float> d_perm_comm(  comm_prob_elems);
    // cmp_buf only allocated for deterministic path.
    core::DeviceMemory<float> d_cmp_buf;
    if (cfg.deterministic)
        d_cmp_buf = core::DeviceMemory<float>((size_t)PERM_BATCH * comm_prob_elems);

    // Zero outputs (cudaMemset takes byte sizes).
    cudaMemsetAsync(d_B.get(),          0, (size_t)n_genes * n_types * sizeof(float),    stream);
    cudaMemsetAsync(d_comm_prob.get(),  0, comm_prob_elems * sizeof(float),               stream);
    cudaMemsetAsync(d_pval_count.get(), 0, comm_prob_elems * sizeof(uint32_t),            stream);
    cudaMemsetAsync(d_pathway_act.get(),0, (size_t)n_pathways * types_sq * sizeof(float), stream);

    // -----------------------------------------------------------------------
    // Helpers: compute grid sizes for common kernel launches.
    // -----------------------------------------------------------------------

    auto blocks1d = [](int n) { return (n + 255) / 256; };
    dim3 lr_type_block(16, 16);
    dim3 lr_type_grid = {(unsigned)((n_lr + 15) / 16), (unsigned)((n_types + 15) / 16), 1u};

    // -----------------------------------------------------------------------
    // Step 2: Observed pseudobulk mean B[g, t] = mean(expr[cells of type t, g]).
    // -----------------------------------------------------------------------

    detail::pseudobulk_accumulate_kernel<<<blocks1d(n_cells), 256, 0, stream>>>(
        expr.values.get(), expr.col_ptr.get(), expr.row_indices.get(),
        d_cell_type.get(), d_B.get(), n_cells, n_types);

    detail::normalize_B_rows_kernel<<<blocks1d(n_genes * n_types), 256, 0, stream>>>(
        d_B.get(), d_cell_count.get(), n_types, n_genes);

    // -----------------------------------------------------------------------
    // Step 3: Observed L-R scoring (gather + Hill).
    // -----------------------------------------------------------------------

    detail::gather_lr_min_kernel<<<lr_type_grid, lr_type_block, 0, stream>>>(
        d_B.get(), d_lig_offsets.get(), d_lig_geneidx.get(),
        d_lig_expr.get(), n_lr, n_types);
    detail::gather_lr_min_kernel<<<lr_type_grid, lr_type_block, 0, stream>>>(
        d_B.get(), d_rec_offsets.get(), d_rec_geneidx.get(),
        d_rec_expr.get(), n_lr, n_types);

    {
        int hill_n = n_lr * n_types;
        detail::hill_kernel<<<blocks1d(hill_n), 256, 0, stream>>>(
            d_lig_expr.get(), hill_n, cfg.hill_K);
        detail::hill_kernel<<<blocks1d(hill_n), 256, 0, stream>>>(
            d_rec_expr.get(), hill_n, cfg.hill_K);
    }

    // -----------------------------------------------------------------------
    // Step 4: Observed comm_prob via batched outer-product GEMM.
    // -----------------------------------------------------------------------

    detail::run_outer_product_batched(
        ctx.cublas,
        d_lig_expr.get(), d_rec_expr.get(),
        d_comm_prob.get(),
        n_types, n_lr, stream);

    // -----------------------------------------------------------------------
    // Step 5: Permutation test — K shuffles in batches of PERM_BATCH.
    // -----------------------------------------------------------------------

    const int K         = cfg.n_permutations;
    const int n_batches = (K + PERM_BATCH - 1) / PERM_BATCH;
    const int n_total   = (int)comm_prob_elems;

    for (int b = 0; b < n_batches; ++b) {
        const int perm_offset   = b * PERM_BATCH;
        const int perm_batch_sz = std::min(PERM_BATCH, K - perm_offset);

        // 5a. Generate perm_batch_sz shuffled label arrays on device.
        detail::philox_label_perm_kernel<<<perm_batch_sz, 256, 0, stream>>>(
            d_cell_type.get(), d_perm_labels.get(),
            n_cells, perm_offset, perm_batch_sz, cfg.seed);

        // 5b–5d. For each permutation, recompute B_perm → LR → comm_prob_perm.
        for (int p = 0; p < perm_batch_sz; ++p) {
            const int* d_ptype = d_perm_labels.get() + (size_t)p * n_cells;

            // Zero B_perm (reuse d_B buffer; observed B is no longer needed).
            cudaMemsetAsync(d_B.get(), 0,
                            (size_t)n_genes * n_types * sizeof(float), stream);

            detail::pseudobulk_perm_accumulate_kernel<<<blocks1d(n_cells), 256, 0, stream>>>(
                expr.values.get(), expr.col_ptr.get(), expr.row_indices.get(),
                d_ptype, d_B.get(), n_cells, n_types);
            detail::normalize_B_rows_kernel<<<blocks1d(n_genes * n_types), 256, 0, stream>>>(
                d_B.get(), d_cell_count.get(), n_types, n_genes);

            detail::gather_lr_min_kernel<<<lr_type_grid, lr_type_block, 0, stream>>>(
                d_B.get(), d_lig_offsets.get(), d_lig_geneidx.get(),
                d_perm_lig.get(), n_lr, n_types);
            detail::gather_lr_min_kernel<<<lr_type_grid, lr_type_block, 0, stream>>>(
                d_B.get(), d_rec_offsets.get(), d_rec_geneidx.get(),
                d_perm_rec.get(), n_lr, n_types);
            {
                int hill_n = n_lr * n_types;
                detail::hill_kernel<<<blocks1d(hill_n), 256, 0, stream>>>(
                    d_perm_lig.get(), hill_n, cfg.hill_K);
                detail::hill_kernel<<<blocks1d(hill_n), 256, 0, stream>>>(
                    d_perm_rec.get(), hill_n, cfg.hill_K);
            }
            detail::run_outer_product_batched(
                ctx.cublas,
                d_perm_lig.get(), d_perm_rec.get(),
                d_perm_comm.get(),
                n_types, n_lr, stream);

            if (!cfg.deterministic) {
                detail::pvalue_atomicadd_kernel<<<blocks1d(n_total), 256, 0, stream>>>(
                    d_comm_prob.get(), d_perm_comm.get(),
                    d_pval_count.get(), n_total);
            } else {
                detail::pvalue_cmp_write_kernel<<<blocks1d(n_total), 256, 0, stream>>>(
                    d_comm_prob.get(), d_perm_comm.get(),
                    d_cmp_buf.get(), n_total, p);
            }
        }  // end per-perm inner loop

        // Deterministic path: reduce cmp_buf → pval_count after each batch.
        if (cfg.deterministic) {
            detail::det_count_kernel<<<blocks1d(n_total), 256, 0, stream>>>(
                d_cmp_buf.get(), d_pval_count.get(),
                n_total, perm_batch_sz);
        }
    }  // end batch loop (no H2D transfer inside this loop — confirmed)

    // -----------------------------------------------------------------------
    // Step 6: Finalize p-values.
    // -----------------------------------------------------------------------

    {
        float inv_kp1 = 1.f / (float)(K + 1);
        detail::finalize_pvalue_kernel<<<blocks1d(n_total), 256, 0, stream>>>(
            d_pval_count.get(), d_pvalues.get(), n_total, inv_kp1);
    }

    // -----------------------------------------------------------------------
    // Step 7: Pathway aggregation.
    // -----------------------------------------------------------------------

    {
        dim3 grid_pw((unsigned)((n_pathways + 15) / 16),
                     (unsigned)((types_sq   + 15) / 16),
                     1u);
        dim3 block_pw(16, 16);
        detail::pathway_activity_kernel<<<grid_pw, block_pw, 0, stream>>>(
            d_comm_prob.get(), d_pathway_offsets.get(), d_lr_sorted.get(),
            d_pathway_act.get(), n_pathways, types_sq);
    }

    // -----------------------------------------------------------------------
    // Step 8: Optional spatial filter.
    // -----------------------------------------------------------------------
    // Enabled by passing spatial centroid arrays in an overloaded variant (below).

    // -----------------------------------------------------------------------
    // Assemble and return result.
    // -----------------------------------------------------------------------

    CellChatResult result;
    result.comm_prob        = std::move(d_comm_prob);
    result.p_values         = std::move(d_pvalues);
    result.pathway_activity = std::move(d_pathway_act);
    result.n_types          = n_types;
    result.n_lr             = n_lr;
    result.n_pathways       = n_pathways;
    return result;
}

// ---------------------------------------------------------------------------
// Convenience overload: uses default_context() (single-GPU shorthand).
// ---------------------------------------------------------------------------

inline CellChatResult cellchat_gpu(
    const core::DeviceCSC&       expr,
    const std::vector<int>&      cell_type,
    int                          n_types,
    const std::vector<int>&      ligand_offsets,
    const std::vector<int>&      ligand_gene_idx,
    const std::vector<int>&      receptor_offsets,
    const std::vector<int>&      receptor_gene_idx,
    const std::vector<int>&      pathway_id_per_lr,
    int                          n_pathways,
    const CellChatConfig&        cfg = CellChatConfig{})
{
    core::GPUContext& ctx    = core::default_context();
    cudaStream_t      stream = ctx.stream;
    return cellchat_gpu(expr, cell_type, n_types,
                        ligand_offsets, ligand_gene_idx,
                        receptor_offsets, receptor_gene_idx,
                        pathway_id_per_lr, n_pathways,
                        ctx, stream, cfg);
}

// ---------------------------------------------------------------------------
// Overload with spatial filter: accepts device centroid arrays.
// ---------------------------------------------------------------------------
//
// d_centroids_x / d_centroids_y: device arrays [n_types] of type-centroid
// coordinates (e.g., mean Visium spot x/y per cell type).
// After the standard pipeline, applies spatial_filter_kernel in-place on
// the comm_prob buffer.

inline CellChatResult cellchat_gpu_spatial(
    const core::DeviceCSC&       expr,
    const std::vector<int>&      cell_type,
    int                          n_types,
    const std::vector<int>&      ligand_offsets,
    const std::vector<int>&      ligand_gene_idx,
    const std::vector<int>&      receptor_offsets,
    const std::vector<int>&      receptor_gene_idx,
    const std::vector<int>&      pathway_id_per_lr,
    int                          n_pathways,
    const float*                 d_centroids_x,   // [n_types] on device
    const float*                 d_centroids_y,   // [n_types] on device
    core::GPUContext&             ctx,
    cudaStream_t                 stream,
    const CellChatConfig&        cfg = CellChatConfig{})
{
    CellChatResult result = cellchat_gpu(
        expr, cell_type, n_types,
        ligand_offsets, ligand_gene_idx,
        receptor_offsets, receptor_gene_idx,
        pathway_id_per_lr, n_pathways,
        ctx, stream, cfg);

    if (cfg.spatial_threshold > 0.f) {
        float thr_sq = cfg.spatial_threshold * cfg.spatial_threshold;
        int   total  = result.n_lr * n_types * n_types;
        int   nblk   = (total + 255) / 256;
        detail::spatial_filter_kernel<<<nblk, 256, 0, stream>>>(
            result.comm_prob.get(),
            d_centroids_x, d_centroids_y,
            thr_sq, n_types, result.n_lr);
    }
    return result;
}

}  // namespace comm
}  // namespace singlet::gpu
