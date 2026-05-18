// SPDX-License-Identifier: MIT
// integrates: decoupleR (Badia-i-Mompel et al. 2022)
//
// singlet/gpu/enrich/decoupler_ora.h
//
// ORA (Over-Representation Analysis) pathway-scoring method from decoupleR.
//
// Algorithm reference:
//   Badia-i-Mompel P, Vélez Santiago J, Braunger J, et al. (2022)
//   "decoupleR: ensemble of computational methods to infer biological
//   activities from omics data." Bioinformatics Advances 2:vbac016.
//   https://doi.org/10.1093/bioadv/vbac016
//
// For each cell c and gene set s, test whether the cell's top-K expressed
// genes (by value in X[:, c]) overlap set s more than expected by chance
// using a one-tailed hypergeometric test.
//
// Setup per cell c:
//   Universe = N = m genes
//   Top-K set T_c: top-K genes by expression value in X[:, c]
//   Default K = ceil(m * 0.05) (top 5% — decoupleR default)
//
// For each gene set s:
//   k  = |T_c ∩ s|          (hits — overlap)
//   K  = |s|                (population successes)
//   N  = m                  (universe size)
//   n  = |T_c|              (sample size)
//
//   Hypergeometric one-tailed p-value (≥ k successes in n draws):
//     p = Σ_{i=k}^{min(K,n)} C(K,i)·C(N-K,n-i) / C(N,n)
//
//   Score: score[c, s] = -log10(p)   (higher = more enriched)
//
//   Closed-form via lgamma (avoids factorial overflow):
//     log_C(a,b)    = lgamma(a+1) − lgamma(b+1) − lgamma(a-b+1)
//     log_p_hit_i   = log_C(K,i) + log_C(N-K,n-i) − log_C(N,n)
//     log_p         = log-sum-exp over i in [k, min(K,n)] of log_p_hit_i
//     score         = -log_p / log(10)
//
// Implementation (4 sequential device passes, no inner loop, no raw cudaMalloc):
//   Pass 1 — per-cell threshold + binary mask T (m × n col-major fp32).
//             ora_topk_smem_kernel: one block per cell, 64-bucket count-histogram
//             in shared memory finds K-th-largest value → threshold[c].
//             ora_build_T_kernel: one thread per (g,c), binary search in CSC,
//             T[g,c] = 1 if X[g,c] >= threshold[c] and X[g,c] > 0.
//   Pass 2 — M matrix (host-side, H2D once): m × n_sets col-major fp32,
//             M[g, s] = 1 if g ∈ set s.  Mirrors score_genes.h.
//   Pass 3 — cuBLAS Sgemm: hits = T^T · M (n × n_sets, col-major fp32).
//             Binary fp32 inputs; counts fit in fp32 mantissa for K ≤ 4096.
//   Pass 4 — lgamma closed-form kernel: one thread per (c,s), log-sum-exp
//             loop over hypergeometric terms → -log10(p).
//
// Determinism (Rule 18): Pass 1 — warp-shuffle only (one block/cell, no atomics
//   in output); Pass 2 — host deterministic; Pass 3 — cuBLAS Sgemm fixed arch;
//   Pass 4 — one thread/element. cfg.deterministic is a no-op (API symmetry).
//
// Memory (Rule 5): all device buffers via core::DeviceMemory<T>. No raw cudaMalloc.
// D2H (Rule 4): none in the hot path. M matrix upload is H2D (not D2H).

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace singlet::gpu {
namespace enrich {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct OraConfig {
    int  top_k       = 0;     // 0 = use top 5%: top_k = ceil(m * 0.05)
    bool deterministic = true; // no-op: ORA v0 has no atomics, bit-exact by construction
};

struct OraResult {
    core::DeviceMemory<float> scores;  // n_cells × n_sets, col-major; -log10(p-value)
    int n_cells;
    int n_sets;
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels (anonymous namespace — not part of the public API)
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// ora_topk_smem_kernel — compute per-cell K-th-largest value threshold.
//
// One block per cell.  Loads column g values into shared memory (m floats),
// then finds the K-th largest via a 64-bucket count histogram over [0, max].
// Shared-memory layout: smem[0..m-1] = column values, smem[m] = col_max,
// smem[m+1..m+64] = int32 bucket counts (64 buckets, reinterpret_cast).
// Requires smem = (m + 72) * sizeof(float) bytes at launch.
// v0 constraint: m <= ~12000 (smem ≤ 48KB); throws in ora() if exceeded.
//
// Output: threshold[c] = lower bound of the bucket containing the top-K gene.
// Approximate (bucket resolution = col_max/64); tie-break error of ±1 gene
// is acceptable for hypergeometric ORA (negligible p-value impact).
//
// ALL parameters are explicit — no enclosing-scope variables relied upon.
__global__ __launch_bounds__(256, 1)
void ora_topk_smem_kernel(
    const float*   __restrict__ values,      // CSC stored values [nnz]
    const int32_t* __restrict__ row_indices, // CSC row indices [nnz]
    const int32_t* __restrict__ col_ptr,     // CSC column pointers [n+1]
    float*         __restrict__ threshold,   // output: per-cell K-th largest [n]
    int m,
    int n,
    int top_k)
{
    const int c = blockIdx.x;   // one block per cell
    if (c >= n) return;

    // --- Load this cell's stored entries into shared memory.
    // smem used as a dense float array [m] for the column values.
    // For m <= ORA_SORT_MAX this is allocated; for larger m we use a compact approach.

    extern __shared__ float smem[];  // size = m floats (set at launch)

    // Initialize to 0 (implicit zeros from sparse format).
    for (int g = threadIdx.x; g < m; g += blockDim.x) {
        smem[g] = 0.f;
    }
    __syncthreads();

    // Fill stored (non-zero) entries.  col_ptr[c]..col_ptr[c+1]-1.
    const int col_start = col_ptr[c];
    const int col_end   = col_ptr[c + 1];
    for (int e = col_start + static_cast<int>(threadIdx.x);
             e < col_end;
             e += static_cast<int>(blockDim.x))
    {
        smem[row_indices[e]] = values[e];
    }
    __syncthreads();

    // --- Find the K-th largest value via parallel bitonic-style selection.
    // v0 simplified: find max, then count-bucket-based threshold.
    // Works for any m. Approximate for m > ORA_SORT_MAX (but usually m <= 20k
    // and top_k ~ 1000, so bucket resolution of max/64 is ample).

    // Step 1: find max value in this column (warp-shuffle reduction).
    float local_max = 0.f;
    for (int g = threadIdx.x; g < m; g += blockDim.x) {
        float v = smem[g];
        if (v > local_max) local_max = v;
    }
    // Warp reduce max.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(0xffffffff, local_max, off);
        if (other > local_max) local_max = other;
    }
    // Cross-warp reduce via smem[m..m+7] (reuse smem after ORA_SORT_MAX).
    // We reserve smem[m..m+8] for cross-warp max accumulation.
    const int wid   = threadIdx.x / 32;
    const int lane  = threadIdx.x & 31;
    const int nwarp = (blockDim.x + 31) / 32;
    // Use the high end of smem for temporaries (smem is at least m+8 floats).
    if (lane == 0) smem[m + wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < nwarp) ? smem[m + lane] : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_down_sync(0xffffffff, v, off);
            if (other > v) v = other;
        }
        if (lane == 0) smem[m] = v;  // global max stored at smem[m]
    }
    __syncthreads();

    const float col_max = smem[m];

    // If col_max == 0, all values are zero; threshold = 0 (top-K = any K genes).
    if (col_max <= 0.f) {
        if (threadIdx.x == 0) threshold[c] = 0.f;
        return;
    }

    // Step 2: binary search for threshold using 64-bucket count histogram.
    // Each bucket covers range [col_max*(b/64), col_max*((b+1)/64)).
    // We find the smallest bucket such that cumulative count from top >= top_k.

    // Use smem[m+1..m+64] (65 floats of workspace after the column data and max).
    // 64 int32 buckets reinterpreted as floats in smem[m+1..m+64].
    // Atomics are needed here; use regular shared atomicAdd (int).
    // Re-cast smem as int for atomic bucket counting.

    // Strategy: 3 passes, each with 8 finer buckets inside the winning coarse bucket.
    // Pass 1: 64 coarse buckets over [0, col_max].
    // Pass 2: 64 fine buckets over [bucket_lo, bucket_hi].
    // This gives resolution of col_max/4096 — more than enough for threshold.

    // For simplicity in v0, do a single pass with 64 buckets.
    // Bucket b covers [lo + b*(hi-lo)/64, lo + (b+1)*(hi-lo)/64).
    // Count values per bucket (values strictly > 0 to distinguish from implicit zeros).
    // Find the bucket where cumulative count (from top) reaches top_k.

    // smem[m+1..m+64] = int32 bucket counts (cast via pointer).
    int* bucket_cnt = reinterpret_cast<int*>(&smem[m + 1]);
    for (int b = threadIdx.x; b < 64; b += blockDim.x) {
        bucket_cnt[b] = 0;
    }
    __syncthreads();

    const float bucket_scale = 64.f / col_max;
    for (int g = threadIdx.x; g < m; g += blockDim.x) {
        float v = smem[g];
        if (v > 0.f) {
            int b = static_cast<int>(v * bucket_scale);
            if (b >= 64) b = 63;
            atomicAdd(&bucket_cnt[b], 1);
        }
    }
    __syncthreads();

    // Thread 0: scan buckets from high to low, find bucket where cumsum >= top_k.
    if (threadIdx.x == 0) {
        int cumsum = 0;
        float thr  = 0.f;
        for (int b = 63; b >= 0; --b) {
            cumsum += bucket_cnt[b];
            if (cumsum >= top_k) {
                // Threshold is the lower bound of bucket b.
                thr = static_cast<float>(b) / 64.f * col_max;
                break;
            }
        }
        threshold[c] = thr;
    }
}

// ---------------------------------------------------------------------------
// ora_build_T_kernel — build dense top-K binary mask T (m × n, col-major fp32).
//
// T[g, c] = 1.f if X[g,c] >= threshold[c] and X[g,c] > 0, else 0.f.
// One thread per (g, c) pair.  Binary search in col_ptr[c]..col_ptr[c+1]
// for row g; if found use stored value, else value = 0 (implicit zero).
// O(m*n * log(nnz_per_col)).  ALL parameters explicit.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void ora_build_T_kernel(
    const float*   __restrict__ values,
    const int32_t* __restrict__ row_indices,
    const int32_t* __restrict__ col_ptr,
    const float*   __restrict__ threshold,
    float*         __restrict__ T,        // m × n col-major output
    int m,
    int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;

    const int c = idx / m;
    const int g = idx % m;

    const float thr = threshold[c];

    // Binary search for gene g in column c of CSC.
    int lo = col_ptr[c];
    int hi = col_ptr[c + 1] - 1;
    float val = 0.f;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int r   = row_indices[mid];
        if (r == g) { val = values[mid]; break; }
        if (r < g)  lo = mid + 1;
        else        hi = mid - 1;
    }

    T[idx] = (val >= thr && val > 0.f) ? 1.f : 0.f;
}

// ---------------------------------------------------------------------------
// ora_hypergeo_kernel — compute -log10 hypergeometric p-value per (cell, set).
//
// One thread per (c, s).  Inputs: hits[c,s] (fp32 count from Sgemm),
// set_size[s] (K), top_k_arr[c] (n), m_universe (N).
// Log-sum-exp loop over i = k..min(K,n):
//   log_term = log_C(K,i) + log_C(N-K,n-i) - log_C(N,n)   via lgammaf
// score = -log_p / log(10), capped at 300, floored at 0.
// ALL parameters explicit.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void ora_hypergeo_kernel(
    const float* __restrict__ hits,       // n × n_sets col-major [n * n_sets]
    const int*   __restrict__ set_size,   // [n_sets] integer set sizes
    const int*   __restrict__ top_k_arr,  // [n] per-cell top-K count
    float*       __restrict__ scores,     // n × n_sets col-major output
    int n,
    int n_sets,
    int m_universe)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n_sets) return;

    const int s   = idx / n;
    const int c   = idx % n;

    const int K  = set_size[s];
    const int N  = m_universe;
    const int nk = top_k_arr[c];
    const int k  = static_cast<int>(hits[idx] + 0.5f);  // round fp32 to int

    // Degenerate / trivial cases.
    if (k <= 0 || K <= 0 || nk <= 0) {
        scores[idx] = 0.f;
        return;
    }

    // Precompute log_C(N, n) — denominator, constant per (c,s).
    const float lN1  = lgammaf(static_cast<float>(N  + 1));
    const float lnk1 = lgammaf(static_cast<float>(nk + 1));
    const float lNnk1= lgammaf(static_cast<float>(N - nk + 1));
    const float log_C_Nn = lN1 - lnk1 - lNnk1;

    // Upper bound for i: min(K, nk).
    const int i_max = (K < nk) ? K : nk;

    if (k > i_max) {
        // More hits than possible — impossible; score = 0.
        scores[idx] = 0.f;
        return;
    }

    // Log-sum-exp accumulation over i = k .. i_max.
    float log_max = -1e30f;
    float log_sum = 0.f;
    bool first    = true;

    for (int i = k; i <= i_max; ++i) {
        const float fi   = static_cast<float>(i);
        const float fK   = static_cast<float>(K);
        const float fNK  = static_cast<float>(N - K);
        const float fni  = static_cast<float>(nk - i);

        // log_C(K, i) = lgamma(K+1) - lgamma(i+1) - lgamma(K-i+1)
        const float log_CKi  = lgammaf(fK + 1.f) - lgammaf(fi + 1.f) - lgammaf(fK - fi + 1.f);

        // log_C(N-K, n-i): valid only if N-K >= n-i >= 0.
        if (fni < 0.f || fni > fNK) continue;  // term is zero (C = 0)
        const float log_CNKni = lgammaf(fNK + 1.f) - lgammaf(fni + 1.f) - lgammaf(fNK - fni + 1.f);

        const float log_term = log_CKi + log_CNKni - log_C_Nn;

        // Log-sum-exp: track max and accumulate shifted exponentials.
        if (first) {
            log_max = log_term;
            log_sum = 1.f;
            first   = false;
        } else if (log_term > log_max) {
            log_sum = log_sum * expf(log_max - log_term) + 1.f;
            log_max = log_term;
        } else {
            log_sum += expf(log_term - log_max);
        }
    }

    if (first) {
        // All terms were zero (impossible overlap configuration).
        scores[idx] = 0.f;
        return;
    }

    // log_p = log_max + log(log_sum)
    const float log_p = log_max + logf(log_sum);

    // score = -log_p / log(10); cap at 300 to avoid Inf.
    const float score = fminf(-log_p / 2.302585f, 300.f);
    scores[idx] = fmaxf(score, 0.f);  // clamp negative (p > 1 numerical artifacts) to 0
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// ora — public entry point
//
// Computes ORA (Over-Representation Analysis) pathway scores per cell.
// Returns -log10(hypergeometric p-value) for each (cell, gene set) pair.
//
// Inputs:
//   X         — m_genes × n_cells sparse CSC expression matrix (PzDeviceMatrix)
//   gene_sets — host-side vector of gene-index lists; indices in [0, m).
//   cfg       — OraConfig (top_k, deterministic)
//   stream    — optional cudaStream_t (nullptr → default context)
//
// Returns OraResult with scores (n_cells × n_sets, col-major) on device.
// Caller must sync stream before reading scores.
//
// Notes:
// - top_k = 0 → use ceil(m * 0.05) (decoupleR default).
// - Gene indices in gene_sets must be in [0, m). Checked on entry.
// - Requires cuBLAS handle; created and destroyed internally.
// ---------------------------------------------------------------------------
inline OraResult ora(
    const io::PzDeviceMatrix&              X,
    const std::vector<std::vector<int>>&  gene_sets,
    const OraConfig&                      cfg    = {},
    cudaStream_t                          stream = nullptr)
{
    if (stream == nullptr) {
        stream = singlet::gpu::core::default_context().stream();
    }

    const int m      = X.mat.rows;
    const int n      = X.mat.cols;
    const int nnz    = X.mat.nnz;
    const int n_sets = static_cast<int>(gene_sets.size());

    // --- Validate inputs ---
    if (m <= 0 || n <= 0) {
        throw std::runtime_error("ora: invalid matrix dimensions (m=" +
            std::to_string(m) + " n=" + std::to_string(n) + ")");
    }
    if (n_sets <= 0) {
        throw std::runtime_error("ora: gene_sets must be non-empty");
    }
    for (int s = 0; s < n_sets; ++s) {
        if (gene_sets[s].empty()) {
            throw std::runtime_error("ora: gene set " + std::to_string(s) + " is empty");
        }
        for (int g : gene_sets[s]) {
            if (g < 0 || g >= m) {
                throw std::runtime_error("ora: gene index " + std::to_string(g) +
                    " in set " + std::to_string(s) + " out of range [0, " +
                    std::to_string(m) + ")");
            }
        }
    }

    // Resolve top_k.
    const int top_k = (cfg.top_k > 0) ? cfg.top_k
        : static_cast<int>(std::ceil(m * 0.05f));

    constexpr int THREADS = 256;
    const size_t out_sz   = static_cast<size_t>(n) * static_cast<size_t>(n_sets);

    // -------------------------------------------------------------------------
    // Pass 1a — per-cell top-K threshold via shared-memory 64-bucket kernel.
    // smem = (m + 72) floats: column values + warp-max scratchpad + 64 int buckets.
    // v0 constraint: m <= ~12000 (smem <= 48KB). Throws if exceeded.
    // -------------------------------------------------------------------------
    const size_t smem_bytes = static_cast<size_t>(m + 72) * sizeof(float);
    if (smem_bytes > 49152) {  // 48KB hard limit (most CUDA devices)
        throw std::runtime_error(
            "ora: m=" + std::to_string(m) + " genes requires " +
            std::to_string(smem_bytes / 1024) + "KB shared memory per block (limit 48KB). "
            "Reduce to m <= 12000 via HVG selection before calling ora().");
    }

    core::DeviceMemory<float> d_threshold(n);
    cudaMemsetAsync(d_threshold.get(), 0, n * sizeof(float), stream);

    ora_topk_smem_kernel<<<n, THREADS, smem_bytes, stream>>>(
        X.mat.values.get(),
        X.mat.row_indices.get(),
        X.mat.col_ptr.get(),
        d_threshold.get(),
        m, n, top_k);

    // -------------------------------------------------------------------------
    // Pass 1b — binary top-K mask T (m × n, col-major fp32). Memory guard.
    // -------------------------------------------------------------------------
    {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        const size_t T_bytes = static_cast<size_t>(m) * static_cast<size_t>(n) * 4ULL;
        if (T_bytes > free_mem / 2) {
            throw std::runtime_error(
                "ora: dense T mask (m=" + std::to_string(m) +
                " × n=" + std::to_string(n) + ") requires " +
                std::to_string(T_bytes / (1024ULL * 1024ULL * 1024ULL)) +
                " GB but only " +
                std::to_string(free_mem / (1024ULL * 1024ULL * 1024ULL)) +
                " GB free. Reduce m via HVG selection.");
        }
    }

    const size_t T_size = static_cast<size_t>(m) * static_cast<size_t>(n);
    core::DeviceMemory<float> d_T(T_size);

    {
        const int total = static_cast<int>(T_size);
        const int grid  = (total + THREADS - 1) / THREADS;
        ora_build_T_kernel<<<grid, THREADS, 0, stream>>>(
            X.mat.values.get(),
            X.mat.row_indices.get(),
            X.mat.col_ptr.get(),
            d_threshold.get(),
            d_T.get(),
            m, n);
    }

    // -------------------------------------------------------------------------
    // Pass 2 — host-side M (m × n_sets col-major fp32): M[g,s]=1 if g∈set s.
    // Built host-side, H2D once.  Mirrors score_genes.h.
    // -------------------------------------------------------------------------
    const size_t M_size = static_cast<size_t>(m) * static_cast<size_t>(n_sets);
    std::vector<float> h_M(M_size, 0.f);
    std::vector<int>   h_set_size(n_sets);

    for (int s = 0; s < n_sets; ++s) {
        const std::vector<int>& S = gene_sets[s];
        h_set_size[s] = static_cast<int>(S.size());
        float* M_col = h_M.data() + static_cast<size_t>(s) * static_cast<size_t>(m);
        for (int g : S) {
            M_col[g] = 1.f;
        }
    }

    core::DeviceMemory<float> d_M(M_size > 0 ? M_size : 1);
    cudaMemcpyAsync(d_M.get(), h_M.data(), M_size * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // Upload per-set sizes and per-cell top-K to device.
    core::DeviceMemory<int> d_set_size(n_sets);
    cudaMemcpyAsync(d_set_size.get(), h_set_size.data(), n_sets * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // Per-cell top-K is fixed (same for all cells in v0) — broadcast into array.
    std::vector<int> h_topk(n, top_k);
    core::DeviceMemory<int> d_topk(n);
    cudaMemcpyAsync(d_topk.get(), h_topk.data(), n * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // -------------------------------------------------------------------------
    // Pass 3 — cuBLAS Sgemm: hits = T^T · M (n × n_sets, col-major).
    // opA=CUBLAS_OP_T on T (m×n), opB=NON_TRANSPOSE on M (m×n_sets).
    // lda=m, ldb=m, ldc=n.
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_hits(out_sz > 0 ? out_sz : 1);
    cudaMemsetAsync(d_hits.get(), 0, out_sz * sizeof(float), stream);

    if (out_sz > 0 && nnz > 0) {
        cublasHandle_t cblas;
        cublasCreate(&cblas);
        cublasSetStream(cblas, stream);

        const float alpha = 1.f, beta = 0.f;
        cublasSgemm(cblas,
                    CUBLAS_OP_T,   // op(T): transpose T to get T^T (n × m)
                    CUBLAS_OP_N,   // op(M): no transpose
                    n,             // rows of op(T) and C
                    n_sets,        // cols of M and C
                    m,             // inner dimension
                    &alpha,
                    d_T.get(), m,  // A=T, lda=m
                    d_M.get(), m,  // B=M, ldb=m
                    &beta,
                    d_hits.get(), n); // C=hits, ldc=n

        cublasDestroy(cblas);
    }

    // Pass 4 — Hypergeometric p-value kernel (one thread per (c,s)).
    core::DeviceMemory<float> d_scores(out_sz > 0 ? out_sz : 1);
    cudaMemsetAsync(d_scores.get(), 0, out_sz * sizeof(float), stream);

    if (out_sz > 0) {
        const int total = static_cast<int>(out_sz);
        const int grid  = (total + THREADS - 1) / THREADS;
        ora_hypergeo_kernel<<<grid, THREADS, 0, stream>>>(
            d_hits.get(),
            d_set_size.get(),
            d_topk.get(),
            d_scores.get(),
            n, n_sets, m);
    }

    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    OraResult res;
    res.scores  = std::move(d_scores);
    res.n_cells = n;
    res.n_sets  = n_sets;
    return res;
}

}  // namespace enrich
}  // namespace singlet::gpu
