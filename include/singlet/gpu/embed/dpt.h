// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: Haghverdi et al. 2016 (Diffusion Pseudotime)
//
// singlet-gpu/embed/dpt.h
//
// DPT — Diffusion Pseudotime.
//
// Algorithm reference:
//   Haghverdi L, Büttner M, Wolf FA, Buettner F, Theis FJ (2016)
//   "Diffusion pseudotime robustly reconstructs lineage branching."
//   Nat Methods 13:845-848. https://doi.org/10.1038/nmeth.3971
//   scanpy implementation: scanpy.tl.dpt
//
// Given a kNN graph (KnnResult from singlet_gpu::graph::knn) with n cells and
// k neighbors per cell, and a root cell index r, compute a per-cell pseudotime
// dpt[c] based on diffusion-map distances from the root.
//
// Algorithm (6 passes, GPU):
//   Pass 1 — per-cell local bandwidth σ_i = median of k distances.
//             One block per cell; thread 0 sorts k distances via selection sort.
//   Pass 2 — build dense symmetric Gaussian similarity W (n × n):
//             cudaMemset W to 0, then one thread per kNN edge:
//             W[i, j] = exp(-d[i,j]^2 / (σ_i * σ_j)).
//   Pass 3 — symmetrize: W[i,j] = (W[i,j] + W[j,i]) / 2 (in-place, upper only).
//   Pass 4 — row-normalize: T[i,j] = W[i,j] / rowsum_i (row-stochastic Markov T).
//             Per-row warp-shuffle reduction + per-element divide.
//   Pass 5 — cuSOLVER cusolverDnSsyevd: eigenvalues (ascending) + eigenvectors of T.
//   Pass 6 — per-cell DPT distance from root:
//             dpt[c] = sqrt(Σ_k (λ_k/(1-λ_k))^2 * (V[r,k] - V[c,k])^2)
//             summed over the top n_eigenvecs components (skip trivial λ≈1).
//
// Memory (Rule 5): all device buffers via core::DeviceMemory<T>. No raw cudaMalloc.
// D2H (Rule 4): one D2H scalar (info from cuSOLVER) is the canonical one-shot exception.
// Determinism (Rule 18): cuSOLVER Ssyevd is deterministic at fp32 on a fixed architecture.
//   cfg.deterministic is a documentation flag (no-op).
// Memory guard: dense T = n^2 * 4 bytes. Throws if > 50% of free GPU memory.
//   For n > ~5000, hint user to subset (Lanczos v1).
// CYCLE-142: initial implementation.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/graph/knn.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>

// Local CUSOLVER_CHECK — throws on cuSOLVER error. core/types.h ships only
// CUDA_CHECK / CUSPARSE_CHECK / CUBLAS_CHECK; cuSOLVER callers roll their own.
#ifndef CUSOLVER_CHECK
#  define CUSOLVER_CHECK(call) \
    do { cusolverStatus_t _s = (call); \
         if (_s != CUSOLVER_STATUS_SUCCESS) \
            throw std::runtime_error(std::string("cuSOLVER error at ") \
                + __FILE__ + ":" + std::to_string(__LINE__) \
                + " status=" + std::to_string(_s)); } while (0)
#endif

namespace singlet_gpu {
namespace embed {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct DptConfig {
    int   root_cell   = 0;      // index of root cell in [0, n)
    int   n_eigenvecs = 15;     // number of diffusion components used (skip trivial λ=1)
    float eps_lambda  = 1e-6f;  // clamp (1-λ) to max(eps_lambda, 1-λ) for stability
    bool  deterministic = true; // no-op: cuSOLVER Ssyevd is deterministic at fp32
};

struct DptResult {
    core::DeviceMemory<float> pseudotime;  // n_cells; pseudotime[root] ≈ 0
    int n_cells;
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels (anonymous namespace — not part of the public API)
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// dpt_sigma_kernel — compute per-cell local bandwidth σ_i = median of k
// distances (Pass 1).
//
// One block per cell; thread 0 runs a selection sort over the k distances to
// find the median. For k = 15 (typical), this is 15 × 7 = 105 comparisons —
// negligible. Using thread 0 only is deterministic and avoids shared-memory
// synchronization overhead.
//
// d_dists: distances layout [n * k] row-major (row i = distances for cell i).
// d_sigma: output σ[n].
// n, k: dimensions.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(1, 1)
void dpt_sigma_kernel(
    const float* __restrict__ d_dists,  // [n * k]
    float*       __restrict__ d_sigma,  // [n]
    int n, int k)
{
    const int cell = static_cast<int>(blockIdx.x);
    if (cell >= n) return;

    // Copy k distances into local registers / stack for in-place selection sort.
    // k <= 64 in practice; VLA stack is fine at k <= 32. Use fixed array size 64.
    float buf[64];
    const int kk = (k < 64) ? k : 64;
    const float* row = d_dists + (ptrdiff_t)cell * k;
    for (int i = 0; i < kk; ++i) buf[i] = row[i];

    // Selection sort to find median position (kk/2).
    const int med_idx = kk / 2;
    for (int i = 0; i <= med_idx; ++i) {
        int min_i = i;
        for (int j = i + 1; j < kk; ++j) {
            if (buf[j] < buf[min_i]) min_i = j;
        }
        // Swap.
        float tmp   = buf[i];
        buf[i]      = buf[min_i];
        buf[min_i]  = tmp;
    }

    d_sigma[cell] = buf[med_idx];
}

// ---------------------------------------------------------------------------
// dpt_fill_w_kernel — fill dense W from kNN edges (Pass 2).
//
// One thread per stored kNN edge (total n*k threads).
// W[i, j] = exp(-d[i,j]^2 / (σ_i * σ_j)).
// W is n × n col-major (ld = n): index W[i + j*n].
// Zero-initialized by caller (cudaMemset).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void dpt_fill_w_kernel(
    const int*   __restrict__ d_neighbors,  // [n * k] neighbor indices
    const float* __restrict__ d_dists,      // [n * k] distances
    const float* __restrict__ d_sigma,      // [n] per-cell bandwidth
    float*       __restrict__ d_W,          // [n * n] col-major, pre-zeroed
    int n, int k)
{
    const int edge = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                   + static_cast<int>(threadIdx.x);
    if (edge >= n * k) return;

    const int i  = edge / k;
    const int j  = d_neighbors[edge];
    const float d = d_dists[edge];
    const float si = d_sigma[i];
    const float sj = d_sigma[j];
    const float denom = si * sj;
    const float w = (denom > 0.f) ? expf(-(d * d) / denom) : 0.f;
    // W is col-major: W[i, j] = W[i + j*n].
    d_W[i + (ptrdiff_t)j * n] = w;
}

// ---------------------------------------------------------------------------
// dpt_symmetrize_kernel — symmetrize W in-place: W[i,j] = (W[i,j]+W[j,i])/2
// (Pass 3).
//
// One thread per upper-triangle element (i < j). Updates both W[i,j] and
// W[j,i] simultaneously. Diagonal is left as-is (self-similarity = 1 from
// the fill, but is handled gracefully in row-normalization).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void dpt_symmetrize_kernel(
    float* __restrict__ d_W,  // [n * n] col-major, modified in-place
    int n)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    // Map linear index to upper-triangle (i < j).
    // Total upper-triangle elements = n*(n-1)/2.
    // Use row/col mapping: iterate over (i, j) pairs with i < j.
    // idx maps to (i, j) via: j = floor((1 + sqrt(1 + 8*idx))/2), i = idx - j*(j-1)/2.
    const int total_upper = n * (n - 1) / 2;
    if (idx >= total_upper) return;

    // Reverse-map idx → (i, j) with i < j.
    // j satisfies j*(j-1)/2 <= idx < j*(j+1)/2 → j = floor((1 + sqrt(1+8*idx))/2).
    int j = static_cast<int>((1.f + sqrtf(1.f + 8.f * static_cast<float>(idx))) * 0.5f);
    // Correct for floating-point rounding.
    while (j * (j - 1) / 2 > idx) --j;
    while ((j + 1) * j / 2 <= idx) ++j;
    const int i = idx - j * (j - 1) / 2;

    // W is col-major: W[i,j] at offset i + j*n; W[j,i] at j + i*n.
    const float wij = d_W[i + (ptrdiff_t)j * n];
    const float wji = d_W[j + (ptrdiff_t)i * n];
    const float sym = (wij + wji) * 0.5f;
    d_W[i + (ptrdiff_t)j * n] = sym;
    d_W[j + (ptrdiff_t)i * n] = sym;
}

// ---------------------------------------------------------------------------
// dpt_rowsum_kernel — compute per-row sum of W (Pass 4a).
//
// One block per row (cell). Threads stride over the n columns and warp-shuffle
// reduce into d_rowsum[row].
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void dpt_rowsum_kernel(
    const float* __restrict__ d_W,       // [n * n] col-major
    float*       __restrict__ d_rowsum,  // [n] output
    int n)
{
    const int row  = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int wid  = static_cast<int>(threadIdx.x) / 32;
    const int nwarp = static_cast<int>(blockDim.x) / 32;

    if (row >= n) return;

    float acc = 0.f;
    // W is col-major: W[row, col] = W[row + col*n].
    for (int col = static_cast<int>(threadIdx.x); col < n; col += static_cast<int>(blockDim.x)) {
        acc += d_W[row + (ptrdiff_t)col * n];
    }

    // Warp reduction.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    __shared__ float smem[8];
    if (lane == 0) smem[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        float val = (lane < nwarp) ? smem[lane] : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            val += __shfl_down_sync(0xffffffffu, val, off);
        }
        if (lane == 0) d_rowsum[row] = val;
    }
}

// ---------------------------------------------------------------------------
// dpt_symnorm_kernel — symmetric normalization W → T_sym in-place (Pass 4b).
//
// T_sym[i, j] = W[i, j] / sqrt(rowsum[i] · rowsum[j])
//
// WHY symmetric (not row-stochastic): cusolverDnSsyevd assumes a SYMMETRIC
// input matrix. The asymmetric Markov T = D^{-1} W has the SAME eigenvalues
// as T_sym = D^{-1/2} W D^{-1/2}, and eigenvectors of T_sym are valid
// substitutes in the DPT formula (which uses squared diffs of eigenvector
// components — sign and scaling factor of D^{-1/2} cancel).
//
// Bug fix CYCLE-142.1 (job 369371 → 369xxx): original kernel used
// row-normalize (T = D^{-1} W), passed asymmetric T to Ssyevd, got
// effectively-random eigenvectors → Spearman(dpt, chain_index) ≈ 0.
// One thread per element. W is col-major: element (i, j) at offset i + j*n.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void dpt_symnorm_kernel(
    float*       __restrict__ d_W,       // [n * n] col-major, modified in-place → T_sym
    const float* __restrict__ d_rowsum,  // [n]
    int n)
{
    const int ij = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                 + static_cast<int>(threadIdx.x);
    if (ij >= n * n) return;

    const int i = ij % n;   // row (col-major: inner index = row)
    const int j = ij / n;   // col
    const float rs_i = d_rowsum[i];
    const float rs_j = d_rowsum[j];
    const float denom = sqrtf(rs_i * rs_j);
    if (denom > 0.f) d_W[ij] /= denom;
    else             d_W[ij]  = 0.f;
}

// ---------------------------------------------------------------------------
// dpt_pseudotime_kernel — compute per-cell DPT distance from root (Pass 6).
//
// One thread per cell. Iterates over the top n_eigenvecs eigenvalues (last
// n_eigenvecs entries in ascending λ array), skips the trivial λ ≈ 1
// eigenvector (the largest), and accumulates:
//   dpt[c] = sqrt(Σ_k (λ_k / max(eps, 1 - λ_k))^2 * (V[r, k] - V[c, k])^2)
//
// V is col-major (n × n, eigenvectors as columns): V[c, col_k] = V[c + col_k*n].
// Eigenvalues λ are sorted ascending: top n_eigenvecs are at [n-n_eigenvecs..n-1].
// The largest eigenvalue (index n-1, ≈ 1.0) is skipped.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void dpt_pseudotime_kernel(
    const float* __restrict__ d_lambda,       // [n] eigenvalues, ascending
    const float* __restrict__ d_V,            // [n * n] col-major eigenvectors
    float*       __restrict__ d_dpt,          // [n] output pseudotime
    int n, int root, int n_eigenvecs, float eps_lambda)
{
    const int cell = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                   + static_cast<int>(threadIdx.x);
    if (cell >= n) return;

    float acc = 0.f;

    // Top n_eigenvecs are at indices [n - n_eigenvecs .. n - 1].
    // Skip index n-1 (trivial λ ≈ 1). Use n_eigenvecs-1 components.
    const int start_idx = n - n_eigenvecs;  // inclusive
    const int end_idx   = n - 2;            // inclusive (skip n-1)

    for (int k = start_idx; k <= end_idx; ++k) {
        if (k < 0) continue;
        const float lam  = d_lambda[k];
        const float denom = (1.f - lam > eps_lambda) ? (1.f - lam) : eps_lambda;
        const float scale = lam / denom;
        // V is col-major: V[cell, k] = d_V[cell + k*n].
        const float vr = d_V[root + (ptrdiff_t)k * n];
        const float vc = d_V[cell + (ptrdiff_t)k * n];
        const float diff = vr - vc;
        acc += scale * scale * diff * diff;
    }

    d_dpt[cell] = sqrtf(acc);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// dpt() — public entry point
//
// Inputs:
//   knn  — KnnResult from singlet_gpu::graph::compute_knn (uses .neighbors,
//           .distances, .n, .k).
//   cfg  — DptConfig (root_cell, n_eigenvecs, eps_lambda, deterministic).
//   stream — optional CUDA stream (nullptr → default context stream).
//
// Returns DptResult with device-resident pseudotime[n_cells].
// pseudotime[root_cell] ≈ 0. Caller must sync stream before reading.
//
// Memory guard: dense T = n^2 * 4 bytes. Throws if > 50% free GPU memory.
// For n > ~5000 the guard will typically fire; use Lanczos v1 for larger n.
// ---------------------------------------------------------------------------
inline DptResult dpt(
    const graph::KnnResult& knn,
    const DptConfig&        cfg    = {},
    cudaStream_t            stream = nullptr)
{
    if (stream == nullptr) {
        stream = singlet_gpu::core::default_context().stream();
    }

    const int n = knn.n;
    const int k = knn.k;

    // Dimension validation.
    if (n <= 0)
        throw std::runtime_error("dpt: empty knn graph (n=" + std::to_string(n) + ")");
    if (k <= 0)
        throw std::runtime_error("dpt: k must be > 0 (k=" + std::to_string(k) + ")");
    if (cfg.root_cell < 0 || cfg.root_cell >= n)
        throw std::runtime_error(
            "dpt: root_cell=" + std::to_string(cfg.root_cell) +
            " out of range [0, " + std::to_string(n) + ")");
    if (cfg.n_eigenvecs < 2)
        throw std::runtime_error(
            "dpt: n_eigenvecs must be >= 2 (got " + std::to_string(cfg.n_eigenvecs) + ")");
    if (cfg.n_eigenvecs >= n)
        throw std::runtime_error(
            "dpt: n_eigenvecs=" + std::to_string(cfg.n_eigenvecs) +
            " must be < n=" + std::to_string(n));
    if (knn.distances.size() == 0)
        throw std::runtime_error("dpt: knn.distances is empty — kNN must include distances");

    // -------------------------------------------------------------------------
    // Memory guard: dense T is n * n * 4 bytes. Reject if > 50% free GPU mem.
    // -------------------------------------------------------------------------
    {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        const size_t needed = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);
        if (needed > free_mem / 2) {
            const double needed_gb = static_cast<double>(needed) / (1024.0 * 1024.0 * 1024.0);
            const double free_gb   = static_cast<double>(free_mem) / (1024.0 * 1024.0 * 1024.0);
            throw std::runtime_error(
                "dpt: dense transition matrix requires " +
                std::to_string(needed_gb).substr(0, 5) + " GB (n=" + std::to_string(n) +
                " cells, n^2 floats) but only " +
                std::to_string(free_gb).substr(0, 5) + " GB free. "
                "Subset to n <= 5000 cells or await Lanczos v1 support.");
        }
    }

    constexpr int THREADS = 256;

    // -------------------------------------------------------------------------
    // Pass 1 — per-cell local bandwidth σ_i = median(distances[i, 0..k-1]).
    //
    // One block per cell; thread 0 sorts k distances via selection sort.
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_sigma(static_cast<size_t>(n));
    dpt_sigma_kernel<<<n, 1, 0, stream>>>(
        knn.distances.get(), d_sigma.get(), n, k);

    // -------------------------------------------------------------------------
    // Pass 2 — build dense W (n × n, col-major) from kNN edges.
    //
    // cudaMemset W to 0 first, then write Gaussian similarities.
    // -------------------------------------------------------------------------
    const size_t W_sz = static_cast<size_t>(n) * static_cast<size_t>(n);
    core::DeviceMemory<float> d_W(W_sz);
    CUDA_CHECK(cudaMemsetAsync(d_W.get(), 0, W_sz * sizeof(float), stream));

    {
        const int total_edges = n * k;
        const int grid = (total_edges + THREADS - 1) / THREADS;
        dpt_fill_w_kernel<<<grid, THREADS, 0, stream>>>(
            knn.neighbors.get(), knn.distances.get(), d_sigma.get(),
            d_W.get(), n, k);
    }

    // -------------------------------------------------------------------------
    // Pass 3 — symmetrize W: W[i,j] = (W[i,j] + W[j,i]) / 2.
    //
    // One thread per upper-triangle element; updates both W[i,j] and W[j,i].
    // -------------------------------------------------------------------------
    {
        const int upper_tri = n * (n - 1) / 2;
        const int grid = (upper_tri + THREADS - 1) / THREADS;
        dpt_symmetrize_kernel<<<grid, THREADS, 0, stream>>>(d_W.get(), n);
    }

    // -------------------------------------------------------------------------
    // Pass 4 — row-normalize W → T (row-stochastic Markov matrix).
    //
    // Step 4a: per-row sum (one block per row, warp-shuffle reduction).
    // Step 4b: per-element divide T[i,j] = W[i,j] / rowsum[i].
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_rowsum(static_cast<size_t>(n));
    dpt_rowsum_kernel<<<n, THREADS, 0, stream>>>(d_W.get(), d_rowsum.get(), n);

    {
        const int grid = (static_cast<int>(W_sz) + THREADS - 1) / THREADS;
        dpt_symnorm_kernel<<<grid, THREADS, 0, stream>>>(d_W.get(), d_rowsum.get(), n);
    }

    // -------------------------------------------------------------------------
    // Pass 5 — cuSOLVER cusolverDnSsyevd: eigendecompose T (symmetric).
    //
    // Returns eigenvalues λ in ascending order, eigenvectors V as columns
    // (col-major, in-place in d_W). LOWER fill mode.
    // d_info: one D2H scalar (canonical one-shot exception to Rule 4).
    // -------------------------------------------------------------------------
    cusolverDnHandle_t solver_h = singlet_gpu::core::default_context().solver();
    cusolverDnSetStream(solver_h, stream);

    // Eigenvalue output (ascending order).
    core::DeviceMemory<float> d_lambda(static_cast<size_t>(n));

    // Query workspace.
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSsyevd_bufferSize(
        solver_h,
        CUSOLVER_EIG_MODE_VECTOR,  // compute eigenvectors
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_W.get(),                 // T (n × n col-major, in-place)
        n,                         // leading dimension
        d_lambda.get(),
        &lwork));

    core::DeviceMemory<float> d_work(lwork > 0 ? static_cast<size_t>(lwork) : 1);
    core::DeviceMemory<int>   d_info(1);

    CUSOLVER_CHECK(cusolverDnSsyevd(
        solver_h,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_W.get(),      // on entry: T (col-major); on exit: eigenvectors V (cols)
        n,
        d_lambda.get(), // on exit: eigenvalues ascending
        d_work.get(),
        lwork,
        d_info.get()));

    // Sync and check eigendecomp info (one D2H scalar — Rule 4 one-shot exception).
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int h_info = 0;
    cudaMemcpy(&h_info, d_info.get(), sizeof(int), cudaMemcpyDeviceToHost);
    if (h_info != 0) {
        throw std::runtime_error(
            "dpt: cusolverDnSsyevd failed (info=" + std::to_string(h_info) + "). "
            "T may not be symmetric or GPU memory is corrupt.");
    }
    // d_W now holds the eigenvectors V (n × n col-major, columns are eigenvectors).
    // d_lambda holds eigenvalues ascending.

    // -------------------------------------------------------------------------
    // Pass 6 — per-cell DPT pseudotime from root.
    //
    // dpt[c] = sqrt(Σ_k (λ_k / max(eps, 1-λ_k))^2 * (V[r,k] - V[c,k])^2)
    // Summed over k = [n - n_eigenvecs .. n - 2] (skip trivial λ ≈ 1 at n-1).
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_dpt(static_cast<size_t>(n));
    {
        const int grid = (n + THREADS - 1) / THREADS;
        dpt_pseudotime_kernel<<<grid, THREADS, 0, stream>>>(
            d_lambda.get(), d_W.get(), d_dpt.get(),
            n, cfg.root_cell, cfg.n_eigenvecs, cfg.eps_lambda);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    DptResult res;
    res.pseudotime = std::move(d_dpt);
    res.n_cells    = n;
    return res;
}

}  // namespace embed
}  // namespace singlet_gpu
