// SPDX-License-Identifier: MIT
// integrates: Coifman & Lafon 2005 (Diffusion Maps)
//
// singlet/gpu/embed/diffmap.h
//
// Diffusion Map embedding.
//
// Algorithm reference:
//   Coifman RR, Lafon S, Lee AB, et al. (2005)
//   "Geometric diffusions as a tool for harmonic analysis and structure
//    definition of data: Diffusion maps." PNAS 102:7426-7431.
//   scanpy implementation: scanpy.tl.diffmap
//
// Given a kNN graph (KnnResult from singlet::gpu::graph::knn) with n cells and
// k neighbors per cell, compute a low-dimensional diffusion-map embedding.
//
// Algorithm (6 passes, GPU) — shares passes 1-5 with dpt.h (CYCLE-142):
//   Pass 1 — per-cell local bandwidth σ_i = median of k distances.
//   Pass 2 — build dense symmetric Gaussian similarity W (n × n).
//   Pass 3 — symmetrize: W[i,j] = (W[i,j] + W[j,i]) / 2.
//   Pass 4 — symmetric normalize: T_sym = D^{-1/2} W D^{-1/2}.
//   Pass 5 — cuSOLVER Ssyevd: eigenvalues (ascending) + eigenvectors.
//   Pass 6 — scale top n_comps eigenvectors by λ^t (diffusion time).
//             embedding[c, k] = λ_k^t * V[c, lam_idx]
//             for lam_idx = n-2-k (skip trivial λ ≈ 1 at n-1).
//
// Output: dense (n_cells × n_comps) embedding col-major.
//
// Memory (Rule 5): all device buffers via core::DeviceMemory<T>. No raw cudaMalloc.
// D2H (Rule 4): one D2H scalar (info from cuSOLVER) is the canonical one-shot exception.
// Determinism (Rule 18): cuSOLVER Ssyevd is deterministic at fp32 on a fixed architecture.
// Memory guard: dense T = n^2 * 4 bytes. Throws if > 50% of free GPU memory.
// CYCLE-150: initial implementation.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/graph/knn.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>

// Local CUSOLVER_CHECK — throws on cuSOLVER error. core/types.h ships only
// SINGLET_GPU_CUDA_CHECK / CUSPARSE_CHECK / CUBLAS_CHECK; cuSOLVER callers roll their own.
#ifndef CUSOLVER_CHECK
#  define CUSOLVER_CHECK(call) \
    do { cusolverStatus_t _s = (call); \
         if (_s != CUSOLVER_STATUS_SUCCESS) \
            throw std::runtime_error(std::string("cuSOLVER error at ") \
                + __FILE__ + ":" + std::to_string(__LINE__) \
                + " status=" + std::to_string(_s)); } while (0)
#endif

namespace singlet::gpu {
namespace embed {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct DiffmapConfig {
    int   n_comps      = 15;     // number of diffusion components (excludes trivial λ=1)
    int   t            = 1;      // diffusion time (eigenvalues raised to this power)
    float eps_lambda   = 1e-6f;  // guard for very small λ
    bool  deterministic = true;  // no-op: cuSOLVER Ssyevd is deterministic at fp32
};

struct DiffmapResult {
    core::DeviceMemory<float> embedding;    // n_cells × n_comps col-major
    core::DeviceMemory<float> eigenvalues;  // n_comps (descending — largest non-trivial first)
    int n_cells, n_comps;
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels (anonymous namespace — not part of the public API)
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// diffmap_sigma_kernel — compute per-cell local bandwidth σ_i = median of k
// distances (Pass 1).
//
// One block per cell; thread 0 runs a selection sort over the k distances.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(1, 1)
void diffmap_sigma_kernel(
    const float* __restrict__ d_dists,  // [n * k]
    float*       __restrict__ d_sigma,  // [n]
    int n, int k)
{
    const int cell = static_cast<int>(blockIdx.x);
    if (cell >= n) return;

    float buf[64];
    const int kk = (k < 64) ? k : 64;
    const float* row = d_dists + (ptrdiff_t)cell * k;
    for (int i = 0; i < kk; ++i) buf[i] = row[i];

    const int med_idx = kk / 2;
    for (int i = 0; i <= med_idx; ++i) {
        int min_i = i;
        for (int j = i + 1; j < kk; ++j) {
            if (buf[j] < buf[min_i]) min_i = j;
        }
        float tmp   = buf[i];
        buf[i]      = buf[min_i];
        buf[min_i]  = tmp;
    }

    d_sigma[cell] = buf[med_idx];
}

// ---------------------------------------------------------------------------
// diffmap_fill_w_kernel — fill dense W from kNN edges (Pass 2).
//
// W[i, j] = exp(-d[i,j]^2 / (σ_i * σ_j)).
// W is n × n col-major (ld = n). Zero-initialized by caller.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void diffmap_fill_w_kernel(
    const int*   __restrict__ d_neighbors,  // [n * k] neighbor indices
    const float* __restrict__ d_dists,      // [n * k] distances
    const float* __restrict__ d_sigma,      // [n] per-cell bandwidth
    float*       __restrict__ d_W,          // [n * n] col-major, pre-zeroed
    int n, int k)
{
    const int edge = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                   + static_cast<int>(threadIdx.x);
    if (edge >= n * k) return;

    const int i   = edge / k;
    const int j   = d_neighbors[edge];
    const float d = d_dists[edge];
    const float si = d_sigma[i];
    const float sj = d_sigma[j];
    const float denom = si * sj;
    const float w = (denom > 0.f) ? expf(-(d * d) / denom) : 0.f;
    d_W[i + (ptrdiff_t)j * n] = w;
}

// ---------------------------------------------------------------------------
// diffmap_symmetrize_kernel — symmetrize W in-place: W[i,j] = (W[i,j]+W[j,i])/2
// (Pass 3). One thread per upper-triangle element.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void diffmap_symmetrize_kernel(
    float* __restrict__ d_W,  // [n * n] col-major, modified in-place
    int n)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    const int total_upper = n * (n - 1) / 2;
    if (idx >= total_upper) return;

    int j = static_cast<int>((1.f + sqrtf(1.f + 8.f * static_cast<float>(idx))) * 0.5f);
    while (j * (j - 1) / 2 > idx) --j;
    while ((j + 1) * j / 2 <= idx) ++j;
    const int i = idx - j * (j - 1) / 2;

    const float wij = d_W[i + (ptrdiff_t)j * n];
    const float wji = d_W[j + (ptrdiff_t)i * n];
    const float sym = (wij + wji) * 0.5f;
    d_W[i + (ptrdiff_t)j * n] = sym;
    d_W[j + (ptrdiff_t)i * n] = sym;
}

// ---------------------------------------------------------------------------
// diffmap_rowsum_kernel — compute per-row sum of W (Pass 4a).
//
// One block per row; warp-shuffle reduction into d_rowsum[row].
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void diffmap_rowsum_kernel(
    const float* __restrict__ d_W,       // [n * n] col-major
    float*       __restrict__ d_rowsum,  // [n] output
    int n)
{
    const int row   = static_cast<int>(blockIdx.x);
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int wid   = static_cast<int>(threadIdx.x) / 32;
    const int nwarp = static_cast<int>(blockDim.x) / 32;

    if (row >= n) return;

    float acc = 0.f;
    for (int col = static_cast<int>(threadIdx.x); col < n; col += static_cast<int>(blockDim.x)) {
        acc += d_W[row + (ptrdiff_t)col * n];
    }

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
// diffmap_symnorm_kernel — symmetric normalization W → T_sym in-place (Pass 4b).
//
// T_sym[i, j] = W[i, j] / sqrt(rowsum[i] · rowsum[j])
// Input to cusolverDnSsyevd must be symmetric; D^{-1/2} W D^{-1/2} satisfies
// this requirement (eigenvectors equivalent to Markov T up to D^{1/2} scaling).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void diffmap_symnorm_kernel(
    float*       __restrict__ d_W,       // [n * n] col-major, modified in-place
    const float* __restrict__ d_rowsum,  // [n]
    int n)
{
    const int ij = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                 + static_cast<int>(threadIdx.x);
    if (ij >= n * n) return;

    const int i = ij % n;
    const int j = ij / n;
    const float rs_i = d_rowsum[i];
    const float rs_j = d_rowsum[j];
    const float denom = sqrtf(rs_i * rs_j);
    if (denom > 0.f) d_W[ij] /= denom;
    else             d_W[ij]  = 0.f;
}

// ---------------------------------------------------------------------------
// diffmap_scale_kernel — diffusion map embedding: copy top n_comps eigenvectors
// (skip trivial λ=1) and scale by λ^t (Pass 6).
//
// cuSOLVER returns eigenvalues ascending. Top n_comps non-trivial eigenvectors
// are at lambda indices [n-n_comps-1 .. n-2] (skip n-1 = trivial λ ≈ 1).
// embedding[c, k] = λ_k^t * V[c, lam_idx],  lam_idx = n-2-k (k=0 = largest).
// Output is col-major: embedding[c + k*n].
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void diffmap_scale_kernel(
    const float* __restrict__ d_lambda,  // [n] ascending eigenvalues
    const float* __restrict__ d_V,       // [n * n] col-major eigenvectors
    float*       __restrict__ d_embed,   // [n * n_comps] col-major output
    int n, int n_comps, int t)
{
    int cell = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
             + static_cast<int>(threadIdx.x);
    if (cell >= n) return;

    for (int k = 0; k < n_comps; ++k) {
        int lam_idx = n - 2 - k;  // descending: k=0 gets largest non-trivial
        if (lam_idx < 0) {
            d_embed[cell + (ptrdiff_t)k * n] = 0.f;
            continue;
        }
        float lam   = d_lambda[lam_idx];
        float scale = (t == 1) ? lam : powf(lam, static_cast<float>(t));
        float v     = d_V[cell + (ptrdiff_t)lam_idx * n];
        d_embed[cell + (ptrdiff_t)k * n] = scale * v;
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// diffmap() — public entry point
//
// Inputs:
//   knn    — KnnResult from singlet::gpu::graph::compute_knn.
//   cfg    — DiffmapConfig (n_comps, t, eps_lambda, deterministic).
//   stream — optional CUDA stream (nullptr → default context stream).
//
// Returns DiffmapResult with device-resident embedding[n_cells * n_comps]
// (col-major) and eigenvalues[n_comps] (descending, non-trivial).
// Caller must sync stream before reading.
//
// Memory guard: dense T = n^2 * 4 bytes. Throws if > 50% free GPU memory.
// ---------------------------------------------------------------------------
inline DiffmapResult diffmap(
    const graph::KnnResult& knn,
    const DiffmapConfig&    cfg    = {},
    cudaStream_t            stream = nullptr)
{
    if (stream == nullptr) {
        stream = singlet::gpu::core::default_context().stream();
    }

    const int n = knn.n;
    const int k = knn.k;

    if (n <= 0)
        throw std::runtime_error("diffmap: empty knn graph (n=" + std::to_string(n) + ")");
    if (k <= 0)
        throw std::runtime_error("diffmap: k must be > 0 (k=" + std::to_string(k) + ")");
    if (cfg.n_comps < 1)
        throw std::runtime_error(
            "diffmap: n_comps must be >= 1 (got " + std::to_string(cfg.n_comps) + ")");
    if (cfg.n_comps >= n - 1)
        throw std::runtime_error(
            "diffmap: n_comps=" + std::to_string(cfg.n_comps) +
            " must be < n-1=" + std::to_string(n - 1));
    if (cfg.t < 1)
        throw std::runtime_error(
            "diffmap: t must be >= 1 (got " + std::to_string(cfg.t) + ")");
    if (knn.distances.size() == 0)
        throw std::runtime_error("diffmap: knn.distances is empty — kNN must include distances");

    // -------------------------------------------------------------------------
    // Memory guard: dense T = n * n * 4 bytes. Reject if > 50% free GPU mem.
    // -------------------------------------------------------------------------
    {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        const size_t needed = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);
        if (needed > free_mem / 2) {
            const double needed_gb = static_cast<double>(needed) / (1024.0 * 1024.0 * 1024.0);
            const double free_gb   = static_cast<double>(free_mem)  / (1024.0 * 1024.0 * 1024.0);
            throw std::runtime_error(
                "diffmap: dense transition matrix requires " +
                std::to_string(needed_gb).substr(0, 5) + " GB (n=" + std::to_string(n) +
                " cells, n^2 floats) but only " +
                std::to_string(free_gb).substr(0, 5) + " GB free. "
                "Subset to n <= 5000 cells or await Lanczos v1 support.");
        }
    }

    constexpr int THREADS = 256;

    // -------------------------------------------------------------------------
    // Pass 1 — per-cell local bandwidth σ_i = median(distances[i, 0..k-1]).
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_sigma(static_cast<size_t>(n));
    diffmap_sigma_kernel<<<n, 1, 0, stream>>>(
        knn.distances.get(), d_sigma.get(), n, k);

    // -------------------------------------------------------------------------
    // Pass 2 — build dense W (n × n, col-major) from kNN edges.
    // -------------------------------------------------------------------------
    const size_t W_sz = static_cast<size_t>(n) * static_cast<size_t>(n);
    core::DeviceMemory<float> d_W(W_sz);
    SINGLET_GPU_CUDA_CHECK(cudaMemsetAsync(d_W.get(), 0, W_sz * sizeof(float), stream));

    {
        const int total_edges = n * k;
        const int grid = (total_edges + THREADS - 1) / THREADS;
        diffmap_fill_w_kernel<<<grid, THREADS, 0, stream>>>(
            knn.neighbors.get(), knn.distances.get(), d_sigma.get(),
            d_W.get(), n, k);
    }

    // -------------------------------------------------------------------------
    // Pass 3 — symmetrize W: W[i,j] = (W[i,j] + W[j,i]) / 2.
    // -------------------------------------------------------------------------
    {
        const int upper_tri = n * (n - 1) / 2;
        const int grid = (upper_tri + THREADS - 1) / THREADS;
        diffmap_symmetrize_kernel<<<grid, THREADS, 0, stream>>>(d_W.get(), n);
    }

    // -------------------------------------------------------------------------
    // Pass 4 — symmetric normalize W → T_sym (D^{-1/2} W D^{-1/2}).
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_rowsum(static_cast<size_t>(n));
    diffmap_rowsum_kernel<<<n, THREADS, 0, stream>>>(d_W.get(), d_rowsum.get(), n);

    {
        const int grid = (static_cast<int>(W_sz) + THREADS - 1) / THREADS;
        diffmap_symnorm_kernel<<<grid, THREADS, 0, stream>>>(d_W.get(), d_rowsum.get(), n);
    }

    // -------------------------------------------------------------------------
    // Pass 5 — cuSOLVER Ssyevd: eigendecompose T_sym (symmetric).
    // -------------------------------------------------------------------------
    cusolverDnHandle_t solver_h = singlet::gpu::core::default_context().solver();
    cusolverDnSetStream(solver_h, stream);

    core::DeviceMemory<float> d_lambda(static_cast<size_t>(n));

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSsyevd_bufferSize(
        solver_h,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_W.get(),
        n,
        d_lambda.get(),
        &lwork));

    core::DeviceMemory<float> d_work(lwork > 0 ? static_cast<size_t>(lwork) : 1);
    core::DeviceMemory<int>   d_info(1);

    CUSOLVER_CHECK(cusolverDnSsyevd(
        solver_h,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_W.get(),       // on entry: T_sym; on exit: eigenvectors V (cols)
        n,
        d_lambda.get(),  // on exit: eigenvalues ascending
        d_work.get(),
        lwork,
        d_info.get()));

    // Sync and check eigendecomp info (one D2H scalar — Rule 4 one-shot exception).
    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    int h_info = 0;
    cudaMemcpy(&h_info, d_info.get(), sizeof(int), cudaMemcpyDeviceToHost);
    if (h_info != 0) {
        throw std::runtime_error(
            "diffmap: cusolverDnSsyevd failed (info=" + std::to_string(h_info) + "). "
            "T_sym may not be symmetric or GPU memory is corrupt.");
    }
    // d_W holds eigenvectors V (n × n col-major). d_lambda holds eigenvalues ascending.

    // -------------------------------------------------------------------------
    // Pass 6 — diffusion map: scale top n_comps eigenvectors by λ^t.
    //
    // lam_idx k=0 → index n-2 (largest non-trivial); skip n-1 (trivial λ ≈ 1).
    // Output embedding[c, k] = λ_k^t * V[c, n-2-k], col-major: c + k*n.
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_embed(
        static_cast<size_t>(n) * static_cast<size_t>(cfg.n_comps));
    {
        const int grid = (n + THREADS - 1) / THREADS;
        diffmap_scale_kernel<<<grid, THREADS, 0, stream>>>(
            d_lambda.get(), d_W.get(), d_embed.get(),
            n, cfg.n_comps, cfg.t);
    }

    // Copy top n_comps eigenvalues (descending) to output buffer.
    // d_lambda is ascending; top n_comps non-trivial are at [n-n_comps-1 .. n-2].
    // We copy them in reverse so output[0] = largest non-trivial.
    core::DeviceMemory<float> d_eig_out(static_cast<size_t>(cfg.n_comps));
    for (int k = 0; k < cfg.n_comps; ++k) {
        int src_idx = n - 2 - k;
        if (src_idx < 0) {
            // Fill with 0 if n_comps exceeds available non-trivial eigenvalues.
            float zero = 0.f;
            cudaMemcpyAsync(d_eig_out.get() + k, &zero, sizeof(float),
                            cudaMemcpyHostToDevice, stream);
        } else {
            cudaMemcpyAsync(d_eig_out.get() + k,
                            d_lambda.get() + src_idx,
                            sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }
    }

    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    DiffmapResult res;
    res.embedding   = std::move(d_embed);
    res.eigenvalues = std::move(d_eig_out);
    res.n_cells     = n;
    res.n_comps     = cfg.n_comps;
    return res;
}

}  // namespace embed
}  // namespace singlet::gpu
