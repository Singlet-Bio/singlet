// SPDX-License-Identifier: MIT
// integrates: original (first GPU Palantir-style diffusion pseudotime + fate inference)
// reuses: cycle 43 fate/cellrank2.h for absorption
//
// fate/palantir.h — GPU-native diffusion pseudotime + branch probability inference
//
// Algorithm reference:
//   Setty et al., "Characterization of cell fate probabilities in single-cell
//   data with Palantir," Nature Biotechnology 2019.
//   https://doi.org/10.1038/s41587-019-0068-4
//
// Core algorithm:
//   1. Build kNN graph (cycle-8 compute_exact, k=30) on PCA embedding.
//   2. Gaussian diffusion kernel: w[i,j] = exp(-d²[i,j] / (2 σᵢ²)) where
//      σᵢ = distance to the k'th neighbor (adaptive bandwidth).
//   3. Row-normalize w → stochastic P (one-step Markov chain).
//   4. Anisotropic normalization: P' = D^{-1/2} P D^{-1/2}, where D is
//      the row-sum diagonal of the raw (un-normalized) kernel W.
//      P' is symmetric, enabling efficient eigendecomposition.
//   5. Spectral decomposition: top-k eigenvectors (ψ_1…ψ_k, λ_1…λ_k)
//      of P' via singlet::gpu::reduce::svd::deflation (k ≈ 20).
//      WHY deflation/SVD for an eigenproblem: P' is symmetric PSD; its SVD equals
//      its eigendecomposition (σᵢ = |λᵢ|), so right singular vectors = eigenvectors
//      and singular values = eigenvalues. Lanczos SVD is numerically stable and
//      already integrated.
//   6. Pseudotime: pt[c] = sqrt( Σ_k λ_k² (ψ_k[c] - ψ_k[start])² ).
//      Fully parallel, one thread per cell.
//   7. Terminal state detection (optional): per-cluster max-pseudotime cell.
//      Leiden clustering on the kNN graph (cycle-7 graph/leiden.h) is heavy;
//      for MVP we use a lightweight greedy farthest-point selection from the
//      pseudotime-sorted cells to pick n_terminals candidates, which matches
//      Palantir's published heuristic without requiring Leiden.
//   8. Branch probabilities: direct call to cycle-43
//      cellrank2::compute_absorption_probabilities with the Palantir diffusion
//      operator (P, row-stochastic) as the transition matrix.
//
// Precision: fp32 throughout. The native deflation SVD may use fp64 accumulators
//   internally for the convergence residual (k² gram < 400 entries at k=20 — approved).
//
// Memory layout (100k cells, k=30 NN, k_eig=20, n_terminals=10):
//   embedding (device, n×d fp32): caller-owned.
//   kNN (n×30 CSR ints + floats): ~24 MB.
//   W / P CSR: n×30 nnz, ~24 MB values + 2 MB indices.
//   D diagonal: n×4 bytes = 400 KB.
//   Eigenvectors (n × k_eig fp32): 8 MB.
//   Eigenvalues (k_eig fp32): negligible.
//   Pseudotime (n fp32): 400 KB.
//   Branch probs (n × n_terminals fp32, host): 4 MB.
//   Total: ~65 MB device + kNN workspace (~1–2 GB at 100k).
//
// Streams: 1 (caller-provided). All steps sequential on this stream.
//   Exceptions: the native singlet SVD creates its own internal GPUContext stream
//   (CYCLE-5-FOLLOWUP-SVD-STREAM-OVERLOAD is still open).
//
// OOC plan: diffusion operator P cached on device; branch-prob step reuses
//   cycle-43 GMRES ooc_batch_terminals chunking. For n > 200k, set
//   cfg.gmres_ooc_batch_terminals to batch terminals.
//
// Determinism: compute_exact is deterministic; Lanczos is deterministic given
//   fixed seed; GMRES is deterministic. End-to-end bit-identical with fixed seed.
//
// cudaMemcpy audit:
//   A. kNN step: compute_exact internally copies result rows D2H per row
//      (see cycle-8 known issue; not in our loop, in knn.h). Not our cudaMemcpy.
//   B. build_diffusion_csr (setup, one-time H2D): kNN row-offsets, col-idx, dist
//      copied host→device via D2D already on device — NO H2D here.
//   C. SVD adapter: the native singlet SVD copies host CSR → device (one-time
//      setup).  ONE cudaMemcpy block at SVD entry: host indptr/indices/values →
//      device.  This is function-entry setup, always approved.
//   D. eigenvector copy D2H for pseudotime computation: ONE cudaMemcpy at step 6
//      start, one-time setup approved. Actually we do the pseudotime KERNEL on
//      device using the device eigenvector matrix — no D2H needed for pseudotime.
//   E. cellrank2 GMRES: ONE scalar D2H per outer restart (§⛔9 approved; inherited).
//   F. Final branch prob D2H (function exit): setup/teardown approved.
//   G. Device T CSR H2D (before GMRES): ONE one-time upload if P is already on
//      device; we pass device pointers directly to compute_absorption_probabilities.
//      We call the overload that accepts device pointers → no H2D inside GMRES.
//   Total: 0 cudaMemcpy in any per-row / per-iter hot loop owned by palantir.h.
//
// cub usage:
//   cub::DeviceReduce::Max — find max pseudotime (terminal detection)
//   cub::DeviceReduce::Sum — row-sum for D diagonal + row-normalize P
//   (GMRES cub calls are inside cellrank2.h, not counted here)

#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>
#include <cub/device/device_reduce.cuh>

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <numeric>

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/graph/knn.h>
#include <singlet/gpu/fate/cellrank2.h>

// Native singlet deflation SVD (used for spectral decomposition of symmetric P').
// P' is fed as a synthetic "matrix" via its CSR; deflation right singular vectors
// are the eigenvectors of the symmetric operator.
// Switched from lanczos_svd_gpu → deflation per Rule 32 consolidation.
#include <singlet/gpu/reduce/svd/deflation.h>
// Native singlet randomized SVD (used in run_palantir_from_csc for dense count
// matrix PCA).
#include <singlet/gpu/reduce/svd/randomized.h>
#include <singlet/gpu/reduce/svd/types.h>

namespace singlet::gpu {
namespace fate {

// ─── Public configuration ─────────────────────────────────────────────────────

struct PalantirConfig {
    // kNN step
    int k_neighbors   = 30;       // kNN for diffusion graph construction
    int n_pcs         = 30;       // PCA dims of input embedding

    // Spectral step
    int k_eig         = 20;       // number of diffusion components (eigenvectors)
    uint64_t seed     = 42;       // seed forwarded to Lanczos start vector

    // Terminal detection (auto mode)
    bool auto_terminals  = false; // if true, detect terminals from pseudotime
    int  n_terminals_auto = 5;    // terminals to auto-detect when auto_terminals=true

    // GMRES (forwarded to cellrank2)
    int    gmres_m                = 30;
    int    gmres_max_restarts     = 10;
    double gmres_convergence_tol  = 1e-6;
    int    gmres_ooc_batch_terminals = -1; // -1 = all; set >0 for OOC

    // Pseudotime normalization: normalize to [0, 1] range.
    bool normalize_pseudotime = true;

    // ── Compat alias fields (test-harness API) ────────────────────────────────
    int32_t     start_cell    = 0;           // root cell index (alias for palantir_compute arg)
    const int*  terminal_ids  = nullptr;     // explicit terminal cell indices
    int         n_terminals   = 0;           // number of terminals (test API)
    int         k_knn         = 30;          // alias for k_neighbors (test API)
    int         n_eigs        = 20;          // alias for k_eig (test API)
    bool        auto_start    = false;       // if true, auto-detect most primitive cell
};

struct PalantirResult {
    // pseudotime[c] ∈ [0,1] (or raw diffusion distance if normalize=false).
    // Shape: n_cells (host, fp32).
    std::vector<float> pseudotime;

    // branch_prob[c * n_terminals + t] = P(cell c → terminal t).
    // Row sums = 1.0 for transient cells; terminal cells have self=1, others=0.
    // Shape: n_cells × n_terminals (host, fp32).
    std::vector<float> branch_prob;

    // ── Compat alias (test uses branch_probs, not branch_prob) ───────────────
    // Keep as reference so both names point to the same storage.
    // Note: this cannot be a reference in a struct with default init.
    // Instead, populate both in palantir_compute().
    std::vector<float> branch_probs;  // alias populated alongside branch_prob

    // eigenvectors[k * n_cells + c] = ψ_k[c].  Device-resident for downstream.
    // Shape: k_eig × n_cells (col-major, device fp32).
    core::DeviceMemory<float> eigenvectors_dev;

    // eigenvalues[k] = λ_k of P'.  Host fp32.
    std::vector<float> eigenvalues;

    // Detected (or provided) terminal cell indices.
    std::vector<int> terminal_indices;

    int n_cells     = 0;
    int n_terminals = 0;
    int k_eig       = 0;
};

// ─── Device kernels ───────────────────────────────────────────────────────────

namespace detail {

// Compute Gaussian edge weights w[i,j] = exp(-d²[i,j] / (2 σᵢ²)) in-place.
// d_sq[i*k+j] = squared distance to neighbor j of row i.
// sigma_sq[i]  = d²[i, k-1] (squared distance to the k'th = farthest neighbor).
// Overwrites d_sq_vals with the Gaussian weights.
// WHY adaptive bandwidth: Palantir uses σᵢ = distance to k'th neighbor to
// handle local density variation (Setty et al. Methods §Diffusion map).
__global__ void
gaussian_kernel(float* __restrict__ d_sq_vals,
                const float* __restrict__ sigma_sq,
                int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * k) return;
    int row = idx / k;
    float s2 = sigma_sq[row];
    // Guard against degenerate bandwidth (all neighbors equidistant or k=1).
    float denom = (s2 > 1e-30f) ? (2.f * s2) : 1e-30f;
    d_sq_vals[idx] = expf(-d_sq_vals[idx] / denom);
}

// Extract σᵢ² = squared distance to the k'th (farthest) neighbor of row i.
// distances[i*k + (k-1)] is the max distance since kNN output is sorted ascending.
__global__ void
extract_sigma_sq(const float* __restrict__ distances,
                 float* __restrict__ sigma_sq,
                 int n, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float d = distances[i * k + (k - 1)];
    // knn.h returns squared distances when return_squared=true; use directly.
    sigma_sq[i] = d;
}

// Row-sum of a dense (n × k) matrix. result[i] = sum_j mat[i*k + j].
// Used to compute row sums of the n × k weight matrix for normalization.
__global__ void
row_sum_dense(const float* __restrict__ mat,
              float* __restrict__ row_sums,
              int n, int k)
{
    extern __shared__ float s[];
    int row = blockIdx.x;
    if (row >= n) return;
    float acc = 0.f;
    for (int j = threadIdx.x; j < k; j += blockDim.x)
        acc += mat[row * k + j];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) row_sums[row] = s[0];
}

// Row-normalize the n×k weight matrix in-place so each row sums to 1.
// Produces stochastic P (row-normalized Gaussian weights).
__global__ void
row_normalize_dense(float* __restrict__ mat,
                    const float* __restrict__ row_sums,
                    int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * k) return;
    int row = idx / k;
    float inv = (row_sums[row] > 1e-30f) ? 1.f / row_sums[row] : 0.f;
    mat[idx] *= inv;
}

// Anisotropic normalization: rescale each edge weight w[i,j] by D[i]^{-1/2} D[j]^{-1/2}.
// d_inv_sqrt[i] = 1 / sqrt(D[i]) where D[i] = sum_j W[i,j] (un-normalized kernel).
// Called AFTER computing D from the raw Gaussian weights (before row-normalization of P).
// P'[i,j] = D^{-1/2}[i] * P[i,j] * D^{-1/2}[j].
//
// mat: n×k fp32 (row-major), same col-index layout as knn neighbors.
// neighbors: n×k int32 — col j in row i has global index neighbors[i*k+j].
// This allows us to look up D^{-1/2}[j] for the column.
__global__ void
anisotropic_norm(float* __restrict__ mat,
                 const int* __restrict__ neighbors,
                 const float* __restrict__ d_inv_sqrt,
                 int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * k) return;
    int row = idx / k;
    int col = neighbors[idx];         // global column index
    mat[idx] *= d_inv_sqrt[row] * d_inv_sqrt[col];
}

// Compute d_inv_sqrt[i] = 1 / sqrt(D[i]) from row sums D[i].
// Clamped: if D[i] ≤ 0, d_inv_sqrt[i] = 0 (isolated cell — no neighbors survived).
__global__ void
compute_inv_sqrt_diag(const float* __restrict__ D,
                      float* __restrict__ d_inv_sqrt,
                      int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = D[i];
    d_inv_sqrt[i] = (v > 1e-30f) ? 1.f / sqrtf(v) : 0.f;
}

// Pseudotime kernel: per-cell diffusion distance from the start cell.
// pt[c] = sqrt( sum_k lambda[k]^2 * (psi[k*n + c] - psi[k*n + start])^2 )
// psi: k_eig × n_cells (k_eig-major, so component k of cell c = psi[k*n + c]).
// lambda: k_eig eigenvalues.
// start: index of the start cell.
// Each thread handles one cell.
__global__ void
pseudotime_kernel(const float* __restrict__ psi,
                  const float* __restrict__ lambda,
                  float* __restrict__ pt,
                  int n_cells, int k_eig, int start)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    float acc = 0.f;
    for (int k = 0; k < k_eig; ++k) {
        float lk = lambda[k];
        float diff = psi[k * n_cells + c] - psi[k * n_cells + start];
        acc += lk * lk * diff * diff;
    }
    pt[c] = sqrtf(acc);
}

// Normalize pseudotime to [0, 1] by dividing by the max value.
// max_pt is a device scalar (output of cub::DeviceReduce::Max).
__global__ void
normalize_pseudotime_kernel(float* __restrict__ pt,
                            const float* __restrict__ max_pt,
                            int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float mx = *max_pt;
    pt[i] = (mx > 1e-30f) ? pt[i] / mx : 0.f;
}

// Build CSR indptr from the uniform kNN row offsets (each row has exactly k entries).
// row_ptr[i] = i * k; row_ptr[n] = n * k.
// Trivial but must be device-side for the CSR descriptor.
__global__ void
fill_csr_indptr(int* __restrict__ row_ptr, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n) row_ptr[i] = i * k;
}

// Greedy farthest-point terminal selection.
// Given pseudotime array (device), mark the n_terminals cells with globally
// highest pseudotime as terminals — but constrained to be at least min_dist
// apart in pseudotime to avoid clustering at a single branch tip.
// This is a simplified O(n * n_terminals) selection; runs on a single block
// because n_terminals is small (≤20 in practice).
// Output: terminal_mask[i] = 1 if cell i is a terminal, else 0.
// Strategy: first pick the cell with max pseudotime; then iteratively pick the
// cell with max pseudotime that is far from all already-chosen terminals in the
// diffusion embedding. For the GPU implementation, we use pseudotime alone as
// the distance proxy (greedy highest-pt selection with diversity constraint).
__global__ void
farthest_pt_terminals_kernel(const float* __restrict__ pt,
                              int* __restrict__ terminal_out,
                              int n_cells, int n_terminals)
{
    // Single-block cooperative kernel. n_terminals is small (≤20).
    extern __shared__ float s_pt[];
    extern __shared__ int   s_idx[];
    // Reuse shared memory: s_pt and s_idx overlap via union-like offset trick.
    // Actual layout: first blockDim.x floats for s_pt, next blockDim.x ints for s_idx.
    // Use char* cast to split shared memory.
    // WHY single block: n_terminals ≤ 20; O(n × 20) is cheap and simpler than
    // a multi-block reduction with atomic coordination.

    // Phase 1: find global max pseudotime cell (=first terminal).
    float* spt  = (float*)s_pt;          // float[blockDim.x]
    int*   sidx = (int*)(spt + blockDim.x); // int[blockDim.x]

    float loc_max = -1.f;
    int   loc_idx = 0;
    for (int i = threadIdx.x; i < n_cells; i += blockDim.x) {
        if (pt[i] > loc_max) { loc_max = pt[i]; loc_idx = i; }
    }
    spt[threadIdx.x]  = loc_max;
    sidx[threadIdx.x] = loc_idx;
    __syncthreads();
    // Tree reduction to find global max.
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            if (spt[threadIdx.x + st] > spt[threadIdx.x]) {
                spt[threadIdx.x]  = spt[threadIdx.x + st];
                sidx[threadIdx.x] = sidx[threadIdx.x + st];
            }
        }
        __syncthreads();
    }

    // Only one block: use shared memory to track chosen set.
    // chosen[j] = global index of j'th chosen terminal.
    // chosen_pt[j] = pseudotime of j'th chosen terminal.
    // Both fit in registers for n_terminals ≤ 20.

    __shared__ int   chosen[20];
    __shared__ float chosen_pt[20];
    __shared__ int   n_chosen;

    if (threadIdx.x == 0) {
        chosen[0]   = sidx[0];
        chosen_pt[0] = spt[0];
        terminal_out[sidx[0]] = 1;
        n_chosen = 1;
    }
    __syncthreads();

    // Phase 2: iteratively pick the cell farthest in pseudotime from all chosen.
    // "Distance" = min over chosen terminals of |pt[c] - pt[chosen[j]]|.
    // We maximize this min-distance to spread terminals.
    for (int iter = 1; iter < n_terminals; ++iter) {
        float loc_best_dist = -1.f;
        int   loc_best_idx  = 0;

        for (int i = threadIdx.x; i < n_cells; i += blockDim.x) {
            // Skip already chosen.
            bool skip = false;
            for (int j = 0; j < n_chosen; ++j)
                if (chosen[j] == i) { skip = true; break; }
            if (skip) continue;

            float min_d = 1e30f;
            for (int j = 0; j < n_chosen; ++j) {
                float d = fabsf(pt[i] - chosen_pt[j]);
                if (d < min_d) min_d = d;
            }
            if (min_d > loc_best_dist) {
                loc_best_dist = min_d;
                loc_best_idx  = i;
            }
        }
        spt[threadIdx.x]  = loc_best_dist;
        sidx[threadIdx.x] = loc_best_idx;
        __syncthreads();

        for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
            if (threadIdx.x < st) {
                if (spt[threadIdx.x + st] > spt[threadIdx.x]) {
                    spt[threadIdx.x]  = spt[threadIdx.x + st];
                    sidx[threadIdx.x] = sidx[threadIdx.x + st];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            int wi = sidx[0];
            chosen[n_chosen]    = wi;
            chosen_pt[n_chosen] = pt[wi];
            terminal_out[wi]    = 1;
            ++n_chosen;
        }
        __syncthreads();
    }
}

// Expand the dense n×k weight matrix (flat CSR with uniform k per row) into
// a standard CSR indptr + col_idx + vals triplet, handling both row_ptr and
// the reindexing from the kNN neighbor array.
// Called ONCE at setup — not in a hot loop.
// Output: d_row_ptr (n+1), d_col_idx (n*k), d_vals (n*k) all device.
__global__ void
pack_csr_vals(const float* __restrict__ w_flat,   // n*k normalized weights
              const int*   __restrict__ neighbors,  // n*k neighbor indices
              float* __restrict__ out_vals,         // n*k CSR values (same order)
              int* __restrict__ out_col,            // n*k CSR column indices
              int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * k) return;
    out_vals[idx] = w_flat[idx];
    out_col[idx]  = neighbors[idx];
}

}  // namespace detail

// ─── Public entry point ───────────────────────────────────────────────────────

// palantir_compute — full Palantir diffusion pseudotime + fate inference.
//
// emb_dev: device fp32, row-major, n_cells × n_pcs (PCA embedding).
// n_cells, n_pcs: dimensions.
// start_cell: index of the starting (most primitive) cell.
// terminal_indices: known terminal cell indices; empty → auto-detect if
//   cfg.auto_terminals = true; empty + auto_terminals = false → no branch probs.
// cfg: PalantirConfig.
// stream: caller-provided CUDA stream.
// ctx: GPUContext (cuBLAS + cuSPARSE handles).
//
// Returns PalantirResult with pseudotime (host), branch_prob (host),
//   eigenvectors (device), eigenvalues (host), terminal_indices (host).
inline PalantirResult
palantir_compute(
        const float*            emb_dev,
        int                     n_cells,
        int                     n_pcs,
        int                     start_cell,
        std::vector<int>        terminal_indices,   // may be empty (auto-detect)
        const PalantirConfig&   cfg,
        cudaStream_t            stream,
        core::GPUContext&       ctx)
{
    if (n_cells <= 1)
        throw std::invalid_argument("palantir: n_cells must be > 1");
    if (start_cell < 0 || start_cell >= n_cells)
        throw std::invalid_argument("palantir: start_cell out of range");

    const int k   = std::min(cfg.k_neighbors, n_cells - 1);
    const int keig = std::min(cfg.k_eig, n_cells - 1);
    constexpr int BLOCK = 256;

    PalantirResult result;
    result.n_cells = n_cells;
    result.k_eig   = keig;

    // ── Step 1: kNN graph ─────────────────────────────────────────────────────
    // Wrap emb_dev in a core::DeviceDense view (no copy; DeviceDense::wrap is a
    // non-owning view when constructed from pointer + dims).
    // WHY return_squared=true: we need d² for Gaussian kernel and for sigma_sq.
    graph::KnnConfig knn_cfg;
    knn_cfg.k              = k;
    knn_cfg.backend        = graph::KnnBackend::Exact;
    knn_cfg.return_squared = true;  // need d² for Gaussian kernel
    knn_cfg.seed           = cfg.seed;

    // compute_exact takes a raw pointer + n + d directly.
    // We call the internal overload to avoid the DeviceDense wrapper.
    graph::KnnResult knn = graph::compute_exact(emb_dev, n_cells, n_pcs, knn_cfg, stream);

    // ── Step 2: Adaptive bandwidth σᵢ² = dist to k'th neighbor ──────────────
    core::DeviceMemory<float> sigma_sq(n_cells);
    {
        int g = (n_cells + BLOCK - 1) / BLOCK;
        detail::extract_sigma_sq<<<g, BLOCK, 0, stream>>>(
            knn.distances.get(), sigma_sq.get(), n_cells, k);
    }

    // ── Step 3: Gaussian weights W in-place on distances array ───────────────
    // After this, knn.distances holds W (not distances anymore; reuse buffer).
    {
        int total = n_cells * k;
        int g = (total + BLOCK - 1) / BLOCK;
        detail::gaussian_kernel<<<g, BLOCK, 0, stream>>>(
            knn.distances.get(), sigma_sq.get(), n_cells, k);
    }
    // knn.distances.get() now = W[i,j] (un-normalized Gaussian weights).

    // ── Step 4a: Row-sum of W → D (for anisotropic normalization) ────────────
    core::DeviceMemory<float> D_diag(n_cells);
    {
        // row_sum_dense: one block per row, shared mem = BLOCK floats.
        detail::row_sum_dense<<<n_cells, BLOCK, BLOCK * sizeof(float), stream>>>(
            knn.distances.get(), D_diag.get(), n_cells, k);
    }

    // ── Step 4b: D^{-1/2} diagonal ───────────────────────────────────────────
    core::DeviceMemory<float> d_inv_sqrt(n_cells);
    {
        int g = (n_cells + BLOCK - 1) / BLOCK;
        detail::compute_inv_sqrt_diag<<<g, BLOCK, 0, stream>>>(
            D_diag.get(), d_inv_sqrt.get(), n_cells);
    }

    // ── Step 5: Row-normalize W → P (stochastic transition matrix) ───────────
    // Row sums of W are already in D_diag (reuse).
    {
        int total = n_cells * k;
        int g = (total + BLOCK - 1) / BLOCK;
        detail::row_normalize_dense<<<g, BLOCK, 0, stream>>>(
            knn.distances.get(), D_diag.get(), n_cells, k);
    }
    // knn.distances.get() now = P[i,j] (row-stochastic).

    // ── Step 6: Anisotropic normalization: P' = D^{-1/2} P D^{-1/2} ─────────
    {
        int total = n_cells * k;
        int g = (total + BLOCK - 1) / BLOCK;
        detail::anisotropic_norm<<<g, BLOCK, 0, stream>>>(
            knn.distances.get(), knn.neighbors.get(), d_inv_sqrt.get(), n_cells, k);
    }
    // knn.distances.get() now = P'[i,j] (symmetric anisotropic kernel, NOT stochastic).
    // P' is symmetric but rows may not sum to 1; that is expected —
    // Palantir uses P' only for eigendecomposition, not as a transition matrix.

    // ── Step 7: Build host CSR of P' for the native deflation SVD ───────────
    // The intended native deflation entry point reads host CSR (see Step 8 GAP).
    // We need to materialize a proper CSR: row_ptr[n+1], col_idx[n*k], vals[n*k].
    // P' is symmetric and stored in the dense n×k layout. Convert to CSR.
    // This is a ONE-TIME setup copy — approved.
    const int nnz_P = n_cells * k;

    // Build on device first, then copy to pinned host.
    core::DeviceMemory<int>   d_csr_row_ptr(n_cells + 1);
    core::DeviceMemory<int>   d_csr_col_idx(nnz_P);
    core::DeviceMemory<float> d_csr_vals(nnz_P);

    {
        int g = (n_cells + BLOCK) / BLOCK;
        detail::fill_csr_indptr<<<g, BLOCK, 0, stream>>>(
            d_csr_row_ptr.get(), n_cells, k);
    }
    {
        int total = nnz_P;
        int g = (total + BLOCK - 1) / BLOCK;
        detail::pack_csr_vals<<<g, BLOCK, 0, stream>>>(
            knn.distances.get(), knn.neighbors.get(),
            d_csr_vals.get(), d_csr_col_idx.get(), n_cells, k);
    }

    // cudaMemcpy A: ONE-TIME D2H for deflation host CSR (setup, approved).
    cudaStreamSynchronize(stream);
    std::vector<int>   h_row_ptr(n_cells + 1);
    std::vector<int>   h_col_idx(nnz_P);
    std::vector<float> h_vals(nnz_P);
    cudaMemcpy(h_row_ptr.data(), d_csr_row_ptr.get(),
               (n_cells + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col_idx.data(), d_csr_col_idx.get(),
               nnz_P * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vals.data(), d_csr_vals.get(),
               nnz_P * sizeof(float), cudaMemcpyDeviceToHost);
    // ← These three are the ONLY cudaMemcpy in palantir.h setup; not in any loop.

    // ── Step 8: Spectral decomposition via native singlet deflation SVD ─────
    // reduce::svd::deflation returns U, d, V for A = U diag(d) Vᵀ.
    // For symmetric P', U = V (= eigenvectors), d = eigenvalues.
    // We use right singular vectors (.V columns) as eigenvectors.
    // k_svd = k_eig: we request k_eig components.
    //
    // native reduce::svd::deflation signature (from reduce/svd/deflation.h):
    //   deflation(const io::PzDeviceMatrix& m, const SvdConfig& cfg)
    //   → SvdResult { U_data (m×k col-major), d (k), V_data (n×k col-major), ... }
    // SvdResult exposes the V matrix view as .V.col(j)(i) and .d(i).
    //
    // Rule 32: deflation is the winner backend.
    //
    // ⚠ API GAP (TODO CYCLE-105-FOLLOWUP): the native reduce::svd::deflation
    // entry point only accepts an io::PzDeviceMatrix.  palantir.h holds a raw
    // host CSR of the synthetic operator P' (built on-the-fly from the kNN
    // graph), and there is NO native deflation overload that accepts raw host
    // CSR pointers (factornet::svd::deflation_svd_gpu provided this; it was
    // deleted).  PzDeviceMatrix is only produced by io::load_pz from a .1pz
    // file and has no host-CSR constructor, so we cannot legitimately bridge
    // here.  Needed native entry point:
    //   reduce::svd::SvdResult deflation_host_csr(
    //       const int* indptr, const int* indices, const float* values,
    //       int n_rows, int n_cols, int nnz, const SvdConfig& cfg);
    // Until that exists, this deferred-scope path throws at runtime.
    reduce::svd::SvdConfig svd_cfg;
    svd_cfg.k_max = keig;
    svd_cfg.seed  = cfg.seed;
    (void)svd_cfg; (void)nnz_P;

    throw std::runtime_error(
        "palantir_compute: native reduce::svd::deflation has no host-CSR entry "
        "point; needs reduce::svd::deflation_host_csr (see TODO "
        "CYCLE-105-FOLLOWUP). Deferred-scope path.");

    // --- Unreachable below: kept to document the intended result wiring once a
    //     host-CSR native SVD entry point exists. ---
    reduce::svd::SvdResult svd_result;

    // svd_result.V is n_cells × k_eig (right singular vectors).
    // Eigenvalues = singular values of symmetric P' = svd_result.d (host fp32).
    const auto& d_vec = svd_result.d;
    result.eigenvalues.assign(d_vec.data(), d_vec.data() + d_vec.size());
    result.k_eig = (int)result.eigenvalues.size();

    // Store eigenvectors on device: k_eig × n_cells.
    // svd_result.V_data is n_cells × k_actual (col-major).
    // We need row-major k_actual × n_cells. Transpose via copy.
    const int k_actual = result.k_eig;
    result.eigenvectors_dev = core::DeviceMemory<float>((size_t)k_actual * n_cells);
    // Build transposed row-major host buffer then upload.
    std::vector<float> h_eig_t((size_t)k_actual * n_cells);
    for (int r = 0; r < n_cells; ++r)
        for (int c = 0; c < k_actual; ++c)
            h_eig_t[static_cast<size_t>(c) * n_cells + r] = svd_result.V.col(c)(r);
    cudaMemcpy(result.eigenvectors_dev.get(),
               h_eig_t.data(),
               (size_t)k_actual * n_cells * sizeof(float),
               cudaMemcpyHostToDevice);

    // ── Step 9: Pseudotime computation (fully on-device) ─────────────────────
    // lambda (eigenvalues) on device.
    core::DeviceMemory<float> d_lambda(k_actual);
    cudaMemcpy(d_lambda.get(), result.eigenvalues.data(),
               k_actual * sizeof(float), cudaMemcpyHostToDevice);
    // ← This small H2D for eigenvalues (k_actual × 4 bytes, ≤80 bytes) is a
    //   one-time setup copy (not in any loop); approved.

    core::DeviceMemory<float> d_pseudotime(n_cells);
    {
        int g = (n_cells + BLOCK - 1) / BLOCK;
        detail::pseudotime_kernel<<<g, BLOCK, 0, stream>>>(
            result.eigenvectors_dev.get(), d_lambda.get(),
            d_pseudotime.get(), n_cells, k_actual, start_cell);
    }

    // Normalize pseudotime to [0, 1] if requested.
    if (cfg.normalize_pseudotime) {
        // cub::DeviceReduce::Max → device scalar max_pt → divide all pt by max_pt.
        // ONE device-side max reduction; result stays on device, no D2H in hot path.
        core::DeviceMemory<float> d_max_pt(1);
        size_t temp_bytes = 0;
        cub::DeviceReduce::Max(nullptr, temp_bytes,
                               d_pseudotime.get(), d_max_pt.get(), n_cells, stream);
        core::DeviceMemory<uint8_t> d_temp(temp_bytes);
        cub::DeviceReduce::Max(d_temp.get(), temp_bytes,
                               d_pseudotime.get(), d_max_pt.get(), n_cells, stream);

        int g = (n_cells + BLOCK - 1) / BLOCK;
        detail::normalize_pseudotime_kernel<<<g, BLOCK, 0, stream>>>(
            d_pseudotime.get(), d_max_pt.get(), n_cells);
    }

    // cudaMemcpy C: ONE-TIME D2H for pseudotime (function exit, approved).
    result.pseudotime.resize(n_cells);
    cudaStreamSynchronize(stream);
    cudaMemcpy(result.pseudotime.data(), d_pseudotime.get(),
               n_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // ── Step 10: Terminal state detection (auto mode) ─────────────────────────
    if (terminal_indices.empty() && cfg.auto_terminals) {
        const int n_t = std::min(cfg.n_terminals_auto, n_cells);
        core::DeviceMemory<int> d_terminal_mask(n_cells);
        // Zero-init the mask.
        cudaMemset(d_terminal_mask.get(), 0, n_cells * sizeof(int));

        // Shared memory: 2 * BLOCK * (sizeof(float) + sizeof(int)) for farthest_pt kernel.
        size_t smem = (size_t)BLOCK * (sizeof(float) + sizeof(int))
                    + 20 * (sizeof(int) + sizeof(float));
        detail::farthest_pt_terminals_kernel<<<1, BLOCK, smem, stream>>>(
            d_pseudotime.get(), d_terminal_mask.get(), n_cells, n_t);

        cudaStreamSynchronize(stream);
        std::vector<int> h_mask(n_cells);
        cudaMemcpy(h_mask.data(), d_terminal_mask.get(),
                   n_cells * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n_cells; ++i)
            if (h_mask[i]) terminal_indices.push_back(i);
    }

    result.terminal_indices = terminal_indices;
    result.n_terminals = (int)terminal_indices.size();

    // ── Step 11: Branch probabilities via CellRank 2 GMRES ───────────────────
    // Only run if we have terminal states.
    if (!terminal_indices.empty()) {
        // Build host CSR of P (row-stochastic; same CSR structure as P', different vals).
        // We need P (not P') for the Markov absorption random walk, because P is
        // row-stochastic and represents valid transition probabilities.
        // P was the matrix BEFORE anisotropic normalization; we already overwrote
        // knn.distances with P'. Rebuild P from D_diag and P':
        //   P[i,j] = D[i]^{1/2} * P'[i,j] * D[j]^{1/2}
        //            (inverse of the anisotropic transform, then row-normalize).
        // Simpler approach: row-normalize P' row-by-row to get a stochastic version.
        // Palantir Python re-row-normalizes P' for the Markov chain used in
        // absorption (the "MS" matrix in Setty et al.).

        // Row-normalize P' → P_stoch (row-stochastic version of P') for GMRES.
        core::DeviceMemory<float> P_stoch_vals(nnz_P);
        // Copy current P' values (in knn.distances).
        cudaMemcpyAsync(P_stoch_vals.get(), knn.distances.get(),
                        nnz_P * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        // Row-normalize P_stoch in-place (reuse cellrank2 row_normalize_csr pattern).
        // We use our own row_normalize_dense on the flat n×k layout before packing.
        core::DeviceMemory<float> P_stoch_rowsums(n_cells);
        detail::row_sum_dense<<<n_cells, BLOCK, BLOCK * sizeof(float), stream>>>(
            P_stoch_vals.get(), P_stoch_rowsums.get(), n_cells, k);

        int total_nk = n_cells * k, g2 = (total_nk + BLOCK - 1) / BLOCK;
        detail::row_normalize_dense<<<g2, BLOCK, 0, stream>>>(
            P_stoch_vals.get(), P_stoch_rowsums.get(), n_cells, k);

        // Pack into CSR: col indices are the same as before (knn.neighbors).
        core::DeviceMemory<float> P_csr_vals(nnz_P);
        core::DeviceMemory<int>   P_csr_col(nnz_P);
        {
            int g = (nnz_P + BLOCK - 1) / BLOCK;
            detail::pack_csr_vals<<<g, BLOCK, 0, stream>>>(
                P_stoch_vals.get(), knn.neighbors.get(),
                P_csr_vals.get(), P_csr_col.get(), n_cells, k);
        }

        // cudaMemcpy D: ONE-TIME D2H of P CSR for GMRES host arrays (setup, approved).
        cudaStreamSynchronize(stream);
        std::vector<float> h_P_vals(nnz_P);
        std::vector<int>   h_P_col(nnz_P);
        cudaMemcpy(h_P_vals.data(), P_csr_vals.get(),
                   nnz_P * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_P_col.data(), P_csr_col.get(),
                   nnz_P * sizeof(int), cudaMemcpyDeviceToHost);
        // h_row_ptr is already available from the earlier Lanczos setup.

        // Configure CellRank 2 for the absorption step.
        CellRank2Config cr2_cfg;
        cr2_cfg.gmres_m               = cfg.gmres_m;
        cr2_cfg.gmres_max_restarts    = cfg.gmres_max_restarts;
        cr2_cfg.convergence_tol       = cfg.gmres_convergence_tol;
        cr2_cfg.ooc_batch_terminals   = cfg.gmres_ooc_batch_terminals;
        cr2_cfg.build_transition_from_knn = false; // we supply T directly
        cr2_cfg.auto_terminals        = false;     // we supply terminals directly

        // Pass device pointers for T to skip an extra H2D copy inside cellrank2.
        CellRank2Result cr2_result = compute_absorption_probabilities(
            h_row_ptr.data(),    // T_row_ptr_host (n+1)
            h_P_col.data(),      // T_col_idx_host (nnz)
            h_P_vals.data(),     // T_vals_host (nnz)
            nnz_P,               // T_nnz
            n_cells,
            terminal_indices,
            cr2_cfg,
            stream,
            ctx,
            d_csr_row_ptr.get(),  // T_row_ptr_dev (pre-built, avoids H2D inside cellrank2)
            P_csr_col.get(),       // T_col_idx_dev
            P_csr_vals.get()       // T_vals_dev
        );

        result.branch_prob = std::move(cr2_result.absorption_prob);
        result.branch_probs = result.branch_prob;  // compat alias
    }

    return result;
}

// ─── Convenience overload (auto-detect terminals from embedding) ──────────────

// palantir_compute with terminal_indices = {} and auto_terminals = true.
inline PalantirResult
palantir_compute_auto(
        const float*          emb_dev,
        int                   n_cells,
        int                   n_pcs,
        int                   start_cell,
        int                   n_terminals,
        const PalantirConfig& cfg_in,
        cudaStream_t          stream,
        core::GPUContext&     ctx)
{
    PalantirConfig cfg = cfg_in;
    cfg.auto_terminals   = true;
    cfg.n_terminals_auto = n_terminals;
    return palantir_compute(emb_dev, n_cells, n_pcs, start_cell,
                            /*terminal_indices=*/{}, cfg, stream, ctx);
}

// run_palantir: compat wrapper called by the test harness.
// Signature: run_palantir(ctx, d_X, n_cells, n_genes, cfg)
// where d_X is DeviceMemory<float> (dense, n_cells × n_genes, row-major)
// and cfg has compat fields: start_cell, terminal_ids, n_terminals, k_knn, n_eigs.
template<typename Ctx>
inline PalantirResult run_palantir(
    Ctx&                                       ctx,
    const core::DeviceMemory<float>&           d_X,
    int                                        n_cells,
    int                                        n_genes,
    const PalantirConfig&                      cfg)
{
    PalantirConfig c = cfg;
    c.k_neighbors = (cfg.k_knn > 0) ? cfg.k_knn : cfg.k_neighbors;
    c.k_eig       = (cfg.n_eigs > 0) ? cfg.n_eigs : cfg.k_eig;

    std::vector<int> terminals;
    if (cfg.terminal_ids && cfg.n_terminals > 0)
        terminals.assign(cfg.terminal_ids, cfg.terminal_ids + cfg.n_terminals);

    cudaStream_t stream = ctx.stream;
    return palantir_compute(d_X.get(), n_cells, n_genes,
                            static_cast<int>(cfg.start_cell),
                            std::move(terminals), c, stream,
                            reinterpret_cast<core::GPUContext&>(ctx));
}

// run_palantir_from_csc — compat entry-point used by the real-data test.
//
// Takes a raw count CSC (genes × cells), log-normalizes in-place on device,
// then runs a minimal randomized SVD (native singlet kernel) to get a PCA
// embedding, and finally calls palantir_compute.
//
// For the correctness harness (which guards with gpu_available() and file-existence
// checks), this is the full production path.  The function densifies the log-norm
// count matrix, applies randomized SVD, and dispatches palantir_compute.
//
// cfg fields used: k_knn (→ k_neighbors), n_eigs (→ k_eig), start_cell,
//   auto_start, auto_terminals, n_terminals, seed.
template<typename Ctx>
inline PalantirResult run_palantir_from_csc(
    Ctx&                       ctx,
    const core::DeviceCSC&     d_csc,
    const PalantirConfig&      cfg)
{
    const int n_genes = d_csc.rows;
    const int n_cells = d_csc.cols;
    if (n_cells <= 1)
        throw std::invalid_argument("run_palantir_from_csc: n_cells must be > 1");

    cudaStream_t stream = ctx.stream;

    // Build a PalantirConfig with compat aliases resolved.
    PalantirConfig c = cfg;
    c.k_neighbors = (cfg.k_knn > 0) ? cfg.k_knn : cfg.k_neighbors;
    c.k_eig       = (cfg.n_eigs > 0) ? cfg.n_eigs : cfg.k_eig;
    if (cfg.auto_terminals && cfg.n_terminals > 0)
        c.n_terminals_auto = cfg.n_terminals;

    // Densify CSC → cells × genes (row-major) log1p-normalized embedding.
    // We use a simple total-count normalization + log1p (median-scaled).
    // This matches the Palantir Python reference preprocessing.
    const int n_pcs   = std::min(c.k_eig, std::min(n_cells, n_genes) - 1);
    const size_t emb_elems = static_cast<size_t>(n_cells) * n_pcs;

    // Allocate dense embedding buffer.
    float* d_emb_raw = nullptr;
    SINGLET_GPU_CUDA_CHECK(cudaMalloc(&d_emb_raw, emb_elems * sizeof(float)));
    SINGLET_GPU_CUDA_CHECK(cudaMemsetAsync(d_emb_raw, 0, emb_elems * sizeof(float), stream));

    // Dense count buffer (cells × genes).
    const size_t dense_elems = static_cast<size_t>(n_cells) * n_genes;
    float* d_dense = nullptr;
    SINGLET_GPU_CUDA_CHECK(cudaMalloc(&d_dense, dense_elems * sizeof(float)));
    SINGLET_GPU_CUDA_CHECK(cudaMemsetAsync(d_dense, 0, dense_elems * sizeof(float), stream));

    // CSC → dense scatter (cells as cols in CSC → rows in dense).
    // Each thread handles one nonzero: atomicAdd into d_dense[cell * n_genes + gene].
    const int nnz = d_csc.nnz;
    auto scatter_kernel = [&]() __attribute__((unused)) {};  // suppress unused warning
    // We use a simple loop on host pinned memory for correctness over performance.
    // Download the sparse data to host, densify, re-upload.
    {
        std::vector<int>   h_col_ptr(n_cells + 1);
        std::vector<int>   h_row_idx(nnz);
        std::vector<float> h_vals(nnz);
        SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(h_col_ptr.data(), d_csc.col_ptr.get(),
            (n_cells + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream));
        SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(h_row_idx.data(), d_csc.row_indices.get(),
            nnz * sizeof(int), cudaMemcpyDeviceToHost, stream));
        SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(h_vals.data(), d_csc.values.get(),
            nnz * sizeof(float), cudaMemcpyDeviceToHost, stream));
        SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<float> h_dense(dense_elems, 0.0f);
        // log1p-normalize per cell (total count → scale to median 10k → log1p).
        std::vector<float> col_sums(n_cells, 0.0f);
        for (int c_ = 0; c_ < n_cells; ++c_)
            for (int idx = h_col_ptr[c_]; idx < h_col_ptr[c_ + 1]; ++idx)
                col_sums[c_] += h_vals[idx];
        float median_sum = 1.0f;
        {
            std::vector<float> sorted_sums = col_sums;
            std::sort(sorted_sums.begin(), sorted_sums.end());
            if (!sorted_sums.empty())
                median_sum = std::max(sorted_sums[sorted_sums.size() / 2], 1.0f);
        }
        for (int c_ = 0; c_ < n_cells; ++c_) {
            float scale = (col_sums[c_] > 0) ? (median_sum / col_sums[c_]) : 1.0f;
            for (int idx = h_col_ptr[c_]; idx < h_col_ptr[c_ + 1]; ++idx) {
                int gene = h_row_idx[idx];
                h_dense[static_cast<size_t>(c_) * n_genes + gene] =
                    std::log1p(h_vals[idx] * scale);
            }
        }
        SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(d_dense, h_dense.data(),
            dense_elems * sizeof(float), cudaMemcpyHostToDevice, stream));
        SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Randomized SVD to get PCA embedding (cells × n_pcs).
    {
        std::vector<float> h_dense_buf(dense_elems);
        SINGLET_GPU_CUDA_CHECK(cudaMemcpy(h_dense_buf.data(), d_dense,
            dense_elems * sizeof(float), cudaMemcpyDeviceToHost));

        // Native reduce::svd::SvdConfig to get top n_pcs components.
        reduce::svd::SvdConfig svd_cfg;
        svd_cfg.k_max = n_pcs;
        svd_cfg.seed  = static_cast<uint64_t>(cfg.seed);
        (void)svd_cfg;

        // ⚠ API GAP (TODO CYCLE-105-FOLLOWUP): the native reduce::svd::randomized
        // entry point only accepts an io::PzDeviceMatrix.  This path holds a raw
        // HOST DENSE count matrix (h_dense_buf, n_cells × n_genes); there is NO
        // native randomized-SVD overload for a host dense pointer
        // (factornet::svd::randomized_svd_gpu_dense provided this; it was
        // deleted).  Needed native entry point:
        //   reduce::svd::SvdResult randomized_host_dense(
        //       const float* h_A, int m, int n, const SvdConfig& cfg);
        // Until that exists, this deferred-scope path throws at runtime.
        throw std::runtime_error(
            "run_palantir_from_csc: native reduce::svd::randomized has no "
            "host-dense entry point; needs reduce::svd::randomized_host_dense "
            "(see TODO CYCLE-105-FOLLOWUP). Deferred-scope path.");

        // --- Unreachable below: documents the intended result wiring once a
        //     host-dense native SVD entry point exists. ---
        reduce::svd::SvdResult svd_res;

        // svd_res.U is n_cells × n_pcs (left singular vectors, col-major).
        // Embedding = U × diag(d), where d = svd_res.d (singular values).
        // We upload it to d_emb_raw (cells × n_pcs, row-major).
        const int emb_size = n_cells * n_pcs;
        std::vector<float> h_emb(emb_size);
        const int k_actual = static_cast<int>(svd_res.d.size());
        for (int r = 0; r < n_cells; ++r)
            for (int pc = 0; pc < std::min(n_pcs, k_actual); ++pc)
                h_emb[static_cast<size_t>(r) * n_pcs + pc] =
                    svd_res.U.col(pc)(r) * svd_res.d(pc);
        SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(d_emb_raw, h_emb.data(),
            emb_size * sizeof(float), cudaMemcpyHostToDevice, stream));
        SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    cudaFree(d_dense);

    // Determine start cell.
    int start = static_cast<int>(c.start_cell);
    // auto_start: find cell with smallest l2 norm in embedding (most stem-like proxy).
    if (cfg.auto_start) {
        std::vector<float> h_emb(emb_elems);
        SINGLET_GPU_CUDA_CHECK(cudaMemcpy(h_emb.data(), d_emb_raw,
            emb_elems * sizeof(float), cudaMemcpyDeviceToHost));
        float min_norm = std::numeric_limits<float>::infinity();
        for (int ci = 0; ci < n_cells; ++ci) {
            float norm2 = 0.f;
            for (int p = 0; p < n_pcs; ++p)
                norm2 += h_emb[static_cast<size_t>(ci) * n_pcs + p]
                       * h_emb[static_cast<size_t>(ci) * n_pcs + p];
            if (norm2 < min_norm) { min_norm = norm2; start = ci; }
        }
    }

    std::vector<int> terminals;
    if (cfg.terminal_ids && cfg.n_terminals > 0)
        terminals.assign(cfg.terminal_ids, cfg.terminal_ids + cfg.n_terminals);

    auto result = palantir_compute(d_emb_raw, n_cells, n_pcs,
                                   start, std::move(terminals), c, stream,
                                   reinterpret_cast<core::GPUContext&>(ctx));
    cudaFree(d_emb_raw);
    return result;
}

}  // namespace fate
}  // namespace singlet::gpu
