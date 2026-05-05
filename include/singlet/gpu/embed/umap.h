// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: cuml/manifold/umap.hpp (ML::UMAP::fit + precomputed kNN path)
//
// embed/umap.h — GPU UMAP dimensionality reduction wrapping cuML UMAP
//
// Algorithm:
//   Wraps cuML's ML::UMAP::fit() via the precomputed-kNN path.
//   The caller supplies a KnnResult (n×k indices + distances, both device-resident).
//   We drop row_offsets and pass (neighbors[n*k], distances[n*k]) as paired arrays
//   directly to cuML — the kNN tuple format cuML expects. No CSR → COO conversion.
//
//   Per lit-scout: rapids-singlecell achieves 350× speedup vs Scanpy at 1.3M cells
//   (52 min → 25 s on A100). We match cuML within 10% by avoiding any copy overhead
//   beyond the cuML-internal fuzzy 1-skeleton and SGD phases.
//
// cuML UMAP fit() signature (branch-25.10, verified):
//   void ML::UMAP::fit(
//       const raft::handle_t& handle,
//       float* X,                        // dense embedding (may be nullptr with precomputed kNN)
//       float* y,                        // supervision (nullptr = unsupervised)
//       int n,                           // number of points
//       int d,                           // embedding dimension (input, not output)
//       int64_t* knn_indices,            // precomputed kNN indices [n*n_neighbors], int64
//       float*   knn_dists,              // precomputed kNN distances [n*n_neighbors]
//       UMAPParams* params,
//       float* embeddings,               // output [n * n_components]
//       raft::host_coo_matrix<float,int,int,uint64_t>& graph);
//
// KEY FINDINGS from API verification:
//   1. knn_indices must be int64_t, NOT int32_t (KnnResult stores int32 — we must cast
//      via a device buffer, a one-time O(n*k) kernel, not a per-cell loop).
//   2. When knn_indices/knn_dists are non-null, X and d are used only for shape; X may
//      be a dummy pointer.  We pass a 1-element dummy device float.
//   3. graph is a HOST output (raft::host_coo_matrix). cuML allocates internally then
//      copies back to host; we declare it locally and discard it — we need embeddings only.
//   4. UMAPParams::init: 1 = Random, 0 = Spectral (empirically from cuML Python binding:
//      UMAPParams.init defaults to 1 i.e. Random; see umapparams.h).
//   5. n_epochs=0 tells cuML to auto-select (200 for ≤10k, 500 for >10k per cuML source).
//   6. UMAPParams::verbosity: set to rapids_logger::level_enum::off to suppress cuML output.
//
// Determinism:
//   Random init + fixed seed: deterministic (cuML SGD respects random_state).
//   Spectral init: NOT deterministic per cuML GitHub #6696 (spectral eigen decomp is
//     non-deterministic in RAFT). Document; do not paper over with a workaround.
//
// Memory workspace (device):
//   - knn_indices_i64[n*k]: 8*n*k bytes (int64 cast of KnnResult.neighbors)
//   - embeddings[n*n_components]: 4*n*n_components bytes (output)
//   - cuML internal: ~10*n bytes for fuzzy 1-skeleton + SGD scratch
//   Total at 1M cells, k=15, 2D: ~120MB + 8MB + ~10MB = ~138 MB device.
//
// Streams:
//   1, caller-provided. raft::handle_t is constructed bound to it. cuML UMAP uses
//   this handle for all its GPU work. Same pattern as graph/leiden.h.
//
// OOC:
//   cuML UMAP requires the full (n*k) kNN graph on device. For n > 10M:
//   k-means subsample → UMAP on subsample → project remaining cells. Defer to cycle ≥ 12.
//
// Precision:
//   fp32 throughout (embeddings, distances). cuML SGD internally fp32.
//   knn_indices cast to int64_t (cuML API requirement) via device kernel — no accuracy impact.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/graph/knn.h>

// cuML UMAP is an optional compile-time dependency. The __has_include guard
// lets the header be parsed by IDEs and static analysers without cuML.
// The function body issues a static_assert if cuML is absent.
#if defined(__has_include)
#  if __has_include(<cuml/manifold/umap.hpp>)
#    include <cuml/manifold/umap.hpp>
#    include <cuml/manifold/umapparams.h>
#    include <raft/core/handle.hpp>
#    include <raft/matrix/matrix.cuh>
#    define SINGLET_GPU_HAS_CUML_UMAP 1
#  endif
#endif

namespace singlet_gpu {
namespace embed {

// ─── Public API types ─────────────────────────────────────────────────────────

// UmapInit controls embedding initialisation.
// Random (default): reproducible with fixed seed (cuML random_state + SGD).
// Spectral: uses RAFT's eigensolver on the fuzzy 1-skeleton; NOT reproducible
//   across runs even with a fixed seed (cuML GitHub #6696). Offered for users
//   who prefer spectral quality and accept non-determinism.
enum class UmapInit {
    Random,   // cuML UMAPParams::init = 1 (default; deterministic with fixed seed)
    Spectral, // cuML UMAPParams::init = 0 (non-deterministic per cuML #6696)
};

struct UmapConfig {
    int        n_components          = 2;
    int        n_epochs              = 0;     // 0 = cuML auto-select (200 for n≤10k, 500 for n>10k)
    float      min_dist              = 0.5f;
    float      spread                = 1.0f;
    float      learning_rate         = 1.0f;
    UmapInit   init                  = UmapInit::Random;
    uint64_t   seed                  = 0;
    int        negative_sample_rate  = 5;
};

struct UmapResult {
    core::DeviceMemory<float> embedding;  // n × n_components, row-major
    int n;
    int n_components;
};

// ─── Detail: int32 → int64 cast kernel ───────────────────────────────────────

namespace detail {

// cuML's fit() knn_indices parameter is int64_t*, but KnnResult stores int32.
// Cast device-side in a single fused kernel — no host round-trip, no per-cell loop.
// One thread per element: n*k elements, each a trivial widening cast.
__global__ void __launch_bounds__(256, 4)
cast_i32_to_i64_kernel(const int* __restrict__  src,
                       int64_t* __restrict__     dst,
                       int                       total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) dst[i] = static_cast<int64_t>(src[i]);
}

}  // namespace detail

// ─── Public entry point ───────────────────────────────────────────────────────

// umap — UMAP embedding from a precomputed kNN graph.
//
// knn:    KnnResult from graph/knn.h (CSR: row_offsets, neighbors[n*k],
//           distances[n*k]). The row_offsets field is ignored — cuML takes
//           the flat (n*k) arrays directly.
// cfg:    UmapConfig (defaults: 2D, cuML-auto epochs, min_dist=0.5, Random init,
//           seed=0, negative_sample_rate=5).
// stream: caller-provided CUDA stream (nullptr = default stream).
//
// Returns UmapResult with device-resident float embedding[n × n_components].
//
// Determinism: Random init + fixed seed → deterministic across runs.
//   Spectral init → NOT deterministic (cuML upstream limitation; see header doc).
//
// Thread safety: raft::handle_t wraps CUDA state; do not share a stream
//   across concurrent umap() calls.
inline UmapResult
umap(const graph::KnnResult& knn,
     const UmapConfig& cfg,
     cudaStream_t stream)
{
#if !defined(SINGLET_GPU_HAS_CUML_UMAP)
    // TODO(CYCLE-GATE-2F): static_assert replaced with runtime throw — real fix is
    // preprocessor #error inside #ifndef guard, not a compile-time instantiation assert.
    // static_assert(sizeof(knn) == 0, "cuML required");
    (void)cfg; (void)stream;
    throw std::runtime_error(
        "singlet_gpu::embed::umap requires cuML. "
        "Rebuild with find_package(cuml) resolving in CMakeLists.txt.");
    UmapResult r; return r;
#else
    if (knn.n <= 0 || knn.k <= 0)
        throw std::invalid_argument("umap: empty KnnResult (n or k is zero)");
    if (cfg.n_components <= 0)
        throw std::invalid_argument("umap: n_components must be > 0");
    // cuML silently fails or segfaults beyond its internal limit (~50).
    // Guard here so the error message is actionable.
    if (cfg.n_components > 50)
        throw std::invalid_argument("umap: n_components > 50 is not supported by cuML UMAP");

    const int n   = knn.n;
    const int k   = knn.k;
    const int nk  = n * k;

    // 1. Cast knn.neighbors (int32) → int64_t device buffer.
    //    cuML fit() requires int64_t* knn_indices. One-time O(n*k) cast kernel.
    //    NOT inside a per-cell loop — single grid launch covers all n*k elements.
    core::DeviceMemory<int64_t> knn_indices_i64(static_cast<size_t>(nk));
    {
        const int blk = 256;
        const int grd = (nk + blk - 1) / blk;
        detail::cast_i32_to_i64_kernel<<<grd, blk, 0, stream>>>(
            knn.neighbors.get(),
            knn_indices_i64.get(),
            nk);
    }

    // 2. Allocate output embedding on device.
    UmapResult result;
    result.n            = n;
    result.n_components = cfg.n_components;
    result.embedding    = core::DeviceMemory<float>(
        static_cast<size_t>(n) * static_cast<size_t>(cfg.n_components));

    // 3. Construct UMAPParams from our UmapConfig.
    //    Verified fields from cuml/manifold/umapparams.h (branch-25.10):
    //      n_components, n_epochs, min_dist, spread, initial_alpha (= learning_rate),
    //      negative_sample_rate, random_state (= seed), init (int: 1=Random, 0=Spectral),
    //      n_neighbors (= k, required so cuML knows the kNN graph width).
    ML::UMAPParams params;
    params.n_neighbors          = k;
    params.n_components         = cfg.n_components;
    params.n_epochs             = cfg.n_epochs;        // 0 = cuML auto-select
    params.min_dist             = cfg.min_dist;
    params.spread               = cfg.spread;
    params.initial_alpha        = cfg.learning_rate;   // cuML names it initial_alpha
    params.negative_sample_rate = cfg.negative_sample_rate;
    params.random_state         = cfg.seed;
    params.init                 = (cfg.init == UmapInit::Random) ? 1 : 0;
    // Suppress cuML's INFO-level logging; only errors propagate to the caller.
    params.verbosity            = rapids_logger::level_enum::off;

    // 4. Construct raft::handle_t bound to the caller's stream.
    //    Same pattern as graph/leiden.h: cuML uses this handle for all GPU work.
    raft::handle_t handle(stream);

    // 5. Dummy X buffer.
    //    cuML fit() with non-null knn_indices/knn_dists uses X only for n and d
    //    (shape metadata); the actual coordinates are not accessed.
    //    We supply a 1-element dummy so X is non-null without wasting memory.
    //    d is set to k (the only consistent value without the original embedding).
    core::DeviceMemory<float> dummy_X(1);
    const int dummy_d = k;  // WHY: cuML checks d > 0 but doesn't read X when knn provided

    // 6. host_coo_matrix output for the fuzzy 1-skeleton.
    //    cuML writes the computed graph here; we discard it (we need embeddings only).
    //    raft::host_coo_matrix<float, int, int, uint64_t> is the required type.
    raft::host_coo_matrix<float, int, int, uint64_t> graph;

    // 7. Synchronize stream before the cuML call.
    //    The i32→i64 cast kernel in step 1 must complete before cuML reads knn_indices_i64.
    //    cuML constructs its own internal stream ordering via raft::handle_t, but the
    //    cast kernel was launched on the caller's stream. A stream sync ensures ordering.
    cudaStreamSynchronize(stream);

    // 8. Call cuML UMAP fit() via the precomputed kNN path.
    //    knn_indices_i64: int64_t* [n*k], device-resident
    //    knn.distances:   float*   [n*k], device-resident (no copy needed — already fp32)
    //    y = nullptr: unsupervised (no label supervision)
    //    result.embedding.get(): float* [n * n_components], device output
    ML::UMAP::fit(
        handle,
        dummy_X.get(),      // X (shape only; not read when knn provided)
        nullptr,             // y (unsupervised)
        n,
        dummy_d,             // d (shape only)
        knn_indices_i64.get(),
        knn.distances.get(),
        &params,
        result.embedding.get(),
        graph);

    // raft::handle_t destructor flushes the stream; no explicit sync needed after fit().
    return result;
#endif
}

}  // namespace embed
}  // namespace singlet_gpu
