// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: factornet/nmf/fit_chunked_gpu.cuh — factornet::nmf::nmf_chunked_gpu<float>
//
// Adapter: singlet_gpu::reduce::nmf::chunked_fit
//
// This is the streaming / out-of-core NMF entry point.  Unlike fit() and cv_fit(),
// it does NOT take a PzDeviceMatrix — it takes a PzDataLoader reference, which
// implements factornet::io::DataLoader<float> and yields Eigen sparse chunks from
// one or more .1pz files without holding the full matrix in memory.
//
// Algorithm: chunked block coordinate descent on GPU.
//   H-update: iterate DataLoader::next_forward()  → upload each Chunk to device → SpMM.
//   W-update: iterate DataLoader::next_transpose() → upload each Chunk to device → SpMM.
//   G (k×k) and W (k×m) stay device-resident across chunks.
//   H (k×n) is accumulated chunk-by-chunk (only the current H slice is device-resident).
//   Reference: factornet/nmf/fit_chunked_gpu.cuh §"Algorithm" + loader.hpp.
//
// Non-zero-copy path: PzDataLoader converts each .1pz chunk to Eigen::SparseMatrix<float>
//   (see streaming/pz_data_loader.h). This is the ONLY non-zero-copy step in the
//   library. The Eigen allocation happens once per chunk; factornet then uploads
//   the chunk to device (a second allocation).  Total per-chunk cost:
//     host alloc: chunk_cols * avg_nnz_per_col * 8 bytes (Eigen float + int32 index)
//     PCIe transfer: same bytes
//   For default chunk_cols=100k and avg density 1k nnz/col: ~800 MB/chunk, ~40 ms PCIe.
//
// Time complexity: O(nnz_total * k * iters) — same as in-memory, amortized over chunks.
// Workspace budget (device):
//   W: k × m (grows with m — full rows resident always).
//   H slice: k × chunk_cols (constant per chunk).
//   G: k × k.
//   Panel upload: chunk_cols × avg_nnz_per_col × 8 bytes (uploaded and discarded per chunk).
//   For k=20, m=30k genes, chunk_cols=100k: W≈24 MB, G<1 MB, panel≈800 MB, H slice≈8 MB.
//
// Streams: factornet creates GPUContext internally (no external stream).
// Precision: fp32 data; fp64 Gram accumulation (k²).
// Determinism: seeded via config.seed.
//
// OOC plan: this IS the OOC plan.  PzDataLoader slides a window of chunk_cols columns
//   across the full matrix.  Memory at any moment: 1 chunk (Eigen) + 1 panel (device).
//   For 1M+ cells: set chunk_cols=100k → 10 forward passes per epoch.
//   Future: factornet PR to accept device pointers directly in DataLoader::Chunk,
//   eliminating the Eigen allocation.  Tracked as CYCLE-6-FOLLOWUP-FACTORNET-DEVICE-LOADER.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

// WHY: factornet/nmf/fit_chunked_gpu.cuh references factornet::constants::EPS
// but the factornet source defines no such sub-namespace (upstream omission).
// Inject the constant here before including the header so both g++ and nvcc
// can resolve it.  Value 1e-10 matches factornet's kl_epsilon() in constants.hpp.
namespace factornet {
namespace constants {
static constexpr double EPS = 1e-10;
}  // namespace constants
}  // namespace factornet

#include <factornet/nmf/fit_chunked_gpu.cuh>
#include <singlet-gpu/reduce/nmf/types.h>

// Forward-declare PzDataLoader to avoid including the full streaming header here.
// Callers that use chunked_fit must include streaming/pz_data_loader.h themselves.
namespace singlet_gpu {
namespace io {
class PzDataLoader;
}  // namespace io
}  // namespace singlet_gpu

// Full include needed because we pass PzDataLoader& to factornet::nmf::nmf_chunked_gpu.
#include <singlet-gpu/streaming/pz_data_loader.h>

namespace singlet_gpu {
namespace reduce {
namespace nmf {

// chunked_fit — streaming NMF adapter.
//
// loader: a PzDataLoader opened on one or more .1pz files.
//   Provides forward + transpose iteration over column panels (chunks).
//   PzDataLoader::chunk_cols() controls panel width (default 100k).
//
// cfg: standard NmfConfig.  cfg.rank, cfg.seed, cfg.max_iter, cfg.tol apply.
//   Solver mode 2 (MU) is recommended for chunked NMF — it avoids the Gram-inversion
//   step that requires the full matrix (factornet auto-selects when solver_mode=3).
//
// W_init / H_init: optional warm-start.  H_init must match n = loader.cols().
//   Useful for incremental re-training when new samples are appended.
//
// Returns NmfResult with W (m×k), d (k), H (k×n), convergence history.
inline NmfResult chunked_fit(io::PzDataLoader& loader,
                             const NmfConfig& cfg,
                             const DenseMatrix* W_init = nullptr,
                             const DenseMatrix* H_init = nullptr)
{
    // PzDataLoader inherits from factornet::io::DataLoader<float>.
    // Pass as the base interface — factornet handles all chunk iteration.
    return ::factornet::nmf::nmf_chunked_gpu<float>(
        static_cast<::factornet::io::DataLoader<float>&>(loader),
        cfg,
        W_init,
        H_init);
}

}  // namespace nmf
}  // namespace reduce
}  // namespace singlet_gpu
