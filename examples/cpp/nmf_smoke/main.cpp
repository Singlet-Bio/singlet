// SPDX-License-Identifier: GPL-2.0-or-later
// examples/nmf_smoke/main.cpp
//
// Smoke test: load a .1pz, run nmf::fit at k=10, print loss.
// Confirms the native NMF kernel executes end-to-end on GPU.
//
// Usage:
//   ./nmf_smoke /path/to/counts.1pz
//   ./nmf_smoke /path/to/counts.1pz 20   # rank 20

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/reduce/nmf/fit.h>
#include <singlet/gpu/reduce/nmf/types.h>
#include <singlet/gpu/version.h>

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    std::printf("singlet-gpu %d.%d.%d (commit %s)\n",
                singlet::gpu::version_major(),
                singlet::gpu::version_minor(),
                singlet::gpu::version_patch(),
                singlet::gpu::commit_sha());

    if (argc < 2) {
        std::printf("Usage: %s <path.1pz> [rank=10]\n", argv[0]);
        std::printf("  Loads the matrix, runs NMF at the given rank, prints loss.\n");
        return 0;
    }

    const std::string path = argv[1];
    const int k = (argc >= 3) ? std::atoi(argv[2]) : 10;

    std::printf("Loading %s ...\n", path.c_str());
    auto mat = singlet::gpu::io::load_pz(path, /*stream=*/nullptr, /*keep_host_pinned=*/true);
    cudaStreamSynchronize(mat.producer_stream);

    std::printf("  %d genes x %d cells, %lld nnz\n",
                mat.n_genes, mat.n_cells,
                static_cast<long long>(mat.mat.nnz));

    std::printf("Running NMF at rank k=%d (MU solver) ...\n", k);

    singlet::gpu::reduce::nmf::NmfConfig cfg;
    cfg.rank         = k;
    cfg.max_iter     = 50;
    cfg.solver_mode  = 2;   // MU explicitly
    cfg.init_mode    = 0;   // random init (no SVD dep for smoke test)
    cfg.seed         = 42;
    cfg.verbose      = false;

    auto result = singlet::gpu::reduce::nmf::fit(mat, cfg);

    std::printf("NMF done: %d iters, converged=%d\n",
                result.iterations, result.converged ? 1 : 0);
    if (!result.train_loss.empty()) {
        std::printf("  initial loss: %.4g\n", result.train_loss.front());
        std::printf("  final   loss: %.4g\n", result.train_loss.back());

        // Check that loss is strictly finite and decreasing (convergence sanity).
        float first_loss = result.train_loss.front();
        float last_loss  = result.train_loss.back();
        if (last_loss < first_loss) {
            std::printf("  loss converges: YES (delta=%.4g)\n", first_loss - last_loss);
        } else {
            std::printf("  loss converges: NO (last >= first — check implementation)\n");
        }
    }

    std::printf("W shape: %d x %d\n", result.W.m, result.W.n);
    std::printf("H shape: %d x %d\n", result.H.m, result.H.n);

    if (mat.producer_stream)
        cudaStreamDestroy(mat.producer_stream);

    return 0;
}
