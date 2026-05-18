// SPDX-License-Identifier: GPL-2.0-or-later
// examples/cpp_minimal/main.cpp — public API smoke test.
//
// Compiles only against the umbrella `<singlet/gpu/singlet::gpu.hpp>`. If
// this file fails to build, the public API surface is broken — Rule 23 +
// release-policy violation.
//
// Run:
//   ./cpp_minimal                        # version + load library only
//   ./cpp_minimal /path/to/exon_counts.1pz   # also load a real .1pz

#include <singlet/gpu/singlet::gpu.hpp>

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
        std::printf("Pass a .1pz path to test the loader.\n");
        return 0;
    }

    const std::string path = argv[1];
    auto mat = singlet::gpu::load_pz(path);
    cudaStreamSynchronize(mat.producer_stream);

    std::printf("loaded %s: %d genes x %d cells, %lld nnz\n",
                path.c_str(),
                mat.n_genes,
                mat.n_cells,
                static_cast<long long>(mat.mat.nnz));

    if (!mat.meta.gsm_id.empty()) {
        std::printf("sample: %s  (organism=%s, protocol=%s)\n",
                    mat.meta.gsm_id.c_str(),
                    mat.meta.organism.c_str(),
                    mat.meta.protocol.c_str());
    }

    if (mat.producer_stream) {
        cudaStreamDestroy(mat.producer_stream);
    }
    return 0;
}
