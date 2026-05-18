// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_nmf_profile_c86.cpp
//
// OPTIM-NMF-K50 Phase B — verbose per-phase timing at k=20 vs k=50.
// Uses factornet's built-in verbose timing (enabled via cfg.verbose = true)
// which prints per-component wall time:
//   H:  gram / rhs / nnls / norm
//   W:  gram / rhs / nnls / norm
//   loss (every loss_every iterations)
//
// Also reports:
//   - Iteration count at convergence (or max_iter if not converged)
//   - Auto-solver switch iteration (MU → CD switch in auto mode)
//   - cd_max_iter inner CD iterations per outer iter
//   - Total wall time (cudaEvent-based, excluding load_pz)
//
// Run TWO configurations:
//   A) k=20, solver_mode=3 (auto) — baseline that is fast (391ms)
//   B) k=50, solver_mode=3 (auto) — regressed (15,980ms)
//   C) k=50, solver_mode=2 (MU)  — control: shows MU cost without CD switch
//
// This is a READ-ONLY profiling driver. No kernel modifications.
// nsys profile + ncu are run externally (in the SLURM script).
//
// Node: g050 or g051 (H100 NVL) — same hardware as the cycle 85 regression.

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet/gpu/bench/harness.h>
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/reduce/nmf/fit.h>
#include <singlet/gpu/reduce/nmf/types.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
using namespace singlet::gpu;

struct ProfileRun {
    const char* label;
    int k;
    int solver_mode;  // 0=CD, 1=Cholesky, 2=MU, 3=auto
    int max_iter;
    int cd_max_iter;
};

static void run_profile(const io::PzDeviceMatrix& mat, const ProfileRun& p, cudaStream_t stream) {
    reduce::nmf::NmfConfig cfg;
    cfg.rank        = p.k;
    cfg.max_iter    = p.max_iter;
    cfg.solver_mode = p.solver_mode;
    cfg.cd_max_iter = p.cd_max_iter;
    cfg.verbose     = true;   // Enable factornet's per-phase GPU timer output
    cfg.seed        = 42u;

    // Time the full fit() call with cudaEvent
    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    std::printf("\n");
    std::printf("===================================================================\n");
    std::printf("PROFILE  label=%s  k=%d  solver=%d  max_iter=%d  cd_max_iter=%d\n",
                p.label, p.k, p.solver_mode, p.max_iter, p.cd_max_iter);
    std::printf("===================================================================\n");
    std::fflush(stdout);

    cudaEventRecord(ev_start, stream);

    // Call the un-patched path (bypass the Cycle 86 MU-forcing fix) so we
    // profile the ORIGINAL regression behavior. We do this by calling
    // factornet::nmf::nmf_fit_gpu directly with the raw config.
    //
    // For the MU-control run (solver_mode=2) we call via the adapter (which
    // does not override solver_mode when it is not 3).
    //
    // Note: factornet NMF takes host CSC pointers (pinned host buffers).
    auto result = ::factornet::nmf::nmf_fit_gpu<float>(
        mat.host_indptr.get(),
        mat.host_indices.get(),
        mat.host_values.get(),
        static_cast<int>(mat.mat.rows),
        static_cast<int>(mat.mat.cols),
        static_cast<int>(mat.mat.nnz),
        cfg,
        /*W_init=*/nullptr,
        /*H_init=*/nullptr);

    cudaEventRecord(ev_end, stream);
    cudaStreamSynchronize(stream);

    float wall_ms = 0.0f;
    cudaEventElapsedTime(&wall_ms, ev_start, ev_end);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);

    std::printf("RESULT  label=%s  k=%d  iterations=%d  converged=%d"
                "  final_tol=%.6f  wall_ms=%.1f\n",
                p.label, p.k, result.iterations,
                (int)result.converged, result.final_tol, wall_ms);
    std::printf("W shape: %d x %d   H shape: %d x %d\n",
                (int)result.W.rows(), (int)result.W.cols(),
                (int)result.H.rows(), (int)result.H.cols());
    std::fflush(stdout);
}

int main() {
    if (!gpu_available()) { skip_no_gpu(); return 0; }

    // Use gene_counts.1pz (singlet pipeline output; counts.1pz does not exist)
    std::string pz_path = std::string(CANONICAL_SAMPLE_DIR) + "/gene_counts.1pz";
    if (!fs::exists(pz_path)) {
        // fallback: try exon_counts.1pz
        pz_path = std::string(CANONICAL_SAMPLE_DIR) + "/exon_counts.1pz";
    }
    if (!fs::exists(pz_path)) {
        skip_no_sample(pz_path);
        std::printf("SKIP: no gene_counts.1pz or exon_counts.1pz found at %s\n",
                    CANONICAL_SAMPLE_DIR);
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    io::PzDeviceMatrix mat = io::load_pz(pz_path, stream, /*keep_host_pinned=*/true);
    cudaStreamSynchronize(stream);

    std::printf("=== OPTIM-NMF-K50 Phase B — verbose per-phase profiling ===\n");
    std::printf("Matrix: %lld rows x %lld cols  nnz=%lld\n",
                (long long)mat.mat.rows, (long long)mat.mat.cols,
                (long long)mat.mat.nnz);
    std::printf("Node: see SLURM_NODELIST env var\n");
    std::printf("cd_max_iter default = 100 (constants::CD_MAXIT)\n");
    std::printf("auto_pilot_iters    = 5  (config default)\n\n");
    std::fflush(stdout);

    // Run A: k=20, auto solver (should be ~391ms, stays on MU or MU→CD transition)
    run_profile(mat, {"k20_auto",  20, 3, 100, 100}, stream);

    // Run B: k=50, auto solver (15,980ms regression — auto switches MU→CD)
    run_profile(mat, {"k50_auto",  50, 3, 100, 100}, stream);

    // Run C: k=50, forced MU (control — how fast is MU alone at k=50)
    run_profile(mat, {"k50_MU",    50, 2, 100, 100}, stream);

    // Run D: k=50, forced CD (control — how slow is pure CD at k=50)
    run_profile(mat, {"k50_CD",    50, 0, 100, 100}, stream);

    // Run E: k=50, auto solver, cd_max_iter=10 (isolate cd_max_iter vs iteration count)
    run_profile(mat, {"k50_auto_cd10", 50, 3, 100, 10}, stream);

    cudaStreamDestroy(stream);
    return 0;
}
