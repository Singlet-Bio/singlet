// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/state/cycle67_nmf_diag_driver.cpp
//
// Cycle 67 NMF k=50 regression diagnostic.
// Compiled as a CUDA TU (includes factornet GPU headers).

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/reduce/nmf/fit.h>
#include <singlet-gpu/reduce/nmf/types.h>

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <string>
#include <vector>

using namespace singlet_gpu;

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", \
            (int)_e, cudaGetErrorString(_e), __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)

// Wall-clock timer in milliseconds (host-side).
static double wall_ms() {
    using clk = std::chrono::high_resolution_clock;
    static thread_local clk::time_point t0 = clk::now();
    auto now = clk::now();
    return std::chrono::duration<double, std::milli>(now - t0).count();
}

// Run NMF once, return: wall_ms, iterations, converged, train_loss.
struct DiagResult {
    double wall_ms;
    int    iterations;
    bool   converged;
    float  train_loss;
    std::string tag;
};

static DiagResult run_nmf(const io::PzDeviceMatrix& mat,
                           reduce::nmf::NmfConfig cfg,
                           const std::string& tag)
{
    DiagResult r;
    r.tag = tag;

    CUDA_CHECK(cudaDeviceSynchronize());
    double t0 = wall_ms();
    try {
        auto res = reduce::nmf::fit(mat, cfg);
        CUDA_CHECK(cudaDeviceSynchronize());
        r.wall_ms   = wall_ms() - t0;
        r.iterations = res.iterations;
        r.converged  = res.converged;
        r.train_loss = res.train_loss;
    } catch (const std::exception& e) {
        r.wall_ms   = -1.0;
        r.iterations = -1;
        r.converged  = false;
        r.train_loss = -1.0f;
        std::printf("[%s] EXCEPTION: %s\n", tag.c_str(), e.what());
    }
    return r;
}

static void print_result(const DiagResult& r) {
    std::printf("  %-50s  wall=%8.1fms  iters=%4d  converged=%s  loss=%.6f\n",
                r.tag.c_str(), r.wall_ms, r.iterations,
                r.converged ? "yes" : "no ", r.train_loss);
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    // Optional: pass PZ path as argv[1].
    std::string pz_path = "/mnt/projects/debruinz_project/singlify_pipeline/"
                          "quant/scrna/GSE127/GSE127918/GSM4037629/gene_counts.1pz";
    if (argc > 1) pz_path = argv[1];

    std::printf("=== Cycle 67 NMF k=50 Regression Diagnostic ===\n");
    std::printf("PZ: %s\n\n", pz_path.c_str());
    std::fflush(stdout);

    // Load matrix.
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    io::PzDeviceMatrix mat;
    try {
        mat = io::load_pz(pz_path, stream, /*keep_host_pinned=*/true);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR loading PZ: %s\n", e.what());
        return 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::printf("Loaded: %d genes x %d cells, nnz=%d\n\n",
                (int)mat.mat.rows, (int)mat.mat.cols, (int)mat.mat.nnz);

    std::vector<DiagResult> results;

    // -----------------------------------------------------------------------
    // Panel A: k sweep {10,20,30,40,50} — production config.
    // Config: solver_mode=1 (Cholesky), max_iter=200, tol=1e-4, cd_max_iter default.
    // Hypothesis: if iterations at k=50 are 5-10x higher than at k=20, convergence
    // is the culprit. If iterations are similar, per-iteration cost is the culprit.
    // -----------------------------------------------------------------------
    std::printf("=== Panel A: k sweep {10,20,30,40,50} (production config) ===\n");
    std::printf("Config: solver_mode=1 (Cholesky+clip), max_iter=200, tol=1e-4f, cd_max_iter=default\n");
    std::printf("Primary regression hypothesis: Cholesky cost O(k^3) vs CD O(k^2*cd_iter)\n\n");
    for (int k : {10, 20, 30, 40, 50}) {
        reduce::nmf::NmfConfig cfg;
        cfg.rank          = k;
        cfg.max_iter      = 200;
        cfg.tol           = 1e-4f;
        cfg.seed          = 42;
        cfg.solver_mode   = 1;  // Cholesky+clip (production default)
        cfg.track_loss_history = true;  // enables result.iterations to be accurate
        std::string tag = "A_k=" + std::to_string(k) + "_cholesky_max200";
        auto r = run_nmf(mat, cfg, tag);
        print_result(r);
        results.push_back(r);
    }

    // -----------------------------------------------------------------------
    // Panel B: k=50, max_iter forcing {200,100,50,20}.
    // If wall_ms scales linearly with max_iter: convergence is culprit (too many iters).
    // If wall_ms is similar across max_iter: per-iteration cost at k=50 is culprit.
    // -----------------------------------------------------------------------
    std::printf("\n=== Panel B: k=50, max_iter forcing {200,100,50,20} ===\n");
    std::printf("If wall scales linearly with max_iter → convergence bug.\n");
    std::printf("If wall is flat vs max_iter → per-iteration cost regression.\n\n");
    for (int mi : {200, 100, 50, 20}) {
        reduce::nmf::NmfConfig cfg;
        cfg.rank          = 50;
        cfg.max_iter      = mi;
        cfg.tol           = 1e-10f;  // disable early-stop → always runs mi iters
        cfg.seed          = 42;
        cfg.solver_mode   = 1;
        cfg.track_loss_history = true;
        std::string tag = "B_k=50_maxiter=" + std::to_string(mi) + "_notol";
        auto r = run_nmf(mat, cfg, tag);
        print_result(r);
        results.push_back(r);
    }

    // -----------------------------------------------------------------------
    // Panel C: k=50, init_mode=0 (random) vs init_mode=1 (Lanczos SVD warm-start).
    // If Lanczos init converges in fewer iterations at k=50, init is a fix path.
    // -----------------------------------------------------------------------
    std::printf("\n=== Panel C: k=50, init_mode random(0) vs Lanczos(1) ===\n");
    std::printf("Tests whether SVD warm-start reduces iterations at k=50.\n\n");
    for (int im : {0, 1}) {
        reduce::nmf::NmfConfig cfg;
        cfg.rank          = 50;
        cfg.max_iter      = 200;
        cfg.tol           = 1e-4f;
        cfg.seed          = 42;
        cfg.solver_mode   = 1;
        cfg.init_mode     = im;
        cfg.track_loss_history = true;
        std::string tag = std::string("C_k=50_init=") + (im == 0 ? "random" : "lanczos");
        auto r = run_nmf(mat, cfg, tag);
        print_result(r);
        results.push_back(r);
    }

    // -----------------------------------------------------------------------
    // Panel D: k=50, solver_mode=0 (CD) vs solver_mode=1 (Cholesky+clip).
    // GUIDE.md notes Cholesky is faster for k<20; CD is better for large k.
    // If CD is significantly faster at k=50: production default is wrong.
    // -----------------------------------------------------------------------
    std::printf("\n=== Panel D: k=50, solver_mode CD(0) vs Cholesky(1) ===\n");
    std::printf("GUIDE.md: Cholesky faster for k<20; CD better for large k.\n");
    std::printf("Cholesky inner cost: O(k^3) per H-update column.\n");
    std::printf("CD inner cost: O(k * cd_max_iter) per column.\n\n");
    for (int sm : {0, 1}) {
        reduce::nmf::NmfConfig cfg;
        cfg.rank          = 50;
        cfg.max_iter      = 200;
        cfg.tol           = 1e-4f;
        cfg.seed          = 42;
        cfg.solver_mode   = sm;
        cfg.track_loss_history = true;
        std::string tag = std::string("D_k=50_solver=") + (sm == 0 ? "cd" : "cholesky");
        auto r = run_nmf(mat, cfg, tag);
        print_result(r);
        results.push_back(r);
    }

    // -----------------------------------------------------------------------
    // Panel E: k=50 CD solver (solver_mode=0), cd_max_iter sweep {100,50,20,5}.
    // Quantifies the inner-loop contribution at k=50.
    // -----------------------------------------------------------------------
    std::printf("\n=== Panel E: k=50 CD solver, cd_max_iter {100,50,20,5} ===\n");
    std::printf("Quantifies how much of k=50 wall time is in the CD inner loop.\n\n");
    for (int cdi : {100, 50, 20, 5}) {
        reduce::nmf::NmfConfig cfg;
        cfg.rank          = 50;
        cfg.max_iter      = 200;
        cfg.tol           = 1e-4f;
        cfg.seed          = 42;
        cfg.solver_mode   = 0;  // CD
        cfg.cd_max_iter   = cdi;
        cfg.track_loss_history = true;
        std::string tag = "E_k=50_cd_cditer=" + std::to_string(cdi);
        auto r = run_nmf(mat, cfg, tag);
        print_result(r);
        results.push_back(r);
    }

    // -----------------------------------------------------------------------
    // Summary table.
    // -----------------------------------------------------------------------
    std::printf("\n=== SUMMARY TABLE ===\n");
    std::printf("%-52s  %9s  %6s  %10s  %10s\n",
                "tag", "wall_ms", "iters", "converged", "train_loss");
    std::printf("%-52s  %9s  %6s  %10s  %10s\n",
                "---", "-------", "-----", "---------", "----------");
    for (auto& r : results) {
        std::printf("%-52s  %9.1f  %6d  %10s  %10.6f\n",
                    r.tag.c_str(), r.wall_ms, r.iterations,
                    r.converged ? "yes" : "no", r.train_loss);
    }

    // Convergence vs per-iteration diagnosis.
    std::printf("\n=== DIAGNOSIS ===\n");
    // Find Panel A results for k=20 and k=50.
    DiagResult* r20 = nullptr;
    DiagResult* r50 = nullptr;
    for (auto& r : results) {
        if (r.tag == "A_k=20_cholesky_max200") r20 = &r;
        if (r.tag == "A_k=50_cholesky_max200") r50 = &r;
    }
    if (r20 && r50 && r20->iterations > 0 && r50->iterations > 0) {
        double iter_ratio = (double)r50->iterations / r20->iterations;
        double wall_ratio = r50->wall_ms / r20->wall_ms;
        double per_iter_20 = r20->wall_ms / r20->iterations;
        double per_iter_50 = r50->wall_ms / r50->iterations;
        std::printf("k=20: wall=%.1fms iters=%d → per_iter=%.2fms\n",
                    r20->wall_ms, r20->iterations, per_iter_20);
        std::printf("k=50: wall=%.1fms iters=%d → per_iter=%.2fms\n",
                    r50->wall_ms, r50->iterations, per_iter_50);
        std::printf("iter ratio (k50/k20) = %.2fx   wall ratio = %.2fx\n",
                    iter_ratio, wall_ratio);
        std::printf("per-iter ratio (k50/k20) = %.2fx\n", per_iter_50 / per_iter_20);
        if (iter_ratio > 3.0) {
            std::printf("PRIMARY CULPRIT: CONVERGENCE — k=50 needs %.1fx more iterations.\n",
                        iter_ratio);
            std::printf("  Fix candidate: increase tol at k>30, or switch to Lanczos init.\n");
        } else if (per_iter_50 / per_iter_20 > 2.0) {
            std::printf("PRIMARY CULPRIT: PER-ITERATION COST — k=50 is %.1fx slower per iter.\n",
                        per_iter_50 / per_iter_20);
            std::printf("  Fix candidate: switch solver_mode 1→0 (CD) for k>20.\n");
            std::printf("  Cholesky NNLS inner cost is O(k^3): 50^3/20^3 = %.1fx expected.\n",
                        (50.0*50*50) / (20.0*20*20));
        } else {
            std::printf("MIXED: both convergence and per-iteration cost contribute.\n");
        }
    }

    std::printf("\nDIAG COMPLETE\n");
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
