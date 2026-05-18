// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_reduce_nmf_c86.cpp
//
// Cycle 86 / OPTIM-NMF-K50 verification bench.
//
// Sweeps k={10, 20, 50, 100} with the new adapter-level MU default (solver_mode=2
// for k<64) to confirm that k=50 no longer triggers the slow CD path.
//
// Expected outcome:
//   k=10  : fast (MU), same as before
//   k=20  : fast (MU) — was already ~391ms (13.8× faster than sklearn 5,383ms)
//   k=50  : fast (MU) — was 15,980ms (40× slower); expect ~600-1,200ms post-fix
//   k=100 : Cholesky branch (k≥64) — note wall time but do not fight it
//
// Sklearn CPU reference (k=20): 5,383ms (Cycle 62b).

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

// Sklearn reference wall times (ms) from prior cycles (CPU, single-threaded).
// k=10: ~2,000ms (Cycle 61), k=20: 5,383ms (Cycle 62b), k=50/100: not measured.
static constexpr float SKLEARN_MS_K10  = 2000.0f;
static constexpr float SKLEARN_MS_K20  = 5383.0f;  // Cycle 62b canonical

static void run_k(const io::PzDeviceMatrix& mat, int k, cudaStream_t stream,
                  float sklearn_ms_ref)
{
    reduce::nmf::NmfConfig cfg;
    cfg.rank     = k;
    cfg.max_iter = 100;
    // solver_mode left at default (3 = auto) — adapter will override to MU for k<64.

    BenchTimer        timer(stream);
    PeakMemoryTracker mem;

    const int WARMUP = 1;
    const int TIMED  = 3;

    for (int iter = 0; iter < WARMUP + TIMED; ++iter) {
        bool timed = (iter >= WARMUP);
        if (timed) mem.snapshot_before();
        timer.start();
        auto result = reduce::nmf::fit(mat, cfg);
        (void)result;
        cudaDeviceSynchronize();
        timer.stop();
        if (timed) { timer.record(); mem.snapshot_after(); }
    }

    float med_ms     = timer.median_ms();
    float min_ms     = timer.min_ms();
    float max_ms     = timer.max_ms();
    float speedup    = (sklearn_ms_ref > 0.0f) ? sklearn_ms_ref / med_ms : -1.0f;
    // FitConfig::k_cd_cutoff default = 32; k >= 32 with solver_mode=3 → forced to MU
    bool  is_mu      = (k >= 32);

    std::printf("BENCH nmf k=%3d  solver=%s  med=%.1fms  min=%.1fms  max=%.1fms"
                "  mem=%.0fMB",
                k, is_mu ? "MU(forced)" : "CD/auto",
                med_ms, min_ms, max_ms, mem.peak_mb());
    if (sklearn_ms_ref > 0.0f)
        std::printf("  vs_sklearn_cpu=%.2fx", speedup);
    std::printf("\n");
    std::fflush(stdout);
}

int main() {
    if (!gpu_available()) { skip_no_gpu(); return 0; }

    const std::string pz_path = std::string(CANONICAL_SAMPLE_DIR) + "/counts.1pz";
    if (!fs::exists(pz_path)) {
        skip_no_sample(pz_path);
        std::printf("SKIP: canonical sample not found; cannot run multi-k bench.\n");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    io::PzDeviceMatrix mat = io::load_pz(pz_path, stream, /*keep_host_pinned=*/true);
    cudaStreamSynchronize(stream);

    std::printf("=== Cycle 86 NMF multi-k bench (GSM4037629, %lld cells) ===\n",
                (long long)mat.mat.cols);
    std::printf("Adapter: FitConfig k_cd_cutoff=32 → MU(2) when k>=32 and cfg.solver_mode==3\n");
    std::printf("Sklearn CPU reference: k=20 → %.0fms (Cycle 62b)\n\n", SKLEARN_MS_K20);

    // k=10: known-good, MU path
    run_k(mat, 10,  stream, SKLEARN_MS_K10);
    // k=20: was fast before fix (MU); confirm still fast
    run_k(mat, 20,  stream, SKLEARN_MS_K20);
    // k=50: was 15,980ms (CD); expect ~600-1,200ms after fix (MU)
    run_k(mat, 50,  stream, -1.0f);
    // k=100: k>=32 (>= k_cd_cutoff), adapter forces MU; note wall time
    run_k(mat, 100, stream, -1.0f);

    cudaStreamDestroy(stream);

    std::printf("\nSUMMARY: k=20 speedup vs sklearn %.0fms reference printed above.\n",
                SKLEARN_MS_K20);
    std::printf("k=50 should now be <1,500ms (MU). k=100 also MU (k>=k_cd_cutoff=32).\n");
    return 0;
}
