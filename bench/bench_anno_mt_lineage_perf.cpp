// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_anno_mt_lineage_perf.cpp
//
// Performance benchmark: singlet::gpu::anno::call_clones (mt_lineage.h)
//   vs mgatk R (cycle 15).
//
// Design reference: singlet/gpu/state/designs/15-mt-lineage.md
// Kernel:           include/singlet/gpu/anno/mt_lineage.h
//
// Staged pipeline (NOT timed):
//   - Load mt_alleles.1pz from GSM4037629 if present (alt + depth CSC matrices).
//   - If absent: synthesize a tiny 80-site × 11560-cell sparse CSC pair.
//
// Timed (cuEvents):
//   - call_clones(alt_csc, depth_csc, cfg, stream)
//   Upstream lognorm / HVG / PCA are NOT timed.
//
// Warmup: 3 discarded iterations.
// Timed:  5 iterations.
//
// SOTA reference: bench/refs/mt_lineage_ref.py (mgatk Python/R subprocess).
//   Falls back to TBD if mgatk is not installed.
//
// Skip conditions:
//   - No CUDA-capable GPU: print message, return 0.

#include "singlet/gpu/bench/harness.h"

#include <singlet/gpu/anno/mt_lineage.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ─── Constants ────────────────────────────────────────────────────────────────

static constexpr int      kNCells   = 11560;  // canonical sample cell count
static constexpr int      kNSites   = 80;     // synthetic MT site count (realistic)
static constexpr float    kAltDens  = 0.15f;  // ~15% cells have alt alleles at each site
static constexpr float    kDepDens  = 0.90f;  // ~90% cells have depth at each site
static constexpr uint64_t kSeed     = 0xDEADMT42ULL;

// GSM4037629 mt_alleles path — contains two layers: alt counts + total depth.
static const char* kMtAllelePath =
    "/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna/"
    "GSE127/GSE127918/GSM4037629/mt_alleles.1pz";

// ─── Helpers ──────────────────────────────────────────────────────────────────

namespace {

// Synthesize a sparse CSC for MT allele counting (n_cells cols, n_sites rows).
// alt_csc: ~15% cells have nonzero alt alleles per site.
// dep_csc: ~90% cells have depth per site.
singlet::gpu::core::DeviceCSC make_mt_csc(
        int n_cells, int n_sites, float density, float val_max,
        uint64_t seed, cudaStream_t stream)
{
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution bern(density);
    std::uniform_real_distribution<float> vdist(1.f, val_max);

    // CSC: rows=n_sites, cols=n_cells.
    std::vector<int>   col_ptr(n_cells + 1, 0);
    std::vector<int>   row_idx;
    std::vector<float> vals;

    for (int c = 0; c < n_cells; ++c) {
        col_ptr[c] = static_cast<int>(row_idx.size());
        for (int s = 0; s < n_sites; ++s) {
            if (bern(rng)) {
                row_idx.push_back(s);
                vals.push_back(vdist(rng));
            }
        }
    }
    col_ptr[n_cells] = static_cast<int>(row_idx.size());
    const int64_t nnz = static_cast<int64_t>(row_idx.size());

    singlet::gpu::core::DeviceMemory<int>   d_cp(n_cells + 1);
    singlet::gpu::core::DeviceMemory<int>   d_ri(nnz);
    singlet::gpu::core::DeviceMemory<float> d_vl(nnz);
    cudaMemcpyAsync(d_cp.data(), col_ptr.data(), (n_cells + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ri.data(), row_idx.data(), nnz * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_vl.data(), vals.data(), nnz * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return singlet::gpu::core::DeviceCSC(n_sites, n_cells, nnz,
                                        std::move(d_cp), std::move(d_ri), std::move(d_vl));
}

// Run mgatk SOTA reference subprocess.
// bench/refs/mt_lineage_ref.py accepts --n-cells and --n-sites, emits
// SOTA_WALL_SEC / SOTA_MEM_MB on stdout.
std::pair<float, float> run_mt_lineage_sota(int n_cells, int n_sites)
{
    std::string cmd =
        "python3 bench/refs/mt_lineage_ref.py"
        " --n-cells " + std::to_string(n_cells) +
        " --n-sites " + std::to_string(n_sites) +
        " 2>/dev/null";
    float wall_sec = -1.f, mem_mb = -1.f;
    FILE* fp = popen(cmd.c_str(), "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "SOTA_WALL_SEC=", 14) == 0) wall_sec = atof(line + 14);
            if (strncmp(line, "SOTA_MEM_MB=",   12) == 0) mem_mb   = atof(line + 12);
        }
        pclose(fp);
    }
    return {wall_sec, mem_mb};
}

}  // anonymous namespace

// ─── Main ─────────────────────────────────────────────────────────────────────

int main()
{
    if (!gpu_available()) {
        skip_no_gpu();
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    int   n_cells   = kNCells;
    int   n_sites   = kNSites;
    const char* scale_label = "tiny-synthetic";

    // ── Load or synthesize alt + depth matrices (NOT timed) ──────────────────
    // The mt_alleles.1pz file stores alt and depth as two separate layers.
    // For the bench we construct a synthetic pair directly when the file is absent.
    // WHY separate CSCs: call_clones() takes alt_counts and depth_counts separately
    // (both are DeviceCSC with rows=MT sites, cols=cells).
    singlet::gpu::core::DeviceCSC alt_csc, dep_csc;
    bool from_file = false;

    if (fs::exists(kMtAllelePath)) {
        try {
            // Attempt to load mt_alleles.1pz.
            // The .1pz bundles alt and depth as separate layers; pz_device_loader.h
            // exposes them via load_pz_mt() which returns a pair of DeviceCSC.
            // If the loader API is not yet finalized, fall through to synthetic.
            // (No io loader for MT yet — use synthetic.)
            std::printf("[mt_lineage_bench] mt_alleles.1pz found but MT loader not yet "
                        "finalized — using synthetic data.\n");
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[mt_lineage_bench] load failed: %s\n", ex.what());
        }
    }

    if (!from_file) {
        std::printf("[mt_lineage_bench] Synthesizing MT CSCs: %d sites × %d cells\n",
                    n_sites, n_cells);
        alt_csc = make_mt_csc(n_cells, n_sites, kAltDens, 30.f, kSeed,            stream);
        dep_csc = make_mt_csc(n_cells, n_sites, kDepDens, 200.f, kSeed ^ 0xFFFF,  stream);
    }

    // ── Config ────────────────────────────────────────────────────────────────
    singlet::gpu::anno::MtLineageConfig cfg;
    cfg.min_depth     = 5;
    cfg.min_cells_alt = 3;
    cfg.min_vaf       = 0.01f;
    cfg.max_em_iters  = 50;
    cfg.em_tol        = 1e-4f;
    cfg.min_K         = 2;
    cfg.max_K         = 5;  // smaller sweep for bench speed
    cfg.seed          = kSeed;

    // ── Warmup ────────────────────────────────────────────────────────────────
    std::printf("[mt_lineage_bench] Warming up call_clones (%d iters)...\n",
                BENCH_WARMUP_ITERS);
    for (int i = 0; i < BENCH_WARMUP_ITERS; ++i) {
        try {
            auto r = singlet::gpu::anno::call_clones(alt_csc, dep_csc, cfg, stream);
            cudaStreamSynchronize(stream);
            (void)r;
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[mt_lineage_bench] warmup error: %s\n", ex.what());
            cudaStreamDestroy(stream);
            return 0;
        }
    }

    // ── Timed iterations ──────────────────────────────────────────────────────
    BenchTimer        timer(stream);
    PeakMemoryTracker mem_tracker;

    for (int i = 0; i < BENCH_TIMED_ITERS; ++i) {
        mem_tracker.snapshot_before();
        timer.start();
        auto r = singlet::gpu::anno::call_clones(alt_csc, dep_csc, cfg, stream);
        timer.stop();
        timer.record();
        mem_tracker.snapshot_after();
        (void)r;
        std::printf("[mt_lineage_bench] iter %d done\n", i);
    }

    std::printf("[mt_lineage_bench] median=%.1fms min=%.1fms max=%.1fms mem=%.1fMB\n",
                timer.median_ms(), timer.min_ms(), timer.max_ms(), mem_tracker.peak_mb());

    // ── SOTA reference ────────────────────────────────────────────────────────
    auto [ref_wall, ref_mem] = run_mt_lineage_sota(n_cells, n_sites);

    // ── Log BenchRow ──────────────────────────────────────────────────────────
    BenchRow row;
    row.date          = current_date_str();
    row.feature       = "anno/mt_lineage";
    row.scale         = scale_label;
    row.impl          = "singlet-gpu";
    row.wall_ms_min   = timer.min_ms();
    row.wall_ms_med   = timer.median_ms();
    row.wall_ms_max   = timer.max_ms();
    row.mem_mb_peak   = mem_tracker.peak_mb();
    row.cells_per_sec = throughput(n_cells, row.wall_ms_med);
    row.sota_wall_sec = ref_wall;
    row.sota_mem_mb   = ref_mem;
    row.commit        = "no-git";
    log_row(row);

    std::printf("[mt_lineage_bench] Done.\n");
    cudaStreamDestroy(stream);
    return 0;
}
