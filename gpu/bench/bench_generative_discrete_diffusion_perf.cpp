// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_generative_discrete_diffusion_perf.cpp
//
// Performance benchmark: singlet_gpu::generative::train_discrete_diffusion
//
// Design reference: singlet-gpu/state/designs/30-discrete-diffusion.md
// Kernel reference:  include/singlet-gpu/generative/discrete_diffusion.h
//
// Staged pipeline:
//   load .1pz counts (or synthesize) → train_discrete_diffusion (fit only)
//
// Timing scope: train_discrete_diffusion() only (excludes load).
//   Sample step is NOT timed here — it is included as a note below.
//
// SOTA baseline: NONE.  discrete_diffusion for single-cell counts is first-GPU
//   and first of its kind.  This driver records "NO_SOTA" in the registry row
//   (sota_wall=0, ratio_wall=0) and logs just our numbers.
//
// 3 warmup + 5 timed iterations.
//
// Skip conditions:
//   - No CUDA-capable GPU → exit 0
//   - Sample not found → use tiny synthetic (500 spots × 200 genes), scale="tiny"
//
// NOTE on sample step: sample_cells() is a secondary output of the trained model.
// It is O(n_timesteps × n_samples) and separate from training. Timing it would
// require storing the trained model across bench iterations. Users who want to
// bench sampling should run the sample step independently.  A future driver
// bench_generative_discrete_diffusion_sample_perf.cpp can add that measurement.

#include <singlet_gpu/bench/harness.h>

#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/generative/discrete_diffusion.h>
#include <singlet-gpu/core/types.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ─── Constants ────────────────────────────────────────────────────────────────

static constexpr uint64_t kSeed         = 0xC0FFEE;
static constexpr int      kWarmup       = 3;
static constexpr int      kTimed        = 5;
// Reduced epoch count for bench timing: production default is 200.
// 10 epochs gives a stable GPU timing signal without multi-minute runs.
static constexpr int      kBenchEpochs  = 10;
static constexpr int      kBatchSize    = 256;
static constexpr int      kSyntheticN   = 2000;   // cells (small for synthetic bench)
static constexpr int      kSyntheticM   = 500;    // genes

static const char* kSamplePath =
    "/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/"
    "GSE127/GSE127918/GSM4037629/counts.1pz";

// ─── Helpers ──────────────────────────────────────────────────────────────────

namespace {

// Build a synthetic CSC count matrix (n_genes × n_cells) and upload to device.
singlet_gpu::core::DeviceCSC make_synthetic_csc(
        int n_genes, int n_cells, uint64_t seed, cudaStream_t stream)
{
    using namespace singlet_gpu::core;
    const double density = 0.05;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<int>   col_ptr(n_cells + 1, 0);
    std::vector<int>   row_idx;
    std::vector<float> values;
    row_idx.reserve(static_cast<size_t>(n_genes) * n_cells / 20);
    values.reserve(row_idx.capacity());

    for (int c = 0; c < n_cells; ++c) {
        col_ptr[c] = static_cast<int>(row_idx.size());
        for (int g = 0; g < n_genes; ++g) {
            if (uni(rng) < density) {
                row_idx.push_back(g);
                values.push_back(static_cast<float>(static_cast<int>(uni(rng) * 20.0) + 1));
            }
        }
    }
    col_ptr[n_cells] = static_cast<int>(row_idx.size());
    const int64_t nnz = static_cast<int64_t>(row_idx.size());

    DeviceMemory<int>   d_col_ptr(n_cells + 1);
    DeviceMemory<int>   d_row_idx(nnz);
    DeviceMemory<float> d_values (nnz);
    cudaMemcpyAsync(d_col_ptr.get(), col_ptr.data(), (n_cells + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_row_idx.get(), row_idx.data(), nnz * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_values.get(),  values.data(),  nnz * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return DeviceCSC(n_genes, n_cells, nnz,
                     std::move(d_col_ptr), std::move(d_row_idx), std::move(d_values));
}

// NO_SOTA: invoke bench/refs/discrete_diffusion_ref.py which writes a placeholder
// JSON row with wall_ms=-1 and impl="NO_SOTA".
singlet_gpu::bench::RefResult run_no_sota_ref()
{
    namespace bench = singlet_gpu::bench;
    fs::create_directories("/tmp/singlet_gpu_bench");
    const std::string out_path = "/tmp/singlet_gpu_bench/dd_bench_out.json";

    const std::string script = std::string(BENCH_REFS_DIR) + "/discrete_diffusion_ref.py";
    std::string cmd = "python3 " + script +
                      " --timing-json " + out_path +
                      " 2>/dev/null";
    return bench::run_python_reference(cmd, out_path);
}

}  // anonymous namespace

// ─── Main ─────────────────────────────────────────────────────────────────────

int main()
{
    namespace bench = singlet_gpu::bench;

    // ── GPU check ──────────────────────────────────────────────────────────────
    int n_dev = 0;
    cudaGetDeviceCount(&n_dev);
    if (n_dev == 0) {
        bench::skip("bench_generative_discrete_diffusion_perf", "no CUDA-capable GPU");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // ── Stage 1: load or synthesize matrix ────────────────────────────────────
    int    n_cells     = kSyntheticN;
    int    n_genes     = kSyntheticM;
    const char* scale_label = "tiny";
    bool from_file = false;

    singlet_gpu::core::DeviceCSC mat;

    if (fs::exists(kSamplePath)) {
        try {
            auto loaded = singlet_gpu::io::load_pz(kSamplePath, stream);
            cudaStreamSynchronize(stream);
            n_cells     = loaded.mat.cols;
            n_genes     = loaded.mat.rows;
            mat         = std::move(loaded.mat);
            scale_label = "10k";
            from_file   = true;
            std::printf("[dd_bench] .1pz: %d genes × %d cells\n", n_genes, n_cells);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[dd_bench] .1pz load failed: %s — synthetic\n", ex.what());
        }
    }

    if (!from_file) {
        std::printf("[dd_bench] Synthesizing CSC %d × %d (tiny)\n", n_genes, n_cells);
        mat = make_synthetic_csc(n_genes, n_cells, kSeed, stream);
    }

    // ── Discrete diffusion config (bench-tuned) ────────────────────────────────
    singlet_gpu::generative::DiscreteDiffusionConfig cfg;
    cfg.n_timesteps  = 100;
    cfg.vocab_size   = 16;
    cfg.d_hidden     = 128;   // smaller than production (256) to reduce per-iter time
    cfg.n_heads      = 4;
    cfg.n_layers     = 2;     // reduced from 4 for bench speed
    cfg.n_epochs     = kBenchEpochs;
    cfg.batch_size   = kBatchSize;
    cfg.learning_rate = 1e-4f;
    cfg.seed         = kSeed;

    // ── Warmup (3 discarded iterations) ───────────────────────────────────────
    std::printf("[dd_bench] Warming up (%d iters, %d epochs each)...\n",
                kWarmup, kBenchEpochs);
    for (int i = 0; i < kWarmup; ++i) {
        auto res = singlet_gpu::generative::train_discrete_diffusion(mat, cfg, stream);
        cudaStreamSynchronize(stream);
        (void)res;
    }

    // ── Timed iterations (5) ──────────────────────────────────────────────────
    bench::BenchTimer     timer;
    bench::PeakMemTracker mem_tracker;
    std::vector<double>   wall_ms_vec(kTimed);

    for (int i = 0; i < kTimed; ++i) {
        mem_tracker.sample_before();
        timer.start(stream);
        auto res = singlet_gpu::generative::train_discrete_diffusion(mat, cfg, stream);
        timer.stop(stream);
        cudaStreamSynchronize(stream);
        mem_tracker.sample_after();
        wall_ms_vec[i] = timer.elapsed_ms();
        double final_loss = res.loss_history.empty() ? 0.0
                          : static_cast<double>(res.loss_history.back());
        std::printf("[dd_bench] iter %d: %.1f ms  final_loss=%.4f\n",
                    i, wall_ms_vec[i], final_loss);
        (void)res;
    }

    // ── Statistics ─────────────────────────────────────────────────────────────
    std::sort(wall_ms_vec.begin(), wall_ms_vec.end());
    double wall_median = wall_ms_vec[kTimed / 2];
    double wall_min    = wall_ms_vec.front();
    double wall_max    = wall_ms_vec.back();
    double peak_mb     = mem_tracker.peak_delta_mb();
    double tput        = bench::throughput(n_cells, wall_median);

    std::printf("[dd_bench] Result: median=%.1fms min=%.1fms max=%.1fms "
                "mem=%.1fMB cells/s=%.0f  [NO_SOTA — first GPU impl]\n",
                wall_median, wall_min, wall_max, peak_mb, tput);

    // ── NO_SOTA reference ──────────────────────────────────────────────────────
    // discrete_diffusion_ref.py writes a placeholder row with impl="NO_SOTA".
    bench::RefResult ref = run_no_sota_ref();
    (void)ref;  // RefResult.wall_ms will be -1; ratio_wall logged as 0 (N/A)

    // ── Log BenchRow ───────────────────────────────────────────────────────────
    bench::BenchRow row;
    row.date          = bench::today_iso();
    row.feature       = "generative/discrete_diffusion";
    row.scale         = scale_label;
    row.impl          = "singlet-gpu";
    row.wall_ms       = wall_median;
    row.mem_mb        = peak_mb;
    row.cells_per_sec = tput;
    // NO_SOTA: leave sota fields at -1 (harness prints "—" for these).
    row.sota_wall     = -1.0;
    row.sota_mem      = -1.0;
    row.ratio_wall    = 0.0;
    row.ratio_mem     = 0.0;
    row.commit        = bench::git_short_sha();
    bench::log_row(row);

    cudaStreamDestroy(stream);
    return 0;
}
