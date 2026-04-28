// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_spatial_stagate_perf.cpp
//
// Performance benchmark: singlet_gpu::spatial::run_stagate vs STAGATE Python.
//
// Design reference: singlet-gpu/state/designs/29-stagate.md
// Kernel reference:  include/singlet-gpu/spatial/stagate.h
//
// Staged pipeline:
//   load .1pz counts + optional spatial_coords.parquet (or synthesize 2D layout)
//   → run_stagate (GAT autoencoder + optional Leiden)
//
// Timing scope: run_stagate() only (excludes load and coord synthesis).
//   train is the primary timed operation; the optional Leiden clustering post-step
//   is included because it runs inside run_stagate() and is inseparable from the
//   returned StagateResult.
//
// 3 warmup + 5 timed iterations.
// SOTA reference: bench/refs/stagate_ref.py (STAGATE Python via subprocess).
//   If STAGATE Python is absent the driver logs sota_wall=TBD and continues.
//
// Skip conditions:
//   - No CUDA-capable GPU (cudaGetDeviceCount == 0) → exit 0
//   - Sample not found → use synthetic 500 spots × 200 genes, scale="tiny"

#include <singlet_gpu/bench/harness.h>

#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/spatial/stagate.h>
#include <singlet-gpu/core/types.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ─── Constants ────────────────────────────────────────────────────────────────

static constexpr uint64_t kSeed        = 0xC0FFEE;
static constexpr int      kWarmup      = 3;
static constexpr int      kTimed       = 5;
// Smaller epoch count for bench timing — keeps per-iteration wall time finite.
// The production default (500) would run for minutes on 10k+ spots.
static constexpr int      kBenchEpochs = 20;

static const char* kSamplePath =
    "/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/"
    "GSE127/GSE127918/GSM4037629/counts.1pz";

// Optional spatial_coords.parquet at the same directory.
static const char* kCoordsPath =
    "/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/"
    "GSE127/GSE127918/GSM4037629/spatial_coords.parquet";

// ─── Helpers ──────────────────────────────────────────────────────────────────

namespace {

// Build a synthetic CSC count matrix (n_genes × n_spots) and upload to device.
singlet_gpu::core::DeviceCSC make_synthetic_csc(
        int n_genes, int n_spots, uint64_t seed, cudaStream_t stream)
{
    using namespace singlet_gpu::core;
    const double density = 0.05;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<int>   col_ptr(n_spots + 1, 0);
    std::vector<int>   row_idx;
    std::vector<float> values;
    row_idx.reserve(static_cast<size_t>(n_genes) * n_spots / 20);
    values.reserve(row_idx.capacity());

    for (int c = 0; c < n_spots; ++c) {
        col_ptr[c] = static_cast<int>(row_idx.size());
        for (int g = 0; g < n_genes; ++g) {
            if (uni(rng) < density) {
                row_idx.push_back(g);
                values.push_back(static_cast<float>(static_cast<int>(uni(rng) * 20.0) + 1));
            }
        }
    }
    col_ptr[n_spots] = static_cast<int>(row_idx.size());
    const int64_t nnz = static_cast<int64_t>(row_idx.size());

    DeviceMemory<int>   d_col_ptr(n_spots + 1);
    DeviceMemory<int>   d_row_idx(nnz);
    DeviceMemory<float> d_values (nnz);
    cudaMemcpyAsync(d_col_ptr.get(), col_ptr.data(), (n_spots + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_row_idx.get(), row_idx.data(), nnz * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_values.get(),  values.data(),  nnz * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return DeviceCSC(n_genes, n_spots, nnz,
                     std::move(d_col_ptr), std::move(d_row_idx), std::move(d_values));
}

// Generate synthetic 2D hexagonal spatial coordinates (n × 2) and upload.
// Uses a simple row-major hex grid so spots have plausible nearest-neighbor structure.
singlet_gpu::core::DeviceDense make_synthetic_coords(
        int n_spots, uint64_t seed, cudaStream_t stream)
{
    // Side length of hex grid: ceil(sqrt(n_spots))
    int side = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n_spots))));
    std::vector<float> host;
    host.reserve(n_spots * 2);
    for (int row = 0; row < side && static_cast<int>(host.size()) / 2 < n_spots; ++row) {
        float offset = (row % 2) ? 0.5f : 0.0f;
        for (int col = 0; col < side && static_cast<int>(host.size()) / 2 < n_spots; ++col) {
            host.push_back(static_cast<float>(col) + offset);
            host.push_back(static_cast<float>(row) * 0.866f);  // sin(60°)
        }
    }
    host.resize(n_spots * 2);

    singlet_gpu::core::DeviceMemory<float> dev(static_cast<size_t>(n_spots) * 2);
    cudaMemcpyAsync(dev.get(), host.data(), host.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return singlet_gpu::core::DeviceDense(n_spots, 2, std::move(dev));
    (void)seed;
}

// Write float32 (N, 2) coords as .npy for the Python reference.
void write_npy_f32_2d(const std::string& path, const float* data, int n, int d)
{
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                      std::to_string(n) + ", " + std::to_string(d) + "), }";
    constexpr int kPre = 10;
    int pad = ((static_cast<int>(hdr.size()) + kPre + 63) / 64) * 64 - kPre;
    while (static_cast<int>(hdr.size()) < pad - 1) hdr += ' ';
    hdr += '\n';
    uint16_t hl = static_cast<uint16_t>(hdr.size());
    std::ofstream f(path, std::ios::binary);
    const char magic[] = "\x93NUMPY";
    f.write(magic, 6); f.put(1); f.put(0);
    f.write(reinterpret_cast<const char*>(&hl), 2);
    f.write(hdr.data(), hl);
    f.write(reinterpret_cast<const char*>(data), static_cast<size_t>(n) * d * sizeof(float));
}

// Write CSC binary (shared dump_csc format) for the Python reference.
void write_csc_bin(const std::string& path,
                   const float* h_vals, const int* h_indptr, const int* h_rowids,
                   int n_rows, int n_cols, int64_t nnz)
{
    std::ofstream f(path, std::ios::binary);
    const uint32_t magic = 0x43535343u;
    f.write(reinterpret_cast<const char*>(&magic), 4);
    const uint32_t nr = static_cast<uint32_t>(n_rows);
    const uint32_t nc = static_cast<uint32_t>(n_cols);
    f.write(reinterpret_cast<const char*>(&nr), 4);
    f.write(reinterpret_cast<const char*>(&nc), 4);
    const uint64_t nz = static_cast<uint64_t>(nnz);
    f.write(reinterpret_cast<const char*>(&nz), 8);
    f.write(reinterpret_cast<const char*>(h_vals),   nnz * sizeof(float));
    f.write(reinterpret_cast<const char*>(h_indptr), (n_cols + 1) * sizeof(int));
    f.write(reinterpret_cast<const char*>(h_rowids), nnz * sizeof(int));
}

// Invoke bench/refs/stagate_ref.py and return (wall_ms, mem_mb).
// Returns wall_ms=-1, mem_mb=-1 if the script is absent or fails.
singlet_gpu::bench::RefResult run_stagate_reference(
        const singlet_gpu::core::DeviceCSC&   mat,
        const singlet_gpu::core::DeviceDense& coords,
        int n_spots, int n_genes, int64_t nnz,
        int n_neighbors, int n_epochs)
{
    namespace bench = singlet_gpu::bench;
    fs::create_directories("/tmp/singlet_gpu_bench");
    const std::string mat_path    = "/tmp/singlet_gpu_bench/stagate_bench_mat.bin";
    const std::string coords_path = "/tmp/singlet_gpu_bench/stagate_bench_coords.npy";
    const std::string out_path    = "/tmp/singlet_gpu_bench/stagate_bench_out.json";

    // Download CSC + coords to host.
    std::vector<float> h_vals  (nnz);
    std::vector<int>   h_indptr(n_spots + 1);
    std::vector<int>   h_rowids(nnz);
    std::vector<float> h_coords(n_spots * 2);
    cudaMemcpy(h_vals.data(),   mat.values.get(),     nnz * sizeof(float),         cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indptr.data(), mat.col_ptr.get(),    (n_spots + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rowids.data(), mat.row_indices.get(),nnz * sizeof(int),           cudaMemcpyDeviceToHost);
    cudaMemcpy(h_coords.data(), coords.data().get(), n_spots * 2 * sizeof(float), cudaMemcpyDeviceToHost);

    write_csc_bin(mat_path, h_vals.data(), h_indptr.data(), h_rowids.data(),
                  n_genes, n_spots, nnz);
    write_npy_f32_2d(coords_path, h_coords.data(), n_spots, 2);

    const std::string script = std::string(BENCH_REFS_DIR) + "/stagate_ref.py";
    std::string cmd = "python3 " + script +
                      " --input "       + mat_path    +
                      " --coords "      + coords_path +
                      " --n-neighbors " + std::to_string(n_neighbors) +
                      " --n-epochs "    + std::to_string(n_epochs)    +
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
        bench::skip("bench_spatial_stagate_perf", "no CUDA-capable GPU");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // ── Stage 1: load or synthesize matrix + coords ────────────────────────────
    int    n_spots     = 500;
    int    n_genes     = 200;
    int64_t nnz        = 0;
    const char* scale_label = "tiny";
    bool from_file = false;

    singlet_gpu::core::DeviceCSC   mat;
    singlet_gpu::core::DeviceDense coords;

    if (fs::exists(kSamplePath)) {
        try {
            auto loaded = singlet_gpu::io::load_pz(kSamplePath, stream);
            cudaStreamSynchronize(stream);
            n_spots     = loaded.mat.cols;
            n_genes     = loaded.mat.rows;
            nnz         = loaded.mat.nnz;
            mat         = std::move(loaded.mat);
            scale_label = "10k";
            from_file   = true;
            std::printf("[stagate_bench] .1pz: %d genes × %d spots, nnz=%lld\n",
                        n_genes, n_spots, (long long)nnz);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[stagate_bench] .1pz load failed: %s — synthetic\n", ex.what());
        }
    }

    if (!from_file) {
        std::printf("[stagate_bench] Synthesizing CSC %d × %d (tiny)\n", n_genes, n_spots);
        mat = make_synthetic_csc(n_genes, n_spots, kSeed, stream);
        nnz = mat.nnz;
    }

    // Spatial coords: prefer real parquet (if spatial sample), else synthetic hex grid.
    // For this scRNA sample (no Visium), always synthesize 2D coords.
    std::printf("[stagate_bench] Synthesizing hex spatial coords for %d spots\n", n_spots);
    coords = make_synthetic_coords(n_spots, kSeed, stream);

    // ── STAGATE config (bench-tuned epoch count) ───────────────────────────────
    singlet_gpu::spatial::StagateConfig cfg;
    cfg.n_neighbors     = 6;
    cfg.d_hidden        = 256;
    cfg.d_embed         = 64;
    cfg.n_epochs        = kBenchEpochs;  // reduced for bench timing
    cfg.learning_rate   = 1e-3f;
    cfg.run_post_leiden = false;          // exclude Leiden from timing for comparability
    cfg.seed            = kSeed;

    // ── Warmup (3 discarded iterations) ───────────────────────────────────────
    std::printf("[stagate_bench] Warming up (%d iters, %d epochs each)...\n",
                kWarmup, kBenchEpochs);
    for (int i = 0; i < kWarmup; ++i) {
        auto res = singlet_gpu::spatial::run_stagate(mat, coords, cfg, stream);
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
        auto res = singlet_gpu::spatial::run_stagate(mat, coords, cfg, stream);
        timer.stop(stream);
        cudaStreamSynchronize(stream);
        mem_tracker.sample_after();
        wall_ms_vec[i] = timer.elapsed_ms();
        std::printf("[stagate_bench] iter %d: %.1f ms  final_loss=%.4f\n",
                    i, wall_ms_vec[i],
                    res.loss_history.empty() ? 0.0f : res.loss_history.back());
        (void)res;
    }

    // ── Statistics ─────────────────────────────────────────────────────────────
    std::sort(wall_ms_vec.begin(), wall_ms_vec.end());
    double wall_median = wall_ms_vec[kTimed / 2];
    double wall_min    = wall_ms_vec.front();
    double wall_max    = wall_ms_vec.back();
    double peak_mb     = mem_tracker.peak_delta_mb();
    double tput        = bench::throughput(n_spots, wall_median);

    std::printf("[stagate_bench] Result: median=%.1fms min=%.1fms max=%.1fms "
                "mem=%.1fMB spots/s=%.0f\n",
                wall_median, wall_min, wall_max, peak_mb, tput);

    // ── SOTA reference (STAGATE Python) ───────────────────────────────────────
    bench::RefResult ref = run_stagate_reference(mat, coords, n_spots, n_genes, nnz,
                                                  cfg.n_neighbors, kBenchEpochs);

    // ── Log BenchRow ───────────────────────────────────────────────────────────
    bench::BenchRow row;
    row.date          = bench::today_iso();
    row.feature       = "spatial/stagate";
    row.scale         = scale_label;
    row.impl          = "singlet-gpu";
    row.wall_ms       = wall_median;
    row.mem_mb        = peak_mb;
    row.cells_per_sec = tput;
    row.sota_wall     = ref.wall_ms;
    row.sota_mem      = ref.mem_mb;
    row.ratio_wall    = (ref.wall_ms > 0) ? ref.wall_ms / wall_median : 0.0;
    row.ratio_mem     = (ref.mem_mb  > 0) ? ref.mem_mb  / peak_mb    : 0.0;
    row.commit        = bench::git_short_sha();
    bench::log_row(row);

    cudaStreamDestroy(stream);
    return 0;
}
