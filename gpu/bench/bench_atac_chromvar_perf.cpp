// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_atac_chromvar_perf.cpp
//
// Performance benchmark: singlet_gpu::atac::chromvar vs chromVAR R.
//
// Design reference: singlet-gpu/state/designs/34-atac-chromvar.md
// Kernel reference:  include/singlet-gpu/atac/chromvar.h
//
// Staged pipeline:
//   1. Load fragments.1pz (n_peaks × n_cells) OR synthesize ATAC accessibility.
//   2. Synthesize motif_in_peak binary matrix (n_motifs × n_peaks).
//   3. Time chromvar() call (all steps: observed score, background, permutations).
//
// Timing scope: chromvar() only (excludes data load + motif synthesis).
// Background n_bg=50, permutations=50 (reduced for bench repeatability).
// 3 warmup + 5 timed iterations.
// SOTA reference: bench/refs/chromvar_ref.py (chromVAR R via subprocess).
//
// Skip conditions:
//   - No CUDA-capable GPU
//   - DeviceCSC construction fails for accessibility

#include <singlet_gpu/bench/harness.h>

#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/atac/chromvar.h>
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/core/memory.h>

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

static constexpr int      kNPeaks     = 2000;   // synthetic peak count (small for bench)
static constexpr int      kNMotifs    = 50;     // motifs
static constexpr int      kNBg        = 50;     // background iterations
static constexpr int      kNPerm      = 50;     // permutations (reduced from 1000)
static constexpr int      kCellBatch  = 2048;   // cells per OOC batch
static constexpr uint64_t kSeed       = 0xC0FFEE + 34;
static constexpr int      kWarmup     = 3;
static constexpr int      kTimed      = 5;
static constexpr int      kSyntheticN = 5000;   // cells (accessibility)

// fragments.1pz is the canonical ATAC input (peaks × cells, uint32 counts).
// Fall back to synthetic if absent.
static const char* kSamplePath =
    "/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/"
    "GSE127/GSE127918/GSM4037629/counts.1pz";

// ─── Helpers ──────────────────────────────────────────────────────────────────

namespace {

// Build a synthetic sparse CSC (peaks × cells) for ATAC accessibility.
// Uses Bernoulli density ~3% — typical for single-cell ATAC-seq data.
singlet_gpu::core::DeviceCSC make_atac_csc(
        int n_peaks, int n_cells, uint64_t seed, cudaStream_t stream)
{
    using namespace singlet_gpu::core;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> udist(0.0f, 1.0f);
    const double density = 0.03;

    std::vector<int>   col_ptr(n_cells + 1, 0);
    std::vector<int>   row_idx;
    std::vector<float> values;
    row_idx.reserve(static_cast<size_t>(n_peaks * n_cells) * 4);
    values.reserve(row_idx.capacity());

    for (int c = 0; c < n_cells; ++c) {
        col_ptr[c] = static_cast<int>(row_idx.size());
        for (int p = 0; p < n_peaks; ++p) {
            if (udist(rng) < density) {
                row_idx.push_back(p);
                // Fragment count in [1,5] — typical for sparse ATAC
                values.push_back(static_cast<float>(
                    1 + static_cast<int>(udist(rng) * 5.0f)));
            }
        }
    }
    col_ptr[n_cells] = static_cast<int>(row_idx.size());
    const int64_t nnz = static_cast<int64_t>(row_idx.size());

    DeviceMemory<int>   d_col_ptr(n_cells + 1);
    DeviceMemory<int>   d_row_idx(nnz);
    DeviceMemory<float> d_values(nnz);
    cudaMemcpyAsync(d_col_ptr.get(), col_ptr.data(), (n_cells + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_row_idx.get(), row_idx.data(), nnz * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_values.get(),  values.data(),  nnz * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return DeviceCSC(n_peaks, n_cells, nnz,
                     std::move(d_col_ptr), std::move(d_row_idx), std::move(d_values));
}

// Build a binary motif × peak CSC (sparse — each motif hits ~5% of peaks).
singlet_gpu::core::DeviceCSC make_motif_csc(
        int n_motifs, int n_peaks, uint64_t seed, cudaStream_t stream)
{
    using namespace singlet_gpu::core;
    std::mt19937_64 rng(seed ^ 0xABC123);
    std::uniform_real_distribution<float> udist(0.0f, 1.0f);
    const double density = 0.05;

    // motif_in_peak stored as CSC on peaks dimension:
    // rows=motifs, cols=peaks → CSC: col_ptr has size n_peaks+1
    std::vector<int>   col_ptr(n_peaks + 1, 0);
    std::vector<int>   row_idx;
    std::vector<float> values;
    row_idx.reserve(static_cast<size_t>(n_motifs * n_peaks) * 6);
    values.reserve(row_idx.capacity());

    for (int p = 0; p < n_peaks; ++p) {
        col_ptr[p] = static_cast<int>(row_idx.size());
        for (int m = 0; m < n_motifs; ++m) {
            if (udist(rng) < density) {
                row_idx.push_back(m);
                values.push_back(1.0f);   // binary
            }
        }
    }
    col_ptr[n_peaks] = static_cast<int>(row_idx.size());
    const int64_t nnz = static_cast<int64_t>(row_idx.size());

    DeviceMemory<int>   d_col_ptr(n_peaks + 1);
    DeviceMemory<int>   d_row_idx(nnz);
    DeviceMemory<float> d_values(nnz);
    cudaMemcpyAsync(d_col_ptr.get(), col_ptr.data(), (n_peaks + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_row_idx.get(), row_idx.data(), nnz * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_values.get(),  values.data(),  nnz * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return DeviceCSC(n_motifs, n_peaks, nnz,
                     std::move(d_col_ptr), std::move(d_row_idx), std::move(d_values));
}

// Upload per-peak float vectors to device.
singlet_gpu::core::DeviceMemory<float> upload_f32(
        const std::vector<float>& h, cudaStream_t stream)
{
    singlet_gpu::core::DeviceMemory<float> d(h.size());
    cudaMemcpyAsync(d.get(), h.data(), h.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    return d;
}

// Write float32 .npy for R reference.
void write_npy_f32_2d(const std::string& path, const float* data, int rows, int cols) {
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                      std::to_string(rows) + ", " + std::to_string(cols) + "), }";
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
    f.write(reinterpret_cast<const char*>(data), rows * cols * sizeof(float));
}

void write_npy_f32_1d(const std::string& path, const float* data, int n) {
    write_npy_f32_2d(path, data, n, 1);
}

singlet_gpu::bench::RefResult run_chromvar_reference(
        const singlet_gpu::core::DeviceCSC& accessibility,
        const singlet_gpu::core::DeviceCSC& motif_in_peak,
        const singlet_gpu::core::DeviceMemory<float>& peak_gc,
        int n_peaks, int n_cells, int n_motifs)
{
    namespace bench = singlet_gpu::bench;
    fs::create_directories("/tmp/singlet_gpu_bench");

    const std::string access_path = "/tmp/singlet_gpu_bench/chromvar_bench_access.npy";
    const std::string motif_path  = "/tmp/singlet_gpu_bench/chromvar_bench_motif.npy";
    const std::string gc_path     = "/tmp/singlet_gpu_bench/chromvar_bench_gc.npy";
    const std::string out_path    = "/tmp/singlet_gpu_bench/chromvar_bench_out.json";

    // Download accessibility dense matrix (peaks × cells) for R reference.
    // We download as a dense float32 matrix.
    std::vector<float> h_access(static_cast<size_t>(n_peaks) * n_cells, 0.0f);
    {
        // Reconstruct dense from CSC on host.
        int64_t nnz = accessibility.nnz;
        std::vector<float> h_vals(nnz);
        std::vector<int>   h_indptr(n_cells + 1);
        std::vector<int>   h_rowids(nnz);
        cudaMemcpy(h_vals.data(),   accessibility.values.get(),     nnz * sizeof(float),         cudaMemcpyDeviceToHost);
        cudaMemcpy(h_indptr.data(), accessibility.col_ptr.get(),    (n_cells + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rowids.data(), accessibility.row_indices.get(),nnz * sizeof(int),            cudaMemcpyDeviceToHost);
        for (int c = 0; c < n_cells; ++c) {
            for (int k = h_indptr[c]; k < h_indptr[c + 1]; ++k) {
                int peak = h_rowids[k];
                h_access[static_cast<size_t>(peak) * n_cells + c] = h_vals[k];
            }
        }
    }
    write_npy_f32_2d(access_path, h_access.data(), n_peaks, n_cells);

    // Download motif_in_peak as dense (n_motifs × n_peaks).
    std::vector<float> h_motif(static_cast<size_t>(n_motifs) * n_peaks, 0.0f);
    {
        int64_t nnz = motif_in_peak.nnz;
        std::vector<float> h_vals(nnz);
        std::vector<int>   h_indptr(n_peaks + 1);
        std::vector<int>   h_rowids(nnz);
        cudaMemcpy(h_vals.data(),   motif_in_peak.values.get(),     nnz * sizeof(float),         cudaMemcpyDeviceToHost);
        cudaMemcpy(h_indptr.data(), motif_in_peak.col_ptr.get(),    (n_peaks + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rowids.data(), motif_in_peak.row_indices.get(),nnz * sizeof(int),            cudaMemcpyDeviceToHost);
        for (int p = 0; p < n_peaks; ++p) {
            for (int k = h_indptr[p]; k < h_indptr[p + 1]; ++k) {
                int motif = h_rowids[k];
                h_motif[static_cast<size_t>(motif) * n_peaks + p] = h_vals[k];
            }
        }
    }
    write_npy_f32_2d(motif_path, h_motif.data(), n_motifs, n_peaks);

    // Download per-peak GC content.
    std::vector<float> h_gc(n_peaks);
    cudaMemcpy(h_gc.data(), peak_gc.get(), n_peaks * sizeof(float), cudaMemcpyDeviceToHost);
    write_npy_f32_1d(gc_path, h_gc.data(), n_peaks);

    const std::string script = std::string(BENCH_REFS_DIR) + "/chromvar_ref.py";
    std::string cmd = "python3 " + script +
                      " --accessibility " + access_path +
                      " --motifs "        + motif_path  +
                      " --gc "            + gc_path     +
                      " --timing-json "   + out_path    +
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
        bench::skip("bench_atac_chromvar_perf", "no CUDA-capable GPU");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // ── Stage 1: build ATAC accessibility matrix ───────────────────────────────
    int n_peaks = kNPeaks;
    int n_cells = kSyntheticN;
    const char* scale_label = "10k-synthetic";

    // Try loading the canonical sample to get n_cells; keep synthetic peak count.
    if (fs::exists(kSamplePath)) {
        try {
            auto loaded = singlet_gpu::io::load_pz(kSamplePath, stream);
            cudaStreamSynchronize(stream);
            n_cells     = loaded.mat.cols;
            scale_label = "10k";
            std::printf("[chromvar_bench] .1pz cells=%d (peaks synthetic=%d)\n",
                        n_cells, n_peaks);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[chromvar_bench] .1pz load failed: %s — synthetic\n",
                         ex.what());
        }
    } else {
        std::printf("[chromvar_bench] No .1pz — synthetic %d peaks × %d cells\n",
                    n_peaks, n_cells);
    }

    std::printf("[chromvar_bench] Building ATAC CSC %d peaks × %d cells...\n",
                n_peaks, n_cells);
    singlet_gpu::core::DeviceCSC accessibility =
        make_atac_csc(n_peaks, n_cells, kSeed, stream);
    cudaStreamSynchronize(stream);

    // ── Stage 2: build motif_in_peak binary CSC ────────────────────────────────
    std::printf("[chromvar_bench] Building motif-in-peak CSC %d motifs × %d peaks...\n",
                kNMotifs, n_peaks);
    singlet_gpu::core::DeviceCSC motif_in_peak =
        make_motif_csc(kNMotifs, n_peaks, kSeed, stream);
    cudaStreamSynchronize(stream);

    // ── Stage 3: per-peak GC content + mean accessibility (required metadata) ──
    std::mt19937_64 rng(kSeed ^ 0x1234);
    std::uniform_real_distribution<float> gc_dist(0.3f, 0.7f);   // GC in [0.3, 0.7]
    std::uniform_real_distribution<float> ma_dist(0.01f, 0.5f);  // mean access

    std::vector<float> h_gc(n_peaks), h_ma(n_peaks);
    for (int i = 0; i < n_peaks; ++i) {
        h_gc[i] = gc_dist(rng);
        h_ma[i] = ma_dist(rng);
    }
    auto d_gc = upload_f32(h_gc, stream);
    auto d_ma = upload_f32(h_ma, stream);
    cudaStreamSynchronize(stream);

    // ── chromVAR config (reduced permutations for bench speed) ─────────────────
    singlet_gpu::atac::ChromVarConfig cfg;
    cfg.n_background_peaks = kNBg;
    cfg.n_permutations     = kNPerm;   // 50 rather than 1000
    cfg.cell_batch         = kCellBatch;
    cfg.bg_batch           = kNBg;
    cfg.perm_inner_batch   = 50;
    cfg.seed               = kSeed;

    // ── Warmup ─────────────────────────────────────────────────────────────────
    std::printf("[chromvar_bench] Warming up (%d iters)...\n", kWarmup);
    for (int i = 0; i < kWarmup; ++i) {
        auto res = singlet_gpu::atac::chromvar(
            accessibility, motif_in_peak, d_gc, d_ma, cfg, stream);
        cudaStreamSynchronize(stream);
        (void)res;
    }

    // ── Timed iterations ───────────────────────────────────────────────────────
    bench::BenchTimer     timer;
    bench::PeakMemTracker mem_tracker;
    std::vector<double>   wall_ms_vec(kTimed);

    for (int i = 0; i < kTimed; ++i) {
        mem_tracker.sample_before();
        timer.start(stream);
        auto res = singlet_gpu::atac::chromvar(
            accessibility, motif_in_peak, d_gc, d_ma, cfg, stream);
        timer.stop(stream);
        cudaStreamSynchronize(stream);
        mem_tracker.sample_after();
        wall_ms_vec[i] = timer.elapsed_ms();
        std::printf("[chromvar_bench] iter %d: %.1f ms\n", i, wall_ms_vec[i]);
        (void)res;
    }

    // ── Statistics ─────────────────────────────────────────────────────────────
    std::sort(wall_ms_vec.begin(), wall_ms_vec.end());
    double wall_median = wall_ms_vec[kTimed / 2];
    double wall_min    = wall_ms_vec.front();
    double wall_max    = wall_ms_vec.back();
    double peak_mb     = mem_tracker.peak_delta_mb();
    double cells_sec   = bench::throughput(n_cells, wall_median);

    std::printf("[chromvar_bench] Result: median=%.1fms min=%.1fms max=%.1fms "
                "mem=%.1fMB cells/s=%.0f\n",
                wall_median, wall_min, wall_max, peak_mb, cells_sec);

    // ── SOTA reference ─────────────────────────────────────────────────────────
    bench::RefResult ref = run_chromvar_reference(accessibility, motif_in_peak, d_gc,
                                                   n_peaks, n_cells, kNMotifs);

    // ── Log BenchRow ───────────────────────────────────────────────────────────
    bench::BenchRow row;
    row.date          = bench::today_iso();
    row.feature       = "atac/chromvar";
    row.scale         = scale_label;
    row.impl          = "singlet-gpu";
    row.wall_ms       = wall_median;
    row.mem_mb        = peak_mb;
    row.cells_per_sec = cells_sec;
    row.sota_wall     = ref.wall_ms;
    row.sota_mem      = ref.mem_mb;
    row.ratio_wall    = (ref.wall_ms > 0) ? ref.wall_ms / wall_median : 0.0;
    row.ratio_mem     = (ref.mem_mb  > 0) ? ref.mem_mb  / peak_mb    : 0.0;
    row.commit        = bench::git_short_sha();
    bench::log_row(row);

    cudaStreamDestroy(stream);
    return 0;
}
