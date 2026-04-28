// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_spatial_flash_deconv_perf.cpp
//
// Performance benchmark: singlet_gpu::spatial::flash_deconv vs cell2location Python.
//
// Design reference: singlet-gpu/state/designs/33-flash-deconv.md
// Kernel reference:  include/singlet-gpu/spatial/flash_deconv.h
//
// Staged pipeline:
//   load .1pz counts (spatial spots × genes) + synthesize reference matrix B
//   → flash_deconv(Y, B, cfg, stream)
//
// Timing scope: flash_deconv() only (excludes matrix load).
//   Reference aggregation, leverage-score sketching, ADMM, bootstrap are all
//   inside flash_deconv() and are included in the timing.
//
// 3 warmup + 5 timed iterations.
// SOTA reference: bench/refs/flash_deconv_ref.py (cell2location Python via subprocess).
//   If cell2location is absent the driver logs sota_wall=TBD and continues.
//
// Skip conditions:
//   - No CUDA-capable GPU → exit 0
//   - Sample not found → use synthetic 500 spots × 200 genes, scale="tiny"

#include <singlet_gpu/bench/harness.h>

#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/spatial/flash_deconv.h>
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

static constexpr uint64_t kSeed       = 0xC0FFEE;
static constexpr int      kWarmup     = 3;
static constexpr int      kTimed      = 5;
static constexpr int      kNTypes     = 8;   // synthetic cell-type count
static constexpr int      kSyntheticN = 500; // spots
static constexpr int      kSyntheticM = 200; // genes

static const char* kSamplePath =
    "/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/"
    "GSE127/GSE127918/GSM4037629/counts.1pz";

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

// Synthesize a reference mean expression matrix B (n_types × n_genes) on device.
// Each type has a distinct gene signature with Gaussian noise.
singlet_gpu::core::DeviceMemory<float> make_reference_B(
        int n_types, int n_genes, uint64_t seed, cudaStream_t stream)
{
    std::mt19937_64 rng(seed ^ 0xABCDEFull);
    std::normal_distribution<float> norm(0.0f, 0.1f);

    std::vector<float> host(static_cast<size_t>(n_types) * n_genes, 0.0f);
    // Each type gets ~200 marker genes with higher mean expression.
    int markers_per_type = std::max(1, n_genes / std::max(1, n_types));
    for (int t = 0; t < n_types; ++t) {
        int base = (t * markers_per_type) % n_genes;
        for (int j = 0; j < markers_per_type; ++j) {
            int g = (base + j) % n_genes;
            host[static_cast<size_t>(t) * n_genes + g] =
                std::max(0.0f, 1.0f + norm(rng));
        }
        // Low background across all genes.
        for (int g = 0; g < n_genes; ++g) {
            host[static_cast<size_t>(t) * n_genes + g] +=
                std::max(0.0f, 0.05f + norm(rng) * 0.01f);
        }
    }

    singlet_gpu::core::DeviceMemory<float> dev(static_cast<size_t>(n_types) * n_genes);
    cudaMemcpyAsync(dev.get(), host.data(),
                    static_cast<size_t>(n_types) * n_genes * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return dev;
}

// Write float32 dense matrix (rows × cols, row-major) as .npy.
void write_npy_f32(const std::string& path, const float* data, int rows, int cols)
{
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
    f.write(reinterpret_cast<const char*>(data),
            static_cast<size_t>(rows) * cols * sizeof(float));
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

// Invoke bench/refs/flash_deconv_ref.py and return (wall_ms, mem_mb).
singlet_gpu::bench::RefResult run_flash_deconv_reference(
        const singlet_gpu::core::DeviceCSC&         mat,
        const singlet_gpu::core::DeviceMemory<float>& d_B,
        int n_spots, int n_genes, int n_types, int64_t nnz)
{
    namespace bench = singlet_gpu::bench;
    fs::create_directories("/tmp/singlet_gpu_bench");
    const std::string mat_path = "/tmp/singlet_gpu_bench/flash_deconv_bench_mat.bin";
    const std::string ref_path = "/tmp/singlet_gpu_bench/flash_deconv_bench_ref.npy";
    const std::string out_path = "/tmp/singlet_gpu_bench/flash_deconv_bench_out.json";

    // Download CSC + reference matrix to host.
    std::vector<float> h_vals  (nnz);
    std::vector<int>   h_indptr(n_spots + 1);
    std::vector<int>   h_rowids(nnz);
    std::vector<float> h_B     (static_cast<size_t>(n_types) * n_genes);
    cudaMemcpy(h_vals.data(),   mat.values.get(),     nnz * sizeof(float),         cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indptr.data(), mat.col_ptr.get(),    (n_spots + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rowids.data(), mat.row_indices.get(),nnz * sizeof(int),           cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B.data(),      d_B.get(),
               static_cast<size_t>(n_types) * n_genes * sizeof(float),            cudaMemcpyDeviceToHost);

    write_csc_bin(mat_path, h_vals.data(), h_indptr.data(), h_rowids.data(),
                  n_genes, n_spots, nnz);
    write_npy_f32(ref_path, h_B.data(), n_types, n_genes);

    const std::string script = std::string(BENCH_REFS_DIR) + "/flash_deconv_ref.py";
    std::string cmd = "python3 " + script +
                      " --input "        + mat_path +
                      " --reference "    + ref_path +
                      " --timing-json "  + out_path +
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
        bench::skip("bench_spatial_flash_deconv_perf", "no CUDA-capable GPU");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // ── Stage 1: load or synthesize count matrix ───────────────────────────────
    int    n_spots     = kSyntheticN;
    int    n_genes     = kSyntheticM;
    int    n_types     = kNTypes;
    int64_t nnz        = 0;
    const char* scale_label = "tiny";
    bool from_file = false;

    singlet_gpu::core::DeviceCSC mat;

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
            std::printf("[flash_deconv_bench] .1pz: %d genes × %d spots, nnz=%lld\n",
                        n_genes, n_spots, (long long)nnz);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[flash_deconv_bench] .1pz load failed: %s — synthetic\n",
                         ex.what());
        }
    }

    if (!from_file) {
        std::printf("[flash_deconv_bench] Synthesizing CSC %d × %d (tiny)\n",
                    n_genes, n_spots);
        mat = make_synthetic_csc(n_genes, n_spots, kSeed, stream);
        nnz = mat.nnz;
    }

    // ── Stage 2: synthesize reference matrix B (n_types × n_genes) ─────────────
    std::printf("[flash_deconv_bench] Synthesizing reference B: %d types × %d genes\n",
                n_types, n_genes);
    auto d_B = make_reference_B(n_types, n_genes, kSeed, stream);

    // ── FlashDeconv config ─────────────────────────────────────────────────────
    singlet_gpu::spatial::FlashDeconvConfig cfg;
    cfg.sketch_size    = std::min(200, n_genes);   // smaller sketch for tiny synth
    cfg.max_admm_iter  = 20;   // reduced for bench timing (production: 50)
    cfg.admm_tol       = 1e-4f;
    cfg.l1_weight      = 0.01f;
    cfg.spatial_lambda = 0.0f;
    cfg.n_bootstrap    = 3;    // reduced (production: 20) to keep bench time finite
    cfg.seed           = kSeed;
    cfg.spot_batch     = std::min(n_spots, 50000);

    // ── Warmup (3 discarded iterations) ───────────────────────────────────────
    std::printf("[flash_deconv_bench] Warming up (%d iters)...\n", kWarmup);
    for (int i = 0; i < kWarmup; ++i) {
        auto res = singlet_gpu::spatial::flash_deconv(mat, d_B, n_types, cfg, stream);
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
        auto res = singlet_gpu::spatial::flash_deconv(mat, d_B, n_types, cfg, stream);
        timer.stop(stream);
        cudaStreamSynchronize(stream);
        mem_tracker.sample_after();
        wall_ms_vec[i] = timer.elapsed_ms();
        std::printf("[flash_deconv_bench] iter %d: %.1f ms  admm_iters=%d\n",
                    i, wall_ms_vec[i], res.n_admm_iters_used);
        (void)res;
    }

    // ── Statistics ─────────────────────────────────────────────────────────────
    std::sort(wall_ms_vec.begin(), wall_ms_vec.end());
    double wall_median = wall_ms_vec[kTimed / 2];
    double wall_min    = wall_ms_vec.front();
    double wall_max    = wall_ms_vec.back();
    double peak_mb     = mem_tracker.peak_delta_mb();
    double tput        = bench::throughput(n_spots, wall_median);

    std::printf("[flash_deconv_bench] Result: median=%.1fms min=%.1fms max=%.1fms "
                "mem=%.1fMB spots/s=%.0f\n",
                wall_median, wall_min, wall_max, peak_mb, tput);

    // ── SOTA reference (cell2location Python) ──────────────────────────────────
    bench::RefResult ref = run_flash_deconv_reference(mat, d_B, n_spots, n_genes, n_types, nnz);

    // ── Log BenchRow ───────────────────────────────────────────────────────────
    bench::BenchRow row;
    row.date          = bench::today_iso();
    row.feature       = "spatial/flash_deconv";
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
