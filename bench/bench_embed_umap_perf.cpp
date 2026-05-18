// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_embed_umap_perf.cpp
//
// Performance benchmark: singlet::gpu::embed::umap vs cuML UMAP.
//
// Design reference: singlet/gpu/state/designs/53-bench-harness.md
// Kernel reference:  include/singlet/gpu/embed/umap.h
//
// Staged pipeline (inline):
//   synthetic embedding → compute_knn k=15 → run_umap
//
// Timing scope: run_umap() only (excludes load + kNN).
// 3 warmup + 5 timed iterations.
// SOTA reference: bench/refs/umap_ref.py (cuML UMAP via precomputed kNN).
//
// Skip conditions:
//   - No CUDA-capable GPU
//   - kNN prerequisite fails

#include <singlet/gpu/bench/harness.h>

#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/graph/knn.h>
#include <singlet/gpu/embed/umap.h>
#include <singlet/gpu/core/types.h>

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

static constexpr int      kK        = 15;
static constexpr int      kPCADim   = 50;
static constexpr uint64_t kSeed     = 0xC0FFEE;
static constexpr int      kWarmup   = 3;
static constexpr int      kTimed    = 5;

static const char* kSamplePath =
    "/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna/"
    "GSE127/GSE127918/GSM4037629/counts.1pz";

// ─── Helpers ──────────────────────────────────────────────────────────────────

namespace {

singlet::gpu::core::DeviceDense make_synthetic_embedding(
        int n, int d, uint64_t seed, cudaStream_t stream)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> host(static_cast<size_t>(n) * d);
    for (auto& v : host) v = dist(rng);
    singlet::gpu::core::DeviceMemory<float> dev(static_cast<size_t>(n) * d);
    cudaMemcpyAsync(dev.get(), host.data(), host.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    singlet::gpu::core::DeviceDense mat;
    mat.rows = n; mat.cols = d;
    mat.data = std::move(dev);
    return mat;
}

// Write .npy helper (float32 or int32).
static void write_npy(const std::string& path, const void* data, size_t elem_sz,
                      const std::string& dtype_str, int rows, int cols)
{
    std::string hdr = "{'descr': '" + dtype_str + "', 'fortran_order': False, "
                      "'shape': (" + std::to_string(rows) + ", " + std::to_string(cols) + "), }";
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
    f.write(reinterpret_cast<const char*>(data), rows * cols * elem_sz);
}

// Run Python reference: bench/refs/umap_ref.py.
singlet::gpu::bench::RefResult run_umap_reference(
        const singlet::gpu::graph::KnnResult& knn, int n, int k)
{
    namespace bench = singlet::gpu::bench;
    fs::create_directories("/tmp/singlet_gpu_bench");
    const std::string idx_path  = "/tmp/singlet_gpu_bench/umap_bench_idx.npy";
    const std::string dist_path = "/tmp/singlet_gpu_bench/umap_bench_dist.npy";
    const std::string out_path  = "/tmp/singlet_gpu_bench/umap_bench_out.json";

    std::vector<int>   h_idx (static_cast<size_t>(n) * k);
    std::vector<float> h_dist(static_cast<size_t>(n) * k);
    cudaMemcpy(h_idx.data(),  knn.neighbors.get(), h_idx.size()  * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dist.data(), knn.distances.get(), h_dist.size() * sizeof(float), cudaMemcpyDeviceToHost);
    write_npy(idx_path,  h_idx.data(),  sizeof(int),   "<i4", n, k);
    write_npy(dist_path, h_dist.data(), sizeof(float), "<f4", n, k);

    const std::string script = std::string(BENCH_REFS_DIR) + "/umap_ref.py";
    std::string cmd = "python3 " + script +
                      " --indices "     + idx_path  +
                      " --distances "   + dist_path +
                      " --n-neighbors " + std::to_string(k) +
                      " --seed 42"
                      " --timing-json " + out_path +
                      " 2>/dev/null";
    return bench::run_python_reference(cmd, out_path);
}

}  // anonymous namespace

// ─── Main ─────────────────────────────────────────────────────────────────────

int main()
{
    namespace bench = singlet::gpu::bench;

    // ── GPU check ──────────────────────────────────────────────────────────────
    int n_dev = 0;
    cudaGetDeviceCount(&n_dev);
    if (n_dev == 0) {
        bench::skip("bench_embed_umap_perf", "no CUDA-capable GPU");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // ── Stage 1: determine cell count ──────────────────────────────────────────
    int emb_n = 10000;
    const char* scale_label = "10k-synthetic";

    if (fs::exists(kSamplePath)) {
        try {
            auto mat = singlet::gpu::io::load_pz(kSamplePath, stream);
            cudaStreamSynchronize(stream);
            emb_n       = mat.mat.cols;
            scale_label = "10k";
            std::printf("[umap_bench] .1pz: %d cells\n", emb_n);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[umap_bench] .1pz load failed: %s\n", ex.what());
        }
    } else {
        std::printf("[umap_bench] Sample not found — synthetic n=%d\n", emb_n);
    }

    auto embedding = make_synthetic_embedding(emb_n, kPCADim, kSeed, stream);

    // ── Stage 2: kNN (prerequisite) ────────────────────────────────────────────
    singlet::gpu::graph::KnnConfig knn_cfg;
    knn_cfg.k       = kK;
    knn_cfg.backend = singlet::gpu::graph::KnnBackend::Exact;
    knn_cfg.seed    = kSeed;

    std::printf("[umap_bench] Building kNN graph k=%d...\n", kK);
    singlet::gpu::graph::KnnResult knn_result;
    try {
        knn_result = singlet::gpu::graph::compute_knn(embedding, knn_cfg, stream);
        cudaStreamSynchronize(stream);
    } catch (const std::exception& ex) {
        bench::skip("bench_embed_umap_perf",
                    std::string("kNN prerequisite failed: ") + ex.what());
        cudaStreamDestroy(stream);
        return 0;
    }
    std::printf("[umap_bench] kNN done.\n");

    // ── UMAP config ────────────────────────────────────────────────────────────
    singlet::gpu::embed::UmapConfig umap_cfg;
    umap_cfg.n_components = 2;
    umap_cfg.n_epochs     = 0;   // cuML auto-select
    umap_cfg.min_dist     = 0.5f;
    umap_cfg.spread       = 1.0f;
    umap_cfg.init         = singlet::gpu::embed::UmapInit::Random;
    umap_cfg.seed         = kSeed;

    // ── Warmup ─────────────────────────────────────────────────────────────────
    std::printf("[umap_bench] Warming up (%d iters)...\n", kWarmup);
    for (int i = 0; i < kWarmup; ++i) {
        auto res = singlet::gpu::embed::umap(knn_result, umap_cfg, stream);
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
        auto res = singlet::gpu::embed::umap(knn_result, umap_cfg, stream);
        timer.stop(stream);
        cudaStreamSynchronize(stream);
        mem_tracker.sample_after();
        wall_ms_vec[i] = timer.elapsed_ms();
        std::printf("[umap_bench] iter %d: %.1f ms\n", i, wall_ms_vec[i]);
        (void)res;
    }

    // ── Statistics ─────────────────────────────────────────────────────────────
    std::sort(wall_ms_vec.begin(), wall_ms_vec.end());
    double wall_median = wall_ms_vec[kTimed / 2];
    double wall_min    = wall_ms_vec.front();
    double wall_max    = wall_ms_vec.back();
    double peak_mb     = mem_tracker.peak_delta_mb();
    double throughput  = bench::throughput(emb_n, wall_median);

    std::printf("[umap_bench] Result: median=%.1fms min=%.1fms max=%.1fms "
                "mem=%.1fMB cells/s=%.0f\n",
                wall_median, wall_min, wall_max, peak_mb, throughput);

    // ── SOTA reference ─────────────────────────────────────────────────────────
    bench::RefResult ref = run_umap_reference(knn_result, emb_n, kK);

    // ── Log BenchRow ───────────────────────────────────────────────────────────
    bench::BenchRow row;
    row.date          = bench::today_iso();
    row.feature       = "embed/umap";
    row.scale         = scale_label;
    row.impl          = "singlet-gpu";
    row.wall_ms       = wall_median;
    row.mem_mb        = peak_mb;
    row.cells_per_sec = throughput;
    row.sota_wall     = ref.wall_ms;
    row.sota_mem      = ref.mem_mb;
    row.ratio_wall    = (ref.wall_ms > 0) ? ref.wall_ms / wall_median : 0.0;
    row.ratio_mem     = (ref.mem_mb  > 0) ? ref.mem_mb  / peak_mb    : 0.0;
    row.commit        = bench::git_short_sha();
    bench::log_row(row);

    cudaStreamDestroy(stream);
    return 0;
}
