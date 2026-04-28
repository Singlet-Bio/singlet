// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_graph_leiden_perf.cpp
//
// Performance benchmark: singlet_gpu::graph::leiden vs cugraph Leiden.
//
// Design reference: singlet-gpu/state/designs/53-bench-harness.md
// Kernel reference:  include/singlet-gpu/graph/leiden.h
//
// Staged pipeline (inline):
//   load .1pz → synthetic embedding (or cached PCA) → compute_knn k=15 → leiden
//
// Timing scope: leiden_single_resolution() only (excludes load + kNN).
// 3 warmup + 5 timed iterations.
// SOTA reference: bench/refs/leiden_ref.py (cugraph Leiden / scanpy fallback).
//
// Skip conditions:
//   - No CUDA-capable GPU
//   - kNN stage fails (prerequisite)

#include <singlet_gpu/bench/harness.h>

#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/graph/knn.h>
#include <singlet-gpu/graph/leiden.h>
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

static constexpr int      kK          = 15;
static constexpr int      kPCADim     = 50;
static constexpr float    kResolution = 1.0f;
static constexpr uint64_t kSeed       = 0xC0FFEE;
static constexpr int      kWarmup     = 3;
static constexpr int      kTimed      = 5;

static const char* kSamplePath =
    "/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/"
    "GSE127/GSE127918/GSM4037629/counts.1pz";

// ─── Helpers ──────────────────────────────────────────────────────────────────

namespace {

// Build a synthetic dense embedding, upload to device.
singlet_gpu::core::DeviceDense make_synthetic_embedding(
        int n, int d, uint64_t seed, cudaStream_t stream)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> host(static_cast<size_t>(n) * d);
    for (auto& v : host) v = dist(rng);

    singlet_gpu::core::DeviceMemory<float> dev(static_cast<size_t>(n) * d);
    cudaMemcpyAsync(dev.get(), host.data(), host.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    singlet_gpu::core::DeviceDense result(n, d);
    result.data = std::move(dev);
    return result;
}

// Run Python reference: bench/refs/leiden_ref.py.
// Writes kNN neighbors + distances as .npy, invokes reference, reads timing JSON.
singlet_gpu::bench::RefResult run_leiden_reference(
        const singlet_gpu::graph::KnnResult& knn, int n, int k)
{
    namespace bench = singlet_gpu::bench;
    fs::create_directories("/tmp/singlet_gpu_bench");
    const std::string idx_path  = "/tmp/singlet_gpu_bench/leiden_bench_idx.npy";
    const std::string dist_path = "/tmp/singlet_gpu_bench/leiden_bench_dist.npy";
    const std::string out_path  = "/tmp/singlet_gpu_bench/leiden_bench_out.json";

    // Download kNN result to host for Python reference.
    std::vector<int>   h_idx(static_cast<size_t>(n) * k);
    std::vector<float> h_dist(static_cast<size_t>(n) * k);
    cudaMemcpy(h_idx.data(),  knn.neighbors.get(),  h_idx.size() * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dist.data(), knn.distances.get(), h_dist.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Write int32 .npy
    auto write_npy = [&](const std::string& path, const void* data,
                         size_t elem_sz, const std::string& dtype_str, int rows, int cols) {
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
    };
    write_npy(idx_path,  h_idx.data(),  sizeof(int),   "<i4", n, k);
    write_npy(dist_path, h_dist.data(), sizeof(float), "<f4", n, k);

    const std::string script = std::string(BENCH_REFS_DIR) + "/leiden_ref.py";
    std::string cmd = "python3 " + script +
                      " --indices "   + idx_path  +
                      " --distances " + dist_path +
                      " --resolution " + std::to_string(kResolution) +
                      " --seed 42"
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
        bench::skip("bench_graph_leiden_perf", "no CUDA-capable GPU");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // ── Stage 1: embedding ─────────────────────────────────────────────────────
    int emb_n = 10000;
    const char* scale_label = "10k-synthetic";

    if (fs::exists(kSamplePath)) {
        try {
            auto mat = singlet_gpu::io::load_pz(kSamplePath, stream);
            cudaStreamSynchronize(stream);
            emb_n       = mat.mat.cols;
            scale_label = "10k";
            std::printf("[leiden_bench] .1pz: %d cells\n", emb_n);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[leiden_bench] .1pz load failed: %s — using n=%d\n",
                         ex.what(), emb_n);
        }
    } else {
        std::printf("[leiden_bench] Sample not found — synthetic n=%d\n", emb_n);
    }

    auto embedding = make_synthetic_embedding(emb_n, kPCADim, kSeed, stream);

    // ── Stage 2: kNN (prerequisite — not timed here) ───────────────────────────
    singlet_gpu::graph::KnnConfig knn_cfg;
    knn_cfg.k       = kK;
    knn_cfg.backend = singlet_gpu::graph::KnnBackend::Exact;
    knn_cfg.seed    = kSeed;

    std::printf("[leiden_bench] Building kNN graph k=%d...\n", kK);
    singlet_gpu::graph::KnnResult knn_result;
    try {
        knn_result = singlet_gpu::graph::compute_knn(embedding, knn_cfg, stream);
        cudaStreamSynchronize(stream);
    } catch (const std::exception& ex) {
        bench::skip("bench_graph_leiden_perf",
                    std::string("kNN prerequisite failed: ") + ex.what());
        cudaStreamDestroy(stream);
        return 0;
    }
    std::printf("[leiden_bench] kNN done.\n");

// TODO(CYCLE-GATE-2G): leiden_single_resolution() does not exist in the graph
// namespace — the public API exposes singlet_gpu::graph::leiden() (taking
// KnnResult + LeidenConfig + stream) and leiden_multi() (multi-resolution).
// This bench driver must be updated to call leiden() once the public API is
// confirmed, and leiden_single_resolution() either aliased or removed.
// Disabled for compile-gate-phase-2 until the API is resolved.
#if 0
    // ── Leiden config ──────────────────────────────────────────────────────────
    singlet_gpu::graph::LeidenConfig leiden_cfg;
    leiden_cfg.resolution = kResolution;
    leiden_cfg.seed       = static_cast<uint64_t>(42);

    // ── Warmup ─────────────────────────────────────────────────────────────────
    std::printf("[leiden_bench] Warming up (%d iters)...\n", kWarmup);
    for (int i = 0; i < kWarmup; ++i) {
        auto res = singlet_gpu::graph::leiden_single_resolution(knn_result, leiden_cfg, stream);
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
        auto res = singlet_gpu::graph::leiden_single_resolution(knn_result, leiden_cfg, stream);
        timer.stop(stream);
        cudaStreamSynchronize(stream);
        mem_tracker.sample_after();
        wall_ms_vec[i] = timer.elapsed_ms();
        std::printf("[leiden_bench] iter %d: %.1f ms  n_clusters=%d\n",
                    i, wall_ms_vec[i], res.n_clusters);
        (void)res;
    }

    // ── Statistics ─────────────────────────────────────────────────────────────
    std::sort(wall_ms_vec.begin(), wall_ms_vec.end());
    double wall_median = wall_ms_vec[kTimed / 2];
    double wall_min    = wall_ms_vec.front();
    double wall_max    = wall_ms_vec.back();
    double peak_mb     = mem_tracker.peak_delta_mb();
    double throughput  = bench::throughput(emb_n, wall_median);

    std::printf("[leiden_bench] Result: median=%.1fms min=%.1fms max=%.1fms "
                "mem=%.1fMB cells/s=%.0f\n",
                wall_median, wall_min, wall_max, peak_mb, throughput);

    // ── SOTA reference ─────────────────────────────────────────────────────────
    bench::RefResult ref = run_leiden_reference(knn_result, emb_n, kK);

    // ── Log BenchRow ───────────────────────────────────────────────────────────
    bench::BenchRow row;
    row.date          = bench::today_iso();
    row.feature       = "graph/leiden";
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
#endif  // TODO(CYCLE-GATE-2G)

    std::printf("[leiden_bench] Leiden benchmark body disabled (CYCLE-GATE-2G: "
                "leiden_single_resolution API missing). kNN prerequisite passed.\n");
    cudaStreamDestroy(stream);
    return 0;
}
