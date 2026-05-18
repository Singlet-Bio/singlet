// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_de_wilcoxon_perf.cpp
//
// Performance benchmark: singlet::gpu::de::wilcoxon_de vs scanpy rank_genes_groups.
//
// Design reference: singlet/gpu/state/designs/53-bench-harness.md
// Kernel reference:  include/singlet/gpu/de/wilcoxon.h
//
// Staged pipeline (inline):
//   load .1pz (or tiny synthetic) → compute_knn k=15 → leiden → wilcoxon_de
//   Timing scope: wilcoxon_de() only.
//
// The Leiden stage provides cluster labels from which wilcoxon_de() picks
// differentially expressed genes (one-vs-rest, all clusters).
//
// 3 warmup + 5 timed iterations.
// SOTA reference: bench/refs/wilcoxon_ref.py (scanpy rank_genes_groups).
//
// Skip conditions:
//   - No CUDA-capable GPU
//   - Upstream kNN/Leiden fails

#include <singlet/gpu/bench/harness.h>

#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/graph/knn.h>
#include <singlet/gpu/graph/leiden.h>
#include <singlet/gpu/de/wilcoxon.h>
#include <singlet/gpu/de/types.h>
#include <singlet/gpu/core/types.h>

#include <cuda_runtime.h>
#include <cusparse.h>

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

static constexpr int      kK          = 15;
static constexpr int      kPCADim     = 50;
static constexpr float    kResolution = 1.0f;
static constexpr uint64_t kSeed       = 0xC0FFEE;
static constexpr int      kWarmup     = 3;
static constexpr int      kTimed      = 5;
static constexpr int      kSyntheticN = 10000;  // cells
static constexpr int      kSyntheticM = 2000;   // genes (small for speed)

static const char* kSamplePath =
    "/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna/"
    "GSE127/GSE127918/GSM4037629/counts.1pz";

// ─── Helpers ──────────────────────────────────────────────────────────────────

namespace {

// Build a synthetic sparse CSC count matrix (genes × cells) with Poisson-like
// structure and upload to device.  nnz density ~5%.
singlet::gpu::core::DeviceCSC make_synthetic_csc(
        int n_genes, int n_cells, uint64_t seed, cudaStream_t stream)
{
    using namespace singlet::gpu::core;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> pois(0.0f, 1.0f);

    const double density = 0.05;
    std::vector<int>   col_ptr(n_cells + 1, 0);
    std::vector<int>   row_idx;
    std::vector<float> values;
    row_idx.reserve(static_cast<size_t>(n_genes * n_cells) * (int)(density * 10));
    values.reserve(row_idx.capacity());

    for (int c = 0; c < n_cells; ++c) {
        col_ptr[c] = static_cast<int>(row_idx.size());
        for (int g = 0; g < n_genes; ++g) {
            if (pois(rng) < density) {
                row_idx.push_back(g);
                // Values in [1, 20] — Poisson-ish
                values.push_back(static_cast<float>(
                    static_cast<int>(pois(rng) * 20.0f) + 1));
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

    // Construct the DeviceCSC (SparseMatrixGPU<float> from factornet).
    DeviceCSC mat;
    mat.rows = n_genes; mat.cols = n_cells; mat.nnz = nnz;
    mat.col_ptr     = std::move(d_col_ptr);
    mat.row_indices = std::move(d_row_idx);
    mat.values      = std::move(d_values);
    return mat;
}

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

// write_csc_bin (0x43535343-magic CSC binary for the Python reference) now
// lives in the shared harness: singlet::gpu::bench::write_csc_bin.

// Write int32 .npy helper.
void write_npy_i32(const std::string& path, const int* data, int n)
{
    std::string hdr = "{'descr': '<i4', 'fortran_order': False, 'shape': (" +
                      std::to_string(n) + ",), }";
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
    f.write(reinterpret_cast<const char*>(data), n * sizeof(int));
}

// Run scanpy reference.
singlet::gpu::bench::RefResult run_wilcoxon_reference(
        const singlet::gpu::core::DeviceCSC&        mat,
        const singlet::gpu::core::DeviceMemory<int>& d_labels,
        int n_cells, int n_genes, int64_t nnz)
{
    namespace bench = singlet::gpu::bench;
    fs::create_directories("/tmp/singlet_gpu_bench");
    const std::string mat_path    = "/tmp/singlet_gpu_bench/wilcoxon_bench_mat.bin";
    const std::string labels_path = "/tmp/singlet_gpu_bench/wilcoxon_bench_labels.npy";
    const std::string out_path    = "/tmp/singlet_gpu_bench/wilcoxon_bench_out.json";

    // Download matrix CSC to host.
    std::vector<float> h_vals  (nnz);
    std::vector<int>   h_indptr(n_cells + 1);
    std::vector<int>   h_rowids(nnz);
    std::vector<int>   h_labels(n_cells);
    cudaMemcpy(h_vals.data(),   mat.values.get(),     nnz * sizeof(float),          cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indptr.data(), mat.col_ptr.get(),    (n_cells + 1) * sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rowids.data(), mat.row_indices.get(),nnz * sizeof(int),            cudaMemcpyDeviceToHost);
    cudaMemcpy(h_labels.data(), d_labels.get(),      n_cells * sizeof(int),        cudaMemcpyDeviceToHost);

    bench::write_csc_bin(mat_path, h_vals.data(), h_indptr.data(), h_rowids.data(),
                         n_genes, n_cells, nnz);
    write_npy_i32(labels_path, h_labels.data(), n_cells);

    const std::string script = std::string(BENCH_REFS_DIR) + "/wilcoxon_ref.py";
    std::string cmd = "python3 " + script +
                      " --input "  + mat_path    +
                      " --labels " + labels_path +
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
        bench::skip("bench_de_wilcoxon_perf", "no CUDA-capable GPU");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // ── Stage 1: load or synthesize matrix ─────────────────────────────────────
    int   n_cells = kSyntheticN;
    int   n_genes = kSyntheticM;
    int64_t nnz   = 0;
    const char* scale_label = "10k-synthetic";
    singlet::gpu::core::DeviceCSC mat;
    bool from_file = false;

    if (fs::exists(kSamplePath)) {
        try {
            auto loaded = singlet::gpu::io::load_pz(kSamplePath, stream);
            cudaStreamSynchronize(stream);
            n_cells     = loaded.mat.cols;
            n_genes     = loaded.mat.rows;
            nnz         = loaded.mat.nnz;
            mat         = std::move(loaded.mat);
            scale_label = "10k";
            from_file   = true;
            std::printf("[wilcoxon_bench] .1pz: %d genes × %d cells, nnz=%lld\n",
                        n_genes, n_cells, (long long)nnz);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[wilcoxon_bench] .1pz load failed: %s — synthetic\n",
                         ex.what());
        }
    }
    if (!from_file) {
        std::printf("[wilcoxon_bench] Synthesizing CSC %d × %d\n", n_genes, n_cells);
        mat = make_synthetic_csc(n_genes, n_cells, kSeed, stream);
        nnz = mat.nnz;
    }

    // ── Stage 2: embedding → kNN → Leiden (for cluster labels) ────────────────
    auto embedding = make_synthetic_embedding(n_cells, kPCADim, kSeed, stream);

    singlet::gpu::graph::KnnConfig knn_cfg;
    knn_cfg.k       = kK;
    knn_cfg.backend = singlet::gpu::graph::KnnBackend::Exact;
    knn_cfg.seed    = kSeed;

    singlet::gpu::graph::KnnResult knn_result;
    try {
        knn_result = singlet::gpu::graph::compute_knn(embedding, knn_cfg, stream);
        cudaStreamSynchronize(stream);
    } catch (const std::exception& ex) {
        bench::skip("bench_de_wilcoxon_perf",
                    std::string("kNN prerequisite failed: ") + ex.what());
        cudaStreamDestroy(stream);
        return 0;
    }

    singlet::gpu::graph::LeidenConfig leiden_cfg;
    leiden_cfg.resolution = kResolution;
    leiden_cfg.seed       = 42;
    singlet::gpu::graph::LeidenResult leiden_result;
    int n_clusters = 0;
    try {
        leiden_result = singlet::gpu::graph::leiden(knn_result, leiden_cfg, stream);
        cudaStreamSynchronize(stream);
        n_clusters = leiden_result.n_clusters;
        std::printf("[wilcoxon_bench] Leiden: %d clusters\n", n_clusters);
    } catch (const std::exception& ex) {
        bench::skip("bench_de_wilcoxon_perf",
                    std::string("Leiden prerequisite failed: ") + ex.what());
        cudaStreamDestroy(stream);
        return 0;
    }

    // ── Wilcoxon config ────────────────────────────────────────────────────────
    singlet::gpu::de::WilcoxonConfig wil_cfg;
    wil_cfg.n_bins       = 4096;
    wil_cfg.top_n        = 100;
    wil_cfg.gene_tile    = 1024;
    wil_cfg.deterministic = false;

    // ── Warmup ─────────────────────────────────────────────────────────────────
    std::printf("[wilcoxon_bench] Warming up (%d iters)...\n", kWarmup);
    for (int i = 0; i < kWarmup; ++i) {
        auto res = singlet::gpu::de::wilcoxon_de(mat, leiden_result.labels, n_clusters, wil_cfg, stream);
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
        auto res = singlet::gpu::de::wilcoxon_de(mat, leiden_result.labels, n_clusters, wil_cfg, stream);
        timer.stop(stream);
        cudaStreamSynchronize(stream);
        mem_tracker.sample_after();
        wall_ms_vec[i] = timer.elapsed_ms();
        std::printf("[wilcoxon_bench] iter %d: %.1f ms\n", i, wall_ms_vec[i]);
        (void)res;
    }

    // ── Statistics ─────────────────────────────────────────────────────────────
    std::sort(wall_ms_vec.begin(), wall_ms_vec.end());
    double wall_median = wall_ms_vec[kTimed / 2];
    double wall_min    = wall_ms_vec.front();
    double wall_max    = wall_ms_vec.back();
    double peak_mb     = mem_tracker.peak_delta_mb();
    double throughput  = bench::throughput(n_cells, wall_median);

    std::printf("[wilcoxon_bench] Result: median=%.1fms min=%.1fms max=%.1fms "
                "mem=%.1fMB cells/s=%.0f\n",
                wall_median, wall_min, wall_max, peak_mb, throughput);

    // ── SOTA reference ─────────────────────────────────────────────────────────
    bench::RefResult ref = run_wilcoxon_reference(mat, leiden_result.labels,
                                                   n_cells, n_genes, nnz);

    // ── Log BenchRow ───────────────────────────────────────────────────────────
    bench::BenchRow row;
    row.date          = bench::today_iso();
    row.feature       = "de/wilcoxon";
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
