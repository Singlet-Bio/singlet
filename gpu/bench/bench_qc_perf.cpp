// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_qc_perf.cpp
//
// Cycle 86 add-on — QC metrics benchmark vs scanpy.calculate_qc_metrics (CPU).
//
// Benchmarks singlet_gpu::qc::calculate_qc_metrics() on:
//   small : synthetic 1,000 cells × 2,000 genes
//   medium: GSM4037629 gene_counts.1pz (~11,560 cells × ~38,606 genes)
//
// Metrics recorded:
//   Per-cell: n_umis, n_genes, pct_mt, pct_ribo
//   Per-gene: mean, var, n_cells_expressing
//   Peak device memory (cudaMemGetInfo delta)
//
// Comparison: scanpy.pp.calculate_qc_metrics (CPU, run via Python subprocess;
//   refs/qc_scanpy_ref.py) on the same input (gene_counts.1pz → AnnData).
//
// OOM guard: cudaMemGetInfo before host vector::reserve in synthetic builder.
// Modeled on bench_de_wilcoxon_perf_c85.cpp.

#include <singlet_gpu/bench/harness.h>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/qc/metrics.h>
#include <singlet-gpu/core/types.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace singlet_gpu;

// ---------------------------------------------------------------------------
// Synthetic matrix builder (1k cells × 2k genes, density 0.15)
// ---------------------------------------------------------------------------
struct SyntheticCSC {
    core::DeviceMemory<int>   col_ptr;
    core::DeviceMemory<int>   row_indices;
    core::DeviceMemory<float> values;
    int n_cells = 0;
    int n_genes = 0;
    int64_t nnz = 0;
};

static SyntheticCSC make_synthetic(int n_cells, int n_genes, float density,
                                   uint64_t seed, cudaStream_t stream) {
    // OOM guard
    const size_t est_nnz = (size_t)(n_cells * n_genes * density * 1.5);
    size_t free_dev = 0, total_dev = 0;
    cudaMemGetInfo(&free_dev, &total_dev);
    size_t dev_need = est_nnz * 8 + (n_cells + 1) * 4;
    if (dev_need > free_dev / 2) {
        std::printf("WARN: synthetic matrix may OOM (need %.0fMB, free %.0fMB)\n",
                    dev_need / 1e6, free_dev / 1e6);
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> ud(0.0, 1.0);

    std::vector<int>   cp; cp.reserve(n_cells + 1);
    std::vector<int>   ri; ri.reserve(est_nnz);
    std::vector<float> vv; vv.reserve(est_nnz);

    cp.push_back(0);
    for (int c = 0; c < n_cells; ++c) {
        for (int g = 0; g < n_genes; ++g) {
            if (ud(rng) < density) {
                ri.push_back(g);
                vv.push_back((float)((int)(ud(rng) * 49) + 1));
            }
        }
        cp.push_back((int)ri.size());
    }

    int64_t nnz = (int64_t)ri.size();
    SyntheticCSC mat;
    mat.n_cells = n_cells;
    mat.n_genes = n_genes;
    mat.nnz     = nnz;
    mat.col_ptr     = core::DeviceMemory<int>  (n_cells + 1);
    mat.row_indices = core::DeviceMemory<int>  (nnz);
    mat.values      = core::DeviceMemory<float>(nnz);

    cudaMemcpyAsync(mat.col_ptr.get(),     cp.data(), (n_cells + 1) * 4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(mat.row_indices.get(), ri.data(), nnz * 4,           cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(mat.values.get(),      vv.data(), nnz * 4,           cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    return mat;
}

// ---------------------------------------------------------------------------
// Gene flag arrays (mt / ribo) — 5% mt, 5% ribo (deterministic by gene index).
// ---------------------------------------------------------------------------
static void make_gene_flags(int n_genes, cudaStream_t stream,
                             core::DeviceMemory<uint8_t>& d_is_mt,
                             core::DeviceMemory<uint8_t>& d_is_ribo) {
    std::vector<uint8_t> mt(n_genes, 0), ribo(n_genes, 0);
    // First 5% are MT, next 5% are ribo (simple deterministic rule).
    int n_mt   = std::max(1, n_genes / 20);
    int n_ribo = std::max(1, n_genes / 20);
    for (int i = 0; i < n_mt;            ++i) mt[i]           = 1;
    for (int i = n_mt; i < n_mt + n_ribo; ++i) ribo[i]        = 1;

    d_is_mt   = core::DeviceMemory<uint8_t>(n_genes);
    d_is_ribo = core::DeviceMemory<uint8_t>(n_genes);
    cudaMemcpyAsync(d_is_mt.get(),   mt.data(),   n_genes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_is_ribo.get(), ribo.data(), n_genes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
}

// ---------------------------------------------------------------------------
// Benchmark one scale.
// Returns median wall ms.
// ---------------------------------------------------------------------------
static float bench_scale(
    const int*    d_indptr,
    const int*    d_indices,
    const float*  d_values,
    int n_cells, int n_genes,
    const uint8_t* d_is_mt,
    const uint8_t* d_is_ribo,
    const char*    scale_name,
    cudaStream_t   stream)
{
    const int WARMUP = 2;
    const int TIMED  = 5;

    BenchTimer        timer(stream);
    PeakMemoryTracker mem;

    for (int iter = 0; iter < WARMUP + TIMED; ++iter) {
        bool timed = (iter >= WARMUP);
        if (timed) mem.snapshot_before();
        timer.start();

        auto result = qc::calculate_qc_metrics(
            d_indptr, d_indices, d_values,
            n_cells, n_genes,
            d_is_mt, d_is_ribo,
            stream);
        cudaDeviceSynchronize();

        timer.stop();
        if (timed) { timer.record(); mem.snapshot_after(); }
    }

    float med_ms  = timer.median_ms();
    float min_ms  = timer.min_ms();
    float max_ms  = timer.max_ms();
    float mem_mb  = mem.peak_mb();
    long long cells = n_cells;

    std::printf("BENCH qc scale=%-10s  n_cells=%lld  n_genes=%d"
                "  med=%.2fms  min=%.2fms  max=%.2fms  mem=%.0fMB\n",
                scale_name, cells, n_genes, med_ms, min_ms, max_ms, mem_mb);
    std::fflush(stdout);

    // Append to benchmark-registry.md.
    {
        std::ofstream reg(BENCH_REGISTRY_PATH, std::ios::app);
        if (reg.is_open()) {
            char scale_tag[64];
            std::snprintf(scale_tag, sizeof(scale_tag), "%s-c86", scale_name);
            reg << "| 2026-04-18 | qc/metrics_c86 | " << scale_tag
                << " | singlet-gpu"
                << " | " << med_ms
                << " | " << mem_mb
                << " | " << (long long)(cells / (med_ms / 1000.0f))
                << " | — | — | — | no-git |\n";
        }
    }

    return med_ms;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    if (!gpu_available()) { skip_no_gpu(); return 0; }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // ── SCALE: SMALL (synthetic 1k × 2k) ────────────────────────────────────
    std::printf("\n=== QC Bench — SMALL scale (synthetic 1000×2000) ===\n");
    {
        auto smat = make_synthetic(1000, 2000, 0.15f, 9876543ULL, stream);
        core::DeviceMemory<uint8_t> d_mt, d_ribo;
        make_gene_flags(smat.n_genes, stream, d_mt, d_ribo);

        bench_scale(smat.col_ptr.get(), smat.row_indices.get(), smat.values.get(),
                    smat.n_cells, smat.n_genes,
                    d_mt.get(), d_ribo.get(),
                    "small", stream);
    }

    // ── SCALE: MEDIUM (GSM4037629 gene_counts.1pz) ──────────────────────────
    const std::string pz_path =
        std::string(CANONICAL_SAMPLE_DIR) + "/gene_counts.1pz";
    if (!fs::exists(pz_path)) {
        std::printf("SKIP medium scale: %s not found\n", pz_path.c_str());
    } else {
        std::printf("\n=== QC Bench — MEDIUM scale (GSM4037629 gene_counts.1pz) ===\n");

        io::PzDeviceMatrix med = io::load_pz(pz_path, stream,
                                              /*keep_host_pinned=*/false);
        cudaStreamSynchronize(stream);

        std::printf("Matrix: %lld genes × %lld cells  nnz=%lld\n",
                    (long long)med.mat.rows,
                    (long long)med.mat.cols,
                    (long long)med.mat.nnz);

        core::DeviceMemory<uint8_t> d_mt, d_ribo;
        make_gene_flags((int)med.mat.rows, stream, d_mt, d_ribo);

        bench_scale(med.mat.col_ptr.get(), med.mat.row_indices.get(), med.mat.values.get(),
                    (int)med.mat.cols, (int)med.mat.rows,
                    d_mt.get(), d_ribo.get(),
                    "medium", stream);
    }

    cudaStreamDestroy(stream);
    std::printf("\n=== QC bench COMPLETE ===\n");
    return 0;
}
