// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_perturbation_cpa_perf.cpp
//
// Performance benchmark: singlet_gpu::perturbation::train_perturb_graph (CPA)
// vs CPA Python (scvi-tools / original PyTorch CPA implementation).
//
// Design reference: singlet-gpu/state/designs/32-perturbation-cpa.md
// Kernel reference:  include/singlet-gpu/perturbation/perturb_graph.h
//
// Staged pipeline:
//   1. Load counts.1pz (or synthetic 10k × 5k matrix).
//   2. Synthesize perturbation labels + dose (uniform random, n_perts=20).
//   3. Time train_perturb_graph() fit phase (all epochs).
//
// Timing scope: train_perturb_graph() only (excludes data load + label gen).
// 3 warmup + 5 timed iterations.
// SOTA reference: bench/refs/cpa_ref.py — CPA Python (PyTorch) fit timing.
//
// Skip conditions:
//   - No CUDA-capable GPU
//   - DeviceCSC construction fails

#include <singlet_gpu/bench/harness.h>

#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/perturbation/perturb_graph.h>
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

static constexpr int      kNPerts     = 20;
static constexpr int      kNEpochs    = 30;   // reduced epochs for bench timing
static constexpr uint64_t kSeed       = 0xDEAD;
static constexpr int      kWarmup     = 3;
static constexpr int      kTimed      = 5;
static constexpr int      kSyntheticN = 10000;  // cells
static constexpr int      kSyntheticM = 5000;   // genes

static const char* kSamplePath =
    "/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/"
    "GSE127/GSE127918/GSM4037629/counts.1pz";

// ─── Helpers ──────────────────────────────────────────────────────────────────

namespace {

// Build a synthetic sparse CSC count matrix (genes × cells) with ~5% density.
singlet_gpu::core::DeviceCSC make_synthetic_csc(
        int n_genes, int n_cells, uint64_t seed, cudaStream_t stream)
{
    using namespace singlet_gpu::core;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> udist(0.0f, 1.0f);
    const double density = 0.05;

    std::vector<int>   col_ptr(n_cells + 1, 0);
    std::vector<int>   row_idx;
    std::vector<float> values;
    row_idx.reserve(static_cast<size_t>(n_genes * n_cells) * 6);
    values.reserve(row_idx.capacity());

    for (int c = 0; c < n_cells; ++c) {
        col_ptr[c] = static_cast<int>(row_idx.size());
        for (int g = 0; g < n_genes; ++g) {
            if (udist(rng) < density) {
                row_idx.push_back(g);
                values.push_back(static_cast<float>(
                    static_cast<int>(udist(rng) * 20.0f) + 1));
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
    return DeviceCSC(n_genes, n_cells, nnz,
                     std::move(d_col_ptr), std::move(d_row_idx), std::move(d_values));
}

// Write a float32 .npy for the reference script.
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

void write_npy_i32(const std::string& path, const int* data, int n) {
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

// Write CSC binary (same format used by other bench drivers).
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

singlet_gpu::bench::RefResult run_cpa_reference(
        const singlet_gpu::core::DeviceCSC& mat,
        const std::vector<int>&   pert_labels,
        const std::vector<float>& dose,
        int n_genes, int n_cells, int64_t nnz, int n_perts, int n_epochs)
{
    namespace bench = singlet_gpu::bench;
    fs::create_directories("/tmp/singlet_gpu_bench");
    const std::string mat_path    = "/tmp/singlet_gpu_bench/cpa_bench_mat.bin";
    const std::string labels_path = "/tmp/singlet_gpu_bench/cpa_bench_labels.npy";
    const std::string dose_path   = "/tmp/singlet_gpu_bench/cpa_bench_dose.npy";
    const std::string out_path    = "/tmp/singlet_gpu_bench/cpa_bench_out.json";

    // Download matrix CSC to host.
    std::vector<float> h_vals  (nnz);
    std::vector<int>   h_indptr(n_cells + 1);
    std::vector<int>   h_rowids(nnz);
    cudaMemcpy(h_vals.data(),   mat.values.get(),     nnz * sizeof(float),         cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indptr.data(), mat.col_ptr.get(),    (n_cells + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rowids.data(), mat.row_indices.get(),nnz * sizeof(int),           cudaMemcpyDeviceToHost);

    write_csc_bin(mat_path, h_vals.data(), h_indptr.data(), h_rowids.data(),
                  n_genes, n_cells, nnz);
    write_npy_i32(labels_path, pert_labels.data(), n_cells);
    write_npy_f32_2d(dose_path, dose.data(), n_cells, 1);

    const std::string script = std::string(BENCH_REFS_DIR) + "/cpa_ref.py";
    std::string cmd = "python3 " + script +
                      " --input "   + mat_path    +
                      " --labels "  + labels_path +
                      " --dose "    + dose_path   +
                      " --n-perts " + std::to_string(n_perts)  +
                      " --epochs "  + std::to_string(n_epochs) +
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
        bench::skip("bench_perturbation_cpa_perf", "no CUDA-capable GPU");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // ── Stage 1: load or synthesize matrix ─────────────────────────────────────
    int     n_cells     = kSyntheticN;
    int     n_genes     = kSyntheticM;
    int64_t nnz         = 0;
    const char* scale_label = "10k-synthetic";
    singlet_gpu::core::DeviceCSC mat;
    bool from_file = false;

    if (fs::exists(kSamplePath)) {
        try {
            auto loaded = singlet_gpu::io::load_pz(kSamplePath, stream);
            cudaStreamSynchronize(stream);
            n_cells     = loaded.mat.cols;
            n_genes     = loaded.mat.rows;
            nnz         = loaded.mat.nnz;
            mat         = std::move(loaded.mat);
            scale_label = "10k";
            from_file   = true;
            std::printf("[cpa_bench] .1pz: %d genes × %d cells, nnz=%lld\n",
                        n_genes, n_cells, (long long)nnz);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[cpa_bench] .1pz load failed: %s — synthetic\n", ex.what());
        }
    }
    if (!from_file) {
        std::printf("[cpa_bench] Synthesizing CSC %d × %d\n", n_genes, n_cells);
        mat = make_synthetic_csc(n_genes, n_cells, kSeed, stream);
        nnz = mat.nnz;
    }

    // ── Stage 2: synthesize perturbation labels + dose ─────────────────────────
    std::mt19937_64 rng(kSeed);
    std::uniform_int_distribution<int>    pert_dist(0, kNPerts - 1);
    std::uniform_real_distribution<float> dose_dist(0.1f, 1.0f);
    std::vector<int>   pert_labels(n_cells);
    std::vector<float> dose(n_cells);
    for (int i = 0; i < n_cells; ++i) {
        pert_labels[i] = pert_dist(rng);
        dose[i]        = dose_dist(rng);
    }

    // ── CPA config ─────────────────────────────────────────────────────────────
    singlet_gpu::perturbation::PerturbGraphConfig cfg;
    cfg.d_latent    = 64;
    cfg.d_hidden    = 128;
    cfg.n_epochs    = kNEpochs;
    cfg.batch_size  = 256;
    cfg.use_gnn     = false;  // GNN deferred (CYCLE-32-FOLLOWUP-GNN-EXTENSION)
    cfg.seed        = kSeed;

    // ── Build GPUContext (stream + handles from core/handles.h) ────────────────
    singlet_gpu::core::GPUContext ctx(stream);

    // ── Warmup ─────────────────────────────────────────────────────────────────
    std::printf("[cpa_bench] Warming up (%d iters × %d epochs)...\n", kWarmup, kNEpochs);
    for (int i = 0; i < kWarmup; ++i) {
        auto res = singlet_gpu::perturbation::train_perturb_graph(
            mat, pert_labels, dose, kNPerts, cfg, ctx);
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
        auto res = singlet_gpu::perturbation::train_perturb_graph(
            mat, pert_labels, dose, kNPerts, cfg, ctx);
        timer.stop(stream);
        cudaStreamSynchronize(stream);
        mem_tracker.sample_after();
        wall_ms_vec[i] = timer.elapsed_ms();
        std::printf("[cpa_bench] iter %d: %.1f ms  epoch_loss[last]=%.4f\n",
                    i, wall_ms_vec[i],
                    res.epoch_losses.empty() ? 0.f : res.epoch_losses.back());
        (void)res;
    }

    // ── Statistics ─────────────────────────────────────────────────────────────
    std::sort(wall_ms_vec.begin(), wall_ms_vec.end());
    double wall_median = wall_ms_vec[kTimed / 2];
    double wall_min    = wall_ms_vec.front();
    double wall_max    = wall_ms_vec.back();
    double peak_mb     = mem_tracker.peak_delta_mb();
    double cells_sec   = bench::throughput(n_cells, wall_median);

    std::printf("[cpa_bench] Result: median=%.1fms min=%.1fms max=%.1fms "
                "mem=%.1fMB cells/s=%.0f\n",
                wall_median, wall_min, wall_max, peak_mb, cells_sec);

    // ── SOTA reference ─────────────────────────────────────────────────────────
    bench::RefResult ref = run_cpa_reference(mat, pert_labels, dose,
                                              n_genes, n_cells, nnz,
                                              kNPerts, kNEpochs);

    // ── Log BenchRow ───────────────────────────────────────────────────────────
    bench::BenchRow row;
    row.date          = bench::today_iso();
    row.feature       = "perturbation/cpa";
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
