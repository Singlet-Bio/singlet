// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_de_donor_pseudobulk_perf.cpp
//
// Performance benchmark: singlet::gpu::de::donor_pseudobulk_de (cycle 15)
//   vs DESeq2 R (pydeseq2 fallback).
//
// Design reference: singlet/gpu/state/designs/15-donor-pseudobulk.md
// Kernel:           include/singlet/gpu/de/donor_pseudobulk.h
//
// Staged pipeline (NOT timed):
//   - Load exon_counts.1pz from GSM4037629 if present; else synthetic CSC.
//   - Synthesize cluster_labels and donor_labels from donor_assignments.tsv
//     if present; else synthetic (10 clusters × 5 donors).
//
// Timed (cuEvents):
//   - donor_pseudobulk_de(mat, cluster_labels, n_clusters,
//                         donor_labels, n_donors, cfg, stream)
//   Upstream lognorm / PCA / Leiden are NOT timed.
//
// Warmup: 3 discarded iterations.
// Timed:  5 iterations.
//
// SOTA reference: bench/refs/donor_pseudobulk_ref.py (DESeq2 R or pydeseq2).
//
// Skip conditions:
//   - No CUDA-capable GPU: print message, return 0.

#include "singlet/gpu/bench/harness.h"

#include <singlet/gpu/de/donor_pseudobulk.h>
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

static constexpr int      kNCells    = 10000;  // synthetic cell count
static constexpr int      kNGenes    = 2000;   // synthetic gene count
static constexpr int      kNClusters = 4;      // cluster count
static constexpr int      kNDonors   = 5;      // donor count (≥3 required by NB GLM)
static constexpr float    kDensity   = 0.05f;
static constexpr uint64_t kSeed      = 0xDEADBEEF01ULL;

static const char* kSamplePath =
    "/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna/"
    "GSE127/GSE127918/GSM4037629/counts.1pz";

static const char* kDonorAssignPath =
    "/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna/"
    "GSE127/GSE127918/GSM4037629/donor_assignments.tsv";

// ─── Helpers ──────────────────────────────────────────────────────────────────

namespace {

// Sparse CSC for count matrix (n_genes rows, n_cells cols).
singlet::gpu::core::DeviceCSC make_count_csc(
        int n_genes, int n_cells, float density, uint64_t seed, cudaStream_t stream)
{
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution bern(density);
    std::uniform_int_distribution<int> cnt(1, 20);

    std::vector<int>   col_ptr(n_cells + 1, 0);
    std::vector<int>   row_idx;
    std::vector<float> vals;

    for (int c = 0; c < n_cells; ++c) {
        col_ptr[c] = static_cast<int>(row_idx.size());
        for (int g = 0; g < n_genes; ++g) {
            if (bern(rng)) { row_idx.push_back(g); vals.push_back(float(cnt(rng))); }
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
    return singlet::gpu::core::DeviceCSC(n_genes, n_cells, nnz,
                                        std::move(d_cp), std::move(d_ri), std::move(d_vl));
}

// Upload int labels to device.
singlet::gpu::core::DeviceMemory<int> upload_labels(
        const std::vector<int>& h_labels, cudaStream_t stream)
{
    singlet::gpu::core::DeviceMemory<int> d(h_labels.size());
    cudaMemcpyAsync(d.data(), h_labels.data(), h_labels.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return d;
}

// Try to parse donor_assignments.tsv to extract donor_id per cell.
// Returns empty vector on failure (caller uses synthetic labels).
std::vector<int> try_load_donor_labels(const char* path, int n_cells)
{
    std::ifstream f(path);
    if (!f) return {};
    std::string line;
    std::getline(f, line);  // header
    std::vector<int> donor_ids;
    donor_ids.reserve(n_cells);
    while (std::getline(f, line)) {
        // Expected columns: cell_barcode, donor_id, prob_max, prob_doublet
        // donor_id is a string like "donor_1"; we parse the numeric suffix.
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        auto tab2 = line.find('\t', tab + 1);
        std::string donor_str = line.substr(tab + 1, tab2 - tab - 1);
        // Extract trailing integer, e.g. "donor_3" → 2 (0-indexed).
        int id = 0;
        for (char ch : donor_str) if (ch >= '0' && ch <= '9') id = id * 10 + (ch - '0');
        donor_ids.push_back(id - 1);  // convert 1-indexed label to 0-indexed
    }
    if ((int)donor_ids.size() != n_cells) return {};
    return donor_ids;
}

// Run DESeq2 / pydeseq2 SOTA reference.
// bench/refs/donor_pseudobulk_ref.py accepts --n-cells / --n-genes / --n-clusters / --n-donors.
std::pair<float, float> run_pseudobulk_sota(
        int n_cells, int n_genes, int n_clusters, int n_donors)
{
    std::string cmd =
        "python3 bench/refs/donor_pseudobulk_ref.py"
        " --n-cells "    + std::to_string(n_cells)    +
        " --n-genes "    + std::to_string(n_genes)    +
        " --n-clusters " + std::to_string(n_clusters) +
        " --n-donors "   + std::to_string(n_donors)   +
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

    int   n_cells    = kNCells;
    int   n_genes    = kNGenes;
    int   n_clusters = kNClusters;
    int   n_donors   = kNDonors;
    const char* scale_label = "tiny-synthetic";

    // ── Load or synthesize matrix (NOT timed) ─────────────────────────────────
    singlet::gpu::core::DeviceCSC mat;
    bool from_file = false;

    if (fs::exists(kSamplePath)) {
        try {
            auto loaded = singlet::gpu::io::load_pz(kSamplePath, stream);
            cudaStreamSynchronize(stream);
            n_cells    = loaded.mat.cols;
            n_genes    = loaded.mat.rows;
            mat        = std::move(loaded.mat);
            scale_label = "10k";
            from_file  = true;
            std::printf("[pseudobulk_bench] .1pz loaded: %d genes × %d cells\n",
                        n_genes, n_cells);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[pseudobulk_bench] .1pz load failed: %s\n", ex.what());
        }
    }

    if (!from_file) {
        std::printf("[pseudobulk_bench] Synthesizing CSC: %d genes × %d cells\n",
                    n_genes, n_cells);
        mat = make_count_csc(n_genes, n_cells, kDensity, kSeed, stream);
    }

    // ── Build cluster + donor labels (NOT timed) ──────────────────────────────
    std::mt19937_64 rng_lbl(kSeed ^ 0xLBL);
    std::uniform_int_distribution<int> cl_dist(0, n_clusters - 1);
    std::uniform_int_distribution<int> dn_dist(0, n_donors - 1);

    std::vector<int> h_cluster_labels(n_cells);
    std::vector<int> h_donor_labels(n_cells);

    if (from_file) {
        auto loaded_donors = try_load_donor_labels(kDonorAssignPath, n_cells);
        if (!loaded_donors.empty()) {
            h_donor_labels = std::move(loaded_donors);
            // Determine n_donors from the loaded labels.
            n_donors = *std::max_element(h_donor_labels.begin(), h_donor_labels.end()) + 1;
            if (n_donors < 3) {
                std::fprintf(stderr,
                    "[pseudobulk_bench] loaded donors=%d < 3, using synthetic donors\n",
                    n_donors);
                n_donors = kNDonors;
                loaded_donors.clear();
            }
        }
        if (loaded_donors.empty()) {
            std::uniform_int_distribution<int> dn2(0, n_donors - 1);
            for (int& d : h_donor_labels) d = dn2(rng_lbl);
        }
        // Synthesize cluster labels (Leiden not staged; just random for bench).
        for (int& c : h_cluster_labels) c = cl_dist(rng_lbl);
    } else {
        for (int& c : h_cluster_labels) c = cl_dist(rng_lbl);
        for (int& d : h_donor_labels)   d = dn_dist(rng_lbl);
    }

    auto d_cluster = upload_labels(h_cluster_labels, stream);
    auto d_donor   = upload_labels(h_donor_labels,   stream);

    // ── Config ────────────────────────────────────────────────────────────────
    singlet::gpu::de::DonorPseudobulkConfig cfg;
    cfg.min_cells_per_pseudobulk = 3;
    cfg.max_irls_iters           = 20;
    cfg.irls_tol                 = 1e-4f;
    cfg.max_dispersion_iters     = 3;
    cfg.apeglm_shrinkage         = false;  // disable for speed in bench
    cfg.top_n                    = 50;

    // ── Warmup ────────────────────────────────────────────────────────────────
    std::printf("[pseudobulk_bench] Warming up (%d iters)...\n", BENCH_WARMUP_ITERS);
    for (int i = 0; i < BENCH_WARMUP_ITERS; ++i) {
        try {
            auto r = singlet::gpu::de::donor_pseudobulk_de(
                mat, d_cluster, n_clusters, d_donor, n_donors, cfg, stream);
            cudaStreamSynchronize(stream);
            (void)r;
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[pseudobulk_bench] warmup error: %s — skipping\n",
                         ex.what());
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
        auto r = singlet::gpu::de::donor_pseudobulk_de(
            mat, d_cluster, n_clusters, d_donor, n_donors, cfg, stream);
        timer.stop();
        timer.record();
        mem_tracker.snapshot_after();
        (void)r;
        std::printf("[pseudobulk_bench] iter %d done\n", i);
    }

    std::printf("[pseudobulk_bench] median=%.1fms min=%.1fms max=%.1fms mem=%.1fMB\n",
                timer.median_ms(), timer.min_ms(), timer.max_ms(), mem_tracker.peak_mb());

    // ── SOTA reference ────────────────────────────────────────────────────────
    auto [ref_wall, ref_mem] = run_pseudobulk_sota(n_cells, n_genes, n_clusters, n_donors);

    // ── Log BenchRow ──────────────────────────────────────────────────────────
    BenchRow row;
    row.date          = current_date_str();
    row.feature       = "de/donor_pseudobulk";
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

    std::printf("[pseudobulk_bench] Done.\n");
    cudaStreamDestroy(stream);
    return 0;
}
