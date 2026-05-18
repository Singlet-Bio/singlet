// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_gsea_fgsea_perf.cpp
//
// Performance benchmark: singlet::gpu::gsea::fgsea + singlet::gpu::gsea::aucell
//   vs fgsea R + AUCell R (cycle 11).
//
// Design reference: singlet/gpu/state/designs/11-gsea-fgsea.md
// Kernels:          include/singlet/gpu/gsea/fgsea.h
//                   include/singlet/gpu/gsea/aucell.h
//
// Staged pipeline (NOT timed):
//   - Build synthetic ranked gene-stat vector (m=22000 genes, ~100 pathways).
//   - Build GeneSetDB (host-side setup).
//
// Timed (per-target, cuEvents):
//   - Part A: fgsea() GPU kernel.
//   - Part B: aucell() GPU kernel on a tiny synthetic CSC (~11560 cells).
//
// Warmup: 3 discarded iterations per target.
// Timed:  5 iterations per target.
//
// SOTA reference: bench/refs/fgsea_ref.py dispatches fgsea R or AUCell R via
// subprocess; returns SOTA_WALL_SEC / SOTA_MEM_MB on stdout (common.py protocol).
//
// Skip conditions:
//   - No CUDA-capable GPU: print message, return 0.

#include "singlet/gpu/bench/harness.h"

#include <singlet/gpu/gsea/fgsea.h>
#include <singlet/gpu/gsea/aucell.h>
#include <singlet/gpu/anno/types.h>
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

static constexpr int      kNGenes    = 22000;  // genes for fgsea stats
static constexpr int      kNPathways = 100;
static constexpr int      kMinSet    = 15;
static constexpr int      kMaxSet    = 200;
static constexpr int      kNCells    = 11560;  // AUCell: canonical sample count
static constexpr uint64_t kSeed      = 0xFEEDCAFE42ULL;

// ─── Helpers ──────────────────────────────────────────────────────────────────

namespace {

singlet::gpu::anno::GeneSetDB make_synthetic_genesets(
        int n_genes, int n_pathways, int min_set, int max_set, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> size_dist(min_set, max_set);
    std::uniform_int_distribution<int> gene_dist(0, n_genes - 1);

    singlet::gpu::anno::GeneSetDB db;
    db.set_names.resize(n_pathways);
    db.member_gene_indices.resize(n_pathways);

    for (int p = 0; p < n_pathways; ++p) {
        db.set_names[p] = "SYN_" + std::to_string(p);
        int sz = size_dist(rng);
        std::vector<bool> chosen(n_genes, false);
        std::vector<int>  members;
        members.reserve(sz);
        while ((int)members.size() < sz) {
            int g = gene_dist(rng);
            if (!chosen[g]) { chosen[g] = true; members.push_back(g); }
        }
        std::sort(members.begin(), members.end());
        db.member_gene_indices[p] = std::move(members);
    }
    return db;
}

singlet::gpu::core::DeviceMemory<float> make_stats_device(
        int n_genes, uint64_t seed, cudaStream_t stream)
{
    std::mt19937_64 rng(seed ^ 0xBEEF);
    std::normal_distribution<float> ndist(0.f, 1.f);
    std::vector<float> h(n_genes);
    for (auto& v : h) v = ndist(rng);
    std::sort(h.begin(), h.end(), std::greater<float>());

    singlet::gpu::core::DeviceMemory<float> d(n_genes);
    cudaMemcpyAsync(d.data(), h.data(), n_genes * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return d;
}

singlet::gpu::core::DeviceCSC make_tiny_csc(
        int n_genes, int n_cells, float density, uint64_t seed, cudaStream_t stream)
{
    std::mt19937_64 rng(seed ^ 0xCAFE);
    std::bernoulli_distribution bern(density);
    std::uniform_int_distribution<int> cnt(1, 15);

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
    cudaMemcpyAsync(d_cp.data(), col_ptr.data(), (n_cells+1)*sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ri.data(), row_idx.data(), nnz*sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_vl.data(), vals.data(), nnz*sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return singlet::gpu::core::DeviceCSC(n_genes, n_cells, nnz,
                                        std::move(d_cp), std::move(d_ri), std::move(d_vl));
}

// Invoke fgsea_ref.py; returns (wall_sec, mem_mb) or (-1,-1) on failure.
// Writes stats + pathway JSON to /tmp, then reads from stdout (common.py protocol).
std::pair<float, float> run_fgsea_sota(
        const singlet::gpu::core::DeviceMemory<float>& d_stats,
        const singlet::gpu::anno::GeneSetDB& db)
{
    fs::create_directories("/tmp/singlet_gpu_bench");

    // Download stats to host.
    const int n = static_cast<int>(d_stats.size());
    std::vector<float> h_stats(n);
    cudaMemcpy(h_stats.data(), d_stats.data(), n * sizeof(float), cudaMemcpyDeviceToHost);

    // Write stats .npy.
    const std::string stats_path = "/tmp/singlet_gpu_bench/fgsea_stats.npy";
    {
        std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                          std::to_string(n) + ",), }";
        constexpr int kPre = 10;
        int pad = ((static_cast<int>(hdr.size()) + kPre + 63) / 64) * 64 - kPre;
        while (static_cast<int>(hdr.size()) < pad - 1) hdr += ' ';
        hdr += '\n';
        uint16_t hl = static_cast<uint16_t>(hdr.size());
        std::ofstream f(stats_path, std::ios::binary);
        const char magic[] = "\x93NUMPY";
        f.write(magic, 6); f.put(1); f.put(0);
        f.write(reinterpret_cast<const char*>(&hl), 2);
        f.write(hdr.data(), hl);
        f.write(reinterpret_cast<const char*>(h_stats.data()), n * sizeof(float));
    }

    // Write pathways JSON.
    const std::string pw_path = "/tmp/singlet_gpu_bench/fgsea_pathways.json";
    {
        std::ofstream f(pw_path);
        f << "{\n";
        for (int p = 0; p < (int)db.set_names.size(); ++p) {
            f << "  \"" << db.set_names[p] << "\": [";
            for (int i = 0; i < (int)db.member_gene_indices[p].size(); ++i) {
                if (i) f << ',';
                f << db.member_gene_indices[p][i];
            }
            f << "]" << (p+1 < (int)db.set_names.size() ? "," : "") << "\n";
        }
        f << "}\n";
    }

    // Launch reference script.
    std::string script = "bench/refs/fgsea_ref.py";
    std::string cmd = "python3 " + script +
                      " --mode fgsea"
                      " --stats " + stats_path +
                      " --pathways " + pw_path +
                      " --n-genes " + std::to_string(n) +
                      " 2>/dev/null";

    // Parse common.py protocol from stdout.
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return {-1.f, -1.f};
    float wall_sec = -1.f, mem_mb = -1.f;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "SOTA_WALL_SEC=", 14) == 0) wall_sec = atof(line + 14);
        if (strncmp(line, "SOTA_MEM_MB=",   12) == 0) mem_mb   = atof(line + 12);
    }
    pclose(fp);
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

    // ── Build synthetic data (NOT timed) ──────────────────────────────────────
    std::printf("[fgsea_bench] Building data: m=%d genes, %d pathways, %d cells\n",
                kNGenes, kNPathways, kNCells);

    auto d_stats  = make_stats_device(kNGenes, kSeed, stream);
    auto genesets = make_synthetic_genesets(kNGenes, kNPathways, kMinSet, kMaxSet, kSeed);
    auto aucell_mat = make_tiny_csc(kNGenes, kNCells, 0.05f, kSeed, stream);

    // ── fgsea config ──────────────────────────────────────────────────────────
    singlet::gpu::gsea::FgseaConfig fgsea_cfg;
    fgsea_cfg.min_perm     = 1000;
    fgsea_cfg.max_perm     = 10000;
    fgsea_cfg.seed         = kSeed;
    fgsea_cfg.min_set_size = kMinSet;
    fgsea_cfg.max_set_size = kMaxSet;

    // ── Warmup: fgsea ──────────────────────────────────────────────────────────
    std::printf("[fgsea_bench] Warmup fgsea (%d iters)...\n", BENCH_WARMUP_ITERS);
    for (int i = 0; i < BENCH_WARMUP_ITERS; ++i) {
        auto r = singlet::gpu::gsea::fgsea(d_stats, genesets, fgsea_cfg, stream);
        cudaStreamSynchronize(stream);
        (void)r;
    }

    // ── Timed: fgsea ─────────────────────────────────────────────────────────
    BenchTimer        fgsea_timer(stream);
    PeakMemoryTracker fgsea_mem;
    for (int i = 0; i < BENCH_TIMED_ITERS; ++i) {
        fgsea_mem.snapshot_before();
        fgsea_timer.start();
        auto r = singlet::gpu::gsea::fgsea(d_stats, genesets, fgsea_cfg, stream);
        fgsea_timer.stop();
        fgsea_timer.record();
        fgsea_mem.snapshot_after();
        (void)r;
        std::printf("[fgsea_bench] fgsea iter %d: %.1f ms\n", i,
                    fgsea_timer.max_ms());  // max_ms is last recorded
    }

    // ── SOTA ref: fgsea R ─────────────────────────────────────────────────────
    auto [ref_fgsea_wall, ref_fgsea_mem] = run_fgsea_sota(d_stats, genesets);

    // ── Log fgsea BenchRow ────────────────────────────────────────────────────
    {
        BenchRow row;
        row.date         = current_date_str();
        row.feature      = "gsea/fgsea";
        row.scale        = "tiny";
        row.impl         = "singlet-gpu";
        row.wall_ms_min  = fgsea_timer.min_ms();
        row.wall_ms_med  = fgsea_timer.median_ms();
        row.wall_ms_max  = fgsea_timer.max_ms();
        row.mem_mb_peak  = fgsea_mem.peak_mb();
        row.cells_per_sec = throughput(kNGenes, row.wall_ms_med);
        row.sota_wall_sec = ref_fgsea_wall;
        row.sota_mem_mb   = ref_fgsea_mem;
        row.commit        = "no-git";
        log_row(row);
    }

    // ── AUCell config ─────────────────────────────────────────────────────────
    singlet::gpu::gsea::AUCellConfig aucell_cfg;
    aucell_cfg.top_k_genes = 500;
    aucell_cfg.n_bins      = 4096;
    aucell_cfg.cell_tile   = 4096;

    // ── Warmup: AUCell ────────────────────────────────────────────────────────
    std::printf("[fgsea_bench] Warmup AUCell (%d iters)...\n", BENCH_WARMUP_ITERS);
    for (int i = 0; i < BENCH_WARMUP_ITERS; ++i) {
        auto r = singlet::gpu::gsea::aucell(aucell_mat, genesets, aucell_cfg, stream);
        cudaStreamSynchronize(stream);
        (void)r;
    }

    // ── Timed: AUCell ─────────────────────────────────────────────────────────
    BenchTimer        aucell_timer(stream);
    PeakMemoryTracker aucell_mem;
    for (int i = 0; i < BENCH_TIMED_ITERS; ++i) {
        aucell_mem.snapshot_before();
        aucell_timer.start();
        auto r = singlet::gpu::gsea::aucell(aucell_mat, genesets, aucell_cfg, stream);
        aucell_timer.stop();
        aucell_timer.record();
        aucell_mem.snapshot_after();
        (void)r;
    }

    // ── SOTA ref: AUCell R ────────────────────────────────────────────────────
    // Reuse fgsea_ref.py with --mode aucell.
    std::string aucell_cmd =
        "python3 bench/refs/fgsea_ref.py"
        " --mode aucell"
        " --pathways /tmp/singlet_gpu_bench/fgsea_pathways.json"
        " --n-genes " + std::to_string(kNGenes) +
        " --n-cells " + std::to_string(kNCells) +
        " 2>/dev/null";
    float ref_aucell_wall = -1.f, ref_aucell_mem = -1.f;
    {
        FILE* fp = popen(aucell_cmd.c_str(), "r");
        if (fp) {
            char line[256];
            while (fgets(line, sizeof(line), fp)) {
                if (strncmp(line, "SOTA_WALL_SEC=", 14) == 0) ref_aucell_wall = atof(line+14);
                if (strncmp(line, "SOTA_MEM_MB=",   12) == 0) ref_aucell_mem  = atof(line+12);
            }
            pclose(fp);
        }
    }

    // ── Log AUCell BenchRow ───────────────────────────────────────────────────
    {
        BenchRow row;
        row.date         = current_date_str();
        row.feature      = "gsea/aucell";
        row.scale        = "tiny";
        row.impl         = "singlet-gpu";
        row.wall_ms_min  = aucell_timer.min_ms();
        row.wall_ms_med  = aucell_timer.median_ms();
        row.wall_ms_max  = aucell_timer.max_ms();
        row.mem_mb_peak  = aucell_mem.peak_mb();
        row.cells_per_sec = throughput(kNCells, row.wall_ms_med);
        row.sota_wall_sec = ref_aucell_wall;
        row.sota_mem_mb   = ref_aucell_mem;
        row.commit        = "no-git";
        log_row(row);
    }

    std::printf("[fgsea_bench] Done.\n");
    cudaStreamDestroy(stream);
    return 0;
}
