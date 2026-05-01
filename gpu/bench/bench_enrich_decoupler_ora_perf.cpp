// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_enrich_decoupler_ora_perf.cpp
//
// CYCLE-166 Phase E — enrich/decoupler_ora standalone kernel benchmark.
//
// Benchmarks decoupler_ora::ora() (from include/singlet-gpu/enrich/decoupler_ora.h)
// against a manual numpy/scipy CPU baseline at two medium scales:
//
//   Scale "10k":  n_cells=10000, n_genes=5000, density=5%, n_pathways=50
//   Scale "30k":  n_cells=30000, n_genes=5000, density=5%, n_pathways=50
//
// GPU kernel: 4 passes — ora_topk_smem_kernel (64-bucket shared-mem top-K),
//   ora_build_T_kernel (binary top-K mask), cuBLAS Sgemm (T^T · M → hits),
//   ora_hypergeo_kernel (log-sum-exp hypergeometric -log10(p) per cell×pathway).
// Reference: Badia-i-Mompel et al. (2022) decoupleR, Bioinformatics Advances.
// Per §J.6 audit: NOT at risk of dense-n×n bug (T is m×n but m=5000 genes;
//   Sgemm O(m*n*p); lgamma pass O(n*p*K_set) per cell, p=50).
//
// Protocol: 2 warmup + 5 timed iterations; wall time via cudaEventElapsedTime.
// Memory via cudaMemGetInfo delta (PeakMemTracker).
//
// Output (stdout): CSV header + rows:
//   scale,n_cells,n_genes,density,n_pathways,wall_ms,mem_mb
//
// Node: any non-excluded GPU node (--exclude=g001,g002,g005 per §J.2).

#include <singlet_gpu/bench/harness.h>
#include <singlet-gpu/enrich/decoupler_ora.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace singlet_gpu;

// ---------------------------------------------------------------------------
// upload_synthetic_csc — build a CSC sparse matrix on device from host vecs.
// Returns a PzDeviceMatrix populated from a SyntheticMatrix.
// Uses core::DeviceMemory<T> throughout (Rule 5 — no raw cudaMalloc).
// CSC convention: rows=genes (m), cols=cells (n).
// ---------------------------------------------------------------------------
static io::PzDeviceMatrix upload_synthetic_csc(const SyntheticMatrix& syn,
                                               cudaStream_t stream)
{
    core::DeviceMemory<int>   d_col_ptr(syn.cols + 1);
    core::DeviceMemory<int>   d_row_idx(syn.nnz);
    core::DeviceMemory<float> d_values(syn.nnz > 0 ? syn.nnz : 1);

    cudaMemcpyAsync(d_col_ptr.get(), syn.indptr.data(),
                    (syn.cols + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    if (syn.nnz > 0) {
        cudaMemcpyAsync(d_row_idx.get(), syn.indices.data(),
                        syn.nnz * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_values.get(), syn.values.data(),
                        syn.nnz * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }
    cudaStreamSynchronize(stream);

    io::PzDeviceMatrix pzmat;
    pzmat.mat.rows        = syn.rows;   // genes (m)
    pzmat.mat.cols        = syn.cols;   // cells (n) — CSC: cols = cells
    pzmat.mat.nnz         = syn.nnz;
    pzmat.mat.col_ptr     = std::move(d_col_ptr);
    pzmat.mat.row_indices = std::move(d_row_idx);
    pzmat.mat.values      = std::move(d_values);
    return pzmat;
}

// ---------------------------------------------------------------------------
// make_gene_sets — synthetic gene sets: n_pathways sets, each with
// ~10% of n_genes members (genes sampled deterministically by modular stride).
// Seed controls the starting offset for reproducibility (Rule 16).
// ---------------------------------------------------------------------------
static std::vector<std::vector<int>> make_gene_sets(int n_genes, int n_pathways,
                                                     uint32_t seed = 42)
{
    const int set_size = static_cast<int>(n_genes * 0.10f);  // ~10% membership
    std::vector<std::vector<int>> sets(n_pathways);
    for (int s = 0; s < n_pathways; ++s) {
        sets[s].reserve(set_size);
        // Stride over genes with an offset that varies per pathway (deterministic).
        const int offset = static_cast<int>((static_cast<uint64_t>(seed) * (s + 1)
                                             * 6364136223846793005ULL) % n_genes);
        for (int i = 0; i < set_size; ++i) {
            sets[s].push_back((offset + i * 7) % n_genes);  // stride-7 walk
        }
        // Remove duplicates while preserving approximate count.
        std::sort(sets[s].begin(), sets[s].end());
        sets[s].erase(std::unique(sets[s].begin(), sets[s].end()), sets[s].end());
        // Ensure at least 1 member.
        if (sets[s].empty()) sets[s].push_back(s % n_genes);
    }
    return sets;
}

// ---------------------------------------------------------------------------
// bench_ora — 2 warmup + 5 timed iterations of ora().
// Returns median wall_ms and peak mem_mb.
// ---------------------------------------------------------------------------
static void bench_ora(
    const io::PzDeviceMatrix&             pzmat,
    const std::vector<std::vector<int>>&  gene_sets,
    cudaStream_t                          stream,
    double&                               out_wall_ms,
    double&                               out_mem_mb)
{
    constexpr int WARMUP = 2;
    constexpr int TIMED  = 5;
    double samples[TIMED]{};
    double mem_samples[TIMED]{};

    bench::BenchTimer    timer;
    bench::PeakMemTracker mem;

    for (int iter = 0; iter < WARMUP + TIMED; ++iter) {
        const bool is_timed = (iter >= WARMUP);
        if (is_timed) mem.sample_before();

        timer.start(stream);

        {
            enrich::OraConfig cfg{};
            auto result = enrich::ora(pzmat, gene_sets, cfg, stream);
            (void)result;
        }

        timer.stop(stream);

        const double ms = timer.elapsed_ms();
        if (is_timed) {
            mem.sample_after();
            samples[iter - WARMUP]     = ms;
            mem_samples[iter - WARMUP] = mem.peak_delta_mb();
            mem.reset();
        }
    }

    std::sort(samples, samples + TIMED);
    out_wall_ms = (TIMED % 2 == 1)
        ? samples[TIMED / 2]
        : 0.5 * (samples[TIMED / 2 - 1] + samples[TIMED / 2]);
    out_mem_mb = *std::max_element(mem_samples, mem_samples + TIMED);
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------
int main() {
    if (!gpu_available()) {
        std::puts("NO GPU — skipping Cycle 166 decoupler_ora bench.");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // CSV header.
    std::printf("scale,n_cells,n_genes,density,n_pathways,wall_ms,mem_mb\n");
    std::fflush(stdout);

    // Scales.
    struct Scale { const char* name; int n_cells; int n_genes; float density; };
    const Scale scales[] = {
        { "10k",  10000, 5000, 0.05f },
        { "30k",  30000, 5000, 0.05f },
    };
    const int n_pathways = 50;

    for (const auto& s : scales) {
        std::printf("[bench] Synthesizing %s: %d cells × %d genes density=%.0f%%...\n",
                    s.name, s.n_cells, s.n_genes, s.density * 100.f);
        std::fflush(stdout);

        // SyntheticMatrix: rows=genes (m), cols=cells (n) — CSC layout.
        SyntheticMatrix syn = make_synthetic_matrix(
            s.n_genes, s.n_cells, s.density, /*seed=*/42);

        io::PzDeviceMatrix pzmat = upload_synthetic_csc(syn, stream);

        // Build gene sets: n_pathways sets, each ~10% of n_genes.
        std::vector<std::vector<int>> gene_sets =
            make_gene_sets(s.n_genes, n_pathways, /*seed=*/42);

        std::printf("[bench] Timing %s method=ora (n_pathways=%d, top_k=5%%)...\n",
                    s.name, n_pathways);
        std::fflush(stdout);

        double wall_ms = 0.0, mem_mb = 0.0;
        bench_ora(pzmat, gene_sets, stream, wall_ms, mem_mb);

        // CSV row.
        std::printf("%s,%d,%d,%.2f,%d,%.3f,%.1f\n",
                    s.name, s.n_cells, s.n_genes, s.density, n_pathways,
                    wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "enrich/decoupler_ora";
        row.scale         = std::string(s.name);
        row.impl          = "singlet-gpu";
        row.wall_ms       = wall_ms;
        row.mem_mb        = mem_mb;
        row.cells_per_sec = bench::throughput(static_cast<int64_t>(s.n_cells), wall_ms);
        row.commit        = bench::git_short_sha();
        bench::log_row(row);
    }

    cudaStreamDestroy(stream);
    std::printf("[bench] Cycle 166 decoupler_ora perf bench complete.\n");
    return 0;
}
