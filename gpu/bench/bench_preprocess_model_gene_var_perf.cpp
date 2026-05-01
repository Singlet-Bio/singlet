// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_preprocess_model_gene_var_perf.cpp
//
// CYCLE-162 Phase E — preprocess/model_gene_var standalone kernel benchmark.
//
// Benchmarks model_gene_var() (include/singlet-gpu/preprocess/model_gene_var.h)
// against synthetic CSC log-normalized matrices at two medium scales:
//
//   Scale "10k":  n_cells=10000, n_genes=5000, density=5%
//   Scale "30k":  n_cells=30000, n_genes=5000, density=5%
//
// Kernel: GPU port of scran::modelGeneVarByPoisson (Lun-McCarthy-Marioni 2016).
// Algorithm: O(nnz) scatter passes + cub::DeviceRadixSort top-N HVG selection.
// Per §J.6 audit: NOT at risk of the diffmap/dpt dense-n×n scaling bug.
//
// Protocol: 2 warmup + 5 timed iterations; wall time via cudaEventElapsedTime.
// Memory via cudaMemGetInfo delta (PeakMemTracker).
//
// Output (stdout): CSV header + rows:
//   scale,n_cells,n_genes,density,n_top,wall_ms,mem_mb
//
// Node: any non-excluded GPU node (--exclude=g001,g002,g005 per §J.2).

#include <singlet_gpu/bench/harness.h>
#include <singlet-gpu/preprocess/model_gene_var.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

using namespace singlet_gpu;

// ---------------------------------------------------------------------------
// upload_synthetic_csc — build a CSC sparse matrix on device from host vecs.
// Returns a PzDeviceMatrix populated from the SyntheticMatrix.
// Uses core::DeviceMemory<T> throughout (Rule 5 — no raw cudaMalloc).
// ---------------------------------------------------------------------------
static io::PzDeviceMatrix upload_synthetic_csc(const SyntheticMatrix& syn,
                                               cudaStream_t stream)
{
    core::DeviceMemory<int>   d_col_ptr(syn.cols + 1);
    core::DeviceMemory<int>   d_row_idx(syn.nnz);
    core::DeviceMemory<float> d_values(syn.nnz);

    cudaMemcpyAsync(d_col_ptr.get(), syn.indptr.data(),
                    (syn.cols + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_row_idx.get(), syn.indices.data(),
                    syn.nnz * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_values.get(), syn.values.data(),
                    syn.nnz * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
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
// bench_scale — 2 warmup + 5 timed iterations of model_gene_var.
// Returns median wall_ms and peak mem_mb.
// ---------------------------------------------------------------------------
static void bench_scale(
    const io::PzDeviceMatrix&           pzmat,
    const preprocess::ModelGeneVarConfig& cfg,
    cudaStream_t                        stream,
    double& out_wall_ms,
    double& out_mem_mb)
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
        // Call the CYCLE-127 model_gene_var kernel — result discarded (wall-time only).
        // model_gene_var syncs internally before returning (Rule 9 / header doc).
        auto result = preprocess::model_gene_var(pzmat, cfg, stream);
        (void)result;
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
        std::puts("NO GPU — skipping Cycle 162 model_gene_var bench.");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // CSV header.
    std::printf("scale,n_cells,n_genes,density,n_top,wall_ms,mem_mb\n");
    std::fflush(stdout);

    // Config: n_top=2000, min_mean=0 (no filter — pure bench of the full path).
    preprocess::ModelGeneVarConfig cfg;
    cfg.n_top    = 2000;
    cfg.min_mean = 0.0f;

    // Scales: 10k and 30k cells, 5000 genes, density 5%.
    struct Scale { const char* name; int n_cells; int n_genes; float density; };
    const Scale scales[] = {
        { "10k",  10000, 5000, 0.05f },
        { "30k",  30000, 5000, 0.05f },
    };

    for (const auto& s : scales) {
        std::printf("[bench] Synthesizing %s: %d cells × %d genes density=%.0f%%...\n",
                    s.name, s.n_cells, s.n_genes, s.density * 100.f);
        std::fflush(stdout);

        // SyntheticMatrix: rows=genes (m), cols=cells (n) — CSC layout.
        SyntheticMatrix syn = make_synthetic_matrix(
            s.n_genes, s.n_cells, s.density, /*seed=*/42);

        io::PzDeviceMatrix pzmat = upload_synthetic_csc(syn, stream);

        double wall_ms = 0.0, mem_mb = 0.0;
        bench_scale(pzmat, cfg, stream, wall_ms, mem_mb);

        // CSV row.
        std::printf("%s,%d,%d,%.2f,%d,%.3f,%.1f\n",
                    s.name, s.n_cells, s.n_genes, s.density, cfg.n_top,
                    wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "preprocess/model_gene_var";
        row.scale         = s.name;
        row.impl          = "singlet-gpu";
        row.wall_ms       = wall_ms;
        row.mem_mb        = mem_mb;
        row.cells_per_sec = bench::throughput(static_cast<int64_t>(s.n_cells), wall_ms);
        row.commit        = bench::git_short_sha();
        bench::log_row(row);
    }

    cudaStreamDestroy(stream);
    std::printf("[bench] Cycle 162 model_gene_var perf bench complete.\n");
    return 0;
}
