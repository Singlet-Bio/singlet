// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_enrich_decoupler_ulm_perf.cpp
//
// CYCLE-164 Phase E — enrich/decoupler_ulm standalone kernel benchmark.
//
// Benchmarks decoupler_ulm::ulm() (from include/singlet/gpu/enrich/decoupler_ulm.h)
// against a manual numpy CPU baseline at two medium scales:
//
//   Scale "10k":  n_cells=10000, n_genes=5000, density=5%, n_pathways=50
//   Scale "30k":  n_cells=30000, n_genes=5000, density=5%, n_pathways=50
//
// GPU kernel: 5 passes — per-cell mean (scatter+divide), per-pathway W stats
//   (fused warp-shuffle), var_W finalization, cuSPARSE SpMM (X^T · W), score
//   kernel (OLS slope β_1[c,p] = cov(X_c,W_p)/var_W[p]).
// Reference: Badia-i-Mompel et al. (2022) decoupleR, Bioinformatics Advances.
// Per §J.6 audit: NOT at risk of dense-n×n bug (SpMM O(nnz × p)).
//
// Protocol: 2 warmup + 5 timed iterations; wall time via cudaEventElapsedTime.
// Memory via cudaMemGetInfo delta (PeakMemTracker).
//
// Output (stdout): CSV header + rows:
//   scale,n_cells,n_genes,density,n_pathways,wall_ms,mem_mb
//
// Node: any non-excluded GPU node (--exclude=g001,g002,g005 per §J.2).

#include <singlet/gpu/bench/harness.h>
#include <singlet/gpu/enrich/decoupler_ulm.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace singlet::gpu;

// ---------------------------------------------------------------------------
// upload_synthetic_csc — build a CSC sparse matrix on device from host vecs.
// Returns a PzDeviceMatrix populated from a SyntheticMatrix.
// Uses core::DeviceMemory<T> throughout (Rule 5 — no raw cudaMalloc).
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
// make_weight_matrix_host — dense W (n_genes × n_pathways, col-major) fp32.
// Values drawn from a simple LCG uniform in [-1, 1] with given seed.
// Matches the scipy rng(42).standard_normal pattern in the Python ref
// (same relative distribution; seeded for reproducibility — Rule 16).
// ---------------------------------------------------------------------------
static std::vector<float> make_weight_matrix_host(int n_genes, int n_pathways,
                                                   uint32_t seed = 42)
{
    std::vector<float> W(static_cast<size_t>(n_genes) * n_pathways);
    uint64_t rng = static_cast<uint64_t>(seed) ^ 0xCAFEBABE12345678ull;
    auto next_rng = [&]() -> uint64_t {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return rng;
    };
    for (auto& v : W) {
        // Map [0, 2^32) → [-1, 1].
        uint32_t u = static_cast<uint32_t>(next_rng() & 0xFFFFFFFFull);
        v = (static_cast<float>(u) / 2147483648.0f) - 1.0f;
    }
    return W;
}

// ---------------------------------------------------------------------------
// bench_ulm — 2 warmup + 5 timed iterations of ulm().
// Returns median wall_ms and peak mem_mb.
// ---------------------------------------------------------------------------
static void bench_ulm(
    const io::PzDeviceMatrix& pzmat,
    const float*              d_W,
    int                       n_pathways,
    cudaStream_t              stream,
    double&                   out_wall_ms,
    double&                   out_mem_mb)
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
            enrich::UlmConfig cfg{};
            auto result = enrich::ulm(pzmat, d_W, n_pathways, cfg, stream);
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
        std::puts("NO GPU — skipping Cycle 164 decoupler_ulm bench.");
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

        // Build dense W (n_genes × n_pathways, col-major) on host, upload.
        std::vector<float> h_W = make_weight_matrix_host(s.n_genes, n_pathways, /*seed=*/42);
        const size_t W_bytes = static_cast<size_t>(s.n_genes) * n_pathways * sizeof(float);
        core::DeviceMemory<float> d_W_buf(static_cast<size_t>(s.n_genes) * n_pathways);
        cudaMemcpyAsync(d_W_buf.get(), h_W.data(), W_bytes, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        std::printf("[bench] Timing %s method=ulm...\n", s.name);
        std::fflush(stdout);

        double wall_ms = 0.0, mem_mb = 0.0;
        bench_ulm(pzmat, d_W_buf.get(), n_pathways, stream, wall_ms, mem_mb);

        // CSV row.
        std::printf("%s,%d,%d,%.2f,%d,%.3f,%.1f\n",
                    s.name, s.n_cells, s.n_genes, s.density, n_pathways,
                    wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "enrich/decoupler_ulm";
        row.scale         = std::string(s.name);
        row.impl          = "singlet-gpu";
        row.wall_ms       = wall_ms;
        row.mem_mb        = mem_mb;
        row.cells_per_sec = bench::throughput(static_cast<int64_t>(s.n_cells), wall_ms);
        row.commit        = bench::git_short_sha();
        bench::log_row(row);
    }

    cudaStreamDestroy(stream);
    std::printf("[bench] Cycle 164 decoupler_ulm perf bench complete.\n");
    return 0;
}
