// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_graph_kmeans_perf.cpp
//
// CYCLE-172 Phase E — graph/kmeans standalone kernel benchmark.
//
// Benchmarks graph::kmeans() (from include/singlet/gpu/graph/kmeans.h)
// against sklearn KMeans CPU baseline at two medium scales:
//
//   Scale "10k":  n_cells=10000, n_pcs=50, k=10, max_iter=20
//   Scale "30k":  n_cells=30000, n_pcs=50, k=10, max_iter=20
//
// GPU kernel: Forgy init (host mt19937 random column copy); Lloyd loop:
//   (a) km_col_norm_kernel — ||C[:,k]||² per centroid (warp-shuffle reduction).
//   (b) cublasSgemm D[n×k] = X^T · C (cuBLAS Sgemm, n×k col-major).
//   (c) km_assign_kernel — dist[c,k] = x_norm[c] + c_norm[k] - 2·D[c,k],
//       label[c] = argmin_k  (one block per cell, warp-shuffle min).
//   (d) km_count_changes_kernel — D2H scalar change count (convergence check).
//   (e) km_scatter_kernel — atomic-scatter X[:,c] into C_new[:,label[c]].
//   (f) km_centroid_divide_kernel — C[:,k] /= count[k].
//   Final: km_inertia_kernel — sum_c ||X[:,c] - C[:,label[c]]||².
// Reference: Lloyd (1982) IEEE TIT 28:129-137.
//
// Protocol:
//   Embedding — synthetic random fp32 (n × d), LCG seed=42, col-major.
//   kmeans() — 2 warmup + 5 timed via cudaEvent (bench::BenchTimer).
//   Memory    — bench::PeakMemTracker (cudaMemGetInfo delta).
//
// Output (stdout): CSV header + rows:
//   scale,n_cells,n_pcs,k,max_iter,wall_ms,mem_mb
//
// Node: any non-excluded GPU node (--exclude=g001,g002,g005 per §J.2).

#include <singlet/gpu/bench/harness.h>
#include <singlet/gpu/graph/kmeans.h>
#include <singlet/gpu/core/types.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

using namespace singlet::gpu;

// ---------------------------------------------------------------------------
// make_synthetic_embedding — random fp32 col-major embedding (d × n).
// LCG seed=42 for reproducibility without <random>.
// kmeans() expects d_pcs × n_cells col-major (same layout as d_X in header).
// ---------------------------------------------------------------------------
static core::DeviceMemory<float> make_synthetic_embedding(int n, int d,
                                                           cudaStream_t stream)
{
    const size_t total = static_cast<size_t>(d) * static_cast<size_t>(n);
    std::vector<float> h_emb(total);

    uint64_t rng = 42ULL ^ 0xDEADBEEFCAFEBABEull;
    for (size_t i = 0; i < total; ++i) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        float u = static_cast<float>(static_cast<uint32_t>(rng >> 32))
                  / static_cast<float>(0xFFFFFFFFu);
        h_emb[i] = u * 2.0f - 1.0f;
    }

    core::DeviceMemory<float> dev_data(total);
    cudaMemcpyAsync(dev_data.get(), h_emb.data(), total * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return dev_data;
}

// ---------------------------------------------------------------------------
// bench_kmeans — 2 warmup + 5 timed iterations of graph::kmeans().
// Embedding is pre-built (untimed).
// Returns median wall_ms and peak mem_mb via out params.
// ---------------------------------------------------------------------------
static void bench_kmeans(
    const float*  d_X,
    int           d_pcs,
    int           n_cells,
    int           k_clusters,
    int           max_iter,
    cudaStream_t  stream,
    double&       out_wall_ms,
    double&       out_mem_mb)
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
            graph::KmeansConfig cfg{};
            cfg.max_iter    = max_iter;
            cfg.tol_changes = 0;
            cfg.seed        = 42ULL;
            auto result = graph::kmeans(d_X, d_pcs, n_cells, k_clusters,
                                        cfg, stream);
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
        std::puts("NO GPU — skipping Cycle 172 graph/kmeans bench.");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // CSV header.
    std::printf("scale,n_cells,n_pcs,k,max_iter,wall_ms,mem_mb\n");
    std::fflush(stdout);

    // Scales.
    struct Scale { const char* name; int n_cells; int n_pcs; int k; };
    const Scale scales[] = {
        { "10k", 10000, 50, 10 },
        { "30k", 30000, 50, 10 },
    };
    const int MAX_ITER = 20;

    for (const auto& s : scales) {
        std::printf("[bench] Synthesizing %s: %d PCs × %d cells (col-major)...\n",
                    s.name, s.n_pcs, s.n_cells);
        std::fflush(stdout);

        // Build col-major embedding d_pcs × n_cells (untimed).
        core::DeviceMemory<float> d_X =
            make_synthetic_embedding(s.n_cells, s.n_pcs, stream);

        std::printf("[bench] Timing %s kmeans() (k=%d, max_iter=%d)...\n",
                    s.name, s.k, MAX_ITER);
        std::fflush(stdout);

        double wall_ms = 0.0, mem_mb = 0.0;
        bench_kmeans(d_X.get(), s.n_pcs, s.n_cells, s.k, MAX_ITER,
                     stream, wall_ms, mem_mb);

        // CSV row.
        std::printf("%s,%d,%d,%d,%d,%.3f,%.1f\n",
                    s.name, s.n_cells, s.n_pcs, s.k, MAX_ITER,
                    wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "graph/kmeans";
        row.scale         = std::string(s.name);
        row.impl          = "singlet-gpu";
        row.wall_ms       = static_cast<double>(wall_ms);
        row.mem_mb        = static_cast<double>(mem_mb);
        row.cells_per_sec = bench::throughput(static_cast<int64_t>(s.n_cells),
                                               wall_ms);
        row.commit        = bench::git_short_sha();
        bench::log_row(row);
    }

    cudaStreamDestroy(stream);
    std::printf("[bench] Cycle 172 graph/kmeans perf bench complete.\n");
    return 0;
}
