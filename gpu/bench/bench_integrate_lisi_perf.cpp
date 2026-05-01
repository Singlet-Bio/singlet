// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_integrate_lisi_perf.cpp
//
// CYCLE-169 Phase E — integrate/lisi standalone kernel benchmark.
//
// Benchmarks integrate::lisi() (from include/singlet-gpu/integrate/lisi.h)
// against a manual numpy CPU baseline at two medium scales:
//
//   Scale "10k":  n_cells=10000, n_pcs=50, k=10, n_batches=4
//   Scale "30k":  n_cells=30000, n_pcs=50, k=10, n_batches=4
//
// GPU kernel: one block per cell; shared-memory histogram of n_labels ints;
//   accumulates label counts for k neighbors via atomicAdd or serial scan;
//   thread 0 computes D = sum(p_l^2) and writes LISI = 1/D.
// Reference: Korsunsky I et al. (2019) "Harmony" Nat Methods 16:1289-1296.
//
// Protocol:
//   kNN build — single untimed warmup (Exact, KnnBackend::Exact, L2, k=10).
//   LISI timing — 2 warmup + 5 timed via cudaEvent (bench::BenchTimer).
//   Memory       — bench::PeakMemTracker (cudaMemGetInfo delta).
//   Batch labels — round-robin: label[c] = c % n_batches (device-side).
//
// Output (stdout): CSV header + rows:
//   scale,n_cells,n_pcs,k,n_batches,wall_ms,mem_mb
//
// Node: any non-excluded GPU node (--exclude=g001,g002,g005 per §J.2).

#include <singlet_gpu/bench/harness.h>
#include <singlet-gpu/graph/knn.h>
#include <singlet-gpu/integrate/lisi.h>
#include <singlet-gpu/core/types.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

using namespace singlet_gpu;

// ---------------------------------------------------------------------------
// make_synthetic_embedding — random fp32 row-major embedding (n × d).
// LCG seed=42 for reproducibility without <random>.
// ---------------------------------------------------------------------------
static core::DeviceDense make_synthetic_embedding(int n, int d,
                                                   cudaStream_t stream)
{
    const size_t total = static_cast<size_t>(n) * d;
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

    core::DeviceDense emb(n, d);
    emb.data = std::move(dev_data);
    return emb;
}

// ---------------------------------------------------------------------------
// make_round_robin_labels — device int[n] with label[c] = c % n_batches.
// Filled on the CPU and uploaded once (labels are constant across all LISI
// bench iterations — only the LISI kernel execution is timed).
// ---------------------------------------------------------------------------
static core::DeviceMemory<int> make_round_robin_labels(int n, int n_batches,
                                                        cudaStream_t stream)
{
    std::vector<int> h_label(n);
    for (int c = 0; c < n; ++c) h_label[c] = c % n_batches;

    core::DeviceMemory<int> d_label(static_cast<size_t>(n));
    cudaMemcpyAsync(d_label.get(), h_label.data(), n * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return d_label;
}

// ---------------------------------------------------------------------------
// bench_lisi — 2 warmup + 5 timed iterations of lisi().
// kNN result and labels are pre-built (untimed).
// Returns median wall_ms and peak mem_mb via out params.
// ---------------------------------------------------------------------------
static void bench_lisi(
    const graph::KnnResult&      knn,
    const core::DeviceMemory<int>& d_label,
    int                           n_batches,
    cudaStream_t                  stream,
    double&                       out_wall_ms,
    double&                       out_mem_mb)
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
            integrate::LisiConfig cfg{};
            auto result = integrate::lisi(knn, d_label.get(), n_batches,
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
        std::puts("NO GPU — skipping Cycle 169 integrate/lisi bench.");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // CSV header.
    std::printf("scale,n_cells,n_pcs,k,n_batches,wall_ms,mem_mb\n");
    std::fflush(stdout);

    // Scales.
    struct Scale { const char* name; int n_cells; int n_pcs; };
    const Scale scales[] = {
        { "10k", 10000, 50 },
        { "30k", 30000, 50 },
    };
    const int K         = 10;
    const int N_BATCHES = 4;

    for (const auto& s : scales) {
        std::printf("[bench] Synthesizing %s: %d cells × %d PCs...\n",
                    s.name, s.n_cells, s.n_pcs);
        std::fflush(stdout);

        // Build embedding (untimed).
        core::DeviceDense emb = make_synthetic_embedding(s.n_cells, s.n_pcs,
                                                          stream);

        // Build kNN graph (untimed warmup — not part of LISI timing).
        std::printf("[bench] Building kNN (k=%d, Exact, untimed)...\n", K);
        std::fflush(stdout);

        graph::KnnConfig knn_cfg;
        knn_cfg.k       = K;
        knn_cfg.backend = graph::KnnBackend::Exact;
        knn_cfg.metric  = graph::DistanceMetric::L2;
        graph::KnnResult knn = graph::compute_knn(emb, knn_cfg, stream);

        // Build round-robin batch labels (untimed).
        core::DeviceMemory<int> d_label =
            make_round_robin_labels(s.n_cells, N_BATCHES, stream);

        std::printf("[bench] Timing %s lisi() (n_batches=%d k=%d)...\n",
                    s.name, N_BATCHES, K);
        std::fflush(stdout);

        double wall_ms = 0.0, mem_mb = 0.0;
        bench_lisi(knn, d_label, N_BATCHES, stream, wall_ms, mem_mb);

        // CSV row.
        std::printf("%s,%d,%d,%d,%d,%.3f,%.1f\n",
                    s.name, s.n_cells, s.n_pcs, K, N_BATCHES,
                    wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "integrate/lisi";
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
    std::printf("[bench] Cycle 169 integrate/lisi perf bench complete.\n");
    return 0;
}
