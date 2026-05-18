// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_embed_diffmap_perf.cpp
//
// CYCLE-159 Phase E — embed/diffmap standalone kernel benchmark.
//
// Benchmarks diffmap() (include/singlet/gpu/embed/diffmap.h)
// against synthetic DENSE PCA-reduced embeddings at two medium scales:
//
//   Scale "10k":  n_cells=10000, n_pcs=50, density=1.0 (dense), k=10
//   Scale "30k":  n_cells=30000, n_pcs=50, density=1.0 (dense), k=10
//
// Protocol:
//   - kNN is built ONCE per scale (Exact backend, untimed warmup).
//   - diffmap() is called 2 warmup + 5 timed iterations via cudaEventElapsedTime.
//   - Memory via cudaMemGetInfo delta (PeakMemTracker).
//   - ONLY diffmap() is timed — not the kNN build (kNN is pre-built input).
//
// Config: DiffmapConfig{ n_comps=5, t=1 } (scanpy.tl.diffmap defaults).
//
// NOTE: diffmap builds a dense n×n transition matrix in Pass 2. At n=30k that
// is 30000^2 * 4B = 3.6 GB — ensure the target GPU has >= 8 GB VRAM or the
// built-in memory guard will throw.
//
// Output (stdout): CSV header + rows:
//   scale,n_cells,n_pcs,k,n_components,wall_ms,mem_mb
//
// Reference: Coifman & Lafon 2005 (PNAS 102:7426-7431)
//            scanpy.tl.diffmap (Wolf et al. 2018)
// Node: g050 (H100 sm_90-compatible, 80 GB HBM).

#include <singlet/gpu/bench/harness.h>
#include <singlet/gpu/embed/diffmap.h>
#include <singlet/gpu/graph/knn.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

using namespace singlet::gpu;

// ---------------------------------------------------------------------------
// make_synthetic_dense_embedding
//
// Generate a synthetic dense PCA embedding (n_cells × n_pcs) on host using a
// simple LCG PRNG, then upload to device.  Diffusion map operates on PCA output
// which is always dense (no sparsity), so density=1.0 here.
//
// Returns core::DeviceDense (row-major, n_cells rows × n_pcs cols).
// Using DeviceMemory<float> + DeviceDense::wrap avoids raw cudaMalloc (Rule 5).
// ---------------------------------------------------------------------------
static core::DeviceDense make_synthetic_dense_embedding(
    int n_cells, int n_pcs, uint32_t seed, cudaStream_t stream)
{
    const size_t total = static_cast<size_t>(n_cells) * n_pcs;

    // Generate host-side data using XorShift LCG for reproducibility.
    std::vector<float> host_data(total);
    uint64_t rng = static_cast<uint64_t>(seed) ^ 0xDEADBEEFCAFEBABEull;
    auto next_rng = [&]() -> uint64_t {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return rng;
    };
    for (size_t i = 0; i < total; ++i) {
        // Map [0, UINT64_MAX] → [-3, 3] to simulate standardized PCA coordinates.
        double u = static_cast<double>(next_rng()) / static_cast<double>(UINT64_MAX);
        host_data[i] = static_cast<float>((u - 0.5) * 6.0);
    }

    // Upload to device via RAII DeviceMemory (Rule 5).
    core::DeviceMemory<float> d_data(total);
    cudaMemcpyAsync(d_data.get(), host_data.data(),
                    total * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Wrap into DeviceDense (row-major: rows=n_cells, cols=n_pcs).
    core::DeviceDense emb;
    emb.rows = n_cells;
    emb.cols = n_pcs;
    emb.data = std::move(d_data);
    return emb;
}

// ---------------------------------------------------------------------------
// bench_scale
//
// Runs 2 warmup + 5 timed iterations of diffmap() on a pre-built KnnResult.
// Only the diffmap() call is timed — kNN must be pre-built by the caller.
// Returns median wall_ms and peak mem_mb.
// ---------------------------------------------------------------------------
static void bench_scale(
    const graph::KnnResult&  knn,
    const embed::DiffmapConfig& cfg,
    cudaStream_t stream,
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
        // Call the CYCLE-150 diffmap kernel — result discarded (wall-time only).
        // diffmap() calls cudaStreamSynchronize internally (Pass 5 cuSOLVER sync).
        auto result = embed::diffmap(knn, cfg, stream);
        // Ensure all GPU work is complete before stopping the timer.
        cudaStreamSynchronize(stream);
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
        std::puts("NO GPU — skipping Cycle 159 diffmap bench.");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // CSV header.
    std::printf("scale,n_cells,n_pcs,k,n_components,wall_ms,mem_mb\n");
    std::fflush(stdout);

    // Config: match scanpy.tl.diffmap defaults (n_comps=5, t=1).
    embed::DiffmapConfig cfg;
    cfg.n_comps = 5;
    cfg.t       = 1;

    // kNN config: Exact backend (n < 50k), k=10 neighbors, L2 distance.
    graph::KnnConfig knn_cfg;
    knn_cfg.k       = 10;
    knn_cfg.backend = graph::KnnBackend::Exact;
    knn_cfg.metric  = graph::DistanceMetric::L2;

    // Scales: 10k and 30k cells, 50 PCs (PCA-reduced dense embeddings).
    struct Scale { const char* name; int n_cells; int n_pcs; };
    const Scale scales[] = {
        { "10k",  10000, 50 },
        { "30k",  30000, 50 },
    };

    for (const auto& s : scales) {
        std::printf("[bench] Synthesizing %s: %d cells × %d PCs (dense)...\n",
                    s.name, s.n_cells, s.n_pcs);
        std::fflush(stdout);

        // Build synthetic dense embedding (row-major, n_cells × n_pcs).
        core::DeviceDense emb = make_synthetic_dense_embedding(
            s.n_cells, s.n_pcs, /*seed=*/42, stream);

        // Build kNN graph ONCE (untimed warmup — not the kernel under test).
        std::printf("[bench] Building kNN k=%d Exact for %s (untimed)...\n",
                    knn_cfg.k, s.name);
        std::fflush(stdout);
        graph::KnnResult knn = graph::compute_knn(emb, knn_cfg, stream);
        cudaStreamSynchronize(stream);

        // Run 2 warmup + 5 timed iterations of diffmap only.
        std::printf("[bench] Timing diffmap n_comps=%d t=%d for %s...\n",
                    cfg.n_comps, cfg.t, s.name);
        std::fflush(stdout);

        double wall_ms = 0.0, mem_mb = 0.0;
        bench_scale(knn, cfg, stream, wall_ms, mem_mb);

        // CSV row.
        std::printf("%s,%d,%d,%d,%d,%.3f,%.1f\n",
                    s.name, s.n_cells, s.n_pcs, knn_cfg.k, cfg.n_comps,
                    wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "embed/diffmap";
        row.scale         = s.name;
        row.impl          = "singlet-gpu";
        row.wall_ms       = wall_ms;
        row.mem_mb        = mem_mb;
        row.cells_per_sec = bench::throughput(static_cast<int64_t>(s.n_cells), wall_ms);
        row.commit        = bench::git_short_sha();
        bench::log_row(row);
    }

    cudaStreamDestroy(stream);
    std::printf("[bench] Cycle 159 diffmap perf bench complete.\n");
    return 0;
}
