// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_embed_dpt_perf.cpp
//
// CYCLE-161 Phase E — embed/dpt standalone kernel benchmark.
//
// §J.6 AT-RISK audit (CYCLE-160): dpt uses the same dense n×n Gaussian
// similarity W + cusolverDnSsyevd eigendecomp as diffmap (CYCLE-159). At
// n=10k diffmap was 14× slower than scanpy ARPACK; at n=30k it crashed.
// Hypothesis: dpt at n=10k will be 5-20× slower than sc.tl.dpt (sparse
// ARPACK via the pre-computed diffmap eigenvectors).
//
// Benchmarks dpt() (include/singlet-gpu/embed/dpt.h) against synthetic
// DENSE PCA-reduced embeddings at one medium scale (30k skipped — same
// cuSOLVER kernel pattern that crashed diffmap in CYCLE-159):
//
//   Scale "10k": n_cells=10000, n_pcs=50, density=1.0 (dense), k=10
//
// Protocol:
//   - kNN is built ONCE per scale (Exact backend, untimed warmup).
//   - dpt() is called 2 warmup + 5 timed iterations via cudaEventElapsedTime.
//   - Memory via cudaMemGetInfo delta (PeakMemTracker).
//   - ONLY dpt() is timed — not kNN build.
//
// Config: DptConfig{ root_cell=0, n_eigenvecs=15, eps_lambda=1e-6 }
//   (mirrors sc.tl.dpt n_comps=15 and iroot=0 defaults).
//
// NOTE: dpt builds a dense n×n transition matrix in Pass 2. At n=10k that
// is 10000^2 * 4B = 400 MB — well within GPU VRAM limits. At n=30k that
// would be 3.6 GB + cuSOLVER workspace; skipped pending Lanczos v1 work.
//
// Output (stdout): CSV header + rows:
//   scale,n_cells,n_pcs,k,n_eigenvecs,wall_ms,mem_mb
//
// Reference: Haghverdi et al. 2016 (Nat Methods 13:845-848)
//            scanpy.tl.dpt (Wolf et al. 2018)
// Node: g050 (H100 sm_90-compatible, 80 GB HBM).

#include <singlet_gpu/bench/harness.h>
#include <singlet-gpu/embed/dpt.h>
#include <singlet-gpu/graph/knn.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

using namespace singlet_gpu;

// ---------------------------------------------------------------------------
// make_synthetic_dense_embedding
//
// Generate a synthetic dense PCA embedding (n_cells × n_pcs) on host using a
// simple XorShift LCG PRNG, then upload to device.  DPT operates on PCA output
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
// Runs 2 warmup + 5 timed iterations of dpt() on a pre-built KnnResult.
// Only the dpt() call is timed — kNN must be pre-built by the caller.
// Returns median wall_ms and peak mem_mb.
// ---------------------------------------------------------------------------
static void bench_scale(
    const graph::KnnResult&  knn,
    const embed::DptConfig&  cfg,
    cudaStream_t             stream,
    double&                  out_wall_ms,
    double&                  out_mem_mb)
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
        // Call the CYCLE-142 dpt kernel — result discarded (wall-time only).
        // dpt() calls cudaStreamSynchronize internally (Pass 5 cuSOLVER sync).
        auto result = embed::dpt(knn, cfg, stream);
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
        std::puts("NO GPU — skipping Cycle 161 dpt bench.");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // CSV header.
    std::printf("scale,n_cells,n_pcs,k,n_eigenvecs,wall_ms,mem_mb\n");
    std::fflush(stdout);

    // Config: n_eigenvecs=15 matches sc.tl.dpt n_comps=15; root_cell=0 matches
    // adata.uns['iroot'] = 0 in the scanpy baseline.
    embed::DptConfig cfg;
    cfg.root_cell   = 0;
    cfg.n_eigenvecs = 15;
    cfg.eps_lambda  = 1e-6f;

    // kNN config: Exact backend (n < 50k), k=10 neighbors, L2 distance.
    graph::KnnConfig knn_cfg;
    knn_cfg.k       = 10;
    knn_cfg.backend = graph::KnnBackend::Exact;
    knn_cfg.metric  = graph::DistanceMetric::L2;

    // Scales: 10k cells, 50 PCs (dense PCA embedding).
    // 30k is SKIPPED — same dense cuSOLVER pattern that crashed diffmap
    // (CYCLE-159): 30k^2 * 4B = 3.6 GB + cuSOLVER workspace. If 10k passes,
    // retry 30k in a dedicated follow-up cycle once Lanczos v1 is available.
    struct Scale { const char* name; int n_cells; int n_pcs; };
    const Scale scales[] = {
        { "10k",  10000, 50 },
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

        // Run 2 warmup + 5 timed iterations of dpt only.
        std::printf("[bench] Timing dpt n_eigenvecs=%d root_cell=%d for %s...\n",
                    cfg.n_eigenvecs, cfg.root_cell, s.name);
        std::fflush(stdout);

        double wall_ms = 0.0, mem_mb = 0.0;
        try {
            bench_scale(knn, cfg, stream, wall_ms, mem_mb);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[bench] ERROR during dpt for %s: %s\n",
                         s.name, e.what());
            std::printf("%s,%d,%d,%d,%d,-1.0,-1.0\n",
                        s.name, s.n_cells, s.n_pcs, knn_cfg.k, cfg.n_eigenvecs);
            std::fflush(stdout);
            continue;
        }

        // CSV row.
        std::printf("%s,%d,%d,%d,%d,%.3f,%.1f\n",
                    s.name, s.n_cells, s.n_pcs, knn_cfg.k, cfg.n_eigenvecs,
                    wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "embed/dpt";
        row.scale         = s.name;
        row.impl          = "singlet-gpu";
        row.wall_ms       = wall_ms;
        row.mem_mb        = mem_mb;
        row.cells_per_sec = bench::throughput(static_cast<int64_t>(s.n_cells), wall_ms);
        row.commit        = bench::git_short_sha();
        bench::log_row(row);
    }

    cudaStreamDestroy(stream);
    std::printf("[bench] Cycle 161 dpt perf bench complete.\n");
    return 0;
}
