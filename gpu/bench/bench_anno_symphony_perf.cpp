// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_anno_symphony_perf.cpp
//
// CYCLE-178 Phase E — anno/symphony standalone kernel benchmark.
//
// Benchmarks anno::symphony_predict() (include/singlet-gpu/anno/symphony.h)
// against a manual NumPy CPU baseline (no Python Symphony pkg dep) at two
// medium scales:
//
//   Scale "10k":  n_cells=10000, n_features=50 PCs, n_classes=20, n_centroids=20.
//   Scale "30k":  n_cells=30000, n_features=50 PCs, n_classes=20, n_centroids=20.
//
// n_centroids=20 matches celltypist's 20 classes for cross-comparability.
//
// GPU kernel: 5 sequential device passes, no D2H in hot path.
//   Pass 1 — Standardize:  Z_std[g,c] = (Z[g,c] - mu[g]) / max(sigma[g], eps_sigma)
//             One element-wise kernel (n_features × n_cells threads).
//   Pass 2 — PCA project:  X_pca = W_pca^T · Z_std  (cuBLAS Sgemm).
//   Pass 3 — Distance:     ||C_ref||², ||X_pca||², Sgemm cross, combine kernel.
//   Pass 4 — Soft assign:  1/max(d,eps) per element, warp-shuffle L1 normalize.
//   Pass 5 — Label transfer: label_prob = P_label^T · s_soft (Sgemm) + argmax.
// Reference: Kang JB, Nathan A, Weinand K et al. (2021) Nat Commun 12:5890.
//
// Protocol:
//   Z (n_features × n_cells), W_pca (n_features × k_pcs),
//   C_ref (n_centroids × k_pcs), P_label (n_centroids × n_classes),
//   mu (n_features), sigma (n_features):
//     all synthetic fp32, xorshift64 seed=42, Gaussian-scale values.
//   symphony_predict() — 2 warmup + 5 timed (cudaEvent).
//   Memory — bench::PeakMemTracker (cudaMemGetInfo delta).
//
// Output (stdout): CSV header + rows:
//   scale,n_cells,n_features,k_pcs,n_centroids,n_classes,wall_ms,mem_mb
//
// §J.7 prediction (4-axis §J.7 formula):
//   SOTA structure: manual NumPy, vectorized standardization + Sgemm-backed PCA
//     projection + per-cell distance loop + softmax + argmax.  No per-batch
//     Python loops; standardize + PCA are tight numpy.  Distance+softmax are
//     also vectorized numpy (broadcasting).  → python_overhead_multiplier = 1×
//     (BLAS-tight with low Python overhead).
//   No dense intermediates >100 MB at 30k × 50 → memory_bandwidth_advantage = 1×.
//   GPU compute: 5 passes vs celltypist's 3 → medium/heavy (extra Sgemm cross +
//     combine + L1-norm passes). GPU_per_cell_ms slightly higher than celltypist.
//   Prediction: 30-100× class (similar to celltypist 50× but slightly lower
//     since GPU has 2 more passes and SOTA is vectorized numpy without per-batch
//     Python overhead).
//
// Node: any non-excluded GPU node (--exclude=g001,g002,g005 per §J.2).

#include <singlet_gpu/bench/harness.h>
#include <singlet-gpu/anno/symphony.h>
#include <singlet-gpu/core/types.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

using namespace singlet_gpu;

// ---------------------------------------------------------------------------
// make_synth_matrix — synthetic fp32 col-major matrix (nrows x ncols) on device.
// Xorshift64 seed=42, values in [-1, 1].
// ---------------------------------------------------------------------------
static core::DeviceMemory<float> make_synth_matrix(int nrows, int ncols,
                                                    uint64_t& rng,
                                                    cudaStream_t stream)
{
    const size_t n = static_cast<size_t>(nrows) * static_cast<size_t>(ncols);
    std::vector<float> h(n);
    auto xorshift = [&]() -> float {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return static_cast<float>(static_cast<int64_t>(rng & 0xFFFFFFFFull) -
                                  0x7FFFFFFFll) * (1.0f / 2147483647.0f);
    };
    for (size_t i = 0; i < n; ++i) h[i] = xorshift();

    core::DeviceMemory<float> d(n);
    cudaMemcpyAsync(d.get(), h.data(), n * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return d;
}

// Positive-only vector for sigma (guard against sigma=0 in standardization).
static core::DeviceMemory<float> make_synth_positive(int n,
                                                      uint64_t& rng,
                                                      cudaStream_t stream)
{
    std::vector<float> h(static_cast<size_t>(n));
    auto xorshift = [&]() -> float {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        // Map to (0, 2] via abs.
        float v = static_cast<float>(static_cast<int64_t>(rng & 0xFFFFFFFFull) -
                                     0x7FFFFFFFll) * (1.0f / 2147483647.0f);
        return v < 0.f ? -v + 0.01f : v + 0.01f;
    };
    for (int i = 0; i < n; ++i) h[static_cast<size_t>(i)] = xorshift();

    core::DeviceMemory<float> d(static_cast<size_t>(n));
    cudaMemcpyAsync(d.get(), h.data(), static_cast<size_t>(n) * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return d;
}

// Row-normalize a col-major (k_clusters x n_classes) matrix on host so rows
// sum to 1 (P_label must be a valid probability distribution for symphony).
static core::DeviceMemory<float> make_synth_plabel(int k_clusters, int n_classes,
                                                    uint64_t& rng,
                                                    cudaStream_t stream)
{
    const size_t total = static_cast<size_t>(k_clusters) * static_cast<size_t>(n_classes);
    std::vector<float> h(total);
    auto xorshift_pos = [&]() -> float {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        float v = static_cast<float>(rng & 0xFFFFFFFFull) * (1.0f / 4294967295.0f);
        return v + 1e-4f;  // strictly positive
    };
    for (size_t i = 0; i < total; ++i) h[i] = xorshift_pos();

    // Row-normalize: P_label is (k_clusters x n_classes) col-major.
    // Row k spans elements h[k + p*k_clusters] for p in [0, n_classes).
    for (int k = 0; k < k_clusters; ++k) {
        float row_sum = 0.f;
        for (int p = 0; p < n_classes; ++p)
            row_sum += h[static_cast<size_t>(k) + static_cast<size_t>(p) * static_cast<size_t>(k_clusters)];
        for (int p = 0; p < n_classes; ++p)
            h[static_cast<size_t>(k) + static_cast<size_t>(p) * static_cast<size_t>(k_clusters)] /= row_sum;
    }

    core::DeviceMemory<float> d(total);
    cudaMemcpyAsync(d.get(), h.data(), total * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return d;
}

// ---------------------------------------------------------------------------
// bench_symphony — 2 warmup + 5 timed iterations of symphony_predict().
// Returns median wall_ms and peak mem_mb via out params.
// ---------------------------------------------------------------------------
static void bench_symphony(
    const float*  d_Z,          // n_features x n_cells col-major
    int           n_features,
    int           n_cells,
    const float*  d_W_pca,      // n_features x k_pcs col-major
    int           k_pcs,
    const float*  d_C_ref,      // n_centroids x k_pcs col-major
    int           n_centroids,
    const float*  d_P_label,    // n_centroids x n_classes col-major
    int           n_classes,
    const float*  d_mu,         // n_features
    const float*  d_sigma,      // n_features (positive)
    cudaStream_t  stream,
    double&       out_wall_ms,
    double&       out_mem_mb)
{
    constexpr int WARMUP = 2;
    constexpr int TIMED  = 5;
    double samples[TIMED]{};
    double mem_samples[TIMED]{};

    bench::BenchTimer     timer;
    bench::PeakMemTracker mem;

    anno::SymphonyConfig cfg{};

    for (int iter = 0; iter < WARMUP + TIMED; ++iter) {
        const bool is_timed = (iter >= WARMUP);
        if (is_timed) mem.sample_before();

        timer.start(stream);

        {
            auto result = anno::symphony_predict(
                d_Z, n_features, n_cells,
                d_W_pca, k_pcs,
                d_C_ref, n_centroids,
                d_P_label, n_classes,
                d_mu, d_sigma,
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
        std::puts("NO GPU — skipping Cycle 178 anno/symphony bench.");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // CSV header.
    std::printf("scale,n_cells,n_features,k_pcs,n_centroids,n_classes,wall_ms,mem_mb\n");
    std::fflush(stdout);

    // Scales: 10k / 30k cells, 50 PCs, n_centroids=20, n_classes=20.
    struct Scale {
        const char* name;
        int n_cells;
        int n_features;
        int k_pcs;
        int n_centroids;
        int n_classes;
    };
    const Scale scales[] = {
        { "10k", 10000, 50, 50, 20, 20 },
        { "30k", 30000, 50, 50, 20, 20 },
    };

    for (const auto& s : scales) {
        std::printf("[bench] Synthesizing %s: %d features × %d cells, "
                    "k_pcs=%d, n_centroids=%d, n_classes=%d...\n",
                    s.name, s.n_features, s.n_cells,
                    s.k_pcs, s.n_centroids, s.n_classes);
        std::fflush(stdout);

        uint64_t rng = 42ULL ^ 0xDEADBEEFCAFEBABEull;

        // Query expression matrix: n_features x n_cells col-major.
        core::DeviceMemory<float> d_Z =
            make_synth_matrix(s.n_features, s.n_cells, rng, stream);

        // PCA loadings: n_features x k_pcs col-major.
        core::DeviceMemory<float> d_W_pca =
            make_synth_matrix(s.n_features, s.k_pcs, rng, stream);

        // Reference centroids: n_centroids x k_pcs col-major.
        core::DeviceMemory<float> d_C_ref =
            make_synth_matrix(s.n_centroids, s.k_pcs, rng, stream);

        // Label distribution: n_centroids x n_classes col-major, rows sum to 1.
        core::DeviceMemory<float> d_P_label =
            make_synth_plabel(s.n_centroids, s.n_classes, rng, stream);

        // Reference mean: n_features.
        core::DeviceMemory<float> d_mu =
            make_synth_matrix(s.n_features, 1, rng, stream);

        // Reference std: n_features (positive).
        core::DeviceMemory<float> d_sigma =
            make_synth_positive(s.n_features, rng, stream);

        std::printf("[bench] %s: Timing anno::symphony_predict "
                    "(n_features=%d, n_cells=%d, k_pcs=%d, "
                    "n_centroids=%d, n_classes=%d)...\n",
                    s.name, s.n_features, s.n_cells, s.k_pcs,
                    s.n_centroids, s.n_classes);
        std::fflush(stdout);

        double wall_ms = 0.0, mem_mb = 0.0;
        bench_symphony(
            d_Z.get(), s.n_features, s.n_cells,
            d_W_pca.get(), s.k_pcs,
            d_C_ref.get(), s.n_centroids,
            d_P_label.get(), s.n_classes,
            d_mu.get(), d_sigma.get(),
            stream, wall_ms, mem_mb);

        // CSV row.
        std::printf("%s,%d,%d,%d,%d,%d,%.3f,%.1f\n",
                    s.name, s.n_cells, s.n_features, s.k_pcs,
                    s.n_centroids, s.n_classes,
                    wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "anno/symphony";
        row.scale         = std::string(s.name);
        row.impl          = "singlet-gpu";
        row.wall_ms       = wall_ms;
        row.mem_mb        = mem_mb;
        row.cells_per_sec = bench::throughput(static_cast<int64_t>(s.n_cells),
                                               wall_ms);
        row.commit        = bench::git_short_sha();
        bench::log_row(row);
    }

    cudaStreamDestroy(stream);
    std::printf("[bench] Cycle 178 anno/symphony perf bench complete.\n");
    return 0;
}
