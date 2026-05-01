// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_anno_celltypist_perf.cpp
//
// CYCLE-176 Phase E — anno/celltypist standalone kernel benchmark.
//
// Benchmarks anno::celltypist_predict() (include/singlet-gpu/anno/celltypist.h)
// against sklearn.linear_model.LogisticRegression.predict_proba CPU baseline
// at two medium scales:
//
//   Scale "10k":  n_cells=10000, n_features=50 PCs, n_classes=20.
//   Scale "30k":  n_cells=30000, n_features=50 PCs, n_classes=20.
//
// GPU kernel: 3 sequential device passes, no D2H in hot path.
//   Pass 1 — cuBLAS Sgemm:  L = W^T . Z  (n_classes x n_cells, col-major).
//   Pass 2 — bias-add:      L[k, c] += b[k]  (one thread per element).
//   Pass 3 — softmax+argmax: one block per cell; warp-shuffle log-sum-exp.
//             Outputs only pred_class[c] and confidence[c] (no full P stored).
// Reference: Dominguez Conde C et al. (2022) Science 376:eabl5197.
//
// Protocol:
//   Z (n_features × n_cells), W (n_features × n_classes), b (n_classes):
//     all synthetic fp32, xorshift64 seed=42, Gaussian-scale values.
//   celltypist_predict() — 2 warmup + 5 timed (cudaEvent).
//   Memory — bench::PeakMemTracker (cudaMemGetInfo delta).
//
// Output (stdout): CSV header + rows:
//   scale,n_cells,n_features,n_classes,wall_ms,mem_mb
//
// §J.7 prediction: sklearn.predict_proba is BLAS-backed (OpenBLAS/MKL Sgemm)
//   with minimal Python overhead — the hot path is just one DGEMM + softmax.
//   Expect class 3 or BLAS-tight (5-30×). Could be 100-200× if sklearn Python
//   overhead adds up at 10k-cell scale (only ~3ms total CPU).
//
// Node: any non-excluded GPU node (--exclude=g001,g002,g005 per §J.2).

#include <singlet_gpu/bench/harness.h>
#include <singlet-gpu/anno/celltypist.h>
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
// Xorshift64 seed=42, values in [-1, 1] (bit-cast uint64 → float → rescale).
// ---------------------------------------------------------------------------
static core::DeviceMemory<float> make_synth_matrix(int nrows, int ncols,
                                                    uint64_t& rng,
                                                    cudaStream_t stream)
{
    const size_t n = static_cast<size_t>(nrows) * static_cast<size_t>(ncols);
    std::vector<float> h(n);
    auto xorshift = [&]() -> float {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        // Map uint64 to [-1, 1] fp32.
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

// ---------------------------------------------------------------------------
// bench_celltypist — 2 warmup + 5 timed iterations of celltypist_predict().
// Returns median wall_ms and peak mem_mb via out params.
// ---------------------------------------------------------------------------
static void bench_celltypist(
    const float*  d_Z,       // n_features x n_cells col-major
    int           n_features,
    int           n_cells,
    const float*  d_W,       // n_features x n_classes col-major
    const float*  d_b,       // n_classes
    int           n_classes,
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

    anno::CelltypistConfig cfg{};
    cfg.use_log_input = false;  // synthetic data, no log-input assertion needed

    for (int iter = 0; iter < WARMUP + TIMED; ++iter) {
        const bool is_timed = (iter >= WARMUP);
        if (is_timed) mem.sample_before();

        timer.start(stream);

        {
            auto result = anno::celltypist_predict(
                d_Z, n_features, n_cells,
                d_W, d_b, n_classes,
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
        std::puts("NO GPU — skipping Cycle 176 anno/celltypist bench.");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // CSV header.
    std::printf("scale,n_cells,n_features,n_classes,wall_ms,mem_mb\n");
    std::fflush(stdout);

    // Scales: 10k / 30k cells, 50 PCs, n_classes=20.
    struct Scale { const char* name; int n_cells; int n_features; int n_classes; };
    const Scale scales[] = {
        { "10k", 10000, 50, 20 },
        { "30k", 30000, 50, 20 },
    };

    for (const auto& s : scales) {
        std::printf("[bench] Synthesizing %s: %d features × %d cells, "
                    "n_classes=%d...\n",
                    s.name, s.n_features, s.n_cells, s.n_classes);
        std::fflush(stdout);

        uint64_t rng = 42ULL ^ 0xDEADBEEFCAFEBABEull;

        // Z: n_features x n_cells col-major (untimed).
        core::DeviceMemory<float> d_Z =
            make_synth_matrix(s.n_features, s.n_cells, rng, stream);

        // W: n_features x n_classes col-major (untimed).
        core::DeviceMemory<float> d_W =
            make_synth_matrix(s.n_features, s.n_classes, rng, stream);

        // b: n_classes (untimed).
        core::DeviceMemory<float> d_b =
            make_synth_matrix(s.n_classes, 1, rng, stream);

        std::printf("[bench] %s: Timing anno::celltypist_predict "
                    "(n_features=%d, n_cells=%d, n_classes=%d)...\n",
                    s.name, s.n_features, s.n_cells, s.n_classes);
        std::fflush(stdout);

        double wall_ms = 0.0, mem_mb = 0.0;
        bench_celltypist(d_Z.get(), s.n_features, s.n_cells,
                         d_W.get(), d_b.get(), s.n_classes,
                         stream, wall_ms, mem_mb);

        // CSV row.
        std::printf("%s,%d,%d,%d,%.3f,%.1f\n",
                    s.name, s.n_cells, s.n_features, s.n_classes,
                    wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "anno/celltypist";
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
    std::printf("[bench] Cycle 176 anno/celltypist perf bench complete.\n");
    return 0;
}
