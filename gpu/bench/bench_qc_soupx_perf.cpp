// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_qc_soupx_perf.cpp
//
// CYCLE-180 Phase E — qc/soupx standalone kernel benchmark.
//
// Benchmarks qc::soupx() (include/singlet-gpu/qc/soupx.h)
// against a manual scipy/numpy Python SoupX equivalent at two scales:
//
//   Scale "10k":  n_droplets=10000, n_genes=3000, lower=100.
//   Scale "30k":  n_droplets=30000, n_genes=3000, lower=100.
//
// Synthetic raw 10X-style sparse CSC (col = droplet, row = gene):
//   ~80% of droplets are "empty" (UMI total <= lower, drawn from Poisson(30)).
//   ~20% are "candidates" / cells (UMI total > lower, drawn from Poisson(200)+lower+1).
//   Gene probabilities follow a Zipf-like distribution (pi[g] ∝ 1/(g+1)).
//   Density ~3% per candidate droplet.
//
// GPU kernel: 5-pass:
//   Pass 1 — per-droplet UMI sum t[j] (warp/col from CSC);
//   Pass 2 — ambient scatter (atomic/nnz for empty cols) + normalize → pi[g];
//   Pass 3 — top-ambient-gene mask (one D2H of pi, host sort+threshold, H2D mask);
//   Pass 4 — per-cell rho_c (warp/col, top_amb_mask weighted);
//   Pass 5 — dense corrected output (cudaMemset + nnz-overwrite;
//             stored: max(0, x - rho_c * t_c * pi[g]);
//             implicit zeros: max(0, 0 - ...) = 0 from cudaMemset).
// Reference: Young MD, Behjati S (2020) GigaScience 9:giaa151.
//
// Protocol:
//   Synthetic PzDeviceMatrix constructed on device via host→device copy.
//   soupx() — 2 warmup + 5 timed (cudaEvent).
//   Memory — bench::PeakMemTracker (cudaMemGetInfo delta).
//
// Output (stdout): CSV header + rows:
//   scale,n_droplets,n_genes,lower,wall_ms,mem_mb
//
// §J.7 prediction (4-axis formula):
//   SOTA structure: manual numpy/scipy — fully vectorized, no Python inner loop
//     (np.sum/slice on CSC, argpartition, multiply-subtract-clip over sparse nnz).
//     python_overhead_multiplier = 1× (tight vectorized ops, no per-element Python).
//   Dense intermediates: GPU outputs dense m×n (114 MB at 10k; 342 MB at 30k).
//     CPU outputs sparse corrections (no dense intermediate).
//     memory_bandwidth_advantage = 1× (GPU dense write offsets HBM bandwidth gain;
//     CPU stays sparse; overall wash at these scales).
//   GPU compute: light-medium (5 passes: warp colsum, atomic scatter, topK mask,
//     warp rho, dense memset + nnz-write). No heavy BLAS.
//   → Predicted speedup: class 3, 10-30× (same as wsum/decoupler family).
//     Lower bound of class 3 because GPU materializes dense output that CPU avoids.
//
// §J.8: harness API verified from bench_qc_empty_drops_perf.cpp grep:
//   bench::BenchTimer timer; timer.start(stream); timer.stop(stream); timer.elapsed_ms()
//   bench::PeakMemTracker mem; mem.sample_before(); mem.sample_after();
//     mem.peak_delta_mb(); mem.reset()
//   bench::BenchRow row; row.date/feature/scale/impl/wall_ms/mem_mb/cells_per_sec/commit
//   bench::today_iso(); bench::git_short_sha(); bench::throughput(n_cells, wall_ms)
//   bench::log_row(row); gpu_available()
//
// Node: any non-excluded GPU node (--exclude=g001,g002,g005 per §J.2).

#include <singlet_gpu/bench/harness.h>
#include <singlet-gpu/qc/soupx.h>
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace singlet_gpu;

// ---------------------------------------------------------------------------
// make_synth_10x_csc
//
// Builds a synthetic raw 10X-style CSC matrix on the device:
//   Rows = genes (n_genes), Cols = droplets (n_droplets)
//   ~frac_empty fraction of droplets are "empty" (UMI <= lower).
//   Remaining droplets are cells/candidates (UMI > lower).
//   Gene frequency follows Zipf: pi[g] ∝ 1/(g+1), normalized.
//   Counts per droplet drawn by multinomial on host (seeded xorshift).
//
// Returns an io::PzDeviceMatrix (CSC: col_ptr, row_indices, values).
// ---------------------------------------------------------------------------
static io::PzDeviceMatrix make_synth_10x_csc(
    int n_genes, int n_droplets, int lower,
    float frac_empty, cudaStream_t stream)
{
    // --- Zipf ambient profile (n_genes long) ---
    std::vector<float> pi(n_genes);
    float pi_sum = 0.f;
    for (int g = 0; g < n_genes; ++g) {
        pi[g] = 1.f / static_cast<float>(g + 1);
        pi_sum += pi[g];
    }
    for (int g = 0; g < n_genes; ++g) pi[g] /= pi_sum;

    // Cumulative distribution for multinomial sampling via xorshift.
    std::vector<float> cdf(n_genes);
    float cum = 0.f;
    for (int g = 0; g < n_genes; ++g) { cum += pi[g]; cdf[g] = cum; }
    cdf[n_genes - 1] = 1.f;

    // --- Per-droplet UMI totals ---
    uint64_t rng = 42ULL ^ 0xCAFEBABEDEADBEEFull;
    auto xorshift = [&]() -> uint64_t {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return rng;
    };
    auto uniform01 = [&]() -> float {
        return static_cast<float>(xorshift() & 0xFFFFFFFFull) *
               (1.f / 4294967296.f);
    };
    // Knuth Poisson for small lambda.
    auto poisson_approx = [&](float mean) -> int {
        float L = expf(-mean);
        float p = 1.f; int k = 0;
        do { p *= uniform01(); ++k; } while (p > L && k < static_cast<int>(2 * mean + 20));
        return k - 1;
    };

    int n_empty = static_cast<int>(frac_empty * n_droplets);
    std::vector<int> umi(n_droplets);
    for (int j = 0; j < n_empty; ++j)
        umi[j] = std::max(1, poisson_approx(static_cast<float>(lower) * 0.3f));
    for (int j = n_empty; j < n_droplets; ++j)
        umi[j] = lower + 1 + static_cast<int>(poisson_approx(200.f));

    // --- Build CSC: sample per-droplet multinomial counts ---
    std::vector<int32_t> h_col_ptr(n_droplets + 1, 0);
    std::vector<int32_t> h_row_idx;
    std::vector<float>   h_vals;
    h_row_idx.reserve(static_cast<size_t>(n_droplets) * 30);
    h_vals.reserve(static_cast<size_t>(n_droplets) * 30);

    for (int j = 0; j < n_droplets; ++j) {
        std::vector<int> gene_cnt(n_genes, 0);
        const int t_j = umi[j];
        for (int it = 0; it < t_j; ++it) {
            float u = uniform01();
            if (u >= 1.f) u = 0.9999999f;
            int lo = 0, hi = n_genes - 1;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (cdf[mid] < u) lo = mid + 1; else hi = mid;
            }
            gene_cnt[lo]++;
        }
        int nnz_j = 0;
        for (int g = 0; g < n_genes; ++g) {
            if (gene_cnt[g] > 0) {
                h_row_idx.push_back(static_cast<int32_t>(g));
                h_vals.push_back(static_cast<float>(gene_cnt[g]));
                ++nnz_j;
            }
        }
        h_col_ptr[j + 1] = h_col_ptr[j] + nnz_j;
    }

    const int nnz = static_cast<int>(h_vals.size());

    // Upload to device.
    core::DeviceMemory<int32_t> d_col_ptr(static_cast<size_t>(n_droplets + 1));
    core::DeviceMemory<int32_t> d_row_idx(static_cast<size_t>(nnz));
    core::DeviceMemory<float>   d_vals(static_cast<size_t>(nnz));

    cudaMemcpyAsync(d_col_ptr.get(), h_col_ptr.data(),
                    static_cast<size_t>(n_droplets + 1) * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    if (nnz > 0) {
        cudaMemcpyAsync(d_row_idx.get(), h_row_idx.data(),
                        static_cast<size_t>(nnz) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_vals.get(), h_vals.data(),
                        static_cast<size_t>(nnz) * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }
    cudaStreamSynchronize(stream);

    io::PzDeviceMatrix mat{};
    mat.mat.rows       = n_genes;
    mat.mat.cols       = n_droplets;
    mat.mat.nnz        = nnz;
    mat.mat.col_ptr    = std::move(d_col_ptr);
    mat.mat.row_indices = std::move(d_row_idx);
    mat.mat.values     = std::move(d_vals);

    return mat;
}

// ---------------------------------------------------------------------------
// bench_soupx — 2 warmup + 5 timed iterations of qc::soupx().
// Returns median wall_ms and peak mem_mb via out params.
// ---------------------------------------------------------------------------
static void bench_soupx(
    const io::PzDeviceMatrix& X,
    const qc::SoupxConfig& cfg,
    cudaStream_t stream,
    double& out_wall_ms,
    double& out_mem_mb)
{
    constexpr int WARMUP = 2;
    constexpr int TIMED  = 5;
    double samples[TIMED]{};
    double mem_samples[TIMED]{};

    bench::BenchTimer     timer;
    bench::PeakMemTracker mem;

    for (int iter = 0; iter < WARMUP + TIMED; ++iter) {
        const bool is_timed = (iter >= WARMUP);
        if (is_timed) mem.sample_before();

        timer.start(stream);

        {
            auto result = qc::soupx(X, cfg, stream);
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
        std::puts("NO GPU — skipping Cycle 180 qc/soupx bench.");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // CSV header.
    std::printf("scale,n_droplets,n_genes,lower,wall_ms,mem_mb\n");
    std::fflush(stdout);

    struct Scale {
        const char* name;
        int n_droplets;
        int n_genes;
        int lower;
    };
    const Scale scales[] = {
        { "10k", 10000, 3000, 100 },
        { "30k", 30000, 3000, 100 },
    };

    for (const auto& s : scales) {
        std::printf("[bench] Synthesizing %s: %d genes × %d droplets, lower=%d...\n",
                    s.name, s.n_genes, s.n_droplets, s.lower);
        std::fflush(stdout);

        io::PzDeviceMatrix X = make_synth_10x_csc(
            s.n_genes, s.n_droplets, s.lower,
            /*frac_empty=*/0.8f, stream);

        std::printf("[bench] %s: nnz=%d, n_droplets=%d. "
                    "Timing qc::soupx (5-pass)...\n",
                    s.name, X.mat.nnz, s.n_droplets);
        std::fflush(stdout);

        qc::SoupxConfig cfg{};
        cfg.lower            = s.lower;
        cfg.top_ambient_frac = 0.10f;
        cfg.min_rho          = 0.0f;
        cfg.max_rho          = 0.9f;
        cfg.deterministic    = true;

        double wall_ms = 0.0, mem_mb = 0.0;
        try {
            bench_soupx(X, cfg, stream, wall_ms, mem_mb);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[bench] ERROR %s: %s\n", s.name, e.what());
            std::printf("%s,%d,%d,%d,ERROR,ERROR\n",
                        s.name, s.n_droplets, s.n_genes, s.lower);
            std::fflush(stdout);
            continue;
        }

        // CSV row.
        std::printf("%s,%d,%d,%d,%.3f,%.1f\n",
                    s.name, s.n_droplets, s.n_genes, s.lower,
                    wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "qc/soupx";
        row.scale         = std::string(s.name);
        row.impl          = "singlet-gpu";
        row.wall_ms       = wall_ms;
        row.mem_mb        = mem_mb;
        row.cells_per_sec = bench::throughput(static_cast<int64_t>(s.n_droplets),
                                               wall_ms);
        row.commit        = bench::git_short_sha();
        bench::log_row(row);
    }

    cudaStreamDestroy(stream);
    std::printf("[bench] Cycle 180 qc/soupx perf bench complete.\n");
    return 0;
}
