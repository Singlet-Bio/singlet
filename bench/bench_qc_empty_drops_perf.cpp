// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_qc_empty_drops_perf.cpp
//
// CYCLE-179 Phase E — qc/empty_drops standalone kernel benchmark.
//
// Benchmarks qc::empty_drops() (include/singlet/gpu/qc/empty_drops.h)
// against a manual scipy/numpy Python MC baseline at two scales:
//
//   Scale "10k":  n_droplets=10000, n_genes=3000, lower=100, niters=10000.
//   Scale "30k":  n_droplets=30000, n_genes=3000, lower=100, niters=10000.
//
// Synthetic raw 10X-style sparse CSC (col = droplet, row = gene):
//   ~80% of droplets are "empty" (UMI total <= lower, drawn from Poisson(30)).
//   ~20% are "candidates" (UMI total > lower, drawn from NegBin mean=300).
//   Counts are sparse: gene probabilities follow a Zipf-like distribution
//   (first 500 genes carry 80% of mass). Density ~3% per candidate droplet.
//
// GPU kernel: 6-pass: col_sum → mark_empty → ambient_scatter+normalize
//   → obs_LL (warp/cand) → MC p-values (cuRAND Philox4x32, smem CDF,
//     binary-search categorical sampling, one block/cand)
//   → BH FDR (host-side sort+scan at D2H boundary) → scatter is_cell.
// Reference: Lun ATL et al. (2019) Genome Biology 20:63.
//
// Protocol:
//   Synthetic PzDeviceMatrix constructed on device via host→device copy.
//   empty_drops() — 2 warmup + 5 timed (cudaEvent).
//   Memory — bench::PeakMemTracker (cudaMemGetInfo delta).
//
// Output (stdout): CSV header + rows:
//   scale,n_droplets,n_genes,n_cand,lower,niters,wall_ms,mem_mb
//
// §J.7 prediction (4-axis formula):
//   SOTA structure: scipy multinomial sampling in a Python loop per droplet
//     per MC iteration → python_overhead_multiplier = 10-100× (pure Python
//     inner loop per draw per iter). No vectorized bulk path exists for this
//     exact computation.
//   Dense intermediates: each MC iter needs t_j draws per candidate; no
//     large dense matrix materialised on CPU → memory_bandwidth_advantage = 1×.
//   GPU compute: medium-heavy (cuRAND init per cand-thread; smem CDF build;
//     O(t_j × log(m)) binary search per MC iter; BH on host).
//   Predicted speedup: 50-200× (class 2-3).
//
// §J.8: harness API verified from bench_anno_symphony_perf.cpp grep:
//   bench::BenchTimer timer; timer.start(stream); timer.stop(stream); timer.elapsed_ms()
//   bench::PeakMemTracker mem; mem.sample_before(); mem.sample_after();
//     mem.peak_delta_mb(); mem.reset()
//   bench::BenchRow row; row.date/feature/scale/impl/wall_ms/mem_mb/cells_per_sec/commit
//   bench::today_iso(); bench::git_short_sha(); bench::throughput(n_cells, wall_ms)
//   bench::log_row(row); gpu_available()
//
// Node: any non-excluded GPU node (--exclude=g001,g002,g005 per §J.2).

#include <singlet/gpu/bench/harness.h>
#include <singlet/gpu/qc/empty_drops.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace singlet::gpu;

// ---------------------------------------------------------------------------
// make_synth_10x_csc
//
// Builds a synthetic raw 10X-style CSC matrix on the device:
//   Rows = genes (n_genes), Cols = droplets (n_droplets)
//   ~frac_empty fraction of droplets are "empty" (UMI <= lower).
//   Remaining droplets are "candidates" (UMI > lower).
//   Gene frequency follows Zipf: pi[g] ∝ 1/(g+1), normalized.
//   Counts per droplet drawn by multinomial(t_j, pi) on host (seeded xorshift).
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
    // Simple Poisson-ish approx: use Geometric(p) to control mean.
    auto poisson_approx = [&](float mean) -> int {
        // Knuth Poisson for small lambda; clamp at 2*mean+10.
        float L = expf(-mean);
        float p = 1.f; int k = 0;
        do { p *= uniform01(); ++k; } while (p > L && k < static_cast<int>(2 * mean + 20));
        return k - 1;
    };

    int n_empty = static_cast<int>(frac_empty * n_droplets);
    int n_cand  = n_droplets - n_empty;
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

    // For each droplet draw umi[j] genes via binary search on cdf.
    for (int j = 0; j < n_droplets; ++j) {
        // Accumulate counts per gene.
        std::vector<int> gene_cnt(n_genes, 0);
        const int t_j = umi[j];
        for (int it = 0; it < t_j; ++it) {
            float u = uniform01();
            if (u >= 1.f) u = 0.9999999f;
            // Binary search on cdf.
            int lo = 0, hi = n_genes - 1;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (cdf[mid] < u) lo = mid + 1; else hi = mid;
            }
            gene_cnt[lo]++;
        }
        // Emit only nonzero gene counts (sparse).
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

    // Construct PzDeviceMatrix.
    io::PzDeviceMatrix mat{};
    mat.mat.rows     = n_genes;
    mat.mat.cols     = n_droplets;
    mat.mat.nnz      = nnz;
    mat.mat.col_ptr  = std::move(d_col_ptr);
    mat.mat.row_indices = std::move(d_row_idx);
    mat.mat.values   = std::move(d_vals);

    (void)n_cand;
    return mat;
}

// ---------------------------------------------------------------------------
// bench_empty_drops — 2 warmup + 5 timed iterations of qc::empty_drops().
// Returns median wall_ms and peak mem_mb via out params.
// ---------------------------------------------------------------------------
static void bench_empty_drops(
    const io::PzDeviceMatrix& X,
    const qc::EmptyDropsConfig& cfg,
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
            auto result = qc::empty_drops(X, cfg, stream);
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
        std::puts("NO GPU — skipping Cycle 179 qc/empty_drops bench.");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // CSV header.
    std::printf("scale,n_droplets,n_genes,lower,niters,wall_ms,mem_mb\n");
    std::fflush(stdout);

    struct Scale {
        const char* name;
        int n_droplets;
        int n_genes;
        int lower;
        int niters;
    };
    const Scale scales[] = {
        { "10k", 10000, 3000, 100, 10000 },
        { "30k", 30000, 3000, 100, 10000 },
    };

    for (const auto& s : scales) {
        std::printf("[bench] Synthesizing %s: %d genes × %d droplets, "
                    "lower=%d, niters=%d...\n",
                    s.name, s.n_genes, s.n_droplets, s.lower, s.niters);
        std::fflush(stdout);

        io::PzDeviceMatrix X = make_synth_10x_csc(
            s.n_genes, s.n_droplets, s.lower,
            /*frac_empty=*/0.8f, stream);

        std::printf("[bench] %s: nnz=%d, n_droplets=%d. "
                    "Timing qc::empty_drops (6-pass)...\n",
                    s.name, X.mat.nnz, s.n_droplets);
        std::fflush(stdout);

        qc::EmptyDropsConfig cfg{};
        cfg.lower          = s.lower;
        cfg.niters         = s.niters;
        cfg.fdr_thresh     = 0.001f;
        cfg.seed           = 42ULL;
        cfg.max_candidates = 50000;

        double wall_ms = 0.0, mem_mb = 0.0;
        try {
            bench_empty_drops(X, cfg, stream, wall_ms, mem_mb);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[bench] ERROR %s: %s\n", s.name, e.what());
            std::printf("%s,%d,%d,%d,%d,ERROR,ERROR\n",
                        s.name, s.n_droplets, s.n_genes, s.lower, s.niters);
            std::fflush(stdout);
            continue;
        }

        // CSV row.
        std::printf("%s,%d,%d,%d,%d,%.3f,%.1f\n",
                    s.name, s.n_droplets, s.n_genes,
                    s.lower, s.niters,
                    wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "qc/empty_drops";
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
    std::printf("[bench] Cycle 179 qc/empty_drops perf bench complete.\n");
    return 0;
}
