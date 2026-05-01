// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_enrich_decoupler_viper_perf.cpp
//
// CYCLE-167 Phase E — enrich/decoupler_viper standalone kernel benchmark.
//
// Benchmarks decoupler_viper::viper() (from include/singlet-gpu/enrich/decoupler_viper.h)
// against a manual numpy/scipy CPU baseline at two medium scales:
//
//   Scale "10k":  n_cells=10000, n_genes=5000, density=5%, n_reg=50
//   Scale "30k":  n_cells=30000, n_genes=5000, density=5%, n_reg=50
//
// GPU kernel: 3 passes — viper_fill_dense_kernel (CSC → dense m×n),
//   CUB DeviceSegmentedRadixSort (per-cell ascending sort → permutation),
//   viper_assign_t1_kernel (normcdfinvf qnorm per rank → T1 m×n),
//   cuBLAS Sgemm (T1^T · W → NES, n×n_reg col-major),
//   viper_l1_norm_kernel + viper_scale_kernel (per-regulon L1 normalisation).
// Reference: Alvarez MJ et al. (2016) "VIPER/aREA" Nat Genet 48:838-847.
// V0 simplification: NES = ES1 = (T1^T · W) / L1(W).
//
// Protocol: 2 warmup + 5 timed iterations; wall time via cudaEventElapsedTime.
// Memory via cudaMemGetInfo delta (PeakMemoryTracker).
//
// Output (stdout): CSV header + rows:
//   scale,n_cells,n_genes,density,n_reg,wall_ms,mem_mb
//
// Node: any non-excluded GPU node (--exclude=g001,g002,g005 per §J.2).

#include <singlet_gpu/bench/harness.h>
#include <singlet-gpu/enrich/decoupler_viper.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace singlet_gpu;

// ---------------------------------------------------------------------------
// upload_synthetic_csc — build a CSC sparse matrix on device from host vecs.
// Returns a PzDeviceMatrix populated from a SyntheticMatrix.
// Uses core::DeviceMemory<T> throughout (Rule 5 — no raw cudaMalloc).
// CSC convention: rows=genes (m), cols=cells (n).
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
// make_weight_matrix — dense (n_genes × n_reg) signed weight matrix W.
//
// Values drawn from a deterministic pseudo-random sequence in [-1, 1].
// Uses LCG seeded at 99 to avoid correlation with X (seed 42).
// W is col-major float32, uploaded to device.
// ---------------------------------------------------------------------------
static core::DeviceMemory<float> make_weight_matrix(int n_genes, int n_reg,
                                                     cudaStream_t stream)
{
    const size_t total = static_cast<size_t>(n_genes) * n_reg;
    std::vector<float> h_W(total);

    // LCG seed=99 (matches Python ref rng=99).
    uint64_t state = 99ULL;
    for (size_t i = 0; i < total; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        // Map to [-1, 1]: use upper 32 bits.
        float u = static_cast<float>(static_cast<uint32_t>(state >> 32))
                  / static_cast<float>(0xFFFFFFFFu);
        h_W[i] = u * 2.0f - 1.0f;
    }

    core::DeviceMemory<float> d_W(total);
    cudaMemcpyAsync(d_W.get(), h_W.data(),
                    total * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return d_W;
}

// ---------------------------------------------------------------------------
// bench_viper — 2 warmup + 5 timed iterations of viper().
// Returns median wall_ms and peak mem_mb.
// ---------------------------------------------------------------------------
static void bench_viper(
    const io::PzDeviceMatrix& pzmat,
    const float*              d_W,
    int                       n_reg,
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
            enrich::ViperConfig cfg{};
            auto result = enrich::viper(pzmat, d_W, n_reg, cfg, stream);
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
        std::puts("NO GPU — skipping Cycle 167 decoupler_viper bench.");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // CSV header.
    std::printf("scale,n_cells,n_genes,density,n_reg,wall_ms,mem_mb\n");
    std::fflush(stdout);

    // Scales.
    struct Scale { const char* name; int n_cells; int n_genes; float density; };
    const Scale scales[] = {
        { "10k",  10000, 5000, 0.05f },
        { "30k",  30000, 5000, 0.05f },
    };
    const int n_reg = 50;

    for (const auto& s : scales) {
        std::printf("[bench] Synthesizing %s: %d cells × %d genes density=%.0f%%...\n",
                    s.name, s.n_cells, s.n_genes, s.density * 100.f);
        std::fflush(stdout);

        // SyntheticMatrix: rows=genes (m), cols=cells (n) — CSC layout.
        SyntheticMatrix syn = make_synthetic_matrix(
            s.n_genes, s.n_cells, s.density, /*seed=*/42);

        io::PzDeviceMatrix pzmat = upload_synthetic_csc(syn, stream);

        // Build dense weight matrix W: n_genes × n_reg, col-major fp32.
        core::DeviceMemory<float> d_W =
            make_weight_matrix(s.n_genes, n_reg, stream);

        std::printf("[bench] Timing %s method=viper (n_reg=%d)...\n",
                    s.name, n_reg);
        std::fflush(stdout);

        double wall_ms = 0.0, mem_mb = 0.0;
        bench_viper(pzmat, d_W.get(), n_reg, stream, wall_ms, mem_mb);

        // CSV row.
        std::printf("%s,%d,%d,%.2f,%d,%.3f,%.1f\n",
                    s.name, s.n_cells, s.n_genes, s.density, n_reg,
                    wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "enrich/decoupler_viper";
        row.scale         = std::string(s.name);
        row.impl          = "singlet-gpu";
        row.wall_ms       = static_cast<float>(wall_ms);
        row.mem_mb        = static_cast<float>(mem_mb);
        row.cells_per_sec = bench::throughput(static_cast<int64_t>(s.n_cells), wall_ms);
        row.commit        = bench::git_short_sha();
        bench::log_row(row);
    }

    cudaStreamDestroy(stream);
    std::printf("[bench] Cycle 167 decoupler_viper perf bench complete.\n");
    return 0;
}
