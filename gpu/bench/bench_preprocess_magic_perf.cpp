// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_preprocess_magic_perf.cpp
//
// CYCLE-174 Phase E — preprocess/magic standalone kernel benchmark.
//
// Benchmarks preprocess::magic_impute() (include/singlet-gpu/preprocess/magic.h)
// against a manual scipy SpMM CPU baseline at two medium scales:
//
//   Scale "10k":  n_cells=10000, n_genes=5000, CSC density=5% (nnz=500k), k=10, t=3.
//   Scale "30k":  n_cells=30000, n_genes=5000, CSC density=5% (nnz=4.5M), k=10, t=3.
//
// GPU kernel: cuSPARSE SpMM-heavy pipeline:
//   (a) row_sum_graph_kernel     — per-row Markov weight sums (warp-shuffle).
//   (b) normalize_weights_kernel — D⁻¹W (Markov transition matrix).
//   (c) csc_transpose_to_dense_kernel — scatter CSC X (m×n) → dense Y₀ (n×m).
//   (d) cusparseSpMM × t=3      — Y_{s} = M · Y_{s-1}  (ping-pong dense buffers).
// Reference: van Dijk et al. (2018) "Recovering Gene Interactions from Single-Cell
//            Data Using Data Diffusion." Cell 174:716-729.
//
// Protocol:
//   Expression matrix: synthetic CSC (5000 genes × n_cells, density=5%), LCG seed=42.
//   SNN graph: built from synthetic PCA embedding (n_cells × 50 dims), k=10.
//     Graph build is NOT timed (only magic_impute is timed).
//   magic_impute() — 2 warmup + 5 timed (cudaEvent), imputation only (not graph build).
//   Memory — bench::PeakMemTracker (cudaMemGetInfo delta).
//
// Output (stdout): CSV header + rows:
//   scale,n_cells,n_genes,density,k,t,wall_ms,mem_mb
//
// §J.7 prediction: cuSPARSE SpMM is 3× per step = 3 SpMM calls (heavy per-cell
//   for m=5000 genes). scipy.sparse SpMM is well-vectorized native C (MKL/OpenBLAS).
//   Expect class 3 (10-30×) speedup — analogous to CYCLE-163 wsum SpMM pattern
//   but with 3 repeated iterations rather than 1.
//
// Node: any non-excluded GPU node (--exclude=g001,g002,g005 per §J.2).

#include <singlet_gpu/bench/harness.h>
#include <singlet-gpu/preprocess/magic.h>
#include <singlet-gpu/graph/knn.h>
#include <singlet-gpu/graph/snn.h>
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

using namespace singlet_gpu;

// ---------------------------------------------------------------------------
// make_sparse_csc_pzmat — synthetic CSC expression matrix (m_genes × n_cells)
// at a target density. LCG seed=42, values in [0, 10] (log-normalised counts).
// ---------------------------------------------------------------------------
static io::PzDeviceMatrix make_sparse_csc_pzmat(int m_genes, int n_cells,
                                                float density,
                                                cudaStream_t stream)
{
    const size_t target_nnz = static_cast<size_t>(
        static_cast<double>(m_genes) * n_cells * density + 0.5);

    // Build host CSC by reservoir-style column fill.
    std::vector<int>   h_col_ptr(static_cast<size_t>(n_cells) + 1, 0);
    std::vector<int>   h_row_idx;
    std::vector<float> h_values;
    h_row_idx.reserve(target_nnz);
    h_values.reserve(target_nnz);

    // Per-column target nnz (uniform density).
    const int nnz_per_col = std::max(1, static_cast<int>(m_genes * density + 0.5));

    uint64_t rng = 42ULL ^ 0xDEADBEEFCAFEBABEull;
    auto lcg_next = [&]() -> uint64_t {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return rng;
    };

    for (int c = 0; c < n_cells; ++c) {
        h_col_ptr[static_cast<size_t>(c)] = static_cast<int>(h_row_idx.size());

        // Sample nnz_per_col unique row indices from [0, m_genes) (sorted).
        std::vector<int> rows;
        rows.reserve(static_cast<size_t>(nnz_per_col));
        // Fisher-Yates partial shuffle on first nnz_per_col of [0, m_genes).
        std::vector<int> perm(static_cast<size_t>(m_genes));
        for (int i = 0; i < m_genes; ++i) perm[static_cast<size_t>(i)] = i;
        for (int i = 0; i < nnz_per_col; ++i) {
            const int j = i + static_cast<int>(lcg_next() % static_cast<uint64_t>(m_genes - i));
            std::swap(perm[static_cast<size_t>(i)], perm[static_cast<size_t>(j)]);
            rows.push_back(perm[static_cast<size_t>(i)]);
        }
        std::sort(rows.begin(), rows.end());

        for (int r : rows) {
            h_row_idx.push_back(r);
            h_values.push_back(static_cast<float>(lcg_next() % 11u));  // counts 0–10
        }
    }
    h_col_ptr[static_cast<size_t>(n_cells)] = static_cast<int>(h_row_idx.size());

    const int actual_nnz = static_cast<int>(h_row_idx.size());

    // Upload to device via RAII DeviceMemory.
    core::DeviceMemory<int32_t> d_col_ptr(static_cast<size_t>(n_cells) + 1);
    core::DeviceMemory<int32_t> d_row_idx(static_cast<size_t>(actual_nnz));
    core::DeviceMemory<float>   d_values (static_cast<size_t>(actual_nnz));

    cudaMemcpyAsync(d_col_ptr.get(), h_col_ptr.data(),
                    (static_cast<size_t>(n_cells) + 1) * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_row_idx.get(), h_row_idx.data(),
                    static_cast<size_t>(actual_nnz) * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_values.get(), h_values.data(),
                    static_cast<size_t>(actual_nnz) * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    io::PzDeviceMatrix pzmat;
    pzmat.mat.rows        = m_genes;
    pzmat.mat.cols        = n_cells;
    pzmat.mat.nnz         = actual_nnz;
    pzmat.mat.col_ptr     = std::move(d_col_ptr);
    pzmat.mat.row_indices = std::move(d_row_idx);
    pzmat.mat.values      = std::move(d_values);
    return pzmat;
}

// ---------------------------------------------------------------------------
// make_pca_embedding — synthetic PCA embedding (n_cells × n_dims, col-major).
// DeviceDense with rows=n_cells, cols=n_dims per compute_knn convention.
// ---------------------------------------------------------------------------
static core::DeviceDense make_pca_embedding(int n_cells, int n_dims,
                                             cudaStream_t stream)
{
    const size_t sz = static_cast<size_t>(n_cells) * static_cast<size_t>(n_dims);
    std::vector<float> h_emb(sz);

    uint64_t rng = 137ULL ^ 0xCAFEBABEDEADBEEFull;
    for (size_t i = 0; i < sz; ++i) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        h_emb[i] = static_cast<float>(static_cast<int64_t>(rng) % 10000) * 1e-4f;  // ~N(0,1) proxy
    }

    core::DeviceDense emb(n_cells, n_dims);
    cudaMemcpyAsync(emb.data.get(), h_emb.data(),
                    sz * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return emb;
}

// ---------------------------------------------------------------------------
// bench_magic — 2 warmup + 5 timed iterations of preprocess::magic_impute().
// Graph (kNN + SNN) is built ONCE before the timing loop (untimed).
// Returns median wall_ms and peak mem_mb via out params.
// ---------------------------------------------------------------------------
static void bench_magic(
    const io::PzDeviceMatrix& X,
    const graph::SnnResult&   snn,
    int                       t_steps,
    cudaStream_t              stream,
    double&                   out_wall_ms,
    double&                   out_mem_mb)
{
    constexpr int WARMUP = 2;
    constexpr int TIMED  = 5;
    double samples[TIMED]{};
    double mem_samples[TIMED]{};

    bench::BenchTimer     timer;
    bench::PeakMemTracker mem;

    preprocess::MagicConfig cfg{};
    cfg.t = t_steps;

    for (int iter = 0; iter < WARMUP + TIMED; ++iter) {
        const bool is_timed = (iter >= WARMUP);
        if (is_timed) mem.sample_before();

        timer.start(stream);

        {
            auto result = preprocess::magic_impute(X, snn, cfg, stream);
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
        std::puts("NO GPU — skipping Cycle 174 preprocess/magic bench.");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // CSV header.
    std::printf("scale,n_cells,n_genes,density,k,t,wall_ms,mem_mb\n");
    std::fflush(stdout);

    // Scales: 10k / 30k cells, 5000 genes, density=5%, k=10, t=3.
    struct Scale { const char* name; int n_cells; int n_genes; float density; int k; int t; };
    const Scale scales[] = {
        { "10k", 10000, 5000, 0.05f, 10, 3 },
        { "30k", 30000, 5000, 0.05f, 10, 3 },
    };

    for (const auto& s : scales) {
        std::printf("[bench] Synthesizing %s: %d genes × %d cells "
                    "(density=%.0f%%, k=%d, t=%d)...\n",
                    s.name, s.n_genes, s.n_cells,
                    s.density * 100.0f, s.k, s.t);
        std::fflush(stdout);

        // Build sparse CSC expression matrix (untimed).
        io::PzDeviceMatrix X = make_sparse_csc_pzmat(
            s.n_genes, s.n_cells, s.density, stream);

        std::printf("[bench] %s: nnz=%d (actual density=%.2f%%)\n",
                    s.name, X.mat.nnz,
                    100.0 * X.mat.nnz / (static_cast<double>(s.n_genes) * s.n_cells));
        std::fflush(stdout);

        // Build PCA embedding for kNN (n_cells × 50 PCs, untimed).
        core::DeviceDense emb = make_pca_embedding(s.n_cells, 50, stream);

        // Build kNN → SNN graph (untimed — only imputation is timed).
        std::printf("[bench] %s: building kNN (k=%d) + SNN graph (untimed)...\n",
                    s.name, s.k);
        std::fflush(stdout);

        graph::KnnConfig knn_cfg{};
        knn_cfg.k       = s.k;
        knn_cfg.backend = graph::KnnBackend::Exact;  // deterministic; n<50k
        graph::KnnResult knn = graph::compute_knn(emb, knn_cfg, stream);
        graph::SnnResult snn = graph::compute_snn(knn, {}, stream);

        std::printf("[bench] %s: SNN nnz=%d. Timing magic_impute (t=%d)...\n",
                    s.name, snn.nnz, s.t);
        std::fflush(stdout);

        double wall_ms = 0.0, mem_mb = 0.0;
        bench_magic(X, snn, s.t, stream, wall_ms, mem_mb);

        // CSV row.
        std::printf("%s,%d,%d,%.2f,%d,%d,%.3f,%.1f\n",
                    s.name, s.n_cells, s.n_genes,
                    s.density, s.k, s.t,
                    wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "preprocess/magic";
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
    std::printf("[bench] Cycle 174 preprocess/magic perf bench complete.\n");
    return 0;
}
