// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_embed_dendrogram_perf.cpp
//
// CYCLE-173 Phase E — embed/dendrogram standalone kernel benchmark.
//
// Benchmarks embed::dendrogram() (include/singlet/gpu/embed/dendrogram.h)
// against a manual scipy/numpy CPU baseline at two medium scales:
//
//   Scale "10k":  n_cells=10000, n_genes=50 (PCA embedding as dense CSC),
//                 n_clusters=20, density=1.0 (dense), round-robin labels.
//   Scale "30k":  n_cells=30000, n_genes=50, n_clusters=20, density=1.0.
//
// GPU kernel: 6 passes:
//   (a) dgram_centroid_scatter_kernel — atomic-scatter nnz into μ (one block/cell).
//   (b) dgram_centroid_divide_kernel  — μ[g,k] /= n_per_cluster[k].
//   (c) dgram_col_mean_kernel         — per-column mean of μ (one block/cluster).
//   (d) dgram_center_kernel           — subtract column mean (element-wise).
//   (e) dgram_col_norm_kernel + dgram_normalize_kernel — L2 normalize each column.
//   (f) cublasSgemm corr = μ_norm^T · μ_norm (k × k); dgram_dist_kernel d = 1 - corr.
//   Host UPGMA — O(k³) pure C++ on the k×k distance matrix (k ≤ 50 → trivial).
// Reference: scanpy.tl.dendrogram (Wolf et al. 2018 Genome Biol 19:15).
//
// Protocol:
//   Embedding — dense synthetic fp32 (50 PCs × n_cells), col-major.
//     Wrapped as PzDeviceMatrix (CSC with nnz = 50 × n_cells, density=1.0).
//     Round-robin cluster labels: label[c] = c % n_clusters.
//   dendrogram() — 2 warmup + 5 timed via cudaEvent (bench::BenchTimer).
//   Memory        — bench::PeakMemTracker (cudaMemGetInfo delta).
//
// Output (stdout): CSV header + rows:
//   scale,n_cells,n_pcs,n_clusters,wall_ms,mem_mb
//
// §J.7 prediction: dominated by atomic centroid scatter (light per-cell, bounded by
//   n_genes=50) + small k×k Sgemm (k=20, trivial) + host UPGMA (O(k³), ~8000 ops).
//   Total GPU should be sub-ms. scipy hierarchy has Python orchestration overhead.
//   Expect class 2 (50-200×) speedup.
//
// Node: any non-excluded GPU node (--exclude=g001,g002,g005 per §J.2).

#include <singlet/gpu/bench/harness.h>
#include <singlet/gpu/embed/dendrogram.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

using namespace singlet::gpu;

// ---------------------------------------------------------------------------
// make_dense_csc_pzmat — build a dense CSC PzDeviceMatrix (n_genes × n_cells)
// from a synthetic fp32 column-major embedding.
//
// For dendrogram, the "expression matrix" is the PCA embedding treated as dense.
// We store it as CSC with all entries populated (density=1.0): nnz = n_genes*n_cells,
// row_indices[j + c*n_genes] = j, col_ptr[c] = c*n_genes.
// ---------------------------------------------------------------------------
static io::PzDeviceMatrix make_dense_csc_pzmat(int n_genes, int n_cells,
                                               cudaStream_t stream)
{
    const size_t nnz   = static_cast<size_t>(n_genes) * static_cast<size_t>(n_cells);
    const size_t n_col = static_cast<size_t>(n_cells);

    // Host-side CSC construction.
    std::vector<int>   h_col_ptr(n_col + 1);
    std::vector<int>   h_row_idx(nnz);
    std::vector<float> h_values(nnz);

    // col_ptr[c] = c * n_genes.
    for (size_t c = 0; c <= n_col; ++c)
        h_col_ptr[c] = static_cast<int>(c * static_cast<size_t>(n_genes));

    // row_indices and values: column c → rows 0..n_genes-1.
    // Values: simple LCG in [0, 10] (counts-like).
    uint64_t rng = 42ULL ^ 0xDEADBEEFCAFEBABEull;
    for (size_t i = 0; i < nnz; ++i) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        h_row_idx[i] = static_cast<int>(i % static_cast<size_t>(n_genes));
        h_values[i]  = static_cast<float>(rng % 11u);  // integer counts 0–10
    }

    // Upload to device via RAII DeviceMemory (Rule 5).
    core::DeviceMemory<int>   d_col_ptr(n_col + 1);
    core::DeviceMemory<int>   d_row_idx(nnz);
    core::DeviceMemory<float> d_values(nnz);

    cudaMemcpyAsync(d_col_ptr.get(), h_col_ptr.data(),
                    (n_col + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_row_idx.get(), h_row_idx.data(),
                    nnz * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_values.get(), h_values.data(),
                    nnz * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    io::PzDeviceMatrix pzmat;
    pzmat.mat.rows        = n_genes;
    pzmat.mat.cols        = n_cells;
    pzmat.mat.nnz         = static_cast<int>(nnz);
    pzmat.mat.col_ptr     = std::move(d_col_ptr);
    pzmat.mat.row_indices = std::move(d_row_idx);
    pzmat.mat.values      = std::move(d_values);
    return pzmat;
}

// ---------------------------------------------------------------------------
// make_roundrobin_labels — device int[n_cells] with label[c] = c % n_clusters.
// ---------------------------------------------------------------------------
static core::DeviceMemory<int> make_roundrobin_labels(int n_cells, int n_clusters,
                                                       cudaStream_t stream)
{
    std::vector<int> h_labels(static_cast<size_t>(n_cells));
    for (int c = 0; c < n_cells; ++c)
        h_labels[static_cast<size_t>(c)] = c % n_clusters;

    core::DeviceMemory<int> d_labels(static_cast<size_t>(n_cells));
    cudaMemcpyAsync(d_labels.get(), h_labels.data(),
                    static_cast<size_t>(n_cells) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return d_labels;
}

// ---------------------------------------------------------------------------
// bench_dendrogram — 2 warmup + 5 timed iterations of embed::dendrogram().
// Returns median wall_ms and peak mem_mb via out params.
// ---------------------------------------------------------------------------
static void bench_dendrogram(
    const io::PzDeviceMatrix& X,
    const int*                d_label,
    int                       k_clusters,
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

    embed::DendrogramConfig cfg{};

    for (int iter = 0; iter < WARMUP + TIMED; ++iter) {
        const bool is_timed = (iter >= WARMUP);
        if (is_timed) mem.sample_before();

        timer.start(stream);

        {
            auto result = embed::dendrogram(X, d_label, k_clusters, cfg, stream);
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
        std::puts("NO GPU — skipping Cycle 173 embed/dendrogram bench.");
        return 0;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // CSV header.
    std::printf("scale,n_cells,n_pcs,n_clusters,wall_ms,mem_mb\n");
    std::fflush(stdout);

    // Scales: 10k / 30k cells, 50 PCs as dense CSC, k=20 clusters.
    struct Scale { const char* name; int n_cells; int n_pcs; int k; };
    const Scale scales[] = {
        { "10k", 10000, 50, 20 },
        { "30k", 30000, 50, 20 },
    };

    for (const auto& s : scales) {
        std::printf("[bench] Synthesizing %s: %d genes(PCs) × %d cells "
                    "(dense CSC, round-robin k=%d labels)...\n",
                    s.name, s.n_pcs, s.n_cells, s.k);
        std::fflush(stdout);

        // Build dense CSC PzDeviceMatrix (untimed).
        io::PzDeviceMatrix X = make_dense_csc_pzmat(s.n_pcs, s.n_cells, stream);

        // Build round-robin labels (untimed).
        core::DeviceMemory<int> d_labels =
            make_roundrobin_labels(s.n_cells, s.k, stream);

        std::printf("[bench] Timing %s dendrogram() (k_clusters=%d)...\n",
                    s.name, s.k);
        std::fflush(stdout);

        double wall_ms = 0.0, mem_mb = 0.0;
        bench_dendrogram(X, d_labels.get(), s.k, stream, wall_ms, mem_mb);

        // CSV row.
        std::printf("%s,%d,%d,%d,%.3f,%.1f\n",
                    s.name, s.n_cells, s.n_pcs, s.k, wall_ms, mem_mb);
        std::fflush(stdout);

        // Registry row.
        bench::BenchRow row;
        row.date          = bench::today_iso();
        row.feature       = "embed/dendrogram";
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
    std::printf("[bench] Cycle 173 embed/dendrogram perf bench complete.\n");
    return 0;
}
