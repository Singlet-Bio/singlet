// SPDX-License-Identifier: MIT
// singlet/gpu/tests/qc_metrics_correctness.cpp
//
// Correctness harness for singlet::gpu::qc::calculate_qc_metrics(),
// filter_cells(), and filter_genes().
//
// Written against the design doc at:
//   singlet/gpu/state/designs/06b-qc-metrics.md
// NOT against the kernel source — authored in parallel with qc/metrics.h.
//
// Tolerances (from design doc):
//   n_umis  : bit-exact vs manual colsum (fp32 accumulation — same order)
//   n_genes : bit-exact vs manual col nnz
//   pct_mt  : relative error <= 1e-5 vs manual computation
//   pct_ribo: relative error <= 1e-5 vs manual computation
//   gene_mean: relative error <= 1e-5 vs numpy mean (all cells, including zeros)
//
// Test cases:
//   QcMetrics_TinySynthetic_NUmis       — n_umis == manual colsum
//   QcMetrics_TinySynthetic_NGenes      — n_genes == manual nnz
//   QcMetrics_TinySynthetic_PctMt       — pct_mt within 1e-5
//   QcMetrics_TinySynthetic_GeneMean    — gene_mean within 1e-5
//   QcMetrics_EdgeCase_EmptyColumn      — cells with 0 nnz handled correctly
//   QcMetrics_EdgeCase_ZeroVarianceGene — gene expressed uniformly has var=0
//   FilterCells_DimensionsCorrect       — filter retains expected #cells
//   FilterGenes_MinCells                — no gene with < min_cells_per_gene remains

#include <singlet/gpu/qc/metrics.h>
#include <singlet/gpu/core/types.h>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: is a GPU available?
// ---------------------------------------------------------------------------
static bool gpu_available() {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n > 0;
}

// ---------------------------------------------------------------------------
// Helper: build a random CSC on host with fixed seed.
//   n_genes x n_cells, density = prob of any entry being nonzero.
//   Values drawn from Poisson(lambda) approximated as truncated uniform [1,10].
// ---------------------------------------------------------------------------
struct HostCSC {
    int n_genes, n_cells, nnz;
    std::vector<int>   indptr;   // [n_cells+1]
    std::vector<int>   indices;  // [nnz]  — row ids
    std::vector<float> values;   // [nnz]
};

static HostCSC make_random_csc(int n_genes, int n_cells, float density,
                                uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::bernoulli_distribution bern(density);
    std::uniform_int_distribution<int> val_dist(1, 10);

    HostCSC h;
    h.n_genes = n_genes;
    h.n_cells = n_cells;
    h.indptr.resize(n_cells + 1, 0);

    for (int j = 0; j < n_cells; ++j) {
        for (int i = 0; i < n_genes; ++i) {
            if (bern(rng)) {
                h.indices.push_back(i);
                h.values.push_back((float)val_dist(rng));
            }
        }
        h.indptr[j + 1] = (int)h.indices.size();
    }
    h.nnz = (int)h.values.size();
    return h;
}

// ---------------------------------------------------------------------------
// Helper: upload HostCSC to a DeviceCSC.
// ---------------------------------------------------------------------------
static singlet::gpu::core::DeviceCSC upload(const HostCSC& h) {
    singlet::gpu::core::DeviceCSC d(h.n_genes, h.n_cells, h.nnz);
    if (h.nnz > 0) {
        d.col_ptr.upload(h.indptr.data(), h.n_cells + 1);
        d.row_indices.upload(h.indices.data(), h.nnz);
        d.values.upload(h.values.data(), h.nnz);
    } else {
        // upload indptr even for zero-nnz matrices
        d.col_ptr.upload(h.indptr.data(), h.n_cells + 1);
    }
    return d;
}

// ---------------------------------------------------------------------------
// Reference: manual column sums and nnz counts.
// ---------------------------------------------------------------------------
static std::vector<float> ref_col_sums(const HostCSC& h) {
    std::vector<float> s(h.n_cells, 0.0f);
    for (int j = 0; j < h.n_cells; ++j) {
        for (int k = h.indptr[j]; k < h.indptr[j + 1]; ++k)
            s[j] += h.values[k];
    }
    return s;
}

static std::vector<int> ref_col_nnz(const HostCSC& h) {
    std::vector<int> c(h.n_cells);
    for (int j = 0; j < h.n_cells; ++j)
        c[j] = h.indptr[j + 1] - h.indptr[j];
    return c;
}

static std::vector<float> ref_pct_mt(const HostCSC& h,
                                      const std::vector<uint8_t>& is_mt) {
    std::vector<float> r(h.n_cells, 0.0f);
    for (int j = 0; j < h.n_cells; ++j) {
        float total = 0, mt = 0;
        for (int k = h.indptr[j]; k < h.indptr[j + 1]; ++k) {
            total += h.values[k];
            if (is_mt[h.indices[k]]) mt += h.values[k];
        }
        r[j] = (total > 0.0f) ? (mt / total * 100.0f) : 0.0f;
    }
    return r;
}

static std::vector<float> ref_gene_mean(const HostCSC& h) {
    // Mean over ALL n_cells (including unexpressed zeros) — matches scanpy.
    std::vector<double> sums(h.n_genes, 0.0);
    for (int k = 0; k < h.nnz; ++k)
        sums[h.indices[k]] += h.values[k];
    std::vector<float> m(h.n_genes);
    for (int i = 0; i < h.n_genes; ++i)
        m[i] = (float)(sums[i] / h.n_cells);
    return m;
}

// ---------------------------------------------------------------------------
// Test: n_umis == manual column sum (fp32, same-order accumulation)
// ---------------------------------------------------------------------------
TEST(QcMetrics, TinySynthetic_NUmis) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU available";

    HostCSC h = make_random_csc(200, 500, 0.15f, 42);
    auto d = upload(h);

    // MT mask: genes 0-9, ribo mask: genes 10-19
    std::vector<uint8_t> is_mt_h(200, 0), is_ribo_h(200, 0);
    for (int i = 0; i < 10; ++i) is_mt_h[i] = 1;
    for (int i = 10; i < 20; ++i) is_ribo_h[i] = 1;

    singlet::gpu::core::DeviceMemory<uint8_t> d_is_mt(200), d_is_ribo(200);
    d_is_mt.upload(is_mt_h.data(), 200);
    d_is_ribo.upload(is_ribo_h.data(), 200);

    auto res = singlet::gpu::qc::calculate_qc_metrics(
        d.col_ptr.get(), d.row_indices.get(), d.values.get(),
        500, 200, d_is_mt.get(), d_is_ribo.get(), nullptr);

    cudaDeviceSynchronize();

    std::vector<float> h_n_umis(500);
    res.n_umis.download(h_n_umis.data(), 500);

    auto ref = ref_col_sums(h);
    for (int j = 0; j < 500; ++j) {
        // The kernel accumulates in the same fp32 order as the reference
        // (stride loop is deterministic for a single block). Exact match expected.
        EXPECT_FLOAT_EQ(h_n_umis[j], ref[j]) << "cell " << j;
    }
}

// ---------------------------------------------------------------------------
// Test: n_genes == manual nnz per column (integer, exact)
// ---------------------------------------------------------------------------
TEST(QcMetrics, TinySynthetic_NGenes) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU available";

    HostCSC h = make_random_csc(200, 500, 0.15f, 42);
    auto d = upload(h);

    std::vector<uint8_t> is_mt_h(200, 0), is_ribo_h(200, 0);
    singlet::gpu::core::DeviceMemory<uint8_t> d_is_mt(200), d_is_ribo(200);
    d_is_mt.upload(is_mt_h.data(), 200);
    d_is_ribo.upload(is_ribo_h.data(), 200);

    auto res = singlet::gpu::qc::calculate_qc_metrics(
        d.col_ptr.get(), d.row_indices.get(), d.values.get(),
        500, 200, d_is_mt.get(), d_is_ribo.get(), nullptr);

    cudaDeviceSynchronize();

    std::vector<int> h_n_genes(500);
    res.n_genes.download(h_n_genes.data(), 500);

    auto ref = ref_col_nnz(h);
    for (int j = 0; j < 500; ++j)
        EXPECT_EQ(h_n_genes[j], ref[j]) << "cell " << j;
}

// ---------------------------------------------------------------------------
// Test: pct_mt within 1e-5 relative error
// ---------------------------------------------------------------------------
TEST(QcMetrics, TinySynthetic_PctMt) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU available";

    HostCSC h = make_random_csc(200, 500, 0.15f, 42);
    auto d = upload(h);

    std::vector<uint8_t> is_mt_h(200, 0), is_ribo_h(200, 0);
    for (int i = 0; i < 10; ++i) is_mt_h[i] = 1;
    singlet::gpu::core::DeviceMemory<uint8_t> d_is_mt(200), d_is_ribo(200);
    d_is_mt.upload(is_mt_h.data(), 200);
    d_is_ribo.upload(is_ribo_h.data(), 200);

    auto res = singlet::gpu::qc::calculate_qc_metrics(
        d.col_ptr.get(), d.row_indices.get(), d.values.get(),
        500, 200, d_is_mt.get(), d_is_ribo.get(), nullptr);

    cudaDeviceSynchronize();

    std::vector<float> h_pct_mt(500);
    res.pct_mt.download(h_pct_mt.data(), 500);

    auto ref = ref_pct_mt(h, is_mt_h);
    constexpr float TOL = 1e-5f;
    for (int j = 0; j < 500; ++j) {
        float err = std::abs(h_pct_mt[j] - ref[j]);
        float denom = std::max(std::abs(ref[j]), 1e-6f);
        EXPECT_LE(err / denom, TOL) << "cell " << j
            << " got=" << h_pct_mt[j] << " ref=" << ref[j];
    }
}

// ---------------------------------------------------------------------------
// Test: gene_mean within 1e-5 relative error vs manual (fp64 reference)
// ---------------------------------------------------------------------------
TEST(QcMetrics, TinySynthetic_GeneMean) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU available";

    HostCSC h = make_random_csc(200, 500, 0.15f, 42);
    auto d = upload(h);

    std::vector<uint8_t> is_mt_h(200, 0), is_ribo_h(200, 0);
    singlet::gpu::core::DeviceMemory<uint8_t> d_is_mt(200), d_is_ribo(200);
    d_is_mt.upload(is_mt_h.data(), 200);
    d_is_ribo.upload(is_ribo_h.data(), 200);

    auto res = singlet::gpu::qc::calculate_qc_metrics(
        d.col_ptr.get(), d.row_indices.get(), d.values.get(),
        500, 200, d_is_mt.get(), d_is_ribo.get(), nullptr);

    cudaDeviceSynchronize();

    std::vector<float> h_gene_mean(200);
    res.gene_mean.download(h_gene_mean.data(), 200);

    auto ref = ref_gene_mean(h);
    constexpr float TOL = 1e-5f;
    for (int i = 0; i < 200; ++i) {
        float err   = std::abs(h_gene_mean[i] - ref[i]);
        float denom = std::max(std::abs(ref[i]), 1e-6f);
        EXPECT_LE(err / denom, TOL) << "gene " << i
            << " got=" << h_gene_mean[i] << " ref=" << ref[i];
    }
}

// ---------------------------------------------------------------------------
// Test: empty columns (cells with 0 expressed genes) produce:
//   n_umis=0, n_genes=0, pct_mt=0, pct_ribo=0
// ---------------------------------------------------------------------------
TEST(QcMetrics, EdgeCase_EmptyColumn) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU available";

    // 10 genes, 5 cells; only cell 2 has data.
    HostCSC h;
    h.n_genes = 10;
    h.n_cells = 5;
    h.indptr  = {0, 0, 0, 3, 3, 3};  // cells 0,1,3,4 are empty
    h.indices = {0, 4, 7};
    h.values  = {2.0f, 5.0f, 1.0f};
    h.nnz     = 3;
    auto d = upload(h);

    std::vector<uint8_t> is_mt_h(10, 0), is_ribo_h(10, 0);
    is_mt_h[0] = 1;  // gene 0 is MT
    singlet::gpu::core::DeviceMemory<uint8_t> d_is_mt(10), d_is_ribo(10);
    d_is_mt.upload(is_mt_h.data(), 10);
    d_is_ribo.upload(is_ribo_h.data(), 10);

    auto res = singlet::gpu::qc::calculate_qc_metrics(
        d.col_ptr.get(), d.row_indices.get(), d.values.get(),
        5, 10, d_is_mt.get(), d_is_ribo.get(), nullptr);

    cudaDeviceSynchronize();

    std::vector<float> h_n_umis(5);
    std::vector<int>   h_n_genes(5);
    std::vector<float> h_pct_mt(5);
    res.n_umis.download(h_n_umis.data(), 5);
    res.n_genes.download(h_n_genes.data(), 5);
    res.pct_mt.download(h_pct_mt.data(), 5);

    // Empty cells
    for (int j : {0, 1, 3, 4}) {
        EXPECT_FLOAT_EQ(h_n_umis[j],  0.0f) << "empty cell " << j << " n_umis";
        EXPECT_EQ      (h_n_genes[j], 0)    << "empty cell " << j << " n_genes";
        EXPECT_FLOAT_EQ(h_pct_mt[j],  0.0f) << "empty cell " << j << " pct_mt";
    }
    // Cell 2: 2+5+1=8, pct_mt = 2/8*100 = 25
    EXPECT_FLOAT_EQ(h_n_umis[2], 8.0f);
    EXPECT_EQ      (h_n_genes[2], 3);
    EXPECT_NEAR    (h_pct_mt[2], 25.0f, 1e-4f);
}

// ---------------------------------------------------------------------------
// Test: gene expressed at the same value in every cell → variance = 0
// ---------------------------------------------------------------------------
TEST(QcMetrics, EdgeCase_ZeroVarianceGene) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU available";

    // 3 genes, 4 cells; gene 1 is expressed at value=3 in every cell.
    HostCSC h;
    h.n_genes = 3;
    h.n_cells = 4;
    // CSC layout: for each cell, one nnz for gene 1 with value 3.
    h.indptr  = {0, 1, 2, 3, 4};
    h.indices = {1, 1, 1, 1};
    h.values  = {3.0f, 3.0f, 3.0f, 3.0f};
    h.nnz     = 4;
    auto d = upload(h);

    std::vector<uint8_t> is_mt_h(3, 0), is_ribo_h(3, 0);
    singlet::gpu::core::DeviceMemory<uint8_t> d_is_mt(3), d_is_ribo(3);
    d_is_mt.upload(is_mt_h.data(), 3);
    d_is_ribo.upload(is_ribo_h.data(), 3);

    auto res = singlet::gpu::qc::calculate_qc_metrics(
        d.col_ptr.get(), d.row_indices.get(), d.values.get(),
        4, 3, d_is_mt.get(), d_is_ribo.get(), nullptr);

    cudaDeviceSynchronize();

    std::vector<float> h_gene_var(3);
    res.gene_var.download(h_gene_var.data(), 3);

    // Gene 1: expressed uniformly at 3 → var should be 0.
    EXPECT_NEAR(h_gene_var[1], 0.0f, 1e-5f) << "uniform gene should have ~0 variance";
    // Genes 0 and 2: never expressed → mean=0, var=0.
    EXPECT_NEAR(h_gene_var[0], 0.0f, 1e-5f);
    EXPECT_NEAR(h_gene_var[2], 0.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Test: filter_cells retains expected number of cells
// ---------------------------------------------------------------------------
TEST(FilterCells, DimensionsCorrect) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU available";

    // 50 genes, 100 cells; density 0.2 (about 10 genes/cell on average).
    HostCSC h = make_random_csc(50, 100, 0.2f, 7);
    auto d = upload(h);

    std::vector<uint8_t> is_mt_h(50, 0), is_ribo_h(50, 0);
    for (int i = 0; i < 5; ++i) is_mt_h[i] = 1;
    singlet::gpu::core::DeviceMemory<uint8_t> d_is_mt(50), d_is_ribo(50);
    d_is_mt.upload(is_mt_h.data(), 50);
    d_is_ribo.upload(is_ribo_h.data(), 50);

    auto qc = singlet::gpu::qc::calculate_qc_metrics(
        d.col_ptr.get(), d.row_indices.get(), d.values.get(),
        100, 50, d_is_mt.get(), d_is_ribo.get(), nullptr);

    cudaDeviceSynchronize();

    // Count how many cells should pass the filter manually.
    std::vector<int>   h_n_genes(100);
    std::vector<float> h_n_umis(100), h_pct_mt(100);
    qc.n_genes.download(h_n_genes.data(), 100);
    qc.n_umis.download(h_n_umis.data(), 100);
    qc.pct_mt.download(h_pct_mt.data(), 100);

    singlet::gpu::qc::FilterConfig cfg;
    cfg.min_genes = 5;
    cfg.max_pct_mt = 50.0f;

    int expected_kept = 0;
    for (int j = 0; j < 100; ++j) {
        if ((float)h_n_genes[j] >= cfg.min_genes && h_pct_mt[j] <= cfg.max_pct_mt)
            ++expected_kept;
    }

    auto out = singlet::gpu::qc::filter_cells(d, qc, cfg, nullptr);

    EXPECT_EQ(out.cols, expected_kept)
        << "filter_cells kept wrong number of columns";
    EXPECT_EQ(out.rows, d.rows)
        << "filter_cells must not change row count";
}

// ---------------------------------------------------------------------------
// Test: filter_genes — no gene with fewer than min_cells_per_gene expressing
//       cells remains in the output.
// ---------------------------------------------------------------------------
TEST(FilterGenes, MinCells) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU available";

    HostCSC h = make_random_csc(50, 100, 0.2f, 13);
    auto d = upload(h);

    std::vector<uint8_t> is_mt_h(50, 0), is_ribo_h(50, 0);
    singlet::gpu::core::DeviceMemory<uint8_t> d_is_mt(50), d_is_ribo(50);
    d_is_mt.upload(is_mt_h.data(), 50);
    d_is_ribo.upload(is_ribo_h.data(), 50);

    auto qc = singlet::gpu::qc::calculate_qc_metrics(
        d.col_ptr.get(), d.row_indices.get(), d.values.get(),
        100, 50, d_is_mt.get(), d_is_ribo.get(), nullptr);

    cudaDeviceSynchronize();

    // Download gene_n_cells for reference count.
    std::vector<int> h_gene_n_cells(50);
    qc.gene_n_cells.download(h_gene_n_cells.data(), 50);

    constexpr int MIN_CELLS = 10;
    int expected_genes = 0;
    for (int i = 0; i < 50; ++i)
        if (h_gene_n_cells[i] >= MIN_CELLS) ++expected_genes;

    singlet::gpu::qc::FilterConfig cfg;
    cfg.min_cells_per_gene = MIN_CELLS;

    auto out = singlet::gpu::qc::filter_genes(d, qc, cfg, nullptr);

    EXPECT_EQ(out.rows, expected_genes)
        << "filter_genes kept wrong number of genes";
    EXPECT_EQ(out.cols, d.cols)
        << "filter_genes must not change column count";
}
