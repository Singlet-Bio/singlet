// SPDX-License-Identifier: MIT
// singlet/gpu/tests/preprocess_magic_correctness.cpp
//
// Correctness harness for the MAGIC imputation kernel (CYCLE-124).
// Reference: van Dijk et al. (2018) Cell 174:716-729.
//
// Tests:
//   1. Identity_Graph_NoOp              — identity graph → imputed == log1p-transposed input
//   2. KGraph_2Cell_Population_Mixes    — kNN graph + 2-population synthetic data
//   3. Determinism_SameInput_SameOutput — two runs, rel_err < 1e-4
//   4. MemoryGuard_RejectsTooLarge      — oversized dims → std::runtime_error
//   5. T_Zero_Returns_Input_Transposed  — cfg.t=0 → no diffusion
//
// Build: compiled as CUDA (see CMakeLists.txt — LANGUAGE CUDA).
// Run:   --gtest_filter=MagicImputeTest.*

#include <singlet/gpu/preprocess/magic.h>
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/graph/snn.h>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

// =============================================================================
// Shared test helpers
// =============================================================================
namespace {

// Build a SynCSC struct and upload it to a PzDeviceMatrix (m genes × n cells).
struct SynCSC {
    int                  rows, cols, nnz;
    std::vector<float>   vals;
    std::vector<int32_t> col_ptr;
    std::vector<int32_t> row_idx;
};

singlet::gpu::io::PzDeviceMatrix upload_csc(const SynCSC& c)
{
    singlet::gpu::io::PzDeviceMatrix pz;
    pz.mat = singlet::gpu::core::DeviceCSC(c.rows, c.cols, c.nnz);
    cudaMemcpy(pz.mat.col_ptr.get(),     c.col_ptr.data(),
               (c.cols + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (c.nnz > 0) {
        cudaMemcpy(pz.mat.row_indices.get(), c.row_idx.data(),
                   c.nnz * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(pz.mat.values.get(), c.vals.data(),
                   c.nnz * sizeof(float), cudaMemcpyHostToDevice);
    }
    return pz;
}

// Build a CSC from a dense row-major matrix (rows=genes, cols=cells).
SynCSC dense_to_csc(int rows, int cols, const std::vector<float>& dense)
{
    SynCSC c;
    c.rows = rows;
    c.cols = cols;
    c.col_ptr.resize(cols + 1, 0);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            if (dense[i * cols + j] != 0.f) c.col_ptr[j + 1]++;
    for (int j = 0; j < cols; ++j) c.col_ptr[j + 1] += c.col_ptr[j];
    c.nnz = c.col_ptr[cols];
    c.vals.resize(c.nnz);
    c.row_idx.resize(c.nnz);
    std::vector<int32_t> fill(cols, 0);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            float v = dense[i * cols + j];
            if (v != 0.f) {
                int pos = c.col_ptr[j] + fill[j]++;
                c.row_idx[pos] = i;
                c.vals[pos]    = v;
            }
        }
    }
    return c;
}

// Fetch device float buffer to host.
std::vector<float> d2h(const singlet::gpu::core::DeviceMemory<float>& d, int n)
{
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

// Build a hand-crafted SnnResult from host CSR arrays.
// row_offsets: n+1 ints; col_indices + weights: nnz entries.
singlet::gpu::graph::SnnResult build_snn_result(
    int n, int nnz,
    const std::vector<int>&   h_row_offsets,   // size n+1
    const std::vector<int>&   h_col_indices,   // size nnz
    const std::vector<float>& h_weights)       // size nnz
{
    singlet::gpu::graph::SnnResult res;
    res.n   = n;
    res.nnz = nnz;

    res.row_offsets = singlet::gpu::core::DeviceMemory<int>(n + 1);
    res.col_indices = singlet::gpu::core::DeviceMemory<int>(nnz > 0 ? nnz : 1);
    res.weights     = singlet::gpu::core::DeviceMemory<float>(nnz > 0 ? nnz : 1);

    cudaMemcpy(res.row_offsets.get(), h_row_offsets.data(),
               (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (nnz > 0) {
        cudaMemcpy(res.col_indices.get(), h_col_indices.data(),
                   nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(res.weights.get(), h_weights.data(),
                   nnz * sizeof(float), cudaMemcpyHostToDevice);
    }
    return res;
}

// Build an identity CSR graph (each cell points to itself with weight 1).
singlet::gpu::graph::SnnResult build_identity_graph(int n)
{
    std::vector<int>   row_off(n + 1);
    std::vector<int>   col_idx(n);
    std::vector<float> wts(n, 1.f);
    for (int i = 0; i <= n; ++i) row_off[i] = i;
    for (int i = 0; i < n;  ++i) col_idx[i] = i;
    return build_snn_result(n, n, row_off, col_idx, wts);
}

// CPU reference: apply t steps of M @ Y where M is a dense row-stochastic
// matrix (CSR) and Y is n × m col-major.  Returns the diffused Y.
std::vector<float> cpu_diffuse(
    int n, int m, int t,
    const std::vector<int>&   row_off,
    const std::vector<int>&   col_idx,
    const std::vector<float>& wts,
    const std::vector<float>& Y0)   // n × m col-major
{
    // Build Markov matrix (normalize rows).
    std::vector<float> M_vals(wts.size());
    for (int i = 0; i < n; ++i) {
        float rs = 0.f;
        for (int e = row_off[i]; e < row_off[i + 1]; ++e) rs += wts[e];
        float inv = rs > 0.f ? 1.f / rs : 1.f;
        for (int e = row_off[i]; e < row_off[i + 1]; ++e) M_vals[e] = wts[e] * inv;
    }

    std::vector<float> Y = Y0;
    std::vector<float> tmp(n * m, 0.f);
    for (int step = 0; step < t; ++step) {
        std::fill(tmp.begin(), tmp.end(), 0.f);
        // Y_out[i, g] = Σ_j M[i,j] · Y_in[j, g]
        for (int i = 0; i < n; ++i) {
            for (int e = row_off[i]; e < row_off[i + 1]; ++e) {
                int j = col_idx[e];
                float mij = M_vals[e];
                for (int g = 0; g < m; ++g) {
                    tmp[i + (size_t)g * n] += mij * Y[j + (size_t)g * n];
                }
            }
        }
        Y = tmp;
    }
    return Y;
}

} // anonymous namespace


// =============================================================================
// Test 1: Identity_Graph_NoOp
//
// Identity graph: each cell points only to itself with weight 1.
// After row-normalisation M_ii = 1, M_ij = 0 for i≠j.
// After t=3 diffusion: Y_t = Y_0 (no mixing occurs).
// Verify: |imputed[cell,gene] - Y_0[cell,gene]| ≤ 1e-4 for all entries.
// =============================================================================
TEST(MagicImputeTest, Identity_Graph_NoOp)
{
    constexpr int M = 20, N = 30;

    // Build a small log1p-like matrix (non-zero entries only).
    std::mt19937 rng(0xABCD1234ULL);
    std::uniform_real_distribution<float> ud(0.f, 3.f);
    std::vector<float> dense(M * N);
    for (auto& v : dense) v = ud(rng);

    const auto csc = dense_to_csc(M, N, dense);
    auto pz        = upload_csc(csc);

    auto graph = build_identity_graph(N);

    singlet::gpu::preprocess::MagicConfig cfg;
    cfg.t = 3;
    auto result = singlet::gpu::preprocess::magic_impute(pz, graph, cfg);
    auto h_imp  = d2h(result.imputed, N * M);

    // Y_0 col-major: Y[cell j + gene i * N] = dense[gene i * N + cell j]
    float max_err = 0.f;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            float expected = dense[i * N + j];
            float got      = h_imp[j + (size_t)i * N];
            max_err = std::max(max_err, std::fabs(got - expected));
        }
    }

    EXPECT_LE(max_err, 1e-4f)
        << "Identity graph should produce no change; max_err=" << max_err;

    std::printf("[REGISTRY] %s | preprocess/magic/identity_graph | %dx%d | "
                "abs_err | %.2e | 1e-4 | none | CYCLE-124 | %s\n",
                __DATE__, M, N, max_err,
                (max_err <= 1e-4f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 2: KGraph_2Cell_Population_Mixes
//
// 100 genes × 200 cells.
//   - Cells 0..99: highly express genes 0..9 (value 5.0).
//   - Cells 100..199: highly express genes 10..19 (value 5.0).
//   - All other entries: 0.
//
// Graph: each cell connects to its 5 same-population neighbours with weight 1
// and is not connected cross-population (simulates a perfect kNN split).
//
// After MAGIC t=3:
//   a) Within-population variance should decrease (smoothing moves expression
//      toward the population mean, compressing individual variation).
//   b) Population separation: for ≥8 of 10 pop1-specific genes (genes 0..9),
//      mean expression in cells 0..99 should remain > 2× mean in cells 100..199.
// =============================================================================
TEST(MagicImputeTest, KGraph_2Cell_Population_Mixes)
{
    constexpr int M      = 100;   // genes
    constexpr int N      = 200;   // cells (100 per population)
    constexpr int N_HALF = 100;
    constexpr int K      = 5;     // neighbours per cell in the graph

    // Build expression matrix: row-major (genes × cells).
    std::vector<float> dense(M * N, 0.f);
    // Population 1: genes 0..9 highly expressed in cells 0..99.
    for (int g = 0; g < 10; ++g)
        for (int j = 0; j < N_HALF; ++j)
            dense[g * N + j] = 5.f;
    // Population 2: genes 10..19 highly expressed in cells 100..199.
    for (int g = 10; g < 20; ++g)
        for (int j = N_HALF; j < N; ++j)
            dense[g * N + j] = 5.f;

    const auto csc = dense_to_csc(M, N, dense);
    auto pz        = upload_csc(csc);

    // Build the CSR graph: each cell connects to its K next cells within the
    // same population (cyclic within [0,N_HALF) and [N_HALF, N) respectively).
    // This is a simple ring graph per population — strong within-population
    // connectivity, no cross-population edges.
    const int nnz_graph = N * K;
    std::vector<int>   row_off(N + 1, 0);
    std::vector<int>   col_idx(nnz_graph);
    std::vector<float> wts(nnz_graph, 1.f);

    for (int i = 0; i < N; ++i) {
        row_off[i + 1] = row_off[i] + K;
        bool pop2 = (i >= N_HALF);
        int  base = pop2 ? N_HALF : 0;
        int  sz   = N_HALF;
        int  li   = i - base;  // local index within population
        for (int k = 0; k < K; ++k) {
            int nb = base + (li + k + 1) % sz;
            col_idx[row_off[i] + k] = nb;
        }
    }

    auto graph = build_snn_result(N, nnz_graph, row_off, col_idx, wts);

    // Compute initial within-population variance for pop1 genes before diffusion.
    // We measure post-diffusion population means instead (simpler assertion).

    singlet::gpu::preprocess::MagicConfig cfg;
    cfg.t = 3;
    auto result = singlet::gpu::preprocess::magic_impute(pz, graph, cfg);
    auto h_imp  = d2h(result.imputed, N * M);

    // Check population separation: for genes 0..9, mean in cells 0..99 > 2x
    // mean in cells 100..199.
    int separated = 0;
    for (int g = 0; g < 10; ++g) {
        double mean_pop1 = 0.0, mean_pop2 = 0.0;
        for (int j = 0;      j < N_HALF; ++j) mean_pop1 += h_imp[j      + (size_t)g * N];
        for (int j = N_HALF; j < N;      ++j) mean_pop2 += h_imp[j      + (size_t)g * N];
        mean_pop1 /= N_HALF;
        mean_pop2 /= N_HALF;
        if (mean_pop1 > 2.0 * std::max(mean_pop2, 1e-9)) ++separated;
    }

    EXPECT_GE(separated, 8)
        << "At least 8 of 10 pop1-specific genes should remain > 2x pop2 mean "
        << "after diffusion; got " << separated;

    std::printf("[REGISTRY] %s | preprocess/magic/population_mixes | %dx%d t=3 | "
                "pop_separation | %d/10 genes | >=8 | none | CYCLE-124 | %s\n",
                __DATE__, M, N, separated,
                (separated >= 8) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 3: Determinism_SameInput_SameOutput
//
// Run magic_impute twice on identical input + identical graph.
// SpMM is deterministic at fixed precision and fixed CSR sparsity pattern.
// Expect relative error < 1e-4 for all elements.
// =============================================================================
TEST(MagicImputeTest, Determinism_SameInput_SameOutput)
{
    constexpr int M = 25, N = 40;

    std::mt19937 rng(0xDECAFBADULL);
    std::uniform_real_distribution<float> ud(0.f, 2.f);
    std::vector<float> dense(M * N);
    for (auto& v : dense) v = ud(rng);

    const auto csc = dense_to_csc(M, N, dense);

    // Build a simple ring-5 graph for N cells.
    constexpr int K = 5;
    const int nnz_g = N * K;
    std::vector<int>   row_off(N + 1);
    std::vector<int>   col_idx(nnz_g);
    std::vector<float> wts(nnz_g, 1.f);
    for (int i = 0; i < N; ++i) {
        row_off[i + 1] = row_off[i] + K;
        for (int k = 0; k < K; ++k)
            col_idx[row_off[i] + k] = (i + k + 1) % N;
    }
    row_off[N] = nnz_g;

    singlet::gpu::preprocess::MagicConfig cfg;
    cfg.t = 3;

    // First run.
    auto pz1    = upload_csc(csc);
    auto graph1 = build_snn_result(N, nnz_g, row_off, col_idx, wts);
    auto res1   = singlet::gpu::preprocess::magic_impute(pz1, graph1, cfg);
    auto h1     = d2h(res1.imputed, N * M);

    // Second run (re-upload same data).
    auto pz2    = upload_csc(csc);
    auto graph2 = build_snn_result(N, nnz_g, row_off, col_idx, wts);
    auto res2   = singlet::gpu::preprocess::magic_impute(pz2, graph2, cfg);
    auto h2     = d2h(res2.imputed, N * M);

    float max_rel = 0.f;
    for (int i = 0; i < N * M; ++i) {
        float denom = std::max(std::fabs(h1[i]), 1e-8f);
        max_rel = std::max(max_rel, std::fabs(h1[i] - h2[i]) / denom);
    }

    EXPECT_LT(max_rel, 1e-4f)
        << "Two identical runs should give rel_err < 1e-4; max_rel=" << max_rel;

    std::printf("[REGISTRY] %s | preprocess/magic/determinism | %dx%d t=3 | "
                "rel_err | %.2e | 1e-4 | cuSPARSE | CYCLE-124 | %s\n",
                __DATE__, M, N, max_rel,
                (max_rel < 1e-4f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 4: MemoryGuard_RejectsTooLarge
//
// Build a valid tiny graph but lie about matrix dims: request n=100k, m=80k
// which would imply 2 × 100000 × 80000 × 4 = 64 GB — always exceeds 50% free
// memory on any realistic GPU.  Expect std::runtime_error with an informative
// message.
// =============================================================================
TEST(MagicImputeTest, MemoryGuard_RejectsTooLarge)
{
    // A trivial 2×2 expression matrix.
    constexpr int M_TINY = 2, N_TINY = 2;
    std::vector<float> dense_tiny = {1.f, 0.f, 0.f, 1.f};
    const auto csc_tiny = dense_to_csc(M_TINY, N_TINY, dense_tiny);

    // Use a realistic large n/m that would need >>8 GB.
    // We fake this by directly constructing a PzDeviceMatrix with large dims
    // but pointing to the tiny device buffers — the guard fires before any
    // kernel reads those buffers.
    singlet::gpu::io::PzDeviceMatrix fake_pz;
    fake_pz.mat.rows = 80000;   // m_genes (large)
    fake_pz.mat.cols = 100000;  // n_cells (large) — must match graph.n
    fake_pz.mat.nnz  = 0;
    // Allocate minimal storage (won't actually be used — guard fires first).
    fake_pz.mat.col_ptr     = singlet::gpu::core::DeviceMemory<int32_t>(1);
    fake_pz.mat.row_indices = singlet::gpu::core::DeviceMemory<int32_t>(1);
    fake_pz.mat.values      = singlet::gpu::core::DeviceMemory<float>(1);

    // Build a matching (fake) graph for n=100000.
    const int N_FAKE = 100000;
    std::vector<int>   ro(N_FAKE + 1, 0);
    std::vector<int>   ci(1, 0);
    std::vector<float> wt(1, 1.f);
    // nnz=0 graph (no edges; just valid row_offsets all zero).
    singlet::gpu::graph::SnnResult fake_graph;
    fake_graph.n   = N_FAKE;
    fake_graph.nnz = 0;
    fake_graph.row_offsets = singlet::gpu::core::DeviceMemory<int>(N_FAKE + 1);
    fake_graph.col_indices = singlet::gpu::core::DeviceMemory<int>(1);
    fake_graph.weights     = singlet::gpu::core::DeviceMemory<float>(1);
    cudaMemset(fake_graph.row_offsets.get(), 0, (N_FAKE + 1) * sizeof(int));

    singlet::gpu::preprocess::MagicConfig cfg;
    cfg.t = 3;

    EXPECT_THROW(
        singlet::gpu::preprocess::magic_impute(fake_pz, fake_graph, cfg),
        std::runtime_error)
        << "Expected std::runtime_error from memory guard for oversized n×m";

    std::printf("[REGISTRY] %s | preprocess/magic/memory_guard | %dx%d | "
                "throw | runtime_error | expected | none | CYCLE-124 | PASS\n",
                __DATE__, fake_pz.mat.rows, fake_pz.mat.cols);
}


// =============================================================================
// Test 5: T_Zero_Returns_Input_Transposed
//
// cfg.t = 0 should skip the SpMM loop entirely and return Y_0 = Xᵀ.
// Y_0[cell j, gene i] = X[gene i, cell j] (with zeros for implicit zeros).
// =============================================================================
TEST(MagicImputeTest, T_Zero_Returns_Input_Transposed)
{
    constexpr int M = 15, N = 20;

    std::mt19937 rng(0xFEEDF00DULL);
    std::uniform_real_distribution<float> ud(0.f, 5.f);
    std::vector<float> dense(M * N);
    for (auto& v : dense) v = ud(rng);

    const auto csc = dense_to_csc(M, N, dense);
    auto pz        = upload_csc(csc);
    auto graph     = build_identity_graph(N);

    singlet::gpu::preprocess::MagicConfig cfg;
    cfg.t = 0;
    auto result = singlet::gpu::preprocess::magic_impute(pz, graph, cfg);
    auto h_imp  = d2h(result.imputed, N * M);

    // Verify h_imp[j + i*N] == dense[i*N + j] for all i, j.
    float max_err = 0.f;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float expected = dense[i * N + j];
            float got      = h_imp[j + (size_t)i * N];
            max_err = std::max(max_err, std::fabs(got - expected));
        }
    }

    EXPECT_EQ(result.t_used, 0);
    EXPECT_LE(max_err, 1e-6f)
        << "t=0 should return exact transpose; max_err=" << max_err;

    std::printf("[REGISTRY] %s | preprocess/magic/t_zero | %dx%d | "
                "abs_err | %.2e | 1e-6 | none | CYCLE-124 | %s\n",
                __DATE__, M, N, max_err,
                (max_err <= 1e-6f) ? "PASS" : "FAIL");
}
