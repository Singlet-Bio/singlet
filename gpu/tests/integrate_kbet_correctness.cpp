// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/integrate_kbet_correctness.cpp
//
// Correctness harness for the kBET kernel (CYCLE-140).
// Reference: Buttner M, Miao Z, Wolf FA, et al. (2019) Nat Methods 16:43-49.
//
// Tests:
//   1. Kbet_PerfectMixing_LowChi2         — balanced kNN → chi2 ≈ 0, reject_rate ≈ 0
//   2. Kbet_FullSegregation_HighChi2      — segregated kNN → chi2 large, reject_rate ≈ 1
//   3. Kbet_FourBatchesBalanced_LowChi2   — 4-batch perfect balance → chi2 ≈ 0
//   4. Kbet_Determinism_BitIdentical      — same input → bit-identical output
//   5. Kbet_RejectRate_Monotonic          — increasing segregation → reject_rate non-decreasing
//
// Build: compiled as CUDA (see CMakeLists.txt — LANGUAGE CUDA).
// Run:   --gtest_filter=KbetTest.*

#include <singlet-gpu/integrate/kbet.h>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

// =============================================================================
// Shared helpers
// =============================================================================
namespace {

// Upload a host int array to device memory.
singlet_gpu::core::DeviceMemory<int> upload_int(const std::vector<int>& h)
{
    singlet_gpu::core::DeviceMemory<int> d(h.size());
    cudaMemcpy(d.get(), h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice);
    return d;
}

// Build a KnnResult by hand from host arrays (no distances needed for kBET).
singlet_gpu::graph::KnnResult build_knn_result(
    int n, int k,
    const std::vector<int>& h_row_offsets,
    const std::vector<int>& h_neighbors)
{
    singlet_gpu::graph::KnnResult res;
    res.n = n;
    res.k = k;
    res.backend_used = singlet_gpu::graph::KnnBackend::Exact;

    res.row_offsets = singlet_gpu::core::DeviceMemory<int>(n + 1);
    res.neighbors   = singlet_gpu::core::DeviceMemory<int>(static_cast<size_t>(n) * k);
    // distances not required by kBET — leave as default (empty DeviceMemory).

    cudaMemcpy(res.row_offsets.get(), h_row_offsets.data(),
               (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(res.neighbors.get(), h_neighbors.data(),
               static_cast<size_t>(n) * k * sizeof(int), cudaMemcpyHostToDevice);
    return res;
}

// Build uniform row_offsets: offsets[i] = i*k.
std::vector<int> make_row_offsets(int n, int k)
{
    std::vector<int> off(n + 1);
    for (int i = 0; i <= n; ++i) off[i] = i * k;
    return off;
}

// Fetch device float buffer to host.
std::vector<float> d2h_float(const singlet_gpu::core::DeviceMemory<float>& d, int n)
{
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d.get(), static_cast<size_t>(n) * sizeof(float),
               cudaMemcpyDeviceToHost);
    return h;
}

} // anonymous namespace


// =============================================================================
// Test 1: Kbet_PerfectMixing_LowChi2
//
// 100 cells, k=10, 2 batches each at 50% global.
// Each cell's k=10 neighbors: 5 from batch 0, 5 from batch 1.
// n_exp[0] = 10 * 50/100 = 5.0; n_exp[1] = 5.0.
// chi2[c] = (5-5)^2/5 + (5-5)^2/5 = 0.
// Expected: mean_chi2 ≈ 0, reject_rate ≈ 0.
// =============================================================================
TEST(KbetTest, Kbet_PerfectMixing_LowChi2)
{
    constexpr int N = 100, K = 10, N_BATCHES = 2;

    // Batch labels: cells 0..49 = batch 0, cells 50..99 = batch 1.
    std::vector<int> h_batch(N);
    for (int c = 0; c < N; ++c) h_batch[c] = (c < N / 2) ? 0 : 1;

    // Each cell: 5 neighbors from batch 0 (cells 0..4), 5 from batch 1 (cells 50..54).
    std::vector<int> h_nbr(static_cast<size_t>(N) * K);
    for (int c = 0; c < N; ++c) {
        for (int j = 0; j < 5; ++j) h_nbr[c * K + j]     = j;       // batch 0
        for (int j = 0; j < 5; ++j) h_nbr[c * K + 5 + j] = 50 + j; // batch 1
    }

    auto knn    = build_knn_result(N, K, make_row_offsets(N, K), h_nbr);
    auto d_bat  = upload_int(h_batch);

    singlet_gpu::integrate::KbetConfig cfg;
    cfg.deterministic = true;
    auto res = singlet_gpu::integrate::kbet(knn, d_bat.get(), N_BATCHES, cfg);

    EXPECT_NEAR(res.mean_chi2,   0.f, 0.01f)
        << "Perfect mixing: mean_chi2 should be ~0; got " << res.mean_chi2;
    EXPECT_NEAR(res.reject_rate, 0.f, 0.05f)
        << "Perfect mixing: reject_rate should be ~0; got " << res.reject_rate;

    auto h_chi2 = d2h_float(res.chi2, N);
    float max_chi2 = *std::max_element(h_chi2.begin(), h_chi2.end());
    EXPECT_NEAR(max_chi2, 0.f, 0.01f)
        << "Perfect mixing: all per-cell chi2 should be ~0; max=" << max_chi2;

    std::printf("[REGISTRY] %s | integrate/kbet/perfect_mixing | n=%d k=%d batches=%d | "
                "mean_chi2 | %.4f | 0.01 | none | CYCLE-140 | %s\n",
                __DATE__, N, K, N_BATCHES, res.mean_chi2,
                (res.mean_chi2 < 0.01f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 2: Kbet_FullSegregation_HighChi2
//
// 100 cells, k=10, 2 batches at 50% global.
// Each cell's k=10 neighbors: all from the SAME batch as the cell.
// n_obs[own]=10, n_obs[other]=0; n_exp[own]=5, n_exp[other]=5.
// chi2[c] = (10-5)^2/5 + (0-5)^2/5 = 25/5 + 25/5 = 10.
// df=1; chi2_crit(0.95, df=1) = 3.84; 10 >> 3.84 → all reject.
// Expected: mean_chi2 ≈ 10, reject_rate ≈ 1.
// =============================================================================
TEST(KbetTest, Kbet_FullSegregation_HighChi2)
{
    constexpr int N = 100, K = 10, N_BATCHES = 2;

    std::vector<int> h_batch(N);
    for (int c = 0; c < N; ++c) h_batch[c] = (c < N / 2) ? 0 : 1;

    // Each cell: all 10 neighbors from own batch.
    std::vector<int> h_nbr(static_cast<size_t>(N) * K);
    for (int c = 0; c < N; ++c) {
        int anchor = (c < N / 2) ? 0 : 50;
        for (int j = 0; j < K; ++j) h_nbr[c * K + j] = anchor + j;
    }

    auto knn   = build_knn_result(N, K, make_row_offsets(N, K), h_nbr);
    auto d_bat = upload_int(h_batch);

    singlet_gpu::integrate::KbetConfig cfg;
    cfg.deterministic = true;
    auto res = singlet_gpu::integrate::kbet(knn, d_bat.get(), N_BATCHES, cfg);

    // chi2 per cell = 10; n_exp[b]=5 so denom = max(5,1)=5.
    // chi2 = (10-5)^2/5 + (0-5)^2/5 = 5 + 5 = 10.
    EXPECT_NEAR(res.mean_chi2, 10.f, 0.5f)
        << "Full segregation: mean_chi2 should be ~10; got " << res.mean_chi2;
    EXPECT_GT(res.reject_rate, 0.9f)
        << "Full segregation: reject_rate should be ~1; got " << res.reject_rate;

    std::printf("[REGISTRY] %s | integrate/kbet/full_segregation | n=%d k=%d batches=%d | "
                "reject_rate | %.4f | 0.9 | none | CYCLE-140 | %s\n",
                __DATE__, N, K, N_BATCHES, res.reject_rate,
                (res.reject_rate > 0.9f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 3: Kbet_FourBatchesBalanced_LowChi2
//
// 200 cells, k=20, 4 batches at 25% global (50 cells each).
// Each cell: 5 neighbors per batch (exactly balanced).
// n_exp[b] = 20 * 50/200 = 5.0 for each b.
// chi2[c] = Σ_4 (5-5)^2/5 = 0.
// Expected: mean_chi2 ≈ 0, reject_rate ≈ 0.
// =============================================================================
TEST(KbetTest, Kbet_FourBatchesBalanced_LowChi2)
{
    constexpr int N = 200, K = 20, N_BATCHES = 4;
    constexpr int CELLS_PER_BATCH = N / N_BATCHES;  // 50

    std::vector<int> h_batch(N);
    for (int c = 0; c < N; ++c) h_batch[c] = c / CELLS_PER_BATCH;

    // Each cell: 5 neighbors from each batch.
    // Batch b starts at cell b * CELLS_PER_BATCH.
    std::vector<int> h_nbr(static_cast<size_t>(N) * K);
    for (int c = 0; c < N; ++c) {
        for (int b = 0; b < N_BATCHES; ++b) {
            int anchor = b * CELLS_PER_BATCH;
            for (int j = 0; j < 5; ++j) {
                h_nbr[c * K + b * 5 + j] = anchor + j;
            }
        }
    }

    auto knn   = build_knn_result(N, K, make_row_offsets(N, K), h_nbr);
    auto d_bat = upload_int(h_batch);

    singlet_gpu::integrate::KbetConfig cfg;
    cfg.deterministic = true;
    auto res = singlet_gpu::integrate::kbet(knn, d_bat.get(), N_BATCHES, cfg);

    EXPECT_NEAR(res.mean_chi2,   0.f, 0.01f)
        << "Four batches balanced: mean_chi2 should be ~0; got " << res.mean_chi2;
    EXPECT_NEAR(res.reject_rate, 0.f, 0.05f)
        << "Four batches balanced: reject_rate should be ~0; got " << res.reject_rate;

    std::printf("[REGISTRY] %s | integrate/kbet/four_batches_balanced | n=%d k=%d batches=%d | "
                "mean_chi2 | %.4f | 0.01 | none | CYCLE-140 | %s\n",
                __DATE__, N, K, N_BATCHES, res.mean_chi2,
                (res.mean_chi2 < 0.01f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 4: Kbet_Determinism_BitIdentical
//
// Same kNN + same batch labels → two runs produce bit-identical output.
// rel_err = 0 (exact integer-count chi2 path is fully deterministic).
// =============================================================================
TEST(KbetTest, Kbet_Determinism_BitIdentical)
{
    constexpr int N = 80, K = 10, N_BATCHES = 3;

    std::vector<int> h_batch(N);
    for (int c = 0; c < N; ++c) h_batch[c] = c % N_BATCHES;

    // Neighbors: fixed pattern mixing all 3 batches unevenly.
    // Batch anchors: 0→cell 0, 1→cell 3, 2→cell 6 (mod N_BATCHES).
    std::vector<int> h_nbr(static_cast<size_t>(N) * K);
    for (int c = 0; c < N; ++c) {
        h_nbr[c * K + 0] = 0;   // batch 0
        h_nbr[c * K + 1] = 3;   // batch 1 (h_batch[3]=0? no, 3%3=0)
        // Use explicit anchors: batch 0 = cells 0,6,... batch 1 = cells 1,7,...
        h_nbr[c * K + 0] = 0;   // batch (0%3=0)
        h_nbr[c * K + 1] = 1;   // batch (1%3=1)
        h_nbr[c * K + 2] = 2;   // batch (2%3=2)
        h_nbr[c * K + 3] = 3;   // batch (3%3=0)
        h_nbr[c * K + 4] = 4;   // batch (4%3=1)
        h_nbr[c * K + 5] = 5;   // batch (5%3=2)
        h_nbr[c * K + 6] = 6;   // batch (6%3=0)
        h_nbr[c * K + 7] = 7;   // batch (7%3=1)
        h_nbr[c * K + 8] = 8;   // batch (8%3=2)
        h_nbr[c * K + 9] = 9;   // batch (9%3=0)
    }

    auto d_bat = upload_int(h_batch);
    singlet_gpu::integrate::KbetConfig cfg;
    cfg.deterministic = true;

    // Run 1.
    auto knn1 = build_knn_result(N, K, make_row_offsets(N, K), h_nbr);
    auto res1 = singlet_gpu::integrate::kbet(knn1, d_bat.get(), N_BATCHES, cfg);
    auto h1   = d2h_float(res1.chi2, N);
    auto p1   = d2h_float(res1.pvalue, N);

    // Run 2 (re-upload).
    auto knn2 = build_knn_result(N, K, make_row_offsets(N, K), h_nbr);
    auto res2 = singlet_gpu::integrate::kbet(knn2, d_bat.get(), N_BATCHES, cfg);
    auto h2   = d2h_float(res2.chi2, N);
    auto p2   = d2h_float(res2.pvalue, N);

    int chi2_mismatches   = 0;
    int pvalue_mismatches = 0;
    for (int c = 0; c < N; ++c) {
        if (h1[c] != h2[c])   ++chi2_mismatches;
        if (p1[c] != p2[c])   ++pvalue_mismatches;
    }

    EXPECT_EQ(chi2_mismatches, 0)
        << "Deterministic mode: chi2 must be bit-identical; mismatches=" << chi2_mismatches;
    EXPECT_EQ(pvalue_mismatches, 0)
        << "Deterministic mode: pvalue must be bit-identical; mismatches=" << pvalue_mismatches;
    EXPECT_EQ(res1.mean_chi2,   res2.mean_chi2)
        << "mean_chi2 mismatch: " << res1.mean_chi2 << " vs " << res2.mean_chi2;
    EXPECT_EQ(res1.reject_rate, res2.reject_rate)
        << "reject_rate mismatch: " << res1.reject_rate << " vs " << res2.reject_rate;

    std::printf("[REGISTRY] %s | integrate/kbet/determinism | n=%d k=%d batches=%d | "
                "bit_identical | chi2=%d pval=%d mismatches | 0 | none | CYCLE-140 | %s\n",
                __DATE__, N, K, N_BATCHES, chi2_mismatches, pvalue_mismatches,
                (chi2_mismatches == 0 && pvalue_mismatches == 0) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 5: Kbet_RejectRate_Monotonic
//
// Gradually shift kNN from perfectly balanced to fully segregated.
// reject_rate should be non-decreasing as segregation increases.
//
// 100 cells, k=10, 2 batches at 50%.
// segregation_levels: 0, 2, 4, 6, 8, 10 extra cells from own batch
// (i.e., own_count = 5, 6, 7, 8, 9, 10; other_count = 5, 4, 3, 2, 1, 0).
// =============================================================================
TEST(KbetTest, Kbet_RejectRate_Monotonic)
{
    constexpr int N = 100, K = 10, N_BATCHES = 2;

    std::vector<int> h_batch(N);
    for (int c = 0; c < N; ++c) h_batch[c] = (c < N / 2) ? 0 : 1;
    auto d_bat = upload_int(h_batch);

    // own_counts[i]: number of neighbors from cell's own batch.
    const int own_counts[] = {5, 6, 7, 8, 9, 10};
    constexpr int N_LEVELS = 6;

    float prev_rate = -1.f;
    bool monotonic = true;

    for (int lvl = 0; lvl < N_LEVELS; ++lvl) {
        const int own  = own_counts[lvl];
        const int other = K - own;

        std::vector<int> h_nbr(static_cast<size_t>(N) * K);
        for (int c = 0; c < N; ++c) {
            int own_anchor   = (c < N / 2) ? 0 : 50;
            int other_anchor = (c < N / 2) ? 50 : 0;
            for (int j = 0; j < own;   ++j) h_nbr[c * K + j]         = own_anchor   + j;
            for (int j = 0; j < other; ++j) h_nbr[c * K + own + j]   = other_anchor + j;
        }

        auto knn = build_knn_result(N, K, make_row_offsets(N, K), h_nbr);
        singlet_gpu::integrate::KbetConfig cfg;
        cfg.deterministic = true;
        auto res = singlet_gpu::integrate::kbet(knn, d_bat.get(), N_BATCHES, cfg);

        std::printf("[REGISTRY] %s | integrate/kbet/monotonic | own=%d other=%d | "
                    "reject_rate | %.4f | monotonic | none | CYCLE-140 | -\n",
                    __DATE__, own, other, res.reject_rate);

        // Allow a small epsilon for the transition at the boundary.
        if (prev_rate > res.reject_rate + 0.01f) {
            monotonic = false;
            ADD_FAILURE() << "Monotonicity violated at level " << lvl
                          << ": prev_rate=" << prev_rate
                          << " > cur_rate=" << res.reject_rate;
        }
        prev_rate = res.reject_rate;
    }

    // At full segregation (own=10), reject_rate should be ~1.
    EXPECT_GT(prev_rate, 0.9f)
        << "At full segregation reject_rate should be > 0.9; got " << prev_rate;

    EXPECT_TRUE(monotonic) << "reject_rate must be non-decreasing with segregation";

    std::printf("[REGISTRY] %s | integrate/kbet/monotonic_summary | n=%d k=%d batches=%d | "
                "monotonic | %s | none | none | CYCLE-140 | %s\n",
                __DATE__, N, K, N_BATCHES,
                monotonic ? "true" : "false",
                monotonic ? "PASS" : "FAIL");
}
