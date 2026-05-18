// SPDX-License-Identifier: MIT
// singlet/gpu/tests/integrate_lisi_correctness.cpp
//
// Correctness harness for the LISI kernel (CYCLE-133).
// Reference: Korsunsky et al. (2019) Nat Methods 16:1289-1296.
//
// Tests:
//   1. Lisi_PerfectMixing_HighScore    — 5/5 split → LISI ≈ 2.0
//   2. Lisi_NoMixing_LowScore          — all same label → LISI ≈ 1.0
//   3. Lisi_FourBatches_BalancedMaxLISI — 5-5-5-5 split → LISI ≈ 4.0
//   4. Lisi_Determinism_BitIdentical   — same input → bit-identical output
//   5. Lisi_SingleLabel_DegenerateCase — n_labels=1 → LISI = 1.0, no NaN/Inf
//
// Build: compiled as CUDA (see CMakeLists.txt — LANGUAGE CUDA).
// Run:   --gtest_filter=LisiTest.*

#include <singlet/gpu/integrate/lisi.h>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

// =============================================================================
// Shared helpers
// =============================================================================
namespace {

// Upload a host int array to device memory.
singlet::gpu::core::DeviceMemory<int> upload_int(const std::vector<int>& h)
{
    singlet::gpu::core::DeviceMemory<int> d(h.size());
    cudaMemcpy(d.get(), h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice);
    return d;
}

// Build a KnnResult by hand from host arrays.
// row_offsets: n+1 ints where row_offsets[i] = i*k (uniform).
// neighbors:   n*k ints (neighbor cell indices).
// distances:   n*k floats (dummy — not used by LISI, but KnnResult requires it).
singlet::gpu::graph::KnnResult build_knn_result(
    int n, int k,
    const std::vector<int>&   h_row_offsets,  // size n+1
    const std::vector<int>&   h_neighbors,    // size n*k
    const std::vector<float>& h_distances)    // size n*k
{
    singlet::gpu::graph::KnnResult res;
    res.n = n;
    res.k = k;
    res.backend_used = singlet::gpu::graph::KnnBackend::Exact;

    res.row_offsets = singlet::gpu::core::DeviceMemory<int>(n + 1);
    res.neighbors   = singlet::gpu::core::DeviceMemory<int>(static_cast<size_t>(n) * k);
    res.distances   = singlet::gpu::core::DeviceMemory<float>(static_cast<size_t>(n) * k);

    cudaMemcpy(res.row_offsets.get(), h_row_offsets.data(),
               (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(res.neighbors.get(), h_neighbors.data(),
               static_cast<size_t>(n) * k * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(res.distances.get(), h_distances.data(),
               static_cast<size_t>(n) * k * sizeof(float), cudaMemcpyHostToDevice);
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
std::vector<float> d2h_float(const singlet::gpu::core::DeviceMemory<float>& d, int n)
{
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d.get(), static_cast<size_t>(n) * sizeof(float),
               cudaMemcpyDeviceToHost);
    return h;
}

// Fetch device int buffer to host.
std::vector<int> d2h_int(const singlet::gpu::core::DeviceMemory<int>& d, int n)
{
    std::vector<int> h(n);
    cudaMemcpy(h.data(), d.get(), static_cast<size_t>(n) * sizeof(int),
               cudaMemcpyDeviceToHost);
    return h;
}

} // anonymous namespace


// =============================================================================
// Test 1: Lisi_PerfectMixing_HighScore
//
// 100 cells, k=10 neighbors, 2 batch labels.
// Each cell has exactly 5 neighbors from label 0 and 5 from label 1.
// Expected: p_0 = p_1 = 0.5 → D = 0.5² + 0.5² = 0.5 → LISI = 2.0.
// Tolerance: |LISI - 2.0| < 0.01 for all cells.
// =============================================================================
TEST(LisiTest, Lisi_PerfectMixing_HighScore)
{
    constexpr int N = 100, K = 10, N_LABELS = 2;

    // Labels: cells 0..49 → label 0; cells 50..99 → label 1.
    std::vector<int> h_label(N);
    for (int c = 0; c < N; ++c) h_label[c] = (c < N / 2) ? 0 : 1;

    // Neighbors: each cell gets 5 from label-0 (cells 0..4) and 5 from
    // label-1 (cells 50..54) regardless of which cell it is.
    std::vector<int> h_nbr(static_cast<size_t>(N) * K);
    for (int c = 0; c < N; ++c) {
        for (int j = 0; j < 5; ++j)  h_nbr[c * K + j]     = j;        // label 0
        for (int j = 0; j < 5; ++j)  h_nbr[c * K + 5 + j] = 50 + j;  // label 1
    }
    std::vector<float> h_dist(static_cast<size_t>(N) * K, 1.f);

    auto knn   = build_knn_result(N, K, make_row_offsets(N, K), h_nbr, h_dist);
    auto d_lbl = upload_int(h_label);

    singlet::gpu::integrate::LisiConfig cfg;
    cfg.deterministic = true;
    auto res = singlet::gpu::integrate::lisi(knn, d_lbl.get(), N_LABELS, cfg);

    auto h_lisi = d2h_float(res.lisi, N);

    float max_err = 0.f;
    const float expected = 2.f;
    for (int c = 0; c < N; ++c)
        max_err = std::max(max_err, std::fabs(h_lisi[c] - expected));

    EXPECT_LT(max_err, 0.01f)
        << "Perfect 5/5 mixing: expected LISI=2.0 for all cells; max_err=" << max_err;

    std::printf("[REGISTRY] %s | integrate/lisi/perfect_mixing | n=%d k=%d labels=%d | "
                "abs_err | %.4f | 0.01 | none | CYCLE-133 | %s\n",
                __DATE__, N, K, N_LABELS, max_err,
                (max_err < 0.01f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 2: Lisi_NoMixing_LowScore
//
// 100 cells, k=10 neighbors, 2 batch labels.
// Each cell's 10 neighbors all have the same label as the cell itself
// (perfect segregation).
// Expected: one p_l = 1.0, rest = 0.0 → D = 1.0 → LISI = 1.0.
// Tolerance: |LISI - 1.0| < 0.01.
// =============================================================================
TEST(LisiTest, Lisi_NoMixing_LowScore)
{
    constexpr int N = 100, K = 10, N_LABELS = 2;

    // Labels: cells 0..49 → label 0; cells 50..99 → label 1.
    std::vector<int> h_label(N);
    for (int c = 0; c < N; ++c) h_label[c] = (c < N / 2) ? 0 : 1;

    // Neighbors: label-0 cells get neighbors from label-0 only (cells 5..14);
    //            label-1 cells get neighbors from label-1 only (cells 55..64).
    std::vector<int> h_nbr(static_cast<size_t>(N) * K);
    for (int c = 0; c < N; ++c) {
        int base = (c < N / 2) ? 5 : 55;  // all within same-label group
        for (int j = 0; j < K; ++j) h_nbr[c * K + j] = base + j;
    }
    std::vector<float> h_dist(static_cast<size_t>(N) * K, 1.f);

    auto knn   = build_knn_result(N, K, make_row_offsets(N, K), h_nbr, h_dist);
    auto d_lbl = upload_int(h_label);

    singlet::gpu::integrate::LisiConfig cfg;
    cfg.deterministic = true;
    auto res = singlet::gpu::integrate::lisi(knn, d_lbl.get(), N_LABELS, cfg);

    auto h_lisi = d2h_float(res.lisi, N);

    float max_err = 0.f;
    const float expected = 1.f;
    for (int c = 0; c < N; ++c)
        max_err = std::max(max_err, std::fabs(h_lisi[c] - expected));

    EXPECT_LT(max_err, 0.01f)
        << "No mixing: expected LISI=1.0 for all cells; max_err=" << max_err;

    std::printf("[REGISTRY] %s | integrate/lisi/no_mixing | n=%d k=%d labels=%d | "
                "abs_err | %.4f | 0.01 | none | CYCLE-133 | %s\n",
                __DATE__, N, K, N_LABELS, max_err,
                (max_err < 0.01f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 3: Lisi_FourBatches_BalancedMaxLISI
//
// 200 cells, k=20 neighbors, 4 batch labels.
// Each cell's 20 neighbors are split evenly: 5 from each of 4 batches.
// Expected: p_l = 0.25 for l in {0,1,2,3} →
//   D = 4 × (0.25)² = 4 × 0.0625 = 0.25 → LISI = 4.0 (maximum for 4 labels).
// Tolerance: |LISI - 4.0| < 0.01.
// =============================================================================
TEST(LisiTest, Lisi_FourBatches_BalancedMaxLISI)
{
    constexpr int N = 200, K = 20, N_LABELS = 4;

    // Labels: cells 0..49 → 0, 50..99 → 1, 100..149 → 2, 150..199 → 3.
    std::vector<int> h_label(N);
    for (int c = 0; c < N; ++c) h_label[c] = c / 50;

    // Neighbors: 5 from each label group (fixed anchor cells per group).
    // Group anchors: label 0 → cells 0..4, label 1 → cells 50..54,
    //                label 2 → cells 100..104, label 3 → cells 150..154.
    const int anchor[4] = {0, 50, 100, 150};
    std::vector<int> h_nbr(static_cast<size_t>(N) * K);
    for (int c = 0; c < N; ++c) {
        int pos = 0;
        for (int lbl = 0; lbl < N_LABELS; ++lbl)
            for (int j = 0; j < K / N_LABELS; ++j)
                h_nbr[c * K + pos++] = anchor[lbl] + j;
    }
    std::vector<float> h_dist(static_cast<size_t>(N) * K, 1.f);

    auto knn   = build_knn_result(N, K, make_row_offsets(N, K), h_nbr, h_dist);
    auto d_lbl = upload_int(h_label);

    singlet::gpu::integrate::LisiConfig cfg;
    cfg.deterministic = true;
    auto res = singlet::gpu::integrate::lisi(knn, d_lbl.get(), N_LABELS, cfg);

    auto h_lisi = d2h_float(res.lisi, N);

    float max_err = 0.f;
    const float expected = 4.f;
    for (int c = 0; c < N; ++c)
        max_err = std::max(max_err, std::fabs(h_lisi[c] - expected));

    EXPECT_LT(max_err, 0.01f)
        << "4-batch balanced: expected LISI=4.0; max_err=" << max_err;

    std::printf("[REGISTRY] %s | integrate/lisi/four_batch_balanced | n=%d k=%d labels=%d | "
                "abs_err | %.4f | 0.01 | none | CYCLE-133 | %s\n",
                __DATE__, N, K, N_LABELS, max_err,
                (max_err < 0.01f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 4: Lisi_Determinism_BitIdentical
//
// Same kNN + same labels → two runs produce bit-identical float output.
// rel_err = 0.0 (exact match, not just close).
// =============================================================================
TEST(LisiTest, Lisi_Determinism_BitIdentical)
{
    constexpr int N = 80, K = 10, N_LABELS = 3;

    std::vector<int> h_label(N);
    for (int c = 0; c < N; ++c) h_label[c] = c % N_LABELS;

    // Neighbors: each cell gets 3-3-4 split across labels (deterministic).
    std::vector<int> h_nbr(static_cast<size_t>(N) * K);
    for (int c = 0; c < N; ++c) {
        // 4 from label 0, 3 from label 1, 3 from label 2.
        h_nbr[c * K + 0] = 0; h_nbr[c * K + 1] = 3; h_nbr[c * K + 2] = 6; h_nbr[c * K + 3] = 9;
        h_nbr[c * K + 4] = 1; h_nbr[c * K + 5] = 4; h_nbr[c * K + 6] = 7;
        h_nbr[c * K + 7] = 2; h_nbr[c * K + 8] = 5; h_nbr[c * K + 9] = 8;
    }
    std::vector<float> h_dist(static_cast<size_t>(N) * K, 1.f);

    auto d_lbl = upload_int(h_label);

    singlet::gpu::integrate::LisiConfig cfg;
    cfg.deterministic = true;

    // Run 1.
    auto knn1 = build_knn_result(N, K, make_row_offsets(N, K), h_nbr, h_dist);
    auto res1 = singlet::gpu::integrate::lisi(knn1, d_lbl.get(), N_LABELS, cfg);
    auto h1   = d2h_float(res1.lisi, N);

    // Run 2 (re-upload).
    auto knn2 = build_knn_result(N, K, make_row_offsets(N, K), h_nbr, h_dist);
    auto res2 = singlet::gpu::integrate::lisi(knn2, d_lbl.get(), N_LABELS, cfg);
    auto h2   = d2h_float(res2.lisi, N);

    int mismatches = 0;
    for (int c = 0; c < N; ++c)
        if (h1[c] != h2[c]) ++mismatches;

    EXPECT_EQ(mismatches, 0)
        << "Deterministic mode: two identical runs must be bit-identical; "
        << "mismatches=" << mismatches;

    std::printf("[REGISTRY] %s | integrate/lisi/determinism | n=%d k=%d labels=%d | "
                "bit_identical | %d mismatches | 0 | none | CYCLE-133 | %s\n",
                __DATE__, N, K, N_LABELS, mismatches,
                (mismatches == 0) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 5: Lisi_SingleLabel_DegenerateCase
//
// All cells have label 0; n_labels = 1.
// p_0 = 1.0 → D = 1.0 → LISI = 1.0 for every cell.
// Also verify: no NaN, no Inf in output.
// =============================================================================
TEST(LisiTest, Lisi_SingleLabel_DegenerateCase)
{
    constexpr int N = 50, K = 8, N_LABELS = 1;

    std::vector<int> h_label(N, 0);  // all label 0

    // Neighbors: each cell's 8 neighbors are cells 0..7 (all label 0).
    std::vector<int> h_nbr(static_cast<size_t>(N) * K);
    for (int c = 0; c < N; ++c)
        for (int j = 0; j < K; ++j) h_nbr[c * K + j] = j;
    std::vector<float> h_dist(static_cast<size_t>(N) * K, 1.f);

    auto knn   = build_knn_result(N, K, make_row_offsets(N, K), h_nbr, h_dist);
    auto d_lbl = upload_int(h_label);

    singlet::gpu::integrate::LisiConfig cfg;
    cfg.deterministic = true;
    auto res = singlet::gpu::integrate::lisi(knn, d_lbl.get(), N_LABELS, cfg);

    auto h_lisi = d2h_float(res.lisi, N);

    float max_err = 0.f;
    int nan_inf_count = 0;
    const float expected = 1.f;
    for (int c = 0; c < N; ++c) {
        if (std::isnan(h_lisi[c]) || std::isinf(h_lisi[c])) ++nan_inf_count;
        max_err = std::max(max_err, std::fabs(h_lisi[c] - expected));
    }

    EXPECT_EQ(nan_inf_count, 0)
        << "Single-label degenerate case: no NaN/Inf expected; found " << nan_inf_count;
    EXPECT_LT(max_err, 0.01f)
        << "Single-label degenerate case: expected LISI=1.0; max_err=" << max_err;

    std::printf("[REGISTRY] %s | integrate/lisi/single_label | n=%d k=%d labels=%d | "
                "abs_err | %.4f nan_inf=%d | 0.01 | none | CYCLE-133 | %s\n",
                __DATE__, N, K, N_LABELS, max_err, nan_inf_count,
                (max_err < 0.01f && nan_inf_count == 0) ? "PASS" : "FAIL");
}
