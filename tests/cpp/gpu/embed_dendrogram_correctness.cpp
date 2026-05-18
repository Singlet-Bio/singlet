// SPDX-License-Identifier: MIT
// singlet/gpu/tests/embed_dendrogram_correctness.cpp
//
// Correctness harness for the dendrogram kernel (CYCLE-146).
// Reference: scanpy.tl.dendrogram wrapping scipy UPGMA average-linkage.
//
// Tests:
//   1. Dendrogram_TinyClosedForm             — 5 genes × 6 cells × 3 clusters.
//                                              Verify distance matrix abs_err < 1e-3.
//   2. Dendrogram_PerfectlySeparatedClusters — 50 genes × 100 cells × 3 clusters.
//                                              Inter-cluster dist >> intra-cluster dist.
//   3. Dendrogram_LinkageMonotonic           — UPGMA Z distances monotonically non-decreasing.
//   4. Dendrogram_Determinism_SameInput      — rel_err < 1e-4 between two runs.
//   5. Dendrogram_EmptyCluster_Handled       — cluster with 0 cells; no NaN/Inf.
//
// Build: compiled as CUDA (see CMakeLists.txt — LANGUAGE CUDA).
// Run:   --gtest_filter=DendrogramTest.*

#include <singlet/gpu/embed/dendrogram.h>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

// =============================================================================
// Shared helpers
// =============================================================================
namespace {

// Upload host float array to DeviceMemory<float>.
singlet::gpu::core::DeviceMemory<float> upload_float(const std::vector<float>& h)
{
    singlet::gpu::core::DeviceMemory<float> d(h.size());
    cudaMemcpy(d.get(), h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

// Upload host int array to DeviceMemory<int>.
singlet::gpu::core::DeviceMemory<int> upload_int(const std::vector<int>& h)
{
    singlet::gpu::core::DeviceMemory<int> d(h.size());
    cudaMemcpy(d.get(), h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice);
    return d;
}

// Fetch device float buffer to host.
std::vector<float> d2h_float(const singlet::gpu::core::DeviceMemory<float>& d, size_t n)
{
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

// Build a PzDeviceMatrix from dense col-major host matrix (m × n).
// Converts to CSC (keeps all non-zero values; for testing we include all).
singlet::gpu::io::PzDeviceMatrix build_pz(
    int m, int n, const std::vector<float>& h_dense)
{
    // Build CSC from dense col-major.
    std::vector<int>   col_ptr(static_cast<size_t>(n + 1), 0);
    std::vector<int>   row_idx;
    std::vector<float> values;

    for (int c = 0; c < n; ++c) {
        col_ptr[static_cast<size_t>(c)] = static_cast<int>(row_idx.size());
        for (int r = 0; r < m; ++r) {
            const float v = h_dense[static_cast<size_t>(r) + static_cast<size_t>(c) * m];
            // Include all values (even zero) for small tests; harmless.
            if (v != 0.f) {
                row_idx.push_back(r);
                values.push_back(v);
            }
        }
    }
    col_ptr[static_cast<size_t>(n)] = static_cast<int>(row_idx.size());
    const int nnz = static_cast<int>(row_idx.size());

    singlet::gpu::io::PzDeviceMatrix pz;
    pz.mat.rows = m;
    pz.mat.cols = n;
    pz.mat.nnz  = nnz;
    pz.mat.col_ptr     = upload_int(col_ptr);
    pz.mat.row_indices = upload_int(row_idx);
    pz.mat.values      = upload_float(values);
    return pz;
}

// Reference: compute cluster centroids on host (col-major output m × k).
std::vector<float> ref_centroids(
    int m, int n, const std::vector<float>& X,
    const std::vector<int>& label, int k)
{
    std::vector<float> mu(static_cast<size_t>(m) * k, 0.f);
    std::vector<int>   cnt(static_cast<size_t>(k), 0);
    for (int c = 0; c < n; ++c) {
        const int lbl = label[static_cast<size_t>(c)];
        cnt[static_cast<size_t>(lbl)]++;
        for (int g = 0; g < m; ++g)
            mu[static_cast<size_t>(g) + static_cast<size_t>(lbl) * m]
                += X[static_cast<size_t>(g) + static_cast<size_t>(c) * m];
    }
    for (int kk = 0; kk < k; ++kk) {
        if (cnt[static_cast<size_t>(kk)] == 0) continue;
        const float inv = 1.f / static_cast<float>(cnt[static_cast<size_t>(kk)]);
        for (int g = 0; g < m; ++g)
            mu[static_cast<size_t>(g) + static_cast<size_t>(kk) * m] *= inv;
    }
    return mu;
}

// Reference: Pearson correlation distance matrix (k × k, row-major == col-major for symmetric).
std::vector<float> ref_distance(int m, int k, const std::vector<float>& mu)
{
    std::vector<float> dist(static_cast<size_t>(k) * k, 0.f);
    for (int i = 0; i < k; ++i) {
        for (int j = i; j < k; ++j) {
            const float* ci = mu.data() + static_cast<size_t>(i) * m;
            const float* cj = mu.data() + static_cast<size_t>(j) * m;
            float sum_i = 0.f, sum_j = 0.f;
            for (int g = 0; g < m; ++g) { sum_i += ci[g]; sum_j += cj[g]; }
            const float mi = sum_i / m, mj = sum_j / m;
            float num = 0.f, var_i = 0.f, var_j = 0.f;
            for (int g = 0; g < m; ++g) {
                const float ai = ci[g] - mi, aj = cj[g] - mj;
                num   += ai * aj;
                var_i += ai * ai;
                var_j += aj * aj;
            }
            const float denom = std::sqrt(var_i * var_j);
            const float corr  = (denom > 1e-9f) ? num / denom : 0.f;
            const float d     = 1.f - corr;
            dist[static_cast<size_t>(i) * k + j] = d;
            dist[static_cast<size_t>(j) * k + i] = d;
        }
    }
    return dist;
}

bool has_nan_or_inf(const std::vector<float>& v)
{
    for (float x : v) {
        if (!std::isfinite(x)) return true;
    }
    return false;
}

} // anonymous namespace


// =============================================================================
// Test 1: Dendrogram_TinyClosedForm
//
// 5 genes × 6 cells × 3 clusters (2 cells each).
// Hand-crafted X with distinct cluster patterns.
// GPU result vs CPU reference: abs_err < 1e-3 on distance matrix.
// =============================================================================
TEST(DendrogramTest, Dendrogram_TinyClosedForm)
{
    constexpr int M = 5, N = 6, K = 3;

    // X: 5 × 6 col-major.
    // Cluster 0: cells 0,1 — high genes 0,1; low genes 2,3,4.
    // Cluster 1: cells 2,3 — high genes 2,3; low genes 0,1,4.
    // Cluster 2: cells 4,5 — high genes 4; low genes 0,1,2,3.
    std::vector<float> h_X = {
        // cell 0      cell 1      cell 2      cell 3      cell 4      cell 5
        3.f,1.f,0.f,0.f,0.f,  // gene 0
        3.f,1.f,0.f,0.f,0.f,  // gene 1
        0.f,0.f,4.f,2.f,0.f,  // gene 2
        0.f,0.f,4.f,2.f,0.f,  // gene 3
        0.f,0.f,0.f,0.f,5.f   // gene 4
    };
    // Re-layout as col-major m×n: X[g + c*m]
    std::vector<float> h_Xcol(M * N);
    // h_X is already row-major gene-first; convert to col-major.
    // Actually let's just re-build as col-major directly.
    // gene g, cell c: col-major = h_Xcol[g + c*M]
    // gene0: cells={3,1,0,0,0,0}; gene1: cells={3,1,0,0,0,0}
    // gene2: cells={0,0,4,2,0,0}; gene3: cells={0,0,4,2,0,0}
    // gene4: cells={0,0,0,0,5,3}
    const float vals[M][N] = {
        {3.f, 1.f, 0.f, 0.f, 0.f, 0.f},  // gene 0
        {3.f, 1.f, 0.f, 0.f, 0.f, 0.f},  // gene 1
        {0.f, 0.f, 4.f, 2.f, 0.f, 0.f},  // gene 2
        {0.f, 0.f, 4.f, 2.f, 0.f, 0.f},  // gene 3
        {0.f, 0.f, 0.f, 0.f, 5.f, 3.f},  // gene 4
    };
    for (int g = 0; g < M; ++g)
        for (int c = 0; c < N; ++c)
            h_Xcol[static_cast<size_t>(g) + static_cast<size_t>(c) * M] = vals[g][c];

    std::vector<int> h_label = {0, 0, 1, 1, 2, 2};

    auto pz = build_pz(M, N, h_Xcol);
    auto d_label_mem = upload_int(h_label);

    auto res = singlet::gpu::embed::dendrogram(pz, d_label_mem.get(), K);

    // GPU distance matrix (k × k col-major → symmetric, same as row-major).
    auto h_gpu_dist = d2h_float(res.distance, static_cast<size_t>(K) * K);

    // CPU reference.
    auto h_mu_ref   = ref_centroids(M, N, h_Xcol, h_label, K);
    auto h_dist_ref = ref_distance(M, K, h_mu_ref);

    // Abs error on distance matrix.
    float max_abs_err = 0.f;
    for (int i = 0; i < K * K; ++i) {
        // GPU is col-major, ref is row-major. For symmetric matrix: same.
        const float err = std::fabs(h_gpu_dist[static_cast<size_t>(i)]
                                  - h_dist_ref[static_cast<size_t>(i)]);
        max_abs_err = std::max(max_abs_err, err);
    }

    EXPECT_LT(max_abs_err, 1e-3f)
        << "TinyClosedForm: max abs err on distance matrix = " << max_abs_err;

    // Diagonal should be 0.
    for (int k = 0; k < K; ++k) {
        // col-major: d[k,k] = d[k + k*K]
        EXPECT_NEAR(h_gpu_dist[static_cast<size_t>(k) + static_cast<size_t>(k) * K], 0.f, 1e-3f)
            << "diagonal[" << k << "] should be 0";
    }

    std::printf("[REGISTRY] %s | embed/dendrogram/tiny_closed_form | m=%d n=%d k=%d | "
                "abs_err | %.2e | 1e-3 | none | CYCLE-146 | %s\n",
                __DATE__, M, N, K, max_abs_err,
                (max_abs_err < 1e-3f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 2: Dendrogram_PerfectlySeparatedClusters
//
// 50 genes × 100 cells × 3 clusters. Each cluster has very different mean
// expression. Inter-cluster distance should be >> intra-cluster distance
// (clusters with themselves have dist = 0; all off-diagonal > 0.5).
// =============================================================================
TEST(DendrogramTest, Dendrogram_PerfectlySeparatedClusters)
{
    constexpr int M = 50, N = 100, K = 3;
    // Cluster 0: cells 0..33  — only genes 0..16 expressed (=1).
    // Cluster 1: cells 34..66 — only genes 17..33 expressed (=1).
    // Cluster 2: cells 67..99 — only genes 34..49 expressed (=1).
    std::vector<float> h_X(M * N, 0.f);
    std::vector<int>   h_label(N);

    for (int c = 0; c < N; ++c) {
        int lbl;
        int g_start, g_end;
        if (c < 34)       { lbl = 0; g_start = 0;  g_end = 17; }
        else if (c < 67)  { lbl = 1; g_start = 17; g_end = 34; }
        else              { lbl = 2; g_start = 34; g_end = 50; }
        h_label[static_cast<size_t>(c)] = lbl;
        for (int g = g_start; g < g_end; ++g)
            h_X[static_cast<size_t>(g) + static_cast<size_t>(c) * M] = 1.f;
    }

    auto pz = build_pz(M, N, h_X);
    auto d_label_mem = upload_int(h_label);

    auto res = singlet::gpu::embed::dendrogram(pz, d_label_mem.get(), K);
    auto h_dist = d2h_float(res.distance, static_cast<size_t>(K) * K);

    EXPECT_FALSE(has_nan_or_inf(h_dist))
        << "Distance matrix contains NaN or Inf";

    // Diagonal should be near 0.
    for (int k = 0; k < K; ++k) {
        EXPECT_NEAR(h_dist[static_cast<size_t>(k) + static_cast<size_t>(k) * K], 0.f, 1e-3f)
            << "diagonal[" << k << "] should be 0";
    }

    // Off-diagonal: perfectly separated clusters → corr ≈ 0 → dist ≈ 1.
    float min_off_diag = 1e9f;
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            if (i == j) continue;
            min_off_diag = std::min(min_off_diag,
                h_dist[static_cast<size_t>(i) * K + j]);
        }
    }
    EXPECT_GT(min_off_diag, 0.5f)
        << "Perfectly separated clusters: min off-diagonal distance = "
        << min_off_diag << " should be > 0.5";

    std::printf("[REGISTRY] %s | embed/dendrogram/separated_clusters | m=%d n=%d k=%d | "
                "min_off_diag | %.4f | >0.5 | none | CYCLE-146 | %s\n",
                __DATE__, M, N, K, min_off_diag,
                (min_off_diag > 0.5f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 3: Dendrogram_LinkageMonotonic
//
// UPGMA guarantee: linkage distances must be non-decreasing.
// Use the separated-cluster data from Test 2 (K=5 for a longer linkage sequence).
// =============================================================================
TEST(DendrogramTest, Dendrogram_LinkageMonotonic)
{
    constexpr int M = 30, N = 150, K = 5;
    // 5 non-overlapping gene blocks; 30 cells per cluster.
    std::vector<float> h_X(M * N, 0.f);
    std::vector<int>   h_label(N);
    const int genes_per = M / K;  // 6 genes per cluster

    for (int c = 0; c < N; ++c) {
        const int lbl = c / (N / K);
        h_label[static_cast<size_t>(c)] = (lbl < K) ? lbl : K - 1;
        for (int g = lbl * genes_per; g < (lbl + 1) * genes_per && g < M; ++g)
            h_X[static_cast<size_t>(g) + static_cast<size_t>(c) * M] = 1.f;
    }

    auto pz = build_pz(M, N, h_X);
    auto d_label_mem = upload_int(h_label);

    auto res = singlet::gpu::embed::dendrogram(pz, d_label_mem.get(), K);

    ASSERT_EQ(static_cast<int>(res.linkage.size()), (K - 1) * 4)
        << "Linkage should have (k-1)*4 = " << (K - 1) * 4 << " elements";

    bool monotonic = true;
    float prev_d = -1e9f;
    for (int step = 0; step < K - 1; ++step) {
        const float d = res.linkage[static_cast<size_t>(step) * 4 + 2];
        if (d < prev_d - 1e-6f) {
            monotonic = false;
            break;
        }
        prev_d = d;
    }

    EXPECT_TRUE(monotonic)
        << "UPGMA linkage distances should be non-decreasing (no inversions)";

    std::printf("[REGISTRY] %s | embed/dendrogram/linkage_monotonic | k=%d | "
                "monotonic | %s | true | none | CYCLE-146 | %s\n",
                __DATE__, K, monotonic ? "true" : "false",
                monotonic ? "PASS" : "FAIL");
}


// =============================================================================
// Test 4: Dendrogram_Determinism_SameInput
//
// Two runs with identical input → rel_err < 1e-4 on distance matrix.
// atomicAdd is non-deterministic but bounded.
// =============================================================================
TEST(DendrogramTest, Dendrogram_Determinism_SameInput)
{
    constexpr int M = 20, N = 60, K = 3;

    std::vector<float> h_X(M * N);
    std::vector<int>   h_label(N);
    for (int c = 0; c < N; ++c) {
        h_label[static_cast<size_t>(c)] = c % K;
        for (int g = 0; g < M; ++g)
            h_X[static_cast<size_t>(g) + static_cast<size_t>(c) * M] =
                static_cast<float>((g + c) % 7) * 0.5f;
    }

    // Run 1.
    {
        auto pz1 = build_pz(M, N, h_X);
        auto d_lbl1 = upload_int(h_label);
        auto res1 = singlet::gpu::embed::dendrogram(pz1, d_lbl1.get(), K);
        auto h_dist1 = d2h_float(res1.distance, static_cast<size_t>(K) * K);

        // Run 2.
        auto pz2 = build_pz(M, N, h_X);
        auto d_lbl2 = upload_int(h_label);
        auto res2 = singlet::gpu::embed::dendrogram(pz2, d_lbl2.get(), K);
        auto h_dist2 = d2h_float(res2.distance, static_cast<size_t>(K) * K);

        int   mismatches  = 0;
        float max_rel_err = 0.f;
        for (int i = 0; i < K * K; ++i) {
            const float denom = std::max(std::fabs(h_dist1[static_cast<size_t>(i)]), 1e-9f);
            const float rel_err = std::fabs(h_dist1[static_cast<size_t>(i)]
                                          - h_dist2[static_cast<size_t>(i)]) / denom;
            if (rel_err > 1e-4f) {
                ++mismatches;
                max_rel_err = std::max(max_rel_err, rel_err);
            }
        }

        EXPECT_EQ(mismatches, 0)
            << "Determinism: " << mismatches << " elements with rel_err > 1e-4; "
            << "max_rel_err=" << max_rel_err;

        std::printf("[REGISTRY] %s | embed/dendrogram/determinism | m=%d n=%d k=%d | "
                    "rel_err | max=%.2e | 1e-4 | none | CYCLE-146 | %s\n",
                    __DATE__, M, N, K, max_rel_err,
                    (mismatches == 0) ? "PASS" : "FAIL");
    }
}


// =============================================================================
// Test 5: Dendrogram_EmptyCluster_Handled
//
// One cluster with 0 assigned cells. Centroid is 0; correlation with any
// non-degenerate centroid is undefined (0/0) — eps guard returns 0.
// Verify no NaN/Inf in distance matrix or linkage.
// =============================================================================
TEST(DendrogramTest, Dendrogram_EmptyCluster_Handled)
{
    constexpr int M = 10, N = 20, K = 3;
    // Assign all cells to clusters 0 and 1; cluster 2 is empty.
    std::vector<float> h_X(M * N);
    std::vector<int>   h_label(N);
    for (int c = 0; c < N; ++c) {
        h_label[static_cast<size_t>(c)] = c % 2;  // only labels 0, 1
        for (int g = 0; g < M; ++g)
            h_X[static_cast<size_t>(g) + static_cast<size_t>(c) * M] =
                static_cast<float>((g * c + 1) % 5) * 1.f;
    }

    auto pz = build_pz(M, N, h_X);
    auto d_label_mem = upload_int(h_label);

    // Should not throw; empty cluster centroid is zero.
    singlet::gpu::embed::DendrogramResult res;
    ASSERT_NO_THROW({
        res = singlet::gpu::embed::dendrogram(pz, d_label_mem.get(), K);
    });

    auto h_dist = d2h_float(res.distance, static_cast<size_t>(K) * K);

    EXPECT_FALSE(has_nan_or_inf(h_dist))
        << "Distance matrix should have no NaN/Inf even with an empty cluster";

    EXPECT_FALSE(has_nan_or_inf(res.linkage))
        << "Linkage array should have no NaN/Inf even with an empty cluster";

    std::printf("[REGISTRY] %s | embed/dendrogram/empty_cluster | m=%d n=%d k=%d | "
                "no_nan_inf | %s | true | none | CYCLE-146 | %s\n",
                __DATE__, M, N, K,
                (!has_nan_or_inf(h_dist) && !has_nan_or_inf(res.linkage)) ? "true" : "false",
                (!has_nan_or_inf(h_dist) && !has_nan_or_inf(res.linkage)) ? "PASS" : "FAIL");
}
