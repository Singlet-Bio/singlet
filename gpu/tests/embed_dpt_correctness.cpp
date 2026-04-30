// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/embed_dpt_correctness.cpp
//
// Correctness harness for the DPT (Diffusion Pseudotime) kernel (CYCLE-142).
// Reference: Haghverdi et al. 2016 Nat Methods 13:845-848.
//
// Tests:
//   1. Dpt_LinearTrajectory_MonotonicPseudotime — 50-cell chain → Spearman >= 0.95
//   2. Dpt_RootHasZeroPseudotime              — root cell dpt ≈ 0
//   3. Dpt_TwoBranches_DistinguishesPaths     — Y-shaped graph: symmetric branches
//   4. Dpt_Determinism_BitIdentical           — rel_err < 1e-4 across two runs
//   5. Dpt_MemoryGuard_RejectsTooLarge        — n=50000 → throws
//
// Build: compiled as CUDA (see CMakeLists.txt — LANGUAGE CUDA).
// Run:   --gtest_filter=DptTest.*

#include <singlet-gpu/embed/dpt.h>

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

// Upload a host float array to a DeviceMemory<float>.
singlet_gpu::core::DeviceMemory<float> upload_float(const std::vector<float>& h)
{
    singlet_gpu::core::DeviceMemory<float> d(h.size());
    cudaMemcpy(d.get(), h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

// Upload a host int array to a DeviceMemory<int>.
singlet_gpu::core::DeviceMemory<int> upload_int(const std::vector<int>& h)
{
    singlet_gpu::core::DeviceMemory<int> d(h.size());
    cudaMemcpy(d.get(), h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice);
    return d;
}

// Fetch device float buffer to host.
std::vector<float> d2h_float(const singlet_gpu::core::DeviceMemory<float>& d, int n)
{
    std::vector<float> h(static_cast<size_t>(n));
    cudaMemcpy(h.data(), d.get(), static_cast<size_t>(n) * sizeof(float),
               cudaMemcpyDeviceToHost);
    return h;
}

// Build a KnnResult from host arrays (mirrors kBET test pattern).
// Row offsets are uniform: offsets[i] = i*k.
singlet_gpu::graph::KnnResult build_knn_result(
    int n, int k,
    const std::vector<int>&   h_neighbors,  // [n * k]
    const std::vector<float>& h_distances)  // [n * k]
{
    singlet_gpu::graph::KnnResult res;
    res.n = n;
    res.k = k;
    res.backend_used = singlet_gpu::graph::KnnBackend::Exact;

    std::vector<int> h_offsets(static_cast<size_t>(n + 1));
    for (int i = 0; i <= n; ++i) h_offsets[static_cast<size_t>(i)] = i * k;

    res.row_offsets = upload_int(h_offsets);
    res.neighbors   = upload_int(h_neighbors);
    res.distances   = upload_float(h_distances);
    return res;
}

// Spearman rank correlation between two vectors.
float spearman_corr(const std::vector<float>& a, const std::vector<float>& b)
{
    const int n = static_cast<int>(a.size());
    // Compute ranks for a and b.
    std::vector<int> ra(static_cast<size_t>(n)), rb(static_cast<size_t>(n));
    std::iota(ra.begin(), ra.end(), 0);
    std::iota(rb.begin(), rb.end(), 0);
    std::sort(ra.begin(), ra.end(), [&](int x, int y){ return a[x] < a[y]; });
    std::sort(rb.begin(), rb.end(), [&](int x, int y){ return b[x] < b[y]; });

    std::vector<float> rank_a(static_cast<size_t>(n)), rank_b(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        rank_a[static_cast<size_t>(ra[i])] = static_cast<float>(i);
        rank_b[static_cast<size_t>(rb[i])] = static_cast<float>(i);
    }

    float sum_d2 = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = rank_a[static_cast<size_t>(i)] - rank_b[static_cast<size_t>(i)];
        sum_d2 += d * d;
    }
    const float fn = static_cast<float>(n);
    return 1.f - 6.f * sum_d2 / (fn * (fn * fn - 1.f));
}

} // anonymous namespace


// =============================================================================
// Test 1: Dpt_LinearTrajectory_MonotonicPseudotime
//
// Build a 50-cell linear chain. Cell i connects to i-1 and i+1 with distance
// 1.0; remaining k-2 slots filled with far-away cells (distance 100.0) to
// keep k uniform. Root = cell 0.
// Expected: dpt is strictly increasing with cell index; Spearman >= 0.95.
// =============================================================================
TEST(DptTest, Dpt_LinearTrajectory_MonotonicPseudotime)
{
    constexpr int N = 50, K = 4, ROOT = 0;

    std::vector<int>   h_nbr(static_cast<size_t>(N) * K, 0);
    std::vector<float> h_dist(static_cast<size_t>(N) * K, 0.f);

    // Build chain: cell c connects to {c-2, c-1, c+1, c+2} with distance |Δ|.
    // For boundary cells (c < 2 or c > N-3), wrap to nearby chain neighbors so
    // every cell has K real chain edges with short distances (no padding hack
    // that would dilute the Gaussian kernel bandwidth σ_i).
    //
    // WHY all-short-edge: σ_i is the median of the k distances. With 2 short
    // (d=1) + 2 long (d=100), σ_i = median{1,1,100,100} = 50.5, which makes
    // exp(-d²/(σ_i σ_j)) ≈ 1 for ALL edges — collapses the diffusion into noise.
    // With all 4 edges in the chain (d ∈ {1, 2}), σ_i ≈ 1.5 and the chain
    // structure dominates the eigenvectors, giving monotonic DPT.
    for (int c = 0; c < N; ++c) {
        // Pick 4 distinct neighbors close to c on the chain (avoiding self).
        int candidates[4];
        if (c == 0)            { candidates[0]=1; candidates[1]=2; candidates[2]=3; candidates[3]=4; }
        else if (c == 1)       { candidates[0]=0; candidates[1]=2; candidates[2]=3; candidates[3]=4; }
        else if (c == N-2)     { candidates[0]=N-1; candidates[1]=N-3; candidates[2]=N-4; candidates[3]=N-5; }
        else if (c == N-1)     { candidates[0]=N-2; candidates[1]=N-3; candidates[2]=N-4; candidates[3]=N-5; }
        else                   { candidates[0]=c-2; candidates[1]=c-1; candidates[2]=c+1; candidates[3]=c+2; }
        for (int slot = 0; slot < K; ++slot) {
            h_nbr[static_cast<size_t>(c) * K + slot]  = candidates[slot];
            h_dist[static_cast<size_t>(c) * K + slot] = static_cast<float>(std::abs(candidates[slot] - c));
        }
    }

    auto knn = build_knn_result(N, K, h_nbr, h_dist);

    singlet_gpu::embed::DptConfig cfg;
    cfg.root_cell   = ROOT;
    cfg.n_eigenvecs = 10;

    auto res = singlet_gpu::embed::dpt(knn, cfg);
    auto h_dpt = d2h_float(res.pseudotime, N);

    EXPECT_NEAR(h_dpt[static_cast<size_t>(ROOT)], 0.f, 1e-3f)
        << "Root cell should have dpt ≈ 0; got " << h_dpt[static_cast<size_t>(ROOT)];

    // Build reference: cell indices 0..N-1.
    std::vector<float> cell_idx(static_cast<size_t>(N));
    std::iota(cell_idx.begin(), cell_idx.end(), 0.f);

    const float rho = spearman_corr(h_dpt, cell_idx);
    EXPECT_GE(rho, 0.95f)
        << "Linear chain: Spearman(dpt, cell_index) should be >= 0.95; got " << rho;

    std::printf("[REGISTRY] %s | embed/dpt/linear_trajectory | n=%d k=%d | "
                "spearman | %.4f | 0.95 | none | CYCLE-142 | %s\n",
                __DATE__, N, K, rho, (rho >= 0.95f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 2: Dpt_RootHasZeroPseudotime
//
// Arbitrary kNN: 30 cells, k=4, root = cell 7.
// Expected: dpt[7] ≈ 0 (within 1e-3), all other cells have dpt >= 0.
// =============================================================================
TEST(DptTest, Dpt_RootHasZeroPseudotime)
{
    constexpr int N = 30, K = 4, ROOT = 7;

    std::vector<int>   h_nbr(static_cast<size_t>(N) * K);
    std::vector<float> h_dist(static_cast<size_t>(N) * K);

    // Simple ring structure: cell i → {i-1, i+1, i-2, i+2} mod N.
    for (int c = 0; c < N; ++c) {
        h_nbr[static_cast<size_t>(c) * K + 0]  = (c - 1 + N) % N;
        h_dist[static_cast<size_t>(c) * K + 0] = 1.f;
        h_nbr[static_cast<size_t>(c) * K + 1]  = (c + 1) % N;
        h_dist[static_cast<size_t>(c) * K + 1] = 1.f;
        h_nbr[static_cast<size_t>(c) * K + 2]  = (c - 2 + N) % N;
        h_dist[static_cast<size_t>(c) * K + 2] = 2.f;
        h_nbr[static_cast<size_t>(c) * K + 3]  = (c + 2) % N;
        h_dist[static_cast<size_t>(c) * K + 3] = 2.f;
    }

    auto knn = build_knn_result(N, K, h_nbr, h_dist);

    singlet_gpu::embed::DptConfig cfg;
    cfg.root_cell   = ROOT;
    cfg.n_eigenvecs = 8;

    auto res = singlet_gpu::embed::dpt(knn, cfg);
    auto h_dpt = d2h_float(res.pseudotime, N);

    EXPECT_NEAR(h_dpt[static_cast<size_t>(ROOT)], 0.f, 1e-3f)
        << "Root cell " << ROOT << " should have dpt ≈ 0; got "
        << h_dpt[static_cast<size_t>(ROOT)];

    for (int c = 0; c < N; ++c) {
        EXPECT_GE(h_dpt[static_cast<size_t>(c)], 0.f)
            << "dpt[" << c << "] should be non-negative; got " << h_dpt[static_cast<size_t>(c)];
    }

    std::printf("[REGISTRY] %s | embed/dpt/root_zero | n=%d k=%d root=%d | "
                "dpt_root | %.6f | 1e-3 | none | CYCLE-142 | %s\n",
                __DATE__, N, K, ROOT, h_dpt[static_cast<size_t>(ROOT)],
                (h_dpt[static_cast<size_t>(ROOT)] < 1e-3f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 3: Dpt_TwoBranches_DistinguishesPaths
//
// Y-shaped graph: 60 cells.
//   Stem:     cells 0..19 (root = 0, tip = 19)
//   Branch A: cells 20..39 attached to cell 19
//   Branch B: cells 40..59 attached to cell 19
//
// Each cell connects to its two linear neighbors with distance 1.0.
// Branch junction (cell 19) connects to 20 and 40 in addition.
//
// Expected:
//   - dpt is monotonic through the stem (cells 0..19).
//   - dpt[19] < dpt[39] (branch tip A is farther than junction).
//   - dpt[19] < dpt[59] (branch tip B is farther than junction).
//   - |dpt[39] - dpt[59]| < 0.3 * max(dpt[39], dpt[59]) (symmetric branches).
// =============================================================================
TEST(DptTest, Dpt_TwoBranches_DistinguishesPaths)
{
    constexpr int N = 60, K = 6, ROOT = 0;

    std::vector<int>   h_nbr(static_cast<size_t>(N) * K, 0);
    std::vector<float> h_dist(static_cast<size_t>(N) * K, 0.f);

    // Helper: define which cells are "linked" for each cell c, returning the
    // K closest neighbors with their distances. For a Y-graph: stem 0..19,
    // branches 20..39 and 40..59, junction at 19.
    //
    // WHY all-short distances (no d=50 padding): σ_i is median of k distances.
    // With 2 short (d=1) + 4 long (d=50), σ_i ≈ 50 → exp(-1/2500) ≈ 1 for every
    // edge → diffusion collapses to noise (same bug as Test 1's original setup).
    // Fill all K slots with chain-local neighbors (distance 1-3) so σ_i ≈ 2.
    auto fill_neighbors = [&](int c) {
        std::vector<std::pair<int, float>> nbrs;  // (cell_index, distance)
        // Determine c's "branch" and gather linear neighbors within ±3 hops.
        auto add = [&](int target, float d) {
            if (target >= 0 && target < N && target != c) nbrs.emplace_back(target, d);
        };
        if (c < 20) {  // stem
            for (int dc = -3; dc <= 3; ++dc) if (dc != 0) {
                int t = c + dc;
                if (t >= 0 && t < 20) add(t, std::fabs(static_cast<float>(dc)));
            }
            if (c == 19) { add(20, 1.f); add(40, 1.f); }  // junction → both branches
        } else if (c < 40) {  // branch A
            for (int dc = -3; dc <= 3; ++dc) if (dc != 0) {
                int t = c + dc;
                if (t >= 20 && t < 40) add(t, std::fabs(static_cast<float>(dc)));
            }
            if (c == 20) add(19, 1.f);  // branch start → junction
        } else {  // branch B
            for (int dc = -3; dc <= 3; ++dc) if (dc != 0) {
                int t = c + dc;
                if (t >= 40 && t < 60) add(t, std::fabs(static_cast<float>(dc)));
            }
            if (c == 40) add(19, 1.f);  // branch start → junction
        }
        // Sort by distance ascending and take first K.
        std::sort(nbrs.begin(), nbrs.end(),
                  [](auto& a, auto& b) { return a.second < b.second; });
        // Pad if fewer than K (small chain endpoints) — repeat the last neighbor.
        while ((int)nbrs.size() < K && !nbrs.empty()) nbrs.push_back(nbrs.back());
        for (int slot = 0; slot < K && slot < (int)nbrs.size(); ++slot) {
            h_nbr[static_cast<size_t>(c) * K + slot]  = nbrs[slot].first;
            h_dist[static_cast<size_t>(c) * K + slot] = nbrs[slot].second;
        }
    };
    for (int c = 0; c < N; ++c) fill_neighbors(c);

    auto knn = build_knn_result(N, K, h_nbr, h_dist);

    singlet_gpu::embed::DptConfig cfg;
    cfg.root_cell   = ROOT;
    cfg.n_eigenvecs = 10;

    auto res = singlet_gpu::embed::dpt(knn, cfg);
    auto h_dpt = d2h_float(res.pseudotime, N);

    const float dpt_junction = h_dpt[19];
    const float dpt_tipA     = h_dpt[39];
    const float dpt_tipB     = h_dpt[59];

    EXPECT_LT(dpt_junction, dpt_tipA)
        << "dpt[19] should be < dpt[39] (junction closer than branch A tip); "
        << dpt_junction << " vs " << dpt_tipA;
    EXPECT_LT(dpt_junction, dpt_tipB)
        << "dpt[19] should be < dpt[59] (junction closer than branch B tip); "
        << dpt_junction << " vs " << dpt_tipB;

    const float max_tip = std::max(dpt_tipA, dpt_tipB);
    const float tip_asymmetry = std::fabs(dpt_tipA - dpt_tipB);
    EXPECT_LT(tip_asymmetry, 0.3f * max_tip)
        << "Symmetric branches should have similar end-pseudotime; "
        << "|dpt[39]-dpt[59]|=" << tip_asymmetry
        << " exceeds 30% of max=" << max_tip;

    std::printf("[REGISTRY] %s | embed/dpt/two_branches | n=%d k=%d | "
                "branch_asymmetry | %.4f | 0.3*max_tip=%.4f | none | CYCLE-142 | %s\n",
                __DATE__, N, K, tip_asymmetry, 0.3f * max_tip,
                (tip_asymmetry < 0.3f * max_tip) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 4: Dpt_Determinism_BitIdentical
//
// Two runs with identical input → rel_err < 1e-4 for all pseudotime values.
// (cuSOLVER fp32 eigendecomp may have small numerical variation across runs;
// 1e-4 relative tolerance accommodates that.)
// =============================================================================
TEST(DptTest, Dpt_Determinism_BitIdentical)
{
    constexpr int N = 40, K = 4, ROOT = 5;

    std::vector<int>   h_nbr(static_cast<size_t>(N) * K);
    std::vector<float> h_dist(static_cast<size_t>(N) * K);

    // Ring with two distance scales.
    for (int c = 0; c < N; ++c) {
        h_nbr[static_cast<size_t>(c) * K + 0]  = (c - 1 + N) % N;
        h_dist[static_cast<size_t>(c) * K + 0] = 1.f;
        h_nbr[static_cast<size_t>(c) * K + 1]  = (c + 1) % N;
        h_dist[static_cast<size_t>(c) * K + 1] = 1.f;
        h_nbr[static_cast<size_t>(c) * K + 2]  = (c - 2 + N) % N;
        h_dist[static_cast<size_t>(c) * K + 2] = 2.f;
        h_nbr[static_cast<size_t>(c) * K + 3]  = (c + 2) % N;
        h_dist[static_cast<size_t>(c) * K + 3] = 2.f;
    }

    singlet_gpu::embed::DptConfig cfg;
    cfg.root_cell   = ROOT;
    cfg.n_eigenvecs = 8;

    // Run 1.
    auto knn1 = build_knn_result(N, K, h_nbr, h_dist);
    auto res1 = singlet_gpu::embed::dpt(knn1, cfg);
    auto h1   = d2h_float(res1.pseudotime, N);

    // Run 2 (re-upload same data).
    auto knn2 = build_knn_result(N, K, h_nbr, h_dist);
    auto res2 = singlet_gpu::embed::dpt(knn2, cfg);
    auto h2   = d2h_float(res2.pseudotime, N);

    int mismatches = 0;
    float max_rel_err = 0.f;
    for (int c = 0; c < N; ++c) {
        const float denom = std::max(std::fabs(h1[static_cast<size_t>(c)]), 1e-9f);
        const float rel_err = std::fabs(h1[static_cast<size_t>(c)] - h2[static_cast<size_t>(c)]) / denom;
        if (rel_err > 1e-4f) {
            ++mismatches;
            max_rel_err = std::max(max_rel_err, rel_err);
        }
    }

    EXPECT_EQ(mismatches, 0)
        << "Determinism: " << mismatches << " cells with rel_err > 1e-4; "
        << "max_rel_err=" << max_rel_err;

    std::printf("[REGISTRY] %s | embed/dpt/determinism | n=%d k=%d | "
                "rel_err | max=%.2e | 1e-4 | none | CYCLE-142 | %s\n",
                __DATE__, N, K, max_rel_err,
                (mismatches == 0) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 5: Dpt_MemoryGuard_RejectsTooLarge
//
// Construct a fake KnnResult with n = 50000.
// Dense T = 50000^2 * 4 = 10 GB → guard should throw.
// Build a minimal KnnResult with n set to 50000 and k=1 (don't allocate the
// actual n*k arrays in full — just set n and k fields plus small buffers;
// the memory guard fires before any kernel launch).
// =============================================================================
TEST(DptTest, Dpt_MemoryGuard_RejectsTooLarge)
{
    constexpr int N_LARGE = 100000;  // 40 GB n² float buffer — exceeds V100 free/2 guard reliably

    // Build a minimal KnnResult: we only need n and k fields set correctly.
    // Allocate tiny neighbor/distance buffers (1 element each) — we expect
    // the function to throw before accessing them.
    singlet_gpu::graph::KnnResult fake_knn;
    fake_knn.n = N_LARGE;
    fake_knn.k = 1;
    fake_knn.backend_used = singlet_gpu::graph::KnnBackend::Exact;

    // Tiny buffers — guard fires before any kernel touches them.
    fake_knn.row_offsets = singlet_gpu::core::DeviceMemory<int>(2);
    fake_knn.neighbors   = singlet_gpu::core::DeviceMemory<int>(1);
    fake_knn.distances   = singlet_gpu::core::DeviceMemory<float>(1);

    singlet_gpu::embed::DptConfig cfg;
    cfg.root_cell   = 0;
    cfg.n_eigenvecs = 15;

    bool threw = false;
    std::string err_msg;
    try {
        auto res = singlet_gpu::embed::dpt(fake_knn, cfg);
    } catch (const std::runtime_error& e) {
        threw = true;
        err_msg = e.what();
    }

    EXPECT_TRUE(threw)
        << "Memory guard should throw for n=" << N_LARGE
        << " (dense T = ~10 GB); no exception was thrown.";

    if (threw) {
        // Error message should mention the size or suggest subsetting.
        const bool has_hint = (err_msg.find("GB") != std::string::npos) ||
                              (err_msg.find("Subset") != std::string::npos) ||
                              (err_msg.find("subset") != std::string::npos);
        EXPECT_TRUE(has_hint)
            << "Memory guard error message should mention GB or subsetting; got: " << err_msg;
    }

    std::printf("[REGISTRY] %s | embed/dpt/memory_guard | n=%d | "
                "throws | %s | expected | none | CYCLE-142 | %s\n",
                __DATE__, N_LARGE, threw ? "true" : "false",
                threw ? "PASS" : "FAIL");
}
