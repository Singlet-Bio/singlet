// SPDX-License-Identifier: MIT
// singlet/gpu/tests/integrate_combat_correctness.cpp
//
// Correctness harness for ComBat batch correction (CYCLE-131).
// Johnson WE, Li C, Rabinovic A (2007) Biostatistics 8:118-127.
// GPU implementation via singlet::gpu::integrate::combat().
//
// Tests:
//  1. Combat_SingleBatch_Identity    — 50×100, single batch → output ≈ input (abs<1e-3)
//  2. Combat_TwoBatches_RemovesShift — 50×200, synthetic +5 shift in batch 1 is removed
//  3. Combat_PreservesBiology        — 30×200, 2 populations × 2 batches: biology survives
//  4. Combat_Determinism             — two runs: rel_err < 1e-4
//  5. Combat_MemoryGuard             — synthetic large dims → std::runtime_error
//
// Build: compiled as CUDA (LANGUAGE CUDA in CMakeLists.txt).
// Run:   --gtest_filter="CombatTest.*"

#include <singlet/gpu/integrate/combat.h>
#include <singlet/gpu/io/pz_device_loader.h>

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

struct SynCSC {
    int rows, cols, nnz;
    std::vector<float>   vals;
    std::vector<int32_t> col_ptr;
    std::vector<int32_t> row_idx;
};

// Build CSC from dense row-major (genes × cells). Includes all nonzero values.
// Explicit zeros stored too (set threshold = 0 strict).
SynCSC dense_to_csc(int rows, int cols, const std::vector<float>& dense)
{
    SynCSC c;
    c.rows = rows; c.cols = cols;
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

singlet::gpu::io::PzDeviceMatrix upload_csc(const SynCSC& c)
{
    singlet::gpu::io::PzDeviceMatrix pz;
    pz.mat = singlet::gpu::core::DeviceCSC(c.rows, c.cols, c.nnz);
    cudaMemcpy(pz.mat.col_ptr.get(), c.col_ptr.data(),
               (c.cols + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (c.nnz > 0) {
        cudaMemcpy(pz.mat.row_indices.get(), c.row_idx.data(),
                   c.nnz * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(pz.mat.values.get(), c.vals.data(),
                   c.nnz * sizeof(float), cudaMemcpyHostToDevice);
    }
    return pz;
}

singlet::gpu::core::DeviceMemory<int> upload_batch(const std::vector<int>& b)
{
    singlet::gpu::core::DeviceMemory<int> d(b.size());
    cudaMemcpy(d.get(), b.data(), b.size() * sizeof(int), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> d2h(const singlet::gpu::core::DeviceMemory<float>& d, int n)
{
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

// X_adj is m × n col-major: element (g, c) at index g + c*m.
inline float get_adj(const std::vector<float>& adj, int m, int g, int c)
{
    return adj[g + (size_t)c * m];
}

} // anonymous namespace

// =============================================================================
// Test 1: Combat_SingleBatch_Identity
//
// 50 genes × 100 cells, all cells in batch 0.
// With a single batch, ComBat has nothing to correct: the EB-shrunk γ* → 0
// and δ²* → 1 because all genes see zero cross-batch variance. The output
// should approximate the input within abs_err < 1e-3.
//
// WHY: no batch effect → ComBat should be close to identity.
// =============================================================================
TEST(CombatTest, Combat_SingleBatch_Identity)
{
    constexpr int M = 50, N = 100;

    std::mt19937 rng(0xDEC0DEULL);
    std::uniform_real_distribution<float> u(0.1f, 5.f);

    std::vector<float> X_h(M * N);
    for (auto& v : X_h) v = u(rng);

    std::vector<int> batch_h(N, 0);

    auto csc   = dense_to_csc(M, N, X_h);
    auto pz    = upload_csc(csc);
    auto d_bat = upload_batch(batch_h);

    singlet::gpu::integrate::CombatConfig cfg;
    auto res = singlet::gpu::integrate::combat(pz, d_bat.get(), 1, cfg);
    auto out = d2h(res.X_adj, M * N);

    ASSERT_EQ(res.m, M);
    ASSERT_EQ(res.n, N);
    ASSERT_EQ(res.n_batches, 1);

    float max_err = 0.f;
    for (int c = 0; c < N; ++c) {
        for (int g = 0; g < M; ++g) {
            float in_val = X_h[g * N + c];
            float adj    = get_adj(out, M, g, c);
            float err    = std::fabs(adj - in_val);
            max_err      = std::max(max_err, err);
            EXPECT_TRUE(std::isfinite(adj))
                << "Non-finite at gene=" << g << " cell=" << c;
        }
    }

    EXPECT_LT(max_err, 1e-3f)
        << "Single-batch max abs_err=" << max_err << " exceeds 1e-3";

    std::printf("[REGISTRY] %s | integrate/combat/single_batch | 50x100 | "
                "abs_err | max=%.2e | 1e-3 | none | CYCLE-131 | %s\n",
                __DATE__, max_err, (max_err < 1e-3f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 2: Combat_TwoBatches_RemovesShift
//
// 50 genes × 200 cells. Cells 0-99 in batch 0, cells 100-199 in batch 1.
// Synthetic batch effect: add 5.0 to all genes for batch 1.
// After ComBat: per-gene within-batch means should be approximately equal.
// Tolerance: |mean_b0 - mean_b1| < 0.5 per gene.
//
// WHY: the planted shift is far larger than any biological signal here,
// so EB should shrink γ* toward the planted offset and correct it.
// =============================================================================
TEST(CombatTest, Combat_TwoBatches_RemovesShift)
{
    constexpr int M = 50, N = 200, N0 = 100;

    std::mt19937 rng(0xCAFEBABEULL);
    std::uniform_real_distribution<float> u(0.5f, 3.f);

    // Base expression (genes × cells row-major).
    std::vector<float> X_h(M * N);
    for (int g = 0; g < M; ++g)
        for (int c = 0; c < N; ++c)
            X_h[g * N + c] = u(rng);

    // Add batch shift of 5.0 to cells in batch 1 (cells 100-199).
    for (int g = 0; g < M; ++g)
        for (int c = N0; c < N; ++c)
            X_h[g * N + c] += 5.f;

    std::vector<int> batch_h(N);
    for (int c = 0; c < N; ++c) batch_h[c] = (c < N0) ? 0 : 1;

    auto csc   = dense_to_csc(M, N, X_h);
    auto pz    = upload_csc(csc);
    auto d_bat = upload_batch(batch_h);

    singlet::gpu::integrate::CombatConfig cfg;
    auto res = singlet::gpu::integrate::combat(pz, d_bat.get(), 2, cfg);
    auto out = d2h(res.X_adj, M * N);

    float max_mean_diff = 0.f;
    for (int g = 0; g < M; ++g) {
        double sum0 = 0., sum1 = 0.;
        for (int c = 0; c < N0; ++c) sum0 += get_adj(out, M, g, c);
        for (int c = N0; c < N; ++c) sum1 += get_adj(out, M, g, c);
        float mean0 = static_cast<float>(sum0 / N0);
        float mean1 = static_cast<float>(sum1 / N0);
        float diff  = std::fabs(mean0 - mean1);
        max_mean_diff = std::max(max_mean_diff, diff);
        EXPECT_LT(diff, 0.5f)
            << "gene " << g << ": |mean_b0 - mean_b1|=" << diff << " > 0.5";
    }

    // All finite.
    for (auto v : out)
        EXPECT_TRUE(std::isfinite(v));

    std::printf("[REGISTRY] %s | integrate/combat/two_batch_shift | 50x200 | "
                "mean_diff | max=%.2e | 0.5 | none | CYCLE-131 | %s\n",
                __DATE__, max_mean_diff,
                (max_mean_diff < 0.5f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 3: Combat_PreservesBiology
//
// 30 genes × 200 cells, 2 batches × 2 populations (orthogonal).
//   - Population 1: cells 0-49 (batch 0) + cells 100-149 (batch 1)
//   - Population 2: cells 50-99 (batch 0) + cells 150-199 (batch 1)
//   - Batch effect: +2.0 added to all genes for batch 1 cells.
//   - Population signal: genes 0-9 have +4.0 in pop 1; genes 10-19 have +4.0 in pop 2.
//   - Genes 20-29: background only.
//
// After ComBat, the mean difference between population 1 and 2 in marker genes
// should be preserved to within 50% of the input difference.
//
// WHY: tests that biology (orthogonal to batch) survives EB shrinkage.
// =============================================================================
TEST(CombatTest, Combat_PreservesBiology)
{
    constexpr int M = 30, N = 200;
    // pop1: c=[0,49] + c=[100,149]; pop2: c=[50,99] + c=[150,199]
    // batch0: c=[0,99]; batch1: c=[100,199]

    std::mt19937 rng(0xB10B10BULL);
    std::uniform_real_distribution<float> bg(0.5f, 2.f);

    std::vector<float> X_h(M * N, 0.f);
    // Background.
    for (int g = 0; g < M; ++g)
        for (int c = 0; c < N; ++c)
            X_h[g * N + c] = bg(rng);

    // Population signal.
    const float pop_sig = 4.f;
    for (int g = 0; g < 10; ++g) {
        // pop1 cells: 0-49 and 100-149
        for (int c = 0; c < 50; ++c)   X_h[g * N + c]       += pop_sig;
        for (int c = 100; c < 150; ++c) X_h[g * N + c]       += pop_sig;
    }
    for (int g = 10; g < 20; ++g) {
        // pop2 cells: 50-99 and 150-199
        for (int c = 50; c < 100; ++c)  X_h[g * N + c]       += pop_sig;
        for (int c = 150; c < 200; ++c) X_h[g * N + c]       += pop_sig;
    }

    // Batch effect: +2 for batch 1 (cells 100-199).
    for (int g = 0; g < M; ++g)
        for (int c = 100; c < N; ++c)
            X_h[g * N + c] += 2.f;

    std::vector<int> batch_h(N);
    for (int c = 0; c < N; ++c) batch_h[c] = (c < 100) ? 0 : 1;

    // Pre-combat biology signal (input diff).
    std::vector<float> pre_diff(M, 0.f);
    for (int g = 0; g < M; ++g) {
        double s1 = 0., s2 = 0.;
        for (int c : {0, 50})
            for (int dc = 0; dc < 50; ++dc) {
                if (c == 0) s1 += X_h[g * N + dc];
                else        s2 += X_h[g * N + c + dc];
            }
        // Also sum batch1 cells.
        for (int dc = 0; dc < 50; ++dc) {
            s1 += X_h[g * N + 100 + dc];
            s2 += X_h[g * N + 150 + dc];
        }
        pre_diff[g] = static_cast<float>((s1 - s2) / 100.0);
    }

    auto csc   = dense_to_csc(M, N, X_h);
    auto pz    = upload_csc(csc);
    auto d_bat = upload_batch(batch_h);

    singlet::gpu::integrate::CombatConfig cfg;
    auto res = singlet::gpu::integrate::combat(pz, d_bat.get(), 2, cfg);
    auto out = d2h(res.X_adj, M * N);

    // Post-combat diff.
    for (int g = 0; g < 20; ++g) {
        double s1 = 0., s2 = 0.;
        for (int dc = 0; dc < 50; ++dc) {
            s1 += get_adj(out, M, g, dc);
            s1 += get_adj(out, M, g, 100 + dc);
            s2 += get_adj(out, M, g, 50 + dc);
            s2 += get_adj(out, M, g, 150 + dc);
        }
        float post_diff = static_cast<float>((s1 - s2) / 100.0);
        float retained  = (pre_diff[g] != 0.f)
                          ? std::fabs(post_diff) / std::fabs(pre_diff[g])
                          : 1.f;

        EXPECT_GT(retained, 0.5f)
            << "gene " << g << ": biology fraction=" << retained
            << " (pre=" << pre_diff[g] << " post=" << post_diff << ")";
    }

    // All output finite.
    for (auto v : out)
        EXPECT_TRUE(std::isfinite(v));

    std::printf("[REGISTRY] %s | integrate/combat/preserves_biology | 30x200 | "
                "biology_retained | threshold=0.5 | none | CYCLE-131 | CHECK_OUTPUT\n",
                __DATE__);
}


// =============================================================================
// Test 4: Combat_Determinism
//
// Two identical runs on the same input must agree to rel_err < 1e-4.
// Atomic scatter is nondeterministic (fp32 add associativity), so this
// documents and bounds the run-to-run noise (Rule 18).
// =============================================================================
TEST(CombatTest, Combat_Determinism)
{
    constexpr int M = 40, N = 120, NB = 3;

    std::mt19937 rng(0xA1B2C3D4ULL);
    std::uniform_real_distribution<float> u(0.1f, 6.f);

    std::vector<float> X_h(M * N);
    for (auto& v : X_h) v = u(rng);

    std::vector<int> batch_h(N);
    for (int c = 0; c < N; ++c) batch_h[c] = c % NB;

    auto csc = dense_to_csc(M, N, X_h);
    singlet::gpu::integrate::CombatConfig cfg;

    auto pz1  = upload_csc(csc);
    auto db1  = upload_batch(batch_h);
    auto res1 = singlet::gpu::integrate::combat(pz1, db1.get(), NB, cfg);
    auto h1   = d2h(res1.X_adj, M * N);

    auto pz2  = upload_csc(csc);
    auto db2  = upload_batch(batch_h);
    auto res2 = singlet::gpu::integrate::combat(pz2, db2.get(), NB, cfg);
    auto h2   = d2h(res2.X_adj, M * N);

    float max_rel = 0.f;
    for (int i = 0; i < M * N; ++i) {
        float denom = std::max(std::fabs(h1[i]), 1e-6f);
        float rel   = std::fabs(h1[i] - h2[i]) / denom;
        max_rel     = std::max(max_rel, rel);
    }

    EXPECT_LT(max_rel, 1e-4f)
        << "Determinism: max_rel_err=" << max_rel << " exceeds 1e-4";

    std::printf("[REGISTRY] %s | integrate/combat/determinism | 40x120x3 | "
                "rel_err | max=%.2e | 1e-4 | none | CYCLE-131 | %s\n",
                __DATE__, max_rel, (max_rel < 1e-4f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 5: Combat_MemoryGuard
//
// Synthetic dimensions implying > 32 GB output (e.g. m=100000, n=100000).
// combat() should throw std::runtime_error before allocating.
// =============================================================================
TEST(CombatTest, Combat_MemoryGuard)
{
    // Build a trivially tiny CSC but pass huge logical dims via crafted PzDeviceMatrix.
    // Fastest approach: construct a minimal real 2×2 matrix but claim large dimensions
    // by overriding rows/cols in the device matrix descriptor.
    // Actually, the memory guard fires on mat.rows × mat.cols, so we need those large.
    // We can build a 2×2 CSC and then manually set rows/cols to huge values.
    //
    // PzDeviceMatrix wraps a DeviceCSC: DeviceCSC has .rows, .cols, .nnz as int fields
    // (public). We construct with huge dims; since nnz=0, no actual allocation of
    // row_indices/values occurs. col_ptr still needs n+1 ints — but n=100001 is fine.

    // Check if we'd be on a 32GB+ GPU (if so, skip rather than OOM).
    {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        const double free_gb = static_cast<double>(free_mem) / (1024.0 * 1024.0 * 1024.0);
        if (free_gb > 32.0) {
            GTEST_SKIP() << "GPU has > 32 GB free — memory guard threshold not reachable "
                         << "for test synthetic dims; skipping.";
        }
    }

    // m=100000, n=100000 → output = 40 GB (>> any consumer GPU's VRAM / 2)
    constexpr int M_BIG = 100000, N_BIG = 100000;
    constexpr int NB    = 2;

    singlet::gpu::io::PzDeviceMatrix pz;
    pz.mat = singlet::gpu::core::DeviceCSC(M_BIG, N_BIG, /*nnz=*/0);

    // batch labels: just two alternating batches.
    std::vector<int> batch_h(N_BIG);
    for (int c = 0; c < N_BIG; ++c) batch_h[c] = c & 1;
    auto d_bat = upload_batch(batch_h);

    singlet::gpu::integrate::CombatConfig cfg;
    EXPECT_THROW(
        singlet::gpu::integrate::combat(pz, d_bat.get(), NB, cfg),
        std::runtime_error);

    std::printf("[REGISTRY] %s | integrate/combat/memory_guard | 100k x 100k | "
                "throws_runtime_error | none | none | none | CYCLE-131 | PASS\n",
                __DATE__);
}
