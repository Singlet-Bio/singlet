// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/enrich_decoupler_viper_correctness.cpp
//
// Correctness harness for decoupleR VIPER/aREA NES scoring (CYCLE-137).
//
// Tests:
//  1. Viper_TinyClosedForm             — 5×4×2, hand-computed T1/NES, abs_err<1e-3
//  2. Viper_AllPositiveWeights_HighRankGenesScoreHigh — 50×100×1, sign check
//  3. Viper_NegativeWeights_FlipsSign  — same setup, sign flip on negative weights
//  4. Viper_Determinism_BitIdentical   — rel_err = 0 across two runs
//  5. Viper_MemoryGuard_RejectsTooLarge — >32 GB T1 → runtime_error
//
// Build: compiled as CUDA (see CMakeLists.txt — LANGUAGE CUDA).
// Run:   --gtest_filter="ViperTest.*"

#include <singlet-gpu/enrich/decoupler_viper.h>
#include <singlet-gpu/io/pz_device_loader.h>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

// =============================================================================
// Shared test helpers
// =============================================================================

namespace {

struct SynCSC {
    int                  rows, cols, nnz;
    std::vector<float>   vals;
    std::vector<int32_t> col_ptr;
    std::vector<int32_t> row_idx;
};

// dense_to_csc: dense is row-major (rows=genes, cols=cells): dense[g*cols + c].
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
                int pos       = c.col_ptr[j] + fill[j]++;
                c.row_idx[pos] = i;
                c.vals[pos]    = v;
            }
        }
    }
    return c;
}

singlet_gpu::io::PzDeviceMatrix upload_csc(const SynCSC& c)
{
    singlet_gpu::io::PzDeviceMatrix pz;
    pz.mat = singlet_gpu::core::DeviceCSC(c.rows, c.cols, c.nnz);
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

std::vector<float> d2h(const singlet_gpu::core::DeviceMemory<float>& d, int n)
{
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

singlet_gpu::core::DeviceMemory<float> upload_W(const std::vector<float>& W_h)
{
    singlet_gpu::core::DeviceMemory<float> d_W(W_h.size());
    cudaMemcpy(d_W.get(), W_h.data(), W_h.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    return d_W;
}

// ---------------------------------------------------------------------------
// cpu_viper — CPU reference for VIPER v0 simplified NES = ES1.
//
// Inputs:
//   X: m×n col-major (X[g + c*m]). W: m×p col-major (W[g + r*m]).
//   Returns NES: n×p col-major (NES[c + r*n]).
//
// Algorithm:
//   For each cell c:
//     1. Collect dense column X[:,c] (m values; zeros for missing).
//     2. Rank 0..m-1 ascending by expression (stable argsort).
//     3. T1[gene, c] = normppf((rank+1)/(m+1)).
//   NES[c, r] = (Σ_g W[g,r] * T1[g,c]) / (Σ_g |W[g,r]|).
//
// normppf via Acklam (2003) rational approximation — CPU only, not the CUDA
// normcdfinvf. Coefficients from Peter Acklam's algorithm (freely cited).
// Sufficient accuracy for abs_err < 1e-3 test tolerances.
//
// Reference comment for Test 1 (5 genes × 4 cells × 2 regulons):
//   Cell 0 expression (row-major, then col c=0): [3, 0, 1, 0, 2]
//   Sorted ascending: [0, 1, 2, 3, 5] → ranks: gene1=0, gene2=1, gene4=2,
//   gene0=3, gene3=--> wait: gene3=0 too. Stable sort:
//   values sorted: 0(gene1), 0(gene3), 1(gene2), 2(gene4), 3(gene0)
//   ranks (0-based): gene1→0, gene3→1, gene2→2, gene4→3, gene0→4
//   T1 quantiles (r+1)/6: 1/6, 2/6, 3/6, 4/6, 5/6
//   T1 via normppf: -0.9674, -0.4307, 0.0000, 0.4307, 0.9674
//   (scipy.stats.norm.ppf([1/6, 2/6, 3/6, 4/6, 5/6]) =
//    [-0.9674, -0.4307, 0.0000, 0.4307, 0.9674])
// ---------------------------------------------------------------------------
static double acklam_normppf(double p)
{
    // Coefficients from Acklam (2003): accurate to ~5e-9 in (0,1).
    static const double a[] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    static const double b[] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static const double c_[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    static const double d[] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00
    };
    const double p_low  = 0.02425;
    const double p_high = 1.0 - p_low;
    double q, r, x;
    if (p < p_low) {
        q = std::sqrt(-2.0 * std::log(p));
        x = (((((c_[0]*q+c_[1])*q+c_[2])*q+c_[3])*q+c_[4])*q+c_[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= p_high) {
        q = p - 0.5;
        r = q * q;
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c_[0]*q+c_[1])*q+c_[2])*q+c_[3])*q+c_[4])*q+c_[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
    return x;
}

// cpu_viper: X_col is m×n col-major (X_col[g + c*m]), W_col is m×p col-major.
std::vector<float> cpu_viper(
    int m, int n, int p,
    const std::vector<float>& X_col,
    const std::vector<float>& W_col)
{
    const double m1_inv = 1.0 / static_cast<double>(m + 1);
    // Build T1: m × n col-major.
    std::vector<double> T1(static_cast<size_t>(m) * n, 0.0);

    for (int c = 0; c < n; ++c) {
        const float* Xc = X_col.data() + static_cast<size_t>(c) * m;
        // Stable argsort ascending.
        std::vector<int> idx(m);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(),
            [&](int a, int b) { return Xc[a] < Xc[b]; });
        // Assign T1.
        double* T1c = T1.data() + static_cast<size_t>(c) * m;
        for (int r = 0; r < m; ++r) {
            double q = static_cast<double>(r + 1) * m1_inv;
            T1c[idx[r]] = acklam_normppf(q);
        }
    }

    // NES[c, r] = (Σ_g W[g,r] * T1[g,c]) / (Σ_g |W[g,r]|).
    std::vector<float> nes(static_cast<size_t>(n) * p, 0.f);
    for (int r = 0; r < p; ++r) {
        const float* Wr = W_col.data() + static_cast<size_t>(r) * m;
        double l1 = 0.0;
        for (int g = 0; g < m; ++g) l1 += std::fabs(static_cast<double>(Wr[g]));
        if (l1 < 1e-12) l1 = 1e-12;
        for (int c = 0; c < n; ++c) {
            const double* T1c = T1.data() + static_cast<size_t>(c) * m;
            double dot = 0.0;
            for (int g = 0; g < m; ++g)
                dot += static_cast<double>(Wr[g]) * T1c[g];
            nes[c + static_cast<size_t>(r) * n] = static_cast<float>(dot / l1);
        }
    }
    return nes;
}

// rowmaj_to_colmaj: X_row[g*n + c] → X_col[g + c*m].
std::vector<float> rowmaj_to_colmaj(const std::vector<float>& X_row, int m, int n)
{
    std::vector<float> X_col(static_cast<size_t>(m) * n);
    for (int g = 0; g < m; ++g)
        for (int c = 0; c < n; ++c)
            X_col[g + static_cast<size_t>(c) * m] = X_row[g * n + c];
    return X_col;
}

} // anonymous namespace


// =============================================================================
// VIPER Tests
// =============================================================================

// =============================================================================
// Test 1: Viper_TinyClosedForm
//
// 5 genes × 4 cells × 2 regulons. Hand-constructed X and W.
// CPU reference NES via cpu_viper (stable argsort + Acklam qnorm).
// GPU NES (sort-based + normcdfinvf) must match within abs_err < 1e-3.
//
// Scipy verification comment (python):
//   X[:, 0] = [3, 0, 1, 0, 2]  → sorted order: [0, 0, 1, 2, 3] (genes 1,3,2,4,0)
//   T1 quantiles: [1/6, 2/6, 3/6, 4/6, 5/6]
//   scipy.stats.norm.ppf([1/6, 2/6, 3/6, 4/6, 5/6]):
//     gene1 → -0.9674, gene3 → -0.4307, gene2 → 0.0000, gene4 → 0.4307, gene0 → 0.9674
// =============================================================================
TEST(ViperTest, Viper_TinyClosedForm)
{
    constexpr int M = 5, N = 4, P = 2;

    // Row-major X (genes × cells): X_row[g*N + c]
    // clang-format off
    std::vector<float> X_row = {
        // gene 0: c0  c1  c2  c3
                   3,  0,  1,  2,
        // gene 1
                   0,  2,  0,  1,
        // gene 2
                   1,  1,  3,  0,
        // gene 3
                   0,  0,  0,  1,
        // gene 4
                   2,  3,  1,  2,
    };
    // Col-major W (m × p): W_h[g + r*M]
    // regulon 0: weights [+1, -1, +0.5, 0, +2]  (activating/repressing mix)
    // regulon 1: weights [+0.5, +1, -0.5, +1, 0]
    std::vector<float> W_h = {
         1.0f, -1.0f,  0.5f,  0.0f,  2.0f,   // regulon 0, genes 0..4
         0.5f,  1.0f, -0.5f,  1.0f,  0.0f,   // regulon 1, genes 0..4
    };
    // clang-format on

    // CPU reference.
    auto X_col = rowmaj_to_colmaj(X_row, M, N);
    auto ref   = cpu_viper(M, N, P, X_col, W_h);

    // GPU.
    auto csc = dense_to_csc(M, N, X_row);
    auto pz  = upload_csc(csc);
    auto dW  = upload_W(W_h);

    singlet_gpu::enrich::ViperConfig cfg;
    auto res = singlet_gpu::enrich::viper(pz, dW.get(), P, cfg);
    auto gpu = d2h(res.nes, N * P);

    ASSERT_EQ(res.n_cells,    N);
    ASSERT_EQ(res.n_regulons, P);

    float max_err = 0.f;
    for (int i = 0; i < N * P; ++i) {
        float err = std::fabs(gpu[i] - ref[i]);
        max_err   = std::max(max_err, err);
        EXPECT_NEAR(gpu[i], ref[i], 1e-3f)
            << "element " << i << " (regulon=" << i/N << " cell=" << i%N
            << "): gpu=" << gpu[i] << " ref=" << ref[i];
    }

    std::printf("[REGISTRY] %s | enrich/decoupler_viper/tiny_cf | 5x4x2 | "
                "abs_err | max=%.2e | 1e-3 | none | CYCLE-137 | %s\n",
                __DATE__, max_err, (max_err < 1e-3f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 2: Viper_AllPositiveWeights_HighRankGenesScoreHigh
//
// 50 genes × 100 cells × 1 regulon.
// Regulon = genes [0..9], all weights +1 (target set at the top of the gene list).
// Cell 0:  genes 0..9 expressed highly (value 10.0), genes 10..49 expressed low (0.1).
// Cell 99: genes 0..9 expressed low (0.1), genes 10..49 expressed high (10.0).
//
// Expected:
//   NES[0, 0]  > 1.0  (cell 0: target genes at top of ranking → high TF activity)
//   NES[99, 0] < -1.0 (cell 99: target genes at bottom → low TF activity)
// =============================================================================
TEST(ViperTest, Viper_AllPositiveWeights_HighRankGenesScoreHigh)
{
    constexpr int M = 50, N = 100, P = 1;

    // Row-major X (genes × cells): X_row[g*N + c]
    std::vector<float> X_row(M * N, 0.f);
    for (int g = 0; g < M; ++g) {
        for (int c = 0; c < N; ++c) {
            if (c == 0) {
                // Cell 0: target genes (0..9) high, rest low.
                X_row[g * N + c] = (g < 10) ? 10.f : 0.1f;
            } else if (c == N - 1) {
                // Cell 99: target genes (0..9) low, rest high.
                X_row[g * N + c] = (g < 10) ? 0.1f : 10.f;
            } else {
                X_row[g * N + c] = 1.f;  // neutral middle cells
            }
        }
    }

    // Col-major W: regulon 0 = genes 0..9 with weight +1; genes 10..49 = 0.
    std::vector<float> W_h(M * P, 0.f);
    for (int g = 0; g < 10; ++g) W_h[g] = 1.f;

    auto csc = dense_to_csc(M, N, X_row);
    auto pz  = upload_csc(csc);
    auto dW  = upload_W(W_h);

    singlet_gpu::enrich::ViperConfig cfg;
    auto res = singlet_gpu::enrich::viper(pz, dW.get(), P, cfg);
    auto gpu = d2h(res.nes, N * P);  // NES[c + 0*N] = NES[c, 0]

    // Cell 0 (high target): NES should be clearly positive.
    EXPECT_GT(gpu[0], 1.0f)
        << "Cell 0 (high-rank target genes): expected NES > 1.0, got " << gpu[0];
    // Cell 99 (low target): NES should be clearly negative.
    EXPECT_LT(gpu[N - 1], -1.0f)
        << "Cell 99 (low-rank target genes): expected NES < -1.0, got " << gpu[N - 1];

    std::printf("[REGISTRY] %s | enrich/decoupler_viper/pos_weights | 50x100x1 | "
                "sign_check | c0=%.3f c99=%.3f | c0>1 c99<-1 | none | CYCLE-137 | %s\n",
                __DATE__, gpu[0], gpu[N - 1],
                (gpu[0] > 1.0f && gpu[N - 1] < -1.0f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 3: Viper_NegativeWeights_FlipsSign
//
// Same expression setup as Test 2, but regulon weights = -1 instead of +1.
// Negative mode-of-regulation means high expression → LOW TF activity.
//
// Expected (sign-flipped from Test 2):
//   NES[0, 0]  < -1.0  (cell 0: target genes high but weight=-1 → low activity)
//   NES[99, 0] >  1.0  (cell 99: target genes low but weight=-1 → high activity)
// =============================================================================
TEST(ViperTest, Viper_NegativeWeights_FlipsSign)
{
    constexpr int M = 50, N = 100, P = 1;

    std::vector<float> X_row(M * N, 0.f);
    for (int g = 0; g < M; ++g) {
        for (int c = 0; c < N; ++c) {
            if (c == 0) {
                X_row[g * N + c] = (g < 10) ? 10.f : 0.1f;
            } else if (c == N - 1) {
                X_row[g * N + c] = (g < 10) ? 0.1f : 10.f;
            } else {
                X_row[g * N + c] = 1.f;
            }
        }
    }

    // Same structure as Test 2 but weights are -1.
    std::vector<float> W_h(M * P, 0.f);
    for (int g = 0; g < 10; ++g) W_h[g] = -1.f;

    auto csc = dense_to_csc(M, N, X_row);
    auto pz  = upload_csc(csc);
    auto dW  = upload_W(W_h);

    singlet_gpu::enrich::ViperConfig cfg;
    auto res = singlet_gpu::enrich::viper(pz, dW.get(), P, cfg);
    auto gpu = d2h(res.nes, N * P);

    EXPECT_LT(gpu[0], -1.0f)
        << "Cell 0 (neg weights, high target): expected NES < -1.0, got " << gpu[0];
    EXPECT_GT(gpu[N - 1], 1.0f)
        << "Cell 99 (neg weights, low target): expected NES > 1.0, got " << gpu[N - 1];

    std::printf("[REGISTRY] %s | enrich/decoupler_viper/neg_weights | 50x100x1 | "
                "sign_check | c0=%.3f c99=%.3f | c0<-1 c99>1 | none | CYCLE-137 | %s\n",
                __DATE__, gpu[0], gpu[N - 1],
                (gpu[0] < -1.0f && gpu[N - 1] > 1.0f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 4: Viper_Determinism_BitIdentical
//
// Two consecutive viper() calls on the same data must produce bit-identical results
// (rel_err = 0). Sort-based ranking + cuBLAS Sgemm at fp32 are deterministic
// for fixed inputs on the same architecture.
// =============================================================================
TEST(ViperTest, Viper_Determinism_BitIdentical)
{
    constexpr int M = 30, N = 50, P = 4;

    std::mt19937 rng(0x13579BDF);
    std::poisson_distribution<int>         pd(3);
    std::uniform_real_distribution<float>  uw(-1.f, 1.f);

    std::vector<float> X_row(M * N);
    for (auto& v : X_row) v = static_cast<float>(pd(rng));
    std::vector<float> W_h(M * P);
    for (auto& v : W_h) v = uw(rng);

    auto csc = dense_to_csc(M, N, X_row);
    singlet_gpu::enrich::ViperConfig cfg;

    auto pz1 = upload_csc(csc);
    auto dW1 = upload_W(W_h);
    auto r1  = singlet_gpu::enrich::viper(pz1, dW1.get(), P, cfg);
    auto h1  = d2h(r1.nes, N * P);

    auto pz2 = upload_csc(csc);
    auto dW2 = upload_W(W_h);
    auto r2  = singlet_gpu::enrich::viper(pz2, dW2.get(), P, cfg);
    auto h2  = d2h(r2.nes, N * P);

    float max_rel = 0.f;
    for (int i = 0; i < N * P; ++i) {
        float denom = std::max(std::fabs(h1[i]), 1e-8f);
        float rel   = std::fabs(h1[i] - h2[i]) / denom;
        max_rel     = std::max(max_rel, rel);
        EXPECT_EQ(h1[i], h2[i])
            << "Determinism: runs differ at element " << i
            << " (h1=" << h1[i] << " h2=" << h2[i] << ")";
    }

    std::printf("[REGISTRY] %s | enrich/decoupler_viper/determinism | 30x50x4 | "
                "rel_err | %.2e | 0 | none | CYCLE-137 | %s\n",
                __DATE__, max_rel, (max_rel == 0.f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 5: Viper_MemoryGuard_RejectsTooLarge
//
// Synthetic dimensions that imply a T1 buffer > 32 GB:
//   m = 100000, n = 100000 → T1 = 100k × 100k × 4 bytes = 40 GB > 32 GB.
// viper() must throw std::runtime_error before any device allocation.
// =============================================================================
TEST(ViperTest, Viper_MemoryGuard_RejectsTooLarge)
{
    // Build a tiny valid CSC but override mat.rows/cols to large synthetic dims.
    // We use a 2×2 matrix as the real CSC, then patch the dims to trigger the guard.
    const int M_real = 2, N_real = 2;
    std::vector<float> X_tiny = { 1.f, 0.f, 0.f, 1.f };  // 2×2 identity (row-major)
    auto csc_tiny = dense_to_csc(M_real, N_real, X_tiny);
    auto pz_tiny  = upload_csc(csc_tiny);

    // Override dimensions in the PzDeviceMatrix to synthetic large values.
    // This is for the test only — the actual CSC data is tiny (no OOM risk).
    pz_tiny.mat.rows = 100000;  // 100k genes
    pz_tiny.mat.cols = 100000;  // 100k cells → 40 GB T1 > 32 GB guard

    singlet_gpu::core::DeviceMemory<float> d_W_tiny(2);  // minimal W (2 entries)
    float w_val = 1.f;
    cudaMemcpy(d_W_tiny.get(), &w_val, sizeof(float), cudaMemcpyHostToDevice);

    singlet_gpu::enrich::ViperConfig cfg;
    cfg.max_dense_gb = 32.f;  // default guard

    bool threw = false;
    try {
        auto res = singlet_gpu::enrich::viper(pz_tiny, d_W_tiny.get(), 1, cfg);
    } catch (const std::runtime_error& e) {
        threw = true;
        std::printf("  [Expected exception]: %s\n", e.what());
    }

    EXPECT_TRUE(threw)
        << "MemoryGuard: expected runtime_error for >32 GB T1 buffer, but none was thrown";

    std::printf("[REGISTRY] %s | enrich/decoupler_viper/memory_guard | 100k×100k | "
                "throws | %s | yes | none | CYCLE-137 | %s\n",
                __DATE__, threw ? "yes" : "no", threw ? "PASS" : "FAIL");
}
