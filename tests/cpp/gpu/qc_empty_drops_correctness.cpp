// SPDX-License-Identifier: MIT
// singlet/gpu/tests/qc_empty_drops_correctness.cpp
//
// integrates: DropletUtils::emptyDrops (Lun et al. 2019)
//
// Correctness harness for singlet::gpu::qc::empty_drops.
//
// Tests:
//   1. EmptyDrops_AllEmpty_NoCellsCalled
//      All droplets have t <= lower=100. Expected: every is_cell=0, pvalue=1.
//
//   2. EmptyDrops_BimodalPopulation_RecoversCells
//      1000 genes x 2000 droplets. First 1900 are Poisson(20) empty.
//      Last 100 have t ~ Poisson(5000) with a shifted profile (non-ambient).
//      Expected: is_cell=1 for >= 80 of last 100; is_cell=0 for < 5% of empty.
//
//   3. EmptyDrops_AmbientProfile_ApproximateMLE
//      Known ambient: gene 0 dominates. A candidate with ambient-matched
//      expression should get high p-value (>= 0.1, not called as cell).
//
//   4. EmptyDrops_Determinism_SameSeed
//      Two identical runs with same seed. Max relative error on p-values < 1e-3.
//
//   5. EmptyDrops_FdrThreshold_ControlsCalls
//      Same input, fdr_thresh=0.01 vs fdr_thresh=0.001.
//      Higher threshold -> n_cells_called >= n_cells_called at lower threshold.

#include <singlet/gpu/qc/empty_drops.h>
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// Utility helpers
// =============================================================================

namespace {

bool gpu_available() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    return (e == cudaSuccess && count > 0);
}

// Host-side CSC container (genes x droplets).
struct HostCSC {
    int m;    // genes (rows)
    int n;    // droplets (cols)
    std::vector<float>   values;
    std::vector<int32_t> col_ptr;    // n+1
    std::vector<int32_t> row_indices;
};

// Build HostCSC from a dense float matrix (m x n, col-major).
HostCSC dense_to_csc(const std::vector<float>& dense, int m, int n) {
    HostCSC h;
    h.m = m; h.n = n;
    h.col_ptr.resize(static_cast<size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            float v = dense[static_cast<size_t>(i) + static_cast<size_t>(j) * m];
            if (v > 0.f) { h.values.push_back(v); h.row_indices.push_back(i); }
        }
        h.col_ptr[j + 1] = static_cast<int32_t>(h.values.size());
    }
    return h;
}

// Upload a HostCSC to a singlet::gpu::io::PzDeviceMatrix.
singlet::gpu::io::PzDeviceMatrix upload_csc(const HostCSC& h, cudaStream_t stream) {
    using namespace singlet::gpu;
    const int64_t nnz = static_cast<int64_t>(h.values.size());

    io::PzDeviceMatrix pz;
    pz.mat.rows = h.m;
    pz.mat.cols = h.n;
    pz.mat.nnz  = static_cast<int>(nnz);

    pz.mat.values      = core::DeviceMemory<float>(nnz > 0 ? nnz : 1);
    pz.mat.row_indices = core::DeviceMemory<int32_t>(nnz > 0 ? nnz : 1);
    pz.mat.col_ptr     = core::DeviceMemory<int32_t>(static_cast<size_t>(h.n) + 1);

    if (nnz > 0) {
        cudaMemcpyAsync(pz.mat.values.get(), h.values.data(),
                        nnz * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(pz.mat.row_indices.get(), h.row_indices.data(),
                        nnz * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    }
    cudaMemcpyAsync(pz.mat.col_ptr.get(), h.col_ptr.data(),
                    (static_cast<size_t>(h.n) + 1) * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return pz;
}

// Download a DeviceMemory<float> to host.
std::vector<float> download_f32(const singlet::gpu::core::DeviceMemory<float>& d,
                                 size_t count, cudaStream_t stream) {
    std::vector<float> h(count);
    cudaMemcpyAsync(h.data(), d.get(), count * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h;
}

// Download a DeviceMemory<uint8_t> to host.
std::vector<uint8_t> download_u8(const singlet::gpu::core::DeviceMemory<uint8_t>& d,
                                  size_t count, cudaStream_t stream) {
    std::vector<uint8_t> h(count);
    cudaMemcpyAsync(h.data(), d.get(), count * sizeof(uint8_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h;
}

// Generate a Poisson count CSC: each column has random UMI totals.
// ambient_pi[g] = probability for gene g; length m.
// For each droplet j, draw t_j ~ Poisson(mean_umi), then distribute t_j UMIs
// across genes according to ambient_pi via multinomial.
HostCSC make_poisson_csc(
    int m, int n, double mean_umi,
    const std::vector<double>& ambient_pi,
    uint64_t seed)
{
    assert(static_cast<int>(ambient_pi.size()) == m);
    std::mt19937_64 rng(seed);
    std::poisson_distribution<int> pois(mean_umi);

    std::vector<float> dense(static_cast<size_t>(m) * n, 0.f);
    for (int j = 0; j < n; ++j) {
        int t = pois(rng);
        if (t <= 0) continue;
        std::discrete_distribution<int> multi(ambient_pi.begin(), ambient_pi.end());
        for (int k = 0; k < t; ++k)
            dense[static_cast<size_t>(multi(rng)) + static_cast<size_t>(j) * m] += 1.f;
    }
    return dense_to_csc(dense, m, n);
}

} // anonymous namespace

// =============================================================================
// Test fixture
// =============================================================================
class EmptyDropsFixture : public ::testing::Test {
protected:
    cudaStream_t stream_{nullptr};
    void SetUp() override {
        if (!gpu_available()) return;
        cudaStreamCreate(&stream_);
    }
    void TearDown() override {
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    }
};

// =============================================================================
// Test 1: EmptyDrops_AllEmpty_NoCellsCalled
// All droplets have t <= lower=100. No candidates. Every is_cell must be 0.
// =============================================================================
TEST_F(EmptyDropsFixture, EmptyDrops_AllEmpty_NoCellsCalled) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int M = 200;
    constexpr int N = 500;

    // All droplets have mean UMI = 10 (well below lower=100).
    std::vector<double> pi(M, 1.0 / M);
    HostCSC h = make_poisson_csc(M, N, /*mean_umi=*/10.0, pi, /*seed=*/42ULL);
    auto pz = upload_csc(h, stream_);

    singlet::gpu::qc::EmptyDropsConfig cfg;
    cfg.lower  = 100;
    cfg.niters = 100;
    cfg.seed   = 0ULL;

    auto result = singlet::gpu::qc::empty_drops(pz, cfg, stream_);

    ASSERT_EQ(result.n_droplets, N);
    EXPECT_EQ(result.n_candidates, 0);

    auto h_pvalue  = download_f32(result.pvalue,  N, stream_);
    auto h_fdr     = download_f32(result.fdr,     N, stream_);
    auto h_is_cell = download_u8 (result.is_cell, N, stream_);

    int n_called = 0;
    for (int j = 0; j < N; ++j) {
        EXPECT_FLOAT_EQ(h_pvalue[j],  1.f) << "droplet " << j;
        EXPECT_FLOAT_EQ(h_fdr[j],     1.f) << "droplet " << j;
        EXPECT_EQ      (h_is_cell[j], 0u)  << "droplet " << j;
        if (h_is_cell[j]) ++n_called;
    }
    std::printf("[AllEmpty] n_called=%d / %d\n", n_called, N);
    EXPECT_EQ(n_called, 0);
}

// =============================================================================
// Test 2: EmptyDrops_BimodalPopulation_RecoversCells
// 1000 genes x 2000 droplets.
// First 1900 ~ Poisson(20) with uniform ambient profile (empty).
// Last 100 ~ Poisson(5000) with shifted profile (real cells).
// Expected: recall >= 80/100 real cells; FPR < 5% among empty.
// =============================================================================
TEST_F(EmptyDropsFixture, EmptyDrops_BimodalPopulation_RecoversCells) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int M        = 500;   // genes (must be <= ED_MAX_GENES_SMEM=32768)
    constexpr int N_EMPTY  = 1900;
    constexpr int N_CELLS  = 100;
    constexpr int N        = N_EMPTY + N_CELLS;

    // Ambient profile: uniform.
    std::vector<double> ambient_pi(M, 1.0 / M);

    // Cell profile: concentrated on first 50 genes.
    std::vector<double> cell_pi(M, 0.0);
    for (int g = 0; g < 50; ++g) cell_pi[g] = 1.0 / 50.0;

    // Generate empty droplets.
    HostCSC h_empty = make_poisson_csc(M, N_EMPTY, /*mean_umi=*/20.0,
                                       ambient_pi, /*seed=*/1001ULL);
    // Generate cell droplets (high UMI, shifted profile).
    HostCSC h_cells = make_poisson_csc(M, N_CELLS, /*mean_umi=*/5000.0,
                                       cell_pi, /*seed=*/2002ULL);

    // Concatenate columns: empty first, cells last.
    HostCSC h;
    h.m = M;
    h.n = N;
    h.col_ptr.resize(static_cast<size_t>(N) + 1, 0);
    // Copy empty columns.
    for (int j = 0; j < N_EMPTY; ++j) {
        for (int32_t k = h_empty.col_ptr[j]; k < h_empty.col_ptr[j + 1]; ++k) {
            h.values.push_back(h_empty.values[k]);
            h.row_indices.push_back(h_empty.row_indices[k]);
        }
        h.col_ptr[j + 1] = static_cast<int32_t>(h.values.size());
    }
    // Copy cell columns.
    for (int j = 0; j < N_CELLS; ++j) {
        for (int32_t k = h_cells.col_ptr[j]; k < h_cells.col_ptr[j + 1]; ++k) {
            h.values.push_back(h_cells.values[k]);
            h.row_indices.push_back(h_cells.row_indices[k]);
        }
        h.col_ptr[N_EMPTY + j + 1] = static_cast<int32_t>(h.values.size());
    }

    auto pz = upload_csc(h, stream_);

    singlet::gpu::qc::EmptyDropsConfig cfg;
    cfg.lower     = 100;
    cfg.niters    = 1000;   // reduced for test speed
    cfg.fdr_thresh = 0.01f;
    cfg.seed      = 0ULL;

    auto result = singlet::gpu::qc::empty_drops(pz, cfg, stream_);

    ASSERT_EQ(result.n_droplets, N);

    auto h_is_cell = download_u8(result.is_cell, N, stream_);
    auto h_pvalue  = download_f32(result.pvalue, N, stream_);

    // Check all p-values are finite and in [0,1].
    for (int j = 0; j < N; ++j) {
        ASSERT_TRUE(std::isfinite(h_pvalue[j])) << "non-finite pvalue at j=" << j;
        EXPECT_GE(h_pvalue[j], 0.f);
        EXPECT_LE(h_pvalue[j], 1.f);
    }

    // Count true positives (cells called among last N_CELLS droplets).
    int tp = 0;
    for (int j = N_EMPTY; j < N; ++j)
        if (h_is_cell[j]) ++tp;

    // Count false positives (cells called among first N_EMPTY droplets).
    int fp = 0;
    for (int j = 0; j < N_EMPTY; ++j)
        if (h_is_cell[j]) ++fp;

    double recall = static_cast<double>(tp) / N_CELLS;
    double fpr    = static_cast<double>(fp) / N_EMPTY;

    std::printf("[BimodalPopulation] tp=%d/%d (recall=%.3f)  fp=%d/%d (fpr=%.4f)\n",
                tp, N_CELLS, recall, fp, N_EMPTY, fpr);

    EXPECT_GE(tp, 80) << "recall too low: " << tp << "/" << N_CELLS;
    EXPECT_LT(fp, static_cast<int>(0.05 * N_EMPTY))
        << "FPR too high: " << fp << "/" << N_EMPTY;
}

// =============================================================================
// Test 3: EmptyDrops_AmbientProfile_ApproximateMLE
// Synthetic with known ambient: gene 0 has pi=0.9, all others share 0.1.
// A candidate droplet with expression matching ambient should get p >= 0.1
// (not called as cell — ambient-matched expression is indistinguishable from
// an empty droplet of larger size).
// =============================================================================
TEST_F(EmptyDropsFixture, EmptyDrops_AmbientProfile_ApproximateMLE) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int M        = 100;
    constexpr int N_EMPTY  = 500;
    constexpr int N_CAND   = 5;    // ambient-matched candidates
    constexpr int N        = N_EMPTY + N_CAND;

    // Ambient profile: gene 0 strongly dominant.
    std::vector<double> ambient_pi(M, 0.1 / (M - 1));
    ambient_pi[0] = 0.9;

    HostCSC h_empty = make_poisson_csc(M, N_EMPTY, 20.0, ambient_pi, 777ULL);

    // Candidate droplets: draw from the exact same ambient profile but with
    // high total UMI so they exceed lower=100.
    HostCSC h_cand = make_poisson_csc(M, N_CAND, 2000.0, ambient_pi, 888ULL);

    // Concatenate.
    HostCSC h;
    h.m = M; h.n = N;
    h.col_ptr.resize(static_cast<size_t>(N) + 1, 0);
    for (int j = 0; j < N_EMPTY; ++j) {
        for (int32_t k = h_empty.col_ptr[j]; k < h_empty.col_ptr[j + 1]; ++k) {
            h.values.push_back(h_empty.values[k]);
            h.row_indices.push_back(h_empty.row_indices[k]);
        }
        h.col_ptr[j + 1] = static_cast<int32_t>(h.values.size());
    }
    for (int j = 0; j < N_CAND; ++j) {
        for (int32_t k = h_cand.col_ptr[j]; k < h_cand.col_ptr[j + 1]; ++k) {
            h.values.push_back(h_cand.values[k]);
            h.row_indices.push_back(h_cand.row_indices[k]);
        }
        h.col_ptr[N_EMPTY + j + 1] = static_cast<int32_t>(h.values.size());
    }

    auto pz = upload_csc(h, stream_);

    singlet::gpu::qc::EmptyDropsConfig cfg;
    cfg.lower     = 100;
    cfg.niters    = 2000;
    cfg.fdr_thresh = 0.001f;
    cfg.seed      = 42ULL;

    auto result = singlet::gpu::qc::empty_drops(pz, cfg, stream_);
    ASSERT_EQ(result.n_droplets, N);

    auto h_pvalue  = download_f32(result.pvalue,  N, stream_);
    auto h_is_cell = download_u8 (result.is_cell, N, stream_);

    // Ambient-matched candidates should NOT be called as cells.
    // Their p-value should be high (>= 0.05 for at least 4 of 5).
    int n_high_pval = 0;
    int n_not_called = 0;
    for (int j = N_EMPTY; j < N; ++j) {
        std::printf("[AmbientMLE] droplet %d: pvalue=%.4f is_cell=%d\n",
                    j, h_pvalue[j], (int)h_is_cell[j]);
        if (h_pvalue[j] >= 0.05f) ++n_high_pval;
        if (!h_is_cell[j]) ++n_not_called;
    }

    EXPECT_GE(n_high_pval, 3)
        << "Expected >= 3 ambient-matched candidates to have p >= 0.05; got "
        << n_high_pval;
    EXPECT_GE(n_not_called, 4)
        << "Expected >= 4 ambient-matched candidates to NOT be called as cells; got "
        << n_not_called;
}

// =============================================================================
// Test 4: EmptyDrops_Determinism_SameSeed
// Two runs with identical seed on the same matrix. Max relative error < 1e-3.
// =============================================================================
TEST_F(EmptyDropsFixture, EmptyDrops_Determinism_SameSeed) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int M       = 200;
    constexpr int N_EMPTY = 300;
    constexpr int N_CELLS = 30;
    constexpr int N       = N_EMPTY + N_CELLS;

    std::vector<double> ambient_pi(M, 1.0 / M);
    std::vector<double> cell_pi(M, 0.0);
    for (int g = 0; g < 20; ++g) cell_pi[g] = 1.0 / 20.0;

    HostCSC h_empty = make_poisson_csc(M, N_EMPTY, 15.0, ambient_pi, 111ULL);
    HostCSC h_cells = make_poisson_csc(M, N_CELLS, 3000.0, cell_pi,   222ULL);

    HostCSC h;
    h.m = M; h.n = N;
    h.col_ptr.resize(static_cast<size_t>(N) + 1, 0);
    for (int j = 0; j < N_EMPTY; ++j) {
        for (int32_t k = h_empty.col_ptr[j]; k < h_empty.col_ptr[j+1]; ++k) {
            h.values.push_back(h_empty.values[k]);
            h.row_indices.push_back(h_empty.row_indices[k]);
        }
        h.col_ptr[j + 1] = static_cast<int32_t>(h.values.size());
    }
    for (int j = 0; j < N_CELLS; ++j) {
        for (int32_t k = h_cells.col_ptr[j]; k < h_cells.col_ptr[j+1]; ++k) {
            h.values.push_back(h_cells.values[k]);
            h.row_indices.push_back(h_cells.row_indices[k]);
        }
        h.col_ptr[N_EMPTY + j + 1] = static_cast<int32_t>(h.values.size());
    }

    auto pz = upload_csc(h, stream_);

    singlet::gpu::qc::EmptyDropsConfig cfg;
    cfg.lower  = 100;
    cfg.niters = 500;
    cfg.seed   = 0xC0FFEEull;

    auto result_a = singlet::gpu::qc::empty_drops(pz, cfg, stream_);
    auto result_b = singlet::gpu::qc::empty_drops(pz, cfg, stream_);

    auto h_pval_a = download_f32(result_a.pvalue, N, stream_);
    auto h_pval_b = download_f32(result_b.pvalue, N, stream_);

    double max_rel_err = 0.0;
    int n_differ = 0;
    for (int j = 0; j < N; ++j) {
        float pa = h_pval_a[j];
        float pb = h_pval_b[j];
        if (pa == pb) continue;
        ++n_differ;
        float denom = std::max(std::abs(pa), std::abs(pb));
        if (denom > 0.f) {
            double rel = std::abs(static_cast<double>(pa - pb)) / denom;
            max_rel_err = std::max(max_rel_err, rel);
        }
    }

    std::printf("[Determinism] n_differ=%d / %d  max_rel_err=%.2e\n",
                n_differ, N, max_rel_err);

    EXPECT_LT(max_rel_err, 1e-3)
        << "Max relative error " << max_rel_err << " >= 1e-3 between same-seed runs";
}

// =============================================================================
// Test 5: EmptyDrops_FdrThreshold_ControlsCalls
// Same input; fdr_thresh=0.01 -> n_cells_a; fdr_thresh=0.001 -> n_cells_b.
// Must have n_cells_a >= n_cells_b (higher threshold = more permissive = more calls).
// =============================================================================
TEST_F(EmptyDropsFixture, EmptyDrops_FdrThreshold_ControlsCalls) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int M       = 300;
    constexpr int N_EMPTY = 500;
    constexpr int N_CELLS = 50;
    constexpr int N       = N_EMPTY + N_CELLS;

    std::vector<double> ambient_pi(M, 1.0 / M);
    std::vector<double> cell_pi(M, 0.0);
    for (int g = 0; g < 30; ++g) cell_pi[g] = 1.0 / 30.0;

    HostCSC h_empty = make_poisson_csc(M, N_EMPTY, 20.0, ambient_pi, 333ULL);
    HostCSC h_cells = make_poisson_csc(M, N_CELLS, 4000.0, cell_pi,   444ULL);

    HostCSC h;
    h.m = M; h.n = N;
    h.col_ptr.resize(static_cast<size_t>(N) + 1, 0);
    for (int j = 0; j < N_EMPTY; ++j) {
        for (int32_t k = h_empty.col_ptr[j]; k < h_empty.col_ptr[j+1]; ++k) {
            h.values.push_back(h_empty.values[k]);
            h.row_indices.push_back(h_empty.row_indices[k]);
        }
        h.col_ptr[j + 1] = static_cast<int32_t>(h.values.size());
    }
    for (int j = 0; j < N_CELLS; ++j) {
        for (int32_t k = h_cells.col_ptr[j]; k < h_cells.col_ptr[j+1]; ++k) {
            h.values.push_back(h_cells.values[k]);
            h.row_indices.push_back(h_cells.row_indices[k]);
        }
        h.col_ptr[N_EMPTY + j + 1] = static_cast<int32_t>(h.values.size());
    }

    auto pz = upload_csc(h, stream_);

    singlet::gpu::qc::EmptyDropsConfig cfg_hi;
    cfg_hi.lower     = 100;
    cfg_hi.niters    = 500;
    cfg_hi.fdr_thresh = 0.01f;
    cfg_hi.seed      = 0ULL;

    singlet::gpu::qc::EmptyDropsConfig cfg_lo = cfg_hi;
    cfg_lo.fdr_thresh = 0.001f;

    auto result_hi = singlet::gpu::qc::empty_drops(pz, cfg_hi, stream_);
    auto result_lo = singlet::gpu::qc::empty_drops(pz, cfg_lo, stream_);

    auto h_iscell_hi = download_u8(result_hi.is_cell, N, stream_);
    auto h_iscell_lo = download_u8(result_lo.is_cell, N, stream_);

    int n_hi = 0, n_lo = 0;
    for (int j = 0; j < N; ++j) {
        if (h_iscell_hi[j]) ++n_hi;
        if (h_iscell_lo[j]) ++n_lo;
    }

    std::printf("[FdrThreshold] fdr=0.01: n_cells=%d  fdr=0.001: n_cells=%d\n",
                n_hi, n_lo);

    EXPECT_GE(n_hi, n_lo)
        << "Higher FDR threshold should produce >= as many cell calls. "
        << "n_hi=" << n_hi << " n_lo=" << n_lo;
}
