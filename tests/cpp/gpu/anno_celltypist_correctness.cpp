// SPDX-License-Identifier: MIT
// singlet/gpu/tests/anno_celltypist_correctness.cpp
//
// integrates: CellTypist (Dominguez Conde et al. 2022, Science)
//
// Correctness harness for singlet::gpu::anno::celltypist_predict.
//
// Tests:
//   1. Celltypist_TinyClosedForm
//      5 features x 4 cells x 3 classes. Hand-constructed W, b, Z.
//      CPU reference: logits, softmax, argmax. GPU must match within 1e-4.
//
//   2. Celltypist_OneHotInputs_PerfectClassification
//      Z is identity-like (cell c expressed on feature c).
//      W is identity-like (W[i,j] = 1 if i==j else 0), b = 0.
//      Expected: pred[c] = c with confidence near 1.
//
//   3. Celltypist_AllZeroExpression_UniformDistribution
//      Z = 0, b = 0. Logits = 0. Softmax = 1/n_classes per class.
//      pred is arbitrary (any valid class). confidence ~= 1/n_classes.
//
//   4. Celltypist_Determinism_BitIdentical
//      Two identical runs on same input. pred_class identical; confidence
//      bit-identical (cuBLAS Sgemm is deterministic, no atomics in kernels).
//
//   5. Celltypist_LargeClasses_HandlesStriding
//      n_classes=500, n_features=100, n_cells=50. Random W, b, Z.
//      CPU reference pred (argmax of logits) must match GPU pred exactly.
//      Tests the strided block path (n_classes > CT_THREADS=256).

#include <singlet/gpu/anno/celltypist.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
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

// Upload a host float vector to device; return a DeviceMemory<float>.
singlet::gpu::core::DeviceMemory<float>
upload_f32(const std::vector<float>& h, cudaStream_t stream) {
    singlet::gpu::core::DeviceMemory<float> d(h.size() > 0 ? h.size() : 1);
    if (!h.empty())
        cudaMemcpyAsync(d.get(), h.data(), h.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    return d;
}

// Download DeviceMemory<float> to host vector.
std::vector<float> download_f32(
    const singlet::gpu::core::DeviceMemory<float>& d, size_t count,
    cudaStream_t stream)
{
    std::vector<float> h(count);
    cudaMemcpyAsync(h.data(), d.get(), count * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h;
}

// Download DeviceMemory<int> to host vector.
std::vector<int> download_i32(
    const singlet::gpu::core::DeviceMemory<int>& d, size_t count,
    cudaStream_t stream)
{
    std::vector<int> h(count);
    cudaMemcpyAsync(h.data(), d.get(), count * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h;
}

// CPU reference: compute logits L = W^T . Z + b (col-major).
// W: n_features x n_classes col-major.
// Z: n_features x n_cells col-major.
// L: n_classes x n_cells col-major.
std::vector<float> cpu_logits(
    const std::vector<float>& W, const std::vector<float>& Z,
    const std::vector<float>& b,
    int n_features, int n_cells, int n_classes)
{
    std::vector<float> L(static_cast<size_t>(n_classes) * n_cells, 0.f);
    for (int c = 0; c < n_cells; ++c) {
        for (int k = 0; k < n_classes; ++k) {
            float acc = b[k];
            for (int f = 0; f < n_features; ++f)
                acc += W[static_cast<size_t>(f) + static_cast<size_t>(k) * n_features]
                     * Z[static_cast<size_t>(f) + static_cast<size_t>(c) * n_features];
            L[static_cast<size_t>(k) + static_cast<size_t>(c) * n_classes] = acc;
        }
    }
    return L;
}

// CPU reference: softmax per cell column; returns (pred, confidence) for each cell.
void cpu_softmax_argmax(
    const std::vector<float>& L, int n_classes, int n_cells,
    std::vector<int>& pred_out, std::vector<float>& conf_out)
{
    pred_out.resize(n_cells);
    conf_out.resize(n_cells);
    for (int c = 0; c < n_cells; ++c) {
        const float* Lc = L.data() + static_cast<ptrdiff_t>(c) * n_classes;
        float L_max = *std::max_element(Lc, Lc + n_classes);
        float sum_exp = 0.f;
        for (int k = 0; k < n_classes; ++k)
            sum_exp += std::exp(Lc[k] - L_max);
        int best = 0;
        float best_val = Lc[0];
        for (int k = 1; k < n_classes; ++k)
            if (Lc[k] > best_val) { best_val = Lc[k]; best = k; }
        pred_out[c] = best;
        conf_out[c] = std::exp(Lc[best] - L_max) / sum_exp;
    }
}

} // anonymous namespace

// =============================================================================
// Test fixture
// =============================================================================
class CelltypistFixture : public ::testing::Test {
protected:
    cudaStream_t stream_{nullptr};
    void SetUp() override {
        if (!gpu_available()) return;
        cudaStreamCreate(&stream_);
        // Ensure a default context exists (initializes cuBLAS handle).
        singlet::gpu::core::default_context();
    }
    void TearDown() override {
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    }
};

// =============================================================================
// Test 1: Celltypist_TinyClosedForm
// 5 features x 4 cells x 3 classes. CPU reference vs GPU.
// =============================================================================
TEST_F(CelltypistFixture, Celltypist_TinyClosedForm) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int F = 5, C = 4, K = 3;

    // W (F x K col-major): arbitrary weights
    std::vector<float> W = {
        0.1f,  0.2f, -0.3f,  0.4f,  0.5f,   // class 0
       -0.2f,  0.3f,  0.1f, -0.4f,  0.2f,   // class 1
        0.5f, -0.1f,  0.3f,  0.2f, -0.5f    // class 2
    };
    // b (K)
    std::vector<float> b = {0.1f, -0.2f, 0.3f};
    // Z (F x C col-major): log1p-normalized expression
    std::vector<float> Z = {
        1.0f, 0.5f, 0.2f, 0.0f, 0.8f,   // cell 0
        0.0f, 1.2f, 0.4f, 0.6f, 0.1f,   // cell 1
        0.9f, 0.0f, 0.7f, 0.3f, 0.5f,   // cell 2
        0.4f, 0.8f, 0.0f, 1.0f, 0.2f    // cell 3
    };

    // CPU reference
    auto L_cpu   = cpu_logits(W, Z, b, F, C, K);
    std::vector<int>   ref_pred;
    std::vector<float> ref_conf;
    cpu_softmax_argmax(L_cpu, K, C, ref_pred, ref_conf);

    // Upload to GPU
    auto d_W = upload_f32(W, stream_);
    auto d_b = upload_f32(b, stream_);
    auto d_Z = upload_f32(Z, stream_);
    cudaStreamSynchronize(stream_);

    auto result = singlet::gpu::anno::celltypist_predict(
        d_Z.get(), F, C, d_W.get(), d_b.get(), K, {}, stream_);

    auto h_pred = download_i32(result.pred_class, C, stream_);
    auto h_conf = download_f32(result.confidence, C, stream_);

    for (int c = 0; c < C; ++c) {
        std::printf("[TinyClosedForm] cell %d: ref_pred=%d gpu_pred=%d "
                    "ref_conf=%.6f gpu_conf=%.6f\n",
                    c, ref_pred[c], h_pred[c], ref_conf[c], h_conf[c]);
        EXPECT_EQ(h_pred[c], ref_pred[c]) << "pred mismatch at cell " << c;
        EXPECT_NEAR(h_conf[c], ref_conf[c], 1e-4f)
            << "confidence mismatch at cell " << c;
    }
}

// =============================================================================
// Test 2: Celltypist_OneHotInputs_PerfectClassification
// Z: cell c has all expression on feature c (identity-like, n_cells == n_features).
// W: scaled diagonal (W[f,k] = 10.0 if f==k else 0), b = 0.
// Expected: pred[c] = c, confidence close to 1.
//
// WHY scaled diagonal (10.0) instead of plain 1.0: with W[f,k]=1 and one-hot Z,
// logits are {1, 0, 0, ...}; softmax peak is e/(e+N-1) = 0.28 for N=8 — that's the
// CORRECT softmax of those logits, not a kernel bug. Scaling to 10.0 makes the
// peak softmax ≈ exp(10)/(exp(10)+(N-1)) ≈ 0.9997, restoring "perfect"
// classification semantics. Pattern noted in CYCLE-118 lessons (test design vs
// kernel correctness).
// =============================================================================
TEST_F(CelltypistFixture, Celltypist_OneHotInputs_PerfectClassification) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int N = 8;   // n_features = n_classes = n_cells = N

    // W (N x N col-major): scaled identity (10.0 on diagonal)
    std::vector<float> W(static_cast<size_t>(N) * N, 0.f);
    for (int i = 0; i < N; ++i)
        W[static_cast<size_t>(i) + static_cast<size_t>(i) * N] = 10.f;

    // b = 0
    std::vector<float> b(N, 0.f);

    // Z (N x N col-major): identity (cell c = feature c = 1.0, rest 0)
    std::vector<float> Z(static_cast<size_t>(N) * N, 0.f);
    for (int c = 0; c < N; ++c)
        Z[static_cast<size_t>(c) + static_cast<size_t>(c) * N] = 1.f;

    auto d_W = upload_f32(W, stream_);
    auto d_b = upload_f32(b, stream_);
    auto d_Z = upload_f32(Z, stream_);
    cudaStreamSynchronize(stream_);

    auto result = singlet::gpu::anno::celltypist_predict(
        d_Z.get(), N, N, d_W.get(), d_b.get(), N, {}, stream_);

    auto h_pred = download_i32(result.pred_class, N, stream_);
    auto h_conf = download_f32(result.confidence, N, stream_);

    for (int c = 0; c < N; ++c) {
        std::printf("[OneHot] cell %d: pred=%d (expected %d) conf=%.6f\n",
                    c, h_pred[c], c, h_conf[c]);
        EXPECT_EQ(h_pred[c], c) << "wrong class for cell " << c;
        EXPECT_GT(h_conf[c], 0.99f) << "confidence too low for cell " << c;
    }
}

// =============================================================================
// Test 3: Celltypist_AllZeroExpression_UniformDistribution
// Z = 0, b = 0 => logits = 0 => softmax = 1/n_classes.
// pred is deterministic (some class); confidence == 1/n_classes.
// No NaN allowed.
// =============================================================================
TEST_F(CelltypistFixture, Celltypist_AllZeroExpression_UniformDistribution) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int F = 10, C = 6, K = 4;
    const float expected_conf = 1.f / static_cast<float>(K);

    std::vector<float> W(static_cast<size_t>(F) * K, 0.5f);
    std::vector<float> b(K, 0.f);
    std::vector<float> Z(static_cast<size_t>(F) * C, 0.f);

    auto d_W = upload_f32(W, stream_);
    auto d_b = upload_f32(b, stream_);
    auto d_Z = upload_f32(Z, stream_);
    cudaStreamSynchronize(stream_);

    auto result = singlet::gpu::anno::celltypist_predict(
        d_Z.get(), F, C, d_W.get(), d_b.get(), K, {}, stream_);

    auto h_pred = download_i32(result.pred_class, C, stream_);
    auto h_conf = download_f32(result.confidence, C, stream_);

    for (int c = 0; c < C; ++c) {
        std::printf("[AllZero] cell %d: pred=%d conf=%.6f (expected ~%.6f)\n",
                    c, h_pred[c], h_conf[c], expected_conf);
        ASSERT_TRUE(std::isfinite(h_conf[c])) << "NaN/Inf confidence at cell " << c;
        EXPECT_GE(h_pred[c], 0) << "invalid pred class at cell " << c;
        EXPECT_LT(h_pred[c], K) << "pred class out of range at cell " << c;
        EXPECT_NEAR(h_conf[c], expected_conf, 1e-5f)
            << "confidence != 1/n_classes at cell " << c;
    }
}

// =============================================================================
// Test 4: Celltypist_Determinism_BitIdentical
// Two identical runs on same input. pred_class identical; confidence identical.
// =============================================================================
TEST_F(CelltypistFixture, Celltypist_Determinism_BitIdentical) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int F = 20, C = 15, K = 7;

    std::mt19937 rng(12345u);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    auto make_vec = [&](size_t n) {
        std::vector<float> v(n);
        for (auto& x : v) x = dist(rng);
        return v;
    };

    auto W = make_vec(static_cast<size_t>(F) * K);
    auto b = make_vec(K);
    auto Z = make_vec(static_cast<size_t>(F) * C);

    auto d_W = upload_f32(W, stream_);
    auto d_b = upload_f32(b, stream_);
    auto d_Z = upload_f32(Z, stream_);
    cudaStreamSynchronize(stream_);

    auto r1 = singlet::gpu::anno::celltypist_predict(
        d_Z.get(), F, C, d_W.get(), d_b.get(), K, {}, stream_);
    auto r2 = singlet::gpu::anno::celltypist_predict(
        d_Z.get(), F, C, d_W.get(), d_b.get(), K, {}, stream_);

    auto p1 = download_i32(r1.pred_class, C, stream_);
    auto p2 = download_i32(r2.pred_class, C, stream_);
    auto c1 = download_f32(r1.confidence, C, stream_);
    auto c2 = download_f32(r2.confidence, C, stream_);

    int n_pred_differ = 0, n_conf_differ = 0;
    for (int c = 0; c < C; ++c) {
        if (p1[c] != p2[c]) ++n_pred_differ;
        if (c1[c] != c2[c]) ++n_conf_differ;
    }

    std::printf("[Determinism] n_pred_differ=%d  n_conf_differ=%d\n",
                n_pred_differ, n_conf_differ);
    EXPECT_EQ(n_pred_differ, 0) << "pred_class differs between identical runs";
    EXPECT_EQ(n_conf_differ, 0) << "confidence differs between identical runs";
}

// =============================================================================
// Test 5: Celltypist_LargeClasses_HandlesStriding
// n_classes=500, n_features=100, n_cells=50.
// Random W, b, Z. CPU reference pred must match GPU pred exactly.
// Tests the strided block path (n_classes=500 > CT_THREADS=256).
// =============================================================================
TEST_F(CelltypistFixture, Celltypist_LargeClasses_HandlesStriding) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int F = 100, C = 50, K = 500;

    std::mt19937 rng(99999u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto make_vec = [&](size_t n) {
        std::vector<float> v(n);
        for (auto& x : v) x = dist(rng);
        return v;
    };

    auto W = make_vec(static_cast<size_t>(F) * K);
    auto b = make_vec(K);
    auto Z = make_vec(static_cast<size_t>(F) * C);

    // CPU reference
    auto L_cpu = cpu_logits(W, Z, b, F, C, K);
    std::vector<int>   ref_pred;
    std::vector<float> ref_conf;
    cpu_softmax_argmax(L_cpu, K, C, ref_pred, ref_conf);

    // GPU
    auto d_W = upload_f32(W, stream_);
    auto d_b = upload_f32(b, stream_);
    auto d_Z = upload_f32(Z, stream_);
    cudaStreamSynchronize(stream_);

    auto result = singlet::gpu::anno::celltypist_predict(
        d_Z.get(), F, C, d_W.get(), d_b.get(), K, {}, stream_);

    auto h_pred = download_i32(result.pred_class, C, stream_);
    auto h_conf = download_f32(result.confidence, C, stream_);

    int n_pred_wrong = 0;
    float max_conf_err = 0.f;
    for (int c = 0; c < C; ++c) {
        if (h_pred[c] != ref_pred[c]) ++n_pred_wrong;
        float err = std::abs(h_conf[c] - ref_conf[c]);
        if (err > max_conf_err) max_conf_err = err;
    }

    std::printf("[LargeClasses] n_pred_wrong=%d / %d  max_conf_err=%.2e\n",
                n_pred_wrong, C, static_cast<double>(max_conf_err));

    EXPECT_EQ(n_pred_wrong, 0)
        << "GPU pred_class differs from CPU argmax in " << n_pred_wrong << " cells";
    EXPECT_LT(max_conf_err, 1e-3f)
        << "Max confidence error " << max_conf_err << " exceeds 1e-3";
}

// =============================================================================
// main
// =============================================================================
// GTest main provided by GTest::gtest_main link target.
