// SPDX-License-Identifier: MIT
// singlet/gpu/tests/anno_symphony_correctness.cpp
//
// integrates: Symphony (Kang et al. 2021, Nat Commun)
//
// Correctness harness for singlet::gpu::anno::symphony_predict.
//
// Tests:
//   1. Symphony_TinyClosedForm
//      3 features x 4 cells x 2 clusters x 2 classes x 2 PCs.
//      Hand-constructed W_pca, C_ref, P_label, mu, sigma, Z_query.
//      CPU reference: standardize → PCA project → distances → soft assign →
//      label transfer → argmax.  GPU must match within 1e-3 abs_err.
//
//   2. Symphony_QueryNearCluster_AssignsClusterLabel
//      10 features x 50 cells x 3 clusters x 3 classes x 3 PCs.
//      Cluster centroids at canonical basis vectors in PCA space.
//      P_label is one-hot per cluster.  Half cells near cluster 0, half near
//      cluster 1.  Expected: pred matches nearest-cluster label.
//
//   3. Symphony_AmbiguousQuery_LowConfidence
//      Query is exactly equidistant from two clusters.
//      Confidence should be lower than the well-separated case.
//
//   4. Symphony_Determinism_BitIdentical
//      Two identical runs.  pred_class identical; confidence bit-identical.
//
//   5. Symphony_DegenerateZeroSigma_Survives
//      sigma_ref has zeros for some features.  With eps_sigma guard,
//      no NaN/Inf in output.

#include <singlet/gpu/anno/symphony.h>
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

singlet::gpu::core::DeviceMemory<float>
upload_f32(const std::vector<float>& h, cudaStream_t stream) {
    singlet::gpu::core::DeviceMemory<float> d(h.size() > 0 ? h.size() : 1);
    if (!h.empty())
        cudaMemcpyAsync(d.get(), h.data(), h.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    return d;
}

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

// CPU reference: full Symphony pipeline.
// All matrices col-major: W_pca (n_features x k_pcs), C_ref (k_clusters x k_pcs),
// P_label (k_clusters x n_classes), Z (n_features x n_cells).
void cpu_symphony(
    const std::vector<float>& Z,     // n_features x n_cells col-major
    const std::vector<float>& W_pca, // n_features x k_pcs col-major
    const std::vector<float>& C_ref, // k_clusters x k_pcs col-major
    const std::vector<float>& P_label, // k_clusters x n_classes col-major
    const std::vector<float>& mu,    // n_features
    const std::vector<float>& sigma, // n_features
    int n_features, int n_cells, int k_pcs, int k_clusters, int n_classes,
    float eps_sigma, float eps_dist,
    std::vector<int>& pred_out, std::vector<float>& conf_out)
{
    // Step 1: standardize
    std::vector<float> Z_std(static_cast<size_t>(n_features) * n_cells);
    for (int c = 0; c < n_cells; ++c)
        for (int g = 0; g < n_features; ++g) {
            float s = sigma[g] < eps_sigma ? eps_sigma : sigma[g];
            Z_std[g + static_cast<ptrdiff_t>(c) * n_features] =
                (Z[g + static_cast<ptrdiff_t>(c) * n_features] - mu[g]) / s;
        }

    // Step 2: PCA project  X_pca = W_pca^T · Z_std  (k_pcs x n_cells)
    std::vector<float> X_pca(static_cast<size_t>(k_pcs) * n_cells, 0.f);
    for (int c = 0; c < n_cells; ++c)
        for (int p = 0; p < k_pcs; ++p) {
            float acc = 0.f;
            for (int g = 0; g < n_features; ++g)
                acc += W_pca[g + static_cast<ptrdiff_t>(p) * n_features]
                     * Z_std[g + static_cast<ptrdiff_t>(c) * n_features];
            X_pca[p + static_cast<ptrdiff_t>(c) * k_pcs] = acc;
        }

    // Step 3: distances
    std::vector<float> dist(static_cast<size_t>(k_clusters) * n_cells);
    for (int c = 0; c < n_cells; ++c) {
        float x_sq = 0.f;
        for (int p = 0; p < k_pcs; ++p) {
            float v = X_pca[p + static_cast<ptrdiff_t>(c) * k_pcs];
            x_sq += v * v;
        }
        for (int k = 0; k < k_clusters; ++k) {
            float c_sq = 0.f, dot = 0.f;
            for (int p = 0; p < k_pcs; ++p) {
                // C_ref (k_clusters x k_pcs) col-major: element (k, p) = C_ref[k + p*k_clusters]
                float cv = C_ref[k + static_cast<ptrdiff_t>(p) * k_clusters];
                c_sq += cv * cv;
                dot  += X_pca[p + static_cast<ptrdiff_t>(c) * k_pcs] * cv;
            }
            float d = x_sq + c_sq - 2.f * dot;
            dist[k + static_cast<ptrdiff_t>(c) * k_clusters] = d > 0.f ? d : 0.f;
        }
    }

    // Step 4: soft assign + normalize
    std::vector<float> s_soft(static_cast<size_t>(k_clusters) * n_cells);
    for (int c = 0; c < n_cells; ++c) {
        float sum_inv = 0.f;
        for (int k = 0; k < k_clusters; ++k) {
            float d = dist[k + static_cast<ptrdiff_t>(c) * k_clusters];
            float inv = 1.f / (d > eps_dist ? d : eps_dist);
            s_soft[k + static_cast<ptrdiff_t>(c) * k_clusters] = inv;
            sum_inv += inv;
        }
        float norm = sum_inv > 0.f ? sum_inv : 1.f;
        for (int k = 0; k < k_clusters; ++k)
            s_soft[k + static_cast<ptrdiff_t>(c) * k_clusters] /= norm;
    }

    // Step 5: label transfer  label_prob = P_label^T · s_soft  (n_classes x n_cells)
    pred_out.resize(n_cells);
    conf_out.resize(n_cells);
    for (int c = 0; c < n_cells; ++c) {
        int   best_cls = 0;
        float best_val = -1.f;
        for (int l = 0; l < n_classes; ++l) {
            float prob = 0.f;
            for (int k = 0; k < k_clusters; ++k)
                // P_label (k_clusters x n_classes) col-major: element (k,l) = P_label[k + l*k_clusters]
                prob += s_soft[k + static_cast<ptrdiff_t>(c) * k_clusters]
                      * P_label[k + static_cast<ptrdiff_t>(l) * k_clusters];
            if (prob > best_val) { best_val = prob; best_cls = l; }
        }
        pred_out[c] = best_cls;
        conf_out[c] = best_val;
    }
}

} // anonymous namespace

// =============================================================================
// Test fixture
// =============================================================================
class SymphonyFixture : public ::testing::Test {
protected:
    cudaStream_t stream_{nullptr};
    void SetUp() override {
        if (!gpu_available()) return;
        cudaStreamCreate(&stream_);
        singlet::gpu::core::default_context();
    }
    void TearDown() override {
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    }
};

// =============================================================================
// Test 1: Symphony_TinyClosedForm
// 3 features x 4 cells x 2 clusters x 2 classes x 2 PCs.
// Hand-constructed inputs; CPU reference vs GPU within 1e-3.
// =============================================================================
TEST_F(SymphonyFixture, Symphony_TinyClosedForm) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int F = 3, C = 4, P = 2, K = 2, L = 2;

    // W_pca (F x P col-major)
    std::vector<float> W_pca = {
        0.8f, 0.4f, 0.2f,   // PC 0
        0.1f, 0.7f, 0.5f    // PC 1
    };
    // C_ref (K x P col-major): cluster k, PC p → C_ref[k + p*K]
    std::vector<float> C_ref = {
        1.5f, -1.0f,   // PC 0: cluster 0 = 1.5, cluster 1 = -1.0
        0.5f,  0.8f    // PC 1: cluster 0 = 0.5, cluster 1 =  0.8
    };
    // P_label (K x L col-major)
    std::vector<float> P_label = {
        0.9f, 0.1f,   // label 0: cluster 0 = 0.9, cluster 1 = 0.1
        0.2f, 0.8f    // label 1: cluster 0 = 0.2, cluster 1 = 0.8
    };
    // mu (F) and sigma (F)
    std::vector<float> mu    = {0.5f, 0.3f, 0.1f};
    std::vector<float> sigma = {1.0f, 0.8f, 0.5f};
    // Z_query (F x C col-major)
    std::vector<float> Z = {
        1.2f, 0.3f, 0.7f,   // cell 0
        0.0f, 1.5f, 0.2f,   // cell 1
        0.8f, 0.6f, 1.0f,   // cell 2
        0.4f, 0.9f, 0.5f    // cell 3
    };

    std::vector<int>   ref_pred;
    std::vector<float> ref_conf;
    cpu_symphony(Z, W_pca, C_ref, P_label, mu, sigma, F, C, P, K, L,
                 1e-3f, 1e-6f, ref_pred, ref_conf);

    auto d_Z      = upload_f32(Z,       stream_);
    auto d_W      = upload_f32(W_pca,   stream_);
    auto d_C      = upload_f32(C_ref,   stream_);
    auto d_P      = upload_f32(P_label, stream_);
    auto d_mu     = upload_f32(mu,      stream_);
    auto d_sigma  = upload_f32(sigma,   stream_);
    cudaStreamSynchronize(stream_);

    auto result = singlet::gpu::anno::symphony_predict(
        d_Z.get(), F, C,
        d_W.get(), P,
        d_C.get(), K,
        d_P.get(), L,
        d_mu.get(), d_sigma.get(),
        {}, stream_);

    auto h_pred = download_i32(result.pred_class, C, stream_);
    auto h_conf = download_f32(result.confidence, C, stream_);

    for (int c = 0; c < C; ++c) {
        std::printf("[TinyClosedForm] cell %d: ref_pred=%d gpu_pred=%d "
                    "ref_conf=%.6f gpu_conf=%.6f\n",
                    c, ref_pred[c], h_pred[c], ref_conf[c], h_conf[c]);
        EXPECT_EQ(h_pred[c], ref_pred[c]) << "pred mismatch at cell " << c;
        EXPECT_NEAR(h_conf[c], ref_conf[c], 1e-3f)
            << "confidence mismatch at cell " << c;
    }
}

// =============================================================================
// Test 2: Symphony_QueryNearCluster_AssignsClusterLabel
// 10 features x 50 cells x 3 clusters x 3 classes x 3 PCs.
// Cluster centroids at canonical basis vectors.  P_label is one-hot.
// =============================================================================
TEST_F(SymphonyFixture, Symphony_QueryNearCluster_AssignsClusterLabel) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int F = 10, C = 50, P = 3, K = 3, L = 3;

    // W_pca (F x P col-major): identity-like first P rows, rest zero.
    std::vector<float> W_pca(static_cast<size_t>(F) * P, 0.f);
    for (int p = 0; p < P; ++p)
        W_pca[p + static_cast<ptrdiff_t>(p) * F] = 1.f;

    // C_ref (K x P col-major): cluster k centroid at e_k in PCA space.
    // C_ref[k + p*K] = 1 if k==p else 0.
    std::vector<float> C_ref(static_cast<size_t>(K) * P, 0.f);
    for (int k = 0; k < K; ++k)
        C_ref[k + static_cast<ptrdiff_t>(k) * K] = 1.f;

    // P_label (K x L col-major): one-hot, cluster k → label k.
    std::vector<float> P_label(static_cast<size_t>(K) * L, 0.f);
    for (int k = 0; k < K; ++k)
        P_label[k + static_cast<ptrdiff_t>(k) * K] = 1.f;

    // mu=0, sigma=1 (no-op standardization).
    std::vector<float> mu(F, 0.f), sigma(F, 1.f);

    // Z (F x C col-major): first 25 cells near cluster 0 (strong feature 0),
    // next 25 cells near cluster 1 (strong feature 1).
    std::mt19937 rng(42u);
    std::uniform_real_distribution<float> noise(-0.05f, 0.05f);
    std::vector<float> Z(static_cast<size_t>(F) * C, 0.f);
    for (int c = 0; c < 25; ++c) {
        Z[0 + static_cast<ptrdiff_t>(c) * F] = 5.f + noise(rng);  // near cluster 0
        for (int g = 1; g < F; ++g)
            Z[g + static_cast<ptrdiff_t>(c) * F] = noise(rng);
    }
    for (int c = 25; c < 50; ++c) {
        Z[1 + static_cast<ptrdiff_t>(c) * F] = 5.f + noise(rng);  // near cluster 1
        for (int g = 0; g < F; ++g)
            if (g != 1) Z[g + static_cast<ptrdiff_t>(c) * F] = noise(rng);
    }

    auto d_Z     = upload_f32(Z,       stream_);
    auto d_W     = upload_f32(W_pca,   stream_);
    auto d_C     = upload_f32(C_ref,   stream_);
    auto d_P     = upload_f32(P_label, stream_);
    auto d_mu    = upload_f32(mu,      stream_);
    auto d_sigma = upload_f32(sigma,   stream_);
    cudaStreamSynchronize(stream_);

    auto result = singlet::gpu::anno::symphony_predict(
        d_Z.get(), F, C, d_W.get(), P, d_C.get(), K,
        d_P.get(), L, d_mu.get(), d_sigma.get(), {}, stream_);

    auto h_pred = download_i32(result.pred_class, C, stream_);

    int n_wrong = 0;
    for (int c = 0; c < 25; ++c)
        if (h_pred[c] != 0) ++n_wrong;
    for (int c = 25; c < 50; ++c)
        if (h_pred[c] != 1) ++n_wrong;

    std::printf("[NearCluster] n_wrong=%d / %d\n", n_wrong, C);
    EXPECT_EQ(n_wrong, 0) << "Some cells mapped to wrong cluster label";
}

// =============================================================================
// Test 3: Symphony_AmbiguousQuery_LowConfidence
// Query exactly equidistant from two clusters.
// confidence should be lower than well-separated case.
// =============================================================================
TEST_F(SymphonyFixture, Symphony_AmbiguousQuery_LowConfidence) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    // 1 feature, 2 cells, 2 clusters, 2 classes, 1 PC.
    constexpr int F = 1, C = 2, P = 1, K = 2, L = 2;

    // W_pca identity (1x1)
    std::vector<float> W_pca = {1.f};
    // Cluster centroids: cluster 0 at +2, cluster 1 at -2 in PCA space.
    // C_ref (K x P col-major): C_ref[k + p*K]
    std::vector<float> C_ref = {2.f, -2.f};  // PC 0
    // P_label: one-hot per cluster.
    std::vector<float> P_label = {
        1.f, 0.f,   // label 0
        0.f, 1.f    // label 1
    };
    std::vector<float> mu(F, 0.f), sigma(F, 1.f);

    // Cell 0: at PCA=0 (equidistant from both clusters — ambiguous).
    // Cell 1: at PCA=2 (right on cluster 0 — high confidence).
    std::vector<float> Z = {0.f, 2.f};   // 1 feature x 2 cells col-major

    auto d_Z     = upload_f32(Z,       stream_);
    auto d_W     = upload_f32(W_pca,   stream_);
    auto d_C     = upload_f32(C_ref,   stream_);
    auto d_P     = upload_f32(P_label, stream_);
    auto d_mu    = upload_f32(mu,      stream_);
    auto d_sigma = upload_f32(sigma,   stream_);
    cudaStreamSynchronize(stream_);

    auto result = singlet::gpu::anno::symphony_predict(
        d_Z.get(), F, C, d_W.get(), P, d_C.get(), K,
        d_P.get(), L, d_mu.get(), d_sigma.get(), {}, stream_);

    auto h_conf = download_f32(result.confidence, C, stream_);
    auto h_pred = download_i32(result.pred_class, C, stream_);

    std::printf("[Ambiguous] cell0_conf=%.6f (ambiguous) cell1_conf=%.6f (clear)\n",
                h_conf[0], h_conf[1]);

    // Ambiguous cell should have conf closer to 0.5 than the clear cell.
    EXPECT_LT(h_conf[0], h_conf[1]) << "Ambiguous cell should have lower confidence";
    // Clear cell should be class 0 (nearest to cluster 0).
    EXPECT_EQ(h_pred[1], 0) << "Well-separated cell near cluster 0 should predict class 0";
    // Confidence is finite.
    ASSERT_TRUE(std::isfinite(h_conf[0])) << "Ambiguous confidence is NaN/Inf";
    ASSERT_TRUE(std::isfinite(h_conf[1])) << "Clear confidence is NaN/Inf";
}

// =============================================================================
// Test 4: Symphony_Determinism_BitIdentical
// Two identical runs. pred_class identical; confidence bit-identical.
// =============================================================================
TEST_F(SymphonyFixture, Symphony_Determinism_BitIdentical) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int F = 20, C = 30, P = 5, K = 4, L = 3;

    std::mt19937 rng(77777u);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    auto make_vec = [&](size_t n) {
        std::vector<float> v(n);
        for (auto& x : v) x = dist(rng);
        return v;
    };

    auto Z       = make_vec(static_cast<size_t>(F) * C);
    auto W_pca   = make_vec(static_cast<size_t>(F) * P);
    auto C_ref   = make_vec(static_cast<size_t>(K) * P);
    auto mu      = make_vec(F);
    auto sigma_v = make_vec(F);
    // Ensure sigma > 0
    for (auto& s : sigma_v) s = 0.1f + s;

    // Build a valid P_label (rows sum to 1)
    std::vector<float> P_label(static_cast<size_t>(K) * L, 0.f);
    for (int k = 0; k < K; ++k) {
        float s = 0.f;
        for (int l = 0; l < L; ++l) {
            P_label[k + static_cast<ptrdiff_t>(l) * K] = dist(rng);
            s += P_label[k + static_cast<ptrdiff_t>(l) * K];
        }
        for (int l = 0; l < L; ++l)
            P_label[k + static_cast<ptrdiff_t>(l) * K] /= s;
    }

    auto d_Z     = upload_f32(Z,       stream_);
    auto d_W     = upload_f32(W_pca,   stream_);
    auto d_C     = upload_f32(C_ref,   stream_);
    auto d_P     = upload_f32(P_label, stream_);
    auto d_mu    = upload_f32(mu,      stream_);
    auto d_sigma = upload_f32(sigma_v, stream_);
    cudaStreamSynchronize(stream_);

    auto r1 = singlet::gpu::anno::symphony_predict(
        d_Z.get(), F, C, d_W.get(), P, d_C.get(), K,
        d_P.get(), L, d_mu.get(), d_sigma.get(), {}, stream_);
    auto r2 = singlet::gpu::anno::symphony_predict(
        d_Z.get(), F, C, d_W.get(), P, d_C.get(), K,
        d_P.get(), L, d_mu.get(), d_sigma.get(), {}, stream_);

    auto p1 = download_i32(r1.pred_class, C, stream_);
    auto p2 = download_i32(r2.pred_class, C, stream_);
    auto c1 = download_f32(r1.confidence, C, stream_);
    auto c2 = download_f32(r2.confidence, C, stream_);

    int n_pred_diff = 0, n_conf_diff = 0;
    for (int c = 0; c < C; ++c) {
        if (p1[c] != p2[c]) ++n_pred_diff;
        if (c1[c] != c2[c]) ++n_conf_diff;
    }
    std::printf("[Determinism] n_pred_diff=%d  n_conf_diff=%d\n",
                n_pred_diff, n_conf_diff);
    EXPECT_EQ(n_pred_diff, 0) << "pred_class differs between identical runs";
    EXPECT_EQ(n_conf_diff, 0) << "confidence differs between identical runs";
}

// =============================================================================
// Test 5: Symphony_DegenerateZeroSigma_Survives
// sigma_ref has zeros for some features.  With eps_sigma guard, no NaN/Inf.
// =============================================================================
TEST_F(SymphonyFixture, Symphony_DegenerateZeroSigma_Survives) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int F = 8, C = 10, P = 2, K = 3, L = 3;

    std::mt19937 rng(31415u);
    std::uniform_real_distribution<float> d(0.f, 1.f);

    std::vector<float> Z(static_cast<size_t>(F) * C);
    for (auto& x : Z) x = d(rng);
    std::vector<float> W_pca(static_cast<size_t>(F) * P);
    for (auto& x : W_pca) x = d(rng) - 0.5f;
    std::vector<float> C_ref(static_cast<size_t>(K) * P);
    for (auto& x : C_ref) x = d(rng) - 0.5f;
    std::vector<float> P_label(static_cast<size_t>(K) * L, 0.f);
    for (int k = 0; k < K; ++k) {
        float s = 0.f;
        for (int l = 0; l < L; ++l) {
            P_label[k + static_cast<ptrdiff_t>(l) * K] = d(rng);
            s += P_label[k + static_cast<ptrdiff_t>(l) * K];
        }
        for (int l = 0; l < L; ++l)
            P_label[k + static_cast<ptrdiff_t>(l) * K] /= s;
    }
    std::vector<float> mu(F, 0.f);
    // sigma has zeros for features 0, 2, 4 — guards required.
    std::vector<float> sigma(F);
    for (int g = 0; g < F; ++g)
        sigma[g] = (g % 2 == 0) ? 0.f : 0.5f;

    auto d_Z     = upload_f32(Z,       stream_);
    auto d_W     = upload_f32(W_pca,   stream_);
    auto d_C     = upload_f32(C_ref,   stream_);
    auto d_P     = upload_f32(P_label, stream_);
    auto d_mu    = upload_f32(mu,      stream_);
    auto d_sigma = upload_f32(sigma,   stream_);
    cudaStreamSynchronize(stream_);

    auto result = singlet::gpu::anno::symphony_predict(
        d_Z.get(), F, C, d_W.get(), P, d_C.get(), K,
        d_P.get(), L, d_mu.get(), d_sigma.get(), {}, stream_);

    auto h_conf = download_f32(result.confidence, C, stream_);
    auto h_pred = download_i32(result.pred_class, C, stream_);

    int n_bad = 0;
    for (int c = 0; c < C; ++c) {
        if (!std::isfinite(h_conf[c])) ++n_bad;
        if (h_pred[c] < 0 || h_pred[c] >= L) ++n_bad;
    }
    std::printf("[ZeroSigma] n_bad=%d / %d\n", n_bad, C);
    EXPECT_EQ(n_bad, 0) << "NaN/Inf or out-of-range pred with zero sigma input";
}

// =============================================================================
// main
// =============================================================================
// GTest main provided by GTest::gtest_main link target.
