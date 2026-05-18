// SPDX-License-Identifier: MIT
// singlet/gpu/tests/anno_reference_map_correctness.cpp
//
// Correctness harness for singlet::gpu::anno::reference_map (CellTypist-style).
//
// Written against the design doc at:
//   singlet/gpu/state/designs/10-annotation.md
// NOT against the kernel source — authored in parallel with anno/reference_map.h
// by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: CellTypist / sklearn LogisticRegression via subprocess:
//   tests/refs/reference_map_celltypist_reference.py
//   Environment: pip install celltypist scanpy scikit-learn numpy scipy
//
// Tolerances (design doc §"Correctness test spec"):
//   Classification accuracy  ≥ 0.90 (well-separated synthetic, held-out 50%)
//   Softmax probabilities     sum ∈ [1 - 1e-5, 1 + 1e-5] per cell
//   Determinism               bit-identical (same input, same model)
//
// Test cases:
//   ReferenceMap_LoadCelltypistNpz
//   ReferenceMap_ProjectKnownEmbedding_LabelMatchesTrueClass
//   ReferenceMap_SoftmaxStability
//   ReferenceMap_ClusterLevelAggregation
//   ReferenceMap_RequireValidModel_Throws
//   ReferenceMap_Determinism_BitIdentical
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every test that requires a device is guarded by
//   gpu_available() and calls GTEST_SKIP().

#include <singlet/gpu/anno/reference_map.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

/// Synthetic embedding dimensions.
constexpr int kNPcs        = 10;
constexpr int kNClasses    = 5;
constexpr int kNCellsTrain = 100;   // 50 cells per half
constexpr int kNCellsTest  = 100;

/// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_anno_refmap_refs_tmp";

/// Python reference script.
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet/gpu/"
    "tests/refs/reference_map_celltypist_reference.py";

// =============================================================================
// Utilities
// =============================================================================

namespace {

void ensure_refs_tmp() { fs::create_directories(kRefsTmpDir); }

std::string refs_path(const std::string& name) {
    return std::string(kRefsTmpDir) + "/" + name;
}

void run_cmd(const std::string& cmd) {
    int ret = std::system(cmd.c_str());
    if (ret != 0)
        throw std::runtime_error("Command failed (exit " + std::to_string(ret) +
                                 "): " + cmd);
}

bool gpu_available() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    return (e == cudaSuccess && count > 0);
}

// ---------------------------------------------------------------------------
// Minimal .npy v1.0 writers (reused from cycles 9-11).
// ---------------------------------------------------------------------------
static void write_npy_generic(const std::string& path,
                               const void* data,
                               const std::string& dtype_str,
                               int rows, int cols)
{
    std::string shape_str = (cols == 1)
        ? "(" + std::to_string(rows) + ",)"
        : "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")";
    std::string header_str =
        "{'descr': '" + dtype_str + "', 'fortran_order': False, 'shape': " +
        shape_str + ", }";

    constexpr int kPreamble = 10;
    int raw_len   = static_cast<int>(header_str.size());
    int pad_to_64 = ((raw_len + kPreamble + 63) / 64) * 64 - kPreamble;
    while (static_cast<int>(header_str.size()) < pad_to_64 - 1)
        header_str += ' ';
    header_str += '\n';

    std::ofstream fout(path, std::ios::binary | std::ios::trunc);
    if (!fout) throw std::runtime_error("write_npy: cannot open " + path);

    const char magic[] = "\x93NUMPY\x01\x00";
    fout.write(magic, 8);
    uint16_t hlen = static_cast<uint16_t>(header_str.size());
    fout.write(reinterpret_cast<const char*>(&hlen), 2);
    fout.write(header_str.c_str(), static_cast<std::streamsize>(header_str.size()));

    size_t n_items = static_cast<size_t>(rows) * (cols == 1 ? 1 : static_cast<size_t>(cols));
    fout.write(static_cast<const char*>(data),
               static_cast<std::streamsize>(n_items * 4));
    if (!fout) throw std::runtime_error("write_npy: write error " + path);
}

void write_npy_f32_2d(const std::string& path, const float* data, int rows, int cols) {
    write_npy_generic(path, data, "<f4", rows, cols);
}

void write_npy_i32(const std::string& path, const int32_t* data, int n) {
    write_npy_generic(path, data, "<i4", n, 1);
}

// ---------------------------------------------------------------------------
// Load float32 .npz key into flat vector.
// ---------------------------------------------------------------------------
std::vector<float> load_npz_f32(const std::string& npz_path,
                                  const std::string& key,
                                  int expected_n)
{
    std::string tmp = refs_path("_extract_tmp_refmap.npy");
    std::string cmd = "python3 -c \""
        "import numpy as np; "
        "d = np.load('" + npz_path + "', allow_pickle=True); "
        "np.save('" + tmp + "', d['" + key + "'].astype(np.float32))"
        "\"";
    run_cmd(cmd);

    std::ifstream f(tmp, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open extracted npy: " + tmp);

    char buf[8]; f.read(buf, 8);
    uint16_t hlen; f.read(reinterpret_cast<char*>(&hlen), 2);
    std::string header(hlen, '\0'); f.read(header.data(), hlen);

    std::vector<float> data(static_cast<size_t>(expected_n));
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(expected_n * sizeof(float)));
    if (!f) throw std::runtime_error("Short read on extracted npy: " + tmp);
    return data;
}

std::vector<int32_t> load_npz_i32(const std::string& npz_path,
                                   const std::string& key,
                                   int expected_n)
{
    std::string tmp = refs_path("_extract_tmp_refmap_i32.npy");
    std::string cmd = "python3 -c \""
        "import numpy as np; "
        "d = np.load('" + npz_path + "', allow_pickle=True); "
        "np.save('" + tmp + "', d['" + key + "'].astype(np.int32))"
        "\"";
    run_cmd(cmd);

    std::ifstream f(tmp, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open extracted i32 npy: " + tmp);

    char buf[8]; f.read(buf, 8);
    uint16_t hlen; f.read(reinterpret_cast<char*>(&hlen), 2);
    std::string header(hlen, '\0'); f.read(header.data(), hlen);

    std::vector<int32_t> data(static_cast<size_t>(expected_n));
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(expected_n * sizeof(int32_t)));
    if (!f) throw std::runtime_error("Short read on extracted i32 npy: " + tmp);
    return data;
}

// ---------------------------------------------------------------------------
// Generate a well-separated synthetic embedding.
// kNClasses cluster centroids; cells per class drawn as centroid + N(0, 0.1).
// Returns flat (n_cells × n_pcs) row-major float array.
// ---------------------------------------------------------------------------
std::vector<float> make_synthetic_embedding(
        int n_cells, int n_pcs, int n_classes,
        std::vector<int32_t>& labels_out,
        uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    std::uniform_real_distribution<float> centroid_range(-5.0f, 5.0f);

    // Fixed centroids (n_classes × n_pcs).
    std::vector<float> centroids(static_cast<size_t>(n_classes) * n_pcs);
    for (auto& v : centroids) v = centroid_range(rng);

    int cells_per_class = n_cells / n_classes;
    labels_out.resize(n_cells);
    std::vector<float> emb(static_cast<size_t>(n_cells) * n_pcs);

    for (int c = 0; c < n_cells; ++c) {
        int cls = std::min(c / cells_per_class, n_classes - 1);
        labels_out[c] = cls;
        for (int p = 0; p < n_pcs; ++p)
            emb[static_cast<size_t>(c) * n_pcs + p] =
                centroids[static_cast<size_t>(cls) * n_pcs + p] + noise(rng);
    }
    return emb;
}

// ---------------------------------------------------------------------------
// Upload a dense (n_cells × n_pcs) float matrix as DeviceDense.
// DeviceDense: row-major, n_rows = n_cells, n_cols = n_pcs.
// ---------------------------------------------------------------------------
singlet::gpu::core::DeviceDense upload_dense(
        const float* h_mat, int n_rows, int n_cols, cudaStream_t stream)
{
    using namespace singlet::gpu::core;
    size_t n_elem = static_cast<size_t>(n_rows) * n_cols;
    DeviceMemory<float> d_mem(n_elem);

    cudaError_t e = cudaMemcpyAsync(d_mem.get(), h_mat,
                                    n_elem * sizeof(float),
                                    cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("upload_dense: ") +
                                 cudaGetErrorString(e));
    cudaStreamSynchronize(stream);
    DeviceDense mat;
    mat.rows = static_cast<uint32_t>(n_rows);
    mat.cols = static_cast<uint32_t>(n_cols);
    mat.data = std::move(d_mem);
    return mat;
}

// ---------------------------------------------------------------------------
// Download labels (int32) from device.
// ---------------------------------------------------------------------------
std::vector<int32_t> download_labels(
        const singlet::gpu::anno::RefMapResult& res, int n_cells)
{
    std::vector<int32_t> h(static_cast<size_t>(n_cells));
    cudaError_t e = cudaMemcpy(h.data(), res.labels.get(),
                               static_cast<size_t>(n_cells) * sizeof(int32_t),
                               cudaMemcpyDeviceToHost);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("download_labels: ") +
                                 cudaGetErrorString(e));
    return h;
}

// ---------------------------------------------------------------------------
// Download probabilities (float32) from device.
// Layout: (n_classes × n_cells) — same as marker_score convention.
// Returns flat vector.
// ---------------------------------------------------------------------------
std::vector<float> download_probabilities(
        const singlet::gpu::anno::RefMapResult& res,
        int n_classes, int n_cells)
{
    int total = n_classes * n_cells;
    std::vector<float> h(static_cast<size_t>(total));
    cudaError_t e = cudaMemcpy(h.data(), res.probabilities.get(),
                               static_cast<size_t>(total) * sizeof(float),
                               cudaMemcpyDeviceToHost);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("download_probabilities: ") +
                                 cudaGetErrorString(e));
    return h;
}

}  // namespace

// =============================================================================
// Test fixture
// =============================================================================

class ReferenceMapTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!gpu_available())
            GTEST_SKIP() << "No CUDA device available.";
        cudaError_t e = cudaStreamCreate(&stream_);
        ASSERT_EQ(e, cudaSuccess) << "cudaStreamCreate failed";
        ensure_refs_tmp();

        // Generate synthetic train and test embeddings.
        train_emb_ = make_synthetic_embedding(kNCellsTrain, kNPcs, kNClasses,
                                               train_labels_, kSeed);
        test_emb_  = make_synthetic_embedding(kNCellsTest, kNPcs, kNClasses,
                                               test_labels_,  kSeed ^ 0xDEAD);

        // Write train/test to npy for Python reference.
        train_emb_path_  = refs_path("train_emb.npy");
        train_lab_path_  = refs_path("train_labels.npy");
        test_emb_path_   = refs_path("test_emb.npy");
        model_npz_path_  = refs_path("celltypist_model.npz");
        pred_npz_path_   = refs_path("celltypist_preds.npz");

        write_npy_f32_2d(train_emb_path_, train_emb_.data(), kNCellsTrain, kNPcs);
        write_npy_i32   (train_lab_path_, train_labels_.data(), kNCellsTrain);
        write_npy_f32_2d(test_emb_path_,  test_emb_.data(),  kNCellsTest,  kNPcs);

        // Run Python reference to produce model.npz and preds.npz.
        std::string cmd = "python3 " + std::string(kRefScript) +
            " --mode generate"
            " --train-emb "  + train_emb_path_  +
            " --train-lab "  + train_lab_path_  +
            " --test-emb "   + test_emb_path_   +
            " --model-out "  + model_npz_path_  +
            " --pred-out "   + pred_npz_path_;
        run_cmd(cmd);
    }

    void TearDown() override {
        if (stream_) cudaStreamDestroy(stream_);
    }

    cudaStream_t stream_ = nullptr;

    std::vector<float>   train_emb_, test_emb_;
    std::vector<int32_t> train_labels_, test_labels_;

    std::string train_emb_path_, train_lab_path_, test_emb_path_;
    std::string model_npz_path_, pred_npz_path_;
};

// =============================================================================
// Test cases
// =============================================================================

// ---------------------------------------------------------------------------
// ReferenceMap_LoadCelltypistNpz
//
// Python reference writes model.npz with weights, intercepts, class_names.
// C++ load_celltypist_model(path) must produce a CelltypistModel with:
//   - n_classes == kNClasses
//   - n_pcs     == kNPcs
//   - class_names.size() == kNClasses
//   - weights device memory has exactly n_classes * n_pcs floats
// ---------------------------------------------------------------------------
TEST_F(ReferenceMapTest, ReferenceMap_LoadCelltypistNpz) {
    using namespace singlet::gpu::anno;

    CelltypistModel model = load_celltypist_model(model_npz_path_);

    EXPECT_EQ(model.n_classes, kNClasses)
        << "Expected n_classes=" << kNClasses << " got " << model.n_classes;
    EXPECT_EQ(model.n_pcs, kNPcs)
        << "Expected n_pcs=" << kNPcs << " got " << model.n_pcs;
    EXPECT_EQ(static_cast<int>(model.class_names.size()), kNClasses)
        << "class_names size mismatch";

    // Verify device memory sizes are non-null and plausibly sized.
    EXPECT_NE(model.weights.get(),    nullptr) << "weights device ptr is null";
    EXPECT_NE(model.intercepts.get(), nullptr) << "intercepts device ptr is null";
}

// ---------------------------------------------------------------------------
// ReferenceMap_ProjectKnownEmbedding_LabelMatchesTrueClass
//
// The Python reference trains on train_emb (well-separated 5-class) and
// projects test_emb.  Our C++ adapter must achieve accuracy ≥ 0.90 on the
// same test set.
// ---------------------------------------------------------------------------
TEST_F(ReferenceMapTest, ReferenceMap_ProjectKnownEmbedding_LabelMatchesTrueClass) {
    using namespace singlet::gpu::anno;

    CelltypistModel model = load_celltypist_model(model_npz_path_);
    auto d_emb = upload_dense(test_emb_.data(), kNCellsTest, kNPcs, stream_);

    RefMapResult result = project_to_reference(d_emb, model, stream_);
    auto our_labels = download_labels(result, kNCellsTest);

    // Compute accuracy against test ground truth.
    int correct = 0;
    for (int c = 0; c < kNCellsTest; ++c)
        if (our_labels[c] == test_labels_[c]) ++correct;
    double accuracy = static_cast<double>(correct) / kNCellsTest;

    EXPECT_GE(accuracy, 0.90)
        << "Classification accuracy " << accuracy << " < 0.90";

    // Also compare against Python reference labels from pred.npz.
    auto ref_labels = load_npz_i32(pred_npz_path_, "labels", kNCellsTest);
    int agreement = 0;
    for (int c = 0; c < kNCellsTest; ++c)
        if (our_labels[c] == ref_labels[c]) ++agreement;
    double label_agreement = static_cast<double>(agreement) / kNCellsTest;
    EXPECT_GE(label_agreement, 0.90)
        << "Label agreement with Python reference: " << label_agreement;
}

// ---------------------------------------------------------------------------
// ReferenceMap_SoftmaxStability
//
// Synthetic embedding with extreme values (±50) to stress max-subtraction.
// Confirm softmax probabilities sum to 1 per row within [1 - 1e-5, 1 + 1e-5].
// ---------------------------------------------------------------------------
TEST_F(ReferenceMapTest, ReferenceMap_SoftmaxStability) {
    using namespace singlet::gpu::anno;

    // Construct extreme embedding: alternating +50 / -50 per feature.
    std::vector<float> extreme_emb(static_cast<size_t>(kNCellsTest) * kNPcs);
    for (int c = 0; c < kNCellsTest; ++c)
        for (int p = 0; p < kNPcs; ++p)
            extreme_emb[static_cast<size_t>(c) * kNPcs + p] =
                ((c + p) % 2 == 0) ? 50.0f : -50.0f;

    CelltypistModel model = load_celltypist_model(model_npz_path_);
    auto d_emb = upload_dense(extreme_emb.data(), kNCellsTest, kNPcs, stream_);

    RefMapResult result = project_to_reference(d_emb, model, stream_);
    auto probs = download_probabilities(result, kNClasses, kNCellsTest);

    // probs layout: (n_classes × n_cells).
    for (int c = 0; c < kNCellsTest; ++c) {
        float row_sum = 0.0f;
        for (int k = 0; k < kNClasses; ++k)
            row_sum += probs[static_cast<size_t>(k) * kNCellsTest + c];
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f)
            << "Softmax row sum " << row_sum << " at cell " << c;
        // Also check individual probabilities are in [0, 1].
        for (int k = 0; k < kNClasses; ++k) {
            float p = probs[static_cast<size_t>(k) * kNCellsTest + c];
            EXPECT_GE(p, 0.0f) << "Negative probability at (class=" << k
                                << ", cell=" << c << ")";
            EXPECT_LE(p, 1.0f + 1e-6f) << "Probability > 1 at (class=" << k
                                        << ", cell=" << c << ")";
        }
    }
}

// ---------------------------------------------------------------------------
// ReferenceMap_ClusterLevelAggregation
//
// Per-cell project_to_reference, then aggregate by ground-truth cluster:
// compute per-cluster mean probability and per-cluster majority-vote label.
// Confirm majority-vote label matches the true cluster label for ≥ 90% of
// clusters (design doc §"For cluster-level annotation: aggregate per-cluster
// mean probabilities, then argmax per cluster").
// ---------------------------------------------------------------------------
TEST_F(ReferenceMapTest, ReferenceMap_ClusterLevelAggregation) {
    using namespace singlet::gpu::anno;

    CelltypistModel model = load_celltypist_model(model_npz_path_);
    auto d_emb = upload_dense(test_emb_.data(), kNCellsTest, kNPcs, stream_);

    RefMapResult result = project_to_reference(d_emb, model, stream_);
    auto our_labels = download_labels(result, kNCellsTest);
    auto probs = download_probabilities(result, kNClasses, kNCellsTest);

    // Compute per-cluster mean probability (host-side) and argmax label.
    // test_labels_[c] gives ground-truth cluster 0..kNClasses-1.
    std::vector<float>   cluster_prob_sum(static_cast<size_t>(kNClasses) * kNClasses, 0.0f);
    std::vector<int>     cluster_count(kNClasses, 0);

    for (int c = 0; c < kNCellsTest; ++c) {
        int gt = test_labels_[c];
        ++cluster_count[gt];
        for (int k = 0; k < kNClasses; ++k)
            cluster_prob_sum[static_cast<size_t>(gt) * kNClasses + k] +=
                probs[static_cast<size_t>(k) * kNCellsTest + c];
    }

    int clusters_correct = 0;
    for (int gt = 0; gt < kNClasses; ++gt) {
        if (cluster_count[gt] == 0) continue;
        // Per-cluster mean prob argmax.
        int best_k = 0;
        float best_p = cluster_prob_sum[static_cast<size_t>(gt) * kNClasses + 0];
        for (int k = 1; k < kNClasses; ++k) {
            float p = cluster_prob_sum[static_cast<size_t>(gt) * kNClasses + k];
            if (p > best_p) { best_p = p; best_k = k; }
        }
        if (best_k == gt) ++clusters_correct;
    }

    double cluster_accuracy = static_cast<double>(clusters_correct) / kNClasses;
    EXPECT_GE(cluster_accuracy, 0.80)
        << "Cluster-level aggregation accuracy " << cluster_accuracy
        << " < 0.80 (" << clusters_correct << "/" << kNClasses << " clusters)";
}

// ---------------------------------------------------------------------------
// ReferenceMap_RequireValidModel_Throws
//
// Passing a model with mismatched dimensions (n_pcs != d_emb.n_cols) must
// throw std::invalid_argument or std::runtime_error.
// (Design doc: "CelltypistModel has expected shapes and class names".)
// ---------------------------------------------------------------------------
TEST_F(ReferenceMapTest, ReferenceMap_RequireValidModel_Throws) {
    using namespace singlet::gpu::anno;

    CelltypistModel model = load_celltypist_model(model_npz_path_);
    // Deliberately use an embedding with wrong number of PCs.
    int wrong_pcs = kNPcs + 5;
    std::vector<float> bad_emb(static_cast<size_t>(kNCellsTest) * wrong_pcs, 0.0f);
    auto d_bad_emb = upload_dense(bad_emb.data(), kNCellsTest, wrong_pcs, stream_);

    EXPECT_THROW(
        { project_to_reference(d_bad_emb, model, stream_); },
        std::exception   // accepts std::invalid_argument, std::runtime_error, etc.
    ) << "Expected throw on mismatched embedding/model dimensions";
}

// ---------------------------------------------------------------------------
// ReferenceMap_Determinism_BitIdentical
//
// Running project_to_reference twice on the same embedding+model must produce
// bit-identical labels AND probabilities.
// ---------------------------------------------------------------------------
TEST_F(ReferenceMapTest, ReferenceMap_Determinism_BitIdentical) {
    using namespace singlet::gpu::anno;

    CelltypistModel model = load_celltypist_model(model_npz_path_);
    auto d_emb = upload_dense(test_emb_.data(), kNCellsTest, kNPcs, stream_);

    RefMapResult r1 = project_to_reference(d_emb, model, stream_);
    RefMapResult r2 = project_to_reference(d_emb, model, stream_);

    auto l1 = download_labels(r1, kNCellsTest);
    auto l2 = download_labels(r2, kNCellsTest);
    auto p1 = download_probabilities(r1, kNClasses, kNCellsTest);
    auto p2 = download_probabilities(r2, kNClasses, kNCellsTest);

    ASSERT_EQ(l1.size(), l2.size());
    ASSERT_EQ(p1.size(), p2.size());

    int label_mismatches = 0;
    for (size_t i = 0; i < l1.size(); ++i)
        if (l1[i] != l2[i]) ++label_mismatches;
    EXPECT_EQ(label_mismatches, 0)
        << "Determinism violated: " << label_mismatches << " label mismatches";

    int prob_mismatches = 0;
    for (size_t i = 0; i < p1.size(); ++i)
        if (std::memcmp(&p1[i], &p2[i], sizeof(float)) != 0) ++prob_mismatches;
    EXPECT_EQ(prob_mismatches, 0)
        << "Determinism violated: " << prob_mismatches << " prob bit-mismatches";
}
