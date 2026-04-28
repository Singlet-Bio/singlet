// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/embed_umap_correctness.cpp
//
// Correctness harness for singlet_gpu::embed::umap.
//
// Written against the design doc at:
//   singlet-gpu/state/designs/08-umap.md
// NOT against the kernel source — this file was authored in parallel with
// umap.h by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: umap-learn Python via subprocess (tests/refs/umap_reference.py).
//   Environment: pip install umap-learn numpy scipy
//
// Tolerances (from design doc §"Correctness test spec"):
//   kNN-preservation@15 (trustworthiness) ≥ 0.85 (tiny synthetic)
//   Trustworthiness@15 ≥ 0.95 (vs umap-learn, both must satisfy independently)
//   Determinism: bit-identical output across two runs with same seed + Random init
//   Shape correctness for n_components variants
//   n_components > 50: cuML error or adapter throws
//   Empty KnnResult: throws std::invalid_argument / std::runtime_error
//
// Test cases:
//   Umap_TinySynthetic_TrustworthinessHigh
//   Umap_VsUmapLearn_BothAchieveTrustworthiness
//   Umap_Determinism_RandomInit_BitIdentical
//   Umap_3D_OutputShape
//   Umap_NComponents1_OutputShape
//   Umap_NComponentsTooLarge_Errors
//   Umap_RequireKnnGraph_Throws
//
// Build: cmake --build build -j (requires FACTORNET_HAS_GPU, CUDA::cudart,
//   cuml). On a CPU-only node where cuML is absent, all GPU tests are skipped
//   via GTEST_SKIP() inside a gpu_available() guard.

// ---- embed + graph headers (written by gpu-kernel-dev) ----------------------
#include <singlet-gpu/embed/umap.h>
#include <singlet-gpu/graph/knn.h>

// ---- test helpers ------------------------------------------------------------
#include "refs/dump_csc.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEE;

/// Tiny synthetic embedding parameters (from design doc).
constexpr int   kTinyN    = 200;   // number of points
constexpr int   kTinyD    = 10;    // input dimensionality
constexpr int   kTinyK    = 15;    // kNN neighbors

/// Trustworthiness thresholds (from design doc).
constexpr double kTrustworthinessThresholdTiny    = 0.85;
constexpr double kTrustworthinessThresholdVsLearn = 0.95;

/// Small graph for edge-case shape tests.
constexpr int kSmallN = 20;
constexpr int kSmallD = 5;
constexpr int kSmallK = 5;

/// Temp directory for Python reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_umap_refs_tmp";

// =============================================================================
// GPU availability guard
// =============================================================================

namespace {

bool gpu_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

// =============================================================================
// Directory helpers
// =============================================================================

void ensure_refs_tmp() {
    fs::create_directories(kRefsTmpDir);
}

std::string refs_path(const std::string& name) {
    return std::string(kRefsTmpDir) + "/" + name;
}

// ---------------------------------------------------------------------------
// run_cmd — execute a shell command; throw on non-zero exit.
// ---------------------------------------------------------------------------
void run_cmd(const std::string& cmd) {
    int ret = std::system(cmd.c_str());
    if (ret != 0)
        throw std::runtime_error("Command failed (exit " + std::to_string(ret) +
                                 "): " + cmd);
}

// =============================================================================
// Minimal NPY v1.0 writer (1-D or 2-D arrays)
// Matches the pattern used in graph_leiden_correctness.cpp.
// =============================================================================

static void write_npy_generic(const std::string& path,
                               const void* data,
                               const std::string& dtype_str,
                               int rows, int cols) {
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

    int item_size = 4;  // float32 and int32 are both 4 bytes
    size_t n_elems = static_cast<size_t>(rows) * (cols == 1 ? 1 : static_cast<size_t>(cols));
    fout.write(static_cast<const char*>(data),
               static_cast<std::streamsize>(n_elems * item_size));
    if (!fout) throw std::runtime_error("write_npy: write error " + path);
}

void write_npy_f32(const std::string& path, const float* data, int rows, int cols) {
    write_npy_generic(path, data, "<f4", rows, cols);
}

void write_npy_i32(const std::string& path, const int32_t* data, int rows, int cols) {
    write_npy_generic(path, data, "<i4", rows, cols);
}

// ---------------------------------------------------------------------------
// load_npy_f32 — load a 2-D float32 .npy into a flat vector<float>.
// Returns the data in row-major order; fills out_rows / out_cols.
// ---------------------------------------------------------------------------
std::vector<float> load_npy_f32(const std::string& path, int* out_rows, int* out_cols) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("load_npy_f32: cannot open " + path);

    // Read 10-byte preamble.
    char magic[6]; fin.read(magic, 6);
    char major, minor; fin.read(&major, 1); fin.read(&minor, 1);
    uint16_t hlen; fin.read(reinterpret_cast<char*>(&hlen), 2);
    std::string hdr(hlen, ' ');
    fin.read(hdr.data(), hlen);

    // Parse shape from header string: look for 'shape': (R, C) or (R,).
    *out_rows = 0; *out_cols = 1;
    {
        auto pos = hdr.find("'shape'");
        if (pos == std::string::npos) pos = hdr.find("shape");
        if (pos != std::string::npos) {
            auto lp = hdr.find('(', pos);
            auto rp = hdr.find(')', lp);
            if (lp != std::string::npos && rp != std::string::npos) {
                std::string shape_str = hdr.substr(lp + 1, rp - lp - 1);
                // Remove whitespace.
                shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '),
                                shape_str.end());
                auto comma = shape_str.find(',');
                if (comma == std::string::npos || comma == shape_str.size() - 1) {
                    // 1-D or trailing comma → 1-D
                    *out_rows = std::stoi(shape_str.substr(0, comma));
                    *out_cols = 1;
                } else {
                    *out_rows = std::stoi(shape_str.substr(0, comma));
                    *out_cols = std::stoi(shape_str.substr(comma + 1));
                }
            }
        }
    }

    size_t n_elems = static_cast<size_t>(*out_rows) * static_cast<size_t>(*out_cols);
    std::vector<float> out(n_elems);
    fin.read(reinterpret_cast<char*>(out.data()),
             static_cast<std::streamsize>(n_elems * sizeof(float)));
    if (!fin) throw std::runtime_error("load_npy_f32: short read from " + path);
    return out;
}

// =============================================================================
// Synthetic embedding generator
//
// Generates an N × D float32 array with values drawn from N(0,1) at the
// given seed.  Used as the "input embedding" for kNN and trustworthiness.
// =============================================================================

std::vector<float> make_random_embedding(int n, int d, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> X(static_cast<size_t>(n) * d);
    for (auto& v : X) v = dist(rng);
    return X;
}

// =============================================================================
// Brute-force kNN on host (for trustworthiness computation)
//
// Returns neighbors[n * k] — each row is the k nearest neighbor indices in
// the input embedding, sorted ascending by Euclidean distance.
// Self is excluded.
// =============================================================================

std::vector<int> brute_knn_host(const std::vector<float>& X, int n, int d, int k) {
    std::vector<int> neighbors(static_cast<size_t>(n) * k, -1);

    for (int i = 0; i < n; ++i) {
        // Compute distances to all other points.
        std::vector<std::pair<float, int>> dists;
        dists.reserve(static_cast<size_t>(n) - 1);
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            float sq = 0.0f;
            for (int dd = 0; dd < d; ++dd) {
                float diff = X[static_cast<size_t>(i) * d + dd] -
                             X[static_cast<size_t>(j) * d + dd];
                sq += diff * diff;
            }
            dists.push_back({sq, j});
        }
        std::partial_sort(dists.begin(),
                          dists.begin() + k,
                          dists.end());
        for (int ki = 0; ki < k; ++ki)
            neighbors[static_cast<size_t>(i) * k + ki] = dists[ki].second;
    }
    return neighbors;
}

// =============================================================================
// Trustworthiness@k
//
// For each point i, let U_i = set of k-nearest neighbors in the OUTPUT space
// that are NOT among the k-nearest neighbors in the INPUT space.
// Trustworthiness@k = 1 - (2 / (n*k*(2n - 3k - 1)))
//                     * sum_i sum_{j in U_i} (r(i,j) - k)
//
// where r(i,j) is the rank of point j in the INPUT k-NN list of point i
// (1-indexed, among all n-1 non-self neighbors).
//
// This is the standard sklearn trustworthiness formula.
//
// Parameters:
//   X_input   : n × d_in  (original high-dim embedding)
//   X_output  : n × d_out (2-D UMAP output)
//   n, d_in, d_out, k
// =============================================================================

double trustworthiness(
        const std::vector<float>& X_input,  int d_in,
        const std::vector<float>& X_output, int d_out,
        int n, int k) {
    // Step 1: compute full rank matrix in INPUT space.
    // rank_in[i][j] = rank of point j among non-self neighbors of i in input space.
    // We only need rank for points that appear in the top-k output neighbors.
    std::vector<std::vector<std::pair<float, int>>> input_sorted(n);
    for (int i = 0; i < n; ++i) {
        input_sorted[i].reserve(static_cast<size_t>(n) - 1);
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            float sq = 0.0f;
            for (int dd = 0; dd < d_in; ++dd) {
                float diff = X_input[static_cast<size_t>(i) * d_in + dd] -
                             X_input[static_cast<size_t>(j) * d_in + dd];
                sq += diff * diff;
            }
            input_sorted[i].push_back({sq, j});
        }
        std::sort(input_sorted[i].begin(), input_sorted[i].end());
    }
    // Build rank lookup: rank_in[i * n + j] = 1-indexed rank of j in input neighbors of i.
    // We lazily compute this only for j values that appear as output neighbors.

    // Step 2: compute k-NN in OUTPUT space.
    std::vector<std::vector<std::pair<float, int>>> output_sorted(n);
    for (int i = 0; i < n; ++i) {
        output_sorted[i].reserve(static_cast<size_t>(n) - 1);
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            float sq = 0.0f;
            for (int dd = 0; dd < d_out; ++dd) {
                float diff = X_output[static_cast<size_t>(i) * d_out + dd] -
                             X_output[static_cast<size_t>(j) * d_out + dd];
                sq += diff * diff;
            }
            output_sorted[i].push_back({sq, j});
        }
        std::sort(output_sorted[i].begin(), output_sorted[i].end());
    }

    // Step 3: accumulate the trustworthiness sum.
    double sum_penalty = 0.0;
    for (int i = 0; i < n; ++i) {
        // Build set of top-k input neighbors.
        std::set<int> top_k_input;
        for (int ki = 0; ki < k && ki < static_cast<int>(input_sorted[i].size()); ++ki)
            top_k_input.insert(input_sorted[i][ki].second);

        // For each point in the top-k output neighbors that is NOT in top-k input:
        for (int ki = 0; ki < k && ki < static_cast<int>(output_sorted[i].size()); ++ki) {
            int j = output_sorted[i][ki].second;
            if (top_k_input.count(j) == 0) {
                // Find rank of j in input space (1-indexed).
                int rank_j = -1;
                for (int ri = 0; ri < static_cast<int>(input_sorted[i].size()); ++ri) {
                    if (input_sorted[i][ri].second == j) {
                        rank_j = ri + 1;  // 1-indexed
                        break;
                    }
                }
                if (rank_j > 0)
                    sum_penalty += static_cast<double>(rank_j - k);
            }
        }
    }

    double normalizer = 2.0 / (static_cast<double>(n) * k *
                               (2.0 * n - 3.0 * k - 1.0));
    return 1.0 - normalizer * sum_penalty;
}

// =============================================================================
// Copy UmapResult embedding from device to host
// =============================================================================

std::vector<float> copy_embedding_to_host(
        const singlet_gpu::embed::UmapResult& result) {
    int n    = result.n;
    int ncomp = result.n_components;
    std::vector<float> h_emb(static_cast<size_t>(n) * ncomp);
    cudaError_t err = cudaMemcpy(
        h_emb.data(),
        result.embedding.get(),
        static_cast<size_t>(n) * ncomp * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("copy_embedding_to_host: cudaMemcpy failed: ") +
            cudaGetErrorString(err));
    return h_emb;
}

// =============================================================================
// Build a KnnResult from host arrays and upload to device
//
// Wraps the host kNN arrays into a singlet_gpu::graph::KnnResult by uploading
// neighbors and distances to device memory.
// The KnnResult holds n * k entries in flat row-major order.
// =============================================================================

singlet_gpu::graph::KnnResult make_knn_result_on_device(
        const std::vector<int32_t>& h_neighbors,
        const std::vector<float>&   h_distances,
        int n, int k) {
    singlet_gpu::graph::KnnResult knn;
    knn.n          = n;
    knn.k          = k;

    // Allocate device memory. KnnResult::neighbors is DeviceMemory<int>.
    size_t n_elems = static_cast<size_t>(n) * k;
    knn.neighbors  = singlet_gpu::core::DeviceMemory<int>(n_elems);
    knn.distances  = singlet_gpu::core::DeviceMemory<float>(n_elems);

    cudaError_t err;
    err = cudaMemcpy(knn.neighbors.get(), h_neighbors.data(),
                     n_elems * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("make_knn_result_on_device: neighbors upload failed");
    err = cudaMemcpy(knn.distances.get(), h_distances.data(),
                     n_elems * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("make_knn_result_on_device: distances upload failed");
    return knn;
}

}  // anonymous namespace

// =============================================================================
// Test fixture
// =============================================================================

class UmapCorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!gpu_available())
            GTEST_SKIP() << "No CUDA device available — skipping UMAP GPU tests.";
        ensure_refs_tmp();
    }
};

// =============================================================================
// TEST 1: Umap_TinySynthetic_TrustworthinessHigh
//
// Generate 200 × 10 random fp32 input.
// Run compute_knn (Exact, k=15) on device.
// Run umap (n_components=2, init=Random, seed=42).
// Compute trustworthiness@15 of the 2D output vs the input embedding.
// Assert ≥ 0.85.
// =============================================================================
TEST_F(UmapCorrectnessTest, Umap_TinySynthetic_TrustworthinessHigh) {
    const int n = kTinyN, d = kTinyD, k = kTinyK;

    // Generate synthetic input.
    std::vector<float> h_X = make_random_embedding(n, d, kSeed);

    // Upload to device and run kNN.
    singlet_gpu::core::DeviceMemory<float> d_X(static_cast<size_t>(n) * d);
    cudaMemcpy(d_X.get(), h_X.data(), static_cast<size_t>(n) * d * sizeof(float),
               cudaMemcpyHostToDevice);

    singlet_gpu::graph::KnnConfig knn_cfg;
    knn_cfg.k      = k;
    knn_cfg.metric = singlet_gpu::graph::DistanceMetric::L2;
    knn_cfg.backend = singlet_gpu::graph::KnnBackend::Exact;
    // compute_knn takes DeviceDense + stream; wrap the raw pointer non-owning.
    singlet_gpu::core::DeviceDense X_dev;
    X_dev.rows = n; X_dev.cols = d;
    X_dev.data = singlet_gpu::core::DeviceMemory<float>::wrap(d_X.get(),
                                                               static_cast<size_t>(n) * d);
    auto knn = singlet_gpu::graph::compute_knn(X_dev, knn_cfg, nullptr);

    // Run UMAP.
    singlet_gpu::embed::UmapConfig umap_cfg;
    umap_cfg.n_components = 2;
    umap_cfg.init         = singlet_gpu::embed::UmapInit::Random;
    umap_cfg.seed         = 42;
    auto result = singlet_gpu::embed::umap(knn, umap_cfg, nullptr);

    ASSERT_EQ(result.n, n);
    ASSERT_EQ(result.n_components, 2);

    // Copy embedding to host.
    auto h_emb = copy_embedding_to_host(result);

    // Compute trustworthiness@k.
    double tw = trustworthiness(h_X, d, h_emb, 2, n, k);
    printf("[test] Umap_TinySynthetic_TrustworthinessHigh: "
           "trustworthiness@%d = %.4f (threshold=%.2f)\n",
           k, tw, kTrustworthinessThresholdTiny);

    EXPECT_GE(tw, kTrustworthinessThresholdTiny)
        << "Trustworthiness@" << k << " = " << tw
        << " < threshold " << kTrustworthinessThresholdTiny;
}

// =============================================================================
// TEST 2: Umap_VsUmapLearn_BothAchieveTrustworthiness
//
// Same 200 × 10 input and kNN as Test 1.
// Run our UMAP and umap-learn (subprocess) on the same precomputed kNN.
// Confirm both achieve trustworthiness@15 ≥ 0.95.
// =============================================================================
TEST_F(UmapCorrectnessTest, Umap_VsUmapLearn_BothAchieveTrustworthiness) {
    const int n = kTinyN, d = kTinyD, k = kTinyK;

    // Verify umap-learn is available before running GPU work.
    {
        int check = std::system("python3 -c 'import umap' 2>/dev/null");
        if (check != 0)
            GTEST_SKIP() << "umap-learn not installed — skipping umap-learn comparison. "
                         << "Install with: pip install umap-learn numpy scipy";
    }

    std::vector<float> h_X = make_random_embedding(n, d, kSeed);

    // Upload to device and run kNN.
    singlet_gpu::core::DeviceMemory<float> d_X(static_cast<size_t>(n) * d);
    cudaMemcpy(d_X.get(), h_X.data(), static_cast<size_t>(n) * d * sizeof(float),
               cudaMemcpyHostToDevice);

    singlet_gpu::graph::KnnConfig knn_cfg;
    knn_cfg.k       = k;
    knn_cfg.metric  = singlet_gpu::graph::DistanceMetric::L2;
    knn_cfg.backend = singlet_gpu::graph::KnnBackend::Exact;
    // compute_knn takes DeviceDense + stream; wrap the raw pointer non-owning.
    singlet_gpu::core::DeviceDense X_dev;
    X_dev.rows = n; X_dev.cols = d;
    X_dev.data = singlet_gpu::core::DeviceMemory<float>::wrap(d_X.get(),
                                                               static_cast<size_t>(n) * d);
    auto knn = singlet_gpu::graph::compute_knn(X_dev, knn_cfg, nullptr);

    // Run our UMAP.
    singlet_gpu::embed::UmapConfig umap_cfg;
    umap_cfg.n_components = 2;
    umap_cfg.init         = singlet_gpu::embed::UmapInit::Random;
    umap_cfg.seed         = 42;
    auto result = singlet_gpu::embed::umap(knn, umap_cfg, nullptr);
    auto h_our_emb = copy_embedding_to_host(result);

    double tw_ours = trustworthiness(h_X, d, h_our_emb, 2, n, k);
    printf("[test] Our UMAP trustworthiness@%d = %.4f\n", k, tw_ours);

    // Fetch kNN neighbors/distances from device for Python reference.
    std::vector<int32_t> h_nb(static_cast<size_t>(n) * k);
    std::vector<float>   h_dist(static_cast<size_t>(n) * k);
    cudaMemcpy(h_nb.data(), knn.neighbors.get(),
               static_cast<size_t>(n) * k * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dist.data(), knn.distances.get(),
               static_cast<size_t>(n) * k * sizeof(float), cudaMemcpyDeviceToHost);

    // Write kNN to .npy files for the Python reference.
    std::string indices_path   = refs_path("umap_indices.npy");
    std::string distances_path = refs_path("umap_distances.npy");
    std::string ref_emb_path   = refs_path("umap_ref_embedding.npy");

    write_npy_i32(indices_path,   h_nb.data(),   n, k);
    write_npy_f32(distances_path, h_dist.data(), n, k);

    // Run umap-learn reference subprocess.
    std::string ref_script =
        std::string(SINGLET_GPU_TEST_DIR) + "/refs/umap_reference.py";
    run_cmd("python3 " + ref_script +
            " --indices "      + indices_path   +
            " --distances "    + distances_path +
            " --output "       + ref_emb_path   +
            " --n-components 2"
            " --n-neighbors "  + std::to_string(k) +
            " --seed 42");

    // Load reference embedding.
    int ref_rows = 0, ref_cols = 0;
    auto h_ref_emb = load_npy_f32(ref_emb_path, &ref_rows, &ref_cols);
    ASSERT_EQ(ref_rows, n);
    ASSERT_EQ(ref_cols, 2);

    double tw_ref = trustworthiness(h_X, d, h_ref_emb, 2, n, k);
    printf("[test] umap-learn trustworthiness@%d = %.4f (threshold=%.2f)\n",
           k, tw_ref, kTrustworthinessThresholdVsLearn);

    EXPECT_GE(tw_ours, kTrustworthinessThresholdVsLearn)
        << "Our UMAP trustworthiness@" << k << " = " << tw_ours
        << " < threshold " << kTrustworthinessThresholdVsLearn;

    EXPECT_GE(tw_ref, kTrustworthinessThresholdVsLearn)
        << "umap-learn trustworthiness@" << k << " = " << tw_ref
        << " < threshold " << kTrustworthinessThresholdVsLearn;
}

// =============================================================================
// TEST 3: Umap_Determinism_RandomInit_BitIdentical
//
// Run our UMAP twice with init=Random + same seed on the same kNN.
// Assert bit-identical float output in the embedding array.
// =============================================================================
TEST_F(UmapCorrectnessTest, Umap_Determinism_RandomInit_BitIdentical) {
    const int n = kTinyN, d = kTinyD, k = kTinyK;

    std::vector<float> h_X = make_random_embedding(n, d, kSeed);

    singlet_gpu::core::DeviceMemory<float> d_X(static_cast<size_t>(n) * d);
    cudaMemcpy(d_X.get(), h_X.data(), static_cast<size_t>(n) * d * sizeof(float),
               cudaMemcpyHostToDevice);

    singlet_gpu::graph::KnnConfig knn_cfg;
    knn_cfg.k       = k;
    knn_cfg.metric  = singlet_gpu::graph::DistanceMetric::L2;
    knn_cfg.backend = singlet_gpu::graph::KnnBackend::Exact;
    // compute_knn takes DeviceDense + stream; wrap the raw pointer non-owning.
    singlet_gpu::core::DeviceDense X_dev;
    X_dev.rows = n; X_dev.cols = d;
    X_dev.data = singlet_gpu::core::DeviceMemory<float>::wrap(d_X.get(),
                                                               static_cast<size_t>(n) * d);
    auto knn = singlet_gpu::graph::compute_knn(X_dev, knn_cfg, nullptr);

    singlet_gpu::embed::UmapConfig umap_cfg;
    umap_cfg.n_components = 2;
    umap_cfg.init         = singlet_gpu::embed::UmapInit::Random;
    umap_cfg.seed         = 42;

    // First run.
    auto result1 = singlet_gpu::embed::umap(knn, umap_cfg, nullptr);
    auto h_emb1  = copy_embedding_to_host(result1);

    // Second run — same config, same knn.
    auto result2 = singlet_gpu::embed::umap(knn, umap_cfg, nullptr);
    auto h_emb2  = copy_embedding_to_host(result2);

    ASSERT_EQ(h_emb1.size(), h_emb2.size());

    // Bit-identical comparison: reinterpret as uint32 to catch -0 vs +0 etc.
    int mismatches = 0;
    for (size_t i = 0; i < h_emb1.size(); ++i) {
        uint32_t u1, u2;
        std::memcpy(&u1, &h_emb1[i], 4);
        std::memcpy(&u2, &h_emb2[i], 4);
        if (u1 != u2) ++mismatches;
    }

    EXPECT_EQ(mismatches, 0)
        << "UMAP with Random init + same seed produced " << mismatches
        << " differing float values out of " << h_emb1.size()
        << " — not bit-identical (violates determinism contract from design doc §Determinism).";
}

// =============================================================================
// TEST 4: Umap_3D_OutputShape
//
// Run with n_components=3.
// Assert result.n_components == 3 and embedding has shape [n, 3].
// =============================================================================
TEST_F(UmapCorrectnessTest, Umap_3D_OutputShape) {
    const int n = kTinyN, d = kTinyD, k = kTinyK;

    std::vector<float> h_X = make_random_embedding(n, d, kSeed);

    singlet_gpu::core::DeviceMemory<float> d_X(static_cast<size_t>(n) * d);
    cudaMemcpy(d_X.get(), h_X.data(), static_cast<size_t>(n) * d * sizeof(float),
               cudaMemcpyHostToDevice);

    singlet_gpu::graph::KnnConfig knn_cfg;
    knn_cfg.k       = k;
    knn_cfg.metric  = singlet_gpu::graph::DistanceMetric::L2;
    knn_cfg.backend = singlet_gpu::graph::KnnBackend::Exact;
    // compute_knn takes DeviceDense + stream; wrap the raw pointer non-owning.
    singlet_gpu::core::DeviceDense X_dev;
    X_dev.rows = n; X_dev.cols = d;
    X_dev.data = singlet_gpu::core::DeviceMemory<float>::wrap(d_X.get(),
                                                               static_cast<size_t>(n) * d);
    auto knn = singlet_gpu::graph::compute_knn(X_dev, knn_cfg, nullptr);

    singlet_gpu::embed::UmapConfig umap_cfg;
    umap_cfg.n_components = 3;
    umap_cfg.init         = singlet_gpu::embed::UmapInit::Random;
    umap_cfg.seed         = 42;
    auto result = singlet_gpu::embed::umap(knn, umap_cfg, nullptr);

    EXPECT_EQ(result.n_components, 3)
        << "Expected n_components=3 in UmapResult";
    EXPECT_EQ(result.n, n)
        << "Expected n=" << n << " in UmapResult";
    // embedding is n * n_components floats.
    EXPECT_EQ(result.embedding.size(), static_cast<size_t>(n) * 3)
        << "Expected embedding buffer of size n * n_components = " << (n * 3);
}

// =============================================================================
// TEST 5: Umap_NComponents1_OutputShape
//
// Run with n_components=1.
// Assert result.n_components == 1 and embedding has shape [n, 1].
// =============================================================================
TEST_F(UmapCorrectnessTest, Umap_NComponents1_OutputShape) {
    const int n = kSmallN, d = kSmallD, k = kSmallK;

    std::vector<float> h_X = make_random_embedding(n, d, kSeed + 1);

    singlet_gpu::core::DeviceMemory<float> d_X(static_cast<size_t>(n) * d);
    cudaMemcpy(d_X.get(), h_X.data(), static_cast<size_t>(n) * d * sizeof(float),
               cudaMemcpyHostToDevice);

    singlet_gpu::graph::KnnConfig knn_cfg;
    knn_cfg.k       = k;
    knn_cfg.metric  = singlet_gpu::graph::DistanceMetric::L2;
    knn_cfg.backend = singlet_gpu::graph::KnnBackend::Exact;
    // compute_knn takes DeviceDense + stream; wrap the raw pointer non-owning.
    singlet_gpu::core::DeviceDense X_dev;
    X_dev.rows = n; X_dev.cols = d;
    X_dev.data = singlet_gpu::core::DeviceMemory<float>::wrap(d_X.get(),
                                                               static_cast<size_t>(n) * d);
    auto knn = singlet_gpu::graph::compute_knn(X_dev, knn_cfg, nullptr);

    singlet_gpu::embed::UmapConfig umap_cfg;
    umap_cfg.n_components = 1;
    umap_cfg.init         = singlet_gpu::embed::UmapInit::Random;
    umap_cfg.seed         = 42;
    auto result = singlet_gpu::embed::umap(knn, umap_cfg, nullptr);

    EXPECT_EQ(result.n_components, 1)
        << "Expected n_components=1 in UmapResult";
    EXPECT_EQ(result.n, n)
        << "Expected n=" << n << " in UmapResult";
    EXPECT_EQ(result.embedding.size(), static_cast<size_t>(n) * 1)
        << "Expected embedding buffer of size n * 1 = " << n;
}

// =============================================================================
// TEST 6: Umap_NComponentsTooLarge_Errors
//
// Run with n_components=51.
// cuML UMAP silently clamps or throws; our adapter must surface a clean error
// (std::runtime_error, std::invalid_argument, or similar) rather than crashing
// or silently truncating.
// =============================================================================
TEST_F(UmapCorrectnessTest, Umap_NComponentsTooLarge_Errors) {
    const int n = kSmallN, d = kSmallD, k = kSmallK;

    std::vector<float> h_X = make_random_embedding(n, d, kSeed + 2);

    singlet_gpu::core::DeviceMemory<float> d_X(static_cast<size_t>(n) * d);
    cudaMemcpy(d_X.get(), h_X.data(), static_cast<size_t>(n) * d * sizeof(float),
               cudaMemcpyHostToDevice);

    singlet_gpu::graph::KnnConfig knn_cfg;
    knn_cfg.k       = k;
    knn_cfg.metric  = singlet_gpu::graph::DistanceMetric::L2;
    knn_cfg.backend = singlet_gpu::graph::KnnBackend::Exact;
    // compute_knn takes DeviceDense + stream; wrap the raw pointer non-owning.
    singlet_gpu::core::DeviceDense X_dev;
    X_dev.rows = n; X_dev.cols = d;
    X_dev.data = singlet_gpu::core::DeviceMemory<float>::wrap(d_X.get(),
                                                               static_cast<size_t>(n) * d);
    auto knn = singlet_gpu::graph::compute_knn(X_dev, knn_cfg, nullptr);

    singlet_gpu::embed::UmapConfig umap_cfg;
    umap_cfg.n_components = 51;  // exceeds cuML UMAP maximum (typically 50)
    umap_cfg.init         = singlet_gpu::embed::UmapInit::Random;
    umap_cfg.seed         = 42;

    EXPECT_THROW(
        { auto result = singlet_gpu::embed::umap(knn, umap_cfg, nullptr); },
        std::exception
    ) << "Expected an exception for n_components=51 (exceeds cuML limit); "
      << "got no exception — adapter must validate n_components before delegating to cuML.";
}

// =============================================================================
// TEST 7: Umap_RequireKnnGraph_Throws
//
// Pass an empty KnnResult (n=0, k=0, null device pointers).
// The adapter must throw (std::invalid_argument or std::runtime_error) rather
// than invoking cuML with garbage inputs.
// =============================================================================
TEST_F(UmapCorrectnessTest, Umap_RequireKnnGraph_Throws) {
    singlet_gpu::graph::KnnResult empty_knn;
    // Default-constructed: n=0, k=0, empty DeviceMemory buffers.

    singlet_gpu::embed::UmapConfig umap_cfg;
    umap_cfg.n_components = 2;
    umap_cfg.init         = singlet_gpu::embed::UmapInit::Random;
    umap_cfg.seed         = 42;

    EXPECT_THROW(
        { auto result = singlet_gpu::embed::umap(empty_knn, umap_cfg, nullptr); },
        std::exception
    ) << "Expected an exception when passing an empty KnnResult to umap(); "
      << "got no exception — adapter must guard against n=0 inputs.";
}

// =============================================================================
// Test directory macro (resolved by CMake via target_compile_definitions)
// =============================================================================
#ifndef SINGLET_GPU_TEST_DIR
#define SINGLET_GPU_TEST_DIR "."
#endif
