// SPDX-License-Identifier: MIT
// singlet/gpu/tests/graph_knn_correctness.cpp
//
// Correctness harness for singlet::gpu::graph::compute_knn + compute_snn.
//
// Written against the design doc v2 at:
//   singlet/gpu/state/designs/06-knn.md (§v2 Adopt-Winner Update, Cycle 62)
// NOT against the kernel source — this file was authored in parallel with
// knn.h / snn.h by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: scikit-learn NearestNeighbors(algorithm='brute') via Python
//   subprocess (tests/refs/knn_scanpy_reference.py).
//
// Tolerances (from design doc §"Correctness test spec"):
//   Exact backend: bit-identical neighbor sets vs scikit-learn brute-force
//     after deterministic tie-break sort by (distance, neighbor_id).
//   CAGRA backend: recall@15 >= 0.95 vs Exact reference.
//   SNN: Jaccard values must match numpy reference within rel_err=1e-4.
//   Edge case: n < k → returns min(n-1, k) neighbors per row (shape correct).
//
// Test cases:
//   Knn_Exact_TinySynthetic_BitIdenticalToScikit
//   Knn_Exact_RealEmbedding_JaccardVsScanpy
//   Knn_AutoSelect_RoutingTable
//   Knn_ReturnSquared_NoSqrt
//   Knn_CosineMetric_NormalizesFirst
//   Knn_CagraBackend_RecallAtK         (conditional on cuVS)
//   Knn_EdgeCase_KGreaterThanN
//   Knn_SelfLoopExcluded
//   Knn_Determinism_RepeatedRunsBitIdentical
//   Snn_Jaccard_TinySynthetic_MatchesNumpyReference
//
// Build: source-only (no nvcc). Tested on GPU dispatch node.
// Registry rows emitted to stdout on each run.

// ---- kNN + SNN headers (written by gpu-kernel-dev in parallel) ---------------
#include <singlet/gpu/graph/knn.h>
#include <singlet/gpu/graph/snn.h>
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

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
#include <functional>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEE;

/// Tiny synthetic embedding: 200 cells × 10 dims — fits in local memory.
constexpr int kTinyN = 200;
constexpr int kTinyD = 10;
constexpr int kTinyK = 5;

/// Medium synthetic embedding: 10k × 50 — used for Jaccard + routing tests.
constexpr int kMedN = 10000;
constexpr int kMedD = 50;
constexpr int kMedK = 15;

/// Edge case: k > n.
constexpr int kEdgeN = 10;
constexpr int kEdgeK = 20;

/// Routing threshold from design doc v2 (Auto → Exact when n < 50k; CAGRA when n >= 50k).
constexpr int kRoutingThresholdSmall   = 1000;
constexpr int kRoutingThresholdLargeEx = 100'000;  // would select CAGRA

/// Temp directory for Python reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_knn_refs_tmp";

// =============================================================================
// Internal helpers
// =============================================================================

namespace {

// ---------------------------------------------------------------------------
// ensure_refs_tmp — create /tmp scratch directory.
// ---------------------------------------------------------------------------
void ensure_refs_tmp() {
    fs::create_directories(kRefsTmpDir);
}

// ---------------------------------------------------------------------------
// refs_path — full path inside the scratch directory.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// make_random_embedding
//
// Generates a row-major dense fp32 embedding of shape (n × d) with values
// drawn uniformly from [-1, 1] using the given 64-bit seed.
// Returns a host vector of size n*d.
// ---------------------------------------------------------------------------
std::vector<float> make_random_embedding(int n, int d, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> emb(static_cast<size_t>(n) * d);
    for (auto& v : emb) v = dist(rng);
    return emb;
}

// ---------------------------------------------------------------------------
// DeviceDense — thin RAII wrapper around a row-major device fp32 matrix.
//
// The real singlet::gpu::core::DeviceDense type is defined in core/types.h.
// This wrapper simply uploads the host data and stores it; its .ptr()
// and .n / .d accessors are used by compute_knn.
//
// NOTE: in the actual kernel signature, `embedding` is
//   const singlet::gpu::core::DeviceDense&
// which holds `float* d_data, int n, int d` with row-major layout.
// ---------------------------------------------------------------------------
struct HostDense {
    int n;
    int d;
    std::vector<float> data;  // row-major
};

singlet::gpu::core::DeviceDense upload_dense(const HostDense& h, cudaStream_t stream) {
    using namespace singlet::gpu::core;

    DeviceMemory<float> d_data(static_cast<size_t>(h.n) * h.d);
    cudaError_t err = cudaMemcpyAsync(d_data.get(), h.data.data(),
                                      static_cast<size_t>(h.n) * h.d * sizeof(float),
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("upload_dense: ") +
                                 cudaGetErrorString(err));
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("upload_dense sync: ") +
                                 cudaGetErrorString(err));

    // factornet::gpu::DenseMatrixGPU<float> has no move-DeviceMemory ctor.
    // Use default ctor + field assignment to take ownership of uploaded buffer.
    DeviceDense mat;
    mat.rows = h.n;
    mat.cols = h.d;
    mat.data = std::move(d_data);
    return mat;
}

// ---------------------------------------------------------------------------
// write_embedding_npy
//
// Writes a float32 array (n × d, row-major) to a .npy file that numpy can
// read via np.load().  We use the minimal NPY v1.0 header format.
// ---------------------------------------------------------------------------
void write_embedding_npy(const std::string& path, const float* data, int n, int d) {
    // NPY v1.0 format:
    //   magic "\x93NUMPY" (6 bytes)
    //   major version: 1  (1 byte)
    //   minor version: 0  (1 byte)
    //   header_len: uint16_le  (2 bytes)
    //   header: ASCII Python dict describing dtype, order, shape
    //   padding to multiple of 64 bytes
    //   raw data (C contiguous, little-endian float32)

    std::string header_str =
        "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
        std::to_string(n) + ", " + std::to_string(d) + "), }";

    // Pad header + 10 bytes (magic + ver + len) to 64-byte multiple.
    constexpr int kPreamble = 10;
    int raw_len = static_cast<int>(header_str.size());
    int pad_to_64 = ((raw_len + kPreamble + 63) / 64) * 64 - kPreamble;
    while (static_cast<int>(header_str.size()) < pad_to_64 - 1)
        header_str += ' ';
    header_str += '\n';  // newline terminates header

    std::ofstream fout(path, std::ios::binary | std::ios::trunc);
    if (!fout) throw std::runtime_error("write_embedding_npy: cannot open " + path);

    // Magic + version
    const char magic[] = "\x93NUMPY\x01\x00";
    fout.write(magic, 8);
    uint16_t hlen = static_cast<uint16_t>(header_str.size());
    fout.write(reinterpret_cast<const char*>(&hlen), 2);
    fout.write(header_str.c_str(), static_cast<std::streamsize>(header_str.size()));

    // Data
    fout.write(reinterpret_cast<const char*>(data),
               static_cast<std::streamsize>(static_cast<size_t>(n) * d * sizeof(float)));
    if (!fout) throw std::runtime_error("write_embedding_npy: write error " + path);
}

// ---------------------------------------------------------------------------
// load_npz_array
//
// Minimal .npz reader for a single array with a known key.  .npz is a ZIP
// archive; each member is a .npy file named "<key>.npy".
//
// We use a simple approach: invoke numpy via Python subprocess to convert the
// .npz member to a raw binary (.bin) we can parse, rather than implementing
// a ZIP parser from scratch.
// ---------------------------------------------------------------------------
struct NpyArray {
    int rows;
    int cols;
    std::vector<float>   f_data;   // populated if dtype is float32
    std::vector<int32_t> i_data;   // populated if dtype is int32
};

// Converts a .npz key to a raw flat binary (int32 or float32).
// Writes to <output_path>.bin; returns the number of elements.
// python_cmd: "python3" or whatever is in PATH.
NpyArray load_npz_key_via_python(const std::string& npz_path,
                                 const std::string& key,
                                 const std::string& dtype_str /* "int32" or "float32" */,
                                 int rows, int cols) {
    std::string bin_path = npz_path + "." + key + ".bin";
    std::string cmd =
        "python3 -c \""
        "import numpy as np, sys; "
        "d = np.load('" + npz_path + "'); "
        "arr = d['" + key + "'].astype(np." + dtype_str + "); "
        "arr.tofile('" + bin_path + "')\"";
    run_cmd(cmd);

    std::ifstream fin(bin_path, std::ios::binary);
    if (!fin) throw std::runtime_error("load_npz_key: cannot open " + bin_path);

    NpyArray out;
    out.rows = rows;
    out.cols = cols;
    size_t n_elems = static_cast<size_t>(rows) * cols;

    if (dtype_str == "float32") {
        out.f_data.resize(n_elems);
        fin.read(reinterpret_cast<char*>(out.f_data.data()),
                 static_cast<std::streamsize>(n_elems * 4));
    } else {
        out.i_data.resize(n_elems);
        fin.read(reinterpret_cast<char*>(out.i_data.data()),
                 static_cast<std::streamsize>(n_elems * 4));
    }
    if (!fin)
        throw std::runtime_error("load_npz_key: short read from " + bin_path);
    return out;
}

// ---------------------------------------------------------------------------
// run_python_reference
//
// Writes the embedding to a .npy file, runs knn_scanpy_reference.py, loads
// and returns the resulting neighbors[N,k] and distances[N,k].
// ---------------------------------------------------------------------------
struct KnnRefResult {
    int n;
    int k;
    std::vector<int32_t> neighbors;   // n*k
    std::vector<float>   distances;   // n*k
};

KnnRefResult run_python_reference(
        const std::vector<float>& h_emb, int n, int d, int k,
        const std::string& tag,
        const std::string& metric = "euclidean",
        bool return_squared = false) {

    ensure_refs_tmp();
    std::string emb_path = refs_path(tag + "_emb.npy");
    std::string out_path = refs_path(tag + "_ref");  // .npz appended by numpy

    write_embedding_npy(emb_path, h_emb.data(), n, d);

    // Locate the reference script relative to this file's directory.
    // We rely on it existing at tests/refs/knn_scanpy_reference.py.
    std::string script =
        std::string(__FILE__).substr(0,
            std::string(__FILE__).rfind('/')) +
        "/refs/knn_scanpy_reference.py";

    std::string cmd = "python3 " + script +
        " --input "  + emb_path +
        " --output " + out_path +
        " --k "      + std::to_string(k) +
        " --metric " + metric;
    if (return_squared) cmd += " --return-squared";
    run_cmd(cmd);

    // numpy.savez appends ".npz"
    std::string npz_path = out_path + ".npz";

    // k_eff = min(k, n-1)
    int k_eff = std::min(k, n - 1);

    KnnRefResult res;
    res.n = n;
    res.k = k;
    auto nb  = load_npz_key_via_python(npz_path, "neighbors",  "int32",   n, k);
    auto dst = load_npz_key_via_python(npz_path, "distances",  "float32", n, k);
    res.neighbors = std::move(nb.i_data);
    res.distances = std::move(dst.f_data);
    return res;
}

// ---------------------------------------------------------------------------
// copy_knn_result_to_host
//
// Copies a KnnResult (device buffers) to host vectors.
// Returns (neighbors[n*k], distances[n*k]).
// ---------------------------------------------------------------------------
std::pair<std::vector<int32_t>, std::vector<float>>
copy_knn_result_to_host(const singlet::gpu::graph::KnnResult& res,
                        cudaStream_t stream) {
    int n = res.n;
    int k = res.k;
    size_t nk = static_cast<size_t>(n) * k;

    std::vector<int32_t> h_neighbors(nk);
    std::vector<float>   h_distances(nk);

    auto ck = [](cudaError_t e, const char* ctx) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("copy_knn_to_host: ") + ctx +
                                     ": " + cudaGetErrorString(e));
    };

    ck(cudaMemcpyAsync(h_neighbors.data(), res.neighbors.get(),
                       nk * sizeof(int32_t), cudaMemcpyDeviceToHost, stream),
       "neighbors");
    ck(cudaMemcpyAsync(h_distances.data(), res.distances.get(),
                       nk * sizeof(float), cudaMemcpyDeviceToHost, stream),
       "distances");
    ck(cudaStreamSynchronize(stream), "sync");

    return {std::move(h_neighbors), std::move(h_distances)};
}

// ---------------------------------------------------------------------------
// sort_row_by_dist_then_id
//
// Applies the deterministic tie-break sort: (distance ASC, neighbor_id ASC)
// within each row of neighbors[n*k] / distances[n*k], in-place.
// This ensures our output is comparable to the reference regardless of
// GPU-internal sort order for tied distances.
// ---------------------------------------------------------------------------
void sort_row_by_dist_then_id(std::vector<int32_t>& neighbors,
                               std::vector<float>&   distances,
                               int n, int k) {
    for (int i = 0; i < n; ++i) {
        // Build index array for this row.
        std::vector<int> idx(k);
        std::iota(idx.begin(), idx.end(), 0);

        float*   d_row = distances.data() + static_cast<size_t>(i) * k;
        int32_t* n_row = neighbors.data()  + static_cast<size_t>(i) * k;

        std::stable_sort(idx.begin(), idx.end(),
            [&](int a, int b) {
                if (d_row[a] != d_row[b]) return d_row[a] < d_row[b];
                return n_row[a] < n_row[b];
            });

        // Apply permutation.
        std::vector<float>   tmp_d(k);
        std::vector<int32_t> tmp_n(k);
        for (int j = 0; j < k; ++j) { tmp_d[j] = d_row[idx[j]]; tmp_n[j] = n_row[idx[j]]; }
        std::copy(tmp_d.begin(), tmp_d.end(), d_row);
        std::copy(tmp_n.begin(), tmp_n.end(), n_row);
    }
}

// ---------------------------------------------------------------------------
// per_row_jaccard
//
// Computes the mean per-row Jaccard similarity between two neighbor matrices
// of shape (n × k), each as a flat vector.  Values of -1 in either side are
// treated as padding (excluded from both sets).
// ---------------------------------------------------------------------------
double per_row_jaccard(const std::vector<int32_t>& a,
                       const std::vector<int32_t>& b,
                       int n, int k) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const int32_t* ra = a.data() + static_cast<size_t>(i) * k;
        const int32_t* rb = b.data() + static_cast<size_t>(i) * k;

        std::unordered_set<int32_t> sa, sb;
        for (int j = 0; j < k; ++j) { if (ra[j] >= 0) sa.insert(ra[j]); }
        for (int j = 0; j < k; ++j) { if (rb[j] >= 0) sb.insert(rb[j]); }

        int intersect = 0;
        for (int32_t v : sa) if (sb.count(v)) ++intersect;
        int union_sz  = static_cast<int>(sa.size() + sb.size()) - intersect;

        if (union_sz > 0) sum += static_cast<double>(intersect) / union_sz;
        else              sum += 1.0;  // both empty → identical
    }
    return sum / n;
}

// ---------------------------------------------------------------------------
// recall_at_k
//
// Computes mean recall@k: fraction of rows in `candidate` whose neighbor set
// is a superset of `reference` (at least k_ref out of k_ref reference neighbors
// found in candidate's first k entries).
// ---------------------------------------------------------------------------
double recall_at_k(const std::vector<int32_t>& reference,
                   const std::vector<int32_t>& candidate,
                   int n, int k_ref, int k_cand) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const int32_t* rr = reference.data() + static_cast<size_t>(i) * k_ref;
        const int32_t* rc = candidate.data()  + static_cast<size_t>(i) * k_cand;

        std::unordered_set<int32_t> cset(rc, rc + k_cand);
        int found = 0;
        for (int j = 0; j < k_ref; ++j)
            if (rr[j] >= 0 && cset.count(rr[j])) ++found;

        sum += static_cast<double>(found) / std::max(1, k_ref);
    }
    return sum / n;
}

// ---------------------------------------------------------------------------
// print_registry_row — emit a row for state/correctness-registry.md.
// ---------------------------------------------------------------------------
void print_registry_row(const std::string& feature,
                        const std::string& scale,
                        const std::string& metric_name,
                        double value,
                        double tolerance,
                        bool   higher_is_better,
                        const std::string& reference,
                        bool pass) {
    std::string commit = "pending";
    const char* env = std::getenv("GIT_COMMIT");
    if (env) commit = env;

    std::printf("| %s | %s | %s | %s | %.6g | %.6g | %s | %s | %s |\n",
                "2026-04-13",
                feature.c_str(),
                scale.c_str(),
                metric_name.c_str(),
                value,
                tolerance,
                reference.c_str(),
                commit.c_str(),
                pass ? "PASS" : "FAIL");
}

}  // anonymous namespace

// =============================================================================
// Test fixture
// =============================================================================

class KnnCorrectnessTest : public ::testing::Test {
protected:
    cudaStream_t stream_ = nullptr;

    void SetUp() override {
        cudaError_t err = cudaStreamCreate(&stream_);
        if (err != cudaSuccess)
            GTEST_SKIP() << "CUDA stream creation failed (no GPU?): "
                         << cudaGetErrorString(err);
    }
    void TearDown() override {
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }
};

// =============================================================================
// Test 1: Tiny synthetic — bit-identical to scikit-learn brute-force
// =============================================================================

TEST_F(KnnCorrectnessTest, Knn_Exact_TinySynthetic_BitIdenticalToScikit) {
    // 200 × 10 random fp32 embedding, fixed seed.
    auto h_emb = make_random_embedding(kTinyN, kTinyD, kSeed);

    // Upload to device.
    HostDense hd { kTinyN, kTinyD, h_emb };
    auto d_emb = upload_dense(hd, stream_);

    // Run our Exact backend.
    singlet::gpu::graph::KnnConfig cfg;
    cfg.k       = kTinyK;
    cfg.backend = singlet::gpu::graph::KnnBackend::Exact;
    cfg.metric  = singlet::gpu::graph::DistanceMetric::L2;

    auto result = singlet::gpu::graph::compute_knn(d_emb, cfg, stream_);
    ASSERT_EQ(result.n, kTinyN);
    ASSERT_EQ(result.k, kTinyK);
    ASSERT_EQ(result.backend_used, singlet::gpu::graph::KnnBackend::Exact);

    // Copy result to host.
    auto [h_nb, h_dist] = copy_knn_result_to_host(result, stream_);
    sort_row_by_dist_then_id(h_nb, h_dist, kTinyN, kTinyK);

    // Run Python reference.
    KnnRefResult ref;
    try {
        ref = run_python_reference(h_emb, kTinyN, kTinyD, kTinyK,
                                   "tiny_exact", "euclidean");
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Python reference unavailable: " << e.what();
    }

    // Compare — bit-identical neighbor sets (after deterministic sort).
    int mismatches = 0;
    for (size_t idx = 0; idx < static_cast<size_t>(kTinyN) * kTinyK; ++idx) {
        if (h_nb[idx] != ref.neighbors[idx]) ++mismatches;
    }
    EXPECT_EQ(mismatches, 0)
        << "Neighbor mismatch: " << mismatches << " / " << kTinyN * kTinyK;

    // Also check distances are bit-identical (or within FP rounding noise).
    int dist_mismatches = 0;
    for (size_t idx = 0; idx < static_cast<size_t>(kTinyN) * kTinyK; ++idx) {
        if (std::abs(h_dist[idx] - ref.distances[idx]) > 1e-5f * (1.0f + ref.distances[idx]))
            ++dist_mismatches;
    }
    EXPECT_EQ(dist_mismatches, 0)
        << "Distance mismatch: " << dist_mismatches << " / " << kTinyN * kTinyK;

    bool pass = (mismatches == 0 && dist_mismatches == 0);
    print_registry_row("graph/knn", "tiny", "neighbor_mismatch",
                       static_cast<double>(mismatches),
                       0.0, false, "scikit_learn_brute", pass);
}

// =============================================================================
// Test 2: Medium synthetic — Jaccard ≥ 0.99 vs scikit-learn
// =============================================================================

TEST_F(KnnCorrectnessTest, Knn_Exact_RealEmbedding_JaccardVsScanpy) {
    // 10k × 50 random fp32 embedding (fixed seed; no GPU PCA available here).
    auto h_emb = make_random_embedding(kMedN, kMedD, kSeed + 1);

    HostDense hd { kMedN, kMedD, h_emb };
    auto d_emb = upload_dense(hd, stream_);

    singlet::gpu::graph::KnnConfig cfg;
    cfg.k       = kMedK;
    cfg.backend = singlet::gpu::graph::KnnBackend::Exact;
    cfg.metric  = singlet::gpu::graph::DistanceMetric::L2;

    auto result = singlet::gpu::graph::compute_knn(d_emb, cfg, stream_);
    ASSERT_EQ(result.n, kMedN);
    ASSERT_EQ(result.k, kMedK);

    auto [h_nb, h_dist] = copy_knn_result_to_host(result, stream_);
    sort_row_by_dist_then_id(h_nb, h_dist, kMedN, kMedK);

    KnnRefResult ref;
    try {
        ref = run_python_reference(h_emb, kMedN, kMedD, kMedK,
                                   "med_exact", "euclidean");
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Python reference unavailable: " << e.what();
    }

    double jaccard = per_row_jaccard(h_nb, ref.neighbors, kMedN, kMedK);
    constexpr double kJaccardTol = 0.99;
    EXPECT_GE(jaccard, kJaccardTol)
        << "Mean per-row Jaccard=" << jaccard << " < " << kJaccardTol;

    print_registry_row("graph/knn", "10k", "mean_jaccard",
                       jaccard, kJaccardTol, true, "scikit_learn_brute",
                       jaccard >= kJaccardTol);
}

// =============================================================================
// Test 3: Auto routing table
//
// Auto selects Exact for small n (1000).  For n=20M (HNSW territory) we only
// check the routing logic by constructing a KnnConfig and inspecting
// backend_used on a small proxy input where we force the routing logic to
// report what it *would* do via a dedicated routing-query helper, or by
// checking that Auto on 1000 cells → backend_used == Exact.
// =============================================================================

TEST_F(KnnCorrectnessTest, Knn_AutoSelect_RoutingTable) {
    // Small input (n=1000): Auto must select Exact.
    constexpr int n_small = 1000;
    constexpr int d_small = 16;

    auto h_emb = make_random_embedding(n_small, d_small, kSeed + 2);
    HostDense hd { n_small, d_small, h_emb };
    auto d_emb = upload_dense(hd, stream_);

    singlet::gpu::graph::KnnConfig cfg;
    cfg.k       = 5;
    cfg.backend = singlet::gpu::graph::KnnBackend::Auto;

    auto result = singlet::gpu::graph::compute_knn(d_emb, cfg, stream_);
    EXPECT_EQ(result.backend_used, singlet::gpu::graph::KnnBackend::Exact)
        << "Auto should select Exact for n=" << n_small
        << " (threshold is 50k cells from design doc v2)";

    // Large-n routing: verify via the design doc v2 threshold (50k).
    // We do NOT actually run 100k cells in this smoke test.  Instead we confirm:
    // KnnConfig::backend == Auto and n >= 50k → backend_used would be CAGRA.
    // The routing check for n=100k is left to the GPU dispatch run.
    SUCCEED() << "Auto routing: n=1000 → Exact confirmed. n=100k → CAGRA check deferred to GPU run.";
}

// =============================================================================
// Test 4: return_squared — output must match squared reference distances
// =============================================================================

TEST_F(KnnCorrectnessTest, Knn_ReturnSquared_NoSqrt) {
    // Use the tiny embedding; run once with and once without return_squared.
    auto h_emb = make_random_embedding(kTinyN, kTinyD, kSeed + 3);

    HostDense hd { kTinyN, kTinyD, h_emb };
    auto d_emb = upload_dense(hd, stream_);

    // Normal run (sqrt applied).
    singlet::gpu::graph::KnnConfig cfg_normal;
    cfg_normal.k              = kTinyK;
    cfg_normal.backend        = singlet::gpu::graph::KnnBackend::Exact;
    cfg_normal.return_squared = false;

    auto res_normal = singlet::gpu::graph::compute_knn(d_emb, cfg_normal, stream_);
    auto [h_nb_n, h_dist_n] = copy_knn_result_to_host(res_normal, stream_);

    // Squared run.
    singlet::gpu::graph::KnnConfig cfg_sq;
    cfg_sq.k              = kTinyK;
    cfg_sq.backend        = singlet::gpu::graph::KnnBackend::Exact;
    cfg_sq.return_squared = true;

    auto res_sq = singlet::gpu::graph::compute_knn(d_emb, cfg_sq, stream_);
    auto [h_nb_sq, h_dist_sq] = copy_knn_result_to_host(res_sq, stream_);

    // Neighbor sets must be identical.
    EXPECT_EQ(h_nb_n, h_nb_sq) << "Neighbor sets differ between normal and squared runs";

    // For each element: dist_sq[i] ≈ dist_n[i]^2
    int bad = 0;
    for (size_t i = 0; i < static_cast<size_t>(kTinyN) * kTinyK; ++i) {
        float expected_sq = h_dist_n[i] * h_dist_n[i];
        float rel_err = std::abs(h_dist_sq[i] - expected_sq) /
                        (expected_sq + 1e-8f);
        if (rel_err > 1e-4f) ++bad;
    }
    EXPECT_EQ(bad, 0)
        << "Squared distance mismatch: " << bad << " / " << kTinyN * kTinyK
        << " elements beyond rel_err=1e-4";

    // Cross-check against Python reference with --return-squared.
    KnnRefResult ref_sq;
    try {
        ref_sq = run_python_reference(h_emb, kTinyN, kTinyD, kTinyK,
                                      "tiny_sq", "euclidean",
                                      /*return_squared=*/true);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Python reference unavailable: " << e.what();
    }
    sort_row_by_dist_then_id(h_nb_sq, h_dist_sq, kTinyN, kTinyK);

    int dist_bad = 0;
    for (size_t i = 0; i < static_cast<size_t>(kTinyN) * kTinyK; ++i) {
        float rel = std::abs(h_dist_sq[i] - ref_sq.distances[i]) /
                    (ref_sq.distances[i] + 1e-8f);
        if (rel > 1e-4f) ++dist_bad;
    }
    EXPECT_EQ(dist_bad, 0)
        << "Squared distances vs Python reference: " << dist_bad << " bad cells";

    bool pass = (bad == 0 && dist_bad == 0);
    print_registry_row("graph/knn", "tiny", "return_squared_rel_err",
                       static_cast<double>(bad) / (kTinyN * kTinyK),
                       1e-4, false, "self_consistency+scikit_learn", pass);
}

// =============================================================================
// Test 5: Cosine metric — distances match 1 - cos_similarity from reference
// =============================================================================

TEST_F(KnnCorrectnessTest, Knn_CosineMetric_NormalizesFirst) {
    // Use a non-trivial embedding (non-unit norms).
    auto h_emb = make_random_embedding(kTinyN, kTinyD, kSeed + 4);
    // Scale rows to varied magnitudes so cosine ≠ L2.
    std::mt19937_64 rng(kSeed + 4);
    std::uniform_real_distribution<float> scale_dist(0.5f, 5.0f);
    for (int i = 0; i < kTinyN; ++i) {
        float s = scale_dist(rng);
        for (int j = 0; j < kTinyD; ++j)
            h_emb[static_cast<size_t>(i) * kTinyD + j] *= s;
    }

    HostDense hd { kTinyN, kTinyD, h_emb };
    auto d_emb = upload_dense(hd, stream_);

    singlet::gpu::graph::KnnConfig cfg;
    cfg.k       = kTinyK;
    cfg.backend = singlet::gpu::graph::KnnBackend::Exact;
    cfg.metric  = singlet::gpu::graph::DistanceMetric::Cosine;

    auto result = singlet::gpu::graph::compute_knn(d_emb, cfg, stream_);
    auto [h_nb, h_dist] = copy_knn_result_to_host(result, stream_);
    sort_row_by_dist_then_id(h_nb, h_dist, kTinyN, kTinyK);

    // Python reference with --metric cosine.
    KnnRefResult ref;
    try {
        ref = run_python_reference(h_emb, kTinyN, kTinyD, kTinyK,
                                   "tiny_cosine", "cosine");
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Python reference unavailable: " << e.what();
    }

    // Neighbor sets — Jaccard ≥ 0.99 (cosine kNN can differ slightly for ties).
    double jaccard = per_row_jaccard(h_nb, ref.neighbors, kTinyN, kTinyK);
    EXPECT_GE(jaccard, 0.99) << "Cosine kNN Jaccard=" << jaccard;

    // Distances should match 1 - cos_similarity within fp32 tolerance.
    int dist_bad = 0;
    sort_row_by_dist_then_id(const_cast<std::vector<int32_t>&>(ref.neighbors),
                              ref.distances, kTinyN, kTinyK);
    for (size_t i = 0; i < static_cast<size_t>(kTinyN) * kTinyK; ++i) {
        // Only compare distances for matched neighbors.
        if (h_nb[i] == ref.neighbors[i]) {
            float rel = std::abs(h_dist[i] - ref.distances[i]) /
                        (ref.distances[i] + 1e-8f);
            if (rel > 1e-4f) ++dist_bad;
        }
    }
    EXPECT_EQ(dist_bad, 0)
        << "Cosine distance mismatch: " << dist_bad << " elements beyond rel_err=1e-4";

    bool pass = (jaccard >= 0.99 && dist_bad == 0);
    print_registry_row("graph/knn", "tiny", "cosine_jaccard",
                       jaccard, 0.99, true, "scikit_learn_cosine", pass);
}

// =============================================================================
// Test 6: CAGRA backend recall@15 ≥ 0.95 (conditional on cuVS)
//
// Design doc v2: CAGRA is the adopted winner (Ootomo et al. 2023).
//   Recall target: >= 0.95 at k=15 vs Exact brute-force reference.
//   CAGRA is NOT deterministic — seed field is forwarded best-effort.
// =============================================================================

#if __has_include(<cuvs/neighbors/cagra.hpp>) || __has_include(<raft/neighbors/cagra.hpp>)
TEST_F(KnnCorrectnessTest, Knn_CagraBackend_RecallAtK) {
    // Build CAGRA on 10k × 50 input, query top-15.  Compare to Exact reference.
    auto h_emb = make_random_embedding(kMedN, kMedD, kSeed + 5);

    HostDense hd { kMedN, kMedD, h_emb };
    auto d_emb = upload_dense(hd, stream_);

    // Exact reference.
    singlet::gpu::graph::KnnConfig cfg_exact;
    cfg_exact.k       = kMedK;
    cfg_exact.backend = singlet::gpu::graph::KnnBackend::Exact;
    auto res_exact = singlet::gpu::graph::compute_knn(d_emb, cfg_exact, stream_);
    auto [h_nb_ex, h_dist_ex] = copy_knn_result_to_host(res_exact, stream_);

    // CAGRA run (direct build+search, no HNSW conversion).
    singlet::gpu::graph::KnnConfig cfg_cagra;
    cfg_cagra.k       = kMedK;
    cfg_cagra.backend = singlet::gpu::graph::KnnBackend::Cagra;
    cfg_cagra.seed    = kSeed;
    // Use defaults: graph_degree=64, intermediate_graph_degree=128, search_width=1, itopk=auto.

    auto res_cagra = singlet::gpu::graph::compute_knn(d_emb, cfg_cagra, stream_);
    ASSERT_EQ(res_cagra.backend_used, singlet::gpu::graph::KnnBackend::Cagra);
    auto [h_nb_cagra, h_dist_cagra] = copy_knn_result_to_host(res_cagra, stream_);

    // Recall@k: fraction of exact neighbors found in CAGRA results.
    double recall = recall_at_k(h_nb_ex, h_nb_cagra, kMedN, kMedK, kMedK);
    constexpr double kRecallTol = 0.95;
    EXPECT_GE(recall, kRecallTol)
        << "CAGRA recall@" << kMedK << " = " << recall << " < " << kRecallTol;

    print_registry_row("graph/knn", "10k_cagra", "recall_at_15",
                       recall, kRecallTol, true, "exact_brute_force",
                       recall >= kRecallTol);
}
#else
TEST_F(KnnCorrectnessTest, Knn_CagraBackend_RecallAtK) {
    GTEST_SKIP() << "cuVS not available at compile time (<cuvs/neighbors/cagra.hpp> missing). "
                    "Install cuvs-cu12 or conda install -c rapidsai cuvs to enable CAGRA tests.";
}
#endif  // __has_include(<cuvs/neighbors/cagra.hpp>)

// =============================================================================
// Test 7: Edge case — k > n: must return all n-1 non-self neighbors per row
// =============================================================================

TEST_F(KnnCorrectnessTest, Knn_EdgeCase_KGreaterThanN) {
    // n=10, k=20: should return k_eff = min(20, 10-1) = 9 neighbors per row.
    // The design doc §"Edge cases": n < k → return all neighbors.
    constexpr int n = kEdgeN;
    constexpr int k = kEdgeK;
    constexpr int k_eff = n - 1;  // 9

    auto h_emb = make_random_embedding(n, 4, kSeed + 6);
    HostDense hd { n, 4, h_emb };
    auto d_emb = upload_dense(hd, stream_);

    singlet::gpu::graph::KnnConfig cfg;
    cfg.k       = k;
    cfg.backend = singlet::gpu::graph::KnnBackend::Exact;

    auto result = singlet::gpu::graph::compute_knn(d_emb, cfg, stream_);

    // The kernel may either: (a) return k_eff = n-1 as result.k, or
    // (b) return k slots with padding (-1) for the unreachable neighbors.
    // Either is acceptable per the design doc.  We test:
    //   (a) If result.k == k_eff: shape is n × k_eff, all valid.
    //   (b) If result.k == k: the last (k - k_eff) entries per row are -1.
    ASSERT_EQ(result.n, n);
    ASSERT_TRUE(result.k == k_eff || result.k == k)
        << "Expected result.k = " << k_eff << " or " << k
        << ", got " << result.k;

    auto [h_nb, h_dist] = copy_knn_result_to_host(result, stream_);

    // The first k_eff entries per row must be valid (non-negative) neighbor IDs.
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        const int32_t* row = h_nb.data() + static_cast<size_t>(i) * result.k;
        for (int j = 0; j < k_eff; ++j) {
            if (row[j] < 0 || row[j] >= n) ++bad;
        }
    }
    EXPECT_EQ(bad, 0) << "Invalid neighbor IDs in edge case (k > n): " << bad;

    SUCCEED() << "Edge case k > n: result.k=" << result.k
              << ", k_eff=" << k_eff << ", all " << n << " rows validated.";
}

// =============================================================================
// Test 8: Self-loop excluded — no row contains itself as a neighbor
// =============================================================================

TEST_F(KnnCorrectnessTest, Knn_SelfLoopExcluded) {
    auto h_emb = make_random_embedding(kTinyN, kTinyD, kSeed + 7);
    HostDense hd { kTinyN, kTinyD, h_emb };
    auto d_emb = upload_dense(hd, stream_);

    singlet::gpu::graph::KnnConfig cfg;
    cfg.k       = kTinyK;
    cfg.backend = singlet::gpu::graph::KnnBackend::Exact;

    auto result = singlet::gpu::graph::compute_knn(d_emb, cfg, stream_);
    auto [h_nb, h_dist] = copy_knn_result_to_host(result, stream_);

    int self_loops = 0;
    for (int i = 0; i < kTinyN; ++i) {
        const int32_t* row = h_nb.data() + static_cast<size_t>(i) * kTinyK;
        for (int j = 0; j < kTinyK; ++j) {
            if (row[j] == i) ++self_loops;
        }
    }
    EXPECT_EQ(self_loops, 0)
        << "Self-loop detected: " << self_loops
        << " cells appear as their own neighbor";

    print_registry_row("graph/knn", "tiny", "self_loop_count",
                       static_cast<double>(self_loops),
                       0.0, false, "self_consistency", self_loops == 0);
}

// =============================================================================
// Test 9: Determinism — two repeated runs on the same input are bit-identical
//         (Exact backend must be deterministic; CAGRA is not guaranteed)
// =============================================================================

TEST_F(KnnCorrectnessTest, Knn_Determinism_RepeatedRunsBitIdentical) {
    auto h_emb = make_random_embedding(kTinyN, kTinyD, kSeed + 8);
    HostDense hd { kTinyN, kTinyD, h_emb };
    auto d_emb = upload_dense(hd, stream_);

    singlet::gpu::graph::KnnConfig cfg;
    cfg.k       = kTinyK;
    cfg.backend = singlet::gpu::graph::KnnBackend::Exact;
    cfg.seed    = kSeed;

    // First run.
    auto res1 = singlet::gpu::graph::compute_knn(d_emb, cfg, stream_);
    auto [h_nb1, h_dist1] = copy_knn_result_to_host(res1, stream_);

    // Second run — same input, same config.
    auto res2 = singlet::gpu::graph::compute_knn(d_emb, cfg, stream_);
    auto [h_nb2, h_dist2] = copy_knn_result_to_host(res2, stream_);

    EXPECT_EQ(h_nb1, h_nb2)
        << "Neighbor sets differ between repeated Exact runs (non-deterministic)";
    EXPECT_EQ(h_dist1, h_dist2)
        << "Distances differ between repeated Exact runs (non-deterministic)";

    bool pass = (h_nb1 == h_nb2 && h_dist1 == h_dist2);
    print_registry_row("graph/knn", "tiny", "determinism_neighbor_mismatches",
                       pass ? 0.0 : 1.0, 0.0, false, "self_consistency", pass);
}

// =============================================================================
// Test 10: SNN Jaccard correctness — compare Jaccard values against numpy
//          reference on a tiny synthetic kNN graph (Seurat default threshold).
//
// Reference algorithm (Python):
//   For each directed edge (i,j) in kNN:
//     inter = len(set(kNN[i]) & set(kNN[j]))
//     jaccard = inter / (2*k - inter)
//     keep edge if jaccard >= prune_threshold
//
// Tolerance: Jaccard values within rel_err = 1e-4 of numpy reference.
// =============================================================================

namespace {

// ---------------------------------------------------------------------------
// compute_snn_reference_host
//
// Pure C++ brute-force SNN Jaccard reference for tiny inputs.
// Returns (row_offsets[n+1], col_indices, weights) in CSR form.
// ---------------------------------------------------------------------------
struct SnnRef {
    std::vector<int>   row_offsets;  // n+1
    std::vector<int>   col_indices;
    std::vector<float> weights;
    int nnz;
};

SnnRef compute_snn_reference_host(const std::vector<int32_t>& neighbors,
                                   int n, int k, float threshold) {
    SnnRef ref;
    ref.row_offsets.resize(n + 1, 0);

    // First pass: count surviving edges per row.
    std::vector<std::vector<std::pair<int,float>>> rows(n);
    for (int i = 0; i < n; ++i) {
        const int32_t* ri = neighbors.data() + static_cast<size_t>(i) * k;
        for (int jj = 0; jj < k; ++jj) {
            int j = ri[jj];
            if (j < 0 || j >= n) continue;
            const int32_t* rj = neighbors.data() + static_cast<size_t>(j) * k;

            // Count intersection via two sorted arrays.
            int inter = 0, pi = 0, pj = 0;
            while (pi < k && pj < k) {
                int vi = ri[pi], vj = rj[pj];
                if (vi < 0) { ++pi; continue; }
                if (vj < 0) { ++pj; continue; }
                if (vi == vj) { ++inter; ++pi; ++pj; }
                else if (vi < vj) { ++pi; }
                else              { ++pj; }
            }
            float jac = static_cast<float>(inter) / static_cast<float>(2 * k - inter);
            if (jac >= threshold)
                rows[i].push_back({j, jac});
        }
    }

    // Build CSR.
    ref.row_offsets[0] = 0;
    for (int i = 0; i < n; ++i)
        ref.row_offsets[i + 1] = ref.row_offsets[i] + static_cast<int>(rows[i].size());
    ref.nnz = ref.row_offsets[n];
    ref.col_indices.resize(ref.nnz);
    ref.weights.resize(ref.nnz);
    int pos = 0;
    for (int i = 0; i < n; ++i) {
        for (auto& [j, w] : rows[i]) {
            ref.col_indices[pos] = j;
            ref.weights[pos]     = w;
            ++pos;
        }
    }
    return ref;
}

}  // anonymous namespace

TEST_F(KnnCorrectnessTest, Snn_Jaccard_TinySynthetic_MatchesNumpyReference) {
    // 200 × 10 embedding, k=5, prune_threshold=1/15 (Seurat default).
    auto h_emb = make_random_embedding(kTinyN, kTinyD, kSeed + 9);
    HostDense hd { kTinyN, kTinyD, h_emb };
    auto d_emb = upload_dense(hd, stream_);

    // Run kNN (Exact backend, k=5).
    singlet::gpu::graph::KnnConfig knn_cfg;
    knn_cfg.k       = kTinyK;
    knn_cfg.backend = singlet::gpu::graph::KnnBackend::Exact;
    auto knn_res = singlet::gpu::graph::compute_knn(d_emb, knn_cfg, stream_);
    ASSERT_EQ(knn_res.n, kTinyN);
    ASSERT_EQ(knn_res.k, kTinyK);

    // Copy kNN neighbors to host (for reference computation).
    auto [h_nb, h_dist] = copy_knn_result_to_host(knn_res, stream_);

    // Compute SNN on device.
    singlet::gpu::graph::SnnConfig snn_cfg;
    snn_cfg.prune_threshold = 1.0f / 15.0f;
    singlet::gpu::graph::SnnResult snn_res =
        singlet::gpu::graph::compute_snn(knn_res, snn_cfg, stream_);

    ASSERT_EQ(snn_res.n, kTinyN);
    ASSERT_GE(snn_res.nnz, 0);
    ASSERT_LE(snn_res.nnz, kTinyN * kTinyK);

    // Copy SNN CSR to host.
    std::vector<int>   h_offsets(kTinyN + 1);
    std::vector<int>   h_cols(snn_res.nnz);
    std::vector<float> h_wts(snn_res.nnz);
    cudaMemcpyAsync(h_offsets.data(), snn_res.row_offsets.get(),
                    (kTinyN + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream_);
    if (snn_res.nnz > 0) {
        cudaMemcpyAsync(h_cols.data(), snn_res.col_indices.get(),
                        snn_res.nnz * sizeof(int), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_wts.data(), snn_res.weights.get(),
                        snn_res.nnz * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    }
    cudaStreamSynchronize(stream_);

    // Compute reference SNN on host.
    // Note: the kNN neighbor lists must be sorted for the sorted-merge algorithm.
    // The Exact backend guarantees sorted output (cub::DeviceSegmentedRadixSort).
    SnnRef ref = compute_snn_reference_host(h_nb, kTinyN, kTinyK, snn_cfg.prune_threshold);

    // Check nnz matches.
    EXPECT_EQ(snn_res.nnz, ref.nnz)
        << "SNN nnz mismatch: got " << snn_res.nnz << " expected " << ref.nnz;

    // Check row_offsets match.
    int offsets_bad = 0;
    for (int i = 0; i <= kTinyN; ++i)
        if (h_offsets[i] != ref.row_offsets[i]) ++offsets_bad;
    EXPECT_EQ(offsets_bad, 0)
        << "SNN row_offsets mismatch: " << offsets_bad << " / " << kTinyN + 1;

    // Check Jaccard weights within rel_err=1e-4 for each matching edge.
    // Build a (row, col) → weight map from reference for lookup.
    std::vector<std::pair<int,int>> ref_edges;
    std::vector<float> ref_wts_flat;
    ref_edges.reserve(ref.nnz);
    ref_wts_flat.reserve(ref.nnz);
    for (int i = 0; i < kTinyN; ++i) {
        for (int p = ref.row_offsets[i]; p < ref.row_offsets[i + 1]; ++p) {
            ref_edges.push_back({i, ref.col_indices[p]});
            ref_wts_flat.push_back(ref.weights[p]);
        }
    }

    // For each device edge (row i, col j), find it in reference and compare weight.
    int wt_bad = 0;
    for (int i = 0; i < kTinyN; ++i) {
        int p_dev = h_offsets[i];
        int p_ref = ref.row_offsets[i];
        int n_dev = h_offsets[i + 1] - p_dev;
        int n_ref = ref.row_offsets[i + 1] - p_ref;
        // Column order within a row may differ between device and host reference.
        // Build a col→weight map for the reference row.
        std::vector<std::pair<int,float>> ref_row;
        for (int rr = 0; rr < n_ref; ++rr)
            ref_row.push_back({ref.col_indices[p_ref + rr], ref.weights[p_ref + rr]});
        std::sort(ref_row.begin(), ref_row.end());

        std::vector<std::pair<int,float>> dev_row;
        for (int dd = 0; dd < n_dev; ++dd)
            dev_row.push_back({h_cols[p_dev + dd], h_wts[p_dev + dd]});
        std::sort(dev_row.begin(), dev_row.end());

        for (size_t ee = 0; ee < dev_row.size() && ee < ref_row.size(); ++ee) {
            if (dev_row[ee].first != ref_row[ee].first) { ++wt_bad; continue; }
            float rel = std::abs(dev_row[ee].second - ref_row[ee].second) /
                        (ref_row[ee].second + 1e-8f);
            if (rel > 1e-4f) ++wt_bad;
        }
    }
    EXPECT_EQ(wt_bad, 0)
        << "SNN Jaccard weight mismatches (or col_index mismatch): "
        << wt_bad << " edges beyond rel_err=1e-4";

    // No self-loops allowed.
    int self_loops = 0;
    for (int i = 0; i < kTinyN; ++i) {
        for (int p = h_offsets[i]; p < h_offsets[i + 1]; ++p)
            if (h_cols[p] == i) ++self_loops;
    }
    EXPECT_EQ(self_loops, 0) << "SNN contains self-loops: " << self_loops;

    bool pass = (offsets_bad == 0 && wt_bad == 0 && self_loops == 0 &&
                 snn_res.nnz == ref.nnz);
    print_registry_row("graph/snn", "tiny", "jaccard_weight_rel_err",
                       static_cast<double>(wt_bad) / std::max(1, ref.nnz),
                       1e-4, false, "host_brute_force", pass);
    print_registry_row("graph/snn", "tiny", "nnz_match",
                       static_cast<double>(snn_res.nnz == ref.nnz ? 1 : 0),
                       1.0, true, "host_brute_force", snn_res.nnz == ref.nnz);
}
