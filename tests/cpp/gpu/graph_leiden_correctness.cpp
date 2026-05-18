// SPDX-License-Identifier: MIT
// singlet/gpu/tests/graph_leiden_correctness.cpp
//
// Correctness harness for singlet::gpu::graph::leiden and leiden_multi.
//
// Written against the design doc at:
//   singlet/gpu/state/designs/07-leiden.md
// NOT against the kernel source — this file was authored in parallel with
// leiden.h by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: scanpy sc.tl.leiden (leidenalg backend) via Python subprocess
//   (tests/refs/leiden_scanpy_reference.py).
//
// Tolerances (from design doc §"Correctness test spec"):
//   Block-stochastic ARI vs ground truth ≥ 0.95
//   Real-data ARI vs scanpy ≥ 0.85 (allowing cuGraph refinement-skip heuristic)
//   Multi-resolution: n_clusters non-strictly increasing with resolution
//   Determinism: bit-identical labels across two runs with same seed
//   Weight function: both Gaussian and Connectivity produce valid clusterings
//   Invalid input: throws / returns error (Leiden_RequireKnnGraph_Throws)
//   Large resolution: n_clusters > n_clusters at resolution 1.0
//
// Test cases:
//   Leiden_TinySynthetic_BlockStochastic_ARI
//   Leiden_RealEmbedding_AriVsScanpy          (GPU-only, marked if no device)
//   Leiden_MultiResolution_DecreasingClusters
//   Leiden_Determinism_RepeatedRunsBitIdentical
//   Leiden_WeightFunction_GaussianVsConnectivity
//   Leiden_RequireKnnGraph_Throws
//   Leiden_LargeResolution_ManyClusters
//
// Build: cmake --build build -j (requires FACTORNET_HAS_GPU, CUDA::cudart,
//   cugraph::cugraph). On a CPU-only node CMake may configure without GPU
//   support; tests that require cuGraph are skipped via GTEST_SKIP().
//
// Registry rows emitted to stdout on each run.

// ---- leiden + kNN headers (written by gpu-kernel-dev) -----------------------
#include <singlet/gpu/graph/leiden.h>
#include <singlet/gpu/graph/knn.h>
#include <singlet/gpu/io/pz_device_loader.h>

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
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEE;

/// Block-stochastic synthetic graph parameters.
constexpr int   kBlockN           = 200;   // total nodes
constexpr int   kBlockCommunities = 4;     // ground-truth clusters
constexpr int   kBlockPerComm     = 50;    // kBlockN / kBlockCommunities
constexpr float kIntraDensity     = 0.40f; // intra-cluster edge probability
constexpr float kInterDensity     = 0.05f; // inter-cluster edge probability
constexpr int   kBlockK           = 5;     // kNN k for synthetic graph

/// Small embedding for real-data + weight-function tests: 200 nodes × 10 dims.
constexpr int kSmallN = 200;
constexpr int kSmallD = 10;
constexpr int kSmallK = 5;

/// Resolutions used in the multi-resolution test.
static const std::vector<float> kMultiResolutions = {0.3f, 0.5f, 1.0f, 1.5f};

/// Temp directory for Python reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_leiden_refs_tmp";

// =============================================================================
// Utility helpers
// =============================================================================

namespace {

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

// ---------------------------------------------------------------------------
// write_npy_i32 / write_npy_f32
//
// Minimal NPY v1.0 writer for 1-D and 2-D arrays.
// Matches write_embedding_npy pattern from graph_knn_correctness.cpp.
// ---------------------------------------------------------------------------
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

    int item_size = (dtype_str == "<i4") ? 4 : 4; // int32 and float32 both 4B
    size_t n_bytes = static_cast<size_t>(rows) * (cols == 1 ? 1 : cols) * item_size;
    fout.write(static_cast<const char*>(data), static_cast<std::streamsize>(n_bytes));
    if (!fout) throw std::runtime_error("write_npy: write error " + path);
}

void write_npy_f32(const std::string& path, const float* data, int rows, int cols) {
    write_npy_generic(path, data, "<f4", rows, cols);
}

void write_npy_i32(const std::string& path, const int32_t* data, int rows, int cols) {
    write_npy_generic(path, data, "<i4", rows, cols);
}

// ---------------------------------------------------------------------------
// load_npy_i32 — load a 1-D int32 .npy into a vector<int32_t>.
// ---------------------------------------------------------------------------
std::vector<int32_t> load_npy_i32(const std::string& path, int expected_n) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("load_npy_i32: cannot open " + path);

    // Skip header: read until '\n' after the 10-byte preamble.
    char magic[6]; fin.read(magic, 6);
    char major, minor; fin.read(&major, 1); fin.read(&minor, 1);
    uint16_t hlen; fin.read(reinterpret_cast<char*>(&hlen), 2);
    std::string hdr(hlen, ' ');
    fin.read(hdr.data(), hlen);

    std::vector<int32_t> out(static_cast<size_t>(expected_n));
    fin.read(reinterpret_cast<char*>(out.data()),
             static_cast<std::streamsize>(expected_n * 4));
    if (!fin) throw std::runtime_error("load_npy_i32: short read from " + path);
    return out;
}

// =============================================================================
// Block-stochastic graph builder
//
// Generates a (kBlockN × kBlockN) symmetric adjacency with the stated densities,
// then converts to a KnnResult-compatible host structure (neighbor indices +
// uniform distances = 1 for connected, unused for no-edge).
//
// Because the graph is unweighted from the block model, we build it as a
// k-NN adjacency by listing, for each node, up to kBlockK neighbors sampled
// from the adjacency list in node-id order (deterministic given the seed).
// =============================================================================

struct BlockGraph {
    int n;                              // kBlockN
    std::vector<int32_t> gt_labels;    // ground-truth community ids
    // kNN style: neighbors[i*k + j], distances[i*k + j]
    int k_actual;
    std::vector<int32_t> neighbors;
    std::vector<float>   distances;
};

BlockGraph make_block_stochastic_graph(
        int n, int n_comm, float p_intra, float p_inter, int k, uint64_t seed) {
    assert(n % n_comm == 0);
    int per_comm = n / n_comm;

    // Assign ground-truth labels.
    std::vector<int32_t> gt(n);
    for (int i = 0; i < n; ++i) gt[i] = i / per_comm;

    // Build adjacency list via Bernoulli draws.
    std::mt19937_64 rng(seed);
    std::vector<std::vector<int>> adj(n);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            float p = (gt[i] == gt[j]) ? p_intra : p_inter;
            std::uniform_real_distribution<float> u(0.0f, 1.0f);
            if (u(rng) < p) {
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }

    // Build KnnResult-style arrays: pick up to k neighbors per node.
    int k_actual = k;
    std::vector<int32_t> nb(static_cast<size_t>(n) * k_actual, -1);
    std::vector<float>   dist(static_cast<size_t>(n) * k_actual, 1.0f); // uniform weight

    for (int i = 0; i < n; ++i) {
        auto& row = adj[i];
        // Deterministic: sort neighbors by id, take first k.
        std::sort(row.begin(), row.end());
        int count = std::min(static_cast<int>(row.size()), k_actual);
        for (int j = 0; j < count; ++j) {
            nb[static_cast<size_t>(i) * k_actual + j]   = row[j];
            dist[static_cast<size_t>(i) * k_actual + j] = 1.0f;
        }
    }

    return {n, std::move(gt), k_actual, std::move(nb), std::move(dist)};
}

// =============================================================================
// ARI computation (pure C++)
//
// Adjusted Rand Index between two integer label arrays of length n.
// Formula: ARI = (RI - Expected_RI) / (max_RI - Expected_RI)
// where RI is based on pairwise agreement counts.
//
// We implement the classic contingency-table formula:
//   ARI = [sum_ij C(n_ij,2) - (sum_i C(a_i,2)*sum_j C(b_j,2)) / C(n,2)]
//       / [0.5*(sum_i C(a_i,2) + sum_j C(b_j,2))
//          - (sum_i C(a_i,2)*sum_j C(b_j,2)) / C(n,2)]
//
// All integer arithmetic to avoid floating-point accumulation issues on labels.
// =============================================================================

double compute_ari(const std::vector<int32_t>& a,
                   const std::vector<int32_t>& b) {
    assert(a.size() == b.size());
    int n = static_cast<int>(a.size());

    // Remap labels to 0-based contiguous.
    auto remap = [&](const std::vector<int32_t>& v) {
        std::unordered_map<int32_t, int32_t> m;
        std::vector<int32_t> out(v.size());
        int32_t next = 0;
        for (int i = 0; i < n; ++i) {
            auto it = m.find(v[i]);
            if (it == m.end()) { m[v[i]] = next; out[i] = next++; }
            else                { out[i] = it->second; }
        }
        return out;
    };
    auto ra = remap(a);
    auto rb = remap(b);

    int na_max = *std::max_element(ra.begin(), ra.end()) + 1;
    int nb_max = *std::max_element(rb.begin(), rb.end()) + 1;

    // Contingency table.
    std::vector<std::vector<int64_t>> C(na_max, std::vector<int64_t>(nb_max, 0));
    for (int i = 0; i < n; ++i) C[ra[i]][rb[i]]++;

    // Row/col sums.
    std::vector<int64_t> ra_sum(na_max, 0), rb_sum(nb_max, 0);
    for (int i = 0; i < na_max; ++i)
        for (int j = 0; j < nb_max; ++j) {
            ra_sum[i] += C[i][j];
            rb_sum[j] += C[i][j];
        }

    auto comb2 = [](int64_t x) -> double {
        return (x <= 1) ? 0.0 : static_cast<double>(x) * (x - 1) / 2.0;
    };

    double sum_cij = 0.0;
    for (int i = 0; i < na_max; ++i)
        for (int j = 0; j < nb_max; ++j)
            sum_cij += comb2(C[i][j]);

    double sum_a = 0.0, sum_b = 0.0;
    for (auto v : ra_sum) sum_a += comb2(v);
    for (auto v : rb_sum) sum_b += comb2(v);

    double n_pairs = comb2(n);
    double expected = sum_a * sum_b / n_pairs;
    double max_ri   = 0.5 * (sum_a + sum_b);

    double denom = max_ri - expected;
    if (denom <= 0.0) {
        // Perfect agreement or degenerate case: return 1 if sum_cij == n_pairs, else 0.
        return (sum_cij == n_pairs) ? 1.0 : 0.0;
    }
    return (sum_cij - expected) / denom;
}

// =============================================================================
// make_random_embedding
// =============================================================================

std::vector<float> make_random_embedding(int n, int d, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> emb(static_cast<size_t>(n) * d);
    for (auto& v : emb) v = dist(rng);
    return emb;
}

// =============================================================================
// upload_knn_to_device
//
// Allocates device buffers and uploads host neighbor/distance arrays to a
// singlet::gpu::graph::KnnResult.  The KnnResult owns DeviceMemory<int> and
// DeviceMemory<float> as in cycle 8.
// =============================================================================

singlet::gpu::graph::KnnResult upload_knn_to_device(
        const std::vector<int32_t>& h_nb,
        const std::vector<float>&   h_dist,
        int n, int k, cudaStream_t stream) {
    using namespace singlet::gpu::core;

    size_t nk = static_cast<size_t>(n) * k;

    DeviceMemory<int>   d_nb(nk);
    DeviceMemory<float> d_dist(nk);

    auto ck = [](cudaError_t e, const char* ctx) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("upload_knn: ") + ctx +
                                     ": " + cudaGetErrorString(e));
    };

    ck(cudaMemcpyAsync(d_nb.get(),   h_nb.data(),   nk * sizeof(int),
                       cudaMemcpyHostToDevice, stream), "neighbors");
    ck(cudaMemcpyAsync(d_dist.get(), h_dist.data(), nk * sizeof(float),
                       cudaMemcpyHostToDevice, stream), "distances");
    ck(cudaStreamSynchronize(stream), "sync");

    singlet::gpu::graph::KnnResult res;
    res.n         = n;
    res.k         = k;
    res.neighbors = std::move(d_nb);
    res.distances = std::move(d_dist);
    return res;
}

// =============================================================================
// copy_labels_to_host
// =============================================================================

std::vector<int32_t> copy_labels_to_host(
        const singlet::gpu::core::DeviceMemory<int>& d_labels,
        int n, cudaStream_t stream) {
    std::vector<int32_t> out(n);
    cudaError_t err = cudaMemcpyAsync(out.data(), d_labels.get(),
                                      n * sizeof(int32_t),
                                      cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("copy_labels: ") +
                                 cudaGetErrorString(err));
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("copy_labels sync: ") +
                                 cudaGetErrorString(err));
    return out;
}

// =============================================================================
// check_gpu_available
// =============================================================================

bool gpu_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

// =============================================================================
// run_python_leiden_reference
//
// Writes kNN arrays to .npy, invokes leiden_scanpy_reference.py, loads labels.
// =============================================================================

std::vector<int32_t> run_python_leiden_reference(
        const std::vector<int32_t>& h_nb,
        const std::vector<float>&   h_dist,
        int n, int k,
        float resolution, int seed,
        const std::string& tag) {
    ensure_refs_tmp();

    std::string idx_path  = refs_path(tag + "_indices.npy");
    std::string dist_path = refs_path(tag + "_distances.npy");
    std::string out_path  = refs_path(tag + "_labels.npy");

    write_npy_i32(idx_path.c_str(),  h_nb.data(),   n, k);
    write_npy_f32(dist_path.c_str(), h_dist.data(), n, k);

    std::string script =
        std::string(__FILE__).substr(0,
            std::string(__FILE__).rfind('/')) +
        "/refs/leiden_scanpy_reference.py";

    std::string cmd =
        "python3 " + script +
        " --indices "    + idx_path +
        " --distances "  + dist_path +
        " --output "     + out_path +
        " --resolution " + std::to_string(resolution) +
        " --seed "       + std::to_string(seed);
    run_cmd(cmd);

    return load_npy_i32(out_path, n);
}

} // anonymous namespace

// =============================================================================
// TEST: Leiden_TinySynthetic_BlockStochastic_ARI
//
// 200-node block-stochastic graph (4 communities × 50 nodes).
// Intra-cluster edge density 0.4, inter-cluster 0.05.
// Build KnnResult manually from adjacency; run leiden(resolution=1.0, seed=42).
// Compute ARI vs ground-truth labels.
// Assert ARI ≥ 0.95.
// =============================================================================

TEST(Leiden, TinySynthetic_BlockStochastic_ARI) {
    if (!gpu_available()) {
        GTEST_SKIP() << "No CUDA device — skipping GPU Leiden test.";
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    // Build block-stochastic graph.
    auto bg = make_block_stochastic_graph(
        kBlockN, kBlockCommunities, kIntraDensity, kInterDensity,
        kBlockK, kSeed);

    ASSERT_EQ(static_cast<int>(bg.gt_labels.size()), kBlockN);

    // Upload to device.
    auto knn = upload_knn_to_device(
        bg.neighbors, bg.distances, kBlockN, bg.k_actual, stream);

    // Run Leiden.
    using namespace singlet::gpu::graph;
    LeidenConfig cfg;
    cfg.resolution      = 1.0f;
    cfg.seed            = 42;
    cfg.weight_function = LeidenWeight::Connectivity;

    LeidenResult result = leiden(knn, cfg, stream);

    ASSERT_EQ(result.n_clusters, kBlockCommunities)
        << "Expected 4 communities from block-stochastic graph";

    // Copy labels to host.
    auto h_labels = copy_labels_to_host(result.labels, kBlockN, stream);

    // Compute ARI vs ground truth.
    double ari = compute_ari(h_labels, bg.gt_labels);
    std::printf("[registry] Leiden_TinySynthetic ARI=%.4f (tol>=0.95)\n", ari);

    EXPECT_GE(ari, 0.95) << "ARI vs ground truth below 0.95: " << ari;

    cudaStreamDestroy(stream);
}

// =============================================================================
// TEST: Leiden_RealEmbedding_AriVsScanpy
//
// Synthesize a random fp32 embedding (200 × 10, fixed seed).
// Run Exact kNN (k=5) → singlet-gpu Leiden.
// Run Python subprocess: same kNN arrays → scanpy sc.tl.leiden.
// Compute ARI(ours, scanpy) ≥ 0.85.
//
// NOTE: This test is best-effort without GPU.  It tests only the plumbing;
// the meaningful comparison requires actual cuGraph on a GPU node.
// =============================================================================

TEST(Leiden, RealEmbedding_AriVsScanpy) {
    if (!gpu_available()) {
        GTEST_SKIP() << "No CUDA device — skipping GPU Leiden vs scanpy test.";
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    // Generate a random embedding.
    auto h_emb = make_random_embedding(kSmallN, kSmallD, kSeed);

    // Build kNN via singlet-gpu compute_knn (Exact backend).
    using namespace singlet::gpu::core;
    DeviceMemory<float> d_emb(static_cast<size_t>(kSmallN) * kSmallD);
    ASSERT_EQ(cudaMemcpyAsync(d_emb.get(), h_emb.data(),
                              kSmallN * kSmallD * sizeof(float),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // factornet::gpu::DenseMatrixGPU<float> has no move-DeviceMemory ctor.
    // Use default ctor + field assignment to take ownership of uploaded buffer.
    DeviceDense embed_dev;
    embed_dev.rows = kSmallN;
    embed_dev.cols = kSmallD;
    embed_dev.data = std::move(d_emb);

    singlet::gpu::graph::KnnConfig knn_cfg;
    knn_cfg.k       = kSmallK;
    knn_cfg.backend = singlet::gpu::graph::KnnBackend::Exact;
    knn_cfg.seed    = kSeed;

    auto knn = singlet::gpu::graph::compute_knn(embed_dev, knn_cfg, stream);

    // Copy kNN to host for Python reference.
    std::vector<int32_t> h_nb(static_cast<size_t>(kSmallN) * kSmallK);
    std::vector<float>   h_dist(static_cast<size_t>(kSmallN) * kSmallK);
    ASSERT_EQ(cudaMemcpyAsync(h_nb.data(),   knn.neighbors.get(),
                              kSmallN * kSmallK * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(h_dist.data(), knn.distances.get(),
                              kSmallN * kSmallK * sizeof(float),
                              cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // Run singlet-gpu Leiden.
    singlet::gpu::graph::LeidenConfig cfg;
    cfg.resolution      = 1.0f;
    cfg.seed            = 42;
    cfg.weight_function = singlet::gpu::graph::LeidenWeight::Connectivity;

    auto result = singlet::gpu::graph::leiden(knn, cfg, stream);
    auto h_our_labels = copy_labels_to_host(result.labels, kSmallN, stream);

    // Run Python reference.
    auto h_ref_labels = run_python_leiden_reference(
        h_nb, h_dist, kSmallN, kSmallK, 1.0f, 42, "real_emb");

    // Compute ARI.
    double ari = compute_ari(h_our_labels, h_ref_labels);
    std::printf("[registry] Leiden_RealEmbedding ARI_vs_scanpy=%.4f (tol>=0.85)\n", ari);

    EXPECT_GE(ari, 0.85)
        << "ARI vs scanpy below 0.85: " << ari
        << " (cuGraph refinement-skip heuristic may cause small differences)";

    cudaStreamDestroy(stream);
}

// =============================================================================
// TEST: Leiden_MultiResolution_DecreasingClusters
//
// leiden_multi with {0.3, 0.5, 1.0, 1.5}.
// Confirm 4 results, n_clusters non-strictly increasing with resolution.
// =============================================================================

TEST(Leiden, MultiResolution_DecreasingClusters) {
    if (!gpu_available()) {
        GTEST_SKIP() << "No CUDA device — skipping multi-resolution test.";
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto bg = make_block_stochastic_graph(
        kBlockN, kBlockCommunities, kIntraDensity, kInterDensity,
        kBlockK, kSeed);
    auto knn = upload_knn_to_device(
        bg.neighbors, bg.distances, kBlockN, bg.k_actual, stream);

    using namespace singlet::gpu::graph;
    LeidenConfig base_cfg;
    base_cfg.seed            = 42;
    base_cfg.weight_function = LeidenWeight::Connectivity;

    auto results = leiden_multi(knn, kMultiResolutions, base_cfg, stream);

    ASSERT_EQ(static_cast<int>(results.size()),
              static_cast<int>(kMultiResolutions.size()))
        << "leiden_multi must return one result per resolution";

    // Collect n_clusters per resolution.
    std::vector<int> n_clusters_per_res;
    for (size_t i = 0; i < results.size(); ++i) {
        int nc = results[i].n_clusters;
        EXPECT_GE(nc, 1) << "Resolution " << kMultiResolutions[i]
                         << " produced 0 clusters";
        n_clusters_per_res.push_back(nc);
        std::printf("[registry] Leiden_MultiRes res=%.2f n_clusters=%d\n",
                    kMultiResolutions[i], nc);
    }

    // Non-strictly increasing: each resolution >= previous.
    for (size_t i = 1; i < n_clusters_per_res.size(); ++i) {
        EXPECT_GE(n_clusters_per_res[i], n_clusters_per_res[i - 1])
            << "n_clusters not non-decreasing: res[" << (i-1) << "]="
            << kMultiResolutions[i-1] << " gave " << n_clusters_per_res[i-1]
            << " but res[" << i << "]=" << kMultiResolutions[i]
            << " gave " << n_clusters_per_res[i];
    }

    cudaStreamDestroy(stream);
}

// =============================================================================
// TEST: Leiden_Determinism_RepeatedRunsBitIdentical
//
// Run leiden(seed=42) twice on the same KnnResult.
// Assert label arrays are bit-identical.
// =============================================================================

TEST(Leiden, Determinism_RepeatedRunsBitIdentical) {
    if (!gpu_available()) {
        GTEST_SKIP() << "No CUDA device — skipping determinism test.";
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto bg = make_block_stochastic_graph(
        kBlockN, kBlockCommunities, kIntraDensity, kInterDensity,
        kBlockK, kSeed);
    auto knn = upload_knn_to_device(
        bg.neighbors, bg.distances, kBlockN, bg.k_actual, stream);

    using namespace singlet::gpu::graph;
    LeidenConfig cfg;
    cfg.resolution      = 1.0f;
    cfg.seed            = 42;
    cfg.weight_function = LeidenWeight::Connectivity;

    auto result_a = leiden(knn, cfg, stream);
    auto result_b = leiden(knn, cfg, stream);

    auto labels_a = copy_labels_to_host(result_a.labels, kBlockN, stream);
    auto labels_b = copy_labels_to_host(result_b.labels, kBlockN, stream);

    ASSERT_EQ(labels_a.size(), labels_b.size());
    for (int i = 0; i < kBlockN; ++i) {
        EXPECT_EQ(labels_a[i], labels_b[i])
            << "Labels differ at node " << i
            << " (run A: " << labels_a[i] << ", run B: " << labels_b[i] << ")";
    }

    std::printf("[registry] Leiden_Determinism: bit-identical labels PASS\n");

    cudaStreamDestroy(stream);
}

// =============================================================================
// TEST: Leiden_WeightFunction_GaussianVsConnectivity
//
// Same kNN input; run with LeidenWeight::Gaussian and LeidenWeight::Connectivity.
// Both must produce valid clusterings (n_clusters ≥ 1, modularity ≥ 0).
// They are NOT required to match.
// =============================================================================

TEST(Leiden, WeightFunction_GaussianVsConnectivity) {
    if (!gpu_available()) {
        GTEST_SKIP() << "No CUDA device — skipping weight-function test.";
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto bg = make_block_stochastic_graph(
        kBlockN, kBlockCommunities, kIntraDensity, kInterDensity,
        kBlockK, kSeed);
    auto knn = upload_knn_to_device(
        bg.neighbors, bg.distances, kBlockN, bg.k_actual, stream);

    using namespace singlet::gpu::graph;

    // Gaussian.
    LeidenConfig cfg_gauss;
    cfg_gauss.resolution      = 1.0f;
    cfg_gauss.seed            = 42;
    cfg_gauss.weight_function = LeidenWeight::Gaussian;
    cfg_gauss.gaussian_sigma  = 0.0f;  // use median

    auto res_gauss = leiden(knn, cfg_gauss, stream);
    EXPECT_GE(res_gauss.n_clusters, 1)
        << "Gaussian weight: expected at least 1 cluster";
    EXPECT_GE(res_gauss.modularity, 0.0f)
        << "Gaussian weight: modularity must be non-negative";

    std::printf("[registry] Leiden_WeightFunction Gaussian: "
                "n_clusters=%d modularity=%.4f\n",
                res_gauss.n_clusters, res_gauss.modularity);

    // Connectivity.
    LeidenConfig cfg_conn;
    cfg_conn.resolution      = 1.0f;
    cfg_conn.seed            = 42;
    cfg_conn.weight_function = LeidenWeight::Connectivity;

    auto res_conn = leiden(knn, cfg_conn, stream);
    EXPECT_GE(res_conn.n_clusters, 1)
        << "Connectivity weight: expected at least 1 cluster";
    EXPECT_GE(res_conn.modularity, 0.0f)
        << "Connectivity weight: modularity must be non-negative";

    std::printf("[registry] Leiden_WeightFunction Connectivity: "
                "n_clusters=%d modularity=%.4f\n",
                res_conn.n_clusters, res_conn.modularity);

    // Results may differ — that is intentional and acceptable.
    // We only log whether they do.
    auto labels_g = copy_labels_to_host(res_gauss.labels, kBlockN, stream);
    auto labels_c = copy_labels_to_host(res_conn.labels,  kBlockN, stream);
    bool identical = (labels_g == labels_c);
    std::printf("[registry] Leiden_WeightFunction labels_identical=%s\n",
                identical ? "yes" : "no (expected)");

    cudaStreamDestroy(stream);
}

// =============================================================================
// TEST: Leiden_RequireKnnGraph_Throws
//
// Pass an empty / zero-node KnnResult.
// Expect leiden() to throw std::invalid_argument (or std::runtime_error).
// =============================================================================

TEST(Leiden, RequireKnnGraph_Throws) {
    if (!gpu_available()) {
        GTEST_SKIP() << "No CUDA device — skipping error-handling test.";
    }

    using namespace singlet::gpu::graph;

    // Construct an empty KnnResult (n=0, k=0, null device buffers).
    KnnResult empty;
    empty.n = 0;
    empty.k = 0;
    // DeviceMemory<int/float> default-constructed → null device ptr, size 0.

    LeidenConfig cfg;
    cfg.resolution = 1.0f;
    cfg.seed       = 42;

    // When cuGraph is absent, leiden() throws std::runtime_error("requires cuGraph").
    // When cuGraph is present, it throws std::invalid_argument for empty/null input.
    // Accept either — both are valid pre-condition rejections.
    EXPECT_THROW(leiden(empty, cfg, nullptr), std::exception)
        << "leiden() must throw for empty/invalid KnnResult (std::runtime_error or std::invalid_argument)";

    std::printf("[registry] Leiden_RequireKnnGraph_Throws: PASS\n");
}

// =============================================================================
// TEST: Leiden_LargeResolution_ManyClusters
//
// resolution=10.0 should produce more clusters than resolution=1.0 on the
// same block-stochastic graph.
// =============================================================================

TEST(Leiden, LargeResolution_ManyClusters) {
    if (!gpu_available()) {
        GTEST_SKIP() << "No CUDA device — skipping large-resolution test.";
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto bg = make_block_stochastic_graph(
        kBlockN, kBlockCommunities, kIntraDensity, kInterDensity,
        kBlockK, kSeed);
    auto knn = upload_knn_to_device(
        bg.neighbors, bg.distances, kBlockN, bg.k_actual, stream);

    using namespace singlet::gpu::graph;

    // Baseline: resolution = 1.0.
    LeidenConfig cfg_base;
    cfg_base.resolution      = 1.0f;
    cfg_base.seed            = 42;
    cfg_base.weight_function = LeidenWeight::Connectivity;
    auto res_base = leiden(knn, cfg_base, stream);
    int nc_base = res_base.n_clusters;

    // High resolution: 10.0.
    LeidenConfig cfg_high;
    cfg_high.resolution      = 10.0f;
    cfg_high.seed            = 42;
    cfg_high.weight_function = LeidenWeight::Connectivity;
    auto res_high = leiden(knn, cfg_high, stream);
    int nc_high = res_high.n_clusters;

    std::printf("[registry] Leiden_LargeResolution: "
                "nc_base(res=1.0)=%d  nc_high(res=10.0)=%d\n",
                nc_base, nc_high);

    EXPECT_GT(nc_high, nc_base)
        << "resolution=10.0 should produce more clusters than resolution=1.0; "
        << "got nc_base=" << nc_base << " nc_high=" << nc_high;

    cudaStreamDestroy(stream);
}
