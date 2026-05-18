// SPDX-License-Identifier: MIT
// singlet/gpu/tests/integrate_bbknn_correctness.cpp
//
// Correctness harness for singlet::gpu::integrate::bbknn.
//
// Written against the design doc at:
//   singlet/gpu/state/designs/12-integration.md
// NOT against the kernel source — authored in parallel with integrate/bbknn.h
// by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference:
//   tests/refs/bbknn_python_reference.py — runs bbknn Python, dumps neighbor graph
//
// Tolerances (design doc §"Correctness test spec"):
//   Neighbor balance:  each cell has k_within ± 1 neighbors per batch
//   BBKNN vs Python:   neighbor Jaccard per cell >= 0.70 on avg (set intersection,
//                      both rank-k lists from the same embedding)
//   Leiden ARI:        >= 0.80 vs unintegrated Leiden on 2-batch data
//
// Test cases:
//   BBKNN_TwoBatch_NeighborBalance
//   BBKNN_VsPython
//   BBKNN_LeidenDownstream
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, singlet-gpu::singlet-gpu, GTest.
//   Every test that requires a GPU device is guarded by gpu_available() and
//   calls GTEST_SKIP() on CPU-only nodes.
//   Leiden downstream test also requires SINGLET_GPU_HAS_CUGRAPH (optional).

// ---- singlet-gpu integration + graph headers (written by gpu-kernel-dev) ---
#include <singlet/gpu/integrate/bbknn.h>
#include <singlet/gpu/graph/knn.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

// For Leiden downstream test (cycle 9).
#ifdef SINGLET_GPU_HAS_CUGRAPH
#include <singlet/gpu/graph/leiden.h>
#endif

// ---- test helpers ------------------------------------------------------------
#include "refs/dump_csc.h"

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
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

/// Tiny synthetic embedding: 400 cells × 20 PCs, 2 balanced batches.
constexpr int kTinyN  = 400;
constexpr int kTinyD  = 20;
constexpr int kTinyB  = 2;
/// Neighbors per batch (k_within) for BBKNN.
constexpr int kKWithin = 3;
/// Total k = kKWithin * n_batches.
constexpr int kKTotal  = kKWithin * kTinyB;

/// Medium: 2000 cells × 50 PCs, 5 batches — for Leiden downstream test.
constexpr int kMedN = 2000;
constexpr int kMedD = 50;
constexpr int kMedB = 5;

/// Number of planted cell types.
constexpr int kNCellTypes = 6;

/// Tolerances.
/// Each cell must have kKWithin ± 1 neighbors in EACH batch.
constexpr int kBalanceTolerance = 1;
/// Minimum average Jaccard similarity vs Python BBKNN.
constexpr double kJaccardMin = 0.70;
/// Minimum Leiden ARI vs unintegrated baseline.
constexpr double kLeidenAriMin = 0.80;

/// Python interpreter.
static const char* kPython = "python3";

/// Path to Python bbknn reference script.
static const char* kBbknnRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet/gpu/"
    "tests/refs/bbknn_python_reference.py";

/// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_bbknn_refs_tmp";

// =============================================================================
// Utility helpers (pattern from cycles 8–13)
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
// Minimal .npy v1.0 writer — float32 2-D row-major.
// ---------------------------------------------------------------------------
static void write_npy_f32_2d(const std::string& path,
                              const float* data, int rows, int cols)
{
    std::string header_str =
        std::string("{'descr': '<f4', 'fortran_order': False, 'shape': (") +
        std::to_string(rows) + ", " + std::to_string(cols) + "), }";
    constexpr int kPreamble = 10;
    int raw_len   = static_cast<int>(header_str.size());
    int pad_to_64 = ((raw_len + kPreamble + 63) / 64) * 64 - kPreamble;
    while (static_cast<int>(header_str.size()) < pad_to_64 - 1)
        header_str += ' ';
    header_str += '\n';

    std::ofstream fout(path, std::ios::binary | std::ios::trunc);
    if (!fout) throw std::runtime_error("write_npy_f32_2d: cannot open " + path);
    const char magic[] = "\x93NUMPY\x01\x00";
    fout.write(magic, 8);
    uint16_t hlen = static_cast<uint16_t>(header_str.size());
    fout.write(reinterpret_cast<const char*>(&hlen), 2);
    fout.write(header_str.c_str(), static_cast<std::streamsize>(header_str.size()));
    fout.write(reinterpret_cast<const char*>(data),
               static_cast<std::streamsize>(
                   static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float)));
    if (!fout) throw std::runtime_error("write_npy_f32_2d: write error " + path);
}

// ---------------------------------------------------------------------------
// Minimal .npy v1.0 writer — int32 1-D.
// ---------------------------------------------------------------------------
static void write_npy_i32_1d(const std::string& path,
                              const int* data, int n)
{
    std::string header_str =
        std::string("{'descr': '<i4', 'fortran_order': False, 'shape': (") +
        std::to_string(n) + ",), }";
    constexpr int kPreamble = 10;
    int raw_len   = static_cast<int>(header_str.size());
    int pad_to_64 = ((raw_len + kPreamble + 63) / 64) * 64 - kPreamble;
    while (static_cast<int>(header_str.size()) < pad_to_64 - 1)
        header_str += ' ';
    header_str += '\n';

    std::ofstream fout(path, std::ios::binary | std::ios::trunc);
    if (!fout) throw std::runtime_error("write_npy_i32_1d: cannot open " + path);
    const char magic[] = "\x93NUMPY\x01\x00";
    fout.write(magic, 8);
    uint16_t hlen = static_cast<uint16_t>(header_str.size());
    fout.write(reinterpret_cast<const char*>(&hlen), 2);
    fout.write(header_str.c_str(), static_cast<std::streamsize>(header_str.size()));
    fout.write(reinterpret_cast<const char*>(data),
               static_cast<std::streamsize>(static_cast<size_t>(n) * sizeof(int32_t)));
    if (!fout) throw std::runtime_error("write_npy_i32_1d: write error " + path);
}

// ---------------------------------------------------------------------------
// Read int32 2-D array from .npy v1.0 file.
// ---------------------------------------------------------------------------
static std::vector<int32_t> read_npy_i32_2d(const std::string& path,
                                              int& rows_out, int& cols_out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("read_npy_i32_2d: cannot open " + path);

    char preamble[10];
    f.read(preamble, 10);
    if (preamble[0] != '\x93' || std::string(preamble + 1, 5) != "NUMPY")
        throw std::runtime_error("read_npy_i32_2d: not a .npy file: " + path);

    uint16_t hlen;
    f.read(reinterpret_cast<char*>(&hlen), 2);
    std::string header(hlen, ' ');
    f.read(header.data(), hlen);

    auto shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos)
        throw std::runtime_error("read_npy_i32_2d: no 'shape' in header");
    auto lp = header.find('(', shape_pos);
    auto rp = header.find(')', lp);
    std::string shape_str = header.substr(lp + 1, rp - lp - 1);
    std::istringstream ss(shape_str);
    std::string tok;
    std::getline(ss, tok, ',');
    rows_out = std::stoi(tok);
    std::getline(ss, tok, ',');
    cols_out = std::stoi(tok);

    size_t nel = static_cast<size_t>(rows_out) * static_cast<size_t>(cols_out);
    std::vector<int32_t> data(nel);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(nel * sizeof(int32_t)));
    if (!f) throw std::runtime_error("read_npy_i32_2d: short read: " + path);
    return data;
}

// ---------------------------------------------------------------------------
// Read .npz file produced by numpy.savez. Parses the ZIP to extract the
// 'indices' and 'distances' arrays.
// Very minimal — only reads .npy members named exactly as given.
// ---------------------------------------------------------------------------
// We cheat: Python npz is a ZIP, which we parse manually.
// Each member is a .npy file stored under the name "<key>.npy".
static std::vector<int32_t> read_npz_i32(const std::string& npz_path,
                                          const std::string& key,
                                          int& rows_out, int& cols_out)
{
    // We use Python to extract just the indices array to a temp .npy file.
    std::string tmp_npy = npz_path + "_" + key + ".npy";
    std::string cmd =
        std::string(kPython) +
        " -c \"import numpy as np; d=np.load('" + npz_path + "'); "
        "np.save('" + tmp_npy + "', d['" + key + "'])\"";
    run_cmd(cmd);

    std::ifstream f(tmp_npy, std::ios::binary);
    if (!f) throw std::runtime_error("read_npz_i32: cannot open extracted: " + tmp_npy);
    char preamble[10];
    f.read(preamble, 10);
    uint16_t hlen;
    f.read(reinterpret_cast<char*>(&hlen), 2);
    std::string header(hlen, ' ');
    f.read(header.data(), hlen);

    auto shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos)
        throw std::runtime_error("read_npz_i32: no shape in " + tmp_npy);
    auto lp = header.find('(', shape_pos);
    auto rp = header.find(')', lp);
    std::string shape_str = header.substr(lp + 1, rp - lp - 1);
    std::istringstream ss(shape_str);
    std::string tok;
    std::getline(ss, tok, ',');
    rows_out = std::stoi(tok);
    std::getline(ss, tok, ',');
    cols_out = std::stoi(tok);

    size_t nel = static_cast<size_t>(rows_out) * static_cast<size_t>(cols_out);
    std::vector<int32_t> data(nel);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(nel * sizeof(int32_t)));
    if (!f) throw std::runtime_error("read_npz_i32: short read: " + tmp_npy);
    fs::remove(tmp_npy);
    return data;
}

// ---------------------------------------------------------------------------
// Synthetic embedding generator (same logic as harmony harness).
// ---------------------------------------------------------------------------
struct SyntheticEmbedding {
    int n_cells, n_dims, n_batches, n_cell_types;
    std::vector<float> embedding;
    std::vector<int>   batch_labels;
    std::vector<int>   ct_labels;
};

SyntheticEmbedding make_synthetic(int n_cells, int n_dims,
                                  int n_batches, int n_cell_types,
                                  uint64_t seed,
                                  float biology_scale = 3.0f,
                                  float batch_scale   = 2.0f,
                                  float noise_scale   = 0.3f)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> ndist(0.0f, 1.0f);

    std::vector<std::vector<float>> bio_centroids(n_cell_types,
                                                  std::vector<float>(n_dims, 0.0f));
    for (int c = 0; c < n_cell_types; ++c) {
        int axis = c % n_dims;
        bio_centroids[c][axis] = biology_scale * static_cast<float>(c + 1);
    }

    std::vector<std::vector<float>> batch_offsets(n_batches,
                                                  std::vector<float>(n_dims, 0.0f));
    for (int b = 0; b < n_batches; ++b) {
        int axis = (n_dims - 1 - b % n_dims);
        batch_offsets[b][axis] = batch_scale * static_cast<float>(b + 1);
    }

    SyntheticEmbedding out;
    out.n_cells      = n_cells;
    out.n_dims       = n_dims;
    out.n_batches    = n_batches;
    out.n_cell_types = n_cell_types;
    out.embedding.resize(static_cast<size_t>(n_cells) * n_dims);
    out.batch_labels.resize(n_cells);
    out.ct_labels.resize(n_cells);

    for (int i = 0; i < n_cells; ++i) {
        int b  = i % n_batches;
        int ct = (i / n_batches) % n_cell_types;
        out.batch_labels[i] = b;
        out.ct_labels[i]    = ct;
        float* row = out.embedding.data() + static_cast<size_t>(i) * n_dims;
        for (int d = 0; d < n_dims; ++d)
            row[d] = bio_centroids[ct][d] + batch_offsets[b][d] + noise_scale * ndist(rng);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Run BBKNN on device; return KnnResult on host.
// Returns (neighbor_indices[N * k_total], n_neighbors_per_cell).
// ---------------------------------------------------------------------------
static singlet::gpu::graph::KnnResult run_bbknn_device(
    const std::vector<float>& h_embedding,
    const std::vector<int>&   h_batch,
    int N, int d, int n_batches,
    int k_within)
{
    using namespace singlet::gpu;
    using namespace singlet::gpu::integrate;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    core::DeviceDense emb_dev;
    emb_dev.rows = static_cast<uint32_t>(N);
    emb_dev.cols = static_cast<uint32_t>(d);
    emb_dev.data = core::DeviceMemory<float>(static_cast<size_t>(N) * d);
    cudaMemcpyAsync(emb_dev.data.get(), h_embedding.data(),
                    static_cast<size_t>(N) * d * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    core::DeviceMemory<int> batch_dev(static_cast<size_t>(N));
    cudaMemcpyAsync(batch_dev.get(), h_batch.data(),
                    static_cast<size_t>(N) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    BbknnConfig cfg;
    cfg.k_within = k_within;

    graph::KnnResult result = bbknn(emb_dev, batch_dev, n_batches, cfg, stream);

    // Copy indices to host (result.neighbors is device int[N * k_total]).
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);

    return result;
}

// ---------------------------------------------------------------------------
// Compute ARI (Adjusted Rand Index) between two label vectors.
// Implements the contingency-table-based formula.
// ---------------------------------------------------------------------------
static double adjusted_rand_index(const std::vector<int>& labels_a,
                                   const std::vector<int>& labels_b)
{
    int N = static_cast<int>(labels_a.size());
    assert(N == static_cast<int>(labels_b.size()));

    // Build contingency table.
    std::unordered_map<int,int> cnt_a, cnt_b;
    std::map<std::pair<int,int>, int> cnt_ab;
    for (int i = 0; i < N; ++i) {
        cnt_a[labels_a[i]]++;
        cnt_b[labels_b[i]]++;
        cnt_ab[{labels_a[i], labels_b[i]}]++;
    }

    auto binom2 = [](int n) -> double {
        return n < 2 ? 0.0 : static_cast<double>(n) * (n - 1) / 2.0;
    };

    double sum_cij2 = 0.0;
    for (auto& kv : cnt_ab) sum_cij2 += binom2(kv.second);

    double sum_ai2 = 0.0;
    for (auto& kv : cnt_a) sum_ai2 += binom2(kv.second);
    double sum_bj2 = 0.0;
    for (auto& kv : cnt_b) sum_bj2 += binom2(kv.second);

    double Nn = binom2(N);
    if (Nn < 1.0) return 1.0;

    double expected = (sum_ai2 * sum_bj2) / Nn;
    double max_term = (sum_ai2 + sum_bj2) / 2.0;
    double denom    = max_term - expected;

    if (std::abs(denom) < 1e-12) return 1.0;
    return (sum_cij2 - expected) / denom;
}

}  // namespace

// =============================================================================
// Test: BBKNN_TwoBatch_NeighborBalance
//
// Run BBKNN on a balanced 2-batch dataset. For every cell confirm it has
// exactly kKWithin ± kBalanceTolerance neighbors from EACH batch.
// =============================================================================
TEST(BbknnCorrectness, BBKNN_TwoBatch_NeighborBalance) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device — skipping GPU test.";

    auto syn = make_synthetic(kTinyN, kTinyD, kTinyB, 4, kSeed);

    // Run BBKNN on device.
    auto result = run_bbknn_device(
        syn.embedding, syn.batch_labels, kTinyN, kTinyD, kTinyB, kKWithin);

    // Copy neighbor indices to host.
    int k_total = kKWithin * kTinyB;
    std::vector<int32_t> h_neighbors(static_cast<size_t>(kTinyN) * k_total);
    cudaMemcpy(h_neighbors.data(), result.neighbors.get(),
               static_cast<size_t>(kTinyN) * k_total * sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    int n_imbalanced = 0;
    for (int i = 0; i < kTinyN; ++i) {
        std::vector<int> per_batch(kTinyB, 0);
        for (int j = 0; j < k_total; ++j) {
            int32_t nb = h_neighbors[static_cast<size_t>(i) * k_total + j];
            if (nb < 0 || nb >= kTinyN) continue;   // padded sentinel
            per_batch[syn.batch_labels[nb]]++;
        }
        for (int b = 0; b < kTinyB; ++b) {
            int dev = std::abs(per_batch[b] - kKWithin);
            if (dev > kBalanceTolerance) {
                ++n_imbalanced;
                break;
            }
        }
    }

    double imbalance_frac = static_cast<double>(n_imbalanced) / kTinyN;
    printf("[BBKNN_TwoBatch_NeighborBalance] "
           "n_imbalanced_cells=%d / %d  frac=%.4f\n",
           n_imbalanced, kTinyN, imbalance_frac);

    EXPECT_EQ(n_imbalanced, 0)
        << n_imbalanced << " / " << kTinyN
        << " cells have neighbor imbalance > " << kBalanceTolerance;

    printf("[registry] 2026-04-13 | integrate/bbknn | tiny | "
           "neighbor_balance_fail_frac | %.4f | 0.0 | self | pending | %s\n",
           imbalance_frac, n_imbalanced == 0 ? "PASS" : "FAIL");
}

// =============================================================================
// Test: BBKNN_VsPython
//
// Compare our BBKNN neighbor graph to the Python bbknn reference.
// Metric: mean per-cell Jaccard similarity between the two neighbor sets.
// Requirement: mean Jaccard >= kJaccardMin.
// =============================================================================
TEST(BbknnCorrectness, BBKNN_VsPython) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device — skipping GPU test.";

    ensure_refs_tmp();

    auto syn = make_synthetic(kTinyN, kTinyD, kTinyB, 4, kSeed);

    std::string emb_path    = refs_path("bbknn_cmp_emb.npy");
    std::string batch_path  = refs_path("bbknn_cmp_batch.npy");
    std::string ref_path    = refs_path("bbknn_cmp_ref.npz");

    write_npy_f32_2d(emb_path, syn.embedding.data(), kTinyN, kTinyD);
    write_npy_i32_1d(batch_path, syn.batch_labels.data(), kTinyN);

    // Run Python bbknn reference.
    {
        std::string cmd =
            std::string(kPython) + " " + kBbknnRefScript +
            " --embedding "               + emb_path +
            " --batch-labels "            + batch_path +
            " --out-neighbors "           + ref_path +
            " --neighbors-within-batch "  + std::to_string(kKWithin);
        run_cmd(cmd);
    }

    // Read Python reference neighbor indices.
    int ref_rows = 0, ref_cols = 0;
    std::vector<int32_t> ref_inds = read_npz_i32(ref_path, "indices", ref_rows, ref_cols);
    ASSERT_EQ(ref_rows, kTinyN);
    int k_ref = ref_cols;   // k_total as seen by Python

    // Run our BBKNN on device.
    int k_total = kKWithin * kTinyB;
    auto result = run_bbknn_device(
        syn.embedding, syn.batch_labels, kTinyN, kTinyD, kTinyB, kKWithin);

    std::vector<int32_t> h_inds(static_cast<size_t>(kTinyN) * k_total);
    cudaMemcpy(h_inds.data(), result.neighbors.get(),
               static_cast<size_t>(kTinyN) * k_total * sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    // Compute mean Jaccard similarity per cell.
    double total_jaccard = 0.0;
    for (int i = 0; i < kTinyN; ++i) {
        std::set<int32_t> our_set, ref_set;
        for (int j = 0; j < k_total; ++j) {
            int32_t v = h_inds[static_cast<size_t>(i) * k_total + j];
            if (v >= 0) our_set.insert(v);
        }
        for (int j = 0; j < k_ref; ++j) {
            int32_t v = ref_inds[static_cast<size_t>(i) * k_ref + j];
            if (v >= 0) ref_set.insert(v);
        }
        // Jaccard = |intersection| / |union|
        int inter = 0;
        for (auto v : our_set)
            if (ref_set.count(v)) ++inter;
        int uni = static_cast<int>(our_set.size() + ref_set.size()) - inter;
        total_jaccard += (uni > 0) ? static_cast<double>(inter) / uni : 1.0;
    }
    double mean_jaccard = total_jaccard / kTinyN;

    printf("[BBKNN_VsPython] mean_jaccard=%.4f (min=%.2f)\n",
           mean_jaccard, kJaccardMin);

    EXPECT_GE(mean_jaccard, kJaccardMin)
        << "Mean Jaccard vs Python BBKNN " << mean_jaccard
        << " < tolerance " << kJaccardMin;

    printf("[registry] 2026-04-13 | integrate/bbknn | tiny | "
           "jaccard_vs_python | %.4f | %.2f | bbknn_python | pending | %s\n",
           mean_jaccard, kJaccardMin,
           mean_jaccard >= kJaccardMin ? "PASS" : "FAIL");
}

// =============================================================================
// Test: BBKNN_LeidenDownstream
//
// Run Leiden clustering on the BBKNN graph for a 2-batch dataset with planted
// cell types. Compare cluster labels via ARI against the ground-truth cell-type
// labels. Requirement: ARI >= kLeidenAriMin.
//
// This test also implicitly checks that BBKNN produces a valid KnnResult that
// the Leiden kernel can consume without modification.
//
// Conditionally compiled: requires SINGLET_GPU_HAS_CUGRAPH. Without cuGraph
// the test is SKIPPED via GTEST_SKIP().
// =============================================================================
TEST(BbknnCorrectness, BBKNN_LeidenDownstream) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device — skipping GPU test.";

#ifndef SINGLET_GPU_HAS_CUGRAPH
    GTEST_SKIP() << "cuGraph not found at build time — "
                    "BBKNN_LeidenDownstream will run only on GPU nodes with cuGraph.";
#else

    auto syn = make_synthetic(kMedN, kMedD, kMedB, kNCellTypes, kSeed + 7,
                              /*biology_scale=*/5.0f,
                              /*batch_scale=*/2.0f,
                              /*noise_scale=*/0.25f);

    // Run BBKNN.
    int k_total = kKWithin * kMedB;
    auto bbknn_result = run_bbknn_device(
        syn.embedding, syn.batch_labels, kMedN, kMedD, kMedB, kKWithin);

    // Run Leiden on the BBKNN graph.
    using namespace singlet::gpu;
    using namespace singlet::gpu::graph;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    LeidenConfig leiden_cfg;
    leiden_cfg.resolution = 1.0f;
    leiden_cfg.max_iter   = 100;
    leiden_cfg.seed       = kSeed;

    LeidenResult leiden_result = leiden(bbknn_result, kMedN, leiden_cfg, stream);

    // Copy Leiden labels to host.
    std::vector<int32_t> h_leiden(kMedN);
    cudaMemcpyAsync(h_leiden.data(), leiden_result.labels.ptr,
                    static_cast<size_t>(kMedN) * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // Convert to std::vector<int> for ARI computation.
    std::vector<int> leiden_labels(h_leiden.begin(), h_leiden.end());

    // ARI vs ground-truth cell-type labels.
    double ari = adjusted_rand_index(leiden_labels, syn.ct_labels);

    printf("[BBKNN_LeidenDownstream] ARI=%.4f (min=%.2f)\n",
           ari, kLeidenAriMin);

    EXPECT_GE(ari, kLeidenAriMin)
        << "Leiden ARI " << ari << " < tolerance " << kLeidenAriMin;

    printf("[registry] 2026-04-13 | integrate/bbknn | medium | "
           "leiden_ari | %.4f | %.2f | self+cugraph | pending | %s\n",
           ari, kLeidenAriMin,
           ari >= kLeidenAriMin ? "PASS" : "FAIL");

#endif  // SINGLET_GPU_HAS_CUGRAPH
}
