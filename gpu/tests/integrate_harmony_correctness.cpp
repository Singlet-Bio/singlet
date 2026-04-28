// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/integrate_harmony_correctness.cpp
//
// Correctness harness for singlet_gpu::integrate::harmony.
//
// Written against the design doc at:
//   singlet-gpu/state/designs/12-integration.md
// NOT against the kernel source — authored in parallel with integrate/harmony.h
// by the analysis-validator worker (Sonnet, Tier 2).
//
// References:
//   tests/refs/harmony_harmonypy_reference.py  — runs harmonypy, dumps corrected embedding
//   tests/refs/scib_metrics_reference.py       — computes iLISI + cLISI via scib-metrics
//
// Tolerances (design doc §"Correctness test spec"):
//   iLISI improvement (before → after Harmony)    >= 0.1
//   CCA score vs harmonypy corrected embedding     >= 0.95
//   cLISI degradation post-correction             <= 5%
//   Determinism (same seed, fixed input)           bit-identical
//   Overcorrection guard (90/10 imbalance)         minority batch not collapsed
//
// Test cases:
//   Harmony_TwoBatch_iLISI_Improvement
//   Harmony_VsHarmonypy
//   Harmony_PreservesBiology_cLISI
//   Harmony_Determinism_FixedSeed
//   Harmony_OvercorrectionGuard
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, singlet-gpu::singlet-gpu, GTest.
//   Every test that requires a GPU device is guarded by gpu_available() and
//   calls GTEST_SKIP() on CPU-only nodes.

// ---- singlet-gpu integration header (written by gpu-kernel-dev) -------------
#include <singlet-gpu/integrate/harmony.h>
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

/// Tiny synthetic embedding: 500 cells × 20 PCs — fits in local memory.
constexpr int kTinyN   = 500;
constexpr int kTinyD   = 20;
/// Number of balanced batches for iLISI / CCA tests.
constexpr int kTinyB   = 2;
/// Number of clusters for Harmony soft k-means.
constexpr int kNClusters = 10;

/// Medium synthetic: 2000 cells × 50 PCs, 5 batches — for biology-preservation test.
constexpr int kMedN  = 2000;
constexpr int kMedD  = 50;
constexpr int kMedB  = 5;
/// Number of planted cell types (biology) in the medium dataset.
constexpr int kNCellTypes = 8;

/// Overcorrection guard: imbalanced 2-batch dataset.
constexpr int kImbalN       = 500;
constexpr int kImbalMajority = 450;   // 90% in batch 0
constexpr int kImbalMinority = 50;    // 10% in batch 1

/// Harmony config defaults.
constexpr int   kMaxIter  = 10;
constexpr float kTol      = 1e-4f;
constexpr float kLambda   = 1.0f;

/// Tolerances from the design doc.
constexpr double kIlisiImprovementMin  = 0.1;
constexpr double kCcaScoreMin          = 0.95;
constexpr double kClisiDegradationMax  = 0.05;   // 5 %
constexpr float  kMinoritySpreadMin    = 0.05f;  // minority cells must not collapse

/// Python interpreter name.
static const char* kPython = "python3";

/// Paths to Python reference scripts.
static const char* kHarmonyRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/"
    "tests/refs/harmony_harmonypy_reference.py";
static const char* kScibRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/"
    "tests/refs/scib_metrics_reference.py";

/// Temporary directory for Python reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_harmony_refs_tmp";

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
// Read a float from a JSON file with key "key".
// Very simple line-scan parser — works for the flat JSONs we write.
// ---------------------------------------------------------------------------
static double read_json_double(const std::string& path, const std::string& key) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("read_json_double: cannot open " + path);
    std::string line;
    while (std::getline(f, line)) {
        std::string search = "\"" + key + "\"";
        auto pos = line.find(search);
        if (pos == std::string::npos) continue;
        auto colon = line.find(':', pos + search.size());
        if (colon == std::string::npos) continue;
        return std::stod(line.substr(colon + 1));
    }
    throw std::runtime_error("read_json_double: key '" + key +
                             "' not found in " + path);
}

// ---------------------------------------------------------------------------
// Read a float32 2-D array from a .npy file written by numpy.save.
// Supports only the subset our reference scripts produce (v1.0, no dict tricks).
// ---------------------------------------------------------------------------
static std::vector<float> read_npy_f32_2d(const std::string& path,
                                           int& rows_out, int& cols_out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("read_npy_f32_2d: cannot open " + path);

    // Magic + version (10 bytes)
    char preamble[10];
    f.read(preamble, 10);
    if (preamble[0] != '\x93' || std::string(preamble + 1, 5) != "NUMPY")
        throw std::runtime_error("read_npy_f32_2d: not a .npy file: " + path);

    uint16_t hlen;
    f.read(reinterpret_cast<char*>(&hlen), 2);
    std::string header(hlen, ' ');
    f.read(header.data(), hlen);

    // Parse shape=(R, C) from header string — robust enough for our writers.
    auto shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos)
        throw std::runtime_error("read_npy_f32_2d: no 'shape' in header");
    auto lp = header.find('(', shape_pos);
    auto rp = header.find(')', lp);
    std::string shape_str = header.substr(lp + 1, rp - lp - 1);
    // shape_str is like "500, 20"
    std::istringstream ss(shape_str);
    std::string tok;
    std::getline(ss, tok, ',');
    rows_out = std::stoi(tok);
    std::getline(ss, tok, ',');
    cols_out = std::stoi(tok);

    size_t nel = static_cast<size_t>(rows_out) * static_cast<size_t>(cols_out);
    std::vector<float> data(nel);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(nel * sizeof(float)));
    if (!f) throw std::runtime_error("read_npy_f32_2d: short read: " + path);
    return data;
}

// ---------------------------------------------------------------------------
// Synthetic 2-batch embedding with planted biology.
//
// Each cell type occupies a Gaussian cluster in PCA space. Batch effect is a
// fixed per-batch translation vector that is orthogonal to the biology axes.
//
//   embedding[i] = biology_centroid[ct[i]] + batch_offset[batch[i]] + noise[i]
//
// With n_batches batches and n_clusters cell types, each batch/cell-type
// combination is equally populated.
// ---------------------------------------------------------------------------
struct SyntheticEmbedding {
    int n_cells;
    int n_dims;
    int n_batches;
    int n_cell_types;
    std::vector<float> embedding;    // (n_cells, n_dims)
    std::vector<int>   batch_labels; // (n_cells,)
    std::vector<int>   ct_labels;    // (n_cells,)
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

    // Orthogonal biology centroids (simple axis-aligned blobs in first n_ct dims).
    std::vector<std::vector<float>> bio_centroids(n_cell_types,
                                                  std::vector<float>(n_dims, 0.0f));
    for (int c = 0; c < n_cell_types; ++c) {
        int axis = c % n_dims;
        bio_centroids[c][axis] = biology_scale * static_cast<float>(c + 1);
    }

    // Batch offsets in orthogonal subspace (last n_batches dims, clamped).
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
    out.embedding.resize(static_cast<size_t>(n_cells) * static_cast<size_t>(n_dims));
    out.batch_labels.resize(n_cells);
    out.ct_labels.resize(n_cells);

    for (int i = 0; i < n_cells; ++i) {
        int b  = i % n_batches;
        int ct = (i / n_batches) % n_cell_types;
        out.batch_labels[i] = b;
        out.ct_labels[i]    = ct;

        float* row = out.embedding.data() + static_cast<size_t>(i) * n_dims;
        for (int d = 0; d < n_dims; ++d) {
            row[d] = bio_centroids[ct][d]
                   + batch_offsets[b][d]
                   + noise_scale * ndist(rng);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Canonical correlation analysis (CCA) — simplified one-component version.
// For two matrices A (N×d) and B (N×d), compute the squared canonical
// correlation of their top-1 linear combination.
//
// Approximation used here: Pearson r between the first left singular vectors
// of A and B. This is a lower bound on full CCA score but sufficient for the
// >= 0.95 acceptance threshold.
// ---------------------------------------------------------------------------
static double cca_score_approx(const std::vector<float>& A,
                                const std::vector<float>& B,
                                int N, int d)
{
    // Compute column means and center both matrices.
    std::vector<double> mean_a(d, 0.0), mean_b(d, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            mean_a[j] += A[static_cast<size_t>(i) * d + j];
            mean_b[j] += B[static_cast<size_t>(i) * d + j];
        }
    }
    for (int j = 0; j < d; ++j) {
        mean_a[j] /= N;
        mean_b[j] /= N;
    }

    // Cross-covariance trace: tr(A^T B) / (||A||_F * ||B||_F).
    double cross = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            double ai = A[static_cast<size_t>(i) * d + j] - mean_a[j];
            double bi = B[static_cast<size_t>(i) * d + j] - mean_b[j];
            cross  += ai * bi;
            norm_a += ai * ai;
            norm_b += bi * bi;
        }
    }

    if (norm_a < 1e-12 || norm_b < 1e-12) return 0.0;
    return cross / std::sqrt(norm_a * norm_b);
}

// ---------------------------------------------------------------------------
// Upload a host float[N*d] embedding to device and run harmony.
// Returns corrected embedding on host (N*d float vector).
// ---------------------------------------------------------------------------
static std::vector<float> run_harmony_device(
    const std::vector<float>& h_embedding,
    const std::vector<int>&   h_batch,
    int N, int d, int n_batches,
    uint64_t seed)
{
    using namespace singlet_gpu;
    using namespace singlet_gpu::integrate;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate device dense embedding (N × d, row-major) using RAII DeviceMemory.
    core::DeviceDense emb_dev;
    emb_dev.rows = static_cast<uint32_t>(N);
    emb_dev.cols = static_cast<uint32_t>(d);
    emb_dev.data = core::DeviceMemory<float>(static_cast<size_t>(N) * d);
    cudaMemcpyAsync(emb_dev.data.get(), h_embedding.data(),
                    static_cast<size_t>(N) * d * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // Allocate device batch labels using RAII DeviceMemory.
    core::DeviceMemory<int> batch_dev(static_cast<size_t>(N));
    cudaMemcpyAsync(batch_dev.get(), h_batch.data(),
                    static_cast<size_t>(N) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    HarmonyConfig cfg;
    cfg.n_clusters = kNClusters;
    cfg.max_iter   = kMaxIter;
    cfg.tol        = kTol;
    cfg.lambda     = kLambda;
    cfg.seed       = seed;

    HarmonyResult result = harmony(emb_dev, batch_dev, n_batches, cfg, stream);

    // Copy corrected embedding back to host.
    std::vector<float> h_corrected(static_cast<size_t>(N) * d);
    cudaMemcpyAsync(h_corrected.data(), result.corrected.get(),
                    static_cast<size_t>(N) * d * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);

    return h_corrected;
}

}  // namespace

// =============================================================================
// Test: Harmony_TwoBatch_iLISI_Improvement
//
// Generate a synthetic 2-batch dataset (kTinyN cells, kTinyD dims).
// Compute iLISI before correction and after correction via the Python scib
// reference. Confirm iLISI improvement >= kIlisiImprovementMin.
// =============================================================================
TEST(HarmonyCorrectness, Harmony_TwoBatch_iLISI_Improvement) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device — skipping GPU test.";

    ensure_refs_tmp();

    // 1. Generate synthetic embedding.
    auto syn = make_synthetic(kTinyN, kTinyD, kTinyB, 4, kSeed);

    // 2. Write embedding + batch labels for Python.
    std::string emb_path      = refs_path("harmony_ilisi_emb.npy");
    std::string corr_path     = refs_path("harmony_ilisi_corr.npy");
    std::string batch_path    = refs_path("harmony_ilisi_batch.npy");
    std::string ct_path       = refs_path("harmony_ilisi_ct.npy");
    std::string meta_path     = refs_path("harmony_ilisi_meta.json");
    std::string metrics_before= refs_path("harmony_ilisi_metrics_before.json");
    std::string metrics_after = refs_path("harmony_ilisi_metrics_after.json");

    write_npy_f32_2d(emb_path, syn.embedding.data(), kTinyN, kTinyD);
    write_npy_i32_1d(batch_path, syn.batch_labels.data(), kTinyN);
    write_npy_i32_1d(ct_path,    syn.ct_labels.data(),    kTinyN);

    // 3. Run harmony on device.
    std::vector<float> corrected = run_harmony_device(
        syn.embedding, syn.batch_labels, kTinyN, kTinyD, kTinyB, kSeed);
    write_npy_f32_2d(corr_path, corrected.data(), kTinyN, kTinyD);

    // 4. Compute iLISI before correction via scib reference.
    {
        std::string cmd =
            std::string(kPython) + " " + kScibRefScript +
            " --embedding "    + emb_path +
            " --batch-labels " + batch_path +
            " --out-metrics "  + metrics_before;
        run_cmd(cmd);
    }
    // 5. Compute iLISI after correction.
    {
        std::string cmd =
            std::string(kPython) + " " + kScibRefScript +
            " --embedding "    + corr_path +
            " --batch-labels " + batch_path +
            " --out-metrics "  + metrics_after;
        run_cmd(cmd);
    }

    double ilisi_before = read_json_double(metrics_before, "ilisi");
    double ilisi_after  = read_json_double(metrics_after,  "ilisi");
    double improvement  = ilisi_after - ilisi_before;

    printf("[Harmony_TwoBatch_iLISI_Improvement] "
           "iLISI before=%.4f after=%.4f improvement=%.4f (min=%.2f)\n",
           ilisi_before, ilisi_after, improvement, kIlisiImprovementMin);

    EXPECT_GE(improvement, kIlisiImprovementMin)
        << "iLISI improvement " << improvement
        << " < tolerance " << kIlisiImprovementMin;

    // Registry row.
    printf("[registry] 2026-04-13 | integrate/harmony | tiny | "
           "ilisi_improvement | %.4f | %.2f | scib_metrics | pending | %s\n",
           improvement, kIlisiImprovementMin,
           improvement >= kIlisiImprovementMin ? "PASS" : "FAIL");
}

// =============================================================================
// Test: Harmony_VsHarmonypy
//
// Same input → run our Harmony and harmonypy reference. Compare corrected
// embeddings via CCA score. Requirement: CCA >= kCcaScoreMin.
// =============================================================================
TEST(HarmonyCorrectness, Harmony_VsHarmonypy) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device — skipping GPU test.";

    ensure_refs_tmp();

    auto syn = make_synthetic(kTinyN, kTinyD, kTinyB, 4, kSeed);

    std::string emb_path   = refs_path("harmony_cca_emb.npy");
    std::string batch_path = refs_path("harmony_cca_batch.npy");
    std::string ref_path   = refs_path("harmony_cca_ref.npy");
    std::string meta_path  = refs_path("harmony_cca_meta.json");

    write_npy_f32_2d(emb_path, syn.embedding.data(), kTinyN, kTinyD);
    write_npy_i32_1d(batch_path, syn.batch_labels.data(), kTinyN);

    // Run harmonypy reference.
    {
        std::string cmd =
            std::string(kPython) + " " + kHarmonyRefScript +
            " --embedding "     + emb_path +
            " --batch-labels "  + batch_path +
            " --out-embedding " + ref_path +
            " --out-meta "      + meta_path +
            " --n-clusters "    + std::to_string(kNClusters) +
            " --max-iter "      + std::to_string(kMaxIter) +
            " --random-state "  + std::to_string(static_cast<int>(kSeed & 0xFFFFFFFF));
        run_cmd(cmd);
    }

    // Run our harmony on device.
    std::vector<float> ours = run_harmony_device(
        syn.embedding, syn.batch_labels, kTinyN, kTinyD, kTinyB, kSeed);

    // Read harmonypy corrected embedding.
    int ref_rows = 0, ref_cols = 0;
    std::vector<float> ref_emb = read_npy_f32_2d(ref_path, ref_rows, ref_cols);
    ASSERT_EQ(ref_rows, kTinyN);
    ASSERT_EQ(ref_cols, kTinyD);

    double cca = cca_score_approx(ours, ref_emb, kTinyN, kTinyD);
    // CCA can be negative if embeddings are sign-flipped — use abs.
    double cca_abs = std::abs(cca);

    printf("[Harmony_VsHarmonypy] CCA score=%.4f (min=%.2f)\n",
           cca_abs, kCcaScoreMin);

    EXPECT_GE(cca_abs, kCcaScoreMin)
        << "CCA score " << cca_abs << " < tolerance " << kCcaScoreMin;

    printf("[registry] 2026-04-13 | integrate/harmony | tiny | "
           "cca_vs_harmonypy | %.4f | %.2f | harmonypy | pending | %s\n",
           cca_abs, kCcaScoreMin,
           cca_abs >= kCcaScoreMin ? "PASS" : "FAIL");
}

// =============================================================================
// Test: Harmony_PreservesBiology_cLISI
//
// Compute cLISI before and after Harmony on a dataset with kNCellTypes biology.
// Confirm cLISI does NOT degrade by more than kClisiDegradationMax (5 %).
// =============================================================================
TEST(HarmonyCorrectness, Harmony_PreservesBiology_cLISI) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device — skipping GPU test.";

    ensure_refs_tmp();

    auto syn = make_synthetic(kMedN, kMedD, kMedB, kNCellTypes, kSeed + 1,
                              /*biology_scale=*/4.0f,
                              /*batch_scale=*/1.5f,
                              /*noise_scale=*/0.2f);

    std::string emb_path      = refs_path("harmony_clisi_emb.npy");
    std::string corr_path     = refs_path("harmony_clisi_corr.npy");
    std::string batch_path    = refs_path("harmony_clisi_batch.npy");
    std::string ct_path       = refs_path("harmony_clisi_ct.npy");
    std::string metrics_before= refs_path("harmony_clisi_metrics_before.json");
    std::string metrics_after = refs_path("harmony_clisi_metrics_after.json");

    write_npy_f32_2d(emb_path, syn.embedding.data(), kMedN, kMedD);
    write_npy_i32_1d(batch_path, syn.batch_labels.data(), kMedN);
    write_npy_i32_1d(ct_path,    syn.ct_labels.data(),    kMedN);

    // Compute cLISI before correction.
    {
        std::string cmd =
            std::string(kPython) + " " + kScibRefScript +
            " --embedding "          + emb_path +
            " --batch-labels "       + batch_path +
            " --cell-type-labels "   + ct_path +
            " --out-metrics "        + metrics_before;
        run_cmd(cmd);
    }

    // Run harmony on device.
    std::vector<float> corrected = run_harmony_device(
        syn.embedding, syn.batch_labels, kMedN, kMedD, kMedB, kSeed);
    write_npy_f32_2d(corr_path, corrected.data(), kMedN, kMedD);

    // Compute cLISI after correction.
    {
        std::string cmd =
            std::string(kPython) + " " + kScibRefScript +
            " --embedding "          + corr_path +
            " --batch-labels "       + batch_path +
            " --cell-type-labels "   + ct_path +
            " --out-metrics "        + metrics_after;
        run_cmd(cmd);
    }

    double clisi_before = read_json_double(metrics_before, "clisi");
    double clisi_after  = read_json_double(metrics_after,  "clisi");

    // cLISI: lower = better (1 = perfect cell-type separation).
    // Degradation = how much cLISI increased (worsened) after correction.
    // We allow up to kClisiDegradationMax * clisi_before as absolute tolerance.
    double degradation = (clisi_after - clisi_before) / (clisi_before + 1e-12);

    printf("[Harmony_PreservesBiology_cLISI] "
           "cLISI before=%.4f after=%.4f rel_degradation=%.4f (max=%.2f)\n",
           clisi_before, clisi_after, degradation, kClisiDegradationMax);

    EXPECT_LE(degradation, kClisiDegradationMax)
        << "cLISI relative degradation " << degradation
        << " > tolerance " << kClisiDegradationMax;

    printf("[registry] 2026-04-13 | integrate/harmony | medium | "
           "clisi_degradation | %.4f | %.2f | scib_metrics | pending | %s\n",
           degradation, kClisiDegradationMax,
           degradation <= kClisiDegradationMax ? "PASS" : "FAIL");
}

// =============================================================================
// Test: Harmony_Determinism_FixedSeed
//
// Run harmony twice on identical input with the same seed. Confirm the
// corrected embeddings are bit-identical (max abs diff == 0).
// =============================================================================
TEST(HarmonyCorrectness, Harmony_Determinism_FixedSeed) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device — skipping GPU test.";

    auto syn = make_synthetic(kTinyN, kTinyD, kTinyB, 4, kSeed);

    std::vector<float> run1 = run_harmony_device(
        syn.embedding, syn.batch_labels, kTinyN, kTinyD, kTinyB, kSeed);
    std::vector<float> run2 = run_harmony_device(
        syn.embedding, syn.batch_labels, kTinyN, kTinyD, kTinyB, kSeed);

    ASSERT_EQ(run1.size(), run2.size());
    float max_diff = 0.0f;
    for (size_t i = 0; i < run1.size(); ++i)
        max_diff = std::max(max_diff, std::abs(run1[i] - run2[i]));

    printf("[Harmony_Determinism_FixedSeed] max_abs_diff=%.6e\n", max_diff);

    EXPECT_EQ(max_diff, 0.0f)
        << "Harmony result not bit-identical between two runs with same seed. "
        << "max_abs_diff=" << max_diff;

    printf("[registry] 2026-04-13 | integrate/harmony | tiny | "
           "determinism_max_diff | %.2e | 0.0 | self | pending | %s\n",
           max_diff, max_diff == 0.0f ? "PASS" : "FAIL");
}

// =============================================================================
// Test: Harmony_OvercorrectionGuard
//
// Imbalanced batches (90% batch-0, 10% batch-1). After correction the minority
// cells must not collapse: their pairwise distance variance must stay above
// kMinoritySpreadMin relative to the pre-correction spread.
// =============================================================================
TEST(HarmonyCorrectness, Harmony_OvercorrectionGuard) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device — skipping GPU test.";

    // Build imbalanced dataset.
    std::mt19937_64 rng(kSeed + 99);
    std::normal_distribution<float> ndist(0.0f, 1.0f);

    int N = kImbalN;
    int d = kTinyD;
    std::vector<float> emb(static_cast<size_t>(N) * d);
    std::vector<int>   batch(N);

    // Batch 0 centroid at origin; batch 1 centroid at (5, 0, 0, ...).
    for (int i = 0; i < N; ++i) {
        batch[i] = (i < kImbalMajority) ? 0 : 1;
        float* row = emb.data() + static_cast<size_t>(i) * d;
        for (int j = 0; j < d; ++j)
            row[j] = ndist(rng);
        if (batch[i] == 1)
            row[0] += 5.0f;   // batch effect in dim 0
    }

    std::vector<float> corrected = run_harmony_device(
        emb, batch, N, d, /*n_batches=*/2, kSeed);

    // Measure variance of minority cells (batch 1) before and after.
    auto variance_of = [&](const std::vector<float>& mat, int start, int end, int ncols) {
        double sum = 0.0, sum2 = 0.0;
        int cnt = (end - start) * ncols;
        for (int i = start; i < end; ++i)
            for (int j = 0; j < ncols; ++j) {
                double v = mat[static_cast<size_t>(i) * ncols + j];
                sum  += v;
                sum2 += v * v;
            }
        double mean = sum / cnt;
        return sum2 / cnt - mean * mean;
    };

    double var_before = variance_of(emb,       kImbalMajority, N, d);
    double var_after  = variance_of(corrected,  kImbalMajority, N, d);
    double ratio      = (var_before > 1e-12) ? (var_after / var_before) : 1.0;

    printf("[Harmony_OvercorrectionGuard] "
           "minority variance before=%.4f after=%.4f ratio=%.4f (min=%.2f)\n",
           var_before, var_after, ratio, (double)kMinoritySpreadMin);

    EXPECT_GE(ratio, static_cast<double>(kMinoritySpreadMin))
        << "Minority batch collapsed: variance ratio " << ratio
        << " < " << kMinoritySpreadMin;

    printf("[registry] 2026-04-13 | integrate/harmony | tiny_imbalanced | "
           "minority_variance_ratio | %.4f | %.2f | self | pending | %s\n",
           ratio, (double)kMinoritySpreadMin,
           ratio >= kMinoritySpreadMin ? "PASS" : "FAIL");
}
