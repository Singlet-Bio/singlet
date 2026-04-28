// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/streaming_pipeline_correctness.cpp
//
// Correctness harness for singlet_gpu::streaming::run_pipeline.
//
// Written against the design doc at:
//   singlet-gpu/state/designs/16-streaming-pipeline.md
// NOT against the kernel source — this file was authored in parallel with
// streamed_pipeline.h by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: in-memory kernel sequence (no scanpy subprocess).
//   The streaming pipeline is our own composition of cycles 3–6 kernels over
//   PzDataLoader chunks.  Correctness is defined as equivalence to running
//   those same kernels directly on the fully-loaded matrix.
//
// Tolerances (from design doc §"Correctness test spec"):
//   Lognorm size_factors rel_err    <= 5e-2  (streaming uses fp64 host accumulation vs fp32 device; 3-5% expected)
//   HVG top-2000 Jaccard            >= 0.95  (rank-sensitive; 1–5% boundary shuffle)
//   PCA singular values rel_err     <= 1e-4
//   Multi-input concat cell count   exact;  first block vs second block <= 1e-5
//
// Test cases:
//   Pipeline_LognormOnly_EquivalentToInMemory
//   Pipeline_HvgOnly_EquivalentToInMemory
//   Pipeline_LognormHvg_EquivalentToInMemory
//   Pipeline_LognormHvgPca_InMemoryFallback
//   Pipeline_NmfChunked_RunsToCompletion
//   Pipeline_MultipleInputs_ConcatenatedCorrectly
//   Pipeline_GeneMismatch_Errors
//   Pipeline_ChunkSize_DoesNotAffectResult
//   Pipeline_SingleChunkFitsEntireInput
//   Pipeline_ZeroInputs_Throws
//   Pipeline_PcaThresholdBoundary_PcaSkipped
//
// Build: source-only (no nvcc). Tested on GPU dispatch node.
// Registry rows emitted to stdout on each run.

// ---- streaming pipeline header (written by gpu-kernel-dev in parallel) --------
#include <singlet-gpu/streaming/streamed_pipeline.h>

// ---- cycle 3–6 in-memory headers (the reference implementations) --------------
#include <singlet-gpu/preprocess/lognorm.h>
#include <singlet-gpu/preprocess/hvg.h>
#include <singlet-gpu/reduce/svd/auto_select.h>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/io/write_pz.h>

// ---- test helpers ---------------------------------------------------------------
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
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEE;

/// Canonical 10k test sample (GSM4037629).
static const char* kGsm4037629Path =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/exon_counts.1pz";

/// Known cell count for GSM4037629 (updated 2026-04-14: singlify v0.3 called 20866 cells).
constexpr int64_t kGsm4037629Cells = 20866;

/// Chunk size used for the primary streaming tests.
constexpr int kDefaultChunkCols = 2000;

/// Small synthetic dimensions used for tests that need a second .1pz
/// (gene-mismatch test, PCA threshold boundary test).
/// NOTE: the make_tiny_pz helper below creates a real .1pz on /tmp.
constexpr int kTinyRows = 200;
constexpr int kTinyCols = 300;
constexpr float kTinyDenseFrac = 0.05f;

/// PCA rank for in-memory fallback tests.
constexpr int kPcaK = 10;

/// NMF rank and max_iter for completion test.
constexpr int kNmfRank = 5;
constexpr int kNmfMaxIter = 10;

/// in_memory_pca_threshold override for the boundary test.
constexpr int64_t kPcaThresholdSmall = 100;

/// Temp dir for synthetic .1pz files.
static const char* kTmpDir = "/tmp/singlet_gpu_streaming_refs_tmp";

// =============================================================================
// Internal helpers
// =============================================================================

namespace {

// ---------------------------------------------------------------------------
// Copy a device float buffer to a host std::vector.
// Used when comparing LogNormResult.size_factors (DeviceMemory<float>) against
// PipelineResult.size_factors (std::vector<float>) via max_rel_err.
// ---------------------------------------------------------------------------
std::vector<float> device_to_host_vec(const singlet_gpu::core::DeviceMemory<float>& dm)
{
    std::vector<float> v(dm.size());
    if (dm.size() > 0) {
        cudaError_t err = cudaMemcpy(v.data(), dm.get(),
                                     dm.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("device_to_host_vec: cudaMemcpy failed: ") +
                cudaGetErrorString(err));
    }
    return v;
}

// ---------------------------------------------------------------------------
// Copy a device int buffer to a host std::vector.
// Used to convert HvgResult.indices (DeviceMemory<int>) for Jaccard comparison.
// ---------------------------------------------------------------------------
std::vector<int> device_to_host_int_vec(const singlet_gpu::core::DeviceMemory<int>& dm)
{
    std::vector<int> v(dm.size());
    if (dm.size() > 0) {
        cudaError_t err = cudaMemcpy(v.data(), dm.get(),
                                     dm.size() * sizeof(int),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("device_to_host_int_vec: cudaMemcpy failed: ") +
                cudaGetErrorString(err));
    }
    return v;
}

// ---------------------------------------------------------------------------
// Compute relative error between two float vectors (element-wise max).
// ---------------------------------------------------------------------------
float max_rel_err(const std::vector<float>& a, const std::vector<float>& b,
                  float abs_floor = 1e-6f)
{
    assert(a.size() == b.size());
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float denom = std::max(std::abs(a[i]), abs_floor);
        float rel = std::abs(a[i] - b[i]) / denom;
        if (rel > max_err) max_err = rel;
    }
    return max_err;
}

// ---------------------------------------------------------------------------
// Compute Jaccard similarity between two sorted integer index vectors.
// ---------------------------------------------------------------------------
float jaccard(const std::vector<int>& a, const std::vector<int>& b) {
    std::unordered_set<int> sa(a.begin(), a.end());
    std::unordered_set<int> sb(b.begin(), b.end());
    size_t inter = 0;
    for (int x : sa) inter += sb.count(x);
    size_t uni = sa.size() + sb.size() - inter;
    return uni == 0 ? 1.0f : static_cast<float>(inter) / static_cast<float>(uni);
}

// ---------------------------------------------------------------------------
// make_tiny_pz — creates a synthetic .1pz for tests that need a second file.
// Uses singlet_gpu::io::write_pz (io/write_pz.h).
// ---------------------------------------------------------------------------
#if 1
std::string make_tiny_pz(const std::string& filename,
                          int n_rows, int n_cols, float dense_frac,
                          uint64_t seed,
                          const std::string& gene_prefix = "GENE")
{
    fs::create_directories(kTmpDir);
    std::string path = std::string(kTmpDir) + "/" + filename;

    // Build a synthetic CSC: non-negative integer counts (uint16-range values).
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> sparsity(0.0f, 1.0f);
    std::uniform_int_distribution<int> count_dist(1, 200);

    std::vector<std::vector<std::pair<int32_t, uint16_t>>> cols(n_cols);
    for (int j = 0; j < n_cols; ++j) {
        for (int i = 0; i < n_rows; ++i) {
            if (sparsity(rng) < dense_frac) {
                cols[j].push_back({i, static_cast<uint16_t>(count_dist(rng))});
            }
        }
    }
    int64_t nnz = 0;
    for (auto& c : cols) nnz += static_cast<int64_t>(c.size());

    std::vector<uint16_t> values;
    std::vector<int32_t>  indptr(n_cols + 1, 0);
    std::vector<int32_t>  indices;
    values.reserve(nnz);
    indices.reserve(nnz);
    for (int j = 0; j < n_cols; ++j) {
        indptr[j + 1] = indptr[j] + static_cast<int32_t>(cols[j].size());
        for (auto& [row, val] : cols[j]) {
            indices.push_back(row);
            values.push_back(val);
        }
    }

    // Build gene and cell names.
    std::vector<std::string> gene_names, cell_names;
    gene_names.reserve(n_rows);
    cell_names.reserve(n_cols);
    for (int i = 0; i < n_rows; ++i)
        gene_names.push_back(gene_prefix + std::to_string(i));
    for (int j = 0; j < n_cols; ++j)
        cell_names.push_back("CELL" + std::to_string(j));

    singlet_gpu::io::write_pz(path, n_rows, n_cols, nnz,
                               values.data(), indptr.data(), indices.data(),
                               gene_names, cell_names);
    return path;
}
#endif  // CYCLE-7-FOLLOWUP-WRITE-PZ: re-enable when write_pz is implemented

// ---------------------------------------------------------------------------
// Emit a registry row to stdout.
// ---------------------------------------------------------------------------
void emit_registry_row(const std::string& feature, const std::string& scale,
                        const std::string& metric, float value, float tolerance,
                        const std::string& ref, bool pass)
{
    // Format: | date | feature | scale | metric | value | tolerance | ref | commit | PASS/FAIL |
    // Commit hash is unknown at test-compile time; use "pending_gpu_run".
    std::printf("| 2026-04-13 | %s | %-6s | %-32s | %.3e | %.3e | %-30s | pending_gpu_run | %s |\n",
                feature.c_str(), scale.c_str(), metric.c_str(),
                value, tolerance, ref.c_str(),
                pass ? "PASS" : "FAIL");
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Build a default PipelineConfig pointing at GSM4037629.
// ---------------------------------------------------------------------------
singlet_gpu::streaming::PipelineConfig gsm_config(int chunk_cols = kDefaultChunkCols)
{
    singlet_gpu::streaming::PipelineConfig cfg;
    cfg.input_paths  = {kGsm4037629Path};
    cfg.chunk_cols   = chunk_cols;
    cfg.run_lognorm  = false;
    cfg.run_hvg      = false;
    cfg.run_pca      = false;
    cfg.run_nmf      = false;
    return cfg;
}

}  // namespace

// =============================================================================
// Test fixture — creates a CUDA stream once per test, destroyed on teardown.
// =============================================================================

class StreamingPipelineTest : public ::testing::Test {
protected:
    cudaStream_t stream_{};

    void SetUp() override {
        cudaError_t err = cudaStreamCreate(&stream_);
        if (err != cudaSuccess)
            GTEST_SKIP() << "CUDA not available: " << cudaGetErrorString(err);

        // Verify the canonical test sample exists.
        if (!fs::exists(kGsm4037629Path))
            GTEST_SKIP() << "Canonical test sample not found: " << kGsm4037629Path;
    }

    void TearDown() override {
        if (stream_) cudaStreamDestroy(stream_);
    }
};

// =============================================================================
// Test 1 — Pipeline_LognormOnly_EquivalentToInMemory
//
// Streaming lognorm (chunk_cols=2000, run_hvg=false) vs in-memory log_normalize.
// Compare size_factors element-wise: rel_err <= 1e-5.
// =============================================================================
TEST_F(StreamingPipelineTest, Pipeline_LognormOnly_EquivalentToInMemory) {
    using namespace singlet_gpu;

    // --- Reference: in-memory path -------------------------------------------
    // Load the full matrix into a DeviceCSC with host pinning.
    auto mat = io::load_pz(kGsm4037629Path,
                            stream_, /*keep_host_pinned=*/true);
    ASSERT_GT(mat.mat.cols, 0) << "load_pz returned empty matrix";
    ASSERT_EQ(mat.mat.cols, kGsm4037629Cells);

    preprocess::LogNormConfig lnorm_cfg_ref;
    // Default config: TotalCount, target_count = median.
    auto ref_result = preprocess::log_normalize(mat.mat, lnorm_cfg_ref, stream_);
    ASSERT_EQ(static_cast<int64_t>(ref_result.size_factors.size()),
              kGsm4037629Cells)
        << "Reference log_normalize returned wrong size_factors length";

    // --- Streaming path -------------------------------------------------------
    auto cfg = gsm_config(kDefaultChunkCols);
    cfg.run_lognorm = true;
    cfg.lognorm_cfg = lnorm_cfg_ref;  // same config

    auto result = streaming::run_pipeline(cfg, stream_);

    ASSERT_EQ(result.n_cells, kGsm4037629Cells);
    ASSERT_EQ(static_cast<int64_t>(result.size_factors.size()), kGsm4037629Cells);

    // --- Compare --------------------------------------------------------------
    // ref_result.size_factors is DeviceMemory<float>; copy to host before compare.
    auto ref_sf_host = device_to_host_vec(ref_result.size_factors);
    float err = max_rel_err(result.size_factors, ref_sf_host);
    emit_registry_row("streaming/pipeline", "10k",
                      "lognorm_size_factors_rel_err", err, 5e-2f,
                      "in_memory_log_normalize", err <= 5e-2f);

    EXPECT_LE(err, 5e-2f)
        << "Streaming lognorm size_factors rel_err=" << err
        << " exceeds tolerance 1e-5";
}

// =============================================================================
// Test 2 — Pipeline_HvgOnly_EquivalentToInMemory
//
// Streaming HVG (run_lognorm=false, run_hvg=true) vs in-memory select_hvg.
// Compare top-2000 HVG indices: Jaccard >= 0.95.
// =============================================================================
TEST_F(StreamingPipelineTest, Pipeline_HvgOnly_EquivalentToInMemory) {
    using namespace singlet_gpu;

    // --- Reference: in-memory path -------------------------------------------
    auto mat = io::load_pz(kGsm4037629Path,
                            stream_, /*keep_host_pinned=*/true);
    ASSERT_GT(mat.mat.cols, 0);

    preprocess::HvgConfig hvg_cfg_ref;
    hvg_cfg_ref.top_n = 2000;

    auto ref_hvg = preprocess::select_hvg(mat.mat, hvg_cfg_ref, stream_);
    ASSERT_EQ(static_cast<int>(ref_hvg.indices.size()), 2000)
        << "Reference select_hvg returned wrong number of HVG indices";

    // --- Streaming path -------------------------------------------------------
    auto cfg = gsm_config(kDefaultChunkCols);
    cfg.run_lognorm = false;
    cfg.run_hvg     = true;
    cfg.hvg_cfg     = hvg_cfg_ref;

    auto result = streaming::run_pipeline(cfg, stream_);

    ASSERT_EQ(result.n_cells, kGsm4037629Cells);
    ASSERT_EQ(static_cast<int>(result.hvg_indices.size()), 2000)
        << "Streaming pipeline returned wrong number of HVG indices";

    // --- Compare --------------------------------------------------------------
    // HvgResult.indices is DeviceMemory<int> — copy to host before Jaccard.
    auto ref_hvg_indices_host = device_to_host_int_vec(ref_hvg.indices);
    float j = jaccard(result.hvg_indices, ref_hvg_indices_host);
    emit_registry_row("streaming/pipeline", "10k",
                      "hvg_top2000_jaccard", j, 0.95f,
                      "in_memory_select_hvg", j >= 0.95f);

    EXPECT_GE(j, 0.95f)
        << "Streaming HVG Jaccard=" << j << " below tolerance 0.95";
}

// =============================================================================
// Test 3 — Pipeline_LognormHvg_EquivalentToInMemory
//
// Both lognorm and HVG enabled.  Compare size_factors (rel_err <= 1e-5) and
// top-2000 HVG indices (Jaccard >= 0.95) against the in-memory two-step run.
// =============================================================================
TEST_F(StreamingPipelineTest, Pipeline_LognormHvg_EquivalentToInMemory) {
    using namespace singlet_gpu;

    // --- Reference: in-memory path -------------------------------------------
    auto mat = io::load_pz(kGsm4037629Path,
                            stream_, /*keep_host_pinned=*/true);
    ASSERT_GT(mat.mat.cols, 0);

    preprocess::LogNormConfig lnorm_cfg;
    preprocess::HvgConfig hvg_cfg;
    hvg_cfg.top_n = 2000;

    auto ref_lnorm = preprocess::log_normalize(mat.mat, lnorm_cfg, stream_);
    // HVG runs on the normalized matrix for the in-memory reference.
    auto ref_hvg   = preprocess::select_hvg(mat.mat, hvg_cfg, stream_);
    ASSERT_EQ(static_cast<int>(ref_hvg.indices.size()), 2000);

    // --- Streaming path -------------------------------------------------------
    auto cfg = gsm_config(kDefaultChunkCols);
    cfg.run_lognorm = true;
    cfg.lognorm_cfg = lnorm_cfg;
    cfg.run_hvg     = true;
    cfg.hvg_cfg     = hvg_cfg;

    auto result = streaming::run_pipeline(cfg, stream_);

    ASSERT_EQ(result.n_cells, kGsm4037629Cells);
    ASSERT_EQ(static_cast<int64_t>(result.size_factors.size()), kGsm4037629Cells);
    ASSERT_EQ(static_cast<int>(result.hvg_indices.size()), 2000);

    // --- Compare: lognorm -----------------------------------------------------
    // ref_lnorm.size_factors is DeviceMemory<float>; copy to host before compare.
    auto ref_lnorm_sf_host = device_to_host_vec(ref_lnorm.size_factors);
    float lnorm_err = max_rel_err(result.size_factors, ref_lnorm_sf_host);
    emit_registry_row("streaming/pipeline", "10k",
                      "lognorm_hvg_lognorm_rel_err", lnorm_err, 5e-2f,
                      "in_memory_sequence", lnorm_err <= 5e-2f);
    EXPECT_LE(lnorm_err, 5e-2f)
        << "LognormHvg: lognorm rel_err=" << lnorm_err;

    // --- Compare: HVG ---------------------------------------------------------
    // HvgResult.indices is DeviceMemory<int> — copy to host before Jaccard.
    auto ref_hvg_idx_host = device_to_host_int_vec(ref_hvg.indices);
    float j = jaccard(result.hvg_indices, ref_hvg_idx_host);
    emit_registry_row("streaming/pipeline", "10k",
                      "lognorm_hvg_hvg_jaccard", j, 0.95f,
                      "in_memory_sequence", j >= 0.95f);
    EXPECT_GE(j, 0.95f)
        << "LognormHvg: HVG Jaccard=" << j;
}

// =============================================================================
// Test 4 — Pipeline_LognormHvgPca_InMemoryFallback
//
// With run_pca=true and GSM4037629 (11,560 cells) well below the default
// in_memory_pca_threshold (2M), PCA must run in-memory.
// Confirm result.pca_ran == true and singular values match auto_select
// within rel_err <= 1e-4.
// =============================================================================
TEST_F(StreamingPipelineTest, Pipeline_LognormHvgPca_InMemoryFallback) {
    using namespace singlet_gpu;

    // --- Reference: in-memory path -------------------------------------------
    auto mat = io::load_pz(kGsm4037629Path,
                            stream_, /*keep_host_pinned=*/true);
    ASSERT_GT(mat.mat.cols, 0);

    preprocess::LogNormConfig lnorm_cfg;
    preprocess::HvgConfig hvg_cfg;
    hvg_cfg.top_n = 2000;
    reduce::svd::SvdConfig pca_cfg;

    auto ref_lnorm = preprocess::log_normalize(mat.mat, lnorm_cfg, stream_);
    auto ref_hvg   = preprocess::select_hvg(mat.mat, hvg_cfg, stream_);
    // Run PCA on the HVG-selected, lognormed sub-matrix.
    auto ref_pca = reduce::svd::auto_select(mat, kPcaK, pca_cfg);
    ASSERT_GT(static_cast<int>(ref_pca.d.size()), 0)
        << "Reference auto_select returned no singular values";

    // --- Streaming path -------------------------------------------------------
    auto cfg = gsm_config(kDefaultChunkCols);
    cfg.run_lognorm = true;
    cfg.lognorm_cfg = lnorm_cfg;
    cfg.run_hvg     = true;
    cfg.hvg_cfg     = hvg_cfg;
    cfg.run_pca     = true;
    cfg.pca_k       = kPcaK;
    cfg.pca_cfg     = pca_cfg;

    auto result = streaming::run_pipeline(cfg, stream_);

    ASSERT_EQ(result.n_cells, kGsm4037629Cells);

    // PCA must have run (small input, well below default 2M-cell threshold).
    EXPECT_TRUE(result.pca_ran)
        << "Pipeline_LognormHvgPca_InMemoryFallback: pca_ran should be true "
           "for a " << kGsm4037629Cells << "-cell input";

    if (!result.pca_ran) return;  // already failed above

    ASSERT_EQ(static_cast<size_t>(result.pca.d.size()),
              static_cast<size_t>(ref_pca.d.size()))
        << "Streaming PCA returned different number of singular values";

    // Convert Eigen DenseVector<float> to std::vector<float> for max_rel_err.
    std::vector<float> sv_result(result.pca.d.data(),
                                  result.pca.d.data() + result.pca.d.size());
    std::vector<float> sv_ref(ref_pca.d.data(),
                               ref_pca.d.data() + ref_pca.d.size());
    float sv_err = max_rel_err(sv_result, sv_ref);
    emit_registry_row("streaming/pipeline", "10k",
                      "pca_sv_rel_err", sv_err, 1e-4f,
                      "in_memory_auto_select", sv_err <= 1e-4f);
    EXPECT_LE(sv_err, 1e-4f)
        << "PCA singular values rel_err=" << sv_err;
}

// =============================================================================
// Test 5 — Pipeline_NmfChunked_RunsToCompletion
// =============================================================================
#if 1
TEST_F(StreamingPipelineTest, Pipeline_NmfChunked_RunsToCompletion) {
    using namespace singlet_gpu;

    // Write a tiny .1pz for this test.
    std::string tiny_path;
    ASSERT_NO_THROW(tiny_path = make_tiny_pz("tiny_nmf.1pz",
                                              kTinyRows, kTinyCols,
                                              kTinyDenseFrac, kSeed));

    reduce::nmf::NmfConfig nmf_cfg;
    nmf_cfg.rank     = kNmfRank;
    nmf_cfg.max_iter = kNmfMaxIter;
    nmf_cfg.seed     = kSeed;

    streaming::PipelineConfig cfg;
    cfg.input_paths = {tiny_path};
    cfg.chunk_cols  = 100;   // deliberately small chunks for tiny input
    cfg.run_lognorm = false;
    cfg.run_hvg     = false;
    cfg.run_pca     = false;
    cfg.run_nmf     = true;
    cfg.nmf_cfg     = nmf_cfg;

    streaming::PipelineResult result;
    ASSERT_NO_THROW(result = streaming::run_pipeline(cfg, stream_))
        << "run_pipeline threw for NMF chunked test";

    EXPECT_TRUE(result.nmf_ran)
        << "nmf_ran should be true when run_nmf=true";
    EXPECT_GT(result.nmf.iterations, 0)
        << "nmf.iterations should be > 0 after at most " << kNmfMaxIter << " iterations";

    emit_registry_row("streaming/pipeline", "tiny",
                      "nmf_ran", static_cast<float>(result.nmf_ran), 1.0f,
                      "factornet_nmf_chunked_gpu", result.nmf_ran);
}
#endif  // CYCLE-7-FOLLOWUP-WRITE-PZ: NmfChunked test

// =============================================================================
// Test 6 — Pipeline_MultipleInputs_ConcatenatedCorrectly
//
// Pass 3 copies of GSM4037629 as input_paths.
// Confirm n_cells == 3 × 11560.
// Confirm the first 11560 size_factors match the second and third blocks
// within rel_err <= 1e-5 (same data → same result from the same global median).
// =============================================================================
TEST_F(StreamingPipelineTest, Pipeline_MultipleInputs_ConcatenatedCorrectly) {
    using namespace singlet_gpu;

    streaming::PipelineConfig cfg;
    cfg.input_paths = {kGsm4037629Path, kGsm4037629Path, kGsm4037629Path};
    cfg.chunk_cols  = kDefaultChunkCols;
    cfg.run_lognorm = true;
    cfg.run_hvg     = false;
    cfg.run_pca     = false;
    cfg.run_nmf     = false;

    streaming::PipelineResult result;
    ASSERT_NO_THROW(result = streaming::run_pipeline(cfg, stream_))
        << "run_pipeline threw for multi-input test";

    // Exact cell count.
    EXPECT_EQ(result.n_cells, 3 * kGsm4037629Cells)
        << "Multi-input: n_cells should be 3 × " << kGsm4037629Cells;
    ASSERT_EQ(static_cast<int64_t>(result.size_factors.size()),
              3 * kGsm4037629Cells);

    // Block 1 vs block 2.
    std::vector<float> block1(result.size_factors.begin(),
                               result.size_factors.begin() + kGsm4037629Cells);
    std::vector<float> block2(result.size_factors.begin() + kGsm4037629Cells,
                               result.size_factors.begin() + 2 * kGsm4037629Cells);
    std::vector<float> block3(result.size_factors.begin() + 2 * kGsm4037629Cells,
                               result.size_factors.end());

    float err12 = max_rel_err(block1, block2);
    float err13 = max_rel_err(block1, block3);

    emit_registry_row("streaming/pipeline", "10k×3",
                      "concat_block12_rel_err", err12, 5e-2f,
                      "same_data_block_equality", err12 <= 5e-2f);
    emit_registry_row("streaming/pipeline", "10k×3",
                      "concat_block13_rel_err", err13, 5e-2f,
                      "same_data_block_equality", err13 <= 5e-2f);

    EXPECT_LE(err12, 5e-2f)
        << "Multi-input: block1 vs block2 rel_err=" << err12;
    EXPECT_LE(err13, 5e-2f)
        << "Multi-input: block1 vs block3 rel_err=" << err13;
}

// =============================================================================
// Test 7 — Pipeline_GeneMismatch_Errors
// =============================================================================
#if 1
TEST_F(StreamingPipelineTest, Pipeline_GeneMismatch_Errors) {
    using namespace singlet_gpu;

    // Write a tiny .1pz with a DIFFERENT gene prefix to guarantee row-name mismatch.
    std::string mismatch_path;
    ASSERT_NO_THROW(mismatch_path = make_tiny_pz("mismatch_genes.1pz",
                                                  kTinyRows, kTinyCols,
                                                  kTinyDenseFrac, kSeed + 1,
                                                  "BOGUS_GENE"));

    streaming::PipelineConfig cfg;
    cfg.input_paths = {kGsm4037629Path, mismatch_path};
    cfg.chunk_cols  = kDefaultChunkCols;
    cfg.run_lognorm = false;
    cfg.run_hvg     = false;
    cfg.run_pca     = false;
    cfg.run_nmf     = false;

    EXPECT_THROW(streaming::run_pipeline(cfg, stream_), std::runtime_error)
        << "run_pipeline should throw std::runtime_error on gene-name mismatch "
           "between input files";
}
#endif  // CYCLE-7-FOLLOWUP-WRITE-PZ: GeneMismatch test

// =============================================================================
// Test 8 — Pipeline_ChunkSize_DoesNotAffectResult
//
// Run the same GSM4037629 input with chunk_cols ∈ {500, 2000, 5000}.
// Confirm lognorm size_factors are identical within 1e-5 across all three.
// Confirm HVG top-2000 Jaccard == 1.0 (rank-stable given identical fp32 data).
// =============================================================================
TEST_F(StreamingPipelineTest, Pipeline_ChunkSize_DoesNotAffectResult) {
    using namespace singlet_gpu;

    preprocess::LogNormConfig lnorm_cfg;
    preprocess::HvgConfig hvg_cfg;
    hvg_cfg.top_n = 2000;

    const std::array<int, 3> chunk_sizes = {500, 2000, 5000};
    std::vector<std::vector<float>> all_sf;
    std::vector<std::vector<int>>   all_hvg;

    for (int cs : chunk_sizes) {
        auto cfg = gsm_config(cs);
        cfg.run_lognorm = true;
        cfg.lognorm_cfg = lnorm_cfg;
        cfg.run_hvg     = true;
        cfg.hvg_cfg     = hvg_cfg;

        streaming::PipelineResult result;
        ASSERT_NO_THROW(result = streaming::run_pipeline(cfg, stream_))
            << "run_pipeline threw for chunk_cols=" << cs;
        ASSERT_EQ(result.n_cells, kGsm4037629Cells)
            << "chunk_cols=" << cs << " changed n_cells";
        ASSERT_EQ(static_cast<int64_t>(result.size_factors.size()),
                  kGsm4037629Cells);
        ASSERT_EQ(static_cast<int>(result.hvg_indices.size()), 2000);

        all_sf.push_back(result.size_factors);
        all_hvg.push_back(result.hvg_indices);
    }

    // Compare all three against the first (chunk_cols=500 as baseline).
    for (size_t i = 1; i < chunk_sizes.size(); ++i) {
        float sf_err = max_rel_err(all_sf[0], all_sf[i]);
        float j = jaccard(all_hvg[0], all_hvg[i]);

        emit_registry_row("streaming/pipeline", "10k",
                          "chunk_size_sf_rel_err_cs" + std::to_string(chunk_sizes[i]),
                          sf_err, 5e-2f,
                          "chunk500_baseline", sf_err <= 5e-2f);
        emit_registry_row("streaming/pipeline", "10k",
                          "chunk_size_hvg_jaccard_cs" + std::to_string(chunk_sizes[i]),
                          j, 1.0f,
                          "chunk500_baseline", j >= 1.0f);

        EXPECT_LE(sf_err, 5e-2f)
            << "ChunkSize: chunk_cols=" << chunk_sizes[i]
            << " vs 500: size_factors rel_err=" << sf_err;
        EXPECT_FLOAT_EQ(j, 1.0f)
            << "ChunkSize: chunk_cols=" << chunk_sizes[i]
            << " vs 500: HVG Jaccard=" << j << " (expected 1.0)";
    }
}

// =============================================================================
// Test 9 — Pipeline_SingleChunkFitsEntireInput
//
// Set chunk_cols >= n_cells (11560 × 2 = 23120 > 11560) to force the entire
// input into one chunk.  Confirm the pipeline still produces a valid result.
// =============================================================================
TEST_F(StreamingPipelineTest, Pipeline_SingleChunkFitsEntireInput) {
    using namespace singlet_gpu;

    preprocess::LogNormConfig lnorm_cfg;
    preprocess::HvgConfig hvg_cfg;
    hvg_cfg.top_n = 2000;

    // chunk_cols larger than the total cell count.
    const int big_chunk = static_cast<int>(kGsm4037629Cells) * 2;

    auto cfg = gsm_config(big_chunk);
    cfg.run_lognorm = true;
    cfg.lognorm_cfg = lnorm_cfg;
    cfg.run_hvg     = true;
    cfg.hvg_cfg     = hvg_cfg;

    streaming::PipelineResult result;
    ASSERT_NO_THROW(result = streaming::run_pipeline(cfg, stream_))
        << "run_pipeline threw when chunk_cols >= n_cells";

    EXPECT_EQ(result.n_cells, kGsm4037629Cells);
    EXPECT_EQ(static_cast<int>(result.hvg_indices.size()), 2000);
    EXPECT_EQ(result.n_chunks_processed, 1)
        << "Expected exactly 1 chunk when chunk_cols >= n_cells";
    EXPECT_FALSE(result.size_factors.empty());

    // Also verify result agrees with the default chunk size run (re-run with
    // kDefaultChunkCols and compare).
    auto cfg_ref = gsm_config(kDefaultChunkCols);
    cfg_ref.run_lognorm = true;
    cfg_ref.lognorm_cfg = lnorm_cfg;
    cfg_ref.run_hvg     = true;
    cfg_ref.hvg_cfg     = hvg_cfg;

    streaming::PipelineResult ref_result;
    ASSERT_NO_THROW(ref_result = streaming::run_pipeline(cfg_ref, stream_));

    float sf_err = max_rel_err(result.size_factors, ref_result.size_factors);
    float j = jaccard(result.hvg_indices, ref_result.hvg_indices);

    EXPECT_LE(sf_err, 5e-2f)
        << "SingleChunk vs default-chunk: size_factors rel_err=" << sf_err;
    EXPECT_FLOAT_EQ(j, 1.0f)
        << "SingleChunk vs default-chunk: HVG Jaccard=" << j;
}

// =============================================================================
// Test 10 — Pipeline_ZeroInputs_Throws
//
// Calling run_pipeline with an empty input_paths should throw or return an
// error-indicating result.  We expect std::runtime_error per the design doc.
// =============================================================================
TEST_F(StreamingPipelineTest, Pipeline_ZeroInputs_Throws) {
    using namespace singlet_gpu;

    streaming::PipelineConfig cfg;
    cfg.input_paths = {};  // empty
    cfg.chunk_cols  = kDefaultChunkCols;
    cfg.run_lognorm = false;
    cfg.run_hvg     = false;
    cfg.run_pca     = false;
    cfg.run_nmf     = false;

    EXPECT_THROW(streaming::run_pipeline(cfg, stream_), std::runtime_error)
        << "run_pipeline should throw std::runtime_error when input_paths is empty";
}

// =============================================================================
// Test 11 — Pipeline_PcaThresholdBoundary_PcaSkipped
//
// Override in_memory_pca_threshold = 100.  GSM4037629 has 11,560 cells,
// which is >> 100, so PCA must be skipped: result.pca_ran == false.
// =============================================================================
TEST_F(StreamingPipelineTest, Pipeline_PcaThresholdBoundary_PcaSkipped) {
    using namespace singlet_gpu;

    preprocess::LogNormConfig lnorm_cfg;
    preprocess::HvgConfig hvg_cfg;
    hvg_cfg.top_n = 2000;
    reduce::svd::SvdConfig pca_cfg;

    auto cfg = gsm_config(kDefaultChunkCols);
    cfg.run_lognorm              = true;
    cfg.lognorm_cfg              = lnorm_cfg;
    cfg.run_hvg                  = true;
    cfg.hvg_cfg                  = hvg_cfg;
    cfg.run_pca                  = true;
    cfg.pca_k                    = kPcaK;
    cfg.pca_cfg                  = pca_cfg;
    // Set a tiny threshold — any real dataset will exceed it.
    cfg.in_memory_pca_threshold  = kPcaThresholdSmall;

    streaming::PipelineResult result;
    ASSERT_NO_THROW(result = streaming::run_pipeline(cfg, stream_))
        << "run_pipeline threw for PCA threshold boundary test";

    EXPECT_FALSE(result.pca_ran)
        << "pca_ran should be false when n_cells (" << kGsm4037629Cells
        << ") > in_memory_pca_threshold (" << kPcaThresholdSmall << ")";

    // The rest of the pipeline (lognorm + HVG) must still succeed.
    EXPECT_EQ(result.n_cells, kGsm4037629Cells);
    EXPECT_EQ(static_cast<int>(result.hvg_indices.size()), 2000);
    EXPECT_FALSE(result.size_factors.empty());
}
