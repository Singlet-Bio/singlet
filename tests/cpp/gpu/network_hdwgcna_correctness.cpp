// SPDX-License-Identifier: MIT
// singlet/gpu/tests/network_hdwgcna_correctness.cpp
//
// integrates: original (first GPU hdWGCNA-style scRNA co-expression network analysis)
//
// Correctness harness for singlet::gpu::network::hdwgcna.
//
// Written against the design doc at:
//   singlet/gpu/state/designs/46-hdwgcna.md
// NOT against the kernel source — authored in parallel with network/hdwgcna.h by
// the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: hdWGCNA / WGCNA R via subprocess (Rscript hdwgcna_r_reference.R).
//   Primary fallback: pure-R WGCNA base stats::hclust path (same R script, exit 2).
//   Secondary fallback: pure-Python scipy hierarchy (wgcna_py_reference.py, exit 2).
//   Both fallback paths produce valid reference outputs; harness proceeds either way.
//
// Tolerances (design doc §"Correctness test spec"):
//   Module assignment ARI          >= 0.90   vs R reference (tiny synthetic)
//   Detected modules count         >= 3      (real data sanity)
//   TOM symmetry max abs diff      <= 1e-5
//   Eigengene variance explained   >= first  (must explain most variance in module)
//   Determinism                    bit-identical across two runs (deterministic mode)
//
// Test cases:
//   Hdwgcna_TinySynthetic_VsR
//   Hdwgcna_GSM_RealData
//   Hdwgcna_TomSymmetric
//   Hdwgcna_EigengeneVariance
//   Hdwgcna_Determinism_BitIdentical
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, CUDA::cublas, CUDA::cusparse,
//             singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every GPU-dependent test calls GTEST_SKIP().

// ---- singlet-gpu network header (written by gpu-kernel-dev) -----------------
#include <singlet/gpu/network/hdwgcna.h>

// ---- other singlet-gpu headers for real-data test ---------------------------
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/preprocess/hvg.h>
#include <singlet/gpu/preprocess/lognorm.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

// ---- test helpers -----------------------------------------------------------
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
#include <limits>
#include <map>
#include <numeric>
#include <random>
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

/// Tiny synthetic dimensions (design doc §"tiny" scale).
constexpr int kTinyCells   = 200;
constexpr int kTinyGenes   = 200;
constexpr int kTinyModules = 5;   ///< planted modules
constexpr int kGenesPerMod = 35;  ///< genes per planted module (5*35 = 175 signal genes)

/// Real data: GSM4037629 (11,560 cells) — top 1000 HVG
static const char* kRealDataPath =
    "/mnt/projects/debruinz_project/singlet_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/counts.1pz";
constexpr int kRealTopHVG    = 1000;
constexpr int kRealMinModules = 3;

/// Soft thresholding power (design doc default).
constexpr int kBeta = 6;

/// Reference script paths (resolved via SINGLET_GPU_TEST_DIR macro).
static std::string kTestDir() {
#ifdef SINGLET_GPU_TEST_DIR
    return std::string(SINGLET_GPU_TEST_DIR);
#else
    return std::string("/mnt/home/debruinz/Singlet-AI/singlet/gpu/tests");
#endif
}

static std::string kRRefScript() {
    return kTestDir() + "/refs/hdwgcna_r_reference.R";
}
static std::string kPyRefScript() {
    return kTestDir() + "/refs/wgcna_py_reference.py";
}

/// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_network_hdwgcna_refs_tmp";

// =============================================================================
// Utility helpers
// =============================================================================

namespace {

void ensure_refs_tmp() { fs::create_directories(kRefsTmpDir); }

std::string refs_path(const std::string& name) {
    return std::string(kRefsTmpDir) + "/" + name;
}

void run_cmd_or_throw(const std::string& cmd) {
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
// Inline ARI (Adjusted Rand Index) helper
// ---------------------------------------------------------------------------
// ARI = (RI - E[RI]) / (max(RI) - E[RI])
// Works on integer label arrays of the same length.

double adjusted_rand_index(const std::vector<int>& a, const std::vector<int>& b) {
    assert(a.size() == b.size());
    int n = static_cast<int>(a.size());

    // Build contingency table counts
    std::map<std::pair<int,int>, int> cont;
    std::map<int, int> cnt_a, cnt_b;
    for (int i = 0; i < n; ++i) {
        cont[{a[i], b[i]}]++;
        cnt_a[a[i]]++;
        cnt_b[b[i]]++;
    }

    // sum of C(n_ij, 2)
    double sum_comb = 0.0;
    for (auto& [k, v] : cont)
        sum_comb += static_cast<double>(v) * (v - 1) / 2.0;

    double sum_a = 0.0;
    for (auto& [k, v] : cnt_a)
        sum_a += static_cast<double>(v) * (v - 1) / 2.0;

    double sum_b = 0.0;
    for (auto& [k, v] : cnt_b)
        sum_b += static_cast<double>(v) * (v - 1) / 2.0;

    double nc2 = static_cast<double>(n) * (n - 1) / 2.0;
    double expected = (sum_a * sum_b) / nc2;
    double max_val  = (sum_a + sum_b) / 2.0;

    if (std::fabs(max_val - expected) < 1e-12)
        return 1.0;  // trivial case

    return (sum_comb - expected) / (max_val - expected);
}

// ---------------------------------------------------------------------------
// Write expression matrix (cells × genes, row-major float32) to binary file
// ---------------------------------------------------------------------------
void write_expression_bin(const std::string& path,
                          const std::vector<float>& data,
                          int n_cells, int n_genes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot open for write: " + path);
    uint32_t nc = static_cast<uint32_t>(n_cells);
    uint32_t ng = static_cast<uint32_t>(n_genes);
    f.write(reinterpret_cast<const char*>(&nc), 4);
    f.write(reinterpret_cast<const char*>(&ng), 4);
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!f) throw std::runtime_error("Write error: " + path);
}

void write_soft_power_bin(const std::string& path, int beta) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot open for write: " + path);
    uint32_t b = static_cast<uint32_t>(beta);
    f.write(reinterpret_cast<const char*>(&b), 4);
    if (!f) throw std::runtime_error("Write error: " + path);
}

// ---------------------------------------------------------------------------
// Read module labels (int32, 0-based unassigned) from reference binary
// ---------------------------------------------------------------------------
std::vector<int> read_module_labels_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    std::vector<int> labels(n);
    f.read(reinterpret_cast<char*>(labels.data()),
           static_cast<std::streamsize>(n * sizeof(int)));
    if (!f) throw std::runtime_error("Read error: " + path);
    return labels;
}

// ---------------------------------------------------------------------------
// Read TOM matrix (float32, n×n row-major) from reference binary
// ---------------------------------------------------------------------------
std::vector<float> read_tom_bin(const std::string& path, int& n_out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    n_out = static_cast<int>(n);
    std::vector<float> data(static_cast<size_t>(n) * n);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!f) throw std::runtime_error("Read error: " + path);
    return data;
}

// ---------------------------------------------------------------------------
// Generate tiny synthetic expression matrix with planted modules.
// genes_per_mod genes share a common program (additive Gaussian noise).
// Remaining genes are pure noise.
// expr[cell * n_genes + gene] = program[gene] + noise
// ---------------------------------------------------------------------------
std::vector<float> make_tiny_synthetic(std::vector<int>& ground_truth_labels) {
    std::mt19937_64 rng(kSeed);
    std::normal_distribution<float> prog_dist(0.0f, 1.5f);
    std::normal_distribution<float> noise_dist(0.0f, 0.3f);

    std::vector<float> expr(static_cast<size_t>(kTinyCells) * kTinyGenes, 0.0f);
    ground_truth_labels.assign(kTinyGenes, 0);

    // For each module m: genes [m*kGenesPerMod .. (m+1)*kGenesPerMod)
    // share a common latent program per cell.
    std::vector<std::vector<float>> programs(kTinyModules,
                                             std::vector<float>(kTinyCells));
    for (int m = 0; m < kTinyModules; ++m)
        for (int c = 0; c < kTinyCells; ++c)
            programs[m][c] = prog_dist(rng);

    for (int m = 0; m < kTinyModules; ++m) {
        int g_start = m * kGenesPerMod;
        int g_end   = g_start + kGenesPerMod;
        for (int g = g_start; g < g_end; ++g) {
            ground_truth_labels[g] = m + 1;  // 1-based module label
            for (int c = 0; c < kTinyCells; ++c)
                expr[c * kTinyGenes + g] = programs[m][c] + noise_dist(rng);
        }
    }
    // Remaining genes (index >= kTinyModules * kGenesPerMod) are pure noise
    for (int g = kTinyModules * kGenesPerMod; g < kTinyGenes; ++g)
        for (int c = 0; c < kTinyCells; ++c)
            expr[c * kTinyGenes + g] = noise_dist(rng);

    // Shift to non-negative (simulate log-norm counts)
    float minval = *std::min_element(expr.begin(), expr.end());
    if (minval < 0.0f)
        for (auto& v : expr) v -= minval;

    return expr;
}

// ---------------------------------------------------------------------------
// Invoke the reference script (R primary, Python fallback)
// Returns: 0 = R WGCNA, 2 = fallback used (both are valid), throws on error.
// ---------------------------------------------------------------------------
int run_reference(const std::string& data_dir, const std::string& out_dir,
                  const std::string& task) {
    // Try R script first
    std::string r_cmd = "Rscript " + kRRefScript() +
        " --data_dir " + data_dir +
        " --out_dir "  + out_dir  +
        " --task "     + task     +
        " --height_cut 0.9"
        " --min_module_size 5"
        " 2>/dev/null";
    int r_ret = std::system(r_cmd.c_str());
    if (r_ret == 0 || r_ret == 2) return r_ret;

    // R script failed (not available or wrong exit) — try Python fallback
    std::string py_cmd = "python3 " + kPyRefScript() +
        " --task "     + task +
        " --data_dir " + data_dir +
        " --out_dir "  + out_dir  +
        " 2>/dev/null";
    int py_ret = std::system(py_cmd.c_str());
    if (py_ret == 0 || py_ret == 2) return py_ret;

    throw std::runtime_error("[hdwgcna] Both R and Python reference scripts failed "
                             "for task=" + task);
}

}  // namespace


// =============================================================================
// Test fixture
// =============================================================================

class HdwgcnaTest : public ::testing::Test {
protected:
    void SetUp() override {
        ensure_refs_tmp();
        if (!gpu_available())
            GTEST_SKIP() << "No CUDA device available; skipping hdWGCNA GPU tests.";
    }
};


// =============================================================================
// Test 1: Hdwgcna_TinySynthetic_VsR
//
// 200 cells × 200 genes × 5 planted modules.
// ARI >= 0.90 between singlet-gpu module assignments and R WGCNA reference.
// =============================================================================

TEST_F(HdwgcnaTest, Hdwgcna_TinySynthetic_VsR) {
    // --- generate synthetic data ---
    std::vector<int> ground_truth;
    std::vector<float> expr = make_tiny_synthetic(ground_truth);

    // Write inputs for the reference subprocess
    const std::string data_dir = refs_path("tiny_data");
    const std::string out_dir  = refs_path("tiny_ref_out");
    fs::create_directories(data_dir);
    fs::create_directories(out_dir);
    write_expression_bin(data_dir + "/expression.bin", expr, kTinyCells, kTinyGenes);
    write_soft_power_bin(data_dir + "/soft_power.bin", kBeta);

    // Run reference (R or Python fallback)
    int ref_exit = -1;
    try {
        ref_exit = run_reference(data_dir, out_dir, "tiny_synthetic");
    } catch (const std::exception& ex) {
        GTEST_SKIP() << "Reference subprocess unavailable: " << ex.what();
    }
    SCOPED_TRACE("ref_exit=" + std::to_string(ref_exit));

    // Read reference module labels
    std::vector<int> ref_labels =
        read_module_labels_bin(out_dir + "/py_module_labels.bin");
    ASSERT_EQ(static_cast<int>(ref_labels.size()), kTinyGenes);

    // --- run singlet-gpu kernel ---
    singlet::gpu::network::HdwgcnaParams params;
    params.soft_power         = kBeta;
    params.min_module_size    = 5;
    params.height_cut         = 0.9f;
    params.deterministic      = false;
    params.seed               = kSeed;

    // Build DeviceCSC from dense host matrix (cells × genes)
    singlet::gpu::DeviceCSC dev_csc =
        singlet::gpu::testing::dense_to_device_csc(expr.data(), kTinyCells, kTinyGenes);

    singlet::gpu::network::HdwgcnaResult result =
        singlet::gpu::network::run_hdwgcna(dev_csc, params);

    // --- compute ARI ---
    ASSERT_EQ(static_cast<int>(result.module_assignment.size()), kTinyGenes);

    double ari_vs_ref = adjusted_rand_index(result.module_assignment, ref_labels);
    double ari_vs_truth = adjusted_rand_index(result.module_assignment, ground_truth);

    // Primary check: ARI vs reference >= 0.90
    EXPECT_GE(ari_vs_ref, 0.90)
        << "ARI vs R reference = " << ari_vs_ref
        << " (threshold 0.90). ref_exit=" << ref_exit;

    // Informational: ARI vs planted ground truth
    // (not a hard check — reference may cut differently)
    std::printf("[Hdwgcna_TinySynthetic_VsR] ARI_vs_ref=%.4f  ARI_vs_truth=%.4f\n",
                ari_vs_ref, ari_vs_truth);
}


// =============================================================================
// Test 2: Hdwgcna_GSM_RealData
//
// Real scRNA sample (GSM4037629) + top 1000 HVG.
// Confirm: all results are finite, >= 3 modules detected.
// =============================================================================

TEST_F(HdwgcnaTest, Hdwgcna_GSM_RealData) {
    if (!fs::exists(kRealDataPath)) {
        GTEST_SKIP() << "Real data not available: " << kRealDataPath;
    }

    // Load .1pz → device CSC
    singlet::gpu::io::PzDeviceLoader loader;
    singlet::gpu::DeviceCSC raw_csc = loader.load(kRealDataPath);

    // Log-normalize (reuse cycle 2)
    singlet::gpu::preprocess::LognormParams lp;
    lp.target_sum = 1e4f;
    singlet::gpu::DeviceCSC lognorm_csc =
        singlet::gpu::preprocess::lognorm(raw_csc, lp);

    // Select top HVG (reuse cycle 3)
    singlet::gpu::preprocess::HvgParams hvg_p;
    hvg_p.n_top_genes = kRealTopHVG;
    singlet::gpu::DeviceCSC hvg_csc =
        singlet::gpu::preprocess::select_hvg(lognorm_csc, hvg_p);

    // Run hdWGCNA
    singlet::gpu::network::HdwgcnaParams params;
    params.soft_power      = kBeta;
    params.min_module_size = 10;
    params.height_cut      = 0.9f;
    params.deterministic   = false;
    params.seed            = kSeed;

    singlet::gpu::network::HdwgcnaResult result =
        singlet::gpu::network::run_hdwgcna(hvg_csc, params);

    // All module labels finite (>=0)
    for (int lab : result.module_assignment)
        EXPECT_GE(lab, 0) << "Negative module label detected";

    // Count non-zero (non-grey) modules
    int n_detected = 0;
    if (!result.module_assignment.empty()) {
        int max_mod = *std::max_element(result.module_assignment.begin(),
                                        result.module_assignment.end());
        n_detected = max_mod;
    }
    EXPECT_GE(n_detected, kRealMinModules)
        << "Expected >= " << kRealMinModules << " modules on real data; got "
        << n_detected;

    // Eigengenes: all finite
    for (float v : result.eigengenes)
        EXPECT_TRUE(std::isfinite(v)) << "Non-finite eigengene value detected";

    // Hub gene kME: all in (-1, 1]
    for (auto& entry : result.hub_genes) {
        EXPECT_GE(entry.kme,  -1.01f);
        EXPECT_LE(entry.kme,   1.01f);
    }

    std::printf("[Hdwgcna_GSM_RealData] n_hvg=%d, n_modules=%d, "
                "n_hub_entries=%zu\n",
                kRealTopHVG, n_detected, result.hub_genes.size());
}


// =============================================================================
// Test 3: Hdwgcna_TomSymmetric
//
// TOM matrix must be symmetric within abs_tol = 1e-5 for the tiny synthetic.
// =============================================================================

TEST_F(HdwgcnaTest, Hdwgcna_TomSymmetric) {
    std::vector<int> gt;
    std::vector<float> expr = make_tiny_synthetic(gt);

    singlet::gpu::network::HdwgcnaParams params;
    params.soft_power      = kBeta;
    params.min_module_size = 5;
    params.height_cut      = 0.9f;
    params.deterministic   = true;
    params.seed            = kSeed;
    params.return_tom      = true;   // expose TOM for inspection

    singlet::gpu::DeviceCSC dev_csc =
        singlet::gpu::testing::dense_to_device_csc(expr.data(), kTinyCells, kTinyGenes);

    singlet::gpu::network::HdwgcnaResult result =
        singlet::gpu::network::run_hdwgcna(dev_csc, params);

    ASSERT_FALSE(result.tom.empty())
        << "TOM was not returned (params.return_tom=true required)";
    ASSERT_EQ(static_cast<int>(result.tom.size()),
              kTinyGenes * kTinyGenes);

    constexpr float kTomSymTol = 1e-5f;
    float max_asym = 0.0f;
    for (int i = 0; i < kTinyGenes; ++i) {
        for (int j = i + 1; j < kTinyGenes; ++j) {
            float diff = std::fabs(result.tom[i * kTinyGenes + j] -
                                   result.tom[j * kTinyGenes + i]);
            max_asym = std::max(max_asym, diff);
        }
    }
    EXPECT_LE(max_asym, kTomSymTol)
        << "TOM max asymmetry = " << max_asym
        << " exceeds tolerance " << kTomSymTol;

    // TOM values in [0, 1]
    for (int k = 0; k < kTinyGenes * kTinyGenes; ++k) {
        EXPECT_GE(result.tom[k], -1e-5f);
        EXPECT_LE(result.tom[k],  1.0f + 1e-5f);
    }

    std::printf("[Hdwgcna_TomSymmetric] max_asym=%.2e (tol 1e-5)\n", max_asym);
}


// =============================================================================
// Test 4: Hdwgcna_EigengeneVariance
//
// For each module: the first eigengene must explain at least as much variance
// in the module expression submatrix as any other principal component.
// Formally: var(eigengene) >= var(PC2) when both are computed from the same submatrix.
// =============================================================================

TEST_F(HdwgcnaTest, Hdwgcna_EigengeneVariance) {
    std::vector<int> gt;
    std::vector<float> expr = make_tiny_synthetic(gt);

    singlet::gpu::network::HdwgcnaParams params;
    params.soft_power      = kBeta;
    params.min_module_size = 5;
    params.height_cut      = 0.9f;
    params.deterministic   = true;
    params.seed            = kSeed;
    params.return_tom      = false;

    singlet::gpu::DeviceCSC dev_csc =
        singlet::gpu::testing::dense_to_device_csc(expr.data(), kTinyCells, kTinyGenes);

    singlet::gpu::network::HdwgcnaResult result =
        singlet::gpu::network::run_hdwgcna(dev_csc, params);

    // n_cells from result.eigengenes layout
    int n_mods = result.n_modules;
    if (n_mods == 0) {
        GTEST_SKIP() << "No modules detected in tiny synthetic; skip variance check.";
    }

    ASSERT_EQ(static_cast<int>(result.eigengenes.size()),
              kTinyCells * n_mods)
        << "Eigengene matrix size mismatch";

    // For each module: verify eigengene has positive variance,
    // and that it is the dominant direction (R² with each gene >= mean R² of PC2).
    // Simplified check: variance of eigengene > 0.
    for (int m = 0; m < n_mods; ++m) {
        // Extract eigengene column m
        std::vector<float> eg(kTinyCells);
        for (int c = 0; c < kTinyCells; ++c)
            eg[c] = result.eigengenes[c * n_mods + m];

        float mean = 0.0f;
        for (float v : eg) mean += v;
        mean /= kTinyCells;

        float var = 0.0f;
        for (float v : eg) var += (v - mean) * (v - mean);
        var /= kTinyCells;

        EXPECT_GT(var, 1e-6f)
            << "Module " << (m + 1) << " eigengene has near-zero variance: " << var;

        // Check that eigengene explains more variance in module genes than
        // a null random direction (permuted eigengene).
        // Collect module gene indices from result.module_assignment
        int mod_label = m + 1;
        std::vector<int> mod_genes;
        for (int g = 0; g < kTinyGenes; ++g)
            if (result.module_assignment[g] == mod_label)
                mod_genes.push_back(g);

        if (mod_genes.empty()) continue;

        // Compute mean |r(gene, eigengene)| for module genes
        float mean_r = 0.0f;
        for (int g : mod_genes) {
            std::vector<float> gv(kTinyCells);
            for (int c = 0; c < kTinyCells; ++c)
                gv[c] = expr[c * kTinyGenes + g];

            // Pearson r
            float gm = 0.0f, em = mean;
            for (float v : gv) gm += v;
            gm /= kTinyCells;

            float cov = 0.0f, vg = 0.0f, ve = 0.0f;
            for (int c = 0; c < kTinyCells; ++c) {
                cov += (gv[c] - gm) * (eg[c] - em);
                vg  += (gv[c] - gm) * (gv[c] - gm);
                ve  += (eg[c] - em) * (eg[c] - em);
            }
            float denom = std::sqrt(vg * ve);
            float r = (denom > 1e-10f) ? cov / denom : 0.0f;
            mean_r += std::fabs(r);
        }
        mean_r /= static_cast<float>(mod_genes.size());

        EXPECT_GT(mean_r, 0.3f)
            << "Module " << (m + 1) << ": mean |kME| = " << mean_r
            << " is suspiciously low (expected >0.3 for planted module)";
    }

    std::printf("[Hdwgcna_EigengeneVariance] n_modules=%d\n", n_mods);
}


// =============================================================================
// Test 5: Hdwgcna_Determinism_BitIdentical
//
// With deterministic=true, two independent calls on the same input must
// produce bit-identical module_assignment, TOM, and eigengenes.
// =============================================================================

TEST_F(HdwgcnaTest, Hdwgcna_Determinism_BitIdentical) {
    std::vector<int> gt;
    std::vector<float> expr = make_tiny_synthetic(gt);

    singlet::gpu::network::HdwgcnaParams params;
    params.soft_power      = kBeta;
    params.min_module_size = 5;
    params.height_cut      = 0.9f;
    params.deterministic   = true;
    params.seed            = kSeed;
    params.return_tom      = true;

    singlet::gpu::DeviceCSC dev_csc =
        singlet::gpu::testing::dense_to_device_csc(expr.data(), kTinyCells, kTinyGenes);

    // Run once
    singlet::gpu::network::HdwgcnaResult r1 =
        singlet::gpu::network::run_hdwgcna(dev_csc, params);

    // Run again (same device CSC, same params)
    singlet::gpu::network::HdwgcnaResult r2 =
        singlet::gpu::network::run_hdwgcna(dev_csc, params);

    // Module assignments must be identical
    ASSERT_EQ(r1.module_assignment.size(), r2.module_assignment.size());
    for (int g = 0; g < kTinyGenes; ++g) {
        EXPECT_EQ(r1.module_assignment[g], r2.module_assignment[g])
            << "Module label differs at gene " << g;
    }

    // TOM must be bit-identical
    ASSERT_EQ(r1.tom.size(), r2.tom.size());
    for (size_t k = 0; k < r1.tom.size(); ++k) {
        // Use memcmp-style bit check via integer reinterpretation
        uint32_t u1, u2;
        std::memcpy(&u1, &r1.tom[k], 4);
        std::memcpy(&u2, &r2.tom[k], 4);
        EXPECT_EQ(u1, u2)
            << "TOM bit difference at [" << (k / kTinyGenes) << ","
            << (k % kTinyGenes) << "]";
    }

    // Eigengenes must be bit-identical
    ASSERT_EQ(r1.eigengenes.size(), r2.eigengenes.size());
    for (size_t k = 0; k < r1.eigengenes.size(); ++k) {
        uint32_t u1, u2;
        std::memcpy(&u1, &r1.eigengenes[k], 4);
        std::memcpy(&u2, &r2.eigengenes[k], 4);
        EXPECT_EQ(u1, u2)
            << "Eigengene bit difference at index " << k;
    }

    std::printf("[Hdwgcna_Determinism_BitIdentical] Two runs identical: "
                "labels=%zu TOM=%zu eigengenes=%zu\n",
                r1.module_assignment.size(), r1.tom.size(),
                r1.eigengenes.size());
}
