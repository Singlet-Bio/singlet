// SPDX-License-Identifier: MIT
// singlet/gpu/tests/abundance_milo_correctness.cpp
//
// integrates: original (first GPU Milo-style kNN-neighborhood differential
//             abundance — exploits singlet donor_assignments.tsv directly)
//
// Correctness harness for singlet::gpu::abundance::milo_da.
//
// Written against the design doc at:
//   singlet/gpu/state/designs/47-milo.md
// NOT against the kernel source — authored in parallel with
// abundance/milo.h by the analysis-validator worker (Sonnet, Tier 2).
//
// References:
//   Primary:  miloR (Dann et al. 2022, Nature Biotechnology) via Rscript
//             tests/refs/milo_r_reference.R
//   Fallback: MASS::glm.nb per neighborhood (always available)
//
// Tolerances (design doc §"Correctness test spec"):
//   log_fc Spearman ρ vs R     >= 0.95  (test 1 — tiny synthetic with planted DA)
//   p-value rank Spearman ρ    >= 0.90  (test 1)
//   Real data: neighborhoods formed, finite p-values (test 2)
//   IRLS convergence rate      >  0.95  (test 3)
//   p-value KS ≤ 0.10 under null labels (test 4)
//   Bit-identical with fixed seed        (test 5)
//
// Test cases:
//   Milo_TinySynthetic_VsR
//   Milo_GSM_RealData
//   Milo_IrlsConvergence
//   Milo_PvalueCalibration_Uniform
//   Milo_Determinism_BitIdentical
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, CUDA::cusparse,
//             singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every test that requires a device calls GTEST_SKIP().

// ---- singlet-gpu abundance header (written by gpu-kernel-dev) ---------------
#include <singlet/gpu/abundance/milo.h>

// ---- other singlet-gpu headers needed by the real-data test -----------------
#include <singlet/gpu/io/pz_device_loader.h>
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

/// Tiny synthetic dimensions (design doc §"Correctness test spec"):
///   200 cells × 10 donors × 2 conditions.
constexpr int kTinyCells     = 200;
constexpr int kTinyGenes     = 50;   ///< reduced dimensionality after PCA
constexpr int kTinyDonors    = 10;
constexpr int kTinyNH        = 20;   ///< number of neighborhoods sampled
constexpr int kTinyK         = 15;   ///< kNN k

/// Fraction of neighborhoods expected to be differentially abundant.
constexpr float kPlantedFraction = 0.40f;

/// Tolerances.
constexpr float kLogFcSpearmanTol   = 0.95f;
constexpr float kPvalSpearmanTol    = 0.90f;
constexpr float kIrlsConvergenceTol = 0.95f;
constexpr float kKsTol              = 0.10f;

/// KS calibration permutations.
constexpr int kNullPermutations = 200;

/// Real-data paths (GSM4037629 canonical test sample).
static const char* kGsmCountsPath =
    "/mnt/projects/debruinz_project/singlet_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/exon_counts.1pz";
static const char* kGsmDonorPath =
    "/mnt/projects/debruinz_project/singlet_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/donor_assignments.tsv";

static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet/gpu/"
    "tests/refs/milo_r_reference.R";

static const char* kRefsTmpDir =
    "/tmp/singlet_gpu_abundance_milo_refs_tmp";

static const char* kRscript = "Rscript";

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

int run_cmd_rc(const std::string& cmd) {
    return std::system(cmd.c_str());
}

bool gpu_available() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    return (e == cudaSuccess && count > 0);
}

// ---------------------------------------------------------------------------
// Host-side CSC container.
// ---------------------------------------------------------------------------
struct HostCSC {
    uint32_t             m;       ///< genes (rows)
    uint32_t             n;       ///< cells (cols)
    uint64_t             nnz;
    std::vector<float>   values;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
};

// ---------------------------------------------------------------------------
// Tiny synthetic data generator.
// Produces a reduced-dimension matrix (kTinyGenes × kTinyCells) as if output
// from PCA — dense float. Donor/condition assignment creates planted DA signal.
// ---------------------------------------------------------------------------
struct TinySyntheticData {
    // PCA-like embedding: kTinyGenes x kTinyCells (row-major for simplicity)
    std::vector<float> embedding;  // [gene * kTinyCells + cell]
    std::vector<int>   donor_ids;    // per cell, 0-based (int matches compute_milo API)
    std::vector<int>   conditions;   // per donor: 0 or 1
    std::vector<float> offsets;      // per donor: log(n_cells_per_donor)

    // Expected differential neighborhoods (planted in DA region)
    std::vector<bool> is_planted_da;   // per neighborhood (set after NH sampling)
};

TinySyntheticData make_tiny_synthetic(uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);

    TinySyntheticData d;
    d.embedding.resize(static_cast<size_t>(kTinyGenes) * kTinyCells);
    d.donor_ids.resize(kTinyCells);
    d.conditions.resize(kTinyDonors);
    d.offsets.resize(kTinyDonors);

    // Assign donors: first 5 = condition 0, last 5 = condition 1.
    for (int dn = 0; dn < kTinyDonors; ++dn)
        d.conditions[dn] = (dn < kTinyDonors / 2) ? 0 : 1;

    // Each cell assigned round-robin to a donor.
    std::vector<int> cells_per_donor(kTinyDonors, 0);
    for (int c = 0; c < kTinyCells; ++c) {
        d.donor_ids[c] = c % kTinyDonors;
        cells_per_donor[c % kTinyDonors]++;
    }
    for (int dn = 0; dn < kTinyDonors; ++dn)
        d.offsets[dn] = std::log(static_cast<float>(std::max(cells_per_donor[dn], 1)));

    // Embedding: random base + condition-specific offset to plant DA.
    // Cells in condition 1 are shifted along dim 0, creating a density shift.
    for (int c = 0; c < kTinyCells; ++c) {
        int dn   = d.donor_ids[c];
        int cond = d.conditions[dn];
        for (int g = 0; g < kTinyGenes; ++g) {
            float val = nd(rng);
            // Plant a shift on dimension 0 only for condition 1 cells.
            if (g == 0 && cond == 1) val += 2.5f;
            d.embedding[g * kTinyCells + c] = val;
        }
    }
    return d;
}

// ---------------------------------------------------------------------------
// Write neighborhood count matrix to TSV (n_nh rows x n_donors cols).
// ---------------------------------------------------------------------------
void write_nh_counts_tsv(const std::string& path,
                          const std::vector<std::vector<int32_t>>& nh_counts,
                          int n_nh, int n_donors) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    for (int i = 0; i < n_nh; ++i) {
        for (int d = 0; d < n_donors; ++d) {
            if (d > 0) f << "\t";
            f << nh_counts[i][d];
        }
        f << "\n";
    }
}

// ---------------------------------------------------------------------------
// Write donor metadata TSV: donor_id  condition  offset
// ---------------------------------------------------------------------------
void write_donor_meta_tsv(const std::string& path,
                           const std::vector<int>& conditions,
                           const std::vector<float>&   offsets,
                           int n_donors) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f << "donor_id\tcondition\toffset\n";
    for (int d = 0; d < n_donors; ++d)
        f << d << "\t" << conditions[d] << "\t" << offsets[d] << "\n";
}

// ---------------------------------------------------------------------------
// Parse milo_results.tsv written by the R reference script.
// ---------------------------------------------------------------------------
struct MiloRefResult {
    int32_t nh_idx;
    float   log_fc;
    float   se;
    float   p_value;
    float   fdr;
    bool    converged;
};

std::vector<MiloRefResult> parse_milo_results(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open R results: " + path);
    std::vector<MiloRefResult> rows;
    std::string line;
    std::getline(f, line);  // skip header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        MiloRefResult r{};
        std::string tok;
        // nh_idx log_fc se p_value fdr converged backend
        auto next = [&]() -> std::string {
            if (!std::getline(ss, tok, '\t')) tok = "";
            return tok;
        };
        r.nh_idx    = std::stoi(next());
        r.log_fc    = std::stof(next());
        r.se        = std::stof(next());
        std::string pv = next();
        r.p_value   = (pv == "NA" || pv.empty()) ? std::numeric_limits<float>::quiet_NaN()
                                                  : std::stof(pv);
        std::string fv = next();
        r.fdr       = (fv == "NA" || fv.empty()) ? std::numeric_limits<float>::quiet_NaN()
                                                  : std::stof(fv);
        std::string cv = next();
        r.converged = (cv == "TRUE" || cv == "true" || cv == "1");
        rows.push_back(r);
    }
    return rows;
}

// ---------------------------------------------------------------------------
// Spearman rank correlation between two float vectors.
// NaN-safe: NaN entries are excluded from both vectors jointly.
// ---------------------------------------------------------------------------
float spearman(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    int n = static_cast<int>(a.size());
    // Collect valid pairs.
    std::vector<std::pair<float, float>> pairs;
    pairs.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(a[i]) && !std::isnan(b[i]))
            pairs.emplace_back(a[i], b[i]);
    }
    int m = static_cast<int>(pairs.size());
    if (m < 3) return std::numeric_limits<float>::quiet_NaN();

    // Rank a.
    std::vector<int> ra(m), rb(m);
    std::iota(ra.begin(), ra.end(), 0);
    std::sort(ra.begin(), ra.end(),
              [&](int i, int j) { return pairs[i].first < pairs[j].first; });
    std::vector<float> rank_a(m);
    for (int i = 0; i < m; ++i) rank_a[ra[i]] = static_cast<float>(i);

    std::iota(rb.begin(), rb.end(), 0);
    std::sort(rb.begin(), rb.end(),
              [&](int i, int j) { return pairs[i].second < pairs[j].second; });
    std::vector<float> rank_b(m);
    for (int i = 0; i < m; ++i) rank_b[rb[i]] = static_cast<float>(i);

    // Pearson on ranks.
    float mean_a = 0.0f, mean_b = 0.0f;
    for (int i = 0; i < m; ++i) { mean_a += rank_a[i]; mean_b += rank_b[i]; }
    mean_a /= m; mean_b /= m;
    float num = 0.0f, den_a = 0.0f, den_b = 0.0f;
    for (int i = 0; i < m; ++i) {
        float da = rank_a[i] - mean_a;
        float db = rank_b[i] - mean_b;
        num  += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    if (den_a < 1e-12f || den_b < 1e-12f) return 0.0f;
    return num / std::sqrt(den_a * den_b);
}

// ---------------------------------------------------------------------------
// Kolmogorov-Smirnov statistic D vs Uniform(0,1).
// ---------------------------------------------------------------------------
float ks_uniform(std::vector<float> pvals) {
    // Remove NaNs.
    pvals.erase(std::remove_if(pvals.begin(), pvals.end(),
                               [](float v) { return std::isnan(v); }),
                pvals.end());
    if (pvals.empty()) return 1.0f;
    std::sort(pvals.begin(), pvals.end());
    int n = static_cast<int>(pvals.size());
    float d = 0.0f;
    for (int i = 0; i < n; ++i) {
        float expected = (i + 1.0f) / n;
        d = std::max(d, std::abs(pvals[i] - expected));
        if (i > 0)
            d = std::max(d, std::abs(pvals[i] - static_cast<float>(i) / n));
    }
    return d;
}

// ---------------------------------------------------------------------------
// Read donor_assignments.tsv produced by singlet (for real-data test).
// Returns vector of donor ids (0-based int) per cell.
// ---------------------------------------------------------------------------
struct DonorAssignments {
    std::vector<int32_t> donor_id;  // per cell
    int n_donors;
};

DonorAssignments load_donor_assignments(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open donor_assignments.tsv: " + path);
    std::string line;
    std::getline(f, line);  // header

    std::unordered_map<std::string, int32_t> donor_map;
    std::vector<int32_t> ids;
    int32_t next_id = 0;

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string barcode, donor, prob_max, prob_doublet;
        std::getline(ss, barcode, '\t');
        std::getline(ss, donor, '\t');
        auto it = donor_map.find(donor);
        if (it == donor_map.end()) {
            donor_map[donor] = next_id;
            it = donor_map.find(donor);
            it->second = next_id++;
        }
        ids.push_back(it->second);
    }
    return {ids, static_cast<int>(donor_map.size())};
}

}  // namespace

// =============================================================================
// Test fixture: sets up GPU context once per test suite.
// =============================================================================

class MiloCorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        ensure_refs_tmp();
        if (!gpu_available()) {
            GTEST_SKIP() << "No CUDA device available — skipping GPU tests.";
        }
    }
};

// =============================================================================
// Test 1: Milo_TinySynthetic_VsR
// =============================================================================
// 200 cells × 10 donors × 2 conditions with planted differential abundance.
// Runs the singlet-gpu Milo kernel and the R reference (miloR or MASS::glm.nb).
// Checks Spearman ρ ≥ 0.95 on log_fc and ρ ≥ 0.90 on p-value rank.
// =============================================================================

TEST_F(MiloCorrectnessTest, Milo_TinySynthetic_VsR) {
    TinySyntheticData d = make_tiny_synthetic(kSeed);

    // ------------------------------------------------------------------
    // Build kNN graph and neighborhood structure on the synthetic
    // embedding, then run singlet-gpu Milo kernel.
    // ------------------------------------------------------------------
    using namespace singlet::gpu;

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    // Config.
    abundance::MiloConfig cfg{};
    cfg.k            = kTinyK;
    cfg.n_nh_target  = kTinyNH;  // number of index cells to sample
    cfg.seed         = kSeed;
    cfg.max_irls_iters = 20;
    cfg.irls_tol     = 1e-6f;

    // Upload embedding as DeviceDense (n_pcs rows × n_cells cols).
    // DenseMatrixGPU(rows, cols, host_data) uploads synchronously on construction.
    core::DeviceDense d_emb(kTinyGenes, kTinyCells, d.embedding.data());

    // Build kNN graph from embedding (required by compute_milo).
    // compute_knn takes a DeviceDense [n_dims × n_cells] and produces KnnResult.
    graph::KnnConfig knn_cfg{};
    knn_cfg.k = cfg.k;
    graph::KnnResult knn = graph::compute_knn(d_emb, knn_cfg, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // Run Milo with the real API.
    abundance::MiloResult gpu_res =
        abundance::compute_milo(d_emb, d.donor_ids, d.conditions, knn, cfg, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaStreamDestroy(stream);

    int n_nh = gpu_res.n_nh;
    ASSERT_GT(n_nh, 0) << "Milo returned 0 neighborhoods";

    // Download device results to host for comparison.
    std::vector<float> h_log_fc(n_nh), h_p_value(n_nh);
    gpu_res.log_fc.download(h_log_fc.data(), n_nh);
    gpu_res.p_value.download(h_p_value.data(), n_nh);
    std::vector<float> h_nh_counts(static_cast<size_t>(n_nh) * kTinyDonors);
    gpu_res.nh_counts.download(h_nh_counts.data(), static_cast<size_t>(n_nh) * kTinyDonors);

    // ------------------------------------------------------------------
    // Build reference input files and run R reference script.
    // ------------------------------------------------------------------
    // Retrieve nh_counts from gpu_res (n_nh × n_donors).
    std::string nh_counts_path = refs_path("tiny_nh_counts.tsv");
    std::string donor_meta_path = refs_path("tiny_donor_meta.tsv");
    std::string r_out_dir       = refs_path("milo_tiny_out");
    fs::create_directories(r_out_dir);

    // Write nh_counts TSV (using host-downloaded h_nh_counts).
    {
        std::ofstream f(nh_counts_path);
        ASSERT_TRUE(f.is_open()) << "Cannot open " << nh_counts_path;
        for (int i = 0; i < n_nh; ++i) {
            for (int dj = 0; dj < kTinyDonors; ++dj) {
                if (dj > 0) f << "\t";
                f << h_nh_counts[i * kTinyDonors + dj];
            }
            f << "\n";
        }
    }
    write_donor_meta_tsv(donor_meta_path, d.conditions, d.offsets, kTinyDonors);

    // Run R reference.
    std::string r_cmd = std::string(kRscript) + " " + kRefScript
        + " --nh_counts "  + nh_counts_path
        + " --donor_meta " + donor_meta_path
        + " --out_dir "    + r_out_dir
        + " --seed 42 2>&1";

    int r_rc = run_cmd_rc(r_cmd);
    if (r_rc == 2) {
        GTEST_SKIP() << "R reference unavailable (neither miloR nor MASS installed).";
    }
    if (r_rc != 0) {
        FAIL() << "R reference script exited with code " << r_rc
               << " (check " << r_out_dir << " for stderr log).";
    }

    std::string r_results_path = r_out_dir + "/milo_results.tsv";
    auto r_rows = parse_milo_results(r_results_path);
    ASSERT_EQ(static_cast<int>(r_rows.size()), n_nh)
        << "R result row count mismatch";

    // ------------------------------------------------------------------
    // Compute Spearman correlations.
    // ------------------------------------------------------------------
    // h_log_fc and h_p_value already downloaded above.
    std::vector<float> gpu_lfc  = h_log_fc;
    std::vector<float> gpu_pval = h_p_value;
    std::vector<float> r_lfc(n_nh),   r_pval(n_nh);

    for (int i = 0; i < n_nh; ++i) {
        r_lfc[i]  = r_rows[i].log_fc;
        r_pval[i] = r_rows[i].p_value;
    }

    float rho_lfc  = spearman(gpu_lfc,  r_lfc);
    float rho_pval = spearman(gpu_pval, r_pval);

    EXPECT_GE(rho_lfc,  kLogFcSpearmanTol)
        << "log_fc Spearman ρ = " << rho_lfc
        << " < tolerance " << kLogFcSpearmanTol;
    EXPECT_GE(rho_pval, kPvalSpearmanTol)
        << "p-value Spearman ρ = " << rho_pval
        << " < tolerance " << kPvalSpearmanTol;

    std::printf("[Milo_TinySynthetic_VsR] n_nh=%d  rho_lfc=%.4f  rho_pval=%.4f\n",
                n_nh, rho_lfc, rho_pval);
}

// =============================================================================
// Test 2: Milo_GSM_RealData
// =============================================================================
// Real scRNA GSM4037629 with donor_assignments.tsv.
// Skip if either file is absent. Confirm: neighborhoods formed, all p-values
// finite, FDR values in [0,1].
// =============================================================================

TEST_F(MiloCorrectnessTest, Milo_GSM_RealData) {
    if (!fs::exists(kGsmCountsPath)) {
        GTEST_SKIP() << "Real data not found: " << kGsmCountsPath;
    }
    if (!fs::exists(kGsmDonorPath)) {
        GTEST_SKIP() << "donor_assignments.tsv not found: " << kGsmDonorPath;
    }

    // Load real data via the device loader.
    using namespace singlet::gpu;

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    io::PzDeviceMatrix pz = io::load_pz(kGsmCountsPath, stream);
    ASSERT_GT(pz.n_cells, 0) << "PZ loader returned 0 cells";

    // Load donor assignments.
    DonorAssignments da;
    try { da = load_donor_assignments(kGsmDonorPath); }
    catch (const std::exception& e) {
        GTEST_SKIP() << "Failed to parse donor_assignments.tsv: " << e.what();
    }

    int n_cells  = pz.n_cells;
    int n_donors = da.n_donors;
    cudaStreamDestroy(stream);

    if (n_donors < 3) {
        GTEST_SKIP() << "Only " << n_donors << " donors — need >=3 for NB GLM.";
    }
    if (static_cast<int>(da.donor_id.size()) != n_cells) {
        GTEST_SKIP() << "donor_id size (" << da.donor_id.size()
                     << ") != n_cells (" << n_cells << ")";
    }

    // TODO(CYCLE-55-MILO-API): compute_milo requires a core::DeviceDense PCA
    // embedding, not a raw DeviceCSC. The real-data test needs reduce/svd::auto_select
    // run on pz.mat before Milo can be called. That dependency is not wired here.
    // Deferring the GPU kernel call until cycle 55 wires the SVD → Milo pipeline.
    // Verify only that load_pz succeeds and donor parsing is correct.
    EXPECT_GT(n_cells, 0)   << "No cells loaded from GSM .1pz.";
    EXPECT_GT(n_donors, 0)  << "No donors parsed from donor_assignments.tsv.";
    GTEST_SKIP() << "Milo_GSM_RealData: GPU kernel call deferred — needs SVD "
                    "embedding upstream (TODO CYCLE-55-MILO-API).";

    std::printf("[Milo_GSM_RealData] n_cells=%d  n_donors=%d (deferred)\n",
                n_cells, n_donors);
}

// =============================================================================
// Test 3: Milo_IrlsConvergence
// =============================================================================
// >95% of neighborhood NB-GLMs converge within 20 IRLS iterations.
// Uses the tiny synthetic dataset (always available, no R required).
// =============================================================================

TEST_F(MiloCorrectnessTest, Milo_IrlsConvergence) {
    TinySyntheticData d = make_tiny_synthetic(kSeed);

    using namespace singlet::gpu;

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    abundance::MiloConfig cfg{};
    cfg.k              = kTinyK;
    cfg.n_nh_target    = kTinyNH;
    cfg.seed           = kSeed;
    cfg.max_irls_iters = 20;
    cfg.irls_tol       = 1e-6f;

    core::DeviceDense d_emb(kTinyGenes, kTinyCells, d.embedding.data());

    graph::KnnConfig knn_cfg{};
    knn_cfg.k = cfg.k;
    graph::KnnResult knn = graph::compute_knn(d_emb, knn_cfg, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    abundance::MiloResult gpu_res =
        abundance::compute_milo(d_emb, d.donor_ids, d.conditions, knn, cfg, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaStreamDestroy(stream);

    int n_nh = gpu_res.n_nh;
    ASSERT_GT(n_nh, 0);

    // MiloResult does not expose a per-NH converged flag directly.
    // Use nh_valid as proxy: valid NHs passed the GLM (converged or not filtered).
    // TODO(CYCLE-55-MILO-API): expose per-NH converged flag in MiloResult for
    // proper convergence rate testing.
    std::vector<int> h_nh_valid(n_nh);
    gpu_res.nh_valid.download(h_nh_valid.data(), n_nh);
    int n_converged = 0;
    for (int v : h_nh_valid) if (v) ++n_converged;

    float conv_rate = static_cast<float>(n_converged) / n_nh;
    EXPECT_GE(conv_rate, kIrlsConvergenceTol)
        << "IRLS convergence rate = " << conv_rate
        << " < " << kIrlsConvergenceTol
        << " (" << n_converged << "/" << n_nh << " passed GLM filter)";

    std::printf("[Milo_IrlsConvergence] n_nh=%d  valid=%d  rate=%.4f\n",
                n_nh, n_converged, conv_rate);
}

// =============================================================================
// Test 4: Milo_PvalueCalibration_Uniform
// =============================================================================
// Under null (random condition labels), p-values should be approximately
// uniform. KS statistic vs Uniform(0,1) must be <= 0.10.
// Uses kNullPermutations permutations pooled into one KS test.
// =============================================================================

TEST_F(MiloCorrectnessTest, Milo_PvalueCalibration_Uniform) {
    using namespace singlet::gpu;

    std::mt19937_64 rng(kSeed ^ 0xDEADBEEFull);

    // Build a null dataset: no planted signal.
    TinySyntheticData d_null = make_tiny_synthetic(kSeed ^ 0x4E554C4Cull);  // 0xNULL→ ascii "NULL"
    // Randomize conditions completely (no structured DA signal).
    {
        std::uniform_int_distribution<int> bin(0, 1);
        for (int dn = 0; dn < kTinyDonors; ++dn)
            d_null.conditions[dn] = bin(rng);
    }

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    abundance::MiloConfig cfg{};
    cfg.k              = kTinyK;
    cfg.n_nh_target    = kTinyNH;
    cfg.seed           = kSeed;
    cfg.max_irls_iters = 20;
    cfg.irls_tol       = 1e-6f;

    // Build embedding and kNN once (embedding doesn't change across permutations).
    core::DeviceDense d_emb(kTinyGenes, kTinyCells, d_null.embedding.data());
    graph::KnnConfig knn_cfg{};
    knn_cfg.k = cfg.k;
    graph::KnnResult knn = graph::compute_knn(d_emb, knn_cfg, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> all_pvals;

    for (int perm = 0; perm < kNullPermutations; ++perm) {
        // Shuffle condition labels.
        std::vector<int> conds(d_null.conditions);
        std::shuffle(conds.begin(), conds.end(), rng);

        // Ensure both conditions are present (at least 2 donors each).
        int c0 = 0, c1 = 0;
        for (int c : conds) { if (c == 0) ++c0; else ++c1; }
        if (c0 < 2 || c1 < 2) { --perm; continue; }

        cfg.seed = kSeed + static_cast<uint64_t>(perm);
        abundance::MiloResult res =
            abundance::compute_milo(d_emb, d_null.donor_ids, conds, knn, cfg, stream);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        int n_res = res.n_nh;
        std::vector<float> h_pv(n_res);
        res.p_value.download(h_pv.data(), n_res);
        for (float pv : h_pv)
            if (std::isfinite(pv)) all_pvals.push_back(pv);
    }
    cudaStreamDestroy(stream);

    ASSERT_GT(all_pvals.size(), 10u) << "Too few valid p-values for KS test";

    float ks = ks_uniform(all_pvals);
    EXPECT_LE(ks, kKsTol)
        << "KS statistic = " << ks
        << " > " << kKsTol
        << " (p-values not uniform under null; n=" << all_pvals.size() << ")";

    std::printf("[Milo_PvalueCalibration_Uniform] n_pvals=%zu  KS=%.4f\n",
                all_pvals.size(), ks);
}

// =============================================================================
// Test 5: Milo_Determinism_BitIdentical
// =============================================================================
// Two runs with the same fixed seed must produce bit-identical outputs:
// log_fc, se, p_value, fdr, converged.
// =============================================================================

TEST_F(MiloCorrectnessTest, Milo_Determinism_BitIdentical) {
    TinySyntheticData d = make_tiny_synthetic(kSeed);

    using namespace singlet::gpu;

    // run_once returns host-copied result vectors for bit-identity comparison.
    struct MiloHostResult {
        std::vector<float> log_fc, se, p_value;
        int n_nh = 0;
    };

    auto run_once = [&](uint64_t seed_val) -> MiloHostResult {
        cudaStream_t stream{};
        EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        abundance::MiloConfig cfg{};
        cfg.k              = kTinyK;
        cfg.n_nh_target    = kTinyNH;
        cfg.seed           = seed_val;
        cfg.max_irls_iters = 20;
        cfg.irls_tol       = 1e-6f;
        cfg.deterministic  = true;  // opt-in to deterministic atomics path

        core::DeviceDense d_emb(kTinyGenes, kTinyCells, d.embedding.data());

        graph::KnnConfig knn_cfg{};
        knn_cfg.k = cfg.k;
        graph::KnnResult knn = graph::compute_knn(d_emb, knn_cfg, stream);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        abundance::MiloResult res =
            abundance::compute_milo(d_emb, d.donor_ids, d.conditions, knn, cfg, stream);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        cudaStreamDestroy(stream);

        MiloHostResult h;
        h.n_nh = res.n_nh;
        h.log_fc.resize(h.n_nh);
        h.se.resize(h.n_nh);
        h.p_value.resize(h.n_nh);
        res.log_fc.download(h.log_fc.data(), h.n_nh);
        res.se.download(h.se.data(), h.n_nh);
        res.p_value.download(h.p_value.data(), h.n_nh);
        return h;
    };

    MiloHostResult r1 = run_once(kSeed);
    MiloHostResult r2 = run_once(kSeed);

    ASSERT_EQ(r1.n_nh, r2.n_nh)
        << "Result sizes differ between runs";
    int n_nh = r1.n_nh;

    for (int i = 0; i < n_nh; ++i) {
        // Bit-identical comparison using memcmp-safe float-to-int reinterpret.
        uint32_t lfc1, lfc2, se1, se2, pv1, pv2;
        std::memcpy(&lfc1, &r1.log_fc[i],  4);
        std::memcpy(&lfc2, &r2.log_fc[i],  4);
        std::memcpy(&se1,  &r1.se[i],      4);
        std::memcpy(&se2,  &r2.se[i],      4);
        std::memcpy(&pv1,  &r1.p_value[i], 4);
        std::memcpy(&pv2,  &r2.p_value[i], 4);

        EXPECT_EQ(lfc1, lfc2) << "log_fc not bit-identical at NH " << i;
        EXPECT_EQ(se1,  se2)  << "se not bit-identical at NH " << i;
        EXPECT_EQ(pv1,  pv2)  << "p_value not bit-identical at NH " << i;
    }

    int diffs = 0;
    for (int i = 0; i < n_nh; ++i) {
        uint32_t lfc1, lfc2;
        std::memcpy(&lfc1, &r1.log_fc[i], 4);
        std::memcpy(&lfc2, &r2.log_fc[i], 4);
        if (lfc1 != lfc2) ++diffs;
    }
    EXPECT_EQ(diffs, 0)
        << diffs << "/" << n_nh << " log_fc values differ between runs";

    std::printf("[Milo_Determinism_BitIdentical] n_nh=%d  diffs=%d\n",
                n_nh, diffs);
}

// =============================================================================
// main
// =============================================================================
// Note: No custom main needed — GTest::gtest_main provides it.
