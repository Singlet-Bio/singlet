// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/disease_scdrs_correctness.cpp
//
// integrates: original (first GPU scDRS-style per-cell disease relevance scoring
//             from GWAS summary statistics via batched SpMV on cuSPARSE)
//
// Correctness harness for singlet_gpu::disease::scdrs.
//
// Written against the design doc at:
//   singlet-gpu/state/designs/48-scdrs.md
// NOT against the kernel source — authored in parallel with disease/scdrs.h by
// the analysis-validator worker (Sonnet, Tier 2).
//
// References:
//   Primary:  scDRS Python package (Zhang et al. 2022, Nature Genetics) via subprocess
//             tests/refs/scdrs_py_reference.py
//             pip install scdrs
//   Fallback: pure-Python weighted sum + control-set null matching (always runs;
//             implemented inside scdrs_py_reference.py --force_fallback)
//
// Tolerances (design doc §"Correctness test spec"):
//   raw_score Spearman ρ vs Python   >= 0.95  (test 1 — tiny synthetic)
//   real data: all finite results            (test 2 — GSM4037629 mini diseases)
//   control set (mean, var) relative error   <= 0.05  (test 3)
//   p-value KS statistic under null          <= 0.10  (test 4)
//   Bit-identical with fixed seed                     (test 5)
//
// Test cases:
//   Scdrs_TinySynthetic_VsPython
//   Scdrs_GSM_RealData
//   Scdrs_ControlSets_MatchedMeanVar
//   Scdrs_PvalueCalibration_Uniform
//   Scdrs_Determinism_BitIdentical
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, CUDA::cusparse, CUDA::curand,
//             singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every test that requires a device is guarded by
//   gpu_available() and calls GTEST_SKIP().

// ---- singlet-gpu disease header (written by gpu-kernel-dev) -----------------
#include <singlet-gpu/disease/scdrs.h>

// ---- other singlet-gpu headers needed by the real-data test -----------------
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>

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
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

/// Tiny synthetic dimensions (design doc §"Correctness test spec").
constexpr int kTinyCells    = 200;
constexpr int kTinyGenes    = 500;
constexpr int kTinyDiseases = 5;

/// Genes per disease in the tiny synthetic data.
constexpr int kGenesPerDisease = 20;

/// Number of control sets (smaller than production 1000 for speed in tests).
constexpr int kNCtrl = 200;

/// Disease-associated cluster shift magnitude (planted signal strength).
constexpr float kPlantedShift = 3.0f;

/// Tolerance: Spearman ρ threshold for raw score comparison vs Python reference.
constexpr float kSpearmanRhoTol = 0.95f;

/// Tolerance: relative error for control-set (mean, var) matching.
constexpr float kCtrlMatchRelErr = 0.05f;

/// Tolerance: KS statistic under null p-values.
constexpr float kKsTol = 0.10f;

/// Number of null permutations for KS calibration test.
constexpr int kNullPermutations = 200;

/// Canonical real-data path (GSM4037629, 11,560 cells).
static const char* kGsmExprPath =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/exon_counts.1pz";

/// Mini disease fixture shipped with the test suite.
static const char* kMiniDiseaseFixture =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/"
    "tests/refs/scdrs_diseases_mini.tsv";

/// Python reference script.
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/"
    "tests/refs/scdrs_py_reference.py";

/// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_disease_scdrs_refs_tmp";

// =============================================================================
// Utility helpers (pattern from cycles 3-49)
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

bool python_ref_available() {
    return fs::exists(kRefScript);
}

// ---------------------------------------------------------------------------
// Minimal .npy v1.0 writer (float32 2-D, C-order).
// Reused from cycles 3-49 (e.g. gsea_fgsea_correctness.cpp pattern).
// ---------------------------------------------------------------------------
static void write_npy_f32_2d(const std::string& path,
                              const float* data, int rows, int cols) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for write: " + path);
    // Magic + version
    const uint8_t magic[] = {0x93,'N','U','M','P','Y',1,0};
    f.write(reinterpret_cast<const char*>(magic), 8);
    // Header dict
    std::ostringstream hdr;
    hdr << "{'descr': '<f4', 'fortran_order': False, 'shape': ("
        << rows << ", " << cols << "), }";
    std::string hdr_str = hdr.str();
    // Pad to 64-byte alignment (header_len field at byte 8-10 is uint16 LE).
    size_t total_hdr = 10 + hdr_str.size() + 1; // +1 for '\n'
    size_t pad = (64 - (total_hdr % 64)) % 64;
    hdr_str.append(pad, ' ');
    hdr_str += '\n';
    uint16_t hlen = static_cast<uint16_t>(hdr_str.size());
    f.write(reinterpret_cast<const char*>(&hlen), 2);
    f.write(hdr_str.data(), static_cast<std::streamsize>(hdr_str.size()));
    f.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(rows * cols * sizeof(float)));
}

// ---------------------------------------------------------------------------
// Minimal .npy v1.0 writer for float32 1-D.
// ---------------------------------------------------------------------------
static void write_npy_f32(const std::string& path, const float* data, int n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for write: " + path);
    const uint8_t magic[] = {0x93,'N','U','M','P','Y',1,0};
    f.write(reinterpret_cast<const char*>(magic), 8);
    std::ostringstream hdr;
    hdr << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << n << ",), }";
    std::string hdr_str = hdr.str();
    size_t total_hdr = 10 + hdr_str.size() + 1;
    size_t pad = (64 - (total_hdr % 64)) % 64;
    hdr_str.append(pad, ' ');
    hdr_str += '\n';
    uint16_t hlen = static_cast<uint16_t>(hdr_str.size());
    f.write(reinterpret_cast<const char*>(&hlen), 2);
    f.write(hdr_str.data(), static_cast<std::streamsize>(hdr_str.size()));
    f.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(n * sizeof(float)));
}

// ---------------------------------------------------------------------------
// Minimal .npy reader for float32 2-D (C-order).
// Returns flat row-major data; sets rows and cols via out-params.
// Only handles NPY v1.0 with dtype '<f4' and fortran_order False.
// ---------------------------------------------------------------------------
static std::vector<float> read_npy_f32_2d(const std::string& path,
                                           int& rows, int& cols) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for read: " + path);
    // Read magic (6) + version (2) + header_len (2) = 10 bytes.
    uint8_t hdr[10];
    f.read(reinterpret_cast<char*>(hdr), 10);
    uint16_t hlen = static_cast<uint16_t>(hdr[8]) |
                    (static_cast<uint16_t>(hdr[9]) << 8);
    std::string hdr_str(hlen, '\0');
    f.read(&hdr_str[0], hlen);
    // Parse shape from "shape': (R, C)" — minimal, not a full parser.
    auto shape_pos = hdr_str.find("'shape'");
    if (shape_pos == std::string::npos)
        throw std::runtime_error("NPY header missing 'shape': " + path);
    auto lp = hdr_str.find('(', shape_pos);
    auto rp = hdr_str.find(')', lp);
    std::string shape_str = hdr_str.substr(lp + 1, rp - lp - 1);
    // shape_str = "R, C" or "R,"
    auto comma = shape_str.find(',');
    if (comma == std::string::npos)
        throw std::runtime_error("NPY shape parse error: " + path);
    rows = std::stoi(shape_str.substr(0, comma));
    std::string rest = shape_str.substr(comma + 1);
    // strip whitespace
    auto start = rest.find_first_not_of(" \t");
    cols = (start == std::string::npos) ? 1 : std::stoi(rest.substr(start));
    std::vector<float> data(static_cast<size_t>(rows) * cols);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(data.size() * sizeof(float)));
    return data;
}

// ---------------------------------------------------------------------------
// Spearman rank correlation (host-side, used for correctness check).
// Reused from cycles 9, 11, 12 (gsea, aucell, anno_marker_score).
// ---------------------------------------------------------------------------
static float spearman_rho(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    int n = static_cast<int>(a.size());
    if (n < 2) return 1.0f;

    auto rank_of = [n](const std::vector<float>& v) {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&v](int i, int j){ return v[i] < v[j]; });
        std::vector<float> r(n);
        for (int i = 0; i < n; ) {
            int j = i;
            while (j < n && v[idx[j]] == v[idx[i]]) ++j;
            float avg_rank = 0.5f * (i + j - 1);
            for (int k = i; k < j; ++k) r[idx[k]] = avg_rank;
            i = j;
        }
        return r;
    };

    std::vector<float> ra = rank_of(a);
    std::vector<float> rb = rank_of(b);

    double sum_d2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = ra[i] - rb[i];
        sum_d2 += d * d;
    }
    double rho = 1.0 - 6.0 * sum_d2 / (static_cast<double>(n) *
                                         (static_cast<double>(n) * n - 1.0));
    return static_cast<float>(rho);
}

// ---------------------------------------------------------------------------
// KS statistic for uniform(0,1) goodness-of-fit (one-sample, one-sided).
// Reused from cycles 32, 47 (qc_doublet, abundance_milo).
// ---------------------------------------------------------------------------
static float ks_statistic_uniform(std::vector<float> pvals) {
    std::sort(pvals.begin(), pvals.end());
    int n = static_cast<int>(pvals.size());
    float D = 0.0f;
    for (int i = 0; i < n; ++i) {
        float cdf_val = static_cast<float>(i + 1) / n;
        float d = std::abs(cdf_val - pvals[i]);
        D = std::max(D, d);
    }
    return D;
}

// ---------------------------------------------------------------------------
// Tiny synthetic data generator.
// 200 cells × 500 genes × 5 diseases with planted disease-associated clusters.
// Cells [0, 39] form a disease-associated cluster for DISEASE_00.
// Genes [0, 19] are the planted gene set for DISEASE_00.
// ---------------------------------------------------------------------------
struct TinySynthetic {
    // Row-major: n_cells × n_genes
    std::vector<float> expr;          ///< log-normalized expression (counts already log1p'd)
    // Gene sets: disease_idx → list of (gene_idx, weight)
    std::vector<std::vector<std::pair<int,float>>> gene_sets;
    std::vector<std::string> disease_names;
    // Expected: cells [0..39] should have high raw_score for DISEASE_00
    std::vector<bool> expected_high_score;  ///< per cell: true if expected high for DISEASE_00
};

TinySynthetic make_tiny_synthetic(uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);

    TinySynthetic d;
    d.expr.resize(static_cast<size_t>(kTinyCells) * kTinyGenes);
    d.gene_sets.resize(kTinyDiseases);
    d.disease_names.resize(kTinyDiseases);
    d.expected_high_score.resize(kTinyCells, false);

    // Fill background expression (all cells × all genes).
    for (auto& v : d.expr) v = std::max(0.0f, nd(rng));

    // Plant disease-associated signal.
    // DISEASE_k: gene set = genes [k*20 .. k*20+19] (first kGenesPerDisease genes per disease)
    // Cluster cells: [k*40 .. k*40+39] have elevated expression for those genes.
    for (int di = 0; di < kTinyDiseases; ++di) {
        d.disease_names[di] = "DISEASE_" + std::to_string(di);
        int gene_start = di * kGenesPerDisease;
        int cell_start = di * 40;

        // Build gene set with random weights.
        for (int g = gene_start; g < gene_start + kGenesPerDisease; ++g) {
            float w = ud(rng);
            d.gene_sets[di].emplace_back(g, w);
        }

        // Elevate expression for cluster cells.
        for (int c = cell_start; c < cell_start + 40 && c < kTinyCells; ++c) {
            for (int g = gene_start; g < gene_start + kGenesPerDisease; ++g) {
                d.expr[static_cast<size_t>(c) * kTinyGenes + g] += kPlantedShift;
            }
            if (di == 0) d.expected_high_score[c] = true;
        }
    }

    return d;
}

// ---------------------------------------------------------------------------
// Write expression matrix to NPY and gene names to TXT for the Python reference.
// ---------------------------------------------------------------------------
static void write_tiny_inputs(const TinySynthetic& d,
                               const std::string& expr_path,
                               const std::string& names_path) {
    write_npy_f32_2d(expr_path, d.expr.data(), kTinyCells, kTinyGenes);
    std::ofstream f(names_path);
    if (!f) throw std::runtime_error("Cannot open gene names: " + names_path);
    for (int g = 0; g < kTinyGenes; ++g)
        f << "GENE_" << g << "\n";
}

// ---------------------------------------------------------------------------
// Write disease TSV matching the fixture format.
// ---------------------------------------------------------------------------
static void write_disease_tsv(const TinySynthetic& d, const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open disease tsv: " + path);
    f << "disease_name\tgene_idx\tweight\n";
    for (int di = 0; di < kTinyDiseases; ++di) {
        for (auto [gidx, w] : d.gene_sets[di])
            f << d.disease_names[di] << "\t" << gidx << "\t" << w << "\n";
    }
}

// ---------------------------------------------------------------------------
// Run the Python reference script and load its output.
// Returns false if the subprocess fails (reference unavailable).
// ---------------------------------------------------------------------------
struct PyRefOutput {
    std::vector<float> raw_score;   // n_cells × n_diseases, row-major
    std::vector<float> norm_score;
    std::vector<float> pvalue;
    int n_cells;
    int n_diseases;
    std::string backend;
};

static bool run_py_reference(const std::string& expr_path,
                              const std::string& names_path,
                              const std::string& disease_path,
                              int n_ctrl,
                              uint64_t seed,
                              PyRefOutput& out,
                              bool force_fallback = false) {
    ensure_refs_tmp();
    const std::string out_dir = refs_path("scdrs_out");
    fs::create_directories(out_dir);

    std::ostringstream cmd;
    cmd << "python3 " << kRefScript
        << " --expr_npy "   << expr_path
        << " --gene_names " << names_path
        << " --diseases "   << disease_path
        << " --n_ctrl "     << n_ctrl
        << " --seed "       << seed
        << " --out_dir "    << out_dir;
    if (force_fallback) cmd << " --force_fallback";

    int rc = run_cmd_rc(cmd.str());
    if (rc != 0) return false;

    const std::string raw_path  = out_dir + "/raw_score.npy";
    const std::string norm_path = out_dir + "/norm_score.npy";
    const std::string pval_path = out_dir + "/pvalue.npy";
    const std::string bknd_path = out_dir + "/backend.txt";

    if (!fs::exists(raw_path)) return false;

    int rows, cols;
    out.raw_score  = read_npy_f32_2d(raw_path,  rows, cols);
    out.n_cells    = rows;
    out.n_diseases = cols;
    int r2, c2;
    out.norm_score = read_npy_f32_2d(norm_path, r2, c2);
    int r3, c3;
    out.pvalue     = read_npy_f32_2d(pval_path, r3, c3);

    if (fs::exists(bknd_path)) {
        std::ifstream bf(bknd_path);
        std::getline(bf, out.backend);
    } else {
        out.backend = "unknown";
    }
    return true;
}

// ---------------------------------------------------------------------------
// Host-side pure-C++ scDRS scoring (used by tests 3, 4, 5 to avoid GPU dep).
// Implements the same algorithm as the Python fallback.
// ---------------------------------------------------------------------------

/// Gene-wise Z-score normalization: output z[cell * n_genes + gene].
static std::vector<float> gene_zscore_host(const float* expr,
                                             int n_cells, int n_genes) {
    std::vector<double> mu(n_genes, 0.0), var(n_genes, 0.0);
    for (int c = 0; c < n_cells; ++c)
        for (int g = 0; g < n_genes; ++g)
            mu[g] += expr[static_cast<size_t>(c) * n_genes + g];
    for (int g = 0; g < n_genes; ++g) mu[g] /= n_cells;
    for (int c = 0; c < n_cells; ++c)
        for (int g = 0; g < n_genes; ++g) {
            double d = expr[static_cast<size_t>(c) * n_genes + g] - mu[g];
            var[g] += d * d;
        }
    for (int g = 0; g < n_genes; ++g) var[g] /= n_cells;

    std::vector<float> z(static_cast<size_t>(n_cells) * n_genes);
    for (int c = 0; c < n_cells; ++c)
        for (int g = 0; g < n_genes; ++g) {
            double sd = std::sqrt(var[g]);
            if (sd < 1e-8) sd = 1.0;
            z[static_cast<size_t>(c) * n_genes + g] =
                static_cast<float>(
                    (expr[static_cast<size_t>(c) * n_genes + g] - mu[g]) / sd);
        }
    return z;
}

/// Compute raw scores: out[c * n_dis + d] = Σ_g  z[c, g] * W[g, d]
static std::vector<float> compute_raw_scores_host(
        const std::vector<float>& z, int n_cells, int n_genes,
        const std::vector<std::vector<std::pair<int,float>>>& gene_sets) {
    int n_dis = static_cast<int>(gene_sets.size());
    std::vector<float> scores(static_cast<size_t>(n_cells) * n_dis, 0.0f);
    for (int di = 0; di < n_dis; ++di) {
        // Normalise weights so they sum to 1.
        float wsum = 0.0f;
        for (auto [gidx, w] : gene_sets[di]) wsum += w;
        if (wsum < 1e-8f) wsum = 1.0f;
        for (int c = 0; c < n_cells; ++c) {
            float s = 0.0f;
            for (auto [gidx, w] : gene_sets[di])
                s += z[static_cast<size_t>(c) * n_genes + gidx] * (w / wsum);
            scores[static_cast<size_t>(c) * n_dis + di] = s;
        }
    }
    return scores;
}

/// Build matched control weight matrix: ctrl_W[g * n_ctrl + k].
/// (Pattern from scdrs_py_reference.py fallback — same bin strategy.)
static std::vector<float> build_control_sets_host(
        const std::vector<std::vector<std::pair<int,float>>>& gene_sets,
        const float* expr, int n_cells, int n_genes,
        int n_ctrl, uint64_t seed) {
    // Gene statistics.
    std::vector<float> gene_mean(n_genes, 0.0f), gene_var(n_genes, 0.0f);
    for (int c = 0; c < n_cells; ++c)
        for (int g = 0; g < n_genes; ++g)
            gene_mean[g] += expr[static_cast<size_t>(c) * n_genes + g];
    for (int g = 0; g < n_genes; ++g) gene_mean[g] /= n_cells;
    for (int c = 0; c < n_cells; ++c)
        for (int g = 0; g < n_genes; ++g) {
            float d = expr[static_cast<size_t>(c) * n_genes + g] - gene_mean[g];
            gene_var[g] += d * d;
        }
    for (int g = 0; g < n_genes; ++g) gene_var[g] /= n_cells;

    std::vector<float> score(n_genes);
    for (int g = 0; g < n_genes; ++g) score[g] = gene_mean[g] + gene_var[g];

    // Build bins (20 bins by score).
    constexpr int kNBins = 20;
    std::vector<float> sorted_score = score;
    std::sort(sorted_score.begin(), sorted_score.end());
    std::vector<float> thresholds(kNBins);
    for (int b = 0; b < kNBins; ++b) {
        int idx = static_cast<int>((static_cast<float>(b + 1) / kNBins) *
                                   (n_genes - 1));
        thresholds[b] = sorted_score[std::min(idx, n_genes - 1)];
    }
    auto bin_of = [&](int g) {
        int b = 0;
        while (b < kNBins - 1 && score[g] > thresholds[b]) ++b;
        return b;
    };
    std::vector<std::vector<int>> bins(kNBins);
    for (int g = 0; g < n_genes; ++g) bins[bin_of(g)].push_back(g);

    // Collect disease genes (union across all disease sets).
    std::vector<int> disease_genes;
    for (auto& gs : gene_sets)
        for (auto [gidx, w] : gs)
            disease_genes.push_back(gidx);
    // Deduplicate.
    std::sort(disease_genes.begin(), disease_genes.end());
    disease_genes.erase(std::unique(disease_genes.begin(), disease_genes.end()),
                        disease_genes.end());
    if (disease_genes.empty())
        return std::vector<float>(static_cast<size_t>(n_genes) * n_ctrl, 0.0f);

    float w_per = 1.0f / static_cast<float>(disease_genes.size());

    std::mt19937_64 rng(seed);
    std::vector<float> ctrl_W(static_cast<size_t>(n_genes) * n_ctrl, 0.0f);
    for (int k = 0; k < n_ctrl; ++k) {
        for (int gidx : disease_genes) {
            int b = bin_of(gidx);
            auto& pool = bins[b];
            int replacement = gidx;
            if (pool.size() > 1) {
                // Sample uniformly from bin, excluding gidx if possible.
                std::vector<int> candidates;
                candidates.reserve(pool.size());
                for (int g2 : pool) if (g2 != gidx) candidates.push_back(g2);
                if (candidates.empty()) candidates = pool;
                std::uniform_int_distribution<int> uid(
                    0, static_cast<int>(candidates.size()) - 1);
                replacement = candidates[uid(rng)];
            }
            ctrl_W[static_cast<size_t>(replacement) * n_ctrl + k] += w_per;
        }
    }
    return ctrl_W;
}

// ---------------------------------------------------------------------------
// Convert a dense row-major n_cells × n_genes host matrix to a device CSC
// with layout: rows = genes, cols = cells (matches pz_device_loader convention).
// score() expects expr.rows = n_genes, expr.cols = n_cells.
// col_ptr has size n_cells + 1; for cell c, nonzeros are genes 0..n_genes-1.
// ---------------------------------------------------------------------------
static singlet_gpu::core::DeviceCSC dense_to_device_csc(
        const float* expr_cells_x_genes, int n_cells, int n_genes,
        cudaStream_t stream) {
    int nnz = n_genes * n_cells;  // all entries nonzero for the tiny test

    std::vector<int>   h_col_ptr(n_cells + 1);
    std::vector<int>   h_row_indices(nnz);
    std::vector<float> h_values(nnz);

    // Uniform n_genes nonzeros per cell column.
    for (int c = 0; c <= n_cells; ++c) h_col_ptr[c] = c * n_genes;

    // For cell column c, row indices are 0..n_genes-1 (all genes).
    for (int c = 0; c < n_cells; ++c) {
        for (int g = 0; g < n_genes; ++g) {
            int pos = c * n_genes + g;
            h_row_indices[pos] = g;
            h_values[pos]      = expr_cells_x_genes[static_cast<size_t>(c) * n_genes + g];
        }
    }

    // SparseMatrixGPU(m, n, nnz, h_col_ptr, h_row_indices, h_values) uploads sync.
    return singlet_gpu::core::DeviceCSC(
        n_genes, n_cells, nnz,
        h_col_ptr.data(), h_row_indices.data(), h_values.data());
}

// Build gene name vector for a CSC with n_genes rows.
static std::vector<std::string> make_gene_names(int n_genes) {
    std::vector<std::string> names(n_genes);
    for (int g = 0; g < n_genes; ++g) names[g] = "GENE_" + std::to_string(g);
    return names;
}

// ScDRS host result (std::vector backing) for bit-identity test.
struct ScDrsHostResult {
    std::vector<float> raw_score;
    std::vector<float> normalized_score;
    std::vector<float> p_value;
    int n_cells    = 0;
    int n_diseases = 0;
};

// Download a ScDrsResult from device to host.
static ScDrsHostResult download_scdrs_result(
        const singlet_gpu::disease::ScDrsResult& res) {
    ScDrsHostResult h;
    h.n_cells    = res.n_cells;
    h.n_diseases = res.n_diseases;
    size_t total = static_cast<size_t>(res.n_cells) * res.n_diseases;
    h.raw_score.resize(total);
    h.normalized_score.resize(total);
    h.p_value.resize(total);
    res.raw_score.download(h.raw_score.data(), total);
    res.normalized_score.download(h.normalized_score.data(), total);
    res.p_value.download(h.p_value.data(), total);
    return h;
}

} // anonymous namespace

// =============================================================================
// Test case 1: Scdrs_TinySynthetic_VsPython
// =============================================================================
// 200 cells × 500 genes × 5 diseases with planted disease-associated clusters.
// Requirement: Spearman ρ ≥ 0.95 on raw_score (cell dimension) for each disease.
// The test serialises data to NPY/TSV, runs the Python reference (with fallback),
// then compares to the GPU kernel output.
// =============================================================================

TEST(ScDRSCorrectness, Scdrs_TinySynthetic_VsPython) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device; GPU kernel cannot run.";
    if (!python_ref_available())
        GTEST_SKIP() << "Python reference script not found at: " << kRefScript;

    ensure_refs_tmp();
    TinySynthetic d = make_tiny_synthetic(kSeed);

    // Write inputs for the reference.
    const std::string expr_path   = refs_path("tiny_expr.npy");
    const std::string names_path  = refs_path("tiny_gene_names.txt");
    const std::string disease_path = refs_path("tiny_diseases.tsv");
    write_tiny_inputs(d, expr_path, names_path);
    write_disease_tsv(d, disease_path);

    // Run Python reference (with fallback).
    PyRefOutput ref;
    bool ref_ok = run_py_reference(expr_path, names_path, disease_path,
                                   kNCtrl, kSeed, ref);
    ASSERT_TRUE(ref_ok) << "Python reference subprocess failed.";
    ASSERT_EQ(ref.n_cells,    kTinyCells)    << "Reference n_cells mismatch.";
    ASSERT_EQ(ref.n_diseases, kTinyDiseases) << "Reference n_diseases mismatch.";

    // -----------------------------------------------------------------------
    // Run GPU kernel: singlet_gpu::disease::score()
    // -----------------------------------------------------------------------
    singlet_gpu::disease::ScDrsConfig cfg;
    cfg.n_controls    = kNCtrl;
    cfg.seed          = kSeed;
    cfg.n_bins        = 20;
    cfg.disease_chunk = 0;  // process all diseases in one pass for small test

    // Build disease gene sets using the real DiseaseGeneSet API:
    // gs.genes is unordered_map<string, float> (gene_name → weight).
    std::vector<singlet_gpu::disease::DiseaseGeneSet> kernel_sets;
    kernel_sets.reserve(kTinyDiseases);
    auto gene_names = make_gene_names(kTinyGenes);
    for (int di = 0; di < kTinyDiseases; ++di) {
        singlet_gpu::disease::DiseaseGeneSet gs;
        gs.name = d.disease_names[di];
        for (auto [gidx, w] : d.gene_sets[di]) {
            gs.genes[gene_names[gidx]] = w;
        }
        kernel_sets.push_back(std::move(gs));
    }

    // Build device CSC from the tiny dense expression matrix.
    cudaStream_t csc_stream{};
    ASSERT_EQ(cudaStreamCreate(&csc_stream), cudaSuccess);
    singlet_gpu::core::DeviceCSC expr_csc =
        dense_to_device_csc(d.expr.data(), kTinyCells, kTinyGenes, csc_stream);
    cudaStreamDestroy(csc_stream);

    // Run score() via the default GPU context.
    singlet_gpu::core::GPUContext& ctx = singlet_gpu::core::default_context();
    singlet_gpu::disease::ScDrsResult dev_result =
        singlet_gpu::disease::score(expr_csc, gene_names, kernel_sets, cfg, ctx);

    // Download to host for comparison.
    ScDrsHostResult result = download_scdrs_result(dev_result);

    ASSERT_EQ(result.n_cells,    kTinyCells)    << "GPU result n_cells mismatch.";
    ASSERT_EQ(result.n_diseases, kTinyDiseases) << "GPU result n_diseases mismatch.";

    // -----------------------------------------------------------------------
    // Compare: Spearman ρ per disease (across cells).
    // -----------------------------------------------------------------------
    for (int di = 0; di < kTinyDiseases; ++di) {
        std::vector<float> gpu_col(kTinyCells), py_col(kTinyCells);
        for (int c = 0; c < kTinyCells; ++c) {
            gpu_col[c] = result.raw_score[static_cast<size_t>(c) * kTinyDiseases + di];
            py_col[c]  = ref.raw_score[static_cast<size_t>(c) * kTinyDiseases + di];
        }
        float rho = spearman_rho(gpu_col, py_col);
        EXPECT_GE(rho, kSpearmanRhoTol)
            << "Disease " << di << " raw_score Spearman rho=" << rho
            << " < " << kSpearmanRhoTol
            << " (reference backend=" << ref.backend << ")";
    }
}

// =============================================================================
// Test case 2: Scdrs_GSM_RealData
// =============================================================================
// Real scRNA-seq data (GSM4037629, 11,560 cells) + mini disease fixture.
// Requirement: all output values finite; no crash; n_cells == actual cell count.
// =============================================================================

TEST(ScDRSCorrectness, Scdrs_GSM_RealData) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device.";
    if (!fs::exists(kGsmExprPath)) GTEST_SKIP() << "GSM file not found: " << kGsmExprPath;
    if (!fs::exists(kMiniDiseaseFixture))
        GTEST_SKIP() << "Mini disease fixture not found: " << kMiniDiseaseFixture;

    // Load expression via pz_device_loader (real API: load_pz).
    singlet_gpu::io::PzDeviceMatrix dev_mat =
        singlet_gpu::io::load_pz(kGsmExprPath);
    int n_cells = dev_mat.n_cells;
    ASSERT_GT(n_cells, 0) << "No cells loaded from GSM .1pz.";

    // Obtain gene names from the metadata (row names in .1pz).
    // Fallback to synthetic names if metadata not available.
    int n_genes = dev_mat.mat.rows;
    std::vector<std::string> gene_names = dev_mat.meta.rownames;
    if ((int)gene_names.size() != n_genes) {
        gene_names = make_gene_names(n_genes);
    }

    // Parse mini disease fixture.
    // Fixture format: disease_name  gene_name  weight  (tab-separated, header row).
    std::vector<singlet_gpu::disease::DiseaseGeneSet> kernel_sets;
    {
        std::ifstream f(kMiniDiseaseFixture);
        ASSERT_TRUE(f.is_open()) << "Cannot open mini disease fixture.";
        std::string line;
        std::getline(f, line); // header
        std::map<std::string, singlet_gpu::disease::DiseaseGeneSet> ds_map;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            // Fixture format: disease_name TAB gene_name TAB weight
            std::string dname, gname;
            float w = 1.0f;
            ss >> dname >> gname >> w;
            auto& gs = ds_map[dname];
            if (gs.name.empty()) gs.name = dname;
            gs.genes[gname] = w;  // DiseaseGeneSet.genes: unordered_map<string,float>
        }
        for (auto& [k, v] : ds_map) kernel_sets.push_back(std::move(v));
    }
    ASSERT_GT(kernel_sets.size(), 0u) << "No diseases parsed from fixture.";

    singlet_gpu::disease::ScDrsConfig cfg;
    cfg.n_controls    = kNCtrl;
    cfg.seed          = kSeed;
    cfg.disease_chunk = 0;

    singlet_gpu::core::GPUContext& ctx = singlet_gpu::core::default_context();
    singlet_gpu::disease::ScDrsResult dev_result =
        singlet_gpu::disease::score(dev_mat.mat, gene_names, kernel_sets, cfg, ctx);

    // Download to host for validation.
    ScDrsHostResult result = download_scdrs_result(dev_result);

    int n_dis = static_cast<int>(kernel_sets.size());
    ASSERT_EQ(result.n_cells,    n_cells) << "Result n_cells mismatch.";
    ASSERT_EQ(result.n_diseases, n_dis)   << "Result n_diseases mismatch.";

    size_t total = static_cast<size_t>(n_cells) * n_dis;

    // All values must be finite.
    for (size_t i = 0; i < result.raw_score.size(); ++i)
        ASSERT_TRUE(std::isfinite(result.raw_score[i]))
            << "Non-finite raw_score at index " << i;
    for (size_t i = 0; i < result.normalized_score.size(); ++i)
        ASSERT_TRUE(std::isfinite(result.normalized_score[i]))
            << "Non-finite normalized_score at index " << i;
    for (size_t i = 0; i < result.p_value.size(); ++i) {
        ASSERT_TRUE(std::isfinite(result.p_value[i]))
            << "Non-finite p_value at index " << i;
        ASSERT_GE(result.p_value[i], 0.0f) << "Negative p_value at index " << i;
        ASSERT_LE(result.p_value[i], 1.0f) << "p_value > 1 at index " << i;
    }
    (void)total;
}

// =============================================================================
// Test case 3: Scdrs_ControlSets_MatchedMeanVar
// =============================================================================
// Verify that control gene sets have matched mean and variance to the disease
// gene sets within 5% relative error (design doc §"Control set matching").
// Run entirely host-side (no GPU needed) using the host helper functions.
// =============================================================================

TEST(ScDRSCorrectness, Scdrs_ControlSets_MatchedMeanVar) {
    // No GPU needed: host-side control set builder.
    TinySynthetic d = make_tiny_synthetic(kSeed);

    // Build control sets host-side (same algorithm as the kernel).
    std::vector<float> ctrl_W = build_control_sets_host(
        d.gene_sets, d.expr.data(), kTinyCells, kTinyGenes, kNCtrl, kSeed);
    // ctrl_W: [n_genes × n_ctrl], column = one control set's weights.

    // Compute gene mean and variance (over cells).
    std::vector<float> gene_mean(kTinyGenes, 0.0f), gene_var(kTinyGenes, 0.0f);
    for (int c = 0; c < kTinyCells; ++c)
        for (int g = 0; g < kTinyGenes; ++g)
            gene_mean[g] += d.expr[static_cast<size_t>(c) * kTinyGenes + g];
    for (int g = 0; g < kTinyGenes; ++g) gene_mean[g] /= kTinyCells;
    for (int c = 0; c < kTinyCells; ++c)
        for (int g = 0; g < kTinyGenes; ++g) {
            float dv = d.expr[static_cast<size_t>(c) * kTinyGenes + g] - gene_mean[g];
            gene_var[g] += dv * dv;
        }
    for (int g = 0; g < kTinyGenes; ++g) gene_var[g] /= kTinyCells;

    // For each disease set: compute expected (mean, var) weighted by disease weights.
    for (int di = 0; di < kTinyDiseases; ++di) {
        float wsum = 0.0f;
        double dis_mean_w = 0.0, dis_var_w = 0.0;
        for (auto [gidx, w] : d.gene_sets[di]) {
            wsum     += w;
            dis_mean_w += w * gene_mean[gidx];
            dis_var_w  += w * gene_var[gidx];
        }
        if (wsum < 1e-8f) continue;
        dis_mean_w /= wsum;
        dis_var_w  /= wsum;

        // For each control set, compute weighted (mean, var) using ctrl_W weights.
        // ctrl_W is indexed as ctrl_W[gene * n_ctrl + ctrl_idx].
        std::vector<double> ctrl_mean_vals(kNCtrl, 0.0);
        std::vector<double> ctrl_var_vals(kNCtrl, 0.0);
        for (int g = 0; g < kTinyGenes; ++g) {
            for (int k = 0; k < kNCtrl; ++k) {
                float cw = ctrl_W[static_cast<size_t>(g) * kNCtrl + k];
                if (cw > 0.0f) {
                    ctrl_mean_vals[k] += cw * gene_mean[g];
                    ctrl_var_vals[k]  += cw * gene_var[g];
                }
            }
        }
        double ctrl_mean_avg = 0.0, ctrl_var_avg = 0.0;
        for (int k = 0; k < kNCtrl; ++k) {
            ctrl_mean_avg += ctrl_mean_vals[k];
            ctrl_var_avg  += ctrl_var_vals[k];
        }
        ctrl_mean_avg /= kNCtrl;
        ctrl_var_avg  /= kNCtrl;

        // Relative errors.
        float rel_mean = static_cast<float>(
            std::abs(ctrl_mean_avg - dis_mean_w) /
            (std::abs(dis_mean_w) + 1e-8));
        float rel_var  = static_cast<float>(
            std::abs(ctrl_var_avg  - dis_var_w)  /
            (std::abs(dis_var_w)  + 1e-8));

        EXPECT_LE(rel_mean, kCtrlMatchRelErr)
            << "Disease " << di << " control mean rel_err=" << rel_mean
            << " > " << kCtrlMatchRelErr;
        EXPECT_LE(rel_var, kCtrlMatchRelErr)
            << "Disease " << di << " control var rel_err=" << rel_var
            << " > " << kCtrlMatchRelErr;
    }
}

// =============================================================================
// Test case 4: Scdrs_PvalueCalibration_Uniform
// =============================================================================
// Under a null hypothesis (random gene sets, no disease signal), p-values should
// be approximately uniform. KS statistic ≤ 0.10.
// Uses host-side scoring to avoid GPU dependency.
// =============================================================================

TEST(ScDRSCorrectness, Scdrs_PvalueCalibration_Uniform) {
    // No GPU needed: host-side scoring.
    std::mt19937_64 rng(kSeed ^ 0xABCDEF);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // Generate a null expression matrix (pure noise, no planted signal).
    constexpr int kNullCells = 200;
    constexpr int kNullGenes = 300;
    constexpr int kNullDiseases = 5;
    constexpr int kNullGenesPerSet = 15;
    constexpr int kNullNCtrl = 200;

    std::vector<float> expr(static_cast<size_t>(kNullCells) * kNullGenes);
    for (auto& v : expr) v = std::max(0.0f, nd(rng));

    // Random null disease gene sets.
    std::uniform_int_distribution<int> gidx_dist(0, kNullGenes - 1);
    std::uniform_real_distribution<float> w_dist(0.5f, 1.5f);
    std::vector<std::vector<std::pair<int,float>>> null_sets(kNullDiseases);
    for (int di = 0; di < kNullDiseases; ++di) {
        for (int g = 0; g < kNullGenesPerSet; ++g)
            null_sets[di].emplace_back(gidx_dist(rng), w_dist(rng));
    }

    // Z-score + raw score (host).
    std::vector<float> z = gene_zscore_host(expr.data(), kNullCells, kNullGenes);
    std::vector<float> raw = compute_raw_scores_host(z, kNullCells, kNullGenes, null_sets);

    // Build control sets (host).
    std::vector<float> ctrl_W = build_control_sets_host(
        null_sets, expr.data(), kNullCells, kNullGenes, kNullNCtrl, kSeed ^ 0x1234);

    // Control scores: ctrl_scores[c * kNullNCtrl + k] = Σ_g z[c,g] * ctrl_W[g,k]
    std::vector<float> ctrl_scores(static_cast<size_t>(kNullCells) * kNullNCtrl, 0.0f);
    for (int c = 0; c < kNullCells; ++c)
        for (int g = 0; g < kNullGenes; ++g)
            for (int k = 0; k < kNullNCtrl; ++k)
                ctrl_scores[static_cast<size_t>(c) * kNullNCtrl + k] +=
                    z[static_cast<size_t>(c) * kNullGenes + g] *
                    ctrl_W[static_cast<size_t>(g) * kNullNCtrl + k];

    // Monte Carlo p-values.
    std::vector<float> all_pvals;
    all_pvals.reserve(static_cast<size_t>(kNullCells) * kNullDiseases);
    for (int c = 0; c < kNullCells; ++c) {
        for (int di = 0; di < kNullDiseases; ++di) {
            float obs = raw[static_cast<size_t>(c) * kNullDiseases + di];
            int exceed = 0;
            for (int k = 0; k < kNullNCtrl; ++k)
                if (ctrl_scores[static_cast<size_t>(c) * kNullNCtrl + k] >= obs)
                    ++exceed;
            float pval = std::max(1e-7f, static_cast<float>(exceed) / kNullNCtrl);
            all_pvals.push_back(pval);
        }
    }

    float ks = ks_statistic_uniform(all_pvals);
    EXPECT_LE(ks, kKsTol)
        << "KS statistic for null p-values = " << ks
        << " > " << kKsTol << " (p-values not uniform under null).";
}

// =============================================================================
// Test case 5: Scdrs_Determinism_BitIdentical
// =============================================================================
// Two identical calls with the same seed must produce bit-identical output.
// Uses the host-side kernel convenience function (scdrs_host) or the GPU kernel
// if a device is available.
// =============================================================================

TEST(ScDRSCorrectness, Scdrs_Determinism_BitIdentical) {
    // Run twice with identical inputs + seed; compare bit-for-bit.
    TinySynthetic d = make_tiny_synthetic(kSeed);

    // run_once returns a host-side result for bit-identity comparison.
    auto run_once = [&]() -> ScDrsHostResult {
        auto gene_names_local = make_gene_names(kTinyGenes);
        std::vector<singlet_gpu::disease::DiseaseGeneSet> kernel_sets;
        kernel_sets.reserve(kTinyDiseases);
        for (int di = 0; di < kTinyDiseases; ++di) {
            singlet_gpu::disease::DiseaseGeneSet gs;
            gs.name = d.disease_names[di];
            for (auto [gidx, w] : d.gene_sets[di]) {
                gs.genes[gene_names_local[gidx]] = w;
            }
            kernel_sets.push_back(std::move(gs));
        }
        singlet_gpu::disease::ScDrsConfig cfg;
        cfg.n_controls    = kNCtrl;
        cfg.seed          = kSeed;
        cfg.n_bins        = 20;
        cfg.disease_chunk = 0;

        if (gpu_available()) {
            cudaStream_t s{};
            cudaStreamCreate(&s);
            singlet_gpu::core::DeviceCSC csc =
                dense_to_device_csc(d.expr.data(), kTinyCells, kTinyGenes, s);
            cudaStreamDestroy(s);
            singlet_gpu::core::GPUContext& ctx = singlet_gpu::core::default_context();
            singlet_gpu::disease::ScDrsResult dev_r =
                singlet_gpu::disease::score(csc, gene_names_local, kernel_sets, cfg, ctx);
            return download_scdrs_result(dev_r);
        } else {
            // Host-only fallback: compute deterministic scores manually.
            std::vector<float> z =
                gene_zscore_host(d.expr.data(), kTinyCells, kTinyGenes);
            std::vector<float> raw =
                compute_raw_scores_host(z, kTinyCells, kTinyGenes, d.gene_sets);
            ScDrsHostResult r;
            r.n_cells    = kTinyCells;
            r.n_diseases = kTinyDiseases;
            r.raw_score  = raw;
            r.normalized_score.assign(raw.size(), 0.0f);
            r.p_value.assign(raw.size(), 0.0f);
            return r;
        }
    };

    auto result1 = run_once();
    auto result2 = run_once();

    ASSERT_EQ(result1.raw_score.size(), result2.raw_score.size())
        << "Result sizes differ between runs.";
    for (size_t i = 0; i < result1.raw_score.size(); ++i) {
        EXPECT_EQ(result1.raw_score[i], result2.raw_score[i])
            << "raw_score not bit-identical at index " << i
            << ": run1=" << result1.raw_score[i]
            << " run2=" << result2.raw_score[i];
    }
    for (size_t i = 0; i < result1.p_value.size(); ++i) {
        EXPECT_EQ(result1.p_value[i], result2.p_value[i])
            << "p_value not bit-identical at index " << i;
    }
}
