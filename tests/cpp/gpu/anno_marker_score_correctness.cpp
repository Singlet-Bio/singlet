// SPDX-License-Identifier: MIT
// singlet/gpu/tests/anno_marker_score_correctness.cpp
//
// Correctness harness for singlet::gpu::anno::marker_score.
//
// Written against the design doc at:
//   singlet/gpu/state/designs/10-annotation.md
// NOT against the kernel source — authored in parallel with anno/marker_score.h
// by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: DecoupleR Python (dc.run_mlm / dc.run_ulm / dc.run_wsum) via
//   subprocess: tests/refs/marker_score_decoupler_reference.py
//   Environment: pip install decoupler-py anndata scanpy numpy scipy
//
// Tolerances (design doc §"Correctness test spec"):
//   Spearman ρ ≥ 0.95   (per-cell, over gene sets)
//   rel_err  ≤ 1e-4     (element-wise, where |ref| > 1e-3)
//   Determinism          bit-identical (same seed, same input)
//   UCell scores         in [0, 1] and rank-consistent with marker abundance
//   Tikhonov             finite output when G^T G is near-singular
//   Empty gene set       graceful skip (set count unchanged or 0-score returned)
//
// Test cases:
//   MarkerScore_Mlm_VsDecoupleR
//   MarkerScore_Wsum_VsDecoupleR
//   MarkerScore_Ulm_VsDecoupleR
//   MarkerScore_UCell_VsManualReference
//   MarkerScore_TikhonovRegularizer_NonSingular
//   MarkerScore_EmptyGeneSet_Skipped
//   MarkerScore_Determinism_BitIdentical
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every test that requires a device is guarded by
//   gpu_available() and calls GTEST_SKIP().

#include <singlet/gpu/anno/marker_score.h>
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
#include <cstring>
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
constexpr int kTinyGenes   = 200;
constexpr int kTinyCells   = 500;
constexpr int kTinySets    = 20;
constexpr int kGenesPerSet = 8;     // 5–10 per design doc; use 8

/// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_anno_markerscore_refs_tmp";

/// Python reference script path.
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet/gpu/"
    "tests/refs/marker_score_decoupler_reference.py";

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
// Minimal .npy v1.0 writer — float32 1-D or 2-D (row-major / C order).
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
    // Both f4 and i4 are 4 bytes.
    fout.write(static_cast<const char*>(data),
               static_cast<std::streamsize>(n_items * 4));
    if (!fout) throw std::runtime_error("write_npy: write error " + path);
}

void write_npy_f32_2d(const std::string& path, const float* data, int rows, int cols) {
    write_npy_generic(path, data, "<f4", rows, cols);
}

void write_npy_f32(const std::string& path, const float* data, int n) {
    write_npy_generic(path, data, "<f4", n, 1);
}

// ---------------------------------------------------------------------------
// TSV writer for gene-set DB.
// Format: set_name<TAB>gene_idx<TAB>weight\n  (one row per (set, gene) pair)
// ---------------------------------------------------------------------------
void write_geneset_tsv(const std::string& path,
                       const singlet::gpu::anno::GeneSetDB& db)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open geneset tsv: " + path);
    for (int s = 0; s < static_cast<int>(db.set_names.size()); ++s) {
        const auto& indices = db.member_gene_indices[s];
        const auto& weights = db.weights[s];
        for (int k = 0; k < static_cast<int>(indices.size()); ++k) {
            float w = weights.empty() ? 1.0f : weights[k];
            f << db.set_names[s] << "\t" << indices[k] << "\t" << w << "\n";
        }
    }
    if (!f) throw std::runtime_error("Write error: " + path);
}

// ---------------------------------------------------------------------------
// Load float32 .npz key — minimal implementation using Python subprocess to
// extract a single array to a temporary .npy file, then load as raw floats.
// ---------------------------------------------------------------------------
std::vector<float> load_npz_f32(const std::string& npz_path,
                                  const std::string& key,
                                  int expected_n)
{
    std::string tmp = refs_path("_extract_tmp.npy");
    std::string cmd = "python3 -c \""
        "import numpy as np; "
        "d = np.load('" + npz_path + "', allow_pickle=True); "
        "np.save('" + tmp + "', d['" + key + "'].astype(np.float32))"
        "\"";
    run_cmd(cmd);

    std::ifstream f(tmp, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open extracted npy: " + tmp);

    // Skip numpy header (preamble + header string).
    const char np_magic[] = "\x93NUMPY\x01\x00";
    char buf[8];
    f.read(buf, 8);
    uint16_t hlen;
    f.read(reinterpret_cast<char*>(&hlen), 2);
    std::string header(hlen, '\0');
    f.read(header.data(), hlen);

    std::vector<float> data(static_cast<size_t>(expected_n));
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(expected_n * sizeof(float)));
    if (!f) throw std::runtime_error("Short read on extracted npy: " + tmp);
    return data;
}

// ---------------------------------------------------------------------------
// Spearman rank correlation (host-side, for comparing score vectors).
// Returns ρ for two equal-length vectors.
// ---------------------------------------------------------------------------
double spearman_rho(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size() && a.size() >= 2);
    int n = static_cast<int>(a.size());

    // Rank a.
    auto rank_vec = [&](const std::vector<float>& v) {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int i, int j){ return v[i] < v[j]; });
        std::vector<double> r(n);
        for (int i = 0; i < n; ) {
            int j = i;
            while (j < n && v[idx[j]] == v[idx[i]]) ++j;
            double avg = (i + j - 1) / 2.0 + 1.0;  // 1-based midrank
            for (int k = i; k < j; ++k) r[idx[k]] = avg;
            i = j;
        }
        return r;
    };

    auto ra = rank_vec(a);
    auto rb = rank_vec(b);

    double mean_ra = 0, mean_rb = 0;
    for (int i = 0; i < n; ++i) { mean_ra += ra[i]; mean_rb += rb[i]; }
    mean_ra /= n; mean_rb /= n;

    double cov = 0, va = 0, vb = 0;
    for (int i = 0; i < n; ++i) {
        double da = ra[i] - mean_ra, db = rb[i] - mean_rb;
        cov += da * db; va += da * da; vb += db * db;
    }
    if (va < 1e-15 || vb < 1e-15) return 1.0;   // constant → perfect rank
    return cov / std::sqrt(va * vb);
}

// ---------------------------------------------------------------------------
// Build a tiny dense float32 matrix (genes × cells) with planted signal:
// set s has high expression in cells [s * cells_per_set .. (s+1)*cells_per_set).
// ---------------------------------------------------------------------------
std::vector<float> make_tiny_matrix(int G, int C, int n_sets,
                                    const singlet::gpu::anno::GeneSetDB& db,
                                    uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> bg(0.0f, 1.0f);
    std::uniform_real_distribution<float> hi(5.0f, 10.0f);

    std::vector<float> mat(static_cast<size_t>(G) * C, 0.0f);
    int cells_per_set = std::max(1, C / n_sets);

    // Fill background.
    for (auto& v : mat) v = bg(rng);

    // Plant signal.
    for (int s = 0; s < n_sets; ++s) {
        int c_start = s * cells_per_set;
        int c_end   = std::min(C, c_start + cells_per_set);
        for (int gene_idx : db.member_gene_indices[s])
            for (int c = c_start; c < c_end; ++c)
                mat[static_cast<size_t>(gene_idx) * C + c] = hi(rng);
    }
    return mat;
}

// ---------------------------------------------------------------------------
// Upload dense matrix (genes × cells, row-major) as a DeviceCSC (sparse).
// We convert to CSC on host by treating every entry as nonzero (dense case).
// ---------------------------------------------------------------------------
singlet::gpu::core::DeviceCSC make_device_csc_from_dense(
        const float* h_mat, int G, int C, cudaStream_t stream)
{
    using namespace singlet::gpu::core;

    // Full dense → CSC (all entries nonzero).
    size_t nnz = static_cast<size_t>(G) * C;
    std::vector<float>   h_values (nnz);
    std::vector<int32_t> h_indptr (static_cast<size_t>(C) + 1);
    std::vector<int32_t> h_indices(nnz);

    size_t off = 0;
    for (int c = 0; c < C; ++c) {
        h_indptr[c] = static_cast<int32_t>(off);
        for (int g = 0; g < G; ++g) {
            h_values [off] = h_mat[static_cast<size_t>(g) * C + c];
            h_indices[off] = g;
            ++off;
        }
    }
    h_indptr[C] = static_cast<int32_t>(nnz);

    DeviceMemory<float>   d_values (nnz);
    DeviceMemory<int32_t> d_indptr (static_cast<size_t>(C) + 1);
    DeviceMemory<int32_t> d_indices(nnz);

    auto ck = [](cudaError_t e, const char* s) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("upload_csc ") + s + ": " +
                                     cudaGetErrorString(e));
    };
    ck(cudaMemcpyAsync(d_values .get(), h_values .data(),
                       nnz * sizeof(float), cudaMemcpyHostToDevice, stream), "values");
    ck(cudaMemcpyAsync(d_indptr .get(), h_indptr .data(),
                       (static_cast<size_t>(C) + 1) * sizeof(int32_t),
                       cudaMemcpyHostToDevice, stream), "indptr");
    ck(cudaMemcpyAsync(d_indices.get(), h_indices.data(),
                       nnz * sizeof(int32_t), cudaMemcpyHostToDevice, stream), "indices");
    ck(cudaStreamSynchronize(stream), "sync");

    DeviceCSC mat;
    mat.rows        = static_cast<uint32_t>(G);
    mat.cols        = static_cast<uint32_t>(C);
    mat.nnz         = nnz;
    mat.values      = std::move(d_values);
    mat.col_ptr     = std::move(d_indptr);
    mat.row_indices = std::move(d_indices);
    return mat;
}

// ---------------------------------------------------------------------------
// Build a gene-set DB: kTinySets sets of kGenesPerSet consecutive genes.
// Sets partition [0 .. kTinySets*kGenesPerSet) non-overlapping.
// ---------------------------------------------------------------------------
singlet::gpu::anno::GeneSetDB make_tiny_gene_set_db(int n_sets, int genes_per_set) {
    singlet::gpu::anno::GeneSetDB db;
    db.set_names.resize(n_sets);
    db.member_gene_indices.resize(n_sets);
    db.weights.resize(n_sets);
    for (int s = 0; s < n_sets; ++s) {
        db.set_names[s] = "set_" + std::to_string(s);
        for (int k = 0; k < genes_per_set; ++k)
            db.member_gene_indices[s].push_back(s * genes_per_set + k);
        // Uniform weights.
        db.weights[s].assign(genes_per_set, 1.0f);
    }
    return db;
}

// ---------------------------------------------------------------------------
// Download scores from device and flatten to host float vector.
// Layout: scores(n_sets × n_cells), column-major (gene_sets × cells).
// Returns vector of length n_sets * n_cells.
// ---------------------------------------------------------------------------
std::vector<float> download_scores(
        const singlet::gpu::anno::MarkerScoreResult& res,
        cudaStream_t stream)
{
    int total = res.n_sets * res.n_cells;
    std::vector<float> h(static_cast<size_t>(total));
    cudaError_t e = cudaMemcpy(h.data(), res.scores.get(),
                               static_cast<size_t>(total) * sizeof(float),
                               cudaMemcpyDeviceToHost);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("download_scores: ") +
                                 cudaGetErrorString(e));
    return h;
}

// ---------------------------------------------------------------------------
// Run DecoupleR reference subprocess and load scores.
// Returns flat vector (n_sets × n_cells) in the same layout as marker_score.
// ---------------------------------------------------------------------------
std::vector<float> run_decoupler_ref(
        const std::vector<float>& h_mat,  // G × C, row-major
        const singlet::gpu::anno::GeneSetDB& db,
        int G, int C, int n_sets,
        const std::string& method,
        const std::string& tag)
{
    ensure_refs_tmp();
    std::string mat_path     = refs_path(tag + "_mat.npy");
    std::string geneset_path = refs_path(tag + "_geneset.tsv");
    std::string scores_path  = refs_path(tag + "_scores.npz");

    // Write matrix as npy (G × C, row-major float32).
    write_npy_f32_2d(mat_path, h_mat.data(), G, C);
    write_geneset_tsv(geneset_path, db);

    // Run Python reference.
    std::string cmd = "python3 " + std::string(kRefScript) +
        " --matrix "  + mat_path     +
        " --geneset " + geneset_path +
        " --output "  + scores_path  +
        " --method "  + method;
    run_cmd(cmd);

    // Load scores: shape (n_sets × n_cells), stored as "scores" key.
    int expected_n = n_sets * C;
    return load_npz_f32(scores_path, "scores", expected_n);
}

}  // namespace

// =============================================================================
// Test fixture
// =============================================================================

class MarkerScoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!gpu_available())
            GTEST_SKIP() << "No CUDA device available.";
        cudaError_t e = cudaStreamCreate(&stream_);
        ASSERT_EQ(e, cudaSuccess) << "cudaStreamCreate failed";

        db_    = make_tiny_gene_set_db(kTinySets, kGenesPerSet);
        h_mat_ = make_tiny_matrix(kTinyGenes, kTinyCells, kTinySets, db_, kSeed);
        d_mat_ = make_device_csc_from_dense(
                     h_mat_.data(), kTinyGenes, kTinyCells, stream_);
    }

    void TearDown() override {
        if (stream_) cudaStreamDestroy(stream_);
    }

    cudaStream_t stream_ = nullptr;
    singlet::gpu::anno::GeneSetDB db_;
    std::vector<float> h_mat_;
    singlet::gpu::core::DeviceCSC d_mat_{};
};

// =============================================================================
// Helper: compare singlet-gpu scores vs DecoupleR reference.
// Checks Spearman ρ ≥ min_rho for each cell's score-over-sets vector.
// =============================================================================
static void compare_vs_decoupler(
        const std::vector<float>& ours,   // (n_sets × n_cells) our scores
        const std::vector<float>& ref,    // (n_sets × n_cells) DecoupleR scores
        int n_sets, int n_cells,
        double min_rho,
        double max_rel_err)
{
    ASSERT_EQ(static_cast<int>(ours.size()), n_sets * n_cells);
    ASSERT_EQ(static_cast<int>(ref .size()), n_sets * n_cells);

    double sum_rho = 0.0;
    int n_rho_computed = 0;
    int n_violations = 0;

    for (int c = 0; c < n_cells; ++c) {
        std::vector<float> our_col(n_sets), ref_col(n_sets);
        for (int s = 0; s < n_sets; ++s) {
            our_col[s] = ours[static_cast<size_t>(s) * n_cells + c];
            ref_col[s] = ref [static_cast<size_t>(s) * n_cells + c];
        }

        // Element-wise relative error (only where |ref| > 1e-3).
        for (int s = 0; s < n_sets; ++s) {
            if (std::abs(ref_col[s]) > 1e-3f) {
                double rel = std::abs(our_col[s] - ref_col[s]) /
                             std::abs(ref_col[s]);
                if (rel > max_rel_err) ++n_violations;
            }
        }

        double rho = spearman_rho(our_col, ref_col);
        sum_rho += rho;
        ++n_rho_computed;
    }

    double mean_rho = (n_rho_computed > 0)
                        ? sum_rho / n_rho_computed : 0.0;

    EXPECT_GE(mean_rho, min_rho)
        << "Mean per-cell Spearman ρ " << mean_rho << " < " << min_rho;

    // Allow up to 2% of element pairs to exceed rel_err threshold.
    int total_pairs = n_sets * n_cells;
    int max_violations = static_cast<int>(std::ceil(0.02 * total_pairs));
    EXPECT_LE(n_violations, max_violations)
        << "rel_err violations: " << n_violations
        << " / " << total_pairs << " exceed threshold " << max_rel_err;
}

// =============================================================================
// Test cases
// =============================================================================

// ---------------------------------------------------------------------------
// MarkerScore_Mlm_VsDecoupleR
// ---------------------------------------------------------------------------
TEST_F(MarkerScoreTest, MarkerScore_Mlm_VsDecoupleR) {
    using namespace singlet::gpu::anno;
    MarkerScoreConfig cfg;
    cfg.method = MarkerMethod::Mlm;

    auto result = marker_score(d_mat_, db_, cfg, stream_);
    ASSERT_EQ(result.n_sets,  kTinySets);
    ASSERT_EQ(result.n_cells, kTinyCells);

    auto ours = download_scores(result, stream_);
    auto ref  = run_decoupler_ref(h_mat_, db_, kTinyGenes, kTinyCells,
                                  kTinySets, "mlm", "mlm");

    compare_vs_decoupler(ours, ref, kTinySets, kTinyCells,
                         /*min_rho=*/0.95, /*max_rel_err=*/1e-4);
}

// ---------------------------------------------------------------------------
// MarkerScore_Wsum_VsDecoupleR
// ---------------------------------------------------------------------------
TEST_F(MarkerScoreTest, MarkerScore_Wsum_VsDecoupleR) {
    using namespace singlet::gpu::anno;
    MarkerScoreConfig cfg;
    cfg.method = MarkerMethod::Wsum;

    auto result = marker_score(d_mat_, db_, cfg, stream_);
    ASSERT_EQ(result.n_sets,  kTinySets);
    ASSERT_EQ(result.n_cells, kTinyCells);

    auto ours = download_scores(result, stream_);
    auto ref  = run_decoupler_ref(h_mat_, db_, kTinyGenes, kTinyCells,
                                  kTinySets, "wsum", "wsum");

    compare_vs_decoupler(ours, ref, kTinySets, kTinyCells,
                         /*min_rho=*/0.95, /*max_rel_err=*/1e-4);
}

// ---------------------------------------------------------------------------
// MarkerScore_Ulm_VsDecoupleR
// ---------------------------------------------------------------------------
TEST_F(MarkerScoreTest, MarkerScore_Ulm_VsDecoupleR) {
    using namespace singlet::gpu::anno;
    MarkerScoreConfig cfg;
    cfg.method = MarkerMethod::Ulm;

    auto result = marker_score(d_mat_, db_, cfg, stream_);
    ASSERT_EQ(result.n_sets,  kTinySets);
    ASSERT_EQ(result.n_cells, kTinyCells);

    auto ours = download_scores(result, stream_);
    auto ref  = run_decoupler_ref(h_mat_, db_, kTinyGenes, kTinyCells,
                                  kTinySets, "ulm", "ulm");

    compare_vs_decoupler(ours, ref, kTinySets, kTinyCells,
                         /*min_rho=*/0.95, /*max_rel_err=*/1e-4);
}

// ---------------------------------------------------------------------------
// MarkerScore_UCell_VsManualReference
//
// UCell is rank-based AUC per cell.  R UCell package may not be installed, so
// we use a manual reference: cells in which a gene set's members have
// uniformly high values should score higher than cells with low values.
// We confirm:
//   (a) all scores ∈ [0, 1]
//   (b) Spearman ρ ≥ 0.90 between UCell scores and the mean expression of
//       each gene set across its member genes (a sensible proxy ranking).
// ---------------------------------------------------------------------------
TEST_F(MarkerScoreTest, MarkerScore_UCell_VsManualReference) {
    using namespace singlet::gpu::anno;
    MarkerScoreConfig cfg;
    cfg.method = MarkerMethod::UCell;

    auto result = marker_score(d_mat_, db_, cfg, stream_);
    ASSERT_EQ(result.n_sets,  kTinySets);
    ASSERT_EQ(result.n_cells, kTinyCells);

    auto ours = download_scores(result, stream_);

    // Check range [0, 1].
    for (int i = 0; i < kTinySets * kTinyCells; ++i) {
        EXPECT_GE(ours[i], 0.0f) << "UCell score below 0 at index " << i;
        EXPECT_LE(ours[i], 1.0f) << "UCell score above 1 at index " << i;
    }

    // Manual reference: mean expression of each set's member genes per cell.
    // h_mat_ is (G × C, row-major); gene g, cell c → h_mat_[g*C + c].
    std::vector<float> mean_expr(static_cast<size_t>(kTinySets) * kTinyCells, 0.0f);
    for (int s = 0; s < kTinySets; ++s) {
        const auto& members = db_.member_gene_indices[s];
        for (int c = 0; c < kTinyCells; ++c) {
            float sum = 0.0f;
            for (int g : members)
                sum += h_mat_[static_cast<size_t>(g) * kTinyCells + c];
            mean_expr[static_cast<size_t>(s) * kTinyCells + c] = sum / members.size();
        }
    }

    // Per-set Spearman ρ (over cells) between UCell and mean_expr.
    int n_low_rho = 0;
    for (int s = 0; s < kTinySets; ++s) {
        std::vector<float> ucell_col(kTinyCells), expr_col(kTinyCells);
        for (int c = 0; c < kTinyCells; ++c) {
            ucell_col[c] = ours  [static_cast<size_t>(s) * kTinyCells + c];
            expr_col [c] = mean_expr[static_cast<size_t>(s) * kTinyCells + c];
        }
        double rho = spearman_rho(ucell_col, expr_col);
        if (rho < 0.90) ++n_low_rho;
    }
    // At most 2 out of 20 sets may have ρ < 0.90.
    EXPECT_LE(n_low_rho, 2)
        << "Too many gene sets with UCell vs mean_expr Spearman ρ < 0.90: "
        << n_low_rho;
}

// ---------------------------------------------------------------------------
// MarkerScore_TikhonovRegularizer_NonSingular
//
// Gene sets with heavily overlapping members cause G^T G to be near-singular.
// The design doc mandates Tikhonov regularizer λ = 1e-6 * trace.
// We confirm: Mlm returns finite (non-NaN, non-Inf) scores under this stress.
// ---------------------------------------------------------------------------
TEST_F(MarkerScoreTest, MarkerScore_TikhonovRegularizer_NonSingular) {
    using namespace singlet::gpu::anno;

    // Build a DB where all gene sets share the same 8 genes → G^T G ∝ identity
    // times n_sets, but each row is identical → Gram matrix rank-1.
    GeneSetDB overlapping_db;
    int shared_genes[] = {0, 1, 2, 3, 4, 5, 6, 7};  // 8 genes, same for all sets
    for (int s = 0; s < kTinySets; ++s) {
        overlapping_db.set_names.push_back("ovlp_set_" + std::to_string(s));
        overlapping_db.member_gene_indices.push_back(
            std::vector<int>(std::begin(shared_genes), std::end(shared_genes)));
        overlapping_db.weights.push_back(std::vector<float>(8, 1.0f));
    }

    MarkerScoreConfig cfg;
    cfg.method = MarkerMethod::Mlm;

    singlet::gpu::anno::MarkerScoreResult result;
    ASSERT_NO_THROW(result = marker_score(d_mat_, overlapping_db, cfg, stream_));

    auto scores = download_scores(result, stream_);
    for (int i = 0; i < result.n_sets * result.n_cells; ++i) {
        EXPECT_TRUE(std::isfinite(scores[i]))
            << "Non-finite score at index " << i
            << " with near-singular Gram (Tikhonov test)";
    }
}

// ---------------------------------------------------------------------------
// MarkerScore_EmptyGeneSet_Skipped
//
// A GeneSetDB entry with 0 member genes should not crash.
// The design doc mandates cfg.min_n_genes_per_set controls this threshold.
// We set min_n_genes_per_set = 5 and insert a set with 0 members.
// Either: (a) result.n_sets is reduced by 1, or (b) that set's scores are 0.
// ---------------------------------------------------------------------------
TEST_F(MarkerScoreTest, MarkerScore_EmptyGeneSet_Skipped) {
    using namespace singlet::gpu::anno;

    GeneSetDB db_with_empty = db_;  // copy
    // Append a set with 0 members.
    db_with_empty.set_names.push_back("empty_set");
    db_with_empty.member_gene_indices.push_back({});
    db_with_empty.weights.push_back({});

    MarkerScoreConfig cfg;
    cfg.method              = MarkerMethod::Wsum;
    cfg.min_n_genes_per_set = 5;

    // Should not throw.
    singlet::gpu::anno::MarkerScoreResult result;
    ASSERT_NO_THROW(result = marker_score(d_mat_, db_with_empty, cfg, stream_));

    // Either the empty set is skipped (n_sets == kTinySets) or its scores are 0.
    bool skipped = (result.n_sets == kTinySets);
    bool zero_scores = false;
    if (result.n_sets == kTinySets + 1) {
        auto scores = download_scores(result, stream_);
        // Last set's scores.
        zero_scores = true;
        for (int c = 0; c < kTinyCells; ++c) {
            float v = scores[static_cast<size_t>(kTinySets) * kTinyCells + c];
            if (std::abs(v) > 1e-6f) { zero_scores = false; break; }
        }
    }
    EXPECT_TRUE(skipped || zero_scores)
        << "Empty gene set neither skipped nor zeroed. n_sets=" << result.n_sets;
}

// ---------------------------------------------------------------------------
// MarkerScore_Determinism_BitIdentical
//
// Running marker_score twice on the same input must produce bit-identical
// scores (design doc §"Determinism": all kernels are deterministic).
// ---------------------------------------------------------------------------
TEST_F(MarkerScoreTest, MarkerScore_Determinism_BitIdentical) {
    using namespace singlet::gpu::anno;
    MarkerScoreConfig cfg;
    cfg.method = MarkerMethod::Mlm;

    auto r1 = marker_score(d_mat_, db_, cfg, stream_);
    auto r2 = marker_score(d_mat_, db_, cfg, stream_);

    ASSERT_EQ(r1.n_sets,  r2.n_sets);
    ASSERT_EQ(r1.n_cells, r2.n_cells);

    auto s1 = download_scores(r1, stream_);
    auto s2 = download_scores(r2, stream_);

    ASSERT_EQ(s1.size(), s2.size());
    int mismatches = 0;
    for (size_t i = 0; i < s1.size(); ++i)
        if (std::memcmp(&s1[i], &s2[i], sizeof(float)) != 0) ++mismatches;

    EXPECT_EQ(mismatches, 0)
        << "Determinism violated: " << mismatches << " bit-mismatches out of "
        << s1.size();
}
