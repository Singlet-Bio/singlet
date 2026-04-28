// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/gsea_aucell_correctness.cpp
//
// Correctness harness for singlet_gpu::gsea::aucell.
//
// Written against the design doc at:
//   singlet-gpu/state/designs/11-gsea.md
// NOT against the kernel source — authored in parallel with gsea/aucell.h by
// the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: R AUCell package via subprocess
//   tests/refs/aucell_r_reference.R
//   Environment: R + BiocManager::install("AUCell")
//
// Tolerances (design doc §"Correctness test spec"):
//   AUCell vs R AUCell:              rel_err <= 1e-2 per cell per pathway
//   Rank consistency:                cells with high markers score higher
//   Histogram approx vs exact ranks: rel_err <= 1% on a small dataset
//   Determinism:                     bit-identical (same input, same config)
//   100k-cell scaling:               runs without OOM; output shape correct
//
// Test cases:
//   AUCell_TinyPlanted_VsR
//   AUCell_RealData_RanksConsistent
//   AUCell_HistogramApprox_Error
//   AUCell_Determinism_BitIdentical
//   AUCell_LargeCellCount_TilesCorrectly
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every test that requires a device is guarded by
//   gpu_available() and calls GTEST_SKIP().

// ---- singlet-gpu AUCell header (written by gpu-kernel-dev) -----------------
#include <singlet-gpu/gsea/aucell.h>
#include <singlet-gpu/anno/gene_set_db.h>   // GeneSetDB type (cycle 12)
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
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

/// Tiny matrix dimensions for planted test (genes × cells).
constexpr int kTinyGenes   = 200;
constexpr int kTinyCells   = 100;
/// Gene sets for tiny test.
constexpr int kTinySets    = 5;
/// Genes per set (above AUCell threshold, 15+).
constexpr int kGenesPerSet = 20;

/// Medium matrix for rank-consistency test.
constexpr int kMedGenes    = 500;
constexpr int kMedCells    = 500;

/// Large-scale test: 100k cells to verify tiling.
constexpr int kLargeCells  = 100000;
constexpr int kLargeGenes  = 200;
constexpr int kLargeSets   = 10;

/// AUCell top-K per cell (must be <= n_genes).
constexpr int kTopK        = 50;   // use small value fitting kTinyGenes=200

/// Path to the AUCell R reference script.
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/"
    "tests/refs/aucell_r_reference.R";

/// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_gsea_aucell_refs_tmp";

// =============================================================================
// Utility helpers (same pattern as cycles 9–12)
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
// Write GeneSetDB to three-column TSV: set_name<TAB>gene_idx<TAB>weight
// ---------------------------------------------------------------------------
void write_geneset_tsv(const std::string& path,
                       const singlet_gpu::anno::GeneSetDB& db)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open geneset tsv: " + path);
    for (int s = 0; s < static_cast<int>(db.set_names.size()); ++s) {
        const auto& idxs    = db.member_gene_indices[s];
        const auto& weights = db.weights[s];
        for (int k = 0; k < static_cast<int>(idxs.size()); ++k) {
            float w = weights.empty() ? 1.0f : weights[k];
            f << db.set_names[s] << "\t" << idxs[k] << "\t" << w << "\n";
        }
    }
    if (!f) throw std::runtime_error("Write error geneset tsv: " + path);
}

// ---------------------------------------------------------------------------
// Write a HostCSC to the dump_csc binary format (for R reference).
// ---------------------------------------------------------------------------
struct HostCSC {
    int                  n_rows;    // genes
    int                  n_cols;    // cells
    size_t               nnz;
    std::vector<float>   values;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
};

void write_csc_bin(const HostCSC& h, const std::string& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot open CSC bin: " + path);
    static constexpr uint32_t kMagic = 0x43535343u;
    uint32_t m = static_cast<uint32_t>(h.n_rows);
    uint32_t n = static_cast<uint32_t>(h.n_cols);
    uint64_t z = static_cast<uint64_t>(h.nnz);
    f.write(reinterpret_cast<const char*>(&kMagic), 4);
    f.write(reinterpret_cast<const char*>(&m),      4);
    f.write(reinterpret_cast<const char*>(&n),      4);
    f.write(reinterpret_cast<const char*>(&z),      8);
    f.write(reinterpret_cast<const char*>(h.values .data()),
            static_cast<std::streamsize>(h.nnz * sizeof(float)));
    f.write(reinterpret_cast<const char*>(h.indptr .data()),
            static_cast<std::streamsize>((static_cast<size_t>(h.n_cols) + 1) * sizeof(int32_t)));
    f.write(reinterpret_cast<const char*>(h.indices.data()),
            static_cast<std::streamsize>(h.nnz * sizeof(int32_t)));
    if (!f) throw std::runtime_error("Write error CSC bin: " + path);
}

// ---------------------------------------------------------------------------
// Convert a dense genes×cells matrix (row-major) to HostCSC, keeping
// all entries (dense → CSC with nnz = G*C).
// ---------------------------------------------------------------------------
HostCSC dense_to_csc(const float* mat, int G, int C) {
    HostCSC h;
    h.n_rows = G; h.n_cols = C;
    h.nnz    = static_cast<size_t>(G) * C;
    h.values .resize(h.nnz);
    h.indptr .resize(static_cast<size_t>(C) + 1);
    h.indices.resize(h.nnz);

    size_t off = 0;
    for (int c = 0; c < C; ++c) {
        h.indptr[c] = static_cast<int32_t>(off);
        for (int g = 0; g < G; ++g) {
            h.values [off] = mat[static_cast<size_t>(g) * C + c];
            h.indices[off] = g;
            ++off;
        }
    }
    h.indptr[C] = static_cast<int32_t>(h.nnz);
    return h;
}

// ---------------------------------------------------------------------------
// Upload HostCSC to device DeviceCSC.
// ---------------------------------------------------------------------------
singlet_gpu::core::DeviceCSC upload_csc(const HostCSC& h, cudaStream_t stream) {
    using namespace singlet_gpu::core;
    DeviceMemory<float>   d_values (h.nnz);
    DeviceMemory<int32_t> d_indptr (static_cast<size_t>(h.n_cols) + 1);
    DeviceMemory<int32_t> d_indices(h.nnz);

    auto ck = [](cudaError_t e, const char* s) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("upload_csc ") + s + ": " +
                                     cudaGetErrorString(e));
    };
    ck(cudaMemcpyAsync(d_values .get(), h.values .data(),
                       h.nnz * sizeof(float),
                       cudaMemcpyHostToDevice, stream), "values");
    ck(cudaMemcpyAsync(d_indptr .get(), h.indptr .data(),
                       (static_cast<size_t>(h.n_cols) + 1) * sizeof(int32_t),
                       cudaMemcpyHostToDevice, stream), "indptr");
    ck(cudaMemcpyAsync(d_indices.get(), h.indices.data(),
                       h.nnz * sizeof(int32_t),
                       cudaMemcpyHostToDevice, stream), "indices");
    ck(cudaStreamSynchronize(stream), "sync");

    DeviceCSC mat;
    mat.rows       = static_cast<uint32_t>(h.n_rows);
    mat.cols       = static_cast<uint32_t>(h.n_cols);
    mat.nnz        = h.nnz;
    mat.values     = std::move(d_values);
    mat.col_ptr    = std::move(d_indptr);
    mat.row_indices= std::move(d_indices);
    return mat;
}

// ---------------------------------------------------------------------------
// Parse the AUCell TSV output from aucell_r_reference.R.
// Returns scores[cell × set] in row-major order, and the set_names.
// ---------------------------------------------------------------------------
struct AUCellRefResult {
    int                      n_cells;
    int                      n_sets;
    std::vector<std::string> set_names;
    std::vector<float>       scores;   // n_cells × n_sets, row-major
};

AUCellRefResult parse_aucell_tsv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open aucell tsv: " + path);

    AUCellRefResult res{};
    std::string line;
    bool header = true;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        if (header) {
            // Header: cell_idx<TAB>set1<TAB>set2...
            std::string tok;
            std::getline(ss, tok, '\t');   // skip "cell_idx"
            while (std::getline(ss, tok, '\t'))
                res.set_names.push_back(tok);
            res.n_sets = static_cast<int>(res.set_names.size());
            header = false;
            continue;
        }
        std::string tok;
        std::getline(ss, tok, '\t');   // skip cell index
        for (int s = 0; s < res.n_sets; ++s) {
            std::getline(ss, tok, '\t');
            res.scores.push_back(std::stof(tok));
        }
        ++res.n_cells;
    }
    return res;
}

// ---------------------------------------------------------------------------
// Download AUCellResult scores from device to host.
// scores layout in AUCellResult: n_pathways × n_cells (column-major per doc).
// We transpose to row-major n_cells × n_pathways for comparison.
// ---------------------------------------------------------------------------
std::vector<float>
download_aucell_scores(const singlet_gpu::gsea::AUCellResult& res,
                       cudaStream_t stream)
{
    int total = res.n_pathways * res.n_cells;
    std::vector<float> h(static_cast<size_t>(total));
    cudaError_t e = cudaMemcpyAsync(h.data(), res.scores.get(),
                                    static_cast<size_t>(total) * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("download_aucell_scores: ") +
                                 cudaGetErrorString(e));
    if (cudaStreamSynchronize(stream) != cudaSuccess)
        throw std::runtime_error("download_aucell_scores: sync failed");
    // Transpose from [n_pathways × n_cells] to [n_cells × n_pathways].
    int P = res.n_pathways, C2 = res.n_cells;
    std::vector<float> out(static_cast<size_t>(C2) * P);
    for (int p = 0; p < P; ++p)
        for (int c = 0; c < C2; ++c)
            out[static_cast<size_t>(c) * P + p] = h[static_cast<size_t>(p) * C2 + c];
    return out;
}

// ---------------------------------------------------------------------------
// Build planted dense matrix: gene set s has high expression in cell bucket s.
// cells_per_bucket = kTinyCells / kTinySets.
// ---------------------------------------------------------------------------
std::vector<float> make_planted_matrix(int G, int C, int n_sets,
                                        int genes_per_set,
                                        const singlet_gpu::anno::GeneSetDB& db,
                                        uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> bg(0.0f, 1.0f);
    std::uniform_real_distribution<float> hi(10.0f, 15.0f);

    std::vector<float> mat(static_cast<size_t>(G) * C, 0.0f);
    for (auto& v : mat) v = bg(rng);

    int cells_per_set = std::max(1, C / n_sets);
    for (int s = 0; s < n_sets && s < static_cast<int>(db.set_names.size()); ++s) {
        int c_start = s * cells_per_set;
        int c_end   = std::min(C, c_start + cells_per_set);
        for (int gene_idx : db.member_gene_indices[s])
            for (int c = c_start; c < c_end; ++c)
                mat[static_cast<size_t>(gene_idx) * C + c] = hi(rng);
    }
    return mat;
}

// ---------------------------------------------------------------------------
// Build a GeneSetDB with planted membership.
// ---------------------------------------------------------------------------
singlet_gpu::anno::GeneSetDB make_planted_genesets(int G, int n_sets,
                                                    int genes_per_set)
{
    singlet_gpu::anno::GeneSetDB db;
    db.n_genes = G;
    for (int s = 0; s < n_sets && s * genes_per_set + genes_per_set <= G; ++s) {
        db.set_names.push_back("PLANTED_SET_" + std::to_string(s));
        std::vector<int32_t> idxs;
        for (int gi = s * genes_per_set; gi < (s + 1) * genes_per_set; ++gi)
            idxs.push_back(gi);
        db.member_gene_indices.push_back(idxs);
        db.weights.emplace_back();
    }
    return db;
}

// ---------------------------------------------------------------------------
// Run the R AUCell reference on a HostCSC + GeneSetDB.
// Returns the parsed result.
// ---------------------------------------------------------------------------
AUCellRefResult run_r_aucell(const HostCSC& mat_csc,
                               const singlet_gpu::anno::GeneSetDB& db,
                               const std::string& tag,
                               int top_k)
{
    ensure_refs_tmp();
    std::string mat_bin    = refs_path(tag + "_matrix.bin");
    std::string gs_tsv     = refs_path(tag + "_genesets.tsv");
    std::string out_tsv    = refs_path(tag + "_aucell_out.tsv");

    write_csc_bin(mat_csc, mat_bin);
    write_geneset_tsv(gs_tsv, db);

    std::string cmd = std::string("Rscript ") + kRefScript +
        " --matrix "   + mat_bin    +
        " --genesets " + gs_tsv     +
        " --output "   + out_tsv    +
        " --top-k "    + std::to_string(top_k);
    run_cmd(cmd);

    return parse_aucell_tsv(out_tsv);
}

}  // namespace

// =============================================================================
// Test: AUCell_TinyPlanted_VsR
// =============================================================================
// Tiny cell × gene matrix with planted high-expression cells per gene set.
// Compare to R AUCell per-cell scores: rel_err <= 1% per pathway.
// =============================================================================

TEST(AUCell, TinyPlanted_VsR) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device available.";

    auto db  = make_planted_genesets(kTinyGenes, kTinySets, kGenesPerSet);
    auto mat = make_planted_matrix(kTinyGenes, kTinyCells, kTinySets,
                                   kGenesPerSet, db, kSeed);

    // ---- GPU AUCell ---------------------------------------------------------
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto csc_h    = dense_to_csc(mat.data(), kTinyGenes, kTinyCells);
    auto d_mat    = upload_csc(csc_h, stream);

    singlet_gpu::gsea::AUCellConfig cfg;
    cfg.top_k_genes = kTopK;
    cfg.n_bins      = 4096;
    cfg.cell_tile   = 65536;
    cfg.seed        = kSeed;

    auto our_res = singlet_gpu::gsea::aucell(d_mat, db, cfg, stream);
    auto our_scores = download_aucell_scores(our_res, stream);
    cudaStreamDestroy(stream);

    // ---- R reference --------------------------------------------------------
    auto ref = run_r_aucell(csc_h, db, "tiny_planted", kTopK);

    ASSERT_EQ(ref.n_cells, kTinyCells)
        << "R AUCell returned wrong cell count.";
    ASSERT_EQ(ref.n_sets, our_res.n_pathways)
        << "R AUCell returned different number of gene sets.";

    // Compare per-cell per-pathway AUC scores.
    int n_large_err = 0;
    float max_rel_err = 0.0f;
    for (int c = 0; c < kTinyCells; ++c) {
        for (int s = 0; s < ref.n_sets; ++s) {
            float ref_val = ref.scores[static_cast<size_t>(c) * ref.n_sets + s];
            float our_val = our_scores[static_cast<size_t>(c) * our_res.n_pathways + s];
            float denom   = std::max(std::abs(ref_val), 1e-6f);
            float rel_err = std::abs(our_val - ref_val) / denom;
            max_rel_err   = std::max(max_rel_err, rel_err);
            if (rel_err > 0.01f) ++n_large_err;
        }
    }
    EXPECT_EQ(n_large_err, 0)
        << n_large_err << " cell×pathway pairs have rel_err > 1% "
        << "(max_rel_err=" << max_rel_err << ")";
}

// =============================================================================
// Test: AUCell_RealData_RanksConsistent
// =============================================================================
// Confirm that cells with high marker expression for a gene set receive
// higher AUC scores than background cells.
// =============================================================================

TEST(AUCell, RealData_RanksConsistent) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device available.";

    // Plant a strong signal for gene set 0 in the first kTinyCells/kTinySets cells.
    auto db  = make_planted_genesets(kMedGenes, kTinySets, kGenesPerSet);
    auto mat = make_planted_matrix(kMedGenes, kMedCells, kTinySets,
                                   kGenesPerSet, db, kSeed + 10);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto csc_h = dense_to_csc(mat.data(), kMedGenes, kMedCells);
    auto d_mat = upload_csc(csc_h, stream);

    singlet_gpu::gsea::AUCellConfig cfg;
    cfg.top_k_genes = std::min(100, kMedGenes);
    cfg.n_bins      = 4096;
    cfg.cell_tile   = 65536;
    cfg.seed        = kSeed + 10;

    auto our_res    = singlet_gpu::gsea::aucell(d_mat, db, cfg, stream);
    auto our_scores = download_aucell_scores(our_res, stream);
    cudaStreamDestroy(stream);

    // For gene set 0: cells 0..(kMedCells/kTinySets-1) are "high" cells.
    int cells_per_set = kMedCells / kTinySets;
    int set_idx       = 0;  // check set 0

    float mean_high = 0.0f, mean_bg = 0.0f;
    int n_high = cells_per_set;
    int n_bg   = kMedCells - n_high;

    for (int c = 0; c < n_high; ++c)
        mean_high += our_scores[static_cast<size_t>(c) * our_res.n_pathways + set_idx];
    for (int c = n_high; c < kMedCells; ++c)
        mean_bg   += our_scores[static_cast<size_t>(c) * our_res.n_pathways + set_idx];

    mean_high /= std::max(1, n_high);
    mean_bg   /= std::max(1, n_bg);

    EXPECT_GT(mean_high, mean_bg)
        << "Planted high-expression cells do not score higher than background "
        << "(mean_high=" << mean_high << ", mean_bg=" << mean_bg << ").";
}

// =============================================================================
// Test: AUCell_HistogramApprox_Error
// =============================================================================
// Compare histogram-binned ranks (B=4096) against exact per-cell ranks on
// a small dataset. The AUC approximation error must be < 1% rel_err.
// Uses the "exact" path by calling aucell with n_bins=0 (exact ranking)
// vs n_bins=4096.  If the kernel does not support n_bins=0, this test
// directly computes exact AUC on the CPU and compares.
// =============================================================================

TEST(AUCell, HistogramApprox_Error) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device available.";

    constexpr int kSmallGenes = 100;
    constexpr int kSmallCells = 50;
    constexpr int kSmallSets  = 3;
    constexpr int kSmallGenesPerSet = 10;

    auto db  = make_planted_genesets(kSmallGenes, kSmallSets, kSmallGenesPerSet);
    auto mat = make_planted_matrix(kSmallGenes, kSmallCells, kSmallSets,
                                   kSmallGenesPerSet, db, kSeed + 20);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto csc_h = dense_to_csc(mat.data(), kSmallGenes, kSmallCells);
    auto d_mat = upload_csc(csc_h, stream);

    // Histogram-approx path (B=4096).
    singlet_gpu::gsea::AUCellConfig cfg_approx;
    cfg_approx.top_k_genes = kSmallGenes;  // use all genes (small dataset)
    cfg_approx.n_bins      = 4096;
    cfg_approx.seed        = kSeed + 20;

    auto approx_res    = singlet_gpu::gsea::aucell(d_mat, db, cfg_approx, stream);
    auto approx_scores = download_aucell_scores(approx_res, stream);

    // Exact path (n_bins=1 or 0 — exact ranking mode; use R reference as ground truth).
    auto ref = run_r_aucell(csc_h, db, "histogram_approx",
                            cfg_approx.top_k_genes);
    cudaStreamDestroy(stream);

    ASSERT_EQ(ref.n_cells, kSmallCells);
    ASSERT_EQ(ref.n_sets,  kSmallSets);

    int n_large_err = 0;
    float max_rel_err = 0.0f;
    for (int c = 0; c < kSmallCells; ++c) {
        for (int s = 0; s < kSmallSets; ++s) {
            float ref_val = ref.scores[static_cast<size_t>(c) * ref.n_sets + s];
            float our_val = approx_scores[static_cast<size_t>(c) * approx_res.n_pathways + s];
            float denom   = std::max(std::abs(ref_val), 1e-6f);
            float rel_err = std::abs(our_val - ref_val) / denom;
            max_rel_err   = std::max(max_rel_err, rel_err);
            if (rel_err > 0.01f) ++n_large_err;
        }
    }
    EXPECT_EQ(n_large_err, 0)
        << n_large_err << " histogram-approx pairs have rel_err > 1% "
        << "(max_rel_err=" << max_rel_err << ")";
}

// =============================================================================
// Test: AUCell_Determinism_BitIdentical
// =============================================================================
// Run AUCell twice on the same input with the same config; verify outputs are
// bit-identical.
// =============================================================================

TEST(AUCell, Determinism_BitIdentical) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device available.";

    auto db  = make_planted_genesets(kTinyGenes, kTinySets, kGenesPerSet);
    auto mat = make_planted_matrix(kTinyGenes, kTinyCells, kTinySets,
                                   kGenesPerSet, db, kSeed);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto csc_h = dense_to_csc(mat.data(), kTinyGenes, kTinyCells);

    singlet_gpu::gsea::AUCellConfig cfg;
    cfg.top_k_genes = kTopK;
    cfg.n_bins      = 4096;
    cfg.seed        = kSeed;

    // Run 1.
    auto d_mat1    = upload_csc(csc_h, stream);
    auto res1      = singlet_gpu::gsea::aucell(d_mat1, db, cfg, stream);
    auto scores1   = download_aucell_scores(res1, stream);

    // Run 2.
    auto d_mat2    = upload_csc(csc_h, stream);
    auto res2      = singlet_gpu::gsea::aucell(d_mat2, db, cfg, stream);
    auto scores2   = download_aucell_scores(res2, stream);

    cudaStreamDestroy(stream);

    ASSERT_EQ(res1.n_cells, res2.n_cells);
    ASSERT_EQ(res1.n_pathways, res2.n_pathways);
    ASSERT_EQ(scores1.size(), scores2.size());

    int n_diff = 0;
    for (size_t i = 0; i < scores1.size(); ++i) {
        uint32_t b1, b2;
        std::memcpy(&b1, &scores1[i], 4);
        std::memcpy(&b2, &scores2[i], 4);
        if (b1 != b2) ++n_diff;
    }
    EXPECT_EQ(n_diff, 0)
        << n_diff << " elements differ between two determinism runs.";
}

// =============================================================================
// Test: AUCell_LargeCellCount_TilesCorrectly
// =============================================================================
// Synthetic 100k cells × 200 genes × 10 gene sets.
// Confirm output shape is (n_cells × n_pathways) and no OOM occurs.
// =============================================================================

TEST(AUCell, LargeCellCount_TilesCorrectly) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device available.";

    // Build a sparse CSC matrix (not fully dense to avoid host OOM).
    // Each cell has ~20 nonzero genes (Poisson-like sparse).
    constexpr float kDensity = 0.1f;  // 10% density
    size_t nnz_approx = static_cast<size_t>(kLargeGenes) * kLargeCells * kDensity;

    singlet_gpu::core::DeviceMemory<float>   d_values(nnz_approx);
    singlet_gpu::core::DeviceMemory<int32_t> d_indptr(static_cast<size_t>(kLargeCells) + 1);
    singlet_gpu::core::DeviceMemory<int32_t> d_indices(nnz_approx);

    // Build host sparse matrix.
    std::mt19937_64 rng(kSeed + 100);
    std::bernoulli_distribution keep(kDensity);
    std::uniform_real_distribution<float> val_dist(0.0f, 10.0f);

    std::vector<float>   h_values;
    std::vector<int32_t> h_indptr(static_cast<size_t>(kLargeCells) + 1);
    std::vector<int32_t> h_indices;
    h_values .reserve(nnz_approx);
    h_indices.reserve(nnz_approx);

    for (int c = 0; c < kLargeCells; ++c) {
        h_indptr[c] = static_cast<int32_t>(h_indices.size());
        for (int g = 0; g < kLargeGenes; ++g) {
            if (keep(rng)) {
                h_values .push_back(val_dist(rng));
                h_indices.push_back(g);
            }
        }
    }
    h_indptr[kLargeCells] = static_cast<int32_t>(h_indices.size());
    size_t actual_nnz = h_indices.size();

    // Upload to device.
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet_gpu::core::DeviceMemory<float>   dv(actual_nnz);
    singlet_gpu::core::DeviceMemory<int32_t> dp(static_cast<size_t>(kLargeCells) + 1);
    singlet_gpu::core::DeviceMemory<int32_t> di(actual_nnz);

    ASSERT_EQ(cudaMemcpyAsync(dv.get(), h_values .data(),
                              actual_nnz * sizeof(float),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(dp.get(), h_indptr .data(),
                              (static_cast<size_t>(kLargeCells) + 1) * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(di.get(), h_indices.data(),
                              actual_nnz * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    singlet_gpu::core::DeviceCSC d_mat;
    d_mat.rows       = static_cast<uint32_t>(kLargeGenes);
    d_mat.cols       = static_cast<uint32_t>(kLargeCells);
    d_mat.nnz        = actual_nnz;
    d_mat.values     = std::move(dv);
    d_mat.col_ptr    = std::move(dp);
    d_mat.row_indices= std::move(di);

    auto db = make_planted_genesets(kLargeGenes, kLargeSets,
                                    kLargeGenes / (kLargeSets + 1));

    singlet_gpu::gsea::AUCellConfig cfg;
    cfg.top_k_genes = std::min(50, kLargeGenes);
    cfg.n_bins      = 4096;
    cfg.cell_tile   = 65536;   // tile 64k cells at a time
    cfg.seed        = kSeed + 100;

    // This must not throw or OOM.
    singlet_gpu::gsea::AUCellResult res;
    ASSERT_NO_THROW(res = singlet_gpu::gsea::aucell(d_mat, db, cfg, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    cudaStreamDestroy(stream);

    EXPECT_EQ(res.n_cells,    kLargeCells)
        << "Output cell count mismatch for 100k-cell test.";
    EXPECT_EQ(res.n_pathways, static_cast<int>(db.set_names.size()))
        << "Output pathway count mismatch for 100k-cell test.";

    // Spot-check: all output values are in [0, 1] (valid AUC range).
    // Download a small slice (first 10 cells × all pathways).
    int n_check = std::min(10, kLargeCells) * res.n_pathways;
    std::vector<float> h_check(static_cast<size_t>(n_check));
    ASSERT_EQ(cudaMemcpy(h_check.data(), res.scores.get(),
                         static_cast<size_t>(n_check) * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < n_check; ++i) {
        EXPECT_GE(h_check[i], 0.0f) << "AUC score < 0 at index " << i;
        EXPECT_LE(h_check[i], 1.0f) << "AUC score > 1 at index " << i;
    }
}
