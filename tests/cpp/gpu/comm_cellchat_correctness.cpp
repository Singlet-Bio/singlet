// SPDX-License-Identifier: MIT
// singlet/gpu/tests/comm_cellchat_correctness.cpp
//
// integrates: original (first GPU CellChat-style cell-cell communication)
//
// Correctness harness for singlet::gpu::comm::CellChat.
//
// Written against the design doc at:
//   singlet/gpu/state/designs/37-cellchat-gpu.md
// NOT against the kernel source — authored in parallel with
// comm/cellchat.h by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: CellChat R via Rscript subprocess (Jin et al. Nat Comm 2021)
//   tests/refs/cellchat_r_reference.R
//   Environment: R >= 4.0 + install.packages("CellChat")
//   Exit code 2 from the R script → GTEST_SKIP (CellChat not installed).
//   Exit code != 0 and != 2 → GTEST_FAIL (script error).
//
//   Pure-R fallback: when CellChat is absent, the reference script computes
//   pseudobulk mean expression + Hill-function comm_prob from scratch so that
//   tests comparing against R always have a reference path.
//
// Tolerances (design doc §"Correctness test spec"):
//   comm_prob Spearman ρ vs R        >= 0.90  (test 1, 200 cells × 100 genes × 3 types × 10 LR)
//   Real data: finite results, edges > 0      (test 2)
//   p-value KS statistic under null  <= 0.10  (test 3, K=100)
//   pathway Spearman ρ vs R          >= 0.95  (test 4)
//   Determinism: bit-identical (fixed seed)   (test 5)
//
// Test cases:
//   CellChat_TinySynthetic_VsR
//   CellChat_GSM_RealData
//   CellChat_PermutationPvalues_Calibrated
//   CellChat_PathwayAggregation_VsR
//   CellChat_Determinism_BitIdentical
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, CUDA::cusparse, CUDA::cublas,
//             CUDA::curand, singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every test that requires a device calls GTEST_SKIP().

// ---- singlet-gpu CellChat header (written by gpu-kernel-dev) ----------------
#include <singlet/gpu/comm/cellchat.h>

// ---- other singlet-gpu headers needed by the real-data test -----------------
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/core/memory.h>

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
#include <map>
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

/// Tiny synthetic dimensions (design doc §"Correctness test spec", case 1).
constexpr int kTinyCells  = 200;
constexpr int kTinyGenes  = 100;
constexpr int kTinyTypes  =   3;
constexpr int kTinyLR     =  10;   // LR pairs used in the tiny test (from mini DB)

/// Permutation count for calibration test (design doc: K=100 for tests).
constexpr int kPermCount  = 100;

/// KS tolerance for null p-value uniformity (test 3).
constexpr double kKsTol   = 0.10;

/// Spearman ρ tolerance for comm_prob vs R (test 1).
constexpr double kCommRhoTol = 0.90;

/// Spearman ρ tolerance for pathway aggregation vs R (test 4).
constexpr double kPathRhoTol = 0.95;

/// Hill function K parameter (CellChat default: 0.5).
constexpr float kHillK = 0.5f;

/// Path to the CellChat R reference script.
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet/gpu/"
    "tests/refs/cellchat_r_reference.R";

/// Path to the mini LR database fixture.
static const char* kMiniDbPath =
    "/mnt/home/debruinz/Singlet-AI/singlet/gpu/"
    "tests/refs/cellchatdb_mini.tsv";

/// Real scRNA .1pz path (skip gracefully if absent).
static const char* kGsmExprPath =
    "/mnt/projects/debruinz_project/singlet_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/exon_counts.1pz";

/// Real cell-type annotation TSV (skip gracefully if absent).
/// Expected format: one integer (0-based type index) per line.
static const char* kGsmCellTypesPath =
    "/mnt/projects/debruinz_project/singlet_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/cell_type_labels.tsv";

/// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_comm_cellchat_refs_tmp";

// =============================================================================
// Utility helpers (pattern from cycles 3–36)
// =============================================================================

namespace {

void ensure_refs_tmp() { fs::create_directories(kRefsTmpDir); }

std::string refs_path(const std::string& name) {
    return std::string(kRefsTmpDir) + "/" + name;
}

/// Run a shell command; throw on non-zero exit.
void run_cmd(const std::string& cmd) {
    int ret = std::system(cmd.c_str());
    if (ret != 0)
        throw std::runtime_error(
            "Command failed (exit " + std::to_string(ret) + "): " + cmd);
}

/// Run a shell command; return the exit code without throwing.
int run_cmd_rc(const std::string& cmd) {
    return std::system(cmd.c_str());
}

bool gpu_available() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    return (e == cudaSuccess && count > 0);
}

// ---------------------------------------------------------------------------
// Write a plain numeric matrix to TSV (no header, no row names).
// Row-major layout.
// ---------------------------------------------------------------------------
void write_matrix_tsv(const std::string& path,
                      const std::vector<float>& data,
                      int n_rows, int n_cols)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open for write: " + path);
    for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
            if (c > 0) f << '\t';
            f << data[static_cast<size_t>(r) * n_cols + c];
        }
        f << '\n';
    }
    if (!f) throw std::runtime_error("Write error: " + path);
}

// ---------------------------------------------------------------------------
// Write a vector of ints (one per line).
// ---------------------------------------------------------------------------
void write_int_column(const std::string& path, const std::vector<int>& vec) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open for write: " + path);
    for (int v : vec) f << v << '\n';
    if (!f) throw std::runtime_error("Write error: " + path);
}

// ---------------------------------------------------------------------------
// Read a plain numeric TSV (no header) into a flat vector (row-major).
// Sets *n_rows and *n_cols.
// ---------------------------------------------------------------------------
std::vector<double> read_matrix_tsv(const std::string& path,
                                     int* n_rows, int* n_cols)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open tsv: " + path);

    std::vector<double> vals;
    *n_rows = 0; *n_cols = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        double v; int col = 0;
        while (ss >> v) { vals.push_back(v); ++col; }
        if (*n_cols == 0) *n_cols = col;
        ++(*n_rows);
    }
    return vals;
}

// ---------------------------------------------------------------------------
// Spearman rank correlation between two equal-length vectors.
// Inline implementation (pattern from cycles 7, 11, 13, 36).
// ---------------------------------------------------------------------------
double spearman(const std::vector<double>& a, const std::vector<double>& b)
{
    assert(a.size() == b.size());
    int n = static_cast<int>(a.size());
    if (n < 2) return 1.0;

    auto rank_of = [&](const std::vector<double>& v) -> std::vector<double> {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int i, int j){ return v[i] < v[j]; });
        std::vector<double> r(n);
        for (int i = 0; i < n; ) {
            int j = i;
            while (j < n && v[idx[j]] == v[idx[i]]) ++j;
            double avg_rank = (i + j - 1) / 2.0 + 1.0;
            for (int k = i; k < j; ++k) r[idx[k]] = avg_rank;
            i = j;
        }
        return r;
    };

    auto ra = rank_of(a);
    auto rb = rank_of(b);
    double sum_d2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = ra[i] - rb[i];
        sum_d2 += d * d;
    }
    return 1.0 - 6.0 * sum_d2 / (static_cast<double>(n) * (n * n - 1));
}

// ---------------------------------------------------------------------------
// Kolmogorov-Smirnov statistic for testing uniformity of p in [0, 1].
// ---------------------------------------------------------------------------
double ks_uniform(std::vector<double> pvals) {
    std::sort(pvals.begin(), pvals.end());
    int n = static_cast<int>(pvals.size());
    double D = 0.0;
    for (int i = 0; i < n; ++i) {
        double expected = (i + 1.0) / n;
        D = std::max(D, std::abs(pvals[i] - expected));
    }
    return D;
}

// ---------------------------------------------------------------------------
// LR database row: pathway, ligand gene name, receptor gene name.
// ---------------------------------------------------------------------------
struct LRPair {
    std::string pathway;
    std::string ligand;
    std::string receptor;
};

// ---------------------------------------------------------------------------
// Parse the mini LR database TSV.
// Format: pathway <TAB> ligand <TAB> receptor  (no header).
// ---------------------------------------------------------------------------
std::vector<LRPair> parse_lr_db(const std::string& path, int max_pairs = -1) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open LR DB: " + path);
    std::vector<LRPair> db;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        LRPair p;
        if (!(ss >> p.pathway >> p.ligand >> p.receptor)) continue;
        db.push_back(std::move(p));
        if (max_pairs > 0 && static_cast<int>(db.size()) >= max_pairs) break;
    }
    return db;
}

// ---------------------------------------------------------------------------
// Build a gene name → index map.  Genes are named "gene_0" ... "gene_{n-1}".
// ---------------------------------------------------------------------------
std::string gene_name(int idx) {
    return "gene_" + std::to_string(idx);
}

// ---------------------------------------------------------------------------
// Generate a tiny synthetic log-normalized expression matrix.
// Shape: n_genes × n_cells (matching CellChat R convention: rows = genes).
// Returns flat row-major float vector [n_genes × n_cells].
// Values drawn from Log-Normal to mimic scRNA-seq.
// ---------------------------------------------------------------------------
std::vector<float> make_synthetic_expr(int n_genes, int n_cells,
                                       const std::vector<int>& cell_types,
                                       int n_types,
                                       const std::vector<LRPair>& lr_db,
                                       uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::lognormal_distribution<float> lognorm(0.0f, 1.2f);
    std::bernoulli_distribution sparse(0.85);  // ~85% zeros

    std::vector<float> mat(static_cast<size_t>(n_genes) * n_cells, 0.0f);

    for (int g = 0; g < n_genes; ++g)
        for (int c = 0; c < n_cells; ++c)
            if (!sparse(rng))
                mat[static_cast<size_t>(g) * n_cells + c] = lognorm(rng);

    // Plant signal: for a handful of LR pairs, boost ligand expression in
    // cell-type 0 and receptor expression in cell-type 1, so there is a
    // non-trivial communication edge to compare against R.
    int n_plant = std::min(3, static_cast<int>(lr_db.size()));
    for (int lr = 0; lr < n_plant; ++lr) {
        // ligand gene index = lr (modulo n_genes)
        int lig_g = lr % n_genes;
        int rec_g = (lr + 1) % n_genes;
        for (int c = 0; c < n_cells; ++c) {
            if (cell_types[c] == 0)
                mat[static_cast<size_t>(lig_g) * n_cells + c] += 2.0f;
            if (cell_types[c] == 1)
                mat[static_cast<size_t>(rec_g) * n_cells + c] += 2.0f;
        }
    }

    return mat;
}

// ---------------------------------------------------------------------------
// Build a sub-selection of LR pairs remapped to synthetic gene indices.
// For the synthetic test, map ligand → "gene_{lr_idx % n_genes}" and
// receptor → "gene_{(lr_idx+1) % n_genes}".
// Returns the same number of rows as requested, as a fresh LRPair list
// whose ligand/receptor names match the synthetic gene_* scheme.
// ---------------------------------------------------------------------------
std::vector<LRPair> remap_lr_to_synthetic(const std::vector<LRPair>& db,
                                          int n_genes, int n_pairs)
{
    std::vector<LRPair> out;
    for (int i = 0; i < n_pairs; ++i) {
        LRPair p;
        p.pathway  = (i < static_cast<int>(db.size())) ? db[i].pathway
                                                        : ("PW" + std::to_string(i));
        p.ligand   = gene_name(i % n_genes);
        p.receptor = gene_name((i + 1) % n_genes);
        out.push_back(std::move(p));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Write LR DB as a TSV (pathway ligand receptor, no header).
// ---------------------------------------------------------------------------
void write_lr_db_tsv(const std::string& path, const std::vector<LRPair>& db) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open lr_db tsv: " + path);
    for (const auto& p : db)
        f << p.pathway << '\t' << p.ligand << '\t' << p.receptor << '\n';
    if (!f) throw std::runtime_error("Write error lr_db tsv: " + path);
}

// ---------------------------------------------------------------------------
// host_dense_to_device_csc: convert a genes×cells row-major float matrix
// (already dense) to a DeviceCSC on the GPU.
// ---------------------------------------------------------------------------
singlet::gpu::core::DeviceCSC host_dense_to_device_csc(
    const float* data, int n_genes, int n_cells, cudaStream_t stream)
{
    // Build CSC on host: column-major, one column per cell.
    // Every gene-row is stored per cell column.
    std::vector<int>   h_col_ptr(n_cells + 1);
    std::vector<int>   h_row_idx;
    std::vector<float> h_vals;

    for (int c = 0; c < n_cells; ++c) {
        h_col_ptr[c] = static_cast<int>(h_vals.size());
        for (int g = 0; g < n_genes; ++g) {
            float v = data[g * n_cells + c];
            if (v != 0.0f) {
                h_row_idx.push_back(g);
                h_vals.push_back(v);
            }
        }
    }
    h_col_ptr[n_cells] = static_cast<int>(h_vals.size());

    int nnz = static_cast<int>(h_vals.size());
    singlet::gpu::core::DeviceCSC d;
    d.rows = n_genes;
    d.cols = n_cells;
    d.nnz  = nnz;
    d.col_ptr    = singlet::gpu::core::DeviceMemory<int>(n_cells + 1);
    d.row_indices= singlet::gpu::core::DeviceMemory<int>(std::max(nnz, 1));
    d.values     = singlet::gpu::core::DeviceMemory<float>(std::max(nnz, 1));

    cudaMemcpyAsync(d.col_ptr.get(), h_col_ptr.data(),
                    (n_cells + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    if (nnz > 0) {
        cudaMemcpyAsync(d.row_indices.get(), h_row_idx.data(),
                        nnz * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d.values.get(), h_vals.data(),
                        nnz * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    cudaStreamSynchronize(stream);
    return d;
}

// ---------------------------------------------------------------------------
// lr_pairs_to_index_arrays: convert LRPair[] + gene_name_to_idx map into
// the flattened index arrays expected by cellchat_gpu().
// Gene names are "gene_0" .. "gene_{n_genes-1}".
// Returns: ligand_offsets, ligand_gene_idx, receptor_offsets,
//          receptor_gene_idx, pathway_id_per_lr, n_pathways.
// ---------------------------------------------------------------------------
struct LRIndexArrays {
    std::vector<int> ligand_offsets;
    std::vector<int> ligand_gene_idx;
    std::vector<int> receptor_offsets;
    std::vector<int> receptor_gene_idx;
    std::vector<int> pathway_id_per_lr;
    int              n_pathways;
};

LRIndexArrays lr_pairs_to_index_arrays(
    const std::vector<LRPair>& db,
    const std::map<std::string,int>& gene_to_idx)
{
    // Build pathway name → id map (in order of first occurrence).
    std::map<std::string,int> pw_to_id;
    std::vector<int> pathway_id_per_lr;
    for (const auto& p : db) {
        if (!pw_to_id.count(p.pathway)) {
            int id = static_cast<int>(pw_to_id.size());
            pw_to_id[p.pathway] = id;
        }
        pathway_id_per_lr.push_back(pw_to_id.at(p.pathway));
    }
    int n_pathways = static_cast<int>(pw_to_id.size());
    int n_lr = static_cast<int>(db.size());

    // For this test, each LR pair has exactly one ligand and one receptor gene.
    std::vector<int> lig_offsets(n_lr + 1), rec_offsets(n_lr + 1);
    std::vector<int> lig_idx, rec_idx;
    for (int i = 0; i < n_lr; ++i) {
        lig_offsets[i] = static_cast<int>(lig_idx.size());
        auto it = gene_to_idx.find(db[i].ligand);
        if (it != gene_to_idx.end()) lig_idx.push_back(it->second);
        rec_offsets[i] = static_cast<int>(rec_idx.size());
        it = gene_to_idx.find(db[i].receptor);
        if (it != gene_to_idx.end()) rec_idx.push_back(it->second);
    }
    lig_offsets[n_lr] = static_cast<int>(lig_idx.size());
    rec_offsets[n_lr] = static_cast<int>(rec_idx.size());

    LRIndexArrays out;
    out.ligand_offsets   = std::move(lig_offsets);
    out.ligand_gene_idx  = std::move(lig_idx);
    out.receptor_offsets = std::move(rec_offsets);
    out.receptor_gene_idx= std::move(rec_idx);
    out.pathway_id_per_lr= std::move(pathway_id_per_lr);
    out.n_pathways       = n_pathways;
    return out;
}

// Convenience: build gene_to_idx for synthetic "gene_0"..."gene_{n-1}" scheme.
std::map<std::string,int> synthetic_gene_map(int n_genes) {
    std::map<std::string,int> m;
    for (int i = 0; i < n_genes; ++i)
        m["gene_" + std::to_string(i)] = i;
    return m;
}

// ---------------------------------------------------------------------------
// compute_cellchat: bridge adapter that converts LRPair[] to index arrays
// and calls the actual cellchat_gpu() API.
// ---------------------------------------------------------------------------
singlet::gpu::comm::CellChatResult compute_cellchat(
    const singlet::gpu::core::DeviceCSC& expr,
    const int*   cell_types_ptr,
    int          n_cells,
    int          n_types,
    const LRPair* lr_db_ptr,
    int          n_lr,
    const singlet::gpu::comm::CellChatConfig& cfg,
    singlet::gpu::core::GPUContext&  /*ctx*/,
    cudaStream_t                     /*stream*/)
{
    std::vector<int> cell_type(cell_types_ptr, cell_types_ptr + n_cells);
    std::vector<LRPair> db(lr_db_ptr, lr_db_ptr + n_lr);

    // Build gene name map from expr shape (n_genes rows)
    auto gene_map = synthetic_gene_map(expr.rows);
    auto idx = lr_pairs_to_index_arrays(db, gene_map);

    return singlet::gpu::comm::cellchat_gpu(
        expr, cell_type, n_types,
        idx.ligand_offsets, idx.ligand_gene_idx,
        idx.receptor_offsets, idx.receptor_gene_idx,
        idx.pathway_id_per_lr, idx.n_pathways,
        cfg);
}

// ---------------------------------------------------------------------------
// Build a CellChatConfig from the design doc defaults.
// ---------------------------------------------------------------------------
singlet::gpu::comm::CellChatConfig make_config(uint64_t seed    = kSeed,
                                              int      n_perm  = kPermCount,
                                              float    hill_k  = kHillK,
                                              bool     deterministic = false)
{
    singlet::gpu::comm::CellChatConfig cfg;
    cfg.seed          = seed;
    cfg.n_permutations = n_perm;
    cfg.hill_K        = hill_k;
    cfg.deterministic = deterministic;
    return cfg;
}

}  // anonymous namespace

// =============================================================================
// Test fixture: device context + stream, created once per test.
// =============================================================================

class CellChatTest : public ::testing::Test {
protected:
    // GPUContext is non-copyable; obtain the process-wide singleton reference.
    singlet::gpu::core::GPUContext* ctx_ptr_ = nullptr;
    cudaStream_t                   stream_{};

    void SetUp() override {
        if (!gpu_available()) GTEST_SKIP() << "No CUDA device available.";
        ctx_ptr_ = &singlet::gpu::core::default_context();
        stream_  = ctx_ptr_->stream;
        ensure_refs_tmp();
    }

    void TearDown() override {
        // GPUContext lifecycle is managed by default_context() singleton; no cleanup.
        stream_  = {};
        ctx_ptr_ = nullptr;
    }
};

// =============================================================================
// Test 1 — CellChat_TinySynthetic_VsR
//
// 200 cells × 100 genes × 3 cell types × 10 L-R pairs synthetic.
// Spearman ρ ≥ 0.90 on comm_prob (flattened n_types*n_types*n_lr) vs R.
// =============================================================================

TEST_F(CellChatTest, CellChat_TinySynthetic_VsR) {
    // --- 1. Parse mini LR DB and remap to synthetic gene names ----------------
    auto db_raw = parse_lr_db(kMiniDbPath, kTinyLR);
    ASSERT_GE(static_cast<int>(db_raw.size()), kTinyLR)
        << "Mini LR DB has fewer than " << kTinyLR << " pairs.";
    auto lr_db = remap_lr_to_synthetic(db_raw, kTinyGenes, kTinyLR);

    // --- 2. Build synthetic cell types (balanced across 3 types) --------------
    std::vector<int> cell_types(kTinyCells);
    for (int c = 0; c < kTinyCells; ++c)
        cell_types[c] = c % kTinyTypes;

    // --- 3. Build synthetic expression (genes × cells) ------------------------
    auto expr_flat = make_synthetic_expr(kTinyGenes, kTinyCells,
                                         cell_types, kTinyTypes,
                                         lr_db, kSeed);

    // --- 4. Write inputs for R reference --------------------------------------
    const std::string expr_tsv    = refs_path("cc_tiny_expr.tsv");
    const std::string ctypes_tsv  = refs_path("cc_tiny_cell_types.tsv");
    const std::string lr_tsv      = refs_path("cc_tiny_lr_db.tsv");
    const std::string r_prob_tsv  = refs_path("cc_tiny_r_prob.tsv");
    const std::string r_pval_tsv  = refs_path("cc_tiny_r_pval.tsv");
    const std::string r_path_tsv  = refs_path("cc_tiny_r_pathway.tsv");

    write_matrix_tsv(expr_tsv, expr_flat, kTinyGenes, kTinyCells);
    write_int_column(ctypes_tsv, cell_types);
    write_lr_db_tsv(lr_tsv, lr_db);

    // --- 5. Run R reference ---------------------------------------------------
    std::string r_cmd =
        std::string("Rscript ") + kRefScript +
        " --expr "          + expr_tsv    +
        " --cell_types "    + ctypes_tsv  +
        " --lr_db "         + lr_tsv      +
        " --n_cells "       + std::to_string(kTinyCells)  +
        " --n_genes "       + std::to_string(kTinyGenes)  +
        " --n_types "       + std::to_string(kTinyTypes)  +
        " --n_lr "          + std::to_string(kTinyLR)     +
        " --K "             + std::to_string(kPermCount)  +
        " --seed 42"                                        +
        " --output_prob "   + r_prob_tsv  +
        " --output_pval "   + r_pval_tsv  +
        " --output_pathway "+ r_path_tsv  +
        " 2>&1";

    int rc = run_cmd_rc(r_cmd);
    if (rc == 2) GTEST_SKIP() << "CellChat R not installed; skipping reference diff.";
    ASSERT_EQ(rc, 0) << "R reference script failed (exit " << rc << ")";

    // --- 6. Read R reference comm_prob ----------------------------------------
    int r_nrows = 0, r_ncols = 0;
    auto r_prob = read_matrix_tsv(r_prob_tsv, &r_nrows, &r_ncols);
    ASSERT_EQ(r_nrows, kTinyTypes * kTinyTypes) << "R prob row count mismatch";
    ASSERT_EQ(r_ncols, kTinyLR)                << "R prob col count mismatch";

    // --- 7. Upload expression to device and run singlet-gpu CellChat ----------
    // Expression shape: genes × cells CSC (matches pz_device_loader convention)
    using DevCSC = singlet::gpu::core::DeviceCSC;
    DevCSC expr_csc = host_dense_to_device_csc(
        expr_flat.data(), kTinyGenes, kTinyCells, stream_);

    auto cfg = make_config(kSeed, kPermCount);
    singlet::gpu::comm::CellChatResult result =
        compute_cellchat(
            expr_csc,
            cell_types.data(),
            kTinyCells,
            kTinyTypes,
            lr_db.data(),
            static_cast<int>(lr_db.size()),
            cfg,
            *ctx_ptr_,
            stream_);

    // --- 8. Copy GPU comm_prob to host ----------------------------------------
    int n_out = kTinyTypes * kTinyTypes * kTinyLR;
    std::vector<float> gpu_prob(n_out);
    cudaError_t ce = cudaMemcpyAsync(
        gpu_prob.data(), result.comm_prob.get(),
        static_cast<size_t>(n_out) * sizeof(float),
        cudaMemcpyDeviceToHost, stream_);
    ASSERT_EQ(ce, cudaSuccess) << cudaGetErrorString(ce);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    // --- 9. Flatten and compute Spearman ρ ------------------------------------
    // GPU layout: [s * n_types + r, lr] row-major — matches R reference layout.
    std::vector<double> gpu_prob_d(n_out), r_prob_d(n_out);
    for (int i = 0; i < n_out; ++i) {
        gpu_prob_d[i] = static_cast<double>(gpu_prob[i]);
        r_prob_d[i]   = r_prob[i];
    }

    double rho = spearman(gpu_prob_d, r_prob_d);
    EXPECT_GE(rho, kCommRhoTol)
        << "CellChat_TinySynthetic_VsR: comm_prob Spearman ρ = " << rho
        << " < threshold " << kCommRhoTol;

    // Cleanup device result
    // result is RAII; cleanup is automatic on scope exit.
}

// =============================================================================
// Test 2 — CellChat_GSM_RealData
//
// Load real scRNA (GSM4037629) with cell-type annotations.
// Skip if either file is absent.
// Confirm: all comm_prob values are finite, at least one non-zero edge exists.
// =============================================================================

TEST_F(CellChatTest, CellChat_GSM_RealData) {
    if (!fs::exists(kGsmExprPath)) {
        GTEST_SKIP() << "Real scRNA .1pz not found: " << kGsmExprPath;
    }
    if (!fs::exists(kGsmCellTypesPath)) {
        GTEST_SKIP() << "Cell-type annotations not found: " << kGsmCellTypesPath;
    }

    // --- 1. Load expression via device loader ----------------------------------
    using DevCSC = singlet::gpu::core::DeviceCSC;
    DevCSC expr_csc = singlet::gpu::io::load_pz(kGsmExprPath, stream_).mat;

    int n_cells = static_cast<int>(expr_csc.cols);
    int n_genes = static_cast<int>(expr_csc.rows);

    // --- 2. Load cell-type labels ---------------------------------------------
    std::vector<int> cell_types;
    {
        std::ifstream f(kGsmCellTypesPath);
        ASSERT_TRUE(f.good()) << "Cannot open cell types: " << kGsmCellTypesPath;
        int v;
        while (f >> v) cell_types.push_back(v);
    }
    ASSERT_EQ(static_cast<int>(cell_types.size()), n_cells)
        << "Cell-type label count != n_cells";

    int n_types = *std::max_element(cell_types.begin(), cell_types.end()) + 1;
    ASSERT_GT(n_types, 1) << "Need at least 2 cell types for communication.";

    // --- 3. Parse mini LR DB --------------------------------------------------
    auto lr_db = parse_lr_db(kMiniDbPath);
    ASSERT_FALSE(lr_db.empty()) << "Mini LR DB is empty.";

    // --- 4. Run singlet-gpu CellChat ------------------------------------------
    auto cfg = make_config(kSeed, /*K=*/50);
    singlet::gpu::comm::CellChatResult result =
        compute_cellchat(
            expr_csc,
            cell_types.data(),
            n_cells,
            n_types,
            lr_db.data(),
            static_cast<int>(lr_db.size()),
            cfg,
            *ctx_ptr_,
            stream_);

    // --- 5. Copy comm_prob to host and validate --------------------------------
    int n_out = n_types * n_types * static_cast<int>(lr_db.size());
    std::vector<float> prob(n_out);
    cudaError_t ce = cudaMemcpyAsync(
        prob.data(), result.comm_prob.get(),
        static_cast<size_t>(n_out) * sizeof(float),
        cudaMemcpyDeviceToHost, stream_);
    ASSERT_EQ(ce, cudaSuccess) << cudaGetErrorString(ce);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    // All values must be finite.
    for (int i = 0; i < n_out; ++i)
        EXPECT_TRUE(std::isfinite(prob[i])) << "Non-finite comm_prob at index " << i;

    // At least one non-zero communication edge.
    bool any_nonzero = std::any_of(prob.begin(), prob.end(),
                                   [](float v){ return v > 0.0f; });
    EXPECT_TRUE(any_nonzero) << "All communication probabilities are zero.";

    // result is RAII; cleanup is automatic on scope exit.
}

// =============================================================================
// Test 3 — CellChat_PermutationPvalues_Calibrated
//
// Under the null (random cell-type labels with balanced assignment):
// p-values should be uniform over [0, 1].
// KS statistic ≤ 0.10, K=100 permutations.
// =============================================================================

TEST_F(CellChatTest, CellChat_PermutationPvalues_Calibrated) {
    // --- 1. Build null expression: random labels, no planted signal -----------
    constexpr int kNullCells  = 200;
    constexpr int kNullGenes  =  50;
    constexpr int kNullTypes  =   4;
    constexpr int kNullLR     =   5;

    std::mt19937_64 rng(kSeed ^ 0xBEEFull);
    std::lognormal_distribution<float> lognorm(0.0f, 1.0f);
    std::bernoulli_distribution sparse(0.80);

    // Random expression: genes × cells
    std::vector<float> expr_flat(static_cast<size_t>(kNullGenes) * kNullCells);
    for (auto& v : expr_flat)
        v = sparse(rng) ? 0.0f : lognorm(rng);

    // Balanced cell types
    std::vector<int> cell_types(kNullCells);
    for (int c = 0; c < kNullCells; ++c)
        cell_types[c] = c % kNullTypes;

    // Simple LR pairs pointing to existing gene indices
    std::vector<LRPair> lr_db(kNullLR);
    for (int i = 0; i < kNullLR; ++i) {
        lr_db[i].pathway  = "PW" + std::to_string(i);
        lr_db[i].ligand   = gene_name(i % kNullGenes);
        lr_db[i].receptor = gene_name((i + 2) % kNullGenes);
    }

    // --- 2. Upload to device --------------------------------------------------
    using DevCSC = singlet::gpu::core::DeviceCSC;
    DevCSC expr_csc = host_dense_to_device_csc(
        expr_flat.data(), kNullGenes, kNullCells, stream_);

    auto cfg = make_config(kSeed, kPermCount);
    singlet::gpu::comm::CellChatResult result =
        compute_cellchat(
            expr_csc,
            cell_types.data(),
            kNullCells,
            kNullTypes,
            lr_db.data(),
            kNullLR,
            cfg,
            *ctx_ptr_,
            stream_);

    // --- 3. Copy p-values to host ---------------------------------------------
    int n_out = kNullTypes * kNullTypes * kNullLR;
    std::vector<float> pvals(n_out);
    cudaError_t ce = cudaMemcpyAsync(
        pvals.data(), result.p_values.get(),
        static_cast<size_t>(n_out) * sizeof(float),
        cudaMemcpyDeviceToHost, stream_);
    ASSERT_EQ(ce, cudaSuccess) << cudaGetErrorString(ce);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    // --- 4. KS test for uniformity -------------------------------------------
    // Exclude diagonal (self-communication) and collect off-diagonal p-values.
    std::vector<double> pvals_offdiag;
    for (int s = 0; s < kNullTypes; ++s) {
        for (int r = 0; r < kNullTypes; ++r) {
            if (s == r) continue;
            for (int lr = 0; lr < kNullLR; ++lr) {
                int idx = (s * kNullTypes + r) * kNullLR + lr;
                float p = pvals[idx];
                if (std::isfinite(p))
                    pvals_offdiag.push_back(static_cast<double>(p));
            }
        }
    }

    ASSERT_FALSE(pvals_offdiag.empty()) << "No off-diagonal p-values collected.";
    double ks = ks_uniform(pvals_offdiag);
    EXPECT_LE(ks, kKsTol)
        << "CellChat_PermutationPvalues_Calibrated: KS stat = " << ks
        << " > threshold " << kKsTol
        << " (p-values not uniform under null; n=" << pvals_offdiag.size() << ")";

    // result is RAII; cleanup is automatic on scope exit.
}

// =============================================================================
// Test 4 — CellChat_PathwayAggregation_VsR
//
// Pathway-level aggregation (sum over LR pairs per pathway):
// Spearman ρ ≥ 0.95 vs R reference.
// Uses the same tiny synthetic data as Test 1.
// =============================================================================

TEST_F(CellChatTest, CellChat_PathwayAggregation_VsR) {
    // --- 1. Set up same synthetic data as Test 1 ------------------------------
    auto db_raw = parse_lr_db(kMiniDbPath, kTinyLR);
    ASSERT_GE(static_cast<int>(db_raw.size()), kTinyLR)
        << "Mini LR DB has fewer than " << kTinyLR << " pairs.";
    auto lr_db = remap_lr_to_synthetic(db_raw, kTinyGenes, kTinyLR);

    std::vector<int> cell_types(kTinyCells);
    for (int c = 0; c < kTinyCells; ++c)
        cell_types[c] = c % kTinyTypes;

    auto expr_flat = make_synthetic_expr(kTinyGenes, kTinyCells,
                                         cell_types, kTinyTypes,
                                         lr_db, kSeed);

    // --- 2. Write inputs and run R reference ----------------------------------
    const std::string expr_tsv    = refs_path("cc_path_expr.tsv");
    const std::string ctypes_tsv  = refs_path("cc_path_cell_types.tsv");
    const std::string lr_tsv      = refs_path("cc_path_lr_db.tsv");
    const std::string r_prob_tsv  = refs_path("cc_path_r_prob.tsv");
    const std::string r_pval_tsv  = refs_path("cc_path_r_pval.tsv");
    const std::string r_path_tsv  = refs_path("cc_path_r_pathway.tsv");

    write_matrix_tsv(expr_tsv, expr_flat, kTinyGenes, kTinyCells);
    write_int_column(ctypes_tsv, cell_types);
    write_lr_db_tsv(lr_tsv, lr_db);

    std::string r_cmd =
        std::string("Rscript ") + kRefScript +
        " --expr "          + expr_tsv   +
        " --cell_types "    + ctypes_tsv +
        " --lr_db "         + lr_tsv     +
        " --n_cells "       + std::to_string(kTinyCells)  +
        " --n_genes "       + std::to_string(kTinyGenes)  +
        " --n_types "       + std::to_string(kTinyTypes)  +
        " --n_lr "          + std::to_string(kTinyLR)     +
        " --K "             + std::to_string(kPermCount)  +
        " --seed 42"                                        +
        " --output_prob "   + r_prob_tsv +
        " --output_pval "   + r_pval_tsv +
        " --output_pathway "+ r_path_tsv +
        " 2>&1";

    int rc = run_cmd_rc(r_cmd);
    if (rc == 2) GTEST_SKIP() << "CellChat R not installed; skipping pathway diff.";
    ASSERT_EQ(rc, 0) << "R reference script failed (exit " << rc << ")";

    // Determine number of unique pathways from the remapped LR DB.
    std::vector<std::string> pathways_ordered;
    {
        std::set<std::string> seen;
        for (const auto& p : lr_db) {
            if (!seen.count(p.pathway)) {
                pathways_ordered.push_back(p.pathway);
                seen.insert(p.pathway);
            }
        }
    }
    int n_pathways = static_cast<int>(pathways_ordered.size());

    // --- 3. Read R pathway activity -------------------------------------------
    int r_nrows = 0, r_ncols = 0;
    auto r_path = read_matrix_tsv(r_path_tsv, &r_nrows, &r_ncols);
    ASSERT_EQ(r_nrows, kTinyTypes * kTinyTypes) << "R pathway row count mismatch";
    ASSERT_EQ(r_ncols, n_pathways)               << "R pathway col count mismatch";

    // --- 4. Run singlet-gpu CellChat ------------------------------------------
    using DevCSC = singlet::gpu::core::DeviceCSC;
    DevCSC expr_csc = host_dense_to_device_csc(
        expr_flat.data(), kTinyGenes, kTinyCells, stream_);

    auto cfg = make_config(kSeed, kPermCount);
    singlet::gpu::comm::CellChatResult result =
        compute_cellchat(
            expr_csc,
            cell_types.data(),
            kTinyCells,
            kTinyTypes,
            lr_db.data(),
            static_cast<int>(lr_db.size()),
            cfg,
            *ctx_ptr_,
            stream_);

    // --- 5. Copy pathway_activity to host -------------------------------------
    int n_out = kTinyTypes * kTinyTypes * n_pathways;
    std::vector<float> gpu_path(n_out);
    cudaError_t ce = cudaMemcpyAsync(
        gpu_path.data(), result.pathway_activity.get(),
        static_cast<size_t>(n_out) * sizeof(float),
        cudaMemcpyDeviceToHost, stream_);
    ASSERT_EQ(ce, cudaSuccess) << cudaGetErrorString(ce);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    // --- 6. Spearman ρ on flattened pathway activity -------------------------
    std::vector<double> gpu_path_d(n_out), r_path_d(n_out);
    for (int i = 0; i < n_out; ++i) {
        gpu_path_d[i] = static_cast<double>(gpu_path[i]);
        r_path_d[i]   = r_path[i];
    }

    double rho = spearman(gpu_path_d, r_path_d);
    EXPECT_GE(rho, kPathRhoTol)
        << "CellChat_PathwayAggregation_VsR: Spearman ρ = " << rho
        << " < threshold " << kPathRhoTol;

    // result is RAII; cleanup is automatic on scope exit.
}

// =============================================================================
// Test 5 — CellChat_Determinism_BitIdentical
//
// Run compute_cellchat twice with the same fixed seed and verify bit-identical
// comm_prob and p-value outputs.  Uses cfg.deterministic = true.
// =============================================================================

TEST_F(CellChatTest, CellChat_Determinism_BitIdentical) {
    // Build tiny synthetic inputs
    auto db_raw = parse_lr_db(kMiniDbPath, kTinyLR);
    ASSERT_GE(static_cast<int>(db_raw.size()), kTinyLR);
    auto lr_db = remap_lr_to_synthetic(db_raw, kTinyGenes, kTinyLR);

    std::vector<int> cell_types(kTinyCells);
    for (int c = 0; c < kTinyCells; ++c)
        cell_types[c] = c % kTinyTypes;

    auto expr_flat = make_synthetic_expr(kTinyGenes, kTinyCells,
                                         cell_types, kTinyTypes,
                                         lr_db, kSeed);

    using DevCSC = singlet::gpu::core::DeviceCSC;

    // Run 1
    DevCSC expr_csc_1 = host_dense_to_device_csc(
        expr_flat.data(), kTinyGenes, kTinyCells, stream_);
    auto cfg1 = make_config(kSeed, kPermCount, kHillK, /*deterministic=*/true);
    auto res1 = compute_cellchat(
        expr_csc_1, cell_types.data(), kTinyCells, kTinyTypes,
        lr_db.data(), static_cast<int>(lr_db.size()), cfg1, *ctx_ptr_, stream_);

    int n_out = kTinyTypes * kTinyTypes * kTinyLR;
    std::vector<float> prob1(n_out), pval1(n_out);
    cudaMemcpyAsync(prob1.data(), res1.comm_prob.get(),
                    static_cast<size_t>(n_out) * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(pval1.data(), res1.p_values.get(),
                    static_cast<size_t>(n_out) * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    // res1 is RAII; cleanup automatic.

    // Run 2 (same seed, same deterministic flag)
    DevCSC expr_csc_2 = host_dense_to_device_csc(
        expr_flat.data(), kTinyGenes, kTinyCells, stream_);
    auto cfg2 = make_config(kSeed, kPermCount, kHillK, /*deterministic=*/true);
    auto res2 = compute_cellchat(
        expr_csc_2, cell_types.data(), kTinyCells, kTinyTypes,
        lr_db.data(), static_cast<int>(lr_db.size()), cfg2, *ctx_ptr_, stream_);

    std::vector<float> prob2(n_out), pval2(n_out);
    cudaMemcpyAsync(prob2.data(), res2.comm_prob.get(),
                    static_cast<size_t>(n_out) * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(pval2.data(), res2.p_values.get(),
                    static_cast<size_t>(n_out) * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    // res2 is RAII; cleanup automatic.

    // Bit-identical check
    for (int i = 0; i < n_out; ++i) {
        EXPECT_EQ(prob1[i], prob2[i])
            << "comm_prob mismatch at index " << i
            << ": run1=" << prob1[i] << " run2=" << prob2[i];
        EXPECT_EQ(pval1[i], pval2[i])
            << "pval mismatch at index " << i
            << ": run1=" << pval1[i] << " run2=" << pval2[i];
    }
}
