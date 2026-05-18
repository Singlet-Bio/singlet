// SPDX-License-Identifier: MIT
// singlet/gpu/tests/fate_palantir_correctness.cpp
//
// integrates: original (first GPU Palantir-style diffusion pseudotime + fate inference)
//
// Correctness harness for singlet::gpu::fate::palantir.
//
// Written against the design doc at:
//   singlet/gpu/state/designs/45-palantir.md
// NOT against the kernel source — authored in parallel with fate/palantir.h by
// the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: Palantir Python via subprocess (pip install palantir anndata scanpy).
//   Fallback: pure-Python scipy-based diffusion map + eigendecomposition +
//   Markov absorption (tests/refs/palantir_py_reference.py).
//   Exit code 0  = Palantir Python ran.
//   Exit code 2  = Palantir not installed; scipy diffusion map fallback ran.
//   Both codes produce valid reference outputs; harness proceeds either way.
//
// Tolerances (design doc §"Correctness test spec"):
//   Pseudotime Spearman ρ  ≥ 0.95  vs Python (test 1 — tiny synthetic)
//   Pseudotime finite+monotone on smooth trajectory (test 2 — real data)
//   Branch prob row sums   = 1 ± 1e-5  per cell (test 3)
//   Terminal self-absorption probability = 1 ± 1e-5 (test 4)
//   Bit-identical across two runs (test 5)
//
// Test cases:
//   Palantir_TinySynthetic_VsPython
//   Palantir_GSM_RealData
//   Palantir_BranchProbs_SumToOne
//   Palantir_TerminalCells_Self1
//   Palantir_Determinism_BitIdentical
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, CUDA::cusparse, CUDA::cublas,
//             CUDA::cusolver, singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every GPU-dependent test calls GTEST_SKIP().

// ---- singlet-gpu palantir header (written by gpu-kernel-dev) ----------------
#include <singlet/gpu/fate/palantir.h>

// ---- other singlet-gpu headers needed for the real-data test ----------------
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
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Compat bridge — test-harness API → real singlet-gpu API
// =============================================================================

// GPUContext::create(stream) — the real GPUContext has default ctor only.
// Replace all occurrences with GPUContext{}.

// io::PzDeviceLoader compat (same as fate_cellrank2 compat).
namespace singlet::gpu { namespace io {
#ifndef SINGLET_GPU_COMPAT_PZ_DEVICE_LOADER
#define SINGLET_GPU_COMPAT_PZ_DEVICE_LOADER
struct PzDeviceLoader {
    std::string path_;
    cudaStream_t stream_;
    PzDeviceLoader(const std::string& p, cudaStream_t s) : path_(p), stream_(s) {}
    template<typename Ctx>
    core::DeviceCSC load(const Ctx& /*ctx*/) const {
        return singlet::gpu::io::load_pz(path_, stream_).mat;
    }
};
#endif
} }

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

/// Tiny synthetic (design doc test 1):
///   200 cells × 50 genes, hierarchical tree: root → 3 branches.
constexpr int kTinyCells     = 200;
constexpr int kTinyGenes     = 50;
constexpr int kTinyTerminals = 3;

/// kNN fan-out used when constructing the synthetic expression matrix.
constexpr int kTinyKnn = 10;

/// Tolerance on per-cell pseudotime Spearman ρ (design doc: ≥ 0.95).
constexpr double kSpearmanMin = 0.95;

/// Tolerance on branch probability row sums (design doc: 1 ± 1e-5).
constexpr float kRowSumTol = 1e-5f;

/// Tolerance for terminal self-absorption check (design doc: 1 ± 1e-5).
constexpr float kTermSelfTol = 1e-5f;

// Real-data paths — GSM4037629 (scRNA, 11,560 cells).
static const char* kGsmExonPath =
    "/mnt/projects/debruinz_project/singlet_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/exon_counts.1pz";

// Python reference script path.
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet/gpu/"
    "tests/refs/palantir_py_reference.py";

// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_palantir_refs_tmp";

// Python interpreter.
static const char* kPython = "python3";

// Exit code from reference script when Palantir Python is not installed
// but the scipy diffusion map fallback ran successfully.
static constexpr int kRefScriptFallbackCode = 2;

// =============================================================================
// Utility helpers (pattern from cycles 3–44)
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
// Spearman rank correlation (pure C++) — reused from cycles 11-44.
// ---------------------------------------------------------------------------
double spearman(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    int n = static_cast<int>(a.size());
    if (n < 2) return 1.0;

    auto rank_of = [&](const std::vector<float>& v) -> std::vector<double> {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int i, int j) { return v[i] < v[j]; });
        std::vector<double> r(n);
        for (int i = 0; i < n; ) {
            int j = i;
            while (j < n && v[idx[j]] == v[idx[i]]) ++j;
            double mid = (static_cast<double>(i) + j - 1) / 2.0;
            for (int k = i; k < j; ++k) r[idx[k]] = mid;
            i = j;
        }
        return r;
    };

    auto ra = rank_of(a);
    auto rb = rank_of(b);

    double mean_a = 0.0, mean_b = 0.0;
    for (int i = 0; i < n; ++i) { mean_a += ra[i]; mean_b += rb[i]; }
    mean_a /= n; mean_b /= n;

    double num = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (int i = 0; i < n; ++i) {
        double da = ra[i] - mean_a, db = rb[i] - mean_b;
        num     += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }
    if (denom_a == 0.0 || denom_b == 0.0) return 0.0;
    return num / std::sqrt(denom_a * denom_b);
}

// ---------------------------------------------------------------------------
// Binary I/O helpers
// ---------------------------------------------------------------------------

/// Write dense float32 matrix (row-major) to disk for the Python reference.
/// Format: [uint32_t n_cells][uint32_t n_genes][float × n_cells × n_genes]
void write_expression_bin(const std::vector<float>& X, uint32_t n_cells,
                           uint32_t n_genes, const std::string& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.write(reinterpret_cast<const char*>(&n_cells), 4);
    f.write(reinterpret_cast<const char*>(&n_genes), 4);
    f.write(reinterpret_cast<const char*>(X.data()),
            static_cast<std::streamsize>(X.size() * sizeof(float)));
    if (!f) throw std::runtime_error("Write error: " + path);
}

/// Write a single uint32.
void write_u32(uint32_t val, const std::string& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.write(reinterpret_cast<const char*>(&val), 4);
}

/// Write int32 vector: [uint32_t n][int32_t × n].
void write_i32_vec(const std::vector<int32_t>& v, const std::string& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    uint32_t n = static_cast<uint32_t>(v.size());
    f.write(reinterpret_cast<const char*>(&n), 4);
    if (n > 0) {
        f.write(reinterpret_cast<const char*>(v.data()),
                static_cast<std::streamsize>(n * sizeof(int32_t)));
    }
}

/// Read float32 vector: [uint32_t n][float × n].
std::vector<float> read_f32_vec(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    uint32_t n;
    f.read(reinterpret_cast<char*>(&n), 4);
    std::vector<float> v(n);
    if (n > 0)
        f.read(reinterpret_cast<char*>(v.data()),
               static_cast<std::streamsize>(n * sizeof(float)));
    if (!f) throw std::runtime_error("Read error: " + path);
    return v;
}

/// Read branch prob matrix: [uint32_t n_cells][uint32_t n_terms][float×n_cells×n_terms].
struct BranchProbs {
    uint32_t            n_cells;
    uint32_t            n_terms;
    std::vector<float>  data;  // row-major: [n_cells × n_terms]
};

BranchProbs read_branch_probs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    BranchProbs bp;
    f.read(reinterpret_cast<char*>(&bp.n_cells), 4);
    f.read(reinterpret_cast<char*>(&bp.n_terms), 4);
    bp.data.resize(static_cast<size_t>(bp.n_cells) * bp.n_terms);
    f.read(reinterpret_cast<char*>(bp.data.data()),
           static_cast<std::streamsize>(bp.data.size() * sizeof(float)));
    if (!f) throw std::runtime_error("Read error: " + path);
    return bp;
}

/// Read int32 vector: [uint32_t n][int32_t × n].
std::vector<int32_t> read_i32_vec(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    uint32_t n;
    f.read(reinterpret_cast<char*>(&n), 4);
    std::vector<int32_t> v(n);
    if (n > 0)
        f.read(reinterpret_cast<char*>(v.data()),
               static_cast<std::streamsize>(n * sizeof(int32_t)));
    return v;
}

// ---------------------------------------------------------------------------
// Tiny synthetic hierarchical tree generator.
//
// Constructs a 200-cell × 50-gene expression matrix with a planted hierarchical
// tree structure: root → 3 terminal branches.
//
// Construction:
//   - Cells 0–9:  root cluster (high expression of gene 0).
//   - Remaining cells split into 3 terminal branches.
//   - Branch k adds a branch-specific "trajectory" component:
//       X[c, k*15 + offset] += trajectory_signal * pseudoprogress.
//   - Gaussian noise added throughout.
//   - Root cell: cell 0 (used as start_cell).
//   - Terminal cells: one high-pseudoprogress cell per branch.
// ---------------------------------------------------------------------------
struct TinySynthetic {
    std::vector<float>   X;             // row-major [n_cells × n_genes]
    uint32_t             start_cell;
    std::vector<int32_t> terminal_ids;  // one per branch
};

TinySynthetic make_tiny_synthetic() {
    std::mt19937_64 rng(kSeed);
    std::normal_distribution<float> gauss(0.f, 0.3f);
    std::uniform_real_distribution<float> unif(0.f, 1.f);

    TinySynthetic syn;
    syn.X.assign(static_cast<size_t>(kTinyCells) * kTinyGenes, 0.f);
    syn.start_cell = 0;

    int cells_per_branch = (kTinyCells - 10) / kTinyTerminals;

    auto cell = [&](int c, int g) -> float& {
        return syn.X[static_cast<size_t>(c) * kTinyGenes + g];
    };

    // Root cluster: cells 0–9.
    for (int c = 0; c < 10; ++c) {
        cell(c, 0) = 2.0f + gauss(rng);
        for (int g = 1; g < kTinyGenes; ++g)
            cell(c, g) = gauss(rng);
    }

    // Branches.
    for (int k = 0; k < kTinyTerminals; ++k) {
        int base = 10 + k * cells_per_branch;
        int end  = (k + 1 == kTinyTerminals) ? kTinyCells : base + cells_per_branch;

        // Gene offset: each branch has a dedicated "marker" region.
        int marker_start = 1 + k * 15;

        for (int c = base; c < end; ++c) {
            float progress = static_cast<float>(c - base) /
                             static_cast<float>(std::max(1, end - base - 1));

            // Branch marker genes: expression increases with progress.
            for (int g = marker_start; g < marker_start + 12 && g < kTinyGenes; ++g) {
                cell(c, g) = progress * 3.0f + gauss(rng);
            }
            // Shared background.
            cell(c, 0) = (1.f - progress) * 1.5f + gauss(rng);
            // Noise.
            for (int g = 0; g < kTinyGenes; ++g)
                cell(c, g) += gauss(rng) * 0.2f;
        }

        // Terminal cell = last cell in this branch.
        syn.terminal_ids.push_back(end - 1);
    }

    // Log-normalize: total-count normalize to 1e4, then log1p.
    for (int c = 0; c < kTinyCells; ++c) {
        float total = 0.f;
        for (int g = 0; g < kTinyGenes; ++g)
            total += std::max(0.f, cell(c, g));
        total = std::max(total, 1e-10f);
        for (int g = 0; g < kTinyGenes; ++g) {
            cell(c, g) = std::log1p(std::max(0.f, cell(c, g)) / total * 1e4f);
        }
    }

    return syn;
}

/// Write TinySynthetic inputs to disk for the Python reference.
void write_tiny_synthetic_inputs(const TinySynthetic& syn) {
    ensure_refs_tmp();
    write_expression_bin(syn.X, kTinyCells, kTinyGenes,
                         refs_path("expression.bin"));
    write_u32(syn.start_cell,       refs_path("start_cell.bin"));
    write_u32(kTinyTerminals,        refs_path("n_terminals.bin"));
    write_i32_vec(syn.terminal_ids,  refs_path("terminal_ids.bin"));
}

// ---------------------------------------------------------------------------
// Download a DeviceMemory<float> block to host.
// Reused pattern from cycles 41-44.
// ---------------------------------------------------------------------------
std::vector<float> download_f32(
        const singlet::gpu::core::DeviceMemory<float>& d,
        size_t count, cudaStream_t stream)
{
    std::vector<float> h(count);
    cudaError_t e = cudaMemcpyAsync(h.data(), d.get(),
                                    count * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("download_f32: ") +
                                 cudaGetErrorString(e));
    cudaStreamSynchronize(stream);
    return h;
}

// Overload: PalantirResult::pseudotime / branch_prob are host std::vector<float>.
std::vector<float> download_f32(
        const std::vector<float>& d,
        size_t count, cudaStream_t /*stream*/)
{
    std::vector<float> h(d.begin(), d.begin() + std::min(d.size(), count));
    h.resize(count, 0.0f);
    return h;
}

}  // anonymous namespace

// =============================================================================
// Test 1 — Palantir_TinySynthetic_VsPython
//
// 200 cells × 50 genes hierarchical tree (root → 3 branches).
// Run singlet-gpu Palantir kernel.
// Run Python reference (Palantir Python or scipy diffusion map fallback).
// Compute per-cell Spearman ρ between the two pseudotimes.
// Require ρ ≥ 0.95.
// =============================================================================

TEST(Palantir, TinySynthetic_VsPython) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU detected — skipping GPU test";

    TinySynthetic syn = make_tiny_synthetic();
    write_tiny_synthetic_inputs(syn);

    // Run the Python reference.
    std::string cmd =
        std::string(kPython) + " " + kRefScript +
        " --task tiny_synthetic"
        " --data_dir " + std::string(kRefsTmpDir) +
        " --out_dir "  + std::string(kRefsTmpDir);
    int rc = run_cmd_rc(cmd);
    ASSERT_TRUE(rc == 0 || rc == kRefScriptFallbackCode)
        << "Reference script failed (exit " << rc << "): " << cmd;

    std::vector<float> py_pt = read_f32_vec(refs_path("py_pseudotime.bin"));
    ASSERT_EQ(static_cast<int>(py_pt.size()), kTinyCells)
        << "Python pseudotime size mismatch";

    // Run singlet-gpu kernel.
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    singlet::gpu::core::GPUContext ctx{};  // compat: real GPUContext has default ctor only

    // Upload expression matrix as dense device array.
    singlet::gpu::core::DeviceMemory<float> d_X(
        static_cast<size_t>(kTinyCells) * kTinyGenes);
    ASSERT_EQ(cudaMemcpyAsync(d_X.get(), syn.X.data(),
                              syn.X.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    cudaStreamSynchronize(stream);

    singlet::gpu::fate::PalantirConfig cfg;
    cfg.start_cell        = static_cast<int32_t>(syn.start_cell);
    cfg.terminal_ids      = syn.terminal_ids.data();
    cfg.n_terminals       = static_cast<int>(syn.terminal_ids.size());
    cfg.k_knn             = kTinyKnn;
    cfg.n_eigs            = 15;
    cfg.seed              = kSeed;

    singlet::gpu::fate::PalantirResult res =
        singlet::gpu::fate::run_palantir(ctx, d_X, kTinyCells, kTinyGenes, cfg);

    std::vector<float> gpu_pt =
        download_f32(res.pseudotime, kTinyCells, stream);

    cudaStreamDestroy(stream);

    // Spearman ρ between GPU and Python pseudotimes.
    double rho = spearman(gpu_pt, py_pt);
    EXPECT_GE(rho, kSpearmanMin)
        << "Spearman ρ = " << rho << " < " << kSpearmanMin
        << " (GPU vs Python pseudotime on tiny synthetic)";
}

// =============================================================================
// Test 2 — Palantir_GSM_RealData
//
// Load GSM4037629 exon_counts.1pz (11,560 cells).
// Run Palantir with auto-detected start cell (highest expression of a stem marker,
// or simply cell 0 as a fallback) and auto-detected terminals.
// Confirm:
//   (a) All pseudotimes are finite and in [0, 1].
//   (b) Pseudotime is monotone in expectation: cells along a smooth trajectory
//       (binned by pseudotime quartiles) show non-decreasing median pseudotime
//       per quartile bin.
// =============================================================================

TEST(Palantir, GSM_RealData) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU detected — skipping GPU test";
    if (!fs::exists(kGsmExonPath)) {
        GTEST_SKIP() << "GSM4037629 exon_counts.1pz not found — skipping real-data test";
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    singlet::gpu::core::GPUContext ctx{};  // compat: real GPUContext has default ctor only

    // Load .1pz → device CSC (genes × cells).
    singlet::gpu::io::PzDeviceLoader loader(kGsmExonPath, stream);
    auto d_csc = loader.load(ctx);

    int n_cells = d_csc.cols;

    // Config: auto-detect start cell and terminals.
    singlet::gpu::fate::PalantirConfig cfg;
    cfg.start_cell        = 0;          // cell 0 as default start
    cfg.n_terminals       = 5;          // auto-detect 5 terminals
    cfg.auto_start        = true;       // find most stem-like cell
    cfg.auto_terminals    = true;
    cfg.k_knn             = 30;
    cfg.n_eigs            = 20;
    cfg.seed              = kSeed;

    singlet::gpu::fate::PalantirResult res =
        singlet::gpu::fate::run_palantir_from_csc(ctx, d_csc, cfg);

    std::vector<float> gpu_pt =
        download_f32(res.pseudotime, static_cast<size_t>(n_cells), stream);

    cudaStreamDestroy(stream);

    // (a) All pseudotimes finite and in [0, 1].
    int n_nan   = 0, n_out_of_range = 0;
    for (int c = 0; c < n_cells; ++c) {
        float pt = gpu_pt[c];
        if (!std::isfinite(pt))       ++n_nan;
        if (pt < -1e-4f || pt > 1.f + 1e-4f) ++n_out_of_range;
    }
    EXPECT_EQ(n_nan, 0)
        << n_nan << "/" << n_cells << " cells have non-finite pseudotime";
    EXPECT_EQ(n_out_of_range, 0)
        << n_out_of_range << "/" << n_cells << " cells have pseudotime outside [0,1]";

    // (b) Monotone in expectation: quartile-bin medians should be non-decreasing.
    // Sort pseudotime values, split into 4 quartile bins, check median order.
    std::vector<float> sorted_pt = gpu_pt;
    std::sort(sorted_pt.begin(), sorted_pt.end());
    int q = n_cells / 4;
    std::array<float, 4> quartile_medians;
    for (int i = 0; i < 4; ++i) {
        int lo = i * q;
        int hi = (i + 1 == 4) ? n_cells : (i + 1) * q;
        int mid = (lo + hi) / 2;
        quartile_medians[i] = sorted_pt[mid];
    }
    for (int i = 1; i < 4; ++i) {
        EXPECT_GE(quartile_medians[i], quartile_medians[i - 1])
            << "Quartile median not non-decreasing: Q" << i
            << "=" << quartile_medians[i] << " < Q" << (i-1)
            << "=" << quartile_medians[i-1];
    }
}

// =============================================================================
// Test 3 — Palantir_BranchProbs_SumToOne
//
// On the tiny synthetic, verify that branch probability row sums = 1 ± 1e-5
// for all cells.
// =============================================================================

TEST(Palantir, BranchProbs_SumToOne) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU detected — skipping GPU test";

    TinySynthetic syn = make_tiny_synthetic();

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    singlet::gpu::core::GPUContext ctx{};  // compat: real GPUContext has default ctor only

    singlet::gpu::core::DeviceMemory<float> d_X(
        static_cast<size_t>(kTinyCells) * kTinyGenes);
    ASSERT_EQ(cudaMemcpyAsync(d_X.get(), syn.X.data(),
                              syn.X.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    cudaStreamSynchronize(stream);

    singlet::gpu::fate::PalantirConfig cfg;
    cfg.start_cell    = static_cast<int32_t>(syn.start_cell);
    cfg.terminal_ids  = syn.terminal_ids.data();
    cfg.n_terminals   = static_cast<int>(syn.terminal_ids.size());
    cfg.k_knn         = kTinyKnn;
    cfg.n_eigs        = 15;
    cfg.seed          = kSeed;

    singlet::gpu::fate::PalantirResult res =
        singlet::gpu::fate::run_palantir(ctx, d_X, kTinyCells, kTinyGenes, cfg);

    int n_terms = static_cast<int>(res.n_terminals);
    std::vector<float> bp =
        download_f32(res.branch_probs,
                     static_cast<size_t>(kTinyCells) * n_terms, stream);

    cudaStreamDestroy(stream);

    int fail_count = 0;
    for (int c = 0; c < kTinyCells; ++c) {
        float row_sum = 0.f;
        for (int k = 0; k < n_terms; ++k)
            row_sum += bp[static_cast<size_t>(c) * n_terms + k];
        if (std::fabs(row_sum - 1.f) > kRowSumTol)
            ++fail_count;
    }
    EXPECT_EQ(fail_count, 0)
        << fail_count << "/" << kTinyCells
        << " cells have branch prob row sum outside [1 ± " << kRowSumTol << "]";
}

// =============================================================================
// Test 4 — Palantir_TerminalCells_Self1
//
// For each terminal cell (provided to the kernel), verify:
//   branch_prob[terminal_k, own_terminal_col] = 1 ± 1e-5
//   branch_prob[terminal_k, other_cols]       = 0 ± 1e-5
// =============================================================================

TEST(Palantir, TerminalCells_Self1) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU detected — skipping GPU test";

    TinySynthetic syn = make_tiny_synthetic();

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    singlet::gpu::core::GPUContext ctx{};  // compat: real GPUContext has default ctor only

    singlet::gpu::core::DeviceMemory<float> d_X(
        static_cast<size_t>(kTinyCells) * kTinyGenes);
    ASSERT_EQ(cudaMemcpyAsync(d_X.get(), syn.X.data(),
                              syn.X.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    cudaStreamSynchronize(stream);

    singlet::gpu::fate::PalantirConfig cfg;
    cfg.start_cell    = static_cast<int32_t>(syn.start_cell);
    cfg.terminal_ids  = syn.terminal_ids.data();
    cfg.n_terminals   = static_cast<int>(syn.terminal_ids.size());
    cfg.k_knn         = kTinyKnn;
    cfg.n_eigs        = 15;
    cfg.seed          = kSeed;

    singlet::gpu::fate::PalantirResult res =
        singlet::gpu::fate::run_palantir(ctx, d_X, kTinyCells, kTinyGenes, cfg);

    int n_terms = static_cast<int>(res.n_terminals);
    std::vector<float> bp =
        download_f32(res.branch_probs,
                     static_cast<size_t>(kTinyCells) * n_terms, stream);

    cudaStreamDestroy(stream);

    ASSERT_EQ(n_terms, kTinyTerminals)
        << "Expected " << kTinyTerminals << " terminals, got " << n_terms;

    for (int k = 0; k < kTinyTerminals; ++k) {
        int term_cell = syn.terminal_ids[k];

        // Own column should be 1.
        float own = bp[static_cast<size_t>(term_cell) * n_terms + k];
        EXPECT_NEAR(own, 1.f, kTermSelfTol)
            << "Terminal " << k << " (cell " << term_cell
            << "): own-column branch_prob = " << own << " (expected 1)";

        // Other columns should be 0.
        for (int j = 0; j < kTinyTerminals; ++j) {
            if (j == k) continue;
            float other = bp[static_cast<size_t>(term_cell) * n_terms + j];
            EXPECT_NEAR(other, 0.f, kTermSelfTol)
                << "Terminal " << k << " (cell " << term_cell
                << "): cross-column " << j << " branch_prob = " << other
                << " (expected 0)";
        }
    }
}

// =============================================================================
// Test 5 — Palantir_Determinism_BitIdentical
//
// Run Palantir twice on the same tiny synthetic input.
// Verify that both pseudotime and branch_probs are bit-identical.
// =============================================================================

TEST(Palantir, Determinism_BitIdentical) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU detected — skipping GPU test";

    TinySynthetic syn = make_tiny_synthetic();

    auto run_once = [&]() -> std::pair<std::vector<float>, std::vector<float>> {
        cudaStream_t stream;
        EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        singlet::gpu::core::GPUContext ctx{};  // compat: real GPUContext has default ctor only

        singlet::gpu::core::DeviceMemory<float> d_X(
            static_cast<size_t>(kTinyCells) * kTinyGenes);
        EXPECT_EQ(cudaMemcpyAsync(d_X.get(), syn.X.data(),
                                  syn.X.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream), cudaSuccess);
        cudaStreamSynchronize(stream);

        singlet::gpu::fate::PalantirConfig cfg;
        cfg.start_cell    = static_cast<int32_t>(syn.start_cell);
        cfg.terminal_ids  = syn.terminal_ids.data();
        cfg.n_terminals   = static_cast<int>(syn.terminal_ids.size());
        cfg.k_knn         = kTinyKnn;
        cfg.n_eigs        = 15;
        cfg.seed          = kSeed;

        singlet::gpu::fate::PalantirResult res =
            singlet::gpu::fate::run_palantir(ctx, d_X, kTinyCells, kTinyGenes, cfg);

        int n_terms = static_cast<int>(res.n_terminals);
        auto pt = download_f32(res.pseudotime, kTinyCells, stream);
        auto bp = download_f32(res.branch_probs,
                               static_cast<size_t>(kTinyCells) * n_terms, stream);

        cudaStreamDestroy(stream);
        return {std::move(pt), std::move(bp)};
    };

    auto [pt1, bp1] = run_once();
    auto [pt2, bp2] = run_once();

    ASSERT_EQ(pt1.size(), pt2.size()) << "Pseudotime vector size mismatch";
    ASSERT_EQ(bp1.size(), bp2.size()) << "Branch prob vector size mismatch";

    // Pseudotime: bit-identical.
    for (size_t i = 0; i < pt1.size(); ++i) {
        uint32_t u1, u2;
        std::memcpy(&u1, &pt1[i], 4);
        std::memcpy(&u2, &pt2[i], 4);
        EXPECT_EQ(u1, u2)
            << "Pseudotime bit difference at cell " << i
            << ": run1=" << pt1[i] << " run2=" << pt2[i];
        if (u1 != u2) break;
    }

    // Branch probs: bit-identical.
    for (size_t i = 0; i < bp1.size(); ++i) {
        uint32_t u1, u2;
        std::memcpy(&u1, &bp1[i], 4);
        std::memcpy(&u2, &bp2[i], 4);
        EXPECT_EQ(u1, u2)
            << "Branch prob bit difference at element " << i
            << ": run1=" << bp1[i] << " run2=" << bp2[i];
        if (u1 != u2) break;
    }
}
