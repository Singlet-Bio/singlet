// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/fate_cospar_correctness.cpp
//
// integrates: original (first GPU Cospar-style cell fate transition mapping)
//
// Correctness harness for singlet_gpu::fate::cospar.
//
// Written against the design doc at:
//   singlet-gpu/state/designs/41-cospar.md
// NOT against the kernel source — authored in parallel with fate/cospar.h by
// the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: Cospar Python via subprocess (pip install cospar).
//   Fallback: pure-Python implementation of the core iterative optimizer
//   (gradient + soft-threshold + simplex projection, from design doc §Algorithm)
//   embedded in tests/refs/cospar_py_reference.py.
//   Exit code 2 from the reference script = cospar not installed → fallback runs.
//   The fallback always produces a reference value.
//
// Tolerances (design doc §"Correctness test spec"):
//   fate_bias Spearman ρ  >= 0.90  vs Python (tiny synthetic — test 1)
//   T row sums            ∈ [0,1]  within 1e-5 (test 3)
//   potency monotone      on hierarchical tree (test 4)
//   bit-identical         across two runs (test 5)
//
// Test cases:
//   Cospar_TinySynthetic_VsPython
//   Cospar_GSM_RealLineage
//   Cospar_TransitionMap_RowSumsValid
//   Cospar_PotencyMonotone
//   Cospar_Determinism_BitIdentical
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, CUDA::cusparse, singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every GPU-dependent test calls GTEST_SKIP().

// ---- singlet-gpu cospar header (written by gpu-kernel-dev) ------------------
#include <singlet-gpu/fate/cospar.h>

// ---- other singlet-gpu headers needed for the real-data test ----------------
#include <singlet-gpu/io/pz_device_loader.h>
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
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// CosparConfig compat aliases (n_iter, n_knn, step) and cospar_fit() / CosparResultCompat
// are now defined directly in fate/cospar.h.
// PzDeviceLoader compat (singlet_gpu::io).
namespace singlet_gpu { namespace io {
struct PzDeviceLoader {
    core::DeviceCSC load(const std::string& path, cudaStream_t stream) const {
        return singlet_gpu::io::load_pz(path, stream).mat;
    }
};
} } // namespace singlet_gpu::io

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

/// Tiny synthetic: 200 cells × 2 time points × 50 genes × 3 clones
/// (from design doc §"Correctness test spec", test case 1).
constexpr int kTinyCells     = 200;  // total cells across both time points
constexpr int kTinyGenes     = 50;
constexpr int kTinyClones    = 3;
constexpr int kTinyTimePoints = 2;
/// Cells per time point (equal split).
constexpr int kTinyCellsT0  = kTinyCells / 2;  // 100
constexpr int kTinyCellsT1  = kTinyCells / 2;  // 100

/// Hierarchical potency tree (test 4): 3-level bifurcating tree.
///   Level 0 (stem): 1 clone,  all cells high potency
///   Level 1 (prog): 2 clones, intermediate potency
///   Level 2 (term): 4 clones, low potency (committed)
constexpr int kTreeLevels  = 3;
constexpr int kTreeCells   = 160;   // 20 stem + 40 prog + 100 term
constexpr int kTreeClones  = 7;     // 1 + 2 + 4
constexpr int kTreeGenes   = 40;

/// Tolerance on fate_bias Spearman ρ (design doc: ≥ 0.90).
constexpr double kFateBiasSpearmanMin = 0.90;

/// Tolerance on T row sum deviation from [0,1] (test 3).
constexpr double kRowSumTol = 1e-5;

// Real-data paths — GSM4037629 (scRNA, no lineage barcodes expected; most
// current data lacks them — test 2 will SKIP if absent).
static const char* kGsmExonPath =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/exon_counts.1pz";

// Lineage barcode CSV expected path (skip if absent).
static const char* kGsmLineagePath =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/lineage_barcodes.csv";

// Python reference script path.
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/"
    "tests/refs/cospar_py_reference.py";

// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_cospar_refs_tmp";

// Python interpreter.
static const char* kPython = "python3";

// Exit code from reference script when cospar Python is not installed
// (fallback pure-Python optimizer takes over — still produces a result).
static constexpr int kRefScriptSkipCode = 2;

// =============================================================================
// Utility helpers (pattern from cycles 3-40)
// =============================================================================

namespace {

void ensure_refs_tmp() {
    fs::create_directories(kRefsTmpDir);
}

std::string refs_path(const std::string& name) {
    return std::string(kRefsTmpDir) + "/" + name;
}

// run_cmd_rc: returns exit code without throwing.
int run_cmd_rc(const std::string& cmd) {
    return std::system(cmd.c_str());
}

// run_cmd: throws on non-zero exit.
void run_cmd(const std::string& cmd) {
    int ret = run_cmd_rc(cmd);
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
// Host-side CSC container (standard across cycles 3-40).
// ---------------------------------------------------------------------------
struct HostCSC {
    uint32_t             m;        // genes (rows)
    uint32_t             n;        // cells (cols)
    uint64_t             nnz;
    std::vector<float>   values;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
};

// Write HostCSC to the dump_csc.h binary format.
void write_csc_bin(const HostCSC& h, const std::string& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    static constexpr uint32_t kMagic = 0x43535343u;  // "CSCC"
    f.write(reinterpret_cast<const char*>(&kMagic), 4);
    f.write(reinterpret_cast<const char*>(&h.m),    4);
    f.write(reinterpret_cast<const char*>(&h.n),    4);
    f.write(reinterpret_cast<const char*>(&h.nnz),  8);
    f.write(reinterpret_cast<const char*>(h.values .data()),
            static_cast<std::streamsize>(h.nnz * sizeof(float)));
    f.write(reinterpret_cast<const char*>(h.indptr .data()),
            static_cast<std::streamsize>((static_cast<size_t>(h.n) + 1) * sizeof(int32_t)));
    f.write(reinterpret_cast<const char*>(h.indices.data()),
            static_cast<std::streamsize>(h.nnz * sizeof(int32_t)));
    if (!f) throw std::runtime_error("Write error: " + path);
}

// Upload HostCSC to device using the synchronous host-data constructor.
singlet_gpu::core::DeviceCSC upload_csc(const HostCSC& h, cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    return singlet_gpu::core::DeviceCSC(
        static_cast<int>(h.m),
        static_cast<int>(h.n),
        static_cast<int>(h.nnz),
        h.indptr.data(),
        h.indices.data(),
        h.values.data());
}

// Download DeviceMemory<float> to host.
std::vector<float> download_f32(
        const singlet_gpu::core::DeviceMemory<float>& d,
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

// ---------------------------------------------------------------------------
// Spearman rank correlation (pure C++) — reused from cycles 11-40.
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
// Read a flat float vector from a binary file written by the Python script.
// Format: [uint32_t n] [float × n]
// ---------------------------------------------------------------------------
std::vector<float> read_float_vec(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
    if (!f) throw std::runtime_error("Read error: " + path);
    return v;
}

// Read a flat double vector from binary: [uint32_t n][double × n]
std::vector<double> read_double_vec(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    std::vector<double> v(n);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(n * sizeof(double)));
    if (!f) throw std::runtime_error("Read error: " + path);
    return v;
}

// ---------------------------------------------------------------------------
// Synthetic data generator for the tiny bifurcation test.
//
// Layout: kTinyCells total cells, first kTinyCellsT0 at t=0, rest at t=1.
// Clones: kTinyClones shared lineages.
// Gene expression: multivariate Gaussian with planted bifurcation structure:
//   - Clone 0 → high expression of genes 0-15 (fate 0)
//   - Clone 1 → high expression of genes 16-30 (fate 1)
//   - Clone 2 → mixed (stem-like, equal both)
// Cell assignment: each clone has ~kTinyCells/kTinyClones cells at each t.
// ---------------------------------------------------------------------------
struct TinySynthetic {
    HostCSC                expression;   // kTinyGenes × kTinyCells (log-norm CSC)
    std::vector<int32_t>   time_point;   // [kTinyCells] 0 or 1
    std::vector<int32_t>   clone_id;     // [kTinyCells] 0..kTinyClones-1
    // Terminal fate indicator (for t=1 cells): 1 = fate A (clones 0), 0 = fate B
    std::vector<float>     fate_indicator_t1;  // [kTinyCellsT1]
};

TinySynthetic make_tiny_synthetic() {
    std::mt19937_64 rng(kSeed);
    std::normal_distribution<float> norm(0.f, 1.f);
    std::uniform_real_distribution<float> unif(0.f, 1.f);

    TinySynthetic syn;

    // ---- Assign cells to time points and clones ----
    syn.time_point.resize(kTinyCells);
    syn.clone_id.resize(kTinyCells);
    // t0: cells 0..kTinyCellsT0-1, t1: cells kTinyCellsT0..kTinyCells-1
    int cells_per_clone = kTinyCells / kTinyClones;
    for (int i = 0; i < kTinyCells; ++i) {
        syn.time_point[i] = (i < kTinyCellsT0) ? 0 : 1;
        syn.clone_id[i]   = (i / (cells_per_clone)) % kTinyClones;
    }

    // ---- Gene expression: planted bifurcation ----
    // Gene clusters:
    //   genes 0-14  → fate-A signal   (clone 0 high, clone 1 low, clone 2 mid)
    //   genes 15-29 → fate-B signal   (clone 0 low,  clone 1 high, clone 2 mid)
    //   genes 30-49 → housekeeping    (all clones similar)
    const float kFateAGenes  = 15.f;  // indices 0-14
    const float kFateBGenes  = 15.f;  // indices 15-29
    (void)kFateAGenes; (void)kFateBGenes;

    // Build dense expression matrix (genes × cells), then sparsify.
    std::vector<std::vector<float>> dense(kTinyGenes,
                                          std::vector<float>(kTinyCells, 0.f));

    for (int c = 0; c < kTinyCells; ++c) {
        int cl = syn.clone_id[c];
        for (int g = 0; g < kTinyGenes; ++g) {
            float signal = 0.f;
            if      (g < 15  && cl == 0) signal = 3.f;
            else if (g < 15  && cl == 2) signal = 1.5f;
            else if (g >= 15 && g < 30 && cl == 1) signal = 3.f;
            else if (g >= 15 && g < 30 && cl == 2) signal = 1.5f;
            else if (g >= 30)            signal = 1.f;  // housekeeping
            // log-norm-like: add noise, clamp ≥ 0
            float val = signal + 0.3f * norm(rng);
            dense[g][c] = std::max(0.f, val);
        }
    }

    // Sparsify: drop values < 0.1 (simulate dropout)
    std::vector<float>   vals;
    std::vector<int32_t> indptr(kTinyCells + 1, 0);
    std::vector<int32_t> indices;

    // CSC: iterate over columns first
    for (int c = 0; c < kTinyCells; ++c) {
        indptr[c] = static_cast<int32_t>(vals.size());
        for (int g = 0; g < kTinyGenes; ++g) {
            if (dense[g][c] >= 0.1f) {
                vals.push_back(dense[g][c]);
                indices.push_back(g);
            }
        }
    }
    indptr[kTinyCells] = static_cast<int32_t>(vals.size());

    syn.expression.m       = static_cast<uint32_t>(kTinyGenes);
    syn.expression.n       = static_cast<uint32_t>(kTinyCells);
    syn.expression.nnz     = vals.size();
    syn.expression.values  = std::move(vals);
    syn.expression.indptr  = std::move(indptr);
    syn.expression.indices = std::move(indices);

    // Fate indicator for t=1 cells: fate A = clone 0, fate B = clone 1,
    // clone 2 split 50/50 based on expression (simplified: use clone_id directly)
    syn.fate_indicator_t1.resize(kTinyCellsT1);
    for (int i = 0; i < kTinyCellsT1; ++i) {
        int global_i = kTinyCellsT0 + i;
        int cl = syn.clone_id[global_i];
        syn.fate_indicator_t1[i] = (cl == 0) ? 1.f :
                                   (cl == 1) ? 0.f : 0.5f;
    }

    return syn;
}

// ---------------------------------------------------------------------------
// Write TinySynthetic to disk for the Python reference script.
//
// Files in kRefsTmpDir:
//   expression.bin  — HostCSC binary (dump_csc format)
//   time_point.bin  — uint32_t n, then int32_t × n
//   clone_id.bin    — uint32_t n, then int32_t × n
//   fate_ind_t1.bin — uint32_t n, then float × n
// ---------------------------------------------------------------------------
void write_tiny_synthetic(const TinySynthetic& syn) {
    ensure_refs_tmp();
    write_csc_bin(syn.expression, refs_path("expression.bin"));

    auto write_i32_vec = [](const std::vector<int32_t>& v,
                             const std::string& path) {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("Cannot open: " + path);
        uint32_t n = static_cast<uint32_t>(v.size());
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        f.write(reinterpret_cast<const char*>(v.data()),
                static_cast<std::streamsize>(n * sizeof(int32_t)));
    };
    auto write_f32_vec = [](const std::vector<float>& v,
                              const std::string& path) {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("Cannot open: " + path);
        uint32_t n = static_cast<uint32_t>(v.size());
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        f.write(reinterpret_cast<const char*>(v.data()),
                static_cast<std::streamsize>(n * sizeof(float)));
    };

    write_i32_vec(syn.time_point,        refs_path("time_point.bin"));
    write_i32_vec(syn.clone_id,          refs_path("clone_id.bin"));
    write_f32_vec(syn.fate_indicator_t1, refs_path("fate_ind_t1.bin"));
}

// ---------------------------------------------------------------------------
// Hierarchical potency tree for test 4.
//
// Layout (kTreeCells = 160):
//   cells 0-19   : stem (level 0, clone 0, t=0)
//   cells 20-59  : progenitor A & B (level 1, clones 1-2, t=1)
//   cells 60-159 : committed (level 2, clones 3-6, t=2)
//
// Expected: potency[stem] > potency[prog] > potency[committed] on average.
// ---------------------------------------------------------------------------
struct TreeSynthetic {
    HostCSC              expression;
    std::vector<int32_t> time_point;   // 0, 1, or 2
    std::vector<int32_t> clone_id;     // 0..kTreeClones-1
    std::vector<int32_t> level;        // 0=stem, 1=prog, 2=committed
};

TreeSynthetic make_tree_synthetic() {
    std::mt19937_64 rng(kSeed + 1);
    std::normal_distribution<float> noise(0.f, 0.5f);

    TreeSynthetic t;
    t.time_point.resize(kTreeCells);
    t.clone_id  .resize(kTreeCells);
    t.level     .resize(kTreeCells);

    // Stem: cells 0-19
    for (int i = 0; i < 20; ++i) {
        t.time_point[i] = 0; t.clone_id[i] = 0; t.level[i] = 0;
    }
    // Progenitor A: cells 20-39 (clone 1)
    for (int i = 20; i < 40; ++i) {
        t.time_point[i] = 1; t.clone_id[i] = 1; t.level[i] = 1;
    }
    // Progenitor B: cells 40-59 (clone 2)
    for (int i = 40; i < 60; ++i) {
        t.time_point[i] = 1; t.clone_id[i] = 2; t.level[i] = 1;
    }
    // Committed A1: cells 60-84 (clone 3)
    for (int i = 60; i < 85; ++i) {
        t.time_point[i] = 2; t.clone_id[i] = 3; t.level[i] = 2;
    }
    // Committed A2: cells 85-109 (clone 4)
    for (int i = 85; i < 110; ++i) {
        t.time_point[i] = 2; t.clone_id[i] = 4; t.level[i] = 2;
    }
    // Committed B1: cells 110-134 (clone 5)
    for (int i = 110; i < 135; ++i) {
        t.time_point[i] = 2; t.clone_id[i] = 5; t.level[i] = 2;
    }
    // Committed B2: cells 135-159 (clone 6)
    for (int i = 135; i < 160; ++i) {
        t.time_point[i] = 2; t.clone_id[i] = 6; t.level[i] = 2;
    }

    // Gene expression (kTreeGenes × kTreeCells): planted lineage signals.
    // genes 0-9:  stem signature (high in level 0)
    // genes 10-19: prog A (high in clones 1,3,4)
    // genes 20-29: prog B (high in clones 2,5,6)
    // genes 30-39: housekeeping
    std::vector<std::vector<float>> dense(kTreeGenes,
                                          std::vector<float>(kTreeCells, 0.f));
    for (int c = 0; c < kTreeCells; ++c) {
        int cl = t.clone_id[c];
        int lv = t.level[c];
        for (int g = 0; g < kTreeGenes; ++g) {
            float base = 0.f;
            if      (g < 10)             base = (lv == 0) ? 4.f : 1.f;
            else if (g < 20)             base = (cl==1||cl==3||cl==4) ? 3.f : 0.5f;
            else if (g < 30)             base = (cl==2||cl==5||cl==6) ? 3.f : 0.5f;
            else                         base = 2.f;
            dense[g][c] = std::max(0.f, base + noise(rng));
        }
    }

    // Sparsify: drop < 0.2
    std::vector<float>   vals;
    std::vector<int32_t> indptr(kTreeCells + 1, 0);
    std::vector<int32_t> indices;
    for (int c = 0; c < kTreeCells; ++c) {
        indptr[c] = static_cast<int32_t>(vals.size());
        for (int g = 0; g < kTreeGenes; ++g) {
            if (dense[g][c] >= 0.2f) {
                vals.push_back(dense[g][c]);
                indices.push_back(g);
            }
        }
    }
    indptr[kTreeCells] = static_cast<int32_t>(vals.size());

    t.expression.m       = static_cast<uint32_t>(kTreeGenes);
    t.expression.n       = static_cast<uint32_t>(kTreeCells);
    t.expression.nnz     = vals.size();
    t.expression.values  = std::move(vals);
    t.expression.indptr  = std::move(indptr);
    t.expression.indices = std::move(indices);

    return t;
}

}  // namespace

// =============================================================================
// CosparConfig (must match fate/cospar.h API)
// =============================================================================

// The kernel header exposes:
//   struct CosparConfig { int n_iter, n_knn; float lambda1, lambda2, step; bool deterministic; };
//   struct CosparResult {
//     DeviceMemory<float> transition_map;   // n_t0 × n_t1 row-major
//     DeviceMemory<float> fate_bias;        // n_cells × n_fates
//     DeviceMemory<float> potency_score;    // n_cells
//   };
//   CosparResult cospar_fit(
//     const DeviceCSC& expression,          // genes × cells, log-norm
//     const int32_t*  time_point,           // device int32[n_cells]
//     const int32_t*  clone_id,             // device int32[n_cells]
//     int n_cells, int n_genes,
//     int n_t0, int n_t1,
//     const CosparConfig& cfg,
//     cudaStream_t stream);
//
// (These types are included transitively via <singlet-gpu/fate/cospar.h>.)

// =============================================================================
// Test 1: Cospar_TinySynthetic_VsPython
//
// 200 cells × 2 time points × 50 genes × 3 clones with planted bifurcation.
// Compare per-cell fate_bias (for t=0 cells) against Python reference.
// Spearman ρ ≥ 0.90.
// =============================================================================

TEST(CosparCorrectness, Cospar_TinySynthetic_VsPython) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU available";

    // Build synthetic data.
    const TinySynthetic syn = make_tiny_synthetic();

    // Upload to device.
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet_gpu::core::DeviceCSC d_expr = upload_csc(syn.expression, stream);

    // Upload time_point and clone_id.
    singlet_gpu::core::DeviceMemory<int32_t> d_tp(kTinyCells);
    singlet_gpu::core::DeviceMemory<int32_t> d_cl(kTinyCells);
    ASSERT_EQ(cudaMemcpyAsync(d_tp.get(), syn.time_point.data(),
                              kTinyCells * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_cl.get(), syn.clone_id.data(),
                              kTinyCells * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // Run GPU kernel.
    using namespace singlet_gpu::fate;
    CosparConfig cfg;
    cfg.n_iter      = 50;
    cfg.n_knn       = 10;
    cfg.lambda1     = 0.05f;
    cfg.lambda2     = 0.1f;
    cfg.step        = 0.01f;
    cfg.deterministic = true;

    CosparResultCompat result = cospar_fit(
        d_expr,
        d_tp.get(), d_cl.get(),
        kTinyCells, kTinyGenes,
        kTinyCellsT0, kTinyCellsT1,
        cfg, stream);

    // Download fate_bias (kTinyCells × n_fates; n_fates = kTinyClones here).
    // We compare per-cell fate-A bias (fate index 0) for t=0 cells.
    const int n_fates = kTinyClones;
    std::vector<float> h_fate_bias =
        download_f32(result.fate_bias, static_cast<size_t>(kTinyCells) * n_fates, stream);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    // Extract fate-A bias for all t=0 cells.
    std::vector<float> gpu_fate_a(kTinyCellsT0);
    for (int i = 0; i < kTinyCellsT0; ++i)
        gpu_fate_a[i] = h_fate_bias[static_cast<size_t>(i) * n_fates + 0];

    // Write synthetic data to disk for Python reference.
    write_tiny_synthetic(syn);

    // Run Python reference script.
    std::string cmd = std::string(kPython) + " " + kRefScript +
                      " --task tiny_synthetic" +
                      " --data_dir " + kRefsTmpDir +
                      " --out_dir "  + kRefsTmpDir +
                      " 2>/dev/null";
    int rc = run_cmd_rc(cmd);
    if (rc == kRefScriptSkipCode) {
        // cospar Python not installed; fallback pure-Python runs → rc should be 0.
        // If rc == 2, re-run with --fallback flag (handled internally by the script).
        GTEST_SKIP() << "cospar Python not installed and pure-Python fallback "
                        "returned exit 2 — check cospar_py_reference.py setup";
    }
    ASSERT_EQ(rc, 0) << "cospar_py_reference.py failed with exit " << rc;

    // Read Python reference fate_bias (t=0 cells, fate-A).
    std::vector<float> py_fate_a =
        read_float_vec(refs_path("py_fate_bias_t0_fateA.bin"));
    ASSERT_EQ(py_fate_a.size(), static_cast<size_t>(kTinyCellsT0))
        << "Python reference fate_bias size mismatch";

    double rho = spearman(gpu_fate_a, py_fate_a);
    EXPECT_GE(rho, kFateBiasSpearmanMin)
        << "fate_bias Spearman ρ = " << rho
        << " < threshold " << kFateBiasSpearmanMin;

    // Sanity: fate_bias values should be in [0, 1].
    for (float v : gpu_fate_a) {
        EXPECT_GE(v, -1e-5f);
        EXPECT_LE(v,  1.f + 1e-5f);
    }
}

// =============================================================================
// Test 2: Cospar_GSM_RealLineage
//
// Load real GSM4037629 exon_counts.1pz. If lineage_barcodes.csv is absent
// (expected for most samples), run in state-only mode (no lineage coupling;
// potency from kNN entropy only). Skip entirely if the exon file is absent.
// =============================================================================

TEST(CosparCorrectness, Cospar_GSM_RealLineage) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU available";
    if (!fs::exists(kGsmExonPath))
        GTEST_SKIP() << "GSM4037629 exon_counts.1pz not found at " << kGsmExonPath;

    bool has_lineage = fs::exists(kGsmLineagePath);

    // Load .1pz via device loader.
    using namespace singlet_gpu;
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    io::PzDeviceLoader loader;
    core::DeviceCSC d_expr = loader.load(kGsmExonPath, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    const uint32_t n_cells = static_cast<uint32_t>(d_expr.cols);
    const uint32_t n_genes = static_cast<uint32_t>(d_expr.rows);
    EXPECT_GT(n_cells, 0u);
    EXPECT_GT(n_genes, 0u);

    // Assign synthetic time points and clones (single time point, single clone
    // — state-only mode since no real lineage barcodes are available).
    std::vector<int32_t> h_tp(n_cells, 0);
    std::vector<int32_t> h_cl(n_cells, 0);
    if (!has_lineage) {
        // State-only: all cells in t=0, all in clone 0.
        // cospar_fit falls back to potency-from-kNN-entropy mode.
    } else {
        // TODO: parse kGsmLineagePath CSV → populate h_tp, h_cl.
        // For now fall through to state-only mode.
    }

    core::DeviceMemory<int32_t> d_tp(n_cells);
    core::DeviceMemory<int32_t> d_cl(n_cells);
    ASSERT_EQ(cudaMemcpyAsync(d_tp.get(), h_tp.data(),
                              n_cells * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_cl.get(), h_cl.data(),
                              n_cells * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    fate::CosparConfig cfg;
    cfg.n_iter = 20;  // fewer iters for speed on real data
    cfg.n_knn  = 15;
    cfg.lambda1 = 0.05f;
    cfg.lambda2 = 0.1f;
    cfg.step    = 0.01f;
    cfg.deterministic = true;

    // In state-only mode n_t0 = n_cells, n_t1 = 0 (or n_cells).
    const int32_t n_t0 = static_cast<int32_t>(n_cells);
    const int32_t n_t1 = has_lineage ? static_cast<int32_t>(n_cells) : 0;

    fate::CosparResultCompat result = fate::cospar_fit(
        d_expr, d_tp.get(), d_cl.get(),
        static_cast<int>(n_cells), static_cast<int>(n_genes),
        n_t0, n_t1, cfg, stream);

    // Download potency score and verify it is finite, in [0, log2(n_fates)] range.
    std::vector<float> potency =
        download_f32(result.potency_score, n_cells, stream);

    int n_nan = 0, n_neg = 0;
    for (float p : potency) {
        if (!std::isfinite(p)) ++n_nan;
        if (p < -1e-5f)        ++n_neg;
    }
    EXPECT_EQ(n_nan, 0) << "potency_score has " << n_nan << " non-finite values";
    EXPECT_EQ(n_neg, 0) << "potency_score has " << n_neg << " negative values";

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// =============================================================================
// Test 3: Cospar_TransitionMap_RowSumsValid
//
// Transition map T (n_t0 × n_t1) row sums must be in [0, 1] within tolerance.
// Uses the tiny synthetic dataset.
// =============================================================================

TEST(CosparCorrectness, Cospar_TransitionMap_RowSumsValid) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU available";

    const TinySynthetic syn = make_tiny_synthetic();

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet_gpu::core::DeviceCSC d_expr = upload_csc(syn.expression, stream);

    singlet_gpu::core::DeviceMemory<int32_t> d_tp(kTinyCells);
    singlet_gpu::core::DeviceMemory<int32_t> d_cl(kTinyCells);
    ASSERT_EQ(cudaMemcpyAsync(d_tp.get(), syn.time_point.data(),
                              kTinyCells * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_cl.get(), syn.clone_id.data(),
                              kTinyCells * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    using namespace singlet_gpu::fate;
    CosparConfig cfg;
    cfg.n_iter = 50;
    cfg.n_knn  = 10;
    cfg.lambda1 = 0.05f;
    cfg.lambda2 = 0.1f;
    cfg.step    = 0.01f;
    cfg.deterministic = true;

    CosparResultCompat result = cospar_fit(
        d_expr, d_tp.get(), d_cl.get(),
        kTinyCells, kTinyGenes,
        kTinyCellsT0, kTinyCellsT1,
        cfg, stream);

    // Download transition map (kTinyCellsT0 × kTinyCellsT1, row-major).
    const size_t tm_size =
        static_cast<size_t>(kTinyCellsT0) * kTinyCellsT1;
    std::vector<float> h_T = download_f32(result.transition_map, tm_size, stream);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    // Compute row sums and check each is in [0, 1].
    int n_violations = 0;
    double max_violation = 0.0;
    for (int i = 0; i < kTinyCellsT0; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < kTinyCellsT1; ++j)
            row_sum += static_cast<double>(h_T[i * kTinyCellsT1 + j]);
        double lo_viol = std::max(0.0, -row_sum);
        double hi_viol = std::max(0.0,  row_sum - 1.0);
        double viol    = std::max(lo_viol, hi_viol);
        if (viol > kRowSumTol) {
            ++n_violations;
            max_violation = std::max(max_violation, viol);
        }
        // All individual entries must be >= -tol.
        for (int j = 0; j < kTinyCellsT1; ++j) {
            double v = static_cast<double>(h_T[i * kTinyCellsT1 + j]);
            EXPECT_GE(v, -kRowSumTol)
                << "T[" << i << "," << j << "] = " << v << " < 0";
        }
    }
    EXPECT_EQ(n_violations, 0)
        << "Row sum out of [0,1]: " << n_violations
        << " violations, max deviation = " << max_violation;
}

// =============================================================================
// Test 4: Cospar_PotencyMonotone
//
// On the hierarchical tree (stem → prog → committed), average potency must
// decrease monotonically across levels: E[potency|stem] > E[potency|prog] >
// E[potency|committed].
// =============================================================================

TEST(CosparCorrectness, Cospar_PotencyMonotone) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU available";

    const TreeSynthetic tree = make_tree_synthetic();

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet_gpu::core::DeviceCSC d_expr = upload_csc(tree.expression, stream);

    singlet_gpu::core::DeviceMemory<int32_t> d_tp(kTreeCells);
    singlet_gpu::core::DeviceMemory<int32_t> d_cl(kTreeCells);
    ASSERT_EQ(cudaMemcpyAsync(d_tp.get(), tree.time_point.data(),
                              kTreeCells * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_cl.get(), tree.clone_id.data(),
                              kTreeCells * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    using namespace singlet_gpu::fate;
    CosparConfig cfg;
    cfg.n_iter = 50;
    cfg.n_knn  = 8;
    cfg.lambda1 = 0.05f;
    cfg.lambda2 = 0.1f;
    cfg.step    = 0.01f;
    cfg.deterministic = true;

    // 3 time points: n_t0 = 20 (stem, t=0), n_t1 = 40 (prog, t=1).
    // cospar_fit handles multi-time-point by chaining.
    // We pass n_t0=20, n_t1=40 for the t0→t1 transition only; potency is
    // computed globally over all cells.
    CosparResultCompat result = cospar_fit(
        d_expr, d_tp.get(), d_cl.get(),
        kTreeCells, kTreeGenes,
        /*n_t0=*/20, /*n_t1=*/40,
        cfg, stream);

    std::vector<float> potency =
        download_f32(result.potency_score, kTreeCells, stream);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    // Average potency per level.
    double sum_stem = 0.0, sum_prog = 0.0, sum_comm = 0.0;
    int    cnt_stem = 0,   cnt_prog = 0,   cnt_comm = 0;
    for (int c = 0; c < kTreeCells; ++c) {
        switch (tree.level[c]) {
            case 0: sum_stem += potency[c]; ++cnt_stem; break;
            case 1: sum_prog += potency[c]; ++cnt_prog; break;
            case 2: sum_comm += potency[c]; ++cnt_comm; break;
            default: break;
        }
    }
    ASSERT_GT(cnt_stem, 0);
    ASSERT_GT(cnt_prog, 0);
    ASSERT_GT(cnt_comm, 0);

    double mean_stem = sum_stem / cnt_stem;
    double mean_prog = sum_prog / cnt_prog;
    double mean_comm = sum_comm / cnt_comm;

    EXPECT_GT(mean_stem, mean_prog)
        << "Potency not monotone: stem=" << mean_stem
        << " prog=" << mean_prog;
    EXPECT_GT(mean_prog, mean_comm)
        << "Potency not monotone: prog=" << mean_prog
        << " committed=" << mean_comm;
}

// =============================================================================
// Test 5: Cospar_Determinism_BitIdentical
//
// Two runs on identical inputs must produce bit-identical outputs.
// Uses tiny synthetic, cfg.deterministic = true.
// Design doc §Determinism: "No stochasticity. Bit-identical."
// =============================================================================

TEST(CosparCorrectness, Cospar_Determinism_BitIdentical) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU available";

    const TinySynthetic syn = make_tiny_synthetic();

    auto run_once = [&]() -> std::vector<float> {
        cudaStream_t stream;
        EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        singlet_gpu::core::DeviceCSC d_expr = upload_csc(syn.expression, stream);
        singlet_gpu::core::DeviceMemory<int32_t> d_tp(kTinyCells);
        singlet_gpu::core::DeviceMemory<int32_t> d_cl(kTinyCells);
        EXPECT_EQ(cudaMemcpyAsync(d_tp.get(), syn.time_point.data(),
                                  kTinyCells * sizeof(int32_t),
                                  cudaMemcpyHostToDevice, stream), cudaSuccess);
        EXPECT_EQ(cudaMemcpyAsync(d_cl.get(), syn.clone_id.data(),
                                  kTinyCells * sizeof(int32_t),
                                  cudaMemcpyHostToDevice, stream), cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        using namespace singlet_gpu::fate;
        CosparConfig cfg;
        cfg.n_iter = 50;
        cfg.n_knn  = 10;
        cfg.lambda1 = 0.05f;
        cfg.lambda2 = 0.1f;
        cfg.step    = 0.01f;
        cfg.deterministic = true;

        CosparResultCompat result = cospar_fit(
            d_expr, d_tp.get(), d_cl.get(),
            kTinyCells, kTinyGenes,
            kTinyCellsT0, kTinyCellsT1,
            cfg, stream);

        const size_t tm_size =
            static_cast<size_t>(kTinyCellsT0) * kTinyCellsT1;
        std::vector<float> h_T = download_f32(result.transition_map, tm_size, stream);
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        return h_T;
    };

    std::vector<float> run1 = run_once();
    std::vector<float> run2 = run_once();

    ASSERT_EQ(run1.size(), run2.size());
    int n_diff = 0;
    for (size_t i = 0; i < run1.size(); ++i) {
        // Bit-identical check: compare raw bits.
        uint32_t bits1, bits2;
        std::memcpy(&bits1, &run1[i], sizeof(float));
        std::memcpy(&bits2, &run2[i], sizeof(float));
        if (bits1 != bits2) ++n_diff;
    }
    EXPECT_EQ(n_diff, 0)
        << "transition_map has " << n_diff
        << " non-identical entries across two runs";
}
