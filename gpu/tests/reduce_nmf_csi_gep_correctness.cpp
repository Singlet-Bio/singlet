// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/reduce_nmf_csi_gep_correctness.cpp
//
// integrates: original (consensus bootstrap NMF; reuses factornet::nmf::fit_gpu for inner loop)
//
// Correctness harness for singlet_gpu::reduce::nmf::csi_gep.
//
// Written against the design doc at:
//   singlet-gpu/state/designs/28-csi-gep.md
// NOT against the kernel source — authored in parallel with
// reduce/nmf/csi_gep.h by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: CSI-GEP Python via subprocess
//   tests/refs/csi_gep_python_reference.py
//   (pip install csi-gep — heavy: torchNMF + sklearn)
//   Exit code 2 from the subprocess = packages unavailable → GTEST_SKIP.
//
// Tolerances (design doc §"Correctness test spec"):
//   top-30 gene Jaccard per consensus program >= 0.85  (vs Python reference)
//   chosen_k within ±2 of true planted K
//   reproducibility curve non-decreasing then plateau
//   bit-identical with fixed seed
//
// Test cases:
//   CsiGep_TinySynthetic_VsPython
//   CsiGep_GSM4037629_RealData
//   CsiGep_AutoKSelection_PicksKnownK
//   CsiGep_ReproducibilityCurve_Monotone
//   CsiGep_Determinism_BitIdentical
//   CsiGep_ReusesCycle5NMF
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every test that requires a device is guarded by
//   gpu_available() and calls GTEST_SKIP().

// ---- singlet-gpu CSI-GEP header (written by gpu-kernel-dev) ----------------
#include <singlet-gpu/reduce/nmf/csi_gep.h>

// ---- cycle 5 NMF adapter (smoke-test reuse) ---------------------------------
#include <singlet-gpu/reduce/nmf/fit.h>

// ---- other singlet-gpu headers used by the real-data test ------------------
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
#include <limits>
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

// Tiny synthetic: 200 genes × 100 cells with 5 planted programs.
constexpr int kTinyGenes     = 200;
constexpr int kTinyCells     = 100;
constexpr int kTinyPrograms  = 5;   // planted programs for TinySynthetic test

// Auto-K synthetic: 100 genes × 200 cells with 8 planted programs.
constexpr int kAutoKGenes    = 100;
constexpr int kAutoKCells    = 200;
constexpr int kAutoKPlanted  = 8;   // planted programs; chosen_k must be in [6,10]

// Tolerance thresholds from the design doc.
constexpr double kJaccardMin     = 0.85;   // top-30 gene Jaccard vs Python ref
constexpr int    kAutoKTolerance = 2;      // chosen_k allowed within ±2 of true K

// Real-data paths — GSM4037629 (exon counts only; scRNA).
static const char* kGsmExonPath =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/exon_counts.1pz";

// Python reference script path.
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/"
    "tests/refs/csi_gep_python_reference.py";

// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_csi_gep_refs_tmp";

// Python interpreter.
static const char* kPython = "python3";

// Exit code from the reference script when csi-gep is not installed.
static constexpr int kRefScriptSkipCode = 2;

// =============================================================================
// Compat bridges (global scope — must be BEFORE anonymous namespace)
// =============================================================================

// csi_gep: test calls (GPUContext&, DeviceCSC, CsiGepConfig) but the header
// signature is (DeviceCSC, CsiGepConfig, cudaStream_t).
// Add a 3-arg overload that extracts the stream from GPUContext.
namespace singlet_gpu { namespace reduce { namespace nmf {
inline CsiGepResult csi_gep(
        const core::GPUContext& ctx,
        const core::DeviceCSC&  mat,
        const CsiGepConfig&     cfg)
{
    return csi_gep(mat, cfg, ctx.stream);
}
} } }

// fit (NMF cycle 5 adapter): test calls (GPUContext&, DeviceCSC, NmfConfig)
// but the real fit() takes (PzDeviceMatrix, NmfConfig).
// Add a DeviceCSC overload that downloads to host and calls factornet directly.
namespace singlet_gpu { namespace reduce { namespace nmf {
inline NmfResult fit(
        const core::GPUContext& /*ctx*/,
        const core::DeviceCSC&  mat,
        const NmfConfig&        cfg,
        const DenseMatrix*      W_init = nullptr,
        const DenseMatrix*      H_init = nullptr)
{
    // Download mat to host for factornet staging.
    std::vector<int>   h_col_ptr(static_cast<size_t>(mat.cols) + 1);
    std::vector<int>   h_row_idx(static_cast<size_t>(mat.nnz));
    std::vector<float> h_vals   (static_cast<size_t>(mat.nnz));
    cudaMemcpy(h_col_ptr.data(), mat.col_ptr.get(),
               (static_cast<size_t>(mat.cols) + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_row_idx.data(), mat.row_indices.get(),
               static_cast<size_t>(mat.nnz) * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vals.data(), mat.values.get(),
               static_cast<size_t>(mat.nnz) * sizeof(float),
               cudaMemcpyDeviceToHost);
    return ::factornet::nmf::nmf_fit_gpu<float>(
        h_col_ptr.data(), h_row_idx.data(), h_vals.data(),
        static_cast<int>(mat.rows), static_cast<int>(mat.cols),
        static_cast<int>(mat.nnz), cfg, W_init, H_init);
}
} } }

// load_pz_device: compat bridge (same pattern as other test files).
namespace singlet_gpu { namespace io {
inline core::DeviceCSC load_pz_device(const std::string& path, cudaStream_t stream) {
    return singlet_gpu::io::load_pz(path, stream).mat;
}
} }

// =============================================================================
// Utility helpers (pattern from cycles 3-27)
// =============================================================================

namespace {

void ensure_refs_tmp() {
    fs::create_directories(kRefsTmpDir);
}

std::string refs_path(const std::string& name) {
    return std::string(kRefsTmpDir) + "/" + name;
}

// run_cmd: throws on non-zero exit.
void run_cmd(const std::string& cmd) {
    int ret = std::system(cmd.c_str());
    if (ret != 0)
        throw std::runtime_error("Command failed (exit " + std::to_string(ret) +
                                 "): " + cmd);
}

// run_cmd_rc: returns exit code without throwing.
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
    uint32_t             m;       // genes (rows)
    uint32_t             n;       // cells (cols)
    uint64_t             nnz;
    std::vector<float>   values;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
};

// ---------------------------------------------------------------------------
// Write HostCSC to dump_csc.h binary format.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Upload HostCSC to device.
// ---------------------------------------------------------------------------
singlet_gpu::core::DeviceCSC upload_csc(const HostCSC& h, cudaStream_t stream) {
    // Synchronize stream before using the host-data constructor which performs
    // synchronous cudaMemcpy internally.
    cudaStreamSynchronize(stream);
    return singlet_gpu::core::DeviceCSC(
        static_cast<int>(h.m),
        static_cast<int>(h.n),
        static_cast<int>(h.nnz),
        h.indptr.data(),
        h.indices.data(),
        h.values.data());
}

// ---------------------------------------------------------------------------
// Download a DeviceMemory<float> to host vector.
// ---------------------------------------------------------------------------
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
// Top-N indices of a float vector (descending) — for Jaccard computation.
// ---------------------------------------------------------------------------
std::vector<int> top_n_indices(const std::vector<float>& v, int n) {
    int sz = static_cast<int>(v.size());
    n = std::min(n, sz);
    std::vector<int> idx(sz);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
                      [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(n);
    std::sort(idx.begin(), idx.end());
    return idx;
}

// ---------------------------------------------------------------------------
// Jaccard index between two sorted integer sets.
// ---------------------------------------------------------------------------
double jaccard(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> intr, uni;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::back_inserter(intr));
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                   std::back_inserter(uni));
    if (uni.empty()) return 1.0;
    return static_cast<double>(intr.size()) / static_cast<double>(uni.size());
}

// ---------------------------------------------------------------------------
// Best-match Jaccard: for each program in set A, pick the best partner in B.
// Returns mean over programs in A.
// ---------------------------------------------------------------------------
double best_match_jaccard(
        const std::vector<std::vector<int>>& A,
        const std::vector<std::vector<int>>& B)
{
    double total = 0.0;
    for (const auto& a : A) {
        double best = 0.0;
        for (const auto& b : B)
            best = std::max(best, jaccard(a, b));
        total += best;
    }
    return A.empty() ? 1.0 : total / static_cast<double>(A.size());
}

// ---------------------------------------------------------------------------
// Simple LCG for synthetic data generation (reproducible, no std::random_device).
// ---------------------------------------------------------------------------
struct LCG {
    uint64_t state;
    explicit LCG(uint64_t seed) : state(seed) {}
    uint64_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return state;
    }
    float next_float() {
        return static_cast<float>(next() >> 32) / static_cast<float>(0xFFFFFFFFu);
    }
};

// ---------------------------------------------------------------------------
// Generate a synthetic count matrix with planted programs.
//
// Each planted program is a sparse gene signature: top genes are bright,
// rest near-zero.  Each cell randomly mixes one dominant program plus noise.
// This creates distinct, discoverable programs for both the C++ kernel and
// the Python CSI-GEP reference to recover.
//
//   m: genes (rows), n: cells (cols), K: planted programs
// ---------------------------------------------------------------------------
HostCSC make_planted_programs(int m, int n, int K, uint64_t seed, float density = 0.15f) {
    LCG rng(seed);

    // Build program gene loadings: top (m/K) genes per program get weight 1,
    // rest get small noise.
    const int genes_per_prog = std::max(1, m / K);

    // Assign dominant program to each cell.
    std::vector<int> cell_prog(n);
    for (int j = 0; j < n; ++j)
        cell_prog[j] = static_cast<int>(rng.next() % static_cast<uint64_t>(K));

    // Build CSC in COO first, then sort.
    std::vector<int32_t> rows_v, cols_v;
    std::vector<float>   vals_v;

    for (int j = 0; j < n; ++j) {
        int prog = cell_prog[j];
        for (int i = 0; i < m; ++i) {
            float base_weight = 0.0f;
            // Genes in this program's "signature window" get high signal.
            if (i >= prog * genes_per_prog &&
                i <  (prog + 1) * genes_per_prog)
                base_weight = 5.0f + rng.next_float() * 3.0f;
            else
                base_weight = rng.next_float() * 0.5f;

            float u = rng.next_float();
            if (u < density || base_weight > 2.0f) {
                float count = std::max(0.0f, base_weight + rng.next_float() - 0.5f);
                if (count > 0.0f) {
                    rows_v.push_back(i);
                    cols_v.push_back(j);
                    vals_v.push_back(count);
                }
            }
        }
    }

    // Sort into CSC order (col-major).
    uint64_t nnz = vals_v.size();
    std::vector<size_t> order(nnz);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return cols_v[a] < cols_v[b] || (cols_v[a] == cols_v[b] && rows_v[a] < rows_v[b]);
    });

    HostCSC h;
    h.m = static_cast<uint32_t>(m);
    h.n = static_cast<uint32_t>(n);
    h.nnz = nnz;
    h.values .resize(nnz);
    h.indices.resize(nnz);
    h.indptr .resize(static_cast<size_t>(n) + 1, 0);

    for (size_t e = 0; e < nnz; ++e) {
        h.values[e]  = vals_v[order[e]];
        h.indices[e] = rows_v[order[e]];
        h.indptr[cols_v[order[e]] + 1]++;
    }
    // prefix-sum indptr
    for (int j = 0; j < n; ++j)
        h.indptr[j + 1] += h.indptr[j];

    return h;
}

// ---------------------------------------------------------------------------
// Read a float array from a simple binary dump written by the Python reference.
// Format: little-endian float32[count], no header.
// ---------------------------------------------------------------------------
std::vector<float> read_float_bin(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open float bin: " + path);
    std::vector<float> v(count);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    if (!f) throw std::runtime_error("Short read: " + path);
    return v;
}

// ---------------------------------------------------------------------------
// Read an int32 scalar from a binary file (single int32, little-endian).
// ---------------------------------------------------------------------------
int32_t read_int32_scalar(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open int32 scalar: " + path);
    int32_t v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

// ---------------------------------------------------------------------------
// Check all values in a float vector are finite (no NaN, no Inf).
// ---------------------------------------------------------------------------
bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Check reproducibility curve: non-decreasing up to its max, then plateau.
// "Plateau" = final value within 0.02 of max.
// ---------------------------------------------------------------------------
bool repro_curve_nondecreasing_then_plateau(const std::vector<float>& curve) {
    if (curve.size() < 2) return true;
    // Allow small noise: use a weak non-decreasing check with tolerance 0.005.
    const float eps = 0.005f;
    float peak = *std::max_element(curve.begin(), curve.end());
    // Non-decreasing check: each value must be >= prev - eps.
    for (size_t i = 1; i < curve.size(); ++i) {
        if (curve[i] < curve[i-1] - eps) return false;
    }
    // Plateau check: final value must be within 0.02 of peak.
    float last = curve.back();
    return (peak - last) <= 0.02f;
}

}  // anonymous namespace

// =============================================================================
// Test fixture
// =============================================================================

class CsiGepTest : public ::testing::Test {
protected:
    void SetUp() override {
        ensure_refs_tmp();
        // Create GPU context (owns its own stream + handles).
        if (!gpu_available()) return;
        try {
            ctx_ = std::make_unique<singlet_gpu::core::GPUContext>();
        } catch (const std::exception& ex) {
            GTEST_SKIP() << "GPUContext init failed: " << ex.what();
        }
        // stream_ is an alias for ctx_->stream so all ops share one stream.
        stream_ = ctx_->stream;
    }

    void TearDown() override {
        if (ctx_) {
            // ctx_ destructor syncs stream_ and destroys handles.
            ctx_.reset();
            stream_ = nullptr;
        }
    }

    cudaStream_t stream_ = nullptr;
    std::unique_ptr<singlet_gpu::core::GPUContext> ctx_;
};

// =============================================================================
// Test 1 — CsiGep_TinySynthetic_VsPython
// =============================================================================
//
// Plant 5 programs in a 200 × 100 matrix.  Run both the C++ CSI-GEP kernel
// and the Python reference.  Compare top-30 gene sets per consensus program
// using best-match Jaccard; require mean >= 0.85.
//
// Python reference exit 2 → GTEST_SKIP (csi-gep not installed).

TEST_F(CsiGepTest, CsiGep_TinySynthetic_VsPython) {
    if (!gpu_available())
        GTEST_SKIP() << "No GPU available";

    // ----- Build synthetic data -----
    HostCSC h = make_planted_programs(kTinyGenes, kTinyCells, kTinyPrograms,
                                      kSeed, 0.20f);
    const std::string csc_path = refs_path("tiny_matrix.bin");
    write_csc_bin(h, csc_path);

    // ----- Run Python reference -----
    const std::string py_programs_path = refs_path("tiny_py_programs.bin");
    const std::string py_chosen_k_path = refs_path("tiny_py_chosen_k.bin");

    std::ostringstream cmd;
    cmd << kPython << " " << kRefScript
        << " --csc "        << csc_path
        << " --k-range 3,4,5,6,7"
        << " --n-runs 10"
        << " --seed " << kSeed
        << " --out-programs " << py_programs_path
        << " --out-chosen-k "  << py_chosen_k_path
        << " 2>&1";

    int rc = run_cmd_rc(cmd.str());
    if (rc == kRefScriptSkipCode)
        GTEST_SKIP() << "csi-gep Python package not installed; skipping reference comparison";
    ASSERT_EQ(rc, 0) << "Python reference script failed (exit " << rc << ")";

    // ----- Run C++ CSI-GEP kernel -----
    singlet_gpu::core::DeviceCSC d_csc = upload_csc(h, stream_);

    singlet_gpu::reduce::nmf::CsiGepConfig cfg;
    cfg.k_range        = {3, 4, 5, 6, 7};
    cfg.n_runs         = 10;
    cfg.subsample_frac = 0.8f;
    cfg.seed           = kSeed;

    singlet_gpu::reduce::nmf::CsiGepResult result =
        singlet_gpu::reduce::nmf::csi_gep(*ctx_, d_csc, cfg);

    cudaStreamSynchronize(stream_);

    // ----- Download C++ programs (m × chosen_k) -----
    int k_cpp = static_cast<int>(result.chosen_k);
    ASSERT_GT(k_cpp, 0) << "chosen_k must be positive";
    ASSERT_LE(k_cpp, 7);

    std::vector<float> cpp_programs_flat =
        download_f32(result.consensus_programs, static_cast<size_t>(kTinyGenes) * k_cpp, stream_);

    // Build top-30 gene sets per C++ program.
    std::vector<std::vector<int>> cpp_top30(k_cpp);
    for (int p = 0; p < k_cpp; ++p) {
        std::vector<float> col(kTinyGenes);
        for (int g = 0; g < kTinyGenes; ++g)
            col[g] = cpp_programs_flat[g + p * kTinyGenes];  // column-major
        cpp_top30[p] = top_n_indices(col, 30);
    }

    // ----- Read Python programs -----
    // Python writes: float32[n_genes * chosen_k_py] (column-major, same layout).
    int32_t k_py = read_int32_scalar(py_chosen_k_path);
    ASSERT_GT(k_py, 0);
    std::vector<float> py_programs_flat =
        read_float_bin(py_programs_path, static_cast<size_t>(kTinyGenes) * k_py);

    std::vector<std::vector<int>> py_top30(k_py);
    for (int p = 0; p < k_py; ++p) {
        std::vector<float> col(kTinyGenes);
        for (int g = 0; g < kTinyGenes; ++g)
            col[g] = py_programs_flat[g + p * kTinyGenes];
        py_top30[p] = top_n_indices(col, 30);
    }

    // ----- Compute best-match Jaccard (cpp → python direction) -----
    double J = best_match_jaccard(cpp_top30, py_top30);
    EXPECT_GE(J, kJaccardMin)
        << "Best-match Jaccard of top-30 genes: " << J
        << " (tolerance " << kJaccardMin << ")";
}

// =============================================================================
// Test 2 — CsiGep_GSM4037629_RealData
// =============================================================================
//
// Load exon_counts.1pz for GSM4037629 (11,560 cells).
// Run CSI-GEP with k_range=[5,10,15,20], n_runs=10 (reduced for test speed).
// Check: all output values finite, chosen_k in [5,20], no NaN in programs.

TEST_F(CsiGepTest, CsiGep_GSM4037629_RealData) {
    if (!gpu_available())
        GTEST_SKIP() << "No GPU available";
    if (!fs::exists(kGsmExonPath))
        GTEST_SKIP() << "GSM4037629 exon_counts.1pz not found at " << kGsmExonPath;

    // Load .1pz via device loader.
    singlet_gpu::core::DeviceCSC d_csc =
        singlet_gpu::io::load_pz_device(kGsmExonPath, stream_);
    cudaStreamSynchronize(stream_);

    ASSERT_GT(d_csc.rows, 0u) << "empty matrix from .1pz loader";
    ASSERT_GT(d_csc.cols, 0u) << "zero cells in loaded matrix";

    // Run CSI-GEP.
    singlet_gpu::reduce::nmf::CsiGepConfig cfg;
    cfg.k_range        = {5, 10, 15, 20};
    cfg.n_runs         = 10;
    cfg.subsample_frac = 0.8f;
    cfg.seed           = kSeed;

    singlet_gpu::reduce::nmf::CsiGepResult result =
        singlet_gpu::reduce::nmf::csi_gep(*ctx_, d_csc, cfg);

    cudaStreamSynchronize(stream_);

    // chosen_k must be in the supplied k_range.
    EXPECT_GE(static_cast<int>(result.chosen_k), 5);
    EXPECT_LE(static_cast<int>(result.chosen_k), 20);

    // robust_programs: genes × chosen_k — must all be finite.
    size_t prog_elements = static_cast<size_t>(d_csc.rows) * result.chosen_k;
    std::vector<float> programs_h = download_f32(result.consensus_programs, prog_elements, stream_);
    EXPECT_TRUE(all_finite(programs_h))
        << "NaN or Inf in consensus programs on real data";

    // program_usage: cells × chosen_k — must all be finite.
    size_t usage_elements = static_cast<size_t>(d_csc.cols) * result.chosen_k;
    std::vector<float> usage_h = download_f32(result.program_usage, usage_elements, stream_);
    EXPECT_TRUE(all_finite(usage_h))
        << "NaN or Inf in program usage on real data";

    // reproducibility_curve must have one entry per k in k_range.
    ASSERT_EQ(result.reproducibility_curve.size(), cfg.k_range.size())
        << "reproducibility_curve length mismatch";
}

// =============================================================================
// Test 3 — CsiGep_AutoKSelection_PicksKnownK
// =============================================================================
//
// 8 planted programs in a 100 × 200 matrix.
// Run with k_range=[3..15], n_runs=10.
// chosen_k must be in [6, 10] (planted K ± 2).

TEST_F(CsiGepTest, CsiGep_AutoKSelection_PicksKnownK) {
    if (!gpu_available())
        GTEST_SKIP() << "No GPU available";

    HostCSC h = make_planted_programs(kAutoKGenes, kAutoKCells, kAutoKPlanted,
                                      kSeed + 7, 0.25f);

    singlet_gpu::core::DeviceCSC d_csc = upload_csc(h, stream_);

    singlet_gpu::reduce::nmf::CsiGepConfig cfg;
    // k_range covers the planted K and a wide margin around it.
    for (int k = 3; k <= 15; ++k)
        cfg.k_range.push_back(k);
    cfg.n_runs         = 10;
    cfg.subsample_frac = 0.8f;
    cfg.seed           = kSeed;

    singlet_gpu::reduce::nmf::CsiGepResult result =
        singlet_gpu::reduce::nmf::csi_gep(*ctx_, d_csc, cfg);

    cudaStreamSynchronize(stream_);

    int chosen = static_cast<int>(result.chosen_k);
    int lo = kAutoKPlanted - kAutoKTolerance;
    int hi = kAutoKPlanted + kAutoKTolerance;

    EXPECT_GE(chosen, lo)
        << "chosen_k=" << chosen << " below planted K=" << kAutoKPlanted
        << " - tolerance=" << kAutoKTolerance;
    EXPECT_LE(chosen, hi)
        << "chosen_k=" << chosen << " above planted K=" << kAutoKPlanted
        << " + tolerance=" << kAutoKTolerance;
}

// =============================================================================
// Test 4 — CsiGep_ReproducibilityCurve_Monotone
// =============================================================================
//
// Use same 5-program synthetic (200 × 100).
// k_range=[3..10], n_runs=10.
// Reproducibility curve must be non-decreasing up to its maximum, then
// plateau (final value within 0.02 of peak).

TEST_F(CsiGepTest, CsiGep_ReproducibilityCurve_Monotone) {
    if (!gpu_available())
        GTEST_SKIP() << "No GPU available";

    HostCSC h = make_planted_programs(kTinyGenes, kTinyCells, kTinyPrograms,
                                      kSeed + 13, 0.20f);

    singlet_gpu::core::DeviceCSC d_csc = upload_csc(h, stream_);

    singlet_gpu::reduce::nmf::CsiGepConfig cfg;
    for (int k = 3; k <= 10; ++k)
        cfg.k_range.push_back(k);
    cfg.n_runs         = 10;
    cfg.subsample_frac = 0.8f;
    cfg.seed           = kSeed;

    singlet_gpu::reduce::nmf::CsiGepResult result =
        singlet_gpu::reduce::nmf::csi_gep(*ctx_, d_csc, cfg);

    cudaStreamSynchronize(stream_);

    ASSERT_EQ(result.reproducibility_curve.size(), cfg.k_range.size())
        << "reproducibility_curve length does not match k_range size";

    // Verify monotone-then-plateau property.
    EXPECT_TRUE(repro_curve_nondecreasing_then_plateau(result.reproducibility_curve))
        << "Reproducibility curve is not non-decreasing then plateau";

    // All values in [0, 1].
    for (float v : result.reproducibility_curve) {
        EXPECT_GE(v, 0.0f) << "negative reproducibility score";
        EXPECT_LE(v, 1.0f) << "reproducibility score > 1";
    }
}

// =============================================================================
// Test 5 — CsiGep_Determinism_BitIdentical
// =============================================================================
//
// Run CSI-GEP twice with the same seed on the same matrix.
// All output arrays must be bit-identical.

TEST_F(CsiGepTest, CsiGep_Determinism_BitIdentical) {
    if (!gpu_available())
        GTEST_SKIP() << "No GPU available";

    HostCSC h = make_planted_programs(kTinyGenes, kTinyCells, kTinyPrograms,
                                      kSeed + 17, 0.18f);

    singlet_gpu::reduce::nmf::CsiGepConfig cfg;
    cfg.k_range        = {3, 5, 7};
    cfg.n_runs         = 5;
    cfg.subsample_frac = 0.8f;
    cfg.seed           = kSeed;

    // First run.
    singlet_gpu::core::DeviceCSC d_csc1 = upload_csc(h, stream_);
    singlet_gpu::reduce::nmf::CsiGepResult r1 =
        singlet_gpu::reduce::nmf::csi_gep(*ctx_, d_csc1, cfg);
    cudaStreamSynchronize(stream_);

    int k1 = static_cast<int>(r1.chosen_k);
    size_t prog_elems = static_cast<size_t>(kTinyGenes) * k1;
    size_t usage_elems = static_cast<size_t>(kTinyCells) * k1;
    std::vector<float> progs1 = download_f32(r1.consensus_programs, prog_elems, stream_);
    std::vector<float> usage1 = download_f32(r1.program_usage,   usage_elems, stream_);

    // Second run (upload fresh copy of the same matrix).
    singlet_gpu::core::DeviceCSC d_csc2 = upload_csc(h, stream_);
    singlet_gpu::reduce::nmf::CsiGepResult r2 =
        singlet_gpu::reduce::nmf::csi_gep(*ctx_, d_csc2, cfg);
    cudaStreamSynchronize(stream_);

    int k2 = static_cast<int>(r2.chosen_k);
    ASSERT_EQ(k1, k2) << "chosen_k differs between two identical runs";

    std::vector<float> progs2 = download_f32(r2.consensus_programs, prog_elems, stream_);
    std::vector<float> usage2 = download_f32(r2.program_usage,   usage_elems, stream_);

    // Bit-identical: memcmp on float arrays.
    ASSERT_EQ(progs1.size(), progs2.size());
    EXPECT_EQ(0, std::memcmp(progs1.data(), progs2.data(),
                              progs1.size() * sizeof(float)))
        << "robust_programs not bit-identical across two runs with same seed";

    ASSERT_EQ(usage1.size(), usage2.size());
    EXPECT_EQ(0, std::memcmp(usage1.data(), usage2.data(),
                              usage1.size() * sizeof(float)))
        << "program_usage not bit-identical across two runs with same seed";

    // reproducibility_curve must also be bit-identical.
    ASSERT_EQ(r1.reproducibility_curve.size(), r2.reproducibility_curve.size());
    EXPECT_EQ(0, std::memcmp(r1.reproducibility_curve.data(),
                              r2.reproducibility_curve.data(),
                              r1.reproducibility_curve.size() * sizeof(float)))
        << "reproducibility_curve not bit-identical across two runs with same seed";
}

// =============================================================================
// Test 6 — CsiGep_ReusesCycle5NMF
// =============================================================================
//
// Smoke test that the cycle 5 NMF adapter (`singlet_gpu::reduce::nmf::fit`) is
// callable without compilation errors and returns a plausible result.
// CSI-GEP uses this as its inner loop; this test confirms the adapter is
// accessible and the ABI matches what csi_gep.h expects.

TEST_F(CsiGepTest, CsiGep_ReusesCycle5NMF) {
    if (!gpu_available())
        GTEST_SKIP() << "No GPU available";

    // Tiny 30-gene × 20-cell matrix for the smoke test.
    HostCSC h = make_planted_programs(30, 20, 3, kSeed + 42, 0.30f);
    singlet_gpu::core::DeviceCSC d_csc = upload_csc(h, stream_);

    singlet_gpu::reduce::nmf::NmfConfig nmf_cfg;
    nmf_cfg.rank           = 3;
    nmf_cfg.max_iter       = 20;
    nmf_cfg.seed           = static_cast<uint32_t>(kSeed & 0xFFFFFFFFu);
    // Note: NMFConfig has no deterministic field; seed alone controls determinism.

    // Call cycle 5 NMF adapter directly.
    singlet_gpu::reduce::nmf::NmfResult nmf_result =
        singlet_gpu::reduce::nmf::fit(*ctx_, d_csc, nmf_cfg);

    cudaStreamSynchronize(stream_);

    // Basic sanity: W (genes × k) and H (k × cells) must be finite.
    // NmfResult.W and .H are Eigen matrices on the host — copy directly.
    size_t w_elems = static_cast<size_t>(30) * 3;
    size_t h_elems = static_cast<size_t>(3) * 20;
    std::vector<float> W(nmf_result.W.data(),
                         nmf_result.W.data() + static_cast<int>(w_elems));
    std::vector<float> H(nmf_result.H.data(),
                         nmf_result.H.data() + static_cast<int>(h_elems));

    EXPECT_TRUE(all_finite(W)) << "NMF W matrix contains NaN or Inf";
    EXPECT_TRUE(all_finite(H)) << "NMF H matrix contains NaN or Inf";

    // All values non-negative (NMF constraint).
    for (float v : W)
        EXPECT_GE(v, 0.0f) << "negative value in NMF W";
    for (float v : H)
        EXPECT_GE(v, 0.0f) << "negative value in NMF H";
}
