// SPDX-License-Identifier: MIT
// singlet/gpu/tests/spatial_flash_deconv_correctness.cpp
//
// integrates: original (first GPU FlashDeconv with leverage-score sketching + ADMM)
//
// Correctness harness for singlet::gpu::spatial::flash_deconv.
//
// Written against the design doc at:
//   singlet/gpu/state/designs/33-flash-deconv.md
// NOT against the kernel source — authored in parallel with
// spatial/flash_deconv.h by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: cell2location Python via subprocess
//   tests/refs/flash_deconv_cell2location_reference.py
//   (pip install cell2location pymc scvi-tools anndata numpy scipy)
//   Falls back to scipy NNLS if cell2location is not installed.
//   Exit code 2 = all packages unavailable → GTEST_SKIP.
//
// Tolerances (design doc §"Correctness test spec"):
//   Spearman ρ >= 0.90   per-spot cell-type abundance vs cell2location (or scipy NNLS)
//   sum-to-1 constraint: |row_sum - 1| <= 1e-4 per spot
//   Non-negative W: min(W) >= -1e-6
//   Finite results: no NaN / Inf
//   Bootstrap uncertainty: uncertainty > 0, max(uncertainty) <= 1.0
//   Determinism: bit-identical W with fixed seed across two calls
//   Spatial regularization: neighboring spots have correlated abundances
//   Leverage-score sketching: top-K genes retained by recall check
//
// Test cases:
//   FlashDeconv_TinySynthetic_VsCell2Location
//   FlashDeconv_GSM4037629_RealData
//   FlashDeconv_LeverageScoreSketching_RecallTopGenes
//   FlashDeconv_BootstrapUncertainty_NonZero
//   FlashDeconv_Determinism_BitIdentical
//   FlashDeconv_SpatialRegularization_SmoothsAbundance
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, CUDA::cublas, CUDA::cusolver,
//             CUDA::curand, CUDA::cusparse, singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every device test is guarded by gpu_available()
//   and calls GTEST_SKIP().

// ---- singlet-gpu flash_deconv header (written by gpu-kernel-dev) ---------------
#include <singlet/gpu/spatial/flash_deconv.h>

// ---- other singlet-gpu headers used by real-data / upload tests ---------------
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

// ---- test helpers -------------------------------------------------------------
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
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

// Tiny synthetic: 200 spots × 100 genes × 5 cell types (design doc spec).
constexpr int kTinySpots  = 200;
constexpr int kTinyGenes  = 100;
constexpr int kTinyTypes  = 5;

// Leverage-score sketching test: sketch retains top-K out of N genes.
constexpr int kSketchSpots = 100;
constexpr int kSketchGenes = 200;
constexpr int kSketchTypes = 4;
constexpr int kSketchK     = 50;    // default sketch size (design doc: K=500 on real data)

// Bootstrap uncertainty test.
constexpr int kBootSpots  = 80;
constexpr int kBootGenes  = 60;
constexpr int kBootTypes  = 3;
constexpr int kBootSamples = 20;   // design doc: K=20 bootstrap iterations

// Spatial regularization test: 12×12 grid of spots (144 total).
constexpr int kSpatRows  = 12;
constexpr int kSpatCols  = 12;
constexpr int kSpatSpots = kSpatRows * kSpatCols;
constexpr int kSpatGenes = 80;
constexpr int kSpatTypes = 3;

// Correctness tolerance.
constexpr double kSpearmanMinRho  = 0.90;
constexpr double kSumToOneTol     = 1e-4;
constexpr double kNonNegTol       = -1e-6;
constexpr double kMaxUncertainty  = 1.0;
constexpr double kSpatCorrMin     = 0.30;  // relaxed: spatial regularization smoother

// Reference script.
#ifndef SINGLET_GPU_TEST_DIR
#define SINGLET_GPU_TEST_DIR "/mnt/home/debruinz/Singlet-AI/singlet/gpu/tests"
#endif

static const char* kRefScript =
    SINGLET_GPU_TEST_DIR "/refs/flash_deconv_cell2location_reference.py";

static const char* kRefsTmpDir = "/tmp/singlet_gpu_flash_deconv_refs_tmp";
static const char* kPython     = "python3";
static constexpr int kRefSkipCode = 2;

// GSM4037629 scRNA counts (used as a stand-in for a Visium matrix in the real-data
// test since the pipeline's canonical test sample is scRNA; the test only validates
// finite results + sum-to-1, not biology).
static const char* kGSM4037629CountPath =
    "/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna/"
    "GSE127/GSE127918/GSM4037629/exon_counts.1pz";

// =============================================================================
// Utility helpers (pattern from cycles 3-32)
// =============================================================================

namespace {

void ensure_refs_tmp() { fs::create_directories(kRefsTmpDir); }

std::string refs_path(const std::string& name) {
    return std::string(kRefsTmpDir) + "/" + name;
}

// run_cmd: throws on non-zero exit.
void run_cmd(const std::string& cmd) {
    int ret = std::system(cmd.c_str());
    if (ret != 0)
        throw std::runtime_error("Command failed (exit " +
                                 std::to_string(ret) + "): " + cmd);
}

// run_cmd_rc: returns exit code without throwing.
int run_cmd_rc(const std::string& cmd) { return std::system(cmd.c_str()); }

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
    uint32_t             n;       // spots (cols)
    uint64_t             nnz;
    std::vector<float>   values;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
};

// Write HostCSC in dump_csc.h "CSCC" binary format.
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

// Write reference mean matrix in "REFM" binary format.
// Format: [uint32 magic=REFM][uint32 n_types][uint32 n_genes][float32 * n_types * n_genes]
void write_refm_bin(const std::vector<float>& B,
                    uint32_t n_types, uint32_t n_genes,
                    const std::string& path) {
    static constexpr uint32_t kMagic = 0x5245464Du;  // "REFM"
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.write(reinterpret_cast<const char*>(&kMagic),   4);
    f.write(reinterpret_cast<const char*>(&n_types),  4);
    f.write(reinterpret_cast<const char*>(&n_genes),  4);
    f.write(reinterpret_cast<const char*>(B.data()),
            static_cast<std::streamsize>(n_types * n_genes * sizeof(float)));
    if (!f) throw std::runtime_error("Write error: " + path);
}

// Read dense float32 matrix written by Python's write_dense_mat.
// Format: [uint32 magic=DMT\0][uint32 rows][uint32 cols][float32 * rows * cols]
struct DenseMat {
    uint32_t             rows;
    uint32_t             cols;
    std::vector<float>   data;  // row-major
};

DenseMat read_dense_npz_W(const std::string& npz_path) {
    // The Python reference writes a .npz; parse via a small Python one-liner.
    std::string tmp_bin = npz_path + ".W.bin";
    static constexpr uint32_t kMagic = 0x444D5400u;  // "DMT\0"
    std::ostringstream cmd;
    cmd << kPython << " -c \""
        << "import numpy as np, struct, sys; "
        << "d = np.load('" << npz_path << "'); "
        << "W = d['W'].astype(np.float32); "
        << "rows, cols = W.shape; "
        << "with open('" << tmp_bin << "', 'wb') as f: "
        << "  f.write(struct.pack('<I', 0x444D5400)); "
        << "  f.write(struct.pack('<II', rows, cols)); "
        << "  f.write(W.tobytes()); "
        << "sys.exit(0)"
        << "\"";
    run_cmd(cmd.str());

    std::ifstream f(tmp_bin, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + tmp_bin);
    uint32_t magic, rows, cols;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&rows),  4);
    f.read(reinterpret_cast<char*>(&cols),  4);
    if (magic != kMagic)
        throw std::runtime_error("read_dense_npz_W: bad magic");
    DenseMat dm;
    dm.rows = rows;
    dm.cols = cols;
    dm.data.resize(static_cast<size_t>(rows) * cols);
    f.read(reinterpret_cast<char*>(dm.data.data()),
           static_cast<std::streamsize>(dm.data.size() * sizeof(float)));
    return dm;
}

// ---------------------------------------------------------------------------
// Upload a HostCSC to device (reused from STAGATE cycle 29).
// ---------------------------------------------------------------------------
singlet::gpu::core::DeviceCSC upload_csc(const HostCSC& h, cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    return singlet::gpu::core::DeviceCSC(
        static_cast<int>(h.m), static_cast<int>(h.n), static_cast<int>(h.nnz),
        h.indptr.data(), h.indices.data(), h.values.data());
}

// ---------------------------------------------------------------------------
// Spearman rank correlation (host-side, from cycles 11-32).
// ---------------------------------------------------------------------------
double spearman_rho(const std::vector<double>& x,
                    const std::vector<double>& y) {
    assert(x.size() == y.size());
    int n = static_cast<int>(x.size());
    if (n < 2) return 0.0;

    auto rank_of = [&](const std::vector<double>& v) {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return v[a] < v[b]; });
        std::vector<double> r(n);
        for (int i = 0; i < n; ) {
            int j = i;
            while (j < n && v[idx[j]] == v[idx[i]]) ++j;
            double avg = 0.5 * (i + j - 1);
            for (int k = i; k < j; ++k) r[idx[k]] = avg;
            i = j;
        }
        return r;
    };

    auto rx = rank_of(x);
    auto ry = rank_of(y);
    double d2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = rx[i] - ry[i];
        d2 += d * d;
    }
    return 1.0 - 6.0 * d2 / (static_cast<double>(n) *
                               (static_cast<double>(n) * n - 1.0));
}

// ---------------------------------------------------------------------------
// Synthetic data generators.
// ---------------------------------------------------------------------------

// Build a tiny synthetic spatial deconvolution dataset:
//   B  (n_types × n_genes)  — reference mean profiles (log1p-normalised)
//   Y  (n_spots × n_genes)  — spatial expression = W_true @ B + noise
//   W_true (n_spots × n_types) — ground truth proportions (simplex rows)
// Returns HostCSC for Y (genes × spots, CSC layout).
struct SyntheticDeconv {
    HostCSC            Y_csc;       // genes × spots
    std::vector<float> B;           // n_types × n_genes, row-major
    std::vector<float> W_true;      // n_spots × n_types, row-major
    uint32_t           n_types;
    uint32_t           n_genes;
    uint32_t           n_spots;
};

SyntheticDeconv make_synthetic(uint32_t n_spots, uint32_t n_genes,
                               uint32_t n_types, uint64_t seed,
                               float noise_level = 0.1f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::normal_distribution<float>       gauss(0.0f, 1.0f);

    // B: reference expression profiles.  Each type has a distinct "marker gene"
    // cluster to make the problem informative.
    std::vector<float> B(static_cast<size_t>(n_types) * n_genes, 0.0f);
    for (uint32_t t = 0; t < n_types; ++t) {
        // Assign n_genes/n_types marker genes to each type.
        uint32_t genes_per_type = std::max(1u, n_genes / n_types);
        uint32_t start_gene = t * genes_per_type;
        uint32_t end_gene   = std::min(n_genes, start_gene + genes_per_type);
        for (uint32_t g = 0; g < n_genes; ++g) {
            float base = (g >= start_gene && g < end_gene) ? 3.0f : 0.3f;
            B[t * n_genes + g] = base * (0.5f + 0.5f * uni(rng));
        }
    }

    // W_true: simplex-constrained proportions per spot.
    std::vector<float> W_true(static_cast<size_t>(n_spots) * n_types, 0.0f);
    for (uint32_t s = 0; s < n_spots; ++s) {
        float row_sum = 0.0f;
        for (uint32_t t = 0; t < n_types; ++t) {
            float v = std::max(0.0f, gauss(rng) + 1.0f);
            W_true[s * n_types + t] = v;
            row_sum += v;
        }
        if (row_sum < 1e-12f) row_sum = 1.0f;
        for (uint32_t t = 0; t < n_types; ++t)
            W_true[s * n_types + t] /= row_sum;
    }

    // Y: sparse gene × spot matrix from W @ B + noise (drop low-value entries).
    // We construct in dense col-major (spot = col), then convert to CSC.
    std::vector<std::vector<std::pair<int32_t, float>>> cols(n_spots);
    uint64_t total_nnz = 0;
    for (uint32_t s = 0; s < n_spots; ++s) {
        for (uint32_t g = 0; g < n_genes; ++g) {
            float v = 0.0f;
            for (uint32_t t = 0; t < n_types; ++t)
                v += W_true[s * n_types + t] * B[t * n_genes + g];
            v += noise_level * std::max(0.0f, gauss(rng));
            // Simulate count data: round and keep if >= threshold.
            float cnt = std::round(v * 10.0f);
            if (cnt > 0.5f) {
                cols[s].push_back({static_cast<int32_t>(g), cnt});
                ++total_nnz;
            }
        }
    }

    HostCSC csc;
    csc.m   = n_genes;
    csc.n   = n_spots;
    csc.nnz = total_nnz;
    csc.values .reserve(total_nnz);
    csc.indices.reserve(total_nnz);
    csc.indptr .resize(static_cast<size_t>(n_spots) + 1, 0);
    for (uint32_t s = 0; s < n_spots; ++s) {
        csc.indptr[s + 1] = csc.indptr[s] +
                            static_cast<int32_t>(cols[s].size());
        for (auto& [g, v] : cols[s]) {
            csc.indices.push_back(g);
            csc.values .push_back(v);
        }
    }

    SyntheticDeconv out;
    out.Y_csc   = std::move(csc);
    out.B       = std::move(B);
    out.W_true  = std::move(W_true);
    out.n_types = n_types;
    out.n_genes = n_genes;
    out.n_spots = n_spots;
    return out;
}

// Build 2-D grid spatial coordinates (row-major raster).
std::vector<float> make_grid_coords(int rows, int cols) {
    std::vector<float> xy;
    xy.reserve(static_cast<size_t>(rows * cols * 2));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            xy.push_back(static_cast<float>(c));  // x
            xy.push_back(static_cast<float>(r));  // y
        }
    return xy;
}

// Pearson correlation between two float vectors (host-side).
double pearson(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    int n = static_cast<int>(a.size());
    double ma = 0.0, mb = 0.0;
    for (int i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double num = 0.0, da = 0.0, db = 0.0;
    for (int i = 0; i < n; ++i) {
        double ea = a[i] - ma, eb = b[i] - mb;
        num += ea * eb;
        da  += ea * ea;
        db  += eb * eb;
    }
    if (da < 1e-15 || db < 1e-15) return 0.0;
    return num / std::sqrt(da * db);
}

}  // anonymous namespace

// =============================================================================
// PzDeviceLoader compat bridge (io::load_pz is the real API)
// =============================================================================
namespace singlet::gpu { namespace io {
struct PzDeviceLoader {
    std::string path_;
    explicit PzDeviceLoader(const std::string& p) : path_(p) {}
    core::DeviceCSC load_csc(cudaStream_t stream) const {
        return singlet::gpu::io::load_pz(path_, stream).mat;
    }
};
} }

// =============================================================================
// FlashDeconv result types (matching flash_deconv.h public API)
// =============================================================================

// Compat: the test was designed against a planned API that differs from the actual
// flash_deconv.h API.  We provide compat types and a bridge function.

// The actual FlashDeconvConfig (from flash_deconv.h) has different field names.
// The actual FlashDeconvResult has device-side fields, not host-side.
// Provide compat types and a bridge at GLOBAL scope.

// FlashDeconvConfigCompat: test-side config with old field names.
namespace singlet::gpu { namespace spatial {

struct FlashDeconvConfigCompat {
    int      sketch_k          = 500;
    int      admm_max_iter     = 50;
    float    admm_tol          = 1e-4f;
    float    admm_lambda       = 1e-2f;
    float    spatial_lambda    = 0.0f;
    int      bootstrap_samples = 0;
    uint64_t seed              = 0xC0FFEEull;
    int      spatial_knn_k    = 6;
};

// FlashDeconvResultCompat: host-side result as expected by test.
struct FlashDeconvResultCompat {
    std::vector<float>   W;
    std::vector<float>   uncertainty;
    std::vector<int32_t> top_gene_indices;
    uint32_t n_spots = 0;
    uint32_t n_types = 0;
    uint32_t n_genes = 0;
};

// Bridge: converts FlashDeconvConfigCompat → FlashDeconvConfig and builds
// fake reference_counts/reference_cell_types from the B matrix.
inline FlashDeconvResultCompat flash_deconv_fit(
    const core::DeviceCSC&             Y_csc,
    const core::DeviceMemory<float>&   B_device,   // n_types × n_genes, row-major
    const core::DeviceMemory<float>*   coords_xy,
    uint32_t                           n_types,
    uint32_t                           n_genes,
    const FlashDeconvConfigCompat&     cfg,
    cudaStream_t                       stream)
{
    const int n_spots = (int)Y_csc.cols;

    // Map compat config to actual config.
    FlashDeconvConfig real_cfg;
    real_cfg.sketch_size   = cfg.sketch_k;
    real_cfg.max_admm_iter = cfg.admm_max_iter;
    real_cfg.admm_tol      = cfg.admm_tol;
    real_cfg.l1_weight     = cfg.admm_lambda;
    real_cfg.spatial_lambda = cfg.spatial_lambda;
    real_cfg.n_bootstrap   = cfg.bootstrap_samples > 0 ? cfg.bootstrap_samples : 0;
    real_cfg.seed          = cfg.seed;

    // Build a fake reference_counts CSC from the B matrix (n_types × n_genes, row-major).
    // We create n_types synthetic "reference cells" (one per type), one cell per type.
    // reference_cell_types = [0, 1, 2, ..., n_types-1].
    // reference_counts CSC: n_genes rows × n_types cols (one cell per type).
    int nt = (int)n_types;
    int ng = (int)n_genes;

    // Download B from device.
    std::vector<float> h_B((size_t)nt * ng);
    cudaMemcpy(h_B.data(), B_device.get(),
               h_B.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Build CSC for reference: rows = genes, cols = types.
    std::vector<int>   ref_indptr(nt + 1, 0);
    std::vector<int>   ref_indices;
    std::vector<float> ref_vals;
    for (int t = 0; t < nt; ++t) {
        for (int g = 0; g < ng; ++g) {
            float v = h_B[(size_t)t * ng + g];
            if (v > 0.0f) {
                ref_indices.push_back(g);
                ref_vals.push_back(v);
            }
        }
        ref_indptr[t + 1] = (int)ref_indices.size();
    }
    int ref_nnz = (int)ref_vals.size();
    core::DeviceCSC ref_csc(ng, nt, ref_nnz,
                            ref_indptr.data(), ref_indices.data(), ref_vals.data());

    // Build reference_cell_types: one cell per type.
    std::vector<int> h_ct(nt);
    for (int t = 0; t < nt; ++t) h_ct[t] = t;
    core::DeviceMemory<int> d_ct(nt);
    d_ct.upload(h_ct.data(), nt);

    // Call actual flash_deconv.
    FlashDeconvResult real_result = flash_deconv(
        Y_csc, ref_csc, d_ct, nt, coords_xy, real_cfg, stream);
    cudaStreamSynchronize(stream);

    // Download result.
    FlashDeconvResultCompat out;
    out.n_spots = (uint32_t)n_spots;
    out.n_types = (uint32_t)nt;
    out.n_genes = (uint32_t)ng;
    size_t W_sz = (size_t)n_spots * nt;
    out.W.resize(W_sz);
    cudaMemcpy(out.W.data(), real_result.abundance.get(),
               W_sz * sizeof(float), cudaMemcpyDeviceToHost);
    if (cfg.bootstrap_samples > 0) {
        out.uncertainty.resize(W_sz);
        cudaMemcpy(out.uncertainty.data(), real_result.uncertainty.get(),
                   W_sz * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return out;
}

} } // namespace singlet::gpu::spatial

// =============================================================================
// Test fixture
// =============================================================================

class FlashDeconvTest : public ::testing::Test {
protected:
    void SetUp() override {
        ensure_refs_tmp();
        if (!gpu_available()) {
            GTEST_SKIP() << "No CUDA device available — skipping FlashDeconv tests.";
        }
        cudaStreamCreate(&stream_);
    }
    void TearDown() override {
        if (gpu_available()) cudaStreamDestroy(stream_);
    }
    cudaStream_t stream_{};
};

// =============================================================================
// TC-1: FlashDeconv_TinySynthetic_VsCell2Location
// =============================================================================

TEST_F(FlashDeconvTest, FlashDeconv_TinySynthetic_VsCell2Location) {
    // Build synthetic dataset.
    auto synth = make_synthetic(kTinySpots, kTinyGenes, kTinyTypes, kSeed);

    // Write inputs for the Python reference.
    const std::string counts_path  = refs_path("fd_tiny_counts.bin");
    const std::string refm_path    = refs_path("fd_tiny_refm.bin");
    const std::string ref_out_path = refs_path("fd_tiny_ref_out.npz");

    write_csc_bin(synth.Y_csc, counts_path);
    write_refm_bin(synth.B, kTinyTypes, kTinyGenes, refm_path);

    // Run Python reference.
    std::ostringstream cmd;
    cmd << kPython << " " << kRefScript
        << " --counts "    << counts_path
        << " --ref-means " << refm_path
        << " --out "       << ref_out_path
        << " --n-epochs 300 --seed 0 2>/dev/null";

    int rc = run_cmd_rc(cmd.str());
    if (rc == kRefSkipCode) {
        GTEST_SKIP() << "Reference packages unavailable (exit 2) — skipping "
                        "cell2location comparison.";
    }
    ASSERT_EQ(rc, 0) << "Reference subprocess failed (exit " << rc << ").";

    // Load reference W_ref (n_spots × n_types).
    DenseMat ref = read_dense_npz_W(ref_out_path);
    ASSERT_EQ(ref.rows, static_cast<uint32_t>(kTinySpots));
    ASSERT_EQ(ref.cols, static_cast<uint32_t>(kTinyTypes));

    // Run FlashDeconv kernel.
    auto Y_dev = upload_csc(synth.Y_csc, stream_);

    using namespace singlet::gpu::core;
    DeviceMemory<float> B_dev(static_cast<size_t>(kTinyTypes) * kTinyGenes);
    cudaMemcpyAsync(B_dev.get(), synth.B.data(),
                    synth.B.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    singlet::gpu::spatial::FlashDeconvConfigCompat cfg;
    cfg.sketch_k = kSketchK;
    cfg.seed     = kSeed;

    auto result = singlet::gpu::spatial::flash_deconv_fit(
        Y_dev, B_dev, nullptr, kTinyTypes, kTinyGenes, cfg, stream_);

    ASSERT_EQ(result.n_spots, static_cast<uint32_t>(kTinySpots));
    ASSERT_EQ(result.n_types, static_cast<uint32_t>(kTinyTypes));

    // Validate W properties.
    for (uint32_t s = 0; s < kTinySpots; ++s) {
        float row_sum = 0.0f;
        for (uint32_t t = 0; t < kTinyTypes; ++t) {
            float w = result.W[s * kTinyTypes + t];
            EXPECT_GE(w, static_cast<float>(kNonNegTol))
                << "W[" << s << "," << t << "] = " << w;
            EXPECT_TRUE(std::isfinite(w))
                << "W[" << s << "," << t << "] is not finite";
            row_sum += w;
        }
        EXPECT_NEAR(row_sum, 1.0f, static_cast<float>(kSumToOneTol))
            << "Row " << s << " sum = " << row_sum;
    }

    // Compute per-spot Spearman ρ and average over all spots.
    double rho_sum = 0.0;
    int    n_valid = 0;
    for (uint32_t s = 0; s < kTinySpots; ++s) {
        std::vector<double> w_ours(kTinyTypes), w_ref(kTinyTypes);
        for (uint32_t t = 0; t < kTinyTypes; ++t) {
            w_ours[t] = static_cast<double>(result.W[s * kTinyTypes + t]);
            w_ref [t] = static_cast<double>(ref.data[s * kTinyTypes + t]);
        }
        double r = spearman_rho(w_ours, w_ref);
        if (std::isfinite(r)) { rho_sum += r; ++n_valid; }
    }
    double mean_rho = n_valid > 0 ? rho_sum / n_valid : 0.0;
    EXPECT_GE(mean_rho, kSpearmanMinRho)
        << "Mean per-spot Spearman ρ = " << mean_rho
        << " (threshold " << kSpearmanMinRho << ")";
}

// =============================================================================
// TC-2: FlashDeconv_GSM4037629_RealData
// =============================================================================

TEST_F(FlashDeconvTest, FlashDeconv_GSM4037629_RealData) {
    // Load GSM4037629 exon counts as a stand-in Visium-like matrix.
    // The test validates: finite W, sum-to-1, non-negative.  We do NOT check
    // biology (the sample is scRNA, not Visium) — this is a smoke test.

    if (!fs::exists(kGSM4037629CountPath)) {
        GTEST_SKIP() << "GSM4037629 not found at " << kGSM4037629CountPath;
    }

    using namespace singlet::gpu;
    auto loader  = io::PzDeviceLoader(kGSM4037629CountPath);
    auto Y_dev   = loader.load_csc(stream_);   // genes × cells

    // Use a tiny synthetic reference (10 types, same gene dimension) so we
    // only test the mechanics, not accuracy.
    const uint32_t n_types = 10;
    const uint32_t n_genes = (uint32_t)Y_dev.rows;   // match loaded gene count

    std::mt19937_64 rng(kSeed ^ 0xBEEF);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::vector<float> B_host(static_cast<size_t>(n_types) * n_genes);
    for (auto& v : B_host) v = uni(rng);

    core::DeviceMemory<float> B_dev(static_cast<size_t>(n_types) * n_genes);
    cudaMemcpyAsync(B_dev.get(), B_host.data(),
                    B_host.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    singlet::gpu::spatial::FlashDeconvConfigCompat cfg;
    cfg.sketch_k = 500;
    cfg.seed     = kSeed;

    auto result = spatial::flash_deconv_fit(
        Y_dev, B_dev, nullptr, n_types, n_genes, cfg, stream_);

    ASSERT_EQ(result.n_types, n_types);
    ASSERT_GT(result.n_spots, 0u);

    for (uint32_t s = 0; s < result.n_spots; ++s) {
        float row_sum = 0.0f;
        for (uint32_t t = 0; t < n_types; ++t) {
            float w = result.W[s * n_types + t];
            EXPECT_TRUE(std::isfinite(w))
                << "W[" << s << "," << t << "] = " << w;
            EXPECT_GE(w, static_cast<float>(kNonNegTol))
                << "W[" << s << "," << t << "] negative";
            row_sum += w;
        }
        EXPECT_NEAR(row_sum, 1.0f, static_cast<float>(kSumToOneTol))
            << "Row " << s << " sum = " << row_sum;
    }
}

// =============================================================================
// TC-3: FlashDeconv_LeverageScoreSketching_RecallTopGenes
// =============================================================================

TEST_F(FlashDeconvTest, FlashDeconv_LeverageScoreSketching_RecallTopGenes) {
    // Construct a reference B where the top-kSketchK genes have dramatically
    // higher variance across cell types.  Leverage-score sketching should
    // retain these informative genes with high recall.

    const uint32_t n_spots = kSketchSpots;
    const uint32_t n_genes = kSketchGenes;
    const uint32_t n_types = kSketchTypes;
    const int      K       = kSketchK;

    std::mt19937_64 rng(kSeed ^ 0x1234);
    std::normal_distribution<float> gauss;
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    // B: first K genes have high variance across types; rest are low-signal.
    std::vector<float> B_host(static_cast<size_t>(n_types) * n_genes, 0.0f);
    for (uint32_t t = 0; t < n_types; ++t) {
        for (uint32_t g = 0; g < n_genes; ++g) {
            if (g < static_cast<uint32_t>(K))
                B_host[t * n_genes + g] = 5.0f * uni(rng);   // high signal
            else
                B_host[t * n_genes + g] = 0.05f * uni(rng);  // low signal
        }
    }

    auto synth = make_synthetic(n_spots, n_genes, n_types, kSeed ^ 0x5678,
                                /*noise_level=*/0.05f);
    // Override B with our structured one.
    synth.B = B_host;

    auto Y_dev = upload_csc(synth.Y_csc, stream_);

    using namespace singlet::gpu::core;
    DeviceMemory<float> B_dev(static_cast<size_t>(n_types) * n_genes);
    cudaMemcpyAsync(B_dev.get(), B_host.data(),
                    B_host.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    singlet::gpu::spatial::FlashDeconvConfigCompat cfg;
    cfg.sketch_k = K;
    cfg.seed     = kSeed;

    auto result = singlet::gpu::spatial::flash_deconv_fit(
        Y_dev, B_dev, nullptr, n_types, n_genes, cfg, stream_);

    // top_gene_indices must have exactly K entries.
    ASSERT_EQ(static_cast<int>(result.top_gene_indices.size()), K);

    // Count how many of the top-K returned genes fall in [0, K).
    // Expectation: high-signal genes are preferentially selected.
    int recall = 0;
    for (int32_t idx : result.top_gene_indices)
        if (idx >= 0 && idx < K) ++recall;

    double recall_rate = static_cast<double>(recall) / K;
    // Design doc: leverage-score sketching should recover the high-variance
    // genes at >> random chance (random would give 50% recall here since K = n/2).
    // Require at least 65% recall (conservative; strong signal should yield >90%).
    EXPECT_GE(recall_rate, 0.65)
        << "Leverage-score recall = " << recall_rate
        << " (expected >= 0.65 for high-variance first half)";
}

// =============================================================================
// TC-4: FlashDeconv_BootstrapUncertainty_NonZero
// =============================================================================

TEST_F(FlashDeconvTest, FlashDeconv_BootstrapUncertainty_NonZero) {
    auto synth = make_synthetic(kBootSpots, kBootGenes, kBootTypes,
                                kSeed ^ 0xABCD);

    auto Y_dev = upload_csc(synth.Y_csc, stream_);

    using namespace singlet::gpu::core;
    DeviceMemory<float> B_dev(static_cast<size_t>(kBootTypes) * kBootGenes);
    cudaMemcpyAsync(B_dev.get(), synth.B.data(),
                    synth.B.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    singlet::gpu::spatial::FlashDeconvConfigCompat cfg;
    cfg.sketch_k          = 30;   // reduced for speed
    cfg.bootstrap_samples = kBootSamples;
    cfg.seed              = kSeed;

    auto result = singlet::gpu::spatial::flash_deconv_fit(
        Y_dev, B_dev, nullptr, kBootTypes, kBootGenes, cfg, stream_);

    ASSERT_EQ(result.uncertainty.size(),
              static_cast<size_t>(kBootSpots) * kBootTypes)
        << "Uncertainty output has wrong size.";

    // Every uncertainty entry must be non-negative and finite.
    float min_unc = std::numeric_limits<float>::infinity();
    float max_unc = -std::numeric_limits<float>::infinity();
    for (float u : result.uncertainty) {
        EXPECT_TRUE(std::isfinite(u)) << "Uncertainty is not finite: " << u;
        EXPECT_GE(u, 0.0f)           << "Uncertainty is negative: " << u;
        min_unc = std::min(min_unc, u);
        max_unc = std::max(max_unc, u);
    }

    // At least some uncertainty should be strictly positive (bootstrap has
    // non-degenerate variance when sketch_k > 1 and data varies).
    EXPECT_GT(max_unc, 0.0f)
        << "All uncertainty values are 0 — bootstrap has no variance.";

    // Maximum uncertainty should be bounded (design doc: within 20% of
    // cell2location, which has max ≤ 1.0 since proportions are in [0,1]).
    EXPECT_LE(max_unc, static_cast<float>(kMaxUncertainty))
        << "Max uncertainty " << max_unc
        << " exceeds bound " << kMaxUncertainty;
}

// =============================================================================
// TC-5: FlashDeconv_Determinism_BitIdentical
// =============================================================================

TEST_F(FlashDeconvTest, FlashDeconv_Determinism_BitIdentical) {
    auto synth = make_synthetic(kTinySpots, kTinyGenes, kTinyTypes, kSeed);

    auto Y_dev1 = upload_csc(synth.Y_csc, stream_);
    auto Y_dev2 = upload_csc(synth.Y_csc, stream_);

    using namespace singlet::gpu::core;
    auto make_B_dev = [&]() {
        DeviceMemory<float> b(static_cast<size_t>(kTinyTypes) * kTinyGenes);
        cudaMemcpyAsync(b.get(), synth.B.data(),
                        synth.B.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
        return b;
    };

    singlet::gpu::spatial::FlashDeconvConfigCompat cfg;
    cfg.sketch_k = kSketchK;
    cfg.seed     = kSeed;  // same seed both times

    auto B1 = make_B_dev();
    auto r1 = singlet::gpu::spatial::flash_deconv_fit(
        Y_dev1, B1, nullptr, kTinyTypes, kTinyGenes, cfg, stream_);

    auto B2 = make_B_dev();
    auto r2 = singlet::gpu::spatial::flash_deconv_fit(
        Y_dev2, B2, nullptr, kTinyTypes, kTinyGenes, cfg, stream_);

    ASSERT_EQ(r1.W.size(), r2.W.size());
    for (size_t i = 0; i < r1.W.size(); ++i) {
        EXPECT_EQ(r1.W[i], r2.W[i])
            << "W differs at element " << i << ": "
            << r1.W[i] << " vs " << r2.W[i];
    }

    // top_gene_indices should also be bit-identical.
    ASSERT_EQ(r1.top_gene_indices.size(), r2.top_gene_indices.size());
    for (size_t i = 0; i < r1.top_gene_indices.size(); ++i) {
        EXPECT_EQ(r1.top_gene_indices[i], r2.top_gene_indices[i])
            << "top_gene_indices[" << i << "] differs.";
    }
}

// =============================================================================
// TC-6: FlashDeconv_SpatialRegularization_SmoothsAbundance
// =============================================================================

TEST_F(FlashDeconvTest, FlashDeconv_SpatialRegularization_SmoothsAbundance) {
    // Build a 12×12 grid dataset with spatial_lambda=0 vs spatial_lambda=1.
    // With regularization, neighboring spots on the grid should have more
    // correlated abundances than without.

    auto synth = make_synthetic(kSpatSpots, kSpatGenes, kSpatTypes,
                                kSeed ^ 0xDEAD, /*noise_level=*/0.5f);

    auto Y_dev_base = upload_csc(synth.Y_csc, stream_);
    auto Y_dev_reg  = upload_csc(synth.Y_csc, stream_);

    using namespace singlet::gpu::core;
    auto make_B_dev = [&]() {
        DeviceMemory<float> b(static_cast<size_t>(kSpatTypes) * kSpatGenes);
        cudaMemcpyAsync(b.get(), synth.B.data(),
                        synth.B.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
        return b;
    };

    // Build spatial coordinates on a 12×12 grid.
    auto xy_host = make_grid_coords(kSpatRows, kSpatCols);
    DeviceMemory<float> xy_dev(static_cast<size_t>(kSpatSpots) * 2);
    cudaMemcpyAsync(xy_dev.get(), xy_host.data(),
                    xy_host.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    // Run without spatial regularization.
    singlet::gpu::spatial::FlashDeconvConfigCompat cfg_base;
    cfg_base.sketch_k       = 30;
    cfg_base.spatial_lambda = 0.0f;
    cfg_base.seed           = kSeed;

    auto B1   = make_B_dev();
    auto r_base = singlet::gpu::spatial::flash_deconv_fit(
        Y_dev_base, B1, nullptr, kSpatTypes, kSpatGenes, cfg_base, stream_);

    // Run with spatial regularization.
    singlet::gpu::spatial::FlashDeconvConfigCompat cfg_reg;
    cfg_reg.sketch_k       = 30;
    cfg_reg.spatial_lambda = 1.0f;
    cfg_reg.spatial_knn_k  = 4;   // 4-connectivity on grid
    cfg_reg.seed           = kSeed;

    auto B2   = make_B_dev();
    auto r_reg = singlet::gpu::spatial::flash_deconv_fit(
        Y_dev_reg, B2, &xy_dev, kSpatTypes, kSpatGenes, cfg_reg, stream_);

    // Measure spatial autocorrelation for each cell type:
    // Compare abundance of each spot to its right-neighbor (col+1) on the grid.
    // With regularization, correlation across neighbors should be higher.
    double corr_base_sum = 0.0, corr_reg_sum = 0.0;
    int    n_pairs = 0;

    for (int t = 0; t < static_cast<int>(kSpatTypes); ++t) {
        std::vector<float> left_base, right_base;
        std::vector<float> left_reg,  right_reg;

        for (int r = 0; r < kSpatRows; ++r) {
            for (int c = 0; c + 1 < kSpatCols; ++c) {
                int s_l = r * kSpatCols + c;
                int s_r = r * kSpatCols + (c + 1);
                left_base .push_back(r_base.W[s_l * kSpatTypes + t]);
                right_base.push_back(r_base.W[s_r * kSpatTypes + t]);
                left_reg  .push_back(r_reg .W[s_l * kSpatTypes + t]);
                right_reg .push_back(r_reg .W[s_r * kSpatTypes + t]);
            }
        }

        corr_base_sum += pearson(left_base, right_base);
        corr_reg_sum  += pearson(left_reg,  right_reg);
        ++n_pairs;
    }

    double mean_corr_base = n_pairs > 0 ? corr_base_sum / n_pairs : 0.0;
    double mean_corr_reg  = n_pairs > 0 ? corr_reg_sum  / n_pairs : 0.0;

    // Regularized output must have higher spatial correlation than unregularized.
    EXPECT_GT(mean_corr_reg, mean_corr_base)
        << "Spatial regularization did not increase neighbor correlation. "
        << "corr_base=" << mean_corr_base << " corr_reg=" << mean_corr_reg;

    // Regularized W must remain sum-to-1 and non-negative.
    for (uint32_t s = 0; s < static_cast<uint32_t>(kSpatSpots); ++s) {
        float row_sum = 0.0f;
        for (uint32_t ty = 0; ty < static_cast<uint32_t>(kSpatTypes); ++ty) {
            float w = r_reg.W[s * kSpatTypes + ty];
            EXPECT_GE(w, static_cast<float>(kNonNegTol));
            EXPECT_TRUE(std::isfinite(w));
            row_sum += w;
        }
        EXPECT_NEAR(row_sum, 1.0f, static_cast<float>(kSumToOneTol));
    }
}
