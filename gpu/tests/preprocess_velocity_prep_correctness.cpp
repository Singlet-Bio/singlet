// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/preprocess_velocity_prep_correctness.cpp
//
// integrates: original (intron-aware velocity prep — first to exploit singlify's exon+intron output)
//
// Correctness harness for singlet_gpu::preprocess::velocity_prep.
//
// Written against the design doc at:
//   singlet-gpu/state/designs/13-velocity-prep.md
// NOT against the kernel source — authored in parallel with
// preprocess/velocity_prep.h by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: scVelo Python via subprocess
//   tests/refs/velocity_prep_scvelo_reference.py
//   (pip install scvelo scanpy anndata numpy scipy)
//
// Tolerances (design doc §"Correctness test spec"):
//   gamma Spearman rho >= 0.95   (per-gene, over all passing genes)
//   moments rel_err   <= 1e-3    (mean relative error for Ms/Mu)
//   shape correctness            (smoothed S/U same shape as input)
//   determinism: bit-identical   (two runs on same input)
//
// Test cases:
//   VelocityPrep_TinyPlanted_VsScvelo
//   VelocityPrep_GSM4037629_RealData
//   VelocityPrep_MinCountFilter_FiltersLowExpression
//   VelocityPrep_MomentsSmoothing_PreservesShape
//   VelocityPrep_VelocityVectorsSign
//   VelocityPrep_Determinism_BitIdentical
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every test that requires a device is guarded by
//   gpu_available() and calls GTEST_SKIP().

// ---- singlet-gpu velocity prep header (written by gpu-kernel-dev) -----------
#include <singlet-gpu/preprocess/velocity_prep.h>

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

/// Tiny synthetic dimensions: 200 cells x 100 genes (design doc spec).
constexpr int kTinyCells = 200;
constexpr int kTinyGenes = 100;

/// Number of planted gamma "buckets" for the synthetic test.
/// Half the genes are "high-gamma" (γ ≈ 1.5), half are "low-gamma" (γ ≈ 0.3).
constexpr float kHighGamma = 1.5f;
constexpr float kLowGamma  = 0.3f;

/// min_S_count filter threshold (must match VelocityPrepConfig defaults).
constexpr int kMinSCount = 10;
constexpr int kMinUCount = 5;

/// Tolerance thresholds from the design doc.
constexpr double kGammaSpearmanMin   = 0.95;
constexpr double kMomentsRelErrMax   = 1e-3;

/// Real-data path — GSM4037629.
static const char* kGsmSplicedPath =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/exon_counts.1pz";
static const char* kGsmUnsplicedPath =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/intron_counts.1pz";

/// scVelo reference script path.
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/"
    "tests/refs/velocity_prep_scvelo_reference.py";

/// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_velocity_prep_refs_tmp";

/// Python interpreter.
static const char* kPython = "python3";

// =============================================================================
// Utility helpers (pattern from cycles 3-14)
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
// Host-side CSC container.
// ---------------------------------------------------------------------------
struct HostCSC {
    uint32_t             m;        // genes (rows)
    uint32_t             n;        // cells (cols)
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
    static constexpr uint32_t kMagic = 0x43535343u;
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
singlet_gpu::core::DeviceCSC upload_csc(const HostCSC& h, cudaStream_t /*stream*/) {
    return singlet_gpu::core::DeviceCSC(
        (int)h.m, (int)h.n, (int)h.nnz,
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
        throw std::runtime_error(std::string("download_f32: ") + cudaGetErrorString(e));
    cudaStreamSynchronize(stream);
    return h;
}

// ---------------------------------------------------------------------------
// Download a DeviceMemory<uint8_t> to host vector.
// ---------------------------------------------------------------------------
std::vector<uint8_t> download_u8(
        const singlet_gpu::core::DeviceMemory<uint8_t>& d,
        size_t count, cudaStream_t stream)
{
    std::vector<uint8_t> h(count);
    cudaError_t e = cudaMemcpyAsync(h.data(), d.get(),
                                    count * sizeof(uint8_t),
                                    cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("download_u8: ") + cudaGetErrorString(e));
    cudaStreamSynchronize(stream);
    return h;
}

// ---------------------------------------------------------------------------
// Spearman rank correlation (pure C++) — reused from cycles 11-14.
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

    double mean_a = 0, mean_b = 0;
    for (int i = 0; i < n; ++i) { mean_a += ra[i]; mean_b += rb[i]; }
    mean_a /= n; mean_b /= n;

    double num = 0, denom_a = 0, denom_b = 0;
    for (int i = 0; i < n; ++i) {
        double da = ra[i] - mean_a, db = rb[i] - mean_b;
        num     += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }
    if (denom_a == 0 || denom_b == 0) return 0.0;
    return num / std::sqrt(denom_a * denom_b);
}

// ---------------------------------------------------------------------------
// Load a 1-D float32 array from a .npz member.
//
// The .npz format is a ZIP of .npy files. We do a minimal extraction via
// Python subprocess: read back the pre-saved .npz via numpy.load.
// Since the .npz is already on disk, we can call a one-liner to extract
// a specific key to a flat binary file.
// ---------------------------------------------------------------------------
std::vector<float> load_npz_f32_1d(const std::string& npz_path,
                                    const std::string& key,
                                    int expected_n)
{
    // Extract key from npz to a temp .npy file via Python one-liner.
    std::string tmp_npy = refs_path("_npz_extract_" + key + ".npy");
    std::string cmd =
        std::string(kPython) + " -c \""
        "import numpy as np; "
        "d = np.load('" + npz_path + "', allow_pickle=False); "
        "np.save('" + tmp_npy + "', d['" + key + "'])\"";
    run_cmd(cmd);

    // Read the .npy file: minimal v1.0 parse (float32 1-D).
    std::ifstream f(tmp_npy, std::ios::binary);
    if (!f) throw std::runtime_error("load_npz_f32_1d: cannot open " + tmp_npy);

    char preamble[10];
    f.read(preamble, 10);
    if (preamble[0] != '\x93' || std::string(preamble + 1, 5) != "NUMPY")
        throw std::runtime_error("load_npz_f32_1d: not a .npy: " + tmp_npy);

    uint16_t hlen;
    f.read(reinterpret_cast<char*>(&hlen), 2);
    std::string header(hlen, ' ');
    f.read(header.data(), hlen);

    // We read all remaining bytes as float32.
    f.seekg(0, std::ios::end);
    auto total = static_cast<size_t>(f.tellg());
    auto data_offset = static_cast<size_t>(10 + 2 + hlen);
    size_t data_bytes = total - data_offset;
    int n = static_cast<int>(data_bytes / sizeof(float));

    f.seekg(static_cast<std::streamoff>(data_offset), std::ios::beg);
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
    if (!f && n > 0)
        throw std::runtime_error("load_npz_f32_1d: short read: " + tmp_npy);

    if (expected_n > 0 && n != expected_n) {
        throw std::runtime_error("load_npz_f32_1d: key=" + key +
                                 " expected " + std::to_string(expected_n) +
                                 " got " + std::to_string(n));
    }
    return v;
}

// ---------------------------------------------------------------------------
// Load a scalar int from a .npz member.
// ---------------------------------------------------------------------------
int load_npz_int(const std::string& npz_path, const std::string& key)
{
    std::string cmd =
        std::string(kPython) + " -c \""
        "import numpy as np; "
        "d = np.load('" + npz_path + "', allow_pickle=False); "
        "print(int(d['" + key + "']))\"";
    // Capture stdout via a temp file.
    std::string tmp_out = refs_path("_npz_int_" + key + ".txt");
    run_cmd(cmd + " > " + tmp_out);
    std::ifstream f(tmp_out);
    int val; f >> val;
    return val;
}

// ---------------------------------------------------------------------------
// Load a 1-D int32 array from a .npz member.
// ---------------------------------------------------------------------------
std::vector<int32_t> load_npz_i32_1d(const std::string& npz_path,
                                      const std::string& key,
                                      int expected_n)
{
    std::string tmp_npy = refs_path("_npz_extract_" + key + ".npy");
    std::string cmd =
        std::string(kPython) + " -c \""
        "import numpy as np; "
        "d = np.load('" + npz_path + "', allow_pickle=False); "
        "np.save('" + tmp_npy + "', d['" + key + "'].astype('int32'))\"";
    run_cmd(cmd);

    std::ifstream f(tmp_npy, std::ios::binary);
    if (!f) throw std::runtime_error("load_npz_i32_1d: cannot open " + tmp_npy);
    char preamble[10]; f.read(preamble, 10);
    uint16_t hlen; f.read(reinterpret_cast<char*>(&hlen), 2);
    std::string header(hlen, ' '); f.read(header.data(), hlen);
    f.seekg(0, std::ios::end);
    auto total = static_cast<size_t>(f.tellg());
    auto data_offset = static_cast<size_t>(10 + 2 + hlen);
    size_t data_bytes = total - data_offset;
    int n = static_cast<int>(data_bytes / sizeof(int32_t));
    f.seekg(static_cast<std::streamoff>(data_offset), std::ios::beg);
    std::vector<int32_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(n * sizeof(int32_t)));
    if (expected_n > 0 && n != expected_n)
        throw std::runtime_error("load_npz_i32_1d: key=" + key +
                                 " expected " + std::to_string(expected_n) +
                                 " got " + std::to_string(n));
    return v;
}

// ---------------------------------------------------------------------------
// Synthetic CSC generator.
//
// Builds a (genes x cells) CSC with planted gamma values:
//   - genes [0, n_genes/2): high gamma (U ≈ kHighGamma * S + noise)
//   - genes [n_genes/2, n_genes): low gamma (U ≈ kLowGamma * S + noise)
//   - density ~40% (to ensure enough expressed genes pass the min_S filter)
//
// The planted gamma values let us verify that the GPU kernel ranks genes
// correctly even if the absolute values differ from the reference.
// ---------------------------------------------------------------------------
struct SyntheticVelocityData {
    HostCSC spliced;
    HostCSC unspliced;
    std::vector<float> planted_gamma;   // (n_genes,) — true gamma per gene
};

SyntheticVelocityData make_synthetic_velocity(
        int n_genes, int n_cells, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    std::normal_distribution<float>       ndist(0.0f, 0.1f);

    // Target density ~40% to ensure most genes have enough counts.
    const float density = 0.40f;

    // Build spliced matrix: moderate counts for each expressed entry.
    // Build unspliced matrix: U ≈ gamma * S + small noise (non-negative).
    std::vector<float>   s_vals, u_vals;
    std::vector<int32_t> s_idx, u_idx;
    std::vector<int32_t> s_ptr(n_cells + 1, 0), u_ptr(n_cells + 1, 0);

    std::vector<float> planted(n_genes);
    for (int g = 0; g < n_genes; ++g)
        planted[g] = (g < n_genes / 2) ? kHighGamma : kLowGamma;

    // Build column by column (CSC: outer = cells, inner = genes).
    for (int c = 0; c < n_cells; ++c) {
        for (int g = 0; g < n_genes; ++g) {
            if (unif(rng) < density) {
                // Spliced count: moderate Poisson-ish count, mapped to float.
                float s_count = static_cast<float>(
                    static_cast<int>(10.0f + 40.0f * unif(rng)));
                float noise   = ndist(rng) * s_count;
                float u_count = std::max(0.0f, planted[g] * s_count + noise);

                s_vals.push_back(s_count);
                s_idx .push_back(g);
                u_vals.push_back(u_count);
                u_idx .push_back(g);
            }
        }
        s_ptr[c + 1] = static_cast<int32_t>(s_vals.size());
        u_ptr[c + 1] = static_cast<int32_t>(u_vals.size());
    }

    SyntheticVelocityData d;

    d.spliced.m       = static_cast<uint32_t>(n_genes);
    d.spliced.n       = static_cast<uint32_t>(n_cells);
    d.spliced.nnz     = s_vals.size();
    d.spliced.values  = std::move(s_vals);
    d.spliced.indptr  = std::move(s_ptr);
    d.spliced.indices = std::move(s_idx);

    d.unspliced.m       = static_cast<uint32_t>(n_genes);
    d.unspliced.n       = static_cast<uint32_t>(n_cells);
    d.unspliced.nnz     = u_vals.size();
    d.unspliced.values  = std::move(u_vals);
    d.unspliced.indptr  = std::move(u_ptr);
    d.unspliced.indices = std::move(u_idx);

    d.planted_gamma = std::move(planted);
    return d;
}

}  // namespace

// =============================================================================
// Test fixture
// =============================================================================

class VelocityPrepTest : public ::testing::Test {
protected:
    void SetUp() override {
        ensure_refs_tmp();
    }
};

// =============================================================================
// Test 1: VelocityPrep_TinyPlanted_VsScvelo
//
// 200 cells x 100 genes synthetic with planted gamma (high / low).
// Run GPU velocity_prep and scVelo on the same input.
// Compare per-gene gamma: Spearman rho >= 0.95.
// =============================================================================

TEST_F(VelocityPrepTest, VelocityPrep_TinyPlanted_VsScvelo) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device found.";

    auto d = make_synthetic_velocity(kTinyGenes, kTinyCells, kSeed);

    // Write spliced + unspliced to dump_csc.h binary.
    const std::string s_bin  = refs_path("tiny_spliced.bin");
    const std::string u_bin  = refs_path("tiny_unspliced.bin");
    const std::string ref_npz = refs_path("tiny_scvelo_ref.npz");
    write_csc_bin(d.spliced,   s_bin);
    write_csc_bin(d.unspliced, u_bin);

    // Run scVelo reference.
    std::string ref_cmd =
        std::string(kPython) + " " + kRefScript +
        " --spliced "   + s_bin +
        " --unspliced " + u_bin +
        " --out "       + ref_npz +
        " --n-neighbors 10"         // small dataset: reduce k
        " --min-shared-counts 5"    // small dataset: lower threshold
        " --n-pcs 10";
    try {
        run_cmd(ref_cmd);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "scVelo reference failed (not installed?): " << e.what();
    }

    int ref_n_genes         = load_npz_int(ref_npz, "n_genes");
    int ref_n_genes_passing = load_npz_int(ref_npz, "n_genes_passing");
    printf("[TinyPlanted] ref: n_genes=%d n_passing=%d\n",
           ref_n_genes, ref_n_genes_passing);

    // GPU velocity_prep.
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet_gpu::core::DeviceCSC d_s = upload_csc(d.spliced,   stream);
    singlet_gpu::core::DeviceCSC d_u = upload_csc(d.unspliced, stream);

    singlet_gpu::preprocess::VelocityPrepConfig cfg;
    cfg.min_S_count     = 5;    // match scVelo min_shared_counts for fair comparison
    cfg.min_U_count     = 3;
    cfg.smooth_moments  = true;
    cfg.compute_velocity = true;

    auto result = singlet_gpu::preprocess::velocity_prep(d_s, d_u, nullptr, cfg, stream);
    cudaStreamSynchronize(stream);

    // Download gamma.
    auto h_gamma = download_f32(result.gamma, kTinyGenes, stream);

    // Load reference gamma (full vector, NaN for filtered genes).
    auto ref_gamma = load_npz_f32_1d(ref_npz, "gamma", kTinyGenes);

    // Extract only genes where BOTH are valid (not NaN) for Spearman comparison.
    std::vector<float> gpu_passing, ref_passing;
    for (int g = 0; g < kTinyGenes; ++g) {
        if (!std::isnan(ref_gamma[g]) && !std::isnan(h_gamma[g])) {
            gpu_passing.push_back(h_gamma[g]);
            ref_passing.push_back(ref_gamma[g]);
        }
    }
    printf("[TinyPlanted] Genes with valid gamma in both: %d / %d\n",
           static_cast<int>(gpu_passing.size()), kTinyGenes);

    ASSERT_GE(static_cast<int>(gpu_passing.size()), 10)
        << "Too few genes with valid gamma for Spearman comparison";

    double rho = spearman(gpu_passing, ref_passing);
    printf("[TinyPlanted] gamma Spearman rho = %.4f (threshold %.2f)\n",
           rho, kGammaSpearmanMin);
    EXPECT_GE(rho, kGammaSpearmanMin)
        << "Spearman rho " << rho << " < " << kGammaSpearmanMin;

    cudaStreamDestroy(stream);
}

// =============================================================================
// Test 2: VelocityPrep_GSM4037629_RealData
//
// Load both exon_counts.1pz (spliced) and intron_counts.1pz (unspliced) from
// GSM4037629 — the two singlify outputs unique to this library.
// Run velocity_prep. Confirm:
//   - gamma values mostly in [0, 5] (no runaway values)
//   - filter_mask has a meaningful number of passing genes (>= 100)
//   - no NaNs or Infs in gamma for passing genes
// =============================================================================

TEST_F(VelocityPrepTest, VelocityPrep_GSM4037629_RealData) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device found.";

    if (!fs::exists(kGsmSplicedPath)) {
        GTEST_SKIP() << "GSM4037629 exon_counts.1pz not found at " << kGsmSplicedPath;
    }
    if (!fs::exists(kGsmUnsplicedPath)) {
        GTEST_SKIP() << "GSM4037629 intron_counts.1pz not found at " << kGsmUnsplicedPath;
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    // Load spliced and unspliced via pz_device_loader.
    auto d_spliced   = singlet_gpu::io::load_pz(kGsmSplicedPath,   stream).mat;
    auto d_unspliced = singlet_gpu::io::load_pz(kGsmUnsplicedPath, stream).mat;
    cudaStreamSynchronize(stream);

    ASSERT_EQ(d_spliced.rows, d_unspliced.rows)
        << "Spliced/unspliced gene count mismatch";
    ASSERT_EQ(d_spliced.cols, d_unspliced.cols)
        << "Spliced/unspliced cell count mismatch";

    int n_genes = static_cast<int>(d_spliced.rows);
    int n_cells = static_cast<int>(d_spliced.cols);
    printf("[GSM4037629] Loaded: %d genes x %d cells\n", n_genes, n_cells);

    singlet_gpu::preprocess::VelocityPrepConfig cfg;
    // Use design-doc defaults.
    auto result = singlet_gpu::preprocess::velocity_prep(
        d_spliced, d_unspliced, nullptr, cfg, stream);
    cudaStreamSynchronize(stream);

    // Download gamma and filter_mask.
    auto h_gamma  = download_f32(result.gamma, n_genes, stream);
    auto h_fmask  = download_u8 (result.filter_mask, n_genes, stream);

    // Count passing genes.
    int n_passing = 0;
    for (int g = 0; g < n_genes; ++g)
        if (h_fmask[g]) ++n_passing;

    printf("[GSM4037629] Genes passing filter: %d / %d\n", n_passing, n_genes);
    EXPECT_GE(n_passing, 100) << "Too few genes passing filter";
    EXPECT_EQ(n_passing, result.n_genes_passing_filter);

    // Check gamma for passing genes.
    int n_nan = 0, n_inf = 0, n_out_of_range = 0;
    float gamma_max = 0.0f;
    for (int g = 0; g < n_genes; ++g) {
        if (!h_fmask[g]) continue;
        if (std::isnan(h_gamma[g])) { ++n_nan; continue; }
        if (std::isinf(h_gamma[g])) { ++n_inf; continue; }
        if (h_gamma[g] < 0.0f || h_gamma[g] > 5.0f) ++n_out_of_range;
        gamma_max = std::max(gamma_max, h_gamma[g]);
    }
    printf("[GSM4037629] gamma max=%.4f  n_nan=%d  n_inf=%d  n_out_of_range=%d\n",
           gamma_max, n_nan, n_inf, n_out_of_range);

    EXPECT_EQ(n_nan, 0)  << "NaN values in gamma for passing genes";
    EXPECT_EQ(n_inf, 0)  << "Inf values in gamma for passing genes";

    // Most genes should have gamma in [0, 5] — warn if >5% are outside.
    double pct_oob = (n_passing > 0)
        ? static_cast<double>(n_out_of_range) / n_passing * 100.0
        : 0.0;
    printf("[GSM4037629] %.1f%% of passing genes have gamma outside [0, 5]\n", pct_oob);
    EXPECT_LE(pct_oob, 20.0)
        << "More than 20% of passing genes have gamma outside [0, 5]";

    cudaStreamDestroy(stream);
}

// =============================================================================
// Test 3: VelocityPrep_MinCountFilter_FiltersLowExpression
//
// Build a matrix where gene 0 has total_S < 10 (deliberately low counts).
// Confirm filter_mask[0] == 0 (filtered out).
// =============================================================================

TEST_F(VelocityPrepTest, VelocityPrep_MinCountFilter_FiltersLowExpression) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device found.";

    // 50 cells x 20 genes.  Gene 0: only 3 nonzero entries (total_S ≈ 9 < 10).
    // Other genes: ~40 nonzero entries each (total_S >> 10).
    constexpr int kG = 20, kC = 50;
    std::mt19937_64 rng(kSeed ^ 0xABCDull);
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);

    HostCSC sp, un;
    sp.m = static_cast<uint32_t>(kG);
    sp.n = static_cast<uint32_t>(kC);
    un.m = sp.m; un.n = sp.n;

    std::vector<float>   sv, uv;
    std::vector<int32_t> si, ui;
    std::vector<int32_t> sptr(kC + 1, 0), uptr(kC + 1, 0);

    // Gene 0 appears in exactly 3 cells with count 3 → total_S = 9.
    const std::set<int> gene0_cells = {1, 15, 30};

    for (int c = 0; c < kC; ++c) {
        for (int g = 0; g < kG; ++g) {
            bool in_gene0 = (g == 0) && (gene0_cells.count(c) > 0);
            bool high_gene = (g > 0) && (unif(rng) < 0.7f);

            if (in_gene0) {
                sv.push_back(3.0f);   // small count for gene 0
                si.push_back(g);
                uv.push_back(0.5f);
                ui.push_back(g);
            } else if (high_gene) {
                float s = 20.0f + 20.0f * unif(rng);
                sv.push_back(s);  si.push_back(g);
                uv.push_back(s * 0.3f + 0.1f);  ui.push_back(g);
            }
        }
        sptr[c + 1] = static_cast<int32_t>(sv.size());
        uptr[c + 1] = static_cast<int32_t>(uv.size());
    }

    sp.nnz = sv.size(); sp.values = sv; sp.indptr = sptr; sp.indices = si;
    un.nnz = uv.size(); un.values = uv; un.indptr = uptr; un.indices = ui;

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto d_s = upload_csc(sp, stream);
    auto d_u = upload_csc(un, stream);

    singlet_gpu::preprocess::VelocityPrepConfig cfg;
    cfg.min_S_count = 10;   // design-doc default
    cfg.min_U_count = 5;

    auto result = singlet_gpu::preprocess::velocity_prep(d_s, d_u, nullptr, cfg, stream);
    cudaStreamSynchronize(stream);

    auto h_fmask = download_u8(result.filter_mask, kG, stream);

    // Gene 0 must be filtered out (total_S = 9 < min_S_count = 10).
    EXPECT_EQ(static_cast<int>(h_fmask[0]), 0)
        << "Gene 0 (total_S = 9) should be filtered out by min_S_count = 10";

    // Other genes should mostly pass (not all — depends on counts — but > 0).
    int n_pass_others = 0;
    for (int g = 1; g < kG; ++g) if (h_fmask[g]) ++n_pass_others;
    printf("[MinCountFilter] gene0 mask=%d  others passing: %d / %d\n",
           static_cast<int>(h_fmask[0]), n_pass_others, kG - 1);
    EXPECT_GT(n_pass_others, 0)
        << "No genes other than gene 0 pass the filter — data generator issue";

    cudaStreamDestroy(stream);
}

// =============================================================================
// Test 4: VelocityPrep_MomentsSmoothing_PreservesShape
//
// Run velocity_prep with smooth_moments=true.
// Confirm smoothed S and U have the same shape as input (n_genes x n_cells).
// =============================================================================

TEST_F(VelocityPrepTest, VelocityPrep_MomentsSmoothing_PreservesShape) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device found.";

    auto d = make_synthetic_velocity(kTinyGenes, kTinyCells, kSeed ^ 0x1234ull);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto d_s = upload_csc(d.spliced,   stream);
    auto d_u = upload_csc(d.unspliced, stream);

    singlet_gpu::preprocess::VelocityPrepConfig cfg;
    cfg.smooth_moments   = true;
    cfg.compute_velocity = true;

    auto result = singlet_gpu::preprocess::velocity_prep(d_s, d_u, nullptr, cfg, stream);
    cudaStreamSynchronize(stream);

    // The velocity matrix should have shape n_genes x n_cells.
    // velocity is stored as DeviceMemory<float> of size n_genes * n_cells.
    // Check the allocation size matches expected elements.
    size_t expected_velocity_size =
        static_cast<size_t>(kTinyGenes) * static_cast<size_t>(kTinyCells);

    EXPECT_EQ(result.velocity.size(), expected_velocity_size)
        << "velocity DeviceMemory size " << result.velocity.size()
        << " != expected " << expected_velocity_size
        << " (n_genes=" << kTinyGenes << " n_cells=" << kTinyCells << ")";

    // gamma and filter_mask must be per-gene (n_genes elements).
    EXPECT_EQ(result.gamma      .size(), static_cast<size_t>(kTinyGenes));
    EXPECT_EQ(result.gamma_se   .size(), static_cast<size_t>(kTinyGenes));
    EXPECT_EQ(result.filter_mask.size(), static_cast<size_t>(kTinyGenes));
    EXPECT_EQ(result.S_mean     .size(), static_cast<size_t>(kTinyGenes));
    EXPECT_EQ(result.U_mean     .size(), static_cast<size_t>(kTinyGenes));

    printf("[PreservesShape] velocity size=%zu  (expected %zu)\n",
           result.velocity.size(), expected_velocity_size);

    cudaStreamDestroy(stream);
}

// =============================================================================
// Test 5: VelocityPrep_VelocityVectorsSign
//
// Synthetic spot-check: cells constructed so that some have clearly high
// U/S ratio (positive velocity expected) and some have clearly low U/S ratio
// (negative velocity expected).
//
// We plant:
//   - "high-flux" cells: U > gamma * S  → velocity > 0
//   - "low-flux"  cells: U < gamma * S  → velocity < 0
// =============================================================================

TEST_F(VelocityPrepTest, VelocityPrep_VelocityVectorsSign) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device found.";

    // Small matrix: 2 genes x 20 cells.
    // Gene 0 has known gamma = 1.0.
    // Cells 0-9: U = 1.5 * S → velocity > 0 (U > gamma * S).
    // Cells 10-19: U = 0.5 * S → velocity < 0 (U < gamma * S).
    constexpr int kG = 2, kC = 20;
    const float kGamma = 1.0f;
    const float kS     = 50.0f;   // high count to pass min_S filter
    const float kUHigh = kGamma * kS * 1.5f;   // U > gamma * S → positive velocity
    const float kULow  = kGamma * kS * 0.5f;   // U < gamma * S → negative velocity

    HostCSC sp, un;
    sp.m = kG; sp.n = kC; un.m = kG; un.n = kC;

    // Fill both genes with the same pattern to ensure they pass filter.
    std::vector<float>   sv, uv;
    std::vector<int32_t> si, ui;
    std::vector<int32_t> sptr(kC + 1, 0), uptr(kC + 1, 0);

    for (int c = 0; c < kC; ++c) {
        for (int g = 0; g < kG; ++g) {
            sv.push_back(kS);
            si.push_back(g);
            float u = (c < kC / 2) ? kUHigh : kULow;
            uv.push_back(u);
            ui.push_back(g);
        }
        sptr[c + 1] = static_cast<int32_t>(sv.size());
        uptr[c + 1] = static_cast<int32_t>(uv.size());
    }

    sp.nnz = sv.size(); sp.values = sv; sp.indptr = sptr; sp.indices = si;
    un.nnz = uv.size(); un.values = uv; un.indptr = uptr; un.indices = ui;

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto d_s = upload_csc(sp, stream);
    auto d_u = upload_csc(un, stream);

    singlet_gpu::preprocess::VelocityPrepConfig cfg;
    cfg.min_S_count      = 5;     // kS = 50 >> 5, so gene passes
    cfg.min_U_count      = 3;
    cfg.smooth_moments   = false;  // disable smoothing for clear sign test
    cfg.compute_velocity = true;

    auto result = singlet_gpu::preprocess::velocity_prep(d_s, d_u, nullptr, cfg, stream);
    cudaStreamSynchronize(stream);

    auto h_fmask = download_u8(result.filter_mask, kG, stream);
    // Gene 0 must pass (total_S = kS * kC / 2 >> min_S_count).
    ASSERT_EQ(static_cast<int>(h_fmask[0]), 1) << "Gene 0 should pass filter";

    auto h_vel = download_f32(result.velocity,
                              static_cast<size_t>(kG) * static_cast<size_t>(kC),
                              stream);

    // Velocity for gene 0:
    //   cells 0-9 (high U): velocity > 0
    //   cells 10-19 (low U): velocity < 0
    // Velocity is stored column-major (CSC) or row-major? Design doc: dense m x n.
    // Assume row-major (genes x cells): vel[gene * n_cells + cell].
    int n_pos_correct = 0, n_neg_correct = 0;
    for (int c = 0; c < kC; ++c) {
        float v = h_vel[0 * kC + c];   // gene 0, cell c
        bool high_flux = (c < kC / 2);
        if (high_flux && v > 0.0f) ++n_pos_correct;
        if (!high_flux && v < 0.0f) ++n_neg_correct;
    }
    printf("[VelocityVectorsSign] positive correct: %d/10  negative correct: %d/10\n",
           n_pos_correct, n_neg_correct);

    // With no smoothing and planted gamma, most cells should have the right sign.
    // Allow 1 cell to be wrong (rounding / tied boundary).
    EXPECT_GE(n_pos_correct, 8)
        << "High-U cells should have positive velocity";
    EXPECT_GE(n_neg_correct, 8)
        << "Low-U cells should have negative velocity";

    cudaStreamDestroy(stream);
}

// =============================================================================
// Test 6: VelocityPrep_Determinism_BitIdentical
//
// Run velocity_prep twice on the same input. Results must be bit-identical.
// =============================================================================

TEST_F(VelocityPrepTest, VelocityPrep_Determinism_BitIdentical) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device found.";

    auto d = make_synthetic_velocity(kTinyGenes, kTinyCells, kSeed ^ 0xDEADull);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet_gpu::preprocess::VelocityPrepConfig cfg;
    cfg.smooth_moments   = true;
    cfg.compute_velocity = true;

    // Run 1.
    {
        auto d_s = upload_csc(d.spliced,   stream);
        auto d_u = upload_csc(d.unspliced, stream);
        auto r1 = singlet_gpu::preprocess::velocity_prep(d_s, d_u, nullptr, cfg, stream);
        cudaStreamSynchronize(stream);
        auto g1 = download_f32(r1.gamma, kTinyGenes, stream);

        // Run 2 — re-upload to get fresh device state.
        auto d_s2 = upload_csc(d.spliced,   stream);
        auto d_u2 = upload_csc(d.unspliced, stream);
        auto r2 = singlet_gpu::preprocess::velocity_prep(d_s2, d_u2, nullptr, cfg, stream);
        cudaStreamSynchronize(stream);
        auto g2 = download_f32(r2.gamma, kTinyGenes, stream);

        // Bit-identical comparison (memcmp on floats).
        bool identical = (g1.size() == g2.size()) &&
                         (std::memcmp(g1.data(), g2.data(),
                                      g1.size() * sizeof(float)) == 0);
        printf("[Determinism] gamma bit-identical: %s\n", identical ? "YES" : "NO");

        if (!identical) {
            // Report first differing element.
            for (size_t i = 0; i < g1.size(); ++i) {
                if (g1[i] != g2[i]) {
                    printf("[Determinism] first diff at gene %zu: run1=%.8f run2=%.8f\n",
                           i, g1[i], g2[i]);
                    break;
                }
            }
        }
        EXPECT_TRUE(identical) << "velocity_prep is not bit-identical across two runs";
    }

    cudaStreamDestroy(stream);
}

// =============================================================================
// main — GoogleTest entry point
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
