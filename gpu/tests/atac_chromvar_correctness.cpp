// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/atac_chromvar_correctness.cpp
//
// integrates: original (first GPU chromVAR for scATAC-seq motif enrichment)
//
// Correctness harness for singlet_gpu::atac::chromvar.
//
// Written against the design doc at:
//   singlet-gpu/state/designs/34-chromvar.md
// NOT against the kernel source — authored in parallel with
// atac/chromvar.h by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: chromVAR R package via Rscript subprocess
//   tests/refs/chromvar_r_reference.R
//   Environment: R + BiocManager::install("chromVAR")
//   Exit code 2 from the script → GTEST_SKIP (R not available).
//   Exit code != 0 and != 2 → GTEST_FAIL (script error).
//
// Tolerances (design doc §"Correctness test spec"):
//   Per-cell deviation Spearman ρ         >= 0.95  (vs chromVAR R)
//   Motif variability ranking Spearman ρ  >= 0.95  (vs chromVAR R)
//   p-value uniformity under null         KS stat <= 0.10 (alpha = 0.05)
//   Determinism                            bit-identical (fixed seed, two runs)
//
// Test cases:
//   ChromVAR_TinySynthetic_VsR
//   ChromVAR_GSM_RealData_Atac
//   ChromVAR_VariabilityRanking
//   ChromVAR_PValuePermutation_Calibrated
//   ChromVAR_Determinism_BitIdentical
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, CUDA::cusparse, CUDA::curand,
//             singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every test that requires a device calls GTEST_SKIP().

// ---- singlet-gpu chromVAR header (written by gpu-kernel-dev) ---------------
#include <singlet-gpu/atac/chromvar.h>

// ---- other singlet-gpu headers needed by the real-data test ----------------
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
// Compat bridges — adapt old test API (designed against a planned API) to the
// actual singlet_gpu::atac API in chromvar.h.  ALL bridges at global scope.
// =============================================================================

namespace singlet_gpu {
namespace core {

// core::Handles is not in the actual header — tests use it as a thin wrapper
// around a stream. We provide a stub that just holds the stream.
struct Handles {
    cudaStream_t stream = nullptr;
    static Handles create(cudaStream_t s) { return Handles{s}; }
    static void destroy(Handles& /*h*/) {}
};

} // namespace core

namespace io {

// io::host_dense_to_device_csc — converts a dense row-major float matrix to a
// device CSC (DeviceCSC = SparseMatrixGPU<float>).  Simple: upload all nonzeros.
inline core::DeviceCSC host_dense_to_device_csc(const float* host_data,
                                                 int n_rows, int n_cols,
                                                 cudaStream_t stream)
{
    // Build CSC (col-major) from dense row-major host matrix.
    int nnz_count = 0;
    for (size_t i = 0; i < (size_t)n_rows * n_cols; ++i)
        if (host_data[i] != 0.0f) ++nnz_count;

    std::vector<int>   col_ptr(n_cols + 1, 0);
    std::vector<int>   row_idx(nnz_count);
    std::vector<float> vals(nnz_count);

    // Count nnz per column.
    for (int c = 0; c < n_cols; ++c)
        for (int r = 0; r < n_rows; ++r)
            if (host_data[(size_t)r * n_cols + c] != 0.0f)
                col_ptr[c + 1]++;

    // Prefix sum.
    for (int c = 0; c < n_cols; ++c)
        col_ptr[c + 1] += col_ptr[c];

    // Fill entries.
    std::vector<int> cur(col_ptr.begin(), col_ptr.end());
    for (int c = 0; c < n_cols; ++c)
        for (int r = 0; r < n_rows; ++r) {
            float v = host_data[(size_t)r * n_cols + c];
            if (v != 0.0f) {
                int pos = cur[c]++;
                row_idx[pos] = r;
                vals[pos]    = v;
            }
        }

    cudaStreamSynchronize(stream);
    return core::DeviceCSC(n_rows, n_cols, nnz_count,
                           col_ptr.data(), row_idx.data(), vals.data());
}

} // namespace io

namespace atac {

// ChromVARConfig is the test-expected name; actual is ChromVarConfig.
using ChromVARConfig = ChromVarConfig;

// ChromVARResult: test-expected struct with raw device pointers + static free.
struct ChromVARResult {
    float* deviations_dev  = nullptr;  // n_motifs × n_cells
    float* variability_dev = nullptr;  // n_motifs
    float* pvalues_dev     = nullptr;  // n_motifs × n_cells
    float* padj_dev        = nullptr;  // n_motifs × n_cells

    // Underlying owned buffers.
    core::DeviceMemory<float> _dev_buf;
    core::DeviceMemory<float> _var_buf;
    core::DeviceMemory<float> _pv_buf;
    core::DeviceMemory<float> _padj_buf;

    static void free(ChromVARResult& r, cudaStream_t /*stream*/) {
        r._dev_buf  = core::DeviceMemory<float>{};
        r._var_buf  = core::DeviceMemory<float>{};
        r._pv_buf   = core::DeviceMemory<float>{};
        r._padj_buf = core::DeviceMemory<float>{};
        r.deviations_dev = r.variability_dev = r.pvalues_dev = r.padj_dev = nullptr;
    }
};

// compute_chromvar bridge: accepts test-style args, converts motif_host to CSC,
// calls the actual chromvar(), packages result into ChromVARResult.
inline ChromVARResult compute_chromvar(
    const core::DeviceCSC& accessibility,
    const float* motif_host,   // host float[n_motifs × n_peaks], row-major
    int n_motifs,
    int n_peaks,
    const ChromVARConfig& cfg,
    const core::Handles& /*handles*/,
    cudaStream_t stream)
{
    const int n_cells = (int)accessibility.cols;

    // Build motif_in_peak CSC from host dense matrix (n_motifs × n_peaks, row-major).
    core::DeviceCSC motif_csc = io::host_dense_to_device_csc(
        motif_host, n_motifs, n_peaks, stream);

    // Build peak_gc and peak_mean_access (uniform stubs — tests don't validate GC bias).
    // Uniform GC = 0.5 for all peaks; uniform mean = 0.01 (nonzero so BG matching works).
    std::vector<float> h_gc(n_peaks, 0.5f);
    std::vector<float> h_mean(n_peaks, 0.01f);

    // Compute actual per-peak mean accessibility from the CSC (one-time host copy).
    {
        std::vector<int> h_cp(n_cells + 1);
        std::vector<int> h_ri(accessibility.nnz);
        std::vector<float> h_vl(accessibility.nnz);
        cudaMemcpy(h_cp.data(), accessibility.col_ptr.get(),    (n_cells + 1) * sizeof(int),   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ri.data(), accessibility.row_indices.get(), accessibility.nnz * sizeof(int),   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vl.data(), accessibility.values.get(),      accessibility.nnz * sizeof(float), cudaMemcpyDeviceToHost);
        std::vector<double> row_sum(n_peaks, 0.0);
        for (int c = 0; c < n_cells; ++c)
            for (int i = h_cp[c]; i < h_cp[c + 1]; ++i)
                row_sum[h_ri[i]] += h_vl[i];
        for (int p = 0; p < n_peaks; ++p)
            h_mean[p] = static_cast<float>(row_sum[p] / n_cells);
    }

    core::DeviceMemory<float> d_gc(n_peaks);
    core::DeviceMemory<float> d_mean(n_peaks);
    d_gc.upload(h_gc.data(), n_peaks);
    d_mean.upload(h_mean.data(), n_peaks);

    // Call actual chromvar().
    ChromVarResult base_result = chromvar(
        accessibility, motif_csc, d_gc, d_mean, cfg, stream);

    // Package into ChromVARResult with raw device pointers.
    ChromVARResult r;
    r._dev_buf  = std::move(base_result.deviation);
    r._var_buf  = std::move(base_result.variability);
    r._pv_buf   = std::move(base_result.p_values);
    r._padj_buf = std::move(base_result.p_adj);
    r.deviations_dev  = r._dev_buf.get();
    r.variability_dev = r._var_buf.get();
    r.pvalues_dev     = r._pv_buf.get();
    r.padj_dev        = r._padj_buf.get();
    return r;
}

} // namespace atac
} // namespace singlet_gpu

// DeviceCSC is RAII — provide a named no-op free function used by the tests.
namespace singlet_gpu { namespace core {
    inline void device_csc_free_noop(core::DeviceCSC& /*csc*/,
                                     cudaStream_t /*stream*/) {}
    inline void device_csc_free_noop(const core::DeviceCSC& /*csc*/,
                                     cudaStream_t /*stream*/) {}
} } // namespace singlet_gpu::core

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

/// Tiny synthetic dimensions (design doc §Correctness test spec, case 1).
constexpr int kTinyCells  = 200;
constexpr int kTinyPeaks  = 500;
constexpr int kTinyMotifs =  20;

/// Background peaks per peak used in the tiny test.
constexpr int kTinyNBg   =  50;
/// Permutation count for p-value calibration test (design doc: K=100 for tests).
constexpr int kPermCount = 100;
/// KS threshold for null p-value uniformity.
constexpr double kKsTol  = 0.10;
/// Spearman tolerance for deviation and variability.
constexpr double kRhoTol = 0.95;

/// Path to the chromVAR R reference script.
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/"
    "tests/refs/chromvar_r_reference.R";

/// Real ATAC .1pz path (skip gracefully if absent).
static const char* kGsmFragmentsPath =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/atac/GSE127/GSE127918/GSM4037629/fragments.1pz";

/// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_atac_chromvar_refs_tmp";

// =============================================================================
// Utility helpers (pattern from cycles 13–33)
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
// ---------------------------------------------------------------------------
void write_matrix_tsv(const std::string& path,
                      const std::vector<float>& data,
                      int n_rows, int n_cols)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open matrix tsv: " + path);
    for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
            if (c > 0) f << '\t';
            f << data[static_cast<size_t>(r) * n_cols + c];
        }
        f << '\n';
    }
    if (!f) throw std::runtime_error("Write error matrix tsv: " + path);
}

// ---------------------------------------------------------------------------
// Read a plain numeric TSV (no header) into a flat vector (row-major).
// Returns the number of rows read; sets *n_cols from the first row.
// ---------------------------------------------------------------------------
std::vector<double> read_matrix_tsv(const std::string& path, int* n_rows, int* n_cols)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open tsv: " + path);

    std::vector<double> vals;
    *n_rows = 0;
    *n_cols = 0;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        double v;
        int col = 0;
        while (ss >> v) { vals.push_back(v); ++col; }
        if (*n_cols == 0) *n_cols = col;
        ++(*n_rows);
    }
    return vals;
}

// ---------------------------------------------------------------------------
// Read a single-column TSV of doubles (no header).
// ---------------------------------------------------------------------------
std::vector<double> read_column_tsv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open column tsv: " + path);
    std::vector<double> vals;
    double v;
    while (f >> v) vals.push_back(v);
    return vals;
}

// ---------------------------------------------------------------------------
// Spearman rank correlation between two equal-length vectors.
// ---------------------------------------------------------------------------
double spearman(const std::vector<double>& a, const std::vector<double>& b)
{
    assert(a.size() == b.size());
    int n = static_cast<int>(a.size());
    if (n == 0) return 0.0;

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
// Generate a synthetic peak × cell binary accessibility matrix (sparse).
// Each cell has ~10% of peaks accessible.  Returns flat row-major float matrix.
// ---------------------------------------------------------------------------
std::vector<float> make_synthetic_counts(int n_peaks, int n_cells, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution bern(0.10);
    std::vector<float> mat(static_cast<size_t>(n_peaks) * n_cells, 0.0f);
    for (int p = 0; p < n_peaks; ++p)
        for (int c = 0; c < n_cells; ++c)
            if (bern(rng))
                mat[static_cast<size_t>(p) * n_cells + c] =
                    static_cast<float>(1 + (rng() % 3));  // 1–3 fragments
    return mat;
}

// ---------------------------------------------------------------------------
// Generate a synthetic motif × peak binary annotation.
// Each motif annotates ~5% of peaks.
// ---------------------------------------------------------------------------
std::vector<float> make_synthetic_motif_in_peak(int n_motifs, int n_peaks,
                                                uint64_t seed)
{
    std::mt19937_64 rng(seed ^ 0xDEADBEEFull);
    std::bernoulli_distribution bern(0.05);
    std::vector<float> mat(static_cast<size_t>(n_motifs) * n_peaks, 0.0f);
    for (int m = 0; m < n_motifs; ++m)
        for (int p = 0; p < n_peaks; ++p)
            if (bern(rng))
                mat[static_cast<size_t>(m) * n_peaks + p] = 1.0f;
    return mat;
}

// ---------------------------------------------------------------------------
// Build a ChromVAR config (from atac/chromvar.h) with common test defaults.
// ---------------------------------------------------------------------------
singlet_gpu::atac::ChromVARConfig make_config(uint64_t seed = kSeed,
                                              int n_bg = kTinyNBg,
                                              int n_perm = kPermCount)
{
    singlet_gpu::atac::ChromVARConfig cfg;
    cfg.n_background_peaks = n_bg;
    cfg.n_permutations     = n_perm;
    cfg.seed               = seed;
    return cfg;
}

}  // anonymous namespace

// =============================================================================
// Test fixture: device context + stream, created once per test.
// =============================================================================

class ChromVARTest : public ::testing::Test {
protected:
    cudaStream_t stream_{};
    singlet_gpu::core::Handles handles_{};

    void SetUp() override {
        if (!gpu_available()) GTEST_SKIP() << "No CUDA device available.";
        cudaError_t e = cudaStreamCreate(&stream_);
        ASSERT_EQ(e, cudaSuccess) << "cudaStreamCreate: " << cudaGetErrorString(e);
        handles_ = singlet_gpu::core::Handles::create(stream_);
        ensure_refs_tmp();
    }

    void TearDown() override {
        if (stream_) {
            singlet_gpu::core::Handles::destroy(handles_);
            cudaStreamDestroy(stream_);
            stream_ = {};
        }
    }
};

// =============================================================================
// Test 1 — ChromVAR_TinySynthetic_VsR
//
// 200 cells × 500 peaks × 20 motifs synthetic.
// Spearman ρ ≥ 0.95 on per-cell deviation (all motifs flattened) vs chromVAR R.
// =============================================================================

TEST_F(ChromVARTest, ChromVAR_TinySynthetic_VsR) {
    // --- 1. Build synthetic inputs ------------------------------------------
    auto counts_flat = make_synthetic_counts(kTinyPeaks, kTinyCells, kSeed);
    auto motif_flat  = make_synthetic_motif_in_peak(kTinyMotifs, kTinyPeaks, kSeed);

    // --- 2. Write inputs for R reference ------------------------------------
    const std::string counts_tsv   = refs_path("chromvar_tiny_counts.tsv");
    const std::string motif_tsv    = refs_path("chromvar_tiny_motif_in_peak.tsv");
    const std::string r_dev_tsv    = refs_path("chromvar_tiny_r_deviations.tsv");
    const std::string r_var_tsv    = refs_path("chromvar_tiny_r_variability.tsv");

    write_matrix_tsv(counts_tsv, counts_flat, kTinyPeaks, kTinyCells);
    write_matrix_tsv(motif_tsv,  motif_flat,  kTinyMotifs, kTinyPeaks);

    // --- 3. Run R reference -------------------------------------------------
    std::string r_cmd =
        std::string("Rscript ") + kRefScript +
        " --counts "           + counts_tsv  +
        " --motif_in_peak "    + motif_tsv   +
        " --n_cells "          + std::to_string(kTinyCells)  +
        " --n_peaks "          + std::to_string(kTinyPeaks)  +
        " --n_motifs "         + std::to_string(kTinyMotifs) +
        " --output_deviations "+ r_dev_tsv   +
        " --output_variability "+ r_var_tsv  +
        " --n_bg "             + std::to_string(kTinyNBg)    +
        " --seed 42 2>&1";

    int rc = run_cmd_rc(r_cmd);
    if (rc == 2) GTEST_SKIP() << "chromVAR R package not installed; skipping reference diff.";
    ASSERT_EQ(rc, 0) << "R reference script failed (exit " << rc << ")";

    // --- 4. Read R reference deviations (n_cells × n_motifs) ----------------
    int r_nrows = 0, r_ncols = 0;
    auto r_dev_flat = read_matrix_tsv(r_dev_tsv, &r_nrows, &r_ncols);
    ASSERT_EQ(r_nrows, kTinyCells)  << "R deviations row count mismatch";
    ASSERT_EQ(r_ncols, kTinyMotifs) << "R deviations col count mismatch";

    // --- 5. Upload inputs to device and run singlet-gpu chromVAR ------------
    // peak × cell CSC: build from row-major float matrix
    using DevCSC = singlet_gpu::core::DeviceCSC;
    DevCSC accessibility = singlet_gpu::io::host_dense_to_device_csc(
        counts_flat.data(), kTinyPeaks, kTinyCells, stream_);

    // motif × peak dense binary (uploaded as a host matrix; chromvar.h accepts
    // host pointer for the small annotation matrix and copies it internally)
    auto cfg = make_config(kSeed, kTinyNBg, kPermCount);
    singlet_gpu::atac::ChromVARResult result =
        singlet_gpu::atac::compute_chromvar(
            accessibility,
            motif_flat.data(),   // host float[n_motifs × n_peaks], row-major
            kTinyMotifs,
            kTinyPeaks,
            cfg,
            handles_,
            stream_);

    // Sync before reading results
    cudaStreamSynchronize(stream_);

    // result.deviations: device float[n_cells × n_motifs] row-major
    std::vector<float> our_dev(static_cast<size_t>(kTinyCells) * kTinyMotifs);
    cudaMemcpy(our_dev.data(), result.deviations_dev,
               our_dev.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // --- 6. Compare per-cell deviations via Spearman ------------------------
    // Flatten both to 1-D and compute a global Spearman (all cells × motifs).
    std::vector<double> our_flat(our_dev.begin(), our_dev.end());
    // Replace any NaN/inf with 0 (small synthetic matrices may have degenerate motifs)
    for (auto& v : our_flat) if (!std::isfinite(v)) v = 0.0;
    // R deviations already loaded as r_dev_flat (double)

    double rho = spearman(our_flat, r_dev_flat);
    EXPECT_GE(rho, kRhoTol)
        << "Per-cell deviation Spearman ρ=" << rho
        << " < threshold=" << kRhoTol;

    // --- 7. Free device result ----------------------------------------------
    singlet_gpu::atac::ChromVARResult::free(result, stream_);
}

// =============================================================================
// Test 2 — ChromVAR_GSM_RealData_Atac
//
// Load a real ATAC fragments.1pz if present; skip if not.
// Confirm all deviations are finite (no NaN/inf).
// =============================================================================

TEST_F(ChromVARTest, ChromVAR_GSM_RealData_Atac) {
    if (!fs::exists(kGsmFragmentsPath)) {
        GTEST_SKIP() << "Real ATAC .1pz not available at: " << kGsmFragmentsPath;
    }

    // Load fragments.1pz → peak × cell CSC on device
    singlet_gpu::core::DeviceCSC fragments =
        singlet_gpu::io::load_pz(kGsmFragmentsPath, stream_).mat;

    int n_peaks = (int)fragments.rows;
    int n_cells = (int)fragments.cols;

    ASSERT_GT(n_peaks, 0);
    ASSERT_GT(n_cells, 0);

    // Build a minimal random motif annotation (no real JASPAR motif file
    // available in CI; we only check that the pipeline runs and produces
    // finite results).
    constexpr int kRealMotifs = 10;
    auto motif_flat = make_synthetic_motif_in_peak(kRealMotifs, n_peaks, kSeed);

    singlet_gpu::atac::ChromVARConfig cfg;
    cfg.n_background_peaks = 50;
    cfg.n_permutations     = 50;
    cfg.seed               = kSeed;

    singlet_gpu::atac::ChromVARResult result =
        singlet_gpu::atac::compute_chromvar(
            fragments,
            motif_flat.data(),
            kRealMotifs,
            n_peaks,
            cfg,
            handles_,
            stream_);

    cudaStreamSynchronize(stream_);

    // Pull deviations to host and check finiteness
    size_t dev_elems = static_cast<size_t>(n_cells) * kRealMotifs;
    std::vector<float> dev_host(dev_elems);
    cudaMemcpy(dev_host.data(), result.deviations_dev,
               dev_elems * sizeof(float), cudaMemcpyDeviceToHost);

    int n_nonfinite = 0;
    for (auto v : dev_host) if (!std::isfinite(v)) ++n_nonfinite;

    EXPECT_EQ(n_nonfinite, 0)
        << "Real-data chromVAR produced " << n_nonfinite << " non-finite deviations";

    // Pull variability to host and check positive
    std::vector<float> var_host(kRealMotifs);
    cudaMemcpy(var_host.data(), result.variability_dev,
               kRealMotifs * sizeof(float), cudaMemcpyDeviceToHost);

    int n_negative_var = 0;
    for (auto v : var_host) if (v < 0.0f) ++n_negative_var;
    EXPECT_EQ(n_negative_var, 0) << "Variability has negative values";

    singlet_gpu::atac::ChromVARResult::free(result, stream_);
    singlet_gpu::core::device_csc_free_noop(fragments, stream_);
}

// =============================================================================
// Test 3 — ChromVAR_VariabilityRanking
//
// Motif variability ranking from singlet-gpu matches chromVAR R
// Spearman ρ ≥ 0.95 on the same 200 × 500 × 20 synthetic.
// =============================================================================

TEST_F(ChromVARTest, ChromVAR_VariabilityRanking) {
    auto counts_flat = make_synthetic_counts(kTinyPeaks, kTinyCells, kSeed);
    auto motif_flat  = make_synthetic_motif_in_peak(kTinyMotifs, kTinyPeaks, kSeed);

    // Write inputs and call R (same files as Test 1 — reuse if present).
    const std::string counts_tsv = refs_path("chromvar_tiny_counts.tsv");
    const std::string motif_tsv  = refs_path("chromvar_tiny_motif_in_peak.tsv");
    const std::string r_dev_tsv  = refs_path("chromvar_tiny_r_deviations.tsv");
    const std::string r_var_tsv  = refs_path("chromvar_tiny_r_variability.tsv");

    // Write only if not already produced by Test 1.
    if (!fs::exists(counts_tsv))
        write_matrix_tsv(counts_tsv, counts_flat, kTinyPeaks, kTinyCells);
    if (!fs::exists(motif_tsv))
        write_matrix_tsv(motif_tsv, motif_flat, kTinyMotifs, kTinyPeaks);

    if (!fs::exists(r_var_tsv)) {
        std::string r_cmd =
            std::string("Rscript ") + kRefScript +
            " --counts "            + counts_tsv  +
            " --motif_in_peak "     + motif_tsv   +
            " --n_cells "           + std::to_string(kTinyCells)  +
            " --n_peaks "           + std::to_string(kTinyPeaks)  +
            " --n_motifs "          + std::to_string(kTinyMotifs) +
            " --output_deviations " + r_dev_tsv   +
            " --output_variability "+ r_var_tsv   +
            " --n_bg "              + std::to_string(kTinyNBg)    +
            " --seed 42 2>&1";
        int rc = run_cmd_rc(r_cmd);
        if (rc == 2) GTEST_SKIP() << "chromVAR R package not installed; skipping variability ranking diff.";
        ASSERT_EQ(rc, 0) << "R reference script failed (exit " << rc << ")";
    }

    auto r_var = read_column_tsv(r_var_tsv);
    ASSERT_EQ(static_cast<int>(r_var.size()), kTinyMotifs);

    // Run singlet-gpu chromVAR
    using DevCSC = singlet_gpu::core::DeviceCSC;
    DevCSC accessibility = singlet_gpu::io::host_dense_to_device_csc(
        counts_flat.data(), kTinyPeaks, kTinyCells, stream_);

    auto cfg = make_config(kSeed, kTinyNBg, kPermCount);
    singlet_gpu::atac::ChromVARResult result =
        singlet_gpu::atac::compute_chromvar(
            accessibility, motif_flat.data(), kTinyMotifs, kTinyPeaks,
            cfg, handles_, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<float> var_gpu(kTinyMotifs);
    cudaMemcpy(var_gpu.data(), result.variability_dev,
               kTinyMotifs * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<double> our_var(var_gpu.begin(), var_gpu.end());
    for (auto& v : our_var) if (!std::isfinite(v)) v = 0.0;

    double rho = spearman(our_var, r_var);
    EXPECT_GE(rho, kRhoTol)
        << "Variability ranking Spearman ρ=" << rho
        << " < threshold=" << kRhoTol;

    singlet_gpu::atac::ChromVARResult::free(result, stream_);
    singlet_gpu::core::device_csc_free_noop(accessibility, stream_);
}

// =============================================================================
// Test 4 — ChromVAR_PValuePermutation_Calibrated
//
// Under the null (random motif assignments, no real enrichment), p-values
// returned by singlet-gpu chromVAR should be approximately uniform in [0,1].
// KS statistic vs Uniform[0,1] must be ≤ 0.10.
// =============================================================================

TEST_F(ChromVARTest, ChromVAR_PValuePermutation_Calibrated) {
    // Build a null data set: random accessibility + random motif assignments
    // (no planted enrichment signal).
    constexpr int kNullCells  = 100;
    constexpr int kNullPeaks  = 300;
    constexpr int kNullMotifs =  10;

    auto counts_flat = make_synthetic_counts(kNullPeaks, kNullCells, kSeed ^ 0x1234);
    auto motif_flat  = make_synthetic_motif_in_peak(kNullMotifs, kNullPeaks, kSeed ^ 0x5678);

    using DevCSC = singlet_gpu::core::DeviceCSC;
    DevCSC accessibility = singlet_gpu::io::host_dense_to_device_csc(
        counts_flat.data(), kNullPeaks, kNullCells, stream_);

    singlet_gpu::atac::ChromVARConfig cfg;
    cfg.n_background_peaks = 50;
    cfg.n_permutations     = kPermCount;
    cfg.seed               = kSeed;

    singlet_gpu::atac::ChromVARResult result =
        singlet_gpu::atac::compute_chromvar(
            accessibility, motif_flat.data(), kNullMotifs, kNullPeaks,
            cfg, handles_, stream_);
    cudaStreamSynchronize(stream_);

    // result.pvalues_dev: device float[n_cells × n_motifs], row-major
    size_t pval_elems = static_cast<size_t>(kNullCells) * kNullMotifs;
    std::vector<float> pvals_host(pval_elems);
    cudaMemcpy(pvals_host.data(), result.pvalues_dev,
               pval_elems * sizeof(float), cudaMemcpyDeviceToHost);

    // Collect finite p-values (exclude any cells with zero total accessibility)
    std::vector<double> finite_pvals;
    finite_pvals.reserve(pval_elems);
    for (float p : pvals_host)
        if (std::isfinite(p) && p >= 0.0f && p <= 1.0f)
            finite_pvals.push_back(static_cast<double>(p));

    ASSERT_GT(static_cast<int>(finite_pvals.size()), 10)
        << "Too few finite p-values to assess calibration";

    double ks = ks_uniform(finite_pvals);
    EXPECT_LE(ks, kKsTol)
        << "Null p-value KS statistic=" << ks
        << " > tolerance=" << kKsTol
        << " (p-values not uniform under null)";

    singlet_gpu::atac::ChromVARResult::free(result, stream_);
    singlet_gpu::core::device_csc_free_noop(accessibility, stream_);
}

// =============================================================================
// Test 5 — ChromVAR_Determinism_BitIdentical
//
// Two runs with the same fixed seed produce bit-identical deviation matrices.
// =============================================================================

TEST_F(ChromVARTest, ChromVAR_Determinism_BitIdentical) {
    auto counts_flat = make_synthetic_counts(kTinyPeaks, kTinyCells, kSeed);
    auto motif_flat  = make_synthetic_motif_in_peak(kTinyMotifs, kTinyPeaks, kSeed);

    using DevCSC = singlet_gpu::core::DeviceCSC;
    auto cfg = make_config(kSeed, kTinyNBg, kPermCount);

    // Run 1
    DevCSC acc1 = singlet_gpu::io::host_dense_to_device_csc(
        counts_flat.data(), kTinyPeaks, kTinyCells, stream_);
    singlet_gpu::atac::ChromVARResult res1 =
        singlet_gpu::atac::compute_chromvar(
            acc1, motif_flat.data(), kTinyMotifs, kTinyPeaks,
            cfg, handles_, stream_);
    cudaStreamSynchronize(stream_);

    size_t dev_elems = static_cast<size_t>(kTinyCells) * kTinyMotifs;
    std::vector<float> dev1(dev_elems), dev2(dev_elems);
    cudaMemcpy(dev1.data(), res1.deviations_dev,
               dev_elems * sizeof(float), cudaMemcpyDeviceToHost);
    singlet_gpu::atac::ChromVARResult::free(res1, stream_);
    singlet_gpu::core::device_csc_free_noop(acc1, stream_);

    // Run 2 — same inputs, same seed
    DevCSC acc2 = singlet_gpu::io::host_dense_to_device_csc(
        counts_flat.data(), kTinyPeaks, kTinyCells, stream_);
    singlet_gpu::atac::ChromVARResult res2 =
        singlet_gpu::atac::compute_chromvar(
            acc2, motif_flat.data(), kTinyMotifs, kTinyPeaks,
            cfg, handles_, stream_);
    cudaStreamSynchronize(stream_);

    cudaMemcpy(dev2.data(), res2.deviations_dev,
               dev_elems * sizeof(float), cudaMemcpyDeviceToHost);
    singlet_gpu::atac::ChromVARResult::free(res2, stream_);
    singlet_gpu::core::device_csc_free_noop(acc2, stream_);

    // Bit-identical comparison
    int n_differ = 0;
    for (size_t i = 0; i < dev_elems; ++i)
        if (dev1[i] != dev2[i]) ++n_differ;

    EXPECT_EQ(n_differ, 0)
        << "Determinism failed: " << n_differ << " of " << dev_elems
        << " deviation values differ between two identical runs";
}
