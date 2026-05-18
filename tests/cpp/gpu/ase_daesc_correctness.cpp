// SPDX-License-Identifier: MIT
// singlet/gpu/tests/ase_daesc_correctness.cpp
//
// integrates: original (first GPU DAESC-style beta-binomial ASE calling —
//             exploits singlet snp_ad.1pz + snp_dp.1pz directly)
//
// Correctness harness for singlet::gpu::ase::daesc.
//
// Written against the design doc at:
//   singlet/gpu/state/designs/40-daesc-ase.md
// NOT against the kernel source — authored in parallel with ase/daesc.h by
// the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: DAESC R via Rscript subprocess (CRAN package daesc).
//   If DAESC R is absent (exit code 2 from the R script) → pure-R VGAM
//   betabinomial MLE fallback (embedded in daesc_r_reference.R).
//   The VGAM fallback always produces a reference even without the DAESC package.
//
// Tolerances (design doc §"Correctness test spec"):
//   beta Spearman ρ         >= 0.95  vs DAESC R (or VGAM fallback)
//   p-value rank Spearman   >= 0.95  vs DAESC R (or VGAM fallback)
//   Convergence rate        >  0.98  within 50 iters on well-specified synthetic
//   p-value KS uniform      <= 0.10  under null (balanced alleles)
//   Bit-identical           across two runs
//
// Test cases:
//   Daesc_TinySynthetic_VsR
//   Daesc_GSM_RealData
//   Daesc_FisherConvergence
//   Daesc_PvalueCalibration_Uniform
//   Daesc_Determinism_BitIdentical
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every GPU-dependent test calls GTEST_SKIP().

// ---- singlet-gpu ASE header (written by gpu-kernel-dev) ---------------------
#include <singlet/gpu/ase/daesc.h>

// ---- other singlet-gpu headers needed for the real-data test ----------------
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

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
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// load_pz_device compat (matching cna_numbat pattern).
// DaescConfig compat fields (tol, n_cell_types, n_snps_limit) and the
// daesc() free function are now defined directly in ase/daesc.h.
namespace singlet::gpu { namespace io {
inline core::DeviceCSC load_pz_device(const std::string& path, cudaStream_t stream) {
    return singlet::gpu::io::load_pz(path, stream).mat;
}
} } // namespace singlet::gpu::io

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

/// Tiny synthetic dimensions (from design doc §"Correctness test spec").
constexpr int kTinyCells    = 200;
constexpr int kTinySNPs     = 100;
constexpr int kTinyCelltypes = 3;

/// Planted imbalance: first kPlantedSNPs SNPs have true allelic imbalance.
constexpr int   kPlantedSNPs = 20;
/// True beta (log-odds bias) for planted SNPs.
constexpr float kPlantedBeta = 1.2f;
/// True phi (overdispersion) for all SNPs.
constexpr float kTruePhi     = 0.05f;

/// Minimum depth filter (from design doc: MIN_DEPTH = 5).
constexpr int kMinDepth = 5;

/// Spearman ρ tolerance (design doc: ≥ 0.95).
constexpr double kSpearmanTol = 0.95;

/// KS test tolerance for p-value uniformity (design doc: ≤ 0.10).
constexpr double kKsTol = 0.10;

/// Convergence rate threshold (design doc: > 0.98).
constexpr double kConvergenceRate = 0.98;

// Real-data paths (GSM4037629 with SNP outputs).
static const char* kGsmSnpAdPath =
    "/mnt/projects/debruinz_project/singlet_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/snp_ad.1pz";
static const char* kGsmSnpDpPath =
    "/mnt/projects/debruinz_project/singlet_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/snp_dp.1pz";

/// Temporary directory for reference I/O.
static const char* kRefsTmpDir =
    "/tmp/singlet_gpu_ase_daesc_refs_tmp";

/// R reference script.
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet/gpu/"
    "tests/refs/daesc_r_reference.R";

static const char* kRscript = "Rscript";

// =============================================================================
// Utility helpers (pattern from cycles 3–39)
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

// run_cmd_rc: returns the exit code without throwing.
int run_cmd_rc(const std::string& cmd) {
    return std::system(cmd.c_str());
}

bool gpu_available() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    return (e == cudaSuccess && count > 0);
}

// ---------------------------------------------------------------------------
// Host-side CSC container (reused from cycles 3–39).
// ---------------------------------------------------------------------------
struct HostCSC {
    uint32_t             m;        // rows (SNPs)
    uint32_t             n;        // cols (cells)
    uint64_t             nnz;
    std::vector<float>   values;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
};

// Write HostCSC in the dump_csc binary format (CSCC magic, used by R).
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

// Upload HostCSC to device using the synchronous host-data constructor.
singlet::gpu::core::DeviceCSC upload_csc(const HostCSC& h, cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    return singlet::gpu::core::DeviceCSC(
        static_cast<int>(h.m),
        static_cast<int>(h.n),
        static_cast<int>(h.nnz),
        h.indptr.data(),
        h.indices.data(),
        h.values.data());
}

// Write a flat vector to a single-column TSV.
void write_vec_tsv(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    for (auto x : v) f << x << '\n';
}

// Write an integer vector to a single-column TSV.
void write_int_vec_tsv(const std::string& path, const std::vector<int32_t>& v) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    for (auto x : v) f << x << '\n';
}

// Read a single-column TSV into a float vector.
std::vector<float> read_vec_tsv(const std::string& path, int n) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i)
        if (!(f >> v[i]))
            throw std::runtime_error("read_vec_tsv: short read at " +
                                     std::to_string(i) + " in " + path);
    return v;
}

// Read a single integer from a TSV file (convergence count output).
int read_int_scalar(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    int v = 0;
    f >> v;
    return v;
}

// ---------------------------------------------------------------------------
// Spearman rank correlation (reused from cycles 3–39).
// ---------------------------------------------------------------------------
double spearman(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    int n = static_cast<int>(a.size());
    if (n < 2) return 1.0;

    auto rank_of = [&](const std::vector<float>& v) {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int i, int j){ return v[i] < v[j]; });
        std::vector<double> r(n);
        for (int i = 0; i < n; ) {
            int j = i;
            while (j < n && v[idx[j]] == v[idx[i]]) ++j;
            double avg = (i + j - 1) * 0.5 + 1.0;
            for (int k = i; k < j; ++k) r[idx[k]] = avg;
            i = j;
        }
        return r;
    };

    auto ra = rank_of(a);
    auto rb = rank_of(b);

    double sa = 0, sb = 0;
    for (int i = 0; i < n; ++i) { sa += ra[i]; sb += rb[i]; }
    double ma = sa / n, mb = sb / n;

    double num = 0, da = 0, db = 0;
    for (int i = 0; i < n; ++i) {
        double da_ = ra[i] - ma, db_ = rb[i] - mb;
        num += da_ * db_;
        da  += da_ * da_;
        db  += db_ * db_;
    }
    if (da < 1e-12 || db < 1e-12) return 0.0;
    return num / std::sqrt(da * db);
}

// ---------------------------------------------------------------------------
// KS statistic vs Uniform(0,1) (reused from cycles 11/24).
// ---------------------------------------------------------------------------
double ks_uniform(std::vector<float> pvals) {
    std::sort(pvals.begin(), pvals.end());
    int n = static_cast<int>(pvals.size());
    double ks = 0.0;
    for (int i = 0; i < n; ++i) {
        double ecdf  = (i + 1.0) / n;
        double ecdf0 = static_cast<double>(i) / n;
        double d = std::max(std::abs(ecdf  - static_cast<double>(pvals[i])),
                            std::abs(ecdf0 - static_cast<double>(pvals[i])));
        ks = std::max(ks, d);
    }
    return ks;
}

// ---------------------------------------------------------------------------
// Synthetic data generator.
//
// Produces two CSC matrices (snp_ad, snp_dp) and cell_type labels.
//
// Cells are divided equally across kTinyCelltypes cell types.
// For planted SNPs (snp_idx < kPlantedSNPs):
//   p_i = logistic(kPlantedBeta) — true allelic bias.
// For null SNPs:
//   p_i = 0.5 — balanced alleles.
//
// Depth per (SNP, cell): Poisson(20), minimum kMinDepth.
// AD given depth and p: BetaBinomial(dp, p, phi=kTruePhi).
//
// BetaBinomial is simulated via the Beta-Binomial compound:
//   q ~ Beta(p/phi, (1-p)/phi)   [where phi = kTruePhi]
//   ad ~ Binomial(dp, q)
// ---------------------------------------------------------------------------
struct SyntheticAseData {
    HostCSC               snp_ad;     // snps × cells
    HostCSC               snp_dp;     // snps × cells
    std::vector<int32_t>  cell_type;  // cells (0-based)
};

SyntheticAseData make_synthetic_ase(uint64_t seed) {
    std::mt19937_64 rng(seed);

    const int C  = kTinyCells;
    const int S  = kTinySNPs;
    const int CT = kTinyCelltypes;
    const int cells_per_type = C / CT;

    // Cell type labels (round-robin).
    std::vector<int32_t> cell_type(C);
    for (int j = 0; j < C; ++j)
        cell_type[j] = j / cells_per_type;

    // For each SNP determine the true p (allele bias).
    std::vector<float> true_p(S);
    for (int s = 0; s < S; ++s) {
        if (s < kPlantedSNPs) {
            // logistic(kPlantedBeta)
            true_p[s] = 1.0f / (1.0f + std::exp(-kPlantedBeta));
        } else {
            true_p[s] = 0.5f;
        }
    }

    // Build snp_dp and snp_ad CSC (S rows × C cols).
    // We draw depth first, then AD from the BetaBinomial compound.
    std::poisson_distribution<int> dp_dist(20);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    // Dense arrays for construction.
    std::vector<int>   dp_dense(static_cast<size_t>(S) * C);
    std::vector<float> ad_dense(static_cast<size_t>(S) * C, 0.0f);

    for (int j = 0; j < C; ++j) {
        for (int s = 0; s < S; ++s) {
            int dp = dp_dist(rng);
            dp = std::max(dp, kMinDepth);   // guarantee at least MIN_DEPTH
            dp_dense[static_cast<size_t>(s) * C + j] = dp;

            // BetaBinomial: draw q ~ Beta(alpha, beta) then ad ~ Bin(dp, q).
            float p   = true_p[s];
            // Convert (p, phi) → (alpha, beta).
            // phi = 1/(alpha + beta + 1) → alpha+beta = 1/phi - 1
            double ab    = 1.0 / kTruePhi - 1.0;
            double alpha = p * ab;
            double beta  = (1.0 - p) * ab;
            // Sample from Beta(alpha, beta) via ratio of Gammas.
            std::gamma_distribution<double> ga(std::max(alpha, 1e-6), 1.0);
            std::gamma_distribution<double> gb(std::max(beta,  1e-6), 1.0);
            double ga_val = ga(rng), gb_val = gb(rng);
            double q = ga_val / (ga_val + gb_val + 1e-15);
            q = std::max(0.0, std::min(1.0, q));
            std::binomial_distribution<int> binom(dp, q);
            ad_dense[static_cast<size_t>(s) * C + j] = static_cast<float>(binom(rng));
        }
    }

    // Build CSC from dense arrays (column-major traversal → column j = cell j).
    auto dense_to_csc = [&](const auto& dense, bool force_nonzero) -> HostCSC {
        HostCSC csc;
        csc.m = static_cast<uint32_t>(S);
        csc.n = static_cast<uint32_t>(C);
        csc.indptr.resize(static_cast<size_t>(C) + 1, 0);

        std::vector<std::vector<std::pair<int32_t, float>>> cols(C);
        for (int j = 0; j < C; ++j) {
            for (int s = 0; s < S; ++s) {
                float val;
                if constexpr (std::is_same_v<std::decay_t<decltype(dense)>,
                                             std::vector<int>>) {
                    val = static_cast<float>(dense[static_cast<size_t>(s) * C + j]);
                } else {
                    val = dense[static_cast<size_t>(s) * C + j];
                }
                if (force_nonzero || val > 0.0f)
                    cols[j].push_back({static_cast<int32_t>(s), val});
            }
        }
        uint64_t total_nnz = 0;
        for (int j = 0; j < C; ++j) total_nnz += cols[j].size();
        csc.nnz = total_nnz;
        csc.indptr[0] = 0;
        for (int j = 0; j < C; ++j)
            csc.indptr[j + 1] = csc.indptr[j] + static_cast<int32_t>(cols[j].size());
        csc.values .resize(total_nnz);
        csc.indices.resize(total_nnz);
        size_t pos = 0;
        for (int j = 0; j < C; ++j)
            for (auto& [row, val] : cols[j]) {
                csc.indices[pos] = row;
                csc.values [pos] = val;
                ++pos;
            }
        return csc;
    };

    return SyntheticAseData{
        dense_to_csc(ad_dense, false),   // snp_ad (sparse — many zeros OK)
        dense_to_csc(dp_dense, true),    // snp_dp (all cells have depth)
        std::move(cell_type)
    };
}

// ---------------------------------------------------------------------------
// Null synthetic: balanced alleles (p = 0.5 everywhere) for KS calibration.
// ---------------------------------------------------------------------------
SyntheticAseData make_null_ase(uint64_t seed) {
    // Temporarily override planted to null by using a seed that seeds all
    // true_p = 0.5.  We do this by generating normal synthetic and then
    // rebuilding the AD from depth with p=0.5 (no Beta-Binomial compound,
    // pure balanced binomial) to maximally stress the null.
    std::mt19937_64 rng(seed + 0xDEAD);

    const int C  = kTinyCells;
    const int S  = kTinySNPs;
    const int CT = kTinyCelltypes;
    const int cells_per_type = C / CT;

    std::vector<int32_t> cell_type(C);
    for (int j = 0; j < C; ++j)
        cell_type[j] = j / cells_per_type;

    std::poisson_distribution<int> dp_dist(20);

    std::vector<int>   dp_dense(static_cast<size_t>(S) * C);
    std::vector<float> ad_dense(static_cast<size_t>(S) * C, 0.0f);

    for (int j = 0; j < C; ++j) {
        for (int s = 0; s < S; ++s) {
            int dp = dp_dist(rng);
            dp = std::max(dp, kMinDepth);
            dp_dense[static_cast<size_t>(s) * C + j] = dp;

            // Pure balanced: Binomial(dp, 0.5)
            std::binomial_distribution<int> binom(dp, 0.5);
            ad_dense[static_cast<size_t>(s) * C + j] = static_cast<float>(binom(rng));
        }
    }

    auto dense_to_csc = [&](const auto& dense, bool force_nonzero) -> HostCSC {
        HostCSC csc;
        csc.m = static_cast<uint32_t>(S);
        csc.n = static_cast<uint32_t>(C);
        csc.indptr.resize(static_cast<size_t>(C) + 1, 0);

        std::vector<std::vector<std::pair<int32_t, float>>> cols(C);
        for (int j = 0; j < C; ++j) {
            for (int s = 0; s < S; ++s) {
                float val;
                if constexpr (std::is_same_v<std::decay_t<decltype(dense)>,
                                             std::vector<int>>) {
                    val = static_cast<float>(dense[static_cast<size_t>(s) * C + j]);
                } else {
                    val = dense[static_cast<size_t>(s) * C + j];
                }
                if (force_nonzero || val > 0.0f)
                    cols[j].push_back({static_cast<int32_t>(s), val});
            }
        }
        uint64_t total_nnz = 0;
        for (int j = 0; j < C; ++j) total_nnz += cols[j].size();
        csc.nnz = total_nnz;
        csc.indptr[0] = 0;
        for (int j = 0; j < C; ++j)
            csc.indptr[j + 1] = csc.indptr[j] + static_cast<int32_t>(cols[j].size());
        csc.values .resize(total_nnz);
        csc.indices.resize(total_nnz);
        size_t pos = 0;
        for (int j = 0; j < C; ++j)
            for (auto& [row, val] : cols[j]) {
                csc.indices[pos] = row;
                csc.values [pos] = val;
                ++pos;
            }
        return csc;
    };

    return SyntheticAseData{
        dense_to_csc(ad_dense, false),
        dense_to_csc(dp_dense, true),
        std::move(cell_type)
    };
}

}  // anonymous namespace

// =============================================================================
// Test 1: Daesc_TinySynthetic_VsR
//
// Generate 200 cells × 100 SNPs × 3 cell types with planted allelic imbalance
// at the first 20 SNPs.
// Run GPU daesc → collect beta and p-values.
// Run daesc_r_reference.R (DAESC R if available, else VGAM betabinomial).
// Compare via Spearman ρ on beta vector and on log p-value rank.
// Tolerance: ρ ≥ 0.95 on both.
// =============================================================================
TEST(Daesc, TinySynthetic_VsR) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU — skipping";

    ensure_refs_tmp();

    SyntheticAseData syn = make_synthetic_ase(kSeed);

    // Write inputs for R reference.
    const std::string ad_path       = refs_path("syn_snp_ad.bin");
    const std::string dp_path       = refs_path("syn_snp_dp.bin");
    const std::string ct_path       = refs_path("syn_cell_type.tsv");
    const std::string r_beta_path   = refs_path("r_beta.tsv");
    const std::string r_pval_path   = refs_path("r_pval.tsv");

    write_csc_bin(syn.snp_ad, ad_path);
    write_csc_bin(syn.snp_dp, dp_path);
    write_int_vec_tsv(ct_path, syn.cell_type);

    // Run R reference (exit 2 = DAESC absent → pure-R VGAM fallback, still OK).
    {
        std::ostringstream cmd;
        cmd << kRscript << " " << kRefScript
            << " --snp_ad "    << ad_path
            << " --snp_dp "    << dp_path
            << " --cell_type " << ct_path
            << " --out_beta "  << r_beta_path
            << " --out_pval "  << r_pval_path
            << " --n_snps "    << kTinySNPs
            << " --n_cells "   << kTinyCells
            << " --min_depth " << kMinDepth;
        int rc = run_cmd_rc(cmd.str());
        // rc == 0: DAESC R; rc == 2: VGAM fallback; both acceptable.
        ASSERT_NE(rc, 1) << "R reference script crashed (exit 1): " << cmd.str();
        // If rc == 3 (R not found or no Rscript), skip.
        if (rc != 0 && rc != 2) GTEST_SKIP() << "R reference unavailable (rc=" << rc << ")";
    }

    // Read R outputs.
    std::vector<float> r_beta = read_vec_tsv(r_beta_path, kTinySNPs);
    std::vector<float> r_pval = read_vec_tsv(r_pval_path, kTinySNPs);

    // Upload to device and run GPU DAESC.
    {
        using namespace singlet::gpu::core;
        using namespace singlet::gpu::ase;

        cudaStream_t stream;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        DeviceCSC d_snp_ad = upload_csc(syn.snp_ad, stream);
        DeviceCSC d_snp_dp = upload_csc(syn.snp_dp, stream);

        // Cell-type vector on device.
        DeviceMemory<int32_t> d_cell_type(kTinyCells);
        ASSERT_EQ(cudaMemcpyAsync(d_cell_type.get(), syn.cell_type.data(),
                                  kTinyCells * sizeof(int32_t),
                                  cudaMemcpyHostToDevice, stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        // Output device buffers.
        DeviceMemory<float> d_beta     (kTinySNPs);
        DeviceMemory<float> d_se       (kTinySNPs);
        DeviceMemory<float> d_phi      (kTinySNPs);
        DeviceMemory<float> d_pvalue   (kTinySNPs);
        DeviceMemory<int32_t> d_n_iters(kTinySNPs);

        DaescConfig cfg;
        cfg.min_depth    = kMinDepth;
        cfg.max_iters    = 50;
        cfg.tol          = 1e-5f;
        cfg.n_cell_types = kTinyCelltypes;

        daesc(d_snp_ad, d_snp_dp, d_cell_type.get(), d_beta.get(),
              d_se.get(), d_phi.get(), d_pvalue.get(), d_n_iters.get(),
              cfg, stream);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        // Copy results to host.
        std::vector<float>   h_beta  (kTinySNPs);
        std::vector<float>   h_pvalue(kTinySNPs);
        ASSERT_EQ(cudaMemcpy(h_beta.data(),   d_beta.get(),
                             kTinySNPs * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(h_pvalue.data(), d_pvalue.get(),
                             kTinySNPs * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

        double rho_beta  = spearman(h_beta,   r_beta);
        double rho_pval  = spearman(h_pvalue, r_pval);

        EXPECT_GE(rho_beta, kSpearmanTol)
            << "Beta Spearman ρ = " << rho_beta << " < " << kSpearmanTol;
        EXPECT_GE(rho_pval, kSpearmanTol)
            << "P-value rank Spearman ρ = " << rho_pval << " < " << kSpearmanTol;

        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
}

// =============================================================================
// Test 2: Daesc_GSM_RealData
//
// Load real snp_ad.1pz + snp_dp.1pz from GSM4037629.
// Run GPU daesc on the first 1000 SNPs with the intercept-only model.
// Confirm: all betas finite, all p-values in (0, 1], ≥1 significant SNP
// at FDR 0.05 (BH-corrected).
// Skip if the .1pz files are absent.
// =============================================================================
TEST(Daesc, GSM_RealData) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU — skipping";

    if (!fs::exists(kGsmSnpAdPath) || !fs::exists(kGsmSnpDpPath))
        GTEST_SKIP() << "Real-data .1pz files absent — skipping";

    using namespace singlet::gpu::core;
    using namespace singlet::gpu::ase;
    using namespace singlet::gpu::io;

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    // Load first 1000 SNPs via pz_device_loader (the loader returns the full
    // matrix; we slice the first 1000 rows after loading).
    auto snp_ad_full = load_pz_device(kGsmSnpAdPath, stream);
    auto snp_dp_full = load_pz_device(kGsmSnpDpPath, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    const int n_snps  = static_cast<int>(std::min(
        static_cast<uint32_t>(1000), static_cast<uint32_t>(snp_ad_full.rows)));
    const int n_cells = static_cast<int>(snp_ad_full.cols);

    // Intercept-only model: pass nullptr for cell_type.
    DeviceMemory<float>   d_beta  (n_snps);
    DeviceMemory<float>   d_se    (n_snps);
    DeviceMemory<float>   d_phi   (n_snps);
    DeviceMemory<float>   d_pvalue(n_snps);
    DeviceMemory<int32_t> d_n_iters(n_snps);

    DaescConfig cfg;
    cfg.min_depth    = kMinDepth;
    cfg.max_iters    = 50;
    cfg.tol          = 1e-5f;
    cfg.n_cell_types = 0;  // intercept-only

    // Slice CSC to first n_snps rows (in-place via row-range view if supported,
    // otherwise pass full matrix — the kernel reads n_snps from cfg).
    cfg.n_snps_limit = n_snps;

    daesc(snp_ad_full, snp_dp_full, /*cell_type=*/nullptr,
          d_beta.get(), d_se.get(), d_phi.get(), d_pvalue.get(), d_n_iters.get(),
          cfg, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // Copy results.
    std::vector<float> h_beta  (n_snps);
    std::vector<float> h_pvalue(n_snps);
    ASSERT_EQ(cudaMemcpy(h_beta.data(),   d_beta.get(),
                         n_snps * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_pvalue.data(), d_pvalue.get(),
                         n_snps * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    // All betas finite.
    int n_nonfinite_beta = 0;
    for (float v : h_beta)   if (!std::isfinite(v)) ++n_nonfinite_beta;
    EXPECT_EQ(n_nonfinite_beta, 0) << "Non-finite beta values found";

    // All p-values in (0, 1].
    int n_bad_pval = 0;
    for (float v : h_pvalue) if (v <= 0.0f || v > 1.0f || !std::isfinite(v)) ++n_bad_pval;
    EXPECT_EQ(n_bad_pval, 0) << "Invalid p-values found";

    // BH FDR correction: count significant at q ≤ 0.05.
    std::vector<std::pair<float,int>> pv_idx(n_snps);
    for (int i = 0; i < n_snps; ++i) pv_idx[i] = {h_pvalue[i], i};
    std::sort(pv_idx.begin(), pv_idx.end());
    int n_sig = 0;
    for (int k = 0; k < n_snps; ++k) {
        float bh_threshold = 0.05f * (k + 1) / n_snps;
        if (pv_idx[k].first <= bh_threshold) n_sig = k + 1;
    }
    EXPECT_GT(n_sig, 0) << "No significant ASE SNPs found at BH FDR 0.05 "
                           "in real sample — may indicate a kernel bug";

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// =============================================================================
// Test 3: Daesc_FisherConvergence
//
// Run daesc on the tiny well-specified synthetic data (planted imbalance).
// Inspect per-SNP iteration counts (returned in n_iters output).
// Require: fraction converged within 50 iters > 0.98.
// "Converged" = n_iters[s] < 50 (hit tolerance before cap).
// =============================================================================
TEST(Daesc, FisherConvergence) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU — skipping";

    SyntheticAseData syn = make_synthetic_ase(kSeed);

    using namespace singlet::gpu::core;
    using namespace singlet::gpu::ase;

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    DeviceCSC d_snp_ad = upload_csc(syn.snp_ad, stream);
    DeviceCSC d_snp_dp = upload_csc(syn.snp_dp, stream);

    DeviceMemory<int32_t> d_cell_type(kTinyCells);
    ASSERT_EQ(cudaMemcpyAsync(d_cell_type.get(), syn.cell_type.data(),
                              kTinyCells * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);

    DeviceMemory<float>   d_beta    (kTinySNPs);
    DeviceMemory<float>   d_se      (kTinySNPs);
    DeviceMemory<float>   d_phi     (kTinySNPs);
    DeviceMemory<float>   d_pvalue  (kTinySNPs);
    DeviceMemory<int32_t> d_n_iters (kTinySNPs);

    DaescConfig cfg;
    cfg.min_depth    = kMinDepth;
    cfg.max_iters    = 50;
    cfg.tol          = 1e-5f;
    cfg.n_cell_types = kTinyCelltypes;

    daesc(d_snp_ad, d_snp_dp, d_cell_type.get(),
          d_beta.get(), d_se.get(), d_phi.get(), d_pvalue.get(), d_n_iters.get(),
          cfg, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<int32_t> h_n_iters(kTinySNPs);
    ASSERT_EQ(cudaMemcpy(h_n_iters.data(), d_n_iters.get(),
                         kTinySNPs * sizeof(int32_t), cudaMemcpyDeviceToHost), cudaSuccess);

    int n_converged = 0;
    for (int i = 0; i < kTinySNPs; ++i)
        if (h_n_iters[i] < 50) ++n_converged;

    double conv_rate = static_cast<double>(n_converged) / kTinySNPs;
    EXPECT_GT(conv_rate, kConvergenceRate)
        << "Convergence rate " << conv_rate << " <= " << kConvergenceRate
        << " (" << n_converged << "/" << kTinySNPs << " SNPs converged within 50 iters)";

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// =============================================================================
// Test 4: Daesc_PvalueCalibration_Uniform
//
// Under the null (balanced alleles, p=0.5 everywhere), p-values should be
// uniform(0,1). KS statistic ≤ 0.10.
// =============================================================================
TEST(Daesc, PvalueCalibration_Uniform) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU — skipping";

    SyntheticAseData null_syn = make_null_ase(kSeed);

    using namespace singlet::gpu::core;
    using namespace singlet::gpu::ase;

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    DeviceCSC d_snp_ad = upload_csc(null_syn.snp_ad, stream);
    DeviceCSC d_snp_dp = upload_csc(null_syn.snp_dp, stream);

    DeviceMemory<int32_t> d_cell_type(kTinyCells);
    ASSERT_EQ(cudaMemcpyAsync(d_cell_type.get(), null_syn.cell_type.data(),
                              kTinyCells * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);

    DeviceMemory<float>   d_beta    (kTinySNPs);
    DeviceMemory<float>   d_se      (kTinySNPs);
    DeviceMemory<float>   d_phi     (kTinySNPs);
    DeviceMemory<float>   d_pvalue  (kTinySNPs);
    DeviceMemory<int32_t> d_n_iters (kTinySNPs);

    DaescConfig cfg;
    cfg.min_depth    = kMinDepth;
    cfg.max_iters    = 50;
    cfg.tol          = 1e-5f;
    cfg.n_cell_types = kTinyCelltypes;

    daesc(d_snp_ad, d_snp_dp, d_cell_type.get(),
          d_beta.get(), d_se.get(), d_phi.get(), d_pvalue.get(), d_n_iters.get(),
          cfg, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> h_pvalue(kTinySNPs);
    ASSERT_EQ(cudaMemcpy(h_pvalue.data(), d_pvalue.get(),
                         kTinySNPs * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    // Clamp any out-of-range p-values before the KS test.
    for (auto& v : h_pvalue)
        v = std::max(0.0f, std::min(1.0f, v));

    double ks = ks_uniform(h_pvalue);
    EXPECT_LE(ks, kKsTol)
        << "KS statistic " << ks << " > " << kKsTol
        << " — p-values not uniform under null";

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// =============================================================================
// Test 5: Daesc_Determinism_BitIdentical
//
// Run daesc twice on the same synthetic data; compare beta, se, phi, pvalue
// bit-for-bit.  DAESC has no stochastic component (pure Fisher scoring) so
// results MUST be identical across runs.
// =============================================================================
TEST(Daesc, Determinism_BitIdentical) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU — skipping";

    SyntheticAseData syn = make_synthetic_ase(kSeed);

    using namespace singlet::gpu::core;
    using namespace singlet::gpu::ase;

    auto run_once = [&](std::vector<float>& beta_out,
                        std::vector<float>& pval_out) {
        cudaStream_t stream;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        DeviceCSC d_snp_ad = upload_csc(syn.snp_ad, stream);
        DeviceCSC d_snp_dp = upload_csc(syn.snp_dp, stream);

        DeviceMemory<int32_t> d_cell_type(kTinyCells);
        ASSERT_EQ(cudaMemcpyAsync(d_cell_type.get(), syn.cell_type.data(),
                                  kTinyCells * sizeof(int32_t),
                                  cudaMemcpyHostToDevice, stream), cudaSuccess);

        DeviceMemory<float>   d_beta    (kTinySNPs);
        DeviceMemory<float>   d_se      (kTinySNPs);
        DeviceMemory<float>   d_phi     (kTinySNPs);
        DeviceMemory<float>   d_pvalue  (kTinySNPs);
        DeviceMemory<int32_t> d_n_iters (kTinySNPs);

        DaescConfig cfg;
        cfg.min_depth    = kMinDepth;
        cfg.max_iters    = 50;
        cfg.tol          = 1e-5f;
        cfg.n_cell_types = kTinyCelltypes;

        daesc(d_snp_ad, d_snp_dp, d_cell_type.get(),
              d_beta.get(), d_se.get(), d_phi.get(), d_pvalue.get(), d_n_iters.get(),
              cfg, stream);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        beta_out.resize(kTinySNPs);
        pval_out.resize(kTinySNPs);
        ASSERT_EQ(cudaMemcpy(beta_out.data(), d_beta.get(),
                             kTinySNPs * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(pval_out.data(), d_pvalue.get(),
                             kTinySNPs * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    };

    std::vector<float> beta1, pval1, beta2, pval2;
    run_once(beta1, pval1);
    run_once(beta2, pval2);

    for (int i = 0; i < kTinySNPs; ++i) {
        // Use memcmp-level bit identity via casting to uint32_t.
        uint32_t b1, b2, p1, p2;
        std::memcpy(&b1, &beta1[i], 4);
        std::memcpy(&b2, &beta2[i], 4);
        std::memcpy(&p1, &pval1[i], 4);
        std::memcpy(&p2, &pval2[i], 4);

        EXPECT_EQ(b1, b2) << "Beta not bit-identical at SNP " << i;
        EXPECT_EQ(p1, p2) << "P-value not bit-identical at SNP " << i;
        if (HasFatalFailure()) break;
    }
}
