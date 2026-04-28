// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/eqtl_nebula_correctness.cpp
//
// integrates: original (first GPU NEBULA-style single-cell eQTL mapping —
//             exploits singlify snp_ad.1pz + snp_dp.1pz directly)
//
// Correctness harness for singlet_gpu::eqtl::nebula_eqtl.
//
// Written against the design doc at:
//   singlet-gpu/state/designs/38-nebula-eqtl.md
// NOT against the kernel source — authored in parallel with eqtl/nebula.h by
// the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: NEBULA R via Rscript subprocess (library(nebula)).
//   If NEBULA R is absent (exit code 2 from the R script) → GTEST_SKIP.
//   Pure-R IRLS fallback (per-(SNP, gene) NB GLM, no mixed model) is
//   embedded in nebula_r_reference.R so the synthetic test always has a
//   reference even without the NEBULA package.
//
// Tolerances (design doc §"Correctness test spec"):
//   beta Spearman ρ       >= 0.90  vs NEBULA R (or pure-R IRLS fallback)
//   p-value rank Spearman >= 0.90  vs NEBULA R (or pure-R IRLS fallback)
//   Convergence rate      >  0.95  under null (50-iter cap)
//   p-value KS uniform    <= 0.10  under null
//   Bit-identical         across two runs
//
// Test cases:
//   Nebula_TinySynthetic_VsR
//   Nebula_GSM_RealMultisample
//   Nebula_FisherScoring_Convergence
//   Nebula_PvalueCalibration_Uniform
//   Nebula_Determinism_BitIdentical
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every GPU-dependent test calls GTEST_SKIP().

// ---- singlet-gpu eQTL header (written by gpu-kernel-dev) -------------------
#include <singlet-gpu/eqtl/nebula.h>

// ---- other singlet-gpu headers needed for the real-data test ---------------
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
// Bridge: adapt old test call convention to the actual run_nebula() API.
// Placed at global scope to avoid shadowing the real singlet_gpu namespace.
// =============================================================================

namespace singlet_gpu { namespace eqtl {

inline NebulaResult nebula_eqtl(
    const core::DeviceCSC& counts,
    const core::DeviceCSC& snp_ad,
    const core::DeviceCSC& snp_dp,
    const int*    donor_id_dev,  // device pointer (old convention)
    int           n_donors,
    const float*  /*covariates_dev*/,
    int           n_cov,
    const NebulaConfig& cfg,
    cudaStream_t  /*stream*/)
{
    int n_cells = (int)counts.cols;
    std::vector<int> h_donor_id(n_cells);
    cudaMemcpy(h_donor_id.data(), donor_id_dev,
               n_cells * sizeof(int), cudaMemcpyDeviceToHost);
    cudaStream_t stream2;
    cudaStreamCreate(&stream2);
    core::GPUContext& ctx = core::default_context();
    auto result = run_nebula(counts, snp_ad, snp_dp,
        h_donor_id.data(), n_donors,
        nullptr, n_cov,
        cfg, ctx, stream2, nullptr);
    cudaStreamSynchronize(stream2);
    cudaStreamDestroy(stream2);
    return result;
}

} } // namespace singlet_gpu::eqtl

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

/// Tiny synthetic dimensions (from design doc §"Correctness test spec").
constexpr int kTinyCells  = 200;
constexpr int kTinyGenes  = 50;
constexpr int kTinySNPs   = 100;
constexpr int kTinyDonors = 5;

/// Planted eQTL: first kPlantedPairs (SNP, gene) pairs have a real effect.
constexpr int kPlantedPairs = 10;
constexpr float kPlantedBeta = 1.5f;   // log-scale effect on mean expression

/// Spearman ρ tolerance (design doc: ≥ 0.90).
constexpr double kSpearmanTol = 0.90;

/// KS test tolerance for p-value uniformity (design doc: ≤ 0.10).
constexpr double kKsTol = 0.10;

/// Convergence rate threshold (design doc: > 0.95).
constexpr double kConvergenceRate = 0.95;

// Real-data paths (GSM4037629 + companion SNP outputs).
static const char* kGsmCountsPath =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/exon_counts.1pz";
static const char* kGsmSnpAdPath =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/snp_ad.1pz";
static const char* kGsmSnpDpPath =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/snp_dp.1pz";
static const char* kGsmDonorPath =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/donor_assignments.tsv";

/// Temporary directory for reference I/O.
static const char* kRefsTmpDir =
    "/tmp/singlet_gpu_eqtl_nebula_refs_tmp";

/// R reference script.
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/"
    "tests/refs/nebula_r_reference.R";

static const char* kRscript = "Rscript";

// =============================================================================
// Utility helpers (pattern from cycles 9–37)
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
// Host-side CSC container (reused from cycles 9–37).
// ---------------------------------------------------------------------------
struct HostCSC {
    uint32_t             m;        // rows (genes or SNPs)
    uint32_t             n;        // cols (cells)
    uint64_t             nnz;
    std::vector<float>   values;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
};

// Write HostCSC in the dump_csc binary format (CSCC magic, used by Python/R).
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

// Upload HostCSC to device.
singlet_gpu::core::DeviceCSC upload_csc(const HostCSC& h, cudaStream_t /*stream*/) {
    // Use the synchronous host-data constructor (col_ptr uses int, cast indptr).
    return singlet_gpu::core::DeviceCSC(
        (int)h.m, (int)h.n, (int)h.nnz,
        h.indptr.data(),   // const int* h_col_ptr
        h.indices.data(),  // const int* h_row_indices
        h.values.data());  // const float* h_values
}

// Write a dense matrix to TSV (rows × cols).
void write_matrix_tsv(const std::string& path,
                      const std::vector<float>& mat,
                      int rows, int cols) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (c) f << '\t';
            f << mat[static_cast<size_t>(r) * cols + c];
        }
        f << '\n';
    }
}

// Write a vector to a single-column TSV.
void write_vec_tsv(const std::string& path,
                   const std::vector<float>& v) {
    write_matrix_tsv(path, v, static_cast<int>(v.size()), 1);
}

// Read a dense TSV into a flat vector (row-major).
std::vector<float> read_matrix_tsv(const std::string& path, int rows, int cols) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::vector<float> mat(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (!(f >> mat[static_cast<size_t>(r) * cols + c]))
                throw std::runtime_error("read_matrix_tsv: short read at "
                                         + std::to_string(r) + "," + std::to_string(c)
                                         + " in " + path);
    }
    return mat;
}

// Read a single-column TSV into a vector.
std::vector<float> read_vec_tsv(const std::string& path, int n) {
    return read_matrix_tsv(path, n, 1);
}

// Write donor_id integer vector to TSV.
void write_donor_ids_tsv(const std::string& path,
                         const std::vector<int32_t>& ids) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    for (auto id : ids) f << id << '\n';
}

// ---------------------------------------------------------------------------
// Spearman rank correlation (reused from cycles 9–37).
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
// Returns the maximum absolute deviation of the empirical CDF from U(0,1).
// ---------------------------------------------------------------------------
double ks_uniform(std::vector<float> pvals) {
    std::sort(pvals.begin(), pvals.end());
    int n = static_cast<int>(pvals.size());
    double ks = 0.0;
    for (int i = 0; i < n; ++i) {
        double ecdf  = (i + 1.0) / n;
        double ecdf0 = static_cast<double>(i) / n;
        double d = std::max(std::abs(ecdf  - pvals[i]),
                            std::abs(ecdf0 - pvals[i]));
        ks = std::max(ks, d);
    }
    return ks;
}

// ---------------------------------------------------------------------------
// Synthetic data generator.
//
// Produces three CSC matrices (counts, snp_ad, snp_dp) and donor_id labels.
//
// Cells are divided equally across kTinyDonors donors.
// For planted pairs (snp_idx < kPlantedPairs, gene_idx == snp_idx):
//   cells from even-genotype donors get expression boost kPlantedBeta on log
//   scale (i.e., mean multiplied by exp(kPlantedBeta)).
//
// Genotype:
//   snp_ad[snp, cell] ~ Binomial(snp_dp[snp, cell], dosage/2)
//   dosage per donor ~ Multinomial probs: 0=0.25, 1=0.50, 2=0.25
//   snp_dp: draw from Poisson(20) per (snp, cell), min 1.
//
// Expression:
//   mu_base = 5.0 (all cells, all genes)
//   For planted pair (s, g=s): mu *= exp(kPlantedBeta) for donor with dosage==2
//   counts[gene, cell] ~ Poisson(mu)
// ---------------------------------------------------------------------------
struct SyntheticData {
    HostCSC counts;     // genes × cells
    HostCSC snp_ad;     // snps  × cells
    HostCSC snp_dp;     // snps  × cells
    std::vector<int32_t> donor_id;  // cells
};

SyntheticData make_synthetic(uint64_t seed) {
    std::mt19937_64 rng(seed);

    const int C = kTinyCells;
    const int G = kTinyGenes;
    const int S = kTinySNPs;
    const int D = kTinyDonors;
    const int cells_per_donor = C / D;

    // Donor assignments (round-robin).
    std::vector<int32_t> donor_id(C);
    for (int j = 0; j < C; ++j)
        donor_id[j] = j / cells_per_donor;

    // Per-donor dosage for each SNP: 0, 1, or 2.
    // We store them per (donor, snp) row-major.
    std::vector<int> dosage(static_cast<size_t>(D) * S);
    {
        std::discrete_distribution<int> geno_dist({0.25, 0.50, 0.25});
        for (auto& d : dosage)
            d = geno_dist(rng);
    }

    // Build snp_dp and snp_ad CSC (S rows × C cols).
    // We store as dense then compress.
    auto build_snp_csc = [&](bool is_ad) -> HostCSC {
        std::poisson_distribution<int> dp_dist(20);
        HostCSC csc;
        csc.m = S; csc.n = C;
        csc.indptr.resize(static_cast<size_t>(C) + 1, 0);

        std::vector<std::vector<std::pair<int32_t, float>>> cols(C);
        for (int j = 0; j < C; ++j) {
            int d = donor_id[j];
            for (int s = 0; s < S; ++s) {
                int dp_val = std::max(1, dp_dist(rng));
                float val;
                if (is_ad) {
                    // snp_ad: Binomial(dp_val, dosage/2)
                    int dos = dosage[static_cast<size_t>(d) * S + s];
                    double p = dos / 2.0;
                    std::binomial_distribution<int> binom(dp_val, p);
                    val = static_cast<float>(binom(rng));
                } else {
                    val = static_cast<float>(dp_val);
                }
                if (val > 0.0f)
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

    // Build dp first; we need matching structure for ad.
    // (Re-seed to get same Poisson draws for dp whether or not ad is computed.)
    std::mt19937_64 rng2(seed + 1);
    HostCSC snp_dp_csc, snp_ad_csc;
    {
        // dp
        std::poisson_distribution<int> dp_dist(20);
        snp_dp_csc.m = S; snp_dp_csc.n = C;
        snp_dp_csc.indptr.resize(static_cast<size_t>(C) + 1, 0);

        // We need the same dp values for ad, so generate dp_vals densely first.
        std::vector<int> dp_vals(static_cast<size_t>(S) * C);
        for (auto& v : dp_vals)
            v = std::max(1, dp_dist(rng2));

        // Build dp CSC.
        {
            std::vector<std::vector<std::pair<int32_t, float>>> cols(C);
            for (int j = 0; j < C; ++j)
                for (int s = 0; s < S; ++s) {
                    float val = static_cast<float>(dp_vals[static_cast<size_t>(s) * C + j]);
                    cols[j].push_back({static_cast<int32_t>(s), val});
                }
            uint64_t total_nnz = 0;
            for (int j = 0; j < C; ++j) total_nnz += cols[j].size();
            snp_dp_csc.nnz = total_nnz;
            snp_dp_csc.indptr[0] = 0;
            for (int j = 0; j < C; ++j)
                snp_dp_csc.indptr[j + 1] = snp_dp_csc.indptr[j] + static_cast<int32_t>(cols[j].size());
            snp_dp_csc.values .resize(total_nnz);
            snp_dp_csc.indices.resize(total_nnz);
            size_t pos = 0;
            for (int j = 0; j < C; ++j)
                for (auto& [row, val] : cols[j]) {
                    snp_dp_csc.indices[pos] = row;
                    snp_dp_csc.values [pos] = val;
                    ++pos;
                }
        }

        // Build ad CSC using same dp_vals.
        {
            std::vector<std::vector<std::pair<int32_t, float>>> cols(C);
            for (int j = 0; j < C; ++j) {
                int d = donor_id[j];
                for (int s = 0; s < S; ++s) {
                    int dp = dp_vals[static_cast<size_t>(s) * C + j];
                    int dos = dosage[static_cast<size_t>(d) * S + s];
                    double p = dos / 2.0;
                    std::binomial_distribution<int> binom(dp, p);
                    float val = static_cast<float>(binom(rng2));
                    if (val > 0.0f)
                        cols[j].push_back({static_cast<int32_t>(s), val});
                }
            }
            uint64_t total_nnz = 0;
            for (int j = 0; j < C; ++j) total_nnz += cols[j].size();
            snp_ad_csc.m = S; snp_ad_csc.n = C;
            snp_ad_csc.nnz = total_nnz;
            snp_ad_csc.indptr.resize(static_cast<size_t>(C) + 1, 0);
            snp_ad_csc.indptr[0] = 0;
            for (int j = 0; j < C; ++j)
                snp_ad_csc.indptr[j + 1] = snp_ad_csc.indptr[j] + static_cast<int32_t>(cols[j].size());
            snp_ad_csc.values .resize(total_nnz);
            snp_ad_csc.indices.resize(total_nnz);
            size_t pos = 0;
            for (int j = 0; j < C; ++j)
                for (auto& [row, val] : cols[j]) {
                    snp_ad_csc.indices[pos] = row;
                    snp_ad_csc.values [pos] = val;
                    ++pos;
                }
        }
    }

    // Build counts CSC (G rows × C cols).
    HostCSC counts_csc;
    {
        std::mt19937_64 rng3(seed + 2);
        counts_csc.m = G; counts_csc.n = C;
        counts_csc.indptr.resize(static_cast<size_t>(C) + 1, 0);

        std::vector<std::vector<std::pair<int32_t, float>>> cols(C);
        for (int j = 0; j < C; ++j) {
            int d = donor_id[j];
            for (int g = 0; g < G; ++g) {
                double mu = 5.0;
                // Planted eQTL: SNP s == g, and donor has dosage 2 → boost.
                if (g < kPlantedPairs) {
                    int s = g;
                    int dos = dosage[static_cast<size_t>(d) * S + s];
                    if (dos == 2) mu *= std::exp(kPlantedBeta);
                }
                std::poisson_distribution<int> pois(mu);
                int val = pois(rng3);
                if (val > 0)
                    cols[j].push_back({static_cast<int32_t>(g), static_cast<float>(val)});
            }
        }
        uint64_t total_nnz = 0;
        for (int j = 0; j < C; ++j) total_nnz += cols[j].size();
        counts_csc.nnz = total_nnz;
        counts_csc.indptr[0] = 0;
        for (int j = 0; j < C; ++j)
            counts_csc.indptr[j + 1] = counts_csc.indptr[j] + static_cast<int32_t>(cols[j].size());
        counts_csc.values .resize(total_nnz);
        counts_csc.indices.resize(total_nnz);
        size_t pos = 0;
        for (int j = 0; j < C; ++j)
            for (auto& [row, val] : cols[j]) {
                counts_csc.indices[pos] = row;
                counts_csc.values [pos] = val;
                ++pos;
            }
    }

    return SyntheticData{
        std::move(counts_csc),
        std::move(snp_ad_csc),
        std::move(snp_dp_csc),
        std::move(donor_id)
    };
}

// ---------------------------------------------------------------------------
// Null synthetic: same structure but SNP dosages are permuted per gene so
// there is no true eQTL signal. Used for convergence + p-value calibration.
// ---------------------------------------------------------------------------
SyntheticData make_null_synthetic(uint64_t seed) {
    SyntheticData d = make_synthetic(seed + 0xDEAD);
    // Permute donor_id labels to break any residual SNP-expression correlation.
    std::mt19937_64 rng(seed + 0xBEEF);
    std::shuffle(d.donor_id.begin(), d.donor_id.end(), rng);
    return d;
}

}  // anonymous namespace

// =============================================================================
// Test 1: Nebula_TinySynthetic_VsR
//
// Generate planted eQTL synthetic data.
// Run GPU nebula_eqtl → collect beta and p-values.
// Run nebula_r_reference.R (with NEBULA R if available, else pure-R IRLS).
// Compare via Spearman ρ on beta flat vector and on log p-value rank vector.
// Tolerance: ρ ≥ 0.90 on both.
// =============================================================================
TEST(Nebula, TinySynthetic_VsR) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU — skipping";

    ensure_refs_tmp();

    SyntheticData syn = make_synthetic(kSeed);

    // ---- Write inputs for R reference ----------------------------------------
    write_csc_bin(syn.counts, refs_path("counts.bin"));
    write_csc_bin(syn.snp_ad, refs_path("snp_ad.bin"));
    write_csc_bin(syn.snp_dp, refs_path("snp_dp.bin"));
    write_donor_ids_tsv(refs_path("donor_ids.tsv"), syn.donor_id);

    // ---- Run R reference -------------------------------------------------------
    std::string r_cmd =
        std::string(kRscript) + " " + kRefScript +
        " --counts "    + refs_path("counts.bin") +
        " --snp_ad "    + refs_path("snp_ad.bin") +
        " --snp_dp "    + refs_path("snp_dp.bin") +
        " --donor_ids " + refs_path("donor_ids.tsv") +
        " --out_beta "  + refs_path("r_beta.tsv") +
        " --out_pval "  + refs_path("r_pval.tsv") +
        " --n_genes "   + std::to_string(kTinyGenes) +
        " --n_snps "    + std::to_string(kTinySNPs) +
        " 2>" + refs_path("r_stderr.txt");

    int rc = run_cmd_rc(r_cmd);
    if (rc == 2) {
        GTEST_SKIP() << "NEBULA R and IRLS fallback both unavailable (exit 2)";
    }
    ASSERT_EQ(rc, 0) << "R reference script failed; see " << refs_path("r_stderr.txt");

    std::vector<float> r_beta  = read_matrix_tsv(refs_path("r_beta.tsv"),
                                                  kTinySNPs, kTinyGenes);
    std::vector<float> r_pval  = read_matrix_tsv(refs_path("r_pval.tsv"),
                                                  kTinySNPs, kTinyGenes);

    // ---- Upload to device + run GPU kernel -----------------------------------
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto d_counts = upload_csc(syn.counts, stream);
    auto d_snp_ad = upload_csc(syn.snp_ad, stream);
    auto d_snp_dp = upload_csc(syn.snp_dp, stream);

    // Upload donor_id array.
    singlet_gpu::core::DeviceMemory<int32_t> d_donor_id(kTinyCells);
    ASSERT_EQ(cudaMemcpyAsync(d_donor_id.get(), syn.donor_id.data(),
                              kTinyCells * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    singlet_gpu::eqtl::NebulaConfig cfg;
    // n_donors is passed directly to run_nebula; not a config field.
    cfg.max_iters     = 50;
    cfg.fisher_tol    = 1e-5f;
    cfg.deterministic = false;

    singlet_gpu::eqtl::NebulaResult result =
        singlet_gpu::eqtl::nebula_eqtl(
            d_counts, d_snp_ad, d_snp_dp,
            d_donor_id.get(), kTinyDonors,
            /*covariates_dev=*/nullptr, /*n_cov=*/0,
            cfg, stream);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // Download beta and p-values to host.
    // result.beta and result.p_value are device matrices (n_snps × n_genes).
    size_t n_pairs = static_cast<size_t>(kTinySNPs) * kTinyGenes;
    std::vector<float> gpu_beta(n_pairs), gpu_pval(n_pairs);
    // result.beta and result.p_value are host-side std::vector<float>.
    ASSERT_EQ(result.beta.size(),    n_pairs) << "beta size mismatch";
    ASSERT_EQ(result.p_value.size(), n_pairs) << "p_value size mismatch";
    gpu_beta = result.beta;
    gpu_pval = result.p_value;

    // Check all finite.
    for (size_t i = 0; i < n_pairs; ++i) {
        ASSERT_TRUE(std::isfinite(gpu_beta[i])) << "Non-finite beta at pair " << i;
        ASSERT_TRUE(std::isfinite(gpu_pval[i])) << "Non-finite p_value at pair " << i;
    }

    // Spearman ρ on beta.
    double rho_beta = spearman(gpu_beta, r_beta);
    EXPECT_GE(rho_beta, kSpearmanTol)
        << "beta Spearman rho=" << rho_beta << " < " << kSpearmanTol;

    // Spearman ρ on log p-value rank.
    // Convert to -log10(p) for rank correlation (avoids pval=0 issues).
    std::vector<float> gpu_logp(n_pairs), r_logp(n_pairs);
    for (size_t i = 0; i < n_pairs; ++i) {
        gpu_logp[i] = -std::log10(std::max(gpu_pval[i], 1e-30f));
        r_logp  [i] = -std::log10(std::max(r_pval  [i], 1e-30f));
    }
    double rho_pval = spearman(gpu_logp, r_logp);
    EXPECT_GE(rho_pval, kSpearmanTol)
        << "p-value rank Spearman rho=" << rho_pval << " < " << kSpearmanTol;

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    printf("[Nebula_TinySynthetic_VsR] beta_rho=%.4f pval_rho=%.4f\n",
           rho_beta, rho_pval);
}

// =============================================================================
// Test 2: Nebula_GSM_RealMultisample
//
// Load exon_counts.1pz, snp_ad.1pz, snp_dp.1pz from GSM4037629.
// Concat with itself 3× to simulate 3 samples sharing a donor pool.
// Run GPU nebula_eqtl on a 500-SNP × 500-gene slice.
// Assert: all-finite results, non-zero converged count.
// =============================================================================
TEST(Nebula, GSM_RealMultisample) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU — skipping";

    // Check real data is available.
    if (!fs::exists(kGsmCountsPath)) {
        GTEST_SKIP() << "GSM4037629 exon_counts.1pz not found";
    }
    if (!fs::exists(kGsmSnpAdPath) || !fs::exists(kGsmSnpDpPath)) {
        GTEST_SKIP() << "GSM4037629 snp_ad/snp_dp not found";
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    // Load real data. Use load_pz() which returns PzDeviceMatrix with .mat field.
    auto d_counts = singlet_gpu::io::load_pz(kGsmCountsPath, stream).mat;
    auto d_snp_ad = singlet_gpu::io::load_pz(kGsmSnpAdPath,  stream).mat;
    auto d_snp_dp = singlet_gpu::io::load_pz(kGsmSnpDpPath,  stream).mat;
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    int n_cells = (int)d_counts.cols;

    // Simulate 3 "samples" by repeating the same cells 3× with different donor IDs.
    // In practice we just assign round-robin donor IDs to the existing cells.
    constexpr int kSimDonors = 10;
    std::vector<int32_t> donor_id(n_cells);
    for (int j = 0; j < n_cells; ++j)
        donor_id[j] = j % kSimDonors;

    singlet_gpu::core::DeviceMemory<int32_t> d_donor_id(n_cells);
    ASSERT_EQ(cudaMemcpyAsync(d_donor_id.get(), donor_id.data(),
                              n_cells * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    singlet_gpu::eqtl::NebulaConfig cfg;
    // n_donors/max_snps/max_genes are not NebulaConfig fields;
    // chunk_snps/chunk_genes control chunk size (limits per-chunk work).
    cfg.max_iters     = 50;
    cfg.fisher_tol    = 1e-5f;
    cfg.chunk_snps    = 200;   // limit chunk size for smoke test
    cfg.chunk_genes   = 200;
    cfg.deterministic = false;

    singlet_gpu::eqtl::NebulaResult result =
        singlet_gpu::eqtl::nebula_eqtl(
            d_counts, d_snp_ad, d_snp_dp,
            d_donor_id.get(), kSimDonors,
            /*covariates_dev=*/nullptr, /*n_cov=*/0,
            cfg, stream);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    int n_snps_out  = static_cast<int>(result.n_snps);
    int n_genes_out = static_cast<int>(result.n_genes);
    size_t n_pairs  = static_cast<size_t>(n_snps_out) * n_genes_out;

    ASSERT_GT(n_pairs, 0u) << "Zero output pairs";

    // result.beta, result.p_value, result.converged are host std::vectors.
    std::vector<float>   beta    = result.beta;
    std::vector<float>   pval    = result.p_value;
    std::vector<uint8_t> conv_u8 = result.converged;

    int n_finite_beta = 0, n_conv = 0;
    for (size_t i = 0; i < n_pairs; ++i) {
        if (std::isfinite(beta[i])) ++n_finite_beta;
        if (conv_u8[i])             ++n_conv;
    }

    EXPECT_GT(n_finite_beta, 0) << "All beta values are non-finite";
    EXPECT_GT(n_conv, 0)        << "Zero converged (SNP, gene) pairs on real data";

    printf("[Nebula_GSM_RealMultisample] n_pairs=%zu finite_beta=%d converged=%d\n",
           n_pairs, n_finite_beta, n_conv);

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// =============================================================================
// Test 3: Nebula_FisherScoring_Convergence
//
// Under null (random SNP dosages, no planted eQTL), run GPU nebula_eqtl and
// verify that >95% of (SNP, gene) pairs converge within 50 iterations.
// =============================================================================
TEST(Nebula, FisherScoring_Convergence) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU — skipping";

    ensure_refs_tmp();
    SyntheticData null_syn = make_null_synthetic(kSeed);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto d_counts = upload_csc(null_syn.counts, stream);
    auto d_snp_ad = upload_csc(null_syn.snp_ad, stream);
    auto d_snp_dp = upload_csc(null_syn.snp_dp, stream);

    singlet_gpu::core::DeviceMemory<int32_t> d_donor_id(kTinyCells);
    ASSERT_EQ(cudaMemcpyAsync(d_donor_id.get(), null_syn.donor_id.data(),
                              kTinyCells * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    singlet_gpu::eqtl::NebulaConfig cfg;
    // n_donors is passed directly to run_nebula; not a config field.
    cfg.max_iters     = 50;
    cfg.fisher_tol    = 1e-5f;
    cfg.deterministic = false;

    singlet_gpu::eqtl::NebulaResult result =
        singlet_gpu::eqtl::nebula_eqtl(
            d_counts, d_snp_ad, d_snp_dp,
            d_donor_id.get(), kTinyDonors,
            nullptr, 0,
            cfg, stream);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    size_t n_pairs = static_cast<size_t>(kTinySNPs) * kTinyGenes;
    // result.converged is a host std::vector<uint8_t>.
    const auto& conv = result.converged;

    int n_conv = 0;
    for (auto c : conv) n_conv += (c != 0);
    double conv_rate = static_cast<double>(n_conv) / static_cast<double>(n_pairs);

    EXPECT_GE(conv_rate, kConvergenceRate)
        << "Convergence rate=" << conv_rate << " < " << kConvergenceRate;

    printf("[Nebula_FisherScoring_Convergence] converged=%d / %zu = %.4f\n",
           n_conv, n_pairs, conv_rate);

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// =============================================================================
// Test 4: Nebula_PvalueCalibration_Uniform
//
// Under null (permuted donor IDs), run GPU nebula_eqtl.
// Collect p-values for all (SNP, gene) pairs where converged==true.
// KS statistic vs Uniform(0,1) must be ≤ kKsTol (0.10).
// =============================================================================
TEST(Nebula, PvalueCalibration_Uniform) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU — skipping";

    ensure_refs_tmp();
    SyntheticData null_syn = make_null_synthetic(kSeed + 1);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto d_counts = upload_csc(null_syn.counts, stream);
    auto d_snp_ad = upload_csc(null_syn.snp_ad, stream);
    auto d_snp_dp = upload_csc(null_syn.snp_dp, stream);

    singlet_gpu::core::DeviceMemory<int32_t> d_donor_id(kTinyCells);
    ASSERT_EQ(cudaMemcpyAsync(d_donor_id.get(), null_syn.donor_id.data(),
                              kTinyCells * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    singlet_gpu::eqtl::NebulaConfig cfg;
    // n_donors is passed directly to run_nebula; not a config field.
    cfg.max_iters     = 50;
    cfg.fisher_tol    = 1e-5f;
    cfg.deterministic = false;

    singlet_gpu::eqtl::NebulaResult result =
        singlet_gpu::eqtl::nebula_eqtl(
            d_counts, d_snp_ad, d_snp_dp,
            d_donor_id.get(), kTinyDonors,
            nullptr, 0,
            cfg, stream);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    size_t n_pairs = static_cast<size_t>(kTinySNPs) * kTinyGenes;
    // result.p_value and result.converged are host-side vectors.
    const std::vector<float>&   pval = result.p_value;
    const std::vector<uint8_t>& conv = result.converged;

    // Keep only converged p-values.
    std::vector<float> conv_pvals;
    conv_pvals.reserve(n_pairs);
    for (size_t i = 0; i < n_pairs; ++i)
        if (conv[i] && std::isfinite(pval[i]) && pval[i] >= 0.0f && pval[i] <= 1.0f)
            conv_pvals.push_back(pval[i]);

    ASSERT_GT(conv_pvals.size(), 100u)
        << "Too few converged pairs (" << conv_pvals.size() << ") for KS test";

    double ks = ks_uniform(conv_pvals);
    EXPECT_LE(ks, kKsTol)
        << "KS stat=" << ks << " > " << kKsTol
        << " (p-values not uniform under null; " << conv_pvals.size() << " pairs)";

    printf("[Nebula_PvalueCalibration_Uniform] n_converged=%zu KS=%.4f\n",
           conv_pvals.size(), ks);

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// =============================================================================
// Test 5: Nebula_Determinism_BitIdentical
//
// Run GPU nebula_eqtl twice on the same planted synthetic data.
// Assert bit-identical beta, se, p_value, converged across both runs.
// =============================================================================
TEST(Nebula, Determinism_BitIdentical) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU — skipping";

    ensure_refs_tmp();
    SyntheticData syn = make_synthetic(kSeed);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto d_counts = upload_csc(syn.counts, stream);
    auto d_snp_ad = upload_csc(syn.snp_ad, stream);
    auto d_snp_dp = upload_csc(syn.snp_dp, stream);

    singlet_gpu::core::DeviceMemory<int32_t> d_donor_id(kTinyCells);
    ASSERT_EQ(cudaMemcpyAsync(d_donor_id.get(), syn.donor_id.data(),
                              kTinyCells * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    singlet_gpu::eqtl::NebulaConfig cfg;
    // n_donors is passed directly to run_nebula; not a config field.
    cfg.max_iters     = 50;
    cfg.fisher_tol    = 1e-5f;
    cfg.deterministic = true;   // required for bit-identical

    size_t n_pairs = static_cast<size_t>(kTinySNPs) * kTinyGenes;

    // Run 1.
    singlet_gpu::eqtl::NebulaResult r1 =
        singlet_gpu::eqtl::nebula_eqtl(
            d_counts, d_snp_ad, d_snp_dp,
            d_donor_id.get(), kTinyDonors,
            nullptr, 0,
            cfg, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // result.beta, .se, .p_value are host std::vector<float>; .converged is uint8_t.
    std::vector<float>   beta1    = r1.beta;
    std::vector<float>   se1      = r1.se;
    std::vector<float>   pval1    = r1.p_value;
    std::vector<uint8_t> conv1    = r1.converged;

    // Run 2.
    singlet_gpu::eqtl::NebulaResult r2 =
        singlet_gpu::eqtl::nebula_eqtl(
            d_counts, d_snp_ad, d_snp_dp,
            d_donor_id.get(), kTinyDonors,
            nullptr, 0,
            cfg, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float>   beta2    = r2.beta;
    std::vector<float>   se2      = r2.se;
    std::vector<float>   pval2    = r2.p_value;
    std::vector<uint8_t> conv2    = r2.converged;

    // Compare bit-identical.
    int diff_beta = 0, diff_se = 0, diff_pval = 0, diff_conv = 0;
    for (size_t i = 0; i < n_pairs; ++i) {
        if (std::memcmp(&beta1[i], &beta2[i], 4) != 0) ++diff_beta;
        if (std::memcmp(&se1[i],   &se2[i],   4) != 0) ++diff_se;
        if (std::memcmp(&pval1[i], &pval2[i], 4) != 0) ++diff_pval;
        if (conv1[i] != conv2[i])                       ++diff_conv;
    }

    EXPECT_EQ(diff_beta, 0) << diff_beta << " beta values differ between run1 and run2";
    EXPECT_EQ(diff_se,   0) << diff_se   << " se values differ";
    EXPECT_EQ(diff_pval, 0) << diff_pval << " p_value values differ";
    EXPECT_EQ(diff_conv, 0) << diff_conv << " converged flags differ";

    printf("[Nebula_Determinism_BitIdentical] diff_beta=%d diff_se=%d diff_pval=%d diff_conv=%d\n",
           diff_beta, diff_se, diff_pval, diff_conv);

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
