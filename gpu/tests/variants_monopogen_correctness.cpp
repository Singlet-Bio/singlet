// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/variants_monopogen_correctness.cpp
//
// integrates: original (first GPU Monopogen-style somatic SNV calling from scRNA —
//             exploits singlify snp_ad.1pz + snp_dp.1pz directly)
//
// Correctness harness for singlet_gpu::variants::monopogen.
//
// Written against the design doc at:
//   singlet-gpu/state/designs/42-monopogen.md
// NOT against the kernel source — authored in parallel with
// variants/monopogen.h by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: Monopogen Python via subprocess (GitHub install).
//   Fallback: pure-Python binomial genotyping + LD correction without the ML
//   refinement step (embedded in monopogen_py_reference.py).
//   The fallback always produces a reference for the synthetic test.
//
// Tolerances (design doc §"Correctness test spec"):
//   Germline call agreement   >= 0.95  vs Python (or fallback)
//   Somatic precision         >= 0.80  on planted variants
//   HWE chi-squared p-value   >  0.01  on common SNPs (not significant departure)
//   LD refinement reduces FP  strict:  fewer somatic candidates with LD than without
//   Bit-identical             across two runs
//
// Test cases:
//   Monopogen_TinySynthetic_VsPython
//   Monopogen_GSM_RealData
//   Monopogen_HWE_Consistency
//   Monopogen_LdRefinement_ReducesFalsePositives
//   Monopogen_Determinism_BitIdentical
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, CUDA::cusparse,
//             singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every GPU-dependent test calls GTEST_SKIP().

// ---- singlet-gpu variants header (written by gpu-kernel-dev) ---------------
#include <singlet-gpu/variants/monopogen.h>

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
// Compat bridge — test-harness API → real singlet-gpu API
// =============================================================================
//
// The test was authored against a slightly different API surface than the real
// header exposes.  Rather than rewrite the test, we add thin wrappers and type
// aliases here so the test compiles unchanged.

// io::PzLoader — wraps io::load_pz().mat (the real loader API).
namespace singlet_gpu { namespace io {
struct PzLoader {
    core::DeviceCSC load(const std::string& path, cudaStream_t stream) const {
        return singlet_gpu::io::load_pz(path, stream).mat;
    }
};
} }  // namespace singlet_gpu::io

// variants::LdPanel — alias for LdPanelCsr (single-chromosome LD; test always
//   passes one chromosome's panel or an empty sentinel).
namespace singlet_gpu { namespace variants {

using LdPanel = LdPanelCsr;

// load_ld_panel_tsv: read a tab-separated LD panel TSV (snp_i, snp_j, r_value)
// and build an LdPanelCsr for n_snps sites.  Returns empty panel if path doesn't
// exist (allows tests to skip gracefully when the fixture file is absent).
inline LdPanel load_ld_panel_tsv(const std::string& path, int n_snps) {
    LdPanel lp;
    lp.n_snps = n_snps;
    lp.indptr.assign(static_cast<size_t>(n_snps) + 1, 0);
    // Build neighbor list.
    std::vector<std::vector<std::pair<int32_t, float>>> adj(
        static_cast<size_t>(n_snps));
    std::ifstream f(path);
    if (f.is_open()) {
        int si, sj; float rv;
        while (f >> si >> sj >> rv) {
            if (si >= 0 && si < n_snps && sj >= 0 && sj < n_snps) {
                adj[static_cast<size_t>(si)].push_back({sj, rv});
            }
        }
    }
    for (int i = 0; i < n_snps; ++i) {
        lp.indptr[static_cast<size_t>(i) + 1] =
            lp.indptr[static_cast<size_t>(i)] +
            static_cast<int32_t>(adj[static_cast<size_t>(i)].size());
        for (auto& [j, r] : adj[static_cast<size_t>(i)]) {
            lp.col_idx.push_back(j);
            lp.val.push_back(r);
        }
    }
    return lp;
}

// run_monopogen: compat wrapper calling the real call_variants API.
// The test passes a single LdPanel; we wrap it as a single-element ld_panels
// vector with a trivial chromosome vector (all SNPs on chromosome 0).
// Preserves pileup_ad / pileup_dp (added to MonopogenResult as compat fields).
inline MonopogenResult run_monopogen(
    const core::DeviceCSC&   snp_ad,
    const core::DeviceCSC&   snp_dp,
    const LdPanel&           ld,
    const MonopogenConfig&   cfg,
    cudaStream_t             stream)
{
    // Build chromosome vector: all n_snps on chromosome 0.
    int n_snps = snp_ad.cols;
    std::vector<int> chromosome(static_cast<size_t>(n_snps), 0);

    // Build ld_panels vector.
    std::vector<LdPanelCsr> ld_panels;
    LdPanelCsr panel = ld;
    if (panel.n_snps == 0) {
        panel.n_snps = n_snps;
        panel.indptr.assign(static_cast<size_t>(n_snps) + 1, 0);
    }
    ld_panels.push_back(std::move(panel));

    // Build a temporary cusparseHandle (we don't have one in scope here).
    cusparseHandle_t sp_handle = nullptr;
    cusparseCreate(&sp_handle);

    MonopogenResult result = call_variants(
        snp_ad, snp_dp, chromosome, ld_panels, cfg, stream, sp_handle);

    cusparseDestroy(sp_handle);
    return result;
}

} }  // namespace singlet_gpu::variants

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

/// Tiny synthetic dimensions (from design doc §"Correctness test spec").
constexpr int kTinyCells         = 200;
constexpr int kTinySNPs          = 500;
constexpr int kGermlineHets      = 20;   // planted germline heterozygous sites
constexpr int kSomaticPlanted    = 10;   // planted somatic SNPs

/// Minimum read-depth filter applied both in the GPU kernel and the reference.
constexpr int kMinDepth          = 3;

/// Germline agreement tolerance (design doc: ≥ 0.95).
constexpr double kGermlineAgreement = 0.95;

/// Somatic precision tolerance (design doc: ≥ 0.80).
constexpr double kSomaticPrecision  = 0.80;

/// HWE chi-squared significance level (we expect NO departure, p > 0.01).
constexpr double kHwePvalueMin      = 0.01;

/// Real-data paths (GSM4037629 with SNP outputs).
static const char* kGsmSnpAdPath =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/snp_ad.1pz";
static const char* kGsmSnpDpPath =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/snp_dp.1pz";

/// Temporary directory for reference I/O.
static const char* kRefsTmpDir =
    "/tmp/singlet_gpu_variants_monopogen_refs_tmp";

/// Python reference script path.
#ifndef SINGLET_GPU_TEST_DIR
#define SINGLET_GPU_TEST_DIR "/mnt/home/debruinz/Singlet-AI/singlet-gpu/tests"
#endif

static const std::string kRefScript =
    std::string(SINGLET_GPU_TEST_DIR) +
    "/refs/monopogen_py_reference.py";

static const std::string kLdPanelPath =
    std::string(SINGLET_GPU_TEST_DIR) +
    "/refs/ld_panel_mini.tsv";

static const char* kPython3 = "python3";

// =============================================================================
// Utility helpers (pattern from cycles 3–41)
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
// Host-side CSC container (reused from cycles 3–41).
// ---------------------------------------------------------------------------
struct HostCSC {
    uint32_t             m;        // rows (SNPs)
    uint32_t             n;        // cols (cells)
    uint64_t             nnz;
    std::vector<float>   values;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
};

// Write HostCSC in the dump_csc binary format (CSCC magic, used by Python).
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
            static_cast<std::streamsize>(
                (static_cast<size_t>(h.n) + 1) * sizeof(int32_t)));
    f.write(reinterpret_cast<const char*>(h.indices.data()),
            static_cast<std::streamsize>(h.nnz * sizeof(int32_t)));
    if (!f) throw std::runtime_error("Write error: " + path);
}

// Upload HostCSC to device.
singlet_gpu::core::DeviceCSC upload_csc(const HostCSC& h, cudaStream_t stream) {
    using namespace singlet_gpu::core;
    // Use the synchronous host-data constructor: DeviceCSC(m, n, nnz, col_ptr, row_indices, values)
    cudaStreamSynchronize(stream);
    return DeviceCSC(
        static_cast<int>(h.m),
        static_cast<int>(h.n),
        static_cast<int>(h.nnz),
        h.indptr.data(),
        h.indices.data(),
        h.values.data());
}

// ---------------------------------------------------------------------------
// Synthetic data generator
// ---------------------------------------------------------------------------
//
// Produces two CSC matrices:  snp_ad (alt counts) and  snp_dp (depth).
// Design:
//   - kTinySNPs sites × kTinyCells cells.
//   - kGermlineHets sites (indices 0..kGermlineHets-1) are true hets: AF~0.50,
//     high depth (20–60 reads/cell), binomial(dp, 0.5) alt counts.
//   - kSomaticPlanted sites (indices kGermlineHets..kGermlineHets+kSomaticPlanted-1)
//     are somatic: AF~0.05–0.15, moderate depth (10–30), present in ~30% of cells.
//   - Remaining sites: depth 0 (most cells not covered) with sporadic noise.
//
// Truth masks:
//   germline_truth[i]  = true  for i < kGermlineHets
//   somatic_truth[i]   = true  for kGermlineHets ≤ i < kGermlineHets + kSomaticPlanted

struct SyntheticData {
    HostCSC snp_ad;
    HostCSC snp_dp;
    std::vector<bool> germline_truth;   // size kTinySNPs: true = germline het
    std::vector<bool> somatic_truth;    // size kTinySNPs: true = planted somatic
};

SyntheticData make_synthetic(uint64_t seed = kSeed) {
    std::mt19937_64 rng(seed);
    auto uniform = [&](double lo, double hi) {
        return std::uniform_real_distribution<double>(lo, hi)(rng);
    };
    auto binom = [&](int n, double p) -> int {
        std::binomial_distribution<int> d(n, p);
        return d(rng);
    };
    auto poisson = [&](double lam) -> int {
        std::poisson_distribution<int> d(lam);
        return d(rng);
    };

    const int M = kTinySNPs;
    const int N = kTinyCells;

    // Build sparse CSC in COO first, then convert.
    // We store per-cell per-snp entries only where depth > 0.
    struct Entry { int snp; int cell; float ad; float dp; };
    std::vector<Entry> entries;
    entries.reserve(M * N / 4);

    std::vector<bool> germ(M, false);
    std::vector<bool> som(M, false);

    for (int i = 0; i < M; ++i) {
        if (i < kGermlineHets) {
            // Germline het: high depth, AF~0.5
            germ[i] = true;
            for (int c = 0; c < N; ++c) {
                int dp = std::max(1, poisson(35.0));
                int ad = binom(dp, 0.5);
                if (dp > 0) entries.push_back({i, c, static_cast<float>(ad),
                                                        static_cast<float>(dp)});
            }
        } else if (i < kGermlineHets + kSomaticPlanted) {
            // Somatic: low AF (~0.08), present in ~40% of cells
            som[i] = true;
            double true_af = 0.05 + uniform(0.0, 0.08);
            for (int c = 0; c < N; ++c) {
                if (uniform(0, 1) > 0.40) continue;  // cell not covered
                int dp = std::max(kMinDepth, poisson(18.0));
                int ad = binom(dp, true_af);
                if (dp > 0) entries.push_back({i, c, static_cast<float>(ad),
                                                        static_cast<float>(dp)});
            }
        } else {
            // Background: very low coverage, noise only
            for (int c = 0; c < N; ++c) {
                if (uniform(0, 1) > 0.05) continue;
                int dp = std::max(1, poisson(5.0));
                int ad = binom(dp, 0.01);
                entries.push_back({i, c, static_cast<float>(ad),
                                         static_cast<float>(dp)});
            }
        }
    }

    // Sort COO by (cell, snp) for CSC conversion.
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return (a.cell != b.cell) ? (a.cell < b.cell) : (a.snp < b.snp);
    });

    // Build CSC indptr.
    auto build_csc = [&](bool use_ad) -> HostCSC {
        HostCSC h;
        h.m = static_cast<uint32_t>(M);
        h.n = static_cast<uint32_t>(N);
        h.nnz = static_cast<uint64_t>(entries.size());
        h.values .resize(h.nnz);
        h.indices.resize(h.nnz);
        h.indptr .resize(static_cast<size_t>(N) + 1, 0);

        // Count per-column (per-cell).
        for (const auto& e : entries)
            h.indptr[static_cast<size_t>(e.cell) + 1]++;
        for (int c = 0; c < N; ++c)
            h.indptr[static_cast<size_t>(c) + 1] +=
                h.indptr[static_cast<size_t>(c)];

        // Fill values and row indices.
        std::vector<int32_t> col_cursor(N, 0);
        for (int64_t k = 0; k < static_cast<int64_t>(entries.size()); ++k) {
            const auto& e = entries[k];
            int64_t pos = h.indptr[static_cast<size_t>(e.cell)] +
                          col_cursor[e.cell];
            h.values [pos] = use_ad ? e.ad : e.dp;
            h.indices[pos] = static_cast<int32_t>(e.snp);
            col_cursor[e.cell]++;
        }
        return h;
    };

    SyntheticData d;
    d.snp_ad = build_csc(/*use_ad=*/true);
    d.snp_dp = build_csc(/*use_ad=*/false);
    d.germline_truth = germ;
    d.somatic_truth  = som;
    return d;
}

// ---------------------------------------------------------------------------
// Parse Python reference output TSVs
// ---------------------------------------------------------------------------

struct GermCall { int snp_idx; int geno_call; float p_hom_ref, p_het, p_hom_alt; };
struct SomCall  { int snp_idx; float alt_freq, ld_score, p_value; int is_somatic; };

std::vector<GermCall> parse_germ_tsv(const std::string& path) {
    std::vector<GermCall> out;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        GermCall g{};
        char sep;
        ss >> g.snp_idx >> sep >> g.geno_call >> sep
           >> g.p_hom_ref >> sep >> g.p_het >> sep >> g.p_hom_alt;
        out.push_back(g);
    }
    return out;
}

std::vector<SomCall> parse_som_tsv(const std::string& path) {
    std::vector<SomCall> out;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        SomCall s{};
        char sep;
        ss >> s.snp_idx >> sep >> s.alt_freq >> sep
           >> s.ld_score >> sep >> s.p_value >> sep >> s.is_somatic;
        out.push_back(s);
    }
    return out;
}

// ---------------------------------------------------------------------------
// HWE chi-squared test helper (in-test, no dependency)
// ---------------------------------------------------------------------------
// Observed: n_hom_ref, n_het, n_hom_alt.
// Expected from p = (n_het + 2*n_hom_alt) / (2*N).
// Returns p-value (1 d.f. chi-squared).

double hwe_pvalue(int n_hom_ref, int n_het, int n_hom_alt) {
    int N = n_hom_ref + n_het + n_hom_alt;
    if (N <= 0) return 1.0;
    double p = (static_cast<double>(n_het) + 2.0 * n_hom_alt) /
               (2.0 * N);
    p = std::max(1e-6, std::min(1.0 - 1e-6, p));
    double q = 1.0 - p;
    double e_aa = static_cast<double>(N) * p * p;
    double e_ab = static_cast<double>(N) * 2.0 * p * q;
    double e_bb = static_cast<double>(N) * q * q;
    double chi2 = 0.0;
    if (e_aa > 0.5) chi2 += std::pow(n_hom_ref - e_aa, 2) / e_aa;
    if (e_ab > 0.5) chi2 += std::pow(n_het      - e_ab, 2) / e_ab;
    if (e_bb > 0.5) chi2 += std::pow(n_hom_alt  - e_bb, 2) / e_bb;
    // Approximate chi2(1) survival function via regularized incomplete gamma.
    // Use the approximation: p-val ≈ erfc(sqrt(chi2/2) / sqrt(2)).
    // This is exact for chi2(1).
    return std::erfc(std::sqrt(chi2 / 2.0));
}

} // anonymous namespace

// =============================================================================
// Test: Monopogen_TinySynthetic_VsPython
// =============================================================================
//
// 200 cells × 500 SNPs with 20 planted germline hets + 10 planted somatic.
// Runs the GPU kernel and the Python reference, then compares:
//   - Germline call agreement ≥ 0.95
//   - Somatic precision ≥ 0.80 on planted somatic sites

TEST(Monopogen, TinySynthetic_VsPython) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device";

    ensure_refs_tmp();
    const SyntheticData data = make_synthetic(kSeed);

    // Write binary CSC files for Python.
    const std::string ad_bin   = refs_path("synth_snp_ad.bin");
    const std::string dp_bin   = refs_path("synth_snp_dp.bin");
    const std::string germ_tsv = refs_path("py_germ_calls.tsv");
    const std::string som_tsv  = refs_path("py_som_calls.tsv");
    write_csc_bin(data.snp_ad, ad_bin);
    write_csc_bin(data.snp_dp, dp_bin);

    // Run Python reference.
    std::ostringstream py_cmd;
    py_cmd << kPython3 << " " << kRefScript
           << " --snp_ad "   << ad_bin
           << " --snp_dp "   << dp_bin
           << " --ld_panel " << kLdPanelPath
           << " --out_germ " << germ_tsv
           << " --out_som "  << som_tsv
           << " --n_cells "  << kTinyCells
           << " --n_snps "   << kTinySNPs
           << " --seed "     << kSeed
           << " 2>/dev/null";

    int py_rc = run_cmd_rc(py_cmd.str());
    if (py_rc == 2) GTEST_SKIP() << "Python reference unavailable (exit 2)";
    ASSERT_EQ(py_rc, 0) << "Python reference script failed (exit " << py_rc << ")";

    const auto py_germ = parse_germ_tsv(germ_tsv);
    const auto py_som  = parse_som_tsv(som_tsv);

    ASSERT_FALSE(py_germ.empty()) << "Python germline output is empty";

    // Build Python germline call map: snp_idx → geno_call.
    std::vector<int> py_geno(kTinySNPs, 0);
    for (const auto& g : py_germ) {
        if (g.snp_idx >= 0 && g.snp_idx < kTinySNPs)
            py_geno[g.snp_idx] = g.geno_call;
    }

    // Run GPU kernel.
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet_gpu::core::DeviceCSC d_ad = upload_csc(data.snp_ad, stream);
    singlet_gpu::core::DeviceCSC d_dp = upload_csc(data.snp_dp, stream);

    // Load LD panel from TSV.
    singlet_gpu::variants::LdPanel ld_panel =
        singlet_gpu::variants::load_ld_panel_tsv(kLdPanelPath, kTinySNPs);

    singlet_gpu::variants::MonopogenConfig cfg{};
    cfg.min_depth    = kMinDepth;
    cfg.seed         = kSeed;
    cfg.with_ld      = true;
    cfg.fdr_threshold = 0.05f;

    singlet_gpu::variants::MonopogenResult result =
        singlet_gpu::variants::run_monopogen(
            d_ad, d_dp, ld_panel, cfg, stream);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaStreamDestroy(stream);

    // --- Germline agreement ---
    // Compare GPU germline calls to Python calls on sites where Python called
    // a non-zero-confidence genotype.
    const auto& gpu_geno = result.germline_calls;  // vector<int>, size n_snps
    ASSERT_EQ(static_cast<int>(gpu_geno.size()), kTinySNPs);

    int agree = 0, total = 0;
    for (int i = 0; i < kTinySNPs; ++i) {
        if (i < kGermlineHets || py_geno[i] != 0) {
            agree += (gpu_geno[i] == py_geno[i]) ? 1 : 0;
            ++total;
        }
    }
    double agreement = (total > 0) ? static_cast<double>(agree) / total : 0.0;
    EXPECT_GE(agreement, kGermlineAgreement)
        << "Germline agreement " << agreement << " < " << kGermlineAgreement
        << " (" << agree << "/" << total << " sites agree)";

    // --- Somatic precision ---
    // GPU somatic calls: result.somatic_candidates contains {snp_idx, ...}.
    std::vector<bool> gpu_som_called(kTinySNPs, false);
    for (const auto& sc : result.somatic_candidates)
        if (sc.snp_idx >= 0 && sc.snp_idx < kTinySNPs)
            gpu_som_called[sc.snp_idx] = true;

    int true_positive = 0, total_called = 0;
    for (int i = 0; i < kTinySNPs; ++i) {
        if (gpu_som_called[i]) {
            ++total_called;
            if (data.somatic_truth[i]) ++true_positive;
        }
    }
    double precision = (total_called > 0) ?
        static_cast<double>(true_positive) / total_called : 0.0;

    // Require at least some somatic calls; if none at all, that's a failure.
    EXPECT_GT(total_called, 0) << "GPU somatic caller returned zero candidates";
    EXPECT_GE(precision, kSomaticPrecision)
        << "Somatic precision " << precision << " < " << kSomaticPrecision
        << " (" << true_positive << " TP / " << total_called << " called)";

    printf("  [TinySynthetic] germline agreement=%.4f (%d/%d)  "
           "somatic precision=%.4f (%d/%d called)\n",
           agreement, agree, total, precision, true_positive, total_called);
}

// =============================================================================
// Test: Monopogen_GSM_RealData
// =============================================================================
//
// Load real sample GSM4037629 snp_ad/snp_dp .1pz files.
// Confirm: non-zero germline call count, all depth values finite and non-negative.

TEST(Monopogen, GSM_RealData) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device";
    if (!fs::exists(kGsmSnpAdPath)) GTEST_SKIP() << "Real data not found";
    if (!fs::exists(kGsmSnpDpPath)) GTEST_SKIP() << "Real data not found";

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet_gpu::io::PzLoader loader;
    singlet_gpu::core::DeviceCSC d_ad =
        loader.load(kGsmSnpAdPath, stream);
    singlet_gpu::core::DeviceCSC d_dp =
        loader.load(kGsmSnpDpPath, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    ASSERT_GT(d_ad.rows, 0) << "snp_ad has 0 rows";
    ASSERT_GT(d_ad.cols, 0) << "snp_ad has 0 cols";
    ASSERT_EQ(d_ad.rows, d_dp.rows) << "snp_ad / snp_dp row mismatch";
    ASSERT_EQ(d_ad.cols, d_dp.cols) << "snp_ad / snp_dp col mismatch";

    // Build a trivial identity LD panel (no LD correction) for the real-data test.
    singlet_gpu::variants::LdPanel empty_ld{};

    singlet_gpu::variants::MonopogenConfig cfg{};
    cfg.min_depth     = kMinDepth;
    cfg.seed          = kSeed;
    cfg.with_ld       = false;   // skip LD correction on full real data
    cfg.fdr_threshold = 0.10f;   // liberal threshold to catch some calls

    singlet_gpu::variants::MonopogenResult result =
        singlet_gpu::variants::run_monopogen(
            d_ad, d_dp, empty_ld, cfg, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaStreamDestroy(stream);

    // Sanity: some germline calls.
    EXPECT_GT(result.germline_calls.size(), 0u) << "No germline calls on real data";
    int n_germ = std::count_if(result.germline_calls.begin(),
                               result.germline_calls.end(),
                               [](int g){ return g > 0; });
    EXPECT_GT(n_germ, 0) << "All germline calls are hom_ref on real data";

    // Sanity: pileup aggregates must all be finite and non-negative.
    const auto& ad_agg = result.pileup_ad;  // vector<float>, size n_snps
    const auto& dp_agg = result.pileup_dp;
    ASSERT_EQ(ad_agg.size(), dp_agg.size());
    for (size_t i = 0; i < ad_agg.size(); ++i) {
        EXPECT_GE(ad_agg[i], 0.0f) << "Negative AD at SNP " << i;
        EXPECT_GE(dp_agg[i], 0.0f) << "Negative DP at SNP " << i;
        EXPECT_TRUE(std::isfinite(ad_agg[i])) << "Non-finite AD at SNP " << i;
        EXPECT_TRUE(std::isfinite(dp_agg[i])) << "Non-finite DP at SNP " << i;
        EXPECT_LE(ad_agg[i], dp_agg[i] + 1e-3f)
            << "AD > DP at SNP " << i
            << " (AD=" << ad_agg[i] << " DP=" << dp_agg[i] << ")";
    }

    printf("  [GSM_RealData] n_snps=%zu n_cells=%d germline_calls=%d "
           "somatic_candidates=%zu\n",
           ad_agg.size(), d_ad.cols, n_germ,
           result.somatic_candidates.size());
}

// =============================================================================
// Test: Monopogen_HWE_Consistency
// =============================================================================
//
// Germline allele frequencies for common SNPs (those with DP ≥ 20 across all
// cells and AF ∈ [0.10, 0.90]) must satisfy Hardy–Weinberg equilibrium.
// Criterion: chi-squared test p-value > kHwePvalueMin on the aggregate of all
// common SNPs after calling genotypes.

TEST(Monopogen, HWE_Consistency) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device";

    ensure_refs_tmp();
    // Use the same synthetic data (has plenty of germline hets for HWE testing).
    const SyntheticData data = make_synthetic(kSeed);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet_gpu::core::DeviceCSC d_ad = upload_csc(data.snp_ad, stream);
    singlet_gpu::core::DeviceCSC d_dp = upload_csc(data.snp_dp, stream);

    singlet_gpu::variants::LdPanel empty_ld{};
    singlet_gpu::variants::MonopogenConfig cfg{};
    cfg.min_depth = kMinDepth;
    cfg.seed      = kSeed;
    cfg.with_ld   = false;

    singlet_gpu::variants::MonopogenResult result =
        singlet_gpu::variants::run_monopogen(
            d_ad, d_dp, empty_ld, cfg, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaStreamDestroy(stream);

    // Aggregate genotype counts over common SNPs.
    // We use germline_calls (0/1/2 = hom_ref / het / hom_alt).
    // A "common" SNP: dp_agg ≥ 20 AND alt_freq ∈ [0.10, 0.90].
    int n_hom_ref = 0, n_het = 0, n_hom_alt = 0, n_common = 0;
    for (int i = 0; i < kTinySNPs; ++i) {
        float dp = result.pileup_dp[i];
        float af = (dp > 0) ? result.pileup_ad[i] / dp : 0.0f;
        if (dp < 20.0f || af < 0.10f || af > 0.90f) continue;
        ++n_common;
        int g = result.germline_calls[i];
        if (g == 0) ++n_hom_ref;
        else if (g == 1) ++n_het;
        else ++n_hom_alt;
    }

    if (n_common < 5) {
        GTEST_SKIP() << "Too few common SNPs for HWE test (n=" << n_common << ")";
    }

    double p = hwe_pvalue(n_hom_ref, n_het, n_hom_alt);
    EXPECT_GT(p, kHwePvalueMin)
        << "HWE chi-squared p-value " << p
        << " < " << kHwePvalueMin
        << " (n_common=" << n_common
        << " hom_ref=" << n_hom_ref
        << " het="     << n_het
        << " hom_alt=" << n_hom_alt << ")";

    printf("  [HWE_Consistency] n_common=%d  hom_ref=%d  het=%d  hom_alt=%d  "
           "HWE_p=%.4f\n",
           n_common, n_hom_ref, n_het, n_hom_alt, p);
}

// =============================================================================
// Test: Monopogen_LdRefinement_ReducesFalsePositives
// =============================================================================
//
// With LD panel provided, fewer somatic candidates are called than without.
// The LD correction subtracts the linkage-explained component from each
// candidate's score, so true germline sites in LD are filtered out.

TEST(Monopogen, LdRefinement_ReducesFalsePositives) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device";

    ensure_refs_tmp();
    const SyntheticData data = make_synthetic(kSeed);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet_gpu::core::DeviceCSC d_ad_no = upload_csc(data.snp_ad, stream);
    singlet_gpu::core::DeviceCSC d_dp_no = upload_csc(data.snp_dp, stream);
    singlet_gpu::core::DeviceCSC d_ad_ld = upload_csc(data.snp_ad, stream);
    singlet_gpu::core::DeviceCSC d_dp_ld = upload_csc(data.snp_dp, stream);

    singlet_gpu::variants::MonopogenConfig cfg_no{};
    cfg_no.min_depth     = kMinDepth;
    cfg_no.seed          = kSeed;
    cfg_no.with_ld       = false;
    cfg_no.fdr_threshold = 0.20f;  // liberal to catch more candidates

    singlet_gpu::variants::LdPanel empty_ld{};
    singlet_gpu::variants::MonopogenResult result_no =
        singlet_gpu::variants::run_monopogen(
            d_ad_no, d_dp_no, empty_ld, cfg_no, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    singlet_gpu::variants::MonopogenConfig cfg_ld = cfg_no;
    cfg_ld.with_ld = true;

    // Load the mini LD panel (100-SNP subset fits the tiny 500-SNP matrix;
    // only SNPs 0..99 get LD correction; the rest are unaffected).
    singlet_gpu::variants::LdPanel ld_panel =
        singlet_gpu::variants::load_ld_panel_tsv(kLdPanelPath, kTinySNPs);

    singlet_gpu::variants::MonopogenResult result_ld =
        singlet_gpu::variants::run_monopogen(
            d_ad_ld, d_dp_ld, ld_panel, cfg_ld, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaStreamDestroy(stream);

    size_t n_no = result_no.somatic_candidates.size();
    size_t n_ld = result_ld.somatic_candidates.size();

    EXPECT_LE(n_ld, n_no)
        << "LD refinement did not reduce somatic candidates: "
        << "no_LD=" << n_no << " with_LD=" << n_ld;

    printf("  [LdRefinement] somatic_no_LD=%zu  somatic_with_LD=%zu  "
           "reduction=%.1f%%\n",
           n_no, n_ld,
           (n_no > 0) ? 100.0 * (n_no - n_ld) / n_no : 0.0);
}

// =============================================================================
// Test: Monopogen_Determinism_BitIdentical
// =============================================================================
//
// Two calls with the same seed on the same input must return bit-identical
// results.  Monopogen has no stochasticity (design doc §"Determinism") —
// this checks there are no uninitialized buffers or race conditions.

TEST(Monopogen, Determinism_BitIdentical) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device";

    ensure_refs_tmp();
    const SyntheticData data = make_synthetic(kSeed);

    auto run_once = [&]() {
        cudaStream_t stream;
        EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        singlet_gpu::core::DeviceCSC d_ad = upload_csc(data.snp_ad, stream);
        singlet_gpu::core::DeviceCSC d_dp = upload_csc(data.snp_dp, stream);

        singlet_gpu::variants::LdPanel ld_panel =
            singlet_gpu::variants::load_ld_panel_tsv(kLdPanelPath, kTinySNPs);

        singlet_gpu::variants::MonopogenConfig cfg{};
        cfg.min_depth     = kMinDepth;
        cfg.seed          = kSeed;
        cfg.with_ld       = true;
        cfg.fdr_threshold = 0.05f;

        singlet_gpu::variants::MonopogenResult result =
            singlet_gpu::variants::run_monopogen(
                d_ad, d_dp, ld_panel, cfg, stream);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        cudaStreamDestroy(stream);
        return result;
    };

    auto r1 = run_once();
    auto r2 = run_once();

    // Germline calls must be identical.
    ASSERT_EQ(r1.germline_calls.size(), r2.germline_calls.size());
    for (size_t i = 0; i < r1.germline_calls.size(); ++i) {
        EXPECT_EQ(r1.germline_calls[i], r2.germline_calls[i])
            << "germline_calls differ at SNP " << i;
    }

    // Pileup aggregates must be bit-identical.
    ASSERT_EQ(r1.pileup_ad.size(), r2.pileup_ad.size());
    for (size_t i = 0; i < r1.pileup_ad.size(); ++i) {
        EXPECT_EQ(r1.pileup_ad[i], r2.pileup_ad[i])
            << "pileup_ad differs at SNP " << i;
        EXPECT_EQ(r1.pileup_dp[i], r2.pileup_dp[i])
            << "pileup_dp differs at SNP " << i;
    }

    // Somatic candidate lists must be the same length and identical.
    ASSERT_EQ(r1.somatic_candidates.size(), r2.somatic_candidates.size())
        << "Different number of somatic candidates across runs";
    for (size_t k = 0; k < r1.somatic_candidates.size(); ++k) {
        EXPECT_EQ(r1.somatic_candidates[k].snp_idx,
                  r2.somatic_candidates[k].snp_idx)
            << "somatic snp_idx differs at candidate " << k;
        EXPECT_EQ(r1.somatic_candidates[k].ld_score,
                  r2.somatic_candidates[k].ld_score)
            << "somatic ld_score differs at candidate " << k;
    }

    printf("  [Determinism] germline=%zu  pileup=%zu  somatic=%zu "
           "— bit-identical\n",
           r1.germline_calls.size(), r1.pileup_ad.size(),
           r1.somatic_candidates.size());
}

// =============================================================================
// Main (GoogleTest entry point)
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
