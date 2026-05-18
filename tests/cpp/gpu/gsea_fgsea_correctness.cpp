// SPDX-License-Identifier: MIT
// singlet/gpu/tests/gsea_fgsea_correctness.cpp
//
// Correctness harness for singlet::gpu::gsea::fgsea.
//
// Written against the design doc at:
//   singlet/gpu/state/designs/11-gsea.md
// NOT against the kernel source — authored in parallel with gsea/fgsea.h by
// the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: R fgsea package via subprocess
//   tests/refs/fgsea_r_reference.R
//   Environment: R + BiocManager::install("fgsea")
//
// Tolerances (design doc §"Correctness test spec"):
//   ES rel_err per pathway            <= 1e-4
//   p-value rank Spearman rho         >= 0.95
//   Top-10 enriched pathways Jaccard  >= 0.90  (vs R fgsea, by NES rank)
//   Determinism                        bit-identical (same seed, same input)
//   BH q-values                        not all trivially zero
//   min_set_size filter                gene sets below threshold are absent
//
// Test cases:
//   Fgsea_TinyPlanted_VsR
//   Fgsea_RealMSigDB_TopHallmarks
//   Fgsea_AdaptivePermutation_StopsEarly
//   Fgsea_Determinism_BitIdentical
//   Fgsea_BHCorrection_NotAllZero
//   Fgsea_EdgeCase_TinyGeneSet
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every test that requires a device is guarded by
//   gpu_available() and calls GTEST_SKIP().

// ---- singlet-gpu GSEA header (written by gpu-kernel-dev) -------------------
#include <singlet/gpu/gsea/fgsea.h>
#include <singlet/gpu/anno/gene_set_db.h>   // GeneSetDB type (cycle 12)
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

/// Synthetic stats vector size (m = number of genes).
constexpr int kTinyM           = 200;
/// Number of planted gene sets for tiny test.
constexpr int kTinySets        = 5;
/// Genes per planted set.
constexpr int kGenesPerSet     = 20;
/// fgsea min/max set size matching R reference defaults.
constexpr int kMinSetSize      = 15;
constexpr int kMaxSetSize      = 500;
/// Number of top pathways for Jaccard check.
constexpr int kTopN            = 10;

/// Path to the fgsea R reference script.
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet/gpu/"
    "tests/refs/fgsea_r_reference.R";

/// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_gsea_fgsea_refs_tmp";

// =============================================================================
// Utility helpers (pattern from cycles 9–12)
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
// Minimal .npy v1.0 writer (float32 1-D).
// ---------------------------------------------------------------------------
static void write_npy_f32(const std::string& path, const float* data, int n) {
    std::string header_str =
        std::string("{'descr': '<f4', 'fortran_order': False, 'shape': (") +
        std::to_string(n) + ",), }";
    constexpr int kPreamble = 10;
    int raw_len   = static_cast<int>(header_str.size());
    int pad_to_64 = ((raw_len + kPreamble + 63) / 64) * 64 - kPreamble;
    while (static_cast<int>(header_str.size()) < pad_to_64 - 1)
        header_str += ' ';
    header_str += '\n';

    std::ofstream fout(path, std::ios::binary | std::ios::trunc);
    if (!fout) throw std::runtime_error("write_npy_f32: cannot open " + path);
    const char magic[] = "\x93NUMPY\x01\x00";
    fout.write(magic, 8);
    uint16_t hlen = static_cast<uint16_t>(header_str.size());
    fout.write(reinterpret_cast<const char*>(&hlen), 2);
    fout.write(header_str.c_str(), static_cast<std::streamsize>(header_str.size()));
    fout.write(reinterpret_cast<const char*>(data),
               static_cast<std::streamsize>(static_cast<size_t>(n) * sizeof(float)));
    if (!fout) throw std::runtime_error("write_npy_f32: write error " + path);
}

// ---------------------------------------------------------------------------
// Write stats to a two-column TSV: gene_idx<TAB>stat
// ---------------------------------------------------------------------------
void write_stats_tsv(const std::string& path, const std::vector<float>& stats) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open stats tsv: " + path);
    for (int i = 0; i < static_cast<int>(stats.size()); ++i)
        f << i << "\t" << stats[i] << "\n";
    if (!f) throw std::runtime_error("Write error stats tsv: " + path);
}

// ---------------------------------------------------------------------------
// Write GeneSetDB to a three-column TSV: set_name<TAB>gene_idx<TAB>weight
// ---------------------------------------------------------------------------
void write_geneset_tsv(const std::string& path,
                       const singlet::gpu::anno::GeneSetDB& db)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open geneset tsv: " + path);
    for (int s = 0; s < static_cast<int>(db.set_names.size()); ++s) {
        const auto& idxs    = db.member_gene_indices[s];
        const auto& weights = db.weights[s];
        for (int k = 0; k < static_cast<int>(idxs.size()); ++k) {
            float w = weights.empty() ? 1.0f : weights[k];
            f << db.set_names[s] << "\t" << idxs[k] << "\t" << w << "\n";
        }
    }
    if (!f) throw std::runtime_error("Write error geneset tsv: " + path);
}

// ---------------------------------------------------------------------------
// Parse the fgsea TSV output written by fgsea_r_reference.R.
// Returns a vector of (pathway, es, nes, pval, padj, size) rows.
// ---------------------------------------------------------------------------
struct FgseaRefRow {
    std::string pathway;
    float es;
    float nes;
    float pval;
    float padj;
    int   size;
    int   n_leading_edge;
};

std::vector<FgseaRefRow> parse_fgsea_tsv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open fgsea tsv: " + path);

    std::vector<FgseaRefRow> rows;
    std::string line;
    bool header = true;
    while (std::getline(f, line)) {
        if (header) { header = false; continue; }   // skip header row
        if (line.empty()) continue;
        std::istringstream ss(line);
        FgseaRefRow row{};
        std::string tok;
        std::getline(ss, row.pathway, '\t');
        std::getline(ss, tok, '\t'); row.es   = std::stof(tok);
        std::getline(ss, tok, '\t'); row.nes  = std::stof(tok);
        std::getline(ss, tok, '\t'); row.pval = std::stof(tok);
        std::getline(ss, tok, '\t'); row.padj = std::stof(tok);
        std::getline(ss, tok, '\t'); row.size = std::stoi(tok);
        std::getline(ss, tok, '\t'); row.n_leading_edge = std::stoi(tok);
        rows.push_back(std::move(row));
    }
    return rows;
}

// ---------------------------------------------------------------------------
// Spearman rank correlation (host-side).
// ---------------------------------------------------------------------------
double spearman_rho(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size() && a.size() >= 2);
    int n = static_cast<int>(a.size());

    auto rank_vec = [&](const std::vector<float>& v) {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int i, int j){ return v[i] < v[j]; });
        std::vector<double> r(n);
        for (int i = 0; i < n; ) {
            int j = i;
            while (j < n && v[idx[j]] == v[idx[i]]) ++j;
            double avg = (i + j - 1) / 2.0 + 1.0;
            for (int k2 = i; k2 < j; ++k2) r[idx[k2]] = avg;
            i = j;
        }
        return r;
    };

    auto ra = rank_vec(a);
    auto rb = rank_vec(b);

    double mean_ra = 0, mean_rb = 0;
    for (int i = 0; i < n; ++i) { mean_ra += ra[i]; mean_rb += rb[i]; }
    mean_ra /= n; mean_rb /= n;

    double cov = 0, va = 0, vb = 0;
    for (int i = 0; i < n; ++i) {
        double da = ra[i] - mean_ra, db = rb[i] - mean_rb;
        cov += da * db; va += da * da; vb += db * db;
    }
    if (va < 1e-15 || vb < 1e-15) return 1.0;
    return cov / std::sqrt(va * vb);
}

// ---------------------------------------------------------------------------
// Jaccard similarity between two ranked name lists (top-N intersection/union).
// ---------------------------------------------------------------------------
double jaccard(const std::vector<std::string>& a,
               const std::vector<std::string>& b)
{
    std::set<std::string> sa(a.begin(), a.end());
    std::set<std::string> sb(b.begin(), b.end());
    int inter = 0;
    for (const auto& x : sa)
        if (sb.count(x)) ++inter;
    int uni = static_cast<int>(sa.size() + sb.size()) - inter;
    if (uni == 0) return 1.0;
    return static_cast<double>(inter) / uni;
}

// ---------------------------------------------------------------------------
// Build a synthetic stats vector with planted positive/negative signals.
// Gene set s has genes [s * genesPerSet .. (s+1)*genesPerSet) planted with
// high positive stats (set 0) or high negative stats (set 1); rest are ~N(0,1).
// ---------------------------------------------------------------------------
std::vector<float> make_planted_stats(int m, int n_sets, int genes_per_set,
                                      uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> bg(0.0f, 1.0f);
    std::normal_distribution<float> pos(5.0f, 0.5f);
    std::normal_distribution<float> neg(-5.0f, 0.5f);

    std::vector<float> stats(m);
    for (auto& v : stats) v = bg(rng);

    for (int s = 0; s < n_sets && s * genes_per_set + genes_per_set <= m; ++s) {
        auto& dist = (s % 2 == 0) ? pos : neg;
        for (int gi = s * genes_per_set; gi < (s + 1) * genes_per_set; ++gi)
            stats[gi] = dist(rng);
    }
    return stats;
}

// ---------------------------------------------------------------------------
// Build a GeneSetDB matching the planted stats layout above.
// ---------------------------------------------------------------------------
singlet::gpu::anno::GeneSetDB make_planted_genesets(int m, int n_sets,
                                                    int genes_per_set) {
    singlet::gpu::anno::GeneSetDB db;
    db.n_genes = m;
    for (int s = 0; s < n_sets && s * genes_per_set + genes_per_set <= m; ++s) {
        db.set_names.push_back("PLANTED_SET_" + std::to_string(s));
        std::vector<int32_t> idxs;
        for (int gi = s * genes_per_set; gi < (s + 1) * genes_per_set; ++gi)
            idxs.push_back(gi);
        db.member_gene_indices.push_back(idxs);
        db.weights.emplace_back();  // uniform weights
    }
    return db;
}

// ---------------------------------------------------------------------------
// Upload a float host vector to a DeviceMemory<float>.
// ---------------------------------------------------------------------------
singlet::gpu::core::DeviceMemory<float>
upload_stats(const std::vector<float>& h, cudaStream_t stream) {
    singlet::gpu::core::DeviceMemory<float> d(h.size());
    cudaError_t e = cudaMemcpyAsync(d.get(), h.data(),
                                    h.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("upload_stats: ") +
                                 cudaGetErrorString(e));
    if (cudaStreamSynchronize(stream) != cudaSuccess)
        throw std::runtime_error("upload_stats: sync failed");
    return d;
}

// ---------------------------------------------------------------------------
// Run the R fgsea reference for a given stats + genesets pair.
// Returns the parsed rows sorted by pathway name.
// ---------------------------------------------------------------------------
std::vector<FgseaRefRow>
run_r_fgsea(const std::vector<float>& stats,
            const singlet::gpu::anno::GeneSetDB& db,
            const std::string& tag,
            int min_size = kMinSetSize,
            int max_size = kMaxSetSize,
            int nperm    = 10000,
            int seed     = 42)
{
    ensure_refs_tmp();
    std::string stats_tsv   = refs_path(tag + "_stats.tsv");
    std::string gs_tsv      = refs_path(tag + "_genesets.tsv");
    std::string out_tsv     = refs_path(tag + "_fgsea_out.tsv");

    write_stats_tsv(stats_tsv, stats);
    write_geneset_tsv(gs_tsv, db);

    std::string cmd = std::string("Rscript ") + kRefScript +
        " --stats "    + stats_tsv  +
        " --genesets " + gs_tsv     +
        " --output "   + out_tsv    +
        " --minSize "  + std::to_string(min_size) +
        " --maxSize "  + std::to_string(max_size) +
        " --nperm "    + std::to_string(nperm)    +
        " --seed "     + std::to_string(seed);
    run_cmd(cmd);

    return parse_fgsea_tsv(out_tsv);
}

}  // namespace

// =============================================================================
// Test: Fgsea_TinyPlanted_VsR
// =============================================================================
// m=200 stats with planted up/down gene sets; compare ES per pathway:
//   rel_err <= 1e-4 and p-value Spearman rho >= 0.95.
// =============================================================================

TEST(Fgsea, TinyPlanted_VsR) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device available.";

    auto stats = make_planted_stats(kTinyM, kTinySets, kGenesPerSet, kSeed);
    auto db    = make_planted_genesets(kTinyM, kTinySets, kGenesPerSet);

    // ---- Run our GPU fgsea --------------------------------------------------
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto d_stats = upload_stats(stats, stream);

    singlet::gpu::gsea::FgseaConfig cfg;
    cfg.min_set_size = kMinSetSize;
    cfg.max_set_size = kMaxSetSize;
    cfg.min_perm     = 1000;
    cfg.max_perm     = 10000;
    cfg.seed         = kSeed;

    singlet::gpu::gsea::FgseaResult our_res =
        singlet::gpu::gsea::fgsea(d_stats, db, cfg, stream);

    cudaStreamDestroy(stream);

    // ---- Run R fgsea reference ----------------------------------------------
    auto ref_rows = run_r_fgsea(stats, db, "tiny_planted");

    // Build lookup maps: pathway -> result
    std::map<std::string, singlet::gpu::gsea::PathwayResult> our_map;
    for (const auto& pr : our_res.pathways)
        our_map[pr.name] = pr;

    std::map<std::string, FgseaRefRow> ref_map;
    for (const auto& row : ref_rows)
        ref_map[row.pathway] = row;

    // Compare ES per pathway that appears in both (after filtering).
    std::vector<float> our_pvals, ref_pvals;
    int n_compared = 0;

    for (const auto& [pname, ref_row] : ref_map) {
        auto it = our_map.find(pname);
        if (it == our_map.end()) continue;
        const auto& our_row = it->second;

        // ES relative error.
        float ref_es = ref_row.es;
        float our_es = our_row.es;
        float denom  = std::max(std::abs(ref_es), 1e-6f);
        float rel_err = std::abs(our_es - ref_es) / denom;
        EXPECT_LE(rel_err, 1e-4f)
            << "ES rel_err too large for pathway " << pname
            << ": our=" << our_es << " ref=" << ref_es;

        our_pvals.push_back(our_row.p_value);
        ref_pvals.push_back(ref_row.pval);
        ++n_compared;
    }

    ASSERT_GE(n_compared, 2)
        << "Too few pathways survived filtering to compare p-values.";

    double rho = spearman_rho(our_pvals, ref_pvals);
    EXPECT_GE(rho, 0.95)
        << "p-value Spearman rho below tolerance: " << rho;
}

// =============================================================================
// Test: Fgsea_RealMSigDB_TopHallmarks
// =============================================================================
// Synthetic preranked stats on kTinyM genes; 50 gene sets simulating MSigDB
// Hallmarks (each ~20 genes in kTinyM space). Compare top-10 enriched pathways
// by NES rank: Jaccard >= 0.90 vs R fgsea.
// =============================================================================

TEST(Fgsea, RealMSigDB_TopHallmarks) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device available.";

    constexpr int kHallmarkSets = 50;
    constexpr int kHallmarkGenesPerSet = 20;   // adjusted to kTinyM=200 space

    // Build synthetic stats: standard normal.
    std::mt19937_64 rng(kSeed + 1);
    std::normal_distribution<float> ndist(0.0f, 1.0f);
    std::vector<float> stats(kTinyM);
    for (auto& v : stats) v = ndist(rng);

    // Build 50 gene sets with random membership in [0, kTinyM).
    singlet::gpu::anno::GeneSetDB db;
    db.n_genes = kTinyM;
    std::uniform_int_distribution<int> gene_dist(0, kTinyM - 1);
    for (int s = 0; s < kHallmarkSets; ++s) {
        db.set_names.push_back("HALLMARK_" + std::to_string(s));
        std::set<int> members_set;
        while (static_cast<int>(members_set.size()) < kHallmarkGenesPerSet)
            members_set.insert(gene_dist(rng));
        std::vector<int32_t> idxs(members_set.begin(), members_set.end());
        db.member_gene_indices.push_back(idxs);
        db.weights.emplace_back();
    }

    // ---- GPU fgsea ----------------------------------------------------------
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto d_stats = upload_stats(stats, stream);

    singlet::gpu::gsea::FgseaConfig cfg;
    cfg.min_set_size = kMinSetSize;
    cfg.max_set_size = kMaxSetSize;
    cfg.seed         = kSeed + 1;

    auto our_res = singlet::gpu::gsea::fgsea(d_stats, db, cfg, stream);
    cudaStreamDestroy(stream);

    // ---- R reference --------------------------------------------------------
    auto ref_rows = run_r_fgsea(stats, db, "hallmarks");

    // Sort our results by descending NES.
    std::vector<std::string> our_top, ref_top;
    {
        auto sorted_ours = our_res.pathways;
        std::sort(sorted_ours.begin(), sorted_ours.end(),
                  [](const singlet::gpu::gsea::PathwayResult& a,
                     const singlet::gpu::gsea::PathwayResult& b){
                      return std::abs(a.nes) > std::abs(b.nes);
                  });
        for (int i = 0; i < std::min(kTopN, static_cast<int>(sorted_ours.size())); ++i)
            our_top.push_back(sorted_ours[i].name);
    }
    {
        auto sorted_ref = ref_rows;
        std::sort(sorted_ref.begin(), sorted_ref.end(),
                  [](const FgseaRefRow& a, const FgseaRefRow& b){
                      return std::abs(a.nes) > std::abs(b.nes);
                  });
        for (int i = 0; i < std::min(kTopN, static_cast<int>(sorted_ref.size())); ++i)
            ref_top.push_back(sorted_ref[i].pathway);
    }

    ASSERT_GE(static_cast<int>(our_top.size()), std::min(kTopN, kHallmarkSets))
        << "Too few pathways in our result.";
    ASSERT_GE(static_cast<int>(ref_top.size()), std::min(kTopN, kHallmarkSets))
        << "Too few pathways in R reference result.";

    double j = jaccard(our_top, ref_top);
    EXPECT_GE(j, 0.90)
        << "Top-10 pathway Jaccard below tolerance: " << j;
}

// =============================================================================
// Test: Fgsea_AdaptivePermutation_StopsEarly
// =============================================================================
// Confirm that pathways with high effect sizes (extreme planted sets) have
// n_perms_used < min_perm + 1 (early stop), while borderline pathways may
// reach max_perm. At least one pathway must stop early.
// =============================================================================

TEST(Fgsea, AdaptivePermutation_StopsEarly) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device available.";

    // Use a very strong planted signal so early stopping is guaranteed.
    constexpr int kM = 200;
    constexpr int kSetsStrong = 2;
    constexpr int kGenesStrong = 30;

    std::mt19937_64 rng(kSeed + 2);
    std::normal_distribution<float> bg(0.0f, 0.1f);
    std::normal_distribution<float> sig(20.0f, 0.1f);  // extreme signal
    std::vector<float> stats(kM);
    for (auto& v : stats) v = bg(rng);
    for (int gi = 0; gi < kGenesStrong; ++gi)
        stats[gi] = sig(rng);   // strong upregulation for first set

    singlet::gpu::anno::GeneSetDB db;
    db.n_genes = kM;
    for (int s = 0; s < kSetsStrong; ++s) {
        db.set_names.push_back("EXTREME_SET_" + std::to_string(s));
        std::vector<int32_t> idxs;
        for (int gi = s * kGenesStrong; gi < (s + 1) * kGenesStrong && gi < kM; ++gi)
            idxs.push_back(gi);
        db.member_gene_indices.push_back(idxs);
        db.weights.emplace_back();
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto d_stats = upload_stats(stats, stream);

    singlet::gpu::gsea::FgseaConfig cfg;
    cfg.min_set_size = 5;   // relax to include our small planted sets
    cfg.max_set_size = 500;
    cfg.min_perm     = 1000;
    cfg.max_perm     = 10000;
    cfg.seed         = kSeed + 2;

    auto res = singlet::gpu::gsea::fgsea(d_stats, db, cfg, stream);
    cudaStreamDestroy(stream);

    // At least one pathway with extreme ES should stop before min_perm.
    bool any_early_stop = false;
    for (const auto& pr : res.pathways) {
        if (pr.n_perms_used < cfg.min_perm)
            any_early_stop = true;
    }
    EXPECT_TRUE(any_early_stop)
        << "Expected at least one pathway to use early stopping "
        << "(adaptive permutation scheme not firing).";

    // Sanity: no pathway exceeds max_perm.
    for (const auto& pr : res.pathways) {
        EXPECT_LE(pr.n_perms_used, cfg.max_perm)
            << "Pathway " << pr.name << " used more perms than max_perm.";
    }
}

// =============================================================================
// Test: Fgsea_Determinism_BitIdentical
// =============================================================================
// Same seed, same input → bit-identical ES, NES, p-values, q-values.
// =============================================================================

TEST(Fgsea, Determinism_BitIdentical) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device available.";

    auto stats = make_planted_stats(kTinyM, kTinySets, kGenesPerSet, kSeed);
    auto db    = make_planted_genesets(kTinyM, kTinySets, kGenesPerSet);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet::gpu::gsea::FgseaConfig cfg;
    cfg.min_set_size = kMinSetSize;
    cfg.max_set_size = kMaxSetSize;
    cfg.seed         = kSeed;

    auto d_stats1 = upload_stats(stats, stream);
    auto res1 = singlet::gpu::gsea::fgsea(d_stats1, db, cfg, stream);

    auto d_stats2 = upload_stats(stats, stream);
    auto res2 = singlet::gpu::gsea::fgsea(d_stats2, db, cfg, stream);

    cudaStreamDestroy(stream);

    ASSERT_EQ(res1.pathways.size(), res2.pathways.size())
        << "Different number of pathways on two determinism runs.";

    // Sort both by pathway name for stable comparison.
    auto sort_by_name = [](singlet::gpu::gsea::FgseaResult& r) {
        std::sort(r.pathways.begin(), r.pathways.end(),
                  [](const singlet::gpu::gsea::PathwayResult& a,
                     const singlet::gpu::gsea::PathwayResult& b){
                      return a.name < b.name;
                  });
    };
    sort_by_name(res1);
    sort_by_name(res2);

    for (size_t i = 0; i < res1.pathways.size(); ++i) {
        const auto& pr1 = res1.pathways[i];
        const auto& pr2 = res2.pathways[i];
        EXPECT_EQ(pr1.name, pr2.name);
        // Bit-identical floats: compare using memcmp-safe integer representation.
        uint32_t es1_bits, es2_bits, nes1_bits, nes2_bits,
                 p1_bits,  p2_bits,  q1_bits,  q2_bits;
        std::memcpy(&es1_bits,  &pr1.es,      4);
        std::memcpy(&es2_bits,  &pr2.es,      4);
        std::memcpy(&nes1_bits, &pr1.nes,     4);
        std::memcpy(&nes2_bits, &pr2.nes,     4);
        std::memcpy(&p1_bits,   &pr1.p_value, 4);
        std::memcpy(&p2_bits,   &pr2.p_value, 4);
        std::memcpy(&q1_bits,   &pr1.q_value, 4);
        std::memcpy(&q2_bits,   &pr2.q_value, 4);
        EXPECT_EQ(es1_bits,  es2_bits)  << "ES not bit-identical for " << pr1.name;
        EXPECT_EQ(nes1_bits, nes2_bits) << "NES not bit-identical for " << pr1.name;
        EXPECT_EQ(p1_bits,   p2_bits)   << "p_value not bit-identical for " << pr1.name;
        EXPECT_EQ(q1_bits,   q2_bits)   << "q_value not bit-identical for " << pr1.name;
    }
}

// =============================================================================
// Test: Fgsea_BHCorrection_NotAllZero
// =============================================================================
// BH-adjusted q-values must not be trivially zero for all pathways.
// =============================================================================

TEST(Fgsea, BHCorrection_NotAllZero) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device available.";

    auto stats = make_planted_stats(kTinyM, kTinySets, kGenesPerSet, kSeed);
    auto db    = make_planted_genesets(kTinyM, kTinySets, kGenesPerSet);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet::gpu::gsea::FgseaConfig cfg;
    cfg.min_set_size = kMinSetSize;
    cfg.max_set_size = kMaxSetSize;
    cfg.seed         = kSeed;

    auto d_stats = upload_stats(stats, stream);
    auto res = singlet::gpu::gsea::fgsea(d_stats, db, cfg, stream);
    cudaStreamDestroy(stream);

    // At least one q-value must be > 0 (BH applied, not trivially zeroed out).
    bool any_nonzero = false;
    for (const auto& pr : res.pathways)
        if (pr.q_value > 0.0f) any_nonzero = true;
    EXPECT_TRUE(any_nonzero)
        << "All q-values are zero — BH correction may not be applied.";

    // No q-value should be NaN.
    for (const auto& pr : res.pathways)
        EXPECT_FALSE(std::isnan(pr.q_value))
            << "NaN q-value for pathway " << pr.name;

    // All p-values in [0, 1].
    for (const auto& pr : res.pathways) {
        EXPECT_GE(pr.p_value, 0.0f) << "Negative p-value for " << pr.name;
        EXPECT_LE(pr.p_value, 1.0f) << "p-value > 1 for " << pr.name;
    }
}

// =============================================================================
// Test: Fgsea_EdgeCase_TinyGeneSet
// =============================================================================
// A gene set with 5 members (below min_set_size=15) must be filtered out of
// the result — it must not appear in res.pathways.
// =============================================================================

TEST(Fgsea, EdgeCase_TinyGeneSet) {
    if (!gpu_available()) GTEST_SKIP() << "No GPU device available.";

    constexpr int kM = 200;

    // Build one large valid set + one tiny set below min_set_size.
    singlet::gpu::anno::GeneSetDB db;
    db.n_genes = kM;

    // Large set: 20 genes (passes min_size=15 filter).
    db.set_names.push_back("LARGE_SET");
    std::vector<int32_t> large_idxs;
    for (int i = 0; i < 20; ++i) large_idxs.push_back(i);
    db.member_gene_indices.push_back(large_idxs);
    db.weights.emplace_back();

    // Tiny set: 5 genes (below min_size=15, must be filtered).
    db.set_names.push_back("TINY_SET_BELOW_THRESHOLD");
    std::vector<int32_t> tiny_idxs;
    for (int i = 100; i < 105; ++i) tiny_idxs.push_back(i);
    db.member_gene_indices.push_back(tiny_idxs);
    db.weights.emplace_back();

    std::mt19937_64 rng(kSeed + 5);
    std::normal_distribution<float> ndist(0.0f, 1.0f);
    std::vector<float> stats(kM);
    for (auto& v : stats) v = ndist(rng);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet::gpu::gsea::FgseaConfig cfg;
    cfg.min_set_size = kMinSetSize;   // =15; tiny set has 5 genes → filtered
    cfg.max_set_size = kMaxSetSize;
    cfg.seed         = kSeed + 5;

    auto d_stats = upload_stats(stats, stream);
    auto res = singlet::gpu::gsea::fgsea(d_stats, db, cfg, stream);
    cudaStreamDestroy(stream);

    // "TINY_SET_BELOW_THRESHOLD" must not appear in results.
    for (const auto& pr : res.pathways) {
        EXPECT_NE(pr.name, "TINY_SET_BELOW_THRESHOLD")
            << "Tiny gene set was not filtered by min_set_size.";
    }

    // "LARGE_SET" should appear (20 >= 15).
    bool large_present = false;
    for (const auto& pr : res.pathways)
        if (pr.name == "LARGE_SET") large_present = true;
    EXPECT_TRUE(large_present)
        << "LARGE_SET (20 genes >= min_size=15) was unexpectedly filtered.";
}
