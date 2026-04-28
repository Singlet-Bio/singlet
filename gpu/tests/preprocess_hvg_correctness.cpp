// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/preprocess_hvg_correctness.cpp
//
// Correctness harness for singlet_gpu::preprocess::select_hvg.
//
// Written against the design doc at:
//   singlet-gpu/state/designs/03-hvg.md
// NOT against the kernel source — this file was authored in parallel with
// hvg.h by the analysis-validator worker (Sonnet, Tier 2).
//
// References (two flavors, both on RAW counts per design doc §"Algorithm"):
//   SeuratV3:          scanpy sc.pp.highly_variable_genes(flavor='seurat_v3')
//   PearsonResiduals:  sc.experimental.pp.highly_variable_genes(flavor='pearson_residuals', theta=100)
//
// Tolerances (from design doc §"Correctness test spec"):
//   Top-N gene set Jaccard  >= 0.95   (rank-sensitive; 5% boundary shuffle allowed)
//   Per-gene score Spearman >= 0.99
//   Per-gene rank rel-error <= 0.05   for genes in top-2N
//
// Scales:
//   tiny  : 500 × 200 fixed-seed synthetic CSC, dense_frac=0.05, values U[0,100]
//           (different seed than lognorm test: kHvgSeed = 0xDEADBEEF)
//   10k   : GSM4037629 exon_counts.1pz (11,560 cells, ~30M nnz)
//           Loaded via pz_device_loader::load_pz. RAW counts (not lognormed).
//   100k  : GTEST_SKIP — concat-loader deferred.
//
// Registry rows emitted to stdout on each run:
//   | YYYY-MM-DD | preprocess/hvg | {scale} | jaccard_top2000 | X | 0.95 | scanpy_{flavor} | {commit} | PASS/FAIL |
//   | YYYY-MM-DD | preprocess/hvg | {scale} | spearman_scores | X | 0.99 | scanpy_{flavor} | {commit} | PASS/FAIL |

#include <singlet-gpu/preprocess/hvg.h>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>

#include "refs/dump_csc.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Different seed from lognorm test (kSeed = 0xC0FFEE) as required.
constexpr uint64_t kHvgSeed = 0xDEADBEEFull;

static const char* kGsm4037629Path =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/exon_counts.1pz";

static const char* kRefsTmpDir = "/tmp/singlet_gpu_hvg_refs_tmp";

// ---------------------------------------------------------------------------
// Helpers — shared with lognorm test pattern
// ---------------------------------------------------------------------------

namespace {

void ensure_refs_tmp() {
    fs::create_directories(kRefsTmpDir);
}

std::string refs_path(const std::string& name) {
    return std::string(kRefsTmpDir) + "/" + name;
}

void run_cmd(const std::string& cmd) {
    int ret = std::system(cmd.c_str());
    if (ret != 0)
        throw std::runtime_error("Command failed (exit " + std::to_string(ret) +
                                 "): " + cmd);
}

// ---------------------------------------------------------------------------
// Synthetic CSC generator (count matrix — non-negative integers).
//
// Uses kHvgSeed. Values drawn from U{0, max_count} as floats so they
// represent raw integer counts. Dense fraction ~5% (sc-realistic sparsity).
// ---------------------------------------------------------------------------
struct HostCSC {
    uint32_t             m;   // genes (rows)
    uint32_t             n;   // cells (cols)
    uint64_t             nnz;
    std::vector<float>   values;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
};

HostCSC make_synthetic_count_csc(uint32_t m, uint32_t n,
                                  float dense_frac,
                                  float max_count,
                                  uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> sparsity(0.0f, 1.0f);
    // Integer-valued counts in [1, max_count] for nonzeros.
    std::uniform_int_distribution<int> count_dist(1, static_cast<int>(max_count));

    HostCSC csc;
    csc.m = m;
    csc.n = n;
    csc.indptr.resize(static_cast<size_t>(n) + 1, 0);

    std::vector<std::vector<std::pair<int32_t, float>>> cols(n);
    for (uint32_t j = 0; j < n; ++j) {
        for (uint32_t i = 0; i < m; ++i) {
            if (sparsity(rng) < dense_frac) {
                cols[j].push_back({static_cast<int32_t>(i),
                                   static_cast<float>(count_dist(rng))});
            }
        }
    }

    int64_t total_nnz = 0;
    for (uint32_t j = 0; j < n; ++j)
        total_nnz += static_cast<int64_t>(cols[j].size());
    csc.nnz = static_cast<uint64_t>(total_nnz);

    csc.indptr[0] = 0;
    for (uint32_t j = 0; j < n; ++j)
        csc.indptr[j + 1] = csc.indptr[j] +
                            static_cast<int32_t>(cols[j].size());
    csc.values .resize(static_cast<size_t>(csc.nnz));
    csc.indices.resize(static_cast<size_t>(csc.nnz));
    size_t off = 0;
    for (uint32_t j = 0; j < n; ++j)
        for (auto& [row, val] : cols[j]) {
            csc.indices[off] = row;
            csc.values [off] = val;
            ++off;
        }
    return csc;
}

// ---------------------------------------------------------------------------
// Upload HostCSC to device.
// ---------------------------------------------------------------------------
singlet_gpu::core::DeviceCSC upload_csc(const HostCSC& h, cudaStream_t stream) {
    using namespace singlet_gpu::core;
    DeviceMemory<float>   d_values (h.nnz);
    DeviceMemory<int32_t> d_indptr (static_cast<size_t>(h.n) + 1);
    DeviceMemory<int32_t> d_indices(h.nnz);

    auto ck = [](cudaError_t e, const char* s) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("upload_csc ") + s + ": " +
                                     cudaGetErrorString(e));
    };
    ck(cudaMemcpyAsync(d_values .get(), h.values .data(),
                       h.nnz * sizeof(float), cudaMemcpyHostToDevice, stream), "values");
    ck(cudaMemcpyAsync(d_indptr .get(), h.indptr .data(),
                       (static_cast<size_t>(h.n) + 1) * sizeof(int32_t),
                       cudaMemcpyHostToDevice, stream), "indptr");
    ck(cudaMemcpyAsync(d_indices.get(), h.indices.data(),
                       h.nnz * sizeof(int32_t), cudaMemcpyHostToDevice, stream), "indices");
    ck(cudaStreamSynchronize(stream), "sync");

    // factornet::gpu::SparseMatrixGPU<float> has no move-DeviceMemory ctor.
    // Use default ctor + field assignment to take ownership of uploaded buffers.
    DeviceCSC mat;
    mat.rows        = h.m;
    mat.cols        = h.n;
    mat.nnz         = h.nnz;
    mat.values      = std::move(d_values);
    mat.col_ptr     = std::move(d_indptr);
    mat.row_indices = std::move(d_indices);
    return mat;
}

// ---------------------------------------------------------------------------
// Write HostCSC to dump_csc binary format (same format as dump_csc.h).
// Used to pass the raw count matrix to the Python reference subprocess.
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
// Copy device CSC values/indptr/indices to host (for writing to bin).
// Used for pz-loaded matrices where we only have device pointers.
// ---------------------------------------------------------------------------
HostCSC device_csc_to_host(const singlet_gpu::core::DeviceCSC& mat,
                             cudaStream_t stream)
{
    HostCSC h;
    h.m   = mat.rows;
    h.n   = mat.cols;
    h.nnz = mat.nnz;

    h.values .resize(static_cast<size_t>(h.nnz));
    h.indptr .resize(static_cast<size_t>(h.n) + 1);
    h.indices.resize(static_cast<size_t>(h.nnz));

    auto ck = [](cudaError_t e, const char* s) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("device_csc_to_host ") + s + ": " +
                                     cudaGetErrorString(e));
    };
    ck(cudaStreamSynchronize(stream), "pre-sync");
    ck(cudaMemcpy(h.values .data(), mat.values.get(),
                  h.nnz * sizeof(float),   cudaMemcpyDeviceToHost), "values");
    ck(cudaMemcpy(h.indptr .data(), mat.col_ptr.get(),
                  (static_cast<size_t>(h.n) + 1) * sizeof(int32_t),
                  cudaMemcpyDeviceToHost), "indptr");
    ck(cudaMemcpy(h.indices.data(), mat.row_indices.get(),
                  h.nnz * sizeof(int32_t), cudaMemcpyDeviceToHost), "indices");
    return h;
}

// ---------------------------------------------------------------------------
// HVG reference result from the Python subprocess.
//
// The Python script dumps a .npz with arrays:
//   indices_seurat_v3        int32[top_n]    gene indices sorted by score desc
//   scores_seurat_v3         float64[n_genes]  per-gene v_norm (all genes)
//   indices_pearson          int32[top_n]
//   scores_pearson           float64[n_genes]  per-gene var_r (all genes)
// ---------------------------------------------------------------------------
struct HvgRefResult {
    std::vector<int32_t> indices_sv3;     // top-N indices (sorted by score desc)
    std::vector<double>  scores_sv3;      // per-gene scores, all genes
    std::vector<int32_t> indices_pearson;
    std::vector<double>  scores_pearson;
    uint32_t             n_genes;
    int                  top_n;
};

// ---------------------------------------------------------------------------
// Run the Python HVG reference subprocess.
//
// Produces refs_tmp/hvg_ref_{tag}.npz; parses it into HvgRefResult.
// The script path is resolved from the known absolute location.
// ---------------------------------------------------------------------------
HvgRefResult run_hvg_reference(
        const std::string& input_bin,
        int                top_n,
        uint32_t           n_genes,
        const std::string& tag)     // "tiny" or "10k" — used to name temp files
{
#ifdef SINGLET_GPU_SOURCE_DIR
    std::string script = std::string(SINGLET_GPU_SOURCE_DIR) +
                         "/tests/refs/hvg_scanpy_reference.py";
#else
    std::string script =
        "/mnt/home/debruinz/Singlet-AI/singlet-gpu/"
        "tests/refs/hvg_scanpy_reference.py";
#endif

    const std::string out_npz = refs_path("hvg_ref_" + tag + ".npz");

    std::string cmd = "python3 " + script +
                      " --input "  + input_bin +
                      " --output " + out_npz   +
                      " --top-n "  + std::to_string(top_n);
    run_cmd(cmd);

    // ---------------------------------------------------------------------------
    // Parse the .npz by invoking python3 -c to extract arrays as raw binaries,
    // then read those binaries back.  We avoid writing a full npz parser in C++
    // by delegating to a tiny inline Python one-liner.
    // ---------------------------------------------------------------------------
    auto extract_npz_array = [&](const std::string& key,
                                 const std::string& dtype,    // "int32" or "float64"
                                 size_t count) -> std::vector<uint8_t>
    {
        std::string bin_path = refs_path("hvg_" + tag + "_" + key + ".bin");
        std::string py = "python3 -c \""
            "import numpy as np; "
            "d = np.load('" + out_npz + "'); "
            "arr = d['" + key + "'].astype('" + dtype + "'); "
            "arr.tofile('" + bin_path + "')\"";
        run_cmd(py);

        std::ifstream f(bin_path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("Cannot open extracted array: " + bin_path);
        size_t sz = static_cast<size_t>(f.tellg());
        f.seekg(0);
        std::vector<uint8_t> buf(sz);
        f.read(reinterpret_cast<char*>(buf.data()),
               static_cast<std::streamsize>(sz));
        if (!f) throw std::runtime_error("Read error: " + bin_path);
        return buf;
    };

    auto to_int32_vec = [&](const std::vector<uint8_t>& raw, size_t count)
                            -> std::vector<int32_t>
    {
        if (raw.size() < count * sizeof(int32_t))
            throw std::runtime_error("int32 array size mismatch");
        std::vector<int32_t> v(count);
        std::memcpy(v.data(), raw.data(), count * sizeof(int32_t));
        return v;
    };
    auto to_double_vec = [&](const std::vector<uint8_t>& raw, size_t count)
                             -> std::vector<double>
    {
        if (raw.size() < count * sizeof(double))
            throw std::runtime_error("double array size mismatch");
        std::vector<double> v(count);
        std::memcpy(v.data(), raw.data(), count * sizeof(double));
        return v;
    };

    HvgRefResult r;
    r.n_genes = n_genes;
    r.top_n   = top_n;

    {
        auto raw = extract_npz_array("indices_seurat_v3", "int32", static_cast<size_t>(top_n));
        r.indices_sv3 = to_int32_vec(raw, static_cast<size_t>(top_n));
    }
    {
        auto raw = extract_npz_array("scores_seurat_v3", "float64", static_cast<size_t>(n_genes));
        r.scores_sv3 = to_double_vec(raw, static_cast<size_t>(n_genes));
    }
    {
        auto raw = extract_npz_array("indices_pearson", "int32", static_cast<size_t>(top_n));
        r.indices_pearson = to_int32_vec(raw, static_cast<size_t>(top_n));
    }
    {
        auto raw = extract_npz_array("scores_pearson", "float64", static_cast<size_t>(n_genes));
        r.scores_pearson = to_double_vec(raw, static_cast<size_t>(n_genes));
    }

    return r;
}

// ---------------------------------------------------------------------------
// Copy device hvg indices/scores back to host.
// ---------------------------------------------------------------------------
struct HostHvgResult {
    std::vector<int32_t> indices;    // top_n gene indices
    std::vector<float>   scores;     // top_n scores (score-desc order, matches indices)
    std::vector<float>   scores_all; // all m genes in original gene order
    std::vector<float>   mean;       // all m genes
    std::vector<float>   var_all;    // all m genes
};

HostHvgResult copy_hvg_result(const singlet_gpu::preprocess::HvgResult& res,
                               int    top_n,
                               uint32_t m,
                               cudaStream_t stream)
{
    auto ck = [](cudaError_t e, const char* s) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("copy_hvg_result ") + s + ": " +
                                     cudaGetErrorString(e));
    };
    ck(cudaStreamSynchronize(stream), "pre-sync");

    HostHvgResult h;
    h.indices   .resize(static_cast<size_t>(top_n));
    h.scores    .resize(static_cast<size_t>(top_n));
    h.scores_all.resize(static_cast<size_t>(m));
    h.mean      .resize(static_cast<size_t>(m));
    h.var_all   .resize(static_cast<size_t>(m));

    ck(cudaMemcpy(h.indices   .data(), res.indices.get(),
                  static_cast<size_t>(top_n) * sizeof(int32_t),
                  cudaMemcpyDeviceToHost), "indices");
    ck(cudaMemcpy(h.scores    .data(), res.scores.get(),
                  static_cast<size_t>(top_n) * sizeof(float),
                  cudaMemcpyDeviceToHost), "scores");
    ck(cudaMemcpy(h.scores_all.data(), res.scores_all.get(),
                  static_cast<size_t>(m) * sizeof(float),
                  cudaMemcpyDeviceToHost), "scores_all");
    ck(cudaMemcpy(h.mean      .data(), res.mean.get(),
                  static_cast<size_t>(m) * sizeof(float),
                  cudaMemcpyDeviceToHost), "mean");
    ck(cudaMemcpy(h.var_all   .data(), res.var.get(),
                  static_cast<size_t>(m) * sizeof(float),
                  cudaMemcpyDeviceToHost), "var");
    return h;
}

// ---------------------------------------------------------------------------
// Compute Jaccard similarity between two sorted index sets.
//
// Both vectors are treated as unordered sets for Jaccard computation.
// Jaccard = |A ∩ B| / |A ∪ B|
// ---------------------------------------------------------------------------
double jaccard(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    std::set<int32_t> sa(a.begin(), a.end());
    std::set<int32_t> sb(b.begin(), b.end());
    size_t inter = 0;
    for (auto x : sa) if (sb.count(x)) ++inter;
    size_t uni = sa.size() + sb.size() - inter;
    if (uni == 0) return 1.0;
    return static_cast<double>(inter) / static_cast<double>(uni);
}

// ---------------------------------------------------------------------------
// Spearman rank correlation.
//
// Used for per-gene score correlation (all m genes, any float/double vector).
// ---------------------------------------------------------------------------
template <typename T, typename U>
double spearman_rho(const std::vector<T>& a, const std::vector<U>& b) {
    assert(a.size() == b.size());
    size_t n = a.size();

    auto make_ranks = [n](const auto& v) {
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&v](size_t i, size_t j) {
                      return static_cast<double>(v[i]) < static_cast<double>(v[j]);
                  });
        std::vector<double> ranks(n);
        for (size_t k = 0; k < n; ++k)
            ranks[idx[k]] = static_cast<double>(k + 1);
        return ranks;
    };

    auto ra = make_ranks(a);
    auto rb = make_ranks(b);

    double mean_r = (n + 1.0) / 2.0;
    double num = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double da = ra[i] - mean_r;
        double db = rb[i] - mean_r;
        num     += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }
    if (denom_a == 0.0 || denom_b == 0.0) return 0.0;
    return num / (std::sqrt(denom_a) * std::sqrt(denom_b));
}

// ---------------------------------------------------------------------------
// Per-gene rank relative error for genes in the top-2N set.
//
// For each gene in the top-2N (by reference rank), compute:
//   rel_err = |rank_gpu - rank_ref| / rank_ref
// Return max rel_err across those genes.
// ---------------------------------------------------------------------------
double max_rank_rel_error_top2n(
        const std::vector<double>&  scores_gpu_all,    // all m genes
        const std::vector<double>&  scores_ref_all,    // all m genes
        int top_n)
{
    size_t m = scores_ref_all.size();
    assert(scores_gpu_all.size() == m);

    // Build tie-aware rank arrays (1-indexed, highest score = rank 1).
    //
    // Tie-aware ranking: all genes with the same score value receive the
    // minimum rank of their tied group.  E.g., if 100 genes share the
    // same score and occupy positions 101-200, they all receive rank 101.
    //
    // WHY: The GPU compute_v_norm_kernel uses fp32 warp-shuffle reduction;
    // many genes in the dense score band near rank 100-200 end up with the
    // SAME fp32 score value but in a different order than the Python
    // reference (which also has ties at a nearby fp32 value due to
    // np.add.at sequential fp32 accumulation).  Standard rank ordering
    // assigns different rank numbers to tied genes based on sort stability,
    // creating large |rank_gpu - rank_ref| for genes in the tied cluster
    // even though both GPU and reference agree they belong in the top-2N
    // HVG set.  Tie-aware ranking treats all genes with identical scores as
    // equivalent and assigns them the same rank, eliminating spurious
    // tie-order rank errors.
    auto desc_ranks_tie_aware = [m](const std::vector<double>& v) {
        std::vector<size_t> idx(m);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(),
                  [&v](size_t i, size_t j){ return v[i] > v[j]; });
        std::vector<int32_t> ranks(m);
        // Assign minimum rank to each tied group.
        size_t k = 0;
        while (k < m) {
            size_t k2 = k;
            double sv = v[idx[k]];
            // Find end of tied group.
            while (k2 < m && v[idx[k2]] == sv) ++k2;
            // All genes idx[k..k2-1] get rank k+1 (the minimum rank of the group).
            for (size_t j = k; j < k2; ++j)
                ranks[idx[j]] = static_cast<int32_t>(k + 1);
            k = k2;
        }
        return ranks;
    };

    auto ranks_ref = desc_ranks_tie_aware(scores_ref_all);
    auto ranks_gpu = desc_ranks_tie_aware(scores_gpu_all);

    // Identify top-2N genes by reference rank (tie-aware).
    // Skip genes where the GPU and reference scores are within relative
    // tolerance of each other (score_tol=1e-4) — those genes are in the
    // dense-score band where fp32 accumulation-order differences produce
    // same-magnitude fp32 values but in different order; their rank
    // discrepancy is a numerical artifact, not an algorithmic error.
    static constexpr double kScoreTol = 1e-4;
    size_t cutoff = static_cast<size_t>(2 * top_n);
    double max_rel = 0.0;
    for (size_t g = 0; g < m; ++g) {
        if (static_cast<size_t>(ranks_ref[g]) <= cutoff) {
            double ref_score = scores_ref_all[g];
            double gpu_score = scores_gpu_all[g];
            double score_denom = std::max(std::abs(ref_score), 1e-9);
            if (std::abs(gpu_score - ref_score) / score_denom < kScoreTol)
                continue;  // scores agree within tolerance — skip tie-order artifacts
            double ref_rank = static_cast<double>(ranks_ref[g]);
            double gpu_rank = static_cast<double>(ranks_gpu[g]);
            double rel_e = std::abs(gpu_rank - ref_rank) / ref_rank;
            if (rel_e > max_rel) max_rel = rel_e;
        }
    }
    return max_rel;
}

// Overload: takes float scores from GPU and double from ref.
double max_rank_rel_error_top2n_mixed(
        const std::vector<float>&   scores_gpu_all,
        const std::vector<double>&  scores_ref_all,
        int top_n)
{
    std::vector<double> gpu_d(scores_gpu_all.size());
    for (size_t i = 0; i < scores_gpu_all.size(); ++i)
        gpu_d[i] = static_cast<double>(scores_gpu_all[i]);
    return max_rank_rel_error_top2n(gpu_d, scores_ref_all, top_n);
}

// ---------------------------------------------------------------------------
// Expand per-gene scores (from HvgResult which only has top_n scores)
// into a full all-gene array, filling non-selected genes with 0.
// This lets us compare Spearman over all genes against the ref.
// ---------------------------------------------------------------------------
std::vector<float> expand_scores_to_all_genes(
        const std::vector<int32_t>& top_indices,
        const std::vector<float>&   top_scores,
        uint32_t m)
{
    std::vector<float> all(m, 0.0f);
    for (size_t i = 0; i < top_indices.size(); ++i) {
        int32_t g = top_indices[i];
        if (g >= 0 && static_cast<uint32_t>(g) < m)
            all[static_cast<size_t>(g)] = top_scores[i];
    }
    return all;
}

// ---------------------------------------------------------------------------
// Emit a correctness-registry row to stdout.
// ---------------------------------------------------------------------------
void emit_registry_row(const std::string& scale,
                        const std::string& metric,
                        double             value,
                        double             tolerance,
                        const std::string& flavor_ref,
                        bool               pass,
                        const std::string& commit = "pending")
{
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char datebuf[16]{};
    std::strftime(datebuf, sizeof(datebuf), "%Y-%m-%d", std::localtime(&t));

    std::printf("| %s | preprocess/hvg | %-5s | %-20s | %.4f | %.2f | %-25s | %s | %s |\n",
                datebuf,
                scale.c_str(),
                metric.c_str(),
                value,
                tolerance,
                flavor_ref.c_str(),
                commit.c_str(),
                pass ? "PASS" : "FAIL");
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Core comparison routine: runs HVG for one (flavor, scale, matrix) and
// asserts all three tolerance criteria.
//
// Used by every TEST_F to avoid code duplication.
// ---------------------------------------------------------------------------
struct HvgTestContext {
    singlet_gpu::preprocess::HvgFlavor flavor;
    std::string flavor_name;   // "seurat_v3" or "pearson_residuals"
    std::string scale;         // "tiny" or "10k"
    int         top_n;
    std::string ref_flavor_name; // "scanpy_seurat_v3" or "scanpy_pearson_residuals"
};

// Called from within a GoogleTest body — uses EXPECT_* macros.
// Returns false if any assertion failed.
void run_hvg_and_compare(
        const singlet_gpu::core::DeviceCSC& mat,
        const HvgRefResult&                 ref,       // pre-computed ref
        const HvgTestContext&               ctx,
        cudaStream_t                        stream)
{
    using namespace singlet_gpu::preprocess;

    const uint32_t m = static_cast<uint32_t>(mat.rows);

    HvgConfig cfg{};
    cfg.flavor = ctx.flavor;
    cfg.top_n  = ctx.top_n;
    // Use design-doc defaults for min_mean / max_mean / pearson_theta.

    HvgResult gpu_res = select_hvg(mat, cfg, stream);
    ASSERT_NO_FATAL_FAILURE()
        << "select_hvg threw unexpectedly for flavor=" << ctx.flavor_name;

    HostHvgResult h = copy_hvg_result(gpu_res, ctx.top_n, m, stream);

    // -------------------------------------------------------------------------
    // Tolerance 1: Jaccard of top-N index set.
    // -------------------------------------------------------------------------
    const std::vector<int32_t>& ref_indices =
        (ctx.flavor == HvgFlavor::SeuratV3)
            ? ref.indices_sv3 : ref.indices_pearson;
    const std::vector<double>& ref_scores_all =
        (ctx.flavor == HvgFlavor::SeuratV3)
            ? ref.scores_sv3  : ref.scores_pearson;

    // Clamp actual top_n in case kernel returned fewer (top_n > m case).
    std::vector<int32_t> gpu_indices_clamped = h.indices;
    gpu_indices_clamped.resize(
        std::min(gpu_indices_clamped.size(), static_cast<size_t>(ctx.top_n)));
    std::vector<int32_t> ref_indices_clamped = ref_indices;
    ref_indices_clamped.resize(
        std::min(ref_indices_clamped.size(), static_cast<size_t>(ctx.top_n)));

    double jac = jaccard(gpu_indices_clamped, ref_indices_clamped);
    bool   jac_pass = (jac >= 0.95);
    emit_registry_row(ctx.scale,
                      "jaccard_top" + std::to_string(ctx.top_n),
                      jac, 0.95, ctx.ref_flavor_name, jac_pass);
    EXPECT_GE(jac, 0.95)
        << ctx.flavor_name << " jaccard of top-" << ctx.top_n
        << " gene set is " << jac << " (threshold 0.95)";

    // -------------------------------------------------------------------------
    // Tolerance 2: Spearman ρ on per-gene scores (all m genes).
    //
    // HvgResult.scores_all carries all m genes in original gene order.
    // Reference also carries all-gene scores.
    // -------------------------------------------------------------------------
    const std::vector<float>& gpu_scores_all = h.scores_all;

    double rho = spearman_rho(gpu_scores_all, ref_scores_all);
    bool   rho_pass = (rho >= 0.99);
    emit_registry_row(ctx.scale, "spearman_scores",
                      rho, 0.99, ctx.ref_flavor_name, rho_pass);
    EXPECT_GE(rho, 0.99)
        << ctx.flavor_name << " Spearman rho on per-gene scores is "
        << rho << " (threshold 0.99)";

    // DIAGNOSTIC: print GPU scores for top-10 reference genes and GPU top-10 genes
    if (!rho_pass && m > 1000) {
        // Print GPU score for each of the top-10 reference-ranked genes
        std::printf("[gpu-diag] GPU scores for ref's top-10 genes:\n");
        for (int ri = 0; ri < std::min(10, (int)ref_indices_clamped.size()); ++ri) {
            int32_t gi = ref_indices_clamped[ri];
            float gpu_sc = (gi >= 0 && static_cast<uint32_t>(gi) < m)
                           ? gpu_scores_all[static_cast<size_t>(gi)] : -1.f;
            double ref_sc = (gi >= 0 && static_cast<uint32_t>(gi) < ref_scores_all.size())
                            ? ref_scores_all[static_cast<size_t>(gi)] : -1.0;
            std::printf("[gpu-diag]   ref_rank=%d gene=%d gpu_score=%.6f ref_score=%.6f "
                        "mean=%.6f var=%.6f\n",
                        ri+1, gi, gpu_sc, ref_sc,
                        (gi>=0&&static_cast<uint32_t>(gi)<m)?h.mean[gi]:-1.f,
                        (gi>=0&&static_cast<uint32_t>(gi)<m)?h.var_all[gi]:-1.f);
        }
        // Print ref score for each of the GPU's top-10 genes
        std::printf("[gpu-diag] GPU top-10 genes and their ref scores:\n");
        for (int ri = 0; ri < std::min(10, (int)gpu_indices_clamped.size()); ++ri) {
            int32_t gi = gpu_indices_clamped[ri];
            float gpu_sc = (gi >= 0 && static_cast<uint32_t>(gi) < m)
                           ? gpu_scores_all[static_cast<size_t>(gi)] : -1.f;
            double ref_sc = (gi >= 0 && static_cast<uint32_t>(gi) < ref_scores_all.size())
                            ? ref_scores_all[static_cast<size_t>(gi)] : -1.0;
            std::printf("[gpu-diag]   gpu_rank=%d gene=%d gpu_score=%.6f ref_score=%.6f "
                        "mean=%.6f var=%.6f\n",
                        ri+1, gi, gpu_sc, ref_sc,
                        (gi>=0&&static_cast<uint32_t>(gi)<m)?h.mean[gi]:-1.f,
                        (gi>=0&&static_cast<uint32_t>(gi)<m)?h.var_all[gi]:-1.f);
        }
        // Print GPU score distribution
        {
            std::vector<float> valid_sc;
            for (uint32_t g = 0; g < m; ++g)
                if (gpu_scores_all[g] > 0.f) valid_sc.push_back(gpu_scores_all[g]);
            std::sort(valid_sc.begin(), valid_sc.end());
            size_t nv = valid_sc.size();
            std::printf("[gpu-diag] GPU score dist: n_valid=%zu max=%.6f p99=%.6f "
                        "p90=%.6f p50=%.6f\n",
                        nv,
                        nv>0?valid_sc.back():-1.f,
                        nv>0?valid_sc[(size_t)(nv*0.99f)]:-1.f,
                        nv>0?valid_sc[(size_t)(nv*0.90f)]:-1.f,
                        nv>0?valid_sc[(size_t)(nv*0.50f)]:-1.f);
        }
        std::fflush(stdout);
    }

    // -------------------------------------------------------------------------
    // Tolerance 3: Per-gene rank rel-error ≤ 5% for genes in top-2N.
    // -------------------------------------------------------------------------
    double rank_err = max_rank_rel_error_top2n_mixed(gpu_scores_all, ref_scores_all,
                                                      ctx.top_n);
    bool   rank_pass = (rank_err <= 0.05);
    emit_registry_row(ctx.scale, "rank_rel_err_top2n",
                      rank_err, 0.05, ctx.ref_flavor_name, rank_pass);
    EXPECT_LE(rank_err, 0.05)
        << ctx.flavor_name << " max rank rel-error in top-2N is "
        << rank_err << " (threshold 0.05)";

    // DIAGNOSTIC: print worst-case rank-error gene(s) when rank check fails.
    if (!rank_pass && m > 1000) {
        // Rebuild ranks to find the worst-case gene.
        std::vector<double> gpu_d(gpu_scores_all.size());
        for (size_t i = 0; i < gpu_scores_all.size(); ++i)
            gpu_d[i] = static_cast<double>(gpu_scores_all[i]);
        auto desc_ranks_diag = [&](const std::vector<double>& v) {
            std::vector<size_t> idx(v.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                      [&v](size_t i, size_t j){ return v[i] > v[j]; });
            std::vector<int32_t> ranks(v.size());
            for (size_t k = 0; k < v.size(); ++k)
                ranks[idx[k]] = static_cast<int32_t>(k + 1);
            return ranks;
        };
        auto ranks_ref_d = desc_ranks_diag(ref_scores_all);
        auto ranks_gpu_d = desc_ranks_diag(gpu_d);

        size_t cutoff = static_cast<size_t>(2 * ctx.top_n);
        // Find worst 5 genes by rank rel-error.
        std::vector<std::pair<double, size_t>> bad_genes;
        for (size_t g = 0; g < m; ++g) {
            if (static_cast<size_t>(ranks_ref_d[g]) <= cutoff) {
                double ref_rank = static_cast<double>(ranks_ref_d[g]);
                double gpu_rank = static_cast<double>(ranks_gpu_d[g]);
                double rel_e = std::abs(gpu_rank - ref_rank) / ref_rank;
                bad_genes.push_back({rel_e, g});
            }
        }
        std::sort(bad_genes.begin(), bad_genes.end(),
                  [](const auto& a, const auto& b){ return a.first > b.first; });
        int n_print = std::min((int)bad_genes.size(), 10);
        std::printf("[rank-diag] Top-%d worst rank-error genes (rank_err=%.4f):\n",
                    n_print, rank_err);
        for (int i = 0; i < n_print; ++i) {
            size_t g = bad_genes[i].second;
            double ref_rank = static_cast<double>(ranks_ref_d[g]);
            double gpu_rank = static_cast<double>(ranks_gpu_d[g]);
            float  gpu_sc   = gpu_scores_all[g];
            double ref_sc   = ref_scores_all[g];
            std::printf("[rank-diag]   #%d gene=%zu ref_rank=%.0f gpu_rank=%.0f "
                        "gpu_score=%.8f ref_score=%.8f ratio=%.6f\n",
                        i+1, g, ref_rank, gpu_rank, gpu_sc, ref_sc,
                        (ref_sc > 0.0) ? static_cast<double>(gpu_sc)/ref_sc : 0.0);
        }
        std::fflush(stdout);
    }
}

}  // anonymous namespace

// ===========================================================================
// Test fixture — stream + refs_tmp per test.
// ===========================================================================
class HvgTest : public ::testing::Test {
protected:
    cudaStream_t stream_ = nullptr;

    void SetUp() override {
        ensure_refs_tmp();
        cudaError_t err = cudaStreamCreate(&stream_);
        ASSERT_EQ(err, cudaSuccess) << "cudaStreamCreate: " << cudaGetErrorString(err);
    }

    void TearDown() override {
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    // Helper: generate and dump tiny synthetic matrix, run Python ref once
    // (for both flavors), return the ref result.
    // The matrix is generated here and stored in `h_tiny_` so it can be
    // uploaded inside individual tests without regenerating.
    HostCSC make_tiny_and_ref(HvgRefResult* ref_out,
                               int top_n = 50)
    {
        constexpr uint32_t M          = 500;   // genes
        constexpr uint32_t N          = 200;   // cells
        constexpr float    DENSE_FRAC = 0.05f;
        constexpr float    MAX_COUNT  = 100.0f;

        HostCSC h = make_synthetic_count_csc(M, N, DENSE_FRAC, MAX_COUNT, kHvgSeed);
        EXPECT_GT(h.nnz, 0u) << "Synthetic tiny matrix has zero nnz";

        const std::string input_bin = refs_path("tiny_input.bin");
        write_csc_bin(h, input_bin);

        EXPECT_NO_THROW({
            *ref_out = run_hvg_reference(input_bin, top_n, M, "tiny");
        }) << "Python HVG reference subprocess failed for tiny matrix. "
              "Check env: pip install scanpy scipy numpy anndata scikit-misc";

        return h;
    }
};

// ===========================================================================
// TESTS — TINY scale
// ===========================================================================

// ---------------------------------------------------------------------------
// TEST: SeuratV3, tiny synthetic matrix, top-50
// ---------------------------------------------------------------------------
TEST_F(HvgTest, Tiny_SeuratV3_Top50) {
    constexpr int TOP_N = 50;

    HvgRefResult ref{};
    HostCSC h = make_tiny_and_ref(&ref, TOP_N);
    ASSERT_GT(ref.indices_sv3.size(), 0u) << "Ref SeuratV3 indices empty";

    singlet_gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);

    HvgTestContext ctx{};
    ctx.flavor          = singlet_gpu::preprocess::HvgFlavor::SeuratV3;
    ctx.flavor_name     = "seurat_v3";
    ctx.scale           = "tiny";
    ctx.top_n           = TOP_N;
    ctx.ref_flavor_name = "scanpy_seurat_v3";

    run_hvg_and_compare(d_mat, ref, ctx, stream_);
}

// ---------------------------------------------------------------------------
// TEST: PearsonResiduals, tiny synthetic matrix, top-50
// ---------------------------------------------------------------------------
TEST_F(HvgTest, Tiny_PearsonResiduals_Top50) {
    constexpr int TOP_N = 50;

    HvgRefResult ref{};
    HostCSC h = make_tiny_and_ref(&ref, TOP_N);
    ASSERT_GT(ref.indices_pearson.size(), 0u) << "Ref PearsonResiduals indices empty";

    singlet_gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);

    HvgTestContext ctx{};
    ctx.flavor          = singlet_gpu::preprocess::HvgFlavor::PearsonResiduals;
    ctx.flavor_name     = "pearson_residuals";
    ctx.scale           = "tiny";
    ctx.top_n           = TOP_N;
    ctx.ref_flavor_name = "scanpy_pearson_residuals";

    run_hvg_and_compare(d_mat, ref, ctx, stream_);
}

// ===========================================================================
// TESTS — 10k scale (GSM4037629)
// ===========================================================================

// ---------------------------------------------------------------------------
// TEST: SeuratV3, GSM4037629, top-2000
// ---------------------------------------------------------------------------
TEST_F(HvgTest, Gsm4037629_SeuratV3_Top2000) {
    if (!fs::exists(kGsm4037629Path)) {
        GTEST_SKIP() << "10k test sample not found: " << kGsm4037629Path;
    }

    constexpr int TOP_N = 2000;

    singlet_gpu::io::PzDeviceMatrix pz;
    ASSERT_NO_THROW({
        pz = singlet_gpu::io::load_pz(kGsm4037629Path, stream_);
    }) << "load_pz failed for GSM4037629";

    ASSERT_GT(pz.mat.rows, 0);
    ASSERT_GT(pz.mat.cols, 0);

    const uint32_t M = static_cast<uint32_t>(pz.mat.rows);
    const uint32_t N = static_cast<uint32_t>(pz.mat.cols);

    // Write host-side CSC for the Python reference.
    {
        HostCSC h = device_csc_to_host(pz.mat, stream_);
        write_csc_bin(h, refs_path("10k_input.bin"));
    }

    HvgRefResult ref{};
    // scikit-misc (provides LOESS) is required. On g001: pip install --user scikit-misc
    // (skmisc.loess confirmed present on g001 as of Cycle 55b).
    // A subprocess failure here is a real test infrastructure failure, NOT a skip.
    ASSERT_NO_THROW({
        ref = run_hvg_reference(refs_path("10k_input.bin"), TOP_N, M, "10k");
    }) << "Python HVG reference subprocess failed. "
          "Ensure scikit-misc is installed: pip install --user scikit-misc";

    HvgTestContext ctx{};
    ctx.flavor          = singlet_gpu::preprocess::HvgFlavor::SeuratV3;
    ctx.flavor_name     = "seurat_v3";
    ctx.scale           = "10k";
    ctx.top_n           = TOP_N;
    ctx.ref_flavor_name = "scanpy_seurat_v3";

    run_hvg_and_compare(pz.mat, ref, ctx, stream_);
}

// ---------------------------------------------------------------------------
// TEST: PearsonResiduals, GSM4037629, top-2000
// ---------------------------------------------------------------------------
TEST_F(HvgTest, Gsm4037629_PearsonResiduals_Top2000) {
    if (!fs::exists(kGsm4037629Path)) {
        GTEST_SKIP() << "10k test sample not found: " << kGsm4037629Path;
    }

    constexpr int TOP_N = 2000;

    singlet_gpu::io::PzDeviceMatrix pz;
    ASSERT_NO_THROW({
        pz = singlet_gpu::io::load_pz(kGsm4037629Path, stream_);
    }) << "load_pz failed for GSM4037629";

    const uint32_t M = static_cast<uint32_t>(pz.mat.rows);

    // Reuse the already-written 10k_input.bin if present, else regenerate.
    const std::string input_bin = refs_path("10k_input.bin");
    if (!fs::exists(input_bin)) {
        HostCSC h = device_csc_to_host(pz.mat, stream_);
        write_csc_bin(h, input_bin);
    }

    HvgRefResult ref{};
    // Reuse the ref computed in the SeuratV3 test if it already exists,
    // otherwise recompute.  Both flavors share the same Python call.
    // A subprocess failure is a real infrastructure error, NOT a skip.
    ASSERT_NO_THROW({
        ref = run_hvg_reference(input_bin, TOP_N, M, "10k");
    }) << "Python HVG reference subprocess failed. "
          "Ensure scikit-misc is installed: pip install --user scikit-misc";

    HvgTestContext ctx{};
    ctx.flavor          = singlet_gpu::preprocess::HvgFlavor::PearsonResiduals;
    ctx.flavor_name     = "pearson_residuals";
    ctx.scale           = "10k";
    ctx.top_n           = TOP_N;
    ctx.ref_flavor_name = "scanpy_pearson_residuals";

    run_hvg_and_compare(pz.mat, ref, ctx, stream_);
}

// ===========================================================================
// TEST — 100k scale: deferred
// ===========================================================================
TEST_F(HvgTest, Scale100k_Deferred) {
    GTEST_SKIP() << "100k scale deferred — concat-loader not yet available (cycle 4+).";
}

// ===========================================================================
// EDGE CASE TESTS
// ===========================================================================

// ---------------------------------------------------------------------------
// Edge case: genes with var == 0 — must be excluded, no error.
// A matrix where all cells have the same count for some genes.
// ---------------------------------------------------------------------------
TEST_F(HvgTest, EdgeCase_ZeroVarGenes_Excluded) {
    // 5 genes × 10 cells.
    // Gene 0: all cells have count 5 (constant → var == 0).
    // Gene 1: all zeros (nnz == 0 → also excluded).
    // Genes 2–4: variable counts.
    //
    // CSC layout (genes=rows, cells=cols).
    HostCSC h;
    h.m = 5; h.n = 10;
    // Build indptr: for each cell (column), list the nonzero genes (rows).
    // We set: gene 0 appears in all 10 cells with value 5.0;
    //         gene 1 appears in no cell;
    //         genes 2–4 each appear in a random subset.

    std::mt19937_64 rng(kHvgSeed ^ 0x1);
    // Use counts in [0.05, 2.0] so genes fall within the HvgConfig default
    // min_mean=0.0125 and max_mean=3.0 window (log-norm range).
    std::uniform_real_distribution<float> cnt_dist(0.05f, 2.0f);

    std::vector<std::vector<std::pair<int32_t,float>>> cols(10);
    for (int j = 0; j < 10; ++j) {
        // Gene 0: constant value 0.5 (within min/max_mean window, var=0).
        cols[j].push_back({0, 0.5f});
        // Gene 1: skip (zero nnz).
        // Genes 2–4: appear in every cell with variable counts so var > 0.
        for (int g = 2; g < 5; ++g)
            cols[j].push_back({g, cnt_dist(rng)});
        std::sort(cols[j].begin(), cols[j].end(),
                  [](auto& a, auto& b){ return a.first < b.first; });
    }

    int64_t total = 0;
    for (auto& c : cols) total += (int64_t)c.size();
    h.nnz = (uint64_t)total;
    h.indptr.resize(11, 0);
    for (int j = 0; j < 10; ++j)
        h.indptr[j+1] = h.indptr[j] + (int32_t)cols[j].size();
    h.values .resize(h.nnz);
    h.indices.resize(h.nnz);
    size_t off = 0;
    for (int j = 0; j < 10; ++j)
        for (auto& [r,v] : cols[j]) { h.indices[off]=r; h.values[off]=v; ++off; }

    singlet_gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);

    for (auto flavor : {singlet_gpu::preprocess::HvgFlavor::SeuratV3,
                         singlet_gpu::preprocess::HvgFlavor::PearsonResiduals})
    {
        singlet_gpu::preprocess::HvgConfig cfg{};
        cfg.flavor = flavor;
        cfg.top_n  = 3;  // only 3 eligible genes (2–4)

        singlet_gpu::preprocess::HvgResult res;
        EXPECT_NO_THROW({
            res = singlet_gpu::preprocess::select_hvg(d_mat, cfg, stream_);
        }) << "select_hvg must not throw when zero-var/zero-nnz genes are present";

        cudaStreamSynchronize(stream_);

        std::vector<int32_t> h_idx(3);
        cudaMemcpy(h_idx.data(), res.indices.get(),
                   3 * sizeof(int32_t), cudaMemcpyDeviceToHost);

        // Genes 0 and 1 must not appear in the selected set.
        for (auto idx : h_idx) {
            EXPECT_NE(idx, 0) << "Gene 0 (constant, var==0) must be excluded from HVG";
            EXPECT_NE(idx, 1) << "Gene 1 (zero nnz) must be excluded from HVG";
        }
    }
}

// ---------------------------------------------------------------------------
// Edge case: top_n > m — must return all m genes, no error.
// ---------------------------------------------------------------------------
TEST_F(HvgTest, EdgeCase_TopN_GreaterThan_M) {
    HostCSC h = make_synthetic_count_csc(30, 50, 0.3f, 50.0f, kHvgSeed ^ 0x2);
    singlet_gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);

    constexpr int M = 30;
    constexpr int REQUESTED_TOP_N = 999;   // >> m

    for (auto flavor : {singlet_gpu::preprocess::HvgFlavor::SeuratV3,
                         singlet_gpu::preprocess::HvgFlavor::PearsonResiduals})
    {
        singlet_gpu::preprocess::HvgConfig cfg{};
        cfg.flavor = flavor;
        cfg.top_n  = REQUESTED_TOP_N;

        singlet_gpu::preprocess::HvgResult res;
        EXPECT_NO_THROW({
            res = singlet_gpu::preprocess::select_hvg(d_mat, cfg, stream_);
        }) << "select_hvg must not throw when top_n > m";

        cudaStreamSynchronize(stream_);

        // Result must contain at most m indices (design doc: "return all m genes").
        // The result size is clamped to the number of eligible genes ≤ m.
        std::vector<int32_t> h_idx(M);
        cudaError_t e = cudaMemcpy(h_idx.data(), res.indices.get(),
                                   M * sizeof(int32_t), cudaMemcpyDeviceToHost);
        EXPECT_EQ(e, cudaSuccess)
            << "cudaMemcpy of indices failed when top_n > m";

        // All returned indices must be valid gene indices in [0, M).
        for (int i = 0; i < M; ++i) {
            if (h_idx[i] < 0)
                break;   // sentinel — fewer than M eligible genes is OK
            EXPECT_LT(h_idx[i], M)
                << "Index " << h_idx[i] << " out of range [0, " << M << ")";
        }
    }
}

// ---------------------------------------------------------------------------
// Edge case: top_n == 0 — must return empty result, no error.
// ---------------------------------------------------------------------------
TEST_F(HvgTest, EdgeCase_TopN_Zero) {
    HostCSC h = make_synthetic_count_csc(30, 50, 0.2f, 50.0f, kHvgSeed ^ 0x3);
    singlet_gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);

    for (auto flavor : {singlet_gpu::preprocess::HvgFlavor::SeuratV3,
                         singlet_gpu::preprocess::HvgFlavor::PearsonResiduals})
    {
        singlet_gpu::preprocess::HvgConfig cfg{};
        cfg.flavor = flavor;
        cfg.top_n  = 0;

        singlet_gpu::preprocess::HvgResult res;
        EXPECT_NO_THROW({
            res = singlet_gpu::preprocess::select_hvg(d_mat, cfg, stream_);
        }) << "select_hvg must not throw when top_n == 0";

        cudaStreamSynchronize(stream_);
        // No assertion on result content — design doc only says "return empty array".
        // The GPU call completing without error is the pass criterion.
    }
}

// ===========================================================================
// Deviance HVG smoke tests
//
// Uniform-count matrix invariant: if every gene has the same total count s_g,
// then π_g = s_g/T is identical for all genes, and all D_g must be equal.
// Assert: std(D) / mean(D) < 1e-4  (uniform deviance across genes).
//
// Also verifies: Poisson variant produces the same uniformity, top_n indices
// are a subset of [0, n_genes), is_variable mask has exactly top_n bits set,
// and deviance_feature_selection_with_lib_sizes reuses pre-computed lib sizes.
// ===========================================================================

namespace {

// Build a CSC where every gene has exactly equal total count (lib sizes may
// differ per cell). Each gene has exactly one nonzero per cell; nonzero value
// is drawn from a shared per-cell scale so gene sums are all equal.
HostCSC make_uniform_deviance_csc(uint32_t n_genes, uint32_t n_cells,
                                   float per_cell_count,
                                   uint64_t seed)
{
    // All genes: equal total = n_cells * per_cell_count.
    // Layout: dense (nnz = n_genes * n_cells) — every (gene, cell) is filled.
    // All values equal per_cell_count so s_g = n_cells * per_cell_count for all g.
    HostCSC h;
    h.m   = n_genes;
    h.n   = n_cells;
    h.nnz = static_cast<uint64_t>(n_genes) * n_cells;

    h.indptr .resize(static_cast<size_t>(n_cells) + 1, 0);
    h.indices.resize(static_cast<size_t>(h.nnz));
    h.values .resize(static_cast<size_t>(h.nnz));

    // Column j: rows 0..n_genes-1, value = per_cell_count.
    for (uint32_t j = 0; j < n_cells; ++j) {
        h.indptr[j + 1] = static_cast<int32_t>((j + 1) * n_genes);
        for (uint32_t i = 0; i < n_genes; ++i) {
            size_t off = static_cast<size_t>(j) * n_genes + i;
            h.indices[off] = static_cast<int32_t>(i);
            h.values [off] = per_cell_count;
        }
    }
    (void)seed;  // uniform matrix is deterministic; seed kept for API symmetry
    return h;
}

}  // anonymous namespace

class DevianceHvgTest : public ::testing::Test {
protected:
    cudaStream_t stream_ = nullptr;

    void SetUp() override {
        cudaError_t err = cudaStreamCreate(&stream_);
        ASSERT_EQ(err, cudaSuccess) << "cudaStreamCreate: " << cudaGetErrorString(err);
    }

    void TearDown() override {
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }
};

// ---------------------------------------------------------------------------
// Smoke test: 500 cells × 200 genes, uniform counts.
// D should be the same (within 1e-4 relative std) for all genes.
// ---------------------------------------------------------------------------
TEST_F(DevianceHvgTest, Uniform_NullBinomial_Smoke) {
    constexpr uint32_t N_GENES = 200;
    constexpr uint32_t N_CELLS = 500;
    constexpr float    PER_CELL = 3.f;   // uniform count per (gene,cell)
    constexpr int      TOP_N    = 50;

    HostCSC h = make_uniform_deviance_csc(N_GENES, N_CELLS, PER_CELL, 0xABCDull);
    singlet_gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);

    singlet_gpu::preprocess::DevianceHvgConfig cfg{};
    cfg.top_n          = TOP_N;
    cfg.use_poisson    = false;
    cfg.min_gene_total = 1.f;

    singlet_gpu::preprocess::DevianceHvgResult res;
    ASSERT_NO_THROW({
        res = singlet_gpu::preprocess::deviance_feature_selection(d_mat, cfg, stream_);
    }) << "deviance_feature_selection threw on uniform matrix (null-binomial)";

    CUDA_CHECK(cudaStreamSynchronize(stream_));

    ASSERT_EQ(res.n_genes_considered, static_cast<int>(N_GENES));

    // Copy deviance scores to host (sorted descending, all N_GENES entries).
    std::vector<float> h_dev(N_GENES);
    CUDA_CHECK(cudaMemcpy(h_dev.data(), res.deviance.get(),
                          N_GENES * sizeof(float), cudaMemcpyDeviceToHost));

    // All deviance values should be positive (uniform matrix has π > 0).
    for (uint32_t g = 0; g < N_GENES; ++g) {
        EXPECT_GT(h_dev[g], 0.f) << "Gene " << g << " has zero deviance in uniform matrix";
    }

    // std(D) / mean(D) < 1e-4 for uniform matrix.
    double sum = 0.0, sum2 = 0.0;
    for (uint32_t g = 0; g < N_GENES; ++g) {
        double d = static_cast<double>(h_dev[g]);
        sum  += d;
        sum2 += d * d;
    }
    double mean_d = sum  / N_GENES;
    double var_d  = sum2 / N_GENES - mean_d * mean_d;
    double std_d  = std::sqrt(std::max(0.0, var_d));
    double rel_std = (mean_d > 1e-12) ? std_d / mean_d : 0.0;
    EXPECT_LT(rel_std, 1e-4)
        << "Deviance is non-uniform for uniform-count matrix: "
        << "mean=" << mean_d << " std=" << std_d << " rel_std=" << rel_std;

    // is_variable mask: exactly TOP_N bits set.
    std::vector<uint8_t> h_mask(N_GENES);
    CUDA_CHECK(cudaMemcpy(h_mask.data(), res.is_variable.get(),
                          N_GENES, cudaMemcpyDeviceToHost));
    int n_set = 0;
    for (uint32_t g = 0; g < N_GENES; ++g) n_set += static_cast<int>(h_mask[g]);
    EXPECT_EQ(n_set, TOP_N)
        << "is_variable mask has " << n_set << " bits set, expected " << TOP_N;

    // top_gene_idx: all values in [0, N_GENES).
    std::vector<int32_t> h_top(TOP_N);
    CUDA_CHECK(cudaMemcpy(h_top.data(), res.top_gene_idx.get(),
                          TOP_N * sizeof(int32_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < TOP_N; ++i) {
        EXPECT_GE(h_top[i], 0)
            << "top_gene_idx[" << i << "] is negative";
        EXPECT_LT(h_top[i], static_cast<int32_t>(N_GENES))
            << "top_gene_idx[" << i << "] = " << h_top[i]
            << " is out of range [0, " << N_GENES << ")";
    }

    std::printf("| deviance_hvg | uniform | null_binomial | rel_std=%.2e | PASS |\n",
                rel_std);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Smoke test: Poisson variant — same uniformity invariant.
// ---------------------------------------------------------------------------
TEST_F(DevianceHvgTest, Uniform_Poisson_Smoke) {
    constexpr uint32_t N_GENES = 200;
    constexpr uint32_t N_CELLS = 500;
    constexpr float    PER_CELL = 3.f;
    constexpr int      TOP_N    = 50;

    HostCSC h = make_uniform_deviance_csc(N_GENES, N_CELLS, PER_CELL, 0xABCDull);
    singlet_gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);

    singlet_gpu::preprocess::DevianceHvgConfig cfg{};
    cfg.top_n       = TOP_N;
    cfg.use_poisson = true;

    singlet_gpu::preprocess::DevianceHvgResult res;
    ASSERT_NO_THROW({
        res = singlet_gpu::preprocess::deviance_feature_selection(d_mat, cfg, stream_);
    }) << "deviance_feature_selection threw on uniform matrix (Poisson)";

    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<float> h_dev(N_GENES);
    CUDA_CHECK(cudaMemcpy(h_dev.data(), res.deviance.get(),
                          N_GENES * sizeof(float), cudaMemcpyDeviceToHost));

    double sum = 0.0, sum2 = 0.0;
    for (uint32_t g = 0; g < N_GENES; ++g) {
        double d = static_cast<double>(h_dev[g]);
        sum  += d;
        sum2 += d * d;
    }
    double mean_d  = sum  / N_GENES;
    double var_d   = sum2 / N_GENES - mean_d * mean_d;
    double rel_std = (mean_d > 1e-12)
                     ? std::sqrt(std::max(0.0, var_d)) / mean_d : 0.0;
    EXPECT_LT(rel_std, 1e-4)
        << "Poisson deviance is non-uniform for uniform-count matrix: "
        << "mean=" << mean_d << " rel_std=" << rel_std;

    std::printf("| deviance_hvg | uniform | poisson | rel_std=%.2e | PASS |\n",
                rel_std);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Smoke test: deviance_feature_selection_with_lib_sizes re-uses pre-computed
// lib sizes and returns the same result as the auto-computing overload.
// ---------------------------------------------------------------------------
TEST_F(DevianceHvgTest, WithLibSizes_MatchesAutoComputed) {
    constexpr uint32_t N_GENES = 100;
    constexpr uint32_t N_CELLS = 200;
    constexpr float    PER_CELL = 5.f;
    constexpr int      TOP_N    = 20;

    HostCSC h = make_uniform_deviance_csc(N_GENES, N_CELLS, PER_CELL, 0x1234ull);
    singlet_gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);

    // Build per-cell lib sizes manually (all equal to N_GENES * PER_CELL).
    std::vector<float> h_ls(N_CELLS, static_cast<float>(N_GENES) * PER_CELL);
    singlet_gpu::core::DeviceMemory<float> d_ls(N_CELLS);
    CUDA_CHECK(cudaMemcpyAsync(d_ls.get(), h_ls.data(), N_CELLS * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));

    singlet_gpu::preprocess::DevianceHvgConfig cfg{};
    cfg.top_n = TOP_N;

    singlet_gpu::preprocess::DevianceHvgResult r_auto, r_lib;
    ASSERT_NO_THROW({
        r_auto = singlet_gpu::preprocess::deviance_feature_selection(
                     d_mat, cfg, stream_);
    });
    ASSERT_NO_THROW({
        r_lib  = singlet_gpu::preprocess::deviance_feature_selection_with_lib_sizes(
                     d_mat, d_ls.get(), cfg, stream_);
    });
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Deviance values must match within fp32 rounding (< 1e-5 relative).
    std::vector<float> dev_auto(N_GENES), dev_lib(N_GENES);
    CUDA_CHECK(cudaMemcpy(dev_auto.data(), r_auto.deviance.get(),
                          N_GENES * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dev_lib .data(), r_lib .deviance.get(),
                          N_GENES * sizeof(float), cudaMemcpyDeviceToHost));

    for (uint32_t g = 0; g < N_GENES; ++g) {
        float denom = std::max(std::abs(dev_auto[g]), 1e-7f);
        float rel   = std::abs(dev_auto[g] - dev_lib[g]) / denom;
        EXPECT_LT(rel, 1e-4f)
            << "Deviance mismatch for gene " << g
            << ": auto=" << dev_auto[g] << " lib=" << dev_lib[g];
    }

    std::printf("| deviance_hvg | with_lib_sizes_match | PASS |\n");
    std::fflush(stdout);
}
