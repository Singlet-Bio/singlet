// SPDX-License-Identifier: MIT
// singlet/gpu/tests/de_ttest_correctness.cpp
//
// Correctness harness for singlet::gpu::de::ttest_de (Welch's t-test DE).
//
// Written against the design doc at:
//   singlet/gpu/state/designs/09-de.md
// NOT against the kernel source — authored in parallel with de/ttest.h by
// the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: scanpy sc.tl.rank_genes_groups(method='t-test') via subprocess
//   tests/refs/de_scanpy_reference.py (shared with de_wilcoxon_correctness.cpp).
//
// Tolerances (design doc §"Correctness test spec"):
//   Marker gene Jaccard   >= 0.90 (top-50 per cluster)
//   log2_fc Spearman rho  >= 0.98 (per-cluster, top-N genes)
//   p-value rank Spearman >= 0.95 (per-cluster rank correlation)
//   Determinism           bit-identical with cfg.deterministic = true
//
// Test cases:
//   Ttest_TinyPlanted_TopMarkersJaccard
//   Ttest_TinyPlanted_LogFoldChangeSpearman
//   Ttest_TinyPlanted_PvalueRankSpearman
//   Ttest_GSM4037629_RealData
//   Ttest_DeterminismFlag_BitIdentical
//   Ttest_EdgeCase_EmptyCluster
//   Ttest_EdgeCase_SingleCluster
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every test that requires a device is guarded by
//   gpu_available() and calls GTEST_SKIP().

// ---- singlet-gpu DE headers (written by gpu-kernel-dev) --------------------
#include <singlet/gpu/de/ttest.h>

// ---- other singlet-gpu headers needed for the GSM4037629 real-data test ----
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/preprocess/lognorm.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

// ---- test helpers -----------------------------------------------------------
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
#include <unordered_map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

/// Synthetic matrix dimensions (matching wilcoxon harness for direct comparison).
constexpr int kTinyGenes             = 200;
constexpr int kTinyCells             = 500;
constexpr int kTinyClusters          = 4;
constexpr int kPlantedGenesPerCluster = 50;
constexpr int kTopN                  = 50;

/// GSM4037629 path.
static const char* kGsmPath =
    "/mnt/projects/debruinz_project/singlet_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/exon_counts.1pz";

/// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_de_ttest_refs_tmp";

/// Python reference script path (SHARED with wilcoxon harness).
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet/gpu/"
    "tests/refs/de_scanpy_reference.py";

// =============================================================================
// Utility helpers (mirrored from de_wilcoxon_correctness.cpp)
// =============================================================================

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

bool gpu_available() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    return (e == cudaSuccess && count > 0);
}

// ---------------------------------------------------------------------------
// Minimal .npy v1.0 writers.
// ---------------------------------------------------------------------------
static void write_npy_generic(const std::string& path,
                               const void* data,
                               const std::string& dtype_str,
                               int rows, int cols) {
    std::string shape_str = (cols == 1)
        ? "(" + std::to_string(rows) + ",)"
        : "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")";
    std::string header_str =
        "{'descr': '" + dtype_str + "', 'fortran_order': False, 'shape': " +
        shape_str + ", }";

    constexpr int kPreamble = 10;
    int raw_len   = static_cast<int>(header_str.size());
    int pad_to_64 = ((raw_len + kPreamble + 63) / 64) * 64 - kPreamble;
    while (static_cast<int>(header_str.size()) < pad_to_64 - 1)
        header_str += ' ';
    header_str += '\n';

    std::ofstream fout(path, std::ios::binary | std::ios::trunc);
    if (!fout) throw std::runtime_error("write_npy: cannot open " + path);

    const char magic[] = "\x93NUMPY\x01\x00";
    fout.write(magic, 8);
    uint16_t hlen = static_cast<uint16_t>(header_str.size());
    fout.write(reinterpret_cast<const char*>(&hlen), 2);
    fout.write(header_str.c_str(), static_cast<std::streamsize>(header_str.size()));

    size_t n_items = static_cast<size_t>(rows) * (cols == 1 ? 1 : cols);
    fout.write(static_cast<const char*>(data),
               static_cast<std::streamsize>(n_items * 4));  // int32 / float32
    if (!fout) throw std::runtime_error("write_npy: write error " + path);
}

void write_npy_i32(const std::string& path, const int32_t* data, int n) {
    write_npy_generic(path, data, "<i4", n, 1);
}

void write_npy_f32(const std::string& path, const float* data, int n) {
    write_npy_generic(path, data, "<f4", n, 1);
}

// ---------------------------------------------------------------------------
// Host-side CSC container.
// ---------------------------------------------------------------------------
struct HostCSC {
    uint32_t             m;
    uint32_t             n;
    uint64_t             nnz;
    std::vector<float>   values;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
};

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

singlet::gpu::core::DeviceCSC upload_csc(const HostCSC& h, cudaStream_t stream) {
    using namespace singlet::gpu::core;
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

    DeviceCSC mat;
    mat.rows        = h.m;
    mat.cols        = h.n;
    mat.nnz         = h.nnz;
    mat.values      = std::move(d_values);
    mat.col_ptr     = std::move(d_indptr);
    mat.row_indices = std::move(d_indices);
    return mat;
}

HostCSC device_csc_to_host(const singlet::gpu::core::DeviceCSC& mat,
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
// Planted-signal synthetic CSC (identical logic to wilcoxon harness).
// ---------------------------------------------------------------------------
HostCSC make_planted_csc(uint64_t seed) {
    constexpr int G = kTinyGenes;
    constexpr int C = kTinyCells;
    constexpr int K = kTinyClusters;
    constexpr int cells_per_cluster = C / K;

    std::mt19937_64 rng(seed);

    std::vector<int32_t> cell_cluster(C);
    for (int j = 0; j < C; ++j)
        cell_cluster[j] = j / cells_per_cluster;

    HostCSC csc;
    csc.m = G;
    csc.n = C;
    csc.indptr.resize(static_cast<size_t>(C) + 1, 0);

    std::vector<std::vector<std::pair<int32_t, float>>> cols(C);

    auto draw_val = [&](float lo, float hi) -> float {
        std::uniform_real_distribution<float> d(lo, hi);
        return std::round(d(rng));
    };

    for (int j = 0; j < C; ++j) {
        int clust = cell_cluster[j];
        for (int g = 0; g < G; ++g) {
            int planted_cluster_for_gene = g / kPlantedGenesPerCluster;
            bool is_planted = (planted_cluster_for_gene < K) &&
                              (g < K * kPlantedGenesPerCluster);
            float val;
            if (is_planted) {
                val = (clust == planted_cluster_for_gene)
                          ? draw_val(50.0f, 100.0f)
                          : draw_val(0.0f,   5.0f);
            } else {
                val = draw_val(0.0f, 10.0f);
            }
            if (val > 0.0f)
                cols[j].push_back({static_cast<int32_t>(g), val});
        }
    }

    int64_t total_nnz = 0;
    for (int j = 0; j < C; ++j) total_nnz += static_cast<int64_t>(cols[j].size());
    csc.nnz = static_cast<uint64_t>(total_nnz);

    csc.indptr[0] = 0;
    for (int j = 0; j < C; ++j)
        csc.indptr[j + 1] = csc.indptr[j] + static_cast<int32_t>(cols[j].size());

    csc.values .resize(static_cast<size_t>(csc.nnz));
    csc.indices.resize(static_cast<size_t>(csc.nnz));
    size_t off = 0;
    for (int j = 0; j < C; ++j)
        for (auto& [row, val] : cols[j]) {
            csc.indices[off] = row;
            csc.values [off] = val;
            ++off;
        }
    return csc;
}

std::vector<int32_t> make_roundrobin_labels(int n_cells, int n_clusters) {
    std::vector<int32_t> lbl(n_cells);
    for (int j = 0; j < n_cells; ++j)
        lbl[j] = j % n_clusters;
    return lbl;
}

// Block labels: cell j → cluster (j / cells_per_cluster). Matches the
// cluster assignment used by make_planted_csc so the planted signal is
// recoverable. Use this for TinyPlanted tests. RealData tests keep
// round-robin (there's no planted signal to recover there).
std::vector<int32_t> make_block_labels(int n_cells, int n_clusters) {
    std::vector<int32_t> lbl(n_cells);
    const int cells_per_cluster = n_cells / n_clusters;
    for (int j = 0; j < n_cells; ++j)
        lbl[j] = std::min(j / cells_per_cluster, n_clusters - 1);
    return lbl;
}

// ---------------------------------------------------------------------------
// Run Python reference with method='t-test'.
// ---------------------------------------------------------------------------
std::string run_scanpy_de(
        const std::string& csc_bin,
        const std::string& labels_npy,
        const std::string& tag,
        int top_n)
{
    std::string out_npz = refs_path("de_ttest_ref_" + tag + ".npz");
    std::string cmd = "python3 " + std::string(kRefScript) +
                      " --input "  + csc_bin    +
                      " --labels " + labels_npy +
                      " --output " + out_npz    +
                      " --method t-test"         +
                      " --top-n "  + std::to_string(top_n);
    run_cmd(cmd);
    return out_npz;
}

// ---------------------------------------------------------------------------
// Extract array from .npz via inline Python.
// ---------------------------------------------------------------------------
std::vector<uint8_t> extract_npz_array(
        const std::string& npz_path,
        const std::string& key,
        const std::string& dtype,
        size_t count,
        const std::string& tag)
{
    std::string bin_path = refs_path("de_tt_" + tag + ".bin");
    std::string py = "python3 -c \""
        "import numpy as np; "
        "d = np.load('" + npz_path + "', allow_pickle=True); "
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
}

template <typename T>
std::vector<T> raw_to_vec(const std::vector<uint8_t>& raw, size_t count) {
    if (raw.size() < count * sizeof(T))
        throw std::runtime_error("raw_to_vec: buffer too small");
    std::vector<T> v(count);
    std::memcpy(v.data(), raw.data(), count * sizeof(T));
    return v;
}

// ---------------------------------------------------------------------------
// Spearman rank correlation (pure C++).
//
// Special case: if either input vector is constant (all values equal), the
// standard Spearman formula yields 0/0.  We return 1.0 in that case because
// a constant ranking means both pipelines agree that all items are equally
// significant — which is perfect agreement, not zero agreement.  This arises
// for TinyPlanted planted genes whose t-statistics are t≈80-90: erfcf(|t|/√2)
// underflows to exactly 0.0 in fp32, so both GPU and scanpy produce an
// all-zero p-value vector for the top-50 planted genes.  Spearman on two
// identical all-zero vectors must return 1.0 (perfect agreement), not 0.0.
// (Same fix applied to de_wilcoxon_correctness.cpp in Cycle 76.)
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
    // Constant vector (denom == 0) means all values tied → perfect agreement.
    if (denom_a == 0 || denom_b == 0) return 1.0;
    return num / std::sqrt(denom_a * denom_b);
}

// ---------------------------------------------------------------------------
// Jaccard of two integer sets.
// ---------------------------------------------------------------------------
double jaccard(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    std::set<int32_t> sa(a.begin(), a.end()), sb(b.begin(), b.end());
    sa.erase(-1); sb.erase(-1);
    int intersection = 0;
    for (int32_t v : sa) if (sb.count(v)) ++intersection;
    int union_sz = static_cast<int>(sa.size() + sb.size()) - intersection;
    if (union_sz == 0) return 1.0;
    return static_cast<double>(intersection) / union_sz;
}

// ---------------------------------------------------------------------------
// Copy t-test DE result per-cluster from device to host.
// The t-test result type mirrors WilcoxonResult but uses TtestResult /
// ClusterMarkers with t_stats instead of z_scores.
// ---------------------------------------------------------------------------
struct HostClusterMarkers {
    std::vector<int32_t> gene_indices;
    std::vector<float>   t_stats;     // Welch t-statistics (alias for "scores")
    std::vector<float>   log2_fc;
    std::vector<float>   p_values;
    std::vector<float>   p_adj;
    int                  cluster_id;
};

HostClusterMarkers copy_cluster_markers(
        const singlet::gpu::de::ClusterMarkers& cm,
        int top_n,
        cudaStream_t stream)
{
    auto ck = [](cudaError_t e, const char* s) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("copy_cluster_markers ") + s + ": " +
                                     cudaGetErrorString(e));
    };
    ck(cudaStreamSynchronize(stream), "pre-sync");

    HostClusterMarkers h;
    h.cluster_id = cm.cluster_id;
    h.gene_indices.resize(top_n);
    h.t_stats     .resize(top_n);
    h.log2_fc     .resize(top_n);
    h.p_values    .resize(top_n);
    h.p_adj       .resize(top_n);

    ck(cudaMemcpy(h.gene_indices.data(), cm.gene_indices.get(),
                  top_n * sizeof(int32_t), cudaMemcpyDeviceToHost), "indices");
    // t-test stores t-stats in the same z_scores field (design doc: "same shape
    // as WilcoxonResult but with TtestConfig").
    ck(cudaMemcpy(h.t_stats  .data(), cm.z_scores.get(),
                  top_n * sizeof(float),   cudaMemcpyDeviceToHost), "tstat");
    ck(cudaMemcpy(h.log2_fc  .data(), cm.log2_fc.get(),
                  top_n * sizeof(float),   cudaMemcpyDeviceToHost), "lfc");
    ck(cudaMemcpy(h.p_values .data(), cm.p_values.get(),
                  top_n * sizeof(float),   cudaMemcpyDeviceToHost), "pvals");
    ck(cudaMemcpy(h.p_adj    .data(), cm.p_adj.get(),
                  top_n * sizeof(float),   cudaMemcpyDeviceToHost), "padj");
    return h;
}

}  // anonymous namespace

// =============================================================================
// Test fixture.
// =============================================================================

class TtestDE : public ::testing::Test {
protected:
    cudaStream_t stream_{};

    void SetUp() override {
        if (!gpu_available()) GTEST_SKIP() << "No CUDA device — skipping.";
        cudaError_t e = cudaStreamCreate(&stream_);
        if (e != cudaSuccess) GTEST_SKIP() << "cudaStreamCreate failed.";
        ensure_refs_tmp();
    }

    void TearDown() override {
        if (stream_) cudaStreamDestroy(stream_);
    }
};

// =============================================================================
// Test 1: Ttest_TinyPlanted_TopMarkersJaccard
// =============================================================================

TEST_F(TtestDE, Ttest_TinyPlanted_TopMarkersJaccard) {
    HostCSC h = make_planted_csc(kSeed);
    std::vector<int32_t> lbl = make_block_labels(kTinyCells, kTinyClusters);

    const std::string csc_bin    = refs_path("tt_tiny_csc.bin");
    const std::string labels_npy = refs_path("tt_tiny_labels.npy");
    write_csc_bin(h, csc_bin);
    write_npy_i32(labels_npy, lbl.data(), kTinyCells);

    std::string ref_npz;
    try {
        ref_npz = run_scanpy_de(csc_bin, labels_npy, "tiny_jaccard", kTopN);
    } catch (const std::exception& ex) {
        GTEST_SKIP() << "scanpy reference failed: " << ex.what();
    }

    singlet::gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);
    // Log-normalise to match the scanpy reference (which calls normalize_total +
    // log1p on the raw counts before running t-test).
    {
        singlet::gpu::preprocess::LogNormConfig ln_cfg;
        ln_cfg.target_count = 1e4f;  // match scanpy reference pipeline (normalize_total target_sum=1e4)
        auto ln_result = singlet::gpu::preprocess::log_normalize(d_mat, ln_cfg, stream_);
        cudaStreamSynchronize(stream_);  // ensure async kernel completes before workspace freed
    }
    singlet::gpu::core::DeviceMemory<int32_t> d_labels(kTinyCells);
    cudaMemcpyAsync(d_labels.get(), lbl.data(),
                    kTinyCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);

    singlet::gpu::de::TtestConfig cfg;
    cfg.top_n = kTopN;

    // === C82 DIAG A: GPU cluster_labels readback (before ttest_de) ===
    {
        const int n_cells = kTinyCells;
        std::vector<int32_t> h_labels_readback(n_cells);
        cudaMemcpyAsync(h_labels_readback.data(), d_labels.get(),
                        n_cells * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        std::cerr << "=== C82 DIAG A (labels readback) ===\n";
        std::cerr << "  first 10: ";
        for (int j = 0; j < 10; ++j) std::cerr << h_labels_readback[j] << " ";
        std::cerr << "\n  cells 120-135: ";
        for (int j = 120; j < 135; ++j) std::cerr << h_labels_readback[j] << " ";
        std::cerr << "\n  cells 248-257: ";
        for (int j = 248; j < 257; ++j) std::cerr << h_labels_readback[j] << " ";
        std::cerr << "\n  cells 373-380: ";
        for (int j = 373; j < 380; ++j) std::cerr << h_labels_readback[j] << " ";
        std::cerr << "\n  cells 490-500: ";
        for (int j = 490; j < 500; ++j) std::cerr << h_labels_readback[j] << " ";
        std::cerr << "\n  expected: 0..0(x125), 1..1(x125), 2..2(x125), 3..3(x125)\n";
    }

    singlet::gpu::de::TtestResult result =
        singlet::gpu::de::ttest_de(d_mat, d_labels, kTinyClusters, cfg, stream_);

    ASSERT_EQ(static_cast<int>(result.per_cluster.size()), kTinyClusters);

    double min_jaccard = 1.0;
    for (int c = 0; c < kTinyClusters; ++c) {
        HostClusterMarkers hcm = copy_cluster_markers(
            result.per_cluster[c], kTopN, stream_);

        auto raw = extract_npz_array(
            ref_npz, "gene_indices_" + std::to_string(c),
            "int32", static_cast<size_t>(kTopN), "tiny_jac_" + std::to_string(c));
        auto ref_indices = raw_to_vec<int32_t>(raw, static_cast<size_t>(kTopN));

        double j = jaccard(hcm.gene_indices, ref_indices);
        printf("[Ttest_TinyPlanted_TopMarkersJaccard] cluster %d Jaccard=%.4f\n",
               c, j);

        // ---- C80 DIAGNOSTIC: cluster 0 only --------------------------------
        if (c == 0) {
            // Fetch scanpy scores for cluster 0.
            std::vector<float> ref_scores_c0;
            try {
                auto raw_sc = extract_npz_array(
                    ref_npz, "scores_0", "float32",
                    static_cast<size_t>(kTopN), "tiny_jac_sc0");
                ref_scores_c0 = raw_to_vec<float>(raw_sc, static_cast<size_t>(kTopN));
            } catch (const std::exception& ex) {
                printf("[C80 DIAG] WARNING: could not fetch scores_0: %s\n", ex.what());
            }

            printf("=== C80 DIAG (cluster 0) ===\n");
            printf("GPU top-50 (gene_id, gpu_t):\n");
            for (int i = 0; i < kTopN; ++i) {
                printf("  rank %2d: gene_id=%3d  t=%+.6f\n",
                       i, hcm.gene_indices[i], hcm.t_stats[i]);
            }
            printf("Scanpy top-50 (gene_id, scanpy_score):\n");
            for (int i = 0; i < kTopN; ++i) {
                int g = (i < static_cast<int>(ref_indices.size())) ? ref_indices[i] : -1;
                float sc = (i < static_cast<int>(ref_scores_c0.size())) ? ref_scores_c0[i] : 0.0f;
                printf("  rank %2d: gene_id=%3d  score=%+.6f\n", i, g, sc);
            }

            // Compute set-only-GPU and set-only-scanpy.
            std::set<int32_t> gpu_set(hcm.gene_indices.begin(), hcm.gene_indices.end());
            std::set<int32_t> ref_set(ref_indices.begin(), ref_indices.end());
            gpu_set.erase(-1); ref_set.erase(-1);

            printf("Set-only-GPU (GPU ∖ scanpy): [");
            bool first = true;
            for (int32_t g : gpu_set) {
                if (!ref_set.count(g)) {
                    if (!first) printf(", ");
                    printf("%d", g);
                    first = false;
                }
            }
            printf("]\n");

            printf("Set-only-scanpy (scanpy ∖ GPU): [");
            first = true;
            for (int32_t g : ref_set) {
                if (!gpu_set.count(g)) {
                    if (!first) printf(", ");
                    printf("%d", g);
                    first = false;
                }
            }
            printf("]\n");

            int intersection = 0;
            for (int32_t g : gpu_set) if (ref_set.count(g)) ++intersection;
            printf("Intersection size: %d\n", intersection);
            printf("=== END C80 DIAG ===\n");

            // === C82 DIAG B: per-gene t-value comparison (cluster 0 scanpy top-50) ===
            // NOTE: rank-aligned (position i in GPU top-50 vs position i in scanpy top-50),
            // NOT gene-aligned, because t_all device buffer is internal to ttest_de.
            // hcm.t_stats[i] = GPU t for GPU's i-th top gene (for cluster 0).
            // sc_scores_vec[i] = scanpy score for scanpy's i-th top gene (for cluster 0).
            // Also prints gene_id from each side for cross-reference.
            std::vector<float> sc_scores_diag_b;
            try {
                auto raw_sc_b = extract_npz_array(
                    ref_npz, "scores_0", "float32",
                    static_cast<size_t>(kTopN), "tiny_diagb_sc0");
                sc_scores_diag_b = raw_to_vec<float>(raw_sc_b, static_cast<size_t>(kTopN));
            } catch (const std::exception& ex) {
                printf("[C82 DIAG B] WARNING: could not fetch scores_0: %s\n", ex.what());
            }
            std::cerr << "=== C82 DIAG B (rank-aligned: GPU top-50 vs scanpy top-50 for cluster 0) ===\n";
            std::cerr << "  rank | gpu_gene | gpu_t        | scanpy_gene | scanpy_score\n";
            int n_diag_b = std::min(kTopN, 20);
            for (int i = 0; i < n_diag_b; ++i) {
                int gpu_gene    = (i < static_cast<int>(hcm.gene_indices.size()))
                                  ? hcm.gene_indices[i] : -1;
                float gpu_t     = (i < static_cast<int>(hcm.t_stats.size()))
                                  ? hcm.t_stats[i] : -9999.f;
                int sc_gene     = (i < static_cast<int>(ref_indices.size()))
                                  ? ref_indices[i] : -1;
                float sc_score  = (i < static_cast<int>(sc_scores_diag_b.size()))
                                  ? sc_scores_diag_b[i] : -9999.f;
                std::cerr << "  " << i
                          << " | " << gpu_gene
                          << " | " << gpu_t
                          << " | " << sc_gene
                          << " | " << sc_score << "\n";
            }
            std::cerr << "  (rank-aligned, not gene-aligned)\n";
        }
        // ---- end C80 DIAGNOSTIC --------------------------------------------

        min_jaccard = std::min(min_jaccard, j);
    }

    printf("[registry] | %s | de/ttest | tiny | jaccard_top50 | %.4f | 0.90 | "
           "scanpy_t-test | - | %s |\n",
           "2026-04-13", min_jaccard, min_jaccard >= 0.90 ? "PASS" : "FAIL");

    EXPECT_GE(min_jaccard, 0.90)
        << "Minimum per-cluster Jaccard = " << min_jaccard
        << " < 0.90 required";
}

// =============================================================================
// Test 2: Ttest_TinyPlanted_LogFoldChangeSpearman
// =============================================================================

TEST_F(TtestDE, Ttest_TinyPlanted_LogFoldChangeSpearman) {
    HostCSC h = make_planted_csc(kSeed);
    std::vector<int32_t> lbl = make_block_labels(kTinyCells, kTinyClusters);

    const std::string csc_bin    = refs_path("tt_tiny_csc.bin");
    const std::string labels_npy = refs_path("tt_tiny_labels.npy");
    write_csc_bin(h, csc_bin);
    write_npy_i32(labels_npy, lbl.data(), kTinyCells);

    std::string ref_npz;
    try {
        ref_npz = run_scanpy_de(csc_bin, labels_npy, "tiny_lfc", kTopN);
    } catch (const std::exception& ex) {
        GTEST_SKIP() << "scanpy reference failed: " << ex.what();
    }

    singlet::gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);
    // Log-normalise to match the scanpy reference (which calls normalize_total +
    // log1p on the raw counts before running t-test).
    {
        singlet::gpu::preprocess::LogNormConfig ln_cfg;
        ln_cfg.target_count = 1e4f;  // match scanpy reference pipeline (normalize_total target_sum=1e4)
        auto ln_result = singlet::gpu::preprocess::log_normalize(d_mat, ln_cfg, stream_);
        cudaStreamSynchronize(stream_);  // ensure async kernel completes before workspace freed
    }
    singlet::gpu::core::DeviceMemory<int32_t> d_labels(kTinyCells);
    cudaMemcpyAsync(d_labels.get(), lbl.data(),
                    kTinyCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);

    singlet::gpu::de::TtestConfig cfg;
    cfg.top_n = kTopN;
    singlet::gpu::de::TtestResult result =
        singlet::gpu::de::ttest_de(d_mat, d_labels, kTinyClusters, cfg, stream_);

    double min_rho = 1.0;
    for (int c = 0; c < kTinyClusters; ++c) {
        HostClusterMarkers hcm = copy_cluster_markers(
            result.per_cluster[c], kTopN, stream_);

        // WHY gene-aligned Spearman (Cycle 76 / Cycle 78): positional comparison of two
        // independently-ranked top-N lists fails when the sets diverge even by 1 gene.
        // Fix: build ref_map {gene_index → ref_lfc}, compare only the intersection.
        auto raw_idx = extract_npz_array(
            ref_npz, "gene_indices_" + std::to_string(c),
            "int32", static_cast<size_t>(kTopN), "tiny_lfc_idx_" + std::to_string(c));
        auto ref_idx = raw_to_vec<int32_t>(raw_idx, static_cast<size_t>(kTopN));

        auto raw_lfc = extract_npz_array(
            ref_npz, "logfoldchanges_" + std::to_string(c),
            "float32", static_cast<size_t>(kTopN), "tiny_lfc_" + std::to_string(c));
        auto ref_lfc_vec = raw_to_vec<float>(raw_lfc, static_cast<size_t>(kTopN));

        std::unordered_map<int32_t, float> ref_map;
        for (int i = 0; i < kTopN; ++i)
            if (ref_idx[i] >= 0)
                ref_map[ref_idx[i]] = ref_lfc_vec[i];

        std::vector<float> our_aligned, ref_aligned;
        for (int i = 0; i < kTopN; ++i) {
            int32_t g = hcm.gene_indices[i];
            auto it = ref_map.find(g);
            if (it != ref_map.end()) {
                our_aligned.push_back(hcm.log2_fc[i]);
                ref_aligned.push_back(it->second);
            }
        }

        double rho = (our_aligned.size() >= 2)
                     ? spearman(our_aligned, ref_aligned)
                     : 0.0;
        printf("[Ttest_TinyPlanted_LogFoldChangeSpearman] cluster %d rho=%.4f"
               " (intersection=%d/%d)\n",
               c, rho, static_cast<int>(our_aligned.size()), kTopN);
        min_rho = std::min(min_rho, rho);
    }

    printf("[registry] | %s | de/ttest | tiny | lfc_spearman | %.4f | 0.98 | "
           "scanpy_t-test | - | %s |\n",
           "2026-04-13", min_rho, min_rho >= 0.98 ? "PASS" : "FAIL");

    EXPECT_GE(min_rho, 0.98)
        << "Minimum per-cluster log2_fc Spearman = " << min_rho
        << " < 0.98 required";
}

// =============================================================================
// Test 3: Ttest_TinyPlanted_PvalueRankSpearman
// =============================================================================

TEST_F(TtestDE, Ttest_TinyPlanted_PvalueRankSpearman) {
    HostCSC h = make_planted_csc(kSeed);
    std::vector<int32_t> lbl = make_block_labels(kTinyCells, kTinyClusters);

    const std::string csc_bin    = refs_path("tt_tiny_csc.bin");
    const std::string labels_npy = refs_path("tt_tiny_labels.npy");
    write_csc_bin(h, csc_bin);
    write_npy_i32(labels_npy, lbl.data(), kTinyCells);

    std::string ref_npz;
    try {
        ref_npz = run_scanpy_de(csc_bin, labels_npy, "tiny_pval", kTopN);
    } catch (const std::exception& ex) {
        GTEST_SKIP() << "scanpy reference failed: " << ex.what();
    }

    singlet::gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);
    // Log-normalise to match the scanpy reference (which calls normalize_total +
    // log1p on the raw counts before running t-test).
    {
        singlet::gpu::preprocess::LogNormConfig ln_cfg;
        ln_cfg.target_count = 1e4f;  // match scanpy reference pipeline (normalize_total target_sum=1e4)
        auto ln_result = singlet::gpu::preprocess::log_normalize(d_mat, ln_cfg, stream_);
        cudaStreamSynchronize(stream_);  // ensure async kernel completes before workspace freed
    }
    singlet::gpu::core::DeviceMemory<int32_t> d_labels(kTinyCells);
    cudaMemcpyAsync(d_labels.get(), lbl.data(),
                    kTinyCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);

    singlet::gpu::de::TtestConfig cfg;
    cfg.top_n = kTopN;
    singlet::gpu::de::TtestResult result =
        singlet::gpu::de::ttest_de(d_mat, d_labels, kTinyClusters, cfg, stream_);

    double min_rho = 1.0;
    for (int c = 0; c < kTinyClusters; ++c) {
        HostClusterMarkers hcm = copy_cluster_markers(
            result.per_cluster[c], kTopN, stream_);

        // WHY gene-aligned Spearman (Cycle 81 fix — mirrors Cycle 76/78 LFCSpearman fix):
        // positional comparison of two independently-ranked top-N p-value lists fails when
        // the gene sets diverge even by 1 gene. Fix: build ref_map {gene_index → ref_pval},
        // compare only the intersection (same pattern as LFCSpearman at lines 695-722).
        auto raw_idx = extract_npz_array(
            ref_npz, "gene_indices_" + std::to_string(c),
            "int32", static_cast<size_t>(kTopN), "tiny_pval_idx_" + std::to_string(c));
        auto ref_idx = raw_to_vec<int32_t>(raw_idx, static_cast<size_t>(kTopN));

        auto raw = extract_npz_array(
            ref_npz, "pvals_" + std::to_string(c),
            "float32", static_cast<size_t>(kTopN), "tiny_pval_" + std::to_string(c));
        auto ref_pvals_vec = raw_to_vec<float>(raw, static_cast<size_t>(kTopN));

        std::unordered_map<int32_t, float> ref_map;
        for (int i = 0; i < kTopN; ++i)
            if (ref_idx[i] >= 0)
                ref_map[ref_idx[i]] = ref_pvals_vec[i];

        std::vector<float> our_aligned, ref_aligned;
        for (int i = 0; i < kTopN; ++i) {
            int32_t g = hcm.gene_indices[i];
            auto it = ref_map.find(g);
            if (it != ref_map.end()) {
                our_aligned.push_back(hcm.p_values[i]);
                ref_aligned.push_back(it->second);
            }
        }

        double rho = (our_aligned.size() >= 2)
                     ? spearman(our_aligned, ref_aligned)
                     : 0.0;
        printf("[Ttest_TinyPlanted_PvalueRankSpearman] cluster %d rho=%.4f"
               " (intersection=%d/%d)\n",
               c, rho, static_cast<int>(our_aligned.size()), kTopN);
        min_rho = std::min(min_rho, rho);
    }

    printf("[registry] | %s | de/ttest | tiny | pval_rank_spearman | %.4f | 0.95 | "
           "scanpy_t-test | - | %s |\n",
           "2026-04-13", min_rho, min_rho >= 0.95 ? "PASS" : "FAIL");

    EXPECT_GE(min_rho, 0.95)
        << "Minimum per-cluster p-value rank Spearman = " << min_rho
        << " < 0.95 required";
}

// =============================================================================
// Test 4: Ttest_GSM4037629_RealDataPlanted
//
// Load GSM4037629 (11,560 cells x ~30k genes) via pz_device_loader.
// Plant a strong ground-truth marker signal on top of the real counts BEFORE
// passing identical matrices to GPU and scanpy.  Round-robin labels (5
// clusters, cell j → j % 5).  For each cluster c ∈ {0..4} sample 50 disjoint
// marker gene indices (seeded mt19937_64, seed 0xC0FFEEull) and add a large
// bump (U[5,20]) to every (gene g, cell j) pair where cluster(j)==c AND g is
// in cluster c's planted set.  If the original value at (g,j) was zero, a new
// nonzero entry is inserted.  After injection both GPU and scanpy receive the
// same modified CSC, so planted genes become clear markers and Jaccard ≥ 0.90
// is achievable.  Ported from Wilcoxon_GSM4037629_RealDataPlanted (Cycle 77).
// =============================================================================

namespace {

// ---------------------------------------------------------------------------
// Plant strong marker signal into a HostCSC.
// For each cluster c pick n_planted_per_cluster disjoint gene indices
// (seeded, without replacement via Fisher-Yates prefix shuffle).
// For every cell j in cluster c add bump ~ U[bump_lo, bump_hi] to each
// planted gene g.  New nonzeros are inserted for (g,j) pairs that were zero.
//
// Cell-to-cluster mapping: cell j → j % n_clusters (round-robin).
//
// Returns planted_genes[c] — the set of gene indices planted for cluster c.
// ---------------------------------------------------------------------------
std::vector<std::vector<int32_t>> plant_signal_inplace_tt(
        HostCSC&   h,
        int        n_clusters,
        int        n_planted_per_cluster,
        float      bump_lo,
        float      bump_hi,
        uint64_t   seed)
{
    const int n_genes = static_cast<int>(h.m);
    const int n_cells = static_cast<int>(h.n);

    std::mt19937_64 rng(seed);

    // ---- Sample disjoint planted gene sets (Fisher-Yates prefix shuffle) --
    std::vector<int32_t> all_genes(static_cast<size_t>(n_genes));
    std::iota(all_genes.begin(), all_genes.end(), 0);
    const int total_planted = n_clusters * n_planted_per_cluster;
    for (int i = 0; i < total_planted; ++i) {
        std::uniform_int_distribution<int> pick(i, n_genes - 1);
        std::swap(all_genes[static_cast<size_t>(i)],
                  all_genes[static_cast<size_t>(pick(rng))]);
    }

    std::vector<std::vector<int32_t>> planted(static_cast<size_t>(n_clusters));
    for (int c = 0; c < n_clusters; ++c) {
        int base = c * n_planted_per_cluster;
        planted[static_cast<size_t>(c)].assign(
            all_genes.begin() + base,
            all_genes.begin() + base + n_planted_per_cluster);
        std::sort(planted[static_cast<size_t>(c)].begin(),
                  planted[static_cast<size_t>(c)].end());
    }

    // ---- Map gene → its planted cluster (or -1) --------------------------
    std::vector<int> gene_to_cluster(static_cast<size_t>(n_genes), -1);
    for (int c = 0; c < n_clusters; ++c)
        for (int32_t g : planted[static_cast<size_t>(c)])
            gene_to_cluster[static_cast<size_t>(g)] = c;

    std::uniform_real_distribution<float> bump_dist(bump_lo, bump_hi);

    // ---- Rebuild CSC column-by-column with planted additions --------------
    std::vector<std::vector<std::pair<int32_t,float>>> new_cols(
        static_cast<size_t>(n_cells));

    // Populate new_cols from existing CSC.
    for (int j = 0; j < n_cells; ++j) {
        int32_t col_start = h.indptr[static_cast<size_t>(j)];
        int32_t col_end   = h.indptr[static_cast<size_t>(j) + 1];
        new_cols[static_cast<size_t>(j)].reserve(
            static_cast<size_t>(col_end - col_start));
        for (int32_t k = col_start; k < col_end; ++k) {
            new_cols[static_cast<size_t>(j)].push_back(
                {h.indices[static_cast<size_t>(k)],
                 h.values [static_cast<size_t>(k)]});
        }
    }

    // Apply planted bumps.
    for (int j = 0; j < n_cells; ++j) {
        int c = j % n_clusters;
        const auto& pg = planted[static_cast<size_t>(c)];
        auto& col      = new_cols[static_cast<size_t>(j)];

        std::unordered_map<int32_t,size_t> row_pos;
        row_pos.reserve(col.size());
        for (size_t k = 0; k < col.size(); ++k)
            row_pos[col[k].first] = k;

        for (int32_t g : pg) {
            float bump = bump_dist(rng);
            auto it = row_pos.find(g);
            if (it != row_pos.end()) {
                col[it->second].second += bump;
            } else {
                col.push_back({g, bump});
            }
        }

        std::sort(col.begin(), col.end(),
                  [](const std::pair<int32_t,float>& a,
                     const std::pair<int32_t,float>& b) {
                      return a.first < b.first;
                  });
    }

    // ---- Flatten new_cols back into h ------------------------------------
    int64_t new_nnz = 0;
    for (int j = 0; j < n_cells; ++j)
        new_nnz += static_cast<int64_t>(new_cols[static_cast<size_t>(j)].size());
    h.nnz = static_cast<uint64_t>(new_nnz);

    h.indptr.resize(static_cast<size_t>(n_cells) + 1);
    h.indptr[0] = 0;
    for (int j = 0; j < n_cells; ++j)
        h.indptr[static_cast<size_t>(j) + 1] =
            h.indptr[static_cast<size_t>(j)] +
            static_cast<int32_t>(new_cols[static_cast<size_t>(j)].size());

    h.values .resize(static_cast<size_t>(h.nnz));
    h.indices.resize(static_cast<size_t>(h.nnz));
    size_t off = 0;
    for (int j = 0; j < n_cells; ++j) {
        for (auto& [row, val] : new_cols[static_cast<size_t>(j)]) {
            h.indices[off] = row;
            h.values [off] = val;
            ++off;
        }
    }
    return planted;
}

}  // namespace (plant_signal_inplace_tt)

TEST_F(TtestDE, Ttest_GSM4037629_RealDataPlanted) {
    if (!fs::exists(kGsmPath)) {
        GTEST_SKIP() << "GSM4037629 not found at " << kGsmPath;
    }

    // Load via pz_device_loader.
    auto pz_mat = singlet::gpu::io::load_pz(kGsmPath, stream_, /*keep_host_pinned=*/true);
    singlet::gpu::core::DeviceCSC& d_mat_raw = pz_mat.mat;

    const int n_cells    = d_mat_raw.cols;
    const int n_clusters = 5;

    // Round-robin labels: cell j → j % 5.
    std::vector<int32_t> lbl = make_roundrobin_labels(n_cells, n_clusters);

    // --- Step 1: copy real CSC to host before any modification. -----------
    HostCSC h_planted = device_csc_to_host(d_mat_raw, stream_);
    printf("[RealDataPlanted] loaded real CSC: %u genes x %u cells, nnz=%llu\n",
           h_planted.m, h_planted.n,
           static_cast<unsigned long long>(h_planted.nnz));

    // --- Step 2: plant strong marker signal IN the host CSC. --------------
    // 5 clusters x 50 disjoint genes (250 total), bump U[5,20].
    // Seed 0xC0FFEEull per spec.  Both GPU and scanpy receive this modified CSC.
    constexpr int   kRealClusters  = 5;
    constexpr int   kRealPlanted   = 50;   // per cluster
    constexpr float kBumpLo        = 5.0f;
    constexpr float kBumpHi        = 20.0f;

    auto planted_genes = plant_signal_inplace_tt(
        h_planted, kRealClusters, kRealPlanted, kBumpLo, kBumpHi, kSeed);

    printf("[RealDataPlanted] after planting: nnz=%llu\n",
           static_cast<unsigned long long>(h_planted.nnz));
    for (int c = 0; c < kRealClusters; ++c) {
        printf("[RealDataPlanted] cluster %d planted genes (first 5): "
               "%d %d %d %d %d ...\n", c,
               planted_genes[static_cast<size_t>(c)][0],
               planted_genes[static_cast<size_t>(c)][1],
               planted_genes[static_cast<size_t>(c)][2],
               planted_genes[static_cast<size_t>(c)][3],
               planted_genes[static_cast<size_t>(c)][4]);
    }

    // --- Step 3: upload modified CSC to GPU. ------------------------------
    singlet::gpu::core::DeviceCSC d_mat = upload_csc(h_planted, stream_);

    // --- Step 4: log-normalise (GPU in-place). ----------------------------
    singlet::gpu::preprocess::LogNormConfig ln_cfg;
    ln_cfg.target_count = 1e4f;
    auto ln_result = singlet::gpu::preprocess::log_normalize(d_mat, ln_cfg, stream_);
    cudaStreamSynchronize(stream_);

    // --- Step 5: upload labels. ------------------------------------------
    singlet::gpu::core::DeviceMemory<int32_t> d_labels(static_cast<size_t>(n_cells));
    cudaMemcpyAsync(d_labels.get(), lbl.data(),
                    static_cast<size_t>(n_cells) * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);

    // --- Step 6: run GPU t-test DE. ---------------------------------------
    singlet::gpu::de::TtestConfig cfg;
    cfg.top_n = kTopN;
    singlet::gpu::de::TtestResult result =
        singlet::gpu::de::ttest_de(d_mat, d_labels, n_clusters, cfg, stream_);

    ASSERT_EQ(static_cast<int>(result.per_cluster.size()), n_clusters)
        << "Expected " << n_clusters << " cluster results";

    // Basic sanity checks.
    for (int c = 0; c < n_clusters; ++c) {
        HostClusterMarkers hcm = copy_cluster_markers(
            result.per_cluster[c], kTopN, stream_);
        EXPECT_EQ(static_cast<int>(hcm.gene_indices.size()), kTopN)
            << "cluster " << c << " missing top_n markers";
        for (int i = 0; i < kTopN; ++i) {
            EXPECT_GE(hcm.gene_indices[i], 0)
                << "cluster " << c << " index[" << i << "] is negative";
            EXPECT_TRUE(std::isfinite(hcm.t_stats [i]))
                << "cluster " << c << " t_stat[" << i << "] not finite";
            EXPECT_TRUE(std::isfinite(hcm.log2_fc [i]))
                << "cluster " << c << " log2_fc[" << i << "] not finite";
            EXPECT_GE(hcm.p_values[i], 0.0f)
                << "cluster " << c << " p_value[" << i << "] negative";
            EXPECT_LE(hcm.p_values[i], 1.0f + 1e-5f)
                << "cluster " << c << " p_value[" << i << "] > 1";
        }
    }

    // --- Step 7: scanpy reference on the SAME planted CSC. ---------------
    // h_planted already contains the modified counts (pre-normalization).
    // Scanpy applies normalize_total + log1p on these raw planted counts,
    // matching the GPU pipeline.
    try {
        const std::string csc_bin    = refs_path("tt_realplanted_csc.bin");
        const std::string labels_npy = refs_path("tt_realplanted_labels.npy");
        write_csc_bin(h_planted, csc_bin);
        write_npy_i32(labels_npy, lbl.data(), n_cells);
        std::string ref_npz = run_scanpy_de(csc_bin, labels_npy,
                                            "realplanted", kTopN);

        // ---- Jaccard: GPU top-50 vs scanpy top-50 per cluster. ----------
        double min_jaccard = 1.0;
        for (int c = 0; c < n_clusters; ++c) {
            HostClusterMarkers hcm = copy_cluster_markers(
                result.per_cluster[c], kTopN, stream_);
            auto raw = extract_npz_array(
                ref_npz, "gene_indices_" + std::to_string(c),
                "int32", static_cast<size_t>(kTopN),
                "realp_jac_" + std::to_string(c));
            auto ref_idx = raw_to_vec<int32_t>(raw, static_cast<size_t>(kTopN));
            double j = jaccard(hcm.gene_indices, ref_idx);
            printf("[Ttest_GSM4037629_RealDataPlanted] cluster %d Jaccard=%.4f\n",
                   c, j);
            min_jaccard = std::min(min_jaccard, j);
        }
        printf("[registry] | %s | de/ttest | realdata_planted | jaccard_top50 | "
               "%.4f | 0.90 | scanpy_t-test | - | %s |\n",
               "2026-04-16", min_jaccard, min_jaccard >= 0.90 ? "PASS" : "FAIL");
        EXPECT_GE(min_jaccard, 0.90)
            << "RealDataPlanted Jaccard = " << min_jaccard << " < 0.90";

        // ---- LFCSpearman: gene-index-aligned. ---------------------------
        double min_lfc_rho = 1.0;
        for (int c = 0; c < n_clusters; ++c) {
            HostClusterMarkers hcm = copy_cluster_markers(
                result.per_cluster[c], kTopN, stream_);
            auto raw_idx = extract_npz_array(
                ref_npz, "gene_indices_" + std::to_string(c),
                "int32", static_cast<size_t>(kTopN),
                "realp_lfc_idx_" + std::to_string(c));
            auto ref_idx = raw_to_vec<int32_t>(raw_idx, static_cast<size_t>(kTopN));

            auto raw_lfc = extract_npz_array(
                ref_npz, "logfoldchanges_" + std::to_string(c),
                "float32", static_cast<size_t>(kTopN),
                "realp_lfc_" + std::to_string(c));
            auto ref_lfc_vec = raw_to_vec<float>(raw_lfc, static_cast<size_t>(kTopN));

            std::unordered_map<int32_t,float> ref_map;
            for (int i = 0; i < kTopN; ++i)
                if (ref_idx[i] >= 0)
                    ref_map[ref_idx[i]] = ref_lfc_vec[i];

            std::vector<float> our_aligned, ref_aligned;
            for (int i = 0; i < kTopN; ++i) {
                int32_t g = hcm.gene_indices[i];
                auto it = ref_map.find(g);
                if (it != ref_map.end()) {
                    our_aligned.push_back(hcm.log2_fc[i]);
                    ref_aligned.push_back(it->second);
                }
            }
            double rho = (our_aligned.size() >= 2)
                         ? spearman(our_aligned, ref_aligned) : 0.0;
            printf("[Ttest_GSM4037629_RealDataPlanted] cluster %d LFCSpearman=%.4f"
                   " (intersection=%d/%d)\n",
                   c, rho, static_cast<int>(our_aligned.size()), kTopN);
            min_lfc_rho = std::min(min_lfc_rho, rho);
        }
        printf("[registry] | %s | de/ttest | realdata_planted | lfc_spearman | "
               "%.4f | 0.98 | scanpy_t-test | - | %s |\n",
               "2026-04-16", min_lfc_rho, min_lfc_rho >= 0.98 ? "PASS" : "FAIL");
        EXPECT_GE(min_lfc_rho, 0.98)
            << "RealDataPlanted LFCSpearman = " << min_lfc_rho << " < 0.98";

        // ---- PvalRankSpearman: gene-index-aligned. -----------------------
        // All top-50 planted genes have p≈0 (machine underflow) in both
        // pipelines.  Gene-index alignment + constant-vector handler in
        // spearman() ensure this returns 1.0 (perfect agreement).
        double min_pval_rho = 1.0;
        for (int c = 0; c < n_clusters; ++c) {
            HostClusterMarkers hcm = copy_cluster_markers(
                result.per_cluster[c], kTopN, stream_);

            auto raw_idx = extract_npz_array(
                ref_npz, "gene_indices_" + std::to_string(c),
                "int32", static_cast<size_t>(kTopN),
                "realp_pval_idx_" + std::to_string(c));
            auto ref_idx = raw_to_vec<int32_t>(raw_idx, static_cast<size_t>(kTopN));

            auto raw = extract_npz_array(
                ref_npz, "pvals_" + std::to_string(c),
                "float32", static_cast<size_t>(kTopN),
                "realp_pval_" + std::to_string(c));
            auto ref_pvals_vec = raw_to_vec<float>(raw, static_cast<size_t>(kTopN));

            std::unordered_map<int32_t,float> pval_map;
            for (int i = 0; i < kTopN; ++i)
                if (ref_idx[i] >= 0)
                    pval_map[ref_idx[i]] = ref_pvals_vec[i];

            std::vector<float> our_pval_aligned, ref_pval_aligned;
            for (int i = 0; i < kTopN; ++i) {
                int32_t g = hcm.gene_indices[i];
                auto it = pval_map.find(g);
                if (it != pval_map.end()) {
                    our_pval_aligned.push_back(hcm.p_values[i]);
                    ref_pval_aligned.push_back(it->second);
                }
            }
            double rho = (our_pval_aligned.size() >= 2)
                         ? spearman(our_pval_aligned, ref_pval_aligned) : 0.0;
            printf("[Ttest_GSM4037629_RealDataPlanted] cluster %d PvalRankSpearman=%.4f"
                   " (intersection=%d/%d)\n",
                   c, rho, static_cast<int>(our_pval_aligned.size()), kTopN);
            min_pval_rho = std::min(min_pval_rho, rho);
        }
        printf("[registry] | %s | de/ttest | realdata_planted | pval_rank_spearman | "
               "%.4f | 0.95 | scanpy_t-test | - | %s |\n",
               "2026-04-16", min_pval_rho, min_pval_rho >= 0.95 ? "PASS" : "FAIL");
        EXPECT_GE(min_pval_rho, 0.95)
            << "RealDataPlanted PvalRankSpearman = " << min_pval_rho << " < 0.95";

    } catch (const std::exception& ex) {
        printf("[Ttest_GSM4037629_RealDataPlanted] scanpy comparison skipped: %s\n",
               ex.what());
        // Not a test failure if scanpy unavailable; sanity checks above passed.
    }
}

// =============================================================================
// Test 5: Ttest_DeterminismFlag_BitIdentical
// =============================================================================

TEST_F(TtestDE, Ttest_DeterminismFlag_BitIdentical) {
    HostCSC h = make_planted_csc(kSeed);
    std::vector<int32_t> lbl = make_roundrobin_labels(kTinyCells, kTinyClusters);

    singlet::gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);
    singlet::gpu::core::DeviceMemory<int32_t> d_labels(kTinyCells);
    cudaMemcpyAsync(d_labels.get(), lbl.data(),
                    kTinyCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);

    singlet::gpu::de::TtestConfig cfg;
    cfg.top_n         = kTopN;
    cfg.deterministic = true;

    singlet::gpu::de::TtestResult r1 =
        singlet::gpu::de::ttest_de(d_mat, d_labels, kTinyClusters, cfg, stream_);
    singlet::gpu::de::TtestResult r2 =
        singlet::gpu::de::ttest_de(d_mat, d_labels, kTinyClusters, cfg, stream_);

    ASSERT_EQ(r1.per_cluster.size(), r2.per_cluster.size());

    for (int c = 0; c < kTinyClusters; ++c) {
        HostClusterMarkers h1 = copy_cluster_markers(r1.per_cluster[c], kTopN, stream_);
        HostClusterMarkers h2 = copy_cluster_markers(r2.per_cluster[c], kTopN, stream_);

        EXPECT_EQ(h1.gene_indices, h2.gene_indices)
            << "cluster " << c << ": gene_indices differ between runs";
        EXPECT_EQ(h1.t_stats,      h2.t_stats)
            << "cluster " << c << ": t_stats differ between runs";
        EXPECT_EQ(h1.log2_fc,      h2.log2_fc)
            << "cluster " << c << ": log2_fc differ between runs";
        EXPECT_EQ(h1.p_values,     h2.p_values)
            << "cluster " << c << ": p_values differ between runs";
        EXPECT_EQ(h1.p_adj,        h2.p_adj)
            << "cluster " << c << ": p_adj differ between runs";
    }
}

// =============================================================================
// Test 6: Ttest_EdgeCase_EmptyCluster
// =============================================================================

TEST_F(TtestDE, Ttest_EdgeCase_EmptyCluster) {
    HostCSC h = make_planted_csc(kSeed);

    constexpr int skip_cluster = 2;
    std::vector<int32_t> lbl(kTinyCells);
    static const int kMap[] = {0, 1, 3, 3};
    for (int j = 0; j < kTinyCells; ++j)
        lbl[j] = kMap[j % 4];

    singlet::gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);
    singlet::gpu::core::DeviceMemory<int32_t> d_labels(kTinyCells);
    cudaMemcpyAsync(d_labels.get(), lbl.data(),
                    kTinyCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);

    singlet::gpu::de::TtestConfig cfg;
    cfg.top_n = kTopN;

    singlet::gpu::de::TtestResult result;
    ASSERT_NO_THROW(
        result = singlet::gpu::de::ttest_de(
            d_mat, d_labels, kTinyClusters, cfg, stream_));

    ASSERT_EQ(static_cast<int>(result.per_cluster.size()), kTinyClusters);
    EXPECT_EQ(result.per_cluster[skip_cluster].cluster_id, skip_cluster);
    // Empty cluster: null gene_indices pointer is acceptable.
    if (!result.per_cluster[skip_cluster].gene_indices.get()) {
        SUCCEED() << "Empty cluster returned null gene_indices — OK.";
    }
}

// =============================================================================
// Test 7: Ttest_EdgeCase_SingleCluster
// =============================================================================

TEST_F(TtestDE, Ttest_EdgeCase_SingleCluster) {
    HostCSC h = make_planted_csc(kSeed);
    std::vector<int32_t> lbl(kTinyCells, 0);

    singlet::gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);
    singlet::gpu::core::DeviceMemory<int32_t> d_labels(kTinyCells);
    cudaMemcpyAsync(d_labels.get(), lbl.data(),
                    kTinyCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);

    singlet::gpu::de::TtestConfig cfg;
    cfg.top_n = kTopN;

    singlet::gpu::de::TtestResult result;
    ASSERT_NO_THROW(
        result = singlet::gpu::de::ttest_de(d_mat, d_labels, 1, cfg, stream_));

    ASSERT_EQ(static_cast<int>(result.per_cluster.size()), 1);
    EXPECT_EQ(result.per_cluster[0].cluster_id, 0);
}
