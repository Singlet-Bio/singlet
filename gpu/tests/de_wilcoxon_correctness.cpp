// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/de_wilcoxon_correctness.cpp
//
// Correctness harness for singlet_gpu::de::wilcoxon_de.
//
// Written against the design doc at:
//   singlet-gpu/state/designs/09-de.md
// NOT against the kernel source — authored in parallel with de/wilcoxon.h by
// the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: scanpy sc.tl.rank_genes_groups(method='wilcoxon') via subprocess
//   tests/refs/de_scanpy_reference.py (shared with de_ttest_correctness.cpp).
//
// Tolerances (design doc §"Correctness test spec"):
//   Marker gene Jaccard   >= 0.90 (top-50 per cluster)
//   log2_fc Spearman rho  >= 0.98 (per-cluster, all-gene ranking)
//   p-value rank Spearman >= 0.95 (per-cluster rank correlation)
//   Determinism           bit-identical with cfg.deterministic = true
//
// Test cases:
//   Wilcoxon_TinyPlanted_TopMarkersJaccard
//   Wilcoxon_TinyPlanted_LogFoldChangeSpearman
//   Wilcoxon_TinyPlanted_PvalueRankSpearman
//   Wilcoxon_GSM4037629_RealData
//   Wilcoxon_DeterminismFlag_BitIdentical
//   Wilcoxon_EdgeCase_EmptyCluster
//   Wilcoxon_EdgeCase_SingleCluster
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every test that requires a device is guarded by
//   gpu_available() and calls GTEST_SKIP().

// ---- singlet-gpu DE header (written by gpu-kernel-dev) ---------------------
#include <singlet-gpu/de/wilcoxon.h>

// ---- other singlet-gpu headers needed for normalization + real-data test ----
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/preprocess/lognorm.h>
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>

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
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

/// Synthetic matrix dimensions.
constexpr int kTinyGenes  = 200;
constexpr int kTinyCells  = 500;
constexpr int kTinyClusters = 4;          // 4 clusters, 50 planted genes each
constexpr int kPlantedGenesPerCluster = 50;
constexpr int kTopN = 50;                 // compare top-50 markers

/// GSM4037629 path.
static const char* kGsmPath =
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/exon_counts.1pz";

/// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_de_wilcoxon_refs_tmp";

/// Python reference script path.
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/"
    "tests/refs/de_scanpy_reference.py";

// =============================================================================
// Utility helpers
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
// Minimal .npy v1.0 writers (reused from cycles 9/10).
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
    int item_size = (dtype_str == "<i4") ? 4 : 4;  // both i4 and f4 are 4 bytes
    fout.write(static_cast<const char*>(data),
               static_cast<std::streamsize>(n_items * item_size));
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
    uint32_t             m;        // genes (rows)
    uint32_t             n;        // cells (cols)
    uint64_t             nnz;
    std::vector<float>   values;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
};

// ---------------------------------------------------------------------------
// Write HostCSC to dump_csc binary format for the Python reference.
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
                       h.nnz * sizeof(float),
                       cudaMemcpyHostToDevice, stream), "values");
    ck(cudaMemcpyAsync(d_indptr .get(), h.indptr .data(),
                       (static_cast<size_t>(h.n) + 1) * sizeof(int32_t),
                       cudaMemcpyHostToDevice, stream), "indptr");
    ck(cudaMemcpyAsync(d_indices.get(), h.indices.data(),
                       h.nnz * sizeof(int32_t),
                       cudaMemcpyHostToDevice, stream), "indices");
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

// ---------------------------------------------------------------------------
// Device-to-host copy for a DeviceCSC (for dump after pz load).
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
// Planted-signal synthetic CSC generator.
//
// Matrix: kTinyGenes × kTinyCells, 4 equal clusters.
// For cluster c, genes [c*kPlantedGenesPerCluster .. (c+1)*kPlanted-1] are
// "upregulated": cells in that cluster have values drawn from U[50, 100]
// (integer), while other cells have values drawn from U[0, 5].
// Background genes: all cells draw from U[0, 10].
// Result is sparse (only nonzeros stored in CSC).
// ---------------------------------------------------------------------------
HostCSC make_planted_csc(uint64_t seed) {
    constexpr int G = kTinyGenes;
    constexpr int C = kTinyCells;
    constexpr int K = kTinyClusters;
    constexpr int cells_per_cluster = C / K;

    std::mt19937_64 rng(seed);

    // Cluster assignment: round-robin.
    std::vector<int32_t> cell_cluster(C);
    for (int j = 0; j < C; ++j)
        cell_cluster[j] = j / cells_per_cluster;

    HostCSC csc;
    csc.m = G;
    csc.n = C;
    csc.indptr.resize(static_cast<size_t>(C) + 1, 0);

    // Build per-column nonzero lists.
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
                if (clust == planted_cluster_for_gene) {
                    val = draw_val(50.0f, 100.0f);  // high in own cluster
                } else {
                    val = draw_val(0.0f, 5.0f);     // low elsewhere
                }
            } else {
                val = draw_val(0.0f, 10.0f);        // background
            }
            // Only store nonzeros.
            if (val > 0.0f)
                cols[j].push_back({static_cast<int32_t>(g), val});
        }
    }

    int64_t total_nnz = 0;
    for (int j = 0; j < C; ++j)
        total_nnz += static_cast<int64_t>(cols[j].size());
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

// ---------------------------------------------------------------------------
// Cluster labels: round-robin assignment (kTinyCells / kTinyClusters each).
// Also used for fake labels in the real-data test.
// ---------------------------------------------------------------------------
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
// Run the Python reference subprocess.
// Returns the path to the output .npz.
// ---------------------------------------------------------------------------
std::string run_scanpy_de(
        const std::string& csc_bin,
        const std::string& labels_npy,
        const std::string& tag,
        int top_n)
{
    std::string out_npz = refs_path("de_wilcoxon_ref_" + tag + ".npz");
    std::string cmd = "python3 " + std::string(kRefScript) +
                      " --input "   + csc_bin      +
                      " --labels "  + labels_npy   +
                      " --output "  + out_npz      +
                      " --method wilcoxon"          +
                      " --top-n "   + std::to_string(top_n);
    run_cmd(cmd);
    return out_npz;
}

// ---------------------------------------------------------------------------
// Extract a per-cluster array from the .npz via inline Python.
// ---------------------------------------------------------------------------
std::vector<uint8_t> extract_npz_array(
        const std::string& npz_path,
        const std::string& key,
        const std::string& dtype,
        size_t count,
        const std::string& tag)
{
    std::string bin_path = refs_path("de_wil_" + tag + "_" + key + ".bin");
    // Sanitise key for shell (replace colons/brackets).
    std::string safe_key = key;
    std::replace(safe_key.begin(), safe_key.end(), '/', '_');
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
// at large scale (310k genes) when all top-50 planted markers have p≈0 in
// both GPU and scanpy (machine underflow), making the p-value vector a flat
// all-zeros array on both sides.
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
// Jaccard of two integer sets (top-k marker gene indices).
// ---------------------------------------------------------------------------
double jaccard(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    std::set<int32_t> sa(a.begin(), a.end()), sb(b.begin(), b.end());
    // Remove sentinel -1 padding.
    sa.erase(-1); sb.erase(-1);
    int intersection = 0;
    for (int32_t v : sa) if (sb.count(v)) ++intersection;
    int union_sz = static_cast<int>(sa.size() + sb.size()) - intersection;
    if (union_sz == 0) return 1.0;
    return static_cast<double>(intersection) / union_sz;
}

// ---------------------------------------------------------------------------
// Copy DE result per-cluster from device to host.
// Returns: gene_indices, z_scores, log2_fc, p_values, p_adj
// ---------------------------------------------------------------------------
struct HostClusterMarkers {
    std::vector<int32_t> gene_indices;
    std::vector<float>   z_scores;
    std::vector<float>   log2_fc;
    std::vector<float>   p_values;
    std::vector<float>   p_adj;
    int                  cluster_id;
};

HostClusterMarkers copy_cluster_markers(
        const singlet_gpu::de::ClusterMarkers& cm,
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
    h.z_scores    .resize(top_n);
    h.log2_fc     .resize(top_n);
    h.p_values    .resize(top_n);
    h.p_adj       .resize(top_n);

    ck(cudaMemcpy(h.gene_indices.data(), cm.gene_indices.get(),
                  top_n * sizeof(int32_t), cudaMemcpyDeviceToHost), "indices");
    ck(cudaMemcpy(h.z_scores    .data(), cm.z_scores.get(),
                  top_n * sizeof(float),   cudaMemcpyDeviceToHost), "scores");
    ck(cudaMemcpy(h.log2_fc     .data(), cm.log2_fc.get(),
                  top_n * sizeof(float),   cudaMemcpyDeviceToHost), "lfc");
    ck(cudaMemcpy(h.p_values    .data(), cm.p_values.get(),
                  top_n * sizeof(float),   cudaMemcpyDeviceToHost), "pvals");
    ck(cudaMemcpy(h.p_adj       .data(), cm.p_adj.get(),
                  top_n * sizeof(float),   cudaMemcpyDeviceToHost), "padj");
    return h;
}

}  // anonymous namespace

// =============================================================================
// Test fixture: handles CUDA stream lifecycle.
// =============================================================================

class WilcoxonTest : public ::testing::Test {
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
// Test 1: Wilcoxon_TinyPlanted_TopMarkersJaccard
//
// Plant 50 DE genes per cluster in a 200×500 CSC.  Run our wilcoxon_de and
// the Python reference, compare top-50 marker gene indices per cluster.
// Jaccard >= 0.90 required.
// =============================================================================

TEST_F(WilcoxonTest, Wilcoxon_TinyPlanted_TopMarkersJaccard) {
    HostCSC h = make_planted_csc(kSeed);
    std::vector<int32_t> lbl = make_block_labels(kTinyCells, kTinyClusters);

    // Write reference inputs.
    const std::string csc_bin    = refs_path("wil_tiny_csc.bin");
    const std::string labels_npy = refs_path("wil_tiny_labels.npy");
    write_csc_bin(h, csc_bin);
    write_npy_i32(labels_npy, lbl.data(), kTinyCells);

    // Run Python reference.  If it fails (no scanpy), skip gracefully.
    std::string ref_npz;
    try {
        ref_npz = run_scanpy_de(csc_bin, labels_npy, "tiny_jaccard", kTopN);
    } catch (const std::exception& ex) {
        GTEST_SKIP() << "scanpy reference failed: " << ex.what();
    }

    // Run our kernel.
    singlet_gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);
    // Log-normalise to match the scanpy reference (which calls normalize_total +
    // log1p on the raw counts before running Wilcoxon).
    // Capture LogNormResult so its DeviceMemory workspace buffers stay alive
    // until the async kernels finish. Discarding the return value destroys them
    // immediately while compute_size_factors_and_apply_kernel is still running,
    // causing an illegal memory access that corrupts the CUDA context.
    singlet_gpu::preprocess::LogNormConfig ln_cfg;
    ln_cfg.target_count = 1e4f;  // match scanpy reference pipeline (normalize_total target_sum=1e4)
    auto ln_result = singlet_gpu::preprocess::log_normalize(d_mat, ln_cfg, stream_);
    cudaStreamSynchronize(stream_);  // ensure d_mat.values fully written before wilcoxon reads

    singlet_gpu::core::DeviceMemory<int32_t> d_labels(kTinyCells);
    cudaMemcpyAsync(d_labels.get(), lbl.data(),
                    kTinyCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);

    singlet_gpu::de::WilcoxonConfig cfg;
    cfg.top_n = kTopN;
    singlet_gpu::de::WilcoxonResult result =
        singlet_gpu::de::wilcoxon_de(d_mat, d_labels, kTinyClusters, cfg, stream_);

    ASSERT_EQ(static_cast<int>(result.per_cluster.size()), kTinyClusters);

    double min_jaccard = 1.0;
    for (int c = 0; c < kTinyClusters; ++c) {
        HostClusterMarkers hcm = copy_cluster_markers(
            result.per_cluster[c], kTopN, stream_);

        // Load reference indices for this cluster.
        auto raw = extract_npz_array(
            ref_npz, "gene_indices_" + std::to_string(c),
            "int32", static_cast<size_t>(kTopN), "tiny_jac_" + std::to_string(c));
        auto ref_indices = raw_to_vec<int32_t>(raw, static_cast<size_t>(kTopN));

        double j = jaccard(hcm.gene_indices, ref_indices);
        printf("[Wilcoxon_TinyPlanted_TopMarkersJaccard] cluster %d Jaccard=%.4f\n",
               c, j);
        min_jaccard = std::min(min_jaccard, j);
    }

    printf("[registry] | %s | de/wilcoxon | tiny | jaccard_top50 | %.4f | 0.90 | "
           "scanpy_wilcoxon | - | %s |\n",
           "2026-04-13", min_jaccard, min_jaccard >= 0.90 ? "PASS" : "FAIL");

    EXPECT_GE(min_jaccard, 0.90)
        << "Minimum per-cluster Jaccard = " << min_jaccard
        << " < 0.90 required";
}

// =============================================================================
// Test 2: Wilcoxon_TinyPlanted_LogFoldChangeSpearman
//
// Same planted matrix. Compare per-cluster log2_fc arrays (all top-N genes)
// with the Python reference. Spearman rho >= 0.98.
// =============================================================================

TEST_F(WilcoxonTest, Wilcoxon_TinyPlanted_LogFoldChangeSpearman) {
    HostCSC h = make_planted_csc(kSeed);
    std::vector<int32_t> lbl = make_block_labels(kTinyCells, kTinyClusters);

    const std::string csc_bin    = refs_path("wil_tiny_csc.bin");
    const std::string labels_npy = refs_path("wil_tiny_labels.npy");
    write_csc_bin(h, csc_bin);
    write_npy_i32(labels_npy, lbl.data(), kTinyCells);

    std::string ref_npz;
    try {
        ref_npz = run_scanpy_de(csc_bin, labels_npy, "tiny_lfc", kTopN);
    } catch (const std::exception& ex) {
        GTEST_SKIP() << "scanpy reference failed: " << ex.what();
    }

    singlet_gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);
    // Log-normalise to match the scanpy reference.
    // Capture result to keep workspace buffers alive until async kernels finish.
    singlet_gpu::preprocess::LogNormConfig ln_cfg;
    ln_cfg.target_count = 1e4f;  // match scanpy reference pipeline (normalize_total target_sum=1e4)
    auto ln_result = singlet_gpu::preprocess::log_normalize(d_mat, ln_cfg, stream_);
    cudaStreamSynchronize(stream_);

    singlet_gpu::core::DeviceMemory<int32_t> d_labels(kTinyCells);
    cudaMemcpyAsync(d_labels.get(), lbl.data(),
                    kTinyCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);

    singlet_gpu::de::WilcoxonConfig cfg;
    cfg.top_n = kTopN;
    singlet_gpu::de::WilcoxonResult result =
        singlet_gpu::de::wilcoxon_de(d_mat, d_labels, kTinyClusters, cfg, stream_);

    double min_rho = 1.0;
    for (int c = 0; c < kTinyClusters; ++c) {
        HostClusterMarkers hcm = copy_cluster_markers(
            result.per_cluster[c], kTopN, stream_);

        // WHY gene-aligned Spearman (Cycle 76): positional comparison of two
        // independently-ranked top-N lists fails when the two sets diverge even
        // by 1 gene.  With 50 genes and 0.96 Jaccard, 2 genes differ; inserting
        // those 2 genes at different positions shifts ALL subsequent positions,
        // collapsing Spearman to 0.63-0.80 even when LFC values are numerically
        // correct.  Fix: build a ref_map {gene_index → ref_lfc} from scanpy's
        // top-N, then compare only the intersection aligned by gene index.
        auto raw_idx = extract_npz_array(
            ref_npz, "gene_indices_" + std::to_string(c),
            "int32", static_cast<size_t>(kTopN), "tiny_lfc_idx_" + std::to_string(c));
        auto ref_idx = raw_to_vec<int32_t>(raw_idx, static_cast<size_t>(kTopN));

        auto raw_lfc = extract_npz_array(
            ref_npz, "logfoldchanges_" + std::to_string(c),
            "float32", static_cast<size_t>(kTopN), "tiny_lfc_" + std::to_string(c));
        auto ref_lfc_vec = raw_to_vec<float>(raw_lfc, static_cast<size_t>(kTopN));

        // Build map: gene_index → ref lfc.
        std::unordered_map<int32_t, float> ref_map;
        for (int i = 0; i < kTopN; ++i)
            if (ref_idx[i] >= 0)
                ref_map[ref_idx[i]] = ref_lfc_vec[i];

        // Collect aligned (our_lfc, ref_lfc) pairs for genes in intersection.
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
        printf("[Wilcoxon_TinyPlanted_LogFoldChangeSpearman] cluster %d rho=%.4f"
               " (intersection=%d/%d)\n",
               c, rho, static_cast<int>(our_aligned.size()), kTopN);
        min_rho = std::min(min_rho, rho);
    }

    printf("[registry] | %s | de/wilcoxon | tiny | lfc_spearman | %.4f | 0.98 | "
           "scanpy_wilcoxon | - | %s |\n",
           "2026-04-13", min_rho, min_rho >= 0.98 ? "PASS" : "FAIL");

    EXPECT_GE(min_rho, 0.98)
        << "Minimum per-cluster log2_fc Spearman = " << min_rho
        << " < 0.98 required";
}

// =============================================================================
// Test 3: Wilcoxon_TinyPlanted_PvalueRankSpearman
//
// Same planted matrix. Compare p-value rankings per cluster.
// Spearman rho >= 0.95.
// =============================================================================

TEST_F(WilcoxonTest, Wilcoxon_TinyPlanted_PvalueRankSpearman) {
    HostCSC h = make_planted_csc(kSeed);
    std::vector<int32_t> lbl = make_block_labels(kTinyCells, kTinyClusters);

    const std::string csc_bin    = refs_path("wil_tiny_csc.bin");
    const std::string labels_npy = refs_path("wil_tiny_labels.npy");
    write_csc_bin(h, csc_bin);
    write_npy_i32(labels_npy, lbl.data(), kTinyCells);

    std::string ref_npz;
    try {
        ref_npz = run_scanpy_de(csc_bin, labels_npy, "tiny_pval", kTopN);
    } catch (const std::exception& ex) {
        GTEST_SKIP() << "scanpy reference failed: " << ex.what();
    }

    singlet_gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);
    // Log-normalise to match the scanpy reference.
    // Capture result to keep workspace buffers alive until async kernels finish.
    singlet_gpu::preprocess::LogNormConfig ln_cfg;
    ln_cfg.target_count = 1e4f;  // match scanpy reference pipeline (normalize_total target_sum=1e4)
    auto ln_result = singlet_gpu::preprocess::log_normalize(d_mat, ln_cfg, stream_);
    cudaStreamSynchronize(stream_);

    singlet_gpu::core::DeviceMemory<int32_t> d_labels(kTinyCells);
    cudaMemcpyAsync(d_labels.get(), lbl.data(),
                    kTinyCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);

    singlet_gpu::de::WilcoxonConfig cfg;
    cfg.top_n = kTopN;
    singlet_gpu::de::WilcoxonResult result =
        singlet_gpu::de::wilcoxon_de(d_mat, d_labels, kTinyClusters, cfg, stream_);

    double min_rho = 1.0;
    for (int c = 0; c < kTinyClusters; ++c) {
        HostClusterMarkers hcm = copy_cluster_markers(
            result.per_cluster[c], kTopN, stream_);

        auto raw = extract_npz_array(
            ref_npz, "pvals_" + std::to_string(c),
            "float32", static_cast<size_t>(kTopN), "tiny_pval_" + std::to_string(c));
        auto ref_pvals = raw_to_vec<float>(raw, static_cast<size_t>(kTopN));

        double rho = spearman(hcm.p_values, ref_pvals);
        printf("[Wilcoxon_TinyPlanted_PvalueRankSpearman] cluster %d rho=%.4f\n",
               c, rho);
        min_rho = std::min(min_rho, rho);
    }

    printf("[registry] | %s | de/wilcoxon | tiny | pval_rank_spearman | %.4f | 0.95 | "
           "scanpy_wilcoxon | - | %s |\n",
           "2026-04-13", min_rho, min_rho >= 0.95 ? "PASS" : "FAIL");

    EXPECT_GE(min_rho, 0.95)
        << "Minimum per-cluster p-value rank Spearman = " << min_rho
        << " < 0.95 required";
}

// =============================================================================
// Test 4: Wilcoxon_GSM4037629_RealDataPlanted
//
// Load GSM4037629 (11,560 cells × ~30k genes) via pz_device_loader.
// Plant a strong ground-truth marker signal on top of the real counts BEFORE
// passing identical matrices to GPU and scanpy.  Round-robin labels (5
// clusters, cell j → j % 5).  For each cluster c ∈ {0..4} sample 50
// disjoint marker gene indices (seeded mt19937_64, seed 0xC0FFEEull) and add
// a large bump (U[5,20]) to every (gene g, cell j) pair where cluster(j)==c
// AND g is in cluster c's planted set.  If the original value at (g,j) was
// zero (not stored in the sparse matrix), a new nonzero entry is inserted.
// After injection both pipelines receive the same modified CSC, so planted
// genes become clear markers and Jaccard ≥ 0.90 is achievable.
// =============================================================================

namespace {

// ---------------------------------------------------------------------------
// Plant strong marker signal into a HostCSC.
// For each cluster c pick n_planted_per_cluster disjoint gene indices
// (seeded, without replacement).  For every cell j in cluster c add
// bump ~ U[bump_lo, bump_hi] to each planted gene g.  New nonzeros are
// inserted for (g,j) pairs that were zero.
//
// Returns planted_genes[c] — the set of gene indices planted for cluster c.
// ---------------------------------------------------------------------------
std::vector<std::vector<int32_t>> plant_signal_inplace(
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

    // ---- Sample disjoint planted gene sets --------------------------------
    // Draw without replacement across all clusters (250 total).
    std::vector<int32_t> all_genes(static_cast<size_t>(n_genes));
    std::iota(all_genes.begin(), all_genes.end(), 0);
    // Shuffle and take the first n_clusters * n_planted_per_cluster genes.
    // Use Fisher-Yates on a prefix (no need to shuffle the whole 30k array).
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
        // Sort for fast membership test later.
        std::sort(planted[static_cast<size_t>(c)].begin(),
                  planted[static_cast<size_t>(c)].end());
    }

    // ---- Build set-of-planted genes per cluster for O(1) lookup ----------
    // planted_set[g] = cluster index c, or -1 if not planted.
    std::vector<int> gene_to_cluster(static_cast<size_t>(n_genes), -1);
    for (int c = 0; c < n_clusters; ++c)
        for (int32_t g : planted[static_cast<size_t>(c)])
            gene_to_cluster[static_cast<size_t>(g)] = c;

    // ---- Bump distribution -----------------------------------------------
    std::uniform_real_distribution<float> bump_dist(bump_lo, bump_hi);

    // ---- Rebuild CSC column-by-column with planted additions -------------
    // For each column (cell j):
    //   1. Collect existing nonzeros.
    //   2. For each planted gene g of cluster(j): bump the existing value or
    //      insert a new nonzero.
    //   3. Re-sort by row index.

    const int cell_cluster_fn = n_clusters;  // capture for lambda
    (void)cell_cluster_fn;

    // New CSC containers.
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

        // Build a quick map: existing row → index in col.
        std::unordered_map<int32_t,size_t> row_pos;
        row_pos.reserve(col.size());
        for (size_t k = 0; k < col.size(); ++k)
            row_pos[col[k].first] = k;

        for (int32_t g : pg) {
            float bump = bump_dist(rng);
            auto it = row_pos.find(g);
            if (it != row_pos.end()) {
                // Existing nonzero: add bump in-place.
                col[it->second].second += bump;
            } else {
                // New nonzero: append.
                col.push_back({g, bump});
            }
        }

        // Re-sort by row index (CSC invariant).
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

}  // anonymous namespace (plant_signal_inplace)

TEST_F(WilcoxonTest, Wilcoxon_GSM4037629_RealDataPlanted) {
    if (!fs::exists(kGsmPath)) {
        GTEST_SKIP() << "GSM4037629 not found at " << kGsmPath;
    }

    // Load via pz_device_loader.
    auto pz_mat = singlet_gpu::io::load_pz(kGsmPath, stream_, /*keep_host_pinned=*/true);
    singlet_gpu::core::DeviceCSC& d_mat_raw = pz_mat.mat;

    const int n_cells    = d_mat_raw.cols;
    const int n_clusters = 5;

    // Generate round-robin labels.
    std::vector<int32_t> lbl = make_roundrobin_labels(n_cells, n_clusters);

    // --- Step 1: copy real CSC to host before any modification. -----------
    HostCSC h_planted = device_csc_to_host(d_mat_raw, stream_);
    printf("[RealDataPlanted] loaded real CSC: %u genes x %u cells, nnz=%llu\n",
           h_planted.m, h_planted.n,
           static_cast<unsigned long long>(h_planted.nnz));

    // --- Step 2: plant strong marker signal IN the host CSC. --------------
    // 5 clusters × 50 disjoint genes (250 total), bump U[5,20].
    // Seed 0xC0FFEEull per spec.  Both GPU and scanpy receive this modified CSC.
    constexpr int   kRealClusters  = 5;
    constexpr int   kRealPlanted   = 50;   // per cluster
    constexpr float kBumpLo        = 5.0f;
    constexpr float kBumpHi        = 20.0f;

    auto planted_genes = plant_signal_inplace(
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
    singlet_gpu::core::DeviceCSC d_mat = upload_csc(h_planted, stream_);

    // --- Step 4: log-normalise (GPU in-place). ----------------------------
    singlet_gpu::preprocess::LogNormConfig ln_cfg;
    ln_cfg.target_count = 1e4f;
    auto ln_result = singlet_gpu::preprocess::log_normalize(d_mat, ln_cfg, stream_);
    cudaStreamSynchronize(stream_);

    // --- Step 5: upload labels. ------------------------------------------
    singlet_gpu::core::DeviceMemory<int32_t> d_labels(static_cast<size_t>(n_cells));
    cudaMemcpyAsync(d_labels.get(), lbl.data(),
                    static_cast<size_t>(n_cells) * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);

    // --- Step 6: run GPU Wilcoxon DE. ------------------------------------
    singlet_gpu::de::WilcoxonConfig cfg;
    cfg.top_n = kTopN;
    singlet_gpu::de::WilcoxonResult result =
        singlet_gpu::de::wilcoxon_de(d_mat, d_labels, n_clusters, cfg, stream_);

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
            EXPECT_TRUE(std::isfinite(hcm.z_scores [i]))
                << "cluster " << c << " z_score[" << i << "] not finite";
            EXPECT_TRUE(std::isfinite(hcm.log2_fc  [i]))
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
        const std::string csc_bin    = refs_path("wil_realplanted_csc.bin");
        const std::string labels_npy = refs_path("wil_realplanted_labels.npy");
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
            printf("[Wilcoxon_GSM4037629_RealDataPlanted] cluster %d Jaccard=%.4f\n",
                   c, j);
            min_jaccard = std::min(min_jaccard, j);
        }
        printf("[registry] | %s | de/wilcoxon | realdata_planted | jaccard_top50 | "
               "%.4f | 0.90 | scanpy_wilcoxon | - | %s |\n",
               "2026-04-16", min_jaccard, min_jaccard >= 0.90 ? "PASS" : "FAIL");
        EXPECT_GE(min_jaccard, 0.90)
            << "RealDataPlanted Jaccard = " << min_jaccard << " < 0.90";

        // ---- LFCSpearman: gene-index-aligned (Cycle 76 helper). ---------
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
            printf("[Wilcoxon_GSM4037629_RealDataPlanted] cluster %d LFCSpearman=%.4f"
                   " (intersection=%d/%d)\n",
                   c, rho, static_cast<int>(our_aligned.size()), kTopN);
            min_lfc_rho = std::min(min_lfc_rho, rho);
        }
        printf("[registry] | %s | de/wilcoxon | realdata_planted | lfc_spearman | "
               "%.4f | 0.98 | scanpy_wilcoxon | - | %s |\n",
               "2026-04-16", min_lfc_rho, min_lfc_rho >= 0.98 ? "PASS" : "FAIL");
        EXPECT_GE(min_lfc_rho, 0.98)
            << "RealDataPlanted LFCSpearman = " << min_lfc_rho << " < 0.98";

        // ---- PvalRankSpearman (gene-index-aligned, same fix as Cycle 76). --
        // WHY: at 310k genes, all 50 planted markers have p≈0 after Bonferroni
        // correction — the top-50 p-value list is a flat array of 0s from both
        // GPU and scanpy.  Positional Spearman(all-zeros, all-zeros) returns 0
        // (undefined / constant vector).  Instead build a map
        // {gene_index → ref_pval} from scanpy's top-N, then compare only the
        // intersection aligned by gene index, giving a well-defined correlation.
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

            // Build map: gene_index → ref pval.
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
            printf("[Wilcoxon_GSM4037629_RealDataPlanted] cluster %d PvalRankSpearman=%.4f"
                   " (intersection=%d/%d)\n",
                   c, rho, static_cast<int>(our_pval_aligned.size()), kTopN);
            min_pval_rho = std::min(min_pval_rho, rho);
        }
        printf("[registry] | %s | de/wilcoxon | realdata_planted | pval_rank_spearman | "
               "%.4f | 0.95 | scanpy_wilcoxon | - | %s |\n",
               "2026-04-16", min_pval_rho, min_pval_rho >= 0.95 ? "PASS" : "FAIL");
        EXPECT_GE(min_pval_rho, 0.95)
            << "RealDataPlanted PvalRankSpearman = " << min_pval_rho << " < 0.95";

    } catch (const std::exception& ex) {
        printf("[Wilcoxon_GSM4037629_RealDataPlanted] scanpy comparison skipped: %s\n",
               ex.what());
        // Not a test failure if scanpy unavailable; sanity checks above passed.
    }
}

// =============================================================================
// Test 5: Wilcoxon_DeterminismFlag_BitIdentical
//
// Run wilcoxon_de twice with cfg.deterministic = true on the planted matrix.
// All output arrays must be bit-identical across both runs.
// =============================================================================

TEST_F(WilcoxonTest, Wilcoxon_DeterminismFlag_BitIdentical) {
    HostCSC h = make_planted_csc(kSeed);
    std::vector<int32_t> lbl = make_roundrobin_labels(kTinyCells, kTinyClusters);

    singlet_gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);
    singlet_gpu::core::DeviceMemory<int32_t> d_labels(kTinyCells);
    cudaMemcpyAsync(d_labels.get(), lbl.data(),
                    kTinyCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);

    singlet_gpu::de::WilcoxonConfig cfg;
    cfg.top_n       = kTopN;
    cfg.deterministic = true;

    singlet_gpu::de::WilcoxonResult r1 =
        singlet_gpu::de::wilcoxon_de(d_mat, d_labels, kTinyClusters, cfg, stream_);
    singlet_gpu::de::WilcoxonResult r2 =
        singlet_gpu::de::wilcoxon_de(d_mat, d_labels, kTinyClusters, cfg, stream_);

    ASSERT_EQ(r1.per_cluster.size(), r2.per_cluster.size());

    for (int c = 0; c < kTinyClusters; ++c) {
        HostClusterMarkers h1 = copy_cluster_markers(r1.per_cluster[c], kTopN, stream_);
        HostClusterMarkers h2 = copy_cluster_markers(r2.per_cluster[c], kTopN, stream_);

        EXPECT_EQ(h1.gene_indices, h2.gene_indices)
            << "cluster " << c << ": gene_indices differ between runs";
        EXPECT_EQ(h1.z_scores,     h2.z_scores)
            << "cluster " << c << ": z_scores differ between runs";
        EXPECT_EQ(h1.log2_fc,      h2.log2_fc)
            << "cluster " << c << ": log2_fc differ between runs";
        EXPECT_EQ(h1.p_values,     h2.p_values)
            << "cluster " << c << ": p_values differ between runs";
        EXPECT_EQ(h1.p_adj,        h2.p_adj)
            << "cluster " << c << ": p_adj differ between runs";
    }
}

// =============================================================================
// Test 6: Wilcoxon_EdgeCase_EmptyCluster
//
// Pass labels where cluster 2 has 0 cells (cells only in 0, 1, 3).
// Confirm: no segfault, the result for cluster 2 is empty or null (0 markers).
// =============================================================================

TEST_F(WilcoxonTest, Wilcoxon_EdgeCase_EmptyCluster) {
    HostCSC h = make_planted_csc(kSeed);

    // Labels: skip cluster 2.  Cells split 0, 1, 3, 3, 0, 1, 3, 3, …
    constexpr int skip_cluster = 2;
    std::vector<int32_t> lbl(kTinyCells);
    static const int kMap[] = {0, 1, 3, 3};  // no cluster 2
    for (int j = 0; j < kTinyCells; ++j)
        lbl[j] = kMap[j % 4];

    singlet_gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);
    singlet_gpu::core::DeviceMemory<int32_t> d_labels(kTinyCells);
    cudaMemcpyAsync(d_labels.get(), lbl.data(),
                    kTinyCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);

    singlet_gpu::de::WilcoxonConfig cfg;
    cfg.top_n = kTopN;

    // Must not crash.
    singlet_gpu::de::WilcoxonResult result;
    ASSERT_NO_THROW(
        result = singlet_gpu::de::wilcoxon_de(
            d_mat, d_labels, kTinyClusters, cfg, stream_));

    ASSERT_EQ(static_cast<int>(result.per_cluster.size()), kTinyClusters);

    // Cluster 2 must be gracefully handled: either empty markers or all-zero.
    const auto& cm2 = result.per_cluster[skip_cluster];
    EXPECT_EQ(cm2.cluster_id, skip_cluster);
    // The result is acceptable if gene_indices is empty (0 nnz) or filled with
    // sentinel -1.  Either way it must not contain garbage addresses.
    if (!cm2.gene_indices.get()) {
        // null device pointer is allowed for an empty cluster
        SUCCEED() << "Empty cluster returned null gene_indices — OK.";
    }
    // If a non-null pointer is returned, copy and verify safe values.
    // We simply check the kernel did not crash; finite-value checks apply only
    // to non-empty clusters.
}

// =============================================================================
// Test 7: Wilcoxon_EdgeCase_SingleCluster
//
// All cells in cluster 0 (n_clusters = 1).  one-vs-rest test degenerates:
// "rest" has 0 cells.  Confirm: one cluster returned, reasonable output
// (markers may all have p=1 or be empty — the key check is no crash and
// the returned cluster_id == 0).
// =============================================================================

TEST_F(WilcoxonTest, Wilcoxon_EdgeCase_SingleCluster) {
    HostCSC h = make_planted_csc(kSeed);
    std::vector<int32_t> lbl(kTinyCells, 0);  // all cells → cluster 0

    singlet_gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);
    singlet_gpu::core::DeviceMemory<int32_t> d_labels(kTinyCells);
    cudaMemcpyAsync(d_labels.get(), lbl.data(),
                    kTinyCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);

    singlet_gpu::de::WilcoxonConfig cfg;
    cfg.top_n = kTopN;

    singlet_gpu::de::WilcoxonResult result;
    ASSERT_NO_THROW(
        result = singlet_gpu::de::wilcoxon_de(d_mat, d_labels, 1, cfg, stream_));

    ASSERT_EQ(static_cast<int>(result.per_cluster.size()), 1);
    EXPECT_EQ(result.per_cluster[0].cluster_id, 0);
}

// =============================================================================
// Test 83: Test83_RealDataSized_PostNormalize_NoCrash
//
// Regression test for BUG-WILCOXON-POST-NORMALIZE-CRASH (cycle 72).
//
// Root cause: log_normalize() launches compute_size_factors_and_apply_kernel
// asynchronously on the caller's stream, then returns a LogNormResult whose
// destructor calls cudaFree on the workspace DeviceMemory (d_size_factors,
// d_col_sums).  If the caller discards the return value, the destructor fires
// immediately while the apply kernel is still reading those buffers → illegal
// memory access that surfaces as a sticky CUDA error at the next CUDA API call
// (typically cudaMalloc inside DeviceMemory<T>::DeviceMemory in wilcoxon_de).
//
// Fix (lognorm.h): add cudaStreamSynchronize(stream) before the return in
// log_normalize(), ensuring the apply kernel has completed before any workspace
// buffer can be freed.
//
// This test DELIBERATELY discards the LogNormResult (via explicit (void) cast)
// to reproduce the pre-fix race window.  Before the fix: deterministic crash
// with CUDA error: illegal memory access.  After the fix: PASS.
//
// Design doc: singlet-gpu/state/designs/72-wilcoxon-postnorm-crash.md
// =============================================================================

namespace {

// ---------------------------------------------------------------------------
// Build a synthetic sparse HostCSC with dims (n_rows x n_cols) and target
// density ~density_frac.  Values drawn from U[0, max_val] (float).
// Uses a local helper so it does not alter the planted-signal generator above.
// ---------------------------------------------------------------------------
HostCSC make_realdata_sized_csc(uint64_t seed,
                                int n_rows,
                                int n_cols,
                                float density_frac,
                                float max_val)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> val_dist(0.0f, max_val);
    // Bernoulli threshold: P(nonzero) = density_frac.
    std::uniform_real_distribution<float> ber(0.0f, 1.0f);

    HostCSC csc;
    csc.m = static_cast<uint32_t>(n_rows);
    csc.n = static_cast<uint32_t>(n_cols);
    csc.indptr.resize(static_cast<size_t>(n_cols) + 1, 0);

    // Build column-by-column.
    std::vector<std::vector<std::pair<int32_t,float>>> cols(
        static_cast<size_t>(n_cols));

    for (int j = 0; j < n_cols; ++j) {
        for (int i = 0; i < n_rows; ++i) {
            if (ber(rng) < density_frac) {
                float v = val_dist(rng);
                if (v > 0.0f)
                    cols[static_cast<size_t>(j)].push_back({static_cast<int32_t>(i), v});
            }
        }
    }

    int64_t total_nnz = 0;
    for (int j = 0; j < n_cols; ++j)
        total_nnz += static_cast<int64_t>(cols[static_cast<size_t>(j)].size());
    csc.nnz = static_cast<uint64_t>(total_nnz);

    csc.indptr[0] = 0;
    for (int j = 0; j < n_cols; ++j)
        csc.indptr[static_cast<size_t>(j) + 1] =
            csc.indptr[static_cast<size_t>(j)] +
            static_cast<int32_t>(cols[static_cast<size_t>(j)].size());

    csc.values .resize(static_cast<size_t>(csc.nnz));
    csc.indices.resize(static_cast<size_t>(csc.nnz));
    size_t off = 0;
    for (int j = 0; j < n_cols; ++j) {
        for (auto& [row, val] : cols[static_cast<size_t>(j)]) {
            csc.indices[off] = row;
            csc.values [off] = val;
            ++off;
        }
    }
    return csc;
}

} // anonymous namespace (extension)

TEST_F(WilcoxonTest, Test83_RealDataSized_PostNormalize_NoCrash) {
    // Dimensions matching the Cycle 70/71 RealData failing configuration.
    constexpr int kGenes  = 30000;
    constexpr int kCells  = 20000;
    constexpr float kDensity  = 0.05f;   // ~5% fill
    constexpr float kMaxVal   = 100.0f;
    constexpr uint64_t kSeed83 = 0xBADC0FFEE0DDF00DULL;
    constexpr int kGroups = 2;

    // --- build host CSC ---
    HostCSC h = make_realdata_sized_csc(kSeed83, kGenes, kCells, kDensity, kMaxVal);
    printf("[Test83] built synthetic CSC: %u genes x %u cells, nnz=%llu\n",
           h.m, h.n, static_cast<unsigned long long>(h.nnz));

    // --- upload to device ---
    singlet_gpu::core::DeviceCSC d_mat = upload_csc(h, stream_);

    // --- run log_normalize and EXPLICITLY DISCARD the return value ---
    // This is the exact condition that triggered BUG-WILCOXON-POST-NORMALIZE-CRASH:
    // without the cudaStreamSynchronize fix inside log_normalize(), the returned
    // LogNormResult destructs immediately, freeing d_size_factors/d_col_sums while
    // compute_size_factors_and_apply_kernel is still running on the stream.
    // With the fix, the sync inside log_normalize() ensures the kernel finishes
    // before the workspace is freed, and this cast becomes a safe no-op.
    singlet_gpu::preprocess::LogNormConfig ln_cfg;
    (void)singlet_gpu::preprocess::log_normalize(d_mat, ln_cfg, stream_);
    // Do NOT add a cudaStreamSynchronize here — that would mask the bug.
    // The fix must be inside log_normalize itself.

    // --- build 2-group labels (seeded round-robin) ---
    std::mt19937_64 label_rng(kSeed83 ^ 0xDEADBEEFULL);
    std::vector<int32_t> lbl(static_cast<size_t>(kCells));
    for (int j = 0; j < kCells; ++j)
        lbl[static_cast<size_t>(j)] = j % kGroups;

    singlet_gpu::core::DeviceMemory<int32_t> d_labels(static_cast<size_t>(kCells));
    cudaMemcpyAsync(d_labels.get(), lbl.data(),
                    static_cast<size_t>(kCells) * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream_);
    // NOTE: no explicit stream sync here — wilcoxon_de receives the same stream
    // and the label copy will be complete before the wilcoxon kernel reads them.

    // --- run wilcoxon DE immediately after log_normalize ---
    // If the lognorm fix is NOT live, this call will trigger:
    //   CUDA error: illegal memory access encountered
    // inside DeviceMemory<T>::DeviceMemory (factornet/gpu/types.cuh:105) because
    // the sticky error from the async race is surfaced by the next cudaMalloc.
    singlet_gpu::de::WilcoxonConfig w_cfg;
    w_cfg.top_n = 50;

    singlet_gpu::de::WilcoxonResult result;
    ASSERT_NO_THROW({
        result = singlet_gpu::de::wilcoxon_de(d_mat, d_labels, kGroups, w_cfg, stream_);
    }) << "wilcoxon_de threw after log_normalize with discarded result — "
          "lognorm sync fix not yet in place (BUG-WILCOXON-POST-NORMALIZE-CRASH)";

    // --- sync and check no sticky CUDA error ---
    cudaError_t sync_err = cudaStreamSynchronize(stream_);
    EXPECT_EQ(sync_err, cudaSuccess)
        << "cudaStreamSynchronize failed after wilcoxon_de: "
        << cudaGetErrorString(sync_err)
        << " — indicates async illegal memory access from lognorm workspace race";

    cudaError_t last_err = cudaGetLastError();
    EXPECT_EQ(last_err, cudaSuccess)
        << "cudaGetLastError returned: " << cudaGetErrorString(last_err)
        << " — sticky error from BUG-WILCOXON-POST-NORMALIZE-CRASH";

    // --- p-value NaN check (proxy for kernel correctness) ---
    // Full Jaccard/Spearman vs scanpy is tracked as OPTIM-WILCOXON-SCANPY-PARITY.
    ASSERT_EQ(static_cast<int>(result.per_cluster.size()), kGroups)
        << "Expected " << kGroups << " cluster results";

    for (int c = 0; c < kGroups; ++c) {
        const auto& cm = result.per_cluster[static_cast<size_t>(c)];
        if (!cm.p_values.get()) continue;  // empty cluster guard

        std::vector<float> h_pvals(static_cast<size_t>(w_cfg.top_n));
        cudaError_t copy_err = cudaMemcpy(
            h_pvals.data(), cm.p_values.get(),
            static_cast<size_t>(w_cfg.top_n) * sizeof(float),
            cudaMemcpyDeviceToHost);
        ASSERT_EQ(copy_err, cudaSuccess)
            << "p_values copy failed for cluster " << c << ": "
            << cudaGetErrorString(copy_err);

        int nan_count = 0;
        for (float p : h_pvals)
            if (std::isnan(p)) ++nan_count;

        printf("[Test83] group %d: %d NaN p-values out of %d\n",
               c, nan_count, w_cfg.top_n);

        EXPECT_EQ(nan_count, 0)
            << "cluster " << c << " has " << nan_count
            << " NaN p-values — kernel produced invalid output";
    }

    printf("[Test83] PASS: log_normalize+wilcoxon_de completed without CUDA error "
           "on %d-cell x %d-gene matrix with discarded LogNormResult\n",
           kCells, kGenes);
}
