// SPDX-License-Identifier: MIT
// singlet/gpu/tests/qc_doublet_correctness.cpp
//
// integrates: original (Scrublet-style synthetic doublet scoring on GPU)
//
// Correctness harness for singlet::gpu::qc::doublet_score.
//
// Written against the design doc at:
//   singlet/gpu/state/designs/31-doublet-detection.md
// NOT against the kernel source — authored in parallel with
// qc/doublet_score.h by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: scrublet Python via subprocess
//   tests/refs/doublet_scrublet_reference.py
//   (pip install scrublet)
//   Exit code 2 from the subprocess = scrublet unavailable → GTEST_SKIP.
//
// Tolerances (design doc §"Correctness test spec"):
//   Spearman ρ ≥ 0.95 vs scrublet      (Doublet_TinySynthetic_VsScrublet)
//   finite results + rate in [0, 20%]  (Doublet_GSM4037629_RealData)
//   ROC AUC ≥ 0.85                     (Doublet_AutoThreshold_ROC)
//   bit-identical with fixed seed      (Doublet_Determinism_BitIdentical)
//   scores correlate ≥ 0.9 across n_synth_frac settings
//                                      (Doublet_NSynth_Sensitivity)
//
// Test cases:
//   Doublet_TinySynthetic_VsScrublet
//   Doublet_GSM4037629_RealData
//   Doublet_AutoThreshold_ROC
//   Doublet_Determinism_BitIdentical
//   Doublet_NSynth_Sensitivity
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, CUDA::curand, singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every test that requires a device is guarded by
//   gpu_available() and calls GTEST_SKIP().

// ---- singlet-gpu doublet scoring header (written by gpu-kernel-dev) ----------
#include <singlet/gpu/qc/doublet_score.h>

// ---- other singlet-gpu headers used by the real-data test -------------------
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
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Compat bridges — adapt test API (designed against an earlier API) to the
// actual singlet::gpu::qc and singlet::gpu::io APIs.  All at global scope.
// =============================================================================

namespace singlet::gpu {
namespace io {

// io::PzDeviceLoader: simple class wrapping load_pz().
struct PzDeviceLoader {
    std::string path;
    explicit PzDeviceLoader(const std::string& p) : path(p) {}
    core::DeviceCSC load(cudaStream_t stream) const {
        return singlet::gpu::io::load_pz(path, stream).mat;
    }
};

} // namespace io

namespace qc {

// qc::DoubletConfig = qc::DoubletScoreConfig (alias).
using DoubletConfig = DoubletScoreConfig;

// qc::DoubletResult: test-expected struct with threshold field.
struct DoubletResult {
    float threshold = 0.5f;
    int   n_doublets = 0;
};

// Raw-pointer doublet_score bridge: wraps d_pca into a DeviceDense and calls
// the actual doublet_score(), then copies scores to d_scores_out.
inline void doublet_score(
    const float*           d_pca,     // device float[n_cells × n_pcs], row-major
    int                    n_cells,
    int                    n_pcs,
    float*                 d_scores_out,
    const DoubletScoreConfig& cfg,
    cudaStream_t           stream)
{
    // Wrap raw pointer into DeviceDense (no ownership transfer).
    // DeviceDense default ctor + set fields manually.
    core::DeviceDense embedding;
    embedding.rows = n_cells;
    embedding.cols = n_pcs;
    // DeviceMemory wrapping an existing raw pointer is not supported (RAII owns memory).
    // We copy into a new DeviceMemory to avoid ownership conflicts.
    embedding.data = core::DeviceMemory<float>((size_t)n_cells * n_pcs);
    cudaMemcpyAsync(embedding.data.get(), d_pca,
                    (size_t)n_cells * n_pcs * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    DoubletScoreResult result = ::singlet::gpu::qc::doublet_score(embedding, cfg, stream);
    cudaStreamSynchronize(stream);

    // Copy scores to caller-provided output buffer.
    cudaMemcpyAsync(d_scores_out, result.score.get(),
                    (size_t)n_cells * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
}

// doublet_score_with_result: same but also fills a DoubletResult.
inline void doublet_score_with_result(
    const float*           d_pca,
    int                    n_cells,
    int                    n_pcs,
    float*                 d_scores_out,
    DoubletResult&         out_result,
    const DoubletScoreConfig& cfg,
    cudaStream_t           stream)
{
    core::DeviceDense embedding;
    embedding.rows = n_cells;
    embedding.cols = n_pcs;
    embedding.data = core::DeviceMemory<float>((size_t)n_cells * n_pcs);
    cudaMemcpyAsync(embedding.data.get(), d_pca,
                    (size_t)n_cells * n_pcs * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    DoubletScoreResult result = ::singlet::gpu::qc::doublet_score(embedding, cfg, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpyAsync(d_scores_out, result.score.get(),
                    (size_t)n_cells * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    out_result.threshold  = result.threshold_used;
    out_result.n_doublets = result.n_predicted_doublets;
}

} // namespace qc
} // namespace singlet::gpu

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

// Tiny synthetic: 200 cells × 50 genes.  5% planted doublets = 10 cells.
constexpr int kTinyCells     = 200;
constexpr int kTinyGenes     = 50;
constexpr int kTinyK         = 20;   // kNN neighbors (design doc default)
constexpr double kTinyDoubletFrac = 0.05;   // 5% planted doublets

// ROC synthetic: 300 cells × 80 genes, 10% planted doublets.
constexpr int kRocCells      = 300;
constexpr int kRocGenes      = 80;
constexpr double kRocDoubletFrac = 0.10;

// n_synth sensitivity test: 300 cells × 60 genes.
constexpr int kSensCells     = 300;
constexpr int kSensGenes     = 60;

// Tolerance thresholds from the design doc.
constexpr double kSpearmanMin       = 0.95;  // vs scrublet
constexpr double kRocAucMin         = 0.85;  // synthetic planted doublets
constexpr double kNSynthCorrMin     = 0.90;  // cross n_synth_frac
constexpr double kMaxDoubletRate    = 0.20;  // real-data sanity
constexpr double kMinDoubletRate    = 0.00;

// n_synth_frac values for sensitivity test.
static const std::vector<double> kNSynthFracs = {0.10, 0.25, 0.50};

// Real-data path — GSM4037629 gene_counts.
static const char* kGsmCountsPath =
    "/mnt/projects/debruinz_project/singlet_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/gene_counts.1pz";

// Reference script path.
#ifndef SINGLET_GPU_TEST_DIR
#  define SINGLET_GPU_TEST_DIR "."
#endif
static const char* kRefScript =
    SINGLET_GPU_TEST_DIR "/refs/doublet_scrublet_reference.py";

// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_doublet_refs_tmp";

// Python interpreter.
static const char* kPython = "python3";

// Exit code from the reference script when scrublet is not installed.
static constexpr int kRefScriptSkipCode = 2;

// =============================================================================
// Utility helpers (pattern from cycles 3-30)
// =============================================================================

namespace {

void ensure_refs_tmp() {
    fs::create_directories(kRefsTmpDir);
}

std::string refs_path(const std::string& name) {
    return std::string(kRefsTmpDir) + "/" + name;
}

// run_cmd_rc: returns raw exit code without throwing.
int run_cmd_rc(const std::string& cmd) {
    return std::system(cmd.c_str());
}

bool gpu_available() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    return (e == cudaSuccess && count > 0);
}

// ---------------------------------------------------------------------------
// Host-side dense PCA/embedding container (n_cells × n_pcs, row-major).
// ---------------------------------------------------------------------------
struct HostDense {
    uint32_t             n_rows;   // cells
    uint32_t             n_cols;   // PCs (dims)
    std::vector<float>   data;     // row-major: data[i * n_cols + j]
};

// ---------------------------------------------------------------------------
// Host-side CSC container (genes × cells).
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
// Write HostCSC to dump_csc.h binary format ("CSCC" magic).
// ---------------------------------------------------------------------------
void write_csc_bin(const HostCSC& h, const std::string& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    static constexpr uint32_t kMagic = 0x43535343u;
    f.write(reinterpret_cast<const char*>(&kMagic),  4);
    f.write(reinterpret_cast<const char*>(&h.m),     4);
    f.write(reinterpret_cast<const char*>(&h.n),     4);
    f.write(reinterpret_cast<const char*>(&h.nnz),   8);
    f.write(reinterpret_cast<const char*>(h.values .data()),
            static_cast<std::streamsize>(h.nnz * sizeof(float)));
    f.write(reinterpret_cast<const char*>(h.indptr .data()),
            static_cast<std::streamsize>((static_cast<size_t>(h.n) + 1) * sizeof(int32_t)));
    f.write(reinterpret_cast<const char*>(h.indices.data()),
            static_cast<std::streamsize>(h.nnz * sizeof(int32_t)));
    if (!f) throw std::runtime_error("Write error: " + path);
}

// ---------------------------------------------------------------------------
// Read doublet_scores float array from reference .npz.
// Minimal NPZ: only handles the 'doublet_scores' key as a raw npy entry.
// NPZ = ZIP of .npy files named "doublet_scores.npy", etc.
// We shell out to python3 to extract rather than parsing ZIP/npy by hand.
// ---------------------------------------------------------------------------
std::vector<double> read_npz_doublet_scores(const std::string& npz_path,
                                            int n_cells) {
    // Write a tiny Python extractor script.
    std::string extract_script = refs_path("_extract_scores.py");
    std::string scores_txt     = refs_path("_extract_scores.txt");
    {
        std::ofstream s(extract_script);
        s << "import numpy as np, sys\n";
        s << "d = np.load('" << npz_path << "', allow_pickle=False)\n";
        s << "scores = d['doublet_scores'].astype(float)\n";
        s << "np.savetxt('" << scores_txt << "', scores, delimiter='\\n')\n";
    }
    int rc = run_cmd_rc(std::string(kPython) + " " + extract_script +
                        " 2>/dev/null");
    if (rc != 0)
        throw std::runtime_error("Failed to extract doublet_scores from NPZ: " + npz_path);

    std::vector<double> scores;
    std::ifstream in(scores_txt);
    double v;
    while (in >> v) scores.push_back(v);
    if (static_cast<int>(scores.size()) != n_cells)
        throw std::runtime_error("Expected " + std::to_string(n_cells) +
                                 " scores, got " + std::to_string(scores.size()));
    return scores;
}

// ---------------------------------------------------------------------------
// Read the PCA embedding (n_cells × n_pcs, float32) exported by the scrublet
// reference script into a HostDense.  Falls back to an empty HostDense (n_cols=0)
// if the 'pca_embedding' key is absent (older reference output).
// ---------------------------------------------------------------------------
HostDense read_npz_pca_embedding(const std::string& npz_path, int n_cells) {
    std::string extract_script = refs_path("_extract_pca.py");
    std::string pca_bin        = refs_path("_extract_pca.bin");
    {
        std::ofstream s(extract_script);
        s << "import numpy as np, sys, struct\n";
        s << "d = np.load('" << npz_path << "', allow_pickle=False)\n";
        s << "if 'pca_embedding' not in d:\n";
        s << "    open('" << pca_bin << "', 'wb').write(struct.pack('<II', 0, 0))\n";
        s << "    sys.exit(0)\n";
        s << "pca = d['pca_embedding'].astype(np.float32)\n";
        s << "nr, nc = pca.shape\n";
        s << "with open('" << pca_bin << "', 'wb') as f:\n";
        s << "    f.write(struct.pack('<II', nr, nc))\n";
        s << "    f.write(pca.tobytes())\n";
    }
    int rc = run_cmd_rc(std::string(kPython) + " " + extract_script + " 2>/dev/null");

    HostDense out;
    out.n_rows = 0; out.n_cols = 0;
    if (rc != 0) return out;

    std::ifstream f(pca_bin, std::ios::binary);
    if (!f) return out;
    uint32_t nr = 0, nc = 0;
    f.read(reinterpret_cast<char*>(&nr), 4);
    f.read(reinterpret_cast<char*>(&nc), 4);
    if (nr == 0 || nc == 0) return out;
    if (static_cast<int>(nr) != n_cells) return out;

    out.n_rows = nr;
    out.n_cols = nc;
    out.data.resize(static_cast<size_t>(nr) * nc);
    f.read(reinterpret_cast<char*>(out.data.data()),
           static_cast<std::streamsize>(static_cast<size_t>(nr) * nc * sizeof(float)));
    if (!f) { out.n_rows = 0; out.n_cols = 0; out.data.clear(); }
    return out;
}

// ---------------------------------------------------------------------------
// Spearman rank correlation between two equal-length vectors.
// ---------------------------------------------------------------------------
double spearman(const std::vector<double>& x, const std::vector<double>& y) {
    assert(x.size() == y.size());
    int n = static_cast<int>(x.size());
    if (n < 2) return 0.0;

    auto rank_vec = [&](const std::vector<double>& v) {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b){ return v[a] < v[b]; });
        std::vector<double> ranks(n);
        for (int i = 0; i < n; ) {
            int j = i;
            while (j < n && v[idx[j]] == v[idx[i]]) ++j;
            double avg = (i + j - 1) * 0.5;
            for (int k = i; k < j; ++k) ranks[idx[k]] = avg;
            i = j;
        }
        return ranks;
    };
    auto rx = rank_vec(x);
    auto ry = rank_vec(y);
    double mx = 0, my = 0;
    for (int i = 0; i < n; ++i) { mx += rx[i]; my += ry[i]; }
    mx /= n; my /= n;
    double num = 0, dx2 = 0, dy2 = 0;
    for (int i = 0; i < n; ++i) {
        double dx = rx[i] - mx, dy = ry[i] - my;
        num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
    }
    if (dx2 < 1e-15 || dy2 < 1e-15) return 0.0;
    return num / std::sqrt(dx2 * dy2);
}

// ---------------------------------------------------------------------------
// Pearson correlation.
// ---------------------------------------------------------------------------
double pearson(const std::vector<double>& x, const std::vector<double>& y) {
    assert(x.size() == y.size());
    int n = static_cast<int>(x.size());
    if (n < 2) return 0.0;
    double mx = 0, my = 0;
    for (int i = 0; i < n; ++i) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    double num = 0, dx2 = 0, dy2 = 0;
    for (int i = 0; i < n; ++i) {
        double dx = x[i] - mx, dy = y[i] - my;
        num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
    }
    if (dx2 < 1e-15 || dy2 < 1e-15) return 0.0;
    return num / std::sqrt(dx2 * dy2);
}

// ---------------------------------------------------------------------------
// ROC AUC (trapezoidal rule).
// labels: 1 = doublet, 0 = singlet.
// ---------------------------------------------------------------------------
double roc_auc(const std::vector<double>& scores,
               const std::vector<int>& labels) {
    assert(scores.size() == labels.size());
    int n = static_cast<int>(scores.size());
    int n_pos = 0, n_neg = 0;
    for (int l : labels) {
        if (l == 1) ++n_pos;
        else        ++n_neg;
    }
    if (n_pos == 0 || n_neg == 0) return 0.5;

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b){ return scores[a] > scores[b]; });

    double auc = 0.0;
    int tp = 0, fp = 0;
    int prev_fp = 0;
    double prev_tpr = 0.0;
    for (int i : idx) {
        if (labels[i] == 1) {
            ++tp;
        } else {
            ++fp;
            double tpr = static_cast<double>(tp) / n_pos;
            double fpr_step = 1.0 / n_neg;
            auc += (tpr + prev_tpr) * 0.5 * fpr_step;
            prev_tpr = tpr;
        }
    }
    return auc;
}

// ---------------------------------------------------------------------------
// Generate tiny synthetic Poisson count data (genes × cells, CSC).
// ---------------------------------------------------------------------------
HostCSC make_synthetic_counts(int m, int n, uint64_t seed,
                               float mean_count = 3.0f) {
    std::mt19937_64 rng(seed);
    std::exponential_distribution<float> mean_dist(1.0f / mean_count);

    std::vector<float>   vals;
    std::vector<int32_t> indptr(static_cast<size_t>(n) + 1, 0);
    std::vector<int32_t> idx;

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            float lam = mean_dist(rng);
            std::poisson_distribution<int> pd(static_cast<double>(lam));
            int cnt = pd(rng);
            if (cnt > 0) {
                vals.push_back(static_cast<float>(cnt));
                idx.push_back(i);
            }
        }
        indptr[static_cast<size_t>(j) + 1] =
            static_cast<int32_t>(static_cast<int>(vals.size()));
    }

    HostCSC h;
    h.m = static_cast<uint32_t>(m);
    h.n = static_cast<uint32_t>(n);
    h.nnz = vals.size();
    h.values  = std::move(vals);
    h.indptr  = std::move(indptr);
    h.indices = std::move(idx);
    return h;
}

// ---------------------------------------------------------------------------
// Plant doublets into a count matrix (in-place).
// For each planted doublet cell, overwrite it with the average of two random
// real singlet cells' expression vectors.
// Returns the set of doublet indices (into the n_cells columns).
// ---------------------------------------------------------------------------
std::vector<int> plant_doublets(HostCSC& h, double doublet_frac, uint64_t seed) {
    int n_cells   = static_cast<int>(h.n);
    int n_doublets = static_cast<int>(std::round(n_cells * doublet_frac));
    n_doublets = std::max(1, std::min(n_doublets, n_cells / 2));

    std::mt19937_64 rng(seed ^ 0xDEADBEEFull);
    std::uniform_int_distribution<int> cell_dist(0, n_cells - 1);

    std::vector<int> doublet_cols;
    doublet_cols.reserve(n_doublets);

    // Choose the last n_doublets columns as doublets (reproducible).
    for (int d = n_cells - n_doublets; d < n_cells; ++d)
        doublet_cols.push_back(d);

    int m = static_cast<int>(h.m);

    for (int d : doublet_cols) {
        // Pick two random singlet cells (from the first n_cells - n_doublets).
        int a = cell_dist(rng) % (n_cells - n_doublets);
        int b = cell_dist(rng) % (n_cells - n_doublets);

        // Dense column extraction.
        std::vector<float> col_a(m, 0.0f), col_b(m, 0.0f);
        for (int32_t k = h.indptr[a]; k < h.indptr[a + 1]; ++k)
            col_a[h.indices[k]] = h.values[k];
        for (int32_t k = h.indptr[b]; k < h.indptr[b + 1]; ++k)
            col_b[h.indices[k]] = h.values[k];

        // Average and rewrite column d in the CSC.
        // First, zero out current column d.
        h.indptr[d + 1] = h.indptr[d];  // clear entries (simplified — rebuild below)
        // For correctness in this test helper we rebuild from scratch.
        // NOTE: this modifies h.indptr only for column d; a full rebuild is
        //       required after all plantings. We do a full-matrix rebuild below.
        (void)col_a; (void)col_b;
    }

    // Full CSC rebuild from dense representation (simpler for planted data).
    // Build dense matrix: h.m × h.n.
    std::vector<std::vector<float>> dense(m, std::vector<float>(n_cells, 0.0f));
    for (int j = 0; j < n_cells; ++j)
        for (int32_t k = h.indptr[j]; k < h.indptr[j + 1]; ++k)
            dense[h.indices[k]][j] = h.values[k];

    // Overwrite doublet columns with averaged pairs.
    std::mt19937_64 rng2(seed ^ 0xDEADBEEFull);
    std::uniform_int_distribution<int> cell_dist2(0, n_cells - n_doublets - 1);
    for (int d : doublet_cols) {
        int a = cell_dist2(rng2);
        int b = cell_dist2(rng2);
        for (int i = 0; i < m; ++i)
            dense[i][d] = (dense[i][a] + dense[i][b]) * 0.5f;
    }

    // Re-compress to CSC.
    std::vector<float>   vals;
    std::vector<int32_t> indptr(static_cast<size_t>(n_cells) + 1, 0);
    std::vector<int32_t> idx;
    for (int j = 0; j < n_cells; ++j) {
        for (int i = 0; i < m; ++i) {
            if (dense[i][j] > 0.0f) {
                vals.push_back(dense[i][j]);
                idx.push_back(i);
            }
        }
        indptr[j + 1] = static_cast<int32_t>(vals.size());
    }
    h.values  = std::move(vals);
    h.indptr  = std::move(indptr);
    h.indices = std::move(idx);
    h.nnz     = static_cast<uint64_t>(h.values.size());

    return doublet_cols;
}

// ---------------------------------------------------------------------------
// Upload HostDense to device (row-major float array).
// Returns device pointer managed by DeviceMemory<float>.
// ---------------------------------------------------------------------------
singlet::gpu::core::DeviceMemory<float>
upload_dense(const HostDense& h, cudaStream_t stream) {
    singlet::gpu::core::DeviceMemory<float> d_data(
        static_cast<size_t>(h.n_rows) * h.n_cols);
    cudaError_t e = cudaMemcpyAsync(
        d_data.get(), h.data.data(),
        static_cast<size_t>(h.n_rows) * h.n_cols * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("upload_dense: ") + cudaGetErrorString(e));
    cudaStreamSynchronize(stream);
    return d_data;
}

// ---------------------------------------------------------------------------
// Download a DeviceMemory<float> to host vector.
// ---------------------------------------------------------------------------
std::vector<float> download_f32(
        const singlet::gpu::core::DeviceMemory<float>& d,
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
// Simple PCA stub: row-normalize + random projection to n_pcs.
// Used to produce a PCA-like embedding for the doublet scorer input.
// (Production code uses singlet::gpu::reduce::pca; this test fixture avoids
//  cross-cycle dependencies and keeps the harness self-contained.)
// ---------------------------------------------------------------------------
HostDense simple_pca_projection(const HostCSC& csc, int n_pcs, uint64_t seed) {
    int n_cells = static_cast<int>(csc.n);
    int n_genes = static_cast<int>(csc.m);
    int k = std::min(n_pcs, n_genes);

    // Random projection matrix: genes × n_pcs, Gaussian N(0, 1/sqrt(n_genes)).
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f / std::sqrt(static_cast<float>(n_genes)));
    std::vector<float> proj(static_cast<size_t>(n_genes) * k);
    for (auto& v : proj) v = nd(rng);

    HostDense out;
    out.n_rows = static_cast<uint32_t>(n_cells);
    out.n_cols = static_cast<uint32_t>(k);
    out.data.assign(static_cast<size_t>(n_cells) * k, 0.0f);

    // Sparse matrix-vector multiply: (n_cells × n_genes) × (n_genes × k).
    // CSC is genes × cells; iterate over columns (cells).
    for (int j = 0; j < n_cells; ++j) {
        float* row = &out.data[static_cast<size_t>(j) * k];
        for (int32_t p = csc.indptr[j]; p < csc.indptr[j + 1]; ++p) {
            int gene = csc.indices[p];
            float val = csc.values[p];
            const float* proj_row = &proj[static_cast<size_t>(gene) * k];
            for (int q = 0; q < k; ++q)
                row[q] += val * proj_row[q];
        }
    }
    return out;
}

}  // anonymous namespace

// =============================================================================
// DoubletFixture: stream + handles, created once per test.
// =============================================================================

class DoubletFixture : public ::testing::Test {
protected:
    cudaStream_t stream_{nullptr};

    void SetUp() override {
        if (!gpu_available()) return;
        cudaError_t e = cudaStreamCreate(&stream_);
        if (e != cudaSuccess)
            FAIL() << "cudaStreamCreate failed: " << cudaGetErrorString(e);
    }

    void TearDown() override {
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    }
};

// =============================================================================
// Test 1: Doublet_TinySynthetic_VsScrublet
//
// 200-cell synthetic with planted 5% doublets (10 cells).
// Run singlet::gpu::qc::doublet_score on a PCA projection of the counts.
// Run scrublet via subprocess on the raw counts.
// Compare per-cell doublet scores: Spearman ρ ≥ 0.95.
// =============================================================================

TEST_F(DoubletFixture, Doublet_TinySynthetic_VsScrublet) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";
    ensure_refs_tmp();

    // Generate synthetic counts and plant doublets.
    HostCSC counts = make_synthetic_counts(kTinyGenes, kTinyCells, kSeed);
    std::vector<int> doublet_cols = plant_doublets(counts, kTinyDoubletFrac, kSeed + 1);

    // Write counts to binary for the Python reference.
    std::string counts_bin = refs_path("tiny_synth_counts.bin");
    std::string ref_npz    = refs_path("tiny_synth_scrublet.npz");
    write_csc_bin(counts, counts_bin);

    // Run scrublet reference.
    std::ostringstream cmd;
    cmd << kPython << " " << kRefScript
        << " --counts " << counts_bin
        << " --out "    << ref_npz
        << " --n-synth-frac 0.25"
        << " --k " << kTinyK
        << " --seed 0"
        << " 2>&1";
    int rc = run_cmd_rc(cmd.str());
    if (rc == kRefScriptSkipCode)
        GTEST_SKIP() << "scrublet not installed — skipping Spearman reference diff";
    ASSERT_EQ(rc, 0) << "doublet_scrublet_reference.py failed (exit " << rc << ")";

    // Read scrublet scores.
    std::vector<double> ref_scores = read_npz_doublet_scores(ref_npz, kTinyCells);

    // Run singlet-gpu doublet scorer on the SAME PCA embedding that scrublet used.
    // Using a different embedding (e.g. random projection) produces Spearman ≈ 0.24
    // even when the kernel is algorithmically correct, because the kNN graphs diverge.
    // The reference script exports scrublet's internal PCA as 'pca_embedding' in the npz.
    HostDense pca = read_npz_pca_embedding(ref_npz, kTinyCells);
    if (pca.n_cols == 0) {
        // Fallback if the reference script used an older version without pca_embedding.
        pca = simple_pca_projection(counts, /*n_pcs=*/10, kSeed + 2);
    }
    singlet::gpu::core::DeviceMemory<float> d_pca = upload_dense(pca, stream_);

    singlet::gpu::qc::DoubletConfig cfg;
    cfg.n_synth_frac = 0.25f;
    cfg.k            = kTinyK;
    cfg.seed         = static_cast<uint64_t>(kSeed);

    singlet::gpu::core::DeviceMemory<float> d_scores(kTinyCells);
    singlet::gpu::qc::doublet_score(
        d_pca.get(),
        static_cast<int>(pca.n_rows),
        static_cast<int>(pca.n_cols),
        d_scores.get(),
        cfg,
        stream_);
    cudaStreamSynchronize(stream_);

    std::vector<float> gpu_scores_f32 = download_f32(d_scores, kTinyCells, stream_);
    std::vector<double> gpu_scores(gpu_scores_f32.begin(), gpu_scores_f32.end());

    // All scores must be finite and in [0, 1].
    for (int i = 0; i < kTinyCells; ++i) {
        ASSERT_TRUE(std::isfinite(gpu_scores[i]))
            << "non-finite score at cell " << i;
        EXPECT_GE(gpu_scores[i], 0.0);
        EXPECT_LE(gpu_scores[i], 1.0);
    }

    double rho = spearman(gpu_scores, ref_scores);
    std::printf("[Doublet_TinySynthetic_VsScrublet] Spearman ρ = %.4f  (threshold ≥ %.2f)\n",
                rho, kSpearmanMin);
    EXPECT_GE(rho, kSpearmanMin)
        << "Spearman rho " << rho << " < " << kSpearmanMin;
}

// =============================================================================
// Test 2: Doublet_GSM4037629_RealData
//
// Load gene_counts.1pz for GSM4037629 (11,560 cells) via pz_device_loader.
// Run PCA (via simple projection stub) → doublet_score.
// Confirm: all scores finite, doublet rate in [0%, 20%].
// =============================================================================

TEST_F(DoubletFixture, Doublet_GSM4037629_RealData) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";
    if (!fs::exists(kGsmCountsPath))
        GTEST_SKIP() << "GSM4037629 gene_counts.1pz not found: " << kGsmCountsPath;

    // Load via pz_device_loader.
    singlet::gpu::io::PzDeviceLoader loader(kGsmCountsPath);
    auto device_csc = loader.load(stream_);   // genes × cells, device CSC

    uint32_t n_cells = static_cast<uint32_t>(device_csc.cols);
    uint32_t n_genes = static_cast<uint32_t>(device_csc.rows);

    // Copy to host for the PCA stub.
    // (Production would call singlet::gpu::reduce::pca directly on the device.)
    // Access device CSC internals for host copy.
    HostCSC h;
    h.m   = n_genes;
    h.n   = n_cells;
    h.nnz = device_csc.nnz;
    h.values .resize(static_cast<size_t>(h.nnz));
    h.indptr .resize(static_cast<size_t>(n_cells) + 1);
    h.indices.resize(static_cast<size_t>(h.nnz));
    cudaMemcpy(h.values .data(), device_csc.values.get(),
               static_cast<size_t>(h.nnz) * sizeof(float),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h.indptr .data(), device_csc.col_ptr.get(),
               (static_cast<size_t>(n_cells) + 1) * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h.indices.data(), device_csc.row_indices.get(),
               static_cast<size_t>(h.nnz) * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // PCA projection: 50 PCs.
    HostDense pca = simple_pca_projection(h, /*n_pcs=*/50, kSeed);
    singlet::gpu::core::DeviceMemory<float> d_pca = upload_dense(pca, stream_);

    singlet::gpu::qc::DoubletConfig cfg;
    cfg.n_synth_frac = 0.25f;
    cfg.k            = 20;
    cfg.seed         = kSeed;

    singlet::gpu::core::DeviceMemory<float> d_scores(n_cells);
    singlet::gpu::qc::doublet_score(
        d_pca.get(),
        static_cast<int>(n_cells),
        static_cast<int>(pca.n_cols),
        d_scores.get(),
        cfg,
        stream_);
    cudaStreamSynchronize(stream_);

    std::vector<float> scores_f32 = download_f32(d_scores, n_cells, stream_);

    // Sanity checks.
    int n_nonfinite = 0;
    int n_outside   = 0;
    for (float s : scores_f32) {
        if (!std::isfinite(s)) ++n_nonfinite;
        if (s < 0.0f || s > 1.0f) ++n_outside;
    }
    EXPECT_EQ(n_nonfinite, 0) << n_nonfinite << " non-finite scores";
    EXPECT_EQ(n_outside,   0) << n_outside   << " scores outside [0,1]";

    // Doublet rate check.
    double mean_score = 0.0;
    for (float s : scores_f32) mean_score += static_cast<double>(s);
    mean_score /= static_cast<double>(n_cells);

    // Auto-threshold doublet rate.
    // Use the default threshold (>0.5 as a rough proxy if auto-threshold
    // unavailable — doublet_score populates DoubletResult.threshold).
    singlet::gpu::qc::DoubletResult result;
    singlet::gpu::qc::doublet_score_with_result(
        d_pca.get(),
        static_cast<int>(n_cells),
        static_cast<int>(pca.n_cols),
        d_scores.get(),
        result,
        cfg,
        stream_);
    cudaStreamSynchronize(stream_);

    double threshold = static_cast<double>(result.threshold);
    int n_doublets = 0;
    for (float s : scores_f32)
        if (static_cast<double>(s) >= threshold) ++n_doublets;
    double doublet_rate = static_cast<double>(n_doublets) /
                          static_cast<double>(n_cells);

    std::printf("[Doublet_GSM4037629_RealData] n_cells=%u  mean_score=%.4f  "
                "threshold=%.4f  doublet_rate=%.3f\n",
                n_cells, mean_score, threshold, doublet_rate);

    EXPECT_GE(doublet_rate, kMinDoubletRate);
    EXPECT_LE(doublet_rate, kMaxDoubletRate)
        << "doublet_rate " << doublet_rate << " > " << kMaxDoubletRate;
}

// =============================================================================
// Test 3: Doublet_AutoThreshold_ROC
//
// 300-cell synthetic with planted 10% doublets.
// Run doublet_score with auto-threshold.
// Compute ROC AUC of returned scores vs planted ground truth labels.
// Require AUC ≥ 0.85.
// =============================================================================

TEST_F(DoubletFixture, Doublet_AutoThreshold_ROC) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    HostCSC counts = make_synthetic_counts(kRocGenes, kRocCells, kSeed + 10);
    std::vector<int> doublet_cols = plant_doublets(counts, kRocDoubletFrac, kSeed + 11);

    // Ground-truth labels: 1 = doublet, 0 = singlet.
    std::vector<int> labels(kRocCells, 0);
    for (int d : doublet_cols) labels[d] = 1;

    HostDense pca = simple_pca_projection(counts, /*n_pcs=*/10, kSeed + 12);
    singlet::gpu::core::DeviceMemory<float> d_pca = upload_dense(pca, stream_);

    singlet::gpu::qc::DoubletConfig cfg;
    cfg.n_synth_frac = 0.25f;
    cfg.k            = 20;
    cfg.seed         = kSeed + 10;

    singlet::gpu::core::DeviceMemory<float> d_scores(kRocCells);
    singlet::gpu::qc::DoubletResult result;
    singlet::gpu::qc::doublet_score_with_result(
        d_pca.get(),
        kRocCells,
        static_cast<int>(pca.n_cols),
        d_scores.get(),
        result,
        cfg,
        stream_);
    cudaStreamSynchronize(stream_);

    std::vector<float>  scores_f32 = download_f32(d_scores, kRocCells, stream_);
    std::vector<double> scores(scores_f32.begin(), scores_f32.end());

    double auc = roc_auc(scores, labels);
    std::printf("[Doublet_AutoThreshold_ROC] ROC AUC = %.4f  "
                "(threshold ≥ %.2f)  n_planted=%d\n",
                auc, kRocAucMin, static_cast<int>(doublet_cols.size()));
    EXPECT_GE(auc, kRocAucMin)
        << "ROC AUC " << auc << " < " << kRocAucMin;
}

// =============================================================================
// Test 4: Doublet_Determinism_BitIdentical
//
// Run doublet_score twice with the same fixed seed on the same input.
// Confirm the two score vectors are bit-identical.
// =============================================================================

TEST_F(DoubletFixture, Doublet_Determinism_BitIdentical) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    HostCSC counts = make_synthetic_counts(kTinyGenes, kTinyCells, kSeed + 20);
    HostDense pca = simple_pca_projection(counts, /*n_pcs=*/10, kSeed + 21);
    singlet::gpu::core::DeviceMemory<float> d_pca = upload_dense(pca, stream_);

    singlet::gpu::qc::DoubletConfig cfg;
    cfg.n_synth_frac = 0.25f;
    cfg.k            = kTinyK;
    cfg.seed         = kSeed;

    // First run.
    singlet::gpu::core::DeviceMemory<float> d_scores_a(kTinyCells);
    singlet::gpu::qc::doublet_score(
        d_pca.get(), kTinyCells, static_cast<int>(pca.n_cols),
        d_scores_a.get(), cfg, stream_);
    cudaStreamSynchronize(stream_);
    std::vector<float> scores_a = download_f32(d_scores_a, kTinyCells, stream_);

    // Second run (same cfg, same seed).
    singlet::gpu::core::DeviceMemory<float> d_scores_b(kTinyCells);
    singlet::gpu::qc::doublet_score(
        d_pca.get(), kTinyCells, static_cast<int>(pca.n_cols),
        d_scores_b.get(), cfg, stream_);
    cudaStreamSynchronize(stream_);
    std::vector<float> scores_b = download_f32(d_scores_b, kTinyCells, stream_);

    ASSERT_EQ(scores_a.size(), scores_b.size());
    int n_differ = 0;
    for (int i = 0; i < kTinyCells; ++i) {
        // Bit-identical: compare uint32 representation.
        uint32_t ua, ub;
        std::memcpy(&ua, &scores_a[i], 4);
        std::memcpy(&ub, &scores_b[i], 4);
        if (ua != ub) ++n_differ;
    }
    std::printf("[Doublet_Determinism_BitIdentical] n_differ=%d / %d\n",
                n_differ, kTinyCells);
    EXPECT_EQ(n_differ, 0)
        << n_differ << " cells differ between two identical-seed runs";
}

// =============================================================================
// Test 5: Doublet_NSynth_Sensitivity
//
// Vary n_synth_frac ∈ {0.10, 0.25, 0.50} on the same input.
// Require Pearson correlation ≥ 0.90 between any pair of score vectors.
// =============================================================================

TEST_F(DoubletFixture, Doublet_NSynth_Sensitivity) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    HostCSC counts = make_synthetic_counts(kSensGenes, kSensCells, kSeed + 30);
    HostDense pca = simple_pca_projection(counts, /*n_pcs=*/15, kSeed + 31);
    singlet::gpu::core::DeviceMemory<float> d_pca = upload_dense(pca, stream_);

    std::vector<std::vector<float>> all_scores;

    for (double frac : kNSynthFracs) {
        singlet::gpu::qc::DoubletConfig cfg;
        cfg.n_synth_frac = static_cast<float>(frac);
        cfg.k            = 20;
        cfg.seed         = kSeed;

        singlet::gpu::core::DeviceMemory<float> d_scores(kSensCells);
        singlet::gpu::qc::doublet_score(
            d_pca.get(), kSensCells, static_cast<int>(pca.n_cols),
            d_scores.get(), cfg, stream_);
        cudaStreamSynchronize(stream_);
        all_scores.push_back(download_f32(d_scores, kSensCells, stream_));
        std::printf("[Doublet_NSynth_Sensitivity] n_synth_frac=%.2f  "
                    "mean_score=%.4f\n",
                    frac,
                    std::accumulate(all_scores.back().begin(),
                                    all_scores.back().end(), 0.0f) /
                    static_cast<float>(kSensCells));
    }

    // Compare all pairs.
    int n_fracs = static_cast<int>(kNSynthFracs.size());
    for (int a = 0; a < n_fracs; ++a) {
        for (int b = a + 1; b < n_fracs; ++b) {
            std::vector<double> sa(all_scores[a].begin(), all_scores[a].end());
            std::vector<double> sb(all_scores[b].begin(), all_scores[b].end());
            double r = pearson(sa, sb);
            std::printf("[Doublet_NSynth_Sensitivity] Pearson r(frac=%.2f, frac=%.2f) = %.4f "
                        "(threshold ≥ %.2f)\n",
                        kNSynthFracs[a], kNSynthFracs[b], r, kNSynthCorrMin);
            EXPECT_GE(r, kNSynthCorrMin)
                << "Pearson r(" << kNSynthFracs[a] << ", " << kNSynthFracs[b]
                << ") = " << r << " < " << kNSynthCorrMin;
        }
    }
}
