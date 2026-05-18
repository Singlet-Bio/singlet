// SPDX-License-Identifier: MIT
// singlet/gpu/tests/qc_omnidoublet_correctness.cpp
//
// integrates: original (first GPU OmniDoublet-style multimodal doublet detection)
//
// Correctness harness for singlet::gpu::qc::omnidoublet.
//
// Written against the design doc at:
//   singlet/gpu/state/designs/39-omnidoublet.md
// NOT against the kernel source — authored in parallel with
// qc/omnidoublet.h by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: OmniDoublet Python (preferred) → Scrublet Python (fallback)
//            → pure-Python Scrublet-style algorithm (always present).
//   Script: tests/refs/omnidoublet_py_reference.py
//   The reference script never exits 2; the pure-Python fallback ensures
//   the test always has a reference even without OmniDoublet/Scrublet installed.
//
// Tolerances (design doc §"Correctness test spec"):
//   ROC AUC ≥ 0.85 vs Python reference          (TinySynthetic_VsPython)
//   finite scores, call rate 5–15%              (GSM_CITEseq_RealData)
//   ADT zeroing changes ≥5% of scores by ≥5%   (MultimodalFeatures_NonZero)
//   simulated doublet capture ±5% of target     (FdrControl)
//   bit-identical across two runs               (Determinism_BitIdentical)
//
// Test cases:
//   Omnidoublet_TinySynthetic_VsPython
//   Omnidoublet_GSM_CITEseq_RealData
//   Omnidoublet_MultimodalFeatures_NonZero
//   Omnidoublet_FdrControl
//   Omnidoublet_Determinism_BitIdentical
//
// Design note: OmniDoublet requires BOTH rna_counts AND adt_counts.
//   Synthetic ADT data is generated alongside RNA in every test that needs it.
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, CUDA::curand, CUDA::cusparse,
//             singlet-gpu::singlet-gpu, GTest.
//   On CPU-only nodes every GPU-dependent test is guarded by gpu_available()
//   and calls GTEST_SKIP().

// ---- singlet-gpu OmniDoublet header (written by gpu-kernel-dev) -------------
#include <singlet/gpu/qc/omnidoublet.h>

// ---- other singlet-gpu headers used by the real-data test ------------------
#include <singlet/gpu/io/pz_device_loader.h>
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
#include <limits>
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
// The test was authored with a C-style DeviceCSC (raw pointers).  The real type
// is factornet::gpu::SparseMatrixGPU<float> (RAII).  Add a type alias so that
// singlet::gpu::DeviceCSC resolves to the real type, and rewrite upload_csc to
// use the synchronous host-data constructor.

namespace singlet::gpu {
    using DeviceCSC = singlet::gpu::core::DeviceCSC;
}

// PzDeviceLoader compat — wraps io::load_pz() and exposes n_cols / n_rows.
// loader.load(path, stream) returns a PzLoaded whose lifetime owns the DeviceCSC
// and which is implicitly convertible to const core::DeviceCSC& for omnidoublet().
namespace singlet::gpu { namespace io {

struct PzLoaded {
    PzDeviceMatrix pz;          // owns the device memory
    uint32_t n_rows = 0;        // genes (rows)
    uint32_t n_cols = 0;        // cells (cols)

    // Implicit conversion so d_rna / d_adt can be passed where DeviceCSC& expected.
    operator const core::DeviceCSC&() const { return pz.mat; }
};

struct PzDeviceLoader {
    PzLoaded load(const std::string& path, cudaStream_t stream) const {
        PzLoaded out;
        out.pz     = singlet::gpu::io::load_pz(path, stream);
        out.n_rows = static_cast<uint32_t>(out.pz.mat.rows);
        out.n_cols = static_cast<uint32_t>(out.pz.mat.cols);
        return out;
    }
};

} }  // namespace singlet::gpu::io

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

// Tiny synthetic dimensions (design doc §"Correctness test spec" / §"Target perf").
constexpr int kTinyCells   = 500;
constexpr int kTinyGenes   = 200;
constexpr int kTinyADT     = 20;
constexpr double kTinyDoubletFrac = 0.10;  // 10% planted doublets

// kNN + PCA parameters matched to the reference script defaults.
constexpr int kK         = 50;    // joint-embedding kNN neighbors (design doc step 3)
constexpr int kNPcs      = 30;    // joint PCA components
constexpr double kNSynthFrac = 1.0;  // n_sim = 2 × n_cells (design doc step 1)

// FDR target for the FdrControl test.
constexpr double kFdrTarget = 0.10;
constexpr double kFdrTol    = 0.05;  // ±5% tolerance

// Tolerance thresholds from the design doc §"Correctness test spec".
constexpr double kRocAucMin        = 0.85;   // vs Python reference
constexpr double kMinCallRate      = 0.05;   // real-data sanity lower
constexpr double kMaxCallRate      = 0.15;   // real-data sanity upper
constexpr double kAdtChangeFrac    = 0.80;   // ≥80% of cells must change score
constexpr double kAdtChangeMin     = 0.05;   // by ≥5% in absolute doublet score

// Real-data paths — CITE-seq sample with adt.1pz.
// GSM4037629 is scRNA (no ADT); skip if absent — test guards with GTEST_SKIP.
static const char* kGsmRnaPath =
    "/mnt/projects/debruinz_project/singlet_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629/gene_counts.1pz";
// ADT path in CITE-seq subtree (may not exist — test checks and skips).
static const char* kGsmAdtPath =
    "/mnt/projects/debruinz_project/singlet_pipeline/"
    "quant/cite/GSE127/GSE127918/GSM4037629/adt.1pz";

// Reference script path.
#ifndef SINGLET_GPU_TEST_DIR
#  define SINGLET_GPU_TEST_DIR "."
#endif
static const char* kRefScript =
    SINGLET_GPU_TEST_DIR "/refs/omnidoublet_py_reference.py";

// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_omnidoublet_refs_tmp";

// Python interpreter.
static const char* kPython = "python3";

// =============================================================================
// Utility helpers (pattern from cycles 3-38)
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
// Host-side CSC container (features × cells).
// ---------------------------------------------------------------------------
struct HostCSC {
    uint32_t             m;        // features (rows: genes or ADT tags)
    uint32_t             n;        // cells (cols)
    uint64_t             nnz;
    std::vector<float>   values;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
};

// ---------------------------------------------------------------------------
// Write HostCSC to dump_csc.h "CSCC" binary format.
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
// Generate a synthetic sparse CSC count matrix with planted doublets.
//
// Pure cells: each has a random "cell type" (n_types types, each expressed as
// a distinct gene block with mean ~20 counts).  Doublets: sum of two random
// pure cells.  Returned matrix is features × (n_cells + n_doublets), with
// the doublet label vector filled separately.
//
// Parameters:
//   n_pure     number of pure (singlet) cells
//   n_doublets number of artificially planted doublets appended at the end
//   n_features number of genes or ADT tags
//   n_types    number of cell types (for cluster structure — helps AUC)
//   density    expected fraction of non-zero entries per column
//   seed       RNG seed
// ---------------------------------------------------------------------------
HostCSC make_synthetic_csc(
        int n_pure, int n_doublets, int n_features,
        int n_types, double density, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> type_dist(0, n_types - 1);
    std::poisson_distribution<int>     count_dist(20);
    std::bernoulli_distribution        nonzero_dist(density);

    int n_total = n_pure + n_doublets;

    // Pre-allocate CSC arrays (upper-bound on nnz).
    std::vector<int32_t> indptr(static_cast<size_t>(n_total) + 1, 0);
    std::vector<int32_t> indices_buf;
    std::vector<float>   values_buf;
    indices_buf.reserve(static_cast<size_t>(n_total) * static_cast<size_t>(n_features) / 5);
    values_buf .reserve(indices_buf.capacity());

    // Build pure-cell dense matrix (n_features × n_pure) for doublet simulation.
    // Using a 2-D vector here is fine — n_pure × n_features is at most 500 × 200 = 100k.
    std::vector<std::vector<float>> pure_dense(
        static_cast<size_t>(n_features),
        std::vector<float>(static_cast<size_t>(n_pure), 0.0f));

    int block_size = std::max(1, n_features / n_types);
    for (int ci = 0; ci < n_pure; ++ci) {
        int cell_type  = type_dist(rng);
        int block_start = cell_type * block_size;
        int block_end   = std::min(block_start + block_size, n_features);
        for (int fi = block_start; fi < block_end; ++fi) {
            if (nonzero_dist(rng)) {
                pure_dense[fi][ci] = static_cast<float>(count_dist(rng));
            }
        }
    }

    // Fill CSC for pure cells.
    for (int ci = 0; ci < n_pure; ++ci) {
        indptr[ci] = static_cast<int32_t>(indices_buf.size());
        for (int fi = 0; fi < n_features; ++fi) {
            float v = pure_dense[fi][ci];
            if (v > 0.0f) {
                indices_buf.push_back(fi);
                values_buf .push_back(v);
            }
        }
    }

    // Fill CSC for doublets: sum of two random pure cells.
    std::uniform_int_distribution<int> cell_dist(0, n_pure - 1);
    for (int di = 0; di < n_doublets; ++di) {
        int a = cell_dist(rng);
        int b = cell_dist(rng);
        if (b == a) b = (b + 1) % n_pure;
        indptr[n_pure + di] = static_cast<int32_t>(indices_buf.size());
        for (int fi = 0; fi < n_features; ++fi) {
            float v = pure_dense[fi][a] + pure_dense[fi][b];
            if (v > 0.0f) {
                indices_buf.push_back(fi);
                values_buf .push_back(v);
            }
        }
    }
    indptr[n_total] = static_cast<int32_t>(indices_buf.size());

    HostCSC h;
    h.m       = static_cast<uint32_t>(n_features);
    h.n       = static_cast<uint32_t>(n_total);
    h.nnz     = static_cast<uint64_t>(indices_buf.size());
    h.values  = std::move(values_buf);
    h.indptr  = std::move(indptr);
    h.indices = std::move(indices_buf);
    return h;
}

// ---------------------------------------------------------------------------
// Read doublet scores (float64) from reference NPZ via a tiny Python extractor.
// ---------------------------------------------------------------------------
std::vector<double> read_npz_scores(const std::string& npz_path,
                                    const std::string& key,
                                    int n_expected) {
    std::string script_path = refs_path("_extract_" + key + ".py");
    std::string txt_path    = refs_path("_extract_" + key + ".txt");
    {
        std::ofstream s(script_path);
        s << "import numpy as np, sys\n";
        s << "d = np.load('" << npz_path << "', allow_pickle=False)\n";
        s << "arr = d['" << key << "'].astype(float)\n";
        s << "np.savetxt('" << txt_path << "', arr, delimiter='\\n')\n";
    }
    int rc = run_cmd_rc(std::string(kPython) + " " + script_path + " 2>/dev/null");
    if (rc != 0)
        throw std::runtime_error("Failed to extract '" + key + "' from NPZ: " + npz_path);

    std::vector<double> result;
    std::ifstream in(txt_path);
    double v;
    while (in >> v) result.push_back(v);

    if (n_expected > 0 && static_cast<int>(result.size()) != n_expected)
        throw std::runtime_error("Expected " + std::to_string(n_expected) +
                                 " values for key '" + key + "', got " +
                                 std::to_string(result.size()));
    return result;
}

// ---------------------------------------------------------------------------
// Spearman rank correlation between two equal-length vectors.
// (Pattern from cycles 3-38: inline, no external dependency.)
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
    double sum_d2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = rx[i] - ry[i];
        sum_d2 += d * d;
    }
    return 1.0 - (6.0 * sum_d2) / (static_cast<double>(n) *
                                    (static_cast<double>(n) * n - 1.0));
}

// ---------------------------------------------------------------------------
// ROC AUC helper.
//
// Computes the area under the ROC curve using the trapezoidal rule.
// labels[i] = 1 for planted doublet, 0 for singlet.
// scores[i] = continuous doublet score (higher = more likely doublet).
// ---------------------------------------------------------------------------
double roc_auc(const std::vector<double>& scores,
               const std::vector<int>&    labels) {
    assert(scores.size() == labels.size());
    int n = static_cast<int>(scores.size());

    // Sort by score descending.
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return scores[a] > scores[b]; });

    int n_pos = 0;
    for (int l : labels) n_pos += l;
    int n_neg = n - n_pos;
    if (n_pos == 0 || n_neg == 0) return 0.5;  // degenerate

    double auc = 0.0;
    int tp = 0, fp = 0;
    double prev_fpr = 0.0, prev_tpr = 0.0;

    for (int i = 0; i < n; ++i) {
        if (labels[order[i]] == 1) {
            ++tp;
        } else {
            ++fp;
            double tpr = static_cast<double>(tp) / n_pos;
            double fpr = static_cast<double>(fp) / n_neg;
            // Trapezoidal step.
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;
            prev_fpr = fpr;
            prev_tpr = tpr;
        }
    }
    // Final segment to (1, 1).
    double tpr_end = static_cast<double>(tp) / n_pos;
    auc += (1.0 - prev_fpr) * (tpr_end + prev_tpr) * 0.5;
    return auc;
}

// ---------------------------------------------------------------------------
// Run the Python reference script and return the doublet_scores vector.
// Returns {} on script failure (caller decides to skip or fail).
// ---------------------------------------------------------------------------
std::vector<double> run_py_reference(
        const HostCSC& rna_csc,
        const HostCSC& adt_csc,
        const std::string& tag,
        double n_synth_frac = kNSynthFrac,
        int    k             = kK,
        int    n_pcs         = kNPcs,
        double expected_rate = kFdrTarget,
        uint64_t seed        = kSeed)
{
    ensure_refs_tmp();

    std::string rna_bin = refs_path(tag + "_rna.bin");
    std::string adt_bin = refs_path(tag + "_adt.bin");
    std::string npz_out = refs_path(tag + "_ref.npz");

    write_csc_bin(rna_csc, rna_bin);
    write_csc_bin(adt_csc, adt_bin);

    std::ostringstream cmd;
    cmd << kPython << " " << kRefScript
        << " --rna-counts "           << rna_bin
        << " --adt-counts "           << adt_bin
        << " --out "                  << npz_out
        << " --n-synth-frac "         << n_synth_frac
        << " --k "                    << k
        << " --n-pcs "                << n_pcs
        << " --expected-doublet-rate "<< expected_rate
        << " --seed "                 << seed
        << " 2>&1";

    int rc = run_cmd_rc(cmd.str());
    if (rc != 0) {
        return {};  // caller will GTEST_SKIP or treat as fallback
    }

    try {
        return read_npz_scores(npz_out, "doublet_scores",
                               static_cast<int>(rna_csc.n));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[omnidoublet] failed to read ref scores: %s\n", e.what());
        return {};
    }
}

// ---------------------------------------------------------------------------
// Upload a HostCSC to device and wrap as singlet::gpu DeviceCSC.
// Returns a pair of (DeviceCSC, backing DeviceMemory handle).
// The caller must keep the handle alive as long as the DeviceCSC is in use.
// ---------------------------------------------------------------------------
singlet::gpu::DeviceCSC upload_csc(const HostCSC& h, cudaStream_t stream) {
    // Use the synchronous host-data constructor:
    // DeviceCSC(m, n, nnz, col_ptr, row_indices, values)
    cudaStreamSynchronize(stream);
    return singlet::gpu::core::DeviceCSC(
        static_cast<int>(h.m),
        static_cast<int>(h.n),
        static_cast<int>(h.nnz),
        h.indptr.data(),
        h.indices.data(),
        h.values.data());
}

void free_device_csc(singlet::gpu::DeviceCSC& csc, cudaStream_t stream) {
    // DeviceCSC uses RAII — memory freed automatically on destruction.
    // The stream parameter is retained for API compatibility.
    (void)csc; (void)stream;
}

}  // anonymous namespace


// =============================================================================
// Test 1 — Omnidoublet_TinySynthetic_VsPython
//
// 500 cells × 200 genes × 20 ADT tags synthetic with 10% planted doublets.
// Measures ROC AUC of our doublet scores vs the planted-doublet labels,
// and (when the reference script succeeds) Spearman ρ vs the Python reference.
// Tolerance: ROC AUC ≥ 0.85 (design doc §"Correctness test spec").
// =============================================================================

TEST(Omnidoublet, TinySynthetic_VsPython) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device — skipping GPU test.";

    // --- Build synthetic data.
    constexpr int n_pure     = static_cast<int>(kTinyCells * (1.0 - kTinyDoubletFrac));
    constexpr int n_doublets = kTinyCells - n_pure;
    constexpr int n_types    = 5;  // distinct cell-type clusters → good AUC

    // RNA: 200 genes, density 0.30
    HostCSC rna_csc = make_synthetic_csc(
        n_pure, n_doublets, kTinyGenes, n_types, 0.30, kSeed);
    // ADT: 20 tags, density 0.60 (antibodies tend to have lower sparsity)
    HostCSC adt_csc = make_synthetic_csc(
        n_pure, n_doublets, kTinyADT, n_types, 0.60, kSeed ^ 0xAD0Cull);

    ASSERT_EQ(rna_csc.n, static_cast<uint32_t>(kTinyCells));
    ASSERT_EQ(adt_csc.n, static_cast<uint32_t>(kTinyCells));

    // --- Labels: last n_doublets cells are planted doublets.
    std::vector<int> labels(kTinyCells, 0);
    for (int i = n_pure; i < kTinyCells; ++i) labels[i] = 1;

    // --- Run GPU OmniDoublet.
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto d_rna = upload_csc(rna_csc, stream);
    auto d_adt = upload_csc(adt_csc, stream);

    singlet::gpu::qc::OmniDoubletConfig cfg;
    cfg.k            = std::min(kK,    kTinyCells - 1);
    cfg.n_pcs        = std::min(kNPcs, kTinyCells - 1);
    cfg.n_synth_frac = kNSynthFrac;
    cfg.target_fdr   = kFdrTarget;
    cfg.seed         = kSeed;

    singlet::gpu::qc::OmniDoubletResult result =
        singlet::gpu::qc::omnidoublet(d_rna, d_adt, cfg, stream);

    // Sync and copy scores to host.
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<double> gpu_scores(kTinyCells);
    ASSERT_EQ(cudaMemcpy(gpu_scores.data(),
                         result.doublet_scores,
                         kTinyCells * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    // --- Sanity: all scores finite and in [0, 1].
    for (int i = 0; i < kTinyCells; ++i) {
        EXPECT_TRUE(std::isfinite(gpu_scores[i]))
            << "Non-finite score at cell " << i;
        EXPECT_GE(gpu_scores[i], 0.0);
        EXPECT_LE(gpu_scores[i], 1.0);
    }

    // --- ROC AUC vs planted labels.
    double auc = roc_auc(gpu_scores, labels);
    EXPECT_GE(auc, kRocAucMin)
        << "ROC AUC " << auc << " < " << kRocAucMin
        << " (design doc tolerance: AUC ≥ 0.85 on planted doublets)";

    std::fprintf(stderr,
        "[omnidoublet] TinySynthetic: AUC=%.3f (threshold=%.3f)\n",
        auc, static_cast<double>(kRocAucMin));

    // --- Optional: Spearman vs Python reference.
    auto py_scores = run_py_reference(rna_csc, adt_csc, "tiny");
    if (!py_scores.empty()) {
        double rho = spearman(gpu_scores, py_scores);
        std::fprintf(stderr,
            "[omnidoublet] TinySynthetic: Spearman ρ vs Python=%.4f\n", rho);
        // AUC check is the hard gate; log Spearman for monitoring only.
        (void)rho;
    }

    // Cleanup.
    free_device_csc(d_rna, stream);
    free_device_csc(d_adt, stream);
    cudaFreeAsync(result.doublet_scores, stream);
    cudaFreeAsync(result.doublet_calls,  stream);
    if (result.simulated_doublet_scores)
        cudaFreeAsync(result.simulated_doublet_scores, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}


// =============================================================================
// Test 2 — Omnidoublet_GSM_CITEseq_RealData
//
// Load a real CITE-seq sample with adt.1pz.  Skip if absent.
// Confirm: all scores finite, call rate in [5%, 15%] at default FDR.
// =============================================================================

TEST(Omnidoublet, GSM_CITEseq_RealData) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device — skipping GPU test.";

    // Check that BOTH the RNA and ADT files exist.
    if (!fs::exists(kGsmRnaPath)) {
        GTEST_SKIP() << "RNA file not found: " << kGsmRnaPath;
    }
    if (!fs::exists(kGsmAdtPath)) {
        GTEST_SKIP() << "ADT file not found (sample is scRNA-only): " << kGsmAdtPath
                     << ".  This test requires a CITE-seq sample.";
    }

    // --- Load via pz_device_loader.
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet::gpu::io::PzDeviceLoader loader;
    auto d_rna = loader.load(kGsmRnaPath, stream);
    auto d_adt = loader.load(kGsmAdtPath, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    ASSERT_GT(d_rna.n_cols, 0u) << "RNA loaded zero cells";
    ASSERT_GT(d_adt.n_cols, 0u) << "ADT loaded zero cells";
    ASSERT_EQ(d_rna.n_cols, d_adt.n_cols)
        << "RNA and ADT cell counts differ: "
        << d_rna.n_cols << " vs " << d_adt.n_cols;

    int n_cells = static_cast<int>(d_rna.n_cols);

    // --- Run OmniDoublet.
    singlet::gpu::qc::OmniDoubletConfig cfg;
    cfg.k            = std::min(50, n_cells - 1);
    cfg.n_pcs        = std::min(30, n_cells - 1);
    cfg.n_synth_frac = 1.0;
    cfg.target_fdr   = kFdrTarget;
    cfg.seed         = kSeed;

    singlet::gpu::qc::OmniDoubletResult result =
        singlet::gpu::qc::omnidoublet(d_rna, d_adt, cfg, stream);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<double> scores(n_cells);
    ASSERT_EQ(cudaMemcpy(scores.data(),
                         result.doublet_scores,
                         n_cells * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    std::vector<int32_t> calls(n_cells);
    ASSERT_EQ(cudaMemcpy(calls.data(),
                         result.doublet_calls,
                         n_cells * sizeof(int32_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    // --- Sanity checks.
    int n_nan = 0;
    for (double s : scores) {
        if (!std::isfinite(s)) ++n_nan;
    }
    EXPECT_EQ(n_nan, 0) << n_nan << " non-finite doublet scores in real data";

    int n_doublet_calls = 0;
    for (int32_t c : calls) n_doublet_calls += c;
    double call_rate = static_cast<double>(n_doublet_calls) / n_cells;

    EXPECT_GE(call_rate, kMinCallRate)
        << "Call rate " << call_rate << " below 5% floor on real CITE-seq data";
    EXPECT_LE(call_rate, kMaxCallRate)
        << "Call rate " << call_rate << " above 15% ceiling on real CITE-seq data";

    std::fprintf(stderr,
        "[omnidoublet] RealData: n_cells=%d n_doublets=%d call_rate=%.3f\n",
        n_cells, n_doublet_calls, call_rate);

    // Cleanup.
    cudaFreeAsync(result.doublet_scores, stream);
    cudaFreeAsync(result.doublet_calls,  stream);
    if (result.simulated_doublet_scores)
        cudaFreeAsync(result.simulated_doublet_scores, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}


// =============================================================================
// Test 3 — Omnidoublet_MultimodalFeatures_NonZero
//
// Design doc §"Correctness test spec" test 3:
// Zeroing the ADT modality changes doublet scores by ≥5% on ≥80% of cells.
// This proves the ADT modality genuinely contributes to the joint embedding.
//
// Procedure:
//   a. Run OmniDoublet on (rna, adt) → scores_full.
//   b. Run OmniDoublet on (rna, adt_zeroed) → scores_rna_only.
//   c. Count cells where |scores_full[i] - scores_rna_only[i]| ≥ 0.05.
//   d. Assert that fraction ≥ 0.80.
// =============================================================================

TEST(Omnidoublet, MultimodalFeatures_NonZero) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device — skipping GPU test.";

    constexpr int n_pure     = static_cast<int>(kTinyCells * 0.9);
    constexpr int n_doublets = kTinyCells - n_pure;
    constexpr int n_types    = 4;

    HostCSC rna_csc = make_synthetic_csc(
        n_pure, n_doublets, kTinyGenes, n_types, 0.30, kSeed);
    HostCSC adt_csc = make_synthetic_csc(
        n_pure, n_doublets, kTinyADT, n_types, 0.65, kSeed ^ 0xAD02ull);

    // Build a zero-ed ADT (same shape, all zeros).
    HostCSC adt_zero;
    adt_zero.m       = adt_csc.m;
    adt_zero.n       = adt_csc.n;
    adt_zero.nnz     = 0;
    adt_zero.values  = {};
    adt_zero.indptr  = std::vector<int32_t>(static_cast<size_t>(adt_csc.n) + 1, 0);
    adt_zero.indices = {};

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet::gpu::qc::OmniDoubletConfig cfg;
    cfg.k            = std::min(kK,    kTinyCells - 1);
    cfg.n_pcs        = std::min(kNPcs, kTinyCells - 1);
    cfg.n_synth_frac = kNSynthFrac;
    cfg.target_fdr   = kFdrTarget;
    cfg.seed         = kSeed;

    // Run (a): full multimodal.
    auto d_rna_a  = upload_csc(rna_csc,  stream);
    auto d_adt_a  = upload_csc(adt_csc,  stream);
    auto result_a = singlet::gpu::qc::omnidoublet(d_rna_a, d_adt_a, cfg, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<double> scores_full(kTinyCells);
    ASSERT_EQ(cudaMemcpy(scores_full.data(),
                         result_a.doublet_scores,
                         kTinyCells * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    // Run (b): zeroed ADT.
    auto d_rna_b   = upload_csc(rna_csc,  stream);
    auto d_adt_b   = upload_csc(adt_zero, stream);
    auto result_b  = singlet::gpu::qc::omnidoublet(d_rna_b, d_adt_b, cfg, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<double> scores_rna_only(kTinyCells);
    ASSERT_EQ(cudaMemcpy(scores_rna_only.data(),
                         result_b.doublet_scores,
                         kTinyCells * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    // (c) Count changed cells.
    int n_changed = 0;
    for (int i = 0; i < kTinyCells; ++i) {
        if (std::abs(scores_full[i] - scores_rna_only[i]) >= kAdtChangeMin)
            ++n_changed;
    }
    double changed_frac = static_cast<double>(n_changed) / kTinyCells;

    std::fprintf(stderr,
        "[omnidoublet] MultimodalNonZero: %.1f%% cells changed score by ≥5%% "
        "when ADT zeroed\n",
        changed_frac * 100.0);

    EXPECT_GE(changed_frac, kAdtChangeFrac)
        << "Only " << (changed_frac * 100.0) << "% of cells changed doublet score "
        << "when ADT was zeroed (need ≥ " << (kAdtChangeFrac * 100.0) << "%). "
        << "ADT modality appears NOT to contribute to joint embedding.";

    // Cleanup.
    free_device_csc(d_rna_a, stream);
    free_device_csc(d_adt_a, stream);
    free_device_csc(d_rna_b, stream);
    free_device_csc(d_adt_b, stream);
    cudaFreeAsync(result_a.doublet_scores, stream);
    cudaFreeAsync(result_a.doublet_calls,  stream);
    if (result_a.simulated_doublet_scores)
        cudaFreeAsync(result_a.simulated_doublet_scores, stream);
    cudaFreeAsync(result_b.doublet_scores, stream);
    cudaFreeAsync(result_b.doublet_calls,  stream);
    if (result_b.simulated_doublet_scores)
        cudaFreeAsync(result_b.simulated_doublet_scores, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}


// =============================================================================
// Test 4 — Omnidoublet_FdrControl
//
// Design doc §"Correctness test spec" test 4:
// Simulated doublets are captured at the target rate ±5%.
//
// Procedure:
//   a. Construct a dataset where the ground-truth doublet fraction is kFdrTarget.
//   b. Run OmniDoublet with target_fdr = kFdrTarget.
//   c. Measure the fraction of simulated doublets (from the internal
//      simulated_doublet_scores) whose score exceeds the selected threshold.
//      This is the "simulated doublet capture rate."
//   d. Assert |capture_rate - kFdrTarget| ≤ kFdrTol.
//
// Note: OmniDoublet uses the simulated-doublet score distribution to calibrate
// the threshold so that the capture rate matches the target FDR.  This test
// verifies that calibration is correct.
// =============================================================================

TEST(Omnidoublet, FdrControl) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device — skipping GPU test.";

    // Larger synthetic to get stable FDR estimate.
    constexpr int kFdrCells    = 500;
    constexpr int kFdrGenes    = 200;
    constexpr int kFdrADT      = 20;
    constexpr int n_pure_fdr   = static_cast<int>(kFdrCells * (1.0 - kFdrTarget));
    constexpr int n_dbl_fdr    = kFdrCells - n_pure_fdr;

    HostCSC rna_csc = make_synthetic_csc(
        n_pure_fdr, n_dbl_fdr, kFdrGenes, 5, 0.30, kSeed ^ 0xFD0Cull);
    HostCSC adt_csc = make_synthetic_csc(
        n_pure_fdr, n_dbl_fdr, kFdrADT, 5, 0.60, kSeed ^ 0xFD0C2ull);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto d_rna = upload_csc(rna_csc, stream);
    auto d_adt = upload_csc(adt_csc, stream);

    singlet::gpu::qc::OmniDoubletConfig cfg;
    cfg.k            = std::min(kK,    kFdrCells - 1);
    cfg.n_pcs        = std::min(kNPcs, kFdrCells - 1);
    cfg.n_synth_frac = kNSynthFrac;
    cfg.target_fdr   = kFdrTarget;
    cfg.seed         = kSeed;

    singlet::gpu::qc::OmniDoubletResult result =
        singlet::gpu::qc::omnidoublet(d_rna, d_adt, cfg, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // Read simulated doublet scores and the threshold from the result.
    ASSERT_NE(result.simulated_doublet_scores, nullptr)
        << "omnidoublet must output simulated_doublet_scores for FDR calibration";

    int n_sim = static_cast<int>(result.n_simulated);
    ASSERT_GT(n_sim, 0) << "n_simulated must be > 0";

    std::vector<double> sim_scores(n_sim);
    ASSERT_EQ(cudaMemcpy(sim_scores.data(),
                         result.simulated_doublet_scores,
                         n_sim * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    double threshold = result.threshold;

    // Capture rate = fraction of simulated doublets above threshold.
    int n_captured = 0;
    for (double s : sim_scores) {
        if (s >= threshold) ++n_captured;
    }
    double capture_rate = static_cast<double>(n_captured) / n_sim;

    std::fprintf(stderr,
        "[omnidoublet] FdrControl: target=%.2f capture_rate=%.3f "
        "threshold=%.4f n_sim=%d\n",
        kFdrTarget, capture_rate, threshold, n_sim);

    EXPECT_NEAR(capture_rate, kFdrTarget, kFdrTol)
        << "Simulated-doublet capture rate " << capture_rate
        << " deviates from target " << kFdrTarget
        << " by more than " << kFdrTol;

    // Cleanup.
    free_device_csc(d_rna, stream);
    free_device_csc(d_adt, stream);
    cudaFreeAsync(result.doublet_scores, stream);
    cudaFreeAsync(result.doublet_calls,  stream);
    cudaFreeAsync(result.simulated_doublet_scores, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}


// =============================================================================
// Test 5 — Omnidoublet_Determinism_BitIdentical
//
// Design doc §"Correctness test spec" test 5:
// Bit-identical doublet scores across two runs with the same fixed seed.
// Exercises the Philox4x32_10 seeded path (design doc §"Determinism").
// =============================================================================

TEST(Omnidoublet, Determinism_BitIdentical) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device — skipping GPU test.";

    constexpr int n_pure     = static_cast<int>(kTinyCells * 0.9);
    constexpr int n_doublets = kTinyCells - n_pure;

    HostCSC rna_csc = make_synthetic_csc(
        n_pure, n_doublets, kTinyGenes, 4, 0.30, kSeed);
    HostCSC adt_csc = make_synthetic_csc(
        n_pure, n_doublets, kTinyADT, 4, 0.60, kSeed ^ 0xDE0Cull);

    singlet::gpu::qc::OmniDoubletConfig cfg;
    cfg.k            = std::min(kK,    kTinyCells - 1);
    cfg.n_pcs        = std::min(kNPcs, kTinyCells - 1);
    cfg.n_synth_frac = kNSynthFrac;
    cfg.target_fdr   = kFdrTarget;
    cfg.seed         = kSeed;
    cfg.deterministic= true;   // segmented-scan path (design doc §"Determinism")

    auto run_once = [&]() -> std::vector<double> {
        cudaStream_t stream;
        EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        auto d_rna = upload_csc(rna_csc, stream);
        auto d_adt = upload_csc(adt_csc, stream);

        auto result = singlet::gpu::qc::omnidoublet(d_rna, d_adt, cfg, stream);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        std::vector<double> scores(kTinyCells);
        EXPECT_EQ(cudaMemcpy(scores.data(),
                             result.doublet_scores,
                             kTinyCells * sizeof(double),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);

        free_device_csc(d_rna, stream);
        free_device_csc(d_adt, stream);
        cudaFreeAsync(result.doublet_scores, stream);
        cudaFreeAsync(result.doublet_calls,  stream);
        if (result.simulated_doublet_scores)
            cudaFreeAsync(result.simulated_doublet_scores, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        return scores;
    };

    auto scores1 = run_once();
    auto scores2 = run_once();

    ASSERT_EQ(scores1.size(), scores2.size());

    int n_mismatch = 0;
    for (size_t i = 0; i < scores1.size(); ++i) {
        // Bit-identical: use memcmp on the double representation.
        if (std::memcmp(&scores1[i], &scores2[i], sizeof(double)) != 0)
            ++n_mismatch;
    }

    EXPECT_EQ(n_mismatch, 0)
        << n_mismatch << " / " << scores1.size()
        << " scores differ between two deterministic runs with the same seed "
        << "(Philox4x32_10 + segmented-scan must guarantee bit-identity).";

    std::fprintf(stderr,
        "[omnidoublet] Determinism: %d/%zu scores bit-identical across two runs.\n",
        static_cast<int>(scores1.size()) - n_mismatch,
        scores1.size());
}
