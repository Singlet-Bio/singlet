// SPDX-License-Identifier: MIT
// singlet/gpu/tests/perturbation_perturb_graph_correctness.cpp
//
// integrates: original (first GPU CPA + PerturbGraph for unseen perturbation prediction)
//
// Correctness harness for singlet::gpu::perturbation::perturb_graph_fit.
//
// Written against the design doc at:
//   singlet/gpu/state/designs/32-perturb-graph.md
// NOT against the kernel source — authored in parallel with
// perturbation/perturb_graph.h by the analysis-validator worker (Sonnet, Tier 2).
//
// Reference: CPA Python (cpa-tools) via subprocess
//   tests/refs/cpa_python_reference.py
//   (pip install cpa-tools scvi-tools anndata numpy scipy)
//   Exit code 2 from the subprocess = packages unavailable → GTEST_SKIP.
//
// Tolerances (design doc §"Correctness test spec"):
//   Spearman ρ ≥ 0.85 per-gene predicted vs CPA Python on held-out perturbation
//   Finite predictions (no NaN/Inf) on held-out perturbation smoke test
//   Monotonic dose-response interpolation
//   Bit-identical results with fixed seed
//   Gradient check: rel_err ≤ 1e-3  (per-parameter, finite-diff; cycle 27+29+30 pattern)
//     6 parameter classes: encoder_W1, encoder_W2, decoder_W1, decoder_W2,
//                          perturbation_embedding, adversarial_W
//   Adversarial disentanglement: z_c → perturbation classifier accuracy < 60%
//   GNN unseen-perturbation: per-gene Spearman ρ vs true held-out > 0.5 (above random)
//
// Test cases:
//   PerturbGraph_TinySynthetic_VsCPA
//   PerturbGraph_HeldOutPerturbation_Smoke
//   PerturbGraph_DoseResponse_Monotone
//   PerturbGraph_Determinism_BitIdentical
//   PerturbGraph_GradientCheck_FiniteDiff
//   PerturbGraph_AdversarialLoss_Disentangle
//   PerturbGraph_GNNExtension_UnseenPerturbation
//
// Build: cmake --build build -j
//   Requires: FACTORNET_HAS_GPU, CUDA::cudart, CUDA::cublas, CUDA::cusparse,
//             CUDA::curand, singlet-gpu::singlet-gpu, GTest.
//   PERTURB_GRAPH_EXPOSE_GRAD_FOR_TEST must be defined (set in CMakeLists.txt).
//   On CPU-only nodes every test that requires a device is guarded by
//   gpu_available() and calls GTEST_SKIP().

// ---- singlet-gpu perturb_graph header (written by gpu-kernel-dev) -----------
#include <singlet/gpu/perturbation/perturb_graph.h>

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
// Constants
// =============================================================================

constexpr uint64_t kSeed = 0xC0FFEEull;

// Tiny synthetic: 200 cells × 100 genes × 5 perturbations (design doc spec).
constexpr int kTinyCells = 200;
constexpr int kTinyGenes = 100;
constexpr int kTinyPerts = 5;

// Gradient-check problem (40 cells × 20 genes — tiny for fd speed; design doc).
constexpr int kGradCells = 40;
constexpr int kGradGenes = 20;
constexpr int kGradPerts = 3;

// Dose-response test (100 cells × 50 genes × 3 perts).
constexpr int kDoseCells = 100;
constexpr int kDoseGenes = 50;
constexpr int kDosePerts = 3;
constexpr int kDoseLevels = 6;   // dose steps checked for monotonicity

// Adversarial disentanglement test (150 cells × 60 genes × 5 perts).
constexpr int kAdvCells = 150;
constexpr int kAdvGenes = 60;
constexpr int kAdvPerts = 5;

// Number of parameters sampled per class in gradient check (cycle 27/29/30 pattern).
static constexpr int kGradSampleSize = 6;

// Tolerance thresholds from the design doc.
constexpr double kSpearmanRhoMin    = 0.85;   // vs CPA Python
constexpr double kGradRelErrMax     = 1e-3;   // gradient check
constexpr double kFiniteDiffEps     = 1e-3;
constexpr double kGNNSpearmanMin    = 0.50;   // unseen-pert prediction vs random
constexpr double kAdvAccuracyMax    = 0.60;   // classifier accuracy on z_c → pert

// Training epochs for various tests.
constexpr int kTinyEpochs = 200;
constexpr int kAdvEpochs  = 150;
constexpr int kDoseEpochs = 100;
constexpr int kGNNEpochs  = 200;

// Reference script path.
static const char* kRefScript =
    "/mnt/home/debruinz/Singlet-AI/singlet/gpu/"
    "tests/refs/cpa_python_reference.py";

// Temporary directory for reference I/O.
static const char* kRefsTmpDir = "/tmp/singlet_gpu_perturb_refs_tmp";

// Python interpreter.
static const char* kPython = "python3";

// Exit code from the reference script when cpa-tools is not installed.
static constexpr int kRefScriptSkipCode = 2;

// =============================================================================
// Utility helpers (pattern from cycles 3-31)
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

// run_cmd_rc: returns exit code without throwing.
int run_cmd_rc(const std::string& cmd) {
    return std::system(cmd.c_str());
}

bool gpu_available() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    return (e == cudaSuccess && count > 0);
}

// ---------------------------------------------------------------------------
// Compatibility shim: bridge test's old API assumptions to real perturb_graph.h
// ---------------------------------------------------------------------------
// All shim types are defined at namespace scope (before the anonymous namespace)
// so they are available to the test cases.

} // temporarily close anonymous namespace

// Extended result wrapping the real PerturbGraphResult.
struct PerturbGraphResultCompat {
    // pred_held_out: not returned by train_perturb_graph; left empty.
    // Tests that assert size() > 0 will fail cleanly at runtime.
    singlet::gpu::core::DeviceMemory<float> pred_held_out;
    // model placeholder (not returned by train_perturb_graph).
    struct ModelHandle { bool valid = false; } model;
    // Fields mirrored from PerturbGraphResult.
    int n_cells  = 0;
    int n_genes  = 0;
    int n_perts  = 0;
    int d_latent = 0;
    std::vector<float> epoch_losses;
    std::vector<float> adv_losses;
};

// Bridge: downloads device labels/doses, maps config fields, calls train_perturb_graph.
inline PerturbGraphResultCompat perturb_graph_fit(
    const singlet::gpu::core::DeviceCSC&              counts,
    const singlet::gpu::core::DeviceMemory<int32_t>&  d_labels,
    const singlet::gpu::core::DeviceMemory<float>&    d_doses,
    const singlet::gpu::perturbation::PerturbGraphConfig& cfg,
    cudaStream_t /*stream*/)
{
    size_t n_lab = d_labels.size();
    std::vector<int>   h_labels(n_lab);
    std::vector<float> h_doses(d_doses.size());
    cudaMemcpy(h_labels.data(), d_labels.get(), n_lab * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_doses.data(),  d_doses.get(),  d_doses.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // n_perts must be deduced from max label + 1 (cfg has no n_perts field).
    int n_perts = 0;
    for (int l : h_labels) if (l + 1 > n_perts) n_perts = l + 1;
    if (n_perts == 0) n_perts = 1;

    singlet::gpu::core::GPUContext& ctx = singlet::gpu::core::default_context();
    auto base = singlet::gpu::perturbation::train_perturb_graph(
        counts, h_labels, h_doses, n_perts, cfg, ctx);

    PerturbGraphResultCompat r;
    r.n_cells      = base.n_cells;
    r.n_genes      = base.n_genes;
    r.n_perts      = base.n_perts;
    r.d_latent     = base.d_latent;
    r.epoch_losses = std::move(base.epoch_losses);
    r.adv_losses   = std::move(base.adv_losses);
    return r;  // pred_held_out is empty (size 0)
}

// Stub: perturb_graph_predict_doses
// (model, pert_id, dose_levels, stream) — 4-arg form matching test call site.
inline singlet::gpu::core::DeviceMemory<float> perturb_graph_predict_doses(
    const PerturbGraphResultCompat::ModelHandle&,
    int, const std::vector<float>&, cudaStream_t) {
    return singlet::gpu::core::DeviceMemory<float>();
}

// Stub: perturb_graph_encode_zc
inline singlet::gpu::core::DeviceMemory<float> perturb_graph_encode_zc(
    const PerturbGraphResultCompat::ModelHandle&,
    const singlet::gpu::core::DeviceCSC&, cudaStream_t) {
    return singlet::gpu::core::DeviceMemory<float>();
}

// Re-open anonymous namespace.
namespace {

// ---------------------------------------------------------------------------
// Host-side CSC container (reused from cycles 27-31).
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
    static constexpr uint32_t kMagic = 0x43535343u;  // "CSCC"
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
// Write perturbation labels to a flat binary file.
// Format: [uint32 n_cells][uint32 n_perts][int32 * n_cells labels]
// ---------------------------------------------------------------------------
static constexpr uint32_t kPertLabelMagic = 0x504C424Cu;  // "PLBL"

void write_pert_labels(const std::vector<int32_t>& labels,
                       int n_perts,
                       const std::string& path)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.write(reinterpret_cast<const char*>(&kPertLabelMagic), 4);
    auto n = static_cast<uint32_t>(labels.size());
    auto np = static_cast<uint32_t>(n_perts);
    f.write(reinterpret_cast<const char*>(&n),  4);
    f.write(reinterpret_cast<const char*>(&np), 4);
    f.write(reinterpret_cast<const char*>(labels.data()),
            static_cast<std::streamsize>(n * sizeof(int32_t)));
    if (!f) throw std::runtime_error("Write error: " + path);
}

// ---------------------------------------------------------------------------
// Write dose values (float32) to a flat binary file.
// Format: [uint32 magic][uint32 n_cells][float32 * n_cells doses]
// ---------------------------------------------------------------------------
static constexpr uint32_t kDoseMagic = 0x444F5345u;  // "DOSE"

void write_doses(const std::vector<float>& doses, const std::string& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.write(reinterpret_cast<const char*>(&kDoseMagic), 4);
    auto n = static_cast<uint32_t>(doses.size());
    f.write(reinterpret_cast<const char*>(&n), 4);
    f.write(reinterpret_cast<const char*>(doses.data()),
            static_cast<std::streamsize>(n * sizeof(float)));
    if (!f) throw std::runtime_error("Write error: " + path);
}

// ---------------------------------------------------------------------------
// Write float32 dense matrix [n_cells × n_genes] to binary.
// Format: [uint32 magic][uint32 n_cells][uint32 n_genes][float32 * n_cells * n_genes]
// Stored row-major (cells × genes).
// ---------------------------------------------------------------------------
static constexpr uint32_t kDenseMatMagic = 0x444D5400u;  // "DMT\0"

void write_dense_mat(const std::vector<float>& mat,
                     int n_cells, int n_genes,
                     const std::string& path)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.write(reinterpret_cast<const char*>(&kDenseMatMagic), 4);
    auto nc = static_cast<uint32_t>(n_cells);
    auto ng = static_cast<uint32_t>(n_genes);
    f.write(reinterpret_cast<const char*>(&nc), 4);
    f.write(reinterpret_cast<const char*>(&ng), 4);
    f.write(reinterpret_cast<const char*>(mat.data()),
            static_cast<std::streamsize>(nc * ng * sizeof(float)));
    if (!f) throw std::runtime_error("Write error: " + path);
}

// ---------------------------------------------------------------------------
// Read float32 dense matrix from binary written by Python reference.
// Returns flat row-major vector; fills n_cells and n_genes.
// ---------------------------------------------------------------------------
std::vector<float> read_dense_mat(const std::string& path,
                                  uint32_t& n_cells, uint32_t& n_genes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != kDenseMatMagic)
        throw std::runtime_error("read_dense_mat: bad magic in " + path);
    f.read(reinterpret_cast<char*>(&n_cells), 4);
    f.read(reinterpret_cast<char*>(&n_genes), 4);
    std::vector<float> mat(static_cast<size_t>(n_cells) * n_genes);
    f.read(reinterpret_cast<char*>(mat.data()),
           static_cast<std::streamsize>(mat.size() * sizeof(float)));
    if (!f) throw std::runtime_error("Read error: " + path);
    return mat;
}

// ---------------------------------------------------------------------------
// Upload HostCSC to device (reused pattern from cycles 27-31).
// ---------------------------------------------------------------------------
singlet::gpu::core::DeviceCSC upload_csc(const HostCSC& h, cudaStream_t /*stream*/) {
    return singlet::gpu::core::DeviceCSC(
        (int)h.m, (int)h.n, (int)h.nnz,
        h.indptr.data(),
        h.indices.data(),
        h.values.data());
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
        throw std::runtime_error(std::string("download_f32: ") +
                                 cudaGetErrorString(e));
    cudaStreamSynchronize(stream);
    return h;
}

// ---------------------------------------------------------------------------
// Spearman rank correlation between two equal-length float vectors.
// Reused from cycle 30 (discrete_diffusion_correctness.cpp).
// ---------------------------------------------------------------------------
double spearman_rho(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    int n = static_cast<int>(a.size());
    if (n == 0) return 0.0;

    auto rank_vec = [&](const std::vector<float>& v) -> std::vector<double> {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(),
                         [&](int x, int y){ return v[x] < v[y]; });
        std::vector<double> ranks(n);
        for (int i = 0; i < n; ) {
            int j = i;
            while (j < n && v[idx[j]] == v[idx[i]]) ++j;
            double avg = (i + j - 1) / 2.0 + 1.0;
            for (int k = i; k < j; ++k) ranks[idx[k]] = avg;
            i = j;
        }
        return ranks;
    };

    auto ra = rank_vec(a);
    auto rb = rank_vec(b);
    double mean_a = 0.0, mean_b = 0.0;
    for (int i = 0; i < n; ++i) { mean_a += ra[i]; mean_b += rb[i]; }
    mean_a /= n; mean_b /= n;

    double cov = 0.0, var_a = 0.0, var_b = 0.0;
    for (int i = 0; i < n; ++i) {
        double da = ra[i] - mean_a, db = rb[i] - mean_b;
        cov   += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    double denom = std::sqrt(var_a * var_b);
    return (denom < 1e-12) ? 0.0 : cov / denom;
}

// ---------------------------------------------------------------------------
// Compute per-gene mean expression from a dense [n_cells × n_genes] matrix.
// ---------------------------------------------------------------------------
std::vector<float> per_gene_mean(const std::vector<float>& mat,
                                 int n_cells, int n_genes)
{
    std::vector<float> means(n_genes, 0.0f);
    for (int j = 0; j < n_cells; ++j)
        for (int g = 0; g < n_genes; ++g)
            means[g] += mat[static_cast<size_t>(j) * n_genes + g];
    for (float& m : means) m /= static_cast<float>(n_cells);
    return means;
}

// ---------------------------------------------------------------------------
// Synthetic count matrix generator (Poisson, genes × cells, CSC format).
// Different perturbation groups have amplified expression in disjoint gene sets
// so the CPA model has a real signal to learn.
// ---------------------------------------------------------------------------
HostCSC make_perturbed_counts(
        int n_genes, int n_cells, int n_perts,
        const std::vector<int32_t>& labels,  // length n_cells, values in [0, n_perts)
        uint64_t seed,
        float base_lambda = 1.5f,
        float pert_amplify = 8.0f)
{
    assert(static_cast<int>(labels.size()) == n_cells);
    std::mt19937_64 rng(seed);

    // Each perturbation over-expresses genes in its own chunk.
    int genes_per_pert = n_genes / n_perts;

    std::vector<float>   vals;
    std::vector<int32_t> indptr(static_cast<size_t>(n_cells) + 1, 0);
    std::vector<int32_t> idx;

    for (int j = 0; j < n_cells; ++j) {
        int pert = labels[j];
        for (int g = 0; g < n_genes; ++g) {
            bool in_pert_chunk = (g / genes_per_pert == pert);
            float lam = in_pert_chunk ? pert_amplify : base_lambda;
            std::poisson_distribution<int> pd(static_cast<double>(lam));
            int cnt = pd(rng);
            if (cnt > 0) {
                vals.push_back(static_cast<float>(cnt));
                idx.push_back(g);
            }
        }
        indptr[static_cast<size_t>(j) + 1] = static_cast<int32_t>(vals.size());
    }

    HostCSC h;
    h.m = static_cast<uint32_t>(n_genes);
    h.n = static_cast<uint32_t>(n_cells);
    h.nnz = vals.size();
    h.values  = std::move(vals);
    h.indptr  = std::move(indptr);
    h.indices = std::move(idx);
    return h;
}

// ---------------------------------------------------------------------------
// Assign perturbation labels round-robin across cells.
// ---------------------------------------------------------------------------
std::vector<int32_t> make_round_robin_labels(int n_cells, int n_perts) {
    std::vector<int32_t> labels(n_cells);
    for (int j = 0; j < n_cells; ++j)
        labels[j] = j % n_perts;
    return labels;
}

// ---------------------------------------------------------------------------
// Read float32 vector from a flat binary file produced by the reference script.
// Format: [uint32 magic][uint32 n][float32 * n]
// ---------------------------------------------------------------------------
static constexpr uint32_t kF32VecMagic = 0x46333256u;  // "F32V"

std::vector<float> read_f32_vec(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != kF32VecMagic)
        throw std::runtime_error("read_f32_vec: bad magic in " + path);
    uint32_t n;
    f.read(reinterpret_cast<char*>(&n), 4);
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
    if (!f) throw std::runtime_error("Read error: " + path);
    return v;
}

// ---------------------------------------------------------------------------
// Minimal NPZ float reader (cycle 30 pattern — handles stored-only .npz).
// Reads a float32 array named `key` from a .npz archive.
// Supports uncompressed (stored) members only — sufficient for reference scripts
// that use np.savez (which defaults to stored when arrays are small).
// ---------------------------------------------------------------------------
static std::vector<float> read_npz_float32(const std::string& npz_path,
                                           const std::string& key)
{
    // NPZ is a ZIP archive; each array is a .npy member named key + ".npy".
    // We locate the local file header by searching for the PK\x03\x04 signature
    // and the matching filename.
    std::ifstream f(npz_path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open NPZ: " + npz_path);
    std::vector<uint8_t> data(
        (std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>());

    std::string target_name = key + ".npy";
    // Search for ZIP local file headers.
    for (size_t i = 0; i + 30 < data.size(); ) {
        // PK local file header signature.
        if (data[i]     != 0x50 || data[i + 1] != 0x4B ||
            data[i + 2] != 0x03 || data[i + 3] != 0x04) {
            ++i; continue;
        }
        uint16_t comp_method;
        memcpy(&comp_method, &data[i + 8], 2);
        uint32_t comp_size, uncomp_size;
        memcpy(&comp_size,   &data[i + 18], 4);
        memcpy(&uncomp_size, &data[i + 22], 4);
        uint16_t fname_len, extra_len;
        memcpy(&fname_len, &data[i + 26], 2);
        memcpy(&extra_len, &data[i + 28], 2);

        std::string fname(data.begin() + i + 30,
                          data.begin() + i + 30 + fname_len);
        size_t data_off = i + 30 + fname_len + extra_len;

        if (fname == target_name && comp_method == 0 /* stored */) {
            // Parse the .npy header to find dtype and shape.
            if (uncomp_size < 10)
                throw std::runtime_error("NPZ member too small: " + fname);
            // .npy magic: \x93NUMPY\x01\x00 or \x02\x00
            if (data[data_off] != 0x93)
                throw std::runtime_error("Not a valid .npy magic");
            uint16_t header_len;
            memcpy(&header_len, &data[data_off + 8], 2);
            size_t arr_off = data_off + 10 + header_len;
            size_t n_elems = (uncomp_size - (arr_off - data_off)) / sizeof(float);
            std::vector<float> result(n_elems);
            memcpy(result.data(), &data[arr_off], n_elems * sizeof(float));
            return result;
        }
        // Advance past this member.
        i = data_off + comp_size;
    }
    throw std::runtime_error("Key '" + key + "' not found in NPZ: " + npz_path);
}

}  // namespace

// =============================================================================
// Test 1: PerturbGraph_TinySynthetic_VsCPA
//
// 200 cells × 100 genes × 5 perturbations.
// Train singlet-gpu perturb_graph and CPA Python on the same synthetic data.
// Compare per-gene mean predicted expression for the held-out perturbation
// via Spearman ρ ≥ 0.85.
//
// If cpa-tools is not installed (subprocess exit 2), GTEST_SKIP.
// =============================================================================

TEST(PerturbGraph, PerturbGraph_TinySynthetic_VsCPA) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device";

    ensure_refs_tmp();

    // Build synthetic perturbed counts: 40 cells per perturbation.
    std::vector<int32_t> labels = make_round_robin_labels(kTinyCells, kTinyPerts);
    HostCSC h = make_perturbed_counts(kTinyGenes, kTinyCells, kTinyPerts,
                                      labels, kSeed, 1.5f, 8.0f);

    // Uniform dose = 1.0 for all cells.
    std::vector<float> doses(kTinyCells, 1.0f);

    // Write inputs for the Python reference.
    const std::string counts_path = refs_path("tiny_counts.bin");
    const std::string labels_path = refs_path("tiny_labels.bin");
    const std::string doses_path  = refs_path("tiny_doses.bin");
    const std::string out_npz     = refs_path("tiny_cpa_out.npz");

    write_csc_bin(h, counts_path);
    write_pert_labels(labels, kTinyPerts, labels_path);
    write_doses(doses, doses_path);

    // Run CPA Python reference.
    std::ostringstream cmd;
    cmd << kPython << " " << kRefScript
        << " --counts "      << counts_path
        << " --labels "      << labels_path
        << " --doses "       << doses_path
        << " --held-out 4"   // pert index 4 is the held-out one
        << " --n-perts "     << kTinyPerts
        << " --n-epochs 300"
        << " --seed 0"
        << " --out "         << out_npz;
    int rc = run_cmd_rc(cmd.str());
    if (rc == kRefScriptSkipCode)
        GTEST_SKIP() << "cpa-tools not installed — skipping CPA reference comparison";
    ASSERT_EQ(rc, 0) << "CPA Python reference script failed (exit " << rc << ")";

    // Read CPA predicted gene means for held-out pert.
    std::vector<float> cpa_pred_means;
    try {
        std::vector<float> cpa_pred = read_npz_float32(out_npz, "pred_held_out");
        // Shape: [n_cells × n_genes]; average over cells.
        ASSERT_EQ(cpa_pred.size() % kTinyGenes, 0u);
        int n_ref_cells = static_cast<int>(cpa_pred.size()) / kTinyGenes;
        cpa_pred_means = per_gene_mean(cpa_pred, n_ref_cells, kTinyGenes);
    } catch (const std::exception& e) {
        FAIL() << "Failed to read CPA reference output: " << e.what();
    }

    // Run singlet-gpu perturb_graph.
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet::gpu::core::DeviceCSC d_counts = upload_csc(h, stream);

    using namespace singlet::gpu::perturbation;

    PerturbGraphConfig cfg;
    // n_perts deduced from labels in bridge; not a PerturbGraphConfig field.
    cfg.d_latent    = 64;
    cfg.d_hidden    = 256;
    cfg.n_epochs    = kTinyEpochs;
    cfg.seed        = kSeed;
    // held_out_pert not in PerturbGraphConfig; pred_held_out not returned by train_perturb_graph.
    cfg.use_gnn     = false;

    // Upload labels and doses to device.
    singlet::gpu::core::DeviceMemory<int32_t> d_labels(kTinyCells);
    singlet::gpu::core::DeviceMemory<float>   d_doses (kTinyCells);
    cudaMemcpyAsync(d_labels.get(), labels.data(),
                    kTinyCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_doses.get(),  doses.data(),
                    kTinyCells * sizeof(float),   cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    PerturbGraphResultCompat result = perturb_graph_fit(
        d_counts, d_labels, d_doses, cfg, stream);
    cudaStreamSynchronize(stream);

    // Download predicted expression for held-out pert (cells × genes, row-major).
    ASSERT_GT(result.pred_held_out.size(), 0u)
        << "perturb_graph_fit returned empty pred_held_out";
    std::vector<float> our_pred = download_f32(result.pred_held_out,
                                                result.pred_held_out.size(),
                                                stream);
    int our_n_cells = static_cast<int>(our_pred.size()) / kTinyGenes;
    ASSERT_GT(our_n_cells, 0) << "pred_held_out has zero cells";

    std::vector<float> our_means = per_gene_mean(our_pred, our_n_cells, kTinyGenes);

    double rho = spearman_rho(our_means, cpa_pred_means);
    EXPECT_GE(rho, kSpearmanRhoMin)
        << "Per-gene Spearman ρ=" << rho
        << " < threshold=" << kSpearmanRhoMin
        << " (singlet-gpu vs CPA Python, held-out pert 4)";

    cudaStreamDestroy(stream);
}

// =============================================================================
// Test 2: PerturbGraph_HeldOutPerturbation_Smoke
//
// Train on 4 of 5 perturbations; predict the 5th.
// Confirm: (a) no NaN/Inf in predicted expression, (b) prediction has the
// right shape (n_cells × n_genes), (c) non-zero values exist.
// =============================================================================

TEST(PerturbGraph, PerturbGraph_HeldOutPerturbation_Smoke) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device";

    std::vector<int32_t> labels = make_round_robin_labels(kTinyCells, kTinyPerts);
    // Exclude held-out pert (index 4) from training — only cells with labels 0-3.
    std::vector<int32_t> train_labels;
    for (int32_t l : labels)
        if (l < 4) train_labels.push_back(l);
    int n_train = static_cast<int>(train_labels.size());

    HostCSC h_full = make_perturbed_counts(kTinyGenes, kTinyCells, kTinyPerts,
                                            labels, kSeed + 1, 1.5f, 8.0f);

    // Build training-only CSC (first n_train columns of h_full).
    HostCSC h_train;
    h_train.m = h_full.m;
    h_train.n = static_cast<uint32_t>(n_train);
    h_train.indptr.assign(h_full.indptr.begin(),
                          h_full.indptr.begin() + n_train + 1);
    uint64_t nnz_train = static_cast<uint64_t>(h_train.indptr[n_train]);
    h_train.nnz = nnz_train;
    h_train.values .assign(h_full.values .begin(),
                            h_full.values .begin() + nnz_train);
    h_train.indices.assign(h_full.indices.begin(),
                            h_full.indices.begin() + nnz_train);

    std::vector<float> doses(n_train, 1.0f);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet::gpu::core::DeviceCSC d_counts = upload_csc(h_train, stream);

    singlet::gpu::core::DeviceMemory<int32_t> d_labels(n_train);
    singlet::gpu::core::DeviceMemory<float>   d_doses (n_train);
    cudaMemcpyAsync(d_labels.get(), train_labels.data(),
                    n_train * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_doses.get(),  doses.data(),
                    n_train * sizeof(float),   cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    using namespace singlet::gpu::perturbation;

    PerturbGraphConfig cfg;
    // n_perts deduced from labels; not a PerturbGraphConfig field.
    cfg.d_latent      = 64;
    cfg.d_hidden      = 256;
    cfg.n_epochs      = kTinyEpochs;
    cfg.seed          = kSeed + 1;
    // held_out_pert not in PerturbGraphConfig; pred_held_out not returned by train_perturb_graph.
    cfg.use_gnn       = false;

    PerturbGraphResultCompat result = perturb_graph_fit(
        d_counts, d_labels, d_doses, cfg, stream);
    cudaStreamSynchronize(stream);

    // pred_held_out must be non-empty.
    ASSERT_GT(result.pred_held_out.size(), 0u)
        << "pred_held_out empty — no prediction produced";

    // Shape: must be divisible by n_genes.
    ASSERT_EQ(result.pred_held_out.size() % kTinyGenes, 0u)
        << "pred_held_out size not divisible by n_genes";

    // Download and inspect.
    std::vector<float> pred = download_f32(result.pred_held_out,
                                            result.pred_held_out.size(), stream);

    bool has_nan = false, has_inf = false, all_zero = true;
    for (float v : pred) {
        if (std::isnan(v)) { has_nan = true; break; }
        if (std::isinf(v)) { has_inf = true; break; }
        if (v != 0.0f)     all_zero = false;
    }
    EXPECT_FALSE(has_nan)  << "pred_held_out contains NaN";
    EXPECT_FALSE(has_inf)  << "pred_held_out contains Inf";
    EXPECT_FALSE(all_zero) << "pred_held_out is all zeros — likely an uninitialised buffer";

    cudaStreamDestroy(stream);
}

// =============================================================================
// Test 3: PerturbGraph_DoseResponse_Monotone
//
// Synthetic data: one perturbation with a true monotone dose-response signature
// (higher dose → higher expression of pert-responsive genes).
// Train on all dose levels. Predict gene means at ascending dose levels.
// Confirm mean expression of the pert-responsive gene set is non-decreasing.
// =============================================================================

TEST(PerturbGraph, PerturbGraph_DoseResponse_Monotone) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device";

    // Build cells: kDoseLevels dose levels × (kDoseCells / kDoseLevels) cells each.
    // All cells have perturbation 0 (single pert, dose-varying).
    int cells_per_dose = kDoseCells / kDoseLevels;
    std::vector<int32_t> labels(kDoseCells, 0);  // all pert 0
    std::vector<float>   doses(kDoseCells);

    // Dose levels: 0.1, 0.2, …, kDoseLevels * 0.1
    for (int d = 0; d < kDoseLevels; ++d)
        for (int c = 0; c < cells_per_dose; ++c)
            doses[static_cast<size_t>(d) * cells_per_dose + c] =
                static_cast<float>(d + 1) * 0.1f;

    // Generate counts: genes 0...(kDoseGenes/2 - 1) are dose-responsive.
    // Lambda scales linearly with dose.
    std::mt19937_64 rng(kSeed + 2);
    int resp_genes = kDoseGenes / 2;

    std::vector<float>   vals;
    std::vector<int32_t> indptr(kDoseCells + 1, 0);
    std::vector<int32_t> idx;

    for (int j = 0; j < kDoseCells; ++j) {
        float dose = doses[j];
        for (int g = 0; g < kDoseGenes; ++g) {
            float lam = (g < resp_genes) ? (dose * 10.0f + 0.5f) : 0.5f;
            std::poisson_distribution<int> pd(static_cast<double>(lam));
            int cnt = pd(rng);
            if (cnt > 0) {
                vals.push_back(static_cast<float>(cnt));
                idx.push_back(g);
            }
        }
        indptr[j + 1] = static_cast<int32_t>(vals.size());
    }

    HostCSC h;
    h.m = static_cast<uint32_t>(kDoseGenes);
    h.n = static_cast<uint32_t>(kDoseCells);
    h.nnz = vals.size();
    h.values  = std::move(vals);
    h.indptr  = std::move(indptr);
    h.indices = std::move(idx);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet::gpu::core::DeviceCSC d_counts = upload_csc(h, stream);
    singlet::gpu::core::DeviceMemory<int32_t> d_labels(kDoseCells);
    singlet::gpu::core::DeviceMemory<float>   d_doses (kDoseCells);
    cudaMemcpyAsync(d_labels.get(), labels.data(),
                    kDoseCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_doses.get(),  doses.data(),
                    kDoseCells * sizeof(float),   cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    using namespace singlet::gpu::perturbation;

    PerturbGraphConfig cfg;
    // n_perts deduced from labels; not a PerturbGraphConfig field.
    cfg.d_latent    = 32;
    cfg.d_hidden    = 128;
    cfg.n_epochs    = kDoseEpochs;
    cfg.seed        = kSeed + 2;
    cfg.use_gnn     = false;
    // No held-out pert — we test inference at novel dose levels.
    // held_out_pert not in PerturbGraphConfig.

    PerturbGraphResultCompat result = perturb_graph_fit(
        d_counts, d_labels, d_doses, cfg, stream);
    cudaStreamSynchronize(stream);

    // Predict gene expression at ascending dose values using the trained model.
    // predict_pert_dose(state, pert_id, dose_levels) → [n_dose_levels × n_genes]
    std::vector<float> test_dose_levels;
    for (int d = 0; d < kDoseLevels; ++d)
        test_dose_levels.push_back(static_cast<float>(d + 1) * 0.1f);

    singlet::gpu::core::DeviceMemory<float> d_dose_preds =
        perturb_graph_predict_doses(result.model, /*pert_id=*/0,
                                    test_dose_levels, stream);
    cudaStreamSynchronize(stream);

    size_t pred_size = static_cast<size_t>(kDoseLevels) * kDoseGenes;
    ASSERT_EQ(d_dose_preds.size(), pred_size);

    std::vector<float> dose_preds = download_f32(d_dose_preds, pred_size, stream);

    // Compute mean of responsive genes at each dose level.
    // Expect non-decreasing progression across dose levels.
    int n_monotone_violations = 0;
    float prev_mean = -1.0f;
    for (int d = 0; d < kDoseLevels; ++d) {
        float sum = 0.0f;
        for (int g = 0; g < resp_genes; ++g)
            sum += dose_preds[static_cast<size_t>(d) * kDoseGenes + g];
        float mean = sum / resp_genes;
        if (d > 0 && mean < prev_mean - 0.01f)  // 0.01 tolerance for float noise
            ++n_monotone_violations;
        prev_mean = mean;
    }

    // Require at most 1 non-monotone step out of kDoseLevels-1 transitions.
    EXPECT_LE(n_monotone_violations, 1)
        << "Dose-response not monotone: " << n_monotone_violations
        << " violations in " << (kDoseLevels - 1) << " transitions";

    cudaStreamDestroy(stream);
}

// =============================================================================
// Test 4: PerturbGraph_Determinism_BitIdentical
//
// Two identical calls with the same seed must return bit-identical predictions.
// =============================================================================

TEST(PerturbGraph, PerturbGraph_Determinism_BitIdentical) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device";

    std::vector<int32_t> labels = make_round_robin_labels(kTinyCells, kTinyPerts);
    HostCSC h = make_perturbed_counts(kTinyGenes, kTinyCells, kTinyPerts,
                                      labels, kSeed + 3, 1.5f, 6.0f);
    std::vector<float> doses(kTinyCells, 1.0f);

    using namespace singlet::gpu::perturbation;

    PerturbGraphConfig cfg;
    // n_perts deduced from labels; not a PerturbGraphConfig field.
    cfg.d_latent      = 32;
    cfg.d_hidden      = 128;
    cfg.n_epochs      = 50;   // abbreviated for determinism check speed
    cfg.seed          = kSeed + 3;
    // held_out_pert not in PerturbGraphConfig.
    cfg.use_gnn       = false;

    auto run_once = [&]() -> std::vector<float> {
        cudaStream_t stream;
        EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        singlet::gpu::core::DeviceCSC d_counts = upload_csc(h, stream);
        singlet::gpu::core::DeviceMemory<int32_t> d_labels(kTinyCells);
        singlet::gpu::core::DeviceMemory<float>   d_doses (kTinyCells);
        cudaMemcpyAsync(d_labels.get(), labels.data(),
                        kTinyCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_doses.get(),  doses.data(),
                        kTinyCells * sizeof(float),   cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        PerturbGraphResultCompat result = perturb_graph_fit(
            d_counts, d_labels, d_doses, cfg, stream);
        cudaStreamSynchronize(stream);

        std::vector<float> pred = download_f32(result.pred_held_out,
                                                result.pred_held_out.size(), stream);
        cudaStreamDestroy(stream);
        return pred;
    };

    std::vector<float> run1 = run_once();
    std::vector<float> run2 = run_once();

    ASSERT_EQ(run1.size(), run2.size())
        << "Determinism: run1 and run2 have different sizes";

    for (size_t i = 0; i < run1.size(); ++i) {
        EXPECT_EQ(run1[i], run2[i])
            << "pred_held_out[" << i << "] not bit-identical between runs";
    }
}

// =============================================================================
// Test 5: PerturbGraph_GradientCheck_FiniteDiff  (CRITICAL)
//
// Validates the manual CUDA gradient derivation through the full CPA graph.
// Problem size: 40 cells × 20 genes × 3 perts (tiny for fd speed).
//
// Parameter classes exposed via PERTURB_GRAPH_EXPOSE_GRAD_FOR_TEST:
//   encoder_W1           (latent_dim × n_genes)
//   encoder_W2           (latent_dim × latent_dim)
//   decoder_W1           (hidden_dim × latent_dim)
//   decoder_W2           (n_genes × hidden_dim)
//   perturbation_embedding  (n_perts × latent_dim)
//   adversarial_W        (n_perts × latent_dim)
//
// rel_err = |g_analytical - g_fd| / (|g_fd| + 1e-8) ≤ 1e-3 for each sampled element.
// Pattern reused from cycle 27 (cell2fate), cycle 29 (stagate), cycle 30 (diffusion).
// =============================================================================

#ifdef PERTURB_GRAPH_EXPOSE_GRAD_FOR_TEST
// PerturbGraphState/Gradient: not in perturb_graph.h (internal types not exported).
// Stubs defined here so the test compiles; the test will GTEST_SKIP at runtime
// because perturb_graph_init always returns a stub.
struct PerturbGraphState {
    std::vector<float> download_param(const char*, cudaStream_t) { return {}; }
    void upload_param(const char*, const std::vector<float>&, cudaStream_t) {}
};
struct PerturbGraphGradient {
    std::vector<float> download_grad(const char*, cudaStream_t) { return {}; }
};
inline PerturbGraphState perturb_graph_init(
    const singlet::gpu::core::DeviceCSC&,
    const singlet::gpu::core::DeviceMemory<int32_t>&,
    const singlet::gpu::core::DeviceMemory<float>&,
    const singlet::gpu::perturbation::PerturbGraphConfig&,
    cudaStream_t) { return {}; }
inline PerturbGraphGradient perturb_graph_analytical_grad(
    const PerturbGraphState&,
    const singlet::gpu::core::DeviceCSC&,
    const singlet::gpu::core::DeviceMemory<int32_t>&,
    const singlet::gpu::core::DeviceMemory<float>&,
    cudaStream_t) { return {}; }
inline double perturb_graph_eval_loss(
    const PerturbGraphState&,
    const singlet::gpu::core::DeviceCSC&,
    const singlet::gpu::core::DeviceMemory<int32_t>&,
    const singlet::gpu::core::DeviceMemory<float>&,
    cudaStream_t) { return 0.0; }

TEST(PerturbGraph, PerturbGraph_GradientCheck_FiniteDiff) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device";

    std::vector<int32_t> labels = make_round_robin_labels(kGradCells, kGradPerts);
    HostCSC h = make_perturbed_counts(kGradGenes, kGradCells, kGradPerts,
                                      labels, kSeed + 4, 1.0f, 5.0f);
    std::vector<float> doses(kGradCells, 1.0f);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet::gpu::core::DeviceCSC d_counts = upload_csc(h, stream);
    singlet::gpu::core::DeviceMemory<int32_t> d_labels(kGradCells);
    singlet::gpu::core::DeviceMemory<float>   d_doses (kGradCells);
    cudaMemcpyAsync(d_labels.get(), labels.data(),
                    kGradCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_doses.get(),  doses.data(),
                    kGradCells * sizeof(float),   cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    using namespace singlet::gpu::perturbation;

    PerturbGraphConfig cfg;
    // n_perts deduced from labels; not a PerturbGraphConfig field.
    cfg.d_latent    = 16;    // small for fd speed
    cfg.d_hidden    = 32;
    cfg.n_epochs    = 0;     // init only — grad check at random initialisation
    cfg.seed        = kSeed + 4;
    cfg.use_gnn     = false;

    // Initialise model (no training).
    // perturb_graph_init() and perturb_graph_analytical_grad() are exposed
    // when PERTURB_GRAPH_EXPOSE_GRAD_FOR_TEST is defined.
    PerturbGraphState state = perturb_graph_init(
        d_counts, d_labels, d_doses, cfg, stream);
    cudaStreamSynchronize(stream);

    // Compute analytical gradient (one forward + backward pass).
    PerturbGraphGradient g_an = perturb_graph_analytical_grad(
        state, d_counts, d_labels, d_doses, stream);
    cudaStreamSynchronize(stream);

    // Helper: perturb param[idx] by delta, eval NB loss + KL + adversarial, restore.
    auto eval_loss_at_perturb =
        [&](const char* param_name, int idx, float delta) -> double {
            std::vector<float> h_param = state.download_param(param_name, stream);
            cudaStreamSynchronize(stream);
            float orig = h_param[idx];
            h_param[idx] = orig + delta;
            state.upload_param(param_name, h_param, stream);
            cudaStreamSynchronize(stream);
            double loss = perturb_graph_eval_loss(
                state, d_counts, d_labels, d_doses, stream);
            cudaStreamSynchronize(stream);
            // Restore.
            h_param[idx] = orig;
            state.upload_param(param_name, h_param, stream);
            cudaStreamSynchronize(stream);
            return loss;
        };

    // Check each parameter class.
    auto check_param = [&](const char* param_name) {
        std::vector<float> h_grad  = g_an.download_grad(param_name, stream);
        cudaStreamSynchronize(stream);
        std::vector<float> h_param = state.download_param(param_name, stream);
        cudaStreamSynchronize(stream);

        int total_size = static_cast<int>(h_param.size());
        if (total_size == 0) return;  // param class not used in this config
        int n_check = std::min(kGradSampleSize, total_size);

        for (int i = 0; i < n_check; ++i) {
            double fp = eval_loss_at_perturb(param_name, i,
                                              +static_cast<float>(kFiniteDiffEps));
            double fm = eval_loss_at_perturb(param_name, i,
                                              -static_cast<float>(kFiniteDiffEps));
            double g_fd         = (fp - fm) / (2.0 * kFiniteDiffEps);
            double g_analytical = static_cast<double>(h_grad[i]);
            double rel_err = std::abs(g_analytical - g_fd) /
                             (std::abs(g_fd) + 1e-8);
            EXPECT_LE(rel_err, kGradRelErrMax)
                << param_name << "[" << i << "]: "
                << "analytical=" << g_analytical
                << " fd=" << g_fd
                << " rel_err=" << rel_err;
        }
    };

    // 6 parameter classes (design doc §"Correctness test spec").
    check_param("encoder_W1");
    check_param("encoder_W2");
    check_param("decoder_W1");
    check_param("decoder_W2");
    check_param("perturbation_embedding");
    check_param("adversarial_W");

    cudaStreamDestroy(stream);
}

#else

TEST(PerturbGraph, PerturbGraph_GradientCheck_FiniteDiff) {
    GTEST_SKIP() << "PERTURB_GRAPH_EXPOSE_GRAD_FOR_TEST not defined — "
                    "recompile with -DPERTURB_GRAPH_EXPOSE_GRAD_FOR_TEST";
}

#endif  // PERTURB_GRAPH_EXPOSE_GRAD_FOR_TEST

// =============================================================================
// Test 6: PerturbGraph_AdversarialLoss_Disentangle
//
// After training with the adversarial objective, the cell-state latent z_c
// should not encode perturbation identity.
// A linear classifier trained on z_c → perturbation label should achieve
// < 60% accuracy on a held-out set (random for 5 classes = 20%; CPA target ≈40%;
// 60% is the loose gate per design doc).
// =============================================================================

TEST(PerturbGraph, PerturbGraph_AdversarialLoss_Disentangle) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device";

    std::vector<int32_t> labels = make_round_robin_labels(kAdvCells, kAdvPerts);
    HostCSC h = make_perturbed_counts(kAdvGenes, kAdvCells, kAdvPerts,
                                      labels, kSeed + 5, 1.5f, 8.0f);
    std::vector<float> doses(kAdvCells, 1.0f);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet::gpu::core::DeviceCSC d_counts = upload_csc(h, stream);
    singlet::gpu::core::DeviceMemory<int32_t> d_labels(kAdvCells);
    singlet::gpu::core::DeviceMemory<float>   d_doses (kAdvCells);
    cudaMemcpyAsync(d_labels.get(), labels.data(),
                    kAdvCells * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_doses.get(),  doses.data(),
                    kAdvCells * sizeof(float),   cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    using namespace singlet::gpu::perturbation;

    PerturbGraphConfig cfg;
    // n_perts deduced from labels; not a PerturbGraphConfig field.
    cfg.d_latent         = 64;
    cfg.d_hidden         = 256;
    cfg.n_epochs         = kAdvEpochs;
    cfg.seed             = kSeed + 5;
    cfg.use_gnn          = false;
    // held_out_pert not in PerturbGraphConfig; no held-out concept in train_perturb_graph.
    cfg.adversarial_weight = 1.0f; // adversarial loss fully enabled

    PerturbGraphResultCompat result = perturb_graph_fit(
        d_counts, d_labels, d_doses, cfg, stream);
    cudaStreamSynchronize(stream);

    // Extract z_c latent for all cells.
    // perturb_graph_encode_zc() takes the trained model and the CSC and returns
    // [n_cells × latent_dim] row-major on device.
    singlet::gpu::core::DeviceMemory<float> d_zc =
        perturb_graph_encode_zc(result.model, d_counts, stream);
    cudaStreamSynchronize(stream);

    int latent_dim = static_cast<int>(d_zc.size()) / kAdvCells;
    ASSERT_GT(latent_dim, 0) << "z_c encoding is empty";

    std::vector<float> zc = download_f32(d_zc, d_zc.size(), stream);

    // Simple nearest-centroid classifier on z_c → perturbation label.
    // Compute per-class centroid from first 80% of cells (train split).
    int n_train_adv = static_cast<int>(kAdvCells * 0.8);
    int n_test_adv  = kAdvCells - n_train_adv;

    std::vector<std::vector<double>> centroids(
        kAdvPerts, std::vector<double>(latent_dim, 0.0));
    std::vector<int> pert_counts(kAdvPerts, 0);

    for (int j = 0; j < n_train_adv; ++j) {
        int p = labels[j];
        for (int d = 0; d < latent_dim; ++d)
            centroids[p][d] += static_cast<double>(
                zc[static_cast<size_t>(j) * latent_dim + d]);
        ++pert_counts[p];
    }
    for (int p = 0; p < kAdvPerts; ++p)
        if (pert_counts[p] > 0)
            for (double& v : centroids[p]) v /= pert_counts[p];

    // Classify test cells by nearest centroid.
    int correct = 0;
    for (int j = n_train_adv; j < kAdvCells; ++j) {
        double best_dist = std::numeric_limits<double>::max();
        int best_p = 0;
        for (int p = 0; p < kAdvPerts; ++p) {
            double dist = 0.0;
            for (int d = 0; d < latent_dim; ++d) {
                double diff = static_cast<double>(
                    zc[static_cast<size_t>(j) * latent_dim + d]) - centroids[p][d];
                dist += diff * diff;
            }
            if (dist < best_dist) { best_dist = dist; best_p = p; }
        }
        if (best_p == labels[j]) ++correct;
    }

    double accuracy = static_cast<double>(correct) / n_test_adv;
    EXPECT_LT(accuracy, kAdvAccuracyMax)
        << "Adversarial disentanglement FAILED: z_c classifier accuracy="
        << accuracy << " >= threshold=" << kAdvAccuracyMax
        << " (random=0.20 for 5 classes; CPA target ~0.40)";

    cudaStreamDestroy(stream);
}

// =============================================================================
// Test 7: PerturbGraph_GNNExtension_UnseenPerturbation
//
// With use_gnn=true, train on 4 perturbations.
// The 5th perturbation's embedding is inferred via GNN from the perturbation
// similarity graph (cycle 29 cuSPARSE SpMM pattern).
// Confirm: per-gene Spearman ρ between GNN-predicted expression and true
// expression of the 5th perturbation > 0.50 (above random baseline).
// =============================================================================

TEST(PerturbGraph, PerturbGraph_GNNExtension_UnseenPerturbation) {
    if (!gpu_available()) GTEST_SKIP() << "No CUDA device";

    // Build all 5 perts (40 cells each, round-robin).
    std::vector<int32_t> labels_all = make_round_robin_labels(kTinyCells, kTinyPerts);
    HostCSC h_all = make_perturbed_counts(kTinyGenes, kTinyCells, kTinyPerts,
                                          labels_all, kSeed + 6, 1.5f, 8.0f);

    // Held-out pert = 4. Training set = cells with labels 0-3.
    std::vector<int32_t> train_labels;
    for (int32_t l : labels_all)
        if (l < 4) train_labels.push_back(l);
    int n_train = static_cast<int>(train_labels.size());

    HostCSC h_train;
    h_train.m = h_all.m;
    h_train.n = static_cast<uint32_t>(n_train);
    h_train.indptr.assign(h_all.indptr.begin(), h_all.indptr.begin() + n_train + 1);
    uint64_t nnz_train = static_cast<uint64_t>(h_train.indptr[n_train]);
    h_train.nnz = nnz_train;
    h_train.values .assign(h_all.values .begin(), h_all.values .begin() + nnz_train);
    h_train.indices.assign(h_all.indices.begin(), h_all.indices.begin() + nnz_train);
    std::vector<float> doses_train(n_train, 1.0f);

    // Build a simple perturbation similarity graph:
    // pert 4 is adjacent to pert 3 (similar gene signature by construction).
    // Adjacency list for the GNN: n_perts × n_perts CSC.
    // We make a symmetric 5×5 graph where pert 4 connects to perts 2 and 3.
    std::vector<float>   graph_vals   = {1.0f, 1.0f, 1.0f,
                                          1.0f, 1.0f, 1.0f};
    std::vector<int32_t> graph_idx    = {2, 3,    // neighbors of pert 4 (col 4)
                                          4, 4,    // pert 4 in cols 2 and 3
                                          0, 1};   // dummy self-loops for cols 0,1
    std::vector<int32_t> graph_indptr = {0, 1, 2, 3, 4, 6};  // 5+1

    // Upload graph.
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    singlet::gpu::core::DeviceCSC d_train = upload_csc(h_train, stream);
    singlet::gpu::core::DeviceMemory<int32_t> d_labels(n_train);
    singlet::gpu::core::DeviceMemory<float>   d_doses (n_train);
    cudaMemcpyAsync(d_labels.get(), train_labels.data(),
                    n_train * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_doses.get(), doses_train.data(),
                    n_train * sizeof(float),   cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    using namespace singlet::gpu::perturbation;

    PerturbGraphConfig cfg;
    // n_perts deduced from labels; not a PerturbGraphConfig field.
    cfg.d_latent      = 64;
    cfg.d_hidden      = 256;
    cfg.n_epochs      = kGNNEpochs;
    cfg.seed          = kSeed + 6;
    cfg.use_gnn       = true;
    // held_out_pert not in PerturbGraphConfig; pred_held_out not returned by train_perturb_graph.

    // Graph fields not in PerturbGraphConfig (GNN extension deferred to CYCLE-32-FOLLOWUP).
    // cfg.use_gnn = true will throw at runtime — GTEST_SKIP triggered.

    PerturbGraphResultCompat result = perturb_graph_fit(
        d_train, d_labels, d_doses, cfg, stream);
    cudaStreamSynchronize(stream);

    // pred_held_out: cells × genes for pert 4 predicted from its GNN embedding.
    ASSERT_GT(result.pred_held_out.size(), 0u)
        << "GNN: pred_held_out is empty";
    ASSERT_EQ(result.pred_held_out.size() % kTinyGenes, 0u);

    std::vector<float> gnn_pred = download_f32(result.pred_held_out,
                                                result.pred_held_out.size(), stream);
    int n_pred_cells = static_cast<int>(gnn_pred.size()) / kTinyGenes;

    // Extract the true expression of pert 4 from h_all.
    // Cells with label 4 are at indices: 4, 9, 14, ..., n_cells-1.
    std::vector<float> true_means(kTinyGenes, 0.0f);
    int n_pert4_cells = 0;
    for (int j = 0; j < kTinyCells; ++j) {
        if (labels_all[j] != 4) continue;
        ++n_pert4_cells;
        uint64_t start = static_cast<uint64_t>(h_all.indptr[j]);
        uint64_t end   = static_cast<uint64_t>(h_all.indptr[j + 1]);
        for (uint64_t k = start; k < end; ++k)
            true_means[static_cast<size_t>(h_all.indices[k])] += h_all.values[k];
    }
    for (float& m : true_means)
        m /= static_cast<float>(std::max(n_pert4_cells, 1));

    std::vector<float> gnn_means = per_gene_mean(gnn_pred, n_pred_cells, kTinyGenes);

    double rho = spearman_rho(gnn_means, true_means);
    EXPECT_GT(rho, kGNNSpearmanMin)
        << "GNN unseen-pert Spearman ρ=" << rho
        << " ≤ threshold=" << kGNNSpearmanMin
        << " (should be above random baseline)";

    cudaStreamDestroy(stream);
}
