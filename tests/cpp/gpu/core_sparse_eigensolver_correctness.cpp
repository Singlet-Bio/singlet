// SPDX-License-Identifier: MIT
// singlet/gpu/tests/core_sparse_eigensolver_correctness.cpp
//
// Correctness harness for core/sparse_eigensolver.h::top_k_eigsh_lobpcg
// (CYCLE-182 Phase D).
// Reference: scipy.sparse.linalg.eigsh (ARPACK backend).
//
// Tests:
//   1. TinyEigSymmetric         — n=40, K=5, tridiagonal; abs eigenvalue error <= 1e-4,
//                                 eigenvector cos-sim >= 0.9999 (non-degenerate).
//   2. ScaleSmokeMedium         — n=10000, K=15; scipy reference (.npy file on disk);
//                                 relative eigenvalue error <= 1e-4, cos-sim >= 0.999.
//   3. Determinism_BitIdentical — n=1000, K=10; same seed → rel_err == 0.0 on eigenvalues.
//   4. Convergence_OnConvergent — n=1000, K=10, well-conditioned ring;
//                                 iters_run < max_iter, converged == true.
//   5. KneeOversampleRobustness — n=500, K=10, 4 clustered eigenvalues;
//                                 all top-K eigenvectors recovered (cos-sim >= 0.99).
//
// Build: compiled as CUDA (LANGUAGE CUDA in CMakeLists.txt — kernel includes device code).
// Run:   --gtest_filter=SparseEigTest.*
//
// Test 2 dependency: run tests/refs/sparse_eig_reference.py first to produce
//   /tmp/singlet_gpu_sparse_eig_refs_tmp/sparse_eig_medium_ref.npz
// If the file is absent the test is skipped with a descriptive message.

#include <singlet/gpu/core/sparse_eigensolver.h>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Tolerance constants (explicit; never lower to make a test pass)
// =============================================================================

/// Test 1: TinyEigSymmetric
constexpr float kTiny_AbsEigvalErr  = 1e-4f;   // absolute eigenvalue error
constexpr float kTiny_CosSim        = 0.9999f;  // eigenvector cosine similarity

/// Test 2: ScaleSmokeMedium
constexpr float kMedium_RelEigvalErr = 1e-4f;   // relative eigenvalue error
constexpr float kMedium_CosSim       = 0.999f;  // eigenvector cosine similarity

/// Test 3: Determinism — exact reproducibility with same seed
constexpr float kDeterminism_RelErr = 0.0f;     // must be bit-identical

/// Test 4: Convergence
/// iters_run < max_iter  and  converged == true (no numeric threshold beyond that)

/// Test 5: KneeOversampleRobustness
constexpr float kKnee_CosSim = 0.99f;           // cos-sim even with clustered eigenvalues

// Fixed seed (Rule 11).
constexpr uint64_t kSeed = 0xC0FFEEull;

/// Path where the scipy reference npz for Test 2 is expected.
static const char* kMediumRefDir  = "/tmp/singlet_gpu_sparse_eig_refs_tmp";
static const char* kMediumRefFile =
    "/tmp/singlet_gpu_sparse_eig_refs_tmp/sparse_eig_medium_ref.npz";

// =============================================================================
// Shared helpers
// =============================================================================
namespace {

// -------------------------------------------------------------------------
// GPU availability guard.
// -------------------------------------------------------------------------
bool gpu_available()
{
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    return (e == cudaSuccess && count > 0);
}

// -------------------------------------------------------------------------
// Host <-> device transfer helpers.
// -------------------------------------------------------------------------
singlet::gpu::core::DeviceMemory<float> upload_float(const std::vector<float>& h)
{
    singlet::gpu::core::DeviceMemory<float> d(h.size());
    cudaMemcpy(d.get(), h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

singlet::gpu::core::DeviceMemory<int> upload_int(const std::vector<int>& h)
{
    singlet::gpu::core::DeviceMemory<int> d(h.size());
    cudaMemcpy(d.get(), h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> d2h_float(const singlet::gpu::core::DeviceMemory<float>& d, int n)
{
    std::vector<float> h(static_cast<size_t>(n));
    cudaMemcpy(h.data(), d.get(), static_cast<size_t>(n) * sizeof(float),
               cudaMemcpyDeviceToHost);
    return h;
}

// -------------------------------------------------------------------------
// Build a SparseSymmetricCSR on device from host COO triplets.
// Caller provides a symmetric CSR (row_ptr[n+1], col_idx[nnz], values[nnz]).
// -------------------------------------------------------------------------
singlet::gpu::core::SparseSymmetricCSR make_csr(
    int n, int nnz,
    const std::vector<float>& h_val,
    const std::vector<int>&   h_row,
    const std::vector<int>&   h_col)
{
    singlet::gpu::core::SparseSymmetricCSR A;
    A.n   = n;
    A.nnz = nnz;
    A._d_values  = upload_float(h_val);
    A._d_row_ptr = upload_int(h_row);
    A._d_col_idx = upload_int(h_col);
    A.values  = A._d_values.get();
    A.row_ptr = A._d_row_ptr.get();
    A.col_idx = A._d_col_idx.get();
    return A;
}

// -------------------------------------------------------------------------
// Build symmetric CSR for a tridiagonal matrix with given diagonal
// and off-diagonal value (symmetric, so upper+lower stored).
//
// For a symmetric matrix A = D + off * (sub + super) where D = diag(diag_vals).
// nnz = n + 2*(n-1) for the full symmetric pattern.
// -------------------------------------------------------------------------
singlet::gpu::core::SparseSymmetricCSR build_tridiagonal(
    int n,
    const std::vector<float>& diag_vals,  // length n
    float off_diag)
{
    // Build CSR row by row.
    std::vector<int>   row_ptr(static_cast<size_t>(n + 1), 0);
    std::vector<int>   col_idx;
    std::vector<float> values;
    col_idx.reserve(static_cast<size_t>(n + 2 * (n - 1)));
    values.reserve(static_cast<size_t>(n + 2 * (n - 1)));

    for (int r = 0; r < n; ++r) {
        row_ptr[static_cast<size_t>(r)] = static_cast<int>(col_idx.size());
        if (r > 0) {
            col_idx.push_back(r - 1);
            values.push_back(off_diag);
        }
        col_idx.push_back(r);
        values.push_back(diag_vals[static_cast<size_t>(r)]);
        if (r < n - 1) {
            col_idx.push_back(r + 1);
            values.push_back(off_diag);
        }
    }
    row_ptr[static_cast<size_t>(n)] = static_cast<int>(col_idx.size());
    const int nnz = static_cast<int>(col_idx.size());
    return make_csr(n, nnz, values, row_ptr, col_idx);
}

// -------------------------------------------------------------------------
// Build a random sparse symmetric matrix.
// Generates a symmetric adjacency-like matrix with ~sparsity fraction of
// off-diagonal non-zeros, adds a scaled identity to ensure positive definiteness.
// -------------------------------------------------------------------------
singlet::gpu::core::SparseSymmetricCSR build_random_sparse_symmetric(
    int n, float sparsity, float diag_shift, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> udist(0.f, 1.f);
    std::uniform_real_distribution<float> vdist(0.1f, 1.f);

    // Build upper triangular COO; then symmetrize.
    std::vector<std::pair<int,int>> edges;
    std::vector<float> edge_vals;
    for (int r = 0; r < n; ++r) {
        for (int c = r + 1; c < n; ++c) {
            if (udist(rng) < sparsity) {
                float v = vdist(rng);
                edges.emplace_back(r, c);
                edge_vals.push_back(v);
            }
        }
    }

    // Build CSR.
    std::vector<std::vector<std::pair<int, float>>> adj(static_cast<size_t>(n));
    for (int r = 0; r < n; ++r)
        adj[static_cast<size_t>(r)].emplace_back(r, diag_shift);  // diagonal
    for (size_t e = 0; e < edges.size(); ++e) {
        int r = edges[e].first, c = edges[e].second;
        float v = edge_vals[e];
        adj[static_cast<size_t>(r)].emplace_back(c, v);
        adj[static_cast<size_t>(c)].emplace_back(r, v);
    }
    for (auto& row : adj)
        std::sort(row.begin(), row.end());

    std::vector<int>   row_ptr(static_cast<size_t>(n + 1), 0);
    std::vector<int>   col_idx;
    std::vector<float> values;
    for (int r = 0; r < n; ++r) {
        row_ptr[static_cast<size_t>(r)] = static_cast<int>(col_idx.size());
        for (auto& [c, v] : adj[static_cast<size_t>(r)]) {
            col_idx.push_back(c);
            values.push_back(v);
        }
    }
    row_ptr[static_cast<size_t>(n)] = static_cast<int>(col_idx.size());
    const int nnz = static_cast<int>(col_idx.size());
    return make_csr(n, nnz, values, row_ptr, col_idx);
}

// -------------------------------------------------------------------------
// Cosine similarity between two unit-length-ish host vectors.
// Returns |cos_sim| (absorb eigenvector sign ambiguity).
// -------------------------------------------------------------------------
float abs_cos_sim(const float* a, const float* b, int n)
{
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na  += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb  += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (na < 1e-30 || nb < 1e-30) return 0.f;
    return static_cast<float>(std::fabs(dot) / std::sqrt(na * nb));
}

// -------------------------------------------------------------------------
// Minimal .npy v1.0 writer for float32 arrays (row-major 2-D or 1-D).
// -------------------------------------------------------------------------
static void write_npy_f32(const std::string& path, const float* data, int rows, int cols)
{
    std::string shape_str = (cols == 1)
        ? "(" + std::to_string(rows) + ",)"
        : "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")";
    std::string header_str =
        "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";

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
    const size_t n_items = static_cast<size_t>(rows) * (cols == 1 ? 1 : static_cast<size_t>(cols));
    fout.write(reinterpret_cast<const char*>(data),
               static_cast<std::streamsize>(n_items * sizeof(float)));
    if (!fout) throw std::runtime_error("write_npy_f32: write error " + path);
}

// -------------------------------------------------------------------------
// Load float32 array from a key inside a .npz via python3 one-liner.
// Returns empty vector and sets *ok=false if extraction fails.
// -------------------------------------------------------------------------
std::vector<float> load_npz_f32(const std::string& npz, const std::string& key,
                                  int expected_n, bool* ok)
{
    *ok = false;
    const std::string tmp = std::string(kMediumRefDir) + "/_extract_" + key + ".npy";
    std::string cmd = "python3 -c \""
        "import numpy as np; "
        "d = np.load('" + npz + "', allow_pickle=True); "
        "np.save('" + tmp + "', d['" + key + "'].astype(np.float32))"
        "\" 2>/dev/null";
    if (std::system(cmd.c_str()) != 0) return {};

    std::ifstream f(tmp, std::ios::binary);
    if (!f) return {};
    // Skip npy header: magic(6) + version(2) + hlen(2) + header
    char magic[8]; f.read(magic, 8); if (!f) return {};
    uint16_t hlen = 0; f.read(reinterpret_cast<char*>(&hlen), 2); if (!f) return {};
    f.ignore(hlen);
    std::vector<float> out(static_cast<size_t>(expected_n));
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(expected_n * sizeof(float)));
    if (!f && !f.eof()) return {};
    *ok = true;
    return out;
}

} // anonymous namespace


// =============================================================================
// GTest fixture
// =============================================================================
class SparseEigTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!gpu_available()) GTEST_SKIP() << "No CUDA GPU found — skipping.";
    }
};


// =============================================================================
// Test 1: TinyEigSymmetric
//
// n=40, K=5, tridiagonal matrix with analytically known eigenvalues.
// For the symmetric tridiagonal A = 2I + off*(sub + super), the eigenvalues are:
//   λ_j = 2 + 2*off_diag * cos(j*π/(n+1)),  j = 1..n  (descending from j=1 largest)
//
// Pass criterion:
//   abs eigenvalue error  <= 1e-4
//   eigenvector cos-sim   >= 0.9999 (non-degenerate; all eigenvalues distinct for n=40)
// =============================================================================
TEST_F(SparseEigTest, TinyEigSymmetric)
{
    constexpr int   N        = 40;
    constexpr int   K        = 5;
    constexpr float kDiag    = 2.0f;
    constexpr float kOffDiag = 1.0f;   // off-diagonal element

    // Build n=40 symmetric tridiagonal.
    std::vector<float> diag_vals(static_cast<size_t>(N), kDiag);
    auto A = build_tridiagonal(N, diag_vals, kOffDiag);

    singlet::gpu::core::SparseEigConfig<float> cfg;
    cfg.n_components = K;
    cfg.max_iter     = 300;
    cfg.tolerance    = 1e-7f;
    cfg.seed         = static_cast<uint64_t>(kSeed);

    auto result = singlet::gpu::core::top_k_eigsh_lobpcg(A, cfg, nullptr);
    ASSERT_TRUE(result.converged) << "LOBPCG should converge on n=40 tridiagonal";

    auto h_eig = d2h_float(result.eigenvalues, K);
    auto h_vec = d2h_float(result.eigenvectors, N * K);

    // Analytical eigenvalues in descending order (largest j first = largest λ).
    // λ_j = kDiag + 2*kOffDiag * cos(j*π/(N+1))  for j = 1..N.
    // Top K = largest K eigenvalues → j = 1..K (cos is decreasing from j=1 to N).
    std::vector<float> ref_eig(static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) {
        int j = k + 1;   // j=1 → largest eigenvalue
        ref_eig[static_cast<size_t>(k)] =
            kDiag + 2.0f * kOffDiag * std::cos(static_cast<float>(j) *
                static_cast<float>(M_PI) / static_cast<float>(N + 1));
    }

    // Analytical eigenvectors: v_k[i] = sin(i*j*π/(N+1)) (1-indexed), normalized.
    std::vector<float> ref_vec(static_cast<size_t>(N * K));
    for (int k = 0; k < K; ++k) {
        int j = k + 1;
        float norm = 0.f;
        for (int i = 1; i <= N; ++i) {
            float v = std::sin(static_cast<float>(i) * static_cast<float>(j) *
                               static_cast<float>(M_PI) / static_cast<float>(N + 1));
            ref_vec[static_cast<size_t>(i - 1 + k * N)] = v;
            norm += v * v;
        }
        norm = std::sqrt(norm);
        for (int i = 0; i < N; ++i)
            ref_vec[static_cast<size_t>(i + k * N)] /= norm;
    }

    // Check eigenvalues (absolute error).
    float max_abs_eig_err = 0.f;
    for (int k = 0; k < K; ++k) {
        float err = std::fabs(h_eig[static_cast<size_t>(k)] - ref_eig[static_cast<size_t>(k)]);
        max_abs_eig_err = std::max(max_abs_eig_err, err);
        EXPECT_LE(err, kTiny_AbsEigvalErr)
            << "Eigenvalue[" << k << "]: got " << h_eig[static_cast<size_t>(k)]
            << ", ref " << ref_eig[static_cast<size_t>(k)];
    }

    // Check eigenvectors (cos-sim, absorbing sign ambiguity).
    float min_cos = 1.f;
    for (int k = 0; k < K; ++k) {
        float cs = abs_cos_sim(h_vec.data() + k * N,
                               ref_vec.data() + k * N, N);
        min_cos = std::min(min_cos, cs);
        EXPECT_GE(cs, kTiny_CosSim)
            << "Eigenvector[" << k << "] cos-sim=" << cs << " < " << kTiny_CosSim;
    }

    std::printf("[REGISTRY] %s | core/sparse_eigensolver/tiny_eig_symmetric | n=%d K=%d | "
                "abs_eig_err | %.2e | %.2e | analytic | CYCLE-182 | %s\n",
                __DATE__, N, K, max_abs_eig_err, kTiny_AbsEigvalErr,
                (max_abs_eig_err <= kTiny_AbsEigvalErr) ? "PASS" : "FAIL");
    std::printf("[REGISTRY] %s | core/sparse_eigensolver/tiny_eig_symmetric | n=%d K=%d | "
                "min_vec_cos_sim | %.6f | %.4f | analytic | CYCLE-182 | %s\n",
                __DATE__, N, K, min_cos, kTiny_CosSim,
                (min_cos >= kTiny_CosSim) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 2: ScaleSmokeMedium
//
// n=10000, K=15.  Reference eigenvalues + eigenvectors from scipy.sparse.linalg.eigsh
// (ARPACK backend), pre-generated by tests/refs/sparse_eig_reference.py and stored
// at kMediumRefFile.  If the file is absent the test is skipped.
//
// Pass criterion:
//   relative eigenvalue error <= 1e-4
//   eigenvector cos-sim       >= 0.999
// =============================================================================
TEST_F(SparseEigTest, ScaleSmokeMedium)
{
    if (!fs::exists(kMediumRefFile)) {
        GTEST_SKIP()
            << "scipy reference not found at " << kMediumRefFile << ". "
            << "Run tests/refs/sparse_eig_reference.py first.";
    }

    constexpr int N = 10000;
    constexpr int K = 15;

    // Load reference eigenvalues and eigenvectors from npz.
    bool ok_eig = false, ok_vec = false;
    auto ref_eig = load_npz_f32(kMediumRefFile, "eigenvalues",  K,       &ok_eig);
    auto ref_vec = load_npz_f32(kMediumRefFile, "eigenvectors", N * K,   &ok_vec);
    if (!ok_eig || !ok_vec) {
        GTEST_SKIP() << "Failed to extract arrays from " << kMediumRefFile
                     << " — check scipy reference script output.";
    }

    // Load the sparse matrix CSR arrays written by the reference script.
    bool ok_val = false, ok_rp = false, ok_ci = false;
    // The reference script also writes the CSR arrays so both sides use the same A.
    auto h_val = load_npz_f32(kMediumRefFile, "csr_values",  -1,  &ok_val);
    // CSR int arrays: load via float extraction then reinterpret.
    // The reference script stores them as int32 but npy extraction via Python
    // casts to float32 for load_npz_f32. We load row_ptr and col_idx as int.
    // Use a dedicated Python one-liner for int arrays.
    auto load_npz_i32 = [&](const std::string& key) -> std::vector<int> {
        const std::string tmp = std::string(kMediumRefDir) + "/_extract_" + key + ".npy";
        std::string cmd = "python3 -c \""
            "import numpy as np; "
            "d = np.load('" + std::string(kMediumRefFile) + "', allow_pickle=True); "
            "np.save('" + tmp + "', d['" + key + "'].astype(np.int32))"
            "\" 2>/dev/null";
        if (std::system(cmd.c_str()) != 0) return {};
        std::ifstream f(tmp, std::ios::binary);
        if (!f) return {};
        char magic[8]; f.read(magic, 8);
        uint16_t hlen = 0; f.read(reinterpret_cast<char*>(&hlen), 2);
        f.ignore(hlen);
        // Read rest.
        f.seekg(0, std::ios::end);
        long total = static_cast<long>(f.tellg()) - 10 - hlen;
        f.seekg(10 + hlen, std::ios::beg);
        std::vector<int> out(static_cast<size_t>(total / 4));
        f.read(reinterpret_cast<char*>(out.data()), total);
        return out;
    };

    auto h_row = load_npz_i32("csr_row_ptr");
    auto h_col = load_npz_i32("csr_col_idx");

    if (!ok_val || h_row.empty() || h_col.empty()) {
        GTEST_SKIP() << "Failed to load CSR arrays from " << kMediumRefFile
                     << " — re-run sparse_eig_reference.py.";
    }

    const int nnz = static_cast<int>(h_val.size());
    auto A = make_csr(N, nnz, h_val, h_row, h_col);

    singlet::gpu::core::SparseEigConfig<float> cfg;
    cfg.n_components = K;
    cfg.max_iter     = 300;
    cfg.tolerance    = 1e-6f;
    cfg.seed         = 42;

    auto result = singlet::gpu::core::top_k_eigsh_lobpcg(A, cfg, nullptr);

    auto h_eig = d2h_float(result.eigenvalues, K);
    auto h_vec = d2h_float(result.eigenvectors, N * K);

    // Eigenvalue relative error.
    float max_rel_err = 0.f;
    for (int k = 0; k < K; ++k) {
        float ref = ref_eig[static_cast<size_t>(k)];
        float err = std::fabs(h_eig[static_cast<size_t>(k)] - ref) /
                    std::max(std::fabs(ref), 1e-9f);
        max_rel_err = std::max(max_rel_err, err);
        EXPECT_LE(err, kMedium_RelEigvalErr)
            << "Eigenvalue[" << k << "]: got " << h_eig[static_cast<size_t>(k)]
            << ", ref " << ref;
    }

    // Eigenvector cos-sim.
    float min_cos = 1.f;
    for (int k = 0; k < K; ++k) {
        float cs = abs_cos_sim(h_vec.data() + k * N,
                               ref_vec.data() + k * N, N);
        min_cos = std::min(min_cos, cs);
        EXPECT_GE(cs, kMedium_CosSim)
            << "Eigenvector[" << k << "] cos-sim=" << cs;
    }

    std::printf("[REGISTRY] %s | core/sparse_eigensolver/scale_smoke_medium | n=%d K=%d | "
                "rel_eig_err | %.2e | %.2e | scipy_arpack | CYCLE-182 | %s\n",
                __DATE__, N, K, max_rel_err, kMedium_RelEigvalErr,
                (max_rel_err <= kMedium_RelEigvalErr) ? "PASS" : "FAIL");
    std::printf("[REGISTRY] %s | core/sparse_eigensolver/scale_smoke_medium | n=%d K=%d | "
                "min_vec_cos_sim | %.6f | %.4f | scipy_arpack | CYCLE-182 | %s\n",
                __DATE__, N, K, min_cos, kMedium_CosSim,
                (min_cos >= kMedium_CosSim) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 3: Determinism_BitIdentical
//
// n=1000, K=10, fixed seed=42.  Two runs with the same sparse matrix and config
// must produce identical eigenvalues (rel_err == 0.0).
//
// (Eigenvectors may differ in sign per column but their norms and eigenvalues
// must match exactly under deterministic mode.)
// =============================================================================
TEST_F(SparseEigTest, Determinism_BitIdentical)
{
    constexpr int N = 1000;
    constexpr int K = 10;

    auto A1 = build_random_sparse_symmetric(N, 0.01f, 10.f, kSeed);
    auto A2 = build_random_sparse_symmetric(N, 0.01f, 10.f, kSeed);

    singlet::gpu::core::SparseEigConfig<float> cfg;
    cfg.n_components  = K;
    cfg.max_iter      = 200;
    cfg.tolerance     = 1e-6f;
    cfg.seed          = 42;
    cfg.deterministic = true;

    auto r1 = singlet::gpu::core::top_k_eigsh_lobpcg(A1, cfg, nullptr);
    auto r2 = singlet::gpu::core::top_k_eigsh_lobpcg(A2, cfg, nullptr);

    auto h_eig1 = d2h_float(r1.eigenvalues, K);
    auto h_eig2 = d2h_float(r2.eigenvalues, K);

    float max_rel_err = 0.f;
    int   mismatches  = 0;
    for (int k = 0; k < K; ++k) {
        const float denom = std::max(std::fabs(h_eig1[static_cast<size_t>(k)]), 1e-9f);
        const float err = std::fabs(h_eig1[static_cast<size_t>(k)] -
                                    h_eig2[static_cast<size_t>(k)]) / denom;
        if (err > 0.f) {
            ++mismatches;
            max_rel_err = std::max(max_rel_err, err);
        }
    }

    EXPECT_EQ(mismatches, 0)
        << "Determinism: " << mismatches << " eigenvalue(s) differ; "
        << "max_rel_err=" << max_rel_err;

    // Also verify eigenvector norms are consistent (should be 1 for orthonormal output).
    auto h_vec1 = d2h_float(r1.eigenvectors, N * K);
    auto h_vec2 = d2h_float(r2.eigenvectors, N * K);
    for (int k = 0; k < K; ++k) {
        // Compare |cos-sim| — should be 1.0 (same column, possibly sign-flipped).
        float cs = abs_cos_sim(h_vec1.data() + k * N, h_vec2.data() + k * N, N);
        EXPECT_GE(cs, 0.9999f)
            << "Eigenvector[" << k << "] from two identical runs has cos-sim=" << cs;
    }

    std::printf("[REGISTRY] %s | core/sparse_eigensolver/determinism | n=%d K=%d | "
                "rel_err | max=%.2e | 0.0 | self | CYCLE-182 | %s\n",
                __DATE__, N, K, max_rel_err,
                (mismatches == 0) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 4: Convergence_OnConvergent
//
// n=1000, K=10, well-conditioned ring-like sparse matrix with distinct
// eigenvalues (no degeneracy, diagonal shift >> off-diagonal).
//
// Pass criterion:
//   result.iters_run < cfg.max_iter
//   result.converged == true
// =============================================================================
TEST_F(SparseEigTest, Convergence_OnConvergent)
{
    constexpr int N       = 1000;
    constexpr int K       = 10;
    constexpr int MAX_IT  = 200;

    // Tridiagonal with large diagonal shift → large spectral gap → fast convergence.
    std::vector<float> diag_vals(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        diag_vals[static_cast<size_t>(i)] = 20.f + static_cast<float>(i) * 0.01f;
    auto A = build_tridiagonal(N, diag_vals, 1.f);

    singlet::gpu::core::SparseEigConfig<float> cfg;
    cfg.n_components = K;
    cfg.max_iter     = MAX_IT;
    cfg.tolerance    = 1e-6f;
    cfg.seed         = 42;

    auto result = singlet::gpu::core::top_k_eigsh_lobpcg(A, cfg, nullptr);

    EXPECT_TRUE(result.converged)
        << "LOBPCG should converge on well-conditioned tridiagonal (n=" << N << ")";
    EXPECT_LT(result.iters_run, MAX_IT)
        << "iters_run=" << result.iters_run << " should be < max_iter=" << MAX_IT;

    // Sanity: eigenvalues should be positive (diag >> off-diag by construction).
    auto h_eig = d2h_float(result.eigenvalues, K);
    for (int k = 0; k < K; ++k) {
        EXPECT_GT(h_eig[static_cast<size_t>(k)], 0.f)
            << "Eigenvalue[" << k << "]=" << h_eig[static_cast<size_t>(k)]
            << " should be positive";
    }

    std::printf("[REGISTRY] %s | core/sparse_eigensolver/convergence | n=%d K=%d | "
                "iters_run | %d | <%d | self | CYCLE-182 | %s\n",
                __DATE__, N, K, result.iters_run, MAX_IT,
                (result.converged && result.iters_run < MAX_IT) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 5: KneeOversampleRobustness
//
// n=500, K=10.  Hand-built matrix with 4 clustered eigenvalues near the "knee"
// of the spectrum (eigenvalues 7-10 are all ≈ 5.0, a typical LOBPCG stress case).
// Tests that the block-size oversample keeps all top-K eigenvectors recoverable
// even with degeneracy.
//
// Pass criterion:
//   For each of the top-K returned eigenvectors, the best cos-sim match among
//   the reference eigenvectors (from a 2× oversample run) is >= 0.99.
//   This verifies the clustered block is spanned even without exact pairing.
// =============================================================================
TEST_F(SparseEigTest, KneeOversampleRobustness)
{
    constexpr int N = 500;
    constexpr int K = 10;

    // Build a block-diagonal matrix:
    //   - First 6 blocks of size 1 with eigenvalues 10, 9, 8, 7, 6, 5.5 (non-clustered).
    //   - One block of size 4 with all eigenvalues ≈ 5.0 (clustered).
    //   - Remaining 490 rows with eigenvalue 1.0.
    //
    // Implemented as: A = diag(prescribed_spectrum), stored as CSR diagonal.
    // The eigenvectors are the standard basis vectors (easy to check).

    std::vector<float> spectrum(static_cast<size_t>(N), 1.0f);
    spectrum[0] = 10.f;
    spectrum[1] =  9.f;
    spectrum[2] =  8.f;
    spectrum[3] =  7.f;
    spectrum[4] =  6.f;
    spectrum[5] =  5.5f;
    spectrum[6] =  5.01f;   // clustered block, 4 eigenvalues ≈ 5
    spectrum[7] =  5.005f;
    spectrum[8] =  5.002f;
    spectrum[9] =  5.001f;

    // Build CSR for pure diagonal.
    std::vector<int>   row_ptr(static_cast<size_t>(N + 1));
    std::vector<int>   col_idx(static_cast<size_t>(N));
    std::vector<float> values(static_cast<size_t>(N));
    for (int i = 0; i <= N; ++i) row_ptr[static_cast<size_t>(i)] = i;
    for (int i = 0; i < N; ++i) {
        col_idx[static_cast<size_t>(i)] = i;
        values[static_cast<size_t>(i)]  = spectrum[static_cast<size_t>(i)];
    }
    auto A = make_csr(N, N, values, row_ptr, col_idx);

    // Reference run with 2× oversample (K + K extra components).
    singlet::gpu::core::SparseEigConfig<float> cfg_ref;
    cfg_ref.n_components = K + K;   // oversample to ensure clustered block is spanned
    cfg_ref.max_iter     = 300;
    cfg_ref.tolerance    = 1e-7f;
    cfg_ref.seed         = 42;
    auto ref = singlet::gpu::core::top_k_eigsh_lobpcg(A, cfg_ref, nullptr);
    auto h_ref_vec = d2h_float(ref.eigenvectors, N * (K + K));

    // Production run with exactly K components.
    singlet::gpu::core::SparseEigConfig<float> cfg;
    cfg.n_components = K;
    cfg.max_iter     = 300;
    cfg.tolerance    = 1e-7f;
    cfg.seed         = 42;
    auto result = singlet::gpu::core::top_k_eigsh_lobpcg(A, cfg, nullptr);
    auto h_vec = d2h_float(result.eigenvectors, N * K);

    // For each of the top-K production eigenvectors, find the best-match in the
    // reference 2K-oversample set. Best match must be >= kKnee_CosSim.
    float min_best_cos = 1.f;
    for (int k = 0; k < K; ++k) {
        float best = 0.f;
        for (int r = 0; r < K + K; ++r) {
            float cs = abs_cos_sim(h_vec.data() + k * N,
                                   h_ref_vec.data() + r * N, N);
            best = std::max(best, cs);
        }
        min_best_cos = std::min(min_best_cos, best);
        EXPECT_GE(best, kKnee_CosSim)
            << "Eigenvector[" << k << "]: best cos-sim to any reference vec = " << best
            << " < " << kKnee_CosSim << " (clustered spectrum may have caused dropout)";
    }

    // Eigenvalues should cover the correct range.
    auto h_eig = d2h_float(result.eigenvalues, K);
    EXPECT_GE(h_eig[0], 9.f) << "Top eigenvalue should be near 10; got " << h_eig[0];
    EXPECT_GE(h_eig[K - 1], 4.9f)
        << "K-th eigenvalue should be near 5; got " << h_eig[K - 1];

    std::printf("[REGISTRY] %s | core/sparse_eigensolver/knee_oversample | n=%d K=%d | "
                "min_best_cos_sim | %.6f | %.2f | self_oversample | CYCLE-182 | %s\n",
                __DATE__, N, K, min_best_cos, kKnee_CosSim,
                (min_best_cos >= kKnee_CosSim) ? "PASS" : "FAIL");
}
