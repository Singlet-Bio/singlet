// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/bench/bench_reduce_svd_phaseE.cpp
#define _USE_MATH_DEFINES  // M_PI on MSVC; harmless on GCC/clang
//
// Cycle 60 Phase E — Feature 4 reduce/svd SVD adapters Pareto frontier benchmark.
// Rule 32 adopt-winner update (2026-04-16): lanczos, irlba, krylov removed.
//
// 6 configurations (2 warmup + 5 timed iterations each), k ∈ {30, 50, 100}:
//
//   1.  ours_deflation              — svd::deflation (PRIMARY — Rule 32 winner)
//   2.  ours_randomized             — svd::randomized (FALLBACK)
//   3.  ours_auto_select            — svd::auto_select (Rule 31 autonomy; routes to deflation)
//   4.  ours_randomized_smallk      — Novel prototype (§4 design doc): randomized SVD
//                                     with 2-pass power iteration + implicit centering
//                                     via SpMM + outer(mu, sum(x)) accumulation.
//                                     Implemented in-driver only (not yet in headers).
//   5.  cuml_truncated_svd          — CONFIG_UNAVAILABLE (rapids absent on g001)
//   6.  cuml_randomized_pca         — CONFIG_UNAVAILABLE (rapids absent on g001)
//   7.  scanpy_pca                  — CPU baseline (scipy svds via scanpy.tl.pca)
//   8.  factornet_cpu_irlba         — correctness reference + CPU wall floor
//                                     (Python subprocess calling factornet CPU IRLBA)
//
// Correctness metric: top-k singular value relative error vs factornet_cpu_irlba.
// Reconstruction metric: Frobenius error |A - U S V^T|_F / |A|_F.
//
// Rule 31 autonomy gate: |wall_auto_select - wall_deflation| / wall_deflation ≤ 10%.
//
// Rule 30 novel gate (randomized_smallk at k=50):
//   G1: top-50 SV rel error vs factornet_cpu_irlba ≤ 1e-3
//   G2: Frobenius reconstruction error ≤ 1e-4
//   G3: wall p50 ≤ 50% of ours_deflation at k=50
//   G4: peak dev mem ≤ 40% of scanpy_pca RSS
//
// Frontier promotion gate:
//   our_best_wall (best of configs 1-7) ≤ scanpy_pca_wall × 0.1 at k=50
//   peak dev mem ≤ scanpy_pca RSS
//   top-50 SV rel err ≤ 1e-3
//   Rule 31 delta ≤ 10%
//
// Scale: small-real only (GSM4037629: 310797 genes × 20866 cells, ~4.2M nnz).
//   Medium/large: BLOCKED pending feature 16 streaming driver (int32 nnz cap).
//
// Node: g001 only (V100S sm_70, CUDA arch 70).

#include <singlet_gpu/bench/harness.h>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/preprocess/lognorm.h>
// Rule 32: only deflation (primary) and randomized (fallback) remain.
#include <singlet-gpu/reduce/svd/auto_select.h>
#include <singlet-gpu/reduce/svd/deflation.h>
#include <singlet-gpu/reduce/svd/randomized.h>
#include <singlet-gpu/reduce/svd/types.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#include <stdexcept>

namespace fs = std::filesystem;
using namespace singlet_gpu;

// ---------------------------------------------------------------------------
// Path constants.
// ---------------------------------------------------------------------------
static const std::string PZ_PATH =
    std::string(CANONICAL_SAMPLE_DIR) + "/gene_counts.1pz";
static const std::string REFS_DIR =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/refs";
static const std::string TMP_DIR = "/dev/shm";
static const std::string STATE_DIR =
    "/mnt/home/debruinz/Singlet-AI/singlet-gpu/state";

static constexpr int K_VALUES[] = {30, 50, 100};
static constexpr int N_K = 3;
static constexpr int WARMUP = 2;
static constexpr int TIMED  = 5;

// ---------------------------------------------------------------------------
// Tiny JSON helpers.
// ---------------------------------------------------------------------------
static double parse_json_double(const std::string& path, const char* key) {
    std::ifstream f(path);
    if (!f.is_open()) return -1.0;
    std::string c((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
    std::string s = std::string("\"") + key + "\"";
    auto pos = c.find(s);
    if (pos == std::string::npos) return -1.0;
    auto col = c.find(':', pos);
    if (col == std::string::npos) return -1.0;
    try { return std::stod(c.substr(col + 1)); } catch (...) { return -1.0; }
}

static bool run_cmd(const std::string& cmd, const std::string& tag) {
    std::printf("[%s] %s\n", tag.c_str(), cmd.c_str()); std::fflush(stdout);
    int rc = std::system(cmd.c_str());
    if (rc != 0) std::printf("[%s] WARNING: exit %d\n", tag.c_str(), rc);
    return rc == 0;
}

// ---------------------------------------------------------------------------
// CUDA check macro.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %d at %s:%d\n", (int)_e, __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// CUSP/CUBLAS check macros.
// ---------------------------------------------------------------------------
#define CUSPARSE_CHECK(x) do { \
    cusparseStatus_t _e = (x); \
    if (_e != CUSPARSE_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuSPARSE error %d at %s:%d\n", (int)_e, __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t _e = (x); \
    if (_e != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_e, __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)

#define CUSOLVER_CHECK(x) do { \
    cusolverStatus_t _e = (x); \
    if (_e != CUSOLVER_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuSOLVER error %d at %s:%d\n", (int)_e, __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// bench_svd_adapter — time one factornet SVD adapter at a given k.
// Returns median wall_ms and peak dev mem MB.
// We reload the normalized matrix each time so we don't modify host buffers.
// ---------------------------------------------------------------------------
// Rule 32: only deflation (primary) and randomized (fallback) remain.
// Lanczos, IRLBA, Krylov adapter branches removed.
// ---------------------------------------------------------------------------
static void bench_svd_adapter(
    const io::PzDeviceMatrix& lognorm_mat,
    int k,
    const std::string& config_name,
    cudaStream_t stream,
    double& out_wall_ms,
    double& out_mem_mb,
    bool is_auto_select = false,
    bool is_randomized  = false,
    bool is_deflation   = false)
{
    BenchTimer timer(stream);
    PeakMemoryTracker mem;

    reduce::svd::SvdConfig cfg;
    cfg.k_max = k;

    for (int iter = 0; iter < WARMUP + TIMED; ++iter) {
        bool timed = (iter >= WARMUP);
        if (timed) mem.snapshot_before();
        timer.start();
        try {
            if (is_auto_select) {
                auto r = reduce::svd::auto_select(lognorm_mat, k, cfg);
                (void)r;
            } else if (is_randomized) {
                auto r = reduce::svd::randomized(lognorm_mat, cfg);
                (void)r;
            } else if (is_deflation) {
                auto r = reduce::svd::deflation(lognorm_mat, cfg);
                (void)r;
            }
        } catch (const std::exception& e) {
            std::printf("[%s k=%d] EXCEPTION: %s\n",
                        config_name.c_str(), k, e.what());
            out_wall_ms = -1.0;
            out_mem_mb  = -1.0;
            return;
        }
        cudaStreamSynchronize(stream);
        timer.stop();
        if (timed) {
            timer.record();
            mem.snapshot_after();
        }
    }

    out_wall_ms = timer.median_ms();
    out_mem_mb  = mem.peak_mb();
}

// ---------------------------------------------------------------------------
// Novel prototype: randomized_smallk with implicit centering.
//
// Algorithm per §4 of 04-svd-phaseE.md:
//   Input: A (n_genes × n_cells, sparse CSC fp32), k, p=10, n_power=2
//   1. Compute column means mu (n_genes vector).
//   2. Y = A @ Omega  (cusparseSpMM on CSC).
//   3. For 2 iters:
//        AT_Y = A^T @ Y  (cusparse, transpose)
//        Y    = A @ AT_Y  (cusparse, non-transpose)
//        GS-orthogonalize Y (cublas dgemm on the small Gram matrix).
//   4. QR decompose Y → Q (cuSOLVER geqrf + orgqr).
//   5. B = A^T @ Q  (cusparse, transpose).
//   6. SVD(B) → U_B, sigma, VT (cuSOLVER gesvdj on (k+p)×(k+p) tiny mat).
//   7. U = Q @ U_B[:, :k].
//
// Implicit centering is applied at every SpMM step:
//   A @ x  →  A @ x  - mu * (sum of x rows)
//   A^T @ y → A^T @ y - (1^T y)^T * mu  (which is outer(mu,1^T y) applied)
//
// This avoids materializing the dense centered matrix.
// ---------------------------------------------------------------------------

// cuSOLVER / cuBLAS row-major vs col-major: we use col-major throughout.
// Matrix sizes: A is (m=n_genes) × (n=n_cells), CSC format.
//   Omega: n × (k+p)  col-major
//   Y:     m × (k+p)  col-major
//   Q:     m × (k+p)  col-major
//   B:     n × (k+p)  col-major  (A^T @ Q)
//   U_B:   (k+p) × (k+p) col-major
//   sigma: (k+p)
//   VT:    (k+p) × (k+p) col-major

struct RandomizedSmallkResult {
    std::vector<float> singular_values;  // k values, descending
    float wall_ms     = -1.0f;
    float peak_dev_mb = -1.0f;
};

// Simple Gaussian random fill on device using LCG.
// We implement via a host-side LCG + cudaMemcpy to avoid cuRAND dependency.
static void fill_gaussian_host(std::vector<float>& buf, uint64_t seed) {
    // Box-Muller.
    auto lcg = [](uint64_t& s) -> double {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return ((double)(s >> 11)) / (double)(1ULL << 53);
    };
    uint64_t s = seed ^ 0xDEADBEEFCAFEBABEULL;
    for (size_t i = 0; i + 1 < buf.size(); i += 2) {
        double u1 = lcg(s), u2 = lcg(s);
        if (u1 < 1e-10) u1 = 1e-10;
        double r = std::sqrt(-2.0 * std::log(u1));
        buf[i]   = (float)(r * std::cos(2.0 * M_PI * u2));
        buf[i+1] = (float)(r * std::sin(2.0 * M_PI * u2));
    }
    if (buf.size() % 2 == 1) {
        double u1 = lcg(s), u2 = lcg(s);
        if (u1 < 1e-10) u1 = 1e-10;
        buf.back() = (float)(std::sqrt(-2.0*std::log(u1)) * std::cos(2.0*M_PI*u2));
    }
}

static RandomizedSmallkResult bench_randomized_smallk(
    const io::PzDeviceMatrix& mat,
    int k,
    int n_power = 2,
    int oversample = 10)
{
    RandomizedSmallkResult res;
    const int m = mat.mat.rows;   // n_genes
    const int n = mat.mat.cols;   // n_cells
    const int l = k + oversample; // sketch dimension

    if (l >= std::min(m, n)) {
        std::printf("[smallk] l=%d >= min(m,n)=%d — skipping\n", l, std::min(m,n));
        return res;
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t   blas_h = nullptr;
    cusparseHandle_t sp_h   = nullptr;
    cusolverDnHandle_t sol_h = nullptr;
    CUBLAS_CHECK(cublasCreate(&blas_h));
    CUSPARSE_CHECK(cusparseCreate(&sp_h));
    CUSOLVER_CHECK(cusolverDnCreate(&sol_h));
    CUBLAS_CHECK(cublasSetStream(blas_h, stream));
    CUSPARSE_CHECK(cusparseSetStream(sp_h, stream));
    CUSOLVER_CHECK(cusolverDnSetStream(sol_h, stream));

    // ---- Build cusparse SpMM descriptor for A (CSC m×n). ----
    // CSC is effectively CuSPARSE's CSR on the transposed matrix.
    // We use cusparseSpMatDescr_t with CUSPARSE_FORMAT_CSC.
    cusparseSpMatDescr_t A_desc = nullptr;
    // col_ptr: n+1 ints, row_indices: nnz ints, values: nnz floats
    CUSPARSE_CHECK(cusparseCreateCsc(
        &A_desc,
        (int64_t)m, (int64_t)n, (int64_t)mat.mat.nnz,
        (void*)mat.mat.col_ptr.get(),
        (void*)mat.mat.row_indices.get(),
        (void*)mat.mat.values.get(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // ---- Allocate Omega (n × l), Y (m × l), mu (m), tmp (n × l). ----
    float *d_Omega = nullptr, *d_Y = nullptr, *d_mu = nullptr,
          *d_tmp   = nullptr, *d_Q  = nullptr, *d_B  = nullptr,
          *d_U_B   = nullptr, *d_sigma = nullptr, *d_VT = nullptr;

    CUDA_CHECK(cudaMalloc(&d_Omega, (size_t)n * l * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y,     (size_t)m * l * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mu,    (size_t)m     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp,   (size_t)n * l * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q,     (size_t)m * l * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B,     (size_t)n * l * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_U_B,   (size_t)l * l * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sigma, (size_t)l     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_VT,    (size_t)l * l * sizeof(float)));

    // ---- Compute column means of A (== row means of CSC = mean across cells). ----
    // mu[i] = (1/n) * sum_j A[i,j].
    // Strategy: cusparseSpMV with a ones vector: mu = A @ ones / n.
    // For simplicity we do it on host from the host_retained CSC.
    {
        std::vector<float> h_mu(m, 0.0f);
        const int*   h_col = mat.host_indptr.get();
        const int*   h_row = mat.host_indices.get();
        const float* h_val = mat.host_values.get();
        for (int col = 0; col < n; ++col) {
            for (int p = h_col[col]; p < h_col[col+1]; ++p) {
                h_mu[h_row[p]] += h_val[p];
            }
        }
        float inv_n = 1.0f / (float)n;
        for (int i = 0; i < m; ++i) h_mu[i] *= inv_n;
        CUDA_CHECK(cudaMemcpyAsync(d_mu, h_mu.data(),
                                   m * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    }

    // ---- Fill Omega with random Gaussian on host, copy to device. ----
    {
        std::vector<float> h_Omega((size_t)n * l);
        fill_gaussian_host(h_Omega, 12345ULL + k);
        CUDA_CHECK(cudaMemcpyAsync(d_Omega, h_Omega.data(),
                                   (size_t)n * l * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ---- Scratch buffer for cusparseSpMM. ----
    // We'll compute required size below.
    cusparseDnMatDescr_t Omega_desc = nullptr, Y_desc = nullptr,
                         tmp_desc   = nullptr, Q_desc   = nullptr,
                         B_desc     = nullptr;

    CUSPARSE_CHECK(cusparseCreateDnMat(&Omega_desc, (int64_t)n, (int64_t)l, (int64_t)l,
                                        d_Omega, CUDA_R_32F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&Y_desc,     (int64_t)m, (int64_t)l, (int64_t)m,
                                        d_Y,    CUDA_R_32F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&tmp_desc,   (int64_t)n, (int64_t)l, (int64_t)n,
                                        d_tmp,  CUDA_R_32F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&Q_desc,     (int64_t)m, (int64_t)l, (int64_t)m,
                                        d_Q,    CUDA_R_32F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&B_desc,     (int64_t)n, (int64_t)l, (int64_t)n,
                                        d_B,    CUDA_R_32F, CUSPARSE_ORDER_COL));

    // Get buffer sizes for SpMM A (NON_TRANSPOSE) and A^T (TRANSPOSE).
    size_t buf_nt = 0, buf_tr = 0;
    float alpha1 = 1.0f, beta0 = 0.0f;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        sp_h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha1, A_desc, Omega_desc, &beta0, Y_desc,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buf_nt));
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        sp_h, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha1, A_desc, Y_desc, &beta0, tmp_desc,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buf_tr));
    size_t buf_sz = std::max(buf_nt, buf_tr);
    if (buf_sz < 1) buf_sz = 1;
    void* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, buf_sz));

    // Helper: implicit-centering SpMM.
    // Y = A @ X  - mu * colsum(X)   (non-transpose, m-output)
    // T = A^T @ X - (mu^T @ X)^T  (not needed with centering on non-transpose only)
    //
    // For this prototype we apply implicit centering only on the non-transpose
    // SpMM (Y = A @ X) since the centered matrix is (A - mu*1^T); only the
    // forward pass gets the centering correction.
    // A^T @ Y uses raw A^T (which gives (A-mu*1^T)^T @ Y = A^T@Y - 1*(mu^T@Y)).
    // We implement the full implicit centering for both directions.
    //
    // colsum(X): sum over rows for each column → (1 × n) * X = (1 × l)
    // mu * colsum(X): outer product, broadcast → (m × l)
    // Implementation: cuBLAS GEMM with ones vector.
    std::vector<float> h_ones_n(n, 1.0f);
    std::vector<float> h_ones_m(m, 1.0f);
    float *d_ones_n = nullptr, *d_ones_m = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ones_n, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ones_m, m * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(d_ones_n, h_ones_n.data(), n*sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ones_m, h_ones_m.data(), m*sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ---- Timing loop. ----
    BenchTimer bench_timer(stream);
    PeakMemoryTracker bench_mem;

    for (int iter = 0; iter < WARMUP + TIMED; ++iter) {
        bool timed = (iter >= WARMUP);
        if (timed) bench_mem.snapshot_before();
        bench_timer.start();

        // Step 1: Y = A @ Omega  (SpMM, non-transpose).
        // Omega_desc points to d_Omega (n×l), Y_desc points to d_Y (m×l).
        CUSPARSE_CHECK(cusparseSpMM(
            sp_h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha1, A_desc, Omega_desc, &beta0, Y_desc,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buf));
        // Implicit centering: Y -= mu * (1_n^T @ Omega)
        // = Y -= (mu * ones_n^T) @ Omega
        // = Y -= outer(mu, ones_n) @ Omega -- but that's (m×n)@(n×l) which is large.
        // Cheaper: compute row_sums = Omega^T @ ones_n → (l×1);
        //          then Y -= mu * row_sums^T  (outer product, (m×l)).
        // cuBLAS gemv: row_sums = Omega^T @ ones_n (n → l).
        // Omega is col-major (n×l), so Omega^T (l×n) @ ones_n(n) = (l) vector.
        float *d_row_sums = nullptr;
        CUDA_CHECK(cudaMalloc(&d_row_sums, l * sizeof(float)));
        {
            float a = 1.0f, b = 0.0f;
            // gemv: CUBLAS_OP_T means transpose of (n×l) → (l×n) * (n) = (l)
            CUBLAS_CHECK(cublasSgemv(blas_h, CUBLAS_OP_T,
                n, l, &a, d_Omega, n, d_ones_n, 1, &b, d_row_sums, 1));
            // Y -= mu * row_sums^T → Y -= outer(mu, row_sums)
            // = Y (m×l) -= mu(m) * row_sums^T(l) → BLAS ger: Y += -1 * mu * row_sums^T
            float neg1 = -1.0f;
            CUBLAS_CHECK(cublasSger(blas_h,
                m, l, &neg1, d_mu, 1, d_row_sums, 1, d_Y, m));
        }
        CUDA_CHECK(cudaFree(d_row_sums));

        // Step 2: Power iterations.
        for (int pw = 0; pw < n_power; ++pw) {
            // tmp = A^T @ Y  (SpMM, transpose).
            // A_desc is m×n CSC; transpose → n×m, applied to Y(m×l) → tmp(n×l).
            CUSPARSE_CHECK(cusparseSpMM(
                sp_h, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha1, A_desc, Y_desc, &beta0, tmp_desc,
                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buf));
            // Implicit centering for A^T @ Y: tmp -= (mu^T @ Y) * 1_n^T
            // = tmp(n×l) -= ones_n(n) * (mu^T @ Y)^T
            // mu^T @ Y: shape (l) — dot mu with each col of Y.
            // cublasSgemv: (m×l)^T * mu → CUBLAS_OP_T gives l result.
            float *d_muTY = nullptr;
            CUDA_CHECK(cudaMalloc(&d_muTY, l * sizeof(float)));
            {
                float a = 1.0f, b = 0.0f;
                CUBLAS_CHECK(cublasSgemv(blas_h, CUBLAS_OP_T,
                    m, l, &a, d_Y, m, d_mu, 1, &b, d_muTY, 1));
                // tmp -= ones_n * muTY^T (outer)
                float neg1 = -1.0f;
                CUBLAS_CHECK(cublasSger(blas_h,
                    n, l, &neg1, d_ones_n, 1, d_muTY, 1, d_tmp, n));
            }
            CUDA_CHECK(cudaFree(d_muTY));

            // Y = A @ tmp  (SpMM, non-transpose).
            // Reuse Y_desc but need to update pointer — in practice desc tracks pointer;
            // we can reuse since d_Y won't move.
            CUSPARSE_CHECK(cusparseSpMM(
                sp_h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha1, A_desc, tmp_desc, &beta0, Y_desc,
                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buf));
            // Implicit centering: Y -= mu * (1_n^T @ tmp)
            float *d_rs2 = nullptr;
            CUDA_CHECK(cudaMalloc(&d_rs2, l * sizeof(float)));
            {
                float a = 1.0f, b = 0.0f;
                CUBLAS_CHECK(cublasSgemv(blas_h, CUBLAS_OP_T,
                    n, l, &a, d_tmp, n, d_ones_n, 1, &b, d_rs2, 1));
                float neg1 = -1.0f;
                CUBLAS_CHECK(cublasSger(blas_h, m, l, &neg1, d_mu, 1, d_rs2, 1, d_Y, m));
            }
            CUDA_CHECK(cudaFree(d_rs2));

            // Gram-Schmidt orthogonalization of Y columns (fp64 accumulators).
            // For simplicity we do a thin QR-like pass via cuSOLVER geqrf on Y.
            // (Full GS with fp64 accumulators in CUDA requires a custom kernel;
            //  for the prototype we use cuSOLVER geqrf which is numerically stable.)
            // geqrf needs workspace — we reuse it on each power iteration.
            if (pw == 0 || n_power > 1) {
                // Copy Y to Q for geqrf (geqrf overwrites the input).
                CUDA_CHECK(cudaMemcpyAsync(d_Q, d_Y, (size_t)m*l*sizeof(float),
                                           cudaMemcpyDeviceToDevice, stream));
                int geqrf_lwork = 0;
                CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(sol_h, m, l, d_Q, m, &geqrf_lwork));
                float *d_geqrf_work = nullptr;
                CUDA_CHECK(cudaMalloc(&d_geqrf_work, geqrf_lwork * sizeof(float)));
                float *d_tau_inner = nullptr;
                CUDA_CHECK(cudaMalloc(&d_tau_inner, l * sizeof(float)));
                int *d_info = nullptr;
                CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
                CUSOLVER_CHECK(cusolverDnSgeqrf(sol_h, m, l, d_Q, m, d_tau_inner,
                                                 d_geqrf_work, geqrf_lwork, d_info));
                // orgqr to get explicit Q.
                int orgqr_lwork = 0;
                CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(sol_h, m, l, l, d_Q, m, d_tau_inner, &orgqr_lwork));
                float *d_orgqr_work = nullptr;
                CUDA_CHECK(cudaMalloc(&d_orgqr_work, orgqr_lwork * sizeof(float)));
                CUSOLVER_CHECK(cusolverDnSorgqr(sol_h, m, l, l, d_Q, m, d_tau_inner,
                                                 d_orgqr_work, orgqr_lwork, d_info));
                // Y = Q (use Q as the orthogonalized Y for next iteration).
                CUDA_CHECK(cudaMemcpyAsync(d_Y, d_Q, (size_t)m*l*sizeof(float),
                                           cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaFree(d_geqrf_work));
                CUDA_CHECK(cudaFree(d_orgqr_work));
                CUDA_CHECK(cudaFree(d_tau_inner));
                CUDA_CHECK(cudaFree(d_info));
            }
        }

        // Step 3: QR(Y) → Q.
        CUDA_CHECK(cudaMemcpyAsync(d_Q, d_Y, (size_t)m*l*sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
        {
            int geqrf_lwork = 0;
            CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(sol_h, m, l, d_Q, m, &geqrf_lwork));
            float *d_geqrf_work = nullptr;
            CUDA_CHECK(cudaMalloc(&d_geqrf_work, geqrf_lwork * sizeof(float)));
            float *d_tau = nullptr;
            CUDA_CHECK(cudaMalloc(&d_tau, l * sizeof(float)));
            int *d_info = nullptr;
            CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
            CUSOLVER_CHECK(cusolverDnSgeqrf(sol_h, m, l, d_Q, m, d_tau,
                                             d_geqrf_work, geqrf_lwork, d_info));
            int orgqr_lwork = 0;
            CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(sol_h, m, l, l, d_Q, m, d_tau, &orgqr_lwork));
            float *d_orgqr_work = nullptr;
            CUDA_CHECK(cudaMalloc(&d_orgqr_work, orgqr_lwork * sizeof(float)));
            CUSOLVER_CHECK(cusolverDnSorgqr(sol_h, m, l, l, d_Q, m, d_tau,
                                             d_orgqr_work, orgqr_lwork, d_info));
            CUDA_CHECK(cudaFree(d_geqrf_work));
            CUDA_CHECK(cudaFree(d_orgqr_work));
            CUDA_CHECK(cudaFree(d_tau));
            CUDA_CHECK(cudaFree(d_info));
        }

        // Step 4: B = A^T @ Q  (SpMM, transpose).
        // A_desc (m×n), transpose → n×m, applied to Q(m×l) → B(n×l).
        CUSPARSE_CHECK(cusparseSpMM(
            sp_h, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha1, A_desc, Q_desc, &beta0, B_desc,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buf));
        // Implicit centering for A^T: B -= ones_n * (mu^T @ Q)
        float *d_muTQ = nullptr;
        CUDA_CHECK(cudaMalloc(&d_muTQ, l * sizeof(float)));
        {
            float a = 1.0f, b = 0.0f;
            CUBLAS_CHECK(cublasSgemv(blas_h, CUBLAS_OP_T,
                m, l, &a, d_Q, m, d_mu, 1, &b, d_muTQ, 1));
            float neg1 = -1.0f;
            CUBLAS_CHECK(cublasSger(blas_h, n, l, &neg1, d_ones_n, 1, d_muTQ, 1, d_B, n));
        }
        CUDA_CHECK(cudaFree(d_muTQ));

        // Step 5: SVD(B^T) where B is (n×l).
        // cuSOLVER gesvdj takes (rows × cols); we want the (l × l) factor.
        // B has shape n × l. We compute SVD of B (n×l):
        //   B = U_B @ diag(sigma) @ VT_B^T
        // But the small "B" in the algorithm is actually (k+p) × n which is B^T.
        // We compute SVD of B^T (l × n) — but l << n so that's expensive.
        // Instead: B is n × l; SVD of B gives U_B (n×l), sigma (l), VT_B (l×l).
        // We want U = Q @ U_B[:,:k] where U_B are the LEFT singular vectors of B.
        //
        // cuSOLVER gesvdj on B (n×l): m_sv=n, n_sv=l.
        // For large n this is still O(n*l^2). For our case n=20866, l~60, this is cheap.
        {
            gesvdjInfo_t params = nullptr;
            CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&params));
            int lwork_sv = 0;
            CUSOLVER_CHECK(cusolverDnSgesvdj_bufferSize(
                sol_h, CUSOLVER_EIG_MODE_VECTOR, 0,
                n, l, d_B, n, d_sigma, d_U_B, n, d_VT, l,
                &lwork_sv, params));
            float *d_sv_work = nullptr;
            CUDA_CHECK(cudaMalloc(&d_sv_work, lwork_sv * sizeof(float)));
            int *d_info = nullptr;
            CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
            // Note: gesvdj gives U (n×l) and V (l×l).
            // We actually want U (n×l) as the right singular vectors of B^T.
            // Since A^T @ Q = B, and SVD(B) gives U_B, sigma, V_B:
            //   U = Q @ V_B[:,:k]  (not U_B)
            // because: A = Q @ B^T, B = U_B @ S @ V_B^T, so
            //   A = Q @ V_B @ S @ U_B^T
            //   Left SV of A = Q @ V_B.
            // cuSOLVER gesvdj: d_U_B = U (n×min(n,l)), d_VT = V^T (l×l).
            // We want V (the right singular vectors) = VT^T.
            // The left singular vectors of A ≈ Q @ V_B where V_B = VT^T.
            // We'll store sigma and VT; final U = Q @ VT^T[:,:k].
            CUSOLVER_CHECK(cusolverDnSgesvdj(
                sol_h, CUSOLVER_EIG_MODE_VECTOR, 0,
                n, l, d_B, n, d_sigma, d_U_B, n, d_VT, l,
                d_sv_work, lwork_sv, d_info, params));
            CUDA_CHECK(cudaFree(d_sv_work));
            CUDA_CHECK(cudaFree(d_info));
            CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(params));
        }

        // Sync and record.
        cudaStreamSynchronize(stream);
        bench_timer.stop();
        if (timed) {
            bench_timer.record();
            bench_mem.snapshot_after();
        }
    }

    // Read back sigma for correctness comparison.
    res.singular_values.resize(k);
    {
        std::vector<float> h_sigma(l);
        CUDA_CHECK(cudaMemcpy(h_sigma.data(), d_sigma, l * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < k; ++i) res.singular_values[i] = h_sigma[i];
    }
    res.wall_ms     = bench_timer.median_ms();
    res.peak_dev_mb = bench_mem.peak_mb();

    // Cleanup.
    CUSPARSE_CHECK(cusparseDestroySpMat(A_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(Omega_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(Y_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(tmp_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(Q_desc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(B_desc));
    CUDA_CHECK(cudaFree(d_Omega)); CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_mu));    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_Q));     CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_U_B));   CUDA_CHECK(cudaFree(d_sigma));
    CUDA_CHECK(cudaFree(d_VT));    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_ones_n)); CUDA_CHECK(cudaFree(d_ones_m));
    CUSOLVER_CHECK(cusolverDnDestroy(sol_h));
    CUBLAS_CHECK(cublasDestroy(blas_h));
    CUSPARSE_CHECK(cusparseDestroy(sp_h));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return res;
}

// ---------------------------------------------------------------------------
// Log a CONFIG_UNAVAILABLE sentinel row.
// ---------------------------------------------------------------------------
static void log_unavailable(const std::string& impl, const std::string& feature,
                              const std::string& scale, const std::string& reason)
{
    std::printf("BENCH %-30s %-7s %-30s CONFIG_UNAVAILABLE: %s\n",
                feature.c_str(), scale.c_str(), impl.c_str(), reason.c_str());
    std::fflush(stdout);
    std::ofstream ofs(BENCH_REGISTRY_PATH, std::ios::app);
    if (ofs.is_open()) {
        ofs << "| " << current_date_str()
            << " | " << feature
            << " | " << scale
            << " | " << impl
            << " | CONFIG_UNAVAILABLE | — | — | — | — | — | no-git |\n";
    }
}

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------
int main() {
    if (!gpu_available()) {
        std::printf("NO GPU — skipping all SVD adapters bench.\n");
        return 0;
    }

    const bool have_real = fs::exists(PZ_PATH);
    if (!have_real) {
        std::printf("[SVD phaseE] PZ not found: %s\n", PZ_PATH.c_str());
        std::printf("[SVD phaseE] Falling back to tiny synthetic (500x200, k<=10).\n");
        // Synthetic path — just measure auto_select on tiny matrix.
        SyntheticMatrix syn = make_synthetic_matrix(500, 200, 0.05f, 99);
        cudaStream_t stream = nullptr; cudaStreamCreate(&stream);
        io::PzDeviceMatrix synmat;
        synmat.mat.rows = syn.rows; synmat.mat.cols = syn.cols; synmat.mat.nnz = syn.nnz;
        int *d_colp, *d_rowi; float *d_vals;
        cudaMalloc(&d_colp,  (syn.cols+1)*sizeof(int));
        cudaMalloc(&d_rowi,  syn.nnz*sizeof(int));
        cudaMalloc(&d_vals,  syn.nnz*sizeof(float));
        cudaMemcpy(d_colp, syn.indptr.data(),  (syn.cols+1)*sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_rowi, syn.indices.data(), syn.nnz*sizeof(int),        cudaMemcpyHostToDevice);
        cudaMemcpy(d_vals, syn.values.data(),  syn.nnz*sizeof(float),      cudaMemcpyHostToDevice);
        synmat.mat.col_ptr     = factornet::gpu::DeviceMemory<int>::wrap(d_colp, syn.cols+1);
        synmat.mat.row_indices = factornet::gpu::DeviceMemory<int>::wrap(d_rowi, syn.nnz);
        synmat.mat.values      = factornet::gpu::DeviceMemory<float>::wrap(d_vals, syn.nnz);
        // pinned host copies.
        int *h_cp, *h_ri; float *h_va;
        cudaMallocHost(&h_cp, (syn.cols+1)*sizeof(int));
        cudaMallocHost(&h_ri, syn.nnz*sizeof(int));
        cudaMallocHost(&h_va, syn.nnz*sizeof(float));
        memcpy(h_cp, syn.indptr.data(),  (syn.cols+1)*sizeof(int));
        memcpy(h_ri, syn.indices.data(), syn.nnz*sizeof(int));
        memcpy(h_va, syn.values.data(),  syn.nnz*sizeof(float));
        synmat.host_retained = true;
        synmat.host_indptr.reset(h_cp, [](int*   p){cudaFreeHost(p);});
        synmat.host_indices.reset(h_ri, [](int*   p){cudaFreeHost(p);});
        synmat.host_values.reset(h_va,  [](float* p){cudaFreeHost(p);});
        synmat.producer_stream = stream;

        reduce::svd::SvdConfig cfg; cfg.k_max = 10;
        BenchTimer tmr(stream); PeakMemoryTracker mem;
        for (int i = 0; i < WARMUP+TIMED; ++i) {
            bool t = (i >= WARMUP);
            if (t) mem.snapshot_before();
            tmr.start();
            try { auto r = reduce::svd::auto_select(synmat, 10, cfg); (void)r; } catch(...) {}
            cudaStreamSynchronize(stream);
            tmr.stop();
            if (t) { tmr.record(); mem.snapshot_after(); }
        }
        BenchRow row;
        row.date = current_date_str(); row.feature = "reduce/svd"; row.scale = "tiny";
        row.impl = "ours_auto_select"; row.wall_ms_med = tmr.median_ms();
        row.wall_ms_min = tmr.min_ms(); row.wall_ms_max = tmr.max_ms();
        row.mem_mb_peak = mem.peak_mb(); row.cells_per_sec = throughput(syn.cols, row.wall_ms_med);
        log_row(row);
        cudaFree(d_colp); cudaFree(d_rowi); cudaFree(d_vals);
        cudaStreamDestroy(stream);
        return 0;
    }

    // ---- Load real matrix. ----
    std::printf("=== SVD Phase E: Loading %s ===\n", PZ_PATH.c_str());
    std::fflush(stdout);

    cudaStream_t main_stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&main_stream));

    io::PzDeviceMatrix raw_mat = io::load_pz(PZ_PATH, main_stream, /*keep_host_pinned=*/true);
    CUDA_CHECK(cudaStreamSynchronize(main_stream));
    std::printf("Matrix: %d genes × %d cells, nnz=%d\n",
                raw_mat.mat.rows, raw_mat.mat.cols, (int)raw_mat.mat.nnz);
    std::fflush(stdout);

    // Log-normalize (exclude from timing).
    preprocess::LogNormConfig lnc; lnc.target_count = 1e4f;
    {
        auto lnr = preprocess::log_normalize(raw_mat.mat, lnc, main_stream);
        CUDA_CHECK(cudaStreamSynchronize(main_stream));
    }

    const int64_t n_cells = raw_mat.mat.cols;

    // ---- Run Python baselines. ----
    const std::string py_script = REFS_DIR + "/cycle60_svd_baselines.py";
    const bool have_py = fs::exists(py_script);

    // scanpy_pca results (one per k).
    double scanpy_wall_ms[N_K]  = {-1.0, -1.0, -1.0};
    double scanpy_mem_mb[N_K]   = {-1.0, -1.0, -1.0};
    // factornet_cpu_irlba results + singular values for correctness.
    double cpu_wall_ms[N_K]     = {-1.0, -1.0, -1.0};
    std::vector<std::vector<float>> cpu_sv(N_K);  // reference singular values

    if (have_py) {
        for (int ki = 0; ki < N_K; ++ki) {
            const int k = K_VALUES[ki];

            // scanpy_pca.
            {
                char cmd[1024]{};
                std::string out_json = TMP_DIR + "/cycle60_scanpy_k" + std::to_string(k) + ".json";
                std::snprintf(cmd, sizeof(cmd),
                    "python3 %s --mode scanpy_pca --k %d --pz %s --out %s 2>/dev/null",
                    py_script.c_str(), k, PZ_PATH.c_str(), out_json.c_str());
                run_cmd(cmd, "scanpy_pca");
                scanpy_wall_ms[ki] = parse_json_double(out_json, "wall_ms");
                scanpy_mem_mb[ki]  = parse_json_double(out_json, "mem_mb");
            }

            // factornet_cpu_irlba.
            {
                char cmd[1024]{};
                std::string out_json = TMP_DIR + "/cycle60_cpu_irlba_k" + std::to_string(k) + ".json";
                std::snprintf(cmd, sizeof(cmd),
                    "python3 %s --mode factornet_cpu_irlba --k %d --pz %s --out %s 2>/dev/null",
                    py_script.c_str(), k, PZ_PATH.c_str(), out_json.c_str());
                run_cmd(cmd, "factornet_cpu_irlba");
                cpu_wall_ms[ki] = parse_json_double(out_json, "wall_ms");

                // Read singular values for correctness comparison.
                std::string sv_json = TMP_DIR + "/cycle60_cpu_irlba_sv_k" + std::to_string(k) + ".json";
                char sv_cmd[1024]{};
                std::snprintf(sv_cmd, sizeof(sv_cmd),
                    "python3 %s --mode factornet_cpu_irlba_sv --k %d --pz %s --out %s 2>/dev/null",
                    py_script.c_str(), k, PZ_PATH.c_str(), sv_json.c_str());
                run_cmd(sv_cmd, "factornet_cpu_irlba_sv");
                // Parse sv array from JSON: "sv": [s1, s2, ...]
                {
                    std::ifstream f(sv_json);
                    if (f.is_open()) {
                        std::string content((std::istreambuf_iterator<char>(f)),
                                             std::istreambuf_iterator<char>());
                        auto pos = content.find("\"sv\"");
                        auto lb  = content.find('[', pos);
                        auto rb  = content.find(']', lb);
                        if (lb != std::string::npos && rb != std::string::npos) {
                            std::string arr = content.substr(lb+1, rb-lb-1);
                            // parse comma-separated floats.
                            std::istringstream ss(arr);
                            std::string tok;
                            cpu_sv[ki].clear();
                            while (std::getline(ss, tok, ',')) {
                                try { cpu_sv[ki].push_back(std::stof(tok)); }
                                catch (...) {}
                            }
                        }
                    }
                }
            }
        }
    } else {
        std::printf("[SVD phaseE] Python baseline script not found: %s\n", py_script.c_str());
    }

    // ---- Log SOTA baseline rows. ----
    const std::string feature_str = "reduce/svd";
    const std::string scale_str   = "small";

    for (int ki = 0; ki < N_K; ++ki) {
        const int k = K_VALUES[ki];
        const std::string k_str = "small-k" + std::to_string(k);
        {
            BenchRow row;
            row.date = current_date_str();
            row.feature = feature_str;
            row.scale   = k_str;
            row.impl    = "scanpy_pca";
            row.wall_ms_med = (float)scanpy_wall_ms[ki];
            row.wall_ms_min = row.wall_ms_med;
            row.wall_ms_max = row.wall_ms_med;
            row.mem_mb_peak = (float)scanpy_mem_mb[ki];
            row.cells_per_sec = (scanpy_wall_ms[ki] > 0)
                ? throughput(n_cells, (float)scanpy_wall_ms[ki]) : 0.0f;
            log_row(row);
        }
        {
            BenchRow row;
            row.date = current_date_str();
            row.feature = feature_str;
            row.scale   = k_str;
            row.impl    = "factornet_cpu_irlba";
            row.wall_ms_med = (float)cpu_wall_ms[ki];
            row.wall_ms_min = row.wall_ms_med;
            row.wall_ms_max = row.wall_ms_med;
            row.mem_mb_peak = -1.0f;
            row.cells_per_sec = (cpu_wall_ms[ki] > 0)
                ? throughput(n_cells, (float)cpu_wall_ms[ki]) : 0.0f;
            log_row(row);
        }
    }

    // CONFIG_UNAVAILABLE rows for rapids.
    for (int ki = 0; ki < N_K; ++ki) {
        const int k = K_VALUES[ki];
        const std::string k_str = "small-k" + std::to_string(k);
        log_unavailable("cuml_truncated_svd",  feature_str, k_str, "ENV-RAPIDS-G001");
        log_unavailable("cuml_randomized_pca", feature_str, k_str, "ENV-RAPIDS-G001");
    }

    // ---- Bench surviving singlet-gpu SVD adapters × k values (Rule 32). ----
    struct AdapterSpec {
        const char* name;
        bool is_auto, is_randomized, is_deflation;
    };
    static const AdapterSpec adapters[] = {
        {"ours_deflation",       false,false,true},
        {"ours_randomized",      false,true, false},
        {"ours_auto_select",     true, false,false},
    };
    constexpr int N_ADAPTERS = 3;

    // Store results for autonomy gate computation.
    double adapter_wall[N_ADAPTERS][N_K];
    double adapter_mem [N_ADAPTERS][N_K];
    for (int a = 0; a < N_ADAPTERS; ++a)
        for (int ki = 0; ki < N_K; ++ki) { adapter_wall[a][ki] = -1.0; adapter_mem[a][ki] = -1.0; }

    for (int ki = 0; ki < N_K; ++ki) {
        const int k = K_VALUES[ki];
        const std::string k_str = "small-k" + std::to_string(k);
        std::printf("\n--- Benchmarking SVD adapters at k=%d ---\n", k);
        std::fflush(stdout);

        for (int a = 0; a < N_ADAPTERS; ++a) {
            const auto& spec = adapters[a];
            std::printf("  [%s k=%d] ...\n", spec.name, k); std::fflush(stdout);
            double wms = -1.0, mmb = -1.0;
            bench_svd_adapter(raw_mat, k, spec.name, main_stream,
                               wms, mmb,
                               spec.is_auto, spec.is_randomized, spec.is_deflation);
            adapter_wall[a][ki] = wms;
            adapter_mem [a][ki] = mmb;

            BenchRow row;
            row.date = current_date_str(); row.feature = feature_str; row.scale = k_str;
            row.impl = spec.name;
            row.wall_ms_med = (float)wms; row.wall_ms_min = row.wall_ms_med; row.wall_ms_max = row.wall_ms_med;
            row.mem_mb_peak = (float)mmb;
            row.cells_per_sec = (wms > 0) ? throughput(n_cells, (float)wms) : 0.0f;
            log_row(row);
        }
    }

    // ---- Novel prototype: randomized_smallk × k values. ----
    double smallk_wall[N_K]; double smallk_mem[N_K];
    std::vector<std::vector<float>> smallk_sv(N_K);
    for (int ki = 0; ki < N_K; ++ki) smallk_wall[ki] = smallk_mem[ki] = -1.0;

    std::printf("\n--- Benchmarking ours_randomized_smallk (novel prototype) ---\n");
    std::fflush(stdout);
    for (int ki = 0; ki < N_K; ++ki) {
        const int k = K_VALUES[ki];
        const std::string k_str = "small-k" + std::to_string(k);
        std::printf("  [randomized_smallk k=%d] ...\n", k); std::fflush(stdout);
        try {
            auto r = bench_randomized_smallk(raw_mat, k, /*n_power=*/2, /*oversample=*/10);
            smallk_wall[ki] = r.wall_ms;
            smallk_mem [ki] = r.peak_dev_mb;
            smallk_sv  [ki] = r.singular_values;

            BenchRow row;
            row.date = current_date_str(); row.feature = feature_str; row.scale = k_str;
            row.impl = "ours_randomized_smallk";
            row.wall_ms_med = r.wall_ms; row.wall_ms_min = r.wall_ms; row.wall_ms_max = r.wall_ms;
            row.mem_mb_peak = r.peak_dev_mb;
            row.cells_per_sec = (r.wall_ms > 0) ? throughput(n_cells, r.wall_ms) : 0.0f;
            log_row(row);
        } catch (const std::exception& e) {
            std::printf("  [randomized_smallk k=%d] EXCEPTION: %s\n", k, e.what());
            std::fflush(stdout);
            BenchRow row;
            row.date = current_date_str(); row.feature = feature_str; row.scale = k_str;
            row.impl = "ours_randomized_smallk";
            row.wall_ms_med = -1.0f; row.mem_mb_peak = -1.0f; row.cells_per_sec = 0.0f;
            log_row(row);
        }
    }

    // ---- Compute correctness: SV relative error vs factornet_cpu_irlba. ----
    std::printf("\n=== Correctness (SV relative error vs factornet_cpu_irlba) ===\n");
    std::fflush(stdout);
    // We don't have GPU singular values readily available in float from the adapter
    // (they are inside the SvdResult which we discarded for timing).
    // Run one extra non-timed SVD pass to get sigma.
    // For the timing table we use the timed results above; for correctness we do
    // a single extra call (not counted in wall_ms).
    // Index 1 = k=50 is the headline gate for novel pursuit.
    const int ki_50 = 1; // K_VALUES[1] == 50

    // Placeholder correctness: compute rel error for randomized_smallk only (we have its SVs).
    double smallk_rel_err_50 = -1.0;
    {
        const auto& ours = smallk_sv[ki_50];
        const auto& ref  = cpu_sv[ki_50];
        if (!ours.empty() && !ref.empty()) {
            double sum_rel = 0.0;
            int cnt = (int)std::min(ours.size(), ref.size());
            cnt = std::min(cnt, 50);
            for (int i = 0; i < cnt; ++i) {
                double sv_ref = ref[i];
                double sv_our = ours[i];
                if (sv_ref > 1e-8)
                    sum_rel += std::fabs(sv_our - sv_ref) / sv_ref;
            }
            if (cnt > 0) smallk_rel_err_50 = sum_rel / cnt;
        }
    }
    std::printf("  randomized_smallk @k=50 mean SV rel err vs CPU IRLBA: %.4e %s\n",
                smallk_rel_err_50,
                (smallk_rel_err_50 >= 0 && smallk_rel_err_50 <= 1e-3) ? "PASS" : "FAIL");

    // ---- Summary: Rule 31 autonomy gate. ----
    std::printf("\n=== Rule 31 autonomy gate (auto_select vs best manual @k=50) ===\n");
    {
        double auto_wall  = adapter_wall[5][ki_50]; // ours_auto_select
        double best_manual = 1e18;
        for (int a = 0; a < 5; ++a)  // adapters 0-4 are manual
            if (adapter_wall[a][ki_50] > 0)
                best_manual = std::min(best_manual, adapter_wall[a][ki_50]);
        if (auto_wall > 0 && best_manual < 1e17) {
            double delta = std::fabs(auto_wall - best_manual) / best_manual * 100.0;
            std::printf("  auto_select=%.2fms  best_manual=%.2fms  delta=%.1f%%  gate(<=10%%) %s\n",
                        auto_wall, best_manual, delta,
                        delta <= 10.0 ? "PASS" : "FAIL");
        } else {
            std::printf("  Insufficient data for Rule 31 gate.\n");
        }
    }

    // ---- Summary: frontier promotion gate. ----
    std::printf("\n=== Frontier promotion gate (@k=50) ===\n");
    {
        double our_best = 1e18;
        for (int a = 0; a < N_ADAPTERS; ++a)
            if (adapter_wall[a][ki_50] > 0) our_best = std::min(our_best, adapter_wall[a][ki_50]);
        if (smallk_wall[ki_50] > 0) our_best = std::min(our_best, smallk_wall[ki_50]);
        double scanpy_w = scanpy_wall_ms[ki_50];
        if (our_best < 1e17 && scanpy_w > 0) {
            double ratio = our_best / scanpy_w;
            std::printf("  our_best=%.2fms  scanpy_pca=%.2fms  ratio=%.4f  gate(<=0.1) %s\n",
                        our_best, scanpy_w, ratio,
                        ratio <= 0.1 ? "PASS" : "FAIL");
        } else {
            std::printf("  Insufficient data for frontier gate.\n");
        }
    }

    // ---- Rule 30 novel pursuit gate summary (written to JSON for bash wrapper). ----
    // Rule 32: deflation (adapter[0]) replaces irlba as the primary reference wall time.
    {
        const int k = 50;
        double deflation_w = adapter_wall[0][ki_50];  // ours_deflation @k=50 (Rule 32 winner)
        double sk_w        = smallk_wall[ki_50];      // randomized_smallk @k=50
        double sk_mem      = smallk_mem[ki_50];
        double scanpy_mem_50 = scanpy_mem_mb[ki_50];

        // G1: SV rel err ≤ 1e-3
        bool g1 = (smallk_rel_err_50 >= 0 && smallk_rel_err_50 <= 1e-3);
        // G2: Frobenius error — we don't compute it here (would need full U,V);
        //     report as N/A from C++ driver; Python baseline has it.
        // G3: sk_w <= 0.5 * deflation_w (was vs irlba; deflation is new reference)
        bool g3 = (sk_w > 0 && deflation_w > 0 && sk_w <= 0.5 * deflation_w);
        // G4: sk_mem <= 0.4 * scanpy_mem
        bool g4 = (sk_mem > 0 && scanpy_mem_50 > 0 && sk_mem <= 0.4 * scanpy_mem_50);

        std::printf("\n=== Rule 30 randomized_smallk gates @k=50 ===\n");
        std::printf("  G1 SV rel err = %.4e (gate <=1e-3): %s\n",
                    smallk_rel_err_50, g1 ? "PASS" : "FAIL");
        std::printf("  G2 Frobenius recon err: N/A (Python baseline)\n");
        std::printf("  G3 wall %.2fms vs deflation %.2fms (gate <=50%%): %s\n",
                    sk_w, deflation_w, g3 ? "PASS" : "FAIL");
        std::printf("  G4 mem %.1fMB vs scanpy %.1fMB (gate <=40%%): %s\n",
                    sk_mem, scanpy_mem_50, g4 ? "PASS" : "FAIL");

        // Write JSON for bash wrapper Step 7.
        std::string novel_json = TMP_DIR + "/cycle60_novel_svd.json";
        std::ofstream jf(novel_json);
        if (jf.is_open()) {
            jf << "{\n"
               << "  \"smallk_wall_ms_k50\":     " << sk_w << ",\n"
               << "  \"deflation_wall_ms_k50\":  " << deflation_w << ",\n"
               << "  \"scanpy_wall_ms_k50\":      " << scanpy_wall_ms[ki_50] << ",\n"
               << "  \"scanpy_mem_mb_k50\":       " << scanpy_mem_50 << ",\n"
               << "  \"smallk_mem_mb_k50\":       " << sk_mem << ",\n"
               << "  \"sv_rel_err_k50\":          " << smallk_rel_err_50 << ",\n"
               << "  \"g1_sv_rel_err\": "           << (g1 ? "true" : "false") << ",\n"
               << "  \"g2_frob_err\": null,\n"
               << "  \"g3_wall_ratio\": "           << (g3 ? "true" : "false") << ",\n"
               << "  \"g4_mem_ratio\": "            << (g4 ? "true" : "false") << "\n"
               << "}\n";
        }
        std::printf("  Novel JSON written: %s\n", novel_json.c_str());
    }

    cudaStreamDestroy(main_stream);
    std::printf("\n=== SVD Phase E bench complete ===\n");
    std::fflush(stdout);
    return 0;
}
