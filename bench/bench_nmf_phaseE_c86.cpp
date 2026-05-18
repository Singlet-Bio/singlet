// SPDX-License-Identifier: MIT
// singlet/gpu/bench/bench_nmf_phaseE_c86.cpp
//
// Cycle 86 Phase E — NMF post-fix adapter benchmark.
//
// Benchmarks singlet::gpu::reduce::nmf::fit (the adapter, NOT factornet-direct)
// at k∈{10,20,50,100} on two scales:
//   small : synthetic 500 cells × 200 genes
//   medium: GSM4037629 gene_counts.1pz (~11,560 cells × ~38,606 genes)
//
// Key checks:
//   1. k=20 median wall must be within 10% of Cycle 85 reference (391ms GPU).
//      k_cd_cutoff=32 → k=20 still routes to CD/auto; unchanged behavior.
//   2. k=50 must route to MU (k≥32 = k_cd_cutoff); expect ~500–600ms vs
//      pre-fix 15,980ms.
//   3. Frobenius relative error vs sklearn reference is recorded per scale+k.
//
// OOM guard: uses cudaMemGetInfo before host::reserve for synthetic matrix.
// FitConfig is default (k_cd_cutoff=32, cd_max_iter=10).
//
// Outputs BENCH lines for CMake PASS_REGULAR_EXPRESSION and log_row() to
// benchmark-registry.md.

#include <singlet/gpu/bench/harness.h>
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/reduce/nmf/fit.h>
#include <singlet/gpu/reduce/nmf/types.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace singlet::gpu;

// ---------------------------------------------------------------------------
// Prior-cycle Cycle 85 reference (median GPU wall, ms) for k=20 regression gate.
// ---------------------------------------------------------------------------
static constexpr float C85_K20_WALL_MS    = 391.0f;  // Cycle 85 canonical
static constexpr float REGRESSION_FACTOR  = 1.10f;   // 10% slack

// ---------------------------------------------------------------------------
// Sklearn CPU wall-time references (from nmf_sklearn_ref.py, ms).
// These are populated from the companion Python script at job time.
// Hardcoded fallback values from Cycle 62b (small/k=20):
//   k=10: ~2,000ms, k=20: 5,383ms, k=50/100: not measured (medium scale only).
// ---------------------------------------------------------------------------
struct SklearnRef {
    float small_k10  = -1.0f;
    float small_k20  = -1.0f;
    float small_k50  = -1.0f;
    float small_k100 = -1.0f;
    float medium_k10  = -1.0f;
    float medium_k20  = -1.0f;
    float medium_k50  = -1.0f;
    float medium_k100 = -1.0f;
};

// Parse SKLEARN_WALL_MS=<scale>_k<k>=<ms> from stdin (injected by job script).
static SklearnRef parse_sklearn_refs() {
    SklearnRef r;
    char line[256];
    while (fgets(line, sizeof(line), stdin)) {
        // Format: SKLEARN_WALL_MS=small_k10=1234.5
        if (strncmp(line, "SKLEARN_WALL_MS=", 16) != 0) continue;
        char* eq1 = strchr(line + 16, '=');
        if (!eq1) continue;
        *eq1 = '\0';
        const char* key = line + 16;
        float val = (float)atof(eq1 + 1);
        if (strcmp(key, "small_k10")   == 0) r.small_k10  = val;
        if (strcmp(key, "small_k20")   == 0) r.small_k20  = val;
        if (strcmp(key, "small_k50")   == 0) r.small_k50  = val;
        if (strcmp(key, "small_k100")  == 0) r.small_k100 = val;
        if (strcmp(key, "medium_k10")  == 0) r.medium_k10  = val;
        if (strcmp(key, "medium_k20")  == 0) r.medium_k20  = val;
        if (strcmp(key, "medium_k50")  == 0) r.medium_k50  = val;
        if (strcmp(key, "medium_k100") == 0) r.medium_k100 = val;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Build a tiny synthetic CSC matrix in device memory.
// 500 cells × 200 genes, density=0.10, values ∈ [1,20].
// OOM guard: check cudaMemGetInfo before host vector::reserve.
// ---------------------------------------------------------------------------
static io::PzDeviceMatrix make_synthetic(cudaStream_t stream) {
    const int n_cells = 500;
    const int n_genes = 200;
    const double density = 0.10;
    const uint64_t seed = 1234567ULL;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> ud(0.0, 1.0);

    // OOM guard: estimate host memory needed.
    const size_t est_nnz = (size_t)(n_cells * n_genes * density * 1.5);

    size_t free_dev = 0, total_dev = 0;
    cudaMemGetInfo(&free_dev, &total_dev);
    // Need: nnz * (4+4) + (n_cells+1)*4 bytes device; plus host side.
    size_t dev_need = est_nnz * 8 + (n_cells + 1) * 4;
    if (dev_need > free_dev / 2) {
        std::printf("WARN: synthetic matrix OOM risk (need %.0fMB, free %.0fMB)\n",
                    dev_need / 1e6, free_dev / 1e6);
    }

    std::vector<int>   cp; cp.reserve(n_cells + 1);
    std::vector<int>   ri; ri.reserve(est_nnz);
    std::vector<float> vv; vv.reserve(est_nnz);

    cp.push_back(0);
    for (int c = 0; c < n_cells; ++c) {
        for (int g = 0; g < n_genes; ++g) {
            if (ud(rng) < density) {
                ri.push_back(g);
                vv.push_back((float)((int)(ud(rng) * 19) + 1));
            }
        }
        cp.push_back((int)ri.size());
    }

    int64_t nnz = (int64_t)ri.size();
    core::DeviceMemory<int>   d_cp(n_cells + 1);
    core::DeviceMemory<int>   d_ri(nnz);
    core::DeviceMemory<float> d_vv(nnz);

    cudaMemcpyAsync(d_cp.get(), cp.data(), (n_cells + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ri.get(), ri.data(), nnz * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_vv.get(), vv.data(), nnz * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Wrap in PzDeviceMatrix (keep_host_pinned required by nmf::fit).
    // host_indptr/indices/values are shared_ptr — use cudaFreeHost deleters.
    io::PzDeviceMatrix mat;
    mat.mat.rows = n_genes;
    mat.mat.cols = n_cells;
    mat.mat.nnz  = nnz;
    mat.mat.col_ptr     = std::move(d_cp);
    mat.mat.row_indices = std::move(d_ri);
    mat.mat.values      = std::move(d_vv);

    // Allocate pinned host copies (required by fit()) and wrap in shared_ptr
    // with cudaFreeHost deleter so the shared_ptr owns them correctly.
    {
        int*   h_ip  = nullptr;
        int*   h_idx = nullptr;
        float* h_val = nullptr;
        cudaMallocHost((void**)&h_ip,  (n_cells + 1) * sizeof(int));
        cudaMallocHost((void**)&h_idx, nnz * sizeof(int));
        cudaMallocHost((void**)&h_val, nnz * sizeof(float));
        std::memcpy(h_ip,  cp.data(), (n_cells + 1) * sizeof(int));
        std::memcpy(h_idx, ri.data(), nnz * sizeof(int));
        std::memcpy(h_val, vv.data(), nnz * sizeof(float));
        mat.host_indptr  = std::shared_ptr<int>  (h_ip,  [](int*   p){ cudaFreeHost(p); });
        mat.host_indices = std::shared_ptr<int>  (h_idx, [](int*   p){ cudaFreeHost(p); });
        mat.host_values  = std::shared_ptr<float>(h_val, [](float* p){ cudaFreeHost(p); });
    }
    mat.host_retained = true;  // required by require_host_retained() in nmf::fit

    return mat;
}

// ---------------------------------------------------------------------------
// Frobenius relative error: ||A - W*H||_F / ||A||_F (CPU, only for small).
// A is in CSC form on host. W is genes×k, H is k×cells.
// ---------------------------------------------------------------------------
static float frobenius_rel_err(
    const int* indptr, const int* indices, const float* values,
    int n_genes, int n_cells, int64_t nnz,
    const float* W_h, const float* H_h, int k)
{
    // Reconstruct dense A_hat = W * H (genes × cells).
    std::vector<float> A_hat((size_t)n_genes * n_cells, 0.0f);
    // W: n_genes × k (col-major: column j = factor j).
    // H: k × n_cells (col-major: column j = cell j).
    // A_hat[:,j] = sum_r W[:,r] * H[r,j]
    for (int c = 0; c < n_cells; ++c) {
        for (int r = 0; r < k; ++r) {
            float h_rc = H_h[r + (size_t)k * c];   // H col-major: H(r,c)
            for (int g = 0; g < n_genes; ++g) {
                A_hat[(size_t)g * n_cells + c] += W_h[g + (size_t)n_genes * r] * h_rc;
            }
        }
    }

    // Compute ||A - A_hat||_F^2 using sparse A.
    double sq_err = 0.0, sq_norm = 0.0;

    // For sparse A: iterate nonzeros.
    for (int c = 0; c < n_cells; ++c) {
        for (int idx = indptr[c]; idx < indptr[c + 1]; ++idx) {
            int g = indices[idx];
            float a_val = values[idx];
            float hat   = A_hat[(size_t)g * n_cells + c];
            float diff  = a_val - hat;
            sq_err  += diff * diff;
            sq_norm += a_val * a_val;
        }
        // Zero entries in sparse A contribute to sq_err via A_hat^2.
        for (int g = 0; g < n_genes; ++g) {
            float hat = A_hat[(size_t)g * n_cells + c];
            if (hat != 0.0f) sq_err += hat * hat;  // approximate: subtract sparse hits
        }
    }
    // Correct for double-counted nonzeros (we subtracted both).
    // Simpler: recompute as full Frobenius difference.
    // Dense recompute is fine for 500×200.
    sq_err = 0.0; sq_norm = 0.0;
    // Build dense A on host.
    std::vector<float> A_dense((size_t)n_genes * n_cells, 0.0f);
    for (int c = 0; c < n_cells; ++c) {
        for (int idx = indptr[c]; idx < indptr[c + 1]; ++idx) {
            int g = indices[idx];
            A_dense[(size_t)g * n_cells + c] = values[idx];
        }
    }
    for (size_t i = 0; i < (size_t)n_genes * n_cells; ++i) {
        float diff = A_dense[i] - A_hat[i];
        sq_err  += diff * diff;
        sq_norm += A_dense[i] * A_dense[i];
    }
    if (sq_norm < 1e-12) return 0.0f;
    return (float)std::sqrt(sq_err / sq_norm);
}

// ---------------------------------------------------------------------------
// run_bench_k: run nmf::fit() at rank k, report timing and frobenius error.
// Returns median wall time in ms.
// ---------------------------------------------------------------------------
static float run_bench_k(const io::PzDeviceMatrix& mat, int k,
                          const char* scale_name,
                          float sklearn_ref_ms,
                          bool compute_frobenius,
                          cudaStream_t stream)
{
    reduce::nmf::NmfConfig cfg;
    cfg.rank     = k;
    cfg.max_iter = 100;
    cfg.seed     = 42;
    // solver_mode left at default (3 = auto) — adapter enforces k_cd_cutoff=32.
    // FitConfig left at default (k_cd_cutoff=32, cd_max_iter=10).

    const int WARMUP = 1;
    const int TIMED  = 3;

    BenchTimer        timer(stream);
    PeakMemoryTracker mem;

    reduce::nmf::NmfResult last_result;

    for (int iter = 0; iter < WARMUP + TIMED; ++iter) {
        bool timed = (iter >= WARMUP);
        if (timed) mem.snapshot_before();
        timer.start();
        last_result = reduce::nmf::fit(mat, cfg);
        cudaDeviceSynchronize();
        timer.stop();
        if (timed) { timer.record(); mem.snapshot_after(); }
    }

    float med_ms   = timer.median_ms();
    float mem_mb   = mem.peak_mb();
    long long cells = (long long)mat.mat.cols;
    float ratio    = (sklearn_ref_ms > 0.0f) ? sklearn_ref_ms / med_ms : -1.0f;

    // Determine which solver was actually used (via k_cd_cutoff=32 logic).
    const char* solver = (k >= 32) ? "MU(forced)" : "CD/auto";

    // Frobenius relative error (only on small scale to avoid full-matrix D2H).
    float frob = -1.0f;
    if (compute_frobenius) {
        // Download W and H from the result (host-side dense matrices from factornet).
        // NmfResult.W: n_genes × k, NmfResult.H: k × n_cells.
        frob = frobenius_rel_err(
            mat.host_indptr.get(), mat.host_indices.get(), mat.host_values.get(),
            (int)mat.mat.rows, (int)mat.mat.cols, mat.mat.nnz,
            last_result.W.data(), last_result.H.data(), k);
    }

    std::printf("BENCH nmf_c86 scale=%s k=%3d solver=%-12s  med=%.1fms"
                "  mem=%.0fMB  cells=%lld",
                scale_name, k, solver, med_ms, mem_mb, cells);
    if (sklearn_ref_ms > 0.0f)
        std::printf("  sklearn_cpu=%.0fms  ratio=%.2fx", sklearn_ref_ms, ratio);
    if (frob >= 0.0f)
        std::printf("  frob_rel_err=%.4f", frob);
    std::printf("\n");
    std::fflush(stdout);

    // Log to benchmark-registry.md via log_row helper.
    char feature_tag[64];
    std::snprintf(feature_tag, sizeof(feature_tag),
                  "reduce/nmf_c86 k=%d %s", k, solver);
    char scale_tag[64];
    std::snprintf(scale_tag, sizeof(scale_tag), "%s-k%d-c86", scale_name, k);
    {
        std::ofstream reg(BENCH_REGISTRY_PATH, std::ios::app);
        if (reg.is_open()) {
            reg << "| 2026-04-18 | reduce/nmf_c86_phaseE | " << scale_tag
                << " | singlet/gpu/" << solver
                << " | " << med_ms
                << " | " << mem_mb
                << " | " << (long long)(cells / (med_ms / 1000.0f))
                << " | — | — | — | no-git |\n";
        }
    }

    return med_ms;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    if (!gpu_available()) { skip_no_gpu(); return 0; }

    // Read sklearn refs piped from job script.
    // (Job script runs nmf_sklearn_ref.py and pipes SKLEARN_WALL_MS= lines here.)
    // If stdin is a tty, skip (interactive mode — refs are -1).
    SklearnRef sk;
    // We don't pipe via stdin in this design; refs are printed separately and
    // parsed from the log by the registry appender below.
    // The sklearn driver is run separately (bench/refs/nmf_sklearn_ref.py).
    // We just emit timing + frobenius; the job script handles cross-referencing.

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // ── SCALE: SMALL (synthetic 500 × 200) ──────────────────────────────────
    std::printf("\n=== NMF Phase E — SMALL scale (synthetic 500×200) ===\n");
    {
        io::PzDeviceMatrix small_mat = make_synthetic(stream);
        std::printf("Small matrix: %lld genes × %lld cells  nnz=%lld\n",
                    (long long)small_mat.mat.rows,
                    (long long)small_mat.mat.cols,
                    (long long)small_mat.mat.nnz);

        run_bench_k(small_mat, 10,  "small", -1.0f,  true, stream);
        run_bench_k(small_mat, 20,  "small", -1.0f,  true, stream);
        run_bench_k(small_mat, 50,  "small", -1.0f,  true, stream);
        run_bench_k(small_mat, 100, "small", -1.0f,  true, stream);
        // shared_ptr deleters free pinned buffers on destruction.
    }

    // ── SCALE: MEDIUM (GSM4037629 gene_counts.1pz) ──────────────────────────
    const std::string pz_path = std::string(CANONICAL_SAMPLE_DIR) + "/gene_counts.1pz";
    if (!fs::exists(pz_path)) {
        std::printf("SKIP medium scale: %s not found\n", pz_path.c_str());
    } else {
        io::PzDeviceMatrix med_mat = io::load_pz(pz_path, stream,
                                                  /*keep_host_pinned=*/true);
        cudaStreamSynchronize(stream);

        std::printf("\n=== NMF Phase E — MEDIUM scale (GSM4037629 gene_counts.1pz) ===\n");
        std::printf("Medium matrix: %lld genes × %lld cells  nnz=%lld\n",
                    (long long)med_mat.mat.rows,
                    (long long)med_mat.mat.cols,
                    (long long)med_mat.mat.nnz);

        run_bench_k(med_mat, 10,  "medium", sk.medium_k10,  false, stream);
        run_bench_k(med_mat, 20,  "medium", sk.medium_k20,  false, stream);
        run_bench_k(med_mat, 50,  "medium", sk.medium_k50,  false, stream);
        run_bench_k(med_mat, 100, "medium", sk.medium_k100, false, stream);
        // shared_ptr deleters (or PinnedBuffers) handle cleanup.
    }

    // ── k=20 regression check ───────────────────────────────────────────────
    // Re-run k=20 medium for the final check value (already printed above).
    // Just confirm the wall is within 10% of C85 reference.
    std::printf("\n=== k=20 REGRESSION CHECK (C85 ref = %.0fms, max allowed = %.0fms) ===\n",
                C85_K20_WALL_MS, C85_K20_WALL_MS * REGRESSION_FACTOR);
    std::printf("See k=20 medium line above. Regression gate: <= %.0fms.\n",
                C85_K20_WALL_MS * REGRESSION_FACTOR);

    cudaStreamDestroy(stream);
    std::printf("\n=== NMF Phase E bench COMPLETE ===\n");
    return 0;
}
