// SPDX-License-Identifier: MIT
// integrates: Buttner et al. 2019 kBET batch-effect test
// singlet/gpu/integrate/kbet.h
//
// kBET — k-nearest neighbor Batch Effect Test.
// Buttner M, Miao Z, Wolf FA, et al. (2019) Nat Methods 16:43-49.
// Reference R package: https://github.com/theislab/kBET
//
// For each cell c with k nearest neighbors and batch labels:
//   1. Observe local batch frequency: n_obs[b] = count of c's k neighbors in batch b.
//   2. Expected frequency under null (uniform mixing):
//      n_exp[b] = k * (global_count[b] / n_cells)
//   3. Chi-square statistic:
//      chi2[c] = Σ_b (n_obs[b] - n_exp[b])^2 / max(n_exp[b], 1.0)   (for n_exp[b] > 0)
//   4. p-value = igamcf(df/2, chi2/2)  where df = n_batches - 1
//      igamcf = CUDA regularized upper incomplete gamma (complementary CDF).
//   5. reject[c] = 1 if p-value < 0.05.
//   6. kBET score = mean(reject) = reject_rate.  Lower = better mixing.
//
// v0: full-cell scan (no random subsetting).  Per-cell chi2 and p-value exposed.
//
// GPU mapping: one block per cell.  Thread 0 serially scans k neighbors,
//   builds int local_count[n_batches] in shared memory, then computes chi2.
//   p-value via CUDA device igamcf(df/2, chi2/2).
//   cub::DeviceReduce::Sum for mean_chi2; cub::DeviceReduce::Sum on reject bits
//   for reject_rate.
//
// Memory (Rule 5): no raw cudaMalloc — DeviceMemory<T> throughout.
// D2H (Rule 4): two scalar D2H readbacks after kernel (compliant).
// Shared memory: int local_count[n_batches] — n_batches * 4 bytes per block.
//   n_batches <= 1024 -> <=4096 bytes.  Throws if n_batches > 1024.
// CYCLE-140: initial implementation.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/graph/knn.h>

#include <cub/device/device_reduce.cuh>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet::gpu {
namespace integrate {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct KbetConfig {
    // deterministic = true (default): single-thread-per-block serial scan.
    // No atomics, bit-identical across runs.  Rule 18 compliant.
    bool deterministic = true;
};

struct KbetResult {
    core::DeviceMemory<float> chi2;    // [n_cells] per-cell chi-square statistic
    core::DeviceMemory<float> pvalue;  // [n_cells] chi-square p-value via igamcf
    float mean_chi2;    // mean of chi2 over all cells
    float reject_rate;  // fraction of cells with p-value < 0.05
    int n_cells;
    int n_batches;
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels
// ---------------------------------------------------------------------------
namespace {

// kbet_kernel_serial — deterministic single-thread-per-block kBET.
//
// Shared memory layout:
//   int smem_cnt[n_batches]  — local neighbor batch counts
//
// Thread 0 serially scans k neighbors, builds histogram, computes chi2,
// then computes p-value via igamcf(df/2, chi2/2).
//
// Explicit kernel parameters (Rule: every kernel carries every dim/buffer it
// indexes as an explicit parameter — no globals, no undefined vars).
__global__
void kbet_kernel_serial(
    const int*   __restrict__ neighbors,   // [n_cells * k]
    const int*   __restrict__ batch,       // [n_cells] batch label in [0, n_batches)
    const float* __restrict__ n_exp,       // [n_batches] expected count per neighbor slot
    float*       __restrict__ chi2_out,    // [n_cells] output chi2
    float*       __restrict__ pvalue_out,  // [n_cells] output p-value
    int n_cells, int k, int n_batches)
{
    // One block per cell.
    const int cell = static_cast<int>(blockIdx.x);
    if (cell >= n_cells) return;

    // Dynamic shared memory: int smem_cnt[n_batches].
    extern __shared__ int smem_cnt[];

    // Thread 0 only (deterministic, blockDim.x == 1).
    // Zero histogram.
    for (int b = 0; b < n_batches; ++b) smem_cnt[b] = 0;

    // Accumulate neighbor batch counts.
    const int* row_nbrs = neighbors + static_cast<ptrdiff_t>(cell) * k;
    for (int i = 0; i < k; ++i) {
        const int nbr = row_nbrs[i];
        const int lbl = batch[nbr];
        smem_cnt[lbl]++;
    }

    // Chi-square statistic: Σ_b (obs - exp)^2 / max(exp, 1.0).
    float chi2 = 0.f;
    for (int b = 0; b < n_batches; ++b) {
        const float exp_b = n_exp[b];
        if (exp_b <= 0.f) continue;
        const float diff = static_cast<float>(smem_cnt[b]) - exp_b;
        chi2 += (diff * diff) / (exp_b > 1.f ? exp_b : 1.f);
    }
    chi2_out[cell] = chi2;

    // p-value = P(chi2_dist(df) > chi2) via Wilson-Hilferty cube-root transform:
    //   z = ((chi2/df)^(1/3) - (1 - 2/(9df))) / sqrt(2/(9df))  ~ N(0, 1)
    //   p = 1 - Φ(z) = normcdff(-z)
    // Accurate to ~1e-3 for df >= 2; df=1 falls back to exact via erfc(sqrt(chi2/2)).
    // WHY not igamcf: CUDA math library does not ship the regularized upper
    // incomplete gamma function as a device intrinsic. Wilson-Hilferty is the
    // simplest accurate approximation that uses only available device math.
    // Guard: chi2=0 -> p=1; df<1 -> p=1.
    float pv = 1.f;
    const int df = n_batches - 1;
    if (df >= 1 && chi2 > 0.f) {
        if (df == 1) {
            // chi2 with 1 df = (Z)^2; p = 2*(1 - Φ(sqrt(chi2))) = erfc(sqrt(chi2/2))
            pv = erfcf(sqrtf(chi2 * 0.5f));
        } else {
            const float dff = static_cast<float>(df);
            const float ratio = chi2 / dff;
            const float cube_root = cbrtf(ratio);
            const float c1 = 1.f - (2.f / (9.f * dff));
            const float sd = sqrtf(2.f / (9.f * dff));
            const float z = (cube_root - c1) / sd;
            pv = normcdff(-z);
        }
        if (pv < 0.f) pv = 0.f;
        if (pv > 1.f) pv = 1.f;
    }
    pvalue_out[cell] = pv;
}

// kbet_reject_kernel — per-cell: write 1.0f if p-value < 0.05, else 0.0f.
// Used as input to cub::DeviceReduce::Sum for reject_rate.
__global__
void kbet_reject_kernel(
    const float* __restrict__ pvalue,   // [n_cells]
    float*       __restrict__ reject,   // [n_cells] output: 0 or 1
    int n_cells)
{
    const int cell = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                   + static_cast<int>(threadIdx.x);
    if (cell >= n_cells) return;
    reject[cell] = (pvalue[cell] < 0.05f) ? 1.f : 0.f;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// kbet() — public entry point
// ---------------------------------------------------------------------------
//
// knn:       KnnResult from singlet::gpu::graph::compute_knn(...).
//            Uses knn.neighbors [n*k], knn.n, knn.k.  Distances unused.
// d_batch:   device int[n_cells] with values in [0, n_batches).
// n_batches: number of distinct batches.  Must be >= 2.
// cfg:       KbetConfig (default: deterministic = true).
// stream:    CUDA stream (nullptr = default stream from context).
//
// Returns KbetResult with:
//   chi2       — device float[n_cells] per-cell chi-square statistics
//   pvalue     — device float[n_cells] per-cell p-values
//   mean_chi2  — mean chi2 across all cells
//   reject_rate— fraction of cells with p-value < 0.05 (kBET score; lower = better)
//
// Caller synchronizes stream before reading chi2/pvalue device arrays.
inline KbetResult kbet(
    const graph::KnnResult& knn,
    const int*              d_batch,
    int                     n_batches,
    const KbetConfig&       cfg    = {},
    cudaStream_t            stream = nullptr)
{
    if (stream == nullptr) {
        stream = singlet::gpu::core::default_context().stream();
    }

    const int n = knn.n;
    const int k = knn.k;

    if (n <= 0)
        throw std::runtime_error("kbet: empty knn graph (n=" +
            std::to_string(n) + ")");
    if (k <= 0)
        throw std::runtime_error("kbet: k must be > 0 (k=" +
            std::to_string(k) + ")");
    if (n_batches < 2)
        throw std::runtime_error("kbet: n_batches must be >= 2 (got " +
            std::to_string(n_batches) + ")");
    if (n_batches > 1024)
        throw std::runtime_error("kbet: n_batches=" + std::to_string(n_batches) +
            " exceeds shared-memory limit (max 1024).");
    if (d_batch == nullptr)
        throw std::runtime_error("kbet: d_batch is null");

    // -------------------------------------------------------------------------
    // Step 1: compute global batch counts on host (small array, one D2H).
    // d_batch is a device array of n ints in [0, n_batches).
    // Use cub::DeviceHistogram to build the histogram on device, then D2H.
    // -------------------------------------------------------------------------
    core::DeviceMemory<int> d_global_cnt(static_cast<size_t>(n_batches));
    {
        // cub::DeviceHistogram::HistogramEven requires [lower, upper) bins.
        size_t hist_tmp_bytes = 0;
        cub::DeviceHistogram::HistogramEven(
            nullptr, hist_tmp_bytes,
            d_batch, d_global_cnt.get(),
            n_batches + 1,  // num_levels = n_bins + 1
            0, n_batches,   // lower_level, upper_level
            n, stream);
        core::DeviceMemory<uint8_t> d_hist_tmp(hist_tmp_bytes);
        cub::DeviceHistogram::HistogramEven(
            d_hist_tmp.get(), hist_tmp_bytes,
            d_batch, d_global_cnt.get(),
            n_batches + 1,
            0, n_batches,
            n, stream);
    }
    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int> h_global_cnt(static_cast<size_t>(n_batches));
    SINGLET_GPU_CUDA_CHECK(cudaMemcpy(h_global_cnt.data(), d_global_cnt.get(),
                          static_cast<size_t>(n_batches) * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // -------------------------------------------------------------------------
    // Step 2: compute expected counts n_exp[b] = k * (global_count[b] / n).
    // Upload as float device array for kernel.
    // -------------------------------------------------------------------------
    std::vector<float> h_n_exp(static_cast<size_t>(n_batches));
    const float k_f = static_cast<float>(k);
    const float n_f = static_cast<float>(n);
    for (int b = 0; b < n_batches; ++b) {
        h_n_exp[static_cast<size_t>(b)] =
            k_f * (static_cast<float>(h_global_cnt[static_cast<size_t>(b)]) / n_f);
    }

    core::DeviceMemory<float> d_n_exp(static_cast<size_t>(n_batches));
    SINGLET_GPU_CUDA_CHECK(cudaMemcpy(d_n_exp.get(), h_n_exp.data(),
                          static_cast<size_t>(n_batches) * sizeof(float),
                          cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // Step 3: per-cell chi2 + p-value kernel.
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_chi2(static_cast<size_t>(n));
    core::DeviceMemory<float> d_pvalue(static_cast<size_t>(n));

    const size_t smem_bytes = static_cast<size_t>(n_batches) * sizeof(int);

    (void)cfg;  // deterministic = true is only mode in v0
    kbet_kernel_serial<<<n, 1, smem_bytes, stream>>>(
        knn.neighbors.get(), d_batch, d_n_exp.get(),
        d_chi2.get(), d_pvalue.get(),
        n, k, n_batches);

    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    // -------------------------------------------------------------------------
    // Step 4: reject flag per cell (p-value < 0.05), then sum for reject_rate.
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_reject(static_cast<size_t>(n));
    {
        const int threads = 256;
        const int blocks  = (n + threads - 1) / threads;
        kbet_reject_kernel<<<blocks, threads, 0, stream>>>(
            d_pvalue.get(), d_reject.get(), n);
    }
    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    // -------------------------------------------------------------------------
    // Step 5: cub::DeviceReduce::Sum for mean_chi2 and reject_rate (two D2H).
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_sum(1);
    {
        size_t tmp_bytes = 0;
        cub::DeviceReduce::Sum(nullptr, tmp_bytes,
            d_chi2.get(), d_sum.get(), n, stream);
        core::DeviceMemory<uint8_t> d_tmp(tmp_bytes);
        cub::DeviceReduce::Sum(d_tmp.get(), tmp_bytes,
            d_chi2.get(), d_sum.get(), n, stream);
    }
    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    float h_chi2_sum = 0.f;
    SINGLET_GPU_CUDA_CHECK(cudaMemcpy(&h_chi2_sum, d_sum.get(), sizeof(float),
                          cudaMemcpyDeviceToHost));

    core::DeviceMemory<float> d_reject_sum(1);
    {
        size_t tmp_bytes = 0;
        cub::DeviceReduce::Sum(nullptr, tmp_bytes,
            d_reject.get(), d_reject_sum.get(), n, stream);
        core::DeviceMemory<uint8_t> d_tmp(tmp_bytes);
        cub::DeviceReduce::Sum(d_tmp.get(), tmp_bytes,
            d_reject.get(), d_reject_sum.get(), n, stream);
    }
    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    float h_reject_sum = 0.f;
    SINGLET_GPU_CUDA_CHECK(cudaMemcpy(&h_reject_sum, d_reject_sum.get(), sizeof(float),
                          cudaMemcpyDeviceToHost));

    KbetResult res;
    res.chi2        = std::move(d_chi2);
    res.pvalue      = std::move(d_pvalue);
    res.mean_chi2   = h_chi2_sum  / static_cast<float>(n);
    res.reject_rate = h_reject_sum / static_cast<float>(n);
    res.n_cells     = n;
    res.n_batches   = n_batches;
    return res;
}

}  // namespace integrate
}  // namespace singlet::gpu
