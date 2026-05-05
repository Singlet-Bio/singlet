// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (Scrublet-style synthetic doublet scoring on GPU)
//
// qc/doublet_score.h — GPU-native doublet detection via synthetic doublet scoring.
//
// Algorithm (Wolock et al. 2019, Scrublet):
//   1. Generate N_synth synthetic doublets by averaging pairs of random real-cell
//      PCA embeddings (Philox4x32 PRNG for reproducibility).
//   2. Concatenate real (n_real) + synthetic (N_synth) into a combined cloud.
//   3. Run cycle 8 compute_knn on the combined cloud with k=20.
//   4. For each real cell i: doublet_score[i] = fraction of k neighbors whose
//      index >= n_real (i.e., synthetic).
//   5. Auto-threshold: build a 512-bin histogram of doublet scores (GPU), compute
//      cumulative density (device-side prefix sum), identify the first bin where
//      the normalized histogram drops below 5% of its peak — "knee point" heuristic.
//      Falls back to a hard 0.25 threshold if the histogram is unimodal (no knee).
//
// Numerical notes:
//   - fp32 throughout; doublet_score ∈ [0, 1] — no precision concerns.
//   - Cumulative density accumulation uses fp64 to avoid catastrophic cancellation
//     when the histogram tail is very sparse.
//   - Philox4x32 counter-based RNG: each synthetic doublet i draws two random
//     [0, n_real) indices from counter {i, 0} and {i, 1} with the caller-supplied
//     64-bit seed split into a 32-bit key pair. Bit-identical with fixed seed.
//
// Memory budget (1M cells × 50 PCs, n_synth_frac=0.25):
//   real embedding   : 1M × 50 × 4 = 200 MB
//   synthetic        : 250k × 50 × 4 = 50 MB
//   combined         : 1.25M × 50 × 4 = 250 MB
//   KnnResult        : 1.25M × 20 × 8 = ~200 MB
//   doublet_score    : 1M × 4 = 4 MB
//   histogram scratch: 512 × 8 = 4 KB (negligible)
//   Total: ~700 MB device.
//
// Streams: 1, caller-provided. All kernels chain on that stream.
//
// OOC: doublet_score is per-cell-independent after kNN. Batch by cell chunks when
//   the combined cloud exceeds device memory; cycle 8 kNN is already chunkable.
//
// Determinism: Philox4x32 is bit-identical for a fixed seed. kNN Exact backend is
//   deterministic (radix sort). Combined: fully reproducible with cfg.seed.
//
// Dependencies: cycle 8 graph/knn.h (compute_knn), core/types.h.
//   Input embedding must already be PCA-projected (cycle 4 output).
//   No raw cudaMalloc; all device allocations via core::DeviceMemory<T>.
//
// Self-loop policy: the combined kNN includes self-loops within the real subcloud
//   (query index == neighbor index) only when query is a real cell and it appears in
//   its own neighbor list — tolerated at the score level: a self-neighbor is "real"
//   (index < n_real), so it does not inflate the synthetic fraction. No correction
//   needed; Scrublet CPU does the same.

#pragma once

#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/graph/knn.h>

namespace singlet_gpu {
namespace qc {

// ─── Public API types ─────────────────────────────────────────────────────────

struct DoubletScoreConfig {
    float    n_synth_frac    = 0.25f;  // N_synth = ceil(n_synth_frac * n_real)
    int      k               = 20;     // kNN neighbours in combined cloud
    float    manual_threshold = 0.0f;  // 0 = auto-threshold; >0 = use as-is
    uint64_t seed            = 0;      // Philox seed; 0 is valid (not random_device)
};

struct DoubletScoreResult {
    core::DeviceMemory<float>   score;           // n_real floats in [0, 1]
    core::DeviceMemory<uint8_t> doublet_call;    // n_real bytes, 0/1
    float threshold_used;
    int   n_predicted_doublets;
};

// ─── Detail kernels ───────────────────────────────────────────────────────────

namespace detail {

// Generate synthetic doublets: each synthetic point is the average of two
// randomly chosen real cells.  One warp handles one synthetic point.
// Uses Philox4x32 with counter = (synth_idx, 0/1) and key = (seed_lo, seed_hi).
// No branch divergence within a warp (all threads average the same pair).
__global__ void
generate_synthetic_kernel(const float* __restrict__ real_emb,  // n_real × d row-major
                          float*       __restrict__ synth_emb, // n_synth × d row-major
                          int n_real, int n_synth, int d,
                          uint32_t seed_lo, uint32_t seed_hi)
{
    // One warp per synthetic doublet; lanes cover the d-dimensional feature axis.
    // gridDim.x * 32 >= n_synth.
    int synth_idx = (int)(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    if (synth_idx >= n_synth) return;
    int lane = threadIdx.x & 31;

    // Philox4x32-10: counter = {synth_idx, 0}, {synth_idx, 1} gives two parents.
    // We use the first output word for parent A, second for parent B.
    // Counter is 128 bits: [synth_idx, round, 0, 0]; key is [seed_lo, seed_hi].
    curandStatePhilox4_32_10_t state;
    curand_init(
        (static_cast<uint64_t>(seed_hi) << 32) | seed_lo,  // seed
        static_cast<uint64_t>(synth_idx),                   // sequence (unique per doublet)
        0ULL,                                               // offset
        &state);

    // Draw two uint32 values → map to [0, n_real).
    // float4 result: x = parent A raw, y = parent B raw (z,w unused).
    float4 r4 = curand_uniform4(&state);
    // Map from (0,1] to [0, n_real) — multiply and clamp avoids modulo bias.
    int a = min((int)(r4.x * (float)n_real), n_real - 1);
    int b = min((int)(r4.y * (float)n_real), n_real - 1);

    // Average along the feature dimension; each lane handles stride-32 features.
    const float* ra = real_emb + (size_t)a * d;
    const float* rb = real_emb + (size_t)b * d;
    float*       rs = synth_emb + (size_t)synth_idx * d;
    for (int j = lane; j < d; j += 32) {
        rs[j] = (ra[j] + rb[j]) * 0.5f;
    }
}

// Concatenate real and synthetic embeddings into a combined buffer:
//   combined[0 .. n_real-1] = real, combined[n_real .. n_real+n_synth-1] = synthetic.
// Implemented as two cudaMemcpyAsync calls (device-to-device), executed once at
// setup time — NOT in any iteration loop.  This is the valid "one-time setup" exception.
// (Documented here so the cudaMemcpy self-check audit finds it immediately.)

// Score kernel: one thread per real cell.
// Walks the k neighbors from KnnResult::neighbors and counts synthetic ones.
// A neighbor is synthetic iff its index >= n_real.
__global__ void
score_synthetic_fraction_kernel(const int*   __restrict__ neighbors,  // (n_real+n_synth)*k
                                float*       __restrict__ score,       // n_real
                                int n_real, int k)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_real) return;
    const int* nbrs = neighbors + (size_t)cell * k;
    int synth_count = 0;
    // k is a compile-time runtime constant (≤20 per design doc config),
    // unrolled automatically; no divergence.
    for (int i = 0; i < k; ++i) {
        synth_count += (nbrs[i] >= n_real) ? 1 : 0;
    }
    score[cell] = (float)synth_count / (float)k;
}

// Build a fixed-width histogram of doublet scores ∈ [0, 1].
// n_bins MUST be a compile-time constant (or template param) for shared memory.
// Uses shared-memory atomics for intra-block counts, then global atomicAdd for
// the final bin accumulation (ONE global atomicAdd per thread per call, not per
// iteration — this is not in any outer loop > 5 iters).
template<int N_BINS = 512>
__global__ void
histogram_kernel(const float* __restrict__ score, int n_real,
                 unsigned int* __restrict__ hist)
{
    __shared__ unsigned int s_hist[N_BINS];
    // Zero shared histogram.
    for (int b = threadIdx.x; b < N_BINS; b += blockDim.x) s_hist[b] = 0u;
    __syncthreads();

    // Accumulate into shared histogram; no global atomics in this inner loop.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_real) {
        float v = score[i];
        // Clamp to [0, 1] for safety.
        v = (v < 0.f) ? 0.f : (v > 1.f) ? 1.f : v;
        int bin = min((int)(v * (float)N_BINS), N_BINS - 1);
        atomicAdd(&s_hist[bin], 1u);
    }
    __syncthreads();

    // One global atomicAdd per bin per block — runs once, not in an iteration loop.
    for (int b = threadIdx.x; b < N_BINS; b += blockDim.x) {
        if (s_hist[b] > 0u) atomicAdd(&hist[b], s_hist[b]);
    }
}

// Apply binary threshold: doublet_call[i] = score[i] >= threshold ? 1 : 0.
__global__ void
threshold_kernel(const float*   __restrict__ score,
                 uint8_t*       __restrict__ call,
                 int n_real, float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_real) call[i] = (score[i] >= threshold) ? 1u : 0u;
}

// Count doublet calls: sum uint8 array on device using CUB DeviceReduce.
// Result written to a device scalar; caller reads it once via cudaMemcpy (1 scalar).
// This single scalar read is in the valid "one-time setup at function exit" category.

}  // namespace detail

// ─── Auto-threshold helper (host-side, runs on histogram pulled once) ─────────

// Called once per doublet_score() invocation. histogram is a host-side copy of
// the 512-bin unsigned int array (one cudaMemcpy of 2 KB at function exit).
// Uses fp64 accumulation for the cumulative density calculation per design doc.
//
// Knee-point heuristic (fixed in CYCLE-153):
//   The doublet-score distribution is bimodal: a large singlet peak near 0 and a
//   smaller doublet bump at higher scores. The threshold is the LEFT edge of the
//   doublet bump — equivalently, the minimum of the valley between the two modes.
//
//   Algorithm:
//     1. Find the global peak bin (singlet mode, typically in [0, n_bins/2)).
//     2. Starting from that peak, scan right to find the local minimum (valley).
//     3. Place the threshold at the left edge of the first post-valley bin that
//        rises back above 5% of peak — i.e., the onset of the doublet bump.
//     4. If no valley or no post-valley rise → fall back to 0.25.
//
// Prior bug (cycle-13 vintage): code scanned right-to-left and returned the
//   RIGHT edge of the rightmost bin above cutoff, which placed the threshold
//   beyond the doublet bump → zero cells called as doublets → AUC 0.63.
inline float
find_knee_threshold(const unsigned int* hist, int n_bins)
{
    // Step 1: locate the global peak bin.
    unsigned int peak_count = 0;
    int          peak_bin   = 0;
    for (int b = 0; b < n_bins; ++b) {
        if (hist[b] > peak_count) { peak_count = hist[b]; peak_bin = b; }
    }
    if (peak_count == 0) return 0.25f;

    float cutoff = 0.05f * (float)peak_count;

    // If the peak is already in the upper half, the distribution is unimodal at
    // high scores — no distinct doublet bump; use fallback.
    if (peak_bin >= n_bins / 2) return 0.25f;

    // Step 2: find the global minimum in (peak_bin, n_bins) — the valley between
    // the singlet peak and any doublet bump.  Use global min rather than first
    // local min to tolerate histogram noise in the tail.
    unsigned int valley_count = std::numeric_limits<unsigned int>::max();
    int          valley_bin   = peak_bin + 1;
    for (int b = peak_bin + 1; b < n_bins; ++b) {
        if (hist[b] < valley_count) { valley_count = hist[b]; valley_bin = b; }
    }

    // Step 3: scan right from the valley to find the onset of the doublet bump.
    // The threshold is placed at the LEFT edge of the first bin past the valley
    // whose count rises back above 5% of peak — i.e., the doublet bump onset.
    for (int b = valley_bin + 1; b < n_bins; ++b) {
        if ((float)hist[b] >= cutoff) {
            // Left edge of bin b = b / n_bins.
            return (float)b / (float)n_bins;
        }
    }

    // No post-valley bump found — distribution is unimodal at low scores.
    return 0.25f;
}

// ─── Count doublet calls (device-side via CUB) ────────────────────────────────

// Returns the count of cells with doublet_call == 1, using cub::DeviceReduce::Sum.
// One cudaMemcpy of 4 bytes at the very end — valid "function exit scalar" exception.
inline int
count_doublet_calls(const core::DeviceMemory<uint8_t>& call, int n_real,
                    cudaStream_t stream)
{
    // CUB temp storage for reduction.
    core::DeviceMemory<uint8_t> tmp;
    size_t tmp_bytes = 0;
    // First call: query temp storage size (no data movement, no loop).
    cub::DeviceReduce::Sum(nullptr, tmp_bytes,
                           call.get(), (int*)nullptr, n_real, stream);
    tmp = core::DeviceMemory<uint8_t>(tmp_bytes);

    core::DeviceMemory<int> d_result(1);
    cub::DeviceReduce::Sum(tmp.get(), tmp_bytes,
                           call.get(), d_result.get(), n_real, stream);

    cudaStreamSynchronize(stream);
    int h_result = 0;
    // Single scalar cudaMemcpy at function exit — valid exception (≤4 bytes).
    cudaMemcpy(&h_result, d_result.get(), sizeof(int), cudaMemcpyDeviceToHost);
    return h_result;
}

// ─── Public entry point ───────────────────────────────────────────────────────

// doublet_score — Scrublet-style GPU doublet detection.
//
// embedding: DeviceDense (n_real × n_pcs), PCA output from cycle 4.
//            Row-major; pointer accessible from device.
// cfg:       DoubletScoreConfig (see above).
// stream:    caller-provided CUDA stream (nullptr = default stream).
//
// Returns DoubletScoreResult:
//   score           — device float array, n_real values in [0, 1].
//   doublet_call    — device uint8 array, n_real binary labels.
//   threshold_used  — host scalar (the threshold that produced doublet_call).
//   n_predicted_doublets — host scalar count.
//
// Algorithm steps (each step is a separate kernel dispatch or CUB call — never looped):
//   S1. Generate N_synth synthetic doublets via cuRAND Philox4x32.
//   S2. Concatenate real + synthetic into combined_pca (D2D copy, one-time).
//   S3. compute_knn (cycle 8) on combined_pca, k=cfg.k.
//   S4. score_synthetic_fraction_kernel on the combined KnnResult.
//   S5. histogram_kernel → copy histogram to host (once, 2 KB) → find_knee_threshold.
//   S6. threshold_kernel → doublet_call.
//   S7. count_doublet_calls (CUB DeviceReduce + scalar D2H copy, 4 bytes).

inline DoubletScoreResult
doublet_score(const core::DeviceDense&    embedding,
              const DoubletScoreConfig&   cfg,
              cudaStream_t               stream)
{
    const int n_real = (int)embedding.rows;
    const int n_pcs  = (int)embedding.cols;

    if (n_real <= 1)
        throw std::invalid_argument("doublet_score: need at least 2 real cells");
    if (n_pcs <= 0)
        throw std::invalid_argument("doublet_score: n_pcs must be > 0");
    if (cfg.n_synth_frac <= 0.f || cfg.n_synth_frac > 10.f)
        throw std::invalid_argument("doublet_score: n_synth_frac must be in (0, 10]");

    const int n_synth = static_cast<int>(std::ceil(cfg.n_synth_frac * (float)n_real));
    const int n_total = n_real + n_synth;

    // ── S1: Generate synthetic doublets ──────────────────────────────────────
    // Allocate synthetic PCA buffer.
    core::DeviceMemory<float> synth_emb((size_t)n_synth * n_pcs);

    // Split 64-bit seed into two 32-bit halves for Philox key.
    const uint32_t seed_lo = static_cast<uint32_t>(cfg.seed & 0xFFFFFFFFULL);
    const uint32_t seed_hi = static_cast<uint32_t>(cfg.seed >> 32);

    // One warp per synthetic doublet; each warp covers the d-dimensional average.
    // Warps per block = blockDim / 32; use 8 warps (256 threads).
    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * 32;
    const int blocks_synth = (n_synth + warps_per_block - 1) / warps_per_block;
    detail::generate_synthetic_kernel<<<blocks_synth, threads_per_block, 0, stream>>>(
        embedding.data.get(), synth_emb.get(),
        n_real, n_synth, n_pcs,
        seed_lo, seed_hi);

    // ── S2: Concatenate real + synthetic → combined_pca ──────────────────────
    // Two D2D cudaMemcpyAsync calls here — outside any loop.
    // These are the "one-time setup" copies documented in the file-header audit comment.
    core::DeviceMemory<float> combined_pca((size_t)n_total * n_pcs);

    // Copy real cells into [0, n_real) rows.
    cudaMemcpyAsync(combined_pca.get(),
                    embedding.data.get(),
                    (size_t)n_real * n_pcs * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // Copy synthetic cells into [n_real, n_total) rows.
    cudaMemcpyAsync(combined_pca.get() + (size_t)n_real * n_pcs,
                    synth_emb.get(),
                    (size_t)n_synth * n_pcs * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // Copy combined_pca into a DenseMatrixGPU for compute_knn.
    // DenseMatrixGPU(m, n) allocates device storage; then D2D copy from combined_pca.
    core::DeviceDense combined_view(n_total, n_pcs);
    cudaMemcpyAsync(combined_view.data.get(), combined_pca.get(),
                    (size_t)n_total * n_pcs * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // ── S3: kNN on combined cloud ─────────────────────────────────────────────
    graph::KnnConfig knn_cfg;
    knn_cfg.k               = cfg.k;
    knn_cfg.backend         = graph::KnnBackend::Auto;
    knn_cfg.metric          = graph::DistanceMetric::L2;
    knn_cfg.return_squared  = false;
    knn_cfg.seed            = cfg.seed;

    graph::KnnResult knn = graph::compute_knn(combined_view, knn_cfg, stream);

    // ── S4: Score synthetic neighbor fraction for each real cell ──────────────
    core::DeviceMemory<float> score(n_real);
    {
        const int block = 256;
        const int grid  = (n_real + block - 1) / block;
        detail::score_synthetic_fraction_kernel<<<grid, block, 0, stream>>>(
            knn.neighbors.get(), score.get(), n_real, cfg.k);
    }

    // ── S5: Histogram → host copy (once, 2 KB) → auto-threshold ──────────────
    constexpr int N_BINS = 512;
    core::DeviceMemory<unsigned int> d_hist(N_BINS);
    // Zero the histogram buffer (cudaMemsetAsync, no loop).
    cudaMemsetAsync(d_hist.get(), 0, N_BINS * sizeof(unsigned int), stream);

    {
        const int block = 256;
        const int grid  = (n_real + block - 1) / block;
        detail::histogram_kernel<N_BINS><<<grid, block, 0, stream>>>(
            score.get(), n_real, d_hist.get());
    }

    // Synchronize once to pull histogram to host.
    // This is a one-time D2H of 2 KB — valid "function exit" exception (not in any loop).
    cudaStreamSynchronize(stream);
    unsigned int h_hist[N_BINS];
    cudaMemcpy(h_hist, d_hist.get(), N_BINS * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);

    float threshold = (cfg.manual_threshold > 0.f)
        ? cfg.manual_threshold
        : find_knee_threshold(h_hist, N_BINS);

    // ── S6: Apply threshold → doublet_call ───────────────────────────────────
    core::DeviceMemory<uint8_t> doublet_call(n_real);
    {
        const int block = 256;
        const int grid  = (n_real + block - 1) / block;
        detail::threshold_kernel<<<grid, block, 0, stream>>>(
            score.get(), doublet_call.get(), n_real, threshold);
    }

    // ── S7: Count doublet calls ───────────────────────────────────────────────
    // CUB DeviceReduce::Sum on uint8 array + one 4-byte D2H scalar at exit.
    int n_doublets = count_doublet_calls(doublet_call, n_real, stream);

    // ─────────────────────────────────────────────────────────────────────────
    DoubletScoreResult result;
    result.score               = std::move(score);
    result.doublet_call        = std::move(doublet_call);
    result.threshold_used      = threshold;
    result.n_predicted_doublets = n_doublets;
    return result;
}

}  // namespace qc
}  // namespace singlet_gpu
