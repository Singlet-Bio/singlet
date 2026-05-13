// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (first GPU Numbat-style CNA detection)
//
// cna/numbat.h — GPU-native Numbat-style copy-number alteration detection
//
// Algorithm reference: Gao et al. 2024 (Numbat), Genome Biology.
// This is the first GPU implementation of Numbat-style CNA detection.
//
// Pipeline (one call to detect_cna):
//   1. Per-gene log expression ratio vs reference normal (or bulk mean).
//   2. Chromosome-sorted tiling: process one chromosome at a time (OOC strategy).
//   3. Direct rolling-window smoothing per chromosome (window ≤15 genes → direct kernel;
//      cuFFT path deferred to follow-up cycle for larger windows).
//   4. HMM forward-backward per cell per chromosome in log-space (fp32).
//      States = {loss=0, neutral=1, gain=2}. Emissions = smoothed log-ratios.
//      Viterbi-style argmax → CNA state per segment.
//   5. Clone clustering: leiden on per-cell CNA pattern (reuses graph/leiden.h).
//      kNN built via compute_exact (direct float pointer path, avoids knn.h wrapper).
//
// Memory layout:
//   Input expression: DeviceCSC (genes × cells), CSC (col = cell).
//   Per-chromosome workspace: n_cells × genes_per_chr × 4 bytes.
//     For 1500 genes/chr × 100k cells: ~600 MB per chromosome tile.
//   HMM state (per chromosome): n_cells × n_segs_chr × 3 × 4 bytes.
//     For 100k × 50 × 3: ~60 MB.
//   Output: n_cells × n_total_segs × 1 byte (uint8 state 0/1/2).
//          n_cells × n_total_segs × 3 × 4 bytes (fp32 posteriors).
//
// OOC: Per-chromosome tile loop keeps workspace bounded to genes_per_chr × n_cells.
//      Each chromosome is independent; results written to the global segment array
//      as each chromosome finishes. Only one chromosome tile is resident at a time.
//
// Streams: 1, caller-provided. Never creates a stream internally.
// Precision: fp32 throughout; HMM in log-space to prevent underflow.
// Determinism: clone-clustering seeded via cfg.seed → leiden.
//
// cuRAND: Not used directly here (Philox is inside leiden.h).
// cuFFT: Not used (smoothing window ≤ MAX_SMOOTH_WINDOW → direct kernel).
//
// Deferred to follow-up:
//   - Allele-specific CNA integration using snp_ad / snp_dp from singlet.
//   - Spatial regularization.
//   - cuFFT-based smoothing for windows > MAX_SMOOTH_WINDOW.
//
// Workspace budget at 100k cells × 22 chromosomes × 50 segments:
//   expr tile (per chr, 1500 genes max): 1500 × 100k × 4 = 600 MB
//   log_ratio + smoothed: 2 × 600 MB (in flight simultaneously)
//   segment means: 100k × 50 × 4 = 20 MB
//   HMM forward+backward+posterior: 100k × 50 × 3 × 3 × 4 = 180 MB
//   cna_states output: 100k × (22×50) × 1 = 110 MB
//   posteriors output: 100k × (22×50) × 3 × 4 = 1.32 GB
//   kNN CNA float copy: 100k × 1100 × 4 = 440 MB
//   Total peak (one chromosome resident, plus outputs): ~3.3 GB (safe on A100/80 GB)
//
// cudaMemcpy self-audit (see inline comments at each call site):
//   [A] cudaMemcpyAsync H→D: seg boundary int arrays per chromosome.
//       Location: chr loop body, after segment assignment.
//       Per-chr, 2 arrays of n_segs_chr ints (≤50 ints × 4 = 200 bytes each).
//       Loop iter count: O(22). VALID: one-time per-chromosome setup.
//   [B] cudaMemcpyAsync D→D: expr tile zero-fill is cudaMemsetAsync (not memcpy).
//   No other cudaMemcpy in hot paths. Scalar copies for leiden n_clusters are
//   inside leiden.h (already audited in cycle 7).

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/graph/leiden.h>
#include <singlet-gpu/graph/knn.h>

#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu {
namespace cna {

// ─── Constants ────────────────────────────────────────────────────────────────

// HMM state indices — fixed, never reordered (affects binary output format).
static constexpr int HMM_LOSS    = 0;
static constexpr int HMM_NEUTRAL = 1;
static constexpr int HMM_GAIN    = 2;
static constexpr int HMM_NSTATES = 3;

// Direct smoothing kernel is O(n_genes × window); cuFFT deferred above this.
static constexpr int MAX_SMOOTH_WINDOW = 15;

// ─── Public API types ─────────────────────────────────────────────────────────

// GeneAnnotation: per-gene chromosome assignment and genomic position.
// Chromosomes are 1-indexed integers (1–22, 23=X, 24=Y, 25=MT).
// Genes must be indexed identically to the rows of the DeviceCSC input.
struct GeneAnnotation {
    std::vector<int>      chr;       // chromosome per gene (1-based)
    std::vector<uint32_t> start_bp;  // genomic start position per gene (bp)
    int n_genes() const { return static_cast<int>(chr.size()); }
};

// SegmentBoundary: one genomic segment = contiguous run of genes on one chromosome.
// gene_start and gene_end use global gene indices (same index space as DeviceCSC rows).
struct SegmentBoundary {
    int chr;        // chromosome (1-based)
    int gene_start; // first gene index (global)
    int gene_end;   // one-past-last gene index (global)
    int seg_idx;    // segment index in the CNA state matrix (0-based)
};

// CnaConfig: tunable parameters. Defaults calibrated for 10x scRNA-seq tumor data.
struct CnaConfig {
    // Direct rolling-window half-width; full window = 2*smooth_half_win+1.
    // Must satisfy (2*smooth_half_win+1) ≤ MAX_SMOOTH_WINDOW.
    int smooth_half_win = 5;  // full window = 11

    // Skip chromosomes with fewer than this many genes (avoids single-gene segs).
    int min_genes_per_chr = 10;

    // Gaussian emission model: per-state mean and σ for the smoothed log2 ratio.
    float loss_mean    = -1.0f;
    float loss_sigma   =  1.0f;
    float neutral_mean =  0.0f;
    float neutral_sigma = 0.5f;
    float gain_mean    =  0.5f;
    float gain_sigma   =  1.0f;

    // HMM off-diagonal transition probability per segment boundary.
    // p(self) = 1 - 2*trans_prob. Typical value: 0.01 → rare state changes.
    float trans_prob = 0.01f;

    // Number of HMM segments per chromosome (genes binned uniformly).
    int segments_per_chr = 50;

    // Whether to include sex chromosomes (23=X, 24=Y) in the analysis.
    bool include_sex_chrs = false;

    // Seed for clone clustering (forwarded to leiden cfg.seed).
    uint64_t seed = 42;

    // kNN k for clone clustering (leiden input graph).
    int knn_k = 15;

    // Leiden config for clone clustering (seed overridden by cfg.seed above).
    graph::LeidenConfig leiden_cfg;
};

// CnaResult: all outputs from detect_cna().
struct CnaResult {
    // Per-cell per-segment CNA state call. Layout: [cell × n_total_segments], uint8.
    // Values: 0=loss, 1=neutral, 2=gain. Row-major (cell is outer dimension).
    core::DeviceMemory<uint8_t> cna_states;
    int n_total_segments = 0;

    // Per-cell per-segment HMM posterior probabilities. Layout:
    // [cell × n_total_segments × HMM_NSTATES], fp32. Sums to 1 along state dim.
    core::DeviceMemory<float> hmm_posteriors;

    // Clone labels from leiden clustering on CNA patterns. [n_cells], int32.
    core::DeviceMemory<int> clone_labels;
    int   n_clones           = 0;
    float leiden_modularity  = 0.f;

    // Segment boundary metadata (host-side, indexed by seg_idx).
    std::vector<SegmentBoundary> segments;
};

// ─── Detail: CUDA kernels ─────────────────────────────────────────────────────

namespace detail {

// Kernel 1: expand a CSC matrix slice (genes in [gene_start, gene_end)) into a
// dense row-major tile. Output tile is zeroed before this kernel; rows outside
// the gene range are silently skipped.
//
// WHY row-major (gene outer, cell inner): the smoothing and log-ratio kernels
// read all cells for one gene → gene-outer layout gives coalesced cell reads.
//
// col_ptr, row_indices, vals: CSC arrays from DeviceCSC (factornet field names).
// tile: (gene_end-gene_start) × n_cells, row-major. Pre-zeroed by caller.
// Each thread handles one cell column; iterates over its CSC column entries.
__global__ void __launch_bounds__(256, 4)
expand_csc_tile_kernel(
    const int*   __restrict__ col_ptr,     // DeviceCSC::col_ptr.data()
    const int*   __restrict__ row_indices, // DeviceCSC::row_indices.data()
    const float* __restrict__ vals,        // DeviceCSC::values.data()
    float* __restrict__       tile,        // n_genes_chr × n_cells, row-major
    int gene_start,
    int gene_end,
    int n_genes_chr,
    int n_cells)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    int p0 = col_ptr[c];
    int p1 = col_ptr[c + 1];
    for (int p = p0; p < p1; ++p) {
        int gene = row_indices[p];
        if (gene < gene_start || gene >= gene_end) continue;
        int local_g = gene - gene_start;
        tile[(size_t)local_g * n_cells + c] = vals[p];
    }
}

// Kernel 2: compute per-gene mean across cells.
// Shared-memory tree reduction; one block per gene.
__global__ void __launch_bounds__(256, 2)
gene_mean_kernel(
    const float* __restrict__ tile,      // n_genes_chr × n_cells, row-major
    float* __restrict__       gene_mean, // n_genes_chr
    int n_genes_chr,
    int n_cells)
{
    extern __shared__ float smem[];
    int g = blockIdx.x;
    if (g >= n_genes_chr) return;

    const float* row = tile + (size_t)g * n_cells;
    float acc = 0.f;
    for (int c = threadIdx.x; c < n_cells; c += blockDim.x)
        acc += row[c];
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        gene_mean[g] = smem[0] / fmaxf((float)n_cells, 1.f);
}

// Kernel 3: log2 expression ratio.
// log_ratio[g,c] = log2((expr[g,c]+1) / (ref[g]+1)).
// When ref pointer is null, caller must pre-fill ref with gene-mean values.
__global__ void __launch_bounds__(256, 4)
log_ratio_kernel(
    const float* __restrict__ tile,
    const float* __restrict__ ref,   // n_genes_chr; never null (caller ensures)
    float* __restrict__       out,   // n_genes_chr × n_cells, row-major
    int n_genes_chr,
    int n_cells)
{
    int g = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (g >= n_genes_chr || c >= n_cells) return;
    float v  = tile[(size_t)g * n_cells + c];
    float r  = ref[g];
    out[(size_t)g * n_cells + c] = log2f((v + 1.f) / (r + 1.f));
}

// Kernel 4: direct rolling-window mean smoothing along the gene axis.
// Window = [g-half_win, g+half_win], boundary-clamped (not reflected).
// WHY 128 threads: register pressure from the per-thread gene loop.
__global__ void __launch_bounds__(128, 4)
smooth_kernel(
    const float* __restrict__ in,     // n_genes_chr × n_cells
    float* __restrict__       out,    // n_genes_chr × n_cells
    int n_genes_chr,
    int n_cells,
    int half_win)
{
    int g = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (g >= n_genes_chr || c >= n_cells) return;

    int g0 = max(0, g - half_win);
    int g1 = min(n_genes_chr - 1, g + half_win);
    float sum = 0.f;
    for (int gi = g0; gi <= g1; ++gi)
        sum += in[(size_t)gi * n_cells + c];
    out[(size_t)g * n_cells + c] = sum / (float)(g1 - g0 + 1);
}

// Kernel 5: per-segment per-cell mean of smoothed log-ratios.
// smoothed: n_genes_chr × n_cells (gene outer, cell inner).
// out: n_cells × n_segs_chr (cell outer, segment inner).
// seg_gs[s], seg_ge[s]: gene range [gs, ge) for segment s (local gene coords).
__global__ void __launch_bounds__(128, 4)
segment_mean_kernel(
    const float* __restrict__ smoothed,
    float* __restrict__       out,
    const int* __restrict__   seg_gs,  // n_segs_chr start (local gene index)
    const int* __restrict__   seg_ge,  // n_segs_chr end   (local gene index, exclusive)
    int n_genes_chr,
    int n_cells,
    int n_segs_chr)
{
    int seg  = blockIdx.x;
    int cell = blockIdx.y * blockDim.x + threadIdx.x;
    if (seg >= n_segs_chr || cell >= n_cells) return;

    int g0 = seg_gs[seg];
    int g1 = seg_ge[seg];
    float sum = 0.f;
    int   cnt = 0;
    for (int g = g0; g < g1 && g < n_genes_chr; ++g, ++cnt)
        sum += smoothed[(size_t)g * n_cells + cell];
    out[(size_t)cell * n_segs_chr + seg] = (cnt > 0) ? sum / (float)cnt : 0.f;
}

// Kernel 6: HMM forward pass in log-space. One block per cell (1 thread).
// Cells are independent → embarrassingly parallel over blocks.
// Segments are sequential within each cell → handled by the 1 thread.
//
// WHY 1 thread per block: n_segs ≤ 50; serialisation cost is ~50 multiply-add
// cycles which is negligible. Using 3 threads for state parallelism would require
// __syncthreads() at each segment, adding overhead that exceeds the benefit at n=50.
__global__ void __launch_bounds__(1, 128)
hmm_forward_kernel(
    const float* __restrict__ seg_means,  // n_cells × n_segs, row-major
    float* __restrict__       log_alpha,   // n_cells × n_segs × 3
    int n_cells, int n_segs,
    // 9 log-transition values (from=row, to=col): loss, neutral, gain.
    float lt00, float lt01, float lt02,
    float lt10, float lt11, float lt12,
    float lt20, float lt21, float lt22,
    float lm0, float li0,   // loss:    log_mean, inv_2sigma2
    float lm1, float li1,   // neutral: log_mean, inv_2sigma2
    float lm2, float li2)   // gain:    log_mean, inv_2sigma2
{
    int cell = blockIdx.x;
    if (cell >= n_cells) return;

    const float* obs = seg_means + (size_t)cell * n_segs;
    float*       a   = log_alpha  + (size_t)cell * n_segs * HMM_NSTATES;

    // Emission function: -0.5*(x-mu)^2 * inv_2sigma2. Normalisation omitted
    // (cancels in posterior; see hmm_posterior_kernel).
    auto emit = [&](float x, int s) -> float {
        float mu, i2;
        if (s == 0) { mu = lm0; i2 = li0; }
        else if (s == 1) { mu = lm1; i2 = li1; }
        else             { mu = lm2; i2 = li2; }
        float d = x - mu;
        return -0.5f * d * d * i2;
    };

    auto ltrans = [&](int f, int t) -> float {
        if (f == 0) { if (t == 0) return lt00; if (t == 1) return lt01; return lt02; }
        if (f == 1) { if (t == 0) return lt10; if (t == 1) return lt11; return lt12; }
        /* f==2 */  { if (t == 0) return lt20; if (t == 1) return lt21; return lt22; }
    };

    // Initialise with uniform prior log(1/3).
    static constexpr float LOG_THIRD = -1.0986122886681098f;
    float ob0 = obs[0];
    for (int s = 0; s < HMM_NSTATES; ++s)
        a[s] = LOG_THIRD + emit(ob0, s);

    // Forward recursion.
    for (int t = 1; t < n_segs; ++t) {
        float ob = obs[t];
        float prev[HMM_NSTATES];
        for (int s = 0; s < HMM_NSTATES; ++s) prev[s] = a[(t-1)*HMM_NSTATES + s];
        for (int s = 0; s < HMM_NSTATES; ++s) {
            // log-sum-exp over predecessor states.
            float best = prev[0] + ltrans(0, s);
            for (int p = 1; p < HMM_NSTATES; ++p) {
                float v = prev[p] + ltrans(p, s);
                if (v > best) best = v;
            }
            float lse = 0.f;
            for (int p = 0; p < HMM_NSTATES; ++p)
                lse += expf(prev[p] + ltrans(p, s) - best);
            a[t*HMM_NSTATES + s] = best + logf(lse) + emit(ob, s);
        }
    }
}

// Kernel 7: HMM backward pass in log-space. One block per cell (1 thread).
__global__ void __launch_bounds__(1, 128)
hmm_backward_kernel(
    const float* __restrict__ seg_means,
    float* __restrict__       log_beta,    // n_cells × n_segs × 3
    int n_cells, int n_segs,
    float lt00, float lt01, float lt02,
    float lt10, float lt11, float lt12,
    float lt20, float lt21, float lt22,
    float lm0, float li0,
    float lm1, float li1,
    float lm2, float li2)
{
    int cell = blockIdx.x;
    if (cell >= n_cells) return;

    const float* obs = seg_means + (size_t)cell * n_segs;
    float*       b   = log_beta   + (size_t)cell * n_segs * HMM_NSTATES;

    auto emit = [&](float x, int s) -> float {
        float mu, i2;
        if (s == 0) { mu = lm0; i2 = li0; }
        else if (s == 1) { mu = lm1; i2 = li1; }
        else             { mu = lm2; i2 = li2; }
        float d = x - mu;
        return -0.5f * d * d * i2;
    };

    auto ltrans = [&](int f, int t) -> float {
        if (f == 0) { if (t == 0) return lt00; if (t == 1) return lt01; return lt02; }
        if (f == 1) { if (t == 0) return lt10; if (t == 1) return lt11; return lt12; }
        /* f==2 */  { if (t == 0) return lt20; if (t == 1) return lt21; return lt22; }
    };

    // Initialise: log beta[T-1] = 0 for all states.
    for (int s = 0; s < HMM_NSTATES; ++s)
        b[(n_segs-1)*HMM_NSTATES + s] = 0.f;

    // Backward recursion.
    for (int t = n_segs - 2; t >= 0; --t) {
        float ob_next = obs[t + 1];
        float next[HMM_NSTATES];
        for (int s = 0; s < HMM_NSTATES; ++s) next[s] = b[(t+1)*HMM_NSTATES + s];
        for (int s = 0; s < HMM_NSTATES; ++s) {
            float best = ltrans(s, 0) + emit(ob_next, 0) + next[0];
            for (int ns = 1; ns < HMM_NSTATES; ++ns) {
                float v = ltrans(s, ns) + emit(ob_next, ns) + next[ns];
                if (v > best) best = v;
            }
            float lse = 0.f;
            for (int ns = 0; ns < HMM_NSTATES; ++ns)
                lse += expf(ltrans(s, ns) + emit(ob_next, ns) + next[ns] - best);
            b[t*HMM_NSTATES + s] = best + logf(lse);
        }
    }
}

// Kernel 8: compute per-segment HMM posterior and argmax CNA state.
// posteriors sum to 1.0 along the state axis for each (cell, segment).
// WHY 1 thread per block: same reasoning as forward/backward.
__global__ void __launch_bounds__(1, 128)
hmm_posterior_kernel(
    const float* __restrict__ log_alpha,
    const float* __restrict__ log_beta,
    float* __restrict__       posteriors,   // n_cells × n_segs × 3
    uint8_t* __restrict__     chr_states,   // n_cells × n_segs
    int n_cells,
    int n_segs)
{
    int cell = blockIdx.x;
    if (cell >= n_cells) return;

    const float* a  = log_alpha  + (size_t)cell * n_segs * HMM_NSTATES;
    const float* b  = log_beta   + (size_t)cell * n_segs * HMM_NSTATES;
    float*       po = posteriors  + (size_t)cell * n_segs * HMM_NSTATES;
    uint8_t*     st = chr_states  + (size_t)cell * n_segs;

    for (int t = 0; t < n_segs; ++t) {
        float lp[HMM_NSTATES];
        float best = -1e30f;
        for (int s = 0; s < HMM_NSTATES; ++s) {
            lp[s] = a[t*HMM_NSTATES + s] + b[t*HMM_NSTATES + s];
            if (lp[s] > best) best = lp[s];
        }
        float sum = 0.f;
        for (int s = 0; s < HMM_NSTATES; ++s)
            sum += expf(lp[s] - best);
        float log_norm = best + logf(sum);

        int   argmax = HMM_NEUTRAL;
        float maxp   = -1.f;
        for (int s = 0; s < HMM_NSTATES; ++s) {
            float p = expf(lp[s] - log_norm);
            po[t*HMM_NSTATES + s] = p;
            if (p > maxp) { maxp = p; argmax = s; }
        }
        st[t] = (uint8_t)argmax;
    }
}

// Kernel 9: scatter per-chromosome state bytes into global output.
__global__ void __launch_bounds__(256, 4)
scatter_states_kernel(
    const uint8_t* __restrict__ chr_st,
    uint8_t* __restrict__       global_st,
    int n_cells, int n_segs_chr, int seg_offset, int n_total_segs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells * n_segs_chr) return;
    int cell = i / n_segs_chr;
    int seg  = i % n_segs_chr;
    global_st[(size_t)cell * n_total_segs + seg_offset + seg]
        = chr_st[(size_t)cell * n_segs_chr + seg];
}

// Kernel 10: scatter per-chromosome posteriors into global output.
__global__ void __launch_bounds__(256, 4)
scatter_posteriors_kernel(
    const float* __restrict__ chr_post,
    float* __restrict__       global_post,
    int n_cells, int n_segs_chr, int seg_offset, int n_total_segs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells * n_segs_chr * HMM_NSTATES) return;
    int cell  = i / (n_segs_chr * HMM_NSTATES);
    int rem   = i % (n_segs_chr * HMM_NSTATES);
    int seg   = rem / HMM_NSTATES;
    int state = rem % HMM_NSTATES;
    global_post[((size_t)cell * n_total_segs + seg_offset + seg) * HMM_NSTATES + state]
        = chr_post[((size_t)cell * n_segs_chr + seg) * HMM_NSTATES + state];
}

// Kernel 11: cast uint8 cna_states → float for kNN input.
__global__ void __launch_bounds__(256, 4)
u8_to_f32_kernel(
    const uint8_t* __restrict__ in,
    float* __restrict__         out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)in[i];
}

// ─── Host helpers ─────────────────────────────────────────────────────────────

// Build uniformly-binned segment boundaries for one chromosome.
// genes_in_chr: sorted gene indices (global) belonging to this chr.
// n_seg: requested segments (clamped to n_genes if fewer genes exist).
// seg_cursor: updated in-place with the global segment index counter.
inline std::vector<SegmentBoundary>
make_uniform_segments(int chr,
                      const std::vector<int>& genes_in_chr,
                      int n_seg,
                      int& seg_cursor)
{
    std::vector<SegmentBoundary> segs;
    int ng = static_cast<int>(genes_in_chr.size());
    if (ng == 0) return segs;
    n_seg = std::min(n_seg, ng);

    int base = ng / n_seg;
    int rem  = ng % n_seg;
    int off  = 0;
    for (int s = 0; s < n_seg; ++s) {
        int cnt = base + (s < rem ? 1 : 0);
        if (cnt == 0) continue;
        SegmentBoundary b;
        b.chr        = chr;
        b.gene_start = genes_in_chr[off];
        b.gene_end   = genes_in_chr[off + cnt - 1] + 1;
        b.seg_idx    = seg_cursor++;
        segs.push_back(b);
        off += cnt;
    }
    return segs;
}

// Compute log-transition matrix from trans_prob scalar.
// Layout: lt[from][to]. p(self) = 1 - 2*tp; p(other) = tp.
inline void build_log_trans(float tp, float (&lt)[HMM_NSTATES][HMM_NSTATES]) {
    for (int f = 0; f < HMM_NSTATES; ++f)
        for (int t = 0; t < HMM_NSTATES; ++t)
            lt[f][t] = logf(f == t ? (1.f - 2.f * tp) : tp);
}

}  // namespace detail

// ─── Public API ───────────────────────────────────────────────────────────────

// detect_cna — main entry point.
//
// expr:   DeviceCSC (genes × cells). Uses field-access style:
//           col_ptr.data(), row_indices.data(), values.data(), .rows, .cols.
//         This matches factornet::gpu::SparseMatrixGPU<float> (cycle 34 lesson).
// annot:  GeneAnnotation mapping gene indices → chromosome + position.
//         Must have n_genes() == expr.rows.
// ref:    Optional device pointer to per-gene reference normal expression
//         (n_genes floats). If nullptr, per-gene bulk mean is used as baseline.
// cfg:    CnaConfig. Defaults are sensible for 10x scRNA-seq tumor data.
// stream: Caller-provided CUDA stream. Never create streams internally.
//
// Returns CnaResult with device-resident arrays (RAII ownership).
inline CnaResult detect_cna(
    const core::DeviceCSC& expr,
    const GeneAnnotation&  annot,
    const float*           ref,
    const CnaConfig&       cfg,
    cudaStream_t           stream)
{
    const int full_win = 2 * cfg.smooth_half_win + 1;
    if (full_win > MAX_SMOOTH_WINDOW)
        throw std::invalid_argument(
            "smooth_half_win too large; full window exceeds MAX_SMOOTH_WINDOW");
    if (cfg.trans_prob <= 0.f || cfg.trans_prob >= 0.5f)
        throw std::invalid_argument("trans_prob must be in (0, 0.5)");

    const int n_genes = expr.rows;
    const int n_cells = expr.cols;

    if (n_genes == 0 || n_cells == 0)
        throw std::invalid_argument("detect_cna: empty expression matrix");
    if (annot.n_genes() != n_genes)
        throw std::invalid_argument("detect_cna: annot size != expr.rows");

    // ── 1. Group genes by chromosome, sort within each chr by position. ──────

    std::vector<std::vector<int>> chr_genes(26);  // index 1..25
    for (int g = 0; g < n_genes; ++g) {
        int c = annot.chr[g];
        if (c < 1 || c > 25) continue;
        if (!cfg.include_sex_chrs && (c == 23 || c == 24)) continue;
        chr_genes[c].push_back(g);
    }
    for (int c = 1; c <= 25; ++c) {
        auto& v = chr_genes[c];
        std::sort(v.begin(), v.end(),
                  [&](int a, int b){ return annot.start_bp[a] < annot.start_bp[b]; });
    }

    // ── 2. Build segment boundary table (host-side). ─────────────────────────

    std::vector<SegmentBoundary> all_segs;
    int seg_cursor = 0;
    for (int c = 1; c <= 25; ++c) {
        const auto& gv = chr_genes[c];
        if ((int)gv.size() < cfg.min_genes_per_chr) continue;
        auto segs = detail::make_uniform_segments(
            c, gv, cfg.segments_per_chr, seg_cursor);
        for (auto& s : segs) all_segs.push_back(s);
    }
    const int n_total_segs = static_cast<int>(all_segs.size());
    if (n_total_segs == 0)
        throw std::runtime_error(
            "detect_cna: no segments built (increase min_genes_per_chr?)");

    // ── 3. Allocate global output arrays. ────────────────────────────────────

    core::DeviceMemory<uint8_t> g_states(
        (size_t)n_cells * n_total_segs);
    core::DeviceMemory<float>   g_post(
        (size_t)n_cells * n_total_segs * HMM_NSTATES);

    // WHY cudaMemset for neutral init: avoids a launch overhead kernel;
    // neutral (1) is the dominant state; overwrites with actual results per chr.
    cudaMemsetAsync(g_states.get(), 1,
                    (size_t)n_cells * n_total_segs, stream);
    cudaMemsetAsync(g_post.get(), 0,
                    (size_t)n_cells * n_total_segs * HMM_NSTATES * sizeof(float), stream);

    // ── 4. Pre-compute HMM parameters (constant across chromosomes). ─────────

    float lt[HMM_NSTATES][HMM_NSTATES];
    detail::build_log_trans(cfg.trans_prob, lt);

    // inv_2sigma2 = 1 / (2σ²) for Gaussian log-emission.
    auto i2s2 = [](float sigma) { return 1.f / (2.f * sigma * sigma); };
    const float li0 = i2s2(cfg.loss_sigma);
    const float li1 = i2s2(cfg.neutral_sigma);
    const float li2 = i2s2(cfg.gain_sigma);

    // ── 5. Per-chromosome tile loop. ─────────────────────────────────────────

    for (int chr = 1; chr <= 25; ++chr) {
        const auto& gv = chr_genes[chr];
        if ((int)gv.size() < cfg.min_genes_per_chr) continue;

        const int n_gc = static_cast<int>(gv.size());  // genes in this chromosome

        // Count segments and find offset for this chromosome.
        int seg_offset = -1, n_sc = 0;
        for (const auto& s : all_segs) {
            if (s.chr != chr) continue;
            if (seg_offset < 0) seg_offset = s.seg_idx;
            ++n_sc;
        }
        if (n_sc == 0) continue;

        // -- 5a. Allocate per-chromosome workspace. ---------------------------
        const int gene_start = gv.front();
        const int gene_end   = gv.back() + 1;

        core::DeviceMemory<float>   tile     ((size_t)n_gc * n_cells);
        core::DeviceMemory<float>   log_ratio((size_t)n_gc * n_cells);
        core::DeviceMemory<float>   smoothed ((size_t)n_gc * n_cells);
        core::DeviceMemory<float>   seg_means((size_t)n_cells * n_sc);
        core::DeviceMemory<float>   la       ((size_t)n_cells * n_sc * HMM_NSTATES);
        core::DeviceMemory<float>   lb       ((size_t)n_cells * n_sc * HMM_NSTATES);
        core::DeviceMemory<float>   chr_post ((size_t)n_cells * n_sc * HMM_NSTATES);
        core::DeviceMemory<uint8_t> chr_st   ((size_t)n_cells * n_sc);

        // Zero the tile before expansion (CSC expansion only writes non-zeros).
        cudaMemsetAsync(tile.get(), 0,
                        (size_t)n_gc * n_cells * sizeof(float), stream);

        // -- 5b. Expand CSC → dense tile. -------------------------------------
        // Use factornet field-access style per cycle 34 lesson.
        {
            const int blk = 256, grd = (n_cells + blk - 1) / blk;
            detail::expand_csc_tile_kernel<<<grd, blk, 0, stream>>>(
                expr.col_ptr.get(),
                expr.row_indices.get(),
                expr.values.get(),
                tile.get(),
                gene_start, gene_end, n_gc, n_cells);
        }

        // -- 5c. Per-gene mean (used as reference when ref==nullptr). ----------
        core::DeviceMemory<float> gmean(n_gc);
        {
            const int shmem = 256 * sizeof(float);
            detail::gene_mean_kernel<<<n_gc, 256, shmem, stream>>>(
                tile.get(), gmean.get(), n_gc, n_cells);
        }
        const float* ref_ptr = (ref != nullptr) ? (ref + gene_start) : gmean.get();

        // -- 5d. Log ratio. ---------------------------------------------------
        {
            dim3 grid(n_gc, (n_cells + 255) / 256);
            detail::log_ratio_kernel<<<grid, 256, 0, stream>>>(
                tile.get(), ref_ptr, log_ratio.get(), n_gc, n_cells);
        }

        // -- 5e. Smoothing. ---------------------------------------------------
        {
            dim3 grid(n_gc, (n_cells + 127) / 128);
            detail::smooth_kernel<<<grid, 128, 0, stream>>>(
                log_ratio.get(), smoothed.get(), n_gc, n_cells, cfg.smooth_half_win);
        }

        // -- 5f. Build segment boundary arrays for this chromosome. -----------
        // Map global gene indices → local (tile-relative) for the segment kernel.
        std::vector<int> seg_gs_h(n_sc), seg_ge_h(n_sc);
        {
            int ls = 0;
            for (const auto& s : all_segs) {
                if (s.chr != chr) continue;
                seg_gs_h[ls] = s.gene_start - gene_start;
                seg_ge_h[ls] = s.gene_end   - gene_start;
                ++ls;
            }
        }

        // ✅ [A] cudaMemcpyAsync H→D: two int arrays of n_sc ints each.
        // One-time setup per chromosome (O(22) total). Not in a per-sample loop.
        core::DeviceMemory<int> d_gs(n_sc), d_ge(n_sc);
        cudaMemcpyAsync(d_gs.get(), seg_gs_h.data(),
                        (size_t)n_sc * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ge.get(), seg_ge_h.data(),
                        (size_t)n_sc * sizeof(int), cudaMemcpyHostToDevice, stream);

        // -- 5g. Segment means. -----------------------------------------------
        {
            dim3 grid(n_sc, (n_cells + 127) / 128);
            detail::segment_mean_kernel<<<grid, 128, 0, stream>>>(
                smoothed.get(), seg_means.get(),
                d_gs.get(), d_ge.get(),
                n_gc, n_cells, n_sc);
        }

        // -- 5h. HMM forward pass. --------------------------------------------
        {
            detail::hmm_forward_kernel<<<n_cells, 1, 0, stream>>>(
                seg_means.get(), la.get(), n_cells, n_sc,
                lt[0][0], lt[0][1], lt[0][2],
                lt[1][0], lt[1][1], lt[1][2],
                lt[2][0], lt[2][1], lt[2][2],
                cfg.loss_mean,    li0,
                cfg.neutral_mean, li1,
                cfg.gain_mean,    li2);
        }

        // -- 5i. HMM backward pass. -------------------------------------------
        {
            detail::hmm_backward_kernel<<<n_cells, 1, 0, stream>>>(
                seg_means.get(), lb.get(), n_cells, n_sc,
                lt[0][0], lt[0][1], lt[0][2],
                lt[1][0], lt[1][1], lt[1][2],
                lt[2][0], lt[2][1], lt[2][2],
                cfg.loss_mean,    li0,
                cfg.neutral_mean, li1,
                cfg.gain_mean,    li2);
        }

        // -- 5j. Posterior + argmax state. ------------------------------------
        {
            detail::hmm_posterior_kernel<<<n_cells, 1, 0, stream>>>(
                la.get(), lb.get(), chr_post.get(), chr_st.get(), n_cells, n_sc);
        }

        // -- 5k. Scatter results into global arrays. --------------------------
        {
            const int tot = n_cells * n_sc;
            const int blk = 256, grd = (tot + blk - 1) / blk;
            detail::scatter_states_kernel<<<grd, blk, 0, stream>>>(
                chr_st.get(), g_states.get(),
                n_cells, n_sc, seg_offset, n_total_segs);
        }
        {
            const int tot = n_cells * n_sc * HMM_NSTATES;
            const int blk = 256, grd = (tot + blk - 1) / blk;
            detail::scatter_posteriors_kernel<<<grd, blk, 0, stream>>>(
                chr_post.get(), g_post.get(),
                n_cells, n_sc, seg_offset, n_total_segs);
        }

        // All per-chromosome DeviceMemory released here (RAII).
    }  // end chromosome loop

    // ── 6. Clone clustering via leiden on the CNA pattern. ──────────────────

    // Cast uint8 states → float for kNN input.
    const size_t n_state_elem = (size_t)n_cells * n_total_segs;
    core::DeviceMemory<float> cna_f(n_state_elem);
    {
        const int blk = 256, grd = (int)((n_state_elem + blk - 1) / blk);
        detail::u8_to_f32_kernel<<<grd, blk, 0, stream>>>(
            g_states.get(), cna_f.get(), (int)n_state_elem);
    }

    // kNN via compute_exact (direct float pointer API; avoids knn.h wrapper
    // that calls .rows()/.cols() methods absent on DenseMatrixGPU fields).
    graph::KnnConfig knn_cfg;
    knn_cfg.k       = std::min(cfg.knn_k, n_cells - 1);
    knn_cfg.backend = graph::KnnBackend::Exact;
    knn_cfg.metric  = graph::DistanceMetric::L2;
    graph::KnnResult knn = graph::compute_exact(
        cna_f.get(), n_cells, n_total_segs, knn_cfg, stream);

    graph::LeidenConfig lc = cfg.leiden_cfg;
    lc.seed = cfg.seed;
    graph::LeidenResult leiden_res = graph::leiden(knn, lc, stream);

    // ── 7. Assemble result. ──────────────────────────────────────────────────

    CnaResult result;
    result.cna_states        = std::move(g_states);
    result.hmm_posteriors    = std::move(g_post);
    result.n_total_segments  = n_total_segs;
    result.clone_labels      = std::move(leiden_res.labels);
    result.n_clones          = leiden_res.n_clusters;
    result.leiden_modularity = leiden_res.modularity;
    result.segments          = std::move(all_segs);
    return result;
}

}  // namespace cna
}  // namespace singlet_gpu
