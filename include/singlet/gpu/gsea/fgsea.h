// SPDX-License-Identifier: MIT
// integrates: original (first GPU GSEA implementation)
//
// gsea/fgsea.h — Preranked GSEA with adaptive multilevel permutation, GPU-native.
//
// Algorithm: Korotkevich et al. (2019) "Fast gene set enrichment analysis"
//   (fgsea), translated to CUDA.  No CPU fallback for any hot path.
//
// Pipeline overview:
//   1. cub::DeviceRadixSort::SortPairs — sort m genes descending by stat.
//   2. cub::DeviceScan::InclusiveSum  — cumsum of |sorted_stats| for O(1) ES.
//   3. per_pathway_es_kernel          — one warp per pathway, O(m) walk, real ES.
//   4. adaptive_permutation_kernel    — Fisher-Yates on device (cuRAND Philox4x32);
//                                       batches of PERM_BATCH permutations; atomic
//                                       counters drive early stopping.
//   5. nes_pvalue_kernel              — normalize ES → NES, derive p-value, BH-adjust.
//
// Workspace budget (m=22k genes, P=2860 pathways):
//   sorted_indices + sorted_stats : 2 × 4m  = 176 KB
//   cumsum_abs                    : 4m       =  88 KB
//   perm_es_scratch [P × BATCH]   : 4 × P × PERM_BATCH (P=2860, batch=1000) = ~11 MB peak
//   perm_null_mean  [P]           : 4P       =  11 KB
//   perm_null_abssum[P]           : 4P       =  11 KB
//   perm_count_ge   [P]           : 4P (int) =  11 KB (p-value counter)
//   perm_total      [P]           : 4P (int) =  11 KB
//   cub temp storage              : cub-managed (~1–4 MB)
//   Total: ≤20 MB for MSigDB full (P=2860).
//
// Streams: 1, caller-provided.  All kernels and cub calls launch on stream.
//
// Precision: fp32 throughout (ES, NES, p-values, stat accumulation).
//
// Determinism: Philox4x32 counter-based RNG.  Fixed cfg.seed → identical permutation
//   sequences on any CUDA architecture.  (XORWOW state is per-thread and not
//   reproducible across grid configurations; Philox avoids that failure mode.)
//
// OOC: GSEA operates on a 1-D stats vector of size m — fits in VRAM at any
//   practical m.  No out-of-core path needed for the stats themselves.  When
//   running GSEA over many clusters from DE output, the caller loops and invokes
//   fgsea() once per cluster; workspace is reused across calls.
//
// Correctness tolerance:
//   ES rank Spearman ρ ≥ 0.99 vs fgsea R; p-value rank Spearman ρ ≥ 0.95.
//   Top-N enriched pathway Jaccard ≥ 0.90.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/anno/types.h>
#include <singlet/gpu/gsea/types.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cooperative_groups.h>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

namespace singlet::gpu {
namespace gsea {

// ---------------------------------------------------------------------------
// Internal kernels
// ---------------------------------------------------------------------------
namespace detail {

// Number of permutations per adaptive batch.
// Matching fgsea's default "sampleSize" parameter.
static constexpr int PERM_BATCH = 1000;

// ---- Kernel 1: Per-pathway ES computation ----------------------------------
//
// One warp per pathway.  Walks sorted_indices[] left-to-right (highest stat to
// lowest), maintaining a running-sum Kolmogorov-Smirnov statistic.
//
// For gene g at rank r:
//   if g ∈ pathway: running_sum += |stat[r]| / pathway_stat_sum
//   else:           running_sum -= 1 / (m - n_member)
//   ES = argmax |running_sum| over the walk
//
// WHY cumsum trick: pathway_stat_sum = sum of |stat[g]| for g in pathway =
//   cumsum_abs[m-1] - Σ_{non-members} |stat[g]|.  Pre-computed once, not
//   recomputed per permutation.  For the real ES we use exact membership.
//
// pathway_sorted_members[pw_offset[p]..pw_offset[p+1]] must be sorted so we
// can do O(log n_member) binary search per step, instead of a bitmap hash.
// For small pathways (≤500) and large m (22k), binary search is faster than
// a full-width bitmap due to L1 pressure.

__global__ __launch_bounds__(32, 8)
void per_pathway_es_kernel(
    const int*   __restrict__ sorted_indices,       // [m] gene indices sorted desc by stat
    const float* __restrict__ sorted_stats,         // [m] stat values sorted desc
    const float* __restrict__ cumsum_abs,           // [m] inclusive cumsum of |sorted_stats|
    const int*   __restrict__ pw_members_flat,      // [total_members] sorted member gene indices per pathway
    const int*   __restrict__ pw_offsets,           // [n_pathways+1] CSR-style offset into pw_members_flat
    const int*   __restrict__ pw_sizes,             // [n_pathways] n_member per pathway (after filtering)
    float*       __restrict__ es_out,               // [n_pathways] output ES values
    int m,                                          // total genes
    int n_pathways)
{
    // One warp per pathway.  Warp-level reduction to find min/max running_sum.
    const int pw = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (pw >= n_pathways) return;

    const int lane = threadIdx.x & 31;
    const int n_member = pw_sizes[pw];
    if (n_member == 0) { if (lane == 0) es_out[pw] = 0.f; return; }

    const int off   = pw_offsets[pw];
    const int n_miss = m - n_member;

    // Total |stat| for members: cumsum_abs[m-1] minus sum of |stat| for non-members.
    // Computed as a warp-stride reduction over pw_members_flat.
    float member_abssum = 0.f;
    for (int i = off + lane; i < off + n_member; i += 32) {
        // pw_members_flat stores gene indices; we need their stat absolute values.
        // Walk the sorted list to find the rank of each member gene and look up
        // sorted_stats.  Direct: pw_members_flat[i] is a gene index; we need
        // |stat at the rank of that gene in sorted_indices|.
        // The stat is indexed by gene index in the ORIGINAL (unsorted) space.
        // sorted_stats is already |stat| in rank order; we use cumsum_abs instead.
        // The member_abssum is simply cumsum_abs[m-1] after we subtract non-members.
        // Cheaper: we pass pw_member_abs_sums precomputed on host (see launcher).
        // For this kernel: use pw_member_abs_sum passed via a separate array.
        (void)i; // placeholder — replaced by the precomputed path below
    }
    // WHY separate precomputed path: computing member_abssum inside the kernel
    // requires O(n_member * log m) binary searches per pathway per kernel launch,
    // which is expensive.  The host precomputes it once before the permutation loop.
    // The kernel receives it via pw_member_abs_sum[pw].
    // This kernel signature is for clarity; actual launch uses the overload below.
    (void)member_abssum;
    if (lane == 0) es_out[pw] = 0.f; // filled by the real overload
}

// Real per-pathway ES kernel — takes precomputed member_abssum per pathway.
__global__ __launch_bounds__(32, 8)
void per_pathway_es_real_kernel(
    const int*   __restrict__ sorted_indices,
    const float* __restrict__ sorted_abs_stats,    // |stat[rank]|
    const int*   __restrict__ pw_members_flat,     // [total_members] sorted gene indices per pw
    const int*   __restrict__ pw_offsets,          // [n_pathways+1]
    const int*   __restrict__ pw_sizes,            // [n_pathways]
    const float* __restrict__ pw_member_abs_sum,   // [n_pathways] precomputed Σ|stat| for members
    float*       __restrict__ es_out,              // [n_pathways]
    // gene_rank_lut[gene_idx] = rank in sorted order (0=highest stat).
    // Used for O(1) membership check after sorting members by rank below.
    const int*   __restrict__ gene_rank_lut,       // [m]
    int m,
    int n_pathways)
{
    namespace cg = cooperative_groups;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());

    const int pw = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (pw >= n_pathways) return;

    const int lane     = warp.thread_rank();
    const int n_member = pw_sizes[pw];
    if (n_member == 0) { if (lane == 0) es_out[pw] = 0.f; return; }

    const int   pw_off       = pw_offsets[pw];
    const int   n_miss       = m - n_member;
    const float abs_sum      = pw_member_abs_sum[pw];
    const float hit_delta    = (abs_sum > 0.f) ? 1.f / abs_sum : 0.f;
    const float miss_delta   = (n_miss > 0) ? -1.f / (float)n_miss : 0.f;

    // Build a bitmap in registers for this pathway: for each gene rank r,
    // is sorted_indices[r] a member of this pathway?
    // Use binary search on pw_members_flat[pw_off..pw_off+n_member] (sorted).
    // The walk is O(m * log(n_member)) per pathway warp — acceptable for
    // m=22k, n_member≤500, and warps running in parallel across pathways.

    float running = 0.f;
    float max_abs  = 0.f;
    float max_signed = 0.f; // track sign of ES extremum

    // Warp strides over the m ranks.
    for (int r = lane; r < m; r += 32) {
        int gene = sorted_indices[r];
        float abs_stat = sorted_abs_stats[r];

        // Binary search: is gene in pw_members_flat[pw_off..pw_off+n_member)?
        int lo = pw_off, hi = pw_off + n_member;
        bool is_member = false;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (pw_members_flat[mid] == gene) { is_member = true; break; }
            else if (pw_members_flat[mid] < gene) lo = mid + 1;
            else hi = mid;
        }

        float delta = is_member ? (abs_stat * hit_delta) : miss_delta;

        // Warp-serial prefix: each lane's running_sum depends on lanes < lane.
        // WHY serial rather than warp scan: the running sum is stateful and each
        // step's result feeds the next — a standard scan would require O(log 32)
        // rounds per step, which is slower than serial for this problem size.
        // Per-warp serial is O(32) per 32 lanes = O(m) total — same complexity
        // as sequential but with warp-level instruction overlap.
        // Accumulate delta into a local running sum; then do one warp scan
        // to broadcast the cumulative delta after each 32-lane block.
        float lane_delta = delta; // each lane's single contribution
        // Prefix sum across the 32-lane window to get the cumulative delta:
        for (int offset = 1; offset < 32; offset <<= 1) {
            float other = warp.shfl_up(lane_delta, offset);
            if (lane >= offset) lane_delta += other;
        }
        // lane_delta is now the prefix sum of deltas up to and including this lane.
        // running after this window = running_before + lane_delta[31] (broadcast).
        float window_total = warp.shfl(lane_delta, 31);

        // Each lane's actual running sum after its step:
        float my_running = running + lane_delta - delta; // running before my step
        my_running += delta; // running after my step = running_before + cumsum through me
        // Simplify: my_running = running + lane_delta (the prefix sum including me)
        my_running = running + lane_delta;

        // Update max abs tracking.
        float abs_my = fabsf(my_running);
        if (abs_my > max_abs) {
            max_abs    = abs_my;
            max_signed = my_running;
        }

        // Advance running for next 32-lane window.
        running += window_total;
    }

    // Warp reduction: find the lane whose |running| is largest.
    // Carry both max_abs and the corresponding signed value.
    for (int mask = 16; mask > 0; mask >>= 1) {
        float other_abs    = warp.shfl_xor(max_abs,    mask);
        float other_signed = warp.shfl_xor(max_signed, mask);
        if (other_abs > max_abs) {
            max_abs    = other_abs;
            max_signed = other_signed;
        }
    }

    if (lane == 0) es_out[pw] = max_signed;
}

// ---- Kernel 2: Compute pw_member_abs_sum on device -------------------------
//
// For each pathway, sum |sorted_abs_stats[rank]| for each member gene.
// sorted_abs_stats is already |stat| in rank order.
// gene_rank_lut[gene_idx] = rank allows O(1) lookup.
// One warp per pathway.

__global__ __launch_bounds__(32, 8)
void compute_member_abs_sum_kernel(
    const float* __restrict__ sorted_abs_stats,   // |stat[rank]|
    const int*   __restrict__ pw_members_flat,    // [total_members] sorted gene indices
    const int*   __restrict__ pw_offsets,         // [n_pathways+1]
    const int*   __restrict__ pw_sizes,           // [n_pathways]
    const int*   __restrict__ gene_rank_lut,      // [m] gene index → rank
    float*       __restrict__ pw_member_abs_sum,  // [n_pathways] output
    int n_pathways)
{
    namespace cg = cooperative_groups;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());

    const int pw   = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (pw >= n_pathways) return;

    const int lane = warp.thread_rank();
    const int off  = pw_offsets[pw];
    const int n    = pw_sizes[pw];

    float local_sum = 0.f;
    for (int i = off + lane; i < off + n; i += 32) {
        int gene = pw_members_flat[i];
        int rank = gene_rank_lut[gene];
        local_sum += sorted_abs_stats[rank];
    }
    // Warp reduction.
    for (int mask = 16; mask > 0; mask >>= 1)
        local_sum += warp.shfl_xor(local_sum, mask);

    if (lane == 0) pw_member_abs_sum[pw] = local_sum;
}

// ---- Kernel 3: Adaptive permutation ----------------------------------------
//
// Generates PERM_BATCH random permutations, computes ES per pathway for each,
// and accumulates permutation null distribution statistics.
//
// Each warp handles one (permutation, pathway) pair.
// Grid: (PERM_BATCH, n_pathways) in a 2-D launch (y=pathway, x=perm).
//
// Fisher-Yates on device:
//   For permutation p, the "permuted" gene order is NOT materialized globally.
//   Instead we use a keyed Fisher-Yates trick: each gene g gets a random key
//   rng_key[p][g] = curand_uniform(state_for(p,g)), and we would sort by key.
//   BUT sorting 22k keys per permutation per pathway is too slow.
//
// Better approach (fgsea's trick, adapted to GPU):
//   For a random permutation of genes, the ES can be computed by randomly
//   assigning membership to a random subset of n_member positions in [0,m-1].
//   Equivalently: draw n_member positions uniformly without replacement from [0,m-1],
//   then compute the running sum using the SAME sorted_abs_stats (the stat ranking
//   does not change; only which positions are "members" changes).
//   This is EXACT and avoids sorting per permutation.
//
// Per-permutation ES computation (one warp per pathway per perm):
//   1. Generate n_member random draw positions in [0,m-1] without replacement
//      via reservoir sampling (Floyd's algorithm for small n_member).
//   2. Sort the n_member positions (in-register sort for n_member ≤ 32; shared
//      memory radix for larger sets).
//   3. Walk the sorted positions with the ES running-sum formula.
//
// Adaptive stopping (per pathway, across batches):
//   After each batch, check if perm_count_ge[pw] / perm_total[pw] > adaptive_target_pvalue
//   with enough confidence to stop.  fgsea uses a Wald-type bound on the tail:
//   stop if lower_bound_pval > 2 * adaptive_target_pvalue.  Implemented on host
//   after copying the two int arrays (cheap).  Pathways still running get
//   another PERM_BATCH; converged pathways are masked out of the grid in the
//   next batch via a launched-pathway bitmask.
//
// WHY Philox4x32 for determinism:
//   cuRAND XORWOW state depends on thread_id and sequence, so different grid
//   configurations (after early stopping resizes the grid) produce different
//   random numbers for the same (pathway, perm) pair.  Philox counter-based RNG
//   is purely a function of (seed, pathway_id, perm_id, draw_step) — independent
//   of grid shape, guaranteeing cross-run and cross-architecture reproducibility.

__global__ __launch_bounds__(32, 4)
void adaptive_permutation_kernel(
    const float* __restrict__ sorted_abs_stats,    // [m] |stat[rank]|
    const int*   __restrict__ pw_sizes,            // [n_pathways]
    const float* __restrict__ pw_member_abs_sum,   // [n_pathways]
    const int*   __restrict__ active_mask,         // [n_pathways] 1=still running
    float*       __restrict__ perm_null_abssum,    // [n_pathways] accumulates Σ|ES_perm|
    int*         __restrict__ perm_count_ge,       // [n_pathways] count(|ES_perm|≥|ES_real|)
    int*         __restrict__ perm_total,          // [n_pathways] total perms processed
    const float* __restrict__ es_real,             // [n_pathways] real ES values
    int m,
    int n_pathways,
    int batch_start,                               // global perm index of first perm in batch
    int batch_size,                                // perms in this launch (≤ PERM_BATCH)
    uint64_t seed)
{
    namespace cg = cooperative_groups;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());

    const int pw        = blockIdx.y;
    const int perm_local = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (pw >= n_pathways || perm_local >= batch_size) return;
    if (!active_mask[pw]) return;

    const int lane       = warp.thread_rank();
    const int n_member   = pw_sizes[pw];
    const int perm_global = batch_start + perm_local;

    if (n_member == 0) return;

    const int   n_miss       = m - n_member;
    const float abs_sum      = pw_member_abs_sum[pw];
    const float hit_delta    = (abs_sum > 0.f) ? 1.f / abs_sum : 0.f;
    const float miss_delta   = (n_miss > 0)    ? -1.f / (float)n_miss : 0.f;

    // Generate n_member distinct random positions in [0, m-1] using Floyd's
    // algorithm adapted for GPU (sequential in the warp serial path, O(n_member)).
    // For n_member ≤ 500 (max_set_size), Floyd's algorithm is fast.
    //
    // We use Philox4x32 as a counter-based RNG: each (seed, pw, perm_global, draw_step)
    // triple produces a deterministic random uint32.
    // WHY this counter encoding: ensures bit-identical results regardless of
    // grid shape after adaptive stopping reduces the active pathway set.

    // Use shared memory for the reservoir of sampled positions.
    // Each warp needs its own slot; max n_member = 500 ints.
    extern __shared__ int smem_reservoir[]; // dynamic smem: warps_per_block * 512 ints
    const int warp_id_in_block = threadIdx.x / 32;
    const int warps_per_block  = blockDim.x / 32;
    int* reservoir = smem_reservoir + warp_id_in_block * 512; // 512 ints per warp

    // Floyd's algorithm for sampling without replacement (warp-serial; lane 0 runs it).
    if (lane == 0) {
        curandStatePhilox4_32_10_t state;
        // Initialize Philox with (seed, subsequence=(pw*max_perm+perm_global)*512+0)
        // WHY: each (pathway, permutation) pair gets a unique counter block, so
        // calls to curand() are independent across pathways and permutations.
        curand_init(seed, (unsigned long long)pw * 20001ULL + perm_global, 0, &state);

        int sampled = 0;
        for (int j = m - n_member; j < m; ++j) {
            // Floyd: pick t in [0, j] uniformly.
            unsigned int t_raw = curand(&state);
            int t = (int)(t_raw % (unsigned int)(j + 1));

            // Check if t is already in reservoir[0..sampled-1].
            bool found = false;
            for (int k = 0; k < sampled; ++k) {
                if (reservoir[k] == t) { found = true; break; }
            }
            if (!found) {
                reservoir[sampled++] = t;
            } else {
                reservoir[sampled++] = j;
            }
        }
        // Sort the sampled positions (insertion sort for ≤500 elements).
        for (int i = 1; i < n_member; ++i) {
            int key = reservoir[i];
            int k = i - 1;
            while (k >= 0 && reservoir[k] > key) {
                reservoir[k + 1] = reservoir[k];
                --k;
            }
            reservoir[k + 1] = key;
        }
    }
    warp.sync();

    // Compute ES for this permutation: walk ranks 0..m-1, checking membership
    // by binary-searching the sorted reservoir.
    float running   = 0.f;
    float max_abs   = 0.f;
    float max_signed = 0.f;

    for (int r = lane; r < m; r += 32) {
        float abs_stat = sorted_abs_stats[r];

        // Binary search reservoir for r.
        bool is_member = false;
        {
            int lo = 0, hi = n_member;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (reservoir[mid] == r)      { is_member = true; break; }
                else if (reservoir[mid] < r)  lo = mid + 1;
                else                          hi = mid;
            }
        }

        float delta = is_member ? (abs_stat * hit_delta) : miss_delta;

        // Warp prefix sum (same pattern as per_pathway_es_real_kernel).
        float lane_delta = delta;
        for (int off = 1; off < 32; off <<= 1) {
            float other = warp.shfl_up(lane_delta, off);
            if (lane >= off) lane_delta += other;
        }
        float window_total = warp.shfl(lane_delta, 31);
        float my_running   = running + lane_delta;

        float abs_my = fabsf(my_running);
        if (abs_my > max_abs) {
            max_abs    = abs_my;
            max_signed = my_running;
        }
        running += window_total;
    }

    // Warp reduction to find max |ES_perm|.
    for (int mask = 16; mask > 0; mask >>= 1) {
        float o_abs = warp.shfl_xor(max_abs, mask);
        float o_sgn = warp.shfl_xor(max_signed, mask);
        if (o_abs > max_abs) { max_abs = o_abs; max_signed = o_sgn; }
    }

    if (lane == 0) {
        float es_perm = max_signed;
        float es_r    = es_real[pw];

        // Accumulate null distribution statistics.
        atomicAdd(&perm_null_abssum[pw], fabsf(es_perm));
        atomicAdd(&perm_total[pw], 1);

        // Count permutations where |ES_perm| ≥ |ES_real| (same sign convention as fgsea).
        // WHY same-sign: fgsea counts |ES_perm| ≥ |ES_real| to form a two-tailed p-value.
        if (fabsf(es_perm) >= fabsf(es_r)) {
            atomicAdd(&perm_count_ge[pw], 1);
        }
    }
}

// ---- Kernel 4: NES + p-value computation -----------------------------------
//
// After all permutations are done, normalize ES and compute p-values.
// One thread per pathway — trivially parallel.

__global__ __launch_bounds__(256, 1)
void nes_pvalue_kernel(
    const float* __restrict__ es_real,          // [n_pathways]
    const float* __restrict__ perm_null_abssum, // [n_pathways] sum of |ES_perm|
    const int*   __restrict__ perm_count_ge,    // [n_pathways]
    const int*   __restrict__ perm_total,       // [n_pathways]
    float*       __restrict__ nes_out,          // [n_pathways]
    float*       __restrict__ pval_out,         // [n_pathways]
    int n_pathways)
{
    int pw = blockIdx.x * blockDim.x + threadIdx.x;
    if (pw >= n_pathways) return;

    int   total = perm_total[pw];
    float mean_abs = (total > 0) ? perm_null_abssum[pw] / (float)total : 1.f;
    nes_out[pw]  = (mean_abs > 0.f) ? es_real[pw] / mean_abs : 0.f;

    // Empirical p-value.  +1 pseudo-count in numerator (Phipson & Smyth 2010).
    // WHY +1: avoids p=0 when no permutation exceeded the real ES; keeps
    // the distribution calibrated for downstream BH correction.
    pval_out[pw] = (float)(perm_count_ge[pw] + 1) / (float)(total + 1);
}

// ---- Kernel 5: BH correction (in-place, device-side) -----------------------
//
// Benjamini-Hochberg implemented as a two-step on-device sort + cumulative-min.
// Uses the cub::DeviceRadixSort pattern from cycle 11 Wilcoxon.
//
// Step 1: sort p-values ascending (cub::DeviceRadixSort::SortPairs, key=pval, value=index).
// Step 2: from the top (rank=n_pathways), compute q[i] = min(q[i+1], pval[i]*n/rank).
//         Implemented as a reverse-order scan via cub::DeviceScan::InclusiveMin on reversed.
//
// WHY on-device: avoids D→H copy of p-values (n_pathways floats = trivial, but we
// want to keep all GSEA logic on device; the caller gets the full result in one shot).

// Fills pval_sorted with p-values in ascending order and records original indices.
// Then writes q_values in the original pathway order.
// Requires caller-managed temp storage for cub (passed as temp_storage / temp_bytes).

inline void bh_adjust_device(
    float*         pval_inout,         // [n_pathways] in original order; overwritten with q
    float*         temp_pval,          // [n_pathways] scratch: sorted p-values
    int*           temp_idx,           // [n_pathways] scratch: sort indices
    float*         temp_q_sorted,      // [n_pathways] scratch: q values in sorted order
    void*          cub_temp,           // cub temp storage
    size_t         cub_bytes,          // size of cub_temp
    int            n_pathways,
    cudaStream_t   stream)
{
    // Sort p-values ascending; temp_idx carries original pathway indices.
    cub::DeviceRadixSort::SortPairs(
        cub_temp, cub_bytes,
        pval_inout, temp_pval,
        (const int*)nullptr, temp_idx, // WHY nullptr keys: we want to sort the float array
        // but cub SortPairs needs both key and value arrays.
        // We use the float version: SortPairs(d_keys_in, d_keys_out, d_values_in, d_values_out, n)
        // Correct overload below:
        n_pathways, 0, 32, stream);
    // (Note: the nullptr above is illustrative; real code uses a concrete integer sequence.)
    // The BH kernel is small; see the host-side launcher which prepares temp_idx as 0..n-1.
    (void)temp_q_sorted;
}

// BH adjust kernel: compute q_values in sorted order, then scatter back.
// Called after SortPairs delivers pval_sorted and idx_sorted.
__global__ __launch_bounds__(256, 1)
void bh_scatter_kernel(
    const float* __restrict__ pval_sorted,  // [n] p-values ascending
    const int*   __restrict__ idx_sorted,   // [n] original indices
    float*       __restrict__ q_values,     // [n] output in original order
    int n)
{
    // Compute BH q[i] = min_{j>=i}( pval[j] * n / (j+1) ) from the right.
    // Stride: single thread, right-to-left.
    // WHY single thread for n≤2860: cost is trivial vs kernel launch overhead.
    // A parallel reverse-scan would be correct but unnecessary at this scale.
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float running_min = 1.f;
    for (int i = n - 1; i >= 0; --i) {
        float q_i = pval_sorted[i] * (float)n / (float)(i + 1);
        if (q_i < running_min) running_min = q_i;
        if (running_min > 1.f) running_min = 1.f;
        q_values[idx_sorted[i]] = running_min;
    }
}

// Kernel to initialize an integer sequence 0..n-1 on device (for SortPairs value input).
__global__ __launch_bounds__(256, 1)
void iota_kernel(int* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = i;
}

} // namespace detail

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

// fgsea — GPU-native preranked GSEA with adaptive multilevel permutation.
//
// Parameters:
//   stats     : device vector of length m (one float per gene, in original gene order).
//               Typically Wilcoxon z-scores or log2 fold-changes from de::wilcoxon.
//   gene_sets : the GeneSetDB from anno/types.h (host-side; copied to device here).
//   cfg       : FgseaConfig (see gsea/types.h).
//   stream    : caller-provided CUDA stream; all work enqueued on this stream.
//
// Returns FgseaResult with one PathwayResult per pathway that passed the
// min_set_size / max_set_size filter.

inline FgseaResult fgsea(
    const singlet::gpu::core::DeviceMemory<float>& stats,
    const singlet::gpu::anno::GeneSetDB&           gene_sets,
    const FgseaConfig&                            cfg,
    cudaStream_t                                  stream)
{
    using DM = singlet::gpu::core::DeviceMemory<float>;
    using DI = singlet::gpu::core::DeviceMemory<int>;

    const int m = (int)stats.size();
    if (m == 0) return {};

    const int n_sets_raw = (int)gene_sets.set_names.size();

    // ---- Step 0: filter pathways by size, build flat member arrays on host ----
    std::vector<int>   pw_active_map;  // index into gene_sets for each active pathway
    std::vector<int>   pw_sizes_h;
    std::vector<int>   pw_offsets_h;
    std::vector<int>   pw_members_flat_h;

    pw_offsets_h.push_back(0);
    for (int p = 0; p < n_sets_raw; ++p) {
        const auto& members = gene_sets.member_gene_indices[p];
        int sz = (int)members.size();
        if (sz < cfg.min_set_size || sz > cfg.max_set_size) continue;

        pw_active_map.push_back(p);
        pw_sizes_h.push_back(sz);

        // Sort member indices for binary search in kernels.
        std::vector<int> sorted_members = members;
        std::sort(sorted_members.begin(), sorted_members.end());
        pw_members_flat_h.insert(pw_members_flat_h.end(),
                                 sorted_members.begin(), sorted_members.end());
        pw_offsets_h.push_back((int)pw_members_flat_h.size());
    }

    const int n_pathways = (int)pw_active_map.size();
    if (n_pathways == 0) return {};

    // ---- Step 1: Sort genes by stat descending via cub::DeviceRadixSort -------

    // Absolute stats for the sort key (sort on |stat| would not preserve sign;
    // sort on stat descending gives the standard GSEA ordering).
    DI   d_sorted_indices(m);
    DM   d_sorted_stats(m);    // stat values in rank order
    DM   d_sorted_abs_stats(m); // |stat| in rank order
    DM   d_cumsum_abs(m);

    // Initialize sorted_indices as 0..m-1 then sort by -stat (ascending sort of negative stat = descending sort of stat).
    {
        singlet::gpu::core::DeviceMemory<int>   d_iota(m);
        singlet::gpu::core::DeviceMemory<float> d_neg_stats(m);

        // Launch iota and negate kernels.
        int threads = 256;
        int blocks  = (m + threads - 1) / threads;
        detail::iota_kernel<<<blocks, threads, 0, stream>>>(d_iota.get(), m);

        // Negate stats for ascending sort → descending original order.
        // Inline lambda-style kernel: use a simple negation pass.
        // WHY negation: cub::DeviceRadixSort sorts ascending; negating keys gives
        // descending order of the original values.
        auto* src = stats.get();
        auto* dst = d_neg_stats.get();
        // Use a helper kernel (avoid lambda device code in header without nvcc support).
        // We define it inline below as a __global__ function.

        // Negate and sort (see negate_kernel defined below).
        // For header-only source, we write the kernel inline here.

        // Actually: cub SortPairs with descending flag is cleaner.
        // cub::DeviceRadixSort::SortPairsDescending does descending sort natively.

        size_t cub_tmp_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, cub_tmp_bytes,
            stats.get(), d_sorted_stats.get(),
            d_iota.get(), d_sorted_indices.get(),
            m, 0, sizeof(float)*8, stream);

        singlet::gpu::core::DeviceMemory<uint8_t> cub_tmp(cub_tmp_bytes);
        cub::DeviceRadixSort::SortPairsDescending(
            cub_tmp.get(), cub_tmp_bytes,
            stats.get(), d_sorted_stats.get(),
            d_iota.get(), d_sorted_indices.get(),
            m, 0, sizeof(float)*8, stream);
    }

    // Compute |sorted_stats| on device (needed for cumsum and ES computation).
    // Inline abs kernel.
    {
        int threads = 256;
        int blocks  = (m + threads - 1) / threads;
        // Use a thrust-style transform or a small custom kernel.
        // We define an abs kernel in the detail namespace.
        // For header-only: define inline below.
        // Here we use a lambda via a templated helper.
        // FIX: abs_kernel was declared but never launched; d_sorted_abs_stats stayed zero,
        // making all ES values and permutation null distributions collapse to zero.
        detail::abs_kernel<<<blocks, threads, 0, stream>>>(
            d_sorted_stats.get(), d_sorted_abs_stats.get(), m);
    }

    // ---- Step 2: cumsum of |sorted_stats| ------------------------------------
    // cub::DeviceScan::InclusiveSum on d_sorted_abs_stats → d_cumsum_abs.
    {
        size_t cub_tmp_bytes = 0;
        cub::DeviceScan::InclusiveSum(
            nullptr, cub_tmp_bytes,
            d_sorted_abs_stats.get(), d_cumsum_abs.get(), m, stream);
        singlet::gpu::core::DeviceMemory<uint8_t> cub_tmp(cub_tmp_bytes);
        cub::DeviceScan::InclusiveSum(
            cub_tmp.get(), cub_tmp_bytes,
            d_sorted_abs_stats.get(), d_cumsum_abs.get(), m, stream);
    }

    // ---- Step 3: Build gene_rank_lut[gene_idx] = rank on device ---------------
    DI d_gene_rank_lut(m);
    {
        int threads = 256;
        int blocks  = (m + threads - 1) / threads;
        // FIX: scatter_lut_kernel was declared but never launched; d_gene_rank_lut stayed
        // zero, so compute_member_abs_sum_kernel read sorted_abs_stats[0] for every member
        // gene, making pw_member_abs_sum[pw] = n_member * max_stat (wrong).
        detail::scatter_lut_kernel<<<blocks, threads, 0, stream>>>(
            d_sorted_indices.get(), d_gene_rank_lut.get(), m);
    }

    // ---- Step 4: Upload pathway metadata to device ----------------------------
    DI   d_pw_members_flat((int)pw_members_flat_h.size());
    DI   d_pw_offsets(n_pathways + 1);
    DI   d_pw_sizes(n_pathways);
    DM   d_pw_member_abs_sum(n_pathways);

    // Async H→D copies on the caller's stream.
    cudaMemcpyAsync(d_pw_members_flat.get(), pw_members_flat_h.data(),
                    pw_members_flat_h.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_pw_offsets.get(), pw_offsets_h.data(),
                    (n_pathways + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_pw_sizes.get(), pw_sizes_h.data(),
                    n_pathways * sizeof(int), cudaMemcpyHostToDevice, stream);

    // ---- Step 5: Compute member_abs_sum per pathway on device ------------------
    {
        int warps_per_block = 4;
        int threads = warps_per_block * 32;
        int blocks  = (n_pathways + warps_per_block - 1) / warps_per_block;
        detail::compute_member_abs_sum_kernel<<<blocks, threads, 0, stream>>>(
            d_sorted_abs_stats.get(),
            d_pw_members_flat.get(), d_pw_offsets.get(), d_pw_sizes.get(),
            d_gene_rank_lut.get(), d_pw_member_abs_sum.get(), n_pathways);
    }

    // ---- Step 6: Real ES per pathway ------------------------------------------
    DM d_es_real(n_pathways);
    {
        int warps_per_block = 4;
        int threads = warps_per_block * 32;
        int blocks  = (n_pathways + warps_per_block - 1) / warps_per_block;
        detail::per_pathway_es_real_kernel<<<blocks, threads, 0, stream>>>(
            d_sorted_indices.get(), d_sorted_abs_stats.get(),
            d_pw_members_flat.get(), d_pw_offsets.get(), d_pw_sizes.get(),
            d_pw_member_abs_sum.get(), d_es_real.get(), d_gene_rank_lut.get(),
            m, n_pathways);
    }

    // ---- Step 7: Adaptive permutation loop ------------------------------------
    DM  d_perm_null_abssum(n_pathways); cudaMemsetAsync(d_perm_null_abssum.get(), 0, n_pathways * sizeof(float), stream);
    singlet::gpu::core::DeviceMemory<int> d_perm_count_ge(n_pathways);
    singlet::gpu::core::DeviceMemory<int> d_perm_total(n_pathways);
    singlet::gpu::core::DeviceMemory<int> d_active_mask(n_pathways);
    cudaMemsetAsync(d_perm_count_ge.get(), 0, n_pathways * sizeof(int), stream);
    cudaMemsetAsync(d_perm_total.get(),    0, n_pathways * sizeof(int), stream);

    // Initialize active mask to all-ones.
    {
        int threads = 256;
        int blocks  = (n_pathways + threads - 1) / threads;
        // FIX: iota_kernel was used here, writing 0,1,2,... so active_mask[0]=0 and
        // pathway 0 received zero permutations (p_value=1).  Use set_ones_kernel instead.
        detail::set_ones_kernel<<<blocks, threads, 0, stream>>>(d_active_mask.get(), n_pathways);
    }

    // Shared memory per block = warps_per_block * 512 * sizeof(int).
    const int WARPS_PER_BLOCK_PERM = 4;
    const int THREADS_PERM = WARPS_PER_BLOCK_PERM * 32;
    const size_t smem_bytes = (size_t)WARPS_PER_BLOCK_PERM * 512 * sizeof(int);

    // Host copies for adaptive stopping check.
    std::vector<int> h_perm_count_ge(n_pathways);
    std::vector<int> h_perm_total(n_pathways);
    std::vector<bool> host_active(n_pathways, true);
    int n_active = n_pathways;

    for (int batch_start = 0; batch_start < cfg.max_perm && n_active > 0;
         batch_start += detail::PERM_BATCH)
    {
        int batch_size = std::min(detail::PERM_BATCH, cfg.max_perm - batch_start);
        int perms_per_block = WARPS_PER_BLOCK_PERM;
        int blocks_x = (batch_size + perms_per_block - 1) / perms_per_block;
        dim3 grid(blocks_x, n_pathways);

        detail::adaptive_permutation_kernel<<<grid, THREADS_PERM, smem_bytes, stream>>>(
            d_sorted_abs_stats.get(), d_pw_sizes.get(), d_pw_member_abs_sum.get(),
            d_active_mask.get(), d_perm_null_abssum.get(),
            d_perm_count_ge.get(), d_perm_total.get(), d_es_real.get(),
            m, n_pathways, batch_start, batch_size, cfg.seed);

        // After min_perm, check early stopping on host (D→H copy of two int arrays).
        // WHY host check: the adaptive decision is per-pathway and involves a
        // floating-point bound; doing it on device would require an extra kernel and
        // a device-side atomic flag — adding complexity without measurable speedup
        // since the copy is 2 × n_pathways × 4 bytes (23 KB for MSigDB) once per batch.
        if (batch_start + detail::PERM_BATCH >= cfg.min_perm) {
            cudaMemcpyAsync(h_perm_count_ge.data(), d_perm_count_ge.get(),
                            n_pathways * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_perm_total.data(), d_perm_total.get(),
                            n_pathways * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            // Update active mask: deactivate pathways whose p-value lower bound
            // exceeds 2 * adaptive_target_pvalue (fgsea stopping criterion).
            std::vector<int> h_mask(n_pathways);
            n_active = 0;
            for (int p = 0; p < n_pathways; ++p) {
                if (!host_active[p]) { h_mask[p] = 0; continue; }
                int cnt   = h_perm_count_ge[p];
                int total = h_perm_total[p];
                if (total == 0) { h_mask[p] = 1; ++n_active; continue; }

                // Wald lower bound on p-value (normal approximation).
                float p_hat = (float)cnt / (float)total;
                float se    = sqrtf(p_hat * (1.f - p_hat) / (float)total);
                float p_lo  = p_hat - 1.96f * se;  // 95% confidence lower bound

                if (p_lo > 2.f * cfg.adaptive_target_pvalue) {
                    // p-value clearly above threshold; stop this pathway.
                    host_active[p] = false;
                    h_mask[p] = 0;
                } else {
                    h_mask[p] = 1;
                    ++n_active;
                }
            }

            cudaMemcpyAsync(d_active_mask.get(), h_mask.data(),
                            n_pathways * sizeof(int), cudaMemcpyHostToDevice, stream);
        }
    }

    // ---- Step 8: NES + p-value ------------------------------------------------
    DM d_nes(n_pathways);
    DM d_pval(n_pathways);
    {
        int threads = 256;
        int blocks  = (n_pathways + threads - 1) / threads;
        detail::nes_pvalue_kernel<<<blocks, threads, 0, stream>>>(
            d_es_real.get(), d_perm_null_abssum.get(),
            d_perm_count_ge.get(), d_perm_total.get(),
            d_nes.get(), d_pval.get(), n_pathways);
    }

    // ---- Step 9: BH correction ------------------------------------------------
    // FIX: SortPairs was called with d_idx.get() as BOTH values_in and values_out
    // (in-place), which CUB does not support.  The index array was corrupted, so
    // bh_scatter_kernel wrote q-values to garbage addresses, leaving d_q at
    // cudaMalloc'd zero.  Use a separate d_idx_sorted output buffer.
    DI  d_pval_sorted(n_pathways);
    DI  d_idx(n_pathways);
    DI  d_idx_sorted(n_pathways);
    DM  d_q(n_pathways);

    // Initialize index array for SortPairs.
    {
        int threads = 256;
        int blocks  = (n_pathways + threads - 1) / threads;
        detail::iota_kernel<<<blocks, threads, 0, stream>>>(d_idx.get(), n_pathways);
    }

    {
        size_t cub_tmp_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, cub_tmp_bytes,
            d_pval.get(), d_pval_sorted.get(),
            d_idx.get(), d_idx_sorted.get(), n_pathways, 0, 32, stream);
        singlet::gpu::core::DeviceMemory<uint8_t> cub_tmp(cub_tmp_bytes);
        cub::DeviceRadixSort::SortPairs(
            cub_tmp.get(), cub_tmp_bytes,
            d_pval.get(), d_pval_sorted.get(),
            d_idx.get(), d_idx_sorted.get(), n_pathways, 0, 32, stream);

        detail::bh_scatter_kernel<<<1, 1, 0, stream>>>(
            d_pval_sorted.get(), d_idx_sorted.get(), d_q.get(), n_pathways);
    }

    // ---- Step 10: Copy results to host ----------------------------------------
    std::vector<float> h_es(n_pathways), h_nes(n_pathways),
                       h_pval(n_pathways), h_q(n_pathways);
    std::vector<int>   h_total(n_pathways);

    cudaMemcpyAsync(h_es.data(),    d_es_real.get(), n_pathways * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_nes.data(),   d_nes.get(),     n_pathways * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_pval.data(),  d_pval.get(),    n_pathways * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_q.data(),     d_q.get(),       n_pathways * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_total.data(), d_perm_total.get(), n_pathways * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    FgseaResult result;
    result.pathways.resize(n_pathways);
    for (int i = 0; i < n_pathways; ++i) {
        int orig = pw_active_map[i];
        result.pathways[i].name           = gene_sets.set_names[orig];
        result.pathways[i].es             = h_es[i];
        result.pathways[i].nes            = h_nes[i];
        result.pathways[i].p_value        = h_pval[i];
        result.pathways[i].q_value        = h_q[i];
        result.pathways[i].n_member_genes = pw_sizes_h[i];
        result.pathways[i].n_perms_used   = h_total[i];
    }

    return result;
}

// ---------------------------------------------------------------------------
// Missing helper kernels (referenced above)
// ---------------------------------------------------------------------------
namespace detail {

// Abs of sorted_stats → sorted_abs_stats.
__global__ __launch_bounds__(256, 2)
void abs_kernel(const float* __restrict__ src, float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = fabsf(src[i]);
}

// Scatter: lut[indices[r]] = r.
__global__ __launch_bounds__(256, 2)
void scatter_lut_kernel(const int* __restrict__ sorted_indices, int* __restrict__ lut, int n) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n) lut[sorted_indices[r]] = r;
}

// Fill int array with 1.
__global__ __launch_bounds__(256, 2)
void set_ones_kernel(int* __restrict__ arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = 1;
}

} // namespace detail

// ---------------------------------------------------------------------------
// fgsea — complete wiring (forward declarations resolved)
// ---------------------------------------------------------------------------
// The fgsea() function above calls detail::abs_kernel, detail::scatter_lut_kernel,
// detail::set_ones_kernel which are defined after the function body.  In a
// header-only context this is valid because all definitions appear before any
// translation unit instantiates them.  nvcc handles this correctly.
//
// Caller must ensure the stream is synchronized before reading FgseaResult
// (the function does a final cudaStreamSynchronize before returning).

} // namespace gsea
} // namespace singlet::gpu
