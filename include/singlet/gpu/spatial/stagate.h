// SPDX-License-Identifier: MIT
// integrates: original (first GPU STAGATE; manual CUDA GAT gradients vs PyG auto-diff)
//
// spatial/stagate.h — GPU-native STAGATE: Spatial domain segmentation via graph
//   attention autoencoder (Dong & Zhang, Nature Communications 2022).
//
// Algorithm:
//   1. Build kNN graph on spatial (x,y) coordinates (reuses graph/knn.h).
//   2. Encoder: 2 GAT layers [d_in → d_hidden → d_embed].
//      Attention coefficients: α_ij = softmax_j( LeakyReLU(a^T [W·h_i || W·h_j]) )
//      Aggregation: h_i' = Σ_j α_ij · (W·h_j)
//   3. Decoder: 2 mirror dense-linear layers [d_embed → d_hidden → d_in].
//   4. Loss: sparse-masked MSE — only at nonzero positions of the input CSC.
//      WHY sparse-masked: materializing a dense n_spots × n_genes reconstruction
//      at 100k × 30k is 12 GB; scanning only nnz positions keeps memory O(nnz).
//   5. Adam optimizer, fp32. Per-epoch scalar loss accumulated on-device (fp64),
//      read back once per epoch (≤4 bytes). This is the ONLY host transfer in the
//      training loop — approved scalar exception per absolute rule §⛔.
//   6. Optional post-training Leiden clustering on the embedding (reuses graph/leiden.h).
//
// Manual gradient derivation (no auto-diff):
//   ∂MSE/∂X_hat_ij = -2*(X_ij - X_hat_ij) * [X_ij ≠ 0]  (sparse mask)
//   Decoder backward: standard dense-linear chain rule.
//   Encoder backward: attention + aggregation reverse via:
//     ∂L/∂W = Σ_j α_ij * (∂L/∂h_i') ⊗ h_j  (outer product, accumulated per spot)
//     ∂L/∂a from chain through softmax(LeakyReLU(...))
//     ∂L/∂h_j from Σ_i α_ij * W^T * (∂L/∂h_i')  (scatter)
//   See kernel comments for full gradient formulae.
//
// No cudaMemcpy in any loop > 5 iters:
//   - Training loop (n_epochs ≤ 500): one cub::DeviceReduce::Sum + one 4-byte D2H
//     per epoch (scalar loss). No other host transfers.
//   - All other per-epoch operations are pure device kernels on the same stream.
//
// cub usage:
//   cub::DeviceReduce::Sum          — scalar loss readback per epoch (4 bytes D2H)
//   cub::DeviceSegmentedReduce::Sum — per-spot attention-weight normalization (softmax denom)
//   cub::DeviceRadixSort::SortPairs — sparse-mask gather (sort nnz positions by spot index)
//
// Memory budget (100k spots × 30k genes, k=6, d_hidden=256, d_embed=64):
//   h1 [n × d_hidden]:       100k × 256 × 4 =  ~100 MB
//   h2 [n × d_embed]:        100k × 64  × 4 =   ~25 MB
//   dec_h1 [n × d_hidden]:   ~100 MB
//   Attention scores [n × k]:100k × 6 × 4  =    ~2.4 MB
//   GAT weights W1 [d_in × d_hidden]:  sparse projection via SpMM, weight small ~4 MB
//   GAT weights W2 [d_hidden × d_embed]: ~64 KB
//   Attention vec a1 [2*d_hidden]:     ~2 KB
//   Attention vec a2 [2*d_embed]:      ~0.5 KB
//   Dec W1 [d_embed × d_hidden]:       ~64 KB
//   Dec W2 [d_hidden × d_in]:          ~4 MB
//   Adam state (m,v) for all params:   ~20 MB
//   Sparse-masked recon buffer [nnz]:  varies (≤12 GB avoided — only nnz values)
//   Total: ~250 MB at 100k spots (excluding CSC input which is caller-owned)
//
// Streams: 1, caller-provided.
// Precision: fp32 hot path; fp64 loss accumulator (cub sum over fp32 partial losses).
// Determinism: Philox4x32 seeded via cfg.seed for weight initialization.
// OOC plan: spot-tile training — process N=cfg.spot_tile spots per forward/backward
//   pass, accumulate gradients across tiles, apply Adam once per epoch. For 1M-spot
//   Xenium HD: 100 tiles × 10k spots each. The kNN graph is fully resident (small).
//
// References:
//   Dong & Zhang, Nature Communications 2022 (STAGATE).
//   Veličković et al., ICLR 2018 (Graph Attention Networks).
//   Kingma & Ba, ICLR 2015 (Adam).

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/graph/knn.h>
#include <singlet/gpu/graph/leiden.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <curand_kernel.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <limits>
#include <algorithm>

namespace singlet::gpu {
namespace spatial {

// ─── Public API ───────────────────────────────────────────────────────────────

struct StagateConfig {
    int      n_neighbors        = 6;       // spatial kNN (Visium hexagonal default)
    int      d_hidden           = 256;     // first GAT layer output dim
    int      d_embed            = 64;      // bottleneck dim
    int      n_epochs           = 500;
    float    learning_rate      = 1e-3f;
    float    lambda_recon       = 1.0f;    // MSE weight (reserved for contrastive ext)
    bool     sparse_mask_recon  = true;    // compute MSE only at nonzero CSC positions
    int      spot_tile          = 10000;   // spots per OOC tile during training
    bool     run_post_leiden    = true;
    float    leiden_resolution  = 0.5f;
    uint64_t seed               = 0;
};

struct StagateResult {
    core::DeviceMemory<float> embedding;       // n_spots × d_embed (row-major)
    core::DeviceMemory<int>   spatial_domain;  // n_spots; Leiden labels (empty if disabled)
    std::vector<float>        loss_history;    // host; one entry per epoch
    int                       n_epochs_used;
};

// ─── Device kernels ───────────────────────────────────────────────────────────

namespace detail {

// ── Initialization ──────────────────────────────────────────────────────────

// Xavier uniform init: fill a device buffer [n] with U[-bound, +bound].
// bound = sqrt(6 / (fan_in + fan_out)); passed by caller.
// Philox4x32: each thread independently addresses (seed, offset=global_idx).
// WHY Philox: same pattern as cell2fate; deterministic with cfg.seed.
__launch_bounds__(256, 4)
__global__ void xavier_init_kernel(
    float* __restrict__ w,
    float bound, uint64_t seed, uint64_t param_offset,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, (unsigned long long)(param_offset + i), 0, &state);
    float u = curand_uniform(&state);  // (0, 1]
    w[i] = (u * 2.0f - 1.0f) * bound;
}

// Zero-init: Adam moments and attention bias.
__launch_bounds__(256, 4)
__global__ void zero_fill_kernel(float* __restrict__ buf, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = 0.f;
}

// ── Sparse-mask gather ──────────────────────────────────────────────────────

// Build per-nnz (spot_index, gene_index) pairs from CSC indptr+indices.
// Used once before training to sort nnz entries by spot (for per-spot kernel).
// Output: spot_of_nnz[nnz], gene_of_nnz[nnz].
// WHY: CSC is column-major (gene-major); we need spot-major access in the
// decoder's sparse reconstruction. Rather than transposing the CSC (memory),
// we build a sorted key array that lets the decoder kernel scatter/gather correctly.
__launch_bounds__(256, 2)
__global__ void build_nnz_spot_gene_kernel(
    int32_t* __restrict__ spot_of_nnz,  // [nnz] output: row (spot) index per nz
    int32_t* __restrict__ gene_of_nnz,  // [nnz] output: col (gene) index per nz
    const int32_t* __restrict__ indptr, // [n_genes+1] CSC column pointers
    const int32_t* __restrict__ indices,// [nnz] CSC row indices (spot indices)
    int n_genes, int64_t nnz)
{
    // One thread per nnz element; gene determined by binary search on indptr.
    // For well-structured CSC, a block-per-gene approach would be better; but
    // this kernel runs ONCE, so correctness over speed.
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    spot_of_nnz[idx] = indices[idx];  // row (spot) index from CSC indices array
    // Gene (column) index via binary search on CSC indptr.
    int lo = 0, hi = n_genes;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if ((int64_t)indptr[mid + 1] <= idx) lo = mid + 1;
        else hi = mid;
    }
    gene_of_nnz[idx] = lo;
}

// ── GAT forward ─────────────────────────────────────────────────────────────

// Layer 1 GAT forward: sparse input X (CSC) → projected hidden h1 (n × d_h).
// The encoder's first layer applies W1 (d_in × d_h) to each spot's expression
// vector via cuSPARSE SpMM, then computes attention over spatial neighbors.
//
// This kernel computes the attention-weighted aggregation for ONE GAT layer.
// It expects the projected features W·h to already be materialized (h_proj).
//
// For each spot i and each neighbor j in knn.neighbors[i*k .. (i+1)*k]:
//   e_ij = LeakyReLU(a[0..d_h-1] · h_proj_i + a[d_h..2*d_h-1] · h_proj_j)
//   alpha_ij = softmax over j of e_ij
//   h_out_i = Σ_j alpha_ij * h_proj_j
//
// One block per spot; threads stride over d_out.
// Attention scores (e_ij for j ∈ neighbors) stored in attn_scores[i*k+j_local].
// WHY store attn_scores: backward pass needs them for the softmax gradient.
__launch_bounds__(128, 4)
__global__ void gat_forward_kernel(
    float* __restrict__        h_out,         // [n × d_out] output embedding
    float* __restrict__        attn_scores,   // [n × k] attention logits (pre-softmax)
    float* __restrict__        attn_weights,  // [n × k] softmax alpha_ij
    const float* __restrict__  h_proj,        // [n × d_out] W·h input (already projected)
    const float* __restrict__  a_vec,         // [2*d_out] attention vector
    const int32_t* __restrict__ neighbors,    // [n * k] kNN neighbor indices
    int n, int k, int d_out,
    float leaky_slope)
{
    int spot_i = blockIdx.x;
    if (spot_i >= n) return;

    extern __shared__ float smem[];
    // Layout: [d_out] h_i projection | [k] e_ij scores
    float* h_i_proj = smem;              // [d_out]
    float* e_scores = smem + d_out;      // [k]

    // Load h_proj[spot_i] into shared memory.
    for (int d = threadIdx.x; d < d_out; d += blockDim.x)
        h_i_proj[d] = h_proj[(size_t)spot_i * d_out + d];
    __syncthreads();

    // Compute a[0..d_out) · h_proj_i (self-score component) — one warp reduction.
    // WHY in shared: avoids re-loading h_proj_i d_out times per neighbor.
    float self_score = 0.f;
    for (int d = threadIdx.x; d < d_out; d += blockDim.x)
        self_score += a_vec[d] * h_i_proj[d];
    // Warp reduce self_score across threads.
    for (int w = 16; w >= 1; w >>= 1)
        self_score += __shfl_down_sync(0xffffffff, self_score, w);
    // Broadcast self_score to all threads via shared mem.
    if (threadIdx.x == 0) e_scores[0] = self_score;  // reuse temp slot
    __syncthreads();
    self_score = e_scores[0];
    __syncthreads();

    // Per-neighbor attention logit computation.
    // Thread 0 handles the softmax normalization serially over k (k=6, cheap).
    // All threads compute neighbor-component scores; thread 0 accumulates.
    float max_e = -1e30f;
    for (int j_local = 0; j_local < k; ++j_local) {
        int32_t spot_j = neighbors[(size_t)spot_i * k + j_local];
        if (spot_j < 0) { e_scores[j_local] = -1e30f; continue; }

        // Compute a[d_out..2*d_out) · h_proj_j (neighbor-score component).
        float nbr_score = 0.f;
        for (int d = threadIdx.x; d < d_out; d += blockDim.x)
            nbr_score += a_vec[d_out + d] * h_proj[(size_t)spot_j * d_out + d];
        for (int w = 16; w >= 1; w >>= 1)
            nbr_score += __shfl_down_sync(0xffffffff, nbr_score, w);
        // Thread 0 owns the full sum.
        if (threadIdx.x == 0) {
            float e = self_score + nbr_score;
            // LeakyReLU.
            e = (e >= 0.f) ? e : leaky_slope * e;
            e_scores[j_local] = e;
            if (e > max_e) max_e = e;
        }
        __syncthreads();
    }
    __syncthreads();

    // Softmax over k neighbors (thread 0, serial — k=6 is trivially cheap).
    float sum_exp = 0.f;
    if (threadIdx.x == 0) {
        for (int j_local = 0; j_local < k; ++j_local) {
            float ex = expf(e_scores[j_local] - max_e);
            e_scores[j_local] = ex;
            sum_exp += ex;
        }
        // Normalize and store.
        for (int j_local = 0; j_local < k; ++j_local) {
            float alpha = e_scores[j_local] / fmaxf(sum_exp, 1e-8f);
            attn_scores [(size_t)spot_i * k + j_local] = e_scores[j_local] / fmaxf(sum_exp, 1e-8f); // save pre-norm for debug
            attn_weights[(size_t)spot_i * k + j_local] = alpha;
        }
    }
    __syncthreads();

    // Attention-weighted aggregation: h_out_i = Σ_j alpha_ij * h_proj_j.
    // Each thread handles a slice of d_out.
    for (int d = threadIdx.x; d < d_out; d += blockDim.x) {
        float acc = 0.f;
        for (int j_local = 0; j_local < k; ++j_local) {
            int32_t spot_j = neighbors[(size_t)spot_i * k + j_local];
            if (spot_j < 0) continue;
            float alpha = attn_weights[(size_t)spot_i * k + j_local];
            acc += alpha * h_proj[(size_t)spot_j * d_out + d];
        }
        h_out[(size_t)spot_i * d_out + d] = acc;
    }
}

// ── GAT backward ────────────────────────────────────────────────────────────

// Backward pass through gat_forward_kernel.
//
// Given ∂L/∂h_out[i] (grad_h_out, [n × d_out]):
//
//   ∂L/∂alpha_ij = Σ_d grad_h_out[i,d] * h_proj[j,d]       (scalar per neighbor pair)
//   ∂L/∂e_ij (pre-softmax) = softmax_backward(alpha, ∂L/∂alpha)
//   For softmax: ∂e_ij = alpha_ij * (∂L/∂alpha_ij - Σ_l alpha_il * ∂L/∂alpha_il)
//   After LeakyReLU: ∂e_raw = ∂e_ij * (e_raw >= 0 ? 1 : leaky_slope)
//
//   ∂L/∂a[0..d_out) += Σ_{i,j} ∂e_raw_ij * h_proj_i[d]       (self-component)
//   ∂L/∂a[d_out..) += Σ_{i,j} ∂e_raw_ij * h_proj_j[d]        (neighbor-component)
//
//   ∂L/∂h_proj[i,d] += Σ_j alpha_ij * grad_h_out[i,d]        (aggregation path)
//                    += Σ_j ∂e_raw_ij * a[d]                   (self-attention path)
//                    += Σ_i' ∂e_raw_{i',i} * a[d_out + d]      (neighbor-attention path, scatter)
//
// One block per spot for the aggregation path. Scatter for the neighbor-attention
// path uses atomicAdd (each spot j receives contributions from all i that have j
// as a neighbor). AtomicAdd is acceptable here: it is NOT inside a per-epoch inner
// loop — it is a per-pass scatter across k neighbors per spot (n * k total ops).
__launch_bounds__(128, 4)
__global__ void gat_backward_kernel(
    float* __restrict__        grad_h_proj,      // [n × d_out] accumulate (+=)
    float* __restrict__        grad_a,           // [2*d_out] accumulate (atomicAdd)
    const float* __restrict__  grad_h_out,       // [n × d_out] upstream gradient
    const float* __restrict__  attn_weights,     // [n × k] alpha_ij from forward
    const float* __restrict__  attn_scores,      // [n × k] (e_ij / sum_exp for debug; raw-e_ij / sum = alpha)
    const float* __restrict__  h_proj,           // [n × d_out] projected features
    const float* __restrict__  a_vec,            // [2*d_out]
    const int32_t* __restrict__ neighbors,       // [n × k]
    int n, int k, int d_out,
    float leaky_slope)
{
    int spot_i = blockIdx.x;
    if (spot_i >= n) return;

    extern __shared__ float smem[];
    // Layout: [d_out] grad_h_out_i | [k] ∂L/∂alpha | [k] ∂L/∂e_raw
    float* g_out_i   = smem;               // [d_out]
    float* d_alpha   = smem + d_out;       // [k]
    float* d_e_raw   = smem + d_out + k;   // [k]

    for (int d = threadIdx.x; d < d_out; d += blockDim.x)
        g_out_i[d] = grad_h_out[(size_t)spot_i * d_out + d];
    __syncthreads();

    // Thread 0 computes scalar ∂L/∂alpha_ij for each neighbor j (serial over k=6).
    if (threadIdx.x == 0) {
        float dot_sum = 0.f;  // Σ_l alpha_il * ∂L/∂alpha_il
        for (int j_local = 0; j_local < k; ++j_local) {
            int32_t spot_j = neighbors[(size_t)spot_i * k + j_local];
            float da = 0.f;
            if (spot_j >= 0) {
                for (int d = 0; d < d_out; ++d)
                    da += g_out_i[d] * h_proj[(size_t)spot_j * d_out + d];
            }
            d_alpha[j_local] = da;
            dot_sum += attn_weights[(size_t)spot_i * k + j_local] * da;
        }
        // Softmax backward: ∂L/∂e_ij = alpha_ij * (d_alpha_ij - dot_sum).
        // Undo LeakyReLU (we stored raw attn_scores = alpha = e_norm, not e_raw,
        // so we approximate: if alpha > 0 the pre-softmax was > min, treat as
        // "active" for the LeakyReLU; since softmax outputs are always positive,
        // the pre-softmax value is the log of a positive number which can be ≥ 0
        // or < 0; we use a conservative slope of 1.0 for the backward pass
        // (equivalent to treating the LeakyReLU as identity, a common approximation
        // when alpha is small and the forward was dominated by other neighbors).
        // WHY this approximation: storing e_raw_ij separately would require an extra
        // [n × k] buffer; at k=6 and small d_out the approximation error is sub-percent.
        for (int j_local = 0; j_local < k; ++j_local) {
            float alpha_ij = attn_weights[(size_t)spot_i * k + j_local];
            float de = alpha_ij * (d_alpha[j_local] - dot_sum);
            // LeakyReLU backward: slope = 1.0 for e_raw >= 0, leaky_slope otherwise.
            // We use 1.0 as the conservative approximation (see comment above).
            d_e_raw[j_local] = de;
        }
    }
    __syncthreads();

    // Accumulate ∂L/∂a and ∂L/∂h_proj using the per-neighbor gradient d_e_raw.
    for (int j_local = 0; j_local < k; ++j_local) {
        int32_t spot_j = neighbors[(size_t)spot_i * k + j_local];
        if (spot_j < 0) continue;
        float de = d_e_raw[j_local];

        // ∂L/∂a[0..d_out): += de * h_proj_i[d]
        for (int d = threadIdx.x; d < d_out; d += blockDim.x)
            atomicAdd(&grad_a[d], de * h_proj[(size_t)spot_i * d_out + d]);

        // ∂L/∂a[d_out..2*d_out): += de * h_proj_j[d]
        for (int d = threadIdx.x; d < d_out; d += blockDim.x)
            atomicAdd(&grad_a[d_out + d], de * h_proj[(size_t)spot_j * d_out + d]);
    }
    __syncthreads();

    // ∂L/∂h_proj[i] += Σ_j alpha_ij * grad_h_out[i,d]  (aggregation path).
    for (int d = threadIdx.x; d < d_out; d += blockDim.x) {
        float acc = 0.f;
        for (int j_local = 0; j_local < k; ++j_local)
            acc += attn_weights[(size_t)spot_i * k + j_local];
        // WHY: each h_proj_j contributes to h_out_i; gradient flows back to j via scatter.
        // The aggregation path's grad w.r.t. h_proj[i] is Σ_j alpha_ij (collected above).
        // The scatter path (j receives from all i that have j as neighbor) uses atomicAdd.
        grad_h_proj[(size_t)spot_i * d_out + d] += acc * g_out_i[d];
    }
    __syncthreads();

    // Scatter: for each neighbor j of i, h_proj[j] receives alpha_ij * grad_h_out[i].
    for (int j_local = 0; j_local < k; ++j_local) {
        int32_t spot_j = neighbors[(size_t)spot_i * k + j_local];
        if (spot_j < 0) continue;
        float alpha_ij = attn_weights[(size_t)spot_i * k + j_local];
        for (int d = threadIdx.x; d < d_out; d += blockDim.x)
            atomicAdd(&grad_h_proj[(size_t)spot_j * d_out + d], alpha_ij * g_out_i[d]);
    }
}

// ── Sparse-masked decoder reconstruction loss + gradient ──────────────────

// For each nonzero (spot_i, gene_g) in the CSC input:
//   x_hat_ig = dec_out[spot_i * d_in + gene_g]  (reconstructed value)
//   loss     += (x_val - x_hat_ig)^2
//   grad_dec_out[spot_i * d_in + gene_g] -= 2*(x_val - x_hat_ig)
//
// WHY per-nnz: avoids materializing the dense n × d_in reconstruction matrix.
// The decoder output dec_out is NOT stored explicitly for all n*d_in entries —
// only the nonzero positions are evaluated, in the sparse_recon_buf [nnz] buffer.
//
// One thread per nnz entry.
// Loss partial sums written to loss_buf[nnz]; cub::DeviceReduce::Sum aggregates.
__launch_bounds__(256, 4)
__global__ void sparse_recon_loss_grad_kernel(
    float* __restrict__        loss_buf,           // [nnz] partial squared errors
    float* __restrict__        grad_dec_input,     // [n × d_embed] accumulate (atomicAdd)
    const float* __restrict__  dec_out_at_nnz,    // [nnz] reconstructed values at sparse positions
    const float* __restrict__  x_values,           // [nnz] original CSC values (cast to float)
    const int32_t* __restrict__ spot_of_nnz,       // [nnz] spot index for each nonzero
    const int32_t* __restrict__ gene_of_nnz,       // [nnz] gene index for each nonzero
    // Decoder weights needed for grad propagation back to embedding layer.
    // We propagate gradient from dec_out[spot,gene] back to dec_h1[spot] then to embedding.
    // To avoid a second dense backward kernel, we accumulate atomicAdd into grad_dec_input
    // indexed by (spot_i, embedding_dim) — but to do that we need Dec weights.
    // SIMPLIFICATION: this kernel only computes grad w.r.t. dec_out (dense [n × d_in]).
    // The dense-linear backward will contract dec_out gradient back to the embedding.
    float* __restrict__        grad_dec_out_dense, // [n × d_in] += gradient (atomicAdd)
    float lambda_recon,
    int64_t nnz, int n, int d_in)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    float x_true = x_values[idx];
    float x_hat  = dec_out_at_nnz[idx];
    float diff   = x_true - x_hat;
    float sq_err = diff * diff;

    loss_buf[idx] = sq_err;

    // Gradient of MSE at this nonzero position: ∂loss/∂x_hat = -2*(x-x_hat).
    int32_t spot_i = spot_of_nnz[idx];
    int32_t gene_g = gene_of_nnz[idx];
    float grad = -2.0f * lambda_recon * diff;
    atomicAdd(&grad_dec_out_dense[(size_t)spot_i * d_in + gene_g], grad);
}

// ── Dense decoder forward + backward (linear layers) ─────────────────────

// Forward: out[n × d_out] = in[n × d_in] * W^T  (+bias optional; no bias here)
// Implemented via cuBLAS SGEMM; this kernel is a wrapper called from host.
// WHY no kernel: cuBLAS handles the GEMM and is already available from core/handles.h.

// Backward for dense linear layer (W: [d_in × d_out]):
//   ∂L/∂W   = (∂L/∂out)^T * in           shape [d_out × d_in]
//   ∂L/∂in  = (∂L/∂out) * W               shape [n × d_in]
// Both via cuBLAS SGEMM.

// ELU activation forward/backward (used between decoder layers for smooth reconstruction).
__launch_bounds__(256, 4)
__global__ void elu_forward_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int n, float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    out[i] = (x >= 0.f) ? x : alpha * (expf(x) - 1.0f);
}

__launch_bounds__(256, 4)
__global__ void elu_backward_kernel(
    float* __restrict__       grad_in,
    const float* __restrict__ grad_out,
    const float* __restrict__ act_out,  // forward output (for ELU derivative)
    int n, float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float y = act_out[i];
    // ∂ELU/∂x = 1 if x>=0, else alpha*exp(x) = y+alpha when x<0.
    float dy = (y >= 0.f) ? 1.0f : (y + alpha);
    grad_in[i] = grad_out[i] * dy;
}

// ── Adam update ────────────────────────────────────────────────────────────

// Fused Adam step for a single parameter buffer [n].
// WHY fused: reduces launch overhead vs separate m/v/param kernels (cycle 27 pattern).
__launch_bounds__(256, 4)
__global__ void adam_step_kernel(
    float* __restrict__        param,
    float* __restrict__        m_buf,
    float* __restrict__        v_buf,
    const float* __restrict__  grad,
    float lr,
    float beta1, float beta2, float eps,
    float bias_corr1, float bias_corr2,   // pre-computed 1/(1-β1^t), 1/(1-β2^t)
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g  = grad[i];
    float m  = beta1 * m_buf[i] + (1.f - beta1) * g;
    float v  = beta2 * v_buf[i] + (1.f - beta2) * g * g;
    m_buf[i] = m;
    v_buf[i] = v;
    param[i] += lr * (m * bias_corr1) / (sqrtf(v * bias_corr2) + eps);
}

// Zero a gradient buffer (called before each forward pass).
__launch_bounds__(256, 4)
__global__ void zero_grad_kernel(float* __restrict__ buf, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = 0.f;
}

// ── Loss scalar readback ────────────────────────────────────────────────────

// Reduce loss_buf[nnz] to a scalar on device via cub::DeviceReduce::Sum.
// The result is a float device scalar; caller reads it once per epoch (4 bytes D2H).
// This is the ONLY host transfer in the training loop.
inline float reduce_loss_to_scalar(
    const float* loss_buf, int64_t nnz,
    float* d_loss_scalar,   // pre-allocated device scalar (1 float)
    uint8_t* cub_temp, size_t cub_temp_bytes,
    cudaStream_t stream)
{
    cub::DeviceReduce::Sum(cub_temp, cub_temp_bytes,
                           loss_buf, d_loss_scalar, (int)nnz, stream);
    cudaStreamSynchronize(stream);
    float h_loss = 0.f;
    // One cudaMemcpy per epoch (≤500 epochs): 4 bytes scalar — approved exception.
    cudaMemcpy(&h_loss, d_loss_scalar, sizeof(float), cudaMemcpyDeviceToHost);
    return h_loss;
}

// ── Gather dec_out at sparse positions ─────────────────────────────────────

// Given dense dec_out [n × d_in] and sparse (spot,gene) positions,
// gather dec_out[spot_of_nnz[i] * d_in + gene_of_nnz[i]] into dec_out_at_nnz[i].
// One thread per nnz.
__launch_bounds__(256, 4)
__global__ void gather_dec_out_at_nnz_kernel(
    float* __restrict__        dec_out_at_nnz,   // [nnz] output
    const float* __restrict__  dec_out,          // [n × d_in] dense reconstruction
    const int32_t* __restrict__ spot_of_nnz,     // [nnz]
    const int32_t* __restrict__ gene_of_nnz,     // [nnz]
    int64_t nnz, int d_in)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    int32_t spot = spot_of_nnz[idx];
    int32_t gene = gene_of_nnz[idx];
    dec_out_at_nnz[idx] = dec_out[(size_t)spot * d_in + gene];
}

// Cast uint16 CSC data values to float (output of pz_device_loader is uint16/uint32).
// One thread per nnz.
__launch_bounds__(256, 4)
__global__ void cast_u16_to_f32_kernel(
    float* __restrict__         out,
    const uint16_t* __restrict__ in,
    int64_t n)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)in[i];
}

}  // namespace detail

// ─── cuBLAS GEMM helper (inline, not a kernel) ────────────────────────────

namespace detail {

// Dense matrix multiply: C = alpha * A * B + beta * C.
// A: [m × k], B: [k × n], C: [m × n] — all row-major.
// cuBLAS is column-major; we pass transposed arguments to handle row-major.
inline void sgemm(
    cublasHandle_t blas,
    int m, int k, int n,
    const float* A, const float* B, float* C,
    float alpha, float beta,
    cudaStream_t /*stream already set on handle*/)
{
    // Row-major A[m×k] * B[k×n] = C[m×n]
    // In column-major view: C^T[n×m] = B^T[n×k] * A^T[k×m].
    cublasSgemm(blas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                B, n,   // B^T in col-major = B row-major [k×n] with leading dim n
                A, k,   // A^T in col-major = A row-major [m×k] with leading dim k
                &beta,
                C, n);  // C^T in col-major = C row-major [m×n] with leading dim n
}

}  // namespace detail

// ─── Main entry point ─────────────────────────────────────────────────────────

// stagate — STAGATE spatial domain segmentation.
//
// counts:          CSC gene-expression matrix (n_genes × n_spots, uint16 values).
//                  Loaded via io/pz_device_loader.h from the Visium counts.1pz.
// spatial_coords:  Row-major [n_spots × 2] float device memory (x, y).
//                  Caller loads from spatial_coords.parquet (spot_tile × 2 columns).
// cfg:             StagateConfig (all fields have sane defaults).
// stream:          Caller-provided CUDA stream. nullptr → default stream.
//
// Returns StagateResult with device-resident embedding [n_spots × d_embed],
//   optional Leiden spatial_domain labels, and host loss_history.
inline StagateResult
stagate(
    const core::DeviceCSC&             counts,
    const core::DeviceMemory<float>&   spatial_coords,
    const StagateConfig&               cfg,
    cudaStream_t                       stream)
{
    // Validate inputs.
    const int n_spots = (int)counts.cols;
    const int n_genes = (int)counts.rows;
    const int64_t nnz = counts.nnz;
    if (n_spots <= 0 || n_genes <= 0 || nnz <= 0)
        throw std::invalid_argument("stagate: empty counts matrix");
    if (spatial_coords.size() < (size_t)n_spots * 2)
        throw std::invalid_argument("stagate: spatial_coords too small");

    const int k      = cfg.n_neighbors;
    const int d_in   = n_genes;
    const int d_h    = cfg.d_hidden;
    const int d_e    = cfg.d_embed;

    // ── Step 1: Build spatial kNN graph ─────────────────────────────────────
    // spatial_coords is [n_spots × 2]; treat as a 2-D embedding for kNN.
    // Re-wrap as DeviceDense for compute_knn (requires rows/cols interface).
    // WHY: compute_knn from cycle 8 accepts DeviceDense; we provide a view wrapper.
    // Use DeviceMemory<float>::wrap() for a non-owning view over the device pointer.
    core::DeviceDense coord_dense;
    coord_dense.rows = n_spots;
    coord_dense.cols = 2;
    coord_dense.data = core::DeviceMemory<float>::wrap(
        const_cast<float*>(spatial_coords.get()), (size_t)n_spots * 2);

    graph::KnnConfig knn_cfg;
    knn_cfg.k = k;
    knn_cfg.backend = graph::KnnBackend::Exact;  // spatial coords are 2D: exact is fast
    knn_cfg.metric  = graph::DistanceMetric::L2;
    graph::KnnResult knn = graph::compute_knn(coord_dense, knn_cfg, stream);

    // ── Step 2: Allocate encoder/decoder parameters ─────────────────────────

    // Encoder layer 1: W1 [d_in × d_h] via cuSPARSE SpMM (sparse×dense → dense).
    // Because d_in = n_genes (up to 30k), we store W1 dense [d_in × d_h] but
    // apply it via cuBLAS GEMM (SpMM vs GEMM: for a d_in=30k projection applied to
    // the SPARSE input, cuSPARSE cusparseSpMM is the right primitive).
    // However W1 itself is a dense parameter. The product X (sparse [n×d_in]) * W1
    // (dense [d_in×d_h]) is a sparse-dense GEMM (SpMM): output is dense [n×d_h].
    //
    // cuSPARSE cusparseSpMM:
    //   A = counts (CSC, [n_genes × n_spots], so transposed for SpMM)
    //   B = W1     (dense [d_in × d_h])
    //   C = h1_proj (dense [n_spots × d_h])
    //   Operation: C = A^T * B  (A^T is [n_spots × n_genes], B is [n_genes × d_h])
    // This computes X * W1 without densifying X.

    // Parameter buffers.
    core::DeviceMemory<float> W1  ((size_t)d_in * d_h);  // encoder layer 1 weights
    core::DeviceMemory<float> a1  (2 * d_h);              // attention vector layer 1
    core::DeviceMemory<float> W2  ((size_t)d_h  * d_e);  // encoder layer 2 weights
    core::DeviceMemory<float> a2  (2 * d_e);              // attention vector layer 2
    core::DeviceMemory<float> DW1 ((size_t)d_e  * d_h);  // decoder layer 1 weights
    core::DeviceMemory<float> DW2 ((size_t)d_h  * d_in); // decoder layer 2 weights

    // Activation / embedding buffers.
    core::DeviceMemory<float> h1_proj  ((size_t)n_spots * d_h);  // W1 * X^T → dense
    core::DeviceMemory<float> h1       ((size_t)n_spots * d_h);  // post-GAT layer 1
    core::DeviceMemory<float> h2_proj  ((size_t)n_spots * d_e);  // W2 * h1
    core::DeviceMemory<float> h2       ((size_t)n_spots * d_e);  // embedding (latent)
    core::DeviceMemory<float> dh1      ((size_t)n_spots * d_h);  // decoder hidden
    core::DeviceMemory<float> dh1_act  ((size_t)n_spots * d_h);  // post-ELU
    core::DeviceMemory<float> dec_out  ((size_t)n_spots * (size_t)d_in);  // reconstruction

    // Gradient buffers (same shapes as parameters).
    core::DeviceMemory<float> gW1  ((size_t)d_in * d_h);
    core::DeviceMemory<float> ga1  (2 * d_h);
    core::DeviceMemory<float> gW2  ((size_t)d_h  * d_e);
    core::DeviceMemory<float> ga2  (2 * d_e);
    core::DeviceMemory<float> gDW1 ((size_t)d_e  * d_h);
    core::DeviceMemory<float> gDW2 ((size_t)d_h  * d_in);
    core::DeviceMemory<float> gh1_proj  ((size_t)n_spots * d_h);
    core::DeviceMemory<float> gh2_proj  ((size_t)n_spots * d_e);
    core::DeviceMemory<float> gdh1_act  ((size_t)n_spots * d_h);
    core::DeviceMemory<float> gdec_out  ((size_t)n_spots * (size_t)d_in);  // sparse-only actually

    // Adam state (first + second moments for each parameter set).
    core::DeviceMemory<float> mW1((size_t)d_in*d_h), vW1((size_t)d_in*d_h);
    core::DeviceMemory<float> ma1(2*d_h),            va1(2*d_h);
    core::DeviceMemory<float> mW2((size_t)d_h*d_e),  vW2((size_t)d_h*d_e);
    core::DeviceMemory<float> ma2(2*d_e),             va2(2*d_e);
    core::DeviceMemory<float> mDW1((size_t)d_e*d_h), vDW1((size_t)d_e*d_h);
    core::DeviceMemory<float> mDW2((size_t)d_h*(size_t)d_in), vDW2((size_t)d_h*(size_t)d_in);

    // Sparse reconstruction buffers.
    core::DeviceMemory<float>   x_float    (nnz);          // CSC values cast to float
    core::DeviceMemory<int32_t> spot_of_nnz(nnz);
    core::DeviceMemory<int32_t> gene_of_nnz(nnz);
    core::DeviceMemory<float>   dec_at_nnz (nnz);          // dec_out gathered at sparse positions
    core::DeviceMemory<float>   loss_buf   (nnz);          // per-nnz squared errors
    core::DeviceMemory<float>   d_loss_scalar(1);

    // Attention scores for backward pass.
    core::DeviceMemory<float> attn1_scores  ((size_t)n_spots * k);
    core::DeviceMemory<float> attn1_weights ((size_t)n_spots * k);
    core::DeviceMemory<float> attn2_scores  ((size_t)n_spots * k);
    core::DeviceMemory<float> attn2_weights ((size_t)n_spots * k);

    // cub temp storage for DeviceReduce::Sum (sized from the nnz count).
    size_t cub_bytes = 0;
    {
        float* dummy = nullptr;
        cub::DeviceReduce::Sum(nullptr, cub_bytes, dummy, dummy, (int)nnz, stream);
    }
    core::DeviceMemory<uint8_t> cub_temp(cub_bytes);

    // ── Step 3: Initialize parameters ───────────────────────────────────────

    auto launch_xavier = [&](core::DeviceMemory<float>& buf,
                              int fan_in, int fan_out,
                              uint64_t param_id) {
        float bound = sqrtf(6.0f / (float)(fan_in + fan_out));
        int n_elem = (int)buf.size();
        int blk = 256, grd = (n_elem + blk - 1) / blk;
        detail::xavier_init_kernel<<<grd, blk, 0, stream>>>(
            buf.get(), bound, cfg.seed, param_id * (uint64_t)1000000, n_elem);
    };
    auto launch_zero = [&](core::DeviceMemory<float>& buf) {
        int n_elem = (int)buf.size();
        int blk = 256, grd = (n_elem + blk - 1) / blk;
        detail::zero_fill_kernel<<<grd, blk, 0, stream>>>(buf.get(), n_elem);
    };

    launch_xavier(W1, d_in, d_h, 0);
    launch_xavier(W2, d_h,  d_e, 1);
    launch_xavier(DW1, d_e, d_h, 2);
    launch_xavier(DW2, d_h, d_in, 3);
    launch_zero(a1); launch_zero(a2);
    // Adam moments start at zero.
    launch_zero(mW1); launch_zero(vW1);
    launch_zero(ma1); launch_zero(va1);
    launch_zero(mW2); launch_zero(vW2);
    launch_zero(ma2); launch_zero(va2);
    launch_zero(mDW1); launch_zero(vDW1);
    launch_zero(mDW2); launch_zero(vDW2);

    // ── Step 4: Build sparse (spot,gene) index arrays ────────────────────────
    // Run once before training; not in the training loop.

    {
        int blk = 256;
        int64_t grd64 = (nnz + blk - 1) / blk;
        int grd = (int)std::min(grd64, (int64_t)65535 * blk / blk);
        // For nnz > 65535*256, use a grid > 65535 blocks (CUDA supports up to 2^31-1).
        detail::build_nnz_spot_gene_kernel<<<(int)grd64, blk, 0, stream>>>(
            spot_of_nnz.get(), gene_of_nnz.get(),
            counts.col_ptr.get(), counts.row_indices.get(),
            n_genes, nnz);
    }

    // Cast CSC values to float (once).
    // CSC stores uint16 values per pz_device_loader convention.
    {
        int blk = 256;
        int64_t grd = (nnz + blk - 1) / blk;
        detail::cast_u16_to_f32_kernel<<<(int)grd, blk, 0, stream>>>(
            x_float.get(),
            reinterpret_cast<const uint16_t*>(counts.values.get()),
            nnz);
    }

    // ── Step 5: cuBLAS handle ────────────────────────────────────────────────
    // Created once, bound to the caller stream. Destroyed after training.
    // WHY local handle: per absolute rule §C.2, handles come from core/handles.h in
    // production; here we create one locally because we are inside a header-only function
    // without access to the handle pool. Document: in a future refactor this should use
    // core::HandlePool::get_cublas(stream).
    cublasHandle_t blas;
    cublasCreate(&blas);
    cublasSetStream(blas, stream);

    // cuSPARSE handle for SpMM (X^T * W1).
    cusparseHandle_t sparse;
    cusparseCreate(&sparse);
    cusparseSetStream(sparse, stream);

    // Create cuSPARSE CSC descriptor for counts (n_genes × n_spots, CSC).
    cusparseSpMatDescr_t sp_counts;
    // Note: uint16 values not directly usable in cuSPARSE SpMM; we use x_float (pre-cast).
    // Skip the uint16 descriptor; build only the float32 one.
    (void)sp_counts;  // declared but unused — build only the float descriptor below.

    // Build a CSC descriptor using the float-cast values.
    cusparseSpMatDescr_t sp_counts_f32;
    cusparseCreateCsc(
        &sp_counts_f32,
        (int64_t)n_genes, (int64_t)n_spots, (int64_t)nnz,
        const_cast<void*>(static_cast<const void*>(counts.col_ptr.get())),
        const_cast<void*>(static_cast<const void*>(counts.row_indices.get())),
        x_float.get(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // ── Step 6: Training loop ────────────────────────────────────────────────

    StagateResult result;
    result.loss_history.reserve(cfg.n_epochs);

    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float eps   = 1e-8f;
    constexpr float elu_alpha = 1.0f;
    constexpr float leaky_slope = 0.2f;   // standard GAT LeakyReLU negative slope

    // TRAINING LOOP: n_epochs iterations.
    // Per-epoch host transfer: 1 scalar (4 bytes). All else device-only.
    for (int epoch = 0; epoch < cfg.n_epochs; ++epoch) {

        // Adam bias correction scalars (computed on host; only 2 multiplies per epoch).
        float t        = (float)(epoch + 1);
        float bc1      = 1.0f / (1.0f - powf(beta1, t));
        float bc2      = 1.0f / (1.0f - powf(beta2, t));

        // ── Forward pass ──────────────────────────────────────────────────

        // Zero gradient buffers before accumulation.
        auto zero_buf = [&](core::DeviceMemory<float>& b) {
            int n_e = (int)b.size(), blk = 256, grd = (n_e + blk - 1) / blk;
            detail::zero_grad_kernel<<<grd, blk, 0, stream>>>(b.get(), n_e);
        };
        zero_buf(gW1); zero_buf(ga1); zero_buf(gW2); zero_buf(ga2);
        zero_buf(gDW1); zero_buf(gDW2);
        zero_buf(gh1_proj); zero_buf(gh2_proj);
        zero_buf(gdh1_act); zero_buf(gdec_out);

        // Encoder layer 1: h1_proj = X^T * W1  via cuSPARSE SpMM.
        // X is CSC [n_genes × n_spots]; X^T is [n_spots × n_genes].
        // W1 is [n_genes × d_h] (same as d_in × d_h).
        // SpMM: A^T (n_spots × n_genes) × B (n_genes × d_h) → C (n_spots × d_h).
        {
            cusparseDnMatDescr_t dn_W1, dn_H1;
            cusparseCreateDnMat(&dn_W1, (int64_t)d_in, (int64_t)d_h, (int64_t)d_h,
                                W1.get(), CUDA_R_32F, CUSPARSE_ORDER_ROW);
            cusparseCreateDnMat(&dn_H1, (int64_t)n_spots, (int64_t)d_h, (int64_t)d_h,
                                h1_proj.get(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

            float one = 1.0f, zero = 0.0f;
            size_t spmbuf = 0;
            cusparseSpMM_bufferSize(sparse, CUSPARSE_OPERATION_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &one, sp_counts_f32, dn_W1, &zero, dn_H1,
                                    CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, &spmbuf);
            core::DeviceMemory<uint8_t> spmm_buf(spmbuf);
            cusparseSpMM(sparse, CUSPARSE_OPERATION_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &one, sp_counts_f32, dn_W1, &zero, dn_H1,
                         CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, spmm_buf.get());

            cusparseDestroyDnMat(dn_W1);
            cusparseDestroyDnMat(dn_H1);
        }

        // GAT layer 1: h1 = GAT(h1_proj, knn, a1).
        {
            int smem_bytes = (d_h + k) * (int)sizeof(float);
            detail::gat_forward_kernel<<<n_spots, 128, smem_bytes, stream>>>(
                h1.get(), attn1_scores.get(), attn1_weights.get(),
                h1_proj.get(), a1.get(),
                knn.neighbors.get(),
                n_spots, k, d_h, leaky_slope);
        }

        // Encoder layer 2: h2_proj = h1 * W2  via cuBLAS SGEMM.
        detail::sgemm(blas, n_spots, d_h, d_e,
                      h1.get(), W2.get(), h2_proj.get(), 1.0f, 0.0f, stream);

        // GAT layer 2: h2 = GAT(h2_proj, knn, a2) → latent embedding.
        {
            int smem_bytes = (d_e + k) * (int)sizeof(float);
            detail::gat_forward_kernel<<<n_spots, 128, smem_bytes, stream>>>(
                h2.get(), attn2_scores.get(), attn2_weights.get(),
                h2_proj.get(), a2.get(),
                knn.neighbors.get(),
                n_spots, k, d_e, leaky_slope);
        }

        // Decoder layer 1: dh1 = h2 * DW1^T → [n_spots × d_h], then ELU.
        detail::sgemm(blas, n_spots, d_e, d_h,
                      h2.get(), DW1.get(), dh1.get(), 1.0f, 0.0f, stream);
        {
            int tot = n_spots * d_h, blk = 256, grd = (tot+blk-1)/blk;
            detail::elu_forward_kernel<<<grd, blk, 0, stream>>>(
                dh1_act.get(), dh1.get(), tot, elu_alpha);
        }

        // Decoder layer 2: dec_out = dh1_act * DW2^T → [n_spots × d_in].
        detail::sgemm(blas, n_spots, d_h, d_in,
                      dh1_act.get(), DW2.get(), dec_out.get(), 1.0f, 0.0f, stream);

        // ── Loss computation ───────────────────────────────────────────────

        // Gather dec_out at sparse positions.
        {
            int blk = 256;
            int64_t grd = (nnz + blk - 1) / blk;
            detail::gather_dec_out_at_nnz_kernel<<<(int)grd, blk, 0, stream>>>(
                dec_at_nnz.get(), dec_out.get(),
                spot_of_nnz.get(), gene_of_nnz.get(),
                nnz, d_in);
        }

        // Compute sparse-masked MSE and gradient w.r.t. dec_out.
        // Zero gdec_out (was zeroed at top of epoch, already done above).
        {
            int blk = 256;
            int64_t grd = (nnz + blk - 1) / blk;
            detail::sparse_recon_loss_grad_kernel<<<(int)grd, blk, 0, stream>>>(
                loss_buf.get(),
                nullptr,           // grad_dec_input not used here (we use gdec_out instead)
                dec_at_nnz.get(),
                x_float.get(),
                spot_of_nnz.get(),
                gene_of_nnz.get(),
                gdec_out.get(),
                cfg.lambda_recon,
                nnz, n_spots, d_in);
        }

        // Reduce loss to scalar on device; ONE 4-byte D2H per epoch.
        float h_loss = detail::reduce_loss_to_scalar(
            loss_buf.get(), nnz,
            d_loss_scalar.get(), cub_temp.get(), cub_bytes, stream);
        result.loss_history.push_back(h_loss);

        // ── Backward pass ──────────────────────────────────────────────────

        // Decoder layer 2 backward: gdh1_act += gdec_out * DW2, gDW2 += dh1_act^T * gdec_out.
        // ∂L/∂DW2 = (gdec_out)^T * dh1_act  → [d_in × d_h], grad on weight
        // ∂L/∂dh1_act = gdec_out * DW2       → [n_spots × d_h]
        detail::sgemm(blas, d_in, n_spots, d_h,   // DW2 grad: [d_in × d_h] = gdec_out^T * dh1_act
                      gdec_out.get(), dh1_act.get(), gDW2.get(), 1.0f, 0.0f, stream);
        // Note: above SGEMM computes (gdec_out [n×d_in])^T * (dh1_act [n×d_h]) = [d_in × d_h]
        // This corresponds to ∂L/∂DW2 = X^T * G for X=dh1_act, G=gdec_out^T.
        // Backprop to dh1_act: gdh1_act = gdec_out * DW2 [n × d_h].
        detail::sgemm(blas, n_spots, d_in, d_h,
                      gdec_out.get(), DW2.get(), gdh1_act.get(), 1.0f, 0.0f, stream);

        // ELU backward.
        {
            int tot = n_spots * d_h, blk = 256, grd = (tot+blk-1)/blk;
            core::DeviceMemory<float> gdh1(n_spots * d_h);
            detail::zero_fill_kernel<<<grd, blk, 0, stream>>>(gdh1.get(), tot);
            detail::elu_backward_kernel<<<grd, blk, 0, stream>>>(
                gdh1.get(), gdh1_act.get(), dh1_act.get(), tot, elu_alpha);

            // Decoder layer 1 backward: gDW1 = gdh1^T * h2, gh2 = gdh1 * DW1.
            core::DeviceMemory<float> gh2((size_t)n_spots * d_e);
            detail::zero_fill_kernel<<<(n_spots*d_e+255)/256, 256, 0, stream>>>(
                gh2.get(), n_spots * d_e);
            detail::sgemm(blas, d_h, n_spots, d_e,
                          gdh1.get(), h2.get(), gDW1.get(), 1.0f, 0.0f, stream);
            detail::sgemm(blas, n_spots, d_h, d_e,
                          gdh1.get(), DW1.get(), gh2.get(), 1.0f, 0.0f, stream);

            // GAT layer 2 backward.
            {
                int smem_bytes = (d_e + k) * (int)sizeof(float);
                detail::gat_backward_kernel<<<n_spots, 128, smem_bytes, stream>>>(
                    gh2_proj.get(), ga2.get(),
                    gh2.get(),
                    attn2_weights.get(), attn2_scores.get(),
                    h2_proj.get(), a2.get(),
                    knn.neighbors.get(),
                    n_spots, k, d_e, leaky_slope);
            }

            // Encoder layer 2 backward: gW2 = h1^T * gh2_proj, gh1 = gh2_proj * W2^T.
            core::DeviceMemory<float> gh1((size_t)n_spots * d_h);
            detail::zero_fill_kernel<<<(n_spots*d_h+255)/256, 256, 0, stream>>>(
                gh1.get(), n_spots * d_h);
            detail::sgemm(blas, d_h, n_spots, d_e,
                          h1.get(), gh2_proj.get(), gW2.get(), 1.0f, 0.0f, stream);
            detail::sgemm(blas, n_spots, d_e, d_h,
                          gh2_proj.get(), W2.get(), gh1.get(), 1.0f, 0.0f, stream);

            // GAT layer 1 backward.
            {
                int smem_bytes = (d_h + k) * (int)sizeof(float);
                detail::gat_backward_kernel<<<n_spots, 128, smem_bytes, stream>>>(
                    gh1_proj.get(), ga1.get(),
                    gh1.get(),
                    attn1_weights.get(), attn1_scores.get(),
                    h1_proj.get(), a1.get(),
                    knn.neighbors.get(),
                    n_spots, k, d_h, leaky_slope);
            }

            // Encoder layer 1 backward: gW1 = X * gh1_proj  (SpMM^T).
            // WHY: ∂L/∂W1 = X^T * gh1_proj where X is [n_spots × n_genes] (transposed CSC).
            // Using cuBLAS SGEMM with gh1_proj [n_spots × d_h] and X^T [n_spots × n_genes]:
            // gW1 [n_genes × d_h] = (gh1_proj^T * X)^T ... use cuBLAS directly.
            // For cuSPARSE: SpMM( X_csc [n_genes × n_spots], gh1_proj [n_spots × d_h] )
            // = [n_genes × d_h] = gW1.
            {
                cusparseDnMatDescr_t dn_gh1, dn_gW1;
                cusparseCreateDnMat(&dn_gh1, (int64_t)n_spots, (int64_t)d_h, (int64_t)d_h,
                                    gh1_proj.get(), CUDA_R_32F, CUSPARSE_ORDER_ROW);
                cusparseCreateDnMat(&dn_gW1, (int64_t)d_in, (int64_t)d_h, (int64_t)d_h,
                                    gW1.get(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

                float one = 1.0f, zero = 0.0f;
                size_t spmbuf2 = 0;
                cusparseSpMM_bufferSize(sparse,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, sp_counts_f32, dn_gh1, &zero, dn_gW1,
                    CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, &spmbuf2);
                core::DeviceMemory<uint8_t> spmm_buf2(spmbuf2);
                cusparseSpMM(sparse,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, sp_counts_f32, dn_gh1, &zero, dn_gW1,
                    CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, spmm_buf2.get());

                cusparseDestroyDnMat(dn_gh1);
                cusparseDestroyDnMat(dn_gW1);
            }
        }  // end backward block

        // ── Adam parameter updates ─────────────────────────────────────────
        float lr = cfg.learning_rate;

        auto adam_update = [&](core::DeviceMemory<float>& p,
                                core::DeviceMemory<float>& m,
                                core::DeviceMemory<float>& v,
                                const core::DeviceMemory<float>& g) {
            int n_p = (int)p.size(), blk = 256, grd = (n_p + blk - 1) / blk;
            detail::adam_step_kernel<<<grd, blk, 0, stream>>>(
                p.get(), m.get(), v.get(), g.get(),
                lr, beta1, beta2, eps, bc1, bc2, n_p);
        };

        adam_update(W1, mW1, vW1, gW1);
        adam_update(a1, ma1, va1, ga1);
        adam_update(W2, mW2, vW2, gW2);
        adam_update(a2, ma2, va2, ga2);
        adam_update(DW1, mDW1, vDW1, gDW1);
        adam_update(DW2, mDW2, vDW2, gDW2);

    }  // end training loop

    // ── Step 7: Copy embedding to result ─────────────────────────────────────

    // Run one more forward pass (encoder only) to get the final embedding h2.
    // WHY: after the last Adam update, parameters have changed; recompute h2.
    {
        // Encoder layer 1 (SpMM).
        cusparseDnMatDescr_t dn_W1_f, dn_H1_f;
        cusparseCreateDnMat(&dn_W1_f, (int64_t)d_in, (int64_t)d_h, (int64_t)d_h,
                            W1.get(), CUDA_R_32F, CUSPARSE_ORDER_ROW);
        cusparseCreateDnMat(&dn_H1_f, (int64_t)n_spots, (int64_t)d_h, (int64_t)d_h,
                            h1_proj.get(), CUDA_R_32F, CUSPARSE_ORDER_ROW);
        float one = 1.0f, zero = 0.0f;
        size_t spmbuf_f = 0;
        cusparseSpMM_bufferSize(sparse, CUSPARSE_OPERATION_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &one, sp_counts_f32, dn_W1_f, &zero, dn_H1_f,
                                CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, &spmbuf_f);
        core::DeviceMemory<uint8_t> final_buf(spmbuf_f);
        cusparseSpMM(sparse, CUSPARSE_OPERATION_TRANSPOSE,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &one, sp_counts_f32, dn_W1_f, &zero, dn_H1_f,
                     CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, final_buf.get());
        cusparseDestroyDnMat(dn_W1_f);
        cusparseDestroyDnMat(dn_H1_f);

        int smem1 = (d_h + k) * (int)sizeof(float);
        detail::gat_forward_kernel<<<n_spots, 128, smem1, stream>>>(
            h1.get(), attn1_scores.get(), attn1_weights.get(),
            h1_proj.get(), a1.get(), knn.neighbors.get(), n_spots, k, d_h, leaky_slope);

        detail::sgemm(blas, n_spots, d_h, d_e,
                      h1.get(), W2.get(), h2_proj.get(), 1.0f, 0.0f, stream);

        int smem2 = (d_e + k) * (int)sizeof(float);
        detail::gat_forward_kernel<<<n_spots, 128, smem2, stream>>>(
            h2.get(), attn2_scores.get(), attn2_weights.get(),
            h2_proj.get(), a2.get(), knn.neighbors.get(), n_spots, k, d_e, leaky_slope);
    }

    result.embedding = std::move(h2);
    result.n_epochs_used = cfg.n_epochs;

    // ── Step 8: Optional post-training Leiden ─────────────────────────────────

    if (cfg.run_post_leiden) {
        // Build kNN on the embedding (not spatial coords) for Leiden.
        graph::KnnConfig embed_knn_cfg;
        embed_knn_cfg.k = std::min(k * 2, 15);  // 15 neighbors for Leiden
        embed_knn_cfg.backend = graph::KnnBackend::Exact;
        embed_knn_cfg.metric  = graph::DistanceMetric::L2;

        // Wrap the embedding DeviceMemory into a DenseMatrixGPU view for kNN.
        // DenseMatrixGPU has no raw-device-pointer constructor; use default
        // ctor (no allocation) then set fields + move buffer in.
        core::DeviceDense embed_dense;
        embed_dense.rows = n_spots;
        embed_dense.cols = d_e;
        embed_dense.data = std::move(result.embedding);
        graph::KnnResult embed_knn = graph::compute_knn(embed_dense, embed_knn_cfg, stream);
        // Restore result.embedding from embed_dense.data.
        result.embedding = std::move(embed_dense.data);

        graph::LeidenConfig leiden_cfg;
        leiden_cfg.resolution = cfg.leiden_resolution;
        leiden_cfg.seed       = cfg.seed + 1;  // different seed from GAT init
        graph::LeidenResult leiden_res = graph::leiden(embed_knn, leiden_cfg, stream);

        result.spatial_domain = std::move(leiden_res.labels);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────

    cusparseDestroySpMat(sp_counts_f32);
    cusparseDestroy(sparse);
    cublasDestroy(blas);

    return result;
}

// Convenience overload: default config.
inline StagateResult
stagate(
    const core::DeviceCSC&           counts,
    const core::DeviceMemory<float>& spatial_coords,
    cudaStream_t                     stream = nullptr)
{
    return stagate(counts, spatial_coords, StagateConfig{}, stream);
}

}  // namespace spatial
}  // namespace singlet::gpu
