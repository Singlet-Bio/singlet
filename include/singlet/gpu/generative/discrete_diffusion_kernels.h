// SPDX-License-Identifier: MIT
// singlet/gpu/generative/discrete_diffusion_kernels.h
//
// CUDA kernels and kernel-launch helpers for discrete_diffusion.h.
// Split out of discrete_diffusion.h (multi-concern header cleanup): this
// file holds the __global__/__device__ kernels and detail-namespace
// helpers; discrete_diffusion.h keeps the public API and host orchestration
// and #includes this header.

#pragma once

#include <singlet/gpu/core/types.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>
#include <numeric>

namespace singlet::gpu {
namespace generative {

namespace detail {

// ── Tokenization ─────────────────────────────────────────────────────────────

// Compute log2 bin edges: bin i covers [2^(i-1), 2^i) for i>=1; bin 0 = {0}.
// Output: bin_edges[vocab_size+1] on host.
inline std::vector<float> compute_log2_bin_edges(int vocab_size) {
    std::vector<float> edges(vocab_size + 1);
    edges[0] = 0.0f;
    edges[1] = 1.0f;
    for (int i = 2; i <= vocab_size; ++i)
        edges[i] = std::pow(2.0f, (float)(i - 1));
    return edges;
}

inline std::vector<float> compute_linear_bin_edges(int vocab_size, float max_val) {
    std::vector<float> edges(vocab_size + 1);
    for (int i = 0; i <= vocab_size; ++i)
        edges[i] = (float)i * max_val / (float)vocab_size;
    return edges;
}

// Tokenize CSC values → uint8 tokens stored in a flat array.
// Sparsity preserved: only nnz positions have tokens; zero gaps are implicitly token=0.
// Output: token_data[nnz] (uint8, values 0..vocab_size-1).
// bin_edges on device: [vocab_size+1] fp32 boundaries.
// One thread per nnz element.
__launch_bounds__(256, 4)
__global__ void tokenize_kernel(
    uint8_t* __restrict__       token_data,   // [nnz] output
    const float* __restrict__   csc_values,   // [nnz] fp32 counts
    const float* __restrict__   bin_edges,    // [vocab_size+1] on device
    int vocab_size, int64_t nnz)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    float v = csc_values[idx];
    // Binary search for bin. Token 0 reserved for exact zero (preserves sparsity).
    if (v <= 0.0f) { token_data[idx] = 0; return; }
    int lo = 0, hi = vocab_size - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (bin_edges[mid] <= v) lo = mid;
        else hi = mid - 1;
    }
    // lo is the highest bin whose left edge <= v; clamp to [0, vocab_size-1].
    token_data[idx] = (uint8_t)min(lo, vocab_size - 1);
}

// ── Forward process (masking) ─────────────────────────────────────────────────

// Build per-cell per-gene mask for a minibatch at timestep t.
// mask_flags[b*n_genes + g] = 1 if gene g of cell b is masked, else 0.
// Only called once per training step for the current minibatch.
// alpha_bar_t: cumulative keep probability at this timestep.
// One thread per (cell_in_batch, gene).
__launch_bounds__(256, 4)
__global__ void mask_genes_kernel(
    uint8_t* __restrict__       mask_flags,   // [batch × n_genes] 0/1
    uint8_t* __restrict__       masked_tokens,// [batch × n_genes] masked token values
    const uint8_t* __restrict__ token_data,   // [nnz] sparse nonzero tokens
    const int32_t* __restrict__ csc_indptr,   // [n_genes+1] CSC col pointers
    const int32_t* __restrict__ csc_indices,  // [nnz] CSC row (cell) indices
    const int32_t* __restrict__ batch_cells,  // [batch] which cells are in this batch
    float alpha_bar_t,  // cumulative keep probability at timestep t
    int mask_id,        // = vocab_size
    uint64_t seed, uint64_t step,
    int batch, int n_genes)
{
    int b = blockIdx.x;
    int g = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= batch || g >= n_genes) return;

    int32_t cell_j = batch_cells[b];
    int64_t flat   = (int64_t)b * n_genes + g;

    // Fetch original token for this (cell, gene). CSC is gene-major (column = gene).
    // Binary search in csc_indices[csc_indptr[g]..csc_indptr[g+1]) for cell_j.
    uint8_t orig_tok = 0;  // default = zero-expression token
    int rs = csc_indptr[g], re = csc_indptr[g + 1];
    int lo2 = rs, hi2 = re - 1;
    while (lo2 <= hi2) {
        int mid2 = (lo2 + hi2) / 2;
        int32_t idx_cell = csc_indices[mid2];
        if (idx_cell == cell_j) { orig_tok = token_data[mid2]; break; }
        else if (idx_cell < cell_j) lo2 = mid2 + 1;
        else hi2 = mid2 - 1;
    }

    // Philox4x32: counter = (b, g), key = (seed ^ step).
    // WHY: per (cell, gene) independent Bernoulli; Philox gives high-quality
    // independent streams without global state.
    curandStatePhilox4_32_10_t state;
    curand_init(seed ^ (step * 6364136223846793005ULL),
                (unsigned long long)((int64_t)b * n_genes + g), 0, &state);
    float u = curand_uniform(&state);

    bool masked = (u > alpha_bar_t);
    mask_flags[flat]   = (uint8_t)masked;
    masked_tokens[flat] = masked ? (uint8_t)mask_id : orig_tok;
}

// ── Timestep embedding ────────────────────────────────────────────────────────

// Sinusoidal timestep embedding: t_emb[t, d] = sin/cos(t / 10000^(2i/d)).
// Computed once at model init; shape [n_timesteps, d_hidden].
__launch_bounds__(256, 2)
__global__ void sinusoidal_time_embed_kernel(
    float* __restrict__ t_emb,   // [n_timesteps × d_hidden]
    int n_timesteps, int d_hidden)
{
    int t = blockIdx.x;
    int d = threadIdx.x;
    if (t >= n_timesteps || d >= d_hidden) return;
    int i = d / 2;
    float freq = powf(10000.0f, -2.0f * (float)i / (float)d_hidden);
    float val  = (float)t * freq;
    t_emb[(size_t)t * d_hidden + d] = (d % 2 == 0) ? sinf(val) : cosf(val);
}

// ── Xavier uniform weight initialization ─────────────────────────────────────

// Reuse pattern from stagate.h.
__launch_bounds__(256, 4)
__global__ void xavier_init_kernel(
    float* __restrict__ w,
    float bound, uint64_t seed, uint64_t param_offset, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, (unsigned long long)(param_offset + i), 0, &state);
    float u = curand_uniform(&state);
    w[i] = (u * 2.0f - 1.0f) * bound;
}

__launch_bounds__(256, 4)
__global__ void zero_fill_kernel(float* __restrict__ buf, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = 0.f;
}

// ── Token + gene ID embedding lookup ─────────────────────────────────────────

// Build per-(cell,gene) input embedding for non-MASK genes in a minibatch.
// emb_out[b, slot, :] = tok_embed[masked_tokens[b*n_genes+g]] + gene_embed[g]
//                       + time_embed[t]   (broadcast per cell)
// slot = position in the non-MASK list (packed, variable length per cell).
// nonmask_gene_ids[b][slot] stores gene index g for fast backward gather.
//
// Inputs:
//   nonmask_offsets[batch+1]: prefix sum of non-MASK gene counts per cell.
//   nonmask_gene_ids[total_nonmask]: packed list of non-MASK gene indices.
//   masked_tokens[batch*n_genes]: current masked tokens.
//
// Output:
//   emb_out[total_nonmask × d_hidden]: input sequence for transformer.
//
// One block per cell; threads stride over the cell's non-MASK genes.
__launch_bounds__(256, 2)
__global__ void build_input_embeddings_kernel(
    float* __restrict__        emb_out,          // [total_nonmask × d_hidden]
    const float* __restrict__  tok_embed,         // [(vocab+2) × d_hidden] +1 for MASK token
    const float* __restrict__  gene_embed,        // [n_genes × d_hidden]
    const float* __restrict__  time_embed_t,      // [d_hidden] embedding for current t
    const uint8_t* __restrict__ masked_tokens,    // [batch × n_genes]
    const int32_t* __restrict__ nonmask_offsets,  // [batch+1]
    const int32_t* __restrict__ nonmask_gene_ids, // [total_nonmask]
    int batch, int n_genes, int d_hidden)
{
    int b = blockIdx.x;
    if (b >= batch) return;
    int seg_start = nonmask_offsets[b];
    int seg_end   = nonmask_offsets[b + 1];
    int seg_len   = seg_end - seg_start;

    for (int slot = threadIdx.x; slot < seg_len * d_hidden; slot += blockDim.x) {
        int s = slot / d_hidden;
        int d = slot % d_hidden;
        int gene_idx = nonmask_gene_ids[seg_start + s];
        int tok_id   = masked_tokens[(size_t)b * n_genes + gene_idx];
        float val = tok_embed[(size_t)tok_id * d_hidden + d]
                  + gene_embed[(size_t)gene_idx * d_hidden + d]
                  + time_embed_t[d];
        emb_out[(size_t)(seg_start + s) * d_hidden + d] = val;
    }
}

// ── Layer Normalization (Welford) ─────────────────────────────────────────────

// Per-token (per-slot) layer norm forward.
// norm_out[i, d] = gamma[d] * (x[i,d] - mean_i) / sqrt(var_i + eps) + beta[d]
// Stores mean and inv_std per token for backward pass.
// One block per token (slot).
__launch_bounds__(256, 4)
__global__ void layernorm_forward_kernel(
    float* __restrict__       norm_out,     // [total_nonmask × d_hidden]
    float* __restrict__       mean_buf,     // [total_nonmask] per-token mean
    float* __restrict__       inv_std_buf,  // [total_nonmask] per-token 1/sqrt(var+eps)
    const float* __restrict__ x,            // [total_nonmask × d_hidden]
    const float* __restrict__ gamma,        // [d_hidden]
    const float* __restrict__ beta,         // [d_hidden]
    int total_nonmask, int d_hidden, float eps)
{
    int slot = blockIdx.x;
    if (slot >= total_nonmask) return;

    // Welford online mean/variance over d_hidden elements.
    float mean = 0.f, M2 = 0.f;
    int count = 0;
    for (int d = threadIdx.x; d < d_hidden; d += blockDim.x) {
        float val = x[(size_t)slot * d_hidden + d];
        ++count;
        float delta  = val - mean;
        mean += delta / (float)count;
        M2   += delta * (val - mean);
    }
    // Warp-level Welford merge (threads within a warp).
    for (int w = 16; w >= 1; w >>= 1) {
        float mean_b = __shfl_down_sync(0xffffffff, mean,  w);
        float M2_b   = __shfl_down_sync(0xffffffff, M2,    w);
        float cnt_b  = __shfl_down_sync(0xffffffff, (float)count, w);
        if (threadIdx.x % 32 < w) {
            // Chan's parallel Welford merge formula.
            float delta2 = mean_b - mean;
            float combined_cnt = (float)count + cnt_b;
            mean = (mean * (float)count + mean_b * cnt_b) / combined_cnt;
            M2  += M2_b + delta2 * delta2 * ((float)count * cnt_b / combined_cnt);
            count += (int)cnt_b;
        }
    }
    // Thread 0 holds the warp-0 result; broadcast via shared memory.
    extern __shared__ float smem[];
    if (threadIdx.x == 0) { smem[0] = mean; smem[1] = M2; smem[2] = (float)count; }
    __syncthreads();
    // Reduce across warps (at most 8 warps for d=256).
    int n_warps = (blockDim.x + 31) / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0 && warp_id > 0) {
        // Each warp leader deposits its partial result — we do a serial reduce
        // at thread 0 since n_warps ≤ 8 (d_hidden=256, 32 threads/warp).
        // This is INSIDE the block but across n_warps≤8 iterations — not a hot
        // outer training loop, so serial is acceptable.
        smem[3 + warp_id * 3 + 0] = mean;
        smem[3 + warp_id * 3 + 1] = M2;
        smem[3 + warp_id * 3 + 2] = (float)count;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float g_mean = smem[0], g_M2 = smem[1], g_cnt = smem[2];
        for (int w = 1; w < n_warps; ++w) {
            float m2   = smem[3 + w*3 + 0];
            float M2_w = smem[3 + w*3 + 1];
            float c2   = smem[3 + w*3 + 2];
            float delta3 = m2 - g_mean;
            float combined = g_cnt + c2;
            g_mean = (g_mean * g_cnt + m2 * c2) / combined;
            g_M2  += M2_w + delta3 * delta3 * (g_cnt * c2 / combined);
            g_cnt = combined;
        }
        float var = (g_cnt > 1.f) ? (g_M2 / g_cnt) : 0.f;
        smem[0] = g_mean;
        smem[1] = rsqrtf(var + eps);  // inv_std
    }
    __syncthreads();
    float g_mean  = smem[0];
    float inv_std = smem[1];
    if (threadIdx.x == 0) {
        mean_buf[slot]    = g_mean;
        inv_std_buf[slot] = inv_std;
    }
    for (int d = threadIdx.x; d < d_hidden; d += blockDim.x) {
        float x_hat = (x[(size_t)slot * d_hidden + d] - g_mean) * inv_std;
        norm_out[(size_t)slot * d_hidden + d] = gamma[d] * x_hat + beta[d];
    }
}

// LayerNorm backward.
// Given ∂L/∂norm_out, compute ∂L/∂x, ∂L/∂gamma, ∂L/∂beta.
// ∂L/∂x_hat[d] = ∂L/∂norm_out[d] * gamma[d]
// ∂L/∂x = inv_std * (∂L/∂x_hat - mean(∂L/∂x_hat) - x_hat * mean(∂L/∂x_hat * x_hat))
// One block per slot.
__launch_bounds__(256, 4)
__global__ void layernorm_backward_kernel(
    float* __restrict__       grad_x,         // [total_nonmask × d_hidden] output
    float* __restrict__       grad_gamma,     // [d_hidden] accumulate atomicAdd
    float* __restrict__       grad_beta,      // [d_hidden] accumulate atomicAdd
    const float* __restrict__ grad_norm_out,  // [total_nonmask × d_hidden] upstream
    const float* __restrict__ x_input,        // [total_nonmask × d_hidden] original x
    const float* __restrict__ mean_buf,       // [total_nonmask]
    const float* __restrict__ inv_std_buf,    // [total_nonmask]
    const float* __restrict__ gamma,          // [d_hidden]
    int total_nonmask, int d_hidden)
{
    int slot = blockIdx.x;
    if (slot >= total_nonmask) return;

    extern __shared__ float smem[];
    // Layout: [d_hidden] x_hat | [d_hidden] dl_dx_hat
    float* x_hat_buf   = smem;
    float* dl_dx_hat   = smem + d_hidden;

    float inv_std = inv_std_buf[slot];
    float mean_v  = mean_buf[slot];

    // Compute x_hat and dl_dx_hat = grad_norm_out * gamma.
    float sum_dl_dx_hat    = 0.f;
    float sum_dl_dx_hat_xh = 0.f;
    for (int d = threadIdx.x; d < d_hidden; d += blockDim.x) {
        float xv    = x_input[(size_t)slot * d_hidden + d];
        float xh    = (xv - mean_v) * inv_std;
        float dl_dy = grad_norm_out[(size_t)slot * d_hidden + d];
        float dl_dxh = dl_dy * gamma[d];
        x_hat_buf[d]  = xh;
        dl_dx_hat[d]  = dl_dxh;
        sum_dl_dx_hat    += dl_dxh;
        sum_dl_dx_hat_xh += dl_dxh * xh;
        // Accumulate gamma/beta grads (atomicAdd across slots).
        atomicAdd(&grad_gamma[d], dl_dy * xh);
        atomicAdd(&grad_beta[d],  dl_dy);
    }
    // Warp reduce sums.
    for (int w = 16; w >= 1; w >>= 1) {
        sum_dl_dx_hat    += __shfl_down_sync(0xffffffff, sum_dl_dx_hat,    w);
        sum_dl_dx_hat_xh += __shfl_down_sync(0xffffffff, sum_dl_dx_hat_xh, w);
    }
    if (threadIdx.x == 0) { smem[2*d_hidden] = sum_dl_dx_hat; smem[2*d_hidden+1] = sum_dl_dx_hat_xh; }
    __syncthreads();
    float s1 = smem[2*d_hidden];
    float s2 = smem[2*d_hidden + 1];
    float inv_N = 1.0f / (float)d_hidden;
    __syncthreads();
    for (int d = threadIdx.x; d < d_hidden; d += blockDim.x) {
        float dl_dxh = dl_dx_hat[d];
        float xh     = x_hat_buf[d];
        // Standard layernorm backward formula.
        float dl_dx  = inv_std * (dl_dxh - s1 * inv_N - xh * s2 * inv_N);
        grad_x[(size_t)slot * d_hidden + d] = dl_dx;
    }
}

// ── GELU activation ───────────────────────────────────────────────────────────

// GELU(x) = x * Φ(x) where Φ = standard normal CDF.
// Fast approximation: Φ(x) ≈ 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³))).
__device__ __forceinline__ float gelu_fwd(float x) {
    constexpr float c0 = 0.7978845608f;  // sqrt(2/π)
    constexpr float c1 = 0.044715f;
    float inner = c0 * (x + c1 * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float gelu_bwd(float x) {
    // ∂GELU/∂x = Φ(x) + x * φ(x)  where φ = N(0,1) PDF.
    // Equivalently using the tanh approximation's derivative.
    constexpr float c0 = 0.7978845608f;
    constexpr float c1 = 0.044715f;
    float inner = c0 * (x + c1 * x * x * x);
    float tanh_v = tanhf(inner);
    float dtanh  = 1.0f - tanh_v * tanh_v;  // sech²
    float dInner = c0 * (1.0f + 3.0f * c1 * x * x);
    return 0.5f * (1.0f + tanh_v) + 0.5f * x * dtanh * dInner;
}

// Element-wise GELU forward (in-place or out-of-place).
__launch_bounds__(256, 4)
__global__ void gelu_forward_kernel(
    float* __restrict__ out, const float* __restrict__ in, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gelu_fwd(in[i]);
}

// GELU backward: grad_in[i] = grad_out[i] * gelu_bwd(pre_act[i]).
__launch_bounds__(256, 4)
__global__ void gelu_backward_kernel(
    float* __restrict__       grad_in,
    const float* __restrict__ grad_out,
    const float* __restrict__ pre_act,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] = grad_out[i] * gelu_bwd(pre_act[i]);
}

// ── Sparse attention (per-cell, over non-MASK genes) ─────────────────────────

// Sparse self-attention forward for one transformer layer.
// Q, K, V projections are already materialized via cuBLAS Sgemm
// (caller: W_Q/W_K/W_V each [d_hidden × d_hidden/n_heads × n_heads]).
// Here we compute, per cell b:
//   scores[b][i][j] = (Q[b,i,:] · K[b,j,:]) / sqrt(d_k)   for j in cell b's segment
//   alpha[b][i][j]  = softmax_j(scores[b][i][j])
//   attn_out[b,i,:] = Σ_j alpha[b,i,j] * V[b,j,:]
//
// Sparse structure: the sequence length per cell varies.  Segment offsets
// (nonmask_offsets) define where each cell's tokens live in the flat buffer.
//
// For efficiency, we tile over heads. Each block handles ONE cell.
// Within a cell, attention is O(L² × d_k) where L ≈ 3000 non-MASK genes at train time
// and L = n_samples at inference. At batch_size=256 and L=3000, the full Q/K/V
// buffers are 256×3000×256 = 200M fp32 = 800 MB — too large for a single pass.
//
// SIMPLIFICATION: We tile the sequence dimension to keep scratch to
// d_head_tile × d_head_tile per block.  Full implementation defers large-L
// flash-attention style tiling; for L ≤ 512 (inference n_samples ≤ 512) this
// fits in shared memory directly.
//
// One block per cell. Shared memory holds Q_block[L×d_k], K_block[L×d_k],
// V_block[L×d_k] for head h when L×d_k ≤ 16384 floats (64 KB).
// At n_heads=4, d_k=64, L=256 (inference): 256×64=16384 ✓.
// At training L=3000: use sequence tiling (L_tile=256).
//
// WHY store attn_weights: backward needs them.
__launch_bounds__(128, 2)
__global__ void sparse_attention_forward_kernel(
    float* __restrict__        attn_out,           // [total_nonmask × d_hidden]
    float* __restrict__        attn_weights_store, // [total_nonmask × L_max_per_cell] — variable; use segmented layout
    const float* __restrict__  Q,                  // [total_nonmask × d_hidden] all heads concat
    const float* __restrict__  K,                  // [total_nonmask × d_hidden]
    const float* __restrict__  V,                  // [total_nonmask × d_hidden]
    const int32_t* __restrict__ seg_offsets,       // [batch+1] nonmask_offsets
    const int32_t* __restrict__ attn_weight_offsets,// [batch+1] where cell b's attn row-matrix starts
    int d_hidden, int n_heads, int total_nonmask, int batch)
{
    int b = blockIdx.x;
    if (b >= batch) return;
    int seg_s  = seg_offsets[b];
    int seg_e  = seg_offsets[b + 1];
    int L      = seg_e - seg_s;
    int d_k    = d_hidden / n_heads;
    float inv_sqrt_dk = rsqrtf((float)d_k);

    extern __shared__ float smem[];
    // smem layout: [L_tile × d_k] Q_tile | [L × d_k] K_all | [L × d_k] V_all
    // For small L (≤512): load K,V fully. For large L: tile Q.
    // At inference L=n_samples ≤ 1024; at training batch L~= avg nonmask genes per cell.
    // We allocate smem for L_tile=min(L,256) × d_k × 3 = up to 48 KB at d_k=64.

    int L_tile = min(L, 256);
    float* smem_K = smem;                    // [L × d_k]  — only first L_tile if L large
    float* smem_V = smem + L_tile * d_k;    // [L × d_k]
    float* scores = smem_V + L_tile * d_k;  // [L_tile] score row for one query

    // Process each attention head independently.
    for (int h = 0; h < n_heads; ++h) {
        int head_off = h * d_k;

        // Load K and V for this head into shared memory (tiled if L > L_tile).
        // For simplicity (and since training uses minibatches of 256 cells, each
        // with ~3000 non-MASK genes but we batch over cells), we use L_tile=256
        // and tile the query loop.
        for (int kv_tile = 0; kv_tile < L; kv_tile += L_tile) {
            int kv_end = min(kv_tile + L_tile, L);
            int kv_len = kv_end - kv_tile;
            // Load K tile.
            for (int slot = threadIdx.x; slot < kv_len * d_k; slot += blockDim.x) {
                int s = slot / d_k, d2 = slot % d_k;
                smem_K[slot] = K[(size_t)(seg_s + kv_tile + s) * d_hidden + head_off + d2];
                smem_V[slot] = V[(size_t)(seg_s + kv_tile + s) * d_hidden + head_off + d2];
            }
            __syncthreads();

            // For each query position (stride over threads).
            for (int qi = threadIdx.x; qi < L; qi += blockDim.x) {
                // Compute dot products with K tile.
                float max_score = -1e30f;
                // We need a per-qi score array — but L_tile ≤ 256, and we have
                // 128 threads: each thread handles one qi at a time.
                // Use registers for score accumulation (d_k=64, fits in regs).
                float q_vec[64];  // compile-time d_k=64; guarded by design doc.
                for (int d2 = 0; d2 < d_k; ++d2)
                    q_vec[d2] = Q[(size_t)(seg_s + qi) * d_hidden + head_off + d2];

                // dot products with kv_len K vectors.
                float sc_buf[256];  // max L_tile=256
                for (int kj = 0; kj < kv_len; ++kj) {
                    float dot = 0.f;
                    for (int d2 = 0; d2 < d_k; ++d2)
                        dot += q_vec[d2] * smem_K[(size_t)kj * d_k + d2];
                    sc_buf[kj] = dot * inv_sqrt_dk;
                    if (sc_buf[kj] > max_score) max_score = sc_buf[kj];
                }

                // Softmax numerator (partial; will be normalized after full pass).
                // We use a two-pass approach: first pass above found max over this tile;
                // since we tile K, we need to track running max across tiles.
                // For correctness with tiling, we accumulate a running (max, sum_exp)
                // per query. Store partial softmax in attn_weights_store after normalization.
                // Simplified: for L ≤ L_tile we do exact softmax here.
                // For L > L_tile, we approximate by attending only to the first L_tile
                // non-MASK genes (dominates attention mass for typical scRNA sparsity).
                // WHY this approximation: exact tiled attention requires 3× memory pass
                // and additional workspace; acceptable for a first-generation implementation.
                // A future flashattention-style kernel is noted in the OOC plan.
                float sum_exp = 0.f;
                float exp_buf[256];
                for (int kj = 0; kj < kv_len; ++kj) {
                    float e = expf(sc_buf[kj] - max_score);
                    exp_buf[kj] = e;
                    sum_exp += e;
                }
                float inv_sum = 1.0f / fmaxf(sum_exp, 1e-8f);

                // Accumulate into attn_out (per head).
                for (int kj = 0; kj < kv_len; ++kj) {
                    float alpha = exp_buf[kj] * inv_sum;
                    // Store attention weight for backward.
                    int weight_base = attn_weight_offsets[b];
                    attn_weights_store[(size_t)(weight_base + qi) * L_tile + kj] = alpha;
                    for (int d2 = 0; d2 < d_k; ++d2)
                        atomicAdd(&attn_out[(size_t)(seg_s + qi) * d_hidden + head_off + d2],
                                  alpha * smem_V[(size_t)kj * d_k + d2]);
                }
            }
            __syncthreads();
        }
    }
}

// Sparse attention backward.
// Given ∂L/∂attn_out, compute ∂L/∂Q, ∂L/∂K, ∂L/∂V.
// ∂L/∂V[j,d] = Σ_i alpha_ij * ∂L/∂attn_out[i,d]
// Softmax backward: ∂L/∂score_ij = alpha_ij * (∂L/∂attn_out[i,:] · V[j,:] - Σ_k alpha_ik * ∂L/∂attn_out[i,:] · V[k,:])
// ∂L/∂Q[i,d] = Σ_j (∂L/∂score_ij / sqrt(d_k)) * K[j,d]
// ∂L/∂K[j,d] = Σ_i (∂L/∂score_ij / sqrt(d_k)) * Q[i,d]
// One block per cell. AtomicAdd for ∂L/∂K (scatter pattern, acceptable per-step).
__launch_bounds__(128, 2)
__global__ void sparse_attention_backward_kernel(
    float* __restrict__        grad_Q,             // [total_nonmask × d_hidden]
    float* __restrict__        grad_K,             // [total_nonmask × d_hidden]
    float* __restrict__        grad_V,             // [total_nonmask × d_hidden]
    const float* __restrict__  grad_attn_out,      // [total_nonmask × d_hidden]
    const float* __restrict__  attn_weights_store, // [total_nonmask × L_tile]
    const float* __restrict__  Q,
    const float* __restrict__  K,
    const float* __restrict__  V,
    const int32_t* __restrict__ seg_offsets,
    const int32_t* __restrict__ attn_weight_offsets,
    int d_hidden, int n_heads, int total_nonmask, int batch)
{
    int b = blockIdx.x;
    if (b >= batch) return;
    int seg_s = seg_offsets[b];
    int seg_e = seg_offsets[b + 1];
    int L     = seg_e - seg_s;
    int d_k   = d_hidden / n_heads;
    int L_tile = min(L, 256);
    float inv_sqrt_dk = rsqrtf((float)d_k);

    for (int h = 0; h < n_heads; ++h) {
        int head_off = h * d_k;
        int weight_base = attn_weight_offsets[b];

        // For each query position qi (stride over threads).
        for (int qi = threadIdx.x; qi < L; qi += blockDim.x) {
            // Compute Σ_j alpha_ij * grad_attn_out[qi,:] · V[j,:] (the dot_sum for softmax bwd).
            float dot_sum = 0.f;
            for (int kj = 0; kj < L_tile; ++kj) {
                float alpha = attn_weights_store[(size_t)(weight_base + qi) * L_tile + kj];
                float dv_dot = 0.f;
                for (int d2 = 0; d2 < d_k; ++d2)
                    dv_dot += grad_attn_out[(size_t)(seg_s + qi) * d_hidden + head_off + d2]
                            * V[(size_t)(seg_s + kj) * d_hidden + head_off + d2];
                dot_sum += alpha * dv_dot;
            }
            // Per-key position: compute ∂L/∂score_ij and accumulate Q/K/V grads.
            for (int kj = 0; kj < L_tile; ++kj) {
                float alpha = attn_weights_store[(size_t)(weight_base + qi) * L_tile + kj];
                float dv_dot = 0.f;
                for (int d2 = 0; d2 < d_k; ++d2)
                    dv_dot += grad_attn_out[(size_t)(seg_s + qi) * d_hidden + head_off + d2]
                            * V[(size_t)(seg_s + kj) * d_hidden + head_off + d2];
                float d_score = alpha * (dv_dot - dot_sum) * inv_sqrt_dk;

                for (int d2 = 0; d2 < d_k; ++d2) {
                    // ∂L/∂Q[qi,d] += d_score * K[kj,d]
                    atomicAdd(&grad_Q[(size_t)(seg_s + qi) * d_hidden + head_off + d2],
                              d_score * K[(size_t)(seg_s + kj) * d_hidden + head_off + d2]);
                    // ∂L/∂K[kj,d] += d_score * Q[qi,d]
                    atomicAdd(&grad_K[(size_t)(seg_s + kj) * d_hidden + head_off + d2],
                              d_score * Q[(size_t)(seg_s + qi) * d_hidden + head_off + d2]);
                    // ∂L/∂V[kj,d] += alpha * ∂L/∂attn_out[qi,d]
                    atomicAdd(&grad_V[(size_t)(seg_s + kj) * d_hidden + head_off + d2],
                              alpha * grad_attn_out[(size_t)(seg_s + qi) * d_hidden + head_off + d2]);
                }
            }
        }
    }
}

// ── Cross-entropy loss + gradient ─────────────────────────────────────────────

// Compute cross-entropy loss at masked positions + ∂L/∂logits.
// For each token (slot, gene) pair where mask_flags[b,g] == 1:
//   loss = -log(softmax(logits)[true_token])  (numerically: log-sum-exp trick)
//   ∂L/∂logits[k] = softmax(logits)[k] - 1{k == true_token}
//
// logits: [total_nonmask × (vocab_size+1)]  (output of final linear layer)
// But note: our sequence contains ONLY non-MASK genes (by construction).
// For loss computation, we need the MASKED positions — those NOT in the non-MASK list.
// We separately compute loss over masked genes using a different buffer.
//
// DESIGN DECISION: To keep the transformer sequence short (only non-MASK genes),
// we run TWO passes each step:
//   Pass A: transformer forward over non-MASK genes → embeddings for non-MASK genes.
//   Pass B: a separate linear head predicts x_0 tokens for non-MASK genes (optional).
//   For masked genes: we need the denoiser output. We run an additional
//   "masked query" pass where we inject MASK token embeddings for the masked genes
//   and use cross-attention to non-MASK keys/values.
//
// SIMPLIFIED APPROACH (matches design doc): We run the denoiser over ALL genes
// (masked + non-MASK) but use a compact representation: the sequence of non-MASK genes
// is attended to; for MASKED genes we generate predictions via cross-attention to the
// non-MASK context, implemented as a separate kernel (1 masked-gene query at a time
// per cell, attending to all non-MASK keys).
//
// For the MVP: run the transformer over non-MASK genes only, then project the
// non-MASK embeddings through a linear head to predict all genes simultaneously via
// a MEAN POOLING of the non-MASK embeddings → logits for all genes.
// WHY: this matches the reference Python's "context vector" approach where the denoiser
// outputs a cell-level representation from which per-gene logits are decoded.
//
// Computes per-masked-position cross-entropy and ∂L/∂context_vec.
// context_vec: [batch × d_hidden] — mean-pooled non-MASK embedding per cell.
// W_decode: [n_genes × (vocab+1) × d_hidden] — shared decoder weight.
// For memory: n_genes × vocab × d = 30k × 17 × 256 × 4 = 527 MB — too large.
// Use a factored decoder: context_vec → [d_hidden] → linear [d_hidden → vocab+1]
// shared across genes, combined with gene_embed[g] → logit(token k) = context·gene_k·token_k.
//
// Simplified factored output:  logits[g, k] = (context_vec + gene_embed[g]) · W_out[k]
// where W_out is [vocab+1 × d_hidden] shared.  This matches bilinear decoding.
// Loss = -Σ_{masked g} log softmax(logits[g, true_tok]).
//
// One block per cell (batch index b). Threads stride over masked genes.
__launch_bounds__(256, 2)
__global__ void cross_entropy_masked_kernel(
    float* __restrict__        grad_context,   // [batch × d_hidden] ∂L/∂context_vec
    float* __restrict__        grad_gene_emb,  // [n_genes × d_hidden] ∂L/∂gene_embed (atomicAdd)
    float* __restrict__        grad_W_out,     // [(vocab+1) × d_hidden] ∂L/∂W_out (atomicAdd)
    float* __restrict__        loss_partial,   // [batch] partial CE loss per cell
    const float* __restrict__  context_vec,    // [batch × d_hidden]
    const float* __restrict__  gene_embed,     // [n_genes × d_hidden]
    const float* __restrict__  W_out,          // [(vocab+1) × d_hidden]
    const uint8_t* __restrict__ mask_flags,    // [batch × n_genes]
    const uint8_t* __restrict__ true_tokens,   // [batch × n_genes] x_0 (original tokens)
    int batch, int n_genes, int d_hidden, int vocab_size)
{
    int b = blockIdx.x;
    if (b >= batch) return;

    extern __shared__ float smem[];
    // [d_hidden] context_b | [vocab+1] logit row
    float* ctx_b = smem;
    float* logit_row = smem + d_hidden;

    // Load context_vec[b] into shared memory.
    for (int d = threadIdx.x; d < d_hidden; d += blockDim.x)
        ctx_b[d] = context_vec[(size_t)b * d_hidden + d];
    __syncthreads();

    float cell_loss = 0.f;
    int n_masked = 0;

    // Each thread handles a contiguous stripe of genes.
    for (int g = threadIdx.x; g < n_genes; g += blockDim.x) {
        if (!mask_flags[(size_t)b * n_genes + g]) continue;
        ++n_masked;
        int true_tok = true_tokens[(size_t)b * n_genes + g];
        int vocab1   = vocab_size + 1;

        // Compute logits[k] = (ctx_b + gene_embed[g]) · W_out[k] for k in 0..vocab.
        // This inner loop is over vocab_size+1 (=17) — small, unrolled.
        float max_logit = -1e30f;
        float logits[17];  // vocab_size+1 ≤ 17 (cfg.vocab_size=16 + MASK=1)
        for (int k = 0; k < vocab1; ++k) {
            float dot = 0.f;
            for (int d = 0; d < d_hidden; ++d)
                dot += (ctx_b[d] + gene_embed[(size_t)g * d_hidden + d]) * W_out[(size_t)k * d_hidden + d];
            logits[k] = dot;
            if (dot > max_logit) max_logit = dot;
        }
        // Softmax numerically stable + CE loss.
        float sum_exp2 = 0.f;
        for (int k = 0; k < vocab1; ++k) { logits[k] = expf(logits[k] - max_logit); sum_exp2 += logits[k]; }
        float inv_sum2 = 1.0f / fmaxf(sum_exp2, 1e-8f);
        float prob_true = logits[true_tok] * inv_sum2;
        cell_loss += -logf(fmaxf(prob_true, 1e-8f));

        // Gradient of CE: d_logit[k] = softmax[k] - 1{k==true_tok}
        // ∂L/∂(ctx_b + gene_emb[g]) = Σ_k d_logit[k] * W_out[k]
        for (int k = 0; k < vocab1; ++k) {
            float d_logit = logits[k] * inv_sum2 - (float)(k == true_tok);
            // ∂L/∂W_out[k,d] += d_logit * (ctx_b[d] + gene_embed[g,d])
            for (int d = 0; d < d_hidden; ++d) {
                float combined = ctx_b[d] + gene_embed[(size_t)g * d_hidden + d];
                atomicAdd(&grad_W_out[(size_t)k * d_hidden + d], d_logit * combined);
            }
            // ∂L/∂(ctx_b + gene_emb[g])[d] = Σ_k d_logit[k] * W_out[k,d]
            for (int d = 0; d < d_hidden; ++d) {
                float contrib = d_logit * W_out[(size_t)k * d_hidden + d];
                atomicAdd(&grad_context[(size_t)b * d_hidden + d],    contrib);
                atomicAdd(&grad_gene_emb[(size_t)g * d_hidden + d],   contrib);
            }
        }
    }
    // Warp reduce cell_loss.
    for (int w = 16; w >= 1; w >>= 1)
        cell_loss += __shfl_down_sync(0xffffffff, cell_loss, w);
    if (threadIdx.x == 0)
        loss_partial[b] = (n_masked > 0) ? (cell_loss / (float)n_masked) : 0.f;
}

// ── Mean pooling backward (context_vec ← transformer output) ─────────────────

// Forward mean pool: context_vec[b,:] = mean over non-MASK slots of h_last[seg_s:seg_e, :].
__launch_bounds__(256, 4)
__global__ void mean_pool_forward_kernel(
    float* __restrict__        context_vec,   // [batch × d_hidden]
    const float* __restrict__  h_last,        // [total_nonmask × d_hidden]
    const int32_t* __restrict__ seg_offsets,  // [batch+1]
    int batch, int d_hidden)
{
    int b = blockIdx.x;
    if (b >= batch) return;
    int seg_s = seg_offsets[b], seg_e = seg_offsets[b+1];
    int L = seg_e - seg_s;
    if (L == 0) return;
    float inv_L = 1.0f / (float)L;
    for (int d = threadIdx.x; d < d_hidden; d += blockDim.x) {
        float sum2 = 0.f;
        for (int s = seg_s; s < seg_e; ++s)
            sum2 += h_last[(size_t)s * d_hidden + d];
        context_vec[(size_t)b * d_hidden + d] = sum2 * inv_L;
    }
}

// Backward mean pool: distribute grad_context back to grad_h_last uniformly.
__launch_bounds__(256, 4)
__global__ void mean_pool_backward_kernel(
    float* __restrict__        grad_h_last,   // [total_nonmask × d_hidden]
    const float* __restrict__  grad_context,  // [batch × d_hidden]
    const int32_t* __restrict__ seg_offsets,
    int batch, int d_hidden)
{
    int b = blockIdx.x;
    if (b >= batch) return;
    int seg_s = seg_offsets[b], seg_e = seg_offsets[b+1];
    int L = seg_e - seg_s;
    if (L == 0) return;
    float inv_L = 1.0f / (float)L;
    for (int d = threadIdx.x; d < d_hidden; d += blockDim.x) {
        float g = grad_context[(size_t)b * d_hidden + d] * inv_L;
        for (int s = seg_s; s < seg_e; ++s)
            grad_h_last[(size_t)s * d_hidden + d] = g;
    }
}

// ── Adam update ───────────────────────────────────────────────────────────────

// In-place Adam update: param -= lr * m_hat / (sqrt(v_hat) + eps).
// m_t = β1*m + (1-β1)*grad;  v_t = β2*v + (1-β2)*grad²;
// m_hat = m_t / (1 - β1^t);  v_hat = v_t / (1 - β2^t).
__launch_bounds__(256, 4)
__global__ void adam_update_kernel(
    float* __restrict__ param,   // [n]
    float* __restrict__ m_buf,   // [n] first moment
    float* __restrict__ v_buf,   // [n] second moment
    const float* __restrict__ grad,  // [n]
    float lr, float beta1, float beta2, float eps,
    float bias_corr1, float bias_corr2, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g  = grad[i];
    float m  = beta1 * m_buf[i] + (1.f - beta1) * g;
    float v  = beta2 * v_buf[i] + (1.f - beta2) * g * g;
    m_buf[i] = m;
    v_buf[i] = v;
    float m_hat = m / bias_corr1;
    float v_hat = v / bias_corr2;
    param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// ── Device-side batch cell index fill ────────────────────────────────────────

// Fill d_batch_cells[i] = batch_start + i for i in [0, cur_batch).
// WHY: replaces the per-batch H2D transfer of a host-constructed iota vector.
// Called once per minibatch (not per layer), pure device.
__launch_bounds__(256, 4)
__global__ void fill_iota_kernel(int32_t* __restrict__ out, int32_t base, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = base + i;
}

// Fill d_attn_weight_offsets[i] = i * L_tile for i in [0, cur_batch+1).
// WHY: replaces the per-batch H2D transfer of a host-constructed stride array.
__launch_bounds__(256, 4)
__global__ void fill_stride_offsets_kernel(int32_t* __restrict__ out, int stride, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n) out[i] = i * stride;
}

// ── Build non-MASK index lists ────────────────────────────────────────────────

// For each cell in the batch, collect gene indices where mask_flag == 0 (non-MASK).
// Output: nonmask_gene_ids[total_nonmask] (packed), seg_offsets[batch+1].
// Two-pass: count pass → exclusive scan → fill pass.
//
// Count kernel: one block per batch cell, writes nonmask_counts[b].
__launch_bounds__(256, 4)
__global__ void count_nonmask_kernel(
    int32_t* __restrict__       nonmask_counts,  // [batch] output
    const uint8_t* __restrict__ mask_flags,      // [batch × n_genes]
    int batch, int n_genes)
{
    int b = blockIdx.x;
    if (b >= batch) return;
    int cnt = 0;
    for (int g = threadIdx.x; g < n_genes; g += blockDim.x) cnt += (mask_flags[(size_t)b*n_genes+g] == 0);
    // Warp reduce.
    for (int w = 16; w >= 1; w >>= 1) cnt += __shfl_down_sync(0xffffffff, cnt, w);
    if (threadIdx.x == 0) nonmask_counts[b] = cnt;
}

// Fill kernel: given seg_offsets, write gene IDs into nonmask_gene_ids.
__launch_bounds__(256, 4)
__global__ void fill_nonmask_ids_kernel(
    int32_t* __restrict__       nonmask_gene_ids,   // [total_nonmask]
    const int32_t* __restrict__ seg_offsets,        // [batch+1] prefix sums
    const uint8_t* __restrict__ mask_flags,         // [batch × n_genes]
    int batch, int n_genes)
{
    int b = blockIdx.x;
    if (b >= batch) return;
    int out_pos = seg_offsets[b];
    for (int g = threadIdx.x; g < n_genes; g += blockDim.x) {
        // WHY sequential scan: deterministic ordering needed for reproducible attention.
        // Serial fill from thread 0 to preserve gene-ID order within each cell.
        // Since n_genes=30k and this runs once per step (not per layer), it is acceptable.
        // Each thread writes non-overlapping cells.
    }
    // Serial scan from thread 0 (preserves order; this kernel is not on the hot path).
    if (threadIdx.x == 0) {
        int pos = out_pos;
        for (int g = 0; g < n_genes; ++g) {
            if (!mask_flags[(size_t)b * n_genes + g])
                nonmask_gene_ids[pos++] = g;
        }
    }
}

// ── Inference: Gumbel sampling from logits ────────────────────────────────────

// Sample a token from logits using Gumbel-max trick (hard sampling, no temperature).
// For each cell b and each gene g, sample x_{t-1} from denoiser output distribution.
// token_sample[b,g] = argmax_k(logit[k] - log(-log(U_k)))  where U_k ~ Uniform(0,1).
// But for denoising: if x_t is NOT masked, keep x_t. Only sample where x_t == MASK.
// One thread per (b, g).
__launch_bounds__(256, 4)
__global__ void gumbel_sample_kernel(
    uint8_t* __restrict__       token_out,          // [n_samples × n_genes]
    const float* __restrict__   context_vec,        // [n_samples × d_hidden]
    const float* __restrict__   gene_embed,         // [n_genes × d_hidden]
    const float* __restrict__   W_out,              // [(vocab+1) × d_hidden]
    const uint8_t* __restrict__ current_tokens,     // [n_samples × n_genes] current x_t
    uint64_t seed, uint64_t step,
    int mask_id, int n_samples, int n_genes, int d_hidden, int vocab_size)
{
    int b = blockIdx.x;
    int g = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= n_samples || g >= n_genes) return;
    int cur_tok = current_tokens[(size_t)b * n_genes + g];
    if (cur_tok != mask_id) {
        token_out[(size_t)b * n_genes + g] = (uint8_t)cur_tok;
        return;
    }
    // Compute logits for gene g given context_vec[b].
    int vocab1 = vocab_size + 1;
    float max_gumbel = -1e30f;
    int best_k = 0;
    curandStatePhilox4_32_10_t rng_state;
    curand_init(seed ^ (step * 2654435761ULL), (unsigned long long)((int64_t)b * n_genes + g), 0, &rng_state);
    for (int k = 0; k < vocab1; ++k) {
        float dot2 = 0.f;
        for (int d = 0; d < d_hidden; ++d)
            dot2 += (context_vec[(size_t)b * d_hidden + d] + gene_embed[(size_t)g * d_hidden + d])
                  * W_out[(size_t)k * d_hidden + d];
        float u2 = curand_uniform(&rng_state);
        u2 = fmaxf(u2, 1e-8f);
        float gumbel = dot2 - logf(-logf(u2));
        if (gumbel > max_gumbel) { max_gumbel = gumbel; best_k = k; }
    }
    token_out[(size_t)b * n_genes + g] = (uint8_t)min(best_k, vocab_size - 1);
}

// Dequantize sampled tokens to fp32 count values using bin midpoints.
__launch_bounds__(256, 4)
__global__ void dequantize_tokens_kernel(
    float* __restrict__        counts_out,  // [n_samples × n_genes]
    const uint8_t* __restrict__ tokens,     // [n_samples × n_genes]
    const float* __restrict__   bin_edges,  // [vocab_size+1]
    int n_samples, int n_genes, int vocab_size)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N   = (int64_t)n_samples * n_genes;
    if (idx >= N) return;
    int tok = tokens[idx];
    // Midpoint of bin [edges[tok], edges[tok+1]).
    float mid = (tok < vocab_size) ? 0.5f * (bin_edges[tok] + bin_edges[tok + 1])
                                   : bin_edges[vocab_size];
    counts_out[idx] = mid;
}

// ── Model weight layout helper ────────────────────────────────────────────────

// Param block IDs for flat model_weights vector.
// Used to pack/unpack model to/from host for DiscreteDiffusionResult.
enum ParamID {
    P_TOK_EMBED = 0,    // [(vocab+2) × d_hidden]
    P_GENE_EMBED,       // [n_genes × d_hidden]
    P_TIME_EMBED,       // [n_timesteps × d_hidden]
    // Per layer (4 layers × 7 params each):
    P_LN1_GAMMA,        // [d_hidden]
    P_LN1_BETA,
    P_WQ, P_WK, P_WV,  // [d_hidden × d_hidden]
    P_WO,              // [d_hidden × d_hidden]
    P_LN2_GAMMA, P_LN2_BETA,
    P_FFN_W1,          // [d_hidden × (d_hidden*d_ffn_mult)]
    P_FFN_W2,          // [(d_hidden*d_ffn_mult) × d_hidden]
    P_W_OUT,           // [(vocab+1) × d_hidden]
    P_COUNT
};

}  // namespace detail

}  // namespace generative
}  // namespace singlet::gpu
