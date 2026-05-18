// SPDX-License-Identifier: MIT
// integrates: original (CPA — first GPU; GNN deferred to followup)
//
// singlet/gpu/perturbation/perturb_graph.h
//
// CYCLE-32 RETRY: Simplified CPA (Compositional Perturbation Autoencoder).
//
// SIMPLIFICATIONS vs design doc (loudly documented):
//   - GNN extension DEFERRED: cfg.use_gnn=true throws std::runtime_error.
//     Tag: CYCLE-32-FOLLOWUP-GNN-EXTENSION
//   - Dose modulator MLP skipped: dose is a scalar broadcast onto z_p.
//   - Batch effect handling skipped.
//   - LOC hard cap: 1500.
//
// Algorithm: Lotfollahi et al. "Predicting single-cell perturbation responses
//   for unseen drugs" Nat Methods 2023. CPA decomposes cell expression into
//   cell-state latent z_c (variational encoder MLP) + additive perturbation
//   latent z_p (embedding lookup), reconstructed through a mirror decoder MLP.
//   An adversarial discriminator with gradient reversal ensures z_c does not
//   encode perturbation identity.
//
// Architecture (CPA-only, simplified):
//   encoder:  n_genes → 256 → d_hid → d_lat×2 (mu + log_sigma; GELU)
//   z_c:      reparameterized: z_mu + eps * exp(z_logsig)
//   z_p:      lookup table n_perts × d_lat; dose scalar broadcast
//   z_total:  z_c + dose * z_p[pert_id]
//   decoder:  d_lat → d_hid → 256 → n_genes (GELU, softplus output)
//   adv disc: d_lat → 64 → n_perts (GRL in backward; CE loss)
//   loss:     NB_LL + kl_weight * KL + adversarial_weight * CE_adv
//
// Workspace at 100k cells × 5k genes × 20 perts, d_lat=64:
//   Encoder/decoder params + Adam: ~15 MB
//   Cell latents (result):         ~25 MB
//   Minibatch buffers:             ~10 MB (batch=256)
//   Total:                         ~50 MB << 200 MB target
//
// Streams:     1, caller-provided.
// Precision:   fp32 hot path; fp64 NB log-likelihood accumulator per element.
// Determinism: cuRAND Philox seeded by cfg.seed; host shuffle seeded by cfg.seed.
// OOC plan:    per-perturbation minibatch streaming. Model params stay on device;
//              input data chunked by batch_size. For datasets larger than device
//              memory, caller should stream .1pz shards via pz_device_loader and
//              call train_perturb_graph with successive CSC windows.

#pragma once

#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet::gpu {
namespace perturbation {

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct PerturbGraphConfig {
    int      d_latent           = 64;
    int      d_hidden           = 128;
    int      n_epochs           = 200;
    int      batch_size         = 256;
    float    learning_rate      = 1e-3f;
    float    kl_weight          = 0.01f;
    float    adversarial_weight = 1.0f;
    bool     use_gnn            = false;  // CYCLE 32: throws if true
    int      gnn_layers         = 2;      // reserved for CYCLE-32-FOLLOWUP-GNN-EXTENSION
    uint64_t seed               = 0;
};

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------
struct PerturbGraphResult {
    core::DeviceMemory<float> pert_embeddings;  // [n_perts × d_latent]
    core::DeviceMemory<float> cell_latents;     // [n_cells × d_latent] z_mu (inference)
    std::vector<float>        epoch_losses;     // [n_epochs] total loss per epoch
    std::vector<float>        adv_losses;       // [n_epochs] adversarial CE per epoch
    int n_cells  = 0;
    int n_genes  = 0;
    int n_perts  = 0;
    int d_latent = 0;
};

// ---------------------------------------------------------------------------
// MLP parameter block: up to 3 layers, all RAII device memory.
// ---------------------------------------------------------------------------
struct MLPParams {
    static constexpr int kMaxLayers = 3;
    core::DeviceMemory<float> W[kMaxLayers];   // [out × in] row-major
    core::DeviceMemory<float> b[kMaxLayers];   // [out]
    core::DeviceMemory<float> mW[kMaxLayers];  // Adam m for W
    core::DeviceMemory<float> vW[kMaxLayers];  // Adam v for W
    core::DeviceMemory<float> mb[kMaxLayers];  // Adam m for b
    core::DeviceMemory<float> vb[kMaxLayers];  // Adam v for b
    core::DeviceMemory<float> gW[kMaxLayers];  // grad W
    core::DeviceMemory<float> gb[kMaxLayers];  // grad b
    int in_dims[kMaxLayers]  = {};
    int out_dims[kMaxLayers] = {};
    int n_layers = 0;
};

// Saved activations (pre and post) for backward.
struct MLPActs {
    static constexpr int kMaxLayers = 3;
    core::DeviceMemory<float> pre[kMaxLayers];   // pre-activation
    core::DeviceMemory<float> post[kMaxLayers];  // post-activation
};

// ---------------------------------------------------------------------------
// detail kernels
// ---------------------------------------------------------------------------
namespace detail {

// ── Grid helper ──────────────────────────────────────────────────────────────
inline dim3 grid1d(int n, int blk = 256) {
    return dim3((n + blk - 1) / blk);
}

// ── GELU forward ──────────────────────────────────────────────────────────
__launch_bounds__(256, 4)
__global__ void gelu_fwd(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    constexpr float kA = 0.7978845608f, kB = 0.044715f;
    y[i] = 0.5f * v * (1.f + tanhf(kA * (v + kB * v * v * v)));
}

// ── GELU backward (accumulate into grad_x) ────────────────────────────────
__launch_bounds__(256, 4)
__global__ void gelu_bwd(const float* __restrict__ x,
                          const float* __restrict__ dy,
                          float* __restrict__       dx,  // accumulate
                          int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    constexpr float kA = 0.7978845608f, kB = 0.044715f;
    float t  = tanhf(kA * (v + kB * v * v * v));
    float dt = 1.f - t * t;
    float d  = kA * (1.f + 3.f * kB * v * v);
    dx[i] += dy[i] * (0.5f * (1.f + t) + 0.5f * v * dt * d);
}

// ── Softplus forward ──────────────────────────────────────────────────────
__launch_bounds__(256, 4)
__global__ void softplus_fwd(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    y[i] = (v > 20.f) ? v : log1pf(expf(v));
}

// ── Softplus backward (accumulate) ─────────────────────────────────────────
__launch_bounds__(256, 4)
__global__ void softplus_bwd(const float* __restrict__ x,
                              const float* __restrict__ dy,
                              float* __restrict__       dx,  // accumulate
                              int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dx[i] += dy[i] / (1.f + expf(-x[i]));
}

// ── Zero buffer ──────────────────────────────────────────────────────────
__launch_bounds__(256, 4)
__global__ void zero_buf(float* __restrict__ buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = 0.f;
}

// ── Scale buffer ─────────────────────────────────────────────────────────
__launch_bounds__(256, 4)
__global__ void scale_buf(float* __restrict__ buf, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] *= s;
}

// ── Add src into dst ──────────────────────────────────────────────────────
__launch_bounds__(256, 4)
__global__ void add_buf(float* __restrict__ dst, const float* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// ── Subtract src from dst ─────────────────────────────────────────────────
__launch_bounds__(256, 4)
__global__ void sub_buf(float* __restrict__ dst, const float* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] -= src[i];
}

// ── Bias add (broadcast over batch) ──────────────────────────────────────
__launch_bounds__(256, 4)
__global__ void bias_add(float* __restrict__ out, const float* __restrict__ b,
                          int batch, int out_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * out_dim) return;
    out[i] += b[i % out_dim];
}

// ── Bias gradient (sum over batch) ──────────────────────────────────────
__launch_bounds__(256, 4)
__global__ void bias_grad(const float* __restrict__ dy, float* __restrict__ db,
                           int batch, int out_dim) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= out_dim) return;
    float s = 0.f;
    for (int b = 0; b < batch; ++b) s += dy[b * out_dim + d];
    db[d] += s;
}

// ── Adam fused step ──────────────────────────────────────────────────────
__launch_bounds__(256, 4)
__global__ void adam_step(float* __restrict__ p, float* __restrict__ m,
                           float* __restrict__ v, const float* __restrict__ g,
                           float lr, float b1, float b2, float eps,
                           float bc1, float bc2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float mi = b1 * m[i] + (1.f - b1) * gi;
    float vi = b2 * v[i] + (1.f - b2) * gi * gi;
    m[i] = mi; v[i] = vi;
    p[i] += lr * (mi * bc1) / (sqrtf(vi * bc2) + eps);
}

// ── Kaiming uniform init ──────────────────────────────────────────────────
__launch_bounds__(256, 4)
__global__ void kaiming_init(float* __restrict__ w, float scale, uint64_t seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t s;
    curand_init(seed, (unsigned long long)i, 0ULL, &s);
    w[i] = (2.f * curand_uniform(&s) - 1.f) * scale;
}

// ── CSC binary-search gather: batch_dense ← CSC columns ──────────────────
// WHY binary search: CSC row indices within a column are sorted ascending.
__launch_bounds__(256, 4)
__global__ void csc_gather(float* __restrict__       out,
                            const int*   __restrict__ indptr,
                            const int*   __restrict__ indices,
                            const float* __restrict__ vals,
                            const int*   __restrict__ cells,
                            int B, int n_genes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * n_genes) return;
    int bi   = i / n_genes;
    int gene = i % n_genes;
    int ci   = cells[bi];
    int lo   = indptr[ci], hi = indptr[ci + 1] - 1;
    float y  = 0.f;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int idx = indices[mid];
        if      (idx == gene) { y = vals[mid]; break; }
        else if (idx  < gene) lo = mid + 1;
        else                  hi = mid - 1;
    }
    out[i] = y;
}

// ── Gather per-cell metadata (pert_id, dose) for a batch ─────────────────
__launch_bounds__(256, 4)
__global__ void gather_meta(int*         __restrict__ out_pert,
                             float*       __restrict__ out_dose,
                             const int*   __restrict__ all_pert,
                             const float* __restrict__ all_dose,
                             const int*   __restrict__ cells,
                             int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    int ci      = cells[i];
    out_pert[i] = all_pert[ci];
    out_dose[i] = all_dose[ci];
}

// ── Split encoder output [B×2d] → z_mu[B×d], z_logsig[B×d] ──────────────
__launch_bounds__(256, 4)
__global__ void split_mu_sig(const float* __restrict__ enc, float* __restrict__ mu,
                              float* __restrict__ ls, int B, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * d) return;
    int bi = i / d, dim = i % d;
    mu[i] = enc[bi * 2 * d + dim];
    ls[i] = enc[bi * 2 * d + d + dim];
}

// ── Merge grad_mu, grad_ls → grad_enc [B×2d] ──────────────────────────────
__launch_bounds__(256, 4)
__global__ void merge_mu_sig(float* __restrict__ genc,
                              const float* __restrict__ gmu, const float* __restrict__ gls,
                              int B, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * d) return;
    int bi = i / d, dim = i % d;
    genc[bi * 2 * d + dim]     = gmu[i];
    genc[bi * 2 * d + d + dim] = gls[i];
}

// ── Reparameterization sample: z = mu + eps*exp(ls) ──────────────────────
__launch_bounds__(256, 4)
__global__ void reparam_fwd(const float* __restrict__ mu, const float* __restrict__ ls,
                             float* __restrict__ z, float* __restrict__ eps_out,
                             uint64_t seed, uint64_t offset, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t s;
    curand_init(seed, (unsigned long long)i, offset, &s);
    float eps  = curand_normal(&s);
    eps_out[i] = eps;
    z[i]       = mu[i] + eps * expf(ls[i]);
}

// ── Reparam backward ───────────────────────────────────────────────────────
__launch_bounds__(256, 4)
__global__ void reparam_bwd(const float* __restrict__ gz, const float* __restrict__ ls,
                             const float* __restrict__ eps, float* __restrict__ gmu,
                             float* __restrict__ gls, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    gmu[i] += gz[i];
    gls[i] += gz[i] * eps[i] * expf(ls[i]);
}

// ── KL divergence: accumulate per-cell, compute grads ─────────────────────
__launch_bounds__(256, 4)
__global__ void kl_fwd(const float* __restrict__ mu, const float* __restrict__ ls,
                        float* __restrict__ loss, float* __restrict__ gmu,
                        float* __restrict__ gls, float kl_w, int B, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * d) return;
    int bi    = i / d;
    float m   = mu[i], l = ls[i], s2 = expf(2.f * l);
    float kl  = 0.5f * (m * m + s2 - 2.f * l - 1.f);
    atomicAdd(&loss[bi], kl_w * kl);
    gmu[i] += kl_w * m;
    gls[i] += kl_w * (s2 - 1.f);
}

// ── Add perturbation embedding: z_total = z_c + dose*z_p[pert_id] ─────────
__launch_bounds__(256, 4)
__global__ void add_pert_emb(float* __restrict__       zt,
                              const float* __restrict__ zc,
                              const float* __restrict__ emb,   // [n_perts × d]
                              const int*   __restrict__ pids,  // [B]
                              const float* __restrict__ dose,  // [B]
                              int B, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * d) return;
    int bi = i / d, dim = i % d;
    zt[i]  = zc[i] + dose[bi] * emb[pids[bi] * d + dim];
}

// ── Gradient of pert embedding ──────────────────────────────────────────────
__launch_bounds__(256, 4)
__global__ void pert_emb_grad(float* __restrict__       gemb,  // [n_perts×d] accumulate
                               const float* __restrict__ gzt,   // [B×d]
                               const int*   __restrict__ pids,  // [B]
                               const float* __restrict__ dose,  // [B]
                               int B, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * d) return;
    int bi = i / d, dim = i % d;
    atomicAdd(&gemb[pids[bi] * d + dim], dose[bi] * gzt[i]);
}

// ── Broadcast-add-scaled: out = z + scale * vec1d ─────────────────────────
// Used at inference to broadcast single pert embedding across batch.
__launch_bounds__(256, 4)
__global__ void bcast_add_scaled(float* __restrict__       out,
                                  const float* __restrict__ zc,
                                  const float* __restrict__ zp,  // [d] single embedding
                                  float scale, int B, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * d) return;
    out[i] = zc[i] + scale * zp[i % d];
}

// ── NB log-likelihood loss + gradient (sparse-masked) ─────────────────────
// Each thread handles one (batch_cell, gene) pair. Observed y is looked up
// from CSC. fp64 lgamma for precision. Gradient ∂(-NB_LL)/∂mu accumulated.
__launch_bounds__(256, 4)
__global__ void nb_loss_kernel(const float* __restrict__ mu,       // [B × n_genes]
                                const int*   __restrict__ indptr,   // [n_cells+1]
                                const int*   __restrict__ indices,  // [nnz]
                                const float* __restrict__ vals,     // [nnz]
                                const int*   __restrict__ cells,    // [B]
                                int B, int n_genes, float theta,
                                float* __restrict__ loss,            // [B] accumulate
                                float* __restrict__ dmu) {           // [B×n_genes] accumulate
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < B * n_genes; idx += blockDim.x * gridDim.x) {
        int bi   = idx / n_genes;
        int gene = idx % n_genes;
        int ci   = cells[bi];
        // Binary search for observed count.
        int lo = indptr[ci], hi = indptr[ci + 1] - 1;
        float y = 0.f;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            int rid = indices[mid];
            if      (rid == gene) { y = vals[mid]; break; }
            else if (rid  < gene) lo = mid + 1;
            else                  hi = mid - 1;
        }
        float mu_v = fmaxf(mu[idx], 1e-6f);
        // NB LL: fp64 for lgamma precision.
        double yd = y, td = theta, md = mu_v;
        double ll = lgamma(yd + td) - lgamma(td) - lgamma(yd + 1.0)
                  + td * log(td / (td + md)) + yd * log(md / (td + md));
        atomicAdd(&loss[bi], (float)-ll);
        // ∂(-NB_LL)/∂mu
        float dmu_v = (float)(((yd + td) / (md + td)) - yd / md);
        dmu[idx] += dmu_v;
    }
}

// ── Adversarial softmax CE + gradient ─────────────────────────────────────
// One thread per cell. Writes per-cell CE loss and gradient.
__launch_bounds__(256, 4)
__global__ void adv_ce_kernel(const float* __restrict__ logits,  // [B × n_perts]
                               const int*   __restrict__ pids,    // [B]
                               float* __restrict__       loss,    // [B]
                               float* __restrict__       glogits, // [B × n_perts]
                               int B, int n_perts) {
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= B) return;
    const float* lg = logits + bi * n_perts;
    float*       gl = glogits + bi * n_perts;
    float mx = lg[0];
    for (int p = 1; p < n_perts; ++p) mx = fmaxf(mx, lg[p]);
    float sumexp = 0.f;
    for (int p = 0; p < n_perts; ++p) sumexp += expf(lg[p] - mx);
    int label = pids[bi];
    loss[bi] = -(lg[label] - mx - logf(sumexp));
    for (int p = 0; p < n_perts; ++p) {
        float soft = expf(lg[p] - mx) / sumexp;
        gl[p] = soft - (p == label ? 1.f : 0.f);
    }
}

// ── Scalar readback: cub reduce + 4-byte D2H (approved once-per-epoch) ────
inline float reduce_scalar(const float* d_buf, int n, float* d_out,
                             uint8_t* tmp, size_t tmp_bytes, cudaStream_t s) {
    cub::DeviceReduce::Sum(tmp, tmp_bytes, d_buf, d_out, n, s);
    cudaStreamSynchronize(s);
    float h = 0.f;
    // cudaMemcpy: 4 bytes, once per epoch — APPROVED EXCEPTION (scalar only).
    cudaMemcpy(&h, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Internal helpers: MLP init, forward, backward, Adam
// ---------------------------------------------------------------------------

// Allocate and Kaiming-init one MLP layer.
inline void init_layer(MLPParams& mlp, int layer, int in, int out,
                        uint64_t seed, cudaStream_t s) {
    int nW = out * in, nb = out;
    mlp.W[layer]  = core::DeviceMemory<float>(nW);
    mlp.b[layer]  = core::DeviceMemory<float>(nb);
    mlp.mW[layer] = core::DeviceMemory<float>(nW);
    mlp.vW[layer] = core::DeviceMemory<float>(nW);
    mlp.mb[layer] = core::DeviceMemory<float>(nb);
    mlp.vb[layer] = core::DeviceMemory<float>(nb);
    mlp.gW[layer] = core::DeviceMemory<float>(nW);
    mlp.gb[layer] = core::DeviceMemory<float>(nb);
    mlp.in_dims[layer]  = in;
    mlp.out_dims[layer] = out;
    detail::kaiming_init<<<detail::grid1d(nW), 256, 0, s>>>(
        mlp.W[layer].get(), sqrtf(2.f / (float)in), seed, nW);
    cudaMemsetAsync(mlp.b[layer].get(),  0, nb * sizeof(float), s);
    cudaMemsetAsync(mlp.mW[layer].get(), 0, nW * sizeof(float), s);
    cudaMemsetAsync(mlp.vW[layer].get(), 0, nW * sizeof(float), s);
    cudaMemsetAsync(mlp.mb[layer].get(), 0, nb * sizeof(float), s);
    cudaMemsetAsync(mlp.vb[layer].get(), 0, nb * sizeof(float), s);
    cudaMemsetAsync(mlp.gW[layer].get(), 0, nW * sizeof(float), s);
    cudaMemsetAsync(mlp.gb[layer].get(), 0, nb * sizeof(float), s);
}

// Ensure activation buffers sized for current batch.
inline void ensure_acts(MLPActs& acts, int layer, int batch, int out_dim) {
    size_t need = (size_t)batch * out_dim;
    if (acts.pre[layer].size() < need) {
        acts.pre[layer]  = core::DeviceMemory<float>(need);
        acts.post[layer] = core::DeviceMemory<float>(need);
    }
}

// MLP linear+activation forward. use_gelu=true → GELU; false → identity.
inline void layer_fwd(cublasHandle_t cb, const MLPParams& mlp, int l, MLPActs& acts,
                       const float* x, float* y, int B, bool gelu, cudaStream_t s) {
    int in = mlp.in_dims[l], out = mlp.out_dims[l];
    ensure_acts(acts, l, B, out);
    float a = 1.f, b = 0.f;
    // Sgemm: y[B×out] = x[B×in] * W^T[in×out], column-major equivalent.
    cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, out, B, in, &a,
                mlp.W[l].get(), in, x, in, &b, y, out);
    detail::bias_add<<<detail::grid1d(B * out), 256, 0, s>>>(y, mlp.b[l].get(), B, out);
    // Save pre-activation for backward.
    cudaMemcpyAsync(acts.pre[l].get(), y, B * out * sizeof(float),
                    cudaMemcpyDeviceToDevice, s);
    if (gelu) {
        detail::gelu_fwd<<<detail::grid1d(B * out), 256, 0, s>>>(y, acts.post[l].get(), B * out);
        cudaMemcpyAsync(y, acts.post[l].get(), B * out * sizeof(float),
                        cudaMemcpyDeviceToDevice, s);
    } else {
        cudaMemcpyAsync(acts.post[l].get(), y, B * out * sizeof(float),
                        cudaMemcpyDeviceToDevice, s);
    }
}

// MLP backward through one layer.
// grad_x: gradient accumulation buffer for layer input (or nullptr if leaf).
inline void layer_bwd(cublasHandle_t cb, const MLPParams& mlp, int l, const MLPActs& acts,
                       const float* x, const float* dy, float* grad_x,
                       float* scratch,  // [B × out_dim] scratch
                       int B, bool gelu, cudaStream_t s) {
    int in = mlp.in_dims[l], out = mlp.out_dims[l];
    int n_pre = B * out;
    // Activation backward into scratch.
    if (gelu) {
        cudaMemsetAsync(scratch, 0, n_pre * sizeof(float), s);
        detail::gelu_bwd<<<detail::grid1d(n_pre), 256, 0, s>>>(
            acts.pre[l].get(), dy, scratch, n_pre);
    } else {
        cudaMemcpyAsync(scratch, dy, n_pre * sizeof(float), cudaMemcpyDeviceToDevice, s);
    }
    // grad_W += scratch^T * x
    float a1 = 1.f, b1 = 1.f;
    cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_T, in, out, B, &a1,
                x, in, scratch, out, &b1, mlp.gW[l].get(), in);
    // grad_b += sum_batch(scratch)
    detail::bias_grad<<<detail::grid1d(out), 256, 0, s>>>(scratch, mlp.gb[l].get(), B, out);
    // grad_x += scratch * W (if not leaf)
    if (grad_x) {
        cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, in, B, out, &a1,
                    mlp.W[l].get(), in, scratch, out, &b1, grad_x, in);
    }
}

// Zero all gradient buffers of an MLP.
inline void zero_grad(MLPParams& mlp, cudaStream_t s) {
    for (int l = 0; l < mlp.n_layers; ++l) {
        cudaMemsetAsync(mlp.gW[l].get(), 0, mlp.out_dims[l] * mlp.in_dims[l] * sizeof(float), s);
        cudaMemsetAsync(mlp.gb[l].get(), 0, mlp.out_dims[l] * sizeof(float), s);
    }
}

// Adam update for all layers.
inline void adam_update(MLPParams& mlp, float lr, float b1, float b2, float eps,
                         float bc1, float bc2, cudaStream_t s) {
    for (int l = 0; l < mlp.n_layers; ++l) {
        int nW = mlp.out_dims[l] * mlp.in_dims[l], nb = mlp.out_dims[l];
        detail::adam_step<<<detail::grid1d(nW), 256, 0, s>>>(
            mlp.W[l].get(), mlp.mW[l].get(), mlp.vW[l].get(),
            mlp.gW[l].get(), lr, b1, b2, eps, bc1, bc2, nW);
        detail::adam_step<<<detail::grid1d(nb), 256, 0, s>>>(
            mlp.b[l].get(), mlp.mb[l].get(), mlp.vb[l].get(),
            mlp.gb[l].get(), lr, b1, b2, eps, bc1, bc2, nb);
    }
}

// ---------------------------------------------------------------------------
// train_perturb_graph — main training entry point
//
// counts_csc:  n_cells × n_genes (cells = columns, per .1pz CSC convention)
// pert_labels: [n_cells] ∈ [0, n_perts)
// dose:        [n_cells] scalar dose per cell (1.0 for binary CRISPR)
// ---------------------------------------------------------------------------
inline PerturbGraphResult train_perturb_graph(
    const core::DeviceCSC&    counts_csc,
    const std::vector<int>&   pert_labels,
    const std::vector<float>& dose,
    int                       n_perts,
    const PerturbGraphConfig& cfg,
    core::GPUContext&         ctx)
{
    if (cfg.use_gnn) {
        throw std::runtime_error(
            "GNN extension deferred to CYCLE-32-FOLLOWUP-GNN-EXTENSION. "
            "Set cfg.use_gnn=false for CPA-only mode.");
    }

    cudaStream_t   s  = ctx.stream;
    cublasHandle_t cb = ctx.cublas;

    const int n  = (int)counts_csc.cols;   // cells
    const int m  = (int)counts_csc.rows;   // genes
    const int d  = cfg.d_latent;
    const int dh = cfg.d_hidden;
    const int B  = cfg.batch_size;
    constexpr int kAdvHid = 64;
    constexpr float kTheta = 1.0f;
    constexpr float kB1 = 0.9f, kB2 = 0.999f, kEps = 1e-8f;

    // ── Build encoder: m → 256 → dh → 2d (GELU, last layer no act)
    MLPParams enc; enc.n_layers = 3;
    init_layer(enc, 0, m,   256,    cfg.seed + 1, s);
    init_layer(enc, 1, 256, dh,     cfg.seed + 2, s);
    init_layer(enc, 2, dh,  2 * d,  cfg.seed + 3, s);

    // ── Build decoder: d → dh → 256 → m (GELU, softplus output)
    MLPParams dec; dec.n_layers = 3;
    init_layer(dec, 0, d,   dh,  cfg.seed + 4, s);
    init_layer(dec, 1, dh,  256, cfg.seed + 5, s);
    init_layer(dec, 2, 256, m,   cfg.seed + 6, s);

    // ── Build adversarial discriminator: d → kAdvHid → n_perts
    MLPParams adv; adv.n_layers = 2;
    init_layer(adv, 0, d,       kAdvHid, cfg.seed + 7, s);
    init_layer(adv, 1, kAdvHid, n_perts, cfg.seed + 8, s);

    // ── Perturbation embeddings + Adam state
    const int pemb_n = n_perts * d;
    core::DeviceMemory<float> pemb(pemb_n), m_pe(pemb_n), v_pe(pemb_n), g_pe(pemb_n);
    detail::kaiming_init<<<detail::grid1d(pemb_n), 256, 0, s>>>(
        pemb.get(), sqrtf(2.f / (float)d), cfg.seed + 9, pemb_n);
    cudaMemsetAsync(m_pe.get(), 0, pemb_n * sizeof(float), s);
    cudaMemsetAsync(v_pe.get(), 0, pemb_n * sizeof(float), s);
    cudaMemsetAsync(g_pe.get(), 0, pemb_n * sizeof(float), s);

    // ── Copy all-cell metadata to device
    core::DeviceMemory<int>   d_all_pert(n);
    core::DeviceMemory<float> d_all_dose(n);
    cudaMemcpyAsync(d_all_pert.get(), pert_labels.data(), n * sizeof(int),   cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_all_dose.get(), dose.data(),        n * sizeof(float), cudaMemcpyHostToDevice, s);

    // ── Device CSC pointers (zero-copy)
    const int*   indptr  = counts_csc.col_ptr.get();
    const int*   indices = counts_csc.row_indices.get();
    const float* vals    = counts_csc.values.get();

    // ── Minibatch working buffers (B × sizes)
    core::DeviceMemory<int>   d_cells(B), d_bpert(B);
    core::DeviceMemory<float> d_bdose(B);
    core::DeviceMemory<float> x_in(B * m);            // gathered dense input
    core::DeviceMemory<float> h0(B * 256), h1(B * dh);
    core::DeviceMemory<float> enc_out(B * 2 * d);
    core::DeviceMemory<float> z_mu(B * d), z_ls(B * d), z_c(B * d), eps(B * d);
    core::DeviceMemory<float> z_tot(B * d);
    core::DeviceMemory<float> dh0(B * dh), dh1(B * 256), dec_pre(B * m), mu_out(B * m);
    core::DeviceMemory<float> ah0(B * kAdvHid), adv_logits(B * n_perts);

    // Loss and gradient buffers
    core::DeviceMemory<float> loss(B), adv_loss(B);
    core::DeviceMemory<float> d_scalar(1), adv_scalar(1);
    core::DeviceMemory<float> g_dmu(B * m);
    core::DeviceMemory<float> g_ztot(B * d), g_zc(B * d);
    core::DeviceMemory<float> g_zmu(B * d), g_zls(B * d);
    core::DeviceMemory<float> g_enc_out(B * 2 * d);
    core::DeviceMemory<float> g_adv_logits(B * n_perts), g_ah0(B * kAdvHid), g_adv_zc(B * d);
    core::DeviceMemory<float> g_h0(B * 256), g_h1(B * dh);
    core::DeviceMemory<float> g_dh0(B * dh), g_dh1(B * 256), g_dec_pre(B * m);
    // Scratch buffers for layer_bwd (one per MLP layer, reused).
    core::DeviceMemory<float> sc256(B * 256), sc_dh(B * dh), sc2d(B * 2 * d);
    core::DeviceMemory<float> sc_m(B * m), sc_np(B * n_perts), sc_ah(B * kAdvHid);

    // CUB temp for scalar reduction.
    size_t cub_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, cub_bytes, loss.get(), d_scalar.get(), B, s);
    core::DeviceMemory<uint8_t> cub_tmp(cub_bytes + 1);

    MLPActs ea, da, aa;

    // Host shuffle state.
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(cfg.seed);

    PerturbGraphResult res;
    res.epoch_losses.resize(cfg.n_epochs, 0.f);
    res.adv_losses.resize(cfg.n_epochs, 0.f);

    // ── Training loop ─────────────────────────────────────────────────────
    //
    // cudaMemcpy audit (self-check):
    //   Line A: cudaMemcpyAsync cells H2D, 1 KB, per-batch — per-batch budget OK.
    //   Line B: cudaMemcpyAsync async D2D inside layer_fwd (pre-act save) — D2D, no PCIe.
    //   Line C: cudaMemcpy 4 bytes D2H in reduce_scalar, once per epoch — APPROVED.
    //   No synchronous cudaMemcpy inside the per-batch inner loop.
    //   CONFIRMED: zero non-approved cudaMemcpy in any loop > 5 iters.
    //
    for (int ep = 0; ep < cfg.n_epochs; ++ep) {
        std::shuffle(order.begin(), order.end(), rng);
        float bc1 = 1.f / (1.f - powf(kB1, (float)(ep + 1)));
        float bc2 = 1.f / (1.f - powf(kB2, (float)(ep + 1)));
        int n_b = 0;

        for (int bs = 0; bs < n; bs += B) {
            int ab = std::min(B, n - bs);
            if (ab < 2) break;

            // [Line A] Upload batch cell indices (1 KB async H2D — per-batch budget OK).
            cudaMemcpyAsync(d_cells.get(), order.data() + bs,
                            ab * sizeof(int), cudaMemcpyHostToDevice, s);
            // Gather per-cell pert/dose metadata for this batch.
            detail::gather_meta<<<detail::grid1d(ab), 256, 0, s>>>(
                d_bpert.get(), d_bdose.get(),
                d_all_pert.get(), d_all_dose.get(), d_cells.get(), ab);

            // Gather dense input from CSC.
            detail::csc_gather<<<detail::grid1d(ab * m), 256, 0, s>>>(
                x_in.get(), indptr, indices, vals, d_cells.get(), ab, m);

            // Zero grad/loss buffers.
            zero_grad(enc, s); zero_grad(dec, s); zero_grad(adv, s);
            cudaMemsetAsync(g_pe.get(), 0, pemb_n * sizeof(float), s);
            cudaMemsetAsync(loss.get(),     0, ab * sizeof(float), s);
            cudaMemsetAsync(adv_loss.get(), 0, ab * sizeof(float), s);
            cudaMemsetAsync(g_zmu.get(),    0, ab * d * sizeof(float), s);
            cudaMemsetAsync(g_zls.get(),    0, ab * d * sizeof(float), s);
            cudaMemsetAsync(g_dmu.get(),    0, ab * m * sizeof(float), s);

            // ── ENCODER FORWARD ─────────────────────────────────────────────
            layer_fwd(cb, enc, 0, ea, x_in.get(), h0.get(),      ab, true,  s);
            layer_fwd(cb, enc, 1, ea, h0.get(),   h1.get(),      ab, true,  s);
            layer_fwd(cb, enc, 2, ea, h1.get(),   enc_out.get(), ab, false, s);
            detail::split_mu_sig<<<detail::grid1d(ab * d), 256, 0, s>>>(
                enc_out.get(), z_mu.get(), z_ls.get(), ab, d);

            // ── REPARAMETERIZATION + KL ──────────────────────────────────────
            detail::reparam_fwd<<<detail::grid1d(ab * d), 256, 0, s>>>(
                z_mu.get(), z_ls.get(), z_c.get(), eps.get(),
                cfg.seed, (uint64_t)ep * 10000 + bs, ab * d);
            detail::kl_fwd<<<detail::grid1d(ab * d), 256, 0, s>>>(
                z_mu.get(), z_ls.get(), loss.get(),
                g_zmu.get(), g_zls.get(), cfg.kl_weight, ab, d);

            // ── ADD PERTURBATION EMBEDDING ───────────────────────────────────
            detail::add_pert_emb<<<detail::grid1d(ab * d), 256, 0, s>>>(
                z_tot.get(), z_c.get(), pemb.get(), d_bpert.get(), d_bdose.get(), ab, d);

            // ── DECODER FORWARD ──────────────────────────────────────────────
            layer_fwd(cb, dec, 0, da, z_tot.get(),  dh0.get(),     ab, true,  s);
            layer_fwd(cb, dec, 1, da, dh0.get(),    dh1.get(),     ab, true,  s);
            layer_fwd(cb, dec, 2, da, dh1.get(),    dec_pre.get(), ab, false, s);
            detail::softplus_fwd<<<detail::grid1d(ab * m), 256, 0, s>>>(
                dec_pre.get(), mu_out.get(), ab * m);

            // ── NB LOSS ──────────────────────────────────────────────────────
            detail::nb_loss_kernel<<<detail::grid1d(ab * m), 256, 0, s>>>(
                mu_out.get(), indptr, indices, vals, d_cells.get(),
                ab, m, kTheta, loss.get(), g_dmu.get());

            // ── ADVERSARIAL FORWARD ──────────────────────────────────────────
            layer_fwd(cb, adv, 0, aa, z_c.get(),        ah0.get(),        ab, true,  s);
            layer_fwd(cb, adv, 1, aa, ah0.get(),         adv_logits.get(), ab, false, s);
            cudaMemsetAsync(g_adv_logits.get(), 0, ab * n_perts * sizeof(float), s);
            detail::adv_ce_kernel<<<detail::grid1d(ab), 256, 0, s>>>(
                adv_logits.get(), d_bpert.get(), adv_loss.get(),
                g_adv_logits.get(), ab, n_perts);
            // Scale adv gradient by adversarial_weight.
            detail::scale_buf<<<detail::grid1d(ab * n_perts), 256, 0, s>>>(
                g_adv_logits.get(), cfg.adversarial_weight, ab * n_perts);

            // ── BACKWARD: decoder ────────────────────────────────────────────
            cudaMemsetAsync(g_dec_pre.get(), 0, ab * m * sizeof(float), s);
            detail::softplus_bwd<<<detail::grid1d(ab * m), 256, 0, s>>>(
                dec_pre.get(), g_dmu.get(), g_dec_pre.get(), ab * m);
            cudaMemsetAsync(g_dh1.get(), 0, ab * 256 * sizeof(float), s);
            layer_bwd(cb, dec, 2, da, dh1.get(),  g_dec_pre.get(), g_dh1.get(), sc_m.get(),  ab, false, s);
            cudaMemsetAsync(g_dh0.get(), 0, ab * dh * sizeof(float), s);
            layer_bwd(cb, dec, 1, da, dh0.get(),  g_dh1.get(),     g_dh0.get(), sc256.get(), ab, true,  s);
            cudaMemsetAsync(g_ztot.get(), 0, ab * d * sizeof(float), s);
            layer_bwd(cb, dec, 0, da, z_tot.get(), g_dh0.get(),    g_ztot.get(), sc_dh.get(), ab, true,  s);

            // grad_z_c += grad_z_total (∂z_total/∂z_c = I).
            // grad_pemb += dose * grad_z_total (perturbation embedding gradient).
            cudaMemsetAsync(g_zc.get(), 0, ab * d * sizeof(float), s);
            detail::add_buf<<<detail::grid1d(ab * d), 256, 0, s>>>(g_zc.get(), g_ztot.get(), ab * d);
            detail::pert_emb_grad<<<detail::grid1d(ab * d), 256, 0, s>>>(
                g_pe.get(), g_ztot.get(), d_bpert.get(), d_bdose.get(), ab, d);

            // ── BACKWARD: adversarial (GRL: subtract from grad_z_c) ──────────
            cudaMemsetAsync(g_ah0.get(), 0, ab * kAdvHid * sizeof(float), s);
            layer_bwd(cb, adv, 1, aa, ah0.get(),  g_adv_logits.get(), g_ah0.get(),   sc_np.get(), ab, false, s);
            cudaMemsetAsync(g_adv_zc.get(), 0, ab * d * sizeof(float), s);
            layer_bwd(cb, adv, 0, aa, z_c.get(),  g_ah0.get(),        g_adv_zc.get(), sc_ah.get(), ab, true,  s);
            // GRL: negate adversarial gradient (subtract from encoder path).
            detail::sub_buf<<<detail::grid1d(ab * d), 256, 0, s>>>(g_zc.get(), g_adv_zc.get(), ab * d);

            // ── BACKWARD: reparam → grad_z_mu, grad_z_ls ────────────────────
            detail::reparam_bwd<<<detail::grid1d(ab * d), 256, 0, s>>>(
                g_zc.get(), z_ls.get(), eps.get(), g_zmu.get(), g_zls.get(), ab * d);

            // ── BACKWARD: encoder ────────────────────────────────────────────
            detail::merge_mu_sig<<<detail::grid1d(ab * d), 256, 0, s>>>(
                g_enc_out.get(), g_zmu.get(), g_zls.get(), ab, d);
            cudaMemsetAsync(g_h1.get(), 0, ab * dh * sizeof(float), s);
            layer_bwd(cb, enc, 2, ea, h1.get(),      g_enc_out.get(), g_h1.get(),  sc2d.get(), ab, false, s);
            cudaMemsetAsync(g_h0.get(), 0, ab * 256 * sizeof(float), s);
            layer_bwd(cb, enc, 1, ea, h0.get(),      g_h1.get(),      g_h0.get(),  sc_dh.get(), ab, true,  s);
            layer_bwd(cb, enc, 0, ea, x_in.get(),    g_h0.get(),      nullptr,      sc256.get(), ab, true,  s);

            // ── ADAM UPDATES ─────────────────────────────────────────────────
            float lr = cfg.learning_rate;
            adam_update(enc, lr,         kB1, kB2, kEps, bc1, bc2, s);
            adam_update(dec, lr,         kB1, kB2, kEps, bc1, bc2, s);
            adam_update(adv, lr * 0.1f,  kB1, kB2, kEps, bc1, bc2, s);  // adv slower
            detail::adam_step<<<detail::grid1d(pemb_n), 256, 0, s>>>(
                pemb.get(), m_pe.get(), v_pe.get(), g_pe.get(),
                lr, kB1, kB2, kEps, bc1, bc2, pemb_n);
            ++n_b;
        }

        // [Line C] Epoch scalar readback: 4 bytes, once per epoch — APPROVED.
        res.epoch_losses[ep] = detail::reduce_scalar(
            loss.get(), B, d_scalar.get(), cub_tmp.get(), cub_bytes, s) /
            (float)std::max(n_b, 1);
        res.adv_losses[ep] = detail::reduce_scalar(
            adv_loss.get(), B, adv_scalar.get(), cub_tmp.get(), cub_bytes, s) /
            (float)std::max(n_b, 1);
    }

    // ── Encode all cells (final z_mu, no noise) for result.cell_latents ───
    res.cell_latents = core::DeviceMemory<float>((size_t)n * d);
    for (int bs = 0; bs < n; bs += B) {
        int ab = std::min(B, n - bs);
        std::vector<int> seq(ab); std::iota(seq.begin(), seq.end(), bs);
        cudaMemcpyAsync(d_cells.get(), seq.data(), ab * sizeof(int), cudaMemcpyHostToDevice, s);
        detail::csc_gather<<<detail::grid1d(ab * m), 256, 0, s>>>(
            x_in.get(), indptr, indices, vals, d_cells.get(), ab, m);
        layer_fwd(cb, enc, 0, ea, x_in.get(), h0.get(),      ab, true,  s);
        layer_fwd(cb, enc, 1, ea, h0.get(),   h1.get(),      ab, true,  s);
        layer_fwd(cb, enc, 2, ea, h1.get(),   enc_out.get(), ab, false, s);
        // Copy only z_mu into result (no noise at inference).
        detail::split_mu_sig<<<detail::grid1d(ab * d), 256, 0, s>>>(
            enc_out.get(), res.cell_latents.get() + bs * d, z_ls.get(), ab, d);
    }

    res.pert_embeddings = std::move(pemb);
    res.n_cells = n; res.n_genes = m; res.n_perts = n_perts; res.d_latent = d;
    cudaStreamSynchronize(s);
    return res;
}

// ---------------------------------------------------------------------------
// predict_perturbation — encode query cells, swap perturbation, decode.
//
// Returns dense [n_query × n_genes] predicted mean expression (fp32).
// WHY dense output: inference output is inherently dense; users subset HVGs.
//
// NOTE: caller must pass the same enc and dec MLPParams used during training.
//       PerturbGraphResult holds pert_embeddings and metadata only.
// ---------------------------------------------------------------------------
inline core::DeviceMemory<float> predict_perturbation(
    const PerturbGraphResult&  res,
    const MLPParams&           enc,
    const MLPParams&           dec,
    const core::DeviceCSC&     query_csc,
    int                        target_pert_id,
    float                      target_dose,
    const PerturbGraphConfig&  cfg,
    core::GPUContext&           ctx)
{
    if (cfg.use_gnn) {
        throw std::runtime_error(
            "GNN extension deferred to CYCLE-32-FOLLOWUP-GNN-EXTENSION.");
    }
    cudaStream_t   s  = ctx.stream;
    cublasHandle_t cb = ctx.cublas;

    const int nq = (int)query_csc.cols;
    const int m  = res.n_genes;
    const int d  = res.d_latent;
    const int dh = cfg.d_hidden;
    const int B  = cfg.batch_size;

    core::DeviceMemory<float> pred((size_t)nq * m);
    core::DeviceMemory<float> x_in(B * m), h0(B * 256), h1(B * dh);
    core::DeviceMemory<float> enc_out(B * 2 * d), z_mu(B * d), z_ls(B * d);
    core::DeviceMemory<float> z_tot(B * d);
    core::DeviceMemory<float> dh0(B * dh), dh1(B * 256), dec_pre(B * m), mu_out(B * m);
    core::DeviceMemory<int>   d_cells(B);
    MLPActs ea, da;

    const float* z_p = res.pert_embeddings.get() + target_pert_id * d;

    for (int bs = 0; bs < nq; bs += B) {
        int ab = std::min(B, nq - bs);
        std::vector<int> seq(ab); std::iota(seq.begin(), seq.end(), bs);
        cudaMemcpyAsync(d_cells.get(), seq.data(), ab * sizeof(int), cudaMemcpyHostToDevice, s);
        detail::csc_gather<<<detail::grid1d(ab * m), 256, 0, s>>>(
            x_in.get(), query_csc.col_ptr.get(), query_csc.row_indices.get(),
            query_csc.values.get(), d_cells.get(), ab, m);
        layer_fwd(cb, enc, 0, ea, x_in.get(),     h0.get(),      ab, true,  s);
        layer_fwd(cb, enc, 1, ea, h0.get(),        h1.get(),      ab, true,  s);
        layer_fwd(cb, enc, 2, ea, h1.get(),        enc_out.get(), ab, false, s);
        detail::split_mu_sig<<<detail::grid1d(ab * d), 256, 0, s>>>(
            enc_out.get(), z_mu.get(), z_ls.get(), ab, d);
        // z_tot = z_mu + dose * z_p (broadcast pert embedding).
        detail::bcast_add_scaled<<<detail::grid1d(ab * d), 256, 0, s>>>(
            z_tot.get(), z_mu.get(), z_p, target_dose, ab, d);
        layer_fwd(cb, dec, 0, da, z_tot.get(),    dh0.get(),     ab, true,  s);
        layer_fwd(cb, dec, 1, da, dh0.get(),      dh1.get(),     ab, true,  s);
        layer_fwd(cb, dec, 2, da, dh1.get(),      dec_pre.get(), ab, false, s);
        detail::softplus_fwd<<<detail::grid1d(ab * m), 256, 0, s>>>(
            dec_pre.get(), mu_out.get(), ab * m);
        cudaMemcpyAsync(pred.get() + bs * m, mu_out.get(),
                        ab * m * sizeof(float), cudaMemcpyDeviceToDevice, s);
    }
    cudaStreamSynchronize(s);
    return pred;
}

}  // namespace perturbation
}  // namespace singlet::gpu
