---
feature: models/scvi (new; not originally in the 17-row roadmap)
module: include/singlet-gpu/models/scvi/
cycle: 65 or later
phase: D+E (foundational implementation + benchmark)
status: draft
mandate_v2_ref: §B (in-house foundation models, bare-metal CUDA)
rule_ref: 27 (foundation models are bare-metal CUDA, not framework wrappers)
parent_feature: 12 (batch integration) — scVI is one of the three integration backends in feature 12
---

# scVI — bare-metal CUDA foundation model

**Rule 27 binding**: this module is **hand-written CUDA** — NO PyTorch, NO LibTorch, NO Python reference at runtime. Python references appear ONLY in correctness tests inside `analysis-validator`. This is the first foundation model in singlet-gpu and sets the pattern for scANVI / totalVI / peakVI / multiVI / future transformer models (Geneformer, scGPT, scFoundation).

## 0. Architecture overview

scVI is a variational autoencoder with:
- **Encoder**: `genes → n_hidden → n_hidden → (μ_z, log σ_z²)` (fully connected, 1–3 hidden layers)
- **Latent**: `z ~ Normal(μ_z, σ_z²)` via reparameterization trick
- **Decoder**: `z → n_hidden → n_hidden → (μ_x, θ, π)` with per-gene parameters
- **Likelihood**: Negative Binomial or Zero-Inflated NB on the raw counts, parameterized by the decoder outputs, scaled by per-cell library size
- **Loss**: `−ELBO = NLL_ZINB + KL(q(z|x) || p(z))`
- **Training**: minibatch Adam on the ELBO with KL warmup schedule

Published reference: Lopez, Regier et al. 2018, extended by Xu et al. 2021 (scANVI).

## 1. Success metrics

On GSM4037629 (20,866 cells × 310,797 genes — pre-filtered to HVG top-2000 for a fair comparison):

- **Wall per epoch (minibatch size=128)**: ≤ `scvi-tools GPU mode` × 0.5 (2× speedup headline).
- **Total training time (400 epochs, standard schedule)**: ≤ scvi-tools × 0.5.
- **Peak device memory (training peak)**: ≤ scvi-tools × 1.0 (not worse).
- **ELBO at convergence**: within 1% of scvi-tools final ELBO on the same dataset + seed.
- **Latent space quality**: scIB `iLISI` (integration LISI score) ≥ 0.95 × scvi-tools iLISI on a multi-sample integration benchmark. `cLISI` (cell-type LISI) within ±5% of scvi-tools.
- **Downstream integration correctness**: kBET > 0.8, ARI of Leiden partitions ≥ 0.95 vs scvi-tools latent.

Frontier gate: dominance on wall AND (ELBO within 1% OR scIB within 5%). Rule 31 autonomy delta ≤ 10%.

## 2. SOTA baselines

| Baseline | Framework | Notes |
|---|---|---|
| **scvi-tools (PyTorch CUDA)** | Python | Reference implementation; the ELBO and scIB gold standard |
| **scvi-tools (Jax backend)** | Python | Recently added Jax mode; may be faster than PyTorch mode |
| **scanvi (scvi-tools)** | Python | Semi-supervised extension; used for celltype label transfer |
| **totalVI (scvi-tools)** | Python | Extension for CITE-seq |
| **peakVI / multiVI** | Python | ATAC / multiome extensions |

All four extensions share the same encoder/decoder backbone — if we get scVI right, the extensions are adapters on top.

## 3. Module layout

```
include/singlet-gpu/models/scvi/
├── config.h              // ScviConfig struct (every hyperparameter)
├── encoder.h             // cuBLAS/cuBLASLt GEMM + fused activations
├── decoder.h             // cuBLAS/cuBLASLt GEMM + ZINB parameter heads
├── loss_zinb.h           // ZINB likelihood + KL divergence, fused
├── loss_nb.h             // NB-only variant (simpler, faster)
├── optimizer_adam.h      // In-house fused Adam/AdamW (no cuDNN, no torch)
├── minibatch_sampler.h   // Device-side index gather from DeviceCSC
├── trainer.h             // Top-level train loop
├── latent.h              // Inference (encode-only) path post-training
└── detail/
    ├── reparam.h         // cuRAND device API reparameterization trick
    ├── layer_norm.h      // Fused block-reduce layer norm
    ├── dropout.h         // cuRAND dropout with reproducible seed
    └── library_size.h    // Per-cell library size (observed or learned)
```

All header-only. All CUDA device functions `inline`. cuBLAS handles from `factornet::gpu::GPUContext`; no new global state.

## 4. Config struct (the Rule 31 autonomy contract)

```cpp
namespace singlet_gpu::models::scvi {

struct ScviConfig {
    // Architecture
    int32_t latent_dim = 10;        // auto: reconstruction-elbow
    int32_t n_hidden = 128;         // auto: 128 for small, 256 for medium
    int32_t n_layers = 1;           // auto: 1 for n_cells<10k, 2 for medium, 3 for large
    float   dropout_rate = 0.1f;    // auto: 0.1 (scvi-tools default)
    bool    use_layer_norm = true;
    bool    use_batch_norm = false; // discouraged; layer norm is strictly better for VAEs

    // Likelihood
    enum class GeneLikelihood { NB, ZINB, Gaussian } gene_likelihood = GeneLikelihood::ZINB;
    enum class Dispersion { GENE, GENE_CELL, GENE_BATCH } dispersion = Dispersion::GENE;
    enum class LibraryPrior { OBSERVED, LEARNED } library_prior = LibraryPrior::OBSERVED;

    // Batch covariates
    int32_t n_batches = 1;
    int32_t n_continuous_cov = 0;
    int32_t n_labels = 0;           // for scANVI

    // Training
    int32_t n_epochs = 400;         // auto: early stop on ELBO plateau
    int32_t batch_size = 128;       // auto: max(128, free_device_mem / per_sample_footprint)
    float   learning_rate = 1e-3f;  // auto: 1e-3 (scvi-tools default)
    float   kl_warmup_epochs = 400; // auto: n_epochs
    float   weight_decay = 1e-6f;
    float   eps = 1e-8f;

    // Determinism
    uint64_t random_seed = 0;
    bool    deterministic = false;  // opt-in

    // Stream
    cudaStream_t stream = 0;
};

struct ScviResult {
    factornet::gpu::DenseMatrixGPU<float> latent_mean;   // n_cells × latent_dim
    factornet::gpu::DenseMatrixGPU<float> latent_var;    // n_cells × latent_dim
    std::vector<float> elbo_trajectory;                  // per-epoch training ELBO
    std::vector<float> kl_trajectory;                    // per-epoch KL component
    std::vector<float> nll_trajectory;                   // per-epoch NLL component
    std::map<std::string, std::string> metadata;         // autotune decisions recorded here
};

// Manual
ScviResult train(const factornet::gpu::SparseMatrixGPU<float>& counts,
                 const ScviConfig& cfg);

// Rule 31 autonomy contract
ScviResult train(const factornet::gpu::SparseMatrixGPU<float>& counts);

// Inference-only path
factornet::gpu::DenseMatrixGPU<float>
encode(const ScviResult& trained_model,
       const factornet::gpu::SparseMatrixGPU<float>& new_counts);

} // namespace singlet_gpu::models::scvi
```

## 5. Training loop (the per-epoch hot path)

```
For epoch in 0..n_epochs:
  For minibatch in minibatch_sampler(counts, batch_size):       // device index gather
    mb_dense = gather_columns(counts, mb_indices)               // sparse → small dense, O(batch×genes)
    lib_size = row_sum(mb_dense)                                 // cub DeviceSegmentedReduce
    mb_norm = log1p(mb_dense / lib_size × 1e4)                   // element-wise fused

    h1 = relu(mb_norm @ W1 + b1) with dropout                   // cuBLAS GEMM + fused activation
    h2 = relu(h1 @ W2 + b2) with dropout
    mu_z = h2 @ W_mu_z + b_mu_z                                  // GEMM
    logvar_z = h2 @ W_lv_z + b_lv_z

    eps = curand_normal(batch_size × latent_dim)                 // reparameterization
    z = mu_z + exp(0.5 × logvar_z) × eps

    d1 = relu(z @ W_d1 + b_d1)
    d2 = relu(d1 @ W_d2 + b_d2)
    mu_x = softmax(d2 @ W_mu_x + b_mu_x) × lib_size              // scaled mean
    theta = exp(d2 @ W_theta + b_theta)                          // dispersion
    pi = sigmoid(d2 @ W_pi + b_pi)                               // ZINB dropout prob

    nll = zinb_nll(mb_dense, mu_x, theta, pi)                    // fused, per-cell sum
    kl = 0.5 × sum(mu_z² + exp(logvar_z) − 1 − logvar_z)         // closed form
    loss = nll + kl_warmup_weight × kl

    grad_W*, grad_b* = backward(loss)                             // hand-written, not autograd
    adam_update(params, grads, m, v, t, lr, beta1, beta2, eps)    // fused kernel
```

### Key performance notes

- **No autograd graph.** Backprop is hand-written per-layer. This is the main cost we pay vs PyTorch (dev time) and the main win we get (no graph overhead).
- **cuBLAS GEMMs dominate wall time.** For `batch=128, n_hidden=128, genes=2000` the encoder GEMM is `128×2000 × 2000×128` = 65 MFLOP per layer. On H100 this is ~50 µs per layer per direction. Forward + backward for 4 layers = ~400 µs per minibatch. For 20k cells / 128 batch = 156 minibatches/epoch × 400 epochs = ~40 sec total. **Wall target: ≤20 sec**.
- **cuBLASLt epilogue fusion** for `relu + dropout` directly in the GEMM call — eliminates one kernel launch per layer.
- **Fused Adam update**: one kernel launch for all parameters at once, strided via a packed parameter layout. Saves ~N_params kernel launches per step.
- **Minibatch sampling on device**: `curand_uniform` for index selection + `gather` for column extraction from the CSC. Zero host roundtrip per minibatch.
- **ZINB NLL** is three terms: zero-inflation logit + log-NB + zero-mixture. Compute in fp32 with `__expf` / `__logf` intrinsics; no fp64.

## 6. Numerical stability plan

- **log1p, log_sigmoid, softplus** for numerical stability where needed (scipy-style).
- **Clamping**: `log σ_z² ∈ [-10, 10]` to prevent degenerate posteriors. `theta ∈ [1e-6, 1e6]`.
- **KL weight clipping**: the warmup schedule starts at 0 and linearly ramps to 1; numerical ELBO can temporarily diverge during warmup if gradients saturate. Clip total gradient norm at 100.
- **fp32 everywhere**; no fp64. Gradient accumulation uses Kahan summation when the batch loss magnitude is above 1e4 (rare but documented).

## 7. Autonomy pass (Rule 31)

| Config field | Auto-tune strategy |
|---|---|
| `latent_dim` | Auto: **reconstruction-ELBO elbow**. Train short (50 epochs) at latent_dim ∈ {5, 10, 20, 50}; pick the smallest where val ELBO plateaus (within 1%). Amortizable via warm-start across the sweep. |
| `n_hidden` | Auto: `128 for n_cells ≤ 20k, 256 for 20k–200k, 512 for >200k`. |
| `n_layers` | Auto: `1 for n_cells ≤ 10k, 2 for 10k–100k, 3 for >100k`. |
| `batch_size` | Auto: `max(128, free_dev_mem / per_sample_footprint_bytes)` where per-sample footprint = `n_genes × 4 + n_hidden × 4 × n_layers × 2`. |
| `n_epochs` | Auto: early stop on val ELBO plateau — stop when the last 20 epochs improve by < 1% of total. Capped at 1000. |
| `kl_warmup_epochs` | Auto: `n_epochs × 1.0`. |
| `learning_rate` | Auto: 1e-3 (scvi-tools default, rarely needs tuning). |
| `gene_likelihood` | Auto: test `log(mean / var)` distribution on device — if most genes show dispersion > Poisson (variance > mean), use NB; if excess zeros are detected via a on-the-fly zero-frequency vs Poisson comparison, use ZINB. |
| `dispersion` | Auto: `GENE` for `n_batches=1`, `GENE_BATCH` for `n_batches>1`. |
| `library_prior` | Auto: `OBSERVED` (scvi-tools default; `LEARNED` is rarely better in practice). |

No-args `scvi::train(counts)` returns a ScviResult with every dial auto, autotune decisions recorded in metadata.

## 8. Out-of-core streaming contract (Rule 14)

Billion-cell scVI training:
- Minibatches stream from `.1pz` shards via the streaming driver (feature 16).
- Model weights stay resident — they're tiny (for `latent_dim=10, n_hidden=128, genes=2000, n_layers=1` the full parameter count is ~550k × 4 bytes = 2.2 MB).
- Checkpoint weights to disk every 10 epochs for crash recovery.
- The minibatch sampler needs the full index set across all shards — maintain a global cell counter and a shard lookup table.

Document here; implement when feature 16 + the full model lands.

## 9. Determinism contract

- Default: non-deterministic via atomic grads in embedding layers + curand thread ordering.
- `deterministic=true` opt-in: segmented scan for all gradient accumulations, deterministic curand via a single block for the reparam noise (slow, documented).
- Seed always exposed in `random_seed`. Setting seed + deterministic=true gives bit-exact reproducibility across runs on the same hardware.

## 10. Correctness test spec

`tests/models_scvi_correctness.cpp`:
1. Train our scVI on GSM4037629 HVG-2000 for 100 epochs with `random_seed=42, deterministic=true`.
2. Python subprocess runs scvi-tools with matching hyperparameters + `seed_everything(42)`.
3. Compare:
   - Final ELBO: relative error ≤ 5% (scvi-tools Adam is slightly different than ours).
   - Latent space geometry: mean pairwise distance correlation ≥ 0.95 between our latent and scvi-tools latent.
   - scIB iLISI on a 2-batch subset: our > scvi-tools × 0.95.
4. Smoke test: `scvi::train(counts)` no-args variant succeeds on a 500-cell synthetic with default config, produces a non-degenerate latent (var > 0 on all dims).

## 11. Phase D+E dispatch spec

This is a large feature. Split into sub-cycles:

- **Sub-cycle 1**: `encoder.h` + `decoder.h` + `loss_zinb.h` + unit tests for forward pass.
- **Sub-cycle 2**: Hand-written backward pass + `optimizer_adam.h` + unit tests for gradient checking against numerical finite differences.
- **Sub-cycle 3**: `minibatch_sampler.h` + `trainer.h` + end-to-end training test on the 500-cell synthetic.
- **Sub-cycle 4**: GSM4037629 HVG-2000 real training + scvi-tools correctness comparison.
- **Sub-cycle 5**: Phase E bench vs scvi-tools PyTorch CUDA + Jax modes. Frontier promotion decision.

Total estimated: 5 cycles at ~1 day each.

## 12. Open questions

- **cuBLASLt epilogue support on sm_70 (V100S)**: some epilogue fusions are Ampere+ only. If V100S doesn't support `relu+dropout` epilogue, we pay one extra launch per layer. Measure during sub-cycle 1.
- **Does scvi-tools's Adam implementation match ours?** They use PyTorch Adam which has a slightly different bias-correction order. Document the difference; expected ELBO gap is ≤ 0.5%.
- **Does the warm-start latent_dim sweep actually beat a fixed-k training?** Needs empirical validation on Sub-cycle 4. If warm-start doesn't help, fall back to fixed `latent_dim=10` default.
- **Memory peak for the backward pass**: hand-written backprop may need activation caching. For 4 layers × batch=128 × n_hidden=256 × 2 (pre + post activation) = ~1 MB per minibatch. Negligible vs the model weights.

## 13. Links

- scvi-tools: https://github.com/scverse/scvi-tools
- Lopez, Regier et al. 2018 "Deep generative modeling for single-cell transcriptomics"
- Xu et al. 2021 "scANVI: probabilistic harmonization and annotation"
- scIB metrics: Luecken et al. 2022 "Benchmarking atlas-level data integration"
- Mandate v2 §B: `CLAUDE.md` search for "In-house foundation models"
- Rule 27: `CLAUDE.md` search for "Foundation models are bare-metal CUDA"
