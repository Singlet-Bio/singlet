---
feature: discrete_diffusion
roadmap_id: 30
module: include/singlet-gpu/generative/discrete_diffusion.h + python/singlet_gpu/generative/diffusion.py + r/R/diffusion.R
status: design
tolerance: per-gene marginal Wasserstein-1 distance ≤ 0.1 vs reference Python; per-cluster cell count rel_err ≤ 5%
target_perf: 100k cells × 30k genes train ≤4h on A100; inference (1k synthetic cells) ≤30s
ooc_plan: gene-batched training; per-cell minibatching naturally chunks
---

## Why this exists

Cycle 29 lit-scout identified **Discrete Diffusion Models for Single-Cell Gene Expression** (bioRxiv Feb 2026) as the top cycle 30 candidate:

1. **No existing GPU-first implementation** in the single-cell field — only PyTorch CPU references exist. **Fifth "first GPU implementation in the field"** candidate after GSEA, MT lineage, donor pseudobulk, Cell2fate.
2. **Native count data modeling** — operates directly on integer counts, no continuous relaxation overhead. Perfect for our `.1pz` input.
3. **High-value applications**: synthetic population generation, batch effect simulation, rare-cell augmentation for downstream model training.
4. **Greenfield**: nobody has shipped this on GPU yet.

This is a STRETCH cycle — diffusion models are non-trivial. We implement a minimum-viable discrete diffusion model with absorbing-state forward process and a transformer denoiser, then validate against the reference Python implementation.

## Algorithm

Discrete diffusion with absorbing state (D3PM-style for count data):

```
Forward process (cell → noisy sequences):
  At time t=0: x_0 = original gene expression vector (integer counts)
  At each step: each gene independently transitions to MASK with probability β_t (linearly increasing schedule)
  After T steps: most genes are MASK; recovers a "noised" cell

Reverse process (denoiser):
  A transformer-style network predicts p(x_0 | x_t, t) for each cell
  Trained to maximize the variational ELBO of the forward + reverse trajectory

Training:
  For each cell j and timestep t in [1, T]:
    Sample x_t by masking x_0 with prob 1 - prod(1 - β_s for s in 1..t)
    Compute predicted x_0 hat = denoiser(x_t, t)
    Loss = cross-entropy(x_0 hat, x_0) at masked positions
    Backprop, Adam update

Inference (sample new cells):
  Start with all-MASK cell
  For t in [T, T-1, ..., 1]:
    x_{t-1} = denoiser(x_t, t)  (sample from the predicted distribution)
  Return x_0
```

For count data, we discretize gene expression into a finite vocabulary (e.g., 0, 1, 2, 4, 8, 16, ..., 2^N). Counts are tokenized; the denoiser predicts the next token per gene.

### Denoiser architecture

A small transformer-style encoder operating on a sequence of `(gene_id, gene_expression_token, timestep_embedding)`. For 30k genes, full attention is O(n²) — infeasible. Use sparse attention:

- Per-cell, attention over only the non-MASK genes (typical ~5-10% of genes).
- Hidden dim 256, 4 attention heads, 4 layers.
- Final layer projects to vocab size (e.g., 16 tokens for power-of-2 binning).

This is a minimal architecture; the reference Python uses larger transformers. We document this as a singlet-gpu trade-off (smaller model, faster training, similar marginals).

## GPU implementation strategy

Native CUDA. The transformer pieces use cuBLAS GEMM for QKV projections + softmax kernels. No PyTorch.

### Layer kernels

- **Token embedding lookup** (one warp per gene token): trivial scatter via cuBLAS.
- **Self-attention** (one block per cell): QKV projection via Sgemm, softmax with max-subtraction (cycle 12 pattern), output projection via Sgemm.
- **Feed-forward**: 2-layer MLP with GELU activation.
- **Layer norm**: per-cell mean+var via Welford (cycle 4 pattern).
- **Cross-entropy loss + gradient**: standard, fp32 hot path with fp64 accumulator (cycle 27 pattern).
- **Adam optimizer**: cycle 27 reuse.

### Forward process (masking)

Per cell, deterministic given seed. Use cuRAND Philox to draw per-gene masks. The masked tokens are replaced by a single MASK_ID.

### Inference (sampling)

Per cell, T sequential denoising steps. Each step is a forward pass through the denoiser. For 1k synthetic cells × T=100 steps × 4 layers: ~400k forward passes — fits in seconds on A100.

## Numerical stability

- fp32 hot path. fp64 loss accumulator.
- Softmax with max-subtraction.
- Layer norm with Welford two-pass (cycle 4 pattern).
- Cross-entropy from logits via the log-sum-exp trick.
- Adam state in fp32.

## Memory layout

- Input: `DeviceCSC` (m × n, fp32 → tokenized to int).
- Tokenized cells: dense `n × m × 1 byte` (assuming vocab ≤256). For 100k × 30k: 3 GB. **Too big**.
- **Sparse storage**: keep only non-zero gene tokens per cell. Average ~3000 non-zero genes per cell → 100k × 3000 × 1 byte = 300 MB. Manageable.
- Denoiser parameters: O(d² × n_layers) — for d=256, n_layers=4: ~1 MB.
- Adam state: 2× the parameter count.
- Total: ~500 MB workspace at 100k cells.

## Streams

One stream, caller-provided. SVI minibatching.

## Out-of-core

Cell minibatching naturally chunks. Gene tokenization is per-cell, no global state.

## Determinism

cuRAND Philox seeded for masking + sampling. Adam updates are deterministic.

## Correctness test spec

Test: `tests/generative_discrete_diffusion_correctness.cpp`.

Reference: Discrete Diffusion Python (bioRxiv reference impl) via subprocess.

Test cases:
1. **`Diffusion_TinySynthetic_TrainsAndSamples`**: 200 × 100 synthetic input. Train for 100 epochs. Sample 50 synthetic cells. Confirm: train loss decreases, sample shape correct, no NaN.
2. **`Diffusion_GSM4037629_RealData_PerGeneMarginal`**: load exon_counts, train, sample 1000 synthetic cells. Compare per-gene marginal Wasserstein-1 distance to original ≤ 0.1 (top 100 most-expressed genes).
3. **`Diffusion_PerClusterPreservation`**: synthetic 3-cluster input, confirm sampled cells preserve cluster proportions within ±5%.
4. **`Diffusion_Determinism_BitIdentical`**: bit-identical with fixed seed.
5. **`Diffusion_GradientCheck_FiniteDiff`**: per-parameter analytical gradient matches finite-difference (cycle 27 pattern).
6. **`Diffusion_VsPythonReference_TopGenes`**: same input, our model + Python reference. Compare top-50 generated genes Spearman ρ ≥ 0.85 (allow for randomness across implementations).

Tolerances per design doc.

## Target performance

| Scale | Cells | Genes | Train epochs | Wall (target) | Reference Python (CPU) |
|---|---|---|---|---|---|
| tiny | 200 | 100 | 100 | <30s | ~5min |
| 10k | 11,560 | 30k | 200 | <20min | ~4h |
| 100k | ~120k | 30k | 200 | <4h | ~24h+ |

## Implementation notes

- Header path: `include/singlet-gpu/generative/discrete_diffusion.h` (~1200 LOC).
- New module path `singlet-gpu/generative/` (NEW).
- Python wrapper: `python/singlet_gpu/generative/diffusion.py` (~250 LOC).
- R wrapper: `r/R/diffusion.R` (~150 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuBLAS + cuRAND + cub.
- Dependencies: cycle 1 (core), cycle 2 (loader).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU discrete diffusion for single-cell counts)` first comment.

## Risks

1. **Diffusion model complexity** is the highest of any feature so far. Manual gradients via the transformer chain rule. Use cycle 27 + cycle 29 patterns. Validate via finite-difference test.
2. **Tokenization choice** (linear vs log binning, vocab size) affects sample quality. Default to log2 binning (vocab=16) per the reference paper.
3. **Reference Python install** is heavy. Make optional.
4. **Sparse attention masking**: needs careful indexing. Test with tiny inputs first.
5. **Time budget**: this is the most ambitious cycle in singlet-gpu history. May need to defer to a 2-cycle implementation if scope balloons.
