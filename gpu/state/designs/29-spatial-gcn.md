---
feature: spatial_gcn
roadmap_id: 29
module: include/singlet-gpu/spatial/stagate.h + python/singlet_gpu/spatial/stagate.py + r/R/spatial.R
status: design
tolerance: spatial domain ARI ≥ 0.85 vs STAGATE_pyG on the DLPFC benchmark; embedding kNN-preservation@15 ≥ 0.80
target_perf: 100k spots × 30k genes × 10 epochs ≤2 min on A100 (STAGATE_pyG ~10 min for the same)
ooc_plan: spot-tile training (process N=10k spot tiles per forward/backward pass, accumulate gradients across tiles)
---

## Why this exists

Cycle 28 (CSI-GEP) is in flight. Cycle 29 is the third Phase B feature — and the **first spatial transcriptomics feature** in singlet-gpu. Per cycle 28 lit-scout:

1. **GraphST** is #1 for spatial domain segmentation (NAR 2025 benchmark, ARI 0.552), but it relies on contrastive learning which is harder to implement from scratch.
2. **STAGATE** is #2 (ARI 0.515) and the cleaner architecture: graph attention auto-encoder. The PyG variant is 10× faster than the original TensorFlow.
3. **Existing reference impls are all PyTorch** (PyG). No native CUDA library to wrap.

**Decision**: cycle 29 implements **STAGATE natively in CUDA** (similar to cycle 27 Cell2fate's manual gradient approach). Graph attention is sparse SpMM + softmax + dense MLP — all kernels we already understand. It's another "first GPU implementation" candidate (the cuVS/cuGraph spatial pieces don't yet ship a GAT autoencoder).

## Algorithm — STAGATE (Dong & Zhang 2022)

```
Input:  X (n_spots × n_genes), spatial_coords (n_spots × 2)
Output: spot_embedding (n_spots × d_embed), spatial_domain[n_spots]

1. Build kNN graph on spatial coordinates (k=6 default, like Visium hexagonal lattice).
2. Encoder layers (2 GAT layers):
     for layer in [1, 2]:
         attention coefficients α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
         h_i' = sum_j α_ij W h_j
         h_i = h_i'
3. Latent embedding: z_i = h_i^(2) (length d_embed = 64 typical).
4. Decoder layers (mirror of encoder):
     symmetric reconstruction X_hat
5. Loss: MSE(X, X_hat) + λ * (optional contrastive term).
6. Optimization: Adam, ~500 epochs.
7. Post-training: cluster z via mclust / Leiden → spatial_domain labels.

The "graph attention" trick: each spot attends only to its k spatial neighbors, with learned attention weights. This combines spatial proximity with feature similarity.
```

## GPU implementation strategy

Native CUDA. Reuse cycle 8 kNN (for the spatial graph) and cycle 9 leiden (for the post-clustering).

### Layer kernels

**`gat_attention_forward`** (one block per spot):
- For spot i, gather k neighbor embeddings via the CSR graph indices (cycle 8 KnnResult format).
- Compute attention scores `s_ij = LeakyReLU(a^T [W·h_i || W·h_j])` for j ∈ neighbors(i).
- Softmax over j → α_ij.
- Aggregate: `h_i' = Σ_j α_ij · W · h_j`.
- Cost: O(k · d) per spot, parallel across spots. For n=100k, k=6, d=64: ~38M ops, trivial on GPU.

**`gat_attention_backward`**:
- Gradient w.r.t. W (linear layer weights), `a` (attention vector), and inputs `h`.
- Use `cub::DeviceSegmentedReduce` for the per-neighbor sum.
- Manual gradient via the chain rule (similar to cycle 17 + cycle 27 patterns).

**`mlp_decoder_forward/backward`**:
- Standard dense linear layers via cuBLAS GEMM.

**`adam_step`**:
- Same as cycle 27 Cell2fate Adam optimizer.

### Loss + convergence

MSE reconstruction loss on `X` vs `X_hat`. Compute via `cub::DeviceReduce::Sum`. One scalar D2H per epoch (≤500 epochs, well within the rule preamble exception).

### Post-training clustering

After STAGATE training, the latent embedding `z` is fed to `singlet_gpu::graph::leiden` (cycle 9 adapter) for domain labels. Reuse, no new code.

## Numerical stability

- fp32 throughout. Adam state in fp32.
- Softmax with max-subtraction trick (cycle 12 pattern reused).
- LeakyReLU is bit-exact in fp32.
- MSE loss computed in fp32 sum + fp64 accumulator (cycle 27 pattern).

## Memory layout

- Input: X dense (n × m × 4) — for 100k × 30k = 12 GB. **Too big for fp32 dense.**
- **Critical**: use the SPARSE input as-is. STAGATE's input matrix is the gene expression — keep as sparse CSC, do `cusparseSpMM` for the first GAT layer's dense projection.
- Encoder hidden states: `h^(1) (n × d_hidden=256)`, `h^(2) (n × d_embed=64)`. For 100k spots: ~100 MB + 25 MB.
- Attention weights per epoch: not stored (recomputed per forward).
- Adam state: 2× the parameter count (~10 MB at typical sizes).
- Decoder reconstruction: dense `n × m × 4` — too big! **Must compute reconstruction loss only at sparse input positions** (mask out zeros). This is a key optimization vs the naive dense decoder.
- Total at 100k spots: ~150 MB workspace.

## Streams

One stream, caller-provided. Per-epoch forward + backward chain on the same stream.

## Out-of-core

Spot-tile training: process N=10k spot tiles, accumulate gradients across tiles, apply Adam update once per epoch. The kNN graph is global (small) so no per-tile overhead. For very large slices (>1M spots, e.g., Xenium HD), tile-based training is mandatory.

## Determinism

cuRAND Philox for layer init weights. Adam updates are deterministic given fixed init.

## Correctness test spec

Test: `tests/spatial_stagate_correctness.cpp`.

Reference: STAGATE_pyG (Python) via subprocess.

Test cases:
1. **`Stagate_TinySynthetic_VsPyG`**: synthetic 200-spot dataset with 4 planted spatial domains. Run our STAGATE and STAGATE_pyG. Compare spatial domain ARI ≥ 0.85.
2. **`Stagate_DLPFC_VsBenchmark`**: load a small DLPFC-style dataset (Visium hexagonal grid). Compare to the published STAGATE result on the same dataset.
3. **`Stagate_EmbeddingShape`**: confirm output `spot_embedding` shape `[n_spots, d_embed]`.
4. **`Stagate_Determinism_BitIdentical`**: bit-identical with fixed seed.
5. **`Stagate_GradientCheck_FiniteDiff`**: per-parameter analytical gradient matches finite-difference within rel_err ≤ 1e-3 (cycle 27 pattern).
6. **`Stagate_PostTrainingLeiden`**: confirm cycle 9 leiden integration produces meaningful domain labels.

Tolerances per design doc.

## Target performance

| Scale | Spots | Genes | Epochs | Wall (target) | STAGATE_pyG (CPU baseline) |
|---|---|---|---|---|---|
| tiny | 200 | 100 | 100 | <5s | ~30s |
| 10k | 10,000 | 30k | 500 | <30s | ~5min |
| 100k | ~100k | 30k | 500 | <2min | ~10min |
| 1M (Xenium HD) | ~1M | 30k | 500 | <20min | not feasible |

## Implementation notes

- Header path: `include/singlet-gpu/spatial/stagate.h` (~800 LOC).
- Python wrapper: `python/singlet_gpu/spatial/stagate.py` (~250 LOC) — adds `stagate(adata, n_neighbors=6, d_embed=64, n_epochs=500, ...)`.
- R wrapper: `r/R/spatial.R` (~150 LOC) — adds `run_stagate(sce, ...)`.
- New module path: `include/singlet-gpu/spatial/` (NEW; first feature in this subtree).
- Build flag: `FACTORNET_HAS_GPU=1`. cuBLAS + cub + cuRAND + cuSPARSE.
- Dependencies: cycle 8 (kNN for spatial graph construction), cycle 9 (leiden for post-clustering).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU STAGATE; manual CUDA gradients vs PyG auto-diff)` first comment.

## Risks

1. **Manual CUDA GAT gradients** are non-trivial. Use the cycle 27 finite-difference test as the gate.
2. **Sparse decoder reconstruction** (compute MSE only at nonzero positions) requires a custom kernel; fall back to dense reconstruction if it gets too complex (memory blow-up).
3. **Spatial coordinates format**: cycle 29 expects `spatial_coords.parquet` with columns `(barcode, x, y)` per singlify Visium output. Xenium / MERSCOPE may have different column names — abstract behind a `read_pz_spatial_sce()` helper.
4. **Hyperparameter tuning** (lit-scout pitfall #2): k, d_embed, learning rate vary across datasets. Expose all as `StagateConfig` fields with sensible defaults.
