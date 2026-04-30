---
feature: perturb_graph
roadmap_id: 32
module: include/singlet-gpu/perturbation/perturb_graph.h + python/singlet_gpu/perturbation/cpa.py + r/R/perturbation.R
status: design
tolerance: per-gene predicted vs observed Spearman ρ ≥ 0.85 on held-out perturbations; cell-level cosine similarity ≥ 0.80
target_perf: 100k cells × 5k HVG × 20 perturbations training ≤4h on A100; inference (predict 1 perturbation × 1k cells) ≤10s
ooc_plan: per-perturbation minibatch streaming
---

## Why this exists

Cycle 31 lit-scout identified **PerturbGraph / CPA-style perturbation response prediction** as the top cycle 32 candidate:

1. **Massive community demand**: 150+ recent perturbation methods papers, no unified GPU baseline exists.
2. **First GPU implementation** in the field — seventh "first GPU" candidate after GSEA, MT lineage, donor pseudobulk, Cell2fate, STAGATE, discrete diffusion.
3. **Aligned with singlify outputs**: exon counts (training), donor labels (cross-donor robustness), GEO metadata (perturbation labels embedded by singlify).
4. **CRISPR screen analysis** is one of the highest-impact downstream applications.

## Algorithm — Compositional Perturbation Autoencoder (CPA) + Graph variant

CPA (Lotfollahi et al. Nat Methods 2023) decomposes cell expression into:
1. **Cell-state latent** (cell type / batch / lineage)
2. **Perturbation latent** (additive to cell-state; per perturbation × dose)
3. **Reconstruction**: decoder maps (cell-state + perturbation_latent) → expression

PerturbGraph (bioRxiv 2026) extends CPA with a GNN over a perturbation similarity graph for **unseen perturbation prediction**.

```
Inputs: counts (n_cells × n_genes), perturbation_labels[n_cells], dose[n_cells], (optional) gene_KG_graph
Outputs: predicted expression for novel (perturbation, dose) pairs

Architecture:
  encoder: 3-layer MLP, n_genes → 256 → 128 → 64 (cell-state latent z_c)
  perturbation_embedding: lookup table, n_perts × 64 (latent z_p[p])
  dose_modulator: MLP(z_p, dose) → modulated z_p_dose
  decoder: 3-layer MLP, (z_c + z_p_dose) → 256 → n_genes
  
  loss = NB log-likelihood(observed, predicted) + KL(z_c || N(0,I)) + adversarial loss to disentangle z_c from perturbation
  
  PerturbGraph extension: z_p[p] is computed via GNN over perturbation graph
                          (genes → perturbed → cell type), allowing unseen perturbations
                          to inherit embeddings from their graph neighbors.
```

## GPU implementation strategy

Native CUDA. Manual gradients via the chain rule (cycle 27/29/30 patterns).

### Layer kernels

- **Encoder MLP** (3 layers): cuBLAS Sgemm for linear, custom GELU activation kernel.
- **Perturbation embedding lookup**: scatter-gather kernel.
- **Dose modulator**: small MLP, cuBLAS.
- **Decoder MLP**: same as encoder (mirror).
- **NB log-likelihood + gradient**: reuse cycle 17 donor pseudobulk pattern.
- **Adversarial loss**: per-cell discriminator (small MLP), GRL (gradient reversal layer) for adversarial training.
- **GNN over perturbation graph** (for PerturbGraph variant): cuSPARSE SpMM for graph propagation; 1-2 layers of standard GCN.

### Training loop

- Adam optimizer (cycle 27 reuse), 200 epochs typical.
- Per-epoch loss readback (4 bytes, approved exception).

## Numerical stability

- fp32 hot path; fp64 NB log-likelihood accumulator (cycle 17 pattern).
- GELU activation: standard tanh-based approximation.
- Adversarial loss can be unstable — clamp gradient norm.

## Memory layout

- Input: counts CSC (n × m).
- Latents: `z_c[n × 64]`, `z_p[n_perts × 64]`.
- MLP parameters: ~5 MB total (small).
- Decoder reconstruction: dense `n × m` — too big! Use sparse-masked reconstruction (cycle 29 STAGATE pattern).
- Adam state: 2× the parameter count.
- Total at 100k cells × 5k HVG × 20 perts: ~200 MB workspace.

## Streams

One stream, caller-provided.

## Out-of-core

Per-perturbation minibatch streaming. The model parameters are global (kept on device), the data is chunked.

## Determinism

cuRAND Philox seeded for layer init + minibatch order.

## Correctness test spec

Test: `tests/perturbation_perturb_graph_correctness.cpp`.

Reference: CPA Python (`pip install cpa-tools`) via subprocess.

Test cases:
1. **`PerturbGraph_TinySynthetic_VsCPA`**: 200 cells × 100 genes × 5 perturbations synthetic. Compare per-gene predicted vs CPA Python: Spearman ρ ≥ 0.85.
2. **`PerturbGraph_HeldOutPerturbation_Smoke`**: train on 4 of 5 perturbations, predict the 5th. Confirm finite predictions, no NaN.
3. **`PerturbGraph_DoseResponse_Monotone`**: synthetic dose-response. Confirm predicted expression interpolates monotonically with dose.
4. **`PerturbGraph_Determinism_BitIdentical`**: bit-identical with fixed seed.
5. **`PerturbGraph_GradientCheck_FiniteDiff`**: critical gradient validation.
6. **`PerturbGraph_AdversarialLoss_Disentangle`**: confirm cell-state latent doesn't encode perturbation identity (classifier accuracy on z_c → perturbation < 60% — random for 5 classes is 20%).
7. **`PerturbGraph_GNNExtension_UnseenPerturbation`**: train on 4 perts, use GNN to predict the 5th from its graph embedding. Confirm prediction quality vs random baseline.

Tolerances per design doc.

## Target performance

| Scale | Cells | Genes | Perts | Wall (target) |
|---|---|---|---|---|
| tiny | 200 | 100 | 5 | <30s |
| 10k | 11,560 | 5k | 20 | <30min |
| 100k | ~120k | 5k | 50 | <4h |

## Implementation notes

- Header path: `include/singlet-gpu/perturbation/perturb_graph.h` (~2500 LOC).
- New module path `singlet-gpu/perturbation/` (NEW).
- Python wrapper: `python/singlet_gpu/perturbation/cpa.py` (~250 LOC).
- R wrapper: `r/R/perturbation.R` (~150 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuBLAS + cuSPARSE + cuRAND + cub.
- Dependencies: cycle 1, cycle 2 (loader). Optional: cycle 8 kNN for GNN graph construction.
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU CPA + PerturbGraph for unseen perturbation prediction)` first comment.

## Risks

1. **Adversarial training is unstable** — clamp gradients, lower learning rate, monitor adversarial loss.
2. **Manual backward through GNN** is non-trivial. Use the cycle 29 STAGATE GAT backward pattern as analog.
3. **CPA Python install** is heavy. Make optional.
4. **GNN extension is the secret sauce** — defer to a follow-up cycle if scope blows up. Ship CPA-only first.
5. **Perturbation labels** must be in `adata.obs['perturbation']` per CPA convention. Document.

## Decision: cycle 32 vs alternatives

Lit-scout's runners-up were spatial deconvolution (RCTD GPU) and DeepVelo. PerturbGraph wins because:
1. **Most novelty** — RCTD is a known algorithm; PerturbGraph is recent (2026 bioRxiv).
2. **Highest community demand** per the perturbation-methods explosion.
3. **Most unique to us** — singlify's donor + GEO metadata embedding makes us the natural place for perturbation-aware single-cell analysis.

Spatial deconvolution and DeepVelo become cycles 33-34 candidates.
