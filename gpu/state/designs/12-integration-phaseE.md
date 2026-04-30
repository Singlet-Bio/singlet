---
feature: 12
module: integrate/harmony.h, integrate/bbknn.h, integrate/scvi_lite.h, integrate/metrics.h
cycle: 70 or later
phase: E (benchmark) + Rule 30 novel pursuit + Rule 31 autonomy
extends: 12-integration.md
status: draft
depends_on: feature 4 (PCA), feature 6 (kNN), models-scvi (scVI backbone)
---

# Feature 12 — Batch integration Phase E (Harmony + BBKNN + scVI-lite)

Feature 12 owns batch integration. Three backends: Harmony (fast iterative), BBKNN (graph-based), scVI (deep generative — shares the models/scvi backbone). This doc specs all three + the scIB metrics evaluation framework.

## 1. Success metrics

On a 5-sample concat where samples are from different GEO studies (synthetic batches from GSM4037629 + 4 peers):

- **Harmony wall p50**: ≤ `harmonypy` wall × 0.1 + ≤ `rapids-singlecell.harmony_gpu` × 0.5.
- **BBKNN wall p50**: ≤ `bbknn` Python wall × 0.1.
- **scVI-lite wall per epoch**: already covered in models-scvi-phaseE; this doc composes it with the integration entry point.
- **scIB correctness** (iLISI, cLISI, ASW, kBET, ARI of Leiden on integrated embedding):
  - Our Harmony ≥ harmonypy × 0.95 on each metric.
  - Our BBKNN ≥ bbknn × 0.95.
  - Our scVI-lite ≥ scvi-tools × 0.95.

Frontier gate: dominance on wall for all three backends, correctness match on all five scIB metrics, Rule 31 autonomy within 10%.

## 2. SOTA baselines

| Baseline | Path | Notes |
|---|---|---|
| **harmonypy** | Python | CPU reference for Harmony |
| **rapids-singlecell harmony_gpu** | cupy | Primary GPU competitor |
| **bbknn** | Python | CPU BBKNN reference |
| **scvi-tools** | PyTorch CUDA | scVI baseline (covered in models-scvi) |
| **scanorama** | Python | Alternative integration, correctness sanity check |
| **Seurat IntegrateData (v4 / v5)** | R / C++ | R gold-standard integration |
| **LIGER** | R | Matrix-factorization integration, correctness comparison |

## 3. Bench configurations

Scales:
- **small**: 5-sample GSE127918 concat (~100k cells, 5 batches) — primary competitive scale
- **medium**: 20-sample synthetic concat (~500k cells, 20 batches)
- **large**: 100-sample synthetic (~5M cells, 100 batches)

Configurations:

1. **`ours_harmony_trust_region`** — Rule 30 novel (§4a)
2. `ours_harmony_iterative` — iterative k-means reference port
3. **`ours_bbknn_soft_neighbor`** — Rule 30 novel (§4b)
4. `ours_scvi_lite` — thin integration wrapper over models/scvi
5. `ours_auto` — Rule 31 backend-picking
6. `harmonypy`
7. `rapids_singlecell_harmony_gpu`
8. `bbknn_python`
9. `scvi_tools` — covered in models-scvi-phaseE bench
10. `seurat_integrate` — R subprocess, small scale only

Metrics: wall p50, peak dev mem, **full scIB suite** per run (iLISI + cLISI + ASW + kBET + ARI).

## 4. Novel pursuit (Rule 30)

### 4a. One-shot Harmony via trust-region optimization

**Standard Harmony**: alternating k-means on the PCA embedding + soft cluster-batch assignment + linear correction. Iterates ~10–20 rounds until convergence. Each round's k-means is the cost driver.

**The trust-region observation**: Harmony's objective (minimize cluster-batch entropy while preserving cluster-embedding fidelity) is a quadratic in the correction term once the cluster assignments are fixed. Instead of re-running k-means every iteration, we can:

1. Run k-means ONCE to get an initial cluster assignment.
2. Compute the analytic trust-region step: `Δembedding = -H^(-1) g` where `H` is the Hessian of the Harmony loss and `g` is its gradient. In Harmony's formulation, H is block-diagonal per cluster and trivially invertible.
3. Update embedding + re-compute assignments ONCE more.
4. Take a final corrective step. Done — 3 passes instead of 20.

**Expected wall**: ~10× over the iterative Harmony. Combined with GPU k-means (cuml / cuVS) the wall drops further.

**Gate**: iLISI ≥ harmonypy × 0.95, cLISI ≥ harmonypy × 0.98, wall ≤ harmonypy × 0.1.

**Risk**: trust-region breaks down when the batch effect is very large (dominates the cluster signal). Detect via initial cluster-batch entropy; fall back to iterative Harmony when `entropy < 0.3 × max_possible`.

### 4b. Soft-neighbor BBKNN

**Standard BBKNN**: for each cell, compute the kNN restricted to each other batch separately, then union the per-batch kNNs as the cell's neighbor set. Produces a graph where each cell has `k × n_batches` neighbors.

**Soft alternative**: weight the per-batch kNN contributions by a **batch-relevance score** estimated from cell-cluster probability × batch confusion. Batches that are "similar" to the query cell contribute more neighbors; batches that are "different" contribute fewer. Not a hard k-nearest-per-batch, but a soft allocation summing to `k`.

Published as "ridge BBKNN" in some recent integration papers. Our contribution: on-device computation in one kernel pass, no pre-clustering needed.

**Expected wall**: ~2× BBKNN (we do more work per cell) BUT produces better cLISI on heterogeneous batch setups where some batches are more "alike" than others. The wall cost may be worth it; measure.

**Gate**: cLISI ≥ BBKNN × 1.05 (we need IMPROVEMENT to justify the wall cost), wall ≤ BBKNN × 3.

### 4c. scIB metrics on device

The scIB evaluation suite (iLISI, cLISI, ASW, kBET, ARI, graph connectivity, NMI) is normally computed in Python with scipy/scikit-learn. Each metric is a reduction on a known graph + labels.

**GPU port**: implement every scIB metric as a fused device kernel taking `(embedding, batch_labels, cell_type_labels)` and returning the scalar metric. One kernel per metric. All metrics batched together in `ours_scib::evaluate()` returning the full tuple.

**Expected wall**: ~100 ms for the full suite on 100k cells vs ~30 sec in Python. 300× headline. Useful for Phase F frontier decisions on every integration cycle AND as a standalone tool for users.

## 5. Autonomy pass (Rule 31)

`integrate::run(embedding, batch_labels)` returns integrated embedding with auto-picked backend.

| Config | Auto |
|---|---|
| `backend` | Auto: `n_batches ≤ 5 → harmony_trust_region`, `n_batches 5–50 → bbknn_soft`, `n_batches > 50 → scvi_lite`. Also: if cell type labels are supplied, scANVI (not specced here) would be preferred — for now just use scvi_lite. |
| `harmony_theta` | Auto: 2.0 (harmonypy default). |
| `harmony_max_iter` | Auto: 3 (with trust-region novel path). |
| `bbknn_neighbors_within_batch` | Auto: `max(3, 15 / n_batches)`. |
| `scvi_latent_dim` | Auto: delegated to scvi autotune (10 default). |
| `scvi_n_epochs` | Auto: delegated to scvi autotune. |
| `scib_evaluate` | Auto: true (always run the metrics on the result; users can ignore if they don't need them). |

## 6. OOC streaming contract

- **Harmony** is embedding-space only, memory small (n_cells × n_pcs × fp32). Works streaming if embedding is pre-computed (upstream PCA handles streaming).
- **BBKNN** needs full graph residency — not streaming-friendly. Fall back to chunked integration where each chunk is integrated separately then stitched via cross-chunk kNN correction.
- **scVI** is streaming-native via feature 16's ScviEpochStage.

Document; most users integrate at small-to-medium scale so OOC is lower priority.

## 7. Determinism contract

- Trust-region Harmony is deterministic up to the initial k-means seed.
- BBKNN is deterministic up to tie-break in the kNN phase.
- scVI is deterministic with `deterministic=true` (models-scvi spec).
- scIB metrics are deterministic.

## 8. Phase E dispatch spec

Dispatch two workers in parallel:

**Worker A (gpu-kernel-dev)**: implement trust-region Harmony + soft-neighbor BBKNN + thin scvi-lite wrapper + scIB metric kernels.

**Worker B (gpu-bench)**: run 10-config × 3-scale benchmark with full scIB evaluation per config. Write to benchmark-registry.md, pareto-frontier.md, novel-attempts.md.

## 9. Open questions

- **Is the trust-region Harmony theoretically justified?** Harmony's original paper uses soft k-means + correction; the trust-region view is a reformulation but needs empirical validation that the second-order step actually reaches the same local optimum as the iterative path.
- **Soft-neighbor BBKNN vs Harmony on the same data**: which produces better scIB scores? Harmony usually wins on iLISI; BBKNN sometimes wins on cLISI. Bench both, let Rule 31 auto-pick per dataset shape.
- **Seurat integration on 100k cells**: will run for hours. Cap at 5k-cell subsample for R baseline; note extrapolation.
- **100-batch synthetic dataset**: needs scaffolding. Consider generating from GSM4037629 with per-batch gene-expression perturbations.

## 10. Links

- Original: `state/designs/12-integration.md`
- Korsunsky et al. 2019 Harmony paper
- Polański et al. 2020 BBKNN
- scvi-tools: https://github.com/scverse/scvi-tools
- Luecken et al. 2022 scIB metrics benchmark
- rapids-singlecell harmony_gpu: https://github.com/scverse/rapids_singlecell
