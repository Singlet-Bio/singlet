---
feature: doublet_detection
roadmap_id: 31
module: include/singlet-gpu/qc/doublet_score.h + python/singlet_gpu/qc/doublet.py + r/R/qc.R
status: design
tolerance: per-cell doublet score Spearman ρ ≥ 0.95 vs scrublet on the same input; ROC AUC ≥ 0.85 on synthetic ground truth
target_perf: 100k cells × 50 PCs doublet scoring ≤500ms on A100; 1M cells ≤5s
ooc_plan: per-batch kNN scoring (already streamable via cycle 8 kNN)
---

## Why this exists

Cycle 30 (discrete diffusion) is the most ambitious cycle so far. Cycle 31 is intentionally small-scope to balance the workload — ~500 LOC, reusing existing cycle 8 kNN infrastructure.

Doublet detection is a critical QC step that singlet-gpu currently lacks. Scrublet (Wolock 2019) and DoubletFinder (McGinnis 2019) are the consensus CPU methods. ScaleSC has a GPU version but it's only available within their pipeline. We provide a standalone GPU doublet scorer.

## Algorithm — Scrublet-style synthetic doublet scoring

```
Inputs: PCA embedding X (n_cells × n_pcs), original counts (optional, for synthetic generation)
Outputs: doublet_score per cell, doublet_call (binary, threshold tunable)

1. Generate synthetic doublets: pick N_synth random pairs of real cells, average their PCA embeddings.
2. Combine real + synthetic into a single point cloud (n_cells + n_synth points).
3. Run kNN (cycle 8) on the combined cloud, k=20.
4. For each real cell: compute the fraction of its k neighbors that are synthetic doublets.
5. doublet_score[i] = synthetic_neighbor_fraction[i].
6. Auto-threshold: pick a doublet score threshold via the "knee point" of the score histogram.
```

The intuition: real doublets cluster near synthetic doublets (both are mixtures of cell types), so they have high synthetic-neighbor fraction.

## GPU implementation strategy

Native CUDA. Reuses cycle 8 `compute_knn` for the kNN step.

### Kernels

**`generate_synthetic_doublets`** (one warp per synthetic pair):
- Sample two random cell indices via cuRAND Philox.
- Average their PCA vectors element-wise.
- Output: `synthetic_pca[N_synth × n_pcs]`.

**`combine_real_and_synthetic`** (trivial concat):
- Output: `combined_pca[(n_cells + N_synth) × n_pcs]` with synthetic at end.

**`compute_knn`**: cycle 8 reuse. Returns `KnnResult` for the combined cloud.

**`score_synthetic_neighbor_fraction`** (one thread per real cell):
- For each real cell i, walk its k neighbors.
- Count how many neighbor indices are ≥ n_cells (i.e., synthetic).
- `doublet_score[i] = synthetic_count / k`.

**`auto_threshold`** (single-block reduction):
- Build a histogram of doublet scores.
- Find the "knee" — first bin where the cumulative density drops below 5% (heuristic).
- Apply threshold to produce binary call.

## Numerical stability

- fp32 throughout. Doublet score is in [0, 1] — no precision concerns.
- Auto-threshold uses fp64 accumulator for cumulative density.

## Memory layout

- Input PCA: `n_cells × n_pcs × 4` bytes.
- Synthetic PCA: `N_synth × n_pcs × 4` bytes (default `N_synth = 0.25 * n_cells`).
- Combined: `(n_cells + N_synth) × n_pcs × 4` bytes.
- KnnResult: standard cycle 8 output.
- doublet_score: `n_cells × 4` bytes.

For 1M cells × 50 PCs: 200 MB embedding + 50 MB synthetic + 250 MB combined + ~200 MB kNN result. Total ~700 MB. Fine.

## Streams

One stream, caller-provided.

## Out-of-core

Doublet scoring is per-cell-independent — naturally batches. The kNN step is the bottleneck and is already chunkable via cycle 8.

## Determinism

cuRAND Philox seeded for synthetic doublet generation. Bit-identical with fixed seed.

## Correctness test spec

Test: `tests/qc_doublet_correctness.cpp`.

Reference: scrublet Python via subprocess.

Test cases:
1. **`Doublet_TinySynthetic_VsScrublet`**: 200-cell synthetic with planted 5% doublets. Compare doublet score Spearman ρ ≥ 0.95.
2. **`Doublet_GSM4037629_RealData`**: load → PCA → doublet score. Confirm finite results, doublet rate in [0, 20%].
3. **`Doublet_AutoThreshold_ROC`**: synthetic with planted doublets. Confirm ROC AUC ≥ 0.85.
4. **`Doublet_Determinism_BitIdentical`**: bit-identical with fixed seed.
5. **`Doublet_NSynth_Sensitivity`**: vary N_synth in [0.1, 0.25, 0.5] × n_cells. Confirm scores correlate ≥ 0.9 across settings.

## Target performance

| Scale | Cells | PCs | Wall (target) |
|---|---|---|---|
| tiny | 200 | 10 | <10ms |
| 10k | 11,560 | 50 | <50ms |
| 100k | ~120k | 50 | <500ms |
| 1M | ~1M | 50 | <5s |

## Implementation notes

- Header path: `include/singlet-gpu/qc/doublet_score.h` (~500 LOC).
- New module path `singlet-gpu/qc/` (NEW; first QC feature in singlet-gpu).
- Python wrapper: `python/singlet_gpu/qc/doublet.py` (~120 LOC) — adds `doublet_score(adata, *, n_synth_frac=0.25, k=20, seed=0, copy=False)`.
- R wrapper: `r/R/qc.R` (~100 LOC) — adds `doublet_score(sce, ...)`.
- Build flag: `FACTORNET_HAS_GPU=1`. cuRAND + cub.
- Dependencies: cycle 8 (kNN), cycle 4 (PCA — for the input embedding).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (Scrublet-style synthetic doublet scoring on GPU)` first comment.

## Risks

1. **N_synth choice** affects sensitivity. Default 0.25 × n_cells per Scrublet recommendation.
2. **Auto-threshold heuristic** is brittle. Expose threshold as a config field.
3. **Scrublet Python install** is heavy. Skip cleanly.
4. **Pre-computed PCA dependency**: doublet_score expects PCA already computed. Document.
