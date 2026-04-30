---
feature: omnidoublet
roadmap_id: 39
module: include/singlet-gpu/qc/omnidoublet.h + python/singlet_gpu/qc/omnidoublet.py + r/R/qc.R
status: design
tolerance: per-cell doublet probability Spearman ρ ≥ 0.90 vs OmniDoublet Python; ROC AUC ≥ 0.90 on the 10x Genomics doublet benchmark
target_perf: 100k cells × 20k genes × 500 ADT tags ≤3 min on A100 (OmniDoublet Python is ~30-60 min)
ooc_plan: artificial-doublet simulation chunked at N=5000 pairs; kNN reuses cycle 8 loader
---

## Why this exists

Cycle 38 lookahead: **OmniDoublet** (Briefings in Bioinformatics 2024) is a multimodal extension of Scrublet/DoubletFinder that integrates RNA + ADT signals for CITE-seq doublet detection. Cycle 31's `qc/doublet_score.h` handles RNA-only doublet scoring; OmniDoublet is a distinct feature because:
1. It consumes `adt.1pz` (the CITE-seq antibody matrix), which no prior cycle uses.
2. It uses joint-embedding kNN rather than RNA-only kNN.
3. It fits a logistic classifier on multimodal doublet features instead of a simple kNN density score.

14th "first GPU" candidate. Pairs naturally with cycles 19-23 CITE-seq support and unblocks high-quality doublet filtering for multimodal datasets.

## Algorithm — OmniDoublet

```
Inputs:
  rna_counts (n_cells × n_genes): spliced UMI counts (from exon_counts.1pz)
  adt_counts (n_cells × n_tags): antibody counts (from adt.1pz)
  (optional) cell_type (n_cells): prior annotation for stratified doublet simulation

Outputs:
  doublet_score (n_cells): probability [0, 1]
  doublet_call (n_cells): binary 0/1 at a target FDR
  simulated_doublet_scores (n_sim): doublet scores on the simulated set (for calibration)

Algorithm:
  1. Artificial doublet simulation:
       a. Sample n_sim = 2 * n_cells pairs (i, j) uniformly from real cells.
       b. artificial_rna[k] = rna_counts[i] + rna_counts[j]  (sum, not average)
       c. artificial_adt[k] = adt_counts[i] + adt_counts[j]
     Result: n_sim synthetic "doublet" cells.
  2. Joint embedding:
       a. Concatenate real + simulated (n_cells + n_sim rows).
       b. Log-normalize each modality independently.
       c. HVG selection on the RNA side (top 2000).
       d. PCA on RNA+ADT joint (stack after PCA on each modality, or CCA).
       e. Result: joint_pca (n_cells + n_sim × n_pcs).
  3. Multimodal kNN: for each real cell, find its k=50 nearest neighbors in joint_pca space.
  4. Doublet features per real cell:
       a. doublet_fraction = fraction of k-NN that are simulated doublets.
       b. mean_distance_to_doublets = mean distance to simulated neighbors.
       c. rna_umi_zscore = z-score of total RNA UMIs.
       d. adt_umi_zscore = z-score of total ADT counts.
  5. Logistic classifier: fit on (simulated=1, real=0) using features from step 4; score each real cell.
  6. FDR-controlled doublet calling at target rate (e.g., 10%).
```

## GPU implementation strategy

Native CUDA. The hot paths are artificial doublet simulation (sparse CSC gather + sum), joint PCA (reuse cycle 4 Randomized SVD), and multimodal kNN (reuse cycle 8 `compute_exact` + the cycle-35 lesson of calling `compute_exact` directly).

### Kernels

1. **Sparse doublet simulation** (`sim_doublets_kernel`): for each (i, j) pair, gather CSC row values from rna_counts[i] and rna_counts[j], sum into a dense row. Output: dense `(n_sim × n_genes)` + dense `(n_sim × n_tags)`. Memory heavy — **tile by n_sim_batch=5000 rows**. Philox4x32_10 for pair sampling.

2. **Log-normalize** (reuse cycle 2 `lognorm`): applied to concatenated real + simulated rows.

3. **HVG** (reuse cycle 3 `hvg`): top 2000 on RNA.

4. **Joint PCA**: factornet `randomized_gpu` on the concatenated (RNA HVG + ADT) matrix. Reuse cycle 4 SVD adapter.

5. **Multimodal kNN** (reuse cycle 8 `compute_exact` directly): 50-NN in joint PCA space.

6. **Doublet features**: per-cell kernel computing fraction of k-NN that are simulated (tagged via the `is_sim[n_cells + n_sim]` flag array), mean distance, UMI z-scores via two-pass Welford.

7. **Logistic classifier**: batched IRLS on (n_cells + n_sim) samples × 4 features. 20 iterations max, fp32 + fp64 Hessian inverse (2×2 to 5×5 analytic). Reuse the cycle 38 Fisher-scoring pattern.

8. **FDR doublet calling**: sort doublet scores, pick threshold that gives target simulated-doublet capture.

## Numerical stability

- fp32 throughout. IRLS uses fp64 for the small-dimensional Hessian inverse.
- Joint PCA via Randomized SVD (fp32 operator, q=3 power iters).

## Memory layout

- Input: `rna_counts` CSC + `adt_counts` CSC via cycle 0 loader.
- Artificial doublet tile: `n_sim_batch × n_genes × 4 bytes`. At 5000 × 20k: 400 MB. **Tile loop** keeps this bounded.
- Joint embedding: `(n_cells + n_sim) × n_pcs × 4 bytes`. At 300k × 50: 60 MB.
- kNN output: `n_cells × k × 8 bytes`. At 100k × 50: 40 MB.
- Total workspace: ~600 MB.

## Streams

One stream. Artificial doublet simulation tiles can overlap with the next tile's compute via a second stream if wall becomes critical.

## Out-of-core

n_sim chunking naturally tiles. n_cells is the outer loop; each tile processes a fixed n_sim_batch.

## Determinism

Philox4x32_10 seeded for doublet pair sampling. Deterministic mode uses segmented reduction for the feature aggregation instead of atomicAdd.

## Correctness test spec

Test: `tests/qc_omnidoublet_correctness.cpp`.

Reference: OmniDoublet Python via subprocess (when available). Fallback: Scrublet Python reference (RNA-only) for the unimodal regression test.

5 test cases:
1. **`Omnidoublet_TinySynthetic_VsPython`**: 500 cells × 200 genes × 20 ADT tags synthetic with 10% planted doublets. ROC AUC ≥ 0.85 vs Python reference.
2. **`Omnidoublet_GSM_CITEseq_RealData`**: load a real CITE-seq sample with adt.1pz. Confirm finite scores, 5-15% call rate at default FDR.
3. **`Omnidoublet_MultimodalFeatures_NonZero`**: confirm the ADT-dependent features contribute (zeroing ADT changes scores by ≥5% on ≥80% of cells).
4. **`Omnidoublet_FdrControl`**: simulated doublets captured at the requested target rate ±5%.
5. **`Omnidoublet_Determinism_BitIdentical`**: bit-identical with fixed seed.

## Target performance

| Scale | Cells | Genes | ADT | Wall (target) | OmniDoublet Py |
|---|---|---|---|---|---|
| tiny | 500 | 200 | 20 | <100ms | ~10s |
| 10k | 11,560 | 20k | 100 | <30s | ~5 min |
| 100k | ~120k | 20k | 500 | <3 min | ~30-60 min |

## Implementation notes

- Header path: `include/singlet-gpu/qc/omnidoublet.h` (~1100 LOC, under 1500 cap).
- Python wrapper: `python/singlet_gpu/qc/omnidoublet.py` (~150 LOC).
- R wrapper: `r/R/qc.R` add function (~50 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuBLAS + cuRAND + cub + cuSPARSE.
- Dependencies: cycle 1 (core), cycle 2 (loader), cycle 3 (lognorm), cycle 4 (Randomized SVD), cycle 8 (kNN), cycle 31 (doublet_score for unimodal comparison in tests).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU OmniDoublet-style multimodal doublet detection)` first comment.
- Use factornet field-access style.
- **Must call `compute_exact` directly**, not the `compute_knn` wrapper, until CYCLE-35-FOLLOWUP-KNN-WRAPPER-FIELD-STYLE is resolved.

## Risks

1. **Cycle 31 overlap**: `qc/doublet_score.h` exists but is RNA-only. Distinct API: `omnidoublet` takes both rna_counts and adt_counts and cannot be called without ADT. Document clearly which to use when.
2. **OmniDoublet Python install** may be heavy. Fall back to Scrublet Python for the RNA regression test; skip the multimodal comparison.
3. **Joint PCA** requires concatenating RNA HVG + ADT. Simple concatenation may be dominated by RNA; OmniDoublet uses CCA. For MVP use concatenation + separate PCA per modality + stack; document the simplification.
4. **n_sim memory**: 2×n_cells simulated cells + n_genes dense tile is ~800 MB at 100k×20k. Tile loop is mandatory, not optional.
