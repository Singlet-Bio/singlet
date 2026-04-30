---
feature: scdrs
roadmap_id: 48
module: include/singlet-gpu/disease/scdrs.h + python/singlet_gpu/disease/scdrs.py + r/R/disease.R
status: design
tolerance: per-cell disease score Spearman ρ ≥ 0.95 vs scDRS Python on shared gene sets; p-value rank Spearman ρ ≥ 0.90
target_perf: 100k cells × 50 diseases × 1000 gene sets × 1000 control sets ≤30s on A100 (scDRS Python is ~5-10 min)
ooc_plan: cell-batched scoring; disease-set chunked
---

## Why this exists

Cycle 47 lookahead (LIANA+, MOFA+, InstaPrism, Baysor all had issues). **scDRS** (Zhang et al. Nature Genetics 2022) is the standard tool for per-cell disease relevance scoring from GWAS summary statistics — no existing GPU implementation. Acknowledged "algorithmically shallow" by the scout but the speedup story (10-15×) is honest and the method is widely used for disease-cell-type association. 23rd "first GPU" candidate (cycle 47 Milo is 23rd; scDRS becomes 24th — renumbering to keep order).

Biological use cases: map GWAS hits to cell types (schizophrenia → neurons? type 2 diabetes → beta cells?), inform clinical translation, link disease genetics to single-cell transcriptomics atlases.

## Algorithm — scDRS

```
Inputs:
  expression (n_cells × n_genes): log-normalized GEX
  disease_gene_sets: list of (disease_name, {gene: weight}) — MAGMA-derived from GWAS sumstats
  (optional) covariates (n_cells × n_cov): age, sex, PCs for score decorrelation

Outputs:
  raw_score (n_cells × n_diseases): weighted expression per cell per disease
  normalized_score (n_cells × n_diseases): z-score vs empirical null from control gene sets
  p_value (n_cells × n_diseases): Wilcoxon-like p-value vs null
  fdr (n_cells × n_diseases): BH-adjusted p-value

Algorithm:
  1. Gene expression Z-score normalization per cell: center + scale across genes.
  2. Raw score per (cell, disease): weighted sum of z-scores for genes in disease gene set.
  3. Control gene sets: sample 1000 matched "control" gene sets with similar mean expression + variance profile.
  4. Empirical null: compute raw scores for each control set → null distribution per cell.
  5. Normalized score: (raw_score - mean_null) / std_null.
  6. Monte Carlo p-value: rank of observed score in the null distribution.
  7. FDR correction.
```

## GPU implementation strategy

Native CUDA. The hot path is computing raw_score + 1000 control-set scores per cell, which is dominated by 1001 sparse weighted sums per cell. Mapped to batched SpMV.

### Kernels

1. **Gene z-score normalization** (`gene_zscore_kernel`): two-pass Welford across cells per gene (reuse cycle 3 HVG pattern).

2. **Control set matching** (one-time setup): sort genes by mean expression + variance, build matched bins, sample control sets with Philox-seeded RNG on host (small work, ~50k gene pool × 1000 sets).

3. **Batched score computation** (`batched_score_kernel`): cuSPARSE SpMM of
   `expression (n_genes × n_cells) @ weights_matrix (n_genes × (n_diseases + n_controls))`
   → `score_matrix (n_cells × (n_diseases + n_controls))`.
   Single SpMM handles both disease sets AND control sets together.

4. **Per-cell null distribution normalization** (`cell_norm_kernel`): for each cell, compute mean + std over the n_controls control scores; normalize each disease score.

5. **Monte Carlo p-value** (`mc_pvalue_kernel`): per (cell, disease), count fraction of controls with score >= observed. Output p-value.

6. **BH FDR correction** via `cub::DeviceSegmentedSort` + cummin (standard pattern).

## Numerical stability

- fp32 throughout. Z-scoring keeps values bounded.
- Two-pass Welford for mean/variance.

## Memory layout

- Input: expression CSC via cycle 0 loader.
- Weights matrix: `n_genes × (n_diseases + n_controls) × 4 bytes`. At 20k × 1050: 84 MB.
- Score matrix: `n_cells × 1050 × 4 bytes`. At 100k × 1050: 420 MB.
- Total: ~550 MB.

## Streams

One stream. SpMM dominates; no overlap needed.

## Out-of-core

Cell-batched at 50k per batch if memory pressure.

## Determinism

Philox-seeded control set sampling. Bit-identical with fixed seed.

## Correctness test spec

Test: `tests/disease_scdrs_correctness.cpp`.

Reference: scDRS Python via subprocess (`pip install scdrs`). Fallback: pure-Python weighted sum + control-set null matching (always runs).

5 test cases:
1. **`Scdrs_TinySynthetic_VsPython`**: 200 cells × 500 genes × 5 diseases synthetic with planted disease-associated cell clusters. Spearman ρ ≥ 0.95 on raw score.
2. **`Scdrs_GSM_RealData`**: real scRNA + a mini disease fixture (5 diseases × 20 genes each). Confirm finite results.
3. **`Scdrs_ControlSets_MatchedMeanVar`**: control sets have matched (mean, var) to disease sets within 5% relative error.
4. **`Scdrs_PvalueCalibration_Uniform`**: under null (random disease sets), p-values uniform (KS ≤ 0.10).
5. **`Scdrs_Determinism_BitIdentical`**: bit-identical with fixed seed.

## Target performance

| Scale | Cells | Diseases | Wall (target) | scDRS Python |
|---|---|---|---|---|
| tiny | 200 | 5 | <100ms | ~3s |
| 10k | 11,560 | 20 | <5s | ~2 min |
| 100k | ~120k | 50 | <30s | ~10 min |

## Implementation notes

- Header path: `include/singlet-gpu/disease/scdrs.h` (~600 LOC, small cycle).
- New module path `singlet-gpu/disease/` (NEW).
- Python wrapper: `python/singlet_gpu/disease/scdrs.py` (~120 LOC).
- R wrapper: `r/R/disease.R` (~100 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuSPARSE + cuBLAS + cuRAND + cub.
- Dependencies: cycle 1 (core), cycle 2 (loader), cycle 3 (hvg for z-score pattern).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU scDRS-style per-cell disease relevance scoring)` first comment.
- Ship a mini disease fixture `tests/refs/scdrs_diseases_mini.tsv` with 5 mock diseases × 20 genes each.

## Risks

1. **MAGMA gene set input format**: accept a pre-computed TSV per disease. Document format.
2. **Control set matching quality**: bin by (mean × var) in 20 bins, sample from same bin. Not as sophisticated as scDRS Python but close enough at target tolerance.
3. **scDRS Python install** via pip. Fall back to pure-Python weighted sum.
4. **Disease gene set overlap**: some genes appear in multiple disease sets. Handled naturally via SpMM with dense weight matrix.
