---
feature: flash_deconv
roadmap_id: 33
module: include/singlet-gpu/spatial/flash_deconv.h + python/singlet_gpu/spatial/deconv.py + r/R/spatial_deconv.R
status: design
tolerance: per-spot cell-type abundance Spearman ρ ≥ 0.90 vs cell2location on the same input; uncertainty bounds within 20% of cell2location
target_perf: 1.6M Visium HD bins × 30 cell types × 5k HVG ≤2.5 min on A100 (cell2location is hours)
ooc_plan: spot-batched ADMM iterations; sketching reduces effective problem size dramatically
---

## Why this exists

Cycle 32 lit-scout's top recommendation: **FlashDeconv** (bioRxiv 2026) — structure-preserving sketching + ADMM for atlas-scale spatial deconvolution. Massive gap between RCTD/cell2location (hours) and FlashDeconv (~2 min). 8th "first GPU implementation" candidate.

Spatial deconvolution = inferring cell-type abundance per Visium / Xenium spot from a reference single-cell atlas. Critical for spatial transcriptomics analysis. singlet-gpu has the spatial coords + count matrices via singlify; this closes the spatial workflow loop.

## Algorithm — FlashDeconv

```
Inputs:
  Y (n_spots × n_genes): spatial expression
  X_ref (n_cells_ref × n_genes): single-cell reference atlas
  cell_types_ref (n_cells_ref): cell type labels for the reference
Outputs:
  W (n_spots × n_types): per-spot cell-type abundance (proportions)
  uncertainty (n_spots × n_types): credible interval width

Algorithm:
  1. Aggregate the reference atlas: B (n_types × n_genes) = mean(X_ref by cell_type)
     This gives a per-cell-type expression profile.
  2. Build the deconvolution problem: Y ≈ W @ B, where W is non-negative and rows sum to 1.
  3. Leverage-score sketching (FlashDeconv's key trick):
       - Compute leverage scores from B's SVD.
       - Subsample genes weighted by leverage scores → keep top-K (K << n_genes).
       - Solve the smaller problem: Y_sub ≈ W @ B_sub.
  4. ADMM solver:
       a. W_t = argmin ||Y_sub - W B_sub||² + λ ||W||_1 (sparsity)
       b. Soft-thresholding for non-negativity
       c. Sum-to-one constraint via projection
       d. Iterate ~50 times until convergence.
  5. Spatial regularization (optional): Laplacian penalty using the spatial kNN graph
     (cycle 8) to encourage spatially coherent cell-type patterns.
  6. Bayesian uncertainty: bootstrap the sketching step → posterior over W.
```

## GPU implementation strategy

Native CUDA. Reuses cycle 8 kNN for the spatial graph + cycle 5 NMF NNLS solver as inner step.

### Kernels

- **Reference aggregation** (one block per type, segmented sum across cells): `cub::DeviceSegmentedReduce::Sum`.
- **Leverage score computation**: SVD on `B^T B` via cuSOLVER (small: n_types × n_types). Leverage = ||B U||² per column.
- **Sketch sampling**: weighted subsample via cuRAND Philox.
- **ADMM iteration**:
  - Primal update: NNLS via cycle 5 NMF adapter pattern (or direct cuBLAS solve for the small problem).
  - Dual update: simple elementwise.
  - Constraint projection: sum-to-1 simplex projection (sort-based, cycle 11 pattern).
- **Spatial regularization**: SpMM with the cycle 8 kNN graph as Laplacian.
- **Bootstrap uncertainty**: re-sample the sketch with different seeds, run K=20 iterations, compute per-element variance.

## Numerical stability

- fp32 hot path; fp64 accumulator for the loss.
- ADMM convergence: relative change in primal residual < 1e-4.

## Memory layout

- Y: dense (n_spots × n_genes) × 4 bytes. For 1.6M spots × 30k genes: 200 GB. **Way too big**.
- **Critical**: use sparse Y from `.1pz`. Most spots are sparse.
- B: dense (n_types × n_genes). For 30 × 30k: 4 MB. Tiny.
- W: dense (n_spots × n_types). For 1.6M × 30: 192 MB.
- Total: ~250 MB workspace beyond input.

## Streams

One stream, caller-provided.

## Out-of-core

Spot-batched ADMM: process N=100k spots at a time, update global W after each batch.

## Determinism

cuRAND Philox seeded for sketching + bootstrap.

## Correctness test spec

Test: `tests/spatial_flash_deconv_correctness.cpp`.

Reference: cell2location Python via subprocess (heavy install).

Test cases:
1. **`FlashDeconv_TinySynthetic_VsCell2Location`**: 200 spots × 100 genes × 5 cell types. Spearman ρ ≥ 0.90 on per-spot abundance.
2. **`FlashDeconv_GSM4037629_RealData`**: real Visium spatial_coords + counts. Confirm finite results, sum-to-1 constraint per spot.
3. **`FlashDeconv_LeverageScoreSketching_RecallTopGenes`**: confirm sketching keeps the most informative genes.
4. **`FlashDeconv_BootstrapUncertainty_NonZero`**: confirm uncertainty bounds are non-zero and reasonable.
5. **`FlashDeconv_Determinism_BitIdentical`**: bit-identical with fixed seed.
6. **`FlashDeconv_SpatialRegularization_SmoothsAbundance`**: with `spatial_lambda=1.0`, confirm neighboring spots have correlated abundances.

## Target performance

| Scale | Spots | Cell types | Wall (target) |
|---|---|---|---|
| tiny | 200 | 5 | <100ms |
| 100k Visium | 100k | 30 | <30s |
| 1.6M Visium HD | 1.6M | 30 | <2.5 min |

## Implementation notes

- Header path: `include/singlet-gpu/spatial/flash_deconv.h` (~800 LOC).
- Python wrapper: `python/singlet_gpu/spatial/deconv.py` (~150 LOC).
- R wrapper: `r/R/spatial_deconv.R` (~120 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuBLAS + cuSOLVER + cuRAND + cub + cuSPARSE.
- Dependencies: cycle 8 (kNN for spatial Laplacian), cycle 5 (NMF NNLS solver inner step).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU FlashDeconv with leverage-score sketching + ADMM)` first comment.

## Risks

1. **Sum-to-1 simplex projection** is non-trivial. Use the well-known sort-based algorithm.
2. **Leverage-score sketching** quality depends on K (sketch size). Default K=500 genes.
3. **cell2location reference install** is very heavy (PyMC + scvi-tools). Skip cleanly.
4. **Reference atlas format**: the singlify output is per-sample; the user must concat multiple samples + assign cell types externally before passing the reference. Document.
