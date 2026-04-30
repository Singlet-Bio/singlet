---
feature: milo
roadmap_id: 47
module: include/singlet-gpu/abundance/milo.h + python/singlet_gpu/abundance/milo.py + r/R/abundance.R
status: design
tolerance: per-neighborhood log-fold-change Spearman ρ ≥ 0.95 vs Milo R; p-value rank Spearman ρ ≥ 0.90
target_perf: 100k cells × 100 donors × 50 kNN-neighborhoods per sample ≤30s on A100 (Milo R is ~20-40 min)
ooc_plan: per-neighborhood batched GLM; kNN reuses cycle 8 compute_exact
---

## Why this exists

Cycle 46 lookahead was weak — LIANA+ overlaps with CellChat, scVI fails the "no existing GPU" constraint. Picked orchestrator-judgment: **Milo** (Dann et al. Nature Biotechnology 2022) — a differential abundance (not expression) testing method that uses kNN neighborhoods to detect cell-state shifts between conditions. **Entirely R** (miloR package). 23rd "first GPU" candidate.

**Distinct from cycles 9, 15 (DE)**:
- Cycle 9/15 DE: test whether gene **expression** differs between conditions.
- Cycle 47 Milo: test whether neighborhood **abundance** (number of cells) differs between conditions.

Milo is the standard for detecting rare cell-state shifts where DE is underpowered. Critical for disease vs healthy comparison, treatment response, development milestones.

## Algorithm — Milo

```
Inputs:
  expression (n_cells × n_genes): log-normalized GEX
  donor_id (n_cells): donor label per cell (from donor_assignments.tsv)
  condition (n_donors): experimental condition per donor (binary or multi-level)
  (optional) design_matrix (n_donors × n_cov): covariates

Outputs:
  neighborhoods (n_neighborhoods × avg_size): kNN neighborhoods (cells each NH contains)
  nh_counts (n_neighborhoods × n_donors): cell count per donor in each neighborhood
  log_fc (n_neighborhoods): log-fold-change of abundance between conditions
  se (n_neighborhoods): standard error
  p_value (n_neighborhoods): Wald test p-value
  fdr (n_neighborhoods): graph-weighted FDR

Algorithm:
  1. kNN graph on PCA: k=30 neighbors per cell.
  2. Neighborhood sampling:
       a. Sample a subset of index cells (every N'th cell, stratified).
       b. Each index cell's kNN ball = one neighborhood.
  3. Per-neighborhood cell count per donor:
       nh_counts[nh, d] = |{c ∈ nh : donor[c] == d}|
  4. Quasi-likelihood negative binomial GLM (edgeR-like) per neighborhood:
       counts ~ NB(mean, dispersion)
       log(mean) = offset + X @ beta
     where offset = log(n_cells_per_donor) and X encodes condition.
  5. Wald test: z = beta_condition / se, p = 2 * Phi(-|z|).
  6. Graph-weighted FDR via kNN-graph adjacency BH (accounts for correlated neighborhoods).
```

## GPU implementation strategy

Native CUDA. Hot paths: kNN sampling + neighborhood-donor count (segmented histogram), batched NB GLM per neighborhood (reuses cycle 15 donor pseudobulk infrastructure).

### Kernels

1. **Neighborhood sampling** (`nh_sample_kernel`): stride sampling with Philox-seeded jitter.

2. **kNN neighborhood expansion**: for each index cell, gather its k neighbors from the cycle 8 kNN graph. Output sparse CSR `(n_neighborhoods × avg_k)`.

3. **Donor count per neighborhood** (`nh_donor_count_kernel`): for each (nh, cell) pair, atomicAdd into `nh_counts[nh, donor[cell]]`. 1 block per NH, 128 threads scanning cells.

4. **Batched NB GLM** (the hot kernel): 1 block per neighborhood, each block runs 20-iter IRLS on its `(n_donors × 2)` design matrix. fp32 hot loop, fp64 2×2 Hessian inverse (cycle 15 + 38 pattern).

5. **Graph-weighted FDR**: build sparse adjacency on the neighborhood graph (NH-NH edges if they share cells), apply spatially-correlated BH correction. Uses cuSPARSE SpMV.

## Numerical stability

- fp32 throughout; fp64 for 2×2 Hessian.
- IRLS with deviance convergence check.

## Memory layout

- Input: expression CSC, donor_id, condition from standard loaders.
- kNN graph: sparse CSR, 100k × 30 = 12 MB.
- Neighborhoods: `n_nh × avg_k × 4 bytes`. At 2000 × 30: 240 KB.
- `nh_counts`: `n_nh × n_donors × 4 bytes`. At 2000 × 100: 800 KB.
- Output: `(log_fc, se, p_value, fdr)`: 4 × 2000 × 4 bytes = 32 KB.
- Total: ~15 MB (tiny).

## Streams

One stream. All kernels lightweight.

## Out-of-core

Not needed at normal scale. Cell-batched kNN if n_cells > 500k.

## Determinism

Philox-seeded neighborhood sampling. IRLS deterministic given inputs. Bit-identical with fixed seed.

## Correctness test spec

Test: `tests/abundance_milo_correctness.cpp`.

Reference: Milo R via Rscript. Fallback: pure-R NB GLM per neighborhood (MASS::glm.nb) — always runs.

5 test cases:
1. **`Milo_TinySynthetic_VsR`**: 200 cells × 10 donors × 2 conditions with planted differential abundance. Spearman ρ ≥ 0.95 on log_fc vs R.
2. **`Milo_GSM_RealData`**: real scRNA sample with donor_assignments.tsv (skip if absent). Confirm neighborhoods formed, finite p-values.
3. **`Milo_IrlsConvergence`**: >95% of neighborhood GLMs converge within 20 IRLS iters.
4. **`Milo_PvalueCalibration_Uniform`**: under null (random condition labels), p-values uniform (KS ≤ 0.10).
5. **`Milo_Determinism_BitIdentical`**: bit-identical with fixed seed.

## Target performance

| Scale | Cells | Donors | NHs | Wall (target) | Milo R |
|---|---|---|---|---|---|
| tiny | 200 | 10 | 20 | <100ms | ~5s |
| 10k | 11,560 | 50 | 500 | <5s | ~5 min |
| 100k | ~120k | 100 | 2000 | <30s | ~30 min |

## Implementation notes

- Header path: `include/singlet-gpu/abundance/milo.h` (~800 LOC).
- New module path `singlet-gpu/abundance/` (NEW).
- Python wrapper: `python/singlet_gpu/abundance/milo.py` (~150 LOC).
- R wrapper: `r/R/abundance.R` (~120 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuSPARSE + cuBLAS + cub.
- Dependencies: cycle 1 (core), cycle 2 (loader), cycle 8 (kNN), cycle 15 (donor pseudobulk NB GLM — can reuse IRLS infrastructure).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU Milo-style kNN-neighborhood differential abundance)` first comment.

## Risks

1. **Graph-weighted FDR** requires NH-NH adjacency based on shared cells. Build via sparse boolean matrix multiply of `(nh × cells) @ (cells × nh)`. Might be expensive at 2000 × 100k.
2. **Milo R install** requires Bioconductor + miloR. MASS::glm.nb fallback always works.
3. **NH definition**: Milo uses index cells every k'th position; the sampling stride is a hyperparameter. Default matches miloR.
4. **Overdispersion estimation**: use edgeR's quasi-likelihood approximation. Document.
