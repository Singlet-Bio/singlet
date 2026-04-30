---
feature: cospar
roadmap_id: 41
module: include/singlet-gpu/fate/cospar.h + python/singlet_gpu/fate/cospar.py + r/R/fate.R
status: design
tolerance: per-cell fate potency Spearman ρ ≥ 0.90 vs Cospar Python; clone-lineage-based transition map Frobenius norm ratio ≤ 1.05
target_perf: 100k cells × 1k lineage barcodes × 20 time points ≤2 min on A100 (Cospar Python is ~5-15 min)
ooc_plan: transition matrix block-sparse; coordinate descent chunked by row-block
---

## Why this exists

Cycle 40 lookahead: **Cospar** (Klein Lab, Nature Biotech 2022, active development through April 2025) is the leading cell fate transition mapping tool — integrates transcriptomic state with lineage barcodes to infer finite-time transition maps. **Entirely Python/numpy/scipy**. Core algorithm is iterative constrained matrix optimization over `(n_cells × n_cells)` transition maps, which naturally maps to GPU. 16th "first GPU" candidate.

Critical use cases: hematopoiesis lineage resolution, cancer drug resistance trajectory, developmental bifurcation driver discovery. Works on any scRNA dataset with time-point + lineage barcode annotations.

## Algorithm — Cospar

```
Inputs:
  expression (n_cells × n_genes): log-normalized GEX (from exon_counts.1pz)
  time_point (n_cells): time bin per cell
  lineage_clone_id (n_cells): shared clone label across time points (from barcodes)
  (optional) cell_type (n_cells): coarse annotation

Outputs:
  transition_map (n_cells_t0 × n_cells_t1): finite-time transition probability
  fate_bias (n_cells × n_fates): per-cell probability of reaching each terminal fate
  potency_score (n_cells): remaining lineage entropy (low = committed, high = stem-like)
  driver_genes (n_fates × n_genes): genes correlated with fate bias

Algorithm:
  1. Preprocess: log-normalize + HVG + joint PCA across time points.
  2. Build state similarity kernel K_state (n_cells × n_cells) from PCA kNN (Gaussian-weighted).
  3. Build lineage coupling L (n_cells_t0 × n_cells_t1) from shared clone IDs: L[i,j] = 1 if cell i at t0 and cell j at t1 share a clone, else 0.
  4. Optimize transition map T ∈ R^(n_t0 × n_t1) via coordinate descent:
       min ||T - L||² + λ1 ||T||_1 + λ2 tr(T^T K_state T)
       subject to T >= 0, row-sums <= 1
     - 50 outer iterations, each iter: gradient step + prox (soft-threshold + sum-to-1 projection).
  5. Compute fate_bias via forward iteration: bias[c] = T^k @ terminal_indicator for k time steps.
  6. Potency score: Shannon entropy of fate_bias per cell.
  7. Driver genes: Pearson correlation of fate_bias vs gene expression.
```

## GPU implementation strategy

Native CUDA. Core operations are sparse SpMM + dense projection + iterative optimization.

### Kernels

1. **kNN state similarity** (reuse cycle 8 `compute_exact`): 50-NN on joint PCA, Gaussian kernel, symmetric k-graph.

2. **Lineage coupling** (`lineage_coupling_kernel`): for each pair (i, j) where `time[i] < time[j]`, check `clone_id[i] == clone_id[j]`. Output sparse COO. `cub::DeviceSelect::Flagged`.

3. **Transition optimization** (the hot loop, 50 outer iters):
   - **Gradient kernel**: `grad = 2*(T - L) + 2*lambda2 * (K_state @ T)`. cuSPARSE SpMM for `K_state @ T`.
   - **Prox step**: elementwise `T = max(0, T - step*grad - lambda1*step)` (soft-threshold + non-neg).
   - **Sum-to-1 projection**: per-row simplex projection (sort-based, cycle 33 FlashDeconv pattern).
   - **Convergence check**: Frobenius residual scalar via `cub::DeviceReduce::Sum`, read once per outer iter (≤4 bytes H2D at outer-loop boundary — valid per absolute rule §⛔9 exceptions).

4. **Fate bias forward iteration**: k=5 SpMV `bias = T @ indicator`.

5. **Potency score**: elementwise Shannon entropy kernel on `bias`.

6. **Driver genes**: Pearson correlation via cycle 36 GRaNIE pattern (1 block per (fate, gene) pair).

## Numerical stability

- fp32 throughout. Soft-threshold + prox is stable.
- Convergence tolerance: relative Frobenius residual < 1e-4.

## Memory layout

- Input: expression CSC via cycle 0 loader.
- Transition map T: dense `(n_t0 × n_t1) × 4 bytes`. For 10k × 10k: 400 MB. Sparse rep if >20k cells per time point.
- Lineage coupling L: sparse COO, few nnz.
- K_state: sparse kNN graph (n_cells × n_cells, ~50 nnz/row).
- cub temps: ~20 MB.
- Total: ~500-800 MB at 100k cells split across time points.

## Streams

One stream. Optimization loop is sequential.

## Out-of-core

Transition map block-sparse at very large scale: 100k × 100k = 40 GB dense — too big. At `n_cells > 50k`, switch to **block-sparse** T with only lineage-coupled blocks stored (falls back to density when lineage coverage is high).

## Determinism

No stochasticity. Bit-identical. `cfg.deterministic=true` is a no-op.

## Correctness test spec

Test: `tests/fate_cospar_correctness.cpp`.

Reference: Cospar Python via subprocess. Fallback: pure-Python implementation of the core iterative optimizer from the Cospar paper equations.

5 test cases:
1. **`Cospar_TinySynthetic_VsPython`**: 200 cells × 2 time points × 50 genes × 3 clones with planted bifurcation. Spearman ρ ≥ 0.90 on fate bias vs Python.
2. **`Cospar_GSM_RealLineage`**: load a real sample with lineage barcodes (skip if unavailable — most current data doesn't have them; fall back to state-only mode).
3. **`Cospar_TransitionMap_RowSumsValid`**: all row sums of T are in [0, 1] within tolerance.
4. **`Cospar_PotencyMonotone`**: on a hierarchical synthetic tree, potency decreases monotonically along committed paths.
5. **`Cospar_Determinism_BitIdentical`**: bit-identical across two runs.

## Target performance

| Scale | Cells | Lineages | Time pts | Wall (target) | Cospar Python |
|---|---|---|---|---|---|
| tiny | 200 | 3 | 2 | <100ms | ~5s |
| 10k | 11,560 | 100 | 5 | <30s | ~3 min |
| 100k | ~120k | 1000 | 20 | <2 min | ~15 min |

## Implementation notes

- Header path: `include/singlet-gpu/fate/cospar.h` (~900 LOC, under 1500 cap).
- New module path `singlet-gpu/fate/` (NEW).
- Python wrapper: `python/singlet_gpu/fate/cospar.py` (~150 LOC).
- R wrapper: `r/R/fate.R` (~120 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuSPARSE + cuBLAS + cub.
- Dependencies: cycle 1 (core), cycle 2 (loader), cycle 3 (lognorm), cycle 4 (PCA), cycle 8 (kNN — call `compute_exact` directly per CYCLE-35-FOLLOWUP).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU Cospar-style cell fate transition mapping)` first comment.
- Use factornet field-access style.

## Risks

1. **Lineage barcode data scarcity**: most scRNA samples lack lineage barcodes. Provide a state-only fallback (no transition optimization, just potency from kNN entropy).
2. **Transition map density**: at 100k cells, dense T is 40 GB. Use block-sparse decomposition by clone + time point; document complexity.
3. **Cospar Python install** requires the Klein Lab package (cospar 0.3.4+). Pure-Python fallback implements only the core optimizer.
4. **Simplex projection**: reuse cycle 33 FlashDeconv's sort-based projection.
