---
feature: palantir
roadmap_id: 45
module: include/singlet-gpu/fate/palantir.h + python/singlet_gpu/fate/palantir.py + r/R/fate.R
status: design
tolerance: per-cell pseudotime Spearman ρ ≥ 0.95 vs Palantir Python; terminal branch probability Spearman ρ ≥ 0.90
target_perf: 100k cells × 10 terminal states × 50-NN graph ≤1 min on A100 (Palantir Python is ~15-30 min)
ooc_plan: diffusion operator cached; random-walk chunked by start-cell block
---

## Why this exists

Cycle 44 lookahead runner-up 1 (chosen because top rec overlapped with cycle 28 CSI-GEP): **Palantir** (Setty et al. Nature Biotech 2019) is the leading diffusion-pseudotime + cell fate inference tool for scRNA lineage mapping. **Entirely Python/scipy**. 21st "first GPU" candidate.

**Complementary to cycles 41, 43** — completes the fate-inference triad:
- **Cycle 41 Cospar**: infers transition map T from **lineage barcodes** via constrained optimization.
- **Cycle 43 CellRank 2**: computes absorption probabilities from a **given** transition matrix via batched GMRES.
- **Cycle 45 Palantir**: builds its own **diffusion operator from scratch** (from kNN graph + Markov chain), then computes pseudotime AND fate probabilities via random walks — no lineage barcodes needed, no external T.

Palantir works on any scRNA dataset with a specified starting cell. Cospar and CellRank 2 have stricter data requirements.

## Algorithm — Palantir

```
Inputs:
  expression (n_cells × n_genes): log-normalized GEX
  start_cell: index of the starting cell (most stem-like)
  (optional) terminal_cells: indices of terminal states (auto-detected if not provided)

Outputs:
  pseudotime (n_cells): scalar [0, 1], distance-from-start in diffusion time
  branch_probs (n_cells × n_terminals): probability of reaching each terminal
  diff_operator (n_cells × n_cells): transition matrix built from diffusion distances

Algorithm:
  1. kNN graph on PCA embedding (reuse cycle 8 compute_exact, k=30).
  2. Diffusion kernel: for each edge (i, j), w[i, j] = exp(-dist² / 2σᵢ²), σᵢ = distance to k'th neighbor.
  3. Row-normalize → stochastic matrix P.
  4. Anisotropic correction: P' = D⁻¹/² P D⁻¹/² where D is the degree diagonal.
  5. Spectral decomposition: top-k eigenvalues + eigenvectors of P' via Lanczos.
  6. Diffusion time: pseudotime[cell] = sqrt( sum over k of λ_k^2 * (ψ_k[cell] - ψ_k[start])² )
     where ψ_k is the k'th eigenvector.
  7. Terminal state detection: cells with maximum pseudotime in each cluster (auto-detect via kNN on pseudotime + eigenvectors).
  8. Branch probabilities: absorbing random walk. For each cell, probability of being absorbed
     into each terminal state, computed via matrix power iteration or direct Markov absorption:
       branch_prob = (I - Q)^(-1) * R where Q = transient-transient, R = transient-absorbing.
     (Similar to cycle 43 CellRank 2 but with Palantir's specific operator.)
```

## GPU implementation strategy

Native CUDA. Reuses cycle 8 kNN, factornet Lanczos SVD for spectral step, and cycle 43 CellRank 2 GMRES for the absorbing random walk.

### Kernels

1. **kNN + diffusion kernel** (`diffusion_op_kernel`):
   - Cycle 8 `compute_exact` for kNN.
   - Fused kernel: edge weight = `expf(-d² / (2*σ²))` per edge.
   - Row-sum + normalize → sparse P in CSR.

2. **Anisotropic correction** (`anisotropic_kernel`):
   - Diagonal D via row-sum of P.
   - Symmetric normalize: `P' = D⁻¹/² @ P @ D⁻¹/²` in-place on the CSR values.

3. **Spectral decomposition** (reuse factornet `lanczos_gpu`): top-k eigenvectors of the symmetric P'. k ≈ 20.

4. **Pseudotime kernel** (`pseudotime_kernel`): per-cell computation
   `pt[c] = sqrt( sum_k (lambda[k]^2 * (psi[k][c] - psi[k][start])^2) )`
   fully parallel.

5. **Terminal state detection** (optional):
   - k-means or Leiden cluster on eigenvectors (reuse cycle 7 leiden.h).
   - Max pseudotime per cluster → terminal candidate set.

6. **Branch probabilities via batched GMRES**: **reuse cycle 43 CellRank 2 kernel directly**. Pass the Palantir diffusion operator + terminal indices; get absorption probabilities.

## Numerical stability

- fp32 throughout.
- Eigendecomposition via factornet Lanczos (already numerically hardened).
- Branch probabilities via cycle 43 GMRES (already fp64 Krylov accumulator).

## Memory layout

- Input: expression CSC via cycle 0 loader.
- Diffusion operator: sparse CSR `n_cells × n_cells`, ~30 nnz/row. At 100k × 30: 24 MB.
- Eigenvectors: `n_cells × k × 4 bytes`. At 100k × 20: 8 MB.
- Pseudotime: `n_cells × 4 bytes`. 400 KB.
- Branch probs: `n_cells × n_terminals × 4 bytes`. At 100k × 10: 4 MB.
- Total: ~50 MB + kNN workspace.

## Streams

One stream. kNN → diffusion → eigendecomp → branch probs is sequential.

## Out-of-core

Diffusion operator cached; branch prob computation reuses cycle 43 GMRES chunking.

## Determinism

cycle 8 `compute_exact` is deterministic; factornet Lanczos is deterministic given the same start vector; cycle 43 GMRES is deterministic. Palantir is bit-identical with fixed seed.

## Correctness test spec

Test: `tests/fate_palantir_correctness.cpp`.

Reference: Palantir Python via subprocess. Fallback: pure-Python scipy-based diffusion map + eigendecomposition + Markov absorption.

5 test cases:
1. **`Palantir_TinySynthetic_VsPython`**: 200 cells × 50 genes synthetic hierarchical tree (root → 3 branches). Spearman ρ ≥ 0.95 on pseudotime vs Python.
2. **`Palantir_GSM_RealData`**: real scRNA sample. Confirm pseudotime finite, monotone increase from provided start cell on a smooth trajectory.
3. **`Palantir_BranchProbs_SumToOne`**: per-cell branch probability row sums = 1 ± 1e-5.
4. **`Palantir_TerminalCells_Self1`**: terminal cells have branch_prob = 1 on their own terminal.
5. **`Palantir_Determinism_BitIdentical`**: bit-identical across two runs.

## Target performance

| Scale | Cells | Terminals | Wall (target) | Palantir Python |
|---|---|---|---|---|
| tiny | 200 | 3 | <100ms | ~5s |
| 10k | 11,560 | 5 | <10s | ~5 min |
| 100k | ~120k | 10 | <1 min | ~20 min |

## Implementation notes

- Header path: `include/singlet-gpu/fate/palantir.h` (~700 LOC, small cycle).
- Python wrapper: `python/singlet_gpu/fate/palantir.py` (~150 LOC).
- R wrapper: `r/R/fate.R` add function (~50 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuSPARSE + cuBLAS + cuSOLVER + cub.
- Dependencies: cycle 1 (core), cycle 2 (loader), cycle 4 (factornet Lanczos SVD), cycle 8 (kNN), cycle 43 (CellRank 2 GMRES).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU Palantir-style diffusion pseudotime + fate inference)` first comment.

## Risks

1. **Symmetric eigendecomposition of sparse P'**: factornet Lanczos handles this. Validate via eigenvalue test.
2. **Palantir Python install** is heavy (scanpy + networkx + palantir). Scipy fallback always runs.
3. **Start cell selection**: for the real-data test, pick the cell with the highest expression of a known stem marker (or just the first cell).
4. **Dependency on cycle 43**: this cycle cannot land until cycle 43 CellRank 2 kernel is source-complete. OK since cycle 43 is already dispatched.
