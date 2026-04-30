---
feature: cellrank2
roadmap_id: 43
module: include/singlet-gpu/fate/cellrank2.h + python/singlet_gpu/fate/cellrank2.py + r/R/cellrank.R
status: design
tolerance: per-cell absorption probability Spearman ρ ≥ 0.95 vs CellRank 2 Python; terminal-state agreement ≥ 0.90
target_perf: 100k cells × 10 terminal states × 50-NN graph ≤1 min on A100 (CellRank 2 Python is ~10-15 min)
ooc_plan: transition matrix is sparse (50-NN); GMRES chunked by right-hand side block
---

## Why this exists

Cycle 42 lookahead runner-up, chosen over the weaker CytoTRACE2 top recommendation: **CellRank 2** (Weiler et al. Nature Methods 2024) is the leading tool for computing cell-fate absorption probabilities from Markov transition matrices on scRNA kNN graphs. **Entirely Python** (scipy sparse linear algebra). 18th "first GPU" candidate.

**Pairs with cycle 41 Cospar**: Cospar *infers* a transition map T from lineage barcodes; CellRank 2 *consumes* a transition matrix T (from velocity, from a kNN graph Laplacian, or from Cospar) and computes per-cell absorption probabilities for terminal states. Together they form a complete fate-inference pipeline.

Biological use cases: hematopoiesis branching, tumor clonal fate trajectories, developmental bifurcation driver discovery, drug perturbation fate shift.

## Algorithm — CellRank 2 absorption probabilities

```
Inputs:
  T (n_cells × n_cells): row-stochastic transition matrix (sparse, ~50 nnz/row)
                         Either from cycle 41 Cospar or from velocity moments (cycle 13).
  terminal_states (n_terminals): indices of absorbing states (or soft memberships)
  (optional) weight_connectivities: mix velocity + connectivity for robust T

Outputs:
  absorption_prob (n_cells × n_terminals): P(cell ends up in each terminal)
  driver_genes (n_terminals × n_genes): genes correlated with fate bias
  lineage_tree: hierarchical lineage tree from absorption profile clustering

Algorithm:
  1. Terminal state identification:
       a. Schur decomposition of T → leading complex eigenvalues (λ ≈ 1).
       b. Stationary distribution clustering → terminal state candidates.
       (OR: user provides terminal states directly.)
  2. Fundamental matrix solve:
       Let Q = T restricted to transient states, R = T from transient to absorbing.
       Solve (I - Q) @ B = R for B ∈ R^(n_transient × n_absorbing).
       This is a sparse linear system with (I - Q) as coefficient matrix.
  3. Iterative solver: GMRES or BiCGStab on the sparse matrix (I - Q).
     - CellRank 2's key engineering: iterative matrix-free solve avoids building (I - Q)⁻¹.
     - Preconditioning: ILU(0) or diagonal.
     - Convergence tolerance: relative residual < 1e-6.
  4. Per-cell absorption probability = row of B for transient cells; identity for absorbing cells.
  5. Driver genes: Pearson correlation of absorption probability vs gene expression.
```

## GPU implementation strategy

Native CUDA. Leverages cuSPARSE sparse matrix-vector products + cuSOLVER's iterative solvers for the GMRES step. The per-terminal-state linear solve is embarrassingly parallel across right-hand sides (batched GMRES).

### Kernels

1. **kNN graph → transition matrix** (`knn_to_transition_kernel`): row-normalize the kNN graph (reuse cycle 8 `compute_exact` for the kNN). Output sparse CSR T.

2. **Terminal state identification via Schur** (cuSOLVER): optional path when user doesn't provide terminals. Reuses cycle 4 factornet SVD infrastructure for the sparse top-k eigenvalue problem via Lanczos on `T^T T`.

3. **Batched GMRES solve** (the hot kernel): solve `(I - Q) @ B = R` for `n_terminals` right-hand sides in parallel.
   - cuSPARSE SpMV inside each GMRES iteration.
   - Arnoldi orthogonalization via cuBLAS dgemv + cublasDgemm.
   - Givens rotations for the Hessenberg QR on device (small, fp64).
   - Convergence check: residual norm via `cub::DeviceReduce::Sum`, one scalar D2H per outer iter (approved §⛔9 exception).
   - Krylov subspace dimension m=30, restart every m iters, max 10 restarts.

4. **Driver gene correlation** (reuse cycle 36 GRaNIE pattern): 1 block per (terminal, gene) pair, fused Pearson.

5. **Lineage tree clustering** (reuse cycle 7 `leiden.h`): optional, on the `(n_cells × n_terminals)` absorption profile.

## Numerical stability

- fp32 SpMV + fp64 Krylov basis accumulator (small, k=30 vectors).
- Relative residual tolerance 1e-6.
- Restart GMRES handles loss of orthogonality in long Krylov runs.

## Memory layout

- Input: sparse T (CSR), n_cells × n_cells, ~50 nnz/row. At 100k × 50: 40 MB.
- Krylov basis: `m × n_cells × 4 bytes`. At 30 × 100k: 12 MB per right-hand side. × n_terminals=10: 120 MB.
- Right-hand side R: `n_cells × n_terminals × 4 bytes`. At 100k × 10: 4 MB.
- Solution B: same shape, 4 MB.
- Total: ~200 MB.

## Streams

One stream. GMRES is sequential; batching across right-hand sides happens inside each kernel.

## Out-of-core

GMRES batched by right-hand side: process 4 terminals at a time if memory tight. Not expected at normal scale.

## Determinism

No stochasticity. Bit-identical across runs.

## Correctness test spec

Test: `tests/fate_cellrank2_correctness.cpp`.

Reference: CellRank 2 Python via subprocess. Fallback: scipy sparse linalg gmres call.

5 test cases:
1. **`Cellrank2_TinySynthetic_VsPython`**: 200 cells × 3 terminal states with planted transition structure. Spearman ρ ≥ 0.95 on absorption prob vs Python.
2. **`Cellrank2_GSM_RealData`**: load real sample, build kNN from PCA, run absorption. Confirm row sums = 1 per cell.
3. **`Cellrank2_GmresConvergence`**: residual norm drops below 1e-6 within 300 SpMV applications on the tiny synthetic.
4. **`Cellrank2_AbsorbingStates_Identity`**: for cells that ARE absorbing states, absorption prob = 1 for own terminal, 0 elsewhere.
5. **`Cellrank2_Determinism_BitIdentical`**: bit-identical across two runs.

## Target performance

| Scale | Cells | Terminals | Wall (target) | CellRank 2 Python |
|---|---|---|---|---|
| tiny | 200 | 3 | <100ms | ~2s |
| 10k | 11,560 | 5 | <5s | ~2 min |
| 100k | ~120k | 10 | <1 min | ~15 min |

## Implementation notes

- Header path: `include/singlet-gpu/fate/cellrank2.h` (~1000 LOC).
- Python wrapper: `python/singlet_gpu/fate/cellrank2.py` (~150 LOC).
- R wrapper: `r/R/cellrank.R` (~120 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuSPARSE + cuBLAS + cuSOLVER + cub.
- Dependencies: cycle 1 (core), cycle 4 (factornet SVD for terminal state ID), cycle 7 (leiden for lineage tree), cycle 8 (kNN), cycle 41 (Cospar — supplies transition matrix T as optional input).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU CellRank 2-style batched GMRES absorption probability)` first comment.

## Risks

1. **cuSOLVER GMRES** is not in every CUDA distribution; fall back to a hand-written batched GMRES in shared memory if unavailable. cuSOLVER 12.x has `cusolverSpCpr*` functions for this.
2. **Ill-conditioned transition matrices** cause GMRES stagnation; add ILU(0) preconditioner.
3. **CellRank 2 Python install** is heavy (anndata + scvelo). Scipy fallback always runs.
4. **Restart overhead**: at high precision, 10 restarts of m=30 Krylov can cost 300 SpMVs per terminal; acceptable at target perf.
