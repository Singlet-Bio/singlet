---
feature: cellchat_gpu
roadmap_id: 37
module: include/singlet-gpu/comm/cellchat.h + python/singlet_gpu/comm/cellchat.py + r/R/comm.R
status: design
tolerance: per-L-R-pair communication probability Spearman ρ ≥ 0.90 vs CellChat R; permutation p-value rank Spearman ρ ≥ 0.90
target_perf: 100k cells × 1000 L-R pairs × 20 cell types × K=1000 permutations ≤10 min on A100 (CellChat R is ~2 hours)
ooc_plan: L-R pair chunking; cell-type batched pseudobulk
---

## Why this exists

Cycle 36 lit-scout: **CellChat** (Jin et al. Nature Protocols 2024, Nat Comm 2024) is the leading cell-cell communication inference tool. Entirely CPU. Permutation test (K=1000 cell-type label shuffles across 20+ cell types × 1000 L-R pairs) is the bottleneck — massively parallel but never ported to GPU. LIANA, CellPhoneDB, NicheNet all CPU as well. 12th "first GPU" candidate. Closes the "who is talking to whom" question for every tumor / developmental / immune dataset.

## Algorithm — CellChat

```
Inputs:
  expression (n_cells × n_genes): log-normalized GEX
  cell_type (n_cells): cluster or annotation label
  lr_db: list of (ligand_genes, receptor_genes, pathway) tuples from CellChatDB
         ~1000-2000 entries for human
  (optional) spatial_coords: for proximity weighting in Visium

Outputs:
  comm_prob (n_types × n_types × n_lr): communication probability
  p_values  (n_types × n_types × n_lr): permutation-based significance
  pathway_activity (n_types × n_types × n_pathways): aggregated by pathway
  network edges: (sender, receiver, lr, prob, pval) sparse list

Algorithm:
  1. Compute per-cell-type mean expression: B[t, g] = mean(expression[cells where cell_type==t, g])
  2. For each L-R pair (ligand_set L, receptor_set R):
       a. ligand_expr[t]   = Hill_function(sum over g in L of B[t, g])
       b. receptor_expr[t] = Hill_function(sum over g in R of B[t, g])
       c. comm_prob[s, r, lr] = ligand_expr[s] * receptor_expr[r]
       d. (optional) multiply by cofactor activation/inhibition terms
  3. Permutation test: shuffle cell_type labels K=1000 times, recompute B, then comm_prob.
     p_value[s, r, lr] = fraction of permutations with prob ≥ observed.
  4. Aggregate by pathway via sum over constituent L-R pairs.
  5. (optional) Spatial filter: zero out communication if sender-receiver spots are > threshold apart.
```

## GPU implementation strategy

Native CUDA. Core operations are one big SpMM + a batched permutation kernel.

### Kernels

1. **Per-cell-type mean expression** (`pseudobulk_mean_kernel`): segmented mean over `expression` rows grouped by `cell_type`. cuSPARSE SpMM of `one_hot_type (n_types × n_cells) × expression (n_cells × n_genes)` / `cell_count[t]` → `B (n_types × n_genes)`. fp32.

2. **L-R pair scoring** (`hill_and_product_kernel`): for each L-R pair, gather the ligand and receptor gene indices, compute the Hill-function of summed expression per cell type, then outer-product to get `n_types × n_types` communication matrix.
   - Best layout: precompute dense `ligand_expr (n_lr × n_types)` and `receptor_expr (n_lr × n_types)` via a small gather kernel. Then the outer product per L-R pair is a batched GEMM.

3. **Permutation test** (the bottleneck): K=1000 iterations. Each iteration:
   - Shuffle `cell_type` labels via Philox4x32_10.
   - Recompute `B_perm` via cuSPARSE SpMM (reuse the kernel from step 1).
   - Recompute `comm_prob_perm` via step 2's batched GEMM.
   - Atomically increment `pvalue_count[s, r, lr]` where `prob_perm ≥ prob_observed`.
   - Batch permutations in groups of 50 (`PERM_BATCH=50`) to amortize kernel launch overhead. `cub::DeviceRadixSort` for the label shuffle on-device.

4. **Pathway aggregation** (`pathway_reduce_kernel`): segmented sum over L-R pairs grouped by pathway. `cub::DeviceSegmentedReduce::Sum`.

5. **Spatial filter** (optional): elementwise zero out comm_prob entries where spatial distance between representative spots exceeds threshold. Reuses cycle 29 spatial distance kernel.

## Numerical stability

- fp32 throughout for the hot path. Hill function `x / (K + x)` is stable.
- Permutation counter in uint32 (K ≤ 2^32). Final p-value = `(count + 1) / (K + 1)` in fp32.
- Pseudobulk mean: straightforward reduction; no Kahan needed at K=1000 samples per cell type.

## Memory layout

- Input: `expression` CSC (n_cells × n_genes) via cycle 0 loader.
- `cell_type`: int array, n_cells.
- `lr_db`: host-side struct, flattened into 3 device arrays: `ligand_offsets[n_lr+1]`, `ligand_gene_idx[L_total]`, `receptor_offsets[n_lr+1]`, `receptor_gene_idx[R_total]`.
- `B`: dense (n_types × n_genes). For 20 × 20k: 1.6 MB. Tiny.
- `comm_prob`: (n_types × n_types × n_lr) × 4 bytes. For 20² × 1000: 1.6 MB.
- `pvalue_count`: same shape, uint32. 1.6 MB.
- `B_perm` per iter: reuses `B` buffer.
- Total: ~50 MB workspace.

## Streams

One stream, caller-provided. All K=1000 permutations on the same stream (the SpMM + GEMM fully occupy the device).

## Out-of-core

L-R pair chunking: process chunks of 256 pairs at a time. Cell-type batched if n_types > 50.

## Determinism

cuRAND Philox4x32_10 seeded for label permutations. `cfg.deterministic=true` uses `cub::DeviceSegmentedReduce` instead of atomic for the p-value counter (the atomicAdd race is the only non-determinism source).

## Correctness test spec

Test: `tests/comm_cellchat_correctness.cpp`.

Reference: CellChat R via Rscript subprocess.

Test cases:
1. **`CellChat_TinySynthetic_VsR`**: 200 cells × 100 genes × 3 cell types × 10 L-R pairs synthetic with known ground truth. Spearman ρ ≥ 0.90 on comm_prob vs R.
2. **`CellChat_GSM_RealData`**: load a real scRNA sample with author cell-type annotations (skip if annotations missing). Confirm finite results, non-zero edge count.
3. **`CellChat_PermutationPvalues_Calibrated`**: under the null (random cell-type labels), p-values are uniformly distributed (KS tolerance 0.10, K=100).
4. **`CellChat_PathwayAggregation_VsR`**: pathway-level aggregation Spearman ρ ≥ 0.95.
5. **`CellChat_Determinism_BitIdentical`**: bit-identical with fixed seed.

## Target performance

| Scale | Cells | Types | LR pairs | Wall (target) | CellChat R |
|---|---|---|---|---|---|
| tiny | 200 | 3 | 10 | <100ms | ~2s |
| 10k | 11,560 | 15 | 1000 | <30s | ~10 min |
| 100k | ~120k | 20 | 2000 | <10 min | ~2 hours |

## Implementation notes

- Header path: `include/singlet-gpu/comm/cellchat.h` (~900 LOC, under 1500 cap).
- New module path `singlet-gpu/comm/` (NEW).
- Python wrapper: `python/singlet_gpu/comm/cellchat.py` (~150 LOC).
- R wrapper: `r/R/comm.R` (~120 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuSPARSE + cuBLAS + cuRAND + cub.
- Dependencies: cycle 1 (core), cycle 2 (loader), cycle 10 (cell-type annotation as input).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU CellChat-style cell-cell communication)` first comment.
- Use factornet field-access style `.col_ptr.data()`, `.row_idx.data()`, `.vals.data()`.

## Risks

1. **CellChatDB parsing**: CellChatDB is an R data object. Ship a pre-parsed TSV (`cellchatdb_v2_human.tsv`) as a test fixture; accept a TSV path in the API.
2. **CellChat R install** is heavy (Seurat + NMF + igraph). Skip cleanly (exit code 2).
3. **Ligand/receptor complex subunit logic**: CellChat uses a min() or geometric-mean over subunits; match R behavior. Document the choice.
4. **Permutation count**: K=1000 is standard; reduce to K=100 for the test to fit in CI.
5. **Cell-type count explosion**: comm_prob scales as `n_types²`. Cap at n_types ≤ 100 with a warning.
