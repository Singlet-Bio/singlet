---
feature: hdwgcna
roadmap_id: 46
module: include/singlet-gpu/network/hdwgcna.h + python/singlet_gpu/network/hdwgcna.py + r/R/network.R
status: design
tolerance: module assignment ARI ≥ 0.90 vs hdWGCNA R on shared HVG; hub gene rank Spearman ρ ≥ 0.95
target_perf: 100k cells × 5000 HVG ≤1 min on A100 (hdWGCNA R is ~20-40 min)
ooc_plan: correlation matrix is dense O(HVG²); 5000×5000×4 bytes = 100 MB — fits; 20k HVG × 20k = 1.6 GB upper bound
---

## Why this exists

Cycle 45 lookahead: **hdWGCNA** (Morabito et al. Cell Reports 2023) extends the classic WGCNA weighted gene co-expression network analysis to single-cell data. **Entirely R**. Distinct from:
- **Cycle 11 fgsea**: ranked GSEA vs a gene list, not module discovery.
- **Cycle 27 Cell2fate / Cycle 29 STAGATE**: different tasks entirely.
- **Cycle 36 GRaNIE**: TF-target GRN inference, not gene co-expression modules.
- **Cycle 28 CSI-GEP**: consensus NMF for gene programs, not network-based module detection.

hdWGCNA builds a gene co-expression network, identifies modules via dynamic tree cutting, and computes eigengenes + hub genes. 22nd "first GPU" candidate.

## Algorithm — hdWGCNA

```
Inputs:
  expression (n_cells × n_genes): log-normalized GEX, filtered to HVG
  (optional) metacell_size: aggregate cells into pseudobulk metacells for stability

Outputs:
  module_assignment (n_genes): module label per gene (0 = unassigned)
  eigengenes (n_cells × n_modules): first PC per module across cells
  hub_genes (n_modules × n_top): top-N hub genes per module by kME
  module_preservation: z-score per module (optional, cross-dataset)

Algorithm:
  1. Metacell construction (optional): kNN aggregate cells into metacells to smooth expression.
  2. Soft thresholding: compute Pearson correlation matrix |r|^beta where beta is chosen
     for scale-free topology. Default beta=6.
  3. Topological overlap matrix (TOM):
       TOM[i,j] = (sum_k(|r[i,k]|^beta * |r[j,k]|^beta) + |r[i,j]|^beta) / (min(k_i, k_j) + 1 - |r[i,j]|^beta)
  4. Distance: dist[i,j] = 1 - TOM[i,j].
  5. Hierarchical clustering: average linkage on the distance matrix.
  6. Dynamic tree cutting: cut the dendrogram at multiple heights, merge similar branches.
  7. Eigengene: first principal component of the expression submatrix for each module.
  8. Hub genes: kME (module eigengene correlation) per gene within its module; top-N are hubs.
```

## GPU implementation strategy

Native CUDA. The hot paths are Pearson correlation matrix (cuBLAS SYRK), topological overlap (custom kernel), hierarchical clustering (per-module problem, `O(n²)` small), and eigengene PCA (one Randomized SVD per module).

### Kernels

1. **Metacell construction** (optional, `metacell_kernel`): reuse cycle 8 kNN, then cub segmented sum over kNN groups.

2. **Pearson correlation matrix** (`correlation_kernel`):
   - Center expression per gene (two-pass Welford).
   - `cuBLAS cublasSsyrk` on centered `(n_genes × n_cells)`: output `(n_genes × n_genes)` symmetric correlation.
   - Divide by sqrt(var[i]*var[j]) elementwise.

3. **Soft power + TOM** (`tom_kernel`):
   - `|r|^beta` elementwise.
   - TOM numerator via cuBLAS GEMM of `(|r|^beta) @ (|r|^beta)`.
   - Denominator via per-row degree computation.

4. **Hierarchical clustering** (`average_linkage_kernel`):
   - Start from distance matrix `1 - TOM`.
   - Iterative min-pair merge via `cub::DeviceReduce::Min`.
   - At each merge, update distance row/col on device.
   - `O(n² log n)` on the GPU.

5. **Dynamic tree cutting** (`dynamic_cut_kernel`):
   - Bottom-up BFS from leaves through the merge tree.
   - Cut when cluster size or height criterion met.
   - Small, host-side OK for the tree traversal (n_genes ≤ 20k).

6. **Eigengene computation**: for each module, extract submatrix, Randomized SVD (reuse cycle 4 factornet adapter), top-1 right singular vector = eigengene.

7. **Hub gene kME**: correlation of each gene in a module with its eigengene. Per-module kernel.

## Numerical stability

- fp32 throughout. Pearson with two-pass Welford.
- Correlation matrix bounded in [-1, 1]; no overflow.
- Beta exponentiation: `powf(|r|, beta)` stable.

## Memory layout

- Input: expression CSC via cycle 0 loader.
- Centered dense: `(n_HVG × n_cells) × 4 bytes`. At 5000 × 100k: 2 GB. Stream/chunk or use cuSPARSE SpMM against centered representation.
- Correlation matrix: `(n_HVG × n_HVG) × 4 bytes`. At 5000 × 5000: 100 MB.
- TOM: same size, 100 MB.
- Distance matrix: same, 100 MB.
- Eigengenes: `n_cells × n_modules × 4 bytes`. At 100k × 50: 20 MB.
- Total: ~2.5 GB at 5k HVG × 100k cells.

## Streams

One stream. Correlation → TOM → clustering is sequential.

## Out-of-core

At n_HVG > 20k, correlation matrix > 1.6 GB: chunk by HVG block and accumulate TOM incrementally.

## Determinism

All kernels deterministic. cuBLAS SYRK is non-deterministic by default — use `cublasSetAtomicsMode(CUBLAS_ATOMICS_NOT_ALLOWED)` for bit-identical. Deterministic mode is opt-in.

## Correctness test spec

Test: `tests/network_hdwgcna_correctness.cpp`.

Reference: hdWGCNA R via Rscript (heavy: WGCNA + Seurat). Fallback: pure-R WGCNA classic path (no hdWGCNA), or pure-Python scipy correlation + scipy.cluster.hierarchy.

5 test cases:
1. **`Hdwgcna_TinySynthetic_VsR`**: 200 cells × 200 genes × 5 planted modules. ARI ≥ 0.90 on module assignment vs R.
2. **`Hdwgcna_GSM_RealData`**: real scRNA sample + top 1000 HVG. Confirm finite results, ≥ 3 modules detected.
3. **`Hdwgcna_TomSymmetric`**: TOM matrix is symmetric within 1e-5.
4. **`Hdwgcna_EigengeneVariance`**: first eigengene explains highest variance of its module expression.
5. **`Hdwgcna_Determinism_BitIdentical`**: bit-identical with deterministic mode.

## Target performance

| Scale | Cells | HVG | Modules | Wall (target) | hdWGCNA R |
|---|---|---|---|---|---|
| tiny | 200 | 200 | 5 | <100ms | ~10s |
| 10k | 11,560 | 2000 | 20 | <10s | ~5 min |
| 100k | ~120k | 5000 | 50 | <1 min | ~30 min |

## Implementation notes

- Header path: `include/singlet-gpu/network/hdwgcna.h` (~1100 LOC).
- New module path `singlet-gpu/network/` (NEW).
- Python wrapper: `python/singlet_gpu/network/hdwgcna.py` (~150 LOC).
- R wrapper: `r/R/network.R` (~120 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuBLAS + cuSPARSE + cub.
- Dependencies: cycle 1 (core), cycle 2 (loader), cycle 3 (hvg for preselection), cycle 4 (factornet Randomized SVD for eigengenes), cycle 8 (kNN for metacells).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU hdWGCNA-style scRNA co-expression network analysis)` first comment.

## Risks

1. **Hierarchical clustering on GPU**: n² min-pair search can be expensive. Use `cub::DeviceReduce::ArgMin` per merge iteration.
2. **WGCNA R install** is heavy (Bioconductor + WGCNA). Fall back to scipy hierarchy.
3. **Metacell path** adds complexity; make it optional and default OFF for MVP.
4. **Dendrogram data structure**: serialize merge tree as `merge[n_merges][2]` int array + `heights[n_merges]` float array, matching scipy's format.
5. **Module preservation** (z-score across datasets) deferred to followup.
