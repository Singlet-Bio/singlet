---
feature: granie_gpu
roadmap_id: 36
module: include/singlet-gpu/grn/granie.h + python/singlet_gpu/grn/granie.py + r/R/grn.R
status: design
tolerance: per-TF-target edge ranking Spearman ρ ≥ 0.90 vs GRaNIE R; top-1000 edge Jaccard ≥ 0.80
target_perf: 100k cells × 10k TFs × 20k target genes × 50k peaks ≤5 min on A100 (GRaNIE R is ~4-6 hours)
ooc_plan: per-TF chunked edge scoring; peak batching for gene-peak assignment
---

## Why this exists

Cycle 36 lit-scout: **GRaNIE** (Gaumondo et al., Nature Methods 2024) is the top multimodal GRN inference method integrating scATAC peaks + scRNA expression + TF motif binding. SCENIC/pySCENIC/Dictys are all CPU. No GPU GRN inference exists in rapids-singlecell, cuml, or cuGraph. 11th "first GPU" candidate. High-value because GRN inference is the bottleneck step in every multiome analysis pipeline.

## Algorithm — GRaNIE

```
Inputs:
  gex (n_cells × n_genes): scRNA gene expression
  peaks (n_cells × n_peaks): scATAC peak accessibility (from fragments.1pz)
  tf_motif_in_peak (n_tfs × n_peaks): which TFs have motif hits in which peaks (binary)
  peak_gene_distance (n_peaks × n_genes): cis-distance (sparse, capped at 500 kb)
  tf_gene_name_map (n_tfs → gene index): TFs that are themselves expressed genes
  (optional) snp_ad/snp_dp: SNP colocalization validation

Outputs:
  tf_activity (n_cells × n_tfs): per-cell TF activity score
  gene_peak_links (n_peaks × n_genes): filtered peak→gene correlation edges
  tf_target_edges: (tf, target, score, p_value) sparse edge list
  grn_network: igraph-compatible edge list with community labels

Algorithm:
  1. Peak filtering: filter peaks by mean accessibility + variance across cells.
  2. Peak-gene correlation: per (peak, gene) in cis window, Pearson correlation.
     Filter pairs with |r| > 0.3 and FDR < 0.05.
  3. TF activity: for each TF t, TF_activity[c,t] = mean(peak_accessibility[c, peaks where t binds]).
  4. TF-target linking: for each TF t and target gene g,
       score(t,g) = corr(TF_activity[*,t], gex[*,g]) × mean(peak_gene_corr for peaks with t binding)
  5. Edge filtering: FDR correction across (TF, target) space.
  6. Community detection: optional Leiden on the filtered TF-target bipartite graph.
  7. Optional SNP colocalization validation from snp_ad/snp_dp.
```

## GPU implementation strategy

Native CUDA. Core operations are sparse matrix multiplies + correlation computations.

### Kernels

1. **Peak filtering** (one block per peak): fused mean + variance via Welford on CSC peaks, threshold filter. Reuses cycle 3 HVG pattern.

2. **Peak-gene correlation** (the main bottleneck): for each (peak, gene) pair in cis window, compute Pearson correlation across cells.
   - Strategy: build a sparse `(peak, gene)` pair list from `peak_gene_distance` (≤500 kb cutoff) → ~5M pairs for 50k peaks × 20k genes.
   - For each pair, reduce `(peak_vals · gene_vals - mean_p * mean_g * n) / (sd_p * sd_g)` via a 1-block-per-pair kernel.
   - Fused on CSC-streamed cell values: each block processes one pair and all n_cells.

3. **TF activity computation**: cuSPARSE SpMM of `tf_motif_in_peak (n_tfs × n_peaks) × peak_accessibility (n_peaks × n_cells)` → `tf_activity (n_tfs × n_cells)`, normalized row-wise by tf_motif_size.

4. **TF-target scoring**: elementwise Pearson over TF activity × gene expression. cuBLAS GEMM-friendly.

5. **FDR correction**: Benjamini-Hochberg in-place sort + rank on device (cycle 11 fgsea pattern).

6. **Community detection**: reuse cycle 7 `leiden.h` on the filtered bipartite edge list.

## Numerical stability

- fp32 throughout. Pearson correlation uses two-pass Welford for mean/variance.
- Correlation denominator epsilon = 1e-8 to avoid div-by-zero on constant-expression genes.

## Memory layout

- Input: `gex` CSC (n_cells × n_genes), `peaks` CSC (n_cells × n_peaks). Both via cycle 0 loader.
- `tf_motif_in_peak`: sparse `n_tfs × n_peaks` (external TF-motif database, e.g., JASPAR).
- Peak-gene pair list: `int2[5M] × 8 bytes = 40 MB`.
- Peak-gene correlation results: `float[5M] × 4 bytes = 20 MB`.
- TF activity: `n_tfs × n_cells × 4 bytes`. For 500 × 100k: 200 MB.
- TF-target scores: `n_tfs × n_genes × 4 bytes`. For 500 × 20k: 40 MB.
- Total: ~350 MB workspace.

## Streams

One stream, caller-provided. Peak-gene correlation is the dominant kernel.

## Out-of-core

Per-TF chunked scoring: process N=100 TFs at a time, aggregate edge list. Peak-gene correlation is one-shot (~20 MB output) so no chunking needed.

## Determinism

cuRAND Philox seeded for FDR null distribution if permutation mode is enabled.

## Correctness test spec

Test: `tests/grn_granie_correctness.cpp`.

Reference: GRaNIE R via Rscript subprocess (Bioconductor).

Test cases:
1. **`GRaNIE_TinySynthetic_VsR`**: 200 cells × 100 genes × 50 peaks × 10 TFs synthetic with planted TF→target relationships. Top-5 edge recall ≥ 0.80 vs R.
2. **`GRaNIE_GSM_RealMultiome`**: load a real multiome sample (skip if unavailable). Confirm finite results, non-zero edge count.
3. **`GRaNIE_PeakGeneCorrelation_VsCpu`**: Pearson ρ between GPU and CPU peak-gene correlation matrix ≥ 0.99.
4. **`GRaNIE_TfActivity_VsR`**: per-cell TF activity Spearman ρ ≥ 0.95 vs GRaNIE R.
5. **`GRaNIE_Determinism_BitIdentical`**: bit-identical with fixed seed.

## Target performance

| Scale | Cells | TFs | Genes | Peaks | Wall (target) |
|---|---|---|---|---|---|
| tiny | 200 | 10 | 100 | 50 | <100ms |
| 10k | 11,560 | 500 | 20k | 50k | <30s |
| 100k | ~120k | 500 | 20k | 50k | <5 min |

GRaNIE R baseline: ~4-6 hours at 100k scale → target ≥50× speedup.

## Implementation notes

- Header path: `include/singlet-gpu/grn/granie.h` (~1000 LOC, under 1500 cap).
- New module path `singlet-gpu/grn/` (NEW).
- Python wrapper: `python/singlet_gpu/grn/granie.py` (~150 LOC).
- R wrapper: `r/R/grn.R` (~120 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuSPARSE + cuBLAS + cuRAND + cub.
- Dependencies: cycle 1 (core), cycle 2 (loader), cycle 7 (leiden for community detection).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU GRaNIE-style multimodal GRN inference)` first comment.

## Risks

1. **Peak-gene correlation scales as O(peak_gene_pairs × n_cells)**. 5M pairs × 100k cells = 500G ops. At 10 TFLOPS on A100 = 50s. Tight but feasible with fused reductions.
2. **TF motif database** is heavy (JASPAR). Accept a pre-computed `tf_motif_in_peak` binary matrix as input rather than parsing motifs on device.
3. **GRaNIE R install** is heavy (Bioconductor + genomics deps). Skip cleanly on missing dependency.
4. **Multiome requirement**: cycle 36 requires both gex AND peaks. For scRNA-only samples, fall back to SCENIC-style TF activity from expression alone and document the fallback.
5. **Cell-type-specific GRN** variant is deferred to a follow-up cycle.
