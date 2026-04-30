---
feature: plaid_progeny
roadmap_id: 44
module: include/singlet-gpu/enrich/ssgsea.h + include/singlet-gpu/enrich/progeny.h + python/singlet_gpu/enrich/{ssgsea,progeny}.py + r/R/enrich.R
status: design
tolerance: per-cell ssGSEA score Spearman ρ ≥ 0.95 vs GSVA R; PROGENy pathway activity Spearman ρ ≥ 0.95 vs decoupleR
target_perf: 100k cells × 1000 gene sets × 20k genes ≤30s on A100 (GSVA R is ~20 min; PROGENy R is ~5 min)
ooc_plan: cell-batched processing; gene-set chunking at 1k sets
---

## Why this exists

Cycle 43 lookahead: **PLAID** (Bioinformatics 2025) is a fast single-sample GSEA method; **PROGENy** (Schubert et al. Nature Commun 2018) is a pathway activity scoring method based on pre-trained gene weights from perturbation experiments. Both are entirely CPU (R / Python). Both are distinct from cycle 11 (fgsea ranked GSEA + AUCell top-N).

Cycles 11 computed:
- **fgsea**: ranked gene list → enrichment across a full ordered list (used for differential expression analysis).
- **AUCell**: per-cell AUC of top-N ranked genes in a gene set.

Cycles 44 computes:
- **ssGSEA** (PLAID): per-cell Kolmogorov-Smirnov enrichment score for each gene set, one score per (cell, set) pair. Used for module scoring at scale.
- **PROGENy**: per-cell pathway activity = weighted sum of expression * pre-trained pathway weight matrix. Used for upstream/downstream signaling inference.

19th + 20th "first GPU" candidates bundled into one cycle because both are fast, share the "score per cell per set" pattern, and are small enough to fit together (~500 LOC each).

## Algorithm — ssGSEA (PLAID)

```
Inputs:
  expression (n_cells × n_genes): log-normalized GEX
  gene_sets: list of (set_name, [gene_idx]) tuples, typically 1000-5000 sets

Outputs:
  ssgsea_score (n_cells × n_sets): per-cell enrichment score
  p_value (n_cells × n_sets): beta-approximation p-value (optional)

Algorithm (per cell):
  1. Rank all genes by expression (ties broken randomly).
  2. For each gene set S:
       a. Create indicator vector v[g] = 1 if g ∈ S, else 0.
       b. Compute weighted ECDF:
          - "hit" step: rank^alpha / sum(rank^alpha for g in S)
          - "miss" step: -1 / (n_genes - |S|)
       c. Enrichment score = max(cumulative_sum) - min(cumulative_sum) (two-sided KS).
```

## Algorithm — PROGENy

```
Inputs:
  expression (n_cells × n_genes): log-normalized GEX
  progeny_weights (n_pathways × n_genes): pre-trained pathway weight matrix
                                           (100 top genes per pathway, non-zero elsewhere zero)

Outputs:
  pathway_activity (n_cells × n_pathways): weighted sum score

Algorithm:
  activity = expression @ progeny_weights^T  (a single cuBLAS GEMM)
  (optional) normalize per pathway: (activity - mean(activity[p])) / std(activity[p])
```

## GPU implementation strategy

Two separate headers, one compilation unit each, but designed together.

### ssGSEA kernels (`enrich/ssgsea.h`)

1. **Per-cell rank** (`rank_expression_kernel`): sort genes by expression within each cell. `cub::BlockRadixSort` per-cell (fast when n_genes ≤ 32k).

2. **Per-(cell, set) KS statistic** (`ks_enrichment_kernel`): one block per (cell, set) pair.
   - Each thread iterates over sorted genes, accumulates running sum with hit/miss step.
   - Warp-shuffle reduction for max - min.
   - Output one fp32 score per (cell, set).

3. **Beta p-value approximation** (`ks_pvalue_kernel`): fit beta distribution to the null distribution of ES across permuted gene sets, compute tail probability.

### PROGENy kernels (`enrich/progeny.h`)

Essentially one cuBLAS `sgemm`: `activity = expression @ progeny_weights^T`. 

1. **Dense matrix multiply**: cuBLAS `sgemm`. Input: sparse CSC expression + dense progeny_weights. Use cuSPARSE `SpMM` with dense result.

2. **Per-pathway normalization** (`pathway_normalize_kernel`): two-pass Welford over cell dim per pathway.

3. **Ship pre-trained weights**: include a `progeny_human_top100.tsv` fixture (~200 KB) with 14 pathways × top-100 genes. Host parses, uploads once.

## Numerical stability

- fp32 throughout. ssGSEA cumulative sums are stable (bounded).
- PROGENy SGEMM in fp32 is stable for log-normalized expression.

## Memory layout

- Input: expression CSC from cycle 0 loader.
- Gene sets: flattened host → device `set_offsets[n_sets+1]` + `gene_idx[total_nnz]`. At 1000 × 50 avg: 200 KB.
- ssGSEA output: `n_cells × n_sets × 4 bytes`. At 100k × 1000: 400 MB.
- PROGENy weights: `14 × n_genes × 4 bytes`. At 14 × 20k: 1.1 MB (tiny).
- PROGENy output: `n_cells × 14 × 4 bytes`. At 100k × 14: 5.6 MB.
- Total: ~500 MB.

## Streams

One stream per module. Both modules can run in parallel at the Python-wrapper level on two streams.

## Out-of-core

Cell-batched: 10k cells per batch. Gene-set chunked at 500 sets per batch.

## Determinism

ssGSEA: stable except for tie-breaking in sort. Use secondary key (gene index) for deterministic ordering. Bit-identical with fixed seed.

PROGENy: fully deterministic (cuBLAS SGEMM).

## Correctness test spec

Test: `tests/enrich_ssgsea_progeny_correctness.cpp`.

Reference: GSVA R (ssGSEA mode) + decoupleR Python (PROGENy). Fallback: pure-Python implementations of both.

6 test cases:
1. **`Ssgsea_TinySynthetic_VsGsva`**: 200 cells × 500 genes × 20 sets synthetic. Spearman ρ ≥ 0.95 on per-cell ssGSEA vs GSVA R.
2. **`Ssgsea_GSM_RealData`**: real scRNA sample. Confirm finite scores, non-zero variance across cells per set.
3. **`Ssgsea_Determinism_BitIdentical`**: bit-identical with fixed seed.
4. **`Progeny_TinySynthetic_VsDecoupleR`**: 200 cells × 500 genes synthetic + mock progeny weights. Spearman ρ ≥ 0.95 vs decoupleR.
5. **`Progeny_HumanTop100_RealData`**: real scRNA sample with shipped progeny weights. Confirm finite.
6. **`Progeny_Determinism_BitIdentical`**: bit-identical.

## Target performance

| Scale | Cells | Sets | Wall (target) | GSVA R |
|---|---|---|---|---|
| tiny | 200 | 20 | <100ms | ~5s |
| 10k | 11,560 | 1000 | <5s | ~3 min |
| 100k | ~120k | 5000 | <30s | ~20 min |

## Implementation notes

- Header path: `include/singlet-gpu/enrich/ssgsea.h` (~500 LOC) + `include/singlet-gpu/enrich/progeny.h` (~400 LOC).
- New module path: `singlet-gpu/enrich/` — NEW, distinct from cycle 11's `gsea/`.
- Python wrapper: `python/singlet_gpu/enrich/ssgsea.py` + `progeny.py` (~100 LOC each).
- R wrapper: `r/R/enrich.R` (~100 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuBLAS + cuSPARSE + cub.
- Dependencies: cycle 1 (core), cycle 2 (loader), cycle 11 (gsea/ module for complementary infrastructure).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU ssGSEA + PROGENy single-sample enrichment)` first comment on each.
- Fixture: `tests/refs/progeny_human_top100.tsv` (14 pathways × 100 genes = 1400 rows).

## Risks

1. **ssGSEA per-cell sort is hot**: `cub::BlockRadixSort` works up to 1024 threads × 32 items = 32k genes per cell. Beyond that, use `cub::DeviceRadixSort` segmented by cell.
2. **n_sets × n_cells output** is 400 MB at 100k × 1000 — chunk by set if pressure is high.
3. **GSVA R install** is heavy (BiocManager + GSVA). Pure-Python fallback via scipy rank + KS.
4. **PROGENy weights distribution**: ship as TSV fixture; document citation.
5. **Cycle 11 overlap**: clearly document that ssGSEA is ORTHOGONAL to fgsea (different algorithm entirely). PROGENy is ORTHOGONAL to AUCell (weighted sum vs top-N AUC).
