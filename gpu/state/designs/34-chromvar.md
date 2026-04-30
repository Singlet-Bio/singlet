---
feature: chromvar_gpu
roadmap_id: 34
module: include/singlet-gpu/atac/chromvar.h + python/singlet_gpu/atac/chromvar.py + r/R/atac.R
status: design
tolerance: per-cell motif deviation Spearman ρ ≥ 0.95 vs chromVAR R; FDR-adjusted p-value rank Spearman ρ ≥ 0.90
target_perf: 100k cells × 500 motifs × 200k peaks ≤30s on A100 (chromVAR R is ~30 min)
ooc_plan: cell-batched scoring; motif PWM matching is per-peak
---

## Why this exists

Cycle 33 lit-scout: **chromVAR-GPU** is the 9th "first GPU implementation" candidate. No GPU motif enrichment scoring exists for scATAC-seq in any major library (chromVAR, scATAC-pro, ArchR, Signac all CPU). Reads singlify's `fragments.1pz` directly. Pairs naturally with cycle 29 STAGATE for spatial + motif activity maps.

## Algorithm — chromVAR (Schep et al. 2017)

```
Inputs:
  fragments (cell × peak): binary or count matrix of accessibility
  motif_in_peak (motif × peak): which motifs are present in which peaks (binary)
  background_peaks: matched background peaks per peak (controls GC + accessibility bias)

Outputs:
  per_cell_deviation (cell × motif): bias-corrected motif accessibility score
  variability (motif): variance across cells (high = informative TF)
  p_values (cell × motif): permutation-based significance

Algorithm:
  1. Compute expected accessibility per peak: E[peak] = mean(accessibility) over cells.
  2. Per cell: observed_score[motif, cell] = sum over peaks of (motif_in_peak[motif, peak] * accessibility[peak, cell]) / total_motif_size[motif]
  3. Per peak, sample N background peaks matched on (GC content, mean accessibility).
  4. Background score per (motif, cell): same as observed but averaged over background peak sets.
  5. Bias-corrected deviation = (observed - background) / background_std.
  6. Variability per motif: var(deviation) across cells.
  7. p-value via permutation of cell labels (one-tail).
```

## GPU implementation strategy

Native CUDA. The core operation is a sparse matrix-matrix multiply: `motif × peak * peak × cell = motif × cell`. cuSPARSE SpMM handles this perfectly.

### Kernels

- **Per-peak expected accessibility**: `cub::DeviceReduce` over rows of accessibility matrix.
- **Background peak matching**: pre-computed on host (one-time setup), uploaded as `background_idx[n_peaks × n_bg]`.
- **Observed score**: cuSPARSE SpMM (motif_in_peak × accessibility) → motif × cell scores.
- **Background score**: same SpMM but with the background-shuffled accessibility (compute N=50 times via Philox seeds, average).
- **Deviation = (observed - mean_bg) / std_bg**: elementwise kernel.
- **Variability per motif**: `cub::DeviceReduce::Sum` per row.
- **p-value via permutation**: shuffle cell labels K=1000 times, recompute deviation, compare to observed.

The permutation is the expensive part. Mitigation: vectorize across permutations (build `K × n_cells` shuffled label matrix, do K SpMMs in batch).

## Numerical stability

- fp32 throughout. Bias-corrected deviation has variance close to 1 by construction; no overflow.
- p-value computed in fp64 to handle small tails.

## Memory layout

- Input: `fragments.1pz` loaded as `peak × cell` CSC (cycle 2 loader).
- `motif_in_peak`: sparse `motif × peak` (loaded from external BED file via host parser).
- Workspace: `motif × cell` deviation matrix (4 bytes × n_motifs × n_cells). For 500 × 100k: 200 MB.
- Background peak storage: `n_peaks × n_bg` int (8 bytes × 200k × 50 = 80 MB).
- Total: ~300 MB.

## Streams

One stream, caller-provided.

## Out-of-core

Cell-batched: process N=50k cells at a time. The motif × peak matrix is global.

## Determinism

cuRAND Philox seeded for background peak sampling + permutation.

## Correctness test spec

Test: `tests/atac_chromvar_correctness.cpp`.

Reference: chromVAR R via Rscript subprocess.

Test cases:
1. **`ChromVAR_TinySynthetic_VsR`**: 200 cells × 500 peaks × 20 motifs synthetic. Spearman ρ ≥ 0.95 on per-cell deviation vs chromVAR R.
2. **`ChromVAR_GSM_RealData_Atac`**: load a real ATAC `.1pz` (when available; skip if not). Confirm finite results.
3. **`ChromVAR_VariabilityRanking`**: confirm motif variability ranking matches chromVAR R Spearman ρ ≥ 0.95.
4. **`ChromVAR_PValuePermutation_Calibrated`**: under the null (random motif assignments), p-values are uniformly distributed.
5. **`ChromVAR_Determinism_BitIdentical`**: bit-identical with fixed seed.

## Target performance

| Scale | Cells | Motifs | Peaks | Wall (target) | chromVAR R (CPU) |
|---|---|---|---|---|---|
| tiny | 200 | 20 | 500 | <100ms | ~5s |
| 10k | 11,560 | 500 | 200k | <5s | ~5min |
| 100k | ~120k | 500 | 200k | <30s | ~30min |

## Implementation notes

- Header path: `include/singlet-gpu/atac/chromvar.h` (~1200 LOC).
- New module path `singlet-gpu/atac/` (NEW).
- Python wrapper: `python/singlet_gpu/atac/chromvar.py` (~150 LOC).
- R wrapper: `r/R/atac.R` (~120 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuSPARSE + cuRAND + cub.
- Dependencies: cycle 1 (core), cycle 2 (loader for fragments.1pz).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU chromVAR for scATAC-seq motif enrichment)` first comment.

## Risks

1. **Background peak matching** is non-trivial: must control for GC content + mean accessibility. Default: K-means cluster peaks into 100 bins, sample within bins.
2. **Permutation count**: K=1000 is the standard; reduce to K=100 for the test.
3. **Motif PWM file format**: chromVAR uses JASPAR; we accept a pre-computed motif × peak BED. Document.
4. **chromVAR R install** is heavy (BiocFileCache + many deps). Skip cleanly.
