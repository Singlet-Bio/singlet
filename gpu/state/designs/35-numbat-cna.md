---
feature: numbat_cna
roadmap_id: 35
module: include/singlet-gpu/cna/numbat.h + python/singlet_gpu/cna/numbat.py + r/R/cna.R
status: design
tolerance: per-cell CNA call agreement ≥ 0.85 vs Numbat R; clone label ARI ≥ 0.80 on tumor samples with known clones
target_perf: 100k cells × 30k genes × 50 segments ≤2 min on A100 (Numbat R is ~30 min)
ooc_plan: per-chromosome segmentation; cell-batched HMM likelihood
---

## Why this exists

Cycle 34 lit-scout: Numbat (Gao et al. 2024) is the top CNA detection method, outperforming CopyKAT/inferCNV by ≥80% accuracy. No GPU implementation exists. 10th "first GPU" candidate. CNA detection is high-value for tumor samples (~40% of GEO catalog).

## Algorithm — Numbat

```
Inputs:
  expression (n_cells × n_genes)
  gene_chr_position (n_genes): chromosome + start position per gene
  reference_normal (optional): reference normal cell expression for log ratio computation

Outputs:
  cna_state (n_cells × n_segments): per-segment copy number state (loss / neutral / gain)
  clone_labels (n_cells): clonal assignment from CNA pattern
  segment_boundaries: chromosome segment definitions

Algorithm:
  1. Compute per-gene log expression ratio vs reference normal (or the bulk mean if no ref).
  2. Smooth log ratios along chromosomes (rolling window per chr).
  3. HMM segmentation per chromosome: states = {loss, neutral, gain}, emissions = smoothed log ratios.
  4. Per-segment per-cell CNA call from the HMM posterior.
  5. Cluster cells by CNA pattern → clone labels.
  6. Optional: integrate with allele-specific information (snp_ad/snp_dp from singlify).
```

## GPU implementation strategy

Native CUDA. Three main kernels:

1. **Per-gene log ratio** (one warp per gene): row-wise mean + per-cell log ratio.
2. **Chromosome smoothing** (per-chromosome convolution): cuFFT for FFT-based convolution OR direct kernel for short windows.
3. **HMM forward-backward** per chromosome per cell: 3-state HMM with Viterbi decoding. ~50 segments per chromosome × 22 chromosomes × 100k cells = 110M HMM cells.
4. **Clone clustering**: cycle 9 leiden on the CNA pattern.

## Numerical stability

- fp32 throughout. HMM in log-space to avoid underflow.
- Per-segment posterior in fp32; segment boundaries fixed.

## Memory layout

- Input: expression CSC (n_cells × n_genes).
- Per-gene log ratio: dense `n_cells × n_genes × 4` bytes — too big for 100k × 30k. Tile by chromosome.
- Per-chromosome workspace: `n_cells × genes_per_chr × 4`. For 1500 genes/chr × 100k: 600 MB. Manageable.
- HMM state: `n_cells × n_segments × 3 × 4` bytes (forward + backward). For 100k × 50 × 3 = 60 MB.
- Output: `n_cells × n_segments × 1` byte (state) + `n_cells × 4` (clone label).

## Streams

One stream, caller-provided. Per-chromosome tile loop.

## Out-of-core

Per-chromosome processing keeps memory bounded. Each chromosome is independent.

## Determinism

cuRAND Philox seeded for clone clustering init.

## Correctness test spec

Test: `tests/cna_numbat_correctness.cpp`.

Reference: Numbat R via Rscript subprocess.

Test cases:
1. **`Numbat_TinySynthetic_VsR`**: 200 cells × 500 genes synthetic with planted CNA in a known segment. Per-cell CNA call agreement ≥ 0.85 vs R.
2. **`Numbat_GSM_RealTumor`**: load a real tumor sample (skip if not available). Confirm finite results, clone labels in [1, 10].
3. **`Numbat_HmmStateProbs_SumToOne`**: per-cell-per-segment HMM posteriors sum to 1.
4. **`Numbat_Determinism_BitIdentical`**: bit-identical with fixed seed.

## Target performance

| Scale | Cells | Segments | Wall (target) |
|---|---|---|---|
| tiny | 200 | 10 | <100ms |
| 10k | 11,560 | 50 | <10s |
| 100k | ~120k | 50 | <2 min |

## Implementation notes

- Header path: `include/singlet-gpu/cna/numbat.h` (~1000 LOC).
- New module path `singlet-gpu/cna/`.
- Python wrapper: `python/singlet_gpu/cna/numbat.py` (~150 LOC).
- R wrapper: `r/R/cna.R` (~120 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuRAND + cub + cuFFT (for smoothing).
- Dependencies: cycle 1, cycle 2, cycle 9 (leiden for clone clustering).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU Numbat-style CNA detection)` first comment.

## Risks

1. **HMM forward-backward in CUDA** is non-trivial. Each cell's HMM is independent → embarrassingly parallel.
2. **Numbat R install** is heavy. Skip cleanly.
3. **Reference normal** is optional; if absent, use the bulk cell mean as the baseline.
4. **Allele-specific extension** (using snp_ad/snp_dp from singlify) is deferred to a follow-up cycle.
