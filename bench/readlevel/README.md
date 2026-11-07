# Read-level outputs: validation

Checks of the read-level members that the Beta freeze keeps against open or
reference tools on real data.

## Reads per UMI against Cell Ranger molecule_info (pbmc1k_v3)

`rpu_vs_molinfo.py` bins Cell Ranger 3.0's `pbmc_1k_v3_molecule_info.h5` (one row per molecule, `count` = reads) into the
engine's 16 bins and compares it with the engine's `reads_per_umi_hist.tsv.gz` over the cells both callers keep.
`jobs/rpu.sbatch` runs it on Clipper against the CPU reference of the frontier gate (commit 654109ad, which gate 803937 found
identical to the GPU path on this sample).

Result of job 803959 (`results/rpu_pbmc1k_v3.json`):

| Metric | Engine | Cell Ranger |
|---|---|---|
| Cells compared | 1,217 | (1,222 filtered) |
| Median of the per-cell mean reads per UMI | 3.59 | 3.65 |
| Molecules with one read | 21.6% | 20.9% |
| Sequencing saturation over these cells | 71.6% | 72.2% |
| Per-cell mean reads per UMI, Pearson / Spearman | 0.967 / 0.961 | |
| Median per-cell relative difference in mean reads per UMI | 2.2% | |
| Pooled histogram, total variation distance | 0.014 | |
| Per-cell histogram, median total variation distance | 0.018 | |

The engine emits 1.48 times as many molecules per cell (log-scale Pearson 0.979) because the histogram covers intronic
molecules as well; Cell Ranger 3.0 counts exonic molecules only. The mean of the last bin counts 16+ as 16 on both sides.
