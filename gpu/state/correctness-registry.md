# singlet-gpu — Correctness Registry

Append-only. One row per `{feature, scale, metric}` reference-diff. Written by `analysis-validator` directly.

## Schema

| date | feature | scale | metric | value | tolerance | reference | commit | pass/fail |
|---|---|---|---|---|---|---|---|---|

## Conventions

- `date`: YYYY-MM-DD.
- `feature`: roadmap-style identifier.
- `scale`: `tiny` | `10k` | `100k` | `1m`.
- `metric`: explicit name — `rel_L2`, `subspace_angle`, `spearman_lfc`, `ari`, `nmi`, `trustworthiness`, `knn_preserve`, `frob_recon`, etc.
- `value`: numerical value of the metric.
- `tolerance`: declared threshold from the design doc.
- `reference`: `factornet_cpu`, `scanpy`, `scran`, `cuml`, `seurat`, `deseq2`, `fgsea`, `scvi`, `harmonypy`, `scib`, `mgatk`, `scvelo`, `umap_learn`, `singler`, `celltypist`, `singlify_io`, etc.
- `commit`: short SHA of the singlet-gpu commit tested.
- `pass/fail`: `PASS` or `FAIL`.

## Rows

(empty — first row added by analysis-validator in cycle 1)
