# Cell calling on independent truth (WIP, stopped 2026-10-26)

Question: is called_v2 2.1 more or less accurate than STARsolo EmptyDrops_CR, CellBender, Cell Ranger (published) and the knee,
on truth that does not come from a cell caller?

## Truth
* Genotype (Kang 2018 b2 ctrl; b1 lanes A/B/C in progress): donor genotypes from half the authors' demuxlet singlets; per-barcode
  LLR (single donor or pair, with ambient contamination) vs the ambient donor mixture (empty plateau barcodes).
* Barnyard (hgmm1k_v3, hgmm5k_v3, hgmm1k_v2): species split vs ambient split.
* Calibration on barcodes far from every threshold: high-UMI cells thinned, plateau empties pooled. Single barcodes are only
  decidable at >= 200 SNP UMIs (Kang) or >= 50-80 species UMIs (barnyard), so the contested zone is scored by a **mixture
  estimate** (per-barcode densities under thinned cells / pooled empties, EM per caller-pattern x UMI bin, bootstrap CIs).
* "intact" variant: damaged barcodes (mito above cells' q99 or intronic below their q01, binomial) do not count as cells.

## Results so far (score.py, mixture, barcodes >= 50 UMIs; 95% bootstrap CI of F1 difference vs called_v2)
| dataset | variant | called_v2 P / R / F1 | EDCR F1 | CellBender F1 | Cell Ranger F1 |
|---|---|---|---|---|---|
| Kang b2 ctrl | lenient | 0.989 / 0.993 / **0.991** | 0.954 [-0.040,-0.033] | 0.985 [-0.009,-0.004] | 0.942 [-0.053,-0.045] |
| Kang b2 ctrl | intact | 0.971 / 0.994 / **0.982** | 0.955 [-0.031,-0.023] | 0.977 [-0.008,-0.003] | 0.943 [-0.043,-0.035] |
| hgmm1k_v3 | intact | 0.959 / 0.875 / **0.915** | 0.897 [-0.025,-0.012] | 0.892 [-0.031,-0.017] | 0.899 [-0.023,-0.010] |

hgmm1k_v3 "lenient" counts species-shifted 100-500 UMI debris as cells (recall ~0.57 for every caller); not meaningful.

**Provisional verdict:** on independent truth called_v2 beats EmptyDrops_CR, Cell Ranger and CellBender on F1 (higher recall,
precision 1-3 points lower). Its 10x-truth "losses" reflect Cell Ranger's conservative calls: reframe those metrics as agreement
with Cell Ranger. Not yet confirmed on hgmm5k_v3, hgmm1k_v2, Kang b1.

## Pending (state under $WORK/state/cellcall_truth/)
* jobs 787567 (truth b1 lanes), 787569 (truth hgmm5k_v3/hgmm1k_v2): `truth/<ds>.truth.{tsv.gz,json}`.
* then `MODE=score` (score.json, posteriors) and `MODE=vscore` (variants.json: FDR 0.01/1e-4, test floor 100, exon-only, mito
  filters; calls already in `truth/<ds>.variants.tsv.gz`).
* CellBender failed on b1 lanes (encoder assertion, also at low LR): no CellBender there.
* variants needs `cellcall.py` (origin/main python/singlet/v2) and `kcommon.py` (bench/called_v2_truth/common.py) next to the scripts
  (already in $WORK/jobs/cellcall_truth).
