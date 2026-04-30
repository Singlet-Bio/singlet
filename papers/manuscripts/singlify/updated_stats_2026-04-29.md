# Singlify Manuscript — Updated Benchmark Stats (2026-04-29)

## Gene Counting (Panel A — SRR32855204, 10x Chromium 3' v3, human PBMC)

| Metric | Manuscript Value | Updated Value | Source |
|--------|-----------------|---------------|--------|
| Gene Pearson r | 0.9960 | **0.9990** | gene_counting.ipynb, 38,606 genes × 2,520 cells |
| Cell UMI Pearson r | 0.9999 | **0.9993** | gene_counting.ipynb, 2,520 shared cells |
| Splice Junction Jaccard | — | **0.9636** | gene_counting.ipynb |
| Gold cell recall | — | **100%** | All 2,520 STARsolo cells found in singlify output |
| Cells called (singlify) | — | 10,341 (EmptyDrops) | E2E Panel A |
| Cells called (STARsolo) | — | 2,520 | E2E Panel A |
| Input reads | 40,358,185 | same | |
| Mapping rate | 82.91% | same | |
| Singlify commit | — | b0fe019 | |

## Sex Calling (Panel F — SRR32855204)

| Metric | Value |
|--------|-------|
| Agreement | 100% (Female) |
| XIST CPM | 570.3 |
| SRY CPM | 0.0 |

## Pipeline Corpus (1,722 samples — updated P23)

| Metric | Value |
|--------|-------|
| Total samples processed | 1,722 |
| Success rate | 38.1% (656 SUCCESS) |
| Protocols auto-detected | 29 |
| Species covered | 8 |
| Total cells | 2,227,350 |
| Avg mapping rate | 79.8% |
| Avg median genes/cell | 447 |
| GEO series | 864 |
| Organism coverage | 100% |
| Title coverage | 100% |

## .1fq Format (12 validation files)

| Metric | Value |
|--------|-------|
| Total reads | 477,650,376 |
| Average bytes/read | 18.6 |
| Codec | ZSTD level 3 |
| Sequence encoding | 2-bit packed |
| Quality mode | 4-bin (2 bits/base) |

## Notes
- Manuscript currently references GSM8313394 (144 cells, older validation)
- Updated Panel A uses SRR32855204 (2,520+ cells, newer build)
- Correlation improved from 0.996 → 0.999 due to bug fixes in multi-mapper merge
- Cell calling discrepancy (10,341 vs 2,520) is under review (CELL-CALLING-REVIEW in dag.md)
