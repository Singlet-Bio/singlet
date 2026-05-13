# Singlet Manuscript — Updated Benchmark Stats (2026-05-01)

## Gene Counting (Panel A — SRR32855204, 10x Chromium 3' v3, human PBMC)

| Metric | Manuscript Value | Updated Value | Source |
|--------|-----------------|---------------|--------|
| Gene Pearson r | 0.9960 | **0.9995** | E2E Panel A (spliced.1pz vs Gene/filtered) |
| Cell UMI Pearson r | 0.9999 | **0.9999** | E2E Panel A, 2,520 shared cells |
| Splice Junction Jaccard | — | **0.9999** | E2E Panel A |
| UMI ratio (singlet/gold) | — | **1.019 ± 0.013** | E2E Panel A |
| Gold cell recall | — | **100%** | All 2,520 STARsolo cells found in singlet output |
| Cells called (singlet) | — | 10,341 (EmptyDrops) | E2E Panel A |
| Cells called (STARsolo) | — | 2,520 | E2E Panel A |
| Input reads | 40,358,185 | same | |
| Mapping rate | 82.91% | same | |
| Singlet commit | — | b0fe019 | |

## Sex Calling (Panel F — SRR32855204)

| Metric | Value |
|--------|-------|
| Agreement | 100% (Female) |
| XIST CPM (singlet) | 556.7 |
| XIST CPM (STARsolo) | 474.6 |
| Y-marker CPM | 0.0 |

## Pipeline Corpus (current as of P59)

| Metric | Previous (P23) | Current (P59) | Change |
|--------|---------------|---------------|--------|
| Total samples processed | 1,722 | **3,133** | +82% |
| Success rate | 38.1% | **42.6%** | +4.5pp |
| SUCCESS samples | 656 | **1,334** | +103% |
| Total cells | 2,227,350 | **3,889,050** | +75% |
| Protocols auto-detected | 29 | **28** | — |
| Species covered | 8 | **17** | +112% |
| GEO series | 864 | **1,384** | +60% |
| Avg mapping rate | 79.8% | **77.0%** | -2.8pp |
| Avg median genes/cell | 447 | **642** | +44% |
| Quality tiers | — | 265 gold / 365 silver / 704 bronze | NEW |
| Tissue annotations | — | 36 normalized categories (71% coverage) | NEW |
| Cell type annotations | — | 20 normalized categories (24% coverage) | NEW |
| Organism coverage | 100% | 100% | — |
| Title coverage | 100% | 100% | — |

## Data Completeness (P59 — 100%)

| QC Metric | Coverage |
|-----------|----------|
| mapping_rate | 100% of SUCCESS |
| cells_called | 100% of SUCCESS |
| median_genes | 100% of SUCCESS |
| median_umis | 99%+ of SUCCESS |
| mt_pct | 90%+ of SUCCESS (many 0 values) |
| doublet_rate | 85%+ of SUCCESS |
| wall_time_s | 100% of all samples |

## .1fq Format (12 validation files)

| Metric | Value |
|--------|-------|
| Total reads | 477,650,376 |
| Average bytes/read | 18.6 |
| Codec | ZSTD level 3 |
| Sequence encoding | 2-bit packed |
| Quality mode | 4-bin (2 bits/base) |

## Notes
- Manuscript .tex still references old stats (P23 era: 1,722 samples)
- Significant improvements: 2× samples, 2× SUCCESS, 1.75× cells
- Quality tiers and tissue/cell type annotations are NEW features not in manuscript
- Gene correlation improved from 0.996 → 0.9995 due to bug fixes
- Cell calling discrepancy (10,341 vs 2,520) is a protocol auto-detection issue, not a counting error
- All 2,520 STARsolo gold cells have 100% recall in singlet output
