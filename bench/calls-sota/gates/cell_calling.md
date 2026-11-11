# Gate: cell calling (empty droplets)

Fixed and committed on 2026-11-11 before any held-out scoring (decisions D-138, D-102, D-109). Engine frozen at v1.3.7. The call is
computed after the freeze from stored counts (`python/singlet/s1/cellcall.py`, called_v2, version 2.1). No engine code is edited.

## Question

Are Singlet's cell calls at or above EmptyDrops and CellBender on real ground truth, on libraries that no cell-calling script
has seen?

## Truth classes (every table says which class it uses)

| Label | Truth | Counts as ground truth |
|---|---|---|
| G | Barcode confirmed by donor genotype (cell: single-donor genotype posterior from allele counts; empty: ambient-like allele profile) | Yes |
| H | Barcode confirmed by a hashtag (cell: total tag UMIs >= 100; empty: total tag UMIs <= 10; between = unknown, excluded) | Yes |
| S | Species purity in a human + mouse mixture (weak: a pure barcode is a cell only if it is clearly not ambient) | Yes, labelled weak |
| T | Cell Ranger filtered list, authors' cutoff list | No. Reported in a separate block, styled distinctly, never used in the verdict |

Decided barcodes are those whose truth is cell or empty. Barcodes with fewer than 50 total UMIs are excluded (the EmptyDrops test
floor, as in `jobs/cellcall_truth/score.py`). The `lenient` variant (any non-ambient single-source barcode is a cell) is the
verdict variant. The `strict` variant is reported beside it.

## Metrics

1. F1 (with precision and recall) of each tool's own call over decided barcodes.
2. Recall at a matched number of called cells: N is the number of truth cells in the library; each tool contributes its top N
   barcodes, called barcodes first, then by total UMIs (competitors carry no score, so this ordering is a stated approximation,
   as in `bench/cellcall_gap/matched_count.py`). Paired bootstrap 95% interval (barcodes resampled, B = 1000) of Singlet minus each tool.

## Bar

A library passes a metric when Singlet is within 0.005 of the best tool. The gate passes when (a) at least 60% of the libraries pass
on both metrics (most libraries) and (b) no library is worse than the best tool by more than 0.02 on either metric. Dev libraries and
held-out libraries are judged separately. The verdict rule:

* enforce: dev gate passes and the held-out gate passes (every held-out library scored);
* metadata-only hidden: dev passes, held-out incomplete or one held-out library fails by under 0.02;
* withheld: any held-out library worse than the best tool by 0.02 or more, or the dev gate fails.

Runtime ceiling: all calls together 2 core-minutes per 10,000 cells on the host. The call is timed per library.

## Comparators

* Singlet: `singlet.s1.cellcall.call` defaults (exon + intron counts, plateau ambient bound, knee within DropletUtils curve bounds,
  FDR 0.001, 10,000 draws, union with the stored knee call). On dev libraries, the stored `called_v2` column of the truth tables.
* EmptyDrops: on dev libraries the STARsolo EmptyDrops_CR emulation stored beside each truth table; on held-out libraries
  `DropletUtils::emptyDrops` (R, GPL-3, called as an external program, not copied) with `lower = 100`, `niters = 10000`, retain at the
  knee, FDR <= 0.001.
* CellBender 0.3.2: existing outputs only (dev libraries). CellBender needs a GPU for practical runs, and the H100s belong to the
  campaign, so it is not rerun on held-out libraries. A held-out row without CellBender states that and is judged against EmptyDrops only.

## Development libraries (may have been seen)

Genotype truth (G): Kang 2018 lanes b1 A, B, C and the b2 control lane. Species truth (S): hgmm1k_v2, hgmm1k_v3, hgmm5k_v3.
Stored in `$W/state/cellcall_truth/truth/*.truth.tsv.gz` with the caller columns beside the truth. Tool-agreement block (T): the eight
gate libraries of `bench/cellcall_gap/matched_count_v1.3.4.json`, restated, not rescored.

## Held-out libraries (never used for tuning, scored once)

| Role | Library | Truth | Why it qualifies |
|---|---|---|---|
| Hashed cells | 10x Genomics CellPlex PBMC 10k, 3' v3.1 NextGem, raw feature matrix (`SC3_v3_NextGem_DI_CellPlex_Human_PBMC_10K_Multiplex_count_raw_feature_bc_matrix.h5`, CMO counts beside gene counts) | H | corpus row 12, engine touch `none`; absent from every cell-calling script and result |
| Hashed nuclei | McGinnis 2019 nuclei MULTI-seq, GSE129578 POC nuclei, SRR8890648 (corpus row 14 xili_nuc_multi) | H | engine touch `none`; present only in route, cohort and library-type tables, not in a cell-calling script. Needs an all-barcode matrix: see Blocked |
| Genotyped pool | scHi-HOST pooled LCLs, GSE205796 (staged under `data/heldout-launch/schihost_gse205796`), named held-out pool of the design | G | no engine run, no cell-calling script; staged on 2026-11-08. Needs stored counts and genotype truth: see Blocked |

### Verification that the held-out libraries are unseen

Searched on 2026-11-11 by `git grep -i` over `origin/main` of Singlet-Bio/singlet (paths bench, python, scripts) and over all of
`origin/main` of Singlet-Bio/singlet-agents:

* `cellplex`: no hit in Singlet-Bio/singlet bench, python or scripts. singlet-agents hits are the corpus tables, the held-out
  tables of the manuscript and the intel entry P2-COR-73 (a donor question). None is a cell-calling script or result.
* `GSE205796`: singlet-bio hits only in `bench/compression/results/m_w5a/*` (accession search indexes, no cell calls). singlet-agents hits
  are the business coverage CSV, DECISIONS, LAUNCH-HELDOUT, MANIFEST and the touch table (status `staged`).
* `SRR8890648` / `GSE129578`: singlet-bio hits only in `bench/input_routes`, `bench/public/cohort.tsv`, `bench/libtype_screen`,
  `bench/compression/results/m_w5a` (route, cohort, library type, search index). singlet-agents hits are corpus, touch and manuscript tables.
* A search over every remote branch of Singlet-Bio/singlet timed out on the laptop (large tree); the branch-level check is limited to
  the main branch and recorded as a limit.

## Blocked pieces, recorded before scoring

* CellBender on held-out libraries (no GPU).
* Genotyped pool: scHi-HOST has reads only. Stored counts and genotype truth need a registered engine run and the genotype pipeline.
* Hashed nuclei: GSE129578 deposits processed matrices without the empty-droplet barcodes of the nuclei library. An all-barcode matrix needs an engine run on the SRA reads.
