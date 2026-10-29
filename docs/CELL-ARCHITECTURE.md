# Cell data architecture: what lives where

> Copy of `docs/CELL-ARCHITECTURE.md` in zdebruine/singlet-website (canonical; added with zdebruine/singlet-website#54). Paths such as
> `schema/`, `scripts/ontology/`, `docs/ONTOLOGY.md` and `docs/COHORT-QUERY.md` are in that repository. Change the canonical copy first, then this one.

Status 2026-10-23, agreed with Zach. Every change that stores, indexes or serves per-sample or per-cell data follows this
document; a change that does not fit it updates this document first. The same text is in the singlet-bio repository's
docs (cell index and annotation work) by reference.

Goal: search about a billion cells by sample metadata and per-cell properties in milliseconds, assemble millions of them
from many studies, and keep Cloudflare compute per request small and bounded.

## 1. Three kinds of data, three stores

| Kind | Examples | Store | Access |
|---|---|---|---|
| **Raw data** | count matrices (exon, intron), junctions, heteroplasmy, VDJ | R2: immutable per-sample `.s1` objects (`docs/format-s1.md`, singlet-bio) | HTTP byte ranges through `data.singlet.bio/s1/*`; never copied into a database |
| **Sample metadata** (tier 1) | study/sample catalog, QC, `sample_attr` facts, ontology terms (`sample_term`), derived sample properties, per-sample aggregates of cell properties | D1 `singlet-catalog` + the sample bit index (`docs/COHORT-QUERY.md`) | SQL / bitsets, milliseconds; at most ~10^5 samples x ~10^3 facts |
| **Cell metadata** (tier 2) | per-cell labels (cell type, chromosomal sex, inferred ancestry), QC counts, feature scores (gene-set scores, malignancy, perturbation) | R2: columnar objects (cell-index shards and per-sample feature sidecars), joined by `(sample ordinal, col)` | ranged reads of only the columns and row groups a query needs |

Rules:
- **No per-cell rows in D1.** A billion cells does not fit D1 (10 GB per database, single-threaded queries, a daily read
  budget). D1 holds per-sample summaries of cell properties, never the cells.
- **A cell is `(sample ordinal, col)`**: `col` is the column in the sample's store `cells` table and matrix order; the ordinal
  is assigned once per GSM by the cell index and never reused. Every per-cell file is in `col` order or carries `col`.
- **Provenance is stored where it varies.** Pipeline version, reference, rev and QC are per sample (D1 + manifest); the
  model or method version is per labels/feature file; per-cell values carry only the value and its score. A cell's full
  provenance is one join, not a repeated string.
- **Ontology ids everywhere**: UBERON tissue, CL cell type, MONDO disease (+ PATO:0000461 normal), PATO sex,
  HANCESTRO ancestry (children of HANCESTRO:0004 only), HsapDv/MmusDv stage. Sample and cell filters take the same ids and
  expand descendants through `ontology_closure` (`docs/ONTOLOGY.md`). An index that uses terms outside the D1 slice adds
  them to the slice (rerun `scripts/ontology/build.ts` with those ids).
- **Genotypes are never stored or served.** Inferred ancestry and chromosomal sex are.

## 2. Tier 1: samples (D1)

| Table | Holds | Status |
|---|---|---|
| `gsm`, `gse_meta`, `sample_qc`, `sample_attr` | catalog, QC, extracted facts | live |
| `ontology_*`, `sample_term`, `ontology_term_count` | ontology terms per sample and counts | live 2026-10-23 |
| `sample_property` (+ `property_value_map`) | derived per-sample properties: donor_status (healthy / diseased), perturbation (treated / genetic / control), sample_kind (primary / cell_line / stem_cell / organoid / xenograft); filters `donor_status=`, `perturbation=`, `sample_kind=` | live 2026-10-23: 26,444 rows |
| `cell_feature` | registry of per-cell features: id, kind, method, version, gene set source and licence, storage location; `POST /api/ingest/cell_feature` | table live, empty |
| `sample_cell_summary` | per sample x (label term or feature): cells, cells above two thresholds, quantiles; `POST /api/ingest/sample_cell_summary` | table live, empty |

`sample_cell_summary` is what makes cell questions cheap: "how many malignant T cells in adult human lung" is a sample
filter (tier 1) joined to per-sample counts; it touches no cell data. It also prunes tier 2: a query for cells with a
feature above a threshold reads only samples whose summary says they have some.

## 3. Tier 2: cells (R2)

- **Cell-index shards** (cell-index session; `python/singlet/s1/atlas.py`, Worker `data.singlet.bio/s1/cidx/*`): fixed core
  columns (cell type, tissue, sample ordinal, col, UMI/feature counts, sex, ancestry and their scores), sorted by
  (cell type, tissue, sample, col), ~2M rows per shard with row-group zone maps.
- **Feature sidecars** (the interchange a feature job writes, one per sample and feature set; mirrors the labels file):
  `<GSM>.features.<feature set>.parquet` with columns `gsm` (string), `col` (int64, unique, the .s1 cells-table column;
  legacy studies: the column position in `exon_counts.1pz`, as `singlet.celllabels.io.read_s1/read_legacy` give it), then one
  column per feature: float32 in [0, 1] for `score` (NaN = not computed), uint16 for `category`, bool for `flag`. Parquet
  metadata `singlet_features_schema = "1"`, `feature_set`, `method_version`. Rows for called cells only. The writer runs a
  guard like `singlet.celllabels.output.assert_public_safe`: exact columns as registered, value ranges, unique `col`, no
  gene or genotype strings. Each feature has a `cell_feature` row (method, gene-set source and release, licence,
  `comparable`).
- **Cross-study comparability**: only `comparable = 'atlas'` features are cross-study filters. That needs a cell-intrinsic
  statistic (rank-based, UCell-like: the score depends on the cell's own gene ranks) or a fixed reference null. Scores
  calibrated within a sample (z against the sample's own random sets) are `comparable = 'sample'`: kept for within-study
  use, never offered as an atlas filter.
- **Stored sidecar**: the validated interchange Parquet itself (no separate binary), content-addressed at
  `s1/cf/<GSM>/<set>.<digest>.parquet` and referenced from the study manifest as `samples[i].features[<set>]`, never inside
  the `.s1` (`atlas.normalize_feature_sidecar`, `atlas.attach_features`; singlet-bio `docs/format-s1-cell-index.md` sections
  4.9-4.10). The Worker never reads sidecars.
- **In the index**: the cell-index builder reads feature sidecars keyed `(gsm, col)` and writes registered features as
  extra shard columns (u8 `round(x*254)`, 255 = NA; u16 codes), so a filter on a feature needs no join. Cell queries name
  features `<set>/<name>` (`ranges`, `categories`). `--d1-export` writes `sample_cell_summary.csv` and `cell_feature.csv` in
  the D1 layouts above (categories as `item = <id>=<category>`; sex only for PATO:0000383/0000384). The index sample table
  also carries the sample properties and disease / developmental stage from `scripts/ontology/export-sample-meta.mjs`.

## 4. Query path and cost budget

1. Natural language -> `interpret` (vocabulary first, model only for the rest) -> a structured query with sample clauses
   and cell clauses, both in ontology ids and registered feature ids.
2. Tier 1 (D1 / bit index): candidate samples and exact counts from `sample_cell_summary`. Most questions end here.
3. Tier 2 (Worker, `POST data.singlet.bio/s1/cidx/query`): `where` = organism, cell_type, tissue, sex, ancestry, samples,
   studies (tier 1's ranked list with `study_rank: "given"`), ranges, categories. Caps set by Zach: at most 10,000 cells from
   at most 10 studies per request, a seeded uniform sample with a cursor for the next page, worst case $0.01 per request,
   daily allowances per subject. Over budget: 413 with a **plan** (index shard keys and a runnable Python snippet);
   `dry_run: true` returns the plan and its cost.
4. Gather: (sample, col) lists -> byte ranges in `.s1` objects, through the Worker within the same caps or by the client
   (Python/R) from the plan. Moving millions of cells is bandwidth-bound (seconds to minutes), not a Worker job.

**Custom scripts** run on the client or on Clipper/Anvil against the plan and the same objects, never as arbitrary code in
a Worker. A script whose output proves useful becomes a registered feature: computed in batch, versioned, written as
sidecars, summarised into `sample_cell_summary`.

## 5. Derived properties to pre-populate

| Property | Level | Source |
|---|---|---|
| healthy vs diseased donor | sample | `sample_term` disease (PATO normal vs MONDO); live, 1,205 healthy / 3,193 diseased samples |
| perturbed vs unperturbed | sample | `treatment_status`, engineered genotypes (KO, flox, Cre, transgenic, CRISPR/sh/si RNA); patient genotypes do not count; live, 8,989 treated / 1,761 genetic / 4,054 control |
| primary / organoid / cell line / stem cell / xenograft | sample | the GEO "cell line" field (known line names), tissue and cell type words; live, 3,353 samples |
| cell type, chromosomal sex, inferred ancestry | cell | annotation pipeline labels |
| gene-set activity (single-cell GSEA, rank-based) | cell | feature sidecar; gene sets with open licences only (MSigDB Hallmark CC BY 4.0, Reactome CC0, GO CC BY 4.0) |
| malignant vs normal | cell | CNV-inference score against the sample's own immune/stromal cells, plus cell type |
| perturbation call (guide or hashtag) | cell | feature sidecar where the assay has guides |

## 6. Who builds what (2026-10-23)

| Piece | Owner |
|---|---|
| Ontology mapping, `sample_term`, term filters | ontology-filters session (this repo, PR #54); live |
| Sample properties, D1 `cell_feature` / `sample_cell_summary` tables, ingest, load of the index's aggregate export | ontology-filters session (schema/035); tables live |
| Natural language -> cell query (`interpret` -> `POST /s1/cidx/query` `where`) | ontology-filters session; not started |
| `--sample-meta` export of sample terms and properties for the index builder | ontology-filters session (`scripts/ontology/export-sample-meta.mjs`) |
| Cell-index shards, feature columns and index registry, per-sample aggregate export, Worker query/gather, per-request budget and plan answers | cell-index session (Singlet-Bio/singlet#15, zdebruine/singlet-website#55) |
| Per-cell labels (cell type, sex, ancestry) | cell-annotation session (singlet-bio `singlet.celllabels`) |
| Feature sidecar computation (gene-set scores, malignancy, perturbation calls) | no owner yet; prior art: singlet-agents `feat/cell-phenotype` `phenotype/states.py` (70 programs, within-sample z, 5-9 s per 10k cells) |
