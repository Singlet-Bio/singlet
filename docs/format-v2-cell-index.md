# Singlet v2 atlas cell index (`v2/cidx/`) and cross-study gather

Status: DRAFT 2 (2026-10-23): adds per-cell feature columns (4.9), per-sample aggregates and their D1 export (4.5, 4.10), seeded
sampling with study ranking and the per-request caps of 10,000 cells and 10 studies (6.3), and the per-request cost budget with plan
answers (8.2). Extends `docs/format-v2.md` (section 8 sketched `v2/cidx/` as "a later addition"). Reference
implementation: `python/singlet/v2/atlas.py` (format, builder, query engine, client), `scripts/v2_cidx_build.py` (builder CLI); the
Worker side is `workers/data-v2/src/{scx,cidx,gather,limits,routes}.ts` in the website repository. Costs: `docs/format-v2-cell-index-costs.md`.
Measurements: `bench/cidx/` (scripts and result files named below). All integers little-endian.

## 1. What it is for

The most common operation on the atlas is expected to be a cell search that spans many studies ("T cells from lung, at least 1,000
UMIs, from donors of any study") followed by fetching some or all of the matching cells. Per-sample `cells` tables (format-v2 5.3) answer
this only after a client has opened every candidate sample; D1 cannot hold 297 M cell rows. The index is a set of immutable,
content-addressed columnar objects in R2 that

* answer counts (also grouped) from a small summary without touching cell rows;
* answer cell lists by reading only the row groups whose zone maps can match, and only the columns the filter needs;
* carry, per sample, a compact copy of the chunk index of every matrix so a gather is planned without reading any sample object;
* are rebuilt per index version when labels change; sample objects are never rewritten.

Two Worker endpoints (`POST /v2/cidx/query`, `POST /v2/cidx/gather`) evaluate queries and stream gathered chunks from inside Cloudflare
under hard per-request caps and per-subject daily quotas. Clients can also read every object directly (`via="direct"`), with identical
results.

Scale assumed: 70,986 samples, ~297 M called cells (uncalled barcode columns are not indexed), 8,202 studies.

## 2. Identifiers

* **Sample ordinal `so`** (u32): assigned once per GSM by the builder, in (GSE number, GSM number) order for the first build and
  appended (`max + 1`) afterwards; never reused or reassigned (`--previous <version>` carries them over). Ordinals of one study are
  therefore mostly contiguous, which makes sample/study zone maps useful.
* **Column `col`** (u32): the cell's position in the sample's `cells` table, which is also its matrix column in every column-aligned
  matrix member (exon, intron, sj, vdj, mt).
* **Cell id** = `(so, col)`; packed as `so * 2^32 + col` where one integer is needed. Valid within an index version. The durable,
  version-independent name of a cell is `(GSM, col)` (sample objects are immutable and their column order is part of the bit-exact
  migration contract) or `(GSM, barcode)`.

## 3. Objects in R2

| key | content | cache |
|---|---|---|
| `v2/cidx/current.json` | `{"format": "singlet-cidx", "version": "<ver>", "directory": "v2/cidx/v/<ver>.json"}` (the only mutable object) | `max-age=60` |
| `v2/cidx/v/<ver>.json` | directory of one index version (section 4.4); `<ver>` = `YYYYMMDDTHHMMZ-<8 hex>`, the hex being a digest of the directory content | immutable |
| `v2/cidx/obj/<d16>.scx` | SCX1 objects: cell shards, the sample table, the summary (section 4) | immutable |
| `v2/cidx/obj/<d16>.json` | plan blocks (section 4.7) | immutable |
| `v2/cidx/obj/<d16>.parquet` | optional Parquet export of each organism partition (section 4.8) | immutable |
| `v2/a/<GSM>/<d16>.parquet` | canonical per-sample label sidecar (section 9), referenced from the study manifest | immutable |

`<d16>` = hex of the first 16 bytes of SHA-256 of the object. Objects are shared between versions when their content is equal. The data
Worker serves all of these by GET with Range/ETag like any other v2 object (keys are matched by anchored patterns).

## 4. Format

### 4.1 SCX1 (shards, sample table, summary)

A small columnar container chosen by measurement (section 5): u8/u16/u32 columns in row groups, each column chunk coded by the cheapest
of three integer encodings, no general-purpose compression, so the Worker decodes with typed arrays and no decompressor.

```
offset 0   FileHeader (64 B)   "SCX1" | u16 version=1 | u16 n_cols | u32 n_groups | u64 n_rows | u32 group_rows | u32 header_len | 36 B zero
           n_cols   x 32 B      name[24] (ASCII, NUL padded) | u8 dtype (1 u8, 2 u16, 3 u32) | 7 B zero
           n_groups x 16 B      u64 first_row | u32 n_rows | 4 B zero
           n_groups x n_cols x 24 B   u64 off (from file start; 0 when len = 0) | u32 len | u8 enc | u8 bits | u16 0 | u32 min | u32 max
           zero padding to header_len (a multiple of 64)
           chunk bodies, each starting at a multiple of 8
```

Encodings: `0 CONST` (every value = `min`, no body); `1 FOR` (`value - min` in `bits` bits, LSB-first in little-endian u32 words,
`ceil(n*bits/32)` words); `2 RLE` (pairs `u32 value, u32 run`, runs add up to the group). The writer picks the shortest. `min`/`max`
are the zone map of every column of every group: a reader prunes groups from the header alone (one read of `header_len` bytes, ~10 KB
for a 2 M-row shard) and then reads the needed chunk bodies of a group in one range (they are adjacent).

A reader rejects a bad magic or version, a header shorter than its tables, a FOR body shorter than `n*bits` bits, RLE runs that do not
add up, and unknown encodings.

### 4.2 Shard schema (version 1)

One row per called cell. Physical column order:

| column | type | meaning |
|---|---|---|
| `ct` | u16 | cell type, dictionary code (`dicts.cell_type`); 0 = no call |
| `tissue` | u16 | tissue (sample-level, UBERON), dictionary code; 0 = unknown |
| `sample` | u32 | sample ordinal `so` |
| `col` | u32 | column in the sample |
| `umi_exon`, `umi_intron`, `feat_exon`, `feat_intron` | u32 | exact counts from the `cells` table (`n_umi_*`, `n_feat_*`) |
| `ct_score` | u8 | cell-type confidence, `round(x*254)`, 255 = not available |
| `sex` | u8 | chromosomal sex call: 0 unknown, 1 female (PATO:0000383), 2 male (PATO:0000384), 3 ambiguous |
| `sex_score` | u8 | posterior of that call, as `ct_score` |
| `anc` | u8 | inferred genetic ancestry category (HANCESTRO), dictionary code; 0 = no call |
| `anc_prob` | u8 | proportion of that ancestry in the donor's estimate, as `ct_score` |
| `f1`, `f2`, ... | u8 / u16 | registered per-cell features (4.9), in registry order |

QC counts are exact (u32), not quantised: a 1/256-octave log code would save 1.9 B/cell (13 %, measured) but makes every threshold
approximate; the simpler semantics won. Scores are u8 because the labels themselves are model estimates with far less than 1/254
resolution. Score codes are `floor(x*254 + 0.5)`; a range bound compares against the code its own value is stored as, so no cell whose
true score satisfies the range is dropped (one up to half a step outside may be kept). Query fields map onto columns (section 6.1);
`n_umi` = `umi_exon + umi_intron`, `n_genes` = `feat_exon`.

Other per-cell values (gene-set scores, malignancy, perturbation calls, doublet or mitochondrial fractions) are registered features
(4.9): extra columns with a registry entry, not schema changes. Readers look columns up by name.

### 4.3 Partitioning, order, sizes

* **Partition** = organism. Within a partition rows are sorted by `(ct, tissue, sample, col)`.
* **Dictionary codes follow the ontology.** Cell types, tissues and ancestry terms get codes in the depth-first order of the
  ontology's primary-parent tree, so a term and its descendants occupy (nearly) contiguous code ranges, and therefore contiguous rows.
  A query for "T cell" reads the shards and groups whose `ct` range overlaps the descendants of CL:0000084 and nothing else; within them,
  rows are grouped by tissue, so "T cells in lung" reads a narrow band.
* **Shard** = 2^21 rows (~22-30 MB): under the data Worker's 32 MiB threshold, so a shard read twice in a data centre is served from the
  Cache API thereafter. **Row group** = 65,536 rows (zone maps, unit of decoding). ~140 shards and ~4,500 groups for the atlas.
* **Postings**: for each sample, the plan block (4.7) lists its runs `[shard index, first row, n rows]` (one per cell type present).
  A query restricted to at most 2,000 samples (`samples` / `studies` filter) reads only the groups those runs touch; larger sample
  sets fall back to zone maps (`sample` min/max is still informative because ordinals are assigned by study).

### 4.4 Directory `v2/cidx/v/<ver>.json`

```json
{"format": "singlet-cidx", "format_version": 1, "version": "20261101T0000Z-8c65e3ff", "created_at": "...",
 "builder": {...}, "group_rows": 65536, "shard_rows": 2097152,
 "columns": [{"name": "ct", "dtype": "u16", "kind": "code:cell_type"}, ...], "sort": ["ct", "tissue", "sample", "col"],
 "score_scale": 254, "na8": 255,
 "dicts": {"organism": [...], "cell_type": [{"code": 3, "id": "CL:0000084", "label": "T cell", "anc": ["CL:0000000", "CL:0000542", ...]}],
           "tissue": [...], "sex": [...], "ancestry": [...]},
 "terms": {"CL:0000542": "lymphocyte", ...},
 "samples": {"key": "v2/cidx/obj/<d16>.scx", "rows": 70986, "max_so": 70986},
 "summary": {"key": "v2/cidx/obj/<d16>.scx", "rows": ...},
 "plan": {"block_size": 256, "blocks": {"0": "v2/cidx/obj/<d16>.json", ...}},
 "shards": ["v2/cidx/obj/<d16>.scx", ...],
 "partitions": [{"organism": 1, "label": "Homo sapiens", "rows": ..., "shards": [
     {"i": 0, "key": "...", "bytes": ..., "rows": ..., "zone": {"ct": [1, 17], "tissue": [...], "sample": [...], "umi_exon": [...], ...},
      "tissues": [codes] or null}]}],
 "exclude": [],
 "totals": {"cells": ..., "samples": ..., "studies": ..., "labeled_samples": ...}}
```

`anc` lists every ancestor of a dictionary term (all paths, `is_a` and, for anatomy, `part_of`); `terms` names every ancestor, so a
query may name an ancestor that no cell carries directly ("lymphocyte"). Code 0 of every dictionary is "unknown". `exclude` lists
sample ordinals hidden from this version (withdrawn or replaced samples between full rebuilds, section 10). Size for the atlas: about
0.3-0.5 MB, parsed once per Worker isolate.

### 4.5 Sample table and per-sample aggregates

SCX1 with one row per indexed sample, ordered by `so`: `so`, `gsm` and `gse` (the accession numbers), `organism`, `tissue` (codes),
`n_columns`, `n_called`, `labeled`, and per-sample aggregates of the called cells:

* `umi_min, umi_q10, umi_q50, umi_q90, umi_max` (n_umi = exon + intron) and `genes_*` (n_genes), u32;
* for every feature `fK`: `fK_n` (cells with a value), `fK_min, fK_q10, fK_q50, fK_q90, fK_max` (codes), and `fK_geJ` (cells at or above
  the feature's J-th registered threshold).

Quantiles are order statistics (`sorted[floor(q (n - 1))]`), so they are stored values and reproducible. It is the ordinal registry
(the next build reads it to keep ordinals), resolves `samples` / `studies` filters, and prunes samples before any shard is read: a range
on `n_umi`, `n_genes` or a feature skips every sample whose [min, max] cannot overlap it (and every sample without values of that
feature). The directory's `sample_stats` maps range fields to these column prefixes.

### 4.6 Summary

SCX1 with one row per (organism, cell type, sex, ancestry, sample): those codes, the sample's tissue, and `n` (cells). Sorted by
organism, cell type, sex, ancestry, sample, so cell-type filters prune its groups too. Every count that needs no per-cell range
(`organism`, `cell_type`, `tissue`, `sex`, `ancestry`, `samples`, `studies`, grouped by any of them or by sample/study) is answered from
it without reading shards. Expected size for the atlas: a few million rows, ~10-20 MB.

### 4.7 Plan blocks (the compact chunk index)

One JSON object per 256 consecutive ordinals:

```json
{"format": "singlet-cidx-plan", "version": 1, "block": 12, "block_size": 256, "samples": [
  {"so": 3072, "gsm": "GSM...", "gse": "GSE...", "key": "v2/s/GSM.../<d16>.s2", "bytes": 3186387, "n_columns": 39217, "n_called": 2722,
   "organism": "Homo sapiens", "annot": "v2/a/GSM.../<d16>.parquet",
   "members": {"exon_counts.1pz": {"off": 181760, "len": 544662, "digest": "...", "prefix": 53144, "chunks_at": 234904, "n_chunks": 10,
                                   "first_col": [0, 4096, 8192, 12288, 16384], "coff": [0, 215402, 349372, 433987, 464727], "codec": 1, "m": 310797},
               "intron_counts.1pz": {...}, "sj_counts.1pz": {...}, "vdj_gene_usage.1pz": {...}, "cells": {"off": 2752, "len": 178978}},
   "runs": [[0, 0, 412], [0, 9120, 380], ...]}]}
```

For every column-aligned matrix member: its byte range in the object, the pz2 prefix length (header + chunk index + tables + rows,
format-v2 4.2; what a decoder needs before any chunk), the absolute offset of the `chunks` section, and the chunk index **truncated to
the chunks that contain called cells** (`first_col` / `coff` with the sentinel of the last kept chunk). Called cells sit in the first
columns of every matrix (columns are roughly in UMI order), so this keeps 30-50 % of the chunks of a typical sample. This answers the
design question "can a compact chunk-index blob avoid the prefix fetch": yes for *planning* (the Worker never reads a sample object to
plan; ~1.3 KB per sample, measured, ~90 MB for the atlas in ~280 blocks), but the *decoder* still needs each member's prefix
(40-70 KB per member on the pilot, dominated by the codec-1 row-key stream). Clients therefore cache prefixes on disk by member digest
(immutable) and tell the Worker which ones they have (`have_prefix`).

### 4.9 Per-cell features

A feature is any number per called cell that a pipeline outside the index computes: a gene-set score (single-cell GSEA, e.g. Hallmark),
a malignancy probability, a perturbation call, a future derived score. The builder takes a registry and one file per sample:

```json
{"features": [{"name": "hallmark_hypoxia", "version": "msigdb-2024.1/ucell-1", "dtype": "u8", "lo": -1, "hi": 1,
               "thresholds": [0.0, 0.5], "description": "...", "producer": "singlet.enrich 0.4"}]}
```

and `<features_dir>/<GSM>.features.parquet` (or `<GSE>/<GSM>...`) with `col` and one float column per feature it provides. Values
are stored as linear codes over `[lo, hi]` (clipped): u8 = `floor((x - lo) / (hi - lo) * 254 + 0.5)`, 255 = missing; u16 the same with
65,534 and 65,535. Bounds compare against their own codes as scores do (4.2). Names are `[a-z][a-z0-9_]*` (at most 48 characters),
unique, not a built-in field; at most 64 features and 4 thresholds each. The builder assigns shard columns `f1, f2, ...` in registry
order and records the registry (with `column`, `scale`, `na`) in the directory under `features`; a query names a feature exactly like a
built-in range field (`"ranges": {"hallmark_hypoxia": [0.3, null]}`) and cell answers return it decoded. Versioning: a feature's
`version` is part of the registry; changing a producer or its parameters is a new version of the feature, and the index is rebuilt
(features, like labels, are inputs of a version). A sample with no feature file gets NA for every feature. A u8 feature costs about
1 byte per cell, a u16 about 2.

### 4.10 Export to the catalog (D1)

`BuildResult.d1_export(dir)` / `scripts/v2_cidx_build.py --d1-export dir` writes the per-sample aggregates as CSV for the catalog,
which owns the tables and loads them (tier 1: sample filters and pruning in D1 / the bit index; tier 2: this index):

* `cidx_sample.csv`: gsm_id, gse_id, so, index_version, organism, tissue, n_called, labeled, umi_min..umi_max, genes_min..genes_max;
* `cidx_sample_celltype.csv`: gsm_id, cell_type (CL id), cell_type_label, n_cells, index_version (cells per CL term per sample, from the
  summary);
* `cidx_sample_feature.csv`: gsm_id, feature, feature_version, n_values, min, q10, q50, q90, max (decoded), threshold, n_ge, index_version
  (one row per registered threshold);
* `cidx_d1_schema.sql`: the suggested tables.

### 4.8 Parquet export

With `--parquet`, the builder also writes each organism partition as one Parquet file (zstd, page index, same columns and order) and
records its key in the partition entry, for DuckDB / duckdb-wasm / Polars users who want SQL over the whole index
(`SELECT ... FROM 'https://data.singlet.bio/v2/cidx/obj/<d16>.parquet'`). Storage only (~$0.06 per month per version); the Worker does
not read it.

## 5. Format choice (measured)

`bench/cidx/synth.py` generates one realistic 4 M-cell partition (1,799 samples; log-normal depth, sub-linear feature counts, Zipf cell
types, per-sample tissue/sex/ancestry, called cells in early columns); `bench/cidx/formats.py` writes it in each candidate format;
`bench/cidx/scan.mjs` scans it in V8 (Node 24, the engine of Workers) with the query "umi_exon >= 1000 and ct == 101, return (sample, col)"
(1.33 M matches). i7-1185G7, 2026-10-23.

| format | bytes/cell | + zstd-3 of the file | JS scan, 4 M rows, 4 columns |
|---|---|---|---|
| Parquet, zstd (hyparquet 1.31 + hyparquet-compressors) | 13.19 | 13.02 | 10.7-11.4 s |
| Parquet, snappy | 14.75 | 13.01 | 6.2-7.9 s |
| Parquet, uncompressed, dictionary + DELTA_BINARY_PACKED | 12.17 | 11.29 | 9.2-11.0 s |
| **SCX1, exact counts** (chosen) | 14.84 | 12.22 | **0.65 s** |
| SCX1, QC counts as 1/256-octave log codes | 12.98 | 11.03 | 0.83 s |

Sorted by `(ct, tissue, sample, col)` with 65,536-row groups (the chosen layout), SCX1 is 14.4 B/cell on the synthetic partition
(`col` 2.1, the four exact counts 7.9, scores 2.8, `sample` 0.2, `ct`/`tissue` ~0); on the real pilot it is **10.4 B/cell**
(423,336 bytes for 40,614 cells). Atlas estimate 3.1-4.3 GB per version.

Decision: SCX1 for what the Worker reads. It is within 10-20 % of Parquet's size and 10-17x faster to scan in a Worker, which is the
difference between ~1 s and ~10 s of CPU for a broad query; general compression would add a decompressor to the Worker and buys only
~1 B/cell. Parquet stays available as an optional export (4.8) because DuckDB/duckdb-wasm read it with zero code.

## 6. `POST /v2/cidx/query`

### 6.1 Request

```json
{"version": "current",
 "where": {"organism": "Homo sapiens", "cell_type": ["CL:0000084"], "tissue": "lung", "sex": ["female"], "ancestry": "HANCESTRO:0005",
           "samples": ["GSM..."], "studies": ["GSE..."],
           "ranges": {"n_umi": [1000, null], "n_genes": [200, 6000], "cell_type_score": [0.5, null], "hallmark_hypoxia": [0.3, null]}},
 "return": "count" | "cells", "by": ["cell_type", "tissue"],
 "n": 10000, "seed": 0, "max_studies": 10, "study_rank": "cells" | "given" | "random", "sampling": "uniform" | "chunks",
 "cursor": "...", "fields": ["ct", "umi_exon", "f1", ...], "dry_run": false}
```

Semantics (identical in `singlet.v2.atlas.Index` and the Worker; the Worker tests check the Worker's answers against golden answers
computed by the Python engine): values in a list are ORed, fields are ANDed. `cell_type`, `tissue`, `ancestry` take ontology ids or
labels (case-insensitive, also of ancestors) and include descendants. `sex` takes female / male / ambiguous / unknown. Ranges are
inclusive, `null` = open; range fields: `n_umi`, `n_umi_exon`, `n_umi_intron`, `n_genes`, `n_feat_exon`, `n_feat_intron`,
`cell_type_score`, `sex_score`, `ancestry_prob`, and every registered feature. Score and feature bounds are applied at the stored
resolution (4.2) and a missing value never satisfies a range. Unknown filters, fields or terms are errors (400), not silently ignored.

### 6.2 Answers

* `return: "count"` without ranges: from the summary (no shard is read): `{"count", "groups": [{"cell_type": "CL:...",
  "cell_type_label": "...", "n_cells": ...}], "source": "summary"}`.
* `return: "count"` with ranges: a row scan; `"source": "scan"`.
* `return: "cells"`: a seeded sample (6.3): `{"n", "cells": {"so", "gsm", "col", "cell_type", ..., "n_umi_exon", ..., <features>},
  "eligible", "eligible_max", "remaining", "sampled", "studies": [{"gse", "eligible_max", "selected"}], "n_studies", "cursor"}`, cells sorted
  by (so, col).
* Every answer carries `version`, `complete`, `cursor`, `rows_scanned`, `reads`, `bytes_read`, `cache_hits`, `ms`, `cost` (the request's
  cost from the reads that missed the cache) and, for scans, `estimate` (its worst case, 8.2).
* `dry_run: true` answers `{plan}` with the worst-case estimate and the shards it would read, and runs nothing.

A cursor is opaque and bound to `(version, query, sampling parameters)`; it is rejected for anything else. For counts it encodes a
position (shard, group, row): a row scan stops after the tier's `maxRowsScanned` rows and returns `complete: false` with a cursor,
and counts are summed over the continuation (`singlet.v2.atlas.count_cells` does this). For cells it encodes the last sampling key of
the page (6.3).

### 6.3 Cell answers: at most 10 studies, at most 10,000 cells, seeded uniform sample

One request returns at most `n <= 10,000` cells drawn from at most `max_studies <= 10` studies:

1. **Studies.** From the summary (no shard read): every study with cells matching all non-range filters, in samples the per-sample
   aggregates do not rule out; `eligible_max` is that count (an upper bound when ranges are given). Ranked by `study_rank`: `cells`
   (default: most eligible cells, then GSE number), `given` (the order of the `studies` filter: the caller's relevance ranking, e.g. from
   the website's search), or `random` (seeded: `hash32(seed, gse, 0x9E3779B9)`). The first `max_studies` are selected; `studies` lists the
   top 50 with their counts and whether they were selected.
2. **Eligible cells.** Every matching cell of the selected studies' samples (postings path; rows decoded = `rows_scanned`).
3. **Sample.** Each eligible cell gets the key `(hash32(seed, so, col), so, col)`, with `hash32(s, a, b) = mix32(mix32(s ^ a) ^ b)` and
   `mix32` the lowbias32 integer hash (u32 arithmetic; identical in Python and the Worker, pinned by golden values). The answer is the `n`
   smallest keys after the cursor: a uniform random sample without replacement of the eligible set, the same for the same seed. The next
   page (`cursor`) continues in key order, so pages of one seed are disjoint and together enumerate the eligible set in a random order:
   "deeper sampling" is repeating the request with the cursor, each repeat one request under the rate limits and daily quota. The Worker
   keeps a bounded heap of `n` keys, so memory does not depend on the eligible count.
4. **`sampling: "chunks"`.** The key is that of the cell's matrix chunk, `hash32(seed ^ 0x5BD1E995, so, chunk)` (exon chunk index from
   the plan block), so whole chunks are taken until `n` cells: cells within a chunk are correlated (neighbouring columns, roughly similar
   depth), but a gather of them reads one chunk per ~chunk-width cells instead of nearly every chunk. Measured on the pilot: 1,000 cells
   uniform 9.2 MB of chunk reads, chunks 0.57 MB; 10,000 cells 9.3 MB vs 3.6 MB.

Why these thresholds: the dollar bound (8.2) holds with a wide margin at 10,000 cells / 10 studies (worst case ~$0.003); what grows with
them is the work per request (rows decoded, bytes gathered, client wait). A uniform sample of 10,000 cells spread over 10 studies touches
nearly every chunk of the selected samples, i.e. moves about the whole called region of those samples (up to ~1 GB with today's chunking),
which is the practical limit; cutting chunks at 48k non-zeros (10.2) or `sampling: "chunks"` lowers it. A larger `n` per request would not
raise the cost much but would raise per-request bytes and latency; fewer studies per request makes ranking matter more. Both caps are
configuration (`CIDX_LIMITS`), not code.

### 6.4 Limits

| | anonymous | API key |
|---|---|---|
| cells per answer | 10,000 | 10,000 |
| studies per answer | 10 | 10 |
| rows decoded per request (row scans) | 8 M | 32 M |
| row-scan rows per UTC day | 200 M | 5 G |
| requests per minute (Rate Limiting binding, per location) | 30 | 300 |

All are configurable without code (`CIDX_LIMITS` JSON var; rates in `wrangler.toml`). Section 8 has the quota mechanics.

## 7. `POST /v2/cidx/gather`

### 7.1 Request

```json
{"version": "20261101T0000Z-8c65e3ff",
 "cells": [[3072, [5, 17, 220]], ["GSM4317426", [0, 1, 2]]],      // or "query": {"where": {...}, "n": 10000, "seed": 0, ...}
 "modalities": ["counts"],                                         // the singlet.v2.client vocabulary
 "have_prefix": ["<member digest>", ...], "cells_table": false, "dry_run": false}
```

`modalities` uses the vocabulary of `singlet.v2.client` (`counts`, `spliced`, `unspliced`, `splice_junctions`, `mt`, `vdj`; aliases
`exon`, `intron`, `sj`, ...); `psi` is derived, so a client gathers `splice_junctions` and computes it. Only called (indexed) columns can
be gathered. `cells_table: true` adds each sample's whole `cells` member (barcodes and per-cell tables) to the stream. A `query` is a
handle: the cells of the same request to `/v2/cidx/query` with `return: "cells"` (the seeded sample of 6.3), gathered in one step.

### 7.2 Plan

From the plan blocks only (no sample object is read to plan): for every sample and member, the chunk runs covering the wanted columns
(touching chunks merged), plus the prefix unless its digest is in `have_prefix`. Pieces of one sample object closer than 64 KiB become
one read (at most 8 MiB). `singlet.v2.atlas.plan_gather` is the reference; the Worker's `planGather` gives the same reads (tested).

Before anything is streamed the Worker checks the per-request caps (cells, studies, samples, planned bytes, planned reads) and the
worst-case cost (8.2), then reserves the planned cells and bytes against the subject's daily quota. A plan over a cap or the budget is
answered 413 with a plan the client can run in smaller requests (`singlet.v2.atlas.gather` batches by 10,000 cells and 10 studies);
`dry_run: true` returns the plan and its estimate as JSON.

### 7.3 Stream `application/x-singlet-gather`

```
"SGA1" | u32 header_len | header JSON {format, version, member_names, samples: [{i, so, gsm, gse, key, n_cells, members:
        {name: {digest, prefix, send_prefix, runs: [[c0, c1, member_off, len], ...], len}}}], reads, bytes, payload, cells}
frames: u32 sample i | u16 member index | u8 kind (1 prefix, 2 run, 3 whole member, 254 error, 255 end) | u8 0 | u64 member offset | u32 len | body
```

Frames arrive in completion order. An error frame (JSON `{sample, key, error}`) reports a failed read and the stream continues; the end
frame's body is the trailer `{ok, errors, reads, bytes_read, bytes_sent, cache_hits, cost, ms}`; the header carries the worst-case
`estimate`. A client adds every run to its decoder at
the given member offset (`singlet._pz2.Member.add_range`) and decodes exactly as for a local file; header digests are checked against
the plan.

### 7.4 Reads inside Cloudflare

Every read goes through the same handler as public GETs (Cache API first, then a ranged R2 read). A Worker invocation can wait on at
most six connections at once (R2 gets and cache lookups count), so one invocation sustains about 6 / (R2 latency) reads per second. A
plan of 48 or more reads is split into up to 16 parts that run in sub-invocations of the same Worker through a service binding to itself
(`SELF`, guarded by the `CIDX_INTERNAL_SECRET` header; sub-invocations are not billed as requests), each with its own six; the parent
copies their frames into its stream whole. Sample objects read by gather are filled into the Cache API on their second ranged miss in an
isolate (not the first, unlike plain ranged GETs: a gather touches many samples once).

### 7.5 Limits

| | anonymous | API key |
|---|---|---|
| cells per request | 10,000 | 10,000 |
| studies per request | 10 | 10 |
| samples per request | 2,000 | 2,000 |
| planned bytes per request | 1 GiB | 1 GiB |
| planned R2 reads per request | 4,000 | 4,000 |
| cells per UTC day | 200,000 | 2 M |
| bytes per UTC day | 4 GiB | 200 GiB |

`singlet.v2.atlas.gather` splits larger requests into batches under the per-request caps (study by study; a sample bigger than a batch is
split by columns) and stitches the result; every batch is a request under the rate limits and daily quota.

## 8. Subjects, rate limits, quotas

* **Subject**: `key:<api key id>` when `Authorization: Bearer <key>` is a valid singlet.bio API key (checked read-only against D1
  `api_keys` by SHA-256, memoised 60 s per isolate; revoked/expired keys get 401), else `anon:<hash of the client address>`.
* **Per-request caps** (6.4, 7.5) and the per-request cost budget (8.2): stateless, always on.
* **Request rate**: Workers Rate Limiting bindings `CIDX_RL_ANON` / `CIDX_RL_KEY` (per location, approximate).
* **Daily allowance**: Durable Object `CidxQuota`, one per subject, single-threaded so check-and-add is atomic: cells, bytes, rows
  scanned and worst-case dollars (anonymous $0.05/day, API key $1.00/day). Gathers reserve their planned cells, bytes and worst-case
  cost up front (no refund on failure). Row-scan queries check that the day's row and dollar budgets have room, then add the rows they
  scanned and their worst case off the latency path, so a subject overshoots by at most one request. Summary counts are not metered.
* Missing bindings switch their layer off (local tests, first deploy); the per-request caps still bound every request.

### 8.2 Per-request cost budget and plan answers

Every request is bounded so that its Cloudflare cost cannot exceed `maxCostUsd` = **$0.01** even if every read misses the cache and
starts a cache fill. Before scanning or reading, the Worker computes the worst case of the plan with the list prices and a CPU model
(`estimateCost`: request + (reads + fills) x Class B + CPU ms (5 fixed, 0.25 per 1,000 rows decoded, 1 per MB copied) + Durable Object
calls); answers report it as `estimate` and, after the work, `cost` from the reads that actually missed. The caps are chosen so that the
largest request they admit stays under the budget (worst gather: 4,064 reads + 1 GiB, ~$0.0030; worst row scan: 32 M rows, ~$0.0003;
tested). If a request's own plan exceeds the budget or a cap (a sample whose selected studies hold more rows than one request may scan, a
gather over its byte or read cap), the answer is 413 with `plan`: the reason, the estimate, the index objects to read (shard keys and sizes)
and a ready-to-run Python snippet (`singlet.v2.atlas.Index(...).sample_cells(...)`, or a batched `gather`) for running it client-side
against the immutable, cacheable index objects, or for narrowing the request. `dry_run` returns the same plan without refusing.

What this bounds per day is in `docs/format-v2-cell-index-costs.md` (worst case per API key per day ~$1.17, per anonymous address ~$0.07).

## 9. Per-cell label sidecars

**Interim interchange** (written by the annotation pipeline, `python/singlet/annotate/output.py` on its branch): one Parquet per
sample, `<GSM>.labels.parquet`, columns `gsm, col, cell_type_cl, cell_type_label, cell_type_score, sex_chrom, sex_score,
ancestry_hancestro, ancestry_prob, model_version`, one row per called cell.

**Canonical sidecar** (`atlas.normalize_sidecar`): validated (one GSM; `col` unique and inside the matrix; CL / HANCESTRO ids
well-formed; sex in female / male / ambiguous; scores in [0, 1]), restricted to called columns, sorted by `col`, strings
dictionary-encoded, zstd, and nothing but the nine columns of `SIDECAR_COLUMNS` (genotype-like content cannot pass because no other
column is written). File metadata: `gsm`, `annot_version`, `cells_digest` (the `cells` member digest the labels were computed on),
`model_version`. Deterministic bytes, so the key is content-addressed: `v2/a/<GSM>/<d16>.parquet`.

**Reference from the manifest** (never inside the immutable `.s2`): `atlas.attach_annotations(manifest, entries, annot_version)` adds

```json
"samples": [{"gsm": "...", ..., "annotations": {"a2026-11": {"key": "v2/a/GSM.../<d16>.parquet", "sha256": "...", "bytes": 1234,
                                                             "n_rows": 3820, "model_version": "...", "cells_digest": "..."}}}],
"annotations_default": "a2026-11"
```

Several annotation versions can coexist; the builder uses `--annot-version` or `annotations_default`, or a local directory of interim
files (`--labels-dir`). Publishing the manifest change goes through the migration's publish step (the manifest `rev` covers sample keys
and study files; MCP "unchanged" answers should also compare `annotations_default`; see 13). Inferred labels are public; genotypes are
never written.

## 10. Versions, rebuilds, retention

1. **Build** a version from the current manifests + sidecars (`scripts/v2_cidx_build.py`). Unchanged inputs give byte-identical objects
   and the same version digest (tested), so a no-op rebuild uploads nothing new.
2. **Upload** objects (`obj/*`, then `v/<ver>.json`), **then** flip `current.json`. Readers that resolved `current` before the flip keep
   using the old version (every object they need is immutable and still there); cursors and query handles name their version.
3. **Retention**: keep the previous two versions' directories and objects for 30 days after they stop being current; delete objects no
   retained version references (they are content-addressed, so a GC lists retained directories and deletes the rest).
4. **New samples between full rebuilds**: a full rebuild is cheap (the atlas index is ~4 GB; the build is dominated by reading 71k
   `cells` tables, about an hour on a cluster node, section 11), so the default is a rebuild per campaign batch. The format also allows
   an incremental version: keep the old partitions' shard entries, append shards built from the new samples only (each shard is sorted on
   its own; nothing assumes order across shards), add their plan-block entries, and list replaced or withdrawn ordinals in `exclude`.
   The current builder does full rebuilds; incremental is a builder change only.
5. **Labels change** (new annotation version): rebuild (the sort order depends on cell type).

### 10.1 What the index does not need from sample objects

Only the `cells` member (counts and the called flag) and each matrix's 256-byte header and chunk index. Sample objects are not modified.

### 10.2 A recommendation for the migration writer (measured)

The pz2 writer picks `chunk_cols` from the average non-zeros per column over *all* columns, most of which are empty droplets, so
sparse samples get 4,096-column chunks whose first chunks hold thousands of called cells (the pilot: every sample uses 4,096). For a
gather that takes a few cells per sample, the bytes moved per cell are the size of the chunk it lives in. `bench/cidx/chunking.py` on the
pilot (14 samples, exon + intron, prefixes excluded):

| chunk cut | bytes to fetch one random called cell | bytes for 1,000 random cells of the study |
|---|---|---|
| current (4,096 columns) | 322,781 | 7.6 MB |
| at most 96,000 non-zeros (and 4,096 columns) | 202,347 | 7.9 MB |
| at most 48,000 non-zeros | 118,715 | 7.9 MB |
| at most 24,000 non-zeros | 61,398 | 7.7 MB |
| at most 12,000 non-zeros | 30,862 | 7.1 MB |

Cutting chunks at ~48k non-zeros (the writer's stated target) instead of a fixed column count makes sparse cross-study picks ~2.7x
cheaper in bytes, egress time and Worker CPU, at no cost for whole-sample reads; readers already accept variable-width chunks (format-v2
4.2: "a chunk may be narrower"). It is a writer-only change, but it must be made **before** the atlas migration campaign, since sample
objects are immutable afterwards.

## 11. Builder

`atlas.build_index(gses, base=..., sample_meta=..., labels_dir=... | annot_version=..., ontology=..., previous=...)` /
`scripts/v2_cidx_build.py`:

* reads each study manifest, then per sample (thread pool): the `cells` member (one range), and for every matrix member the 256-byte
  header and the chunk index (two small ranges);
* labels: `labels_dir/<GSM>.labels.parquet` (or `<GSE>/<GSM>...`), else the manifest's sidecar (sha256 checked);
* sample metadata: `--sample-meta` (gsm, organism, tissue UBERON id, tissue_label; the ontology session's D1 mapping exported as
  Parquet), else the organism from `study_meta.json`;
* ontologies: `--ontology cl-basic.obo --ontology uberon-basic.obo --ontology hancestro.obo` (OBO or JSON), for codes and ancestors;
* writes the server layout under `--out`, prints a report (samples, errors, cells, bytes, requests); a sample that fails is reported and
  left out (exit code 2), never silently.

Resources for the atlas (estimate from the pilot: 127 requests and 2.0 MB read for 14 samples): ~650k range requests and ~10 GB read
(the `cells` tables), about an hour with 32 threads from a cluster node to data.singlet.bio, or minutes from a local mirror. Memory:
labels are held as per-sample category codes, scores as u8; ~45 B per called cell → ~13 GB for 297 M cells plus one organism's sort at
a time (~10 GB): a 64 GB node is enough. Class B cost of a full build: ~$0.25.

## 12. Clients

Python (`singlet.v2.atlas`):

```python
from singlet.v2 import atlas
cells = atlas.find_cells(organism="Homo sapiens", cell_type="T cell", tissue="UBERON:0002048", min_umi=1000,
                         ranges={"hallmark_hypoxia": [0.3, None]}, n=10_000, seed=0)       # seeded sample, <= 10 studies
more  = atlas.find_cells(..., cursor=cells.attrs["cursor"])          # the next, disjoint page of the same sample
n     = atlas.count_cells(cell_type="T cell", by=["tissue", "sex"])  # DataFrame of counts (not sampled)
adata = atlas.gather(cells, modalities=["counts"])  # AnnData: X, layers spliced/unspliced, obs gsm/gse/so/col, var genes
idx   = atlas.Index.open(base="/mirror")            # the same engine (sample_cells, count, find) on a mirror or over ranged GETs
```

`via="worker"` (default for an http base) uses the endpoints; `via="direct"` plans locally and issues the same coalesced range requests
(the comparison path of the cost model). `gather` decodes with `singlet._pz2` in parallel threads, aggregates rows to genes with the
vocabulary packs exactly as `singlet.v2.client` does (bit-exact against `client._read_sample`, tested), caches prefixes on disk by
member digest, and splits large requests into batches of at most 10,000 cells and 10 studies. A refusal for budget raises
`atlas.OverBudget` carrying the Worker's plan. Its `modalities` mirror the vocabulary that `singlet.v2.client` is adding
(`counts`, `spliced`, `unspliced`, `splice_junctions`, `mt`, `vdj`); when that lands, `atlas` should import it instead of its copy.

R and MCP (follow-up, not in this change):

* **R** (`r/R/v2.R`): `singlet_find_cells()` / `singlet_count_cells()` are thin `httr2` POSTs to `/v2/cidx/query` (JSON in, data frame
  out). `singlet_gather()` parses the SGA1 stream (a few dozen lines with `readBin`) and hands prefix + runs to the existing R pz2 decoder
  the way `load` does for ranged reads; the prefix cache keyed by member digest goes in the existing cache directory.
* **MCP** (`functions/_shared/mcp-v2.ts`): `find_cells` should call the index (one `runQuery` against the same objects, or a POST to the
  Worker) instead of scanning per-sample `cells` tables, which caps it at 25 samples today; `count_cells` grouped answers come from the
  summary at no row cost. `slice_cells` stays (byte ranges for one sample); a new `gather_cells` tool should return a ready-to-run Python
  / R snippet and the size of the plan rather than bytes.

## 13. Open points

* MCP "unchanged" answers: include `annotations_default` (and the index version, for index-backed tools) in what `rev` stands for.
* An anonymous global daily budget (one `CidxQuota` object named `anon:global`) would bound distributed anonymous abuse beyond the
  per-address quota; not enabled.
* Doublet, mitochondrial and cell-cycle columns: as soon as the `cells` tables carry them (format-v2 5.3 optional columns), add u8
  columns (schema v2 of the shards) and range fields.
* R2 latency inside Cloudflare was not measurable locally; the fan-out width (16 parts x 6) should be tuned with the first deployed
  measurements (`bench/cidx/measure.py https://data.singlet.bio`).

## 14. Measured on the pilot (local workerd)

GSE145370 (14 samples, 40,614 called cells, 31 MB of sample objects), fake labels of the annotation schema and two fake features
(u8 and u16; shards 545 KB = 13.4 B/cell with them), index built locally, the data
Worker running in workerd (Miniflare 4.20260617, `workers/data-v2/scripts/local.mjs`) with a seeded local R2 and Cache API; client on
the same laptop. Local R2 is a disk, so latencies are a lower bound for request handling, not a prediction of Cloudflare's; bytes, reads
and request counts are exact. `bench/cidx/results_local_pilot.json`; cold = first call (empty memos and cache), warm = third call.

| query | n | cold ms (client / Worker) | warm ms | rows scanned | response bytes | worst-case estimate $ | cost $ (cold) |
|---|---|---|---|---|---|---|---|
| count, summary (cell type) | 24,379 | 273 / 170 | 13 / 4 | 0 | 233 | | 0.0000040 |
| count by cell type x sex, summary | 40,614 | 12 / 6 | 11 / 5 | 0 | 3,134 | | 0.0000004 |
| count, row scan (cell type + n_umi) | 2,466 | 166 / 120 | 26 / 10 | 40,614 | 249 | | 0.0000031 |
| count, two feature ranges | 5,911 | 84 / 48 | 47 / 21 | 40,614 | 249 | | 0.0000009 |
| sample 1,000 cells (cell type + score), uniform | 1,000 | 238 / 214 | 117 / 64 | 40,614 | 126,563 | 0.0000155 | 0.0000016 |
| sample 1,000 cells, chunks | 1,000 | 129 / 115 | 176 / 138 | 40,614 | 125,957 | 0.0000155 | 0.0000009 |
| sample 10,000 of 40,614 cells | 10,000 | 378 / 294 | 313 / 215 | 40,614 | 1,257,243 | 0.0000156 | 0.0000009 |
| dry run of that sample | | 16 | 21 | | 799 | 0.0000156 | |

| gather (`counts`) of a seeded sample | samples | planned bytes (no prefixes cached) | reads | cold fetch / total s | warm fetch / total s | worst-case estimate $ | cost $ (cold) |
|---|---|---|---|---|---|---|---|
| 1 cell | 1 | 940,867 | 1 | 0.04 / 0.95 | 0.03 / 0.35 | 0.0000023 | 0.0000013 |
| 100 cells | 14 | 7,756,281 | 21 | 1.11 / 2.50 | 0.30 / 1.25 | 0.0000168 | 0.0000144 |
| 1,000 cells, uniform | 14 | 9,182,334 | 15 | 0.19 / 1.47 | 0.16 / 1.02 | 0.0000125 | 0.0000007 |
| 1,000 cells, chunks | 1 | 571,230 | 2 | 0.03 / 0.27 | 0.03 / 0.41 | 0.0000030 | 0.0000006 |
| 10,000 cells, uniform | 14 | 9,266,046 | 15 | 0.17 / 1.29 | 0.27 / 1.49 | 0.0000125 | 0.0000007 |
| 10,000 cells, chunks | 6 | 3,628,814 | 12 | 0.66 / 2.16 | 0.12 / 0.82 | 0.0000103 | 0.0000006 |

"total" includes decoding and building the AnnData (parallel decode: 10,000 cells went from 9.4 s single-threaded to 1.6 s with 4
threads); the 1-cell cold total includes loading the vocabulary packs. Correctness: Worker-streamed and direct gathers are bit-identical
to `singlet.v2.client`'s per-sample reader for 2,466 cells across 14 samples, including through the fan-out path.
