# Proposal: a faster, smaller `.singlet` / `.1pz` and the Cloudflare changes to serve it

2026-10-22. Written from the compression study (`REPORT.md`, `notes/frontier_fable.md`) and eight follow-up
audits run today (`notes/w3_total.md`, `w3_fast.md`, `w4_corpus.md`, `w4_atlas.md`, `w4_bugs.md`, `w4_vocab.md`,
`w4_reader.md`, `w4_cf.md`, `w4_mtx.md`) plus the parent's own read-only checks of the live service. Nothing here was
deployed, committed or changed in Cloudflare.

Evidence tags: **[M]** measured by the parent session on the live service; **[V]** verified by an agent on real files,
bit-exact where stated; **[E]** estimated from a measured sample; **[P]** projection from parts, no single run;
**[D]** Cloudflare documentation. Speeds were measured on a shared, throttled 4-core laptop (clock 1.2 to 2.9 GHz): read
every Mnnz/s as a lower bound; "calibrated" means scaled to 3 GHz from cycle counts.

---

## 0. Decisions, in order

| # | Do | Why (one line) | Effort | Risk |
|---|---|---|---|---|
| 1 | **Repair the stale member index and add a size/ETag guard** | Half the atlas serves wrong byte offsets today [M] | 1 to 2 days | low |
| 2 | **Make `data.singlet.bio` cacheable** (Cache Rule, CORS `*`, headers, Smart Tiered Cache) | Every response is `DYNAMIC`; OPTIONS returns 403 [M] | 1 day, no code | low |
| 3 | **Stop shipping what is derivable** (raw barcodes, features, filtered matrix, PSI, ambient, vocab) and move names to packs | About 40 points of live bytes, bit-exact, no new codec [V] | 1 to 2 weeks | medium |
| 4 | **Container v2 + class-order rANS payload** as per-sample, content-addressed objects | 0.41 of today's matrix bytes, no sort, 3 to 10x faster decode [V] | 3 to 5 weeks | medium |
| 5 | **Client rewrite of the read path** (member fetch, parallel ranges, one decode, lazy names, threads) | One sample 17.6 s to about 1.5 s modelled; 6.65 s CPU to about 0.5 s [P] | 2 to 3 weeks | low |
| 6 | **MCP diet + pack ids** | Every answer is sent twice; names cost 7x the tokens of ids [E] | 3 to 5 days | low |
| 7 | **Fix the pipeline defects** (PSI, ARC strand, summary fractions, docs) | See section 7 | varies | varies |

Decisions 1, 2 and 6 need no new format and can ship this week. Decision 3 is the largest single saving and
needs no new codec. Decision 4 is the speed work and should wait for the experiments in section 9 marked **gate**.

---

## 1. What is actually in the live data (corrects the earlier headline)

All from `notes/w4_atlas.md`: 180 uniformly sampled public bundles, range reads of the ZIP directories, 54.5 GB.

* **The catalog total is stale.** The catalog says 3.89 TB for 8,202 bundles. For the 180 sampled bundles the catalog says
  79.1 GB and the live files total 54.5 GB [M, D1 aggregate + HTTP]. Generation A bundles match (21.9 vs 21.2 GB); generation B
  shrank 42% (57.2 vs 33.2 GB) when the restricted-sidecar migration rewrote them. Live atlas is about **2.7 TB** [E], not 3.89 TB.
* **Two generations coexist.** Gen A (48% of bundles, 39% of bytes, built 2026-06-18): `.1pz` members and small JSON/TSV only.
  Gen B (52% of bundles, 61% of bytes, rewritten 2026-10-20/21): the same plus raw STARsolo outputs, ambient profiles and QC tables.
* **No restricted members remain** in the public layer: 0 of 34,252 sampled members. The September index that showed 31% of
  bytes as donor SNP matrices describes files that no longer exist.
* **Where the live bytes are** (share of live bytes, 95% intervals in `results/w4_atlas_tables.md`):

| member group | share | stored as |
|---|---|---|
| exon + intron `.1pz` | 42.9% | STORED zstd frames |
| STARsolo `matrix.mtx` (raw 13.6%, filtered 8.6%) | 22.2% | DEFLATE text, 4.4x |
| `sj_counts` 10.5% + `splice_psi` 6.2% | 16.7% | STORED |
| STARsolo `barcodes.tsv` | 7.1% | DEFLATE, 5.9x |
| `ambient_profile.tsv` | 3.8% | DEFLATE |
| everything else (vdj, mt, QC tsv, features, JSON, zip overhead) | 7.3% | mixed |

* **Cacheability.** 16.1% of bundles hold 52.3% of bytes and exceed the 512 MB CDN limit for non-Enterprise plans [D];
  they can never be edge-cached as single objects. Per sample, nothing exceeds 132 MB (median 25.7 MB), so a per-sample layout is
  100% cache-eligible.
* **Headers** [M]: `cf-cache-status: DYNAMIC` on every response including the immutable index file; no `Cache-Control` on bundles;
  no CORS headers; the OPTIONS preflight returns 403, so a browser page cannot range-read bundles; gen-B objects have no
  `Content-Type`; HTTP/2 is negotiated. Time to first byte for a tail read is about 100 to 275 ms; one stream moves about
  32 MB/s from this laptop (8 parallel streams measured 18 MB/s total here, so parallelism must be re-measured elsewhere).
* **Request shape.** A ZIP directory of a 143-member bundle is 11 KB, so one 64 KB tail read returns the directory (174 of 180
  sampled bundles fit in a 256 KB tail).

## 2. The production defect to fix first: stale member index

Reproduced end to end [M]:

1. The migration (`scripts/restrict_published_bundles.py:240-304`) overwrote bundles in place on 2026-10-20/21 and never refreshes
   the website's cached ZIP index (`bundle_index` in D1) or catalog sizes.
2. `functions/_shared/bundle-reader.ts:243-254` returns any cached row without comparing size or ETag.
3. GSE111113: D1, `/api/bundle/GSE111113/index` and `/api/gse/GSE111113` say 324,676,242 bytes (indexed 2026-09-06); the live object is
   170,777,036 bytes with 179 members, not 215. `/entry?path=samples/GSM3022283/summary.json` returns **502**. The recipe for
   `exon_counts.1pz` is `bytes=5745986-13467255` and the true payload starts 42,723 bytes earlier; the 4 bytes at the advertised start are
   `06 d5 3d 23`, not `TP1Z`. 6 of 6 sampled rewritten bundles mismatch; 2 of 2 untouched ones match.
4. About 52% of bundles (3,700 to 4,900) are affected. The stale value also feeds `get_download_url`, `get_study`,
   `export_manifest` and the cohort manifest.

Fix, no new format (details and file touch points in `notes/w4_cf.md` section 6, step 0): add `etag`/`last_modified` columns; verify the
local-header signature and name on every ranged read and re-index once on mismatch; send `If-Match` on range GETs; list the bucket
once and re-index only mismatching rows with a light mode that rewrites just the directory (about 4,300 rows, never `json_each`);
bump the edge-cache keys; add the refresh call and a read-one-member check to the migration script
(`results/w4_bugs_patches/restrict_published_bundles_index_refresh.patch`, unapplied, tested against a stub).

---

## 3. What can be removed without any new codec (the largest lever)

Every row below is a function of something the client already has, a per-build constant, or a protocol constant.

| member | live share | derivable from | status |
|---|---|---|---|
| filtered `matrix.mtx` + `barcodes.tsv` | 8.6% | raw matrix + `cell_calls.tsv` (keep listed columns, renumber, rewrite dims line) | [V] byte-identical 3/3 |
| raw `barcodes.tsv` | 7.1% | the 10x whitelist of the protocol, sorted; 467 sampled files have exactly **3 distinct contents** | [V] 3/3; 2 equal Track A models byte for byte |
| raw + filtered `features.tsv` | 1.0% | per-build vocabulary (2 distinct contents atlas-wide; order = genome order for human, stored permutation for mouse) | [V] |
| `splice_psi.1pz` | 6.2% | `sj_counts.1pz` with a pinned recipe | [V] 14/14 bit-exact **with caveats**: see below |
| `ambient_profile.tsv` | 3.8% | an integer vector `k` per exon interval; file regenerates byte-identically from `k` (`(k+0.5)/(T+0.5N)`, `%.6g`) | [V] 2/2; `k` costs 37 to 50 KB per sample (order-0) against 3.3 to 4.2 MB deflated today |
| `gene_expression.tsv` | 0.9% | exon + intron gene sums; TPM exact; FPKM normaliser unidentified | [V] 14,932/14,932 genes; partial |
| `cell_qc_metrics.tsv` | 0.8% | rows, `n_umi`, `n_genes`, `intronic_pct` exact (50,119/50,119); `n_reads` is new information | [V] mostly |
| intron row names, `feature_vocab.json` | n/a | exon names (introns are exactly the gaps between a gene's exons: **0 exceptions in all six builds**) | [V] |
| permutation, column pointers, colsums | 6 to 20% of a matrix file | per-row nnz; chunk metadata | [V] 152/152 |

Caveats that change the design [V, `notes/w4_corpus.md`]:

* **PSI rounding is not one rule.** Of 35 samples checked, 8 match round-to-nearest and 30 match truncation; 4 hold an empty stub
  although `sj_counts` has events. The recipe must be pinned **per pipeline version**, or the bundle should stop shipping PSI as a stored
  matrix and expose `psi = f(sj_counts)` computed on read with documented fraction semantics (section 7, item 2).
* **`cell_calls.tsv` is not derivable** for the 4 of 35 samples that used EmptyDrops with real p-values and keep non-called rows.
* **The raw STARsolo matrix** (13.6%) is not derivable but is predictable: where both exist, 99.5% (v3) and 95.4% (v2) of values equal the
  `.1pz` exon gene sum; coded conditionally it costs **8 to 14% of its deflated size** (0.17 to 0.22 MB per sample) against 37 to 55%
  for a standalone CSC coding [E, 2 samples, `notes/w4_mtx.md` section 3]. About 70% of that cost is the 17 to 20% of nonzeros
  in barcodes the `.1pz` dropped (mostly 1-UMI ambient droplets); keeping those droplets in the `.1pz` would shrink the remainder to
  1.3 to 1.5% of today's size.

### Atlas-wide projection (share of live bytes before to after)

`[V]` = verified lever, `[E]` = estimated, `[P]` = projected. Shares overlap slightly (rounded groups), so read the totals to
about 0.2x.

| lever | before | after | tag |
|---|---|---|---|
| exon + intron: container v2 + class-order rANS (x0.41) | 42.9% | 17.6% | [V] (`w3_total_v1` 0.407 on 14 files; 0.415 projected on 77 matrices) |
| raw STARsolo matrix, conditional on exon `.1pz` (x0.12) | 13.6% | 1.6% | [E] |
| filtered matrix + barcodes, raw barcodes, features | 16.7% | 0 | [V] |
| `splice_psi` | 6.2% | 0 | [V] with caveat |
| `sj_counts`, vdj, mt (names to pack, payload recode, x0.45) | 11.9% | 5.4% | [P] (GSE145370: 36.1 to about 16 MB) |
| `ambient_profile` (x0.012 measured), `gene_expression`, `cell_qc`, `star_SJ` | 6.3% | 1.1% | [V]/[E]/[P] |
| other tables, JSON | 4.2% | 2.8% | [P] |
| **Total** | **101.8%** | **28.7%** | **3.5x** |

Verified levers only: 1.9x. Verified plus estimated: 2.8x. Live atlas 2.7 TB becomes about 0.76 TB (all levers), 1.0 TB (verified+estimated),
or 1.4 TB (verified only). **Storage dollars barely move** (R2 is $0.015/GB-month, egress free [D]): about $40 per month today, about $11
to $21 afterwards. The benefit is transfer time and decode time for users, and it is large.

### What the first study's headline looked like against the measurements

| claim | result |
|---|---|
| 14 corpus matrices 114.4 to about 46 MB (2.5x) | **measured one codec**: 114.09 to 46.44 MB, 0.407, 2.46x, every roundtrip exact (`w3_total_v1`) |
| GSE145370 bundle 187 to 37 to 41 MB (4.5x) | **measured 48.16 MB** (3.9x) because only exon/intron were recoded (125.8 to 11.75 MB, 10.7x); sj/vdj/mt (36.1 MB) stay. About 28 MB if they get the same treatment [P] |
| payload -15.1% | holds pooled (-14.8%, 69 matrices) with per-matrix spread -35.9% to -6.9%; 8 barnyard matrices cannot be coded at all (row limit) |
| names identical per build | holds (5 human, 4 mouse, 3 barnyard, 2 zebrafish, 2 macaque, 2 rat studies; 77 matrices) |
| barcodes are whitelist members | holds for 10x v2/v3 only (100%); multiome 0.7%, Visium 0.1%, non-10x 0.02% |
| second-pass "recursion" finds nothing | holds: 0.9991 to 1.0002 on all 14 `w3_total_v1` streams |

---

## 4. Format v2 (what to build)

### 4.1 Principles

1. **Nothing derivable is stored.** Names, permutation, column pointers, colsums, raw barcodes, features, filtered matrix, PSI, ambient
   profile live in packs or are recomputed by pinned recipes (section 5).
2. **Identity is the decoded matrix, not the container bytes.** Container byte identity needs the exact zstd build
   (1.5.7 reproduces 87/87 frames on brain and PBMC files, 1.5.6 reproduces the T-cell and tumour files). A v2 member stores a 16-byte
   digest of the canonical decoded CSC and readers verify it lazily.
3. **Per-sample, content-addressed objects.** A sample's essential members (exon, intron, sj, small metadata) in one STORED container,
   key `s/<GSM>/<digest16>.s2`, `Cache-Control: public, max-age=31536000, immutable`. A rewrite creates a new key, so cached offsets cannot go
   stale again. Only the small study manifest `idx/<GSE>.json` is mutable (TTL minutes) and names the current digests. Median
   object is expected at 3 to 8 MB.
4. **Index first.** Header and chunk table sit at the front so the first range read (4 to 64 KB) already tells the client where any
   column range lives; today names sit at the end and the chunk table behind the permutation (10 requests in 3 rounds before the first cell;
   about 2 requests after) [E4 audit].
5. **Decoder is integer-only, runtime-dispatched** (AVX-512, AVX2, scalar; NEON port described), byte-identical across ISAs.
6. **Legacy view.** Keep `data/<GSE>/<GSE>.singlet` as a compatibility copy with a short TTL; optionally assemble a ZIP64 view in a
   Function from the stored members (CRC-32 and sizes are known) when old bundles are retired.

### 4.2 Matrix member (`1pz2`)

Header (fixed) with magic, version, flags, `m`, `n`, `nnz`, value width, `chunk_cols`, classes per octave, **8-byte pack id** (reference
vocabulary for this build and member kind), matrix digest. Then chunk table (offset, columns, nnz per chunk), row side info (per-row class
from nnz counts, 2.1% of payload), column side info (per-column nnz 1.2%; barcodes 5.0% as sorted whitelist indices where a whitelist exists,
inline sorted-delta otherwise), static per-file tables (0.3%), then chunk streams (91.3%): 32 lock-step rANS lanes, 10-bit probabilities,
contexts rank-position (40) x column-depth (12), "value > 1" flag folded into the gap symbol, residual values in a side stream.

**Row order is the speed decision.** Store rows in the file's frequency-class order (4 classes per octave: +0.006 bits/nnz); the decoder then
emits ascending class-local ids with no sort, and the client receives the matrix plus a row permutation (applied to the names once per file).
An exact canonical mode (original row order) stays available and costs a per-column sort. `w3_fast` built this: payload 8.226 bits/nnz (mean of
14 files, same as Track B), all 14 files and both orders exact.

### 4.3 Measured decode speed

| decoder | Mnnz/s, single thread, measured | calibrated to 3 GHz | note |
|---|---|---|---|
| today's production reader (full decode incl. names) | 4 to 21 (3.9 / 5.4 / 6.8 on three files) | n/a | 52% in a per-column sort on dense files, 86% in name parsing on sparse ones |
| `b_rans16`, canonical order | 34 to 51 | about 80 | sort costs as much as entropy decoding |
| **class-order decoder (`w3_fast`)** | **69 to 89** (117 in the best quiet window) | **159 to 207** (240) | no sort; scalar twin is bit-identical |
| 4 threads, `tcell_5p` exon, class order | 539 (quiet window) | 819 | 8 hyperthreads are slower than 4; memory bandwidth is not the limit |
| `w3_total_v1` full decode incl. names | 17.5 to 72 | n/a | name materialisation is 22 to 63% of it |

Unmet: 200 Mnnz/s single-thread on measured wall clock; the loop still spends 14 to 15 cycles per nonzero against about 4 from a uop count
(unlocated). Smaller tables (80 KB vs 1.9 MB) gave the same speed, so decode is not cache-bound; 64 lanes were slower than 32.

### 4.4 Limits to fix before freezing the format

* **Rows above 2^19.** The lane codec rejects `m > 524,288`; all 8 barnyard matrices failed in `w4_corpus` (barnyard is 2.3% of bytes). Barnyard rows
  are the human rows plus the mouse rows, so either widen row ids to 21 bits or code two row blocks per column. Gate experiment E2.
* **Tiny and degenerate matrices.** An empty matrix (151-byte members) cannot round-trip through `a_names_perm_model`; sparse plate-based and
  Visium samples are 88 to 99% removable but pay fixed per-chunk overhead at 1,024 columns. Gate experiment E6 (chunk size 256 to 4,096).
* **Pipeline-version dependence** of derived members (PSI rule) must be a field in the member table: `recipe_id` plus `pipeline_version`.

---

## 5. Side information: packs, and how Python, R and the MCP use them

### 5.1 What the packs are (`notes/w4_vocab.md`)

| pack | contents | size (all [V], zstd-19 after structured coding) |
|---|---|---|
| vocabulary, human GRCh38-2024-A | exon + intron rows, gene table, `feature_vocab.json` | **1,144,091 B** |
| vocabulary, mouse GRCm39-2024-A | same | **1,529,232 B** |
| zebrafish / macaque / rat | same | 967,625 / 878,852 / 874,975 B |
| barnyard | composite of human + mouse with species tags | **66 B** |
| whitelist, 10x v2 (737K) / v3 (3M) / multiome | sorted-set pack | 3.2 KB / 6.4 KB / 8.5 KB |

Total for all six builds: 5.39 MB, 2.9 to 3.2x smaller than xz-9e of the plain names. Expansion matches the names of every fetched study byte for
byte (30 of 30 study/matrix pairs); intron names cost zero; `feature_vocab.json` regenerates byte for byte. Gate for the id scheme:
`reference_manifest_sha256` is identical across five builds and cannot serve as a vocabulary id; use the sha256 of the canonical pack (8-byte prefix in
headers). Mouse names were checked only on pipeline sha `37e7cbd`; none from the newer `8e1803b` build were in the sample (experiment E9).

### 5.2 Distribution

* **Python wheel:** ship human and mouse vocabulary packs and the three whitelists (2.7 MB plus under 20 KB); other builds download on first use
  to the cache directory with sha256 verification.
* **R:** CRAN's tarball guidance is about 5 MB, so ship only the whitelists and a manifest; fetch packs on first use into `tools::R_user_dir()`
  with sha256 verification (human plus mouse is 2.7 MB and could also fit).
* **Server:** `data.singlet.bio/ref/<id>.pack`, immutable, edge-cached; the content never changes under its id. Old readers keep working on
  the legacy bundles; a revised reference build gets a new pack id.
* **Materialisation speed (C++):** lazy lookup 0.23 to 0.72 microseconds; a full 310k-name array 12 to 38 ms (38 to 89 ms cold from pack bytes).
  The 30 ms full-array target is missed; Python is slower (150 ms to build a `list[str]` of 310k names), so expose lazy names and a numpy-backed
  index by default.
* **Barcodes:** whitelist-index coding saves about 1.5 to 1.7 KB per 1,000 columns; the pack pays for itself in 2 to 5 samples at 1,000
  columns and in under one sample at 10,000+. Non-10x protocols (Quartz-Seq2, CEL-Seq2, MARS-seq2, BD, inDrop, SPLiT-seq, Visium) have no
  usable whitelist and code barcodes inline; the SPLiT-seq round structure is unresolved. Visium spot sets repeat exactly (2 samples share
  4,991 spots) and could be per-slide packs.

### 5.3 MCP (hosted, `functions/mcp.ts`, `_shared/mcp-tools.ts`)

Measured or read from code (`notes/w4_cf.md` section 4, `notes/w4_reader.md` section 4):

* Every result is sent twice, as `content[].text` and `structuredContent` (`mcp-tools.ts:36-38`): send one.
* `export_manifest` sends a 520 KB TSV twice; `/api/cohort/manifest` returned 1,065,743 bytes for 20 samples, of which 1,145,544 characters are a
  `downsample.dropped` list repeated on every page and 27.5 KB is the samples.
* Proposed protocol: each tool takes optional `known_rev`; a study `rev` (hash of bundle ETag, `gse_meta.updated_at`, max
  `sample_qc.updated_at`) lets the server answer `{unchanged:true, rev}`. Responses reference packs by id, never by names:
  `{pack:"<id>", bytes, url, sha256}` once per session; rows are integer indices; a `resolve_features(pack, indices)` tool returns
  names on demand. `get_partial_download` returns `{url, etag, bytes, codec:{id, version}, pack}` so a ranged member can be decoded without extra
  requests. Estimated token saving: 100 interval names cost about 1,290 tokens versus about 191 as indices plus a one-time 47-token descriptor [E].
* Drop per-row loader strings, three URLs per search row, duplicated QC fields, per-call caveats and billing notes; share enum schemas by `$ref`
  (`tools/list` is 15.7 KB). Version the result shape and keep the old one behind `verbose` for one release.

---

## 6. Cloudflare: serving layer changes, in rollout order

Layout facts first (`notes/w4_cf.md`, repos `singlet-website` and `singlet-cloud`): the service is a Pages project with Functions; R2 `singlet-data` is
served at `data.singlet.bio` (location hint ENAM); D1 `singlet-catalog` (378 MB) holds the catalog and caches; main auto-deploys and **previews share
production D1**; D1 has a daily read budget (never `json_each` over `bundle_index`). Cache Reserve does not apply to R2 custom domains [D], so it is
out of the design. The Functions do not appear in the Workers list because they are Pages Functions.

| step | change | files / places | notes |
|---|---|---|---|
| 0 | **Index repair and guard** (section 2) | `schema/031_bundle_index_identity.sql`, `bundle-reader.ts`, `ingest/[table].ts`, tests, `smoke.sh`; migration script patch | before anything else; roll back by Pages rollback |
| 1 | **Make the data host cacheable** | dashboard: Cache Rule "eligible for cache" on `data.singlet.bio`, Edge TTL from origin `Cache-Control`, Smart Tiered Cache on; R2 CORS to `*` (and make OPTIONS succeed); Content-Type and Cache-Control set at upload | keep legacy `*.singlet` at <= 1 h TTL or bypass; old bundles are overwritten in place; test on a staging hostname first |
| 2 | **MCP diet, `rev`/`known_rev`** | `mcp-tools.ts`, `mcp.ts`, `study-core.ts` | no storage change; golden-size tests |
| 3 | **Per-sample content-addressed objects** | producer script (`pack_gse` writer + copy job: about 78k PUTs, under $0.50 of Class A), `getPartialDownload`, entry route | additive; doubles storage while both exist (about +$40 per month) |
| 4 | **Static member/chunk index** `idx/v2/<GSE>.<digest>.idx` | `bundle-reader.ts`, `cohort-manifest.ts` (`loadEntries`, `loadChunkRows`), `schema/033` | D1 keeps `(gse_id, idx_digest, etag, bytes)` only; stop repeating `downsample.dropped` |
| 5 | **Side-information packs** at `ref/<id>.pack` | `mcp-tools.ts`, `study-core.ts` | freeze the format first |
| 6 | **Clients** | `python/singlet/_loader.py`, `fetch.py`, `r/R/load.R` | section 8 |

Cost [D, E]: R2 storage $0.015/GB-month, Class A $4.50 per million, Class B $0.36 per million, egress free. Ops are small next to storage;
caching helps latency and Function load, not the bill. Non-North-America users: the bucket's location hint is fixed at creation; per-sample
objects plus Cache Everything plus Smart Tiered Cache serve hot objects from the user's edge without a second bucket (regional latency not
measured: experiment E8). Dashboard-only checklist (Workers plan, D1 plan and daily usage, existing Cache Rules, CORS as stored, who produced the
immutable index file): `notes/w4_cf.md` section 8.

---

## 7. Defects and data-quality tickets (`notes/w4_bugs.md`, `w4_mtx.md`, `w4_corpus.md`)

Ordered by what to do first. Patches are written and **unapplied** under `results/w4_bugs_patches/`.

| # | defect | evidence | severity | action |
|---|---|---|---|---|
| 1 | Stale member index and catalog sizes after migration | section 2 [M] | critical | step 0 |
| 2 | **Shipped `splice_psi.1pz` is `round(PSI)` in {0,1}** (readers using it as a fraction get an indicator) | shipped = `nearbyint(PSI)` on 4,664,595 of 4,664,595 entries (14 samples); in-tree writer truncates and differs on 56,947 entries; rounding exists in no commit since 2026-04-28, so the production binary came from an uncommitted tree | high | store `lround(PSI*10000)` as uint16 with `value_scale` in the member (3 samples tested: max error 5e-05, +3% size) or compute from `sj_counts` on read; regenerate published files in the same pass as step 0; pin the recipe per pipeline version |
| 3 | **10x multiome (ARC) `.1pz` counts look strand-reversed** | only 3 to 4% of STARsolo nonzeros appear in the `.1pz` gene matrix; top cell has MALAT1 456 in STARsolo but TALAM1 427 and MALAT1 43 in the exon `.1pz`; about 800 samples (1.1%) use the ARC protocol | high if confirmed | confirm against BAM/STAR output (experiment E7) before reprocessing |
| 4 | Possible under-counting: 13 of 14 GSE145370 samples have 50 to 77% of reads flagged `wrong_strand` | from `summary.json`; ordinary 10x v2/v3 samples agree with STARsolo (99.5% / 95.4% equal values), so this is **unverified** and may be a labelling issue | to check | E7 |
| 5 | `summary.json` fractions divide UMIs by reads: `intergenic_fraction` is 94.8 to 99.0% in all 14 samples (`export.h:1336-1343`) | [V] | medium | fix the denominator or rename the field |
| 6 | `cell_calls.tsv`: p-value column is BH FDR or the sentinel 0 for `star_solo`; the default vendored filter is a UMI knee (CellRanger2.2), not EmptyDrops_CR, and the filter used is not recorded; 4 of 35 samples use `emptydrops` | [V] | medium | record the filter; documentation fix |
| 7 | 20 doc/code mismatches in `format-1pz.md`, `format-spec.md`, `format-singlet.md`, `pipeline.md` (value-type codes 6/5/3 in the doc vs 1/2/3 in the writer; `format-spec.md` describes a different format; `CANONICAL_OUTPUT_FORMAT.md` referenced but absent) | table in `notes/w4_bugs.md` section 7 | medium | one spec, generated from the code |
| 8 | `compute_psi` comment says donor events win; code takes first event by string-sorted id (1 of 9,293 junctions in GSM4317429) | [V] | low | two-pass fix in the PSI patch |
| 9 | Migration script hazards: lost-update window (`restrict_published_bundles.py:278`), unverified rollback (`:283`), 64 MiB parts change ETag semantics | review table | medium | patch + 8 MiB parts + verified upload |
| 10 | Writer arithmetic wrap at `max_val+1`, unchecked row indices | unreachable from the pipeline today | low | validation patch |

Reader hardening: 13,812 byte-level mutations of `read_1pz_buffer`: 0 crashes, 0 hangs; the reader verifies the footer CRC and every chunk CRC.
Row and column names are not authenticated (425 rewritten-name cases accepted after recomputing the CRC), another reason to put names in
content-addressed packs.

---

## 8. Client read path: where the time goes and what to build (`notes/w4_reader.md`)

Today, one sample (GSM4317426, exon + intron), download excluded, machine at 96 to 100% load [V]:

| stage | ms | share |
|---|---|---|
| native decode (4 calls: every file decoded twice) | 1,134 | 17.0% |
| pybind conversion (names to `str`, element-wise array copies) | 839 | 12.6% |
| feature-name to gene mapping (Python loops, both passes) | 1,890 | 28.4% |
| `AnnData` + pandas `Index` of names for the layers | 1,530 | 23.0% |
| matrix products and subsets | 796 | 12.0% |
| zip extract, temp file, glue | 465 | 7.0% |
| **total** | **6,654** | |

Name handling is 51% of load time; native decode is 5.6%. `to_anndata(layers=True)` takes 60.3 s on the 14-sample bundle against 23.5 s with
`layers=False`. Four integrity passes (zip CRC-32, SHA-256, file CRC-32, chunk CRC-32) read the same bytes. A cache hit never revalidates an ETag.
R: roughly 10 s for the same sample (two reads 5.1 s, two aggregations 2 x 2.4 s), data stored as double (12 bytes per nonzero per assay), names
about 39 of 46.7 MB.

Build, in this order:

1. **Fetch only what is needed.** `load("GSM...")` pulls the member list from the study manifest and ranges for that sample (modelled: one sample of
   a 187 MB bundle at 100 Mbit/s, 17.6 s whole-bundle to 1.4 to 2.2 s; at 1 Gbit/s 7.7 s to 0.8 to 1.7 s) [P]. Verify `Content-Range` total,
   ETag (`If-Match`) and local-header signature on every ranged read, with a fallback to the live central directory.
2. **One decode, no temp file, no per-element copies.** The prototype (`kernels/w4_reader_dec.h`) reuses the production chunk kernels, decodes chunks
   in parallel into caller-owned int32/uint16/float buffers, never touches names, and supports a chunk range. It is bit-identical to production on 5
   files and 1.5x to 10x faster (name-dominated samples 4 to 10x; dense matrices 1.5x at one thread, up to 4.8x with threads) [V]. Projected sample load
   with glue fixes about 0.4 to 0.6 s [P, unmeasured].
3. **Lazy names**, pack-backed, numpy fixed-width index; gene mapping precomputed in the pack (not Python loops).
4. **Derive `X = spliced + unspliced`** instead of recomputing; stop writing temp files.
5. **Parallel ranges (8 to 16 x 8 MiB), resume, backoff**; skip re-hashing when the ETag matches; revalidate cached bundles with `If-None-Match`; R timeout
   from a size estimate (today `max(600, ...)` fails above about 19 GB at 32 MB/s).
6. **Packs and class-order decode** once the format is frozen.

---

## 9. Experiments still to run (ordered by value per day)

"Gate" means the answer changes what we build. All can run on this laptop unless noted; items marked (quiet) need an otherwise idle machine.

| id | experiment | decides | effort | success criterion |
|---|---|---|---|---|
| E1 | **Idle-machine timing** of the class-order decoder and `w3_total_v1`: single thread, 1/2/4 threads, min of 5, clock logged (quiet) | **gate**: whether 200 Mnnz/s single-thread is real; thread default | 0.5 day | clean numbers; if under 150 calibrated, profile the 14 cycles/nnz |
| E2 | **Widen row ids** (21 bits) or two-block rows for barnyard; size and speed impact | **gate**: barnyard (2.3% of bytes) cannot be coded now | 1 day | all 8 barnyard matrices round-trip, speed within 5% |
| E3 | **Recode `sj_counts`, vdj, mt** with the same machinery: junction names to a per-build pack, counts conditional on gene counts (E5 of the study was never run) | **gate**: 11.9% of live bytes is projected, not measured; sj is the largest remaining payload | 3 days | measured bytes per sample vs the 0.45 assumption |
| E4 | **End-to-end raw `matrix.mtx` conditional codec**, real coder, on 6 samples including 2 large ones; policy for dropped low-count droplets (keep in `.1pz` vs side stream) | 13.6% of live bytes is an estimate from 2 samples | 3 days | within 20% of the 0.12 estimate; ARC anomaly isolated |
| E5 | **Pin the PSI recipe per pipeline version** and verify byte-identical regeneration on 10+ samples per version, including the 4 stub samples; decide stored-scaled vs computed-on-read | 6.2% of bytes and a correctness issue | 2 days | regeneration passes or per-version rule documented |
| E6 | **Chunk size and probability-bit grid** (256/1024/4096 columns; 10/11/12 bits; 4 vs 8 classes per octave; contexts), including column-range decode cost | default parameters; tiny-matrix overhead | 1 day | defaults per size class |
| E7 | **Strand check** on GSE145370 and one ARC sample against BAM or STAR output; count antisense assignment | tickets 3 and 4: reprocess about 800 ARC samples or not | 1 to 2 days | verdict |
| E8 | **Staging Cloudflare test** (new hostname or bucket): Cache Rule plus Smart Tiered Cache on a 100 MB and a 600 MB object; MISS/HIT and Range behaviour; TTFB from 3 regions (cloud VMs); parallel-range throughput off this laptop | **gate** for step 1 and for the per-sample size target; 8 streams were slower than 1 here | 1 day (needs account owner and VMs) | HIT ratio and TTFB table |
| E9 | **Name invariance under the newer pipeline sha** (`8e1803b`) for mouse and the `reference_manifest` id question; SPLiT-seq barcode structure | pack id policy | 0.5 day | no variants, or version the pack |
| E10 | **Wire the prototype decoder into `to_anndata`** and time `load()` end to end in Python and R | confirms the 0.4 to 0.6 s projection | 2 days | measured |
| E11 | **ISA coverage**: AVX2 kernel measured; NEON port plan or build on ARM; runtime dispatch test | most users lack AVX-512 | 2 days | speed table per ISA |
| E12 | **Remaining derivations**: `ambient_profile` integer vector size, `star_SJ.out.tab` vs `sj_counts`, `cell_qc.n_reads`, FPKM normaliser, mouse `features.tsv` order | small bytes, completes the "derived" list | 1 day | byte-identical or documented residual |
| E13 | **Real traffic**: R2 metrics (Class A/B by operation) and zone analytics for `data.singlet.bio`; which members users actually fetch | prioritise hot objects, validates the cost model | owner task | request mix |
| E14 | **c_r2/E1 replication on the diverse corpus** (does the 0.04 to 0.12 bit state-term bound hold on Smart-seq-like and plate protocols?) | **not needed for the plan**; only if someone wants a model-based coder (it would add at most a few percent) | 2 days | optional |

Not worth running (answered, with numbers): shipped learned models, cell-type or latent side information, exon-to-intron joint coding, per-chunk codec
orchestration, a recursive second pass, an ambient coder for empties (0.35% smaller, 7x slower to decode).

## 10. Rollout and risk

1. **This week:** step 0 (index repair), step 1 on a staging hostname then production, MCP diet, PSI patch and migration-script patch review, E1, E2, E7.
2. **Weeks 2 to 3:** derived-member removal for gen B (filtered matrix, raw barcodes, features, PSI, ambient) as a pure re-pack of existing bundles; E3, E4, E5; this
   alone is about 20 points of live bytes with no new codec.
3. **Weeks 3 to 6:** `1pz2` writer and reader, per-sample objects behind a feature flag, packs, static index; E6, E10, E11.
4. **Weeks 5 to 8:** clients and the MCP pack protocol; keep the legacy bundle for at least one release; retire after usage shows migration.
5. **Risks:** a re-index storm (cap at the crank lock rate); caching a bundle that is later overwritten (hence short TTL on legacy objects and immutable keys
   on v2); D1 daily read budget (never scan `bundle_index`; use the bucket listing); derived members depending on a pipeline version (store `recipe_id`);
   mirrored code in `singlet-website` and `singlet-cloud` (land each change twice or extract a shared package first); the two migrations (restricted
   removal now, v2 later) must not rewrite the same objects without an index refresh each time.

## 11. Where each number comes from

| topic | note | tables |
|---|---|---|
| combined codec, 14 files and bundle | `notes/w3_total.md` | `results/w3_total_14.tsv`, `corpus_w3_total/GSE145370.w3.zip` |
| fast decoder | `notes/w3_fast.md` | `results/w3_fast_*` |
| diverse corpus (77 matrices, 39 samples) | `notes/w4_corpus.md` | `results/w4_corpus_perfile.tsv`, `w4_corpus_e4*.tsv` |
| live bundle audit (180 bundles) | `notes/w4_atlas.md` | `results/w4_atlas_tables.md`, `w4_atlas_levers.md`, `w4_atlas_members.tsv` |
| legacy members derivable | `notes/w4_mtx.md` | `results/w4_mtx_*` |
| bugs and patches | `notes/w4_bugs.md` | `results/w4_bugs_patches/`, `w4_bugs_psi_all.tsv` |
| packs | `notes/w4_vocab.md` | `results/w4_vocab_packs/` |
| client path | `notes/w4_reader.md` | `results/w4_reader_*` |
| serving layer | `notes/w4_cf.md` | `results/w4_cf_requests.tsv` |
| theory and earlier synthesis | `notes/frontier_fable.md`, `notes/theory_fable.md`, `REPORT.md` | |

Open caveats stated once: speeds are lower bounds from a throttled shared laptop; the `matrix.mtx` conditional coding rests on 2 normal samples (a third, an ARC
sample, is anomalous); atlas-wide shares come from 180 bundles sampled uniformly over bundles, not bytes (their mean 302.6 MB is below the catalog's stale
474 MB), so confidence intervals are wide on the large-bundle tail (no sampled bundle exceeded 3.84 GB; the catalog maximum is 29 GB); the lever table
adds separately measured components; no Cloudflare resource was changed and the dashboard-only settings are unread.
