# w4_cf: audit of the singlet.bio serving layer (code, live headers, change list)

Status: COMPLETE for the requested deliverables (2026-10-22 07:40 EDT hard stop respected). Owner: `w4_cf`. Files: this note,
`results/w4_cf_requests.tsv` (one row per access path), `tools/` unused. Nothing in any repository was modified; no wrangler, git,
npm, D1/R2/KV or MCP call was made. Live traffic: 24 GET/HEAD to data.singlet.bio (8) and singlet.bio (16) (list in section 9).

Evidence labels used below: **[M]** measured live by me today; **[M-parent]** reported by the coordinator, not re-measured by me;
**[W4]** read from the sibling agent's scan on disk (`results/w4_atlas_*`, 180 bundles drawn uniformly from 8,202); **[C]** read
from code (file:line); **[D]** from Cloudflare docs fetched today; **[?]** cannot be determined from code, see the checklist.

Audited tree: `C:\Users\zacha\Projects\singlet-website` is checked out on branch `fix/publish-exec`, not `main`. The files that matter
(`bundle-reader.ts`, `bundle-core.ts`, `api/bundle/[gse]/[action].ts`, `cache.ts`, `cors.ts`, `manifest-core.ts`, `catalog-refresh.ts`,
`identity.ts`) are byte-identical in `singlet-cloud` (branch `main`), so line numbers hold for both; `mcp.ts`, `mcp-tools.ts`,
`study-core.ts` and `api/ingest/[table].ts` differ only in wording, the completeness block and cohort/search-fact tables.
**`python/singlet/_restricted.py` says `functions/_shared/restricted.ts` filters what the site lists; that file does not exist in
either tree.** Production evidently runs something newer than the audited tree (the parent saw 215 index entries where the API I
called lists 179 for GSE111113). Treat "what the live Function does with restricted members" as unverified.

## 0. The seven things that matter

1. **The cached member index is stale for every bundle the restricted-sidecar migration rewrote, and nothing checks.**
   `getBundleIndex` returns any cached D1 row (`functions/_shared/bundle-reader.ts:243-254`) and never compares size or ETag with
   the live object. The migration (`scripts/restrict_published_bundles.py:240-304`) overwrote objects in place on 2026-10-20/21
   and never calls the site to re-index. [M] GSE111113: D1/API say `bytes=324,676,242`, `indexed_at=2026-09-06T04:15:53Z`; the live
   object is 170,777,036 bytes, ETag `...-3`, Last-Modified 2026-10-20 22:21:57. The recipe from
   `/api/bundle/GSE111113/entry?path=samples/GSM3022283/exon_counts.1pz` says `bytes=5745986-13467255`; the true payload starts at
   5,703,263 (local header at 5,703,179 with `extraLen=20`), so the recipe is 42,723 bytes late and the bytes it points at are not
   a `TP1Z` file. [M-parent] `/entry` returns 502 for small members of such studies. The same stale row feeds
   `get_partial_download`, `/api/cohort/manifest` (`loadEntries`, `cohort-manifest.ts:357-376`) and the 24 h edge cache of recipes.
2. **How many.** [W4] 94 of 180 sampled bundles (52.2%, 95% CI 45-60%, i.e. about 3,700-4,900 of 8,202; 61% of sampled bytes) have
   Last-Modified on 2026-10-20/21. [M] 6 of 6 rewritten studies I checked have a D1 `bytes` that differs from the live size and
   `indexed_at=2026-09-06`; 2 of 2 June controls match exactly. Catalog sizes are stale too: `/api/gse/GSE111113` reports
   `bundle_bytes=324,676,242`, which `get_download_url`, `get_study` and `export_manifest` repeat.
3. **data.singlet.bio is not cached at the edge, and the cause is probably configuration, not size.** [M] Every response is
   `cf-cache-status: DYNAMIC`, including `search-index/v1/index.json` (`Cache-Control: public, max-age=300`) and the `.sbix`
   (`Cache-Control: public, max-age=31536000, immutable`). `.json`, `.sbix` and `.singlet` are not in Cloudflare's default
   cached-extension list [D]; R2 custom domains cache only default types unless a Cache Rule/page rule says Cache Everything [D].
   The comment "Cloudflare serves it from cache" (`bundle-reader.ts:10`) is false for this host. Separately, the 512 MB CDN limit on
   non-Enterprise plans [D] excludes 15% of bundles holding 50% of bytes in my sample [W4] (16% / 52% [M-parent]).
4. **Per-sample data already fits under the cache limit.** [W4] 1,705 sampled samples: median 25.7 MB, p99 111 MB, max 131.6 MB,
   none above 512 MiB. A per-sample (or per-member) object layout makes 100% of the per-sample traffic cacheable.
5. **Cohort and MCP answers are bulky by construction.** `/api/cohort/manifest` returned 1,065,743 bytes for 20 samples [M]:
   1,145,544 characters of it are `downsample.dropped` (17,358 samples), repeated on every page; the 20 samples themselves are 27.5 KB.
   MCP puts every result in `content[].text` and again in `structuredContent` (`mcp-tools.ts:36-38`), so each answer costs twice.
6. **Client behaviour multiplies bytes.** `singlet.load("GSM...")` downloads the whole parent study then filters
   (`python/singlet/_loader.py:509-548`); Python/R have no Range use, no resume, no parallelism, and never revalidate a cached file.
   R's `download.file` runs with `timeout = max(600, old)` (`r/R/load.R:325`): at the measured 32 MB/s that fails above about 19 GB.
7. **Cache Reserve does not help this design.** [D] "Requests to R2 public buckets linked to a zone's domain don't use Cache
   Reserve", and Origin Range requests are not supported from Cache Reserve. Use Cache Rules + Smart Tiered Cache + smaller objects.

## 1. Request and cost map

Full machine-readable version: `results/w4_cf_requests.tsv` (22 rows). Costs: R2 Class B $0.36/M (GetObject, HeadObject),
Class A $4.50/M (Put, UploadPart, Create/CompleteMultipart, List), egress $0, storage $0.015/GB-month [D]. The "R2 ops" column counts
operations R2 bills; "D1 stmts" counts statements the Function issues.

| path | what is requested, from where | cached? | headers | metered | R2 ops / D1 stmts per call |
|---|---|---|---|---|---|
| Browser download (`StudyTable.tsx:139`, `StudyCard.tsx:98`, `DownloadPanel.tsx:83`) | `<a href download>` straight to `data.singlet.bio/data/<GSE>/<GSE>.singlet`; no Function, no D1 | no: DYNAMIC [M] | ETag, Last-Modified, Accept-Ranges, `nosniff`; no Cache-Control, no Content-Disposition; Content-Type missing on ~46% of objects [W4] | nothing | 1 Class B; 0 D1 |
| Browser multi-download (`SelectionBar.tsx:36-46`) | hidden iframes, one per study, 400 ms apart, removed after 60 s | no | as above | nothing | 1 Class B per study |
| Python `singlet.load("GSE")` (`_loader.py:159-282`) | one urllib GET, 64 KiB reads, socket timeout 120 s, UA `singlet-loader/1`; then local SHA-256 of every member (`_integrity.py:180-188`) | disk cache by path only; reused if the zip is structurally complete (`:213-217`); never compared with server ETag/size | | nothing | 1 Class B; 0 D1 |
| Python `load("GSM")` (`:509-548`) | `/api/gsm?q=` then `/api/search?q=` (D1), then the whole parent bundle, then filter | as above | | AI quota not touched | 1 Class B + 1-2 D1-backed GETs |
| R `singlet::load` (`load.R:303-352`) | HEAD for size (`:320`), `download.file`, verify; GSM parent cached in `gsm_parents.tsv` | disk cache | | nothing | 2 Class B |
| MCP `get_download_url` (`mcp.ts:761-788`) | one D1 join `gse` x `gse_meta`; returns URL, `bytes` from `gse.r2_bundle_bytes`, loaders, curl | `no-store` (`mcp.ts:91`) | | key, 60/min, `mcp_usage` | 0 R2; D1: 3 metering + 1 read |
| MCP `get_partial_download` (`mcp-tools.ts:356-419`) | `bundle_index` row (whole entries JSON) + **one 30-byte Range GET to the public host** (`entryRange`, `bundle-reader.ts:309-314`) | none for the subrequest | | key, 60/min | 1 Class B (+2 client Range GETs for exon+intron); D1: 3 metering + 1 row read (8.7 KB median, up to 185 KB [W4]) |
| MCP `list_bundle_files` / `get_modalities` / `get_sample_qc` / `assess_study` | `bundle_index` and/or `sample_qc` rows, `bundle_manifest`; on a miss the Function reads the zip tail + central directory (HEAD + 256 KiB + CD) and, for QC, one subrequest per `summary.json` (6 concurrent, `bundle-core.ts:192-209`) | D1 is the cache | | key, 60/min | miss: 3 + n_samples Class B; hit: 0 |
| MCP `export_manifest` (`mcp-tools.ts:425-495`) | ids: chunks of 90 over `gse` x `gse_meta` x `bundle_manifest` (`manifest-core.ts:41-84`); query: search scan then the same | `no-store` | | key | D1 reads proportional to studies (<=2,000) |
| REST `/api/bundle/:gse/index|samples` (`[action].ts:48-78`) | D1; Cache API per colo | **24 h** (`DAY`, key `v=1`), `X-Edge-Cache` MISS/HIT [M] | `public, max-age=86400, s-maxage=86400` | none (no key), WAF rule 60/min/IP recommended [?] | D1 1-2 reads on miss |
| REST `/api/bundle/:gse/entry` <=4 MB (`[action].ts:110-118`) | D1 row + one Range GET to the public host + `DecompressionStream` inflate | 24 h per (study, path) | | none | 1 Class B + 1 D1 on miss |
| REST `/api/bundle/:gse/entry` >4 MB (`:90-108`) | D1 row + `entryRange` 30-byte Range GET; returns JSON recipe | **24 h: a wrong recipe stays wrong for a day** | | none | 1 Class B + 1 D1 on miss |
| REST `/api/cohort/manifest` (`api/cohort/manifest.ts`) | bit index (isolate memo; fetched with `cf.cacheEverything`, `cohort-index.ts:31-35`), `bundle_index` rows for the studies on the page (chunks of 20, whole JSON), `pz_chunk_index` (empty: every `chunks` and `data_offset` is null [M]) | none | 1.07 MB for 20 samples [M] | identity only | D1: ceil(studies/20) + ceil(samples/90) |
| REST `/api/cohort/resolve` | bit index only, no D1 | 300 s | | identity | 0 |
| REST `/api/manifest` (`api/manifest.ts`) | search scan + `manifestRows`; ignores `gse=` (the smoke test sends it) [M] | 300 s | 518,607 B for 2,000 rows [M] | identity | D1 heavy (hundreds of rows per study) |
| Catalog crank (`catalog-crank.yml:20`, every 3 h, both repos target `https://singlet.bio/api/ingest`) | `refresh-next` (5 count queries even when idle, one scans ~22k-135k rows per comment `catalog-refresh.ts:199-203`), `index-next` | none | | ingest token | D1 reads dominate |
| Search-index publish (`search-index.yml:24`, nightly) | `wrangler r2 object put` of `index.<ver>.sbix` (3.9 MB) and `index.json` (241 KB) | the only objects with Cache-Control, and still DYNAMIC [M] | | CI token | 2 Class A |
| Bundle publish / repack | `pack_gse` then upload (section 3) | | | | multipart Class A |

### 1.1 Quantities for three typical jobs (R2 ops are Class B unless stated)

| job | today | bytes moved | requests (R2 / Function / D1 statements) |
|---|---|---|---|
| **One sample's matrix by the member index** (MCP `list_bundle_files` once, `get_partial_download`, client fetches exon + intron) | 2 MCP calls, then 2 Range GETs. Typical sample: 25.7 MB all members, at most ~16 MB exon+intron (all `.1pz` are 61% of bytes [W4]) | ~16 MB matrix + ~9 KB index row read server-side + ~2.5 KB tokens to the model (about 600 tokens) | R2: 3 (1 header read by the Function + 2 client). Function: 2 invocations. D1: 2 x (3 metering + 1 read) = 8 statements, ~12 rows read, 4 rows written |
| **A whole study** | 1 GET, single stream at ~32 MB/s [M-parent] | mean 474 MB (15 s), median 181 MB [W4] (5.7 s), max 29 GB (15 min); R timeout 600 s is exceeded above ~19 GB | R2: 1 (+1 HEAD for R). D1: 0 |
| **Cohort of 20 samples over 5 studies** | `/api/cohort/manifest` page, then 20 `get_partial_download` calls (60/min limit), 40 client Range GETs | ~320 MB matrices vs ~2.4 GB for the 5 whole studies (mean 474 MB each): 7.4x less; manifest 1.07 MB today (should be ~30 KB) | R2: 20 header reads + 40 client = 60. Function: 21. D1: 20 x 4 metering + 20 bundle_index row reads (5 distinct rows, 9-185 KB each) + manifest queries = ~100 statements |

D1 budget: a metered MCP call costs 3 statements before it does any work (`withinRate` upsert `spend.ts:325-351`, `recordToolCall` upsert
`:355-364`, `allowanceFor` read via `quotaMeta` `mcp.ts:465-481`), plus a key lookup every 60 s per isolate (`identity.ts:124-168`) and, once per
isolate, `ensureCatalogColumns` (8 ALTER attempts, an index, a probe: `catalog-refresh.ts:25-53`) at the start of every POST (`mcp.ts:989`).
The row-read budget is dominated by aggregate scans (facets, stats, `findMatchedControls` reads up to 400 rows per call,
`mcp-tools.ts:559`), not by member-index reads.

## 2. Failure modes

| limit | where it bites | evidence |
|---|---|---|
| Stale cached offsets | every rewritten bundle: wrong recipe, 502 on inflate (deflate fails on garbage), silent garbage for `stored` entries, wrong `bytes` in catalog/manifest sizes | section 0.1-0.2 |
| No signature/name check on the local header | `readEntryFromSource` (`bundle-reader.ts:282-293`) and `entryRange` (`:309-314`) read `nameLen/extraLen` from whatever sits at the offset | code |
| D1 row limit 2 MB per string/row [D] | `getBundleIndex` swallows a failed write (`bundle-reader.ts:267`); the study is then re-parsed over the network on every call (HEAD + 256 KiB tail + CD, then up to MBs of CD). [W4] largest row in 180 bundles is 185 KB (2,243 members); the biggest atlas studies (hundreds of samples x ~36 members) could pass 2 MB: look for "index too large" parks | `ingest/[table].ts:604-616` |
| D1 daily row reads | free tier: 5M rows/day [?: plan unknown]; outage on 2026-09-03 (`cache.ts:4-8`); `020_backfill_usable.sql` forbids `json_each` over all of `bundle_index` | code, CLAUDE.md |
| D1 queries per invocation | 50 free / 1,000 paid [D]; `compare_studies` of 8 studies issues ~8 x 8 queries (`loadStudy` runs 6-7 queries + QC read) | `study-core.ts:123-195`, `mcp-tools.ts:737-760` |
| Subrequests per invocation | 50 free / 10,000 paid [D]; `readSampleSummaries` issues one per sample; the repo has a test for the literal "Too many subrequests" (`catalog-park.test.ts:55`) | suggests the 50 limit was hit [?] |
| 6 simultaneous connections | `CONCURRENCY = 6` (`bundle-core.ts:193`) | [D] |
| 4 MB inflate | `MAX_INFLATE_BYTES` (`bundle-reader.ts:39`); every `.1pz`, `.mtx`, large `.tsv` returns a recipe | code |
| Isolate memory 128 MB [D] | `fetchRange` accepts status 200 as well as 206 (`bundle-reader.ts:67`): a Range-ignoring front (e.g. a future Cache Rule) would buffer a whole 29 GB object | code |
| Edge cache TTL 24 h with no purge | `cachedJson` has no per-key invalidation (`ingest/[table].ts:43-46`); Cache API is per data center, not tiered | code |
| 512 MB CDN object limit [D] | 15-16% of bundles, 50-52% of bytes | [W4], [M-parent] |
| MCP envelope | body <= 256 KB, batch <= 10, 60 calls/min/key (`mcp.ts:320-324`); responses have no size cap | code |
| Doc drift | README says "most tools need no key" (`README.md` MCP section); `callTool` requires a key for all 14 (`mcp.ts:836-849`) | code |

## 3. How bundles are produced, published and refreshed

| step | what happens | cite |
|---|---|---|
| Pack | `pack_gse` writes one ZIP64 per study with `zipfile.ZipFile(..., allowZip64=True)`; `.1pz` STORED, JSON/TSV DEFLATED; members sorted; `ZipInfo(arc)` without a date, so every entry is dated 1980-01-01; `manifest.json` last, with `created_at` and per-member SHA-256. Each member is read fully into memory (`fh.read()`) | `python/singlet/bundle.py:1055-1160` (1062-1084, 1124-1129) |
| Upload (original, 2026-06-19) | not in any audited repo (the `cf_bundle_v3.py` uploader sits in `singlet-website-anvil`, outside my read scope). Evidence from the objects [W4]: all 75 June multipart objects have ETag part counts consistent with **8 MiB parts** (72 of them fit no other common size; boto3/aws-cli default; GSE138867 is `...-23` for 192,567,529 B = 22.96 x 8 MiB [M]), Content-Type `application/octet-stream`, 11 small ones single-part | measured |
| Upload (migration, 2026-10-20/21) | `boto3 upload_file` with `multipart_threshold = multipart_chunksize = 64 MiB`, **no ContentType, no CacheControl**: 83 of 180 objects now have no Content-Type header, ETag part counts fit 64 MiB (GSE111113 170,777,036 B = `...-3`), 11 small ones are single-part `application/zip` [W4] | `restrict_published_bundles.py:65, 95-97, 128-129` |
| Why ETag looks like `-23` | S3 multipart ETag = MD5 of the part MD5s + `-<parts>`; it changes whenever bytes **or the part size** change, and cannot be recomputed from content without knowing the part size. It is still a valid change detector; it is not a content hash | S3 semantics |
| Versioning / overwrite | none: objects are overwritten in place under one key; no object versioning, no content-addressed keys, no `If-Match` on writes | `restrict_published_bundles.py:278` |
| Migrated local headers | members copied with `force_zip64=True`: local header carries `0xFFFFFFFF` sizes and a 20-byte zip64 extra [M: GSE111113 `extraLen=20`]; offsets and sizes in the central directory are normal. Any reader that trusts the local-header sizes is wrong; the site reads sizes from the central directory, which is right | `_restricted.py:123-134` |
| D1 refresh | `bundle_manifest` rows come from the HPC ingest or `refresh-next` phase (c) and only for studies **without** a manifest (`catalog-refresh.ts:181-183, 288-295`); `bundle_index` rows are written the first time a study is read, and again only by `?refresh=1` with the ingest token (`[action].ts:43`), `POST /api/ingest/index-bundle` (`ingest/[table].ts:274-315`), or `index-next` when a cached summary cannot be read (`:563-578`) | code |
| What re-triggers indexing | `PENDING_SQL` (`ingest/[table].ts:498-502`) selects studies with no `bundle_index` row or no QC rows; a fully indexed study is never revisited | code |
| Never checked anywhere | (a) live size or ETag vs `bundle_index.bytes` before using offsets; (b) PK\3\4 signature and entry name at the offset; (c) `If-Match`/`If-None-Match` on any range read; (d) `bundle_manifest.bytes`, `gse.r2_bundle_bytes` vs the object; (e) `Last-Modified` vs `indexed_at`; (f) any alarm when the migration or a repack changes an object; (g) smoke test touches `data.singlet.bio` only through `/api/bundle/GSE138867/index` containing "exon_counts" (`scripts/smoke.sh`), never a Range read or a size comparison |
| Data identity available today | R2 ETag and Last-Modified (not stored by the site); `manifest.json` `created_at`, `singlet_version`, `reference_build` (stored in `bundle_manifest`); per-member SHA-256 in `manifest.json` (not indexed); CRC-32 in the central directory (`crc32` column in [W4] members, not kept in `bundle_index`) | |

## 4. The MCP surface (14 tools, `mcp.ts:98-317`)

`tools/list` is 15,681 characters (about 4,300 tokens) and `initialize` adds 1,584 characters of instructions, per session [C: computed
from the source with `scratchpad/w4cf/toolsize.mjs`]. `search_datasets` (3,418) and `export_manifest` (2,508) are the heavy schemas
because the 28 tissue / 12 disease / 13 assay enum values (944 characters) are repeated in both.
Tokens below are characters / 3.5 and count text + structuredContent together, because both are returned (`mcp-tools.ts:36-38`).

| tool | inputs | output size from the code path | chatter / low-value text |
|---|---|---|---|
| `search_datasets` | query<=500, level, limit<=50, filters | per study: ~1.2 KB structured (bundle_url, study_url, geo_url, `python` + `r` loader strings, `why`, 12 cell types, groups) + ~0.35 KB text; limit 10 = ~15 KB (4k tokens), limit 50 = ~75 KB (21k) | loaders repeated in every row (`mcp-tools.ts:55-58`, used at `mcp.ts:524`); three URLs per row; text restates the structure |
| `get_study` | gse_id | structured: full abstract + <=60 samples with `characteristics` + conditions + publications (with abstracts) + loaders; text repeats title, abstract (<=1,500 chars), counts. [M] REST analogue 9,960 B for 8 samples; 60 samples ~40 KB (11k tokens) | abstract twice; sample characteristics and conditions overlap; loaders |
| `get_download_url` | gse_id | ~0.9 KB | loaders + curl + text restating them; `bytes` stale for rewritten studies |
| `get_atlas_stats` | none | ~3-4 KB (failure categories, accounting) | `as_of`, rules block |
| `get_sample_qc` | gse_id, gsm_ids | ~1.0 KB per sample (0.8 KB structured with 8 derived duplicate fields: `uniquely_mapped_pct`/`mapping_rate_pct`, `median_mito_fraction`/`median_mito_pct`, `sequencing_saturation`/`_pct`, plus `git_sha`, `wall_seconds`, `matrix_bytes`, `usable`; 0.23 KB text line); +`qc_rules` block each call. 58 samples = ~60 KB (17k tokens) | duplicated fields, the same numbers in text; static `qc_rules`/`thresholds` per call |
| `list_bundle_files` | gse_id, gsm_id | structured = the `/api/bundle/index` body ([M] 60,512 B for 58 samples / 583 files; 8,831 B for 8 samples) + a text line per file (~62 B x 583 = 36 KB). Median study ~106 members; the 2,243-member study = ~190 KB + ~140 KB (95k tokens) | every file of every sample listed in both channels, including the ~34 non-matrix files per sample that nobody asks about |
| `get_modalities` | gse_id, gsm_id | static registry rows (up to 29 x ~260 B = 7.5 KB structured, same again as text) + `per_sample` (0.35 KB x samples) + snippets; 58 samples = ~35 KB (10k tokens) | Python and R one-liner for every modality in both channels; `raw_counts` snippet; per-sample modality arrays; static knowledge that belongs in server instructions or a resource |
| `get_partial_download` | gse_id, gsm_id, file | ~1.5 KB in two copies (url, range, curl, a generic zlib Python snippet, `edge_url`) = ~800 tokens | boilerplate curl/Python identical in every answer; offsets stale for rewritten studies |
| `export_manifest` | query/filters/gse_ids, format | **`manifest` body is in the text (capped 60,000 chars, `MANIFEST_TEXT_CAP`) and again uncapped in `structuredContent.manifest`** (`mcp-tools.ts:423, 477, 493`) plus a `studies` id list; 2,000 studies = ~520 KB per copy [M: REST 518,607 B] = ~300k tokens for both | duplicate body; ids listed a third time; curl/wget/python/R renderings are generic |
| `find_matched_controls` | gse_id, min_samples, same_assay | 10 candidates x ~0.9 KB + 4 constant caveats (~700 chars) + D1 `LIMIT 400` read | caveats and loader per row every call |
| `compare_studies` | 2-8 gse_ids | ~1.3 KB per study x up to 8 + differences; ~8 x 8 D1 queries | text restates each field |
| `assess_study` | gse_id, purpose | ~4-8 KB (conditions, per-sample file names, fit checks, `completeness`) | `per_sample_files` list repeated from `list_bundle_files`; text restates |
| `get_cohort` / `save_cohort` | id, token / name, <=2,000 ids | `JSON.stringify(out)` as text **and** structured (`mcp.ts:824`); a 2,000-study cohort echoes 2,000 ids twice | duplicate |
| every result | | `_meta.quota` (~350-600 chars, includes the 260-char `BILLING_NOTE` on every call, `mcp.ts:465-481`) and one D1 read for it | boilerplate in every response |

Resources `singlet://stats` and `singlet://vocab` run aggregate scans per read (`mcp.ts:422-457`). Prompts embed multi-step scripts.

**Stable identifiers that exist now** (none is exposed as a revalidator): bundle ETag/Last-Modified (R2, not stored); `bundle_index.indexed_at`
and `bytes`; `bundle_manifest.manifest_created_at` (`created_at` in `manifest.json`), `singlet_version`, `reference_build`;
`gse_meta.updated_at`; `sample_qc.updated_at`; bit index `index_version` (`index.json`, e.g. `20261022-675f4275`, checksum `675f4275`) and the cohort
`hash` (`cohort/manifest.ts:42`). An "unchanged since you asked" answer needs one new field: a per-study `rev` (see 6, step 5).

## 5. Cloudflare configuration inventory

| item | what the repo records | what I observed live [M] | status |
|---|---|---|---|
| Pages project `singlet`, bindings | `wrangler.toml:1-31`: D1 `DB` `singlet-catalog` (id `fb42f359-...`), R2 `SINGLET_DATA`=`singlet-data`, `USER_DATA`, `HPC_DASHBOARD`, `[ai]`; no `location_hint`, no routes, no cache config. `singlet-cloud` binds the **same D1 database and the same bucket** (its `wrangler.toml`), and runs its own copy of every function and its own 3-hourly crank against `https://singlet.bio/api/ingest` | | `bundle-reader.ts:6-9` claims the R2 binding is not usable in preview [?] |
| R2 custom domain `data.singlet.bio` | named in code (`study-core.ts:21`), `README.md:32`; the binding step is not in the repo | served by Cloudflare (`server: cloudflare`, ORD colo, TTFB median 0.20 s, p90 0.33 s over 180 bundles [W4]) | custom domain exists [?] bucket location hint ENAM [M-parent] |
| Cache Rules / Cache Everything | `launch-runbook.md:50-53` proposes only a Transform Rule `Cache-Control: public, max-age=3600` for `*.singlet` | no Cache-Control on any `.singlet`; DYNAMIC everywhere | **not applied** (the runbook's rule is not live) |
| Transform rules | runbook: `X-Content-Type-Options`, HSTS, Cache-Control | `X-Content-Type-Options: nosniff` present on bundle responses; no `Strict-Transport-Security`; no Cache-Control | partially applied or an R2 default [?] |
| Tiered Cache / Smart Tiered Cache | not mentioned in the repo | | unknown [?] |
| Cache Reserve | not mentioned | not usable on R2 custom domains [D] | skip |
| CORS | bucket-level R2 policy set by the previous session (`docs/HANDOFF.md:17`: GET/HEAD+Range for singlet.bio, www, the alpha preview, localhost:8080; expose Content-Length, Content-Range, Accept-Ranges, ETag). The runbook's `origins: ["*"]` version (`launch-runbook.md:31-43`) differs | `Origin: https://singlet.bio` gets `Access-Control-Allow-Origin: https://singlet.bio`, `Vary: Origin`, exposes `Content-Length,Content-Range,Accept-Ranges,ETag`; `Origin: https://example.org` gets none [W4]. A request without Origin gets no CORS headers (R2 behaviour), which is why the parent saw none | working for listed origins. **Cache Everything will freeze the first requester's ACAO for the cached copy** unless CORS becomes `*` or a Transform Rule sets it [?] |
| WAF / rate limiting | runbook table (`launch-runbook.md:60-71`, `HANDOFF.md:50-56`): /mcp 120/min, nl-search/explain 30/min, `/api/bundle/` 60/min, `/api/` 600/min per IP; the wrangler login has no zone scope so the owner must create them | python-urllib default UA gets 403 from data.singlet.bio (`AGENT_RULES.md:15`): some bot/WAF rule exists on that host | rules unconfirmed [?] |
| Security headers | `_middleware.ts:10-36` for `/api/*`, `/mcp`, `/auth/*`, `sitemap.xml`; `public/_headers:12-24` (assets immutable); `_redirects:12-13` has `/data/* /data/:splat 200` (a self-rewrite) | | fine |
| Write access | the Pages project binds `SINGLET_DATA` (read/write capable); uploads are done from Clipper with S3 keys (`restrict_published_bundles.py:83-97`) and from CI with `wrangler r2 object put` (`publish.sh:19-20`) | | whether production only, or preview too, is [?] |
| Live objects vs the script | `publish.sh:19` uploads the `.sbix` with `--content-type=application/octet-stream` only | the live `.sbix` has `Cache-Control: public, max-age=31536000, immutable`; `index.json` has `public, max-age=300` (script line 20) | a different script or an in-place metadata edit produced the `.sbix` header [?] |

## 6. Change list for the v2 serving design (ordered; step 0 is a fix, not a feature)

Constraints: `main` auto-deploys and previews share production D1 (`CLAUDE.md`); schema files are idempotent numbered files applied by hand
before the code that uses them; there is a post-deploy `smoke.sh`; both `singlet-website` and `singlet-cloud` carry copies of the same files, so
every serving change lands twice (or the shared code moves to one package first).

### Step 0 (do first, no new format): make the cached index trustworthy and repair it
| what | files / tables | notes |
|---|---|---|
| 0.1 Store identity with each index | new `schema/031_bundle_index_identity.sql`: `ALTER TABLE bundle_index ADD COLUMN etag TEXT; ADD COLUMN last_modified TEXT; ADD COLUMN verified_at TEXT;` (the table itself is created on demand at `bundle-reader.ts:198-210`, not in `schema/`). `parseZipSource` returns the HEAD `etag`/`last-modified`; `getBundleIndex` writes them. Code must tolerate rows without the columns (the `ensureCatalogColumns` pattern) | apply 031 first, deploy second |
| 0.2 Guard on every use, without an extra request | in `readEntryFromSource` and `entryRange`/`entryPayloadRange` (`bundle-reader.ts:282-320`): read 30 + len(path) bytes, require signature `PK\3\4` and that the name equals `entry.p`; on mismatch call `getBundleIndex(..., {refresh:true})` once and retry, then 409/502 with a clear error. Send `If-Match: <etag>` on the Range GET (`fetchRange`, `:65-69`) so a changed object answers 412 for free; remove the `status === 200` acceptance (`:67`) | zero extra requests on the hot path |
| 0.3 Find every stale row | list the bucket (S3 `ListObjectsV2`, 9 Class A ops for 8,202 keys, or `env.SINGLET_DATA.list`) and compare `size`/`etag` with `bundle_index.bytes`/`etag`: `SELECT gse_id, bytes FROM bundle_index` is ~8k rows read. **Never `json_each` over `bundle_index`** (`020_backfill_usable.sql:6-9`). Expected: 3,700-4,900 stale rows [W4 CI] | new token-protected `GET /api/ingest/verify-index?n=50` beside `index-next` (`ingest/[table].ts:521`), or a one-off script run by the owner |
| 0.4 Re-index only the central directory | `?refresh=1` exists (`[action].ts:43`) but `POST /api/ingest/index-bundle` also reads every `summary.json` (`:286-298`), 1 subrequest per sample. Add a light mode that rewrites just `bundle_index` (HEAD + 256 KiB tail; 174 of 180 central directories fit the tail [W4]; ~3 subrequests, 1 D1 write per study; ~4,300 studies = ~13k Class B, ~4,300 rows written) | run in 25-study batches under the crank lock |
| 0.5 Fix catalog sizes | `bundle_manifest.bytes`/`n_files` and `gse.r2_bundle_bytes` via the existing token POST `/api/ingest/bundle_manifest` (`ingest/[table].ts:119-132`) using the listing from 0.3 | restores `get_download_url`, `get_study`, `export_manifest` totals |
| 0.6 Drop poisoned edge entries | bump the cache keys `v=1` -> `v=2` at `[action].ts:60,76` and `path=` -> `v=2&path=` at `:123`; Cache API entries from the old key expire by TTL. No per-key purge exists | one-line change |
| 0.7 Make the migration safe to repeat | in `restrict_published_bundles.py` (singlet-bio, owner's repo): after the verified upload, call the re-index endpoint, set `ContentType` and `CacheControl`, and use 8 MiB parts so ETag semantics match the originals | patch file only: I did not edit it |
| 0.8 Tests (vitest, node env, `createTestD1()` as in `bundle-endpoint.test.ts`) | `bundle-reader-guard.test.ts`: cached row + different HEAD size/etag -> re-index; wrong signature at offset -> one refresh then error; `If-Match` header present and 412 handled; row written without new columns still reads. `catalog-park.test.ts` style test for the verify-index selection | `npm run check` gates the merge |
| 0.9 Smoke additions | `scripts/smoke.sh`: Range-read the first 4 bytes at a recipe start for one rewritten and one original study and require `TP1Z`; compare `bytes` in `/api/bundle/GSE111113/index` with `HEAD` content-length | catches a recurrence within one deploy |
Rollback for step 0: the guard only adds a retry and a column; revert the deploy (Pages rollback) and the columns are harmless. Risk: a re-index storm: cap at the crank lock rate (1 s cooldown, `ingest/[table].ts:473-475`) and 25 per call; D1 cost is ~4,300 row writes.

### Step 1: make data.singlet.bio cacheable (dashboard + two header fixes; no code)
Cache Rule on hostname `data.singlet.bio`: Eligible for cache; Edge TTL: use `Cache-Control` from origin, default 1 day for 2xx; Smart Tiered Cache on (free on all plans [D]); set Content-Type and `Cache-Control: public, max-age=31536000, immutable` on content-addressed objects only (index `.sbix`, and later packs and per-sample objects); keep **old whole-study bundles at a short TTL or bypass** because they are overwritten in place and 16% exceed 512 MB anyway. Make CORS `*` on the bucket, or add a Transform Rule that sets `Access-Control-Allow-Origin: *` after the cache, before turning caching on. Test on a staging hostname first for the full-object-fill-on-miss behaviour of Range requests (parent read the Range docs; I did not).
Rollback: delete the rule; zero code. Risk: caching a bundle that is later overwritten serves old bytes until TTL: this is exactly step 0's problem, so do step 0 and keep the TTL on `*.singlet` at <= 1 h.

### Step 2: per-sample objects under 512 MB (the layout change)
Write each sample's members as objects `data.singlet.bio/s/<GSM>/<member>` (or one object per `.1pz`; sample median 25.7 MB, max 131.6 MB [W4]), with `ETag`, `Cache-Control: immutable`, key = URL, no ZIP offset arithmetic. Leave `data/<GSE>/<GSE>.singlet` untouched as the compatibility copy; assemble a ZIP64 "view" in a Function only if the old bundles are later retired (the members are STORED, CRC-32 and sizes are in the central directory, so a streaming assembler needs no recompression).
| touch point | change |
|---|---|
| producer | new `scripts/` copy job (reads own bundle by range, PUT per member; ~78k samples x ~36 members = ~2.8M PUT = ~$13 Class A if per member, ~78k PUT if per sample). Add the writer to `pack_gse` (`bundle.py:1055-1160`) for new studies |
| D1 | new `schema/032_sample_objects.sql`: `sample_object(gsm_id, gse_id, member, key, bytes, etag, crc32, rev, PRIMARY KEY(gsm_id, member))` or no table at all if the static index (step 3) carries it |
| `mcp-tools.ts:getPartialDownload` | answer `{url:"https://data.singlet.bio/s/<GSM>/exon_counts.1pz", etag, bytes}` with no range, no inflate recipe |
| `[action].ts` entry route | for <=4 MB members redirect/serve the object; recipes disappear |
| clients | new Python/R paths fetch members with parallel ranges; old clients keep using the ZIP |
Rollback: stop writing; objects are additive. Cost: doubles storage until old bundles are retired (+~$58/month at today's 3.89 TB).

### Step 3: compact member and chunk index as a static cached object
`data.singlet.bio/idx/v2/<GSE>.<digest>.idx` (immutable, content-addressed) holding per-sample member table plus chunk offsets (what `pz_chunk_index` was meant to hold; it is empty [M]); D1 keeps only `(gse_id, idx_digest, etag, bytes)`. `loadEntries` and `loadChunkRows` (`cohort-manifest.ts:357-399`) and `getBundleIndex` read the static object through `fetch(..., {cf:{cacheEverything:true, cacheTtl}})` as `cohort-index.ts:31-35` already does for the bit index. Touch: `bundle-reader.ts`, `cohort-manifest.ts`, `api/cohort/manifest.ts` (also stop repeating `downsample.dropped` on every page: return its count and a digest, paged on request), new `schema/033_bundle_index_static.sql` (`ALTER TABLE bundle_index ADD COLUMN idx_key TEXT`). Do not drop `pz_chunk_index`. Tests: round-trip of the index codec; cohort manifest page <= 100 KB for 500 samples; stale-digest guard.

### Step 4: content-addressed side-information packs
`data.singlet.bio/ref/<id>.pack` (vocabulary, feature names, derivation recipes) per `REPORT.md` container v2; the website only carries pack ids and decode recipes. Touch: `mcp-tools.ts` (`getModalities`, `getPartialDownload` return `pack` ids), `study-core.ts` (reference build -> pack id), client loaders. Depends on the container v2 spec; do not start before the format is frozen.

### Step 5: MCP diet and "unchanged" answers (can start right after step 0; needs no new format)
1. `toolResult` (`mcp-tools.ts:36-38`): send either a short text or the structure, not both. Keep a 1-2 line text summary and make `structuredContent` the data (est. -40% to -50% tokens on QC/file/modality tools, -50% on `export_manifest`, which today sends the 520 KB TSV twice).
2. Drop per-row loader strings (`LOADERS`, `mcp-tools.ts:55-58`; `mcp.ts:524`), the three URLs per search row (derive from the id), derived duplicate QC fields, the per-call `qc_rules`, caveats and `BILLING_NOTE` (`mcp.ts:465-481`; skip the `allowanceFor` read unless the tool is metered).
3. `list_bundle_files` / `get_modalities`: group by file kind and report counts, return the full list only on `gsm_id` or `detail=true`; move the modality snippets into `instructions` or a resource.
4. Add `rev` per study (hash of bundle ETag, `gse_meta.updated_at`, max `sample_qc.updated_at`) to `get_study`, `list_bundle_files`, `get_sample_qc`, `get_partial_download`; accept optional `known_rev` and answer `{unchanged:true, rev}`. D1: reads the same rows, no new table; the ETag half needs step 0.1.
5. `tools/list` 15.7 KB: share the three enums by `$ref`/`$defs` (-944 chars once) and shorten `search_datasets` and `export_manifest` descriptions.
Tests: golden-size tests for each tool on a fixture (`mcp-auth.test.ts` / `modalities.test.ts` style), `known_rev` round trip. Rollback: output shapes change, so version the tool results (`SERVER_INFO.version`) and keep the old shape behind a `verbose` argument for one release.

### Step 6: clients (separate repository, `singlet-bio/python`, `r/`)
`download()` (`_loader.py:159-282`): HEAD + `If-None-Match`, parallel Range parts (8-16 x 8 MiB), resume of the `.part` file, retry with backoff; skip re-hashing every member (`_integrity.py:180-188`) when the ETag matches; `load("GSM")` fetches that sample's members, not the study; R: raise the timeout from a size estimate, not `max(600, ...)` (`load.R:325`).

### Rollout order (safe with auto-deploy, shared D1, small D1 read budget)
0.1 apply `031` on production D1 (idempotent) -> 0.2/0.6 guard + key bump in a PR (preview shares prod D1, so run `smoke.sh` against the preview) -> merge -> 0.3 list and compare (read-only) -> 0.4 re-index in 25-study batches off-peak, watching D1 rows written (~4,300) -> 0.5 repair catalog sizes -> 0.9 smoke additions -> step 5 MCP diet (no storage change) -> step 1 Cache Rule + CORS `*` on a staging hostname, then production -> steps 2-3 behind a feature flag in `wrangler.toml [vars]` -> step 4 after the format freeze -> step 6 clients. Mirror each PR into `singlet-cloud`.

## 7. Cost table (ranges only)

Today: 3.89 TB x $0.015/GB-month = **$58/month** storage; egress $0. Request volume is **not recoverable from code**: no download counter exists (browser, Python and R fetch from R2 directly; no Function logs them), `/api/stats` has no traffic fields. Read R2 metrics (Class A/B by operation) and zone Analytics for `data.singlet.bio` before trusting any request figure. Class B at $0.36/M means 100M reads/month = $36, so ops are small next to storage; the benefits of caching are latency, throughput and Function load.

| lever | storage / month | ops | notes |
|---|---|---|---|
| Recode to container v2, 2.1x-4.5x smaller (REPORT.md) | $13-$28 (saves $30-$45) | fewer bytes per fetch; no Class A change | one-off rewrite of 8,202 objects = ~8,202 x (parts) Class A, < $5; old copies cost another $58 until deleted |
| Payload only, -15% | $50 (saves $9) | | the -15% is the best bit-exact coder, projected |
| Per-sample objects, same bytes | $58 (+$58 while both exist) | ~78k-2.8M Class A one-off ($0.35-$13) | one-time |
| Cache Rules + Smart Tiered Cache [D] | $0 | Class B falls by the hit ratio; objects >512 MB never hit | free on all plans [D]; value is mostly latency |
| Cache Reserve | not applicable to this R2 domain [D] | $0.015/GB-month, A $4.50/M, B $0.36/M if it were | skip |
| Non-North-America users | | R2 location hint is fixed at bucket creation (ENAM [M-parent]); a second bucket with another hint costs a full copy of what you replicate; per-sample objects + Cache Everything + Smart Tiered Cache put hot objects at the user's edge without a copy | the data-location page was not fetched; verify [?] |

## 8. Checklist for the owner (dashboard-only; none can be read from code)

1. Workers plan for Pages Functions: Free or Paid? (subrequests 50 vs 10,000, CPU 10 ms vs 30 s [D]; `catalog-park.test.ts:55` hints the 50 limit was hit.) D1 plan and today's rows-read/written; was the 2026-09-03 outage on the 5M/day free cap?
2. R2 -> `singlet-data` -> Settings: custom domains (is `data.singlet.bio` the only one?), the bucket location hint, public `r2.dev` URL on/off, CORS policy as stored (`wrangler r2 bucket cors list` output), any lifecycle rule, object versioning.
3. Zone `singlet.bio` -> Caching: Cache Rules (any "Bypass cache" on `data.singlet.bio`?), Page Rules, Tiered Cache / Smart Tiered Cache on or off, Cache Reserve on or off, Development Mode (it forces DYNAMIC), "Browser Cache TTL", "Origin Range Requests".
4. Rules -> Transform: the response-header rule from `launch-runbook.md:50-53` (what is live: `nosniff` yes, HSTS no, Cache-Control no).
5. Security -> WAF: custom rules and rate limiting (the four runbook rules); the rule that returns 403 to the default Python-urllib User-Agent on `data.singlet.bio`; Bot Fight Mode; whether rate-limit rules also see Pages Function subrequests to `data.singlet.bio`.
6. Pages -> singlet -> Settings: production vs preview bindings (is `SINGLET_DATA` bound in preview, as `bundle-reader.ts:6-9` denies?), Preview environment variables, `INGEST_TOKEN_SHA256`, `ANON_IP_SALT`.
7. Analytics: R2 metrics (Class A/B ops by operation, egress), Analytics for `data.singlet.bio` (requests, cache status, top paths, countries, bytes), Pages Functions requests and CPU, D1 rows read/written per day and the biggest queries.
8. Is the previous session's R2 CORS the allow-list in `HANDOFF.md:17` or the `*` in `launch-runbook.md:35`?
9. Who uploaded the live `.sbix` with an immutable Cache-Control (not what `publish.sh:19` does), and which script produced the June 8 MiB-part uploads (`singlet-website-anvil/scripts/anvil/cf_bundle_v3.py` is outside my read scope).
10. After step 0: confirm the number of stale rows with the listing comparison and that `restricted.ts` (referenced by `_restricted.py` docstring) exists on `main`.

## 9. Evidence and reproduction

Live requests (all GET/HEAD, UA `singlet-w4-cf-audit/1`): data.singlet.bio: `search-index/v1/index.json` GET + HEAD, `.sbix` HEAD, GSE138867 bundle HEAD and 2 Range GETs, GSE111113 two Range GETs (local header at 5,703,179; bytes at the recipe start 5,745,986). singlet.bio: `/api/bundle/<gse>/index` for GSE111113, GSE148676, GSE126310, GSE130184, GSE180319, GSE125788, GSE116256, GSE121080, GSE138867; `/entry` for GSE111113 (recipe) and GSE138867 `study_meta.json`; `/api/bundle/GSE138867/samples`; `/api/gse/GSE111113`, `/api/gse/GSE138867`; `/api/cohort/manifest?organism=Mus musculus&max_cells=10000&limit=20`; `/api/manifest?gse=GSE138867&format=tsv`. Raw responses and the parsing snippets are in `C:\Users\zacha\AppData\Local\Temp\claude\C--Users-zacha-Projects-singlet-bio\8e8b9acd-9e9b-409a-a4ac-d2afe710d803\scratchpad\w4cf\`.
Reusable analysis of [W4] data (ETag part-size inference, rewrite share, per-sample size distribution, index-row sizes) is in the note's tables; the `python -P` snippets are short enough to redo from `results/w4_atlas_bundles_raw.tsv` and `results/w4_atlas_members_raw.tsv`.
Not done: Cloudflare R2 data-location page, Range-request caching page, D1 pricing page (D1 free allowances above are from memory: 5M rows read and 100k rows written per day on Free); no estimate of real download volume; singlet-website-anvil and any uploader scripts were out of scope.
