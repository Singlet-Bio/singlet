# m_w5: serving-layer work (W5), change sets for singlet-website

Worktree: `C:\Users\zacha\Projects\singlet-website-v2`, branch `feat/singlet-v2` (from origin/main 0dcf4914a). Nothing committed, pushed or
deployed by W5. Change sets are meant to become separate PRs, in this order: CS1, CS2, CS3, CS4, CS5. Files shared between sets are marked
(shared); apply the sets in order and the later diff is the incremental one.

Status (final for this pass):

| set | state | new tests |
|---|---|---|
| CS1 stale-index guard | done, `npm run check` green | 31 |
| CS2 data Worker `workers/data-v2` | done, verified under `wrangler dev --local`; needs the owner's deploy and a route check | 113 |
| CS3 D1 v2 schema + ingest | done; schema 032 is what W6 posts against (contract below) | 20 |
| CS4 API + MCP on v2 | done: v2 branches with legacy fallback, 3 new tools, diet, rev/known_rev, cohort and export manifests | 127 (serve 14, cells 49, packs 29, mcp 26, api 9) |
| CS5 website panel + docs | done (download panel, file tree, types/client, Docs, DocsMcp) | 8 |

`npm run check` (vitest + strict `tsc -p tsconfig.functions.json` + `wrangler pages functions build` + `tsc -p workers/data-v2`): 42 files, 624 tests,
all green (325 before W5). `npm run build` (vite) green. The Functions bundle is 744 KB uncompressed (`fzstd` adds about 8 KB).

---------------------------------------------------------------------------------------------------------------------------------

## Contract for W6 (campaign): what to POST to the website

`CATALOG_SCHEMA` in `scripts/v2_campaign.py` guessed a row-per-table interface (`POST /api/ingest/<table>` with `{rows}`) and column names that
differ from the ones below. The implemented interface is manifest based, as the brief asked, and is one call per study (or per slice of a
study). Change `catalog-post` to this and drop `CATALOG_SCHEMA`/`catalog-sql` row building (the SQL files are not needed: the endpoint upserts).

```
POST https://singlet.bio/api/ingest/v2-study            header X-Ingest-Token, Content-Type application/json, a non-default User-Agent
{
  "manifest": { ...the v2/m/<GSE>.json object WITHOUT "study_files" (the server ignores it and the body limit is 8 MiB)... },
  "manifest_bytes": 123456, "manifest_sha256": "<64 hex>",             // size/hash of the uploaded manifest object (optional)
  "final": true,                                                         // optional, default true
  "migration": { "status": "verified", "legacy_key": "...", "legacy_etag": "\"...-3\"", "legacy_bytes": 187066350, "v2_bytes": 123456789,
                 "n_samples": 12, "members_by_kind": {"matrix": 24, "recipe": 90, "stored": 100}, "verify_ok": true,
                 "commit_sha": "...", "migrated_at": "...", "verified_at": "...", "ledger_key": "v2/ledger/GSE1.jsonl", "error": null }
}
POST same URL, state only:  { "gse_id": "GSE1", "migration": { "status": "running" | "failed", "error": "..." } }
GET  same URL ?gse=GSE1 (token): { study, migration, sample_rows_by_rev, complete }        // verification after a post
```

* `manifest` is validated, not trusted: `format == "singlet-v2"`, `manifest_version == 1`, `gse`, a 16-hex `rev`, every sample `key` must be
  `v2/s/<its GSM>/<32 hex>.s2`, `members` with integer `kind` (1-5), `off`, `len` (the real manifest member dicts from `migrate._manifest_member`
  pass unchanged: `chunk_index`, `n_chunks`, `codec`, `names`, `recipe`... are stored as given). 422 if a sample lists a restricted member
  (`donor/snp_*`, `ase_counts.tsv`, `mt/*` other than `mt_summary.tsv`, vcf...): such a member must not be in a public object at all.
* At most 400 samples per post. A larger study is sent in slices: every slice carries the same `rev`, `n_samples` (the study total) and `bytes`
  (total over all samples) and `final: false`; the last slice has `final: true` (or omits it). Sample rows are written by every slice; the
  `v2_study` row and the removal of rows of an older `rev` only by the final one.
* Idempotent. A study is served by the site from v2 once `v2_study` exists and its `v2_migration` row (if any) says `verified` or `retired`.
  Post `verified` only after the uploaded objects were re-read and compared (format-v2 section 10); until then post `uploaded`/`running`.
* Order for one study: upload objects and manifest to R2, verify, then this POST. Limits: the Worker is subject to the D1 budget; 400 samples per
  post is about 800 row writes.
* Columns (schema/032_v2_catalog.sql): `v2_study(gse_id PK, manifest_key, rev, bytes, n_samples, created_at, legacy_bytes, manifest_bytes,
  manifest_sha256, n_cells_called, reference_build, packs_json, pipeline_json, updated_at)`; `v2_sample(gse_id, gsm_id, key, bytes, sha256, n_columns,
  n_called, n_members, median_umi, median_genes, modalities, members_json, cell_stats_json, rev, updated_at, PRIMARY KEY (gse_id, gsm_id))`;
  `v2_migration(gse_id PK, status, legacy_key, legacy_etag, legacy_bytes, v2_bytes, n_samples, members_by_kind, verify_ok, commit_sha, migrated_at,
  verified_at, ledger_key, error, updated_at)`.
  **Deviation from the brief:** `v2_sample` is keyed `(gse_id, gsm_id)`, not `gsm_id` alone, because a SuperSeries and its SubSeries pack the same GSM
  (index-next already special-cases this for sample_qc) and their manifests can point at different objects for it. A secondary index serves
  lookups by `gsm_id`. One-line change if the orchestrator wants `gsm_id` alone.

---------------------------------------------------------------------------------------------------------------------------------

## CS1: stale-index guard (ship first)

What it fixes: `bundle_index` rows outlive the object they describe (the 2026-10-20/21 migration overwrote about half the atlas in place);
`getBundleIndex` returned them without any check, so recipes pointed into other bytes (GSE111113: 42,723 bytes late) and `/entry` answered 502.

| change | where |
|---|---|
| Identity stored with each row: `etag`, `last_modified`, `verified_at`. New schema file; the code works without them: the first statement that finds a column missing is remembered for a minute, during which reads select fewer columns and writes use the old INSERT, and the If-Match half is simply off; nothing is probed or created up front, so a read of a stored row is one statement | `schema/031_bundle_index_identity.sql`, `functions/_shared/bundle-reader.ts` |
| Every ranged read sends `If-Match: <etag>` (stored, or pinned from the HEAD that started the read); 412 and 416 are `StaleBundleError`; **only 206 is accepted for a range** (a 200 is refused with its body dropped unread; it used to be buffered, up to 29 GB); weak ETags are never sent | `bundle-reader.ts` (`fetchRangeWithEtag`) |
| Every entry read checks the zip local header at the offset: signature `PK\3\4` and that the name equals the entry's (reads 30 bytes + name, no extra request); mismatch is `StaleBundleError`. A local extra field longer than the read slack gets one more exact read | `bundle-reader.ts` (`payloadOffset`, `readEntryFromSource`, `entryRange`, `entryPayloadRange`) |
| `withIndexRetry`: on `StaleBundleError` re-read the central directory once and run again; a second failure is rethrown. Concurrent re-reads of one study share one read (isolate-level). Used by the `/entry` route, MCP `get_partial_download`, `loadSampleQc`, `index-next` | `bundle-reader.ts`, `functions/api/bundle/[gse]/[action].ts`, `functions/_shared/mcp-tools.ts`, `functions/_shared/bundle-core.ts`, `functions/api/ingest/[table].ts` |
| `getBundleIndex(..., {verify:true})`: one HEAD compares size (and ETag if stored) with the row and re-reads on a difference; a failed HEAD leaves the row. Used by `/api/bundle/:gse/index` (edge-cached 1 day) and MCP `list_bundle_files`, so the size and listing they report are the file's now | same |
| Recipes (`/entry` over 4 MB, `get_partial_download`) and `/index` now carry `etag`; the curl/python snippets send `If-Match` so a client's read fails with 412 instead of returning other bytes after the file is replaced again | `[action].ts`, `mcp-tools.ts`, `bundle-core.ts` |
| `/entry` answers 409 with a clear message (not a cached 200, not a bare 502) when the index and the file still disagree after one refresh | `[action].ts` |
| Edge-cache keys: `index` v=2 -> **v=3** (main was already v=2, from PR #47), `samples` v=1 -> v=2, `entry` `path=` -> `v=2&path=`. Old entries expire by TTL (1 day) | `[action].ts` |
| `readEntry`/`readEntryText` take the index's etag; `parseZipSource` returns `etag`, `last_modified` | `bundle-reader.ts`, `bundle-core.ts`, `catalog-refresh.ts` |
| smoke.sh: for GSE111113 (rewritten in place) and GSE138867 (original): index `bytes` equals the HEAD content-length, and the first 4 bytes of the exon matrix at the offset the API reports are `TP1Z`. Run against production today it FAILS for GSE111113 (index 324,676,242 vs object 170,777,036; reads garbage) and passes for GSE138867, which is the point | `scripts/smoke.sh` |

Files (CS1): `schema/031_bundle_index_identity.sql`, `functions/_shared/bundle-reader.ts`, `functions/_shared/bundle-core.ts`,
`functions/_shared/catalog-refresh.ts` (1 line), `functions/_shared/mcp-tools.ts` (shared with CS4: `getPartialDownload`, `listBundleFiles`),
`functions/api/bundle/[gse]/[action].ts`, `functions/api/ingest/[table].ts` (index-next only), `scripts/smoke.sh` (shared with CS4: the tool count),
`src/test/bundle-reader-guard.test.ts` (31 tests), `src/test/helpers/zip.ts` (zip writer and fake object host).

Verified live (read-only GETs): `data.singlet.bio` honours `If-Match` on ranged reads (match -> 206, mismatch -> 412, weak validator -> 412,
past the end -> 416), so the guard works on the R2 custom domain as it is. With the new code, `entryRange` on live GSE111113 returns header
5,703,179, payload 5,703,263 and `TP1Z`, and a recipe 42,723 bytes late is rejected as stale.

Not in CS1 (by instruction or by size): repair of existing rows (another agent), `verify-index`/light re-index endpoint (w4_cf 0.3/0.4), catalog
size repair (0.5), the `restrict_published_bundles.py` patch (0.7). Residual risk: a recipe cached at the edge for a day (`/entry` over 4 MB) can
still go stale if the file is overwritten again within that day; the recipe now carries the ETag so a client's own read fails safely.
`/api/cohort/manifest` still reads `bundle_index` rows without a guard (it never fetches); CS4 replaces that path for migrated studies.

Owner: apply `schema/031_bundle_index_identity.sql` to production D1 (`wrangler d1 execute singlet-catalog --remote --file=schema/031_bundle_index_identity.sql`)
BEFORE merging the CS1 PR (it is optional for correctness, required for the If-Match half). Run `scripts/smoke.sh <preview-url>` before merging.
After merge, the first read of any stale study refreshes its row by itself; the repair of the other rows is the separate SQL/endpoint work.

---------------------------------------------------------------------------------------------------------------------------------

## CS2: data Worker `workers/data-v2/`

Files: `workers/data-v2/{wrangler.toml,tsconfig.json,README.md,.gitignore}`, `workers/data-v2/src/handler.ts` (all behaviour), `src/index.ts` (entry),
`src/test/data-v2-worker.test.ts` (113 tests, node env, fakes for R2 and the Cache API), `package.json` (the `check` script gains
` && tsc -p workers/data-v2/tsconfig.json --noEmit`; the only change).

Behaviour (full table in the README): GET/HEAD/OPTIONS on `v2/s/<GSM>/<32hex>.s2`, `v2/ref/<32hex>.pack`, `v2/ref/index.json`, `v2/m/<GSE>.json`,
`v2/cells/<GSE>.cells`; everything else 404/405. Single `Range` (206 + Content-Range, 416 + `bytes */size`; multi-range and malformed ranges are
ignored and the object is streamed whole), `ETag`, `If-Match` (412), `If-None-Match` (304), `If-Range` (ETag form), CORS `*` with exposed headers,
`Cache-Control` immutable (1 year) for s/ and ref/<id>.pack and `max-age=300` for m/, cells/ and ref/index.json, correct Content-Type, `no-store`
on errors. Edge cache = Cache API keyed by the canonical URL without query or Range: a hit serves any Range from the cached full object; a miss
without Range streams from R2 and fills the cache (tee up to 8 MiB, a second R2 read above that); a miss with Range serves the range from R2 and
fills the cache in the background (`waitUntil`, one fill per key per isolate); objects over 500 MiB are never cached. Costs one extra Class B
read per cold object per data centre.
Established with a probe Worker on workerd (local) and the docs: `cache.match` slices a cached 200 for a single valid Range (206 with
Content-Range); an unsatisfiable range gives a bare 416 without ETag (the handler re-matches without Range and decides itself); `If-Match` and
`If-Range` are ignored by the Cache API (the handler evaluates every conditional itself); a HEAD never matches; `put` rejects a 206.
Verification: `npx wrangler dev --local --port 8799 --compatibility-date 2026-06-24` (the installed wrangler 4.102 refuses the pinned
`2026-10-01`; the code uses nothing date dependent), seeded local R2 with `wrangler r2 object put ... --local`: cold GET MISS then HIT with the same
ETag, Range on a hit 206, clamp/suffix/open-ended, 416, If-None-Match 304, If-Match 412, If-Range, HEAD, OPTIONS 204, POST 405, `..`/`//`/`%2f`
404; a Range on a cold 12 MiB object returned 206 from R2 and the same Range was a HIT two seconds later (background fill), full GET byte
identical. `npx wrangler deploy --dry-run` bundles at 17 KiB.
Reviewed by W5 after the fork finished (handler read in full): no change made. Open point noted by the fork: Cloudflare may compress `.json` on the
fly and weaken its ETag; only manifests are affected and nothing sends If-Match for them.

Owner: (1) check the compatibility date: the installed wrangler rejects 2026-10-01 locally but deploys of newer wrangler accept it; use
`2026-06-24` if `wrangler deploy` complains. (2) `cd workers/data-v2 && npx wrangler deploy` (account `2f8a8cc95b6bc0be84ac611fe2e5c3c2`).
(3) **Route vs R2 custom domain**: the docs do not say whether a Workers route on `data.singlet.bio/v2/*` wins over the R2 custom-domain binding.
After deploy `curl -sI https://data.singlet.bio/v2/ref/index.json` must show `x-served-by: data-v2`, and `/data/...` URLs must be unchanged. If the
route does not win: remove it and use `routes = [{ pattern = "dv2.singlet.bio", custom_domain = true }]`, then clients use `https://dv2.singlet.bio/v2/`
(the website reads the origin from the Pages variable `V2_ORIGIN`, see `functions/_shared/v2-urls.ts`, added in CS4). (4) Rollback = remove the route; the
Worker never writes to the bucket. (5) Nothing needs a dashboard Cache Rule.

---------------------------------------------------------------------------------------------------------------------------------

## CS3: D1 schema for v2 + ingest endpoint

Files: `schema/032_v2_catalog.sql`, `functions/_shared/v2-catalog.ts` (validation, upsert statements, typed reads), `functions/api/ingest/v2-study.ts`
(POST/GET/OPTIONS, a static sibling of `[table].ts`), `src/test/v2-catalog.test.ts` (20 tests, real SQL through node:sqlite and the real schema file).
Contract and columns: see the section "Contract for W6" above. Read helpers: `getV2State(db, gse)` (two primary-key reads; null unless served from v2),
`v2ServedStudies(db, gses)` (chunks of 90), `getV2Sample`, `getV2Samples` (<= 50 with members), `listV2Samples(db, gse, rev, {limit<=500, after})`
(no JSON columns; keyed by the primary key prefix and `rev`), `v2Available(db)` (table probe: yes forever, no for a minute). Read-budget test:
EXPLAIN QUERY PLAN shows keyed SEARCH for every read and for the stale-row delete; the list query selects no JSON column; there is no
`json_each` anywhere.
Decisions: modalities are computed at ingest from the public member names with the existing registry (`resolveModalities`) and stored as a small
JSON array, so study-level modality questions read short rows; `median_umi`/`median_genes` are typed columns so depth filters run in SQL; the
member dicts are stored whole (`members_json` <= 128 KB per sample) and read one sample at a time.
Owner: apply `schema/032_v2_catalog.sql` to production D1 (idempotent) before CS4 merges. The ingest token is the existing one
(`INGEST_TOKEN_SHA256`).

---------------------------------------------------------------------------------------------------------------------------------

## CS4: API + MCP on v2, with legacy fallback

Rule: a study is **served from v2** when it has a `v2_study` row and its `v2_migration` row (if any) says `verified` or `retired`; no migration row means
served (the publisher only posts a manifest after it verified). `running`/`uploaded`/`failed`/`pending` keep the legacy answer. One call: `getV2State`
(two keyed reads, no probe; before schema 032 the first read finds no table and the next minute is answered without touching D1).

| area | v2 behaviour (legacy path unchanged) |
|---|---|
| `get_download_url` | `format: "singlet-v2"`, `rev`, `manifest_url`, n_samples, v2 bytes, pack ids, `legacy` block (url, bytes) until status is `retired`, `curl -fLO <manifest>`; `include_samples`+`limit<=200`+`after` pages the sample object URLs |
| `get_partial_download` | the member's absolute range inside the immutable sample object (`url`, `range`, `bytes`), kind, codec id/name, chunk facts, pack refs (`names`), `curl`, no If-Match (content addressed); kind 4 members return `range: null` and the recipe; `slice_with` points at `slice_cells` for matrices |
| `list_bundle_files` | grouped summary by default (file kinds of one representative sample, light per-sample rows, `next_after`), `gsm_id` gives that sample's members with ranges, `detail=true` pages 25 samples with their members |
| `get_modalities` | from `v2_sample.modalities` (a short JSON array computed at ingest), no member list read |
| `export_manifest` / `/api/manifest` | rows gain `format`, `v2_manifest_url`, `v2_rev`, `v2_bytes`; `download_url` and `bytes` stay the legacy file until it is retired, then the manifest URL and v2 size; curl lines for a retired study fetch the manifest; TSV gets three columns appended (`format`, `v2_manifest_url`, `v2_rev`) |
| `/api/cohort/manifest` | a v2 study is planned from `v2_sample.members_json` (no legacy index, no `pz_chunk_index`): `studies[].format/rev/url(manifest)`, `samples[].object`, `files[]` with absolute ranges, `chunk_index.range`, `chunk_cols`, `n_chunks`; column keep-lists unchanged. Reads <= 50 samples per query |
| `GET /api/gse/:id`, `get_study`, `assess_study` | `series.v2 {format, rev, manifest_url, n_samples, bytes, legacy_available}`; `bundle_url` is null once retired; the file counts as present for completeness; `assess_study` describes the v2 files |
| `GET /api/bundle/:gse/index` | same shape plus `format`, `rev`, `manifest_url`, `files_truncated`; members listed for the first 100 samples |
| `GET /api/bundle/:gse/entry` | v2: where the member's bytes are (`url`, `range`, `method` pz2/zstd/stored/derived), never inflates |
| `find_cells` (new) | D1 picks samples (`v2_sample` joined to `v2_study` at the current rev, `gsm` for organism/tissue/disease/assay, `sample_qc` usable and tier), then each cells table is read by Range (one read when the table is under 1 MiB; directory + per-column reads otherwise) and filtered in the Worker. `cell_filter`: `called_only` (default true), `min/max_umi` (exon+intron), `min/max_features` (larger of exon, intron genes), `where` on any numeric/bool column. Returns `[start,length]` column runs (<= 100 per sample, `runs_truncated`), exact `n_match`, `next_after`. Bounds: 25 samples, 40 reads of the data host, 6 at a time; stops with `budget_exhausted` and says where to continue |
| `slice_cells` (new) | reads header + chunk index of each matrix (one ranged read), returns the prefix ranges (header, chunk index, tables, rows) and the chunk-run ranges for the named cells (`runs`, `indices`, `range`, none = all), `columns_wanted/covered`, `overread_ratio`; joins runs to stay within `max_ranges` (default 64). Both matrices by default |
| `resolve_features` (new) | `pack` id (16-32 hex) or `gse_id+gsm_id+file(+axis)` -> names for up to 500 indices from `v2/ref/index.json` + the pack file (vocabulary SVP0/SVPD/composite, or whitelist barcodes) |
| rev / known_rev | `rev` on `get_study`, `get_sample_qc`, `list_bundle_files`, `get_modalities`, `get_partial_download`; find/slice/resolve results carry the study `rev`. v2 file rev = manifest `rev`; legacy file rev = digest of the bundle ETag (size before schema 031). `get_study` and `get_sample_qc` combine the file rev with catalog timestamps (`gse.last_updated`, newest `gsm.last_updated`, counts; `sample_qc.updated_at`). `known_rev` equal -> `{gse_id, rev, unchanged: true}` |
| diet | tool results keep their data in `structuredContent` and the text becomes the first paragraph of the old text (<= 700 chars) plus a pointer; `verbose: true` restores the old double shape. One switch, `DEFAULT_VERBOSE` in `mcp-core.ts`. Errors are untouched. Server version 1.1.0 |
| `tools/list` | 17 tools, every one with `verbose`; `known_rev` where it applies. **23.6 KB** (was 15.7 KB; the three new tools are 4.4 KB). The shared-enum saving of w4_cf step 5.5 was not done |

Decisions worth a second look:
* **The diet is a behaviour change for clients that read only `content[0].text`**: they now get a short summary and a pointer. If a target client ignores
  `structuredContent`, flip `DEFAULT_VERBOSE` to true (one line) or have the model pass `verbose: true`; the summary says so.
* `format-v2.md` section 4.1 still describes a 160-byte pz2 header; the code (`include/singlet/pz2/detail/container.h`, `python/singlet/v2/codec.py`) writes a
  **256-byte header with a CRC-32 at 252, sections at 72, total_len at 184**. The reader here (`v2-serve.ts parsePz2Header`) follows the code; fixtures from the
  current Python caught this. Update the spec text.
* The cells table is read with `fzstd` (pure JavaScript, MIT, 0.1.1, added to `package.json`), because the Worker runtime has no zstd. Decoding a 100k-row u32
  column takes about 22 ms warm on node; `find_cells` therefore needs a Workers plan with real CPU time (Paid; the free 10 ms would not do beyond a handful of small
  samples). Frames must carry their content size (the writer sets it); a column over 64 MiB or a table over 50M rows is refused.
* Objects are fetched from the public data host with Range GETs (not the `SINGLET_DATA` binding), like the legacy code, so previews work. If the data Worker
  route does not win over the R2 custom domain, set the Pages variable `V2_ORIGIN` (e.g. `https://dv2.singlet.bio`); every v2 URL in the API comes from it
  (`functions/_shared/v2-urls.ts`).
* The manifest member dict carries `chunk_index: [absolute off, len]` but not the chunk-section offset, so `slice_cells` reads the member header (256 bytes, one
  request together with the chunk index in files written so far). Adding `"chunks": [abs off, len]` to `_manifest_member` would let clients skip that read (W2).
* `v2_sample.modalities` and `median_umi/median_genes` are computed at ingest so searches do not parse member lists.

Files (CS4): new `functions/_shared/{mcp-core,mcp-v2,rev,v2-serve,v2-urls,v2-cells,v2-packs}.ts`; changed `functions/mcp.ts`, `functions/_shared/{mcp-tools,study-core,bundle-core,manifest-core,cohort-manifest,env}.ts`
(`mcp-tools.ts` and `bundle-core.ts` are shared with CS1), `functions/api/bundle/[gse]/[action].ts` (shared with CS1), `functions/api/cohort/manifest.ts`,
`functions/api/manifest.ts`, `functions/api/gse/[id].ts`, `package.json` + `package-lock.json` (fzstd; `package.json` also carries CS2's `check` script line),
`scripts/smoke.sh` (17 tools; shared with CS1), `README.md`; tests `src/test/{mcp-v2,v2-api,v2-serve,v2-cells,v2-packs}.test.ts` and
`src/test/fixtures/v2/**` (a real `.s2` sample object, cells tables of every dtype and encoding, pack files, all written by the Python reference with the generator
scripts `make_fixtures.py` and `packs/make_pack_fixtures.py`; regenerate with `python -P` from a singlet-bio checkout).
Validation beyond the unit tests (not committed): the pack reader agrees with the Python `names_at` on 3000 random indices in each of the 16 shipped packs
(human, mouse, zebrafish, macaque, rat exon+intron, the barnyard composites, four whitelists); cold decode of the human exon pack 546 ms.

Owner for CS4: apply `schema/032_v2_catalog.sql` first (and 031 from CS1); no new binding or secret is needed. Optional Pages variable `V2_ORIGIN`.
`package-lock.json` was regenerated locally with `npm install fzstd@0.1.1 --save-exact`; `lockfile.yml` can regenerate it if preferred.

---------------------------------------------------------------------------------------------------------------------------------

## CS5: website

Files: `src/components/DownloadPanel.tsx` (prop `v2`; legacy unchanged; with the legacy file present it still offers Download and mentions the per-sample layout; once retired it
offers "Manifest" and a curl line for it, loaders unchanged), `src/components/study/FileTreePanel.tsx` ("What's in the study", "Sample objects", a note when members are
listed for the first 100 samples only), `src/pages/StudyDetail.tsx` (passes `series.v2`), `src/integrations/api/types.ts` (`StudyV2`, `GseRow.v2`, `BundleIndexResponse`
optional `format/manifest_url/files_truncated`), `src/integrations/api/client.ts` (parses `v2`; never rebuilds the legacy URL from the bundle key for a retired study),
`src/pages/Docs.tsx` (Files on disk, the HTTP API for per-sample studies, 17 tools), `src/pages/DocsMcp.tsx` (three new tools, new inputs, two example asks),
test `src/test/download-panel.test.tsx` (8). Not done: the Browse page's direct download links (`StudyCard`, `StudyTable`, `SelectionBar`, `CohortDetail`) still build
`data/<GSE>/<GSE>.singlet` from the accession; they must be switched (to the study page or the manifest) before the first legacy bundle is retired. No CHANGELOG entry
yet: add one when the first v2 study is public. Not looked at in a browser (component tests only).

---------------------------------------------------------------------------------------------------------------------------------

## Retirement checklist for the catalog (found while doing CS4; none of it is done)

1. Studies published only as v2 have no `bundle_manifest`/`bundle_index` rows and `gse_meta.has_bundle` is 0: `DOWNLOADABLE_SQL`, the bit index (`downloadable`), search
   and the headline counts key on those. Decide where `has_bundle` is set for v2 studies (the ingest endpoint could set it; left alone on purpose).
2. `loadSampleQc` and the index-next crank read `samples/<GSM>/summary.json` from the legacy bundle for studies without `sample_qc` rows. A v2-only study needs the same
   from the stored `summary.json` member (kind 3, zstd; `fzstd` is available now).
3. Browse links (CS5 note above), `compare_studies` and `find_matched_controls` still report the legacy URL and size.
4. `pz_chunk_index` is not needed for v2; leave it alone until the legacy path is gone.

---------------------------------------------------------------------------------------------------------------------------------

## Mirror into singlet-cloud (read only; branch main at f227fc241, 2026-10-20)

singlet-cloud binds the same D1 and the same bucket, so the two schema files are applied once to the shared database; the cloud repo only needs the SQL files for the record.
The cloud has its own `030_cloud_beta.sql` and `031_cloud_tools.sql`: the same numbers as the website's `030_search_facts.sql` and my `031`, different names and tables, so no content
clash; keep the file names exactly as in the website. `workers/data-v2` is deployed once from the website repo (the cloud dropped `workers/`); do not copy it.

| file | cloud copy vs the website baseline | straight copy? |
|---|---|---|
| `functions/_shared/bundle-reader.ts`, `catalog-refresh.ts`, `api/gse/[id].ts`, `api/manifest.ts`, `components/study/FileTreePanel.tsx` | identical apart from line endings | yes (but `api/gse/[id].ts` and `api/manifest.ts` import `v2-urls.ts`, new) |
| `functions/_shared/manifest-core.ts` | identical apart from line endings; needs `v2-serve.ts` (new) | yes, with the new files |
| `functions/_shared/bundle-core.ts` | cloud lacks the `publicEntries` filter (3 lines) | **no**: the cloud does not import `./restricted` |
| `functions/_shared/mcp-tools.ts` | cloud lacks the `publicEntries` filter and the `completeness` lines (7 lines) | merge by hand |
| `functions/api/bundle/[gse]/[action].ts` | cloud lacks `publicEntries` and has the edge key `v=1` (5 lines) | merge by hand; its `index` key is v=1 there, bump to v=3 as well |
| `functions/mcp.ts` | cloud INSTRUCTIONS and descriptions are worded differently (11 lines) | merge by hand: the INSTRUCTIONS paragraph and a few descriptions conflict |
| `functions/_shared/env.ts` | cloud adds `ADMIN_TOKEN` | merge (add `V2_ORIGIN`) |
| `functions/_shared/study-core.ts` | 26 lines differ (completeness, restricted wording); the cloud has no `completeness.ts` | merge by hand: add the `v2` block, the `bundle_url` rule, `hasFile` |
| `functions/api/ingest/[table].ts` | 70 lines differ | apply only the `index-next` hunk (StaleBundleError retry) by hand |
| `functions/_shared/cohort-manifest.ts`, `functions/api/cohort/manifest.ts` | **absent in the cloud** (no cohort API there) | skip until the cloud has the cohort feature |
| new: `mcp-core.ts`, `mcp-v2.ts`, `rev.ts`, `v2-catalog.ts`, `v2-cells.ts`, `v2-packs.ts`, `v2-serve.ts`, `v2-urls.ts`, `api/ingest/v2-study.ts`, `schema/031_*.sql`, `schema/032_*.sql` | n/a | copy, but **`v2-catalog.ts` and `v2-serve.ts` import `./restricted`, which the cloud does not have** |
| `package.json` / `package-lock.json` | cloud has another license field, other dependencies, no `enrich:*` scripts | add `fzstd` 0.1.1 only; regenerate the lock there |
| `scripts/smoke.sh` | cloud smoke lacks the cohort, restricted, layer and library checks; its tool count is 14 | add the offsets block; set the count to 17 once the tools are mirrored |
| `src/components/DownloadPanel.tsx`, `src/integrations/api/{client,types}.ts`, `src/pages/{Docs,DocsMcp,StudyDetail}.tsx` | different product UI (83 to 146 lines apart) | do not copy; port by hand if the cloud shows downloads |
| tests | self contained except `mcp-v2.test.ts` / `v2-api.test.ts` (read the schema files `001_init_local_nofts.sql`, `010`, `012`, `032`) and the helpers `helpers/zip.ts`, `bitindex-fixture.ts`, `d1-sqlite.ts` | copy with the helpers |

The one decision that blocks a straight copy: the cloud has no `restricted.ts` (the website's public filter for per-donor genotype layers, PR #47). The v2 ingest refuses restricted
members and the v2 listings filter them. If the cloud is meant to serve those layers to controlled-access accounts, `v2-catalog.ts` and `v2-serve.ts` need a flag instead of a hard
filter; if not, mirror `restricted.ts` first.

---------------------------------------------------------------------------------------------------------------------------------

## How to resume / what is not done

* Nothing is committed. Suggested PR order: CS1 (+ apply 031), CS2 (deploy the Worker, check the route), CS3 (+ apply 032), CS4, CS5. CS4 and CS5 also carry CS1's shared files, so merge CS1 first.
* Repair of existing stale `bundle_index` rows (w4_cf 0.3 to 0.5, 0.7): not part of W5 by instruction.
* Not verified live: the data Worker on the real edge (route precedence, Cache API Range on a hit); the Cloudflare plan limits (CPU, subrequests); `find_cells` against real v2 objects
  (only the Python-written fixture and the unit tests).
* The Python-written fixtures pin the 256-byte pz2 header and the CEL2 layout of 2026-10-22; if W1 or W2 change either, regenerate with the two generator scripts and fix
  `parsePz2Header` / `v2-cells.ts`.
