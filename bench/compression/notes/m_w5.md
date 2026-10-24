# m_w5: serving-layer work (W5), change sets for singlet-website

Worktree: `C:\Users\zacha\Projects\singlet-website-v2`, branch `feat/singlet-v2` (from origin/main 0dcf4914a). Nothing committed, pushed or
deployed by W5. The orchestrator commits; the proposed commits and PRs are in "Proposed commits and PRs" below (section 6 of this file).

PR #51 (the three PRs above, merged as ed2f776) is on main and deployed; schema 033 is applied; GSE145370 is the first public v2 study. The follow-up work
(docs copy, CHANGELOG, retirement) is in section 9, in a second worktree.

Status (after the review fixes of 2026-10-22):

| set | state | new tests |
|---|---|---|
| CS1 stale-index guard | done, `npm run check` green; schema 031 is applied in production | 39 (`bundle-reader-guard`) |
| CS2 data Worker `workers/data-v2` | done; deployed as version 732fe7f6 BEFORE the round-2 fill rule (item 8), so it needs one more `wrangler deploy` | 120 |
| CS3 D1 v2 schema + ingest | done; 032 is applied in production (empty tables), **033 is not applied yet** and must be before the ingest code is live | 26 |
| CS4 API + MCP on v2 | done; v2 tools and v2 inputs are gated, the response diet is off by default | 144 (serve 14, cells 49, packs 29, mcp-v2 34, mcp-diet 6, edge-ttl 3, api 9) |
| CS5 website panel | done (download panel, file tree, types/client); the docs copy is held back (follow-up patch) | 8 |

`npm run check` (vitest + strict `tsc -p tsconfig.functions.json` + `wrangler pages functions build` + `tsc -p workers/data-v2`): 44 files, 662 tests,
all green (325 before W5). `npm run build` (vite) green. Not run: a browser, a deployed preview, real v2 objects.

---------------------------------------------------------------------------------------------------------------------------------

## 1. Round 2: the review fixes (ten items)

| # | item | what is in the tree now | tests |
|---|---|---|---|
| 1 | MCP diet off | `DEFAULT_VERBOSE = true`. The whole diet (short text, `verbose` argument, `structuredContent` pointer in the instructions) runs only with the Pages variable **`MCP_DIET=1`**. Without it every answer has the old double shape and the `verbose` argument is not even advertised or read. With it, `summarize` keeps the lines of a list-shaped text in order (no trimming of a row's trailing tabs, lines clipped at 400 characters, at most 100 lines and 9,000 characters) and ends with "... N more lines." plus a pointer; a text that fits is returned whole | `mcp-diet.test.ts` (6): search_datasets (all 10 results with accession, title and the end of the reason), get_sample_qc (totals and a line per sample), export_manifest (headline, whole header row, first rows, count of the rest; a short manifest is identical to the full one), assess_study (title, file, per-sample files, QC, fit checks), plus "diet off changes nothing"; and the `summarize`/`diet` unit tests in `mcp-v2.test.ts` |
| 2 | v2 advertising gated | `find_cells`, `slice_cells`, `resolve_features`, the v2-only inputs (`detail` on list_bundle_files, `include_samples` on get_download_url) and the v2 paragraph of the server instructions appear only when some study is **served** from v2 (a `v2_study` row joined to a `v2_migration` row that is `verified` or `retired`: stricter than "a row exists", on purpose: an `uploaded` or `running` study serves nothing) or when `V2_TOOLS=1`. One D1 read per isolate per 5 minutes (`functions/_shared/v2-gate.ts`, concurrent callers share it; before schema 032, or on any D1 error, the answer is no). The GET description of the server follows the same rule. The tools still answer when called by name. tools/list is 16.3 KB (14 tools, main was 15.7 KB; `known_rev` added), 21.7 KB with the v2 tools, 23.6 KB with v2 and the diet. Docs.tsx, DocsMcp.tsx and README copy about v2 is removed and kept as a follow-up patch (see CS5) | `mcp-v2.test.ts` "tools/list and the instructions" (8): 14 tools by default, 17 once a study is served, not for uploaded/running/failed, retired still counts, `V2_TOOLS=1` makes no D1 read, one read per 5 minutes (fake timers), a database without the v2 tables answers 14, tools answer by name, the GET description |
| 3 | private-upload URLs | `httpBundleSource(url, {noRedirects, userHost})`. Our own host: 206 only (as before). A user-registered URL: a 200 to a ranged request is accepted for ranges ending within 32 MiB (`USER_HOST_FULL_READ_MAX`): at most end+1 bytes are read, the body is cancelled, the slice is returned; beyond 32 MiB a host that ignores Range is refused with a clear error (so a private upload from a host without Range support works up to 32 MiB only). `If-Match` goes to a user host only with a strong ETag taken from that host's own HEAD (never the stored one); a 412 there means "re-HEAD, adopt the new ETag, retry once"; only a second 412 is `StaleBundleError`. `functions/api/projects/[id]/upload/[action].ts` passes `userHost: true` | `bundle-reader-guard.test.ts` (user-host block) |
| 4 | smoke.sh | The offset check is skipped (printed as `ok ... skipped`) for a study whose index JSON contains `"format":"singlet-v2"` and for one whose legacy URL answers 404 to HEAD. The tool count check accepts 14 or 17 | `bash -n` only; run it against the preview or production |
| 5 | ingest semantics | A manifest with no `migration` block is recorded as `uploaded` (never live). A `migration` block must carry a valid `status`. `verified` and `retired` need `verify_ok: true` (422 otherwise). New schema file **033** keys `v2_sample` by `(gse_id, rev, gsm_id)`: the new rev's rows are written beside the served rev's (nobody reads them), the final post flips `v2_study.rev`, deletes the other revs' sample rows and writes the migration row in ONE atomic batch. Readers ask for the rev named by `v2_study` (a subquery in the keyed read), so no reader sees a mix. A non-final slice never touches `v2_study` or `v2_migration` | `v2-catalog.test.ts` (26) |
| 6 | edge cache keys | A study with a `v2_study` or `v2_migration` row is cached at the edge for **5 minutes** (the `X-Singlet-Cache-Ttl` response header, which `cachedJson` turns into `max-age`/`s-maxage` and strips); a study with neither keeps the day. The keys do not carry the rev (that would cost a D1 read before every cache lookup); the short lifetime bounds the staleness instead. A cached answer made before the first v2 row existed lives up to 24 h: see the retirement checklist | `v2-edge-ttl.test.ts` (3) |
| 7 | weak ETags, HEAD timeout | `strongEtag()` filters the recipes' `If-Match` in `/entry`, `get_partial_download` and the python/curl snippets (a weak or empty validator is dropped, never sent). The verify HEAD of `getBundleIndex` runs under `AbortSignal.timeout(5000)`; on timeout the cached row stands | `bundle-reader-guard.test.ts` ("recipes and the verify HEAD") |
| 8 | data Worker fill rule | A cold Range (or a 304/412 revalidation) on an object over **32 MiB** (`RANGE_FILL_MAX_BYTES`) no longer starts a background full-object fill. The first miss of the key is remembered in the isolate (a set of at most 256 keys, oldest forgotten first); the second miss of the same key starts the fill; any full GET fills at once (tee up to 8 MiB, a separate read above). Objects up to 32 MiB behave as before. See `workers/data-v2/README.md` | `data-v2-worker.test.ts` (120; 7 new/changed in "a cold Range on an object over 32 MiB" and the cap test) |
| 9 | `find_cells` CPU | See "CPU and plan" below. Gated by item 2; no other change | none |
| 10 | ingest write budget | See "Rows written" below. The unchanged-rev rule was already in (compare the stored rev first, write nothing when equal, `force: true` to rewrite); the response now also carries `rows_written`, the sum of D1's own `meta.rows_written`, so the first real post measures the cost | `v2-catalog.test.ts` ("posting the rev that is already stored...", "reports the rows D1 says it wrote") |

### Rows written by the v2 ingest (item 10)

D1 bills table rows and index rows. By SQLite's storage rules (not yet measured against D1's meter: the first real post reports `rows_written`):

| write | rows written |
|---|---|
| a new sample row (`v2_sample`: table row, `(gse_id, rev, gsm_id)` primary-key index, `idx_v2_sample_gsm`) | **3** |
| a sample row of an older rev deleted at the flip | 3 |
| a sample re-posted with the rev already stored (no `force`) | **0** (`ON CONFLICT DO NOTHING`, and an unchanged study skips the sample statements altogether) |
| `v2_study` row: new / rev change | 2 / 1 |
| `v2_migration` row: new / update | 3 / 1 (2 when `status` changes: `idx_v2_migration_status`) |
| a whole unchanged post | 0 for the study, 1 to 2 for the migration row only when the body carries a `migration` block |

So a new study of N samples costs about 3N + 5; a rev change costs about 6N + 4 (new rows now, old rows deleted). **An atlas of about 78,000 samples (the size assumed in these estimates; check it against the catalog) is about 234,000 row writes**, more than twice the free plan's cap of 100,000 rows written per day: the initial load of an atlas that size needs the Workers Paid plan (which includes 50 million rows written a month) or a campaign that posts about 30,000 samples a day. The per-post limit stays 400 samples (about 1,200 rows) so one POST is far from any per-request limit.

### CPU and plan (item 9)

The Workers plan of the Pages project is unknown to W5. The Free plan allows 10 ms of CPU per request; Paid allows 30 s by default. `find_cells` decodes zstd in JavaScript (`fzstd`; about 22 ms for a 100k-row column on node, warm) and filters in the Worker, so on Free it will fail with Worker CPU-limit errors on anything but a tiny cells table. `resolve_features` parses a reference pack (cold decode of the human exon pack measured at 546 ms on node) and would exceed 10 ms too. `slice_cells` reads a 256-byte header and a chunk index and does not decompress, so it is fine on either plan. All three are advertised only after a study is served from v2 (item 2); nothing else changes. Before the first v2 study goes public, confirm the plan (dashboard, Workers & Pages, Plans) or accept that `find_cells` and `resolve_features` are Paid-only.

---------------------------------------------------------------------------------------------------------------------------------

## 2. Contract for W6 (campaign): what to POST to the website

`CATALOG_SCHEMA` in `scripts/v2_campaign.py` guessed a row-per-table interface (`POST /api/ingest/<table>` with `{rows}`) and column names that
differ from the ones below. The implemented interface is manifest based and is one call per study (or per slice of a study). Change
`catalog-post` to this and drop `CATALOG_SCHEMA`/`catalog-sql` row building (the SQL files are not needed: the endpoint upserts).

```
POST https://singlet.bio/api/ingest/v2-study            header X-Ingest-Token, Content-Type application/json, a non-default User-Agent
{
  "manifest": { ...the v2/m/<GSE>.json object WITHOUT "study_files" (the server ignores it and the body limit is 8 MiB)... },
  "manifest_bytes": 123456, "manifest_sha256": "<64 hex>",             // size/hash of the uploaded manifest object (optional)
  "final": true,                                                         // optional, default true
  "force": false,                                                        // optional: rewrite the rows of a rev that is already stored
  "migration": { "status": "verified", "verify_ok": true, "legacy_key": "...", "legacy_etag": "\"...-3\"", "legacy_bytes": 187066350, "v2_bytes": 123456789,
                 "n_samples": 12, "members_by_kind": {"matrix": 24, "recipe": 90, "stored": 100},
                 "commit_sha": "...", "migrated_at": "...", "verified_at": "...", "ledger_key": "v2/ledger/GSE1.jsonl", "error": null }
}
POST same URL, state only:  { "gse_id": "GSE1", "migration": { "status": "running" | "failed" | "pending" | ..., "error": "..." } }
GET  same URL ?gse=GSE1 (token): { study, migration, sample_rows_by_rev, complete }        // verification after a post
response: { ok, gse_id, rev, unchanged, samples_written, rows_written, n_samples, final, migration_status, ms }
```

**Changed in round 2 (W6 must follow):**
* **No implicit live.** A manifest posted without a `migration` block is recorded as `uploaded` and is NOT served. A `migration` block must have a `status` from
  `pending | running | uploaded | verified | failed | retired` (`MIGRATION_STATUSES` in `functions/_shared/v2-catalog.ts`). `verified` and `retired` require
  `verify_ok: true` and answer 422 without it; post them only after the bit-for-bit verification of format-v2 section 10 passed.
* **Served rule:** a study is served from v2 only when `v2_study` exists and its `v2_migration.status` is `verified` or `retired`. A study without a migration row is not served.
* **Same rev again = nothing written.** Posting the rev that `v2_study` already holds writes no sample row and no study row (`unchanged: true`, `samples_written: 0`);
  the migration row is still written when the body has a `migration` block (this is how `uploaded` becomes `verified`). A body without a block leaves it alone.
* **A new rev is written beside the served one** and flips on the final post (schema 033); send `final: false` on every slice but the last. A failed upload of a new rev
  therefore leaves the served rev untouched; re-post the same slices (they are `ON CONFLICT DO NOTHING`) or `force`.
* **Order in production:** apply `schema/033_v2_sample_rev_key.sql` BEFORE the ingest code is deployed. Until then the endpoint answers 503 with the message
  "apply schema/032... and schema/033...".
* The `retired` status needs `verify_ok: true` as well (the legacy file is deleted afterwards, so the check is the same).

Unchanged from the first pass:
* `manifest` is validated, not trusted: `format == "singlet-v2"`, `manifest_version == 1`, `gse`, a 16-hex `rev`, every sample `key` must be
  `v2/s/<its GSM>/<32 hex>.s2`, `members` with integer `kind` (1-5), `off`, `len` (the real manifest member dicts from `migrate._manifest_member`
  pass unchanged: `chunk_index`, `n_chunks`, `codec`, `names`, `recipe`... are stored as given). 422 if a sample lists a restricted member
  (`donor/snp_*`, `ase_counts.tsv`, `mt/*` other than `mt_summary.tsv`, vcf...): such a member must not be in a public object at all.
* At most 400 samples per post. A larger study is sent in slices: every slice carries the same `rev`, `n_samples` (the study total) and `bytes`
  (total over all samples) and `final: false`; the last slice has `final: true` (or omits it). Sample rows are written by every slice; the
  `v2_study` row, the removal of rows of an older `rev` and the migration row only by the final one.
* Order for one study: upload objects and manifest to R2, verify, then this POST.
* Columns (schema/032 + 033): `v2_study(gse_id PK, manifest_key, rev, bytes, n_samples, created_at, legacy_bytes, manifest_bytes,
  manifest_sha256, n_cells_called, reference_build, packs_json, pipeline_json, updated_at)`; `v2_sample(gse_id, gsm_id, key, bytes, sha256, n_columns,
  n_called, n_members, median_umi, median_genes, modalities, members_json, cell_stats_json, rev, updated_at, PRIMARY KEY (gse_id, rev, gsm_id))` and an
  index on `gsm_id`; `v2_migration(gse_id PK, status, legacy_key, legacy_etag, legacy_bytes, v2_bytes, n_samples, members_by_kind, verify_ok, commit_sha,
  migrated_at, verified_at, ledger_key, error, updated_at)`.
  **Deviation from the brief:** `v2_sample` is not keyed by `gsm_id` alone, because a SuperSeries and its SubSeries pack the same GSM (index-next already
  special-cases this for sample_qc) and their manifests can point at different objects for it; a secondary index serves lookups by `gsm_id`.

---------------------------------------------------------------------------------------------------------------------------------

## 3. CS1: stale-index guard (ship first)

What it fixes: `bundle_index` rows outlive the object they describe (the 2026-10-20/21 migration overwrote about half the atlas in place);
`getBundleIndex` returned them without any check, so recipes pointed into other bytes (GSE111113: 42,723 bytes late) and `/entry` answered 502.

| change | where |
|---|---|
| Identity stored with each row: `etag`, `last_modified`, `verified_at`. New schema file; the code works without them: the first statement that finds a column missing is remembered for a minute, during which reads select fewer columns and writes use the old INSERT, and the If-Match half is simply off; nothing is probed or created up front, so a read of a stored row is one statement | `schema/031_bundle_index_identity.sql`, `functions/_shared/bundle-reader.ts` |
| Every ranged read sends `If-Match: <etag>` (stored, or pinned from the HEAD that started the read); 412 and 416 are `StaleBundleError`; **only 206 is accepted for a range from our own host** (a 200 is refused with its body dropped unread; it used to be buffered, up to 29 GB); weak ETags are never sent | `bundle-reader.ts` (`fetchRangeWithEtag`, `strongEtag`) |
| User-registered URLs (private uploads) are a separate branch: 200 accepted by reading at most end+1 bytes (ranges within 32 MiB), `If-Match` only with a strong ETag from that host's own HEAD, 412 means re-HEAD and one retry | `bundle-reader.ts` (`httpBundleSource`, `userHost`), `functions/api/projects/[id]/upload/[action].ts` |
| Every entry read checks the zip local header at the offset: signature `PK\3\4` and that the name equals the entry's (reads 30 bytes + name, no extra request); mismatch is `StaleBundleError`. A local extra field longer than the read slack gets one more exact read | `bundle-reader.ts` (`payloadOffset`, `readEntryFromSource`, `entryRange`, `entryPayloadRange`) |
| `withIndexRetry`: on `StaleBundleError` re-read the central directory once and run again; a second failure is rethrown. Concurrent re-reads of one study share one read (isolate-level). Used by the `/entry` route, MCP `get_partial_download`, `loadSampleQc`, `index-next` | `bundle-reader.ts`, `functions/api/bundle/[gse]/[action].ts`, `functions/_shared/mcp-tools.ts`, `functions/_shared/bundle-core.ts`, `functions/api/ingest/[table].ts` |
| `getBundleIndex(..., {verify:true})`: one HEAD (5 s timeout) compares size (and ETag if stored) with the row and re-reads on a difference; a failed or timed-out HEAD leaves the row. Used by `/api/bundle/:gse/index` (edge-cached) and MCP `list_bundle_files`, so the size and listing they report are the file's now | same |
| Recipes (`/entry` over 4 MB, `get_partial_download`) and `/index` carry `etag` (strong only); the curl/python snippets send `If-Match` so a client's read fails with 412 instead of returning other bytes after the file is replaced again | `[action].ts`, `mcp-tools.ts`, `bundle-core.ts` |
| `/entry` answers 409 with a clear message (not a cached 200, not a bare 502) when the index and the file still disagree after one refresh | `[action].ts` |
| Edge-cache keys: `index` v=2 -> **v=3** (main was already v=2, from PR #47), `samples` v=1 -> v=2, `entry` `path=` -> `v=2&path=`. Old entries expire by TTL (1 day) | `[action].ts` |
| `readEntry`/`readEntryText` take the index's etag; `parseZipSource` returns `etag`, `last_modified` | `bundle-reader.ts`, `bundle-core.ts`, `catalog-refresh.ts` |
| smoke.sh: for GSE111113 (rewritten in place) and GSE138867 (original): index `bytes` equals the HEAD content-length, and the first 4 bytes of the exon matrix at the offset the API reports are `TP1Z`; skipped for a study served from v2 or whose legacy file is gone. Run against production before the fix it FAILS for GSE111113 (index 324,676,242 vs object 170,777,036; reads garbage) and passes for GSE138867, which is the point | `scripts/smoke.sh` |

Files (CS1): `schema/031_bundle_index_identity.sql`, `functions/_shared/bundle-reader.ts`, `functions/_shared/bundle-core.ts`,
`functions/_shared/catalog-refresh.ts` (1 line), `functions/_shared/mcp-tools.ts` (shared with CS4: `getPartialDownload`, `listBundleFiles`),
`functions/api/bundle/[gse]/[action].ts`, `functions/api/ingest/[table].ts` (index-next only), `functions/api/projects/[id]/upload/[action].ts` (1 line),
`scripts/smoke.sh` (shared with CS4: the tool count), `src/test/bundle-reader-guard.test.ts` (39 tests), `src/test/helpers/zip.ts` (zip writer and fake object host).

Verified live (read-only GETs): `data.singlet.bio` honours `If-Match` on ranged reads (match -> 206, mismatch -> 412, weak validator -> 412,
past the end -> 416), so the guard works on the R2 custom domain as it is. With the code, `entryRange` on live GSE111113 returns header
5,703,179, payload 5,703,263 and `TP1Z`, and a recipe 42,723 bytes late is rejected as stale. The orchestrator repaired the `bundle_index` rows today
(etag columns NULL, so the If-Match half starts when each row is next refreshed).

Not in CS1: repair of existing rows (done separately), `verify-index`/light re-index endpoint (w4_cf 0.3/0.4), catalog size repair (0.5), the
`restrict_published_bundles.py` patch (0.7). Residual risk: a recipe cached at the edge for a day (`/entry` over 4 MB) can still go stale if the file is
overwritten again within that day; the recipe carries the ETag so a client's own read fails safely. `/api/cohort/manifest` still reads `bundle_index` rows
without a guard for legacy studies (it never fetches); for migrated studies CS4 replaces that path.

---------------------------------------------------------------------------------------------------------------------------------

## 4. CS2, CS3, CS4, CS5

### CS2: data Worker `workers/data-v2/`

Files: `workers/data-v2/{wrangler.toml,tsconfig.json,README.md,.gitignore}`, `workers/data-v2/src/handler.ts` (all behaviour), `src/index.ts` (entry),
`src/test/data-v2-worker.test.ts` (120 tests, node env, fakes for R2 and the Cache API), `package.json` (the `check` script gains
` && tsc -p workers/data-v2/tsconfig.json --noEmit`; that hunk only).

Behaviour (full table in the README): GET/HEAD/OPTIONS on `v2/s/<GSM>/<32hex>.s2`, `v2/ref/<32hex>.pack`, `v2/ref/index.json`, `v2/m/<GSE>.json`,
`v2/cells/<GSE>.cells`; everything else 404/405. Single `Range` (206 + Content-Range, 416 + `bytes */size`; multi-range and malformed ranges are
ignored and the object is streamed whole), `ETag`, `If-Match` (412), `If-None-Match` (304), `If-Range` (ETag form), CORS `*` with exposed headers,
`Cache-Control` immutable (1 year) for s/ and ref/<id>.pack and `max-age=300` for m/, cells/ and ref/index.json, correct Content-Type, `no-store`
on errors. Edge cache = Cache API keyed by the canonical URL without query or Range: a hit serves any Range from the cached full object; a miss
without Range streams from R2 and fills the cache (tee up to 8 MiB, a second R2 read above that); a miss with Range serves the range from R2 and, **for
objects up to 32 MiB**, fills the cache in the background (`waitUntil`, one fill per key per isolate); **for a bigger object the first ranged miss only
records the key** and the second miss of the same key in that isolate starts the fill (round 2, item 8); objects over 500 MiB are never cached. Costs one
extra Class B read per cold object per data centre.
Established with a probe Worker on workerd (local) and the docs: `cache.match` slices a cached 200 for a single valid Range (206 with
Content-Range); an unsatisfiable range gives a bare 416 without ETag (the handler re-matches without Range and decides itself); `If-Match` and
`If-Range` are ignored by the Cache API (the handler evaluates every conditional itself); a HEAD never matches; `put` rejects a 206.
Local verification: `npx wrangler dev --local` with a seeded local R2: cold GET MISS then HIT with the same ETag, Range on a hit 206, clamp/suffix/open-ended,
416, If-None-Match 304, If-Match 412, If-Range, HEAD, OPTIONS 204, POST 405, `..`/`//`/`%2f` 404; a Range on a cold 12 MiB object returned 206 from R2 and the
same Range was a HIT two seconds later (background fill), full GET byte identical. The round-2 rule is covered by unit tests only (not re-run under wrangler dev).
Open point from the fork: Cloudflare may compress `.json` on the fly and weaken its ETag; only manifests are affected and nothing sends If-Match for them.

State: deployed by the orchestrator (version 732fe7f6, route `data.singlet.bio/v2/*` wins, `X-Served-By: data-v2`), `compatibility_date = "2026-09-01"` pinned
in `wrangler.toml` (kept). **That deploy predates the 32 MiB fill rule**; the rule changes cache behaviour only (responses are byte-identical), so it can go
out whenever convenient: `cd workers/data-v2 && npx wrangler deploy`. Rollback = remove the route; the Worker never writes to the bucket.

### CS3: D1 schema for v2 + ingest endpoint

Files: `schema/032_v2_catalog.sql` (applied in production, empty tables), `schema/033_v2_sample_rev_key.sql` (**not applied**), `functions/_shared/v2-catalog.ts`
(validation, upsert statements, typed reads), `functions/api/ingest/v2-study.ts` (POST/GET/OPTIONS, a static sibling of `[table].ts`), `src/test/v2-catalog.test.ts`
(26 tests, real SQL through node:sqlite and the real schema files).
Contract and columns: section 2. Read helpers: `getV2State(db, gse)` (two primary-key reads; null unless served from v2), `v2ServedStudies(db, gses)`
(chunks of 90), `getV2Sample`/`getV2Samples` (<= 50 with members; optional rev, default the study's current rev), `getV2Presence`/`servedState` (study and migration rows,
for the edge TTL), `listV2Samples(db, gse, rev, {limit<=500, after})` (no JSON columns), `v2Available(db)` (try, remember a missing table for a minute). Read-budget test:
EXPLAIN QUERY PLAN shows keyed SEARCH for every read and for the stale-row delete; the list query selects no JSON column; there is no `json_each`.
Decisions: modalities are computed at ingest from the public member names with the existing registry (`resolveModalities`) and stored as a small JSON array;
`median_umi`/`median_genes` are typed columns so depth filters run in SQL; the member dicts are stored whole (`members_json` <= 128 KB per sample) and read one sample at a time.
Owner: apply `schema/033_v2_sample_rev_key.sql` to production D1 before the ingest code is live (`npx wrangler d1 execute singlet-catalog --remote --file=schema/033_v2_sample_rev_key.sql`;
it renames the table, recreates it with the new key, copies the rows and drops the old one: safe on the empty tables). The ingest token is the existing one (`INGEST_TOKEN_SHA256`).

### CS4: API + MCP on v2, with legacy fallback

Rule: a study is **served from v2** when it has a `v2_study` row and its `v2_migration` row says `verified` or `retired` (round 2: no migration row means NOT served).
`running`/`uploaded`/`failed`/`pending` keep the legacy answer. One call: `getV2State` (two keyed reads, no probe; before schema 032 the first read finds no table and
the next minute is answered without touching D1).

| area | v2 behaviour (legacy path unchanged) |
|---|---|
| `get_download_url` | `format: "singlet-v2"`, `rev`, `manifest_url`, n_samples, v2 bytes, pack ids, `legacy` block (url, bytes) until status is `retired`, `curl -fLO <manifest>`; `include_samples`+`limit<=200`+`after` pages the sample object URLs |
| `get_partial_download` | the member's absolute range inside the immutable sample object (`url`, `range`, `bytes`), kind, codec id/name, chunk facts, pack refs (`names`), `curl`, no If-Match (content addressed); kind 4 members return `range: null` and the recipe; `slice_with` points at `slice_cells` for matrices |
| `list_bundle_files` | grouped summary by default (file kinds of one representative sample, light per-sample rows, `next_after`), `gsm_id` gives that sample's members with ranges, `detail=true` pages 25 samples with their members |
| `get_modalities` | from `v2_sample.modalities` (a short JSON array computed at ingest), no member list read |
| `export_manifest` / `/api/manifest` | rows gain `format`, `v2_manifest_url`, `v2_rev`, `v2_bytes`; `download_url` and `bytes` stay the legacy file until it is retired, then the manifest URL and v2 size; curl lines for a retired study fetch the manifest; TSV gets three columns appended (`format`, `v2_manifest_url`, `v2_rev`) |
| `/api/cohort/manifest` | a v2 study is planned from `v2_sample.members_json` (no legacy index, no `pz_chunk_index`): `studies[].format/rev/url(manifest)`, `samples[].object`, `files[]` with absolute ranges, `chunk_index.range`, `chunk_cols`, `n_chunks`; column keep-lists unchanged. Reads <= 50 samples per query |
| `GET /api/gse/:id`, `get_study`, `assess_study` | `series.v2 {format, rev, manifest_url, n_samples, bytes, legacy_available}`; `bundle_url` is null once retired; the file counts as present for completeness; `assess_study` describes the v2 files |
| `GET /api/bundle/:gse/index` | same shape plus `format`, `rev`, `manifest_url`, `files_truncated`; members listed for the first 100 samples; edge-cached 5 minutes while the study has any v2 row |
| `GET /api/bundle/:gse/entry` | v2: where the member's bytes are (`url`, `range`, `method` pz2/zstd/stored/derived), never inflates |
| `find_cells` (new, gated) | D1 picks samples (`v2_sample` joined to `v2_study` at the current rev, `gsm` for organism/tissue/disease/assay, `sample_qc` usable and tier), then each cells table is read by Range (one read when the table is under 1 MiB; directory + per-column reads otherwise) and filtered in the Worker. `cell_filter`: `called_only` (default true), `min/max_umi` (exon+intron), `min/max_features` (larger of exon, intron genes), `where` on any numeric/bool column. Returns `[start,length]` column runs (<= 100 per sample, `runs_truncated`), exact `n_match`, `next_after`. Bounds: 25 samples, 40 reads of the data host, 6 at a time; stops with `budget_exhausted` and says where to continue. CPU: see "CPU and plan" |
| `slice_cells` (new, gated) | reads header + chunk index of each matrix (one ranged read), returns the prefix ranges (header, chunk index, tables, rows) and the chunk-run ranges for the named cells (`runs`, `indices`, `range`, none = all), `columns_wanted/covered`, `overread_ratio`; joins runs to stay within `max_ranges` (default 64). Both matrices by default |
| `resolve_features` (new, gated) | `pack` id (16-32 hex) or `gse_id+gsm_id+file(+axis)` -> names for up to 500 indices from `v2/ref/index.json` + the pack file (vocabulary SVP0/SVPD/composite, or whitelist barcodes) |
| rev / known_rev | `rev` on `get_study`, `get_sample_qc`, `list_bundle_files`, `get_modalities`, `get_partial_download` (advertised for the 14 tools now); find/slice/resolve results carry the study `rev`. v2 file rev = manifest `rev`; legacy file rev = digest of the bundle ETag (size before schema 031). `get_study` and `get_sample_qc` combine the file rev with catalog timestamps (`gse.last_updated`, newest `gsm.last_updated`, counts; `sample_qc.updated_at`). `known_rev` equal -> `{gse_id, rev, unchanged: true}` |
| diet (OFF) | behind `MCP_DIET=1` only (item 1). Server version 1.1.0 |
| `tools/list` | 14 tools by default (16.3 KB), 17 once a study is served from v2 or `V2_TOOLS=1` (21.7 KB); `verbose` on every tool only with `MCP_DIET=1` (23.6 KB with v2) |

Decisions worth a second look:
* **Tool advertising follows the catalog.** The gate reads D1 at most once per isolate per 5 minutes; a Pages deployment where the v2 tables are missing advertises 14 tools. The
  gate requires `verified`/`retired` rather than any `v2_study` row, so a study that is only uploaded, or whose verification failed, never makes the tools appear.
* **Environment variables (all optional, Pages):** `MCP_DIET=1` (the short-text diet), `V2_TOOLS=1` (advertise the v2 tools without waiting for a served study, e.g. for a preview),
  `V2_ORIGIN` (origin of the v2 objects if the data Worker route does not win).
* `format-v2.md` section 4.1 still describes a 160-byte pz2 header; the code (`include/singlet/pz2/detail/container.h`, `python/singlet/v2/codec.py`) writes a
  **256-byte header with a CRC-32 at 252, sections at 72, total_len at 184**. The reader here (`v2-serve.ts parsePz2Header`) follows the code; fixtures from the
  current Python caught this. Update the spec text.
* The cells table is read with `fzstd` (pure JavaScript, MIT, 0.1.1, added to `package.json`), because the Worker runtime has no zstd. Frames must carry their content size
  (the writer sets it); a column over 64 MiB or a table over 50M rows is refused.
* Objects are fetched from the public data host with Range GETs (not the `SINGLET_DATA` binding), like the legacy code, so previews work. If the data Worker
  route did not win over the R2 custom domain, set `V2_ORIGIN`; every v2 URL in the API comes from it (`functions/_shared/v2-urls.ts`). (The route does win, per the orchestrator's deploy check.)
* The manifest member dict carries `chunk_index: [absolute off, len]` but not the chunk-section offset, so `slice_cells` reads the member header (256 bytes, one
  request together with the chunk index in files written so far). Adding `"chunks": [abs off, len]` to `_manifest_member` would let clients skip that read (W2).
* `v2_sample.modalities` and `median_umi/median_genes` are computed at ingest so searches do not parse member lists.

Files (CS4): new `functions/_shared/{mcp-core,mcp-v2,rev,v2-serve,v2-urls,v2-cells,v2-packs,v2-gate}.ts`; changed `functions/mcp.ts`, `functions/_shared/{mcp-tools,study-core,bundle-core,manifest-core,cohort-manifest,env,cache}.ts`
(`mcp-tools.ts` and `bundle-core.ts` are shared with CS1; `cache.ts` gains `CACHE_TTL_HEADER`), `functions/api/bundle/[gse]/[action].ts` (shared with CS1), `functions/api/cohort/manifest.ts`,
`functions/api/manifest.ts`, `functions/api/gse/[id].ts`, `package.json` + `package-lock.json` (fzstd), `scripts/smoke.sh` (14-or-17), `README.md` (one sentence about `rev`/`known_rev`); tests
`src/test/{mcp-v2,mcp-diet,v2-api,v2-edge-ttl,v2-serve,v2-cells,v2-packs}.test.ts` and `src/test/fixtures/v2/**` (a real `.s2` sample object, cells tables of every dtype and encoding, pack
files, all written by the Python reference with the generator scripts `make_fixtures.py` and `packs/make_pack_fixtures.py`; regenerate with `python -P` from a singlet-bio checkout).
Validation beyond the unit tests (not committed): the pack reader agrees with the Python `names_at` on 3000 random indices in each of the 16 shipped packs.

Owner for CS4: schemas 031 and 032 are applied; apply 033. No new binding or secret. `package-lock.json` was regenerated locally with `npm install fzstd@0.1.1 --save-exact`.

### CS5: website

Files: `src/components/DownloadPanel.tsx` (prop `v2`; legacy unchanged; with the legacy file present it still offers Download and mentions the per-sample layout; once retired it
offers "Manifest" and a curl line for it, loaders unchanged), `src/components/study/FileTreePanel.tsx` ("What's in the study", "Sample objects", a note when members are
listed for the first 100 samples only), `src/pages/StudyDetail.tsx` (passes `series.v2`), `src/integrations/api/types.ts` (`StudyV2`, `GseRow.v2`, `BundleIndexResponse`
optional `format/manifest_url/files_truncated`), `src/integrations/api/client.ts` (parses `v2`; never rebuilds the legacy URL from the bundle key for a retired study),
test `src/test/download-panel.test.tsx` (8). All of it shows nothing new until the API returns `series.v2`.
**Docs are held back (round 2, item 2):** `Docs.tsx` is back at HEAD; `DocsMcp.tsx` and `README.md` carry only the `known_rev` copy that is true today. The rest of the copy
(per-sample layout in Files on disk, the HTTP API for per-sample studies, "17 tools", the three new tools and their inputs, two example asks) is in
`bench/compression/notes/m_w5_docs_followup.patch` (a unified diff relative to the committed tree, checked with `git apply --check --ignore-whitespace`); apply it in the commit that makes the first
v2 study public. Not done: the Browse page's direct download links (`StudyCard`, `StudyTable`, `SelectionBar`, `CohortDetail`) still build `data/<GSE>/<GSE>.singlet` from the accession;
they must be switched (to the study page or the manifest) before the first legacy bundle is retired. No CHANGELOG entry yet. Not looked at in a browser (component tests only).

---------------------------------------------------------------------------------------------------------------------------------

## 5. Retirement checklist for the catalog (found while doing CS4; done in section 9, worktree singlet-website-v2b)

1. Studies published only as v2 have no `bundle_manifest`/`bundle_index` rows and `gse_meta.has_bundle` is 0: `DOWNLOADABLE_SQL`, the bit index (`downloadable`), search
   and the headline counts key on those. Decide where `has_bundle` is set for v2 studies (the ingest endpoint could set it; left alone on purpose).
2. `loadSampleQc` and the index-next crank read `samples/<GSM>/summary.json` from the legacy bundle for studies without `sample_qc` rows. A v2-only study needs the same
   from the stored `summary.json` member (kind 3, zstd; `fzstd` is available now).
3. Browse links (CS5 note above), `compare_studies` and `find_matched_controls` still report the legacy URL and size.
4. `pz_chunk_index` is not needed for v2; leave it alone until the legacy path is gone.
5. **Edge cache of legacy answers (round 2, item 6).** `/api/bundle/<GSE>/index` and `/entry` answers made before a study had any v2 row live up to 24 hours in the edge cache. Flipping
   a study to `verified` therefore leaves the legacy answer visible for up to a day. That is harmless while the legacy file exists, so: **do not delete a legacy file (status `retired`)
   until at least 24 hours after the flip to `verified`**, or purge those two URL families for the study. From the first v2 row on, the answers live 5 minutes.
6. The data Worker's cache of `v2/m/<GSE>.json` is 5 minutes and the site's v2 index is 5 minutes: after a rev change readers may see the old rev for up to 5 minutes, never a mix (the
   rev is the key of the sample rows).

---------------------------------------------------------------------------------------------------------------------------------

## 6. Proposed commits and PRs (the orchestrator commits)

Everything below is in the worktree, uncommitted; `git status` lists 26 modified and the untracked files named here. Nothing was verified on a subset: only the full tree passes
`npm run check`. The three code PRs are ordered so each can be merged on its own against `main` + the ones before it; PR 2 is the one that fixes production (stale offsets).

**PR 1: data Worker (CS2)**, can merge now, no dependency.
* `workers/data-v2/**` (wrangler.toml with the pinned `compatibility_date = "2026-09-01"`, tsconfig.json, README.md, .gitignore, src/handler.ts, src/index.ts)
* `src/test/data-v2-worker.test.ts`
* `package.json`: only the `check` script hunk (` && tsc -p workers/data-v2/tsconfig.json --noEmit`)
* After merge: `cd workers/data-v2 && npx wrangler deploy` for the 32 MiB fill rule (cache behaviour only).

**PR 2: serving layer (CS1 + CS3 + CS4), one PR.** The three sets share `bundle-core.ts`, `mcp-tools.ts`, `[action].ts` and `smoke.sh` hunk by hunk (the v2 branch and the guard are interleaved in
`getPartialDownload`, `listBundleFiles` and the `index`/`entry` routes), so they are not separable without hand-merging; with the v2 tables empty and the gate closed, the v2 code is inert
and the visible change is the guard (and `known_rev`). If a guard-only PR is wanted first, the guard files are: whole files `schema/031_bundle_index_identity.sql`,
`functions/_shared/bundle-reader.ts`, `functions/_shared/catalog-refresh.ts`, `functions/api/ingest/[table].ts`, `functions/api/projects/[id]/upload/[action].ts`,
`src/test/bundle-reader-guard.test.ts`, `src/test/helpers/zip.ts`; hand-merged hunks of `bundle-core.ts`, `mcp-tools.ts`, `[action].ts` and the offsets block of `scripts/smoke.sh`. W5 has not prepared or tested that subset.
* Schema: `schema/031_bundle_index_identity.sql` (applied), `schema/032_v2_catalog.sql` (applied), `schema/033_v2_sample_rev_key.sql` (**apply to production before this PR deploys**).
* Functions, new: `functions/_shared/{mcp-core,mcp-v2,rev,v2-catalog,v2-cells,v2-gate,v2-packs,v2-serve,v2-urls}.ts`, `functions/api/ingest/v2-study.ts`.
* Functions, changed: `functions/mcp.ts`, `functions/_shared/{bundle-core,bundle-reader,cache,catalog-refresh,cohort-manifest,env,manifest-core,mcp-tools,study-core}.ts`,
  `functions/api/bundle/[gse]/[action].ts`, `functions/api/cohort/manifest.ts`, `functions/api/gse/[id].ts`, `functions/api/ingest/[table].ts`, `functions/api/manifest.ts`,
  `functions/api/projects/[id]/upload/[action].ts`.
* Other: `scripts/smoke.sh`, `package.json` (the `fzstd` dependency hunk) and `package-lock.json`, `README.md` (the `known_rev` sentence), `src/pages/DocsMcp.tsx` (the `known_rev?` inputs only).
* Tests: `src/test/{bundle-reader-guard,mcp-v2,mcp-diet,v2-api,v2-catalog,v2-cells,v2-edge-ttl,v2-packs,v2-serve}.test.ts`, `src/test/helpers/zip.ts`, `src/test/fixtures/**`.
* Before merge: schema 033 applied; `scripts/smoke.sh <preview>` (expect tools/list 14, GSE111113 offsets ok, GSE138867 offsets ok). After merge: `scripts/smoke.sh https://singlet.bio`.
* Environment variables: none required. `MCP_DIET=1`, `V2_TOOLS=1`, `V2_ORIGIN` optional.

**PR 3: website UI (CS5).** Merge after PR 2 (it reads `series.v2`, absent until a study is served, so it is inert before that).
* `src/components/DownloadPanel.tsx`, `src/components/study/FileTreePanel.tsx`, `src/pages/StudyDetail.tsx`, `src/integrations/api/types.ts`, `src/integrations/api/client.ts`, `src/test/download-panel.test.tsx`.

**Follow-up commit (hold until the first public v2 study):** `git apply --ignore-whitespace bench/compression/notes/m_w5_docs_followup.patch` in the website tree (Docs.tsx, DocsMcp.tsx, README.md copy for per-sample
studies, the three tools, "17 tools"). Add the CHANGELOG entry in the same commit.

Not committed and not in any PR: nothing under `C:\Users\zacha\Projects\singlet-website` was touched; no deploy, no remote D1 write was made by W5.

---------------------------------------------------------------------------------------------------------------------------------

## 7. Mirror into singlet-cloud (read only; branch main at f227fc241, 2026-10-20; not re-compared in round 2)

singlet-cloud binds the same D1 and the same bucket, so the schema files are applied once to the shared database; the cloud repo only needs the SQL files for the record.
The cloud has its own `030_cloud_beta.sql` and `031_cloud_tools.sql`: the same numbers as the website's `030_search_facts.sql` and my `031`, different names and tables, so no content
clash; keep the file names exactly as in the website (`031_bundle_index_identity.sql`, `032_v2_catalog.sql`, `033_v2_sample_rev_key.sql`). `workers/data-v2` is deployed once from the website repo (the cloud dropped `workers/`); do not copy it.

| file | cloud copy vs the website baseline | straight copy? |
|---|---|---|
| `functions/_shared/bundle-reader.ts`, `catalog-refresh.ts`, `api/gse/[id].ts`, `api/manifest.ts`, `components/study/FileTreePanel.tsx` | identical apart from line endings | yes (but `api/gse/[id].ts` and `api/manifest.ts` import `v2-urls.ts`, new) |
| `functions/_shared/manifest-core.ts` | identical apart from line endings; needs `v2-serve.ts` (new) | yes, with the new files |
| `functions/_shared/bundle-core.ts` | cloud lacks the `publicEntries` filter (3 lines) | **no**: the cloud does not import `./restricted` |
| `functions/_shared/mcp-tools.ts` | cloud lacks the `publicEntries` filter and the `completeness` lines (7 lines) | merge by hand |
| `functions/api/bundle/[gse]/[action].ts` | cloud lacks `publicEntries` and has the edge key `v=1` (5 lines) | merge by hand; its `index` key is v=1 there, bump to v=3 as well; also needs the `CACHE_TTL_HEADER` handling in `cache.ts` |
| `functions/_shared/cache.ts` | not compared | add the `CACHE_TTL_HEADER` hunk (one constant, `cachedJson` and `bypass`) |
| `functions/mcp.ts` | cloud INSTRUCTIONS and descriptions are worded differently (11 lines) | merge by hand: the INSTRUCTIONS paragraphs and a few descriptions conflict; the gate (`v2-gate.ts`) and `toolList({v2, diet})` shape is new |
| `functions/_shared/env.ts` | cloud adds `ADMIN_TOKEN` | merge (add `MCP_DIET`, `V2_TOOLS`, `V2_ORIGIN`) |
| `functions/_shared/study-core.ts` | 26 lines differ (completeness, restricted wording); the cloud has no `completeness.ts` | merge by hand: add the `v2` block, the `bundle_url` rule, `hasFile` |
| `functions/api/ingest/[table].ts` | 70 lines differ | apply only the `index-next` hunk (StaleBundleError retry) by hand |
| `functions/api/projects/[id]/upload/[action].ts` | not compared | one line: `{ noRedirects: true, userHost: true }` |
| `functions/_shared/cohort-manifest.ts`, `functions/api/cohort/manifest.ts` | **absent in the cloud** (no cohort API there) | skip until the cloud has the cohort feature |
| new: `mcp-core.ts`, `mcp-v2.ts`, `rev.ts`, `v2-catalog.ts`, `v2-cells.ts`, `v2-gate.ts`, `v2-packs.ts`, `v2-serve.ts`, `v2-urls.ts`, `api/ingest/v2-study.ts`, `schema/031_*.sql`, `schema/032_*.sql`, `schema/033_*.sql` | n/a | copy, but **`v2-catalog.ts` and `v2-serve.ts` import `./restricted`, which the cloud does not have** |
| `package.json` / `package-lock.json` | cloud has another license field, other dependencies, no `enrich:*` scripts | add `fzstd` 0.1.1 only; regenerate the lock there |
| `scripts/smoke.sh` | cloud smoke lacks the cohort, restricted, layer and library checks; its tool count is 14 | add the offsets block (with its two skips); the count check accepts 14 or 17 |
| `src/components/DownloadPanel.tsx`, `src/integrations/api/{client,types}.ts`, `src/pages/{Docs,DocsMcp,StudyDetail}.tsx` | different product UI (83 to 146 lines apart) | do not copy; port by hand if the cloud shows downloads |
| tests | self contained except `mcp-v2.test.ts` / `mcp-diet.test.ts` / `v2-api.test.ts` / `v2-edge-ttl.test.ts` (read the schema files `001_init_local_nofts.sql`, `010`, `012`, `032`, `033`) and the helpers `helpers/zip.ts`, `bitindex-fixture.ts`, `d1-sqlite.ts` | copy with the helpers |

The one decision that blocks a straight copy: the cloud has no `restricted.ts` (the website's public filter for per-donor genotype layers, PR #47). The v2 ingest refuses restricted
members and the v2 listings filter them. If the cloud is meant to serve those layers to controlled-access accounts, `v2-catalog.ts` and `v2-serve.ts` need a flag instead of a hard
filter; if not, mirror `restricted.ts` first.

---------------------------------------------------------------------------------------------------------------------------------

## 8. How to resume / what is not done

* Nothing is committed. Order: PR 1 (and redeploy the Worker), apply schema 033, PR 2, PR 3, then the docs follow-up with the first public study.
* Repair of existing stale `bundle_index` rows (w4_cf 0.3 to 0.5, 0.7): done by the orchestrator, not part of W5.
* Not verified live: the new Worker fill rule on the edge (unit tests only); the Cloudflare plan limits (CPU, subrequests, D1 row-write cap); `find_cells` against real v2 objects
  (only the Python-written fixture and the unit tests); a real ingest POST (the first one reports `rows_written`; compare it with the table above); `scripts/smoke.sh` after the round-2 edits (syntax checked with `bash -n`).
* Not done: Browse download links, the catalog retirement checklist (section 5), the singlet-cloud mirror, the docs follow-up.
* The Python-written fixtures pin the 256-byte pz2 header and the CEL2 layout of 2026-10-22; if W1 or W2 change either, regenerate with the two generator scripts and fix
  `parsePz2Header` / `v2-cells.ts`.

---------------------------------------------------------------------------------------------------------------------------------

## 9. Retirement (2026-10-22): the legacy file can be deleted without a dead link or a 404

Worktree `C:\Users\zacha\Projects\singlet-website-v2b`, branch `feat/singlet-v2-retire`. Nothing committed or deployed.

**Base commit deviation (read first).** The brief said branch from `origin/main`. In the shared object store `origin/main` is still 0dcf4914a (nobody fetched), which does not
contain PR #51, and the only git allowed was `worktree add`, so no fetch. GitHub says the merge commit ed2f776 has parents 7586084c1 (main) and c54fcc578 (the head of
`feat/singlet-v2`); main moved from 0dcf4914a to 7586084c1 only by two `chore(hpc-dashboard)` snapshots (`public/data/hpc/**`). The worktree is therefore cut from **c54fcc578**: the
code of ed2f776 minus those three data files. `git branch --unset-upstream` was run and answered "no upstream" (nothing to unset). To rebase: nothing in this change touches `public/data/hpc`.

State of the checks: `npm run check` green (46 files, 687 tests: 662 before, +22 `retired-study`, +3 `browse-links`), `npm run build` green. Not run: a browser, a deployed preview.

### What "retired" means here, and the rule everything follows

A retirement deletes `data/<GSE>/<GSE>.singlet` and flips `v2_migration.status` to `retired` (W6's `retire-sql`). It leaves every catalog row of the legacy file in place
(`gse.r2_bundle_key`, `gse.r2_bundle_bytes`, `bundle_manifest`, `bundle_index`, `sample_qc`, `gse_meta.has_bundle = 1`). So:
* **No response may name the legacy URL or key of a retired study.** One predicate, `isLegacyAlive(migration)` (`v2-catalog.ts`: status != 'retired'), decides, and every URL goes through
  `legacyUrl` / `downloadUrl` in the new `functions/_shared/file-facts.ts` (the legacy URL builder moved to `data-urls.ts`, still re-exported from `study-core.ts`).
* **The catalog rows are the source of truth after retirement** (search, statistics, sample QC, layer counts); the file is not.
* **A study that never had a legacy file** (no `gse.r2_bundle_key`) is a v2-only study. Convention for W6: post it with `migration.status = "retired"` and `verify_ok: true` ("retired" =
  no legacy object exists), otherwise `legacy_available` would say true and `export_manifest` would offer a legacy URL. The API still never emits a legacy URL for a study with a v2 block and no key.

### Emission sites (every place a legacy URL, key or size could leave the site)

| where | before | now |
|---|---|---|
| `GET /api/gse/:id`, MCP `get_study` (`study-core.ts`) | `bundle_url` null when retired (PR #51); `bundle_key`, `r2_bundle_key` and the legacy size still sent | `bundle_key`, `r2_bundle_key` null; `bundle_bytes`/`r2_bundle_bytes` = v2 size; reference build, pipeline version, packed-at and sample count fall back to the v2 study when `bundle_manifest` has no row |
| `GET /api/search` and `/api/nl-search` rows, MCP `search_datasets` | `bundle_key`/`bundle_bytes` of the legacy file; MCP `bundle_url` built from the accession for every row with `has_bundle` | rows carry `v2` (manifest URL, rev, legacy_available); a retired row has `bundle_key` null and `bundle_bytes` = v2 size; MCP `bundle_url` null, plus `format` and `manifest_url`. One batched read per result page (`attachFileFacts`; none when no study is migrated) |
| `GET /api/gse` list | raw `r2_bundle_key` | `v2` block, key null and v2 size when retired; `has_bundle` filter counts a served v2 study as having a file |
| MCP `find_matched_controls` | `download_url: bundleUrl(...)` for every candidate | manifest URL and v2 size for a retired candidate, `format` added |
| saved cohorts (`getCohort`, `/cohorts/:id` page) | page built `data/<GSE>/<GSE>.singlet` for every item and in the TSV | each public item carries `file_url` and `format`; the page uses it |
| Browse: study card, table row, selection bar (curl lines, "Download N files"), cohort page, study page JSON-LD | `bundleUrl(accession)` | `studyFileUrl(accession, row.v2)`: the single file while it exists, else the manifest (with a tooltip and a comment line in the curl block). The study page also treated a null `bundle_url` as "no file": hasBundle is now `bundle_url || v2` (it hid the pipeline row and the JSON-LD distribution for a retired study) |
| export manifest (REST, MCP), cohort manifest, `get_download_url`, `list_bundle_files`, `get_partial_download`, `/api/bundle/*` | already v2-aware in PR #51 | unchanged (checked by the fixture test); `n_samples_in_file` and `reference_build` of a manifest row fall back to the v2 study when there is no `bundle_manifest` (tested) |
| `find_cells` sample selection (`mcp-v2.ts`) | treated a study with a `v2_study` row and **no migration row** as served (a bug against the served rule) | requires a migration row that is verified or retired |

### What reads the legacy file, and what it does now

| reader | after a retirement |
|---|---|
| `loadSampleQc` (MCP `get_sample_qc`, `assess_study`, `/api/bundle/:gse/samples`) when `sample_qc` has no rows | a study served from v2 is read from its sample objects: `summary.json` member of each `.s2` (kind 3, zstd, `fzstd`), shaped like the legacy summaries, stored in `sample_qc`, usable flag from the v2 matrix sizes (`v2-summaries.ts`). Studies with QC rows read D1 as before |
| `index-next` crank | the legacy pending set leaves retired studies out; a second set (`PENDING_V2_SQL`) takes served studies with no QC rows whose legacy file is retired or never existed and indexes them from the sample objects; `remaining` counts both. `POST index-bundle` does the same for a retired study (`source: "v2"`) |
| `refresh-next` crank | `MANIFEST_PENDING` skips retired studies (no HEAD of a deleted file, no park row); `USABLE_NO_FILE` skips every served study, so a v2-only study is never resolved to `usable = 0` ("no file"); a new phase measures usable flags for served studies from `v2_sample.members_json` (exon + intron matrix bytes > 400 and cells called > 0), counted as `sample_usable_v2`; the cron workflow already loops until every count is 0 |
| `gse_meta.has_bundle` (`recomputeMeta`) | 1 for a legacy key **or** a served v2 study; reference mismatch uses `v2_study.reference_build` when there is no `bundle_manifest`. A final ingest post of a served study with `has_bundle = 0` queues its meta row (one `UPDATE ... WHERE has_bundle = 0`, zero rows written for a study with a legacy file). Studies with a legacy key keep `has_bundle = 1` after retirement, so search, `DOWNLOADABLE_SQL`, the headline counts, the nightly bit index (`downloadable`) and the sitemap keep them |
| layer statistics (`/api/stats/layers`) | retired studies still count through their `bundle_index` row (left in place); studies served from v2 with no `bundle_index` row are counted from `v2_sample.members_json` |
| `/api/bundle/:gse/index?refresh=1` (the reindex workflow) | answers from the v2 catalog for a served study; never touches the deleted file |
| edge cache | answers about a study with any v2 row live 5 minutes (PR #51); answers cached before its first v2 row live up to 24 h, so keep the legacy file at least 24 h after the study becomes `verified` |
| all of the above on a database without the v2 tables | the v2-aware SQL is used only after `v2Available(db)`; otherwise the original statements run (test: "a database without the v2 tables") |

### Tests

* `src/test/retired-study.test.ts` (22): the smoke check on fixtures. A D1 stand-in holds a retired study (GSE1, two samples), a verified study whose legacy file is alive (GSE3) and one never moved (GSE2),
  with the catalog rows a retirement leaves behind; `fetch` serves the v2 objects and records every request. Every endpoint (`/api/gse/:id`, `/api/search` studies and samples, `/api/gse`, `/api/manifest` json/tsv/curl,
  `/api/bundle` index/entry/forced refresh, `/api/cohort/manifest`, `/sitemap.xml`, saved cohort items) and every MCP tool that names a file (`get_study`, `get_download_url`,
  `list_bundle_files`, `get_modalities`, `get_sample_qc`, `assess_study`, `compare_studies`, `get_partial_download`, `search_datasets`, `find_matched_controls`, `export_manifest` in six formats) is searched
  for the legacy URL of the retired study, and no request to a `/data/` URL may be made; the other two studies must still show theirs. Then the second group: summaries from the objects, `index-next` and
  `index-bundle` v2 passes with no `/data/` request, the refresh crank on a v2-only study, `recomputeMeta`, the database without the v2 tables, `/api/gse?has_bundle`, layer counts.
* `src/test/browse-links.test.ts` (3): `studyFileUrl`, the search-row mapping, the selection bar's curl lines.
* `src/test/v2-catalog.test.ts`: the unchanged-rev post now also issues the queue marker for a served study (expectation updated).
* `scripts/smoke.sh`: `SMOKE_RETIRED="GSE… GSE…"` runs the same check against a deployed site for studies that really are retired: detail has no bundle URL and says `legacy_available: false`, no legacy name anywhere in
  detail, bundle index, search row and manifest export, the manifest object answers, and the old object is a 404. With it empty nothing runs. Dry-run against production with GSE145370 (verified, not retired):
  the retired checks FAIL as they should and every other check passes. `bash -n` clean.

### Docs and CHANGELOG

`m_w5_docs_followup.patch` applied as it was (README, Docs.tsx, DocsMcp.tsx: the "17 tools" text, the per-sample layout, the three tools). CHANGELOG.md has a new entry (2026-10-22): the layout, what the API and MCP show, the ETag/`If-Match` recipes,
and what happens when a single file is retired. It is a draft for the owner: it says the Python and R packages read the layout "from the release that includes the per-sample reader" (singlet-bio cfaeb90 has the reader; I did not check a release),
and the docs text of the patch still says `singlet.load` reads both layouts the same way: that holds only for a package release that has cfaeb90. Edit both if the release is later than the docs.

### Order of operations for the first real retirement (W6 / orchestrator)

1. **Flip D1 to `retired` before deleting the object**, not after (W6's `retire` deletes first and `retire-sql` flips afterwards; between the two the site would still emit a dead URL until the flip). Flipping early is safe: v2 serves everything
   and the site stops naming the legacy URL within 5 minutes (edge TTL). Only an old client pinned to `bundle_url` is affected, and the file still exists at that point. Suggested: archive-copy and verify, run the SQL, wait 5 minutes, delete.
2. Keep the legacy file at least 24 h after the study became `verified` (older cached `/index` and `/entry` answers), and as long as the D12 grace period says.
3. After the first real retirement run `SMOKE_RETIRED="<GSE>" bash scripts/smoke.sh https://singlet.bio`.
4. A study that never had a legacy file: post it as `retired` with `verify_ok: true` (see above).
5. The retire SQL could also set `gse.r2_bundle_bytes` to the v2 size so the file-size filter and sort (`max_file_bytes`, `file_size`, facets) use it: those still read the legacy size from `gse.r2_bundle_bytes` for a retired study (display fields are fixed in the API,
   the SQL filters are not). Not done on purpose: it needs a write to a table the retirement does not otherwise touch.

### Not done

* The Python and R clients (singlet-bio, W4) read `bundle_url`/`v2` from `/api/gse/:id`; they are not part of this worktree. With `bundle_url` null and `v2.legacy_available: false` they must use the manifest (cfaeb90 does).
* The size filter/sort and facet `file_size` use `gse.r2_bundle_bytes` (item 5 above). Layer statistics for a v2-only study are one LIKE scan over `v2_sample.members_json` every 12 h (computed and cached, empty today).
* No browser check of the Browse links. No deploy. The singlet-cloud mirror of this section is not prepared (new files: `data-urls.ts`, `file-facts.ts`, `v2-summaries.ts`; changed: `catalog-refresh.ts`, `search-core.ts`, `ingest/[table].ts`, `bundle-core.ts`, `study-core.ts`, `product.ts`, `layers-core.ts`, `api/gse/index.ts`).

### Proposed commits / PRs for this section

One PR from `feat/singlet-v2-retire` (the changes share `catalog-refresh.ts`, `search-core.ts`, `study-core.ts` and the tests; the only independent piece is the docs/CHANGELOG pair, which can be its own first commit):
1. `Docs and CHANGELOG for the first v2 study`: `README.md`, `src/pages/Docs.tsx`, `src/pages/DocsMcp.tsx`, `CHANGELOG.md`.
2. `Retirement: no legacy URL for a retired study, summaries and flags from v2, has_bundle from v2`: everything else listed by `git status` in the worktree (new: `functions/_shared/{data-urls,file-facts,v2-summaries}.ts`,
   `src/test/{retired-study,browse-links}.test.ts`; changed: `functions/_shared/{bundle-core,bundle-reader,catalog-refresh,layers-core,manifest-core,mcp-tools,mcp-v2,product,search-core,study-core,v2-catalog,v2-serve}.ts`,
   `functions/api/gse/index.ts`, `functions/api/ingest/{[table],v2-study}.ts`, `functions/mcp.ts`, `scripts/smoke.sh`, `src/components/browse/{SelectionBar,StudyCard,StudyTable,useSelection}.ts(x)`,
   `src/integrations/api/{client,types}.ts`, `src/pages/{CohortDetail,StudyDetail}.tsx`, `src/test/v2-catalog.test.ts`).
No schema change, no new binding, no Pages variable.
