# "v2" branding inventory: zdebruine/singlet-website (origin/main, 549d67d81)

Source: `git grep` in a checkout of singlet-website origin/main. 780 tracked files. Counts are matching lines (`git grep -c`) unless marked "occ" (occurrences, `-o`). Read-only; nothing edited.

Headline facts
- The format is "v2" everywhere in code: file names (`v2-*.ts`), identifiers (`V2State`, `StudyV2`, `getV2State`...), D1 tables (`v2_*`), R2 prefix (`v2/`), URL path (`/v2/`), Worker name (`data-v2`, `singlet-data-v2`, header `X-Served-By: data-v2`), JSON value `"singlet-v2"`, JSON key `v2`, env vars `V2_ORIGIN` / `V2_TOOLS`, ingest endpoint `/api/ingest/v2-study`.
- No `/api/v2/...` generic API version exists. No `called_v2` anywhere in this repo (0 hits). No `scripts/**` file has v2 in its name. No CI workflow mentions v2 or data-v2 (the Worker is "deployed by the owner, never by CI").
- This repo contains NO copy of `docs/format-v2*.md` or `docs/CELL-ARCHITECTURE.md` (the `docs/` dir has only COHORT-QUERY, COMPLETENESS-REPORT, DATA-AUDIT, DENSE-METADATA, HANDOFF, LANDING, QC, SEARCH-ROADMAP, launch-runbook). Those are only cited by path in code comments (see 5).
- Many code comments cite singlet-bio paths that will also need renaming on that side: `docs/format-v2.md`, `docs/format-v2-cell-index.md`, `docs/format-v2-cell-index-costs.md`, `docs/v2-migration.md`, `python/singlet/v2/{cells,client,codec,packs,atlas}.py`, `singlet.v2.atlas.*` (Python module path, also emitted in user-facing code snippets, see 1), `r/R/v2.R`, `scripts/v2_cidx_build.py`.

---

## 1. Workers

Files in `workers/data-v2/` (line counts with "v2"): `README.md` 28, `src/handler.ts` 24, `src/cidx.ts` 17, `scripts/local.mjs` 7, `src/routes.ts` 7, `src/gather.ts` 6, `wrangler.toml` 8, `src/index.ts` 2, `src/limits.ts` 2, `src/scx.ts` 1. Also `workers/data-v2/.gitignore` and `tsconfig.json` (path only; the directory name itself is `data-v2`, 12 tracked files). Other worker: `workers/alpha-proxy` has no v2 reference.

`workers/data-v2/wrangler.toml` relevant lines (verbatim):
```
# data.singlet.bio/v2/*: read-only front for the v2 objects in R2 (docs/format-v2.md). See README.md.
name = "singlet-data-v2"
main = "src/index.ts"
compatibility_date = "2026-09-01"   # newest date the pinned wrangler 4.102 accepts; deployed 2026-10-22
account_id = "2f8a8cc95b6bc0be84ac611fe2e5c3c2"
workers_dev = false
#   deploy, then `curl -sI https://data.singlet.bio/v2/ref/index.json` must show `x-served-by: data-v2`.
#   routes = [{ pattern = "dv2.singlet.bio", custom_domain = true }]        (commented fallback)
# and point clients (website Pages variable V2_ORIGIN, Python/R loaders) at https://dv2.singlet.bio/v2/.   (comment)
routes = [{ pattern = "data.singlet.bio/v2/*", zone_name = "singlet.bio" }]
services = [{ binding = "SELF", service = "singlet-data-v2" }]        # self service binding names the Worker
[[r2_buckets]]
binding = "DATA"
bucket_name = "singlet-data"
[[d1_databases]]
binding = "DB"
database_name = "singlet-catalog"
database_id = "fb42f359-92a0-452c-91e4-f630f627c4c7"
[[ratelimits]]  name = "CIDX_RL_ANON" namespace_id = "7301"   (and CIDX_RL_KEY "7302")
[[durable_objects.bindings]]  name = "CIDX_QUOTA"  class_name = "CidxQuota"
[[migrations]]
tag = "cidx-quota-v1"        # "v1" here is a DO migration tag, not this format
new_sqlite_classes = ["CidxQuota"]
```
Branding items inside the Worker:
- Worker name `singlet-data-v2` (wrangler.toml:3 and the `SELF` service binding at :23; also `scripts/local.mjs:34,40`). Renaming the Worker name changes the deployed Worker and the SELF service binding must change with it. The R2 bucket (`singlet-data`) and D1 db (`singlet-catalog`) bindings carry no v2.
- Route pattern `data.singlet.bio/v2/*` (wrangler.toml:17). Prefix is also hard-coded in code, see 2.
- Header `X-Served-By: data-v2`: `src/handler.ts:257`, `src/routes.ts:27`; checked in `src/test/data-v2-worker.test.ts`; documented in README:27,131 and wrangler.toml comment; `console.error("data-v2: ...")` log prefixes at `handler.ts:356,397,455,551` and `routes.ts:110` ("data-v2 cidx:").
- Route path handling: `handler.ts:99-110` (nine regexes, see 2), `routes.ts:10` `CIDX_PATHS = new Set(["/v2/cidx/query", "/v2/cidx/gather", "/v2/cidx/_part"])`, `routes.ts:61,69` (equality checks), `routes.ts:100` (`self.fetch("https://data.singlet.bio/v2/cidx/_part", ...)`: internal fan-out URL with the host and v2 prefix hard-coded).
- Python snippets emitted to API users from `src/cidx.ts:926,928,929`: `from singlet.v2 import atlas` (user-facing text in query answers; needs the Python package rename).
- Package/module references in comments: `singlet.v2.atlas.*` (cidx.ts 8,394,404,415,513,616,826,841,1112; gather.ts:100,293), `python/singlet/v2/atlas.py` (scx.ts:2).
- Other worker/config references: `package.json:13` script `check` ends with `tsc -p workers/data-v2/tsconfig.json --noEmit` (directory path). `wrangler.toml` (root, Pages): no v2. `tsconfig*.json`, `vite.config.ts`, `vitest.config.ts`: no v2 (not even `workers`). Pages env var `V2_ORIGIN` is declared in `functions/_shared/env.ts:65` but not set in root wrangler.toml (set in dashboard, if at all).

## 2. R2 key prefixes, `.s2`, URL bases

Prefix occurrences (occ / files, whole repo excluding package-lock): `v2/s/` 81 / 22, `v2/m/` 54 / 18, `v2/ref/` 23 / 8, `v2/ledger/` 6 / 3, `v2/cidx/` 54 / 9, `v2/a/` 2 / 2, `v2/cells/` 7 / 3, `v2/cf/` 0. No other `v2/<word>/` R2 prefix exists in code (the only other `v2/` hits are python module paths `python/singlet/v2/`, fixture dirs, and Google's `oauth2/v2/auth`, see 10).

Where defined vs scattered. There is NOT a single constant. Two independent definition sites plus many literals:
- Website (Pages Functions): `functions/_shared/v2-urls.ts` builds URLs: `v2Url(key)` = origin + "/" + key; `v2ManifestUrl = v2Url(\`v2/m/${gse}.json\`)` (:24), `v2PackUrl` -> `v2/ref/${id32}.pack` (:25), `v2PackIndexUrl` -> `"v2/ref/index.json"` (:26). `functions/_shared/v2-catalog.ts:40` `manifestKey = (gse) => \`v2/m/${gse}.json\``. `V2_ORIGIN_DEFAULT = "https://data.singlet.bio"` (:9; the `/v2` is in the key not the origin). `configureV2(env)` reads env `V2_ORIGIN` (11 uses of `configureV2` across API handlers: gse/[id], manifest, cohort/manifest, mcp, bundle...).
- Website scattered literals: `functions/_shared/bundle-core.ts:138` (`path: \`v2/m/${gse}.json\``), `functions/_shared/v2-packs.ts:32,481,581` (comments), `v2-summaries.ts:6`, `v2-cells.ts:3`, `v2-serve.ts:4`, `functions/api/ingest/v2-study.ts:14` (doc example `"ledger_key": "v2/ledger/GSE1.jsonl"`), `scripts/smoke.sh:123` (`grep -q "v2/m/$gse.json"`), `src/pages/Docs.tsx:233-234` (user-visible URLs `https://data.singlet.bio/v2/s/<GSM>/<digest>.s2` and `https://data.singlet.bio/v2/m/<GSE>.json`), `CHANGELOG.md:13`.
- Worker: no constant; key regexes hard-coded in `workers/data-v2/src/handler.ts:99-110`:
  - `^\/v2\/s\/GSM\d{1,12}\/[0-9a-f]{32}\.s2$`, `^\/v2\/ref\/[0-9a-f]{32}\.pack$`, `^\/v2\/ref\/index\.json$`, `^\/v2\/m\/GSE\d{1,12}\.json$`, `^\/v2\/cells\/GSE\d{1,12}\.cells$`, `^\/v2\/cidx\/obj\/[0-9a-f]{32}\.(scx|parquet)$`, `^\/v2\/cidx\/obj\/[0-9a-f]{32}\.json$`, `^\/v2\/cidx\/v\/\d{8}T\d{4}Z-[0-9a-f]{8}\.json$`, `^\/v2\/cidx\/current\.json$`, `^\/v2\/a\/GSM\d{1,12}\/[0-9a-f]{32}\.parquet$`
  - `gather.ts:176` `SAMPLE_KEY = /^v2\/s\/GSM\d{1,12}\/[0-9a-f]{32}\.s2$/` (the one that validates keys in gather requests)
  - `cidx.ts:198,209` (`"v2/cidx/current.json"`, `` `v2/cidx/v/${v}.json` ``), `routes.ts:10,61,69,100`, `scripts/local.mjs:47` (`walk(join(root, "v2"), "v2/")` to seed local R2).
- Also stored in D1 data: `v2_study.manifest_key` rows hold `v2/m/<GSE>.json`, `v2_sample.key` rows hold `v2/s/<GSM>/<hex>.s2`, `v2_migration.ledger_key` rows hold `v2/ledger/<GSE>.jsonl`. Existing rows and existing R2 objects keep the old keys unless re-keyed/copied (data migration question, not code).
- `.s2` extension: 93 occ total; code: `workers/data-v2/src/handler.ts` (2), `gather.ts` (1), `functions/_shared/v2-catalog.ts` (3, key regex validation of manifest sample keys), `v2-urls.ts` (2), `v2-summaries.ts` (1), `functions/api/ingest/v2-study.ts` (1), `functions/api/ingest/[table].ts` (1), `schema/032_v2_catalog.sql` (1), `src/pages/Docs.tsx` (1). Rest are tests/fixtures (`src/test/data-v2-worker.test.ts` 34, `fixtures/cidx/expected.json` 11, etc.; binary fixture `src/test/fixtures/v2/sample/GSM1000001.s2`).
- URL base `data.singlet.bio/v2`: 17 hits outside tests (workers/data-v2 README/wrangler/routes, v2-urls.ts, Docs.tsx, CHANGELOG, `src/test/*`). The generic base `DATA_BASE = "https://data.singlet.bio"` is also defined at `functions/_shared/data-urls.ts:10` and `src/lib/install-snippets.ts:28` (not v2-branded; legacy paths `/data/<GSE>/<GSE>.singlet`, `/search-index/v1/` are separate).
- Pack side-info file extension `.pack`, `.scx`, `.cells`, `.parquet` are not v2-branded.

## 3. D1 (schema `schema/032_v2_catalog.sql`, `schema/033_v2_sample_rev_key.sql`)

Tables
- `v2_study` (PK `gse_id`): columns `gse_id, manifest_key, rev, bytes, n_samples, created_at, legacy_bytes, manifest_bytes, manifest_sha256, n_cells_called, reference_build, packs_json, pipeline_json, updated_at`. Only the table name carries v2; `manifest_key` values contain `v2/m/`.
- `v2_sample` (PK `(gse_id, gsm_id)` in 032; 033 recreates it, key per its title is `(gse_id, rev, gsm_id)`): columns `gse_id, gsm_id, key, bytes, sha256, n_columns, n_called, n_members, median_umi, median_genes, modalities, members_json, cell_stats_json, rev, updated_at`. `key` values contain `v2/s/`.
- `v2_migration` (PK `gse_id`, status CHECK in pending/running/uploaded/verified/failed/retired): columns `gse_id, status, legacy_key, legacy_etag, legacy_bytes, v2_bytes, n_samples, members_by_kind, verify_ok, commit_sha, migrated_at, verified_at, ledger_key, error, updated_at`. **Column with v2 in its name: `v2_bytes`** (the only one in any table).
- Temporary: `v2_sample_old` (created and dropped in 033).
Indexes: `idx_v2_sample_gsm` (on `v2_sample(gsm_id)`; created in 032, dropped and recreated in 033), `idx_v2_migration_status` (on `v2_migration(status)`). Views: none. Triggers: none.
Other v2-named D1 artefacts: `schema/033_v2_sample_rev_key.sql` migration performs `ALTER TABLE v2_sample RENAME TO v2_sample_old`, so a rename migration for 1.0 can follow that pattern. File names `032_v2_catalog.sql`, `033_v2_sample_rev_key.sql` are themselves branded and are read by tests by name (`src/test/mcp-diet.test.ts:25`, `src/test/retired-study.test.ts:45`, `v2-catalog.test.ts`, `v2-api.test.ts`). `schema/031_bundle_index_identity.sql:5` comment cites `docs/v2-migration.md`. Derived metric key `sample_usable_v2` (counted/returned by catalog refresh, see 4).

Query sites referencing `v2_study` / `v2_sample` / `v2_migration` (non-test, lines with the table name; plus lines with `v2_`prefix in total):
| file | v2_study | v2_sample | v2_migration | all `v2_` lines |
|---|---|---|---|---|
| functions/_shared/v2-catalog.ts | 12 | 6 | 14 | 36 (main DAO: getV2State, getV2Study, getV2Sample(s), listV2Samples, v2ServedStudies, v2Available, upsert at :450-467) |
| functions/_shared/catalog-refresh.ts | 4 | 1 | 3 | 6 (`USABLE_V2_PENDING` :206, :317) |
| functions/api/ingest/v2-study.ts | 3 | 2 | 0 | 6 (POST/GET handler) |
| functions/api/ingest/[table].ts | 1 | 1 | 3 | 4 (`PENDING_V2_SQL` :538, :685, :716) |
| functions/_shared/v2-gate.ts | 2 | 0 | 1 | 2 (`SELECT ... FROM v2_study` visibility gate) |
| functions/_shared/v2-summaries.ts | 0 | 3 | 0 | 3 |
| functions/_shared/layers-core.ts | 1 | 1 | 1 | 3 (:50-52 join) |
| functions/_shared/mcp-v2.ts | 1 | 1 | 1 | 3 (:407 `EXISTS (SELECT 1 FROM v2_migration m ...)`, :461-462) |
| functions/api/gse/index.ts | 1 | 0 | 1 | 1 (:156 `id IN (SELECT s.gse_id FROM v2_study s JOIN v2_migration m ...)`) |
| functions/_shared/rev.ts | 1 | 0 | 0 | 1 |
| functions/_shared/data-urls.ts | 0 | 0 | 1 | 1 (comment) |
| scripts/smoke.sh | 0 | 0 | 1 | 1 (comment) |
| functions/_shared/manifest-core.ts | | | | 12 (these are JSON field names `v2_manifest_url`, `v2_rev`, `v2_bytes`, not SQL) |

`v2_bytes` as a column/JSON field: `functions/_shared/v2-catalog.ts` (67, 82 SELECT list, 152, 286 `raw.v2_bytes`/`migration.v2_bytes`, 450-467 INSERT/ON CONFLICT), `functions/api/ingest/v2-study.ts:13` (example), `functions/_shared/manifest-core.ts` 46,90,102,134 (CSV/TSV export column).
Error-message dependency: `mcp-v2.ts:472` tests `/no such table/i` against the v2_* tables; `mcp-v2.ts:472,478` user strings say "v2 format".

## 4. Pages Functions / API / MCP

Endpoints with v2 in the path
- `POST /api/ingest/v2-study` and `GET /api/ingest/v2-study?gse=GSE...` (`functions/api/ingest/v2-study.ts`, 24 lines; Pages route is by file name). Also called in tests `src/test/retired-study.test.ts:114`, `mcp-diet.test.ts:165`, `v2-api.test.ts`. Header comment lines 2-3 name the endpoint and `schema/032_v2_catalog.sql`.
- `functions/api/ingest/[table].ts` is a generic ingest route (table = param); it is not v2-branded but has 33 v2-mentioning lines (comments, `PENDING_V2_SQL`, `readV2Summaries`, `applyUsableV2`, result `source: "v2"` at :312).
- **No `/api/v2/...` endpoints and no generic API-version "v2" in `functions/`.** The only external `/v2/` path in code is Google OAuth `https://accounts.google.com/o/oauth2/v2/auth` (`functions/_shared/oauth.ts:117`, false positive) and `functions/api/explain.ts:39` `CACHE_VERSION = "v2"` (explain-cache key version, unrelated).

JSON field names and values (user-visible API/MCP contract)
- Value `"singlet-v2"` (format discriminator; 75 occ repo-wide, non-test ~35): `functions/_shared/bundle-core.ts:102,139`, `cohort-manifest.ts:237,261 (union "singlet-v1" | "singlet-v2"),354,373`, `file-facts.ts:17,27`, `manifest-core.ts:41-42,99,163`, `mcp-tools.ts:978`, `mcp-v2.ts:52 (const FORMAT = "singlet-v2", interpolated into text "format ${FORMAT}")`, `v2-catalog.ts:318` (validates incoming manifest: error text `manifest.format must be 'singlet-v2'`; also incoming manifests in R2 `v2/m/*.json` carry `"format": "singlet-v2"`), `functions/api/bundle/[gse]/[action].ts:181`, `functions/mcp.ts:96,925`, `functions/api/cohort/manifest.ts:63`, `functions/api/ingest/v2-study.ts:22`, `src/integrations/api/client.ts:437,439,1040`, `src/integrations/api/types.ts:330,660`, `src/components/study/FileTreePanel.tsx:77`, `src/pages/Docs.tsx:449`, `scripts/smoke.sh:73,118` (greps `"format":"singlet-v2"`). The counterpart `"singlet-v1"` is used for the legacy `.singlet` single file (cohort-manifest.ts:261, manifest-core.ts:42, mcp.ts:695, mcp-tools.ts:637,978, product.ts:471).
- Key `v2` (object, `StudyV2`: `{format, rev, manifest_url, n_samples, bytes, legacy_available}`) in `GET /api/gse/:id` as `series.v2`, in `/api/search` and `/api/gse` list rows as `v2`, in MCP `get_study` as `v2` (`functions/mcp.ts:925`), `search_datasets` rows. Set at `file-facts.ts:70`, `study-core.ts:50,279`, `functions/api/gse/index.ts:193`, `search-core.ts:119,166`; consumed in frontend (`client.ts:280,451,470`, `StudyCard.tsx:98,101`, `StudyTable.tsx:140,143`, `useSelection.ts:15`, `StudyDetail.tsx:224,256,525`, `DownloadPanel.tsx:51`). Documented in `README.md` (17 tools line), `CHANGELOG.md:13,24`, `src/pages/Docs.tsx:235` ("`series.v2` in `GET /api/gse/:id`"). Also `fileRev` returns `source: "v2" | "legacy"` (`rev.ts:33,36`).
- Cohort manifest TSV/JSON export columns: `v2_manifest_url`, `v2_rev`, `v2_bytes` (`functions/_shared/manifest-core.ts:43-46,88-90,100-102,133-134`; asserted in `src/test/mcp-diet.test.ts:148`).
- Catalog-refresh result key `sample_usable_v2` (`catalog-refresh.ts:248`, `[table].ts:41`).
- No `is_v2` / `has_v2` booleans exist. (Related: `has_bundle` is true if `series.v2` exists, `mcp.ts:942`; `legacy_available` is a v2-block field, not v2-branded.)
- Env vars: `V2_ORIGIN` (`functions/_shared/env.ts:65` + local `interface Env` copies in `api/bundle/[gse]/[action].ts:32`, `api/gse/[id].ts:21`, `api/manifest.ts:25`; read in `v2-urls.ts:14-15`; also named in `workers/data-v2/wrangler.toml` comment) and `V2_TOOLS` (`env.ts:63`, `v2-gate.ts:5,15-16`; value "1" forces v2 MCP tools on). Constants: `V2_ORIGIN_DEFAULT`, `V2_GATE_TTL_MS`, `V2_EDGE_TTL` (bundle action :40), `V2_INDEX_FILE_SAMPLES` (bundle-core), `V2_INSTRUCTIONS`, `V2_ONLY`, `V2_PROPS`.
- Source file names with v2 (`git ls-files`): `functions/_shared/{mcp-v2,v2-catalog,v2-cells,v2-gate,v2-load-modalities,v2-packs,v2-serve,v2-summaries,v2-urls}.ts`, `functions/api/ingest/v2-study.ts`. Exported identifiers (counts of occurrences): `V2ManifestError` 37, `StudyV2` 22, `V2Member` 21, `V2FetchError` 20, `getV2State` 18, `v2ManifestUrl` 17, `V2State` 17, `getV2Sample` 11, `configureV2` 11, `V2SampleFull` 11, `V2MigrationRow` 10, `v2Available` 9, `V2ToolCtx` 9, `v2Url` 8, `studyV2Map` 8, `getV2Samples` 8, `v2ServedStudies` 7, `listV2Samples` 7, `getV2Study` 7, `applyUsableV2` 7, `readV2Summaries` 6, `V2StudyRow` 6, `v2ToolsVisible` 5, `v2MatrixBytes` 5, `v2BundleIndexResponse` 5, `isServedFromV2` 5, `getV2Migration` 5, `V2SampleLight` 5, `V2Presence` 5, `getV2Presence` 4, `V2SummaryTarget` 4, `V2PlanStudy` 4, `loadV2Plan` 3, `normalizeStudyV2` 3, `ManifestFileV2` 3, `V2Facts` 3, `resetV2ToolsGate`, `v2PartialDownload`, `v2ListFiles`, `v2Modalities`, `v2DownloadUrl`, `v2PackUrl`, `v2PackIndexUrl`, `v2OnlyLayerRow`, `studyV2`, `PENDING_V2_SQL`, `USABLE_V2_PENDING`, ...

MCP server (`functions/mcp.ts`, 47 lines; `_shared/mcp-v2.ts` 66; `mcp-tools.ts` 26; `mcp-core.ts` 1). Strings to rewrite, verbatim:
- Server instructions `V2_INSTRUCTIONS` (mcp.ts:96): `Studies are being moved to a per-sample format (format "singlet-v2": one immutable object per sample plus a manifest). For those studies get_download_url returns the manifest, list_bundle_files and get_partial_download give byte ranges inside the sample objects, find_cells searches cells by their per-cell statistics, slice_cells gives the byte ranges of just those cells' matrix chunks, and resolve_features turns row numbers into feature names. Studies not moved yet keep the single .singlet file.`
- `V2_ONLY` (mcp.ts:350): `Only studies already moved to the v2 format are searched or sliced. Not metered.`
- `find_cells` description (:357): `Find cells in v2-format samples by per-cell statistics. ...` + V2_ONLY
- `slice_cells` description (:414): `Byte ranges (HTTP Range on the sample object) to read only some cells of one v2 sample's count matrices: ...` + V2_ONLY
- `resolve_features` description (:435): `Row numbers of a v2 count matrix to feature names (or whitelist column numbers to barcodes). ...`
- `V2_PROPS` input descriptions (:466-481): `v2 studies: also list the sample objects.` / `v2 studies: list each sample object with the byte ranges of only these members (25 samples a page). ...` / `v2 studies: every sample's files, 25 samples a page.` / `v2 studies: only the files these need, with their bytes. ...` / `v2 studies: which samples have these, the singlet.load call for them, and with gsm_id that sample's byte ranges. ...` / `v2 studies: instead of \`file\`, the byte ranges of every member these need. ...`
- Other user-visible text: `mcp-v2.ts:292` `(format ${FORMAT}, rev ${s.rev})`, `:148,:155` manifest/Python hint text; `:309` `${gse} has no samples in the v2 catalog.`; `:472` `No study has been migrated to the v2 format yet, so there are no per-cell tables to search.`; `:478` `No v2 sample matches that sample scope. find_cells searches studies that have been migrated to the v2 format.`; `:570` `${gse} is not in the v2 format yet, so cells cannot be sliced by chunk. Use get_partial_download for the legacy file.`; `:690` `${gsm} is not in ${gse} in the v2 catalog.`
- `functions/api/cohort/manifest.ts:63` note: `Studies with format singlet-v2 (studies[].format): one immutable object per sample. files[].range is an absolute byte range inside samples[].object.url; ...`
- `README.md:109` MCP tool list says "and for studies stored in the v2 per-sample f[ormat]..." (line truncated in this inventory).
- Mechanisms: MCP tool visibility gated by `v2ToolsVisible` (functions/_shared/v2-gate.ts) and `toolList({v2,...})`; `mcp.ts:10,79-80,492` version/comments. The three tools themselves are named `find_cells`, `slice_cells`, `resolve_features` (not v2-branded), though `src/test/mcp-v2.test.ts:115` has `V2_TOOL_NAMES`.
- The singlet MCP server instructions delivered to Claude clients (the `## singlet` block in this session) contain the same text: `Studies are being moved to a per-sample format (format "singlet-v2": ...)`.

## 5. Frontend / docs

Files (lines with v2): `src/integrations/api/client.ts` 16, `src/integrations/api/types.ts` 10, `src/components/DownloadPanel.tsx` 6, `src/components/study/FileTreePanel.tsx` 5, `src/pages/StudyDetail.tsx` 4, `src/pages/Docs.tsx` 4 (+1 `format: "singlet-v2"`), `src/components/browse/{useSelection.ts,StudyTable.tsx,StudyCard.tsx}` 2 each, `src/lib/catalog-display.ts` 2 (false positive: 10xv2).
User-visible copy (no "v2" word is shown on screen except Docs.tsx; the rest is identifiers, but copy says "per-sample"):
- `src/pages/Docs.tsx:233-235`: shows `https://data.singlet.bio/v2/s/<GSM>/<digest>.s2`, `https://data.singlet.bio/v2/m/<GSE>.json`, `series.v2` in `GET /api/gse/:id`; `:449`: `format: "singlet-v2"`.
- `FileTreePanel.tsx:77` `const v2 = index.format === "singlet-v2"`; copy "What's in the study", "Study manifest", "Sample objects" (no v2 word).
- `DownloadPanel.tsx:133`: `Also stored as one file per sample (see the manifest).` ; `StudyCard.tsx:101` / `StudyTable.tsx:143` title `Download the manifest of this study's per-sample files`.
- `src/integrations/api/types.ts:330` (`format: "singlet-v2"` in `StudyV2`), `:659-660` (`format?: "singlet-v2"` on bundle index), `client.ts:437-439,1040` (normalizers), `client.ts:1098-1105` (`isManifestFile`, `studyFileUrl` use the v2 block).
Docs / markdown in this repo mentioning the format: `CHANGELOG.md:13,24` (`series.v2`, `https://data.singlet.bio/v2/m/<GSE>.json`, `legacy_available: false` "in the v2 block"), `README.md:104` (`anvil_v2/` dir, false positive path) and `:109` (MCP tools), `workers/data-v2/README.md` (28 lines; title `# data-v2: Worker for data.singlet.bio/v2/*`, table of keys at :11-19, deploy steps :129-135, fallback host `dv2.singlet.bio` at :135 and wrangler.toml:15-16, local dev :111-112,144-146). Code comments cite singlet-bio docs `docs/format-v2.md` (in v2-catalog.ts:3,299, v2-cells.ts:2, v2-packs.ts:2, v2-serve.ts:3, v2-urls.ts:2, cohort-manifest.ts:228, api/ingest/v2-study.ts:10, schema/032:3, workers/data-v2 files), `docs/v2-migration.md` (bundle-reader.ts:17, schema/031:5, schema/032:3), `docs/format-v2-cell-index.md` / `-costs.md` (workers/data-v2/src/{cidx,gather,handler,limits,scx}.ts, README:82-83). `CLAUDE.md` of this repo has no v2 or CELL-ARCHITECTURE mention (grep clean).
JSON-LD / SEO: `StudyDetail.tsx:256` uses `series.v2?.manifest_url` as `contentUrl`.

## 6. Scripts and tooling

- No script file name contains v2. No `scripts/*` publish/migrate/retire/cidx script exists in this repo (those live in singlet-bio: `scripts/v2_cidx_build.py` is only cited in `workers/data-v2/README.md:103`).
- `scripts/smoke.sh` (11 lines): `:67,72-73,78` (served from v2 layout; greps `"format":"singlet-v2"`), `:100-102,110,118,123` (retired-study checks; `v2_migration.status = 'retired'`; greps `"legacy_available":false`, `"bundle_url":null`, `v2/m/$gse.json`), `:161` ("14 tools, or 17 once a study is served from v2"). Env var `SMOKE_RETIRED` is not v2-named.
- `workers/data-v2/scripts/local.mjs` (7 lines): worker name `singlet-data-v2` (:34,:40), `walk(join(root,"v2"),"v2/")` (:47), comments.
- `package.json:13` `check` script includes `tsc -p workers/data-v2/tsconfig.json --noEmit`; no other npm script mentions v2 (`package.json` has 1 v2 line total). Dependency `wrangler ^4.102.0` is a tool version, not branding.
- Other: `scripts/data-audit/checks.mjs:75` (10x chemistry names), `scripts/dev-api/seed.ts:118,122` (10xv2 aliases), `scripts/seed-local-d1.mjs:83-90,138` (10x Chromium v2) , `scripts/search-eval/{README.md,run.ts}` (prompt version "v2"), all false positives (see 10). `scripts/seed-local-d1.mjs` does NOT create v2_* tables (local seed only), but `schema/002_seed_local.sql` is unrelated.
- Env vars containing V2: only `V2_ORIGIN`, `V2_TOOLS` (see 4). Python/R clients referenced in comments: `singlet.load`, `singlet.v2`, `r/R/v2.R`.

## 7. CI (`.github/workflows/*`)

Zero hits for v2 / data-v2 in `.github/**` (workflows: catalog-crank, ci, dashboard-refresh, enrich-ingest, lockfile, reindex-bundles, search-index, smoke). `ci.yml` runs `npm run check` (which compiles `workers/data-v2` via package.json:13), `npm run build`, `npm run lint`; its `paths-ignore: public/data/hpc/**` is not v2. The Worker is deployed manually (`wrangler.toml:2` comment, README:129). `search-index.yml:4` mentions `search-index/v1/` (index-format version v1, not this format). No data-v2 deploy job exists, so no CI rename is required beyond what `package.json` implies if the directory is renamed.

## 8. Tests and fixtures (v2 line counts)

Tests: `src/test/v2-catalog.test.ts` 129, `retired-study.test.ts` 68, `data-v2-worker.test.ts` 59, `mcp-v2.test.ts` 56, `v2-api.test.ts` 35, `data-v2-cidx.test.ts` 29, `download-panel.test.tsx` 17, `mcp-v2-modalities.test.ts` 15, `v2-serve.test.ts` 16, `v2-edge-ttl.test.ts` 12, `mcp-diet.test.ts` 9, `browse-links.test.ts` 8, `study-card.test.ts` 5, `v2-cells.test.ts` 5, `v2-packs.test.ts` 4, plus 1 each in `hpc-dashboard.test.ts` (path `anvil_v2/latest.json`, false positive), `bundle-reader-guard.test.ts` (2; etag string `"v2"`, false positive), `ingest-index.test.ts` (10; local variables `v1`/`v2`/`v3` = index versions, false positive).
Test file names containing v2 (9): `data-v2-cidx.test.ts`, `data-v2-worker.test.ts`, `mcp-v2-modalities.test.ts`, `mcp-v2.test.ts`, `v2-api.test.ts`, `v2-catalog.test.ts`, `v2-cells.test.ts`, `v2-edge-ttl.test.ts`, `v2-packs.test.ts`, `v2-serve.test.ts`.
Fixtures (path contains v2): `src/test/fixtures/v2/` (cells 12 files, packs 13 files incl. `make_pack_fixtures.py`, sample 3 files incl. `GSM1000001.s2` + manifest + expected, `make_fixtures.py`) and `src/test/fixtures/cidx/v2/cidx/` (9 files; objects `.scx`, `.json`, `current.json`, `v/20261101T0000Z-48871622.json`). `src/test/fixtures/cidx/expected.json` (11 lines with `v2/s/...s2` keys). Fixture contents embed keys `v2/s/...`, `v2/cidx/...` and `"format":"singlet-v2"` (e.g. GSM1000001.manifest.json:1, packs/index.json:1, cidx current.json:1), so renaming the prefix requires regenerating or editing fixtures and their golden answers (fixtures generated by singlet-bio `python/singlet/v2/atlas.py`, the cidx ones check golden answers from it). Tests read `schema/032_v2_catalog.sql` and `033_v2_sample_rev_key.sql` by name (mcp-diet.test.ts:25, retired-study.test.ts:45). Tests assert header `x-served-by: data-v2` and URL `https://data.singlet.bio/v2/m/<GSE>.json`.

## 9. `called_v2`

0 occurrences in this repo (case-insensitive `git grep -c -i called_v2` is empty). The related word "called" appears as a v2_sample column `n_called`, a `v2_study.n_cells_called` column, and an MCP `find_cells` filter name `called` (these refer to cell calling and are not "called_v2"). Nothing to list from this repo; `called_v2` lives in singlet-bio.

## 10. False positives (do not rename)

- `package-lock.json`: 29 lines, all base64 `integrity` hashes containing the letters "V2" (e.g. line 260 `...XV2Cy3W8...`). Not branding. (`yarn.lock` absent.)
- 10x chemistry / protocol "v2" (v3/v4 siblings): `schema/002_seed_local.sql` 9 (`10x Chromium v2`, facets JSON), `scripts/seed-local-d1.mjs` 9, `scripts/dev-api/seed.ts` 2 (`10xv2`, `10x-3p-v2`), `src/lib/catalog-display.ts` 2 (`"10xv2": "10x 3' v2"`), `scripts/data-audit/checks.mjs` 1, `docs/DATA-AUDIT.md` 1 (`10xv2`), `functions/_shared/search-core.ts:720` (`LIKE '%v2%' THEN '10xv2'`), `functions/_shared/facets-core.ts:184` (SQL), `functions/_shared/interpret.ts:295,795` (stop-word list "v1","v2","v3","v4" and "v2/v3 with prime"), `public/vignettes/census-vs-singlet/index.html:48` (`10x 3′ v2`), `scripts/enrich/sample-run/{cards.jsonl 26, summaries.tsv 26, full/unplaced-values.tsv 10, sample-lines.jsonl 2}` (`10x 3' v2` data, 64 lines).
- Smart-seq2 etc. and other assay names: none containing "v2" besides above; Smart-seq2 appears as `Smart-seq2` (docs/interpret.ts), not matched as v2 branding.
- Other unrelated "v2" meanings: search prompt/translator versions (`scripts/search-eval/README.md` 6, `run.ts` 2 [`--prompts v1,v2`], `docs/SEARCH-ROADMAP.md` 6 [NL translator v2, MCP tools v2, `?v=2`], `roadmap.md` 1 [interpreter v2], `docs/COMPLETENESS-REPORT.md` 2 [barcode test v2]); `functions/api/explain.ts:39` `CACHE_VERSION = "v2"`; Google OAuth endpoint `functions/_shared/oauth.ts:117` (`o/oauth2/v2/auth`); `src/test/ingest-index.test.ts` 10 (variables `v1`/`v2` = search-index builds; paths `search-index/v1/`); `src/test/bundle-reader-guard.test.ts:540,545` (HTTP ETag literal `"v2"`); `functions/_shared/cohort-index.ts:17` `search-index/v1` (v1 siblings, not v2).
- HPC dashboard data: `public/data/hpc/anvil_v2/{latest.json 13, history/20260918T02Z.json 13}` (2 files; campaign name `"campaign": "v2-human-mouse"`, job names `v2p_...`: this is a pipeline campaign called "v2" on Anvil, a different concept; the directory name `anvil_v2` is referenced by `README.md:104` and `src/test/hpc-dashboard.test.ts:34`). Decide separately whether the campaign name is also branding.
- Binary/graphics: `public/placeholder.svg` 7 and `public/brand/og-default.svg` 3 (SVG path data `...V2...`), PNG/GIF with "v2" byte sequences: `public/brand/lockup-dark.png`, `public/demo/singlet-demo.gif`, `public/demo/singlet-demo-poster.png`, `public/logos/r.png`, `public/og-default.png`, `public/og-image.png`, `src/assets/debruine.png`.
- Tooling versions: `package.json` `wrangler ^4.102.0` (no v2 literal); no node_modules tracked; no minified bundles tracked in git (dist/ is not tracked).
- Not v2: `pz2` (matrix codec name, `functions/_shared/v2-serve.ts`), `SCX1` cell-index shard format, `singlet-v1`, `manifest_version: 1`, `cidx-quota-v1` DO migration tag, `search-index/v1`.

## Summary counts (matching lines, excluding false positives)
- workers/data-v2: 94 lines in 10 files (+ 2 config/support files, 12 tracked files in dir).
- functions (+ schema): ~850 lines in ~30 files; `functions/_shared/v2-catalog.ts` 125, `mcp-v2.ts` 66, `mcp.ts` 47, `catalog-refresh.ts` 37, `v2-serve.ts` 35, `api/ingest/[table].ts` 33, `cohort-manifest.ts` 33, `bundle-core.ts` 29, `mcp-tools.ts` 26, `file-facts.ts` 26, `api/ingest/v2-study.ts` 24, `api/bundle/[gse]/[action].ts` 24, `study-core.ts` 23, `manifest-core.ts` 21; `schema/032` 17, `schema/033` 13.
- src (non-test): ~50 lines in 10 files. src/test: ~480 lines in 29 files + 44 fixture files (all under `fixtures/v2/**`, `fixtures/cidx/**`).
- scripts: `smoke.sh` 11 + `local.mjs` 7 genuine; remaining are false positives.
- CI: 0.
