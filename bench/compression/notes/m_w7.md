# m_w7: serving-layer mirror into singlet-cloud (W7)

Worktree: `C:\Users\zacha\Projects\singlet-cloud-v2`, branch `feat/singlet-v2` (origin/main f227fc241, 2026-10-20). Source of truth: singlet-website-v2 at HEAD
(`f16ee32cd` data Worker, `d4a6546d3` serving layer, `c54fcc578` UI; base `0dcf4914a`). Nothing committed, pushed or deployed; no Cloudflare read or write.

Result: `npm run check` green (vitest 37 files, 418 tests, was 25 files, 194; strict `tsc -p tsconfig.functions.json`; `wrangler pages functions build`
"Compiled Worker successfully") and `npm run build` green. Not run: a browser, a preview deploy, `scripts/smoke.sh` against a live host (syntax checked with `bash -n`).

Method: every changed website file was compared with the cloud copy against the website baseline `0dcf4914a` (CRLF ignored). Files identical to the baseline were copied;
the rest were 3-way merged with `git merge-file` (cloud = ours, `0dcf4914a` = base, website HEAD = theirs) and each conflict resolved by hand, then the merged file was
diffed against the website HEAD file to confirm that the only differences left are the cloud-only lines listed below. All files keep the checkout's CRLF.

## 1. Files changed

Straight copies (cloud was identical to the baseline): `functions/_shared/{bundle-reader,cache,catalog-refresh,manifest-core}.ts`, `functions/api/gse/[id].ts`,
`functions/api/manifest.ts`, `functions/api/projects/[id]/upload/[action].ts` (the `userHost: true` line), `src/components/study/FileTreePanel.tsx`.

New, copied unchanged: `functions/_shared/{mcp-core,mcp-v2,rev,v2-catalog,v2-cells,v2-gate,v2-packs,v2-serve,v2-urls}.ts`, `functions/api/ingest/v2-study.ts`,
`schema/031_bundle_index_identity.sql`, `schema/032_v2_catalog.sql`, `schema/033_v2_sample_rev_key.sql`, tests `bundle-reader-guard`, `mcp-v2`, `mcp-diet`, `v2-catalog`,
`v2-cells`, `v2-edge-ttl`, `v2-packs`, `v2-serve`, `src/test/helpers/zip.ts`, `src/test/fixtures/v2/**` (Python-written `.s2`, cells and pack fixtures plus the two generator scripts).

New, ported (see 2.1): `functions/_shared/restricted.ts` (verbatim copy of the website file, 44 lines).

Hand-merged: `functions/_shared/{bundle-core,mcp-tools,study-core,env}.ts`, `functions/api/bundle/[gse]/[action].ts`, `functions/api/ingest/[table].ts` (index-next hunk only),
`functions/mcp.ts`, `scripts/smoke.sh`, `package.json`, `package-lock.json` (the fzstd hunks of the website lock), `README.md` (the `known_rev` sentence),
`src/pages/DocsMcp.tsx` (the `known_rev?` inputs only), and the UI: `src/components/DownloadPanel.tsx`, `src/integrations/api/{client,types}.ts`, `src/pages/StudyDetail.tsx`.

Tests adapted or new for the cloud: `src/test/v2-api.test.ts` (website file without the `/api/cohort/manifest` block), `src/test/restricted.test.ts` (without the legacy "bundle listing"
test, see 2.1), `src/test/download-panel.test.tsx` (against the cloud panel), `src/test/cloud-mcp-wording.test.ts` (new).

Not copied, on purpose: `workers/data-v2/**` and `src/test/data-v2-worker.test.ts` (deployed from the website repo; the `check` script has no `tsc -p workers/data-v2` step here),
`functions/_shared/cohort-manifest.ts` and `functions/api/cohort/manifest.ts` (the cloud has no cohort API), the cohort block of `v2-api.test.ts`, and its helpers `bitindex-fixture.ts` / `cohort-index`.

## 2. Hand-merge decisions

### 2.1 `restricted.ts`: ported, used by the v2 code only
`v2-catalog.ts` (ingest refuses restricted members with 422) and `v2-serve.ts` (listings drop them) import `isRestrictedMember`. Rather than add a flag, the website file was copied verbatim
(it only needs `SAMPLE_RE` from `bundle-reader`, which the cloud has). The v2 objects live in the shared public bucket, so the same rule has to hold for the cloud's ingest and answers.

**Not adopted: the legacy `publicEntries` filter.** The cloud never got website PR #47: its `modalities.ts` registry still lists `mt_events`, `donor_snp_ad`, `donor_snp_dp`, `ase_counts`, and its
`bundle-core.ts`, `mcp-tools.ts` and `api/bundle/[gse]/[action].ts` read `index.entries` raw. Those three lines were left as the cloud has them (the 3-way merge conflicted there; the cloud
side was kept). Filtering only the listings while `get_modalities` still advertises the layers would be inconsistent, and which layers the cloud serves is a product decision (the
mirror table in m_w5.md section 7 flagged it). So on the cloud a legacy study still lists restricted members; a v2 study never does. To adopt the website policy later: mirror PR #47
(`modalities.ts`, `search-core.ts`, the three `index.entries` sites -> `publicEntries(index.entries)`, `modalities.test.ts`, the smoke restricted-member check) and put the "bundle listing"
test back in `restricted.test.ts`.

### 2.2 `functions/mcp.ts`
Website structure taken: `INSTRUCTIONS` renamed `INSTRUCTIONS_BASE`, `instructions(v2, diet)`, the v2 gate (`v2ToolsVisible`), `toolList({v2, diet})`, `known_rev` inputs, server 1.1.0.
Cloud wording kept in `INSTRUCTIONS_BASE` ("all reprocessed the same way", no "Coverage is partial..." sentence, "splice junctions and PSI"), in the `include_unbuilt` description
("not built yet"), the search result line ("file not built yet") and the study text ("File: not built yet"; the `Files: <manifest>` line for a v2 study is taken from the website).
The cloud has no extra MCP tools, so nothing else was at risk; `get_cohort`/`save_cohort` are in the baseline and stay (14 tools, 17 with the v2 tools).

### 2.3 `mcp-tools.ts`, `bundle-core.ts`, `[action].ts`, `study-core.ts`
* Guard and v2 branches taken whole (`withIndexRetry`, `StaleBundleError` -> 409 or a tool error, `strongEtag` recipes, `v2BundleIndexResponse`, `v2Entry`, `getV2State` in `loadStudy`).
* Cloud lines kept: raw `index.entries` (2.1); "not built yet" in `assess_study`; no `completeness` field or `Completeness:` note (the cloud has no `completeness.ts`).
* `study-core.ts`: the `v2` block and `bundle_url = null` once retired are in; the website's `hasFile: hasBundle || !!v2State` only existed inside `studyCompleteness`, so it is not needed.
  `assess_study` uses `hasFile = !!bundle_url || !!v2`, as in the website.
* `[action].ts` edge keys: `index` v=1 -> **v=3** (the website went v=2 -> v=3; the cloud never had v=2), `samples` v=1 -> v=2, `entry` `path=` -> `v=2&path=`. These caches are per host, so
  the cloud's entries are simply dropped by the new keys. `CACHE_TTL_HEADER` handling comes with the copied `cache.ts` (5 minutes for a study with any v2 row).
* `ingest/[table].ts`: only the `index-next` hunk (retry once on `StaleBundleError`); the 70 lines where the cloud differs are untouched.

### 2.4 Other
* `env.ts`: the cloud's `ADMIN_TOKEN` kept, `MCP_DIET`, `V2_TOOLS`, `V2_ORIGIN` added. No new binding or secret; `V2_ORIGIN` unset means `https://data.singlet.bio/v2/...` (checked in `v2-urls.ts`).
* `package.json`: only `"fzstd": "0.1.1"`. The website's `check` script change (`&& tsc -p workers/data-v2/tsconfig.json --noEmit`) was dropped. `package-lock.json`: the two fzstd hunks applied;
  `npm ci` passes, so `lockfile.yml` has nothing to fix.
* `scripts/smoke.sh`: added the stale-offset block (`smoke_bundle_offsets GSE111113` / `GSE138867`, skipped for a v2-served study or a retired legacy file) and the 14-or-17 tool-count check.
  Not added: the website's restricted-member check on the bundle index (the cloud does not filter, 2.1) and the website-only checks (cohort, library, feedback, layers, completeness).
* `DocsMcp.tsx`: `known_rev?` on the five tools; the cloud's `include_low_qc?` on `search_datasets` and `export_manifest` kept. `Docs.tsx` untouched (held back in the website too: "14 tools").
  README: the `known_rev` sentence only.
* **UI** (the cloud shows downloads, so it was ported by hand, not copied): `types.ts` (`StudyV2`, `GseRow.v2`, `BundleIndexResponse.format/manifest_url/files_truncated`) and `client.ts`
  (`normalizeStudyV2`, no legacy URL rebuilt for a retired study, v2 listing fields) merged cleanly. `DownloadPanel`: `v2` prop, "Manifest" button, curl line for the manifest, "GSE · N sample files"
  with the v2 size (the cloud panel shows sizes), note for the per-sample layout, "Also stored as one file per sample" next to the legacy download; the cloud's extras (usable-sample counts,
  "no usable count data" alert, partly-empty note) are kept and now also apply to a manifest-only study. `StudyDetail`: passes `v2={series.v2}`; the "no usable count data" banner is also
  shown for `series.v2`. The website's language tabs are not in the cloud and were not added.

## 3. Tests run

| command | result |
|---|---|
| `npm ci` | ok (639 packages) |
| `npx tsc -p tsconfig.functions.json --noEmit` (strict) | 0 errors |
| `npm run check` (vitest + strict tsc + `wrangler pages functions build`) | green: 37 files, 418 tests (before: 25 files, 194); "Compiled Worker successfully" |
| `npm run build` (vite) | green |
| `bash -n scripts/smoke.sh` | ok |

New tests per file: bundle-reader-guard 39, mcp-v2 34, mcp-diet 6, v2-api 6 (cohort 3 dropped), v2-catalog 26, v2-cells 49, v2-edge-ttl 3, v2-packs 29, v2-serve 14, restricted 5 (listing test dropped),
download-panel 10 (panel 6 including the cloud's no-usable-data alert and partly-empty note for a manifest-only study, file tree 2, normalisers 2), cloud-mcp-wording 3 (the cloud's "not built yet" wording
for a study with no file, the manifest line for a retired-legacy study, 14 -> 17 tools). Total 224 new. The first merged run had one failure (the legacy listing test of `restricted.test.ts`), which is
decision 2.1.

## 4. Where the cloud differs from the website, and why

1. No `publicEntries` on legacy listings (2.1). v2 paths filter.
2. No cohort API: `cohort-manifest.ts`, `api/cohort/manifest.ts` and their v2 hunk are absent; `v2-api.test.ts` has no cohort block. When the cloud gets the cohort feature, copy both files and the test block.
3. No `completeness` (study-core, mcp-tools, mcp.ts) and the cloud's "not built yet" wording instead of "file pending".
4. The cloud's own MCP instruction text and `include_low_qc` inputs; `ADMIN_TOKEN`; admin/beta/tool-evidence code untouched.
5. UI: cloud panel (sizes, usable-sample alerts) with the v2 additions; no language toggle; the `StudyDetail` banner also covers v2.
6. `check` has no data-Worker typecheck; `smoke.sh` has fewer checks.
7. Schema numbers: the cloud now has two files numbered 031 (`031_cloud_tools.sql`, `031_bundle_index_identity.sql`). Different tables, both additive, no tool applies by number; the website names are kept so
   the record matches. The cloud does not carry the website's 023 and 030 files, only the three the tests and the record need.

## 5. Before this deploys (shared D1, shared bucket)

* Schema 031 and 032 are applied in production (W5/orchestrator); **033 must be applied before the cloud deploys** (m_w5.md CS3: `npx wrangler d1 execute singlet-catalog --remote --file=schema/033_v2_sample_rev_key.sql`),
  because the cloud now has its own `/api/ingest/v2-study` and v2 readers. One application covers both sites. Cloud's own `031_cloud_tools.sql` is a separate file and is not touched here.
* The cloud now exposes a second door to the v2 ingest (same `INGEST_TOKEN_SHA256`). The campaign posts to singlet.bio only; if the commercial site should not accept campaign writes, delete
  `functions/api/ingest/v2-study.ts`; six of the v2 test files import its `onRequestPost` to seed rows (v2-catalog, v2-edge-ttl, v2-api, mcp-v2, mcp-diet, cloud-mcp-wording), so they would need another way to seed.
* The data Worker must serve `data.singlet.bio/v2/*` (deployed from the website repo; the 32 MiB fill rule is a separate redeploy there). Nothing here deploys or changes it.
* Retirement checklist items 1 to 6 of m_w5.md section 5 apply to the cloud as well (including: do not retire a legacy file until 24 h after the flip to `verified`, because cached legacy answers live a day).
  Not done in the website either and therefore not here: Browse download links (`StudyCard` etc. still build `data/<GSE>/<GSE>.singlet`), the JSON-LD `contentUrl` in `StudyDetail` (falls back to the
  legacy URL for a retired study), `has_bundle` for v2-only studies.
* After deploy: `scripts/smoke.sh https://cloud.singlet.bio` (expect tools/list 14, offsets ok for GSE111113 and GSE138867), and hit `/api/bundle/GSE111113/index` to see the `bytes` equal the object size.

## 6. Proposed commit message

```
Mirror the v2 serving layer: stale-index guard, v2-aware API/MCP/UI, known_rev

Port singlet.bio ed2f776 (f16ee32, d4a6546, c54fcc5) minus the data Worker and the cohort API.

- bundle-reader: bundle_index rows carry etag/last_modified (schema 031, already applied to the shared
  D1); every ranged read sends If-Match and checks the zip local header; 412/416/header mismatch
  re-reads the central directory once (withIndexRetry); only 206 is accepted from our own host;
  /api/bundle index is verified with one HEAD; recipes carry a strong etag; index-next retries once.
- v2 catalog (schema 032/033, copied for the record), POST /api/ingest/v2-study, and v2-aware
  /api/gse, /api/bundle index+entry, /api/manifest, MCP get_study/get_download_url/
  list_bundle_files/get_partial_download/get_modalities/assess_study with the legacy answer as
  fallback. find_cells/slice_cells/resolve_features appear in tools/list only once a study is served
  from v2 (v2-gate). rev / known_rev on the five study tools. Edge keys: index v=3, samples v=2,
  entry v=2; a study with any v2 row is cached 5 minutes.
- restricted.ts ported for the v2 paths only; the legacy listings keep their current behaviour.
- UI: DownloadPanel/StudyDetail/client/types/FileTreePanel show the manifest for a v2 study.
- fzstd 0.1.1; smoke.sh offsets check; 224 new tests.

Apply schema/033_v2_sample_rev_key.sql to the shared D1 before deploying.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
```
