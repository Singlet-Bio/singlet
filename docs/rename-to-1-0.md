# Renaming the per-sample format for Singlet 1.0.0 (no "v2" anywhere)

Status: APPROVED 2026-10-25 (Zach) with extension `.s1`, decisions A keep magic bytes, B `called_v2` -> `called_ed`, C rename our own
"v2" names only where trivial, E old Worker keeps `/v2/*` until cleanup. Implemented in this PR (P1) and website PR W1; the R2, D1
and Worker steps (O1-O4) run only on the orchestrator's go-ahead.

Decision being implemented (Zach): "Rename everything before 1.0.0, including the manifest format string, R2 paths and
module names; plan it as one migration-safe change."

The per-sample storage format was built under the working name "v2". It has two parts:

* **Objects:** `.s2` sample objects, study manifests, packs and the cell index in R2.
* **Tooling:** the `singlet.v2` package, `r/R/v2.R`, `scripts/v2_*`, the `data-v2` Worker and the D1 tables `v2_*`.

The release is Singlet 1.0.0. This plan gives the per-sample format one durable name, **store**, with no version word, and
puts versions in fields. It says what keeps its bytes, how the rename lands without a broken reader at any point, and in
what order.

What exists in production today, and is affected:

* **R2:**
  * GSE145370: 14 `.s2` objects, the manifest and the ledger.
  * `v2/ref/`: `index.json` and 161 packs (12 vocabularies, 4 whitelists, 5 junction table versions, 140 junction
    segments).
  * `v2/cidx/current.json` returns 404, so no cell index is published yet.
* **D1:** `v2_study` (1 row), `v2_sample` (14 rows), `v2_migration` (1 row).
* **Worker:** `singlet-data-v2` on `data.singlet.bio/v2/*`.
* **Website:** the API, the MCP server and the frontend read these tables and keys.

No Python wheel or CRAN package with the `v2` reader was ever published (`singlet-bio` is not on PyPI). Old-name readers
therefore exist only in:

* the deployed website, MCP server and Worker;
* GitHub checkouts of `main` and of open branches;
* jobs running on Clipper or Anvil from those checkouts.

The atlas re-run and migration campaign has not started. After it starts, a rename of anything content-addressed would
mean migrating again, so this lands first.

## 1. Names

One word, **store**, names the per-sample object store everywhere: the code, the R2 prefix, the URL, the D1 tables and the
format string. A version, where one is needed, is a field (`manifest_version`, the header `version` u16, the pack index
`version`), never part of a name.

| what | today | 1.0.0 |
|---|---|---|
| Python package | `singlet.v2` (`python/singlet/v2/`, 33 modules + `packs/` + `recipes_data/`) | `singlet.store` (`python/singlet/store/`) |
| Python public names | `V2Bundle`, `V2Sample`, `V2Context`, `V2Error` (two classes), `celllabels.read_v2`, `features.io.is_v2` / `v2_gsms` / `iter_samples(v2=)`, `load(**v2_options)`, `adata.uns["singlet_v2"]`, CLI `verify --v2 DIR`, migrate `--no-cell-call-v2` | `StoreBundle`, `StoreSample`, `StoreContext`, `StoreError`, `read_store`, `is_store` / `store_gsms` / `iter_samples(store=)`, `load(**store_options)`, `adata.uns["singlet_store"]`, `--store DIR` (old spellings kept as deprecated aliases until cleanup) |
| local MCP server (`python/singlet/mcp/server.py`) | output fields `"format": "v2"`, `out["v2"]`, `out["v2_error"]`, generated code `singlet.v2.cells(...)` | `"format": "singlet-store"`, `out["store"]`, `out["store_error"]`, `singlet.cells(...)` |
| R | file `r/R/v2.R`; 51 internal `.v2_*` functions; condition classes `singlet_v2_error`, `singlet_v2_not_found`, `singlet_v2_integrity`, `singlet_v2_modality_unavailable`; `metadata(sce)$singlet_v2`; Rd titles ("Open a study's v2 manifest"); C++ error text `"v2: truncated data"`. **No exported function has v2 in its name** (`singlet_study`, `singlet_study_modalities`, `singlet_cells`). | `r/R/store.R`, `.store_*`, `singlet_store_*` classes (conditions carry both class vectors until cleanup so existing `tryCatch` handlers still match), `metadata(sce)$singlet_store`, titles reworded |
| env vars (clients, tests) | `SINGLET_V2_PACKS`, `SINGLET_V2_CODEC`, `SINGLET_V2_VERIFY`, `SINGLET_V2_RECIPES_CACHE`, `SINGLET_V2_CACHE`, `SINGLET_V2_FULL`, `SINGLET_TEST_V2_CORPUS_ROOT`, `SINGLET_V2_TEST_ROOT`, `SINGLET_V2_TEST_URL`, `SINGLET_V2_EXTRAS_ROOT` | `SINGLET_STORE_*` (the old name is read as a fallback until cleanup) |
| User-Agent | `singlet-v2/1.0`, `singlet-v2-campaign/1.0` | `singlet/<package version>`, `singlet-campaign/<package version>` (no Worker or Function allow-lists a UA, checked in the website repo; the zone rejects only the default Python UA) |
| local cache directories | `<cache>/v2`, `~/.cache/singlet/v2`, `<tmp>/singlet_v2_recipes`, campaign work area `state/v2/` | `<cache>/store`, `~/.cache/singlet/store`, `singlet_store_recipes`, `state/store/` (old caches are simply not reused) |
| ledger fields (`<prefix>/ledger/<GSE>.jsonl`) | `v2_object`, `v2_name`, `v2_kind`, `v2_bytes` | `object`, `name`, `kind`, `bytes`. The ledger is a record, not content-addressed, so existing ledgers are copied unchanged and readers accept both. |
| scripts | `scripts/v2_campaign.py`, `v2_publish.py`, `v2_retire.py`, `v2_cidx_build.py`, `v2_junction_table.py`, `v2_junction_table.sbatch`, `v2_migrate.sbatch` | `scripts/store_campaign.py`, `store_publish.py`, `store_retire.py`, `store_cidx_build.py`, `store_junction_table.py`, `store_junction_table.sbatch`, `store_migrate.sbatch` |
| docs | `docs/format-v2.md`, `format-v2-cell-index.md`, `format-v2-cell-index-costs.md`, `v2-migration.md`, `v2-performance.md` | `docs/format-store.md`, `format-store-cell-index.md`, `format-store-cell-index-costs.md`, `store-migration.md`, `store-performance.md` |
| manifest format string | `"format": "singlet-v2"`, `"manifest_version": 1` | `"format": "singlet-store"`, `"manifest_version": 1` (the existing version field is kept) |
| pack index format | `"format": "singlet-v2-packs"`, `"version": 1` | `"format": "singlet-store-packs"`, `"version": 1` |
| legacy single-file format value in the API | `"singlet-v1"` | `"singlet-bundle"` (it also carries a version word) |
| R2 prefix | `v2/` (`v2/s/`, `v2/m/`, `v2/ref/`, `v2/ledger/`, `v2/cidx/`, `v2/a/`, `v2/cf/`, `v2/x/`, `v2/cells/`) | `store/` with the same sub-prefixes (`store/s/`, `store/m/`, ...) |
| sample object extension | `<d16>.s2` (also the `v2/x/<GSM>/adt.<d16>.s2` extra objects) | `<d16>.s1` |
| URL | `https://data.singlet.bio/v2/...` | `https://data.singlet.bio/store/...` |
| cell index API | `POST /v2/cidx/query`, `/v2/cidx/gather`, `/v2/cidx/_part` | `POST /store/cidx/query`, ... |
| data Worker | `workers/data-v2`, script `singlet-data-v2`, `X-Served-By: data-v2`, log prefix `data-v2:` | `workers/data`, script `singlet-data`, `X-Served-By: singlet-data`, log prefix `singlet-data:` |
| D1 tables | `v2_study`, `v2_sample`, `v2_migration` | `store_study`, `store_sample`, `store_migration` |
| D1 column | `v2_migration.v2_bytes` | `store_migration.store_bytes` |
| D1 indexes | `idx_v2_sample_gsm`, `idx_v2_migration_status` | `idx_store_sample_gsm`, `idx_store_migration_status` |
| API JSON | `series.v2` (and `v2` in search/list rows, MCP `get_study`) | `series.store` |
| cohort export columns | `v2_manifest_url`, `v2_rev`, `v2_bytes` | `store_manifest_url`, `store_rev`, `store_bytes` |
| catalog-refresh key | `sample_usable_v2` | `sample_usable_store` |
| ingest endpoint | `POST/GET /api/ingest/v2-study` | `/api/ingest/store-study` |
| env vars | `V2_ORIGIN`, `V2_TOOLS` (Pages) | `STORE_ORIGIN`, `STORE_TOOLS` |
| website modules | `functions/_shared/v2-*.ts`, `mcp-v2.ts`, `api/ingest/v2-study.ts` and identifiers `V2State`, `StudyV2`, `getV2State`, ... (about 60 names) | `store-*.ts`, `mcp-store.ts`; `StoreState`, `StudyStore`, `getStoreState`, ... |
| MCP text | "format \"singlet-v2\"", "v2 format", "v2 studies", "v2 sample", "v2 catalog" | "format \"singlet-store\"", "per-sample format", "per-sample studies", ... (rewritten, not substituted word for word) |
| cell call | `called_v2`, `called_v2_fdr` (cells-table columns, the manifest/metadata block `called_v2`, `singlet.v2.cellcall`, the C++ `called_v2.h` on the prod branches) | **Decision B below**; recommended `called_ed`, `called_ed_fdr`, block `called_ed` (method version stays a field: `"version": "2.1"`) |

### 1.1 What does not change, and why

These identifiers are bytes inside content-addressed objects. Renaming any of them changes the digest of every object that
contains it, and with it the R2 key, the manifest entry, the D1 `sha256` and the study `rev`.

| identifier | where it sits | what a change would invalidate |
|---|---|---|
| `SINGLET2` | first 8 bytes of every sample object | every sample object's digest16, so its key and sha256 |
| `PZ2M` | first 4 bytes of every matrix member header | the member bytes and the sample object digest. The canonical matrix digest (format section 4.6) does not hash the magic and survives. |
| `PZ2CANON1` | domain string of the canonical matrix digest (`api.h:401`, `matrix.py:124`) | every matrix member's stored `digest`, so the member table and the object digest |
| `CEL2`, and `CEL2CANON1` in the canonical cells digest | cells member | the cells member's content digest and the object digest |
| `SVP0` | vocabulary pack body coder | the pack ids, as for `SPK2` |
| `SPK2` | every pack | every pack id (digest16 of the file). Matrices store the first 8 bytes of the pack id (`row_names_id`), and junction version packs list their segment ids. So every pack and every sample object changes. |
| `SCX1` | cell index shards | every shard key and the index version id. No index is published yet. |
| codec name `pz2` (C++ `singlet::pz2`, `include/singlet/pz2/`, Python `_pz2`, R `pz2_*`) | code only; the bytes are `PZ2M` | nothing in the data, but every binding and the R API |
| legacy names `.1pz`, `TP1Z`, `1PZ02`, member file names | legacy bundles and legacy member names inside objects | legacy regeneration (bit equivalence, D15) |

**Recommendation (Decision A): keep the magic bytes and the codec name, and document them.**

* **Magic bytes.** A magic number is a file signature, not a product name. The `2` records the container generation: the
  legacy `.singlet` zip with `TP1Z` matrices was generation 1. The format docs will say exactly that, next to each magic.
* **Pilot objects.** Keeping the magic lets every existing pilot object, pack and ledger be **copied byte-identically** to
  its new key. Only the key prefix and the extension change, and the digest in the key stays the same. The copy can then
  be verified by sha256 and nothing has to be re-encoded.
* **Codec name.** `pz2` names a codec, the way `bzip2` or `lz4` does. It pairs with the legacy `1pz`. It is not the format
  version.

**If Zach wants the magic changed anyway (option A2), now is the only cheap time.** Proposed values: `SINGLETS` (sample
object), `SGLM` (matrix), `SGLC` and `SGLCCANON1` (cells), `SGLP` (pack), `SGLX` (index shard). What it costs today:

1. Re-wrap the 161 packs. Only the header changes, and the bodies are reused. The 5 junction version packs must be rebuilt
   over the new segment ids. About 1 hour on the laptop.
2. Re-run the pilot migration of GSE145370 with the new writer: 187 MB in, minutes on the laptop. Then verify it against
   the legacy bundle (D3).
3. Readers accept both magic sets until the old pilot objects are deleted. That is a dispatch on 4 bytes in `pz2.h`,
   `container.py`, `cells.py`, `packs.py` and the Worker's `scx.ts`.
4. Regenerate every fixture that embeds the magic bytes (`tests/data`, website `src/test/fixtures/v2/**`, cidx golden
   files).

That adds roughly half a day of agent work and one more verification of the pilot, and it moves the risk of a codec bug
from "none" to "small". After the campaign, the same change means re-encoding the whole atlas, which is not realistic.

## 2. Inventory

Counts are matching lines (`git grep -c`) on `origin/main`: singlet-bio `5c279cd`, singlet-website `549d67d81`. Bench
result tables, manuscripts and notebooks are historical records. They are counted under false positives and are not
renamed.

### 2.1 singlet-bio

The full listing, with file:line for every definition, is
[`rename-to-1-0/inventory-singlet-bio.md`](rename-to-1-0/inventory-singlet-bio.md). In-scope hits total about 2,400
lines, out of 47,583 matching lines in all. Of those, 38,401 are historical `bench/compression` tables, 6,191 are one 10x
chemistry TSV, and 472 are in `papers`.

| surface | files | lines | changes stored or served bytes? |
|---|---|---|---|
| `python/singlet/v2/` (33 modules, 16 packs, `packs/index.json`, 1 `recipes_data` file); intra-package imports are relative (100 lines), with 29 absolute self-references | 50 | 295 | no, except that `packs/index.json` carries `singlet-v2-packs` |
| `singlet.v2` imports elsewhere (`_loader.py` 14, `_io.py`, `bundle.py`, `mcp/server.py`, `celllabels/io.py`, `features/io.py`, `__init__.py:244-245` re-exports `cells` and `open_study`; scripts, bench, tests); `pyproject.toml:108,153` package-data and tool config | about 50 | 235 | no |
| public Python names, local MCP output fields, `uns["singlet_v2"]` | about 12 | about 150 | MCP responses only |
| CLIs `python -m singlet.v2.migrate`, `.verify`, `.packs_build` (called by `scripts/v2_campaign.py`, `v2_migrate.sbatch`, `v2_publish.py`) | 6 | — | no |
| R (`R/v2.R` 228 lines, `load.R` 21, `bundle.R` 3, 6 Rd files, `pz2_io.cpp` messages, `test-v2.R`, `test-v2-modalities.R`) | 10 | about 440 | no |
| `pz2` codec naming (C++ namespace and directory, 14 headers mirrored into `r/inst/include` by `scripts/sync_r_headers.py`, `_pz2` binding, CMake `singlet_pz2`, 12 R `pz2_*_` functions, `PZ2_*` macros and env) | 62 paths | 829 | no. Only the magic bytes and `PZ2CANON1` are stored (section 1.1). |
| magic bytes and digest domains (`SINGLET2` `PZ2M` `SPK2` `CEL2` `CEL2CANON1` `PZ2CANON1` `SCX1` `SVP0`) | about 25 | about 150 | **yes** (section 1.1) |
| scripts: `v2_campaign.py` (130+), `v2_publish.py`, `v2_retire.py`, `v2_cidx_build.py`, `v2_junction_table.py` / `.sbatch`, `v2_migrate.sbatch` (job name `v2mig`, branch `feat/singlet-v2`); referencing scripts: `annotate_campaign.py`, `annotate_sample.py`, `features_run.py` / `.sbatch` | 13 | 283 | no |
| docs: `format-v2.md` (51 inbound references in 41 files, including `CLAUDE.md` and the pz2 headers), `format-v2-cell-index.md`, `-costs.md`, `v2-migration.md`, `v2-performance.md`, plus `called-v2*`; citing docs `cell_annotation.md`, `cell_features.md`, `empty-droplets.md`, `quickstart.md:37`, `CELL-ARCHITECTURE.md`. None is in the mdBook (`SUMMARY.md`). | 22 | about 460 | no |
| wire strings: `singlet-v2` (62 lines; constants `v2/__init__.py:7`, `manifest.py:25`), `singlet-v2-packs`, key prefixes (`migrate.py:436` sample key, `client.py:606`, `packs.py:45` `REMOTE`, `atlas.py:71` `CIDX`, `atlas.py:311,736` sidecars, `migrate.py:1052` ledger), `.s2` (73 lines in 31 files), D1 table names in the campaign scripts, the `/api/ingest/v2-study` endpoint, `SINGLET_V2_*` env vars, User-Agent strings | about 60 | about 600 | **yes** |
| tests: 19 `tests/python/test_v2_*` and helper files (529 lines), 2 R test files; no checked-in `.s2` fixture (`_v2_synth.py` builds them) | 27 | about 580 | no |
| CI | 0 | 0 | no (14 hits, all action tags or 10x chemistry) |
| `called_v2`: columns `called_v2`, `called_v2_fdr` (`cellcall.py:42`, written at `migrate.py:476-478`), meta and manifest key `called_v2`, `n_called_v2`; `bench/called_v2*/`, `docs/called-v2*`, `tests/data/called_v2_golden/` (14 files); on the prod branches also `include/singlet/pileup/called_v2.h` (1,579 lines) and C++ tests | 40 (+11 on the prod branches) | 212 | **yes**: columns and metadata inside every object (Decision B) |
| **false positives (do not rename)** | — | about 45,000 | 10x chemistry (`10xv2`, `SC3Pv2`, `10x-3p-v2`; about 6,350 lines); inDrop v2; min-sketch format v2; doublet detector v2 (`test_doublet_detect_v2.cpp`); "Pipeline V2" (`scripts/singlet_orchestrator_v2.sh`, `pileup_engine.h:4`); legacy `1PZ02` / `singlet.pz_v2`; `cublas_v2.h`; htslib BCFv2.1; S3 `list_objects_v2`; action tags `@v2`; assembly names (`ASM276v2`); gene symbols (`NDUFV2`); `cellguide-v2`; GPU archive scripts; local variables and SIMD registers; `bench/compression/**`, `papers/**` and notebooks (historical). |

Two references point at files that do not exist: `docs/V2_IMPLEMENTATION_PLAN.md`, cited by `scripts/build_*_fbin.py`,
and `include/singlet/pileup/called_v2.h`, which exists only on the prod branches. P1 leaves both alone.

### 2.2 singlet-website

The full listing is [`rename-to-1-0/inventory-singlet-website.md`](rename-to-1-0/inventory-singlet-website.md).

Roughly 1,000 matching lines in about 70 files, plus 44 fixture files. Neither CI nor any script has a `v2` file name, and
the repo holds no `called_v2`.

| category | identifiers | count / files |
|---|---|---|
| data Worker | directory `workers/data-v2/` (12 files); script name `singlet-data-v2` (wrangler.toml, the `SELF` service binding, `scripts/local.mjs`); route `data.singlet.bio/v2/*`; `X-Served-By: data-v2` (`handler.ts:257`, `routes.ts:27`); log prefix `data-v2:`; `CIDX_PATHS` `/v2/cidx/{query,gather,_part}` and the internal fan-out URL `https://data.singlet.bio/v2/cidx/_part` (`routes.ts:10,61,69,100`); `from singlet.v2 import atlas` snippets returned to API users (`cidx.ts:926-929`); `package.json:13` `tsc -p workers/data-v2/...` | 94 lines / 10 files |
| R2 keys | no single constant. Ten key regexes in `workers/data-v2/src/handler.ts:99-110`, `SAMPLE_KEY` in `gather.ts:176`, `cidx.ts:198,209`; website `functions/_shared/v2-urls.ts:24-26`, `v2-catalog.ts:40`, `bundle-core.ts:138`, `Docs.tsx:233-234`, `smoke.sh:123`. Occurrences: `v2/s/` 81, `v2/m/` 54, `v2/cidx/` 54, `v2/ref/` 23, `v2/cells/` 7, `v2/ledger/` 6, `v2/a/` 2; `.s2` 93 | about 30 files |
| D1 schema | `schema/032_v2_catalog.sql`, `033_v2_sample_rev_key.sql` (tests read both by file name); tables `v2_study`, `v2_sample`, `v2_migration`; column `v2_bytes`; indexes `idx_v2_sample_gsm`, `idx_v2_migration_status`; no views or triggers. Row values hold old keys: `manifest_key`, `key`, `ledger_key` | 30 lines / 2 files |
| D1 query sites | `v2-catalog.ts` (36 lines, the main DAO), `catalog-refresh.ts` 6, `api/ingest/v2-study.ts` 6, `api/ingest/[table].ts` 4, `mcp-v2.ts` 3, `layers-core.ts` 3, `v2-summaries.ts` 3, `v2-gate.ts` 2, `api/gse/index.ts` 1, `rev.ts` 1; `mcp-v2.ts:472` matches `/no such table/` on these tables | 11 files |
| API contract | value `"singlet-v2"` (75; about 35 outside tests) and the legacy counterpart `"singlet-v1"`; key `v2` (`series.v2` in `/api/gse/:id`, list and search rows, MCP `get_study`, `StudyDetail.tsx:256` JSON-LD `contentUrl`); export columns `v2_manifest_url`, `v2_rev`, `v2_bytes`; `sample_usable_v2`; `fileRev` `source: "v2"`; endpoint `/api/ingest/v2-study` | about 25 files |
| env and constants | `V2_ORIGIN`, `V2_TOOLS`; `V2_ORIGIN_DEFAULT`, `V2_GATE_TTL_MS`, `V2_EDGE_TTL`, `V2_INDEX_FILE_SAMPLES`, `V2_INSTRUCTIONS`, `V2_ONLY`, `V2_PROPS`, `PENDING_V2_SQL`, `USABLE_V2_PENDING` | — |
| module and identifier names | `functions/_shared/{mcp-v2,v2-catalog,v2-cells,v2-gate,v2-load-modalities,v2-packs,v2-serve,v2-summaries,v2-urls}.ts`, `api/ingest/v2-study.ts`; about 60 exported identifiers (`V2ManifestError` 37, `StudyV2` 22, `V2Member` 21, `getV2State` 18, ...) | 10 files |
| MCP text | server instructions (`mcp.ts:96`, "format \"singlet-v2\""); `V2_ONLY`; the `find_cells`, `slice_cells` and `resolve_features` descriptions; six `V2_PROPS` input descriptions; seven error and hint strings in `mcp-v2.ts` ("v2 catalog", "v2 format"); `cohort/manifest.ts:63`; `README.md:109` | 4 files |
| frontend | `client.ts` 16, `types.ts` 10, `DownloadPanel.tsx` 6, `FileTreePanel.tsx` 5, `StudyDetail.tsx` 4, `Docs.tsx` 5 (the only on-screen "v2": URLs, `series.v2`, `format: "singlet-v2"`), `StudyCard`, `StudyTable`, `useSelection` 2 each | about 50 lines / 10 files |
| docs and comments | `workers/data-v2/README.md` (28; includes the fallback host `dv2.singlet.bio`); `CHANGELOG.md:13,24`; comments citing singlet-bio `docs/format-v2*.md`, `docs/v2-migration.md`, `python/singlet/v2/*`, `r/R/v2.R`, `scripts/v2_cidx_build.py` | — |
| scripts | `scripts/smoke.sh` (11 lines, greps `"format":"singlet-v2"` and `v2/m/`), `workers/data-v2/scripts/local.mjs` (7) | 2 files |
| tests and fixtures | 10 test files named `*v2*` (`v2-catalog.test.ts` 129 lines, `retired-study` 68, `data-v2-worker` 59, `mcp-v2` 56, ...), about 480 lines in 29 test files; fixtures `src/test/fixtures/v2/**` (28 files, including the binary `GSM1000001.s2` generated by singlet-bio) and `fixtures/cidx/v2/cidx/**` (9) plus `fixtures/cidx/expected.json` | 29 + 44 files |
| CI | none. The Worker is deployed by hand, and `ci.yml` compiles it through `npm run check`. | 0 |
| **false positives (do not rename)** | `package-lock.json` hashes (29); 10x chemistry `10xv2` / `10x 3' v2` (about 120 lines: seeds, enrichment data, `search-core.ts:720`, `interpret.ts`); search prompt and translator versions; `explain.ts` `CACHE_VERSION = "v2"`; Google `oauth2/v2/auth`; ETag literals in tests; `search-index/v1`; DO tag `cidx-quota-v1`; SVG and PNG bytes; HPC dashboard `public/data/hpc/anvil_v2/` (campaign `v2-human-mouse`, Decision C) | — |

## 3. Migration safety

The rules for the whole rename:

1. **Readers before writers.** Every reader accepts both names before any writer emits the new ones. "Every reader" means
   the Python and R clients, the Worker, the website and the MCP server.
2. **Copy, never move.** Old R2 keys stay until a cleanup step after the campaign is under way. Old D1 names stay as
   read-only views until the same cleanup.
3. **One name per tree.** Objects under `v2/` keep their old manifests (`singlet-v2`, keys `v2/s/...`). Objects under
   `store/` get new manifests (`singlet-store`, keys `store/s/...`). A manifest never mixes the two trees, so an old
   reader that finds an old manifest also finds every key it names.
4. **New work writes only new names.** That covers the campaign, publishing, junction tables and the cell index.
5. **Every R2, D1 or Worker step is separately approved.** Each one has a rollback.

### 3.1 Dual-read rules

**Python (`singlet.store`) and R:**

* **Locating a study.** Try `store/m/<GSE>.json` first, then `v2/m/<GSE>.json`, then the legacy bundle (the existing
  `_LEGACY_FALLBACK_DEFAULT`).
* **Manifest format.** Accept `format` in {`singlet-store`, `singlet-v2`}.
* **Sample keys and packs.** Accept sample keys matching `^(store|v2)/s/GSM\d+/[0-9a-f]{32}\.(s1|s2)$`. Fetch packs
  from the prefix of the manifest that named them.
* **Cell call column.** Accept the old column name for the new one (`called_v2` read as `called_ed`, if Decision B is
  approved).
* **Temporary alias.** `singlet.v2` stays as a module alias (`sys.modules` entries for each submodule, with a
  `DeprecationWarning`). It covers the open branches and running Clipper jobs that import `singlet.v2.cellcall` and
  similar modules. It is removed in cleanup, before the 1.0.0 tag.

**Worker:**

* Serves `/store/*` and `/v2/*` with the same handler. The R2 key is the path, so both trees are served one to one.
* The cell index endpoints answer under both prefixes.
* `SAMPLE_KEY` in `gather.ts` accepts both forms.

**Website and MCP:**

* Read the `store_*` tables. Until cleanup, the `v2_*` names exist as views.
* Accept both manifest format values on ingest. The validation in `v2-catalog.ts:318` becomes a set.
* The API emits `series.store` and, until cleanup, also `series.v2` with the same object. The cohort export emits both the
  `store_*` and the `v2_*` columns. Only the website frontend and the MCP server read these fields, and both move in the
  same deploy. The alias exists for scripts and notebooks that were written against the API during the pilot.
* `STORE_ORIGIN` falls back to `V2_ORIGIN`.

### 3.2 R2 copy (pilot and reference)

The copy is done by a new script, `scripts/store_rekey.py`, run from the laptop with wrangler in the same way as the pilot
upload. It has three modes:

* **Default: plan only.** Lists every source key, its destination key, and whether the object is copied or rewritten.
* `--apply`: performs the plan.
* `--verify`: checks the result.

| source | destination | how |
|---|---|---|
| `v2/s/GSM*/<d>.s2` (14) | `store/s/GSM*/<d>.s1` | byte copy; sha256 must equal the source and D1 `v2_sample.sha256` |
| `v2/ref/<id>.pack` (161) | `store/ref/<id>.pack` | byte copy; sha256 must equal `index.json` |
| `v2/ref/index.json` | `store/ref/index.json` | rewritten: `format: "singlet-store-packs"`, otherwise identical (no keys inside) |
| `v2/ledger/GSE145370.jsonl` | `store/ledger/GSE145370.jsonl` | byte copy (a historical record; its key strings stay old) |
| `v2/m/GSE145370.json` | `store/m/GSE145370.json` | **rewritten** by `singlet.store.manifest`: `format: "singlet-store"`, sample `key`s re-prefixed. The `rev` is recomputed, because it hashes the sample keys. Uploaded **last**, after every object it names verifies. |
| `v2/a/`, `v2/cf/`, `v2/x/`, `v2/cells/`, `v2/cidx/` | same under `store/` | byte copy for anything present at copy time. Today nothing is: a cidx built later is built under `store/` by `store_cidx_build.py`. |

Verification (`--verify`) checks three things:

* every destination object exists and its sha256 equals its source;
* the new manifest validates with `singlet.store.verify` against the new keys;
* `singlet.load("GSE145370")` through `https://data.singlet.bio/store/` gives an AnnData identical to the load through
  `/v2/`, and to the legacy bundle (X, spliced, unspliced, obs/var names, obs columns). This repeats the pilot check
  from 2026-10-22.

Rollback: delete the `store/` keys. The `v2/` tree was never touched.

### 3.3 D1

One migration file, numbered after whatever the open website PRs (#54) take. It renames the tables and the column, keeps
the old names as views and rewrites the key values:

```sql
ALTER TABLE v2_study     RENAME TO store_study;
ALTER TABLE v2_sample    RENAME TO store_sample;
ALTER TABLE v2_migration RENAME TO store_migration;
ALTER TABLE store_migration RENAME COLUMN v2_bytes TO store_bytes;
-- index renames: DROP INDEX + CREATE INDEX with the new names
UPDATE store_study     SET manifest_key = 'store/' || substr(manifest_key, 4) WHERE manifest_key LIKE 'v2/%';
UPDATE store_sample    SET key = replace('store/' || substr(key, 4), '.s2', '.s1') WHERE key LIKE 'v2/%';
-- rev: set to the recomputed rev of the rewritten manifest (written by store_rekey.py into the SQL it emits)
CREATE VIEW v2_study     AS SELECT * FROM store_study;
CREATE VIEW v2_sample    AS SELECT * FROM store_sample;
CREATE VIEW v2_migration AS SELECT *, store_bytes AS v2_bytes FROM store_migration;
```

`ledger_key` keeps pointing at the old ledger copy; both copies are byte-identical.

**Why a rename plus views, rather than new tables and a copy.** Today the only writer is the ingest endpoint, which is fed
by the campaign and the publisher, and no ingest is running. The rename is atomic, so readers never see two diverging
copies. The views keep the currently deployed Pages build working between the migration and the deploy. Only its writes
would fail, and there are none during that window.

**How it is applied.** The same way as 031 to 034: `wrangler d1 execute --remote --file`, after recording a Time Travel
bookmark.

**Writes.** 16 rows plus the DDL, a few dozen row writes in all.

**Rollback.** Restore the bookmark (`wrangler d1 time-travel restore`), then redeploy the previous Pages build.

**Keys in the window.** Between this migration and the Pages deploy, the old build reads `store/...` keys through the
views. Those keys are served, because the Worker step (3.4) comes first.

### 3.4 Worker

The rename creates a new script, `singlet-data`. It has its own Durable Object namespace for `CidxQuota`, so daily cell
quotas reset once, which is harmless. The rate-limit namespace ids stay the same.

1. Deploy `singlet-data` with the route `data.singlet.bio/store/*` only.
2. Smoke test it:
   * `curl -sI https://data.singlet.bio/store/ref/index.json` shows `x-served-by: singlet-data`;
   * Range requests, CORS and ETag behave as on `/v2/`;
   * a cell index query returns a plan;
   * `/v2/*` still shows `x-served-by: data-v2`.
3. Move `/v2/*` in two deploys:
   * Deploy `singlet-data-v2` without its route. During the seconds before step two, `/v2/*` falls through to the R2
     custom domain, which serves the same bytes without the CORS and Cache headers.
   * Redeploy `singlet-data` with both routes.

   Alternatively, step 3 can wait until cleanup and `singlet-data-v2` keeps serving `/v2/*` alone until then.
   **Recommended: keep the old Worker on `/v2/*` until cleanup.** Then no step has a fall-through window, and rollback is
   simply deleting the new script.

### 3.5 Order of PRs and deploys

| step | repo / action | contents | needs before it | rollback |
|---|---|---|---|---|
| P0 | website: `docs/CELL-ARCHITECTURE.md` | canonical copy first (CLAUDE.md rule): `v2` → `store` in names and keys, plus a note on the transition. The canonical copy so far exists only on website PR #54 (`claude/ontology-filters-46958f`), not on main, so P0 is either a commit on #54 before it merges or a PR right after it. | — | revert PR |
| P1 | singlet-bio PR | `git mv python/singlet/v2 python/singlet/store` and every import; the `singlet.v2` alias; constants `PREFIX = "store/"`, `EXT = ".s1"`, `FORMAT = "singlet-store"`; dual read (3.1); `r/R/store.R` (no exported R name changes; conditions carry both the `singlet_store_*` and the `singlet_v2_*` classes until cleanup); old Python public names and env vars kept as deprecated aliases; scripts renamed; docs renamed with redirect stubs at the old paths; `called_v2` → `called_ed` (if B) in Python and R; `scripts/store_rekey.py`; tests (section 5). CELL-ARCHITECTURE copy synced from P0. | P0 | revert PR. No data is written. |
| W1 | website PR | `workers/data` (dual prefix, new name, header); Pages: table names, `series.store` + alias, export columns, MCP text, env var fallback, ingest endpoint renamed (old path kept as an alias), D1 migration file, fixtures regenerated by P1's writer; smoke script | P1 merged (fixtures come from it) | revert PR |
| O1 | R2 copy (`store_rekey.py --apply`, then `--verify`) | section 3.2 | P1 merged | delete `store/` keys |
| O2 | deploy Worker `singlet-data` on `store/*` | section 3.4, steps 1-2 | W1 merged, O1 verified | delete the new script |
| O3 | D1 migration, then **at once** the Pages deploy of W1 (merge to main deploys) | section 3.3 | O2 smoke-tested | Time Travel restore, then redeploy the previous Pages build |
| O4 | production smoke | `scripts/smoke.sh` (updated), MCP `get_study GSE145370` shows `singlet-store`, the website download panel links `store/` keys, Python and R clients load GSE145370 from `store/` | O3 | as O3 |
| P2 | singlet-bio PR (after #31 merges, if B) | `called_v2` → `called_ed` in C++ (`include/singlet/pileup/called_v2.h` → `called_ed.h`, fixtures, `bench/called_v2*` dirs), pipeline output column names | #31 merged | revert PR |
| C1 | cleanup, after the campaign has written its first full wave and nothing reads `v2/` (Worker logs show no `/v2/` hits for 14 days) | delete `v2/` keys (pilot, ledger, refs; an R2 delete, separately approved), drop the D1 views, remove the `/v2/*` route and the `singlet-data-v2` script, remove the `singlet.v2` alias, the `v2_*` R wrappers, the `series.v2` and export aliases, and old-name reading in the clients | campaign wave 1, 14 quiet days | the deleted objects are copies of `store/` objects; the views and aliases come back with a revert |

The 1.0.0 tag is cut after C1, so the released packages contain no `v2` name. If the tag has to be cut before C1, the
released packages still write only `store` names. What they keep is code that reads the old ones, with no `v2` in any
public name, and C1 removes it in 1.0.1.

## 4. How the PRs are produced

* **Mechanical renames go to Sonnet or Haiku agents with fixed file lists**, taken from the inventory files. These are:
  * `git mv` of the package, scripts, docs, tests and the Worker directory;
  * import rewrites and identifier renames;
  * fixture regeneration;
  * `Rcpp::compileAttributes()` and roxygen;
  * `scripts/sync_r_headers.py`.
* **Done by hand:**
  * the dual-read logic (3.1);
  * `store_rekey.py`;
  * the D1 migration;
  * the Worker routing;
  * the MCP and Docs text.
* **The docs are renamed with `git mv`, keeping history.** Each old path gets a two-line stub pointing at the new one
  until cleanup, so links from other sessions' notes and open PRs keep working.
* **The order inside P1:**
  1. `git mv` and import rewrites (the test suite is green with the alias);
  2. constants and dual read;
  3. public names and aliases;
  4. scripts and docs;
  5. the stray-name test.

  Each is a separate commit, so a review can follow it.

## 5. Tests

* **Old and new readers, both directions.**
  * Python and R unit tests read a fixture study in each tree: an old manifest (`singlet-v2`, `v2/` keys, `.s2`) and a
    new one (`singlet-store`, `store/` keys, `.s1`). Both must give identical matrices and obs.
  * A test asserts that writers never emit an old name: the manifest `format`, key prefix, extension, pack index
    `format`, and (if B) the cell call column.
* **Alias.** `import singlet.v2.cellcall` works and warns. `singlet.v2.cellcall is singlet.store.cellcall`.
* **Byte identity.** `store_rekey.py --verify` (section 3.2) on a local copy of the pilot tree (`bench/compression/corpus`
  plus the downloaded manifest) in CI-free mode. Then the same command against production in O1.
* **Pilot round trip.** `singlet.load("GSE145370")` through `/store/` equals the load through `/v2/` and the legacy bundle
  (X, spliced, unspliced, obs/var names, obs columns). Python and R both run it.
* **Worker.** The existing `data-v2-worker` and `data-v2-cidx` tests are parameterised over both prefixes. The header
  assertion becomes `singlet-data`. A gather request mixing `v2/` and `store/` sample keys is accepted.
* **D1.** A website test applies schema 032, 033 and the new migration to SQLite. It checks two things:
  * old-name queries through the views return the same rows;
  * the new-name DAO reads the rewritten keys.
* **API and MCP.** Snapshot tests for `series.store` and the `series.v2` alias, the export columns, and the MCP
  instructions text containing `singlet-store` and no `v2`.
* **No stray names.** A CI grep test in both repos fails on new `v2` identifiers outside an allowlist. The allowlist
  covers the alias module, the dual-read constants, the 10x chemistry names, the bench and manuscript history and the
  magic-bytes docs. C1 shrinks it to the false positives alone.

## 6. Interactions with open work

| work | touches | conflict | plan |
|---|---|---|---|
| prod/atlas-rerun #31, prod/campaign, prod/modality-fixes #36 (stacked) | adds `include/singlet/pileup/called_v2.h`, `tests/cpp/test_called_v2.cpp`, `bench/called_v2*/` (they import `singlet.v2.cellcall`), golden fixtures | none for P1: they do not edit `python/singlet/v2/`, and the alias keeps their imports working | land #31 / #36 first if they are ready. The C++ `called_v2` rename (P2) follows on main. The campaign must start from a commit that has P1 (and P2 if B), so the objects it writes carry only new names. |
| ADT #29 | edits `docs/format-v2.md`, `python/singlet/v2/client.py`, `r/R/v2.R`; defines `v2/x/<GSM>/adt.<d16>.s2` | real conflict with the `git mv` | merge #29 before P1, or rebase #29 after P1. Git's rename detection carries edits across `git mv`. Its key becomes `store/x/<GSM>/adt.<d16>.s1`. |
| plan/atlas-rerun #30 | `bench/atlas_rerun/cost_model_v2.py` (a cost model revision) | none: it is a model version, not the format | rename optional, to `cost_model.py` |
| cell index, junction table, labels interchange (merged into main via #13) | `singlet.v2.atlas`, `v2/cidx/`, `v2/a/`, `v2/cf/` | covered by P1 / W1 | nothing published under `v2/cidx/` yet. The first build is written to `store/cidx/`. |
| website ontology #54 | new D1 tables (`sample_term`, `sample_property`, `cell_feature`, `sample_cell_summary`) | migration file numbering only | W1's migration takes the next free number after #54 |
| website benchmark #66 | Singlet 1.0 naming, no `v2` files | none | none |
| sessions in flight (adt-modality, vdj-crispr, atac-spatial, mtdna, ensemble classifier, bench-cohort) | may add code under `python/singlet/v2/` or new `v2/...` keys | merge conflicts after P1 | the orchestrator tells them when P1 merges: new code goes in `singlet.store`, new keys under `store/`. The alias keeps their existing branches importable. |
| `public/data/hpc/anvil_v2/` (website), campaign name `v2-human-mouse` | a pipeline campaign label on the HPC dashboard | none | not the format. **Decision C:** leave it (history), or rename the directory for new snapshots. |

## 7. Decisions for Zach

* **A. Magic bytes and the `pz2` codec name.** Recommended: keep them and document them (section 1.1). The alternative
  (A2) is to change them now, at the cost listed there.
* **B. `called_v2`.** Recommended: rename it to `called_ed` / `called_ed_fdr`, with the method version as a field. It is a
  cells-table column and a user-visible obs column, and the campaign will write it into every object, so after the
  campaign it is fixed. Other possible names: `called_emptydrops`, `called_rescue`.
* **C. Other "v2" names that are not the format.** These are the HPC dashboard's `public/data/hpc/anvil_v2/` (campaign
  `v2-human-mouse`), "Pipeline V2" (`scripts/singlet_orchestrator_v2.sh`, unreferenced, and `pileup_engine.h:4`), and
  `test_doublet_detect_v2.cpp`. Recommended:
  * leave the dashboard history;
  * delete the unreferenced orchestrator script in P1;
  * reword the two comments.

  The 10x chemistry names (`10xv2`) are fixed external names and never change.
* **D. Extension.** Decided: `.s1`.
* **E. Old Worker.** Recommended: keep `singlet-data-v2` on `/v2/*` until cleanup (C1), rather than moving the route.
