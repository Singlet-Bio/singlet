# Rename "v2" to "store" (Singlet 1.0): runbook

Code part merged into main on 2026-10-26 (Singlet-Bio/singlet#39: `singlet.store`, `called_ed`, `.s1`; readers accept both names; `singlet.v2` stays as an alias). The production ops (R2 copy to `store/`, D1 rename, store Worker, website #67) are deferred: nothing was executed in R2, D1 or any Worker. Plan: docs/rename-to-1-0.md on branch `claude/rename-to-1-0-447a9c`.

## Singlet-Bio/singlet#39

This PR holds the plan (`docs/rename-to-1-0.md`, approved 2026-10-25) and its singlet-bio implementation (P1). The website half is zdebruine/singlet-website#67.

**Name changes:**

| old | new |
|---|---|
| `singlet.v2` | `singlet.store` (`singlet.v2` is kept as an alias that emits a `DeprecationWarning`) |
| `scripts/v2_*` | `scripts/store_*` |
| `docs/format-v2*.md` | `docs/format-store*.md` |
| `called-v2*` docs, bench dirs and golden data | `called-ed*` |
| manifest `"singlet-v2"` | `"singlet-store"` |
| pack index `"singlet-v2-packs"` | `"singlet-store-packs"` |
| keys `v2/...`, extension `.s2` | `store/...`, extension `.s1` |
| `called_v2` / `called_v2_fdr` | `called_ed` / `called_ed_fdr` |
| `V2Bundle` etc. | `StoreBundle` etc. (old names kept as aliases) |
| `SINGLET_V2_*` | `SINGLET_STORE_*` (old variables still read as a fallback) |
| R `v2.R` and `singlet_v2_*` conditions | `store.R`; conditions carry both classes |

Also:
- `scripts/singlet_orchestrator_v2.sh` is deleted. It was unreferenced.
- The magic bytes and the `pz2` codec name are unchanged (decision A).

**Dual read:**
- Readers accept the old names: `singlet-v2` manifests, `v2/` keys, `.s2`, `called_v2` and `v2_*` ledger fields.
- A study is looked up at `store/m/` first, then `v2/m/`, then the legacy bundle.
- Writers emit only the new names.

**`scripts/store_rekey.py`** copies the published `v2/` tree to `store/`. It has two modes: S3 CopyObject, or a read over the public origin followed by `wrangler r2 object put`. It rewrites the manifest (with a new rev) and the pack index, emits the D1 rev SQL, and has a verify step. A read-only rehearsal against production planned 178 keys, 48.6 MB in all, with every byte copy sha256-checked. Nothing has been written.

**Tests:**
- `tests/python/test_store_*.py` plus `test_cell_features.py`: 431 passed, 34 skipped, with `_pz2` rebuilt from source.
- `test_store_rekey.py`: 2 passed.
- `test_store_dual_read.py` (new): 11 tests, all passing.
- The R suite was not run locally; CI runs it.

**Not done here:**
- the R2, D1 and Worker steps (O1–O4), which wait for the go-ahead;
- the C++ `called_v2.h` rename on the prod branches (P2, after #31 merges);
- cleanup C1.

## Ops runbook (not run yet)

**State:** nothing has been written to R2, D1 or any Worker. The only R2 activity so far was a read-only `plan` (Clipper job 787575): 178 keys, 48.7 MB, GSE145370 rev `87b5fa4cb0a7a0d3` → `d5349cefc5009dbe`.

Run the steps in this order. Each step needs a go-ahead, and each has a rollback.

### 0. Readers first

Merge Singlet-Bio/singlet#39 once its CI is green. Do **not** merge website #67 yet; it is the last action of step 3.

### 1. R2 copy (Clipper)

Only `store/` keys are written; `v2/` is never touched. Copies are server-side, with Content-Type and Cache-Control set explicitly. The job uses the copy of `scripts/store_rekey.sbatch` already at `jobs/storerekey/` on Clipper; `clipper.py put` it again if it changed.

```bash
C="python ~/.claude/skills/clipper/clipper.py"
SHA=<singlet-bio main sha after #39>
$C run "cd /mnt/projects/debruinz_project/debruinz/singlet && COMMIT=$SHA MODE=plan  sbatch --parsable -A debruinz jobs/storerekey/store_rekey.sbatch"
$C run "cd /mnt/projects/debruinz_project/debruinz/singlet && COMMIT=$SHA MODE=apply sbatch --parsable -A debruinz jobs/storerekey/store_rekey.sbatch"   # apply, then verify: size, ETag, Content-Type, Cache-Control
```

Alternative without Clipper (laptop, wrangler): `store_rekey.py plan --http https://data.singlet.bio --gse GSE145370`, then `store_rekey.py wrangler > apply.sh`, then `bash apply.sh`.

**Rollback:** delete the `store/` keys.

### 2. Worker (laptop, wrangler)

`singlet-data` is deployed alongside the old `singlet-data-v2`, which keeps `/v2/*`. Run this from website #67's branch.

```bash
cd workers/data && npx wrangler deploy && npx wrangler secret put CIDX_INTERNAL_SECRET
curl -sI https://data.singlet.bio/store/ref/index.json | grep -i x-served-by     # singlet-data
curl -sI https://data.singlet.bio/v2/ref/index.json    | grep -i x-served-by     # data-v2
curl -s -o /dev/null -w '%{http_code}\n' -H 'Range: bytes=0-63' https://data.singlet.bio/store/s/GSM4317423/af69be74768d618866100f7acecb9535.s1   # 206
COMMIT=$SHA MODE=verify   # or locally: python -P scripts/store_rekey.py verify --out <plan dir> --http https://data.singlet.bio  (sha256 of every key)
```

**Rollback:** `npx wrangler delete --name singlet-data`.

### 3. D1 and Pages, in one window (laptop, wrangler)

Steps 1 and 2 must come first, because 036 rewrites the catalog keys to `store/`.

```bash
npx wrangler d1 time-travel info singlet-catalog          # RECORD the bookmark
npx wrangler d1 execute singlet-catalog --remote --file=schema/036_store_rename.sql
npx wrangler d1 execute singlet-catalog --remote --file=<plan dir>/catalog.sql     # rev 87b5fa4cb0a7a0d3 -> d5349cefc5009dbe
gh pr merge 67 -R zdebruine/singlet-website --merge       # Pages deploys
```

Between 036 and the Pages deploy, the old build reads the `v2_*` views.

**Rollback:** `npx wrangler d1 time-travel restore singlet-catalog --bookmark=<recorded>`, then redeploy the previous Pages build.

### 4. Smoke

```bash
bash scripts/smoke.sh   # website repo
```

Also check:
- `/api/gse/GSE145370` shows `series.store.format == "singlet-store"` and rev `d5349cefc5009dbe`;
- `singlet.load("GSE145370")` gives the same AnnData through `store/` and through `v2/`;
- MCP `get_study` reports `singlet-store`.

### Later: cleanup C1

After the campaign's first wave and 14 days with no `/v2/` hits:
- delete `v2/` in R2;
- drop the `v2_*` views;
- delete the `singlet-data-v2` script;
- remove the old-name readers and the `singlet.v2` alias.

🤖 Generated with [Claude Code](https://claude.com/claude-code)


## zdebruine/singlet-website#67

This is the website half of the Singlet 1.0.0 rename. The plan is in Singlet-Bio/singlet#39 (`docs/rename-to-1-0.md`).

**What changes:**
- **Worker:** `workers/data-v2` is now `workers/data`, deployed as the script `singlet-data` on the route `data.singlet.bio/store/*` only. It answers with `X-Served-By: singlet-data`. The handler also serves the `v2/` keys and `/v2/cidx/*`, so it can take over `/v2/*` at cleanup. Until then the old script `singlet-data-v2` keeps that route.
- **D1, `schema/036_store_rename.sql`:**
  - `v2_study`, `v2_sample` and `v2_migration` become `store_*`, and `v2_bytes` becomes `store_bytes`.
  - Stored keys are rewritten to `store/...` with `.s1`.
  - Views under the old `v2_*` names keep the deployed build working between the migration and the deploy.
- **API:**
  - `series.store`, with `series.v2` kept as an alias.
  - Export columns in both spellings.
  - `/api/ingest/store-study`, with `v2-study` kept as an alias.
  - `STORE_ORIGIN` and `STORE_TOOLS`, falling back to the `V2_*` variables.
  - The MCP text now says "singlet-store" and "per-sample".
- **Fixtures** are moved to `fixtures/store`. The sample object's bytes are unchanged, because the magic bytes are kept.
- **Readers accept the old names until cleanup:** `singlet-v2` manifests, `v2/` keys and `.s2`.

**Deploy order. Each step needs the orchestrator's go-ahead, and none happens on merge alone:**
1. Copy the R2 `v2/` tree to `store/` with `scripts/store_rekey.py` in singlet-bio.
2. Deploy `workers/data` (`npx wrangler deploy`).
3. Record a D1 Time Travel bookmark, apply schema 036 and the rekey `catalog.sql`, then merge this PR right away. Merging deploys Pages.

Checks: `npm run check` passes, with 774 tests (771 before plus 3 new: the 036 migration, old-format ingest, and the Worker serving `/v2/`).

## Ops runbook (not run yet)

**State:** nothing has been written to R2, D1 or any Worker. The only R2 activity so far was a read-only `plan` (Clipper job 787575): 178 keys, 48.7 MB, GSE145370 rev `87b5fa4cb0a7a0d3` → `d5349cefc5009dbe`.

Run the steps in this order. Each step needs a go-ahead, and each has a rollback.

### 0. Readers first

Merge Singlet-Bio/singlet#39 once its CI is green. Do **not** merge website #67 yet; it is the last action of step 3.

### 1. R2 copy (Clipper)

Only `store/` keys are written; `v2/` is never touched. Copies are server-side, with Content-Type and Cache-Control set explicitly. The job uses the copy of `scripts/store_rekey.sbatch` already at `jobs/storerekey/` on Clipper; `clipper.py put` it again if it changed.

```bash
C="python ~/.claude/skills/clipper/clipper.py"
SHA=<singlet-bio main sha after #39>
$C run "cd /mnt/projects/debruinz_project/debruinz/singlet && COMMIT=$SHA MODE=plan  sbatch --parsable -A debruinz jobs/storerekey/store_rekey.sbatch"
$C run "cd /mnt/projects/debruinz_project/debruinz/singlet && COMMIT=$SHA MODE=apply sbatch --parsable -A debruinz jobs/storerekey/store_rekey.sbatch"   # apply, then verify: size, ETag, Content-Type, Cache-Control
```

Alternative without Clipper (laptop, wrangler): `store_rekey.py plan --http https://data.singlet.bio --gse GSE145370`, then `store_rekey.py wrangler > apply.sh`, then `bash apply.sh`.

**Rollback:** delete the `store/` keys.

### 2. Worker (laptop, wrangler)

`singlet-data` is deployed alongside the old `singlet-data-v2`, which keeps `/v2/*`. Run this from website #67's branch.

```bash
cd workers/data && npx wrangler deploy && npx wrangler secret put CIDX_INTERNAL_SECRET
curl -sI https://data.singlet.bio/store/ref/index.json | grep -i x-served-by     # singlet-data
curl -sI https://data.singlet.bio/v2/ref/index.json    | grep -i x-served-by     # data-v2
curl -s -o /dev/null -w '%{http_code}\n' -H 'Range: bytes=0-63' https://data.singlet.bio/store/s/GSM4317423/af69be74768d618866100f7acecb9535.s1   # 206
COMMIT=$SHA MODE=verify   # or locally: python -P scripts/store_rekey.py verify --out <plan dir> --http https://data.singlet.bio  (sha256 of every key)
```

**Rollback:** `npx wrangler delete --name singlet-data`.

### 3. D1 and Pages, in one window (laptop, wrangler)

Steps 1 and 2 must come first, because 036 rewrites the catalog keys to `store/`.

```bash
npx wrangler d1 time-travel info singlet-catalog          # RECORD the bookmark
npx wrangler d1 execute singlet-catalog --remote --file=schema/036_store_rename.sql
npx wrangler d1 execute singlet-catalog --remote --file=<plan dir>/catalog.sql     # rev 87b5fa4cb0a7a0d3 -> d5349cefc5009dbe
gh pr merge 67 -R zdebruine/singlet-website --merge       # Pages deploys
```

Between 036 and the Pages deploy, the old build reads the `v2_*` views.

**Rollback:** `npx wrangler d1 time-travel restore singlet-catalog --bookmark=<recorded>`, then redeploy the previous Pages build.

### 4. Smoke

```bash
bash scripts/smoke.sh   # website repo
```

Also check:
- `/api/gse/GSE145370` shows `series.store.format == "singlet-store"` and rev `d5349cefc5009dbe`;
- `singlet.load("GSE145370")` gives the same AnnData through `store/` and through `v2/`;
- MCP `get_study` reports `singlet-store`.

### Later: cleanup C1

After the campaign's first wave and 14 days with no `/v2/` hits:
- delete `v2/` in R2;
- drop the `v2_*` views;
- delete the `singlet-data-v2` script;
- remove the old-name readers and the `singlet.v2` alias.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

