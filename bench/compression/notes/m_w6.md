# m_w6: migration campaign (W6)

Owner: W6. Paths: `scripts/v2_migrate.sbatch`, `scripts/v2_campaign.py`, `scripts/v2_publish.py`, `scripts/v2_retire.py`,
`tests/python/test_v2_campaign*.py`, this file, `bench/compression/results/m_w6_patches/`. Started 2026-10-22.
**Status (second pass, after W1/W2/W3/W5 landed): aligned with the real transcoder and schema 032/033, rehearsed end to end on the real
corpus bundle with no stand-ins. Nothing has touched R2, D1 or Clipper.** No `git` command was run by W6.

## 0. Status board

| item | state |
|---|---|
| `v2_campaign.py run` against the real `singlet.v2.migrate` / `verify` (flags, output layout, ledger, manifest, pack lists) | done; real run of GSE145370 (section 3) |
| catalog rows and SQL for schema 032/033 (exact columns, apply order, 80,000-row days) | done; equal row by row to what the website's own ingest code writes (section 2) |
| `v2_migrate.sbatch` build step re-checked against `setup.py` (`_pz`, `_pz2`), container self-test in the build | done; Clipper-only parts still unverified (section 5) |
| `v2_publish.py` (new studies), Anvil patch | done; patch regenerated, `git apply --check` passes, nothing applied |
| retirement in three steps (A archive, B catalog flip via `v2_campaign.py retire-sql`, C delete behind catalog age checks), restore, audit; new studies posted as `retired` | done (section 6) |
| tests | 102 in `tests/python/test_v2_campaign*.py`, all pass (section 7) |
| final submission commands | section 6 |

## 1. What changed in this pass

* **Real transcoder** (`m_w2.md`): `migrate --bundle B --out DIR [--gse] [--samples] [--codec] [--created-at]`, `verify --bundle B --v2 DIR
  [--samples] [--quick]`. Output under `DIR/v2/{s,m,ledger,ref}/`. The campaign now (a) merges the transcoder's per-member ledger
  (`v2/ledger/<GSE>.jsonl`, 172 lines for GSE145370: treatment, recipe, verify mode, why a member was stored) into the R2 ledger as
  `migrate:*` events, (b) ignores its `v2/ref/index.json` (the bucket-wide index is published once by `refs-index`), (c) accepts
  `packs` values that are an id, a list of ids (several whitelists) or null, (d) keeps the transcoder's manifest serialisation
  (indent 1, key order) and only updates the `legacy` block it wrote (`etag` and `bytes` from the live object, plus `sha256` and the
  ledger key), (e) for a **new** study (v2_publish) sets `legacy` to null, because the transcoder describes the temporary bundle as
  a legacy object, (f) refuses a v2 object whose manifest lists a restricted member (like the website's endpoint).
* **No thread knob in `migrate`**: `Options.threads = 4` is fixed in `migrate.py` and not reachable from the CLI or the environment,
  so `SINGLET_V2_THREADS` is not set (it would do nothing). A migrate process is one Python thread with short native-encoder bursts, so
  a job runs `cpus - 2` studies at once (`--workers 0`), BLAS threads are pinned to 1 (`--blas-threads`), and
  `SINGLET_PZW_AUTOBUILD=0` stops W3's Windows-only helper build. `--migrate-args` / `--verify-args` pass extra flags (for example
  `--samples`), `--verify-quick` exists but is not for the campaign.
* **Schema 032 + 033** (`singlet-website-v2/schema/`): `catalog-post` and `catalog-sql` follow `v2-catalog.ts` (section 2). `m_w5.md` now
  says the endpoint is manifest based (`POST /api/ingest/v2-study`); that route is kept as the secondary path because no token is available.
* **Plan** changed with the measured speed: a study is one single-threaded migrate process, so a 30 GB bundle needs hours. Studies of
  at least `--big-gb` (5) go to a separate job on a longer partition (section 4).

* **Retirement order changed** (orchestrator, third pass): the old single `retire` command (copy, verify, delete) is gone. `v2_retire.py archive` copies and
  verifies and never deletes (its store guard has no delete permission); `v2_campaign.py retire-sql` flips the catalog; `v2_retire.py delete` removes a
  legacy object only for an archived study whose catalog row has been `retired` >= 5 minutes and `verified` >= 24 hours, judged from an export of D1 that
  the orchestrator provides (the cluster cannot read D1). Studies that never had a legacy file are posted as `retired` directly. Commands: section 6.

## 2. Catalog: rows, SQL, order, budget

`rows/<GSE>.json` holds `{v2_migration, v2_study, v2_sample[]}` written by `run`; `catalog-sql` turns them (or, with `--from-r2`, the
manifests and ledgers in R2) into files under `DIR/dayNN/`.

* **Columns** exactly as `schema/032_v2_catalog.sql` / `033` and `parseSample` / `ingestStatements` in `functions/_shared/v2-catalog.ts`:
  `modalities` is the JSON array of registry names resolved from the member names (the registry is mirrored in `MODALITY_MEMBERS`; a test
  compares it with `modalities.ts` and with `singlet.bundle.MODALITIES` minus the restricted layers), `median_umi` / `median_genes` come
  from `stats`, `members_json` is the member dict as the manifest has it (3.2 KB per sample for GSE145370), `members_by_kind` uses the
  catalog's names (`matrix` / `recipe` / `stored` / `raw`, plus `cells`), `verify_ok` 1, `verified_at` is the time `verify` passed.
  **Equality check**: a test bundles the website's TypeScript with its own esbuild, feeds the POST body this script builds through the
  real `parseIngestBody` + `ingestStatements` into SQLite with the real schema files, applies this script's SQL to a second database, and
  compares every column: 0 differences (14 samples, 2 studies; skipped when node or the worktree are missing).
* **Key of `v2_sample`**: the website's code now writes the 033 key `(gse_id, rev, gsm_id)` and answers 503 if the table still has the 032
  key. `catalog-sql` defaults to 033 (`ON CONFLICT(gse_id, rev, gsm_id) DO NOTHING`: re-applying writes nothing). **Apply 033 before
  these files**; if 033 is not applied the 033 SQL fails with "ON CONFLICT clause does not match" and writes nothing (tested), and
  `--schema 032` (or `D1_SCHEMA=032`) produces the 032 form (`ON CONFLICT(gse_id, gsm_id) DO UPDATE`).
* **Order** (the website's own): first every sample row of the day (`1_sample_*.sql`; with the 033 key they sit beside the served rev
  and nobody reads them), then `2_flip_*.sql`, where each study's three statements stay together in one atomic file: the `v2_study`
  upsert, `DELETE FROM v2_sample WHERE gse_id = ? AND rev != ?`, and the `v2_migration` upsert with status `verified`. So a study is not
  served from v2 until its flip, and `verified` is never applied before its own sample and study rows. A study that failed gets only a
  `failed` migration row; a failed re-run never downgrades a study that was published before (its rows file is left alone).
* **Budget**: whole studies are packed into `day01`, `day02`, ... of at most `--day-budget` (80,000) estimated row writes, counted as 3 per
  sample row, 2 for the study row, 2 for the migration row (the website's notes say 1 and 1 to 2; this is on the safe side). Applying a day
  twice is harmless and cheap (conflicts write nothing). `PLAN.txt` / `plan.json` list studies, samples, writes, files per day; a day
  directory is self-contained (`APPLY.txt` lists the `wrangler d1 execute singlet-catalog --remote --file=dayNN/...` commands in order).
  Files hold at most 1000 statements and 3 MB. A sample row whose SQL would exceed D1's 100 KB statement limit is shrunk by dropping
  optional member fields (`names`, `columns`, `called_source`, `params`, `needs`; never offsets or the chunk index), and a study that
  cannot be made to fit is skipped and named in `PLAN.txt` (exit code 1).
* **Atlas estimate**: about 52,000 sample rows -> 52k x 3 + 8.2k x 4 = about 190,000 writes = **3 days** of 80,000; about 200 MB of SQL (about 25 MB
  as `sql.tar.gz`, which the `catalog-sql` job makes for `clipper.py get`). The retirement (section 6, step B) flips `v2_migration.status` from
  `verified` to `retired` for the archived studies (`UPDATE ... WHERE status = 'verified'`, 3 writes per study, 25k in all, one day). A study that
  never had a legacy file (published by `v2_publish.py`) is posted as `retired` with `verify_ok` 1 from the start, so the site never offers a legacy URL for it.
* `catalog-post` (secondary): `POST /api/ingest/v2-study` with the manifest without `study_files`, slices of at most 400 samples that share
  rev, `n_samples` and `bytes` (`final: false` until the last, which carries `manifest_bytes`, `manifest_sha256` and the migration block),
  User-Agent set, retries on 429/5xx. The body is accepted by the website's real parser (same test).

Sample of what is applied (GSE145370, real run; the SQL path of the rehearsal in section 3):

```
day01/1_sample_0000.sql  (14 statements, 44.7 KB)
INSERT INTO v2_sample (gse_id, gsm_id, key, bytes, sha256, n_columns, n_called, n_members, median_umi, median_genes, modalities, members_json,
  cell_stats_json, rev, updated_at) VALUES ('GSE145370', 'GSM4317423', 'v2/s/GSM4317423/1e71de45f24c1e51985080133bed7ab0.s2', 3184595, '1e71de45...',
  39217, 2722, 11, 106.0, 60.0, '["exon_counts","intron_counts","cell_calls","junctions","splice_psi","mt_heteroplasmy","vdj_gene_usage","summary",
  "pileup_stats","provenance"]', '{"cells":{"kind":2,...},"exon_counts.1pz":{"kind":1,...,"codec":1,"chunk_cols":4096,"n_chunks":10,"chunk_index":[182016,264],...}}',
  '{"median_umi":106.0,"median_genes":60.0}', '31bfd9c808c8f72e', strftime('%Y-%m-%dT%H:%M:%SZ', 'now')) ON CONFLICT(gse_id, rev, gsm_id) DO NOTHING;
day01/2_flip_0000.sql  (3 statements, 2.7 KB)
INSERT INTO v2_study (gse_id, manifest_key, rev, bytes, n_samples, created_at, legacy_bytes, manifest_bytes, manifest_sha256, n_cells_called,
  reference_build, packs_json, pipeline_json, updated_at) VALUES ('GSE145370', 'v2/m/GSE145370.json', '31bfd9c808c8f72e', 31828624, 14,
  '2026-10-22T15:50:42Z', 187066350, 90457, '09a2aa1e...', 40614, 'GRCh38-2024-A', '{"exon":"a34ebca6...","intron":"fd3fcc82...","genes":null,
  "whitelist":"c4661819..."}', '{"git_sha":"8e1803ba...","singlet_version":"0.3.0"}', strftime(...)) ON CONFLICT(gse_id) DO UPDATE SET ...;
DELETE FROM v2_sample WHERE gse_id = 'GSE145370' AND rev != '31bfd9c808c8f72e';
INSERT INTO v2_migration (gse_id, status, legacy_key, legacy_etag, legacy_bytes, v2_bytes, n_samples, members_by_kind, verify_ok, commit_sha,
  migrated_at, verified_at, ledger_key, error, updated_at) VALUES ('GSE145370', 'verified', 'data/GSE145370/GSE145370.singlet',
  '29f661b49cc3bdd631bcea26319fa571', 187066350, 31919081, 14, '{"cells":14,"matrix":56,"recipe":42,"stored":42}', 1, 'abcdef1234567890',
  '2026-10-22T15:50:42Z', '2026-10-22T15:52:31Z', 'v2/ledger/GSE145370.jsonl', NULL, strftime(...)) ON CONFLICT(gse_id) DO UPDATE SET
  status = excluded.status, legacy_key = COALESCE(...), ..., verify_ok = excluded.verify_ok, ..., error = excluded.error, updated_at = strftime(...);
```

`legacy_etag` is stored without quotes (R2 composite ETags for the real bundles, an MD5 for a single-part object); the same value is in the
manifest's `legacy.etag` and is what `v2_retire.py` compares with the live object.

## 3. Real end-to-end rehearsal (no stand-ins)

`v2_campaign.py run --mode full --local-root <dir> --gse GSE145370 --workers 1 --readback sha256` on `bench/compression/corpus/bundles/GSE145370.singlet`
(copied into a directory-backed bucket; the real `python -m singlet.v2.migrate` and `verify`, compiled `_pz2`, W3's recipes; Python 3.11, laptop
i7-1185G7 4C/8T at 3.0 GHz, load 19% before the run, timing lock held; **one run, not a benchmark**; R2 and network are replaced by local disk):

| quantity | value |
|---|---|
| legacy bundle | 187,066,350 B, 14 samples, 143 members |
| v2 sample objects | 31,828,624 B (**5.87x**) + manifest 90,457 B + 3 shared packs 1,147,901 B; study-owned total 31,919,081 B |
| members | 56 pz2 matrices, 14 cells tables, 42 derived, 42 stored; 154 in the manifest |
| objects uploaded | 17 (14 samples, 3 packs) + manifest + ledger (172 transcoder lines, 6 of ours); all read back by SHA-256 |
| `verify` | 1505 checks passed, 0 failed |
| seconds | download 0.4, hash 0.2, **migrate 81.4**, collect (hash 17 objects, validate) 0.2, **verify 26.8**, upload 0.2, manifest 0.0, **total 109.9** |
| throughput | migrate 2.3 MB/s, verify 7.0 MB/s, together **1.7 MB/s of legacy bytes per study process** (about one core) |
| result | status `ok`; legacy sha256 `14fffd79...`, etag recorded in the manifest and the ledger; legacy object untouched |

The same study through `v2_migrate.sbatch` (`RUNTIME=host`, local directories, junctions for `$SRC`): `plan`, `full` (115 s, `workers=4` on a
6-cpu job), `full` again (`skipped_done` in 0 s), `summarize`, `catalog-sql`, `catalog-post --dry-run`, `refs-index` (14 packs + index uploaded, 3
already there; again: nothing) ran as written. The retirement modes were rehearsed the same way after the sequence changed (4 studies from the
stand-in transcoder): `archive-dry`, `archive` in two tasks, `retire-sql`, `delete-dry` / `delete` against an ages file (one study deleted, three
NOT-YET: verified 3 h, retired 2 min, not in the catalog), `audit`, `restore`, `retire-sql` with the restore list.
`catalog-sql` on its rows (section 2) applied to SQLite with the real 032 + 033 files gives 14 `v2_sample` rows, one `v2_study`, one
`v2_migration` with status `verified` and `verify_ok` 1 (and the same after a second application).

## 4. Throughput plan

Measured: 1.7 MB/s of legacy bytes per core for migrate + verify on a quiet laptop core (W2's loaded-laptop run of the whole corpus: about
1 MB/s for migrate alone, 0.6 to 0.7 MB/s on the deep 20 to 50 MB samples, 3 MB/s on the STARsolo ones). Cluster cores are probably not
faster; the table uses 1.0 as the base case. The legacy bundle is read once at 150 MB/s per node (assumed, to be measured), which is never
the limit here. A migrate process is single threaded, so **a study cannot be split across cores**: a 30 GB bundle needs 8 h at 1 MB/s.

Node = 16 cores, `cpus - 2` workers at about 75% efficiency, 8 minutes of start-up per task, all 2.7 TB in non-big studies:

| MB/s per core | node MB/s | campaign node-hours (core-hours) | task length at 32 / 64 / 96 shards | wall at `%16`, 64 shards |
|---|---|---|---|---|
| 0.5 | 6.0 | 134 (2,140) | 4.0 / 2.1 / 1.4 h | 8.3 h |
| **1.0** | 12.0 | 71 (1,140) | 2.1 / **1.1** / 0.8 h | **4.4 h** |
| 1.7 (measured, quiet laptop) | 20.4 | 45 (730) | 1.3 / 0.7 / 0.5 h | 2.8 h |
| 3.0 | 36.0 | 29 (470) | 0.8 / 0.5 / 0.35 h | 1.8 h |

Recommendation: **64 shards of 16 CPUs / 64 GB on `short-cpu`, array `0-63%16`** (first wave `%4`; widen after the pilot's `summarize`), about
42 GB and 128 studies per shard, tasks of about 1 h at the base rate (under the 4 h limit even at 0.5 MB/s, and the deadline guard defers a
study that cannot finish). Total about 1,100 core-hours (range 470 to 2,100), 5 to 10 times what the first estimate of this note said, because
the first estimate assumed 5 MB/s per core.

**Big studies** (at least 5 GB, `plan` writes `big.txt` and the share of the bytes they hold): one worker each, so 5 GB = 1.4 h at 1 MB/s and the
largest (about 30 GB) 8 h. They go in their own array on a longer partition: `--partition=cpu --time=12:00:00 --array=0-7` with `NSHARD=8
GSE_LIST=<big.txt> SCRATCH_BUDGET_GB=1000 MIGRATE_TIMEOUT=43200 VERIFY_TIMEOUT=43200`; each task then runs 14 big studies in parallel. The main
array uses `EXCLUDE_LIST=<big.txt>`. If the listing says the big studies hold most of the bytes, or the `cpu` partition starts days late
(the clipper skill notes it can), split big studies by sample (`migrate --samples` exists) and merge the manifests: a follow-up for W2, not
needed to start.
Per node: scratch about 3x the bundle per study in flight (budget 400 GB; 1000 GB for the big job), memory 2 to 4 GB per worker (W2: the
323 MB STARsolo sample peaks near 2 GB), R2 reads 16 x 1 to 2 MB/s = 16 to 32 MB/s per node, far under the assumed 150 MB/s.
R2 operations about 250k class A (about $1), egress free; v2 adds about 1.6 TB to `singlet-data` until retirement, then 2.7 TB in Infrequent
Access in `singlet-archive` (about $27/month, D12).

## 5. Build step, re-checked against `setup.py` / W1 / W4

`setup.py` builds two `Pybind11Extension`s: `singlet._pz` (`src/bindings/python/_pz.cpp`) and `singlet._pz2` (`_pz2.cpp`, `include_dirs`
`include` + `src/bindings/common`, header-only pz2), both `cxx_std=17`, `libraries=["zstd"]`, `/utf-8` only on Windows, **no `-march`** (pz2
selects AVX2 / AVX-512 at run time from cpuid; the kernels carry per-function target attributes). What the job therefore needs: a C++17 compiler
(the image's GCC 12), Python headers, pybind11, `zstd.h` and a static `libzstd.a`, and `-pthread` (`std::thread` in the encoder). The sbatch:

* builds `libzstd.a` 1.5.6 once (official tarball, checked against the `.sha256` beside it, `make libzstd.a MOREFLAGS=-fPIC`) into
  `$WORK/cache/zstd-1.5.6-static` and points `CPATH` / `LIBRARY_PATH` at it, so `setup.py` is unchanged and the extensions link zstd statically.
  1.5.6 is also the zstd W3 found to write the same `.1pz` frames as 1.5.5 (calibration id `ccd0ac25`), so the legacy writer that `singlet._pz`
  provides reproduces container bytes where the legacy files came from those versions (informational, D15);
* exports `CFLAGS=-pthread LDFLAGS=-pthread` around `build_ext --inplace -j 8`;
* fails early, with a message, if the commit lacks `python/singlet/v2/packs/*.pack` + `index.json`, `python/singlet/v2/recipes_data/`, or
  the `_pz2` source / `setup.py` entry (data files that are easy to leave untracked);
* imports `singlet._pz` and `singlet._pz2` and prints the `.1pz` writer calibration ids (warns if `ccd0ac25` is missing);
* **self-test inside the build lock, before the marker that releases the other tasks**: `python -m pytest tests/python/test_v2_container.py -q`
  (with `SINGLET_PZW_AUTOBUILD=0`) and `--help` of migrate and verify; `MODE=build` additionally runs the campaign tests.

Not verifiable from here: that the image has GCC 12, make and Python headers; the zstd download and its `.sha256` asset; `pip install` of the
dependencies; that `test_v2_container.py` passes with the venv's package versions; that `import singlet` runs under Python 3.12 with the
current dependency pins. `MODE=build` on `debug-cpu` is the first thing to run.

## 6. Procedure: exact commands

`C="python $HOME/.claude/skills/clipper/clipper.py"`, `W=/mnt/projects/debruinz_project/debruinz/singlet`, `SHA=<pushed commit of feat/singlet-v2>`,
`C10=${SHA:0:10}`, `J=$W/src/singlet-py/feat__singlet-v2/scripts/v2_migrate.sbatch` (after `sync.sh` has brought the branch to Clipper).
Prerequisites (orchestrator): the commit is pushed and **contains** `python/singlet/v2/` (packs, recipes_data, `.py`), `src/bindings/python/_pz2.cpp`,
`setup.py` with `_pz2` and these scripts; `~/.config/singlet-r2.env` on Clipper with a token for `singlet-data` (and `singlet-archive` for
the retirement); the restrict migration has finished; schema 032 and 033 applied to D1. Before each submission: `$C free`, `$C when $J`.

```bash
# 1. build + self-test (no R2). Fix whatever it prints before going on.
$C run "sbatch --partition=debug-cpu --time=00:30:00 --array=0 --export=ALL,MODE=build,COMMIT=$SHA $J"

# 2. publish the 16 reference packs and v2/ref/index.json (5.4 MB; idempotent)
$C run "sbatch --partition=debug-cpu --time=00:10:00 --array=0 --export=ALL,MODE=refs-index,APPLY=1,COMMIT=$SHA $J"

# 3. freeze the study list: plan.tsv (all studies, shard column) and big.txt
$C run "sbatch --partition=debug-cpu --time=00:20:00 --array=0 --export=ALL,MODE=plan,NSHARD=64,COMMIT=$SHA $J"
L=$W/state/v2/plan-$C10/plan.tsv ; B=$W/state/v2/plan-$C10/big.txt

# 4. dry run of one shard: real downloads, transcode and verify, no R2 writes
$C run "sbatch --array=0 --export=ALL,MODE=dry,NSHARD=64,LISTING=$L,EXCLUDE_LIST=$B,COMMIT=$SHA $J"

# 5. pilot: 6 studies of at most 2 GB across the size range + the corpus study, with writes and SHA-256 read-back of every object
$C run "sbatch --array=0 --export=ALL,MODE=pilot,PILOT_N=6,GSES=GSE145370,COMMIT=$SHA $J"
$C run "sbatch --partition=debug-cpu --time=00:10:00 --array=0 --export=ALL,MODE=summarize,RUNS=pilot-$C10,COMMIT=$SHA $J"   # per-stage MB/s, projection

# 6. full, normal studies (first wave %4, then `scontrol update JobId=<array> ArrayTaskThrottle=16`)
$C run "sbatch --array=0-63%4 --export=ALL,MODE=full,NSHARD=64,LISTING=$L,EXCLUDE_LIST=$B,COMMIT=$SHA $J"
#    big studies, own array on the long partition (check `$C when` first; use a shorter --time if it starts sooner)
$C run "sbatch --partition=cpu --time=12:00:00 --cpus-per-task=16 --mem=96G --array=0-7 --export=ALL,MODE=full,NSHARD=8,GSE_LIST=$B,LISTING=$L,SCRATCH_BUDGET_GB=1000,MIGRATE_TIMEOUT=43200,VERIFY_TIMEOUT=43200,RUN=full-big-$C10,COMMIT=$SHA $J"
#    resubmit failed or deferred shards with the same command (RESUME=1 is the default); sweep for studies added since the listing was frozen:
$C run "sbatch --array=0-3 --export=ALL,MODE=full,NSHARD=4,COMMIT=$SHA $J"     # no LISTING: a fresh listing; finished studies are skipped

# 7. catalog SQL (primary route): one tar.gz to bring back; the orchestrator applies it with wrangler from the laptop
$C run "sbatch --partition=debug-cpu --time=00:30:00 --array=0 --export=ALL,MODE=catalog-sql,RUNS='full-$C10 full-big-$C10 pilot-$C10',COMMIT=$SHA $J"
$C get $W/state/v2/catalog-sql-$C10/sql.tar.gz <local dir>      # about 25 MB; unpack, then per day: the APPLY.txt of dayNN, in order
#    (MODE=catalog-sql FROM_R2=1 rebuilds the rows from the manifests and ledgers in R2 instead; MODE=catalog-post needs ~/.config/singlet-ingest.env)

# 8. retirement, only after the clients default to v2 (docs/v2-migration.md D12). Three steps; the site never offers a dead legacy URL:
#    (A) archive: copy data/<GSE>/<GSE>.singlet to singlet-archive, verify (size + ETag; SHA-256 read-back when multipart), write the archived list.
#        Deletes nothing; the legacy object is still served.
$C run "sbatch --array=0-7 --export=ALL,MODE=archive-dry,NSHARD=8,COMMIT=$SHA $J"                      # read-only
$C run "sbatch --array=0-7 --export=ALL,MODE=archive,NSHARD=8,COMMIT=$SHA $J"                          # -> $W/state/v2/archive-$C10/archived-<task>.jsonl
#    (B) catalog: SQL from the archived lists; the orchestrator applies it with wrangler from the laptop (verified -> retired, 3 writes per study)
$C run "sbatch --partition=debug-cpu --time=00:10:00 --array=0 --export=ALL,MODE=retire-sql,COMMIT=$SHA $J"   # RUNS defaults to archive-$C10
$C get $W/state/v2/retire-sql-$C10/sql.tar.gz <local dir>       # unpack; run the wrangler lines of APPLY.txt; from then on the site stops offering the file
#    (C) delete from singlet-data, ONLY for archived studies that D1 shows `retired` for >= 5 min AND `verified` for >= 24 h. The cluster cannot read D1,
#        so export the retired rows on the laptop and put the small file on Clipper:
#          wrangler d1 execute singlet-catalog --remote --json --command "SELECT gse_id, verified_at, updated_at FROM v2_migration WHERE status = 'retired'" > ages.json
#          $C put ages.json $W/state/v2/ages.json
#        (CSV with the header gse_id,verified_at,updated_at works too; a `status` column is allowed, only 'retired' rows count)
$C run "sbatch --array=0-7 --export=ALL,MODE=delete-dry,NSHARD=8,AGES=$W/state/v2/ages.json,COMMIT=$SHA $J"   # read-only; NOT-YET lines say what is still waiting
$C run "sbatch --array=0-7 --export=ALL,MODE=delete,NSHARD=8,AGES=$W/state/v2/ages.json,CONFIRM_RETIRE=RETIRE,COMMIT=$SHA $J"
#        run (C) again later (new export of ages.json) until `audit` says nothing is left; studies that are not ready print NOT-YET and exit 0
$C run "sbatch --array=0 --export=ALL,MODE=audit,EXPECT_NONE=1,COMMIT=$SHA $J"                         # exit 1 while any data/*.singlet is left
#    undo for named studies: MODE=restore GSES="GSE... GSE..." APPLY=1 (writes restored-<task>.jsonl into restore-$C10), then step B again with
#    RUNS='archive-<c10> restore-<c10>' flips those studies back to verified; a restored study is skipped by (C).
```

Go / no-go: build prints `_pz and _pz2 import` and the container tests pass; `plan` prints about 8,202 studies and 2.7 TB; dry: every status `dry_ok`
or an explained `blocked_restricted`; pilot: all `ok`, `verify.ok` everywhere, no `etag_note` in the ledgers (if one appears, R2's ETag rule differs
from the documented one; the read-back kept the run safe, but fix `hash_file` before the full run), `summarize` at or above 0.5 MB/s per worker
(if lower, 96 shards); first full wave: `failed` count 0.
Tunables (env): `WORKERS` (cpus - 2), `IO_THREADS` (8), `SCRATCH_BUDGET_GB` (400), `MIN_RATE_MBS` (0.4: a study is not started if it cannot finish at this
rate before the job's deadline), `PILOT_N`, `PILOT_MAX_GB` (2), `RESUME`, `EXCLUDE_LIST`, `GSE_LIST`, `GSES`, `LISTING`, `LIST_ONLY`, `KEEP_SCRATCH=failed`,
`DEADLINE_MARGIN_SEC` (900), `DAY_BUDGET`, `D1_SCHEMA`, `BIG_GB`, `RATE_CPU`, `RUN`. Results and rows: `$W/state/v2/<run>/`; logs `$W/logs/v2mig-<array>_<task>.out`.
Pull only small things: `summarize` output, `plan.tsv`, `big.txt`, `sql.tar.gz`, `results-*.jsonl`.

## 7. Tests (`tests/python/test_v2_campaign*.py`, 102)

`python -P -m pytest tests/python/test_v2_campaign.py tests/python/test_v2_campaign_catalog.py tests/python/test_v2_campaign_retire.py
tests/python/test_v2_campaign_publish.py -q` -> 102 passed (about 3 min on the laptop). Stand-in transcoder follows the real CLI and layout (ledger,
`ref/index.json`, `legacy` block, pack lists); two tests use the real one.

* `test_v2_campaign.py` (38): ETag rule; headers; manifest last; nothing outside `v2/` written; idempotent rerun; resume; every failure mode
  (migrate / verify / upload / transient / legacy replaced / bad names / restricted / missing); dry, list-only, pilot, shards, frozen listing, big-study plan,
  deadline; write guard; `S3Store` against an in-memory S3; plan; summarize; projection.
* `test_v2_campaign_catalog.py` (24): rows and types; restricted and malformed samples refused; modality registry equal to `modalities.ts` and
  `bundle.py`; embedded DDL equal to the website's 032 + 033; SQL quoting; apply order (nothing served while samples load; flip groups atomic); re-apply
  idempotent, new rev replaces the old; 032 variant and the loud failure of the wrong key; day budget keeps whole studies together; oversized rows
  shrunk or skipped; the whole atlas fits 2 to 3 days; rows-dir and `--from-r2` agree; failed studies; restricted member in the v2 object stops the study;
  pack lists; POST slicing, headers, retry, token; **rows equal the website endpoint's output** (TypeScript run through esbuild + node); reference packs
  published once and tampering refused; **the real transcoder and verifier on two samples of the corpus bundle, then SQL into the website's schema files**.
* `test_v2_campaign_retire.py` (29): (A) dry run changes nothing, `--list` required, copy + verify + list with no delete call at all, idempotent, refusals (rebuilt
  bundle, no manifest, missing samples, no verified ledger), completeness gate, a source that changes during the copy is not listed, multipart copy by
  SHA-256, corrupt copy never listed, stable shards; (B) `retire-sql` flips only archived studies, twice is harmless, a restore flips back; (C) dry run
  default and `--apply --confirm RETIRE`, both inputs required, waits (NOT-YET) for the catalog, for 24 h verified and for 5 min retired (thresholds
  inclusive and optional), only archived studies are deleted, re-checks archive / legacy object / v2 objects, ages file formats (CSV, JSON, wrangler JSON,
  status column, timestamp forms), bare study list, a restore cancels the delete; the three steps in order with the catalog in SQLite; restore; audit;
  `S3Store.copy`; A and C through `S3Store`.
* `test_v2_campaign_publish.py` (10): a new study end to end (no `data/` key, `legacy` null, manifest after every object, sidecar to the private bucket
  only when asked, leaked restricted members refused, verify failure, republish uploads nothing, POST and its failure).

## 8. Open items and dependencies

* **Orchestrator**: commit and push (the commit must carry `python/singlet/v2/packs/`, `recipes_data/`, `src/bindings/python/_pz2.cpp`, `setup.py`); R2 token
  scope for `singlet-data` and `singlet-archive` (create it with `wrangler r2 bucket create singlet-archive --storage-class InfrequentAccess`); apply
  033 before any SQL; the restrict migration must be finished; decide the retirement date.
* **W2**: `migrate` has no thread option and runs samples one after another; a `--samples`-based split plus manifest merge would shorten the
  big studies (only needed if `big.txt` holds a large share of the bytes). `pyproject.toml` / `MANIFEST.in` package-data for the packs is also open on your side.
* **W5**: the data Worker's `v2/ledger/` is intentionally not served; `gse.r2_bundle_*` / `gse_meta.has_bundle` for v2-only studies is still W5's
  open item 1 and decides whether the v2-only studies from `v2_publish.py` appear in search (the Anvil patch leaves a marked TODO).
* **Not verified**: any real R2 call (boto3 is not installed on the laptop; `S3Store` ran against an in-memory S3 following R2's documented rules: STANDARD_IA
  on CopyObject/CreateMultipartUpload, If-Match on GET, UploadPartCopy ranges, parts >= 5 MiB and equal, composite ETag; botocore's
  `request_checksum_calculation=when_required` is set as a precaution and is not in R2's docs), the Clipper image toolchain, the zstd download, queue
  behaviour, network throughput, the memory peak of the largest study.
* **Anvil**: needs the `singlet` package of the pinned commit with `singlet.v2` and the compiled `_pz` (the transcoder reads the legacy `.1pz` with it) and
  `_pz2` importable from `pylib_v3`; the patched `cf_pack_v3_prepare.sh` fails fast without them.

## 9. How to resume / change

* A different table layout: `STUDY_COLS` / `SAMPLE_COLS` / `MIGRATION_COLS` and the `sql_*` functions in `v2_campaign.py`; the equality test shows drift.
* Redo studies that are already `ok` (a newer transcoder commit): `--resume` compares only the legacy object, so run
  `--array=0 --export=ALL,MODE=full,NSHARD=1,RESUME=0,GSES="GSE... GSE...",COMMIT=<new sha>`; unchanged objects are skipped by ETag, changed ones get new keys, the
  manifest is replaced last. For everything use `RESUME=0` on all shards.
* A study stuck in a bad state: nothing to clean in R2 (no manifest = invisible); rerun its shard.
* Do not start step A before the clients default to v2. Step C refuses anything the catalog does not show as retired and old enough; `audit --expect-none`
  is the end condition. The age rule is the orchestrator's: 5 minutes since the flip to `retired` and 24 hours since `verified`. If a cached answer built
  *before* the flip can still name the legacy URL for a day, raise `--min-retired-age-minutes` (`MIN_RETIRED_AGE_MINUTES=1445`); nothing else changes.
