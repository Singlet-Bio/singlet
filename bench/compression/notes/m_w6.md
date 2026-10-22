# m_w6: migration campaign (W6)

Owner: W6. Paths: `scripts/v2_migrate.sbatch`, `scripts/v2_campaign.py`, `scripts/v2_publish.py`, `scripts/v2_retire.py`,
`tests/python/test_v2_campaign*.py`, this file, `bench/compression/results/m_w6_patches/`. Started 2026-10-22.
**Status: code and tests complete against stand-ins; nothing has touched R2, D1 or a cluster. The first real run is the build job
then the pilot (section 3).** No `git` command was run by W6.

## 0. Status board

| item | state |
|---|---|
| `v2_campaign.py run` (dry / pilot / full, `--shard`, `--gse`, `--gse-list`, `--listing`, `--resume`) | done, 43 tests |
| `v2_campaign.py plan / summarize / catalog-sql / catalog-post / refs-index` | done, tested (SQL is executed in SQLite; POST against a local HTTP server) |
| `v2_migrate.sbatch` (Slurm array wrapper, build once per commit, all modes) | done; every mode rehearsed end to end on the laptop (`RUNTIME=host`, local directories, stand-in transcoder); the Clipper-specific parts (apptainer, zstd source build, venv, `squeue`) are **unverified** |
| `v2_publish.py` (new studies) | done, 10 tests; Anvil patch written and checked with `git apply --check` (nothing applied) |
| `v2_retire.py` (retire / restore / audit) | done, 19 tests including the boto3 code path against an in-memory S3 |
| notes / plan | this file |

Test run: `python -P -m pytest tests/python/test_v2_campaign.py tests/python/test_v2_campaign_retire.py tests/python/test_v2_campaign_publish.py -q`
-> 72 passed (about 4.5 min on the laptop; it starts a Python process per transcode/verify call). No network, no credentials, no
compiled codec needed.

## 1. What each piece does

### `scripts/v2_campaign.py run`

Per study (details in the module docstring): HEAD the live bundle -> ranged read of its zip directory, **refuse if any restricted
member is still in it** (`blocked_restricted`; the restrict migration must have finished) -> download to node-local scratch with ranged
GETs pinned to the HEAD ETag (`If-Match`; R2 implements it on GET) -> SHA-256 -> re-HEAD (a bundle replaced meanwhile is
`legacy_changed`) -> `python -m singlet.v2.migrate --bundle B --out OUT --gse G` -> validate the tree -> add the `legacy` block to the
manifest -> `python -m singlet.v2.verify --bundle B --v2 OUT` (non-zero = `verify_failed`, **nothing uploaded**) -> upload objects ->
ledger -> manifest LAST -> read the manifest back -> `committed` ledger line.

* Never writes outside `v2/` of the public bucket (the store handed to study code is a `GuardedStore`; any other put, copy or delete
  raises `WriteRefused`; deletes are refused everywhere). Legacy objects are only read.
* Idempotent: an object that exists with the same size and ETag (or the same `sha256` user metadata) is skipped. Part size is fixed
  at 8 MiB (the rule R2 documents: ETag of a multipart object = MD5 of the concatenated part MD5s + `-N`; all parts but the last equal;
  minimum 5 MiB), so the ETag is a function of the bytes. Single-part (<= 8 MiB) uploads are checked against the local MD5, multipart
  against the composite ETag. On any ETag disagreement the object is read back and compared by SHA-256 (accepted with an `etag_note`
  in the ledger, so an unexpected R2 ETag rule shows up in the pilot instead of failing the campaign); a SHA mismatch fails the study.
  `--readback sha256` (default in pilot) re-reads every uploaded object.
* Headers: `application/octet-stream` for `.s2`/`.pack`/`.cells`, `application/json` for manifests and `v2/ref/index.json`;
  `Cache-Control: public, max-age=31536000, immutable` for `v2/s/` and `v2/ref/*.pack`, `public, max-age=300` for `v2/m/`,
  `v2/ref/index.json` and `v2/cells/` (what the data Worker sends), `no-store` for the ledger (the Worker does not serve `v2/ledger/`).
* Failure handling: every study is isolated (`process_study` never raises). Statuses: `ok`, `skipped_done`, `missing`,
  `blocked_restricted`, `deferred` (job out of time), `migrate_failed`, `verify_failed`, `collect_failed`, `upload_failed`,
  `legacy_changed`, `failed`, and for dry runs `dry_ok`, `dry_listed`. The exit code is 1 if any study is in a failure status. A failed
  study leaves no manifest (so it is invisible), at most some immutable orphan objects, and an `error` event in its ledger.
* `--resume`: a study is skipped when `v2/m/<GSE>.json` exists whose `legacy.etag`/`legacy.bytes` equal the live object, every sample
  object is present with its size, and the ledger ends in a committed `ok`. A rebuilt legacy bundle (new ETag) is migrated again.
* Concurrency: `--workers` studies in flight (default cores / `--migrate-threads`), a shared `--io-threads` pool for hashing, HEADs and
  uploads, and a scratch byte budget (`--scratch-budget-gb`, 3x the bundle reserved per study). Studies are processed largest first
  inside a shard. `--deadline-epoch` makes the job stop starting studies that cannot finish (assumes 15 MB/s of legacy bytes).
* Outputs per task: `results-<task>.jsonl` (one row per study: status, stage, legacy etag/bytes/sha256, v2 bytes, members by kind,
  verify result, per-stage seconds, upload counts, the transcoder's `migrate_report.json` if any), `rows/<GSE>.json` (catalog rows),
  `ledger/<GSE>.jsonl` (local copy of the R2 ledger lines of this run), `study-logs/<GSE>.log` (transcoder and verifier output).

### `scripts/v2_migrate.sbatch`

Pins `COMMIT` (a pushed commit of `feat/singlet-v2`), adds a detached worktree `$WORK/src/singlet-py/v2mig__<c10>` from the bare clone
`$WORK/repos/singlet-py.git` (the Clipper name of this repo, as in `restrict_published_bundles.sbatch`), and builds it **once per
commit** under a `flock` (array tasks wait for the first one; or run `MODE=build` first, which also runs the campaign tests in the
job). Everything that compiles or runs Python goes through `apptainer exec` on `$WORK/images/python-3.12-bookworm.sif` (`RUNTIME=host`
to bypass).

zstd (the nodes have no zstd headers): the job downloads the official `zstd-1.5.6.tar.gz` from the GitHub release, checks it against
the `.sha256` published beside it, builds a **static `libzstd.a` with `-fPIC`** once into `$WORK/cache/zstd-1.5.6-static`, and the
extensions link it with `CPATH`/`LIBRARY_PATH` (no change to `setup.py`). Why this and not the alternatives: vendoring the source
would put 1.5 MB of third-party code into a repo W1/W4 own and change their build; a conda/module zstd depends on what the cluster
offers; a per-commit build wastes minutes per task. The static library is built once per cluster, independent of git, and the
extension has no runtime dependency on a library path. Cost: needs outbound HTTPS to github.com (allowed from compute nodes) once.
The venv (`$WORK/venvs/v2tools`, Python 3.12 from the image) gets the package's own `[project].dependencies` read from the commit's
`pyproject.toml` (numpy, scipy, pandas, anndata, ...: `import singlet` is eager, so anndata **is** needed), plus boto3, pybind11,
setuptools, pytest. Extensions: `python setup.py build_ext --inplace -j 8` (so `_pz`, and `_pz2` as soon as W4's `setup.py` lists
it). `REQUIRE_PZ2=1` (default) fails the build if `import singlet._pz2` fails; `REQUIRE_PZ2=0` allows the pure-Python codec (much
slower: do not use for the campaign). All caches live under `$WORK` (`XDG_CACHE_HOME`, `PIP_CACHE_DIR`), scratch under node-local `/tmp`.

### `scripts/v2_publish.py`

New studies. Same inputs as `pack_gse` (`--results`, `--catalog`, optional enriched metadata). It runs the packer into scratch (a
legacy-format bundle used only as the verifier's oracle and deleted afterwards; `pack_gse` stays the single definition of what a
study contains), then the same code path as the campaign (`process_study(provider=...)`): transcode, verify, objects, ledger,
manifest last. Restricted members: with `--private BUCKET` the packer's sidecar is uploaded to
`BUCKET/restricted/<GSE>/<GSE>.restricted.singlet` (verified by SHA-256) **before** the manifest; without it they are not kept (as
today); a public bundle that still contains one is refused. Catalog rows are written and, with `--ingest-url` + `$INGEST_TOKEN`,
POSTed. Exit 0 live and catalogued, 1 not published, 3 live but the POST failed (rerun `catalog-post`). Last stdout line `RESULT {json}`.
Why a temporary legacy bundle instead of reading the pipeline directory directly: W2's transcoder takes a bundle; if W2 adds a
directory source, `provider` can drop the packing step without touching anything else.

Anvil patch (unapplied): `bench/compression/results/m_w6_patches/anvil_v2_publish.patch` (`git apply --check` passes on
singlet-website-anvil, which was not modified). It changes `publish_study_v3.py` (v2 publish replaces pack + `upload-bundles`;
`sync-catalog --all` for gse/gsm stays; verification checks `v2/m/<GSE>.json` size and sha256 metadata, optionally the public URL
with `V2_REQUIRE_PUBLIC=1`), `cf_bundle_v3.py` (`--v2` delegates each study to `v2_publish.py`), and `cf_pack_v3_prepare.sh` (copies
`v2_publish.py`/`v2_campaign.py` from the pinned commit and fails fast unless `singlet.v2.migrate`, `singlet.v2.verify` and
`singlet._pz2` import from `pylib_v3`). **Anvil prerequisite**: the `singlet` package of that commit with `singlet.v2` and a compiled
`_pz2` must be importable from `pylib_v3` (today Anvil has only the standalone `bundle.py`); that is a one-time build on Anvil.
`R2_PRIVATE_BUCKET` (env) turns the sidecar upload on.

### `scripts/v2_retire.py`

`retire` (dry run unless `--apply --confirm RETIRE`): refuses unless the manifest exists, its `legacy.etag`/`bytes` equal the live
object, all sample objects exist with the right size and the ledger ends in a committed `ok`; server-side copy to
`singlet-archive/<same key>` with `StorageClass=STANDARD_IA` (R2 lists STANDARD_IA for CopyObject/CreateMultipartUpload; if a call
rejects it the copy is retried without, and the bucket default applies; create the bucket with
`wrangler r2 bucket create singlet-archive --storage-class InfrequentAccess` to be safe), objects over 4 GiB in 1 GiB ranges with
`UploadPartCopy`; verification: size + ETag when the server kept the source ETag, otherwise SHA-256 of the archived object against the
SHA-256 the campaign recorded in the manifest; re-HEAD of the source, then delete. The study list comes from the manifests (plus
legacy studies without one, which are `REFUSED`), so `--shard I/N` keeps meaning the same studies while other shards delete. `retire
--all` refuses to start while any legacy study lacks a manifest (`--allow-partial` overrides). `restore` copies back and verifies
(`--force` to overwrite a different object). `audit` lists every `data/*.singlet` still in `singlet-data` with manifest/archive flags
(`--expect-none` for exit 1). After a restore of a multipart-copied object the ETag differs from the one recorded in the manifest, so
the study needs a new migration before it can be retired again.

### Catalog rows (item 3)

`rows/<GSE>.json` holds `{v2_migration, v2_study, v2_sample[]}` per study. **The table and column names are an assumption**
(`CATALOG_SCHEMA` at the top of `v2_campaign.py`; names from `docs/format-v2.md` section 8 and the ingest endpoint's `gse_id`/`gsm_id`
convention; JSON values are stored as text because the endpoint rejects objects). `bench/compression/notes/m_w5.md` did not exist
when this was written; when W5's schema is final, edit that one table. Two ways to apply, both ordered samples -> migration ->
study (the study row is what makes a study appear):

* `catalog-post --rows-dir ...` (run it on the cluster, `MODE=catalog-post`): `POST /api/ingest/<table>` with `{"rows": [...]}`,
  `X-Ingest-Token`, a non-default User-Agent (the site answers 403 to the default one), 500 rows per request, retries on 429/5xx.
  Needs W5's `TABLES` entries for the three tables and `~/.config/singlet-ingest.env` (`INGEST_TOKEN=...`) on Clipper.
* `catalog-sql --rows-dir ... --out DIR` (or `--from-r2`, which rebuilds the rows from the manifests and ledgers in R2 and needs no
  rows directory): files `1NNNN_v2_sample.sql`, `5NNNN_v2_migration.sql`, `9NNNN_v2_study.sql` of idempotent `INSERT ... ON CONFLICT DO
  UPDATE` (<= 1000 rows and <= 1.5 MB each, no BEGIN/COMMIT: each file is atomic through D1's import) and `APPLY.txt` listing the
  `wrangler d1 execute singlet-catalog --remote --file=...` commands. About 52k sample rows, so roughly 40 files and 40 to 80 MB: pull
  them with `clipper.py pull` only if the POST route is not available.

## 2. Contracts with the other workstreams (what the campaign assumes)

**W2 (`python -m singlet.v2.migrate` / `verify`)**

* `migrate --bundle PATH --out DIR --gse ID`: writes the R2 key space under `DIR/v2/` (`v2/s/<GSM>/<32 hex>.s2`, `v2/m/<GSE>.json`,
  `v2/ref/<32 hex>.pack`, optional `v2/cells/<GSE>.cells`); a layout without the `v2/` prefix (`DIR/m/...`) is also found. Exit 0 on
  success. Optional `DIR/migrate_report.json` (any JSON, < 200 kB) is copied into the result row. Please honour `SINGLET_V2_THREADS`
  (the campaign sets it, and `OMP/OPENBLAS/MKL_NUM_THREADS`, to `--migrate-threads`, default 2).
* `verify --bundle PATH --v2 DIR`: exit non-zero on any mismatch. It is run **after** the campaign adds the `legacy` block to the
  manifest, so it checks the manifest that will be published; it must accept a `legacy` block (`key`, `etag`, `bytes`, `sha256`,
  `migrated_at`, `ledger`) and must not require the block to be absent.
* Checked by the campaign before any upload: object names are `sha256(object)[:32]` (also packs); the header is `SINGLET2` with the
  GSM id and `file_len` at the offsets of format-v2 section 3; manifest `format == "singlet-v2"`, `gse`, a non-empty `samples[]` with
  `key`, `bytes`, optional `sha256` all matching the files; `packs` values are 32-hex ids whose pack is in `DIR` or already in R2.
  Fields used for the catalog: `rev`, `reference_build`, `manifest_version`, `created_at`, `packs`, `samples[].gsm/key/bytes/sha256/
  n_columns/n_called/members{name:{kind,off,len}}/stats`. Members `kind` may be the integer of format-v2 section 3.
* The migrated manifest must carry no restricted member; the campaign refuses bundles that still do (it does not look inside v2).
* Resource: the campaign assumes a migrate+verify of one study fits in about 16 GB (48 GB node job / 3 workers); tell me if the
  largest study (about 30 GB legacy) needs more.

**W1/W4**: `setup.py build_ext --inplace` in the pinned commit must produce `singlet._pz2` (header-only pz2, `-O2`, no `-march`,
zstd linked statically from `$WORK/cache/zstd-1.5.6-static`, Python 3.12, pybind11). Until it exists the build job fails by design.

**W5**: tables `v2_study`, `v2_sample`, `v2_migration` and their ingest `TABLES` entries (columns in `CATALOG_SCHEMA`); how a
v2-only study is recognised in `gse` (today `gse.r2_bundle_key/bytes/n_gsms`, which `upload-bundles` set: the Anvil patch does not
write them: `V2_GSE_FIELDS` is a marked TODO); the data Worker deployed before `V2_REQUIRE_PUBLIC=1` is used; the format of
`v2/ref/index.json` (this campaign writes `{"format","updated_at","packs":[{id,kind,build,member,n,bytes,sha256}]}` from the pack
headers; confirm with W4's pack reader). Note the Worker serves `v2/cells/` with `max-age=300`, format-v2 section 2 says immutable
per rev.

## 3. Procedure (exact commands)

`C="python $HOME/.claude/skills/clipper/clipper.py"`; `W=/mnt/projects/debruinz_project/debruinz/singlet`. Prerequisites the
orchestrator provides: the commit pushed to GitHub (`SHA=$(git rev-parse origin/feat/singlet-v2)`); `sync.sh` run so that the bare
clone `$W/repos/singlet-py.git` and a worktree of the branch exist (`J=$W/src/singlet-py/feat__singlet-v2/scripts/v2_migrate.sbatch`);
`~/.config/singlet-r2.env` on Clipper with a token that reads/writes `singlet-data` (and creates/writes `singlet-archive` for
retirement); the restrict migration finished for every study. Before each submission: `$C free`, `$C when $J --partition=...`.

```bash
# 1. build + self-test the commit (no R2). Fix anything this prints before going on.
$C run "sbatch --partition=debug-cpu --time=00:30:00 --array=0 --export=ALL,MODE=build,COMMIT=$SHA $J"

# 2. freeze the study list and the shard plan (reads R2: HEADs nothing, lists data/)
$C run "sbatch --partition=debug-cpu --time=00:30:00 --array=0 --export=ALL,MODE=plan,NSHARD=32,COMMIT=$SHA $J"
#    -> $W/state/v2/plan-<c10>/plan.tsv  big.txt ; read the log, pull nothing else
L=$W/state/v2/plan-${SHA:0:10}/plan.tsv

# 3. dry run on one shard: real downloads, transcode and verify, no R2 writes
$C run "sbatch --array=0 --export=ALL,MODE=dry,NSHARD=32,LISTING=$L,COMMIT=$SHA $J"

# 4. pilot: 6 studies across the size range + the 14-sample corpus study, with writes and SHA-256 read-back
$C run "sbatch --array=0 --export=ALL,MODE=pilot,PILOT_N=6,GSES=GSE145370,COMMIT=$SHA $J"
$C run "sbatch --array=0 --export=ALL,MODE=summarize,RUNS=pilot-${SHA:0:10},COMMIT=$SHA $J"   # measured MB/s per stage + projection

# 5. full: first wave small, then widen once the pilot's numbers hold
$C run "sbatch --array=0-31%4 --export=ALL,MODE=full,NSHARD=32,LISTING=$L,COMMIT=$SHA $J"
#    watch: squeue; tail -3 $W/logs/v2mig-<jobid>_<n>.out ; MODE=summarize RUNS=full-<c10>
#    then raise the throttle (scontrol update JobId=<arrayid> ArrayTaskThrottle=16) and resubmit failed shards unchanged
#    (RESUME=1 is the default: finished studies are skipped, only failures and deferred studies run again)

# 6. sweep for studies that appeared after the listing was frozen, and retries
$C run "sbatch --array=0-3 --export=ALL,MODE=full,NSHARD=4,COMMIT=$SHA $J"        # no LISTING = fresh listing

# 7. catalog (after W5 deploys the tables); the cluster talks to the site, nothing goes through the laptop
$C run "sbatch --array=0 --export=ALL,MODE=catalog-post,RUNS='full-${SHA:0:10} pilot-${SHA:0:10}',COMMIT=$SHA $J"
#    or MODE=catalog-sql (add FROM_R2=1 to rebuild from R2), then pull $W/state/v2/catalog-sql-<c10>/sql and run APPLY.txt
$C run "sbatch --array=0 --export=ALL,MODE=refs-index,APPLY=1,COMMIT=$SHA $J"

# 8. retirement, only after clients default to v2 (docs/v2-migration.md D12)
$C run "sbatch --array=0-7 --export=ALL,MODE=retire-dry,NSHARD=8,COMMIT=$SHA $J"                      # read-only
$C run "sbatch --array=0-7 --export=ALL,MODE=retire,NSHARD=8,CONFIRM_RETIRE=RETIRE,COMMIT=$SHA $J"
$C run "sbatch --array=0 --export=ALL,MODE=audit,EXPECT_NONE=1,COMMIT=$SHA $J"                        # exit 1 if any data/*.singlet is left
#    undo for named studies: MODE=restore GSES="GSE... GSE..." APPLY=1
```

Tunables (env, all optional): `WORKERS` (3), `MIGRATE_THREADS` (2), `IO_THREADS` (8), `SCRATCH_BUDGET_GB` (400), `PILOT_N`, `PILOT_MAX_GB`
(8), `RESUME`, `EXCLUDE_LIST` (file of GSEs, e.g. `big.txt`, to run the largest studies in their own job with `WORKERS=1
MIGRATE_THREADS=8`), `KEEP_SCRATCH=failed`, `DEADLINE_MARGIN_SEC` (900), `GSES`, `LISTING`, `LIST_ONLY`.
Default job: `short-cpu`, 8 CPUs, 48 GB, 3:59:00, `--no-requeue`. Logs: `$W/logs/v2mig-<arrayid>_<task>.out`.
Results: `$W/state/v2/<mode>-<c10>/`. Pull only small things: `results-*.jsonl`, the `summarize` output, `plan.tsv`, `big.txt`.

Go / no-go after each phase: build ok and `_pz2` imports; plan prints 8,202 studies and about 2.7 TB; dry: every status `dry_ok` or an
explained `blocked_restricted`; pilot: all `ok`, no `etag_note` in the ledgers (if one appears, R2's ETag rule differs from the
documented one: the read-back made the run safe, but fix `hash_file` before the full run), `verify.ok` everywhere, `summarize`
projection within the budget below; first `full` wave: `failed` count 0.

## 4. Throughput plan

Inputs: 8,202 legacy bundles, about 2.7 TB live (mean 329 MB; the largest bundles are about 30 GB; sample objects up to 132 MB).
Work per study: read the bundle once (download), hash it, decode every legacy matrix twice (migrate and verify), encode, decode v2
once (verify), compress the stored members. Download is ~150 MB/s per node (assumed, to be measured); the compute rate per core is
the unknown. From the codec benches (`results/w3_total_timing.tsv`: legacy decode 4 to 11 Mnnz/s per core, about 2.5 bytes per
nonzero) I expect 5 MB/s of legacy bytes per core for migrate+verify together (range 2 to 20, the zstd-19 pass over stored members is
the other candidate). The `plan`/`summarize` subcommands redo this table from measured numbers.

Node = 8 cores at 75% efficiency (6 effective), download 150 MB/s, 32 shards, 6 minutes of start-up per task (reused build):

| MB/s per core (migrate+verify) | node MB/s | bound | campaign node-hours | task length (16 / 32 / 64 shards) | wall at `%8`, 32 shards |
|---|---|---|---|---|---|
| 2 | 12 | compute | 66 | 4.0 / 2.1 / 1.1 h | 8.2 h |
| 5 | 30 | compute | 28 | 1.7 / 0.9 / 0.5 h | 3.5 h |
| 10 | 60 | compute | 16 | 0.9 / 0.5 / 0.3 h | 2.0 h |
| 20 | 120 | compute | 9.4 | 0.5 / 0.3 / 0.2 h | 1.2 h |
| 40 | 150 | download | 8.2 | 0.4 / 0.26 / 0.18 h | 1.0 h |

Download-only floor: 2.7e6 MB / 150 MB/s = 5.0 node-hours.

Recommendation: **32 shards of about 84 GB and 256 studies each (jobs of 0.3 to 2 h in every row above, under the 4 h limit with
the deadline guard), array `0-31%8`**, first wave `%4` to read the per-node download rate off `summarize` before widening. That is 32
jobs of 8 CPUs / 48 GB (256 CPUs when 8 run at once), 9 to 66 node-hours of Clipper in total, a wall time of 1 to 8 hours plus queueing
(the `short-cpu` partition starts at once when `$C free` shows idle nodes). If the pilot shows < 3 MB/s per core, go to 64 shards; if the
node download rate is the limit at 8 concurrent nodes (aggregate 1.2 GB/s against the campus link), keep `%4`.
Tail risk: one 30 GB study is processed by one worker; at 5 MB/s per core and 2 migrate threads that is 50 minutes (2 h at 2 MB/s). Studies
are started largest first; put anything above `--big-gb` (plan writes `big.txt`) in their own job if the pilot says a single study can
exceed an hour.
Resources per job: scratch 3x the bundle per study in flight (3 workers x 30 GB x 3 = 270 GB of the 3.3 TB node disk, budget 400 GB);
memory 16 GB per worker (to be confirmed with W2's peak RSS on the largest sample).
R2: about 52k sample objects (3 to 17 parts each at 8 MiB) -> roughly 250k class A operations (about $1), 40k ranged GETs of the bundles
(class B), egress free; v2 adds about 1.6 TB to `singlet-data` until retirement, then the archive holds 2.7 TB in Infrequent Access
(about $27/month for the 30-day grace period, as D12). D1: about 68k row writes in total.

## 5. What was verified, what was not

Verified (laptop, stand-ins): all behaviours in section 1 by the tests listed in section 6; the sbatch dispatch for build-less modes
(dry, full shards, pilot, summarize, catalog-sql, catalog-post --dry-run, refs-index, plan, retire-dry, retire with and without the
confirmation, audit, restore) run through `bash` with `RUNTIME=host`; `time_left` parsing of `squeue -o %L`; `bash -n` of the sbatch; the
Anvil patch applies (`git apply --check`) and the patched scripts parse; R2 behaviours checked against Cloudflare's public docs
(STANDARD_IA on CopyObject/CreateMultipartUpload/PutObject, If-Match on GetObject, UploadPartCopy with range, CreateBucket/HeadBucket,
multipart rules: >= 5 MiB, equal parts, composite ETag).

**Not verified** (no R2/D1/cluster access by rule): any real R2 call (boto3 is not even installed on the laptop; `S3Store` was exercised
only against an in-memory fake that follows the S3 semantics above); the botocore options `request_checksum_calculation` and
`response_checksum_validation=when_required` (set when the installed botocore knows them, as a precaution against CRC32 trailers
that some S3-compatible stores reject; not mentioned in R2's docs); the apptainer image having a compiler and Python headers; the zstd
source build with `make libzstd.a MOREFLAGS=-fPIC`; the existence of the `.sha256` asset next to the zstd tarball (a missing one only
prints a warning with the computed hash); `pip install` of the dependency list on the nodes; the real `short-cpu` start behaviour; real
throughput of anything. `MODE=build` and the pilot are designed to expose exactly these.

## 6. Tests (`tests/python/test_v2_campaign*.py`)

* `test_v2_campaign.py` (43): ETag rule; headers per key class; full run (manifest is the last data write of each study, only `v2/`
  written, legacy bytes untouched, ledger/rows/results contents); idempotent rerun (0 objects uploaded); `--resume` and a replaced
  legacy bundle; one failing study (migrate / verify / upload / transient upload errors / legacy replaced during or after the read /
  object named wrongly / restricted members / missing study) while the shard continues; dry and list-only write nothing; pilot, shards,
  exclude and frozen listing; deadline deferral; concurrent workers; the write guard; the boto3 path (`S3Store` on an in-memory S3:
  ranged `If-Match` download, multipart ETags, ETag-rule difference accepted only after a SHA-256 read-back, corrupted object rejected);
  catalog SQL executed in SQLite (idempotent, quoting, a failed rerun keeps the study row), POST ordering/retry/token; refs index; plan;
  summarize; projection arithmetic.
* `test_v2_campaign_retire.py` (19): dry run changes nothing; confirmation phrase; archive -> verify -> delete order; refusal for a
  rebuilt bundle, no manifest, missing samples, no verified ledger; completeness gate; source changing during the copy; multipart copy
  verified by SHA-256; corrupt copy rejected; rerun after a crash between archive and delete; restore (and `--force`); audit; stable
  shards; `S3Store.copy` (single, storage-class fallback, multipart ranges, abort on failure); the whole retire flow through `S3Store`
  including bucket creation.
* `test_v2_campaign_publish.py` (10): dry run; v2 objects without any `data/` key; manifest written after every object and the ledger
  around it; restricted sidecar to the private bucket only when asked; a packer that leaks restricted members is refused; verify
  failure; no finished samples; republish uploads nothing; catalog POST and a failed POST (exit 3).
* `test_v2_campaign_common.py`: helpers only (stand-in transcoder/verifier that writes real `SINGLET2` headers, directory-backed R2,
  in-memory boto3 S3).

## 7. How to resume / change things

* Another table layout from W5: edit `CATALOG_SCHEMA` in `v2_campaign.py`; `catalog-sql --from-r2` rebuilds all rows from R2.
* A different transcoder CLI or output layout: `Ctx.migrate_cmd/verify_cmd` (or `--migrate-cmd/--verify-cmd` as a JSON list),
  `find_v2_root`, `collect_objects`.
* A study stuck in a bad state: nothing to clean up in R2 (no manifest = invisible); rerun the shard.
* Redo studies that are already `ok` (for instance after a transcoder fix in a newer commit): `--resume` compares only the legacy
  object, not the code commit, so a new commit does not invalidate finished studies. Run
  `--array=0 --export=ALL,MODE=full,NSHARD=1,RESUME=0,GSES="GSE... GSE...",COMMIT=<new sha>`; the immutable objects whose bytes did not
  change are skipped by ETag, changed ones get new keys, and the manifest is replaced last. For the whole atlas use `RESUME=0` with all
  shards (it re-reads and re-verifies every bundle).
* Do not run `retire` before `catalog-*` and the client switch; `audit --expect-none` is the end condition.
