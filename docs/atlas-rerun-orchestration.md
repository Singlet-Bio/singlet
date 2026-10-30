# Atlas re-run orchestration: one queue, two clusters, one ledger per cluster

Status: built and smoke-tested on Clipper (2026-10-25, 8 wave-a samples). **Anvil is configured but disabled: no Anvil
login, job or SU has been used.** Code: `scripts/atlas_rerun/` (Python standard library only, plus pandas for `plan`).
Plan of record: `docs/atlas-rerun-plan.md` (waves §1, gates §6, schedule §5, quote §4).

## 1. Design

### Partition by study, decided once, on the laptop

`orchestrator.py plan` reads the universe (`bench/atlas_rerun/build_universe.py`) and ENA's run tables and writes
`plan.tsv.gz`: one row per GEX sample with its study, wave, runs, depth, reference, cost-model estimates and **the
cluster that owns its study**. Every sample of a study goes to one cluster. That means:

* the two clusters never compete for a sample, so no cross-cluster lock or shared database is needed (the two
  clusters share no file system, and the laptop is not always online);
* a study is produced by one cluster with one binary, so the per-study hand-off to the v2 packer stays simple
  (atlas-rerun-plan §7: a study is never half old, half new);
* moving work later is an explicit, auditable step (`reassign`, §4.4), not a race.

Order (policy in `config.json`):
* waves a → b → c → d;
* studies within waves a–c in accession order, wave d by study size (cells), largest first;
* samples within a study in accession order.

**A study runs in the earliest wave any of its samples belongs to** (`policy.study_unit`). A study cannot be
republished until all of its samples are re-run. So its wave-d siblings run together with its wave-a/b samples. This
moves ~10k samples of strand-list studies from wave d into wave b. The wave sizes become a 5,323 / b 16,358 /
c 5,259 / d 43,618 (the plan's table counts samples, not studies). Set `study_unit: false` to keep the plan's
per-sample waves. A study is still handed off only once all its samples are done or parked.

Clipper gets everything by default. With `policy.anvil_su_budget > 0`, Anvil gets whole wave-d studies. They are taken
in wave-d order while their summed SU estimate (cost_model_v2 per sample, whole-node charging, +20% retries) fits the
budget. Studies with a sample over 1.3B pairs stay on Clipper, which has 5-day partitions.

### One controller per cluster, the only writer of its ledger

```
<root>/                       Clipper: /mnt/projects/debruinz_project/debruinz/singlet/atlas-rerun/campaign
  plan/plan.tsv.gz (+ .json)  the plan (copied up from the laptop; small)
  bin/singlet-<sha12>         the pinned binary, copied once and checked by sha256 in every job
  inbox/                      events from jobs, one JSON file each, written by temp file + rename (atomic on NFS)
  events.jsonl                append-only log; single writer = the controller holding lock/
  state.json                  fold of events.jsonl up to a byte offset (a cache: `rebuild` recreates it)
  ledger/{samples,runs,studies}.tsv, ledger/status.json   snapshots, rewritten every tick; what the laptop pulls
  batches/*.jsonl             one per submitted array (the array task index selects the line)
  stage/<GSM>/                staged reads (.1fq, screen JSON); deleted when the sample is done
  out/<GSE>/<GSM>/            public outputs + MANIFEST.sha256 (sha256, bytes, path) + screen.json + run.log
  private/<GSE>/<GSM>/        the pipeline's private/ dir (SNP allele counts), mode 0700, PRIVATE.sha256
  handoff/<GSE>.json          per-study hand-off manifest for the v2 packer
  logs/<GSM>/                 stage logs of every attempt;  slurm/  Slurm stdout
  PAUSE, STOP                 operator flags (§3)
```

Jobs never edit shared state: a job writes one event per state change into `inbox/`. The controller is a small Slurm
job (1 core, 4 GB), and it is the only process that touches the ledger. Each tick:
1. folds the inbox into `events.jsonl`;
2. reconciles with `squeue`/`sacct`: a job that vanished without its final event becomes `lost` with the Slurm
   reason (timeout, OOM, node failure), after one tick of grace for file-system lag;
3. retries or parks failed samples;
4. submits new work in plan order, within the configured caps;
5. writes the hand-off manifests and snapshots.

It takes a lease (`lock/owner.json`, renewed each tick) so two controllers never write at once. Before entering its
loop, the controller queues its successor with `--dependency=afterany:<itself>`, so the loop survives the 4-hour
partition limit with no gap.

Every event carries the attempt number. A late event from an older attempt can never override a newer one, for
example from a requeued scavenger job or a job reported lost that later wrote its event.

The laptop is not in the loop. `pull` copies a cluster's four snapshot files; `status` merges any number of them. It
also warns if a sample has work on more than one cluster.

### Sample states

```
queued -> downloading -> screened -> downloaded -> running -> done
     \-> failed(n) -> requeued (once) | parked          parked = terminal until `unpark`
```

The ledger row (`ledger/samples.tsv`) records:
* cluster, attempt, failures, last reason;
* I/O, compute and node job ids, compute class, threads, partition;
* the time of every transition;
* stage seconds: head screen, prefetch, decode, full screen, pool, compute;
* peak RSS;
* reads available (ENA) and used (`.1fq` footer);
* runs planned, kept and dropped, and chemistry;
* SRA bytes, staged bytes;
* binary sha256, output and private dirs, manifest sha256, output bytes, cells.

`ledger/runs.tsv` has one row per run: keep or drop, the reason, chemistry, whitelist rate, reads, SRA bytes and
timings.

## 2. Jobs

### Clipper: per-sample arrays, chained task by task

Each tick submits batches of up to 16 samples with the same compute class:
* an **I/O array**: `short-cpu`, 2 cores, 16 GB, 4 h, `%8` throttle. Samples whose predicted I/O time ×3 exceeds 4 h go to `cpu` with 2 days;
* a **compute array**: `--dependency=aftercorr:<io array> --kill-on-invalid-dep=yes`. Task *i* starts when I/O task *i* succeeds, and is cancelled if I/O task *i* fails or parks.

Global caps: 24 samples in I/O, 75 in compute, 20 TB staged. With `io.requeue` / class `requeue` set, jobs carry
`--requeue`; give them `partition: "short-cpu,scavenger"` to use the scavenger partition. Every stage starts from a
clean directory, so a requeued job is safe.

**I/O stage** (`stages.io_stage`):
1. Streams the first 1M read pairs of every run (`singlet download SRR --max-reads 1000000`). A run that `download`'s
   non-10x gate rejects (exit 2) is dropped with reason `non10x_pregate`.
2. Runs `singlet screen-runs` on those heads. A dropped run is never downloaded in full. If no run is kept, the sample
   is parked as `no_gex_runs` or `non10x`.
3. `prefetch`es each kept run to node-local `/tmp`, then decodes it locally to the shared stage dir
   (`singlet download <.sra>`).
4. Re-runs `screen-runs` on the full local runs. Its JSON is the authoritative `--run-screen` provenance, and its
   `runs_kept` becomes `runs_used`.
5. One kept run: it becomes `sample.1fq`. Several: the per-run `<run>.1fq` files stay as they are and the engine reads them as
   one stream (`--runs-dir <stage dir> --run-screen screen.json`, PR #82), with no decode, re-encode or pooled copy.

Legacy path, kept for one release: `pipeline.pooled_input: true` pools the kept runs into `sample.1fq` (`singlet decode` each
run, then one `singlet encode --no-dedup --no-trim --quality none`). Per-run input gives the same count values as the pool
(evidence in §8); only barcode column order differs.

**Compute stage** (`stages.compute_stage`):
* runs the pinned binary on `sample.1fq` (or `--runs-dir` for a multi-run sample) with `--pipeline` (the SNP panel is on by default and goes to `private/`),
  `--output-format 1pz`, `--bam-sort-mode disk`, a private genome (`--genome-shared off`) and `--run-screen`;
* sets `SINGLET_REF_BASE` when the non-host screen is on (the default, as recommended in plan §9);
* runs in node-local scratch, then copies the outputs.

Output handling:
* `private/` → `private/<GSE>/<GSM>` (0700);
* everything else except BAM and reads → `out/<GSE>/<GSM>`, through a temp dir and a rename;
* then the sha256 manifests are written.

Compute classes are chosen from the planned depth: the first class whose predicted wall time ×1.6 fits its time limit.

| class | threads | partition, limit | memory | samples (full plan) |
|---|---|---|---|---|
| s4 | 4 | short-cpu, 4 h | 96 GB | 42,462 |
| m8 | 8 | short-cpu, 4 h | 112 GB | 16,541 |
| l8 | 8 | cpu, 2 days | 128 GB | 10,867 |
| xl16 (> 1.3B pairs) | 16 | cpu, 5 days | 192 GB | 688 |

The sample counts are for the full plan with `wall_scale` = 2.0 (§7). With the bare model (`wall_scale` = 1) they
are 64,641 / 4,473 / 756 / 688.

Retry escalation:
* after a timeout, the retry uses the next class;
* after an OOM, it uses 1.5× the memory;
* a compute failure with the staged reads still present re-runs only the compute stage.

### Anvil: I/O arrays on `shared`, whole-node packers

* **I/O arrays** on `shared`: 1 core, 1.9 GB (1 SU/h), throttled.
* **Node jobs** on `wholenode`: 128 cores, charged 128 SU/h, 24 h, `--exclusive`, `--signal=B:TERM@600`.

A node job is built only from samples whose reads are already staged, so no dependency is needed. Batches are filled
to 80% of 112 cores × 24 h by predicted core-hours.

`stages.node_stage`:
* loads one genome per reference into `/dev/shm`;
* starts samples longest first while cores are free and their predicted finish ×1.3 is before the job's end
  (4 threads each, 16 for XL);
* releases the samples it did not start back to `downloaded` for the next node (no failure is counted);
* unloads the genome on exit, including on SIGTERM.

**Budget guard** (every tick, before any Anvil submission):
* spent SU = `sacct` over every campaign job since `campaign_start`, charged as Anvil does: whole node = 128 cores;
  `shared` = max(cores, memory/1.9 GB); `highmem` ×4;
* reserved SU = the full time limit of every campaign job still queued or running;
* the guard stops all submissions when spent + reserved + the next job would pass `su_cap`, or when `mybalance` is
  under `su_floor`.

`su_cap: 0`, the shipped value, blocks everything, and so does `enabled: false`. The two are independent switches.

## 3. Operating (Clipper)

Rules: jobs only, `--account=debruinz`, no work on the login node, downloads only inside jobs, LF line ends. The
commands below go through `python ~/.claude/skills/clipper/clipper.py run '...'`. They only call `sbatch`, `touch` and
`rm`, which the login node allows.

```bash
W=/mnt/projects/debruinz_project/debruinz/singlet
D=$W/src/singlet/<branch with / as __>/scripts/atlas_rerun     # a git checkout; code arrives by git
R=$W/atlas-rerun/campaign
```

**Start**
1. Pin the binary in `config.json`: path, sha256 and commit of the PGO build of step 0. Commit and push, then fetch it
   into `$D`'s checkout from a job.
2. Laptop: `python scripts/atlas_rerun/orchestrator.py plan --universe universe.parquet --runs runs.parquet
   runs_extra.parquet runs_extra2.parquet --out plan.tsv.gz`, then `dryrun --plan plan.tsv.gz --cluster clipper`.
   Read the first submissions and the projection.
3. `clipper.py put plan.tsv.gz $R/plan/plan.tsv.gz`, and the same for `plan.tsv.json`.
4. Start the controller:

   ```bash
   mkdir -p $R/slurm
   sbatch --account=debruinz --partition=short-cpu --cpus-per-task=1 --mem=4G --time=03:55:00 \
     --job-name=ar-ctl-clipper --output=$R/slurm/%x-%j.out $D/templates/controller.sbatch \
     $D/orchestrator.py $D/config.json clipper
   ```

**Watch**:
* `python scripts/atlas_rerun/orchestrator.py pull --cluster clipper && python scripts/atlas_rerun/orchestrator.py
  status campaign_ledgers/clipper`;
* or `cat $R/ledger/status.json` through `clipper.py run`.

The status shows:
* states per wave and cluster;
* samples/h and TB/day over 24 h;
* failures by reason;
* ETA at the 24 h rate;
* model core-h left;
* the median ratio of compute time to the cost model;
* for Anvil, SU spent, reserved and projected.

**Pause**: `touch $R/PAUSE`. Running jobs finish and the ledger keeps folding, but nothing new is submitted.
**Resume**: `rm $R/PAUSE`.

**Stop**: `touch $R/STOP`. The current controller exits at its next tick, and the queued successor exits at once
without queueing another. `scancel` campaign jobs only if you need the cores back. The ledger keeps its state, and
lost jobs are retried by the next controller.

**Restart after a stop**: `rm $R/STOP`, then step 4 again.

**Update code**: pause, wait one tick, `git pull` the checkout from a job, then resume. Running jobs read the code at
start.

**Unpark** (an operator decision, for example after a binary fix): `orchestrator.py unpark --cluster clipper --reason
<text>` or `--gsm A,B`, run in a job on the cluster. Each sample comes back with a fresh attempt number.

**Rebuild the fold** (if `state.json` is ever lost or suspect): `orchestrator.py rebuild --cluster clipper`, in a job.

## 4. Failure handling

| what | how it is caught | what happens |
|---|---|---|
| a stage fails (download, prefetch, decode, screen, pool, binary exit) | the stage's `failed` event with reason and log tail | requeue once (`retry_max: 1`), then park as `retries_exhausted(<reason>)` |
| job killed: timeout, OOM, node failure, preemption without requeue | `squeue` loses it, `sacct` gives the state, 2 ticks in a row | `lost` with the reason → retried once, with escalation for timeout and OOM |
| compute failed, staged reads intact | reason starts with `compute:` and `stage/<GSM>/sample.1fq` exists | only the compute stage runs again |
| the binary exits 8 (`strand_contradiction`): on the unsorted stream the reads contradict the protocol orientation, and a pipe cannot be re-read | `compute_stage` sees rc 8 and emits `strand_retry` | the same command runs again once with `--sorted-bam`, whose strand safety net re-reads the BAM and recounts with the opposite orientation. A second exit 8 fails as `strand_contradiction` |
| nothing to run: no 10x run, or no run passes the screen | the stage exits 3 with `parked` | parked at once, no retry. The chained compute task is cancelled |
| binary differs from the pin | sha256 checked at the start of every stage | `binary_mismatch` failure, so the whole campaign parks visibly. Fix the pin, then unpark |
| staging fills | the controller sums staged bytes | no new I/O until the compute stage drains it |
| a node job (Anvil) ends early | `released` events, or `sacct` timeout/cancel | samples go back to `downloaded` without a failure count |
| two controllers | lease in `lock/` | the second one does not tick |

### 4.4 Moving studies between clusters

`reassign --plan plan.tsv.gz --gse G1,G2 --step release --ledger <pulled ledger of the old cluster>` refuses studies
with any sample past `queued`. Otherwise it marks them `none`.

To move them:
1. push the new plan to the old cluster and let it tick; the studies are now inert there;
2. `reassign --step assign --to anvil`;
3. push the plan to both clusters.

## 5. Hand-off to the v2 packer (interface; packing is not done here)

When every sample of a study is `done` or `parked`, the controller writes `handoff/<GSE>.json`
(`schema: singlet.atlas_rerun.handoff/1`) and logs a `study_handoff` event. The manifest holds:
* plan id, binary sha256 and commit, pipeline settings;
* for each sample: state, `out_dir`, `manifest_sha256`, `out_bytes`, reads available and used, runs planned, kept and
  dropped, chemistry, cells, attempt;
* the parked samples with reasons;
* the private root.

What the packer can rely on:
* `out_dir` holds the `.1pz` members and stats written by `--output-format 1pz`, `screen.json` (runs used, per-run
  verdicts), `run.log`, and `MANIFEST.sha256`. The manifest's sha256 is in the ledger, so the packer must check it
  before packing;
* the outputs contain no BAM and no reads.

What the packer must do:
* never read `private/` into a public object. It lives in a separate tree, and its location is in the manifest only so
  that private storage can be arranged;
* treat a study with parked samples as complete. Whether to publish it is the packer's gate decision (plan §6 item 6).

On Anvil, out/ lives on `$SCRATCH`, which is purged after 30 days. Each handed-off study must be shipped to Clipper
project storage or to a private R2 prefix within that time (§6).

## 6. Adding Anvil later: the steps Zach approves

Nothing below has been done. Each step needs Zach's explicit OK. The hpc skill's rule applies: state the SU estimate
before anything over ~1,000 SU.

1. **Validated days on Clipper.** At least several days of the full campaign on Clipper with:
   * status showing compute/model near 1;
   * failures understood;
   * hand-off manifests consumed by the packer.
2. **Stage Anvil's inputs** (needs SSH to Anvil; no SUs):
   * the reference (GRCh38-2024-A, and GRCm39 if mouse studies go there) under `/anvil/projects/x-bio260157/reference`;
   * whitelists and sra-tools;
   * a git checkout of this branch;
   * the pinned binary, built on Anvil from the same commit, or the Clipper binary if it runs there. Its sha256 must
     equal the pin; a different build needs its own pin and is a separate decision.

   `$PROJECT` has ~2.7 TB free, so out/ goes to `$SCRATCH`, with shipping per study.
3. **Decide the non-host screen on Anvil.** It fits a 257 GB node only once its indices are memory-mapped once per node
   (plan §3). Until then `anvil.nonhost` is `false`, and Anvil studies would lack non-host members. Either land the
   mmap change first, or accept the gap for Anvil's studies.
4. **One-node pilot, ~200 SU.**
   * Set `anvil.enabled: true`, `su_cap: 300`, `node.max_nodes: 1`, and a pilot plan of ~30 wave-d samples (`plan
     --only-gse`).
   * Start the Anvil controller (`controller.sbatch` on `shared`, 1 core, 1.9 GB).
   * Measure EPYC 7763 per-core speed against the cost model, the download rate, and node packing (28 slots, memory).
5. **Set the budget.**
   * `policy.anvil_su_budget` (for example 100,000 of the ~112k balance, keeping `su_floor`);
   * re-run `plan` and check the `dryrun --cluster anvil --enable --su-cap N` projection;
   * `reassign` is not needed for studies that have not started.
6. **Go.** Raise `su_cap` to the approved number and `node.max_nodes` (for example 4–6). Keep `status` on both
   clusters; the budget guard stops at the cap by itself.

## 7. Smoke test on Clipper (2026-10-25)

**Setup.**
* 8 wave-a samples from 6 studies, through the real controller (`config.smoke.json`, root `$W/atlas-rerun/smoke`).
* The binary was `pipeline-benchmark/bin/singlet-campaign`: prod/campaign 02aa9b9, sha256 261d3a50…. That is the
  current best binary, not yet the final pinned PGO build.
* Non-host screen on, SNP panel on.
* The controller ran as a short-cpu job ticking every 2 min. I/O ran on 2 cores / 16 GB; compute on 4 threads / 80 GB
  with a private genome.
* Model = cost_model_v2 for one unpacked 4-thread sample: 300 s + 12.79 s per M pairs (screen on, STAR ×1.54).

| sample | runs | pairs used | head screen | prefetch | decode | pool | compute | model | compute/model | peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| GSM6996272 | 4 | 2.1M | 43 s | 12 s | 5 s | 5 s | 122 s | 326 s | 0.37 | 50 GB |
| GSM6996273 | 4 | 1.8M | 39 s | 12 s | 5 s | 4 s | 142 s | 323 s | 0.44 | 50 GB |
| GSM6996274 | 4 | 4.1M | 68 s | 21 s | 8 s | 9 s | 160 s | 352 s | 0.45 | 51 GB |
| GSM6250705 | 1 | 61.5M | 9 s | 70 s | 38 s | – | 732 s | 1,087 s | 0.67 | 55 GB |
| GSM7036403 | 1 | 111.6M | 21 s | 134 s | 102 s | – | 2,196 s | 1,727 s | **1.27** | 58 GB |
| GSM6470208 (5′ v2) | 2 | 121.8M | 28 s | 158 s | 94 s | 670 s | 4,570 s | 1,858 s | **2.46** | 66 GB |
| GSM7656149 (ENA: 0 reads) | 1 | 263.4M | 454 s (failed, fallback) | 711 s | 310 s | – | 5,097 s | 3,669 s | **1.39** | 76 GB |
| GSM7623467 | 1 | – | parked in 4 s: `download`'s gate saw split-seq (R1 = R2 = 151 bp, 0% whitelist) | | | | | | | |

**What worked end to end.**
* Plan → controller → chained arrays → events → ledger → hand-off: 7 done, 1 parked, and all 6 studies have a
  `handoff/<GSE>.json`. `rebuild` from `events.jsonl` reproduced the ledger.
* Per-run screening and pooling of 4-run and 2-run samples, with runs_used provenance in `pileup_stats.json`.
* Parking with a reason, without a retry.
* Requeue after a failure.
* `private/` split off (0700, PRIVATE.sha256), plus the sha256 manifests.
* `pull` and `status` from the laptop.
* Restarting the controller on new code mid-run.
* `unpark` from staged reads.

**Problems found and fixed during the run.**
1. One run's remote head stream (`download --max-reads 1000000`) was OOM-killed at 16 GB twice; the engine does not
   bound memory with `--max-reads` on that run.
   * Fix: a run whose head cannot be read is now prefetched and judged by the full-run screen.
   * Side effect: Slurm then flags the I/O job OUT_OF_MEMORY although it exited 0, and `aftercorr` cancelled the
     compute task. The controller now resubmits compute alone (`released`, no failure count).
2. `star_Unmapped.out.mate1/2` and `nonhost_reads.fq` were written to out/. They are reads: 40–75% of a sample's
   output bytes (850 MB of 1.1 GB at 112M pairs). They are now excluded.
3. Smaller fixes:
   * epoch times lost precision in the TSV snapshots;
   * the status rate window was wrong;
   * pileup `total_reads` (alignment records, 1.4–2.7× the pairs) was overwriting `reads_used`;
   * `encode` did not understand the `.1fq` header's protocol names, so `--protocol` is no longer passed.

**Timing against the model.**
* Small samples run faster than the model (0.4–0.7×): the fixed 300 s is overstated.
* At 112–263M pairs the compute stage took **1.3–2.5× the model** (1.27, 2.46, 1.39). STAR's disk-backed BAM sort
  took 18 and 46 min at 112M and 122M pairs, against 10.5 and 14 min of mapping. At 263M pairs, mapping took 47 min
  against ~31 min modelled, and the sort 16.5 min. The nodes were shared with other jobs' I/O on `/tmp`, and the 5′ sample has a
  larger BAM (22 GB estimated).
* The PBMC benchmark (66M pairs, an idle node) did not show this. Until the 50-sample validation measures it on more
  samples, `wall_scale` = 2.0 in `config.json`. Class choice and `dryrun` use it.
* At 2× the Clipper calendar becomes compute-bound: ~160 days at 300 cores against ~150 days of network. Anvil SU per
  sample would double as well (the `est_su` column is unscaled).

**Memory.**
* Compute peaked at 50–66 GB RSS up to 122M pairs and at 76 GB at 263M pairs (80 GB jobs). The classes now ask for 96/112/128/192 GB. sacct reports the cgroup limit, because the page cache is counted.
* I/O peaked at 3.4 GB for a plain run. Pooling (`encode` of 122M pairs) and the bad head stream reached the 16 GB
  limit.

**Network.** Prefetch reached ~38 MB/s per stream (26.9 GB in 711 s).

## 8. Beta channel, packing, per-run input (S6, 2026-10-30)

**Node packing.** `common.job_slots` computes the concurrent jobs of a node from its memory: `(node_gb * mem_fill -
shared_index_gb) // job_peak_gb`, capped by `cores // threads`. The inputs are in `clusters.<c>.node`: `job_peak_gb` 33.4 and
`shared_index_gb` 17.8 (deep GSM7036403 with the shared index, #86) and `mem_fill` 0.9. A 512 GB node gives 13 jobs, a 768 GB node
16 at 8 threads (cores bind) and 20 at 4 threads on more cores. `node_stage` reads the node's memory from `SLURM_MEM_PER_NODE` or
`/proc/meminfo`.

**Exit 8.** The compute stage reruns once on `--sorted-bam` and emits `strand_retry`; a second exit 8 fails as
`strand_contradiction`. (Supersedes PR #85.)

**Beta publishing** (`beta.py`, `orchestrator.py beta-publish --cluster C --gse G [--dry-run]`, controller `beta_pass` when
`beta.enabled`). A finished study's hand-off goes to `beta.producer` (a command that writes `<work>/<GSE>.manifest.json` and
the `s1/...` objects it names; the default is `s1_producer.py`, which reuses `scripts/s1_publish.py` against a local store:
out/ -> pack_gse bundle -> `singlet.s1.migrate` -> `verify`, so a study is written about twice; needs `beta.catalog`, the
processing catalog parquet; the `PipelineDirBundle` of rollout plan W5 would remove the bundle detour), then to `scripts/beta/beta_ingest.py` of the website repo
(`beta.uploader`), which is the only writer of Beta's R2 and D1. Retried `beta.retries` times; failed studies again on later
ticks up to `beta.max_attempts`. The target must name Beta: host, `beta.bucket` and `beta.database` contain "beta" and are not
a production name, or nothing runs. A manifest naming a `private/`, restricted, BAM, reads or genotype key is refused.

**Evidence (2026-10-30, Clipper job 793180, `bench/atlas_rerun/perrun_vs_pooled.sbatch`).** GSM6470208 (2 runs, 121.8 M pairs),
binary `singlet-pc-tip` (prod/campaign, #82, R11 on in both), the orchestrator's own io and compute stages, `--output-format mtx`.
A = per-run `--runs-dir`, B = the same staged runs pooled (`pool_runs`, 245 s). All 8 count matrices (exon, intron, ambient-corrected,
ambiguous, sj, splice_psi, mt_heteroplasmy, vdj) have the same 118,371 barcodes and 0 differing entries. 70 files identical; the 6 that
differ are the clip5p probe and rRNA samples (read order), the Q30 line (pool re-encoded with `--quality none`), and timings/paths.
Compute wall 1063 s (A) vs 1016 s (B), peak RSS 37.6 GB both, and A skips the pool (245 s, one copy of the reads).

**End to end on a real sample (2026-10-30, Clipper job 793210, `bench/atlas_rerun/e2e_s1_sample.sbatch`).** GSM6470208 through the
orchestrator's io and compute stages (per-run input, 1pz, `singlet-pc-tip`, 23 min total) and then `s1_producer.py`
(`s1_publish.py --local-root`, worktree `bb__prod__campaign`, catalog = `data/atlas-rerun/samples.parquet`, which has `gse_id`,
`gsm_id`, `organism`, `protocol_inferred`, `processing_status` but no series text). pack, migrate and verify exit 0 and write one
64.9 MB `.s1` object and a manifest. **It is not publishable:** the object has `n_columns` 0, `n_called` 0 and empty `stats`
(the migrator found no non-empty exon matrix, although `summary.json` says 10,239 cells called and the same run's mtx output has
118,371 barcodes), and with no release policy it still carries the layers a policy withholds (`mt_heteroplasmy.1pz`,
`mt_variants.tsv`, `ancestry_call.json`, `ambient_*`, `doublet_scores.tsv`). So nothing was sent to Beta. Needed first: the Beta
policy (`s1_publish.py` has no `--alpha-policy` pass-through) and a migrator that reads this engine's out/ (the W5
`PipelineDirBundle`).
