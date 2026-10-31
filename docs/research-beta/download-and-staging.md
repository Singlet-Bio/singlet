# Download ceiling and the decoupled fetch → stage → drain pipeline

Written 2026-10-30 by S9. Measured on Clipper (jobs only, nothing on the login node), real SRA data, ~1.3 TB downloaded in total and deleted.
Code: `scripts/atlas_rerun/{stagequeue,fetch_worker,drain_worker,launcher}.py`, harness `scripts/atlas_rerun/dlbench/`, tests
`tests/python/test_stage_pipeline.py`. Raw results: `$WORK/jobs/s9/results/*.jsonl` and `$WORK/jobs/s9/{e2e,scale}_<job>/`.

## 0. Summary

| question | answer |
|---|---|
| How fast can Clipper pull from SRA? | **~0.9 GB/s aggregate (7.2–7.7 Gbit/s), ≈ 75 TB/day.** One node with ≥ 96 ranged streams reaches it; 2, 4 and 8 nodes add nothing. The campaign planned on 150 MB/s (13 TB/day) because it used one stream per run (`prefetch`): a single stream is 45–60 MB/s. |
| Where is the limit? | A shared link upstream of the nodes, not our NICs (10 G bond per node), not NAT (nodes have public IPs, default route is the campus gateway) and not `port`. §1. |
| What does the download cost in CPU? | Decoding SRA to the compact `.1fq` is **2.6 core-s per M pairs** (decode + screen, 50 samples). At the ceiling (18 M pairs/s of Lite) that is **~47 cores**: the fetch tier is CPU-bound before it is network-bound. §3. |
| What is staged? | `.1fq` with binned qualities, zstd inside: **20 B/pair** (p10–p90: 12–28), **0.42× the SRA Lite bytes** (Lite: 48 B/pair). §3. |
| Does the GPU outpace the network? | **No, not at this ceiling.** The network delivers 18 M pairs/s (Lite) or 12 M (full `.sra`); one H100 is 8.7–11.2 M pairs/s at S8's target (0.17 M/s per sample measured today). The earlier premise held at 150 MB/s (3 M pairs/s). §7. |
| Calendar | Download alone: 12–14 days (Lite) to 20 days (full `.sra`) at the ceiling, 16–26 days at 0.7 GB/s. Compute then sets the calendar: 300 CPU cores = 7 M pairs/s (36 days); + one GPU at 10 M/s ≈ the network. §7. |
| Stream straight into the GPU? | No: §8. |
| Prototype | Queue, fetch worker, drain worker and launcher run end to end on Clipper (10 runs → 8 staged, 8 done exactly once, kill -9 mid-sample recovered). §4–6. |

## 1. Phase 1: the bandwidth ceiling

Harness `dlbench/dlbench.py`: fixed 25–30 s windows, N streams of one tool, bytes counted from the allocated blocks of the node-local /tmp
files once a second, rate = bytes between second 8–10 and the end of the window. Files deleted per test. Jobs 793532, 793553, 793563
(`$WORK/jobs/s9/results/{single,curlmax,multi,nodes}.jsonl`); about 0.9 TB moved. Sources: NCBI SRA Lite on NCBI's own host (SDL with alternate
locations), S3 `sra-pub-run-odp` full `.sra`; GCS Lite and S3 Lite (`sra-pub-zq-*`) are requester-pays (HTTP 400/403) and unusable.

**(a) Path.** Compute nodes have their own public IPs (egress IP = node IP, e.g. 35.39.212.71), default route via the campus gateway
35.39.212.126, bond0 = 10 Gbit/s (2 × eno*np*), then 148.61.30.5 → 148.61.0.125 → 192.122.182.137 → 207.72.x (Merit) → Internet2 163.253.x. No NAT, nothing
through `port`. sysctl `tcp_rmem` max 4 MiB, cubic: fine for 100 ms-class paths with many streams.

**(b) One node, MB/s (steady state; node g008, Fri evening).**

| streams | NCBI Lite curl | NCBI Lite aria2 | S3 full curl | S3 full aria2 | prefetch × N runs (NCBI / S3) |
|---|---|---|---|---|---|
| 1 | 50 | 128 | 53 | 63 | 52 / 44 |
| 4 | 212 | 381 | 204 | 205 | 195 / 203 |
| 8 | 217 | 374 | 271 | 324 | 302 / - |
| 16 | 329 | 379 | 442 | 435 | 467 / - |
| 32 | 597 | 377 | 613 | 458 | - |
| 64 | 645 | 366 | 819 | 475 | - |
| 96 / 128 / 192 | 884 / 795 / 939 | - | 872 / 866 / 885 | - | - |

* One stream is 45–60 MB/s on either source. Rate grows linearly to ~16 streams and flattens at 0.87–0.94 GB/s from ~96 streams.
* `aria2c` stops at 375–475 MB/s however many splits (the `-x` limit is 16 connections per URI; repeating the URI did not lift it). `prefetch` has no
  stream option: N parallel prefetches are N streams. Many `curl --range` processes, or Python threads with keep-alive connections (the fetch worker), reach
  the plateau. `s5cmd` works (2.8 GB in ~15 s at 8 parts) but is no better than curl.
* One stream per file with N files (multi mode, the pipeline's pattern): NCBI curl 573 / 732 / 644 MB/s at 16 / 32 / 64 files, S3 curl 478 / 626 / 751.

**(c) Several nodes, aggregate MB/s (curl, all nodes start the same second).**

| nodes | NCBI, 32 streams/node | NCBI, 64 | S3, 32 | S3, 64 |
|---|---|---|---|---|
| 1 | 805 | 719 | 582 | 732 |
| 2 | 810 | 897 | 812 | 842 |
| 4 | 857 | 902 | 883 | 909 |
| 8 | 840 | 960 | 815 | 960 |

The aggregate saturates at 0.85–0.96 GB/s from two nodes on, each node getting 1/k of it (8 nodes: 115–130 MB/s each). **The bottleneck is a shared
~7.5 Gbit/s limit between the cluster and the Internet** (the campus/Merit uplink or its gateway); a single well-driven node already fills it. More fetch nodes
only add redundancy.

**(d) Time of day.** Fri 19:36–19:47 as above (cluster: 144–147 running jobs, no other downloader visible). Sat 03:00 and Sat 13:00 repeats are queued
(jobs 793570/793571) and append to `$WORK/jobs/s9/results/tod.jsonl`; results are added to the PR when they land.

**Recommended download configuration.** NCBI SRA Lite from NCBI's host (SDL with `accept-alternate-locations=yes`), S3 full `.sra` as fallback and where
qualities matter (§9). 64–96 concurrent ranged streams in total (8 MiB blocks, keep-alive, per-block retry), as ~12 jobs × 8 streams (the CPU needs the
jobs, not the network). Do not use `prefetch`, `aria2c` or VDB streaming. Other lab downloads share the same 0.9 GB/s; the launcher should not run next
to them.

## 2. Architecture

```
 fetch tier (CPU, ~12 jobs x 4 cores)         stage ($WORK, bounded)              compute tier (on demand)
 SDL -> ranged GETs -> node /tmp  ->  decode  ->  incoming/ --rename--> ready/ --rename--> claimed/ --> singlet --engine gpu|cpu
 md5 while downloading   .sra/.lite  .1fq        (cap: pause when over)   atomic            heartbeat    .s1 + upload (Beta) -> delete
 raw file deleted                    + screen                                                              staged reads deleted last
                                  launcher (Slurm job, sbatch from a compute node): starts fetch jobs while todo exists,
                                  GPU workers when backlog >= X GPU-hours, CPU workers for overflow
```

Nothing but the staged `.1fq` is shared; raw `.sra` lives in node `/tmp` for the length of its decode; outputs live in `/tmp` until uploaded.

## 3. Fetch tier (`fetch_worker.py`)

Per sample: claim a todo item → wait for room under the stage cap → per run: SDL (Lite on NCBI, else S3 full) → `RangeFetcher` (threads, `pwrite`, a
hasher thread md5s the contiguous prefix while the rest downloads; checked against the SDL md5 and size) → `singlet download <local .sra> --quality binned`
→ delete the raw file → `singlet screen-runs` on the staged runs → `manifest.json` (bytes, pairs, per-run bytes, `decode_core_s_per_Mpair`) → rename into `ready/`.
Retries: 6 per block with back-off, then fall back Lite → full; a failed sample returns to `todo/` up to 3 times; a non-10x run (decode gate rc=2) or a sample
with no GEX run is parked in `failed/<gsm>.fetch` with no further work.

**Measured (job 793624, 63 runs of 1–9 GB Lite, 12 workers × 8 streams × 2 decode threads on one 24-core node; 50 staged, 13 parked):**

| | |
|---|---|
| raw Lite → staged | 219 GB → 91 GB for 4.53 G pairs: **48.4 B/pair raw, 20.1 B/pair staged (ratio 0.42; per-sample p10–p90 12–28 B/pair)** |
| decode + screen | median 3.06, volume-weighted **2.63 core-s/M pairs** (min 2.17, max 9.26) |
| network at start | 900–946 MB/s for the first two minutes (everything downloading) |
| network sustained | 300–690 MB/s, load 25–31 on 24 cores: **decode-bound**, 24 cores ≈ 10 M pairs/s ≈ 0.5 GB/s of Lite |
| parked | 13 of 63 (21%): 7 non-10x gate, 6 no GEX run after the screen. Their bytes are downloaded and wasted. |

Consequences:

* **The fetch tier needs ~47 cores for 0.9 GB/s of Lite** (18.4 M pairs/s × 2.6), ~40 more for full `.sra` equivalently (12 M × 2.6 = 31 cores). It does not
  need the network to be spread over many nodes: 12 jobs × 4 cores. Writing `.1fg` directly from the decoder would cut the `.1fq` writer (35 of 53 s in
  ingest-format.md §2) but loses the qualities; see §9.
* **Waste from the head screen.** The old flow screened the first 1 M pairs by VDB streaming before downloading. SRA files are columnar, so no cheap byte range gives
  the head; here the whole run is fetched and the screen runs on the staged `.1fq`. With 21% of random "single-cell" runs rejected, the plan's own filtering
  (a_empty…d_rest are GEX candidates) must keep this much lower; measure it on the first campaign wave and cap with a cheap pre-filter (ENA `library_strategy`, platform,
  read length) if it is above ~5%.
* **Stage traffic is small:** 18.4 M pairs/s × 20 B = 0.37 GB/s of writes to `$WORK`, 0.45 PB over the whole campaign.

## 4. The stage and queue (`stagequeue.py`)

```
todo/<GSM>.json  → fetching/<GSM>.<w>/ (HB) → incoming/<GSM>.<w>/ → ready/<GSM>/ → claimed/<GSM>.<w>/ (HB) → deleting/ → gone
                                                                                          └→ ready/ (reaped) or failed/<GSM>.<n>/ after 3 claims
```

Every transition is one `rename()` within the same file system (atomic on Clipper's project space), so:

* **Exactly one claimant wins**; the loser sees `FileNotFoundError` and takes the next sample.
* **No half samples:** `ready/` appears only after `manifest.json` is written and the files are fsynced; a dead fetch worker leaves an `incoming/` dir that `reap()` removes
  and a `fetching/` claim whose item goes back to `todo/`.
* **Crash during compute:** the claim's heartbeat (every `HB_STALE_S/5`, default 900 s/5) stops; any worker's `reap()` returns the dir to `ready/` with `attempts + 1`
  (to `failed/` after 3). A worker that was merely paused notices that its claim directory is gone (`Heart.lost`), kills its child and publishes nothing: no sample is processed
  and uploaded twice.
* **Delete last:** `finish()` renames `claimed/` → `deleting/` then removes it, only after the publish hook succeeded.
* **Backpressure:** `wait_for_room()` blocks while ready + claimed + incoming bytes (+ the next item's expected size) exceed the cap; bytes come from manifests, never a `du` walk.
* **Scratch:** workers sweep `s9-fetch-*`/`s9-drain-*` dirs of dead pids at start (SIGKILL cannot clean /tmp; this showed up in the test).

**Tested** (`test_stage_pipeline.py`: exclusive claim, crash/attempt limit, fetch crash, backpressure, launcher policy, range fetcher with injected 503s and md5) and on
Clipper (job 793579): 10 runs, cap 3 GB (fetch paused at the cap, resumed as the drain freed space), 8 staged, 2 parked; drain worker A killed with `-9` while holding a claim, the
claim reaped, drain worker B finished all 8: **8 `done` events, 0 duplicates, 0 failed, stage empty**. Job 793691 repeats it with the final code.

## 5. Compute tier (`drain_worker.py`)

Loop: `reap` → `claim_ready` (oldest first; `--min-pairs`/`--max-pairs` let GPU workers take big samples and CPU workers small ones) → heartbeat thread → run
`singlet <staged .1fq | --runs-dir> … --out-prefix /tmp/… --pipeline --output-format 1pz --engine gpu|cpu` (zstd-decoded on the CPU inside the reader; no nvCOMP anywhere) →
publish hook → delete outputs and staged reads → `done` event in the orchestrator inbox (same events as `stages.compute_stage`, so the ledger is unchanged) → next. A worker
exits after `--idle-exit-min` without work; the index lives as long as the process, but the binary reloads it per sample today.

* **Publish hook.** Default `--publish dry-run` records what would be uploaded (public vs private bytes, files); `--publish CMD {gsm} {gse} {out}` runs the Beta uploader.
  *Interface to settle with S6 (#97):* `s1_producer.py`/`beta-publish` work per **study** (a manifest per GSE), not per sample. To meet "nothing kept on Clipper" a
  per-sample `.s1` producer + idempotent manifest append is needed; until then the hook is where it plugs in. Genotype-bearing `private/` files are counted, never uploaded, deleted.
* **Per-sample fixed cost.** CPU engine, 8 threads, 7–15 M-pair samples: 119–206 s each (job 793579); a linear fit of these against the 66.6 M run (462 s at 9 threads, gpu-cr4-clip.md)
  gives ~100 s fixed start-up (index load) + ~5 s per M pairs. At the campaign mean (309 M pairs) that is 7% on the CPU engine; it matters 3–5× more for a GPU engine that is
  several times faster on the variable part. **S8: a resident-index "session" mode (one process, many samples) is worth ~10–25% of GPU wall.** The worker already records `compute_s`
  per sample to measure it.
* Not tested here: `--engine gpu` (no CUDA build in my path; the flag is passed through, the binary `mc99cpu` accepts it).

## 6. Launcher (`launcher.py`)

A one-core Slurm job (3 h 55 min, queues its own successor with `--dependency=afterany`). `sbatch`, `squeue` and `scancel` work from compute nodes (job 793564: submit and cancel
from a node succeeded; `scrontab` is available but unnecessary). Each tick: `reap`, read the backlog, count its own jobs by name (`s9-fetch`, `s9-drain-gpu`, `s9-drain-cpu`), decide:

* fetch jobs: top up to `fetch_jobs` (12 × 4 cores) while `todo/` is not empty; each job pauses itself at the cap;
* GPU worker: start one when `backlog_gpu_h = ready pairs / R_gpu / 3600 ≥ X`; at most `max_gpu` (1) running or pending;
* CPU workers: the part of the backlog the GPU workers cannot clear within `cpu_overflow_h` (3 h) goes to 16-core `short-cpu` jobs, up to `max_cpu`;
* `PAUSE` file in the stage stops all submissions.

**Choice of X.** A GPU worker lives `queue wait + index load + backlog/R_gpu`. It never waits for data if the stage holds more than it drains while it exists, i.e. the
backlog at start ≥ R_gpu × (queue wait + load) and fetch keeps adding at least as much as it drains. X = **1 GPU-hour** (= 0.76 TB of staged `.1fq` at 10 M pairs/s and 21 B/pair)
covers a 15-minute start-up with a 4× margin and amortises the start-up over ≥ 1 h of work; when the network outruns the GPU (§7) the backlog is always above X and the worker simply runs for
its 3 h 59 min and is replaced. If S8's rate goes up 3×, X in hours stays; in TB it triples.

## 7. Sizing model

Inputs: total 2.16 × 10¹³ pairs (1.58 PB full `.sra` at 73 B/pair; 309 M pairs per sample over 70 k samples); Lite 48.4 B/pair measured (0.66×); staged 20.1 B/pair; fetch
decode 2.6 core-s/M; CPU engine 38.6–44.6 host core-s/M (gpu-cr4-clip.md) → 0.022–0.026 M pairs/s/core; GPU engine: 0.17 M pairs/s per sample measured today (66.6 M pairs in 386 s,
9 threads), S8's node bound 8.7–11.2 M pairs/s per H100 (ws-w11-gpu.md), host bound 18–24 M pairs/s per node.

| quantity | value |
|---|---|
| network, Lite / full | 0.9 GB/s → **18.4 / 12.3 M pairs/s**; 0.7 GB/s sustained → 14.3 / 9.6 |
| download calendar, all Lite (1.06 PB) / rollout "half-Lite" (0.96 PB) / all full (1.58 PB) | 13.6 / 12.3 / 20.3 days at 0.9 GB/s; 17.5 / 15.9 / 26.1 days at 0.7 GB/s (was 74–122 at 150 MB/s) |
| fetch jobs | 12 × 4 cores (47 cores of decode for Lite at the ceiling; 31 for full) |
| S8 today (PR #107, GPU engine, everything GPU-side on) | host 12.5–14.4 core-s/M, GPU busy only 3.6–8% → host-bound: one 54-core node ≈ 3.9–4.3 M pairs/s, ~half of S8's GPU-bound 8.7–11.2 |
| compute rate needed to match the network | 18.4 M pairs/s (Lite) |
| compute calendar | 300 CPU cores: 7 M pairs/s → 36 days; one GPU at 10 M/s: 25 days; one GPU + 300 cores: 17 M/s → 15 days; one GPU at today's 0.17 M/s per sample: not a factor until S8's speedup lands (node-level 2.5 M/s: 100 days) |
| stage size | cap = X + start-up buffer: 0.76 TB (X = 1 GPU-h) + 1.4 TB (one hour of fetch at 18.4 M/s × 21 B while a GPU job waits in the queue) ≈ **2.2 TB; run with a 4 TB cap** (`$WORK` has 47 TB free) |
| GPU duty cycle | stage never empty while compute < network, so ~100% of the time a GPU job exists; between jobs the queue wait is covered by the stage. When compute exceeds network (GPU + CPU > 18 M/s) the stage drains and the launcher reduces to bursts: duty = network pairs/s ÷ compute pairs/s |
| fetch duty when compute-bound | the cap pauses fetch; its CPU and the network are idle for 1 − R_compute/R_network (46% with one GPU at 10 M/s), which is why CPU overflow workers pay |
| stage writes | 0.37 GB/s, 0.45 PB over the campaign |

**What this says about the plan.** With the ceiling at 0.9 GB/s the campaign is **compute-bound again** from the moment S8's GPU engine gets to ~10 M pairs/s per H100: the calendar
is then max(download 12–20 d, compute 15–36 d depending on GPUs/cores), not 94–131 days. Clipper has several H100s; "single GPU per job" still allows 2–3 GPU workers in parallel
(`max_gpu`), which is where the remaining calendar is. Anvil is not needed for bandwidth on this evidence (its network would only matter if it is not behind a similar shared link).

## 8. Streaming download → GPU, skipping the disk?

No, and the reason is not that the GPU outpaces the network (it doesn't, §7):

1. **Rates:** the network (18 M pairs/s) is ~2× one H100 (≤ 11): a streaming GPU would be fed by an uncontrolled source; any dip (other lab jobs share the 0.9 GB/s: single-stream rates were 11–56 MB/s
   in W11's busy windows) idles it. The stage decouples that at a cost of 0.37 GB/s of writes and ~2 TB.
2. **CPU on the GPU node:** decode is 2.6 core-s/M; at 10 M pairs/s that is 26 cores, on a node (54 cores) whose host lane for the GPU engine already needs 14–25 core-s/M and is bound at 18–24 M pairs/s.
   Decoding there would take about half of the host lane.
3. **Gating and retries:** 21% of random runs are rejected only after decode and screen (§3); multi-run samples need all runs' screen before any compute (pooling or `--runs-dir`).
   Failed/retried downloads would stall the GPU.
4. **Exactly-once/restart:** staged samples survive a GPU job ending; a stream does not.

The stage costs ~2 TB of `$WORK` and 0.4 GB/s; streaming would save neither the decode CPU nor the bytes.

## 9. Open decisions and risks

* **Qualities.** The stage keeps binned qualities (`--quality binned`; 20 B/pair; ingest-format.md measured `.1fq` without qualities 14–28 B/pair, binned adds 30–51%, format-1fq.md §4.3).
  The SNP and mtDNA callers need them (donor false-alt 4.3% without), but **SRA Lite carries only simplified qualities**, so Lite plus binned qualities may be uninformative for them. If
  the callers need real qualities the download is the full `.sra` (1.58 PB, 20 days at the ceiling, 12 M pairs/s) and Lite saves nothing. Decision for Zach; both paths work in the code
  (`--quality`, and the SDL source order is one list).
* **Per-sample upload** (S6's `beta-publish` is per study) — §5.
* **GPU build for the drain test:** `--engine gpu` through the queue is untested; the CPU engine through the same worker is. S8's speed-up and a resident-index mode change §7's compute row, not the design.
* **Shared link:** other lab downloads (`jehlee`, `mulaae` arrays were running) and other campus traffic share the 0.9 GB/s; the number may be lower at some hours (time-of-day repeats pending).
  The launcher should refuse to start fetch jobs while a competing big download runs; there is no hook for this yet.
* **Parking waste** (§3) and the `staged/raw` ratio are measured on 50 + 8 samples, not the campaign list.
* **Stage on `$WORK`:** writes of 0.37 GB/s and many small `rename`s are modest; the cap keeps it at ≤ 4 TB. Untested: a fetch burst of several hundred small samples in a minute.
