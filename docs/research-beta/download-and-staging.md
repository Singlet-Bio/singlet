# Download ceiling and the decoupled fetch → stage → drain pipeline

Written 2026-10-30/31 by S9. Everything measured on Clipper (jobs only), real SRA data, ~1.8 TB downloaded in total and deleted.
Code: `scripts/atlas_rerun/{stagequeue,ratecap,fetch_worker,drain_worker,sample_publish,prefilter,launcher}.py`, harness `scripts/atlas_rerun/dlbench/`,
test jobs `e2e_stage_pipeline.sbatch`, `e2e_publish_dry.sbatch`, `fetch_tier_scale.sbatch`, `prefilter_eval.sbatch`, `headscreen_eval.sbatch`;
tests `tests/python/test_stage_pipeline.py`. Raw results: `$WORK/jobs/s9/` (`results/*.jsonl`, `e2e_*`, `scale_*`, `pubdry_*`, `prefilter/`).

## 0. Summary

| question | answer |
|---|---|
| Clipper's ceiling from SRA | **~0.9 GB/s (≈ 75 TB/day) at best, 0.65–0.77 GB/s on a busier evening**, shared by all nodes: one node with ≥ 96 ranged streams reaches it, 2/4/8 nodes add nothing. It is a campus-side limit shared with the rest of GVSU, not our NICs/NAT. **Default global cap: 0.6 GB/s** (`stage.json`, one number for all fetch jobs; 0.9 is an edit). §1, §3 |
| What is fetched | **Full-quality `.sra`**, S3 `sra-pub-run-odp` first, NCBI host second; SRA Lite only behind `--allow-lite`. Measured: 70.2 B/pair raw. §3 |
| What is staged | `.1fq`, zstd inside, **binned qualities: 29.4 B/pair (p10–p90 23.6–40.3) = 0.42 × the raw bytes** (Lite: 20.1 B/pair). §3 |
| Fetch tier cost | decode + screen **3.0 core-s per M pairs** → 26 cores for the 0.6 GB/s cap (8.5 M pairs/s), 38 for 0.9. §3 |
| Waste on non-10x / non-GEX runs | **30% of the runs of a random single-cell sample (9 of 30)** are parked after being fetched. Metadata alone cannot fix it (read lengths reject 46% of the good runs). A **1 M-pair head screen before the fetch is exact (73 of 73 runs) and costs a median 28 s**. §3 |
| Pipeline | queue / fetch / drain / launcher run end to end; kill -9 and SIGTERM (preemption) recovered, exactly-once; **`--engine gpu` through the queue works on H100 and V100S** (S5's `singlet-gpu`, sha verified); per-sample Beta publish works in dry run on a real sample. §4–6 |
| **Scavenger has no H100 nodes** | `sinfo`: scavenger = V100S (g001–g004), RTX 8000 (g005, g008) and CPU nodes. The three H100 nodes (g050–g052) are only in `gpu`, `short-gpu`, `debug-gpu`. The launcher defaults to `short-gpu` for H100 workers and `scavenger` for V100S; one config value flips it if the admins add them. §6 |
| Calendar | download alone: **30 days at the 0.6 GB/s cap, 20 days at 0.9** (1.58 PB). Compute is the long pole today (a GPU worker measures ~0.1–0.2 M pairs/s end to end); it matches the cap at ≥ 1.4 M pairs/s per GPU with 6 GPUs. §7 |

## 1. Phase 1: the bandwidth ceiling

Harness `dlbench/dlbench.py`: fixed 25–30 s windows, N streams of one tool, bytes counted from the allocated blocks of the node-local /tmp files once a
second, rate = bytes between second 8–10 and the end of the window. Jobs 793532, 793553, 793563, 793745 (`$WORK/jobs/s9/results/{single,curlmax,multi,nodes,tod}.jsonl`).
Sources: NCBI SRA Lite on NCBI's own host, S3 `sra-pub-run-odp` full `.sra`; GCS Lite and S3 Lite (`sra-pub-zq-*`) are requester-pays (HTTP 400/403): unusable.

**(a) Path.** Compute nodes have their own public IPs (egress IP = node IP), default route via the campus gateway 35.39.212.126, bond0 = 10 Gbit/s, then
148.61.30.5 → 148.61.0.125 → 192.122.182.137 → 207.72.x (Merit) → Internet2. No NAT, nothing through `port`.

**(b) One node, MB/s (steady state, Fri 19:10).**

| streams | NCBI Lite curl | NCBI Lite aria2 | S3 full curl | S3 full aria2 | prefetch × N runs (NCBI / S3) |
|---|---|---|---|---|---|
| 1 | 50 | 128 | 53 | 63 | 52 / 44 |
| 4 | 212 | 381 | 204 | 205 | 195 / 203 |
| 8 | 217 | 374 | 271 | 324 | 302 / - |
| 16 | 329 | 379 | 442 | 435 | 467 / - |
| 32 | 597 | 377 | 613 | 458 | - |
| 64 | 645 | 366 | 819 | 475 | - |
| 96 / 128 / 192 | 884 / 795 / 939 | - | 872 / 866 / 885 | - | - |

One stream is 45–60 MB/s; the rate is linear to ~16 streams and flat at 0.87–0.94 GB/s from ~96. `aria2c` stops at 375–475 MB/s (16 connections per URI), `prefetch` has no stream
option. Many `curl --range` processes or keep-alive threads reach the plateau. Multi mode (one stream per different run, the pipeline's pattern): S3 curl 478 / 626 / 751 at 16 / 32 / 64 files.

**(c) Several nodes, aggregate MB/s (curl, same start second).**

| nodes | NCBI, 32 streams/node | NCBI, 64 | S3, 32 | S3, 64 |
|---|---|---|---|---|
| 1 | 805 | 719 | 582 | 732 |
| 2 | 810 | 897 | 812 | 842 |
| 4 | 857 | 902 | 883 | 909 |
| 8 | 840 | 960 | 815 | 960 |

Saturation from two nodes on; each node gets 1/k. **The limit is a shared ~7.5 Gbit/s upstream link**; a single node fills it, more nodes only add redundancy.

**(d) Time of day.** Fri 19:36–19:47: 0.82–0.96 GB/s (table above). **Fri 22:00 (152 jobs running, `jehlee` 94 and `mulaae` 38 array jobs, plus my own small test downloads): NCBI 767 MB/s and S3 641 MB/s on one
node, 647 MB/s aggregate over four nodes**, i.e. 20–30% lower at the same stream counts. Sat 03:00 and Sat 13:00 repeats are queued (jobs 793570, 793571) and append to `tod.jsonl`; see §10.
The ceiling is therefore ~0.65–0.9 GB/s depending on what the rest of the campus does, which is why the default cap is 0.6.

## 2. Architecture

```
 fetch tier (CPU jobs, 4 cores each)             stage ($WORK, bounded)                  drain workers (GPU/CPU, fungible, preemption-safe)
 [head screen 1M pairs] -> SDL -> ranged GETs -> node /tmp -> decode -> incoming/ -rename-> ready/ -rename-> claimed/ -> singlet --engine gpu|cpu
   global rate cap (stage.json, all jobs)    md5 while downloading   .sra deleted   (cap: pause when over)   atomic     heartbeat    |
                                                                                                                         publish thread (previous sample, beside the next compute):
                                                                                    collect -> s1_producer -> upload objects -> extend live study manifest -> delete staged reads
 launcher (1-core Slurm job; sbatch works from compute nodes): fetch jobs while todo exists; H100 / V100S / CPU drain jobs by backlog and wait
```

## 3. Fetch tier (`fetch_worker.py`) and what it costs

Per sample: claim a todo item → optional head screen → wait for room under the stage cap → per run: SDL → `RangeFetcher` (threads, `pwrite`, md5 of the contiguous prefix while the rest downloads, size + md5 checked,
each block throttled by the shared rate cap) → `singlet download <local .sra> --quality binned` → delete the raw file → `screen-runs` → `manifest.json` → rename into `ready/`.
Sources, in order: full `.sra` on S3 `sra-pub-run-odp`, a full `.sra` on NCBI's host if the SDL lists one, SRA Lite only with `--allow-lite`.

**Global rate cap (`ratecap.py`).** No daemon, no locks: each fetch worker publishes a heartbeat file in `<stage>/rate/` (`{t, active}`, every 2 s) and takes `cap / (workers downloading)` as its share, enforced by a token
bucket shared by its download threads. A worker that is decoding gives its share to the others; a dead or preempted one drops out after 10 s. The cap is read from `<stage>/stage.json` (`rate_cap_MBps`, default 600)
every 2 s, so **0.9 GB/s is `ratecap.write_cap_mbps(stage, 900)` or an edit of that file, no restart.** Validation (job 793706, one node, 6 workers × 16 streams): cap 150 MB/s → NIC receive 160–164 MB/s flat; raised
to 300 mid-run → 290–330 MB/s within seconds. The wire rate is ~7% above the payload cap (TLS/TCP framing and NFS traffic are counted by the NIC), so the default 600 sits at ~640 MB/s on the wire.

**Measured, full `.sra` (job 793706: 30 runs of 1–8 GB Lite size, 21 staged, 9 parked):**

| | full `.sra` | (SRA Lite, job 793624, 50 samples) |
|---|---|---|
| raw bytes | **70.2 B/pair** (189 GB for 2.69 G pairs) | 48.4 |
| staged `.1fq` with binned qualities | **29.4 B/pair** (p10–p90 23.6–40.3) | 20.1 |
| staged / raw | 0.42 | 0.42 |
| decode + screen | **3.0 core-s/M pairs** (volume weighted) | 2.63 |
| parked after the fetch | **9 of 30 (30%)** | 13 of 63 (21%) |

Binned qualities cost +9 B/pair over Lite's simplified ones. The old `.1fq` without qualities was 14–28 B/pair; real qualities add roughly a third.

**Pre-filtering to stop the waste (measured on the 73 runs whose outcome is known: 58 staged, 9 parked at the non-10x gate, 6 parked by the GEX screen).**

* *Metadata (ENA + NCBI SRA XML per-read statistics, `prefilter.py`)* is **not usable**. Many good 10x runs are stored as 2 × 150 or 2 × 101 spots (barcode and cDNA in one read); a "fixed barcode read of 26/28 nt"
  rule rejected 27 of the 58 good runs (46%) while catching 13 of 15 bad ones; keyword rules (`vdj`, `tcr`, `perturb`) reject good GEX libraries from studies that also contain VDJ/CRISPR libraries (5 false rejects). Only
  `library_strategy`/`platform`/`source` are safe, and the plan already applies them. Of 400 random single-cell RNA-Seq runs the length rule would drop 76%: not credible as a filter.
* ***Head screen (`--head-screen`, default on in the launcher)***: stream the first 1 M pairs by VDB (`singlet download --max-reads 1000000`), then `screen-runs`, *before* fetching the run. Job 793716, 8 runs in parallel,
  2 VDB threads each: **73 of 73 verdicts equal the real outcome** (58 keep, 9 non10x, 6 no_gex); wall time **median 28 s (max 68 s) for kept runs, 35 s (max 51) for no-GEX runs, 5 s (max 16) for non-10x runs**.
  Cost-benefit: a kept run pays ~30 s × 2 threads ≈ 60 core-s; a rejected run saves its whole download (≈ 4.6 GB average at 30% rejects) and decode (3.0 core-s/M × 309 M ≈ 930 core-s). At 30% rejects the head screen
  saves ~5× what it costs in CPU and removes the wasted bytes (≈ 30% of 1.58 PB ≈ 0.47 PB would otherwise cross the shared link). **Parked share after the head screen: ~0 by construction** (it is the same screen
  the full run gets; a run whose stream fails or times out is `unknown` and is fetched anyway, then judged by the full screen). Not measured: the head screen inside the running fetch tier (it was tested on its own: job 793716).
  The planned campaign list is already filtered to GEX candidates, so its reject rate is probably below the 30% of a random sample; the screen costs ~60 core-s per kept run either way.

## 4. The stage and queue (`stagequeue.py`)

```
todo/<GSM>.json  → fetching/<GSM>.<w>/ (HB) → incoming/<GSM>.<w>/ → ready/<GSM>/ → claimed/<GSM>.<w>/ (HB) → deleting/ → gone
                                                                                          └→ ready/ (reaped, or SIGTERM: attempt refunded) or failed/<GSM>.<n>/ after 3 claims
```

Every transition is one `rename()` in the same file system (atomic on Clipper's project space): one claimant wins; `ready/` only after the manifest is written and fsynced; a dead fetch worker's item goes back to
`todo/`; a dead drain worker's claim returns to `ready/` when its heartbeat (every `HB_STALE_S/5`, default 300 s) is stale; a paused worker whose claim was reaped notices it (`Heart.lost`) and publishes nothing;
`finish()` renames to `deleting/` then removes. Backpressure: `wait_for_room()` blocks while ready + claimed + incoming bytes exceed the cap. Workers sweep `s9-*` scratch of dead pids at start.

**Preemption (decision 2026-10-30).** On SIGTERM (Slurm preemption, time limit, `--signal=B:TERM@60` forwarded by the batch script's trap) a drain worker kills its child, returns the claim to `ready/` **with the attempt
refunded** and emits `requeued`; a fetch worker returns its item to `todo/` untouched. A SIGKILL is the heartbeat timeout. Nothing is ever uploaded half: objects are content-addressed and re-checked by the server, the
study manifest is written last, the staged reads are deleted only after the publish returned. **Tested on Clipper (jobs 793707 CPU, 793728 H100, 793729 V100S, 793748 H100 with the final code): 6–10 runs, cap 3 GB (fetch paused at the cap),
worker A killed with -9 mid-sample (reaped), worker C killed with SIGTERM mid-sample (exit 143, claim back in `ready/`, attempts 0), worker B finished: every sample has exactly one `done` event, none failed, stage and /tmp empty.**
Not tested live: a real Slurm preemption of a scavenger job (SIGTERM was sent by hand); the `--signal=B:TERM@60` trap in the launcher's batch scripts is untested on Clipper.

## 5. Drain worker, GPU and per-sample publish (`drain_worker.py`, `sample_publish.py`)

Loop: `reap` → `claim_ready` → heartbeat → `singlet <staged .1fq | --runs-dir> … --engine gpu|cpu` (the staged `.1fq` is decoded by the CPU reader; no nvCOMP anywhere) → hand the finished sample to a **publish thread
and claim the next sample at once**, so the GPU does not wait for the producer/upload of the previous sample (the producer + upload took 95–107 s per sample, 15–20% of a 61 M-pair GPU run) → `done` event (same events as
`stages.compute_stage`, ledger unchanged). Scratch may be `/dev/shm`. A worker exits after `--idle-exit-min` without work.

**`--engine gpu` through the queue** (S5's `$WORK/release-gate/bin/d9834b8a/singlet-gpu`, sha256 890fcd92… matches the file beside it): ran on an **H100 NVL (g052, `short-gpu`; jobs 793728, 793748)** and a **Tesla V100S (g001, `scavenger`; job 793729)**,
6 small samples each (7–14 M pairs): all drained exactly once, same preemption results as above. On these small samples the GPU engine is dominated by start-up (130–210 s per sample; the V100S was as fast as the H100).
Real sample **GSM6470208 (61.09 M pairs, 2.80 GB `.sra`) on an H100 (g051): fetch 11 s, decode + screen 81 s (2.59 core-s/M), singlet `--engine gpu` 533 s and 548 s in two runs = 0.112–0.115 M pairs/s end to end**; the number to compare with S8's later work.

**Per-sample Beta publish (`--publish s1` / `s1-dry`)**, built on #97 without editing it: the sample's outputs are collected exactly like the orchestrator does (`stages.collect_outputs`, private/ dropped with the scratch), a one-sample
hand-off is made, `s1_producer.py` (→ `s1_publish.py` pack/migrate/verify, Beta policy) builds the objects (inside the runtime image via `in_runtime.sh`, with the built `s1repo`), the Beta uploader's own `Client`/`upload` primitives push
what the bucket lacks, and then the **live study manifest** (`beta-data.singlet.bio/s1/m/<GSE>.json`, public) is extended with this sample (replacing an older entry of the same sample), `rev` recomputed with the same
`compute_rev` as `singlet.s1.manifest` (a test asserts equality), and posted to `s1-publish` under a per-study lock. Nothing is stored on Clipper. Re-publishing the same objects reproduces the same keys and rev (no writes); **a re-run of the
sample that is not bit-identical gets a new object key (the two GPU runs above gave rev 9510b829… and e320751d…) and replaces the old one in the manifest, which is the uploader's documented behaviour** (orphan objects stay in the bucket).
Target checks are #97's (`beta.check_target`: Beta host/bucket/database only; manifest keys must be public `s1/…`). Verified in **dry run** on the real sample above (jobs 793741, 793749): producer ok, local SHA-256/size/key-digest checks ok,
merged manifest for a study of 1 sample, nothing written. Found and fixed on the way: sourcing `in_runtime.sh` with `bash -c 'source "$0"'` runs the command once during the `source` (its `BASH_SOURCE == $0` guard) and
again explicitly, so the producer ran twice; the wrapper now sources `$1`. **Not run live: no token was used.** Assumption to confirm with S6: `study_files` (feature vocab, study meta) of the newest one-sample manifest are valid for the whole study.

## 6. Launcher (`launcher.py`)

A one-core Slurm job (3 h 55 min, queues its own successor with `--dependency=afterany`); `sbatch`/`squeue`/`scancel` work from compute nodes (job 793564), `scrontab` exists but is not needed. Every submitted job has `--no-requeue`,
`--signal=B:TERM@60` and a TERM trap that forwards to the worker. Per tick: `reap`, backlog, count its own jobs by name and pending age, decide (unit-tested as a pure function; no live submission was made):

* **fetch**: `fetch_jobs` (6 × 4 cores = 24 cores of decode for 8.5 M pairs/s) while `todo/` is not empty, `--head-screen` on;
* **H100 workers** (`s9-drain-h100`, one GPU each, up to 6 = 3 nodes × 2): start only when the backlog is ≥ X H100-hours (X = 1), partition `h100_partition` (default **`short-gpu`**, because the scavenger partition contains no H100);
* **V100S workers** (`s9-drain-v100`, `scavenger`, up to 4): only when the H100 pool is starved, i.e. a pending H100 job has waited more than `fallback_wait_h` (1 h) and the running + young-pending capacity cannot clear the backlog in `overflow_h` (3 h);
* **CPU workers** (`s9-drain-cpu`, 16 cores, `short-cpu`, up to 8): the remaining excess;
* long waits therefore never block progress, and fallback workers need no cancelling: they exit after `idle_exit_min` when the stage is drained. A `PAUSE` file stops submissions. `publish` (`dry-run` | `s1-dry` | `s1`) is a pipeline value;
  `s1` also exports the ingest token from `~/.singlet/beta_ingest_token` into the job's environment only.

**X = 1 H100-hour** covers a 15-minute start-up with a 4× margin and amortises it over ≥ 1 h of work.

## 7. Sizing model (full `.sra`, cap 0.6 GB/s, compute as the long pole)

Inputs: total 2.16 × 10¹³ pairs (1.58 PB at 73 B/pair; 70.2 measured); staged 29.4 B/pair; decode 3.0 core-s/M; GPU worker end to end 0.115 M pairs/s (measured above), 0.17 M pairs/s (S8, one sample, 9 threads); S8's
host cost with everything GPU-side on: 12.5–14.4 core-s/M, GPU busy 3.6–8% (PR #107), so one 56-core H100 node (2 GPUs) is host-bound at ≈ 4 M pairs/s, 2 M per GPU; GPU-bound target 8.7–11.2 M pairs/s per H100; CPU engine 38.6–44.6 core-s/M.

| quantity | at the 0.6 GB/s cap | at 0.9 GB/s |
|---|---|---|
| network in pairs/s (70.2 B/pair) | **8.5 M** | 12.8 M |
| download calendar, 1.58 PB | **30.5 days** | 20.3 days |
| fetch decode cores (3.0 core-s/M) | 26 (6 jobs × 4) | 38 (10 jobs × 4) |
| stage inflow (29.4 B/pair) | 0.25 GB/s, 0.64 PB over the campaign | 0.38 GB/s |
| waste without the head screen (30% of a random sample) | +0.47 PB, ~+13 days of link | |

Compute calendar for 2.16 × 10¹³ pairs with N GPU workers at R M pairs/s each:

| R per worker | 6 workers (all 3 H100 nodes) | workers needed to match 8.5 M/s |
|---|---|---|
| 0.115 (measured end to end today) | 500 days | 74 |
| 0.17 (S8, one sample) | 340 days | 50 |
| 1.0 | 42 days | 9 |
| 1.4 | **30 days = the cap's download time** | 6 |
| 2.0 (host-bound at today's 14 core-s/M with 28 cores per GPU) | 21 days (network-bound at 0.9) | 4 |
| 10 (S8 GPU-bound target) | 4 days | 1 |

* **Compute is the long pole until a worker delivers ≥ 1.4 M pairs/s.** Today it is 7–12× short; this is S8's job. 300 CPU cores add 7 M pairs/s (36 days alone) and are the cheapest way to cover the gap now (CPU workers use the same queue).
* **Stage size.** Compute-bound: the stage is always full, fetch sits at the cap and the network idles. Network-bound: the stage should hold the longest gap between GPU jobs: start-up (15 min) + a `short-gpu`/scavenger wait of up to 1 h
  before the launcher falls back, at 8.5 M pairs/s × 29.4 B = 0.25 GB/s: **0.9 TB for 1 hour, 2 TB for 2.2 hours.** Default cap 2 TB ("a few hours"); X = 1 GPU-hour holds 0.2 TB at 2 M/s, 1.06 TB at 10 M/s.
* **GPU duty.** Workers hold their GPU for as long as the stage has work, so duty ≈ 100% while compute-bound. In the network-bound regime duty = supply ÷ capacity (8.5 ÷ 12 for 6 GPUs at 2 M/s = 71%), carried by bursts of ≥ 1 h
  claimed from the stage; the publish thread keeps the GPU from waiting on uploads.
* **Spare network.** When compute-bound, lowering `fetch_jobs` or the cap costs nothing; raising the cap to 0.9 only helps once R is ≥ 2.1 per GPU (6 GPUs).

## 8. Streaming download → GPU, skipping the disk?

No. The network (8.5 M pairs/s at the cap, 12.8 at 0.9) is **faster than any GPU worker measured so far** (0.1–0.2 M pairs/s each, up to ~2 host-bound), so a streaming GPU would not wait for data today; and in the
regime where it would (R ≥ 1.4 per GPU) the stage is what makes it robust: (1) the link is shared and varies by 20–30% (§1d), a streamed GPU idles on every dip; (2) the decode costs 3.0 core-s/M, i.e. 26 cores at the cap, which
on a GPU node would compete with the host lane (12.5–14.4 core-s/M) that already limits it; (3) 30% of runs are rejected only after the head screen/decode, multi-run samples need every run before compute starts; (4) a preempted
GPU job loses a stream but not a staged sample. The stage costs ≤ 2 TB of `$WORK` and 0.25 GB/s of writes.

## 9. Decisions taken (Zach, 2026-10-30) and what is open

* Full-quality `.sra` (S3 primary), binned qualities in the stage ✔ (implemented, measured). No CPU-node split: the drain worker does staged input → uploaded `.s1` ✔ (the producer/upload runs in the worker's publish thread). GPU drain workers preemption-safe ✔;
  **but scavenger has no H100** (§0): use `short-gpu` for H100 or ask GVSU to add g050–g052 to `scavenger`.
* Rate cap 0.6 GB/s global ✔. 0.9 needs Zach's clearance with GVSU IT: one value.
* *Open:* S6 to confirm the manifest-merge assumption (§5) and run one live publish with a token; `singlet-gpu` ran on a V100S (g001), but its speed on large samples is unmeasured (`v100_rel` = 0.3 is a guess); the GPU engine's
  per-sample start-up (~100 s) argues for a resident-index session mode (S8); the fetch CPU (26 cores) should not be stolen by other lab jobs on `short-cpu`.

## 10. Status of what was queued

* Time-of-day repeats Sat 03:00 / 13:00 (jobs 793570, 793571): results append to `$WORK/jobs/s9/results/tod.jsonl`; this section is updated in a follow-up commit when they land.

## 11. Staged format `.1fg2` and `singlet-stage` (S9 producer, S8 readers; layout in `src/stage/fmt2.h`, spec `staged-format-1fg2.md` on #107)

The fetch tier can write a GPU-native staged file instead of `.1fq`: chunks of 32,768 reads as zstd planes (whitelist-corrected barcode index + status byte, UMI word, sequence id, length, 2-bit binned quality per base, UMI-N exceptions) and one per-sample table of
distinct cDNA sequences (2-bit, N lists), no barcode sort, run table in the footer. `singlet-stage` (CMake target, `src/stage/singlet_stage.cpp`) reads the `.1fq` of a sample's kept runs with the engine's `Reader`, takes protocol, geometry and whitelist from `known_protocols()`,
builds the exact-barcode histogram (exact whitelist hit, UMI without N and not a homopolymer, as `singlet.cpp` pass 1), corrects with `CbCorrector::resolve`, keys the sequence table on the cDNA exactly as the `.1fq` holds it (the engine clips later on the packed codes).
`--verify` re-reads the written file and compares every read with the source.

**Measured (job 793809: 45 runs from the earlier fetch-scale list; 37 staged, 3.91 G reads; 7 stopped at the non-10x gate, 1 protocol with no universal whitelist in R1; `--verify` on 5: 0 mismatches; plus 32.8 M reads verified on 3 more runs, job 793776):**

| | value |
|---|---|
| raw full `.sra` / `.1fq` (binned qualities) / `.1fg2` | 75.1 / 30.1 / **28.1 B per read** (`.1fg2` = 0.93 × `.1fq`); input `.1fq` is barcode-sorted, an unsorted source adds ~2–3 B/read to the `cb`/`umi` planes |
| where the bytes go (B/read) | q 9.5, sequence table 12.2, umi 3.0, sequence id 2.9, cb 0.35, rest < 0.1 |
| CPU | decode `.sra` → `.1fq` **3.20 core-s/M** (unchanged); `singlet-stage` **2.18 core-s/M** (pass 1 1.59: `.1fq` decode + histogram + hash; pass 2 0.59: correct + zstd 3 + CRC) → **5.4 core-s/M in total, +68% over today's 3.2** |
| unique cDNA fraction (distinct / reads) | 0.44 at 20–50 M reads (n = 3), 0.49 at 50–100 M (n = 18), **0.31 at ≥ 100 M reads (n = 16, up to 234 M)**; S8 had expected ~0.54 on deep samples |
| memory | table up to 5.4 GB at 234 M reads (64-byte in-memory slots; 32 GB fetch jobs are enough) |
| limits found | cDNA up to 151 nt occurs: the limit was raised from 127 to 255 (u8 length); GPU pileup with per-base quality handles ≤ 128 nt for now (S8 falls back to the CPU pileup above) |

**Reading it.** The format moves barcode correction, sequence dedup and the chemistry decisions to the CPU nodes and keeps staged bytes at ≈ `.1fq`; it costs 2.2 core-s/M more in the fetch tier (≈ 70 cores more at 8.5 M pairs/s: 26 → ~46 cores at the 0.6 GB/s cap) and removes 9 of the
14 host core-s/M that S8 measured on the GPU node. Part of the added cost is avoidable: a combined decode-and-stage pass would not write and re-read the `.1fq` (the writer was 35 of 53 s in the W11 ingest measurement, the `.1fq` read is most of pass 1); not built or measured.
The sequence ids and the sequence table together are 15 B/read; distinct sequences are only 31% of reads on deep samples, but each costs 2-bit + ids. Quality (9.5 B/read) is the largest single plane.

## 12. Combined pass `.sra` → `.1fg2` (`singlet download --stage-out`) and the raw-barcode plane

`singlet download <sra> --stage-out OUT.1fg2` feeds the SRA writer's flushed blocks (after its polyA trim, `--r2-maxlen`, in SRA order, no barcode sort) to the same `stage::Encoder` as `singlet-stage`: no `.1fq` is written or re-read. The default `.1fq` path is untouched (the block sink is off unless the option is given;
`Writer`, `SraEncoder` and `EncoderConfig` gained an optional sink). Single-run samples with a 10x R1 protocol only; other protocols exit 6 and a multi-run sample needs one histogram over all runs, so both stay on the `.1fq` route (the fetch worker's `--format 1fg2` falls back automatically).
**`PL_RAWCB`** (new plane, S8 agreed): one 8-byte entry (raw barcode word 2 bit/base, read index, N position) for every read that was not an exact whitelist hit, so STAR's exact/corrected/no-match counts and the cell QC stay identical; 3.4–7.9% of reads, 0.2–0.5 B/read.

**Measured, same local `.sra` per run (A = today's `singlet download` to the barcode-sorted `.1fq`; B = A + `singlet-stage`; C = combined), core-s per M reads, 8 runs:**

| run (reads) | A baseline | B two-step | C combined | C − A |
|---|---|---|---|---|
| SRR19421372 (9 M) | 2.82 | – | 3.46 | +0.64 |
| SRR19573735 (11 M) | 3.19 | – | 3.88 | +0.69 |
| SRR30477995 (12 M) | 3.37 | – | 4.40 | +1.03 |
| SRR19544612 (75 M) | 2.87 | 4.93 | 3.73 | +0.86 |
| SRR12455246 (112 M) | 2.88 | 5.24 | 3.72 | +0.84 |
| SRR19421396 (132 M) | 2.95 | 5.02 | 3.53 | +0.58 |
| SRR19649496 (186 M) | 2.68 | 6.61 | 3.59 | +0.91 |
| SRR19572361 (234 M) | 2.50 | 5.69 | 3.14 | +0.64 |

(Baselines vary 2.5–3.4 with node load; B is not shown for the three small runs because their timing included `--verify`; it was 4.3–5.4 in the earlier plain runs.) **The combined pass costs +0.6 to +1.0 core-s/M over today (about +25%), against +1.7 to +4 for the two-step path.** It removes the `.1fq` sort, compress, write and re-read.
In all 8 runs the order-independent digest of C equals that of B (same barcode, status, raw barcode, UMI, cDNA, qualities per read); B passes `--verify` against the `.1fq` read by read.

**Bytes per read, sorted vs unsorted source (the "no barcode sort" question):** `.1fq` / B (`.1fg2` from the barcode-sorted `.1fq`) / C (SRA order):
75 M: 33.1 / 31.5 / 34.1; 112 M: 32.0 / 28.6 / 31.1; 132 M: 26.3 / 20.4 / 23.1; 186 M: 25.9 / 19.6 / 22.7; 234 M: 23.0 / 18.6 / 21.5. Skipping the sort costs **+8 to +16%** in the `.1fg2` (the barcode and UMI planes lose their runs), but even unsorted it is 3% above to 12% below the `.1fq` (deep samples gain from the distinct-sequence table).
Distinct-sequence fraction on these deep runs: 0.56 (75 M), 0.46 (112 M), 0.31 (132 M), 0.28 (186 M), 0.24 (234 M). Encoder table memory 3.1–4.2 GB at 75–234 M reads (64-byte in-memory slots; halving it for ≤127 nt files is possible but not done).

**Open:** S8's `.1fg2` reader and the per-read identity gate (fixtures in `$WORK/jobs/s9/fixtures`); a multi-run part merge; the fetch worker's `--format 1fg2` path is written but not run against a real binary here (the drain only claims `.1fg2` samples with `--accept-1fg2`).
