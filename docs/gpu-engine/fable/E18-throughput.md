# E18: samples in flight per GPU (throughput)

Experiment E18 ("throughput") of the GPU aligner program. Branch `gpu/fable/throughput`, from `gpu/fable-engine` at 44a6e9fa (E1–E15),
with `gpu/fable-engine` merged in at e0e1514a (E16, "tail") as c6ddd3d3. The question: how many read pairs per second does one node
(2 H100 NVL, 54 usable cores, 750 GB) deliver when several samples share each GPU through the engine's device slots
(`SINGLET_ENGINE_GPU_SLOTS`), what limits it, and which configuration the campaign should run.

Everything runs level 5 + ingest (`--engine gpu`, `SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1`), the
configuration E19 is making the default. E18 changed no engine code: new bench scripts under `bench/fable/throughput/` only.

## 0. Answer in brief

- **The curve (pbmc1k_v3, one H100 NVL, 28 cores, `SLOTS=1`, M read pairs/s):**

  | samples in flight k | 1 | 2 | 3 | 4 | 6 | 8 | limit |
  |---|---|---|---|---|---|---|---|
  | 44a6e9fa (E15) | 0.44–0.49 | 0.80 | 1.00 | 1.09–1.12 batch, **1.16** stream | **1.27** stream | – | 1.62 |
  | merged head (E16) | 0.52 (T = 7), 0.57 (T = 9) | – | **1.10** | – | **1.40** stream (1.33 over the span) | @@ | 1.84 |

  **The node (2 GPUs, 54 cores): @@ M pairs/s** at @@, against 1.38–1.44 M pairs/s for the CPU engine on the same node (54–56
  cores / 39 core-s per M).
- **The bottleneck is the device slot, not the cores, the host memory bandwidth or the device memory (§5a).** With `SLOTS=1`
  each sample holds the GPU 41.1 s on 44a6e9fa and 36.1 s on the merged head (T = 7), against 11.6 s of kernels. The limit per GPU
  is 66.6 M pairs / held time: 1.62 and 1.84 M pairs/s. The cores are mostly idle (8.0 core-s per M pairs per run, flat in k:
  ~10 of 28 cores at k = 4).
- **Per phase, at k = 1 (§4):** of the 36.1 s held on the merged head, ~7.9 s is host work with the device idle that could run
  outside the slot without changing any device work:
  - the prefill's host lane wait, record assembly and `readRecord` between lane B and `e2e.build` (5.7 s);
  - since E16's `resetbg`, the pileup's slot is held 2.2 s past `pileupRun` through the accumulator puts and the host pass (the log
    reports 4.1 s held; the sampler sees 6.3 s).
  E16 already took 5 s off the hold (pileup digest and uploads 9.0 → 6.3 s, zero-copy hand-off pass 3.9 → 2.5 s). With the slot held
  only around device work the limit is ~2.4 M pairs/s per GPU; with the index, the non-host index and a context resident (a
  persistent per-GPU process) ~3.0.
- **Below the limit the samples form a convoy (§5b).** At k = 4 the slot is idle 31 % of the steady window although every run
  waits 70–100 s for it: the samples leave their device phases together, run their host-only tails (53 s) and starts (31 s)
  together, and nobody needs the device meanwhile. A FIFO queue model built from the k = 1 profile alone reproduces every
  measured point within 2–5 % (the measured waits per phase match the FIFO model, not the random-order one) and says a FIFO slot needs k ≈ 10–12 per GPU to come within 6 %
  of its limit. **Serving later phases first would reach the limit at k = 5** (model): a change to `lockDevice` worth as much as
  E16.
- **`SLOTS=2` with the `m43` memory split gains 0–8 % (§5c):** two prefills time-slice the device (36–43 s each), and the split
  itself slows a prefill alone from 18.2 to 27.4 s.
- **What bounds k is host memory (§5d):** ~35 GB per sample in flight on 44a6e9fa and ~43 GB on the merged head, the largest piece
  being each process's private 16.8 GB copy of the genome. Device memory does not bound k with `SLOTS=1` (a waiting sample keeps ≤ 0.7 GB of context).
- **Identity holds at every k (§3):** @@ runs compared with their sample's k = 1 run: every count matrix, Solo file and
  `CellReads.stats` identical; only E11's run-dependent files differ. The mapDistinct dumps of all @@ points are identical
  (sha256 77d7bc20008166af).
- **Recommendation (§6):** @@
- **For E17 (§7):** no budget is needed at `SLOTS=1`; for 2 slots, A100 and big genomes the budget must cover a 26.4 GB base before
  the lanes, lane B and the arena that today take free − 3 GB, a fixed 24.9 GB non-host index, and ~0.7 GB per waiting sample.

## 1. Setup

- **Binary:** `singlet-gpu` built by `bench/fable/throughput/jobs/build.sbatch` (job 797401) at 7a2bbb9e, whose engine code is
  gpu/fable-engine 44a6e9fa (E1–E15 merged); sha256 51e0b035…. No engine code changed in E18.
- **Mode:** level 5 + ingest, the command line and environment of E15's gate mode `gpuE5i`:
  `singlet-gpu <s>.1fq --ref-base … --genome-dir …/star_2.7.11b --exons …/genes.gtf --out-prefix <dir>/ --threads T --pipeline
  --output-format 1pz --engine gpu` with `SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1`.
- **k samples in flight** are k independent processes (no fork from a shared parent), each with its own private copy of the genome
  (`Allocated a private copy of the genome`), its own output directory and `--threads T`. They share the device through the
  engine's slot locks (`lockDevice` / `lockDeviceSlot` in `include/singlet/engine_gpu.h`, exclusive `flock`s on
  `/tmp/singlet_engine_gpu_<pci>[.k].lock`). The job's `/tmp` is private to the job (ext4 on /dev/sda1), so other jobs on the node do
  not share the lock files.
- **`run.sbatch` tags** `[mix]k<K>t<T>s<S>[m43][r<R>][p][f]`: K workers take items from a queue; R rounds give K × R items
  ("stream" form: a worker starts its next sample as soon as its previous one ends); `s` sets `SINGLET_ENGINE_GPU_SLOTS`; `m43` the
  two-slot memory split (§7); `p` pins worker w to device w mod GPUs; `f` samples at 0.2 s.
- **Inputs** copied to the job's `/tmp` before each point; outputs and scratch in `/tmp`; page cache warm after the first point
  (the index and genome files are read by every sample).
- **Measurements per point:**
  - throughput = pairs (sum of `Number of input reads` in `Log.final.out`) / span (first start to last end);
  - host CPU from the job's cgroup (`cpu.stat` usage, includes the sampler) and per run from `/usr/bin/time` (user + sys);
  - host memory from the cgroup (`memory.current`, `memory.stat` anon/file/shmem) and per run RSS/PSS;
  - device memory and utilisation from `nvidia-smi` (device-wide and per process);
  - lock waits and holds from the engine's own log lines (`waited … s for it, held it … s` and the pileup's, Solo's and non-host's
    equivalents) and, independently, from `/proc/locks` (the holder and waiters of each lock file at every tick).
- **Nodes:** g050 (jobs A, B), g051 (job C) and @@ (job D), Xeon Gold 6238R (2 × 28 cores, 2 NUMA nodes), 750 GB, 2 H100 NVL 94 GB.
  Slurm gives 54 usable cores per node (`CPUEfctv`). Jobs A, B and C use one GPU and 28 cores (half a node; other jobs may hold the
  other half), job D the whole node.

## 2. The curve, per point

pbmc1k_v3 (66.6 M pairs per sample), level 5 + ingest, one H100 NVL and 28 cores unless stated. "batch": the k samples start
together and the point ends with the last; "stream" (`r`): k workers each run R samples back to back, and the steady rate is k ×
66.6 M / the mean wall of the runs started after the first completion. Lock columns are means per run from the engine's log lines;
"held %" is the share of the span (steady window for streams) in which the sampler saw the slot held. Host and device memory
are in GiB (the cgroup's `memory.current` peak, `nvidia-smi`).

**Engine 44a6e9fa (E1–E15), jobs 797423 (A) and 797443 (B), node g050:**

| point | form | k | T | slots | runs | **pairs/s (M)** span / steady | wall s min / mean / max | core-s/M runs / cgroup | lock wait / held s | held % | host peak GB | device peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k1t28s1 | – | 1 | 28 | 1 | 1 | **0.462** | 144.1 | 9.19 / 9.70 | 0 / 36.9 | 25 | 54 | 90.1 |
| k1t14s1 | – | 1 | 14 | 1 | 1 | **0.486** | 137.0 | 8.43 / 8.87 | 0 / 36.7 | 26 | 54 | 90.3 |
| k1t9s1 | – | 1 | 9 | 1 | 1 | **0.474** | 140.5 | 7.98 / 8.46 | 0 / 37.7 | 26 | 55 | 90.3 |
| k1t7s1 | – | 1 | 7 | 1 | 1 | **0.447** | 149.1 | 7.71 / 8.20 | 0 / 40.1 | 27 | 56 | 90.3 |
| k1t7s1f (B) | – | 1 | 7 | 1 | 1 | **0.435** | 153.0 | 7.71 / 9.00 | 0 / 40.4 | 26 | 52 | 90.3 |
| k2t14s1 | batch | 2 | 14 | 1 | 2 | **0.802** | 143.5 / 154.8 / 166.2 | 8.47 / 8.85 | 13.1 / 36.6 | 44 | 91 | 90.3 |
| k3t9s1 | batch | 3 | 9 | 1 | 3 | **0.995** | 179.4 / 187.7 / 200.7 | 8.03 / 8.37 | 42.1 / 37.1 | 57 | 112 | 90.3 |
| k4t7s1 | batch | 4 | 7 | 1 | 4 | **1.093** | 222.7 / 235.9 / 243.8 | 7.83 / 8.18 | 82.0 / 39.8 | 67 | 148 | 90.1 |
| k4t14s1 | batch | 4 | 14 | 1 | 4 | **1.120** | 214.6 / 226.4 / 238.0 | 8.43 / 8.77 | 72.2 / 36.7 | 65 | 152 | 90.3 |
| k4t7s1r3 (B) | stream | 4 | 7 | 1 | 12 | **1.108 / 1.159** | 225.6 / 236.0 / 260.9 | 7.85 / 8.21 | 79.4 / 41.2 | 69 | 153 | 90.3 |
| k6t7s1r2 (B) | stream | 6 | 7 | 1 | 12 | **1.212 / 1.273** | 303.8 / 323.1 / 355.7 | 7.97 / 8.35 | 154.8 / 41.3 | 78 | 210 | 90.3 |
| k1t14s1m43 | – | 1 | 14 | 1 | 1 | **0.455** | 146.3 | 8.54 / 9.03 | 0 / 45.3 | 31 | 59 | 43.6 |
| k2t14s2m43 | batch | 2 | 14 | 2 | 2 | **0.807** | 163.0 / 164.0 / 165.0 | 8.82 / 9.20 | 0 / 60.0 | 42 | 102 | 88.4 |
| k3t9s2m43 | batch | 3 | 9 | 2 | 3 | **0.993** | 163.8 / 181.7 / 201.2 | 8.30 / 8.64 | 18.6 / 53.6 | 56 | 131 | 84.7 |
| k4t7s2m43 | batch | 4 | 7 | 2 | 4 | **1.178** | 165.8 / 203.2 / 226.1 | 8.05 / 8.38 | 37.9 / 50.2 | 63 | 154 | 87.6 |

**Merged head c6ddd3d3 (E15 + E16), job 797460 (C), node g051:**

| point | form | k | T | slots | runs | **pairs/s (M)** span / steady | wall s min / mean / max | core-s/M runs / cgroup | lock wait / held s | held % | host peak GB | device peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k1t7s1f | – | 1 | 7 | 1 | 1 | **0.522** | 127.7 | 7.70 / 8.80 | 0.4 / 36.1 | 28 | 52 | 90.3 |
| k1t9s1 | – | 1 | 9 | 1 | 1 | **0.573** | 116.3 | 7.89 / 8.29 | 1.4 / 34.0 | 29 | 54 | 90.3 |
| k3t9s1 | batch | 3 | 9 | 1 | 3 | **1.104** | 172.9 / 177.6 / 181.0 | 7.98 / 8.33 | 80.4 / 34.1 | 58 | 141 | 90.3 |
| k6t7s1r2 | stream | 6 | 7 | 1 | 12 | **1.329 / 1.398** | 282.5 / 294.5 / 315.1 | 7.77 / 8.16 | 214.2 / 34.7 | 69 | 274 | 90.4 |
@@C-REST@@

- **The k = 3 re-run on the merged head: 1.104 M pairs/s against 0.995 on 44a6e9fa (+11 %)**, both batch, T = 9, one GPU, 28 cores
  (different nodes, g051 and g050; the k = 1 runs differ by the same order: 0.573 against 0.474 at T = 9). The slot is held 34.1 s
  per run instead of 37.1 s.
- At k = 1 a run on the merged head waits 0.4–1.4 s for its own Solo: E16 runs Solo (STAR's child) beside the pipeline's pileup,
  and they take the same slot.
- Host memory per sample in flight is higher on the merged head (k = 6: 274 GB against 210 GB; anon + shmem at most 255 GB, 43 GB
  per sample): E16 keeps the prefill's data until after STAR's loop and page-locks the hand-off region.

## 3. Identity

Every run of a point with k > 1 is compared with the k = 1 run of the same sample at the same T in the same job (the reference; job C
has its own references). The check (`ident` in `run.sbatch`):
- `bench/star_accel/out_equiv.py` over the whole output directory: identical / differ only in timings or paths / differ;
- `star_Log.final.out`'s statistics (every line after the timing header) and `star_SJ.out.tab`: byte-identical;
- `nonhost_reads.fq`, `star_Unmapped.out.mate1` / `mate2`: byte-identical, else the same records sorted (their record order is
  run-dependent);
- `Gene/CellReads.stats` (copied before the pipeline deletes it): byte-identical, else the same rows sorted;
- item 0 of each point writes mapDistinct's records (`SINGLET_ENGINE_GPU_DUMP`, E6's dump); their sha256 per sample are compared
  across all points of the job.

The run-dependent files, as in E11's gate:
- `pileup_stats.json`, `read_stats.tsv` and `star_Log.std.out` differ (stage timings and wall clocks);
- 2–3 files differ only in timings or paths (the output prefix);
- with `m43`, `summary.json` also differs: its `memory_tier` comes from the process's VmPeak (Small < 8, Medium < 64, Large < 256 GB),
  and the arena cap moves pbmc1k_v3 from Large to Medium.

**Results:** every run of every point is identical to its sample's k = 1 run except for the run-dependent files.

| job (engine) | runs compared | out_equiv: identical / timings-paths / differ | the differing files | `Log.final.out` stats, `SJ.out.tab`, `CellReads.stats` | E6 dumps |
|---|---|---|---|---|---|
| 797423 A (44a6e9fa), SLOTS=1 | 13 | 63 / 3 / 3 | `pileup_stats.json` (`bytes_public`, timings), `read_stats.tsv`, `star_Log.std.out` | identical; the 3 read files byte-identical | all identical, sha256 77d7bc20008166af (E12's) |
| 797423 A, `m43` (SLOTS=1 and 2) | 10 | 63 / 2 / 4 | the same 3 + `summary.json`'s `memory_tier` | identical | identical |
| 797443 B (44a6e9fa) | 24 | 63 / 3 / 3 | as above | identical | identical |
| 797460 C (merged, E16) | @@ | 60 / 3 / 6 | the same 3 + `nonhost_reads.fq`, `star_Unmapped.out.mate1/2` (same records, another order) | identical; the 3 read files hold the same sorted records | identical |
| @@ D (merged, 2 GPUs) | @@ | @@ | @@ | @@ | @@ |

On 44a6e9fa even the three read-order files came out byte-identical at every k; with E16 they are written by a background thread
and their record order varies from run to run (E11's known run-dependent set; E16's own gate lists the same three files).

## 4. What the slot holds, per phase (k = 1, both heads)

A sample takes a device slot five times. Points `k1t7s1f` (pbmc1k_v3 alone, T = 7) ran with the sampler at 0.2 s ticks and
`nvidia-smi` every tick, on 44a6e9fa (job 797443, g050) and on the merged head with E16 (c6ddd3d3, job 797460, g051). The engine's
own timers (`engine_gpu: steps (s)`, the hand-off pass's line, `soloRunKept`, `[gpu_pileup]`, `[nonhost]`, E16's `[e16]` marks)
split each hold; the sampler confirms it (the slot's holder from `/proc/locks`, device utilisation and memory, the run's CPU cores
per tick; `e18_trace.py`).

**44a6e9fa (E15): 41.1 s held per sample, 11.6 s of kernels.**

| phase (code) | held | kernels | transfers, allocation, context | host work with the device idle | could leave the slot as it is |
|---|---|---|---|---|---|
| prefill (`mapDistinct`, `SeqCacheGpuPrefill.cpp`) | **23.1 s** | 8.5 s (lanes A/M/B 2.84 + 2.73 + 2.64, clip/E10/dedup 0.32) | 7.6 s: uploads 1.39 (reads 0.92), the async index upload's wait 0.99, clip/dedup/lane downloads 1.60, `e2e.build` 2.56 (the hand-off records built on the device and staged to the host at 2.7 GB/s), init/alloc/free 1.04 | **6.8 s**: host lane wait 1.79, `records.prep` + `records.copy` 3.07, `readRecord` 0.63, `lanes.host` 0.64, `clip.host` 0.53, `dedup.order` 0.11 | **5.5 s**: the host lane's wait, the records and `readRecord` run between lane B's download and `e2e.build`; release the slot after the lanes and take it again for `e2e.build` |
| hand-off pass (`e2ePass`, `records_gpu.cu`) | **3.9 s** | 0.28 s | 3.6 s: upload 1.36 (pinned, ~6.7 GB, with the page-locking), download 1.86, context create/reset ~0.4 | ~0.2 s | ~0 (transfer-bound: the data were on the device in the prefill) |
| Solo (`soloRunKept`, E11) | **0.7 s** | 0.04 s | 0.53 s context and allocation, 0.10 transfers | – | ~0 |
| pileup (`pileupRun`, `pileup_gpu.cu`) | **9.0 s** | 2.16 s (CUDA events) | ~2.3 s: context, the arena (free − 3 GB = 92 GB), the annotation conversion and pageable uploads (1.1 s before the first kernel), `cudaFree` + `cudaDeviceReset` (0.4 s) | **~4.6 s** on one core: 1.8 s after the kernels (the host fetch of the partition's results, 1.11 s by its timer), then 2.8 s with the arena allocated and idle: the SHA-256 digest of every output, the output copies, the R9 merge sort | **~3.1 s**: the annotation conversion before; the digest, copies and sort after `cudaFree` |
| non-host (`nonhost_gpu.cu`) | **4.4 s** | ~0.6 s (classify 0.4, the index directory) | 3.6 s: the 24.9 GB index read from the page cache by T threads through pinned bounce buffers (~6.9 GB/s) | – | ~0 as it is; ~2 s with the index in page-locked memory before the slot (12.3 GB/s) |
| **total** | **41.1 s** | **11.6 s** | **~17.5 s** | **~11.6 s** | **~8.6 s** |

**c6ddd3d3 (E15 + E16): 36.1 s held per sample (sampler), the same 11.6 s of kernels.** E16 changed the order (the hand-off pass
runs early, during STAR's loop, before Solo) and two holds:

| phase | held | what changed with E16 | host work with the device idle under the slot | could leave the slot |
|---|---|---|---|---|
| prefill | **22.2 s** | nothing material (steps within node noise: kernels 8.6, host lane wait 1.47, records 3.51, readRecord 0.77, lanes.host 0.48, clip.host 0.22, dedup.order 0.11) | 6.6 s | **5.7 s** (as above) |
| hand-off pass (early) | **2.5 s** | zero-copy: upload 0.85 s from the page-locked region (0.46 s of it page-locking), download 1.13 s, kernels 0.30, context 0.19 | – | ~0 |
| Solo | **0.6 s** | – | – | ~0 |
| pileup | **6.3 s** (the log says 4.1) | `pileupRun` 9.0 → 4.1 s: no digest, smaller per-alignment download, uploads from page-locked memory in place (GPU 2.17 s, fetch 0.86 s) | **2.2 s**: with E16's `resetbg` the slot stays held after `pileupRun` returns, through the accumulator puts and the host pass (15–22 cores busy, device idle at 0.5 GB), until the free thread has reset the context (0.42 s) | **2.2 s**: reset and unlock on the free thread right after `pileupRun`, before the puts and the host pass |
| non-host | **4.5 s** | – | – | ~0 as it is; ~2 s with a page-locked index |
| **total** | **36.1 s** | −5.0 s against E15 | ~8.8 s | **~7.9 s** |

The pileup's `device lock: … held 4.13 s` line is printed when `pileupRun` returns; the slot itself is released 2.2 s later by
the free thread (`parent (bg): device reset in 0.423 s, device unlocked`). Only the sampler sees the difference.

With `SINGLET_ENGINE_GPU_SLOTS=1` the slot's limit is 66.6 M pairs per held time:

| what is held per sample (pbmc1k_v3) | held | limit per GPU |
|---|---|---|
| 44a6e9fa, T = 7 | 41.1 s | **1.62 M pairs/s** |
| 44a6e9fa, T = 14 (host lane and records faster with more threads) | 36.7 s | 1.81 |
| merged head (E16), T = 7 | 36.1 s | **1.84** |
| merged head, slot only around device work (prefill −5.7 s, pileup −2.2 s) | ~28.2 s | ~2.36 |
| and a persistent per-GPU process: index, non-host index and a context resident (−1.3 s index upload, −3.6 s non-host index, −~1.5 s context create/reset over the five phases) | ~22 s | ~3.0 |
| kernels plus the transfers no design avoids (reads up, records and results down) | ~16 s | ~4.2 |

The coordinator's ~1.9 M pairs/s assumed 35 s per sample: on the merged head the measured hold is 36.1 s at T = 7.

## 5. Why the curve bends where it does

### 5a. The device slot is the bottleneck, not the cores, the host memory bandwidth or the device memory

- **Cores:** a sample alone takes 137–149 s at T = 7, 9, 14 or 28 (job A): its wall hardly depends on its threads. At k = 4 the
  cgroup uses ~10 of 28 cores on average (8.2 core-s per M pairs × 1.16 M pairs/s ≈ 9.5 cores). The runs' own core-s per M pairs
  stay at 7.7–7.9 at T = 7 from k = 1 to k = 6 (no contention cost).
- **Host memory bandwidth:** the host-only stages run as fast with 4 samples in flight as alone (k4t7s1 against k1t7s1: export 17–18
  s against 17 s, demux 23–24 s against 23 s, the prefill's uploads 5.1–5.6 GB/s for the index and 7.4–12.1 GB/s for the reads
  against 5.4 and 12.1); §5d has where it does show.
- **Device memory:** with `SLOTS=1` one process holds the device at a time; the device peak is the k = 1 peak (90.1–90.4 GB) at every
  k (§7).
- **The slot:** with `SLOTS=1` every device phase of every sample is serialised, and a sample holds the slot 37–41 s (§4). The limit
  is therefore 66.6 M / 41 s = 1.62 M pairs/s per GPU at T = 7 (1.81 at T = 14).

### 5b. Below the limit: a convoy, not a queue that never empties

The slot is far from busy at k = 3–4 although every run waits for it: in the steady window of `k4t7s1r3` it is held **61 %** of
the time, and the sampler never sees it free with a waiter (408 of 1,175 ticks free, all without waiters; 688 ticks held, 560 of
them with 2–3 waiters). The samples move in a convoy:
- a sample needs the slot five times, in 41 s of a 153 s life, but the first visit comes after 31 s of host work (decode, scan,
  references, STAR's start) and the last ends 53 s before the sample does (export, demux, writers);
- when the slot frees, the waiters queue behind each other's long prefills (23 s) — Solo (0.7 s of device work) waited 52–56 s in
  three of the twelve runs — so the samples leave their device phases close together, end close together, and start the next
  ones together: then for ~30 s at the start and ~50 s at the end of each round nobody needs the device.

A closed-queue model reproduces the curve from the k = 1 profile alone (`e18_sim.py`: k workers, each a sequence of host
stretches and slot holds taken from `k1t7s1f`, one slot per GPU). With waiters served first come, first served it matches the
measurements within 2–5 % in rate and within a few seconds in the mean wait of each phase; the measured waits per phase
match first come, first served, not a random choice among the waiters.

44a6e9fa (pairs/s in M; mean waits per run in s, prefill / Solo / pileup / non-host):

| k | model FIFO | model random order | **measured** | model FIFO waits | measured waits |
|---|---|---|---|---|---|
| 1 | 0.435 | 0.435 | 0.435–0.447 (T = 7) | 0 | 0 |
| 2 | 0.816 | 0.816 | 0.802 (batch, T = 14) | 2.1 / 7.1 / 0 / 1.4 | 8.5 / 0.9 / 0 / 0 (batch) |
| 3 | 1.033 | 1.054 | 0.995 (batch, T = 9) | 15.0 / 15.1 / 2.4 / 4.7 | 17.3 / 14.5 / 2.7 / 4.7 (batch) |
| 4 | 1.177 | 1.227 | **1.159** (steady) | 25.5 / 25.3 / 6.3 / 10.2 | 27.7 / 25.1 / 11.1 / 13.6 |
| 6 | 1.335 | 1.513 | **1.273** (steady) | 45.6 / 48.5 / 18.6 / 22.2 | 49.8 / 45.5 / 22.3 / 25.7 |
| 8 | 1.430 | 1.625 | | | |
| 12 | 1.538 | 1.637 | | | |
| ∞ | 1.628 = 66.6 M / 40.9 s | | | | |

Merged head (E16):

| k | model FIFO | model random order | **measured** |
|---|---|---|---|
| 1 | 0.521 | 0.521 | 0.522 (T = 7), 0.573 (T = 9) |
| 3 | 1.133 | 1.149 | **1.104** (batch, T = 9) |
| 6 | 1.465 | 1.606 | **1.398** (steady; 1.329 over the span) |
| 8 | 1.582 | 1.782 | @@ |
| 12 | 1.843 | 1.851 | |
| ∞ | 1.843 = 66.6 M / 36.1 s | | |

Three consequences:
- **k must be several times the naive bound** (life / held = 153 s / 41 s ≈ 3.7) to keep a FIFO slot busy: the model needs
  k ≈ 10–12 per GPU to come within 6 % of the limit on either head.
- **The order in which waiters get the slot is worth as much as E16.** Giving the slot to the sample furthest along (later
  phases before new prefills) reaches the limit at k = 5 on the merged head (1.84 M pairs/s; FIFO 1.38 at k = 5) and at k = 6 on
  44a6e9fa (1.66). New samples then wait at their prefill instead of delaying the short phases of samples about to finish, and
  the in-flight samples (host memory) are halved for the same rate. That is a change to `lockDevice` (for example a second lock
  file that a prefill also takes and that any later phase takes while it waits), outside E18's scope.
- **Batch starts cost 5–10 %** against the stream form at the same k (k = 4: 1.093–1.120 batch, 1.159 stream): the ramp (31 s
  before the first prefill) and the tail (53 s after the last non-host) are paid once per batch.

### 5c. Two slots per GPU (`SLOTS=2`, the `m43` split) do not pay

| k | SLOTS=1 (default memory) | SLOTS=2 + `m43` | prefill held per run, SLOTS=2 |
|---|---|---|---|
| 1 | 0.486 (T = 14) | 0.455 (SLOTS=1 + `m43`) | 27.4 s alone (18.2 s without `m43`) |
| 2 | 0.802 | 0.807 | 40.8 / 42.7 s (both at once) |
| 3 | 0.995 | 0.993 | 40.5 / 40.0 / 24.8 s |
| 4 | 1.093 (T = 7), 1.120 (T = 14) | 1.178 | 36.7 / 27.8 / 36.2 / 26.3 s |

- Without MPS, two processes' kernels on one device time-slice: two prefills at once take 36–43 s each instead of 18–25 s, so the
  device does the same work in the same time. What a second slot can overlap is one sample's host work under its slot with
  another's kernels, which is why k = 4 gains 5–8 %.
- The memory split costs on its own: alone, the prefill holds 27.4 s instead of 18.2 s, because lane B's Work at 21,504 reads per
  batch (instead of free memory) runs 7.5 s instead of 3.0 s, and the 4 M-read chunk adds rounds.
- A second slot also needs every phase inside half the device (§7), which only the `m43` hand split achieves today.
- Fixing the holds (§4) is worth more than a second slot, and keeps the device memory free for one sample's large lanes.

### 5d. What bounds k: host memory

- **Device memory does not bound k with `SLOTS=1`.** The holder's phases end with `cudaDeviceReset`; a waiting sample keeps at most a
  CUDA context, ~0.7 GB (the STAR child's while E15's page-locked Solo inputs live through STAR's loop; on the merged head also the
  early hand-off pass's until the pileup). The sampler's per-process device memory of the non-holders: median 0–0.7 GB in total,
  95th percentile 3.5 GB, maximum 5.0 GB at k = 6. Lane B and the pileup arena take free memory minus 3 GB, so they shrink by that
  much and nothing fails; the device peak stays 90.0–90.4 GB at every k.
- **Host memory does.** Per sample in flight: ~35 GB at the cgroup's peak (k = 4: 153 GB; k = 6: 210 GB, of which anon 188 and
  shmem 39), a run's PSS peak 40–45 GB. The largest single piece is the **private copy of the genome** that every process loads
  (16.8 GB for GRCh38: `Shared memory is not used for genomes. Allocated a private copy of the genome.`): k copies of the same
  index. The E8 hand-off region in `/dev/shm` adds ~6.5 GB per sample.
  - A half node (28 cores, 370 GB cgroup) holds k ≤ 8–9 such samples; a whole node (750 GB) ~18. The mixed set's larger
    samples (nuclei2k_v2, 118.7 M reads) need more (§2).
  - One shared genome per node (STAR's `--genomeLoad LoadAndKeep`, or E14's preload in a persistent process) would cut ~17 GB per
    sample and raise the node's bound to ~30 samples.
- **Cores:** 8.0–8.4 core-s per M pairs per run at every k (T = 7: 7.7–8.0), so at the slot's limit (~1.6–1.8 M pairs/s) a GPU
  needs ~13–15 cores on average. Bursts are higher: at the start of a round the k samples decode and scan together (24 of 28
  cores at k = 6), which stretches the prefill's host steps under the slot (two runs of `k6t7s1r2` held it 32 s instead of 21 s:
  records 5.4–11.0 s instead of 3.1, the reads' pageable upload at 2.3 GB/s instead of 8–12).
- **Host memory bandwidth** shows only there: the host-only stages run as fast at k = 4 as alone (export 17–18 s against 17 s,
  demux 23–24 s against 23 s, the prefill's uploads 5.1–5.6 GB/s for the index and 7.4–12.1 GB/s for the reads), and the pageable
  uploads under the slot slow down at k = 6 when the round's starts coincide.

## 6. Recommendation for the campaign

For the binary of the merged head (or later), level 5 + ingest, on H100 NVL nodes:

| knob | value | why |
|---|---|---|
| `SINGLET_ENGINE_GPU_SLOTS` | **1** (the default) | a second slot gains 0–8 % and needs a hand-tuned memory split (§5c) |
| samples in flight per GPU (k) | **@@** (@@ per node) | a FIFO slot needs many samples to stay busy (§5b); host memory allows ~43 GB per sample on the merged head (§5d) |
| threads per sample (`--threads`, T) | **7** (oversubscribed: k × T ≈ @@ threads on @@ cores) | T only matters through the host work under the slot (the prefill's host lane runs on T − 1 threads); below 7 the hold grows, above it the cores contend at round starts |
| device per worker | **pinned**: `SINGLET_ENGINE_GPU_DEVICE` = worker mod GPUs | @@ |
| scheduling | a pool of k workers per GPU, each starting its next sample when its last one ends (the stream form) | batches pay the 31 s ramp and the 53 s tail per batch: −5–10 % |
| memory knobs (`CHUNK`, `NB*`, `ARENA_GB`) | **none** | with one slot, lane B and the arena may take the whole device |

**Expected:** @@ M pairs/s per node for 3' samples like pbmc1k_v3, i.e. the campaign's ~20.6 T read pairs in ~@@ node-days, against
~170 node-days on the CPU engine (1.4 M pairs/s per node).

**What would change the recommendation:**
- a priority in `lockDevice` (§5b): the same rate at k = 5 per GPU, half the host memory;
- holding the slot only around device work (§4): ~2.4 M pairs/s per GPU, so 4.5–4.8 per node;
- a persistent per-GPU process (resident indexes and context, one genome copy per node): ~3 M pairs/s per GPU and 2–3x more samples
  in flight per node.

## 7. What E17's budget must provide

With `SLOTS=1` (the recommendation) no budget is needed for one H100: one process holds the device at a time, the holder's phases
end with `cudaDeviceReset`, and a waiting sample keeps at most a ~0.7 GB CUDA context (5.0 GB at most for 5 waiters at k = 6, §5d),
which lane B and the arena absorb because they size themselves from free memory. A budget becomes necessary for `SLOTS ≥ 2` (two
device phases at once), for A100-80/40, and for the barnyard genome. Since waiting samples' contexts add up with k, the budget
should count them (k × ~0.7 GB) rather than assume an empty device.

The numbers it has to work with (pbmc1k_v3, GRCh38, H100 NVL 94 GB; engine memory marks are device-wide, from `cudaMemGetInfo`):

| phase | device memory | what sets it | knob today |
|---|---|---|---|
| prefill before the lanes | **26.4 GB** = index 16.4 + reads/clip 9.5 + context 0.5 | genome size; reads per chunk | `SINGLET_ENGINE_GPU_CHUNK` (reads part only) |
| lane A Work | 36 KB per read in a batch: 19.1 GB at the default 524,288 | batch | `SINGLET_ENGINE_GPU_NB` |
| lane M Work | 68 KB per read: 26.9 GB at 393,216 | batch | `SINGLET_ENGINE_GPU_NBM` |
| lane B Work | 550 KB per read: **free − 3 GB** (64 GB here) unless set | free memory | `SINGLET_ENGINE_GPU_NBB` |
| lane outputs | n × NBLOB per chunk (640 / 2048 / 4096 B for A / M / B) | chunk | `SINGLET_ENGINE_GPU_CHUNK` |
| prefill peak | **90.4 GB** (lanes peak) | lane B takes what is left | – |
| hand-off pass | 7.2 GB | reads with a record | none |
| Solo | 5.0 GB | kept inputs (1.6 GB) + work | none |
| pileup | arena **free − 3 GB** (92 GB here); used 26.4 GB (est 27.9) for pbmc1k_v3 | free memory; reads | `SINGLET_GPU_PILEUP_ARENA_GB` (cap; cell partitions K adapt) |
| non-host | **24.9 GB index, fixed** + ~3.9 GB work = 28.8 GB peak | the index | **none** |

- **What a slot needs:** with `SLOTS = s` each process must stay within (94 − margin) / s. Two slots (44–45 GB each) work with
  today's knobs, as the split `m43`:
  `SINGLET_ENGINE_GPU_CHUNK=4194304 SINGLET_ENGINE_GPU_NB=274432 SINGLET_ENGINE_GPU_NBM=155648 SINGLET_ENGINE_GPU_NBB=21504 SINGLET_GPU_PILEUP_ARENA_GB=38`.
  Measured per-process device peaks 43.6–45.2 GB; two processes together 84.7–88.4 GB; identical outputs.
  - Its cost alone: the prefill holds 27.4 s instead of 18.2 s (k1t14s1m43 vs k1t14s1): the smaller lanes run more batches
    (lane B 7.5 s instead of 3.0 s), and the chunk of 4 M reads adds rounds.
- **Three slots do not fit:** (94 − 3) / 3 = 30 GB, but the prefill holds 26.4 GB before its first lane, and the non-host screen
  needs 28.8 GB with no knob.
- **What E17 should provide, in order of use for the campaign:**
  1. One budget variable (`SINGLET_ENGINE_GPU_MEM_GB`, or derived as (device − margin) / slots) that sets the chunk, the three
     lane batches and the pileup arena together, instead of five variables tuned by hand (`m43` is that by hand for 2 slots).
  2. Lane B and the pileup arena sized from the budget, not from free memory. Today a second process that starts its prefill while
     another holds the device sees the other's allocations, so free − 3 GB is a race (harmless only with `SLOTS=1`).
  3. The non-host index inside the budget: it is a fixed 24.9 GB (with ~4 GB of work) and the floor for any slot. It is also loaded
     from the host on every sample (3.6 s under the lock, §4).
  4. The base before the lanes (26.4 GB): the index (16.4 GB) is the floor for GRCh38; reads/clip (9.5 GB) scale with the chunk.
     A barnyard or a larger genome raises the index part, and the budget must then shrink the lanes, not fail (hgmm1k_v3, E15).
  5. The measured peak per phase in the log, so that a slot split can be checked from one run.
- **What E18 says the budget is for:** not for samples in flight on one H100 (`SLOTS=2` gains 0–8 % over `SLOTS=1` at the same k,
  §5), but for A100-40/80, for big genomes, and for a future design in which phases of two samples overlap on purpose (one
  sample's transfers under another's kernels), which needs both to fit at once.

## 8. Against II.1's targets and the CPU engine

| | target / reference | E18 measured |
|---|---|---|
| end to end per H100 (brief, II.1 §2.1.4) | ≥ 10 M pairs/s; P1 kill < 5 M/s | **1.40** (k = 6) – **@@** (k = 8) M pairs/s per GPU in steady state on the merged head; the slot's limit 1.84 |
| node (II.1 §2.1.4) | ≥ 20 M pairs/s | **@@ M pairs/s** (2 GPUs, 54 cores) |
| node, projected in Part I if cores bind | GPU path 2.65–3.07 M pairs/s | cores do not bind: the GPU slot does |
| II.13's balance point | ~3 samples in flight per GPU at 8.5 core-s/M and ~25 GPU-s per sample | the slot is held 36–41 s per sample (not 25), and at k = 3–6 a FIFO slot sits idle 30–40 % of the time in a convoy: k = 3 gives 1.0–1.1 M pairs/s, 60 % of the limit |
| host core-s per M pairs | ≤ 12.7 (cloud break-even, 54-core node); not worse than the CPU path | **7.7–8.5** per run (`time`), 8.2–9.7 for the whole cgroup with the sampler, flat in k |
| CPU engine node throughput | 54–56 cores / 36–40 core-s per M ≈ 1.35–1.56 M pairs/s (1.44 at 56 / 39) | GPU node **@@x** that, using @@ of 54 cores |
| peak device memory per sample (II.1 §2.2) | ≤ 30 GB for two samples per H100 NVL | 90.4 GB (lane B and the arena take free memory); 43.6–45.2 GB under `m43` |

**Where the remaining distance to 10 M pairs/s per GPU is:** at 66.6 M pairs per sample, 10 M pairs/s means 6.7 s of exclusive device
time per sample. The kernels alone are 11.6 s (§4), so even a slot held only for kernels gives ≤ 5.7 M pairs/s per GPU at today's
kernel speed; the slot is held 3.1x the kernel time on the merged head (3.5x on 44a6e9fa). The order of the levers is therefore:
1. hold the slot only around device work (§4: ~7.9 s on the merged head, → ~2.4 M pairs/s per GPU);
2. a persistent per-GPU process with the index, the non-host index and a context resident (~6 s more, → ~3 M pairs/s);
3. a slot order that keeps it busy at small k (§5b), so that the limit is reached with the samples host memory allows;
4. then the kernels (the M5 lanes are 8.2 of the 11.6 s; II.11 puts the exact port near its ceiling, so past ~4–5 M pairs/s per GPU
   the gain is in contract 2).

## 9. Surprises

- **The slot is held three to four times as long as the kernels run** (41 s against 11.6 s on 44a6e9fa; §4). Most of the
  difference is not device work at all: the prefill's host lane and records (5.5 s with the device idle), the pileup's digest and
  copies (2.8 s with a 92 GB arena allocated and idle), and transfers of data that was on the device a minute earlier (the
  hand-off pass, the non-host index).
- **A convoy keeps the slot idle 31 % of the time at k = 4** although every run waits 70–100 s for it (§5b). Samples
  synchronise through the lock; the naive "k = life / held" bound (3.7) is off by 2x.
- **Priority inversion:** Solo needs the device for 0.7 s and waited up to 56 s behind other samples' 23 s prefills.
- **`SLOTS=2` buys almost nothing** (0 to +8 %, §5c): two prefills on one device time-slice (36–43 s each instead of 18–25 s
  alone), and the memory split makes each one slower even alone (27.4 s against 18.2 s).
- **More threads per sample shorten the slot hold:** the prefill's host lane runs on T − 1 threads beside the device lanes and its
  wait after the lanes falls from 1.79 s at T = 7 to 0.07 s at T = 14; the records' assembly too. Since the cores are mostly
  idle, oversubscribing T costs nothing.
- **Each sample keeps a private 16.8 GB copy of the genome** in host memory (`Shared memory is not used for genomes`), so host
  memory, not the device, bounds k on a node.

## 10. What remains

In order of what they buy per GPU (pbmc1k_v3; §4 and §5):
1. **Hold the slot only around device work** (engine, not E18's scope): release after lane B's download and take the slot again for
   `e2e.build` (−5.7 s); reset and unlock right after `pileupRun`, before the puts and the host pass (−2.2 s on the merged head);
   read the non-host index into page-locked memory before taking the slot. On the merged head: 36.1 → ~28 s held, 1.84 → ~2.4 M
   pairs/s per GPU.
2. **A priority in `lockDevice`:** later phases before new prefills. Reaches the slot's limit at k = 5–6 instead of 8 in the model
   (§5b), so fewer samples in flight (host memory) for the same throughput.
3. **A persistent per-GPU process** (E14's recommendation: preload once, fork per sample): index, non-host index and a context
   resident; no per-sample context create/reset, no 16.8 GB genome copy per sample. ~22 s held → ~3 M pairs/s per GPU and 2–3x more
   samples in flight per node by host memory.
4. **E17's budget** for `SLOTS ≥ 2`, A100 and big genomes (§7). On one H100 with today's phases, a second slot is not where the
   throughput is.
5. **Not measured here:** other samples' mixes at node scale beyond the mixed set; A100; the input supply (download/convert) that
   II.1 expects to bind the campaign at ~1 GB/s.

## 11. Files touched

All new, under `bench/fable/throughput/` (no engine code, no other bench code):

| file | what |
|---|---|
| `jobs/build.sbatch` | E15's build job with E18's branch, refs and output directory |
| `jobs/run.sbatch` | the experiment: points `[mix]k<K>t<T>s<S>[m43][r<R>][p][f]`, k worker processes per point taking samples from a queue, the sampler, the per-point analysis, identity against the k = 1 reference, the E6 dump hashes |
| `e18_sampler.py` | per tick: the cgroup's CPU and memory (anon / file / shmem), per run CPU, RSS and PSS (runs found by `E18_COPY` in `/proc/<pid>/environ`), the holder and waiters of every lock file (`/proc/locks`), `nvidia-smi` utilisation and memory per device and per process |
| `e18_point.py` | per run: wall, core-s, RSS, pairs, every lock wait/hold from the engine's log lines, the prefill's steps and bandwidths, stage walls and stage core-s; per point: span, pairs/s, cgroup core-s/M, lock held share, peaks; `--table` |
| `e18_locks.py` | the lock-held intervals per run from the sampler, labelled by phase, with the device's utilisation, memory and the run's CPU in each |
| `e18_steady.py` | the steady-state rate (runs started after the first completion), and the lock held and CPU in that window |
| `e18_trace.py` | one run's 0.2 s trace (lock, device utilisation and memory, CPU cores) between two times |
| `e18_sim.py` | the closed-queue model of §5b: k workers per GPU replaying the k = 1 profile of either head (`e15`, `e16`), waiters served `fifo`, in `random` order, `late` phases first or `early` first; `python3 e18_sim.py fifo 1 e16` |

## 12. Jobs, commits, how to run

**Branch `gpu/fable/throughput`** (from gpu/fable-engine 44a6e9fa):
- 7a2bbb9e: build job.
- a421ccfa: run job, sampler, per-point analysis.
- 9907d324: the prefill's upload bandwidths and step groups in the analysis.
- 8d967a46: lock intervals per run from the sampler (`e18_locks.py`), fine-sampling tag `f`.
- a900b32e: identity checks of a point's runs in parallel (`IDPAR`).
- 86084d19: `e18_steady.py`, `e18_trace.py`.
- c6ddd3d3: merge of gpu/fable-engine e0e1514a (E16 "tail").
- 3b93a792: lock intervals labelled by memory and length (E16's early hand-off pass).
- f3af8b8e: mix points take the MIX list R times.
- d4a7e60e: the steady window (every worker busy).
- 9bb39905: the queue model (`e18_sim.py`).
- This document is the next commit.

**Builds** (short-gpu without a GPU, ~5 min):

| job | commit | engine | output (`$W/jobs/fable/throughput/`) | sha256 |
|---|---|---|---|---|
| 797401 | 7a2bbb9e | 44a6e9fa (E1–E15) | `bin/singlet-gpu` | 51e0b035… |
| 797455 | 3b93a792 | c6ddd3d3 (+ E16) | `bin-e16/singlet-gpu` | 28fe749c… |

**Runs** (short-gpu, H100 NVL, `--nice=10000`):

| job | node | GPUs / cores / mem | binary | points | notes |
|---|---|---|---|---|---|
| 797417 | g051 | debug-gpu, 1 / 8 | bin | pbmc2m `k1t4s1 k2t4s1` | script debugging; k = 2 identical to k = 1, dumps identical |
| 797423 (A) | g050 | 1 / 28 / 370 GB | bin | `k1t28s1 k1t14s1 k1t9s1 k1t7s1 k2t14s1 k3t9s1 k4t7s1 k4t14s1 k1t14s1m43 k2t14s2m43 k3t9s2m43 k4t7s2m43` | 43:43 |
| 797443 (B) | g050 | 1 / 28 / 370 GB | bin | `k1t7s1f k4t7s1r3 k6t7s1r2` | cancelled after `k6t7s1r2` when E16 landed (its mixed-set points are void); a build ran beside it on the same node during `k6t7s1r2` |
| 797460 (C) | g051 | 1 / 28 / 370 GB | bin-e16 | @@ | merged head |
| @@ (D) | @@ | 2 / 54 / 700 GB | bin-e16 | @@ | the node |

**Results:** `$W/jobs/fable/throughput/results/run-<job>/`: `summary.txt` (everything below in one file), `table.md`, `dumps.tsv`,
and per point `point.txt` / `point.json` (analysis), `locks.json`, `samples.jsonl` (the sampler), `runs.tsv`, `meta.json`,
`identity.txt`, and per run `c<i>/` (`time.txt`, `run.log.ts`, `star_Log.out`, `star_Log.final.out`, `CellReads.stats`,
`read_stats.tsv`, `summary.json`, `pileup_stats.json`, `identity.txt`). Logs: `$W/logs/e18-{build,run*}-<job>.out`.

**Run:**
```
sbatch --export=ALL,OUTD=$W/jobs/fable/throughput/bin-e16 bench/fable/throughput/jobs/build.sbatch
sbatch --export=ALL,BINDIR=$W/jobs/fable/throughput/bin-e16,POINTS="k1t7s1f k6t7s1r2" bench/fable/throughput/jobs/run.sbatch
sbatch --gres=gpu:nvidia_h100_nvl:2 -c 54 --mem=700G --export=ALL,BINDIR=…,POINTS="k1t7s1 k12t7s1r2p" bench/fable/throughput/jobs/run.sbatch
```
Analysis on a point directory: `e18_point.py <dir>`, `e18_locks.py <dir>`, `e18_steady.py <dir>`, `e18_trace.py <dir> <from> <to>
[item]`; `e18_point.py --table <run dir>`. The model of §5b: `e18_sim.py` (in this document's appendix; not in the tree).
