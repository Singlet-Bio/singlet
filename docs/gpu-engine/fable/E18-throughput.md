# E18: samples in flight per GPU (throughput)

Experiment E18 ("throughput") of the GPU aligner program. Branch `gpu/fable/throughput`, from `gpu/fable-engine` at 44a6e9fa
(E1–E15), with `gpu/fable-engine` merged in at e0e1514a (E16, "tail") as c6ddd3d3 and at 503c07f7 (E17, "budget") as 5e55b702. The
question: how many read pairs per second does one node (2 H100 NVL, 54 usable cores, 750 GB) deliver when several samples share each
GPU through the engine's device slots (`SINGLET_ENGINE_GPU_SLOTS`), what limits it, and which configuration the campaign should run.

Everything runs level 5 + ingest (`--engine gpu`, `SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1`), the
configuration E19 is making the default. E18 changed no engine code: new bench scripts under `bench/fable/throughput/` only.

## 0. Answer in brief

- **The curve (pbmc1k_v3, one H100 NVL, 28 cores, M read pairs/s; batch at k ≤ 4 unless "stream", steady state for streams):**

  | samples in flight k | 1 | 2 | 3 | 4 | 6 | 8 | limit |
  |---|---|---|---|---|---|---|---|
  | 44a6e9fa (E15), `SLOTS=1` | 0.44–0.49 | 0.80 | 1.00 | 1.09–1.12, **1.16** stream | **1.27** stream | – | 1.62 |
  | E16 head, `SLOTS=1` | 0.52 (T = 7), 0.57 (T = 9) | – | **1.10** | – | **1.40** stream | **1.57** stream | 1.84 |
  | E17 head, `SLOTS=1` | 0.52–0.54 (T = 7), 0.61 (T = 14) | 0.97 | **1.10** | 1.16 | **1.44** stream | – | ~1.84 |
  | **E17 head, `SLOTS=2`** | – | 0.99 | 1.21 | 1.36 | **1.86** stream | **2.17** stream | ~2.8 |

  **The node (2 GPUs, 54 cores, E17 head): 4.39 M pairs/s in steady state with two slots and 8 samples per GPU, pinned** (3.24
  over the span of 32 samples; one slot, 6 per GPU: 2.74), against 1.38–1.44 M pairs/s for the CPU engine on the same node (54–56
  cores / 39 core-s per M): 3.0–3.2x.
- **With one slot the bottleneck is the device slot, not the cores, the host memory bandwidth or the device memory (§5a).** Each
  sample holds the GPU 41.1 s on 44a6e9fa and 36.1 s on the E16 head (T = 7), against 11.6 s of kernels. The limit per GPU is
  66.6 M pairs / held time: 1.62 and 1.84 M pairs/s. The cores are mostly idle (8.0 core-s per M pairs per run, flat in k: 10–17
  of 28 cores).
- **Per phase, at k = 1 (§4):** of the 36.1 s held on the E16 head, ~7.9 s is host work with the device idle that could run
  outside the slot without changing any device work:
  - the prefill's host lane wait, record assembly and `readRecord` between lane B and `e2e.build` (5.7 s);
  - since E16's `resetbg`, the pileup's slot is held 2.2 s past `pileupRun` through the accumulator puts and the host pass (the log
    reports 4.1 s held; the sampler sees 6.3 s).
  E16 already took 5 s off the hold against 44a6e9fa's 41.1 s (pileup 9.0 → 6.3 s, zero-copy hand-off pass 3.9 → 2.5 s). With the
  slot held only around device work the one-slot limit is ~2.4 M pairs/s per GPU; with the index, the non-host index and a context
  resident (a persistent per-GPU process) ~3.0.
- **Below the limit the samples form a convoy (§5b).** At k = 4 the slot is idle 31 % of the steady window although every run
  waits 70–100 s for it: the samples leave their device phases together, run their host-only tails and starts together, and nobody
  needs the device meanwhile. A FIFO queue model built from the k = 1 profile alone reproduces every one-slot point within 1–5 %
  (k = 8 on the E16 head: model 1.582, measured 1.567) and says a FIFO slot needs k ≈ 10–12 per GPU to come within 6 % of its
  limit. Serving later phases first would reach the limit at k = 5 (model).
- **Two slots pay since E17 (§5c): +29 % at k = 6 and +38 % at k = 8 in steady state, 2.17 M pairs/s per GPU**, past the one-slot
  limit, with nothing set but `SINGLET_ENGINE_GPU_SLOTS=2` (E17's budget gives each process 45.5 GB). Two slots let a second sample
  use the device during the first's host work under its slot and the convoy's gaps. With E15's hand-made split (`m43`) two slots
  gained 0–8 %, because the split slowed a prefill alone from 18.2 to 27.4 s. Three slots add nothing over two at k = 3.
- **What bounds k is host memory (§5d):** ~35 GB per pbmc1k_v3 sample in flight on 44a6e9fa and 41–43 GB since E16 (~0.7 GB per M
  read pairs; nuclei2k_v2: 83 GB), the largest piece being each process's private 16.8 GB copy of the genome. A half node (370 GB)
  holds k = 8, the node 16 (607 GB). Device memory does not bound k: a waiting sample keeps ≤ 0.7 GB of context, and E17's lanes stay
  within the free memory.
- **Identity holds at every k and slot count (§3):** 260 runs compared with their sample's k = 1 run in the same job: every count
  matrix, Solo file, `Log.final.out` statistic, `SJ.out.tab` and `CellReads.stats` identical; only E11's run-dependent files differ.
  The mapDistinct dumps of all 41 points are identical (sha256 77d7bc20008166af). One pipeline bug surfaced, independent of k:
  nuclei2k_v2's `nonhost_reads.fq` holds a different record set in each run, because the writer indexes the screen's subsample
  against the whole unmapped file, whose order E16 made run-dependent (§3).
- **Recommendation (§6): `SINGLET_ENGINE_GPU_SLOTS=2` (E17's default budget, no other memory knobs), 8 samples in flight per GPU
  where host memory allows (16 per node; ~0.7 GB per M read pairs per sample), `--threads 7`, a pool of workers that starts the
  next sample when one ends, each worker pinned to a GPU (`SINGLET_ENGINE_GPU_DEVICE` = worker mod 2).** Expected ~4.4 M pairs/s
  per node for samples like pbmc1k_v3: the campaign's ~20.6 T read pairs in ~54 node-days against ~170 on the CPU engine.
- **E17's budget (§7)** provides what E18 found a second slot needs: one budget for the chunk, lanes and arena, sized per slot and
  capped by the free memory. What it does not cover: the non-host screen's fixed 25–29 GB (outside the budget, so 4 slots would not
  fit) and the host memory per sample, which is what bounds k.

## 1. Setup

- **Binaries:** `singlet-gpu` built by `bench/fable/throughput/jobs/build.sbatch` from this branch, three engine heads (§12):
  44a6e9fa (E1–E15; jobs A, B), c6ddd3d3 = E15 + E16 (job C) and 5e55b702 = E15 + E16 + E17 (jobs D, E, F). No engine code changed
  in E18.
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
- **Nodes:** g050 (jobs A, B), g051 (jobs C, D, F) and g052 (job E), Xeon Gold 6238R (2 × 28 cores, 2 NUMA nodes), 750 GB, 2 H100 NVL
  94 GB. Slurm gives 54 usable cores per node (`CPUEfctv`). Jobs A, B, C, D and F use one GPU and 28 cores (half a node; other jobs
  held the other half), job E the whole node.

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

**E16 head c6ddd3d3 (E15 + E16), job 797460 (C), node g051:**

| point | form | k | T | slots | runs | **pairs/s (M)** span / steady | wall s min / mean / max | core-s/M runs / cgroup | lock wait / held s | held % | host peak GB | device peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k1t7s1f | – | 1 | 7 | 1 | 1 | **0.522** | 127.7 | 7.70 / 8.80 | 0.4 / 36.1 | 28 | 52 | 90.3 |
| k1t9s1 | – | 1 | 9 | 1 | 1 | **0.573** | 116.3 | 7.89 / 8.29 | 1.4 / 34.0 | 29 | 54 | 90.3 |
| k3t9s1 | batch | 3 | 9 | 1 | 3 | **1.104** | 172.9 / 177.6 / 181.0 | 7.98 / 8.33 | 80.4 / 34.1 | 58 | 141 | 90.3 |
| k6t7s1r2 | stream | 6 | 7 | 1 | 12 | **1.329 / 1.398** | 282.5 / 294.5 / 315.1 | 7.77 / 8.16 | 214.2 / 34.7 | 69 | 274 | 90.4 |
| k8t7s1r2 | stream | 8 | 7 | 1 | 16 | **1.440 / 1.567** | 324.2 / 361.7 / 415.8 | 7.75 / 8.15 | 284.8 / 33.6 | 84 | 352 | 90.4 |
| mixk1t7s1 | one at a time | 1 | 7 | 1 | 4 | **0.502** | 125.9 / 168.4 / 251.4 | 7.27 / 7.76 | 0.6 / 50.1 | 30 | 115 | 90.3 |
| mixk6t7s1r2 | stream, 8 items | 6 | 7 | 1 | 8 | **1.130** span | 202.3 / 315.9 / 393.9 | 7.37 / 7.78 | 188.7 / 47.5 | 64 (span) | 286 | 90.2 |

- **The k = 3 re-run on the E16 head: 1.104 M pairs/s against 0.995 on 44a6e9fa (+11 %)**, both batch, T = 9, one GPU, 28 cores
  (different nodes, g051 and g050; the k = 1 runs differ by the same order: 0.573 against 0.474 at T = 9). The slot is held 34.1 s
  per run instead of 37.1 s.
- At k = 1 a run on the E16 head waits 0.4–1.4 s for its own Solo: E16 runs Solo (STAR's child) beside the pipeline's pileup,
  and they take the same slot.
- Host memory per sample in flight is higher on the E16 head (k = 6: 274 GB against 210 GB; anon + shmem at most 255 GB, 43 GB
  per sample): E16 keeps the prefill's data until after STAR's loop and page-locks the hand-off region.
- **k = 8 on the E16 head: 1.567 M pairs/s in steady state** (1.440 over the span), the slot held 84 % of the steady window,
  352 GB of the half node's 370 GB cgroup (anon 290 + shmem 54). It is the largest k a half node holds.
- **The mixed set** (pbmc1k_v3 66.6 M pairs, pbmc5p_v1 60.0, neuron1k_v3 92.9, nuclei2k_v2 118.7; 338.2 M pairs in all). Alone, the
  samples hold the slot 37.6 / 36.9 / 60.0 / 65.7 s (sampler): 0.59 s per M pairs against 0.54 for pbmc1k_v3. neuron1k_v3's pileup
  holds it 28.7 s (pbmc1k_v3: 4.2 s), and nuclei2k_v2's prefill 49.6 s. The slot's limit for this mix is 338.2 M / 200.2 s = **1.69 M
  pairs/s per GPU** (pbmc1k_v3 alone: 1.84). With 6 workers and only 8 items, `mixk6t7s1r2` is mostly ramp and tail (1.130 over the
  span; pbmc1k_v3 alone at k = 6: 1.329). Host memory per run scales with the sample: maxRSS 38.7 / 45.1 / 59.9 / 83.4 GB, ~0.65–0.70
  GB per M read pairs with the 16.8 GB genome included.
- The node's other half ran other jobs during job C (E17's gate jobs 797462 and 797468 on the other GPU and 18 cores, an E19 encode
  job): separate GPUs and cores, but the same memory bus and page cache.

**E17 head 5e55b702 (E15 + E16 + E17's budget), job 797485 (D), node g051:**

| point | form | k | T | slots | runs | **pairs/s (M)** | wall s min / mean / max | core-s/M runs / cgroup | lock wait / held s | held % | host peak GB | device peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k1t7s1f | – | 1 | 7 | 1 | 1 | **0.520** | 128.1 | 7.79 / 8.90 | 0.2 / 38.4 | 30 | 52 | 89.6 |
| k1t9s1 | – | 1 | 9 | 1 | 1 | **0.557** | 119.6 | 7.91 / 8.32 | 2.0 / 35.6 | 29 | 54 | 89.3 |
| k1t14s1 | – | 1 | 14 | 1 | 1 | **0.607** | 109.6 | 8.17 / 8.53 | 1.5 / 32.4 | 29 | 56 | 89.3 |
| k2t14s1 | batch | 2 | 14 | 1 | 2 | **0.969** | 134.9 / 136.2 / 137.5 | 8.36 / 8.69 | 34.4 / 31.5 | 45 | 100 | 89.4 |
| k2t14s2 | batch | 2 | 14 | 2 | 2 | **0.987** | 129.3 / 132.2 / 135.0 | 8.61 / 8.92 | 0.8 / 50.7 | 43 | 102 | 86.5 |
| k3t9s1 | batch | 3 | 9 | 1 | 3 | **1.096** | 173.1 / 178.1 / 182.3 | 8.02 / 8.37 | 78.7 / 33.8 | 57 | 142 | 90.1 |
| k3t9s2 | batch | 3 | 9 | 2 | 3 | **1.210** | 153.5 / 158.7 / 165.1 | 8.19 / 8.51 | 31.6 / 45.7 | 48 | 139 | 87.4 |
| k3t9s3 | batch | 3 | 9 | 3 | 3 | **1.206** | 155.7 / 159.7 / 165.7 | 8.71 / 9.04 | 2.3 / 72.8 | 52 | 145 | 87.1 |
| k4t7s1 | batch | 4 | 7 | 1 | 4 | **1.164** | 216.7 / 222.7 / 228.9 | 7.77 / 8.15 | 128.8 / 34.5 | 62 | 184 | 90.1 |
| k4t7s2 | batch | 4 | 7 | 2 | 4 | **1.361** | 149.6 / 181.7 / 195.7 | 7.94 / 8.26 | 41.9 / 45.2 | 56 | 155 | 86.3 |

- **The k = 3 point on three heads** (batch, T = 9, one GPU, 28 cores): 0.995 (44a6e9fa), 1.104 (E16), 1.096 (E17). E17's budget
  does not change the rate at `SLOTS=1`: its default (91.1 GB, and no more than the free memory less 2 GB) gives the lanes almost
  what E12's "free − 3 GB" gave. The slot is held 33.8 s per run at k = 3 (E16: 34.1). The single k = 1 runs held it 38.4 s at T = 7
  (E16: 36.1) and 35.6 s at T = 9 (E16: 34.0), with the prefill's lanes and `e2e.build` 0.3–0.6 s slower each; one run per point,
  so this is within a run's spread.
- **With E17, two slots pay (§5c): +10 % at k = 3 and +17 % at k = 4** against `SLOTS=1` in the same job, using only
  `SINGLET_ENGINE_GPU_SLOTS=2`.

**E17 head, steady state with one and two slots, job 797491 (F), node g051** (held %: each slot's lock in the steady window):

| point | form | k | T | slots | runs | **pairs/s (M)** span / steady | wall s min / mean / max | core-s/M runs / cgroup | lock wait / held s | held % | host peak GB | device peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k1t7s1 | – | 1 | 7 | 1 | 1 | **0.537** | 123.9 | 7.69 / 8.12 | 1.6 / 35.8 | 28 | 52 | 89.3 |
| k6t7s1r2 | stream | 6 | 7 | 1 | 12 | **1.355 / 1.441** | 275.3 / 289.5 / 314.4 | 7.78 / 8.16 | 208.7 / 34.5 | 72 | 267 | 90.1 |
| k6t7s2r2 | stream | 6 | 7 | 2 | 12 | **1.686 / 1.862** | 201.0 / 231.9 / 258.1 | 8.03 / 8.34 | 115.8 / 46.6 | 58 + 47 | 275 | 89.1 |
| k8t7s2r2 | stream | 8 | 7 | 2 | 16 | **1.847 / 2.167** | 219.2 / 279.1 / 343.3 | 8.06 / 8.37 | 147.2 / 47.4 | 80 + 68 | 342 | 89.2 |

- **Two slots at k = 6–8 in steady state: 1.86 and 2.17 M pairs/s per GPU, +29 % over one slot at k = 6 (1.441) and +38 % at k = 8
  (E16 head, one slot: 1.567).** k = 8 with two slots is past the one-slot limit (1.84). Every run identical (§3), no fallback, the
  device at 89.1–89.2 GB.
- The cores stay mostly idle: 14.9 cores (k = 6) and 17.1 cores (k = 8) of 28 in the steady window, 8.0–8.1 core-s per M pairs per
  run (one slot: 7.8).
- Host memory with two slots is the same per sample: k = 8 peaked at 342 GB of the 370 GB cgroup (anon 283 + shmem 48 = 330).
- The other GPU of g051 ran E19's gate (797474) until 13:52 and an s15 job (797746) after it.

**The node: E17 head, 2 GPUs, 54 cores, 730 GB, job 798028 (E), node g052.** `p`: worker w pinned to GPU w mod 2. Lock wait and
held are means per run from the engine's log lines; held % is each lock file's share of the steady window (GPU 0 slot 0 + slot 1,
GPU 1 slot 0 + slot 1), approximate: with 32 runs the sampler's ticks stretched past 5 s for half the span (`nvidia-smi` per process),
which does not touch the rates (from the runs' start and end times) or the cgroup's CPU.

| point | form | k (per GPU) | T | slots | runs | **pairs/s (M)** span / steady | wall s min / mean / max | core-s/M runs / cgroup | lock wait / held s | held % | cores in window | host peak GB (anon + shmem) | device peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k1t7s1p | – | 1 | 7 | 1 | 1 | **0.541** | 123.2 | 7.68 / 8.16 | 0.4 / 31.2 | – | – | 52 | 89.3 |
| k12t7s1r2p | stream | 12 (6) | 7 | 1 | 24 | **2.547 / 2.735** | 282.9 / 306.7 / 342.0 | 8.49 / 8.77 | 158.1 / 32.4 | 73; 75 | 21.6 | 531 (437 + 76) | 90.1 |
| k12t7s2r2p | stream | 12 (6) | 7 | 2 | 24 | **3.050 / 3.779** | 156.0 / 248.8 / 314.8 | 8.95 / 9.18 | 89.0 / 45.5 | 71 + 38; 44 + 29 | 37.1 | 498 (414 + 71) | 88.9 |
| **k16t7s2r2p** | stream | 16 (8) | 7 | 2 | 32 | **3.239 / 4.390** | 172.6 / 297.0 / 400.3 | 8.87 / 9.11 | 115.3 / 44.0 | 50 + 41; 73 + 42 | 44.3 | 607 (541 + 93) | 89.3 |
| k16t7s2r2 | stream, unpinned | 16 | 7 | 2 | 32 | **3.149 / 4.076** | 177.8 / 303.1 / 407.0 | 8.91 / 9.15 | 106.8 / 44.8 | 66 + 45; 63 + 47 | 41.8 | 647 (564 + 94) | 89.5 |

- **The node gives 4.39 M pairs/s in steady state with two slots and 8 samples per GPU, pinned**: 2.03x the half node's 2.17 at the
  same setting (job F), 3.0–3.2x the CPU engine on the same node (1.38–1.44). With one slot and 6 per GPU it gives 2.74 (job F's
  half node: 1.44 per GPU); two slots at the same k give 3.78 (+38 %).
- **Pinning matters on the node: 4.39 pinned against 4.08 unpinned (−7 %), and 4 of the 32 unpinned runs lost the GPU in the non-host
  screen and screened on the CPU** (`cudaMalloc … resource already mapped` twice, `cudaSetDevice(device): CUDA-capable device(s)
  is/are busy or unavailable` twice). Unpinned, a sample's phases land on either GPU (12 of the 32 ran their prefill and their
  non-host screen on different GPUs), so each process can open a context on both. No run fell back in any other point of any job (one
  GPU, or pinned).
- **The cores start to count:** 44 of 54 cores busy on average in the k = 16 window (9.1 core-s per M pairs for the cgroup; per run
  8.9 against 8.1 on the half node, so the runs contend). At ~9 core-s per M pairs, 54 cores carry at most ~6 M pairs/s per node.
- **Host memory:** 607 GB at k = 16 (anon 541 + shmem 93; the slow ticks may miss the true peak) of the 730 GB cgroup; the node
  (735 GB allocatable) does not hold 18 such samples.

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

**Results:** every run of every point (260 runs in jobs A–F) is identical to its sample's k = 1 run except for the run-dependent files
and one pipeline bug in nuclei2k_v2's `nonhost_reads.fq` (below).

| job (engine) | runs compared | out_equiv: identical / timings-paths / differ | the differing files | `Log.final.out` stats, `SJ.out.tab`, `CellReads.stats` | E6 dumps |
|---|---|---|---|---|---|
| 797423 A (44a6e9fa), SLOTS=1 | 13 | 63 / 3 / 3 | `pileup_stats.json` (`bytes_public`, timings), `read_stats.tsv`, `star_Log.std.out` | identical; the 3 read files byte-identical | all identical, sha256 77d7bc20008166af (E12's) |
| 797423 A, `m43` (SLOTS=1 and 2) | 10 | 63 / 2 / 4 | the same 3 + `summary.json`'s `memory_tier` | identical | identical |
| 797443 B (44a6e9fa) | 24 | 63 / 3 / 3 | as above | identical | identical |
| 797460 C (merged, E16) | 40 (4 samples) | 60 / 3 / 6 (pbmc5p_v1: 58 / 3 / 6) | the same 3 + `nonhost_reads.fq`, `star_Unmapped.out.mate1/2` (same records, another order) | identical; the 3 read files hold the same sorted records, except nuclei2k_v2's `nonhost_reads.fq` (below) | identical |
| 797485 D (E17), SLOTS = 1, 2, 3 | 21 | 60 / 3 / 6; with 3 slots 60 / 2 / 7 | as job C; with 3 slots also `summary.json`'s `memory_tier` (Large → Medium: the 30.4 GB budget lowers the process's VmPeak) | identical; the 3 read files hold the same sorted records | identical (10 points) |
| 797491 F (E17), SLOTS = 1, 2 | 40 | 60 / 3 / 6 | as job C | identical; the 3 read files hold the same sorted records | identical (4 points) |
| 798028 E (E17, the node), SLOTS = 1, 2, pinned and unpinned | 112 | 60 / 3 / 6 | as job C | identical, including the 4 unpinned runs that screened non-host reads on the CPU; the 3 read files hold the same sorted records | identical (5 points) |

On 44a6e9fa even the three read-order files came out byte-identical at every k; with E16 they are written by a background thread
and their record order varies from run to run (E11's known run-dependent set; E16's own gate lists the same three files).

**One exception, not caused by k: nuclei2k_v2's `nonhost_reads.fq` holds a different set of records in each run** (528,892,150 bytes
alone; 528,774,096 and 528,848,534 bytes in its two runs at k = 6). Every count matrix, Solo file, `CellReads.stats`, `star_Unmapped`
file (same records) and the non-host screen's own outputs (`nonhost/`) are identical. The cause is in the pipeline, not the engine
(`src/pipeline/singlet.cpp`, the non-host screen):
- the screen classifies a subsample of `star_Unmapped.out.mate1` when it holds more than 20 M reads (by read-name hash; nuclei2k_v2:
  11.4 M kept) and skips empty sequences;
- `nonhost_reads.fq` is then written by reading the whole of `star_Unmapped.out.mate1` again and taking the read at file position i
  when the i-th *subsampled* read was classified (`multi_hits[idx_r]`).
The written reads are therefore not the classified ones whenever the screen subsampled, and since E16 writes `star_Unmapped.out.mate1`
in a run-dependent order, which reads they are changes from run to run. Samples under 20 M unmapped reads (the other three) are not
affected: index i is then the same read in both passes, so the set is the same and only its order varies. The fix belongs to the
pipeline (write the kept reads' names and qualities from the screen's own pass, or apply the same hash filter in the writer), outside
E18's scope.

## 4. What the slot holds, per phase (k = 1, both heads)

A sample takes a device slot five times. Points `k1t7s1f` (pbmc1k_v3 alone, T = 7) ran with the sampler at 0.2 s ticks and
`nvidia-smi` every tick, on 44a6e9fa (job 797443, g050) and on the E16 head (c6ddd3d3, job 797460, g051). The engine's
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
| E16 head, T = 7 | 36.1 s | **1.84** |
| E16 head, slot only around device work (prefill −5.7 s, pileup −2.2 s) | ~28.2 s | ~2.36 |
| and a persistent per-GPU process: index, non-host index and a context resident (−1.3 s index upload, −3.6 s non-host index, −~1.5 s context create/reset over the five phases) | ~22 s | ~3.0 |
| kernels plus the transfers no design avoids (reads up, records and results down) | ~16 s | ~4.2 |

An estimate of ~1.9 M pairs/s assumed 35 s per sample: on the E16 head the measured hold is 36.1 s at T = 7.

## 5. Why the curve bends where it does

### 5a. With one slot, the device slot is the bottleneck, not the cores, the host memory bandwidth or the device memory

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

The slot is far from busy at k = 3–4 although every run waits for it: in the steady window of `k4t7s1r3` it is held **69 %** of
the time, and the sampler never sees it free with a waiter (every free tick has none; most held ticks have 2–3 waiters). The samples move in a convoy:
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

E16 head:

| k | model FIFO | model random order | **measured** |
|---|---|---|---|
| 1 | 0.521 | 0.521 | 0.522 (T = 7), 0.573 (T = 9) |
| 3 | 1.133 | 1.149 | **1.104** (batch, T = 9) |
| 6 | 1.465 | 1.606 | **1.398** (steady; 1.329 over the span) |
| 8 | 1.582 | 1.782 | **1.567** (steady; 1.440 over the span) |
| 12 | 1.843 | 1.851 | |
| ∞ | 1.843 = 66.6 M / 36.1 s | | |

Three consequences:
- **k must be several times the naive bound** (life / held = 153 s / 41 s ≈ 3.7) to keep a FIFO slot busy: the model needs
  k ≈ 10–12 per GPU to come within 6 % of the limit on either head.
- **The order in which waiters get the slot is worth as much as E16.** Giving the slot to the sample furthest along (later
  phases before new prefills) reaches the limit at k = 5 on the E16 head (1.84 M pairs/s; FIFO 1.38 at k = 5) and at k = 6 on
  44a6e9fa (1.66). New samples then wait at their prefill instead of delaying the short phases of samples about to finish, and
  the in-flight samples (host memory) are halved for the same rate. That is a change to `lockDevice` (for example a second lock
  file that a prefill also takes and that any later phase takes while it waits), outside E18's scope.
- **Batch starts cost 5–10 %** against the stream form at the same k (k = 4: 1.093–1.120 batch, 1.159 stream): the ramp (31 s
  before the first prefill) and the tail (53 s after the last non-host) are paid once per batch.

### 5c. Two slots per GPU: no gain with a hand-made split on 44a6e9fa, +10–38 % with E17's budget

`SINGLET_ENGINE_GPU_SLOTS=2` lets two samples hold the device at once. Each must then stay within about half the device. On 44a6e9fa
that took the hand-made split `m43` (§7); on the E17 head the budget does it alone, (device memory − 2 GB) / slots: 45.5 GB for two
slots, 30.4 GB for three (job 797485). All points below are batch, one GPU, 28 cores, M pairs/s:

| k | 44a6e9fa SLOTS=1 | 44a6e9fa SLOTS=2 + `m43` | E17 head SLOTS=1 | **E17 head SLOTS=2** (45.5 GB) | E17 head SLOTS=3 (30.4 GB) |
|---|---|---|---|---|---|
| 1 | 0.486 (T = 14) | 0.455 (SLOTS=1 + `m43`) | 0.607 (T = 14) | – | – |
| 2 (T = 14) | 0.802 | 0.807 | 0.969 | **0.987** (+2 %) | – |
| 3 (T = 9) | 0.995 | 0.993 | 1.096 | **1.210** (+10 %) | 1.206 (+10 %) |
| 4 (T = 7) | 1.093 (1.120 at T = 14) | 1.178 (+8 %) | 1.164 | **1.361** (+17 %) | – |

- **Two prefills at once time-slice the device:** each takes 35–36 s when both run together (k2t14s2) against 17.5 s alone, and
  three at once take 53–59 s each (k3t9s3). Without MPS the device does the same kernel work in the same time.
- **What a second slot buys is overlap:** one sample's host work under its slot (the prefill's host lane and records, the pileup's
  puts and host pass, ~8 s per sample on E16+, §4) and its transfers run while the other's kernels do. A FIFO slot also leaves the
  device idle in the convoy (§5b); with two slots a second sample can start a device phase in those gaps.
- **Why `m43` did not pay and the budget does:** `m43` slowed the prefill alone from 18.2 to 27.4 s (lane B's batch of 21,504 reads ran
  7.5 s instead of 3.0 s; a 4 M chunk). E17's plan at 45.5 GB keeps larger batches (chunk 3.96 M in 8 chunks, NB 438,272 / 180,224 /
  31,744), costing a few seconds instead of nine (E17 measured +5.3 s at 40 GB). On the E16 head the rest of a sample's life is
  also shorter (E16's tail), so the slot binds more at the same k.
- **Three slots fit but add nothing over two at k = 3**: lean mode at 30.4 GB (13 chunks of 2.4 M) and three-way time-slicing.
- **Device memory:** 44.8–45.1 GB per process with two slots and 29.7–30.8 GB with three (the non-host screen's 28.5 GB outside the
  budget), 86.3–87.4 GB for the device. No sample failed or fell back; every output identical to the k = 1 run (§3).

**In steady state (stream, T = 7, one GPU, 28 cores; job 797491 and, for one slot at k = 8, the E16 head's job 797460):**

| k | one slot | **two slots** (E17 budget 45.5 GB) | gain | slot-held share of the window, two slots | GPU util | cores |
|---|---|---|---|---|---|---|
| 6 | 1.441 (E17), 1.398 (E16) | **1.862** | +29 % | 58 % + 47 % | 46 % (one slot: 35 %) | 14.9 (one slot: 10.8) |
| 8 | 1.567 (E16) | **2.167** | +38 % | 80 % + 68 % | 59 % (one slot: 42 %) | 17.1 (one slot: 11.4) |

- **Two slots pass the one-slot limit** (1.84 M pairs/s). A sample holds a slot 47.4 s at k = 8 (34.5 s with one slot), because
  its device phases share the GPU with the other slot's, but two samples hold slots at once: 1.48 slot-seconds per second in the
  window. With both slots always held the limit would be 2 × 66.6 M / 47.4 s ≈ 2.8 M pairs/s per GPU.
- The waits shrink where the convoy hurt most: at k = 8 a run waits 42 / 52 / 9 / 8 s for its prefill / Solo / pileup /
  non-host slot, against 46 / 59 / 23 / 23 s with one slot at k = 6 (job F).
- The device does more work per second (utilisation 59 % against 42 %), with the same kernels: the gain is the overlap of one
  sample's host work under its slot and its transfers with the other's kernels, and the convoy's gaps filled by a second sample.
- The device memory stays within the budget at every k: 89.1–89.2 GB for the device, waiting samples' contexts included (with the
  default budget the lanes also take no more than the free memory less 2 GB).

### 5d. What bounds k: host memory

- **Device memory does not bound k with `SLOTS=1`.** The holder's phases end with `cudaDeviceReset`; a waiting sample keeps at most a
  CUDA context, ~0.7 GB (the STAR child's while E15's page-locked Solo inputs live through STAR's loop; on the E16 head also the
  early hand-off pass's until the pileup). The sampler's per-process device memory of the non-holders: median 0–0.7 GB in total,
  95th percentile 3.5 GB, maximum 5.0 GB at k = 6. Lane B and the pileup arena take free memory minus 3 GB, so they shrink by that
  much and nothing fails; the device peak stays 90.0–90.4 GB at every k. With two slots and E17's budget the device stayed at
  86.3–89.5 GB at every k up to 8 per GPU (§5c).
- **Host memory does.** Per sample in flight, at the cgroup's peak (GiB, `memory.current`, of which anon + shmem cannot be
  reclaimed):
  - 44a6e9fa: ~35 GB (k = 4: 153 GB; k = 6: 210 GB, of which anon 188 and shmem 39); a run's PSS peak 40–45 GB;
  - E16 head: ~43 GB (k = 6: 274 GB, anon + shmem 255; k = 8: 352 GB, anon 290 + shmem 54 = 344, under a 370 GB
    cgroup). E16 keeps the prefill's host data until after STAR's loop and page-locks the hand-off region.
  The largest single piece is the **private copy of the genome** that every process loads (16.8 GB for GRCh38: `Shared memory is
  not used for genomes. Allocated a private copy of the genome.`): k copies of the same index. The E8 hand-off region in
  `/dev/shm` adds ~6.5 GB per sample.
  - A half node (28 cores, 370 GB cgroup) holds k = 8 such samples on the E16 head (k = 8 ran at 352 GB); the whole node (730 GB
    cgroup) held 16 at 607 GB (job E). The mixed set's larger samples (nuclei2k_v2, 118.7 M reads) need more (§2).
  - One shared genome per node (STAR's `--genomeLoad LoadAndKeep`, or E14's preload in a persistent process) would cut ~17 GB per
    sample and raise the node's bound to ~28 samples.
- **Cores:** 8.0–8.4 core-s per M pairs per run at every k (T = 7: 7.7–8.0), so at the slot's limit (~1.6–1.8 M pairs/s) a GPU
  needs ~13–15 cores on average. Bursts are higher: at the start of a round the k samples decode and scan together (24 of 28
  cores at k = 6), which stretches the prefill's host steps under the slot (two runs of `k6t7s1r2` held it 32 s instead of 21 s:
  records 5.4–11.0 s instead of 3.1, the reads' pageable upload at 2.3 GB/s instead of 8–12). With two slots a GPU needs 15–17
  cores at k = 6–8, and the node ran 44 of its 54 cores at k = 16 (9.1 core-s per M pairs for the cgroup, the runs' own 8.9
  against 8.1 on a half node).
- **Host memory bandwidth** shows only there: the host-only stages run as fast at k = 4 as alone (export 17–18 s against 17 s,
  demux 23–24 s against 23 s, the prefill's uploads 5.1–5.6 GB/s for the index and 7.4–12.1 GB/s for the reads), and the pageable
  uploads under the slot slow down at k = 6 when the round's starts coincide.

## 6. Recommendation for the campaign

For the binary of the E17 head (503c07f7) or later, level 5 + ingest, on H100 NVL nodes:

| knob | value | why |
|---|---|---|
| `SINGLET_ENGINE_GPU_SLOTS` | **2** | +29–38 % over one slot at k = 6–8 in steady state (§5c); three slots add nothing over two |
| device memory (`SINGLET_ENGINE_GPU_MEM_GB`, `CHUNK`, `NB*`, `ARENA_GB`) | **none set**: E17's default, (device memory − 2 GB) / slots = 45.5 GB | the budget sizes the chunk, the lanes and the arena per slot and keeps the lanes within the free memory; `m43` by hand is slower (§5c) |
| samples in flight per GPU (k) | **8** (16 per node) where host memory allows, else 6–7 | one more sample per GPU is worth ~+8 % per step from k = 6 to 8 (1.86 → 2.17); host memory is ~0.7 GB per M read pairs per sample (pbmc1k_v3: 41–43 GB; §5d), so 16 samples of ~60 M pairs fill a 735 GB node; with larger samples keep the sum of their read pairs under ~1,000 M per node |
| threads per sample (`--threads`, T) | **7** (k × T = 56 threads on 28 cores per GPU) | T only matters through the host work under the slot (the prefill's host lane runs on T − 1 threads); the cores stay under 2/3 busy at k = 8 per GPU (17 of 28 on a half node, 44 of 54 on the node) |
| device per worker | **pinned**: `SINGLET_ENGINE_GPU_DEVICE` = worker mod GPUs | on the node 4.39 M pairs/s pinned against 4.08 unpinned, and 4 of 32 unpinned runs lost the GPU in the non-host screen and screened on the CPU (§2) |
| scheduling | a pool of k workers per GPU, each starting its next sample when its last one ends (the stream form) | batches pay the ramp and the tail per batch: −5–10 % at k = 4, −8–15 % at k = 6–8 (span vs steady) |

**Expected:** ~2.2 M pairs/s per GPU in steady state for 3' samples like pbmc1k_v3 (measured 2.17 on one GPU), **4.4 M pairs/s per
node** (measured 4.39). The campaign's ~20.6 T read pairs then take ~54 node-days, against ~170 node-days on the CPU engine (1.4 M
pairs/s per node). Samples whose slot holds are longer per read pair (the mixed set's neuron1k_v3 pileup, nuclei2k_v2's prefill: 0.59
s per M pairs against 0.54, §2) run ~10 % slower.

**What would change the recommendation:**
- holding the slot only around device work (§4, ~7.9 s per sample on the E16 head): the one-slot limit 1.84 → ~2.4 M pairs/s per
  GPU, and less host work under a slot for the second slot to hide;
- a priority in `lockDevice` (§5b): the one-slot limit at k = 5 per GPU in the model, i.e. fewer samples in flight for the same rate;
- a persistent per-GPU process (resident indexes and context, one genome copy per node): ~3 M pairs/s per GPU with one slot, and
  2–3x more samples in flight per node by host memory.

## 7. What E17's budget must provide, and what it provides

E18 measured the device memory per phase before E17 existed (44a6e9fa and the E16 head; the table below). E17 (503c07f7) landed
during E18, and jobs 797485 (D) and 797491 (F) ran its binary with its default budget.

With one slot no budget is needed on one H100: one process holds the device at a time, the holder's phases end with
`cudaDeviceReset`, and a waiting sample keeps at most a ~0.7 GB CUDA context (5.0 GB at most for 5 waiters at k = 6, §5d), which
lane B and the arena absorb because they size themselves from free memory. A budget is necessary for `SLOTS ≥ 2` (two device phases
at once), which since E17 is the faster configuration (§5c), and for A100-80/40 and big genomes.

The numbers it had to work with, before E17 (pbmc1k_v3, GRCh38, H100 NVL 94 GB, one slot; engine memory marks are device-wide, from
`cudaMemGetInfo`):

| phase | device memory | what sets it | knob before E17 |
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

- **What a slot needs:** with `SLOTS = s` each process must stay within (device − margin) / s. Before E17, two slots (44–45 GB each)
  took the hand-made split `m43`:
  `SINGLET_ENGINE_GPU_CHUNK=4194304 SINGLET_ENGINE_GPU_NB=274432 SINGLET_ENGINE_GPU_NBM=155648 SINGLET_ENGINE_GPU_NBB=21504 SINGLET_GPU_PILEUP_ARENA_GB=38`.
  Measured per-process device peaks 43.6–45.2 GB; two processes together 84.7–88.4 GB; identical outputs. Its cost alone: the
  prefill held 27.4 s instead of 18.2 s (k1t14s1m43 vs k1t14s1): lane B 7.5 s instead of 3.0 s, and the 4 M-read chunk added rounds.
- **What E17 had to provide, in order of use for the campaign, and what it does (E18's measurements on the E17 head):**
  1. **One budget for the chunk, the three lane batches and the pileup arena,** instead of five variables tuned by hand. *Provided:*
     `SINGLET_ENGINE_GPU_MEM_GB`, by default (device − 2 GB) / slots = 45.5 GB for two slots and 30.4 GB for three. Per process
     44.8–45.1 GB with two slots and 29.7–30.8 GB with three, the device at 86.3–89.2 GB at every k up to 8, every output identical.
  2. **Lane B and the arena sized from the budget, not from the free memory** (a second process starting its prefill sees the
     other's allocations, so "free − 3 GB" was a race). *Provided,* and with the default budget the lanes also take no more than the
     free memory less 2 GB, which absorbs the waiting samples' contexts (k × ~0.7 GB). An explicit `SINGLET_ENGINE_GPU_MEM_GB` has no
     such cap (the tally counts the process's own allocations), so with s slots and w waiting samples it should stay under
     (device − 2 GB − w × 0.7 GB) / s.
  3. **The non-host index inside the budget.** *Not provided:* it stays a fixed 24.9 GB (with ~4 GB of work) outside the budget,
     loaded from the host for every sample (3.6 s under the slot, §4). It fits up to three slots (30.4 GB); four slots (22.8 GB each)
     would not.
  4. **The base before the lanes** (26.4 GB: the 16.4 GB index, the reads and clip): *provided* by E17's lean mode (the index after
     the dedup, one chunk's sequences on the device), which three slots use; a barnyard runs at level 5 (E17).
  5. **The measured peak per phase in the log.** *Provided* for the prefill (the tally's peak and where) and the pileup (arena, budget,
     peak); the hand-off pass, Solo and the non-host screen still log no peak.
- **What E18 says the budget is for:** samples in flight on one H100 (`SLOTS=2`, +29–38 % at k = 6–8), A100-40/80 and big genomes.
  The next step it enables is phases of two samples overlapping on purpose (one sample's transfers under another's kernels).
- **What bounds k is not on the device:** host memory (§5d) — ~0.7 GB per M read pairs per sample, 16.8 GB of it a private genome
  copy per process.

## 8. Against II.1's targets and the CPU engine

| | target / reference | E18 measured |
|---|---|---|
| end to end per H100 (brief, II.1 §2.1.4) | ≥ 10 M pairs/s; P1 kill < 5 M/s | **2.17 M pairs/s per GPU** in steady state (E17 head, two slots, k = 8); one slot: 1.44–1.57 at k = 6–8, limit 1.84 |
| node (II.1 §2.1.4) | ≥ 20 M pairs/s | **4.39 M pairs/s** (2 GPUs, 54 cores; two slots, 16 samples, pinned): 22 % of the target |
| node, projected in Part I if cores bind | GPU path 2.65–3.07 M pairs/s | cores do not bind: the GPU slot does |
| II.13's balance point | ~3 samples in flight per GPU at 8.5 core-s/M and ~25 GPU-s per sample | the slot is held 36–41 s per sample (not 25), and at k = 3–6 a FIFO slot sits idle 30–40 % of the time in a convoy: k = 3 gives 1.0–1.1 M pairs/s, 60 % of the one-slot limit; the balance point is k = 8 with two slots |
| host core-s per M pairs | ≤ 12.7 (cloud break-even, 54-core node); not worse than the CPU path | **7.7–8.4** per run (`time`), 8.2–9.7 for the whole cgroup with the sampler, flat in k and slots |
| CPU engine node throughput | 54–56 cores / 36–40 core-s per M ≈ 1.35–1.56 M pairs/s (1.44 at 56 / 39) | GPU node **3.0–3.2x** that (4.39), using 44 of 54 cores |
| peak device memory per sample (II.1 §2.2) | ≤ 30 GB for two samples per H100 NVL | one slot: 89–90 GB (the budget is the device); two slots: 44.8–45.1 GB (E17's 45.5 GB budget); three slots: 29.7–30.8 GB (30.4 GB, lean) |

**Where the remaining distance to 10 M pairs/s per GPU is:** at 66.6 M pairs per sample, 10 M pairs/s means 6.7 s of exclusive device
time per sample. The kernels alone are 11.6 s (§4), so even a slot held only for kernels gives ≤ 5.7 M pairs/s per GPU at today's
kernel speed; one slot is held 3.1x the kernel time on the E16 head (3.5x on 44a6e9fa). Two slots (2.17 M pairs/s at k = 8) already
recover part of that by overlap. The order of the levers is:
1. hold the slot only around device work (§4: ~7.9 s on the E16 head; one slot → ~2.4 M pairs/s per GPU, and less host work under a
   slot for two slots to hide);
2. a persistent per-GPU process with the index, the non-host index and a context resident (~6 s more, → ~3 M pairs/s with one slot);
3. a slot order that keeps the device busy at small k (§5b), so that the limit is reached with the samples host memory allows;
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
- **Two slots went from nothing to +38 % with E17** (§5c): with E15's hand-made split they gained 0–8 %, because the split made a
  prefill alone 9 s slower; with E17's budget and nothing else set, 2.17 M pairs/s per GPU at k = 8, past the one-slot limit. Two
  prefills at once still time-slice (35–36 s each against 17.5 s alone): the gain is overlap, not parallel kernels.
- **Unpinned samples wander between the GPUs** (job E): without `SINGLET_ENGINE_GPU_DEVICE`, each phase takes the first free slot of
  any GPU, so 12 of 32 samples ran their prefill and non-host screen on different GPUs, 4 of 32 lost the GPU in the non-host screen
  (`resource already mapped`, `device(s) is/are busy or unavailable`) and screened on the CPU, and the node ran 7 % slower than
  pinned. The outputs stayed identical.
- **A pipeline bug, not an engine one:** nuclei2k_v2's `nonhost_reads.fq` differs in content from run to run (§3); the writer reads
  the unmapped file again and indexes it with the screen's subsample.
- **More threads per sample shorten the slot hold:** the prefill's host lane runs on T − 1 threads beside the device lanes and its
  wait after the lanes falls from 1.79 s at T = 7 to 0.07 s at T = 14; the records' assembly too. Since the cores are mostly
  idle, oversubscribing T costs nothing.
- **Each sample keeps a private 16.8 GB copy of the genome** in host memory (`Shared memory is not used for genomes`), so host
  memory, not the device, bounds k on a node.

## 10. What remains

In order of what they buy per GPU (pbmc1k_v3; §4 and §5):
1. **Hold the slot only around device work** (engine, not E18's scope): release after lane B's download and take the slot again for
   `e2e.build` (−5.7 s); reset and unlock right after `pileupRun`, before the puts and the host pass (−2.2 s on the E16 head);
   read the non-host index into page-locked memory before taking the slot. On the E16 head: 36.1 → ~28 s held, one slot's limit
   1.84 → ~2.4 M pairs/s per GPU; with two slots, less host work under a slot to hide.
2. **A priority in `lockDevice`:** later phases before new prefills. Reaches the slot's limit at k = 5–6 instead of 10–12 in the model
   (§5b), so fewer samples in flight (host memory) for the same throughput.
3. **A persistent per-GPU process** (E14's recommendation: preload once, fork per sample): index, non-host index and a context
   resident; no per-sample context create/reset, no 16.8 GB genome copy per sample. ~22 s held → ~3 M pairs/s per GPU and 2–3x more
   samples in flight per node by host memory. At ~9 core-s per M pairs (node, §2) 54 cores carry ~6 M pairs/s, so past ~3 M pairs/s
   per GPU the host work per read pair becomes the next bound.
4. **Two slots at larger k and with the levers above:** k = 8 per GPU is where host memory stops a half node; one shared genome per
   node (item 3) would allow more. The two-slot limit (~2.8 M pairs/s per GPU with both slots always held) is not reached at k = 8.
5. **Keep an unpinned process on one GPU** (engine): `lockDevice` without a pin could prefer the device of the process's earlier
   phases, or the campaign's launcher pins every worker (§6), as job E's pinned points did.
6. **`nonhost_reads.fq` (pipeline):** write the reads the screen classified (§3), so that the file is the same in every run.
7. **Not measured here:** the mixed set with two slots and at node scale; A100; the input supply (download/convert) that II.1
   expects to bind the campaign at ~1 GB/s.

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
- f68bfba9, e49211df, 7155e6c6, 3a22c200: this document's drafts.
- 5e55b702: merge of gpu/fable-engine 503c07f7 (E17 "budget").
- This document is the next commit.

**Builds** (short-gpu without a GPU, ~5 min):

| job | commit | engine | output (`$W/jobs/fable/throughput/`) | sha256 |
|---|---|---|---|---|
| 797401 | 7a2bbb9e | 44a6e9fa (E1–E15) | `bin/singlet-gpu` | 51e0b035… |
| 797455 | 3b93a792 | c6ddd3d3 (+ E16) | `bin-e16/singlet-gpu` | 28fe749c… |
| 797476 | 5e55b702 | 503c07f7 (+ E17) | `bin-e17/singlet-gpu` | 23c4c79a… |

**Runs** (short-gpu, H100 NVL, `--nice=10000`):

| job | node | GPUs / cores / mem | binary | points | notes |
|---|---|---|---|---|---|
| 797417 | g051 | debug-gpu, 1 / 8 | bin | pbmc2m `k1t4s1 k2t4s1` | script debugging; k = 2 identical to k = 1, dumps identical |
| 797423 (A) | g050 | 1 / 28 / 370 GB | bin | `k1t28s1 k1t14s1 k1t9s1 k1t7s1 k2t14s1 k3t9s1 k4t7s1 k4t14s1 k1t14s1m43 k2t14s2m43 k3t9s2m43 k4t7s2m43` | 43:43 |
| 797443 (B) | g050 | 1 / 28 / 370 GB | bin | `k1t7s1f k4t7s1r3 k6t7s1r2` | cancelled after `k6t7s1r2` when E16 landed (its mixed-set points are void); a build ran beside it on the same node during `k6t7s1r2` |
| 797460 (C) | g051 | 1 / 28 / 370 GB | bin-e16 | `k1t7s1f k1t9s1 k3t9s1 k6t7s1r2 k8t7s1r2 mixk1t7s1 mixk6t7s1r2` | E16 head; 1:00:17; E17's gate jobs on the node's other GPU |
| 797485 (D) | g051 | 1 / 28 / 370 GB | bin-e17 | `k1t7s1f k1t9s1 k1t14s1 k2t14s1 k2t14s2 k3t9s1 k3t9s2 k3t9s3 k4t7s1 k4t7s2` | E17 head: the k = 3 re-run, the slot points with the budget's defaults; 32:16 |
| 797491 (F) | g051 | 1 / 28 / 360 GB | bin-e17 | `k1t7s1 k6t7s2r2 k8t7s2r2 k6t7s1r2` | E17 head: two slots in steady state; 33:18 |
| 797490 | – | 2 / 54 / 730 GB | bin-e17 | – | the node job, cancelled while pending (no whole node free) to run job F first |
| 798028 (E) | g052 | 2 / 54 / 730 GB | bin-e17 | `k1t7s1p k16t7s2r2p k12t7s2r2p k16t7s2r2 k12t7s1r2p` | the node; 51:11; waited 6 min for a whole node |

**Results:** `$W/jobs/fable/throughput/results/run-<job>/`: `summary.txt` (everything below in one file), `table.md`, `dumps.tsv`,
and per point `point.txt` / `point.json` (analysis), `locks.json`, `samples.jsonl` (the sampler), `runs.tsv`, `meta.json`,
`identity.txt`, and per run `c<i>/` (`time.txt`, `run.log.ts`, `star_Log.out`, `star_Log.final.out`, `CellReads.stats`,
`read_stats.tsv`, `summary.json`, `pileup_stats.json`, `identity.txt`). Logs: `$W/logs/e18-{build,run*}-<job>.out`.

**Run:**
```
sbatch --export=ALL,OUTD=$W/jobs/fable/throughput/bin-e17 bench/fable/throughput/jobs/build.sbatch
sbatch --mem=360G --export=ALL,BINDIR=$W/jobs/fable/throughput/bin-e17,POINTS="k1t7s1 k8t7s2r2" bench/fable/throughput/jobs/run.sbatch
sbatch --gres=gpu:nvidia_h100_nvl:2 -c 54 --mem=730G   --export=ALL,BINDIR=$W/jobs/fable/throughput/bin-e17,POINTS="k1t7s1p k16t7s2r2p" bench/fable/throughput/jobs/run.sbatch
```
(The node's allocatable memory is 735 GB: `RealMemory` less `MemSpecLimit`; `--mem=740G` is refused.) Analysis on a point
directory: `e18_point.py <dir>`, `e18_locks.py <dir>`, `e18_steady.py <dir>`, `e18_trace.py <dir> <from> <to> [item]`;
`e18_point.py --table <run dir>`. The model of §5b: `e18_sim.py [fifo|random|late|early] [GPUs] [e15|e16]`.
