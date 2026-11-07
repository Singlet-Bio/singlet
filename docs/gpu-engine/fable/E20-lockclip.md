# E20: the device lock around device work only, and STAR's CellRanger4 clip on the device (lockclip)

Experiment E20 ("lockclip") of the GPU aligner program. Branch `gpu/fable/lockclip`, from `gpu/fable-engine` at 78c46552 (E1–E19
merged). It takes the three largest pieces of the GPU path's own host work that E18 (`E18-throughput.md` §4) and II.14 of the north
star named:
1. the device lock held through host work (the prefill's host lane, record assembly and `readRecord`; the pileup's host pass and
   E16's reset);
2. STAR's CellRanger4 (CR4) clip, E13's phase C, on the host once per distinct sequence;
3. the CR4 probe, two STAR children on the CPU.

Everything runs level 5 + ingest (`--engine gpu`, `SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1`), the
default of E19. The CPU engine and the default GPU path (`SINGLET_ENGINE_GPU_E2E=0`) run the code they ran before.

## 0. Answer in brief

E20 is built and measured. It is not gated on all samples and it carries one open risk.

**Takeaway:** E20 shortens the time a sample holds the device lock. Core-seconds per M read pairs fell in one calibration job and stayed flat in another. One of 16 loaded runs fell back to the CPU after a device out-of-memory error.

| Check | Result | Job |
|---|---|---|
| Tier 1, alone, ABBA, g051 | core-s per M 8.94 to 7.79 (-12.8%). Lock held 39.1 s to 27.6 s (-29.4%). Wall 152.9 s to 144.2 s | 800360 |
| Tier 2, 24 runs per leg, stopped after leg 2 | core-s per M 9.89 to 9.92 (flat). Lock held 45.0 s to 36.4 s. Progress rate 1.809 to 2.098 M pairs/s. One device out-of-memory fallback in 16 loaded runs. Verdict REJECT by the tier 2 rule | 800365 |
| Identity against the CPU engine | pbmc1k_v3, neuron1k_v3 and pbmc5p_v1 identical. The other 7 of the 10 gate samples were never run | see §4 |

The decision rests on core-seconds per M and lock held, not on rate. Rates in this report come from "steady" windows that the drain of the queue biases upward.
PR-2 on `perf/gpu-next` carries E20 together with WSHRINK. WSHRINK shrinks lane B memory per read and may remove the out-of-memory fallback. That claim waits for the PR-2 gates.

## 1. The lock

Five phases of a sample take a slot of the device (`lockDevice` / `lockDeviceSlot` in `include/singlet/engine_gpu.h`: an exclusive
`flock` on `/tmp/singlet_engine_gpu_<pci>[.k].lock`). E18 §4 measured 36.1 s held per pbmc1k_v3 sample on the E16 head at T = 7, of which
~7.9 s was host work with the device idle. E20 changes two of the five holds and checks a third.

### 1a. The prefill: two holds, the host work between them (`engine_gpu.cu`, `SeqCacheGpuPrefill.cpp`)

Before, `mapDistinct` held the device from the index upload to the end of E8's `e2e.build`, through the host lane's wait, the host
assembly of the records (`records.prep`, `records.copy`), `readRecord` and E10's host pass over the entries. Now, when E8's records are
built (`in.e2e`) and the caller gives a way to take the device again (`Input::reacquire`):

1. **Hold 1** (unchanged device work): context, index and tables, reads, the clip (§2), E10's entries and masks, the dedup, lanes A,
   M and B. Two things move so that nothing but E11's keep is left on the device afterwards:
   - E10's entries and masks go into E8's shared region right after the clip (`entToRegion`, as E17's lean mode does), not at the
     end;
   - the lanes do not keep their packed results on the device (`resident = false`), so E8's records are later uploaded from the host
     assembly instead of gathered on the device (E17's lean path, `gatherDev = false`).
   Then the index, the lanes' buffers, the reads and the tables are freed (only `dcs`, `buildRecords`' chromosome starts, and E11's
   keep stay), `cudaDeviceSynchronize`, and `in.release()` unlocks.
2. **Unlocked:** the host's copy of the clip lengths (`clip5` / `clip3` per read, which STAR's own clip of the reads without a
   record reads after `mapDistinct`; `clipHostFill`, 0.2–0.5 s on 18 threads, ran in the clip step before), the host lane's wait
   (`hostAll`), the host assembly of the records, `readRecord` (`fillReadRecord`), and E10's host pass over the region's entries
   (zero the entries of reads without a record, list the hook reads; `entHostPass`, moved out of `e2eBuild` so that it can run
   here).
3. **Hold 2:** `in.reacquire()` (`lockDeviceSlot` on the same device: E11's keep and this process's context are there), then
   `e2e.build` (staged upload of the host-assembled records, `buildRecords`, the records into the region, E10's products, E11's keep
   of the records) and the frees.

`star_Log.out` reports both holds:
`ENGINE_GPU: device 0 (...), waited 0 s for it, held it 13.87 s (E20 split: 12.67 s to the lanes' end, released 2.37 s for the host
lane and the records, waited 0.00004 s of it to take it again, 1.20 s for the records on the device)`. "held" is the sum of the two
holds, so E18's `e18_point.py` reads the new total unchanged; `e18_locks.py` labels a sixth sampler interval `prefill2`.
`SINGLET_ENGINE_GPU_SPLIT=0` gives the single hold of the head.

### 1b. The pileup: unlocked right after `pileupRun` (`include/singlet/pileup/pileup_engine.h`)

Since E16's `resetbg`, the pileup's context (with the input's page-locked memory) was reset on the "free2" thread at the very end,
which then unlocked the device: the slot stayed held through the accumulator puts and the host pass (E18: 2.2 s). Now a thread
starts as soon as `pileupRun` returns, beside the puts and the host pass (which read only host memory: `PileupOutput`'s vectors and
the input's host arrays): it unlocks the device, then runs `pileupRelease` (`cudaDeviceReset`, which also unpins the input). The
thread is joined after the host pass; the input is freed after the join, as before.

The unlock goes before the reset because by then the reset is host work: `pileupRun` has freed its arena and fetched its results,
and what is left is the context's teardown and unpinning the input (0.6–0.8 s on pbmc1k_v3). The next holder only creates its own
context meanwhile; the one thing it can see is this process's last ~1–2 GB on the device for under a second (the prefill's index
upload, its first step, needs 17 GB of the 94). `SINGLET_GPU_PILEUP_UNLOCK=reset` gives reset-then-unlock (the order of c849c839),
`=late` E16's order. The log line now reports the real hold:
`[gpu_pileup] device lock: waited 0.3 s, held 2.3 s (E20: pileupRun 2.3 s; the device reset 0.77 s on a thread beside the puts and
the host pass, after the unlock unless SINGLET_GPU_PILEUP_UNLOCK=reset)`.

### 1c. The non-host screen and E11's Solo: unchanged

- **Non-host** (`src/pipeline/singlet.cpp` ~1697): the lock covers `nonhostClassify` only, which is the context, the index and
  Bloom filter from the page cache to the device, the reads' upload, the classify kernels, the results' download and
  `cudaDeviceReset`. The host EM input (`classify_from_groups`) runs after the unlock. Nothing host-only is under the lock; the
  index load (~3.6 s of the 4.4 s) is a file-to-device copy.
- **Solo** (`seqCacheGpuAfterMapping`, E11): `lockDeviceSlot` around `soloRunKept` only, 0.6–1.0 s.

### 1d. The hold per phase

The logged examples in §1a to §1c give these holds. They come from single runs. The per-run hold in §5 is the measured total.

| Phase | Hold | Source |
|---|---|---|
| Prefill, both holds | 13.87 s (12.67 s to the lanes' end, 1.20 s for the records on the device, 2.37 s released between them) | log line quoted in §1a |
| Pileup | 2.3 s (`pileupRun`). The device reset of 0.77 s runs after the unlock | log line quoted in §1b |
| Non-host screen | 4.4 s, of which about 3.6 s is the index load | §1c |
| Solo (E11) | 0.6 to 1.0 s | §1c |

Before E20 the E16 head held the lock 36.1 s per pbmc1k_v3 sample at T = 7. About 7.9 s of that was host work with the device idle (E18 §4).


## 2. The CellRanger4 clip on the device

### 2a. What it replaces

E13's phase C (`cr4Clip`, `src/star/SeqCacheGpuPrefill.cpp`) runs STAR's own `ClipMate` / `ClipCR4` code on the host: opal's
overlap alignment of the 10x TSO against the read's first 91 bases, 64 sequences at a time, then `ClipMate::clip` for the 5' mate
(type 10) and the 3' mate (type 11, polyA), once per distinct cDNA sequence (31.9 M of 66.6 M on pbmc1k_v3). It started on a thread
of STAR's process right after the fork (`startEarly`'s `th0`), beside the CR4 probe: 70–77 thread-s, 6–8 s at T = 18, 11+ s at
T = 7. Its products are the engine's `pre5` / `pre3` (bases clipped at each end per read) and `cr4info` (the 5' clip's opal result,
kept for STAR's own clip of the reads STAR maps itself).

The engine's `k_clip` never ran opal: since 94210b6e it applies the host's `pre5` / `pre3` spans (and the generic N / 3' adapter
rules) on the device. So the device clip is a new kernel, `k_cr4`.

### 2b. `k_cr4` (`src/engine_gpu/engine_gpu.cu`)

One thread per read, over the packed codes already on the device (`dSeq`, `dOff`), before `k_clip`. No dedup is needed: a read's
clip depends on its own bases and length only (E13 checked that opal's batch does not matter), and the device does 66.6 M reads in
0.093 s.

- **opal OV, exactly, in integer registers.** Target: the read's first 91 codes, padded with 4 (N) as `opalFillOneSeq` does; query:
  the adapter's numeric codes (30 for the TSO; the 30-base case is a template with its rows unrolled, 1–40 otherwise). Score matrix as
  `ClipCR4`'s: match 1, N/N 0, anything else −2; gap open 2, extend 2. The column recurrence is opal's (`prevH` 0 and `prevE` −126 per
  query row at the start, per column `uF` −126 and `uH` = `ulH` = 0). Opal's SIMD lanes saturate at 8 bits; with a query of at most
  40 and these scores no cell leaves [−126, 40], so plain `int` arithmetic gives the same values. Result as opal's
  `OPAL_SEARCH_SCORE_END` in mode OV: score = max(last column's max, last row's max); the target end is 90 if the last column's max
  beats the last row's max before the last column, else the first column where the last row's max strictly rose.
- **`ClipCR4`'s decision:** `Lm = endT + 1`; no clip if `score < 20`, or `score == 20 && Lm > 26`, or `score == 21 && Lm > 30`;
  `info = no clip ? 0 : Lm`.
- **`ClipMate::clip`, type 10 (5'):** the N bases (no buffer move for this type), the adapter (`min(info, Lread)`, which moves the
  buffer), `NafterAd`; a mate clipped to nothing counts its whole length.
- **type 11 (3'):** the N bases, then `polyTail3p` over the first `Lr` bases of the moved buffer (the same scan as
  `ClipCR4::polyTail3p`: +1 per A, −2 otherwise, kept while `10·score ≥ 7·length`, stop when `length − score > 27`, at least 20),
  then `NafterAd`.
- Only reads of at most 650 bases (`DEF_readSeqLengthMax`) are clipped, as in `cr4Clip`; `info` is computed for every read.
  `pre5` / `pre3` are capped at 65,535.

`info` (1 B per read, 66.6 MB) comes back to the host (0.06 s) for STAR's own clip of the reads without a record. The clip's
lengths stay on the device for `k_clip`, E10's entries and the dedup of the clipped sequences.

### 2c. Where it is decided (`src/star/SeqCacheGpuPrefill.cpp`)

- `ingPrepare` (STAR's process, after its parameters) builds STAR's ClipMates and takes the device clip when
  `SINGLET_ENGINE_GPU_CR4_DEVICE` is not 0 and they are CellRanger4's (`cr4DevFits`: type 10 with an adapter of 1–40 bases in A, C,
  G, T, N; type 11 polyA). The adapter, `N` and `NafterAd` of both mates go into `Input::clip` (`dev`, `cr4q`, `cr4n5`, ...).
- `startEarly`'s `th0` no longer clips on the host when the device will.
- Otherwise (`SINGLET_ENGINE_GPU_CR4_DEVICE=0`, or ClipMates the device does not take) E13's host clip runs as before.
- `SINGLET_ENGINE_GPU_CR4_CHECK=1` runs both: the host clip as before, and `k_cr4`, whose three arrays are compared byte for byte
  with the host's; the engine logs `E20 CellRanger4 clip check: ... IDENTICAL/DIFFERENT (5' n, 3' n, info n reads differ, first
  read ...)` and uses the host's.
- E13's log line says where the clip ran: `CellRanger4 clip 0 s (on the device: E20, in mapDistinct)`.

### 2d. Identity of the clip

The device clip matches the host clip on the 92.9 M reads of neuron1k_v3. No other sample has a recorded clip comparison. `SINGLET_ENGINE_GPU_CR4_CHECK=1` runs both clips and compares `pre5`, `pre3` and `info` byte for byte. The PR-2 gate runs the 10-sample gate with this switch on.


## 3. The CR4 probe: kept on the host

**Not moved.** The brief allowed keeping the host probe when the device version cannot share the index or reproduce the rates; it
cannot do the first without a restructuring of the STAR start that is larger than the probe's cost, and the second needs STAR
itself for part of the reads and for files the gate compares. Precisely:

1. **The index is not on the device when the probe runs, and the order cannot be swapped.** The probe runs in the pipeline's STAR
   child before `star_main_impl` (`src/pipeline/singlet.cpp` ~7854): its decision removes or keeps the four profile arguments in the
   main STAR's argv (`args.erase(cr4_beg, cr4_end)`). The device index is uploaded inside that main STAR, by the prefill, after
   STAR has built its `Parameters` from that argv, loaded the genome and the transcriptome. Before that the STAR child has no
   `Parameters`, no `Genome`, no CUDA context. So a device probe is one of:
   - **the probe children on the device:** each child uploads the 17.6 GB index itself (E12: 2.6 s at 7 GB/s) under the device
     lock, two more uploads per sample. At one slot that adds ~5–6 s to each sample's hold (the probes' mapping itself is ~0.1 s
     of kernels), which costs more throughput than E20's lock work gains (§1d), to save 19–27 core-s per sample (point 3);
   - **the probe inside the main STAR** after its index upload: the main STAR would start with the profile assumed on, build a
     second `Parameters` for the plain run, run `mapDistinct` on the 200 k reads with each, and restart itself (a new `star_main_impl`
     with the plain argv) when the decision is off. Feasible, but it moves the probe's decision, `cr4_probe.json` and the profile's
     argv from the pipeline into STAR's prefill, and needs everything in point 2.
2. **Records alone do not give the probe's outputs.**
   - **The reads without a record.** The engine leaves reads with N and host-lane failures to STAR (E10/E11: 0.24 % of the reads;
     ~470 of 200 k). They count in "Number of input reads" and some map uniquely, so "Uniquely mapped reads %" to STAR's two
     decimals needs them mapped by STAR with each parameter set (a `ReadAlignChunk` per `Parameters`).
   - **The probes' `SJ.out.tab`.** `cr4_probe/plain_SJ.out.tab` and `cr4_SJ.out.tab` are among the 60 files the gate compares. E10
     gives the junction counts from records, but the file is STAR's collapse and `outputSJ` filter (`outSJfilter*` against the
     annotated junctions) with each `Parameters`, i.e. STAR's own output step run twice in one process.
   - The two `Log.final.out` (skipped by `out_equiv.py` but read by the decision) would be written from E10's counters.
3. **What it would save is small.** With the genome preloaded once per sample (E14), the two probes are mostly STAR start-up (parameters,
   whitelist, transcriptome, Solo setup: ~2 s each) and ~2 s of mapping 200 k reads on 9 threads each. Measured (the new `wait4`
   field, job 798106, T = 18, 9 + 9 threads, concurrent): pbmc1k_v3 plain 5.0 s / 14 core-s and cr4 5.5 s / 13 core-s (gpuT), 4.1 s /
   9.4 and 4.2 s / 9.4 (gpuO0, warm caches); neuron1k_v3 4.8 s / 11 and 5.3 s / 11. Per sample 19–27 core-s, 0.3–0.4 core-s per M
   pairs, off the device's critical path (the prefill starts after them, but the stream overlaps them with other samples' work).
   Since the CR4 clip no longer runs beside them, the probes alone are the start stage's work (§5c).

The probe and the CPU engine's probe are untouched; `cr4_probe.json` is byte-identical in every gate (§4). The one probe change is a
log field: the probes' CPU from `wait4` on the timing line (`CPU (user + system) plain X s, cr4 Y s`), to measure point 3. The
natural home of a device probe is E21's resident per-GPU process, which keeps the index on the device across samples (§7).


## 4. Identity

- pbmc1k_v3, neuron1k_v3 and pbmc5p_v1 are identical to the CPU engine.
- The default path (`SINGLET_ENGINE_GPU_E2E=0`) is identical to E11's gpuD.
- The four mapDistinct dumps of the stream runs match.
- Overnight job 800576 reported identity on 4 of 4 runs. Its summary file is not available, so this line is unreproduced.
- The integration build `gpu/bold-integ` (E20, WSHRINK and the strand fix) differs from b0 only in a provenance path in job 800821, on pbmc1k_v3.
- **Not gated:** the other 7 of the 10 gate samples. All earlier gates ran one slot. Identity at 2 and 3 slots is a PR-2 gate.

## 5. Throughput and core-s

pbmc1k_v3, one slot, T = 7, Fable job 798129 against f73938b2:

| Run | M pairs/s | core-s per M | Lock held per run |
|---|---|---|---|
| f73938b2, k = 1 | 0.532 | 8.73 | 35.2 s |
| E20, k = 1 | 0.555 | 7.50 | 26.2 s |
| f73938b2, k = 8, steady | 1.599 | 8.11 | 31.9 s |
| E20, k = 8, steady | 2.177 | 6.90 | 24.4 s |

The "steady" rates carry the drain bias. The calibration jobs on g051 give a smaller gain in core-seconds per M than job 798129 (tier 1 -12.8%, tier 2 flat). The jobs disagree on core-seconds per M, so the record asks for the range and not a single value. Host peak memory fell from 349 GB to 298 GB in tier 2. Device headroom with all slots held is 2.2 GB for E20 and 3.4 GB for s3 (e42c0636).

The node target of 5 M pairs/s was never measured on this branch. The one-slot target of 2.0 M pairs/s at k = 8 is met in job 798129 and the core-seconds target of 6.5 per M is not met (6.90).

## 6. Surprises

- The tier 2 run (job 800365) showed one device out-of-memory fallback in 16 loaded runs. The failing allocation was `dmal(&dBits, bitsCap * 4)`. That run used 4.5 times the core-seconds of a normal run. The earlier headline rate hid this CPU fallback.
- The calibration jobs disagree. Tier 1 gives -12.8% core-s per M and tier 2 gives no change (9.89 to 9.92). The hold fell in both.
- The handoff row for e20w omitted the fallback risk.

## 7. What remains

1. Gate the 7 samples that never ran, with `SINGLET_ENGINE_GPU_CR4_CHECK=1`.
2. Run identity at 2 and 3 slots, with MPS at k = 8.
3. Test whether WSHRINK removes the device out-of-memory fallback.
4. Run E20 with E21. They touch the same start path and have never run together.
5. Move the CR4 probe to the device. §3 explains why it belongs with E21's resident process.

## 8. Files touched, and why

| File | Change |
|---|---|
| `src/engine_gpu/engine_gpu.cu` | `k_cr4`, the two prefill holds, E10 entries into the shared region |
| `include/singlet/engine_gpu.h` | `Input::reacquire`, the split of the hold |
| `src/star/SeqCacheGpuPrefill.cpp` | the device-clip decision, the clip check, the log lines |
| `include/singlet/pileup/pileup_engine.h` | the pileup's unlock right after `pileupRun`, the reset on a thread |
| `src/pipeline/singlet.cpp` | the probe's CPU time from `wait4` in the log |
| `bench/fable/lockclip/` | build, gate and run jobs, `e20_start.py` |
| `bench/fable/throughput/e18_locks.py` | a sixth sampler interval, `prefill2` |

## 9. Jobs, commits, how to run

| Job | Purpose |
|---|---|
| 798129 | one slot, T = 7, E20 against f73938b2 |
| 800360 | calibration, tier 1, ABBA, g051 |
| 800365 | calibration, tier 2, stopped after leg 2 |
| 800576 | overnight rerun, summary not available |
| 800821 | identity of the integration build with WSHRINK and the strand fix |

Code heads: `de5d85aa` (E20), `aeef7a5e` (`perf/gpu-next-e20`, the e20w binary). Job scripts are in `bench/fable/lockclip/jobs/`. The gate and timing commands for PR-2 are in the pull request description.
