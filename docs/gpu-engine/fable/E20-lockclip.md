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

[[ANSWER]]

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

[[LOCKTABLE]]


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

[[CLIPID]]


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

[[IDENTITY]]

## 5. Throughput and core-s

[[THROUGHPUT]]

## 6. Surprises

[[SURPRISES]]

## 7. What remains

[[REMAINS]]

## 8. Files touched, and why

[[FILES]]

## 9. Jobs, commits, how to run

[[JOBS]]
