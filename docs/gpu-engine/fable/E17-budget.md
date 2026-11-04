# E17: the prefill's device memory budget and 64-bit junction keys (budget)

Experiment E17 ("budget") of the GPU aligner program. Branch `gpu/fable/budget`, from `gpu/fable-engine` at 44a6e9fa (the E15 report),
merged with `gpu/fable-engine` at e0e1514a (E16 "tail") before the final gates. Goal: the GPU path runs every genome, holds a given
device memory budget, and stays byte-identical to the CPU engine. Three parts:
1. a device memory budget for the prefill (`SINGLET_ENGINE_GPU_MEM_GB`; default (device memory − 2 GB) / `SINGLET_ENGINE_GPU_SLOTS`),
   from which the chunk, the lane batches, lane B and the pileup's arena are sized, instead of "all the free memory";
2. 64-bit junction starts in E10's sort key, so that genomes above 2^32 bases (the 5.8 G-base human + mouse barnyard) run at level 5;
3. (optional) STAR's CellRanger4 clip on the device: not done (§7).

GB in this report: 2^30 bytes, as the engine's memory marks and `nvidia-smi`'s MiB (the H100 NVL has 93.1 GB of them).

## 0. Answer in brief

- **The prefill now sizes itself from a device memory budget, and keeps to it.**
  - The budget is `SINGLET_ENGINE_GPU_MEM_GB`, or by default (device memory − 2 GB) / `SINGLET_ENGINE_GPU_SLOTS`: 91.1 GB on the H100 NVL
    with one slot, 45.5 GB with two, 30.4 GB with three.
  - The chunk, the three lanes' batches, lane M's warp-or-thread choice and lane B's batch are planned from the room it leaves. Lane B
    no longer takes "the free memory less 3 GB". Below about 35 GB on pbmc1k_v3 a lean mode keeps only the index and one chunk on the
    device. The pileup's arena is budget − 3 GB.
  - Every device allocation of the prefill is in a tally. It includes lane M's 4 GB warp stack, which the driver reserves for the whole
    device. The tally is within 0.2 GB of the device after every lane of every run.
  - The 1 s `nvidia-smi` peak of the prefill stays under the budget in every run (pbmc1k_v3; budget → peak in GB):
    91.1 → 88.6, 80 → 77.5, 45.5 → 43.8, 40 → 38.0, 38 → 36.2, 30.4 → 29.4, 24 → 23.2.
  - A sample that cannot fit stops before allocating, with the reason: "the index (52.73 GB) and the smallest lanes need 54.84 GB of
    device memory, more than the budget of 40.00 GB ...". `SINGLET_ENGINE_GPU_MEM_FALLBACK=cpu` lets STAR map every read instead.
  - E12's `SINGLET_ENGINE_GPU_NB/_NBM/_NBB/_CHUNK` remain overrides.
  - Two slots now take 45.5 GB each (prefill peak 43.8 GB). Three slots fit too: 30.4 GB each in lean mode (prefill peak 29.4 GB).
- **hgmm1k_v3 (human + mouse barnyard, 5.8 G bases) runs at level 5, identical to the CPU engine (gate (a), 797468).**
  - The junction sort key is `start << gb | gap`, with gb = 32 below 2^32 bases (E10's key, unchanged) and 31 on the barnyard.
  - out_equiv: 49 identical, 2 differ only in timings or paths, and the same 7 run-dependent files as E15. The 10 STAR Solo files are
    identical. `Gene/CellReads.stats` has the same 524,497 rows. `SJ.out.tab` is identical (171,869 junctions), and so are the mapping
    statistics. `nonhost_reads.fq` and `Unmapped.out.mate1/2` have the same records.
  - Level 5 + ingest takes **88.7 s / 437 core-s, against 374.9 s / 1,609 core-s for the CPU engine** (4.2x wall, 3.7x core-s). The
    default GPU path also runs now: 176.9 s. In E15 both failed at lane A's allocation and fell back to the CPU.
  - At the full budget the plan is 2 chunks of 16 M, with NB 524,288 / 311,296 / 52,224. The engine's peak is 89.3 GB.
- **Gate (b), pbmc1k_v3 at level 5 + ingest under 80 / 40 / 24 GB budgets and the full budget (797462, merged with E16): every run is
  identical to E11's kept CPU engine outputs.**
  - Each run has 60 identical files, the 10 Solo files identical, the same 323,542 CellReads.stats rows, `SJ.out.tab` identical
    (126,996 junctions) and the mapping statistics identical.
  - So does a 24 GB run with a forced 16 M chunk, where lanes A and M run in parts.
  - The differing files are the head's own run-dependent ones.
- **The full budget is not slower than the head:** 103.0 / 103.2 s against E16's head at 103.2 / 104.5 s, in the same job on the
  same node (E16's own gate: 102.8 s).
  - The prefill's device peak drops 90.4 → 88.6 GB.
  - Smaller budgets cost time: +2.0 s at 80 GB, +5.3 s at 40 GB, and +18.3 s at 24 GB. At 24 GB lane B's batches shrink to 11 k reads
    because of its 550 KB per read in flight.
- **Gate (c): the default GPU path is unchanged** (797468). Against E11's kept gpuD it has 60 identical files. The only differences
  are timings, paths and the 3 files missing from the kept copy. `SJ.out.tab` and the mapping statistics are identical, and
  CellReads.stats has the same rows.
- **Not inside the budget:**
  - The non-host screen is a fixed extra of 25–29 GB (its 24.9 GB index; §1e).
  - The device lock is untouched. It is held for 16.1 s at the full budget and 32.3 s at 24 GB, and the host assembly of the records
    runs under it. This is E18's lever (§7).
- **Not done:** the optional CellRanger4 clip on the device.
- **Fixed on the way:** a budget failure crashed the process (`fail()` reset the device under E12's staging buffers; §6.1).

## 1. The budget

### 1a. What it bounds

`singlet_engine_gpu::memBudget(total)` (`include/singlet/engine_gpu.h`) is one sample's device memory, in bytes:
- `SINGLET_ENGINE_GPU_MEM_GB=<GB>` when set (`0`: no budget, E12's behaviour: lane B and the pileup's arena take the free memory less 3 GB);
- else (device memory − 2 GB) / `SINGLET_ENGINE_GPU_SLOTS` (slots 1–8, default 1): 91.1 GB on the H100 NVL with one slot, 45.5 GB with
  two, 30.4 GB with three.

It bounds what the sample's processes hold on the device, phase by phase:
- **the prefill** (`mapDistinct`, `src/engine_gpu/engine_gpu.cu`): the context, the index, the reads and their clip, E10's aux, entries
  and masks, the dedup, the lanes A / M / B with their result slots and packed results, E8's records, E10's products and E11's keep. The
  chunk and the batches are derived from it (§1c), and each phase is checked against it before it allocates.
- **the pileup** (`pileupRun`, `src/engine_gpu/pileup_gpu.cu`): its arena is min(free − 3 GB, budget − 3 GB) (the 3 GB are this process's
  context, STAR's and E11's keep, which STAR's process may still hold). `SINGLET_GPU_PILEUP_ARENA_GB` still caps it.
- **not** the non-host screen: its index is a fixed 24.9 GB (§1e).

The default budget is also never more than the free device memory less 2 GB when the lanes are planned (another process's memory is not
taken from it). With `SINGLET_ENGINE_GPU_MEM_GB` set, the budget is taken as given: the caller (E18's scheduler) guarantees it.

### 1b. The tally

Every device allocation of `mapDistinct` and `runLane` goes through `dmal` / `dfree`, which keep a per-pointer size (rounded to the
driver's 2 MB granularity) and a running total; thrust's temporaries go through the same allocator (`TP` = `thrust::cuda::par(TAlloc)`).
The tally starts at the context (the device memory in use at the start, at most 1 GB) and also carries:
- E11's keep (the reads' CR/UR, 16 bytes per read, then 4 per read and 8 per record), which `solo_gpu.cu` allocates itself;
- lane M's warp stack. The warp lane raises `cudaLimitStackSize` to `sizeof(LWM)` + 8 KB per thread, and the driver reserves it at once
  for every thread the device can hold: 4,050 MiB on the H100 NVL (132 SMs × 2,048 threads × ~15 KB). The engine measures it
  (`cudaMemGetInfo` around `cudaDeviceSetLimit`) and logs it with the lane. E12's sizing never saw it: it only shows as free memory that
  drops when lane M starts.

What the tally does not see (modules' temporaries in E8 / E10 / E11, the driver's local memory for the kernels) is covered by a fixed
slack of 1 GB. Measured against the device (`cudaMemGetInfo` after each lane's kernels, the "device X GB in use, tally Y GB" of the lane
log): the tally is within 0.2 GB of the device on every lane of every run. The engine's own mark is the tally's peak plus
0.10–0.18 GB, and the 1 s `nvidia-smi` peak never exceeds that mark by more than 0.05 GB (it can miss a short peak; §3).

### 1c. The plan

Right after the context exists, `mapDistinct` estimates two modes from the sample's sizes (logged as `engine_gpu: E17 budget ...`):
- **normal** (E12's order): the index uploaded on a second thread during the reads' upload, clip and dedup; E10's entries and masks and
  E11's keep on the device until the records; the distinct sequences compacted on the device; the lanes keep their packed results on the
  device for E8's gather. Estimate: context + tables + slack + index + the larger of the reads' phase and (reads + entries + keep + the
  smallest lanes). pbmc1k_v3: 34.8 GB; hgmm1k_v3: 70.3 GB.
- **lean**, when the normal estimate exceeds the budget: the reads' phase runs without the index; E10's entries and masks go to the shared
  region as soon as they exist (the host zeroes those of the reads without a record at the records and lists the hook's reads, the rules
  of `products()`); after the dedup only the index stays on the device (each chunk uploads its own sequences, compacted once on the
  host); E11's keep is made at the records (the aux uploaded again); the lanes' results go to the host and E8's records are uploaded from
  the host assembly (the same bytes as the gather: `SINGLET_ENGINE_GPU_E2E_CHECK`). Estimate: context + tables + slack + the larger of
  the reads' phase and (index + the smallest lanes). pbmc1k_v3: 18.5 GB; hgmm1k_v3: 54.9 GB.

A sample that does not fit even the lean estimate stops before any allocation (§1f).

Then, after the dedup (normal) or the index (lean), from the room left:
- **chunk**: E12's 16 M distinct sequences when lane A's batch and all its result slots fit; else about 15 % of the room for lane A's
  slots (640 + 7 bytes per sequence; lean: plus the chunk's sequences), rounded so that the chunks are even.
- **lane A**: E12's 524,288 reads in flight (36.4 KB of `Work` + `ReadBuf` each), or as many as fit next to all its slots.
- **lane M** (planned when its read count is known): the warp lane when its stack and a batch of at least 32,768 fit next to all its
  slots (2,048 + 7 bytes each); else thread per read (68.4 KB per read in flight). E12's 393,216 at most.
- **lane B**: E12's 131,072 at most (550 KB per read in flight), as many as the room holds next to all its slots (4,096 + 7 bytes each);
  no longer "the free memory less 3 GB".
- While the lanes keep their results on the device (normal), each chunk's room is reduced by 112 bytes for each sequence of this and the
  later chunks (their packed results and offsets).
- A lane whose smallest batch and all its slots do not fit runs in parts (`runLaneParts`: one `runLane` per part, the results joined
  on the host; nothing stays on the device then).
- `SINGLET_ENGINE_GPU_NB`, `_NBM`, `_NBB` and `_CHUNK` remain overrides (E12's switches): a set value is used as given.

Each run logs the derivation, e.g. (pbmc1k_v3, 40 GB):

```
engine_gpu: budget 40.0 GB: chunk 2879488 (11 chunks), NB 311296 / 110592 / 22528 (lane M warp per read); mode normal, E8 records
gathered from the device-resident results; room 16.79 GB after the index and the distinct sequences (23.21 GB in use), 12.49 GB for
the lanes (less the results kept for E8's gather), lane M's warp stack 3.96 GB; tally peak 38.51 GB at laneB (context, own buffers,
e9's keep, lane M's stack), within the budget
engine_gpu: E17 lanes: chunk 0: A 2879488 x 311296, M 630212 x 110592 warp, B 110352 x 22528; chunk 1: ...
```

Before E8's records the phase's additions (the records' arrays, `buildRecords`' output, E10's products, E11's keep) are estimated; if
they do not fit, E8 falls back to the hand-off file path with the reason logged (not reached in any run here).

### 1d. The pileup's arena

`pileupRun` allocated free − 3 GB as one arena up front: 90.4 GB in `nvidia-smi` on pbmc1k_v3 with E16's head, of which the
pileup uses 24.6 GB. It now takes min(free − 3 GB, budget − 3 GB): 88.1 GB at the full budget, 77.0 at 80 GB, 37.0 at 40 GB, 21.0 at
24 GB. The arena's cell partitions adapt to its size: one partition down to 37 GB; three at 21 GB (peak 14.2 GB in the arena, GPU time
2.16 → 4.38 s, the same outputs). If the arena is too small the existing fallback (unified memory, twice the device) runs; it did not
happen here. The log line `[gpu_pileup] on: ... arena X GB (E17 budget Y GB), peak Z GB; cell partitions: k` gives both.

### 1e. The non-host screen: a fixed extra

`nonhostClassify` (`src/engine_gpu/nonhost_gpu.cu`, called from `src/pipeline/singlet.cpp`) loads the viral, bacterial and fungal
minimizer indexes (24,871 MB) and the host Bloom filter onto the device, then the unmapped reads, their minimizers and hits. It runs
after STAR's loop and the pileup, under its own device lock, when the prefill's and the pileup's memory are gone. It is outside E17's
files, so the budget does not size it: it is a fixed extra. On pbmc1k_v3 (3,355,559 unmapped reads) the device holds 25.4 GB once the
index is loaded and 29.3 GB at the classify step (1 s sampler, gate 797453 at the 24 GB budget, where it was alone on the device; the
merged gate's sampler caught only the 25.4 GB). So it fits inside budgets of about 30 GB and more in time (the phases do not overlap
within a sample), and exceeds a 24 GB budget by up to 5 GB. Its failure path (any CUDA error) already falls back to the CPU screen with
the same outputs. To bring it inside the budget: stream the index in parts, or keep one copy per device shared by the slots.

### 1f. Failures

A sample that cannot fit stops with a message, before the allocation that would exceed the budget:

```
EXITING because of FATAL ERROR (singlet --engine gpu, E17): the sample does not fit the device memory budget: the index (52.73 GB) and
the smallest lanes need 54.84 GB of device memory, more than the budget of 40.00 GB (SINGLET_ENGINE_GPU_MEM_GB); raise
SINGLET_ENGINE_GPU_MEM_GB or lower SINGLET_ENGINE_GPU_SLOTS (SINGLET_ENGINE_GPU_MEM_FALLBACK=cpu: STAR maps every read instead)
```

(hgmm1k_v3 at 40 GB, probe 797452; pbmc1k_v3 at 12 GB: "the sample's reads phase (upload, clip, dedup) needs 14.98 GB".)
`SINGLET_ENGINE_GPU_MEM_FALLBACK=cpu` turns it into E12's fallback (STAR maps every read; logged as such). The checks: the reads' phase
(lean), the index and the smallest lanes (both modes, at the start and again after the dedup with the distinct sequences' real size),
each lane before it allocates (`lane X (n reads of chunk c) does not fit: ...`). Other allocation failures keep the old behaviour (STAR
maps every read).

### 1g. Switches

| switch | meaning |
|---|---|
| `SINGLET_ENGINE_GPU_MEM_GB=<GB>` | the budget (GB of 2^30 bytes); `0`: none (E12's sizes) |
| `SINGLET_ENGINE_GPU_SLOTS=<k>` | samples per GPU (existing); the default budget is (device − 2 GB) / k |
| `SINGLET_ENGINE_GPU_MEM_FALLBACK=cpu` | a sample that does not fit runs STAR on every read instead of stopping |
| `SINGLET_ENGINE_GPU_LEAN=1` / `0` | debug: force the lean / normal mode under a budget |
| `SINGLET_ENGINE_GPU_NB`, `_NBM`, `_NBB`, `_CHUNK` | E12's overrides, used as given |

## 2. 64-bit junction keys

E10's `products()` (`src/engine_gpu/hostloop_gpu.cu`) collapses the junctions of all records by sorting one 64-bit key per junction entry,
`start << 32 | gap`, with the record and the junction's values as payload (stable: record order within a key). A start at or above 2^32
was counted and the products were refused ("junction start beyond 32 bits"), so levels 3–5 could not run on the barnyard (5.8 G bases).

The key is now `start << gb | gap`, with the split chosen per genome from its end (`chrStart[nChr]`, passed by `mapDistinct`):
- gb = 32 when the genome ends below 2^32: E10's key, bit for bit (pbmc1k_v3's GRCh38 and every genome gated before);
- gb = 64 − (the bits of the genome's end) above: 31 on the barnyard (the end needs 33 bits), so the gap keeps 31 bits (2 G bases; STAR's
  `alignIntronMax` here is 589,824).

Sorting (start, gap) as one key is the same order as before, so the junctions come out in the same order, and the host decode takes
`start = key >> gb` and the gap from the low bits. An entry whose start or gap does not fit is still counted, and `products()` then
refuses with the count and the bits, as before. `Output::E10::sjGapBits` records the split. One key, one sort: no 128-bit key and no
second sort pass were needed.

Other 32-bit packings checked on the barnyard path: E9/E11's Solo model keeps transcript starts and ends in 64 bits (`trS`, `trE`) and
exon coordinates relative to the transcript in 32 bits, as STAR does; E8's records carry 64-bit positions; the hand-off entries carry
none. The gates (§4) are the check: hgmm1k_v3 at level 5 gives STAR's `SJ.out.tab` and Solo files.

## 3. Budget derivation and peaks

Per budget: the plan (`engine_gpu: budget ...` line), the room left for the lanes after the index and the distinct sequences, the tally's
peak, the engine's own device memory mark (`cudaMemGetInfo`, "peak (lanes, records)"), and the 1 s `nvidia-smi` sampler's peak in the
prefill window (start to "cache filled") and after it (the pileup's arena, the non-host screen). Gate rows: gate 797462 (pbmc1k_v3) and
797468 (hgmm1k_v3), merged binary 709e34ea, T = 18, g051. Probe rows (T = 8, stopped after the prefill): 797433, 797439, 797452 (binaries
883cd7c3 / 63ffe997 / d9d3343f: the same plans as 709e34ea where both ran) and 797457 (709e34ea).

**pbmc1k_v3** (66.6 M reads, 31.7 M distinct; index 16.37 GB, reads 6.09 GB; normal estimate 34.8 GB, lean 18.5 GB):

| budget | mode | chunk (chunks) | NB A / M / B | lane M | room for the lanes | tally peak | engine mark | `nvidia-smi` prefill | after the prefill | pileup arena (used, partitions) |
|---|---|---|---|---|---|---|---|---|---|---|
| none: E16's head (bin-c14) | – | 16 M (2) | 524,288 / 393,216 / free − 3 GB | warp | – | – | – | 90.4 | 90.1 | free − 3 GB (24.6, 1) |
| **91.1** (default, 1 slot) | normal | 16 M (2) | 524,288 / 393,216 / 115,712 | warp | 63.56 | 88.38 (lane B) | 88.56 | **88.6** | 89.3 | 88.1 (24.6, 1) |
| 91.1, the default GPU path (gate (c), 797468) | normal, no results kept | 16 M (2) | 524,288 / 393,216 / 131,072 | warp | 70.34 | 89.65 (lane B) | 89.83 | 89.8 | 88.6 | 88.1 (24.6, 1) |
| **80** | normal | 16 M (2) | 524,288 / 393,216 / 94,208 | warp | 52.49 | 77.36 (lane B) | 77.53 | **77.5** | 78.2 | 77.0 (24.6, 1) |
| 45.5 (default, 2 slots; probe) | normal | 3.96 M (8) | 454,656 / 188,416 / 32,768 | warp | 18.03 | 43.91 (lane B) | 44.09 | 43.8 | – | (42.5) |
| **40** | normal | 2.88 M (11) | 311,296 / 110,592 / 22,528 | warp | 12.49 | 38.51 (lane B) | 38.69 | **38.0** | 38.2 | 37.0 (24.6, 1) |
| 38 (an A100 40 GB; probe) | normal | 2.44 M (13) | 262,144 / 81,920 / 18,432 | warp | 10.49 | 36.50 (lane M) | 36.68 | 36.2 | – | (35.0) |
| 30.4 (default, 3 slots; probe) | lean | 2.64 M (12) | 315,392 / 110,592 / 22,528 | warp | 12.46 | 29.30 (lane M) | 29.40 | 29.4 | – | (27.4) |
| **24** | lean | 1.27 M (25) | 151,552 / 81,920 / 11,264 | thread | 6.10 | 23.00 (lane M) | 23.17 | **23.2** | 25.4 (non-host) | 21.0 (14.2, 3) |
| 24, `CHUNK` 16 M | lean | 16 M (2), A in 6 + 5 parts, M in 4 + 3 | 86,016 / 32,768 / 3,072 | thread | 6.10 | 22.90 (lane M) | 23.07 | 23.1 | 25.4 (non-host) | 21.0 (14.2, 3) |
| 12 | stops before any allocation: "the sample's reads phase (upload, clip, dedup) needs 14.98 GB" | | | | | | | 0.0 | | |

(Arena values in parentheses are budget − 3 GB, from probes that stop after the prefill. The NB in the plan line is lane A's planned
batch; a chunk's own plan can be a little smaller, e.g. 307,200 at 30.4 GB, when the room at the chunk is smaller. Lane B's NB is the
first chunk's.)

**hgmm1k_v3** (63.1 M reads, 24.1 M distinct; barnyard index 52.73 GB, reads 5.76 GB; normal estimate 70.3 GB, lean 54.9 GB):

| budget | mode | chunk (chunks) | NB A / M / B | lane M | room for the lanes | tally peak | engine mark | `nvidia-smi` prefill | after the prefill |
|---|---|---|---|---|---|---|---|---|---|
| none: E15 (gates 796865, 797017) | – | 16 M | 524,288 (also 131,072, 65,536) / … | – | – | – | – | 85–90 at the failure | lane A's `dWork`: out of memory, level 0 |
| **91.1** (default) | normal | 16 M (2) | 524,288 / 311,296 / 52,224 | warp | 28.86 | 89.11 (lane B) | 89.28 | **89.0** | 89.3 (arena 88.1; 16.0 used, 1 partition) |
| 91.1, the default GPU path (gate (c)) | normal, no results kept | 16 M (2) | 524,288 / 393,216 / 66,560 | warp | 34.66 | 90.07 (lane B) | 90.25 | 83.4 | 88.6 |
| 78 (an A100 80 GB; probe) | normal | 3.43 M (7) | 397,312 / 167,936 / 131,072 | warp | 15.78 | 76.32 (lane M) | 76.49 | 76.2 | – |
| 64 (probe) | lean | 2.00 M (12) | 245,760 / 77,824 / 131,072 | warp | 9.72 | 62.91 (lane A) | 63.08 | 63.1 | – |
| 40 (probe) | stops before any allocation: "the index (52.73 GB) and the smallest lanes need 54.84 GB" | | | | | | | 0.5 | |

In every run the `nvidia-smi` prefill peak is under the budget, the tally is within 0.2 GB of the device after every lane's kernels
(6 to 75 lanes per run), and the engine's mark is the tally's peak plus 0.10–0.18 GB (what the tally does not see; the slack is 1 GB).
After the prefill the pileup's arena (budget − 3 GB) and the non-host screen (§1e) set the peak.

## 4. Identity per output

Checks per pair (the gate's, as E11 §2a / E15 §2): `bench/star_accel/out_equiv.py` over the output directory; the 10 STAR Solo files by
`cmp`; `star_Log.final.out`'s mapping statistics; `star_SJ.out.tab` by `cmp`; `Gene/CellReads.stats` (hard-linked before the pipeline
deletes it) by sorted rows; `nonhost_reads.fq` and `star_Unmapped.out.mate1/2` by sorted 4-line records (when both sides have them).

### 4a. Gate (b): pbmc1k_v3, level 5 + ingest under budgets (gate 797462, merged binary 709e34ea)

Every run against E11's kept CPU engine outputs (gate 796377, binary 8d9eaa85):

| run | out_equiv | Solo files | CellReads.stats | SJ.out.tab | mapping statistics |
|---|---|---|---|---|---|
| full budget (91.1 GB), two runs | 60 identical, 1 timing/path, 8 differ | 10 / 10 identical | same 323,542 rows | identical (126,996) | identical |
| 80 GB | 60 identical, 1 timing/path, 8 differ | 10 / 10 | same 323,542 rows | identical (126,996) | identical |
| 40 GB | 60 identical, 2 timing/path, 7 differ | 10 / 10 | same 323,542 rows | identical (126,996) | identical |
| 24 GB (lean) | 60 identical, 2 timing/path, 7 differ | 10 / 10 | same 323,542 rows | identical (126,996) | identical |
| 24 GB, 16 M chunk (lanes in parts) | 60 identical, 2 timing/path, 7 differ | 10 / 10 | same 323,542 rows | identical (126,996) | identical |
| E16's head (bin-c14), two runs | 60 identical, 1 timing/path, 8 differ | 10 / 10 | same 323,542 rows | identical (126,996) | identical |

The differing files are the same in every run, the head's included:
- `nonhost_reads.fq`, `star_Unmapped.out.mate1/2`: absent from E11's kept copy (that gate did not keep them; their records are compared
  in §4b on a fresh CPU run);
- `pileup_stats.json`'s `bytes_public` and `provenance.json`'s whitelist path (the binary's directory);
- `read_stats.tsv`, `star_Log.std.out`: stage timings;
- `summary.json`'s `memory_tier` (Large vs Medium, from the host's peak RSS; at 40 and 24 GB the GPU runs stay under 64 GB and report
  Medium like the CPU engine: 7 differ instead of 8, §6.5).

So the budget changes nothing in the outputs: the lanes' batch sizes, the chunking, the lean mode (entries to the region at the clip,
the late E11 keep, the records uploaded from the host assembly) and the parts all give the head's bytes. The same holds before the merge
(gate 797453, d9d3343f, against E15's head: 60 identical in every run, the same differing files).

### 4b. Gate (a): hgmm1k_v3, CPU engine vs level 5 + ingest (gate 797468)

A fresh CPU engine run and the two GPU paths, all from the merged binary 709e34ea, ran in the same job on g051 at T = 18:

| pair | out_equiv | Solo files | CellReads.stats | SJ.out.tab | mapping statistics | `nonhost_reads.fq`, `Unmapped.out.mate1/2` |
|---|---|---|---|---|---|---|
| CPU vs **level 5 + ingest** | 49 identical, 2 timing/path, 7 differ | 10 / 10 identical | same 524,497 rows | identical (171,869) | identical | same records (309,553 / 1,812,610 / 1,812,610 reads) |
| CPU vs the default GPU path | 49 identical, 2 timing/path, 7 differ | 10 / 10 identical | same 524,497 rows | identical (171,869) | identical | same records |

The 7 differing files are E15's 7 run-dependent ones (E15 §2c, where these GPU paths fell back to level 0):
- `nonhost_reads.fq`, `star_Unmapped.out.mate1/2`: the same size and the same records, in another order. The unmapped file's order
  differs between the CPU engine and every GPU path (E15 §2c).
- `pileup_stats.json`: `bytes_public`, which differs by a few KB in every run. The default path also differs in
  `clip5p/probe/n_reads` (12,192 vs 12,000; E11 saw 12,194 vs 12,000 on pbmc1k_v3).
- `read_stats.tsv`, `star_Log.std.out`: stage lines and timings.
- `summary.json`'s `memory_tier`: Medium (the CPU engine, 62.2 GB peak RSS) vs Large (78.4 / 77.7 GB).

So the barnyard index now maps on the device at level 5 for the first time, with its E10 products (1,898,705 junction entries →
194,331 junctions, 1,753,600 unmapped reads) and E11's Solo on the device. The outputs are the CPU engine's.
- The 31-bit start of the junction key (§2) sorts the same junctions as STAR's CPU path: `SJ.out.tab` is byte-identical.
- At the full budget the plan is the one E12 would choose, except lane B: NB 52,224, against "free − 3 GB" before.
- The pileup's arena is 88.1 GB, of which 16.0 GB is used, in 1 partition.

### 4c. Gate (c): the default GPU path (gate 797468)

The default GPU path (`SINGLET_GPU_PILEUP=on`, no level 5 or ingest), merged binary 709e34ea, pbmc1k_v3:

| pair | out_equiv | Solo files | CellReads.stats | SJ.out.tab | mapping statistics |
|---|---|---|---|---|---|
| E11's kept gpuD (gate 796377, bin-s5) vs gpuD | 60 identical, 2 timing/path, 7 differ | 10 / 10 identical | same 323,542 rows | identical (126,996) | identical |
| E11's kept CPU engine vs gpuD | 60 identical, 1 timing/path, 8 differ | 10 / 10 identical | same 323,542 rows | identical (126,996) | identical |

Against E11's gpuD, the 7 differing files are:
- 3 files missing from the kept copy: `nonhost_reads.fq` and `star_Unmapped.out.mate1/2`;
- `pileup_stats.json`'s `bytes_public`;
- `provenance.json`'s whitelist path, which is the binary's directory;
- `read_stats.tsv` and `star_Log.std.out`, which hold timings.

Against the CPU engine the list adds `summary.json`'s `memory_tier` and the probe's `n_reads` (12,194 vs 12,000), as in E11.

The budget's plan for this path:
- chunk 16 M, NB 524,288 / 393,216 / 131,072;
- the lanes' results go to the host, because this path builds no records on the device: its hand-off goes to the file, as before
  (the plan line still says "E8 records uploaded from the host assembly", a log nit, §6.9);
- the tally peaks at 89.65 GB, the engine's mark at 89.83 GB and the prefill's `nvidia-smi` at 89.8 GB.

hgmm1k_v3's default path is the second row of §4b: identical to the CPU engine, where E15's gpuD fell back to level 0.

## 5. Timings

T = 18, short-gpu node g051 (H100 NVL), one run at a time. Wall time and core-s are for the whole `singlet --pipeline` run (cgroup
`cpu.stat`); "prefill" is STAR's "cache filled ... in N s"; "lock" is how long the prefill held the device lock ("held it").

**pbmc1k_v3, level 5 + ingest, merged binary 709e34ea (gate 797462), in run order:**

| run | wall s | core-s | prefill s | GPU mapping s | lock s |
|---|---|---|---|---|---|
| E16's head (bin-c14, e0e1514a's code), no budget | 103.2 | 554.0 | 18 | – | 16.2 |
| **full budget (91.1 GB)** | **103.0** | **553.7** | 18 | 9.27 | 16.1 |
| 80 GB | 105.1 | 555.7 | 21 | 9.45 | 18.3 |
| 40 GB | 108.4 | 564.5 | 23 | 14.49 | 20.9 |
| 24 GB (lean) | 121.4 | 608.6 | 35 | 24.03 | 32.3 |
| 24 GB, `CHUNK` 16 M (lanes in parts) | 149.9 | 660.0 | 62 | 50.31 | 59.2 |
| E16's head, again | 104.5 | 576.3 | 19 | 9.36 | 16.1 |
| **full budget, again** | **103.2** | **568.1** | 19 | 9.33 | 16.3 |

- **The full budget is not slower than the head:** 103.0 / 103.2 s against 103.2 / 104.5 s in the same job (E16's gate 797428:
  102.8 s). Its plan is E12's except lane B (115,712 reads in flight instead of "free − 3 GB", 114–117 k in E12), so the device work is
  the same; the prefill's device peak is 1.8 GB lower (88.6 vs 90.4 GB) and the pileup's arena 88.1 GB.
- **What a smaller budget costs** (against the full budget's mean, 103.1 s): 80 GB +2.0 s (lane B 94 k; the upload ran slower in that run,
  4.8 vs 2.7 s); 40 GB +5.3 s (11 chunks, lane B 22.5 k: GPU mapping 9.3 → 14.5 s); 24 GB +18.3 s (lean mode: 25 chunks, lane M thread
  per read, lane B 11 k: GPU mapping 24.0 s; the pileup in 3 partitions, +2.2 s GPU). Lane B's 550 KB `Work` per read in flight is what
  small budgets pay for: at 24 GB it takes 12.7 s of the 24.0 s (2.7 s at the full budget).
- The parts (24 GB with a 16 M chunk forced) cost another 28.5 s: a forced chunk the budget would not choose; the derived chunk never
  needs parts on these samples.
- The device lock is held for the prefill's whole device phase and its host assembly (§7); it grows with the GPU mapping time.

**Before the merge** (binary d9d3343f against E15's head bin-m3, gate 797453, g051): head 129.7 / 130.3 s, full budget 129.4 / 129.6 s,
80 GB 127.1 s, 40 GB 131.4 s, 24 GB 145.8 s; the same identity as §4 (60 identical against E11's CPU outputs in every run).

**hgmm1k_v3 (gate 797468):**

| run | wall s | core-s | maxRSS | prefill s | GPU mapping s | lock s | `nvidia-smi` prefill / after GB |
|---|---|---|---|---|---|---|---|
| CPU engine | 374.9 | 1,609.1 | 62.2 GB | – | – | – | 0 / 0 |
| **level 5 + ingest** | **88.7** | **437.5** | 78.4 GB | 17 | 3.79 | 14.7 | 89.0 / 89.3 |
| default GPU path | 176.9 | 863.3 | 77.7 GB | 31 | 3.75 | 10.6 | 83.4 / 88.6 |
| E15: CPU engine (gates 796865 g051, 797017 g050) | 431.7, 345.6 | 1,682.8, 1,587.4 | 62.2 GB | | | | |
| E15: gpuD, gpuE5 (GPU mapping failed: level 0) | 361.2, 373.0 (289.9 with `NB` 65,536) | 1,729.9, 1,783.1 | 78–83 GB | | | | |

- Level 5 + ingest is **4.2x faster in wall time and 3.7x in core-s** than the CPU engine on the barnyard. Its stages:
  - STAR's prefill and map: 8 s;
  - the stream into the pileup: 18 s;
  - the hand-off: 5 s;
  - non-host: 13 s;
  - demux, ancestry and the writers: 30 s.
- hgmm1k_v3 is faster than pbmc1k_v3 (103 s) at level 5. It has 24.1 M distinct sequences against 31.7 M, its GPU mapping takes 3.8 s
  against 9.3 s, and its pileup has no SNP or MT tuples.
- In the default GPU path, STAR's prefill and map stage takes 71 s. That is the replay of the hand-off summaries, as on pbmc1k_v3
  (E11).
- On pbmc1k_v3 in the same job, the default path took 204.5 s / 992.9 core-s (E11's gate 796377: 253.4 s / 996.6 core-s). Its prefill
  `nvidia-smi` peak was 89.8 GB.

## 6. Surprises

1. **The failure path crashed.** The first budget failure (pbmc1k_v3 at 12 GB, probe 797439) ended in "STAR killed by signal 11", with
   no message in `Log.out`. `mapDistinct`'s `fail()` reset the device (`cudaDeviceReset`) while E12's staging buffers (two `Stager`s and
   the `D2H`) still held streams, events and page-locked buffers; their destructors then used the stale handles. A failure after their
   init had never been reached on the gated paths (E15's hgmm1k_v3 failures predate E12's merge). `fail()` now releases them first
   (d9d3343f); the same probe then printed the message (797447) and, with `SINGLET_ENGINE_GPU_MEM_FALLBACK=cpu`, ran STAR on every read
   (797452).
2. **Lane M's warp lane reserves 4 GB at once.** Raising the stack limit makes the driver reserve the stack for every thread the device
   can hold (4,050 MiB on the H100 NVL), device-wide and invisible to `cudaMalloc` accounting. With a budget it must be planned like a
   buffer; below about 30 GB it does not fit next to a useful batch and lane M runs thread per read (24 GB). Thread per read sends 7 more
   sequences to the host lane (85,570 vs 85,563, as E12 found for `SINGLET_ENGINE_GPU_WARPM=0`); the outputs are the same.
3. **The pileup's arena took everything for a quarter of it.** 88–90 GB allocated, 24.6 GB used on pbmc1k_v3: E18's two samples could
   not share the device for this alone. Budget − 3 GB is still mostly idle at the full budget; with the arena sized to the need (a
   pass over the input sizes; outside E17) it would need about its 24.6 GB peak at any budget.
4. **The non-host screen is the largest fixed user after the index.** 29.3 GB at its classify step on pbmc1k_v3, more than the 24 GB
   budget's whole prefill (§1e).
5. **`summary.json`'s `memory_tier` follows the host's peak RSS** (Medium below 64 GB, Large above). The GPU runs at 40 and 24 GB stay
   under 64 GB and report "Medium" like the CPU engine, so their comparison has 7 differing files instead of 8 (§4).
6. **The lean mode's per-chunk host gather cost 3.7 s** (63ffe997: a gather of each chunk's sequences on the host and pageable copies).
   Compacting the distinct sequences once on the host and staging each chunk's slice (d9d3343f) brought it to 0.45 + 0.42 s.
7. **`sbatch --export` splits at commas.** A probe's `RUNS` with `K=V,K=V` lost its last runs (797447); the jobs now get such values
   through the environment (`VAR=... sbatch --export=ALL`).
8. **The step table counts `lean.seq.host` twice in lean runs**: `dedup.order` is measured from the previous step and includes it (0.99 vs
   0.15 s). The total is right. Not fixed (log only).
9. **The plan line names E8's records on the default GPU path**, which has none: "E8 records uploaded from the host assembly" where
   the lanes' results simply go to the host for the hand-off file. Not fixed (log only).

## 7. What remains

**The A100.** Nothing here ran on one. What carries over and what does not:
- The budget is derived from the device: (40 − 2) = 38 GB on an A100 40 GB, 78 GB on an A100 80 GB (one slot). The probes at those
  budgets on the H100 (797457) give the plans: pbmc1k_v3 at 38 GB runs normal (chunk 2.44 M, 13 chunks, NB 262,144 / 81,920 / 18,432,
  peak 36.2 GB); hgmm1k_v3 at 78 GB runs normal (chunk 3.43 M, 7 chunks, peak 76.2 GB). hgmm1k_v3 cannot run on an A100 40 GB at all
  (its index alone is 52.7 GB): it stops with E17's message, or runs STAR on every read with `SINGLET_ENGINE_GPU_MEM_FALLBACK=cpu`.
- Lane M's warp stack is measured on the device (108 SMs × 2,048 threads: about 3.3 GB instead of 4.0).
- The context's size, the driver's granularity and the local memory differ; the tally starts from the measured context and the slack is
  1 GB. The tally-to-device check of §3 should be repeated there (the lane log prints both).
- The kernels' speed on sm_80 (the binary carries sm_80 code) and therefore the time each budget costs are not measured. A 40 GB A100
  runs pbmc1k_v3 like the 40 GB row here: 11–13 chunks, lane B at ~20 k reads in flight.
- The non-host screen needs ~29 GB: it fits a 40 GB A100 alone, not beside another sample's prefill (§1e).

**The device lock (E18's lever).** Not touched. `mapDistinct` releases it (`in.release`) after its last device work; at level 5 that is
after E8's records are built on the device, so the host lane, the host assembly of the records (records.prep, records.copy and
readRecord: ~3.6 s on pbmc1k_v3) and, in lean mode, the chunks' host-side compaction (0.45 s) run under it. Measured hold: 16.1 s at the
full budget, 20.9 s at 40 GB, 32.3 s at 24 GB (§5). Releasing the lock during the host assembly needs the records' device phase moved
after it or split from it (E8 builds the records from the device-resident lane results). The other holders E18 listed (e2ePass,
E11's Solo, the pileup, the non-host screen) are outside `mapDistinct`.

**The budget's gaps.**
- The non-host screen (§1e) and E16's hand-off pass (`e2ePass`, `records_gpu.cu`) are not sized by the budget. The hand-off pass's
  device use is inside the "after the prefill" peaks of §3, which stay within the budget except the non-host screen's at 24 GB.
- The pileup's arena is budget − 3 GB, not its need (§6.3).
- The lean mode's estimate of the reads' phase is a bound, not a plan: a sample whose reads alone (upload, clip, dedup) exceed the
  budget stops (pbmc1k_v3 needs 15 GB). Chunking the reads' phase would lower that floor.
- `runLaneParts` (a lane in parts) is reached only with a forced chunk; it was gated once (24 GB with `CHUNK` 16 M, identical).

**Not done: STAR's CellRanger4 clip on the device** (optional in the brief). The clip stays on the host (E13's ingest computes it
during the decode); the budget work took the time.

## 8. Files touched, and why

| file | change | why |
|---|---|---|
| `src/engine_gpu/engine_gpu.cu` | the budget's tally (`Budget`, `dmal` / `dfree`, `TAlloc` / `TP` for thrust); every allocation of `runLane` and `mapDistinct` through it; lane M's stack measured and tallied; Work and the batch buffers freed before the packed results; `laneSlotB` / `lanePerB` / `laneNeed` / `lanePlan` / `runLaneParts`; the plan after the context (modes normal / lean, estimates, early failures with `Output::budgetFail`); lean mode (index after the dedup, E10's entries to the region at the clip, the late E11 keep, the chunks' sequences compacted on the host and staged, host zeroing of entries and the hook list, E8's records uploaded from the host assembly when the lanes' results are not all on the device); the chunk and lane batches from the room; lane B from the budget; the records' estimate; the E17 log lines; `fail()` releases the staging buffers before the reset; `products()` gets the genome's end | part 1, the 64-bit key's caller |
| `src/engine_gpu/hostloop_gpu.cu`, `.h` | `k_sjfill`'s key `start << gb \| gap` with gb from the genome's end; the host decode; the overflow message; `products(…, genomeEnd)`; without `dEnt` no zeroing and no hook list (the caller does both) | part 2; lean mode |
| `src/engine_gpu/pileup_gpu.cu` | the arena is min(free − 3 GB, budget − 3 GB); the log line | part 1 (the coordinator's follow-up) |
| `include/singlet/engine_gpu.h` | `memBudget`; `Output::{budgetFail, budgetWhy}`; `Output::E10::sjGapBits`; the slots comment | the API of the above |
| `src/star/SeqCacheGpuPrefill.cpp` | after `mapDistinct`: a budget failure stops the run with its message (`SINGLET_ENGINE_GPU_MEM_FALLBACK=cpu`: STAR maps every read) | part 1 |
| `bench/fable/budget/jobs/{build,probe,gate}.sbatch` | build; the prefill probe (`RUNS="label\|sample\|K=V,..."`, `BIN_<label>`); the gate (modes `BASE_TAG` with `ENV_<mode>` / `BIN_<mode>`, the device memory peaks overall, in the prefill window and after it, the E17 lines) | deliverable |
| `docs/gpu-engine/fable/E17-budget.md` | this report | deliverable |

Not changed: `SeqCacheGpuIngest.cpp` (no hook needed), the lock's placement, `nonhost_gpu.cu`, `records_gpu.cu`, `solo_gpu.cu`, STAR.

## 9. Jobs, commits, how to run

**Branch `gpu/fable/budget`** (from `gpu/fable-engine` 44a6e9fa):
- 11bc6d8c: build and probe jobs.
- 883cd7c3: the budget (tally, plan, lanes, lane B and the arena from the budget, failures) and the 64-bit junction keys.
- 9b90e9ec: the gate job.
- 63ffe997: lean mode uploads only the chunk's sequences.
- d9d3343f: `fail()` releases the staging buffers before the device reset; lean mode compacts the distinct sequences once on the host.
- 709e34ea: merge of `gpu/fable-engine` at e0e1514a (E16 "tail"; no conflicts). **Last change to `src/`.**
- this report (the commit after 709e34ea; no code change).

**Builds** (short-gpu, no GPU; `$W/jobs/fable/budget/`):

| job | commit | output |
|---|---|---|
| 797432 | 883cd7c3 | `bin/singlet-gpu` (probe 797433) |
| 797434 | 63ffe997 | `bin2/singlet-gpu` (probe 797439) |
| 797446 | d9d3343f | `bin3/singlet-gpu` (probes 797447, 797452; gate 797453) |
| 797454 | 709e34ea (the merge) | `bin4/singlet-gpu` (probe 797457; gates 797462, 797468) |

Reference binaries: E15's head `$W/jobs/fable/posterior/bin-m3/singlet-gpu` (60a60f92 = 44a6e9fa's code), E16's head
`$W/jobs/fable/tail/bin-c14/singlet-gpu` (a6dfaf9b = e0e1514a's code).

**Probes** (debug-gpu, T = 8, level 5 + ingest, stopped after the prefill):

| job | binary | runs |
|---|---|---|
| 797433 | 883cd7c3 | pbmc1k_v3 at the full budget, 40, 24 GB; hgmm1k_v3 at the full budget |
| 797439 | 63ffe997 | pbmc1k_v3 at 24 GB, 3 slots, 12 GB (the crash, §6.1); hgmm1k_v3 at 64 GB |
| 797447 | d9d3343f | pbmc1k_v3 at 12 GB: the message (the fallback run lost to `--export`'s comma split) |
| 797452 | d9d3343f | pbmc1k_v3 at 12 GB with the CPU fallback, at 24 GB, 3 slots; hgmm1k_v3 at 40 GB (the message) |
| 797457 | 709e34ea | pbmc1k_v3 with 2 slots, at 38 GB; hgmm1k_v3 at 78 GB |

**Gates** (short-gpu, H100 NVL g051, T = 18):

| job | binary | samples, modes | result |
|---|---|---|---|
| 797453 | d9d3343f (before the merge) | pbmc1k_v3: E15's head ×2, full ×2, 80, 40, 24 GB, against E11's kept CPU outputs | identical (§4), §5 |
| 797462 | 709e34ea | pbmc1k_v3: E16's head ×2, full ×2, 80, 40, 24 GB, 24 GB with a 16 M chunk, against E11's kept CPU outputs | gate (b): §3, §4, §5 |
| 797468 | 709e34ea | hgmm1k_v3: cpu, gpuE5i, gpuD; pbmc1k_v3: gpuD against E11's kept gpuD and CPU | gates (a), (c): §3, §4, §5 |

**Results:** `$W/jobs/fable/budget/results/{gate,probe}-<job>/` (`summary.txt`; per gate run `stages.txt`, `starlog_engine.txt`,
`star_Log.out`, `CellReads.stats`, `monitor.tsv`, `run.log.ts`; per probe run `<label>.star_Log.out`, `<label>.monitor.tsv`). Logs:
`$W/logs/e17-{build,probe,gate}-<job>.out`. `$W` = `/mnt/projects/debruinz_project/debruinz/singlet`.

**Run** (values with commas or spaces through the environment, not `--export`, which splits at commas):
- build: `sbatch --export=ALL,OUTD=<dir> bench/fable/budget/jobs/build.sbatch`;
- probe: `BINDIR=<dir> RUNS="b24|pbmc1k_v3|SINGLET_ENGINE_GPU_MEM_GB=24 h|hgmm1k_v3|" sbatch --export=ALL bench/fable/budget/jobs/probe.sbatch`;
- gate (b): `BINDIR=<dir> SAMPLES=pbmc1k_v3 MODES="gpuE5i gpuE5i_b40" ENV_gpuE5i_b40=SINGLET_ENGINE_GPU_MEM_GB=40
  PAIRS="cpu:gpuE5i cpu:gpuE5i_b40" sbatch --export=ALL bench/fable/budget/jobs/gate.sbatch` (pbmc1k_v3's cpu defaults to E11's kept
  outputs; `BIN_<mode>` runs another binary; `OUT_<mode>` / `CRS_<mode>` name kept outputs for a mode not run).

**Switches:** §1g. The default (no switch) is the budget (device − 2 GB) / `SINGLET_ENGINE_GPU_SLOTS`; `SINGLET_ENGINE_GPU_MEM_GB=0`
restores E12's sizes.
