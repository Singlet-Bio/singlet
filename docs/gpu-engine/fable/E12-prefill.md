# E12: the GPU prefill, 38 s of device time cut to 16 s (prefill)

Experiment E12 ("prefill") of the GPU aligner program. Branch `gpu/fable/prefill`, from `gpu/fable-engine` at 8e54be40 (E11), with
`origin/gpu/fable-engine` e2b0000e (E14) merged at the end (a43d4ff4). It works on `mapDistinct` (`src/engine_gpu/engine_gpu.cu`), the
device phase of STAR's prefill under `--engine gpu`: on pbmc1k_v3 at level 5 (T = 18, H100 NVL) the prefill was 58 s, of which the GPU
part was 38 s.

Every change has a switch back to the code before it (§1), and the switches were measured one at a time in one job (§2b).

## 0. Answer in brief

- **Device phase of the prefill (`mapDistinct`, "GPU" in STAR's log): 38.5 s -> 16.2 s** (target <= 25 s). Same job, same node:
  gate 796641 (baseline, instrumented) and 796791 (E12, binary b38601ad), node g050. The final gate 796888 (node g051, merged binary
  a43d4ff4, two runs each) gives 16.9 / 17.4 s against 39.8 / 39.7 s for 8e54be40 and 38.9 / 41.4 s for the merged engine without E12
  (e2b0000e) (STAR's "held it").
- **The prefill stage: 58-59 s -> 37 s** (target <= 45 s): reading reads 17 s (E13's area, untouched), GPU 16 s, bookkeeping 4 s
  (`s8_stages_e11.py`: 59 s / 300 core-s -> 37 s / 299 core-s). In 796888: 37 / 38 s against 61 / 64 s (merged engine) and 62 / 65 s
  (8e54be40). The bookkeeping after the engine is 2.7 s (was 4 s): the sequence cache is no longer re-built at level 5.
- **The whole level-5 run.** 796791 (g050): 213.7 s -> 189.5 s (-24.2 s, -11.3 %; core-s 686 -> 681). 796888 (g051, with E14 merged):
  **161.6 s** (162.0 / 161.2) against **187.2 s** for the merged engine without E12 (185.5 / 188.9): **-25.6 s (-13.7 %)**, core-s
  674 vs 677; and 228.5 s for 8e54be40 (without E14 either).
- **Identity.**
  - The E6 dump of `mapDistinct`'s records (6,053,620,001 bytes: every sequence, blob, offset and `readRecord`) has the same sha256
    with everything on, with `SINGLET_ENGINE_GPU_CFG=old` (the engine before E12), and with each switch off on its own (796791).
  - Base (8e54be40) vs E12, whole run: 63 files identical, 3 differ only in timings or paths, 3 differ in timings and `bytes_public`
    (the same 3 as between two level-5 runs of one binary in E11); CellReads.stats byte-identical; SJ, Log.final.out statistics,
    unmapped and non-host reads identical.
  - **Gate (a), 796888:** CPU engine vs E12 at level 5: **60 identical**, 2 timing/path only, 7 differ: exactly E11's 7 (record order
    in 3 files with the same sorted records, timings in 2, `pileup_stats.json`'s probe count and `bytes_public`, `summary.json`'s
    memory tier). `Gene/CellReads.stats`: the same 323,542 rows. Mapping statistics, `SJ.out.tab` (126,996 junctions), unmapped
    and non-host reads identical.
  - **gpuD of E12 vs E11's kept gpuD (796377):** 60 identical; the rest are the 3 files E11's copy lacks, timings, `bytes_public` and
    the whitelist path. CellReads.stats: same rows. This is the default GPU path, which runs E12's code without E8 (records assembled
    after the device is released, the sequence cache re-sized and 2,839,242 results inserted).
  - E12 vs the merged engine (e2b0000e) and vs 8e54be40 at level 5: 63 identical, 3 timing-only, 3 differ (timings, `bytes_public`).
  - m5 harness (fable_cc 796639 + fable_gate 796640): **1,991,926 / 1,991,926 identical** for all four variants, including the best
    configuration at **3.64 M distinct reads/s** (3.07 for the defaults, 2.51 for m5b).
- **Where the 22 s came from** (one switch at a time, §2b; s = seconds of device phase):

  | change | saves | switch back |
  |---|---|---|
  | the records' host assembly sized once, without zero-fill, copied in parallel | 6.0 s (8.0 -> 1.9) | none (output-identical restructure) |
  | the lanes in m5's best exact configuration, on m5's `stitch.cuh` | 4.5 s (12.7 -> 8.2) | `CFG=old` |
  | device -> host copies through page-locked staging buffers | 3.9 s | `SINGLET_ENGINE_GPU_D2H=pageable` |
  | dedup products on the device | 3.2 s (3.5 -> 0.25) | `SINGLET_ENGINE_GPU_DEDUP=host` |
  | the host lane on a thread pool fed by lane B's batches, during the device lanes | 2.5 s | `SINGLET_ENGINE_GPU_HOSTLANE=after` |
  | staged uploads, the index on a second thread during reads, clip and dedup | 2.6 s | `SINGLET_ENGINE_GPU_UPLOAD=pageable` |

  Of the lanes' 4.5 s, 2.1 s come from `stitch.cuh` alone: the engine compiled a stale copy of m5's file, without E5's and E7's
  defaults (§4).

## 1. What was done

All in `mapDistinct` unless said otherwise. `SINGLET_ENGINE_GPU_CFG=old` turns off every switchable change at once (the lanes,
uploads, dedup, host lane and device -> host copies); the records' restructure (1g) and the sequence cache's sizing (1h) have no
switch.

**1a. Instrumentation (62ca9be3).** A wall clock per step (`engine_gpu: steps (s): ...; total` in `star_Log.out`). A device step
ends where the code synchronises; the steps add up to the total within 0.3 s.

**1b. The m5 best exact configuration (f7335345).** `src/engine_gpu/stitch.cuh` is now a forwarder to `bench/gpu/m5/stitch.cuh`, so the
engine and the harness compile one file (the engine had a copy of m5 811cbe88, before E5 and E7). The kernels of `runStaged` in
`bench/gpu/m5/m5.cu` are ported into `runLane`:
- lane A: 524,288 reads per batch, phase 1 split into seeds (`k_p1s`) and windows (`k_p1w`), phase 3 `k_p3lb` under a 64-register cap
  (`--p3-maxreg-a 64`);
- lane M: 393,216 per batch, warp per read (`k_cost` by windows, descending; `k_p3a` + `k_p3w` on 128 threads; `--warp-m --wsort
  --wthreads 128`). The warp lane is used only when its window loop cannot reach STAR's `alignTranscriptsPerReadNmax` break;
- lane B: 131,072 per batch, or as many as the free device memory holds with 3 GB spare (114-117 k here), persistent windows kernel
  (`k_p1wp`), phase 3 `k_p3e`;
- every batch ends with `k_copy_status`.
`SINGLET_ENGINE_GPU_CFG=old` runs the old `k_p1`/`k_p3` with 262,144 / 262,144 / 65,536; `_NB`, `_NBM`, `_NBB` and `_WARPM=0` set
single knobs.

**1c. Dedup products on the device (3859f65a, 1994295b).** Before: three arrays to the host and two serial passes over 66.6 M reads.
Now:
- `k_heads` marks group boundaries and writes per sorted position k or 0; an inclusive max-scan gives every position its group's first
  position. `k_collide` checks each member byte for byte against the group's first read (before: `k_groups` walked back to it, which
  is quadratic in a group's size; no measurable cost on this sample, 0.06 s either way, but the worst case is gone);
- `k_readrep` writes each read's representative (NONE for a sequence with N), a reduce counts the groups, and `copy_if` compacts the
  representatives without N in hash order. The prefix-key stable sort runs on that array in place; two arrays come back (`readRep`,
  `rep`).

**1d. The host lane during the device lanes (3859f65a).** `runLane` takes a per-batch callback; lane B's callback copies the batch's
statuses and pushes its overflows to a pool of T - 1 threads (`HostPool`), which map them with STAR's limits (`mapRead` on a `Work<Heavy>`
per thread) while the device runs on. Before the records, the main thread waits for the chunk's tasks. The results are stored by
index in the chunk's host lane, so the order is the one before.

**1e. Staged uploads (3859f65a).** Two 64 MB page-locked buffers per direction and stream (`Stager`): the host copies a chunk into one
buffer (8 threads for the reads, 4 for the index) while the other one's DMA runs. The index (17.6 GB) goes on a second host thread and a
non-blocking stream from the start of `mapDistinct`; the reads (6 GB), offsets and E10's aux follow on the main thread. The index thread
is joined after the dedup ("index.wait"). The device-wide syncs used as step markers became `cudaStreamSynchronize(0)`, so that they do
not wait for the index copy. No `cudaHostRegister` of the sources (pinning GBs costs seconds).

**1f. Staged device -> host copies (b38601ad).** The mirror image (`D2H`): the DMA of one 64 MB chunk runs while 16 threads copy the
previous one out, faulting the destination's pages in parallel. Used for the lanes' packed results and offsets, the clip spans, the
dedup products, and E8's records and E10's entries and masks into the shared region (before: fault the region's pages, `cudaHostRegister`,
`cudaMemcpy`, unregister).

**1g. The records' host assembly (b38601ad).** `hostPart` became `hostPrep` (host lane wait, per sequence its kept flag, lengths and
prefix sums) for every chunk, then one allocation of `Output::seq/blob/seqOff/blobOff` at their final size, then `hostCopy` per chunk in
parallel. The four arrays are `NoInitVector`s (E11's allocator: no zero-fill; every element is written by the copies). Before: each chunk
grew them with `resize`, a serial zero-fill of the new bytes plus, for the second chunk, a reallocation that copied the first; 8.0 s on one
thread. The lanes' host results, the clip spans, `recSrc`/`recRead` and the representative-position map are `NoInitVector`s as well.

**1h. After the engine (eba63af0, `src/star/SeqCacheGpuPrefill.cpp`).** The sequence cache is re-created at its fitted size only when
results are inserted; at E10 level >= 3 nothing is inserted, and the 12 GB cache with its zeroed 2 GB index was built for nothing. A log
line gives the wall time of the steps after the engine ("ENGINE_GPU: E12 after the engine (s): ...").

## 2. Per-step tables

### 2a. Before and after (pbmc1k_v3, T = 18, level 5, node g050)

`t` = 8e54be40 + timers (62ca9be3), gate 796641; `d` = b38601ad, gate 796791. Seconds of wall time on the thread that drives the device.

| step | before (`t`) | after (`d`) | what changed |
|---|---|---|---|
| init | 0.15 | 0.25 | + staging buffers (4 x 64 MB page-locked) |
| index.h2d | 3.08 | 0.24 | allocation and memset only; the copy runs on the index thread |
| reads.h2d | 1.12 | 0.60 | staged (6 GB at 11 GB/s) |
| e10.aux.h2d | 0.37 | 0.18 | staged |
| clip (kernel, entries, gather, free, host) | 0.41 | 0.51 | |
| clip.d2h | 0.85 | 0.19 | staged, no zero-fill |
| dedup (kernels, host alloc, d2h, host, order) | 3.56 | 0.24 | on the device |
| index.wait | - | 0.64 | the index (2.56 s from the start, 7.6 GB/s) ends after the dedup |
| lane A / M / B kernels | 4.48 / 4.53 / 3.68 | 2.84 / 2.73 / 2.63 | m5 configuration and `stitch.cuh` |
| lane A / M / B alloc | 0.23 / 0.06 / 0.11 | 0.09 / 0.05 / 0.26 | lane B: 64 GB of Work |
| lane A / M / B d2h | 1.23 / 0.46 / 0.21 | 0.23 / 0.12 / 0.13 | staged, no zero-fill |
| lanes.host | 0.52 | 0.47 | |
| hostlane | 2.52 | 0.003 | 42 core-s on 17 threads during lane B; nothing left to wait for |
| records.host | 7.98 | 1.93 | prep 0.64, alloc 0.00, copy 1.29 |
| readRecord | 0.55 | 0.42 | |
| e2e.build | 2.07 | 1.11 | region copies 0.86 + 0.85 -> 0.38 + 0.36 s (staged) |
| other (index.free, e2e.free, free) | 0.06 | 0.08 | |
| **total** | **38.46** | **16.18** | |

The remaining 16.2 s: lane kernels 8.2 (51 %), records 1.9, uploads and the index wait 1.4, E8/E10 into the region 1.1, lanes' host
work, allocations and copies 1.4, clip 0.7, readRecord 0.4, dedup 0.25.

### 2b. One switch at a time (gate 796791, binary b38601ad, same node, back to back)

Device phase (s); the E6 dump runs after the step line and is not in it.

| run | switch | device phase | vs all on |
|---|---|---|---|
| `d` | all on | 16.18 | |
| `dnew` | all on (with dump) | 15.85 | |
| `dpg` | `SINGLET_ENGINE_GPU_UPLOAD=pageable` | 18.75 | +2.57 |
| `dha` | `SINGLET_ENGINE_GPU_HOSTLANE=after` | 20.54 | +4.36 (hostlane +2.48; reads.h2d 3.32, see §4) |
| `ddh` | `SINGLET_ENGINE_GPU_DEDUP=host` | 19.33 | +3.15 |
| `dd2h` | `SINGLET_ENGINE_GPU_D2H=pageable` | 20.04 | +3.87 |
| `dold` | `SINGLET_ENGINE_GPU_CFG=old` | 31.01 | +14.83 |

`dold` is 7.4 s below `t`: `CFG=old` keeps the shared `stitch.cuh` (2.1 s, below) and the records' restructure (1g, `records.host`
8.0 -> 1.9 s), which has no switch. In gate 796641 the m5 configuration alone (binary f7335345, before 1c-1g): 34.10 s against 36.33 s
with `CFG=old` on the same binary and 38.46 s for `t`.

| step | `t` | m5 binary, `CFG=old` | m5 binary |
|---|---|---|---|
| lane A kernels | 4.48 | 3.80 | 2.84 |
| lane M kernels | 4.53 | 3.55 | 2.73 |
| lane B kernels | 3.68 | 3.21 | 2.63 |
| sum | 12.68 | 10.56 | 8.20 |

### 2c. The prefill stage and the run (`bench/fable/solowire/s8_stages_e11.py`, wall s / core-s)

| stage | base (8e54be40) | E12 (b38601ad) |
|---|---|---|
| decode .1fq -> FASTQ + R11 histogram | 20 / 86.6 | 19 / 84.8 |
| STAR load, CR4 probe, start | 35 / 45.3 | 36 / 45.6 |
| **GPU prefill** | **59 / 300.0** (GPU util 26.6 %) | **37 / 299.2** (35.5 %) |
| STAR per-read host pass | 10 / 20.9 | 9 / 20.7 |
| stream into pileup, hand-off, pileup | 25 / 42.0 | 26 / 51.0 |
| non-host | 21 / 51.6 | 19 / 40.7 |
| export, QC, callers; demux, writers | 38 / 130.5 | 38 / 129.5 |
| **total** | **214 / 686.3** | **189 / 679.8** |

STAR's line for the prefill: "cache filled ... in 58 s (reading reads 16 s, GPU 38 s, cache inserts 4 s)" -> "in 37 s (reading reads
17 s, GPU 16 s, cache inserts 4 s)".

**Final gate 796888** (node g051; two runs each, interleaved; wall s / core-s):

| stage | 8e54be40 | merged engine e2b0000e | **E12 a43d4ff4** |
|---|---|---|---|
| decode .1fq -> FASTQ + R11 histogram | 25 / 88.4, 23 / 89.5 | 20 / 92.7, 20 / 93.8 | 20 / 93.3, 20 / 91.5 |
| STAR load, CR4 probe, start | 39 / 49.3, 39 / 48.8 | 3 / 10.6, 4 / 11.9 | 6 / 21.8, 5 / 19.3 |
| **GPU prefill** | 62 / 311.6, 65 / 311.6 | 61 / 316.1, 64 / 322.3 | **37 / 304.5, 38 / 304.7** |
| STAR per-read host pass | 10, 9 | 10, 9 | 8, 8 |
| stream into pileup, hand-off, pileup | 28, 27 | 27, 28 | 27, 27 |
| non-host | 20, 20 | 20, 20 | 20, 19 |
| export, QC, callers; demux, writers | 39, 41 | 39, 39 | 39, 40 |
| **total wall** | **228.1, 228.9** | **185.5, 188.9** | **162.0, 161.2** |
| core-s | 705.3, 704.8 | 673.8, 679.4 | 679.3, 668.8 |

E14's probe change is in both merged binaries (`STAR load, CR4 probe` 39 -> 3-6 s). E12's run spends 2 s more in that stage than the
merged engine's; the stage script splits it from the prefill at a log line, and the prefill's own clock (STAR's "cache filled ... in
37 s / 38 s") agrees with the table. The device phase on g051 (16.9 / 17.4 s) is ~1 s above g050's (16.2 s): the staged device -> host
copies ran at 5.2-5.6 GB/s there against 7.2-7.6 on g050 (`e2e.build` 1.4-1.5 s against 1.1), and one run's reads upload took 2.0 s.

After the engine, from the new log line (two runs): start 0.30 / 0.29, hand-off records 0.40 / 0.39, hand-off install 0.32 / 0.32, E10
products 1.67 / 1.59 (unmapped records written, level 5's read source, the RNA Q30 adjustment), cache inserts 0.00: 2.7 s. The base
binaries log "cache inserts 4 s" and build the 11,997 MB cache ("SEQCACHE: on, 11997 MB" appears only in the base runs and in gpuD,
which inserts).

## 3. Identity per output

**Gate (a): CPU engine vs E12 level 5 (796888, binary a43d4ff4 for both; `bench/star_accel/out_equiv.py` + the gate's checks).**

| outputs | result |
|---|---|
| count matrices and pileup products, tables, private/ VCFs and sites, non-host EM, rRNA/sex/ancestry calls, `cr4_probe` files (the 60 of E11 §2a) | identical |
| STAR Solo: `Barcodes.stats`, `Gene/Features.stats`, `Gene/Summary.csv`, `Gene/UMIperCellSorted.txt`, `Gene/raw/*`, `Gene/filtered/*` | identical (in the 60) |
| `Gene/CellReads.stats` | same 323,542 rows (sorted; file order differs, as between two CPU runs) |
| `cell_qc_metrics.tsv`, `read_qc_summary.json` | identical (in the 60) |
| `star_SJ.out.tab` | identical (126,996 junctions) |
| `star_Log.final.out` | mapping statistics identical |
| `nonhost_reads.fq`, `star_Unmapped.out.mate1/2` | same records (sorted sha256 c115c734..., 766e191f..., 31fc9ef1...; 428,468 and 3,355,559 reads); order differs |
| 2 files (in E11's gate: `provenance.json`, `ambient_contamination.json`) | timings/paths only (`out_equiv` counts them without naming them) |
| `read_stats.tsv`, `star_Log.std.out` | stage rows and timings |
| `pileup_stats.json` | `clip5p/probe/n_reads` 12,194 vs 12,000 and `bytes_public` |
| `summary.json` | `memory_tier` Medium vs Large |

`out_equiv`: **60 identical, 2 timing/path only, 7 differ**, the 7 of E11's gate (a) and of cpu vs gpuD.

**gpuD (default GPU path, binary a43d4ff4) vs E11's kept gpuD (796377, `$W/jobs/fable/solowire/results/gate-796377/gpuD/out`):** 60
identical, 2 timing/path only, 7 differ: `nonhost_reads.fq` and `star_Unmapped.out.mate1/2` absent from E11's copy, `bytes_public`,
the whitelist path in `provenance.json` (`solowire/bin-s5/` vs `prefill/bin/`), and the timings in `read_stats.tsv` and
`star_Log.std.out`. CellReads.stats: same rows; mapping statistics and `SJ.out.tab` identical.

**Level 5 against the binaries without E12 (796888):** E12 vs 8e54be40 and E12 vs e2b0000e: 63 identical, 3 timing-only, 3 differ
(`read_stats.tsv`, `star_Log.std.out`, `bytes_public`); CellReads.stats byte-identical in file order; the sorted record files
identical. 796641 and 796791 gave the same for the intermediate binaries (f7335345, b38601ad).

**Engine level (796791).** `SINGLET_ENGINE_GPU_DUMP` writes `mapDistinct`'s output: every record's sequence and blob with their
offsets, `readRecord`, the filter parameters and `chrStart` (E6's format). The six dumping runs (`CFG=old`, all on, and each of the four
switches off) gave the same 6,053,620,001 bytes, sha256 77d7bc20008166af.... The lanes split the work differently (the warp lane M maps
7 more sequences on the device: 31,550,994 vs 31,550,987, host lane 85,563 vs 85,570) and the records are the same.

**m5 harness (796639 fable_cc, 796640 fable_gate, commit 0b2e3e94).**

| variant | identical | M distinct reads/s |
|---|---|---|
| m5b defaults | 1,991,926 / 1,991,926 | 2.51 |
| m5 defaults | 1,991,926 / 1,991,926 | 3.07 |
| `--best-order --p3-maxreg-a 64 --batch-b 131072` | 1,991,926 / 1,991,926 | 3.54 |
| + `--warp-m --wsort --wthreads 128` (E12's lanes) | 1,991,926 / 1,991,926 | 3.64 |

## 4. Surprises

- **The largest step was not on the GPU.** `records.host` (8.0 s of 38.5) was the host assembly of `Output`'s arrays, which the
  instrumentation showed for the first time (E11's breakdown had "~9 s unaccounted"). Growing four `std::vector`s per chunk meant a
  serial zero-fill of ~4 GB and a reallocating copy. One allocation without zero-fill and the parallel copies: 1.9 s. The part each
  piece contributed was not measured separately.
- **The engine compiled a stale `stitch.cuh`.** `src/engine_gpu/stitch.cuh` was a copy of m5 811cbe88, without E5's compact state
  (S3, TR32, ENC1) and E7's flat seeds (flatqs8b). Sharing m5's file made the old kernels 17 % faster (12.7 -> 10.6 s) before any lane
  change, and keeps the harness and the engine on one source from now on.
- **The host lane disappears.** 85,563 reads, 42 core-s, mapped by 17 pool threads during lane B's 2.6 s and the record preparation;
  the wait after the lanes is 3 ms. The lane kernels did not slow down (lane B 2.63 s in both).
- **The staged reads upload varies between runs** (2.5-11.9 GB/s, `reads.h2d` 0.55-3.3 s). It competes with the index thread for host
  memory bandwidth and PCIe, and the node runs other jobs. The index itself ran at 5.0-7.6 GB/s (2.6-3.7 s), so the first second of
  the device phase is bounded by host memcpy into page-locked memory, not by PCIe.
- **The dedup's quadratic walk cost nothing here** (`k_groups` 0.06 s): the gain of 1c is the host passes (3.4 s), not the kernel.
- **Lane B fills the device.** With 131,072 reads per batch, lane B's Work is 72 GB; the free-memory fit gives 114,688-116,736 per batch
  (63-64 GB) and the lanes' peak is 90.4 GB of 93.6.

## 5. What remains

- **After the engine** (2.7 s): E10's products step is 1.6 s (writing 3.3 M unmapped records, level 5's read source, the RNA
  adjustment), the hand-off 0.7 s.
- **The lane kernels are now half of the device phase** (8.2 of 16.2 s). Running two lanes at once (lane A of chunk 2 with lanes M/B
  of chunk 1, ~16 % in m5's measurements) was not done: lane B takes all free memory, and `runLane` drives the legacy stream with
  synchronous copies, so concurrent lanes need explicit streams through `runLane`, thrust and the per-batch scans, and a memory split.
- **The index upload** (17.6 GB, 2.6 s) ends 0.6 s after the dedup. It could start before the reads are read (16 s of host work), but
  then the device is held during the read phase, which matters when samples share a GPU (`lockDevice`).
- **The host records at level 5.** `records.prep` + `records.copy` (1.9 s) build `Output::seq/blob`, which level 5 uses only for sizes:
  E8 builds the records on the device. Skipping them needs a fallback when E8's build fails (the hand-off file path reads them).
- **E8/E10 into the region** (0.74 s at 7 GB/s): the parallel host copy out of the staging buffers bounds it.
- **Smaller host steps:** `lanes.host` 0.47 s (serial compactions of the lanes' statuses), `readRecord` 0.42 s, lane B's 64 GB
  allocation and memset 0.26 s.
- **Reading reads** (16-17 s) is E13's.

## 6. Files touched outside `src/engine_gpu/`, and why

| file | change | why |
|---|---|---|
| `include/singlet/engine_gpu.h` | `Output::seq`, `seqOff`, `blob`, `blobOff` are `NoInitVector`s | the one allocation of the records without a zero-fill (1g); the only user, `SeqCacheGpuPrefill.cpp`, reads them through `data()`/`size()`/`[]` |
| `src/star/SeqCacheGpuPrefill.cpp` | the cache re-created when results are inserted; the post-engine step times | 1h |
| `bench/fable/prefill/jobs/{build,gate}.sbatch` | build and gate jobs (from E11's) with binary and environment per mode, E6 dumps hashed per mode, `KEEPOUT` per mode | deliverable |
| `docs/gpu-engine/fable/E12-prefill.md` | this report | deliverable |

`src/engine_gpu/stitch.cuh` is now 5 lines that include `bench/gpu/m5/stitch.cuh`; m5's file is unchanged. Nothing in
`bench/gpu/m5/` changed. The merge of `origin/gpu/fable-engine` (E14) brings E14's files; E12 does not touch them.

## 7. Jobs, commits, how to run

**Branch `gpu/fable/prefill`** (from 8e54be40):
- 62ca9be3: per-step timers; build and gate jobs.
- f7335345: the m5 configuration (shared `stitch.cuh`), `CFG=old`.
- 0b2e3e94: gate `DUMP="mode ..."`.
- 3859f65a: dedup on the device, host-lane pool, staged uploads and the index thread.
- 1994295b: `k_heads` + `k_collide`.
- b38601ad: records' assembly, staged device -> host copies, `NoInitVector` outputs; gate dumps hashed per mode.
- eba63af0: the sequence cache sized on insert; post-engine step times.
- 0fc12e69: gate `KEEPOUT="mode ..."`.
- a43d4ff4: merge of `origin/gpu/fable-engine` e2b0000e (E14).
- This document is the next commit.

**Builds** (short-gpu without a GPU, ~5 min; `$W/jobs/fable/prefill/bin/singlet-<variant>`, commit in `commit-<variant>`):

| job | variant | commit |
|---|---|---|
| (earlier) | base / t / m5 | 8e54be40 / 62ca9be3 / f7335345 |
| 796673 | e12b | 3859f65a |
| 796701 | e12c | 1994295b |
| 796788 | e12d | b38601ad |
| 796805 | e12e | eba63af0 |
| 796884 | e12f | a43d4ff4 (merged) |
| 796885 | eng | e2b0000e (`gpu/fable-engine` with E14, without E12) |

**Gates** (short-gpu, H100 NVL, T = 18, node g050 unless said):

| job | binaries | modes | result |
|---|---|---|---|
| 796639 / 796640 | m5 from 0b2e3e94 | fable_cc + fable_gate (debug-cpu / debug-gpu) | 4 variants, all 1,991,926 / 1,991,926 |
| 796641 | base, t, m5 | `gpuE5_base gpuE5_t gpuE5_m5old gpuE5_m5` | before-table; base vs m5 identical |
| 796791 | base, e12d | `gpuE5_base gpuE5_d` + 6 switch runs, 6 dumps | after-table; switches; dumps identical |
| 796888 (g051) | e12f, base, eng | `cpu gpuD gpuE5_base gpuE5_eng gpuE5 gpuE5_base2 gpuE5_eng2 gpuE5_2`; pairs cpu:gpuE5, E11's gpuD:gpuD, base:gpuE5, eng:gpuE5 | gate (a), timings |

**Results:** `$W/jobs/fable/prefill/results/gate-<job>/` (`summary.txt`, `equiv-*.txt`, per run `starlog_engine.txt`, `star_Log.out`,
`stages.txt`, `CellReads.stats`, `monitor.tsv`). Logs: `$W/logs/e12-{gate,build}-<job>.out`.

**Run:**
- build: `sbatch --export=ALL,VARIANT=<v>[,REV=<commit>] bench/fable/prefill/jobs/build.sbatch`;
- gate: `MODES="cpu gpuE5" BIN_cpu=singlet-<v> BIN_gpuE5=singlet-<v> PAIRS="cpu:gpuE5" sbatch --export=ALL bench/fable/prefill/jobs/gate.sbatch`;
  `ENV_<mode>="K=V ..."` adds environment, `DUMP="<modes>"` compares the E6 dumps, `KEEPOUT="<modes>"` keeps those outputs.

**Switches** (all default on with `--engine gpu`; any of them off, or `SINGLET_ENGINE_GPU_CFG=old` for all):
`SINGLET_ENGINE_GPU_UPLOAD=pageable`, `_D2H=pageable`, `_DEDUP=host`, `_HOSTLANE=after`, `_WARPM=0`, `_NB/_NBM/_NBB=<reads>`.
`star_Log.out` shows `engine_gpu: steps (s): ...`, `engine_gpu: E12 lanes (new): ...`, `engine_gpu: index upload ... staged ...`,
`engine_gpu: E12 device -> host copies staged: ...` and `engine_gpu: E12 host lane during the device lanes: ...`.
