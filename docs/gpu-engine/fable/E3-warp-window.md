# E3 warp-window: the windows of a read stitched across a warp (wave 1, question Q6)

Branch `gpu/fable/warp-window` (from `gpu/fable-engine` cf92cdb9). Code: `bench/gpu/m5/stitch.cuh` (warp lanes section),
`bench/gpu/m5/m5.cu` (`k_p3a`, `k_p3w`, flags `--warp-m --warp-b --warp-a G --wmin-a T --wassign --wsort --wb-cfg
--wthreads --no-wstats --host-emul --tag`), jobs `bench/gpu/m5/jobs/e3_compile.sbatch` (NAME=binary),
`bench/gpu/m5/jobs/e3_run.sbatch` (RUNS=variants, NCUS=Nsight Compute profiles; baseline and variants in one job,
identity checked in each), `bench/gpu/m5/jobs/e3_ncu_source.sbatch`. Results JSON: `bench/fable/warp-window/`.

STATUS: done. Exact (100% identity in every variant) but **no speed-up**: the best warp configuration ties the baseline
(lanes M+B 568.7 vs 563.5 ms); the gate (M+B <= 0.5x) is not met. Section 3.3 says why. After the ballot fix (3.4) lane B
warp-stitched with the in-kernel assignment is 5.7% faster than the baseline lane B, lane M 5% slower. The branch carries
the integration head (merge 80f366fc, all build flavours compile; E3 probes only with `-DM5_E3STATS`); the run on the
merged code (job 795844) was queued when this was written (3.5).

## 1. Design as built

Lanes M and B keep phase 1 (`k_p1`, thread per read: seeds, windows, flanks) and the coalesced assignment scan (`k_p2`, warp
per read) unchanged. Phase 3 (`k_p3`, thread per read: assignment, stitching of every window, selection, encoding) is split
into two kernels when a lane runs warp-stitched:

1. `k_p3a`, thread per read: the assignment (`assignHits`, STAR's order from the scan's marks), exactly as in `k_p3`.
2. `k_p3w`, one warp per read: lane `l` stitches the read's `(32 j + l)`-th window that has aligns (windows without aligns
   are skipped as in STAR; a read with more than 32 such windows takes several rounds), with the unchanged
   `stitchWindowAligns` / `finalizeTranscript` recursion on a per-lane context (`LaneWork`: the explicit stack, the window's
   transcript slots and their pointer permutation, the finalisation scratch) in local memory, which the hardware interleaves
   per lane (coalesced when lanes are converged). After each round the warp merges the round into the read's `keep[]` with
   shuffles (arg-best over the round's windows, a prefix sum of the transcripts each lane keeps, each lane copies its own
   into `keep[]`); lane 0 then runs `multMapSelect`, `mappedFilter` and the encoding (`selectEncode`, split out of
   `mapFinish` unchanged).

Variants behind flags (all in `k_p3w`, the baseline kernels untouched):

- `--wassign`: the assignment moves into `k_p3w` too (`warpAssign`), split over the lanes by window: window `iW` belongs
  to lane `iW % G`. The group walks the read's marked SA entries `G` at a time in STAR's order; lane `q` decodes entry `q`
  of the chunk (SA lookup, strand, sjdb split, window of each part) into shared memory; then each lane applies, in entry
  order, the parts that fall into its own windows (donor before acceptor, as `assignEntry`). `assignAlignToWindow` reads
  and writes only the target window's state (`WA`, `nWA`, `WALrec`, `WlastAnchor`) plus `nW`/`mapMarker` on
  `MARKER_TOO_MANY_ANCHORS_PER_WINDOW`; each window therefore sees exactly STAR's sequence of calls. After the marker
  (`nW = 0`, nothing observable follows in STAR) the chunk's later entries may still be applied by other lanes; if one of
  them overflows a capacity the read goes to the next lane, which is exact.
- Groups of `G` lanes per read (`Grp<G>`: ballots relative to the group, shuffles of width `G`): `G = 32` for lanes M and
  B, `G = 8 / 16 / 32` for lane A (`--warp-a G`). With `--wmin-a T` a lane A batch is sorted by window count and only the
  reads with `>= T` windows go to `k_p3w`; the rest run the baseline `k_p3` (thread per read) on the same sorted
  permutation (`k_p3` takes a permutation already).
- `--wsort`: `k_p3w` takes the batch's reads by window count, most first; `--wthreads`: block size; `--wb-cfg`: lane B's
  per-lane capacities.

`stitchWindowAligns` and `finalizeTranscript` are now templates on the stitching context (`Work<C>` or `LaneWork`), with no
change for `Work<C>`; `mapFinish` is `stitchWindows` + `selectEncode` with the same code. The serial lanes and the host lane
compile to the same logic as before.

Per-lane capacities: lane M's lane context has lane M's own capacities (8 exons, 8 aligns per window, 8 transcripts per
window), so lane M never falls back. Lane B's (configuration 0: 8 exons, 16 aligns, 8 transcripts; configuration 1: 12, 32,
16) are smaller than the read's (20, 50, 32): a read with a window above them is stitched by the serial loop
(`mapFinish`) on lane 0 of its warp, from the same assignment state. The local memory a lane context needs (`sizeof`):
lane M 14,008 B, lane B configuration 0 19,832 B, configuration 1 49,984 B, lane A 19,832 B.

After the merge of the integration head (commit 80f366fc: E1's tail probe, E2's census, E5's exact compact variants on by
default, E6's records) the lanes share E5's layouts: `LaneWork` has `Work`'s M5_FIN1 slot pool (one spare slot as the
finalisation scratch instead of `fin`), the M5_SLOT0 window start (only `maxScore` of slot 0 is written), the M5_TR32
exon rows (`copyFromAny` copies whole rows), and with M5_KEEPIX the warp merge places `keep[p]` at pool entry `NWTR + 2 + p`
(`keepAt`). Under M5_CENSUS the warp kernel fills no census records (the census describes the baseline kernels). The E3
statistics (`--wstats` counters and histograms, the `clock64` probes of `k_p3w`, `warpStitch`, `k_p3a` and the baseline
`k_p3`) compile only with `-DM5_E3STATS`; the default build has none (`--no-wstats` is then a no-op).

## 2. Exactness: why every recording decision is identical

The design note (study Part I 4.4) proposed two passes: pass 1 computes each window's maximum finalised score, a prefix max
gives STAR's `maxScoreMate` at each window's start, pass 2 records with it. While porting it I found that for the reads this
port supports the coupling does not exist, so one pass suffices:

- In `finalizeTranscript` (STAR `stitchWindowAligns.cpp`, the `iA >= nA` branch), for a single-fragment transcript
  `maxScoreMate[iFrag] = max(maxScoreMate[iFrag], Score)` runs just before the recording test
  `Score + range >= wTr[0]->maxScore || Score + range >= maxScoreMate[iFrag]` (variationAdjust is 0: no variation).
- `maxScoreMate` is reset to 0 per read (`ReadAlign::resetN`) and only grows; so at the test it is
  `max(0, every Score finalised so far in the read, including this one)`.
- `wTr[0]` is either the window's initial transcript (`trA` with `maxScore = 0`, written into slot 0 before the recursion) or
  a transcript recorded earlier *in this window*, whose `maxScore` is its own finalised `Score` (`trA.maxScore = Score`
  before `*(wTr[iTr]) = trA`). Pointers at positions `>= nWinTr` are never read except `wTr[0]` when `nWinTr = 0`, which
  holds the initial transcript or a transcript of this window removed as a subset.
- Hence `maxScoreMate >= wTr[0]->maxScore` at every test, and with `range >= 0` (the driver rejects a negative range)
  `Score + range >= maxScoreMate` implies `Score + range >= wTr[0]->maxScore`. The test is exactly
  `Score + range >= wTr[0]->maxScore`, which reads only the window's own state.

So each lane runs its window with its own `maxScoreMate` starting at 0 (any value `>= wTr[0]->maxScore` gives the same
decisions), and the recording of every window is identical to STAR's. For paired-end reads (per-mate `maxScoreMate`
against a combined `wTr[0]`) the argument fails and the two-pass scheme would be needed; the port does not support mates.
The host emulation (`--host-emul`) checks this on every read: it carries STAR's read-wide `maxScoreMate` across the windows
alongside the window-local one and counts tests where they would decide differently (`shadow_diff`) and tests where the
`maxScoreMate` disjunct alone would record (`mate_only`).

The remaining cross-window state of STAR's window loop and how it is reproduced:

1. `trBest`: updated after each window when the window's best transcript `t0 = wTr[0]` has a higher score, or the same score
   and a smaller `gLength` (strict). Starting from `trInit` (score 0, gLength 0) it ends as the first window, in window
   order, whose `t0` is lexicographically best by (score desc, gLength asc). The warp computes it with a shuffle arg-best on
   (score, gLength, window index) per round and the same strict comparison against the running best across rounds.
2. `keep[]` (the port's representation of STAR's `trAll`, `multMapSelect` input): after window `w` the serial loop holds
   exactly the transcripts of windows `<= w` with `maxScore + range >= trBest's score after w`, in window order then slot
   order (a window's transcripts are kept against `runBest = max(trBest, t0)` = the new best, and the earlier ones are
   compacted when the best rises). `multMapSelect` then keeps those with `maxScore + range >= final best`. The warp keeps a
   superset in the same order (appends each round's transcripts above the running best; compacts only when `keep[]` would
   overflow) and `multMapSelect` applies the same final filter, so the selected list, its order and the primary index
   (`trBestKeep`, found as the best window's slot-0 entry) are identical. Consequence: lane-local `keep[]` overflow is rarer
   than in the serial loop (the capacity applies to fewer entries), so a few reads finish in lane M/B instead of moving to
   the next lane, with the same result.
3. The `trNtotal + alignTranscriptsPerWindowNmax >= alignTranscriptsPerReadNmax` break: unreachable here (at most
   `NW x min(NWTR, alignTranscriptsPerWindowNmax)` = 512 (M) or 2,048 (B) transcripts + 100 < 10,000); the driver checks the
   inequality at start-up and refuses the warp lanes otherwise.
4. The step budget: the serial loop overflows when the read's total recursion steps exceed the budget. The enumeration of a
   window does not depend on recording, so its step count is the same in a lane; each lane starts its counter at the read's
   total before the round (and aborts early once over the budget), the warp sums the round, and the read overflows iff the
   total exceeds the budget: the same reads overflow as in the serial loop.
5. Capacities: an overflow of the read's own capacity in any window (exons, transcripts per window) overflows the read as in
   the serial loop (the next lane re-runs it from scratch); a lane capacity below the read's sends the read to the serial
   loop on lane 0 (`WR_FALLBACK`), which is the baseline code.

The enumeration inside a window, the transcripts' fields and every tie rule are the unchanged port; the identity gate below
compares all 1,991,926 results byte for byte with STAR.

## 3. Results

Set: pbmc1k_v3, every 16th distinct cDNA sequence = 1,991,926 distinct reads (4,118,999 reads); STAR truth regenerated
in each job (stardump M5). One H100 NVL (g051), median of 3 timed runs after a warm-up. Lane times are the GPU time of
the lane (phases 1-3 of every batch); M+B is the quantity of the gate. Defaults: batches A/M/B 524,288 / 393,216 /
65,536, budgets 1024 / 2048 / 2048 steps; "big" = 1,048,576 / 524,288 / 131,072 (E1's best).

Jobs (all `#SBATCH --nice=10000`; logs `$W/logs/<name>-<id>.out`, JSON `$W/jobs/fable/warp-window/results/`, copied to
`bench/fable/warp-window/`): compiles 795255 (ca8b980f, failed, section 4), 795258 (1ea23874), 795276 (54e934ff), 795311
(41a4187d = the measured binary), 795731 (8288c0a0); runs 795259 (no output, section 4), **795338** (core variants, lane A),
**795637** (statistics off, best configuration, block sizes, batch sizes), **795580** (Nsight Compute, debug-gpu), 795739
(ballot fix), 795654 / 795662 (Nsight report exports).

Identity: **every run of every variant, 1,991,926 / 1,991,926 distinct reads identical to STAR** (4,118,999 / 4,118,999
weighted), mapped and unmapped. The host emulation of the warp algorithm (`--host-emul`, job 795338, `wasg`) reproduces all
1,991,926 results and reports 0 shadow `maxScoreMate` disagreements and 0 `maxScoreMate`-only records over every
finalisation of lanes M and B: the window-independence argument of section 2 holds on the data.

### 3.1 Lanes M and B (job 795338 = binary 41a4187d, job 795637 = the same binary, statistics off where marked `ns`)

| variant (job) | flags beyond the defaults | lane M ms | lane B ms | M+B ms | M+B / base | GPU ms (A+M+B) | M distinct reads/s |
|---|---|---:|---:|---:|---:|---:|---:|
| base (795338) | (baseline lanes, unchanged) | 311.4 | 252.9 | **564.3** | 1.000 | 793.7 | **2.510** |
| base (795637) | | 310.4 | 253.1 | 563.5 | 1.000 | 792.7 | 2.513 |
| warp (795338) | `--warp-m --warp-b` (assignment `k_p3a`) | 332.8 | 287.5 | 620.2 | 1.099 | 850.2 | 2.343 |
| warpns (795637) | + `--no-wstats` | 330.9 | 283.0 | 613.8 | 1.089 | 843.1 | 2.362 |
| wasg (795338) | `--warp-m --warp-b --wassign` | 379.6 | 292.8 | 672.4 | 1.192 | 902.3 | 2.208 |
| wasgsort (795338) | + `--wsort` | 348.1 | 264.1 | 612.2 | 1.085 | 841.4 | 2.367 |
| wasg1 (795338) | `--wassign --wb-cfg 1` | 380.5 | 265.4 | 645.9 | 1.145 | 875.2 | 2.276 |
| wasgns (795637) | `--wassign --wsort --wb-cfg 1 --no-wstats` | 340.7 | 243.9 | 584.6 | 1.038 | 814.3 | 2.446 |
| best (795637) | `--wsort --wb-cfg 1 --no-wstats` (k_p3a) | 318.7 | 263.4 | 582.1 | 1.033 | 811.3 | 2.455 |
| bestt32 (795637) | best, `--wthreads 32` | 317.8 | 262.3 | 580.0 | 1.029 | 810.0 | 2.459 |
| **bestt128 (795637)** | best, `--wthreads 128` | 311.6 | 257.1 | **568.7** | **1.009** | 798.2 | 2.496 |
| bestt256 (795637) | best, `--wthreads 256` | 320.6 | 263.2 | 583.8 | 1.036 | 813.1 | 2.450 |
| basebig (795637) | big batches | 313.5 | 190.5 | 504.1 | 0.895 | 723.3 | 2.754 |
| bestbig (795637) | best + big batches | 316.9 | 219.9 | 536.9 | 0.953 (1.065 vs basebig) | 756.5 | 2.633 |
| basebig (795338) | big batches | 313.2 | 192.6 | 505.8 | 0.896 | 725.7 | 2.745 |
| wasgbig (795338) | `--wassign` + big | 378.6 | 270.4 | 649.0 | 1.150 (1.283 vs basebig) | 868.6 | 2.293 |
| wasgsortbig (795338) | `--wassign --wsort` + big | 346.6 | 241.2 | 587.8 | 1.042 (1.162 vs basebig) | 807.4 | 2.467 |
| basesmall (795637) | batches M/B 131,072 / 32,768 | 364.0 | 328.5 | 692.5 | 1.229 | 922.1 | 2.160 |
| bestsmall (795637) | best + the same small batches | 351.8 | 309.1 | 660.9 | 1.173 (0.954 vs basesmall) | 890.1 | 2.238 |

**The gate is not met.** The best warp configuration is on par with the baseline (M+B 568.7 vs 563.5 ms, 1.01x) at the
default batches, 6.5% slower at the big batches and 4.6% faster only at small batches (where the batch tail, which the
warp shortens, weighs more). M5 rate before / after: 2.51 / 2.50 M distinct reads/s (default batches), 2.75 / 2.63 (big).

Phase split (ms, run 1; p1 = seeds + windows, scan = `k_p2`, assign = `k_p3a`, p3/warp = `k_p3` or `k_p3w`):

| lane | base: p1 / scan / p3 | best: p1 / scan / assign / warp | wasg: p1 / scan / warp (assignment inside) |
|---|---|---|---|
| M | 73 / 18 / 220 | 73 / 19 / 18 / 210 | 73 / 17 / 290 |
| B | 39 / 7 / 206 | 39 / 7 / 40 / 177 | 39 / 8 / 246 |

The baseline's `k_p3` spends 14% (M) / 12% (B) of its thread cycles in the assignment, so its stitching + selection is about
190 / 180 ms; the warp kernel's is 210 / 177 ms: **stitching the windows across the lanes is not faster than stitching
them in a thread per read**, and the separate `k_p3a` (72 registers, high occupancy) is the cheapest way to do the
assignment (18 / 40 ms).

### 3.2 Lane A (groups of 8/16/32 lanes, reads with >= T windows; job 795338, 795637)

| variant | lane A ms (base 229.4) | lane A phase 3 (base 116.4) | group reads | lanes active |
|---|---:|---:|---:|---:|
| a16w6: G 16, T 6 | 293.5 | 179.9 | 317,683 | 62.5% |
| a8w4: G 8, T 4 | 284.3 | 174.7 | 447,930 | 73.7% |
| a8w8: G 8, T 8 | 306.5 | 193.0 | 225,810 | 77.8% |
| a32w10: G 32, T 10 | 299.0 | 186.0 | 159,491 | 39.7% |
| a16all: G 16, every read | 363.4 | 250.5 | 1,477,367 | 21.6% |
| a16w6 + big | 254.0 (basebig 219.9) | 144.6 (110.4) | 317,683 | 62.5% |
| a16w6ns (`--wsort`, stats off) | 290.1 (base 229.2) | 177 | | |

Lane A's few-window reads lose 25-58%: no lane A variant is a win.

### 3.3 Why: Nsight Compute (job 795580, debug-gpu; the first batch of each lane, `--set full`)

| | `k_p3` lane M (baseline) | `k_p3w` lane M (`--wassign`) | `k_p3w` lane B (`--wassign`, cfg 0) |
|---|---:|---:|---:|
| reads in the launch | 393,216 | 393,216 | 65,536 |
| duration | 154.7 ms | 210.9 ms | 170.1 ms |
| registers / thread, stack | 94, 320 B | 124, 14,320 B | 124, 21,488 B |
| occupancy theoretical / achieved | 31.3% / 25.3% (16.2 warps) | 25% / 17.0% (10.9 warps) | 25% / 16.6% |
| warp instructions executed | 21.69 G | 22.84 G | 14.50 G |
| avg. active threads per warp instruction | 2.29 | **7.02** | 4.48 |
| thread instructions (product) | 49.7 G | **160.3 G** | 65.0 G |
| issued warps per scheduler per cycle | 0.16 | 0.12 | 0.11 |
| no eligible warp | 83.9% | 87.8% | 89.4% |
| L1 hit rate | 76.7% | **48.5%** | 56.8% |
| L2 hit rate | 76.6% | 68.8% | 75.7% |
| DRAM read + write | 72.6 + 37.9 GB | 88.4 + 59.1 GB | 45.5 + 30.0 GB |
| global ld / st instructions | 1.96 / 0.94 G | 0.92 / 0.11 G | 0.88 / 0.28 G |
| local ld / st instructions | 0.05 / 0.05 G | 0.93 / 0.87 G | 0.55 / 0.51 G |
| branch efficiency | 96.3% | 95.9% | 96.9% |

Reading:

1. SIMT efficiency rose 3x as intended (2.29 -> 7.02 active lanes per instruction in lane M), but the warp kernel executed
   3.2x the baseline's thread instructions for the same reads. Part of it is the `--wassign` assignment as first built:
   every lane scanned all 32 entries of every chunk with a marked entry to find its own (a 32-way loop per chunk); in the
   timings it cost ~65 ms in lane M (wasg 290 vs warp 225 ms of kernel) against 18 ms for the thread-per-read `k_p3a`.
   Commit 8288c0a0 replaces the scan with ballots on the owner-lane bits (job 795739 measures it, see 3.4). The rest is the
   warp-collective overhead per round (window list, arg-best, scan, the per-round statistics) executed by all lanes.
2. The lane contexts live in local memory (14 KB per lane in M, 20-50 KB in B): the stitching's transcript traffic moved
   from global (`Work`) to local memory (0.9 G local loads and stores per batch in M), the L1 hit rate fell from 77% to
   48% (the hot slots of 32 lanes x ~11 warps per SM do not fit) and DRAM traffic rose 34%.
3. 124 registers (the warp driver plus the unchanged recursion) cap the occupancy at 25% (17% achieved, 10.9 warps vs
   16.2): with every warp waiting on memory (no eligible warp 88% of the time) fewer warps hide less latency.
4. The windows are tiny (E2: 81% have one align and ~5 steps; here lane M: 34 windows/read, 82% of windows with one align,
   94% with <= 16 steps, 1 transcript in 92%): the per-window fixed cost (window init, slot copies, finalisation, the
   merge) dominates, and spreading it over lanes does not remove it, it only interleaves it.
5. Lane B: 7.3% of its warp reads fall back to the serial loop on lane 0 (a window above the lane capacity, configuration
   0), and that serial fallback takes 24% of the lane-0 cycles outside the assignment (1.0e11 of 4.2e11); configuration 1 (12 exons, 32 aligns,
   16 transcripts; 50 KB per lane) cuts the fallbacks to 1.9% and lane B's warp phase from 246 to 219 ms.
6. Sorting the reads by window count (`--wsort`) helps the warp kernel 11% (as E2 predicted for slot use), block size
   128 helps 2%; neither changes the conclusion.

Register / spill report (ptxas, commit 41a4187d, `-O3 -arch=sm_90`): `k_p3w` lane M (LaneWork<8,8,8>) 124 registers,
14,320 B stack, 0 spill; lane B cfg 0 (<8,16,8>) 124 / 21,488 B / 0; cfg 1 (<12,32,16>) 124 / 51,648 B / 0; lane A G = 8 /
16 / 32 (<8,16,8>) 124-126 / 20,400 B / 0; `k_p3a` 72 / 0 B; baseline `k_p3` A/M/B 94 registers, 576 / 320 / 1,664 B, 0
spill. Full reports: `$W/jobs/fable/warp-window/bin/ptxas_<sha>.txt` and `_summary.txt`.

### 3.4 After the ballot fix (job 795739, binary 8288c0a0, before the merge)

| variant | flags beyond the defaults | lane M ms | lane B ms | M+B ms | M+B / base | GPU ms | M distinct reads/s |
|---|---|---:|---:|---:|---:|---:|---:|
| base | | 315.5 | 254.4 | **569.9** | 1.000 | 802.9 | **2.481** |
| wasgbest | `--warp-m --warp-b --wassign --wsort --wb-cfg 1 --no-wstats --wthreads 128` | 331.2 | **239.9** | 571.1 | 1.002 | 804.5 | 2.476 |
| best128 | the same with `k_p3a` (no `--wassign`) | 315.3 | 260.6 | 575.8 | 1.010 | 809.3 | 2.461 |
| wasgstats | wasgbest with the statistics on | 337.7 | 242.9 | 580.6 | 1.019 | 813.3 | 2.449 |
| basebig | big batches | 323.9 | 202.1 | 526.0 | 0.923 | 746.5 | 2.668 |
| wasgbestbig | wasgbest + big batches | 331.2 | 212.0 | 543.3 | 0.953 (1.033 vs basebig) | 763.6 | 2.609 |

Identity 1,991,926 / 1,991,926 in every variant. The ballot ownership brought the in-kernel assignment from the worst to
the best option for lane B: wasgbest's lane B (239.9 ms) is 5.7% under the baseline's (254.4), its warp kernel 192 ms
against `k_p3`'s 208 (assignment included in both). Lane M stays 5% over (331.2 vs 315.5). Against job 795637's
`wasgns` (the same flags before the fix, 64 threads per block) M+B fell from 584.6 to 571.1 ms. Statistics (wasgstats):
lane M 34.3 windows per read, 71.9% of the lanes busy per round, 237 steps per read, 43 + 895 reads to lane B (exons/keep
capacity); lane B 75.0 windows per read, 80.5% busy, 584 steps per read, 1,005 + 601 of 85,689 reads to the serial loop
(window above 32 aligns, lane capacity). The combination that the lanes suggest, lane M as the baseline and lane B
warp-stitched, would be 315.5 + 239.9 = 555.4 ms (0.975x); it is measured directly on the merged code (3.5, `mixed`).

### 3.5 On the merged code (job 795844, binary 80f366fc = m5e3m)

Run job 795844 (short-gpu, `e3_run.sbatch`, BIN=m5e3m) measures base, best128, wasgbest, `mixed` (lane M baseline, lane B
`--warp-b --wassign --wsort --wb-cfg 1 --wthreads 128`), basebig, wasgbestbig and a second base (noise), plus Nsight
Compute of lane M (`k_p3` vs `k_p3w --wassign`). At the time of writing it was still queued behind higher-priority GPU jobs
of the same account (QOSMaxGRESPerUser, priority 1 with `--nice=10000`). Its results land in
`$W/jobs/fable/warp-window/results/e3_795844_<label>.json(.e3.json)` and the log in `$W/logs/e3-runm-795844.out`;
`python bench/fable/warp-window/e3_table.py <dir> 795844` prints the table above from them. The integration head's own
baseline for comparison: job 795737, lanes A / M / B 212.6 / 268.2 / 222.7 ms, 2.83 M distinct reads/s.

Register / spill report of the merged default build (compile job 795831, ptxas `-O3 -arch=sm_90`, no E3 probes):

| kernel | lane A | lane M | lane B |
|---|---|---|---|
| `k_p3` (baseline) | 79 regs, 624 B stack | 79, 368 B | 80, 1,712 B |
| `k_p3w` | 95-96 (G 32/16/8), 12,720 B | 95, 8,944 B (LaneWork<8,8,8>) | 95, 13,808 B (cfg 0) / 32,144 B (cfg 1) |
| `k_p3a` | 64, 0 B | 64 | 64 |
| `k_p1`, `k_p2` | 95, 42 | 95, 42 | 95, 42 |

No spill anywhere. Against the measured pre-merge binary (41a4187d): `k_p3w` 124 -> 95 registers and lane M's stack
14,320 -> 8,944 B (E5's 32-bit exon rows and FIN1 slots shrink the lane contexts 38%), so its register-limited occupancy
rises from 25% to 31% (5 blocks of 128 threads per SM); the baseline `k_p3` is back at 79-80 registers (37.5%) once the
E3 probes are out of it. With `-DM5_E3STATS` `k_p3a` takes 72 registers (job 795831's flavour check: e3stats, tail, census,
baseline and keepix all compile).


## 4. What failed on the way, and how it was found

- First compile (job 795255): `uint` ambiguous in the host emulation (`m5::uint` vs the system typedef); fixed in 1ea23874.
- The first run job (795259) produced nothing: the job script copied the binary to `$SCR/m5`, the directory the STAR truth
  is written to (`stardump exit=3`, then `m5: t.params.txt: Not a directory` for every variant). Fixed in 2c35a2f1 (the
  binary goes to `$SCR/m5e3`; the job stops if the truth is missing).
- The warp design itself: correct on the first measured run (100% identity, host emulation 0 disagreements), but not
  faster. Found by the lane timings (job 795338), then attributed with the warp-cycle counters (stitch rounds vs merge vs
  select vs assignment vs fallback, lane 0 of each warp) and Nsight Compute (job 795580, raw metrics exported by job
  795662). The CUDA-source page of the reports came out without metrics (job 795654), so no per-line attribution.
- The in-warp assignment's 32-way ownership scan (an obvious O(32) redundancy once the thread-instruction count was
  seen); replaced by ballots (8288c0a0).

- The E3 statistics were compiled into the default build at first, including two `clock64` probes and atomics in the
  baseline `k_p3` (job 795311's report: `k_p3` 94 registers where the integration head has 80, E4's finding). After the
  merge they are behind `-DM5_E3STATS` (default `k_p3` 79-80 registers again).
- The merge (80f366fc): 5 conflicts in `stitch.cuh` and 13 in `m5.cu`, all kept from both sides; `e3_compile.sbatch
  CHECK=1` compiles the e3stats / tail / census / baseline / keepix flavours next to the default one (job 795831: all ok).

## 5. Next steps

1. Do not pursue window-per-lane stitching as built for lanes M/B: at best it ties the thread-per-read kernel. The
   per-window cost is dominated by fixed work on transcript structs (E2, and here: 1 align and ~5 steps per window), so
   the lever is the size of that state, not the parallel schedule: combine with E5's compact transcript state, after
   which lane contexts small enough for registers / L1 could change the balance (the 3x SIMT gain is real; it is eaten by
   local-memory traffic and occupancy).
2. If revisited: a cooperative finalisation (lanes of a warp share one window's finalisation/recording, not one window
   each), lane contexts in shared memory for lane M (14 KB x 32 does not fit; a compact transcript would), and
   `__launch_bounds__` to trade registers for occupancy.
3. Lane A: no group variant wins (25-58% slower); pre-classification by window count works mechanically (`--wmin-a`
   split, one sorted permutation for both kernels) and could carry another phase-3 kernel.
4. Overlapping lanes and carrying lane A's phase-1 state into lanes M/B (E1/E2: 13.7% of M5) are independent of this
   result and remain the larger wins; bigger batches alone gave -10.4% here (M+B 504 vs 564 ms).
