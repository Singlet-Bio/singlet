# E4: phase-3 read order and batch scheduling in M5 (wave 1, questions Q5 and Q3)

Branch `gpu/fable/order` (from `gpu/fable-engine` cf92cdb9). Harness: `bench/gpu/m5/m5.cu` on the 2 M pbmc1k_v3 dump set
(every 16th distinct cDNA sequence, 1,991,926 sequences), Clipper H100 NVL, CUDA 13.3 image. Status: done (rounds 1-5; merged with `gpu/fable-engine` and measured, job 795837).

## Summary

- Every variant of every round is exact: 1,991,926 / 1,991,926 distinct reads (4,118,999 / 4,118,999 weighted) identical to
  STAR, 100% (rounds 1-5: 16 + 18 + 18 + 25 + 4 variants and 9 ncu runs).
- **Surprise: the harness itself**: rounds 1-3 ran a `k_p3` with the `--tail` timing code compiled in (a warp-collective
  convergence point after each read). It costs 14 registers (94 instead of the base's 80 in lanes A and M) and **18% of M5**
  (1028 vs 870 ms) even with the timers off. Round 4 runs the base kernel (tail code only under `--tail`): v0 870.4 ms =
  base build 874.7 ms. Comparisons inside rounds 1-3 are like for like; their absolute times are 18% high.
- **Order** (base kernel, round 4, vs v0 870.4 ms = 2.29 M distinct/s): the engine's prefix order is the best order for
  lanes A and B; every reordering of whole reads loses there: cost ascending (`--bin`) 1.31x slower, cost descending
  1.30x, control-flow signature 1.07x, genome locality 1.11-1.15x (round 1). Lane M is the exception: window count
  descending, stable (`--p3order-m nw-desc`), cuts lane M's phase 3 by 14% and M5 by **4.5% (831.0 ms, 2.40 M/s)**.
- **Scheduling**: a persistent queue inside a batch costs 2% (twice measured), one warp per block 0%, longest-first from an
  atomic counter 1.39x slower, warp-granular longest-first (prefix warps kept whole) +2%, E2's compaction of overflowed reads
  +1% (-0.9% at large batches). A batch lasts as long as its longest warp, which starts at t = 0; no order of whole reads
  shortens that.
- **Why**: `k_p3` issues 1.94 active threads per warp instruction (6% SIMT efficiency, ncu) with 98% branch efficiency: lanes
  idle because their reads are done while one or two reads of the warp go on. Sorting by cost (E2's key) raises active
  threads to 3.39 and cuts warp instructions by 43%, as E2 simulated, yet a lane-A k_p3 launch takes 2.2x longer: warps made only
  of heavy reads become the batch's critical path and the GPU idles (issue slots busy 12% -> 3%).
- **What pays is fewer batch barriers and registers**: lane B in one launch -23% on B; batches 512k / 384k / 128k -12.6%
  (E1: -11.9%); with the best order **727.8 ms (2.74 M/s, 1.20x)**; lane A's `k_p3` capped at 64 registers -5% on A. The
  per-lane best is one flag, `--e4-best`: **709.3 / 711.6 ms, 2.81 / 2.80 M/s, 1.22x** against v0 863.1 / 868.5 ms in the
  same job (round 5), at 80.6 GB of device memory.
- **After the merge with `gpu/fable-engine`** (E5's stitch variants on, job 795837): the gate configuration runs at 704.4 /
  703.2 ms (2.83 M/s); with `--best-order --p3-maxreg-a 64 --batch-b 131072` at **603.4 ms (3.30 M/s, 1.17x)**, identical.
  **`-DM5_NOCLK` is not a win**: without the `clock64()` phase counters lane M's phase 3 is 22% slower (208.9 vs 171.2 ms,
  same 79 registers), lane A +0.5%, lane B -2.6%; M5 +4.5% (734.9 / 736.3 vs 704.4 / 703.2 ms), twice measured.
- **Expected engine gain of the best ordering plus a persistent queue: about 1.03x** (order +4.5%, queue -2%), not the
  1.2-1.5x of the study's section 4.6. The 1.2x above comes from batch sizes, i.e. memory.

## Method

All variants are exact by construction: a read's result depends only on the read, every variant only changes which thread or
time slot handles which read, and every result is written by read index. Each variant still runs the byte-for-byte comparison
with STAR in the same job.

Options added to `m5` (all off by default; the default binary behaves as before). `--queue` and `--fused*` were dropped
in the merge with `gpu/fable-engine` (see "Merge with gpu/fable-engine and M5_NOCLK"); their code is in bf8746492a:

| option | what it does |
|---|---|
| `--presort` | the engine's order: stable sort of the truth reads by their first 32 bases (`engine_gpu.cu` `k_prefix_key`) before mapping. Without it the harness maps in truth-file order (24 sorted runs, one per STAR thread) |
| `--bin` | unchanged: after phase 1, phase-3 threads take the batch's reads sorted ascending by `nW << 40 \| nEntries` (cheap reads first) |
| `--p3order K` | other phase-3 keys, stable radix sort per batch: `cost-asc` (= `--bin`), `cost-desc` (most windows first, then most SA entries; reads without phase 3 last), `nw-desc` (windows only), `locus` (genome bin of the read's first window, `WC[0][WC_gStart]`), `anchor` (genome position of the first anchor's first locus), `nwlocus` (window-count class 1, 2, 3-4, 5-8, ... descending, then the first window's bin), `sig` (the control-flow signature known before phase 3: stitched, nW, nP, nA, descending; ties keep prefix order), `warp-desc` (the warps of the prefix order, 32 consecutive reads, kept whole and in order, run heaviest first by the sum of their windows, then SA entries: longest-first at warp granularity without breaking coherence) |
| `--p3order-a/m/b K` | the same keys for one lane only (overrides `--p3order` for that lane) |
| `--best-order` | E4's best exact order as one flag: `--presort --p3order-m nw-desc` (prefix order everywhere, lane M's phase 3 by window count); later flags override |
| `--p3-block N` | threads per block of the phase-3 kernels (32, 64, 128; default 128). 32 = one warp per block: a finished warp frees its slot instead of waiting for the other 3 warps of its block |
| `--p3-maxreg R`, `--p3-maxreg-a/m/b R` | `k_p3` (all lanes, or one lane) under `__launch_bounds__` capping registers at 64 or 80 (94 without; ptxas: 64 = 148-160 B spills, 80 = 60-68 B; with 32-thread blocks the 80 cap compiles to 72) |
| `--compact` | E2: the phase-3 order puts the reads that already overflowed in phase 1 after the others (stable partition), so their idle threads no longer sit in working warps |
| `--e4-best` | E4's fastest exact configuration: `--best-order --p3-maxreg-a 64 --batch-a 1048576 --batch-m 524288 --batch-b 131072` |
| `--queue` | phase 3 as a persistent kernel: grid = SMs x resident blocks (occupancy API); each warp takes the next reads of the batch, in the `--p3order` order, from an atomic counter (warp-aggregated) |
| `--fused A\|M\|B` | the lane as one persistent kernel with Work per resident thread: a warp takes the next 32 reads (prefix order) from an atomic counter, runs phase 1 (thread per read), scans the SA entries of each of the 32 reads warp-cooperatively (k_p2's loop, ballot into a per-thread bit buffer of `--fused-bits` entries, default 16,384), then phase 3 (thread per read). A read with more entries than the buffer runs `assignAll` itself (the same result). No batches, no host syncs inside the lane |
| `--fused-bps-a/m/b N` | resident blocks per SM of the fused kernel (0 = occupancy limit): the Work memory knob |
| `--fused-lb N` | `__launch_bounds__(128, N)` on the fused kernel (N = 1, 4, 5): fewer registers, more resident blocks |
| `--fused-bits N` | entries of the fused kernel's per-thread SA-entry bit buffer (default 16,384) |
| `--tail` | `%globaltimer` per block (start, end), per warp (end), the threads' busy time and when the last read was taken, per phase-3 launch |
| `--label` | the variant's name in the log and JSON |

Every run prints `m5 E4-order build <sha>; ...` with the compiled-in commit (job 790068 had run a stale binary; this marker
and the binary's sha256 in the job log rule that out).

Scheduling statistics (`--tail`, last timed run, phase-3 kernel only, summed over a lane's launches):
- **slot fill** = sum over blocks of (block lifetime x warps per block) / (resident warp slots x kernel span). 1 - fill is the
  share of warp slots left empty: block-level tails, partial waves and the end-of-batch drain.
- **drain** = (kernel end - the moment the last read was taken) / span: for fixed batches the last block's start, for the
  persistent kernels the time the last read index was handed out.
- **warp balance** = sum of warp lifetimes / sum of (block lifetime x warps per block): slots held by finished warps of a block
  that is still running.
- **lane balance** = sum of the threads' busy time / (32 x sum of warp lifetimes). Only meaningful for the persistent kernels:
  in the fixed-grid kernel a thread's end time is read after the warp reconverges, so it is 1.00 by construction.

Fixed settings in every variant (the engine's): budgets 1024 / 2048 / 2048 steps, batches 262,144 (A), 262,144 (M),
65,536 (B) reads, 128 threads per block; host lane on 24 threads.

## Jobs

| job | what | binary (commit, sha256 prefix) |
|---|---|---|
| 795239 | build (debug-cpu, CUDA 13.3) | `m5-e552a6f32d` (64f7792ec268c121) |
| 795240 | round 1: 16 variants + identity, g051, 665 s | `m5-e552a6f32d` |
| 795260, 795287, 795288 | builds: `--fused-lb`; per-lane order; the base source cf92cdb987 with CUDA 13.3 | `m5-f61996240d`, `m5-8885087332` (992464e2e1642de5), `m5-cf92cdb987` (5d2436f962c276d9) |
| 795289, 795304 | round 2, cancelled while pending (variants added) | - |
| 795307 | build of b861dab5 (`--best-order`) | `m5-b861dab562` |
| 795328, 795334 | builds of 857bd885 (`--p3-block`, `--p3-maxreg`, `sig`) and f8c2ae0d (`warp-desc`) | `m5-857bd88594` (d0b8de7959f02947), `m5-f8c2ae0d12` (618f8fa46e39354a) |
| 795308 | round 2 (`ROUND=2`, script of b861dab5): reference builds, per-lane order, batch sizes, lane B in one launch, fused at 4-5 blocks/SM, equal-memory staged vs fused, `--best-order` | the latest build when it starts |
| 795329 | round 3, cancelled while pending (warp-desc added) | - |
| 795329, 795335 | round 3 on short-gpu, cancelled while pending (moved to debug-gpu) | - |
| 795347 | round 3 (`ROUND=3`, debug-gpu): the E1 follow-ups (32-thread blocks, longest-first from a counter, `sig`, `warp-desc`, register caps, E1's batch sizes) and ncu passes on `k_p3` | `m5-f8c2ae0d12` |
| 795562-795564, 795573-795575 | builds and round 4, cancelled (base-kernel fix, then `--compact`, added first) | - |
| 795645 | build of 1b2c82d8: `k_p3` without tail code unless `--tail`, `--p3-maxreg-a/m/b`, `--compact` | `m5-1b2c82d8c8` (5135a551979cd81a) |
| 795646 | round 4 (`ROUND=4`, debug-gpu, g052): base kernel; reference build, main variants, E2's `--compact`, combinations; ncu | `m5-1b2c82d8c8` |
| 795647 | round 4 on short-gpu, cancelled when 795646 started | - |
| 795757, 795758 | build of the branch tip 04398779 (`--e4-best` since 73a8bb75) and round 5 (`ROUND=5`, debug-gpu, g051: v0 and `--e4-best`, twice each) | `m5-04398779c7` (baaf12c4b93a953c) |
| 795835, 795836 | builds of the merge e51325aa (`bench/fable/order/e4m_cc.sbatch`): default and `-DM5_NOCLK`; `-DM5_TAIL`, `-DM5_CENSUS`, `-DM5_BASELINE` (all compile) | `$W/jobs/fable/order/merge/bin/m5`, `m5n`, `m5t`, `m5c`, `m5b` |
| 795837 | merge gate (`e4m_gate.sbatch`, debug-gpu, g051): default vs `M5_NOCLK`, with E4's flags, twice each for the plain builds | `m5`, `m5n` (e51325aadf) |
| 795838 | the same on short-gpu, cancelled when 795837 started | - |

Results (JSON per variant, logs): `$W/jobs/fable/order/results/<job>/`, copied to `bench/fable/order/results/<job>/`.
Table: `python bench/fable/order/e4_table.py bench/fable/order/results/<job> v0 [--tail]` (round 3: baseline `v0t`).

Registers (ptxas, CUDA 13.3, sm_90): `k_p1` 95, `k_p2` 42, `k_key` 20 (all lanes); `k_p3` 80 / 80 / 94 (A / M / B) in the base
build and round 4, 94 / 94 / 94 with the tail code (rounds 1-3, and `--tail`), `k_p3q` 94-96 (always with tail code). 80
registers = 6 blocks of 128 per SM (37.5% occupancy), 94 = 5 blocks (2,640 resident warps, 31.25%). `k_fused` 134-144 (3 blocks/SM, 1,584 warps); with `--fused-lb 4` 128
registers (40-56 B spills, 4 blocks/SM), `--fused-lb 5` 96 registers (160 B spills, 5 blocks/SM).

## Results, round 1 (job 795240)

Rounds 1-3 ran `k_p3` with the tail code compiled in (see the summary and round 4): their absolute times are about 18% above
the base build's; the comparisons between their variants are like for like (one kernel for all of them).

Every variant: **1,991,926 / 1,991,926 distinct reads identical to STAR (100%)**, 4,118,999 / 4,118,999 reads weighted;
STAR-unmapped 151,095 and mapped 1,840,831 all identical; host lane 5,304 reads, 0 failures. Times: median of 3 timed runs
after a warm-up (run-to-run spread <= 0.6%). Rate = 1,991,926 / (A + M + B).

| variant | options | A ms | M ms | B ms | M5 ms | M distinct/s | vs v0 | Work GB A/M/B | device GB in use (peak) |
|---|---|---|---|---|---|---|---|---|---|
| v0raw | harness order (truth files: 24 sorted runs) | 291.8 | 418.7 | 304.2 | 1014.8 | 1.96 | 1.02x | 12.2/21.8/41.7 | 64.6 |
| **v0** | **presort (the engine's order)** | **294.7** | **423.8** | **312.5** | **1031.0** | **1.93** | **1.00x** | 12.2/21.8/41.7 | 64.6 |
| v0t | v0 + `--tail` (instrumentation cost) | 294.1 | 420.6 | 311.8 | 1026.5 | 1.94 | 1.00x | same | 64.6 |
| V1 v1bin | `--bin` as it exists (cost ascending) | 504.0 | 489.2 | 433.9 | 1427.1 | 1.40 | 0.72x | same | 64.6 |
| V1 v1desc | cost descending (nW, then SA entries) | 499.7 | 441.7 | 454.2 | 1395.6 | 1.43 | 0.74x | same | 64.6 |
| V1 v1nw | nW descending only (stable) | 307.0 | **391.8** | 358.9 | 1057.6 | 1.88 | 0.97x | same | 64.6 |
| V2 v2locus | first window's genome bin | 376.8 | 417.7 | 377.2 | 1171.7 | 1.70 | 0.88x | same | 64.6 |
| V2 v2anchor | first anchor's genome position | 383.8 | 414.2 | 387.3 | 1185.4 | 1.68 | 0.87x | same | 64.6 |
| V2 v2nwlocus | nW class (1, 2, 3-4, 5-8, ...) desc, then bin | 346.2 | 406.5 | 394.5 | 1147.3 | 1.74 | 0.90x | same | 64.6 |
| V3 v3q | queue, prefix order | 306.0 | 428.2 | 313.2 | 1047.4 | 1.90 | 0.98x | same | 64.6 |
| V3 v3qdesc | queue, cost descending (longest first) | 513.3 | 468.6 | 451.3 | 1433.2 | 1.39 | 0.72x | same | 64.6 |
| V3 v3qnwlocus | queue, nwlocus | 353.6 | 424.7 | 390.7 | 1169.0 | 1.70 | 0.88x | same | 64.6 |
| V4 v4big | staged, batches 512k (A) / 384k (M) | **264.9** | 394.8 | 311.7 | **971.3** | **2.05** | **1.06x** | 24.4/32.7/41.7 | 64.6 |
| V4 v4fA | fused lane A (3 blocks/SM) | 545.8 | 422.2 | 311.9 | 1279.9 | 1.56 | 0.81x | **2.4**/21.8/41.7 | 64.6 |
| V4/V5 v4fAMB | fused A, M, B (3 blocks/SM) | 545.0 | 804.1 | 340.6 | 1689.7 | 1.18 | 0.61x | **2.4/4.2/32.2** | 55.2 |
| V4/V5 v4fAMB1 | fused A, M, B, 1 block/SM | 583.9 | 840.3 | 475.7 | 1899.9 | 1.05 | 0.54x | **0.8/1.4/10.7** | **33.7** |

Phase split of v0 (ms, phase 1 / scan / phase 3): A 107 / 17 / 171, M 79 / 18 / 327, B 33 / 7 / 273. Every order variant
changes only phase 3 (phase 1 and the scan are the same in all of them, as they should be).

Phase-3 schedule (`--tail`, last run; fill = warp-slot fill, drain = share of the kernel span after the last read was taken,
warp bal. = warp balance inside blocks):

| variant | A: fill / drain / warp bal. | M: fill / drain / warp bal. | B: fill / drain / warp bal. | launches A/M/B |
|---|---|---|---|---|
| v0t (fixed batches, prefix) | 0.83 / 0.35 / 0.69 | 0.84 / 0.28 / 0.82 | **0.39 / 1.00** / 0.88 | 8 / 3 / 2 |
| v1bin | 0.21 / 0.91 / 0.87 | 0.53 / 0.65 / 0.87 | 0.24 / 1.00 / 0.76 | 8 / 3 / 2 |
| v1nw | 0.53 / 0.59 / 0.78 | 0.85 / 0.24 / 0.84 | 0.32 / 1.00 / 0.87 | 8 / 3 / 2 |
| v2locus | 0.52 / 0.61 / 0.68 | 0.79 / 0.34 / 0.82 | 0.31 / 1.00 / 0.87 | 8 / 3 / 2 |
| v3q (queue) | 0.76 / 0.56 / 0.86 | 0.86 / 0.40 / 0.96 | 0.54 / 1.00 / 0.85 | 8 / 3 / 2 |
| v3qdesc | 0.29 / 0.88 / 0.67 | 0.68 / 0.53 / 0.89 | 0.38 / 1.00 / 0.57 | 8 / 3 / 2 |
| v4fAMB (fused) | **0.99 / 0.03** / 1.00 | **0.98 / 0.07** / 0.99 | 0.80 / 0.51 / 0.89 | 1 / 1 / 1 |
| v4fAMB1 (fused, 1 block/SM) | 0.99 / 0.02 / 1.00 | 0.98 / 0.04 / 0.99 | 0.90 / 0.22 / 0.95 | 1 / 1 / 1 |

### What round 1 says

1. **The prefix order is the best phase-3 order for lanes A and B by a wide margin; every reordering is slower.** Sorting a
   batch by cost (V1, as `--bin` does) makes lane A 1.7x slower (phase 3 171 -> 380 ms) and M5 1.38-1.40x slower.
   Genome-locality orders (V2) cost 1.28x on lane A. The earlier "binning gives no gain" (job 790068, stale binary) was wrong
   in the other direction: binning is a large loss. The mechanism is SIMT coherence, not memory locality: neighbours in prefix
   order are near-identical sequences (same seeds, SA ranges, windows and splice decisions), so a warp of them follows one
   control path. `locus` keeps memory locality (same genome bins) but mixes sequences that stitch differently and loses 1.28x;
   `nw-desc` keeps prefix order inside each window-count class (stable sort) and loses only 1.04x on A. The 2.7x the engine
   got from prefix sorting is mostly this coherence in phase 3, not SAi cache hits in phase 1 (M3: 1.03-1.14x for seeds).
2. **Lane M is the exception: window-count classes help it** (nw-desc: phase 3 327 -> 294 ms, lane M -7.5%). M's reads have
   17-64 windows; equal-nW warps run the window loop the same number of times, and the stable sort keeps prefix order within
   a class. `cost-desc`, which breaks prefix order inside the class by sorting on SA entries, loses (442 ms).
3. **A persistent queue inside a batch gains nothing** (v3q 1.6% slower; warp balance in blocks 0.69 -> 0.86, slot fill
   0.83 -> 0.76). The hardware block scheduler already hands out blocks dynamically; what is left is the drain at the end of
   every batch (35% / 28% of the phase-3 span after the last block started, in A / M), and a queue over one batch has the same
   drain: when the last read is handed out, ~2,640 warps are still busy with about a third of the batch. Longest-first
   (AGAThA-style, v3qdesc) is the worst variant (1.39x slower): with thread-per-read SIMT, grouping heavy reads puts many
   divergent heavy reads in one warp, lengthens the critical path, and destroys coherence.
4. **Lane B is latency-bound, not throughput-bound.** Its 90,740 reads run in 2 launches of <= 65,536 reads, fewer than the
   84,480 resident thread slots, so each launch is one wave and lasts as long as its slowest warp (slot fill 0.39, drain 100%:
   the last block starts at once). No order helps (all are worse); fewer launches or overlap can.
5. **Batch barriers are the tail.** Larger batches (v4big: 4 instead of 8 A launches, 2 instead of 3 M launches) give -10% on
   A, -7% on M, -6% on M5 (971 ms, 2.05 M/s), at 24.4 / 32.7 GB of Work.
6. **The fused persistent kernel (V4) removes the tail completely** (slot fill 0.98-0.99, drain 2-7% in A and M) **and cuts
   Work memory 5x** (A 12.2 -> 2.4 GB, M 21.8 -> 4.2 GB at 3 blocks/SM; 0.8 / 1.4 / 10.7 GB at 1 block/SM, device peak 64.6
   -> 33.7 GB), **but each read costs 1.85x more** (A 545 vs 295 ms; M 804 vs 424 ms). Occupancy is not the reason: at 1
   block/SM (a third of the warps) lane A is only 7% slower than at 3. The likely cause is the megakernel effect (Laine, Karras,
   Aila 2013; Part I 7.2): one kernel holding the seed search, the scan and the stitcher runs every phase at the register
   footprint of the largest (136-144 vs 95), warps in different phases on one SM compete for the instruction cache, and the
   scan of a group's 32 reads runs warp-serially instead of across all warps. Not verified with counters (E1's ncu runs can).

## Results, round 2 (job 795308)

Script of b861dab5 (`ROUND=2`), g050, binary `m5-f8c2ae0d12` (the latest build when it started; the variants it runs behave as in
the build they were added in). Every variant 1,991,926 / 1,991,926 identical. Reference builds in the same job: `base13` is
the branch's base source (cf92cdb9) compiled with CUDA 13.3 by `e4_compile.sbatch`, `w11` the W11 binary of 2026-10-28
(CUDA 12.4.1); both map in the harness order.

| variant | options | A ms | M ms | B ms | total ms | M distinct/s | vs v0 | identical | A/M/B work GB | p3 regs A/M/B | A/M/B p3 slot fill | A/M/B drain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base13 | reference build, harness order | 269.7 | 340.1 | 256.4 | 866.2 | 2.30 | 1.19x | 1991926/1991926 | | | | |
| best | presort, order none/nw-desc/none | 294.8 | 391.5 | 312.3 | 998.5 | 1.99 | 1.03x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | -/-/- | -/-/- |
| f40 | presort, fused AMB bps 4/2/1 lb 4, tail | 529.8 | 779.9 | 474.0 | 1783.7 | 1.12 | 0.58x | 1991926/1991926 | 3.1/2.8/10.7 | 128/128/128 | 0.98/0.98/0.90 | 0.03/0.06/0.22 |
| f40a | presort, order prefix(fused)/nw-desc/none, fused A bps 4/0/0 lb 4, batches 256k/32k/16k, tail | 529.6 | 968.8 | 671.4 | 2169.7 | 0.92 | 0.47x | 1991926/1991926 | 3.1/2.7/10.4 | 128/94/94 | 0.98/0.15/0.12 | 0.04/1.00/1.00 |
| s40 | presort, order none/nw-desc/none, batches 64k/32k/16k, tail | 452.0 | 974.5 | 671.8 | 2098.3 | 0.95 | 0.49x | 1991926/1991926 | 3.1/2.7/10.4 | 94/94/94 | 0.34/0.15/0.12 | 1.00/1.00/1.00 |
| v0 | presort | 294.6 | 421.8 | 312.3 | 1028.7 | 1.94 | 1.00x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | -/-/- | -/-/- |
| v0b | presort | 294.5 | 424.4 | 311.4 | 1030.2 | 1.93 | 1.00x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | -/-/- | -/-/- |
| v0raw | harness order | 292.0 | 418.4 | 303.3 | 1013.7 | 1.96 | 1.01x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | -/-/- | -/-/- |
| v4big | presort, batches 512k/384k/64k | 264.9 | 393.7 | 310.7 | 969.3 | 2.06 | 1.06x | 1991926/1991926 | 24.4/32.7/41.7 | 94/94/94 | -/-/- | -/-/- |
| v4bigB1 | presort, batches 512k/384k/128k | 265.6 | 391.3 | 246.1 | 903.1 | 2.21 | 1.14x | 1991926/1991926 | 24.4/32.7/57.7 | 94/94/94 | -/-/- | -/-/- |
| v4fA4 | presort, order prefix(fused)/none/none, fused A lb 4, tail | 528.3 | 423.4 | 311.4 | 1263.2 | 1.58 | 0.81x | 1991926/1991926 | 3.1/21.8/41.7 | 128/94/94 | 0.99/0.83/0.39 | 0.02/0.29/1.00 |
| v4fA5 | presort, order prefix(fused)/none/none, fused A lb 5, tail | 518.2 | 420.4 | 310.8 | 1249.5 | 1.59 | 0.82x | 1991926/1991926 | 3.9/21.8/41.7 | 96/94/94 | 0.98/0.84/0.40 | 0.03/0.28/1.00 |
| v4fAMB5 | presort, fused AMB lb 5, tail | 517.3 | 719.2 | 299.7 | 1536.3 | 1.30 | 0.67x | 1991926/1991926 | 3.9/7.0/53.7 | 96/96/96 | 0.98/0.96/0.72 | 0.04/0.12/0.56 |
| v5best | presort, order none/nw-desc/none, batches 512k/384k/128k | 265.1 | 364.5 | 245.3 | 874.9 | 2.28 | 1.18x | 1991926/1991926 | 24.4/32.7/57.7 | 94/94/94 | -/-/- | -/-/- |
| v5bestq | presort, order none/nw-desc/none, queue, batches 512k/384k/128k | 281.3 | 385.8 | 231.4 | 898.4 | 2.22 | 1.14x | 1991926/1991926 | 24.4/32.7/57.7 | 94/94/96 | -/-/- | -/-/- |
| v5m | presort, order none/nw-desc/none | 294.4 | 391.8 | 311.9 | 998.0 | 2.00 | 1.03x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | -/-/- | -/-/- |
| v5mt | presort, order none/nw-desc/none, tail | 295.2 | 392.8 | 311.3 | 999.3 | 1.99 | 1.03x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | 0.82/0.86/0.40 | 0.36/0.23/1.00 |
| w11 | reference build, harness order | 271.4 | 343.7 | 255.3 | 870.4 | 2.29 | 1.18x | 1991926/1991926 | | | | |

- **The reference builds are 1.17x faster than the E4 build in the same order** (base13 866.2 ms, w11 870.4, v0raw 1013.7).
  CUDA 13.3 vs 12.4.1 makes no difference (0.5%); the difference is the E4 build's `k_p3`. See Round 4: rounds 1-3 compiled
  the `--tail` timing code into every `k_p3` (94 registers instead of the base's 80 in lanes A and M; phase 3 of A 168.5 vs
  146.3 ms, of B 256.9 vs 208.9 ms). All E4 variants of rounds 1-3 share that kernel, so they compare like for like, but their
  absolute times are 15-23% (phase 3) too high.
- `--best-order` (= `--presort --p3order-m nw-desc`) reproduces v5m: 998.5 vs 998.0 ms, -3.0% against v0.
- **Lane B in one launch** (B batch 131,072 >= its 90,740 reads): lane B 310.7 -> 246.1 ms (-21%), M5 -6.8% on top of the
  512k / 384k A / M batches. v5best (best order + 512k / 384k / 128k): **874.9 ms, 2.28 M/s, 1.18x**. With the queue on top
  (v5bestq) it is 2.7% slower (A and M +6%; only lane B gains, -5.7%).
- The fused lane at 5 blocks/SM (`--fused-lb 5`, 96 registers, 160 B spills) is 5% faster than at 3 blocks/SM (518 vs 546
  ms on lane A) and still 1.76x the staged lane: occupancy is not what makes the fused kernel slow.
- **At equal Work memory (a 40 GB-class budget: 3.1 / 2.8 / 10.7 GB) the fused lanes beat small staged batches**: f40 1783.7
  ms (1.12 M/s) against s40 (batches 64k / 32k / 16k) 2098.3 ms (0.95 M/s). Small staged batches are sub-wave launches (17 M
  launches, 6 B launches, slot fill 0.15 / 0.12) whose time is their slowest warp.
- Noise: v0 / v0b 1028.7 / 1030.2 ms (0.15%).

## Results, round 3 (job 795347): the E1 follow-ups

debug-gpu (8 CPUs), g052, `m5-f8c2ae0d12`, same `k_p3` with the tail code as rounds 1-2. Every variant and every ncu run
1,991,926 / 1,991,926 identical. Baseline `v0t` (presort + `--tail`).

| variant | options | A ms | M ms | B ms | total ms | M distinct/s | vs v0t | identical | A/M/B work GB | p3 regs A/M/B | A/M/B p3 slot fill | A/M/B drain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b32 | presort, p3 block 32, tail | 297.1 | 420.5 | 315.1 | 1032.8 | 1.93 | 0.99x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | 0.61/0.74/0.34 | 0.38/0.29/1.00 |
| b32desc | presort, cost-desc, p3 block 32, tail | 495.3 | 437.9 | 447.7 | 1381.0 | 1.44 | 0.74x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | 0.18/0.55/0.17 | 0.85/0.43/1.00 |
| b32r64 | presort, p3 block 32, maxreg 64, tail | 279.9 | 388.1 | 323.1 | 991.0 | 2.01 | 1.04x | 1991926/1991926 | 12.2/21.8/41.7 | 64/64/64 | 0.57/0.69/0.21 | 0.50/0.43/1.00 |
| b32wdesc | presort, warp-desc, p3 block 32, tail | 290.1 | 429.5 | 317.2 | 1036.9 | 1.92 | 0.99x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | 0.67/0.72/0.34 | 0.29/0.28/1.00 |
| best | presort, order none/nw-desc/none, tail | 296.2 | 390.9 | 311.7 | 998.8 | 1.99 | 1.03x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | 0.82/0.85/0.40 | 0.36/0.23/1.00 |
| bestE1 | presort, order none/nw-desc/none, batches 1024k/512k/128k | 254.9 | 380.4 | 241.8 | 877.1 | 2.27 | 1.17x | 1991926/1991926 | 48.9/43.5/57.7 | 94/94/94 | -/-/- | -/-/- |
| bestE1b32 | presort, order none/nw-desc/none, p3 block 32, batches 1024k/512k/128k | 254.5 | 379.9 | 242.1 | 876.5 | 2.27 | 1.17x | 1991926/1991926 | 48.9/43.5/57.7 | 94/94/94 | -/-/- | -/-/- |
| bestb32 | presort, order none/nw-desc/none, p3 block 32, tail | 294.6 | 390.8 | 315.1 | 1000.4 | 1.99 | 1.03x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | 0.62/0.75/0.34 | 0.37/0.23/1.00 |
| q32desc | presort, cost-desc, queue, p3 block 32, tail | 510.7 | 463.6 | 444.9 | 1419.2 | 1.40 | 0.72x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/96 | 0.20/0.60/0.22 | 0.88/0.53/1.00 |
| qdesc | presort, cost-desc, queue, tail | 512.7 | 464.0 | 449.9 | 1426.6 | 1.40 | 0.72x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/96 | 0.29/0.67/0.38 | 0.88/0.53/1.00 |
| qwdesc | presort, warp-desc, queue, tail | 307.2 | 428.5 | 315.4 | 1051.1 | 1.90 | 0.98x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/96 | 0.78/0.85/0.51 | 0.50/0.39/1.00 |
| r64 | presort, maxreg 64, tail | 280.6 | 387.9 | 320.2 | 988.7 | 2.01 | 1.04x | 1991926/1991926 | 12.2/21.8/41.7 | 64/64/64 | 0.74/0.80/0.24 | 0.50/0.42/1.00 |
| r80 | presort, maxreg 80, tail | 286.4 | 401.6 | 310.9 | 998.9 | 1.99 | 1.03x | 1991926/1991926 | 12.2/21.8/41.7 | 80/80/80 | 0.80/0.83/0.32 | 0.41/0.32/1.00 |
| sig | presort, sig, tail | 370.4 | 405.9 | 352.5 | 1128.8 | 1.76 | 0.91x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | 0.36/0.89/0.34 | 0.80/0.32/1.00 |
| sigm | presort, order none/sig/none, tail | 294.8 | 404.9 | 316.8 | 1016.5 | 1.96 | 1.01x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | 0.82/0.90/0.39 | 0.36/0.31/1.00 |
| v0b | presort | 294.8 | 422.5 | 310.4 | 1027.7 | 1.94 | 1.00x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | -/-/- | -/-/- |
| v0t | presort, tail | 295.1 | 421.9 | 310.3 | 1027.3 | 1.94 | 1.00x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | 0.82/0.84/0.40 | 0.36/0.28/1.00 |
| wdesc | presort, warp-desc, tail | 287.9 | 428.4 | 318.4 | 1034.7 | 1.93 | 0.99x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | 0.85/0.82/0.39 | 0.28/0.28/1.00 |

Phase-3 span against busy time (`--tail`; busy = span x slot fill x warp balance: the duration if every resident warp slot
were busy until its warp ended):

| variant | lane | phase-3 ms | p3 span ms | busy ms | busy / span | slot fill | warp balance | lane balance | drain | resident warps | launches |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b32 | A | 170.6 | 172.2 | 105.3 | 0.61 | 0.61 | 1.00 | 1.00 | 0.38 | 2640 | 8 |
| b32 | M | 323.9 | 324.0 | 238.2 | 0.74 | 0.74 | 1.00 | 1.00 | 0.29 | 2640 | 3 |
| b32 | B | 274.7 | 273.1 | 94.0 | 0.34 | 0.34 | 1.00 | 1.00 | 1.00 | 2640 | 2 |
| b32wdesc | A | 165.9 | 162.9 | 108.7 | 0.67 | 0.67 | 1.00 | 1.00 | 0.29 | 2640 | 8 |
| b32wdesc | M | 332.2 | 332.4 | 240.5 | 0.72 | 0.72 | 1.00 | 1.00 | 0.28 | 2640 | 3 |
| b32wdesc | B | 277.6 | 278.2 | 93.9 | 0.34 | 0.34 | 1.00 | 1.00 | 1.00 | 2640 | 2 |
| q32desc | A | 386.6 | 382.3 | 74.8 | 0.20 | 0.20 | 1.00 | 0.87 | 0.88 | 2640 | 8 |
| q32desc | M | 366.8 | 368.3 | 221.7 | 0.60 | 0.60 | 1.00 | 0.91 | 0.53 | 2640 | 3 |
| q32desc | B | 406.7 | 410.6 | 90.8 | 0.22 | 0.22 | 1.00 | 0.84 | 1.00 | 2048 | 2 |
| qwdesc | A | 183.1 | 179.7 | 122.2 | 0.68 | 0.78 | 0.87 | 0.83 | 0.50 | 2640 | 8 |
| qwdesc | M | 331.3 | 331.9 | 273.0 | 0.82 | 0.85 | 0.96 | 0.83 | 0.39 | 2640 | 3 |
| qwdesc | B | 274.5 | 275.4 | 118.8 | 0.43 | 0.51 | 0.84 | 0.66 | 1.00 | 2048 | 2 |
| r64 | A | 156.4 | 157.2 | 83.3 | 0.53 | 0.74 | 0.71 | 1.00 | 0.50 | 4224 | 8 |
| r64 | M | 291.4 | 291.3 | 194.3 | 0.67 | 0.80 | 0.84 | 1.00 | 0.42 | 4224 | 3 |
| r64 | B | 280.4 | 280.3 | 59.9 | 0.21 | 0.24 | 0.87 | 1.00 | 1.00 | 4224 | 2 |
| sig | A | 245.5 | 244.5 | 69.0 | 0.28 | 0.36 | 0.78 | 1.00 | 0.80 | 2640 | 8 |
| sig | M | 309.3 | 310.9 | 177.0 | 0.57 | 0.89 | 0.64 | 1.00 | 0.32 | 2640 | 3 |
| sig | B | 312.8 | 312.3 | 79.9 | 0.26 | 0.34 | 0.76 | 1.00 | 1.00 | 2640 | 2 |
| v0t | A | 170.5 | 171.1 | 97.2 | 0.57 | 0.82 | 0.69 | 1.00 | 0.36 | 2640 | 8 |
| v0t | M | 325.3 | 325.1 | 224.7 | 0.69 | 0.84 | 0.82 | 1.00 | 0.28 | 2640 | 3 |
| v0t | B | 270.5 | 270.4 | 94.0 | 0.35 | 0.40 | 0.87 | 1.00 | 1.00 | 2640 | 2 |
| wdesc | A | 163.8 | 161.4 | 100.5 | 0.62 | 0.85 | 0.73 | 1.00 | 0.28 | 2640 | 8 |
| wdesc | M | 333.0 | 337.2 | 226.5 | 0.67 | 0.82 | 0.82 | 1.00 | 0.28 | 2640 | 3 |
| wdesc | B | 278.9 | 279.6 | 94.0 | 0.34 | 0.39 | 0.86 | 1.00 | 1.00 | 2640 | 2 |

Other variants: `python bench/fable/order/e4_table.py bench/fable/order/results/795347 v0t --tail`.

ncu, `k_p3` (`--set full`, one launch: lane A's 2nd batch of 262,144 reads, lane M's 1st; `ncu_*.txt` in the results):

| launch | duration ms | active threads per warp instruction | not predicated off | issue slots busy | IPC | warp cycles per issued instr. | achieved / theoretical occupancy | L1 / L2 hit | branch efficiency |
|---|---|---|---|---|---|---|---|---|---|
| v0, lane A | 21.56 | **1.81** (5.7%) | 1.68 | 10.9% | 0.50 | 28.3 | 22.3% / 31.25% | 80.6% / 84.2% | 97.9% |
| warp-desc, lane A | 21.64 | 1.81 | 1.68 | 10.9% | 0.52 | 28.7 | 23.5% / 31.25% | 80.2% / 83.4% | 97.9% |
| 32-thread blocks, lane A | 21.91 | 1.81 | 1.68 | 10.9% | 0.50 | 30.2 | 23.9% / 31.25% | 80.3% / 83.7% | 97.9% |
| sig, lane A | 32.53 | **2.60** | 2.40 | 5.2% | 0.35 | 29.0 | 15.7% / 31.25% | 75.7% / 77.0% | 96.8% |
| v0, lane M | 147.91 | 1.83 | 1.68 | 12.8% | 0.56 | 28.5 | 24.7% / 31.25% | 79.9% / 81.4% | 97.7% |

What round 3 says:
1. **One warp per block does not shorten a batch.** With 32-thread blocks the warp balance is 1 by construction and a
   finished warp frees its slot, but the slot fill drops by the same amount (A 0.82 -> 0.61) and the span is unchanged (172 vs
   171 ms): a batch lasts as long as its longest warp, which starts at t = 0 (E1), and at the end of a batch there are no
   reads left to put in the freed slots. M5 +0.5%.
2. **Longest-first from an atomic counter is the worst schedule again** (q32desc 1.38x, qdesc 1.39x slower), with or without
   one warp per block: grouping the heaviest reads makes the heaviest warps.
3. **The signature order raises SIMT efficiency and loses anyway**: active threads per warp instruction 1.81 -> 2.60 (+44%)
   in lane A, but lane A's phase 3 is 1.44x longer (245 vs 170 ms) and achieved occupancy falls 22% -> 16%: equal signatures
   put the expensive reads of a class into the same warps. Lane M only (sigm): -4% on M, less than nw-desc's -7%.
4. **Warp-granular longest-first (warp-desc) is the only reordering that does not lose on lane A**: the prefix-order warps
   are kept whole (ncu: identical instruction counts, 1.81 active threads, identical divergent branches) and only their start
   order changes. Phase 3 of lane A -4% (163.8 vs 170.5 ms; drain 0.36 -> 0.28), lanes M and B +2-3%, M5 +0.7%.
5. **SIMT efficiency, not scheduling, is the ceiling of phase 3**: 1.81 of 32 lanes do work per issued instruction (5.7%),
   with branch efficiency 98%. The lanes are idle because their reads are done while one or two reads of the warp go on (the
   per-warp tail), not because of divergent branches. No order of whole reads fixes that; it needs either warps that work on
   one read together (E3) or re-packing the still-running reads into full warps.
6. **Register caps help on this kernel** (94 registers): 64 registers (8 blocks/SM, 148-160 B spills) -3.8% on M5 (A -4.9%,
   M -8.1%, B +3.2%), 80 registers -2.8%. Round 4 checks this on the 80-register base kernel.
7. With the best order and E1's batch sizes (1M / 512k / 128k; Work 48.9 / 43.5 / 57.7 GB) M5 is **877.1 ms, 2.27 M/s,
   1.17x** (v0t 1027.3); 32-thread blocks on top change nothing (876.5).

## Results, round 4 (job 795646): the base kernel

debug-gpu (8 CPUs), g052, `m5-1b2c82d8c8`: `k_p3` is the base kernel again (80 / 80 / 94 registers) unless `--tail` is given.
Every variant and every ncu run 1,991,926 / 1,991,926 identical. Baseline v0 (presort, the engine's order and batches).

| variant | options | A ms | M ms | B ms | total ms | M distinct/s | vs v0 | identical | A/M/B work GB | p3 regs A/M/B | A/M/B p3 slot fill | A/M/B drain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b32 | presort, p3 block 32 | 270.1 | 337.8 | 263.7 | 871.5 | 2.29 | 1.00x | 1991926/1991926 | 12.2/21.8/41.7 | 80/80/94 | -/-/- | -/-/- |
| base13 | reference build, harness order | 272.9 | 345.1 | 256.8 | 874.7 | 2.28 | 1.00x | 1991926/1991926 | | | | |
| best | presort, order none/nw-desc/none | 270.3 | 306.9 | 253.8 | 831.0 | 2.40 | 1.05x | 1991926/1991926 | 12.2/21.8/41.7 | 80/80/94 | -/-/- | -/-/- |
| bestE1 | presort, order none/nw-desc/none, batches 1024k/512k/128k | 236.3 | 291.5 | 196.5 | 724.3 | 2.75 | 1.20x | 1991926/1991926 | 48.9/43.5/57.7 | 80/80/94 | -/-/- | -/-/- |
| bestE1cmp | presort, order none+compact/nw-desc+compact/none+compact, batches 1024k/512k/128k | 231.1 | 292.3 | 194.6 | 717.9 | 2.77 | 1.21x | 1991926/1991926 | 48.9/43.5/57.7 | 80/80/94 | -/-/- | -/-/- |
| bestE1r64am | presort, order none/nw-desc/none, batches 1024k/512k/128k | 224.5 | 294.3 | 204.4 | 723.1 | 2.75 | 1.20x | 1991926/1991926 | 48.9/43.5/57.7 | 64/64/94 | -/-/- | -/-/- |
| bestE1wr64am | presort, order warp-desc/nw-desc/none, batches 1024k/512k/128k | 223.8 | 294.8 | 196.4 | 715.0 | 2.79 | 1.22x | 1991926/1991926 | 48.9/43.5/57.7 | 64/64/94 | -/-/- | -/-/- |
| bestcmp | presort, order none+compact/nw-desc+compact/none+compact | 273.7 | 306.9 | 264.8 | 845.5 | 2.36 | 1.03x | 1991926/1991926 | 12.2/21.8/41.7 | 80/80/94 | -/-/- | -/-/- |
| bestr64 | presort, order none/nw-desc/none, maxreg 64 | 262.4 | 319.2 | 260.7 | 842.3 | 2.36 | 1.03x | 1991926/1991926 | 12.2/21.8/41.7 | 64/64/64 | -/-/- | -/-/- |
| bestr64am | presort, order none/nw-desc/none | 262.3 | 319.2 | 253.7 | 835.1 | 2.39 | 1.04x | 1991926/1991926 | 12.2/21.8/41.7 | 64/64/94 | -/-/- | -/-/- |
| bestwA | presort, order warp-desc/nw-desc/none | 269.2 | 305.8 | 253.5 | 828.5 | 2.40 | 1.05x | 1991926/1991926 | 12.2/21.8/41.7 | 80/80/94 | -/-/- | -/-/- |
| cmp | presort | 275.1 | 339.1 | 265.1 | 879.3 | 2.27 | 0.99x | 1991926/1991926 | 12.2/21.8/41.7 | 80/80/94 | -/-/- | -/-/- |
| q | presort, queue | 309.2 | 424.1 | 315.1 | 1048.4 | 1.90 | 0.83x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/96 | -/-/- | -/-/- |
| r64 | presort, maxreg 64 | 263.0 | 328.8 | 261.3 | 853.1 | 2.33 | 1.02x | 1991926/1991926 | 12.2/21.8/41.7 | 64/64/64 | -/-/- | -/-/- |
| sig | presort, sig | 315.5 | 324.8 | 288.4 | 928.6 | 2.15 | 0.94x | 1991926/1991926 | 12.2/21.8/41.7 | 80/80/94 | -/-/- | -/-/- |
| v0 | presort | 273.3 | 343.4 | 253.7 | 870.4 | 2.29 | 1.00x | 1991926/1991926 | 12.2/21.8/41.7 | 80/80/94 | -/-/- | -/-/- |
| v0b | presort | 270.2 | 340.2 | 253.9 | 864.3 | 2.30 | 1.01x | 1991926/1991926 | 12.2/21.8/41.7 | 80/80/94 | -/-/- | -/-/- |
| v0raw | harness order | 272.0 | 341.3 | 256.9 | 870.2 | 2.29 | 1.00x | 1991926/1991926 | 12.2/21.8/41.7 | 80/80/94 | -/-/- | -/-/- |
| v0t | presort, tail | 294.9 | 421.6 | 311.6 | 1028.1 | 1.94 | 0.85x | 1991926/1991926 | 12.2/21.8/41.7 | 94/94/94 | 0.83/0.85/0.39 | 0.36/0.27/1.00 |
| v1bin | presort, cost-asc(--bin) | 410.2 | 389.1 | 341.9 | 1141.2 | 1.75 | 0.76x | 1991926/1991926 | 12.2/21.8/41.7 | 80/80/94 | -/-/- | -/-/- |
| v1desc | presort, cost-desc | 412.4 | 364.2 | 350.8 | 1127.3 | 1.77 | 0.77x | 1991926/1991926 | 12.2/21.8/41.7 | 80/80/94 | -/-/- | -/-/- |
| v1nw | presort, nw-desc | 269.6 | 307.2 | 289.8 | 866.6 | 2.30 | 1.00x | 1991926/1991926 | 12.2/21.8/41.7 | 80/80/94 | -/-/- | -/-/- |
| v4bigB1 | presort, batches 512k/384k/128k | 245.3 | 319.3 | 196.4 | 761.1 | 2.62 | 1.14x | 1991926/1991926 | 24.4/32.7/57.7 | 80/80/94 | -/-/- | -/-/- |
| v5best | presort, order none/nw-desc/none, batches 512k/384k/128k | 245.4 | 285.7 | 196.7 | 727.8 | 2.74 | 1.20x | 1991926/1991926 | 24.4/32.7/57.7 | 80/80/94 | -/-/- | -/-/- |
| wdesc | presort, warp-desc | 269.5 | 347.8 | 269.6 | 886.8 | 2.25 | 0.98x | 1991926/1991926 | 12.2/21.8/41.7 | 80/80/94 | -/-/- | -/-/- |

Phase split (ms, phase 1 / scan / phase 3): v0 A 107.0 / 18.6 / 146.7, M 78.7 / 20.0 / 244.1, B 32.8 / 7.0 / 213.6; best
(M nw-desc) M 78.7 / 17.7 / 210.2; bestE1 A 96.3 / 13.2 / 126.5, M 71.0 / 16.9 / 203.5, B 31.6 / 5.8 / 161.4. Device memory in
use: v0 36.0 / 45.4 / 64.6 GB in lanes A / M / B; v5best 48.2 / 56.3 / 80.6 GB; bestE1 72.7 / 67.1 / 80.6 GB.

ncu, `k_p3` on the base kernel (lane A's 2nd batch, lane M's 1st):

| launch | duration ms | warp instructions | active threads per warp instruction | issue slots busy | achieved / theoretical occupancy | warp cycles per issued instr. | branch efficiency |
|---|---|---|---|---|---|---|---|
| v0, lane A | 18.61 | 2.05 G | **1.94** (6.1%) | 12.1% | 27.3% / 37.5% | 31.7 | 97.5% |
| `--bin`, lane A | **41.38** | **1.17 G** (-43%) | **3.39** | **3.0%** | 20.1% / 37.5% | 36.9 | 94.5% |
| `--compact`, lane A | 18.39 | 1.93 G (-6%) | 2.06 | 10.9% | 27.1% / 37.5% | 31.1 | 97.2% |
| v0, lane M | 113.97 | 13.96 G | 2.35 | 13.1% | 30.3% / 37.5% | 34.2 | 96.0% |

What round 4 says:
1. **The fix restores the base build**: v0 870.4 ms, v0raw 870.2, base13 874.7 (v0 / v0b 870.4 / 864.3: 0.7% noise). The tail
   code alone (v0t) costs 18% (1028.1 ms): phase 3 of A 146.7 -> 170.5, M 244.1 -> 324.0, B 213.6 -> 271.6 ms.
2. **The order conclusions of rounds 1-3 hold on the base kernel**: `--bin` 1.31x slower, cost descending 1.30x, `sig` 1.07x,
   warp-desc 1.02x, nw-desc in every lane 1.00x (M -10.5%, B +14%); **nw-desc on lane M only (`--best-order`) -4.5%**
   (831.0 ms, 2.40 M/s; M's phase 3 244.1 -> 210.2 ms). warp-desc on A on top (bestwA) 828.5 ms, within noise.
3. **E2's cost key does what E2's simulation says and still loses**: sorting a batch by (nW, SA entries) cuts lane A's warp
   instructions by 43% and lifts active threads from 1.94 to 3.39, but the launch takes 2.2x longer with issue slots busy 3%:
   the warps of heavy reads now hold only heavy reads, each of them runs ~3 lanes at a time through 32 long reads, and the
   batch waits for them with the rest of the GPU idle. Slot use is not the objective; the batch's critical path is.
4. **E2's compaction is exact and gains nothing**: `--compact` +1.0% (879.3 ms), +1.7% with the best order, -0.9% with E1's
   batches; ncu shows 6% fewer warp instructions and 1.2% less time in lane A's launch, about what the per-batch partition
   costs. Idle threads of overflowed reads are cheap in a warp that is waiting for its longest read anyway.
5. **Scheduling**: the queue (its kernel still has the tail code) 1048.4 vs v0t 1028.1: +2.0%, as in round 1 (+2.0%). One
   warp per block (b32) 871.5: no effect.
6. **Registers**: a cap of 64 (8 blocks/SM, 148-160 B spills) gives A -3.8%, M -4.3%, B +3.0% on the prefix order (r64 853.1,
   -2.0%); with nw-desc on M the cap stops helping M (319.2 vs 306.9 ms), so it is worth it on lane A only (A -3% at the
   engine's batches, -5% at 1M: 236.3 -> 224.5 ms).
7. **Batches**: 512k / 384k / 128k (v4bigB1) -12.6% (761.1 ms; E1 measured -11.9%); lane B in one launch is most of it (B
   253.7 -> 196.4 ms). With the best order: v5best **727.8 ms, 2.74 M/s, 1.20x**; E1's 1M / 512k / 128k: bestE1 724.3 ms.
8. **Fastest run: bestE1wr64am 715.0 ms, 2.79 M/s, 1.22x** (warp-desc + 64 registers on A, nw-desc + 64 registers on M, E1
   batches). Lanes are independent (a lane's time depends only on its own settings: A of best / v0 / v0b 270.3 / 273.3 / 270.2),
   so the per-lane best is A 224.5 (bestE1r64am), M 291.5 (bestE1), B 196.5 (bestE1): **712.5 ms, 2.80 M/s**. That is
   `--e4-best`.

## Results, round 5 (job 795758): `--e4-best`

debug-gpu, g051, `m5-04398779c7`; v0 and `--e4-best` twice each, all 1,991,926 / 1,991,926 identical.

| variant | options | A ms | M ms | B ms | total ms | M distinct/s | vs v0 | identical | A/M/B work GB | p3 regs A/M/B | A/M/B p3 slot fill | A/M/B drain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4best | presort, order none/nw-desc/none, batches 1024k/512k/128k | 223.9 | 291.8 | 193.7 | 709.3 | 2.81 | 1.22x | 1991926/1991926 | 48.9/43.5/57.7 | 64/80/94 | -/-/- | -/-/- |
| e4bestb | presort, order none/nw-desc/none, batches 1024k/512k/128k | 224.1 | 290.6 | 197.0 | 711.6 | 2.80 | 1.21x | 1991926/1991926 | 48.9/43.5/57.7 | 64/80/94 | -/-/- | -/-/- |
| v0 | presort | 270.7 | 339.0 | 253.3 | 863.1 | 2.31 | 1.00x | 1991926/1991926 | 12.2/21.8/41.7 | 80/80/94 | -/-/- | -/-/- |
| v0b | presort | 273.8 | 340.6 | 254.1 | 868.5 | 2.29 | 0.99x | 1991926/1991926 | 12.2/21.8/41.7 | 80/80/94 | -/-/- | -/-/- |

`--e4-best` runs at **709.3 / 711.6 ms (2.81 / 2.80 M distinct/s), 1.22x** the engine configuration in the same job (v0
863.1 / 868.5 ms), as the per-lane composite of round 4 predicted (712.5 ms).

## Merge with gpu/fable-engine and M5_NOCLK (jobs 795835-795837)

The branch merges `gpu/fable-engine` (f1622f84): E1's tail probe (`-DM5_TAIL`), E2's census (`-DM5_CENSUS`), E5's
`stitch.cuh` variants (S3, TR32, ENC1 on by default; `-DM5_BASELINE` restores the old paths) and E6's files. In `m5.cu`,
E4's `p3one` / `k_p3<C, TAIL>` / `k_p3lb` carry the census buffers (one `CenP3` argument) and E1's `TAIL_BEGIN` /
`TAIL_INFO` / `TAIL_END`; E4's `--tail` flag variable is `g_e4tail` (E1's `tail.cuh` defines `g_tail`). The persistent
`--queue` and `--fused` kernels are dropped (slower, rounds 1-4); those flags now exit with a message. All five flavours
compile (795835, 795836). The default `k_p3` carries no tail code (79 registers, as with `--tail` off before).

`-DM5_NOCLK` (e51325aa) compiles out the `clock64()` phase counters: `M5CLK` / `M5T` / `M5ADD` (stitch.cuh), the recording
timer (`RecClk`), the phase-1 / phase-3 start clocks, and the per-read `atomicAdd` of `w.clk[8]` into `g_clk` (m5.cu
`p3one`, `lane_loop`). The harness then prints no cycle shares.

Registers (ptxas -v, CUDA 13.3, sm_90; `bench/fable/order/results/merge/regs-<flavour>-e51325aadf.tsv`):

| kernel | default (counters) | `M5_NOCLK` | `M5_TAIL` | `M5_CENSUS` | `M5_BASELINE` |
|---|---|---|---|---|---|
| `k_p1` A / M / B | 95 / 95 / 95 | 95 / 95 / 95 | 95 | 95 | 95 |
| `k_p2` | 42 | 42 | 42 | 44 | 42 |
| `k_p3` A / M / B | **79 / 79 / 79** (656 / 400 / 1744 B stack) | **79 / 79 / 79** (same stack) | 79 / 79 / 79 | 84 / 84 / 84 | 80 / 80 / 94 |
| `k_p3<C, true>` (`--tail`) A / M / B | 79 / 79 / 80 | 79 / 79 / 79 | 79 / 79 / 80 | 94 | 94 |
| `k_p3lb` cap 64 (128 x 8) A | 64, spills 64 / 92 B | 64, spills 100 / 140 B | | | |
| `k_p3lb` cap 80 (128 x 6) A | 80, spills 32 / 56 B | 80, spills 32 / 36 B | | | |
| `k_lane<A, local>` (`--mono --local`) | 88 | 104 | | | |

The counters cost no registers in the default kernels: E5's changes already brought `k_p3` from 80 / 80 / 94 to 79 / 79 / 79
(lane B 6 blocks of 128 per SM instead of 5), and `--tail`'s instantiation no longer grows (79-80, was 94).

Job 795837 (debug-gpu, g051, 2 M pbmc1k_v3 set, gate budgets 1024 / 2048 / 2048 and batches 512k / 384k / 64k = `@STD` of
`fable_gate.sbatch`; median of 3 per variant). Every variant 1,991,926 / 1,991,926 distinct (4,118,999 / 4,118,999 weighted)
identical to STAR:

| variant | build | options on top of `@STD` | A ms | M ms | B ms | total ms | M distinct/s | vs v0 |
|---|---|---|---|---|---|---|---|---|
| v0 | default | - | 212.7 | 268.4 | 223.2 | 704.4 | 2.83 | 1.00x |
| v6 | default | - (repeat) | 212.6 | 267.9 | 222.6 | 703.2 | 2.83 | 1.00x |
| v1 | `M5_NOCLK` | - | 212.2 | 305.8 | 217.0 | 734.9 | 2.71 | 0.96x |
| v7 | `M5_NOCLK` | - (repeat) | 212.5 | 305.7 | 218.1 | 736.3 | 2.71 | 0.96x |
| v2 | `M5_NOCLK` | `--best-order --p3-maxreg-a 64` | 200.4 | 281.7 | 218.4 | 700.6 | 2.84 | 1.01x |
| v3 | `M5_NOCLK` | `--best-order --p3-maxreg-a 64 --batch-b 131072` | 200.2 | 282.1 | 161.2 | 643.4 | 3.10 | 1.09x |
| v5 | `M5_NOCLK` | `--e4-best` (batches 1M / 512k / 128k) | 188.9 | 292.2 | 160.8 | 641.9 | 3.10 | 1.10x |
| **v4** | **default** | **`--best-order --p3-maxreg-a 64 --batch-b 131072`** | **200.1** | **235.8** | **167.5** | **603.4** | **3.30** | **1.17x** |

Phase 3 per lane (ms; phase 1 and the scan are within 1 ms between builds):

| | A | M | B |
|---|---|---|---|
| default, `@STD` (v0 / v6) | 98.6 / 98.4 | 171.2 / 169.9 | 178.6 / 179.2 |
| `M5_NOCLK`, `@STD` (v1 / v7) | 99.1 / 99.1 | **208.9 / 208.7** | 173.9 / 174.5 |
| default, best (v4) | 87.2 | 143.9 | 129.3 |
| `M5_NOCLK`, best (v3) | 88.1 | 189.6 | 123.3 |

1. **`M5_NOCLK` is slower, not faster: M5 +4.5%** (734.9 / 736.3 vs 704.4 / 703.2 ms), reproduced within the job. Lane M's
   phase 3 is 22% slower without the counters (208.9 vs 171.2 ms; with lane M's `nw-desc` order 189.6 vs 143.9 ms, +32%), at
   the same 79 registers and stack; lane A is unchanged (+0.5%) and lane B 2.6% faster (173.9 vs 178.6 ms). The `clock64()`
   reads are volatile and order the code around them; without them ptxas schedules the stitching code differently (also
   visible under the 64-register cap, whose spills grow from 64 / 92 to 100 / 140 B), and the result depends on the lane's
   caps: lane M (NW 64, NWA 8) loses, lane B (72.9% of its cycles in stitching) gains a little. The counters themselves
   cost no registers. Not kept as a default; the macro stays for experiments. A per-lane switch (no
   counters in lane B only) would save about 6 ms (1%).
2. **E4's flags on the merged code (counters on): 603.4 ms, 3.30 M/s, 1.17x** against the merged default (v4 vs v0 / v6).
   Lane A -5.9% (`--p3-maxreg-a 64`), lane M -12.1% (`--best-order`: prefix order, `nw-desc` in lane M), lane B -25% (one launch,
   128k batch, 49.9 GB of B Work). The same flags with `M5_NOCLK` (v3) lose lane M's gain (282.1 ms).
3. `--e4-best`'s bigger A / M batches (v5 vs v3, both without counters): lane A -11.3 ms (1M batch), lane M +10.1 ms (512k
   batch): a wash in total, at 19.2 GB (A) + 8.9 GB (M) more Work. Not measured with the counters on.
4. Compared with round 5 (pre-merge, 709.3 ms with `--e4-best`): E5's stitch variants and E4's flags compose; the gate
   configuration of the merged code goes from 704 to 603 ms with E4's flags.

## What to keep

- **`--best-order`** (= `--presort --p3order-m nw-desc`): the best exact order, -4.5% on the base kernel (831.0 vs 870.4 ms),
  -3.0% on the tail kernel (rounds 1-2). For the engine: after phase 1 of a lane-M batch, one stable sort of the batch's
  phase-3 slots by window count (descending); lanes A and B keep the prefix order. Cost: `k_key` + a thrust stable radix sort
  of 64-bit keys per batch, included in the times above.
- **`--e4-best`**: `--best-order`, lane A's `k_p3` at 64 registers, batches 1M / 512k / 128k: 709.3-711.6 ms, 2.80-2.81 M/s,
  1.22x (round 5), at 80.6 GB of device memory (H100 NVL only). Of the 22% the order is 4.5 points, the register cap on A about 1.5, and
  the batch sizes the rest.
- **On the merged code: `--best-order --p3-maxreg-a 64 --batch-b 131072`** with the default build (counters on): 603.4 ms,
  3.30 M/s, 1.17x the merged gate configuration (job 795837).
- **`-DM5_NOCLK` stays off**: it compiles out the `clock64()` phase counters but makes lane M's phase 3 22% slower and M5
  4.5% slower (job 795837). The counters cost no registers and no time; the warp-collective tail code (below) does.
- **Keep instrumentation out of production kernels**: warp-collective timing code after the read costs 14 registers and 18%.
  The engine's `k_p3` should be checked for the same (register count of the deployed kernel vs this base: 80 in A and M).
- **Not kept** (all exact, all measured): `--bin` and cost descending (1.30-1.31x slower), longest-first from a counter
  (1.39x), control-flow signature (1.07x), genome-locality orders (1.11-1.15x), warp-granular longest-first (+2%, -0.4% on A
  only: not worth a sort), the persistent queue inside a batch (+2%), one warp per block (0%), `--compact` (+1%). `--bin`
  should not be turned on in the engine.
- **The fused persistent lane** was a memory option only (`--fused`, `--fused-bps-*`, `--fused-lb`; dropped in the merge,
  code in bf8746492a): no batch tail and
  far less Work (A and M 5x less at 3 blocks/SM; A / M / B 15x / 16x / 3.9x less at 1 block/SM), at 1.6-1.85x the time (its
  kernel also carries the tail code, so those factors are measured against the tail-code staged kernel).

## Expected gain in the engine

Measured on M5 with the base kernel (round 4; one 2 M-read set; the engine maps 16 M-read chunks with the same lane code):

| change | M5 ms | M distinct/s | vs v0 | status |
|---|---|---|---|---|
| v0: engine order (prefix), engine batches 262k / 262k / 64k | 870.4 | 2.29 | 1.00x | measured |
| best order (`--best-order`) | 831.0 | 2.40 | 1.05x | measured |
| best order + persistent queue inside each batch | about 848 | 2.35 | 1.03x | the queue costs 2.0% (measured twice, on the tail kernel) |
| best order + batches 512k / 384k / 128k | 727.8 | 2.74 | 1.20x | measured (v5best) |
| `--e4-best` (+ lane A at 64 registers, batches 1M / 512k / 128k) | 709.3-711.6 | 2.80-2.81 | 1.22x | measured (round 5, against v0 863.1-868.5 in the same job) |
| bound for any phase-3 slot filling | about 595 | 3.35 | 1.46x | idle phase-3 warp-slot time (busy / span A 0.57, M 0.70, B 0.35) applied to the base kernel's phase 3 |

So **combining the best ordering with a persistent queue in the engine is worth about 1.03x**: +4.5% from lane M's order,
-2% from the queue. The queue adds nothing because the waste is not imbalance inside a batch: the hardware block scheduler
already refills slots, and what is left is the critical path of each batch (its longest warp, which starts at t = 0) and
lane B's single sub-wave launch. E1's estimate of a queue floor 277 ms below today and the 1.46x bound above count idle slot
time; neither is reachable by handing out whole reads, because the longest warp is already running from the start.

What would capture it (not measured here):
1. **SIMT efficiency inside phase 3** (6% today): warps that work on one read together (E3's warp-parallel stitching), or
   re-packing the still-running reads of a batch into full warps at step boundaries. This is the only lever on the critical
   path.
2. **Overlap instead of a queue**: lane B of chunk i (one sub-wave launch, 35% busy) concurrently with lane A of chunk i+1 on
   a second stream; bound about 139 ms of B's 214 ms phase 3; needs both lanes' Work resident.
3. **Split Work**: per-read phase-1 state for the whole batch and phase-3 scratch per resident thread, so the 1M / 512k /
   128k batches no longer need 46-635 KB of Work per read (and the 40 GB question below changes).

### Memory: does the engine fit a 40 GB GPU?

Work per batch read is 46,072 B (A), 82,552 B (M), 635,216 B (B); at the engine's batches that is 12.2 / 21.8 / 41.7 GB, and
the harness peaks at 64.6 GB of device memory (in lane B; about 23 GB is genome, SA, SAi, reads and outputs). The faster
configurations need more: 80.6 GB with lane B in one launch. The device footprint is set by lane B's Work, not by the index.
At a 40 GB-class budget (3.1 / 2.8 / 10.7 GB of Work) round 2 measured, on the tail-code kernels: fused lanes at 4 / 2 / 1
blocks/SM (f40) 1783.7 ms (1.12 M/s), small staged batches 64k / 32k / 16k (s40) 2098.3 ms (0.95 M/s), i.e. 1.7-2.0x slower
than the 94 GB configuration of the same kernels. So **a 40 GB fit is possible but costs about 1.7x** with today's Work
layout; the engine's own chunk output buffers (16 M x NBLOB, about 10.7 GB for lane A) must shrink too. A split Work layout
(point 3 above) is what would make 40 GB cheap.
