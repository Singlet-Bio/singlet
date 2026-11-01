# E4: phase-3 read order and batch scheduling in M5 (wave 1, questions Q5 and Q3)

Branch `gpu/fable/order` (from `gpu/fable-engine` cf92cdb9). Harness: `bench/gpu/m5/m5.cu` on the 2 M pbmc1k_v3 dump set
(every 16th distinct cDNA sequence, 1,991,926 sequences), Clipper H100 NVL, CUDA 13.3 image. Status: in progress.

## Method

All variants are exact by construction: a read's result depends only on the read, every variant only changes which thread or
time slot handles which read, and every result is written by read index. Each variant still runs the byte-for-byte comparison
with STAR in the same job.

Options added to `m5` (all off by default; the default binary behaves as before):

| option | what it does |
|---|---|
| `--presort` | the engine's order: stable sort of the truth reads by their first 32 bases (`engine_gpu.cu` `k_prefix_key`) before mapping. Without it the harness maps in truth-file order (24 sorted runs, one per STAR thread) |
| `--bin` | unchanged: after phase 1, phase-3 threads take the batch's reads sorted ascending by `nW << 40 \| nEntries` (cheap reads first) |
| `--p3order K` | other phase-3 keys, stable radix sort per batch: `cost-asc` (= `--bin`), `cost-desc` (most windows first, then most SA entries; reads without phase 3 last), `nw-desc` (windows only), `locus` (genome bin of the read's first window, `WC[0][WC_gStart]`), `anchor` (genome position of the first anchor's first locus), `nwlocus` (window-count class 1, 2, 3-4, 5-8, ... descending, then the first window's bin) |
| `--queue` | phase 3 as a persistent kernel: grid = SMs x resident blocks (occupancy API); each warp takes the next reads of the batch, in the `--p3order` order, from an atomic counter (warp-aggregated) |
| `--fused A\|M\|B` | the lane as one persistent kernel with Work per resident thread: a warp takes the next 32 reads (prefix order) from an atomic counter, runs phase 1 (thread per read), scans the SA entries of each of the 32 reads warp-cooperatively (k_p2's loop, ballot into a per-thread bit buffer of `--fused-bits` entries, default 16,384), then phase 3 (thread per read). A read with more entries than the buffer runs `assignAll` itself (the same result). No batches, no host syncs inside the lane |
| `--fused-bps-a/m/b N` | resident blocks per SM of the fused kernel (0 = occupancy limit): the Work memory knob |
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
| 795289 | round 2: reference builds, per-lane order, batch sizes, lane B in one launch, fused at 4-5 blocks/SM | `m5-8885087332` |

Results (JSON per variant, logs): `$W/jobs/fable/order/results/<job>/`, copied to `bench/fable/order/results/<job>/`.
Table: `python bench/fable/order/e4_table.py bench/fable/order/results/795240 v0`.

Registers (ptxas, CUDA 13.3, sm_90): `k_p1` 95, `k_p2` 42, `k_p3` 94, `k_p3q` 94-96, `k_key` 20 (all lanes): 5 blocks of 128
per SM for phase 1 and phase 3 (2,640 resident warps). `k_fused` 134-144 (3 blocks/SM, 1,584 warps); with `--fused-lb 4` 128
registers (40-56 B spills, 4 blocks/SM), `--fused-lb 5` 96 registers (160 B spills, 5 blocks/SM).

## Results, round 1 (job 795240)

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

ROUND2_PLACEHOLDER

