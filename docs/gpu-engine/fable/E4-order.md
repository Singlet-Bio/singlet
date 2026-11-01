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
| `--p3order K` | other phase-3 keys, stable radix sort per batch: `cost-asc` (= `--bin`), `cost-desc` (most windows first, then most SA entries; reads without phase 3 last), `nw-desc` (windows only), `locus` (genome bin of the read's first window, `WC[0][WC_gStart]`), `anchor` (genome position of the first anchor's first locus), `nwlocus` (window-count class 1, 2, 3-4, 5-8, ... descending, then the first window's bin), `sig` (the control-flow signature known before phase 3: stitched, nW, nP, nA, descending; ties keep prefix order), `warp-desc` (the warps of the prefix order, 32 consecutive reads, kept whole and in order, run heaviest first by the sum of their windows, then SA entries: longest-first at warp granularity without breaking coherence) |
| `--p3order-a/m/b K` | the same keys for one lane only (overrides `--p3order` for that lane) |
| `--best-order` | E4's best exact order as one flag: `--presort --p3order-m nw-desc` (prefix order everywhere, lane M's phase 3 by window count); later flags override |
| `--p3-block N` | threads per block of the phase-3 kernels (32, 64, 128; default 128). 32 = one warp per block: a finished warp frees its slot instead of waiting for the other 3 warps of its block |
| `--p3-maxreg R` | `k_p3` under `__launch_bounds__` capping registers at 64 or 80 (94 without; ptxas: 64 = 148-160 B spills, 80 = 60-68 B; with 32-thread blocks the 80 cap compiles to 72) |
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
| 795335 | round 3 (`ROUND=3`): the E1 follow-ups (32-thread blocks, longest-first from a counter, `sig`, `warp-desc`, register caps, E1's batch sizes) and ncu passes on `k_p3` | `m5-f8c2ae0d12` |

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

## Results, round 2 (job 795308)

ROUND2_RESULTS

## What to keep

- **Order: prefix order (`--presort`) for every lane, plus `--p3order-m nw-desc` for lane M**, now one flag,
  `--best-order`. Exact (a read's result never depends on its slot). Round 1: lane M 423.8 -> 391.8 ms, A and B unchanged,
  M5 1031 -> 999 ms (1.93 -> 1.99 M distinct/s, -3.1%). In the engine this is one stable radix sort of lane M's batch by
  window count after phase 1 (here `k_key` + a thrust stable radix sort of 64-bit keys per batch, inside the timed region).
- **Not kept:** `--bin` (cost ascending: 1.38x slower), cost descending / longest-first (1.35-1.39x slower), genome-locality
  orders (1.11-1.15x slower), the persistent queue inside a batch (1.6% slower, and slower again with any non-prefix order).
  `--bin` should not be turned on in the engine.
- **Batch size is the scheduling lever that pays**: 512k / 384k reads for A / M (v4big) -6% at +12 / +11 GB of Work.
- **The fused persistent lane** is kept only as a memory option (`--fused`, `--fused-bps-*`, `--fused-lb`): no batch tail and
  far less Work (A and M 5x less at 3 blocks/SM; A / M / B 15x / 16x / 3.9x less at 1 block/SM), at 1.6-1.85x the time.

## Expected gain in the engine

Measured on M5 (one 2 M-read set; the engine maps 16 M-read chunks with the same lane code and batch sizes):

| change | M5 ms | M distinct/s | vs v0 | status |
|---|---|---|---|---|
| v0: engine order (prefix), fixed batches | 1031.0 | 1.93 | 1.00x | measured |
| + lane M nw-desc (`--best-order`) | 999.0 | 1.99 | 1.03x | measured (round 1, lanes are independent) |
| best order + persistent queue inside each batch | about 1015 | 1.96 | 1.02x | v3q: the queue adds 16 ms (+1.6%) |
| best order + bigger batches 512k / 384k, no queue | about 942 | 2.12 | 1.09x | estimate: v4big with lane M x 391.8 / 423.8 |
| upper bound of any phase-3 scheduling fix | about 780 | 2.55 | 1.32x | idle phase-3 warp slots of v0t: A 29, M 52, B 167 ms |

So **the best ordering combined with a persistent queue in the engine is worth about 3%, not the 1.2-1.5x of the study's
section 4.6 estimate**, and the queue contributes nothing. The reason is that the waste is not imbalance inside a batch (the
hardware block scheduler already balances that; warp balance 0.69-0.88), it is the end of every batch and lane B: 24% of the
phase-3 warp-slot time is idle, 2/3 of it in lane B, whose 90,740 reads fill only 0.39 of the slots for 273 ms. A queue over
one batch cannot fill that, and a persistent kernel across batches (the fused variant) removes it but pays 1.85x per read.

What would capture it, in order of expected value (none measured here):
1. **Overlap instead of a queue**: run lane B of chunk i (and the last wave of every M batch) concurrently with lane A of
   chunk i+1 on a second stream. Lane B is one-wave and latency-bound, so the other 61% of the slots are free; bound
   about 167 ms of 1031 (16%), needs B's Work and A's Work resident together (about 54 GB at today's batch sizes).
2. **Fewer, larger batches** where memory allows (round 2: lane B in one launch).
3. **Split Work**: per-read phase-1 state (small) for the whole batch, and phase-3 scratch per resident thread, so batches can
   be large without the 46-635 KB per read of today's Work; this is what makes the fused kernel's memory figures (0.8 / 1.4 /
   10.7 GB) reachable without its 1.85x per-read cost.

### Memory: does the engine fit a 40 GB GPU?

Work per batch read is 46,072 B (A), 82,552 B (M), 635,216 B (B); at the engine's batches that is 12.2 / 21.8 / 41.7 GB, and
the harness peaks at 64.6 GB of device memory (in lane B; about 23 GB is genome, SA, SAi, reads and outputs). The device-side
footprint of a 94 GB run is therefore set by lane B's Work, not by the index. A 40 GB fit is plausible only by shrinking lane
B's Work: B batches of 16k reads (10.4 GB; round 2 measures the cost: `s40`) or the fused lane B at 1 block/SM (10.7 GB,
device peak 33.7 GB measured in round 1, 1.84x slower overall). The engine's own chunk output buffers (16 M x NBLOB, about
10.7 GB for lane A) must shrink as well; they are not in the harness.

