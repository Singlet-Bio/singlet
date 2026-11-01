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
- **lane balance** = sum of the threads' busy time / (32 x sum of warp lifetimes): idle lanes of live warps (a warp lives until
  its slowest read is done).
- thread-slot utilisation = the product of the three.

Fixed settings in every variant (the engine's): budgets 1024 / 2048 / 2048 steps, batches 262,144 (A), 262,144 (M),
65,536 (B) reads, 128 threads per block; host lane on 24 threads.

## Jobs

## Results

## What to keep

## Expected gain in the engine
