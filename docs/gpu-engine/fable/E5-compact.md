# E5: compact per-read stitching state in lane A

Branch `gpu/fable/compact` (base cf92cdb9, `gpu/gpu-slots2`). Code: `bench/gpu/m5/stitch.cuh`, `bench/gpu/m5/m5.cu`; jobs:
`bench/gpu/m5/jobs/e5_compile.sbatch` (variants built side by side, ptxas -v kept) and `bench/gpu/m5/jobs/e5_run.sbatch`
(truth regenerated in the job, every variant in one job, optional Nsight Compute on lane A's `k_p3`).

## Question

Lane A (reads with ≤ 16 windows, 100% of the 1,991,926 distinct reads enter it) keeps one `Work<LaneA>` per read in
global memory: 46,104 B, of which 25.5 KB are 35 transcripts (`Tr<8>`, 728 B each: 17 on the recursion stack, 9 window
slots, 8 kept, 1 finalization scratch) and 14.3 KB the window aligns `WA`. One thread maps one read, so a warp's 32
accesses to the same field fall 46 KB apart: 32 sectors for 32 useful words. The hypothesis (AoS): stitching is bound by
this traffic, so copying fewer transcripts, making them smaller, or laying the state out field-major would cut lane A's
time. Gate (study doc): lane A ≤ 0.7× baseline, with results byte-identical to STAR.

## Method

- Data: pbmc1k_v3, the standard 2 M distinct-sequence set (MAXD = 2,000,000: every 16th of the 31,870,813 distinct cDNA sequences,
  1,991,926 sequences carrying 4,118,999 of the 66.6 M reads), STAR 2.7.11b truth regenerated in every job (decode, distinct, stardump, ~3 min).
- GPU: H100 NVL (Clipper), CUDA 13.3, `-O3 -arch=sm_90 -fmad=false`. Runs: `--reps 3` after a warm-up pass, median;
  `--budget-a 1024 --budget-m 2048 --budget-b 2048 --batch-a 524288 --batch-m 393216 --batch-b 65536` (the production
  staged configuration). Identity: every distinct read's encoded result compared byte for byte with STAR's.
- Variants are compile-time macros (`M5_*`, all off = the baseline) or run-time flags, so a baseline and its variants run
  in the same job on the same node.
- Cycle shares come from `clock64` intervals per thread (`clk[0..7]`). An interval includes the time the thread's warp
  spends on other lanes' paths, so the shares say where a thread *waits*, not where the warp *works*; the sub-timers
  inside stitching (stitchAlignToTranscript, finalization, copies) time only the thread's own sections.

## S1: one kernel, `Work` in global memory vs in local memory (job 795235, commit 016f650)

`--mono` runs lane A as one kernel (`k_lane`, persistent threads, 528 blocks × 128 threads = 67,584 threads, each looping
over reads); `--local` keeps that thread's `Work` in local memory instead of a global array. Local memory is the
hardware's own field-major layout: word *i* of the 32 threads of a warp sits in 32 consecutive 4-byte slots, so a warp
access to the same field of 32 reads is one or a few 128-B lines, and it is cached in L1. The two runs execute the same
code on the same reads; only the address of `Work` changes.

| lane A, one kernel | ms (median of 3) | thread cycles: seeds / windows / stitching / select | inside stitching: stitchAlignToTranscript / finalization / stack copies |
|---|---:|---|---|
| `--mono`, Work global, 4 blocks/SM | 978.3 | 41.1 / 10.6 / 44.4 / 3.9 % | 7.5 / 35.7 / 4.9 % |
| `--mono --local`, 4 blocks/SM | **633.7** (1.54× faster) | 71.2 / 18.6 / 9.4 / 0.8 % | 6.7 / 32.8 / 4.1 % |
| `--mono`, Work global, 8 blocks/SM | 904.0 | 48.4 / 12.8 / 36.0 / 2.7 % | 8.3 / 33.4 / 4.8 % |
| `--mono --local`, 8 blocks/SM | **618.3** (1.46× faster) | 71.2 / 18.7 / 9.3 / 0.8 % | 7.3 / 32.7 / 4.0 % |
| staged (production: `k_p1`, `k_p2`, `k_p3`, one thread per read, Work global) | 244.1 (phase 1 98.6, scan 14.2, phase 3 131.3) | 50.6 / 8.7 / 39.8 / 0.9 % | 1.9 / 10.2 / 1.1 % |

Local is **1.54× faster than global** in mono mode, above the 1.2× bar (E1 measured the same pair at 1.60×: 626.7 vs
1002.6 ms). Scaling each share by its run's time, stitching's thread cycles fall about 7× (0.444 × 978 → 0.094 × 634)
while seeds (0.411 × 978 → 0.712 × 634) and windows do not improve. Identity 100% in every run.

The speed-up is real but its cause is not the one the hypothesis names. E1's Nsight Compute comparison of the two
builds: the local build executes the same thread instructions in 25% fewer warp instructions (1.85 vs 1.44 active
threads per instruction) with 45% fewer instruction fetches, and moves **4.3× more** DRAM bytes. At ~2 active lanes per
instruction a warp access costs ~1 sector per active thread whatever the layout (E1: 2.2-2.4 sectors per load request,
2.8-3.6 per store), so coalescing has almost nothing to gain; the local build wins on convergence and instruction
fetch. **Verdict on AoS: not the cost.** A field-major transposition (S5 as posed) is not pursued; S5 below measures
the cheap version, phase-local state in thread-local memory, which E1 also proposes for its own reasons (k_p3's
scratch never needs to be in a global array).

The staged lanes are already 4× faster than mono global (244 vs 978 ms) because they run one thread per read over
batches of 524,288 reads (full occupancy, no persistent-thread tail), so S1's 1.54× says how much the layout costs the
stitching code, not what it costs the staged lane. S5 measures that directly.

## S2: transcript copies in lane A (job 795235, `M5_COPYSTATS`)

Every `Tr::copyFrom` is counted by kind, with the bytes it moves (scalar fields + live exon rows) and its cycles.

| copy | per read | bytes per read | share of stitching cycles |
|---|---:|---:|---:|
| include (stack[d] → stack[d+1]) | 4.577 | 1,057 | 0.5 % |
| exclude (stack[d] → stack[d+1]) | 4.577 | 1,057 | 0.5 % |
| finalization scratch (stack → fin) | 3.537 | 984 | 1.8 % |
| record (fin → window slot) | 2.967 | 821 | 1.4 % |
| slot init (trInit → slot[0], per window) | 2.541 | 536 | 0.3 % |
| keep (slot → keep, per window) | 1.144 | 317 | 0.9 % |
| keep drop (keep compaction) | 0.177 | 49 | 0.1 % |
| multMapSelect compaction | 0 | 0 | 0 |
| **total** | **19.52** | **4,821 (247 B per copy)** | **5.4 %** |

18.77 recursion steps and 2.54 stitched windows per read. A lane A read moves 4.8 KB of transcripts (19.5 copies),
against 46 KB of `Work` and ~450 suffix-array entries scanned.

By the clock, copies are 5.4% of stitching cycles, and stitching is 41% of lane A's thread cycles: removing every copy
would buy ~2% of lane A. The clock understates them. A store retires without waiting for memory; its cost appears
later, as throttling of whichever instruction finds the load/store queue full. E1's Nsight Compute sampling
(`docs/gpu-engine/fable/E1-profile.md`, job 795242) charges `Tr::copyFrom` with 19-22% of k_p3's stall samples in
every lane and about half of k_p3's L2 sectors (k_p3 writes 2.4-2.7× as many L2 sectors as it reads). S3 is therefore
measured by lane A and k_p3 time, and by k_p3's L2 write sectors (ncu), not by the clock.

## S3: fewer copies, exactly (commits 0a05f46, 8382429)

Four compile-time variants; `M5_S3` = (a) + (b) + (c).

- **(a) `M5_REFX`, the exclude branch by reference.** STAR's recursion passes `trA` by value, so the port copied
  `stack[d]` to `stack[d+1]` for the exclude call as well as for the include call. With REFX a frame carries `src`, the
  stack entry that holds its `trA`: an include child gets a fresh copy (`src = d+1`, the one copy left), an exclude child
  inherits its parent's `src` and no copy is made. *Exactness:* `stack[k]` is written only by the include step of frame
  k−1, which runs once, before frame k−1's exclude step and before any of frame k−1's descendants run. While frame d is
  active, the frames that can run are its descendants, which write `stack[d+1..]` only; `src(d) ≤ d`, so `stack[src(d)]`
  holds, at every read, exactly the transcript the copy would have held. The include copy reads `stack[src(d)]` and
  writes `stack[d+1]`, never the same entry. Both branches still visit the same frames in the same order.
- **(b) `M5_FIN1`, finalization copies once.** The port copied `trA` to a scratch `fin`, finalized it, and the recording
  copied `fin` into the slot the permutation `slotP` hands it. With FIN1 the window has one physical slot more, outside
  the permutation (`slotSpare`): the candidate is copied there once and finalized in place; recording at position iTr
  sets `slotP[iTr] = slotSpare` and makes the displaced physical slot `pTr` (STAR's `trArrayPointer[nWinTr]`, the entry
  STAR's copy overwrites, outside the live positions) the next spare. *Exactness:* through `slotP[0..nWinTr−1]` every
  live position names a slot holding the same bytes as in the baseline; the spare is never a live position, so no
  comparison (`blocksOverlap`, the score tests) can see the candidate as an old transcript; a candidate that
  finalization rejects stays in the spare and is overwritten by the next one, unread. Deletions move a live slot to the
  dead position `nWinTr`, as before.
- **(c) `M5_SLOT0`, no slot-0 init copy.** Each window copied `trInit` into slot 0. While `nWinTr == 0` the only read of
  `slot[slotP[0]]` is `.maxScore` in the recording test (`Score + range >= trAll[0]->maxScore`); every other slot read
  is at a position < `nWinTr`, written whole by a recording, and a window that records nothing is skipped before `t0`
  is read. SLOT0 sets just `maxScore`.
- **`M5_KEEPIX`, kept transcripts by index** (needs FIN1). `keep[]` becomes an index list into the slot pool (pool of
  NWTR + 2 + NKEEP slots, `own[]` marks the entries held by `keep`). Keeping a window's transcript moves its index and
  marks it owned; the next window's positions and spare are the first unowned entries (at most NKEEP are owned, so
  NWTR + 2 remain); dropping a kept transcript clears its bit; multMapSelect's compaction moves indices. *Exactness:* an
  owned entry is never a position or the spare, so stitching never writes it; the writes the baseline makes to `keep[k]`
  (Chr/Str/roStr at keep time, roStart/cStart/primaryFlag in multMapSelect) go to `slot[keepP[k]]` through one macro.
  This removes the keep, keep-drop and select copies (1.32 per read).

## S4: 32-bit transcript fields (`M5_TR32`, commit 0a05f46)

STAR's `uint` is 64-bit; the port kept it. TR32 stores every transcript field that holds a read offset, length or
count (exon `EX_R`, `EX_L`, `EX_iFrag`, `EX_sjA`, `shiftSJ`, `intronMotifs`, `nExons`, rStart/roStart/rLength,
Chr/Str/roStr, nMatch/nMM/mappedLength/extendL, the gap and indel counters, nUnique/nAnchor) in an `int32_t`, written by
truncation and read back sign-extended to 64 bits. Genome coordinates (`EX_G`, gStart, gLength, cStart) stay 64-bit.
*Exactness:* arithmetic still happens in 64 bits on the read-back value, exactly as in STAR, so the port is exact iff
every stored value round-trips: every value v with −2³¹ ≤ (i64)v < 2³¹ does, including STAR's unsigned wrap-arounds
(`sjA = (uint)−1`, small negative intermediates). The bounds are read length (≤ 650), exon counts (≤ 20), gap lengths
(< a window's span, < 2²⁴), sjdb and chromosome indices (< 2³¹). `M5_TR32CHECK` counts every store that does not
round-trip, on the device and on the host (the heavy lane runs the same code with STAR's limits); it must be 0.
`Tr<8>` shrinks from 728 B to 440 B, and `Work<LaneA>` by ~10 KB.

## S3 and S4 results (job 795250, commit 0a05f46, node g050, all variants in one job)

| variant | Work A (B) | Tr<8> (B) | lane A ms | phase 1 / scan / phase 3 ms | lane A vs base | lane M ms | lane B ms | GPU M distinct/s | k_p3 A regs | identity |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| baseline (first) | 46,104 | 728 | 245.6 | 98.5 / 14.6 / 132.4 | 1.000 | 321.1 | 256.5 | 2.42 | 80 | 100% |
| (a) REFX | 46,104 | 728 | 235.5 | 98.5 / 14.6 / 122.0 | 0.959 | 307.0 | 246.0 | 2.53 | 80 | 100% |
| (b) FIN1 | 46,112 | 728 | 236.2 | 98.9 / 14.6 / 122.7 | 0.962 | 305.7 | 245.6 | 2.53 | 80 | 100% |
| S3 = (a)+(b)+(c) SLOT0 | 46,112 | 728 | 228.4 | 98.8 / 14.6 / 114.5 | 0.930 | 288.6 | 237.2 | 2.64 | 80 | 100% |
| S4 TR32 alone | 36,024 | 440 | 230.2 | 98.3 / 14.6 / 117.5 | 0.938 | 289.4 | 238.7 | 2.63 | 79 | 100% |
| **S3 + S4** | **36,032** | **440** | **220.3** | 98.3 / 14.6 / **107.3** | **0.898** | **274.1** | **223.8** | **2.77** | 79 | 100% |
| S3 + copy statistics | 46,336 | 728 | 234.6 | 98.8 / 14.6 / 121.1 | (instrumented) | 296.1 | 241.6 | 2.58 | 80 | 100% |
| S3 + S4 + TR32CHECK | 36,032 | 440 | 261.7 | 98.4 / 14.6 / 148.6 | (instrumented) | 337.2 | 253.5 | 2.34 | 80 | 100%, 0 bad stores (device and host) |
| baseline (last) | 46,104 | 728 | 245.2 | 98.5 / 14.6 / 131.5 | 0.998 | 321.6 | 255.8 | 2.42 | 80 | 100% |

Lane A ms are medians of 3; the two baselines bracket the job (245.6 and 245.2 ms). No variant spills (k_p3 576-B
frame, 0 spill stores/loads); thread-cycle stitching share: 39.8% (base), 35.6% (S3), 33.7% (S3 + S4).

- Copies (S3 + statistics): include 4.577, exclude 0, finalize 3.537, record 0, slot init 0, keep 1.144, keep drop
  0.177 per read: **9.44 copies and 2,407 B per read, half the baseline's 19.52 and 4,821 B.** KEEPIX removes the keep
  copies as well (1.32 per read; measured in job 795320).
- **Phase 3 of lane A: 132.4 → 107.3 ms (−19%)**; lane A 245.4 → 220.3 ms (−10%, 0.90×); lanes M (−15%) and B (−13%)
  gain as much or more, since they copy more per read. GPU throughput 2.42 → 2.77 M distinct reads/s (+14%).
- The parts add up roughly (REFX −10 ms, FIN1 −9.4 ms, S3 −17 ms, TR32 −15 ms, S3 + S4 −25 ms): two cheaper copies of a
  smaller transcript, but REFX and FIN1 overlap (both shrink the same store stream).
- **Gate (lane A ≤ 0.7× = 172 ms): not met (0.90×).** Phase 1 (seeds and windows, 98.5 ms) and the scan (14.6 ms) do
  not change, so 0.7× needs phase 3 ≤ 59 ms, 0.44× of its 132 ms; S3 + S4 bring it to 107 ms (0.81×).

## Word-wise output encoder (`M5_BLOBW`, commit 00a4c32; added after E1)

E1 found the byte-wise varint encoder (`Blob::putV`/`put`, one byte store per output byte, each thread writing its own
640-B slot) at 9.4% of lane A's stall samples. BLOBW gathers the bytes in a 64-bit register and stores whole aligned
words. *Exactness:* byte j of the register is the j-th byte put since the word began, and the GPU is little-endian, so
memory holds the same bytes in the same order; `n` and the overflow rule (`full` once `n` reaches `cap`) are unchanged.
`flush()` at the end of encoding stores the last partial word, so the only extra bytes are zeros between `n` and the
next multiple of 8: inside the read's slot (`NBLOB` is a multiple of 8, asserted) and past the length the comparison
reads. Cost: k_p3 lane A goes from 80 to 94 registers (24 → 20 resident warps per SM at 128 threads per block); with
TR32 it stays at 79.

**Result: a regression** (job 795345, same job as the S5 table; lane A median of 3):

| variant | lane A ms | phase 3 ms (last run) | k_p3 regs | vs the variant without BLOBW |
|---|---:|---:|---:|---|
| baseline | 249.1 / 249.8 | 134.6 / 136.0 | 80 | |
| BLOBW | 264.2 | 149.8 | 94 | +15 ms |
| S3 + BLOBW | 248.6 | 134.0 | 94 | +20 ms (S3 alone: 0.93× of its baseline in job 795250) |
| S3 + KEEPIX + TR32 + BLOBW | 230.8 | 115.6 | 79 | 0.925× against 0.898× for S3 + TR32 in job 795250 (KEEPIX and BLOBW together) |

The register growth explains most of it where it happens (94 registers), but the TR32 build at 79 registers still
loses. Lane A's k_p3 is instruction-fetch bound (E1: 51% of stalls, i-cache hit 54%), and the word encoder's `put` is
inlined at every `putV` call site of `encodeTr`: more code in a kernel that is short of i-cache. A word-wise encoder
can still win if it is small: one non-inlined routine over an array of the record's values, or the record's fields
staged to a contiguous buffer and varint-packed in one loop. Not done here.

## S5: phase state in thread-local memory (`--p3local`, `--p1local`; commit 6745dd3)

A field-major transposition of `Work` was not built (E1's profile and S1 above say coalescing is not the cost). The
cheap form of the same idea was: run a staged phase on a per-thread `Work` in local memory, the hardware's own
field-major layout, which is also E1's proposal to keep k_p3's scratch out of the global array.

- `--p3local`: `k_p3` copies the state that crosses kernels (everything but `WA` and the transcript arrays:
  6.3 KB with TR32, 6.6 KB without) and the read buffers from the global `Work` into a local one, and runs phase 3
  there. `--p1local`: `k_p1` runs phase 1 on a local `Work` and writes the crossing state back (`copyPersistent`).
- *Exactness:* the global `Work` buffers are reused across batches without clearing, so the baseline already relies on
  phase 3 writing `WA` and every transcript before reading it; the local copy only removes the stale bytes. Phase 1
  starts from `wbEpoch = 0` (clears the winBin keys) and zeroed counters.

Job 795345 (commit 00a4c32, node g052, debug-gpu, all in one job; lane A median of 3, phases from the last run):

| variant | lane A ms | phase 1 / scan / phase 3 ms | lane M ms | lane B ms | GPU M distinct/s | k_p3 / k_p1 regs (frame) | identity |
|---|---:|---|---:|---:|---:|---|---:|
| baseline (first) | 249.1 | 99.8 / 14.7 / 134.6 | 317.6 | 252.4 | 2.43 | 80 / 95 | 100% |
| baseline `--p3local` | 289.5 | 99.4 / 14.7 / **176.3** | 320.2 | 253.2 | 2.31 | 80 (47,168 B) | 100% |
| S3 + KEEPIX + TR32 + BLOBW | 230.8 | 100.3 / 14.7 / 115.6 | 280.4 | 225.1 | 2.71 | 79 | 100% |
| same, `--p3local` | 238.7 | 100.4 / 14.8 / 123.3 | 279.0 | 226.8 | 2.68 | 72 (37,136 B) | 100% |
| same, `--p1local --p3local` | 248.4 | **110.5** / 14.7 / 123.4 | 279.2 | 227.1 | 2.64 | 94 (36,208 B) | 100% |
| baseline (last) | 249.8 | 99.6 / 14.7 / 136.0 | 318.4 | 255.6 | 2.42 | 80 / 95 | 100% |

**Local memory loses in the staged lane**: phase 3 +31% with 728-B transcripts, +7% with 440-B ones; phase 1 +10%.
Nsight Compute on lane A's `k_p3` (one batch of 524,288 reads, the second `k_p3` launch) shows why:

| k_p3, lane A, 1 batch | baseline | S3 + KEEPIX + TR32 + BLOBW | same, `--p3local` |
|---|---:|---:|---:|
| duration (ms) | 34.54 | **29.03** | 31.48 |
| L2 sectors read / written (M) | 417.4 / 773.4 | 285.0 / **428.7** | 705.5 / 732.3 |
| L1 global load / store sectors (M) | 761.3 / 539.9 | 559.1 / 266.5 | 771.2 / 7.4 |
| L1 local load / store sectors (M) | 25.9 / 22.7 | 25.9 / 22.7 | 455.4 / 393.0 |
| DRAM read / write (GB) | 8.75 / 4.51 | 6.01 / 2.28 | **23.03 / 10.58** |
| warp / thread instructions (G) | 4.06 / 7.81 | 3.87 / 7.52 | 3.84 / 9.50 |
| L1 / L2 hit rate | 78.5% / 81.5% | 80.4% / 79.9% | 58.1% / 57.0% |
| registers, achieved occupancy | 80, 28.0% | 79, 27.9% | 72, 33.1% |

In local memory word *i* of a thread shares a 128-B line with word *i* of 31 other threads. That is ideal when the 32
threads access the same word together, but k_p3 runs at ~2 active threads per warp instruction and its dominant access
is a thread walking its own transcript (copies, finalization, encoding). A 440-B transcript is ~14 sectors in the AoS
global `Work` and 110 sectors in local memory, each sector 7/8 other threads' data that is not wanted at that moment.
L1 and L2 hit rates fall from ~80% to ~57% and DRAM traffic grows 2.8× (E1 saw 4.3× in the mono kernel). The
copy-in of the crossing state adds 2 G thread instructions per batch but few warp instructions (it is convergent).
**A field-major layout in global memory has the same problem for the same access pattern; the AoS layout is the
better one for this code.** The mono kernel's 1.5× local win (S1) comes from convergence and instruction fetch (E1),
which the staged kernels already have.

## Summary

| step | lane A ms (vs baseline in the same job) | phase 3 ms | lane M / B ms | stitching share of lane A thread cycles | Work A (B) | k_p3 A regs, spills | identity |
|---|---|---:|---|---:|---:|---|---:|
| baseline (795250) | 245.4 (1.00) | 132 | 321 / 256 | 39.8% | 46,104 | 80, 0 | 100% |
| S1 mono global / mono local (795235) | 978.3 / 633.7 (local 1.54× faster) | | | 44.4% / 9.4% | 46,104 | 104 / 88, 0 | 100% |
| S2 copies (795235) | 19.52 copies, 4,821 B per read; 5.4% of stitching cycles by the clock | | | | | | 100% |
| S3 REFX + FIN1 + SLOT0 (795250) | 228.4 (0.930) | 114.5 | 289 / 237 | 35.6% | 46,112 | 80, 0 | 100% |
| S4 TR32 (795250) | 230.2 (0.938) | 117.5 | 289 / 239 | 36.3% | 36,024 | 79, 0 | 100%, 0 bad stores |
| **S3 + S4 (795250)** | **220.3 (0.898)** | **107.3** | **274 / 224** | 33.7% | 36,032 | 79, 0 | 100% |
| + KEEPIX + BLOBW (795345) | 230.8 (0.925) | 115.6 | 280 / 225 | | 36,040 | 79, 0 | 100% |
| S5 `--p3local`, baseline (795345) | 289.5 (1.160) | 176.3 | 320 / 253 | | 46,104 local | 80, 0 (47,168-B frame) | 100% |
| S5 `--p3local`, S3 + S4 + KEEPIX + BLOBW (795345) | 238.7 (0.957) | 123.3 | 279 / 227 | | 36,040 local | 72, 0 (37,136-B frame) | 100% |

GPU throughput (all three lanes): 2.42 → **2.77 M distinct reads/s** with S3 + S4.

## Verdict

- **AoS hypothesis: rejected as the cost.** The mono kernel is 1.5× faster with `Work` in local memory, but through
  convergence and instruction fetch (E1's ncu: 4.3× more DRAM bytes, not fewer). In the staged lanes, local memory
  (a field-major layout at 4-byte granularity) is slower and moves 2.8× more DRAM bytes, because k_p3's dominant access
  is one thread walking its own transcript at ~2 active threads per instruction: AoS keeps that walk in few sectors.
- **What pays is moving fewer bytes:** half the copies (S3) and 40% smaller transcripts (S4) cut k_p3's L2 write
  sectors by 45% and its time by 16-19%, lane A by 10%, lanes M and B by 13-15%. All exact, identity 100%, no spills,
  one register fewer.
- **Gate (lane A ≤ 0.7×): not met** (0.90×). Phase 1 (99 ms of 245) and the scan (15 ms) are outside this experiment's
  reach; 0.7× would need phase 3 at 59 ms (0.44×).

## Recommendation

1. Take **S3 + S4** (`-DM5_S3 -DM5_TR32`) into the engine: exact, −10% lane A, −15% lane M, −13% lane B, smaller
   `Work` (lane A 46 → 36 KB, lane B 635 → 549 KB, so larger batches fit).
2. Do not pursue a field-major `Work` or local-memory phases for the staged kernels.
3. Phase 1 is now the larger half of lane A (99 of 220 ms; E1: memory-bound, 71-79%): it is where a 0.7× lane A has to
   come from.
4. If the encoder is revisited (E1: 9.4% of lane A's stall samples), keep it small and out of line; the inlined word
   encoder here lost 10-20 ms.

## Jobs and commits

Branch `gpu/fable/compact`. Commits: 016f650 (copy statistics, mono budget fix, jobs), 0a05f46 (S3, S4), 8382429
(KEEPIX), 6745dd3 (`--p1local`/`--p3local`), b82c906 (compile summary), 00a4c32 (BLOBW, ncu metrics), 51a100b (run job
host threads).

| job | what | commit | node |
|---|---|---|---|
| 795234, 795249, 795257, 795279, 795319 | compiles (debug-cpu; ptxas -v in `jobs/fable/compact/results/ptxas_*.txt`) | 016f650, 0a05f46, 8382429, b82c906, 00a4c32 | |
| 795235 | S1 (mono global/local, 4 and 8 blocks/SM), S2 (copy statistics), staged baseline | 016f650 | g050 |
| 795250 | S3 parts, S3, S4, S3 + S4, TR32CHECK, copy statistics under S3 | 0a05f46 | g050 |
| 795345 | S5 (`--p3local`, `--p1local`), BLOBW, KEEPIX; ncu of k_p3 lane A | 00a4c32 | g052 (debug-gpu) |

Results: `/mnt/projects/debruinz_project/debruinz/singlet/jobs/fable/compact/results/<tag>_v<i>_<binary>.{log,json}`
(tags s1s2, s3s4, d5) and `<tag>_ncu<i>_<binary>.{txt,_raw.csv,ncu-rep}`; job logs in `singlet/logs/e5-run-<job>.out`.
