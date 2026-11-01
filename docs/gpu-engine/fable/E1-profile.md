# E1: where the M5 GPU kernels spend their time (H100 NVL, pbmc1k_v3 2 M set)

Wave 1, experiment E1 of `singlet-agents/docs/GPU_ALIGNER_STUDY_2026-10-31.md` (Part II, questions Q1, Q3, Q4).
Branch `gpu/fable/profile` (from `gpu/fable-engine` = cf92cdb9). Measurement only: the port's logic is unchanged
(stitch.cuh untouched; m5.cu gained a compile-time tail probe `-DM5_TAIL` and a `--mono-budget` knob, both inert in the
default build). Every number below carries the Slurm job that produced it. Identity was 100% in every run
(1,991,926 / 1,991,926 distinct reads, 4,118,999 / 4,118,999 reads; 151,095 STAR-unmapped and 1,840,831 mapped, all identical).

## 0. Answer in brief

Key numbers (H100 NVL, 1,991,926 distinct reads, identity 100% in every job):

| quantity | value | job |
|---|---|---|
| M5 GPU time (default batches) | 860.7 ms = 2.31 M distinct reads/s; lanes A 268.6 / M 337.4 / B 254.7 ms | 795242 |
| kernel split | k_p1 26.0%, k_p2 4.7%, k_p3 69.1%; scan, syncs, memsets, copies and host gaps together 1.3 ms (0.15%) | 795275 |
| lanes M + B | 68.8% of the time for the 26.3% of distinct reads that overflow lane A | 795242 |
| active threads per instruction (of 32) | k_p3 1.94 (A), 2.03 (M), 1.94 (B); k_p1 3.57 / 3.90 / 4.37; k_p2 15.9-17.4 | 795282, 795297 |
| top stall | A k_p3 instruction fetch 51%; M k_p3 memory 47%; B k_p3 memory 54%; k_p1 memory 71-79% | 795282, 795297 |
| occupancy (theoretical / achieved) | k_p1 31 / 22-28%; k_p3 A, M 37.5 / 28-30%; k_p3 B 31 / 19%; limiter: registers everywhere | 795282, 795297 |
| DRAM bandwidth in use | 10-24% of peak | 795282, 795297 |
| sectors per request, Work loads / stores | 2.2-2.4 / 2.8-3.6 (1.1-1.2 / 1.5-1.8 per active thread) | 795282, 795297 |
| empty warp-slot time | 39% of k_p1 + k_p3 (339 ms); longest warp = 67-100% of each k_p3 batch | 795282 |
| bigger batches (1M, 524k, 131k) | 758.6 ms (-11.9%) | 795242 |
| mono, local vs global Work | 626.7 vs 1002.6 ms (1.60x); same thread instructions, 25% fewer warp instructions, 4.3x the DRAM bytes | 795242, 795297 |
| seed stage | ~144 ms = 16.8% of M5 (k_p1 26%); 21.9 M reads/s vs 78-105 M/s standalone | 795242, 795282 |

Ranked bottlenecks (section 10): (1) SIMT divergence of the thread-per-read code, (2) batch tails and slot waste
(39% of slot-time empty), (3) serial per-read work in lanes M and B, (4) transcript copies through the global Work,
(5) the seed stage and its repetition in lanes M and B, (6) register-limited occupancy, (7) byte-wise output encoding.
Verdicts (section 9): (a) partly - copies of Work-resident transcripts are the largest k_p3 function, but the AoS layout
is not the cost at 2 active threads per instruction; (b) true; (c) true as the limiter, secondary as a cost; (d) true,
from serial work per read, not more divergence; (e) false.

## 1. Method

| item | value |
|---|---|
| data | pbmc1k_v3 first 2,000,000 reads -> 1,991,926 distinct cDNA sequences after clipping (stardump M5 truth, regenerated per job in job /tmp, deleted by trap) |
| reference | GRCh38-2024-A STAR 2.7.11b index, copied to job /tmp |
| GPU | one H100 NVL (132 SMs, 95.8 GB, SM clock 1.78 GHz under load), Clipper g050 / g051 / g052 |
| image | `cuda-13-devel.sif` (CUDA 13.3, ncu 2026.2.0, nsys from the same Nsight bundle) |
| build | `nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -Xptxas -v` (job 795229, commit 43c74812 = `m5`; tail build `m5t` with `-DM5_TAIL`, job 795274, commit d41bd534, same registers and stack per kernel as `m5`) |
| default settings | `--batch-a 262144 --batch-m 262144 --batch-b 65536 --budget-a 1024 --budget-m 2048 --budget-b 2048`, 128 threads per block, 3 timed reps after one warm-up rep |
| timing | m5's own CUDA events (median of 3 reps), job 795242 |
| timeline | nsys 2026.2 `--trace cuda`, job 795275; the image's QdstrmImporter lacks `libdw.so.1`, so the .qdstrm was imported on the host (job 795281, `e1_nsys_import.sbatch`) |
| tails | `m5t`: every thread of k_p1 and k_p3 writes `globaltimer` and `clock64` at start and end plus its SM, SA entries, windows and steps; per batch: span, warp-slot occupancy, block-slot occupancy, finish percentiles, longest warp (jobs 795242, 795282) |
| Nsight Compute | `--set full --import-source yes --clock-control none --kernel-name-base mangled`, kernel replay; k_p1/k_p2/k_p3 of one full batch per lane (A: rep 0 batch 1, 262,144 reads; M: rep 0 batch 1, 262,144 reads; B: rep 0 batch 0, 65,536 reads) (jobs 795282, 795297); the mono `k_lane` with global and local Work (job 795297); per-line view exported with `--page source --print-source cuda,sass` (ncu 2026.2 prints line metrics only in that mode; jobs 795296, 795300, 795309) |
| scripts | `bench/gpu/m5/jobs/e1_*.sbatch`, `bench/fable/profile/{tail.cuh,e1_summarize.py,e1_tables.py,ncu_summary.py,e1_lines_by_func.py}` |
| raw reports | `$W/jobs/fable/profile/results/{time-795242,ncu-795275,ncu-795282,ncu-795297}` (.ncu-rep, .nsys-rep, .sqlite, full CSV exports); summaries in `bench/fable/profile/results/` |

Caveats. (1) Lane B was profiled on the warm-up rep's first batch: its k_p1 includes the one-time clear of the
window-bin keys (first use of the Work buffer), so B k_p1 under ncu is 28.1 ms against 25.1 ms in the timed reps; k_p3
is unaffected (120.9 vs 121.7 ms). (2) ncu's stall "samples per line" were folded into functions by the line ranges of
stitch.cuh (`e1_lines_by_func.py`); the stall mix per function uses each line's top two reasons. (3) Per-thread `clock64`
times in a diverged warp include the time the warp spent on the other threads (independent thread scheduling), so the
per-read cost of the slowest reads is attributed to warps, not to single reads.

## 2. Baseline and the time split (Q1)

Job 795242 (m5 at 43c74812), medians of 3 reps; nsys job 795275 (same binary, mean of reps 1-3).

| lane | reads | lane ms | per read | phase 1 (k_p1) ms | scan + 2 syncs + k_p2 ms | phase 3 (k_p3) ms | share of M5 |
|---|---|---|---|---|---|---|---|
| A | 1,991,926 | 268.6 | 0.135 us | 105.7 | 16.8 | 145.9 | 31.2% |
| M | 524,577 | 337.4 | 0.643 us (4.8x A) | 79.2 | 17.9 | 240.9 | 39.2% |
| B | 90,740 | 254.7 | 2.81 us (20.8x A) | 38.8 | 6.8 | 209.1 | 29.6% |
| all | | **860.7** (2.31 M distinct/s) | | 223.7 (26.0%) | 41.5 (4.8%) | 595.9 (69.2%) | |

The nsys timeline (job 795275, 864.4 ms over the three lanes) splits the same way: k_p1 26.0%, k_p2 4.7%, k_p3 69.1%;
thrust scan, the two D2H syncs, memsets and k_copy_status take at most 0.05 ms each per lane (0.26 ms in all), and the
host-side gaps between GPU operations total 1.04 ms (0.12%). Nothing outside the three kernels matters. Per-batch k_p3 times (nsys, ms):
A 15.7, 18.2, 20.7, 22.0, 19.8, 20.0, 19.2, 10.9; M 114.3, 113.1, 15.4 (the third M batch holds 289 reads and still costs
5.0 + 0.5 + 15.5 ms); B 121.7, 86.7. The heavy host lane (5,304 reads) took 0.14 s on 24 host threads and runs after the GPU.

Thread-cycle shares inside the lanes (`clock64`, job 795242): A seeds 51.6% / windows 8.4% / stitching 39.1% / select 0.9%
(450 SA entries per read); M 16.4 / 15.1 / 68.3 / 0.2 (1,933); B 7.0 / 15.1 / 77.8 / 0.1 (3,117). The `copies` counter
inside stitching reads only 1.1-1.5% because it times one copy site; ncu (section 5) puts all transcript copies at 19-22%
of k_p3's stall samples.

## 3. Registers and occupancy

ptxas (job 795229; no spills in any kernel); occupancy at 128 threads per block on 132 SMs (65,536 registers per SM):

| kernel | registers | stack (local) | blocks / SM (limiter) | theoretical occupancy | resident threads |
|---|---|---|---|---|---|
| k_p1 (all lanes) | 95 (96 allocated) | 128 B | 5 (registers) | 31.25% | 84,480 |
| k_p2 (all lanes, 256 threads) | 42 (48) | 0 | 5 (registers) | 62.5% | |
| k_p3 A / M | 80 | 576 B / 320 B | 6 (registers) | 37.5% | 101,376 |
| k_p3 B | 94 (96) | 1,664 B | 5 (registers) | 31.25% | 84,480 |
| k_lane A global / A local / B (mono) | 104 / 88 / 106 | 640 B / 47,232 B / 1,728 B | 4 / 5 / 4 | 25% / 31.25% / 25% | |

Lane B's batch of 65,536 reads is smaller than the 84,480 resident thread slots of k_p1 and k_p3 B: at the default
settings 22% of lane B's slots are empty from the first cycle.

## 4. Nsight Compute per kernel and lane (Q1)

Full set, kernel replay, one whole batch per lane (A and M: 262,144 reads; B: 65,536 reads). Jobs 795282 (A, B) and
795297 (M); `results/ncu-795282/` and `results/ncu-795297/` hold `e1_ncu_kernels.csv` and `summary_ncu.txt`.

| metric | A k_p1 | A k_p3 | M k_p1 | M k_p3 | B k_p1 | B k_p3 |
|---|---|---|---|---|---|---|
| duration ms (one batch) | 13.81 | 17.70 | 38.91 | 113.36 | 28.08 | 120.92 |
| registers / blocks per SM (limiter) | 95 / 5 (regs) | 80 / 6 (regs) | 95 / 5 (regs) | 80 / 6 (regs) | 95 / 5 (regs) | 94 / 5 (regs) |
| occupancy theoretical / achieved % | 31.3 / 26.6 | 37.5 / 27.5 | 31.3 / 28.1 | 37.5 / 29.9 | 31.3 / 22.3 | 31.3 / 18.6 |
| active / eligible warps per scheduler | 4.34 / 0.29 | 4.57 / 0.16 | 4.52 / 0.26 | 4.92 / 0.20 | 3.63 / 0.15 | 3.29 / 0.20 |
| issue slots busy % | 24.1 | 14.5 | 20.8 | 17.8 | 12.9 | 17.9 |
| warp cycles per issued instruction | 18.0 | 31.7 | 21.7 | 27.6 | 28.1 | 18.4 |
| **active threads per instruction (of 32)** | **3.57** | **1.94** | **3.90** | **2.03** | **4.37** | **1.94** |
| stall 1 (share of stall cycles) | long scoreboard 71% | **no instruction 51%** | long scoreboard 78% | long scoreboard 47% | long scoreboard 79% | long scoreboard 54% |
| stall 2 | wait 14% | long scoreboard 33% | wait 12% | no instruction 34% | wait 9% | wait 18% |
| stall 3 | selected 6% | wait 10% | selected 5% | wait 12% | lg throttle 5% | no instruction 17% |
| instruction cache hit % | 85.3 | 53.6 | 93.3 | 61.1 | 96.8 | 69.7 |
| L1 hit % / L2 hit % | 79.8 / 62.6 | 78.4 / 81.0 | 74.0 / 62.3 | 76.7 / 76.3 | 74.2 / 76.1 | 79.1 / 78.4 |
| DRAM % of peak (read + write GB) | 19.0 (8.66 + 1.68) | 9.8 (4.55 + 2.28) | 23.5 (27.3 + 8.7) | 16.8 (49.7 + 25.2) | 17.6 (12.4 + 7.0) | 10.7 (33.0 + 17.9) |
| global load sectors per request (per active thread) | 2.86 (0.80) | 2.19 (1.13) | 3.63 (0.93) | 2.43 (1.20) | 4.41 (1.01) | 2.17 (1.12) |
| global store sectors per request (per active thread) | 3.69 (1.03) | 2.84 (1.47) | 3.69 (0.95) | 3.56 (1.76) | 8.00 (1.83) | 3.17 (1.64) |
| local load / store requests (M) | 4.0 / 2.2 | 7.8 / 6.9 | 6.2 / 2.3 | 58.4 / 50.8 | 1.7 / 0.6 | 57.7 / 48.6 |
| local sectors per request load / store | 1.97 / 3.16 | 1.69 / 1.68 | 1.99 / 3.52 | 1.86 / 1.77 | 1.99 / 3.55 | 1.64 / 1.56 |
| warp instructions per read | 10,786 | 7,832 | 26,794 | 64,712 | 42,122 | 215,457 |
| ns per read (batch time / reads) | 53 | 68 | 148 | 432 | 428 | 1,845 |
| L1 global load / store KB per read | 75.7 / 24.0 | 47.4 / 33.4 | 201.7 / 74.3 | 436.3 / 304.2 | 352.9 / 275.6 | 1,414.9 / 967.8 |
| L2 read / write KB per read | 17.0 / 21.3 | 11.9 / 29.7 | 54.2 / 65.8 | 114.4 / 273.4 | 97.9 / 263.8 | 315.7 / 852.8 |
| DRAM read / write KB per read | 33.1 / 6.4 | 17.4 / 8.7 | 104.1 / 33.1 | 189.4 / 96.0 | 189.7 / 107.4 | 504.3 / 272.7 |

k_p2 (warp per read, 256 threads, 42 registers, 62.5% theoretical occupancy): A 2.21 ms, M 8.65 ms, B 4.62 ms per batch;
15.9-17.4 active threads per instruction, issue slots 41-45% busy, long scoreboard 51-69% (SA reads and the window-bin
lookup). It is the only kernel that keeps half a warp busy, and it is 4.7% of M5.

What the table says:

- **Divergence is near-total.** The thread-per-read kernels run 1.9-2.0 (k_p3) and 3.6-4.4 (k_p1) active threads per
  executed instruction: 6% and 11-14% SIMT efficiency. Branches are 92-97% uniform among the threads still together,
  yet threads drift apart because every loop has a data-dependent trip count (bases to compare, SA entries, windows,
  steps) and nothing reconverges them until the kernel ends.
- **Lane A's k_p3 waits on instruction fetch, not memory.** `no_instruction` is 51% of its stall cycles and the
  instruction cache hits 54% of requests (lane A's k_p1 85%, k_p2 99.9%). Diverged sub-warps walk different parts of one large
  inlined kernel (assignHits + stitching + finalisation + selection + encoding), so the warp's instruction stream keeps
  jumping. Lanes M and B spend proportionally more time inside the same few loops and wait on memory instead
  (long scoreboard 47% and 54%).
- **Nothing is bandwidth-bound.** DRAM runs at 10-24% of peak and L2 at 22-32%; issue slots are 13-24% busy. Every kernel
  is latency-bound with 0.15-0.29 eligible warps per scheduler: on most cycles no warp can issue.
- **Sectors per request look good only because so few threads are active.** 2.2-2.4 sectors per load request is about
  one sector per active thread (1.1-1.2). An AoS layout costs nothing extra when one or two threads issue a load; it would
  cost 32 sectors per request if a warp were converged.
- **Stores dominate the L2 traffic.** k_p3 writes 2.4-2.7x as many L2 sectors as it reads (A 29.7 vs 11.9 KB per read)
  and k_p1 writes as much as it reads; global stores are write-through to L2.
- **Per-read state is re-read and re-written several times.** Lane B's k_p3 loads 1.4 MB and stores 0.97 MB per read
  through L1 against a 635 KB Work; lane M 436 / 304 KB against 83 KB; lane A 47 / 33 KB against 46 KB. Lanes M and B
  miss L2 on that traffic (their batch's Work is 21.6 and 41.6 GB), so they read 189 and 504 KB per read from DRAM.

## 5. Hot source lines and functions (k_p3 lanes A and B)

Warp-stall samples per CUDA line (ncu SourceCounters, `--print-source cuda,sass`), top 10 for k_p3, with the active
threads per executed instruction of the line and its two main stall reasons (`e1_ncu_top_lines.csv`):

| # | lane A k_p3 (job 795282) | share | thr/inst | stalls |
|---|---|---|---|---|
| 1 | stitch.cuh:656 `extendAlign` per-base loop: `G[iG] == 5 ...` (genome byte, read byte) | 7.8% | 1.48 | long sb 50%, wait 26% |
| 2 | stitch.cuh:1191 `Blob::putV` varint loop (output encoding) | 3.9% | 2.78 | no instruction 89% |
| 3 | stitch.cuh:1194 `Blob::put` one byte to global | 3.6% | 2.81 | no instruction 89% |
| 4 | stitch.cuh:157 `Tr::copyFrom` exon rows | 2.8% | 2.26 | no instruction 52%, long sb 37% |
| 5 | stitch.cuh:152 `Tr::copyFrom` gap/indel scalars | 2.7% | 2.74 | long sb 60%, no instruction 31% |
| 6 | stitch.cuh:159 `Tr::copyFrom` junction fields | 2.6% | 2.37 | long sb 56%, no instruction 40% |
| 7 | stitch.cuh:149 `Tr::copyFrom` coordinates | 2.3% | 2.76 | long sb 76% |
| 8 | stitch.cuh:150 `Tr::copyFrom` chr/strand | 1.7% | 2.76 | long sb 66% |
| 9 | stitch.cuh:654 `extendAlign` loop head | 1.7% | 1.56 | wait 42%, no instruction 33% |
| 10 | stitch.cuh:158 `Tr::copyFrom` shiftSJ | 1.6% | 2.42 | long sb 59%, no instruction 38% |

| # | lane B k_p3 (job 795282) | share | thr/inst | stalls |
|---|---|---|---|---|
| 1 | stitch.cuh:656 `extendAlign` per-base loop | 12.3% | 1.95 | long sb 59%, wait 27% |
| 2 | stitch.cuh:159 `Tr::copyFrom` junction fields | 3.7% | 2.43 | long sb 80% |
| 3 | stitch.cuh:157 `Tr::copyFrom` exon rows | 2.8% | 2.22 | long sb 59%, no instruction 20% |
| 4 | stitch.cuh:152 `Tr::copyFrom` gap/indel scalars | 2.7% | 2.97 | long sb 75% |
| 5 | stitch.cuh:749 `stitchAlignToTranscript` junction-shift scoring (read vs two genome positions) | 2.3% | 1.24 | long sb 53%, wait 20% |
| 6 | stitch.cuh:654 `extendAlign` loop head | 2.1% | 2.03 | wait 54%, branch resolving 22% |
| 7 | stitch.cuh:149 `Tr::copyFrom` coordinates | 2.1% | 2.98 | long sb 78% |
| 8 | stitch.cuh:570 `assignAlignToWindow` scan of the window's aligns (Work.WA) | 2.0% | 1.15 | long sb 86% |
| 9 | stitch.cuh:483 `binarySearch2` (annotated junction lookup) | 2.0% | 1.32 | long sb 87% |
| 10 | stitch.cuh:150 `Tr::copyFrom` chr/strand | 1.8% | 2.98 | long sb 87% |

Folded into functions (`e1_lines_by_func.py`, `e1_ncu_by_func.csv`; share of the kernel's stall samples, active threads
per instruction, theoretical L2 sectors of the function's global accesses):

| function (k_p3) | A share | A thr/inst | B share | B thr/inst | what it touches |
|---|---|---|---|---|---|
| `Tr::copyFrom` (transcript copies) | **19.3%** | 2.60 | **21.3%** | 2.74 | Work (stack, trMult, trBest): 51% of A's and 53% of B's k_p3 L2 sectors |
| `extendAlign` | 16.4% | 1.40 | 22.6% | 1.77 | genome and read bytes, one per iteration |
| `stitchAlignToTranscript` | 13.2% | 1.12 | 18.9% | 1.20 | genome, read, transcript in Work |
| `finalizeTranscript` | 12.9% | 2.36 | 10.5% | 3.02 | Work (scores, records) |
| `stitchWindowAligns` (recursion) | 8.6% | 3.02 | 9.7% | 3.56 | Work, local stack |
| `Blob::putV` + `Blob::put` (output encoding) | 9.4% | 2.8 | 0.9% | 1.1 | output buffer, byte by byte |
| `assignAlignToWindow` + `assignEntry` + `assignHits` | 7.8% | 2.2-3.2 | 5.9% | 1.5-2.2 | SA, Work.WA |
| `wbGet` / `binarySearch2` / `mapFinish` body | 3.1 / 1.3 / 3.5% | | 2.1 / 2.5 / 1.9% | | window map, junction DB |

Lane M's k_p3 (job 795297) has the same profile: `Tr::copyFrom` 21.5% (3.25 threads per instruction), `extendAlign` 19.1%,
`stitchAlignToTranscript` 15.0%, `finalizeTranscript` 12.6%, `stitchWindowAligns` 9.8%, assignment 7.5%, `wbGet` 3.4%,
encoding 3.0%. `Tr::copyFrom` holds 51% (A), 53% (M) and 53% (B) of the theoretical L2 sectors of k_p3's sampled lines.

The transcript copies are the largest single function in every lane's k_p3 and half of its L2 traffic. The in-kernel
`copies` clock counter (1.1-1.5% of stitching) times only one of the copy sites and understates them by an order of
magnitude.

## 6. Batch tails (Q3)

Job 795282 (`m5t` at d41bd534; job 795242 with the 43c74812 probe agrees within 2%). For each kernel launch the probe
measures, over the launch's span, the share of resident warp slots (blocks per SM x 4 warps x 132 SMs) that hold an
unfinished warp ("warp-slot busy"). Threads of a warp end together (they reconverge at the kernel's end), so thread-slot
and warp-slot shares coincide; a block slot is busy until the block's last warp ends.

| lane | kernel | span ms | warp-slot busy | idle: finished warps in a live block | idle: grid drain and under-fill | idle slot-time ms |
|---|---|---|---|---|---|---|
| A | k_p1 | 106.2 | 0.766 | 9.0% | 14.4% | 24.9 |
| A | k_p3 | 144.2 | 0.581 | 23.4% | 18.5% | 60.4 |
| M | k_p1 | 78.8 | 0.781 | 5.6% | 16.3% | 17.3 |
| M | k_p3 | 238.2 | 0.694 | 13.3% | 17.3% | 72.9 |
| B | k_p1 | 39.6 | 0.423 | 3.1% | 54.6% | 22.9 |
| B | k_p3 | 210.7 | 0.332 | 5.3% | 61.5% | 140.7 |
| **all** | | 817.7 | | | | **339.1 = 39% of M5 (865 ms in this job)** |

"Idle slot-time ms" is span x (1 - warp-slot busy): the time the kernels would lose if nothing but the empty slots were
removed. Two mechanisms produce it:

- **Block granularity.** A 128-thread block keeps its four warp slots until its slowest warp ends. In lane A's k_p3 this
  alone is 23% of the slot-time; the busy curve of every A batch sits at 0.70-0.80 in mid-batch, not at 1.0 (curves in
  `tail.log`, 20 bins per batch).
- **Batch drain and under-fill.** The last 30-40% of each A and M batch drains to zero; lane B's batch of 65,536 reads
  never fills the 84,480 slots, and its second batch (25,204 reads) fills 30% of them.

The critical path is one warp, and it starts first. In every batch the longest-running warp began at t = 0 (first wave)
and ran for most of the span:

| lane | k_p3 batch spans ms | longest warp ms | longest warp / span |
|---|---|---|---|
| A | 15.6, 17.7, 20.3, 20.8, 19.9, 19.9, 18.6, 10.8 | 13.7, 16.7, 16.6, 21.2, 16.4, 17.5, 15.4, 10.5 | 0.81-1.00 |
| M | 111.3, 111.5, 15.5 | 76.1, 75.2, 15.5 | 0.67, 0.68, 1.00 |
| B | 121.1, 87.2 | 123.3, 87.4 | 1.00, 1.00 |

The slowest 1% of threads hold only 0.9-3.2% of k_p3's thread-time (TAILWHO lines): the tail is not a few reads with huge
total work, it is the serial length of single warps (32 reads that diverge and run largely one after another) and the
slot waste around them. Under a persistent per-lane work queue the floor is max(busy slot-time, longest warp):

| kernel | now ms | floor ms | gain ms |
|---|---|---|---|
| k_p3 A / M / B | 144.2 / 238.2 / 210.7 | 83.8 / 165.3 / 123.3 (B: its longest warp) | 60 / 73 / 87 |
| k_p1 A / M / B | 106.2 / 78.8 / 39.6 | 81.4 / 61.5 / 25.2 (B: its longest warp) | 25 / 17 / 14 |
| total | 817.7 | 540.5 | **277 (32% of M5), an upper bound** |

The floor assumes per-warp times stay as they are with more warps running; they will grow somewhat (the kernels are
latency-bound with 10-23% of DRAM bandwidth in use, section 4). Lane B's floor is its longest warp, so B gains beyond
87 ms only if the slowest reads are split across threads (window parallelism, E3) or spread one per warp.

## 7. Reads in flight, memory and local vs global Work (Q4)

NB sweep, job 795242 (lane times in ms, all variants 100% identical; Work = the per-batch Work buffer):

| lane | reads per batch | lane ms | vs best | Work GB |
|---|---|---|---|---|
| A | 32,768 | 649.4 | +175% | 1.51 |
| A | 65,536 | 417.5 | +77% | 3.02 |
| A | 131,072 | 318.1 | +35% | 6.04 |
| A | 262,144 (default) | 268.6 | +14% | 12.09 |
| A | **524,288** | 245.4 | +3.9% | 24.17 |
| A | 1,048,576 | 236.1 | best | 48.34 |
| M | 32,768 | 632.7 | +95% | 2.71 |
| M | 65,536 | 433.8 | +34% | 5.41 |
| M | 131,072 | 373.5 | +15% | 10.82 |
| M | **262,144** (default) | 337.4 | +4.2% | 21.64 |
| M | 524,288 (all 524,577 reads in 2 batches, the second of 289) | 323.7 | best | 43.28 |
| B | 8,192 | 1016.1 | +414% | 5.20 |
| B | 16,384 | 558.6 | +183% | 10.41 |
| B | 32,768 | 328.0 | +66% | 20.82 |
| B | 65,536 (default) | 254.7 | +29% | 41.63 |
| B | **131,072** (all 90,740 reads in one batch) | 197.6 | best | 57.64 |

Smallest batch within 10% of the best: A 524,288 (24.2 GB), M 262,144 (21.6 GB), B only the single batch (57.6 GB).
Whole M5: default 860.7 ms; (524k, 524k, 131k) 767.2 ms (-10.9%); (1M, 524k, 131k) 758.6 ms (-11.9%, 2.63 M distinct/s).
Occupancy is fixed by registers at every batch size, so the curve measures the batch count (every batch pays a drain)
and, below the resident slot count (84,480-101,376 threads), plain under-fill. The lanes run one after another and free
their buffers, so the peak is the largest lane: 57.6 GB for B in one batch, plus the 17.6 GB human index (study,
section 1.3) = 75 GB on the 95.8 GB H100 NVL. On a 40 GB card about 20 GB remain for Work: A at 262,144 (12.1 GB, +14%),
M at 131,072 (10.8 GB, +15%), B at 32,768 (20.8 GB, +66%); from the sweep's lane times that is about 970 ms, +28% against
the H100's best configuration (not run as one job).

Mono path (`--mono`: one `k_lane` thread per read in a grid-stride loop for lane A; lane M staged; lane B mono with 64
threads), job 795242. Resident blocks per SM: 4 with global Work (104 registers), 5 with local Work (88 registers);
requesting more blocks than are resident only makes the grid-stride chunks smaller.

| Work | blocks per SM requested (resident) | step budget | lane A ms | cycles seeds / windows / stitching / select % | stitching-equivalent ms |
|---|---|---|---|---|---|
| global | 1 (1) | 1024 | 1513.4 | 33.1 / 7.8 / 53.5 / 5.6 | 810 |
| global | 2 (2) | 1024 | 1133.5 | 36.2 / 8.7 / 50.5 / 4.6 | 572 |
| global | 4 (4) | 1024 | 1002.6 | 40.6 / 11.0 / 44.7 / 3.8 | 448 |
| global | 4 (4) | none | 964.3 | 40.6 / 11.0 / 44.6 / 3.8 | 430 |
| global | 8 (4) | 1024 | 879.1 | 48.5 / 13.3 / 35.5 / 2.7 | 312 |
| global | 16 (4) | 1024 | 733.0 | 56.5 / 15.7 / 25.8 / 1.9 | 189 |
| local | 2 (2) | 1024 | 654.4 | 75.7 / 15.8 / 8.0 / 0.5 | 52 |
| local | 4 (4) | 1024 | 626.7 | 71.7 / 18.3 / 9.2 / 0.8 | 58 |
| local | 4 (4) | none | 626.7 | 71.7 / 18.3 / 9.2 / 0.8 | 58 |
| local | 8 (5) | 1024 | 620.8 | 71.7 / 18.4 / 9.1 / 0.8 | 56 |

`--mono --local` vs `--mono` at the default 4 blocks per SM: 626.7 vs 1002.6 ms (1.60x); best vs best 620.8 vs 733.0 ms
(1.18x). The stitching part drops 7.7x (448 -> 58 ms) while the seed part does not move (407 -> 449 ms); ncu below shows
why (convergence and instruction fetch, not memory traffic). With global Work the kernel
also gains 27% from finer grid-stride chunks (1002.6 -> 733.0 ms at the same 4 resident blocks): the tail effect of
section 6 again. With local Work neither residency (2 vs 4 blocks) nor chunk size matters (621-654 ms). The staged lane A
(268.6 ms) stays 2.3x faster than the best mono variant.

ncu on the mono kernel (job 795297; first `k_lane` launch of lane A, all 1,991,926 reads, 4 resident blocks per SM
requested; `results/ncu-795297/`):

| `k_lane` lane A | global Work | local Work |
|---|---|---|
| duration ms | 1010 | 614 (1.64x) |
| registers / resident blocks per SM / occupancy theoretical, achieved % | 104 / 4 / 25, 21.7 | 88 / 5 / 31.3, 23.6 |
| warp instructions executed | 134.8 G | 101.6 G (-25%) |
| active threads per instruction | 1.44 | 1.85 |
| thread instructions executed | 194 G | 188 G (-3%) |
| instruction-cache requests / hit % | 19.5 G / 52.3 | 10.8 G / 58.1 |
| stall mix | no instruction 65%, long scoreboard 16%, wait 10% | long scoreboard 41%, no instruction 39%, wait 11% |
| L2 read / write sectors | 1.33 G / 3.12 G | 3.80 G / 4.56 G |
| DRAM read + write (% of peak) | 66 + 16 GB (2.1%) | 253 + 95 GB (14.4%) |

The local build does the same thread-level work (188 vs 194 G thread instructions) in 25% fewer warp instructions, because
more threads stay together (1.85 vs 1.44 per instruction), and it needs 45% fewer instruction fetches. It moves 4.3x
the DRAM bytes and stalls more on memory, and is still 1.64x faster. The gain of local Work is convergence and
instruction supply, not memory traffic; likely mechanisms (not isolated here) are shorter code for local addressing
(32-bit stack offsets instead of 64-bit address arithmetic for every Work field) and no possible aliasing between the
Work and the genome/SA pointers.

## 8. The seed stage

k_p1 is 223.7 ms = 26.0% of M5 (job 795242). Splitting k_p1 by its thread cycles (seeds vs windows: A 86/14, M 52/48,
B 32/68) gives the seed search 90.9 + 41.2 + 12.3 = **144 ms = 16.8% of M5** and window creation 79 ms (9.2%).

ncu on lane A's k_p1 (job 795282, by function): `compareSeqToGenome` 38.2% of stall samples, SA reads (`packed_get`)
12.1%, `storeAligns` 11.9%, `maxMappableLength(2strands)` 4.9%, `qualitySplit` 1.0%; windows (`createExtendWindowsWithAlign`
8.0%, `wbSet` 6.7%, `wbGet` 4.9%) 19.6%; the rest of `mapPhase1` 10.7%. The kernel is memory-latency bound
(long scoreboard 71% of stall cycles, 3.57 active threads per instruction, 26.6% achieved occupancy, 33 KB of DRAM reads
per read, L2 read hit 23.5%). In lane B, k_p1 is dominated by the window-bin map (`wbSet` 33.6%, `wbGet` 11.5%,
`createExtendWindowsWithAlign` 13.3%; 4,096 open-addressing slots per read).

Rate: lane A seeds 1.99 M reads in ~91 ms = 21.9 M reads/s, against 78-105 M distinct reads/s for M3's standalone exact
seed kernel (study, Part II): inside k_p1 the same search runs 3.6-4.8x slower.

Redundant phase 1: lane M's reads are lane A's overflows and lane B's are lane M's; both lanes run k_p1 again from the
read: 79.2 + 38.8 = 118 ms (13.7% of M5), of which seeds ~53 ms. Lanes A and M have the same seed capacity (NP = 32), and
510,638 of the 524,577 M reads overflowed lane A on windows, i.e. after the seed stage had finished: lane M recomputes
seeds lane A already had.

## 9. Verdicts on the five hypotheses

**(a) "The AoS Work state dominates k_p3": partly true, and not in the way the hypothesis assumed.**
The Work-resident transcript copies are the largest single function of k_p3 in every lane (`Tr::copyFrom` 19-22% of
stall samples) and about half of k_p3's L2 traffic, and k_p3 writes 2.4-2.7x as many L2 sectors as it reads. But the
layout (AoS vs SoA) is not what costs: with 1.9-2.0 active threads per instruction a load already costs about one sector
per active thread (1.1-1.2), and an SoA layout coalesces only threads that execute the same load together. The top stall
of lane A's k_p3 is instruction fetch (51%), not memory (33%); in lanes M and B memory latency leads (47%, 54%) because
their Work does not fit in L2. The local-Work experiment points the same way: moving the whole Work into thread-local
memory made the mono kernel 1.60-1.64x faster (jobs 795242, 795297) while it moved 4.3x more DRAM bytes; the gain came
from 25% fewer warp instructions (1.85 vs 1.44 active threads per instruction) and 45% fewer instruction fetches. Work
traffic is a top-two cost in every lane, the layout is not; what pays is fewer and smaller copies (fewer instructions)
and code that keeps threads together.

**(b) "Batch tails are large": true.** 39% of k_p1 + k_p3 warp-slot time is empty (339 of 865 ms, job 795282): in lane A's
k_p3 23% from finished warps held by unfinished blocks and 19% from the batch drain; in lane B 62% from under-fill and
drain. The longest warp starts at t = 0 and lasts 81-100% of every lane A k_p3 batch, 67-68% of the full M batches and
100% of both B batches. A persistent per-lane queue has a floor 277 ms (32% of M5) below today; bigger batches alone
gave -11.9% (job 795242); finer grid-stride chunks gave -27% on the mono path at constant residency.

**(c) "Occupancy is register-limited": true as a limit, secondary as a cost.** Registers are the occupancy limiter of
every kernel (k_p1 95 -> 5 blocks of 128 per SM, 31.25%; k_p3 A/M 80 -> 6, 37.5%; k_p3 B 94 -> 5, 31.25%; no spills).
The kernels are latency-bound (0.15-0.29 eligible warps per scheduler, issue slots 14-24% busy), so more resident warps
would help, but achieved occupancy is already 10-40% below the theoretical one because of tails (B k_p3 18.6% of 31.25%),
the mono path gained 1.34x from 1 to 2 resident blocks and only 1.13x from 2 to 4, and lane A's leading stall
(instruction fetch) grows with more diverged warps. Expected gain of 80 -> 64 registers (6 -> 8 blocks): 5-15% on the
kernels concerned; worth one cheap run (`__launch_bounds__`, identity check), not a redesign.

**(d) "Lanes M and B take 70% of the time": true (68.8%, 592 of 861 ms, for the 26.3% of distinct reads that overflow
lane A), and the reason is the amount of serial work per read, made worse by memory, not more divergence.** Per k_p3 read
lane M executes 8.3x and lane B 27.5x the warp instructions of lane A (64.7k and 215k vs 7.8k), and the time per read
follows (432 and 1,845 vs 68 ns): about 7-9 ns per thousand warp instructions in all three lanes. Divergence is the same
in all lanes (1.94 / 2.03 / 1.94 active threads per instruction), so it sets the cost per instruction everywhere; what
makes M and B expensive is that one thread runs the whole read (16-128 windows, 1.9-3.1k SA entries, up to 2,048
stitching steps) one step at a time. Their larger state shifts the stall mix from instruction fetch to memory latency
(DRAM per read in k_p3: A 26 KB, M 285 KB, B 777 KB), and lane B adds under-fill and a single-warp critical path
(warp-slot busy 0.33). The lever for M and B is parallelism inside a read (windows and SA entries across the lanes of a
warp, E3), not memory layout alone.

**(e) "The seed stage is small": false.** Seeds cost ~144 ms = 16.8% of M5 and window creation another 79 ms (k_p1 in
total 223.7 ms = 26.0%), above the study's 15% threshold for the 2-bit genome + SA bucket work. The seed search inside
k_p1 runs at 21.9 M reads/s against 78-105 M/s for M3's standalone kernel, and lanes M and B redo all of phase 1
(118 ms, 13.7% of M5) for reads whose seeds lane A had already computed.

## 10. Ranked bottlenecks and wave-2 recommendation

Bottlenecks ranked by the time they hold (each with the evidence above):

| rank | bottleneck | time at stake | evidence |
|---|---|---|---|
| 1 | SIMT divergence of thread-per-read code: 1.9-2.0 (k_p3) and 3.6-4.4 (k_p1) of 32 threads active per instruction | the cost per instruction of 95% of M5 (818 ms in k_p1 + k_p3) | ncu, all lanes; instruction-fetch stalls 51% in A k_p3; issue 14-24% |
| 2 | Batch tails and slot waste | 339 ms of empty slot-time; floor -277 ms (32%) | tail probe; NB sweep -11.9%; mono chunking -27% |
| 3 | Serial per-read work in lanes M and B (one thread walks 16-128 windows) | 592 ms (69%) of M5 | 8.3x / 27.5x A's instructions per read; single-warp critical path in B |
| 4 | Transcript copies through global Work | ~20% of k_p3 samples ~ 120 ms; half of k_p3's L2 traffic | ncu by function, all lanes |
| 5 | Seed stage inside k_p1 and its repetition in lanes M and B | 144 ms seeds + 118 ms redone phase 1 (of which ~53 ms seeds) | clk shares; ncu k_p1; 22 vs 78-105 M/s |
| 6 | Register-limited occupancy | 5-15% of the kernels concerned (estimate) | ptxas, ncu occupancy, mono residency sweep |
| 7 | Byte-wise output encoding (`Blob::putV/put`) | 9.4% of A k_p3 samples ~ 14 ms | ncu lines, 89% instruction-fetch stalls |
| - | not bottlenecks: host gaps, thrust scan, syncs, memsets, k_copy_status | 1.3 ms (0.15%) | nsys |

Wave-2 recommendation, ranked by expected gain per unit of work (gains relative to today's 860.7 ms, not additive):

1. **Batch sizes, no code change.** Lane A 524,288-1,048,576, lane M >= 524,577 (all its reads in one batch; 524,288 left
   a 289-read batch that costs 21 ms), lane B all reads in one batch (57.6 GB of Work + 17.6 GB index on the H100).
   Measured -11.9% (758.6 ms, 2.63 M distinct/s); with M in one batch about -14%. On 40 GB cards see section 7.
2. **Persistent per-lane work queue with warp-granular slot reuse and heavy reads first (E4).** Removes the block
   granularity (23% of A k_p3's slot-time; 32-thread blocks alone would remove it, since at 80 registers 25 one-warp
   blocks fit per SM) and the batch drains. Floor -32%; expected -10 to -18% on top of (1), less on lane B, whose floor
   is its slowest warp.
3. **Transcript copies (E5) plus k_p3-only scratch in thread-local memory.** Copy by reference where STAR's copy is not
   observable, 32-bit fields, and keep the recursion's transcripts (stack, trMult, trA) out of the global Work, which
   only phase 1 -> phase 3 state needs. Copies are ~20% of k_p3's stall samples; the mono local-Work build was 1.6x
   faster through fewer warp instructions and instruction fetches, so the k_p3 version must be measured, not assumed.
   Expected k_p3 -10 to -30% (-60 to -180 ms).
4. **Window and SA-entry parallelism inside a read for lanes M and B (E3).** Attacks ranks 1 and 3 where they cost most
   (69% of M5, lane B's single-warp critical path). Highest ceiling (x2-3 on M/B k_p3, -25 to -40% of M5), highest effort.
5. **Hand lane A's phase-1 state to lane M instead of recomputing it**, then pre-classify reads after the seed stage
   (SOAP3-dp style) so no read runs phase 1 twice: -40 to -55 ms (-5 to -6%). Then the seed search: port M3's kernel /
   2-bit genome compare inside k_p1: -50 to -70 ms (-6 to -8%).
6. **Register caps** (k_p3 A/M at 64, k_p1 at 80) with an identity run: -3 to -8%, an afternoon.
7. **Word-wise output encoding** (build the blob in registers or local memory, store 16-byte words): -1 to -2%.

Items 1-3 and 5-7 together plausibly take M5 from 861 ms to ~500-550 ms (3.6-4.0 M distinct reads/s). The 10 M/s target
needs the divergence fixed (item 4, then the wavefront design of Part I section 5): with a perfect queue and no copies,
k_p3 would still execute one thread in sixteen.

## 11. Surprises

1. Divergence is near-total: 1.94 active threads per instruction in k_p3 (6% SIMT efficiency) in all three lanes, 3.6-4.4
   in k_p1. The AoS-vs-SoA question is moot until warps converge.
2. Lane A's k_p3 is bound by instruction fetch, not memory (`no_instruction` 51% of stall cycles, instruction cache hit 54%);
   the byte-wise output encoder alone is 9.4% of its samples, nearly all instruction-fetch stalls.
3. Transcript copies are the largest function of k_p3 (19-22% of samples, half of its L2 traffic), while the in-kernel
   `copies` clock counter reported 1.1-1.5%.
4. Each batch's time is set by one warp that starts at t = 0 (81-100% of lane A's k_p3 batches, 100% in lane B), and
   finished warps waiting for their block's slowest warp cost as much as the batch drain in lane A (23% vs 19%).
5. The seed search inside k_p1 runs 3.6-4.8x slower than M3's standalone kernel, and lanes M and B recompute it (118 ms of
   phase 1 redone).
6. Stores, not loads, dominate L2 traffic: k_p3 writes 2.4-2.7x the sectors it reads.
7. Thread-local Work wins while moving more memory: the mono local build is 1.64x faster with 4.3x the DRAM traffic,
   because it executes 25% fewer warp instructions for the same thread work and halves instruction fetches.
8. Lane M's third batch holds 289 reads and costs 21 ms (2.4% of M5).
9. Tooling: the image's nsys cannot import its own report (QdstrmImporter needs `libdw.so.1`; imported on the host, job
   795281), and ncu 2026.2 prints per-line metrics only with `--print-source cuda,sass` (plain `cuda` prints source only).

## Appendix: jobs and files

| job | what | node |
|---|---|---|
| 795229 | build `m5` and `m5t` at 43c74812, ptxas register table | c008 |
| 795274 | build `m5t` at d41bd534 (TAILEV/TAILWHO/TAILTOP probes) | c009 |
| 795242 | baseline, nsys attempt, tail (43c74812), NB sweep (5 points), mono global/local x blocks per SM x budget | g050 |
| 795275 | nsys timeline (`.qdstrm`), identity | g052 |
| 795281 | host-side QdstrmImporter + `nsys stats` (CPU) | c009 |
| 795282 | tail (d41bd534); ncu full set lane A and lane B | g052 |
| 795297 | ncu full set lane M; ncu mono `k_lane` global and local Work | g051 |
| 795292, 795296, 795300, 795309 | ncu source-view probe and exports, summaries (CPU) | c008/c009 |

Summaries in the repository: `bench/fable/profile/results/time-795242/` (sweep, mono, tails, logs),
`results/nsys-795275/` (kernel sums, per-batch trace summary), `results/ncu-795282/` and `results/ncu-795297/`
(`e1_ncu_kernels.csv`, `e1_ncu_top_lines.csv`, `e1_ncu_by_func.csv`, `summary_ncu.txt`, tail log),
`results/build-43c74812/` (ptxas output, registers). Raw `.ncu-rep`, `.nsys-rep` and the full per-line CSVs stay on
Clipper under `$W/jobs/fable/profile/results/`.
