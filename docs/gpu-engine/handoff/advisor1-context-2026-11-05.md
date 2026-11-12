# Advisor context: the prefill kernels of the singlet GPU engine

Prepared 2026-11-05 by the gpu/bold worker. Source repo Singlet-Bio/singlet, branch `gpu/bold`, code at commit `c3a6096c1ae8b9cb0eaf950968c7ee7ccce8283d` (same files as `e42c0636`, binary `s3`). Everything marked **measured** has a job id. Everything marked **inference** has not been tested.

## 1. The question

The node processes about 2 M read pairs per second per GPU. The device runs kernels for a small share of the time and fills few of its warp slots when it does. We ask for an outside review of the heaviest kernel and the one or two kernels behind it. The review should say why occupancy is low and which exact-preserving changes would raise device throughput.

## 2. Hardware and software

| Item | Value |
|---|---|
| GPU | NVIDIA H100 NVL, 132 SMs, 93.1 GiB visible, max SM clock 1785 MHz, memory clock 2619 MHz, PCIe gen 5 |
| Driver | 610.43.02 |
| Build | CUDA 12.4 (nvcc 12.4.r12.4, `-DCMAKE_CUDA_ARCHITECTURES=90`, Release), GCC 16.2 conda toolchain for the host |
| Profilers | Nsight Compute 2026.2.1, Nsight Systems 2026.1.3 (CUDA 13.3 toolkit on the node) |
| Node | g050, 56 cores, 768 GB. Jobs use 28 cores and one GPU. |
| Process model | One singlet process per sample. The GPU prefill runs in a forked STAR child. Several processes share one GPU by time slicing (no MPS). |

## 3. What the kernels compute, in STAR terms

The engine is a line-by-line CUDA port of STAR 2.7.11b's read mapping. File `bench/gpu/m5/stitch.cuh` holds the ported STAR code (`mapPhase1`, `assignHits`, `mapFinish`). File `src/engine_gpu/engine_gpu.cu` holds the kernels and the launch code (`runLane`, lines 540 to 680).

Each distinct read sequence goes through three phases. One CUDA thread (or one warp in lane M) owns one read for its whole life.

| Phase | Kernel | STAR code it ports |
|---|---|---|
| 1 seeds | `k_p1s` (persistent warps, seed iterator in shared memory), then `k_p1w` or, in lane B, `k_p1wp` | `ReadAlign_mapOneRead` seed search (`maxMappableLength2strands`), `storeAligns`, `createExtendWindowsWithAlign` |
| 2 hit marking | `k_p2` (warp per read) | SA entry filter (`entryHit`) over the seed's suffix-array range |
| 3 stitching | **`k_p3e`** lane B, **`k_p3lb`** lane A, **`k_p3a` then `k_p3w`** lane M | `assignAlignToWindow`, `stitchPieces`, `stitchWindowAligns` (the include/exclude recursion over the aligns of a window), `extendAlign`, `scoreGenomicLength`, `multMapSelect`, the mapped filters |

Reads are sorted into three lanes by cost. The lane limits are the template parameters `Caps<LREAD, NSPLIT, NP, NW, NWA, NWB, NEX, NWTR, NKEEP, NBLOB>`:

| Lane | Reads (pbmc1k_v3) | Caps | Phase 3 kernel | Work struct per read in flight |
|---|---|---|---|---|
| A, light | every one of the 31.7 M distinct reads enters A first. 27.9 M finish there. | NP 32, NW 16, NWA 16, NWB 256, NEX 8, NWTR 8, NKEEP 8 | `k_p3lb`, thread per read, 64 registers, `__launch_bounds__(128,8)` | 36 KB |
| M, medium | 3.85 M (12%), the reads A could not finish | NP 32, NW 64, NWA 8, NWB 1024, NEX 8, NWTR 8, NKEEP 32 | `k_p3a` then `k_p3w`, warp per read, 98 registers, 4,050 MiB of device stack reserved for the lane | 68 KB |
| B, heavy | 705,895 (2.2%), the reads M could not finish | NP 128, NW 128, NWA 50, NWB 4096, NEX 20, NWTR 32, NKEEP 64. These are STAR's own limits. | **`k_p3e`**, thread per read, 80 registers | **550 KB** |

Lane B is the reads with many windows and many aligns (repeats, multimappers). It is 2.2% of the distinct reads and **39% of the prefill kernel time**.

## 4. Device time per kernel (measured, job 800512, run 1)

One solo pbmc1k_v3 run under Nsight Compute at a 45 GB device budget (the budget of a 2-slot node). 958 launches of 11.31 s summed device time. See `nsys.txt` section 2 for the table. Top five:

| Rank | Kernel | Device s | Share | Launches | Mean ms | Grid | Block | Regs | Occupancy limit from registers (blocks/SM) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `k_p3e`, lane B | 4.45 | 39.3% | 46 | 96.7 | 231 | 128 | 80 | 6 |
| 2 | `k_p3lb`, lane A | 1.51 | 13.4% | 80 | 18.9 | 3090 | 128 | 64 | 8 |
| 3 | `k_p3w`, lane M | 1.45 | 12.8% | 42 | 34.5 | 42399 | 128 | 98 | 4 |
| 4 | `k_p1s`, lane A | 0.92 | 8.1% | 80 | 11.4 | 987 | 128 | 64 | 8 |
| 5 | `k_p1w`, lane M | 0.66 | 5.8% | 42 | 15.8 | 1325 | 128 | 63 | 8 |

**Why the lane B grid is small (measured).** A lane B batch is as many reads as fit in the device budget at 550 KB of Work per read. At a 45 GB budget that is 29.6 thousand reads, so 231 blocks of 128 threads on 132 SMs. With 6 blocks per SM allowed by registers, the device could hold 792 blocks. The launch fills 29% of one wave. At a 91 GB budget (one process alone) the batch is 115,712 reads, about 904 blocks. At the 24 GB budget used in an earlier ncu job (800386) the lane B grid was only 78 blocks. **Memory per read in flight sets the grid, and the grid sets occupancy.**

**Lane B in a k=8 load (measured, job 800350, run c12).** Lane B kernels took 10.4 s against 2.7 s alone. Lane A and M took 5.9 s and 5.8 s against 2.9 s and 2.8 s. Two processes time-slice the GPU, and the large Work buffers of the other process also limit residency.

## 5. Hold split (measured)

The device lock covers the prefill step called `mapDistinct`. Numbers in seconds, from the engine's own step timers (`nsys.txt` section 1).

| Class | Solo, 91 GB budget (job 800349) | k=8 load, 45 GB budget, run c12 of job 800350 | Baseline run c10 of job 800259 |
|---|---|---|---|
| Lock held | 18.4 | 34.4 (waited 27 s first) | 43.7 |
| Lane kernels A, M, B | 8.3 (2.85, 2.76, 2.70) | 22.1 (5.95, 5.77, 10.36) | 14.4 (4.10, 3.95, 6.37) |
| H2D copies | 2.5 | 2.0 | 8.7 |
| D2H copies | 0.8 | 1.1 | 8.0 (staged) |
| Host work under the lock | 3.8 | 4.9 | 8.7 |
| Records on the device | 1.7 | 1.7 | 2.4 |

The prefill is about 45% kernels when alone and 64% in a k=8 load. In a 130 s solo sample the device has any kernel activity for about 9 s (GR active above 50% for 9 s, `nsys.txt` section 3).

## 6. GPU metrics (measured, nsys GPU metrics sampler and nvidia-smi GPM, device wide)

| Window | SMs active | Warps in flight (occupancy) | Unallocated warps in active SMs | SM issue | DRAM read | DRAM write | PCIe rx, tx |
|---|---|---|---|---|---|---|---|
| Solo run, seconds with GR active above 50% (9 s of 136 s) | 71.5% | 24.5% | 47.0% | 17.2% | 11.2% | 4.9% | 0.7%, 1.4% |
| k=8 load, seconds with GR active above 50% (110 s of 288 s) | 62.2% | 17.0% | 45.4% | 12.4% | 7.5% | 4.4% | 2.8%, 2.1% |
| k=8 load, mean over all seconds | 25.4% | 7.5% | 18.0% | 5.1% | 3.3% | 2.0% | 3.3%, 1.9% |
| nvidia-smi dmon GPM, k=8 to k=12 points of job 800347, span means | 24 to 38% (SM activity) | 7 to 10% (SM occupancy) | | | 4 to 6% (DRAM activity) | | rx about 1.8 GB/s, tx about 0.7 GB/s |

`nvidia-smi` "SM utilization" reads 47 to 73% in the same runs. It only says that a kernel ran.

**Reading.** While a kernel runs, the SMs are active but only about a quarter of the warp slots hold warps and DRAM is at 5 to 11%. Neither compute nor memory bandwidth is the limit. The kernels are limited by registers per thread (64 to 98), by small grids (lane B), by divergence inside a thread-per-read recursion, and by the dependent loads of the stitching recursion.

## 7. Divergence and per-kernel counters from the small ncu run (measured, job 800386, 24 GB budget, pbmc2m)

Warp execution efficiency is `smsp__thread_inst_executed_per_inst_executed.ratio` divided by 32. These come from a run with small grids, so use them for the thread behavior and not for occupancy.

| Kernel | Active threads per warp instruction | Warps active, % of peak | Branch targets uniform, % | SM throughput % | DRAM throughput % |
|---|---|---|---|---|---|
| `k_p3e` lane B (grid 78) | 1.85 | 5.1 | 97.0 | 3.1 | 0.5 |
| `k_p3lb` lane A | 2.08 | 29.0 | 97.8 | 10.5 | 8.7 |
| `k_p1s` lane A | 3.2 | 40.5 | 91.2 | 23.0 | 19.0 |
| `k_p1wp` lane B (persistent) | 10.6 | 5.8 | 90.7 | 2.8 | 7.0 |
| `k_p3e` lane M variant (lean) | 2.01 | 17.8 | 96.6 | 12.0 | 8.6 |

In the thread-per-read stitching kernels only about 2 of 32 lanes do useful work per warp instruction, although branches are 97% uniform. That means the 32 threads of a warp execute different reads, and the reads take very different code paths (different loop trip counts and recursion depths), so most lanes idle on most instructions. The full `--set full` details of the top kernels are in `ncu.txt` once job 800527 has run.

## 7b. Nsight Compute --set full of the top three kernels (measured, job 800533; `ncu.txt`, `ncu_source.txt`)

One solo run at a 45 GB budget, two launches per kernel from the middle of the run.

| | `k_p3e` lane B | `k_p3lb` lane A | `k_p3w` lane M |
|---|---|---|---|
| Duration, launch 1 (launch 2 for k_p3e: 104 ms) | 97.7 ms | 17.4 ms | 37.9 ms |
| Grid (launch 1, launch 2), block | 248 and 248, 128 | 3,392 and 384 (0.36 waves for the second), 128 | 45,056 and 21,641, 128 |
| Waves of the device | 0.31 (launch 1) | 3.2 and 0.36 | many |
| Registers per thread | 80 | 64 | 98 |
| Theoretical occupancy and its limit | 37.5% (24 of 64 warps, registers, 6 blocks per SM) | 50% (registers, 8 blocks) | 25% (registers, 4 blocks) |
| Achieved occupancy | 8.7% (5.6 warps per SM) | 35.3% (22.6 warps) | 17.7% (11.3 warps) |
| Active threads per warp instruction | 1.87 | 2.3 | 3.1 |
| Issue slots busy (SM throughput) | 6.1% | 14.0% | 16.3% |
| Eligible warps per scheduler | 0.14 of 1.69 active | 0.22 of 5.98 | 0.19 of 2.89 |
| Warp cycles per issued instruction | 12.8 | 30.9 | 16.5 |
| Top stall | L1TEX scoreboard (loads) 55.7% of cycles | waiting to be selected or instruction cache miss | L1TEX scoreboard |
| DRAM throughput | 2.8% | 12.8% | 15.6% |
| L1 hit rate | 84% | 77% | 61% |
| Local memory (stack and spills) | 5.8% of L1 sectors, 90 M L2 sectors | | |

What the SASS source page shows for `k_p3e` (10,792 SASS lines, 6.43 M stall samples): 52% of stall samples are long scoreboard (a load result not back yet), 28% are fixed-latency wait, 9% selected. The hottest lines are integer compares and adds that run with one active thread per warp (`avg threads` 1.0 to 2.0) right after loads of the per-read Work struct: for example `ISETP.NE ... 0x5` and `ISETP.NE ... 0xb` (tests of a status or op code) executed 8 M times each at 1.0 threads, and the compare and add pairs at the lines near `f4c5f0` to `f4cc80` executed 18.7 M times at 1.0 threads. The thread instruction count is 10.6 G over 5.66 G warp instructions. Divergent branches: 21.6 M.

**Reading.** All three phase 3 kernels are limited by the latency of dependent loads from the per-read Work struct (global and local memory), issued by few lanes of a warp at a time, with few warps resident (registers cap residency at 4 to 8 blocks per SM) and, in lane B, too few blocks to cover the device. Memory bandwidth, compute and the instruction cache are not saturated.

## 8. Device memory per slot (measured, solo run 800349, E17 line)

| Piece | GB | Same in every process |
|---|---|---|
| Genome index and tables | 16.4 | yes (read only) |
| CUDA context | 0.5 | no |
| Reads, packed | 6.1 | no |
| CR4 clip | 7.3 | no |
| E10 aux, entries and masks | 4.5 | no |
| E11 keep | 1.0 | no |
| Lane Work buffers | A 19.1, M 26.9 (incl. 4 GiB stack), B 63.6 at the largest batches | no, they fill the budget |
| Pileup arena | 26.3 peak | no |
| Non-host index (24.9) | outside the budget | yes (read only) |

E17 plan for one sample: 34.8 GB normal mode, 18.5 GB lean mode. At 2 slots the budget is 45.5 GB. At 3 slots (30.3 GB) every sample drops to lean mode.

## 9. The constraint: exact results

The GPU engine must give output byte-identical to the CPU engine (STAR 2.7.11b plus the singlet pipeline). The gate compares 65 output files plus a run-dependent set on 10 samples (gate 798027: all identical). The mapping port matched STAR on 1,991,926 of 1,991,926 sampled distinct sequences and on a whole sample with zero mismatches. Therefore a change to a kernel must keep every alignment, every score, every tie-break and the order of multimapper selection exactly as STAR does. Integer arithmetic only, no floating-point reordering in these kernels. Changes that only reorder which read a thread works on (sorting, batching, work stealing) are fine because reads are independent. Changes to the arithmetic of the stitching recursion are not.

Past attempts (E3 warp-window, E4 order with 81 variants, E7 seeds with 55 variants, `docs/gpu-engine/fable/`) already tried kernel parallelism, batch order and compact state. A persistent-thread stitch and launch-bound changes gave no gain on `k_p3` (`docs/research-beta/gpu-profile-deep.md` section 10.1).

## 10. Questions for the reviewer

1. Lane B (`k_p3e`) carries 39% of kernel time with a grid that fills under one wave and about 2 active lanes per warp. What would you change that keeps results exact? Options we see: a warp or a group of lanes per read for the heavy tail, a smaller Work footprint per read so more reads fit in flight, sorting reads by predicted cost inside the lane, splitting the include/exclude recursion across lanes.
2. Is 80 registers per thread with a 6-block occupancy limit the right target, or would a lower cap (for example 64 as in lane A) win through more resident warps despite spills?
3. The Work struct is 550 KB per read and lives in global memory. Which parts are hot, and could they live in shared memory or registers for most reads?
4. Lane M's `k_p3w` uses a warp per read with 98 registers, and the driver reserves 4,050 MiB of device stack for it. Does the stack reservation hurt occupancy or residency?
5. Four to eight processes share each GPU by time slicing. Would CUDA MPS, or one process launching batches of several samples in separate streams, fill the device better than any single-kernel change?

## 11. Files in this folder

| File | Content |
|---|---|
| `kernel/` | `src/engine_gpu/engine_gpu.cu` (kernels `k_p1s`, `k_p1wp`, `k_p3e`, `k_p3lb`, `k_p3w` and the launch code), `bench/gpu/m5/stitch.cuh` (the ported STAR code: `Work`, `Caps`, `mapPhase1`, `assignHits`, `mapFinish`), `include/singlet/engine_gpu.h`. Commit in `kernel/COMMIT.txt`. |
| `ncu.txt` | Nsight Compute `--set full` details of the top kernels, from job 800527 |
| `ncu_source.txt` | SASS level hot lines per kernel (no CUDA line info, the binary is built without `-lineinfo`) |
| `nsys.txt` | The prefill hold timeline from the engine timers, the per-kernel table, the device wide GPU metrics, the parent process nsys summary |
| `context.md` | This file |
