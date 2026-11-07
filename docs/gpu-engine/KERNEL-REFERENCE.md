# GPU engine kernel reference

This page describes the device kernels of `singlet --engine gpu`. It covers the mapping kernels of `src/engine_gpu/engine_gpu.cu`, the CellRanger4 clip and the pileup collapse. Every fact comes from the source files or from the measured record named in the Source column. A cell that says "Not recorded" has no measured value yet. Fill it from an `ncu` run (recipe in `RUNTIME-REFERENCE.md`).

The operator view is in `docs/gpu-engine/fable/GPU-PATH.md`. The run-time model (lock, slots, switches, build) is in `docs/gpu-engine/RUNTIME-REFERENCE.md`.

**Status of this page.** It is the first draft of the production-kernel documentation. The mapping kernels, `k_cr4` and the R9 collapse are covered. The records, Solo, non-host and the other pileup stages are listed at the end and still need an entry.

## 1. How a read moves through the lanes

Every distinct cDNA sequence enters lane A. A read that overflows a capacity or a step budget gets `Status::Overflow` and moves to the next lane. Lane B holds STAR's own limits for windows and anchors. A read that overflows lane B goes to the host lane, which runs STAR's `mapRead` on CPU threads.

| Lane | Type in source | Windows (NW) | Aligns per window (NWA) | Exons (NEX) | Kept transcripts (NKEEP) | Result slot (NBLOB) | Phase 1 | Phase 3 |
|---|---|---|---|---|---|---|---|---|
| A, light | `LaneA` | 16 | 16 | 8 | 8 | 640 B | `k_p1s`, `k_p1w` | `k_p3lb`, thread per read |
| M, medium | `LaneM` | 64 | 8 | 8 | 32 | 2048 B | `k_p1s`, `k_p1w` | `k_p3a` then `k_p3w`, warp per read |
| B, heavy | `LaneB` | 128 | 50 | 20 | 64 | 4096 B | `k_p1s`, `k_p1wp` | `k_p3e`, thread per read |
| B, WSHRINK | `LaneBc` | 128 | 32 | 20 | 32 | 4096 B | `k_p1s`, `k_p1wp` | `k_wplan`, `k_wkey`, `k_p3c` rounds |
| Host | `Heavy` | 650 | 50 | 1000 | 10000 | not on device | CPU `mapRead` | |

The columns NW, NWA, NEX, NKEEP and NBLOB are the template arguments in `engine_gpu.cu` lines 51 to 58. NW, NWA, NEX and NKEEP are the first four capacities. `LaneBc` is the lane used when `SINGLET_BOLD_WSHRINK=1`.

The common code lives in `bench/gpu/m5/stitch.cuh`. Each device function there is a line-by-line port of the STAR 2.7.11b source vendored in `src/star`. The header names the STAR file above each function.

## 2. Mapping kernels

### 2.1 `k_p1s`: seed search

| Item | Value |
|---|---|
| STAR function | `ReadAlign::mapOneRead` seed loop, `maxMappableLength2strands`, `storeAligns` |
| Inputs | packed read codes `rb` and offsets `roff`, read index list `idx`, `Work<C>` and `ReadBuf` per read in flight, genome view `g` |
| Outputs | seed hits in `Work<C>`, per read status, per read entry counts for phase 2 |
| Memory layout | `Work<C>` array of one batch (`NB` reads). One `SeedIt` per thread in shared memory |
| Launch shape | 128 threads per block. Grid is `min(nSm * bpsS, G)` with `G = ceil(nb / 128)`. The warps are persistent |
| Registers and bounds | `__launch_bounds__(128, M5_P1MINB)` with `M5_P1MINB = 8`. Register count Not recorded |
| Shared memory | 128 `SeedIt` structs per block (`smIt[128]`) |
| Occupancy | Not recorded. `ncu` job 800512 lists `k_p1s` at 0.92 s of 11.31 s device time, 8.1 percent |
| Work balance | Each warp refills lanes without a read from a global counter `dCtr` with a warp-aggregated `atomicAdd` |
| Limits | Reads up to 160 bases in lanes A, M and B (`LREAD`). Reads over 128 bases decline level 5 on this branch |
| Failure modes | A capacity overflow sets `Status::Overflow`. The read moves to the next lane |
| Tie order | A read's seeds are stored in STAR's order. The batch order does not change a read's result |

### 2.2 `k_p1w`, `k_p1wp`: windows

| Item | Value |
|---|---|
| STAR function | `stitchPieces`: `createExtendWindowsWithAlign`, flanks and `assignAlignToWindow` |
| Kernel choice | `k_p1w` is thread per read, `ceil(nb / 128)` blocks. `k_p1wp` is persistent and runs in lane B only (`NW >= 128`), grid `min(nSm * bpsW, G)` |
| Inputs and outputs | `Work<C>`, per read status and per read entry counts `dNent` |
| Registers and occupancy | Not recorded. `k_p1w` is 5.8 percent of device time in `ncu` job 800512 |
| Failure modes | Overflow of the window, align or `winBin` capacity (`OV_WIN`, `OV_WA`, `OV_WINBIN`) |

### 2.3 `k_p2`: suffix array entry scan

| Item | Value |
|---|---|
| Role | Scans the suffix array entries between phase 1 and phase 3 and writes a bit set (`dBits`) |
| Inputs | `Work<C>`, per read entry count and exclusive scan base |
| Outputs | bit set `dBits`. The array grows when `words` exceeds `bitsCap` |
| Launch shape | 256 threads per block, `ceil(nb * 32 / 256)` blocks |
| Memory | The bit set has `(lastBase + lastCount) / 32 + 2` words. It starts at 1 M words. The allocation `dmal(&dBits, bitsCap * 4)` is the one that failed in the device out-of-memory fallback of calibration job 800365 |

### 2.4 `k_p3lb`: lane A phase 3

| Item | Value |
|---|---|
| STAR function | `stitchWindowAligns`, `stitchAlignToTranscript`, `extendAlign`, `blocksOverlap`, `multMapSelect`, `mappedFilter` |
| Launch shape | thread per read, 128 threads per block, `ceil(nb / 128)` blocks, `k_p3lb<C, 128, 8>` |
| Registers | 64. `__launch_bounds__(128, 8)` allows at most 65536 / (128 x 8) registers per thread |
| Occupancy | achieved 35.3 percent (`ncu --set full`, job 800533). 13.4 percent of device time |
| Shared memory | None |
| Per read work bytes | `Work<LaneA>` plus `ReadBuf` plus 17 bytes of batch bookkeeping. The record quotes 36 KB |
| Memory formula | `laneNeed(m, nb, slot, per) = m * slot + max(min(nb, m) * (per + 256), m * 112) + 16 MiB`. Here `slot = 4 + NBLOB + 1 + 2` bytes per read |
| Tie order | The recursion of `stitchWindowAligns` runs on an explicit stack with STAR's visiting order. Transcript slots keep STAR's permutations |
| Failure modes | `Status::Overflow` (capacity or step budget of 1024), `Status::BlobFull` (result larger than the slot) |

### 2.5 `k_p3a` and `k_p3w`: lane M phase 3

| Item | Value |
|---|---|
| Kernels | `k_cost` builds a key per read from its window count. `thrust::sort_by_key` orders reads by windows, most first. `k_p3a` is thread per read and runs `assignHits`. `k_p3w` is warp per read and stitches the read's windows across the 32 lanes (`warpStitch`). Lane 0 selects and encodes (`warpSelect`) |
| Launch shape | `k_p3a`: 128 threads per block. `k_p3w`: `ceil(nb * 32 / wt)` blocks of `wt` threads, `wt = g_wthreads = 128` |
| Registers | 98 |
| Shared memory | `wt * sizeof(AssignEnt) + (wt / 32) * NW * 2` bytes per block. The `AssignEnt` array is unused here and kept for the shared layout |
| Occupancy | achieved 17.7 percent, 12.8 percent of device time (`ncu` jobs 800512 and 800533) |
| Per read work bytes | `Work<LaneM>` plus `ReadBuf` plus 17 plus 36 bytes for the sort key and order. The record quotes 68 KB |
| Stack | The warp lane keeps its stitching context in the thread stack. The prefill raises the stack limit to `max(16 KiB, sizeof(LWM) + 8192)` and counts the reserve (`stackNeed`) against the device budget |
| Fallback | If the stack and a batch of 32768 reads do not fit next to all slots, lane M runs thread per read (`k_p3e`-style code, `perMt`). The log line says "lane M warp per read" or not |
| Tie order | The sort changes only the launch order. Each read's result does not depend on it |
| Failure modes | `WR_OVERFLOW` gives `Status::Overflow`. `WR_FALLBACK` finishes the read with the thread code (`mapFinish`) |

### 2.6 `k_p3e`: lane B phase 3

| Item | Value |
|---|---|
| Launch shape | thread per read, 128 threads per block. `ncu` shows grid 231 and 6 blocks per SM in job 800512 |
| Registers | 80 |
| Occupancy | achieved 8.7 percent, 1.87 active threads per warp instruction, issue slots 6.1 percent (job 800533) |
| Device time | 39.3 percent of the prefill's kernel time, 96.7 ms mean per launch, 46 launches (job 800512) |
| Per read work bytes | `Work<LaneB>` plus `ReadBuf`. The record quotes 550 KB. This is the largest allocation of the prefill |
| Reads in lane B | 2.2 percent of distinct reads on pbmc1k_v3 |
| Limits | STAR's limits for windows (128) and anchors, step budget 2048 |
| Failure modes | Overflow sends the read to the host lane. The log line `[B to host n of m]` counts them |

### 2.7 `k_wplan`, `k_wkey`, `k_p3c`: lane B with WSHRINK

Switch: `SINGLET_BOLD_WSHRINK=1`. The default is off. Default-on waits for the PR-2 gates.

| Item | Value |
|---|---|
| Idea | The windows' align rows move out of `Work<LaneB>` into a pool sized per batch. `k_wplan` computes the rows each read needs. `k_wkey` lists the pending reads and their needs. An exclusive scan gives each pending read its row start. `k_p3c` runs phase 3 for the reads whose rows fit the pool. The others wait for the next round |
| Why it is exact | A read's result does not depend on where its rows sit, so the order of the rounds changes nothing |
| Pool size | `max(NB * g_waSlice, NW * NWA)` aligns. `g_waSlice` is 160 by default (`SINGLET_BOLD_WASLICE`, at least 8) |
| Per read work bytes | The record quotes 549 KB before and 157 KB after, from the bold branch. A read that needs more than 32 aligns per window or 32 kept transcripts ends `Overflow` and goes to the host lane |
| Registers and occupancy | Not recorded for `k_p3c`. The code reads them at run time (`cudaFuncGetAttributes`) and logs them |
| Log lines | `[wshrink: per read N bytes = Work + ReadBuf + pool + 25, batch NB, pool P aligns (G GB)]` and `[wshrink: R pool rounds, D read-rounds deferred]` |
| Measured effect | `k_p3e` time 4.44 s to 2.23 s on the bold branch. Lock held fell by about 4 percent, which is inside the noise. The value is memory headroom |

### 2.8 `k_cr4`: CellRanger4 clip

| Item | Value |
|---|---|
| STAR function | `ClipCR4` with opal overlap alignment of the 10x TSO against the first 91 bases, then `ClipMate::clip` for the 5' mate (type 10) and the 3' mate (type 11, poly A) |
| Inputs | packed read codes, offsets, adapter codes, `Cr4Dev` (`q[40]`, `nq`, `n5`, `after5`, `n3`, `after3`) |
| Outputs | `p5` and `p3` (bases clipped per end, `uint16`, capped at 65,535) and `info` (1 byte per read, the 5' clip's opal result) |
| Launch shape | thread per read, 128 threads per block. `k_cr4<30>` for the 30-base TSO, `k_cr4<0>` for other adapters of up to 40 bases |
| Registers, occupancy, shared memory | Not recorded |
| Memory | `info` is 1 byte per read (66.6 MB on pbmc1k_v3). It returns to the host for STAR's own clip of reads without a record |
| Time | 0.093 s for 66.6 M reads, from the E20 report |
| Exactness | Opal's 8-bit saturating lanes never leave [-126, 40] with a query of at most 40 and these scores. Plain `int` arithmetic gives the same values |
| Limits | Reads up to 650 bases get clipped. `info` is computed for every read |
| Switches | `SINGLET_ENGINE_GPU_CR4_DEVICE=0` runs the host clip. `SINGLET_ENGINE_GPU_CR4_CHECK=1` runs both and compares the three arrays byte for byte |
| Log lines | `E20 CellRanger4 clip check: ... IDENTICAL/DIFFERENT` and `CellRanger4 clip 0 s (on the device: E20, in mapDistinct)` |
| Gate | Checked on neuron1k_v3 (92.9 M reads). The 10-sample gate with the check on is a PR-2 gate |

## 3. Pileup collapse: R9

The pileup collapses junction and SNP molecules per (cell, feature) group with a directional one-mismatch rule (R9 of the order-free rules). The default path uses E23's linear collapse. The older per-group kernel `k_r9` stays behind a switch.

| Item | Value |
|---|---|
| Source | `src/engine_gpu/gpu_pileup/pileup/pileup.cu` |
| Old kernel `k_r9` | One thread per group tests every pair of the group. The cost is quadratic in the group size. On neuron1k_v3 one junction with 14,097 UMIs in one cell took 25.8 s of the pileup's 27 s |
| New path `r9_collapse` | `k_r9_hkey` builds a key per entry from one half of the base-5 UMI digits. A radix sort groups the candidates. `k_r9_pairs` tests each candidate pair with `k_r9`'s rule and unions it at once with `atomicCAS`. `k_r9_flatten` points every entry at its root |
| Why it matches | Two UMIs one base apart agree on one half of their digits, so every one-mismatch pair is a candidate. The union hooks the larger root under the smaller, so each component's root is its smallest entry. That root is the one `k_r9` leaves |
| Gene-aware part | An entry with a gene component takes the first entry of its group with the same component id, from a stable sort by (group, component). `k_cr_*` kernels replay the union in the CPU's order for the intron rule `kUmiGeneCR` |
| Memory | Five dead arrays of the caller are reused as scratch. The reducer's peak does not grow |
| Determinism | Integer sorts, scans and reductions only. No float arithmetic and no atomics on values. The engine builds with `-fmad=false` |
| Switch | `SINGLET_GPU_PILEUP_E23=0` selects the old per-group kernels. Default is on |
| Arena | The pileup arena is the device budget minus 3 GB. About 26 GB on pbmc1k_v3 |
| Fallback | A declined or failed GPU pileup falls back to the CPU pileup with a loud log line. The lines are in `GPU-PATH.md` section 3 |

## 4. Entries still to write

The plan requires one entry per kernel with the same fields as above. These parts of the engine have no entry yet.

| Part | Source | Kernels |
|---|---|---|
| Dedup and clip | `engine_gpu.cu` | `k_hash`, `k_groups`, `k_clip`, `k_gather_clip`, `k_heads`, `k_collide`, `k_readrep`, `k_prefix_key`, `k_rep_len`, `k_rep_copy` |
| Result packing and records | `engine_gpu.cu`, `records_gpu.cu` | `k_pack`, `k_rec_len`, `k_rec_copy`, `k_count`, `k_fill`, `k_crkeys`, `k_eval`, `k_scatter`, `k_pass_eval`, `k_pass_scatter` |
| Per-read products | `hostloop_gpu.cu` | `k_entries`, `k_r11`, `k_mult`, `k_zero`, `k_rec`, `k_sjfill`, `k_sjcounts` |
| Non-host screen | `nonhost_gpu.cu` | `k_count`, `k_fill`, `k_probe`, `k_sum`, `k_write`, `k_first`, `k_gather`, `k_dir` |
| Solo | `solo_gpu.cu` | `kEach` and the Solo library |
| Pileup stages 1 to 7 | `pileup.cu`, `m6.cu` | `k_annotate`, `k_resolve`, `k_lists`, the M6 sort and CSR kernels |
| Legacy kernels | `engine_gpu.cu` | `k_p1`, `k_p3` run only with `SINGLET_ENGINE_GPU_CFG=old` |

Two kernels need a closer look before the next entry. `k_resolve` has an open exon-multimapper defect that blocks PR-4. The two device dedups (E12 and #145) must become one before PR-4 as well.
