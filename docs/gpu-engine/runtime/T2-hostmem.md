# T2 host memory per read (hostcut), rounds 1-3

Task `hostcut` of throughput track 2 (branch `t2/hostcut`). Goal: lower the host memory of one deep item with byte-identical outputs, then see what it buys
the pool's admission. Sizes are GiB unless a column says GB (the pool prints GB for what is GiB here: `RssAnon` kB / 1048576). Sample d2600_3p has
1,303,720,518 reads. "Item peak" is `gpu_pool.cpp`'s definition: the maximum over time of the sum of RssAnon + RssShmem of the item's processes
(`SINGLET_POOL_MEMSAMPLE=2`, `memsamp.tsv`); it counts the shared region of the child and the parent twice. "Cgroup peak" is anon + shmem of the job's cgroup
(10 s samples).

Every switch below is ON by default and `=0` restores the old path; `=check` runs both where that is possible and fails the mapping on a difference.
**The constants proposed at the end are valid only with all of these switches at their defaults.**

## Switches

| switch | default | `=0` | `=check` | what |
|---|---|---|---|---|
| `SINGLET_ENGINE_GPU_LEAN_DROP` | on | slices kept to the end of the lanes | released range filled with 0xA5, an upload that finds one fails | `leanSeq` / `leanOff` pages of a lane chunk released after its upload (`dropPages`) |
| `SINGLET_ENGINE_GPU_CLIP16` | on | `hcs` / `hoffC` (two u64 per read) | both, compared per read at the end of the reads phase | clip amounts as u16 into `out.clip5` / `clip3` (`k_clip16`); `SINGLET_ENGINE_GPU_CLIP16_LIMIT=<n>` lowers the u16 limit to test the fallback to the old arrays |
| `SINGLET_ENGINE_GPU_READREP_FREE` | on | kept to the return | none | `readRep` freed where `rrPos` is built |
| `SINGLET_ENGINE_GPU_VERIFY_FREE` | on | freed after the lanes | none | `bgv.I` / `bgv.gh` freed when the group check ends (the check stays on the host; a device check is not possible in the chunked reads phase) |
| `SINGLET_ENGINE_GPU_AUX_COMPACT` | on | aux kept whole | nothing released, compact CR/UR compared with aux before the keep upload | per chunk CR/UR into 16 B/read, the chunk's aux pages released; a failed attempt that is retried rebuilds aux from the planes (`regenAux`) |
| `SINGLET_ENGINE_GPU_PRE_RELEASE` | on | adjustment and store after the read source | `SINGLET_ENGINE_GPU_EARLY_RELEASE=check` compares the store with the planes | RNA Q30 adjustment and compact store of the unmapped reads on a thread from the return of `mapDistinct`; one block search per thread |

## Round 1 (commits 59abf5f4, 00e01755, 0ca50882; merged 98fa4eeb)

| cut | saved by construction | measured |
|---|---|---|
| LEAN_DROP | 13.0 B/read | -4.9 GiB at the peak (294.2 with it off, 289.3 on) |
| CLIP16 | 16 B/read minus the 4 B/read of clip5/clip3 that now exist earlier = 12 | -14.6 GB anon at `rc.reads`; reads-phase D2H to the host 146.9 / 107.8 s (baseline runs) -> 9.8 s |
| READREP_FREE | 4 B/read | in the sum below |
| VERIFY_FREE | 8 B/read | in the sum below |

d2600_3p alone (jobs 818330, 818372 baseline `bin/multirun`; 818354 mine; set hash 415592bfbf2d21d50d2e9ed5b4003655 in all):
cgroup peak 323.3 / 324.9 -> 292.5; inside `mapDistinct` 323.3 -> 289.3; the peak then sat after the return (bump). Gates: 8/8 IDENTICAL each (default, chunked
`RCHUNK=1048576,RPARTS=3,CHUNK=262144`, checks, `CLIP16_LIMIT=20` for the fallback).

## Round 2 (commits b2fb4c79, 373d74d4, dc759ce8; merged b03f2477; build `t2-hostcut5` 818618)

### What the post-return bump is (job 818528, d2600_3p, sampler every ~2 s from the return; sizes at the e16 marks)

| array | at the return | at the bump (+57..60 s) | needed until |
|---|---|---|---|
| planes pseq 26.9 + qual 26.9 + r1 19.4 + off 9.7 | 82.9 | 82.9 | E22 release (round 1: +62 s): `e11RnaAdjust`, `lateBuild`, `e10BuildSource` |
| aux | 38.9 | 0 | freed at `Pre.cpp:1814`, about +4 s |
| compact store of 170,060,995 unmapped reads | 0 | ~14 (23.8 when built) | Unmapped.out formatting, ~+100 s |
| readRecord (static keep) 4.9, clip5+3 4.9, cr4info 1.2, host record offsets 3.1 | 14.1 | 14.1 | STAR loop |
| private STAR index copy in the child | 16.4 | 16.4 | STAR loop |
| Solo keep, page-locked | 0-7 | 27.3 | `soloRunKept`, ~+85..90 s |
| E2E region (shm, mapped by child and parent) | 62.7-67.6 | 67.6 | parent `e2ePass`, ~+85 s |
| parent E2EPassOut | ~0 | 14 (35.7 at +84) | pileup, later |
| node-shared: holder 17.6, non-host server 24.3-25.6 | 42 | 42 | whole run |

The child's ~246 is anon ~150 + shmem 93.4 (region 67.6 + 25.8 page-locked Solo keep). The code study's ~120 GiB after the return left out the Solo keep, the
parent's E2EPassOut, the private index copy, holder and server, and assumed the planes freed (they were freed 62 s after the return, with the store coexisting).

### Result (same set hash 415592bf... everywhere)

| run | cgroup peak | in `mapDistinct` / after return | item peak |
|---|---|---|---|
| baseline 818330 | 323.3 | 323.3 / 276.8 | 347.3 |
| round 1 (818354) | 292.5 | 289.3 / 292.5 | 349.7 |
| step A (818528) | 286.3 | 286.3 / 285 | 337.1 |
| + AUX_COMPACT, PRE_RELEASE v1 (818571) | 282.2 | 263.1 / 282.2 | 347.5 |
| final `t2-hostcut5` (818641) | 262.2 | 262.2 / 252.3 | 324.2 |

Per process (max anon / shm): STAR child 252.9 / 96.7 (baseline) -> 188.1 / 96.7; parent 113.1 / 67.6 (its maximum is in the tail, unchanged).
d300_3p node pair on g051 (A = baseline 818651, 818739; B = hostcut5 818699, 818721; node under cache pressure in all four, MemFree min 39-51 GB): item peak 90.2 -> 74.4;
RUN 4.065 -> 4.244, NODE steady 8.33 -> 8.72 Mpairs/s (inside the ~4 % noise on a disturbed node), core-s/Mpair 3.865 -> 3.935 (+1.8 %), hold 1 39.45 -> 37.95 s; same set hash 391d8ba7...
Other samples with the new build alone: d700s 149.1 (baseline 190.0, same hash cfaaf2b3...), d1000_3p 210.4, d1000s_3p 211.0.
Gates (V100S, RTX 8000 would have started ~13 h later): AUX_COMPACT 818567 / 818568 / 818569 / 818570 (default, chunked, checks, forced retry with the aux rebuilt);
PRE_RELEASE 818637 / 818638 / 818639 / 818640; all 8/8 IDENTICAL; on V100S `nuclei2k_v2` falls back in the non-chunked default runs (budget 29.73 GB < 35.96 GB), also with the
baseline binary.
Not built: masks (16 B/read; the decision `snpSkipN < 0` comes from sums made from the masks) and an earlier release of the qual / r1 planes (needs a spill or a re-read of the .1fq).

## Round 3: the pool's admission estimate

### Reproduction of the comment's fit on the old build (job 816788, mix3, 118 ok items of 126; script `bench/gpu-runtime/t2-memfit.py`)

The constants in `gpu_pool.cpp` (61.8 + 0.2235 x, from "48.2 + 0.2235 x, rmse 13.6, worst under-prediction 30.3") **could not be reproduced exactly**:

| fit set | n | a | b | rmse | worst under-prediction |
|---|---|---|---|---|---|
| all ok items | 118 | 46.5 | 0.2203 | 15.7 | 33.6 (d500s_3p) |
| without pbmc5p_v1 (110 items, 16 samples) | 110 | 49.0 | 0.2154 | 15.1 | 33.5 (d500s_3p) |
| the comment | 110 | 48.2 | 0.2235 | 13.6 | 30.3 (d500s_3p) |

Same procedure, same worst sample, a within 1-3 GiB, b within 0.002-0.008; rmse and worst under-prediction are 1.5-2 and about 3 GiB larger. The reference for both builds
is the script's fit on all ok items: **46.5 + 0.2203 x, margin 35 -> a = 65.1** on the old build. On the old build the code's constants under-predict d500s_3p by 16 GiB
(estimate 173.7, measured 190.1 mean).

### New build, mix3 with `bin/t2-hostcut5` (parameters of 816788: same mix string, K=3 T=8 slots=2 memgb=31 numa=1 prod=1, `--no-hla`, same xenv; defaults, old constants)

| arm | job | node | RUN | NODE steady | admission waits | cgroup anon+shm peak / memory.peak / oom_kill | MemFree min / pressure / compaction stalls |
|---|---|---|---|---|---|---|---|
| A1 | 818763 | g052 | 126/126 ok, wall 11412 s, 2.497 Mpairs/s, max_item_rss 238 GB | 2.624 | 13 + 13 waits, 10280 + 10289 s | 314.6 GB / not recorded (the harness did not read it) / not recorded (all 126 items exit 0) | 2.3 GB / 89 % / 206,592 |
| A1b (repeat) | 818882 | g052 | 126/126 ok, wall 10633 s, 2.680 Mpairs/s, max_item_rss 248 GB | 2.807 | 13 + 9 waits, 9579 + 9614 s | 308.4 GB / 407.3 GiB / oom_kill 0 | 3.8 GB / 33 % / 204,886 |

Both runs were disturbed by the brief's rule (MemFree min far under 140 GB, pressure far above a few percent); the node was otherwise idle (`sacct -N g052`: only these jobs),
so the pressure is the run's own plus page cache the node held at the start (Cached 448 / 386 GB). Per the round-3 rule both are reported **invalid as rate
measurements** and the A/B (new constants through `SINGLET_POOL_EST_BASE_GB` / `SINGLET_POOL_EST_GB_PER_MPAIR`) was **not run**. Outputs: every sample has one set hash across its items and arms
(d2600 415592bf..., d300_3p 391d8ba7..., d700s cfaaf2b3..., d1000 01e155a8..., d1000s bc100f5b... as in the single-item runs). All 126 items completed in both; 816788 had 8 device-memory failures
(d700s 7 of 8, d1000s 1 of 1) and the same items did not fail here (the device-OOM recovery of rt/devoom had been merged since).
For information: at the cgroup peak A1 had d2600 + pbmc5p in flight, sum of the pool's estimates 428 GB against 314.6 (ratio 1.36); A1b 3 items, 477 against 308.4 (1.55);
816788 (old build, other admission) 6 items, 776 against 415.8 (1.87). The per-process sum is conservative by 1.4-1.9.

Fits on the new build (script, all ok items):

| run | n | a | b | rmse | worst under-prediction | margin (up to 5) | a_new = a + margin - 16.4 |
|---|---|---|---|---|---|---|---|
| A1 | 126 | 48.9 | 0.1590 | 16.2 | 84.6 (d2600_3p, measured 340.8) | 85 | 117.5 |
| A1b | 126 | 49.4 | 0.1572 | 15.7 | 67.8 (d2600_3p, measured 322.1) | 70 | 103.0 |
| A1b without d2600_3p | 125 | 51.3 | 0.1457 | 14.3 | 23.5 (d500s_3p) | 25 | 59.9 |
| A1b against the .1fq size (GiB) | 126 | 48.5 | 6.447 /GiB | 14.8 | 127.0 (d2600_3p) | 130 | 162.1 |

Per-sample mean item peaks, old build (816788) against new (A1b), GiB; the new build's old-constant estimate in brackets:
d2600 342.3 / 322.1 [353]; d1000 270.8 / 213.9 [285]; d1000s failed / 210.4 [286]; d700s 189.8 / 149.7 [219]; d500s 190.1 / 146.9 [174]; hgmm5k 109.6 / 103.7 [132]; d300_3p 90.5 / 74.3 [129];
d300_5p 102.3 / 85.0 [129]; nuclei2k 81.0 / 80.9 [88]; d100s 87.7 / 88.3 [84]; neuron1k 65.9 / 65.7 [83]; hgmm1k_v2 63.3 / 63.7 [79]; pbmc1k_v3 63.4 / 63.3 [77]; hgmm1k_v3 63.2 / 63.4 [76];
pbmc5p 37.7 / 36.2 [75]; pbmc1k_cite 61.6 / 61.7 [72]; pbmc10m 58.9 / 59.3 [64]; pbmc2m 57.7 / 57.5 [62].
The cuts lower the 150 bp deep samples (d500s, d700s, d1000s) and d300 by 15-60 GiB but not the small samples (planes and the tail dominate there) and d2600 by only 20 in the mix
(322-341 against 324 alone). The new points are not on a line (d2600 is far above the line of the rest), so the single worst point sets the margin.

### Constants by the comment's rule

Rule: margin = worst under-prediction rounded up to 5 GiB, genome 16.4 off. On the new build that gives **a = 103.0, b = 0.1572** (A1b) or a = 117.5, b = 0.1590 (A1). They
estimate d2600 at 308 GB (353 now) but d300 at 150 (129 now) and d700s at 213 (219 now): the margin comes from one item, so the rule makes mid-size items more expensive and
d2600 cheaper. **I do not recommend adopting them**: the two runs behind them are invalid by the node rule, the linear model fits the new build poorly, and the code's own
constants are already above the new build's measured peak of every sample class except d100s (4 GiB under; on the old build also d500s, 16 GiB under). If constants are changed they
are valid only with the switches above at their defaults and only for this build; `gpu_pool.cpp` was not edited.

### Two d2600_3p items in one pool (harness `bench/gpu-runtime/jobs/t2-rt_deep2.sbatch`: `--workers 2`, `--exclusive`, `--mem=700G`, one H100, both items the same input)

| arm | job | in flight together | wall item 0 / item 1 | pool wall | cgroup anon+shm peak / memory.peak / oom_kill | set hash of both |
|---|---|---|---|---|---|---|
| old constants (61.8 / 0.2235) | 818769 (g051) | no: item 1 waited 618 s ("estimate 353 GB; free + cache share 227 GB, own items still to allocate 261 GB, reserve 40 GB; projected -388 GB") | 658.3 / 1155.1 s | 1828 s | 275.0 GiB / 378.8 GB / 0 | 415592bf... both |
| new constants 103.0 / 0.1572 (A1b's) | 818931 (g051) | no: item 1 waited 597 s (estimate 308 GB; free + cache share 300 GB, own items still to allocate 215 GB; projected -223 GB) | 637.0 / 619.0 s | 1274 s | 270.3 GiB / 388.8 GB / 0 | 415592bf... both |

With either set the second item starts only when the first ends (the first's charge drops 308 -> 123 -> 80 GB in its phases, but the projected room stays negative until then), so the
new constants did not raise occupancy on this node and there was no oom_kill. Item 1 of the old-constants run took 1155 s against 658 s (system time 5236 s against 713 s), not
explained by the constants (not repeated).

## What is left (not built)
- qual / r1 / pseq / off planes (83 GiB) read again only after hold 2 (adjustment, store of the unmapped reads): earlier release needs a spill or a re-read of the .1fq, or block streaming.
- masks (16 B/read in the region, 15.9 GiB E2EPassOut.lowq): the need is decided after they are built.
- A/B of the new constants on a node that is not at its memory floor with the old ones: not obtained.
