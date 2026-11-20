# T2 recrc: records phase over the budget is built by record chunks instead of falling back (recrc)

Task `recrc` of throughput track 2 (branch `t2/recrc`, from `gpu/throughput-2` edccda43). An exact change: outputs identical. Sizes are GiB (the engine prints "GB" for 2^30 bytes). Build `bin/t2-recrc` (job 818987, `ctest exit 0`), default switches except where stated. No `private/` file and no HLA file was opened (hash sets leave `private/` out).

## What changed

One file, `src/engine_gpu/engine_gpu.cu`, in `mapDistinctBody` (28 lines):

1. New switch `SINGLET_ENGINE_GPU_REC_CHUNK` (default on; `=0` restores today's refusal exactly). In the `if (e2e)` block before the `RC ? e2eBuildRC : e2eBuild` choice a local `recChunks` is set when ALL of these hold: no read chunks (`!RC`), a budget (`cap`), `!gatherDev`, more than one record, `recNeed() > capRec` (the test `e2eBuild` would refuse on), `e10`, `entHost`, `out.readRecord.size() == n`, and (`keepLate` or no Solo model). Then `e2eBuildRC` is called instead of `e2eBuild`. In every other case the old code runs, including its refusal.
2. One log line when the path is taken (in `star_Log.out` via `out.log`, and on stderr, no "fallback" words so `gpuFallbackLines` does not count it): `engine_gpu: E17 records by record chunks (SINGLET_ENGINE_GPU_REC_CHUNK): the records' phase needs about X GB, more than the budget of Y GB`. The host lane's `hBlob`/`hOff` are freed there, as `e2eBuild` does for the non-gather case.
3. `e2eBuildRC`'s record counts: it took them from the host (`hcnt`), which exist only with read chunks (`hostCnt = e2e && RC && rec2 != 1`); outside RC `hcnt` is all zero. Now `devPass1 = !hostCnt || rec2 != 0`, and the uses of `hcnt` (the total check, the choice of counts, the log text) are guarded by `hostCnt` instead of `rec2`. With RC on the conditions are equal to the old ones (checked by reading: `hostCnt` is `rec2 != 1` there), so the RC path is unchanged; without RC the device's counting pass runs (the same code as `SINGLET_ENGINE_GPU_REC2PASS=1`).

## Input-by-input check (from the code, before any run)

| input of `e2eBuildRC` | in the non-RC path | finding |
|---|---|---|
| `out.readRecord` (size n) | filled by `fillReadRecord()` only `if (e10)` (both the E20 split branch and the one-hold branch) | exists when E10 is on; with E10 off it is empty and `e2eBuildRC` refuses ("no records"), so the new path requires `e10` and `out.readRecord.size() == n`. With E10 off the old refusal stays. |
| `recSrc`, `recRead` | filled by `hostCopy` for e2e; `e2eBuild` checks their size | `e2eBuildRC` does not read them (only frees them afterwards, as for e2eBuild). Not needed. |
| `out.seq/blob/seqOff/blobOff` (host-assembled records) | filled by `hostCopy` in both paths | same arrays. In the gather case (`gatherDev`, only when not lean) the device holds the results; the new path is not taken there (`!gatherDev`). Lean mode (both test samples) always has `gatherDev` false. |
| `cr.hBlob`, `cr.hOff` | `e2eBuild` frees them in its non-gather branch | freed in the new branch before the call; `e2eBuildRC` does not use them. |
| `auxCompact` | 0 outside RC (`auxCompact` needs `RC`) | `e2eBuildRC` then keeps from `in.e10.aux` with stride `2 + nw`, the same source as `e2eBuild`'s keep; `if (auxCompact)` blocks are skipped. `RCN` is 0 outside RC, so the keep upload is one piece of `n * st * 8` (the aux size, scoped to its block, freed before the record chunks). |
| E11 keep time | `keepLate = lean`; when not lean the keep is made early (`e10Entries`), and `e2eBuildRC` would make it a second time | the new path requires `keepLate` or no Solo model. |
| E10 entries and masks | in the region at once only when `entHost` (lean or E20 split, via `entToRegion`); otherwise on the device (`dEnt`, `dQm`) | `e2eBuildRC` ends with `entHost ? ... : e10Off("the entries are not in the region")`, so the new path requires `entHost`. Both samples run lean and split. |
| `entPre` / `hkPre` (E20 split pre-pass of the hook list) | the split branch computes `hkPre` when `e10 && entHost`, before the relock | `e2eBuildRC` uses `entPre ? hkPre : entHostPass` exactly as `e2eBuild`'s products path does. Same. |
| E20 split branch | releases the device after the lanes, relocks, sets `gatherDev = false`, computes `recSlots` (`want` only when `!RC && admitOn`) | handled: the new choice is made after it, and `capRec` (raised by SL admission) is what `recNeed()` is compared with. The records phase runs in the second hold, as before. |
| `dcs` (chromosome starts), `RP` | same variables | same. |
| record chunk size | `SINGLET_ENGINE_GPU_RECCHUNK`, else a quarter of `min(g_bud.room(), card free - 2 GiB) - slack` | sized from the budget's room and the card's free memory at that point. This is the sizing rule that keeps the phase within the budget. |

Nothing was missing; the one real gap was the record counts (item 3 above), which only showed up by reading `hostCnt`. No safety check was removed.

## Results

### 1. Build
Job 818987: `ctest exit 0` (TESTS=1).

### 2. Identity gate (default settings; the 8 samples; `ref-t2`)
- V100S (tesla_v100s, g002), job 818990: 8 of 8 `VERDICT ... cpu:gpu IDENTICAL (except E11's run-dependent files ...)`. nuclei2k_v2: level `5 of 5`, wall 225.9 s, `VERDICT nuclei2k_v2 cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 92/5/5)`. Its log: `E17 records by record chunks ...: the records' phase needs about 35.96 GB, more than the budget of 29.73 GB`; `E8 records on the device in 6 record chunks of 15728640`; `budget 29.7 GB ... tally peak 28.73 GB at laneM ..., within the budget`; `E19: level 5 reached`. The other seven: level 5, one fallback line each (the R11 info line), unchanged.
  **fallbacks.txt of nuclei2k_v2 holds a second line**: `[nonhost] engine_gpu nonhost: cudaMalloc ... out of memory | ...; screening on the CPU` (at the end of the run). It is not the records phase. It is also in the level-0 run on the same card in this track (job 818991, and gate 818966), so it is not introduced by this change; but by the gate's rule ("no fallback line other than R11") nuclei2k_v2 on the V100S still does not pass strictly. Not investigated here.
- V100S with `SINGLET_ENGINE_GPU_REC_CHUNK=0`, job 818991: 8 of 8 IDENTICAL; nuclei2k_v2 `0 of 5`, wall 301.0 s, falls back as before ("E8 records not built ... needs about 35.96 GB ... budget of 29.73 GB", "level 0 (requested 5) ... FELL BACK").
- RTX 8000 (g005), job 818992: 8 of 8 IDENTICAL, every sample level 5, nuclei2k_v2 one fallback line (R11), the new line not present (not needed at budget 42.3).
(V100S wall: new 225.9 s on g002, old 301.0 s on g003: different nodes, a single run each; not a controlled comparison.)

### 3. H100, budget 31, 2 slots, 1 worker (job 818993, node g051, `t2-recrc-item.sbatch`)
| item | wall | level | new line | tally peak | card peak (nvidia-smi) | host peak | core set hash |
|---|---|---|---|---|---|---|---|
| d100s_3p | **179.37 s** | 5 | needs 35.07, budget 31.00; 5 record chunks of 14680064 | 30.19 GB at laneB, within the budget | 47.3 GiB | 98.0 GiB | 2601bbe3170102c8c3b83764cce0b4e3 (= T2-fallzone) |
| nuclei2k_v2 | **218.36 s** | 5 | needs 43.93, budget 31.00; 6 record chunks of 13631488 | 30.02 GB at laneB, within the budget | 47.2 GiB | 102.4 GiB | 762a1a32a9e986aaf96bf20f58f0bf30 (= T2-fallzone) |

Tally: the engine prints one peak for the whole mapping, with no OVER; its maximum is in the lanes (laneB), so the records phase is at or below 30.19 / 30.02 GB (an upper bound, not a separate measurement; the engine keeps no per-phase peak).
Walls against T2-fallzone: level 0 358.4 / 341.0 s; level 5 at 48 GiB 160.5 / 188.6; forced read chunks 196.9 / 211.3. This path: 179.4 / 218.4 (d100s_3p 2.0x faster than level 0 and 19 s faster than forced read chunks; nuclei2k_v2 1.6x faster than level 0, 7 s slower than forced read chunks, one run each, the run-to-run variation is about 4 percent).

### 4. Controls, same node and job (job 818995, node g051, arms new / old / old / new)
| sample | wall new (x2) | wall old `=0` (x2) | level | new line | core set hash |
|---|---|---|---|---|---|
| d300_3p (read chunks on) | 138.26, 139.20 | 138.26, 137.81 | 5 | none | d6365aaf0b4735a7d742c9e9b51589fb in all four |
| pbmc1k_v3 (level 5, normal mode) | 50.77, 51.42 | 51.55, 51.14 | 5 | none | 55008fd7a3ba3ca1560476a6aee60f85 in all four |
No wall differs beyond the run-to-run variation; the new path was not taken in either (no log line), as designed. (The `E8 records ... one pass; counts on the host` line of d300_3p is the RC path, unchanged.)

### 5. Three workers on one H100 (job 819002, node g051, budget 31, 2 slots, d100s_3p + nuclei2k_v2 + d300_3p together)
Pool wall 354 s. Items: d100s_3p 310.1 s, nuclei2k_v2 343.7 s, d300_3p 324.9 s; all level 5, all exit 0; the new line for the first two; hashes 2601bbe3..., 762a1a32..., d6365aaf... (equal to the single-worker runs). **Card peak 78.7 GiB** (nvidia-smi, 1 Hz, whole card, including the 18 GiB shared index) against the campaign's 86 GiB bar. Host: cgroup peak 190.5 GiB anon+shmem (memory.peak 209 GiB of 300), no oom events.

## Decision on the default
All tests pass on identity (every hash equal, 16 of 16 gate verdicts identical, controls unchanged) and on memory (tally within the budget, card peak 78.7 under 86), except the strict "no fallback line" reading for nuclei2k_v2 on the V100S, whose extra line is the unrelated, already-present nonhost OOM. Default left **on** (`=0` restores the old behaviour); the orchestrator can decide on the nonhost line.

## Not done / limits
- The tally's peak for the records phase alone is not measured (upper bound above). The records phase chunk is sized from the budget's room, which in these runs left the tally within the budget; on the V100S the lanes (28.73) are the peak.
- Only samples that reach `recNeed > capRec` with E10, entHost and lean are routed; a non-lean sample over the budget, or one with E10 off, still falls back as before.
- 150 bp reads, paired-end (mate-gap refusal remains), and the 3-worker shape with the V100S were not tested.
- The poll for jobs ran every 180 s; no job was cancelled or modified other than mine (818987, 818990-818993, 818995, 819002).

Scripts: `bench/gpu-runtime/jobs/t2-recrc-item.sbatch` (Clipper: `jobs/gpu-runtime/t2-recrc-item.sbatch`, `t2-recrc-sub1.sh`); patch `patches/t2-recrc.diff`.
