# T2 cachetest: whose is the page cache at the start, and what does the pool's cache share do (cachetest)

Measurement and bench scripts only; nothing under `src/` or `include/` changed. Branch `t2/cachetest` (from `gpu/throughput-2` f2e3598b).
All sizes are GiB (2^30), printed as "GB" by the harness, as in the other T2 notes. Every number below is from the jobs' own lines
(`$W/jobs/gpu-runtime/results/rt-node-<job>-<tag>/results.txt`, `cache-<label>.txt`, `cachesamp.tsv`); estimates are marked.

## Summary

| question | measured |
|---|---|
| Whose is the cache at the start? | Clean start (818957, g052, 16:43): Cached 270.0 GB = **88.5 GB our mix source files** (33 %) + 11.7 GB other files of ours (genome index, whitelists, binary) + **169.8 GB not attributable by anything scanned** (63 %). No other input files, no `/tmp/$USER-*` leftovers were in cache. The hypothesis is **partly right**: a third of the start cache is our own source pages; most of it is not. |
| And after the copy? | Cached 421-469 GB, of which 288 GB ours (144 GB node-local copies, dirty at first, + 144 GB source pages). The pool drops the node-local pages at its start; **the source pages stay** (144 GB at the end of the default-share control, 40 GB at the end of the share-1.0 run). |
| Cache share 1.0 against the round-3 baselines | all 126 items ok, wall 6,805 s against 11,412 / 10,633 s, rate_overall **4.187** against 2.497 / 2.680 Mpairs/s (+68 % / +56 %), admission wait 5,925 s against 20,569 / 19,193 s, no oom_kill. The baselines, however, started on a different node state (start Cached 448 / 386 GB). |
| The control I added (default share 0.25, no change, started at Cached 275 GB) | rate_overall **3.173**, wall 8,979 s, wait 15,559 s. It is already 27 % / 18 % faster than the two baselines with the same binary, node and settings. **The baselines do not serve as a control for today's node state; the three runs of 2026-11-19/20 are the comparison.** |
| Share 1.0 against that control | +32.0 % in rate_overall (4.187 / 3.173), wait 5,925 against 15,559 s. One run each; outside the ~4 % run-to-run variation. The start cache of the share-1.0 run held 144 GB more of our source pages (325 against 275 GB; an aborted job of mine had just read the sources), so the node state at the start is not identical. |
| Dropping our source cache at the default share (step 3) against the control | +19.2 % (3.783 / 3.173), MemFree min 140-153 GB instead of 49-56 GB, memory pressure max 1 % instead of 2-3 %, compaction stalls 14,071 instead of 94,608, wait 11,496 s. Share 1.0 is +10.7 % over this arm (4.187 / 3.783, one run each). |
| Cost seen with share 1.0 | MemFree min 2.0-2.3 GB, node memory pressure (avg10) max 49 %, 3.93 M direct-reclaim page scans (the other two runs 0 and 0), 104,570 compaction stalls, `memory.peak` 560.7 GiB (the job asked for 700G): **no OOM kill, no failed item, outputs identical**. Items ran slower (item_wall_mean 262 s against 234 s control) but more of them ran at once. |

Order of the runs, all on g052, exclusive, `--mem=700G`, each alone on the node (`sacct -N g052`; the other agents' jobs ran before and after on the same node, none during):

| job | tag | what | start (EST) | state |
|---|---|---|---|---|
| 818957 | t2ct-S100 (first try) | share 1.0 | 16:43:41 | **cancelled by me after 6 min, before any item**: the job's `wait` waited for the 1 Hz sampler too (bug in my script, fixed); its start snapshot is the only clean-start attribution |
| 818958 | t2ct-S100 | share 1.0 | 16:50:22 | COMPLETED 1:55:19 |
| 818962 | t2ct-DROP | default share, source / leftover cache dropped | 20:30:54 | COMPLETED 2:08:12 |
| 819044 | t2ct-CTRL | default share, nothing else | 03:30:14 (11-20) | COMPLETED 2:31:31 |

Step 4 (an intermediate share) was **not run**: I counted 818957 against the four-run budget and used the last slot for the control (CTRL), which I judged more informative
than a third share value; the queue wait for each run was 1.5-7 h because other agents' jobs held the 3 H100 of the account.

## Arm (identical in all runs and in the round-3 baselines 818763 / 818882, read from their results.txt first line)

Binary `bin/t2-hostcut5/singlet-gpu` (`b21de2cf` + patch; commit-gpu line `b21de2cfb748f257385f4d4973e88f5639acc054`), `K=3 T=8 slots=2 memgb=31 numa=1 PROD=1 PARGS=--no-hla`
(the run's pipeline arguments: `--hla-t1k .../t1k-p3b --no-editing --no-hla`), `LIM=12000`, 126 items (mix3, 18 samples, `MIXORDER` default spread):
`MIX='d2600_3p d1000_3p*2 d1000s_3p d700s_3p*8 d500s_3p*8 d300_3p*20 d300_5p*8 hgmm5k_v3*4 nuclei2k_v2*10 neuron1k_v3*6 d100s_3p*8 pbmc5p_v1*8 pbmc1k_v3*10 hgmm1k_v2*6 hgmm1k_v3*6 pbmc1k_cite_v3*6 pbmc10m*10 pbmc2m*4'`,
`XENV='SINGLET_TAIL_THREADS=16 SINGLET_INGEST_THREADS=16 SINGLET_POOL_MEMSAMPLE=2'` (+ `SINGLET_POOL_CACHE_SHARE=1.0` in S100). `REFMIX` = `mixident.tsv` of 818882.
Job script `t2-cachetest-mix.sbatch` = `t2-rt_mix.sbatch` + the cache snapshots, the 1 Hz sampler and `DROPSRC`; submitted with `sbatch --exclusive -w g052 --time=04:00:00` (the baselines had `--time=04:00:00` and `-c 54`, 700G from the script header; `--exclusive` and `-w g052` added by me).

## Step 1: whose is the cache

Method (`t2-cachetest-fincore.py`, mmap + `mincore`; `fincore` and `vmtouch` are not installed on g052): per-file page-cache residency, counted once per file in the first matching category:
`scr` (this job's node-local dir), `mixsrc` (the 18 shared-file-system `.1fq` the mix reads), `tmpours` (`/tmp/$USER-*`), `data` (every other file under `$W/data/gpu-deep` and `$PB/data`, 2 levels: 141 files, 137 GB on disk),
`ref` (reference genome dir, whitelists, the binary's dir), and from the second try on `envs`, `logs`, `tmpall` (other tenants' `/tmp`, 2 levels). unexplained = Cached - Shmem - sum of the categories (Shmem was 0 in every snapshot).
The category sums printed to the job log of 818957 are wrong (the awk summed the file counts: a bug I fixed afterwards); the numbers below for 818957 are recomputed from its `cache-start.txt`.

| snapshot | Cached | scr | mixsrc | other inputs (`data`) | `/tmp/$USER-*` | ref | explained | **unexplained** |
|---|---|---|---|---|---|---|---|---|
| 818957 start (the clean start; before the copy) | 270.0 | 0 | **88.5** | 0 | 0 | 11.7 | 100.2 | **169.8** |
| 818958 start (7 min after 818957 had copied the sources) | 325.4 | 0 | 144.0 | 0 | 0 | 11.7 | 155.6 | 169.8 |
| 818958 after the copy | 469.4 | 143.9 | 144.0 | 0 | 0 | 11.7 | 299.6 | 169.8 |
| 818958 end (all items done) | 184.3 | 9.3 | 40.2 | 0 | 0 | 16.3 | 65.8 | 118.5 |
| 818958 after the job's cleanup | 175.0 | 0 | 40.2 | 0 | 0 | 16.3 | 56.5 | 118.5 |
| 818962 (DROP) start | 277.8 | 0 | 144.0 | 0 | 0 | 18.1 | 162.1 | 115.7 |
| 818962 after dropping our sources / leftovers, before the copy | 133.9 | 0 | 0 | 0 | 0 | 18.1 | 18.1 | 115.7 |
| 818962 after the copy | 421.7 | 143.9 | 144.0 | 0 | 0 | 18.1 | 306.0 | 115.7 |
| 818962 after dropping each source right after the copy (`dd iflag=nocache count=0`) | 277.8 | 143.9 | 0 | 0 | 0 | 18.1 | 162.1 | 115.7 |
| 818962 end | 143.5 | 9.3 | 0 | 0 | 0 | 18.1 | 27.4 | 116.1 |
| 819044 (CTRL) start | 274.8 | 0 | 141.4 | 0 | 0 | 18.1 | 159.5 | 115.3 |
| 819044 after the copy | 421.3 | 143.9 | 144.0 | 0 | 0 | 18.1 | 306.0 | 115.3 |
| 819044 end | 286.7 | 9.3 | 144.0 | 0 | 0 | 18.1 | 171.4 | 115.3 |
| 819044 after the cleanup | 277.4 | 0 | 144.0 | 0 | 0 | 18.1 | 162.1 | 115.3 |

(The categories `envs`, `logs`, `tmpall` were 0 GB in every snapshot that had them; 818957 had only the first five.)

Per source file in the clean start (cached of size, GiB): d2600_3p 22.3 of 22.7, d500s_3p 15.4 of 17.9, d1000s_3p 9.9 of 25.7, hgmm5k_v3 6.6 of 7.3, d1000_3p 6.1 of 23.2, d700s_3p 5.5 of 16.3, d300_5p 5.2 of 6.5,
nuclei2k_v2 4.1 of 4.4, d100s_3p 3.7 of 4.7, d300_3p 2.9 of 5.6, and 1-2 GB each for the small samples (all in `cache-start.files.tsv` of 818957).

Facts: (a) at a clean start 88.5 of 270 GB (33 %) were our own input source pages and 12 GB more other files of ours; the other 170 GB (63 %) are in none of the places I scanned (not our other inputs, not our `/tmp` leftovers, not other tenants' `/tmp` entries,
not `/dev/shm`, since Shmem was 0). The unexplained part was 169.8 GB at 16:43, 115.7 GB at 20:30 and 115.3 GB at 03:30 and stayed constant inside each job except for the drop in 818958 (118.5 GB at the end): it is the
same set of pages being held between our jobs, not growing with them (estimate: other tenants' files on the shared file system and the system's own files; the method cannot say which). (b) After the copy 61-68 % of the cache is ours
(288 of 421-469 GB), of which the 144 GB of node-local copies are released by the pool at its start. (c) The source pages are the ones our jobs leave behind for the next job: 141-144 GB were in cache at the start of 818958, 818962 and 819044,
because another job of ours (an agent's cggate jobs for 818962 and 819044, my aborted 818957 for 818958) had read the same files shortly before. (d) The round-3 baselines had no attribution; their start caches (386 and 448 GB) can have contained
up to 144 GB of source pages of the same kind (estimate, unproven). The criterion for step 3 (at least 100 GB of the cache at the start or after the copy is our own source files) was met after the copy (144 GB) and at the start of 818958 / 818962 / 819044, but not at the clean start of 818957 (88.5 GB).

## Step 2 and 3: results

RUN lines (verbatim) and the other result lines:

```
RUN t2ct-S100 K=3 T=8 items=126 ok=126 wall=6805s pairs=28493.3M rate_overall=4.187 Mpairs/s item_wall_mean=262s core_s_per_Mpair=7.30 max_item_rss=248GB status=ok     (818958)
RUN t2ct-DROP K=3 T=8 items=126 ok=126 wall=7533s pairs=28493.3M rate_overall=3.783 Mpairs/s item_wall_mean=247s core_s_per_Mpair=7.13 max_item_rss=248GB status=ok    (818962)
RUN t2ct-CTRL K=3 T=8 items=126 ok=126 wall=8979s pairs=28493.3M rate_overall=3.173 Mpairs/s item_wall_mean=234s core_s_per_Mpair=7.05 max_item_rss=248GB status=ok    (819044)
RUN t2h3-A1  K=3 T=8 items=126 ok=126 wall=11412s pairs=28493.3M rate_overall=2.497 Mpairs/s item_wall_mean=228s core_s_per_Mpair=7.10 max_item_rss=238GB status=ok    (818763)
RUN t2h3-A1b K=3 T=8 items=126 ok=126 wall=10633s pairs=28493.3M rate_overall=2.680 Mpairs/s item_wall_mean=227s core_s_per_Mpair=7.13 max_item_rss=248GB status=ok    (818882)
```

| | 818763 (A1, baseline) | 818882 (A1b, baseline) | 819044 CTRL (0.25) | 818962 DROP (0.25 + drop) | 818958 S100 (share 1.0) |
|---|---|---|---|---|---|
| node, start Cached / MemFree (GB) | g052, 448 / 288 | g052, 386 / 349 | g052, 275 / 459 | g052, 278 / 458 (134 after the drop) | g052, 325 / 411 |
| wall (s) | 11,412 | 10,633 | 8,979 | 7,533 | **6,805** |
| rate_overall (Mpairs/s) | 2.497 | 2.680 | 3.173 | 3.783 | **4.187** |
| NODE steady-state rate | 2.624 | 2.807 | 3.281 | 4.020 | 4.980 |
| mid-window rate | 2.809 | 2.708 | 3.522 | 4.293 | 4.718 |
| GPU0 / GPU1 pool wall (s) | 10,985 / 11,412 | 10,381 / 10,448 | 8,977 / 8,700 | 7,533 / 6,932 | 6,805 / 5,559 |
| items ok / failed | 126 / 0 | 126 / 0 | 126 / 0 | 126 / 0 | 126 / 0 |
| admission waits, pool0 + pool1 (count, s) | 13 + 13, 10,280 + 10,289 | 13 + 9, 9,579 + 9,614 | 10 + 11, 8,124 + 7,435 (15,559) | 9 + 8, 5,524 + 5,972 (11,496) | 9 + 18, 3,507 + 2,418 (5,925) |
| host peak anon+shmem (GB) | 315 | 308 | 356 | 405 | 540 |
| cgroup `memory.peak` | not recorded | 407.3 GiB | 417.4 GiB | 472.0 GiB | 560.7 GiB |
| `memory.events` (max / oom / oom_kill) | not recorded | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| card peak GPU0 / GPU1 (GB) | 82.0 / 79.4 | 85.0 / 79.3 | 90.3 / 80.3 | 90.1 / 87.7 | 85.4 / 82.1 |
| node MemFree min (GB; results.txt line, 10 s) | 2.3 | 3.8 | 55.6 | 153.3 | 2.3 |
| MemFree min, 1 Hz sampler (GB) | - | - | 49.0 | 140.4 | 2.0 |
| memory pressure max (cgroup full avg10) | 89 % | 33 % | 2 % | 1 % | 47 % |
| node pressure avg10 some / full max (1 Hz) | - | - | 3.0 / 3.0 % | 3.1 / 0.8 % | 49.0 / 49.0 % |
| direct compaction stalls (run) | 206,592 | 204,886 | 94,608 | 14,071 | 104,570 |
| pgscan_direct / pgsteal_direct / pgscan_kswapd / allocstall / thp_fault_fallback (run) | - | - | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | 3,934,808 / 3,934,640 / 38,225,742 / 3,876 / 274 |
| Cached - Shmem min / max (GB, 1 Hz) | - | - | 277 / 422 | 134 / 289 | 176 / 488 |
| Dirty max (GB, 1 Hz) | - | - | 102.7 | 72.8 | 102.1 |
| cores busy (middle 60 %) | 19.1 | 20.7 | 23.5 | 29.3 | 34.6 |
| IDENT: one hash per sample / against REFMIX (818882) | yes / (A1 vs A1b identical per the hostmem note) | yes / reference | yes / all 18 samples identical | yes / all 18 identical | yes / all 18 identical |

Ratios (rate_overall; single runs, estimates of the effect with the ~4 % run-to-run variation as the floor): S100 / CTRL 1.320; DROP / CTRL 1.192; S100 / DROP 1.107; CTRL / 818763 1.271, CTRL / 818882 1.184; S100 / 818763 1.677, S100 / 818882 1.562;
DROP / 818763 1.515, DROP / 818882 1.412. The two baselines differ from each other by 7.3 %.

Per-sample item wall means (s; baseline A1b / CTRL / DROP / S100): d2600_3p 652 / 651 / 641 / 724; d1000_3p 1136 / 1249 / 1301 / 1362; d1000s_3p 771 / 730 / 786 / 753; d700s_3p 500 / 510 / 509 / 558; d500s_3p 504 / 519 / 535 / 567;
d300_3p 147 / 159 / 172 / 190; d300_5p 166 / 175 / 176 / 212; nuclei2k_v2 411 / 404 / 464 / 489; the small samples are at the noise of their 20-150 s walls. With share 1.0 the larger items are 5-30 % slower than in the control while the node ran faster as a whole.

Flag lines (the harness counts engine lines matching widen|unified|managed|fallback|OOM|cudaMalloc, which includes the normal `[engine_gpu] budget ... room N GB` line): 126 (S100), 129 (DROP), 128 (CTRL) in 126 items, against 127 and 126 in the baselines;
"pool stdout widen/fallback lines" was 0 in S100 (not read for the other two). No failed item, no `abort` line in any of the three runs.

Node validity (the brief's rule: another tenant's job on the node): none of my three runs shared g052 with another job (`sacct -N g052` for the windows; 818985, another agent's, was cancelled at 20:30:24, 30 s before 818962 started, and had run 3 min). With share 1.0, low MemFree is expected, so MemFree is not used as a disturbance
test here. The same sacct check for the baselines is in the hostmem note (only these jobs). Shared storage and the network cannot be excluded for any run.

### What the numbers say, and do not say

- Direction: share 1.0 was faster than the default in this one pair by 32 % and than the drop arm by 11 %; the drop arm was faster than the control by 19 %. Each is one run, from different but similar start states (Cached 275-325 GB). The variation between days is not small: the control and the baselines
  (same binary, node and settings) differ by 18-27 %, so a gain of the size seen for the drop arm (19 %) is about the size of the day-to-day spread between the two controls and the baselines. I cannot separate "the share" from "the node state" with these runs alone; a repeat of CTRL and of S100 (A/B/B/A, as the brief prescribes) would be needed to put a number on the share's effect.
- What changed with share 1.0 in the run: the pools waited 5,925 s instead of 15,559 s (control), the host held 540 GB of anon+shmem instead of 356 (more items in flight at once: cores busy 34.6 against 23.5), and the node paid for it with MemFree 2 GB, pressure up to 49 % and 3.9 M direct-reclaim page scans. The run was still faster.
  No job was killed and no item failed at this node state and `--mem=700G`; this does not show that it is safe at another memory limit (T2-pooladmit.md section 1: a cgroup limit below the node's can bind first).
- Lower node pressure in CTRL / DROP than in the baselines (MemFree min 49-153 GB against 2-4 GB) goes with a lower start cache (275 / 278 against 386 / 448 GB), 6-15x fewer compaction stalls in DROP and 2x fewer in CTRL; the baselines' own cache was higher by 111-173 GB at the start. This is consistent with the idea that the baselines' slowness came from the node state at the start, but I did not run the baseline arm again on a clean state with the instrumentation, and no cause is established.
- The drop arm only removes what is ours; at its end the cache was 143.5 GB (116 GB unexplained, 18 GB reference). It still waited 11,496 s: dropping our pages raised MemFree at the waits (MemFree min 140 GB) but did not remove the waits.

## Scripts and files

- `bench/gpu-runtime/jobs/t2-cachetest-mix.sbatch` (copy on Clipper `$W/jobs/gpu-runtime/t2-cachetest-mix.sbatch`): `t2-rt_mix.sbatch` with `cachesnap start|afterdrop0|aftercopy|afterdrop|end|afterclean`, the 1 Hz sampler, `DROPSRC=1`, `--time=04:00:00`. The two bugs found in the first try are fixed in the committed version (the `wait` waits only for the copies; the category sums use column 4).
- `bench/gpu-runtime/t2-cachetest-fincore.py`: `name:depth:path` categories, mmap + mincore, `--drop` for `posix_fadvise(DONTNEED)`, `--tsv` per-file list.
- `bench/gpu-runtime/t2-cachetest-samp.sh`: 1 Hz node and cgroup memory, pressure, reclaim counters.
- Results on Clipper: `$W/jobs/gpu-runtime/results/rt-node-{818957-t2ct-S100,818958-t2ct-S100,818962-t2ct-DROP,819044-t2ct-CTRL}` (`results.txt`, `cache-*.txt`, `cache-*.files.tsv`, `cachesamp.tsv`, `mixident.tsv`); logs `$W/logs/rt-cachetest-mix-<job>.out`.
- Deviations: (1) 818957 was cancelled by me (own job) after 6 min. (2) 818957 ran the first version of the script (wrong category sums in the log line, no `envs`/`logs`/`tmpall`); its start was recomputed by hand from `cache-start.txt`. 818958, 818962 and 819044 ran the fixed version.
  (3) I wrote two small test files to the login node's `/tmp` (`t2ct-test.tsv`, `t2ct-samp.tsv`) while testing the helper, which is outside `$W/jobs/gpu-runtime`; nothing was deleted remotely.
  (4) The start state of 818958 was not clean (above). (5) Step 4 not run (above).
