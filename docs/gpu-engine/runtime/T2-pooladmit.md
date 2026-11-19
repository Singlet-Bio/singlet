# T2 pool admission: where the gate loses memory (pooladmit)

Task `pooladmit` of throughput track 2 (branch `t2/pooladmit`, measurement and analysis only: no file under `src/` or `include/` changed, `gpu_pool.cpp` not edited).
Build `bin/t2-hostcut5` (v1.4.0 + the hostcut cuts, all switches at their defaults) in every run. Units: the pool prints GiB with the label "GB" (`RssAnon` kB / 1048576); this
document says GiB for figures from the cgroup and from the scripts, and keeps "GB" inside quoted pool lines. Scripts (all in `bench/gpu-runtime/`): `t2-cgsamp.sh` (1 Hz cgroup and node sampler), `jobs/t2-pooladmit-cg.sbatch`
(loss A), `jobs/t2-pooladmit-true.sbatch` (loss B), `t2-pooladmit-cg.py`, `t2-cg-events.py`, `t2-true-peaks.py`, `t2-admit-replay.py` (loss C). The orchestrator's reading of the code
was checked against the data; where it did not hold, it is said below.

## Summary

| loss | measured answer |
|---|---|
| A: the gate does not see the job's memory limit | **The code reading is right and the data show it.** `gateTry` takes MemFree + 0.25 x (Cached - Shmem) of `/proc/meminfo`; the cgroup is read only by `avail_gb()` (stagger line, optional floor). In 818937 (`--mem=200G`, three d300_3p) the gate admitted the third item with `free + cache share` 316.8 GiB while the cgroup had 63.7 GiB of headroom by the `avail_gb` rule (55.5 by `memory.max - memory.current`). The cgroup did not reach its limit (peak `memory.current` 181.6 of 200 GiB, `memory.events` all 0). In the supplementary 818940 (`--mem=150G`, same items, same admissions) the cgroup hit its limit: `max` 18207, `oom` 24, **`oom_kill` 2**: two of the three items were killed. A gate that used the limit together with today's estimates would have refused items 1 and 2 of the 200 GiB run (projected -105 and -147 GiB) although all three fitted: estimate 129 each, sum 387, against a true anon+shmem peak of 157.1 GiB for the three. |
| B: the RSS sum double counts | **Only partly.** RSS-sum / true is 1.5-2.3 for six of the seven samples with 75 M reads or fewer (pbmc5p_v1: 1.00), 0.97-1.18 from 93 M to 1,005 M reads, and 1.30 for d2600_3p (325.6 against 250.5 GiB). The large gap is **the estimate against the true peak**: the pool's estimate (61.8 + 0.2235 x) is above the true peak of all 18 classes (true / estimate 0.40-0.95), by 5 to 103 GiB. Fit of the true peak: 35.50 + 0.1709 x reads(M), rmse 12.8 GiB, worst under-prediction 27.1 GiB (nuclei2k_v2). |
| C: page cache | **Cache is the dominant loss in the replay.** At the 22 (818882) and 26 (818763) logged waits the node's file cache was 328-545 and 341-697 GiB (44-73 % and 45-93 % of the 750 GB), of which the job's cgroup held 76-216 GiB and 252-481 GiB was not the job's (already there at the start: Cached 386 and 448 GB). With cache share 1.0 the replayed gate waits 1,778 s (818882) and 620 s (818763) instead of 17,368 and 18,652 s (replayed baseline; real 19,195 and 20,570): 90 and 97 % of the wait. Charging every item its true figure with the cache at 0.25 leaves 6,119 and 7,914 s: 65 and 58 % saved. Both together 667 and 354 s. This is a replay of the gate's inequality, not a prediction that the run would have been faster or safe (see "what a change to the gate would have to respect"). |

## 1. Loss A: the gate against the job's limit

Harness `jobs/t2-pooladmit-cg.sbatch` (derived from `t2-rt_deep2.sbatch`; `rt_node` and `t2-rt_mix` were not used: both are 2-GPU `--mem=700G` jobs): one `nvidia_h100_nvl:1`, `-c 27`, **not exclusive**, `--mem=200G`,
`bin/t2-hostcut5`, three d300_3p items in one pool, `--workers 3`, `--threads 8`, defaults, `SINGLET_POOL_MEMSAMPLE=2`. `t2-cgsamp.sh` samples once a second beside the pool: the job cgroup
(`/sys/fs/cgroup/system.slice/slurmstepd.scope/job_<id>`, found as `avail_gb()` does) `memory.max`, `memory.current`, `anon`, `file`, `shmem`, `memory.events`, pressure, and `/proc/meminfo` MemFree, Cached, Shmem.

| job | node | `--mem` | memory.max | squeue on the node at start | result |
|---|---|---|---|---|---|
| 818937 | g051 | 200G | 214748364800 B | only this job | 3/3 items exit 0, pool wall 241 s; `memory.events` low 0 high 0 max 0 oom 0 oom_kill 0; `memory.peak` 195254714368 B (181.8 GiB) |
| 818940 (supplementary, not asked for) | g051 | 150G | 161061273600 B | only this job | items 1 and 2 exit 3 (killed), item 0 exit 0; `memory.events` low 0 high 44 **max 18207 oom 24 oom_kill 2** oom_group_kill 0; `memory.peak` = memory.max |

Neither node was disturbed (the job was alone on it; MemFree minimum 184.8 and 212.7 GiB). The pool logged **no "memory admission: item N waits" line in either job** (`memory admission: 0 waits, 0 s waited in all`): the gate passed every start at once, so there is no wait line to put
next to the headroom. What the pool did log is the stagger line, which prints `avail_gb()` (cgroup aware): "available 166 GB" at the start of item 0 and "107 GB" at the start of item 1 in 818937 (118 and 55 GB in 818940); the gate does not use that figure.
The gate's inequality was therefore evaluated at each start from the 1 Hz series (`t2-pooladmit-cg.py`; own = what the items in flight can still allocate by the pool's rule, from `memsamp.tsv`; the same reconstruction reproduces the logged figures of the mix runs to within a few GiB, section 3). Estimate 129 each, reserve 40, GiB:

818937 (`--mem=200G`):
```
cgroup memory.max 200.0 GiB; samples 233 (242 s); memory.events (max oom oom_kill) values seen: ['0 0 0']
peaks (GiB): memory.current 181.6, anon+shmem 157.1, anon 111.2, file 86.2; minimum headroom max-current 18.4 GiB at +148 s; node MemFree min 184.8 GB, Cached max 459.5 GB
"memory admission: item N waits" lines: 0
pool RssAnon (sharedGb) 17.6 GB; estimate per item 129 GB; per-item RSS-sum peak (pool definition): item 0 74.6, item 1 74.7, item 2 74.6
gate at every start (1 Hz series; estimate 129; reserve 40):
  item +s  MemFree  Cached-Shm  freeEff | own(others)  proj=freeEff-own-est | cg max-current  cg avail_gb-rule  | would a cgroup-aware gate admit: min(freeEff, avail_gb) - own - est
  0  +  5    335.4      380.6    430.6 |    0.0    301.7 (alone) |   168.8    171.9 |    43.0
  1  + 45    263.2      389.8    360.7 |   85.3    146.4 |   100.8    109.0 |  -105.2
  2  + 85    219.3      389.9    316.8 |   82.0    105.9 |    55.5     63.7 |  -147.2
at the peak of memory.current (+148 s): current 181.6 GiB = anon 110.7 + file 70.2 + (shmem 46.5 inside both); MemFree 184.8 GB
sum of the three estimates 387 (pool GB = GiB) against memory.max 200 GiB; true anon+shmem peak 157.1 GiB
```
818940 (`--mem=150G`):
```
cgroup memory.max 150.0 GiB; samples 155 (159 s); memory.events (max oom oom_kill) values seen: ['0 0 0', '1709 0 0', '1777 0 0', '18207 24 2', '1963 0 0', '219 0 0', '2871 0 0', '3857 0 0', '4806 0 0', '5510 0 0', '5782 0 0', '7065 0 0', '830 0 0']
peaks (GiB): memory.current 150.0, anon+shmem 147.4, anon 96.6, file 67.4; minimum headroom max-current 0.0 GiB at +99 s; node MemFree min 212.7 GB, Cached max 440.7 GB
"memory admission: item N waits" lines: 0
pool RssAnon (sharedGb) 17.6 GB; estimate per item 129 GB; per-item RSS-sum peak (pool definition): item 0 74.7, item 1 74.6, item 2 64.3
gate at every start (1 Hz series; estimate 129; reserve 40):
  item +s  MemFree  Cached-Shm  freeEff | own(others)  proj=freeEff-own-est | cg max-current  cg avail_gb-rule  | would a cgroup-aware gate admit: min(freeEff, avail_gb) - own - est
  0  +  4    331.2      383.4    427.0 |    0.0    298.1 (alone) |   118.4    121.7 |    -7.2
  1  + 44    260.0      389.8    357.4 |   82.7    145.8 |    46.2     54.4 |  -157.2
  2  + 85    234.4      380.2    329.4 |   81.4    119.1 |    21.5     24.9 |  -185.4
at the peak of memory.current (+99 s): current 150.0 GiB = anon 96.6 + file 52.8 + (shmem 43.1 inside both); MemFree 213.2 GB
sum of the three estimates 387 (pool GB = GiB) against memory.max 150 GiB; true anon+shmem peak 147.4 GiB
```

Reading: the gate counted 360.7 and 316.8 GiB free at the starts of items 1 and 2 (projected +146 and +106 against the threshold 40) while the cgroup had 109.0 and 63.7 GiB by the `avail_gb` rule (100.8 and 55.5 by `memory.max - memory.current`):
252 and 253 GiB more than the cgroup had in the 200 GiB job, 303 and 304 in the 150 GiB job. At no start did the gate count less than the cgroup had. In the 200 GiB run the cgroup stayed inside its limit (minimum headroom 18.4 GiB at +148 s) because the true figures are far
below the estimates; in the 150 GiB run it did not.

What happened at the limit (818940; 1 s series, `t2-cg-events.py`, every 5th second plus every change of `memory.events`; seconds from the sampler start; items started at +4, +44 and +85 s):
```
  t(s)  current  anon  file  shmem  MemFree  Cached(GB)  max/oom/oom_kill  pressure_full_avg10
     0      5.8   0.0   5.6    0.0    345.8      395   0 0 0  0.00
     5     37.7  17.7  19.7   10.8    329.2      393   0 0 0  0.00
    10     58.9  33.0  25.5   24.3    306.2      399   0 0 0  0.00
    15     69.7  40.2  29.3   24.7    293.7      403   0 0 0  0.00
    21     67.5  42.6  24.7   24.7    293.8      398   0 0 0  0.00
    26     77.6  49.4  28.0   25.3    284.0      401   0 0 0  0.00
    31     83.0  49.4  33.4   25.3    279.1      407   0 0 0  0.00
    36     88.6  49.4  38.9   25.3    274.3      412   0 0 0  0.00
    41     99.8  48.6  50.9   34.4    264.4      424   0 0 0  0.00
    46    105.0  51.9  52.7   36.3    258.7      426   0 0 0  0.00
    51    123.9  68.9  54.5   36.3    240.6      428   0 0 0  0.00
    57    134.9  77.3  57.1   36.3    230.4      430   0 0 0  0.00
    62    137.5  78.6  58.4   36.3    228.3      432   0 0 0  0.29
    67    146.8  92.8  53.5   37.0    219.7      427   0 0 0  0.16
    72    148.2  89.4  58.2   48.4    217.8      432   1963 0 0  1.52
    77    145.6  79.4  65.6   55.8    219.1      439   1963 0 0  0.83
    82    132.6  69.5  62.5   57.0    231.1      436   3857 0 0  0.75
    88    139.2  76.0  62.4   53.2    224.3      436   3857 0 0  0.50
    93    136.2  81.8  53.8   42.0    226.2      427   3857 0 0  0.27
    98    150.0  95.4  54.0   43.0    212.7      427   4806 0 0  0.18
   103    137.6  85.5  51.4   47.8    227.2      425   5782 0 0  0.76
   108    142.8  88.7  53.5   53.5    219.1      427   18207 24 2  1.86
   114     70.0  36.3  33.5   33.5    292.3      407   18207 24 2  1.25
   119     70.5  40.9  29.4   29.4    292.8      403   18207 24 2  0.68
   124     88.6  55.5  32.8   30.4    275.7      406   18207 24 2  0.45
   129     90.2  58.0  31.8   29.4    273.0      405   18207 24 2  0.25
   134     87.1  52.2  34.6   29.3    276.0      408   18207 24 2  0.16
   139     67.9  32.8  34.9   29.3    297.4      408   18207 24 2  0.09
   145     67.2  31.9  35.1   29.3    298.1      408   18207 24 2  0.04
   150     59.1  23.3  35.5   29.3    306.4      409   18207 24 2  0.03
   155     36.1  17.7   6.2    0.0    329.8      380   18207 24 2  0.01
```
- `memory.events max` (the counter of reclaim at the limit) began to count at +68 s and had reached 7,065 at +107 s: the cgroup lived at its limit for about 40 s (anon+shmem 147.4 GiB at the peak) before the OOM killer acted at +108 s (oom 24, oom_kill 2; the two younger items were killed, exit 3 with 0 pairs; item 0 finished after 146 s).
- The cgroup's non-shmem file cache (`file - shmem`) was about 22 GiB at +60 s, 10 GiB at +99 s and 2 GiB at +107 s: the job's cache was squeezed to a few GiB and the limit was still reached; the cache bought only the tens of GiB the job had.
- The pressure counter (`full avg10`) stayed at or under 1.9 % through the episode, and the node's MemFree stayed at 213-345 GiB: the limit, not the node, bound the job.
- After the kills `memory.current` fell from 142.8 to 70.0 GiB within 6 s (+108 to +114 s).

## 2. Loss B: true per-item peaks (one item at a time)

Job 818938 (g052, exclusive, `--mem=700G`, one H100, `-c 27`, `bin/t2-hostcut5`, `--workers 1`, `--threads 8`, `SINGLET_POOL_MEMSAMPLE=2`, the 1 Hz sampler with item markers). The mix3 string has **18** sample names (the brief said 17):
one item of each, shallow to deep: pbmc2m pbmc10m pbmc1k_cite_v3 pbmc5p_v1 hgmm1k_v3 pbmc1k_v3 hgmm1k_v2 neuron1k_v3 d100s_3p nuclei2k_v2 d300_5p d300_3p hgmm5k_v3 d500s_3p d700s_3p d1000_3p d1000s_3p d2600_3p. All 18 exit 0; pool wall 5085 s;
`memory.events` all 0; `memory.peak` 522799845376 B (486.9 GiB, includes page cache). Node at start: MemFree 384 GB, Cached 351 GB; the job alone on the node (not disturbed).

Definitions: reads = pairs of `samples.tsv`; est = 61.8 + 0.2235 x; RSSsum = the pool's per-process definition (max over time of the sum of RssAnon+RssShmem of the item's processes, `memsamp.tsv`, 2 s);
TRUE = max of the job cgroup's anon+shmem (1 s) while the item runs, minus the idle baseline of **22.03 GiB** (median of the 5 s before the first start: holder, non-host server, pool). "level", "waited": the level (GiB above the baseline) at the "[engine_gpu] level" line and
at the "[allsite] async: waited" line (the second the line first showed in the item's log, so +-1 s), `t` in seconds after the item's start; "exit": at the item's end; g.lvl / g.wait = the largest later growth above the level at those two lines (GiB):
```
idle baseline before the first item (median anon+shmem of the 5 s before the start): 22.03 GiB; sampler rows 4939
it  sample           reads(M)    est  RSSsum    TRUE  ratio  peak@s | level (t s, GiB)   waited (t s, GiB)  exit (t s,GiB) |  g.lvl g.wait
0   pbmc2m                2.0   62.2    57.5    24.8   2.32       7 | 29, 21.5           31, 22.8           33, 22.3       |      1      0
1   pbmc10m              10.0   64.0    58.4    27.1   2.16       7 | 15, 25.2           19, 23.3           22, 23.1       |      0      1
2   pbmc1k_cite_v3       45.4   71.9    61.9    36.2   1.71      22 | 25, 28.8           35, 27.3           42, 23.4       |      2      0
3   pbmc5p_v1            60.0   75.2    36.8    36.9   1.00      20 | 24, 30.8           44, 29.7           55, 21.3       |      2      2
4   hgmm1k_v3            63.1   75.9    63.1    39.5   1.60      27 | 31, 34.7           47, 30.0           69, 21.7       |      2      2
5   pbmc1k_v3            66.6   76.7    63.6    41.6   1.53      26 | 29, 32.6           44, 29.0           53, 22.4       |      3      1
6   hgmm1k_v2            75.2   78.6    63.5    40.8   1.56      26 | 30, 34.4           47, 28.9           65, 21.8       |      2      0
7   neuron1k_v3          92.9   82.6    65.0    58.6   1.11      76 | 62, 41.7           98, 29.0           110, 22.4      |     17      1
8   d100s_3p            100.0   84.2    82.5    79.6   1.04     132 | 134, 73.2          339, 35.8          356, 21.3      |      4      1
9   nuclei2k_v2         118.7   88.3    80.3    82.9   0.97     180 | 155, 60.0          343, 28.4          355, 22.4      |     23      2
10  d300_5p             299.6  128.8    97.4    82.2   1.18      69 | 86, 59.5           126, 37.5          146, 21.8      |      6      3
11  d300_3p             300.2  128.9    75.9    74.6   1.02      47 | 58, 53.3           108, 42.8          124, 21.7      |     14      0
12  hgmm5k_v3           313.0  131.8   100.4    86.2   1.17      96 | 97, 67.1           160, 41.5          295, 24.1      |     16      2
13  d500s_3p            500.5  173.7   146.2   146.8   1.00     192 | 213, 89.1          379, 70.6          452, 21.0      |     18      4
14  d700s_3p            703.4  219.0   149.6   149.3   1.00     130 | 154, 96.6          384, 70.3          475, 22.0      |     16     10
15  d1000_3p            999.8  285.3   210.4   207.1   1.02     214 | 263, 136.1         536, 86.3          1079, 21.9     |      8     20
16  d1000s_3p          1004.6  286.3   210.4   209.7   1.00     208 | 255, 134.6         537, 93.6          692, 23.0      |     15      8
17  d2600_3p           1303.7  353.2   325.6   250.5   1.30     331 | 333, 170.5         520, 65.5          649, 22.0      |     46      7
fit TRUE    n=18: = 35.50 + 0.1709 x reads(M); rmse 12.8 GiB; worst under-prediction 27.1 GiB (nuclei2k_v2, 118.7 M reads, measured 82.9); margin rule (5 up) 30
fit RSS-sum n=18: = 47.59 + 0.1802 x reads(M); rmse 16.6 GiB; worst under-prediction 43.0 GiB (d2600_3p, 1303.7 M reads, measured 325.6); margin rule (5 up) 45
true peak minus the pool estimate (61.8 + 0.2235 x): pbmc2m -37, pbmc10m -37, pbmc1k_cite_v3 -36, pbmc5p_v1 -38, hgmm1k_v3 -36, pbmc1k_v3 -35, hgmm1k_v2 -38, neuron1k_v3 -24, d100s_3p -5, nuclei2k_v2 -5, d300_5p -47, d300_3p -54, hgmm5k_v3 -46, d500s_3p -27, d700s_3p -70, d1000_3p -78, d1000s_3p -77, d2600_3p -103
max over items of true / estimate: 0.95 (d100s_3p)
```
Reading:
- The levels at the item's end are 21.0-24.1 GiB (baseline 22.03), so the baseline did not drift and the TRUE figures are not inflated by memory left behind by the previous item.
- RSSsum / true by depth: 2.32, 2.16, 1.71, 1.60, 1.53, 1.56 for the six smallest samples other than pbmc5p_v1 (2-75 M reads; pbmc5p_v1 has 1.00 with an RSS-sum of only 36.8), 1.11 neuron1k_v3, 1.04 d100s_3p, 0.97 nuclei2k_v2, 1.18 / 1.02 / 1.17 for d300_5p / d300_3p / hgmm5k_v3, 1.00 / 1.00 / 1.02 / 1.00 for d500s / d700s / d1000 / d1000s, **1.30 for d2600_3p**.
  So the double count is an effect of the shallow samples and of d2600_3p (there the 67.6 GiB E2E region is mapped by child and parent); between 100 M and 1,000 M reads the RSS-sum equals the true peak.
- The estimate is above the true peak for every class (true minus estimate: -5 d100s_3p and nuclei2k_v2, -24 neuron1k_v3, -27 d500s_3p, -35 to -38 for the six shallow samples, -46 to -54 for the d300 class and hgmm5k_v3, -70 d700s_3p, -77 and -78 d1000, -103 d2600_3p). The largest true / estimate is 0.95 (d100s_3p, 79.6 against 84.2).
- The peak sits at or just before the level line for the deep samples (d2600_3p: peak at +331 s, level line at +333 s at 170.5 GiB, so the item has already given back 80 GiB); after the level line it grows by at most 46 GiB (d2600_3p), 23 (nuclei2k_v2), 18 (d500s_3p), 17 (neuron1k_v3), 16 (hgmm5k_v3, d700s_3p), 15 (d1000s_3p), 14 (d300_3p), at most 8 elsewhere; after the "waited" line by at most 20 GiB (d1000_3p), 10 (d700s_3p), 8 (d1000s_3p), at most 7 elsewhere.
- The level at the "waited" line (the pool's plateau, `SINGLET_POOL_TAIL_GB` 80): 65.5 d2600_3p, 70.3 d700s_3p, 70.6 d500s_3p, **86.3 d1000_3p, 93.6 d1000s_3p**, 23-43 for everything below 500 M reads. Two classes are above 80.
- Fit (the procedure of `t2-memfit.py`, all 18 items, true peak against reads in M): **35.50 + 0.1709 x, rmse 12.8 GiB, worst under-prediction 27.1 GiB (nuclei2k_v2, 118.7 M reads, measured 82.9), margin rule 30**. The same fit of this run's RSS-sums: 47.59 + 0.1802 x, rmse 16.6, worst under-prediction 43.0 (d2600_3p, measured 325.6), margin 45.
  This run's RSS-sum of d2600_3p (325.6) lies inside the 322-341 range of the mix runs of hostmem round 3, so single-item and mix figures agree.

## 3. Loss C and the decomposition of the mix waits (replay of 818763 and 818882)

Data: `results/rt-node-818763-t2h3-A1` and `rt-node-818882-t2h3-A1b` (pool logs, `memsamp.tsv`, `mem2.tsv`, `mon.tsv`; no new mix run). `t2-admit-replay.py` re-evaluates, once a second inside each wait window of the head item, the inequality
`freeEff - ownRem - othRem - est - othWait >= 40` together with the two other things that make a head wait (a free worker slot, the 40 s stagger after an item of 100 or more), with the in-flight sets, the other pool and the free memory of the real run.

Inputs it reconstructs. (1) MemFree: `mem2.tsv`, 10 s, linearly interpolated. (2) **Cached - Shmem is in no sample of these runs** (the harness did not record it). It is modelled as MemTotal - MemFree - (cgroup anon+shmem) - K, with K the median over the logged lines of what is left when the logged
`free + cache share` is solved for Cached - Shmem (K = 17.1 GiB for 818882, 11.3 for 818763); residual of the model at the logged lines, model minus logged: mean -1.4 GiB, absolute maximum 14.7 (818882); mean -1.1, maximum 24.8 (818763). The "cacheLg" column below is the logged-line figure, "cacheM" the model.
(3) ownRem / othRem: the estimate (61.8 + 0.2235 x of the item's pairs), lowered at the logged "phase drop" lines, minus held (RSS-sum of the item minus the pool's own RssAnon, `memsamp.tsv`). At the first moment of the waits it reproduces the logged `own` and `other pools` figures to within a few GiB; the exceptions are three waits
where the replay's own figure is larger (818882 pool0 items 12, 27 and 50: 88 against 15, 197 against 113, 139 against 75; probable cause: an item started in the same second and counted by the replay). (4) Variants: `cache1` = cache share 1.0; `true` = the head is charged the true peak of its class (section 2) and every item in flight (maximum of its true profile from now on) minus (its true level now), the point on the item's solo profile being found from its start, its first phase drop (the level line) and its end by piecewise-linear
mapping; `both` = both.

**What it does not do:** it does not simulate knock-on effects. The items that started ahead of the head, the other pool, the free memory and the in-flight sets are those of the real run; only the head's wait is re-timed. A counterfactual start earlier than the real one would have changed everything after it (more items in flight, other
waits, memory). The replayed baseline agrees with the real wait within 15 s in 13 of 22 (818882) and 15 of 26 (818763) waits; the others differ by minutes (818882 pool1 item 1: real 1,800 s bounded start, replayed 1,114 s; pool1 item 9: 1,212 real / 299 replayed; 818763 pool1 item 42: 1,576 real / 457 replayed). That is the model's
error (10 s MemFree, the cache model, the 1 s state of the pool), so differences between variants of a few hundred seconds per wait are not resolved. The "saved" figures below are against the replayed baseline.

Cache while items waited (GiB; from the logged `free + cache share`, MemFree and the cgroup samples):

| | 818882 (A1b, 22 waits) | 818763 (A1, 26 waits) |
|---|---|---|
| node file cache (Cached - Shmem) | 328-545, median 436 (44-73 % of MemTotal 750) | 341-697, median 431 (45-93 %) |
| the job's cgroup (`memory.current - anon - shmem`) | 76-200, median 97 | 77-216, median 104 |
| not the job's (node minus cgroup) | 252-397, median 336 | 257-481, median 315 |
| cgroup anon+shmem / MemFree at those seconds | 72-240, median 184 / 56-219, median 114 | 72-238, median 177 / 12-198, median 131 |
| node at job start | MemFree 349, Cached 386, Shmem 0 GB | MemFree 288, Cached 448, Shmem 0 GB |

Whose: the 252-481 GiB that is not the job's was in the node's cache before the job started (386 and 448 GB at the start, earlier tenants' files) and was not given up during the run. The job's own 76-216 GiB is the harness's copy of the 18 inputs to node-local disk
(144 GB, 41 s; the pool drops the clean pages of the inputs at its start and after decode), the items' outputs written to `/tmp` until the harness hashes and removes them (per item 10-49 GB: d2600_3p 49.0, d1000_3p 20.3 GB in 818882), and files the items read. **The samples cannot separate inputs from outputs** (no per-file accounting); 144 GB and the outputs above are upper bounds.
The first wait of both runs (the d2600_3p head at pool start) saw a node cache of 544 (818882, model; logged-line figure 545) and 637 (818763, model; logged-line figure 697) GiB: 73 % and 85-93 % of the node's memory was page cache before any item had grown, with `free + cache share` 253 and 204 GB.

Replay, 818882 (columns: pool, item, sample, estimate | logged free, own, other pools, projected | MemFree, cgroup anon+shmem, memory.current at the line | cacheLg, cacheM | real wait s, replayed baseline, cache1, true, both):
```
run A1b: 22 waits (pool0 13, pool1 9); MemTotal 750 GB; cache model K = 17.1 GB (median; spread -41.5..72.5 over the lines)
model residual of free + cache share at the logged lines (model - logged), GB: mean -1.4, abs max 14.7
pool item  sample           est   free    own    oth   proj  MemF  cgAS cgCur | cacheLg cacheM | actual    base  cache1 | true  both
0    0     d2600_3p         353    253      0    242   -343   117    72   272 |     545    544 |   1259    1259       0    1155      0
1    1     d700s_3p         219    191    216      0   -244    56   130   330 |     540    547 |   1800    1114       0     208      0
0    1     d1000s_3p        286    250    263      0   -518   141   149   248 |     436    443 |   1140    1140     341    1140    296
1    6     hgmm1k_v2         79    199    146     59    -85    80   181   308 |     474    471 |    124     129       0       0      0
1    9     d300_3p          129    238     64     94    -48   121   184   264 |     468    428 |   1212     299       0      10      0
0    2     d500s_3p         174    268    218      0   -123   159   127   230 |     436    447 |    202     203       0     198      0
0    3     d300_5p          129    178    143     31   -124    63   231   325 |     461    439 |    510     503       0     166      0
1    11    d700s_3p         219    221     76     72   -146   108   206   286 |     453    419 |   1061    1061       0       0      0
0    9     d300_3p          129    216     68     75    -56   112   201   282 |     416    420 |    288     288       0       0      0
0    12    d500s_3p         174    198     15     68    -58    84   190   309 |     455    459 |    394     534      49      49     49
0    19    d500s_3p         174    202    101    112   -185    95   213   295 |     427    425 |   1014    1014       0      68      0
1    18    d700s_3p         219    225    147     31   -172   118   183   271 |     427    431 |   1113    1800       0     233      0
0    27    d700s_3p         219    252    113      0    -80   125   160   269 |     506    447 |    949     952     209     219    209
1    27    d500s_3p         174    206    146     47   -161    95   201   310 |     444    437 |    817     617       0     167      0
0    32    d1000_3p         285    239    162     31   -239   112   161   316 |     508    460 |   1263    1262       0     548      0
1    32    d700s_3p         219    192     95     81   -204   110   240   315 |     328    383 |   1320     853       0     313      0
0    43    d500s_3p         174    314    215      0   -293   219   130   224 |     380    384 |   1071    1207    1066    1207      0
1    42    d700s_3p         219    232    154     37   -178   136   217   303 |     384    380 |   1044    1046       0     224      0
0    45    d300_3p          129    205    103     76   -103   104   209   338 |     406    420 |    662      96       0       0      0
1    49    d700s_3p         219    270    146     35   -129   178   180   261 |     369    375 |   1124    1124       0     101      0
0    50    d500s_3p         174    246     75     66    -68   139   187   298 |     426    406 |    125     160     113     113    113
0    58    d700s_3p         219    292    144      0    -71   197   155   245 |     380    381 |    703     707       0       0      0
page cache at the logged wait lines (GiB; Cached - Shmem from the logged "free + cache share" and MemFree, 0.25 share): node total / job cgroup (memory.current - anon - shmem) / not the job
  node 328..545 (median 436); job 76..200 (median 97); not the job 252..397 (median 336)
  node cache as a share of MemTotal 44..73 %; cgroup anon+shmem 72..240 (median 184); MemFree 56..219 (median 114)
  node start: g052.clipper.gvsu.edu MemFree: 349 GB; Cached: 386 GB; Shmem: 0 GB;
TOTAL actual wait 19195 s; replayed baseline 17368 s; cache1 1778 s (saves 15590 vs replayed baseline)
       true 6119 s (saves 11249); both 667 s (saves 16701)
  variant base  : waits shorter than 10 s: 0 of 22; shorter than half the replayed baseline: 0; wait seconds 17368
  variant cache1: waits shorter than 10 s: 17 of 22; shorter than half the replayed baseline: 20; wait seconds 1778
  variant true  : waits shorter than 10 s: 5 of 22; shorter than half the replayed baseline: 17; wait seconds 6119
  variant both  : waits shorter than 10 s: 18 of 22; shorter than half the replayed baseline: 21; wait seconds 667
  logged lines only (no time evolution): projected + 0.75 x (Cached - Shmem) >= reserve in 20 of 22 lines (cache share 1.0 at the first moment)
  replayed baseline agrees with the real wait within 15 s in 13 of 22 waits (the rest differ by the model error of the 10 s MemFree / the second-by-second state)
```
Replay, 818763:
```
run A1: 26 waits (pool0 13, pool1 13); MemTotal 750 GB; cache model K = 11.3 GB (median; spread -87.9..61.1 over the lines)
model residual of free + cache share at the logged lines (model - logged), GB: mean -1.1, abs max 24.8
pool item  sample           est   free    own    oth   proj  MemF  cgAS cgCur | cacheLg cacheM | actual    base  cache1 | true  both
0    0     d2600_3p         353    204      0    246   -396    30    72   287 |     697    637 |   1800    1800       0    1150      0
1    1     d700s_3p         219    164    234      0   -289    12   112   311 |     607    615 |   1114    1114       0     971      0
1    2     d300_3p          129    247    147      0   -382   127   129   206 |     479    482 |    396     396      99     396      0
1    3     hgmm5k_v3        132    266     76      0   -294   146   112   189 |     481    480 |     84      84       0      84      0
1    9     d300_3p          129    262     75      0   -295   144   114   191 |     470    480 |    903    1800       0    1800      0
0    1     d1000s_3p        286    158    262     13   -404    32   182   301 |     505    525 |   1034    1035     167     948      0
1    11    d700s_3p         219    202    104     83   -203    98   238   333 |     416    403 |    941     132       0      14      0
0    9     d300_3p          129    283    218     57   -121   189   154   245 |     374    396 |    204     204       0       0      0
0    10    nuclei2k_v2       88    203    117    109   -111   118   230   314 |     341    390 |    104     104       0       0      0
0    12    d500s_3p         174    261    109     34    -56   162   183   269 |     394    394 |    946     331       0      47      0
1    16    d300_3p          129    243    152     49    -88   130   148   299 |     453    461 |    128     506       0       0      0
1    18    d700s_3p         219    261     56     51    -65   153   184   274 |     432    402 |    954    1800       0     163      0
0    14    d300_3p          129    222     77      0     16   107   196   327 |     460    436 |    491     491     253     253    253
0    19    d500s_3p         174    254     78    115   -113   150   178   282 |     416    411 |    935      96       0      13      0
1    26    d300_3p          129    184    189      0   -134    64   184   365 |     481    491 |    129     129       0       6      0
1    27    d500s_3p         174    253    137      0    -58   119   183   307 |     535    436 |   1013     626     101     101    101
0    27    d700s_3p         219    215     97     56   -157   112   190   313 |     412    437 |    970     971       0      41      0
1    32    d700s_3p         219    213     98     66   -170   105   202   325 |     433    432 |   1117    1125       0     162      0
0    32    d1000_3p         285    286    199      0   -198   179   144   254 |     430    416 |   1013    1013       0     377      0
1    42    d700s_3p         219    258    151     74   -186   157   177   274 |     405    405 |   1576     457       0       0      0
0    33    d500s_3p         174    287    215      0   -102   183   129   248 |     415    427 |   1195     524       0      10      0
1    49    d700s_3p         219    303    144      0   -233   198   137   234 |     418    403 |    978    1042       0    1042      0
0    43    d500s_3p         174    230    101     81   -126   127   200   305 |     413    412 |    980     980       0     167      0
1    58    d500s_3p         174    279    144     52    -90   165   161   264 |     455    412 |    956    1283       0       0      0
0    50    d500s_3p         174    233     95     51    -87   133   194   298 |     400    412 |    433     433       0     169      0
0    58    d700s_3p         219    257     93      0    -55   158   171   274 |     395    410 |    176     176       0       0      0
page cache at the logged wait lines (GiB; Cached - Shmem from the logged "free + cache share" and MemFree, 0.25 share): node total / job cgroup (memory.current - anon - shmem) / not the job
  node 341..697 (median 431); job 77..216 (median 104); not the job 257..481 (median 315)
  node cache as a share of MemTotal 45..93 %; cgroup anon+shmem 72..238 (median 177); MemFree 12..198 (median 131)
  node start: g052.clipper.gvsu.edu MemFree: 288 GB; Cached: 448 GB; Shmem: 0 GB;
TOTAL actual wait 20570 s; replayed baseline 18652 s; cache1 620 s (saves 18032 vs replayed baseline)
       true 7914 s (saves 10738); both 354 s (saves 18298)
  variant base  : waits shorter than 10 s: 0 of 26; shorter than half the replayed baseline: 0; wait seconds 18652
  variant cache1: waits shorter than 10 s: 22 of 26; shorter than half the replayed baseline: 25; wait seconds 620
  variant true  : waits shorter than 10 s: 7 of 26; shorter than half the replayed baseline: 18; wait seconds 7914
  variant both  : waits shorter than 10 s: 24 of 26; shorter than half the replayed baseline: 25; wait seconds 354
  logged lines only (no time evolution): projected + 0.75 x (Cached - Shmem) >= reserve in 24 of 26 lines (cache share 1.0 at the first moment)
  replayed baseline agrees with the real wait within 15 s in 15 of 26 waits (the rest differ by the model error of the 10 s MemFree / the second-by-second state)
```

Answers:
- (i) Page cache while items waited: 328-697 GiB of 750 (median 431-436), 44-93 % of the node; 76-216 GiB (median 97-104) of it in the job's cgroup, the rest (252-481, median 315-336) not the job's.
- (ii) Would the item have been admitted if the cache had counted fully as free (share 1.0)? **At the first moment of the wait: in 20 of 22 (818882) and 24 of 26 (818763) lines** (logged projected + 0.75 x logged (Cached - Shmem) >= 40); the exceptions are pool0 items 1 (d1000s_3p, projected -191) and 43 (d500s_3p, -8) in 818882, and pool1 item 2 (d300_3p, -23) and pool0 item 1 (d1000s_3p, -25) in 818763.
  Over the whole wait window the replayed wait is under 10 s in 17 of 22 and 22 of 26 waits. If every item in flight and the head were charged its true figure instead (cache share 0.25): under 10 s in 5 of 22 and 7 of 26 waits, and shorter than half the replayed baseline in 17 of 22 and 18 of 26.
- (iii) Wait seconds (sum of both pools):

| run | real | replayed baseline | cache1 | true | both | saved of the replayed baseline by cache1 / true / both |
|---|---|---|---|---|---|---|
| 818882 | 19,195 | 17,368 | 1,778 | 6,119 | 667 | 15,590 (90 %) / 11,249 (65 %) / 16,701 (96 %) |
| 818763 | 20,570 | 18,652 | 620 | 7,914 | 354 | 18,032 (97 %) / 10,738 (58 %) / 18,298 (98 %) |

The two changes overlap: both together leave 1,111 (818882) and 266 s (818763) fewer than cache1 alone and 5,452 and 7,560 s fewer than true figures alone. The wait seconds attributable to the cache share are therefore 90-97 % of the replayed wait, those attributable to the estimate (the "true" column; the estimate's excess over the true figure of section 2, not the RSS double count) 58-65 %.
Waits that stay long under true figures (cache share 0.25): d2600_3p at the first wait of each run (1,155 and 1,150 s), d1000s_3p (1,140 and 948 s), d500s_3p (818882 pool0 item 43, 1,207 s), d700s_3p (818763: 971 and 1,042 s) and d300_3p (818763 pool1 item 9, capped at the 1,800 s bound in the replay, as in its baseline): their true peaks do not fit beside the other pool's items at that node state.

## What a change to the gate would have to respect (measured facts only)

- **The job's limit can bind before the node does.** With `--mem=150G` and three d300_3p items the cgroup was at its limit from +68 s, stayed there for about 40 s while the reclaim counter rose to 7,065, and then killed two of three items (`oom_kill` 2). The job's own cache was down to 2-10 GiB by then. The node's MemFree was 213-345 GiB the whole time. Pressure (`full avg10`) never exceeded 1.9 %, so it gave no warning.
- **How far items went above their figures.** Solo, the true peak never exceeded the pool's estimate (largest true / estimate 0.95, d100s_3p: 79.6 against 84.2 GiB); in the mixes the RSS-sum peaks of hostmem round 3 were above the estimate only for d100s_3p (88.3 against 84). After the "[engine_gpu] level" line an item grew by up to 46 GiB (d2600_3p), 23 (nuclei2k_v2), 18 (d500s_3p), 17 (neuron1k_v3), 16 (hgmm5k_v3, d700s_3p), 15 (d1000s_3p); after the "[allsite] async: waited" line by up to 20 GiB (d1000_3p), 10 (d700s_3p), 8 (d1000s_3p).
- **The plateau constant is exceeded.** `SINGLET_POOL_TAIL_GB` is 80: the true level at the "waited" line is 86.3 (d1000_3p) and 93.6 (d1000s_3p) GiB above the baseline, 65.5-70.6 for d2600_3p / d700s_3p / d500s_3p, 23-43 below 500 M reads; growth after that line up to 20 GiB (d1000_3p).
- **True peak against reads** (one item at a time, this build): 35.50 + 0.1709 x reads(M), rmse 12.8 GiB, worst under-prediction 27.1 GiB at nuclei2k_v2 (118.7 M reads). Class values: 74.6-82.9 GiB for d100s_3p, nuclei2k_v2, d300_5p, d300_3p; 86.2 hgmm5k_v3; 146.8, 149.3, 207.1, 209.7 and 250.5 GiB for d500s, d700s, d1000, d1000s, d2600. The RSS-sum equals the true peak between 100 M and 1,000 M reads, is 1.30 times it for d2600_3p and 1.5-2.3 times for the pbmc/hgmm samples with 75 M reads or fewer.
- **Page cache.** Both mix runs started with 386 and 448 GB of cache that was not theirs and held 328-697 GiB of cache at the waits; MemFree reached 2-4 GB with memory pressure 33-89 % and 205,000-207,000 compaction stalls (hostmem round 3) while the gate admitted at a 25 % cache share, so reclaim was running and costing time in those runs. The replay says a cache share of 1.0 would not have made the gate wait; it does not say that the memory would have been there without those stalls, and no run with a higher share exists. The job's own cache is small and can be squeezed (section 1); the cache of others at the start is the large part and was not given up during a run.
- **The estimate is not the only unknown.** Charging true figures cuts the wait by 58-65 % of the replayed baseline; the d2600_3p head still waits 1,150-1,155 s in both runs because its true peak (250.5 GiB) does not fit beside the other pool's items, and several d700s / d1000s heads still wait 950-1,200 s.
- **Idle baseline of an exclusive job** (holder, server, pool, no item): 22.0 GiB anon+shmem; after an item ends the cgroup returns to 21.0-24.1 GiB.

## Jobs and data

| id | what | node | note |
|---|---|---|---|
| 818937 | loss A, `--mem=200G`, 3 x d300_3p | g051 | oom_kill 0, `memory.peak` 195254714368 |
| 818940 | loss A supplementary, `--mem=150G` | g051 | oom_kill 2, max 18207, oom 24 |
| 818938 | loss B, exclusive, 18 items one at a time | g052 | all exit 0, `memory.peak` 522799845376 |
| 818763, 818882 | mixes replayed in section 3 | g052 | as in T2-hostmem.md |

Results: `$W/jobs/gpu-runtime/results/t2-pooladmit-818937-cg`, `...-818938-true`, `...-818940-cg150` (`cg1hz.tsv`, `markers.tsv`, `pool0/`). No disturbed run, no repeat. No job of other agents was touched.
