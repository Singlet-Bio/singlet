# T2 fallzone: samples too big for the records' phase and too small for read chunks (fallzone)

Task `fallzone` of throughput track 2 (branch `t2/fallzone`, from `gpu/throughput-2` d141971c). Analysis only: no file under `src/` or `include/` changed. Sizes are GiB (the engine prints "GB" for 2^30 bytes; quoted lines keep its label).
Estimates are marked "estimate"; everything else is a log line, a count, or a measurement. Build in every job: `bin/t2-hostcut5`, default switches. No file under any `private/` directory and no HLA file was opened (the hash set below leaves `private/` out).

Scripts (`bench/gpu-runtime/`): `t2-fallzone-census.py` (per sample of named campaign directories: provenance fields, the item's own `out/star_Log.out` engine lines), `t2-fallzone-depth.py` (depth histogram of the corpus table), `jobs/t2-fallzone-item.sbatch` (one pool, items in sequence, per-item peaks and hashes).
Data: `bench/gpu-runtime/data/t2-fallzone-census-live.tsv`, `...-dryrun2.tsv` (one row per run, accession ids only), `data/t2-fallzone-results/*.results.txt` (the four jobs), Clipper copies under `$W/jobs/gpu-runtime/results/t2-fallzone-*`.

## Summary

| question | answer |
|---|---|
| Is the zone a real-data problem? | **Not in the data that exists.** 22 distinct real samples of 70-160 M pairs have an engine level on record (the item's STAR log). Five fell back on the E17 records budget, **all at budgets lower than today's H100 setting**: 4 on the RTX 8000 at 25.9 (dry run 2, holder-on shape, since replaced by 42.3) and 1 on the V100S at 13.4 (pilot-r2a, old shape). Their records' need was 26.3-30.5 GiB (and 13.9 at 13.4), **every one below 31**. At the H100 setting (31, 2 slots) **0 of the 22 would fall back** (largest need 30.46); at the V100S setting (29.7) 2 of the 22 would (30.07, 30.46); at the RTX 8000 setting (42.3) none. The only two samples above 31 are the test classes d100s_3p (35.07) and nuclei2k_v2 (43.93), which have 62 M and 80 M distinct sequences from 100 M and 119 M reads (D / n 0.62, 0.67). In the live pilots (43 GPU runs of 13 samples) every run reached level 5 except one (pilot-r2a, V100S, budget 13.4). |
| How big is the zone? | Corpus table (94,740 eligible samples, 25.65 T pairs): **14,456 samples (15.3 %) lie in 70-160 M pairs, 1.65 T pairs (6.4 % of pairs)**. The 235-sample dry-run set: 39 samples (16.6 %), 4.44 G of 71.9 G pairs (6.2 %). Depth alone does not decide a fallback: the number of distinct sequences does (section 2). |
| What decides it | `recNeed` (section 2): about 440-610 bytes per distinct sequence (all-in, 7 logged points) plus a read term; compared with the budget `SINGLET_ENGINE_GPU_MEM_GB`. Read chunks (RC) are decided earlier, from the read count and read length only (`ctx + tables + slack + readsPhase(lean) > budget`: about 142 M reads at 90 bp, 93 M at 150 bp, H100 at 31; estimates from the formula, checked against the logs). Fall-back = no RC and `recNeed > budget`. The slot count and the card's free memory do not enter either test when the budget is set by the environment (the campaign sets it on the H100). |
| Cost, measured (4 jobs) | d100s_3p: level 0 **358.4 s**, level 5 (budget 48, 1 slot) **160.5 s**, read chunks forced (budget 31) **195.8-196.9 s**. nuclei2k_v2: level 0 **341.0 s**, level 5 **188.6 s**, read chunks forced **211.3 s**. Level 5 is 2.2x and 1.8x faster; forced read chunks recover 82 % and 85 % of that gain. **Output sets identical in all arms** (core set hash equal per sample; the files that differ are the run-dependent ones, see 3.3). |
| Premise check | Right about the cost per sample (about 2x, +150-200 s on these two). Wrong, or at least unproven, about frequency: the real-data evidence has no E17 fall-back at the H100 setting. A bound (estimate): 0 of 22 gives a 95 % upper bound of about 13 % of zone samples; at 5 % of 14,456 zone samples, about 720 samples x 175 s = 35 H100-hours, against about 3,560 H100-hours for the whole corpus at the planning rate of 2.0 M pairs/s (estimate; 1 % of the campaign). |
| V100S 0.53 / RTX 0.41 M pairs/s | **Fall-backs do not explain them.** The four pilot samples (527.6 M pairs) all reached level 5 on both cards (sched-live7v, sched-live9r). The time is in the pileup of GSM7305278 (87 M pairs, 241 s and 319 s of pileup, 0.23 and 0.16 M pairs/s) and in the exports of the 259 M sample; the engine's mapping (`star_seconds`) is 8-20 s per sample on both. |
| A different, larger loss | dry run 2 had **10 level-0 runs of 99 GPU runs that are not the budget fall-back: "E10 (level 5) not used: E8 does not run"** (7 of them 10x 5' samples, 2 microwell-seq, 1 dropseq). On the H100: 121-255 M pairs in 2,263-2,886 s (0.05-0.10 M pairs/s). Not traced beyond the condition list (section 1.4). None in the live pilots. |
| Options | (b) is the cheapest: the RC record-chunk build `e2eBuildRC` already builds the records in several passes within the budget and produces the same bytes; routing the non-RC path to it when `recNeed > budget` is a one-condition change plus a gate (estimate: small, untested). (c) exists in the code (SL admission, `SINGLET_ENGINE_GPU_ADMIT=1`) but is **inert in the campaign**: it needs `budDefault` (no `SINGLET_ENGINE_GPU_MEM_GB`), and the campaign sets the budget. Per-sample budget is not possible; per-pool (per wave) is. |

## 1. How common is it in real campaign data

### 1.1 What the runner keeps

`campaign_provenance.json` has `engine_log.found.fallbacks` (flags for Solo on the host and the like), `run.pairs`, `run.wall_s`, `engine_timing`, `pool.{workers,slots,gpu_mem_gb}`; `telemetry.fell_back_of` reads `run.log` for `FELL BACK` / `level 0 (requested` and the `solo_host` flag.
**In the live pilots `run.log` and `engine.gpu.log.gz` hold only the pool's "start item" line (181 bytes; `engine_log.scope` "sample-lines"), so `telemetry.py` cannot see a records-phase fall-back in these samples** (`found.fallbacks` is `[]` for all). The engine's lines are in each sample's own `out/star_Log.out` (kept beside the outputs: `E19: level N reached (requested 5)`, `engine_gpu: N reads, D distinct`, `E8 records not built on the device (E17 budget: ...)`, `E17 budget ... mode ...`, `P3-MEM-03 reads phase in K read chunks`). `t2-fallzone-census.py` reads those lines (regex hits only). The lane logs `$W/logs/camp-*` are 2-6 KB scheduler logs with no engine lines. `ROUTE.json` exists for 3 samples (`h100:run exit 3`, unrelated).
Campaigns read (named directories, `samples/<shard>/<key>` listed one level): sched-live, sched-live.attempt1, sched-live2, 3, 4, 5v, 6v, 7v, 8, 9r, pilot-r2a, dryrun2. `sched-live5` has no samples; `dryrun`, `dryrun-deep`, `dryrun-dup-a/b` were not read (dryrun2 is the later run of the same set).
The "rate" below is read pairs / the item's wall seconds as the runner recorded it (3 in flight on the H100 pools: queue and pileup overlap included).

### 1.2 Live pilots (levels from each item's STAR log)

Summary per lane kind (runs; level 5; level 0 by E17 budget; level 0 other):

| lane kind | runs | level 5 | level 0, E17 budget | level 0, other | read pairs of the runs in 70-160 M |
|---|---|---|---|---|---|
| H100 (budget 31, 2 slots; sched-live .. live8) | 17 | 17 | 0 | 0 | 3 runs (74, 87, 107 M), all level 5 |
| V100S, current shape (holder off, budget 29.7; live7v) | 4 | 4 | 0 | 0 | 3 runs, all level 5 |
| V100S, older shapes (holder on, budget 13.4; live, live2, live5v, live6v) | 12 | 12 | 0 | 0 | 8 runs, all level 5 |
| V100S, pilot-r2a (budget 13.4; the log's device is 31.7 GB, the route record has no device) | 6 | 5 | 1 (GSM8974227, 76.2 M, need 13.90) | 0 | 1 run, fell back |
| RTX 8000, current shape (holder off, budget 42.3; live9r) | 4 | 4 | 0 | 0 | 3 runs, all level 5 |

H100 (every run level 5):

| campaign | sample | read pairs (M) | distinct sequences (M) | level | fall-back reason | read chunks | budget (GiB) | item wall (s) | rate (M pairs/s) |
|---|---|---|---|---|---|---|---|---|---|
| sched-live | GSM6481641 | 336.9 | 171.3 | 5 | - | 15 | 31.0 | 456 | 0.739 |
| sched-live | GSM7518387 | 385.9 | 83.8 | 5 | - | 12 | 31.0 | 391 | 0.986 |
| sched-live | GSM9134876 | 614.2 | 142.4 | 5 | - | 19 | 31.0 | 571 | 1.076 |
| sched-live2 | GSM6481641 | 336.9 | 171.3 | 5 | - | 15 | 31.0 | 550 | 0.612 |
| sched-live2 | GSM7518387 | 385.9 | 83.8 | 5 | - | 12 | 31.0 | 484 | 0.797 |
| sched-live3 | GSM7518387 | 385.9 | 83.8 | 5 | - | 12 | 31.0 | 573 | 0.673 |
| sched-live3 | GSM9134876 | 614.2 | 142.4 | 5 | - | 19 | 31.0 | 642 | 0.957 |
| sched-live4 | GSM6481641 | 336.9 | 171.3 | 5 | - | 15 | 31.0 | 420 | 0.801 |
| sched-live4 | GSM7518387 | 385.9 | 83.8 | 5 | - | 12 | 31.0 | 430 | 0.897 |
| sched-live4 | GSM9134876 | 614.2 | 142.4 | 5 | - | 19 | 31.0 | 632 | 0.972 |
| sched-live8 | GSM6123263 | 74.3 | 33.1 | 5 | - | 0 | 31.0 | 75 | 0.992 |
| sched-live8 | GSM7305278 | 87.4 | 53.2 | 5 | - | 0 | 31.0 | 393 | 0.222 |
| sched-live8 | GSM8313333 | 106.7 | 33.8 | 5 | - | 0 | 31.0 | 287 | 0.372 |
| sched-live8 | GSM8075917 | 259.2 | 126.4 | 5 | - | 9 | 31.0 | 323 | 0.802 |
| sched-live8 | GSM6481641 | 336.9 | 171.3 | 5 | - | 15 | 31.0 | 534 | 0.631 |
| sched-live8 | GSM7518387 | 385.9 | 83.8 | 5 | - | 12 | 31.0 | 452 | 0.854 |
| sched-live8 | GSM9134876 | 614.2 | 142.4 | 5 | - | 19 | 31.0 | 585 | 1.049 |

V100S:

| campaign | sample | read pairs (M) | distinct sequences (M) | level | fall-back reason | read chunks | budget (GiB) | item wall (s) | rate (M pairs/s) |
|---|---|---|---|---|---|---|---|---|---|
| sched-live | GSM8313333 | 106.7 | 33.8 | 5 | - | 9 | 13.4 | 167 | 0.640 |
| sched-live | GSM8075917 | 259.2 | 126.4 | 5 | - | 21 | 13.4 | 527 | 0.492 |
| sched-live2 | GSM8313333 | 106.7 | 33.8 | 5 | - | 9 | 13.4 | 185 | 0.576 |
| sched-live2 | GSM8075917 | 259.2 | 126.4 | 5 | - | 21 | 13.4 | 521 | 0.498 |
| sched-live5v | GSM6123263 | 74.3 | 33.1 | 5 | - | 7 | 13.4 | 139 | 0.534 |
| sched-live5v | GSM7305278 | 87.4 | 53.2 | 5 | - | 11 | 13.4 | 638 | 0.137 |
| sched-live5v | GSM8313333 | 106.7 | 33.8 | 5 | - | 9 | 13.4 | 165 | 0.647 |
| sched-live5v | GSM8075917 | 259.2 | 126.4 | 5 | - | 21 | 13.4 | 538 | 0.482 |
| sched-live6v | GSM6123263 | 74.3 | 33.1 | 5 | - | 7 | 13.4 | 139 | 0.536 |
| sched-live6v | GSM7305278 | 87.4 | 53.2 | 5 | - | 11 | 13.4 | 640 | 0.136 |
| sched-live6v | GSM8313333 | 106.7 | 33.8 | 5 | - | 9 | 13.4 | 166 | 0.644 |
| sched-live6v | GSM8075917 | 259.2 | 126.4 | 5 | - | 21 | 13.4 | 537 | 0.482 |
| sched-live7v | GSM6123263 | 74.3 | 33.1 | 5 | - | 0 | 29.7 | 107 | 0.697 |
| sched-live7v | GSM7305278 | 87.4 | 53.2 | 5 | - | 0 | 29.7 | 384 | 0.227 |
| sched-live7v | GSM8313333 | 106.7 | 33.8 | 5 | - | 0 | 29.7 | 121 | 0.881 |
| sched-live7v | GSM8075917 | 259.2 | 126.4 | 5 | - | 9 | 29.7 | 373 | 0.694 |

V100S, pilot-r2a (device inferred from the log's "device 31.7 GB", mode and budget 13.4; samples of 31-76 M pairs):

| campaign | sample | read pairs (M) | distinct sequences (M) | level | fall-back reason | read chunks | budget (GiB) | item wall (s) | rate (M pairs/s) |
|---|---|---|---|---|---|---|---|---|---|
| pilot-r2a | GSM5223066 | 31.5 | 16.9 | 5 | - | 0 | 13.4 | 412 | 0.077 |
| pilot-r2a | GSM3576407 | 39.5 | 21.8 | 5 | - | 0 | 13.4 | 667 | 0.059 |
| pilot-r2a | GSM3930530 | 51.6 | 22.4 | 5 | - | 0 | 13.4 | 1426 | 0.036 |
| pilot-r2a | GSM5365964 | 51.8 | 15.5 | 5 | - | 0 | 13.4 | 810 | 0.064 |
| pilot-r2a | GSM8199637 | 55.8 | 11.6 | 5 | - | 0 | 13.4 | 921 | 0.061 |
| pilot-r2a | GSM8974227 | 76.2 | 26.5 | 0 | E17 budget | 0 | 13.4 | 1446 | 0.053 |

RTX 8000 (job 818686, sched-live9r):

| campaign | sample | read pairs (M) | distinct sequences (M) | level | fall-back reason | read chunks | budget (GiB) | item wall (s) | rate (M pairs/s) |
|---|---|---|---|---|---|---|---|---|---|
| sched-live9r | GSM6123263 | 74.3 | 33.1 | 5 | - | 0 | 42.3 | 142 | 0.522 |
| sched-live9r | GSM7305278 | 87.4 | 53.2 | 5 | - | 0 | 42.3 | 533 | 0.164 |
| sched-live9r | GSM8313333 | 106.7 | 33.8 | 5 | - | 0 | 42.3 | 172 | 0.622 |
| sched-live9r | GSM8075917 | 259.2 | 126.4 | 5 | - | 6 | 42.3 | 428 | 0.606 |

(sched-live2 sample GSM9134876 ran on `bigmem`/CPU lanes and has no engine level; the CPU-lane rows of the bigmem and CPU lanes are not in these tables. sched-live samples with no `run` record (SRS7752244, GSM5233208, GSM4829275) have no engine log.)

### 1.3 Dry run 2 (99 GPU runs; the 235-sample set, with the pool shapes of that run: V100S budget 13.4, RTX 8000 budget 25.9, H100 31)

| lane kind | GPU runs | level 5 (and 1 level 3) | level 0, E17 budget | level 0, "E8 does not run" |
|---|---|---|---|---|
| H100 | 33 | 29 | 0 | 4 (GSM6250953 121 M, GSM7658834 131 M, GSM9535813 254 M, GSM7232636 255 M) |
| V100S | 4 | 0 | 0 | 4 (9-69 M pairs) |
| RTX 8000 | 62 | 56 (GSM8810635, 318 M, reached level 3: not traced) | 4 | 2 (2.8 M and 4.4 M pairs) |

Rows in the zone (70-160 M pairs) and every row that did not reach level 5 (all other rows were level 5 with no fall-back; same columns):

H100:

| campaign | sample | read pairs (M) | distinct sequences (M) | level | fall-back reason | read chunks | budget (GiB) | item wall (s) | rate (M pairs/s) |
|---|---|---|---|---|---|---|---|---|---|
| dryrun2 | GSM6250953 | 121.4 | 36.2 | 0 | E8 does not run | 4 | 31.0 | 2301 | 0.053 |
| dryrun2 | GSM7658834 | 130.6 | 42.4 | 0 | E8 does not run | 0 | 31.0 | 2263 | 0.058 |
| dryrun2 | GSM9535813 | 253.6 | 75.7 | 0 | E8 does not run | 5 | 31.0 | 2641 | 0.096 |
| dryrun2 | GSM7232636 | 255.2 | 38.9 | 0 | E8 does not run | 5 | 31.0 | 2886 | 0.088 |

V100S:

| campaign | sample | read pairs (M) | distinct sequences (M) | level | fall-back reason | read chunks | budget (GiB) | item wall (s) | rate (M pairs/s) |
|---|---|---|---|---|---|---|---|---|---|
| dryrun2 | GSM7662904 | 9.3 | 4.6 | 0 | E8 does not run | 0 | 13.4 | 350 | 0.027 |
| dryrun2 | GSM5361253 | 15.3 | 11.9 | 0 | E8 does not run | 0 | 13.4 | 195 | 0.078 |
| dryrun2 | GSM5231122 | 16.3 | 11.1 | 0 | E8 does not run | 0 | 13.4 | 1107 | 0.015 |
| dryrun2 | GSM5682562 | 68.9 | 27.8 | 0 | E8 does not run | 4 | 13.4 | 2075 | 0.033 |

RTX 8000 (budget 25.9: four E17 fall-backs, all in the zone, 78-114 M reads, 52-66 M distinct sequences):

| campaign | sample | read pairs (M) | distinct sequences (M) | level | fall-back reason | read chunks | budget (GiB) | item wall (s) | rate (M pairs/s) |
|---|---|---|---|---|---|---|---|---|---|
| dryrun2 | GSM1241178 | 2.8 | 0.8 | 0 | E8 does not run | 0 | 25.9 | 40 | 0.071 |
| dryrun2 | SRS3702682 | 4.4 | 0.9 | 0 | E8 does not run | 0 | 25.9 | 87 | 0.050 |
| dryrun2 | GSM6647596 | 72.6 | 8.7 | 5 | - | 0 | 25.9 | 130 | 0.559 |
| dryrun2 | GSM6123263 | 74.3 | 33.1 | 5 | - | 0 | 25.9 | 175 | 0.425 |
| dryrun2 | GSM4658517 | 77.9 | 56.8 | 0 | E17 budget | 0 | 25.9 | 870 | 0.090 |
| dryrun2 | GSM3972019 | 82.1 | 57.0 | 5 | - | 0 | 25.9 | 318 | 0.259 |
| dryrun2 | GSM7305278 | 87.4 | 53.2 | 5 | - | 4 | 25.9 | 656 | 0.133 |
| dryrun2 | GSM6025613 | 92.8 | 65.1 | 0 | E17 budget | 0 | 25.9 | 458 | 0.203 |
| dryrun2 | GSM5268288 | 94.8 | 37.7 | 5 | - | 5 | 25.9 | 495 | 0.192 |
| dryrun2 | GSM7157281 | 100.5 | 52.1 | 0 | E17 budget | 0 | 25.9 | 434 | 0.232 |
| dryrun2 | GSM8313333 | 106.7 | 33.8 | 5 | - | 0 | 25.9 | 232 | 0.460 |
| dryrun2 | GSM8871313 | 107.1 | 32.7 | 5 | - | 0 | 25.9 | 210 | 0.510 |
| dryrun2 | GSM7634817 | 109.1 | 38.0 | 5 | - | 0 | 25.9 | 267 | 0.408 |
| dryrun2 | GSM9038586 | 109.1 | 30.6 | 5 | - | 0 | 25.9 | 309 | 0.354 |
| dryrun2 | GSM4771283 | 110.9 | 31.0 | 5 | - | 0 | 25.9 | 233 | 0.477 |
| dryrun2 | GSM8173895 | 113.9 | 66.1 | 0 | E17 budget | 0 | 25.9 | 499 | 0.228 |
| dryrun2 | GSM3891635 | 115.3 | 48.0 | 5 | - | 6 | 25.9 | 466 | 0.247 |
| dryrun2 | GSM7444994 | 115.8 | 43.7 | 5 | - | 0 | 25.9 | 262 | 0.441 |
| dryrun2 | GSM5467117 | 118.6 | 49.1 | 5 | - | 5 | 25.9 | 255 | 0.465 |
| dryrun2 | GSM5341102 | 128.4 | 48.0 | 5 | - | 5 | 25.9 | 326 | 0.394 |
| dryrun2 | GSM5534780 | 132.9 | 50.3 | 5 | - | 7 | 25.9 | 807 | 0.165 |

The five E17 fall-back lines (need in GiB, reads, distinct): GSM8974227 13.90 (76.2 M, 26.5 M; V100S pilot, budget 13.37); GSM4658517 30.07 (77.9 M, 56.8 M); GSM6025613 26.51 (92.8 M, 65.1 M); GSM7157281 26.33 (100.5 M, 52.1 M); GSM8173895 30.46 (113.9 M, 66.1 M) (RTX 8000, "more than the budget of 25.93 GB"). Together with the two test classes: need / D = 437-609 bytes per distinct sequence (563, 568, 437, 543, 495 for the five; 609 and 591 for d100s_3p and nuclei2k_v2).

### 1.4 The other level-0 cause: "E8 does not run"

Ten dry-run runs logged `E10 (level 5) not used: E8 does not run` (the line comes from `SeqCacheGpuPrefill.cpp:1617-1619`, one condition list: the shared-memory region, `SINGLET_ENGINE_GPU_HANDOFF`, `..._HANDOFF_MINQ`, `SINGLET_GPU_PILEUP=on`, hand-off summary, `seqCacheHandoffConfig`, a Solo CB_UMI_Simple type with CB and UMI up to 21 bases). Their provenance `engine_protocol`: 10xv2-5p (7 runs; assay label 10x_5p), microwell-seq (2), dropseq (1, labelled seqwell). `singlet.cpp:6866-6890` creates the region only on the unsorted stream path; a 5' library is restarted on the sorted path (`strand_early_abort`). **This is not traced to the exact clause**; it is reported because it is a larger loss than the E17 zone in this data: 4 H100 runs, 10,091 s for 761 M pairs (0.08 M pairs/s) against about 1 M pairs/s at level 5. The live pilots show none of it (no 5' sample among their level lines).

### 1.5 How many samples and how much of the data lie in 70-160 M pairs

Corpus (`data/corpus-beta/out/samples_pairs.csv.gz`, eligible and not mux-skipped, corrected depth; `t2-fallzone-depth.py`):

```
eligible samples (not mux-skipped) with a depth: 94740, 25.65 T pairs
bin_M_pairs  samples  share_samples_pct  pairs_G  share_pairs_pct  samples_bases_per_pair_ge_200
0-20          13505   14.25     100.6    0.39   2932
20-50         10133   10.70     342.5    1.34   3661
50-70          4538    4.79     267.7    1.04   1357
70-100         4921    5.19     417.5    1.63   1168
100-130        4919    5.19     565.1    2.20    952
130-160        4616    4.87     668.4    2.61    989
160-300       18849   19.90    4241.9   16.54   4091
300-600       24364   25.72   10242.5   39.94   8075
600-1400       7958    8.40    6486.5   25.29   2480
1400-inf        937    0.99    2314.9    9.03    191
zone 70-160 M: 14456 samples (15.26 %), 1650.9 G pairs (6.44 % of all pairs)
```
The last column counts samples with at least 200 bases per pair in the table's base count (both mates; about 119 for 3' v3 at 90 bp), a proxy for 150 bp reads, whose zone is narrower (RC from about 93 M reads, section 2).
The 235-sample dry-run set (`dryrun/out/dryrun_set.csv`, stratified, not the corpus distribution): 55 samples under 70 M (23.4 %, 1.41 G pairs), 10 in 70-100 M, 29 in 100-160 M (together 39 samples, 16.6 %, 4.44 G pairs, 6.2 % of 71.9 G), 141 at 160 M and above (60.0 % of samples, 91.9 % of pairs); 7 of the 39 have more than 100 bytes of FASTQ per pair (long reads).
**Depth alone does not decide a fall-back**: it needs distinct sequences D above about 55-76 M at the H100 setting (section 2). Real zone samples had D / n of 0.12-0.73 (median about 0.4), so most of the zone has D below that.

## 2. What decides it, from the code

All in `src/engine_gpu/engine_gpu.cu` (`mapDistinctBody`, line 1551 on) and `include/singlet/engine_gpu.h`; the call and the retries are in `src/star/SeqCacheGpuPrefill.cpp`.

1. **Budget** `cap = memBudget(total)` (`engine_gpu.h:429`): `SINGLET_ENGINE_GPU_MEM_GB` if set (GiB; 0 = none), else `(total - 2 GiB - sharedIndexReserve) / SINGLET_ENGINE_GPU_SLOTS`. `budDefault` is true only when the variable is unset. `ctx` = this process's context, at most 1 GiB of what the device holds at the start (`g_bud.reset(cap, min(used, 1 GiB))`).
2. **Reads phase and read chunks** (lines 1626-1653). With S = cDNA bases, n reads, nw = ceil(longest / 64):
   `RD = S + 8n`, `AUX = (2+nw) 8 n`, `ENT = 24 n`, `QM = n qmBytes`, `KEEP = 16 n`, `CLIP = n (28 or 29) + S` when the CellRanger4 clip is on, else 0;
   `readsPhase(lean) = max(RD + 28n + AUX + ENT + QM (+KEEP unless lean), RD + ek + CLIP, RD + ek + 53 n, ...)` with `ek = 0` when lean.
   `needNormal = ctx + TAB + SLACK + IDX + max(readsPhase(false), ...)`; **`lean0 = needNormal > cap`**; **`RC = RCHUNK set || (lean0 && ctx + TAB + SLACK + readsPhase(true) > cap)`**. IDX is 0 on the H100 (shared index). SLACK = 1 GiB, TAB about 0.04 GiB. Neither test uses D, the slot count or the card's free memory. `SINGLET_ENGINE_GPU_RCHUNK=<reads per chunk>` forces RC at any n (a test knob; the chunk size is then the user's; without it the chunk is `roomRC/4/perR`, with roomRC from the budget and the card's free memory).
   Thresholds for RC (estimate: usable = budget - ctx - TAB - SLACK; dominant term 2S + 37n bytes with clip on, S + 108 n without; per card in the table below; peakmodel 1.2 gives 144 M and 93 M from the same formula).
3. **Records' need** (`recNeed`, line 3385), evaluated in `e2eBuild` (line 3413) after the lanes and the host assembly of the records:
   `recNeed = used + 28 nRec + blob + seq + 101 nRec + seq/2 + [E10] (24 nRec + 4n + 256 MiB) + [E11] ((keepLate ? AUX + 16n : 0) + 4n + 8 nRec) + SLACK`, with nRec the records (about D), `blob` and `seq` the host records' byte counts (`seq` = 2 x the E8 line's "seq bytes"), `used` the tally (context, tables, E11's keep) and `keepLate = lean`. **Check against the logs:** d100s_3p, nRec 61,828,260, seq 8.6 GiB, blob 4.7 GiB, lean: sum 35.2 GiB against the logged 35.07; nuclei2k_v2, nRec 79,810,823, seq 6.8, blob 13.3: 44.0 against 43.93 (my arithmetic from the e16 marks; the formula as read reproduces both lines to 0.4 %).
   Per distinct sequence the all-in figure is 437-609 B (7 logged points). The n-terms are 8n (normal) or 56n (lean, keepLate adds AUX + 16n: 48n at 90 bp) bytes.
4. **Fall-back** (line 3413 and 3957): `cap && recNeed() > capRec` returns "E8 records not built on the device (E17 budget: ...)"; the caller turns E10/E11 off: level 0, STAR's loop does the per-read work and fills its sequence cache from the host. `capRec = cap` unless **SL admission** raised it: `if (cap && budDefault && !RC && admitOn() && records) { need = recNeed(); want = min(gpuSlots, ceil(need / cap)) }`, then `in.reacquire(want)` and `capRec = max(cap, min(cap x slots, used + free - 0.5 GiB))` (line 3926-3940). The RC records build `e2eBuildRC` (line 3619) has no `recNeed` test; it sizes its record chunks from the room.
5. **Which inputs matter:** budget (environment or total/slots): both tests. Slot count: only through a default budget, and through SL admission (needs `budDefault` and `SINGLET_ENGINE_GPU_ADMIT=1`). Free memory: not in either test (it sizes chunks and the SL admission's room). Read length: S in `readsPhase` (RC) and the bytes of `seq` and `blob` per record (fall-back). Distinct sequences: nRec, blob, seq (fall-back only). Reads n: RC and the n-terms.

Thresholds per card, with the campaign's settings (`config.py`: H100 budget 31.0 via `GPU_MEM_BUDGET_GB`, 2 slots, shared index; V100S and RTX 8000 set no budget, holder off, so the engine takes `total - 2` with 1 slot; the logs: 29.7 and 42.3). **Estimates** from the formulas above (RC) and from the 437-609 B range (fall-back); "falls back" needs both no RC and `D` above the fall-back threshold:

| card, setting | budget | RC from n of (90 bp, clip on / off; 150 bp, clip on) | fall-back threshold on D (437-609 B per sequence) | (reads, distinct) that falls back |
|---|---|---|---|---|
| H100, 31 GiB, 2 slots, shared index | 31.0 | 142 M / 156 M; 93 M | 55-76 M | D >= 55-76 M and n below the RC line: at 90 bp n of 55-142 M with D / n above 0.39-0.53 at the top of the range, 1.0 at the bottom; at 150 bp n of 55-93 M, D / n above 0.6-0.8 |
| V100S, holder off, 1 slot | 29.7 | 139 M / 153 M; 91 M | 52-73 M | the same shape, 3 M lower on every number |
| RTX 8000, holder off, 1 slot | 42.3 | 201 M / 221 M; 131 M | 75-104 M | D >= 75-104 M and n below 201 M (90 bp) / 131 M (150 bp) |
| (RTX 8000 in dry run 2, holder on) | 25.9 | 117 M / 128 M; 76 M | 46-64 M | four real samples fell here (table 1.3) |

Observed against these lines: all 5 real E17 fall-backs have n of 76-114 M and D of 26-66 M at budgets of 13.4 and 25.9; the 22 zone samples have D of 8.7-66.1 M; the two test classes (D 61.8 and 79.8 M) are the only ones above the H100 line. The RC counts agree (RTX at 25.9: 87 M reads, 4 chunks, a 150 bp library; H100 at 31: 121 M, 4 chunks).
**V100S 0.53 and RTX 0.41 M pairs/s against d300_3p's 1.25 and 1.17:** no fall-back is involved (section 1, the pilot rows of 7v and 9r are level 5). d300_3p is one 3' v3 sample with 59.5 M distinct reads and a light pileup; the pilot set is four samples with a 241-319 s pileup on the 87 M sample, and a 259 M sample whose export takes 113-117 s. That is the measured reason in the provenance timings; it is outside this task.

## 3. What it costs, measured

Four single-item jobs (the last one a repeat of one arm with a smaller chunk), one H100 each, not exclusive, node g051, `bin/t2-hostcut5`, defaults, `jobs/t2-fallzone-item.sbatch` (a copy of the peakmodel item job; one pool, one worker, the items in sequence; `SINGLET_E16_TRACE=1`; 1 Hz cgroup and `nvidia-smi` samplers). Same inputs as 818938 (`d100s_3p.1fq`, `nuclei2k_v2.1fq`, copied to node-local disk).

| job | tag | budget | slots | extra | items |
|---|---|---|---|---|---|
| 818970 | l0 | 31 | 2 | none (the campaign's H100 settings) | d100s_3p, nuclei2k_v2 |
| 818968 | l5 | 48 | 1 | none | d100s_3p, nuclei2k_v2 |
| 818969 | rc | 31 | 2 | `SINGLET_ENGINE_GPU_RCHUNK=33554432` | d100s_3p, nuclei2k_v2 |
| 818977 | rc16 | 31 | 2 | `SINGLET_ENGINE_GPU_RCHUNK=16777216` | d100s_3p |

### 3.1 Wall, level, peaks

| sample | arm | wall (s) | engine level | read chunks | card peak, nvidia-smi 1 s (GiB, whole card incl. the 18.0 shared index) | engine's "peak (lanes, records)" (GB in use on the card) | host peak, cgroup anon+shmem (GiB) |
|---|---|---|---|---|---|---|---|
| d100s_3p | level 0 (818970) | 358.4 | 0 (E17 need 35.07 > 31.00) | no | 66.9 | 47.32 | 101.4 |
| d100s_3p | level 5, budget 48 (818968) | **160.5** | 5 | no | 64.2 | 64.25 | 102.9 |
| d100s_3p | RC 32 Mi (818969) | 196.9 | 5 | 3 of 33,554,432 | 60.4 | 47.32 | 98.6 |
| d100s_3p | RC 16 Mi (818977) | 195.8 | 5 | 6 of 16,777,216 | 52.1 | 47.32 | 98.4 |
| nuclei2k_v2 | level 0 (818970) | 341.0 | 0 (need 43.93 > 31.00) | no | 47.2 | 47.16 | 104.7 |
| nuclei2k_v2 | level 5, budget 48 (818968) | **188.6** | 5 | no | 64.1 | 64.12 | 101.7 |
| nuclei2k_v2 | RC 32 Mi (818969) | 211.3 | 5 | 4 of 33,554,432 | 51.5 | 47.16 | 102.1 |
818938 (the finding): d100s_3p 356 s, nuclei2k_v2 355 s, both level 0. Speed-up of level 5 over level 0: 2.23x and 1.81x; forced RC: 1.82x / 1.61x (RC recovers 82 % / 85 % of the gain; RC costs +36 s and +23 s over the one-pass level 5, estimate of the cause: the extra record-chunk passes and the chunked dedup, `dedup 6.6 s` against 0.8 s in the log).
Exact lines (full text in `data/t2-fallzone-results/`):
```
818970 d100s_3p  engine_gpu: E8 records not built on the device (E17 budget: the records' phase needs about 35.07 GB of device memory, more than the budget of 31.00 GB (SINGLET_ENGINE_GPU_MEM_GB)); the hand-off file path is used
818970 d100s_3p  ENGINE_GPU: E19: level 0 reached (requested 5)
818970 nuclei2k  engine_gpu: E8 records not built on the device (E17 budget: the records' phase needs about 43.93 GB of device memory, more than the budget of 31.00 GB (SINGLET_ENGINE_GPU_MEM_GB)); the hand-off file path is used
818968 d100s_3p  engine_gpu: 100002652 reads, 61828260 distinct (22500044 with N), mapped on the GPU 61694886 ...; ENGINE_GPU: E19: level 5 reached (requested 5)
818968 nuclei2k  engine_gpu: 118742315 reads, 79810824 distinct (1641489 with N) ...; ENGINE_GPU: E19: level 5 reached (requested 5)
818969 d100s_3p  engine_gpu: P3-MEM-03 reads phase in 3 read chunks of 33554432: upload 19.73 s ...; E8 records on the device in 5 record chunks of 14680064 (P3-MEM-03) ...
818969 nuclei2k  engine_gpu: P3-MEM-03 reads phase in 4 read chunks of 33554432 ...; E8 records on the device in 6 record chunks of 13631488 (P3-MEM-03)
```
**What the peaks do and do not say.** The card peak is the whole card over the item's window: the shared index holder's 18.0 GiB, the item's engine phase, then its pileup (arena up to 28 GB) and the non-host server's context, so it is not the records' phase. At budget 48 the lanes are sized from the budget (room 36.4 GB; "tally peak 47.12 GB at laneB"), so the 64 GiB peak is the budget's lane size, not what the records need. **The records' phase's own device need was not measured in any arm** (the engine prints only the estimate when it refuses). The estimate of 35.07 / 43.93 reproduces from the formula to 0.4 % (section 2); whether it is conservative (it adds every array of the phase as if all were live at once) is not shown by these runs.
**The forced-RC arms exceeded the budget in the engine's own tally:** d100s_3p at 32 Mi reads per chunk: "tally peak 43.47 GB at read chunk ... OVER the budget 1 times, by up to 12.47 GB"; at 16 Mi: "tally peak 35.14 GB at read chunk ... OVER the budget 1 times, by up to 4.14 GB". nuclei2k_v2 at 32 Mi: "tally peak 30.02 GB at laneB ... within the budget". The chunk size was mine (the knob needs one); the engine's automatic choice (`roomRC/4/perR`, about 20 M reads for this 150 bp sample, estimate) was not exercised because it is made only when RC triggers on its own. Under a two-slot card plan (17.3 + 2 x 30.4 + 3.3 under 86) an overshoot of 4-12 GB would not fit, so these two RC arms show the time and the identity, not a safe memory figure.
The host peak is about the same in every arm (98-105 GiB cgroup anon+shmem, including the resident index and the non-host server; the cgroup floor of these jobs was 0.1 GiB at the first sample, so there is no "above baseline" figure here).

### 3.2 Identity of the output sets

The hash set leaves `private/` out. `core set hash` = md5 of the sorted list of md5s of `star_Solo.out/Gene/{raw,filtered}/{matrix.mtx,barcodes.tsv,features.tsv}`, `star_SJ.out.tab` and every `.pz`/`.pz2`/`.1pz` file at depth 3 or less outside `private/` (18 files).

| sample | level 0 (818970) | level 5 (818968) | RC 32 Mi (818969) | RC 16 Mi (818977) |
|---|---|---|---|---|
| d100s_3p | 2601bbe3170102c8c3b83764cce0b4e3 | 2601bbe3170102c8c3b83764cce0b4e3 | 2601bbe3170102c8c3b83764cce0b4e3 | 2601bbe3170102c8c3b83764cce0b4e3 |
| nuclei2k_v2 | 762a1a32a9e986aaf96bf20f58f0bf30 | 762a1a32a9e986aaf96bf20f58f0bf30 | 762a1a32a9e986aaf96bf20f58f0bf30 | not run |

**Level 0, level 5 and RC give the same core set for both samples.** The 818938 run kept no hash, so level 0 was re-run here (818970).
The listing of all 71 non-private files differs between arms only in: `ambient_contamination.json`, `cr4_probe/{cr4,plain}_Log{,.final}.out`, `pileup_stats.json`, `provenance.json`, `read_stats.tsv`, `star_Log.{out,final.out,progress.out,std.out}`, `summary.json` (14 files; level 5 against RC differ in these too, so they are run-dependent whatever the level) and, between level 0 and level 5 or RC, `nonhost_reads.fq`, `star_Unmapped.out.mate1`, `star_Unmapped.out.mate2` (the read order of the unmapped and non-host reads; for d100s_3p level 5 and RC match each other in these three). I did not diff the contents of `ambient_contamination.json`, `pileup_stats.json`, `read_stats.tsv`, `summary.json` (the scratch directory is removed at the end of the job); their being different between two level-5 arms is the evidence that they are run-dependent (timing fields), not a content check.

## 4. Options, from the code (nothing implemented)

Device plan used below: shared index 17.3 + 2 slots x 30.4 + 3.3 context under the 86 GiB bar on a 94 GiB card (`config.py`; measured 80.7 against a predicted 81.4 with three samples in flight).

**(a) Trigger read chunks from the records' need.** RC is decided at line 1653 from n, S, nw and the clip only; D, blob and seq do not exist yet (D after the dedup, blob and seq after the lanes and `hostAll`). Two ways: (a1) after `e2eBuild` refuses, run `mapDistinct` again with RC forced (the same pattern as `handoverAgain` in `SeqCacheGpuPrefill.cpp:1791`, which already re-runs it): it needs an `Input` field instead of the environment variable, and costs the first attempt again (the refusal came 109 s into d100s_3p's item in 818938; estimate: first attempt about 110 s + RC run about 196 s = about 300 s against 358 s at level 0, so little gain). (a2) decide after the dedup, when D and the distinct sequences' lengths are known (about 22 s into the run; blob per record can be bounded at 165 B/record, the largest seen) and switch to the chunked reads phase then: more work (the reads are on the device in the one-pass path; the chunked path expects its own state), saves the first attempt. Device memory: unchanged by construction if the chunk is sized by the budget (the engine's own estimate said "OVER" for my fixed chunk sizes, section 3.1, so the sizing needs a test on 150 bp samples). Identical outputs: RC's records "can differ in order (ties of 32-base keys), not in content" (code comment); the identity gate for RC samples is not in the current 8-sample gate (all below 119 M); the two samples here were identical.
**(b) Build the records in several passes within the budget.** Already coded: `e2eBuildRC` takes the host-assembled records (`out.seq`, `blob`, offsets) and builds them in record chunks of about a quarter of the room (two passes: counts, then build), with E10's products per chunk and E11's keep in parts. It has no `recNeed` test and is reached only when `RC` is true (line 3957: `RC ? e2eBuildRC : e2eBuild`). Candidate change: `(RC || (cap && recNeed() > capRec)) ? e2eBuildRC : e2eBuild`, in `mapDistinctBody` line 3957 (and the E20 `split` branch before it). What it needs to be checked: that `e2eBuildRC`'s inputs exist in the non-RC lean path (`out.readRecord` is filled by `fillReadRecord` when E10 is on; `auxCompact` is 0 there and the keep reads `in.e10.aux`; `recSrc`/`recRead`; `cr.hBlob` freed). Estimate of the work: a one-line change and a debugging pass, a build (15 min) and the identity gate. Device memory: record chunks are sized from `g_bud.room()` and the card's free memory, so the phase stays within the slot's budget by construction (the budget test is then a sizing rule, not a refusal); two other samples in flight see the same 30.4 plan. Outputs: same bytes by construction (the code builds each chunk and shifts offsets so that "the region holds the same bytes as the whole-sample build"); the identity above (level 5 = level 0 = RC) is for the whole RC path, not for this mix.
**(c) A larger budget while the item holds both slots.** Exists as SL step 2: `SINGLET_ENGINE_GPU_ADMIT=1` lets an item whose `recNeed` exceeds one slot take several slots at the E20 relock (`lockDeviceSlots`, `engine_gpu.h:590`; waits up to `SINGLET_ENGINE_GPU_ADMIT_WAIT`, 120 s, then keeps what it has) and raises `capRec` to `min(cap x slots, used + free - 0.5 GiB)`. **Inert in the campaign for two reasons:** the test needs `budDefault` (line 3926), and the campaign sets `SINGLET_ENGINE_GPU_MEM_GB=31.0` on the H100 pools (`worker.pool_env`); `ADMIT` is not set either. A change would drop `budDefault` from the test (or add a switch) and set `ADMIT=1` in `pool_env`. Device memory: the item holds both slots only for its records' phase, so the card holds the shared index 17.3 + the item up to 2 x 31 = 62 + the other samples' contexts and E11 keeps, as the plan already allows (17.3 + 60.8 + 3.3 = 81.4 under 86; the 35.07 and 43.93 GiB needs fit). Identical outputs: yes, the same `e2eBuild` code with a larger cap. Costs: other samples' engine phases wait for the slots while it holds them (up to 120 s waiting for the second slot, estimate of its own cost), and a need above 62 would still fall back.
**(d) Other.**
 1. `recNeed` is a sum of every array of the phase; if the real peak is lower, a tighter formula (for example not counting `eBuild` and `eIn` together, or E11's `AUX` that is freed) would let the same samples through at 31. Not measured here (section 3.1); the check is one run at budgets 35-36 for d100s_3p, or the engine's own tally printed for the records' phase.
 2. The same RC machinery and `SINGLET_ENGINE_GPU_RCHUNK` give the zone samples 82-85 % of the level-5 gain today with no code change, but with a hand-chosen chunk and an engine tally above the budget on 150 bp reads (3.1): not safe as a campaign setting.
 3. The refusal comes after the lanes (about 110 s of 358). An earlier check at the dedup, when D is known, would save the lanes' work only in option (a2).
 4. `telemetry.py` cannot see a records-phase fall-back in the live pilots (their `run.log` holds one line); the cost model's `p_fb` would be fitted on nothing. A line from the item's `star_Log.out` (`E19: level N reached`) into `engine_log.found` would feed it.

### Campaign side: per-sample budget

Not possible today. `worker.py` computes one `mem_gb` per pool launch (`gpu_mem_budget`: `max(settings.gpu_mem_gb, index + 2)`), checks it with `admit_slots` (predicted peak = shared 17.3 + slots x (budget - 0.6) + 3.3 against the 86 GiB bar) and `pool_env` exports `SINGLET_ENGINE_GPU_MEM_GB` once for the pool; `gpu_pool.cpp` gives every item the pool's environment and only per-item command-line arguments (`it.extra`). **Per pool (wave) it is possible:** a wave of samples of one depth tier could get a larger `mem_gb` with fewer slots: at 48 GiB, one slot predicts 17.3 + 47.4 + 3.3 = 68.0 (within 86), two slots 115 (refused by `admit_slots`, which then drops to one slot). The depth tier is known before the run, D is not, so such a wave would take the whole 70-160 M tier at one slot (which loses the two-slot overlap for all of them: estimate not measured).

## Not done, failed, or uncertain

- The campaign's own telemetry (`run.log`) could not be used for the level; the item's `star_Log.out` was read instead. Only the sched-live*/pilot-r2a/dryrun2 directories listed in 1.1 were read; the 61 `bigmem`/CPU runs of dry run 2 have no engine level and are not counted.
- 22 distinct real samples in the zone is a small and partly stratified sample (dry run 2 is a designed set); the "0 of 22 at 31" figure carries a wide interval.
- The records' phase device need was not measured (3.1). The forced-RC arms exceeded the engine's own budget tally (3.1); a run with the engine's automatic chunk was not possible without a code change.
- The cause of "E8 does not run" for 5'/other protocols was not traced (1.4).
- The all-files comparison of the 14 run-dependent files is by file name and md5 only.
- No implementation, no gate, no build in this task. Four jobs were used (818968, 818969, 818970, 818977).

## 5. Follow-up: the "E8 does not run" level-0 runs (10 of 99 GPU runs of dry run 2)

Analysis only again. Added scripts: `bench/gpu-runtime/t2-fallzone-e8params.py` (STAR parameter block and campaign arguments per GPU run), `t2-fallzone-assays.py` (corpus per assay label), `jobs/t2-fallzone-e8.sbatch` (one staged SRA run, two encodes, one pool). Data: `data/t2-fallzone-e8params-dryrun2.tsv`, `data/t2-fallzone-assays.txt`, `data/t2-fallzone-census-dryrun-other.tsv` (the earlier `dryrun` campaign and the duplicates), `data/t2-fallzone-results/t2-fallzone-818988-e8.results.txt`. One job was used (818988).

### 5.1 Which condition fails, per run

The test (`SeqCacheGpuPrefill.cpp:1617-1619`) needs: the E2E region (`e2eRegion().fd >= 0`), `SINGLET_ENGINE_GPU_HANDOFF`, `..._HANDOFF_MINQ`, `SINGLET_GPU_PILEUP=on`, the hand-off summary not off, `ReadAlign::seqCacheHandoffConfig(P)`, `(tr != nullptr || !quant.gene.yes)`, Solo type CB_UMI_Simple, CB and UMI length <= 21.
`seqCacheHandoffConfig` (`ReadAlign_seqCache.cpp:361`) needs: `soloType == CB_UMI_Simple`, one Solo feature and it is Gene, `readNmates == 1`, no BySJout, no genome transform (SAM, out, quant), `chimSegmentMin == 0`, no WASP, no transcriptome SAM, no GeneCounts, no GeneFull / GeneFull_Ex50pAS / GeneFull_ExonOverIntron, `gType != 101`.
The region and the hand-off variable exist only on the unsorted stream path: `singlet.cpp:6866-6890` unsets the hand-off variable when `stream_unsorted != 1`, and creates the region only inside the `stream_unsorted == 1` block. The stream path is refused (`singlet.cpp:6777-6800`) for: an existing BAM, not `.1fq` mode, `--parallel-pileup`, ATAC, Smart-seq/bulk, **`detected_soloType != CB_UMI_Simple`**, **no whitelist file (STAR then runs CB_samTagOut)**, `--tagged-bam`, `SINGLET_KEEP_STAR_BAM`; and it can be left later by **`restart_on_sorted_path`** (`singlet.cpp:3392`, called at 10040 after the first N reads when `strand_early_abort`, and at 10101 at the end of a pass), which strips the `star_*` files and re-executes the command with `--sorted-bam`.
The STAR parameter blocks of the ten runs (`out/star_Log.out`; `t2-fallzone-e8params.py`) show which applies. Group A is the only one with `soloType CB_UMI_Simple`:

| group | runs (lane, read pairs M) | label / preflight / engine protocol | STAR: soloType, CB, UMI, whitelist; BAM mode | condition that is false | cause |
|---|---|---|---|---|---|
| A: 10x 5' called 3' | GSM5682562 (V100S 68.9), GSM5231122 (V100S 16.3), GSM7662904 (V100S 9.3), GSM6250953 (H100 121.4), GSM7658834 (H100 130.6), GSM7232636 (H100 255.2), GSM9535813 (H100 253.6) | label `10x_5p`; preflight `chemistry_call` **10x-3p-v2** (six `medium` with flag `end_ambiguous:10x-3p-v2\|10x-5p-v2`, GSM9535813 `high` with no flag); `encode_args --protocol 10x-3p-v2`; engine protocol 10xv2-5p | CB_UMI_Simple, 16, 10, 737K-august-2016.txt; **BAM SortedByCoordinate** | region and hand-off variable (the run is on the sorted path); everything else true | the .1fq is encoded with the 3' orientation, the library is 5', the stream's end-of-pass strand check restarts the run on the sorted path (see 5.2) |
| B: microwell-seq | GSM1241178 (RTX 2.8), SRS3702682 (RTX 4.4) | corpus labels 10x_3p_v3 / 10x_3p_v2 (not microwell); engine protocol microwell-seq | **CB_samTagOut**, 18, 6, `/tmp/singlet_combwl_filtered_*.txt` (a concatenated whitelist); BAM SortedByCoordinate | `detected_soloType != CB_UMI_Simple` (`singlet.cpp:6784`), `seqCacheHandoffConfig` (soloType), the Solo-type line of 1618-1619 | complex barcode: the engine builds a concatenated barcode and runs CB_samTagOut (`singlet.cpp:5069-5520`, `7421-7502`), which the stream and E8 do not support |
| C: dropseq | GSM5361253 (V100S 15.3) | label seqwell; engine protocol dropseq | **CB_samTagOut**, 12, 8, whitelist **None**; BAM SortedByCoordinate | `whitelist_file == none` (`singlet.cpp:6788-6791`: "no barcode whitelist file (STAR runs CB_samTagOut)"), and the same Solo-type conditions | no barcode whitelist (barcode discovery) |

HLA is not a cause: all 99 GPU runs of dry run 2, including every level-5 one, passed `--hla-t1k`; E8's test does not read it. Chimeric is not a cause either: the campaign's `--chimeric` is the engine's second pass on the unmapped reads (`singlet.cpp:10561`), not STAR's `chimSegmentMin`, and the one 10x_5p run that reached level 5 had it. Length limits are not a cause (CB 12-18, UMI 6-10).
**Group A against the 5' runs that reach level 5.** In dry run 2 the eight `10x_5p`-labelled GPU runs split 7 / 1: the one at level 5 (GSM6504592, 516 M pairs, H100) had `chemistry_call` **10x-5p-v2**, `high`, `--protocol 10x-5p-v2`, `BAM Unsorted`. Sixteen further GPU runs with a 3'-label and engine protocol 10xv2-5p (no preflight record, no `--protocol`) are all level 5 on the unsorted stream: the engine's own encode-time detection calls them 5' (5.2). So what separates the seven from the others is the `--protocol 10x-3p-v2` the preflight hook passed on a 5' library; paired-end cDNA, clip, barcode mate and HLA are the same. The gate sample pbmc5p_v1 and the mix sample d300_5p are benchmark `.1fq` files whose encode command I did not read; they are consistent with the "no `--protocol`" case.
Evidence that group A is restarted, from the kept data: the final STAR command line has `--outSAMtype BAM SortedByCoordinate` and the campaign's `pipeline_args` do not contain `--sorted-bam`, no stream-exclusion reason of the list above applies (whitelist present, CB_UMI_Simple, `.1fq` mode), and item wall minus the engine's own wall is 252-1,072 s for the seven against a mean of 34 s (RTX 8000) and 107 s (H100) for the level-5 runs:

| run | lane | item wall (s) | engine wall (s) | difference (s) |
|---|---|---|---|---|
| GSM7662904 | V100S | 350 | 98 | 252 |
| GSM7232636 | H100 | 2886 | 2461 | 425 |
| GSM7658834 | H100 | 2263 | 1471 | 793 |
| GSM5231122 | V100S | 1107 | 298 | 809 |
| GSM9535813 | H100 | 2641 | 1670 | 971 |
| GSM5682562 | V100S | 2075 | 1016 | 1059 |
| GSM6250953 | H100 | 2301 | 1230 | 1072 |

The kept logs of dry run 2 hold no `[singlet] STREAM STRAND` line (run.log is one line), so the restart is shown by job 818988 (5.2), not by those runs' own logs.

### 5.2 Is it still so with today's settings and build

`config.py` today: `pipeline_args` = `--ref-base --pipeline --output-format 1pz --no-hla --engine gpu` (+ `--genome-shared off` on the private-index lanes) plus `sample_output_args(assay)` = `--chimeric` for `10x_5p` (VREGION_ASSAYS, D-133); HLA is off for every library (D-169). The protocol reaches the command line only through the `--hooks` class (`worker.py:917-921`, `dec.encode_args`): the preflight hook (`preflight/hook.py:63-71`) turns a high or medium `chemistry_call` into `--protocol <tag>`. **The live pilots ran with no hooks** (`settings.hooks` is empty in all 7 samples of sched-live8; their `protocol_decision.encode_args` is `[]`), and their labels are all `10x_3p_v3`: none of them exercises a 5' library, so the pilots neither show nor exclude group A.
Job 818988 (H100, g051, `bin/t2-hostcut5`, budget 31 / 2 slots, `--no-hla --chimeric`; the staged run SRR25434921 of GSM7662904, 1,989,132 pairs; the `.1fq` of the dry run no longer exists, the `.sra` does) settles the 5' cause:

| arm | encode's own line | level | BAM mode | pipeline lines | item wall (s) |
|---|---|---|---|---|---|
| `auto` (no `--protocol`, as without the preflight hook) | `[protocol_detect] 10x-3p-v2 -> 10x-5p-v2 (cDNA read looks 5': no 3'-kit TSO, poly(T) > poly(A))` | **5** ("E19: level 5 reached (requested 5)") | Unsorted | `5' protocol detected -> using reverse strand`; `BAM: unsorted STAR stream into the pileup` | 65.6 |
| `p3` (`--protocol 10x-3p-v2`, as the hook passed) | the same detect line, then `[1fq-encode] Protocol: 10x-3p-v2` (the flag wins) | **0** ("E10 (level 5) not used: E8 does not run"; "E19: level 0 reached (requested 5)") | SortedByCoordinate | **`STREAM STRAND: 92% of 687804 strand-informative nuclear reads are antisense to the protocol orientation (at the end of the pass); restarting on the sorted path (--sorted-bam)`**; later `Protocol relabeled by strand: id 2 -> 5` | 80.8 |

So: with the flag, today's code and arguments reproduce the level 0; without it, the engine's own detection labels the library 5' at encode and the run is level 5 on the stream. At 2 M pairs the restart costs 15 s (23 %); at 121-255 M it cost 425-1,072 s (table above). Whether today's campaign passes `--protocol` for 5' samples depends only on whether the preflight hook is on in the real run; this was not settled from the code (`campaign.py` default `--hooks` is empty). The other two causes do not depend on settings or build: B and C are decided by the protocol the engine detects (complex barcode, no whitelist) and by the conditions quoted in 5.1, which are the same in `d141971c` (the source read) as in the dry run's logs ("E8 does not run" and CB_samTagOut).

### 5.3 How much of the corpus is affected

Corpus table (eligible, not mux-skipped, 94,740 samples, 25.65 T pairs; `data/t2-fallzone-assays.txt`):

| class | assay labels | samples | share of samples | pairs (G) | share of pairs | basis |
|---|---|---|---|---|---|---|
| A, upper bound | `10x_5p` | 11,579 | 12.22 % | 3,582.7 | 13.97 % | the deciding property (the preflight call) is **not in the corpus table**: the class as a whole is the bound. Dry-run set: 22 `10x_5p` samples, 14 called 3p-v2, 3 called 5p-v2, 1 called 3p-v3, 1 arc-gex, 2 no call (single read, no barcode read); of the 8 on GPU lanes 7 were called 3p-v2. Applying 14 / 22 to the class would give about 7,400 samples and 2.3 T pairs (estimate from 22 samples; wide interval) **if the hook is on** |
| B and C, shown by a run | `microwell_seq`, `seqwell`, `dropseq` (in dry run a `seqwell`-labelled sample ran as the dropseq protocol; two `10x_3p_*`-labelled samples ran as microwell-seq, so labels are not reliable here) | 393 + 625 + 4,359 = 5,377 | 5.68 % | 24.1 + 53.6 + 553.0 = 630.7 | 2.46 % | one to three samples per label seen in the logs |
| B and C, same cause expected but not shown | `bd_rhapsody` 929, `singleron` 468, `hive` 27, `indrop` 414, `dnbelab_c4` 489, `pipseq` 161, `mobidrop` 10, `barcoded_unspecified` 2,681 | 5,179 | 5.47 % | 1,300.6 | 5.07 % | bound only: not traced for these protocols (complex or whitelist-less barcodes are the likely property; no run seen) |
| not affected by these causes | 10x 3' v1-v4, `10x_multiome`, visium and the other spatial labels | the rest | | | | `10x_unspecified` (8,154 samples, 2,056.9 G pairs) is a label "whose chemistry only the engine knows": its 5' share is unknown here |

### 5.4 What the planner does with them today

- **Nothing keeps such a sample off a GPU lane, and nothing learns it.** `costmodel.estimate` refuses a kind only for `disabled_assays` (no prior row sets any), `disabled_genomes`, the GPU depth limit, the depth caps and host memory (`costmodel.py:224-243`); `planner.py`, `placement.py`, `routing.py` have no fall-back or protocol rule.
- **Class key** (`class_keys`, `costmodel.py:88`): `assay_group|human or other|plain or mux|short or long`, with `assay_group` = 3p / 5p / other (`telemetry.assay_group`: `THREE_PRIME_ASSAYS`, `five_prime`, else other); the coarse level drops assay and genome. So `10x_5p` is a class of its own (`5p|human|plain|short`), group A is **not** separated inside it (the preflight call is not a feature), and microwell-seq, dropseq and every other label share `other|...`.
- **Fall-back learning:** `telemetry.fell_back_of` (`telemetry.py:167`) is true only if the sample's `run.log` / provenance engine log holds `FELL BACK` or `level 0 (requested`, or `found.fallbacks` has `solo_host`. The kept `run.log` is the pool's lines that name the sample (`logs.excerpt`, `logs.py:101-109`: engine lines do not name it, `LOG_NAMES` does not include `star_Log.out`). In my census **0 of the 99 dry-run-2 GPU runs and 0 of the 37 live GPU runs are marked** although 14 and 1 were level 0. So `p_fallback` stays at its prior (`p_fb` 0.05, `p_fb_nonhuman` 0.5 for the H100; `costmodel.py:45-59`) and the multiplier is the prior 6.5 (`MULT["fallback"]`, "26/4").
- **What it would do to the 5' class's wall fit:** the five dry-run-2 H100 records of `10x_5p` (4 level 0, 1 level 5) reach `MIN_N = 5`, so the class's own line would be fitted on them as ordinary walls: 10-19 s per M pairs for the four restarted runs against the H100 prior of 0.47 s per M pairs (a 22-40x gap, against the prior fall-back factor 6.5; estimate from the rates: 2301 s / 121.4 M = 19.0, 2263 / 130.6 = 17.3, 2641 / 253.6 = 10.4, 2886 / 255.2 = 11.3 s per M pairs).
- **GPU lane against CPU lane, measured, same samples** (the earlier `dryrun` campaign ran them on CPU lanes: 12 runs per node at `--threads 3`, `--hla` inline, build bc2d8d92; dry run 2: H100 3 in flight or V100S 1 in flight, build b7ff2406; so the lanes differ in shape, HLA and build, and each sample has one run per lane):

| sample | pairs (M) | GPU lane wall (s), level 0 | CPU lane wall (s) |
|---|---|---|---|
| GSM6250953 | 121.4 | 2301 (H100) | 3409 |
| GSM7658834 | 130.6 | 2263 (H100) | 2527 |
| GSM9535813 | 253.6 | 2641 (H100) | 4267 |
| GSM7232636 | 255.2 | 2886 (H100) | 3517 |
| GSM5682562 | 68.9 | 2075 (V100S) | 2090 |
| GSM5231122 | 16.3 | 1107 (V100S) | 1210 |
| GSM7662904 | 9.3 | 350 (V100S) | 356 |
| GSM1241178 (B) | 2.8 | 40 (RTX 8000) | 55 (bigmem) |
| SRS3702682 (B) | 4.4 | 87 (RTX 8000) | 126 (bigmem) |
| GSM5361253 (C) | 15.3 | 195 (V100S) | 148 (V100S, level 0, in `dryrun`) |

A level-0 run on a GPU lane is not slower per sample than the same sample on a CPU lane in these data (nine of ten faster or equal); keeping such samples off GPU lanes would not by itself buy time, the gain is in making them run at level 5.

### 5.5 Options (nothing implemented)

(i) **Static rule on what is known before the run.** For group A: label `10x_5p` with a preflight call of 3p-v2 (or flag `end_ambiguous:*`): do not pass `--protocol` (the `auto` arm above is level 5), or pass the 5' twin. Place: `preflight/hook.py:63-71` (`protocol`), a few lines; evidence: job 818988, 7 of 8 GPU runs. For B and C: label-based routing is possible through `costmodel.estimate`'s `disabled_assays` (an existing switch, `costmodel.py:224`) but the comparison above says a CPU lane is no faster for these samples, and the labels do not match the engine's protocol (two `10x_3p_*` labels ran as microwell-seq), so a label rule would miss them; the detected protocol is known only after the encode. Work: small for A, not worth it for B and C.
(ii) **Make E8 run.** A: nothing is missing in the engine for the 5' protocol (the `auto` arm); what is wasted is the whole first pass, because the early abort is off whenever the GPU pileup is on (`singlet.cpp:6897`: `strand_early_abort = ... && !getenv("SINGLET_GPU_PILEUP")`) and the only check left is the end of the pass (10101). Options: enable the early check in the GPU path (the restart already removes the hand-off files, 3400-3405), or run the strand probe before STAR (the CellRanger4 probe reads 200 k reads) and set the protocol before encode. Work: moderate, needs a gate with a 5' sample; not tested. B and C: the engine lacks E8 for CB_samTagOut: the stream takes CB from CR and needs a whitelist-exact CB_UMI_Simple (`singlet.cpp:6784-6791`), the hand-off summary is defined for CB_UMI_Simple (`seqCacheHandoffConfig`), and the Solo-type and CB-length test at 1618-1619 assumes the simple layout; the complex path concatenates segments before STAR (`singlet.cpp:5516-5520`, `7502`). Work: large (a stream and hand-off for concatenated barcodes and for barcode discovery), estimate only.
(iii) **Telemetry so the planner learns it.** Read the item's `out/star_Log.out` (it is kept): `E19: level N reached (requested M)` and `E10 (level 5) not used: <reason>` into `engine_log.found` (in `logs.parse_log` / `worker.py:1485-1490`, where `found` is built), and have `telemetry.fell_back_of` (`telemetry.py:167`) also test that level; add the strand-restart line of the item's pool log (it names no sample, so `logs.excerpt` drops it; the pool log plus the item index is the way to attribute it). With that the 5' class's `p_fb` and its fitted line would stop mixing restarted and normal runs; a per-sample feature (label 5' and preflight call 3p) would separate group A in the class key. Work: small (a parser and a field).

### 5.6 Not traced, uncertain

- The restart of the seven dry-run-2 runs is inferred (sorted command line, no other exclusion reason, the wall differences) and reproduced on one run of one sample (1.99 M pairs) in job 818988; their own logs do not hold the line.
- Whether the real campaign uses the preflight hook (and so `--protocol`) is not settled; the pilots ran without it and had no 5' sample.
- Groups B and C are settled from the code and the STAR parameter blocks; no run was made for them. The "same cause expected" labels of 5.3 are a bound, not a finding.
- The GPU/CPU comparison mixes lane shapes, builds and HLA settings (5.4) and has one run per lane per sample.
