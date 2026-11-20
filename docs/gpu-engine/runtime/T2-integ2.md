# T2 integ2: the merged head of `gpu/throughput-2` built and run as one binary

Branch `t2/integ2` from `gpu/throughput-2` head `2a779473` (recrc, chimdedup, cggate, chimprune merged; `engine_gpu.cu` merged by hand). No source change.

## Build
Job 819090 -> `bin/t2-integ2`. Patch `patches/t2-integ2.diff` = `git -c core.autocrlf=false diff b21de2cf HEAD -- src include CMakeLists.txt bench/gpu/m5/stitch.cuh tests/cpp`, 224,821 bytes, sha256 `cc1f55c955c4c6a45dbe483260fbf724469dd40072d8acab3892e89a0852aa66` (verified equal after the base64 transfer).
Build log: `100% tests passed out of 189`, `ctest exit 0`. Both `test_chimeric_dedup` and `test_gpu_pool_peak` are in the diff and were built.

## Identity gates (default settings, no `--chimeric`)
- Gate 819093 (REFD = `ref-t2`, GPU engine of the new binary vs baseline CPU references): 8 lines `VERDICT <sample> cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 91|92/5|6|7/4)`, `FALLBACK LINES: 1` per sample.
- Gate 819094 (fresh REFD `ref-t2-integ2`, CPU and GPU engine both of the new binary): 8 lines `VERDICT ... IDENTICAL (except E11's run-dependent files ...)`, same out_equiv counts; `FALLBACK LINES: 0` for the CPU run and `1` for the GPU run of each sample.

## CPU engine: refcmp
Method: `bench/gpu-runtime/jobs/t2-cpucache-refcmp.sbatch` (T2-cpucache), `NEW=ref-t2-integ2/T18`, job 819100, `out_equiv` of the new CPU-engine outputs against `ref-t2/T18`. Per sample: `RESULT: 4 differences`, `readhash identical`, `CellReads.stats identical (sorted rows)`, for all 8 samples. The 4 differing files are the same in all 8 samples: `pileup_stats.json`, `provenance.json` (byte counts `bytes_public`/`bytes_private` off by a few bytes), `read_stats.tsv`, `star_Log.std.out` (same byte sizes). File counts per sample: 91-92 identical, 5-7 differ only in timings/paths, 4 differ (same split as the gates' `out_equiv`).

## `--chimeric` items, one pool (jobs 819091 arm P, 819095 arm O)
Script `bench/gpu-runtime/jobs/t2-integ2-chim.sbatch`. Both on g051 (H100 NVL), `--mem=200G`, 3 workers, T=8, MEM_GB=31, SLOTS=2, `--no-hla`, items GSM6250953 (two SRA runs encoded on the node), d300_5p, pbmc5p_v1, each `--chimeric`. Arm P: `SINGLET_POOL_PEAKADMIT=1`; arm O: unset. Results: `results/t2-integ2-chim-819091-P`, `-819095-O`.

Table md5 (decompressed) and rows, both arms, all equal to the required values: pbmc5p_v1 `a5d71eb6a8b45889fe667bf57f17dce3` 24,141; d300_5p `1763c9dca0013d9bb3e68d7bfc419efe` 102,847; GSM6250953 `3573f1645c4fc6e920d37c70a0033df7` 59,632. All items exit 0.

| | arm P | arm O |
|---|---|---|
| pool wall | 1153 s | 721 s |
| item wall GSM6250953 / d300_5p / pbmc5p_v1 | 628.4 / 501.2 / 155.9 s | 710.7 / 597.2 / 238.1 s |
| second pass (from log epochs) gsm / d300 / pbmc | 415 / 337 / 92 s | 435 / 355 / 94 s |
| overlap | gsm and pbmc5p_v1 together, d300_5p after gsm ended (waited 628 s) | all three together |
| oom_kill | 0 | 0 |
| memory.peak | 153,206,243,328 B (142.7 GiB) | 189,226,565,632 B (176.2 GiB) |

Arm P pool lines: `item N: peak model:` for all three items said `--chimeric: floor 0.5 P + 22.1 + margin` (item 0 GSM: P 76.7, figure 88.7 GiB; item 1 d300_5p: P 59.9, figure 71.9; item 2 pbmc5p_v1: P 17.3, figure 42.8). Baseline B 42.60 GiB. Admission: item 1 waited (`figure 72 GiB; ... job term 156 GiB; ... can still allocate 89 GiB ... projected node 232, job -4; refused by the job term`), item 2 started ahead of it after 30 s, item 1 `admitted after 628 s (projected node 294 GiB, job 67 GiB)`; `1 waits, 628 s waited in all`. Peak figure lines: item 0 88.7 -> 52.0 (P 35.8, line 1: no fallback), then 52.0 unchanged twice (level line, waited line); item 2 42.8 unchanged (twice); item 1 71.9 (level line) -> 64.1 (waited line). Final line: `model exceeded in 0 s of the refreshes (largest excess 0.0 GiB)`; no `model exceeded:` line.
Arm O pool lines: no peak-model lines; `admission: the next item waits (... memory floor 0 GB, available 163 GB)` once; phase drops: item 2 75 -> 30 GB (holds 21), item 1 129 -> 52 (holds 85), item 0 89 -> 36 (holds 46); `0 waits`.

memsamp-cg.tsv (arm P, 2 s samples), during each item's second pass (AS = job anon+shmem, B = 42.6 GiB baseline; the sum of figures includes all items in flight, so item 2's window includes item 0's engine phase):
| item (window epochs) | samples | max AS-B | sum of figures at that sample | max(AS-B minus sum) |
|---|---|---|---|---|
| GSM6250953 (1795165326-1795165741) | 205 | 28.92 GiB | 52.04 | -23.12 |
| d300_5p (1795165905-1795166242) | 167 | 34.51 | 64.12 | -29.61 |
| pbmc5p_v1 (1795165209-1795165301) | 46 | 70.44 | 94.83 | -24.39 |
Whole run (555 samples): held memory never exceeded the sum of figures (largest value of AS-B minus the sum: -9.45 GiB). During the second passes the floor-based figures were above the measurement by 23.1 to 29.6 GiB at the sample where the margin was smallest; the largest gap (figures minus held) in a second pass window was 40.7 / 47.4 / 76.1 GiB.

## Not done / notes
No A/B/B/A repeat; one run per arm (timing differences between arms, e.g. pool wall, are not claimed as gains). No change to constants.
