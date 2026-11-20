# T2-chimshare: the chimeric pass takes the pool's resident genome

Facts only; estimates are marked. Switch `SINGLET_CHIM_SHARE_GENOME` (default on; `=0` restores the old behaviour: the preload is released at the first STAR fork and the pass reads the index files).
Binaries: build 1 `t2-chimshare` (job 819102, `ctest exit 0`, placeholder constant 19.5, used for the memory measurements), final `t2-chimshare-2` (job 819116, `ctest exit 0`, `chimSharedGb` 4.0), sha256 of the binaries in `bin/<name>/BUILD.txt`.

## Change
- `src/pipeline/singlet.cpp`: `chim_share_held` (4844). At the first STAR fork (8086-8091) the parent releases its preload as before, except when `--chimeric` is set, the switch is not `0` and `star_genome_preload_is_resident()` is true: then it keeps it. In the pass's `run_pass` (10690-10692) the parent prints `[singlet] chimeric: genome shared with the pool (SINGLET_CHIM_SHARE_GENOME)` before the fork and releases the preload right after it (`pp != 0`, also on a failed fork). After the whole chimeric block (10782) a still-held preload is released (pass skipped, "needs a FASTQ barcode run" case, or not run). If the dedup self-check refuses and the pass is run a second time, that second run no longer has the preload and reads the files (as before; not exercised).
- After the pass exits the item reads the pass's own `chim_Log.out` (the item deletes it later) and prints the lines `Genome preload: ...` and `Genome load seconds: ...` as `[singlet] chimeric: pass STAR log: <line>` (10702-10708). That is where "taken from singlet's preload ... shared copy-on-write" can be found: the item's stderr (the pool's `<i>.log`). With sharing off or outside a pool only the `Genome load seconds` line appears (no `Genome preload:` line when the process holds no preload).
- `src/star/STAR.cpp` (176-183): `Genome load seconds: x.xx` in `Log.out` (time around `genomeLoad()`; all STAR runs, log only).
- `src/star/GenomePreload.cpp:206`, `include/singlet/star/star_api.h`: `star_genome_preload_is_resident()` (this process's preload is a gpu-pool resident copy: `star_genome_preload_select` copies the state with `resident = true`; a preload read by the item itself has `resident = false`).
- Peak model: `include/singlet/gpu_pool_peak.h`: `chimSharedGb` (`SINGLET_POOL_PEAK_CHIM_SHARED_GB`, default 4.0), `chimShare` (false when `SINGLET_CHIM_SHARE_GENOME=0`, set in `fromEnv`), `chimEff()`; `phaseFigure` uses `chimEff()` instead of `chimGb`; `gpu_pool.cpp` 915 prints the effective figure. `tests/cpp/test_gpu_pool_peak.cpp`: the old chimeric checks use a copy with `chimShare=false` (19.5), new checks for the shared floor and the default.
- Scripts: `bench/gpu-runtime/jobs/t2-chimshare-mem.sbatch` (one item per pool, 1 Hz watcher of the pass), `t2-chimshare-pool.sbatch` (three-item pool, arms in sequence, optional standalone CPU run), `bench/gpu-runtime/t2-chimshare-win.sh` (rise within a pass window).

## Evidence asked for before building
1. The parent writes nothing to the arrays between the two forks. `g_genomePreload` is read or modified only in `src/star/GenomePreload.cpp`, `Genome_genomeLoad.cpp` (202, 273, 311) and `singlet.cpp` 8088/10692/10782 (grep over `src`); `star_main_impl` is called in the item process only inside forked children (singlet.cpp 5651, 8212, 8254, 8321, 9273, 9385, 10689; the other call sites, 1735 and 3726, are the `LoadAndExit` helper and a sub-command). The first child's `star_genome_preload_take` (`GenomePreload.cpp` 217-246) does `delete S; g_genomePreload = nullptr` on the child's copy of the state object and `Genome::freeMemory` (`Genome.cpp` 33-40) `delete[]`s the arrays in the child's address space; after `fork` these are separate address spaces, so the parent's pointer, state object and pages are untouched (the arrays are shared copy-on-write, a free in the child releases only its view).
2. The pass's genome parameters: `base_args` is the first run's `star_args.txt` (same `--genomeDir`, `--genomeLoad NoSharedMemory`); `--sjdbInsert` is not requested: `Parameters.cpp` 1004-1017 sets `sjdbInsert.pass1/pass2` only for `--sjdbFileChrStartEnd`, `--sjdbGTFfile` or two-pass mode, which neither run has, so `Genome_genomeLoad.cpp` 273 (release on insertion) and 202 (release on shared memory) are not reached. `take` (311) compares directory device+inode, Genome size, SA size, SAindex header/size and the files' mtimes (`GenomePreload.cpp` 217-241). If it refuses it logs `not used (<reason>), loading from the files`, releases the references and `genomeLoad` allocates its own arrays and reads the files (`Genome_genomeLoad.cpp` 313-317, 352-374), i.e. today's path. Measured: every pool run took it (`Genome preload: taken from singlet's preload (...)`, genome load 0.15 s).
3. Writes by the pass: it maps with `--outSAMtype None`, chimeric detection, no sjdb insertion; `genomeLoad` writes G/SA/SAindex only on the file-read path (`!genomePreloaded`, 352-374). The pass builds its 2-bit packed genome (`GENOME_PACKED`, `Genome_genomeLoad.cpp` 398) in new memory and calls `madvise(G1, ..., MADV_RANDOM)` (advice only). Any write to a shared page would be copy-on-write private to the child; the measured cgroup rise (below) is the sum of what it actually copied or allocated.

## Memory of the pass, one item at a time, pool of one worker (jobs 819103 = A, 819104 = B; build 1)
Rise = job cgroup anon+shmem, maximum from the pass's start (first sample, when `chim_Log.out` appears) to its maximum; 1 Hz.

| sample | pass threads | sharing on | sharing off | earlier private figure | genome load of the pass on / off |
|---|---|---|---|---|---|
| pbmc5p_v1 | 8 | 1.10 GiB | 16.46 GiB | 16.7 | 0.15 s / 14.73 s |
| d300_5p | 8 | 1.21 | 17.91 | 17.1 | 0.15 / 15.54 |
| GSM6250953 | 8 | 0.23 | 17.22 | 16.6 | 0.15 / 15.96 |
| GSM6250953 | 16 | 2.27 | not run | 18.0 | 0.15 / - |
| GSM6250953 | 27 | 3.61 | not run | 19.3 | 0.15 / - |

The pass child's `RssAnon` is 24.1-27.0 GiB with sharing on as well (the shared pages are in its RSS, but the cgroup counts them once, already held by the pool). Table md5 (decompressed) in all eight runs equals the required one (pbmc5p_v1 `a5d71eb6...` 24,141 rows, d300_5p `1763c9dc...` 102,847, GSM6250953 `3573f164...` 59,632).
Constant by the rule: the largest rise over the three samples at 8 threads and GSM6250953 at 16 and 27 threads is 3.61 GiB, rounded up to the next 0.5 GiB: `chimSharedGb = 4.0` (19.5 for the private index).

## Final binary `t2-chimshare-2`
1. ctest: `ctest exit 0` (build job 819116, `rt-build-819116.out`).
2. Identity gate job 819119 (default settings, `ref-t2`): 8 lines `VERDICT <sample> cpu:gpu IDENTICAL (except E11's run-dependent files; ...)` for pbmc2m, pbmc10m, pbmc1k_v3, pbmc1k_v2, pbmc1k_cite_v3, pbmc5p_v1, neuron1k_v3, nuclei2k_v2; `FALLBACK LINES: 1` in each (the R11 info line only).
3. Pool of three `--chimeric` items, old admission gate, one job (819120, g051, `--mem=200G`, arms S then N on the same node and inputs, 3 workers, 8 threads). The arms overlap their items, so a rise per pass cannot be separated; the single-item runs above are the memory measurement. Tables: S and N both give GSM6250953 `3573f1645c4fc6e920d37c70a0033df7` (59,632 rows), d300_5p `1763c9dca0013d9bb3e68d7bfc419efe` (102,847), pbmc5p_v1 `a5d71eb6a8b45889fe667bf57f17dce3` (24,141). The second-pass counter lines are equal between arms for all three items (junction lines 17,188,238 / 16,886,797 / 3,926,694; kept 4,498,881 / 145,563 / 27,130; unmatched, ambiguous, non-canonical, reoriented, rows, cells all equal).

| item | arm | item wall | pass (fork to pass log line, from log epochs) | pass genome load | `Genome preload` line |
|---|---|---|---|---|---|
| GSM6250953 | S | 658.8 s | 372 s | 0.15 s | taken from singlet's preload (4.79 s, shared copy-on-write) |
| GSM6250953 | N | 711.0 | 422 | 10.13 | none |
| d300_5p | S | 555.7 | 298 | 0.15 | taken |
| d300_5p | N | 597.3 | 340 | 10.18 | none |
| pbmc5p_v1 | S | 220.9 | 73 | 0.15 | taken |
| pbmc5p_v1 | N | 237.6 | 89 | 11.29 | none |

Pool wall: S 670 s, N 722 s. oom_kill 0 in both. Maximum of job anon+shmem over the arm (1 Hz, `cg1hz.tsv`): S 149.5 GiB, N 149.3 GiB; maximum `memory.current` S 168.2, N 174.8 GiB. Anon+shmem rise inside each pass window (other items' phases overlap, so these are not the passes' own): S 0.60 / 1.55 / -3.76 GiB, N 7.75 / 18.19 / 12.26 GiB (gsm / d300_5p / pbmc5p_v1; `t2-chimshare-win.sh`).
4. See the memory table above (build 1; the final binary differs only in `chimSharedGb`).
5. Arm P (job 819121, g051, `SINGLET_POOL_PEAKADMIT=1`, `--mem=200G`, sharing on): item figures 88.7 / 71.9 / 29.3 GiB (GSM6250953 / d300_5p / pbmc5p_v1; margin 12, `--chimeric: floor 0.5 P + 6.6 + margin`, i.e. 2.6432 + chimSharedGb). `peak admission: idle baseline B 42.60 GiB`; item 1 (d300_5p) `waits (figure 72 GiB; ... refused by the job term)` at the start, item 2 started ahead of it after 31 s, item 1 `admitted after 596 s`; `peak admission: model exceeded in 0 s of the refreshes (largest excess 0.0 GiB)`; `memory admission: 1 waits, 596 s waited in all`. Pool wall 1091 s (earlier result 1153 s, d300_5p waited 628 s). cgroup `memory.peak` 153,127,317,504 B (142.6 GiB), oom_kill 0. Table md5s as above. `memsamp-cg.tsv`: 525 samples with a valid baseline, largest AS-B minus the sum of figures -9.49 GiB (held memory never exceeded the sum of figures); maximum AS 105.06 GiB. The three passes took the genome (`Genome load seconds: 0.15`).
6. Standalone, no pool (same job 819121, `--engine cpu`, pbmc5p_v1, 16 threads): `sharing line count: 0`, `pass STAR log: Genome load seconds: 9.18`, table md5 `a5d71eb6a8b45889fe667bf57f17dce3` (24,141 rows), wall 252 s.

## Notes
- Jobs: builds 819102, 819116; gate 819119; measurements 819103, 819104; pools 819120, 819121 (five further jobs).
- Results: `$W/jobs/gpu-runtime/results/t2-chimshare-mem-819103-A`, `-819104-B`, `t2-chimshare-pool-819120-{S,N}`, `t2-chimshare-pool-819121-P` and `t2-chimshare-pool-819121/results.txt`; gate `$W/jobs/gpu-pr2/results/gate-819119`.
