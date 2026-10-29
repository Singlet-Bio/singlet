# W11 host-time study (M10b): where the post-pileup CPU time goes

Measured with an instrumented DEV build of `prod/campaign` (28969b87 + `timers.patch`), 6 real samples, `--threads 4 --pipeline
--output-format 1pz --bam-sort-mode disk --genome-shared off` (+ run screen), `short-cpu`, 8 cores per run. Timers only: no output changes.
The dev binary is `$W/jobs/w11/host/bin/singlet-timers` (PGO, same `pgo_build.sh` and training data as the campaign binary).
`$W/pipeline-benchmark/bin/*` was not touched.

## Files
- `timers.patch` (against 28969b87): `include/singlet/pileup/w11_timers.h` (new) + timers in `export.h`, `donor_demux.h`, `singlet.cpp`.
- `build_timers.sbatch` (job 789193, 7 min), `run_timers.sbatch` (array 789215), `replay_export.sbatch` (job 789253).
- `parse_w11.py`, `make_tables.py`, `parse_replay.py` produce `../results/m10b_timers.json` (per sample: all sections with wall / process cpu /
  thread cpu, the export ledger, demux sizes and K trace, cgroup cross-check), `../results/m10b_ranked.json`, `../results/m10b_replay.json`.
- `runs/<GSM>/` (run.log, `w11.log` = the `[w11t]`/`[w11n]` lines, monitor.tsv = cgroup CPU every 5 s, node/time/mem, small summaries), `replay/`.

## What the timers print
`[w11t] <section> wall_s= cpu_s= thr_cpu_s= t0= tid=` when a section ends.
- `cpu_s` is the CLOCK_PROCESS_CPUTIME_ID delta. It counts every thread, so a section that runs next to another (the `.1pz`/tsv writers and
  the demux thread in phase 3 of the export) also counts the other one's CPU.
- `thr_cpu_s` is CLOCK_THREAD_CPUTIME_ID of the calling thread: exact for a one-thread section, but it misses helper threads the section
  spawns (pz codec, OpenMP, the 8-thread CSC conversion, the all-hardware-threads doublet scorer).
- The stages and the serial part of the export run alone, so their `cpu_s` is exact. Phase-3 sections use `thr_cpu_s`; the rest of the export's
  process CPU is booked as "unattributed". The ledger therefore sums to the process CPU of the export by construction, and the cgroup monitor
  independently agrees with the timers within 1-3% (gap window 66.6 vs 64.6 core-s for GSM4006843; export-to-exit 482.3 vs 483.3 for GSM7656149).
- `[w11n]` lines carry sizes (cells, SNPs, K tried, restarts).

## Core-seconds per section (W = everything after the pileup)

W = gap (end of pileup to start of export: non-host screening, BAM unlink, strand check) + export (`export_results`) + the tail to process exit.
Sections are grouped from the individual timers (`m10b_timers.json` has all ~150). Ranked by pooled core-seconds over the six samples.

| rank | section | GSM4006843 (67 M) | GSM5411066 (70 M) | GSM7901264 (100 M) | GSM7036403 (112 M) | GSM6470208 (122 M) | GSM7656149 (263 M) | sum core-s | per M pairs (pooled) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | demux: auto-K selection (K fits, 2 folds) | 39.6 | 7.2 | 8.1 | 83.0 | 36.9 | 321.6 | 496 | 0.677 |
| 2 | nonhost: k-mer classify (OpenMP) | 20.5 | 20.3 | 40.8 | 29.7 | 77.2 | 126.0 | 315 | 0.429 |
| 3 | nonhost: load 3 indices (viral+bacterial+fungal) | 33.2 | 28.6 | 28.0 | 53.3 | 46.2 | 42.7 | 232 | 0.316 |
| 4 | nonhost: EM deconvolution | 4.9 | 7.8 | 19.2 | 5.8 | 24.7 | 17.8 | 80 | 0.109 |
| 5 | demux: final fit (selected K) | 28.5 | 1.6 | 1.9 | 8.9 | 5.0 | 27.2 | 73 | 0.100 |
| 6 | nonhost: read FASTQ, cell matrix, writes, bloom | 4.0 | 5.7 | 7.7 | 6.2 | 19.3 | 22.6 | 65 | 0.089 |
| 7 | export: COO->CSC (8 threads) | 5.1 | 2.4 | 2.5 | 7.7 | 6.7 | 33.0 | 57 | 0.078 |
| 8 | export: .1pz / tsv writers (own threads, sum of thread CPU) | 5.6 | 3.3 | 3.6 | 8.3 | 8.5 | 14.9 | 44 | 0.060 |
| 9 | export: cell calling (called_v2) | 4.5 | 3.7 | 1.6 | 5.6 | 4.4 | 24.1 | 44 | 0.060 |
| 10 | export: doublet scoring (Scrublet-like) | 10.2 | 1.9 | 0.8 | 7.9 | 12.1 | 7.1 | 40 | 0.054 |
| 11 | export: unattributed (writer codec helper threads, gaps) | 5.8 | 2.5 | 1.1 | 6.7 | 7.0 | 15.9 | 39 | 0.053 |
| 12 | export: ambient (DecontX-like) | 3.9 | 1.6 | 0.5 | 3.6 | 2.2 | 16.4 | 28 | 0.038 |
| 13 | gap: BAM unlink | 1.9 | 2.1 | 3.0 | 5.9 | 8.4 | 1.9 | 23 | 0.032 |
| 14 | export: small callers + cell_qc + donor writes | 1.7 | 0.8 | 0.7 | 3.3 | 6.2 | 9.7 | 22 | 0.031 |
| 15 | export: mt event caller (G6) | 2.0 | 1.3 | 1.0 | 3.1 | 3.1 | 5.4 | 16 | 0.022 |
| 16 | export: mt heteroplasmy calls | 1.7 | 0.5 | 0.4 | 2.5 | 2.2 | 3.8 | 11 | 0.015 |
| 17 | export: saturation curve | 1.6 | 0.6 | 0.2 | 1.3 | 0.8 | 3.0 | 7 | 0.010 |
| 18 | demux: other (covered SNPs, doublet stage, back-assign) | 0.3 | 0.1 | 0.1 | 0.5 | 0.3 | 1.1 | 2 | 0.003 |
| 19 | post-export (provenance, summary, stats json, validate) | 0.1 | 0.1 | 0.1 | 0.2 | 0.1 | 0.4 | 1 | 0.001 |
| 20 | gap: other (strand check, log parse, cleanup) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0 | 0.000 |
| | **TOTAL W (post-pileup host)** | **174.9** | **92.2** | **121.2** | **243.6** | **271.4** | **694.8** | 1598 | 2.179 |
| | W per M pairs | 2.62 | 1.32 | 1.21 | 2.18 | 2.23 | 2.64 | | |
| | check: W from timers (gap+export+post) | 174.9 | 92.2 | 121.2 | 243.6 | 271.4 | 694.8 | | |

per M pairs:
| section | GSM4006843 | GSM5411066 | GSM7901264 | GSM7036403 | GSM6470208 | GSM7656149 |
| demux: auto-K selection (K fits, 2 folds) | 0.593 | 0.104 | 0.081 | 0.744 | 0.303 | 1.221 |
| nonhost: k-mer classify (OpenMP) | 0.308 | 0.292 | 0.406 | 0.266 | 0.634 | 0.478 |
| nonhost: load 3 indices (viral+bacterial+fungal) | 0.498 | 0.411 | 0.280 | 0.478 | 0.379 | 0.162 |
| nonhost: EM deconvolution | 0.073 | 0.112 | 0.191 | 0.052 | 0.203 | 0.067 |
| demux: final fit (selected K) | 0.427 | 0.024 | 0.019 | 0.080 | 0.041 | 0.103 |
| nonhost: read FASTQ, cell matrix, writes, bloom | 0.059 | 0.081 | 0.076 | 0.056 | 0.159 | 0.086 |
| export: COO->CSC (8 threads) | 0.076 | 0.034 | 0.024 | 0.069 | 0.055 | 0.125 |
| export: .1pz / tsv writers (own threads, sum of thread CPU) | 0.084 | 0.048 | 0.036 | 0.074 | 0.070 | 0.056 |
| export: cell calling (called_v2) | 0.067 | 0.053 | 0.016 | 0.050 | 0.036 | 0.092 |
| export: doublet scoring (Scrublet-like) | 0.153 | 0.028 | 0.007 | 0.071 | 0.100 | 0.027 |
| export: unattributed (writer codec helper threads, gaps) | 0.086 | 0.036 | 0.011 | 0.060 | 0.057 | 0.061 |
| export: ambient (DecontX-like) | 0.058 | 0.022 | 0.005 | 0.032 | 0.018 | 0.062 |
| gap: BAM unlink | 0.029 | 0.031 | 0.030 | 0.053 | 0.069 | 0.007 |
| export: small callers + cell_qc + donor writes | 0.026 | 0.012 | 0.007 | 0.030 | 0.051 | 0.037 |
| export: mt event caller (G6) | 0.030 | 0.019 | 0.010 | 0.027 | 0.026 | 0.021 |
| export: mt heteroplasmy calls | 0.025 | 0.007 | 0.004 | 0.022 | 0.018 | 0.014 |
| export: saturation curve | 0.024 | 0.009 | 0.002 | 0.012 | 0.007 | 0.011 |
| demux: other (covered SNPs, doublet stage, back-assign) | 0.004 | 0.001 | 0.001 | 0.005 | 0.003 | 0.004 |
| post-export (provenance, summary, stats json, validate) | 0.002 | 0.001 | 0.001 | 0.002 | 0.001 | 0.002 |
| gap: other (strand check, log parse, cleanup) | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |


Per-sample W per M pairs: 2.62 / 1.32 / 1.21 / 2.18 / 2.23 / 2.64 (pooled 2.18). Export alone per M pairs: 1.65 / 0.40 / 0.22 / 1.28 / 0.78 / 1.84.

## Serial or parallel
- **The export runs on one main thread** except: the CSC conversion (8 threads, 2.6-6 s wall, up to 33 core-s at 263 M), the doublet scorer and the
  ambient EM (they use every hardware thread, `threads=0`, so their CPU is 2-4x their wall), and phase 3, where the `.1pz`/tsv writers (~16 threads, all
  done in 1-5 s) run next to the **demux thread**, which is one thread and outlives them. Demux is **68.6 of 87.3 s** of export wall for GSM4006843 and 351.6 of 431.2 s at 263 M.
- `export_cfg.threads` is 1 in this code path whatever `--threads` is (`demux_input ... threads=1` in every run, also with `--threads 8`), so **the demux fits
  are fully serial** (the two K-selection folds run one after the other and `fit_vb_k` runs its restarts on one worker).
- Export CPU / wall is 1.1-1.4. The non-host screening is mostly serial too (index load, EM), with one OpenMP phase (k-mer classify, 4 threads).
- For contrast the pileup (not part of W) uses 2.6-3.3 cores per wall second at `--threads 4`.

## Demux breakdown
All six samples are single-donor: auto-K selects **K=1** but tries K=1..4 (it stops after `k_select_patience=3` K without a gain, so three K are wasted).
The final fit runs at K=1; the doublet model (K>=2 only), the staged restarts and split-merge never run. `n_donors_cfg=-1` (auto).

| sample | fit cells | covered SNPs / reduced (auto-K) | auto-K core-s | per K (wall s, K=1..4) | PCA (2 folds, thread CPU) | final fit K=1 | demux thread total |
|---|---|---|---|---|---|---|---|
| GSM4006843 | 10,889 | 426,198 / 65,369 | 39.6 | 2.3 / 6.6 / 9.6 / 13.6 | 7.4 | 28.5 | 68.3 |
| GSM5411066 (mouse) | 3,659 | 248,405 / 37,003 | 7.2 | 0.6 / 1.7 / 2.3 / 2.6 | 0.0 | 1.6 | 8.9 |
| GSM7901264 | 1,290 | 359,993 / 35,944 | 8.1 | 0.5 / 1.4 / 1.9 / 2.3 | 2.0 | 1.9 | 10.2 |
| GSM7036403 | 7,445 | 624,025 / 104,974 | 83.0 | 4.8 / 19.3 / 25.4 / 18.6 | 14.8 | 8.9 | 92.5 |
| GSM6470208 (5') | 9,568 | 494,044 / 42,975 | 36.9 | 2.9 / 5.7 / 10.4 / 10.4 | 7.4 | 5.0 | 42.3 |
| GSM7656149 | 3,274 | 1,479,710 / 339,869 | 321.6 | 35.2 / 64.8 / 83.2 / 93.4 | 44.8 | 27.2 | 351.6 |

- **Auto-K is 80-91% of the demux thread in 5 of 6 samples (58% for GSM4006843, whose final fit is long).** Inside it ~80% is `fit_vb_k` (3 PCA-seeded / random
  VB restarts of 60 iterations per fold and K) and 14-24% is the PCA cache; tables, scoring and the distinctness test are < 1% (`count_valid` ~0).
  K=2..4 cost 75% of auto-K (GSM4006843: K=1 2.3 s of 39.6 s) and cannot win in a single-donor sample.
- **What scales it is the number of SNP observations in the fit cells, not the number of cells.** The 263 M sample has the fewest fit cells (3,274) but the most
  SNP data (1.48 M covered SNPs, 340 k kept for the K search, snp_nnz 5.3 M): auto-K 322 core-s. GSM7036403 (7.4 k cells, 105 k reduced SNPs) costs 83 s; GSM4006843 (10.9 k cells, 65 k) 40 s.
  The mouse sample runs the human SNP panel on a mouse genome: 12 k SNP allele entries, so it is cheap (7 s).
- **The final fit at K=1** (one random restart plus up to 200 VB iterations over all covered SNPs) is 28 s on GSM4006843 (42 iterations) and 27 s on GSM7656149, 2-9 s elsewhere. At K=1 it is only a genotype-frequency estimate.
- Back-assignment of the 74-160 k ambient barcodes, covered-SNP extraction (0.2 s thread CPU; 1-15 s counting its helper threads) and the VCF writes are < 2 s together.
- Not exercised: the multi-donor path (final K>=2: 24 staged restarts, doublet model, split-merge). Its timers are in the patch (`staged.*`, `demux.doublet_*`, `doublet_stage.*`) but no sample here is multi-donor.

## Gap before the export (the old "non-host residual")
Essentially all of it is `run_nonhost_em_screening` (viral + bacterial + fungal k-mer screen of 3-6 M unmapped reads) plus the BAM unlink:
- loading the three indices: 28-53 core-s per sample (CPU-bound; 0.3-0.5 per M pairs at 66-110 M, 43 s at 263 M, so close to a per-sample constant);
- OpenMP k-mer classify: 20-126 core-s; EM deconvolution: 5-25 core-s, one thread, it hit the 500-iteration cap in every sample;
- FASTQ read, per-cell matrix, writes: 4-23; BAM unlink 2-8 s of CPU. Strand check and cleanup are ~0. Timers: `nonhost.*`.

## Reconciliation with section 12.3 (M10, monitor deltas)
| sample | M10 export | now export | M10 residual | now gap | M10 W | now W (timers) |
|---|---|---|---|---|---|---|
| GSM4006843 | 121.8 | 110.2 | 79.1 | 64.6 | 200.9 | 174.9 |
| GSM5411066 | 50.6 | 27.6 | 92.8 | 64.5 | 143.4 | 92.2 |
| GSM7901264 | 58.1 | 22.4 | 147.6 | 98.7 | 205.7 | 121.2 |
| GSM7036403 | 202.4 | 142.4 | 92.7 | 101.0 | 295.1 | 243.6 |
| GSM6470208 | 90.2 | 95.5 | 169.6 | 175.8 | 259.8 | 271.4 |
| GSM7656149 | 468.2 | 483.3 | 179.7 | 211.0 | 647.9 | 694.8 |

- The timer W is 2-7% above M10's for the 3 samples at 112 M and deeper, and 13-41% below it for the 3 smaller ones. The residual now has a name: it is the non-host screen plus the BAM unlink; nothing else is in it (non-host is 62-209 core-s of a 65-211 core-s gap; the remainder is the BAM unlink).
- M10's export numbers are windows of 5 s monitor samples (last `[export] Total` seconds + 10 s); they are 1.8-2.6x higher on GSM5411066 and GSM7901264. Those were other executions of the same binary on other nodes.
  Two effects move CPU between runs and are visible here: the doublet scorer and the ambient EM use every hardware thread (CPU 2-4x wall), so their CPU depends on the node, and M10's 10 s tail picks up process teardown. I could not separate them further without re-running M10's runs; the new numbers are exact per section and agree with the cgroup counter.
- Median W per M pairs over the six samples is 2.2 (M10: 2.3); export alone 0.78 (M10: 1.26). The two largest items are the demux auto-K and the non-host screen (auto-K 31% and the non-host screen 43% of the pooled W).

## Thread and contention scaling (step 5, GSM4006843, job 789253, 48-core node c101, Xeon 6730P)
`--bam-file` replay works. A first run with `SINGLET_KEEP_STAR_BAM=1` (`--threads 8`) leaves the sorted BAM and `auto_barcodes.tsv` in /tmp. A replay with
`--bam-file BAM --barcodes auto_barcodes.tsv --cell-calling --umi-len 10 --whitelist WL --metadata-json {"protocol":"10x-3p-v2"}` (STARsolo side files copied) reproduces the
full run's sizes exactly (119,473 barcodes, 13,101 called cells, same demux input, K=1 after 4 K tried) but pileups the BAM on one thread and skips the non-host screen (no unmapped-read files), so it measures the export only.

| run | export wall s | export core-s | demux thread CPU | auto-K | final fit | doublet cpu s |
|---|---|---|---|---|---|---|
| full run, --threads 8 | 90.1 | 115.3 | 72.2 | 44.0 | 27.9 | 10.7 |
| replay --threads 1 | 92.4 | 117.7 | 69.4 | 41.7 | 27.4 | 11.6 |
| replay --threads 4 | 94.2 | 119.3 | 71.1 | 43.0 | 27.8 | 11.3 |
| replay --threads 8 | 91.1 | 116.0 | 67.8 | 40.4 | 27.1 | 11.3 |
| 8 concurrent, --threads 4 (min / max of 8) | 91.1 / 96.0 | 116.6 / 120.8 | 65.5 / 69.7 | 38.8 / 41.9 | 26.4 / 28.1 | 10.3 / 10.9 |

- **Export core-seconds do not change with `--threads`** (116-119) because the export never sees it (`threads=1`). **8 concurrent copies change them by < 3%** (mean +1.5 core-s) on this node (it held only these jobs; a co-tenanted node was not tested).
  A replay peaks at ~11.5 GB RSS, so 8 at once fit easily.
- Cell calling, ambient, CSC conversion and every other per-section number reproduce within noise between the full run, the replays and the 8 concurrent copies.

## Caveats
- Writer sections use thread CPU and miss the pz codec's helper threads; the "unattributed" row (2-6% of W) holds them.
- All six samples are single-donor, so the multi-donor demux path is untimed in practice.
- The step-4 array shows FAILED (exit 1) for every task: the exit code comes from the last command of the job script (a `[ -f ] && ...` loop), after `singlet` returned 0 and every file was copied. The data are complete.
- The first step-5 job (789253) hung in `wait` on its own monitor loop after all work was written, and was cancelled after 36 min. Fixed in `replay_export.sbatch`.
- Core-hours: build ~2, step 4 ~19 (145 task-minutes x 8 cores), step 5 ~29 (48 cores x 36.5 min incl. the hang); about 50 in total.
