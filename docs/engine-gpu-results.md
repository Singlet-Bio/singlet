# GPU engine: measurements (2026-10-28)

What `singlet --engine gpu` is and how to build and check it: [engine-gpu.md](engine-gpu.md). The research record behind every
number here (methods, job ids, raw JSON): `docs/gpu-engine/exploration.md` §12 on the W11 branch, PR Singlet-Bio/singlet#61.
Every number is labelled **measured** (read from a job's output), **derived** (arithmetic on measured numbers), **projected** (a
measured rate applied to another sample or setting) or **modelled** (uses an outside price, power figure or stated assumption).
Hardware: Clipper GPU nodes (2 × NVIDIA H100 NVL 94 GB, 2 × 28-core Xeon, 768 GB; 54 schedulable cores) and CPU nodes (2 × Xeon
Gold 6230R, 52 cores). CPU truth: `prod/campaign` c4a1be38.

## 1. Identity

| test | result | label |
|---|---|---|
| STAR's mapping state per distinct sequence, GPU port vs STAR (its sequence-cache encoding: every transcript field, nW, nTr, best transcript, unmapType) | 1,991,926 / 1,991,926 identical (pbmc1k_v3, every 16th distinct sequence; 151,095 unmapped and 1,840,831 mapped, all identical) | measured |
| whole sample through the engine, STAR verifying every GPU result (`SINGLET_STAR_SEQCACHE_VERIFY=1`) | 66,004,541 reads took the GPU result (99.27%), **0 mismatches** (pbmc1k_v3, 66.6 M read pairs) | measured |
| this PR's binary: `scripts/engine_gpu/identity_test.sh`, pbmc1k_v3 (66.6 M read pairs), final build | **PASS**: verify run 64,775,880 reads took the GPU result, each re-mapped by STAR, **0 mismatches**; GPU-fed vs CPU outputs differ only where two CPU runs differ (below). Earlier builds of this branch: PASS twice more (61.1 M and 64.8 M reads verified, 0 mismatches). Jobs 789898, 789924, 789947; `bench/engine_gpu/results/` | measured |
| file-by-file outputs, `--engine gpu` vs `--engine cpu`, 3 samples in flight on one GPU node (pbmc1k_v3, GSM7036403, GSM7656149: 441.6 M read pairs) | every matrix, cell table, pileup, junction, SNP, mtDNA, QC and non-host read file identical (57–59 files per sample); differing: logs, timings, run summaries and provenance (8 files per sample) and, for GSM7036403, the non-host EM's two output files (its input, `nonhost_reads.fq`, is identical in content; see below) | measured (job 789929) |

Two differences come from the file-based decode `--engine gpu` uses (STAR must read the cDNA reads twice, so they cannot come
through the default FIFO): provenance only. The 5′-prefix probe records 12,194 sampled reads instead of 12,000 in
`pileup_stats.json`, and `read_stats.tsv` gains a decode line; no data output changes.

What differs between any two runs of the CPU engine itself (and therefore also between CPU and GPU runs): the order of records in
`star_Unmapped.out.mate1/2` and `nonhost_reads.fq` (same content), timings in logs and run summaries, and the non-host EM's
iteration count, which follows the order of its input reads (458 in three CPU runs; 458 and 462 in two GPU-fed runs, 0 species
above threshold in all). On GSM7036403 the EM's abundances (`nonhost_em_abundance.tsv`, `nonhost_summary.json`) differed between
the CPU and GPU runs for the same reason; the M13 work on the research branch saw the CPU engine's abundances move in the third
digit between runs too. Sorting the EM's reads would make it deterministic (a CPU-engine change, not in this PR).

## 2. Speed

| | value | label |
|---|---|---|
| GPU mapping rate (seeds → windows → stitching → selection), one H100 NVL | 2.41 M distinct sequences/s (pbmc1k_v3; 4.9 M reads/s at its 2.07 reads per distinct sequence) | measured |
| whole sample on the GPU (31.9 M distinct sequences of 66.6 M read pairs) | 17.1 s GPU time | measured |
| one sample on a GPU node, 24 threads (pbmc1k_v3): `--engine cpu` / `--engine gpu` wall | 418 s / 468 s; the GPU path's prefill is 76 s (reading the reads 27 s, GPU 44 s of which mapping 16 s and dedup 1.6 s, cache inserts 5 s) and runs before STAR; STAR then finds 97.4% of reads in the cache | measured (job 789947) |
| one GPU node, 3 samples in flight (17 threads each, 51 of 54 cores; 441.6 M read pairs), intermediate build: `--engine cpu` / `--engine gpu` | CPU used: **18,136 / 10,411 core-s (41.1 / 23.6 core-s per M read pairs: −43%)**; makespan 1,844 / 2,189 s (the deep sample's 647 s GPU prefill in that build was 4× slower than the final build's; final build (job 789966, same samples): **10,224 core-s (−44%), makespan 1,924 s**; GPU mapping 16 s / 17 s / 73 s for 31.9 M / 34.8 M / 137.2 M distinct sequences) | measured (job 789929) |
| CPU node, 4 copies of pbmc1k_v3 in flight (Xeon 6730P, 64 cores, 512 GB) | 0.394 M read pairs/s, 12 cores busy on average: 53.7 GB per sample, so memory, not cores, limits the samples in flight | measured (CPU-baseline subagent, job 789778) |

In this release the GPU replaces STAR's mapping; STAR's read loading, BAM writing and sorting, the pileup, callers and export still
run on the CPU, so a single sample's wall time is bounded by those stages. The research branch has the next stages ported and
identical (annotation and pileup: 9 of 9 outputs; non-host classification: every `nonhost/` file), not yet wired into the
binary.

## 3. Ranked bottlenecks per GPU node

Before tonight (end of W11 round 2, research branch) and after (this PR), M read pairs/s per node:

| stage | before | after (this PR) | evidence |
|---|---|---|---|
| GPU mapping (M5), 2 GPUs | 7.8–11.2, research binary only | same kernels, now inside `singlet` | in-kernel cycle counters (exploration.md §12.12): 54–85% of M5 time in window assignment and stitching of reads with > 16 windows |
| CPU work per read pair in the engine binary | 41.1 core-s per M read pairs (`--engine cpu` on the GPU node) | 23.6 (`--engine gpu`): −43% | node run's cgroup CPU counter (job 789929) |
| per-sample memory (limits samples in flight on both node types) | 54–74 GB | same, plus the cache for the GPU results (~3× the results: 7 GB for pbmc1k_v3, ~30 GB for a 263 M-pair sample) | `/usr/bin/time` max RSS |
| per-sample wall, one sample alone | 418 s (pbmc1k_v3, 24 threads) | 468 s: the 76 s GPU prefill runs before STAR | job 789947 |

What binds a GPU node with this binary is the CPU stages and per-sample memory, not the GPU (the GPUs read 0% utilisation in almost every 5 s
sample of the node run: `bench/engine_gpu/results/node_run1_gpu_monitor.txt`). Next, in order: (1) start STAR's mapping while the GPU fills the cache instead of after (the
prefill is serial today); (2) wire in the research branch's GPU pileup and non-host stages (identical; they remove 85% of the
remaining CPU work with STAR's BAM sort); (3) lower per-sample memory so more samples fit per node.

Profilers: Nsight Compute could not collect counters on the Clipper GPU nodes (the application aborted under the profiler; likely
the counter-permission setting) and Nsight Systems is not in the CUDA 12.4 devel image; the hot spots come from per-phase
`clock64()` counters compiled into the kernels (`stitch.cuh`, M5CLK) and from CUDA event timings per lane.

## 4. Cost case: projected for the GPU stages, and today

Model: `bench/engine_gpu/cost/cost_model.py` → `cost_case.json`; rates with sources: `bench/engine_gpu/cost/rates.{json,md}`.
CPU baseline (**measured**): `prod/campaign` c4a1be38 sources with 3′/5′ PGO, campaign command line, `--threads 4`, six real samples
(733 M read pairs), 37.6 core-s per M read pairs after crediting the per-node non-host index load (a change that helps both
pipelines). GPU node rates are **projected** from the measured M5 rate and each sample's measured distinct fraction.

| | CPU node (52 cores) | GPU node, M5 as on a whole sample | GPU node, best measured M5 rate | target: M5 at the host bound |
|---|---|---|---|---|
| M read pairs/s per node | 1.39 | 8.7 | 11.2 | 18.0 |
| samples/hour (6-sample mix) | 41 | 256 | 331 | 530 |
| **CPU nodes offset per GPU node** | — | **6.3** | **8.1** | **13.0** |
| $ per sample, cloud on-demand ($3.99 / H100-hour, $0.05 / core-hour) | $0.064 | $0.031 | $0.024 | $0.015 |
| same if cloud CPU cores are 2× faster than Clipper's 2019 Xeons | $0.032 | | | |
| ACCESS credits per sample (Anvil; GPU rate derived) | 1.28 | 0.52 | 0.41 | 0.25 |
| energy per sample, kWh (estimated, not metered) | 0.017 | 0.0085 | 0.0066 | 0.0041 |

**This PR's binary, today** (only mapping on the GPU), with the same prices:

| | CPU node, `--engine cpu` | GPU node, `--engine gpu` | label |
|---|---|---|---|
| CPU core-s per M read pairs | 41.1 (on the GPU node; 37.5 on Clipper CPU nodes) | 23.6 | measured (job 789929) |
| node rate if cores bind (54 or 52 cores) | 1.27–1.39 M pairs/s | 2.29 M pairs/s | derived |
| $ per sample (122 M pairs) at node-hour prices ($2.60 CPU node, $7.98 GPU node) | $0.06–0.07 | $0.12 | modelled |

So **today's binary is not cheaper per sample in the cloud**: it cuts the CPU work by 43%, but the GPUs sit idle most of the
time and a GPU node costs ~3× a CPU node. On an owned cluster (Clipper), where the GPU nodes exist anyway, it frees 43% of their
CPU time per sample with identical outputs. The table above (6–8 CPU nodes per GPU node, 2.0–2.6× cheaper) is the projection for
the research branch's GPU stages (annotation, pileup, non-host; all identical) carrying the CPU work, and for M5 at its measured
rate; it is not what this binary delivers.

## 5. Not done

- GPU mapping at ≥ 10 M distinct sequences/s (2.41 M now).
- Annotation, pileup and non-host classification on the GPU inside the binary (identical on the research branch).
- GPU read clipping (5′ samples with adapter clipping map on the CPU in this release).
- Metered power; CPU node-packing efficiency (single-sample core-s used, so the CPU node rate is an upper bound).
