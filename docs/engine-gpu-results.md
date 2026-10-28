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
| this PR's binary: `scripts/engine_gpu/identity_test.sh` | IDENTITY_RESULT | measured |
| file-by-file outputs, GPU engine vs CPU engine, several samples in flight | NODE_IDENTITY | measured |

What differs between any two runs of the CPU engine itself (and therefore also between CPU and GPU runs): the order of records in
`star_Unmapped.out.mate1/2` and `nonhost_reads.fq` (same content), timings in logs and run summaries, and the non-host EM's
iteration count, which follows the order of its input reads (458 in three CPU runs; 458 and 462 in two GPU-fed runs, 0 species
above threshold in all). Sorting the EM's reads would make that deterministic (a CPU-engine change, not in this PR).

## 2. Speed

| | value | label |
|---|---|---|
| GPU mapping rate (seeds → windows → stitching → selection), one H100 NVL | 2.41 M distinct sequences/s (pbmc1k_v3; 4.9 M reads/s at its 2.07 reads per distinct sequence) | measured |
| whole sample on the GPU (31.9 M distinct sequences of 66.6 M read pairs) | 17.1 s GPU time | measured |
| sample wall time, one sample on a GPU node (24 threads): `--engine cpu` / `--engine gpu` | WALL_CPU / WALL_GPU | measured |
| node throughput, 4 samples in flight on one GPU node (both GPUs, 52 cores): `--engine gpu` / `--engine cpu` | NODE_RATE | measured |

In this release the GPU replaces STAR's mapping; STAR's read loading, BAM writing and sorting, the pileup, callers and export still
run on the CPU, so a single sample's wall time is bounded by those stages. The research branch has the next stages ported and
identical (annotation and pileup: 9 of 9 outputs; non-host classification: every `nonhost/` file), not yet wired into the
binary.

## 3. Ranked bottlenecks per GPU node

Before tonight (end of W11 round 2, research branch) and after (this PR), M read pairs/s per node:

| stage | before | after (this PR) | evidence |
|---|---|---|---|
| GPU mapping (M5), 2 GPUs | 7.8–11.2, research binary only | same kernels, now inside `singlet` | in-kernel cycle counters (exploration.md §12.12): 54–85% of M5 time in window assignment and stitching of reads with > 16 windows |
| host work in the engine binary, 54 cores | — (not runnable on a GPU node) | NODE_HOST | the node run's cgroup CPU counter |
| the CPU stages still in the binary (read load, BAM, sort, pileup, callers, export) | measured as part of the CPU engine: 37.6 core-s per M pairs in total | NODE_CPU_STAGES | node run |

Profilers: Nsight Compute could not collect counters on the Clipper GPU nodes (the application aborted under the profiler; likely
the counter-permission setting) and Nsight Systems is not in the CUDA 12.4 devel image; the hot spots come from per-phase
`clock64()` counters compiled into the kernels (`stitch.cuh`, M5CLK) and from CUDA event timings per lane.

## 4. Cost case

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

In words: one GPU node does the work of 6–8 CPU nodes with identical outputs once the GPU stages carry the node, at 2.0–2.6× lower
cloud cost per sample (1.0–1.3× if cloud CPU cores are twice as fast as the 2019 Xeons the baseline ran on). In this PR's binary
only mapping is on the GPU, so the node rate measured in §2 is what a GPU node delivers today; the table is the rate once the
research branch's GPU stages are wired in.

## 5. Not done

- GPU mapping at ≥ 10 M distinct sequences/s (2.41 M now).
- Annotation, pileup and non-host classification on the GPU inside the binary (identical on the research branch).
- GPU read clipping (5′ samples with adapter clipping map on the CPU in this release).
- Metered power; CPU node-packing efficiency (single-sample core-s used, so the CPU node rate is an upper bound).
