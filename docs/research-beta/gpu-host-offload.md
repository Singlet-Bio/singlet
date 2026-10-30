# S8: host stages of `--engine gpu`, ranked (2026-10-30)

Goal: make the GPU the bottleneck; the host only feeds reads and writes outputs. This is the phase-1 profile (no code changed yet).

## Setup
`prod/campaign` e368626c (plain build and a `-DSTAR_PROFILE` build, Clipper job 793495). One H100 NVL node (g051/g052), 18 threads, 20 cpus, the
default CPU-engine run as baseline. Jobs 793516_{0,1,2} (`bench/engine_gpu_gate/s8_profile.sbatch`); stage split by `s8_stages.py`
(log timestamps + a 1 s sampler of the job's cgroup CPU time and `nvidia-smi` utilization.gpu). K5 is on (CR4 probe -> profile ON in every log).
Every run logged `BAM: unsorted STAR stream into the pileup` and `Pileup: streaming from STAR pipe`; the GPU-pileup runs logged `[gpu_pileup] on`.
**The whitelists must sit at `<binary dir>/../whitelists`**: the first array (793501) had none, silently took the sorted-BAM path and was discarded.

Modes: `cpu` = default engine; `gpuA` = `--engine gpu`, default intron rules (what ships: #80 rules are not on the GPU, so the CPU pileup runs);
`gpuB` = `--engine gpu` + `SINGLET_GPU_PILEUP=on` + `SINGLET_INTRON_RULES=0` (everything that exists on the GPU is on it; this is the profile below).

## Totals
| Sample (reads) | cpu core-s (/M) wall | gpuA core-s (/M) wall | gpuB core-s (/M) wall | GPU busy, gpuB |
|---|---|---|---|---|
| pbmc1k_v3 (66.6 M) | 2627 (39.4) 494 s | 1314 (19.7) 509 s | 961 (14.4) 373 s | ~22 s = 6% |
| neuron1k_v3 (92.9 M) | 2170 (23.4) 566 s | 1637 (17.6) 598 s | 1158 (12.5) 462 s | ~36 s = 8% |
| GSM7036403 (111.6 M) | - | 2339 (21.0) 879 s | 1562 (14.0) 605 s | ~21 s = 3.6% |

GPU busy-s = mean `utilization.gpu` x wall (1 s samples, so +-30%). Host core-s is ~40x the GPU busy-s. Identity, `cpu` vs `gpuA` (pbmc1k_v3, neuron1k_v3):
60 files identical, 3 differ only in timings/paths, 6 differ in logs and read order (`nonhost_reads.fq`, `star_Unmapped.out.mate1/2`, `star_Log.std.out`,
`read_stats.tsv`, probe counts), the same set as the #95/#98 gates; every count output identical.

## Ranked host stages (gpuB, core-s per M reads; range over the three samples; wall = pbmc1k_v3 / GSM7036403)
| # | Stage | core-s/M | wall s | On the GPU? | Expected saving |
|---|---|---|---|---|---|
| 1 | **CPU pileup under the default intron rules** (gpuA only, shipped config) | 10.1-11.2 | 268 / 465 | Yes: kUnique/kNoSplicedIntron/kMtPrefer + kUmiGeneCR mirror into `gpu_pileup` | ~9 (the gpuA -> gpuB gap). **Owned by the mirror-80 session; not touched here** |
| 2 | STAR per-read host pass (readLoad 51, map 24, output 109 thread-s on pbmc1k_v3 `STAR_PROFILE`; 18 threads, wall 34 s) | 3.8-4.1 | 40 / 63 | Yes: sequence-cache lookup + summary replay + hand-off entry emit are table lookups over what the prefill already computed | 3-3.5 |
| 3 | Prefill host work: read the FASTQ text, CR4 opal TSO clip + polyTail3p, upload, cache inserts (engine log: reading 15 s, inserts 8 s, GPU 46 s) | 3.2-3.9 | 68 / 98 | Partly: 2-bit packed reads from #82's source (no text parse), opal clip as a kernel, inserts overlapped | 2-2.5 |
| 4 | `.1fq` -> FASTQ temp files + R11 CB histogram (18 threads) | 1.25-1.3 | 18 / 35 | Mostly: with 2 and 3 the temp FASTQ is not needed; the R11 histogram can be counted on the GPU or on the packed reads | 1-1.2 |
| 5 | Pipe consumer of STAR's unsorted stream into the pileup (1 thread; wall-critical: 23% of pbmc1k_v3 wall) | 1.2-1.35 | 85 / 127 | Disappears when the hand-off entries come straight from stage 2 (no BAM records for these reads) | 1.2 + 85-127 s wall |
| 6 | Export, QC, callers (cell calling, ambient, doublet; 1-2 threads) | 0.84-1.6 | 30 / 75 | Mostly small dense kernels; later | ~0.5, wall-critical on deep samples |
| 7 | Demux, ancestry, writers (1-3 threads) | 0.3-1.1 | 41 / 107 | Writers stay on the host (CPU only writes outputs); VB demux could move | wall mostly |
| 8 | Hand-off pass (`gpu_host_pass`, 1 thread) | 0.44-0.8 | 30 / 46 | Yes | 0.5 + 30-60 s wall |
| 9 | Non-host screening (host part) | 0.5-0.9 | 21 | Already on the GPU; host side is index load + EM | small |
| 10 | STAR load, CR4 probe (200 k reads mapped on the CPU), references | 0.5-0.8 | 37 / 31 | Probe could use the GPU path | ~0.3 |

GPU pileup itself: 1.7 s (pbmc1k_v3) and 0.3 s (GSM7036403) of GPU time. Prefill GPU phase: ~46-50 s of mostly seed/stitch (utilization 11-24% while it runs).

## What this says
- Even with everything that exists on the GPU turned on, the host spends 12.5-14.4 core-s/M and the GPU is idle 92-96% of the wall. Stages 2-5 (about 9 of 14 core-s/M) are
  the "feed and replay" work the brief names; stage 1 is the larger win for the shipped configuration but belongs to mirror-80.
- Wall is dominated by serial pieces, not by core-s: stage 5 + 8 are single-threaded and add ~115-175 s of a 373-605 s wall; export and demux add ~70 s (pbmc1k_v3) and ~180 s (GSM7036403).
- First targets, in this order: (2)+(5)+(8) one change (STAR emits hand-off entries from the cache without the BAM stream: removes the per-read output and the pipe consumer),
  then (3)+(4) (packed reads into the prefill, clip kernel, no temp FASTQ), then the tail (6, 7).
- Caveat on the numbers: stage boundaries come from log markers and the prefill's device-memory window; they are good to a few seconds. `utilization.gpu` is a 1 s point sample.
