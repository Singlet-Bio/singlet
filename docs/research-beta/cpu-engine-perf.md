# CPU engine (`--engine cpu`) profile, stream path, K5 + default rules (2026-10-30)

Phase 1 of the CPU-perf track: where do the core-seconds and the wall go, per thread count, packed or not, before anything is built.
Binary: `prod/campaign` d9834b8a, CPU-only (`-DSINGLET_ENGINE_GPU=OFF`, `-march=x86-64-v3`, non-PGO) plus a `-DSTAR_PROFILE` twin
(`scripts/cpu_perf/build_cpu.sbatch`). Whitelists sit at `bin/../whitelists`; no `SINGLET_WHITELIST_DIR`. Every run was checked for
`BAM: unsorted STAR stream into the pileup`, `Pileup: streaming from`, no `Pileup: parallel`, the whitelist path next to the binary, exit 0
(`STREAM PATH OK`; `scripts/cpu_perf/summarize.py` re-checks it from the logs: OK for all 12 plain runs, and the 3 profile runs show the same
log lines). Raw results: `$WORK/jobs/cpu-perf/results/` on Clipper.

**Caveat on the timings.** The CPU partitions had no start slot for days (other sessions' arrays), so all jobs ran on `scavenger`, at the
same moment, on mixed hardware (Xeon Gold 6226 2.7 GHz and 6230 2.1 GHz), up to three of them on one node. For a DRAM-latency-bound aligner
that is noisy: wall times differ by perhaps +-10%, core-s by a few percent. Treat the ratios as indicative; exclusive-node numbers (S5's
harness) are the release timings. One run per cell, no repeats.

## 1. Thread counts and packing (plain build)

| sample | reads | T | jobs/node | wall s | core-s used | core-s/M | allocated core-s per sample | peak RSS GB |
|---|---|---|---|---|---|---|---|---|
| pbmc1k_v3 | 66.6 M | 4 | 1 | 873 | 2,664 | 40.0 | 3,492 | 30.5 |
| | | 8 | 1 | 554 | 2,773 | 41.6 | 4,432 | 30.5 |
| | | 16 | 1 | 477 | 2,642 | 39.7 | 7,632 | 30.8 |
| | | 4 | 4 (16 cores) | 883 | 2,769 | 41.6 | 3,532 | 30.5 |
| neuron1k_v3 (mouse) | 92.9 M | 4 | 1 | 786 | 2,257 | 24.3 | 3,144 | 33.4 |
| | | 8 | 1 | 711 | 2,378 | 25.6 | 5,688 | 34.4 |
| | | 16 | 1 | 735 | 2,299 | 24.7 | 11,760 | 36.6 |
| | | 4 | 4 | 862 | 2,253 | 24.3 | 3,448 | 33.5 |
| GSM7036403 | 111.6 M | 4 | 1 | 1,210 | 3,676 | 32.9 | 4,840 | 34.1 |
| | | 8 | 1 | 971 | 3,837 | 34.4 | 7,768 | 34.2 |
| | | 16 | 1 | 891 | 3,651 | 32.7 | 14,256 | 34.6 |
| | | 4 | 4 | 1,316 | 3,678 | 32.9 | 5,264 | 34.1 |

core-s used = user+sys from `time -v`. Allocated core-s per sample = cores held x wall / samples on the node (what the queue and Anvil
SUs charge for).

1. **Used core-s is flat in T (within noise).** The work does not grow with threads; what grows is idle allocated cores.
2. **Wall barely improves past 8 threads:** 4 -> 16 threads is 1.8x (pbmc1k_v3), 1.07x (neuron), 1.36x (GSM7036403) for 4x the cores.
   Allocated cost per sample is 2.2x, 3.7x and 2.9x at 16 threads. Many 4-thread jobs win, as in the 29 Oct review.
3. **Packing 4 x 4-thread jobs on 16 cores costs +1% (pbmc1k_v3), +9% (GSM7036403), +10% (neuron) wall**, not the 15-28% of the review.
   The packed jobs shared nodes with other scavenger work, so this is if anything pessimistic; one run each.
4. A 4-thread job uses 3.1 of its 4 cores on average (pbmc1k_v3: 2,664 / 873); section 2 says where the rest goes.
5. Peak RSS 30.5-36.6 GB includes the mapped genome index (shared between jobs on a node through the page cache), so it cannot split
   private from shared. Clipper's 33.4 GB private + 17.8 GB shared is not re-measured here.

## 2. Per stage (4-thread jobs; profile build)

Stage walls and core-s from the job cgroup counter between log markers (`scripts/cpu_perf/stage_profile.py`). On the stream path STAR, the
serial pileup thread and the callers overlap, so one line covers all three; the split inside it is from STAR's own `STAR_PROFILE` counters
and a per-thread CPU sampler (`threads.py`).

| stage | pbmc1k_v3 wall s | core-s | per M | GSM7036403 wall s | core-s | per M |
|---|---|---|---|---|---|---|
| `.1fq` header, pass 1 (CB histogram) | 10.9 | 39 | 0.6 | 13.0 | 50 | 0.45 |
| barcode dictionary, references, genome start | 4.2 | 6.8 | 0.1 | 3.9 | 6.6 | 0.06 |
| **STAR map + serial pileup + callers (concurrent)** | **759.8** | **2,641** | **39.7** | **996.6** | **3,345** | **30.0** |
| non-host screen (EM, 500 iterations) | 36.3 | 52 | 0.8 | 44.5 | 62 | 0.55 |
| export (matrices, QC, demux, writers) | 74.6 | 97 | 1.5 | 175.3 | 256 | 2.3 |
| total | 887 | 2,837 | 42.6 | 1,234 | 3,719 | 33.3 |

Inside the concurrent line (thread-seconds from `STAR_PROFILE`):

| | pbmc1k_v3 T=4 | pbmc1k_v3 T=16 | GSM7036403 T=4 |
|---|---|---|---|
| STAR wall s | 645 | 270 | 819 |
| `map` (seed + windows + stitch) | 1,750 | 1,569 | 1,900 |
| - seed search | 962 | 746 | 1,070 |
| - windows | 245 | 246 | 225 |
| - stitch | 440 | 443 | 451 |
| `readLoad` (parse, clip) | 62 | 64 | 96 |
| `output` (alignment records, barcode, Solo, pipe write) | 541 | 2,293 | 900 |
| - of which `outSAM` (pipe write; blocks when the pileup is behind) | 320 | 2,088 | 577 |
| busiest single thread in the process (CPU-s; the serial pileup thread) | 606 | 324 | 736 |
| `Pileup: N reads` wall (s) | 760 | 372 | 997 |

Reading it:

- **STAR's `map` is 65% of the core-s** (seed search alone 36%, 55% of `map`). That is `src/star/`, S8's code; the CPU engine cannot shrink
  it without changing alignments (K5 profile fixed).
- **The serial pileup thread** is the busiest single thread (606 and 736 thread-s: 9 and 6.6 core-s/M, up to 22% of the run; the 324 at T=16
  is a sampling miss or a faster core, unresolved). **At T=4 it is not the bottleneck** (about 0.7-0.8 busy; `outSAM` is 12% of STAR's
  time). **At T=16 it is:** `outSAM` is 2,088 of about 4,300 thread-s (48% of STAR's threads wait on the pipe) and pileup wall is 372 s
  against STAR's 270 s. That matches the 261k reads/s cap in the brief (73.4 M records / 372 s = 197k/s end to end). Sharding the pileup
  raises the ceiling for big thread counts and shortens wall; it does not reduce core-s and **does not help the 4-thread jobs that are
  the efficient configuration.**
- **The tail after the pileup** (non-host + export: 111 s of 887 for pbmc1k_v3, 220 s of 1,234 for GSM7036403) runs at about 1.3 cores. In
  a 4-core allocation that is about 2.7 idle cores x 111-220 s = 300-590 core-s, 9-12% of the allocated cost. Export's donor demux (auto-K,
  VB) is the single-threaded part (GSM7036403 export: 175 s wall, 256 core-s).
- `.1fq` pass 1 plus the barcode dictionary is 1.4-1.7% of core-s. In this build the decode is inside STAR's reader (`[1fq] read source:
  chunk (decoded in STAR's process)`, `chunkRead` 9-15 thread-s), so there is little temp-FASTQ or extra-decode cost left on the CPU path.

## 3. Ranked levers (nothing built yet)

| rank | lever | owner | est. saving | notes |
|---|---|---|---|---|
| 1 | **Fill the tail**: parallelise export's donor demux and the non-host EM, or overlap a sample's tail with the next sample's mapping | CPU (export, non-host) | 9-12% of allocated core-s (300-590 per sample) | outputs stay identical if demux K and VB are parallelised, not changed |
| 2 | **Anvil packing is memory-bound, not core-bound**: (256 - 17.8) / 33.4 = 7 four-thread jobs per 128-core node (28 cores, 22%). Needs the private footprint per job to fall, or the per-core-billed `shared` partition | CPU + S8 (index and hand-off memory) | up to 4.5x node utilisation on Anvil | measure the private part (RSS minus shared index) first |
| 3 | Serial pileup thread: barcode-sharded order-free workers (R1-R12) | CPU pileup | 0 core-s; wall -20% to -45% at T >= 8 | helps nothing at T=4; worthwhile only if 8-16-thread jobs are wanted |
| 4 | Pileup micro-costs (6.6-9 core-s/M): DirectionalUmiStore linear scan and O(k^2) finalize, chrM COO triplet per base, uint8 clamp | CPU pileup | unknown | **not exercised by these three shallow samples**; a deep sample (GSM7656149, 263 M reads) needs a profile before ranking |
| 5 | Non-host EM: 500 iterations on 3.4-12 M unmapped reads | CPU (non-host) | 0.5-1 core-s/M | not output-identical (known D4b) |
| 6 | `.1fq` pass 1 + dictionary | CPU | 0.5-0.6 core-s/M (1.5%) | small |
| - | STAR seed search (36% of core-s), windows, stitch | S8 (`src/star/`) | - | out of scope for the CPU track |

## 4. Not measured

Depth hazards (lever 4); a repeat on exclusive nodes; Anvil (no runs, by instruction); the private/shared split of the 30-37 GB RSS; the 5'
sample class; more than 4 packed jobs per node.

## 5. Reproduce

`sbatch scripts/cpu_perf/build_cpu.sbatch` (`COMMIT=...`), then `scripts/cpu_perf/submit_matrix.sh`; `profile.sbatch` takes `SET T PACK BINV`.
`summarize.py RESULTS`, `stage_profile.py DIR M`, `threads.py DIR M` summarise (run them as a `debug-cpu` job, not on the login node).

## 4. Private memory per job (step 2)

Deep sample GSM7656149, `--engine cpu`, 4 threads, genome in shared memory (`singlet genome load`, then
`--genome-shared on`; STAR log: "Found genome in shared memory"), scavenger nodes (not exclusive).

| | private genome (793903) | shared genome (793999) | + finalize memory cut (794032) |
|---|---|---|---|
| STAR process anonymous peak | 23.6 GB | 7.2 GB | 7.2 GB |
| singlet main process anonymous peak | 32.0 GB | 32.1 GB (finalize) | 23.8 GB |
| job peak, sum of anonymous (billed basis) | 37.2 GB | 32.1 GB | 22.5 GB |
| billed cores at ~2 GB/core (Anvil `shared`) | 19 | 16-17 | 12 |
| genome, charged once to the loader | | ~23 GB | ~23 GB |

The finalize used to hold the molecule store (13.6 GB), the collapsed molecules (~5 GB) and the multi-gene UMI table
(~10 GB) at once. `DirectionalUmiStore::consume_molecules()` now gives each (barcode, gene) group back as it is
collapsed (same molecules, same order; the per-cell saturation statistics the export reads are computed first and
kept), and the multi-gene table is built in 16 barcode shards, one in memory per thread. Counts and every other
output are byte-identical to the previous run (all files except the five run-dependent ones); `test_umi_consume`
covers the consume path. Node packing: with the genome shared, a node holds one ~23 GB genome plus ~22 GB per job.

## 6. Post-STAR tail (step 3), first change: `--threads` reaches export and finalize

`--threads` set only STAR's thread count. The stream path runs the pileup on one thread (`--pileup-threads 1`) and
`export_cfg.threads` inherited that, so every export stage that takes a thread count (donor auto-K and VB, decontX,
doublets) ran with 1 (auto-K ran at ~1.1 cores for ~350 s on the deep sample while 4 were held). Export and the
multi-gene UMI shards now use `max(pileup threads, --threads)`. The demux reductions are chunked independently of the
thread count (`parallel_det_chunks`), so donor calls do not depend on it.

Deep GSM7656149, 4 threads, scavenger (non-exclusive): export 625 s -> ~400 s, auto-K 350 s -> ~200 s, pack wall
4059 s -> 3938 s together with the memory change; outputs byte-identical to the 1-thread run (except the run-dependent
files). Still serial and next: SNP consensus finalize 34 s, junction UMI finalize 50 s, molecule collapse ~65 s,
multi-gene shards ~50 s (the shard scan repeats the pass 16 times; a single partition pass would be cheaper).

## 7. Compiler flags (LTO, PGO, -march): no measurable gain

Build options `SINGLET_MARCH_FLAG`, `SINGLET_LTO`, `SINGLET_PGO=generate|use` (defaults unchanged) and
`scripts/cpu_perf/build_flags.sbatch` build one variant per job from one commit (9e6f95aa), all `-march=x86-64-v3`. Whole-program
PGO trains on pbmc10m + pbmc2m (the gate samples stay out of the training set; training on the full 66M-read samples took
>2 h instrumented). Runs: `--threads 4`, shared nothing, scavenger (non-exclusive), stream path OK on every run.

Outputs: every variant is byte-identical to plain v3 on pbmc1k_v3, neuron1k_v3 and the deep GSM7656149 (excluded: run-dependent
files, as in section 4).

| sample (node) | v3 | v3-lto | v3-starpgo | v3-pgo | v3-pgo-lto |
|---|---|---|---|---|---|
| pbmc1k_v3 wall s (g008 6230 unless noted) | 840 | 850 (g005) | 878 | 878 | 789 (g001, 6226 @2.7 GHz) |
| neuron1k_v3 wall s | 1972 | 1988 | 2192 | 2165 | 1957 (6226) |
| deep wall s | 3691 | 3668 | 3825 | 3451 (6226) | 3481 (6226) |

The runs landed on Xeon 6230 (2.1 GHz) and 6226 (2.7 GHz) nodes, so only same-node pairs compare: on g008, STAR-only PGO and
whole-program PGO are about 4% *slower* than plain v3 on pbmc1k_v3 and 11% slower on neuron1k_v3; the faster pgo / pgo-lto
rows are the faster CPU. LTO alone is within 1%. Conclusion: keep the current flags. znver3 cannot be measured on Clipper (Intel
only); same ISA level as v3, expected gain is small. An exclusive same-node A/B would be needed to resolve anything under ~5%.
