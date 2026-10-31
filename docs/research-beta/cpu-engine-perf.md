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

## 3b. Deep sample and memory (GSM7656149, 263 M read pairs, T=4, scavenger so non-exclusive, 2026-10-31)

Build 07e71c2d (PR #112) = d9834b8a + indexed UMI pair loops + `[mem]` markers. All count outputs identical to d9834b8a. 47.6 core-s/M,
4,145 s wall, 12,537 core-s (d9834b8a: 12,508 user+sys, 4,216 s; the 2% is within scavenger noise). Stage split: STAR map + serial pileup
89.5% of core-s, non-host 1.5%, export 8.1% (663 s wall at ~1.5 cores).

**Memory, from `/proc/<pid>/smaps_rollup` every 15 s (`scripts/cpu_perf/mem.py`):**

| phase | t (s) | STAR process, anon GB | singlet main, anon GB | total anon GB |
|---|---|---|---|---|
| mapping (concurrent) | 0-2,930 | 22.9 (flat from the first minutes) | 1.4 -> 13.6 (grows with reads seen) | **36.6 at the end of mapping** |
| STAR exits, pileup finalizes | 2,930-3,340 | 3-5 (exiting) | 13.6 -> **32.0 transient** -> 25.8 | <= 36 |
| non-host screen | 3,340-3,470 | - | 26.9 anon **+ 24.3 file-backed** (mmap'd indices, `MinSketchIndex::load`) | RSS 51.3 |
| export | 3,470-4,145 | - | 28.0 -> 29.4 | 29.4 |

- **The 53.8 GB peak RSS is not a private-memory problem.** 24.3 GB of it is the non-host indices mapped read-only from a file; they are
  page cache (reclaimable, shared between jobs on a node) but `smaps` counts them as Private_Clean when one process maps them. The anonymous
  peak is **36.6 GB, at the end of mapping**.
- **Of that, 22.9 GB (63%) is STAR's genome index**, read into anonymous memory by each STAR process, not shared (the earlier "17.8 GB
  shared" premise does not hold in this setup). It is `src/star/`, S8's code. File-backed (mmap) or node-shared index would take the
  anonymous peak to about 14 GB on this deep sample (about 6 GB on a shallow one).
- singlet main's own state grows with depth (13.6 GB at 263 M reads, 5-7 GB on the 66-112 M samples) and then re-allocates during the
  end-of-stream finalize (+18 GB transient, 13.6 -> 32.0, then 25.8 retained). That transient stays under the mapping-phase peak, so it
  does not set the job size today, but it is the floor once the index is shared: then it is the peak.
- **Job size on Anvil `shared` (about 2 GB per core, per the brief; not verified against Anvil):** anonymous peak 36.6 GB -> 19 billed
  cores for a 4-thread job on the deep sample (shallow samples: 28-30 GB -> 14-15 cores). Nothing singlet-side changes this until the
  index leaves anonymous memory.
- **The finalize is probably a serial tail (thread CPU during it not yet checked):** main grows 13.6 -> 32 GB between t=3,088 and 3,341 (about 250 s, 6% of wall) after STAR
  has exited. That is step 3 (fill the tail) material, and where a memory cut inside singlet would come from.

## 4. Not measured

Depth hazards (lever 4); a repeat on exclusive nodes; Anvil (no runs, by instruction); the private/shared split of the 30-37 GB RSS; the 5'
sample class; more than 4 packed jobs per node.

## 5. Reproduce

`sbatch scripts/cpu_perf/build_cpu.sbatch` (`COMMIT=...`), then `scripts/cpu_perf/submit_matrix.sh`; `profile.sbatch` takes `SET T PACK BINV`.
`summarize.py RESULTS`, `stage_profile.py DIR M`, `threads.py DIR M` summarise (run them as a `debug-cpu` job, not on the login node).
