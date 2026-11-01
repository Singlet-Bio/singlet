# Notes for the fable GPU program, from the Beta Launch sessions

Written 2026-11-01 by Claude in Zach's Beta Launch Prep session (the account that built `--engine gpu` on `prod/campaign`: W11, S4, S8, S9, S14).
Zach asked for this: the fable program is now the line that will become the GPU aligner, so these are the things our sessions measured or
decided that the E1-E15 docs do not mention. Nothing here asks you to change direction. Where I computed a number instead of measuring it, it says so.

Your base `cf92cdb9` is our `gpu/gpu-slots2` merge of 2026-10-31 08:41. Everything below is `Singlet-Bio/singlet` unless a path says otherwise.

## 1. Which bound is binding (read this first)

Our node-level model is `docs/research-beta/gpu-profile-deep.md` section 10.2 (on `prod/campaign`). Per H100 node, per GPU: 28 host cores and one device-lock permit.

| quantity | value | source |
|---|---|---|
| measured campaign rate, 8 samples per GPU x 12 threads, 1 permit | 2.0-2.2 M pairs/s (read-weighted mix 1.97) | queue benchmark, `bench/engine_gpu_gate/s8_queue.sbatch` |
| host-CPU bound = 28 cores / (core-s per M pairs) | 1.75 at 16 core-s/M | same |
| device-lock bound = 1 / (lock seconds per pair) | 2.9-3.7 | same, lock busy 57-72% |

Before your work the host CPU was the binding bound, so kernel speedups (`k_p3`, `k_p1`, uploads) could not raise node throughput. Your level 5 at T = 18 is
564 core-s for 66.6 M reads, i.e. 8.5 core-s/M. **My calculation, not a measurement:** that is a host bound of 28 / 8.5 = about 3.3 M pairs/s, which meets
the lock bound (about 3.2). From here the next limit is **device-lock seconds per pair**, not wall time. So:

* Report the lock hold per sample (prefill + pileup + non-host screen, `lockDevice` hold) next to wall and core-s. E12's 38.5 -> 16.2 s device phase is the right
  kind of number; add the other holds. We measured 39.4 s per sample before any of this (24.9 s prefill, 9.7 s pileup, 4.9 s non-host).
* Re-measure under the queue benchmark (8 samples per GPU, 12 threads, one permit), not only single-sample T = 18 or T = 8. We found more samples per GPU (12) is slower, a second lock permit
  gives +4% for 2x the CPU, and NUMA pinning, OpenMP passive waiting and storage staging did nothing.
* Zach's rule for the campaign (2026-10-31): Clipper only, at most 3 concurrent H100s per user (`QOSMaxGRESPerUser`), no Anvil. Fewer lock-seconds per pair is what buys calendar time.

## 2. Memory and deep samples

The campaign admits samples against a fitted memory model (S14's drain worker, `scripts/atlas_rerun/multi_drain.py`): 17.5 GB + 0.39 GB per M reads on the FASTQ route, about 17.5 GB + 0.26 GB per M reads
on the `.1fg2` route. A node mixes several samples; the read-weighted mix peaked at 590 GB cgroup memory.

* Your ingest numbers (45.8 GB at 66.6 M reads, 60.4 GB for neuron1k_v3 at 92.9 M) are better than ours. They need a new fit, because the admission model is what stops the node being oversubscribed.
* The largest sample you gated is 118.7 M reads (nuclei2k_v2). We measured GSM7656149 (263.4 M reads, 135.9 M distinct): prefill hold 95.8 s, STAR child 120 GB high-water on the FASTQ route, 84 GB on `.1fg2`.
  Level 5 keeps whole-sample arrays alive through the prefill, so run level 5 + ingest on a 260 M-read sample before calling it general.
* 5' libraries cost more per read: GSM6470208 (121.8 M reads) writes 331.7 M alignment records against 73.9 M for pbmc1k_v3's 66.6 M reads (2.7 vs 1.1 per read), and STAR's adapter search runs for every read
  (`docs/research-beta/host-cost-model.md`). pbmc5p_v1 at 60 M reads does not stress that.
* Mouse spends about 38 core-s per M pairs in stitch and extend on the CPU lane (S12's profile). Your host-lane pool (E12) matters more there.

## 3. Tools that already exist

* `nsys` follows `exec`, not `fork`. The STAR child, where the prefill runs, is invisible to it. `ncu` counters do work on the Clipper nodes; `perf` does not (`perf_event_paranoid=4`).
* `SINGLET_ENGINE_GPU_TRACE=1` (branch `perf/nvtx-host-ranges`, commit e3432256): the transfer ledger, every `cudaMemcpy` by source line (calls, bytes, host seconds, GB/s, pinned or pageable) plus the GPU time of each launch.
  `SINGLET_NVTX=1` adds named host ranges (`include/singlet/nvtx.h`). `SINGLET_STAR_PHASE_LOG=1` (branch `prof/star-phase-marks`) logs wall, CPU time and RSS of the STAR child at each phase boundary.
* `bench/engine_gpu_gate/s8_stages.py` splits a run's log into stages and adds a 1 s cgroup CPU and `nvidia-smi` sampler.
* Per-sample overheads we saw in a 13 M-read run: `cudaHostAlloc` 8.3 s over 42 calls and `cudaStreamCreate` 7.2 s over 40 calls, made by the loader threads for every sample. Check whether your staged buffers (E12) pay this per sample; a pool avoids it.

## 4. Your E12 overlaps work we already had in flight

`perf/gpu-pinned-dedup` (PR #145, base `perf/gpu-instrumentation`) has pinned multi-thread uploads (653a75b2, `SINGLET_ENGINE_GPU_FASTUP=0` turns it off) and the per-read dedup pass on the device
(3a064a19, `SINGLET_ENGINE_GPU_HOSTDEDUP=1` keeps the host loops). E12 1c ("dedup products on the device", 3.2 s) and its staged uploads (2.6 s) are the same two changes. Both exist; pick one when merging
(they will likely conflict in `src/engine_gpu/engine_gpu.cu`) and keep the better number. They are not additive: our estimates for the lock were -3.4 s (dedup) and -3.1 s (uploads), and you measured 3.2 + 2.6 s on the same two changes.

## 5. `prod/campaign` has moved since your fork; merge it before the next gate

Your gate (a) compares against the CPU engine of your base. Since 2026-10-31 08:41 `prod/campaign` changed output definitions and the files you also edit (`src/star/SeqCacheGpuPrefill.cpp`, `src/pipeline/singlet.cpp`,
`src/star/ReadAlign_outputAlignments.cpp`). Items that touch your device paths:

* **Barcode universe (e1a4217b):** one universe for the stream and sorted path, whitelist barcodes with at least 5 reads after R11 correction, computed in pass 1. E10's device R11 histogram must follow it.
* **Order-free gene tracking of multimappers (90b93489, `note_gene`) and the seed-based UMI collapse (535f0127, R12, off by default).** Anything that combines evidence across reads must be order-independent: `docs/research-beta/order-free-rules.md`.
* **Counts definitions (0cd56886):** `exon_counts` / `intron_counts` are back to the STARsolo Gene / GeneFull split; the alevin-fry USA labels are their own matrices. The GPU pileup mirrors the USA labels and kFlank (03a5fac5), and releases the device lock when the device work is done (aba7f801).
* **Table-fed prefill (9038df44, f5909477):** the GPU engine reads a `.1fg2` table (distinct sequences come from the table, each clipped once, no temp FASTQ). Your E13 ingest feeds the prefill from `.1fq`.
  S9's staging produces `.1fg2` for the drain, so check which route the campaign ships and port E13's packed-buffer ingest to the table route if it is `.1fg2`.
* **Immutable engine binaries (11-01 06:20):** workers, the orchestrator and every sbatch refuse a shared mutable binary path. Fable binaries need commit-identified builds before they can run in the campaign.

## 6. Outputs Zach wants kept (decision of 2026-10-31)

"Keep every output we can" and the viral-read exception. On `prod/campaign` already: reads-per-UMI matrices and per-cell histogram (c552bcae, 2b7898f7), `nonhost_confirmed_reads` (a0b7da9b), chimeric table (`feat/chimeric-table`, S13).
Level 5 skips STAR's per-read loop for all but 0.24% of reads, and `src/star/SeqCache.h` states its exactness holds only with no chimeric detection. **I have not checked how the chimeric table or the TE counts are
produced**, so check whether any of these needs a per-read source that level 5 no longer has, and add them to gate (a) before the Beta Software Freeze.

## 7. Fallbacks and gate pitfalls

* Reads over 128 bases decline the GPU pileup, a forced decline must fall back to the CPU pileup with identical counts, and any CUDA failure releases the device and maps that sample on the CPU. Keep all three at level 5 and gate them
  (`bench/engine_gpu_gate/s8_gate_decline.sbatch`).
* The 10x whitelists must sit at `<binary dir>/../whitelists` or in `SINGLET_WHITELIST_DIR`. Otherwise the run silently takes the sorted-BAM path and the gate compares the wrong thing (job 793501 was discarded for this).
  Since 10-31 `provenance.json` records `pileup_path`, `pileup_path_reason` and `whitelist_missing`; assert on them in the gate.

## 8. Kernel experiments we already ran (so you do not repeat them)

On the unmodified `k_p3` (pbmc1k_v3, 66.6 M reads): bucketing by hit count ascending +2.7 s, descending 10.91 vs 8.72 s; persistent threads with an atomic read counter 8.64 s (no change, still 1.93 active threads per instruction);
launch bounds for 6 / 8 / 12 blocks per SM 8.70 / 8.61 / 9.97 s. Sorting by the true step count gives 99.9% warp efficiency but is not available before the kernel runs. This agrees with your E4 and E7 findings; the useful lever was
registers and batch barriers, as you found. Nsight on `k_p3` showed SM 7.7%, DRAM 7%, so it is not bound by either.

## 9. Lower priority

* **Resident indexes** (`gpu-profile-deep.md` section 13, `bench/engine_gpu_gate/ipc_test.cu`): the 16.9 GB suffix array (3.05 s) and the 24.9 GB non-host index set (4.1 s) are uploaded under the lock for every sample, about 7 s of 39.4.
  A per-GPU index server with CUDA IPC handles is designed and on hold. E14 removed the host-side reload only; the non-host set is untouched.
* **Cross-sample result store:** 49.5% of pbmc1k_v3's reads (19% of its distinct sequences) already occur in GSM7656149, same chemistry; 0 across read lengths. STAR maps from the sequence alone, so a store of mapped distinct
  sequences is exact within one genome index and parameter set. With M5 now a small part of the run, it is optional.
* **Break-even:** the CPU-node break-even is 12.7 core-s per M reads (`host-cost-model.md`). Level 5 at 8.5 is under it.

## Standard we used

For every remaining segment: at a hardware bound within about 20%, or justified in writing (`gpu-profile-deep.md` section 10). Zach asked for "provably optimal" on 2026-10-31, so the E-docs would benefit from a table in that form for
the lock-held phases once the other holds are measured.
