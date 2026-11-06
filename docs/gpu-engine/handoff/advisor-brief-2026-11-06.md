> Archived copy. Paths under a local scratchpad referred to the advisor package at the time; its contents are summarized in HANDOFF-2026-11-06.md and the track summaries on Clipper.

# Advisor brief: GPU STAR + pileup pipeline, results of the 2026-11-05 overnight campaign and the plan ahead

## Goal

The fastest possible pipeline that turns many single-cell samples (10x 3'/5', nuclei, CITE, multiplexed donors) into Singlet `.s1` objects on one node with 2 H100 NVL GPUs (56 cores, 768 GB host RAM) on the Clipper cluster. Stages per sample: FASTQ ingest, barcode/UMI, STAR 2.7.11b alignment (CUDA port, `--engine gpu`), pileup (counts, junctions, heteroplasmy), non-host export, donor demultiplexing, cell calling. The aligner must stay byte-identical to the CPU engine. Iterative floating-point stages, such as donor demux, are accepted by a ground-truth tolerance instead.

## The engine in one paragraph

One process per sample. Reads collapse to distinct sequences, then go through three device lanes: A light (`k_p1s`, `k_p3lb`), M medium (warp per read, `k_p3a`/`k_p3w`), B heavy (`k_p3e`, thread per read, 2.2% of reads and 39% of prefill kernel time). Lane B overflow, judged by a step budget, goes to a host pool that runs CPU `mapRead`. A device lock with 2 slots per GPU serializes the device phase of the processes. Each process uploads its own 16.4 GB genome index. Throughput at k=8 samples in flight, T=7 threads per sample, 2 slots: about 1.75 M read pairs/s per node on g050 (about 2.4 on g051 for the same binary: node effect, so only arms inside one job are compared). At that point 16 to 21 of 28 cores are busy and processes wait on the lock for 30 to 45% of wall. Prior profiling (folder `../advisor/`: `context.md`, `ncu.txt`, `ncu_source.txt`, `nsys.txt`, kernel source under `kernel/`) found lane B at 0.31 waves and 1.9 of 32 active lanes per warp, limited by 549 KB of per-read Work. You ranked MPS, a shared index over CUDA IPC, a batching service, then kernel fixes. That roadmap drove last night.

## Results (full summaries in `summaries/`, rates are progress-weighted M pairs/s)

| Track | Result | Verdict |
|---|---|---|
| e20w host tuning (fast lane) | core-s/M -12.4%, lock held -15%, rate +5.5%, identity 4/4, no fallbacks | Promote |
| e21w (fast lane) | rate +7.9% within noise, core-s and lock flat | Inconclusive |
| WSHRINK: lane B per-batch WA pool, Work 549 KB to 157 KB | lane B batch 31,744 to 112,640 reads, `k_p3e` device time 4.44 to 2.23 s, registers 80 to 52, grid 231 to 626 blocks; outputs identical except a whitelist path string in provenance.json | Strong; timing rerun running (jobs 800764 2 slots, 800765 MPS 3 slots) |
| BSTREAM: lane B on its own stream | no change in kernel time | Dropped |
| MPS | 2 slots: +21% in one A/B/A, +0% in another. k=12, 3 slots: 2.32 with MPS against 2.19 without (+6%). Identity holds under MPS | Small gain; most of the lift comes from 3 slots |
| Budget 91 vs 45 GB (solo) | lane B 2.7 vs 5.3 s, hold 18 vs 21.5 s | Batch size matters; WSHRINK gets it without memory |
| Lane B cost sort | k_p3e 5.07 vs 4.45 s; lane B total 14.7 vs 9.0 s | Rejected |
| 64-register cap | 4.53 vs 4.45 s | Rejected |
| Admission control | A/B/A 1.86, 1.51, 1.63 | Rejected |
| Shared index (holder process, `cudaIpcOpenMemHandle` with LazyEnablePeerAccess) | identity passes in every comparison with and without MPS. Under load (k=8, 2 slots) 0.32 vs 2.41 for b0 in the same job: laneA 500 s vs 6.8 s, laneM 237 vs 6.0 s, laneB 48 vs 9.7 s per run, GPU util 89-100%, holder idle | Bug under investigation by a debug agent (solo vs shared, pointer attributes, ncu PCIe/NVLink bytes, D2D-copy control) |
| Heavy-read CPU/GPU split (STEPS_B 256/512/128, BHOST, auto) | identity runs differ only in same-size unmapped/non-host files (likely order); timing never ran (commit check) | Rerun queued (800769 to 800774) |
| Code-path corpus (16 inputs) | ran only for the shared-index build: strand100 NOT IDENTICAL (strand restart bug, now fixed in PR 194), v5p150/mask3syn/long_q4 fail path rule with disallowed fallbacks (long reads above 128 bases), others pass | Reruns queued for kern2 and split |
| Read collapse and cache (8 samples, CPU truth) | in-sample distinct fraction 31 to 68%. Exact cross-sample cache by full sequence: cost-weighted hit 23 to 37% for 3' blood and cell lines, 12% brain, 3% 5', 0.3% nuclei, still rising with pool size. Caveat: pbmc10m is a subsample of pbmc1k_v3, which inflates the reported 29% mean to about 17-19% without it. Hamming grouping (prefix25, HD<=2, verified) saves 7-9%. Top-1e6 sequence table 2.8%. Barcode+UMI grouping -4.5%. k=25 k-mer classifier for mito/ribo/MALAT1: precision 90-96%, recall 33-56%, changes output | Exact cache is the only exact idea worth building |
| Transcriptome-first (STAR transcriptome pass, 1 mismatch, one gene) | full match with the genome record for 29-32% of all reads (PBMC, neurons), 0% of heavy reads. Nuclei sample invalid (prep processed 15,888 reads) | Little gain where time goes |
| Two GPUs per node baseline | never ran (commit check); queued behind the timings as 800789 | Pending |
| Donor demux ground truth (in-silico mixtures of 4 real single-donor samples at SNP-count level, 2/3/4 donors, 2/5/8% planted doublets) | K given: singlet accuracy 0.51-0.68 for 2 donors (ARI 0 to 0.12), 0.62-0.94 for 3, 0.73-0.99 for 4. Deterministic across repeats and threads; a different seed changes up to 47% of assignments. Auto-K wrong on all 9 (K=1 for 2 donors, K=2 for 3 and 4). Doublet AUROC 0.39-0.48 and no doublet calls. Runtime 5 to 130 s | The demux algorithm is weak; GPU port deferred |

Profiling on hand: `../advisor/ncu.txt` and `ncu_source.txt` (b0 lane B source-level stalls), `../advisor/nsys.txt`, the kern2 ncu table in `summaries/gpu-kern2.md`, lane step times per run in `summaries/gpu-shidx.md` (arm A b0 vs arm B shared), stage core-s per run in `summaries/gpu-fast.md` (scan 28, refs 91, STAR 88 to 38 with e20w, stream 170, handoff 23, nonhost 45, export 37, demux 142 core-s per run of pbmc1k_v3). Note demux at 142 core-s is the largest single CPU stage after stream.

## Plan going forward (what I intend to do; critique it)

1. Merge the promoted pieces into one integration branch once identity and corpus hold: e20w, WSHRINK, strand-restart fix (PR 194), split auto budget if its timing wins. Then one combined exclusive timing at 2 and 3 slots, with and without MPS.
2. Fix the shared index. With it, each GPU holds one index, which frees 16 GB per extra process and allows 3 to 4 samples per GPU in normal mode. Then re-time 3 and 4 slots with MPS.
3. Use both GPUs from one node scheduler (one lock per GPU is in place; the 2-GPU baseline measures scaling).
4. Kernel work on lane B after WSHRINK: re-profile; candidates are genome reads through `__ldg`/texture path for extendAlign, warp-cooperative extension for the longest reads, and a persistent lane B kernel fed by a device queue.
5. Exact cross-sample alignment cache keyed by the full read sequence (128-bit hash plus a length and verification), built per chemistry and tissue, consulted before the device lanes. Start host-side; consider an atlas-scale shared store later.
6. Batching service: one long-lived process per node owns the GPUs and the index and takes distinct-sequence batches from per-sample front ends. This would also give cross-sample dedup in one batch.
7. Donor demux: fix the algorithm on the CPU first (several seeded restarts with the best likelihood, better initialization, a doublet term, a real auto-K criterion), benchmark it against the ground-truth mixtures, then port to the GPU under a tolerance.
8. Device cell counting and pileup tail after that.

## Questions for you

1. Is this the best plan for maximum multi-sample throughput per node? What would you reorder, add or drop? Where is the actual ceiling, and which bound (device lock, host cores, memory, I/O) binds after WSHRINK, shared index and e20w?
2. Read the shared-index symptom and code (`src/engine_gpu/shared_index.cuh` and the `SHC.on` branch of `startIndex` on branch `gpu/bold-shidx` in the repo at `C:\Users\zacha\Projects\singlet-bio\.claude\worktrees\gpu-fast`, readable with `git show origin/gpu/bold-shidx:<path>`). What is the most likely cause of a 75x slowdown on light lanes with identical output?
3. Donor demux: what is wrong with the current approach (`demux/donor_demux.h`, benchmark in `demux/`), and how should it be done? Is the benchmark itself sound (mixing at SNP-count level, donors S1-S4 from GSM4006843, GSM7474576, GSM7901264 and pbmc1k_v2)? What is the state of the art (Vireo, souporcell, demuxlet/freemuxlet, scSplit, cellsnp-lite, Demuxafy ensembles, newer tools), and how should we use them: as a ceiling in the benchmark, as a reference implementation to port, or in the pipeline? Licensing matters: the pipeline is open source and its outputs are public, and genotypes must never be stored in public objects or served.
4. Read collapse and cache: is the exact cache worth building given the numbers, and how would you design it so the aligner output stays byte-identical?
5. Anything in the measurement method you would change (node effect, A/B/B/A design, progress-weighted rate, tolerances)?

Keep the advice concrete: name experiments, expected effect, and the go/no-go rule for each. Rank by expected throughput gain per hour of work.
