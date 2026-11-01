# E16: the level-5 tail (tail)

Experiment E16 ("tail") of the GPU aligner program. Branch `gpu/fable/tail`, from `gpu/fable-engine` at 8e54be40, with
`gpu/fable-engine` merged in twice: at 09e74739 (E14, "probe") and at d3a335b8 (E12, "prefill", and E13, "ingest").

E16 shortens the stretch of a level-5 run (`SINGLET_ENGINE_GPU_E2E=5`, `--engine gpu`, `SINGLET_GPU_PILEUP=on`) that starts when
the prefill releases the device and ends when the GPU pileup's results are in the accumulators. This document calls that stretch
"the tail". It was 34 s on pbmc1k_v3 at T = 18 in E11's gate 796433. The target was 15 s or less, with every output byte-identical
(gate (a)).

All changes are active only at level 5 and can be turned off:
- `SINGLET_E16=0` turns every cut off, and the tail then runs as E11 left it. This is the gate's baseline switch.
- `SINGLET_E16_OFF=name,name,...` turns single cuts off (names in §2b).
- `SINGLET_E16_TRACE=0` turns the `[e16]` timing marks off. They cost one stderr line each and are printed only at level 5.

The CPU engine, the default GPU path and levels 1–4 run the code they ran before (§3c).

## 0. Answer in brief

- **The tail is 8.8 s instead of 36.0 s on pbmc1k_v3 at T = 18** (gate 797132, binary b14fb05f, node g050). Both figures are from
  the same binary in the same job: `SINGLET_E16=0` for "before", two runs for "after" (8.79 s and 8.82 s). The target was ≤ 15 s.

  | piece of the tail (the brief's four parts) | before | after | main cuts |
  |---|---|---|---|
  | STAR from the prefill's end through its loop on the 157 k reads without a record | 9.79 s | 1.94 s | cache, sumthreads, bg, free, chunkpar, chunk |
  | stream end, Solo and STAR's exit (loop end → the pipeline's stream drained) | 3.14 s | 0.01 s | eof, hdr |
  | hand-off pass, input build, `pileupRun`, puts and host pass | 23.09 s | 6.84 s | early, zerocopy, digest, aasplit, regpar, dlpar, rrnabg, resetbg, overlap, … |
  | finalize (collapse) | 0.11 s | 0.11 s | |
  | **tail** | **36.03 s** | **8.79 s** | 22 cuts (§2b) |

  The brief's 34 s (9 + 8 + 16 + 1) came from E11's stage table at 1 s resolution. Here the same stretch is measured with
  timestamped marks from both processes (§2a): 36.0 s before the cuts. The stage table puts the boundaries between the stream and
  the hand-off pass elsewhere, hence its different split.
- **The run: 153.8 s instead of 180.5 s (−26.7 s, −14.8 %)** with the same binary and E14 in both (mean of 154.4 s and 153.2 s).
  Core-s fell from 662 to 650 (−12, −1.9 %). Against E11's final binary in the same job (213.6 s, 213.3 s), E14 and E16 together save
  −59.7 s (−28.0 %) and −35 core-s. E16 is cutting waits, not work, so its core-s saving is small.
- **Gate (a) holds (§3).**
  - **CPU engine vs level 5, final binary:** 60 identical, 2 differ in timings or paths only, 7 differ. The 7 are the same files
    that differ between the CPU engine and the default GPU path in the same job, i.e. E11 §2a's run-dependent files:
    - record order in `nonhost_reads.fq` and `star_Unmapped.out.mate1/2` (same sorted records);
    - timings in `read_stats.tsv` and `star_Log.std.out`;
    - `pileup_stats.json` and `summary.json`'s two GPU-engine fields.
  - The following are identical: `Log.final.out`'s mapping statistics, `SJ.out.tab` (126,996 junctions), and
    `Gene/CellReads.stats` (323,542 rows, compared sorted).
  - **The default GPU path and levels 1, 3 and 4** against E11's kept outputs: 60 identical each (§3c).
- **After merging E12 and E13 (§3d, §4b):** with E12's prefill in the binary, level 5 runs in **127.2 s and 127.7 s**,
  against 153.5 s with `SINGLET_E16=0` in the same job (−26.1 s, −17.0 %) and 213.7 s for E11's binary (−86.3 s, −40.4 %).
  The tail is 8.88 s and 8.85 s against 34.90 s. Gate (a) is the same 60 / 2 / 7 (job 797377, binary d3a335b8).
  - **With E13's ingest** (`SINGLET_ENGINE_GPU_INGEST=1`) that gate found a crash. STAR's child died (SIGSEGV) because E13
    releases the packed reads at the prefill's end, while E16's background thread was still formatting Unmapped.out from them.
  - The fix (d7aafb2c) keeps the packed reads until that thread is joined. With it (job 797398, node g051), level 5 + ingest
    + E16 runs in **101.1 s and 105.1 s** (548 and 566 core-s), against 126.9 s (562 core-s) with `SINGLET_E16=0`: −23.8 s
    (−18.8 %). The tail is 8.83 s and 9.17 s against 34.40 s, and gate (a) holds (CPU vs level 5 + ingest: 60 / 2 / 7, the
    same 7).
- **Where the 27 s came from.** In order of size:
  - the hand-off pass and the input's build no longer copy the region (zerocopy) and no longer wait for STAR (early):
    11.4 s → 1.2 s;
  - `pileupRun` lost its digest (2.5 s), the per-alignment download shrank (aasplit, 1.3 s), and its uploads read page-locked
    memory in place (0.8 s): 9.2 s → 4.0 s;
  - STAR's loop uses all 18 threads instead of 2 (chunk) and its chunks are built in parallel: 5.1 s → 0.9 s;
  - the pipeline now sees the end of the BAM stream when STAR closes it, not when STAR exits (eof): 3.1 s → 0.01 s;
  - the prefill's tail no longer re-creates a 12 GB cache, formats Unmapped.out, or frees GBs on the critical path:
    4.7 s → 1.0 s.
- **What is left (§6)** is mostly device work and its host copies:
  - the early hand-off pass (2.1 s, which the pipeline waits on for 1.0 s);
  - `pileupRun`'s GPU time (2.2 s) and its fetches between partitions (0.8 s);
  - the accumulator puts beside the host pass (1.7 s);
  - the prefill's remaining tail (1.0 s).

## 1. The tail, step by step

Gate 797132, T = 18, node g050, binary b14fb05f. "Before" is `SINGLET_E16=0` (run gpuE5x3) and "after" is the default (gpuE5).
Seconds are counted from the prefill's device release (`mapDistinct` returned). The table comes from
`bench/fable/tail/e16_steps.py` on the runs' logs.

| step | before | after | cut |
|---|---|---|---|
| **STAR's child, prefill's end → loop end** | **0 → 9.79** | **0 → 1.94** | |
| aux free, sequence cache re-created (12 GB, its 1.5 GB index zeroed) | 1.64 | — | cache |
| read → record map copied into the region, summary threads (18 threads, no summaries to make) | 0.36 | 0.54 (the aux free and the copy) | sumthreads |
| alignment counts, region completed (magic written) | 0.17 | 0.28 | |
| Unmapped.out: 3.3 M records formatted and written | 0.91 | — (thread, 0.72 s beside the loop) | bg |
| read source of the 157 k reads without a record | 0.22 | 0.23 | |
| RNA Q30 adjustment | 0.21 | — (same thread, 0.46 s) | bg |
| prefill's end → prefill returned (the decode's and the engine's buffers freed) | 1.19 | — (freed on a thread after the loop, 1.31 s) | free |
| STAR's 18 chunks constructed | 1.34 | 0.21 | chunkpar |
| STAR's loop on 156,931 reads | 3.76 (2 of 18 threads got reads) | 0.68 (18 of 18, 72 chunks) | chunk |
| **loop end → stream drained** | **3.14** | **0.01** | eof |
| the pipeline's BAM header read → references resolved (7 M SNPs) | at 8.58 → 9.49 | at −55.96 → −55.06 (during the prefill) | hdr |
| first BAM record → stream drained | 9.50 → 12.93 | 1.65 → 1.95 | |
| STAR's child: after-mapping steps, frees, `SJ.out.tab`, exit | 9.79 → 11.94 (the pipeline waits for it) | 1.94 → 6.54 (beside the pileup) | eof, solofirst |
| **stream drained → GPU pileup applied** | **12.93 → 36.03** | **1.95 → 8.79** | |
| hand-off pass after the drain | 5.88 | 1.05 (0.96 waiting for the early pass, 0.09 merge) | early |
| … `e2ePass` (device lock 0.18 s before it) | 3.64 | 2.07, from 0.84 s on, beside STAR's loop | early, regpar, dlpar |
| … rRNA detector | 0.31 | — (thread, 0.43 s beside `pileupRun`) | rrnabg |
| … kept reads appended to the collector | 1.60 | 0.09 | zerocopy |
| build input (collector: region copied, 4 copies of 0.6–1.5 s, unmap 1.08 s) | 5.50 | 0.12 | zerocopy |
| `pileupRun` | 9.24 | 4.01 | |
| … context created | 0.14 | — (the early pass's context) | zerocopy |
| … records H2D | 1.08 | 0.29 (page-locked region read in place) | zerocopy |
| … `run_pipeline` (GPU 2.16 s) | 4.75 | 3.22 | aasplit, fetchpar |
| … of which per-alignment results D2H and merge | 1.50 | 0.21 | aasplit |
| … of which host fetches between partitions | 1.08 | 0.84 | fetchpar |
| … digest of every output (4.3 GB) for a log line | 2.50 | — | digest |
| … outputs copied out of the run | 0.37 | 0.06 (+ 0.10 join at the end) | copypar |
| … device reset | 0.11 | — (deferred: 0.38 s on the free thread) | resetbg |
| results put into the accumulators | 1.03 | 1.01 | puts |
| host pass | 1.10 | 0.65 after the puts (they run side by side: 1.66 s for both) | overlap |
| input and output freed | 0.35 | — (thread, 1.36 s) | free2 |
| **tail** | **36.03** | **8.79** | |
| finalize (collapse) | 0.11 | 0.11 | |

The second level-5 run (gpuE5b) gives the same steps within 0.03 s and a tail of 8.82 s.

## 2. What was done

### 2a. Instrumentation (`include/singlet/tail_trace.h`)

`singlet_tail::mark(fmt, ...)` writes one stderr line `[e16] <epoch s.ms> <pid> <text>`. STAR's child and the pipeline share
stderr, so the job's log orders both processes on one clock. Marks are printed only at level 5. The places marked:
- STAR's child: the prefill's tail (cache, summaries, region, Unmapped.out, read source), the chunk construction, the loop (per
  thread: start, first chunk, last chunk mapped, chunk loop left, BAM buffer flushed, chunk and read counts), the BAM close (with
  the processes that still hold the pipe's write end, read from `/proc`), the after-mapping steps, the free of the genome and the
  cache, `SJ.out.tab`, the Solo constructor, the exit;
- the pipeline: the BAM header, the first record, the drain, the hand-off pass (device lock, `e2ePass`, rRNA detector), the
  input's build, `pileupRun`, the accumulator puts, the host pass, the finalize, the reap;
- `e2ePass` and `pileupRun` append their own host-side step timers to their logs (context, register, downloads, reset).

`bench/fable/tail/e16_steps.py RUN_DIR` turns a gate run's log into the step table of §1. It leaves out the marks of the CR4
probe's STAR runs, which the child forks and which print the same marks.

`bench/fable/tail/s8_stages_e16.py RUN_DIR READS_M` is E11's whole-run stage table (§4), adapted to E16 runs:
- the prefill's end is the `prefill: mapDistinct returned` mark, because the early hand-off pass takes the device within a second
  and the device memory no longer drops;
- the stream stage starts at the end of STAR's loop, because the references are resolved during the prefill.

### 2b. The cuts

In the order they were made. Each name is a `SINGLET_E16_OFF` switch. "Effect" is the change of the step it targets at T = 18,
from §1's runs unless a job is named.

| cut | where | what | effect |
|---|---|---|---|
| hdr | `samHeaders.cpp`, `singlet.cpp` | STAR flushes the unsorted BAM's header block to the pipe as soon as it is written. `bgzf_flush` only hands the block to htslib's hFILE buffer, so `singlet_bgzf_hflush` (built in the pipeline against the linked htslib) pushes it to the fd. The BAM's decompressed bytes are the same. | the pipeline resolves its references (0.9 s) 56 s before the prefill ends instead of after STAR's loop |
| chunk | `SeqCacheGpuPrefill.cpp` | E10's read source serves at most ⌈n / (4 × threads)⌉ reads per chunk (2,180 here). Before, the first threads to ask took all 157 k reads. | loop 3.76 → 0.68 s; 2 → 18 threads with reads |
| sumthreads | `SeqCacheGpuPrefill.cpp` | When no summaries are needed (level 5), the 18 summary threads are not started. Each built a `ReadAlign`, a `Transcriptome` copy and an `OutSJ` (leaked) and mapped nothing. | with cache: 2.00 → 0.54 s |
| early | `pileup_engine.h` | The hand-off pass's device part (`e2ePass` and the rRNA detector) runs on a thread once the prefill completes the shared region (magic written). It reads the region through its own mapping while STAR maps and the stream fills the collector. Its results are merged after the stream, as before. | `e2ePass` off the critical path; hand-off pass after the drain 5.88 → 1.05 s |
| zerocopy | `pileup_engine.h`, `gpu_pileup_bridge.h`, `engine_gpu.h`, `records_gpu.cu`, `pileup_gpu.cu` | The early pass keeps the region and the kept reads page-locked in a live device context. The pileup's input reads them in place (`PileupInput::Pre` parts) instead of copying ~6 GB into vectors. `pileupRun` uploads `[pre | vector]` into the same arena arrays, and the host pass reads through `PileupCat` views. | build input 5.50 → 0.12 s; kept reads appended 1.60 → 0.09 s; records H2D 1.08 → 0.29 s; no second context. The reset now unpins the memory (0.11 → 0.40–0.49 s; see resetbg) |
| bg | `SeqCacheGpuPrefill.cpp` | At level 5 the 3.3 M Unmapped.out records of reads with an unmapped record are formatted, and E11's RNA Q30 adjustment runs, on a thread (4 threads under it) started with the read source. `seqCacheGpuAfterMapping` joins it and appends the records after STAR's loop. | 1.12 s off the prefill's tail |
| cache | `SeqCacheGpuPrefill.cpp` | No 12 GB sequence cache is re-created at level 5. No read with a record looks it up or inserts into it, and STAR's start-up cache serves the 157 k reads without a record. **Superseded by E12** (d3a335b8): E12 makes that cache only when results are inserted, which covers every level ≥ 3. | 1.64 s → 0 (with the aux free) |
| digest | `pileup_gpu.cu` | `pileupRun` no longer hashes every output tuple (4.3 GB of chrM tuples) for its log line. | 2.50 s → 0 |
| free | `SeqCacheGpuPrefill.cpp` | The decode's reads and the engine's output (the prefill's `Input`/`Output`) are kept until STAR's loop is done, then freed on a thread in `seqCacheGpuAfterMapping`. The first version freed them on a thread at once; see §5. | prefill returned 1.19 → 0 s |
| free2 | `pileup_engine.h` | The pileup's input (with the region's second mapping) and output are freed on a thread. | 0.35 s → 0 (1.16 s in 796702, where the step also unmapped zerocopy's mapping) |
| puts | `pileup_engine.h` | The 7 accumulator puts run one thread per accumulator, each filling its own vectors in the same order. | 1.08 → 0.87 s (796702 → 796863) |
| aasplit | `pileup.cu`, `pileup_gpu.cu` | The per-alignment results are split on the device before their download: 9 bytes per alignment instead of 48. | merge + per-alignment D2H 1.50 → 0.21 s |
| solofirst | `SeqCacheGpuPrefill.cpp` | After STAR's loop, Solo's device step goes first, before the Unmapped.out append. It then takes the device ahead of `pileupRun` (which holds it 4 s) and waits only for the early pass. | the child's Solo is done at 3.0 s instead of after `pileupRun`; not on the tail's path once eof is in |
| chunkpar | `STAR.cpp`, `ReadAlignChunk.cpp` | STAR's `ReadAlignChunk` objects are constructed one per thread. Their only shared write, `Log.out`'s "Opening the file" lines, takes a lock. | 1.34 → 0.21 s (1.22 s in 796863, where the prefill's free ran beside it) |
| eof | `STAR.cpp` | After the unsorted BAM's close, stdout is pointed at `/dev/null`. The linked htslib's `bgzf_close` leaves fd 1 open, so the pipeline saw the end of the stream only when STAR exited. | loop end → drained 3.14 s (4.66 s in 796863) → 0.01 s |
| overlap | `pileup_engine.h` | The accumulator puts run on a thread beside `gpu_host_pass`, which reads other parts of the output and writes other members. | puts + host pass 2.04 → 1.70 s (x1 → x2) |
| fetchpar | `pileup.cu`, `pileup_gpu.cu` | The pileup's outputs between partitions (junctions, SNPs, mt, indels) are fetched on threads, each output vector by one thread. | host fetch 1.11 → 0.84 s |
| copypar | `pileup_gpu.cu` | The 8 tuple copies out of the run go on threads, joined after the device's release. | outputs copied 0.37 → 0.06 s |
| rrnabg | `pileup_engine.h` | The rRNA detector (on the 0.93 M sampled reads the pass sends to the host) runs on a thread after `e2ePass`, beside the merge, the input's build and `pileupRun`. Its count is added after the host pass. Its thread is joined before the region mapping it reads goes away, on every return path. | 0.43 s off the early pass |
| dlpar | `records_gpu.cu` | Before the kept reads' download, their three arrays (61 M reads, ~2 GB) are reserved and first touched by up to 16 threads. `resize` then fills pages that already exist. | download 1.51 → 0.79 s (touch 0.15 s) |
| regpar | `records_gpu.cu` | Before `cudaHostRegister`, this process's mapping of the region (5.3 GB) is read one byte per page by up to 16 threads. | register 1.00 → 0.49 s (touch 0.03 s); the region's upload also fell, 1.34 → 0.86 s |
| resetbg | `pileup_gpu.cu`, `pileup_engine.h`, `engine_gpu.h` | `pileupRun` leaves its context in place (`PileupOutput::resetDeferred`). The free thread of free2 resets it (`pileupRelease`), unlocks the device, then frees the input whose memory was page-locked in it. | reset 0.49 s → 0 (0.38 s on the thread) |

How the tail fell, build by build (T = 18):

| build | cuts on | tail | job |
|---|---|---|---|
| bin-c3 (eb67bfbe), `SINGLET_E16=0` | none | 34.73 s | 796702 |
| bin-c3 | hdr (not yet reaching the fd), chunk, sumthreads, early, zerocopy, bg, cache | 22.50 s | 796702 |
| bin-c6 (c824d5f1) | + hdr fixed, digest, free (at once), free2, puts, aasplit, solofirst, chunkpar | 14.79 s, 14.62 s | 796863 (node g051) |
| bin-c11 (b14fb05f), `SINGLET_E16_OFF=overlap,fetchpar,copypar,rrnabg,dlpar,regpar,resetbg` | + eof, free (held until after the loop) | 11.67 s | 797132 |
| bin-c11, `SINGLET_E16_OFF=rrnabg,dlpar,regpar,resetbg` | + overlap, fetchpar, copypar | 10.79 s | 797132 |
| bin-c11 | all 22 | **8.79 s, 8.82 s** | 797132 |
| bin-c11, `SINGLET_E16=0` | none | 36.03 s | 797132 |

At T = 8 on debug-gpu (H100 NVL, 8 CPUs), bin-c3 took 23.67 s (796703) and bin-c6 16.84 s (796862).

### 2c. What the cuts did not change

- No output is computed differently. Each cut does one of three things:
  - moves work to another thread, with each output vector still filled by one thread in the same order;
  - drops work whose result was never used (the digest, the summary threads, the second cache);
  - changes when bytes reach a pipe or a page (the header flush, the stdout redirect, first touches).
- `Unmapped.out.mate1/2` were already written in a run-dependent record order (E11 §2a). The bg cut appends the 3.3 M records of
  reads with an unmapped record after STAR's loop instead of before it, so their position in the file changes. The gate compares
  these files as sorted records, as before.
- STAR's own files are unchanged: `Log.final.out`'s mapping statistics and `SJ.out.tab` are identical.
- Two things are visible in logs only:
  - the pileup's digest line reads "digest skipped (E16)" at level 5. Gates that compare that digest (E13 §2, the
    partition-invariance gate) need `SINGLET_E16_OFF=digest`;
  - the hand-off pass's log line says the rRNA detector ran "on a thread (E16)". The count is reported in its own line after the
    host pass.

## 3. Identity per output

### 3a. Gate (a): CPU engine vs level 5 (job 797132, binary b14fb05f; `bench/star_accel/out_equiv.py` + the gate's own checks)

| outputs | result |
|---|---|
| count matrices and pileup products (`exon_counts`, `intron_counts`, `ambiguous_counts`, `sj_counts`, `splice_psi`, `polya_counts`, `mt_heteroplasmy`, `ambient_corrected_counts`, `vdj_gene_usage`; `private/snp_ad`, `snp_dp`, `mt_allele_counts(_rev)`; `mt/mt_events`) | identical |
| tables, VCFs, calls, `rrna_report.json`, `sex_call.json`, `ancestry_call.json`, `cr4_probe` files | identical |
| STAR Solo: the 10 files of E11 §2a | identical |
| `Gene/CellReads.stats` | same 323,542 rows (sorted) |
| `cell_qc_metrics.tsv`, `read_qc_summary.json` | identical |
| `star_SJ.out.tab` | identical (126,996 junctions) |
| `star_Log.final.out` | mapping statistics identical |
| `nonhost_reads.fq`, `star_Unmapped.out.mate1/2` | same records (sorted sha256 c115c73474e99993, 766e191fcc6181a5, 31fc9ef1d3d55a89; 428,468 and 3,355,559 reads); order differs |
| `provenance.json`, `ambient_contamination.json` | timings/paths only |
| `read_stats.tsv`, `star_Log.std.out` | stage rows and timings |
| `pileup_stats.json` | `clip5p/probe/n_reads` 12,194 (CPU) vs 12,000 (GPU engine), `bytes_public` |
| `summary.json` | `memory_tier` Medium (CPU) vs Large (GPU engine) |

`out_equiv`'s count: **60 identical, 2 timing/path only, 7 differ**. The CPU engine vs the default GPU path (gpuD) in the same job
gives the same 60 / 2 / 7, with the same 7 files.

Every intermediate gate gave the same verdict:
- 796702 (bin-c3): CPU vs level 5, and CPU vs level 5 with `SINGLET_E16=0`, each 60 / 2 / 7.
- 796862 (bin-c6, T = 8) and 796863 (bin-c6, T = 18): level 5 against 796702's kept CPU outputs, 60 identical each, and
  CellReads.stats with the same rows. The kept copy lacks the three order-only files.

### 3b. Level 5 against itself (same job)

gpuE5 vs gpuE5b: 60 identical, 3 timing-only, 6 differ. The 6 are:
- read order in 3 files;
- timings in 2 files;
- `pileup_stats.json`'s `bytes_public`.

`CellReads.stats` is byte-identical in file order.

Each cut set against the full set (gate 797370 on 797132's kept outputs: gpuE5x3, gpuE5x1 and gpuE5x2 against gpuE5): 60 identical,
3 timing-only, 3 differ (`bytes_public` and the 2 timing files), and `CellReads.stats` byte-identical in file order. The kept
`out/` copies leave out the 3 order-only files.

### 3c. Default paths and levels 1–4

- **Default GPU path** (797132): gpuD vs E11's kept gpuD outputs (796377): 60 identical. The rest:
  - 3 order-only files that E11 did not keep;
  - `bytes_public`;
  - `provenance.json`'s whitelist path;
  - timings in `read_stats.tsv` and `star_Log.std.out`.

  CellReads.stats has the same rows.
- **CPU engine** (797370: 797132's CPU run against E11's kept one): 60 identical, 2 timing/path only, 4 differ:
  - `bytes_public`;
  - the whitelist path in `provenance.json`;
  - timings in `read_stats.tsv` and `star_Log.std.out`.

  CellReads.stats has the same rows.
- **Levels 1, 3 and 4** (gate 797370, binary b14fb05f, node g052, against E11's kept gpuE, gpuE3 and gpuE4 from 796377): each gives
  60 identical, 2 timing/path only, 7 differ. The 7 are:
  - the 3 order-only files that E11 did not keep;
  - `bytes_public`;
  - the whitelist path;
  - timings in 2 files.

  `Log.final.out`'s mapping statistics and `SJ.out.tab` are identical. CellReads.stats has the same rows: 323,542 at levels 1 and
  3, and 25,754 at level 4, where it is byte-identical in file order.

  Wall times: level 1 217.9 s, level 3 220.3 s, level 4 177.7 s (E11: 251.8, 252.5 and 209.5 s, before E14).

None of the cuts can run below level 5: `singlet_tail::cuts()` requires `SINGLET_ENGINE_GPU_E2E` to start with 5. Code that was
restructured on shared paths keeps its order and work when its cut is off:
- the fetch lambdas in `pileup.cu`;
- `applyOut` in `pileup_engine.h`;
- `gpu_rrna_detect`, which is the same detector loop moved into a member function.

### 3d. After merging E12 and E13 (jobs 797377 and 797398)

Merging `gpu/fable-engine` at fe97c8b4 brought in E12 (the prefill's device phase) and E13 (ingest, off by default). Its
`SeqCacheGpuPrefill.cpp` conflicted in four places:
- E12 makes the sequence cache only when results are inserted. At E10 level ≥ 3 nothing is inserted, so that covers E16's
  "cache" cut, which was dropped. Its name is now a no-op;
- E12 moved `deferInsert` below the insert lambda;
- E12's step timer and E16's mark after the summary threads were both kept;
- E13's `readSourceFree()` was combined with E16's chunk cut.

**Gate 797377** (binary d3a335b8, node g050, T = 18; modes gpuE5 base5 gpuE5x3 gpuE5i cpu gpuE5b gpuD gpuE gpuE3 gpuE4):

| comparison | identical | timing/path only | differ | the rest |
|---|---|---|---|---|
| cpu vs gpuE5 | 60 | 2 | 7 | the same 7 as cpu vs gpuD; sorted records equal (c115c734…, 766e191f…, 31fc9ef1…); mapping statistics, `SJ.out.tab`, CellReads.stats rows identical |
| cpu vs gpuD | 60 | 2 | 7 | same |
| gpuE5 vs gpuE5b | 60 | 3 | 6 | order, timings, `bytes_public`; CellReads.stats byte-identical in file order |
| gpuE5x3 (`SINGLET_E16=0`) vs gpuE5 | 60 | 3 | 6 | same as gpuE5 vs gpuE5b |
| E11's kept gpuD, gpuE, gpuE3, gpuE4, cpu vs this binary's | 60 each | 2 each | 7 each | the 3 order-only files E11 did not keep, `bytes_public`, the whitelist path, 2 timing files; CellReads.stats same rows (level 4: 25,754, byte-identical) |
| cpu vs gpuE5i (E13 ingest) | — | — | — | **STAR's child crashed** (signal 11 after the prefill's end; exit status 3). The gate printed "GATE RUN OK" because it checked only the logs |

**The crash.** With ingest, `e10Record` (Unmapped.out's text) and `e11RnaAdjust` (the RNA Q30 counts) read E13's packed reads
(`ing`, and `ingSeq`/`ingOff` into the prefill's `Input`). E13 releases them at the prefill's end, because its Unmapped.out has
been written by then. E16's bg cut moves that writing to a thread beside STAR's loop, so the thread read freed memory.
- **Fix (d7aafb2c):** when the bg thread runs with ingest, `ingRelease()` waits for the thread's join in
  `seqCacheGpuAfterMapping`, and the prefill's `Input` is held until then. This holds even with the free cut off.
- **Gate:** each run's exit status is now checked (`EXIT STATUS 0`, or `GATE FAIL`).

**Gate 797398** (binary d7aafb2c, node g051, T = 18; modes gpuE5i gpuE5x3 (ingest, `SINGLET_E16=0`) gpuE5 cpu gpuE5ib):

| comparison | identical | timing/path only | differ | the rest |
|---|---|---|---|---|
| cpu vs gpuE5i (ingest) | 60 | 2 | 7 | the same 7 files. `pileup_stats.json` differs only in `bytes_public`: with ingest the 5' probe samples 12,194 reads as on the CPU engine (E13 §4). Sorted records equal; mapping statistics, `SJ.out.tab`, CellReads.stats rows identical |
| cpu vs gpuE5 (no ingest) | 60 | 2 | 7 | as in 797377 |
| gpuE5i vs gpuE5ib | 60 | 3 | 6 | order, timings, `bytes_public`; CellReads.stats byte-identical in file order |
| gpuE5x3 (ingest, `SINGLET_E16=0`) vs gpuE5i | 60 | 3 | 6 | same |
| gpuE5 vs gpuE5i | 60 | 3 | 6 | order, timings, the probe count 12,000 vs 12,194 with `bytes_public`; CellReads.stats byte-identical in file order |

Every run's exit status was 0. The fix changes nothing without ingest: gpuE5 of this binary against the CPU run in the same job
gives the same 60 / 2 / 7, and its tail is 8.79 s.

## 4. Timings

### 4a. The final binary before the merge (gate 797132, b14fb05f, node g050, T = 18)

Wall time and core-s are for the whole `singlet --pipeline` run (cgroup `cpu.stat`). base5 is E11's final binary (673cb317,
`jobs/fable/solowire/bin-s6`) in the same job. Every E16 run includes E14 (merged at 09e74739).

| run | switch | wall s | core-s | wall s/M | core-s/M | maxRSS |
|---|---|---|---|---|---|---|
| cpu | CPU engine | 394.7 | 2507.0 | 5.93 | 37.64 | 29.6 GB |
| gpuD | `--engine gpu` + `SINGLET_GPU_PILEUP=on` | 223.5 | 977.7 | 3.36 | 14.68 | 43.6 GB |
| base5 / base5b | level 5, E11's binary | 213.6 / 213.3 | 685.3 / 683.4 | 3.21 | 10.28 | 61.0 GB |
| gpuE5x3 | level 5, `SINGLET_E16=0` (E11's tail, E14's start) | 180.5 | 662.1 | 2.71 | 9.94 | 61.1 GB |
| gpuE5x1 | level 5, cuts of bin-c7 (§2b) | 155.9 | 648.4 | 2.34 | 9.74 | 61.0 GB |
| gpuE5x2 | level 5, cuts of bin-c8 | 154.6 | 648.1 | 2.32 | 9.73 | 61.0 GB |
| **gpuE5 / gpuE5b** | **level 5, E16** | **154.4 / 153.2** | **650.9 / 648.6** | **2.31** | **9.76** | 61.0 GB |

Level 5 with E16 (mean 153.8 s, 649.7 core-s) compared with:
- the same binary with `SINGLET_E16=0`: −26.7 s (−14.8 %), −12.4 core-s (−1.9 %). The tail fell by 27.2 s;
- E11's level 5: −59.7 s (−28.0 %), −34.6 core-s (−5.1 %). Of this, E14 accounts for −32.9 s (base5 vs x3);
- the default GPU path: 1.45x faster wall, 1.50x fewer core-s;
- the CPU engine: 2.57x faster wall, 3.86x fewer core-s.

**Phase breakdown** (`bench/fable/tail/s8_stages_e16.py`; wall s / core-s; 1 s resolution):

| stage | base5 (E11) | gpuE5x3 (`SINGLET_E16=0`) | **gpuE5** | **gpuE5b** |
|---|---|---|---|---|
| startup | 1 / 1.7 | 1 / 5.2 | 0 / 2.0 | 0 / 2.5 |
| decode .1fq → FASTQ + R11 histogram | 19 / 85.8 | 19 / 87.2 | 21 / 92.5 | 21 / 92.4 |
| probe/ref load | 5 / 8.3 | 5 / 7.8 | 5 / 8.8 | 4 / 6.7 |
| STAR load, CR4 probe, start | 36 / 45.8 | 3 / 15.3 | 5 / 21.6 | 5 / 21.8 |
| GPU prefill (to the device's release) | 58 / 298.6 | 58 / 300.7 | 56 / 297.6 | 56 / 297.6 |
| STAR per-read host pass (prefill's tail and STAR's loop) | 10 / 21.4 | 9 / 19.6 | **2 / 15.2** | **2 / 15.2** |
| STAR stream into pileup | 9 / 16.7 | 10 / 18.1 | **1 / 8.5** | **1 / 4.4** |
| GPU-pileup hand-off + pileup | 17 / 30.9 | 17 / 34.3 | **6 / 33.2** | **6 / 35.3** |
| non-host | 20 / 45.6 | 20 / 42.1 | 19 / 42.3 | 20 / 44.9 |
| export, QC, callers | 17 / 35.0 | 18 / 38.6 | 18 / 36.7 | 17 / 35.3 |
| demux/ancestry/writers | 21 / 95.4 | 20 / 92.7 | 21 / 92.4 | 20 / 91.3 |
| total | 213 / 685.2 | 180 / 661.5 | 154 / 650.8 | 152 / 647.4 |

The three stages of the tail went from 36 s to 9 s, and their core-s only from 72 to 57 (gpuE5). The parallel first touches, puts
and fetches spend core-s, while the waits they removed used none. E14 moved the STAR load and CR4 probe from 36 s to
3–5 s.

### 4b. After merging E12 and E13

Gate 797377 (binary d3a335b8: E12, E13 and E14 merged; node g050; T = 18):

| run | switch | wall s | core-s | wall s/M | core-s/M | maxRSS |
|---|---|---|---|---|---|---|
| cpu | CPU engine | 396.7 | 2498.9 | 5.96 | 37.52 | 29.6 GB |
| gpuD | default GPU path | 198.2 | 961.8 | 2.98 | 14.44 | 43.4 GB |
| gpuE / gpuE3 / gpuE4 | levels 1 / 3 / 4 | 195.6 / 195.6 / 153.1 | 927.2 / 923.2 / 652.7 | | | 43.5 / 61.3 / 61.3 GB |
| base5 | level 5, E11's binary | 213.7 | 684.4 | 3.21 | 10.28 | 61.0 GB |
| gpuE5x3 | level 5, `SINGLET_E16=0` (E12 + E14) | 153.5 | 658.1 | 2.30 | 9.88 | 61.3 GB |
| **gpuE5 / gpuE5b** | **level 5 (E12 + E14 + E16)** | **127.2 / 127.7** | **647.8 / 649.0** | **1.91** | **9.73** | 61.3 GB |

Level 5 (mean 127.4 s, 648.4 core-s) compared with:
- `SINGLET_E16=0`: −26.1 s (−17.0 %), −9.7 core-s. The tail fell by 26.0 s (34.90 → 8.88 / 8.85 s);
- E11's level 5: −86.3 s (−40.4 %), −35.9 core-s (−5.3 %);
- the default GPU path: 1.56x faster wall, 1.48x fewer core-s;
- the CPU engine: 3.11x faster wall, 3.85x fewer core-s.

| stage (wall s / core-s) | base5 (E11) | gpuE5x3 | **gpuE5** | **gpuE5b** |
|---|---|---|---|---|
| decode .1fq → FASTQ + R11 histogram | 20 / 87.3 | 21 / 92.8 | 20 / 87.6 | 21 / 91.5 |
| probe/ref load, startup | 6 / 9.8 | 4 / 8.9 | 5 / 12.6 | 5 / 10.5 |
| STAR load, CR4 probe, start | 36 / 46.0 | 5 / 19.6 | 5 / 21.5 | 6 / 21.8 |
| GPU prefill (to the device's release) | 58 / 299.1 | 31 / 296.0 | 30 / 295.3 | 30 / 295.6 |
| STAR per-read host pass (prefill's tail and loop) | 10 / 21.4 | 8 / 18.7 | **2 / 15.0** | **2 / 15.1** |
| STAR stream into pileup | 9 / 16.7 | 9 / 16.7 | **1 / 7.4** | **0 / 3.7** |
| GPU-pileup hand-off + pileup | 17 / 31.4 | 17 / 29.3 | **6 / 34.9** | **6 / 35.5** |
| non-host | 20 / 44.8 | 20 / 47.2 | 20 / 43.6 | 20 / 46.4 |
| export, QC, callers | 17 / 35.7 | 18 / 38.4 | 17 / 35.0 | 17 / 36.0 |
| demux/ancestry/writers | 21 / 92.1 | 19 / 89.6 | 20 / 94.0 | 21 / 92.4 |
| total | 214 / 684.2 | 153 / 657.2 | 126 / 646.8 | 127 / 648.6 |

With E12's 30 s prefill, the tail's 9 s is now 7 % of the run, against 27 % before the cuts (35 of 153 s). The step table matches
§1 within 0.15 s, except for the prefill's tail. That is 1.17 s here against 1.04 s in 797132, and with the cuts off it is 2.41 s:
E12 had already removed the cache's 1.6 s.

Gate 797398 (binary d7aafb2c, node g051, T = 18), with E13's ingest:

| run | switch | wall s | core-s | maxRSS | tail |
|---|---|---|---|---|---|
| cpu | CPU engine | 402.9 | 2493.2 | 29.6 GB | |
| gpuE5 | level 5 (no ingest) | 128.2 | 651.1 | 61.3 GB | 8.79 s |
| gpuE5x3 | level 5 + ingest, `SINGLET_E16=0` | 126.9 | 561.7 | 46.0 GB | 34.40 s |
| **gpuE5i / gpuE5ib** | **level 5 + ingest + E16** | **101.1 / 105.1** | **548.0 / 565.7** | 46.0 GB | **8.83 / 9.17 s** |

Level 5 + ingest + E16 (mean 103.1 s, 556.9 core-s) compared with:
- `SINGLET_E16=0`: −23.8 s (−18.8 %), −4.8 core-s;
- the CPU engine in the same job: 3.91x faster wall, 4.48x fewer core-s;
- E11's level 5 (213.7 s, 684 core-s in 797377 on g050): −110.6 s (−51.8 %).

The two ingest runs differ by 4 s. Their stage tables put the difference in the probe and prefill stages: 10 + 17 s and
9 + 18 s of wall time, and 107 vs 115 core-s in the prefill. The tail is 0.34 s apart.

| stage (wall s / core-s) | gpuE5x3 (ingest, no E16) | **gpuE5i** | **gpuE5ib** |
|---|---|---|---|
| decode in the pipeline, probe/ref load | 10 / 88.7 | 10 / 89.1 | 10 / 93.4 |
| STAR load, CR4 probe, CR4 clip | 7 / 100.3 | 10 / 123.7 | 9 / 123.4 |
| GPU prefill | 20 / 130.2 | 17 / 107.0 | 18 / 115.3 |
| STAR per-read host pass | 9 / 22.5 | **2 / 17.4** | **2 / 18.9** |
| STAR stream into pileup | 8 / 15.7 | **0 / 3.0** | **0 / 4.9** |
| GPU-pileup hand-off + pileup | 17 / 30.3 | **6 / 35.6** | **6 / 35.4** |
| non-host | 17 / 42.9 | 17 / 42.4 | 19 / 43.0 |
| export, QC, callers | 18 / 38.6 | 18 / 37.5 | 18 / 36.8 |
| demux/ancestry/writers | 20 / 91.5 | 21 / 91.4 | 21 / 93.9 |
| total | 126 / 560.8 | 100 / 547.1 | 104 / 565.0 |

The boundary between the "STAR load" and "GPU prefill" stages comes from the engine log's whole-second prefill time, so 3 s can
move between them (x3: 7 + 20, gpuE5i: 10 + 17). Their sum is 27 s in all three runs. From the prefill's end to the run's end, the
tail is 9 of 64 s with E16 and 34 of 89 s without.

## 5. Surprises

- **The header did not reach the pipe.** `bgzf_flush` hands the compressed block to htslib's hFILE layer, whose buffer reaches
  the fd only when full or closed. The header therefore travelled with the first records after the prefill, and the pipeline
  resolved its 7 M SNP references after STAR's loop. The first fix called `hflush(fp->fp)` from STAR's sources. That crashed
  (gate 796833): STAR's bundled `bgzf.h` declares another `struct BGZF` than the linked system htslib, so `fp->fp` read the wrong
  field. The helper `singlet_bgzf_hflush` is now compiled in the pipeline against the linked htslib.
- **`bgzf_close` leaves stdout open.** With the BAM on stdout, the linked htslib closes its stream but not fd 1, so the
  pipeline's read of the pipe ended only when STAR's process exited. The pipeline was therefore waiting 3–4.7 s for STAR's Solo
  write, its frees, `SJ.out.tab` and its exit. The trace's `/proc` scan of the pipe's write end showed it: "held by: <pid>
  (singlet-gpu) fd 1" before the eof cut, "none" after.
- **2 of 18 threads mapped all 157 k reads.** E10's read source handed out chunks as large as STAR's input buffer allowed, so the
  first threads to ask drained it.
- **The summary threads had nothing to do.** At level 5 no summary is made, but 18 threads were still started, each
  constructing (and leaking) a `ReadAlign`, a `Transcriptome` copy and an `OutSJ`.
- **The prefill re-created a 12 GB cache nobody used.** At level 5 no read looks the cache up, but its 1.5 GB index was zeroed on
  the critical path. E12 found and removed the same thing independently.
- **The pileup's digest cost 2.5 s for one log line.** It hashes every output tuple, 4.3 GB of them mt.
- **Freeing beside STAR's loop slowed the loop.** The first free cut released the prefill's GBs on a thread at once. The kernel
  serialises `munmap` with page faults in the same process, and chunk construction (bin-c6: 1.22 s instead of 0.21 s) and the loop
  fault in fresh memory. Holding the buffers until the loop is done and freeing them then costs nothing on the tail.
- **First touch decides `cudaHostRegister` and the download.** Page-locking the region took 1.0 s, of which half was faulting its
  pages into this process on one thread. The kept reads' download (2 GB) took 1.5 s, of which half was `resize` faulting fresh
  pages. Touching both with 16 threads first halved both steps, and the region's upload fell from 1.34 s to 0.86 s as well.
- **The early pass became the critical path.** Once the stream ends 2 s after the prefill, the pipeline waits for the early pass
  (1.0 s in the final runs, 2.4 s before rrnabg, dlpar and regpar). Making `e2ePass` faster now shortens the tail directly.
- **The puts slow down beside the host pass.** Run side by side, the puts take 1.01 s instead of 0.86 s, and the host pass's own
  1.18 s ends 0.65 s after them. The overlap saves 0.34–0.38 s, not the 0.86 s the puts take alone. Both are memory-bound.
- **E16's bg cut and E13's ingest freed the same memory at different times.** Each was correct on its own branch. Merged, the
  packed reads were released while the background thread still read them, and STAR's child crashed (§3d). The gate's log checks
  did not catch it, because the pipeline itself ran on and wrote its log lines; it now checks each run's exit status.
- **Zero-copy moved cost into the device reset.** With the region and the kept reads page-locked in the pileup's context, the reset
  also unpins 7 GB: 0.40–0.49 s instead of 0.11 s. resetbg moved it to the free thread, under the device lock that `pileupRun`
  took.

## 6. What remains

1. **The early hand-off pass (2.07 s; the pipeline waits 0.96 s for it):**
   - region upload 0.86 s;
   - kept reads' download 0.79 s;
   - page-locking 0.49 s;
   - CR table 0.22 s;
   - context 0.14 s.

   The region is written by STAR's child in the prefill and then page-locked by the pipeline. If the region were page-locked
   before the prefill, by the process that creates it, both the register step and part of the upload would go. The download could
   go straight into the collector's final layout.
2. **`pileupRun` (4.0 s):**
   - GPU time 2.16 s;
   - host fetches between partitions 0.84 s;
   - records H2D 0.29 s;
   - annotation copies 0.18 s.

   The fetches could move to the device's output arena with one download, or overlap the next partition.
3. **The puts and host pass (1.66 s side by side).** The puts copy the device's tuples into the accumulators' hash tables. Building
   those tables from sorted tuples, or giving the GPU path its own accumulator layout, would remove most of it.
4. **The prefill's tail (1.04 s):**
   - aux free and the read → record map copy into the region (0.54 s);
   - region and alignment counts (0.28 s);
   - read source (0.23 s).

   The copy could be written by the device into the region directly. E12 and E13 change this part of the prefill (§3d).
5. **The digest** could be computed on the free thread after `pileupRun` and logged there. Gates comparing it would then not need
   `SINGLET_E16_OFF=digest`.
6. **STAR's child after the loop (4.6 s, now beside the pileup):**
   - Solo waits 0.94 s for the device lock;
   - Solo write 0.64 s;
   - genome free 0.72 s.

   This is not on the tail's path, but the child keeps its memory until it exits at 6.5 s.

## 7. Files touched, and why

| file | change | why |
|---|---|---|
| `include/singlet/tail_trace.h` (new) | marks, `cuts()`, `cut(name)`, pipe-writer scan, per-thread loop timers, chunk log lock | instrumentation and switches |
| `src/star/SeqCacheGpuPrefill.cpp` | marks; chunk, sumthreads, bg (thread, join and append in `seqCacheGpuAfterMapping`), cache (superseded by E12 at the merge), free (held buffers freed after the loop), solofirst | in scope (prefill's tail and after-mapping) |
| `src/star/STAR.cpp` | marks; chunkpar; eof (stdout to `/dev/null` after the BAM's close); per-thread loop line | STAR's loop setup and exit |
| `src/star/ReadAlignChunk.cpp`, `ReadAlignChunk_processChunks.cpp` | chunk log lock (chunkpar); per-thread timers | chunkpar; loop trace |
| `src/star/samHeaders.cpp` | hdr | the header to the pipe at once |
| `src/pipeline/singlet.cpp` | `singlet_bgzf_hflush` (10 lines) | must be compiled against the linked htslib's `struct BGZF` (§5); called only by the hdr cut |
| `include/singlet/pileup/pileup_engine.h` | marks; early (`E16Early`), zerocopy input, rrnabg, overlap, puts, free2, resetbg's release | the hand-off pass, the input build, the apply |
| `include/singlet/pileup/gpu_pileup_bridge.h` | zerocopy: the collector takes the early pass's kept reads (`take_e2e_reads`) and hands them and the region's records to the input in place (`finish(in, zc)`); marks | input build |
| `include/singlet/engine_gpu.h` | `PileupInput::Pre` and `PileupCat` views, `keepPinned`, `PileupOutput::resetDeferred`, `pileupRelease` | zerocopy, resetbg |
| `src/engine_gpu/records_gpu.cu` | `e2ePass` keepPinned (context and page-locking survive), regpar, dlpar, step timers | the early pass |
| `src/engine_gpu/pileup_gpu.cu` | digest, copypar, resetbg, `[pre \| vector]` uploads, step timers | `pileupRun`'s transfers |
| `src/engine_gpu/gpu_pileup/pileup/pileup.cu` | aasplit, fetchpar | `pileupRun`'s downloads |
| `bench/fable/tail/jobs/{build,gate}.sbatch`, `e16_steps.py`, `s8_stages_e16.py` | build and gate jobs (modes gpuE5x1..3 with XENV1..3, gpuE5i), step and stage tables | deliverable |
| `docs/gpu-engine/fable/E16-tail.md` | this report | deliverable |

Not changed: no `Solo*.cpp`, nothing in the device kernels' arithmetic, nothing that runs below level 5 (§3c).

## 8. Jobs, commits, how to run

**Commits on `gpu/fable/tail`:**

| commit | content |
|---|---|
| 48af7732, c8cb91c6 | instrumentation, build and gate jobs |
| 260bbdd1 | hdr (first version), chunk, sumthreads, early, zerocopy |
| c0c25ea8, e7293303, eb67bfbe | bg, cache |
| 0011ff72 | hdr's hflush (first version), digest, free, free2, puts; trace of the pipe's writers |
| 89a70d51, 91e97e9a, ef4fdb0f | aasplit, solofirst, chunkpar |
| c824d5f1 | hdr's helper in the pipeline (the crash fix) |
| ea40283d | eof; free holds until after the loop |
| fe561c3c | overlap, fetchpar, copypar |
| 8667a63d | rrnabg, dlpar, regpar |
| 6e74e728, 902a12bc | step table rows; gate modes gpuE5x1..3 |
| b14fb05f | resetbg |
| 09e74739 | merge of `gpu/fable-engine` (E14) |
| 44b9ee96 | `s8_stages_e16.py` |
| d3a335b8 | merge of `gpu/fable-engine` (E12, E13) |
| 77f8baf1 | gate mode gpuE5i, E16 stage table per run |
| d7aafb2c | bg with E13's ingest: the packed reads kept until the thread's join (the crash fix); gate checks exit status, mode gpuE5ib |
| this document | the next commit |

**Builds** (short-gpu without a GPU, ~5 min each; output `$W/jobs/fable/tail/<dir>/singlet-gpu`):

| dir | commit | used by |
|---|---|---|
| bin-c3 | eb67bfbe | 796702, 796703 |
| bin-c4 / bin-c5 | 91e97e9a / ef4fdb0f | 796833 (crash: hdr's first hflush) |
| bin-c6 | c824d5f1 | 796862, 796863 |
| bin-c7 / bin-c8 | ea40283d / fe561c3c (build 796906) | (measured as x1/x2 in 797132) |
| bin-c11 | b14fb05f (build 796944) | 797132, 797370 |
| bin-c12 | d3a335b8 (build 797375) | 797377 |
| bin-c13 | d7aafb2c (build 797394) | 797398 |

**Gates** (T = 18 on short-gpu unless noted):

| job | binary | modes | result |
|---|---|---|---|
| 796702 | bin-c3 | base5 gpuE5x(`SINGLET_E16=0`) gpuE5 cpu | tail 34.7 → 22.5 s; gate (a) 60 / 2 / 7 |
| 796703 | bin-c3 | gpuE5 (debug-gpu, T = 8) | tail 23.7 s |
| 796833 | bin-c5 | gpuE5 (debug-gpu) | crash in hdr's hflush (§5) |
| 796862 | bin-c6 | gpuE5 (debug-gpu, T = 8) | tail 16.8 s; 60 identical vs 796702's CPU |
| 796863 | bin-c6 | gpuE5 ×2 (node g051) | tail 14.8 / 14.6 s; 60 identical |
| 797132 | bin-c11 | cpu gpuE5 base5 gpuE5x3 gpuE5b base5b gpuE5x1 gpuE5x2 gpuD | the final gate (§1, §3, §4a) |
| 797370 | bin-c11 | gpuE gpuE3 gpuE4 (+ comparisons with E11's kept outputs) | levels 1, 3 and 4 (§3c) |
| 797377 | bin-c12 | gpuE5 base5 gpuE5x3 gpuE5i cpu gpuE5b gpuD gpuE gpuE3 gpuE4 | the merged build (§3d, §4b); gpuE5i crashed |
| 797398 | bin-c13 | gpuE5i gpuE5x3 (ingest, `SINGLET_E16=0`) gpuE5 cpu gpuE5ib | the ingest fix (§3d, §4b) |

Cancelled while queued or within a few minutes, each superseded by a later build: 796799, 796807, 796834, 796889, 796908, 796915,
796983.

**Results:** `$W/jobs/fable/tail/results/gate-<job>/`:
- `summary.txt` and `equiv-*.txt`;
- per run: `e16.txt` (§1's table), `stages.txt` (E11's script), `stages-e16.txt` (from 77f8baf1), `starlog_engine.txt`,
  `star_Log.out`, `CellReads.stats`, `monitor.tsv` and `out/`.

Logs are in `$W/logs/e16-{gate,build}-<job>.out`.

**Run:**
- Build: `sbatch --export=ALL,OUTD=<dir> bench/fable/tail/jobs/build.sbatch`.
- Gate: `sbatch --export=ALL,BINDIR=<dir>,MODES="cpu gpuE5 gpuE5x3",XENV3="SINGLET_E16=0",PAIRS="cpu:gpuE5" bench/fable/tail/jobs/gate.sbatch`.
- Step table of one run: `python3 bench/fable/tail/e16_steps.py <results>/gate-<job>/<mode>`.
- Stage table: `python3 bench/fable/tail/s8_stages_e16.py <run dir> 66.601887`.

**Switches** (level 5 only):
- `SINGLET_E16=0`: all cuts off;
- `SINGLET_E16_OFF=a,b,...`: the named cuts off;
- `SINGLET_E16_TRACE=0`: no marks.
