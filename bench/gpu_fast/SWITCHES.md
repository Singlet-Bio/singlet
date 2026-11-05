# Runtime switches of `singlet --engine gpu`

This page lists the SINGLET_* environment variables that the engine reads at runtime through getenv.
Each one changes the behavior of `--engine gpu`.
Setting one needs no rebuild.

**Code read.** Commit e42c0636, plus the later tree at 0fdec57f, which adds commit 618487d7 and P3-ENG-14 for variable-width low-quality masks.
Every `file:line` below is at e42c0636.
Only one switch exists solely at 0fdec57f. It is marked **[0fdec57f only]**.
Build-time CMake macros are out of scope.

**Output risk column.**
- `none expected` means the code comments say the output is identical by design. This page did not run it.
- `yes` means the output changes.
- `timing`, `memory` and `diag` mean the output stays the same.
- `unclear` means the code does not settle it.

**Defaults that matter.**
`--engine gpu` sets SINGLET_ENGINE_GPU=1 inside the process, at singlet.cpp:3183.
The E2E level, the ingest and the hand-off need SINGLET_GPU_PILEUP=on.
With it, an unset SINGLET_ENGINE_GPU_E2E becomes 5 and an unset SINGLET_ENGINE_GPU_INGEST becomes 1, at engine_gpu.h:134-148.
Without it, only the GPU mapping runs.
The pipeline writes these defaults into the environment before it forks STAR, so STAR's child sees the same values.

## Can demux be skipped?

**Short answer.** Yes. Add `--no-snps` to the harness command line. No environment switch exists.

**What the stage does.**
The log line "[demux] Starting donor demultiplexing" comes from `run_demux`, at donor_demux.h:2067.
Export Phase 3, "Demux + parallel writes", calls it at export.h:1229-1248.
It runs on its own thread beside the matrix writers.
It fits a variational Bayes donor model to the per-cell SNP allele and depth matrices of the called cells.
When K is unset it first picks the donor count by a held-out likelihood search, at donor_demux.h:2145-2150.
It then fits the full model for each candidate K and may run split-merge rounds.
After the join, export writes `donor_assignments.tsv`, per-donor VCFs and coverage under `private/donor`, and the donor-aware chrM outputs, at export.h:1722-1760.
The stage runs only when `--pipeline` is set, a SNP panel is loaded and the SNP matrix holds data, at export.h:1233.

**Where the SNP panel comes from.**
With `--pipeline`, singlet loads a default panel from `<ref-base>/pileup_indices` unless `--snps` or `--no-snps` is given. See singlet.cpp:3873-3895.
The default exists for human and mouse genome directories only.
A genome path with GRCh38 or hg38 selects the human panel.
A path with GRCm39 or mm39 selects the mouse panel.
Other references and mixed references get no panel and therefore no demux.

**The switch.**
`--no-snps` sets `no_default_snps`. See singlet.cpp:3097 and the help text at singlet.cpp:2772.
The panel path stays empty, so the gate at export.h:1233 stays closed.
The harness command line accepts it:

```
singlet <in.1fq> --ref-base ... --genome-dir ... --exons ... --out-prefix ... --threads T --pipeline --output-format 1pz --engine gpu --no-snps
```

The harness passes `--exons`, so the check "at least one of --exons or --snps" at singlet.cpp:3790 passes.

**Caveats of `--no-snps`.**
- The run differs from the default run. The panel's SNPs leave the pileup, so the GPU pileup and the export do less work and some files differ.
- These files are missing or reduced. `private/snp_ad`, `private/snp_dp` and `private/ase_counts.tsv` are skipped at export.h:1299-1316. `donor_assignments.tsv` becomes a header-only stub at export.h:2157. The donor-aware chrM outputs are skipped because K stays 0, at export.h:1752. `summary.json` gets an empty donor_demux_method at export.h:1853.
- Ancestry calling stays on. It uses its own built-in AIM panel. The `aim_ancestry` option defaults to true at pileup_engine.h:138 and 492.
- The marker "[demux] Starting" never appears. The harness should expect that.

**Other CLI options that shorten the stage.**

| Option | Effect | Caveat |
|---|---|---|
| `--n-donors 1` | Skips the VB fit and the K search. Labels every cell donor0. Prints "[demux] K=1 workaround" instead of "[demux] Starting". See export.h:1238 and donor_demux.h:2644-2660. | Still extracts the covered SNPs and still writes the donor VCFs and chrM donor outputs. All donor labels change. |
| `--n-donors K` with K above 1 | Skips the K search and keeps the full VB fit for that K. See donor_demux.h:2145-2147. | Output changes when K differs from the auto choice. |
| Dropping `--pipeline` | Closes the gate at export.h:1233. | Also turns off chrM counting, poly(A) counting and the default directional UMI dedup, at singlet.cpp:3871-3872 and 3897. Unfit for a like-for-like harness. |

No other option skips the stage.
donor_demux.h has no getenv.
export.h reads only SINGLET_POLYA_DUMP, SINGLET_GIT_SHA and SINGLET_REF_MANIFEST_SHA.
Export builds `DonorDemuxConfig` with only `n_donors`, `threads` and `seed = 42` set, at export.h:1245-1247.

## Device slots, device memory and fallback on out of memory

| Switch | Values and default | Effect | Output risk | Source |
|---|---|---|---|---|
| SINGLET_ENGINE_GPU_SLOTS | 1 to 8, default 1 | Samples per GPU. Takes one exclusive lock file per slot in /tmp, named by PCI bus id. Also divides the default memory budget. | none expected | engine_gpu.h:263, 278, 326 |
| SINGLET_ENGINE_GPU_DEVICE | device index, default the first free slot of the first device | Pins the device and waits for a slot of that device. | none expected | engine_gpu.h:277 |
| SINGLET_ENGINE_GPU_MEM_GB | GB as a number, 2^30 bytes each. 0 means no budget. Default is device memory minus 2 GB, divided by slots | Budget per sample for the prefill. The pileup arena gets the budget less 3 GB. With 0, lane B and the arena take the free memory less 3 GB. | none expected | engine_gpu.h:262, engine_gpu.cu:948, pileup_gpu.cu:64 |
| SINGLET_GPU_PILEUP_ARENA_GB | GB as a number, default unset | Caps the GPU pileup arena. The partitions adapt. | none expected | pileup_gpu.cu:66 |
| SINGLET_ENGINE_GPU_MEM_FALLBACK | `cpu`, default unset | Applies when the E17 budget check finds the sample too big. Unset ends the run with a fatal error. `cpu` lets STAR map every read. Whether the GPU pileup still runs after this fallback is unclear. | none expected | SeqCacheGpuPrefill.cpp:1485-1491 |
| SINGLET_ENGINE_GPU_E2E_GB | integer GB, default 512 | Sparse size of the shared region for E8 records. | memory | SeqCacheGpuPrefill.cpp:1439 |
| SINGLET_STAR_SEQCACHE_MB | MB, default 4096. Under `--engine gpu` it grows to 0.6 times the cDNA file size, capped at 98304 | Host memory of STAR's sequence cache. The value 0 removes the cache, and the GPU prefill then returns at once without a log line. | none expected | SeqCache.cpp:65-80, SeqCacheGpuPrefill.cpp:1285 |

Out-of-memory behavior without a switch.
A real device allocation failure inside `mapDistinct` takes the "GPU mapping failed" path, and STAR maps every read. See engine_gpu.cu:898-904 and SeqCacheGpuPrefill.cpp:1296-1299 and 1492-1501.
The GPU pileup retries once in unified memory of twice the device size when its arena runs out. See pileup_gpu.cu:57, 69 and 187.
The comment at pileup_gpu.cu:55-56 says the results stay the same.

## 1. Harness and pipeline-wide gates

| Switch | Values and default | Effect | Output risk | Source |
|---|---|---|---|---|
| SINGLET_ENGINE_GPU | `1` turns it on. `--engine gpu` sets it and `--engine cpu` unsets it | Master gate of every GPU part. Other values mean off at the checks that compare to "1". | none expected | singlet.cpp:3183, 3188, 3358, 4211 |
| SINGLET_GPU_PILEUP | `on` or `compare`, default unset | `on` runs the pileup on the GPU and skips the covered CPU steps. `compare` runs both and compares in process, so it is slow. Other set values count as set in the singlet.cpp checks but run mode 0 in the bridge, so use only `on` or `compare`. Declined with a stderr line for unsupported intron rules, `--sj-all-alignments` and long reads. The read limit is 128 bases here and 512 at 0fdec57f. | none expected | gpu_pileup_bridge.h:34, singlet.cpp:5712-5742 |
| SINGLET_ENGINE_GPU_E2E | first character 1 to 5, default 5 when SINGLET_GPU_PILEUP=on, else 0 | Level of device work. 1 records on the device. 2 per-read entries. 3 STAR takes products from the device. 4 mapped reads skip STAR. 5 STARsolo on the device. Other values mean 0. A set value is kept. | none expected | engine_gpu.h:119, 138, tail_trace.h:20 |
| SINGLET_ENGINE_GPU_INGEST | `1` turns it on, default 1 at level 5 | Decodes the .1fq into the prefill buffers and writes no temp FASTQ. Other set values keep the FASTQ files. The ingest also declines on its own for the conditions at singlet.cpp:4732-4756. | none expected | engine_gpu.h:142-143, singlet.cpp:4224 |
| SINGLET_GPU_HANDOFF | `0`, default unset | Disables the hand-off to the GPU pileup. The ingest declines too. | unclear | singlet.cpp:4740, 5757-5759 |
| SINGLET_GPU_HANDOFF_SUMMARY | `0`, default unset | Turns off the per-sequence summaries. The E8 and E10 paths then fall back with a stderr line. | unclear | SeqCacheGpuPrefill.cpp:1329, 1432, 1580, 1597 |
| SINGLET_GPU_PILEUP_FORCE_DECLINE | any value, default unset | Forces the CPU pileup on GPU-mapped reads. Regression test of the fallback. | none expected | singlet.cpp:4745, 5733 |
| SINGLET_KEEP_STAR_BAM | any value, default unset | Selects the sorted BAM path. Skips the GPU pileup and the hand-off and declines the ingest. Keeps the BAM at the end when set to `1`. | none expected | singlet.cpp:4746, 5688, 8747 |
| SINGLET_REF_BASE | directory path, default unset | Fallback for `--ref-base`. The harness passes the flag. | yes | singlet.cpp:3566, 3879 |
| SINGLET_WHITELIST_DIR | directory path, default unset | Extra barcode whitelist directory when `--whitelist-dir` is absent. | yes | singlet.cpp:602, fg2_source.h:28, run_screen.h:605 |
| SINGLET_MEMORY_TIER | `Small`, `Medium`, `Large` or `XLarge`, default the measured peak RSS | Sets the memory_tier field of summary.json. | yes, one field | singlet.cpp:9003 |

## 2. Read ingest and decode

| Switch | Values and default | Effect | Output risk | Source |
|---|---|---|---|---|
| SINGLET_READ_SOURCE | `chunk`, `fifo` or `file`, default `chunk` | How decoded reads reach STAR. Under `--engine gpu` the cDNA reads are decoded to files first, so `fifo` has no effect there. `file` also sets SINGLET_FILE_DECODE. | none expected | singlet.cpp:3016, 3415, 4214 |
| SINGLET_FILE_DECODE | `1`, default unset | Forces the file decode. Set by `--read-source file`. | none expected | singlet.cpp:4215 |
| SINGLET_READ_SOURCE_THREADS | integer above 0, default max of 2 and min of 4 and threads/4 | Decode threads of the chunk source. | none expected | singlet.cpp:8135, 8145 |
| SINGLET_PASS1_THREADS | integer above 0, default min of threads and blocks | Caps the barcode pass-1 threads. | none expected | singlet.cpp:5257 |
| SINGLET_ENGINE_GPU_INGEST_EARLY | `0`, default unset | The prefill decodes the .1fq when it starts. The default starts the decode beside the genome load. | none expected | singlet.cpp:4904, 7991 |
| SINGLET_ENGINE_GPU_INGEST_CLIPDEDUP | `0`, default on | Turns off the grouping of clipped reads in the ingest. | unclear | SeqCacheGpuPrefill.cpp:787 |
| SINGLET_ENGINE_GPU_FG2_OFF | any value, default unset | Applies to a .1fg2 input only. Keeps the temp FASTQ. A .1fq input ignores it. | none expected | singlet.cpp:4212, SeqCacheGpuPrefill.cpp:1363 |
| SINGLET_GENOME_PRELOAD | `0`, default on | Turns off the background genome preload. | none expected | singlet.cpp:4014 |
| SINGLET_GENOME_PREFETCH | `0`, default on | Turns off the page-cache prefetch of the genome files. The source comment says no output changes. | none expected | singlet.cpp:4026 |
| SINGLET_GENOME_PRELOAD_THREADS | integer above 0, default unset | Applies to the `singlet star` subcommand only. | timing | singlet.cpp:2980 |

## 3. STAR prefill, the device mapping

| Switch | Values and default | Effect | Output risk | Source |
|---|---|---|---|---|
| SINGLET_ENGINE_GPU_CFG | `old`, default new | Restores the configuration before E12. Lanes, host dedup, pageable uploads and the host lane after the lanes. Several switches below act only with it. | none expected | engine_gpu.cu:909 |
| SINGLET_ENGINE_GPU_LEAN | `1` or `0`, default chosen by the budget | `1` forces lean mode and `0` forces normal mode. Lean uploads the index after the dedup. Both need a budget. | none expected | engine_gpu.cu:969 |
| SINGLET_ENGINE_GPU_CHUNK | sequences per chunk, minimum 262144, default 16M | Chunk size. Device lane buffers scale with it. The budget can lower it. | none expected | engine_gpu.cu:1348 |
| SINGLET_ENGINE_GPU_NB | integer, minimum 4096, default 524288 | Batch of lane A. The default is 262144 with CFG=old. | none expected | engine_gpu.cu:1342-1343 |
| SINGLET_ENGINE_GPU_NBM | integer, minimum 4096, default 393216 | Batch of lane M. | none expected | engine_gpu.cu:1344 |
| SINGLET_ENGINE_GPU_NBB | integer, minimum 1024, default 131072 | Batch of lane B. | none expected | engine_gpu.cu:1346 |
| SINGLET_ENGINE_GPU_WARPM | `0`, default on | Lane M runs thread per read. | none expected | engine_gpu.cu:1337 |
| SINGLET_ENGINE_GPU_DEDUP | `host`, default device | Builds the dedup groups on the host. | none expected | engine_gpu.cu:1190 |
| SINGLET_ENGINE_GPU_HOSTLANE | `after`, default concurrent | Runs the host lane after each chunk's device lanes. | none expected | engine_gpu.cu:1404 |
| SINGLET_ENGINE_GPU_UPLOAD | `pageable`, default staged | Uses pageable driver copies for uploads. | none expected | engine_gpu.cu:929 |
| SINGLET_ENGINE_GPU_D2H | `pageable`, default staged | Uses pageable copies from the device. | none expected | engine_gpu.cu:934 |
| SINGLET_ENGINE_GPU_PIN | any value, default unset | Pins the host index. Acts only with UPLOAD=pageable or CFG=old. | none expected | engine_gpu.cu:932 |
| SINGLET_STAR_FASTASSIGN | `0`, default on | Turns off the precomputed strand conversion in seed assignment. The header says it makes the same calls. | none expected | STAR.cpp:237, StarAccel.h:7 |
| SINGLET_STAR_SEQCACHE_VERIFY | `1`, default unset | Cache verify mode. | diag | SeqCache.cpp:93 |
| SINGLET_ENGINE_GPU_CHUNK_SORT | `1`, default unset | Keeps STAR's chunk prefix sort when the cache is prefilled. The comment says only the record order inside a chunk changes. | none expected | ReadAlignChunk_mapChunk.cpp:102 |

## 4. Hand-off pass

| Switch | Values and default | Effect | Output risk | Source |
|---|---|---|---|---|
| SINGLET_HANDOFF_PASS_THREADS | integer, default from the cores, capped at 32 | Threads of the hand-off pass. One thread when fewer than 65536 reads. | none expected | pileup_engine.h:2149, 2319, 2484 |
| SINGLET_HANDOFF_PASS_SERIAL | any value, default unset | Uses one chunk in the poly(A) and V(D)J rebuild. Also leaves the all-cores entries pass for the serial pass at line 2615. Identity of the serial pass is unclear. | unclear | pileup_engine.h:2148, 2615 |
| SINGLET_HANDOFF_LEAN | `0`, `verify` or other, default lean on | `0` builds full BAM records. `verify` runs both and counts differences. | none expected | ReadAlign_seqCache.cpp:158 |
| SINGLET_HANDOFF_QMASK | `0`, default on | Turns off the per-read low-quality masks. With varied base qualities the GPU pileup then reports "not run" and the run can end with exit 9. | yes | SeqCacheGpuPrefill.cpp:1330, 1606, 1770, pileup_engine.h:2058 |
| SINGLET_HANDOFF_CLIPCHECK | `1`, default unset | Compares the GPU clip lengths with STAR's. | diag | SeqCacheGpuPrefill.cpp:1697 |
| SINGLET_ENGINE_GPU_E10_CHECK | `1`, default unset | Level 2 only. Compares STAR's per-read entries with the device's. | diag | SeqCacheGpuPrefill.cpp:1326, 1742 |
| SINGLET_ENGINE_GPU_E10_R11 | file path, default unset | Writes the device's CB histogram. | diag | SeqCacheGpuPrefill.cpp:1518 |
| SINGLET_HANDOFF_QWORDS **[0fdec57f only]** | integer clamped to 2 to 8, default 2 | Mask words per read. Singlet overwrites it for every sample from the cDNA read length, at 0fdec57f singlet.cpp:5732, so an outside value has no effect in the pipeline. | none expected | 0fdec57f include/singlet/qmask_words.h:19 |

## 5. Solo and counting

| Switch | Values and default | Effect | Output risk | Source |
|---|---|---|---|---|
| SINGLET_SOLO_PAR | `0`, default on | Reads STAR's Solo record files in parallel. `0` runs the serial loop. The comment says every later step sees the same data. | none expected | SoloFeature_countCBgeneUMI.cpp:49-50 |
| SINGLET_E15_PERREAD | `1`, default unset | Forces per-read CB quality in the E11 posterior. The ingest ignores it. | none expected | SeqCacheGpuPrefill.cpp:268 |
| SINGLET_E15_KEEP_ON_DEVICE | `1`, default unset | Keeps the Solo inputs on the device while STAR runs. The default moves them to pinned host memory. | none expected | SeqCacheGpuPrefill.cpp:1515 |
| SINGLET_UMI_COLLAPSE | `seed`, default connected components of the directional graph | Changes the UMI collapse rule of the CPU molecule store. A reader exists only in umi_dedup.h. The GPU pileup has none, so GPU and CPU results may differ. | yes | umi_dedup.h:269, 549, 645 |

## 6. GPU pileup

| Switch | Values and default | Effect | Output risk | Source |
|---|---|---|---|---|
| SINGLET_INTRON_RULES | integer, parsed with base 0, default 397 | Intron rule mask. The GPU pileup accepts only bits 1, 4, 8, 128 and 256. Other bits decline it. | yes | pileup_engine.h:428, singlet.cpp:4739, 5719 |
| SINGLET_GPU_PILEUP_QUAL | `force`, default unset | Always uses the per-base masks. | none expected | gpu_pileup_bridge.h:123 |
| SINGLET_GPU_PILEUP_PARTS | integer from 0 to the maximum, default 0 | Forces the number of partitions. | none expected | pileup_gpu.cu:117 |
| SINGLET_GPU_PILEUP_E23 | `0`, default on | Runs the per-group kernels as before E23. | none expected | pileup.cu:782 |
| SINGLET_GPU_PILEUP_PROF | set and other than `0`, default off | Group profile lines. | diag | pileup.cu:781 |

## 7. Non-host screen

| Switch | Values and default | Effect | Output risk | Source |
|---|---|---|---|---|
| SINGLET_NONHOST_NO_MMAP | `1`, default memory-map | Reads the index file into the process. | memory | min_sketch.h:389 |

No environment switch skips the screen.
The GPU screen runs only when SINGLET_ENGINE_GPU=1, the viral, bacterial and fungal index paths are all given and the device is usable. See singlet.cpp:1545.
Otherwise the CPU screen runs.

## 8. Export, demux and doublet

| Switch | Values and default | Effect | Output risk | Source |
|---|---|---|---|---|
| SINGLET_GIT_SHA | string, default unset | Written to provenance in summary.json. | yes, one field | export.h:1838, singlet.cpp:3815, 5532, 5565 |
| SINGLET_REF_MANIFEST_SHA | string, default unset | Written to provenance in summary.json. | yes, one field | export.h:1840, singlet.cpp:3816, 5533, 5566 |
| SINGLET_SPECIES_FILTERS | directory of Bloom filters, default unset | Runs the informational species check on the .1fq and logs one line. Adds time. The source comment says no output changes. | none expected | singlet.cpp:3606, species_detect.h:700 |

Demux has no environment switch. See the first section.

## 9. Memory, threads and tuning

| Switch | Values and default | Effect | Output risk | Source |
|---|---|---|---|---|
| SINGLET_STALL_TIMEOUT | seconds above 0, default 1800 | Watchdog. The run ends with exit 5 after that long without progress. | none expected | stall_watchdog.h:27-38 |
| SINGLET_MEM_BUDGET_BYTES | bytes, default from SLURM, cgroup and MemTotal | Memory budget for the BAM sort bins, read through bamSortByCoordinate.cpp:31. Whether the GPU stream path uses it is unclear. | unclear | bam_sort_budget.h:20 |
| SINGLET_CB_REFRESH_SERIAL | `1`, default parallel | Serial CB tag scan. Sorted-BAM path only. | none expected | singlet.cpp:8677 |
| SINGLET_E16 | `0`, default on at level 5 | Turns off all E16 tail cuts. | none expected | tail_trace.h:30, engine_gpu.h:146 |
| SINGLET_E16_OFF | comma list of cut names, default empty | Turns off the named cuts. The names in the source are early, zerocopy, rrnabg, puts, overlap, free2, aasplit, fetchpar, resetbg, digest, copypar, regpar, dlpar, chunkpar, eof, sumthreads, bg, chunk, free, solofirst and hdr. | none expected | tail_trace.h:38 |

## 10. Diagnostics and dumps

| Switch | Values and default | Effect | Output risk | Source |
|---|---|---|---|---|
| SINGLET_E16_TRACE | `0` or `1`, default 0 on the GPU path | `1` prints "[e16]" timing marks at level 5. | diag | engine_gpu.h:147, tail_trace.h:21 |
| SINGLET_NVTX | any value, default off | NVTX ranges. Needs a build that found nvtx3. | diag | nvtx.h:25 |
| SINGLET_ENGINE_GPU_DUMP | directory path, default unset | E6 debug dump of the device output. | diag | engine_gpu.cu:1935 |
| SINGLET_ENGINE_GPU_E2E_CHECK | any value, default unset | Compares gathered records with host-assembled records. | diag | engine_gpu.cu:1723 |
| SINGLET_SOLO_DUMP | directory path, default unset | E9 debug dump of Solo records. | diag | ReadAlign_outputAlignments.cpp:26 |
| SINGLET_PILEUP_DUMP | directory path, default unset | Writes the pileup accumulators for the W11 harness. | diag | gpu_pileup_dump.h:20, pileup_engine.h:1925 |
| SINGLET_STREAM_TEE | file path, default unset | Copies the stream BAM. | diag | gpu_pileup_dump.h:25 |
| SINGLET_TEE_STREAM_BAM | file path, default unset | Copies the unsorted STAR stream to a file. | diag | singlet.cpp:8176 |
| SINGLET_MOLECULE_DUMP | file path, default unset | Molecule dump. The format is unclear. | diag | pileup_engine.h:430 |
| SINGLET_POLYA_DUMP | file path, default unset | Raw poly(A) records, 24 bytes each. | diag | export.h:1573 |
| SINGLET_POLYA_DEBUG | any value, default unset | Poly(A) rebuild keep and drop counts. | diag | pileup_engine.h:2136 |
| SINGLET_R11_DUMP | file path, default unset | Writes the stream CB corrector's exact histogram. | diag | singlet.cpp:8836 |
| SINGLET_CR4_PROBE | `serial`, default parallel probes | Runs the two CellRanger4 probes one after the other. | none expected | singlet.cpp:8016 |
| SINGLET_EXTRA_SAM_ATTRS | space-separated tags, default unset | Adds STAR SAM attributes. Debug only. The effect on the stream is unclear. | unclear | singlet.cpp:7829 |
| SINGLET_KEEP_ATAC_BAM | directory path, default unset | ATAC only. | none expected | singlet.cpp:8271 |

## Set by singlet, do not set

SINGLET_ENGINE_GPU_HANDOFF, SINGLET_ENGINE_GPU_HANDOFF_MINQ, SINGLET_ENGINE_GPU_FG2, SINGLET_ENGINE_GPU_FG2_CLIP5P, SINGLET_DECODE_R1 and SINGLET_DECODE_R2 are written by the pipeline for STAR's child. See singlet.cpp:4918, 5754, 5762, 5764 and 6586.
Setting SINGLET_ENGINE_GPU_HANDOFF from outside skips the pipeline's own hand-off setup at singlet.cpp:5757.

## Test hooks

SINGLET_TEST_N_MISS_BARCODES, SINGLET_TEST_DROP_READS_PCT, SINGLET_TEST_CB_CHECK and SINGLET_TEST_STAR_START_DELAY_MS exist for tests only.
