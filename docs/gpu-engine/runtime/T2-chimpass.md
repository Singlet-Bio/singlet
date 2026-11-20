# T2 chimpass: where the time of the `--chimeric` second pass goes

Analysis and measurement only. No file under `src/` or `include/` was changed. Three jobs were used (819008, 819009, 819021; none failed). Scripts: `bench/gpu-runtime/jobs/t2-chimpass.sbatch`, `t2-chimpass-watch.sh`, `t2-chimpass-reads.cpp`, `t2-chimpass-summary.py`. Data: `bench/gpu-runtime/data/t2-chimpass/` (results of the three jobs, per-run summaries, the reads analysis). Binary: `bin/t2-hostcut5` (the one of job 818999). Remote results: `$W/jobs/gpu-runtime/results/t2-chimpass-819008-gsm6250953`, `...-819009-gate-mix`, `...-819021-det`.

## Summary (facts; estimates are marked)

- The pass is STAR (the bundled `star_main_impl`) run again in a forked child on 50.77 M reads, with the whole genome index loaded again (10 s), 8 threads, no sequence cache (`SEQCACHE: off (chimeric detection)`), Solo counting on (its output is deleted), then a single-threaded post-process of 17.19 M junction lines (24 s).
- At 8 threads the pass is 866 s: 807 s of mapping (93 %), 10 s genome load and start, 23 s Solo post-map, 24 s table post-process. It uses 8.0 cores for the whole mapping phase and 6,480 CPU-s (user+sys of the STAR child).
- `--runThreadN` of the pass is the item's thread count; it cannot be set separately (no environment variable, no flag). Measured by changing the item's `--threads`: 866 s (8), 466 s (16), 332 s (27); CPU-s 6,480 / 6,495 / 7,146.
- The output table is identical at 8, 16 and 27 threads (md5 of the decompressed `chimeric_junctions.tsv.gz` `3573f164...`). The raw `Chimeric.out.junction` differs in line order only (checked on pbmc5p_v1: sorted data lines identical).
- Of the 50.77 M reads only 17.64 M sequences are distinct (34.7 %). 4.62 M reads give a kept (best-score, canonical) line, and they come from 168,769 distinct sequences; a sequence always gave the same verdict in all its reads (0 of 168,769 mixed).
- 11.36 M reads (22.4 %) give any junction line; 17.19 M lines; 4.50 M kept. By source (upper bound, before the barcode filter): 83.5 % of the canonical best lines come from the 41.1 M unmapped reads, 16.5 % from the 9.6 M soft-clipped reads.
- A length pre-filter has nothing to remove here: 24 of 50.77 M reads are shorter than 24 bases after clipping (average read length 149).

## 1. Anatomy

### 1.1 Which reads go in

`--chimeric` (`singlet.cpp:3982`) sets `chimeric = true` and `SINGLET_CHIM_SOFTCLIP_MIN=15` unless set. While the main pass runs, two FASTQ pairs are written into the item's output directory (cDNA mate = `mate1`, barcode mate = `mate2`):

- `star_Unmapped.out.mate1/2`: the reads STAR's output left unmapped (`outReadsUnmapped Fastx`; at level 5 the engine writes them: `e10WriteUnmapped`, `SeqCacheGpuPrefill.cpp:2318`).
- `star_Softclip.out.mate1/2`: unique mappers with >= 15 unaligned read bases. CPU path: `ReadAlign::softclipRoute` / `outReadsSoftclip` (`ReadAlign_outputAlignments.cpp:182-185, 386-402`, hand-off path 598); the files are opened in `Parameters.cpp:843-847`. Engine path: the device flags the record (bit 2 of `recMapped`, `hostloop_gpu.cu:177`, minimum from `SINGLET_CHIM_SOFTCLIP_MIN` at `:317`) and `SeqCacheGpuPrefill.cpp:2323-2331` writes those reads and logs `ENGINE_GPU: --chimeric soft-clip route: N reads with a record`.

The main STAR's arguments are saved to `star_args.txt` (`singlet.cpp:9378-9381`). The files are plain FASTQ on the item's local disk; no pipe. The records are the original (unclipped) reads with the original barcode mate.

Job 819008, GSM6250953 (121.4 M pairs): 41,132,142 unmapped + 9,636,150 soft-clipped = 50,768,292 reads. File sizes: `star_Unmapped.out.mate1` 13.21 GB, `mate2` 3.01 GB, `star_Softclip.out.mate1` 3.09 GB, `mate2` 0.70 GB, 20.01 GB in all (0.165 GB per M pairs).

### 1.2 The pass (`singlet.cpp:10561-10670`)

- Counts the records of both files by `getline` (`:10579-10582`), logs `chimeric: N reads into the second pass (...)`.
- Builds the argument list from `star_args.txt` minus a drop list (`outSAMtype`, `outStd`, BAM options, `outFileNamePrefix`, `outTmpDir`, `outReadsUnmapped`, `outSAMattributes`, `outSAMunmapped`, `readMapNumber`, `readFilesIn`, `readFilesCommand`, `readFilesType`, `soloFeatures`, `soloOutFileNames`, `soloMultiMappers`). Everything else of the main STAR stays: genome dir, `--runThreadN` (the item's threads), `--genomeLoad NoSharedMemory`, `--clip3pAdapterSeq AAGCAGTGGTATCAACGCAGAGTACATGGG --clip3pAdapterMMp 0.1`, `--outFilterScoreMin 10`, Solo barcode parameters and whitelist, `--soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts`.
- Appended: `--readFilesIn unm1,sc1 unm2,sc2` (the soft-clip file is read after the unmapped one), `--soloFeatures Gene --outSAMtype None --outFileNamePrefix <out>/chim_`, and Arriba's chimeric set: `--chimSegmentMin 10 --chimJunctionOverhangMin 10 --chimScoreMin 1 --chimScoreDropMax 30 --chimScoreJunctionNonGTAG 0 --chimScoreSeparation 1 --alignSJstitchMismatchNmax 5 -1 5 5 --chimSegmentReadGapMax 3 --chimMultimapNmax 50 --chimOutType Junctions --chimOutJunctionFormat 1`.
- `fork()`; the child unsets the engine variables and `SINGLET_CHIM_SOFTCLIP_MIN` and calls `star_main_impl(...)`; the parent `waitpid`s. So the genome index is loaded again in that child (private memory, `NoSharedMemory`), STAR maps with `runThreadN` threads, chimeric detection runs after the normal alignment of each read (`ReadAlign_oneRead.cpp:166-176`, `ReadAlign::chimericDetection`, the new multimapping detection `ChimericDetection::chimericDetectionMult`), Solo counting runs for `Gene` (CB and UMI of each read are parsed once for the junction line, `:166-170`), and the junction lines go to `chim_Chimeric.out.junction` (one buffer per thread, merged at the end).
- Afterwards the parent deletes `chim_Solo.out`, `Softclip.out.*`, `chim_STARgenome` (Solo output is never used).
- `singlet::chimeric::write_table` (`include/singlet/pileup/chimeric_table.h:499`, `build_rows` `:266`): reads the raw junction file line by line (`gzgets`), drops lines with `this_chim_aln_score < bestall_chim_aln_score`, drops non-canonical lines (`jtype < 1`), turns lines to transcript orientation, resolves the cell barcode (exact, else the one unique 1-mismatch neighbour; else `unmatched` / `ambiguous`), collapses to rows of (cell, junction) with read and UMI counts, sorts the rows (total order on barcode and junction coordinates), writes gzip level 6. Single-threaded.
- The pass's own logs and the raw junction file are deleted (`SINGLET_KEEP_CHIMERIC_JUNCTION=1` keeps the junction file).
- `base_args` come from the engine's STAR arguments, so on this 5' library the TSO adapter of the 3' kit is the clip adapter in both passes (the run was encoded without `--protocol`, as 818999).

### 1.3 Where the 866 s go (job 819008, run T8, `T8.summary.txt`)

| phase | start (clock) | length | share |
|---|---|---|---|
| record counting (before the log line) | not measured (<= 9 s in 818999, which had a log line 9 s earlier) | <= 9 s | <= 1 % |
| STAR start, parameters, whitelist, genome index load | 20:06:40 | 10 s | 1.2 % |
| mapping + chimeric detection (8 threads) | 20:06:50 | 807 s | 93.2 % |
| Solo post-map (read Solo files 22 s, collapse, matrix, filtering; output deleted) | 20:20:17 | 23 s | 2.7 % |
| `write_table` (GTF load, 17.19 M lines, collapse, sort, gzip) | 20:20:41 | 24 s | 2.8 % |
| log line to log line | 1795136800 to 1795137666 | 866 s | |

STAR's own log: `Mapping speed 220.20 M reads per hour`; progress 200-209 M reads/h through the 41.1 M unmapped reads, 211-222 M/h during the soft-clip file. 1 Hz watch: 8.0 cores in use in every 30 s window of the mapping (5.9 in the first, 1.6 in the last), child CPU 6,479.8 s (user+sys), cgroup CPU over the watch 6,583.8 s, `VmHWM` 24.0 GiB, cgroup anon+shmem 53.9 GiB before, 70.5 GiB peak (+16.6 GiB), junction file 2.34 GB (first bytes 9 s after the start, so the lines go out with the mapping).
Per read: 128 us CPU (6,480 s / 50.77 M). The first log line of STAR's `Log.out` is `SEQCACHE: off (chimeric detection)` (`SeqCache.cpp:77`: the cache refuses `pCh.segmentMin > 0`, and `seqCacheEncode` refuses `chimRecord`, `ReadAlign_seqCache.cpp:122`).

The item: wall 1,037 s (engine part ~171 s, pass 866 s: 83.5 %), reproducing 818999 exactly: `50768292 reads ... 17188238 junction lines, 4498881 kept, 0 without barcode, 123105 unmatched, 14517 ambiguous, 9749577 non-canonical, 4589683 turned ...`.

STAR's Log.final.out of the pass: input 50,768,292, average read length 149 (after clip), uniquely mapped 14,922,322 (29.39 %), multi 151,292, too many loci 31,549, unmapped too short 34,560,654 (68.08 %), other 1,102,475 (2.17 %), chimeric reads 11,359,390 (22.37 %).

## 2. Scaling

### 2.1 Thread count

Not settable separately: `--runThreadN` is copied from `star_args.txt` (the item's `--threads`, `singlet.cpp:9127-9128`; the pool passes `--threads`); there is no `SINGLET_CHIM_*` variable for it. So the runs below change the item's threads (the main engine at level 5 gets them too; its own wall barely moves: 135.7 s in 818999). Same sample, same `.1fq` (encoded once), same job, one pool run each, `SINGLET_KEEP_CHIMERIC_JUNCTION=1`.

| item threads | pass wall (log to log) | STAR job start to finish | mapping | genome+start | Solo post-map | `write_table` | cores in use (mapping) | STAR child CPU (user+sys) | cgroup CPU | `VmHWM` | cgroup anon+shmem increase | item wall | STAR reads/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 866 s | 840 s | 807 s | 10 s | 23 s | 24 s | 8.0 | 6,480 s | 6,584 s | 24.0 GiB | 16.6 GiB | 1,037.0 s | 220.2 M |
| 16 | 466 s | 439 s | 406 s | 10 s | 23 s | 25 s | 16.0 | 6,495 s | 6,576 s | 25.0 GiB | 18.0 GiB | 629.5 s | 426.0 M |
| 27 | 332 s | 305 s | 267 s | 15 s | 23 s | 25 s | 26.9 | 7,146 s | 7,199 s | 26.2 GiB | 19.3 GiB | 481.4 s | 630.2 M |

Speed-up of the pass: 1.86 x at 2 x threads, 2.61 x at 3.4 x threads (efficiency 0.93 and 0.77); the fixed parts (start 10-15 s, Solo 23 s, table 25 s, ~60 s) do not shrink. CPU-s are constant from 8 to 16 threads and 10 % higher at 27. The cgroup page cache of the node-local files rose to 71.7 / 93.4 / 94.2 GiB (the reads and the junction file).

Determinism (`t2-chimpass-819008/results-819008.txt`):

| threads | md5 of decompressed `chimeric_junctions.tsv.gz` | rows | sorted lines md5 | raw junction file md5 | raw sorted md5 |
|---|---|---|---|---|---|
| 8 | 3573f1645c4fc6e920d37c70a0033df7 | 59,632 lines | 6d7713248f81821a19272012c5098e49 | 5ba315a7... | d7501e09... |
| 16 | 3573f1645c4fc6e920d37c70a0033df7 | 59,632 | 6d7713248f81821a19272012c5098e49 | 4e78ae97... | 990fd7d2... |
| 27 | 3573f1645c4fc6e920d37c70a0033df7 | 59,632 | 6d7713248f81821a19272012c5098e49 | 7d3dd5e4... | 5af86ca7... |

The table is identical at 8, 16, 27 threads. The raw junction file is not byte-identical (17,188,241 lines each: header line, 2 `#` lines, 17,188,238 data lines; sizes 2,339,252,964 / ...970 / ...970). For GSM6250953 I hashed only the whole file (the `#` lines were not separated), so the raw sorted md5 differing there is explained but not proven by this job. The follow-up job 819021 (pbmc5p_v1, 8 and 27 threads) separates them: the raw file md5 differs (3dcb18c0... / 3776afcf...), the data lines without the two `#` lines differ in order only (md5 a2bc6a83... / 592a65b0...) and **the sorted data lines are identical** (`af98b621630e8dd6698eca2dc8dbb1fd` both); the `#` comment lines carry the command line (thread count) and counters. The table md5 of pbmc5p_v1 is the same at both (`a5d71eb6a8b45889fe667bf57f17dce3`).

### 2.2 Gate sample and mix sample (job 819009, 8 threads, `--chimeric`)

| sample | pairs | reads into the pass (unmapped + soft-clip) | pass wall | item wall | junction lines | kept | non-canonical | child CPU | `VmHWM` | read length (STAR) | unmapped "too short" | chimeric reads |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pbmc5p_v1 | 60,000,000 | 4,944,371 (4,818,388 + 125,983) | 131 s (1795139043 to 1795139174) | 196.6 s | 3,926,694 | 27,130 | 2,928,198 | 856.6 s | 24.4 GiB | 90 | 89.30 % | 913,687 (18.48 %) |
| d300_5p | 299,628,383 | 22,372,925 (21,477,879 + 895,046) | 560 s (1795139341 to 1795139901) | 711.0 s | 16,886,797 | 145,563 | 12,840,905 | 4,109.3 s | 27.9 GiB | 90 | 86.11 % | 4,160,956 (18.60 %) |
| GSM6250953 (for comparison) | 121,366,072 | 50,768,292 | 866 s | 1,037.0 s | 17,188,238 | 4,498,881 | 9,749,577 | 6,479.8 s | 24.0 GiB | 149 | 68.08 % | 11,359,390 (22.37 %) |

Pass share of the item: 67 % (pbmc5p_v1), 79 % (d300_5p), 83.5 % (GSM6250953). Reads into the pass per M pairs: 0.082, 0.075, 0.418. Pass seconds per M reads at 8 threads: 26.5, 25.0, 17.1 (the 90-base samples cost more per read than the 149-base one: 173, 184, 128 us CPU per read). Kept lines are 0.7 %, 0.9 % and 26 % of the lines. pbmc5p_v1 re-run in 819021: pass 131 s at 8 threads, 66 s at 27, same lines and kept counts.
All three used `bin/t2-hostcut5`, level 5 (`E19: level 5 reached`), blacklist `none` in all (the reference base has no `pileup_indices/fusion` here, so the blacklist step did nothing; as in 818999).

## 3. How much of the input can matter (GSM6250953, 50,768,292 reads; `reads-analysis-819008.txt`, counts only)

Tool: `t2-chimpass-reads.cpp` on the pass's input files and raw junction file kept by the first run (hard links; 78 s). Sequences compared in full (128-bit hash). The 3' adapter clip of `ClipMate::clip` / `localSearch` is replicated for the clipped length.

### 3.1 Distinct sequences

| set | reads | distinct sequences | share |
|---|---|---|---|
| unmapped | 41,132,142 | 13,735,577 | 33.4 % |
| soft-clip | 9,636,150 | 3,900,547 | 40.5 % |
| both files | 50,768,292 | 17,636,124 | 34.7 % |

Multiplicity (both files): 15,114,737 sequences occur once (15.1 M reads); 1,341,059 twice; 843,464 3-5 times (3.01 M reads); 192,347 6-10 times; 120,478 11-100 times (3.06 M reads); 19,333 101-1,000 times (6.17 M reads); 4,544 1,001-10,000 times (10.69 M reads); 162 sequences more than 10,000 times (8.64 M reads). Most frequent sequence 1,858,007 reads, then 990,279, 429,819.
The pass does nothing like the main engine's dedup: the sequence cache is off for chimeric detection (above), so each of the 50.77 M reads is mapped.
A chimeric alignment depends on the clipped read sequence only: the code path (`oneRead` -> `mapOneRead`, `multMapSelect`, `chimericDetection`) uses `Read1`, its length and the genome; qualities are not used for alignment (`Qual0` feeds the quality histogram and outputs); the line's other columns (read name, CB, UMI) are attached at output (`chimericJunctionOutput`, `ChimericAlign_chimericJunctionOutput.cpp`). Observed: for the 168,769 sequences that gave a kept line in some read, no other read of the same sequence gave none (0 mixed).

### 3.2 Length against the minimum segment

Confirmed from the code: `chimericAlignScore` (`ChimericDetection_chimericDetectionMult.cpp:6-20`) needs `roE > chimSegmentMin + roS + overlap` for both segments (each >= 12 aligned bases with `chimSegmentMin 10`, plus the overlap), and the segments together cover at least 24 read bases; also `trBest->maxScore <= Lread - chimNonchimScoreDropMin(20)` is required to start the detection (`ReadAlign_chimericDetection.cpp`, default `parametersDefault:726`), and the chimeric score must be >= `Lread - chimScoreDropMax(30)`. A read shorter than 24 bases after clipping cannot give a chimeric junction (two minimum segments of 10 would be 20; the strict `>` makes it 24).
Measured: clipped length distribution of the 50.77 M reads: 23 reads with 0-11 bases, 1 with 12-19, 0 with 20-23, none 24-89, **50,768,268 with 90 or more**. Average input read length 149. Over all 17.19 M junction lines the smallest `max_possible_aln_score` (clipped length) is 116 and the smallest drop from it to the non-chimeric score is 20 (the trigger). 0 lines come from reads under 24 bases.

### 3.3 Low-complexity, poly(A)/poly(T), TSO

Definitions: mono = >= 90 % of the clipped bases one base; tandem = period 2-6 repeat with >= 90 % identity (not mono); entropy = base-composition entropy of the clipped read (length >= 12); TSO motifs exact: 5' kit `TTTCTTATATGGG` and its reverse complement `CCCATATAAGAAA`, 3' kit core 16-mer `TATCAACGCAGAGTAC` and reverse complement; "concatemer" = >= 2 non-overlapping hits of any motif. Counts over unmapped + soft-clip reads; "kept reads" = reads with a best-score canonical line (4,621,263 reads in all).

| class | reads | % of 50.77 M | of them with any junction line | of them with a kept (best-score canonical) line |
|---|---|---|---|---|
| poly(A)-dominated | 71,630 | 0.14 % | 40,699 | 57 |
| poly(T)-dominated | 278 | 0.0005 % | 185 | 0 |
| poly(G)-dominated | 547,446 | 1.08 % | 1,217 | 13 |
| poly(C)-dominated | 88 | | 4 | 0 |
| tandem repeat, period 2-6 | 28,296 | 0.06 % | 14,665 | 34 |
| entropy < 1.0 bit | 1,009,907 | 1.99 % | 62,518 | 124 |
| entropy < 1.5 bit (includes the above) | 2,169,636 | 4.27 % | 65,867 | 174 |
| contains N | 151,967 | 0.30 % | 28,476 | 11,638 |
| 5' kit TSO reverse complement `CCCATATAAGAAA` | 8,271,024 | 16.29 % | 3,096,162 | 44,972 |
| 5' kit TSO forward | 84,760 | 0.17 % | 22,917 | 358 |
| 3' kit TSO core (sum of the two strands) | 64,340 | 0.13 % | 383 | 1 |
| any TSO motif | 8,415,901 | 16.58 % | 3,119,438 | 45,331 |
| TSO concatemer (>= 2 hits) | 50,128 | 0.10 % | 5,270 | 18 |

The 3' adapter of the clip step matched (partially) in 12,308,419 reads (24.2 %).

### 3.4 Which reads give the junction lines

- 11,359,390 reads (22.4 % of the input; equal to STAR's "chimeric reads") give at least one of the 17,188,238 lines: 1.51 lines per read. They are 3,237,917 distinct sequences.
- Lines by source of the read: unmapped file 5,972,859, soft-clip file 11,215,379 (65 %). Reads: unmapped 5,419,032 of 41.13 M (13.2 %), soft-clip 5,940,358 of 9.64 M (61.6 %).
- Best-score lines: unmapped 5,856,059, soft-clip 8,530,021. Of those canonical: unmapped 3,869,687, soft-clip 766,816 (total 4,636,503; kept 4,498,881 + unmatched 123,105 + ambiguous 14,517 = 4,636,503, exact). Non-canonical best lines: unmapped 1,986,372, soft-clip 7,763,205.
- Reads with a canonical best line: 4,621,263 (9.1 % of the input): unmapped 3,864,457, soft-clip 756,806, from 168,769 distinct sequences (0.96 % of the 17.64 M).
- So of the 4.50 M kept lines at most (upper bound, the barcode filter removed 137,622 lines, source unknown) 83.5 % came from the 41.1 M unmapped reads and 16.5 % from the 9.6 M soft-clipped reads. The soft-clip route is 19.0 % of the reads (about 19 % of the mapping time, an estimate: the speed is the same).

## 4. Options (nothing implemented; savings are estimates from the measurements above)

### (a) More threads for the pass while the GPU idles
- What changes: `singlet.cpp` pass (`:10561`): give the second STAR its own `--runThreadN` (today copied from `star_args.txt`, `drop`/rebuild list at `:10586-10595`); a variable such as `SINGLET_CHIM_THREADS`; `gpu_pool.cpp` would have to let an item's thread budget change between phases.
- Identical by construction: yes for the table (measured: the table md5 is the same at 8, 16, 27; raw lines the same set).
- To test: gate with the switch on; pool admission with a variable thread count.
- Estimate from the measurements: pass 866 -> 466 -> 332 s (item 1,037 -> 630 -> 481 s). The CPU-s do not fall (6.5 k at 16, 7.1 k at 27), so on a lane with three items in flight on 27 cores the lane's core capacity is the limit: 27 threads for one item takes all cores for 332 s and starves the other two; the gain is the item's latency and the worker place and GPU slot hold time, not core-seconds (estimate: lane throughput roughly unchanged, wall of the item shorter; not measured with competing items).

### (b) Map each distinct sequence once
- What changes: a dedup stage before the pass (distinct by sequence of the clipped read; the engine already dedups on the device: 1.4 s for 121 M reads), or let `SeqCache` cover chimeric detection: `SeqCache::create` (`:77`, the `pCh.segmentMin` refusal), `ReadAlign::seqCacheEncode` (`:122`, `chimRecord`), `ReadAlign_oneRead.cpp:166-176`, and the junction output (`ChimericAlign::chimericJunctionOutput`) so that a cache hit writes the cached chimeric alignments with the new read's name, CB and UMI. Or in `singlet.cpp`: write unique sequences to the pass and expand the junction lines by (sequence id -> member reads) afterwards.
- Identical by construction: the junction lines of a read are a function of its clipped sequence (3.1); the expanded raw file differs only in line order, which already depends on threads; the table is built from CB, UMI and the junction only. Not "by construction" for a hash-only key: compare full sequences (the existing cache does, `SEQCACHE verify`).
- To test: table md5 against the baseline on pbmc5p_v1, d300_5p and GSM6250953 (each a different mix), sorted raw lines equal, reads with N (151,967 reads here), the soft-clip file and the unmapped file merged or not, multi-chimera reads (`chimN` > 1), thread counts.
- Estimate: the pass maps 17.64 M instead of 50.77 M reads (34.7 %). If the per-read cost were equal, the mapping 807 s becomes about 280 s (saving about 527 s of 866 s at 8 threads); the fixed 57 s remain (start 10, Solo 23, table 24 plus the expansion). With 27 threads on top: about 267 s x 0.347 + 63 s = about 155 s (both estimates; the cost per read of the duplicates is unknown: the most frequent 162 sequences hold 8.6 M reads and may be cheaper or dearer than average).

### (c) A provably exact pre-filter
- What the code proves: only "clipped length < 24 cannot give a chimeric junction". It removes 24 of 50.77 M reads here (average read 149, minimum useful 116 observed). Nothing else is provable before mapping: the trigger `maxScore <= Lread - 20` and the segment rules depend on the alignment.
- Not exact but measured (cost = kept reads lost, of 4,621,263): poly(A)/poly(T)/poly(G)/tandem/low entropy < 1.5 (about 2.17 M reads, 4.3 %): 174 kept reads lost (0.004 %); reads with the 5' TSO reverse complement (8.27 M reads, 16.3 %): 44,972 lost (0.97 %); skipping the soft-clip route entirely (9.64 M reads, 19.0 %): up to 766,816 canonical lines lost (16.5 % of the kept candidates). These change the output and would need the recall measurement the brief asks for.
- A provable after-mapping rule exists on the device side only if the device knew the pass's trigger: reads whose main-pass alignment leaves 15-19 bases unaligned and no mismatch have a drop below 20, but the pass's own `trBest` can differ (other parameters: `alignSJstitchMismatchNmax 5 -1 5 5`), so it would need a test; the unaligned length of the soft-clip reads is not recorded, so I cannot count them.

### (d) Defer the pass out of the GPU lane (as `SINGLET_HLA=defer`)
- What would have to be kept: `star_Unmapped.out.mate1/2` and `star_Softclip.out.mate1/2`: 20.0 GB for this sample (13.21 + 3.01 + 3.09 + 0.70 GB; about 0.165 GB per M pairs of the sample, or 0.39 GB per M reads of the pass); the main STAR arguments (`star_args.txt`); the barcode list used for matching (`engine_ptr->barcodes()`), the strand verdict (`reverse_strand`), the GTF, genome dir, whitelist, catalogue and blacklist paths. The raw reads of the item are deleted after processing, so these files are the only input. A smaller form (estimate): distinct sequences (17.6 M, about 3 GB packed) plus per-read (sequence id, CB, UMI) records (50.8 M, about 1 GB), so about 4 GB instead of 20 GB.
- Functions: split the block `singlet.cpp:10561-10670` into a function writing `pending/` and one that runs it, the analogue of `singlet hla --pending` (`singlet.cpp:1250-1284`, `hla_defer` at `:7972-7994`, `hla::run_pending`); a subcommand `singlet chimeric --pending DIR [--threads N]` that runs STAR with the saved arguments and `write_table`, then deletes `pending/`.
- Identical by construction: the same code on the same files, so the table would be identical (the run is thread-independent for the table, 2.1). The output set of the item would lack `chimeric_junctions.tsv.gz` until the deferred step runs (the item's output inventory and validation would need to accept that).
- What the runner needs: a CPU lane that picks up items with a `pending/` directory (6.5-7.1 k core-s and 24-26 GiB of memory for a 121 M-pair sample at any thread count; 20 GB of local or shared disk until it ran); the item's output directory must stay readable until then, and the campaign's upload step must wait for the table.
- Estimate: the item's wall would fall by the pass (866 s of 1,037 s here), i.e. the worker place and the GPU slot are free 83 % earlier; the CPU cost does not disappear.

### (e) Something cheap on the device
- The device pass already routes the soft-clip reads (`hostloop_gpu.cu:177`) but holds one best alignment record per read, not the windows and segments that chimeric detection needs (the only mention of chimeric in `src/engine_gpu` is that route). Reusing the seeds of the main pass for chimeric detection would need the device to keep or re-run the windows for 50.8 M reads (17.6 M distinct) and implement `chimericDetectionMult` (segment pairing, stitching, motif scoring); not cheap. What the device already does cheaply is the dedup (1.4 s for 121 M reads, `engine_gpu: 121366072 reads, 36270543 distinct`): the same key could feed option (b) and could flag which of the 50.8 M reads duplicate another (65.3 % of them do).
- The device maps at about 0.6 M distinct reads per second (36.3 M in 60.6 s) against 0.058 M reads per second for the pass at 8 threads (estimate of the ratio: about 10 x per read, different parameters and no chimeric step).

### (f) Other things the code suggests
- Solo counting inside the pass is wasted: `--soloFeatures Gene` is forced, the pass spends 23 s (2.7 % at 8 threads; the same 23 s at 16 and 27 threads, so 7 % at 27) reading its own Solo files, collapsing UMIs and writing a matrix, then `singlet.cpp` deletes `chim_Solo.out`. Solo is needed only to make STAR parse CB and UMI for the junction line. Skipping the post-map would not change the junction lines (estimate: saves 23 s per item; needs a test of the junction file).
- `write_table` is single-threaded (24-25 s at every thread count: 2.8 % at 8, 7.5 % at 27 threads); 17.19 M lines are parsed although 9.75 M (57 %) are dropped as non-canonical and 2.80 M as worse alignments; filtering in the STAR output (or a parallel parse) would cut it (estimate: to a few seconds).
- 57 % of the junction lines are non-canonical and are dropped after being written (the 2.34 GB raw file): `--chimScoreJunctionNonGTAG 0` makes them score as well as canonical ones in the pass, so changing it would change which chimeras are found (not identical).
- The pass reads the genome index again, privately (+16.6 to +19.3 GiB cgroup memory, `VmHWM` 24-26 GiB, 10-15 s); `--genomeLoad` of the pass could use the shared index segment the pool already keeps (not checked in the code).
- The pass's clip is the main pass's: on this 5' library `AAGCAGTGGTATCAACGCAGAGTACATGGG` (3' kit TSO) is the adapter, while 16.3 % of the pass's reads contain the 5' kit's TSO reverse complement, which neither pass clips; those reads are 3.1 M of the 11.4 M that give junction lines but 45 k of the kept ones.

## Not done, failed, or uncertain
- Mapping time inside the pass was not split between alignment, chimeric detection and CB/UMI parsing (no profiler build; no change to `src/`).
- The record counting time before STAR starts was not measured separately (the only bound is the 9 s of 818999).
- The raw junction file of GSM6250953 was hashed with its `#` lines, so determinism of its data lines across threads rests on the identical table and on the pbmc5p_v1 raw check.
- By-source kept lines are an upper bound (canonical best-score lines); the barcode filter removed 137,622 lines whose source I did not separate.
- The unaligned-base histogram of the 9.6 M soft-clip reads is not available (the route does not record it).
- Lane contention at 16/27 threads with other items in flight was not measured (the runs were alone in the job's 27 cores).
- The job script's trap removes its own node-local scratch directory (`rm -rf $SCR`, as every job script of the track does); no `rm` was sent through the bridge.
- All runs: `bin/t2-hostcut5` (not the head 8da76165); `--no-hla`; budget 31 / 2 slots / 1 worker.
