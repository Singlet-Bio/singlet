# T2 chimprof: where the CPU of the `--chimeric` second pass goes, inside the mapping

Analysis and measurement only. Behaviour of the default build is unchanged; the instrumentation is off unless an environment variable is set. Branch `t2/chimprof` (from `gpu/throughput-2` 401e73bd). Builds `bin/t2-chimprof` (job 819064) and `bin/t2-chimprof-2` (job 819069, built with `EXTRA_CXX=-fno-omit-frame-pointer`): both `ctest exit 0`, 188 of 188 tests passed. Cluster jobs besides the builds: 819065 (kept inputs), 819066 (identity gate), 819068 (first run, pbmc5p_v1), 819071 (final run, both samples): 4 jobs. Scripts: `bench/gpu-runtime/jobs/t2-chimprof-inputs.sbatch`, `jobs/t2-chimprof-run.sbatch`, `t2-chimprof-reads.py`, `t2-chimprof-samples.py`, `t2-chimprof-toy.sh`. Data (counts, hashes, seconds, function names): `bench/gpu-runtime/data/t2-chimprof/`. Remote results: `$W/jobs/gpu-runtime/results/t2-chimprof-819065-in`, `t2-chimprof-819068-pbmc`, `t2-chimprof-819071-final`; kept pass inputs `$W/jobs/gpu-runtime/results/t2-chimprof-inputs/{pbmc5p_v1,gsm}/` (`chim_dedup.mate1/2`, `args.txt`).

## Summary (facts; estimates are marked)

- The mapping inside the pass is dominated by one thing nobody had measured: **the pair loop of `ChimericDetection::chimericDetectionMult`**. It takes 38.1 % of the CPU on GSM6250953 (1,904 of 5,000 CPU-s of the per-read cycle counters; the stack sampler gives 36.9 % of the samples in the pair loop itself, 37.8 % with the stitching and line output of the chimeric code) and 30.4 % on pbmc5p_v1 (229 of 755 CPU-s; sampler 28.9 %, 30.9 % with stitching and output). It visits 85.3 billion window-pair combinations on GSM6250953 (10.07 billion on pbmc5p_v1) to find 12.8 M (7.2 M) candidates; 92.9 % (76.9 %) of its time is spent on reads that end with no junction line. Nearly all of it is arithmetic on pairs that a score bound already excludes (candidate 1 in section 4: exact, estimated to remove at least 90 % of this time).
- Everything else: stitching (`stitchWindowAligns`, `stitchAlignToTranscript`, `extendAlign`) 28 % (GSM6250953) and 31 % (pbmc5p_v1) of the samples; seed search 21.7 % / 21.1 % (`compareSeqToGenome` alone 16 %); seed storage and windows 9.3 % / 13.1 %; input parsing (read load, adapter clip, chunk sort) 1.8 % / 2.3 %; Solo/barcode parsing 0.7 % / 1.0 %; output 0.1 %. No sample has a lock, condition-variable or futex wait in its innermost four frames (wall-clock sampler); read input and all output together are 1.9 % (GSM6250953) and 2.4 % (pbmc5p_v1).
- The cost per read is spread, not concentrated: the costliest 0.1 % / 1 % / 10 % of reads take 4.2 % / 13.0 % / 42.0 % of the CPU on GSM6250953 and 4.1 % / 12.2 % / 36.8 % on pbmc5p_v1. The costliest reads have about 115 (GSM) and 66 (pbmc) windows against 50 and 34 on average, and 40 % (GSM) and 54 % (pbmc) of them are low complexity (entropy under 1.5 bits) against 10 % and 30 % of all reads.
- `perf` could not be used (see Method). The pass can be run on its own from kept inputs (`singlet star <arguments>`): that is how everything below was measured; the raw junction lines are identical in every arm (hash of the sorted lines).
- The pass's genome index is a private copy because the item process drops the pool's shared pages before it forks the pass; nothing in the code or in the kernel objects (no SysV segment, no memfd for the STAR index) lets the pass attach to them (section 5).

## Method actually used, and what was unavailable

1. **Kept inputs (job 819065, one H100 lane, as `t2-chimdedup.sbatch`).** One normal `--chimeric` item run per sample with `SINGLET_CHIM_KEEP_INPUT=1` (new, default off; `singlet.cpp`, in `run_pass` and at the cleanup): the pass input `chim_dedup.mate1/2` and the pass's STAR argument list (`chim_args.txt`, one argument per line) are kept and copied. pbmc5p_v1: 4,535,057 distinct sequences (`chim_dedup.mate1` 883,225,005 bytes, `mate2` 293,667,595 bytes, md5 `701fc7e4fed4c586fc0a1fc00a3a8655` / `435fda859f408be69777ab1f216e7031`); GSM6250953: 17,636,124 distinct sequences (5,526,631,826 and 1,152,873,074 bytes, md5 `26b50d0e94122c1c32d2c54bc01fb468` / `11a24c2e222e171a33df93a4edaa6e2c`). The item tables are the known ones: pbmc5p_v1 `a5d71eb6a8b45889fe667bf57f17dce3` (24,141 rows), GSM6250953 `3573f1645c4fc6e920d37c70a0033df7` (59,632 rows); the item walls were 177.9 s and 811.0 s.
2. **The pass run on its own (jobs 819068, 819071; CPU only, `-p short-gpu --exclude=g050,g051,g052`, no GPU).** `singlet star` with the kept argument list, `--readFilesIn` and `--outFileNamePrefix` replaced, `--runThreadN 8`; the kept input is copied to node-local `/tmp` first. Arms one after the other on the same node: `base` (no instrumentation), `prof` (`SINGLET_CHIM_PROF`), `samp-fp` (`SINGLET_CHIM_SAMPLE`).
3. **Per-read instrumentation, `SINGLET_CHIM_PROF=<prefix>` (`src/star/ChimProf.{h,cpp}`).** Around `ReadAlign::oneRead` and inside the mapping code, behind `if (g_chimProf)` (one load of a global bool when off): one 128-byte record per read into `<prefix>.reads.<k>`: rdtsc cycles of read load, seed search (`mapOneRead` seeding loop), window creation, flank extension and seed assignment (`stitchPieces`), stitching (the window loop of `stitchPieces`), `multMapSelect`, barcode parse (`getCBandUMI`), `chimericDetection` (with the stitching and junction-output parts inside it), output; the counters `Lread`, `nP` (stored seeds), `nA` (SA loci of the stored seeds), windows made, windows with alignments, `nTr`, `maxScore`, `mapMarker`; for the chimeric code the entry flags and the counters of pair visits, scored pairs, candidates, `chimericCheck` passes, stitchings, survivors, lines; flags for the STAR limits (`seedMultimapNmax` dropped a seed, `seedPerWindowNmax`, `alignWindowsPerReadNmax`, `alignTranscriptsPerReadNmax`, `alignTranscriptsPerWindowNmax`, seed above `winAnchorMultimapNmax`, `seedPerReadNmax`, too many anchors, `chimN > chimMultimapNmax`). The read id is the number in the read name (`@<id>` of the dedup input); the analysis joins it to the pass input for the sequence classes. Analysis: `t2-chimprof-reads.py`.
4. **Function-level profile, `SINGLET_CHIM_SAMPLE=<prefix>`, `SINGLET_CHIM_SAMPLE_MODE=fp`.** **`perf` is not usable**: `perf_event_paranoid` is 4 on the compute nodes (g001, g008) and `perf record -F 99 -- /bin/true` fails with "Access to performance monitoring and observability operations is limited" (no CAP_PERFMON); `ptrace_scope` is 1; `gdb`, `eu-stack` and `pstack` do not exist. So the sampling fallback is in the binary: a helper thread sends SIGPROF to every mapping thread (`ReadAlignChunk::processChunks`) 100 times a second of wall-clock time; the handler (on an alternate signal stack) stores the interrupted pc and the return addresses of a frame-pointer walk (`t2-chimprof-2` is built with `-fno-omit-frame-pointer`; a libc leaf function has no frame, so the first return address into the executable found within 320 bytes of the stack pointer is added) into a preallocated per-thread buffer. 100 % of the samples have the interrupted pc as their first frame. `t2-chimprof-samples.py` symbolizes with `nm` and `c++filt` (addresses in libc without exported symbol, 3.2 % of the samples on pbmc5p_v1 and 2.4 % on GSM6250953, are shown as `[libc.so.6 unresolved] <- caller`). The stacks are wall-clock samples, so a thread waiting on a lock would show up; none did. Inlined functions are counted in the function that contains them (`assignAlignToWindow` is partly inlined in `stitchPieces`).
   - Modes: `bt` (`backtrace()`), `fp`, `pc` (interrupted pc only). The first run of the sampler (job 819068, build `t2-chimprof`, mode `bt`) **died with a segmentation fault** about 6 s into the mapping of pbmc5p_v1 (shell message in `pbmc5p_v1.samp.time.txt`, core to apport, not available). The cause was not found; it did not reproduce in a toy genome with 4 and 8 threads, Solo, the same arguments, 100 and 1,000 samples a second on the login node (`t2-chimprof-toy.sh`). Build 2 added the alternate stack and the `fp` walk (and a fault report); with it the sampler ran to the end on both samples (jobs 819071). Modes `bt` and `pc` of build 2 ran only on the toy genome.
5. **Overheads and identity.** pbmc5p_v1 (job 819071, g008, Xeon Gold 6230 2.1 GHz): `base` wall 108.3 s, user 748.7 s, sys 10.6 s; `prof` 110.1 / 761.6 / 11.1 (+1.6 % wall, +1.7 % user); `samp-fp` 108.5 / 749.0 / 11.6 (+0.2 %). GSM6250953: `base` 637.8 / 4,930.6 / 14.7; `prof` 649.6 / 5,027.5 / 16.5 (+1.8 % wall, +2.0 % user); `samp-fp` 639.1 / 4,939.3 / 20.4 (+0.2 %). Job 819068 (g001, Xeon Gold 6226 2.7 GHz, build 1): pbmc5p_v1 `base` 104.4 / 639.2 / 20.0, `prof` 96.0 / 649.2 / 11.8. So the same pass costs 639 or 749 CPU-s on two node types: compare only within a job. The sorted data lines of the raw junction file (header and `#` lines removed, `LC_ALL=C sort`) are identical in every arm: pbmc5p_v1 `643f187626f6ffaa67d0fe9acf324d42` (2,832,142 lines) in all five arms of 819068 and 819071, GSM6250953 `c2a0f02ef3d48de17e89fab98b17c1ac` (6,706,877 lines) in all three. (These are lines of the representative sequences, the read name being the id: not comparable with the `RAWNORM` md5 of `t2-chimdedup`, which is over the expanded lines.) The line counts equal the `chimLines` counters. STAR's own counters are equal too (pbmc5p_v1: 114,684 unique, 4,275,567 too short, 799,963 chimeric reads; GSM6250953: 4,732,909 unique, 12,241,451 too short, 3,237,917 chimeric reads).
6. **Identity gate** (job 819066, build `t2-chimprof`, all switches off, the brief's command; the other-build change is flags only, see the end): 8 verdicts, `FALLBACK LINES: 1` for each sample (the R11 info line):
```
VERDICT pbmc2m cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 91/7/4)
VERDICT pbmc10m cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 91/7/4)
VERDICT pbmc1k_v3 cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 92/7/4)
VERDICT pbmc1k_v2 cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 92/7/4)
VERDICT pbmc1k_cite_v3 cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 92/7/4)
VERDICT pbmc5p_v1 cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 91/5/4)
VERDICT neuron1k_v3 cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 92/6/4)
VERDICT nuclei2k_v2 cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 92/6/4)
```

## 1. Function-level profile of the mapping phase (8 threads, dedup input)

Samples are wall-clock at 100 Hz per thread over the whole life of the mapping threads (processChunks begin to end, so read-chunk input, chunk sort, mapping, chunk output and the final junction-file concatenation are included). Groups: each sample goes to the group of its innermost frame that belongs to one (`t2-chimprof-samples.py`, `GROUPS`). The per-read cycle counters (section 2) give the same picture and agree with the sampler: chimeric code 33.0 % (pbmc5p_v1) / 39.2 % (GSM6250953) of `oneRead` by cycles against 30.9 % / 37.8 % by samples.

Grouping asked for in the brief, both samples (percent of samples; seconds = samples / 100 summed over 8 threads):

| group | GSM6250953 | pbmc5p_v1 |
|---|---|---|
| input parsing (readLoad, chunk read, adapter clip `localSearch`, chunk sort) | 1.76 % (86 s) | 2.34 % (17 s) |
| seed search (`maxMappableLength2strands`, `maxMappableLength`, `compareSeqToGenome`, `storeAligns`) | 21.66 % (1,060 s) | 21.09 % (157 s) |
| seed storage / windows (`stitchPieces` self incl. inlined window creation and assignment, `createExtendWindowsWithAlign`, `assignAlignToWindow`) | 9.25 % (453 s) | 13.13 % (98 s) |
| stitching (`stitchWindowAligns`, `stitchAlignToTranscript`, `extendAlign`, `Transcript` copies, `sjAlignSplit`) | 28.26 % (1,383 s) | 30.90 % (229 s) |
| multimapper selection | 0.19 % (9 s) | 0.26 % (2 s) |
| chimeric: pair loop (`chimericDetectionMult`, `ChimericSegment`, `chimericAlignScore`) | 36.90 % (1,805 s) | 28.91 % (215 s) |
| chimeric: `chimericStitching` | 0.65 % (32 s) | 1.36 % (10 s) |
| chimeric: junction line output | 0.21 % (10 s) | 0.59 % (4 s) |
| chimeric: other (`ChimericAlign` construct and check, entry) | 0.02 % | 0.03 % |
| Solo / barcode (`getCBandUMI`, `hashLookupCB`, `addStats`) | 0.70 % (34 s) | 0.97 % (7 s) |
| output (`outputAlignments`) | 0.12 % (6 s) | 0.08 % (1 s) |
| per-read driver (`oneRead` self) and other | 0.28 % | 0.34 % |
| locks and waiting | 0 | 0 |

Tail: the threads' sample counts differ by 0.47 % (GSM6250953, 60,945 to 61,445 samples) and 2.7 % (pbmc5p_v1, 8,424 to 9,531) of the thread time: the end of the mapping, where threads that run out of chunks wait for the others.

### GSM6250953 (job 819071, `samp-fp`): groups, self and inclusive time

```
# t2-chimprof stack samples of /tmp/debruinz-chimprof-run-819071/gsm/samp-fp/p: 489242 samples of 8 threads at 100 Hz (wall clock): 4892.4 thread-seconds
samples whose first backtrace frame equals the interrupted pc: 100.0 %; leaf unresolved (no symbol): 2.4 %
per thread samples: 60965 61282 60945 61255 61271 61101 60978 61445

## groups (innermost frame that belongs to a group; percent of all samples; seconds = samples / Hz summed over threads)
chimeric: pair loop (chimericDetectionMult, segments, score)                                 180529   36.90 %     1805.3 thread-s
stitching (stitchWindowAligns, stitchAlignToTranscript, extendAlign, scoring)                138284   28.26 %     1382.8 thread-s
seed search (maxMappableLength2strands, suffix array)                                        105965   21.66 %     1059.7 thread-s
seed storage / windows (createExtendWindowsWithAlign, assignAlignToWindow, stitchPieces)      45255    9.25 %      452.6 thread-s
input parsing (readLoad, chunk read, clip, chunk sort)                                         8599    1.76 %       86.0 thread-s
Solo / barcode (getCBandUMI, Solo records)                                                     3424    0.70 %       34.2 thread-s
chimeric: stitching (chimericStitching)                                                        3192    0.65 %       31.9 thread-s
per-read driver (oneRead self, qualHist)                                                       1080    0.22 %       10.8 thread-s
chimeric: junction output (chimericJunctionOutput, formatting)                                 1037    0.21 %       10.4 thread-s
multimapper selection (multMapSelect, mappedFilter)                                             928    0.19 %        9.3 thread-s
output (outputAlignments, SAM/unmapped/SJ output)                                               578    0.12 %        5.8 thread-s
other (main binary, no group)                                                                   306    0.06 %        3.1 thread-s
chimeric: ChimericAlign other (check, construct, generateCigar)                                  34    0.01 %        0.3 thread-s
chimeric: ReadAlign::chimericDetection (entry)                                                   26    0.01 %        0.3 thread-s
other (libc/other library)                                                                        5    0.00 %        0.1 thread-s
```

```
## self time, top 25 (innermost frame)
 17.24 %  ChimericSegment::ChimericSegment
 15.97 %  compareSeqToGenome
  9.73 %  chimericAlignScore
  8.35 %  ChimericDetection::chimericDetectionMult
  7.20 %  stitchWindowAligns
  7.09 %  stitchAlignToTranscript
  6.93 %  ReadAlign::stitchPieces
  6.36 %  extendAlign
  2.75 %  ReadAlign::maxMappableLength2strands
  1.94 %  Transcript::operator=
  1.61 %  maxMappableLength
  1.55 %  ChimericSegment::segmentCheck
  1.31 %  sjAlignSplit
  1.05 %  ReadAlign::assignAlignToWindow
  0.96 %  [libc.so.6 unresolved] <- stitchWindowAligns
  0.88 %  std::vector<int, std::allocator >::operator=(std::vector<int, std::allocator > const&)
  0.83 %  localSearch
  0.81 %  ReadAlign::createExtendWindowsWithAlign
  0.76 %  binarySearch2
  0.71 %  ReadAlign::storeAligns
  0.60 %  ChimericAlign::chimericStitching
  0.48 %  [libm.so.6 unresolved] <- stitchWindowAligns
  0.48 %  blocksOverlap
  0.45 %  findMultRange
  0.38 %  [libc.so.6 unresolved] <- ReadAlign::stitchPieces
```

```
## inclusive time, top 25 (function anywhere in the stack)
 99.07 %  [libc.so.6 unresolved]
 99.06 %  ReadAlignChunk::processChunks
 98.91 %  ReadAlignChunk::mapChunk
 98.58 %  ReadAlign::oneRead
 86.55 %  ThreadControl::threadRAprocessChunks
 58.72 %  ReadAlign::mapOneRead
 37.84 %  ReadAlign::chimericDetection
 36.71 %  ReadAlign::stitchPieces
 24.31 %  stitchWindowAligns
 21.47 %  ReadAlign::maxMappableLength2strands
 17.78 %  maxMappableLength
 17.66 %  ChimericDetection::chimericDetectionMult
 17.24 %  ChimericSegment::ChimericSegment
 15.97 %  compareSeqToGenome
 12.36 %  mapThreadsSpawn
 12.35 %  star_main_impl
 12.35 %  singlet_main
 12.34 %  __libc_start_main@GLIBC_2.2.5
 12.34 %  _start
  9.73 %  chimericAlignScore
  7.85 %  stitchAlignToTranscript
  6.36 %  extendAlign
  2.38 %  Transcript::operator=
  2.17 %  findMultRange
  1.55 %  ChimericSegment::segmentCheck
```

### pbmc5p_v1 (job 819071, `samp-fp`): groups, self and inclusive time

```
# t2-chimprof stack samples of /tmp/debruinz-chimprof-run-819071/pbmc5p_v1/samp-fp/p: 74223 samples of 8 threads at 100 Hz (wall clock): 742.2 thread-seconds
samples whose first backtrace frame equals the interrupted pc: 100.0 %; leaf unresolved (no symbol): 3.2 %
per thread samples: 9526 9517 9258 9531 8424 9498 9488 8981

## groups (innermost frame that belongs to a group; percent of all samples; seconds = samples / Hz summed over threads)
stitching (stitchWindowAligns, stitchAlignToTranscript, extendAlign, scoring)                 22932   30.90 %      229.3 thread-s
chimeric: pair loop (chimericDetectionMult, segments, score)                                  21458   28.91 %      214.6 thread-s
seed search (maxMappableLength2strands, suffix array)                                         15654   21.09 %      156.5 thread-s
seed storage / windows (createExtendWindowsWithAlign, assignAlignToWindow, stitchPieces)       9745   13.13 %       97.5 thread-s
input parsing (readLoad, chunk read, clip, chunk sort)                                         1734    2.34 %       17.3 thread-s
chimeric: stitching (chimericStitching)                                                        1009    1.36 %       10.1 thread-s
Solo / barcode (getCBandUMI, Solo records)                                                      721    0.97 %        7.2 thread-s
chimeric: junction output (chimericJunctionOutput, formatting)                                  439    0.59 %        4.4 thread-s
multimapper selection (multMapSelect, mappedFilter)                                             193    0.26 %        1.9 thread-s
per-read driver (oneRead self, qualHist)                                                        158    0.21 %        1.6 thread-s
other (main binary, no group)                                                                   100    0.13 %        1.0 thread-s
output (outputAlignments, SAM/unmapped/SJ output)                                                61    0.08 %        0.6 thread-s
chimeric: ChimericAlign other (check, construct, generateCigar)                                  12    0.02 %        0.1 thread-s
chimeric: ReadAlign::chimericDetection (entry)                                                    7    0.01 %        0.1 thread-s
```

```
## self time, top 25 (innermost frame)
 16.40 %  compareSeqToGenome
 12.53 %  ChimericSegment::ChimericSegment
 10.83 %  ReadAlign::stitchPieces
  8.73 %  stitchWindowAligns
  8.05 %  chimericAlignScore
  7.41 %  extendAlign
  6.98 %  ChimericDetection::chimericDetectionMult
  5.34 %  stitchAlignToTranscript
  2.13 %  Transcript::operator=
  2.05 %  ReadAlign::maxMappableLength2strands
  1.71 %  sjAlignSplit
  1.60 %  maxMappableLength
  1.24 %  ChimericSegment::segmentCheck
  1.23 %  ChimericAlign::chimericStitching
  1.17 %  [libc.so.6 unresolved] <- stitchWindowAligns
  1.10 %  ReadAlign::assignAlignToWindow
  1.02 %  std::vector<int, std::allocator >::operator=(std::vector<int, std::allocator > const&)
  0.86 %  ReadAlign::createExtendWindowsWithAlign
  0.81 %  localSearch
  0.80 %  binarySearch2
  0.63 %  blocksOverlap
  0.54 %  [libm.so.6 unresolved] <- stitchWindowAligns
  0.49 %  Transcript::Transcript
  0.41 %  findMultRange
  0.41 %  ReadAlign::storeAligns
```

```
## inclusive time, top 25 (function anywhere in the stack)
 98.43 %  ReadAlignChunk::processChunks
 98.41 %  [libc.so.6 unresolved]
 97.94 %  ReadAlignChunk::mapChunk
 97.48 %  ReadAlign::oneRead
 85.64 %  ThreadControl::threadRAprocessChunks
 64.03 %  ReadAlign::mapOneRead
 42.73 %  ReadAlign::stitchPieces
 31.08 %  ReadAlign::chimericDetection
 25.81 %  stitchWindowAligns
 20.85 %  ReadAlign::maxMappableLength2strands
 18.26 %  maxMappableLength
 16.40 %  compareSeqToGenome
 16.14 %  ChimericDetection::chimericDetectionMult
 12.57 %  mapThreadsSpawn
 12.55 %  star_main_impl
 12.54 %  singlet_main
 12.53 %  ChimericSegment::ChimericSegment
 12.53 %  __libc_start_main@GLIBC_2.2.5
 12.50 %  _start
  8.05 %  chimericAlignScore
  7.41 %  extendAlign
  6.13 %  stitchAlignToTranscript
  2.76 %  findMultRange
  2.67 %  Transcript::operator=
  1.71 %  sjAlignSplit
```

Reading the lists: `ChimericSegment::ChimericSegment`, `chimericAlignScore` and `chimericDetectionMult` self time are the pair loop (the constructor is run for every pair); `compareSeqToGenome` is the suffix-array search; `ReadAlign::stitchPieces` self time is the window creation, flanks and seed assignment loops plus the per-window transcript copies; `Transcript::operator=` and the copies of `std::vector<int>` are the `Transcript` copies of the stitching recursion; `[libc.so.6 unresolved] <- stitchWindowAligns` is memcpy of those copies.

## 2. Per-read cost distribution

Cycle counters of `oneRead` for every read (rdtsc, TSC 2.095 GHz), sum over threads. Phase table, all reads:

GSM6250953 (17,636,124 reads, 283.5 us per read, sum 5,000.4 CPU-s against 5,018.9 s user of the process):
```
## 1. where the cycles of oneRead go (all reads)
cLoad             61.5 CPU-s   1.23 %      3.49 us/read
cSeed           1067.0 CPU-s  21.34 %     60.50 us/read
cWinCreate        96.0 CPU-s   1.92 %      5.44 us/read
cWinFlank         18.8 CPU-s   0.38 %      1.07 us/read
cWinAssign       385.2 CPU-s   7.70 %     21.84 us/read
cStitch         1344.8 CPU-s  26.89 %     76.25 us/read
cMult              9.6 CPU-s   0.19 %      0.54 us/read
cBar              22.5 CPU-s   0.45 %      1.27 us/read
cChim           1960.2 CPU-s  39.20 %    111.15 us/read
cOut              16.1 CPU-s   0.32 %      0.91 us/read
other             18.8 CPU-s   0.38 %   (oneRead total minus the phases above: hand-off, qualHist, mappedFilter, transformGenome, loop overhead)
inside cChim: stitching 38.9 CPU-s (2.0 % of cChim), junction output 17.1 CPU-s (0.9 %), rest (pair loop, segment checks, allocation) 1904.3 CPU-s
```

pbmc5p_v1 (4,535,057 reads, 166.4 us per read, sum 754.7 CPU-s against 759.3 s user):
```
## 1. where the cycles of oneRead go (all reads)
cLoad             10.6 CPU-s   1.40 %      2.33 us/read
cSeed            157.5 CPU-s  20.86 %     34.72 us/read
cWinCreate        13.8 CPU-s   1.83 %      3.04 us/read
cWinFlank          3.3 CPU-s   0.44 %      0.74 us/read
cWinAssign        90.3 CPU-s  11.96 %     19.90 us/read
cStitch          216.3 CPU-s  28.66 %     47.69 us/read
cMult              2.2 CPU-s   0.30 %      0.49 us/read
cBar               4.9 CPU-s   0.64 %      1.07 us/read
cChim            249.4 CPU-s  33.04 %     54.99 us/read
cOut               3.0 CPU-s   0.39 %      0.66 us/read
other              3.5 CPU-s   0.47 %   (oneRead total minus the phases above: hand-off, qualHist, mappedFilter, transformGenome, loop overhead)
inside cChim: stitching 13.4 CPU-s (5.4 % of cChim), junction output 6.9 CPU-s (2.7 %), rest (pair loop, segment checks, allocation) 229.1 CPU-s
```

Distribution (share of the total of the `oneRead` cycles taken by the costliest reads):
```
## 2. cost distribution (oneRead cycles per read)
costliest   0.1 % of reads (17636 reads): 4.22 % of the CPU; their mean 11964 us, smallest of them 5459 us
costliest   1.0 % of reads (176361 reads): 13.03 % of the CPU; their mean 3694 us, smallest of them 1853 us
costliest  10.0 % of reads (1763612 reads): 41.96 % of the CPU; their mean 1190 us, smallest of them 588 us
costliest  50.0 % of reads (8818062 reads): 83.91 % of the CPU; their mean 476 us, smallest of them 153 us
per-read us: p10 64.1 median 152.7 p90 588.5 p99 1853.1 p99.9 5458.5 max 329527.4; mean 283.5
```
```
## 2. cost distribution (oneRead cycles per read)
costliest   0.1 % of reads (4535 reads): 4.11 % of the CPU; their mean 6836 us, smallest of them 3539 us
costliest   1.0 % of reads (45351 reads): 12.24 % of the CPU; their mean 2036 us, smallest of them 930 us
costliest  10.0 % of reads (453506 reads): 36.78 % of the CPU; their mean 612 us, smallest of them 308 us
costliest  50.0 % of reads (2267528 reads): 79.96 % of the CPU; their mean 266 us, smallest of them 111 us
per-read us: p10 44.2 median 110.9 p90 308.1 p99 930.3 p99.9 3537.8 max 282091.4; mean 166.4
```

Per-read cost is high for the many reads with many windows: reads with more than 20 windows are 69.6 % of GSM6250953 reads and 90.4 % of the CPU (368 us per read); on pbmc5p_v1 65.0 % of the reads and 83.1 % of the CPU (213 us per read). By stored seeds: reads with 21 or more seeds are 72 % of GSM6250953 reads and 90.2 % of the CPU.

### Class of all reads and of the costliest 1 % (counts only; sequences are not printed)

GSM6250953, all reads:
```
### all reads (17636124 reads, 100.00 % of the CPU)
  Lread     mean    149.55  median     150  p90     150  max      150
  nP        mean     29.10  median      27  p90      46  max       76
  nA        mean   3879.01  median     258  p90   12556  max   115245
  nWinMade  mean     49.72  median      41  p90     103  max      356
  nW        mean     47.70  median      38  p90     102  max      356
  nTr       mean      1.68  median       1  p90       3  max      328
  maxScore/Lread: mean 0.463; maxScore == 0: 3.4 %
  nTr>1 (multimapping): 19.4 %   nTr==0: 3.4 %   nTr>outFilterMultimapNmax(10): 0.8 %
  flag chim_called                                           100.00 %
  flag chim_trigger                                           96.79 %
  flag chim_pairloop                                          93.35 %
  flag chim_record                                            18.36 %
  flag limit seedMultimapNmax (a seed dropped)                53.46 %
  flag limit seedPerWindowNmax                                 5.69 %
  flag limit alignWindowsPerReadNmax                           0.00 %
  flag limit alignTranscriptsPerReadNmax                       0.00 %
  flag limit alignTranscriptsPerWindowNmax                     0.21 %
  flag seed above winAnchorMultimapNmax (not an anchor)       51.39 %
  flag limit seedPerReadNmax                                   0.00 %
  flag marker too many anchors per window                      0.00 %
  flag chimN > chimMultimapNmax                                0.05 %
  chimVisited  sum  85258057368  mean   4834.29  max  11127403
  chimScored   sum  80285145902  mean   4552.31  max  11127403
  chimCand     sum     12773851  mean      0.72  max      4747
  chimCheckOk  sum     12729775  mean      0.72  max      4747
  chimStitched sum     12729775  mean      0.72  max      4747
  chimSurvive  sum      9915236  mean      0.56  max      4739
  chimLines    sum      6706877  mean      0.38  max        50
```

GSM6250953, costliest 1 % (176,361 reads, 13.03 % of the CPU):
```
### costliest 1 % (176361 reads, 13.03 % of the CPU)
  Lread     mean    149.55  median     150  p90     150  max      150
  nP        mean     43.25  median      45  p90      57  max       76
  nA        mean  16640.11  median   14483  p90   29517  max    89464
  nWinMade  mean    115.48  median     119  p90     173  max      356
  nW        mean    115.46  median     119  p90     173  max      356
  nTr       mean      3.26  median       2  p90       7  max      328
  maxScore/Lread: mean 0.283; maxScore == 0: 0.0 %
  nTr>1 (multimapping): 55.1 %   nTr==0: 0.0 %   nTr>outFilterMultimapNmax(10): 3.5 %
  flag chim_called                                           100.00 %
  flag chim_trigger                                           99.47 %
  flag chim_pairloop                                          99.47 %
  flag chim_record                                             7.47 %
  flag limit seedMultimapNmax (a seed dropped)                81.55 %
  flag limit seedPerWindowNmax                                27.87 %
  flag limit alignWindowsPerReadNmax                           0.00 %
  flag limit alignTranscriptsPerReadNmax                       0.00 %
  flag limit alignTranscriptsPerWindowNmax                    11.45 %
  flag seed above winAnchorMultimapNmax (not an anchor)       99.93 %
  flag limit seedPerReadNmax                                   0.00 %
  flag marker too many anchors per window                      0.00 %
  flag chimN > chimMultimapNmax                                1.35 %
  chimVisited  sum  21795246424  mean 123583.14  max  11127403
  chimScored   sum  20999845663  mean 119073.07  max  11127403
  chimCand     sum      1261949  mean      7.16  max      4747
  chimCheckOk  sum      1258693  mean      7.14  max      4747
  chimStitched sum      1258693  mean      7.14  max      4747
  chimSurvive  sum      1062357  mean      6.02  max      4739
  chimLines    sum       121584  mean      0.69  max        50
```

pbmc5p_v1, all reads:
```
### all reads (4535057 reads, 100.00 % of the CPU)
  Lread     mean     90.16  median      91  p90      91  max       91
  nP        mean     14.99  median      14  p90      21  max       33
  nA        mean   3331.42  median     524  p90   10372  max    52758
  nWinMade  mean     34.38  median      27  p90      68  max      244
  nW        mean     33.97  median      27  p90      68  max      244
  nTr       mean      2.43  median       1  p90       5  max      311
  maxScore/Lread: mean 0.363; maxScore == 0: 2.3 %
  nTr>1 (multimapping): 49.9 %   nTr==0: 2.3 %   nTr>outFilterMultimapNmax(10): 1.8 %
  flag chim_called                                           100.00 %
  flag chim_trigger                                           98.85 %
  flag chim_pairloop                                          96.51 %
  flag chim_record                                            17.64 %
  flag limit seedMultimapNmax (a seed dropped)                64.50 %
  flag limit seedPerWindowNmax                                 3.83 %
  flag limit alignWindowsPerReadNmax                           0.00 %
  flag limit alignTranscriptsPerReadNmax                       0.00 %
  flag limit alignTranscriptsPerWindowNmax                     0.18 %
  flag seed above winAnchorMultimapNmax (not an anchor)       59.21 %
  flag limit seedPerReadNmax                                   0.00 %
  flag marker too many anchors per window                      0.00 %
  flag chimN > chimMultimapNmax                                0.16 %
  chimVisited  sum  10070406605  mean   2220.57  max  10032960
  chimScored   sum   9674646171  mean   2133.30  max  10032960
  chimCand     sum      7220866  mean      1.59  max      5800
  chimCheckOk  sum      7217243  mean      1.59  max      5800
  chimStitched sum      7217243  mean      1.59  max      5800
  chimSurvive  sum      5355061  mean      1.18  max      3032
  chimLines    sum      2832142  mean      0.62  max        50
```

pbmc5p_v1, costliest 1 % (45,351 reads, 12.24 % of the CPU):
```
### costliest 1 % (45351 reads, 12.24 % of the CPU)
  Lread     mean     90.07  median      91  p90      91  max       91
  nP        mean     18.06  median      19  p90      24  max       32
  nA        mean  11555.34  median   11238  p90   19697  max    48651
  nWinMade  mean     66.31  median      65  p90     115  max      244
  nW        mean     66.30  median      65  p90     115  max      244
  nTr       mean     10.64  median       3  p90      35  max      311
  maxScore/Lread: mean 0.443; maxScore == 0: 0.0 %
  nTr>1 (multimapping): 69.4 %   nTr==0: 0.0 %   nTr>outFilterMultimapNmax(10): 24.8 %
  flag chim_called                                           100.00 %
  flag chim_trigger                                           85.18 %
  flag chim_pairloop                                          85.17 %
  flag chim_record                                            17.16 %
  flag limit seedMultimapNmax (a seed dropped)                62.88 %
  flag limit seedPerWindowNmax                                24.39 %
  flag limit alignWindowsPerReadNmax                           0.00 %
  flag limit alignTranscriptsPerReadNmax                       0.00 %
  flag limit alignTranscriptsPerWindowNmax                    16.01 %
  flag seed above winAnchorMultimapNmax (not an anchor)       99.66 %
  flag limit seedPerReadNmax                                   0.00 %
  flag marker too many anchors per window                      0.00 %
  flag chimN > chimMultimapNmax                                5.33 %
  chimVisited  sum   2727979783  mean  60152.58  max  10032960
  chimScored   sum   2676890650  mean  59026.06  max  10032960
  chimCand     sum      1319012  mean     29.08  max      5800
  chimCheckOk  sum      1318954  mean     29.08  max      5800
  chimStitched sum      1318954  mean     29.08  max      5800
  chimSurvive  sum       827010  mean     18.24  max      3032
  chimLines    sum        63076  mean      1.39  max        50
```

Outcome classes of all reads:
```
## 4. outcome classes (all reads): mean us per read and share of the CPU
  nW == 0 (no window with an alignment)       605544 reads (  3.4 %)     112.9 us/read    1.37 % of CPU
  nW == 1                                       6092 reads (  0.0 %)      58.0 us/read    0.01 % of CPU
  nW 2-5                                      492183 reads (  2.8 %)      62.0 us/read    0.61 % of CPU
  nW 6-20                                    4250312 reads ( 24.1 %)      89.7 us/read    7.62 % of CPU
  nW > 20                                   12281993 reads ( 69.6 %)     368.0 us/read   90.39 % of CPU
  by read length (Lread after clip):
    Lread 0-89:        24 reads      5.3 us/read   0.00 % of CPU
    Lread 90-99:         2 reads     17.7 us/read   0.00 % of CPU
    Lread 100-129:       320 reads     86.6 us/read   0.00 % of CPU
    Lread 130-150:  17635778 reads    283.5 us/read 100.00 % of CPU
  by nP (stored seeds):
    nP 0-0:       493 reads     10.3 us/read   0.00 % of CPU
    nP 1-2:       358 reads     18.8 us/read   0.00 % of CPU
    nP 3-5:       994 reads     43.2 us/read   0.00 % of CPU
    nP 6-10:    174604 reads     58.3 us/read   0.20 % of CPU
    nP 11-20:   4738329 reads    101.2 us/read   9.59 % of CPU
    nP 21-inf:  12721346 reads    354.6 us/read  90.21 % of CPU
```
```
## 4. outcome classes (all reads): mean us per read and share of the CPU
  nW == 0 (no window with an alignment)       106545 reads (  2.3 %)      60.2 us/read    0.85 % of CPU
  nW == 1                                       5945 reads (  0.1 %)      61.6 us/read    0.05 % of CPU
  nW 2-5                                       51314 reads (  1.1 %)      74.8 us/read    0.51 % of CPU
  nW 6-20                                    1424768 reads ( 31.4 %)      82.2 us/read   15.51 % of CPU
  nW > 20                                    2946485 reads ( 65.0 %)     212.8 us/read   83.08 % of CPU
  by read length (Lread after clip):
    Lread 0-89:   1494209 reads    170.3 us/read  33.71 % of CPU
    Lread 90-99:   3040848 reads    164.5 us/read  66.29 % of CPU
  by nP (stored seeds):
    nP 0-0:       173 reads      7.6 us/read   0.00 % of CPU
    nP 1-2:       152 reads     23.5 us/read   0.00 % of CPU
    nP 3-5:      7477 reads     44.6 us/read   0.04 % of CPU
    nP 6-10:    636093 reads     76.3 us/read   6.43 % of CPU
    nP 11-20:   3347307 reads    155.6 us/read  69.02 % of CPU
    nP 21-inf:    543855 reads    340.2 us/read  24.51 % of CPU
```

Low-complexity and motif classes of the T2-chimpass 3.3 definitions (applied to the unclipped sequence of the pass input; the baseline is a random sample of 200,000 reads; the cost columns are per class, a read can be in several):
```
## 5. read classes (T2-chimpass 3.3 definitions, on the unclipped sequence of the pass input)
sequences found for 374323 of 374323 selected reads
### random sample (baseline) (200000 reads classified)
  polyA           0.44 % of the reads; mean    928.3 us/read;   1.44 % of the group CPU
  polyT           0.00 % of the reads; mean   2877.5 us/read;   0.02 % of the group CPU
  polyG           0.52 % of the reads; mean    459.1 us/read;   0.84 % of the group CPU
  polyC           0.00 % of the reads; mean    119.8 us/read;   0.00 % of the group CPU
  tandem2-6       0.14 % of the reads; mean   1135.6 us/read;   0.58 % of the group CPU
  entropy<1.5     9.74 % of the reads; mean    572.6 us/read;  19.71 % of the group CPU
  entropy<1.0     3.11 % of the reads; mean    649.4 us/read;   7.15 % of the group CPU
  hasN            0.58 % of the reads; mean    167.1 us/read;   0.34 % of the group CPU
  TSO5rc         31.00 % of the reads; mean    206.2 us/read;  22.59 % of the group CPU
  TSO5fwd         0.46 % of the reads; mean    204.8 us/read;   0.33 % of the group CPU
  TSO3core        0.36 % of the reads; mean    421.3 us/read;   0.54 % of the group CPU
  any low-complexity (mono, tandem, entropy<1.5)    9.74 % of the reads; mean    572.6 us/read;  19.71 % of the group CPU
### costliest 1 % (176361 reads classified)
  polyA           5.48 % of the reads; mean   3636.3 us/read;   5.39 % of the group CPU
  polyT           0.06 % of the reads; mean   3809.3 us/read;   0.07 % of the group CPU
  polyG           1.75 % of the reads; mean   9821.4 us/read;   4.64 % of the group CPU
  polyC           0.00 % of the reads; mean   3603.6 us/read;   0.00 % of the group CPU
  tandem2-6       2.97 % of the reads; mean   3925.0 us/read;   3.16 % of the group CPU
  entropy<1.5    40.05 % of the reads; mean   4714.7 us/read;  51.12 % of the group CPU
  entropy<1.0    19.76 % of the reads; mean   5437.2 us/read;  29.09 % of the group CPU
  hasN            0.18 % of the reads; mean   3703.6 us/read;   0.18 % of the group CPU
  TSO5rc         11.73 % of the reads; mean   3203.2 us/read;  10.17 % of the group CPU
  TSO5fwd         0.07 % of the reads; mean   2631.6 us/read;   0.05 % of the group CPU
  TSO3core        0.15 % of the reads; mean   2405.7 us/read;   0.10 % of the group CPU
  any low-complexity (mono, tandem, entropy<1.5)   40.05 % of the reads; mean   4714.7 us/read;  51.12 % of the group CPU
### costliest 0.1 % (17636 reads classified)
  polyA           7.30 % of the reads; mean   8391.4 us/read;   5.12 % of the group CPU
  polyT           0.09 % of the reads; mean   8909.4 us/read;   0.06 % of the group CPU
  polyG           7.48 % of the reads; mean  18701.5 us/read;  11.69 % of the group CPU
  polyC           0.00 % of the reads; mean      0.0 us/read;   0.00 % of the group CPU
  tandem2-6       4.52 % of the reads; mean   8300.1 us/read;   3.14 % of the group CPU
  entropy<1.5    68.37 % of the reads; mean  13642.3 us/read;  77.96 % of the group CPU
  entropy<1.0    42.52 % of the reads; mean  14403.9 us/read;  51.19 % of the group CPU
  hasN            0.24 % of the reads; mean  10001.8 us/read;   0.20 % of the group CPU
  TSO5rc          4.18 % of the reads; mean   6779.6 us/read;   2.37 % of the group CPU
  TSO5fwd         0.02 % of the reads; mean   6035.0 us/read;   0.01 % of the group CPU
  TSO3core        0.02 % of the reads; mean   9062.2 us/read;   0.01 % of the group CPU
  any low-complexity (mono, tandem, entropy<1.5)   68.37 % of the reads; mean  13642.3 us/read;  77.96 % of the group CPU
```
```
## 5. read classes (T2-chimpass 3.3 definitions, on the unclipped sequence of the pass input)
sequences found for 243392 of 243392 selected reads
### random sample (baseline) (200000 reads classified)
  polyA           0.29 % of the reads; mean    293.9 us/read;   0.50 % of the group CPU
  polyT           0.01 % of the reads; mean    175.3 us/read;   0.01 % of the group CPU
  polyG           0.74 % of the reads; mean     71.1 us/read;   0.32 % of the group CPU
  polyC           0.01 % of the reads; mean     69.2 us/read;   0.01 % of the group CPU
  tandem2-6       0.59 % of the reads; mean    674.6 us/read;   2.38 % of the group CPU
  entropy<1.5    29.47 % of the reads; mean    191.9 us/read;  33.88 % of the group CPU
  entropy<1.0     6.02 % of the reads; mean    236.5 us/read;   8.53 % of the group CPU
  hasN            0.34 % of the reads; mean    254.1 us/read;   0.52 % of the group CPU
  TSO5rc          0.96 % of the reads; mean    114.1 us/read;   0.65 % of the group CPU
  TSO5fwd         0.35 % of the reads; mean    112.4 us/read;   0.23 % of the group CPU
  TSO3core       36.37 % of the reads; mean     86.4 us/read;  18.82 % of the group CPU
  any low-complexity (mono, tandem, entropy<1.5)   29.48 % of the reads; mean    191.9 us/read;  33.88 % of the group CPU
### costliest 1 % (45351 reads classified)
  polyA           2.87 % of the reads; mean   1342.5 us/read;   1.89 % of the group CPU
  polyT           0.02 % of the reads; mean   1194.1 us/read;   0.01 % of the group CPU
  polyG           0.82 % of the reads; mean   4376.9 us/read;   1.76 % of the group CPU
  polyC           0.01 % of the reads; mean   1745.7 us/read;   0.01 % of the group CPU
  tandem2-6      13.71 % of the reads; mean   2188.8 us/read;  14.74 % of the group CPU
  entropy<1.5    54.93 % of the reads; mean   2213.5 us/read;  59.71 % of the group CPU
  entropy<1.0    23.78 % of the reads; mean   2200.0 us/read;  25.69 % of the group CPU
  hasN            1.13 % of the reads; mean   2154.4 us/read;   1.20 % of the group CPU
  TSO5rc          0.11 % of the reads; mean   1682.5 us/read;   0.09 % of the group CPU
  TSO5fwd         0.01 % of the reads; mean   1262.9 us/read;   0.00 % of the group CPU
  TSO3core        1.69 % of the reads; mean   1369.2 us/read;   1.14 % of the group CPU
  any low-complexity (mono, tandem, entropy<1.5)   54.94 % of the reads; mean   2213.4 us/read;  59.72 % of the group CPU
### costliest 0.1 % (4535 reads classified)
  polyA           0.02 % of the reads; mean   6226.8 us/read;   0.02 % of the group CPU
  polyT           0.00 % of the reads; mean      0.0 us/read;   0.00 % of the group CPU
  polyG           2.25 % of the reads; mean  11450.6 us/read;   3.77 % of the group CPU
  polyC           0.00 % of the reads; mean      0.0 us/read;   0.00 % of the group CPU
  tandem2-6      20.13 % of the reads; mean   5646.3 us/read;  16.63 % of the group CPU
  entropy<1.5    65.05 % of the reads; mean   7257.5 us/read;  69.06 % of the group CPU
  entropy<1.0    31.31 % of the reads; mean   6525.0 us/read;  29.89 % of the group CPU
  hasN            0.77 % of the reads; mean  11882.9 us/read;   1.34 % of the group CPU
  TSO5rc          0.07 % of the reads; mean   4843.1 us/read;   0.05 % of the group CPU
  TSO5fwd         0.00 % of the reads; mean      0.0 us/read;   0.00 % of the group CPU
  TSO3core        0.31 % of the reads; mean   5608.2 us/read;   0.25 % of the group CPU
  any low-complexity (mono, tandem, entropy<1.5)   65.05 % of the reads; mean   7257.5 us/read;  69.06 % of the group CPU
```

Findings: (a) the costliest reads are those with many windows (115.5 against 49.7 on average on GSM6250953; 66.3 against 34.4 on pbmc5p_v1) and many stored SA loci (`nA` 16,640 against 3,879; 11,555 against 3,331), most of them non-anchor seeds; (b) 81.6 % (GSM) and 62.9 % (pbmc) of them have a seed dropped by `seedMultimapNmax`, 27.9 % / 24.4 % hit `seedPerWindowNmax`, 11.5 % / 16.0 % hit `alignTranscriptsPerWindowNmax`; `alignWindowsPerReadNmax`, `alignTranscriptsPerReadNmax` and `seedPerReadNmax` are never hit by any read; (c) the chimeric detection ran (trigger passed) for 99.5 % (GSM) and 85.2 % (pbmc) of the costliest 1 %, found a line for 7.5 % and 17.2 % of them; (d) low-complexity reads (entropy under 1.5 bits) are 40.1 % (GSM) and 54.3 % (pbmc) of the costliest 1 %, 68.4 % and 64.8 % of the costliest 0.1 %, against 9.7 % and 29.6 % of a random sample, and they take 51 % and 59 % of the CPU of that 1 %.

## 3. Entry into the chimeric detection (counters, every read)

Conditions, from the code (`ReadAlign_oneRead.cpp`, `ReadAlign_chimericDetection.cpp`, `ChimericDetection_chimericDetectionMult.cpp`; values of the pass in brackets):
1. `oneRead` mapped the read itself (not a cache hit, not a GPU hand-off or E10 record: none applies in the pass) and `!peOv.yes` (no mate merging: single mate here); `getCBandUMI` is called first for the junction line (`P.pCh.segmentMin>0 && P.pSolo.type>0`).
2. `chimericDetection()`: `P.pCh.segmentMin > 0` (10), `P.outFilterBySJoutStage <= 1` (0), `P.pCh.multimapNmax != 0` (50; 0 would select the old detection).
3. The trigger: `trBest->maxScore <= readLength[0]+readLength[1] - P.pCh.nonchimScoreDropMin` (the cDNA mate's clipped length minus 20; the barcode mate is not an alignment mate). A read without a window has `trBest = trInit`, `maxScore` 0 and passes.
4. In `chimericDetectionMult`: `minScoreToConsider = max(chimScoreMin 1, trBest->maxScore + 1, Lread - chimScoreDropMax 30)`. For every transcript (window, alignment) the segment check: `rLength >= chimSegmentMin` (10) and `intronMotifs[0] == 0`; for every later transcript the same, consistent strands, then `chimericAlignScore`: both segments longer than `chimSegmentMin + overlap`, different mates or a read gap of at most `chimSegmentReadGapMax` (3), score = sum of the two `maxScore` minus the overlap; a pair counts when the score is at least `minScoreToConsider`; then `chimericCheck` (junction overhang at least `chimJunctionOverhangMin` 10), `chimericStitching` (rescoring, can fail), the best score raising `minScoreToConsider` by `chimMultimapScoreRange`; no line when no candidate survived or `chimN > chimMultimapNmax` (50).

Counters of the pass (a window-pair combination is a visited pair of transcripts `i < j` in window order, `chimVisited`; scored = after the segment and strand checks, `chimericAlignScore` called):

| | GSM6250953 | pbmc5p_v1 |
|---|---|---|
| reads in the pass (distinct sequences) | 17,636,124 | 4,535,057 |
| entered `chimericDetection()` past the early returns | 17,636,124 (100 %) | 4,535,057 (100 %) |
| passed the trigger (`chimericDetectionMult` called) | 17,069,216 (96.79 %) | 4,483,075 (98.85 %) |
| reached the pair loop (a segment passed `segmentCheck`) | 16,463,693 (93.35 %) | 4,376,608 (96.51 %) |
| window-pair combinations visited | 85,258,057,368 | 10,070,406,605 |
| scored by `chimericAlignScore` | 80,285,145,902 | 9,674,646,171 |
| candidates (score at least `minScoreToConsider`) | 12,773,851 | 7,220,866 |
| passing `chimericCheck` = stitched | 12,729,775 | 7,217,243 |
| surviving the stitching | 9,915,236 | 5,355,061 |
| reads with at least one line / lines | 3,237,917 (18.36 %) / 6,706,877 | 799,963 (17.64 %) / 2,832,142 |
| CPU of `chimericDetection()` | 1,960.2 CPU-s = 39.2 % of `oneRead` | 249.4 CPU-s = 33.0 % |
| of it, on reads that end with no junction line | 1,820.8 CPU-s = **92.9 %** | 191.8 CPU-s = **76.9 %** |
| of it, `chimericStitching` / line output / the rest (pair loop, segment objects) | 38.9 / 17.1 / 1,904.3 CPU-s | 13.4 / 6.9 / 229.1 CPU-s |
| barcode parse (`getCBandUMI`) before it | 22.5 CPU-s = 0.45 % | 4.9 CPU-s = 0.64 % |

Full blocks (with the split of the chimeric time by the size of the pair loop; in both samples the reads with more than 1,000 scored pairs take 94.0 % (GSM) and 85.6 % (pbmc) of the chimeric time, 8.6 M and 1.7 M reads):

```
## 3. chimeric detection (counters of every read; trigger = trBest->maxScore <= Lread - chimNonchimScoreDropMin(20), see T2-chimprof.md)
reads mapped by the pass 17636124
reads that entered chimericDetection() past its early returns: 17636124 (100.00 %)
reads passing the trigger (chimericDetectionMult called): 17069216 (96.79 %)
reads that reached the pair loop (some segment passed segmentCheck): 16463693 (93.35 %)
reads with at least one junction line: 3237917 (18.36 %); lines 6706877
total chimVisited  85258057368
total chimScored   80285145902
total chimCand     12773851
total chimCheckOk  12729775
total chimStitched 12729775
total chimSurvive  9915236
chimericDetection() time: 1960.2 CPU-s = 39.20 % of oneRead; reads with no junction line: 1820.8 CPU-s = 92.88 % of it
  reads passing the trigger without a line: 1820.7 CPU-s (92.88 % of chim time); reads not passing the trigger: 0.0 CPU-s (0.00 %)
  reads in the pair loop without a line: 1820.7 CPU-s (92.88 %)
  barcode parse (getCBandUMI, once per read before chimericDetection): 22.5 CPU-s = 0.45 % of oneRead
  chim time by pair-loop size (chimScored per read): 
    chimScored 0-0: 1173200 reads, chim 0.1 CPU-s (0.00 % of chim), stitching 0.0 CPU-s
    chimScored 1-1: 37942 reads, chim 0.3 CPU-s (0.01 % of chim), stitching 0.1 CPU-s
    chimScored 2-10: 221198 reads, chim 1.6 CPU-s (0.08 % of chim), stitching 0.9 CPU-s
    chimScored 11-100: 1774215 reads, chim 15.8 CPU-s (0.81 % of chim), stitching 7.3 CPU-s
    chimScored 101-1000: 5827990 reads, chim 99.7 CPU-s (5.09 % of chim), stitching 18.9 CPU-s
    chimScored 1001-inf: 8601579 reads, chim 1842.7 CPU-s (94.01 % of chim), stitching 11.7 CPU-s
```
```
## 3. chimeric detection (counters of every read; trigger = trBest->maxScore <= Lread - chimNonchimScoreDropMin(20), see T2-chimprof.md)
reads mapped by the pass 4535057
reads that entered chimericDetection() past its early returns: 4535057 (100.00 %)
reads passing the trigger (chimericDetectionMult called): 4483075 (98.85 %)
reads that reached the pair loop (some segment passed segmentCheck): 4376608 (96.51 %)
reads with at least one junction line: 799963 (17.64 %); lines 2832142
total chimVisited  10070406605
total chimScored   9674646171
total chimCand     7220866
total chimCheckOk  7217243
total chimStitched 7217243
total chimSurvive  5355061
chimericDetection() time: 249.4 CPU-s = 33.04 % of oneRead; reads with no junction line: 191.8 CPU-s = 76.91 % of it
  reads passing the trigger without a line: 191.8 CPU-s (76.91 % of chim time); reads not passing the trigger: 0.0 CPU-s (0.00 %)
  reads in the pair loop without a line: 191.8 CPU-s (76.90 %)
  barcode parse (getCBandUMI, once per read before chimericDetection): 4.9 CPU-s = 0.64 % of oneRead
  chim time by pair-loop size (chimScored per read): 
    chimScored 0-0: 159400 reads, chim 0.0 CPU-s (0.01 % of chim), stitching 0.0 CPU-s
    chimScored 1-1: 10123 reads, chim 0.1 CPU-s (0.02 % of chim), stitching 0.0 CPU-s
    chimScored 2-10: 27052 reads, chim 0.2 CPU-s (0.07 % of chim), stitching 0.1 CPU-s
    chimScored 11-100: 526664 reads, chim 2.8 CPU-s (1.12 % of chim), stitching 0.8 CPU-s
    chimScored 101-1000: 2086159 reads, chim 32.8 CPU-s (13.15 % of chim), stitching 3.9 CPU-s
    chimScored 1001-inf: 1725659 reads, chim 213.6 CPU-s (85.63 % of chim), stitching 8.7 CPU-s
```

The maximum of visited pairs of one read is 11,127,403 (GSM) and 10,032,960 (pbmc). A candidate costs the stitching and allocation only: 12.7 M stitchings took 38.9 CPU-s.

## 4. Candidates for a speed-up

Seconds are CPU-seconds (thread-seconds) of the standalone pass at 8 threads on job 819071's node, from the per-read counters and the sampler; wall estimates divide by 8 and assume linear scaling (marked estimates). Nothing below was implemented.

### Exact (same junction lines for every read)

**1. Restrict the pair loop to the transcripts that can be part of a candidate** (`ChimericDetection_chimericDetectionMult.cpp` lines 50-102, the two nested loops from `ChimericSegment seg1(...)` to the end of the `iW1` loop, and `ChimericSegment::ChimericSegment`/`segmentCheck`).
- Time addressed: GSM6250953 1,904 CPU-s of the counters (the "rest" of the chimeric time: 38.1 % of `oneRead`; the sampler: 1,805 thread-s in the pair loop group = 36.9 %); pbmc5p_v1 229 CPU-s (30.4 %; sampler 215 thread-s = 28.9 %).
- Estimate of the saving (estimate): at least 90 % of it, i.e. about 1,700 to 1,850 CPU-s on GSM6250953 (about 210 to 230 s of the 638 s standalone pass at 8 threads, 33 to 36 %) and about 200 CPU-s on pbmc5p_v1 (about 25 s of 108 s, 23 %). What stays: one pass over the n transcripts of the read (n is about 48 windows on average, one or a few transcripts each) to find the participants, and the pairs among the participants; the 12.7 M (7.2 M) candidates and their stitching are unchanged.
- Why the output cannot change (proof sketch from the code). The visited pairs are exactly the pairs `i < j` of the flattened list of transcripts `trAll[iW][iA]` (window order, then alignment order), visited in lexicographic order. For a pair the loop does, with no side effect before the score test: construct two `ChimericSegment`s (read-only on the transcripts), `segmentCheck` for both (`rLength >= segmentMin` and `intronMotifs[0] == 0`), the strand test, `chimericAlignScore`, which returns 0 or `maxScore_i + maxScore_j - overlap` with `overlap >= 0` (unsigned), and acts only if the score is at least `minScoreToConsider`. `minScoreToConsider` never decreases inside the loop (it is raised once a better chimera is stored). Let `min0` be its value before the loop and `maxS` the largest `maxScore` over the transcripts that pass `segmentCheck`. A pair can act only if `maxScore_i + maxScore_j >= min0`, which needs `maxScore_i >= min0 - maxS` and `maxScore_j >= min0 - maxS`. So let P be the transcripts that pass `segmentCheck` and have `maxScore >= min0 - maxS` (a pass over the n transcripts, reading `rLength`, `intronMotifs[0]`, `maxScore`). Looping over `i < j` in P in the same order visits every pair that can act, in the same order, with the same `minScoreToConsider` at each of them, and skips only pairs that the original code rejects without effect. The candidates, `chimAligns`, `chimScoreBest`, the order of the stored chimeras and the lines are therefore the same. (`trBest->maxScore` is the maximum of all `maxScore`, so P is also empty when `min0 > 2 * maxS`.)
- Verification: the standalone job on the kept inputs (`t2-chimprof-run.sbatch`, `ARMS=base`) must give the sorted-raw-line hashes above (`643f187626f6ffaa67d0fe9acf324d42`, 2,832,142 lines; `c2a0f02ef3d48de17e89fab98b17c1ac`, 6,706,877 lines); then `t2-chimdedup.sbatch` on pbmc5p_v1 and GSM6250953 with `RAWNORM` equal to `d9e7359a0c10b09163e2622063149394` (3,926,694 lines) and `5202513e0679f97a1e700c494b3bfca2` (17,188,238 lines) and the table md5 `a5d71eb6...`, `3573f164...`, `1763c9dc...` (d300_5p); a unit test that feeds the old and the new function the same random transcript sets (including equal scores, overlaps, strands 0/1/2, `minScoreToConsider` raised mid-loop) and compares `chimAligns`.

**2. `extendAlign`** (6.36 % of the samples on GSM6250953, 7.41 % on pbmc5p_v1): a vectorized or word-wise comparison of read and genome with the same mismatch count and score bookkeeping would give the same result. Time addressed 311 thread-s (GSM), 55 thread-s (pbmc); the saving is not estimated; the proof is a per-function equivalence test on random inputs plus the hash check.

**3. Cheaper copies of `Transcript` in the stitching recursion** (`stitchWindowAligns.cpp` lines 19 and 317, `stitchPieces` `trA=*trInit` and `*(trAll[iW1][0])=trA`; `Transcript.h` members `cigar`, `varInd`, `varGenCoord`, `varReadCoord`, `varAllele`, `alignGenes`, which are empty in this configuration).
- Time addressed: `Transcript::operator=` 1.94 % + `std::vector<int>::operator=` 0.88 % + the libc rows under `stitchWindowAligns` and `stitchPieces` 0.96 % + 0.38 % of the samples on GSM6250953 (4.2 %, about 205 thread-s); on pbmc5p_v1 2.13 % + 1.02 % + 1.17 % (4.3 %, about 32 thread-s). An upper bound (the libc rows may include memset); the saving is a part of it (estimate: 1 to 2 % of the pass, 6 to 13 s of 638 s on GSM6250953).
- Why exact: the same values are assigned; only the work on empty containers and unused bytes is skipped. Verification: the sorted-raw-line hashes above.

**4. Skip the barcode whitelist matching in the pass** (`SoloReadBarcode::getCBandUMI` calls `matchCBtoWL`, `hashLookupCB`, `addStats`; the junction line prints the raw substrings `cbSeq`, `umiSeq` only, see T2-chimdedup 1.1).
- Time addressed: Solo/barcode group 0.70 % (34 thread-s, 4 s of wall) on GSM6250953, 0.97 % (7 thread-s) on pbmc5p_v1. Small; with `SINGLET_CHIM_SKIP_SOLO` (exists, off) the post-map is already skipped (about 10 s, T2-chimdedup 5).

**5. Share the genome index with the pool's copy** (section 5): 10 to 15 s of start per pass and 17,159,424 kB of private memory (Pss) of each running pass. Not a mapping cost.

**Not provable from the code (checked, not a candidate):**
- A pre-filter that proves "no junction line" from the seeds or from `trBest` before the stitching: a line needs two transcripts with `maxScore_i + maxScore_j >= min0` (at least `Lread - 30`); the score of a transcript is bounded by the read length only (extension in `stitchWindowAligns` can add the whole unaligned remainder of the read to a seed's window), so no seed-level bound excludes a window. Pruning windows before `stitchWindowAligns` is therefore not exact, and the per-window data to size it were not recorded.
- The seed search (21 to 22 %, `compareSeqToGenome` 16 %) and the window assignment (7.7 to 12 %): already optimized by the existing prefix sort of the chunk and `SINGLET_STAR_FASTASSIGN`; no repeated work across reads was found (the pass already maps each distinct sequence once).
- Allocation and locking: no locks (zero samples), no waits; `new Transcript` in `chimericStitching` is 12.7 M stitchings, 0.65 % (GSM) / 1.36 % (pbmc) of the samples; junction output formatting 0.2 % / 0.6 %; the per-thread junction file and its concatenation at the thread end do not appear in the profile (below 0.01 %).

### Faster but changes the output (not pursued; what it changes)
- Lower `--seedMultimapNmax` (10,000), `--winAnchorMultimapNmax` (50), `--seedPerWindowNmax` (50) or `--alignTranscriptsPerWindowNmax` (100): 53.5 % (GSM) and 64.5 % (pbmc) of the reads already have a seed dropped by `seedMultimapNmax`, 51 % / 59 % have a seed that is not an anchor; the costliest reads have 99.9 % non-anchor seeds and 16,640 / 11,555 SA loci on average. Changes the windows, the transcripts, and so the chimeras and the non-chimeric best score printed in each line.
- Skip reads of low-complexity classes before mapping: entropy under 1.5 bits is 9.7 % of the GSM6250953 reads and 19.7 % of their CPU (29.6 % and 34.2 % on pbmc5p_v1). T2-chimpass 3.3 measured the loss on GSM6250953 as 174 kept reads of 4,621,263 (0.004 %) for the class set poly(A/T/G/C), tandem, entropy under 1.5. Changes the lines of those reads.
- Skip reads with the 5' kit TSO reverse complement: 31.0 % of the GSM6250953 reads, 22.6 % of the CPU; T2-chimpass 3.3: 44,972 kept reads lost (0.97 %).
- Cap the number of visited pairs or transcripts per read in the pair loop (reads with more than 1,000 scored pairs carry 94 % / 86 % of the chimeric time): changes the lines of the reads above the cap (not measured).
- Change `--chimScoreDropMax`, `--chimScoreJunctionNonGTAG`, `--chimMultimapNmax`: change which chimeras are found.
- Stop STAR from computing what is not used by the table (non-chimeric alignment output, `Log.final.out` statistics, Solo counts): the junction lines are unchanged but the pass's counters and `#` comment lines change; `trBest->maxScore` is printed in every line and `nW`/`nTr` feed `Log.final.out` only, so only the parts that do not touch `trBest` and the windows could go; not examined further.

## 5. Is the genome index of the pass a private copy by necessity?

Facts from the code:
- The pool (`gpu_pool.cpp:532-542`) reads each genome once with `star_genome_preload_resident` (`GenomePreload.cpp:173`): Genome, SA, SAindex in STAR's layout in **anonymous private memory of the pool process** (`new[]`, transparent hugepages), no System V segment, no memfd. An item's process, forked from the pool, calls `star_genome_preload_select` (`gpu_pool.cpp:604`): it inherits the pages copy-on-write.
- The item's `star_args.txt` has `--genomeLoad NoSharedMemory` (the pass's `args.txt` too). The main STAR child takes the pages (`star_genome_preload_take`, `Genome_genomeLoad.cpp:311`). Right after that fork the item process drops its references: `singlet.cpp:8082`, `if (star_pid != 0) star_genome_preload_release();` (a `delete[]` of the arrays in the item process).
- The pass is forked from that item process (`singlet.cpp`, `run_pass`) when the main STAR is long gone. The item process holds no preload any more (`g_genomePreload` is null), so `star_genome_preload_take` has nothing to take and `Genome::genomeLoad` reads the three files into new private arrays; the pool also dropped the files' page cache after its own read (`SINGLET_GENOME_DROPCACHE`, default on).

Facts from a job (819065, one snapshot 25 s after the pass of each item started; `memprobe.*` in the data directory):
- `ipcs -m`: 0 segments in both items.
- The pool process (`gpu-pool`): Rss 18,613,916 kB, three anonymous mappings of 1,529,168 kB, 12,489,568 kB and 3,140,688 kB (Genome, SA, SAindex), 17,152,000 kB of them AnonHugePages.
- The pass child (the process with `chim_Log.out` open): Rss 25,108,676 kB (pbmc5p_v1) and 26,479,260 kB (GSM6250953), Pss 21,572,458 kB and 22,246,651 kB; three anonymous mappings of exactly the same sizes (1,529,168 / 12,489,568 / 3,140,688 kB), each with Pss equal to Rss, Shared_Clean and Shared_Dirty 0, Private_Dirty 17,159,424 kB in all, AnonHugePages 17,152,000 kB: a private copy, not shared with the pool's. Only the child's `[heap]` (1,181,184 kB mapping, Rss 1,071,804 kB, Shared_Dirty 817,300 kB in the pbmc5p_v1 snapshot) is shared copy-on-write with the item process; the index is not.
- The memfd mappings of 25.4 GB in the second `gpu-pool` process (`/memfd:singlet-nhidx`) belong to the non-host server, not to the STAR index.

So: the pass can only attach to pages it inherits by fork (there is no named segment to attach to). Keeping the pool's copy would need the item process to keep its `GenomePreloadState` (not call `star_genome_preload_release` at the fork of the main STAR, or re-select the resident genome with `star_genome_preload_select` before forking the pass); the pass child would then take the arrays by `star_genome_preload_take` under the same conditions as the main STAR (same index directory, sizes and modification times; the pass uses the same `--genomeDir`, `--genomeLoad NoSharedMemory` and no sequence or junction insertion), sharing the pool's pages copy-on-write (STAR never writes G, SA or SAindex). What it would save (facts to estimate from): 17,159,424 kB of private memory per running pass and the 10 s (GSM6250953, pbmc5p_v1: 10 to 15 s in T2-chimpass) of the start. Not tested; no change made.

## 6. Instrumentation (what was added)

| file | what |
|---|---|
| `src/star/ChimProf.h`, `ChimProf.cpp` (new) | `g_chimProf`, the per-thread record buffers, the stack sampler (modes `bt`, `fp`, `pc`; helper thread, SIGPROF, alternate stack, fault report), meta file |
| `ReadAlign_oneRead.cpp`, `ReadAlign_mapOneRead.cpp`, `ReadAlign_storeAligns.cpp`, `ReadAlign_stitchPieces.cpp`, `ReadAlign_createExtendWindowsWithAlign.cpp`, `ReadAlign_assignAlignToWindow.cpp`, `stitchWindowAligns.cpp`, `ReadAlign_chimericDetection.cpp`, `ChimericDetection_chimericDetectionMult.cpp` | `CP_*` macros: each is `if (g_chimProf)` around a counter or an rdtsc |
| `ReadAlignChunk_processChunks.cpp` | `chimProfThreadBegin()` / `chimProfThreadEnd()` around the thread's work |
| `src/pipeline/singlet.cpp` | `SINGLET_CHIM_KEEP_INPUT=1` keeps `chim_dedup.mate1/2` and writes `chim_args.txt` (default off) |

Switches: `SINGLET_CHIM_PROF=<prefix>` (per-read records), `SINGLET_CHIM_SAMPLE=<prefix>` with `SINGLET_CHIM_SAMPLE_MODE=fp|bt|pc` and `SINGLET_CHIM_SAMPLE_HZ` (default 100), `SINGLET_CHIM_KEEP_INPUT=1`. Use the first two only with `singlet star` (a whole item run would instrument the main pass too).

## Not done, failed, uncertain
- `perf`: not permitted on the compute nodes; no hardware counters (cache misses, branch misses), only wall-clock stacks.
- The first sampler run crashed (segmentation fault, cause not found, section Method 4); the profile tables are from the `fp` mode of build 2 (frame pointers on). Frame-pointer unwinding loses the immediate caller of a libc leaf function when none of the 40 words above the stack pointer is a return address into the executable; those samples are in the `[libc.so.6 unresolved]` rows (2 to 3 %).
- Build 2 was not run through the identity gate (flags `-fno-omit-frame-pointer` and the sampler code, which only runs with the switch on; the gate of build 1 passed). Frame pointers cost a few percent of speed in the STAR objects (not isolated: the node differs between jobs); the arms of one job use the same binary.
- Timing is node dependent: pbmc5p_v1 `base` is 639 user-s on g001 (Xeon Gold 6226) and 749 on g008 (Xeon Gold 6230). Section 1 and 2 are from g008 (job 819071).
- Savings of the candidates are estimates from the profile; none was implemented or run. No per-window data (scores, bounds) were recorded, so a window-pruning scheme cannot be sized.
- d300_5p was not profiled. The pass's own `chim_Log.out` is deleted by the item, so the log line about the genome preload of the pass was not available; the memory maps are the evidence.
- The samples of `bt` and `pc` modes of build 2 exist only for the toy genome.
- The kept inputs stay in `$W/jobs/gpu-runtime/results/t2-chimprof-inputs/` (about 7.9 GB).
