# E9: STARsolo's Gene outputs on the GPU

Experiment E9 ("solo") of wave 2b of the GPU aligner program. Branch `gpu/fable/solo` (from 7ed4e9f0). A standalone library
(`src/engine_gpu/solo.cuh`, `solo_gpu.h`, `solo_gpu.cu`) and a driver (`bench/fable/solo/solo_test.cu`); nothing is wired into the engine.
The only change to `src/star` is a debug dump behind `SINGLET_SOLO_DUMP` (off by default, section 3).

## 0. Answer in brief

- **All 10 gated files are byte-identical to STAR's on pbmc1k_v3** (66,601,887 reads), from the GPU path and from the host (OpenMP)
  path: `Barcodes.stats`, `Gene/Features.stats`, `Gene/raw/{barcodes,features,matrix}`, `Gene/filtered/{barcodes,features,matrix}`,
  `Gene/UMIperCellSorted.txt` and `Gene/CellReads.stats` (run 796095, commit ff2b5dc0; first reached at 2a884cc1, runs 796088/796089).
  CellReads.stats is identical when its rows are written in the order STAR's `unordered_map` gave them in that run; STAR's own row order
  changes from run to run (section 6); without an order the library writes the same rows sorted by whitelist index.
- **Per read, 0 differences**: gene annotation (nTr, gene set size, gene, mito, exonic) for the 66,444,956 reads with a device record;
  cbMatch after the UMI check, first whitelist match and the record-time CellReads flag for all 66,601,887 reads.
- **Time**: the whole Gene path on one H100 NVL takes 74 ms of device compute (gene annotation 34, CB 15, per read 16, sort 8, 1MM_All
  1.3, compaction 0.5) + 35-63 ms to bring the compact results to the host + 0.75 s for filtering and writing the 10 files (0.5-0.58 s
  with CellReads rows sorted). STAR's Solo post-map step alone takes 15-16 s at `--runThreadN 8` on the same input, on top of the per-read
  Solo work inside its mapping threads (not timed separately).
- **Not ported**: `Gene/Summary.csv` (needs the read-quality histograms and the mapping statistics; nothing in the pipeline reads it),
  other features (GeneFull, Velocyto, SJ), other UMI/cell-filter/multimapper options. Inputs not yet on the device: CR/UR came from STAR's
  per-read dump in this test (E8's hand-off pass has them on the device), and the 156,931 reads whose sequence has an N (mapped by STAR on
  the host) take their gene annotation from STAR.

## 1. What STARsolo computes in this pipeline

The pipeline runs STAR 2.7.11b (vendored, `src/star`) with `--soloType CB_UMI_Simple --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17
--soloUMIlen 12 --soloBarcodeReadLength 0 --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloCellReadStats Standard` and the defaults
for everything else (`star_Log.out` of the truth run): feature `Gene`, `--soloStrand Forward`, `--soloUMIdedup 1MM_All`,
`--soloUMIfiltering -`, `--soloMultiMappers Unique`, `--soloCellFilter CellRanger2.2 3000 0.99 10` (no random numbers: EmptyDrops is not
used), `--outFilterMultimapNmax 10`, `--outSAMmultNmax -1`, no `--sjdbGTFfile` at mapping (the transcript model is the index's
`transcriptInfo.tab` / `exonInfo.tab` / `geneInfo.tab`). `MATCH_CellRanger` is not defined (`src/star/CMakeLists.txt`), so the CB
posterior is computed in `float`. The pipeline's R1 qualities are constant `F` (STAR sees `F` for every CB base), so the posterior of a CR
is a function of its candidates' exact counts only.

## 2. The port

Every rule is a `__host__ __device__` function in `solo.cuh` that names the STAR code it transcribes; `solo_gpu.cu` runs each one per item
on the device (grid-stride kernels, CUB sorts) or on the host (OpenMP over the same functions, `__gnu_parallel::sort`). All counters are
integer atomics, so every output is independent of thread order.

| stage | per | rule (STAR 2.7.11b, `src/star`) |
|---|---|---|
| (b) gene annotation | record (distinct sequence) | `e6::recordCount` (E6's mappedFilter port) for unmapped; then over all `nTr` transcripts of the blob: mito set (`Genome.cpp:201`, an absent name inserts `nChr`), `Transcriptome::classifyAlign` (`Transcriptome_classifyAlign.cpp:177-267`: `binarySearch1a` over `trS`, the backward walk while `trEmax >= aGend`, the strand test, `alignToTranscript` `:8-90` with `binarySearch_leLeft`, `serviceFuns.cpp:212/239`), gene set size and its single gene, `ovType` exonic. = `ReadAlign::seqCacheHandoffSummary` (`ReadAlign_seqCache.cpp:325`) without the mapping stats and junctions |
| (b) | read | the read's record annotation, or (read without a record) the host's |
| (a) CB | distinct CR | `matchCBtoWL` (`SoloReadBarcode_getCBandUMI.cpp:9-80`): `convertNuclStrToInt64` (`SequenceFuns.cpp:262`), exact, one-N (4 bases at the N, in order), 1MM (last base first, `jj` 1..3), `mm1_multi` keeps every match. The whitelist is the sorted, deduplicated array STAR builds (`ParametersSolo.cpp:311-336`); a binary search returns the index `hashLookupCB` returns |
| (a) | read | `convertCheckUMI` (`:127`) replaces cbMatch with -23 / -24 (`:291`); `addStats` (`:87`): Barcodes.stats and the exact-match histogram |
| (a) | distinct CR with >1 match | the posterior (`SoloReadFeature_inputRecords.cpp:85-121`): exact count + 1 (`Solo.cpp:66-70`), `pin = (float)(count * pow(10, -q/10))` with the host's `pow`, float `ptot`, strict `>` for the maximum, `pmax >= 0.975f * ptot` |
| (c) record + inputRecords | read | `SoloReadFeature::record` (`SoloReadFeature_record.cpp:20-204`): the flag (genomeU/M, mito, exonic, featureU/M, cbMatch), CBnotInPasslist counts, Features.stats noUnmapped / noNoFeature / MultiFeature / subMultiFeatureMultiGenomic, `cbReadCount` of every match (`:189`); `inputRecords` (`SoloReadFeature_inputRecords.cpp:44-176`): the final CB, yessubWLmatchExact / noTooManyWLmatches, countedU, cbPerfect / cbMMunique / cbMMmultiple, the CB's CellReads row or CBnotInPasslist, the `(CB, gene, UMI)` key of a counted read |
| (c) collapse | distinct (CB, gene, UMI) | radix sort of 63-bit keys (CB 22, gene 17, UMI 24 bits), unique; a second order with the UMI halves swapped (`umiSwapHalves`, `ParametersSolo.cpp:500`); in each order the pairs that share the upper half and differ in one 2-bit symbol (`collapseUMIwith1MMlowHalf`, `SoloFeature_collapseUMI_Graph.cpp:80-127`) are unioned; the roots per (CB, gene) are the 1MM_All count (`umiArrayCorrect_Graph` `:16-77`: uncoloured UMIs + connected components of the colour graph = the connected components of the Hamming-1 graph) |
| (d) stats | CB | `collapseUMIall` (`SoloFeature_collapseUMIall.cpp:17-26`): yesUMIs, yesCellBarcodes, yesWLmatch, yessubWLmatch_UniqueFeature; nCB = whitelist entries with `cbReadCount > 0` (`SoloFeature_sumThreads.cpp:82-98`) |
| (e) filter | host | CellRanger2.2 (`SoloFeature_cellFiltering.cpp:37-66`): `maxind = int(round(3000 * (1 - 0.99)))`, `nUMImax = sorted[min(nCB-1, maxind)]`, `nUMImin = max(int(round(nUMImax / 10.0)), 1)` |
| (f) writers | host | `statsOut` (`setw(50)` / `setw(15)`), `outputResults` (`SoloFeature_outputResults.cpp`), `statsOutput` (`SoloFeature_statsOutput.cpp:73-116`) |

CellReads.stats rows are written in the iteration order of STAR's `unordered_map` (`SoloFeature_statsOutput.cpp:102`). The writer takes
the keys' insertion order: with it, it builds the same `std::unordered_map` (`reserve(nCB*3/2)`, `SoloFeature_countCBgeneUMI.cpp:43`) and
iterates it; without it, rows are sorted by whitelist index.

## 3. Inputs and the hook

The library takes what the engine already has after `mapDistinct` and E8's hand-off pass:

- the distinct sequences' records: the blobs in E6's format (`blob`, `blobOff`, `seqOff`, `RecParams`), one per distinct sequence the
  device mapped (31,636,557 on pbmc1k_v3);
- per read: CR and UR (3 bits per base, `HandoffReadEntry` packing) and the record index (`mapDistinct`'s `readRecord`); a read without a
  record (its sequence has an N and STAR maps it on the host: 105,089 distinct sequences, 156,931 reads on pbmc1k_v3) carries the host's
  annotation (`RecAnn`: nTr, gene set size, the single gene, mito, exonic);
- the transcript model of the index (`transcriptInfo.tab`, `exonInfo.tab`, `geneInfo.tab`, `chrName.txt`) and the whitelist file.

The test reads them from the truth run: the records and `readRecord` from E6's dump (`SINGLET_ENGINE_GPU_DUMP`, 6.05 GB), CR, UR and the
host-mapped reads' annotation from a dump of what STAR's Solo saw per read. That dump is the only change to `src/star`, a debug hook that is
off by default:

| file | change |
|---|---|
| `src/star/ReadAlign.h` | `soloDumpChunk`, the thread's buffer, `soloDumpRead()`; `soloDumpFlushAll()` |
| `src/star/ReadAlign.cpp` | `soloDumpChunk = iChunk` (names the thread's file) |
| `src/star/ReadAlign_outputAlignments.cpp` | with `SINGLET_SOLO_DUMP=<dir>`: after each of the two `soloRead->record` calls, one 48-byte record per read (iRead, CR, UR, the gene, the record flag, cbMatch, the first whitelist match, nTr, gene set size, mito/exonic bits); per thread `<dir>/solo_<chunk>.bin` |
| `src/star/STAR.cpp` | `soloDumpFlushAll()` before Solo (the STAR child leaves with `_exit`) |

77 lines in all. With the variable unset nothing changes; with it set the Solo outputs are the same as without it (the dump and plain runs
of truth job 796074 give byte-identical Barcodes.stats, Features.stats, raw and filtered files and UMIperCellSorted.txt; CellReads.stats has
the same rows in another order, section 6).

The truth job (796074, commit b1fa696b, T=8, one H100 NVL) runs the singlet build of STAR twice on pbmc1k_v3 (66,601,887 reads): `dump`
(both dumps on) and `plain`. The driver (`bench/fable/solo/solo_test.cu`) loads the dumps and the index tables, runs `soloRun` on the host
and/or the device `REPS` times, checks every read's annotation, barcode and record flag against STAR's dump, writes `star_Solo.out` with
`soloWrite` (once with the CellReads.stats rows in STAR's order, once sorted) and compares each file with STAR's byte for byte (the first
differing line when they differ).

STAR's CellReads.stats row order is replayed this way: STAR fills one `unordered_map` from the threads' Solo files, file after file
(`SoloFeature_countCBgeneUMI.cpp`), so the keys arrive as, for each thread in turn, the first occurrence of each row CB in that thread's
reads. The driver rebuilds that sequence from the dump's per-thread files and `rowCb` (per read, the CB whose row counts it), and
`soloWrite` inserts it into a `std::unordered_map` with the same `reserve` and iterates it.

## 4. Identity

Run 796095 (commit ff2b5dc0, node g052, H100 NVL, 18 CPU threads; truth 796074 `dump` run). Both paths, every file:

| file | bytes | lines | GPU | host |
|---|---:|---:|---|---|
| `Barcodes.stats` | 792 | 12 | identical | identical |
| `Gene/Features.stats` | 726 | 11 | identical | identical |
| `Gene/raw/barcodes.tsv` | 62,668,800 | 3,686,400 | identical | identical |
| `Gene/raw/features.tsv` | 1,627,536 | 38,606 | identical | identical |
| `Gene/raw/matrix.mtx` | 56,944,066 | 3,686,802 | identical | identical |
| `Gene/filtered/barcodes.tsv` | 19,312 | 1,136 | identical | identical |
| `Gene/filtered/features.tsv` | 1,627,536 | 38,606 | identical | identical |
| `Gene/filtered/matrix.mtx` | 31,515,344 | 2,675,097 | identical | identical |
| `Gene/UMIperCellSorted.txt` | 503,185 | 233,950 | identical | identical |
| `Gene/CellReads.stats` | 18,353,470 | 323,543 | identical (STAR's row order replayed) | identical (same) |
| `Gene/Summary.csv` | 647 | | not ported | not ported |

The numbers the files carry: 903,105 distinct CRs, 707,240 whitelist matches, 36,678,568 counted reads, 11,500,780 distinct
(CB, gene, UMI), 3,686,799 (CB, gene) entries after 1MM_All, nCB 333,210, CellRanger2.2 nUMImax 19,191, nUMImin 1,919, 1,136 cells.
Exon overflow (a record with more than 32 exons, which the device rule would flag): 0.

Per read against STAR's dump (both paths, every rep): nTr, gene set size, gene, mito, exonic 0 differences of 66,444,956; cbMatch, first
whitelist match, record flag 0 differences of 66,601,887.

The truth run's own consistency (truth 796074): the `dump` and `plain` runs and the E8 gate's files have the same hashes for every file
but CellReads.stats; the two runs' CellReads.stats have the same 323,541 rows (sorted-row hash a4a92b97b5d141cb) in different orders.

## 5. Timing

Run 796095, ms. The GPU numbers are reps 1-2 (rep 0 includes first-touch costs: sort 12.0 ms); the host numbers are reps 0-2 with 18
OpenMP threads (a test of the same code; not a production path).

| stage | GPU (H100 NVL) | host, 18 threads | STAR 2.7.11b (singlet build), T=8 |
|---|---:|---:|---|
| gene annotation (per record, then per read) | 34.1-34.2 | 1,213-1,237 | in the mapping threads (hand-off summary / `classifyAlign`) |
| CB (distinct CRs, whitelist, UMI check, Barcodes.stats, posterior) | 15.0 | 3,957-3,986 | in the mapping threads (`getCBandUMI`) |
| per read (record + inputRecords rules, keys) | 15.7-16.7 | 2,393-2,396 | `record` in the mapping threads + reading the Solo files 5-6 s |
| sort + unique + swapped-halves order | 7.7 | 717-724 | counting + 1MM_All collapse 9-10 s |
| 1MM_All (union-find) + counts per CB | 1.3 | 68-91 | (in the above) |
| compaction (CBs with a count or a row) | 0.5 | 24 | |
| download (compact results) | 53.4-63.2 (rep 0: 34.6) | 73 | |
| **compute, total** | **129-138** (74 without the download) | **8,472-8,506** | **15-16 s post-map** (01:48:24 to 01:48:39 in the dump run) |
| filtering + 10 files, STAR's CellReads order | 774 | 749 | raw matrix ~1 s, filtering + filtered < 1 s (in the 15 s) |
| filtering + 10 files, CellReads rows sorted | 577 | 501 | |
| upload (test only: the inputs come from the host dumps) | 998-1,314 (all reps) | 0 | |

STAR's post-map Solo here is already singlet's parallel `inputRecords` (`SoloFeature_countCBgeneUMI.cpp:45-104`, on under
`SINGLET_ENGINE_GPU`). Its per-read Solo work inside the mapping threads (barcode extraction and whitelist match, the gene part of the
hand-off summary for 31.6 M records, which is in the 6 s "summaries" line of the E8 log, and writing the per-thread Solo temp files) is not timed separately in the release build (`STAR_PROFILE` compiled out).

Two optimisations after identity, both kept identical: the per-whitelist arrays (3,686,400 x 19 counters) were downloaded whole
(412-429 ms); a compaction on the device to the ~333 k CBs with a count or a row cut that to 35-63 ms. The writers went from 3.0 s to 0.75 s by
formatting integers and barcodes directly into one buffer per file (no `snprintf`, no per-barcode strings).

## 6. Surprises

1. **STAR's CellReads.stats is not reproducible run to run.** Its rows come from an `unordered_map` filled from the threads' Solo files
   in thread order (`SoloFeature_countCBgeneUMI.cpp:97-103`, the serial loop `:108-111` inserts in the same order); which thread mapped
   which chunk changes between runs, so the insertion order and the iteration order change. The two truth runs (same binary, same reads)
   differ in byte order and agree as sorted rows. The library reproduces a run's bytes exactly given that run's insertion order (the driver
   replays it from the dump's per-thread files), and otherwise writes rows sorted by whitelist index. The pipeline reads the file by
   barcode (`include/singlet/pileup/export.h:1008`, `star_cell_read_stats.h`), so the sorted order is safe and deterministic.
2. **Whitelist parsing quirk**: `ParametersSolo.cpp:320` keeps a whitelist line when `convertNuclStrToInt64` returns non-zero, which drops
   only barcodes with an N at the first base (`posN == 0`) and keeps the others with N read as A. Ported as is; the 10x whitelist has no N.
3. **The mitochondrial set has an extra member**: `Genome.cpp:201` inserts `nChr` for each name of `chrM M MT` that the genome lacks, so
   the set is {22, 194}; harmless (no alignment is on chromosome nChr), ported as is.
4. **Everything matched on the first full run** (2a884cc1): the float posterior with the host's `pow`, the 1MM_All colour graph as
   Hamming-1 components, the read-flag rules and the CellRanger2.2 rounding needed no second iteration. The only debugging was in the
   test harness.
5. **Reads with an N in the sequence** (105,089 distinct sequences, 156,931 reads, 0.24%) have no device record (STAR maps them on the
   host lane), so their gene annotation is an input the library does not compute.
6. **Downstream use**: in this repository only `Gene/filtered/barcodes.tsv` (cell calling, `export.h:607`) and `Gene/CellReads.stats`
   (per-cell QC, `export.h:1008`) are read; the bundle skips `star_Solo.out` (`python/singlet/bundle.py:1115`), and the legacy `.s1`
   recipes read the raw/filtered matrices of older samples. `Summary.csv` is not read anywhere.

## 7. Wiring

Recommended: run `soloRun(device)` in the engine right after E8 has built the records on the device, and have STAR skip its Solo work.

- **Inputs on the device**: the records are already there (E8, no upload); the per-read record index is `mapDistinct`'s; CR and UR per
  read are in E8's hand-off pass (it builds the distinct-CR table on the device: 897,422 distinct CRs in 17 ms in the truth log); upload
  once per index the transcript model (~20 MB) and the whitelist (29.5 MB). The 1-1.3 s upload in this test disappears.
- **Reads without a device record** (sequences with N): STAR keeps `classifyAlign` for those reads only and hands their `RecAnn`
  (8 bytes per read, 157k reads) to the engine before `soloRun`; or the engine maps them. The library already takes `hostAnn`.
- **What remains of STAR's per-read loop for Solo**: for reads with a device record, nothing: `readBar->getCBandUMI` (CB/UMI extraction,
  whitelist match, Barcodes.stats), `soloRead->record` (gene annotation, flags, the Solo temp files) and the post-map `Solo` step
  (`inputRecords`, collapse, filtering, outputs) can all be skipped behind one switch. For host-lane reads, `classifyAlign` stays.
- **Outputs**: `soloWrite` writes STAR's files byte for byte; write CellReads.stats rows sorted by whitelist index (STAR's own order is
  run-dependent). If the E8 gate keeps hashing CellReads.stats, it should hash sorted rows (it already cannot pin STAR's order).
- **Still to port before switching STAR's Solo off**: `Summary.csv` (Q30 of CB+UMI and RNA bases from the quality histograms, the mapping
  numbers from `g_statsAll`, and cell numbers that `soloWrite` already has: nCells, reads and UMIs in cells, median genes and UMIs per
  cell, saturation from counted reads vs UMIs). Nothing in the pipeline reads it, but STAR writes it and the E8 gate hashes it.
- **Scope guard**: the port covers `CB_UMI_Simple`, Gene, `--soloStrand Forward`, `1MM_All`, Unique, `1MM_multi_Nbase_pseudocounts`,
  CellRanger2.2, `--soloCellReadStats Standard`; key limits whitelist <= 2^22 entries, genes <= 2^17, UMI <= 12 bases, CB/UMI <= 21
  bases in the 3-bit read packing. Any other Solo configuration should keep STAR's path.
- **Device memory**: not measured here; the largest buffers are per read (66.6 M x ~30 bytes) and the keys (36.7 M x 8 bytes, double
  buffered), roughly 3 GB by estimate, released before the next stage where possible.

## 8. Jobs, commits, files

| job | what | result |
|---|---|---|
| 796064 | build of the singlet binary with the hook | cancelled (resubmitted) |
| 796066 | build of the singlet binary with the hook (b1fa696b) | completed |
| 796074 | truth: pbmc1k_v3, T=8, `dump` and `plain` runs | completed; files and hashes in `bench/fable/solo/results/truth-796074-summary.txt` |
| 796083 | driver compile (debug-cpu, QOS-blocked) | cancelled |
| 796084 | driver compile (2a884cc1) | completed |
| 796087 | run on debug-gpu (QOS-blocked) | cancelled |
| 796088 | run, host only (2a884cc1) | 10 of 10 identical |
| 796089 | run, host + GPU, 3 reps (2a884cc1) | 10 of 10 identical on both |
| 796092 | driver compile (fffd9903) | failed (driver printf typo) |
| 796094 | driver compile (ff2b5dc0) | completed |
| 796095 | run, host + GPU, 3 reps (ff2b5dc0) | 10 of 10 identical on both; log `bench/fable/solo/results/run-796095.txt` |

Commits on `gpu/fable/solo` (from 7ed4e9f0): b1fa696b (hook, build and truth jobs), 2a884cc1 (library and driver), aedc35c0 (doc draft),
fffd9903 (device compaction, faster writers), ff2b5dc0 (driver fix), and the commit that adds this document's results.

Files: `src/engine_gpu/solo.cuh` (rules), `src/engine_gpu/solo_gpu.h`, `src/engine_gpu/solo_gpu.cu` (stages, writers),
`bench/fable/solo/solo_test.cu` (driver), `bench/fable/solo/jobs/{build_singlet,truth,compile,run}.sbatch`,
`bench/fable/solo/results/` (logs; no read-level data). The read-level dumps
(`$W/jobs/fable/solo/scratch/truth-796074`, 8.7 GB) were deleted after run 796095.

Reproduce: `sbatch bench/fable/solo/jobs/build_singlet.sbatch`; `sbatch bench/fable/solo/jobs/truth.sbatch` (needs the dumps, so it
must be re-run once the scratch is deleted); `sbatch --partition=short-gpu bench/fable/solo/jobs/compile.sbatch`;
`TRUTH=<truth job> MODE=both REPS=3 sbatch --partition=short-gpu --cpus-per-task=18 --mem=150G bench/fable/solo/jobs/run.sbatch`.
