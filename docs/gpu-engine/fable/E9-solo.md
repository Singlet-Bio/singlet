# E9: STARsolo's Gene outputs on the GPU

Experiment E9 ("solo") of wave 2b of the GPU aligner program. Branch `gpu/fable/solo` (from 7ed4e9f0). A standalone library
(`src/engine_gpu/solo.cuh`, `solo_gpu.h`, `solo_gpu.cu`) and a driver (`bench/fable/solo/solo_test.cu`); nothing is wired into the engine.
The only change to `src/star` is a debug dump behind `SINGLET_SOLO_DUMP` (off by default, section 3).

## 0. Answer in brief

RESULTS_BRIEF

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

RESULTS_INPUTS

## 4. Identity

RESULTS_IDENTITY

## 5. Timing

RESULTS_TIMING

## 6. Surprises

RESULTS_SURPRISES

## 7. Wiring

RESULTS_WIRING

## 8. Jobs, commits, files

RESULTS_JOBS
