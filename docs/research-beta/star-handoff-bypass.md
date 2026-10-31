# STAR bypass for hand-off reads: scope (S8)

Status: measured and scoped, not built. Everything below is from the code at `prod/campaign` plus the runs named in each line.

## Why

`--engine gpu` with the GPU pileup on: the GPU prefill maps every distinct sequence, the pileup gets the alignments from the hand-off, and
STAR still walks every read to produce its own bookkeeping. Measured on SRR19573735 (11.33 M reads, 18 STAR threads, quiet node, jobs 794539 /
794577 / 794604 / 794613), core-s of the STAR child process:

| step | core-s | core-s/M |
|---|---|---|
| genome load (fixed per sample) | 7.9 | 0.70 |
| prefill: read + clip, from the table or the FASTQ | 6.4-7.9 | 0.56-0.70 |
| prefill: mapDistinct host side | 20.4 | 1.80 |
| prefill: hand-off records (host) | 14.5 | 1.28 |
| mapping pass (per read) | 46.3-47.1 | 4.1 |
| solo post-map: records read back 6.2, UMI collapse 2.1, matrix 0.4, filtering 0.3, stats 0.2 | 9.3 | 0.82 |

`STAR_PROFILE` (SRR19421372, 13.2 M reads): per-read thread-s chunkRead 4.1, readLoad 6.8, map 1.9, output 14.5 (outMapped 1.2, outBarcode 3.7,
outSolo 3.1, the rest = hand-off entry, summary replay, stats adds 6.3). Only 7% of the pass is mapping.

## What STAR does for one hand-off read (`ReadAlign::outputHandoffSummary`, `handoffReadEntry`)

1. `statsRA`: mapped unique / multi counts, mismatch / insertion / deletion counts and lengths, splice counts, mapped bases and portion
   (these end in `Log.final.out`).
2. `chunkOutSJ`: the read's junction records (these end in `SJ.out.tab`).
3. `readAnnot`: the Gene feature set and overlap type from the sequence's summary.
4. `getCBandUMI`: CB / UMI of the read, the 1MM_multi_Nbase_pseudocounts correction and its `SoloReadBarcodeStats` counters (`Barcodes.stats`).
5. `handoffReadEntry`: the per-read entry for the GPU pileup (corrected barcode, UMI, quality counters, flags, per-base mask).
6. `SoloRead::record`: the read's flag bits (genomeU/M, mito, exonic / intronic / AS, featureU/M, cbMatch) and, for a read with a CB match, a
   (CB, gene, UMI) record in the thread's stream. After mapping these are read back (`inputRecords`) to fill `CellReads.stats` and to feed the
   UMI collapse and matrices.

## Which STAR outputs something consumes (default stream-path run)

Consumed (must stay exact): `Log.final.out` (singlet.cpp:8294, export.h:1765, validate_output.h, catalog `star_log`; compared since #130),
`SJ.out.tab` (export.h:1487), `CellReads.stats` columns cbMatch, genomeU, genomeM, exonic, intronic, exonicAS, intronicAS, mito
(star_cell_read_stats.h). Not consumed by a default run or by the .s1 builder (python/singlet/bundle.py drops `star_Solo.out`): the Gene
matrices, `Features.stats`, `Barcodes.stats`, `Summary.csv`, `UMIperCellSorted.txt`, `nUMIunique` / `nGenesUnique`. The only reader of
`Gene/filtered/barcodes.tsv` is the cell-caller fallback when called_v2 is not used (export.h:598-652).

## Counter by counter: can it be computed by group-by, without a per-read walk?

For a hand-off read, everything above is a function of three things: the read's CB status (known from the `.1fg2` planes `PL_CB`, `PL_CBST`,
`PL_RAWCB`), its sequence's summary (the GPU record: nTr, mito, overlap type, gene set, mismatches, junctions), and the number of reads with
that (CB, sequence). So:

* `Log.final.out` counters 1: sum over sequences of `summary x multiplicity`; multiplicity comes from the histogram of `PL_SEQ`. Exact.
* `SJ.out.tab` 2: the per-read junction records collapse by (start, gap) with summed counts and max overhang: `record x multiplicity`. Needs a check
  of `sjdbFilter` / collapse arithmetic (`Junction::collapseOneSJ`), then exact.
* `CellReads.stats` flag counts 6: per (CB, sequence) pair, `multiplicity` added to the sequence's flag set, with `cbMatch` from the CB status. Exact.
* `Barcodes.stats` 4: per-CB-status counts. Not consumed, so it may be dropped, but the correction itself (R11, `CbCorrector`) already runs in singlet.
* Hand-off entries 5: planes + GPU records. Per-base quality mask: `PL_Q` (binned qualities) gives the same bins the CPU uses (ILLUMINA4), so
  the mask is a function of the plane; needs the exact `minbaseq` rule on binned values (the pileup already depends on this identity).
* Per-read order: the entries are indexed by read number (iReadAll); the plane order is the read order. `Unmapped.out.mate*` carry the STAR read
  index in the name line, so hand-off reads must keep their numbers if they stay out of STAR's input.

## Open questions (a counter that cannot be reproduced exactly blocks that read class)

1. Does anything downstream use `iReadAll` of hand-off reads other than the entries (STAR read numbering in the name lines of the unmapped files,
   which are not hand-off reads)? If the chunk source skips hand-off reads, numbering of the others must not change.
2. `readN`, `readBases`, "Average input read length", chunk-level counters in `Log.final.out`: derived from all reads, including skipped ones.
3. Multi-mapping hand-off reads (`nTr > 1`): STAR picks `trMult[0]`'s chromosome for mito and the stand-in transcripts; confirm the summary
   carries exactly what `record()` needs (it does today, since `outputHandoffSummary` builds the stand-ins from it).
4. Reads whose CB status is not exact (corrected, no match, two N): they are in `PL_RAWCB` and take the same branches as in the live run.

## Plan, with the gate for each step

1. Bulk statistics from the planes and the summaries, written to the same places STAR writes them, behind an env switch, run next to the live
   per-read pass and compared: `Log.final.out`, `SJ.out.tab`, `CellReads.stats` consumed columns, hand-off entries, masks (shadow mode, no change
   to outputs).
2. When the shadow comparison is exact on pbmc1k_v3, neuron1k_v3, nuclei2k_v2 and the 3 `.1fg2` fixtures: the chunk source skips hand-off reads and
   STAR adds the bulk results. Gate: CPU = GPU byte-identical, ctest, `Log.final.out` compared (#130), `CellReads.stats` consumed columns, a decline /
   fallback test (any read class that cannot be reproduced runs through STAR as today).
3. Then: stop building per-read Solo records when nothing consumes them, and the stats-only post-map (measured upper bound 0.26 core-s/M, below the 0.3 bar, so only if it comes for free).
