# D-150 non-10x validation: pass bars (fixed before any scoring, 2026-11-11)

Real data and real ground truth only. Patched engine = gate/d150-encodable-tags (PR 246). Prefixes of the deposited reads are used
(never a full download); depth fractions are reported next to every comparison.

## Libraries (from the accession search of 2026-11-11)
| Platform | Library | Design | Ground truth |
|---|---|---|---|
| Drop-seq | GSM1544798 (SRR1748411, Macosko 2015 HEK/3T3 mixture, ThousandSTAMPs) | human-mouse mixture | authors' HUMAN and MOUSE digital expression files; 1020 STAMPs |
| Drop-seq | GSM1626793 (Macosko 2015, P14 mouse retina rep 1) | ordinary library | authors' digital expression file |
| inDrop | GSM2230757 (Baron 2016, human islet 1) | ordinary human library | authors' umifm counts |
| BD Rhapsody | GSM4764685 (GSE157424, glioblastoma WTA, sample-tag pool, cartridge SC106) | ordinary human WTA | authors' DBEC MolsPerCell file |
| BD Rhapsody | GSM6513561 (sample-tag library) | must be flagged | none (negative control) |

No human-mouse mixture with raw reads exists for inDrop or BD Rhapsody (searched GEO, SRA, ENA, ArrayExpress), so those two platforms
have no purity or doublet bar. The nearest mixture, ChronoSeq K562/EL4 (E-MTAB-15956), is another platform and is not used.
Drop-seq has no ordinary human library with an authors' matrix in this set; the mouse retina library stands in.

## Bars
Auto-detection (no forced protocol): the detected tag must be the platform's own (dropseq, indrop, bd-rhapsody).

Drop-seq barnyard (author cells = the 1020 barcodes with the most UMIs in the authors' files):
1. cells called between 0.7x and 1.3x of 1020 at the prefix depth.
2. barcode recall of author cells among called cells >= 0.70 and precision >= 0.80.
3. species purity: median major-species fraction >= 0.95, and at least 90% of called cells have a major fraction >= 0.90.
4. mixed-cell rate (minor-species fraction > 0.10) within 3 percentage points of the authors' rate.
5. per-cell total UMI Spearman with the authors' counts on shared barcodes >= 0.90.
6. run status success.

Ordinary libraries (Drop-seq retina, inDrop human, BD WTA):
1. cells called between 0.6x and 1.6x of the authors' cell count (BD: 0.8x to 6x of the one-tag matrix, because the deposit pools four tags).
2. median genes per cell between 0.4x and 1.25x of the authors' median (the prefix is a few percent of the deposit depth).
3. run status success (a flagged status is a FAIL for an ordinary library).

Negative control (BD sample-tag GSM6513561): status must not be success.

## Bulk check
The 32 dry-run rejects (dryrun monitor latest.json) minus the six suspected false rejects (GSM3141013, GSM8193087, GSM2581275,
GSM9062044, GSM9062045, GSM6513561) are the 26 real rejects (bulk, ribosome profiling, miRNA, plate, ATAC, no barcode read). A sample
PASSES when it is rejected (gate or encoder abort, no output) or flagged (summary status other than success). It FAILS when the run
ends with status success and 10 or more called cells. Prefix: 4 M spots, auto protocol, GRCh38.

## Bulk-like guard (engine, non-10x protocols only)
A library is flagged (status align_bulk_like) when its barcode rank curve has no knee (no 5x UMI drop within a 5x step in rank), when one barcode
holds more than 5% of all UMIs, or when more than half of all UMIs sit in distinct barcodes. The first version of the knee test used the called
cells and flagged the full-depth Macosko barnyard (the cell caller over-calls there); it was replaced by the caller-independent test above.

## Held-out libraries per platform (registered 2026-11-12, in the commit that precedes any run of them; D-152)
Chosen after the development fixes were designed and before any of these reads were processed. Each is run once, after development ends, with
auto-detected protocol, on the first 30 M read pairs (the same prefix depth as the development libraries, so the depth fractions are stated
next to every number), GRCh38 (human) or GRCm39 (mouse) or the human-mouse barnyard reference (Mereu Drop-seq, which mixes species).
Bars are those of the development libraries of the platform, unchanged; where the truth carries cell barcodes, barcode recall and precision
(>= 0.70 and >= 0.80) are bars as in the Drop-seq barnyard.

| Platform | Library | Design | Truth |
|---|---|---|---|
| Drop-seq | GSM3911759 (GSE133540, Mereu 2020, library Dropseq_N701), run SRR9621741, 106 M pairs, R1 = 76 nt (CB 12 + UMI 8 + poly(T)), R2 = 76 nt cDNA | human-mouse mixture: PBMC 60%, mouse colon 30%, HEK293T 6%, NIH3T3 3%, dog MDCK 1% | authors' per-cell tables GSE133540_Dropseq_human_exp_mat.tsv.gz and ..._mouse_exp_mat.tsv.gz, columns Dropseq_N701_<barcode>: 1,000 cells of this library (862 human, 73 mouse, 49 doublet, 16 dog by the authors' labels) |
| Seq-Well | GSM4693356 (GSE150672, Hughes 2020, "S3Comp1217" PBMC), run SRR12314741, 443 M pairs, R1 = 20 nt, R2 = 50 nt | healthy human PBMC | authors' GSM4693356_SeqWell.V2.Counts1.csv.gz, 1,968 cells with 12-mer barcodes as column names |
| inDrop | GSM3911812 (GSE133548, Mereu 2020, 1CellBio inDrop), run SRR9621794, 143 M pairs, R1 = 60 nt cDNA, R2 = 50 nt barcode read (BC1 8, W1 22, BC2 8, UMI 6, poly(T)) | human-mouse reference sample of Mereu 2020 | authors' GSE133548_inDrop_human_exp_mat.tsv.gz (and _mouse_), 2,988 cells; the deposit has no matrix per library beyond these, barcode orientation unverified, so only the ordinary-library bars apply (cells, genes per cell, status) |
| BD Rhapsody | GSM7177494 (GSE229765, PyMT mammary tumor, fresh, BD Rhapsody WTA), WTA run SRR24175927 (570 M pairs; the sibling run SRR24175928 is the sample-tag library and is not used), R1 = R2 = 76 nt | mouse tumor, WTA | authors' GSM7177494_PyMT_Fresh_BD_3772cells_matrix.mtx.gz with genes and barcodes; barcodes are BD Cell_Index integers, so only the ordinary-library bars apply |

Bulk libraries for the guard (registered 2026-11-12): human RNA-seq read pairs, 4 M spots each, forced through a non-10x tag (dropseq for all;
indrop for the first, bd-rhapsody for the second as well): SRR1039508 (airway, 2x63), ERR188044 (Geuvadis LCL, 2x76), SRR896663 (SEQC
sample A, 2x100), SRR12188073 (2x150). Pass rule as in "Bulk check": the run ends rejected, flagged, or with fewer than 10 called cells.
