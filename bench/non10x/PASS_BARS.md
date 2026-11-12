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
