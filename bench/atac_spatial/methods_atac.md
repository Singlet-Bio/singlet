**Data.** 10x Genomics public datasets (CC BY 4.0), each processed from its published FASTQs: scATAC PBMC 5k Next GEM
(Cell Ranger ATAC 1.2.0 outputs), and the ATAC library of Multiome PBMC 10k (Cell Ranger ARC 2.0.0). Multiome PBMC 3k
was used for development and is not scored. Reference: 10x GRCh38-2024-A for every tool.

**Singlet 1.0.0.** `singlet encode` (R1, barcode read, R3), then the pipeline with STAR in paired-end DNA mode (no
splicing) and the ATAC post-alignment step. That step detects the barcode window and corrects barcodes against the 10x
whitelist; for Multiome it translates them to GEX barcodes. It then builds Tn5-shifted fragments (MAPQ ≥ 30,
deduplicated) and calls peaks on cut sites with the MACS2 recipe Signac/ArchR use (q < 0.05). It also computes Cell
Ranger / ArchR / Signac QC definitions and calls cells with a mixture model on fragments in peaks (joint with GEX UMIs for
Multiome). 16 threads.

**Competitor.** The open-source route from FASTQs to a Signac/ArchR object, on the same FASTQs, whitelist, reference and
threads:
- Chromap 0.2 (`--preset atac`, whitelist correction, ARC barcode translation for Multiome);
- peaks: MACS3 as Signac's `CallPeaks` runs it;
- cells: SnapATAC2 TSS enrichment plus ArchR's default cell filter (≥ 1,000 fragments, TSS enrichment ≥ 4).

**Truth.** Cell Ranger's published cell calls, per-barcode metrics, fragments and peaks. The Cell Ranger ATAC 1.2 outputs
of the 5k dataset are on hg19; their peaks and fragments are lifted to GRCh38 with UCSC's hg19ToHg38 chain (both ends
must land on the same chromosome and strand). Fragments or peaks that do not lift are dropped; this affects both tools
the same way.

**Metrics.** Each is computed per dataset.
- Cell-call Jaccard with Cell Ranger.
- Spearman correlation of fragments per cell with Cell Ranger, over Cell Ranger's cells, chrM excluded.
- Fraction of Cell Ranger's fragments of its cells on chr21 that a tool reproduces exactly (chromosome, start, end,
  barcode).
- Peak F1: the harmonic mean of the fraction of Cell Ranger peaks overlapped by a tool's peaks and the fraction of the
  tool's peaks overlapping Cell Ranger peaks.
- TSS fragments per cell against Cell Ranger. This metric has no open competitor, so it uses a threshold rule.

Every comparison metric is match-or-beat against the competitor, with no tolerance.
