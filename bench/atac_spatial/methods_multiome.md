**Data.** 10x Multiome PBMC 10k (granulocytes sorted out), Cell Ranger ARC 2.0.0 published outputs, processed from its
GEX and ATAC FASTQs with 10x GRCh38-2024-A. Multiome PBMC 3k was used for development and is not scored.

**Singlet 1.0.0.** The GEX library goes through the standard pipeline (10x-arc-gex whitelist). The ATAC library goes
through the ATAC path with `--paired-gex <GEX run>`:
- ATAC barcodes are corrected against the ARC ATAC whitelist and translated to the gel bead's GEX barcode. The pair list
  was rebuilt from 10x public ARC per-barcode tables. Leave-one-dataset-out, it covers 100% of the called cells of each of
  7 public ARC runs.
- Cells are called with a two-dimensional mixture over GEX UMIs and ATAC fragments in peaks, the joint call ARC makes.

**Competitor.** The usual open-source route calls each modality separately and intersects the results:
- GEX cells: Singlet's GEX call, EmptyDrops (called_v2);
- ATAC cells: ArchR's default filter (≥ 1,000 fragments, TSS enrichment ≥ 4) on Chromap + SnapATAC2 fragments from the
  same FASTQs.

**Truth.** ARC's `is_cell` and its `gex_barcode` to `atac_barcode` pairs.

**Metrics.**
- Joint cell-call Jaccard with ARC: match-or-beat against the competitor.
- Barcode pairing accuracy: the fraction of ARC cells reported by Singlet whose ATAC barcode equals ARC's. This is a
  threshold rule, ≥ 0.999.

ATAC fragment, peak and QC agreement for this library is scored under `atac`. GEX counts are scored under `gex-counts`.
