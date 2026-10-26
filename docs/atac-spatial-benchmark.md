# ATAC, Multiome and Visium in Singlet 1.0.0

Singlet's pipeline already had ATAC and Visium code paths, but neither had been checked against the tools people use. This
page records what was wrong, what changed, the QC rules, and how the result compares with Cell Ranger ARC, Cell Ranger
ATAC and Space Ranger (published outputs), and with open-source pipelines run on the same FASTQs.

Code: `include/singlet/pileup/atac_*.h`, `visium_tissue.h`, `src/pipeline/singlet.cpp` (ATAC branch, Visium block,
`singlet atac` subcommand). Harness: `bench/atac_spatial/`. Storage proposal: `docs/format-v2.md` section 14. Public results:
`bench/public/results/{atac,multiome,spatial}.json` (contract `bench/public/CONTRACT.md`).

RESULTS_PLACEHOLDER

## 1. What was wrong

An audit of the ATAC and Visium paths (2026-10-25) found these problems.

**ATAC**

* **Fragment coordinates.** A fragment was built from read 1's position plus |TLEN|. When read 1 was the reverse-strand
  mate (about half of all pairs), the fragment was shifted right by its own length. This affected every fragment, peak,
  TSS and FRiP number downstream. The unit tests only used forward read-1 records.
* **Barcodes.** The raw i5 read was used as the barcode, with no whitelist correction and no orientation detection.
  Every barcode with a sequencing error became its own "cell". Multiome's 24 bp barcode read (8 bp spacer + 16 bp barcode)
  was taken whole, so no barcode could match. The ATAC side of the ARC whitelist and the ATAC→GEX pairing were missing, so
  Multiome ATAC could not be joined to its GEX cells.
* **Barcode discovery** before alignment kept raw 16-mers with at least 100 reads. 24 bp barcodes were never counted,
  and the run failed with "no barcodes discovered".
* **TSS "enrichment"** was the fraction of fragments near a gene start, not an enrichment score. Without `--exons` it was
  0 for every barcode, so the TSS filter rejected every cell.
* **FRiP** was computed against 500 bp bins with at least 3 reads, before any peaks existed.
* **Peaks** were a 500 bp-bin Poisson test with no q-value, run on all barcodes, with bin-aligned ends.
* **Cell calling** was an Otsu split on total fragments plus the broken TSS filter.
* There was no nucleosome signal, no gene activity, and the read-pair count (column 5 of `fragments.tsv.gz`) was always 1.
* In the atlas, the library-type screen (`docs/library-type-screen.md`) found 139 ATAC libraries processed as RNA.

**Visium**

* There was no spot list, so the pipeline could not tell Visium from 3' v3 (same read geometry) and could not correct
  spot barcodes. Spot barcodes were "discovered" like droplet barcodes.
* Spots went through droplet cell calling (EmptyDrops). On GSM7090362, 58 of about 4,000 spots were "cells".
* Spatial coordinates came only from a Space Ranger `tissue_positions.csv` passed by hand. The atlas runs never had one.

## 2. What changed

### ATAC (`atac_pipeline.h`, run after STAR in PE-DNA mode, or `singlet atac --bam ...`)

1. **Barcodes** (`atac_barcodes.h`). 200,000 fragment reads choose the barcode window: offset and strand, scored against
   each candidate whitelist by exact hit rate. The candidates are `737K-cratac-v1` and the ARC ATAC list with GEX pairs.
   A count pass then builds the prior. Each read's barcode is resolved by exact match, or else by the most supported
   Hamming-1 neighbour, accepted only if its posterior is at least 0.975 (Cell Ranger's rule, without base qualities).
   With ARC pairs, output barcodes are GEX barcodes. Pre-alignment discovery is skipped for ATAC.
2. **Fragments** (`atac_fragment.h`). Each fragment comes from the leftmost mate (TLEN > 0): [pos, pos+TLEN), shifted
   +4/−5, MAPQ ≥ 30, proper pairs only. Duplicates (same start, end and barcode) are collapsed, and their read pairs are
   counted in column 5. STAR now keeps mates that protrude past each other (`--alignEndsProtrude 100 ConcordantPair`):
   these are fragments shorter than the read.
3. **Peaks** (`atac_peak_caller.h`). This is the MACS2 recipe that Signac and ArchR run for single-cell ATAC, on Tn5 cut
   sites:
   * a 200 bp tag around each cut site;
   * local lambda = max(genome-wide, 10 kb);
   * Poisson p-values, Benjamini–Hochberg q < 0.05 (the MACS2 default, which also gave the best peak F1 on the development data), summits;
   * computed on a 10 bp grid.
4. **QC** (`atac_metrics.h`). Each metric follows a named tool's definition so it can be compared:
   * fragments, mito fragments, duplicates, and TSS fragments (±1 kb): Cell Ranger;
   * TSS enrichment: ArchR (±50 bp vs flanks at 1.9–2 kb);
   * nucleosome signal: Signac (147–294 bp / < 147 bp);
   * peak-region fragments, cut sites and FRiP: Cell Ranger.
5. **Cells** (`atac_cell_caller.h::call_cells_mixture`). A two-component mixture on log10 fragments-in-peaks. With the
   paired GEX run (`--paired-gex`), the mixture is joint (GEX UMIs × ATAC fragments in peaks), which is how Cell Ranger
   ARC calls Multiome cells.
6. **Matrices**:
   * `atac_peak_counts`: cut sites, peaks × columns;
   * `atac_gene_activity`: Signac GeneActivity (gene body + 2 kb upstream, protein-coding).

   Columns are barcodes with ≥ 100 fragments, or that are called.
7. **Outputs**: `fragments.tsv.gz` (bgzip + tabix), `peaks.bed`, `atac_barcode_metrics.tsv`, `atac_cells.tsv`, the two
   matrices, and `atac_summary.json` (QC verdict).

### Visium (`visium_tissue.h`, Visium block of `singlet.cpp`)

1. **Spot list.** `visium-v1_coordinates.tsv` lists the 4,992 spots with array row/col. It was rebuilt from the
   `tissue_positions_list.csv` of three 10x public datasets, all identical (`bench/atac_spatial/make_lists.py`). Column
   barcodes are matched exactly, or else to the unique Hamming-1 spot.
2. **Tissue from counts.** A two-component mixture on log10 UMIs per spot gives each spot a probability. Mean-field
   smoothing on the hexagonal array (an Ising prior that tissue is contiguous) then refines it. Per-spot evidence is capped
   at a logit of ±4, so six concordant neighbours can fill a low-count spot inside tissue and remove a speck outside it.
   A slide with a single population is all tissue if its median is ≥ 500 UMIs, else empty. This needs no image.
   Space Ranger uses the H&E image, which GEO depositions often lack.
3. **Called set** = tissue spots. `cell_calls.tsv` is rewritten (`call_method = visium_tissue`), so the cells table's
   `called` flag means "spot under tissue".
4. **Outputs**:
   * `spatial_positions.tsv`: one row per column, with spot, array row/col, `in_tissue`, `tissue_prob`, `primary` and UMIs,
     plus Space Ranger's `in_tissue` and pixel coordinates when the authors' positions file is given;
   * `spatial_summary.json`: QC verdict.

## 3. QC rules (pass / fail)

ATAC (`atac_summary.json` → `qc`). Each rule must hold:

| metric | rule | source of the number |
|---|---|---|
| valid barcode fraction (MAPQ ≥ 30 pairs) | ≥ 0.75 | Cell Ranger ATAC warns below 0.85; 0.75 is the failure level |
| median TSS enrichment of cells | ≥ 4 | ArchR's default cell filter |
| FRiP of cells (fragments in peaks / fragments) | ≥ 0.20 | Cell Ranger ATAC / ARC alert level |
| fraction of fragments in cells | ≥ 0.30 | Cell Ranger ATAC alert level |
| cells | ≥ 100 | minimum for any per-sample analysis |
| median nucleosome signal of cells | ≤ 4 | Signac vignette cut |

Visium (`spatial_summary.json` → `qc`):

| metric | rule | source |
|---|---|---|
| fraction of UMIs on slide spots | ≥ 0.80 | a library whose barcodes are not Visium spots is not Visium (or not this slide version) |
| tissue spots | ≥ 50 | Space Ranger alert level |
| fraction of UMIs in tissue spots | ≥ 0.50 | Space Ranger warns on "fraction reads in spots under tissue" below 50% |
| median UMIs per tissue spot | ≥ 500 | a section with less is not usable for spot-level analysis |
| tissue and background separate | mixture means > 2 pooled SD apart | otherwise tissue cannot be told from background by counts |

## 4. Benchmark design

BENCH_PLACEHOLDER
