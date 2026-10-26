# ATAC, Multiome and Visium in Singlet 1.0.0

Singlet's pipeline already had ATAC and Visium code paths, but neither had been checked against the tools people use. This
page records what was wrong, what changed, the QC rules, and how the result compares with Cell Ranger ARC, Cell Ranger
ATAC and Space Ranger (published outputs), and with open-source pipelines run on the same FASTQs.

Code: `include/singlet/pileup/atac_*.h`, `visium_tissue.h`, `src/pipeline/singlet.cpp` (ATAC branch, Visium block,
`singlet atac` subcommand). Harness: `bench/atac_spatial/`. Storage proposal: `docs/format-v2.md` section 14. Public results:
`bench/public/results/{atac,multiome,spatial}.json` (contract `bench/public/CONTRACT.md`).

## Results at a glance

Each comparison starts from the same FASTQs, reference (10x 2024-A) and threads (16). Cell Ranger and Space Ranger are
the truth (10x published outputs). The competitor is the open-source route a user would take:
- ATAC: Chromap + MACS3 (Signac recipe) + SnapATAC2 / ArchR filter.
- Visium: STARsolo with the same spot list; for tissue, an Otsu threshold.

Tolerances are the half-width of the bootstrap 95% CI of the paired Singlet − competitor difference (section 4).

| dataset | metric | Singlet | competitor | verdict |
|---|---|---|---|---|
| scATAC PBMC 5k (held out) | cell calls, Jaccard with Cell Ranger | **0.950** | 0.884 | pass |
| | fragments per cell, Spearman | **0.9967** | 0.9966 | pass |
| | Cell Ranger fragments reproduced exactly (chr21) | **0.912** | 0.906 | pass |
| | peaks, F1 vs Cell Ranger | **0.897** | 0.892 | pass |
| | TSS fragments per cell, Spearman | 0.994 | – (≥ 0.9) | pass |
| Multiome PBMC 10k (held out) | see `bench/public/results/atac.json`, `multiome.json` | MO10K | | |
| Multiome PBMC 3k (development) | joint cell calls, Jaccard with ARC | 0.987 | 0.960 | (dev) |
| | ATAC→GEX barcode pairing identical to ARC | 1.000 | – | (dev) |
| | exact fragments / peak F1 | 0.980 / 0.872 | 0.975 / 0.868 | (dev) |
| Visium mouse brain | UMIs per tissue spot, Spearman with Space Ranger | 0.9948 | **0.9982** | fail |
| | per-spot profile, median Pearson | 0.966 | **0.989** | fail |
| | molecules per gene, Spearman | 0.976 | **0.984** | fail |
| | tissue (no image), Cohen's κ with Space Ranger's image call | **0.850** | 0.776 | pass |
| Visium FFPE human breast | UMIs per tissue spot, Spearman | 0.9995 | **0.9996** | fail |
| | per-spot profile, median Pearson | 0.969 | **0.973** | fail |
| | tissue κ | **0.840** | 0.827 | pass |

Runtime, scATAC 5k (215 M read pairs), 16 threads:
- Singlet (encode + STAR + post-alignment): 35 min wall, 2.7 CPU-h, 19 GB.
- Chromap + MACS3 + SnapATAC2: 8 + 15 min, 1.6 CPU-h, 21 GB.

Singlet is an exact genome-wide aligner run as part of the full single-cell pipeline. Chromap is a fast approximate mapper
built for chromatin data. The difference is a resource, not a gate (Pareto figures in the contract documents).

**What fails, and why.** Visium spot counts trail STARsolo slightly on both slides. The gap is in Singlet's gene
counting, which Visium shares with every single-cell sample; it is not in the Visium handling.
- On GRCm39, Singlet loses almost all molecules of mt-Co2, mt-Atp6, mt-Co3 and mt-Nd3, for example mt-Co3: 71 vs
  Space Ranger's 1.7 M. These genes lie in the region covered by the chr1 NUMT, so their reads are 2-locus multimappers.
  STARsolo and Cell Ranger count a multimapper that overlaps a single gene; Singlet drops it. The same loss appears in
  every single-cell mouse run, for example neuron1k_v3: mt-Atp6 16 vs STARsolo 3,978. This is reported to the
  orchestrator for the gex-counts owner.
- On the FFPE probe slide (no mito probes), the remaining difference is under 0.5% in any metric.

Visium handling itself works: spot assignment, strand auto-detection on FFPE, and the image-free tissue call, which beats
the Otsu baseline on both slides.

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

* **Datasets.** All are 10x Genomics public datasets (CC BY 4.0), as deposited.
  * scATAC PBMC 5k Next GEM, Cell Ranger ATAC 1.2.0. Its outputs are on hg19; peaks and fragments are lifted to GRCh38
    with UCSC's chain.
  * Multiome PBMC 10k (granulocytes sorted out), Cell Ranger ARC 2.0.0.
  * Multiome PBMC 3k, ARC 2.0.0: development data, not scored.
  * Visium mouse brain sagittal anterior, Space Ranger 1.1.0.
  * Visium FFPE human breast cancer, Space Ranger 1.3.0. It is the only human Visium dataset 10x serves with FASTQs.
* **Held-out discipline.** Every parameter change after the first runs was chosen on Multiome 3k:
  * the density-valley ATAC-only cell cut;
  * STAR `Extend5pOfReads12`;
  * peak q 0.05, from a sweep over 1e-4 to 0.05.

  The 5k scATAC cell count (6,895 with the first caller) was seen before the cell caller changed, and the change was
  made on 3k evidence alone. The Visium tissue-model constants were fixed on synthetic data before any Visium run.
* **Tolerances.** Each match-or-beat rule carries the half-width of the bootstrap 95% CI of the paired difference
  between Singlet and the competitor. The resampling unit is:
  * called barcodes for cell calls;
  * cells for per-cell correlations;
  * spots for Visium;
  * genes for gene totals;
  * chromosomes for peaks.

  Fragment recall uses 1.96 × the binomial SE.

  These margins are tiny (≤ 0.007), so close calls are decided almost strictly.
* **Harness.** `bench/atac_spatial/submit.sh` runs the whole benchmark on Clipper:
  * download, comparator env, barcode lists and build;
  * the Singlet runs;
  * the Chromap and STARsolo comparators;
  * `make_artifacts.py` (small per-barcode, per-spot and per-gene tables and peak BEDs) and `make_provenance.py`.

  `bench/public/capabilities/{atac,multiome,spatial}.py` turn the artifacts into the contract documents in a few minutes
  on a laptop.

## 5. Barcode lists

Rebuilt only from 10x public datasets (CC BY 4.0), never from 10x software. They are kept on Clipper
(`pipeline-benchmark/whitelists/atacsp/`) and are not yet in git, pending the licence decision:

| list | source | size | check |
|---|---|---|---|
| `visium-v1_coordinates.tsv` / `visium-v1.txt` | `tissue_positions_list.csv` of 3 Visium datasets | 4,992 spots | identical across the 3 slides; 97.5% of mouse-brain reads match; 7 barcodes shared with 3M-february-2018 |
| `737K-arc-v1-atac-gex.tsv.gz` | `per_barcode_metrics.csv` of 7 ARC runs | 736,198 pairs (of 736,320 GEX barcodes) | leave-one-run-out: 100% of each run's cells known |
| `737K-cratac-v1.txt` | `singlecell.csv` of 12 Cell Ranger ATAC runs | 736,795 | leave-one-run-out ≥ 99.95% of cells; the bundled file of that name is a copy of `737K-august-2016.txt` and shares 10,801 barcodes with it |

## 6. Atlas implications

* The atlas tags 756 samples as Spatial. Two issues affect them:
  * Without the spot list, Visium libraries are detected as 3' v3, their spots are not corrected, and EmptyDrops
    "calls" a few dozen spots.
  * With the list installed, detection, spot correction, tissue calls and coordinates come out of the normal pipeline.
* ATAC libraries processed as RNA (139 in `docs/library-type-screen.md`) and the Multiome ATAC halves need the ATAC
  path: `--protocol 10x-atac` at encode, or detection from the Tn5 signature. Their RNA-mode objects should not be
  migrated.
* Mouse samples are affected by the chr1 NUMT mito-gene loss above. It changes pct_mt and mito-gene counts in every mouse
  sample. It needs a fix in the GEX counting before the re-run.

## 7. Open items

* Licence answer for the three barcode lists, then bundle them, replacing the wrong `737K-cratac-v1.txt`.
* Merge into the campaign binary. The results here use `singlet-atacsp`: this branch merged onto `prod/modality-fixes`.
* Visium HD and Slide-seq are not handled: different barcode geometry and bin aggregation. Visium v2 / CytAssist slides
  need their own spot lists, which no 10x public dataset with FASTQs supplies yet.
* The `fragments` modality reader (byte-range region queries in the client) is specified (format-v2 section 14) but not
  implemented.
