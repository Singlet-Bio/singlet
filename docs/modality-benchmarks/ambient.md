# Ambient RNA and barnyard doublets: Singlet vs SoupX, DecontX, CellBender, Scrublet, scDblFinder

Benchmark of the `singlet-fixed-det` binary on a species-mixing library with a known answer (2026-10-24). Scripts and result
tables: `bench/modalities/ambient/`.

## Verdicts

| Modality | Verdict | Key number |
|---|---|---|
| Ambient contamination estimate (`ambient_contamination.tsv`) | **worse** | mean rho 0.27% vs truth 1.01% (0.27x); per-cell Spearman 0.41 against 0.60-0.70 for CellBender / DecontX with background |
| Ambient correction (`correct_counts`) | **missing** | the CLI never writes a corrected matrix; with the formula applied by hand it removes 0.0005% of UMIs (SoupX 1.0%, CellBender 0.9%) |
| Ambient profile (`ambient_profile.tsv`) | works | exon-level profile over 50,000 empty droplets; its species composition was not separately checked (gene-level pool is 27.8% mouse) |
| `ambient_contamination.json` summary | **broken** | `estimated_contamination_fraction` is always 0 |
| Barnyard species call (`barnyard_classification.tsv`) | **equivalent** on called cells, misleading as written | 421 human / 458 mouse / 1 doublet among 880 cells, the same as an independent Cell Ranger count; but the file covers 111,861 barcodes and the log reports `doublet=69093` |
| Singlet general doublet score (`doublet_scores.tsv`) | **worse** | flags 220 of 880 cells (25%); truth has 1 heterotypic doublet; on synthetic doublets precision 0.20, recall 1.00 |

## Dataset and truth

10x `hgmm_1k_v3` (HEK293T + NIH3T3, 3' v3, 76.7 M read pairs), joint GRCh38 + GRCm39 2024-A reference
(`cellarium/reference/GRCh38-GRCm39-barnyard`). FASTQs and the Cell Ranger filtered h5 were already on Clipper
(`pipeline-benchmark/data/hgmm1k_v3`).

* **Cells:** the 880 cells Singlet called (pipeline default, STARsolo `CellRanger2.2`; Cell Ranger calls 1,046, CellBender 1,048).
  Every tool is scored on these 880. Species by gene prefix (ENSG / ENSMUSG): 421 human (mouse fraction <= 5%), 458 mouse, 1 mixed.
* **Ambient pool composition:** barcodes with 1-100 UMI (Singlet's own rule, top 50,000): 27.75% mouse UMIs. Other pool choices:
  27.2% (10-100 UMI), 27.3% (20-200), 32.8% (1-20 UMI).
* **True contamination:** in a pure human cell every mouse read is ambient, so
  `rho_true = mouse_fraction / mouse_share_of_pool` (human share for mouse cells). Mean over the 879 pure cells 1.01%
  (UMI-weighted 0.92%); 1.27 / 0.89 / 0.87% in the low / mid / high depth thirds. Per-cell truth is Poisson-noisy, so per-cell
  correlations have a ceiling below 1; the bias ratios are the robust column.
* **Doublet truth:** minor-species fraction >= 0.2 (0.1 and 0.3 give the same set; Cell Ranger counts agree). Only **one** such cell
  exists among the 880 (minor fraction 0.35), so a second, synthetic test adds 44 human+mouse sums (5%).

## Commands

All jobs: `--account=debruinz`, `short-cpu` (CellBender `short-gpu`), outputs under `.../singlet/modality-bench/ambient/`.

```
singlet_run.sbatch   singlet-fixed-det encode ...; singlet-fixed-det s.1fq --ref-base $REFBASE --genome-dir $REF/star_2.7.11b \
                       --exons $REF/genes/genes.gtf --threads 16 --pipeline --output-format mtx --raw-matrix   (7 min, 60 GB)
setup_env.sbatch     conda env envs/ambient: R 4.4, SoupX 1.6, celda 1.22 (DecontX), scDblFinder 1.23, scrublet, scanpy
prep.sbatch          prep.py: species, pool, truth, clusters, trimmed raw matrix (barcodes >= 1 UMI)
tools.sbatch         run_r_tools.R (SoupX autoEstCont, DecontX x2, scDblFinder), run_scrublet.py
cellbender.sbatch    cellbender remove-background --cuda --expected-cells 1000 --total-droplets-included 6000 --epochs 150 --fpr 0.01
                       (CellBender 0.3.2 container, 12.5 min)
evaluate.sbatch      evaluate.py; dbl.sbatch: dbl_synth.py, dbl_harness.cpp (detect_doublets compiled unchanged), dbl_tools.py
```

Inputs: the raw barcode x gene matrix is Singlet's own STAR pass (`star_Solo.out/Gene/raw`); `--raw-matrix` writes nothing (below).
SoupX clusters are 6 k-means clusters on the PCA (igraph was not installed, so no Leiden); SoupX with species labels as clusters
failed ("No plausible marker genes"). DecontX was run with its own clustering ("default") and with 20,000 empty droplets (10-100 UMI)
as background. Singlet's correction was reproduced by applying `correct_counts()` (`ambient_correction.h:236-253`) to the exon
matrix with the profile and per-cell rho the binary wrote.

## Ambient estimate against truth

Mean over 879 pure cells (truth 1.01%). "Implied" = fraction of a cell's UMIs removed by the tool.

| Estimator | mean | ratio to truth | Spearman | Pearson | low / mid / high depth (truth 1.27 / 0.89 / 0.87%) |
|---|---|---|---|---|---|
| Singlet `rho` | 0.27% | 0.27 | 0.41 | 0.35 | 0.43 / 0.26 / 0.12% |
| SoupX (global rho 0.010, implied) | 0.98% | 0.97 | 0.49 | 0.51 | 1.02 / 0.97 / 0.94% |
| DecontX default | 0.44% | 0.44 | 0.38 | 0.49 | 0.60 / 0.51 / 0.21% |
| DecontX with background | 0.74% | 0.73 | 0.69 | 0.74 | 0.82 / 0.66 / 0.75% |
| CellBender | 1.42% | 1.40 | 0.60 | 0.48 | 2.71 / 0.98 / 0.54% |

Singlet's estimate is `S / N_cell` (mean UMIs per empty droplet over cell depth, `ambient_correction.h:180-190`), so it is a
function of depth alone: it falls 3.6x from the low to the high depth third, while the truth is nearly flat (1.27 to 0.87%). It
sees no species or gene information. Its median rho is 0.27% and 129 of 880 cells sit on the 0.001 floor.

## Correction: residual cross-species counts and over-correction

Pure cells only. Cross-species UMIs before correction: 98,443 (Singlet exon matrix: 94,880).

| Tool | cross-species UMIs left | own-species UMIs removed | removed / expected own-species ambient | total UMIs removed | top-100 marker genes kept (human / mouse) |
|---|---|---|---|---|---|
| Singlet (formula, not produced by CLI) | 100.0% | 0.0005% | 0.001 | 0.0005% | n/a (exon level) |
| SoupX | 23.1% | 0.67% | 1.25 | 1.00% | 99.7 / 99.8% |
| DecontX default | 14.7% | 0.0002% | 0.000 | 0.38% | 100.0 / 100.0% |
| DecontX with background | 23.1% | 0.43% | 0.81 | 0.85% | 99.8 / 99.9% |
| CellBender | 11.5% | 0.56% | 1.03 | 0.90% | 99.0 / 99.3% |

"Expected own-species ambient" is inferred from each cell's cross-species reads and the pool composition; a value near 1 means the
tool removed about the amount of own-species ambient that must be there. Markers are the 100 genes per species with the highest
cell-to-ambient fraction ratio (mean >= 0.3 UMI per cell). No tool over-corrects here: marker loss is at most 1%. DecontX default
removes cross-species reads (85%) while removing almost no own-species reads, which only works because it uses the species
structure of the cells; it removes less than half the true amount overall. The cross-species residual of 11-23% is not a floor
estimate: some of it is real mouse-human misassignment, not ambient.

## Barnyard and doublets

**Species call.** On the 880 cells `classify_barnyard` gives 421 human / 458 mouse / 1 doublet, identical to the species labels
used here (it is the same rule). As written, `barnyard_classification.tsv` covers all 111,861 barcodes in the matrix: 16,318 human,
8,223 mouse, 69,093 "doublet", 18,088 "ambiguous" (empty droplets with a few UMIs of each species), and the stderr line prints
those totals. Anyone reading the log would conclude a 63% doublet rate.

**Natural cells** (1 heterotypic doublet, 880 cells):

| Caller | called | TP | FP |
|---|---|---|---|
| Singlet barnyard | 1 | 1 | 0 |
| Singlet `detect_doublets` | 220 | 1 | 219 |
| scDblFinder | 17 | 1 | 16 |
| Scrublet (expected rate 0.03 / 0.08) | 1 / 1 | 1 | 0 |

**With 44 synthetic human+mouse doublets added** (45 positives in 924 columns; Singlet's header compiled unchanged and fed its exon
matrix):

| Caller | called | precision | recall | F1 | AUROC | false positives on natural cells |
|---|---|---|---|---|---|---|
| Singlet `detect_doublets` | 224 | 0.20 | 1.00 | 0.33 | 0.987 | 179 |
| scDblFinder | 27 | 1.00 | 0.60 | 0.75 | 1.000 | 0 |
| Scrublet 0.03 | 41 | 0.95 | 0.87 | 0.91 | 0.993 | 2 |
| Scrublet 0.08 | 43 | 0.93 | 0.89 | 0.91 | 0.993 | 3 |

Singlet's score ranks the doublets well (AUROC 0.987) but its threshold is far too low: the GMM threshold (clamped to >= 0.15,
`doublet_detect.h:172-267`) lands inside the singlet distribution, and 25% of cells are flagged. Only heterotypic (cross-species) doublets exist in the truth; homotypic doublets are invisible to every tool here.

## What is missing or broken (file:line, no code changed)

1. **No corrected matrix is ever written.** `run_ambient_correction` is declared (`include/singlet/pileup/export.h:185`) and
   consumed (`export.h:736`, `:1107`) but nothing in `src/pipeline/singlet.cpp` sets it; there is no CLI flag. The log always says
   `corrected=no`. Fix: add a flag, and apply it to the gene-level matrix.
2. **The correction would be a no-op at exon level.** `round(rho * N * frac_g)` (`ambient_correction.h:250-251`) with the profile
   spread over 696,825 exon features is below 0.5 for almost every entry (0.0005% of UMIs removed on this library).
3. **`estimated_contamination_fraction` is always 0.** `export.h:709` averages only cells with `n_genes_used > 0`;
   `ambient_correction.h:193` sets `n_genes_used = 0` for every cell.
4. **The estimator ignores the data's contamination signal.** `ambient_correction.h:180-190` (rho = mean empty-droplet UMIs / cell
   UMIs). A fix needs a gene-level signal: SoupX-style non-expressed genes per cluster, or the species structure when available.
   The header comment (`:17-27`) documents why the earlier marker-gene version failed; cluster-aware markers avoid that circularity.
5. **`--raw-matrix` is a no-op.** `write_raw_matrix` is set (`singlet.cpp:2715`, `:7988`) and declared (`export.h:191`) but never
   read. The "filtered" matrix (`filtered_feature_bc_matrix/`) contains 111,861 barcodes, not the 880 called cells.
6. **Barnyard summary covers all barcodes**, not the called cells (`export.h:1064-1078`, `classify_barnyard(gene_csc, ...)`).
7. **`detect_doublets` threshold** (see above) and it runs on the exon matrix with 200 HVG chosen by dispersion
   (`doublet_detect.h:326-357`; `export.h:789`).
8. `ambient_profile.h` is not included anywhere and defines a second `singlet::AmbientProfile` that would clash with
   `ambient_correction.h:59` if both were included.

## Caveats

* One library, 880 cells, two cell lines; truth assumes all cross-species reads are ambient and the pool composition is the same for
  every droplet. The mouse share of the pool moves truth by about +-20% (1-20 UMI pool: 32.8%).
* The cell set is Singlet's (STARsolo default call), not Cell Ranger's 1,046; tools with their own cell calls were scored on the
  same 880.
* SoupX used k-means clusters; with species labels it failed. SoupX's reported global rho is 0.01 (its default floor is 0.01; the
  range was widened to 0.001 and it still returned 0.01 with 247 estimates). DecontX and CellBender hyperparameters are defaults
  (CellBender: expected cells 1000, 6000 droplets, FPR 0.01).
* A repeat of the same binary on the same input gave 220 doublet calls; an earlier `singlet-fixed` run gave 232.
* The synthetic doublets are plain sums of two cells, the easiest case for simulation-based callers.
