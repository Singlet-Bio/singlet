# Research Alpha benchmarks: methods and limitations

Status: 2026-10-26. Code: branch `claude/alpha-bench` of Singlet-Bio/singlet, folder `bench/alpha/`. Results: `results/`
here (the same files the Clipper job wrote to `$WORK/alpha_bench/results/`).

These benchmarks measure the atlas **as it is published today** (Research Alpha). They do not measure the fixed engine.
Every document carries `engine: "alpha (published atlas)"` and a `dataset_scope`. The A and B documents follow the
public results contract (schema 1.1.0, `bench/public/contract.py` on `claude/bench-public-cohort-485cf7`), and all three
pass `contract.validate`. The A documents use Alpha strata as dataset ids, and those ids are not rows of
`bench/public/cohort.tsv`. Validating them with `--cohort` needs Alpha rows added there, or a separate Alpha cohort file.

## A. Published bundles vs the authors' Cell Ranger matrices (159 samples picked, 135 scored)

**Sample choice** (`select_samples.py`, seed 7).

The Alpha public set has 47,702 samples. A sample is in it when all of these hold:
- its study is not withheld;
- the sample is not excluded (wrong strand, non-RNA library or empty matrix);
- the study is downloadable now;
- the sample is usable with QC tier 0.

From this set:
- 1,527 candidates were screened, one per study, across human/mouse × 3' v2 / 3' v3 / 5'.
- 793 had a filtered matrix in their GEO **sample** folder: a 10x `.h5` or a matrix/barcodes/features triplet, with no "raw" in the file name.
- Nuclei were recognised from the GEO sample text.
- Up to 10 samples were kept per stratum: species × {3' v2, 3' v3, 5', nuclei} × {<3,000, ≥3,000 Singlet cells}. Each kept sample is from a different study.
- That gave 159 samples in 16 strata. Human nuclei small had only 9.

**Data.** Singlet's side is read from data.singlet.bio by exact HTTP ranges:
- the zip central directory;
- then exon_counts, intron_counts, cell_calls, summary.json and pileup_stats.

No request touches a restricted member. The authors' matrix comes straight from GEO.

**Metrics** (`alpha_compare.py`, `aggregate.py`):
- **Recovery** = called by both / authors' cells.
- **Precision** = called by both / Singlet's cells.
- **UMIs per cell**, Singlet / authors, on cells both called.
- **Per-cell log1p Pearson correlation** after binomially thinning both profiles to half the smaller per-cell total. The noise floor is two independent thinnings of the authors' profile.
- **Pseudobulk Spearman** over common genes.
- **reads_used / reads_available**: summary.json input reads over all SRA spots of the sample (NCBI runinfo).

Singlet's exon or exon+intron layer is chosen per sample, whichever fits the authors' pseudobulk better: exon only in 85 samples, exon+intron in 52 (Cell Ranger ≥7 counts introns). Results are medians over samples with bootstrap 95% CIs, pooled and per stratum.

**Headline (135 scored samples):**

| metric | median (95% CI) | rule | result |
|---|---|---|---|
| authors' cells recovered | **0.65** (0.60–0.72) | ≥0.95 | fail |
| precision of Singlet's calls | **1.00** (0.999–1.00) | ≥0.95 | pass |
| authors' cells present in Singlet's barcode matrix | 1.00 | info | — |
| UMIs per cell, Singlet / authors | **0.26** (0.23–0.30) | 1.0 ± 10% | fail |
| per-cell correlation at equal depth | **0.565** vs noise floor 0.602 | match floor | fail (gap ≈ 0.04) |
| pseudobulk Spearman | **0.952** (0.949–0.958) | ≥0.95 | pass |
| reads used / available in SRA | **0.10** (0.087–0.12) | info | — |

Recovery by stratum (medians) runs from 0.46 (mouse 5' small) to 0.91 (mouse 3' v2 large). Every cell the authors called
is in Singlet's barcode matrix, so the missed cells are a calling problem, not missing data. The UMI ratio tracks the
share of reads used (`fig_umi_ratio_vs_reads_used`).

**Not scored (24):**
- 16 deposits were unfiltered (raw barcodes);
- 5 bundles returned HTTP 404 during the final run. All are mouse nuclei; they were scored in an earlier run, and their studies were probably repacked or withdrawn meanwhile;
- 2 had barcodes that do not match Singlet's;
- 1 h5 had features that do not match its rows.

## B. The campaign engine at campaign settings on 7 public 10x datasets

**What existed.** `pipeline-benchmark/runs/<ds>/singlet-baseline-t4` existed: the atlas engine b1335a8, 4 threads, but **all reads**. Nothing existed at the 30M cap, so the cap was run.

**How it was run** (`engine_build.sbatch`, `engine_run.sbatch`):
- **Binary:** git 8e1803ba, the build most Alpha samples report (v0.3.0-pilot-freeze).
- **Reads:** the head 30M read pairs of the lane-sorted FASTQs.
- **Command:** `--threads 4 --pipeline --output-format 1pz`, plus the human-only campaign flags (`--snps --cascade auto --txome-index --te-classify on --te-db`).

**How it was scored** (`engine_score.py`), with the pipeline benchmark's loaders and scorers, against:
- STARsolo EmptyDrops_CR on all reads;
- 10x's published Cell Ranger matrices;
- barnyard species truth;
- CITE-seq ADT clusters.

| dataset | recovery vs 10x CR (STARsolo) | same engine, all reads | precision | exon UMIs / STARsolo Gene | equal-depth r (floor) |
|---|---|---|---|---|---|
| pbmc1k_v3 | 0.92 (0.99) | 0.93 | 0.99 | 0.82 | 0.80 (0.81) |
| pbmc1k_v2 | 0.98 (1.00) | 0.98 | 1.00 | 1.06 | 0.83 (0.84) |
| pbmc1k_cite_v3 | 0.82 (1.00) | 0.82 | 1.00 | 0.95 | 0.82 (0.83) |
| neuron1k_v3 | 0.84 (0.96) | 0.85 | 1.00 | 0.58 | 0.68 (0.69) |
| hgmm1k_v3 | 0.84 (1.00) | 0.84 | 1.00 | 0.51 | 0.74 (0.75) |
| pbmc5p_v1 | 0.80 (0.94) | 0.80 | 1.00 | **0.05** | **0.01 (0.23)**, 92% wrong strand |
| nuclei2k_v2 | no 10x matrix | — | — | 0.69 | 0.48 (0.50) |

- **Barnyard:** the mixed-species rate is 0.11%, against STARsolo 0.48% and Cell Ranger 0.67%. The cross-species UMI share is 0.50%, against 0.51% and 0.45%. Both pass.
- **CITE-seq:** the ARI of RNA clusters vs ADT clusters is 0.55, against STARsolo 0.44 and Cell Ranger 0.57. Pass, within the tolerance set by the gap between those two.
- **Main finding:** the 30M cap barely changes cell recovery here; all reads give the same numbers. The cells are missed by the knee filter, not by the cap.
- **The 5' library** is counted antisense, as `pipeline-validation.md` §4.6 reported.

## C. Atlas QC census (`alpha_census.json`)

**Atlas size.**

| level | studies | samples | cells |
|---|---|---|---|
| bundles before exclusions | 8,381 | 72,386 | 305.5M |
| after exclusions | 7,391 | 51,879 | 239.6M |
| public (the site's basis) | 7,040 | 47,702 | 228.0M |

By species and chemistry, see `public_by_species` and `public_by_chemistry` in the JSON. Human is 27,222 samples and mouse 18,675.

**Reads used**, from each published sample's own `summary.json`, for 47,214 samples:
- 89% used exactly 30M reads.
- Over the SRA spots in Singlet's catalog, the median share used is **0.10** (p10–p90 0.04–0.42).
- 92% used less than half of their reads.
- 37% of samples have more than one run, and every run after the first was dropped.

**Engine builds.** Alpha comes from three builds:

| build | samples |
|---|---|
| 8e1803ba | 27,881 |
| 37e7cbdf | 18,134 |
| b1335a8 (reports 1.0.0) | 1,199 |

## Limitations

- **The authors' matrices are not uniform truth.** Cell Ranger versions differ, and some authors filtered cells further before depositing. Both lower recovery without being Singlet's error.
- **Sample choice is biased.** Only samples with a deposited filtered matrix in the GEO sample folder qualify, which favours newer studies. The strata are equally weighted, not weighted by their share of the atlas.
- **Nuclei and 5' labels come from GEO text and curation**, so some may be wrong.
- **Mouse strand is unscreened in Alpha.**
- **The equal-depth correlation is not the atlas's real depth.** It is measured at half the smaller depth, so its absolute values are lower than at full depth.
- **reads_available overstates GEX reads** for samples that also hold non-GEX runs.
- **B approximates SRA read order.** "Head of the lane-sorted FASTQs" stands in for "head of the SRA run".
- **B memory:** hgmm1k_v3 needed 120 GB, where the campaign requested 64 GB.
- **B runtimes are not representative.** Singlet's runtimes include an 18-minute transcriptome-index load for `--cascade auto` from NFS, though the cascade never engaged.
- **Rule tolerances:**
  - Recovery and precision use ≥0.95, the Cell Ranger vs EmptyDrops_CR Jaccard on identical reads.
  - The UMI ratio uses ±10%.
  - The equal-depth correlation must match the floor within the bootstrap CI.
- **No simulation is used anywhere.**

## Compute

About 32.5 CPU-h allocated on Clipper, short-cpu and debug-cpu.

| jobs | what |
|---|---|
| 787763 | build |
| 787770, 787956 | engine runs; 787770_6 was killed for memory and rerun as 787956 |
| 787959, 787985 | engine scoring |
| 787768, 787781, 787965, 787984 | per-sample comparisons |
| 787767 | catalog reads |
| 787918 | summary.json census, 8 shards |
| 787960, 787986, 787961 | finalize; 787961 was a test |
