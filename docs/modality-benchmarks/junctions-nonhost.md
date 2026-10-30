# Benchmark: splice junctions / PSI and non-host detection

Binary `singlet-fixed-det` (1.0.0), GRCh38-2024-A, run on GVSU Clipper. All numbers are from the jobs listed under
Reproduction; small result files are in `bench/modalities/sj_nonhost/results/`.

## Verdicts

| Modality | Reference | Verdict |
|---|---|---|
| Junction counts (per cell) | STARsolo `--soloFeatures SJ`, same reads | **Equivalent** for junctions with at least 10 molecules (r = 0.994, per-cell totals r = 0.9999). **Worse** at low counts: Singlet applies no junction filter (177,593 junctions vs 89,541; 134,242 have a single count). |
| PSI (emitted `splice_psi`) | PSI recomputed from STARsolo counts, same definition | **Equivalent and the legacy 0/1 bug is absent** (continuous float values, 649 distinct). Of limited use: 96.7 % of entries are exactly 1.0, and the grouping is not a cassette-exon PSI. |
| Cassette-exon PSI from Singlet junction counts | PSI from STARsolo counts | **Equivalent** (pseudobulk r = 0.996, MAE 0.008 at coverage >= 20). |
| Non-host, sample level | Kraken2 (viral DB) + minimap2 oracle, SARS-CoV-2 infected vs mock | **Equivalent** on the true positive (554,070 vs 553,261 reads), **better** on the clean sample (PBMC: no species called; Kraken reports 34k viral reads, all host-derived artefacts). |
| Non-host, per cell | same | **Equivalent** to Kraken2 for the confirmed virus (Spearman 0.985 on per-cell UMIs), **broken by default**: the per-cell table is not written on a fresh output directory (`src/pipeline/singlet.cpp:1613`). |

One further defect affects both modalities on PBMC: Singlet clips 11 bases from every R2 read on a false-positive
"constant 5' prefix" (details below).

## Datasets

| Label | Data | Reads used | Truth |
|---|---|---|---|
| PBMC | 10x PBMC 1k v3 (3' v3), `pipeline-benchmark/data/pbmc1k_v3` | 66.6 M pairs (2 lanes) | uninfected; junctions/PSI against STARsolo |
| mock | GSE166766 / GSM5082289 (SRR13711613), Ravindra et al. 2021, mock human bronchial epithelium, 10x 3' v3 | first 50 M spots | no SARS-CoV-2 |
| dpi2 | GSM5082291 (SRR13711615), same study, 2 days post infection | first 50 M spots | SARS-CoV-2 infected |

ENA serves only the cDNA read of these runs (one FASTQ per run, barcodes lost), so the reads were fetched with
`fastq-dump -X 50000000 --split-files` (I1 8 nt, R1 28 nt, R2 91 nt) inside a job.

Truth is sample-level only. Not every dpi2 cell is infected: with minimap2 against NC_045512.2 (oracle), 41.6 % of dpi2 cells carry
at least one SARS-CoV-2 UMI, so the best achievable AUROC for "mock vs dpi2" per cell is about 0.71 to 0.73. The mock is not perfectly clean
either: the oracle finds 3,246 SARS-CoV-2 reads (13 UMIs in 7 cells), presumably index hopping or ambient RNA.

## 1. Splice junctions and PSI (PBMC)

### Setup

* Singlet: `bench/atlas_rerun` run `perf/pbmc1k_v3_det/t4_r1` (no screen): `sj_counts.mtx.gz` (junction x barcode, integer),
  `splice_psi.mtx`, `splice_events.tsv`, `star_SJ.out.tab`.
* STARsolo 2.7.11b on the same FASTQs (`starsolo_sj_pbmc.sbatch`): `CB_UMI_Simple`, 3M-february-2018 whitelist,
  `--soloFeatures Gene SJ`, `--soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts`, `--outFilterScoreMin 10`, EmptyDrops_CR cell filter (1,227 cells).
  Two runs: raw reads ("default") and `--clip5pNbases 11` ("clip-matched", see the clipping finding).
* `sj_compare.py`: Singlet coordinates are STAR coordinates minus 1 on the intron start (found automatically, 97,357 of 177,593 junctions match STAR's filtered list).
  Singlet counts are molecules: its total 4.27 M compares with 4.00 M STARsolo UMIs, while STAR's unique-read total is 7.80 M.
  Per-cell comparisons use the 1,227 STARsolo-called cells.

### Alignment input differs: 5' clip false positive

Singlet logs `Detected 11bp constant 5' prefix in R2 reads` on PBMC and clips 11 bases from all cDNA reads. The R2 base composition
at positions 1 to 14 is flat (each base 0.15 to 0.34; `r2_composition.sbatch`), so no such prefix exists. The detector
(`src/pipeline/singlet.cpp:4538-4610`) looks at the first 25 reads of the .1fq, which is sorted by barcode, so duplicate reads of one molecule
look like a constant prefix. It did not fire on mock or dpi2. Effect: STAR reports 7.87 M splices in Singlet's run vs 9.21 M on the raw reads
(unique mapping 57.92 M vs 58.47 M reads); STARsolo with `--clip5pNbases 11` reproduces Singlet (7.89 M splices). The clip-matched run isolates the counting layer.

### Junction counts

| Metric | STARsolo default | STARsolo clip-matched |
|---|---|---|
| Junctions with any count (Singlet / STARsolo) | 177,593 / 95,213 | 177,593 / 89,541 |
| Jaccard, junctions with >= 1 molecule | 0.418 | 0.475 |
| Jaccard, >= 3 | 0.815 | 0.884 |
| Jaccard, >= 10 | 0.855 | 0.923 |
| Jaccard, >= 30 | 0.869 | 0.941 |
| Count correlation, both >= 10 (Pearson on log10, Spearman) | 0.991, 0.980 | 0.994, 0.992 |
| Median Singlet / STARsolo count ratio (both >= 10) | 0.91 | 1.00 |
| Junctions >= 3 only in Singlet / only in STARsolo | 2,349 / 558 | 2,013 / 469 |
| Per-cell total junction counts, Pearson across 1,227 cells | 0.9998 | 0.9999 |
| Median distinct junctions per cell (Singlet / STARsolo) | 911 / 885 | 911 / 812 |
| Median per-cell Jaccard of junction sets | 0.80 | 0.87 |
| (cell, junction) entries nonzero in both: exact equal | 82.0 % | 94.0 % |
| (cell, junction) Pearson, both nonzero | 0.990 | 0.993 |

Singlet agrees with STARsolo wherever STARsolo reports a junction. Disagreement is concentrated in low counts, because STARsolo only
reports junctions that pass STAR's `outSJfilter*` rules (novel and non-canonical junctions need several unique reads) and Singlet keeps every
junction seen (134,242 of 177,593 have exactly one count). STAR's own `SJ.out.tab` counts reads, not molecules, so it is not a count reference (median ratio 0.55).

### PSI

Singlet's `splice_psi` (`include/singlet/pileup/splice_psi.h`) is the junction's share of the counts of its donor group or acceptor group in the same cell
(16,077 events, 840,751 entries).

* **0/1 bug (`docs/v2-migration.md` D8): not present.** Header `real`, float values, 649 distinct values, no zeros, min 0.003.
  Recomputing from `sj_counts` reproduces the file (max |difference| 5e-7, float32 rounding).
* 96.7 % of entries are exactly 1.0: in most cells only one junction of an event is observed. Absence means "not observed", not 0.
* Each junction is assigned to one event only (donor or acceptor, whichever event id sorts first as a string, `splice_psi.h:96-135`), so the value depends on an
  arbitrary tie-break rather than on biology.
* Singlet PSI vs the same definition applied to STARsolo counts, 640,698 common (cell, junction) entries (default / clip-matched):
  Pearson 0.931 / 0.947, MAE 0.0033 / 0.0022, 98.9 % / 99.4 % within 0.05.

Cassette exons computed client-side from junction counts (annotated GTF triples inclusion-1, inclusion-2, skipping; PSI = mean inclusion / (mean inclusion + skipping)).
Of 54,411 annotated cassette triples, 1,841 to 2,014 have all three junctions observed.

| Cassette PSI, Singlet vs STARsolo | default | clip-matched |
|---|---|---|
| Pseudobulk (called cells), coverage >= 20: n, Pearson, MAE | 836, 0.995, 0.0138 | 805, 0.996, 0.0076 |
| Pseudobulk, coverage >= 50: n, Pearson, MAE | 495, 0.999, 0.0079 | 482, 0.999, 0.0043 |
| Per cell, coverage >= 5: pairs, Pearson, MAE | 34,002, 0.9996, 0.0030 | 33,714, 0.9998, 0.0016 |

### Other findings (junction outputs)

* `splice_psi.mtx`, `splice_psi_features.tsv`, `splice_psi_barcodes.tsv` are gzip files under non-`.gz` names (magic bytes `1f 8b`); `sj_counts.1pz` and `splice_psi.1pz` are 151-byte
  empty stubs with `--output-format mtx` (`warnings: no_1pz_output_format`). Writers: `include/singlet/pileup/export.h:1117-1139`.
* The sj matrix has 153,798 barcode columns (all barcodes with a junction), not only called cells.

## 2. Non-host detection

### Setup

* Singlet: `singlet-fixed-det` with `SINGLET_REF_BASE` set (viral, bacterial, fungal `.snhskidx` plus 21-mer host Bloom filter), 16 threads, `--pipeline --output-format mtx --raw-matrix`
  (`singlet_nonhost.sbatch`). `nonhost/` is created before the run (see bug 1). Cells = STAR Solo filtered barcodes (mock 14,013; dpi2 11,567; PBMC 1,136).
* Kraken2 2.17.2 with the Langmead prebuilt viral-only database (k2_viral 2024-06-05, 0.65 GB), run on the same STAR-unmapped cDNA reads that Singlet screens
  (`star_Unmapped.out.mate1`), confidence 0.0 and 0.1; per-cell attribution from the barcode mate with 1-mismatch correction to the called cells; reads rolled up to species;
  molecules = distinct (cell, UMI) (`kraken_run.sbatch`, `kraken_percell.py`). A standard database was not used because of its size.
* Oracle: minimap2 `-x sr` of the same reads against NC_045512.2 (>= 60 nt aligned, >= 90 % identity), same per-cell logic (`oracle_run.sbatch`).

### Sample level

| Sample | Unmapped reads screened | Singlet EM species | Singlet reads assigned | Kraken2 (conf 0) top species, reads | Oracle SARS-CoV-2 reads |
|---|---|---|---|---|---|
| dpi2 | 2,334,542 | 1: NC_045512.2 SARS-CoV-2, abundance 0.78, HIGH | 554,070 | SARS-related CoV 553,261; BeAn 58058 virus 7,843; HERV-K 7,019 | 488,083 |
| mock | 1,886,673 | 0 | 0 | BeAn 58058 virus 8,834; HERV-K 5,149; SARS-related CoV 3,914 | 3,246 |
| PBMC | 2,575,933 | 0 | 0 | BeAn 58058 virus 23,850; HERV-K 10,833; phiX174 221 | 0 |

* Correct virus: both tools. Singlet names it as the index entry `NC_045512.2 Severe acute respiratory syndrome coronavirus 2 isolate Wuhan-Hu-1`.
* The Kraken2 viral-only database has no human sequence, so human-derived reads (HERV-K, a repeat-rich BeAn 58058 region) classify as viral: 1.4 % of PBMC unmapped reads.
  Singlet's host Bloom filter plus EM threshold suppresses them (no species called in PBMC or mock). Treat Kraken2's PBMC output as a database artefact of using a viral-only DB.
* Singlet's detection floor is `max(500, 0.5 % of screened reads)` reads (`src/pipeline/singlet.cpp:1549`, 9.4 k reads in mock, 11.7 k in dpi2). The 3.9 k real SARS-CoV-2 reads in the mock are below it,
  so a contamination or infection under 0.5 % of unmapped reads is not reported. Not titrated here.
* `nonhost_summary.json` kingdom fractions count raw k-mer hits before EM (`singlet.cpp:1646-1656`): PBMC reports `fungal_fraction` 0.110 and mock 0.080 with no species called
  (the unmapped reads include poly-A and low-complexity sequence; `nonhost_reads.fq` begins with `AAAAAAAA...`). Do not read these fractions as detections.

### Per cell: mock (14,013 cells) vs dpi2 (11,567 cells)

| Per-cell SARS-CoV-2 UMI from | AUROC mock vs dpi2 | cells >= 1 UMI, dpi2 | cells >= 1 UMI, mock | total UMI dpi2 / mock |
|---|---|---|---|---|
| Singlet (confirmed species) | 0.727 | 45.5 % | 0.07 % (10 cells) | 261,792 / 17 |
| Kraken2 conf 0 | 0.732 | 46.4 % | 0.06 % | 263,342 / 16 |
| Kraken2 conf 0.1 | 0.731 | 46.3 % | 0.06 % | 263,183 / 16 |
| minimap2 oracle | 0.708 | 41.6 % | 0.05 % | 233,341 / 13 |
| Singlet, all viral records | 0.695 | 71.8 % | 43.0 % | 272,984 / 8,832 |
| Kraken2 conf 0, all viral | 0.715 | 72.3 % | 39.5 % | 273,638 / 8,044 |

Within dpi2, per-cell SARS-CoV-2 UMI Spearman: Singlet vs Kraken2 0.985, Singlet vs oracle 0.930, Kraken2 vs oracle 0.935. Agreement with the oracle on "cell has >= 1 UMI":
Singlet 4,747 of 4,815 oracle-positive cells (recall 0.986), 5,264 called (precision 0.902); Kraken2 4,813 recalled (0.9996), 5,367 called (0.897). For >= 3 UMI: Singlet recall 0.985,
precision 0.886; Kraken2 recall 1.00, precision 0.884. The "extra" calls versus the oracle are cells where the 21-mer screens find viral reads shorter than the oracle's 60 nt requirement,
and are shared by both tools.

The all-viral rows show the noise floor: Singlet's per-cell table contains every sub-threshold hit, so 1,128 of 1,136 PBMC cells (99 %) have at least one "viral" UMI (Kraken2: 1,132). Per-cell
analysis must filter to species the sample-level EM confirmed (Singlet) or to a named taxon (Kraken2).

### Runtime and memory

| Step | Wall | Peak RSS |
|---|---|---|
| Singlet PBMC without screen (`t4_r1`, 4 threads) | 18:40 | 18.6 GB |
| Singlet PBMC with screen (`t4_nonhost_r1`, `_r2`, 4 threads) | 24:01, 23:16 | 55.3 GB |
| Screen cost (difference) | about 4.6 to 5.4 min | about +36.7 GB |
| Singlet mock / dpi2 (50 M pairs, 16 threads, full pipeline incl. screen) | 11:27 / 11:03 | 53.9 / 55.5 GB |
| Kraken2 viral DB, 2.6 M reads, 16 threads (includes DB load) | 2.3 s | 1.4 GB |
| minimap2 oracle, 2 M reads | 2 to 3 s | 0.4 GB |

The comparison is not like for like: Kraken2 ran on pre-extracted unmapped reads against a viral-only database, while Singlet loads about 25 GB of viral, bacterial and fungal indices
and the host filter. Kraken2 per-cell attribution in Python was not timed.

## Defects and what a fix needs

1. **Per-cell non-host table lost on a fresh output directory.** `NonHostCellMatrix::write` (`nonhost_cell_matrix.h:280-406`) is called at `singlet.cpp:1613` with `out_prefix + "/nonhost"`; that directory is
   created only afterwards at `singlet.cpp:1621`. The PBMC run `t4_nonhost_r1` logged `[pz_writer] Cannot open .../nonhost/nonhost_per_cell.1pz` and `nonhost/` held only the summary and EM table.
   Fix: move `create_directories` before line 1596. Workaround used here: `mkdir OUT/nonhost` before the run.
2. **Per-cell table names only EM-confirmed species.** `snames` and `kmap_cell` are filled from `em_result` only (`singlet.cpp:1604-1609`); every other hit appears with its internal numeric id as name
   and no kingdom label beyond the hit category. Fix: pass the full index name table, and add a column marking EM-confirmed species.
3. **Missing species FASTA for secondary alignment.** `reference/nonhost/viral_genomes/species_8877.fna` does not exist, so the confirmed SARS-CoV-2 gets no alignment or coverage output.
4. **5' prefix detector false positive** (`singlet.cpp:4538-4610`) described above. Fix: sample reads across the whole file (or require agreement across distinct barcodes) and add a flag to disable it.
5. **No junction filtering in `sj_counts`** (177,593 junctions, 134,242 singletons). Fix: apply STAR's `outSJfilter*` rules or a minimum count over the sample when writing the feature list.
6. **Mislabelled compressed outputs** (`export.h:1117-1139`, see above).

## Caveats

* Single replicate per sample; the first 50 M spots of each GSE166766 run, not the whole run (about 600 M spots each).
* Cell-level truth does not exist; the oracle is a read-level alignment, not an independent cell annotation. AUROC is bounded by the fraction of infected cells.
* Kraken2 used a viral-only database; a standard or PlusPF database would remove human-derived false positives but is 80 GB or more.
* Only SARS-CoV-2 was tested for correct-virus identification; bacterial and fungal paths were exercised only as false-positive checks. No sensitivity titration.
* Junction/PSI comparison is one dataset (PBMC 3' v3); STARsolo is run with 16 threads and a different EmptyDrops setting than Singlet's own cell call (1,227 vs 1,136 cells), so cell sets were taken from STARsolo.

## Reproduction

Scripts in `bench/modalities/sj_nonhost/`; work dir `/mnt/projects/debruinz_project/debruinz/singlet/modality-bench/sj_nonhost/`.

```
sbatch starsolo_sj_pbmc.sbatch                         # STARsolo Gene+SJ, raw reads
sbatch starsolo_sj_pbmc.sbatch clip11 '--clip5pNbases 11'
sbatch sj_compare.sbatch ; sbatch sj_compare.sbatch clip11     # -> results/pbmc1k_v3_sj_compare*.json
sbatch r2_composition.sbatch                           # R2 base composition, positions 1-14
sbatch setup_kraken.sbatch ; sbatch setup_sra.sbatch ; sbatch setup_mm2.sbatch
sbatch fetch_gse166766.sbatch SRR13711613 mock 50000000 ; sbatch fetch_gse166766.sbatch SRR13711615 dpi2 50000000
sbatch singlet_nonhost.sbatch mock ; sbatch singlet_nonhost.sbatch dpi2 ; sbatch pbmc_nonhost_rerun.sbatch
sbatch kraken_run.sbatch LABEL MATE1 MATE2 CELLS ; sbatch oracle_run.sbatch LABEL MATE1 MATE2 CELLS
sbatch nonhost_compare.sbatch                          # -> results/nonhost_compare.json
```
