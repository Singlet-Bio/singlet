# Expression-side SOTA: pre-registered benchmark plan and results

Owner: expression-sota session. Engine under test: `prod/campaign` d9834b8a (intron_rules = 389 default). Written 2026-10-30.
Rules: real data with real ground truth only; competitors are open (no 10x software, no MiXCR); one primary metric per modality,
fixed here **before** any new run. A modality "reaches SOTA" when its primary metric is within the stated margin of, or better than,
the best open competitor on held-out data. Results are appended below each plan row as they land; failures are reported as failures.

Existing infrastructure reused: `bench/public/` (contract 1.1.0, harness, `capabilities/spliced_unspliced.py` on
`bench/public-counts`, `ambient-rna` on `bench/public-ambient-doublets`), `docs/called-ed.md`, `docs/modality-benchmarks/*`,
latent-doublet study (#43, `docs/beta-doublet-decision.md` on `docs/beta-doublet-decision`).

## Plan (status: v1, datasets are verified on Clipper/ENA before compute)

| # | Modality | Datasets (truth) | Best open competitor | Primary metric (pre-registered) | Pass |
|---|---|---|---|---|---|
| 1 | Spliced / unspliced / ambiguous | pbmc1k_v3, hgmm_1k_v3 (barnyard), one 10x snRNA nuclei set, Kang lane; truth-like: cross-species intron leak (hgmm), nuclei intron fraction | alevin-fry USA, STARsolo Velocyto | Per-gene Pearson r of pooled **unspliced** UMI vs each competitor, median over datasets (old: 0.83 vs competitors' mutual 0.96); secondary: share of the 42 contract checks passing, hgmm cross-species intron rate | r >= (competitor-vs-competitor r) - 0.02 |
| 2 | Empty droplets / cell calling | Kang 8-donor lane (genotype truth), hgmm (species truth), GSE163793, GSM6470208 | EmptyDrops (DropletUtils), CellBender | F1 of called cells vs genotype/species-confirmed cells (old: 0.991); settle `called` vs `called_ed` on GSM6470208 by the same F1 | F1 >= best competitor - 0.005; the call Beta shows = higher F1 |
| 3 | Ambient (R10c) | hgmm_1k_v3 (rho_true from mouse reads in human cells), Kang genotype lane (true 2.48%) | SoupX, DecontX, CellBender | Absolute relative error of the mean contamination fraction (old: 0.27x on hgmm, 3.43 vs 2.48 on Kang) and per-cell Spearman vs truth; plus fraction of ambient UMIs removed | rel. err <= best competitor; Spearman >= best - 0.05 |
| 4 | Expression doublets | the 17 labelled datasets of #43 (genotype / hashtag / MULTI-seq / species), leave-one-study-out | scDblFinder, Scrublet, DoubletFinder | Mean AUPRC over datasets with paired bootstrap CI (old: 0.501 prod, combo_prod_atlas 0.557, scDblFinder 0.559) | combo >= scDblFinder with the CI excluding a loss > 0.01 |
| 5 | VDJ | 10x public 5' PBMC/T-cell VDJ-T + VDJ-B libraries, paired-chain truth from the library's own paired alpha/beta (bulk-confirmed clonotype tables in published sets, e.g. donor-matched TCR-seq) | TRUST4 | Recall of the published/paired-chain clonotype per cell, and precision of productive CDR3 calls (chain-pair F1) | F1 >= TRUST4 - 0.02 |
| 6 | Non-host transcripts | SARS-CoV-2 infected scRNA-seq (public SRA, known viral genes/sgRNAs), one HIV or other infection set; spiked truth: hgmm for the species level | Kraken2 (read level), viral-aware STARsolo (genome + viral contigs) | Per-cell viral gene-level UMI r vs the viral-augmented STARsolo count, and read-level sensitivity/precision of the pathogen call vs Kraken2 | r >= 0.95 and precision >= Kraken2 |
| 7a | TE counts | 10x PBMC (and a hESC or K562 set that scTE used) | scTE, SoloTE, IRescue | Per-subfamily pooled r vs best competitor; per-cell r | r >= 0.9 vs the best competitor, no systematic family bias |
| 7b | Chimeric junctions | K562 scRNA-seq (BCR-ABL1) or a fusion-bearing tumor line | STAR-Fusion / the published fusion call | Recovery of the known fusion in the expected fraction of cells; false chimeric rate in a fusion-free line | known fusion found, FP/cell reported |
| 7c | Reads-per-UMI, saturation | pbmc1k_v3, downsampling series | Preseq-style saturation curve; STARsolo `Summary.csv` sequencing saturation | Saturation error at 25/50/75% down-sampling vs actual | abs error < 0.02 |
| 7d | RNA editing panel | PBMC and a set with a published Alu editing index | RNAEditingIndexer / published AEI values | Per-sample AEI correlation | r >= 0.9 vs published |
| 8 | HTO | an independent hashed set not used to pick the caller (Kang-genotype-labelled hashed lane, or 10x TotalSeq hashing set with a genotype key), caller frozen from earlier runs | Seurat HTODemux, demuxEM, GMM-Demux | Macro F1 of singlet/doublet/negative vs genotype key (old: 0.92 vs 0.94, caller selected on the same data) | F1 >= best - 0.01 on the held-out set |

## Amendments (orchestrator, 2026-10-30)

1. USA: agreement with competitors is not truth. Co-primary checks: nuclei intron fraction and hgmm cross-species intron leak; report all three.
2. Doublets: a tie with scDblFinder within the CI counts as SOTA. Ship the combiner only if no worse than scDblFinder at the CI bound; report per-dataset wins.
3. Kang and hgmm data are reused in place on `$WORK`.

## Scope widening (Zach via orchestrator, 2026-10-30)

Order: USA -> cells/ambient -> doublets -> TE -> fusions -> reads-per-UMI -> VDJ, non-host, editing, HTO.

* **TE:** class / family / subfamily / locus from Dfam/RepeatMasker hg38. Subfamily counts by multimapper EM (young L1HS/AluY/SVA
  distributed, not dropped); locus-level only where uniquely mappable, with a mappability flag. Competitors scTE (subfamily),
  SoloTE and Stellarscope (locus), IRescue. Truth: molecule-matched long reads (`bench/ground-truth`, PBMC-E, LongBench under
  `$W/ground-truth/raw`) at locus level, plus biology checks (HERVH in ESCs).
* **Fusions:** no fusion contigs in the primary genome (breaks the K5 gate). STAR chimeric detection on, junctions annotated against
  open catalogues (Mitelman, ChimerDB, CTAT; not COSMIC). Optional second pass on unmapped reads against a breakpoint index built
  from open catalogues, with a gate showing main counts unchanged. Truth: K562 BCR-ABL1 and CCLE-documented lines; per-cell
  sensitivity and false positives on a fusion-free sample (PBMC).
* **Reads per UMI:** a cell x gene reads-count matrix (same layout and pz2 codec as the UMI matrix) plus a per-cell reads-per-UMI
  histogram; report bytes added per sample.
* **Doublets:** improve the caller, aiming for a clear CI win on AUPRC over scDblFinder, on genotype-labelled pools.

## Engine-change rules

CPU reference first; GPU engine mirrors via S8; gate = CPU vs `--engine gpu` byte-identical and ctest green; one draft PR per
modality fix against `prod/campaign`, handed to S1.

## Results

(none yet)
