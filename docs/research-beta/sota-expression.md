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

### 1. Spliced / unspliced / ambiguous (2026-10-31, binary = pin d9834b8a, intron_rules 389)

Fresh runs of `mc99cpu` are byte-identical to S5's gate runs of the pin (`cmp` on `intron_counts.mtx.gz`), so these are the pin's numbers.
Competitors: STARsolo 2.7.11b Velocyto, alevin-fry 0.18 USA (existing runs). Cells = STARsolo EmptyDrops_CR calls. Driver
`bench/expression_sota/usa_metrics.py`, outputs `bench/expression_sota/results/usa/`.

Primary: pooled per-gene unspliced r (Pearson on log1p of summed UMI). Bar = the two competitors' mutual r.

| dataset | Singlet~STARsolo | Singlet~alevin-fry | STARsolo~alevin-fry (bar) |
|---|---|---|---|
| pbmc1k_v3 | 0.987 | 0.963 | 0.969 |
| pbmc5p_v1 | 0.982 | 0.952 | 0.965 |
| neuron1k_v3 | 0.983 | 0.961 | 0.970 |
| nuclei2k_v2 | 0.963 | 0.949 | 0.959 |
| hgmm1k_v3 | 0.987 | 0.959 | 0.966 |

Pass (r >= bar - 0.02) on 5/5 vs STARsolo; vs alevin-fry 5/5 within 0.02 (worst nuclei2k_v2 -0.010, pbmc5p_v1 -0.013). The "0.83 vs
0.96" of the old result does not reproduce at the pooled-gene level on the pin. With the #80 rules off (`mc96off`, pbmc1k_v3 and
neuron1k_v3) the pooled r is the same to 3 decimals (0.988/0.964, 0.982/0.961): the rules do not move this metric.

Co-primary 1, unspliced share U/(S+A+U) (no competitor needed to read it; nuclei should be intron rich):

| dataset | Singlet | STARsolo Velocyto | alevin-fry |
|---|---|---|---|
| pbmc1k_v3 | 0.312 | 0.347 | 0.318 |
| pbmc5p_v1 | 0.094 | 0.101 | 0.117 |
| neuron1k_v3 | 0.321 | 0.359 | 0.350 |
| nuclei2k_v2 | 0.599 | 0.675 | 0.647 |
| hgmm1k_v3 | 0.210 | 0.222 | 0.212 |

Singlet's unspliced share is the lowest of the three on nuclei2k_v2 (8% below alevin-fry, 11% below STARsolo) and 10% below STARsolo
on neuron/pbmc3'. It matches alevin-fry on pbmc1k_v3 and hgmm. The tools disagree among themselves by up to 8%, and no truth
resolves which share is right; so this is reported, not claimed.

Co-primary 2, cross-species leak on hgmm1k_v3 (fraction of intron UMI in a pure-species cell that maps to the other species; exon in brackets):
Singlet 0.29% (0.38%), STARsolo 0.27% (0.31%), alevin-fry 0.43% (0.68%). Singlet is within 0.02 points of STARsolo and better than alevin-fry.

Per-cell median r of the unspliced vector is 0.80-0.96 (3'), 0.37-0.59 (5', depth) for every tool pair; Singlet~STARsolo is the
highest pair on 4/5 datasets.

Verdict: **at SOTA by agreement and by leak; the remaining gap is the per-cell unspliced total (S5 gate: 0.21 vs 0.027 ratio
distance on pbmc1k_v2)**, which is driven by STARsolo's Velocyto being the outlier (Singlet matches alevin-fry's share on pbmc1k_v3
and hgmm) and by a 8-11% lower share on nuclei. No engine change proposed yet; the ambiguous layer exists (`ambiguous_counts`), no
addition needed. A real-truth check of the share (metabolic labelling, or long reads from `bench/ground-truth`) is the open item.

### 2. Empty droplets and cell calling (2026-10-31, pin d9834b8a)

Truth that is not a cell caller (`bench/cellcall_truth`, branch `bench/cellcall-independent-truth`): Kang b2 ctrl donor genotypes
(per-barcode SNP likelihood vs the ambient donor mixture) and hgmm1k_v3 species purity; contested barcodes scored by a mixture
estimate with bootstrap CIs. Calls re-scored from the pin's own `cell_calls.tsv` (`bench/expression_sota/cellcall_rescore.py`;
the pin differs from the earlier calls on 98 of 22,605 Kang barcodes and 3 of 90,082 hgmm barcodes). Barcodes >= 50 UMI.

| dataset (truth) | variant | caller | P | R | F1 | F1 minus called_v2 (95% CI) |
|---|---|---|---|---|---|---|
| Kang b2 ctrl (genotype) | lenient | **called_v2 (pin)** | 0.9905 | 0.9905 | **0.9905** | |
| | | STARsolo EmptyDrops_CR | 0.9990 | 0.9136 | 0.9544 | [-0.040, -0.032] |
| | | CellBender | 0.9848 | 0.9843 | 0.9846 | [-0.008, -0.003] |
| | | Cell Ranger (authors) | 0.9989 | 0.8914 | 0.9421 | [-0.052, -0.044] |
| | | knee (`called`) | 0.9990 | 0.8805 | 0.9360 | [-0.058, -0.050] |
| Kang b2 ctrl | intact | **called_v2** | 0.9722 | 0.9920 | **0.9820** | |
| | | EmptyDrops_CR / CellBender / knee | | | 0.9548 / 0.9767 / 0.9371 | all CIs below 0 |
| hgmm1k_v3 (species) | intact | **called_v2** | 0.9585 | 0.8740 | **0.9143** | |
| | | EmptyDrops_CR / CellBender / Cell Ranger | | | 0.8973 / 0.8919 / 0.8985 | [-0.024,-0.011] / [-0.031,-0.016] / [-0.023,-0.009] |
| | | knee | 0.9875 | 0.7748 | 0.8683 | [-0.058, -0.034] |

Re-confirmed: **F1 0.991 on genotype truth** (0.9905 lenient, 0.982 with damaged barcodes excluded), above every competitor with a
CI that excludes zero; hgmm1k_v3 species truth likewise. The knee loses 0.05 F1 on both. hgmm1k_v3 "lenient" is not meaningful
(species-shifted debris counted as cells; recall ~0.57 for every caller).

**Which call Beta shows: `called_v2` (the pipeline default, `called_ed`), not the knee `called`.** It is the best F1 on both
independent truths, and the knee's recall is 0.88 (Kang) and 0.77 (hgmm). I did not examine GSM6470208 itself (4,791 vs 10,239): there is no
genotype or species truth for it, so the choice rests on the datasets above and the hashtag check in `docs/called-ed.md` (81% of
v2's additions on a super-loaded hashed PBMC library are stained single cells, against 3.8% singlet recall for the knee). Users who want
a stricter set filter on `fdr` and total UMI.

**nuclei2k_v2 is a real over-call, not only a circular-truth artefact.** The pin calls 3,045 barcodes; EmptyDrops_CR calls 2,202
(Gene) / 2,151 (GeneFull) and all of them are inside the v2 set (recall 1.00, precision 0.72 / 0.71, matching the gate's 0.77
against the other reference). The 843 extras (outside EDCR Gene) have median 387 UMI, 152 genes, mito 8.9% and intronic 42%;
the shared nuclei have 3,253 UMI, 1,762 genes, mito 2.4%, intronic 62%; ambient barcodes (20-100 UMI) have intronic 41%. Their
intronic fraction is ambient-like, not nuclei-like, and mito is 4-6x the nuclei. By contrast in GSE157827 brain nuclei the extras
looked like nuclei (intronic 0.60-0.64). A simple quantile "intact" filter (mito q99 / intronic q01 of the shared set) does not
separate them (828 of 843 pass), so the fix needs a per-barcode test, not a threshold: candidate = add an intronic-fraction
likelihood against the ambient profile to the EmptyDrops test for libraries whose called cells are intron rich. Not implemented:
no independent truth for nuclei exists in the cohort, so a change would be tuned on the same evidence it is judged by. **Open: truth for
nuclei (a hashed or genotype-pooled snRNA-seq set).**

(next: ambient)
