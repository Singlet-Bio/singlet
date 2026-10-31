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

**U and U+A share of all UMI (orchestrator follow-up):**

| dataset | Singlet U / U+A | STARsolo Velocyto U / U+A | alevin-fry U / U+A |
|---|---|---|---|
| pbmc1k_v3 | 0.312 / 0.324 | 0.347 / 0.446 | 0.318 / 0.787 |
| pbmc5p_v1 | 0.094 / 0.101 | 0.101 / 0.200 | 0.117 / 0.539 |
| neuron1k_v3 | 0.321 / 0.331 | 0.359 / 0.425 | 0.350 / 0.854 |
| nuclei2k_v2 | 0.599 / 0.628 | 0.675 / 0.741 | 0.647 / 0.967 |
| hgmm1k_v3 | 0.210 / 0.211 | 0.222 / 0.305 | 0.212 / 0.814 |

Singlet's ambiguous layer is tiny (0.2-3% of UMI; STARsolo 7-10%, alevin-fry 32-60%) and Singlet's U+A is still below the competitors' U on
nuclei2k_v2 (0.628 vs 0.647 / 0.675) and on neuron1k_v3 (0.331 vs 0.350 / 0.359). So the nuclei gap is **not** only a classification
choice (molecules filed under A instead of U); Singlet files these molecules as spliced (S) under its collapsed-gene rules, e.g. reads
that touch an exon and an intron. Whether that is right is decided by long-read truth (next).

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

| Gaublomme 2019 mouse nuclei (hashtag truth; positive = hashtag UMI above the empty-droplet p99 and one tag >= 60%) | | knee `called` | | | | precision 0.978 (2,691 called) |
| | | **called_v2** (additions are 828 of its 3,519) | | | | precision **0.90** (additions 64.9% positive, those with >= 400 UMI 87%); caveat: the antibody stains nuclear pores, so debris and ambient nuclear fragments can be tagged too; recall not measured (no hashtag-negative real nuclei exist) |

Re-confirmed: **F1 0.991 on genotype truth** (0.9905 lenient, 0.982 with damaged barcodes excluded), above every competitor with a
CI that excludes zero; hgmm1k_v3 species truth likewise. The knee loses 0.05 F1 on both. hgmm1k_v3 "lenient" is not meaningful
(species-shifted debris counted as cells; recall ~0.57 for every caller).

**Which call Beta shows: `called_v2` (the pipeline default, `called_ed`), not the knee `called`.** It is the best F1 on both
independent truths, and the knee's recall is 0.88 (Kang) and 0.77 (hgmm). I did not examine GSM6470208 itself (4,791 vs 10,239): there is no
genotype or species truth for it, so the choice rests on the datasets above and the hashtag check in `docs/called-ed.md` (81% of
v2's additions on a super-loaded hashed PBMC library are stained single cells, against 3.8% singlet recall for the knee). Users who want
a stricter set filter on `fdr` and total UMI.

**(Superseded by the hashtag result below: the over-call is smaller than this paragraph suggests.) nuclei2k_v2 over-call.** The pin calls 3,045 barcodes; EmptyDrops_CR calls 2,202
(Gene) / 2,151 (GeneFull) and all of them are inside the v2 set (recall 1.00, precision 0.72 / 0.71, matching the gate's 0.77
against the other reference). The 843 extras (outside EDCR Gene) have median 387 UMI, 152 genes, mito 8.9% and intronic 42%;
the shared nuclei have 3,253 UMI, 1,762 genes, mito 2.4%, intronic 62%; ambient barcodes (20-100 UMI) have intronic 41%. Their
intronic fraction is ambient-like, not nuclei-like, and mito is 4-6x the nuclei. By contrast in GSE157827 brain nuclei the extras
looked like nuclei (intronic 0.60-0.64). A simple quantile "intact" filter (mito q99 / intronic q01 of the shared set) does not
separate them (828 of 843 pass), so the fix needs a per-barcode test, not a threshold: candidate = add an intronic-fraction
likelihood against the ambient profile to the EmptyDrops test for libraries whose called cells are intron rich. Not implemented:
no independent truth for nuclei exists in the cohort, so a change would be tuned on the same evidence it is judged by. **Open: truth for
nuclei (a hashed or genotype-pooled snRNA-seq set).**

**Nuclei truth found, and it weakens the over-call claim (2026-10-31).** Gaublomme et al. 2019 (Nat Commun 10:2907) deposit a mouse
cortex nucleus-hashing library: RNA SRR8703773 (10x 3' v2, 26.9 M reads) and its antibody-hashtag library SRR8703774 (eight hashtags found by
k-mer tally; the human and species-mixing runs are controlled access, so there is no human/mouse nuclei barnyard). Fetched in a Clipper job,
Singlet (build 92095e29, GRCm39) calls 2,691 by the knee and 3,519 by called_v2 (+828: the same +30% as nuclei2k_v2's +633/+843). A barcode is hashtag-positive when
its hashtag UMI exceeds the 99th percentile of empty droplets (66 UMI) and one tag holds >= 60%.

| set | n | hashtag-positive |
|---|---|---|
| knee-called | 2,691 | 97.8% |
| called_v2 additions | 828 | **64.9%** |
| of which RNA UMI 20-100 / 100-200 / 200-400 / 400+ | 35 / 124 / 257 / 412 | 34% / 21% / 55% / **87%** |
| uncalled barcodes at 20-50 / 50-100 / 100-200 / 200-400 RNA UMI | 22,254 / 279 / 41 / 19 | 0.9% / 11% / 51% / 68% |

So about two thirds of the additions are hashtag-confirmed nuclei, and the half with >= 400 UMI is 87% confirmed; the additions below 200 UMI
(159 barcodes) are no more hashtag-positive than uncalled barcodes of the same RNA depth. The hashtag-based precision of called_v2 is about 0.90 (knee
0.98); EmptyDrops_CR's smaller set on nuclei2k_v2 is therefore **not** established as the truth, and "precision 0.72 against EDCR" overstates the error.
Caveat: the antibody stains nuclear pores, so debris and ambient nuclear fragments can carry tags too; positivity does not prove an intact nucleus.

The pre-registered intronic-composition test (frozen before scoring: binomial test of intronic fraction against the ambient fraction, active when
the knee cells' median intronic fraction exceeds ambient by >= 0.15, BH 0.01, additions only) does not earn a PR. On this library it removes 329 of 828
additions of which 48% are hashtag-positive (kept: 76%), a weak split; precision rises 0.90 to 0.94 while recall of the positives falls 5%, so F1 is
unchanged (0.947 vs 0.947). It is inactive on pbmc1k_v3 and hgmm1k_v3 (intronic margins 0.10 and 0.13) and active on nuclei2k_v2 (removes 326 of 633).
A UMI floor on the additions (drop < 200 UMI) would give F1 0.96, but that was found after looking at this library and is not validated.

**Pre-registered hypothesis (not shipped; orchestrator decision 2026-10-31):** in an intron-rich library (median intronic fraction of the knee-called
barcodes exceeds the ambient pool's by >= 0.15), a barcode added by called_v2 beyond the knee is kept only if it has >= 200 UMI. Found after looking at the
Gaublomme library, where it gives F1 0.96 against 0.947; the expectation is F1 ~0.96 on a new set. It is to be validated on a second hashed or
genotype-pooled nuclei set with raw reads, if one turns up, before any PR. Until then called_v2 ships unchanged for nuclei, `fdr` is available for filtering.

**Nuclei ambient: a real, large bug (2026-10-31).** In the Gaublomme library, Xist is expressed only in female nuclei, so Xist UMI in male-hash nuclei is ambient
(four of the eight hashtags are female: median Xist share 0.22-0.25% of UMI; four are male: ~0.003%). With the pool's Xist share a_X = 0.151% (uncalled barcodes with
1-100 UMI), the Xist-based contamination of the 1,344 male nuclei is Xist share / a_X = **<= 2.3%** (UMI-weighted 2.4%); this is an upper bound, because male nuclei have low native Xist (Zach), so the true value is lower. Singlet (92095e29) reports per-cell `rho_total`
**27.6%** mean (12x), exon-basis `rho` 37% (16x), and removes 25% of the nuclei's UMIs (`bench/expression_sota/nh_xist.py`, `results/nuclei/xist_ambient.json`).
nuclei2k_v2 shows the same signature (mean rho 29%, removed 21% of UMI). The ambient pool of a nuclei library is debris with a nuclear profile (its Xist share
equals that of the female nuclei themselves), so "the part of a nucleus that looks like the pool" is most of the nucleus; the estimator cannot separate soup from native
expression there. Second, sex-independent check (`nuc_markers.py`, `results/nuclei/markers_*.json`): genes exclusive to one brain cell class appear in nuclei of
another class only through soup (or low native expression, so these are upper bounds). Oligodendrocyte markers (Plp1 Mbp Mobp Mog Mag Cldn11) in the 2,597 neuronal
nuclei of the Gaublomme library: <= **9.2%** of their UMI is soup (marker share in neurons / marker share in the pool); microglia markers in neurons <= 25% (34 UMI only);
astrocyte markers in neurons <= 69% (Slc1a2 and Aqp4 are expressed natively in neurons, so this one is uninformative). Singlet reports 27.6% (mean rho_total) there,
at least 3x the oligodendrocyte bound and 12x the Xist bound. On nuclei2k_v2 the same check gives <= 28% (oligodendrocyte in neurons), which does **not** exclude Singlet's 23%:
there is no sex truth in that library, so its over-estimate is shown by the Gaublomme library only. Candidate fix, not yet built: detect the case (pool profile close to the called cells' profile) and report rho as NA with the corrected matrix equal to the
raw counts, instead of a 12x wrong number; a criterion needs the pool-to-cell similarity measured on the nuclei and cell libraries we have.

### 3. Ambient RNA (R10c)### 3. Ambient RNA (R10c) (2026-10-31, pin d9834b8a)

Truth 1, real tissue: **Kang b2 ctrl donor genotypes** (12,137 confident genotype singlets). A read carrying an allele the cell's donor does
not have comes from another donor's RNA through the ambient pool; per-cell truth rho = foreign reads / expected foreign reads, error-rate
corrected (error rate 0.12%, pooled truth **1.48%**). Method: `kang_ambient_artifacts.py` (bench/public-ambient-doublets, from PR #43);
tools were run earlier on the same raw matrix (SoupX auto, DecontX default and with 20k-droplet background, CellBender 0.3.2 default call;
the CellBender call with `--expected-cells` did not produce output). Scoring: `bench/expression_sota/ambient_kang_eval.py`, results
`bench/expression_sota/results/ambient/kang_ctrl_pin.json`. Implied contamination = 1 - UMI after / UMI before.

| tool | mean | ratio to truth | abs log2 ratio | per-cell Spearman | worst cell-type ratio |
|---|---|---|---|---|---|
| **Singlet `rho` (estimate)** | 2.01% | 1.36 | 0.44 | 0.093 | NK 2.6, CD8 2.3 |
| **Singlet corrected-counts removal** | 1.68% | **1.13** | **0.17** | 0.081 | DC 2.2, NK 2.2 |
| SoupX | 0.95% | 0.64 | 0.64 | 0.032 | Megakaryocyte 0.2 |
| DecontX default | 9.2% | 6.22 | 2.64 | 0.133 | DC 23 |
| DecontX with background | 1.60% | **1.07** | **0.10** | 0.020 | CD8 3.7 |
| CellBender (default call) | 1.07% | 0.72 | 0.47 | -0.085 | DC 0.2, NK 2.6 |

On the mean, the pin's removal is 1.13x truth: second only to DecontX with background (1.07), clearly ahead of SoupX (0.64), CellBender
(0.72) and DecontX default (6.2x). The reported `rho` is 1.36x (it over-estimates more than the amount actually removed). Every tool's
per-cell ordering is weak against this truth (|Spearman| <= 0.13; per-cell truth is Poisson noisy), so per-cell Spearman does not
discriminate on tissue; Singlet's 0.09 is second to DecontX default (which is 6x off on the mean). Rare clusters are over-corrected by
2x for Singlet (DC, NK), as on the earlier Kang lanes (`fix/ambient-calibration`: 8-10x on the old estimator, 7.3-8.4x with the unmerged
cluster shrinkage, held-out lanes); the pin is far better than that here because its corrected removal is bounded by the formula.

Truth 2, species mixing (hgmm1k_v3, held-out hgmm1k_v2), from S5's gate at the pin: 9 of 10 gating metrics pass; the failing one is
the contamination-estimate bias on hgmm1k_v3 (|log2| 0.28 vs SoupX 0.038).

**Which ambient number is "the estimate" (2026-10-31, pin; `bench/expression_sota/ambient_hgmm_eval.py`).** The run writes several summaries that disagree
by a factor of two, and the headline `estimated_contamination_fraction` is the worst of them:

| library | truth | `estimated_contamination_fraction` (= mean of per-cell `rho`) | median `rho` | UMI-weighted `rho` | removed / observed UMI |
|---|---|---|---|---|---|
| hgmm1k_v3 (species; 998 pure cells; cell-mean truth 1.14%, UMI-weighted ~0.92%) | 1.14% | 2.11% (1.85x) | 0.92% | 0.94% | 0.94% |
| Kang b2 ctrl (genotype; pooled truth on 12,137 confident singlets 1.48%) | 1.48% | 4.71% (3.2x) | 1.51% | 2.82% | 2.82% |
| pbmc1k_v3 | n/a | 2.71% | 1.18% | 2.29% | 2.29% |
| nuclei2k_v2 | n/a | 29.4% | 27.1% | 21.2% | 21.2% |

Per cell on hgmm1k_v3 (998 pure cells): the exon-basis `rho` has ratio 1.56 and Spearman 0.79; `rho_total` (ambient share of all UMIs, already in the TSV as its last
column) ratio 1.17, Spearman 0.79; the realised removal fraction ratio 1.08, Spearman 0.82. On the Kang confident singlets the realised removal is 1.13x truth
and the reported `rho` 1.36x. So the per-cell estimator is reasonable on cells that are clearly cells; the large errors are in the headline mean, which
the low-depth and outlier cells inflate (hgmm 45 low-depth cells, Kang 16.5k called cells of which 12.1k are confident), and in the exon-basis `rho`.
No code change that is both small and supported: switching the headline to the UMI-weighted removal fixes hgmm (1.02x) but leaves Kang at 1.9x
because the called set includes non-singlet barcodes the truth excludes, and nuclei2k_v2 at 21% has no truth (a nuclei library with 27% median `rho`
looks wrong, but nothing here shows it). **No ambient PR from me**: the real fix is the v2 estimator on `fix/ambient-calibration`, which its own bench still has at 7-8x on
held-out Kang lanes. Recommendation for the site: show `rho_total` per cell, not the mean `rho`.

Verdict: **at the mean-calibration level SOTA on tissue (within 0.07 of the best tool) but not on species mixing (0.28 vs 0.04)**; per-cell
ordering is not a differentiator on this truth. Candidate engine change: calibrate the reported `rho` to the removal it applies and shrink
rare-cluster estimates (`fix/ambient-calibration` has the code, unmerged). Not done: needs a decision on whether the 0.28 hgmm bias justifies
a PR; the cluster shrinkage did not fix the rare-cluster ratio on held-out lanes in its own bench.

### 4. Expression doublets (2026-10-31; analysis of the #43 latent-qc results, no new compute)

Source: `bench/latent_qc/doublets/results/auprc_wide.csv` (study/latent-qc, PR #43; copied to `bench/expression_sota/results/doublets/`). 17
real-label datasets (genotype demuxlet, hashtag, MULTI-seq, species), leave-one-study-out training for the combiners, no synthetic sets.
Paired bootstrap of the AUPRC difference vs scDblFinder (dataset resampling; study-clustered resampling in brackets):

| method | mean diff vs scDblFinder | 95% CI | study-clustered CI | wins |
|---|---|---|---|---|
| production `detect_doublets` | -0.058 | [-0.092, -0.027] | [-0.113, -0.020] | 1/17 |
| `combo_prod_atlas` (no R) | -0.002 | [-0.029, +0.023] | [-0.040, +0.016] | 9/17 |
| `combo_all` (needs R + Solo) | +0.013 | [-0.002, +0.026] | [-0.010, +0.021] | 12/17 |

By kind of label, mean diff vs scDblFinder [95% CI over datasets], wins:

| truth | n | production | combo_prod_atlas | combo_all | scDblFinder (mean AUPRC) |
|---|---|---|---|---|---|
| genotype (Kang demuxlet lanes) | 6 | 0.482 | **0.580**, +0.037 [+0.017,+0.051], 5/6 | 0.574, +0.030 [+0.019,+0.039], 6/6 | 0.544 |
| hashtag / MULTI-seq | 8 | 0.478 | 0.491, -0.012 [-0.057,+0.024], 3/8 | 0.514, +0.011 [-0.011,+0.031], 5/8 | 0.503 |
| species mixture (hgmm) | 2 | 0.808 | 0.923, -0.076 [-0.105,-0.046], 0/2 | 0.966, -0.033 [-0.045,-0.021], 0/2 | 0.998 |
| 293T demuxlet | 1 | 0.190 | 0.214 | 0.227 | 0.213 |

Reading. Overall the combiner is a statistical tie with scDblFinder; the production score is clearly worse. Caveats: (a) the six
genotype lanes are one study (shared donors and pool), so +0.037 is one effective sample, not six; (b) the two species datasets are
easy heterotypic doublets that scDblFinder scores at ~1.0 and every other method loses, which is not the single-donor situation this
feature is for; (c) the combiner is a Python model, not in the binary, and its atlas features need `singlet.celltype_nn`.
**Against the orchestrator's rule (ship only if no worse than scDblFinder at the CI bound): `combo_prod_atlas` fails** (lower bound -0.029
dataset-level, -0.040 study-clustered); `combo_all` is closest (-0.002 / -0.010) but needs R. A "clear CI win" is not available from the
current feature set: the best open method gains +0.013 and its CI includes 0.

**First attempt at a native caller (2026-10-31): did not beat scDblFinder.** `bench/expression_sota/doublets/sdfl.py` is an R-free
reimplementation of the scDblFinder idea (cluster-pair artificial doublets, kNN features at four k, gradient-boosted
artificial-vs-observed classifier, two rounds, out-of-fold scores), unsupervised so no label leakage. Same 17 datasets and labels:

| method | mean AUPRC | vs scDblFinder (95% CI over datasets) | wins | Kang 6 lanes | hgmm 2 |
|---|---|---|---|---|---|
| scDblFinder | 0.559 | | | 0.544 | 0.998 |
| production `detect_doublets` | 0.501 | -0.058 | 1/17 | 0.482 | 0.808 |
| `sdfl` heterotypic pairs | 0.510 | -0.049 [-0.087,-0.015] | 3/17 | 0.527 | 0.780 |
| `sdfl` with 1/3 homotypic pairs | 0.489 | -0.069 [-0.114,-0.032] | 1/17 | 0.512 | 0.735 |
| `combo_prod_atlas` (needs the atlas model) | 0.557 | -0.001 | 9/17 | 0.580 | 0.923 |

It is a slight gain over production (+0.009) and clearly below scDblFinder, with the largest loss on the easy species mixtures (0.78 vs
1.00). What scDblFinder does that this does not: doublet generation tied to cluster pairs it then scores per cluster pair, many more features
(including per-cell kNN ratios at several k over a larger PC space and cell-type-specific doublet enrichment), and an iterative loop that
retrains on the identified doublets. These are the next things to add; I have not tuned on the evaluation sets.

**hgmm diagnosis (2026-10-31): coarse clusters do not close the gap.** Simulating heterotypic pairs from 3 or 8 k-means clusters (the orchestrator's
suggestion; the parent total is already a feature) gives hm-6k / hm-12k AUPRC 0.89 / 0.87 (k = 3) and 0.80 / 0.81 (k = 8) against scDblFinder 1.00 / 1.00 and the default
(sqrt(N)/4 clusters) 0.81 / 0.76, and loses elsewhere (mean over 17: 0.471 for k = 3, 0.508 for k = 8, 0.510 default; scDblFinder 0.559). The hgmm matrices of Xi & Li 2021 hold only human genes
(ENSG, 15 k), so a human+mouse doublet is a human cell with ortholog-mapped mouse reads; the right simulation is not what scDblFinder's clustering step is
doing, and its near-perfect score there is unexplained. Stopped here; the sdfl feature stays a stacker input, not a standalone caller.

**sdfl as a stacker feature (leave-one-study-out, same recipe as #43; `bench/expression_sota/doublets/stack.py`, per dataset
`results/doublets/stack.csv`).** The replication of `combo_prod_atlas` gives 0.5573 (reported 0.557). Adding sdfl:

| stacker | mean AUPRC | paired gain over combo_prod_atlas (dataset CI / study-clustered CI) | wins | vs scDblFinder (0.5587) |
|---|---|---|---|---|
| combo_prod_atlas | 0.5573 | | | -0.001 [-0.032,+0.025] |
| + sdfl | 0.5684 | **+0.0111** [+0.0070,+0.0150] / [+0.0066,+0.0133] | 16/17 | +0.010 [-0.020,+0.037], study-clustered [-0.033,+0.025] |
| + sdfl + sdfl (homotypic variant) | 0.5707 | **+0.0134** [+0.0095,+0.0172] / [+0.0030,+0.0151] | 16/17 | +0.012 [-0.019,+0.040] |
| production + umi + genes + sdfl (no atlas model) | 0.537 | | | |

The sdfl feature is a real, small, consistent gain over the existing combiner (16 of 17 datasets, CIs exclude 0 at dataset and study
level). Against scDblFinder the stacker is now nominally ahead (+0.010 to +0.012) but the CI still contains 0, so this is a tie, not the
"clear CI win". **The hgmm gap is not closed:** hm-6k/hm-12k reach 0.95/0.91 against scDblFinder's 1.00/1.00 (-0.07 on the pair), so the
artificial-doublet simulation is still missing the signal that makes species-mixture doublets trivial for scDblFinder. Without an
atlas model the stacker is worse than scDblFinder (0.537). Not yet diagnosed.

Not done: a better caller. What the data say about where a gain can come from: the win is on genotype-labelled PBMC (the tissue-like case);
hashtag cell lines and nuclei are ties; the combiners lose where one library is two easily separable populations. Candidate work (needs
Clipper compute and a decision on scope): native (non-R) artificial-doublet kNN score as an extra feature, and a stacker trained on the
genotype-labelled lanes of a second, independent study (the batch-1 and stim lanes are the same study) so the gain can be shown on a held-out
study. Until then, ship nothing new for doublets: genotype doublets for pools with >= 2 donors (F1 0.77-0.87, S10), production score
otherwise.

### 5-7. TE, chimeric junctions, reads per UMI: what exists and what each needs (2026-10-31)

State of the engine at d9834b8a (read from the source, nothing run yet):

* **TE.** `--te-classify on` (needs `--cascade on` and `--te-db te_sketch.bin`) is a 21-mer sketch classifier at **Dfam family level** with
  an EM, run on reads the cascade routes to L2 (`te_classifier.h`, `te_counter.h`, output `te_counts.1pz`). There is no genome-alignment TE
  counting, no subfamily or locus level, and no multimapper EM over alignments. STAR's multimapper limit decides whether young L1/Alu reads
  reach any counter at all.
  Needed for the requested hierarchy: a TE annotation (RepeatMasker/Dfam hg38 BED with class/family/subfamily/locus) and a **separate counting
  pass over multimapping alignments** (NH > 1) that never touches the gene counts, with locus-level counts only for uniquely mappable
  loci (flag from a mappability track). Gate: gene matrices byte-identical with the TE pass on and off, CPU vs GPU byte-identical.
* **Chimeric junctions.** STAR's chimeric code is compiled in (`include/singlet/star/Chimeric*.h`), but no per-cell chimeric table is
  written. Needed: switch on `--chimSegmentMin` style detection, write (barcode, donor, acceptor, supporting UMIs), annotate against
  Mitelman/ChimerDB/CTAT. Gate: main counts unchanged with detection on (chimeric detection can change which reads STAR reports as
  linear, so this must be measured, not assumed).
* **Reads per UMI.** Per-cell reads and unique UMIs already exist (`saturation.h`: `reads_per_umi` per barcode, `saturation_metrics.tsv`).
  Missing: a reads-count matrix (cell x gene, same pz2 layout as the UMI matrix) and the per-cell histogram of reads per UMI. This is a
  counting-stage change: the dedup step must carry the read multiplicity of each (cell, gene, UMI) it collapses.
* **Truth.** `bench/ground-truth` and `$WORK/ground-truth/raw` (molecule-matched long reads) were named in the request but are not on
  Clipper or in the repo. TE locus truth, and the USA share question, are blocked on that data.

Each of the three is a CPU change plus a GPU mirror (S8) and a byte-identity gate; they are separate engine PRs, not benchmarking tasks.

(next: VDJ, non-host, editing, HTO)
