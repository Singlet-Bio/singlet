# Latent-embedding and atlas-trained doublet and ambient calling

Study, 2026-10-25. Question from Zach: can neural-network / latent-embedding models, trained on data the atlas already holds,
call doublets ("several cell types in one droplet") and ambient RNA better than Singlet's per-sample methods? Code and result
tables: `bench/latent_qc/` (`doublets/`, `ambient_truth/`, `ambient_atlas/`, `ambient_kang/`). All compute ran on Clipper
(`bench/latent_qc/common.sh`); nothing was written to R2 or D1. Every number reported here is measured on **real data with an
experimental ground truth**: genotype, hashtag, MULTI-seq and species doublet labels; cross-species and cross-donor ambient reads.
Simulated doublets appear only inside methods as training signal (Scrublet, Solo, scDblFinder and the production scorer all
work that way); no reported score is on simulated data.

## Recommendation

| | Adopt? | Numbers (real truth) | Cost per 10k cells | Where |
|---|---|---|---|---|
| Atlas cell type encoder as a doublet feature (kNN fraction of simulated doublets in the embedding of `singlet.celltype_nn`) | **yes, as a feature inside a combiner** | alone: equal to production (AUROC +0.008, AUPRC 0.000 mean over 17 datasets); inside the combiner it adds +0.017 median AUPRC, better on 16/17 datasets (Wilcoxon p < 1e-4) | 6-9 s on 8 CPU threads (embedding real + 2x simulated cells, kNN) | post-hoc pass on stored counts, next to the cell type sidecar inference (same network, same forward pass) |
| Cross-study doublet combiner (gradient boosting on production score + atlas features, trained on real experimental labels of other studies) | **yes** | AUPRC +0.060 median over production (15/17 datasets better, p = 0.0004); mean AUPRC 0.557 vs 0.501; Kang 2.1 on Singlet counts 0.935 / 0.698 vs 0.919 / 0.632 | about 10 s (2 s production score + 6-9 s atlas features) | post-hoc on stored counts; model trained once on the 16 Xi & Li datasets + Kang |
| scDblFinder | not needed if the combiner is adopted | equal to the combiner (AUPRC difference -0.001, p = 0.64) | 44 s, single-threaded R | would add an R stack to the campaign |
| Solo-style per-sample VAE | **no** | worse than production on 16/17 datasets (AUPRC -0.099 median) | 167 s of GPU time | |
| "Two cell types in one droplet" by NNLS of atlas or sample centroids | **no** | AUROC 0.62 (atlas centroids) / 0.70 (sample clusters), far below every simulation-based score | 1-23 s | |
| Genotype doublets (already in production, `donor_assignments.tsv prob_doublet`) | **keep, and prefer whenever a pool has >= 2 donors** | Kang 2.1: AUROC 0.969 / AUPRC 0.897; recall 0.89 on doublets that look like one cell type, where expression methods reach 0.47-0.59 | in the run | production binary |
| Atlas prior over ambient profiles | **no** (optional fallback for pools < 3,000 UMIs, ~3% of samples) | the per-sample empirical profile matches the prior at the atlas' median pool (268k UMIs); the prior wins only below ~3k UMIs | negligible | post-hoc |
| Latent-kNN per-cell ambient estimator (scVI / PCA neighbourhood "native expression") | **no** | barnyard, 5 held-out libraries: Spearman 0.918 vs production 0.935; ratio to truth 0.86 vs 0.98 | 270 s single-threaded for 11k cells, 25x production | |
| Production ambient estimator (`correct_ambient_gene_level_split`) | keep; fix two issues | barnyard held-out Spearman 0.935, ratio 0.98; real tissue (Kang genotype truth) mean 3.4% vs truth 2.5%, but per-cell spread overstated about 10x and estimates depend on the cell set | 1-2 s | production binary |

In short: **latent models do not beat Singlet's per-sample methods on their own; the atlas cell type network is worth adding as one
feature of a small cross-study doublet combiner (+0.06 AUPRC, scDblFinder-level accuracy without R), and genotype doublets remain
far better than any expression method wherever donors are pooled.** For ambient RNA the production estimator is already at the
truth's noise ceiling on barnyard libraries; the open problems are its per-cell calibration in real tissue and its sensitivity to the
cell set (section 3.4), not a missing atlas prior.

## 1. Data

**Doublet truth (real experimental labels).** The 16 real datasets of Xi & Li 2021 (*Cell Systems*, Zenodo 4062232; public count
matrices with doublet labels from the original experiments): demuxlet genotype calls (Kang 2018 lanes 1A/1B/1C/2ctrl/2stim, J293t),
cell hashing (pbmc-ch, cline-ch, mkidney-ch), MULTI-seq (HEK-HMEC, HMEC-orig, HMEC-rep, nuc, pdx), species mixing (10x hm-6k,
hm-12k). 500 to 26,426 cells, 2.5% to 37% doublets. Plus **Kang 2018 lane 2.1 as Singlet counts it** (prod/modality-fixes, 13,737
cells with demuxlet labels, 11.6% doublets), the lane behind the production numbers. pbmc-2ctrl-dm is the same lane counted by Cell
Ranger 1.0 (the authors' matrix); both are kept and always held out together (section 2.2). The labels see heterotypic doublets
across donors / hashtags / species only; same-donor or same-hashtag doublets are invisible to the truth.

**Ambient truth (real).**
* Seven human-mouse barnyard libraries: hgmm1k_v3 as Singlet counts it (the tuning library) and 10x's published Cell Ranger
  matrices of hgmm1k_v3, 5k_v3, 10k_v3, 1k_v2 (Cell Ranger 3.0.0) and 6k_v2, 12k_v2 (2.1.0), from
  `https://cf.10xgenomics.com/samples/cell-exp/<version>/<name>/` (`bench/latent_qc/ambient_truth/RESULTS.md` lists the files).
  Truth as in `docs/modality-benchmarks/ambient.md` (branch plan/atlas-rerun): in a pure human cell every mouse UMI is ambient.
  No 10x software was run.
* Real tissue: Kang 2.1 (8 pooled PBMC donors), the production end-to-end run on prod/modality-fixes. A read in a donor's singlet
  that carries an allele the donor does not have came from the ambient pool (section 3.3).
* Atlas: 390 human and 71 mouse samples of 271 + 61 studies over 20 tissue groups, read from the public legacy bundles
  (`bench/latent_qc/ambient_atlas/samples_full.tsv`).

## 2. Doublets

### 2.1 Methods (all post-hoc on a sample's count matrix)

| id | method |
|---|---|
| `prod` | production scorer `detect_doublets` (prod/modality-fixes, doublet_detect.h through the existing dbl_run harness; gene level) |
| `scrublet` | Scrublet 0.2.3 defaults |
| `scdblfinder` | scDblFinder 1.23 defaults (R) |
| `solo` | Solo-style: scVI-like VAE (NB decoder, 10-d latent, 128 units) per sample on 2,000 HVGs, classifier on the latent, real vs 2n simulated doublets |
| `atlas_knn` | the cell type network `singlet.celltype_nn` m1_notissue (trained on 2.8M CELLxGENE cells; `docs/celltype_nn.md` on branch claude/celltype-nn) used as a fixed encoder: 2,048-d hidden layer of real cells and 2n simulated doublets, PCA-50, fraction of simulated cells among the k = 0.5 sqrt(n) nearest neighbours |
| `atlas_clf` | 5-fold cross-fitted classifier, real vs simulated, on the same embedding |
| `ct_entropy`, `ct_maxprob` | uncertainty of the network's calibrated cell type distribution |
| `nnls`, `atlas_nnls` | explicit "two cell types in one droplet" test: relative residual drop of the best non-negative two-centroid fit over the best single centroid of the cell's profile; centroids = the sample's own Leiden clusters (`nnls`) or 856 atlas cell-type centroids from the network's CELLxGENE corpus (`atlas_nnls`) |
| `combo_*` | cross-study supervised combiner: gradient boosting on within-sample rank-transformed scores (`combo_all`: all features; `combo_no_atlas`; `combo_atlas`; `combo_prod_atlas`: depth, genes, production score and the four atlas features), trained on the real labels of every other study (leave-one-study-out; the six Kang lanes form one study, the five MULTI-seq sets another, the two 10x hgmm sets a third) |
| `rankavg` | unsupervised mean of the within-sample ranks of scDblFinder, Solo, atlas_knn and production |
| `vireo_geno`, `prod_geno` | genotype doublet probability (vireo; Singlet's fixed demux), Kang 2.1 only |

### 2.2 Results (17 real-label datasets; AUROC / AUPRC)

Means over datasets (`bench/latent_qc/doublets/results/summary_real.csv`; per-study means in `per_study_means.csv` give the same
order):

| method | AUROC | AUPRC | AUPRC vs production: median delta, datasets better, Wilcoxon p | s / 10k cells |
|---|---|---|---|---|
| production | 0.787 | 0.501 | | 2.0 |
| Scrublet | 0.761 | 0.466 | -0.059, 4/17, 0.064 | 7.7 |
| scDblFinder | 0.798 | 0.559 | +0.027, 16/17, 0.0008 | 43.6 |
| Solo-style VAE | 0.753 | 0.412 | -0.099, 1/17, < 1e-4 | 167 (GPU) |
| atlas_knn | 0.795 | 0.501 | +0.009, 9/17, 0.68 | 1.3 (GPU) / 6-9 (CPU) |
| atlas_clf | 0.798 | 0.484 | +0.007, 11/17, 1.0 | 17 (GPU) / ~160 (CPU) |
| ct_entropy | 0.625 | 0.248 | -0.256, 0/17 | 1.2 |
| nnls (sample clusters) | 0.704 | 0.309 | -0.174, 2/17 | 23 |
| atlas_nnls (atlas centroids) | 0.617 | 0.160 | -0.301, 0/17 | 1.0 |
| combo_no_atlas | 0.798 | 0.547 | +0.034, 14/17, 0.001 | |
| **combo_prod_atlas** | 0.801 | 0.557 | **+0.060, 15/17, 0.0004** | ~10 |
| combo_all | 0.803 | 0.572 | +0.056, 16/17, < 1e-4 | |
| rankavg (no labels) | 0.804 | 0.561 | +0.047, 16/17, < 1e-4 | |

Paired between the top methods (AUPRC): combo_all vs combo_no_atlas +0.017 median, 16/17, p < 1e-4 (the atlas features add
information that the per-sample scores lack); combo_all vs scDblFinder +0.015, 13/17, p = 0.08; combo_prod_atlas vs scDblFinder
+0.000, 9/17, p = 0.64; combo_all vs rankavg +0.013, 10/17, p = 0.13 (supervision across studies helps little beyond a plain rank
average).

Kang 2.1 on Singlet's own counts (the production benchmark lane):

| method | AUROC | AUPRC |
|---|---|---|
| production | 0.919 | 0.632 |
| Scrublet | 0.880 | 0.582 |
| scDblFinder | 0.927 | 0.642 |
| Solo-style VAE | 0.876 | 0.566 |
| atlas_knn | 0.923 | 0.649 |
| combo_prod_atlas | **0.935** | **0.698** |
| vireo genotype | 0.950 | 0.842 |
| Singlet genotype (`prob_doublet`) | **0.969** | **0.897** |

The production numbers reproduce the doublet-fix benchmark (0.915 / 0.629 on the exon-mapped input; the brief's 0.90 / 0.57 is an
earlier run). Per dataset (`auroc_wide.csv`, `auprc_wide.csv`): the atlas features help most in the low-depth PBMC lanes
(pbmc-1A/1B/1C AUPRC 0.35 / 0.26 / 0.39 production -> 0.45 / 0.40 / 0.55 combo_prod_atlas) and in human-mouse mixtures they lose to
scDblFinder (hm-12k AUPRC 0.89 vs 1.00). J293t (500 cells, two cell lines) defeats every expression method (AUROC 0.46-0.57).

### 2.3 What did not work, and why

* **Two cell types by NNLS.** A best-two-centroid fit improves on a single centroid for most cells, singlets included: cell
  states are continuous, Poisson noise rewards an extra free component, and doublet-rich clusters form their own centroids. On
  atlas centroids the cross-platform gene bias (CELLxGENE counts vs Singlet's) dominates. The explicit test is the weakest score.
* **Cell type uncertainty.** A doublet of two lineages spreads the network's mass, but so do the many singlets of transitional or
  poorly covered types; entropy alone is near chance in cell lines (AUROC 0.42-0.52) and 0.62-0.72 in PBMCs.
* **Solo.** A 10-d per-sample latent loses the information the doublet classifier needs; Scrublet's PCA-kNN and the production
  scorer use more dimensions directly.
* **The atlas network on other pipelines' matrices.** On the Xi & Li matrices the network gives a confident (>= 0.9) lineage to only
  1-7% of PBMC singlets, against 40% on Singlet's counts of the same Kang lane (`homotypic_composition.csv`); its training augmentation
  targets Singlet's counts. Its doublet gain on Singlet data is therefore, if anything, underestimated by the Xi & Li sets.

### 2.4 Homotypic blind spot (real labels)

Real genotype / hashtag doublets of the PBMC pools, split by what the atlas network sees (recall at the truth rate;
`homotypic_real.csv`; only datasets where the network is confident on singlets):

| method | Kang 2.1: doublet looks like one lineage (26% of doublets) | Kang 2.1: two lineages (41%) | pbmc-ch: one lineage (22%) | pbmc-ch: two lineages (17%) |
|---|---|---|---|---|
| production | 0.53 | 0.77 | 0.69 | 0.77 |
| scDblFinder | 0.59 | 0.76 | 0.62 | 0.76 |
| atlas_knn | 0.47 | 0.79 | 0.62 | 0.81 |
| combo_prod_atlas | 0.50 | 0.77 | 0.66 | 0.84 |
| Scrublet / Solo | 0.30 / 0.18 | 0.75 / 0.78 | 0.27 / 0.30 | 0.74 / 0.72 |
| vireo genotype | **0.89** | 0.83 | | |
| Singlet genotype | **0.89** | 0.89 | | |

A quarter of real cross-donor PBMC doublets look like one cell type. Expression methods catch about half of them (mostly through the
doubled library size; UMI count alone gets 0.15-0.57), genotype doublets catch 0.89. No latent or atlas model closes this gap: it
is information the expression profile does not hold. Same-donor homotypic doublets are invisible to every method and to the truth.

## 3. Ambient RNA

### 3.1 Barnyard truth, 7 real libraries (`bench/latent_qc/ambient_truth/RESULTS.md`)

Spearman against truth (ratio of mean estimate to mean truth). Hyperparameters of new methods were tuned on hgmm1k_v3 (Singlet
counts) only; hgmm1k_v3 CR is the same sample; five libraries are held out.

| method | 1k_v3 Singlet (tuned) | 1k_v3 CR | 5k_v3 | 10k_v3 | 1k_v2 | 6k_v2 | 12k_v2 | held-out mean |
|---|---|---|---|---|---|---|---|---|
| production | 0.810 (0.92) | 0.917 (1.04) | 0.987 (0.98) | 0.782 (0.82) | 0.938 (1.19) | 0.979 (0.94) | 0.989 (0.95) | **0.935 (0.98)** |
| latent-kNN, PCA, k = 50 | 0.864 (1.05) | 0.782 (1.01) | 0.951 (0.83) | 0.944 (0.51) | 0.935 (0.85) | 0.899 (0.78) | 0.900 (0.82) | 0.926 (0.76) |
| latent-kNN, PCA, k = 30 | 0.877 (1.18) | 0.715 (1.18) | 0.939 (0.92) | 0.941 (0.61) | 0.934 (0.96) | 0.885 (0.88) | 0.892 (0.92) | 0.918 (0.86) |
| latent-kNN, VAE latent, k = 30 | 0.884 (1.20) | 0.671 (1.10) | 0.947 (0.89) | 0.938 (0.59) | 0.923 (0.95) | 0.876 (0.87) | 0.839 (0.90) | 0.905 (0.84) |
| cluster centroid (same EM) | 0.773 (1.01) | 0.876 (1.17) | 0.828 (0.42) | 0.784 (0.13) | 0.710 (0.66) | 0.935 (0.33) | 0.885 (0.31) | 0.828 (0.37) |
| DecontX + empty-droplet background | 0.790 (0.75) | 0.730 (0.76) | 0.845 (0.94) | 0.452 (0.38) | 0.399 (0.66) | 0.802 (0.78) | 0.674 (0.64) | 0.634 (0.68) |

* A finer latent neighbourhood ("native expression" from the cell's k nearest neighbours instead of its cluster) beats the cluster
  centroid inside the same EM, but not production's off-gene estimator; it wins only on the tuning library and 10k_v3 and
  under-estimates heavily contaminated libraries.
* On an independent-noise truth (estimate on one random half of the genes, truth from the other half) production scores 0.744 and
  latent-kNN 0.73.
* The truth's own reproducibility (split-half over each cell's UMIs, Spearman-Brown) is 0.83-0.98; production's 0.81 on the
  1k_v3 tuning library is within 0.02 of that ceiling, and it reaches 0.98-0.99 on 5k / 6k / 12k.
* Atlas-marker estimators cannot be built for HEK293T / NIH3T3 (their only reliably silent genes are the other species'). The oracle
  (true species-specific genes) needs about 1,000 marker genes to match production on the independent truth; 100 are not enough.
* Cost on one thread: production 4.8 s (5k cells) / 10.7 s (11k); latent-kNN 115 / 274 s, 4-7 GB; DecontX 75-290 s.

### 3.2 Ambient profiles across the atlas (`bench/latent_qc/ambient_atlas/RESULTS.md`)

* **Pools are large.** Uncalled low-UMI barcodes (below the uncalled median and 10% of the called median) hold a median 268k UMIs
  per sample (5th percentile 20k; 3% of samples below 10k). Production's rule (1-100 UMIs) gives a median 1.15M.
* **The profile is mostly the sample's own cells.** R^2 of a sample's ambient log-profile: own cell mean 0.83, same-tissue atlas
  mean of other studies 0.59, both 0.85. Study explains far more variance than tissue (eta^2 0.56 vs 0.10); tissue is recognisable
  from the ambient profile (5-NN accuracy 0.44 over 20 groups, chance 0.07), slightly less than from the cells (0.47).
* **Recurrent ambient markers** (enriched in the pool over the sample's own cells): hemoglobin (HBB > 2x in 68% of samples, median
  log2 3.4; strongest outside blood: adipose 4.8, thyroid / brain 4.2, heart 3.6), mitochondrial transcripts (MT-ATP8 / ND6 about
  2x; nuclei pools 3.6x more mitochondrial than nuclei), myeloid and cytotoxic transcripts (S100A8/9, C1QA, NKG7, GNLY), platelet
  PPBP in blood, and secreted tissue products in their tissue (liver ALB 1.8, APOC3 3.0, TTR 2.5; pancreas PNLIP 1.7, CTRB1 1.2;
  lung SCGB1A1 1.8; heart NPPA 2.0). MALAT1 is depleted, not enriched; immunoglobulins are enriched only in blood and adipose.
  Several top genes (GAREM2, LINGO1, SLC26A3, unannotated ids) look like pileup artefacts of the legacy counts.
* **Atlas prior test** (study-grouped cross-validation; KL in bits to a held-out half of the pool): an atlas prior (own cells x an
  atlas-learned per-gene enrichment, with tissue / global components) predicts the profile better than the sample's own pool
  sampled to n UMIs for every n up to 30,000 (0.345 vs 3.19 at n = 100, 0.49 at 30k), and ties the whole half-pool (0.345 vs 0.444,
  shrinkage 0.317). Almost all of the prior is the sample's own cells (0.416); the atlas enrichment factor adds 13%, tissue terms
  4%. With real pools of ~1e5 UMIs the empirical profile is already close to its target; the prior matters only for the ~3% of
  samples with < 3,000 pool UMIs.

### 3.3 Real-tissue truth from genotypes (Kang 2.1; `bench/latent_qc/ambient_kang/`)

In a donor's confident singlet (12,035 cells), reads that carry alleles the donor lacks (at sites where the donor is homozygous in
the pooled reads of its own cells) come from other donors through the ambient pool. With `f_s` the pool allele frequency measured
on the 138,015 ambient barcodes and sequencing error `e = 0.0017` measured where the pool is monomorphic,
`rho = (foreign - e * reads) / sum(reads * q_s)`. Median 167 informative reads per cell; per-cell truth is noisy (split-half
Spearman-Brown 0.38), so calibration is checked on groups (95% bootstrap intervals):

| group | cells | genotype truth | production run (mean rho) |
|---|---|---|---|
| all singlets | 12,035 | **2.48% [2.34, 2.64]** | 3.43% |
| CD4 T | 5,051 | 1.93% [1.75, 2.10] | 2.19% |
| CD8 T | 1,071 | 1.45% [1.14, 1.76] | 2.97% |
| B | 1,297 | 2.67% [2.28, 3.01] | 2.74% |
| CD14+ monocytes | 2,710 | 3.00% [2.74, 3.35] | 4.66% |
| FCGR3A+ monocytes | 702 | 2.36% [1.85, 2.95] | 5.35% |
| NK | 902 | 2.23% [1.77, 2.82] | 4.96% |
| dendritic | 199 | 2.12% [1.59, 2.65] | 6.99% |
| megakaryocytes | 103 | 9.31% [5.78, 12.46] | 11.93% |
| production rho quintile 1 / 3 / 5 | 2,407 each | 2.51% / 2.14% / 3.19% | 0.87% / 2.16% / 9.43% |

Production's level is right to within 1.4x (and finds the megakaryocyte excess), but its per-cell spread is not: cells it rates
0.9% and 9.4% differ by 0.7 points in truth. Rare clusters (DC, NK, FCGR3A+ monocytes) are over-estimated 2-3x. This is the first
real-tissue check of the estimator; the barnyard libraries (two cell lines, trivially separable) cannot show it.

### 3.4 The estimator depends on the cell set

The production run passes the caller's 17,004 cells (`ambient_contamination.tsv`); re-running the same function (harness, same
pool of 50,000 barcodes / 304,719 UMIs) reproduces it (mean rho 0.059 vs 0.060). Given instead the 14,407 knee-filter cells of
`cell_calls.tsv` (STARsolo CellRanger2.2, the call the legacy atlas carries), the same lane gives mean rho 0.100 (median 0.049),
i.e. about 2x, and 4x the genotype truth (9.8% over the truth singlets). Truncating the matrix to barcodes with >= 20 UMIs (as the
legacy atlas keeps only barcodes with >= 100 reads) changes the estimate less, and downward with the caller's cells (0.059 -> 0.038;
with the knee cells, >= 5-50 UMIs give 0.10-0.12).
Consequence: the atlas-wide production estimates of section 3.2's samples (median per-sample rho 0.19, 14% at the 0.5 cap;
`ambient_atlas/rho_by_sample.csv`) are inflated by the legacy knee-filter cell calls and should not be read as atlas contamination
levels; nuclei > cells (0.30 vs 0.16) and the tissue order may still hold. The cause inside the estimator (probably uncalled real
cells above the 100-UMI pool limit, or the knee set's cluster structure) was not isolated.

## 4. Cost and where it runs

| step | per 10k cells | runs on | needs realignment? |
|---|---|---|---|
| production doublet score | 2 s | binary (already) | no |
| atlas encoder features (embedding of real + 2x simulated cells, kNN) | 6-9 s CPU (8 threads), 1.3 s GPU | post-hoc job, next to the cell type sidecar inference (`scripts/celltype_nn/infer_campaign.py`, same network) | no: stored gene counts of the called cells |
| doublet combiner (gradient boosting, ~15 features) | < 1 s | post-hoc | no |
| genotype doublets | in the run | binary, needs the SNP panel | yes (reads) |
| production ambient estimate | 1-2 s | binary (in the run) | the run sees every barcode; a post-hoc rerun on stored counts sees only barcodes with >= 100 reads and estimates lower (section 3.4) |

The doublet pieces run on stored counts: they read only the called cells' gene counts (exon + intron), which the v2 `.s2` objects keep.
The ambient estimator should stay in the binary: its pool is the 1-100 UMI barcodes, and stored matrices keep only barcodes with
>= 100 reads. Outputs are per-cell scores and belong in the per-sample label sidecar keyed by
`(sample ordinal, col)` (`docs/CELL-ARCHITECTURE.md`), not in D1; the combiner model is a versioned artifact like the cell type
network. Nothing here touches genotypes in public objects: genotype doublet probabilities are a per-cell label, the allele counts
stay private.

## 5. Follow-ups found (not done here)

1. Doublet combiner in production: train on the 16 Xi & Li sets + Kang (all labels real), ship the model with the cell type network,
   run it in the post-hoc label job; report `doublet_score` and `is_doublet` with the 10x multiplet-rate threshold as now.
2. Ambient estimator: per-cell calibration in tissue (shrink rho toward the sample mean; the per-cell spread is ~10x too wide on Kang)
   and the cell-set sensitivity (section 3.4). Genotype pools give real tissue truth for both; more pooled lanes (Kang batch 1 A/B/C,
   other multiplexed studies in the atlas once the re-run has SNP counts) would make it a benchmark.
3. Prefer genotype doublets: in multiplexed samples combine `prob_doublet` with the expression score (a genotype doublet is
   decisive; same-donor doublets still need expression).

## Caveats

* Xi & Li matrices are other pipelines' counts (Cell Ranger, Drop-seq tools); Kang 2.1 and hgmm1k_v3 are the only doublet / ambient
  truth sets counted by Singlet. hgmm1k_v3 and Kang lane 2.1 (kang2018_ctrl) were used to tune production earlier and the ambient
  latent estimators here; their numbers are not held out.
* AUPRC depends on the doublet rate; paired comparisons are within dataset. Studies are unequal in size (6 Kang lanes, 5 MULTI-seq
  sets); per-study means are in `per_study_means.csv` and give the same ranking.
* The combiner's gain is measured leave-one-study-out on 8 studies; a production model would be trained on all of them, and its gain
  on new protocols (5', nuclei, Flex) is untested.
* The Kang ambient truth assumes the pool composition is the same in every droplet and sees only cross-donor ambient; same-donor
  ambient is counted through `f_s` (measured on the pool itself), so it estimates total contamination under that assumption.
* The atlas ambient track uses legacy bundle counts (inflated top genes, >= 100-read barcodes only, knee-filter cell calls); its pool
  is "uncalled low-UMI", not verified-empty (a stricter pool is less cell-like: KL to cells 0.92 vs 0.40 bits).
* CellBender was not rerun (0.60 Spearman in the earlier benchmark).

## Reproduce

`bench/latent_qc/doublets/`: `fetch_xili.sbatch` (Zenodo download, rds -> mtx), `prep.sbatch` (datasets), `centroids.sbatch`,
`cpu.sbatch` and `gpu.sbatch` (arrays over datasets), `eval.sbatch` (lineage probabilities, `evaluate.py`), `cputime.sbatch`.
`bench/latent_qc/ambient_truth/`: see its RESULTS.md. `bench/latent_qc/ambient_atlas/`: see its RESULTS.md.
`bench/latent_qc/ambient_kang/`: `run.sbatch` (dumps, harness on full and truncated inputs, genotype truth), `runcells.sbatch`.
