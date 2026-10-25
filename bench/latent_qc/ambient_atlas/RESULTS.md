# Track A1: ambient (empty-droplet) profiles across the atlas and an atlas prior

Branch `study/latent-qc-ambient-atlas`. All numbers below are measured on real atlas data (gene-level exon+intron counts of the
legacy `.singlet` bundles); the only resampling is subsampling UMIs of real ambient pools. Nothing synthetic, no R2/D1 writes.

## Data and method (short)

* Selection (`select_samples.py`, `samples.tsv`): round-robin over 20 coarse tissue groups (115 distinct tissue labels among the
  human samples that passed), <= 3 samples per study, primary/unlabelled sample kinds only (no cell lines, organoids, xenografts).
  514 requested (460 human, 54 mouse); 488 fetched with the existing range-reading fetcher (26 samples absent from their bundle,
  e.g. no count matrix), plus 49 cached samples of the empty-droplet study. Records were built for 480 samples; skipped:
  69 wrong strand (`wrong_strand/(wrong_strand+exon_hits) > 0.5`), 4 with < 20 called cells (no 0x0 matrices turned up).
  `samples_full.tsv` = samples with a record and a tissue label: **390 human (271 studies) + 71 mouse (61 studies)**; the analysis uses
  those with pool UMI >= 2,000 (385 human, 70 mouse).
* Per sample (`extract.py`): gene-level counts exon+intron summed; core genes = Ensembl ids present in >= 95% of human samples
  (38,606 genes; 78,298 for mouse); profiles are renormalised over core genes.
  * called profile `c_g`: sum over `called` barcodes (STARsolo CellRanger2.2 knee filter).
  * ambient pool, rule 1 (main): uncalled barcodes with total UMI < median(uncalled) **and** < 10% of the called-cell median.
    Median pool = 36k barcodes / 268k UMIs (below). Sensitivity pools: rule 2 = uncalled with <= 100 UMIs (the production rule),
    rule 3 = uncalled below the 25th percentile of uncalled **and** < 5% of the called median.
  * pool halves A/B: pool-1 barcodes randomly split in two; B is the held-out "truth", A is the source of subsampled UMIs.
  * enrichment `log2((a_g+0.5)/N_a) / ((c_g+0.5)/N_c)`, evaluated only where max(pool count, expected pool count under `c`) >= 5.
* Nuclei vs cells: `intron_frac_called > 0.45` (histogram of the called-cell intronic fraction is bimodal with a valley at 0.4-0.5);
  70 / 390 human samples are "nuclei". Metadata text mentions nuclei for only 24 samples and agrees poorly, so the intronic fraction is used.
* **3'/5' could not be determined**: `protocol_name` in the bundles only says chemistry (10xv2 133, 10xv3 232, others 25) and only one
  sample's characteristics mention 5'; 5'-vs-3' stratification is therefore not reported (the 69 wrong-strand samples that were dropped
  may be enriched for 5' libraries; this was not checked).
* Prior test (`prior_test.py`, `prior_summarize.py`): study-grouped 20-fold (a sample's study is never in the training set of its own
  prior, weights or hyper-parameters; hyper-parameters are cross-fitted by an inner study-grouped 5-fold inside the training studies).
  Predictors of half-pool B (KL in bits = cross-entropy minus entropy of B, also total variation and Spearman over genes with B > 0):
  (a) own called-cell mean; (b) own cell mean x atlas per-gene enrichment `(mean a_g / mean c_g)^lambda` learned on other studies
  (lambda chosen in {0.25..1}, 1.0 in 16/20 folds); (c) tissue-mean ambient of other studies (global mean if the tissue is absent);
  (d) linear mixture of cell, enrichment-reweighted, tissue, global, uniform with weights fitted by EM on cross-fitted training predictions
  (mean weights 0.10 / 0.80 / 0.05 / 0.05 / 0.003); (e) n = 100 ... 30,000 UMIs drawn without replacement from half A (3 draws each),
  alone (pseudocount kappa spread uniformly, kappa tuned on training studies) and as `(x + kappa*prior)/(n + kappa)` toward the tissue
  prior and toward the mixture prior (kappa tuned per n on training studies). "All" = the whole half A.
  KL is computed against a noisy half-pool, so it contains a constant sampling-noise offset common to every predictor; differences are valid, absolute values are inflated.

## 1. Where the real samples sit: pool size

Rule-1 pool UMIs (390 human): 5th pct 19.9k, 10th 41k, 25th 123k, **median 268k**, 75th 532k, 90th 995k; median 36k barcodes.
3.3% of samples have < 10k pool UMIs, 7.4% < 30k, 20.8% < 100k. Production rule (<= 100 UMIs, 50k-barcode cap reached in 82% of samples):
median 1.15M UMIs. (Caveat: the matrices only keep barcodes with >= 100 raw reads.) **Real atlas samples are almost all in the regime of >= 4e4 pool UMIs.**

## 2. Composition (rule-1 pool vs called cells, human; medians)

| class | ambient | cells |
|---|---|---|
| mitochondrial (chrM) | 3.6% (mean 8.3%) | 3.3% (mean 4.7%) |
| MALAT1 | 1.5% | 2.4% |
| ribosomal proteins | 13.9% | 12.0% |
| HBB+HBA1+HBA2 | 0.01% (mean 0.41%) | 0.00% (mean 0.47%) |
| immunoglobulin | 0.01% (mean 1.2%) | 0.00% (mean 1.6%) |

Per sample, ambient mt share / called mt share has median 1.47. Nuclei: ambient mt 2.5% vs 0.7% in nuclei, ribosomal 2.6% vs 0.6%
(nuclei pools are mt-enriched 3.6x relative to nuclei). Tissue medians of ambient mt: gut 20.9% (cells 14.9%), liver 7.9% (cells 2.2%), lung 5.5%, heart 5.2% (cells 2.1%), kidney 3.8% (cells 1.7%); ambient bone-marrow HBB+HBA 2.8%.
Per-sample table: `results/sample_summary.csv`.

## 3. Atlas ambient markers (`markers_overall.csv`, `markers_by_tissue.csv`, `results/targeted.csv`)

Overall (385 human samples; genes evaluable in >= 30% of samples, mean ambient CPM >= 50), median per-sample log2(ambient/cells), fraction of samples > 2x:

| gene | n eval | median log2 enr | frac > 2x | amb CPM | cell CPM |
|---|---|---|---|---|---|
| HBB | 233 | 3.40 | 0.68 | 3497 | 4094 |
| HBA2 | 138 | 2.88 | 0.67 | 420 | 392 |
| HBA1 | 113 | 2.10 | 0.58 | 256 | 235 |
| MT-ATP8 | 292 | 1.03 | 0.52 | 575 | 215 |
| MT-ND6 | 301 | 0.97 | 0.50 | 261 | 119 |
| MT-ATP6 | 367 | 0.70 | 0.39 | 10367 | 5486 |
| MT-CO3 / ND4L / ND2 / CYB / CO2 / ND1 | 340-374 | 0.54-0.64 | 0.35-0.41 | 1k-12.6k | 0.5k-7.1k |
| S100A8 / S100A9 | 208 / 242 | 0.82 / 0.69 | 0.41 / 0.40 | 1063 / 1486 | 844 / 1120 |
| C1QA, AIF1, RNASE1, CCL5, NKG7, GNLY, TRBC1/2 | 150-210 | 0.7-1.1 | 0.43-0.51 | 90-440 | 60-420 |
| GAREM2, LINGO1, SLC26A3, SNHG9/25 | 159-300 | 1.5-4.0 | 0.62-0.94 | 60-340 | 22-267 |
| unannotated ENSG00000280441 / 278996 / 289474 / 289901 | 158-370 | 0.8-2.0 | 0.44-0.75 | 445-2313 | 192-846 |

Classes: among the top 200 enriched genes 12 are MT, 9 ribosomal, 2 hemoglobin, the rest mostly immune/myeloid, nuclear-encoded
mitochondrial (ATP5*, NDUF*, COX7A1) and a handful of lncRNA / unannotated ids. The unannotated Ensembl ids, GAREM2, LINGO1 and SLC26A3
are suspicious (probably multimapping / overlapping-locus or intronic artefacts of the pileup, not biology); they are reported, not interpreted.
MALAT1 is *depleted* in ambient (median log2 -0.58, 2% of samples > 2x), contrary to the expectation for fragile cells.

Hemoglobin: HBB is > 2x enriched in ambient in 68% of samples where it is evaluable, strongest outside blood/marrow
(adipose 4.8, thyroid/endocrine 4.2, brain 4.2, oral/nasal 3.8, heart 3.6, skin 3.1, liver 3.1 median log2 enrichment: blood contamination), while
in bone marrow it is not enriched (-0.24; erythroid cells themselves carry it, 39,000 CPM ambient vs 64,000 in cells).
Tissue panel (`targeted.csv`; median log2 amb/cell, n = evaluable samples):
liver ALB 1.76 (18; 20,100 vs 6,800 CPM), APOC3 3.02 (17), TTR 2.48, APOA1 2.17; pancreas PNLIP 1.66 (8), CTRB1 1.19, REG1A 1.17,
CELA3A 0.89, PRSS2 0.73, PRSS1 0.49 (10), CPA1 0.30 (9), INS 0.41, GCG 0.31 (the pancreas panel is evaluable in only 8-14 of 19 samples);
lung SFTPC -0.03 (13; 6,400 vs 6,700 CPM, i.e. ambient SFTPC is high but not enriched over its cell level), SCGB1A1 1.84, SFTPA1 1.16;
heart NPPA 1.95, MB 1.69; brain PLP1 1.73; blood PPBP 3.08 (platelets), HBB 4.60, JCHAIN 1.15, IGKC 0.91; bone marrow ELANE 2.93.
Immunoglobulins are **not** globally enriched (IGKC -0.13, JCHAIN -0.46 over all samples), only in blood/adipose.
Ranking genes by "ambient excess over other tissues" (`tissue_excess` rows of `markers_by_tissue.csv`) returns the tissue's own
cell-type markers (INS 18.3, GCG 13.0, ALB 12.1, SFTPC 10.6, MYL7 12.0, DCD 11.2 log2 vs other tissues), i.e. the ambient profile of a
tissue is dominated by its own cells' transcripts, as expected; what is *enriched relative to the sample's own cells* is the smaller
set above (hemoglobin, mt, stress/secreted, some tissue products).

## 4. Structure (`results/structure_knn.csv`, `results/r2_by_sample.csv`, `fig1_structure.png`)

Profiles are log2(CPM+5) over 15,973 expressed core genes, PCA(30), cosine 5-NN with **neighbours restricted to other studies**.

| | ambient pool | called cells | majority class |
|---|---|---|---|
| tissue (20 groups) 5-NN accuracy | **0.44** | **0.47** | 0.07 (label-shuffled 0.05) |
| chemistry (10xv2/v3/other) | 0.70 | 0.76 | 0.60 |
| nuclei vs cells | 0.93 | 0.96 | 0.82 |

Tissue silhouette (cosine) is ~ -0.03 for both (clusters are not tight). Adjusted eta-squared per gene (median over genes): tissue 0.10 (ambient)
vs 0.14 (cells); **study** 0.56 (ambient) vs 0.77 (cells): batch/study explains far more than tissue.
Mouse (70 samples): tissue 5-NN 0.23 (ambient) vs 0.27 (cells), chance 0.07; study eta-squared 0.79.

Variance of one sample's ambient log-CPM profile across genes (R^2, median over 385 samples): own cell mean **0.83**, atlas mean of the same tissue in
other studies 0.59, mean of all other studies 0.56, own cells + tissue mean 0.85 (mean 0.77 / 0.55 / 0.51 / 0.79). So the ambient profile is
explained mostly by the sample's own cells; the tissue atlas mean adds ~0.02 R^2 on top.

## 5. Atlas-prior test (`prior_test.csv`, `prior_summary.csv`, `prior_contrasts.csv`, `fig2_prior_test.png`)

Study-grouped CV; 333 samples that have >= 30,000 UMIs in half A (all numbers below on this common subset, mean KL in bits, lower is better; 384 samples for "all"):

| UMIs from own pool | empirical alone | + tissue-mean prior | + atlas mixture prior |
|---|---|---|---|
| 100 | 3.19 | 1.02 | 0.343 |
| 300 | 2.60 | 0.97 | 0.339 |
| 1,000 | 1.86 | 0.84 | 0.332 |
| 3,000 | 1.36 | 0.73 | 0.316 |
| 10,000 | 0.82 | 0.54 | 0.283 |
| 30,000 | 0.49 | 0.39 | 0.246 |
| all of half A (median ~134k; 384 samples) | 0.444 | 0.411 | 0.317 |

Predictors that use **no** UMIs of the sample's pool (same 333 samples; KL / TV / Spearman): own cells 0.416 / 0.252 / 0.812;
own cells x atlas enrichment 0.360 / 0.233 / 0.819; **atlas mixture 0.345 / 0.231 / 0.821**; tissue-mean ambient of other studies 1.145 / 0.433 / 0.687;
global mean ambient 1.293 / 0.454 / 0.663. (384 samples: cells 0.558, enrichment 0.505, mixture 0.468, tissue mean 1.319.)
Empirical alone at n = 30,000: TV 0.22, Spearman 0.65; with all of half A: KL 0.444, Spearman 0.76.

Findings:
* The atlas mixture prior (no pool data at all) beats the empirical profile of n UMIs for every n up to 30,000 (empirical alone better than the prior in 0% / 0% / 1% / 2% / 4% / 17% of samples at n = 100 / 300 / 1k / 3k / 10k / 30k) and ties it at the whole half-pool (prior better in 40% of 384 samples; mean difference +0.025 bits in favour of empirical).
* Almost all of that prior's power is the sample's **own called-cell mean**: the atlas-learned enrichment factor reduces KL by 13% (0.416 -> 0.360) and tissue/global components by a further 4% (-> 0.345). The tissue-mean ambient alone is a poor predictor (1.15), worse than the cells.
* Shrinking the empirical profile toward the prior beats empirical alone at **every n** (paired, study-bootstrap 95% CI excludes 0 at every n; at "all": -0.126 bits [-0.156, -0.098], better in 89% of samples).
* Own pool UMIs add information beyond the prior only from ~3,000 UMIs (mean 0.316 vs prior-only 0.345; better in 72% of samples; at 30,000: 0.246, -0.10 bits, -29%). Below ~1,000 UMIs the shrunk estimate equals the prior (0.343 vs 0.345). So: prior matters decisively for pools < ~3k UMIs, helps modestly (0.03-0.1 bits) for 3k-30k, and still reduces KL by 0.126 bits (29%) at the median pool of the atlas (>= 1e5 UMIs) by variance reduction.
* **Real regime** (section 1): median pool 268k UMIs, 5th percentile 20k. Only ~3% of samples are below 10k UMIs, so the prior's decisive regime covers few atlas samples; for the rest a per-sample empirical profile is already close to the target and an atlas prior gives a modest refinement, not a change in kind.

## 6. Production gene-level estimator on every sample (`rho_by_sample.csv`, `results/rho/`, `fig3_rho.png`)

`correct_ambient_gene_level_split` (prod/modality-fixes @ 4f110f2, default `AmbientDecontConfig`, exon and intron matrices supplied separately, 4 threads;
`harness_split.cpp`). 389/390 human samples returned ok. Pool as the production rule defines it (uncalled 1..100 UMIs, top 50,000 barcodes; cap hit in 82%).
Runtime per sample: median 0.48 s estimator (0.70 s with load), p90 0.96 s (1.34 s), max 2.5 s.
Mean rho per sample: quantiles 10/25/50/75/90% = 0.053 / 0.097 / **0.189** / 0.312 / 0.420 (UMI-weighted median 0.150). 13.6% of samples have median cell rho >= 0.45 (rho is capped at 0.5), 27% have mean rho > 0.3.
Setting the pool cap to rule-1's UMI limit changes little (Spearman of per-sample mean rho 0.985; median 0.197 vs 0.189).

Median of per-sample mean rho (IQR) by tissue group (human):

| tissue | n | median | q25-q75 |
|---|---|---|---|
| heart | 18 | 0.298 | 0.137-0.339 |
| lymphoid | 18 | 0.280 | 0.186-0.335 |
| kidney | 17 | 0.265 | 0.158-0.428 |
| brain | 22 | 0.261 | 0.198-0.324 |
| adipose | 20 | 0.223 | 0.151-0.443 |
| liver | 20 | 0.223 | 0.070-0.327 |
| other | 21 | 0.220 | 0.092-0.398 |
| breast | 20 | 0.209 | 0.156-0.405 |
| reproductive | 21 | 0.190 | 0.124-0.275 |
| oral/nasal | 22 | 0.160 | 0.108-0.256 |
| gut | 19 | 0.159 | 0.059-0.303 |
| blood | 26 | 0.159 | 0.071-0.242 |
| bone marrow | 22 | 0.157 | 0.080-0.234 |
| lung | 18 | 0.155 | 0.088-0.247 |
| prostate | 21 | 0.147 | 0.082-0.288 |
| skin | 15 | 0.136 | 0.092-0.192 |
| thyroid/endocrine | 12 | 0.135 | 0.106-0.227 |
| eye | 20 | 0.133 | 0.072-0.173 |
| muscle | 18 | 0.116 | 0.062-0.345 |
| pancreas | 19 | 0.092 | 0.062-0.210 |

Nuclei 0.295 (n = 70, IQR 0.21-0.41) vs cells 0.161 (n = 319, 0.08-0.28). Chemistry: 10xv2 0.190 (133), 10xv3 0.172 (232), drop-seq 0.363 (5),
10x multiome 0.358 (3), BD Rhapsody 0.214 (7) (counts of non-10x chemistries are tiny). Mouse (70 samples, `results/mouse/rho_by_sample.csv`): median 0.172, nuclei 0.42 (n = 5).
Per-sample mean rho correlates only weakly with the ambient mt/hemoglobin/MALAT1 share (Spearman -0.10 / -0.17 / -0.01).

## Caveats

* **The ambient "truth" is the low-UMI uncalled barcode pool, not a verified empty-droplet set.** Uncalled barcodes can hold cell-like material, so the own-cell-mean advantage in section 5 is partly the pool resembling cells. Pool-rule sensitivity (`results/pool_sens.csv`, 362 samples): Spearman rule1 vs production rule 0.96 (median; IQR 0.93-0.98), rule1 vs stricter rule 3 0.85 (0.76-0.91); KL(pool || own cells) is 0.40 bits for rule 1, 0.29 for the production rule and 0.92 for the strictest, i.e. the emptier the pool, the less it looks like the cells. The prior test was run only on rule 1; the value of "own cells" as a predictor may be lower for a stricter pool. The human-mouse benchmark (another track) has real truth.
* No genotype or simulated data were used. Subsamples of half A are resamples of real pool UMIs. The truth half B is noisy (median ~134k UMIs), so absolute KL has a noise offset.
* Atlas counts are inflated for high-expressed genes and the pileup has artefacts (several top "ambient markers" are unannotated or odd loci). The matrices only contain barcodes with >= 100 raw reads, so the production rule's 1..100 UMI pool is a truncated population here; rho values are therefore tied to this truncated input and have no ground truth in this track. Many samples sit near the estimator's cap (rho_max = 0.5).
* 3'/5' not determinable; 69 wrong-strand samples excluded (13% of those processed). Tissue labels are coarse groups (20 groups; 115 raw labels), "other" is a catch-all. Core-gene restriction (95% presence) drops genes only present in larger references.
* Hyper-parameters for the mixture and kappa were tuned on training studies with an inner cross-fit; reported differences between predictors use the study-grouped folds, but multiple n and predictors were compared without correction.

## Files and commands

`select_samples.py` -> `samples.tsv`; `fetch.sbatch` (+`fetch_driver.py`, array 0-39 %10, short-cpu, 1 CPU, ~40 CPU-h total); `extract.sbatch` (`extract.py` + `harness_split.cpp`, array of 20, 4 CPU,
~ 30 CPU-h); `analyze.sbatch STEP` (a / mouse / prior / rho / rhomouse / targeted / poolsens; edit the `case` word, `NUC=0.45`); `prior_summarize.py`, `prior_extra.py`, `make_figs.py` run locally on the small CSVs in `results/`.
Compute: a few tens of CPU-hours in total (fetch + extract arrays plus ~1 h of analysis jobs). Large intermediates (`npz/`, `rec/`, `rho/`) stay on Clipper under `$WORK/latent_qc/ambient_atlas/`.
