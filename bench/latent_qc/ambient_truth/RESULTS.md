# Track A2: per-cell ambient RNA against real barnyard truth (2026-10-25)

Question: can a latent-neighbourhood ("native expression") or atlas-marker model estimate per-cell ambient contamination better
than Singlet's production estimator (`correct_ambient_gene_level`, prod/modality-fixes)? Everything below is scored on **real**
human-mouse mixture libraries with cross-species truth; nothing is simulated (the earlier simulated "noise ceiling" was removed; see
"Truth noise"). Code: `bench/latent_qc/ambient_truth/`, raw per-library CSVs: `results/raw/`, merged: `results/*.csv`,
all tables as printed by `aggregate.py`: `results/tables.txt`.

## Verdict

* **No latent model beat production on held-out libraries.** Mean Spearman over the 5 libraries not used for tuning (5k_v3, 10k_v3,
  1k_v2, 6k_v2, 12k_v2): production **0.935**, latent-kNN (k=50) 0.926, (k=30) 0.918, VAE-latent kNN 0.905, cluster-centroid 0.828,
  DecontX+background 0.634. Mean ratio to truth: production 0.98, kNN k=30 0.86, kNN k=50 0.76 (under-estimates in the heavily
  contaminated libraries). On the gene-split truth (independent noise) production 0.744 vs kNN 0.73.
* The kNN estimator wins only where it was tuned (hgmm1k_v3 Singlet counts: Spearman 0.877 vs 0.810) and on hgmm10k_v3 (0.941 vs
  0.782); it loses on 1k_v3 (Cell Ranger counts, same sample as the tuning library), 5k, 6k and 12k (0.72-0.94 vs 0.92-0.99).
* Headroom above production is small: the truth itself is only reproducible to a Spearman of 0.83-0.98 (split-half, Spearman-Brown),
  and production sits at 0.78-0.99 on the full-gene truth.
* **Atlas markers:** a meaningful atlas marker set cannot be built for HEK293T/NIH3T3 (see below), so only the oracle (species-specific
  genes) is reported. It is an upper bound that needs about 1000 marker genes to reach production's accuracy on the independent
  (gene-split) truth; with 100 genes it is below production.
* Cost: latent-kNN costs 115 s (5k cells) to 274 s (11k cells) on one CPU thread, 4-7 GB; production costs 4.8 s and 10.7 s
  (1.8 GB) on the same libraries, DecontX 75-290 s. The kNN estimator does not meet the 1 CPU-minute budget as written.

## Libraries and truth

Truth as in `docs/modality-benchmarks/ambient.md`: in a pure human cell every mouse UMI is ambient, so
`rho_true = mouse_fraction / mouse_share_of_pool` (human share for mouse cells); pool = barcodes with 1-100 UMI, top 50,000 by UMI;
pure cell = minor-species fraction <= 5% (the rest are excluded from scoring but are in every method's input).
Counts are 10x Cell Ranger public matrices (raw and filtered), **not Singlet's pipeline counts**, except the first row.

| library | chemistry / source | cells (pure) | pool mouse share | mean truth |
|---|---|---|---|---|
| hgmm1k_v3_singlet | v3, Singlet STARsolo counts, 880 Singlet-called cells (the previous benchmark; **tuning library**) | 880 (879) | 0.277 | 1.01% |
| hgmm1k_v3 | v3, Cell Ranger 3.0.0 hg19+mm10 (same sample as above; **not independent of tuning**) | 1046 (1023) | 0.291 | 1.03% |
| hgmm5k_v3 | v3, CR 3.0.0 | 5366 (4899) | 0.409 | 2.70% |
| hgmm10k_v3 | v3, CR 3.0.0 | 11456 (9619) | 0.217 | 4.49% |
| hgmm1k_v2 | v2, CR 3.0.0 | 1080 (1052) | 0.289 | 2.44% |
| hgmm6k_v2 | v2, CR 2.1.0 (hg19 / mm10 matrices stacked over the same barcodes) | 6806 (6558) | 0.360 | 3.21% |
| hgmm12k_v2 | v2, CR 2.1.0 | 12820 (11742) | 0.363 | 3.78% |

Downloads probed (`probe.sbatch`): exist: 3.0.0 hgmm_1k_v3, 5k_v3, 10k_v3, 1k_v2 (raw+filtered h5); 2.1.0 hgmm_1k, 6k, 12k (tar.gz).
HTTP 403 (do not exist): 3.1.0 5k_hgmm_v3_nextgem / hgmm_5k_v3_nextgem, 3.0.2 hgmm_1k_v3, 1.1.0 hgmm_*, 3.0.0 hgmm_5k_v2, 2.1.0
hgmm_1k_v2, 3.0.0 hgmm_1k_v2 tarball. hgmm_1k (2.1.0) was not run (redundant with 1k_v2). Libraries with heavy mixing (10k: 16% of called
cells have >5% minor-species reads) lose those cells from scoring, which truncates the truth distribution.

## Two truths

* **Full-gene truth** (as in the previous benchmark): estimator and truth use the same reads, so an estimator that effectively
  counts cross-species reads is rewarded by shared noise (the oracle marker with all genes scores 0.996-1.000, by construction).
* **Gene-split truth** (new, real data): genes are split at random in halves A/B (fixed seed). Every method (production
  harness, DecontX, kNN, markers) is run on half A only and scored against the truth computed from half B (and vice versa); the
  mean of A->B and B->A is reported. Estimator and truth noise are then independent, at the price of halving the reads in both
  (so scores are lower, and `marker:all` ~0.57-0.97 is its practical ceiling).

## Headline: Spearman against truth (ratio of mean estimate to mean truth in brackets)

Full-gene truth. Hyperparameters of the new estimators were chosen on hgmm1k_v3_singlet only (column 2 is tuned, column 3 is the
same sample re-counted; columns 4-8 are held out).

| method | 1k_v3 Singlet (tuned) | 1k_v3 CR | 5k_v3 | 10k_v3 | 1k_v2 | 6k_v2 | 12k_v2 | held-out mean (5 libs) |
|---|---|---|---|---|---|---|---|---|
| production | 0.810 (0.92) | 0.917 (1.04) | 0.987 (0.98) | 0.782 (0.82) | 0.938 (1.19) | 0.979 (0.94) | 0.989 (0.95) | **0.935** (0.98) |
| DecontX + background | 0.790 (0.75) | 0.730 (0.76) | 0.845 (0.94) | 0.452 (0.38) | 0.399 (0.66) | 0.802 (0.78) | 0.674 (0.64) | 0.634 (0.68) |
| DecontX default | 0.380 (0.44) | 0.779 (0.45) | 0.826 (0.67) | 0.610 (0.39) | 0.440 (0.31) | 0.749 (0.28) | 0.945 (0.24) | 0.714 (0.38) |
| latent-kNN PCA k=30 (hier) | 0.877 (1.18) | 0.715 (1.18) | 0.939 (0.92) | 0.941 (0.61) | 0.934 (0.96) | 0.885 (0.88) | 0.892 (0.92) | 0.918 (0.86) |
| latent-kNN PCA k=50 (hier) | 0.864 (1.05) | 0.782 (1.01) | 0.951 (0.83) | 0.944 (0.51) | 0.935 (0.85) | 0.899 (0.78) | 0.900 (0.82) | 0.926 (0.76) |
| latent-kNN PCA k=15 (hier) | 0.825 (1.41) | 0.604 (1.42) | 0.918 (1.04) | 0.906 (0.75) | 0.889 (1.09) | 0.813 (1.00) | 0.829 (1.05) | 0.871 (0.99) |
| latent-kNN VAE k=30 (hier) | 0.884 (1.20) | 0.671 (1.10) | 0.947 (0.89) | 0.938 (0.59) | 0.923 (0.95) | 0.876 (0.87) | 0.839 (0.90) | 0.905 (0.84) |
| latent-kNN PCA k=30, no cluster smoothing | 0.791 (1.37) | 0.687 (1.42) | 0.907 (1.03) | 0.947 (0.68) | 0.914 (1.11) | 0.872 (0.92) | 0.879 (1.03) | 0.904 (0.95) |
| cluster centroid (k-means, K=sqrt(n/20)) | 0.773 (1.01) | 0.876 (1.17) | 0.828 (0.42) | 0.784 (0.13) | 0.710 (0.66) | 0.935 (0.33) | 0.885 (0.31) | 0.828 (0.37) |
| oracle markers, 100 genes | 0.832 (0.91) | 0.865 (0.91) | 0.950 (0.81) | 0.882 (0.87) | 0.888 (0.85) | 0.935 (0.87) | 0.957 (0.89) | 0.922 (0.86) |
| oracle markers, 1000 genes | 0.947 (0.93) | 0.949 (0.97) | 0.990 (0.97) | 0.982 (0.99) | 0.958 (0.93) | 0.982 (0.95) | 0.990 (0.97) | 0.980 (0.96) |

Gene-split (independent-noise) truth, Spearman, mean of A->B and B->A:

| method | 1k_v3 Singlet (tuned) | 1k_v3 CR | 5k_v3 | 10k_v3 | 1k_v2 | 6k_v2 | 12k_v2 | held-out mean |
|---|---|---|---|---|---|---|---|---|
| production | 0.544 | 0.700 | 0.837 | 0.918 | 0.693 | 0.549 | 0.721 | **0.744** |
| DecontX + background | 0.593 | 0.449 | 0.709 | 0.438 | 0.259 | 0.504 | 0.597 | 0.501 |
| latent-kNN PCA k=30 (hier) | 0.637 | 0.571 | 0.797 | 0.902 | 0.707 | 0.624 | 0.632 | 0.732 |
| latent-kNN PCA k=50 (hier) | 0.621 | 0.579 | 0.814 | 0.859 | 0.693 | 0.633 | 0.640 | 0.728 |
| latent-kNN VAE k=30 (hier) | 0.634 | 0.557 | 0.755 | 0.855 | 0.694 | 0.582 | 0.584 | 0.694 |
| cluster centroid | 0.538 | 0.650 | 0.692 | 0.453 | 0.529 | 0.510 | 0.652 | 0.567 |
| oracle markers, 100 genes | 0.487 | 0.570 | 0.784 | 0.955 | 0.650 | 0.515 | 0.659 | 0.713 |
| oracle markers, 1000 genes | 0.575 | 0.690 | 0.849 | 0.970 | 0.703 | 0.550 | 0.702 | 0.755 |
| oracle markers, all other-species genes | 0.602 | 0.696 | 0.866 | 0.972 | 0.725 | 0.560 | 0.712 | 0.767 |

Pearson and ratio tables for both truths, and per-depth-third means, are in `results/tables.txt`. Depth thirds (estimate / truth,
full-gene): production 0.82-1.43 across thirds and libraries (it is within +-10% in most cells; 1k_v2 low third 1.43); kNN k=30
0.57-1.28 (flat in depth, but biased low on 10k).

## Correction table (full-gene truth; expected removal for the model-based estimators)

Cross-species UMIs left after correction (%, lower is better) and own-species UMIs removed / expected own-species ambient (1.0 = right
amount; the previous benchmark's SoupX 1.25, CellBender 1.03). kNN removal is the posterior ambient responsibility summed over entries
(expected counts, not rounded).

| method | residual %: 1k_v3 S / 1k_v3 CR / 5k / 10k / 1k_v2 / 6k / 12k | own/expected: same order |
|---|---|---|
| production | 8.4 / 11.8 / 3.9 / 30.6 / 4.4 / 2.3 / 1.1 | 0.95 / 1.04 / 0.89 / 0.58 / 1.04 / 0.89 / 0.90 |
| DecontX + background | 22.2 / 23.0 / 24.5 / 74.1 / 40.3 / 24.1 / 39.9 | 0.81 / 0.88 / 0.90 / 0.54 / 0.68 / 0.84 / 0.72 |
| kNN k=30 (hier) | 16.8 / 20.8 / 20.4 / 41.5 / 16.2 / 19.8 / 18.1 | 1.39 / 1.60 / 1.13 / 0.67 / 1.19 / 0.98 / 1.09 |
| kNN k=50 (hier) | 18.4 / 22.4 / 24.1 / 49.2 / 18.6 / 26.1 / 23.7 | 1.19 / 1.35 / 0.99 / 0.54 / 1.03 / 0.85 / 0.94 |
| cluster centroid | 23.1 / 30.5 / 61.9 / 88.2 / 43.8 / 68.7 / 69.6 | 1.05 / 1.12 / 0.41 / 0.11 / 0.65 / 0.30 / 0.31 |

Production removes the cross-species reads best (1-12%, except 10k at 31%) while removing about the expected amount of own-species
ambient, i.e. its per-cell scale and its gene-level removal are both right on these libraries.

## Truth noise (real-data analysis; replaces the earlier simulated ceiling)

This table analyses the noise of the truth itself and is not a method score. Method: each pure cell's real UMIs are split at random in
two halves (exact hypergeometric split of the observed cross-species count), `rho_true` is computed from each half, and the halves are
correlated across cells; Spearman-Brown `2r/(1+r)` gives the reproducibility of the full-depth truth. "Reliability" is
(observed variance of truth - mean Poisson noise variance)/observed variance; sqrt is the Pearson ceiling of any estimator.

| library | split-half Spearman | Spearman-Brown Spearman | split-half Pearson | Spearman-Brown Pearson | reliability Pearson ceiling |
|---|---|---|---|---|---|
| 1k_v3 Singlet | 0.716 | 0.834 | 0.926 | 0.962 | 0.980 |
| 1k_v3 CR | 0.737 | 0.849 | 0.882 | 0.937 | 0.968 |
| 5k_v3 | 0.952 | 0.976 | 0.948 | 0.974 | 0.987 |
| 10k_v3 | 0.921 | 0.959 | 0.949 | 0.974 | 0.986 |
| 1k_v2 | 0.781 | 0.877 | 0.846 | 0.917 | 0.960 |
| 6k_v2 | 0.815 | 0.898 | 0.910 | 0.953 | 0.976 |
| 12k_v2 | 0.891 | 0.942 | 0.908 | 0.952 | 0.975 |

Reading: on the 1k libraries (small cross-species counts) the truth is reproducible to a Spearman of only ~0.83-0.88, so production's 0.81
on 1k_v3 Singlet is already within ~0.02 of what any estimator can show; headroom exists only on the larger libraries and there
production is at 0.98-0.99 (5k, 6k, 12k) or limited by 10k's excluded mixed cells. A Pearson ceiling of ~0.96-0.99 (reliability)
is the less informative bound because heavy-tailed truth dominates Pearson.

## Method notes

* **Production**: `harness.cpp` (copied unchanged from prod/modality-fixes `bench/modalities/ambient_fix`) compiled against `$PROD`
  headers, default config, on the full trimmed raw matrix; scores reproduce the previous result on 1k_v3 Singlet (Spearman 0.81, Pearson
  0.87, ratio 0.92 vs 0.93). Output identical for 1 and 16 threads was not rechecked here.
* **DecontX** (celda, R): `decontX(counts)` default clustering, and with a 20,000-droplet 10-100 UMI background; seed 1.
* **latent-kNN mixture** (`methods.py`): x_c ~ Multinomial(N_c, (1-rho_c) q_c + rho_c a); a = ambient pool profile (+0.5); q_c = the cell's
  k nearest neighbours' summed counts with each neighbour's estimated ambient removed ((sum_j x_j - rho_j N_j a)+), self excluded, smoothed
  with `tau` pseudo-UMIs towards a target profile (global native profile, or with `hier=1` the cell's k-means cluster profile);
  rho_c by safeguarded-Newton 1-D maximum likelihood (concave), optional shared Beta prior (mean = median of previous pass, strength
  `kappa` pseudo-reads); 3 refinement passes. Latent = PCA(20) of 2000 HVG log1p-CP10k, or a per-sample VAE (10-d, MLP 128x2,
  multinomial likelihood, 150 epochs, H100). Cluster variant = same model with all cluster members as neighbours.
* **Hyperparameter choice** (tuning library hgmm1k_v3_singlet only, `tune.sbatch`, `tune2.sbatch`, grid in `results/raw/*singlet*`):
  chosen by Spearman on the gene-split truth (independent noise): k in {15,30,50}, tau in {1..10000}, kappa in {0,100,1000}, hier in {0,1}.
  Best: k=30, tau=10000, kappa=1000, hier=1 (gene-split Spearman 0.637, ratio 1.07). **tau=10000 is the edge of the grid** and `hier=0`
  collapses there; a larger tau was not tried. Unsmoothed variants (tau<=10) over-estimate 1.6-2.6x because native genes missed by the
  neighbours are attributed to ambient.
* **Oracle markers** (`marker:m=N`): for a cell of species s, the N most abundant ambient genes of the other species; rho_hat =
  reads in the set / (N_c * ambient mass of the set). It uses the true species label, so it is an upper bound for any marker method.
  An atlas-derived marker set (genes an atlas says the cell's lineage does not express and that are ambient-enriched) cannot be built
  here: HEK293T and NIH3T3 are not lineages in the atlas, and in these libraries the only genes that are reliably "non-expressed" are
  the other species' genes, which is the oracle. This test therefore says nothing about atlas markers in real tissue.

## Caveats

* Cell Ranger counts (hg19+mm10, 10x public data), not Singlet's; post-hoc method comparison only. Libraries differ a lot in mixing
  (1.0-4.5% mean truth, 10k has 16% mixed cells excluded); only 7 library-views and 2 cell lines.
* hgmm1k_v3 CR is the same sample as the tuning library, so only 5 libraries are held out. Cell lines are a poor test of "native
  profile from neighbours": the two species are trivially separable, so any cluster-aware method gets the species off-genes for free.
  Real tissues with continuous cell states would favour finer neighbourhoods more than this test can show.
* The truth assumes all cross-species reads are ambient and the pool composition applies to every droplet; true heterotypic doublets
  with minor fraction < 5% are in the truth.
* Full-gene scores are inflated for estimators that read the cross-species genes (shared noise); use the gene-split table for
  comparisons. Gene-split halves the reads in both estimator and truth.
* Cost numbers are single runs on shared nodes (`short-cpu`, one thread for kNN, production t1 = 1 thread); RSS includes loading the
  library's trimmed raw matrix (0.7-2.0 M barcodes). The kNN solver is numpy, not optimised (no numba / C++); a compiled solver would
  be several times faster but not obviously 100x. CellBender was not rerun (previous 1k_v3: Spearman 0.60, ratio 1.40).
* VAE: one seed, no tuning.

## Cost (one CPU thread, one process per method)

| method | library (cells, barcodes) | wall s | peak RSS GB |
|---|---|---|---|
| production (harness, 1 thread) | 5k_v3 (5,366; 1.20 M) | 4.8 | 0.9 |
| production (harness, 1 thread) | 10k_v3 (11,456; 1.98 M) | 10.7 (6.4 at 16 threads) | 1.8 |
| DecontX default / with background | 5k_v3 | 101 / 75 | n/a |
| DecontX default / with background | 10k_v3 | 292 / 174 | n/a |
| latent-kNN PCA k=30, hier | 5k_v3 | 115 | 3.7 |
| latent-kNN PCA k=30, hier | 10k_v3 | 274 | 7.2 |
| latent-kNN PCA k=30, no hier | 5k_v3 / 10k_v3 | 55 / 129 | 2.3 / 4.0 |
| cluster centroid | 5k_v3 / 10k_v3 | 116 / 264 | 2.8 / 5.5 |
| oracle markers (100) | 5k_v3 / 10k_v3 | 0.45 / 1.0 | 1.3 / 2.4 |

Per-sample estimate for a 10k-cell sample on CPU: production ~11 s; latent-kNN ~2-4.5 min; DecontX 3-5 min. The VAE adds ~1 min of
GPU time per sample (H100) on top of the kNN stage.

## Commands

```
probe.sbatch      HEAD-probe of candidate URLs
prep.sbatch       prep_lib.py LIB   (array 0-6; download raw+filtered, build X.npz, species by ENSMUSG, cell calls)
external.sbatch   dump_inputs.py; harness.cpp (g++ -std=c++20 -O2 -pthread -I$PROD/include) threads 1 and 16; decontx.R  (array 0-20: lib x {full,A,B})
score.sbatch      score_ext.py LIB full full / A B / B A
tune.sbatch, tune2.sbatch   run_method.py on hgmm1k_v3_singlet (grid)
vae.sbatch        vae_latent.py (H100)
eval.sbatch       run_method.py LIB {full full | A B | B A} <fixed spec list>    (array 0-6)
cost.sbatch       one process per method, 1 CPU
ceiling.sbatch    ceiling.py (truth noise: split-half, reliability)
python aggregate.py   (local; reads results/raw/*.csv)
```
All jobs `--account=debruinz`, `short-cpu` (VAE `short-gpu`), outputs under `$WORK/latent_qc/ambient_truth/` on Clipper.
