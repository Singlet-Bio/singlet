# w3_coexp: within-cell co-expression (hub genes) as codec context

Status: IN PROGRESS (stage 1 running). Owner: w3_coexp agent. Updated as results land; sections marked TODO are not yet measured.

## 0. Question and plan

Every codec so far models features as independent given the column depth (plus in-file adaptation of the per-feature rate).
The remaining structure in a called cell is the dependence between features of the same cell. The decoder can condition on
genes it already decoded in this cell, so this information is NOT bounded by the entropy of a cell-type label.

Stage 1 (python, `tools/w3_coexp_prep.py`, `tools/w3_coexp_s1.py`): gene level, called cells, presence only. Baseline =
per-gene x depth-bucket Bernoulli (16 log buckets of interval nnz, in-file, class pooled + shrunk, best shrink on the scored
half). Conditional = baseline logit + delta_g + int8 weights x centred hub presence, on the top-G targets (G = 2048 / 4096), K hubs
(K = 16 / 64 / 256) chosen by greedy variance reduction (pivoted Cholesky of the depth-residual correlation matrix) or by
raw presence variance. Two out-of-sample regimes: (i) weights trained on other studies (leave-one-study-out pool of
corpus_train + the other corpus files of the same build), (ii) weights trained on one half of this file and scored on the other
half (cross-fitted; transmitted weights counted as side info). In-sample is reported only as a labelled upper bound.
Decision rule from the brief: if the out-of-sample gain is < 0.3 bits/nnz on the PBMC/T-cell files, stop after writing up.

Bookkeeping facts used throughout (measured, `results/w3_coexp_prep.log`): gene-level nnz of called cells is 87-99% of the
interval-level nnz (pbmc_v2_typical: 2.224 M gene entries vs 2.402 M interval entries), so a saving per gene-level decision is
within ~10% of the same saving per interval-level nnz. All gains below are given per INTERVAL nnz of the file's called cells
(the unit of c_r2's 5.5-6.0 bits/nnz) and per gene nnz.

## Results

TODO (stage 1 table).
