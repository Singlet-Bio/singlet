# Expression doublets: gate result (2026-11-11)

Gate: `bench/calls-sota/gates/doublets.md` (committed before any held-out run). Scorer: `doublet/score_v2` at origin/main b7130d38. Comparator:
scDblFinder, defaults. Raw outputs: `$W/bench/calls-sota/doublets/` on Clipper (`*.result.json`, `*.scores.tsv.gz`, `posthoc.json`).

## Leakage check

| Question | Answer |
|---|---|
| Truth or test labels used by any feature? | No. `compute_inputs` takes counts, gene names, species and the engine score. |
| Packaged stacker trained on the evaluation sets? | Yes, on all 17 development datasets. Their in-sample scores are not evidence. Only the leave-one-study-out (LOSO) refits count: mean AUPRC difference to scDblFinder -0.005, 95% CI [-0.022, +0.011] over datasets, [-0.033, +0.005] clustered by study. |
| Does the engine score used at call time match the one used in training? | Yes. `doublet_detect.h` at v1.3.7 is identical to the pinned training commit (`git diff` empty). |
| Atlas encoder overlap with test data? | Earlier nearest-neighbor check on the 17 sets: no cell at cosine 0.98 or above (max 0.84). Not rerun on the two new libraries (limit). |
| Selection effect | Feature set and learner were chosen among eleven variants on the same 17 sets, so the development number is slightly optimistic. |

## Held-out, pre-registered rules (scored once)

| Library | Truth | Doublets / scored | v2 AUPRC | scDblFinder AUPRC | Difference [95% CI] | Structural-method AUROC |
|---|---|---|---|---|---|---|
| CellPlex PBMC 10k | hashtag | 990 / 13,213 | 0.166 | 0.096 | +0.070 [+0.062, +0.080] | 0.32 (scDblFinder), 0.31 (engine score) |
| hgmm 5k v3 | species | 179 / 5,047 | 0.456 | 0.443 | +0.013 [-0.002, +0.029] | 0.47, 0.47 |

Both libraries fail a sanity check: structural methods (scDblFinder, the engine score) should rank true doublets above singlets, and here they
rank them below chance. The fixed rules label low-depth cells as doublets (tag noise in CellPlex, ambient RNA of a barnyard with 75% human
ambient), so the truth is confounded with depth. These rows are therefore not verdict-bearing evidence of superiority.

## Post hoc depth-robust rescoring (written after seeing the above; labeled post hoc)

CellPlex: at least 500 tag UMIs; doublet when the second tag has at least 100 UMIs and 25% of tag UMIs, singlet when it has at most 5%.
hgmm 5k: at least 5,000 gene UMIs; doublet at 20% minor species or more, singlet at 5% or less.

| Library | Doublets / scored | v2 AUPRC | scDblFinder AUPRC | Difference [95% CI] | AUROC v2 / scDblFinder |
|---|---|---|---|---|---|
| CellPlex PBMC 10k | 438 / 10,100 | 0.195 | 0.165 | +0.030 [+0.014, +0.046] | 0.81 / 0.70 |
| hgmm 5k v3 | 78 / 4,735 | 0.949 | 0.952 | -0.003 [-0.036, +0.027] | 0.998 / 0.993 |

## Verdict

Parity numerically met (development LOSO -0.005, held-out differences +0.070 and +0.013 as registered, +0.030 and -0.003 post hoc; no library
below -0.02). Enforcement not met: the pooled study-clustered interval is [-0.033, +0.005], lower bound below the -0.01 margin. Verdict:
**metadata only, hidden by default**. A larger real-truth set (genotype pools such as the Neavin village and scHi-HOST) is the route to the
interval; two new libraries cannot narrow it.

Runtime (engine score plus package inputs, wall seconds times 4 allocated threads, an upper bound): 0.49 (CellPlex, 14,273 cells) and 0.90
(hgmm 5k, 5,370 cells, fixed costs dominate) core-minutes per ten thousand cells, within the 2 core-minute ceiling for all calls; above the
0.5 sub-budget set in the gate for the small library (CPU time was not measured separately). scDblFinder took 88 s and 40 s.
