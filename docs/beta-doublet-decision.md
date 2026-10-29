# Beta decision: expression doublets and the #43 combiner (2026-10-29)

Evidence: `study/latent-qc` (#43, `docs/latent-doublet-ambient.md`, `bench/latent_qc/doublets/results/`). Real experimental
labels only (genotype, hashtag, MULTI-seq, species), 17 datasets, leave-one-study-out training; no reported number is on
simulated doublets.

| method | mean AUPRC (17 datasets) | Kang 2.1 on Singlet counts (AUROC / AUPRC) | cost / 10k cells |
|---|---|---|---|
| production `detect_doublets` | 0.501 | 0.919 / 0.632 | 2 s |
| scDblFinder (R) | 0.559 | 0.927 / 0.642 | 44 s, needs R |
| combo_prod_atlas (production score + 4 atlas features) | 0.557 | 0.935 / 0.698 | about 10 s |
| combo_all (adds scDblFinder, Solo, Scrublet scores) | 0.572 | | needs R and a VAE |
| genotype doublets (`prob_doublet`), pools with >= 2 donors | | 0.969 / 0.897 | in the run |

## Decision
1. **Genotype doublets stay the doublet call whenever a pool has two or more donors** (recall 0.89 on doublets that look like
   one cell type, expression methods 0.47-0.59).
2. **Expression doublets for single-donor samples: adopt `combo_prod_atlas` as a registered, versioned `cell_feature`**
   (CELL-ARCHITECTURE: computed in batch on stored counts next to the cell-type inference, never in the engine, never in a
   Worker). It beats production by +0.060 median AUPRC (15/17, p = 0.0004) and equals scDblFinder (-0.001, p = 0.64) without
   an R stack. It does not use `combo_all`: that needs R scDblFinder in the campaign.
3. **The 0.56 target is not met by margin**: mean 0.557 vs scDblFinder 0.559 is a statistical tie, and the gate metric
   (match_or_beat scDblFinder, AUPRC) should carry a tolerance justified by the bootstrap CI over the 17 datasets, not 0.
   If the release gate must exceed 0.56 outright, the only validated option is `combo_all` (0.572), which costs R.
4. Not adopted: Solo-style VAE (worse on 16/17), NNLS two-cell-type test (AUROC 0.62-0.70), atlas-only scores (tie with
   production).

## Not yet done (needs implementation, not a decision)
* The combiner is not in any binary or batch job; the atlas features need `singlet.celltype_nn` (PR #32). A batch job that
  writes the `doublet_score_v2` feature per sample, a held-out check on the Kang lanes **not** in the training set (the
  training set contains them: report leave-one-study-out numbers only), and the `bench/public` doublets capability file
  (currently missing; hgmm libraries have < 30 truth doublets, so Kang and the Xi & Li sets must carry the gate).
