# Counting parity: B18 and B17 (2026-11-16)

Development libraries only (nuclei2k_v2, neuron1k_v3, pbmc1k_v2). Comparators: STARsolo Gene, GeneFull and Velocyto from STAR 2.7.11b. Scorer: `run_capability.py`. Sweep tooling: `bench/counting/`.

## B18 nuclei exonic and total counts: already passes

The failure registered in LAUNCH-ORACLES came from the older molecule rules (7053). The shipped default (61837) passes every exonic and total bar on nuclei2k_v2. The multi-gene UMI rule (bit 128) is required: dropping it (61709) brings the failures back. No engine change is needed.

## B17 unspliced gene correlation: not closed

| library | default (61837) | STARsolo Velocyto | bar |
|---|---|---|---|
| nuclei2k_v2 | 0.9610 | 0.9644 | fail |
| neuron1k_v3 | 0.9857 | 0.9869 | fail |
| pbmc1k_v2 | 0.9195 | 0.9189 | pass |

Tried: bits 1024, 2048, 512 and 1024+512 (no change in outcome), and a new Velocyto-style labeling bit (65536: a molecule touching both intron and exon-only reads becomes Ambiguous, as in STARsolo). The new bit left unspliced gene correlation unchanged on nuclei2k_v2 (0.9610) and neuron1k_v3 (0.9857) and made unspliced UMI-ratio distance worse on pbmc1k_v2, so it was dropped from the PR.

Remaining gap sources, none cheap: Velocyto classifies per transcript intersection (exon model versus intron model), UMI correction there starts from the Gene feature, and STARsolo's multi-gene rule also checks uncorrected UMI read counts. Matching this means a second labeling pipeline, not a rule tweak.

Per D-111, B17 turns diagnostic once its ground-truth counterpart passes. B9 passes (F1 0.713 versus Velocyto 0.706). Recommendation: report B17 as diagnostic in the preprint.

## Identity impact

No engine code changes in this PR, so the 10x counts stay byte-identical to the frozen build. As a check, the build with the (now removed) default-off bit was compared against the v1.4.0 CPU reference on all 8 gate samples plus pbmc5p: every count matrix and count file was identical. Only logs, timings and git SHA differed, plus one timing field in the HLA manifest, which the harness flags because it is not on its allowlist.
