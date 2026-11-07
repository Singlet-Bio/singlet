# SPDX-License-Identifier: MIT
"""Expression doublet score for single-donor samples: a cross-study gradient-boosted combiner over the production doublet score,
library size and four features of the atlas cell-type network, computed in batch on stored counts and written as the per-sample
feature sidecar ``qc/doublet_score`` (docs/doublet_score.md, docs/CELL-ARCHITECTURE.md section 3).

Genotype doublets (``donor_assignments.tsv`` ``prob_doublet``) stay the doublet call for pools with two or more donors."""

__version__ = "1.0.0"
