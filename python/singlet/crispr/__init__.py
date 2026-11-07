# SPDX-License-Identifier: MIT
"""CRISPR guide-capture libraries as a Singlet modality.

* :mod:`singlet.crispr.guides`: guide libraries (Cell Ranger feature references, author tables) and the protospacer
  geometry in R2, measured from the reads.
* :mod:`singlet.crispr.count`: guide UMIs per (cell, guide) on the ADT counting core (Cell Ranger feature-barcoding logic).
* :mod:`singlet.crispr.assign`: per-cell guide calls (ambient-calibrated test; Cell Ranger GMM and Poisson-Gaussian
  mixture for comparison).
* :mod:`singlet.crispr.qc`: library QC with pass/fail thresholds.
* :mod:`singlet.crispr.members`: the members (``crispr_counts.1pz``, ``crispr_guides.json``).

Design and benchmark: ``docs/vdj-crispr-modalities.md``.
"""

CRISPR_SCHEMA = "1"
