# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.de — GPU-native differential expression analysis.

Exposes:
    pseudobulk_de — donor-aware pseudobulk DE via cycle-17 NB GLM kernel.

Underlying C++ cycles:
    - cycle 9  (de/*.h) — Wilcoxon, t-test, pseudobulk NB GLM (reduce/de).
    - cycle 17 (de/donor_pseudobulk.h) — donor-aware pseudobulk NB GLM.

This module extends the cycle-9 DE namespace with the donor-aware pseudobulk
path introduced in cycle 17.  The cycle-9 rank_genes_groups / Wilcoxon wrappers
live in ``singlet_gpu.tools`` (cycles 19/21).

CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE: ``_core.donor_pseudobulk_de``
must be exposed by the cycle-22 pybind11 binding extension.
"""

from .pseudobulk import pseudobulk_de

__all__ = ["pseudobulk_de"]
