# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.enrichment — GPU-native gene-set enrichment analysis.

Exposes:
    run_gsea   — preranked GSEA via cycle-13 fgsea kernel (decoupleR-compatible).
    run_aucell — AUCell scoring via cycle-13 aucell kernel (decoupleR-compatible).

Underlying C++ cycle: cycle 13 (gsea/fgsea.h, gsea/aucell.h).

decoupleR parity
----------------
Both functions accept ``net`` as a ``pd.DataFrame`` with ``source`` and
``target`` columns (decoupleR convention).  Results are written to
``adata.obs`` (GSEA) or ``adata.obsm`` (AUCell) using decoupleR-compatible
key names.

CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE: ``_core.fgsea`` and
``_core.aucell`` are expected to have been added by the cycle-22 binding
extension.  Both wrappers raise ``AttributeError`` with this tag if the
bindings are missing.
"""

from .gsea import run_gsea
from .aucell import run_aucell

__all__ = ["run_gsea", "run_aucell"]
