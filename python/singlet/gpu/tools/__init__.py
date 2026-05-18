# SPDX-License-Identifier: MIT
"""
singlet.gpu.tools — GPU-native analysis tools.

Exposes (drop-in replacements for scanpy.tl.*):

    leiden(adata, ...)              — Leiden community detection
    umap(adata, ...)                — UMAP dimensionality reduction
    rank_genes_groups(adata, ...)   — Differential expression
    score_genes(adata, ...)         — Marker gene scoring
    celltypist_predict(adata, ...)  — Reference-based cell type annotation

Underlying C++ cycles:
    - cycle 9  (graph/leiden.h)          → leiden
    - cycle 10 (embed/umap.h)            → umap
    - cycle 11 (de/{wilcoxon,ttest}.h)   → rank_genes_groups
    - cycle 12 (anno/{marker_score,      → score_genes, celltypist_predict
                       reference_map}.h)
"""

from .leiden import leiden
from .umap import umap
from .rank_genes_groups import rank_genes_groups
from .markers import score_genes, celltypist_predict

__all__ = [
    "leiden",
    "umap",
    "rank_genes_groups",
    "score_genes",
    "celltypist_predict",
]
