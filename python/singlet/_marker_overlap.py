# SPDX-License-Identifier: MIT
"""Marker gene overlap analysis."""

from __future__ import annotations

import pandas as pd
from anndata import AnnData


def marker_gene_overlap(
    adata: AnnData,
    reference_markers: dict[str, list[str]],
    *,
    key: str = "rank_genes_groups",
    n_genes: int = 100,
    method: str = "overlap_count",
    normalize: str | None = None,
) -> pd.DataFrame:
    """Compare DE results with known marker gene sets.

    Useful for automatic cell type annotation by comparing DE markers
    against reference signatures.

    Parameters
    ----------
    adata
        Annotated data matrix with DE results.
    reference_markers
        Dictionary mapping reference cell type names to gene lists.
    key
        Key in .uns containing DE results.
    n_genes
        Number of top DE genes per group to compare.
    method
        Comparison method:
        - 'overlap_count': Number of overlapping genes
        - 'overlap_coef': Overlap coefficient (|A∩B| / min(|A|,|B|))
        - 'jaccard': Jaccard index (|A∩B| / |A∪B|)
    normalize
        Normalization: None, 'reference' (by ref size), or 'data' (by n_genes).

    Returns
    -------
    DataFrame with groups as rows and reference types as columns.
    Values are the overlap metric.
    """
    if key not in adata.uns:
        raise KeyError(f"'{key}' not found in .uns. Run singlet.rank_genes_groups() first.")

    de_results = adata.uns[key]
    names = de_results["names"]
    if hasattr(names, "dtype") and names.dtype.names is not None:
        # Structured numpy array (scanpy-style)
        de_groups = list(names.dtype.names)
    else:
        # Plain dict (singlet-style)
        de_groups = list(names.keys())

    # Build overlap matrix
    rows = []
    for group in de_groups:
        top_genes = set(names[group][:n_genes])
        row = {}
        for ref_name, ref_genes in reference_markers.items():
            ref_set = set(ref_genes)
            overlap = top_genes & ref_set

            if method == "overlap_count":
                score = len(overlap)
            elif method == "overlap_coef":
                min_size = min(len(top_genes), len(ref_set))
                score = len(overlap) / max(min_size, 1)
            elif method == "jaccard":
                union = top_genes | ref_set
                score = len(overlap) / max(len(union), 1)
            else:
                raise ValueError(
                    f"Unknown method '{method}'. Use 'overlap_count', 'overlap_coef', or 'jaccard'."
                )

            if normalize == "reference":
                score = score / max(len(ref_set), 1)
            elif normalize == "data":
                score = score / max(n_genes, 1)

            row[ref_name] = score
        rows.append(row)

    return pd.DataFrame(rows, index=de_groups)
