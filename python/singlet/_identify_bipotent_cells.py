# SPDX-License-Identifier: MIT
"""Identify bipotent/multipotent cells co-expressing multiple lineage markers.

Provides singlet.identify_bipotent_cells() — find cells in transition states
by scoring co-expression of lineage-specific gene sets and detecting cells
with high entropy or balanced lineage scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


def identify_bipotent_cells(
    adata: "AnnData",
    lineage_genes: dict[str, list[str]],
    *,
    threshold: float = 0.3,
    method: str = "entropy",
    use_rep: str = "X_pca",
    layer: str | None = None,
) -> "AnnData":
    """Identify cells co-expressing markers of multiple lineages.

    Finds cells in transition/bipotent states by scoring co-expression of
    lineage-specific genes and detecting cells where multiple lineage programs
    are simultaneously active.

    Parameters
    ----------
    adata
        Annotated data matrix with gene expression data.
    lineage_genes
        Dictionary mapping lineage names to lists of marker genes.
        E.g., ``{"erythroid": ["HBB", "HBA1"], "myeloid": ["LYZ", "CST3"]}``.
    threshold
        Score threshold above which a cell is classified as bipotent.
        For 'entropy' method: fraction of max possible entropy (0-1).
        For 'score_ratio' method: minimum ratio of 2nd/1st lineage score.
    method
        Scoring method. One of:
        - 'entropy': high Shannon entropy of lineage scores = multipotent
        - 'score_ratio': ratio of top-2 lineage scores close to 1 = bipotent
    use_rep
        Representation in ``.obsm`` to use for neighbor lookups (unused
        currently but reserved for future smoothing).
    layer
        Layer to use for gene expression. If None, uses ``.X``.

    Returns
    -------
    AnnData
        Input ``adata`` with added fields:

        - ``adata.obs['bipotent_score']``: continuous bipotency score (0-1)
        - ``adata.obs['top_lineage']``: lineage with highest score per cell
        - ``adata.obs['is_bipotent']``: boolean, True if score > threshold

    Examples
    --------
    >>> import singlet
    >>> lineages = {
    ...     "erythroid": ["HBB", "HBA1", "HBA2"],
    ...     "myeloid": ["LYZ", "CST3", "AIF1"],
    ... }
    >>> adata = singlet.identify_bipotent_cells(adata, lineages)
    >>> adata.obs["is_bipotent"].sum()
    """
    import scipy.sparse as sp

    if method not in ("entropy", "score_ratio"):
        msg = f"method must be 'entropy' or 'score_ratio', got {method!r}"
        raise ValueError(msg)

    if not lineage_genes:
        msg = "lineage_genes must be a non-empty dictionary"
        raise ValueError(msg)

    if len(lineage_genes) < 2:
        msg = "lineage_genes must contain at least 2 lineages"
        raise ValueError(msg)

    # Get expression matrix
    if layer is not None:
        expr = adata.layers[layer]
    else:
        expr = adata.X

    if sp.issparse(expr):
        expr = expr.toarray()
    expr = np.asarray(expr, dtype=np.float64)

    # Build gene name -> index mapping
    gene_idx = {g: i for i, g in enumerate(adata.var_names)}

    # Score each lineage per cell
    lineage_names = list(lineage_genes.keys())
    n_lineages = len(lineage_names)
    scores = np.zeros((adata.n_obs, n_lineages), dtype=np.float64)

    for col, (lineage, genes) in enumerate(lineage_genes.items()):
        valid_idx = [gene_idx[g] for g in genes if g in gene_idx]
        if valid_idx:
            lineage_expr = expr[:, valid_idx]
            # Mean expression of lineage genes per cell
            scores[:, col] = lineage_expr.mean(axis=1)

    # Normalize scores to sum to 1 per cell (probability distribution)
    row_sums = scores.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    probs = scores / row_sums

    if method == "entropy":
        # Shannon entropy of lineage score distribution
        # Max entropy = log(n_lineages) when all equal
        max_entropy = np.log(n_lineages)
        # Compute entropy, handling zeros
        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.where(probs > 0, np.log(probs), 0.0)
        entropy = -np.sum(probs * log_probs, axis=1)
        # Normalize to [0, 1]
        bipotent_score = entropy / max_entropy if max_entropy > 0 else entropy
    else:
        # score_ratio: ratio of 2nd highest to highest
        sorted_probs = np.sort(probs, axis=1)
        top1 = sorted_probs[:, -1]
        top2 = sorted_probs[:, -2]
        # Ratio of 2nd/1st (1.0 means perfectly balanced)
        with np.errstate(divide="ignore", invalid="ignore"):
            bipotent_score = np.where(top1 > 0, top2 / top1, 0.0)

    # Determine top lineage per cell
    top_lineage_idx = np.argmax(scores, axis=1)
    top_lineage = pd.Categorical(
        [lineage_names[i] for i in top_lineage_idx],
        categories=lineage_names,
    )

    # Store results
    adata.obs["bipotent_score"] = bipotent_score.astype(np.float32)
    adata.obs["top_lineage"] = top_lineage
    adata.obs["is_bipotent"] = bipotent_score >= threshold

    return adata
