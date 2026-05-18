# SPDX-License-Identifier: MIT
"""Gene set scoring for AnnData objects.

Provides singlet.score_genes() — scores cells based on expression of a gene
set relative to a control set, similar to Seurat's AddModuleScore.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


def score_genes(
    adata,
    gene_list: list[str],
    *,
    score_name: str = "score",
    ctrl_size: int = 50,
    n_bins: int = 25,
    layer: Optional[str] = None,
    inplace: bool = True,
) -> Optional["np.ndarray"]:
    """Score cells by average expression of a gene set.

    Uses the method from Tirosh et al. (2016): for each gene in the set,
    a control set of genes with similar average expression is selected.
    The score is the mean expression of the gene set minus the mean
    expression of the control set.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Should be log-normalized.
    gene_list : list[str]
        List of gene names to score.
    score_name : str, default "score"
        Name of the score column added to adata.obs (if inplace=True).
    ctrl_size : int, default 50
        Number of control genes per expression bin.
    n_bins : int, default 25
        Number of bins for grouping genes by expression level.
    layer : str or None, default None
        Layer to use for expression values. If None, uses adata.X.
    inplace : bool, default True
        If True, stores score in adata.obs[score_name].
        If False, returns score array.

    Returns
    -------
    numpy.ndarray or None
        Score array (n_cells,) if inplace=False.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> s_genes = ["MCM5", "PCNA", "TYMS", "MCM2", "MCM4"]
    >>> singlet.score_genes(adata, s_genes, score_name="S_score")
    >>> adata.obs["S_score"].head()
    """
    import numpy as np
    import scipy.sparse as sp

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(f"score_genes() requires an AnnData object, got {type(adata).__name__}")

    if not gene_list:
        raise ValueError("gene_list must not be empty.")

    var_names = list(adata.var_names)

    # Filter to genes present in the dataset
    gene_list_valid = [g for g in gene_list if g in var_names]
    if not gene_list_valid:
        raise ValueError("None of the genes in gene_list are found in adata.var_names.")

    # Get expression matrix
    if layer is not None and layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X

    # Compute mean expression per gene
    if sp.issparse(X):
        gene_means = np.asarray(X.mean(axis=0)).ravel()
    else:
        gene_means = np.mean(X, axis=0).ravel()

    # Bin genes by mean expression
    n_genes_total = len(var_names)
    n_bins = min(n_bins, n_genes_total)
    sorted_indices = np.argsort(gene_means)
    bin_size = max(1, n_genes_total // n_bins)

    gene_to_bin = np.zeros(n_genes_total, dtype=int)
    for i, idx in enumerate(sorted_indices):
        gene_to_bin[idx] = min(i // bin_size, n_bins - 1)

    # Build control gene set
    rng = np.random.default_rng(0)
    ctrl_genes_idx = set()
    gene_list_idx = [var_names.index(g) for g in gene_list_valid]

    for gene_idx in gene_list_idx:
        gene_bin = gene_to_bin[gene_idx]
        # Find all genes in same bin (excluding gene_list genes)
        bin_genes = [
            i for i in range(n_genes_total) if gene_to_bin[i] == gene_bin and i not in gene_list_idx
        ]
        if len(bin_genes) == 0:
            continue
        # Sample ctrl_size genes from this bin
        n_sample = min(ctrl_size, len(bin_genes))
        sampled = rng.choice(bin_genes, size=n_sample, replace=False)
        ctrl_genes_idx.update(sampled)

    ctrl_genes_idx = list(ctrl_genes_idx)

    # Compute scores
    if sp.issparse(X):
        score_set = np.asarray(X[:, gene_list_idx].mean(axis=1)).ravel()
        if ctrl_genes_idx:
            score_ctrl = np.asarray(X[:, ctrl_genes_idx].mean(axis=1)).ravel()
        else:
            score_ctrl = np.zeros(adata.n_obs)
    else:
        score_set = X[:, gene_list_idx].mean(axis=1).ravel()
        if ctrl_genes_idx:
            score_ctrl = X[:, ctrl_genes_idx].mean(axis=1).ravel()
        else:
            score_ctrl = np.zeros(adata.n_obs)

    scores = score_set - score_ctrl

    if inplace:
        adata.obs[score_name] = scores
        return None
    return scores
