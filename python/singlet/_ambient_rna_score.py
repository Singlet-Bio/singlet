# SPDX-License-Identifier: MIT
"""Ambient RNA contamination scoring per cell.

Provides singlet.ambient_rna_score() — estimate ambient RNA contamination
by comparing each cell's expression profile to an ambient (empty droplet) profile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from anndata import AnnData


def ambient_rna_score(
    adata: AnnData,
    *,
    empty_droplet_profile: np.ndarray | None = None,
    n_top_ambient: int = 100,
    method: str = "cosine",
) -> AnnData:
    """Estimate ambient RNA contamination score per cell.

    Compares each cell's expression profile to an estimated ambient RNA
    profile (from empty droplets). Higher scores indicate more contamination.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix (raw counts preferred).
    empty_droplet_profile : numpy.ndarray or None, default None
        Pre-computed ambient RNA profile (1D array, one value per gene).
        If None, estimates the ambient profile from genes with highest
        dropout rate and lowest variance (top ``n_top_ambient`` genes).
    n_top_ambient : int, default 100
        Number of top ambient-associated genes to use when estimating
        the ambient profile automatically.
    method : str, default 'cosine'
        Similarity method: 'cosine' (cosine similarity) or 'correlation'
        (Pearson correlation with the ambient profile).

    Returns
    -------
    anndata.AnnData
        The input adata with ``adata.obs['ambient_rna_score']`` added.
        Score is in [0, 1] for cosine, or [-1, 1] for correlation.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.ambient_rna_score(adata, method='cosine')
    >>> adata.obs['ambient_rna_score'].describe()
    """
    import numpy as np
    import scipy.sparse as sp

    if method not in ("cosine", "correlation"):
        msg = f"method must be 'cosine' or 'correlation', got {method!r}"
        raise ValueError(msg)

    # Get expression matrix as dense
    mat = adata.X
    if sp.issparse(mat):
        mat_dense = np.asarray(mat.todense(), dtype=np.float64)
    else:
        mat_dense = np.asarray(mat, dtype=np.float64)

    n_cells, n_genes = mat_dense.shape

    # Estimate ambient profile if not provided
    if empty_droplet_profile is not None:
        ambient_profile = np.asarray(empty_droplet_profile, dtype=np.float64).ravel()
        if ambient_profile.shape[0] != n_genes:
            msg = (
                f"empty_droplet_profile length ({ambient_profile.shape[0]}) "
                f"does not match number of genes ({n_genes})"
            )
            raise ValueError(msg)
    else:
        # Estimate ambient profile from genes with high dropout + low variance
        # Dropout rate: fraction of cells with zero expression
        if sp.issparse(adata.X):
            nonzero_per_gene = np.asarray((adata.X > 0).sum(axis=0)).ravel()
        else:
            nonzero_per_gene = np.asarray((mat_dense > 0).sum(axis=0)).ravel()
        dropout_rate = 1.0 - (nonzero_per_gene / n_cells)

        # Variance per gene
        gene_var = np.var(mat_dense, axis=0)
        # Normalize variance to [0, 1] for ranking
        var_max = gene_var.max()
        if var_max > 0:
            gene_var_norm = gene_var / var_max
        else:
            gene_var_norm = gene_var

        # Score: high dropout + low variance → likely ambient
        # ambient_gene_score = dropout_rate - gene_var_norm
        ambient_gene_score = dropout_rate - gene_var_norm

        # Select top ambient genes
        n_select = min(n_top_ambient, n_genes)
        top_ambient_idx = np.argsort(ambient_gene_score)[::-1][:n_select]

        # Ambient profile: mean expression across all cells for these genes
        # (ambient RNA is ubiquitous low-level expression)
        ambient_profile = np.zeros(n_genes, dtype=np.float64)
        ambient_profile[top_ambient_idx] = np.mean(mat_dense[:, top_ambient_idx], axis=0)

    # Compute similarity of each cell to ambient profile
    if method == "cosine":
        # Cosine similarity: dot(cell, ambient) / (||cell|| * ||ambient||)
        ambient_norm = np.linalg.norm(ambient_profile)
        if ambient_norm == 0:
            scores = np.zeros(n_cells, dtype=np.float64)
        else:
            cell_norms = np.linalg.norm(mat_dense, axis=1)
            # Avoid division by zero
            cell_norms = np.where(cell_norms == 0, 1.0, cell_norms)
            dots = mat_dense @ ambient_profile
            scores = dots / (cell_norms * ambient_norm)
            # Clip to [0, 1] (negative cosine similarity doesn't make sense here)
            scores = np.clip(scores, 0.0, 1.0)
    else:
        # Pearson correlation
        ambient_centered = ambient_profile - ambient_profile.mean()
        ambient_std = np.std(ambient_profile)
        if ambient_std == 0:
            scores = np.zeros(n_cells, dtype=np.float64)
        else:
            cell_means = mat_dense.mean(axis=1, keepdims=True)
            cell_centered = mat_dense - cell_means
            cell_stds = np.std(mat_dense, axis=1)
            cell_stds = np.where(cell_stds == 0, 1.0, cell_stds)
            corr = (cell_centered @ ambient_centered) / (n_genes * cell_stds * ambient_std)
            scores = corr

    adata.obs["ambient_rna_score"] = scores

    return adata
