"""Preprocessing recipes (standard workflows)."""

from __future__ import annotations

from anndata import AnnData


def recipe_seurat(
    adata: AnnData,
    *,
    min_genes: int = 200,
    min_cells: int = 3,
    n_top_genes: int = 2000,
    target_sum: float = 10000,
    max_value: float = 10,
    copy: bool = False,
) -> AnnData | None:
    """Seurat-style preprocessing recipe.

    Steps: filter_cells → filter_genes → normalize → log1p →
    highly_variable_genes → scale (with clipping).

    Parameters
    ----------
    adata
        Annotated data matrix (raw counts).
    min_genes
        Minimum genes per cell.
    min_cells
        Minimum cells per gene.
    n_top_genes
        Number of highly variable genes.
    target_sum
        Target sum for normalization.
    max_value
        Maximum value after scaling (clip).
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True.
    """
    import singlet

    adata = adata.copy() if copy else adata

    singlet.filter_cells(adata, min_genes=min_genes, inplace=True)
    singlet.filter_genes(adata, min_cells=min_cells, inplace=True)
    singlet.normalize(adata, target_sum=target_sum)
    singlet.highly_variable_genes(adata, n_top_genes=n_top_genes)
    singlet.scale(adata, max_value=max_value)

    return adata if copy else None


def recipe_zheng17(
    adata: AnnData,
    *,
    n_top_genes: int = 1000,
    log: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """Zheng et al. 2017 preprocessing recipe.

    Steps: filter_genes → normalize (to median) → log1p →
    highly_variable_genes → subset to HVGs.

    Used in the original 10X Genomics PBMC dataset analyses.

    Parameters
    ----------
    adata
        Annotated data matrix (raw counts).
    n_top_genes
        Number of highly variable genes to keep.
    log
        Whether to log-transform.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True. Subsets to HVGs.
    """
    import numpy as np
    from scipy.sparse import issparse

    import singlet

    adata = adata.copy() if copy else adata

    singlet.filter_genes(adata, min_cells=1, inplace=True)

    # Normalize to median total count
    if issparse(adata.X):
        total_counts = np.asarray(adata.X.sum(axis=1)).flatten()
    else:
        total_counts = np.sum(adata.X, axis=1)

    median_count = float(np.median(total_counts))
    singlet.normalize(adata, target_sum=median_count, log=log)

    singlet.highly_variable_genes(adata, n_top_genes=n_top_genes)

    # Subset to HVGs
    if "highly_variable" in adata.var.columns:
        adata._inplace_subset_var(adata.var["highly_variable"].values)

    return adata if copy else None
