"""Basic quality-control filtering for AnnData objects.

Provides singlet.filter_cells() and singlet.filter_genes() so users can do
basic QC without needing scanpy installed. These are intentionally simple —
for advanced filtering, use scanpy.pp.filter_cells/filter_genes.
"""

from __future__ import annotations

from typing import Optional


def filter_cells(
    adata,
    *,
    min_genes: Optional[int] = None,
    max_genes: Optional[int] = None,
    min_counts: Optional[int] = None,
    max_counts: Optional[int] = None,
    inplace: bool = False,
):
    """Filter cells by number of detected genes or total counts.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    min_genes : int, optional
        Minimum number of genes detected (non-zero) per cell.
    max_genes : int, optional
        Maximum number of genes detected per cell.
    min_counts : int, optional
        Minimum total counts per cell.
    max_counts : int, optional
        Maximum total counts per cell.
    inplace : bool, default False
        If True, modifies adata in-place and returns None.
        If False, returns a filtered copy.

    Returns
    -------
    anndata.AnnData or None
        Filtered AnnData (if inplace=False), or None (if inplace=True).

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> filtered = singlet.filter_cells(adata, min_genes=200, max_genes=5000)
    >>> filtered.shape[0] < adata.shape[0]  # some cells removed
    True
    """
    import numpy as np
    import scipy.sparse as sp

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(f"filter_cells() requires an AnnData object, got {type(adata).__name__}")

    X = adata.X
    mask = np.ones(X.shape[0], dtype=bool)

    if min_genes is not None or max_genes is not None:
        if sp.issparse(X):
            genes_per_cell = np.diff(X.tocsr().indptr)
        else:
            genes_per_cell = np.count_nonzero(X, axis=1)

        if min_genes is not None:
            mask &= genes_per_cell >= min_genes
        if max_genes is not None:
            mask &= genes_per_cell <= max_genes

    if min_counts is not None or max_counts is not None:
        if sp.issparse(X):
            counts_per_cell = np.asarray(X.sum(axis=1)).ravel()
        else:
            counts_per_cell = X.sum(axis=1)

        if min_counts is not None:
            mask &= counts_per_cell >= min_counts
        if max_counts is not None:
            mask &= counts_per_cell <= max_counts

    if inplace:
        # In-place subsetting
        idx = np.where(mask)[0]
        adata._inplace_subset_obs(idx)
        return None
    else:
        return adata[mask].copy()


def filter_genes(
    adata,
    *,
    min_cells: Optional[int] = None,
    max_cells: Optional[int] = None,
    min_counts: Optional[int] = None,
    max_counts: Optional[int] = None,
    inplace: bool = False,
):
    """Filter genes by number of cells expressing them or total counts.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    min_cells : int, optional
        Minimum number of cells in which the gene must be detected.
    max_cells : int, optional
        Maximum number of cells in which the gene is detected.
    min_counts : int, optional
        Minimum total counts for a gene across all cells.
    max_counts : int, optional
        Maximum total counts for a gene across all cells.
    inplace : bool, default False
        If True, modifies adata in-place and returns None.
        If False, returns a filtered copy.

    Returns
    -------
    anndata.AnnData or None
        Filtered AnnData (if inplace=False), or None (if inplace=True).

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> filtered = singlet.filter_genes(adata, min_cells=3)
    >>> filtered.shape[1] < adata.shape[1]  # some genes removed
    True
    """
    import numpy as np
    import scipy.sparse as sp

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(f"filter_genes() requires an AnnData object, got {type(adata).__name__}")

    X = adata.X
    mask = np.ones(X.shape[1], dtype=bool)

    if min_cells is not None or max_cells is not None:
        if sp.issparse(X):
            cells_per_gene = np.diff(X.tocsc().indptr)
        else:
            cells_per_gene = np.count_nonzero(X, axis=0)

        if min_cells is not None:
            mask &= cells_per_gene >= min_cells
        if max_cells is not None:
            mask &= cells_per_gene <= max_cells

    if min_counts is not None or max_counts is not None:
        if sp.issparse(X):
            counts_per_gene = np.asarray(X.sum(axis=0)).ravel()
        else:
            counts_per_gene = X.sum(axis=0)

        if min_counts is not None:
            mask &= counts_per_gene >= min_counts
        if max_counts is not None:
            mask &= counts_per_gene <= max_counts

    if inplace:
        idx = np.where(mask)[0]
        adata._inplace_subset_var(idx)
        return None
    else:
        return adata[:, mask].copy()
