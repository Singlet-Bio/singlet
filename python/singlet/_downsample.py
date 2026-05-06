"""Count downsampling."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def downsample_counts(
    adata: AnnData,
    *,
    total_counts: int | None = None,
    counts_per_cell: int | None = None,
    random_state: int = 0,
    replace: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """Downsample counts from a count matrix.

    If `total_counts` is specified, the total number of counts is downsampled.
    If `counts_per_cell` is specified, each cell is downsampled independently.
    Exactly one must be specified.

    Parameters
    ----------
    adata
        Annotated data matrix (counts).
    total_counts
        Target total counts across all cells.
    counts_per_cell
        Target counts per cell. Cells with fewer counts are unchanged.
    random_state
        Random seed.
    replace
        Sample with replacement.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True. Modifies .X in place.
    """
    from scipy.sparse import csr_matrix, issparse

    if total_counts is None and counts_per_cell is None:
        raise ValueError("Specify either total_counts or counts_per_cell.")
    if total_counts is not None and counts_per_cell is not None:
        raise ValueError("Specify only one of total_counts or counts_per_cell.")

    adata = adata.copy() if copy else adata
    rng = np.random.default_rng(random_state)

    if issparse(adata.X):
        X = np.asarray(adata.X.todense())
    else:
        X = np.asarray(adata.X).copy()

    X = X.astype(np.float64)

    if counts_per_cell is not None:
        for i in range(X.shape[0]):
            cell_total = int(X[i].sum())
            if cell_total <= counts_per_cell:
                continue
            X[i] = _downsample_cell(X[i], counts_per_cell, rng, replace)
    else:
        current_total = int(X.sum())
        if current_total > total_counts:
            flat = X.flatten()
            probs = flat / flat.sum()
            sampled = rng.multinomial(total_counts, probs)
            X = sampled.reshape(X.shape).astype(np.float64)

    if issparse(adata.X):
        adata.X = csr_matrix(X.astype(np.float32))
    else:
        adata.X = X.astype(np.float32)

    return adata if copy else None


def _downsample_cell(cell_counts: np.ndarray, target: int, rng, replace: bool) -> np.ndarray:
    """Downsample a single cell's counts."""
    total = cell_counts.sum()
    if total == 0:
        return cell_counts
    probs = cell_counts / total
    sampled = rng.multinomial(target, probs)
    return sampled.astype(np.float64)
