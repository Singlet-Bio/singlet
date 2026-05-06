"""Splicing ratio computation.

Provides singlet.splicing_ratio() — compute the fraction of spliced
transcripts per gene per cell from spliced/unspliced count layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def splicing_ratio(
    adata: "AnnData",
    spliced_layer: str = "spliced",
    unspliced_layer: str = "unspliced",
    *,
    min_counts: int = 10,
) -> "AnnData":
    """Compute splicing ratio (fraction spliced) per gene per cell.

    Calculates spliced / (spliced + unspliced) for each gene in each cell,
    masking entries where total counts are below min_counts.

    Parameters
    ----------
    adata
        Annotated data matrix with spliced and unspliced layers.
    spliced_layer
        Name of the layer containing spliced counts. Default 'spliced'.
    unspliced_layer
        Name of the layer containing unspliced counts. Default 'unspliced'.
    min_counts
        Minimum total counts (spliced + unspliced) to compute ratio.
        Entries below this threshold are set to NaN. Default 10.

    Returns
    -------
    AnnData
        Returns adata with:
        - adata.layers['splicing_ratio']: fraction spliced per gene per cell
        - adata.obs['mean_splicing_ratio']: per-cell average splicing ratio

    Raises
    ------
    TypeError
        If adata is not an AnnData object.
    KeyError
        If spliced_layer or unspliced_layer not in adata.layers.
    ValueError
        If min_counts < 0.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.splicing_ratio(adata)
    >>> adata.obs['mean_splicing_ratio'].describe()
    """
    import numpy as np
    import scipy.sparse as sp

    # Validate input
    if not hasattr(adata, "layers"):
        msg = "splicing_ratio requires an AnnData object, got " + type(adata).__name__
        raise TypeError(msg)

    if spliced_layer not in adata.layers:
        msg = f"'{spliced_layer}' not found in adata.layers"
        raise KeyError(spliced_layer)

    if unspliced_layer not in adata.layers:
        msg = f"'{unspliced_layer}' not found in adata.layers"
        raise KeyError(unspliced_layer)

    if min_counts < 0:
        msg = f"min_counts must be >= 0, got {min_counts}"
        raise ValueError(msg)

    # Get data as dense arrays for computation
    spliced = adata.layers[spliced_layer]
    unspliced = adata.layers[unspliced_layer]

    if sp.issparse(spliced):
        spliced = spliced.toarray()
    else:
        spliced = np.asarray(spliced, dtype=np.float64)

    if sp.issparse(unspliced):
        unspliced = unspliced.toarray()
    else:
        unspliced = np.asarray(unspliced, dtype=np.float64)

    spliced = spliced.astype(np.float64)
    unspliced = unspliced.astype(np.float64)

    # Compute total and ratio
    total = spliced + unspliced

    # Initialize ratio with NaN
    ratio = np.full_like(total, np.nan)

    # Only compute where total >= min_counts and total > 0
    mask = (total >= min_counts) & (total > 0)
    ratio[mask] = spliced[mask] / total[mask]

    # Store in layers
    adata.layers["splicing_ratio"] = ratio.astype(np.float32)

    # Per-cell mean splicing ratio (nanmean across genes)
    with np.errstate(all="ignore"):
        mean_ratio = np.nanmean(ratio, axis=1)

    # If all genes are NaN for a cell, nanmean gives NaN — keep as NaN
    adata.obs["mean_splicing_ratio"] = mean_ratio.astype(np.float64)

    return adata
