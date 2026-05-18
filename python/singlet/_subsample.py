# SPDX-License-Identifier: MIT
"""Subsampling utilities for AnnData objects.

Provides singlet.subsample() for random downsampling of cells,
useful for quickly exploring large datasets or balancing groups.
"""

from __future__ import annotations

from typing import Optional


def subsample(
    adata,
    *,
    n_obs: Optional[int] = None,
    fraction: Optional[float] = None,
    random_state: int = 0,
    copy: bool = True,
):
    """Subsample cells from an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    n_obs : int or None
        Number of cells to keep. Mutually exclusive with fraction.
    fraction : float or None
        Fraction of cells to keep (0 < fraction <= 1).
        Mutually exclusive with n_obs.
    random_state : int, default 0
        Random seed for reproducibility.
    copy : bool, default True
        If True, returns a subsampled copy.
        If False, modifies adata in-place and returns None.

    Returns
    -------
    anndata.AnnData or None
        Subsampled AnnData (if copy=True), or None (if copy=False).

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")  # large dataset
    >>> small = singlet.subsample(adata, n_obs=1000)
    >>> small.shape[0]
    1000
    """
    import numpy as np

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(f"subsample() requires an AnnData object, got {type(adata).__name__}")

    if n_obs is not None and fraction is not None:
        raise ValueError("Specify either n_obs or fraction, not both.")

    if n_obs is None and fraction is None:
        raise ValueError("Must specify either n_obs or fraction.")

    n_total = adata.shape[0]

    if fraction is not None:
        if not (0 < fraction <= 1):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        n_obs = max(1, int(n_total * fraction))

    n_obs = min(n_obs, n_total)

    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_total, size=n_obs, replace=False)
    indices.sort()

    if copy:
        return adata[indices].copy()
    else:
        adata._inplace_subset_obs(indices)
        return None
