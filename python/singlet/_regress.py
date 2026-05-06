"""Regression of confounding variables from expression data.

Provides singlet.regress_out() — removes effects of specified variables
(e.g., total_counts, percent_mito) from each gene via linear regression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


def regress_out(
    adata,
    keys: list[str],
    *,
    inplace: bool = True,
) -> Optional["np.ndarray"]:
    """Regress out unwanted sources of variation.

    For each gene, fits a linear model with the specified variables
    and returns residuals + intercept. This removes confounding effects
    (e.g., cell cycle, mitochondrial contamination, library size).

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Should be log-normalized.
    keys : list[str]
        Columns in adata.obs to regress out.
    inplace : bool, default True
        If True, modifies adata.X in place.
        If False, returns corrected matrix.

    Returns
    -------
    numpy.ndarray or None
        Corrected expression matrix if inplace=False.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.regress_out(adata, ["total_counts", "pct_counts_mt"])
    """
    import numpy as np
    import scipy.sparse as sp

    if not hasattr(adata, "X") or not hasattr(adata, "obs"):
        raise TypeError(f"regress_out() requires an AnnData object, got {type(adata).__name__}")

    if not keys:
        raise ValueError("keys must not be empty.")

    for key in keys:
        if key not in adata.obs.columns:
            raise KeyError(f"'{key}' not found in adata.obs.columns")

    # Get expression as dense float64
    X = adata.X
    if sp.issparse(X):
        X = np.asarray(X.todense(), dtype=np.float64)
    else:
        X = np.array(X, dtype=np.float64)

    n_cells = X.shape[0]

    # Build design matrix: intercept + covariates
    design = np.ones((n_cells, 1 + len(keys)), dtype=np.float64)
    for i, key in enumerate(keys):
        values = adata.obs[key].values
        if hasattr(values, "codes"):
            design[:, i + 1] = values.codes.astype(np.float64)
        else:
            design[:, i + 1] = np.array(values, dtype=np.float64)

    # Solve normal equations: beta = (X^T X)^-1 X^T Y
    # Use least squares for numerical stability
    beta, _, _, _ = np.linalg.lstsq(design, X, rcond=None)

    # Residuals + intercept (keeps mean expression)
    X_corrected = X - design[:, 1:] @ beta[1:, :]

    if inplace:
        adata.X = X_corrected
        return None
    return X_corrected
