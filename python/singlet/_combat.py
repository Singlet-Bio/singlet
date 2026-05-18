# SPDX-License-Identifier: MIT
"""ComBat batch correction for AnnData objects.

Implements the ComBat algorithm (Johnson et al., Biostatistics 2007) for
removing batch effects from expression data using empirical Bayes.
Operates directly on the expression matrix (unlike Harmony which works on PCA).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


def combat(
    adata,
    key: str,
    *,
    covariates: Optional[list[str]] = None,
    inplace: bool = True,
) -> Optional["np.ndarray"]:
    """Remove batch effects using ComBat (empirical Bayes).

    Adjusts gene expression values to remove batch effects while
    preserving biological variation. Works on log-normalized data.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Should be log-normalized.
    key : str
        Column in adata.obs identifying batch membership.
    covariates : list[str] or None, default None
        Columns in adata.obs to protect as biological covariates.
        These effects are preserved during correction.
    inplace : bool, default True
        If True, modifies adata.X in place.
        If False, returns corrected expression matrix.

    Returns
    -------
    numpy.ndarray or None
        Corrected expression matrix (n_cells × n_genes) if inplace=False.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.combat(adata, key="batch")
    """
    import numpy as np
    import scipy.sparse as sp

    if not hasattr(adata, "X") or not hasattr(adata, "obs"):
        raise TypeError(f"combat() requires an AnnData object, got {type(adata).__name__}")

    if key not in adata.obs.columns:
        raise KeyError(f"'{key}' not found in adata.obs.columns")

    # Get expression matrix as dense float64
    X = adata.X
    if sp.issparse(X):
        X = np.asarray(X.todense(), dtype=np.float64)
    else:
        X = np.array(X, dtype=np.float64)

    n_cells, n_genes = X.shape

    # Encode batches
    batch_labels = adata.obs[key].values
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)

    if n_batches < 2:
        if inplace:
            adata.X = X.copy()
            return None
        return X.copy()

    batch_to_idx = {b: i for i, b in enumerate(unique_batches)}
    batch_indices = np.array([batch_to_idx[b] for b in batch_labels])

    # Build design matrix
    # Intercept + batch indicators (reference coding: first batch = reference)
    n_covars = 0
    covar_matrix = None
    if covariates:
        valid_covars = [c for c in covariates if c in adata.obs.columns]
        if valid_covars:
            covar_matrix = np.zeros((n_cells, len(valid_covars)), dtype=np.float64)
            for ci, col in enumerate(valid_covars):
                vals = adata.obs[col].values
                if hasattr(vals, "codes"):
                    covar_matrix[:, ci] = vals.codes.astype(np.float64)
                else:
                    covar_matrix[:, ci] = np.array(vals, dtype=np.float64)
            n_covars = covar_matrix.shape[1]

    # Design matrix: intercept + covariates
    n_design = 1 + n_covars
    design = np.ones((n_cells, n_design), dtype=np.float64)
    if covar_matrix is not None:
        design[:, 1:] = covar_matrix

    # Step 1: Standardize data
    # Fit linear model (intercept + covariates) to get residuals
    # Then estimate batch effects on residuals
    beta_hat = np.linalg.lstsq(design, X, rcond=None)[0]  # (n_design × n_genes)
    X_hat = design @ beta_hat  # fitted values (biological signal)

    # Standardize: center and scale

    # Per-batch mean and variance of residuals
    batch_means = np.zeros((n_batches, n_genes), dtype=np.float64)
    batch_vars = np.zeros((n_batches, n_genes), dtype=np.float64)
    batch_sizes = np.zeros(n_batches, dtype=int)

    for b in range(n_batches):
        mask = batch_indices == b
        batch_sizes[b] = mask.sum()
        if batch_sizes[b] > 0:
            batch_data = X[mask] - X_hat[mask]
            batch_means[b] = batch_data.mean(axis=0)
            if batch_sizes[b] > 1:
                batch_vars[b] = batch_data.var(axis=0, ddof=1)
            else:
                batch_vars[b] = 1.0

    # Pool variance estimate
    pooled_var = np.zeros(n_genes, dtype=np.float64)
    for b in range(n_batches):
        if batch_sizes[b] > 1:
            pooled_var += (batch_sizes[b] - 1) * batch_vars[b]
    pooled_var /= max(n_cells - n_batches, 1)
    pooled_var[pooled_var == 0] = 1.0
    pooled_std = np.sqrt(pooled_var)

    # Step 2: Empirical Bayes estimation of batch parameters
    # Estimate gamma (location) and delta (scale) using EB shrinkage

    # Standardized batch effects
    gamma_hat = batch_means / pooled_std[np.newaxis, :]

    # EB shrinkage for gamma (location parameter)
    gamma_bar = gamma_hat.mean(axis=1, keepdims=True)  # (n_batches, 1)
    tau2 = gamma_hat.var(axis=1, keepdims=True)  # (n_batches, 1)
    tau2[tau2 == 0] = 1.0

    # Posterior gamma (shrunk toward overall mean)
    # gamma_star = (tau2 * gamma_hat + var_hat * gamma_bar) / (tau2 + var_hat)
    # Simplified: use moderate shrinkage
    n_g = n_genes
    gamma_star = (n_g * tau2 * gamma_hat + gamma_bar) / (n_g * tau2 + 1)

    # EB shrinkage for delta (variance parameter)
    delta_hat = batch_vars / pooled_var[np.newaxis, :]
    delta_hat[delta_hat == 0] = 1.0

    # Use method of moments for inverse gamma prior
    delta_bar = delta_hat.mean(axis=1, keepdims=True)
    delta_star = delta_hat.copy()
    # Moderate shrinkage toward batch mean
    for b in range(n_batches):
        if batch_sizes[b] > 2:
            shrink = 0.5
            delta_star[b] = shrink * delta_hat[b] + (1 - shrink) * delta_bar[b]

    delta_star[delta_star == 0] = 1.0

    # Step 3: Correct data
    X_corrected = X.copy()

    for b in range(n_batches):
        mask = batch_indices == b
        if not mask.any():
            continue

        # Subtract batch location effect
        X_corrected[mask] -= gamma_star[b] * pooled_std

        # Scale by batch variance effect
        sqrt_delta = np.sqrt(delta_star[b])
        sqrt_delta[sqrt_delta == 0] = 1.0
        X_corrected[mask] = (X_corrected[mask] - X_hat[mask]) / sqrt_delta + X_hat[mask]

    if inplace:
        adata.X = X_corrected
        return None
    return X_corrected
