"""Variational inference for single-cell latent space.

Provides singlet.variational_inference() — a lightweight probabilistic PCA
with optional batch correction (Bayesian PPCA), producing a latent embedding
and batch-corrected normalized expression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def variational_inference(
    adata: "AnnData",
    n_latent: int = 10,
    *,
    n_epochs: int = 50,
    batch_key: str | None = None,
    random_state: int = 0,
) -> "AnnData":
    """Compute variational latent space via probabilistic PCA.

    Implements Bayesian probabilistic PCA (PPCA) with optional batch
    correction. The model assumes:

        z ~ N(0, I)        (latent variables)
        x | z ~ N(Wz + mu + batch_effect, sigma^2 I)

    Parameters are estimated via EM algorithm. When ``batch_key`` is
    provided, batch-specific means are estimated and removed to produce
    a corrected expression layer.

    Parameters
    ----------
    adata
        Annotated data matrix (cells × genes). Should be log-normalized
        or otherwise preprocessed.
    n_latent
        Number of latent dimensions.
    n_epochs
        Number of EM iterations.
    batch_key
        Key in ``adata.obs`` for batch labels. If provided, batch effects
        are estimated and corrected.
    random_state
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        Input ``adata`` with:
        - ``adata.obsm['X_scvi']``: latent representation (cells × n_latent).
        - ``adata.layers['scvi_normalized']``: batch-corrected normalized
          expression (cells × genes).
    """
    import numpy as np
    import scipy.sparse as sp

    rng = np.random.default_rng(random_state)

    # Extract dense matrix
    X = adata.X
    if sp.issparse(X):
        X = np.asarray(X.toarray(), dtype=np.float64)
    else:
        X = np.asarray(X, dtype=np.float64)

    n_cells, n_genes = X.shape

    # Ensure n_latent is valid
    n_latent = min(n_latent, n_genes - 1, n_cells - 1)

    # Compute batch effects if batch_key provided
    batch_effects = np.zeros((n_cells, n_genes), dtype=np.float64)
    if batch_key is not None and batch_key in adata.obs.columns:
        batches = adata.obs[batch_key].values
        unique_batches = np.unique(batches)
        global_mean = X.mean(axis=0)

        batch_means = {}
        for batch in unique_batches:
            mask = batches == batch
            batch_means[batch] = X[mask].mean(axis=0) - global_mean

        for batch in unique_batches:
            mask = batches == batch
            batch_effects[mask] = batch_means[batch]

    # Remove batch effects for PPCA
    X_corrected = X - batch_effects

    # EM algorithm for probabilistic PCA
    # Initialize
    mu = X_corrected.mean(axis=0)
    X_centered = X_corrected - mu

    # Initialize W randomly
    W = rng.normal(0, 0.1, size=(n_genes, n_latent))
    sigma2 = 1.0

    for _epoch in range(n_epochs):
        # E-step: compute expected latent variables
        # M = W^T W + sigma^2 I
        M = W.T @ W + sigma2 * np.eye(n_latent)
        M_inv = np.linalg.inv(M)

        # E[z | x] = M^{-1} W^T (x - mu)
        # Shape: (n_latent, n_genes) @ (n_genes, n_cells) -> (n_latent, n_cells)
        Ez = (M_inv @ W.T @ X_centered.T).T  # (n_cells, n_latent)

        # E[zz^T | x] = sigma^2 M^{-1} + E[z]E[z]^T
        Ez_cov = sigma2 * M_inv  # (n_latent, n_latent) - shared part

        # M-step: update W and sigma^2
        # W_new = (sum_n x_n E[z_n]^T) (sum_n E[z_n z_n^T])^{-1}
        # sum_n E[z_n z_n^T] = n_cells * Ez_cov + Ez^T @ Ez
        sum_Ezz = n_cells * Ez_cov + Ez.T @ Ez  # (n_latent, n_latent)
        sum_xEz = X_centered.T @ Ez  # (n_genes, n_latent)

        W = sum_xEz @ np.linalg.inv(sum_Ezz)

        # sigma^2 = (1/ND) sum_n ||x_n - mu||^2 - 2 E[z_n]^T W^T x_n
        #           + Tr(E[z_n z_n^T] W^T W)
        recon = Ez @ W.T  # (n_cells, n_genes)
        residuals = X_centered - recon
        sigma2 = np.mean(residuals**2)
        sigma2 = max(sigma2, 1e-6)  # numerical stability

    # Final latent representation
    M = W.T @ W + sigma2 * np.eye(n_latent)
    M_inv = np.linalg.inv(M)
    latent = (M_inv @ W.T @ X_centered.T).T  # (n_cells, n_latent)

    # Batch-corrected normalized expression: reconstruction without batch
    normalized = latent @ W.T + mu

    adata.obsm["X_scvi"] = latent.astype(np.float32)
    adata.layers["scvi_normalized"] = normalized.astype(np.float32)

    return adata
