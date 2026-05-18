# SPDX-License-Identifier: MIT
"""Multi-modal factor analysis (MOFA-lite).

Joint SVD/NMF decomposition across multiple modalities to discover
shared latent factors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


def multiome_factor_analysis(
    adata: "AnnData",
    modality_keys: list[str],
    *,
    n_factors: int = 20,
    method: str = "svd",
    random_state: int = 0,
) -> "AnnData":
    """Joint factor analysis across multiple modalities (MOFA-lite).

    Performs dimensionality reduction on concatenated modality matrices,
    recovering shared latent factors and per-modality loadings.

    Parameters
    ----------
    adata
        Annotated data matrix. Modalities should be stored in
        ``adata.obsm`` or ``adata.layers``.
    modality_keys
        List of keys to look up in ``adata.obsm`` first, then
        ``adata.layers``. Each must be a cell × feature matrix.
    n_factors
        Number of latent factors to compute.
    method
        Decomposition method: ``'svd'`` (truncated SVD) or ``'nmf'``
        (non-negative matrix factorization).
    random_state
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        The input ``adata`` with:
        - ``adata.obsm['X_mofa']``: cell × n_factors factor matrix.
        - ``adata.uns['mofa_loadings']``: dict mapping each modality key
          to its n_factors × n_features loading matrix.
        - ``adata.uns['mofa_params']``: dict of parameters used.

    Raises
    ------
    KeyError
        If a modality key is not found in ``adata.obsm`` or ``adata.layers``.
    ValueError
        If ``method`` is not 'svd' or 'nmf', or if modalities have
        inconsistent numbers of cells, or n_factors exceeds feature count.
    """
    from scipy.sparse import issparse

    valid_methods = {"svd", "nmf"}
    if method not in valid_methods:
        msg = f"Method must be one of {valid_methods}, got '{method}'"
        raise ValueError(msg)

    n_cells = adata.n_obs
    matrices = []
    feature_dims = []

    for key in modality_keys:
        mat = _get_modality_matrix(adata, key)
        if mat.shape[0] != n_cells:
            msg = f"Modality '{key}' has {mat.shape[0]} cells, expected {n_cells}"
            raise ValueError(msg)
        # Convert sparse to dense
        if issparse(mat):
            mat = np.asarray(mat.todense())
        elif not isinstance(mat, np.ndarray):
            mat = np.asarray(mat)
        matrices.append(mat.astype(np.float64))
        feature_dims.append(mat.shape[1])

    # Concatenate modalities along feature axis
    concatenated = np.hstack(matrices)
    total_features = concatenated.shape[1]

    if n_factors > min(n_cells, total_features):
        msg = f"n_factors={n_factors} exceeds min(n_cells={n_cells}, n_features={total_features})"
        raise ValueError(msg)

    # Decomposition
    if method == "svd":
        factors, loadings = _svd_decompose(concatenated, n_factors, random_state)
    else:
        factors, loadings = _nmf_decompose(concatenated, n_factors, random_state)

    # Split loadings back per modality
    mofa_loadings = {}
    offset = 0
    for key, dim in zip(modality_keys, feature_dims):
        mofa_loadings[key] = loadings[:, offset : offset + dim]
        offset += dim

    adata.obsm["X_mofa"] = factors
    adata.uns["mofa_loadings"] = mofa_loadings
    adata.uns["mofa_params"] = {
        "modality_keys": modality_keys,
        "n_factors": n_factors,
        "method": method,
        "random_state": random_state,
    }

    return adata


def _get_modality_matrix(adata: "AnnData", key: str) -> np.ndarray:
    """Retrieve modality matrix from obsm or layers."""
    if key in adata.obsm:
        return adata.obsm[key]
    if key in adata.layers:
        return adata.layers[key]
    msg = f"Key '{key}' not found in adata.obsm or adata.layers"
    raise KeyError(msg)


def _svd_decompose(
    data: np.ndarray, n_factors: int, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Truncated SVD decomposition."""
    from sklearn.decomposition import TruncatedSVD

    # Center data for SVD
    mean = np.mean(data, axis=0)
    centered = data - mean

    svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
    factors = svd.fit_transform(centered)
    loadings = svd.components_  # n_factors × n_features

    return factors, loadings


def _nmf_decompose(
    data: np.ndarray, n_factors: int, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative matrix factorization."""
    from sklearn.decomposition import NMF

    # NMF requires non-negative input — shift if needed
    min_val = np.min(data)
    if min_val < 0:
        data_nn = data - min_val
    else:
        data_nn = data

    model = NMF(n_components=n_factors, random_state=random_state, max_iter=500)
    factors = model.fit_transform(data_nn)
    loadings = model.components_  # n_factors × n_features

    return factors, loadings
