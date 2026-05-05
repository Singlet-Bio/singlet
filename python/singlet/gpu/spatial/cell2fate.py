# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.spatial.cell2fate — GPU-native RNA velocity module decomposition.

Underlying C++: cycle 27, ``embed/cell2fate.h``.
Algorithm: Poisson SVI over K interpretable velocity modules
           (Lederer & Trynka, Nature Methods 2025).

API
---
``fit``     — train the model; returns a Cell2FateModel wrapper.
``predict`` — retrieve per-cell kinetic rates + velocities from a Cell2FateModel.

The Cell2FateModel wraps the raw _core.Cell2FateResult and exposes
AnnData-friendly accessors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import anndata


class Cell2FateModel:
    """
    Trained Cell2fate model.  Wraps ``_core.Cell2FateResult``.

    Attributes
    ----------
    result : _core.Cell2FateResult
        Raw C++ result (device buffers).
    gene_names : list[str]
        Gene names matching the training matrix rows.
    """

    def __init__(self, result, gene_names: list):
        self.result = result
        self.gene_names = gene_names

    # ------------------------------------------------------------------ #
    # Kinetic rate accessors (host copies for downstream use)            #
    # ------------------------------------------------------------------ #

    def kinetic_rates(self) -> dict:
        """
        Return per-gene kinetic rate estimates as a dict of numpy arrays.

        Returns
        -------
        dict with keys "alpha", "beta", "gamma" (each [n_genes]),
        and "alpha_var", "beta_var", "gamma_var" for posterior variances.
        """
        try:
            import cupy as cp

            return {
                "alpha": cp.asarray(self.result.alpha_view).get(),
                "beta": cp.asarray(self.result.beta_view).get(),
                "gamma": cp.asarray(self.result.gamma_view).get(),
                "alpha_var": cp.asarray(self.result.alpha_var_view).get(),
                "beta_var": cp.asarray(self.result.beta_var_view).get(),
                "gamma_var": cp.asarray(self.result.gamma_var_view).get(),
            }
        except ImportError:
            raise ImportError("singlet.gpu.spatial.cell2fate requires cupy.")

    def module_loadings(self) -> np.ndarray:
        """Return [K × n_genes] module loading matrix (host numpy)."""
        import cupy as cp

        K = self.result.K_used
        n_genes = len(self.gene_names)
        return cp.asarray(self.result.module_loadings_view).reshape(K, n_genes).get()

    def cell_activities(self, adata: "anndata.AnnData") -> np.ndarray:
        """Return [n_cells × K] cell activity matrix (host numpy)."""
        import cupy as cp

        K = self.result.K_used
        return cp.asarray(self.result.cell_activities_view).reshape(-1, K).get()

    def top_genes_per_module(self, n: int = 20) -> list:
        """Return list of length K, each a list of gene names for top-n genes."""
        import cupy as cp

        K = self.result.K_used
        idx = cp.asarray(self.result.module_topgenes_view).reshape(K, n).get()
        return [[self.gene_names[i] for i in row] for row in idx.astype(int)]

    def to_anndata(
        self, adata: "anndata.AnnData", copy: bool = False
    ) -> Optional["anndata.AnnData"]:
        """
        Write Cell2fate results to an AnnData object.

        Written keys:
        - ``adata.varm['cell2fate_module_loadings']``   — [n_genes × K]
        - ``adata.obsm['cell2fate_activities']``        — [n_cells × K]
        - ``adata.uns['cell2fate_kinetics']``           — per-gene rate dict
        - ``adata.uns['cell2fate_params']``             — training parameters
        """
        working = adata.copy() if copy else adata
        K = self.result.K_used
        working.varm["cell2fate_module_loadings"] = self.module_loadings().T  # n_genes × K
        working.obsm["cell2fate_activities"] = self.cell_activities(adata)
        working.uns["cell2fate_kinetics"] = self.kinetic_rates()
        working.uns["cell2fate_params"] = {
            "K_used": K,
            "n_iters_used": self.result.n_iters_used,
        }
        return working if copy else None

    def __repr__(self) -> str:
        return (
            f"<Cell2FateModel K={self.result.K_used} "
            f"iters={self.result.n_iters_used} "
            f"n_genes={len(self.gene_names)}>"
        )


def fit(
    adata: "anndata.AnnData",
    spliced_layer: str = "spliced",
    unspliced_layer: str = "unspliced",
    *,
    n_modules: int = 10,
    max_iter: int = 2000,
    batch_size: int = 1024,
    learning_rate: float = 0.01,
    n_posterior_draws: int = 100,
    top_n_per_module: int = 20,
    beta_kl_max: float = 1.0,
    kl_anneal_fraction: float = 0.5,
    convergence_tol: float = 1e-4,
    convergence_patience: int = 10,
    compute_velocity: bool = False,
    gene_tile: int = 1024,
    stream=None,
    seed: int = 0,
) -> Cell2FateModel:
    """
    Fit a Cell2fate model to spliced + unspliced count matrices.

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.layers[spliced_layer]`` and
        ``adata.layers[unspliced_layer]`` (cells × genes).
    spliced_layer : str
        Layer key for spliced counts.
    unspliced_layer : str
        Layer key for unspliced counts.
    n_modules : int
        Number of velocity modules K.
    compute_velocity : bool
        Emit full [n_cells × n_genes] velocity matrix (expensive: 12 GB at 100k×30k).
    stream : int or None
    seed : int

    Returns
    -------
    Cell2FateModel
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "cell2fate_fit"):
        raise AttributeError(
            "_core.cell2fate_fit is not available.  "
            "Ensure the cycle-52a binding extension has been compiled."
        )

    try:
        import cupy as cp

        try:
            import cupyx.scipy.sparse as csp  # cupy >= 14
        except ImportError:
            import cupy.sparse as csp  # cupy < 14 fallback
        import scipy.sparse as sp

        def _layer_to_csc(layer_key):
            X = adata.layers[layer_key]
            if X.shape[0] == adata.n_obs:
                X = X.T  # ensure genes × cells
            if sp.issparse(X):
                return csp.csc_matrix(X)
            return csp.csc_matrix(cp.array(X))

        d_spliced = _layer_to_csc(spliced_layer)
        d_unspliced = _layer_to_csc(unspliced_layer)
    except ImportError as e:
        raise ImportError(f"singlet.gpu.spatial.cell2fate.fit requires cupy.  Original error: {e}")

    result = _core.cell2fate_fit(
        d_spliced,
        d_unspliced,
        n_modules=n_modules,
        max_iter=max_iter,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_posterior_draws=n_posterior_draws,
        top_n_per_module=top_n_per_module,
        beta_kl_max=beta_kl_max,
        kl_anneal_fraction=kl_anneal_fraction,
        convergence_tol=convergence_tol,
        convergence_patience=convergence_patience,
        compute_velocity=compute_velocity,
        gene_tile=gene_tile,
        stream=stream,
        seed=seed,
    )
    return Cell2FateModel(result, gene_names=list(adata.var_names))


__all__ = ["fit", "Cell2FateModel"]
