# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.perturbation.perturb_graph — GPU-native CPA perturbation autoencoder.

Underlying C++: cycle 32, ``perturbation/perturb_graph.h``.
Algorithm: Compositional Perturbation Autoencoder (Lotfollahi et al., Nat Methods 2023).

API
---
``fit``                — train CPA; returns a PerturbGraphModel.
``predict_perturbation`` — predict post-perturbation expression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import anndata


class PerturbGraphModel:
    """
    Trained CPA model wrapper.

    Attributes
    ----------
    result : _core.PerturbGraphResult
        C++ result (device buffers for pert_embeddings + cell_latents).
    pert_names : list[str]
        Perturbation names in index order.
    """

    def __init__(self, result, pert_names: list):
        self._result    = result
        self.pert_names = pert_names

    @property
    def epoch_losses(self) -> list:
        """Total training loss per epoch."""
        return self._result.epoch_losses

    @property
    def adv_losses(self) -> list:
        """Adversarial CE loss per epoch."""
        return self._result.adv_losses

    def pert_embeddings(self) -> np.ndarray:
        """Return [n_perts × d_latent] perturbation embedding (host numpy)."""
        import cupy as cp
        n_perts = self._result.n_perts
        d       = self._result.d_latent
        return cp.asarray(self._result.pert_embeddings_view).reshape(n_perts, d).get()

    def cell_latents(self) -> np.ndarray:
        """Return [n_cells × d_latent] basal cell latents (host numpy)."""
        import cupy as cp
        d = self._result.d_latent
        return cp.asarray(self._result.cell_latents_view).reshape(-1, d).get()

    def predict_perturbation(
        self,
        adata_query: "anndata.AnnData",
        target_pert: str,
        *,
        target_dose: float = 1.0,
        stream=None,
    ) -> np.ndarray:
        """
        Predict post-perturbation expression for query cells.

        Parameters
        ----------
        adata_query : AnnData
            genes × cells count matrix for query cells.
        target_pert : str
            Perturbation name (must be in ``self.pert_names``).
        target_dose : float
            Dose scalar.
        stream : int or None

        Returns
        -------
        numpy.ndarray
            [n_query_cells × n_genes] float32 predicted expression.
        """
        if target_pert not in self.pert_names:
            raise ValueError(
                f"'{target_pert}' not in trained perturbation set.  "
                f"Available: {self.pert_names}"
            )
        pert_id = self.pert_names.index(target_pert)

        try:
            import cupy as cp
            try:
                import cupyx.scipy.sparse as csp  # cupy >= 14
            except ImportError:
                import cupy.sparse as csp         # cupy < 14 fallback
            import scipy.sparse as sp

            X = adata_query.X
            if X.shape[0] == adata_query.n_obs:
                X = X.T
            mat = csp.csc_matrix(X) if sp.issparse(X) else csp.csc_matrix(cp.array(X))
        except ImportError as e:
            raise ImportError(
                "singlet_gpu.perturbation.perturb_graph requires cupy.  "
                f"Original error: {e}"
            )

        view = self._result.predict_perturbation(mat, pert_id, target_dose, stream)
        n_query = adata_query.n_obs
        n_genes = self._result.n_genes
        arr = cp.asarray(view).reshape(n_query, n_genes)
        return arr.get()

    def __repr__(self) -> str:
        return (
            f"<PerturbGraphModel n_perts={self._result.n_perts} "
            f"d_latent={self._result.d_latent}>"
        )


def fit(
    adata: "anndata.AnnData",
    pert_key: str = "perturbation",
    *,
    d_latent: int = 64,
    d_hidden: int = 128,
    n_epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    kl_weight: float = 0.01,
    adversarial_weight: float = 1.0,
    stream=None,
    seed: int = 0,
) -> PerturbGraphModel:
    """
    Train a CPA compositional perturbation autoencoder.

    Parameters
    ----------
    adata : AnnData
        ``adata.X`` is (cells × genes) or (genes × cells) count matrix.
        ``adata.obs[pert_key]`` must contain perturbation labels (string or int).
    pert_key : str
        Column in ``adata.obs`` with perturbation identities.
    stream : int or None
    seed : int

    Returns
    -------
    PerturbGraphModel
    """
    import singlet_gpu._core as _core

    if not hasattr(_core, "train_perturb_graph"):
        raise AttributeError(
            "_core.train_perturb_graph is not available.  "
            "Ensure the cycle-52a binding extension has been compiled."
        )

    try:
        import cupy as cp
        try:
            import cupyx.scipy.sparse as csp  # cupy >= 14
        except ImportError:
            import cupy.sparse as csp         # cupy < 14 fallback
        import scipy.sparse as sp

        X = adata.X
        if X.shape[0] == adata.n_obs:
            X = X.T
        mat = csp.csc_matrix(X) if sp.issparse(X) else csp.csc_matrix(cp.array(X))
    except ImportError as e:
        raise ImportError(
            "singlet_gpu.perturbation.perturb_graph.fit requires cupy.  "
            f"Original error: {e}"
        )

    # Encode perturbation labels as integers.
    raw_labels = adata.obs[pert_key].values
    if raw_labels.dtype.kind not in ("i", "u"):
        pert_names = list(dict.fromkeys(raw_labels))
        label_map  = {n: i for i, n in enumerate(pert_names)}
        int_labels = np.array([label_map[l] for l in raw_labels], dtype=np.int32)
    else:
        pert_names = [str(i) for i in range(int(raw_labels.max()) + 1)]
        int_labels = np.asarray(raw_labels, dtype=np.int32)

    d_pert_ids = cp.asarray(int_labels)

    result = _core.train_perturb_graph(
        mat, d_pert_ids,
        None,    # dose: 1.0 for all cells
        d_latent=d_latent,
        d_hidden=d_hidden,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        kl_weight=kl_weight,
        adversarial_weight=adversarial_weight,
        stream=stream,
        seed=seed,
    )
    return PerturbGraphModel(result, pert_names=pert_names)


__all__ = ["fit", "PerturbGraphModel"]
