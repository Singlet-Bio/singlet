# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.qc.omnidoublet — Multimodal CITE-seq doublet detection.

Underlying C++: cycle 39, ``qc/omnidoublet.h``.
Algorithm: joint RNA+ADT PCA embedding + logistic IRLS classifier
           (OmniDoublet, Briefings in Bioinformatics 2024).

For RNA-only datasets use ``singlet.gpu.qc.doublet_score.run_doublet_score``.

API
---
``run_omni_doublet`` — CITE-seq doublet detection; writes results to AnnData.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import anndata


def run_omni_doublet(
    adata: anndata.AnnData,
    adt_key: str = "adt",
    *,
    n_sim_mult: int = 2,
    n_pcs_rna: int = 30,
    n_pcs_adt: int = 20,
    n_hvg: int = 2000,
    k: int = 50,
    irls_max_iter: int = 20,
    irls_tol: float = 1e-5,
    target_doublet_rate: float = 0.10,
    deterministic: bool = False,
    obs_score_key: str = "omni_doublet_score",
    obs_call_key: str = "omni_doublet_call",
    stream=None,
    seed: int = 0,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """
    Multimodal CITE-seq doublet detection (cycle 39).

    Parameters
    ----------
    adata : AnnData
        RNA data in ``adata.X`` (cells × genes).
        ADT data in ``adata.obsm[adt_key]`` (cells × tags) or
        ``adata.layers[adt_key]`` (cells × tags).
    adt_key : str
        Key in ``adata.obsm`` (preferred) or ``adata.layers`` for ADT counts.
    n_sim_mult : int
        Simulated doublets = n_sim_mult × n_cells.
    n_pcs_rna : int
        RNA PCA components for joint embedding.
    n_pcs_adt : int
        ADT PCA components for joint embedding.
    n_hvg : int
        HVG count for RNA PCA input.
    k : int
        kNN neighbours in joint embedding.
    target_doublet_rate : float
        Expected doublet fraction for FDR threshold.
    deterministic : bool
        Use segmented scan for doublet features (reproducible with different GPU configs).
    obs_score_key : str
        Key written to ``adata.obs`` for per-cell doublet scores.
    obs_call_key : str
        Key written to ``adata.obs`` for binary doublet calls.
    stream : int or None
    seed : int
    copy : bool

    Returns
    -------
    None or AnnData
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "omni_doublet"):
        raise AttributeError(
            "_core.omni_doublet is not available.  "
            "Ensure the cycle-52a binding extension has been compiled."
        )

    working = adata.copy() if copy else adata

    try:
        import cupy as cp

        try:
            import cupyx.scipy.sparse as csp  # cupy >= 14
        except ImportError:
            import cupy.sparse as csp  # cupy < 14 fallback
        import scipy.sparse as sp

        # RNA: genes × cells CSC.
        X_rna = working.X
        if X_rna.shape[0] == working.n_obs:
            X_rna = X_rna.T
        d_rna = csp.csc_matrix(X_rna) if sp.issparse(X_rna) else csp.csc_matrix(cp.array(X_rna))

        # ADT: tags × cells CSC.
        if adt_key in working.obsm:
            X_adt = working.obsm[adt_key]
        elif adt_key in working.layers:
            X_adt = working.layers[adt_key]
        else:
            raise KeyError(
                f"ADT data not found.  Expected adata.obsm['{adt_key}'] "
                f"or adata.layers['{adt_key}']."
            )
        if X_adt.shape[0] == working.n_obs:
            X_adt = X_adt.T
        d_adt = csp.csc_matrix(X_adt) if sp.issparse(X_adt) else csp.csc_matrix(cp.array(X_adt))
    except ImportError as e:
        raise ImportError(
            f"singlet.gpu.qc.run_omni_doublet requires cupy.  Original error: {e}"
        ) from e

    result = _core.omni_doublet(
        d_rna,
        d_adt,
        n_sim_mult=n_sim_mult,
        n_pcs_rna=n_pcs_rna,
        n_pcs_adt=n_pcs_adt,
        n_hvg=n_hvg,
        k=k,
        irls_max_iter=irls_max_iter,
        irls_tol=irls_tol,
        target_doublet_rate=target_doublet_rate,
        deterministic=deterministic,
        stream=stream,
        seed=seed,
    )

    working.obs[obs_score_key] = cp.asarray(result.doublet_score_view).get()
    working.obs[obs_call_key] = cp.asarray(result.doublet_call_view).get().astype(bool)
    working.uns["omni_doublet_params"] = {
        "n_sim_mult": n_sim_mult,
        "threshold_used": float(result.threshold_used),
        "n_predicted_doublets": int(result.n_predicted_doublets),
        "target_doublet_rate": target_doublet_rate,
        "seed": seed,
    }

    return working if copy else None


__all__ = ["run_omni_doublet"]
