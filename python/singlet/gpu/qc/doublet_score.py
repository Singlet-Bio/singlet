# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.qc.doublet_score — Scrublet-style RNA-only doublet detection.

Underlying C++: cycle 31, ``qc/doublet_score.h``.
Algorithm: synthetic doublet PCA scoring (Wolock et al. 2019, Scrublet).

For CITE-seq (RNA + ADT) use ``singlet.gpu.qc.omnidoublet.run_omni_doublet``.

API
---
``run_doublet_score`` — score doublets from a PCA embedding; writes to AnnData.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import anndata


def run_doublet_score(
    adata: "anndata.AnnData",
    *,
    embedding_key: str = "X_pca",
    n_synth_frac: float = 0.25,
    k: int = 20,
    manual_threshold: float = 0.0,
    obs_score_key: str = "doublet_score",
    obs_call_key: str = "doublet_call",
    stream=None,
    seed: int = 0,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    Scrublet-style doublet detection on a PCA embedding (cycle 31).

    Requires ``adata.obsm[embedding_key]`` to exist (e.g., run
    ``singlet.gpu.reduce.pca(adata)`` first).

    Parameters
    ----------
    adata : AnnData
        Must contain a PCA embedding in ``adata.obsm[embedding_key]``.
    embedding_key : str
        Key in ``adata.obsm`` for the (n_cells, n_pcs) float32 embedding.
    n_synth_frac : float
        N_synth = ceil(n_synth_frac × n_cells).
    k : int
        kNN neighbours in combined (real + synthetic) cloud.
    manual_threshold : float
        0 = auto-threshold via histogram knee; >0 = use as the score cut-off.
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

    if not hasattr(_core, "doublet_score"):
        raise AttributeError(
            "_core.doublet_score is not available.  "
            "Ensure the cycle-52a binding extension has been compiled."
        )

    if embedding_key not in adata.obsm:
        raise KeyError(
            f"adata.obsm['{embedding_key}'] not found.  "
            "Run PCA first (e.g., singlet.gpu.reduce.pca(adata))."
        )

    working = adata.copy() if copy else adata

    try:
        import cupy as cp

        emb = np.asarray(working.obsm[embedding_key], dtype=np.float32)
        d_emb = cp.asarray(emb)
    except ImportError as e:
        raise ImportError(f"singlet.gpu.qc.run_doublet_score requires cupy.  Original error: {e}")

    result = _core.doublet_score(
        d_emb,
        n_synth_frac=n_synth_frac,
        k=k,
        manual_threshold=manual_threshold,
        stream=stream,
        seed=seed,
    )

    working.obs[obs_score_key] = cp.asarray(result.score_view).get()
    working.obs[obs_call_key] = cp.asarray(result.doublet_call_view).get().astype(bool)
    working.uns["doublet_score_params"] = {
        "embedding_key": embedding_key,
        "n_synth_frac": n_synth_frac,
        "k": k,
        "threshold_used": float(result.threshold_used),
        "n_predicted_doublets": int(result.n_predicted_doublets),
        "seed": seed,
    }

    return working if copy else None


__all__ = ["run_doublet_score"]
