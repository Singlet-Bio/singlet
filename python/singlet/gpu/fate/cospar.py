# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.fate.cospar — GPU-native CoSpar cell fate transition mapping.

Underlying C++ cycle: cycle 41 (fate/cospar.h —
``singlet::gpu::fate::run_cospar`` kernel).

Reference: Wang et al., "CoSpar: integrating lineage and transcriptomes for
robust trajectory inference," Nature Biotechnology 2022.
https://doi.org/10.1038/s41587-022-01209-1

Result location
---------------
When an AnnData is provided (``run_from_anndata``):

- ``adata.obsm['fate_bias']``      — float32 (n_cells × n_fates).
- ``adata.obs['potency']``         — float32 per-cell potency score.
- ``adata.uns['cospar_params']``   — run parameters.

For raw array usage see ``run_from_csc``.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_core():
    import singlet.gpu._core as _core

    if not hasattr(_core, "fate") or not hasattr(_core.fate, "cospar"):
        raise AttributeError(
            "_core.fate.cospar is not available.  "
            "The C++ extension must be compiled on a CUDA-capable node.  "
            "Run: pip install singlet[gpu] on a GPU node."
        )
    return _core.fate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_from_csc(
    mat,  # DeviceCsc
    clone_ids: np.ndarray,  # int32 (n_cells,)
    time_labels: np.ndarray,  # int32 (n_cells,)
    embedding: np.ndarray,  # float32 (n_cells, n_pcs)
    *,
    k_neighbors: int = 50,
    max_iters: int = 50,
    lambda1: float = 1e-3,
    lambda2: float = 1.0,
    knn_sigma: float = 0.1,
    convergence_tol: float = 1e-4,
    n_fate_bias_steps: int = 5,
    max_cells_per_timepoint: int = 30000,
    stream=None,
    seed: int = 0,
):
    """
    GPU-native CoSpar cell fate mapping (raw CSC input).

    Parameters
    ----------
    mat : DeviceCsc
        Expression matrix (genes × cells) on device.
    clone_ids : np.ndarray int32, shape (n_cells,)
        Clone identifier per cell (-1 = no clone / state-only mode).
    time_labels : np.ndarray int32, shape (n_cells,)
        Integer time-point label per cell (0, 1, 2, ...).
    embedding : np.ndarray float32, shape (n_cells, n_pcs)
        Pre-computed PCA embedding.
    k_neighbors : int, default 50
        kNN for state-similarity kernel construction.
    max_iters : int, default 50
        Proximal gradient outer iterations.
    lambda1 : float, default 1e-3
        L1 regularisation weight.
    lambda2 : float, default 1.0
        State-smoothness weight (K_state @ T term).
    knn_sigma : float, default 0.1
        Gaussian bandwidth for K_state.
    convergence_tol : float, default 1e-4
        Relative Frobenius residual threshold for early stopping.
    n_fate_bias_steps : int, default 5
        Forward propagation steps for fate_bias computation.
    max_cells_per_timepoint : int, default 30000
        Scale guard: raises if any time-point exceeds this count (dense T limit).
    stream : int or None
        CUDA stream pointer (``cupy.cuda.Stream.ptr``).
    seed : int, default 0
        Unused (algorithm is deterministic); kept for API consistency.

    Returns
    -------
    CosparResult
        ``.fate_bias`` (n_cells × n_fates numpy float32)
        ``.potency_score`` (n_cells numpy float32)
        ``.driver_genes`` (n_fates × n_genes numpy float32)
        ``.transition_map_view(pair_idx)`` — __cuda_array_interface__ dict.
    """
    fate = _require_core()
    clone_ids = np.asarray(clone_ids, dtype=np.int32)
    time_labels = np.asarray(time_labels, dtype=np.int32)
    embedding = np.asarray(embedding, dtype=np.float32)

    return fate.cospar(
        mat,
        clone_ids,
        time_labels,
        embedding,
        k_neighbors=k_neighbors,
        max_iters=max_iters,
        lambda1=float(lambda1),
        lambda2=float(lambda2),
        knn_sigma=float(knn_sigma),
        convergence_tol=float(convergence_tol),
        n_fate_bias_steps=int(n_fate_bias_steps),
        max_cells_per_timepoint=int(max_cells_per_timepoint),
        stream=stream,
        seed=int(seed),
    )


def run_from_anndata(
    adata: anndata.AnnData,
    clone_key: str = "clone_id",
    time_key: str = "time",
    basis: str = "X_pca",
    *,
    k_neighbors: int = 50,
    max_iters: int = 50,
    lambda1: float = 1e-3,
    lambda2: float = 1.0,
    knn_sigma: float = 0.1,
    convergence_tol: float = 1e-4,
    n_fate_bias_steps: int = 5,
    max_cells_per_timepoint: int = 30000,
    stream=None,
    seed: int = 0,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """
    GPU-native CoSpar cell fate mapping from an AnnData object.

    Reads the expression matrix from ``adata.X`` (must be a DeviceCsc or
    convertible to one via ``singlet.gpu.load_pz``), clone labels from
    ``adata.obs[clone_key]``, time labels from ``adata.obs[time_key]``,
    and the PCA embedding from ``adata.obsm[basis]``.

    Parameters
    ----------
    adata : AnnData
        Input single-cell dataset.
    clone_key : str, default ``"clone_id"``
        Column in ``adata.obs`` with integer clone identifiers (-1 = no clone).
    time_key : str, default ``"time"``
        Column in ``adata.obs`` with integer time-point labels.
    basis : str, default ``"X_pca"``
        Key in ``adata.obsm`` for the PCA embedding.
    copy : bool, default False
        Return a modified copy.

    Returns
    -------
    None (in-place) or AnnData (copy=True).

    Notes
    -----
    Writes:

    - ``adata.obsm['fate_bias']``    — float32 (n_cells × n_fates).
    - ``adata.obs['potency']``       — float32.
    - ``adata.uns['cospar_params']`` — run parameters.
    """
    import singlet.gpu

    working = copy_module.copy(adata) if copy else adata

    if clone_key not in working.obs.columns:
        raise KeyError(
            f"clone_key='{clone_key}' not in adata.obs.  "
            f"Available columns: {list(working.obs.columns)}"
        )
    if time_key not in working.obs.columns:
        raise KeyError(
            f"time_key='{time_key}' not in adata.obs.  "
            f"Available columns: {list(working.obs.columns)}"
        )
    if basis not in working.obsm:
        raise KeyError(
            f"basis='{basis}' not in adata.obsm.  Available keys: {list(working.obsm.keys())}"
        )

    clone_ids = working.obs[clone_key].to_numpy(dtype=np.int32)
    time_labels = working.obs[time_key].to_numpy(dtype=np.int32)
    embedding = np.asarray(working.obsm[basis], dtype=np.float32)

    # Load expression matrix to device if not already there.
    if not isinstance(working.X, singlet.gpu.DeviceCsc):
        raise TypeError(
            "adata.X must be a DeviceCsc.  "
            "Call singlet.gpu.load_pz() first, or pass mat directly to run_from_csc()."
        )

    result = run_from_csc(
        working.X,
        clone_ids,
        time_labels,
        embedding,
        k_neighbors=k_neighbors,
        max_iters=max_iters,
        lambda1=lambda1,
        lambda2=lambda2,
        knn_sigma=knn_sigma,
        convergence_tol=convergence_tol,
        n_fate_bias_steps=n_fate_bias_steps,
        max_cells_per_timepoint=max_cells_per_timepoint,
        stream=stream,
        seed=seed,
    )

    working.obsm["fate_bias"] = np.asarray(result.fate_bias, dtype=np.float32)
    working.obs["potency"] = np.asarray(result.potency_score, dtype=np.float32)
    working.uns["cospar_params"] = {
        "k_neighbors": k_neighbors,
        "max_iters": max_iters,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "knn_sigma": knn_sigma,
        "convergence_tol": convergence_tol,
        "n_fate_bias_steps": n_fate_bias_steps,
        "iters_run": result.iters_run,
        "state_only_mode": result.state_only_mode,
        "n_fates": result.n_terminal_fates,
    }

    return working if copy else None


__all__ = ["run_from_csc", "run_from_anndata"]
