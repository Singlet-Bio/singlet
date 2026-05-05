# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.integrate.harmony — GPU-native Harmony batch integration.

Underlying C++ cycle: cycle 14 (integrate/harmony.h —
``singlet::gpu::integrate::harmony`` kernel).

Drop-in for ``sc.external.pp.harmony_integrate``.  All parameter names and
defaults match the harmonypy / scanpy 1.10 external API.  The GPU kernel
runs the full Harmony EM loop (soft k-means + correction) entirely on device.

Result location (matches harmonypy / scanpy external)
------------------------------------------------------
``adata.obsm[adjusted_basis]`` — float32 array (n_cells × n_pcs).

CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE: ``_core.harmony`` must be
exposed by the cycle-22 pybind11 binding extension.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_embedding(
    adata: "anndata.AnnData",
    basis: str,
) -> "np.ndarray":
    """
    Extract the PCA embedding from adata.obsm[basis].

    Returns
    -------
    np.ndarray of shape (n_cells, n_pcs), float32.
    """
    if basis not in adata.obsm:
        raise KeyError(
            f"Embedding key '{basis}' not found in adata.obsm.  "
            f"Available keys: {list(adata.obsm.keys())}.  "
            "Run sc.pp.pca() or singlet.gpu.reduce.svd() first."
        )
    emb = adata.obsm[basis]
    try:
        import cupy as cp

        if isinstance(emb, cp.ndarray):
            return emb.astype(cp.float32, copy=False)
    except ImportError:
        pass
    return np.asarray(emb, dtype=np.float32)


def _resolve_batch_codes(
    adata: "anndata.AnnData",
    key: Union[str, List[str]],
) -> "np.ndarray":
    """
    Build an integer batch-code vector from obs column(s).

    When *key* is a list, cells are grouped by the cross-product of all
    key columns (matching harmonypy's multi-key behaviour).

    Returns
    -------
    np.ndarray of shape (n_cells,), dtype int32.
    """

    keys = [key] if isinstance(key, str) else list(key)
    missing = [k for k in keys if k not in adata.obs.columns]
    if missing:
        raise KeyError(
            f"Batch key(s) {missing} not found in adata.obs.  "
            f"Available columns: {list(adata.obs.columns)}"
        )

    if len(keys) == 1:
        series = adata.obs[keys[0]].astype("category")
        return series.cat.codes.to_numpy(dtype=np.int32)

    # Multi-key: concatenate as tuple-string then encode.
    combined = adata.obs[keys].astype(str).agg("_".join, axis=1).astype("category")
    return combined.cat.codes.to_numpy(dtype=np.int32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def harmony_integrate(
    adata: "anndata.AnnData",
    key: Union[str, List[str]],
    *,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
    n_clusters: int = 20,
    max_iter: int = 10,
    tol: float = 1e-4,
    seed: int = 0,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    GPU-native Harmony batch integration (cycle-14 kernel).

    Mirrors ``sc.external.pp.harmony_integrate`` — parameter names and
    defaults match harmonypy / scanpy so this function is a drop-in
    replacement.

    Harmony corrects a PCA embedding by iteratively soft-clustering cells,
    computing per-cluster batch centroids, and subtracting per-batch
    corrections.  All computation runs on device; only the corrected
    embedding is transferred to host for storage in ``adata.obsm``.

    Parameters
    ----------
    adata : AnnData
        Must contain a PCA embedding in ``adata.obsm[basis]``.
    key : str or list of str
        Observation column(s) in ``adata.obs`` that identify the batch.
        When a list is given, the cross-product of all columns defines
        the batch label (harmonypy convention).
    basis : str, default ``"X_pca"``
        Key in ``adata.obsm`` for the input embedding.
    adjusted_basis : str, default ``"X_pca_harmony"``
        Key in ``adata.obsm`` where the corrected embedding is written.
    n_clusters : int, default 20
        Number of Harmony clusters (K in the soft k-means step).
    max_iter : int, default 10
        Maximum number of Harmony EM iterations.
    tol : float, default 1e-4
        Convergence tolerance on the cluster-assignment entropy.
    seed : int, default 0
        Random seed for cluster initialisation.
    copy : bool, default False
        Return a modified copy of *adata*.  When ``False`` (default),
        *adata* is modified in-place and ``None`` is returned.

    Returns
    -------
    None
        When ``copy=False`` (default, in-place).
    AnnData
        When ``copy=True`` — the modified copy.

    Raises
    ------
    AttributeError
        If ``_core.harmony`` is not available.
        See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE.
    KeyError
        If *basis* is not in ``adata.obsm`` or *key* not in ``adata.obs``.

    Notes
    -----
    **Result location** (matches scanpy external / harmonypy):
    ``adata.obsm[adjusted_basis]`` — float32 array of shape (n_cells, n_pcs).

    **No host copies in the hot path.**  The PCA embedding is transferred
    to device once (if not already there), the full EM loop runs on device,
    and the corrected embedding is transferred to host only at the end for
    ``obsm`` storage.

    Examples
    --------
    Single batch key::

        import singlet.gpu.integrate as sgi
        sgi.harmony_integrate(adata, key="batch")

    Multiple batch keys::

        sgi.harmony_integrate(adata, key=["donor", "sequencing_run"])

    Read the corrected embedding::

        print(adata.obsm['X_pca_harmony'].shape)  # (n_cells, n_pcs)
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "harmony"):
        raise AttributeError(
            "_core.harmony is not available.  "
            "See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE — the cycle-22 "
            "pybind11 binding extension must expose harmony before "
            "singlet.gpu.integrate.harmony_integrate() is callable."
        )

    working = copy_module.copy(adata) if copy else adata

    emb = _get_embedding(working, basis)  # (n_cells, n_pcs) float32
    batch_codes = _resolve_batch_codes(working, key)  # (n_cells,) int32
    n_batches = int(batch_codes.max()) + 1

    # CYCLE-267: binding signature (kw_only after n_batches):
    #   harmony(embedding, batch_labels, n_batches, *, n_clusters, max_iter,
    #           tol, lambda, seed: uint64)
    # Upload to device if not already CAI-exposing (CYCLE-261 pattern).
    if not hasattr(emb, "__cuda_array_interface__"):
        import cupy as cp

        emb = cp.asarray(emb, dtype=cp.float32)
    if not hasattr(batch_codes, "__cuda_array_interface__"):
        import cupy as cp

        batch_codes = cp.asarray(batch_codes, dtype=cp.int32)
    result = _core.harmony(
        emb,
        batch_codes,
        n_batches,
        n_clusters=int(n_clusters),
        max_iter=int(max_iter),
        tol=float(tol),
        seed=int(seed),
    )

    # CYCLE-268: HarmonyResult exposes corrected_view (CAI dict) per
    # _bind_results.hpp (n_iters_used, final_delta, corrected_view, __repr__).
    # The corrected embedding is float32 (n × d, row-major).
    import cupy as cp

    class _CaiView:
        def __init__(self, d):
            self.__cuda_array_interface__ = d

    n_cells = int(emb.shape[0])
    corrected_flat = cp.asarray(_CaiView(result.corrected_view))
    corrected = corrected_flat.reshape(n_cells, -1).get()
    working.obsm[adjusted_basis] = np.asarray(corrected, dtype=np.float32)

    return working if copy else None


__all__ = ["harmony_integrate"]
