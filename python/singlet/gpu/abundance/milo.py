# SPDX-License-Identifier: MIT
"""
singlet.gpu.abundance.milo — GPU-native Milo differential abundance testing.

Underlying C++ cycle: cycle 47 (abundance/milo.h —
``singlet::gpu::abundance::run_milo`` kernel).

Reference: Dann et al., "Differential abundance testing on single-cell data
using k-nearest neighbor graphs," Nature Biotechnology 2022.
https://doi.org/10.1038/s41587-021-01033-z

Result location
---------------
When an AnnData is provided (``run_from_anndata``):

- ``adata.uns['milo']['log_fc']``   — float32 (n_nh,).
- ``adata.uns['milo']['p_value']``  — float32 (n_nh,).
- ``adata.uns['milo']['fdr']``      — float32 (n_nh,).
- ``adata.uns['milo']['nh_index_cells']`` — int32 (n_nh,).
- ``adata.uns['milo']['params']``   — run parameters.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, Optional

import numpy as np

from singlet.gpu._coreutil import require_core

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _view_to_numpy(view, n, dtype=np.float32):
    """Copy a __cuda_array_interface__ scalar-array view to numpy."""
    try:
        import cupy as cp
        arr = cp.ndarray(
            shape=(n,), dtype=cp.float32,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(view["data"][0], n * 4, None), 0)
        ).get()
        return arr.astype(dtype)
    except (ImportError, Exception):
        import ctypes
        buf = (ctypes.c_float * n)()
        ctypes.memmove(ctypes.addressof(buf),
                       ctypes.c_void_p(view["data"][0]), n * 4)
        return np.frombuffer(buf, dtype=np.float32).copy().astype(dtype)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_from_embedding(
    embedding: np.ndarray,
    condition: np.ndarray,
    donor_id: np.ndarray,
    *,
    k: int = 30,
    n_nh_target: int = 2000,
    max_irls_iters: int = 20,
    irls_tol: float = 1e-4,
    min_cells_fraction: float = 0.2,
    min_donors_per_nh: int = 3,
    deterministic: bool = True,
    stream=None,
    seed: int = 42,
):
    """
    GPU-native Milo differential-abundance testing (raw embedding input).

    Parameters
    ----------
    embedding : np.ndarray float32, shape (n_cells, n_dims)
        Low-dimensional cell embedding (e.g. PCA).
    condition : np.ndarray int32, shape (n_cells,)
        Binary condition label (0 / 1) per cell.
    donor_id : np.ndarray int32, shape (n_cells,)
        Donor identifier (0-indexed integers) per cell.
    k : int, default 30
        kNN neighborhood size.
    n_nh_target : int, default 2000
        Target number of neighborhoods.
    max_irls_iters : int, default 20
        IRLS iterations for the negative-binomial GLM.
    irls_tol : float, default 1e-4
        IRLS convergence tolerance.
    min_cells_fraction : float, default 0.2
        Drop neighborhoods with fewer than this fraction of donors present.
    min_donors_per_nh : int, default 3
        Minimum number of donors required in a neighborhood.
    deterministic : bool, default True
        Use scan-based (deterministic) neighborhood count path.
    stream : int or None
        CUDA stream pointer.
    seed : int, default 42
        Philox seed for neighborhood sampling.

    Returns
    -------
    MiloResult
        Device arrays exposed via __cuda_array_interface__.
        Use ``.log_fc_view``, ``.p_value_view``, ``.fdr_view``,
        ``.nh_index_cells_view`` to access results.
    """
    ab = require_core("abundance", "milo")
    embedding = np.asarray(embedding, dtype=np.float32)
    condition = np.asarray(condition, dtype=np.int32)
    donor_id  = np.asarray(donor_id,  dtype=np.int32)

    return ab.milo(
        embedding, condition, donor_id,
        k=int(k),
        n_nh_target=int(n_nh_target),
        max_irls_iters=int(max_irls_iters),
        irls_tol=float(irls_tol),
        min_cells_fraction=float(min_cells_fraction),
        min_donors_per_nh=int(min_donors_per_nh),
        deterministic=bool(deterministic),
        stream=stream,
        seed=int(seed),
    )


def run_from_anndata(
    adata: "anndata.AnnData",
    condition_key: str,
    donor_key: str,
    *,
    basis: str = "X_pca",
    k: int = 30,
    n_nh_target: int = 2000,
    max_irls_iters: int = 20,
    irls_tol: float = 1e-4,
    min_cells_fraction: float = 0.2,
    min_donors_per_nh: int = 3,
    deterministic: bool = True,
    stream=None,
    seed: int = 42,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    GPU-native Milo differential-abundance testing from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.obsm[basis]``, ``adata.obs[condition_key]``,
        and ``adata.obs[donor_key]``.
    condition_key : str
        Column in ``adata.obs`` with binary condition labels (0/1 or two
        unique categorical values; encoded to 0/1 automatically).
    donor_key : str
        Column in ``adata.obs`` with donor identifiers.
    basis : str, default ``"X_pca"``
        Key in ``adata.obsm`` for the cell embedding.
    copy : bool, default False

    Returns
    -------
    None (in-place) or AnnData (copy=True).

    Notes
    -----
    Writes ``adata.uns['milo']`` with keys:

    - ``log_fc``           — float32 (n_nh,).
    - ``se``               — float32 (n_nh,).
    - ``p_value``          — float32 (n_nh,).
    - ``fdr``              — float32 (n_nh,).
    - ``nh_index_cells``   — int32 (n_nh,).
    - ``nh_valid``         — int32 (n_nh,).
    - ``params``           — run parameters.
    """
    working = copy_module.copy(adata) if copy else adata

    if basis not in working.obsm:
        raise KeyError(f"basis='{basis}' not in adata.obsm.")
    if condition_key not in working.obs.columns:
        raise KeyError(f"condition_key='{condition_key}' not in adata.obs.")
    if donor_key not in working.obs.columns:
        raise KeyError(f"donor_key='{donor_key}' not in adata.obs.")

    embedding = np.asarray(working.obsm[basis], dtype=np.float32)

    cond_series = working.obs[condition_key].astype("category")
    if len(cond_series.cat.categories) != 2:
        raise ValueError(
            f"condition_key='{condition_key}' must have exactly 2 unique values, "
            f"got {list(cond_series.cat.categories)}."
        )
    condition = cond_series.cat.codes.to_numpy(dtype=np.int32)

    don_series = working.obs[donor_key].astype("category")
    donor_id   = don_series.cat.codes.to_numpy(dtype=np.int32)

    result = run_from_embedding(
        embedding, condition, donor_id,
        k=k,
        n_nh_target=n_nh_target,
        max_irls_iters=max_irls_iters,
        irls_tol=irls_tol,
        min_cells_fraction=min_cells_fraction,
        min_donors_per_nh=min_donors_per_nh,
        deterministic=deterministic,
        stream=stream,
        seed=seed,
    )

    n_nh = result.n_nh

    working.uns["milo"] = {
        "log_fc":         _view_to_numpy(result.log_fc_view,        n_nh),
        "se":             _view_to_numpy(result.se_view,             n_nh),
        "p_value":        _view_to_numpy(result.p_value_view,        n_nh),
        "fdr":            _view_to_numpy(result.fdr_view,            n_nh),
        "nh_index_cells": _view_to_numpy(result.nh_index_cells_view, n_nh, dtype=np.int32),
        "nh_valid":       _view_to_numpy(result.nh_valid_view,       n_nh, dtype=np.int32),
        "condition_labels": list(cond_series.cat.categories),
        "donor_labels":     list(don_series.cat.categories),
        "params": {
            "k": k,
            "n_nh_target": n_nh_target,
            "min_cells_fraction": min_cells_fraction,
            "min_donors_per_nh": min_donors_per_nh,
            "n_nh_computed": n_nh,
            "seed": seed,
        },
    }

    return working if copy else None


__all__ = ["run_from_embedding", "run_from_anndata"]
