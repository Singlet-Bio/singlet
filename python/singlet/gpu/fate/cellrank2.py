# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.fate.cellrank2 — GPU-native CellRank 2 absorption probability solver.

Underlying C++ cycle: cycle 43 (fate/cellrank2.h —
``singlet::gpu::fate::run_cellrank2`` kernel).

Reference: Weiler et al., "CellRank 2: unified fate mapping in multiview
single-cell data," Nature Methods 2024.
https://doi.org/10.1038/s41592-024-02303-9

Result location
---------------
When an AnnData is provided (``run_from_anndata``):

- ``adata.obsm['X_absorption_prob']`` — float32 (n_cells × n_terminals).
- ``adata.uns['cellrank2_params']``   — run parameters.
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

    if not hasattr(_core, "fate") or not hasattr(_core.fate, "cellrank2"):
        raise AttributeError(
            "_core.fate.cellrank2 is not available.  "
            "The C++ extension must be compiled on a CUDA-capable node."
        )
    return _core.fate


def _csr_to_arrays(T):
    """Extract indptr / indices / data from a CSR matrix (scipy or cupy)."""
    try:
        try:
            import cupyx.scipy.sparse as csp  # cupy >= 14
        except ImportError:
            import cupy.sparse as csp  # cupy < 14 fallback
        if isinstance(T, (csp.csr_matrix, csp.csc_matrix)):
            T_csr = T.tocsr() if hasattr(T, "tocsr") else T
            return (
                np.asarray(T_csr.indptr, dtype=np.int32),
                np.asarray(T_csr.indices, dtype=np.int32),
                np.asarray(T_csr.data, dtype=np.float32),
                int(T_csr.shape[0]),
            )
    except ImportError:
        pass

    # scipy sparse
    import scipy.sparse as sp

    if sp.issparse(T):
        T_csr = T.tocsr()
        return (
            np.asarray(T_csr.indptr, dtype=np.int32),
            np.asarray(T_csr.indices, dtype=np.int32),
            np.asarray(T_csr.data, dtype=np.float32),
            int(T_csr.shape[0]),
        )

    raise TypeError(
        f"transition_matrix must be a scipy.sparse or cupy.sparse matrix, got {type(T).__name__}."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_absorption_probabilities(
    transition_matrix,
    terminal_indices: np.ndarray,
    *,
    gmres_m: int = 30,
    gmres_max_restarts: int = 10,
    convergence_tol: float = 1e-6,
    ooc_batch_terminals: int = -1,
    compute_driver_genes: bool = False,
    mat=None,
    stream=None,
    seed: int = 0,
):
    """
    GPU-native CellRank 2 absorption probability computation (raw inputs).

    Parameters
    ----------
    transition_matrix : scipy.sparse.csr_matrix or cupy.sparse.csr_matrix
        Row-stochastic transition matrix (n_cells × n_cells).
    terminal_indices : np.ndarray int32, shape (n_terminals,)
        Indices of absorbing (terminal) states.
    gmres_m : int, default 30
        Krylov subspace dimension (restart size).
    gmres_max_restarts : int, default 10
        Maximum outer GMRES restarts per terminal.
    convergence_tol : float, default 1e-6
        Relative residual convergence threshold.
    ooc_batch_terminals : int, default -1
        -1 = process all terminals at once; >0 = OOC batch size.
    compute_driver_genes : bool, default False
        If True, compute Pearson correlation of absorption prob vs expression.
        Requires *mat* (DeviceCsc).
    mat : DeviceCsc or None
        Expression matrix for driver gene computation.
    stream : int or None
        CUDA stream pointer.
    seed : int, default 0
        Unused (deterministic); kept for API consistency.

    Returns
    -------
    CellRank2Result
        ``.absorption_prob`` (n_cells × n_terminals numpy float32)
        ``.driver_genes`` (n_terminals × n_genes numpy float32; empty if not requested)
        ``.converged_all`` bool.
    """
    fate = _require_core()
    indptr, indices, data, n_cells = _csr_to_arrays(transition_matrix)
    terminal_indices = np.asarray(terminal_indices, dtype=np.int32)

    return fate.cellrank2(
        indptr,
        indices,
        data,
        n_cells,
        terminal_indices,
        gmres_m=int(gmres_m),
        gmres_max_restarts=int(gmres_max_restarts),
        convergence_tol=float(convergence_tol),
        ooc_batch_terminals=int(ooc_batch_terminals),
        compute_driver_genes=bool(compute_driver_genes),
        mat=mat,
        stream=stream,
        seed=int(seed),
    )


def run_from_anndata(
    adata: "anndata.AnnData",
    transition_matrix_key: str = "T_fwd",
    terminal_states_key: str = "terminal_states",
    *,
    basis_result: str = "X_absorption_prob",
    gmres_m: int = 30,
    gmres_max_restarts: int = 10,
    convergence_tol: float = 1e-6,
    ooc_batch_terminals: int = -1,
    compute_driver_genes: bool = False,
    stream=None,
    seed: int = 0,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    GPU-native CellRank 2 absorption probabilities from an AnnData object.

    Reads the transition matrix from ``adata.obsp[transition_matrix_key]``
    and terminal state indices from ``adata.obs[terminal_states_key]``
    (non-NaN rows are treated as terminal).

    Parameters
    ----------
    adata : AnnData
        Must have ``adata.obsp[transition_matrix_key]`` (CSR sparse matrix)
        and ``adata.obs[terminal_states_key]`` (string or integer labels;
        NaN / negative = transient cell).
    transition_matrix_key : str, default ``"T_fwd"``
        Key in ``adata.obsp`` for the transition matrix.
    terminal_states_key : str, default ``"terminal_states"``
        Column in ``adata.obs`` identifying terminal cells.
    basis_result : str, default ``"X_absorption_prob"``
        Key in ``adata.obsm`` where the result is stored.
    copy : bool, default False

    Returns
    -------
    None (in-place) or AnnData (copy=True).

    Notes
    -----
    Writes:

    - ``adata.obsm[basis_result]``         — float32 (n_cells × n_terminals).
    - ``adata.uns['cellrank2_params']``    — run parameters.
    """
    working = copy_module.copy(adata) if copy else adata

    if transition_matrix_key not in working.obsp:
        raise KeyError(
            f"transition_matrix_key='{transition_matrix_key}' not in adata.obsp.  "
            f"Available keys: {list(working.obsp.keys())}"
        )
    T = working.obsp[transition_matrix_key]

    # Detect terminal indices from obs column (non-NaN rows).
    if terminal_states_key in working.obs.columns:
        col = working.obs[terminal_states_key]
        try:
            terminal_mask = col.notna() & (col != "nan") & (col != "")
        except Exception:
            terminal_mask = ~np.isnan(col.to_numpy(dtype=float, na_value=np.nan))
        terminal_indices = np.where(terminal_mask.to_numpy())[0].astype(np.int32)
    else:
        raise KeyError(f"terminal_states_key='{terminal_states_key}' not in adata.obs.")

    result = compute_absorption_probabilities(
        T,
        terminal_indices,
        gmres_m=gmres_m,
        gmres_max_restarts=gmres_max_restarts,
        convergence_tol=convergence_tol,
        ooc_batch_terminals=ooc_batch_terminals,
        compute_driver_genes=compute_driver_genes,
        stream=stream,
        seed=seed,
    )

    working.obsm[basis_result] = np.asarray(result.absorption_prob, dtype=np.float32)
    working.uns["cellrank2_params"] = {
        "transition_matrix_key": transition_matrix_key,
        "terminal_states_key": terminal_states_key,
        "n_terminals": result.n_terminals,
        "gmres_m": gmres_m,
        "gmres_max_restarts": gmres_max_restarts,
        "convergence_tol": convergence_tol,
        "converged_all": result.converged_all,
    }

    return working if copy else None


__all__ = ["compute_absorption_probabilities", "run_from_anndata"]
