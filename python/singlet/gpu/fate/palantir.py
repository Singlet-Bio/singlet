# SPDX-License-Identifier: MIT
"""
singlet.gpu.fate.palantir — GPU-native Palantir diffusion pseudotime.

Underlying C++ cycle: cycle 45 (fate/palantir.h —
``singlet::gpu::fate::run_palantir`` kernel).

Reference: Setty et al., "Characterization of cell fate probabilities in
single-cell data with Palantir," Nature Biotechnology 2019.
https://doi.org/10.1038/s41587-019-0068-4

Result location
---------------
When an AnnData is provided (``run_from_anndata``):

- ``adata.obs['palantir_pseudotime']``     — float32 pseudotime in [0, 1].
- ``adata.obsm['palantir_branch_prob']``   — float32 (n_cells × n_terminals).
- ``adata.obsm['X_palantir_multiscale']``  — float32 diffusion components.
- ``adata.uns['palantir_params']``         — run parameters.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from singlet.gpu._coreutil import require_core

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_from_embedding(
    embedding: np.ndarray,
    start_cell: int,
    terminal_indices: Optional[np.ndarray] = None,
    *,
    k_neighbors: int = 30,
    n_pcs: int = 30,
    k_eig: int = 20,
    auto_terminals: bool = False,
    n_terminals_auto: int = 5,
    gmres_m: int = 30,
    gmres_max_restarts: int = 10,
    gmres_convergence_tol: float = 1e-6,
    gmres_ooc_batch_terminals: int = -1,
    normalize_pseudotime: bool = True,
    stream=None,
    seed: int = 42,
):
    """
    GPU-native Palantir pseudotime + branch probabilities (raw embedding input).

    Parameters
    ----------
    embedding : np.ndarray float32, shape (n_cells, n_pcs)
        PCA embedding of the cells.
    start_cell : int
        Index of the root cell (lowest pseudotime).
    terminal_indices : np.ndarray int32 or None
        Terminal (absorbing) state cell indices.  If None and
        ``auto_terminals=False``, a greedy farthest-point selection is used.
    k_neighbors : int, default 30
        kNN for diffusion graph construction.
    n_pcs : int, default 30
        PCA dimensionality (must match embedding.shape[1]).
    k_eig : int, default 20
        Number of diffusion components.
    auto_terminals : bool, default False
        If True, auto-detect terminals from pseudotime.
    n_terminals_auto : int, default 5
        Number of terminals to detect when auto_terminals=True.
    gmres_m : int, default 30
        Krylov subspace dimension for branch probability GMRES.
    gmres_max_restarts : int, default 10
        Max GMRES restarts.
    gmres_convergence_tol : float, default 1e-6
        GMRES residual threshold.
    gmres_ooc_batch_terminals : int, default -1
        -1 = all terminals at once; >0 = OOC batch size.
    normalize_pseudotime : bool, default True
        Normalise pseudotime to [0, 1] range.
    stream : int or None
        CUDA stream pointer.
    seed : int, default 42
        Seed for Lanczos randomised start vector.

    Returns
    -------
    PalantirResult
        ``.pseudotime`` (n_cells numpy float32)
        ``.branch_prob`` (n_cells × n_terminals numpy float32)
        ``.eigenvalues`` (k_eig numpy float32)
        ``.terminal_indices`` (n_terminals numpy int32)
        ``.eigenvectors_view`` — __cuda_array_interface__ dict (device).
    """
    fate = require_core("fate", "palantir")
    embedding = np.asarray(embedding, dtype=np.float32)
    if terminal_indices is None:
        terminal_indices = np.array([], dtype=np.int32)
    else:
        terminal_indices = np.asarray(terminal_indices, dtype=np.int32)

    return fate.palantir(
        embedding,
        int(start_cell),
        terminal_indices,
        k_neighbors=int(k_neighbors),
        n_pcs=int(n_pcs),
        k_eig=int(k_eig),
        auto_terminals=bool(auto_terminals),
        n_terminals_auto=int(n_terminals_auto),
        gmres_m=int(gmres_m),
        gmres_max_restarts=int(gmres_max_restarts),
        gmres_convergence_tol=float(gmres_convergence_tol),
        gmres_ooc_batch_terminals=int(gmres_ooc_batch_terminals),
        normalize_pseudotime=bool(normalize_pseudotime),
        stream=stream,
        seed=int(seed),
    )


def run_from_anndata(
    adata: "anndata.AnnData",
    start_cell: int,
    terminal_cells: Optional[List[int]] = None,
    *,
    basis: str = "X_pca",
    k_neighbors: int = 30,
    k_eig: int = 20,
    auto_terminals: bool = False,
    n_terminals_auto: int = 5,
    gmres_m: int = 30,
    gmres_max_restarts: int = 10,
    gmres_convergence_tol: float = 1e-6,
    gmres_ooc_batch_terminals: int = -1,
    normalize_pseudotime: bool = True,
    stream=None,
    seed: int = 42,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    GPU-native Palantir pseudotime from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Must contain a PCA embedding in ``adata.obsm[basis]``.
    start_cell : int
        Index of the root cell.
    terminal_cells : list of int or None
        Terminal cell indices.  None → auto-detect or greedy selection.
    basis : str, default ``"X_pca"``
        Key in ``adata.obsm`` for the PCA embedding.
    copy : bool, default False

    Returns
    -------
    None (in-place) or AnnData (copy=True).

    Notes
    -----
    Writes:

    - ``adata.obs['palantir_pseudotime']``    — float32.
    - ``adata.obsm['palantir_branch_prob']``  — float32 (n_cells × n_terminals).
    - ``adata.obsm['X_palantir_multiscale']`` — float32 (n_cells × k_eig).
    - ``adata.uns['palantir_params']``        — run parameters.
    """
    working = copy_module.copy(adata) if copy else adata

    if basis not in working.obsm:
        raise KeyError(
            f"basis='{basis}' not in adata.obsm.  "
            f"Available keys: {list(working.obsm.keys())}"
        )
    embedding = np.asarray(working.obsm[basis], dtype=np.float32)
    terminal_indices = (
        np.asarray(terminal_cells, dtype=np.int32)
        if terminal_cells is not None
        else None
    )

    result = run_from_embedding(
        embedding, start_cell, terminal_indices,
        k_neighbors=k_neighbors,
        n_pcs=embedding.shape[1],
        k_eig=k_eig,
        auto_terminals=auto_terminals,
        n_terminals_auto=n_terminals_auto,
        gmres_m=gmres_m,
        gmres_max_restarts=gmres_max_restarts,
        gmres_convergence_tol=gmres_convergence_tol,
        gmres_ooc_batch_terminals=gmres_ooc_batch_terminals,
        normalize_pseudotime=normalize_pseudotime,
        stream=stream,
        seed=seed,
    )

    working.obs["palantir_pseudotime"]     = np.asarray(result.pseudotime,   dtype=np.float32)
    working.obsm["palantir_branch_prob"]   = np.asarray(result.branch_prob,  dtype=np.float32)

    # Eigenvectors: copy from device to host for obsm storage.
    n_eig = result.k_eig
    n_c   = result.n_cells
    ev_view = result.eigenvectors_view
    try:
        import cupy as cp
        ev_host = cp.ndarray(
            shape=(n_eig * n_c,), dtype=cp.float32,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(ev_view["data"][0], n_eig * n_c * 4, None),
                0)
        ).get()
    except (ImportError, Exception):
        import ctypes
        buf = (ctypes.c_float * (n_eig * n_c))()
        ctypes.memmove(
            ctypes.addressof(buf),
            ctypes.c_void_p(ev_view["data"][0]),
            n_eig * n_c * 4,
        )
        ev_host = np.frombuffer(buf, dtype=np.float32).copy()

    # Reshape (k_eig × n_cells) → (n_cells × k_eig)
    working.obsm["X_palantir_multiscale"] = ev_host.reshape(n_eig, n_c).T.astype(np.float32)

    working.uns["palantir_params"] = {
        "start_cell": start_cell,
        "basis": basis,
        "k_neighbors": k_neighbors,
        "k_eig": k_eig,
        "n_terminals": result.n_terminals,
        "terminal_indices": result.terminal_indices.tolist(),
        "normalize_pseudotime": normalize_pseudotime,
        "seed": seed,
    }

    return working if copy else None


__all__ = ["run_from_embedding", "run_from_anndata"]
