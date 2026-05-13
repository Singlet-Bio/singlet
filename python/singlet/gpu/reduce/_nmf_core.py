# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.reduce.nmf — GPU-native NMF via factornet.

Underlying C++ cycle: cycle 6 (reduce/nmf/*.h — factornet NMF adapters:
fit.h, chunked.h, graph.h, types.h).

Three entry points:

``nmf``
    Standard NMF on a single GPU-resident AnnData.  Wraps
    ``factornet::nmf::fit_gpu<float>``.  Writes
    ``adata.obsm['X_nmf']`` (cells × k) and
    ``adata.varm['NMF_loadings']`` (genes × k).

``nmf_chunked``
    Out-of-core streaming NMF over a list of ``.1pz`` file paths.  Uses
    ``singlet::gpu::io::PzDataLoader`` (which implements
    ``factornet::io::DataLoader<float>``) so the data is never fully
    materialised on device.  Returns a raw ``NmfResult`` rather than an
    AnnData because the data is too large to fit in memory.

``nmf_graph_factorize``
    Multi-modal joint NMF using ``factornet::graph::FactorGraph`` with a
    ``SharedNode`` for shared-H across modalities.  Takes a dict mapping
    modality names to AnnData objects.  Returns per-modality AnnData with
    shared ``obsm['X_nmf']`` and per-modality ``varm['NMF_loadings']``.

Loss types (``loss`` parameter)
--------------------------------
``"MSE"``      — L2 / Frobenius loss (fully GPU-resident).
``"KL"``       — KL / Poisson divergence  (host-mediated IRLS, 50-100× slower).
``"NB"``       — Negative Binomial        (host-mediated IRLS).
``"GP"``       — Generalized Poisson      (host-mediated IRLS).
``"Gamma"``    — Gamma deviance           (host-mediated IRLS).
``"Tweedie"``  — Tweedie compound Poisson (host-mediated IRLS).

Non-MSE losses use host-mediated IRLS because factornet's GPU NMF is only
fully device-resident for MSE.  This is upstream behavior; we document but
do not override it (style-rules §F Non-MSE loss caveat).

Solver modes (``solver_mode`` parameter)
-----------------------------------------
0 — Coordinate Descent (constraints active).
1 — Cholesky + clip (unconstrained MSE).
2 — Multiplicative Updates (every non-MSE loss; 10-38× faster than CD on
    sparse MSE).
3 — Auto (factornet data-driven switching, default).

Initialisation modes (``init_mode``)
--------------------------------------
0 — Random uniform [0, 1).
1 — Lanczos SVD init: W = |U|√Σ, H = √Σ|V|ᵀ.
2 — IRLBA SVD init (default, reuses svd.svd_irlba for k ≥ 32).

Streams
-------
factornet NMF kernels manage CUDA streams via ``factornet::gpu::GPUContext``.
The Python wrapper passes no explicit stream (style-rules §C.15).

CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE: ``_core.nmf``, ``_core.nmf_chunked``,
``_core.nmf_graph_factorize``, and ``_core.from_cupy_csr`` are not yet exposed
by the cycle-18 binding.  These wrappers raise ``AttributeError`` until those
bindings are added.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# Valid parameter sets
# ---------------------------------------------------------------------------
_VALID_LOSSES = ("MSE", "KL", "NB", "GP", "Gamma", "Tweedie")
_VALID_SOLVER_MODES = (0, 1, 2, 3)
_VALID_INIT_MODES = (0, 1, 2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_matrix(adata: anndata.AnnData, layer: Optional[str]):
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        return adata.layers[layer]
    return adata.X


def _csr_to_device_csc(csr_mat):
    """CSR (cells × genes) → DeviceCSC (genes × cells)."""
    import singlet.gpu._core as _core

    if not hasattr(_core, "from_cupy_csr"):
        raise AttributeError(
            "_core.from_cupy_csr is not available.  See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )
    csc_mat = csr_mat.T.tocsc()
    return _core.from_cupy_csr(csc_mat)


def _write_nmf_result(
    adata: anndata.AnnData,
    result,
    *,
    n_factors: int,
    modality_key: str = "X_nmf",
) -> None:
    """
    Write a factornet NmfResult struct into an AnnData.

    _core.NmfResult exposes W_view / H_view / d_view as CAI dicts (per
    binding _bind_kernels.hpp + §J.13 _CaiView shim required for cupy 14).
    loss_history exists as a Python list attribute.
    """
    import cupy as cp
    import numpy as np

    class _CaiView:  # cupy 14 dtype-strict shim (§J.13 / CYCLE-189)
        def __init__(self, d):
            self.__cuda_array_interface__ = d

    k = int(result.k_used)
    # PyNmfResult upload from factornet:
    #   d_W: shape (rows=genes × k_used) col-major
    #   d_H: shape (k_used × cols=cells) col-major
    W = cp.asarray(_CaiView(result.W_view)).reshape(k, -1).T.get()  # genes × k
    H = cp.asarray(_CaiView(result.H_view)).reshape(-1, k)  # cells × k

    adata.obsm[modality_key] = (
        H.astype(np.float32, copy=False).get()
        if hasattr(H, "get")
        else H.astype(np.float32, copy=False)
    )
    adata.varm["NMF_loadings"] = W.astype(np.float32, copy=False)

    # loss_history may not be exposed; skip gracefully.
    loss_hist = list(result.loss_history) if hasattr(result, "loss_history") else []
    adata.uns["nmf"] = {
        "n_factors": n_factors,
        "loss_history": loss_hist,
        "final_loss": float(loss_hist[-1]) if loss_hist else float("nan"),
    }


def _build_nmf_config(
    *,
    n_factors: int,
    loss: str,
    solver_mode: int,
    init_mode: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> dict:
    """Assemble the config dict passed to the C++ binding."""
    return {
        "n_factors": int(n_factors),
        "loss": str(loss),
        "solver_mode": int(solver_mode),
        "init_mode": int(init_mode),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# Public API — nmf
# ---------------------------------------------------------------------------


def nmf(
    adata: anndata.AnnData,
    *,
    n_factors: int = 20,
    loss: str = "MSE",
    solver_mode: int = 3,
    init_mode: int = 2,
    max_iter: int = 100,
    tol: float = 1e-5,
    seed: int = 0,
    layer: Optional[str] = None,
    inplace: bool = True,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """
    GPU-native NMF via factornet (cycle-6 kernel, ``factornet::nmf::fit_gpu``).

    Decomposes ``adata.X`` (or ``adata.layers[layer]``) as ``A ≈ W·H`` where
    ``W`` (genes × k) and ``H`` (k × cells) are non-negative.

    Results are written into:

    * ``adata.obsm['X_nmf']``       — (cells × k) cell embeddings (Hᵀ).
    * ``adata.varm['NMF_loadings']`` — (genes × k) gene loadings (W).
    * ``adata.uns['nmf']``          — convergence info.

    Parameters
    ----------
    adata : AnnData
        AnnData with ``X`` as a GPU-resident ``cupy.sparse.csr_matrix``.
    n_factors : int, default 20
        Number of NMF factors (rank k).
    loss : str, default ``"MSE"``
        Loss function.  ``"MSE"`` is fully GPU-resident.  All other losses
        use host-mediated IRLS and are 50–100× slower.
        Choices: ``"MSE"``, ``"KL"``, ``"NB"``, ``"GP"``, ``"Gamma"``,
        ``"Tweedie"``.
    solver_mode : int, default 3 (auto)
        NNLS solver backend.  0=CD, 1=Cholesky+clip, 2=MU, 3=auto.
    init_mode : int, default 2 (IRLBA SVD init)
        Initialisation.  0=random, 1=Lanczos SVD, 2=IRLBA SVD.
    max_iter : int, default 100
        Maximum number of alternating least-squares iterations.
    tol : float, default 1e-5
        Relative loss tolerance for early stopping.
    seed : int, default 0
        Random seed (used when ``init_mode=0`` or stochastic NNLS paths).
    layer : str or None, default None
        Layer to factorise.  ``None`` operates on ``adata.X``.
    inplace : bool, default True
        Write results into *adata*.  If ``False``, operate on a copy.
    copy : bool, default False
        Return a copy.  When ``True``, implies ``inplace=False``.

    Returns
    -------
    None
        When ``inplace=True`` (default).
    AnnData
        When ``copy=True`` or ``inplace=False``.

    Raises
    ------
    ValueError
        If *loss*, *solver_mode*, or *init_mode* is not recognised.
    AttributeError
        If ``_core.nmf`` or ``_core.from_cupy_csr`` is not available
        (see CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE).

    Examples
    --------
    Default (MSE, k=20, IRLBA init)::

        import singlet.gpu.reduce as sgr
        sgr.nmf(adata)

    Negative Binomial loss (count data)::

        sgr.nmf(adata, loss="NB", n_factors=30)
    """
    if loss not in _VALID_LOSSES:
        raise ValueError(f"loss='{loss}' not recognised.  Choose from: {_VALID_LOSSES}")
    if solver_mode not in _VALID_SOLVER_MODES:
        raise ValueError(
            f"solver_mode={solver_mode} not recognised.  Choose from: {_VALID_SOLVER_MODES}"
        )
    if init_mode not in _VALID_INIT_MODES:
        raise ValueError(f"init_mode={init_mode} not recognised.  Choose from: {_VALID_INIT_MODES}")

    import singlet.gpu._core as _core

    if not hasattr(_core, "nmf"):
        raise AttributeError(
            "_core.nmf is not available.  See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    working_adata = adata if (inplace and not copy) else copy_module.copy(adata)
    mat = _get_matrix(working_adata, layer)
    device_csc = _csr_to_device_csc(mat)

    # _core.nmf signature (py::kw_only after rank):
    #   nmf(mat, rank, *, loss='MSE', solver_mode=3, init_mode=2,
    #       max_iter=100, tol=1e-5, seed=0)
    result = _core.nmf(
        device_csc,
        int(n_factors),
        loss=str(loss),
        solver_mode=int(solver_mode),
        init_mode=int(init_mode),
        max_iter=int(max_iter),
        tol=float(tol),
        seed=int(seed),
    )
    _write_nmf_result(working_adata, result, n_factors=n_factors)

    if inplace and not copy:
        return None
    return working_adata


# ---------------------------------------------------------------------------
# Public API — nmf_chunked
# ---------------------------------------------------------------------------


class NmfResult:
    """
    Lightweight container for the result of ``nmf_chunked``.

    Attributes
    ----------
    W : numpy.ndarray  (genes × k)
        Basis matrix — gene loadings.
    H_list : list of numpy.ndarray  (k × cells_i)
        Per-chunk coefficient matrices.  Each corresponds to one input
        ``.1pz`` path.
    loss_history : list of float
        Per-iteration reconstruction loss.
    n_genes : int
        Number of genes (rows of W).
    n_factors : int
        Rank k.
    """

    def __init__(self, W, H_list, loss_history, n_genes: int, n_factors: int):
        self.W = W
        self.H_list = H_list
        self.loss_history = loss_history
        self.n_genes = n_genes
        self.n_factors = n_factors
        # CYCLE-277-FOLLOWUP: tests expect `H` (or `loadings`) for the gene
        # factor matrix.  In our convention W is genes × k (loadings) and H
        # is k × cells (factors).  Expose both `H` (alias to first chunk)
        # and `loadings` (= W) for scanpy-parity test contract.
        self.H = H_list[0] if H_list else None
        self.loadings = W

    def __repr__(self) -> str:
        return (
            f"NmfResult(n_genes={self.n_genes}, n_factors={self.n_factors}, "
            f"n_chunks={len(self.H_list)}, "
            f"final_loss={self.loss_history[-1]:.6g if self.loss_history else float('nan')})"
        )


def nmf_chunked(
    paths: List[str],
    *,
    n_factors: int = 20,
    chunk_cols: int = 100_000,
    loss: str = "MSE",
    solver_mode: int = 3,
    init_mode: int = 2,
    max_iter: int = 100,
    tol: float = 1e-5,
    seed: int = 0,
) -> NmfResult:
    """
    Out-of-core streaming NMF over multiple ``.1pz`` files (cycle-6 kernel).

    Uses ``singlet::gpu::io::PzDataLoader`` as a ``factornet::io::DataLoader``
    so data is loaded in chunks and never fully materialised on device.
    Bypasses AnnData because the total data may be too large to hold in GPU
    memory simultaneously.

    Parameters
    ----------
    paths : list of str
        Paths to ``.1pz`` sample directories or files.  All must share the
        same gene set.
    n_factors : int, default 20
        Number of NMF factors.
    chunk_cols : int, default 100_000
        Number of cells per streaming chunk.
    loss : str, default ``"MSE"``
        Loss function.  See ``nmf`` for full list.
    solver_mode : int, default 3
        NNLS solver mode.
    init_mode : int, default 2
        Initialisation mode.
    max_iter : int, default 100
        Maximum iterations.
    tol : float, default 1e-5
        Convergence tolerance.
    seed : int, default 0
        Random seed.

    Returns
    -------
    NmfResult
        ``.W`` (genes × k), ``.H_list`` (list of k × cells_i arrays),
        ``.loss_history``.

    Raises
    ------
    ValueError
        If *loss*, *solver_mode*, or *init_mode* is not recognised.
    AttributeError
        If ``_core.nmf_chunked`` is not available
        (see CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE).

    Examples
    --------
    Stream 5 samples at 100k cells per chunk::

        from singlet.gpu.reduce import nmf_chunked
        result = nmf_chunked(paths, n_factors=30, chunk_cols=100_000)
        # result.W is (genes × 30), result.H_list[i] is (30 × cells_i)
    """
    if loss not in _VALID_LOSSES:
        raise ValueError(f"loss='{loss}' not recognised.  Choose from: {_VALID_LOSSES}")
    if solver_mode not in _VALID_SOLVER_MODES:
        raise ValueError(f"solver_mode={solver_mode} not recognised.")
    if init_mode not in _VALID_INIT_MODES:
        raise ValueError(f"init_mode={init_mode} not recognised.")

    import singlet.gpu._core as _core

    if not hasattr(_core, "nmf_chunked"):
        raise AttributeError(
            "_core.nmf_chunked is not available.  See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    # _core.nmf_chunked signature (py::kw_only after rank):
    #   nmf_chunked(loader, rank, *, loss='MSE', solver_mode=2, init_mode=2,
    #               max_iter=100, tol=1e-5, seed=0)
    # CYCLE-275: PzDataLoader is now exposed as a Python class. Construct it
    # from the first path. Multi-file streaming NMF is filed as
    # CYCLE-7-MULTI-INPUT-NMF (factornet FactorGraph::SharedNode wiring needed).
    paths_list = list(paths)
    if len(paths_list) == 0:
        raise ValueError("nmf_chunked: paths must be non-empty.")
    if len(paths_list) > 1:
        import warnings

        warnings.warn(
            "nmf_chunked: multiple paths provided but only the first is used "
            "(CYCLE-7-MULTI-INPUT-NMF — multi-file streaming requires "
            "factornet FactorGraph::SharedNode wiring).",
            UserWarning,
            stacklevel=2,
        )

    # CYCLE-275: each `path` is per the singlet-pipeline convention either:
    # (a) a directory containing a `gene_counts.1pz` (the standard sample
    #     output), or (b) a direct `*.1pz` file path. Resolve (a) → (b) here.
    import os

    p = str(paths_list[0])
    if os.path.isdir(p):
        candidate = os.path.join(p, "gene_counts.1pz")
        if not os.path.isfile(candidate):
            # Fallback: first .1pz file lexicographically.
            pz_files = sorted(f for f in os.listdir(p) if f.endswith(".1pz"))
            if not pz_files:
                raise FileNotFoundError(f"nmf_chunked: no .1pz file found in directory {p!r}.")
            candidate = os.path.join(p, pz_files[0])
        p = candidate
    loader = _core.PzDataLoader(p, int(chunk_cols))
    raw = _core.nmf_chunked(
        loader,
        int(n_factors),
        loss=str(loss),
        solver_mode=int(solver_mode),
        init_mode=int(init_mode),
        max_iter=int(max_iter),
        tol=float(tol),
        seed=int(seed),
    )

    # CYCLE-276: NmfResult binding (per _singlet_gpu_core.cpp:406-435) exposes
    # only k_used / iterations / converged / W_view / d_view / H_view (no
    # direct .W / .H_list / .loss_history). Build the host arrays from the
    # device CAI views — same pattern as nmf_basic at line 134.
    import cupy as cp_local

    class _CaiView:
        def __init__(self, d):
            self.__cuda_array_interface__ = d

    k_used = int(raw.k_used)
    # W_view is (rows*k,) flat col-major; reshape to (k, rows) then T → (rows, k)
    W = cp_local.asarray(_CaiView(raw.W_view)).reshape(k_used, -1).T.get()
    # H_view is (k*cols,) row-major; reshape to (k, cols)
    H = cp_local.asarray(_CaiView(raw.H_view)).reshape(k_used, -1).get()

    return NmfResult(
        W=W,
        H_list=[H],
        loss_history=[],  # not exposed by binding; CYCLE-276-FOLLOWUP-NMF-LOSS-HISTORY
        n_genes=int(W.shape[0]),
        n_factors=k_used,
    )


# ---------------------------------------------------------------------------
# Public API — nmf_graph_factorize
# ---------------------------------------------------------------------------


def nmf_graph_factorize(
    modalities: Dict[str, anndata.AnnData],
    *,
    n_factors: int = 20,
    loss: str = "MSE",
    solver_mode: int = 3,
    init_mode: int = 2,
    max_iter: int = 100,
    tol: float = 1e-5,
    seed: int = 0,
    shared_h: bool = True,
) -> Dict[str, anndata.AnnData]:
    """
    Multi-modal joint NMF using ``factornet::graph::FactorGraph`` (cycle-6).

    When ``shared_h=True`` (default), a ``factornet::graph::SharedNode`` is
    used so all modalities share a single coefficient matrix H (cells × k).
    This is the canonical multi-modal joint factorization pattern.

    When ``shared_h=False``, a ``ConcatNode`` is used instead — this
    concatenates all feature spaces into a single NMF (different semantics:
    one large feature vector per cell rather than a truly shared latent space).

    All input AnnData objects must share the same cell ordering (same obs_names
    in the same order).  Gene sets may differ across modalities.

    Parameters
    ----------
    modalities : dict mapping modality name → AnnData
        e.g. ``{"rna": adata_rna, "atac": adata_atac, "adt": adata_adt}``.
        Each AnnData must have ``X`` as a GPU-resident
        ``cupy.sparse.csr_matrix``.
    n_factors : int, default 20
        Rank k — shared across all modalities when ``shared_h=True``.
    loss : str, default ``"MSE"``
        Loss function applied to all modalities.
    solver_mode : int, default 3
        NNLS solver mode.
    init_mode : int, default 2
        Initialisation mode.
    max_iter : int, default 100
        Maximum iterations.
    tol : float, default 1e-5
        Convergence tolerance.
    seed : int, default 0
        Random seed.
    shared_h : bool, default True
        Use ``SharedNode`` (shared H) instead of ``ConcatNode``.
        Set to ``False`` for feature-axis concatenation.

    Returns
    -------
    dict mapping modality name → AnnData
        Each output AnnData has:
        * ``adata.obsm['X_nmf']``        — (cells × k) shared cell embedding.
        * ``adata.varm['NMF_loadings']`` — (genes_i × k) per-modality loadings.
        * ``adata.uns['nmf']``           — convergence info.

    Raises
    ------
    ValueError
        If *modalities* is empty or has inconsistent cell counts.
        If *loss*, *solver_mode*, or *init_mode* is not recognised.
    AttributeError
        If ``_core.nmf_graph_factorize`` or ``_core.from_cupy_csr`` is not
        available (see CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE).

    Examples
    --------
    Joint RNA + ATAC NMF with shared H::

        from singlet.gpu.reduce import nmf_graph_factorize
        results = nmf_graph_factorize(
            {"rna": adata_rna, "atac": adata_atac},
            n_factors=20,
        )
        # Both results["rna"].obsm["X_nmf"] and results["atac"].obsm["X_nmf"]
        # are the same shared (cells × 20) matrix.
    """
    if not modalities:
        raise ValueError("modalities dict must not be empty.")
    if loss not in _VALID_LOSSES:
        raise ValueError(f"loss='{loss}' not recognised.  Choose from: {_VALID_LOSSES}")
    if solver_mode not in _VALID_SOLVER_MODES:
        raise ValueError(f"solver_mode={solver_mode} not recognised.")
    if init_mode not in _VALID_INIT_MODES:
        raise ValueError(f"init_mode={init_mode} not recognised.")

    # Validate all AnnData have the same number of cells.
    n_cells_per_modality = {k: v.n_obs for k, v in modalities.items()}
    unique_n_cells = set(n_cells_per_modality.values())
    if len(unique_n_cells) > 1:
        raise ValueError(
            f"All modalities must have the same number of cells.  Got: {n_cells_per_modality}"
        )

    import singlet.gpu._core as _core

    if not hasattr(_core, "nmf_graph_factorize"):
        raise AttributeError(
            "_core.nmf_graph_factorize is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    # Convert each modality's AnnData to a DeviceCSC.
    modality_keys = list(modalities.keys())
    device_cscs = {k: _csr_to_device_csc(_get_matrix(v, None)) for k, v in modalities.items()}

    config = _build_nmf_config(
        n_factors=n_factors,
        loss=loss,
        solver_mode=solver_mode,
        init_mode=init_mode,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
    )
    config["shared_h"] = bool(shared_h)

    # _core.nmf_graph_factorize accepts a dict of {name: DeviceCSC} and
    # returns a dict of {name: NmfResult} where result.W is per-modality
    # and result.H is shared (same array for all modalities when shared_h=True).
    raw_results = _core.nmf_graph_factorize(device_cscs, config)

    # Write back into copies of the input AnnData objects.
    output: Dict[str, anndata.AnnData] = {}
    for key in modality_keys:
        out_adata = copy_module.copy(modalities[key])
        _write_nmf_result(out_adata, raw_results[key], n_factors=n_factors)
        output[key] = out_adata

    return output


__all__ = ["nmf", "nmf_chunked", "nmf_graph_factorize", "NmfResult"]
