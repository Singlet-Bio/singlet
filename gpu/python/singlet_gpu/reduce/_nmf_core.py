# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.reduce.nmf — GPU-native NMF via factornet.

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
    ``singlet_gpu::io::PzDataLoader`` (which implements
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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    import anndata
    import numpy as np


# ---------------------------------------------------------------------------
# Valid parameter sets
# ---------------------------------------------------------------------------
_VALID_LOSSES = ("MSE", "KL", "NB", "GP", "Gamma", "Tweedie")
_VALID_SOLVER_MODES = (0, 1, 2, 3)
_VALID_INIT_MODES = (0, 1, 2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_matrix(adata: "anndata.AnnData", layer: Optional[str]):
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        return adata.layers[layer]
    return adata.X


def _csr_to_device_csc(csr_mat):
    """CSR (cells × genes) → DeviceCSC (genes × cells)."""
    import singlet_gpu._core as _core

    if not hasattr(_core, "from_cupy_csr"):
        raise AttributeError(
            "_core.from_cupy_csr is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )
    csc_mat = csr_mat.T.tocsc()
    rows, cols = csc_mat.shape
    return _core.from_cupy_csr(
        csc_mat.data.__cuda_array_interface__,
        csc_mat.indices.__cuda_array_interface__,
        csc_mat.indptr.__cuda_array_interface__,
        rows, cols, int(csc_mat.nnz),
    )


def _write_nmf_result(
    adata: "anndata.AnnData",
    result,
    *,
    n_factors: int,
    modality_key: str = "X_nmf",
) -> None:
    """
    Write a factornet NmfResult struct into an AnnData.

    Expected result fields (host numpy arrays after device transfer):
      .W   (genes  × k) — basis matrix (gene loadings)
      .H   (k      × cells) — coefficient matrix (cell scores);
                               we transpose to (cells × k) for obsm
      .loss_history (list of floats) — per-iteration loss
    """
    import numpy as np

    # obsm: (cells × k) cell embedding — H transposed
    H_T = result.H.T.astype(np.float32, copy=False)  # (cells × k)
    adata.obsm[modality_key] = H_T

    # varm: (genes × k) gene loadings — W
    adata.varm["NMF_loadings"] = result.W.astype(np.float32, copy=False)

    # uns: store convergence + parameters
    adata.uns["nmf"] = {
        "n_factors":     n_factors,
        "loss_history":  result.loss_history,
        "final_loss":    float(result.loss_history[-1]) if result.loss_history else float("nan"),
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
        "n_factors":   int(n_factors),
        "loss":        str(loss),
        "solver_mode": int(solver_mode),
        "init_mode":   int(init_mode),
        "max_iter":    int(max_iter),
        "tol":         float(tol),
        "seed":        int(seed),
    }


# ---------------------------------------------------------------------------
# Public API — nmf
# ---------------------------------------------------------------------------

def nmf(
    adata: "anndata.AnnData",
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
) -> Optional["anndata.AnnData"]:
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

        import singlet_gpu.reduce as sgr
        sgr.nmf(adata)

    Negative Binomial loss (count data)::

        sgr.nmf(adata, loss="NB", n_factors=30)
    """
    if loss not in _VALID_LOSSES:
        raise ValueError(f"loss='{loss}' not recognised.  Choose from: {_VALID_LOSSES}")
    if solver_mode not in _VALID_SOLVER_MODES:
        raise ValueError(f"solver_mode={solver_mode} not recognised.  Choose from: {_VALID_SOLVER_MODES}")
    if init_mode not in _VALID_INIT_MODES:
        raise ValueError(f"init_mode={init_mode} not recognised.  Choose from: {_VALID_INIT_MODES}")

    import singlet_gpu._core as _core

    if not hasattr(_core, "nmf"):
        raise AttributeError(
            "_core.nmf is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    working_adata = adata if (inplace and not copy) else copy_module.copy(adata)
    mat = _get_matrix(working_adata, layer)
    device_csc = _csr_to_device_csc(mat)

    config = _build_nmf_config(
        n_factors=n_factors, loss=loss, solver_mode=solver_mode,
        init_mode=init_mode, max_iter=max_iter, tol=tol, seed=seed,
    )
    result = _core.nmf(device_csc, config)
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
) -> "NmfResult":
    """
    Out-of-core streaming NMF over multiple ``.1pz`` files (cycle-6 kernel).

    Uses ``singlet_gpu::io::PzDataLoader`` as a ``factornet::io::DataLoader``
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

        from singlet_gpu.reduce import nmf_chunked
        result = nmf_chunked(paths, n_factors=30, chunk_cols=100_000)
        # result.W is (genes × 30), result.H_list[i] is (30 × cells_i)
    """
    if loss not in _VALID_LOSSES:
        raise ValueError(f"loss='{loss}' not recognised.  Choose from: {_VALID_LOSSES}")
    if solver_mode not in _VALID_SOLVER_MODES:
        raise ValueError(f"solver_mode={solver_mode} not recognised.")
    if init_mode not in _VALID_INIT_MODES:
        raise ValueError(f"init_mode={init_mode} not recognised.")

    import singlet_gpu._core as _core

    if not hasattr(_core, "nmf_chunked"):
        raise AttributeError(
            "_core.nmf_chunked is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    config = _build_nmf_config(
        n_factors=n_factors, loss=loss, solver_mode=solver_mode,
        init_mode=init_mode, max_iter=max_iter, tol=tol, seed=seed,
    )
    config["chunk_cols"] = int(chunk_cols)

    # _core.nmf_chunked accepts a list of paths and returns a raw result struct
    # whose fields mirror NmfResult.
    raw = _core.nmf_chunked(list(paths), config)

    return NmfResult(
        W=raw.W,
        H_list=raw.H_list,
        loss_history=raw.loss_history,
        n_genes=int(raw.W.shape[0]),
        n_factors=int(n_factors),
    )


# ---------------------------------------------------------------------------
# Public API — nmf_graph_factorize
# ---------------------------------------------------------------------------

def nmf_graph_factorize(
    modalities: Dict[str, "anndata.AnnData"],
    *,
    n_factors: int = 20,
    loss: str = "MSE",
    solver_mode: int = 3,
    init_mode: int = 2,
    max_iter: int = 100,
    tol: float = 1e-5,
    seed: int = 0,
    shared_h: bool = True,
) -> Dict[str, "anndata.AnnData"]:
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

        from singlet_gpu.reduce import nmf_graph_factorize
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
            f"All modalities must have the same number of cells.  "
            f"Got: {n_cells_per_modality}"
        )

    import singlet_gpu._core as _core

    if not hasattr(_core, "nmf_graph_factorize"):
        raise AttributeError(
            "_core.nmf_graph_factorize is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    # Convert each modality's AnnData to a DeviceCSC.
    modality_keys = list(modalities.keys())
    device_cscs = {k: _csr_to_device_csc(_get_matrix(v, None))
                   for k, v in modalities.items()}

    config = _build_nmf_config(
        n_factors=n_factors, loss=loss, solver_mode=solver_mode,
        init_mode=init_mode, max_iter=max_iter, tol=tol, seed=seed,
    )
    config["shared_h"] = bool(shared_h)

    # _core.nmf_graph_factorize accepts a dict of {name: DeviceCSC} and
    # returns a dict of {name: NmfResult} where result.W is per-modality
    # and result.H is shared (same array for all modalities when shared_h=True).
    raw_results = _core.nmf_graph_factorize(device_cscs, config)

    # Write back into copies of the input AnnData objects.
    output: Dict[str, "anndata.AnnData"] = {}
    for key in modality_keys:
        out_adata = copy_module.copy(modalities[key])
        _write_nmf_result(out_adata, raw_results[key], n_factors=n_factors)
        output[key] = out_adata

    return output


__all__ = ["nmf", "nmf_chunked", "nmf_graph_factorize", "NmfResult"]
