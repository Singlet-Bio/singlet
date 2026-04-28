# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.reduce.svd — GPU-native PCA and low-level SVD backends.

Underlying C++ cycle: cycle 5 (reduce/svd/*.h — factornet 5-method SVD
adapters: lanczos, irlba, randomized, krylov_constrained, deflation,
auto_select).

The module exposes:

``pca``               — scanpy-compatible PCA entry point with backend auto-select.
``svd_lanczos``       — Golub–Kahan bidiagonalization (factornet::svd::lanczos_gpu).
``svd_irlba``         — implicitly-restarted Lanczos  (factornet::svd::irlba_gpu).
``svd_randomized``    — Halko–Martinsson–Tropp        (factornet::svd::randomized_gpu).
``svd_krylov``        — Krylov constrained            (factornet::svd::krylov_gpu).
``svd_deflation``     — rank-1 ALS with deflation     (factornet::svd::deflation_gpu).

All backends use implicit centering (``A(v) = X·v − μ·(1ᵀv)``) implemented
inside factornet — the mean vector μ is precomputed once on device and
passed through.  ``center=True`` is therefore the default and does not
materialise a dense centred matrix.

GPU execution path (pca)
------------------------
1. Extract ``adata.X`` (or ``adata.layers[layer]``) as
   ``cupy.sparse.csr_matrix``.
2. Convert CSR → CSC, wrap in ``singlet_gpu.DeviceCSC`` via ``from_cupy_csr``
   (CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE).
3. Call ``_core.pca`` (or the named backend) — factornet GPU kernels run,
   producing device-resident U, Σ, V matrices.
4. Transfer only the small output matrices (n_cells × n_comps, n_genes × n_comps,
   n_comps scalars) from device to host.  The input CSC stays device-resident.
5. Write into ``adata.obsm['X_pca']``, ``adata.varm['PCs']``,
   ``adata.uns['pca']``.

CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE: ``_core.pca``, ``_core.svd_*``,
and ``_core.from_cupy_csr`` are not yet exposed by the cycle-18 binding.
The wrappers here raise ``AttributeError`` with a clear message until those
bindings are extended.

Streams
-------
The factornet SVD kernels manage their own CUDA streams internally via
``factornet::gpu::GPUContext``.  The Python wrapper passes no explicit
stream — stream lifecycle is owned by the C++ side (style-rules §C.15:
streams passed in by caller, never created inside a kernel; here the "caller"
is the factornet GPUContext pool, not user Python code).
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import anndata
    import numpy as np

# ---------------------------------------------------------------------------
# Valid backend names (matching factornet routing table)
# ---------------------------------------------------------------------------
_VALID_BACKENDS = ("auto", "lanczos", "irlba", "randomized", "krylov", "deflation")


# ---------------------------------------------------------------------------
# Internal helpers (shared with preprocess — duplicated to avoid circular
# imports; a future refactor can move them to a _common.py)
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


def _write_pca_result(adata: "anndata.AnnData", result, *, n_comps: int) -> None:
    """
    Write a PCA result struct into the AnnData in scanpy layout.

    Expected result fields (all host numpy arrays after device transfer):
      .U       (cells  × n_comps) — left  singular vectors (cell embeddings)
      .V       (genes  × n_comps) — right singular vectors (gene loadings)
      .sigma   (n_comps,)         — singular values
    """
    import numpy as np

    # X_pca: cell embeddings (cells × n_comps)
    adata.obsm["X_pca"] = result.U.astype(np.float32, copy=False)

    # PCs: gene loadings (genes × n_comps)  — scanpy stores as varm['PCs']
    adata.varm["PCs"] = result.V.astype(np.float32, copy=False)

    # variance and variance_ratio from singular values
    sigma = result.sigma.astype(np.float64, copy=False)
    variance = sigma ** 2 / max(adata.n_obs - 1, 1)
    total_var = float(adata.X.multiply(adata.X).sum())  # Frobenius² / (n-1)
    denom = total_var / max(adata.n_obs - 1, 1) if total_var > 0 else 1.0
    variance_ratio = variance / denom

    adata.uns["pca"] = {
        "variance":       variance,
        "variance_ratio": variance_ratio,
        "params": {
            "n_comps": n_comps,
            "use_highly_variable": False,  # caller responsible for subsetting
        },
    }


# ---------------------------------------------------------------------------
# Public API — pca (scanpy-compatible)
# ---------------------------------------------------------------------------

def pca(
    adata: "anndata.AnnData",
    *,
    n_comps: int = 50,
    layer: Optional[str] = None,
    backend: str = "auto",
    center: bool = True,
    scale: bool = False,
    seed: int = 0,
    inplace: bool = True,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    GPU-native PCA via factornet 5-method SVD with auto-select (cycle-5 kernel).

    Mirrors ``scanpy.pp.pca`` — parameter names and defaults are identical.

    The backend is selected by factornet's ``auto_select`` routing table:

    * ``k < 32``        → Lanczos
    * ``32 ≤ k < 64``   → Randomized (Halko–Martinsson–Tropp, 3 power iters)
    * ``k ≥ 64``        → IRLBA

    Constrained paths (L1/L2/non-neg) are accessible via the low-level
    ``svd_krylov`` and ``svd_deflation`` functions.

    Parameters
    ----------
    adata : AnnData
        AnnData with ``X`` (or ``layers[layer]``) as a GPU-resident
        ``cupy.sparse.csr_matrix``.
    n_comps : int, default 50
        Number of principal components to compute.
    layer : str or None, default None
        Layer to use.  ``None`` operates on ``adata.X``.
    backend : str, default ``"auto"``
        SVD backend.  One of: ``"auto"`` (factornet routing), ``"lanczos"``,
        ``"irlba"``, ``"randomized"``, ``"krylov"``, ``"deflation"``.
    center : bool, default True
        Implicitly subtract the gene mean (``A(v) = X·v − μ·(1ᵀv)``).
        No dense matrix is materialised — factornet implements the shift.
    scale : bool, default False
        Scale to unit variance after centering.  Passed to factornet; does
        NOT materialise a dense scaled matrix.
    seed : int, default 0
        Random seed for stochastic backends (randomized, IRLBA restarts).
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
        If *backend* is not recognised.
    AttributeError
        If ``_core.pca`` or ``_core.from_cupy_csr`` is not available
        (see CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE).

    Examples
    --------
    Default (auto backend, 50 PCs, matches scanpy)::

        import singlet_gpu.reduce as sgr
        sgr.pca(adata)

    Force IRLBA with 100 PCs::

        sgr.pca(adata, n_comps=100, backend="irlba")
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"backend='{backend}' is not recognised.  "
            f"Choose one of: {_VALID_BACKENDS}"
        )

    import singlet_gpu._core as _core

    if not hasattr(_core, "pca"):
        raise AttributeError(
            "_core.pca is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    working_adata = adata if (inplace and not copy) else copy_module.copy(adata)
    mat = _get_matrix(working_adata, layer)
    device_csc = _csr_to_device_csc(mat)

    svd_config = {
        "n_comps":  int(n_comps),
        "backend":  str(backend),
        "center":   bool(center),
        "scale":    bool(scale),
        "seed":     int(seed),
    }

    result = _core.pca(device_csc, svd_config)
    _write_pca_result(working_adata, result, n_comps=n_comps)

    if inplace and not copy:
        return None
    return working_adata


# ---------------------------------------------------------------------------
# Low-level SVD backends (named entry points for advanced users)
# ---------------------------------------------------------------------------

def svd_lanczos(
    adata: "anndata.AnnData",
    *,
    n_comps: int = 50,
    layer: Optional[str] = None,
    center: bool = True,
    scale: bool = False,
    seed: int = 0,
    inplace: bool = True,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    Golub–Kahan Lanczos bidiagonalization SVD (cycle-5, factornet::svd::lanczos_gpu).

    Best for dominant components with ``k < 32``.  For larger ``k``,
    prefer ``svd_irlba`` or use ``pca`` with ``backend="auto"``.

    See ``pca`` for parameter descriptions.
    """
    import singlet_gpu._core as _core

    if not hasattr(_core, "svd_lanczos"):
        raise AttributeError(
            "_core.svd_lanczos is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    working_adata = adata if (inplace and not copy) else copy_module.copy(adata)
    mat = _get_matrix(working_adata, layer)
    device_csc = _csr_to_device_csc(mat)

    config = {"n_comps": int(n_comps), "center": bool(center),
               "scale": bool(scale), "seed": int(seed)}
    result = _core.svd_lanczos(device_csc, config)
    _write_pca_result(working_adata, result, n_comps=n_comps)

    if inplace and not copy:
        return None
    return working_adata


def svd_irlba(
    adata: "anndata.AnnData",
    *,
    n_comps: int = 50,
    layer: Optional[str] = None,
    center: bool = True,
    scale: bool = False,
    seed: int = 0,
    inplace: bool = True,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    factornet IRLBA (implicitly-restarted Lanczos) SVD backend.

    Best for ``k ≥ 64``.  See ``pca`` for parameter descriptions.
    """
    import singlet_gpu._core as _core

    if not hasattr(_core, "svd_irlba"):
        raise AttributeError(
            "_core.svd_irlba is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    working_adata = adata if (inplace and not copy) else copy_module.copy(adata)
    mat = _get_matrix(working_adata, layer)
    device_csc = _csr_to_device_csc(mat)

    config = {"n_comps": int(n_comps), "center": bool(center),
               "scale": bool(scale), "seed": int(seed)}
    result = _core.svd_irlba(device_csc, config)
    _write_pca_result(working_adata, result, n_comps=n_comps)

    if inplace and not copy:
        return None
    return working_adata


def svd_randomized(
    adata: "anndata.AnnData",
    *,
    n_comps: int = 50,
    n_oversampling: int = 10,
    n_power_iter: int = 3,
    layer: Optional[str] = None,
    center: bool = True,
    scale: bool = False,
    seed: int = 0,
    inplace: bool = True,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    Halko–Martinsson–Tropp randomized SVD (cycle-5, factornet::svd::randomized_gpu).

    Best for ``32 ≤ k < 64``.  Uses *n_power_iter* = 3 power iterations
    (factornet default) for accuracy.  See ``pca`` for standard parameter
    descriptions.

    Parameters
    ----------
    n_oversampling : int, default 10
        Number of oversampling columns added to the sketch (``p`` in HTM).
    n_power_iter : int, default 3
        Number of power iterations for accuracy improvement.
    """
    import singlet_gpu._core as _core

    if not hasattr(_core, "svd_randomized"):
        raise AttributeError(
            "_core.svd_randomized is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    working_adata = adata if (inplace and not copy) else copy_module.copy(adata)
    mat = _get_matrix(working_adata, layer)
    device_csc = _csr_to_device_csc(mat)

    config = {
        "n_comps":        int(n_comps),
        "n_oversampling": int(n_oversampling),
        "n_power_iter":   int(n_power_iter),
        "center":         bool(center),
        "scale":          bool(scale),
        "seed":           int(seed),
    }
    result = _core.svd_randomized(device_csc, config)
    _write_pca_result(working_adata, result, n_comps=n_comps)

    if inplace and not copy:
        return None
    return working_adata


def svd_krylov(
    adata: "anndata.AnnData",
    *,
    n_comps: int = 50,
    layer: Optional[str] = None,
    center: bool = True,
    scale: bool = False,
    seed: int = 0,
    l1: float = 0.0,
    l2: float = 0.0,
    non_negative: bool = False,
    inplace: bool = True,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    Krylov-constrained SVD backend (cycle-5, factornet::svd::krylov_gpu).

    Supports L1/L2 regularisation and non-negativity constraints on the
    singular vectors.  Best for constrained ``k ≥ 8``.

    Parameters
    ----------
    l1 : float, default 0.0
        L1 regularisation weight.
    l2 : float, default 0.0
        L2 regularisation weight.
    non_negative : bool, default False
        Enforce non-negativity on singular vectors.

    See ``pca`` for standard parameter descriptions.
    """
    import singlet_gpu._core as _core

    if not hasattr(_core, "svd_krylov"):
        raise AttributeError(
            "_core.svd_krylov is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    working_adata = adata if (inplace and not copy) else copy_module.copy(adata)
    mat = _get_matrix(working_adata, layer)
    device_csc = _csr_to_device_csc(mat)

    config = {
        "n_comps":      int(n_comps),
        "center":       bool(center),
        "scale":        bool(scale),
        "seed":         int(seed),
        "l1":           float(l1),
        "l2":           float(l2),
        "non_negative": bool(non_negative),
    }
    result = _core.svd_krylov(device_csc, config)
    _write_pca_result(working_adata, result, n_comps=n_comps)

    if inplace and not copy:
        return None
    return working_adata


def svd_deflation(
    adata: "anndata.AnnData",
    *,
    n_comps: int = 4,
    layer: Optional[str] = None,
    center: bool = True,
    scale: bool = False,
    seed: int = 0,
    robust: bool = False,
    inplace: bool = True,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    Rank-1 ALS with deflation SVD (cycle-5, factornet::svd::deflation_gpu).

    Supports Graph Laplacian and robust IRLS paths.  Best for constrained
    ``k < 8`` and robust low-rank decompositions.

    Parameters
    ----------
    robust : bool, default False
        Use IRLS (iteratively re-weighted least squares) for robustness to
        outliers.

    See ``pca`` for standard parameter descriptions.
    """
    import singlet_gpu._core as _core

    if not hasattr(_core, "svd_deflation"):
        raise AttributeError(
            "_core.svd_deflation is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    working_adata = adata if (inplace and not copy) else copy_module.copy(adata)
    mat = _get_matrix(working_adata, layer)
    device_csc = _csr_to_device_csc(mat)

    config = {
        "n_comps": int(n_comps),
        "center":  bool(center),
        "scale":   bool(scale),
        "seed":    int(seed),
        "robust":  bool(robust),
    }
    result = _core.svd_deflation(device_csc, config)
    _write_pca_result(working_adata, result, n_comps=n_comps)

    if inplace and not copy:
        return None
    return working_adata


__all__ = [
    "pca",
    "svd_lanczos",
    "svd_irlba",
    "svd_randomized",
    "svd_krylov",
    "svd_deflation",
]
