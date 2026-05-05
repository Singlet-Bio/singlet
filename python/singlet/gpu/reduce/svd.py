# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.reduce.svd — GPU-native PCA and low-level SVD backends.

After CYCLE-61 winner consolidation, the C++ surface ships only two backends
(deflation = primary, randomized = fallback). After CYCLE-105 these are native
internal kernels; factornet is no longer involved.

The module exposes:

``pca``               — scanpy-compatible PCA entry point with backend auto-select (recommended).
``svd_randomized``    — Halko–Martinsson–Tropp randomized SVD.
``svd_deflation``     — successive rank-1 deflation SVD.

**Deprecated since 0.1.0** (will be removed in 0.2.0 per state/release-policy.md):

``svd_lanczos``       — DEPRECATED. Use ``pca(backend="auto")``.
``svd_irlba``         — DEPRECATED. Use ``pca(backend="auto")``.
``svd_krylov``        — DEPRECATED. Use ``pca(backend="auto")``.

These three backends were dropped from the C++ surface in CYCLE-61 (Rule 32:
adopt-the-winner). The Python functions remain callable for one MINOR with a
``DeprecationWarning`` to give users time to migrate. Calling any of them
forwards to ``pca(backend="auto")`` which currently routes to ``deflation``
with ``randomized`` as automatic fallback.

All backends use implicit centering (``A(v) = X·v − μ·(1ᵀv)``) implemented
inside factornet — the mean vector μ is precomputed once on device and
passed through.  ``center=True`` is therefore the default and does not
materialise a dense centred matrix.

GPU execution path (pca)
------------------------
1. Extract ``adata.X`` (or ``adata.layers[layer]``) as
   ``cupy.sparse.csr_matrix``.
2. Convert CSR → CSC, wrap in ``singlet.gpu.DeviceCSC`` via ``from_cupy_csr``
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
    import singlet.gpu._core as _core

    if not hasattr(_core, "from_cupy_csr"):
        raise AttributeError(
            "_core.from_cupy_csr is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    csc_mat = csr_mat.T.tocsc()
    return _core.from_cupy_csr(csc_mat)


def _write_pca_result(adata: "anndata.AnnData", result, *, n_comps: int) -> None:
    """
    Write a PCA result struct into the AnnData in scanpy layout.

    _core.SvdResult exposes U_view / V_view / d_view as CAI dicts (per
    binding _bind_kernels.hpp:341 + §J.13 _CaiView shim required for cupy 14).
    """
    import numpy as np
    import cupy as cp

    # cupy 14 dtype-strict shim (CYCLE-189 / §J.13).
    class _CaiView:
        def __init__(self, d): self.__cuda_array_interface__ = d

    # PySvdResult fields (per _bind_kernels.hpp:upload_svd_result):
    #   d_U: shape (rows × k_selected) col-major
    #   d_V: shape (cols × k_selected) col-major
    #   d_d: k_selected singular values
    # Each *_view is a CAI dict from make_view_object.
    #
    # CYCLE-257 (CYCLE-199 root cause closed): SvdResult does NOT expose `rows`
    # / `cols` attributes via pybind11 (struct fields exist but only `__repr__`
    # uses them; see _singlet_gpu_core.cpp:360-397). The CYCLE-254 "host-side
    # segfault" was actually this AttributeError that surfaced once the
    # CYCLE-255 status-check audit removed the silent cuSPARSE failures
    # masking it. Wrapper transposed mat to (genes × cells) at line 97 so
    # rows = n_vars and cols = n_obs.
    n_rows = int(adata.n_vars)
    n_cols = int(adata.n_obs)
    k_sel  = int(result.k_selected)
    U = cp.asarray(_CaiView(result.U_view)).reshape(k_sel, n_rows).T.get()
    V = cp.asarray(_CaiView(result.V_view)).reshape(k_sel, n_cols).T.get()
    sigma = cp.asarray(_CaiView(result.d_view)).get()

    # X_pca: cell embeddings (cells × n_comps).  After our CSC layout swap
    # in the wrapper, "rows" of the binding are actually genes (the SVD
    # input was genes × cells), so U here is (genes × k) and V is (cells ×
    # k).  Restore scanpy convention: X_pca = cells × k = our V; PCs = our U.
    adata.obsm["X_pca"] = V.astype(np.float32, copy=False)
    adata.varm["PCs"]   = U.astype(np.float32, copy=False)

    sigma = sigma.astype(np.float64, copy=False)
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

        import singlet.gpu.reduce as sgr
        sgr.pca(adata)

    Force IRLBA with 100 PCs::

        sgr.pca(adata, n_comps=100, backend="irlba")
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"backend='{backend}' is not recognised.  "
            f"Choose one of: {_VALID_BACKENDS}"
        )

    import singlet.gpu._core as _core

    if not hasattr(_core, "pca"):
        raise AttributeError(
            "_core.pca is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    # CYCLE-274 (test_pca_inplace_vs_copy): copy_module.copy(adata) is a
    # shallow copy — adata.obsm is shared with the original, so writing
    # X_pca mutates the input. Use anndata's deep copy instead.
    working_adata = adata if (inplace and not copy) else adata.copy()
    mat = _get_matrix(working_adata, layer)
    device_csc = _csr_to_device_csc(mat)

    # _core.pca signature (py::kw_only after n_comps):
    #   pca(mat, n_comps, *, zero_center=True, scale=False, tol=1e-6,
    #       max_iter=100, seed=0)
    # backend is selected automatically by the C++ side (see svd_auto_select).
    result = _core.pca(
        device_csc, int(n_comps),
        zero_center=bool(center),
        scale=bool(scale),
        seed=int(seed),
    )
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
    DEPRECATED (since 0.1.0; removed in 0.2.0).

    The ``lanczos`` SVD backend was dropped from the C++ surface in CYCLE-61
    (Rule 32 winner consolidation: deflation dominates at every k). Calls now
    forward to ``pca(backend="auto")`` which routes to deflation with
    randomized fallback.
    """
    import warnings
    warnings.warn(
        "svd_lanczos is deprecated since singlet 2.0.0 and will be removed "
        "in 0.2.0. Use pca(backend='auto') instead — it routes to the deflation "
        "winner with randomized fallback. See state/release-policy.md.",
        DeprecationWarning, stacklevel=2,
    )
    return pca(adata, n_comps=n_comps, layer=layer, backend="auto",
               center=center, scale=scale, seed=seed, inplace=inplace, copy=copy)


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
    DEPRECATED (since 0.1.0; removed in 0.2.0).

    The ``irlba`` (implicitly-restarted Lanczos) SVD backend was dropped from
    the C++ surface in CYCLE-61 (Rule 32 winner consolidation). Calls now
    forward to ``pca(backend="auto")``.
    """
    import warnings
    warnings.warn(
        "svd_irlba is deprecated since singlet 2.0.0 and will be removed "
        "in 0.2.0. Use pca(backend='auto') instead. See state/release-policy.md.",
        DeprecationWarning, stacklevel=2,
    )
    return pca(adata, n_comps=n_comps, layer=layer, backend="auto",
               center=center, scale=scale, seed=seed, inplace=inplace, copy=copy)


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
    import singlet.gpu._core as _core

    if not hasattr(_core, "svd_randomized"):
        raise AttributeError(
            "_core.svd_randomized is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    # CYCLE-274 (test_pca_inplace_vs_copy): copy_module.copy(adata) is a
    # shallow copy — adata.obsm is shared with the original, so writing
    # X_pca mutates the input. Use anndata's deep copy instead.
    working_adata = adata if (inplace and not copy) else adata.copy()
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
    DEPRECATED (since 0.1.0; removed in 0.2.0).

    The Krylov-constrained SVD backend (with L1/L2/non-negativity constraints)
    was dropped from the C++ surface in CYCLE-61 (Rule 32 winner consolidation;
    deflation handles all the constraint types Krylov did). Calls now forward
    to ``pca(backend="auto")``. The ``l1``/``l2``/``non_negative`` parameters
    are silently ignored — pass through ``reduce::nmf::fit`` instead if you need
    constrained factorisation.
    """
    import warnings
    warnings.warn(
        "svd_krylov is deprecated since singlet 2.0.0 and will be removed "
        "in 0.2.0. Use pca(backend='auto') for unconstrained SVD; for "
        "constrained factorisation use reduce.nmf.nmf instead. The l1/l2/"
        "non_negative parameters are silently ignored on this deprecated path. "
        "See state/release-policy.md.",
        DeprecationWarning, stacklevel=2,
    )
    return pca(adata, n_comps=n_comps, layer=layer, backend="auto",
               center=center, scale=scale, seed=seed, inplace=inplace, copy=copy)


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
    import singlet.gpu._core as _core

    if not hasattr(_core, "svd_deflation"):
        raise AttributeError(
            "_core.svd_deflation is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    # CYCLE-274 (test_pca_inplace_vs_copy): copy_module.copy(adata) is a
    # shallow copy — adata.obsm is shared with the original, so writing
    # X_pca mutates the input. Use anndata's deep copy instead.
    working_adata = adata if (inplace and not copy) else adata.copy()
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
