# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.preprocess.lognorm — GPU-native total-count normalisation + log1p.

Underlying C++ cycle: cycle 3 (preprocess/lognorm.h —
``singlet::gpu::preprocess::log_normalize`` fused kernel).

Both functions mirror the scanpy.pp.normalize_total / scanpy.pp.log1p
signatures exactly so they can be used as drop-in replacements:

    import singlet.gpu.preprocess as sgpp
    sgpp.normalize_total(adata)     # instead of sc.pp.normalize_total(adata)
    sgpp.log1p(adata)               # instead of sc.pp.log1p(adata)

GPU execution path
------------------
1. Extract ``adata.X`` (or ``adata.layers[layer]``) as a
   ``cupy.sparse.csr_matrix``.  This is already device-resident when the
   AnnData was loaded via ``read_pz_to_anndata``.
2. Convert CSR → CSC if needed (``_csr_to_device_csc``) so the C++ kernel
   can accept a column-major sparse input.
3. Call ``singlet.gpu._core.normalize_total`` / ``_core.log1p`` on the
   ``DeviceCSC`` — no host copies in the hot path.
4. Convert the result ``DeviceCSC`` back to ``cupy.sparse.csr_matrix`` via
   the ``__cuda_array_interface__`` protocol.

CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE: ``_core.normalize_total``,
``_core.log1p``, ``_core.from_cupy_csr``, and ``_core.to_cupy_csr`` are
placeholders awaiting implementation in the cycle-18 binding extension.  The
wrapper code here is written against those names and will raise an
``AttributeError`` until the binding is extended.  A dedicated follow-up
cycle must add those bindings before this module is functional.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_matrix(adata: "anndata.AnnData", layer: Optional[str]):
    """Return the matrix to operate on (X or a layer)."""
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        return adata.layers[layer]
    return adata.X


def _set_matrix(adata: "anndata.AnnData", layer: Optional[str], mat) -> None:
    """Write *mat* back to adata.X or adata.layers[layer]."""
    if layer is not None:
        adata.layers[layer] = mat
    else:
        adata.X = mat


def _csr_to_device_csc(csr_mat):
    """
    Convert a cupy.sparse.csr_matrix to a singlet.gpu.DeviceCsc.

    Requires ``_core.from_cupy_csr`` (CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE).

    The CSR is first converted to CSC (cupy in-place .tocsc()).  Then we
    extract the ``__cuda_array_interface__`` dicts for indptr / indices / data
    and pass the raw device pointers into ``_core.from_cupy_csr``.

    Parameters
    ----------
    csr_mat : cupy.sparse.csr_matrix  (cells × genes — AnnData convention)

    Returns
    -------
    singlet.gpu.DeviceCsc   (genes × cells — C++ kernel convention)
    """
    import singlet.gpu._core as _core  # noqa: F401  (will AttributeError if not built)

    if not hasattr(_core, "from_cupy_csr"):
        raise AttributeError(
            "_core.from_cupy_csr is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE — the cycle-18 "
            "pybind11 binding must be extended to expose from_cupy_csr / "
            "to_cupy_csr before the GPU preprocess kernels can be called."
        )

    # cupy >= 14: import from cupyx.scipy.sparse; fall back to cupy.sparse for < 14.
    try:
        import cupyx.scipy.sparse as csp  # cupy >= 14
    except ImportError:
        import cupy.sparse as csp  # cupy < 14 fallback  # noqa: F401

    # AnnData stores (cells × genes) CSR; the C++ kernel expects (genes × cells) CSC.
    # .T.tocsc() is in-place — no new data allocation.
    csc_mat = csr_mat.T.tocsc()  # genes × cells, CSC

    # _core.from_cupy_csr accepts any object exposing .indptr / .indices / .data
    # whose backing arrays expose __cuda_array_interface__ (cupy csr_/csc_matrix).
    # Pass the cupy matrix directly — the binding extracts pointers and shape.
    return _core.from_cupy_csr(csc_mat)


def _device_csc_to_csr(device_csc):
    """
    Convert a singlet.gpu.DeviceCsc back to a cupy.sparse.csr_matrix.

    Requires ``_core.to_cupy_csr`` (CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE).

    Uses the zero-copy ``__cuda_array_interface__`` views to build a cupy
    csc_matrix and then transposes to (cells × genes) csr — no host traffic.

    Parameters
    ----------
    device_csc : singlet.gpu.DeviceCsc  (genes × cells)

    Returns
    -------
    cupy.sparse.csr_matrix  (cells × genes — AnnData convention)
    """
    import cupy as cp

    import singlet.gpu._core as _core  # noqa: F401

    try:
        import cupyx.scipy.sparse as csp  # cupy >= 14
    except ImportError:
        import cupy.sparse as csp  # cupy < 14 fallback

    # cupy >= 14 dtype-strictness: cp.asarray() rejects bare dicts.  The
    # __cuda_array_interface__ must be an attribute on the source object.
    # Wrap each CAI dict in a one-line shim — works on cupy 13 and 14.
    class _CaiView:
        def __init__(self, d):
            self.__cuda_array_interface__ = d

    if hasattr(_core, "to_cupy_csr"):
        # _core.to_cupy_csr returns a dict {data, indices, indptr, shape, _owner}
        # whose data/indices/indptr values are themselves CAI dicts (from
        # make_view_object — see python/src/_cupy_interop.hpp).
        d = _core.to_cupy_csr(device_csc)
        rows, cols = d["shape"]
        cu_data = cp.asarray(_CaiView(d["data"]))
        cu_indices = cp.asarray(_CaiView(d["indices"]))
        cu_indptr = cp.asarray(_CaiView(d["indptr"]))
    else:
        # Fallback — use __cuda_array_interface__ views from DeviceCsc directly.
        cu_data = cp.asarray(_CaiView(device_csc.data_view))
        cu_indices = cp.asarray(_CaiView(device_csc.indices_view))
        cu_indptr = cp.asarray(_CaiView(device_csc.indptr_view))
        rows = device_csc.rows
        cols = device_csc.cols

    genes_x_cells_csc = csp.csc_matrix((cu_data, cu_indices, cu_indptr), shape=(rows, cols))
    result_csr = genes_x_cells_csc.T.tocsr()
    # Lifetime anchor (CYCLE-214 — fixes CYCLE-199 pca segfault):
    # cu_data/cu_indices/cu_indptr are zero-copy views into device_csc's memory.
    # Without this anchor, device_csc may be GC'd after the wrapper returns,
    # leaving result_csr's pointers dangling.  Subsequent kernel reads
    # (e.g. _core.pca on the next wrapper call) then segfault.  Stashing on
    # result_csr.__dict__ keeps the anchor for the whole lifetime of result_csr.
    try:
        result_csr.__dict__["_singlet_gpu_device_ref"] = device_csc
    except (AttributeError, TypeError):
        # cupy sparse may not expose __dict__ on some versions; fall back to
        # storing on the underlying data ndarray.
        try:
            result_csr.data._singlet_gpu_device_ref = device_csc
        except AttributeError:
            pass  # last-resort no-op; lifetime managed by Python local scope
    return result_csr


def _prepare_adata(
    adata: "anndata.AnnData",
    *,
    inplace: bool,
    copy: bool,
) -> "anndata.AnnData":
    """
    Return the AnnData to operate on based on scanpy inplace/copy semantics.

    scanpy convention:
    - ``inplace=True, copy=False`` (default): operate in-place, return None.
    - ``inplace=False, copy=True``:           operate on a deep copy, return copy.
    - ``inplace=True, copy=True``:            copy wins; operate on copy, return copy.
    - ``inplace=False, copy=False``:          no-op semantics — warn; treat as copy.
    """
    if copy or not inplace:
        return copy_module(adata)
    return adata


# Alias: we shadow the ``copy`` builtin locally — import the module once.
import copy as copy_module  # noqa: E402

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_total(
    adata: "anndata.AnnData",
    *,
    target_sum: Optional[float] = None,
    layer: Optional[str] = None,
    inplace: bool = True,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    Normalize total counts per cell to *target_sum* (GPU-native, cycle-3 kernel).

    Mirrors ``scanpy.pp.normalize_total`` exactly — parameter names and
    defaults are identical so this function is a drop-in replacement.

    Each cell is divided by the sum of its counts, then multiplied by
    ``target_sum``.  When ``target_sum=None``, the median total count across
    all cells is used as the scale factor (scanpy default).

    Parameters
    ----------
    adata : AnnData
        AnnData with ``X`` (or ``layers[layer]``) as a GPU-resident
        ``cupy.sparse.csr_matrix``.
    target_sum : float or None, default None
        Total count to normalise each cell to.  ``None`` uses the median
        total count (scanpy default).
    layer : str or None, default None
        Layer to normalise.  ``None`` operates on ``adata.X``.
    inplace : bool, default True
        Modify *adata* in place.  If ``False``, operate on a copy.
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
    AttributeError
        If ``_core.normalize_total`` or ``_core.from_cupy_csr`` is not
        available (see CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE).

    Examples
    --------
    In-place (default, matches scanpy)::

        import singlet.gpu.preprocess as sgpp
        sgpp.normalize_total(adata)                 # target_sum = median

    With explicit target::

        sgpp.normalize_total(adata, target_sum=1e4)

    On a layer::

        sgpp.normalize_total(adata, layer="raw_counts")
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "normalize_total"):
        raise AttributeError(
            "_core.normalize_total is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    # AnnData.copy() is a deep copy (incl. .X) — required because
    # _core.normalize_total / _core.log1p mutate device data in place.
    # `copy_module.copy()` is a SHALLOW copy that shares .X → original
    # gets mutated too.  See CYCLE-237.
    working_adata = adata if (inplace and not copy) else adata.copy()
    mat = _get_matrix(working_adata, layer)

    device_csc = _csr_to_device_csc(mat)

    # target_sum == None → pass 0.0 to signal "use median" in C++.
    # _core.normalize_total mutates device_csc in place and returns a
    # PyNormalizeResult (size_factors, qc_mask, target_used) — NOT a new matrix.
    # target_sum is keyword-only on the binding side.
    ts = float(target_sum) if target_sum is not None else 0.0
    norm_result = _core.normalize_total(device_csc, target_sum=ts)
    working_adata.uns.setdefault("singlet_gpu", {})["normalize_total"] = {
        "target_used": float(norm_result.target_used),
    }

    result_csr = _device_csc_to_csr(device_csc)
    _set_matrix(working_adata, layer, result_csr)

    if inplace and not copy:
        return None
    return working_adata


def log1p(
    adata: "anndata.AnnData",
    *,
    base: Optional[float] = None,
    layer: Optional[str] = None,
    inplace: bool = True,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    Compute log(1 + X) element-wise on device (GPU-native, cycle-3 kernel).

    Mirrors ``scanpy.pp.log1p`` exactly — parameter names and defaults are
    identical so this function is a drop-in replacement.

    Parameters
    ----------
    adata : AnnData
        AnnData with ``X`` (or ``layers[layer]``) as a GPU-resident
        ``cupy.sparse.csr_matrix``.
    base : float or None, default None
        Base of the logarithm.  ``None`` (default) uses the natural log.
        When *base* is set, the result is divided by ``log(base)``.
    layer : str or None, default None
        Layer to transform.  ``None`` operates on ``adata.X``.
    inplace : bool, default True
        Modify *adata* in place.  If ``False``, operate on a copy.
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
    AttributeError
        If ``_core.log1p`` or ``_core.from_cupy_csr`` is not available
        (see CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE).

    Examples
    --------
    Natural log1p (default, matches scanpy)::

        import singlet.gpu.preprocess as sgpp
        sgpp.log1p(adata)

    Base-2 log1p::

        sgpp.log1p(adata, base=2)
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "log1p"):
        raise AttributeError(
            "_core.log1p is not available.  See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    # AnnData.copy() is a deep copy (incl. .X) — required because
    # _core.normalize_total / _core.log1p mutate device data in place.
    # `copy_module.copy()` is a SHALLOW copy that shares .X → original
    # gets mutated too.  See CYCLE-237.
    working_adata = adata if (inplace and not copy) else adata.copy()
    mat = _get_matrix(working_adata, layer)

    device_csc = _csr_to_device_csc(mat)

    # base == None → pass 0.0 to signal "natural log" in C++.
    # _core.log1p mutates device_csc in place and returns None.
    # base is keyword-only on the binding side.
    b = float(base) if base is not None else 0.0
    _core.log1p(device_csc, base=b)

    result_csr = _device_csc_to_csr(device_csc)
    _set_matrix(working_adata, layer, result_csr)

    if inplace and not copy:
        return None
    return working_adata


__all__ = ["normalize_total", "log1p"]
