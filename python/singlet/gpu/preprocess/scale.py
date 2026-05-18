# SPDX-License-Identifier: MIT
"""
singlet.gpu.preprocess.scale — GPU-native scale + regress_out.

Underlying C++: cycle-103, ``preprocess/scale.h``.
Matches scanpy.pp.scale / scanpy.pp.regress_out signatures.

Functions
---------
scale(adata, max_value=10.0, zero_center=True, layer=None, inplace=True, copy=False)
    Z-score scale expression values gene-wise.

regress_out(adata, keys, layer=None, inplace=True, copy=False)
    Remove the linear effect of confounders via OLS on device.
    ``keys`` is a list of adata.obs column names; builds the design matrix from them.
    At most 32 covariates are supported (C++ QR budget); raises ValueError otherwise.

GPU execution path (scale)
--------------------------
1. Extract adata.var['mean_counts'] / 'var_counts' (pre-computed by calculate_qc_metrics
   or the HVG kernel). If absent, run calculate_qc_metrics first.
2. Upload mean / std vectors to device as cupy arrays.
3. Call _core.scale(DeviceCsc, mean, std, cfg) → DenseResult (n_genes × n_cells,
   row-major float32 on device).
4. Store DenseResult in adata.layers['scale_dense'] (device-resident) so that
   regress_out can consume it without an extra copy.
5. Return None (or copy).

GPU execution path (regress_out)
---------------------------------
1. Retrieve DenseResult from adata.layers['scale_dense'].  If absent, call scale() first.
2. Build design matrix C [n_cells × p] from adata.obs[keys] as column-major float32.
3. Upload C to device.
4. Call _core.regress_out(DenseResult, C, p) → in-place modification.
5. No return value (None).
"""

from __future__ import annotations

import copy as copy_module
import math
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SCALE_LAYER = "_singlet_gpu_scale_dense"   # internal cache key in adata.layers


def _get_matrix(adata: "anndata.AnnData", layer: Optional[str]):
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        return adata.layers[layer]
    return adata.X


def _csr_to_device_csc(csr_mat):
    import singlet.gpu._core as _core
    if not hasattr(_core, "from_cupy_csr"):
        raise AttributeError(
            "_core.from_cupy_csr is not available.  "
            "Ensure pybind11 extension compiled with SINGLET_GPU_BUILD_PYTHON=ON."
        )
    csc_mat = csr_mat.T.tocsc()
    return _core.from_cupy_csr(csc_mat)


def _ensure_gene_stats(adata: "anndata.AnnData", layer: Optional[str],
                       stream) -> None:
    """Run calculate_qc_metrics if mean_counts / var_counts are missing."""
    if "mean_counts" not in adata.var.columns or "var_counts" not in adata.var.columns:
        from singlet.gpu.qc.qc_metrics import calculate_qc_metrics
        calculate_qc_metrics(adata, layer=layer, inplace=True, stream=stream)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scale(
    adata: "anndata.AnnData",
    *,
    max_value: float = 10.0,
    zero_center: bool = True,
    layer: Optional[str] = None,
    inplace: bool = True,
    copy: bool = False,
    stream=None,
) -> Optional["anndata.AnnData"]:
    """
    Z-score scale expression matrix gene-wise (cycle-103, ``preprocess::scale``).

    Mirrors ``scanpy.pp.scale``.  Converts the sparse CSC matrix to a dense
    row-major float32 matrix applying::

        scaled[g, c] = clip((x[g,c] - mean[g]) / std[g], -max_value, max_value)

    Pre-computed ``mean_counts`` and ``var_counts`` are read from ``adata.var``;
    if absent, ``calculate_qc_metrics`` is run automatically.

    The resulting dense device matrix is stored in
    ``adata.layers['_singlet_gpu_scale_dense']`` as a ``DenseResult`` object
    (device-resident, no host copy).  Pass this object directly to
    ``regress_out`` for zero-copy in-place correction.

    Parameters
    ----------
    adata : AnnData
    max_value : float, default 10.0
        Clip scaled values to [-max_value, max_value].  Set to inf to disable.
    zero_center : bool, default True
        Subtract per-gene mean.  False = only unit-variance scaling.
    layer : str or None
    inplace : bool, default True
    copy : bool, default False
    stream : int or None

    Returns
    -------
    None (inplace) or AnnData (copy).
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "scale"):
        raise AttributeError(
            "_core.scale is not available.  "
            "Compile with SINGLET_GPU_BUILD_PYTHON=ON (cycle-103 binding)."
        )

    try:
        import cupy as cp
    except ImportError as e:
        raise ImportError(f"singlet.gpu.preprocess.scale requires cupy.  {e}")

    working = adata if (inplace and not copy) else copy_module.copy(adata)

    _ensure_gene_stats(working, layer, stream)

    mat = _get_matrix(working, layer)
    device_csc = _csr_to_device_csc(mat)

    # Build device mean/std arrays from adata.var.
    mean_np = working.var["mean_counts"].values.astype(np.float32)
    var_np  = working.var["var_counts"].values.astype(np.float32)
    # std = sqrt(var), guarded against negative values from fp32 rounding.
    std_np = np.sqrt(np.maximum(var_np, 0.0)).astype(np.float32)

    d_mean = cp.asarray(mean_np)
    d_std  = cp.asarray(std_np)

    dense_result = _core.scale(
        device_csc,
        d_mean, d_std,
        max_value=float(max_value),
        zero_center=zero_center,
        unit_variance=True,
        stream=stream,
    )

    # Store device-resident DenseResult so regress_out can consume it.
    working.layers[_SCALE_LAYER] = dense_result

    # Also write a scaled_data annotation so users know the layer is available.
    working.uns["scale"] = {
        "max_value":   max_value,
        "zero_center": zero_center,
        "layer_key":   _SCALE_LAYER,
    }

    if inplace and not copy:
        return None
    return working


def regress_out(
    adata: "anndata.AnnData",
    keys: Sequence[str],
    *,
    layer: Optional[str] = None,
    inplace: bool = True,
    copy: bool = False,
    stream=None,
) -> Optional["anndata.AnnData"]:
    """
    Remove the linear effect of confounders in-place (cycle-103, ``preprocess::regress_out``).

    Mirrors ``scanpy.pp.regress_out``.  Uses QR factorization on device
    (cuSOLVER + cuBLAS) to OLS-regress out the columns of ``adata.obs[keys]``
    from the scaled expression matrix.

    Calls ``scale()`` automatically if the dense matrix is not already cached in
    ``adata.layers['_singlet_gpu_scale_dense']``.

    Parameters
    ----------
    adata : AnnData
    keys : list of str
        Names of ``adata.obs`` columns to use as covariates.  All columns must
        be numeric (float32-castable).  At most 32 columns are supported.
    layer : str or None
        Source sparse layer passed to scale() if a pre-scaled matrix is absent.
    inplace : bool, default True
    copy : bool, default False
    stream : int or None

    Returns
    -------
    None (inplace) or AnnData (copy).

    Raises
    ------
    ValueError
        If ``len(keys) > 32`` (C++ QR budget: p ≤ 32).
    KeyError
        If any key in ``keys`` is not in ``adata.obs``.
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "regress_out"):
        raise AttributeError(
            "_core.regress_out is not available.  "
            "Compile with SINGLET_GPU_BUILD_PYTHON=ON (cycle-103 binding)."
        )

    p = len(keys)
    if p == 0:
        raise ValueError("regress_out: keys must be non-empty.")
    if p > 32:
        raise ValueError(
            f"regress_out: {p} covariate keys supplied but the C++ QR kernel "
            "supports at most p=32.  Reduce the number of keys."
        )

    # Validate all keys exist.
    missing = [k for k in keys if k not in adata.obs.columns]
    if missing:
        raise KeyError(
            f"regress_out: keys {missing} not found in adata.obs.  "
            "Available columns: " + str(list(adata.obs.columns[:20]))
        )

    try:
        import cupy as cp
    except ImportError as e:
        raise ImportError(f"singlet.gpu.preprocess.regress_out requires cupy.  {e}")

    working = adata if (inplace and not copy) else copy_module.copy(adata)

    # Ensure scaled dense matrix exists.
    if _SCALE_LAYER not in working.layers:
        scale(working, layer=layer, inplace=True, stream=stream)

    dense_result = working.layers[_SCALE_LAYER]

    # Validate it's a DenseResult (not a cupy/numpy array from a different pipeline).
    if not hasattr(dense_result, "data_view"):
        raise TypeError(
            f"adata.layers['{_SCALE_LAYER}'] is not a DenseResult object.  "
            "Re-run singlet.gpu.preprocess.scale(adata) to regenerate it."
        )

    # Build design matrix C [n_cells × p] col-major float32.
    # scanpy convention: intercept is NOT included; user passes explicit obs columns.
    n_cells = working.n_obs
    C_host = np.zeros((n_cells, p), dtype=np.float32, order="F")  # col-major
    for j, key in enumerate(keys):
        col = working.obs[key].values
        C_host[:, j] = col.astype(np.float32)

    d_C = cp.asarray(C_host)   # device, col-major [n_cells × p]

    # In-place regression: modifies dense_result.dense.data in-place on device.
    _core.regress_out(dense_result, d_C, p=p, stream=stream)

    # Annotate uns for provenance.
    working.uns["regress_out"] = {"keys": list(keys), "p": p}

    if inplace and not copy:
        return None
    return working


__all__ = ["scale", "regress_out"]
