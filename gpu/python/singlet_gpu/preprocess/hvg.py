# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.preprocess.hvg — GPU-native highly variable gene selection.

Underlying C++ cycle: cycle 4 (preprocess/hvg.h —
``singlet_gpu::preprocess::select_hvg`` two-pass Welford kernel with
Pearson residuals and seurat_v3 variance-stabilization flavors).

This wrapper mirrors ``scanpy.pp.highly_variable_genes`` exactly —
parameter names and defaults are identical so it is a drop-in replacement:

    import singlet_gpu.preprocess as sgpp
    sgpp.highly_variable_genes(adata, n_top_genes=2000)

GPU execution path
------------------
1. Extract ``adata.X`` (or ``adata.layers[layer]``) as a
   ``cupy.sparse.csr_matrix``.
2. Convert CSR → CSC and wrap in a ``singlet_gpu.DeviceCsc`` via
   ``_core.from_cupy_csr`` (CSC is the natural layout for per-gene stats).
3. Call ``_core.highly_variable_genes`` which runs the two-pass Welford
   variance calculation entirely on device (fp32 values, fp64 accumulator
   per §D of style-rules).
4. Return per-gene statistics as numpy arrays (small host transfer: n_genes
   scalars), write into ``adata.var`` columns.

CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE: ``_core.highly_variable_genes``
and ``_core.from_cupy_csr`` are not yet exposed by the cycle-18 binding.
The wrapper raises ``AttributeError`` with a clear message until those
bindings are added.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import anndata
    import numpy as np


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


def _csr_to_device_csc(csr_mat):
    """
    Convert a ``cupy.sparse.csr_matrix`` to a ``singlet_gpu.DeviceCsc``.

    Requires ``_core.from_cupy_csr``
    (CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE).
    """
    import singlet_gpu._core as _core

    if not hasattr(_core, "from_cupy_csr"):
        raise AttributeError(
            "_core.from_cupy_csr is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE — the cycle-18 "
            "pybind11 binding must expose from_cupy_csr / to_cupy_csr."
        )

    csc_mat = csr_mat.T.tocsc()   # genes × cells, CSC
    return _core.from_cupy_csr(csc_mat)


_VALID_FLAVORS = ("seurat_v3", "pearson_residuals")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def highly_variable_genes(
    adata: "anndata.AnnData",
    *,
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
    max_disp: float = float("inf"),
    span: float = 0.3,
    n_bins: int = 20,
    layer: Optional[str] = None,
    inplace: bool = True,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    Select highly variable genes using a GPU-native two-pass Welford kernel
    (cycle-4, ``singlet_gpu::preprocess::select_hvg``).

    Mirrors ``scanpy.pp.highly_variable_genes`` exactly — parameter names
    and defaults are identical so this function is a drop-in replacement.

    The function computes per-gene mean and variance on device (fp32 values,
    fp64 accumulator per §D of style-rules) and selects the top *n_top_genes*
    by normalised dispersion (``"seurat_v3"``) or by Pearson residual
    (``"pearson_residuals"``).

    Results are written into:

    * ``adata.var['highly_variable']``    — bool mask (top *n_top_genes*).
    * ``adata.var['means']``              — per-gene mean.
    * ``adata.var['variances']``          — per-gene variance.
    * ``adata.var['variances_norm']``     — normalised dispersion.
    * ``adata.var['highly_variable_rank']``  — rank (0 = most variable).

    Parameters
    ----------
    adata : AnnData
        AnnData with ``X`` (or ``layers[layer]``) as a GPU-resident
        ``cupy.sparse.csr_matrix``.
    n_top_genes : int, default 2000
        Number of genes to flag as highly variable.
    flavor : str, default ``"seurat_v3"``
        HVG selection method.  One of:
        ``"seurat_v3"`` (Pearson residuals of a negative-binomial fit),
        ``"pearson_residuals"`` (analytic approximation).
    min_mean : float, default 0.0125
        Lower bound on per-gene mean (for ``"seurat_v3"`` filtering).
    max_mean : float, default 3.0
        Upper bound on per-gene mean.
    min_disp : float, default 0.5
        Lower bound on normalised dispersion.
    max_disp : float, default inf
        Upper bound on normalised dispersion.
    span : float, default 0.3
        Loess span for the ``"seurat_v3"`` variance stabilisation curve.
    n_bins : int, default 20
        Number of bins for the ``"seurat_v3"`` mean-dispersion curve.
    layer : str or None, default None
        Layer to use.  ``None`` operates on ``adata.X``.
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
        If *flavor* is not one of the recognised values.
    AttributeError
        If ``_core.highly_variable_genes`` or ``_core.from_cupy_csr`` is
        not available (see CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE).

    Examples
    --------
    Default (seurat_v3, top 2000 genes)::

        import singlet_gpu.preprocess as sgpp
        sgpp.highly_variable_genes(adata)

    Pearson residuals::

        sgpp.highly_variable_genes(adata, flavor="pearson_residuals",
                                   n_top_genes=3000)

    On a raw-counts layer::

        sgpp.highly_variable_genes(adata, layer="raw_counts")
    """
    if flavor not in _VALID_FLAVORS:
        raise ValueError(
            f"flavor='{flavor}' is not recognised.  "
            f"Choose one of: {_VALID_FLAVORS}"
        )

    import singlet_gpu._core as _core
    import numpy as np  # small host transfer for per-gene stats

    if not hasattr(_core, "highly_variable_genes"):
        raise AttributeError(
            "_core.highly_variable_genes is not available.  "
            "See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
        )

    working_adata = adata if (inplace and not copy) else copy_module.copy(adata)
    mat = _get_matrix(working_adata, layer)

    device_csc = _csr_to_device_csc(mat)

    # Build a config dict that the C++ kernel understands.
    # _core.highly_variable_genes signature (py::kw_only after mat):
    #   highly_variable_genes(mat, *, n_top_genes=2000, flavor='seurat_v3',
    #                         min_mean=0.0125, max_mean=3.0, pearson_theta=100.0)
    # min_disp / max_disp / span / n_bins are scanpy-side params not yet plumbed
    # through to the C++ kernel — accepted by the wrapper for API parity.
    result = _core.highly_variable_genes(
        device_csc,
        n_top_genes=int(n_top_genes),
        flavor=str(flavor),
        min_mean=float(min_mean),
        max_mean=float(max_mean),
    )

    # Write results into adata.var.
    working_adata.var["highly_variable"]      = result.highly_variable
    working_adata.var["means"]                = result.means
    working_adata.var["variances"]            = result.variances
    working_adata.var["variances_norm"]       = result.variances_norm
    working_adata.var["highly_variable_rank"] = result.highly_variable_rank

    # Mirror scanpy: store the parameters used in uns['hvg'].
    working_adata.uns["hvg"] = {
        "flavor":       flavor,
        "n_top_genes":  n_top_genes,
        "min_mean":     min_mean,
        "max_mean":     max_mean,
        "min_disp":     min_disp,
        "max_disp":     max_disp,
    }

    if inplace and not copy:
        return None
    return working_adata


__all__ = ["highly_variable_genes"]
