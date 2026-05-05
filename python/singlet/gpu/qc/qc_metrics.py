# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.qc.qc_metrics — GPU-native QC metrics + cell/gene filtering.

Underlying C++: cycle-103, ``qc/metrics.h``.
Matches scanpy.pp.calculate_qc_metrics / filter_cells / filter_genes signatures.

Functions
---------
calculate_qc_metrics(adata, qc_vars=("MT", "RIBO"), inplace=True)
    Compute per-cell and per-gene QC statistics.  Writes into adata.obs and
    adata.var exactly as scanpy does.

filter_cells(adata, min_genes, max_genes, min_counts, max_counts, inplace=True)
    Remove cells that fail QC thresholds.  Runs calculate_qc_metrics first
    if the required obs columns are absent.

filter_genes(adata, min_cells, min_counts, inplace=True)
    Remove genes expressed in too few cells.

GPU execution path
------------------
1. Build uint8 gene masks for MT / RIBO prefixes from adata.var.index.
2. Upload masks to device via cupy.
3. Call _core.calculate_qc_metrics(DeviceCsc, is_mt, is_ribo).
4. Download small per-cell / per-gene arrays to host (O(n_cells + n_genes) scalars).
5. Write into adata.obs / adata.var.

For filter_cells/filter_genes:
- Re-run calculate_qc_metrics on device if QC columns are missing.
- Build a boolean keep mask on host from the obs/var thresholds.
- Subset adata[:, keep] / adata[keep, :] — scanpy convention.
- The C++ filter_cells / filter_genes kernels produce the filtered DeviceCsc
  which replaces adata.X.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import anndata


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
    """Convert cupy.sparse.csr_matrix → singlet_gpu DeviceCsc."""
    import singlet.gpu._core as _core
    if not hasattr(_core, "from_cupy_csr"):
        raise AttributeError(
            "_core.from_cupy_csr is not available.  "
            "Ensure pybind11 extension compiled with SINGLET_GPU_BUILD_PYTHON=ON."
        )
    csc_mat = csr_mat.T.tocsc()   # genes × cells, CSC
    return _core.from_cupy_csr(csc_mat)


def _build_gene_mask(var_names, prefix: str) -> np.ndarray:
    """Build uint8 mask: 1 where var_name starts with *prefix* (case-insensitive)."""
    prefix_lower = prefix.lower()
    return np.array(
        [1 if str(g).lower().startswith(prefix_lower) else 0 for g in var_names],
        dtype=np.uint8,
    )


def _run_qc_on_device(adata: "anndata.AnnData", layer: Optional[str],
                      mt_key: str, ribo_key: str, stream):
    """
    Run calculate_qc_metrics on device; return (qc_result, device_csc).

    Both are needed by filter_cells / filter_genes to avoid running twice.
    """
    import singlet.gpu._core as _core
    import cupy as cp

    mat = _get_matrix(adata, layer)
    device_csc = _csr_to_device_csc(mat)

    var_names = adata.var.index.tolist()
    is_mt_np   = _build_gene_mask(var_names, mt_key)
    is_ribo_np = _build_gene_mask(var_names, ribo_key)
    d_is_mt    = cp.asarray(is_mt_np)
    d_is_ribo  = cp.asarray(is_ribo_np)

    qc_result = _core.calculate_qc_metrics(
        device_csc, d_is_mt, d_is_ribo, stream=stream)

    return qc_result, device_csc, is_mt_np, is_ribo_np


def _qc_result_to_obs_var(adata: "anndata.AnnData", qc_result,
                           mt_key: str, ribo_key: str,
                           is_mt_np: np.ndarray, is_ribo_np: np.ndarray) -> None:
    """Download QcResult arrays and write into adata.obs / adata.var."""
    import cupy as cp

    n_umis    = cp.asarray(qc_result.n_umis_view).get()
    n_genes_c = cp.asarray(qc_result.n_genes_view).get()
    pct_mt    = cp.asarray(qc_result.pct_mt_view).get()
    pct_ribo  = cp.asarray(qc_result.pct_ribo_view).get()

    adata.obs["total_counts"]             = n_umis.astype(np.float32)
    adata.obs["n_genes_by_counts"]        = n_genes_c.astype(int)
    adata.obs[f"pct_counts_{mt_key}"]     = pct_mt.astype(np.float32)
    adata.obs[f"pct_counts_{ribo_key}"]   = pct_ribo.astype(np.float32)

    gene_mean   = cp.asarray(qc_result.gene_mean_view).get()
    gene_var    = cp.asarray(qc_result.gene_var_view).get()
    gene_ncells = cp.asarray(qc_result.gene_n_cells_view).get()

    adata.var["mean_counts"]       = gene_mean.astype(np.float32)
    adata.var["var_counts"]        = gene_var.astype(np.float32)
    adata.var["n_cells_by_counts"] = gene_ncells.astype(int)
    adata.var[f"is_{mt_key}"]      = is_mt_np.astype(bool)
    adata.var[f"is_{ribo_key}"]    = is_ribo_np.astype(bool)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_qc_metrics(
    adata: "anndata.AnnData",
    qc_vars: Tuple[str, ...] = ("MT", "RIBO"),
    *,
    layer: Optional[str] = None,
    inplace: bool = True,
    copy: bool = False,
    deterministic: bool = False,
    stream=None,
) -> Optional["anndata.AnnData"]:
    """
    Compute per-cell and per-gene QC metrics on device (cycle-103).

    Mirrors ``scanpy.pp.calculate_qc_metrics``.  Results written into the
    same ``adata.obs`` / ``adata.var`` columns that scanpy produces.

    Parameters
    ----------
    adata : AnnData
        ``X`` (or ``layers[layer]``) must be a GPU-resident
        ``cupy.sparse.csr_matrix``.
    qc_vars : tuple of str, default ("MT", "RIBO")
        Gene-name prefixes identifying groups.  For each prefix ``P``:
        - ``adata.var['is_{P}']`` (bool) is written.
        - ``adata.obs['pct_counts_{P}']`` (float32) is written.
        Matching is case-insensitive (e.g. "MT-CO1" matches prefix "MT").
        Only the first two prefixes are passed to the C++ kernel (is_mt,
        is_ribo); additional prefixes are silently ignored until a future
        multi-mask kernel extension.
    layer : str or None
    inplace : bool, default True
    copy : bool, default False
    deterministic : bool, default False
        True = two-pass gene variance (bit-identical, ~1.5× slower).
    stream : int or None

    Returns
    -------
    None (inplace) or AnnData (copy).

    Columns written to ``adata.obs``::

        total_counts        — total UMI per cell (float32)
        n_genes_by_counts   — number of detected genes per cell (int)
        pct_counts_{P}      — %% counts from group P, for each P in qc_vars

    Columns written to ``adata.var``::

        mean_counts         — mean expression across cells (float32)
        var_counts          — variance of expression across cells (float32)
        n_cells_by_counts   — number of cells expressing this gene (int)
        is_{P}              — bool gene-group mask, for each P in qc_vars
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "calculate_qc_metrics"):
        raise AttributeError(
            "_core.calculate_qc_metrics is not available.  "
            "Compile with SINGLET_GPU_BUILD_PYTHON=ON (cycle-103 binding)."
        )

    try:
        import cupy as cp
    except ImportError as e:
        raise ImportError(
            f"singlet.gpu.qc.calculate_qc_metrics requires cupy.  {e}"
        )

    working = adata if (inplace and not copy) else copy_module.copy(adata)

    mt_key   = qc_vars[0] if len(qc_vars) > 0 else "MT"
    ribo_key = qc_vars[1] if len(qc_vars) > 1 else "RIBO"

    mat = _get_matrix(working, layer)
    device_csc = _csr_to_device_csc(mat)

    var_names  = working.var.index.tolist()
    is_mt_np   = _build_gene_mask(var_names, mt_key)
    is_ribo_np = _build_gene_mask(var_names, ribo_key)
    d_is_mt    = cp.asarray(is_mt_np)
    d_is_ribo  = cp.asarray(is_ribo_np)

    qc_result = _core.calculate_qc_metrics(
        device_csc, d_is_mt, d_is_ribo,
        stream=stream, deterministic=deterministic)

    _qc_result_to_obs_var(working, qc_result, mt_key, ribo_key,
                          is_mt_np, is_ribo_np)

    if inplace and not copy:
        return None
    return working


def filter_cells(
    adata: "anndata.AnnData",
    *,
    min_genes: Optional[int] = None,
    max_genes: Optional[int] = None,
    min_counts: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_pct_mt: Optional[float] = None,
    qc_vars: Tuple[str, ...] = ("MT", "RIBO"),
    layer: Optional[str] = None,
    inplace: bool = True,
    copy: bool = False,
    stream=None,
) -> Optional["anndata.AnnData"]:
    """
    Filter cells by QC thresholds (cycle-103).

    Mirrors ``scanpy.pp.filter_cells``.  Computes QC metrics if not already
    present, then removes cells failing any threshold.

    Parameters
    ----------
    adata : AnnData
    min_genes : int or None   — keep cells with >= min_genes detected genes
    max_genes : int or None   — keep cells with <= max_genes detected genes
    min_counts : int or None  — keep cells with >= min_counts total UMIs
    max_counts : int or None  — keep cells with <= max_counts total UMIs
    max_pct_mt : float or None — keep cells with <= max_pct_mt %% MT counts
    qc_vars : tuple of str    — passed to calculate_qc_metrics if run
    layer, inplace, copy, stream — as for calculate_qc_metrics

    Returns
    -------
    None (inplace) or AnnData (copy).
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "filter_cells"):
        raise AttributeError(
            "_core.filter_cells is not available.  "
            "Compile with SINGLET_GPU_BUILD_PYTHON=ON (cycle-103 binding)."
        )

    try:
        import cupy as cp
        try:
            import cupyx.scipy.sparse as csp  # cupy >= 14
        except ImportError:
            import cupy.sparse as csp         # cupy < 14 fallback
    except ImportError as e:
        raise ImportError(f"singlet.gpu.qc.filter_cells requires cupy.  {e}")

    mt_key   = qc_vars[0] if len(qc_vars) > 0 else "MT"
    ribo_key = qc_vars[1] if len(qc_vars) > 1 else "RIBO"

    # Run QC metrics on device (single pass regardless of whether obs cols exist).
    qc_result, device_csc, is_mt_np, is_ribo_np = _run_qc_on_device(
        adata, layer, mt_key, ribo_key, stream)

    # Build keep mask on host from per-cell arrays.
    n_umis_h  = cp.asarray(qc_result.n_umis_view).get().astype(np.float32)
    n_genes_h = cp.asarray(qc_result.n_genes_view).get().astype(int)
    mt_h      = cp.asarray(qc_result.pct_mt_view).get().astype(np.float32)

    keep = np.ones(adata.n_obs, dtype=bool)
    if min_genes  is not None: keep &= (n_genes_h >= int(min_genes))
    if max_genes  is not None: keep &= (n_genes_h <= int(max_genes))
    if min_counts is not None: keep &= (n_umis_h  >= float(min_counts))
    if max_counts is not None: keep &= (n_umis_h  <= float(max_counts))
    if max_pct_mt is not None: keep &= (mt_h      <= float(max_pct_mt))

    _INF = float("inf")
    filtered_csc = _core.filter_cells(
        device_csc, qc_result,
        min_genes  = float(min_genes)  if min_genes  is not None else 0.0,
        max_genes  = float(max_genes)  if max_genes  is not None else _INF,
        min_umis   = float(min_counts) if min_counts is not None else 0.0,
        max_umis   = float(max_counts) if max_counts is not None else _INF,
        max_pct_mt = float(max_pct_mt) if max_pct_mt is not None else 100.0,
        stream     = stream,
    )

    # Subset AnnData (obs/var/layers/uns) using the host keep mask.
    filtered = adata[keep, :].copy()

    # Replace X with the device-filtered sparse matrix.
    d = _core.to_cupy_csr(filtered_csc)
    # cupy >= 14 dtype-strictness: cp.asarray() rejects bare CAI dicts —
    # __cuda_array_interface__ must be an attribute. Wrap each dict.
    class _CaiView:
        def __init__(self, x): self.__cuda_array_interface__ = x
    filtered.X = csp.csc_matrix(
        (cp.asarray(_CaiView(d["data"])),
         cp.asarray(_CaiView(d["indices"])),
         cp.asarray(_CaiView(d["indptr"]))),
        shape=d["shape"],
    ).T.tocsr()

    # Write QC obs/var into the filtered object.
    _qc_result_to_obs_var(filtered, qc_result, mt_key, ribo_key,
                          is_mt_np, is_ribo_np)
    # Restrict QC obs to kept cells.
    for col in ["total_counts", "n_genes_by_counts",
                f"pct_counts_{mt_key}", f"pct_counts_{ribo_key}"]:
        if col in filtered.obs.columns:
            filtered.obs[col] = filtered.obs[col].values[keep]

    if copy or not inplace:
        return filtered

    # inplace=True: update the caller's adata object by reassigning its arrays.
    # This follows scanpy's own filter_cells inplace pattern.
    adata._X      = filtered.X
    adata._obs    = filtered.obs
    adata._var    = filtered.var
    adata._uns    = filtered.uns
    adata._obsm   = filtered.obsm
    adata._varm   = filtered.varm
    adata._obsp   = filtered.obsp
    adata._varp   = filtered.varp
    adata._layers = filtered.layers
    adata._n_obs  = filtered.n_obs
    adata._n_vars = filtered.n_vars
    return None


def filter_genes(
    adata: "anndata.AnnData",
    *,
    min_cells: Optional[int] = None,
    min_counts: Optional[int] = None,
    qc_vars: Tuple[str, ...] = ("MT", "RIBO"),
    layer: Optional[str] = None,
    inplace: bool = True,
    copy: bool = False,
    stream=None,
) -> Optional["anndata.AnnData"]:
    """
    Filter genes by cell-detection threshold (cycle-103).

    Mirrors ``scanpy.pp.filter_genes``.

    Parameters
    ----------
    adata : AnnData
    min_cells : int or None   — keep genes detected in >= min_cells cells
    min_counts : int or None  — keep genes whose total counts >= min_counts
                                (approximated as mean_counts * n_obs)
    qc_vars, layer, inplace, copy, stream — as for calculate_qc_metrics

    Returns
    -------
    None (inplace) or AnnData (copy).
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "filter_genes"):
        raise AttributeError(
            "_core.filter_genes is not available.  "
            "Compile with SINGLET_GPU_BUILD_PYTHON=ON (cycle-103 binding)."
        )

    try:
        import cupy as cp
        try:
            import cupyx.scipy.sparse as csp  # cupy >= 14
        except ImportError:
            import cupy.sparse as csp         # cupy < 14 fallback
    except ImportError as e:
        raise ImportError(f"singlet.gpu.qc.filter_genes requires cupy.  {e}")

    mt_key   = qc_vars[0] if len(qc_vars) > 0 else "MT"
    ribo_key = qc_vars[1] if len(qc_vars) > 1 else "RIBO"

    qc_result, device_csc, is_mt_np, is_ribo_np = _run_qc_on_device(
        adata, layer, mt_key, ribo_key, stream)

    gene_ncells = cp.asarray(qc_result.gene_n_cells_view).get().astype(int)
    gene_mean   = cp.asarray(qc_result.gene_mean_view).get().astype(np.float32)

    keep = np.ones(adata.n_vars, dtype=bool)
    if min_cells  is not None: keep &= (gene_ncells >= int(min_cells))
    if min_counts is not None:
        total_approx = gene_mean * float(adata.n_obs)
        keep &= (total_approx >= float(min_counts))

    threshold = int(min_cells) if min_cells is not None else 1
    filtered_csc = _core.filter_genes(
        device_csc, qc_result, min_cells=threshold, stream=stream)

    filtered = adata[:, keep].copy()

    d = _core.to_cupy_csr(filtered_csc)
    # cupy >= 14 dtype-strictness: cp.asarray() rejects bare CAI dicts —
    # __cuda_array_interface__ must be an attribute. Wrap each dict.
    class _CaiView:
        def __init__(self, x): self.__cuda_array_interface__ = x
    filtered.X = csp.csc_matrix(
        (cp.asarray(_CaiView(d["data"])),
         cp.asarray(_CaiView(d["indices"])),
         cp.asarray(_CaiView(d["indptr"]))),
        shape=d["shape"],
    ).T.tocsr()

    # Write QC var only for kept genes.
    _qc_result_to_obs_var(filtered, qc_result, mt_key, ribo_key,
                          is_mt_np, is_ribo_np)
    for col in ["mean_counts", "var_counts", "n_cells_by_counts",
                f"is_{mt_key}", f"is_{ribo_key}"]:
        if col in filtered.var.columns:
            filtered.var[col] = filtered.var[col].values[keep]

    if copy or not inplace:
        return filtered

    adata._X      = filtered.X
    adata._obs    = filtered.obs
    adata._var    = filtered.var
    adata._uns    = filtered.uns
    adata._obsm   = filtered.obsm
    adata._varm   = filtered.varm
    adata._obsp   = filtered.obsp
    adata._varp   = filtered.varp
    adata._layers = filtered.layers
    adata._n_obs  = filtered.n_obs
    adata._n_vars = filtered.n_vars
    return None


__all__ = ["calculate_qc_metrics", "filter_cells", "filter_genes"]
