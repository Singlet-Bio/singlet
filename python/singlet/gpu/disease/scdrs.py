# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.disease.scdrs — GPU-native scDRS cell-level disease relevance scoring.

Underlying C++ cycle: cycle 50 (disease/scdrs.h —
``singlet_gpu::disease::run_scdrs`` kernel).

Reference: Zhang et al., "Polygenic enrichment distinguishes disease
associations of individual cells in single-cell RNA-seq data," Nature Genetics 2022.
https://doi.org/10.1038/s41588-022-01167-z

Result location
---------------
When an AnnData is provided (``run_from_anndata``):

- ``adata.obsm['scdrs_raw_score']``        — float32 (n_cells × n_diseases).
- ``adata.obsm['scdrs_norm_score']``       — float32 (n_cells × n_diseases).
- ``adata.obsm['scdrs_p_value']``          — float32 (n_cells × n_diseases).
- ``adata.obsm['scdrs_fdr']``              — float32 (n_cells × n_diseases).
- ``adata.uns['scdrs_params']``            — run parameters.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_core():
    import singlet_gpu._core as _core
    if not hasattr(_core, "disease") or not hasattr(_core.disease, "scdrs"):
        raise AttributeError(
            "_core.disease.scdrs is not available.  "
            "The C++ extension must be compiled on a CUDA-capable node."
        )
    return _core.disease


def _view_to_numpy_2d(view, n_cells, n_diseases):
    """Copy a __cuda_array_interface__ 2D buffer to numpy."""
    n = n_cells * n_diseases
    try:
        import cupy as cp
        arr = cp.ndarray(
            shape=(n,), dtype=cp.float32,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(view["data"][0], n * 4, None), 0)
        ).get()
        return arr.reshape(n_cells, n_diseases)
    except (ImportError, Exception):
        import ctypes
        buf = (ctypes.c_float * n)()
        ctypes.memmove(ctypes.addressof(buf),
                       ctypes.c_void_p(view["data"][0]), n * 4)
        return np.frombuffer(buf, dtype=np.float32).copy().reshape(n_cells, n_diseases)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_from_csc(
    mat,
    gene_sets: Dict[str, Union[List[str], Dict[str, float]]],
    gene_names: List[str],
    *,
    n_controls: int = 1000,
    n_bins: int = 20,
    disease_chunk: int = 5,
    stream=None,
    seed: int = 0,
):
    """
    GPU-native scDRS disease relevance scoring (raw CSC input).

    Parameters
    ----------
    mat : DeviceCsc
        Expression matrix (genes × cells) on device.
    gene_sets : dict
        Mapping disease_name → list of gene names (unweighted) OR
        dict of {gene_name: weight} (MAGMA-style weighted).
    gene_names : list of str
        Gene name for each row of *mat* (length = mat.rows).
    n_controls : int, default 1000
        Number of matched control gene sets per disease.
    n_bins : int, default 20
        Bins per axis for (mean, variance) gene matching.
    disease_chunk : int, default 5
        Diseases processed per OOC pass (0 = all at once).
    stream : int or None
        CUDA stream pointer.
    seed : int, default 0
        Seed for host-side control set sampling.

    Returns
    -------
    ScDrsResult
        ``.raw_score_view``        — __cuda_array_interface__ (n_cells × n_diseases).
        ``.normalized_score_view`` — __cuda_array_interface__.
        ``.p_value_view``          — __cuda_array_interface__.
        ``.fdr_view``              — __cuda_array_interface__.
    """
    dis = _require_core()

    names       = list(gene_sets.keys())
    gene_lists  = []
    weight_lists = []

    for name in names:
        gs = gene_sets[name]
        if isinstance(gs, dict):
            gene_lists.append(list(gs.keys()))
            weight_lists.append(list(gs.values()))
        else:
            gene_lists.append(list(gs))
            weight_lists.append([])   # empty → all 1.0

    return dis.scdrs(
        mat,
        names,
        gene_lists,
        weight_lists,
        np.array([], dtype=np.int32),   # gene_name_index placeholder
        n_controls=int(n_controls),
        n_bins=int(n_bins),
        disease_chunk=int(disease_chunk),
        stream=stream,
        seed=int(seed),
    )


def run_from_anndata(
    adata: "anndata.AnnData",
    gene_sets: Dict[str, Union[List[str], Dict[str, float]]],
    *,
    layer: Optional[str] = None,
    gene_symbol_key: Optional[str] = None,
    n_controls: int = 1000,
    n_bins: int = 20,
    disease_chunk: int = 5,
    stream=None,
    seed: int = 0,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    GPU-native scDRS disease relevance scoring from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        ``adata.X`` (or ``adata.layers[layer]``) must be a DeviceCsc.
    gene_sets : dict
        Mapping disease_name → list of gene names (unweighted) OR
        dict of {gene_name: float weight}.
        Gene names must match ``adata.var_names`` or ``adata.var[gene_symbol_key]``.
    layer : str or None
        Layer key to use instead of ``adata.X``.
    gene_symbol_key : str or None
        Column in ``adata.var`` with gene symbols if ``adata.var_names``
        are not gene symbols.
    n_controls : int, default 1000
        Matched control gene sets per disease.
    n_bins : int, default 20
        (mean, variance) binning for control matching.
    disease_chunk : int, default 5
        OOC batch size (0 = all at once).
    stream : int or None
        CUDA stream pointer.
    seed : int, default 0
    copy : bool, default False

    Returns
    -------
    None (in-place) or AnnData (copy=True).

    Notes
    -----
    Writes:

    - ``adata.obsm['scdrs_raw_score']``   — float32 (n_cells × n_diseases).
    - ``adata.obsm['scdrs_norm_score']``  — float32.
    - ``adata.obsm['scdrs_p_value']``     — float32.
    - ``adata.obsm['scdrs_fdr']``         — float32.
    - ``adata.uns['scdrs_disease_names']`` — list of disease names.
    - ``adata.uns['scdrs_params']``        — run parameters.
    """
    import singlet_gpu

    working = copy_module.copy(adata) if copy else adata

    mat = working.layers[layer] if layer is not None else working.X
    if not isinstance(mat, singlet_gpu.DeviceCsc):
        raise TypeError(
            "Expression matrix must be a DeviceCsc.  "
            "Call singlet_gpu.load_pz() first."
        )

    # Gene names for row mapping.
    if gene_symbol_key is not None and gene_symbol_key in working.var.columns:
        gene_names = list(working.var[gene_symbol_key])
    else:
        gene_names = list(working.var_names)

    result = run_from_csc(
        mat, gene_sets, gene_names,
        n_controls=n_controls,
        n_bins=n_bins,
        disease_chunk=disease_chunk,
        stream=stream,
        seed=seed,
    )

    n_c = result.n_cells
    n_d = result.n_diseases
    disease_names = list(gene_sets.keys())

    working.obsm["scdrs_raw_score"]  = _view_to_numpy_2d(result.raw_score_view,        n_c, n_d)
    working.obsm["scdrs_norm_score"] = _view_to_numpy_2d(result.normalized_score_view, n_c, n_d)
    working.obsm["scdrs_p_value"]    = _view_to_numpy_2d(result.p_value_view,          n_c, n_d)
    working.obsm["scdrs_fdr"]        = _view_to_numpy_2d(result.fdr_view,              n_c, n_d)

    working.uns["scdrs_disease_names"] = disease_names
    working.uns["scdrs_params"] = {
        "n_diseases": n_d,
        "n_controls": n_controls,
        "n_bins": n_bins,
        "disease_chunk": disease_chunk,
        "seed": seed,
    }

    return working if copy else None


__all__ = ["run_from_csc", "run_from_anndata"]
