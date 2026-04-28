# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.network.hdwgcna — GPU-native hdWGCNA co-expression network analysis.

Underlying C++ cycle: cycle 46 (network/hdwgcna.h —
``singlet_gpu::network::run_hdwgcna`` kernel).

Reference: Morabito et al., "hdWGCNA identifies co-expression networks in
high-dimensional transcriptomics data," Cell Reports Methods 2023.
https://doi.org/10.1016/j.crmeth.2023.100498

Result location
---------------
When an AnnData is provided (``run_from_anndata``):

- ``adata.var['hdwgcna_module']``      — int32 per-gene module assignment.
- ``adata.var['hdwgcna_kme']``         — float32 per-gene kME.
- ``adata.obsm['X_hdwgcna_eigen']``    — float32 (n_cells × n_modules) eigengenes.
- ``adata.uns['hdwgcna_hub_genes']``   — dict mapping module_id → gene index list.
- ``adata.uns['hdwgcna_params']``      — run parameters.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_core():
    import singlet_gpu._core as _core
    if not hasattr(_core, "network") or not hasattr(_core.network, "hdwgcna"):
        raise AttributeError(
            "_core.network.hdwgcna is not available.  "
            "The C++ extension must be compiled on a CUDA-capable node."
        )
    return _core.network


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_from_csc(
    mat,
    *,
    soft_power: float = 6.0,
    min_module_size: int = 30,
    height_cut_fraction: float = 0.25,
    n_hub_genes: int = 10,
    deterministic: bool = False,
    ooc_chunk_size: int = 0,
    stream=None,
    seed: int = 42,
):
    """
    GPU-native hdWGCNA co-expression network analysis (raw CSC input).

    Parameters
    ----------
    mat : DeviceCsc
        Expression matrix (genes × cells) on device.  Should be normalised
        and log-transformed.
    soft_power : float, default 6.0
        WGCNA soft-thresholding power beta (6 for unsigned, 12 for signed).
    min_module_size : int, default 30
        Minimum number of genes per module.
    height_cut_fraction : float, default 0.25
        Dendrogram height cut threshold as fraction of max linkage height.
    n_hub_genes : int, default 10
        Top-N hub genes returned per module.
    deterministic : bool, default False
        Pass CUBLAS_ATOMICS_NOT_ALLOWED for deterministic SYRK/GEMM.
    ooc_chunk_size : int, default 0
        Chunk size for OOC pairwise correlation; 0 = auto.
    stream : int or None
        CUDA stream pointer.
    seed : int, default 42
        Seed for randomised SVD used in eigengene computation.

    Returns
    -------
    HdwgcnaResult
        ``.module_assignment`` (n_genes numpy int32)
        ``.eigengenes`` (n_modules × n_cells numpy float32)
        ``.hub_genes`` (n_modules × n_hub_genes numpy int32)
        ``.kme`` (n_genes numpy float32)
    """
    net = _require_core()
    return net.hdwgcna(
        mat,
        soft_power=float(soft_power),
        min_module_size=int(min_module_size),
        height_cut_fraction=float(height_cut_fraction),
        n_hub_genes=int(n_hub_genes),
        deterministic=bool(deterministic),
        ooc_chunk_size=int(ooc_chunk_size),
        stream=stream,
        seed=int(seed),
    )


def run_from_anndata(
    adata: "anndata.AnnData",
    *,
    layer: Optional[str] = None,
    soft_power: float = 6.0,
    min_module_size: int = 30,
    height_cut_fraction: float = 0.25,
    n_hub_genes: int = 10,
    deterministic: bool = False,
    ooc_chunk_size: int = 0,
    stream=None,
    seed: int = 42,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    GPU-native hdWGCNA co-expression network from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        ``adata.X`` (or ``adata.layers[layer]``) must be a DeviceCsc.
    layer : str or None
        Layer key to use instead of ``adata.X``.
    soft_power : float, default 6.0
        WGCNA soft-thresholding power.
    min_module_size : int, default 30
    height_cut_fraction : float, default 0.25
    n_hub_genes : int, default 10
    deterministic : bool, default False
    ooc_chunk_size : int, default 0
    stream : int or None
    seed : int, default 42
    copy : bool, default False

    Returns
    -------
    None (in-place) or AnnData (copy=True).

    Notes
    -----
    Writes:

    - ``adata.var['hdwgcna_module']``   — int32 module label (0 = grey).
    - ``adata.var['hdwgcna_kme']``      — float32 kME per gene.
    - ``adata.obsm['X_hdwgcna_eigen']`` — float32 (n_cells × n_modules).
    - ``adata.uns['hdwgcna_hub_genes']``— {module_id: [gene_idx, ...]}.
    - ``adata.uns['hdwgcna_params']``   — run parameters.
    """
    import singlet_gpu

    working = copy_module.copy(adata) if copy else adata

    mat = working.layers[layer] if layer is not None else working.X
    if not isinstance(mat, singlet_gpu.DeviceCsc):
        raise TypeError(
            "Expression matrix must be a DeviceCsc.  "
            "Call singlet_gpu.load_pz() first."
        )

    result = run_from_csc(
        mat,
        soft_power=soft_power,
        min_module_size=min_module_size,
        height_cut_fraction=height_cut_fraction,
        n_hub_genes=n_hub_genes,
        deterministic=deterministic,
        ooc_chunk_size=ooc_chunk_size,
        stream=stream,
        seed=seed,
    )

    module_assignment = np.asarray(result.module_assignment, dtype=np.int32)
    kme               = np.asarray(result.kme,               dtype=np.float32)
    eigengenes        = np.asarray(result.eigengenes,         dtype=np.float32)
    hub_genes_mat     = np.asarray(result.hub_genes,          dtype=np.int32)

    # eigengenes: (n_modules × n_cells) → (n_cells × n_modules)
    eigengenes_T = eigengenes.reshape(result.n_modules, result.n_cells).T

    working.var["hdwgcna_module"] = module_assignment
    working.var["hdwgcna_kme"]    = kme
    working.obsm["X_hdwgcna_eigen"] = eigengenes_T

    # Build hub gene dict: {module_id (int): [gene_idx, ...]}
    hub_dict: Dict[int, List[int]] = {}
    hub_genes_2d = hub_genes_mat.reshape(result.n_modules, n_hub_genes)
    for mod_i in range(result.n_modules):
        module_label = int(mod_i + 1)  # module labels start at 1; 0 is grey
        hub_dict[module_label] = hub_genes_2d[mod_i, :].tolist()
    working.uns["hdwgcna_hub_genes"] = hub_dict

    working.uns["hdwgcna_params"] = {
        "soft_power": soft_power,
        "min_module_size": min_module_size,
        "height_cut_fraction": height_cut_fraction,
        "n_hub_genes": n_hub_genes,
        "n_modules": result.n_modules,
        "seed": seed,
    }

    return working if copy else None


__all__ = ["run_from_csc", "run_from_anndata"]
