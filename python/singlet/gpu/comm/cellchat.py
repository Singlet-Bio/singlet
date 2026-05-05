# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.comm.cellchat — GPU-native CellChat cell communication scoring.

Underlying C++ cycle: cycle 37 (comm/cellchat.h —
``singlet::gpu::comm::run_cellchat`` kernel).

Reference: Jin et al., "Inference and analysis of cell-cell communication
using CellChat," Nature Communications 2021.
https://doi.org/10.1038/s41467-021-21246-9

Result location
---------------
When an AnnData is provided (``run_from_anndata``):

- ``adata.uns['cellchat']['comm_prob']``         — float32 (n_lr × n_types × n_types).
- ``adata.uns['cellchat']['p_values']``          — float32 (n_lr × n_types × n_types).
- ``adata.uns['cellchat']['pathway_activity']``  — float32 (n_pathways × n_types × n_types).
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    import anndata
    import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_core():
    import singlet.gpu._core as _core

    if not hasattr(_core, "comm") or not hasattr(_core.comm, "cellchat"):
        raise AttributeError(
            "_core.comm.cellchat is not available.  "
            "The C++ extension must be compiled on a CUDA-capable node."
        )
    return _core.comm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_from_csc(
    mat,
    cell_type: np.ndarray,
    lr_ligand: np.ndarray,
    lr_receptor: np.ndarray,
    pathway_id: Optional[np.ndarray] = None,
    *,
    n_pathways: int = 0,
    n_permutations: int = 1000,
    hill_K: float = 0.5,
    deterministic: bool = False,
    stream=None,
    seed: int = 0,
) -> Any:
    """
    GPU-native CellChat communication probability scoring (raw CSC input).

    Parameters
    ----------
    mat : DeviceCsc
        Expression matrix (genes × cells) on device.
    cell_type : np.ndarray int32, shape (n_cells,)
        Integer cell-type label per cell (0-indexed).
    lr_ligand : np.ndarray int32, shape (n_lr,)
        Gene index of the ligand for each LR pair.
    lr_receptor : np.ndarray int32, shape (n_lr,)
        Gene index of the receptor for each LR pair.
    pathway_id : np.ndarray int32, shape (n_lr,) or None
        Pathway index per LR pair.  None → all assigned to pathway 0.
    n_pathways : int, default 0
        Total pathway count (0 = infer from pathway_id.max()+1).
    n_permutations : int, default 1000
        Permutations for p-value estimation.
    hill_K : float, default 0.5
        Hill function half-saturation constant K.
    deterministic : bool, default False
        Use deterministic (scan-based) permutation counting.
    stream : int or None
        CUDA stream pointer.
    seed : int, default 0
        Philox seed for permutation sampling.

    Returns
    -------
    CellChatResult
        ``.comm_prob_view``        — __cuda_array_interface__ [n_lr × n_types²].
        ``.p_values_view``         — __cuda_array_interface__ [n_lr × n_types²].
        ``.pathway_activity_view`` — __cuda_array_interface__ [n_pathways × n_types²].
    """
    comm = _require_core()
    cell_type = np.asarray(cell_type, dtype=np.int32)
    lr_ligand = np.asarray(lr_ligand, dtype=np.int32)
    lr_receptor = np.asarray(lr_receptor, dtype=np.int32)
    if pathway_id is None:
        pathway_id = np.zeros(len(lr_ligand), dtype=np.int32)
    else:
        pathway_id = np.asarray(pathway_id, dtype=np.int32)
    if n_pathways == 0:
        n_pathways = int(pathway_id.max()) + 1

    return comm.cellchat(
        mat,
        cell_type,
        lr_receptor,
        lr_ligand,
        pathway_id,
        n_pathways=int(n_pathways),
        n_permutations=int(n_permutations),
        hill_K=float(hill_K),
        deterministic=bool(deterministic),
        stream=stream,
        seed=int(seed),
    )


def run_from_anndata(
    adata: anndata.AnnData,
    lr_database: pd.DataFrame,
    cell_type_key: str = "cell_type",
    *,
    gene_symbol_key: Optional[str] = None,
    n_permutations: int = 1000,
    hill_K: float = 0.5,
    deterministic: bool = False,
    stream=None,
    seed: int = 0,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """
    GPU-native CellChat communication scoring from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Expression matrix on device (``adata.X`` must be a DeviceCsc).
    lr_database : pd.DataFrame
        Must contain columns ``['ligand', 'receptor', 'pathway']`` with gene
        names matching ``adata.var_names`` (or ``adata.var[gene_symbol_key]``).
    cell_type_key : str, default ``"cell_type"``
        Column in ``adata.obs`` with cell-type labels.
    gene_symbol_key : str or None
        Column in ``adata.var`` with gene symbols, if ``adata.var_names``
        are not gene symbols (e.g. Ensembl IDs).
    n_permutations : int, default 1000
        Permutations for p-value estimation.
    copy : bool, default False

    Returns
    -------
    None (in-place) or AnnData (copy=True).

    Notes
    -----
    Writes ``adata.uns['cellchat']`` with keys:

    - ``comm_prob``        — float32 ndarray (n_lr, n_types, n_types).
    - ``p_values``         — float32 ndarray (n_lr, n_types, n_types).
    - ``pathway_activity`` — float32 ndarray (n_pathways, n_types, n_types).
    - ``lr_pairs``         — list of (ligand, receptor) pairs.
    - ``cell_types``       — ordered list of cell-type labels.
    - ``params``           — run parameters dict.
    """
    import singlet.gpu

    working = copy_module.copy(adata) if copy else adata

    if not isinstance(working.X, singlet.gpu.DeviceCsc):
        raise TypeError("adata.X must be a DeviceCsc.  Call singlet.gpu.load_pz() first.")
    if cell_type_key not in working.obs.columns:
        raise KeyError(f"cell_type_key='{cell_type_key}' not in adata.obs.")

    # Build gene name → row-index map.
    if gene_symbol_key is not None and gene_symbol_key in working.var.columns:
        gene_names = working.var[gene_symbol_key].to_list()
    else:
        gene_names = list(working.var_names)
    gene_idx_map: Dict[str, int] = {g: i for i, g in enumerate(gene_names)}

    # Build LR arrays from database.
    valid_rows = []
    for _, row in lr_database.iterrows():
        lig = str(row.get("ligand", ""))
        rec = str(row.get("receptor", ""))
        if lig in gene_idx_map and rec in gene_idx_map:
            valid_rows.append(row)
    if not valid_rows:
        raise ValueError(
            "No LR pairs from lr_database found in adata.var_names.  "
            "Check gene symbols match between the database and the AnnData."
        )

    import pandas as pd

    valid_df = pd.DataFrame(valid_rows)
    lr_ligand = np.array(
        [gene_idx_map[r["ligand"]] for _, r in valid_df.iterrows()], dtype=np.int32
    )
    lr_receptor = np.array(
        [gene_idx_map[r["receptor"]] for _, r in valid_df.iterrows()], dtype=np.int32
    )

    # Pathway assignment.
    if "pathway" in valid_df.columns:
        pathway_labels = valid_df["pathway"].astype("category")
        pathway_id = pathway_labels.cat.codes.to_numpy(dtype=np.int32)
        pathway_names = list(pathway_labels.cat.categories)
        n_pathways = len(pathway_names)
    else:
        pathway_id = np.zeros(len(lr_ligand), dtype=np.int32)
        pathway_names = ["unknown"]
        n_pathways = 1

    # Cell-type codes.
    ct_series = working.obs[cell_type_key].astype("category")
    cell_type = ct_series.cat.codes.to_numpy(dtype=np.int32)
    cell_types = list(ct_series.cat.categories)
    n_types = len(cell_types)

    result = run_from_csc(
        working.X,
        cell_type,
        lr_ligand,
        lr_receptor,
        pathway_id,
        n_pathways=n_pathways,
        n_permutations=n_permutations,
        hill_K=hill_K,
        deterministic=deterministic,
        stream=stream,
        seed=seed,
    )

    # Copy device results to host.
    n_lr = result.n_lr

    def _d2h(view, shape):
        """Copy a __cuda_array_interface__ dict to a numpy array."""
        try:
            import cupy as cp

            arr_gpu = cp.ndarray(
                shape=(int(np.prod(shape)),),
                dtype=cp.float32,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(view["data"][0], int(np.prod(shape)) * 4, None), 0
                ),
            )
            return arr_gpu.get().reshape(shape)
        except (ImportError, Exception):
            import ctypes

            n = int(np.prod(shape))
            buf = (ctypes.c_float * n)()
            ctypes.memmove(ctypes.addressof(buf), ctypes.c_void_p(view["data"][0]), n * 4)
            return np.frombuffer(buf, dtype=np.float32).copy().reshape(shape)

    comm_prob = _d2h(result.comm_prob_view, (n_lr, n_types, n_types))
    p_values = _d2h(result.p_values_view, (n_lr, n_types, n_types))
    pathway_activity = _d2h(result.pathway_activity_view, (n_pathways, n_types, n_types))

    working.uns["cellchat"] = {
        "comm_prob": comm_prob,
        "p_values": p_values,
        "pathway_activity": pathway_activity,
        "lr_pairs": list(zip(valid_df["ligand"].tolist(), valid_df["receptor"].tolist())),
        "pathways": pathway_names,
        "cell_types": cell_types,
        "params": {
            "n_permutations": n_permutations,
            "hill_K": hill_K,
            "deterministic": deterministic,
            "seed": seed,
        },
    }

    return working if copy else None


__all__ = ["run_from_csc", "run_from_anndata"]
