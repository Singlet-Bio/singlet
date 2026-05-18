# SPDX-License-Identifier: MIT
"""
singlet.gpu.enrichment.score_genes — GPU-native per-cell gene-set scoring.

Underlying C++: cycle 129, ``enrich/score_genes.h``.
Algorithm: Satija et al. 2015 / Seurat AddModuleScore — difference of means
           between target gene set and a size-matched control pool drawn from
           the same expression-mean bins.

Scanpy compatibility
--------------------
``run_score_genes`` mirrors ``scanpy.tl.score_genes``.
Results written to ``adata.obs[f'{score_name}_{set_name}']`` (one column per
gene set), which is identical to the scanpy convention.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import anndata


def run_score_genes(
    adata: "anndata.AnnData",
    gene_lists: Union[dict[str, list[str]], list[list[str]]],
    *,
    gene_names_key: Optional[str] = None,
    score_name: str = "score_genes",
    ctrl_size: int = 50,
    n_bins: int = 25,
    seed: int = 0,
    stream=None,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    Per-cell gene-set scoring (cycle 129, Satija et al. 2015 / Seurat AddModuleScore).

    Mirrors ``scanpy.tl.score_genes`` API.  Results written to
    ``adata.obs[f'{score_name}_{set_name}']`` — one column per gene set.

    Parameters
    ----------
    adata : AnnData
        ``adata.X`` is cells × genes (log-normalized recommended).
    gene_lists : dict[str, list[str]] or list[list[str]]
        Gene sets to score.  If a dict, keys become column name suffixes
        (``f'{score_name}_{key}'``).  If a list, sets are named by index
        (``f'{score_name}_{i}'``).
    gene_names_key : str or None
        Column in ``adata.var`` to use as gene names.  None = ``var_names``.
    score_name : str
        Prefix for columns written to ``adata.obs``.  Default ``"score_genes"``.
    ctrl_size : int
        Number of control genes per set (scanpy default 50).
    n_bins : int
        Expression-mean bins for control sampling (scanpy default 25).
    seed : int
        RNG seed for deterministic control sampling.  Passed through to
        ``std::mt19937`` host-side in the kernel.
    stream : int or None
        CUDA stream handle.  None = default stream.
    copy : bool
        Return a copy of ``adata`` with scores written to ``obs``.
        False (default) = modify in place, return None.

    Returns
    -------
    None or AnnData
        None when ``copy=False`` (inplace), AnnData when ``copy=True``.

    Notes
    -----
    Gene names not found in ``adata.var_names`` (or the ``gene_names_key``
    column) are silently dropped with a ``UserWarning``.  If a gene set has
    zero valid genes after filtering, its score column is set to ``NaN``
    with a ``UserWarning``.
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "score_genes"):
        raise AttributeError(
            "_core.score_genes is not available.  "
            "Ensure the cycle-186 binding extension has been compiled."
        )

    working = adata.copy() if copy else adata

    # ── Build gene name → index lookup ───────────────────────────────────────
    if gene_names_key is not None:
        gene_names = list(working.var[gene_names_key])
    else:
        gene_names = list(working.var_names)

    index_of: dict[str, int] = {g: i for i, g in enumerate(gene_names)}

    # ── Normalise gene_lists to (set_names, set_genes) ──────────────────────
    if isinstance(gene_lists, dict):
        set_names = list(gene_lists.keys())
        set_genes_raw = list(gene_lists.values())
    else:
        set_names = [str(i) for i in range(len(gene_lists))]
        set_genes_raw = list(gene_lists)

    # ── Build index lists; warn on missing genes; mark empty sets ────────────
    gene_sets_idx: list[list[int]] = []
    empty_sets: list[int] = []  # positions that have 0 valid genes

    for pos, (name, raw_genes) in enumerate(zip(set_names, set_genes_raw)):
        indices: list[int] = []
        missing: list[str] = []
        for g in raw_genes:
            if g in index_of:
                indices.append(index_of[g])
            else:
                missing.append(g)
        if missing:
            warnings.warn(
                f"run_score_genes: {len(missing)} gene(s) in set '{name}' not found "
                f"in var_names and will be skipped: {missing[:5]}"
                + (" ..." if len(missing) > 5 else ""),
                UserWarning,
                stacklevel=2,
            )
        if not indices:
            warnings.warn(
                f"run_score_genes: gene set '{name}' has 0 valid genes after "
                "var_names filtering.  Its score column will be NaN.",
                UserWarning,
                stacklevel=2,
            )
            empty_sets.append(pos)
            # Placeholder — kernel rejects empty inner lists.  We'll NaN-fill
            # the column after the GPU call without invoking the kernel for it.
            # Insert a dummy single-gene list to keep set count consistent, then
            # overwrite the column with NaN after the call.
            # Use gene index 0 as a harmless stand-in.
            indices = [0]
        gene_sets_idx.append(indices)

    # ── Upload to GPU and run kernel ─────────────────────────────────────────
    if not hasattr(_core, "from_cupy_csr"):
        raise AttributeError(
            "_core.from_cupy_csr is not available.  "
            "Ensure the singlet_gpu binding extension has been compiled."
        )

    try:
        import cupy as cp
        try:
            import cupyx.scipy.sparse as csp  # cupy >= 14
        except ImportError:
            import cupy.sparse as csp         # cupy < 14 fallback
        import scipy.sparse as sp

        X = working.X
        # adata.X is (n_obs=cells) × (n_vars=genes); kernel expects genes × cells.
        if X.shape[0] == working.n_obs:
            X = X.T  # genes × cells
        if csp.issparse(X):
            # cupy sparse — already on device.  Convert layout to CSC if needed.
            cupy_csc = X if isinstance(X, csp.csc_matrix) else X.tocsc()
            if cupy_csc.dtype != cp.float32:
                cupy_csc = cupy_csc.astype(cp.float32)
        elif sp.issparse(X):
            # scipy sparse on host — upload as cupy CSC.
            cupy_csc = csp.csc_matrix(X.tocsc().astype(np.float32))
        else:
            # Dense.  cupy ndarray exposes __cuda_array_interface__; numpy
            # goes through cp.asarray (uploads to device).
            if hasattr(X, "__cuda_array_interface__"):
                cupy_csc = csp.csc_matrix(X.astype(cp.float32))
            else:
                cupy_csc = csp.csc_matrix(cp.asarray(X, dtype=cp.float32))

        # Convert cupy csc_matrix to PyDeviceCsc (zero-copy device pointer wrap).
        device_mat = _core.from_cupy_csr(cupy_csc)
    except ImportError as e:
        raise ImportError(
            "singlet.gpu.enrichment.run_score_genes requires cupy.  "
            f"Original error: {e}"
        ) from e

    result = _core.score_genes(
        device_mat,
        gene_sets_idx,
        ctrl_size=ctrl_size,
        n_bins=n_bins,
        seed=seed,
        use_set_size_for_ctrl=True,
        stream=stream,
    )

    # ── Retrieve scores and write to adata.obs ───────────────────────────────
    n_cells = result.n_cells
    n_sets  = result.n_sets

    # scores_view is [n_cells × n_sets] col-major float32 on device.
    # cupy >= 14 dtype-strictness: wrap CAI dict so cp.asarray() gets an
    # object with __cuda_array_interface__ as an attribute, not a bare dict.
    class _CaiView:
        def __init__(self, d): self.__cuda_array_interface__ = d
    scores_gpu = cp.asarray(_CaiView(result.scores_view)).reshape(n_sets, n_cells).T
    # scores_gpu is now (n_cells, n_sets) in C order.
    scores_cpu = scores_gpu.get()  # D2H once; shape (n_cells, n_sets)

    for i, name in enumerate(set_names):
        col_name = f"{score_name}_{name}"
        col_data = scores_cpu[:, i].astype(np.float64)
        if i in empty_sets:
            col_data[:] = np.nan
        working.obs[col_name] = col_data

    working.uns[f"{score_name}_params"] = {
        "ctrl_size": ctrl_size,
        "n_bins": n_bins,
        "seed": seed,
        "set_names": set_names,
        "n_sets_empty": len(empty_sets),
    }

    return working if copy else None


__all__ = ["run_score_genes"]
