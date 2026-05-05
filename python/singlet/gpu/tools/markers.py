# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.tools.markers — GPU-native marker scoring and cell type annotation.

Wraps:
    ``singlet::gpu::anno::marker_score``   (cycle 12, ``anno/marker_score.h``)
    ``singlet::gpu::anno::celltypist_project`` (cycle 12, ``anno/reference_map.h``)

Exposes two public functions:

    score_genes(adata, gene_list, ...)      — marker gene scoring
    celltypist_predict(adata, model_path, ...) — CellTypist-style annotation

``score_genes`` mirrors ``scanpy.tl.score_genes`` for API parity.
``celltypist_predict`` has no direct scanpy counterpart; it follows
the CellTypist Python API (``celltypist.annotate(adata, model=...)``)
for parameter naming.

CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND: ``_core.marker_score`` and
``_core.celltypist_project`` are not yet exposed by the cycle-20 binding
extension.  Both functions raise ``AttributeError`` with this tag until
the bindings are added.
"""

from __future__ import annotations

import copy as copy_module
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# RNG helper
# ---------------------------------------------------------------------------


def _resolve_seed(rng: Optional[Union[int, np.random.Generator]]) -> int:
    """Convert scanpy-style ``rng`` to a uint64 seed for C++."""
    if rng is None:
        return 0
    if isinstance(rng, int):
        return rng & 0xFFFF_FFFF_FFFF_FFFF
    return int(np.random.default_rng(rng).integers(0, 2**63))


# ---------------------------------------------------------------------------
# Internal: gene index lookup
# ---------------------------------------------------------------------------


def _gene_indices(var_names: Sequence[str], gene_list: Sequence[str]) -> np.ndarray:
    """
    Return the integer indices of *gene_list* in *var_names*.

    Genes not found in *var_names* are silently skipped with a warning.

    Returns
    -------
    np.ndarray of int32, length ≤ len(gene_list)
    """
    import warnings

    name_to_idx = {n: i for i, n in enumerate(var_names)}
    indices = []
    missing = []
    for g in gene_list:
        idx = name_to_idx.get(g)
        if idx is None:
            missing.append(g)
        else:
            indices.append(idx)
    if missing:
        warnings.warn(
            f"score_genes: {len(missing)} gene(s) in gene_list not found in "
            f"adata.var_names and will be skipped: {missing[:5]}"
            + ("..." if len(missing) > 5 else ""),
            UserWarning,
            stacklevel=3,
        )
    return np.array(indices, dtype=np.int32)


# ---------------------------------------------------------------------------
# score_genes
# ---------------------------------------------------------------------------


def score_genes(
    adata: anndata.AnnData,
    gene_list: Sequence[str],
    *,
    score_name: str = "score",
    method: Literal["mlm", "ulm", "wsum", "ucell"] = "mlm",
    layer: Optional[str] = None,
    use_raw: bool = False,
    ctrl_size: int = 50,
    n_bins: int = 25,
    rng: Optional[Union[int, np.random.Generator]] = None,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """
    Compute a per-cell marker-gene activity score (GPU-native, cycle-12 kernel).

    Mirrors ``scanpy.tl.score_genes`` for the core parameters:

    - Same *gene_list*, *score_name*, *ctrl_size*, *n_bins*, *copy*
      parameter names and semantics.
    - Extends scanpy with the *method* parameter, offering four GPU
      scoring algorithms (mlm, ulm, wsum, UCell).

    Parameters
    ----------
    adata : AnnData
        Input data.  Expression values are read from ``adata.X``,
        ``adata.raw.X`` (when ``use_raw=True``), or
        ``adata.layers[layer]``.
    gene_list : list of str
        Genes in the marker set to score.  Genes not present in
        ``adata.var_names`` are skipped with a warning.
    score_name : str, default 'score'
        Key under which per-cell scores are written in ``adata.obs``.
        Matches scanpy's ``score_name`` parameter.
    method : {'mlm', 'ulm', 'wsum', 'ucell'}, default 'mlm'
        Scoring algorithm:
        - ``'mlm'`` — Multivariate Linear Model (DecoupleR-style).
          Regresses a unit weight-per-gene design matrix against the
          cell expression vector.
        - ``'ulm'`` — Univariate Linear Model: mean z-score over the
          gene set (like enrichment score).
        - ``'wsum'`` — Weighted sum: Σ w_i · x_i (w_i = 1 / len by
          default unless a ``gene_weights`` dict is provided).
        - ``'ucell'`` — UCell rank-based scoring (median normalised
          rank of the gene set within each cell).
    layer : str or None, default None
        Layer to use as expression values.  ``None`` → ``adata.X``.
    use_raw : bool, default False
        Use ``adata.raw.X`` as expression values.
    ctrl_size : int, default 50
        Number of control genes per bin for background correction
        (scanpy ``ctrl_size`` parameter).  Only used by ``'mlm'`` and
        ``'ulm'``; ignored by ``'wsum'`` and ``'ucell'``.
    n_bins : int, default 25
        Number of expression bins for background gene sampling (scanpy
        ``n_bins`` parameter).  Only used by ``'mlm'`` and ``'ulm'``.
    rng : int or np.random.Generator or None, default None
        Random seed for background gene sampling.  ``None`` → seed 0.
    copy : bool, default False
        Return a copy of *adata* with results written to it.  When
        ``False`` (default), *adata* is modified in-place and ``None``
        is returned.

    Returns
    -------
    None
        When ``copy=False`` (default, in-place).
    AnnData
        When ``copy=True`` — the modified copy.

    Raises
    ------
    AttributeError
        If ``_core.marker_score`` is not available.
        See CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND.
    ValueError
        If *gene_list* is empty after filtering against ``adata.var_names``.

    Notes
    -----
    **Result location**:
    ``adata.obs[score_name]`` — float32 per-cell activity score.

    **No host copies**: the expression matrix is already device-resident;
    the scoring kernel runs entirely on device and returns a device array
    that is stored directly in ``obs``.

    Examples
    --------
    Score a T-cell marker set::

        import singlet.gpu.tools as sgt
        sgt.score_genes(adata, ['CD3D', 'CD3E', 'CD3G'], score_name='T_score')
        print(adata.obs['T_score'].describe())

    UCell scoring::

        sgt.score_genes(adata, tcell_markers, method='ucell', score_name='ucell_T')
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "marker_score"):
        raise AttributeError(
            "_core.marker_score is not available.  "
            "See CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND — the cycle-20 "
            "pybind11 binding extension must expose marker_score before "
            "singlet.gpu.tools.score_genes() is callable."
        )

    working = copy_module.copy(adata) if copy else adata
    seed = _resolve_seed(rng)

    gene_indices = _gene_indices(list(working.var_names), list(gene_list))
    if len(gene_indices) == 0:
        raise ValueError("score_genes: no genes from gene_list found in adata.var_names.")

    # Resolve expression matrix.
    if use_raw and working.raw is not None:
        expr = working.raw.X
    elif layer is not None:
        if layer not in working.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        expr = working.layers[layer]
    else:
        expr = working.X

    scores = _core.marker_score(
        expr=expr,
        gene_indices=gene_indices,
        method=method,
        ctrl_size=int(ctrl_size),
        n_bins=int(n_bins),
        seed=seed,
    )

    # Convert device array → pandas Series for adata.obs storage.
    try:
        import cupy as cp

        if isinstance(scores, cp.ndarray):
            scores = scores.get()
    except ImportError:
        pass
    working.obs[score_name] = np.asarray(scores, dtype=np.float32)

    return working if copy else None


# ---------------------------------------------------------------------------
# celltypist_predict
# ---------------------------------------------------------------------------


def celltypist_predict(
    adata: anndata.AnnData,
    model_path: Union[str, Path],
    *,
    key_added: str = "celltypist",
    majority_voting: bool = False,
    over_clustering: Optional[str] = None,
    min_prop: float = 0.0,
    layer: Optional[str] = None,
    use_raw: bool = False,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """
    Reference-based cell type prediction via GPU CellTypist projection.

    Wraps ``singlet::gpu::anno::celltypist_project`` (cycle 12,
    ``anno/reference_map.h``).

    Follows the CellTypist Python API (``celltypist.annotate(adata, model=...)``):

        import singlet.gpu.tools as sgt
        sgt.celltypist_predict(adata, 'Immune_All_High.pkl')

    Parameters
    ----------
    adata : AnnData
        Input data.  Expression values are read from ``adata.X``
        (default), ``adata.raw.X`` (when ``use_raw=True``), or
        ``adata.layers[layer]``.
        **Important**: normalise to 10,000 counts per cell and log1p
        before calling, following CellTypist requirements.  The cycle-12
        kernel does not apply normalisation internally.
    model_path : str or Path
        Path to a CellTypist model file (``*.pkl``) or the
        singlet-gpu native model format (``*.sgm`` — a binary
        representation of the reference centroids + weights).
    key_added : str, default 'celltypist'
        Key under which predicted labels are written in ``adata.obs``.
        Probability scores per cell type are written to
        ``adata.obsm[f'{key_added}_proba']``.
    majority_voting : bool, default False
        Refine predictions by majority voting within a cell cluster.
        Requires *over_clustering*.
    over_clustering : str or None, default None
        Key in ``adata.obs`` containing cluster labels to use for
        majority voting.  Required when ``majority_voting=True``.
    min_prop : float, default 0.0
        Minimum proportion for a cell type to be assigned in majority
        voting.  Cells in clusters where no cell type exceeds *min_prop*
        are labelled ``'Unassigned'``.
    layer : str or None, default None
        Layer to use as expression values.  ``None`` → ``adata.X``.
    use_raw : bool, default False
        Use ``adata.raw.X`` as expression values.
    copy : bool, default False
        Return a copy of *adata* with results written to it.  When
        ``False`` (default), *adata* is modified in-place and ``None``
        is returned.

    Returns
    -------
    None
        When ``copy=False`` (default, in-place).
    AnnData
        When ``copy=True`` — the modified copy.

    Raises
    ------
    AttributeError
        If ``_core.celltypist_project`` is not available.
        See CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND.
    FileNotFoundError
        If *model_path* does not exist.
    ValueError
        If ``majority_voting=True`` but *over_clustering* is ``None``,
        or if *over_clustering* key is not in ``adata.obs``.

    Notes
    -----
    **Result locations**:

    +----------------------------------+--------------------------------------------+
    | ``adata.obs[key_added]``         | Predicted cell type label (str category)   |
    +----------------------------------+--------------------------------------------+
    | ``adata.obsm[key_added+'_proba']``| Float32 probability matrix (cells × types)|
    +----------------------------------+--------------------------------------------+
    | ``adata.uns[key_added]``         | Model metadata dict (source, n_types, …)   |
    +----------------------------------+--------------------------------------------+

    **No host copies**: the expression matrix is already device-resident;
    the reference projection (a batched dot-product + softmax) runs
    entirely on device.  The probability matrix is stored as a cupy array
    in ``obsm``; the argmax label assignment is a tiny reduction.

    Examples
    --------
    Predict cell types with a CellTypist Immune model::

        import singlet.gpu.tools as sgt
        sgt.celltypist_predict(adata, 'models/Immune_All_High.pkl')
        print(adata.obs['celltypist'].value_counts())

    With majority voting on Leiden clusters::

        sgt.celltypist_predict(
            adata,
            'models/Immune_All_High.pkl',
            majority_voting=True,
            over_clustering='leiden',
        )
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "celltypist_project"):
        raise AttributeError(
            "_core.celltypist_project is not available.  "
            "See CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND — the cycle-20 "
            "pybind11 binding extension must expose celltypist_project before "
            "singlet.gpu.tools.celltypist_predict() is callable."
        )

    import pandas as pd

    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"CellTypist model not found: {model_path}")

    if majority_voting and over_clustering is None:
        raise ValueError(
            "majority_voting=True requires over_clustering to be set to a "
            "key in adata.obs containing cluster labels."
        )
    if majority_voting and over_clustering not in adata.obs.columns:
        raise ValueError(f"over_clustering='{over_clustering}' not found in adata.obs.")

    working = copy_module.copy(adata) if copy else adata

    # Resolve expression matrix.
    if use_raw and working.raw is not None:
        expr = working.raw.X
    elif layer is not None:
        if layer not in working.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        expr = working.layers[layer]
    else:
        expr = working.X

    # Match model gene universe to adata.var_names.
    gene_names = list(working.var_names)

    raw_result = _core.celltypist_project(
        expr=expr,
        gene_names=gene_names,
        model_path=str(model_path),
    )
    # raw_result has:
    #   .labels          — list of str, length n_cells
    #   .proba           — device array, shape (n_cells, n_types)
    #   .cell_type_names — list of str, length n_types
    #   .model_meta      — dict

    labels = list(raw_result.labels)

    # Majority voting post-processing (on host — labels are small).
    if majority_voting:
        cluster_col = working.obs[over_clustering].values
        unique_clusters = list(dict.fromkeys(cluster_col))  # preserve order
        label_arr = np.array(labels)
        for cl in unique_clusters:
            mask = cluster_col == cl
            cl_labels = label_arr[mask]
            if len(cl_labels) == 0:
                continue
            unique, counts = np.unique(cl_labels, return_counts=True)
            top_label = unique[np.argmax(counts)]
            top_prop = counts[np.argmax(counts)] / len(cl_labels)
            if top_prop < min_prop:
                label_arr[mask] = "Unassigned"
            else:
                label_arr[mask] = top_label
        labels = list(label_arr)

    # Write results.
    working.obs[key_added] = pd.Categorical(labels)
    working.obsm[f"{key_added}_proba"] = raw_result.proba
    working.uns[key_added] = dict(raw_result.model_meta)
    working.uns[key_added]["cell_type_names"] = list(raw_result.cell_type_names)

    return working if copy else None


__all__ = ["score_genes", "celltypist_predict"]
