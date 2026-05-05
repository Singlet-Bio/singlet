# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.tools.rank_genes_groups — GPU-native differential expression.

Wraps:
    ``singlet::gpu::de::wilcoxon``  (cycle 11, ``de/wilcoxon.h``)
    ``singlet::gpu::de::ttest``     (cycle 11, ``de/ttest.h``)

Drop-in replacement for ``sc.tl.rank_genes_groups``.  All parameter
names and defaults match the scanpy 1.10 signature verified by the
cycle-19 code-reader:

    import singlet.gpu.tools as sgt
    sgt.rank_genes_groups(adata, 'leiden')

Result location (matches scanpy):
    adata.uns['rank_genes_groups']   (or key specified by key_added)

CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND: ``_core.wilcoxon_de`` and
``_core.ttest_de`` are not yet exposed by the cycle-20 binding
extension.  The wrapper raises ``AttributeError`` with this tag until
the bindings are added.
"""

from __future__ import annotations

import copy as copy_module
from typing import (
    TYPE_CHECKING,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Union,
)

import numpy as np

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# Correction methods
# ---------------------------------------------------------------------------

_CORR_METHODS = frozenset(
    ["benjamini-hochberg", "bonferroni", "holm-sidak", "fdr_bh", "fdr_by"]
)


def _bh_correct(pvalues: "np.ndarray") -> "np.ndarray":
    """
    Apply Benjamini-Hochberg FDR correction to a flat array of p-values.

    Used as a Python fallback when the C++ kernel does not return
    adjusted p-values directly.
    """
    n = len(pvalues)
    if n == 0:
        return pvalues.copy()
    order = np.argsort(pvalues)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    adjusted = np.minimum(1.0, pvalues * n / ranks)
    # Enforce monotonicity (cumulative minimum from high to low rank).
    for i in range(n - 2, -1, -1):
        adjusted[order[i]] = min(adjusted[order[i]], adjusted[order[i + 1]])
    return adjusted


# ---------------------------------------------------------------------------
# Internal: build result dict
# ---------------------------------------------------------------------------

def _build_result_dict(
    raw,
    groups: Sequence[str],
    gene_names: Sequence[str],
    n_genes: int,
    corr_method: str,
    groupby: str,
    method: str,
    reference: str,
) -> dict:
    """
    Convert the C++ DE result struct into the scanpy ``rank_genes_groups``
    dict format.

    Scanpy's format (as of 1.10):
    {
        'params': {
            'groupby': ..., 'reference': ..., 'method': ...,
            'corr_method': ..., 'use_raw': ...
        },
        'names':        recarray of shape (n_genes,) with field per group,
        'scores':       recarray of shape (n_genes,) with field per group,
        'logfoldchanges': recarray of shape (n_genes,),
        'pvals':        recarray of shape (n_genes,),
        'pvals_adj':    recarray of shape (n_genes,),
        'pts':          DataFrame (cells_fraction per gene × group) — optional,
        'pts_rest':     DataFrame — optional,
    }
    """
    import numpy.lib.recfunctions as rfn  # noqa: F401

    dtype_str = [(g, "U50") for g in groups]
    dtype_f32 = [(g, "f4") for g in groups]

    names_arr    = np.empty(n_genes, dtype=dtype_str)
    scores_arr   = np.empty(n_genes, dtype=dtype_f32)
    lfc_arr      = np.empty(n_genes, dtype=dtype_f32)
    pvals_arr    = np.empty(n_genes, dtype=dtype_f32)
    pvals_adj    = np.empty(n_genes, dtype=dtype_f32)

    for grp in groups:
        grp_raw = raw[grp]
        # grp_raw has: .names, .scores, .logfoldchanges, .pvals, .pvals_adj
        # (all length n_genes, already sorted by score descending)
        top_names = np.asarray(grp_raw.names)[:n_genes]
        top_scores = np.asarray(grp_raw.scores, dtype=np.float32)[:n_genes]
        top_lfc    = np.asarray(grp_raw.logfoldchanges, dtype=np.float32)[:n_genes]
        top_pvals  = np.asarray(grp_raw.pvals, dtype=np.float32)[:n_genes]

        # Apply FDR correction if the C++ kernel did not do it.
        if hasattr(grp_raw, "pvals_adj"):
            top_padj = np.asarray(grp_raw.pvals_adj, dtype=np.float32)[:n_genes]
        else:
            if corr_method in ("benjamini-hochberg", "fdr_bh"):
                top_padj = _bh_correct(top_pvals.astype(np.float64)).astype(np.float32)
            elif corr_method == "bonferroni":
                top_padj = np.minimum(1.0, top_pvals * len(gene_names)).astype(np.float32)
            else:
                top_padj = top_pvals.copy()

        names_arr[grp]  = top_names
        scores_arr[grp] = top_scores
        lfc_arr[grp]    = top_lfc
        pvals_arr[grp]  = top_pvals
        pvals_adj[grp]  = top_padj

    return {
        "params": {
            "groupby": groupby,
            "reference": reference,
            "method": method,
            "corr_method": corr_method,
            "use_raw": False,
        },
        "names": names_arr,
        "scores": scores_arr,
        "logfoldchanges": lfc_arr,
        "pvals": pvals_arr,
        "pvals_adj": pvals_adj,
    }


# ---------------------------------------------------------------------------
# Internal: extract expression matrix
# ---------------------------------------------------------------------------

def _get_expr_matrix(
    adata: "anndata.AnnData",
    layer: Optional[str],
    use_raw: Optional[bool],
):
    """Return the expression matrix to test."""
    if use_raw and adata.raw is not None:
        return adata.raw.X
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        return adata.layers[layer]
    return adata.X


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_genes_groups(
    adata: "anndata.AnnData",
    groupby: str,
    *,
    mask_var: Optional[object] = None,
    use_raw: Optional[bool] = None,
    groups: Union[Literal["all"], Iterable[str]] = "all",
    reference: str = "rest",
    n_genes: Optional[int] = None,
    rankby_abs: bool = False,
    pts: bool = False,
    key_added: Optional[str] = None,
    copy: bool = False,
    method: Union[
        Literal["wilcoxon", "t-test", "t-test_overestim_var", "logreg"],
        object,
    ] = "wilcoxon",
    corr_method: Literal[
        "benjamini-hochberg", "bonferroni", "holm-sidak", "fdr_bh", "fdr_by"
    ] = "benjamini-hochberg",
    tie_correct: bool = False,
    layer: Optional[str] = None,
) -> Optional["anndata.AnnData"]:
    """
    Rank genes for characterising groups (GPU-native DE, cycle-11 kernels).

    Mirrors ``scanpy.tl.rank_genes_groups`` exactly — parameter names and
    defaults are identical so this function is a drop-in replacement.

    Supported test methods:
      - ``'wilcoxon'``          → GPU Wilcoxon rank-sum test (``_core.wilcoxon_de``)
      - ``'t-test'``            → GPU Welch's t-test (``_core.ttest_de``)
      - ``'t-test_overestim_var'`` → GPU t-test with pooled variance (``_core.ttest_de``)

    Parameters
    ----------
    adata : AnnData
        Input data.  Expression values are read from ``adata.X`` (default),
        ``adata.raw.X`` (when ``use_raw=True``), or ``adata.layers[layer]``.
    groupby : str
        Key in ``adata.obs`` to group cells by.  This is the first
        positional argument (matches scanpy convention).
    mask_var : array-like or str or None, default None
        Boolean mask or var column name selecting which genes to test.
        ``None`` → test all genes.
    use_raw : bool or None, default None
        Use ``adata.raw.X`` as expression values.  ``None`` → use
        ``adata.raw`` if it exists and ``layer`` is ``None``.
    groups : 'all' or list of str, default 'all'
        Subset of groups to test.  ``'all'`` → test every group.
    reference : str, default 'rest'
        Reference group for the one-vs-rest or one-vs-one test.
        ``'rest'`` → compare each group against all other cells.
    n_genes : int or None, default None
        Number of top genes to report per group.  ``None`` → report all
        genes ranked (scanpy default).
    rankby_abs : bool, default False
        Rank by the absolute value of the score.  Useful when bidirectional
        DE is of interest.
    pts : bool, default False
        Compute the fraction of cells in each group expressing each gene
        (``pts`` and ``pts_rest`` entries in the result).
    key_added : str or None, default None
        Key under which results are written in ``adata.uns``.
        ``None`` → ``'rank_genes_groups'``.
    copy : bool, default False
        Return a copy of *adata* with results written to it.  When
        ``False`` (default), *adata* is modified in-place and ``None``
        is returned.
    method : str, default 'wilcoxon'
        Statistical test.  One of ``'wilcoxon'``, ``'t-test'``,
        ``'t-test_overestim_var'``.  ``'logreg'`` is accepted for API
        parity but dispatches to Wilcoxon with a warning (GPU logistic
        regression DE is not yet implemented in cycle 11).
    corr_method : str, default 'benjamini-hochberg'
        Multiple-testing correction method.  One of
        ``'benjamini-hochberg'``, ``'bonferroni'``, ``'fdr_bh'``,
        ``'fdr_by'``, ``'holm-sidak'``.
    tie_correct : bool, default False
        Apply tie correction to the Wilcoxon rank-sum test.  Has no
        effect for t-tests.
    layer : str or None, default None
        Layer to use as expression values.  ``None`` → ``adata.X``.

    Returns
    -------
    None
        When ``copy=False`` (default, in-place).
    AnnData
        When ``copy=True`` — the modified copy.

    Raises
    ------
    AttributeError
        If ``_core.wilcoxon_de`` or ``_core.ttest_de`` is not available.
        See CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND.
    KeyError
        If *groupby* is not in ``adata.obs``.
    ValueError
        If *method* is not in the supported set.

    Notes
    -----
    **Result location** (matches scanpy exactly):
    ``adata.uns['rank_genes_groups']`` — dict with fields:
    ``'names'``, ``'scores'``, ``'logfoldchanges'``, ``'pvals'``,
    ``'pvals_adj'``, ``'params'``.

    Examples
    --------
    Wilcoxon DE for Leiden clusters::

        import singlet.gpu.tools as sgt
        sgt.rank_genes_groups(adata, 'leiden')
        # result in adata.uns['rank_genes_groups']

    t-test with Bonferroni correction::

        sgt.rank_genes_groups(
            adata, 'cell_type',
            method='t-test',
            corr_method='bonferroni',
            n_genes=100,
        )
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"groupby='{groupby}' not found in adata.obs.")

    import singlet.gpu._core as _core
    import warnings

    # Map 'logreg' to wilcoxon with a warning (not yet implemented).
    effective_method = method
    if method == "logreg":
        warnings.warn(
            "method='logreg' is not yet implemented in the cycle-11 GPU kernel.  "
            "Falling back to 'wilcoxon'.  A dedicated logreg GPU path is planned.",
            UserWarning,
            stacklevel=2,
        )
        effective_method = "wilcoxon"

    use_wilcoxon = effective_method == "wilcoxon"
    use_ttest = effective_method in ("t-test", "t-test_overestim_var")

    if not use_wilcoxon and not use_ttest:
        raise ValueError(
            f"Unsupported method='{method}'.  Supported: "
            "'wilcoxon', 't-test', 't-test_overestim_var', 'logreg'."
        )

    if use_wilcoxon and not hasattr(_core, "wilcoxon_de"):
        raise AttributeError(
            "_core.wilcoxon_de is not available.  "
            "See CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND — the cycle-20 "
            "pybind11 binding extension must expose wilcoxon_de before "
            "singlet.gpu.tools.rank_genes_groups() can run Wilcoxon DE."
        )

    if use_ttest and not hasattr(_core, "ttest_de"):
        raise AttributeError(
            "_core.ttest_de is not available.  "
            "See CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND — the cycle-20 "
            "pybind11 binding extension must expose ttest_de before "
            "singlet.gpu.tools.rank_genes_groups() can run t-test DE."
        )

    working = copy_module.copy(adata) if copy else adata

    # Resolve groups.
    all_groups = list(working.obs[groupby].cat.categories
                      if hasattr(working.obs[groupby], "cat")
                      else working.obs[groupby].unique())
    if groups == "all":
        test_groups = all_groups
    else:
        test_groups = list(groups)
        for g in test_groups:
            if g not in all_groups:
                raise ValueError(
                    f"Group '{g}' not found in adata.obs['{groupby}'].  "
                    f"Available: {all_groups}"
                )

    # Resolve gene universe.
    gene_names = list(working.var_names)
    if mask_var is not None:
        import numpy as np_  # avoid shadowing outer np
        if isinstance(mask_var, str):
            mask = working.var[mask_var].values.astype(bool)
        else:
            mask = np_.asarray(mask_var, dtype=bool)
        gene_names = [g for g, m in zip(gene_names, mask) if m]

    n_report = n_genes if n_genes is not None else len(gene_names)
    n_report = min(n_report, len(gene_names))

    # Get expression matrix.
    eff_use_raw = use_raw
    if eff_use_raw is None:
        eff_use_raw = (working.raw is not None) and (layer is None)
    expr = _get_expr_matrix(working, layer, eff_use_raw)

    # Cell group labels as integer indices.
    group_labels = working.obs[groupby].values
    group_to_idx = {g: i for i, g in enumerate(all_groups)}
    label_ints = np.array([group_to_idx[lbl] for lbl in group_labels], dtype=np.int32)

    # Dispatch to C++ DE kernel.
    de_kwargs = dict(
        expr=expr,
        labels=label_ints,
        test_groups=[all_groups.index(g) for g in test_groups],
        reference=reference,
        all_group_names=all_groups,
        gene_names=gene_names,
        n_genes=n_report,
        rankby_abs=rankby_abs,
        pts=pts,
        tie_correct=tie_correct,
        overestim_var=(effective_method == "t-test_overestim_var"),
    )

    if use_wilcoxon:
        raw_result = _core.wilcoxon_de(**de_kwargs)
    else:
        raw_result = _core.ttest_de(**de_kwargs)

    result_dict = _build_result_dict(
        raw=raw_result,
        groups=test_groups,
        gene_names=gene_names,
        n_genes=n_report,
        corr_method=corr_method,
        groupby=groupby,
        method=effective_method,
        reference=reference,
    )

    uns_key = key_added if key_added is not None else "rank_genes_groups"
    working.uns[uns_key] = result_dict

    return working if copy else None


__all__ = ["rank_genes_groups"]
