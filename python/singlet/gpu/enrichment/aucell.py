# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.enrichment.aucell — GPU-native AUCell gene-set scoring.

Underlying C++ cycle: cycle 13 (gsea/aucell.h —
``singlet::gpu::gsea::aucell`` kernel).

decoupleR-compatible API
-------------------------
``run_aucell`` mirrors the ``decoupler.run_aucell`` signature.  The ``net``
parameter is a ``pd.DataFrame`` with at minimum two columns:

    source  — gene-set name (one row per gene-set/gene pair)
    target  — gene name

AUCell scores the per-cell recovery of each gene-set by computing the
Area Under the (per-cell ranked recovery) Curve (AUC) for each gene-set
independently.

Result location
---------------
``adata.obsm['X_aucell']``  — (n_cells × n_sets) float32 AUC scores.
``adata.uns['aucell_params']`` — run parameters.

If *mat* is a ``pd.DataFrame``, a matching DataFrame is returned.

CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE: ``_core.aucell`` must be
exposed by the cycle-22 pybind11 binding extension.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import anndata
    import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_net(net: "pd.DataFrame", source: str, target: str) -> None:
    import pandas as pd

    if not isinstance(net, pd.DataFrame):
        raise TypeError(f"net must be a pd.DataFrame, got {type(net).__name__!r}.")
    missing = [c for c in (source, target) if c not in net.columns]
    if missing:
        raise KeyError(
            f"net DataFrame is missing columns: {missing}.  Available columns: {list(net.columns)}"
        )


def _build_genesets(
    net: "pd.DataFrame",
    source: str,
    target: str,
    min_n: int,
) -> dict:
    """Build a dict {set_name: [gene, ...]} filtered by min_n."""
    genesets: dict = {}
    for set_name, group in net.groupby(source):
        genes = list(group[target].dropna().unique())
        if len(genes) >= min_n:
            genesets[str(set_name)] = genes
    return genesets


def _matrix_from_adata(
    adata: "anndata.AnnData",
    use_raw: bool,
) -> tuple:
    """
    Extract expression matrix and gene names.

    Returns
    -------
    (mat, gene_names)
        mat       — (cells × genes) GPU-resident matrix (cupy sparse CSR).
        gene_names — list of str, length == n_genes.
    """
    if use_raw and adata.raw is not None:
        mat = adata.raw.X
        gene_names = list(adata.raw.var_names)
    else:
        mat = adata.X
        gene_names = list(adata.var_names)
    return mat, gene_names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_aucell(
    mat: Union["anndata.AnnData", "pd.DataFrame"],
    net: "pd.DataFrame",
    *,
    source: str = "source",
    target: str = "target",
    min_n: int = 5,
    seed: int = 42,
    use_raw: bool = False,
    copy: bool = False,
) -> Optional[Union["anndata.AnnData", "pd.DataFrame"]]:
    """
    GPU-native AUCell gene-set scoring (cycle-13 aucell kernel).

    Scores each cell for each gene-set by computing the Area Under the
    recovery Curve of the gene-set genes in the per-cell expression ranking.
    The AUC is normalised by the maximum possible AUC for the set size.

    Mirrors ``decoupler.run_aucell`` parameter names.  Accepts an AnnData
    (per-cell scoring) or a ``pd.DataFrame`` of expression values (bulk).

    Parameters
    ----------
    mat : AnnData or pd.DataFrame
        - AnnData: per-gene expression is ranked per cell.
          Results are written to ``adata.obsm['X_aucell']``.
        - pd.DataFrame (genes × samples): ranked per sample.
          A DataFrame of AUC scores is returned.
    net : pd.DataFrame
        Gene-set membership table with at least two columns.
    source : str, default ``"source"``
        Column in *net* that holds gene-set names.
    target : str, default ``"target"``
        Column in *net* that holds gene names.
    min_n : int, default 5
        Minimum number of genes per set (post-intersection) to be scored.
    seed : int, default 42
        Random seed (used for tie-breaking in rank computation).
    use_raw : bool, default False
        Use ``adata.raw.X`` instead of ``adata.X`` when *mat* is AnnData.
    copy : bool, default False
        Return a modified copy of *mat* when it is an AnnData.
        When ``False`` (default), *mat* is modified in-place and ``None``
        is returned.  Ignored when *mat* is a DataFrame.

    Returns
    -------
    None
        When *mat* is AnnData and ``copy=False`` (default in-place).
    AnnData
        When *mat* is AnnData and ``copy=True``.
    pd.DataFrame
        When *mat* is a ``pd.DataFrame`` (shape: samples × gene-sets).

    Raises
    ------
    AttributeError
        If ``_core.aucell`` is not available.
        See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE.
    KeyError
        If *net* is missing the *source* or *target* columns.
    TypeError
        If *net* is not a ``pd.DataFrame``.

    Notes
    -----
    **Result location** (AnnData mode, matches decoupleR):
    - ``adata.obsm['X_aucell']`` — float32 matrix of shape
      (n_cells × n_sets).  Column order matches ``adata.uns['aucell_params']['set_names']``.
    - ``adata.uns['aucell_params']`` — run parameters + set names.

    **No host copies in the hot path.**  The expression matrix is
    device-resident; the aucell kernel ranks and scores entirely on device.
    Only the compact (cells × n_sets) score matrix is transferred to host
    for writing into ``adata.obsm``.

    Examples
    --------
    Default AUCell::

        import singlet.gpu.enrichment as sge
        sge.run_aucell(adata, net)
        print(adata.obsm['X_aucell'].shape)  # (n_cells, n_sets)

    Copy mode::

        adata2 = sge.run_aucell(adata, net, copy=True)
    """
    import pandas as pd

    _validate_net(net, source, target)
    genesets = _build_genesets(net, source, target, min_n)

    if not genesets:
        raise ValueError(
            f"No gene-sets with ≥{min_n} genes found after filtering net.  "
            "Check min_n or the target column."
        )

    import singlet.gpu._core as _core

    if not hasattr(_core, "aucell"):
        raise AttributeError(
            "_core.aucell is not available.  "
            "See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE — the cycle-22 "
            "pybind11 binding extension must expose aucell before "
            "singlet.gpu.enrichment.run_aucell() is callable."
        )

    set_names = list(genesets.keys())

    # ---- AnnData mode -------------------------------------------------------
    if hasattr(mat, "obs"):
        adata = mat
        working = copy_module.copy(adata) if copy else adata

        expr_mat, gene_names = _matrix_from_adata(working, use_raw)

        # C++ kernel: aucell(cells_x_genes_mat, gene_names, genesets, seed)
        # Returns a (n_cells × n_sets) device array (cupy or raw pointer).
        auc_device = _core.aucell(
            expr_mat,
            gene_names,
            genesets,
            int(seed),
        )

        # Transfer small score matrix to host numpy for obsm storage.
        auc_host = np.asarray(auc_device, dtype=np.float32)  # (n_cells × n_sets)

        working.obsm["X_aucell"] = auc_host
        working.uns["aucell_params"] = {
            "min_n": min_n,
            "seed": seed,
            "source": source,
            "target": target,
            "set_names": set_names,
            "n_sets": len(set_names),
        }

        return working if copy else None

    # ---- DataFrame / bulk mode ----------------------------------------------
    if isinstance(mat, pd.DataFrame):
        gene_names = list(mat.index)
        # mat is (genes × samples); run per sample.
        col_results = {}
        for sample_col in mat.columns:
            stats_col = mat[[sample_col]]  # keep 2D shape for C++ interface
            auc_device = _core.aucell(
                stats_col.values.T.astype(np.float32),  # (1 × genes)
                gene_names,
                genesets,
                int(seed),
            )
            col_results[sample_col] = np.asarray(auc_device, dtype=np.float32).ravel()

        return pd.DataFrame(col_results, index=set_names).T  # samples × sets

    raise TypeError(f"mat must be an AnnData or pd.DataFrame, got {type(mat).__name__!r}.")


__all__ = ["run_aucell"]
