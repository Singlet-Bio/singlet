# SPDX-License-Identifier: MIT
"""
singlet.gpu.enrichment.gsea — GPU-native preranked GSEA.

Underlying C++ cycle: cycle 13 (gsea/fgsea.h —
``singlet::gpu::gsea::fgsea`` kernel).

decoupleR-compatible API
-------------------------
``run_gsea`` mirrors the ``decoupler.run_gsea`` signature.  The ``net``
parameter is a ``pd.DataFrame`` with at minimum two columns:

    source  — gene-set name (one row per gene-set/gene pair)
    target  — gene name

The function accepts either a ranked AnnData or a per-gene statistic
vector (e.g., from differential expression).  When passed an AnnData,
per-gene statistics are derived cell-by-cell from the expression matrix.

Result location
---------------
``adata.obs['gsea_norm_es_<set>']``   — normalised enrichment score per cell.
``adata.obs['gsea_pval_<set>']``      — p-value (beta-approximated or permutation).
``adata.uns['gsea_params']``          — run parameters.

If *mat* is a ``pd.DataFrame`` (e.g., bulk statistics), a ``pd.DataFrame``
is returned directly instead of writing to an AnnData.

CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE: ``_core.fgsea`` must be
exposed by the cycle-22 pybind11 binding extension.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import anndata
    import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_net(net: "pd.DataFrame", source: str, target: str) -> None:
    """Validate that net has the expected source/target columns."""
    import pandas as pd

    if not isinstance(net, pd.DataFrame):
        raise TypeError(f"net must be a pd.DataFrame, got {type(net).__name__!r}.")
    missing = [c for c in (source, target) if c not in net.columns]
    if missing:
        raise KeyError(
            f"net DataFrame is missing columns: {missing}.  "
            f"Available columns: {list(net.columns)}"
        )


def _build_genesets(
    net: "pd.DataFrame",
    source: str,
    target: str,
    min_n: int,
) -> dict:
    """
    Build a dict {set_name: [gene, ...]} from the net DataFrame.

    Gene-sets smaller than *min_n* are silently dropped.
    """
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
    Extract (genes × cells) device matrix and gene names from an AnnData.

    Returns
    -------
    (device_csc_or_cupy, gene_names)  — gene_names is a list of str.
    """
    if use_raw and adata.raw is not None:
        mat = adata.raw.X
        gene_names = list(adata.raw.var_names)
    else:
        mat = adata.X
        gene_names = list(adata.var_names)

    # AnnData stores (cells × genes); C++ expects per-gene stats per cell.
    # For GSEA we need (genes × cells) for the per-gene ranking pass.
    # Transpose lazily — cupy CSC(.T) is a view.
    try:
        try:
            import cupyx.scipy.sparse as csp  # cupy >= 14
        except ImportError:
            import cupy.sparse as csp         # cupy < 14 fallback
        if hasattr(mat, "T"):
            genes_x_cells = mat.T  # cupy CSC or scipy CSC
        else:
            genes_x_cells = mat.T
    except ImportError:
        genes_x_cells = mat.T

    return genes_x_cells, gene_names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_gsea(
    mat: Union["anndata.AnnData", "pd.DataFrame"],
    net: "pd.DataFrame",
    *,
    source: str = "source",
    target: str = "target",
    times: int = 1000,
    min_n: int = 5,
    seed: int = 42,
    use_raw: bool = False,
    copy: bool = False,
) -> Optional[Union["anndata.AnnData", "pd.DataFrame"]]:
    """
    GPU-native preranked GSEA (cycle-13 fgsea kernel).

    Mirrors ``decoupler.run_gsea`` parameter names.  Accepts an AnnData
    (per-cell ranking from the expression matrix) or a ``pd.DataFrame``
    of pre-computed per-gene statistics (bulk mode).

    Parameters
    ----------
    mat : AnnData or pd.DataFrame
        - AnnData: per-gene expression is used as the ranking statistic.
          Results are written to ``adata.obs``.
        - pd.DataFrame (genes × samples): bulk statistics matrix.
          A DataFrame of NES scores is returned.
    net : pd.DataFrame
        Gene-set membership table with at least two columns.  Each row
        represents one gene in one gene-set.
    source : str, default ``"source"``
        Column in *net* that holds gene-set names.
    target : str, default ``"target"``
        Column in *net* that holds gene names.
    times : int, default 1000
        Maximum number of permutations for p-value estimation.
        The C++ kernel uses the FGSEA beta-approximation when
        ``times > 0``; set ``times=0`` to use exact permutation only.
    min_n : int, default 5
        Minimum number of genes a set must have (after intersection with
        the expression matrix gene set) to be scored.
    seed : int, default 42
        Random seed for permutation.
    use_raw : bool, default False
        Use ``adata.raw.X`` instead of ``adata.X`` when *mat* is AnnData.
    copy : bool, default False
        Return a modified copy of *mat* when it is an AnnData.
        When ``False`` (default), *mat* is modified in-place and ``None``
        is returned.  Ignored when *mat* is a DataFrame.

    Returns
    -------
    None
        When *mat* is AnnData and ``copy=False`` (in-place default).
    AnnData
        When *mat* is AnnData and ``copy=True``.
    pd.DataFrame
        When *mat* is a ``pd.DataFrame`` (bulk mode, shape:
        samples × gene-sets with NES values).

    Raises
    ------
    AttributeError
        If ``_core.fgsea`` is not available.
        See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE.
    KeyError
        If *net* is missing the *source* or *target* columns.
    TypeError
        If *net* is not a ``pd.DataFrame``.

    Notes
    -----
    **Result location** (AnnData mode, matches decoupleR):
    - ``adata.obs['gsea_norm_es_<set>']`` — per-cell NES for each gene-set.
    - ``adata.obs['gsea_pval_<set>']``    — per-cell p-value.
    - ``adata.uns['gsea_params']``        — run parameters dict.

    **No host copies in the hot path.**  The expression matrix is already
    device-resident.  The fgsea kernel runs entirely on device; only the
    small (cells × n_sets) NES matrix is transferred to host for writing
    into ``adata.obs``.

    Examples
    --------
    AnnData mode::

        import singlet.gpu.enrichment as sge
        sge.run_gsea(adata, net)
        print(adata.obs.filter(like='gsea_norm_es_').head())

    Bulk / DataFrame mode::

        df_nes = sge.run_gsea(stats_df, net)  # pd.DataFrame

    Copy mode::

        adata2 = sge.run_gsea(adata, net, copy=True)
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

    if not hasattr(_core, "fgsea"):
        raise AttributeError(
            "_core.fgsea is not available.  "
            "See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE — the cycle-22 "
            "pybind11 binding extension must expose fgsea before "
            "singlet.gpu.enrichment.run_gsea() is callable."
        )

    # ---- AnnData mode -------------------------------------------------------
    if hasattr(mat, "obs"):
        adata = mat
        working = copy_module.copy(adata) if copy else adata

        genes_x_cells, gene_names = _matrix_from_adata(working, use_raw)

        # C++ kernel: fgsea(genes_x_cells, gene_names, genesets, times, seed)
        # Returns a dict {set_name: (nes_array, pval_array)} where each array
        # has length == n_cells.
        result = _core.fgsea(
            genes_x_cells,
            gene_names,
            genesets,
            int(times),
            int(seed),
        )

        for set_name, (nes_arr, pval_arr) in result.items():
            safe_name = set_name.replace(" ", "_").replace("/", "_")
            working.obs[f"gsea_norm_es_{safe_name}"] = np.asarray(nes_arr, dtype=np.float32)
            working.obs[f"gsea_pval_{safe_name}"] = np.asarray(pval_arr, dtype=np.float32)

        working.uns["gsea_params"] = {
            "times": times,
            "min_n": min_n,
            "seed": seed,
            "source": source,
            "target": target,
            "n_sets": len(genesets),
        }

        return working if copy else None

    # ---- DataFrame / bulk mode ----------------------------------------------
    if isinstance(mat, pd.DataFrame):
        gene_names = list(mat.index)
        # mat is (genes × samples); we run one GSEA per sample column.
        results_dict: dict = {}
        for sample_col in mat.columns:
            stats_series = mat[sample_col].values.astype(np.float32)
            res = _core.fgsea(
                stats_series,
                gene_names,
                genesets,
                int(times),
                int(seed),
            )
            results_dict[sample_col] = {k: float(v[0][0]) for k, v in res.items()}

        return pd.DataFrame(results_dict).T  # samples × gene-sets

    raise TypeError(
        f"mat must be an AnnData or pd.DataFrame, got {type(mat).__name__!r}."
    )


__all__ = ["run_gsea"]
