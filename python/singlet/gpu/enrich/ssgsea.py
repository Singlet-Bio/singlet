# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.enrich.ssgsea — Per-cell single-sample GSEA enrichment.

Underlying C++: cycle 44, ``enrich/ssgsea.h``.
Algorithm: per-cell KS enrichment over ranked gene expression
           (Barbie et al. 2009; PLAID 2025).

Distinct from ``singlet.gpu.enrichment.run_gsea`` (ranked-list fgsea.h):
- ssgsea scores every (cell, gene-set) pair.
- fgsea scores gene-set enrichment against a ranked statistic vector.

API
---
``run_ssgsea`` — decoupleR-compatible per-cell scoring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import anndata
    import pandas as pd


def _build_genesets_csr(
    net: "pd.DataFrame",
    source: str,
    target: str,
    gene_names: list,
    min_n: int,
) -> tuple:
    """
    Build (set_names, set_offsets, gene_indices) from a decoupleR-style net DataFrame.
    Returns (names, offsets_list, indices_list) or raises ValueError if no sets pass min_n.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    sets = {}
    for set_name, group in net.groupby(source):
        idxs = [gene_to_idx[g] for g in group[target].dropna() if g in gene_to_idx]
        if len(idxs) >= min_n:
            sets[str(set_name)] = idxs
    if not sets:
        raise ValueError(
            f"No gene-sets with ≥{min_n} genes found after intersection with expression matrix."
        )
    names = list(sets.keys())
    offsets = [0]
    indices = []
    for name in names:
        indices.extend(sets[name])
        offsets.append(len(indices))
    return names, offsets, indices


def run_ssgsea(
    mat: "anndata.AnnData",
    net: "pd.DataFrame",
    *,
    source: str = "source",
    target: str = "target",
    alpha: float = 1.0,
    min_n: int = 5,
    cell_tile: int = 8192,
    set_chunk: int = 500,
    normalize: bool = True,
    obsm_key: str = "ssgsea_scores",
    stream=None,
    seed: int = 0,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    Per-cell single-sample GSEA (ssGSEA, cycle 44).

    Parameters
    ----------
    mat : AnnData
        ``mat.X`` is cells × genes.
    net : pd.DataFrame
        Gene-set membership table with ``source`` (set name) and ``target`` (gene name) columns.
    alpha : float
        Rank exponent (1.0 = standard ssGSEA).
    min_n : int
        Minimum genes per set after intersection.
    cell_tile : int
        Cells processed per kernel tile (controls VRAM usage).
    normalize : bool
        Divide ES by n_genes.
    obsm_key : str
        Key written to ``mat.obsm`` with shape (n_cells, n_sets).
    stream : int or None
    seed : int
    copy : bool

    Returns
    -------
    None or AnnData
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "ssgsea"):
        raise AttributeError(
            "_core.ssgsea is not available.  "
            "Ensure the cycle-52a binding extension has been compiled."
        )

    working = mat.copy() if copy else mat
    gene_names = list(working.var_names)

    names, offsets, indices = _build_genesets_csr(net, source, target, gene_names, min_n)

    try:
        import cupy as cp

        try:
            import cupyx.scipy.sparse as csp  # cupy >= 14
        except ImportError:
            import cupy.sparse as csp  # cupy < 14 fallback
        import scipy.sparse as sp

        X = working.X
        if X.shape[0] == working.n_obs:
            X = X.T  # genes × cells
        device_mat = csp.csc_matrix(X) if sp.issparse(X) else csp.csc_matrix(cp.array(X))
    except ImportError as e:
        raise ImportError(f"singlet.gpu.enrich.run_ssgsea requires cupy.  Original error: {e}")

    result = _core.ssgsea(
        device_mat,
        names,
        offsets,
        indices,
        alpha=alpha,
        cell_tile=cell_tile,
        set_chunk=set_chunk,
        normalize=normalize,
        stream=stream,
        seed=seed,
    )

    # scores is [n_sets × n_cells] col-major → transpose to [n_cells × n_sets]
    n_sets = result.n_sets
    n_cells = result.n_cells
    scores_device = cp.asarray(result.scores_view).reshape(n_sets, n_cells)
    scores_host = scores_device.T.get()  # n_cells × n_sets

    working.obsm[obsm_key] = scores_host
    working.uns["ssgsea_gene_set_names"] = names
    working.uns["ssgsea_params"] = {
        "alpha": alpha,
        "normalize": normalize,
        "min_n": min_n,
        "seed": seed,
    }

    return working if copy else None


__all__ = ["run_ssgsea"]
