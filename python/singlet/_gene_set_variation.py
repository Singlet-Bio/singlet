"""Gene Set Variation Analysis (GSVA) and ssGSEA.

Provides singlet.gene_set_variation() — computes per-cell pathway enrichment
scores using either the GSVA or ssGSEA algorithm. Unlike over-representation
analysis, these methods assign a continuous score to each cell for each gene
set, enabling downstream differential pathway analysis.

References
----------
Hänzelmann, S., Castelo, R., & Guinney, J. (2013). GSVA: gene set variation
analysis for microarray and RNA-seq data. BMC Bioinformatics, 14, 7.

Barbie, D. A., et al. (2009). Systematic RNA interference reveals that
oncogenic KRAS-driven cancers require TBK1. Nature, 462(7269), 108-112.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def gene_set_variation(
    adata,
    gene_sets: dict[str, list[str]],
    *,
    method: str = "gsva",
    kcdf: str = "Gaussian",
) -> "pd.DataFrame":
    """Compute per-cell gene set enrichment scores via GSVA or ssGSEA.

    For each cell and gene set, computes a continuous enrichment score that
    reflects the relative expression of genes inside vs. outside the set.
    GSVA uses a non-parametric kernel density estimation approach, while
    ssGSEA uses a rank-weighted running sum statistic.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with genes in columns (adata.var_names) and
        cells in rows. Expression values are taken from adata.X.
    gene_sets : dict[str, list[str]]
        Dictionary mapping pathway/gene set names to lists of gene symbols.
        Genes not found in adata.var_names are silently ignored. Gene sets
        with fewer than 2 matching genes receive NaN scores.
    method : {'gsva', 'ssgsea'}, default 'gsva'
        Algorithm to use:

        - ``'gsva'``: Gene Set Variation Analysis. Applies kernel density
          estimation to rank-transformed expression, then computes a
          random-walk enrichment statistic per cell.
        - ``'ssgsea'``: Single-sample GSEA. Ranks genes per cell, then
          computes a weighted running sum whose integral (area under
          curve) defines the enrichment score.
    kcdf : {'Gaussian', 'Poisson'}, default 'Gaussian'
        Kernel for cumulative density estimation (GSVA method only).

        - ``'Gaussian'``: Standard normal CDF applied to z-scored ranks.
          Appropriate for log-transformed continuous expression.
        - ``'Poisson'``: Rank-based approach with Poisson-like weighting.
          Appropriate for integer count data (e.g., UMI counts).

    Returns
    -------
    pandas.DataFrame
        DataFrame of shape (n_cells, n_gene_sets) with enrichment scores.
        Indexed by adata.obs_names, columns are gene set names. Also stored
        in ``adata.obsm['gsva_scores']``.

    Raises
    ------
    TypeError
        If ``adata`` is not an AnnData object or ``gene_sets`` is not a dict.
    ValueError
        If ``method`` or ``kcdf`` is not a recognized value, or if
        ``gene_sets`` is empty.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> pathways = {
    ...     "cell_cycle": ["CDK1", "MKI67", "TOP2A", "CCNB1"],
    ...     "apoptosis": ["BAX", "BCL2", "CASP3", "CASP9"],
    ... }
    >>> scores = singlet.gene_set_variation(adata, pathways, method="gsva")
    >>> scores.shape[1]
    2
    """
    import numpy as np
    import pandas as pd
    from scipy import sparse

    # --- Input validation ---
    if not hasattr(adata, "obs") or not hasattr(adata, "var_names"):
        raise TypeError(
            f"gene_set_variation() requires an AnnData object, got {type(adata).__name__}"
        )

    if not isinstance(gene_sets, dict):
        raise TypeError(f"gene_sets must be a dict, got {type(gene_sets).__name__}")

    if len(gene_sets) == 0:
        raise ValueError("gene_sets must not be empty")

    method = method.lower()
    if method not in ("gsva", "ssgsea"):
        raise ValueError(f"method must be 'gsva' or 'ssgsea', got {method!r}")

    if kcdf not in ("Gaussian", "Poisson"):
        raise ValueError(f"kcdf must be 'Gaussian' or 'Poisson', got {kcdf!r}")

    # --- Extract dense expression matrix (cells × genes) ---
    expr = adata.X
    if sparse.issparse(expr):
        expr = np.asarray(expr.todense())
    else:
        expr = np.asarray(expr)

    n_cells, n_genes = expr.shape
    var_names = list(adata.var_names)
    gene_to_idx = {g: idx for idx, g in enumerate(var_names)}

    # --- Filter gene sets to genes present in data ---
    filtered_sets: dict[str, list[int]] = {}
    for set_name, genes in gene_sets.items():
        indices = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        filtered_sets[set_name] = indices

    # --- Dispatch to method ---
    if method == "gsva":
        scores_array = _gsva(expr, filtered_sets, n_cells, n_genes, kcdf)
    else:
        scores_array = _ssgsea(expr, filtered_sets, n_cells, n_genes)

    # --- Build output DataFrame ---
    scores_df = pd.DataFrame(
        scores_array,
        index=adata.obs_names,
        columns=list(gene_sets.keys()),
    )
    adata.obsm["gsva_scores"] = scores_df

    return scores_df


def _gsva(
    expr: "np.ndarray",
    filtered_sets: dict[str, list[int]],
    n_cells: int,
    n_genes: int,
    kcdf: str,
) -> "np.ndarray":
    """GSVA algorithm: KDE-transformed ranks + random walk statistic."""
    import numpy as np
    from scipy.stats import norm

    # Step 1: Rank genes per cell (axis=1), using average for ties
    ranks = np.empty_like(expr, dtype=np.float64)
    for cell_idx in range(n_cells):
        ranks[cell_idx] = _rank_row(expr[cell_idx])

    # Step 2: Kernel density estimation to get smoothed cumulative density
    if kcdf == "Gaussian":
        # Z-score the ranks and apply normal CDF
        rank_mean = ranks.mean(axis=1, keepdims=True)
        rank_std = ranks.std(axis=1, keepdims=True)
        # Avoid division by zero for cells with constant expression
        rank_std = np.where(rank_std == 0, 1.0, rank_std)
        z_ranks = (ranks - rank_mean) / rank_std
        density = norm.cdf(z_ranks)
    else:
        # Poisson: use rank / (n_genes + 1) as empirical CDF with
        # Poisson-like weighting (heavier tails for low-expressed genes)
        density = ranks / (n_genes + 1.0)
        # Apply a Poisson-inspired transformation: 1 - exp(-rank)
        density = 1.0 - np.exp(-density * np.log(n_genes))

    # Step 3: Compute enrichment scores for each gene set
    scores = np.empty((n_cells, len(filtered_sets)), dtype=np.float64)

    for set_idx, (set_name, gene_indices) in enumerate(filtered_sets.items()):
        n_in_set = len(gene_indices)
        if n_in_set < 2:
            scores[:, set_idx] = np.nan
            continue

        in_set_mask = np.zeros(n_genes, dtype=bool)
        in_set_mask[gene_indices] = True
        n_not_in_set = n_genes - n_in_set

        # For each cell, compute the random-walk enrichment score
        for cell_idx in range(n_cells):
            cell_density = density[cell_idx]
            # Sort genes by density value (descending)
            sort_order = np.argsort(-cell_density)
            sorted_in_set = in_set_mask[sort_order]

            # Running sum: in-set genes contribute positively,
            # out-of-set genes contribute negatively
            step_up = 1.0 / n_in_set
            step_down = 1.0 / n_not_in_set

            running_sum = np.where(sorted_in_set, step_up, -step_down)
            cumsum = np.cumsum(running_sum)

            # Enrichment score: max deviation (bidirectional)
            max_dev = cumsum.max()
            min_dev = cumsum.min()
            if abs(max_dev) >= abs(min_dev):
                scores[cell_idx, set_idx] = max_dev
            else:
                scores[cell_idx, set_idx] = min_dev

    return scores


def _ssgsea(
    expr: "np.ndarray",
    filtered_sets: dict[str, list[int]],
    n_cells: int,
    n_genes: int,
) -> "np.ndarray":
    """ssGSEA algorithm: rank-weighted running sum integral."""
    import numpy as np

    alpha = 0.25

    # Step 1: Rank genes per cell (ascending, 1-based)
    ranks = np.empty_like(expr, dtype=np.float64)
    for cell_idx in range(n_cells):
        ranks[cell_idx] = _rank_row(expr[cell_idx])

    # Step 2: Compute enrichment scores
    scores = np.empty((n_cells, len(filtered_sets)), dtype=np.float64)

    for set_idx, (set_name, gene_indices) in enumerate(filtered_sets.items()):
        n_in_set = len(gene_indices)
        if n_in_set < 2:
            scores[:, set_idx] = np.nan
            continue

        in_set_mask = np.zeros(n_genes, dtype=bool)
        in_set_mask[gene_indices] = True
        n_not_in_set = n_genes - n_in_set
        penalty = 1.0 / n_not_in_set

        for cell_idx in range(n_cells):
            cell_ranks = ranks[cell_idx]
            # Sort genes by expression rank (descending)
            sort_order = np.argsort(-cell_ranks)
            sorted_in_set = in_set_mask[sort_order]
            sorted_ranks = cell_ranks[sort_order]

            # Weighted ranks for in-set genes
            in_set_ranks_powered = np.where(sorted_in_set, np.abs(sorted_ranks) ** alpha, 0.0)
            sum_in_set_ranks = in_set_ranks_powered.sum()

            if sum_in_set_ranks == 0:
                scores[cell_idx, set_idx] = 0.0
                continue

            # Running sum: in-set adds weighted rank, out-of-set subtracts
            step_values = np.where(
                sorted_in_set,
                in_set_ranks_powered / sum_in_set_ranks,
                -penalty,
            )
            cumsum = np.cumsum(step_values)

            # ssGSEA score = sum of running sum (area under curve)
            scores[cell_idx, set_idx] = cumsum.sum()

    # Step 3: Normalize scores across cells (zero-center, unit variance per set)
    for set_idx in range(scores.shape[1]):
        col = scores[:, set_idx]
        if np.all(np.isnan(col)):
            continue
        col_std = np.nanstd(col)
        if col_std > 0:
            scores[:, set_idx] = (col - np.nanmean(col)) / col_std

    return scores


def _rank_row(row: "np.ndarray") -> "np.ndarray":
    """Rank values in a 1-D array using average method (1-based).

    Equivalent to scipy.stats.rankdata(row, method='average') but avoids
    the overhead of repeated function-call dispatch for many rows.
    """
    import numpy as np

    n = len(row)
    sort_idx = np.argsort(row, kind="mergesort")
    ranked = np.empty(n, dtype=np.float64)

    pos = 0
    while pos < n:
        # Find run of tied values
        tie_start = pos
        while pos < n and row[sort_idx[pos]] == row[sort_idx[tie_start]]:
            pos += 1
        # Assign average rank (1-based)
        avg_rank = (tie_start + pos + 1) / 2.0  # +1 for 1-based
        for tied_pos in range(tie_start, pos):
            ranked[sort_idx[tied_pos]] = avg_rank

    return ranked
