# SPDX-License-Identifier: MIT
"""Cell-cell communication inference (CellPhoneDB-style).

Estimates intercellular signalling by scoring ligand-receptor interactions
between cell type pairs and assessing significance via permutation testing
or expression-based z-scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


def cell_communication(
    adata: "AnnData",
    ligand_receptor_pairs,
    *,
    groupby: str = "leiden",
    method: str = "permutation",
    n_perms: int = 100,
) -> "pd.DataFrame":
    """Infer cell-cell communication from ligand-receptor co-expression.

    For every ligand-receptor pair and every ordered (sender, receiver) cell
    type combination, a communication score is computed as::

        score = mean(ligand expr in sender) * mean(receptor expr in receiver)

    Significance is assessed either by a label-permutation test or by
    computing z-scores across all pair scores.

    Parameters
    ----------
    adata
        Annotated data matrix with expression in ``adata.X``.
    ligand_receptor_pairs
        Either a list of ``(ligand, receptor)`` tuples or a
        :class:`~pandas.DataFrame` with columns ``'ligand'`` and
        ``'receptor'``.
    groupby
        Column in ``adata.obs`` containing cell type labels.
    method
        One of ``'permutation'`` or ``'expression'``.

        * ``'permutation'``: shuffle cell type labels *n_perms* times and
          compute an empirical p-value as the fraction of permuted scores
          >= the observed score.
        * ``'expression'``: compute a z-score for each interaction score
          relative to the distribution of all scores, then derive a
          two-sided p-value.
    n_perms
        Number of permutations for the permutation test. Ignored when
        *method* is ``'expression'``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``ligand``, ``receptor``, ``sender``,
        ``receiver``, ``score``, ``pvalue``, ``fdr``.  Results are sorted
        by ascending ``fdr`` then ``pvalue``.  Also stored in
        ``adata.uns['cell_communication']``.
    """
    import pandas as pd

    if method not in ("permutation", "expression"):
        msg = f"method must be 'permutation' or 'expression', got {method!r}"
        raise ValueError(msg)

    if groupby not in adata.obs.columns:
        msg = f"Column {groupby!r} not found in adata.obs"
        raise KeyError(msg)

    # --- Parse ligand-receptor pairs ---
    lr_pairs = _parse_lr_pairs(ligand_receptor_pairs)

    # --- Filter to pairs where both genes exist in the dataset ---
    var_set = set(adata.var_names)
    valid_pairs = [(lig, rec) for lig, rec in lr_pairs if lig in var_set and rec in var_set]
    if not valid_pairs:
        msg = "No valid ligand-receptor pairs found in adata.var_names"
        raise ValueError(msg)

    # --- Get cell type labels and unique groups ---
    labels = adata.obs[groupby].values
    cell_types = sorted(set(labels))

    # Require at least 1 cell per group (drop empty groups)
    cell_types = [ct for ct in cell_types if np.sum(labels == ct) > 0]

    # --- Precompute expression arrays for valid genes ---
    ligands = list({lig for lig, _ in valid_pairs})
    receptors = list({rec for _, rec in valid_pairs})
    all_genes = list(set(ligands + receptors))

    # Extract dense expression matrix for genes of interest
    gene_idx = [list(adata.var_names).index(g) for g in all_genes]
    expr_matrix = adata.X[:, gene_idx]
    if hasattr(expr_matrix, "toarray"):
        expr_matrix = expr_matrix.toarray()
    expr_matrix = np.asarray(expr_matrix, dtype=np.float64)

    gene_to_col = {g: idx for idx, g in enumerate(all_genes)}

    # --- Compute observed scores ---
    records = []

    # Precompute masks for each cell type
    ct_masks = {ct: labels == ct for ct in cell_types}

    # Precompute mean expression per gene per cell type
    mean_expr = np.zeros((len(all_genes), len(cell_types)), dtype=np.float64)
    ct_index = {ct: idx for idx, ct in enumerate(cell_types)}
    for gi, gene in enumerate(all_genes):
        col = expr_matrix[:, gi]
        for ct in cell_types:
            mask = ct_masks[ct]
            mean_expr[gi, ct_index[ct]] = col[mask].mean()

    # Build result records with observed scores
    observed_scores = []
    result_keys = []  # (pair_idx, sender_idx, receiver_idx)
    for lig, rec in valid_pairs:
        lig_col = gene_to_col[lig]
        rec_col = gene_to_col[rec]
        for sender in cell_types:
            si = ct_index[sender]
            lig_mean = mean_expr[lig_col, si]
            for receiver in cell_types:
                ri = ct_index[receiver]
                rec_mean = mean_expr[rec_col, ri]
                score = lig_mean * rec_mean
                observed_scores.append(score)
                result_keys.append((lig, rec, sender, receiver))

    observed_scores = np.array(observed_scores, dtype=np.float64)

    # --- Significance testing ---
    if method == "permutation":
        pvalues = _permutation_test(
            expr_matrix=expr_matrix,
            labels=labels,
            cell_types=cell_types,
            valid_pairs=valid_pairs,
            gene_to_col=gene_to_col,
            observed_scores=observed_scores,
            n_perms=n_perms,
        )
    else:
        pvalues = _expression_zscore(observed_scores)

    # --- FDR correction (Benjamini-Hochberg) ---
    fdr = _benjamini_hochberg(pvalues)

    # --- Assemble DataFrame ---
    records = []
    for idx, (lig, rec, sender, receiver) in enumerate(result_keys):
        records.append(
            {
                "ligand": lig,
                "receptor": rec,
                "sender": sender,
                "receiver": receiver,
                "score": observed_scores[idx],
                "pvalue": pvalues[idx],
                "fdr": fdr[idx],
            }
        )

    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values(["fdr", "pvalue"]).reset_index(drop=True)

    adata.uns["cell_communication"] = result_df
    return result_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_lr_pairs(ligand_receptor_pairs) -> list[tuple[str, str]]:
    """Convert input to a list of (ligand, receptor) tuples."""
    import pandas as pd

    if isinstance(ligand_receptor_pairs, pd.DataFrame):
        if "ligand" not in ligand_receptor_pairs.columns:
            msg = "DataFrame must have a 'ligand' column"
            raise ValueError(msg)
        if "receptor" not in ligand_receptor_pairs.columns:
            msg = "DataFrame must have a 'receptor' column"
            raise ValueError(msg)
        return list(
            zip(
                ligand_receptor_pairs["ligand"],
                ligand_receptor_pairs["receptor"],
            )
        )

    # Assume iterable of tuples/lists
    return [(str(pair[0]), str(pair[1])) for pair in ligand_receptor_pairs]


def _permutation_test(
    *,
    expr_matrix: np.ndarray,
    labels: np.ndarray,
    cell_types: list[str],
    valid_pairs: list[tuple[str, str]],
    gene_to_col: dict[str, int],
    observed_scores: np.ndarray,
    n_perms: int,
) -> np.ndarray:
    """Compute empirical p-values via label permutation.

    Vectorises computation across all L-R pairs and cell type combinations
    per permutation for efficiency.
    """
    rng = np.random.default_rng(seed=42)
    n_scores = len(observed_scores)
    counts = np.zeros(n_scores, dtype=np.int64)

    n_ct = len(cell_types)
    ct_index = {ct: idx for idx, ct in enumerate(cell_types)}
    n_genes = expr_matrix.shape[1]

    for _ in range(n_perms):
        perm_labels = rng.permutation(labels)

        # Compute mean expression per gene per permuted cell type
        perm_mean = np.zeros((n_genes, n_ct), dtype=np.float64)
        for ci, ct in enumerate(cell_types):
            mask = perm_labels == ct
            if mask.sum() > 0:
                perm_mean[:, ci] = expr_matrix[mask].mean(axis=0)

        # Compute permuted scores in same order as observed
        score_idx = 0
        for lig, rec in valid_pairs:
            lig_col = gene_to_col[lig]
            rec_col = gene_to_col[rec]
            for sender in cell_types:
                si = ct_index[sender]
                lig_mean = perm_mean[lig_col, si]
                for receiver in cell_types:
                    ri = ct_index[receiver]
                    rec_mean = perm_mean[rec_col, ri]
                    perm_score = lig_mean * rec_mean
                    if perm_score >= observed_scores[score_idx]:
                        counts[score_idx] += 1
                    score_idx += 1

    # Empirical p-value: add 1 to numerator/denominator for conservative estimate
    pvalues = (counts + 1) / (n_perms + 1)
    return pvalues


def _expression_zscore(observed_scores: np.ndarray) -> np.ndarray:
    """Compute p-values from z-scores of the score distribution."""
    from scipy.stats import norm

    mean_score = observed_scores.mean()
    std_score = observed_scores.std(ddof=1)

    if std_score == 0 or np.isnan(std_score):
        return np.ones(len(observed_scores), dtype=np.float64)

    z_scores = (observed_scores - mean_score) / std_score
    # Two-sided p-value from z-score
    pvalues = 2.0 * norm.sf(np.abs(z_scores))
    return pvalues


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""
    n_tests = len(pvalues)
    if n_tests == 0:
        return np.array([], dtype=np.float64)

    # Sort p-values and track original order
    order = np.argsort(pvalues)
    sorted_pvals = pvalues[order]

    # BH adjustment: p_adj[i] = p[i] * n / rank
    ranks = np.arange(1, n_tests + 1, dtype=np.float64)
    adjusted = sorted_pvals * n_tests / ranks

    # Enforce monotonicity (from largest rank down)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Restore original order
    fdr = np.empty(n_tests, dtype=np.float64)
    fdr[order] = adjusted
    return fdr
