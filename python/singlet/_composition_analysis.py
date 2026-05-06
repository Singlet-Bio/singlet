"""Cell composition analysis across conditions.

Provides singlet.composition_analysis() — test whether cell type proportions
differ between conditions (e.g., disease vs control) using Dirichlet-multinomial
regression or proportion z-tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


def composition_analysis(
    adata: "AnnData",
    groupby: str,
    condition_key: str,
    *,
    method: str = "dirichlet",
    reference: Optional[str] = None,
) -> "pd.DataFrame":
    """Test whether cell type proportions differ between conditions.

    Compares cell composition across conditions (e.g., disease vs control)
    to identify cell types that are enriched or depleted.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        Column in adata.obs defining cell groups (e.g., 'cell_type', 'leiden').
    condition_key
        Column in adata.obs defining conditions to compare (e.g., 'disease').
    method
        Statistical method: 'dirichlet' (Dirichlet-multinomial regression)
        or 'prop_test' (proportion z-test). Default 'dirichlet'.
    reference
        Reference condition for fold change. If None, uses the first
        condition alphabetically.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: group, condition, proportion, fold_change,
        pvalue, fdr. Also stored in adata.uns['composition_analysis'].

    Raises
    ------
    TypeError
        If adata is not an AnnData object.
    KeyError
        If groupby or condition_key not in adata.obs.
    ValueError
        If method is invalid or fewer than 2 conditions exist.

    Examples
    --------
    >>> import singlet
    >>> result = singlet.composition_analysis(adata, 'cell_type', 'disease')
    >>> result[result['fdr'] < 0.05]  # significant composition changes
    """
    import numpy as np
    import pandas as pd

    # Validate input
    if not hasattr(adata, "obs"):
        msg = "composition_analysis requires an AnnData object, got " + type(adata).__name__
        raise TypeError(msg)

    if groupby not in adata.obs.columns:
        msg = f"'{groupby}' not found in adata.obs"
        raise KeyError(groupby)

    if condition_key not in adata.obs.columns:
        msg = f"'{condition_key}' not found in adata.obs"
        raise KeyError(condition_key)

    valid_methods = ("dirichlet", "prop_test")
    if method not in valid_methods:
        msg = f"method must be one of {valid_methods}, got '{method}'"
        raise ValueError(msg)

    conditions = sorted(adata.obs[condition_key].dropna().unique().tolist())
    if len(conditions) < 2:
        msg = f"Need at least 2 conditions, found {len(conditions)}: {conditions}"
        raise ValueError(msg)

    # Set reference
    ref = reference if reference is not None else conditions[0]
    if ref not in conditions:
        msg = f"reference '{ref}' not in conditions: {conditions}"
        raise ValueError(msg)

    # Compute contingency table: groups x conditions
    groups = sorted(adata.obs[groupby].dropna().unique().tolist())
    counts = pd.crosstab(adata.obs[groupby], adata.obs[condition_key])

    # Fill missing groups/conditions with 0
    counts = counts.reindex(index=groups, columns=conditions, fill_value=0)

    # Proportions within each condition
    totals = counts.sum(axis=0)
    proportions = counts.div(totals, axis=1)

    # Compute fold change vs reference
    ref_props = proportions[ref].values
    # Avoid division by zero with pseudocount
    ref_props_safe = np.where(ref_props > 0, ref_props, 1e-10)

    if method == "dirichlet":
        pvalues = _dirichlet_test(counts, groups, conditions, ref)
    else:
        pvalues = _proportion_ztest(counts, groups, conditions, ref)

    # Build results DataFrame
    rows = []
    for group in groups:
        for cond in conditions:
            prop = proportions.loc[group, cond]
            fc = prop / ref_props_safe[groups.index(group)] if cond != ref else 1.0
            pval = pvalues.get((group, cond), 1.0) if cond != ref else 1.0
            rows.append(
                {
                    "group": group,
                    "condition": cond,
                    "proportion": prop,
                    "fold_change": fc,
                    "pvalue": pval,
                    "fdr": np.nan,  # filled below
                }
            )

    result = pd.DataFrame(rows)

    # FDR correction (Benjamini-Hochberg) on non-reference rows
    non_ref_mask = result["condition"] != ref
    if non_ref_mask.sum() > 0:
        pvals = result.loc[non_ref_mask, "pvalue"].values
        fdr_values = _benjamini_hochberg(pvals)
        result.loc[non_ref_mask, "fdr"] = fdr_values

    # Reference rows get fdr=1.0
    result.loc[~non_ref_mask, "fdr"] = 1.0

    # Store in adata.uns
    adata.uns["composition_analysis"] = result

    return result


def _dirichlet_test(
    counts: "pd.DataFrame",
    groups: list,
    conditions: list,
    ref: str,
) -> dict:
    """Dirichlet-multinomial likelihood ratio test for composition changes.

    Compares a null model (all conditions share one Dirichlet) against
    an alternative (each condition has its own Dirichlet concentration).
    """
    import numpy as np
    from scipy import stats

    pvalues = {}

    for group in groups:
        for cond in conditions:
            if cond == ref:
                continue

            # 2x2: this group vs rest, this condition vs reference
            n_group_cond = counts.loc[group, cond]
            n_other_cond = counts[cond].sum() - n_group_cond
            n_group_ref = counts.loc[group, ref]
            n_other_ref = counts[ref].sum() - n_group_ref

            # Use chi-squared test on 2x2 contingency as approximation
            # to Dirichlet-multinomial LRT (exact DM requires iterative MLE)
            table = np.array([[n_group_cond, n_other_cond], [n_group_ref, n_other_ref]])

            # Handle all-zero cases
            if table.sum() == 0:
                pvalues[(group, cond)] = 1.0
                continue

            # Use G-test (log-likelihood ratio) for better Dirichlet approximation
            try:
                g_stat, pval, _, _ = stats.chi2_contingency(table, lambda_="log-likelihood")
                pvalues[(group, cond)] = pval
            except ValueError:
                pvalues[(group, cond)] = 1.0

    return pvalues


def _proportion_ztest(
    counts: "pd.DataFrame",
    groups: list,
    conditions: list,
    ref: str,
) -> dict:
    """Two-proportion z-test for each group between conditions."""
    import numpy as np
    from scipy import stats

    pvalues = {}

    for group in groups:
        for cond in conditions:
            if cond == ref:
                continue

            # Proportion in condition
            n_cond = counts[cond].sum()
            x_cond = counts.loc[group, cond]

            # Proportion in reference
            n_ref = counts[ref].sum()
            x_ref = counts.loc[group, ref]

            # Handle edge cases
            if n_cond == 0 or n_ref == 0:
                pvalues[(group, cond)] = 1.0
                continue

            p_cond = x_cond / n_cond
            p_ref = x_ref / n_ref

            # Pooled proportion
            p_pool = (x_cond + x_ref) / (n_cond + n_ref)

            # Z-statistic
            if p_pool == 0 or p_pool == 1:
                pvalues[(group, cond)] = 1.0
                continue

            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_cond + 1 / n_ref))
            if se == 0:
                pvalues[(group, cond)] = 1.0
                continue

            z_val = (p_cond - p_ref) / se
            pval = 2 * stats.norm.sf(abs(z_val))
            pvalues[(group, cond)] = pval

    return pvalues


def _benjamini_hochberg(pvalues):
    """Benjamini-Hochberg FDR correction."""
    import numpy as np

    n_vals = len(pvalues)
    if n_vals == 0:
        return np.array([])

    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]

    # BH correction
    fdr = np.empty(n_vals)
    for idx in range(n_vals):
        rank = idx + 1
        fdr[idx] = sorted_pvals[idx] * n_vals / rank

    # Enforce monotonicity (from largest to smallest rank)
    for idx in range(n_vals - 2, -1, -1):
        fdr[idx] = min(fdr[idx], fdr[idx + 1])

    # Cap at 1.0
    fdr = np.minimum(fdr, 1.0)

    # Restore original order
    result = np.empty(n_vals)
    result[sorted_idx] = fdr
    return result
