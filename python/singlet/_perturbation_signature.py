# SPDX-License-Identifier: MIT
"""Perturbation response signature.

Provides singlet.perturbation_signature() — compute per-gene effect size
of a perturbation relative to a control condition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


def perturbation_signature(
    adata: "AnnData",
    condition_key: str,
    *,
    reference: str = "control",
    method: str = "mean_shift",
) -> "pd.DataFrame":
    """Compute perturbation response signature.

    For each non-reference condition, compute per-gene effect sizes
    comparing that condition to the reference (control). Supports
    mean shift, fold change, and Cohen's d.

    Parameters
    ----------
    adata
        Annotated data matrix with condition information in adata.obs.
    condition_key
        Key in adata.obs containing condition/perturbation labels.
    reference
        Label of the reference (control) condition.
    method
        Method for computing effect size:
        - 'mean_shift': difference of means (treated - control).
        - 'fold_change': log2 fold change of means.
        - 'cohen_d': Cohen's d (standardized mean difference).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'gene', 'condition', 'effect_size',
        'pvalue', 'fdr'. One row per gene per non-reference condition.

    Raises
    ------
    ValueError
        If condition_key not found in adata.obs.
        If reference not found in condition values.
        If method is not 'mean_shift', 'fold_change', or 'cohen_d'.
        If fewer than 2 conditions exist.

    Notes
    -----
    Results are also stored in adata.uns['perturbation_signature'] as a dict
    mapping condition names to DataFrames.
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    # --- Validate inputs ---
    if condition_key not in adata.obs.columns:
        msg = (
            f"condition_key '{condition_key}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )
        raise ValueError(msg)

    conditions = adata.obs[condition_key].unique()
    if reference not in conditions:
        msg = (
            f"Reference '{reference}' not found in condition values. Available: {list(conditions)}"
        )
        raise ValueError(msg)

    valid_methods = ("mean_shift", "fold_change", "cohen_d")
    if method not in valid_methods:
        msg = f"method must be one of {valid_methods}, got '{method}'"
        raise ValueError(msg)

    non_ref_conditions = [c for c in conditions if c != reference]
    if len(non_ref_conditions) < 1:
        msg = "Need at least 2 conditions (1 reference + 1 treatment)"
        raise ValueError(msg)

    # --- Extract reference expression ---
    # .to_numpy(): scipy sparse indexing calls ``.nonzero()`` on the mask, which
    # pandas >= 2.0 removed from Series.
    ref_mask = (adata.obs[condition_key] == reference).to_numpy()
    X_ref = adata.X[ref_mask]
    if sp.issparse(X_ref):
        X_ref = X_ref.toarray()
    X_ref = np.asarray(X_ref, dtype=np.float64)

    gene_names = list(adata.var_names)
    all_results = []
    uns_dict = {}

    for cond in non_ref_conditions:
        cond_mask = (adata.obs[condition_key] == cond).to_numpy()
        X_cond = adata.X[cond_mask]
        if sp.issparse(X_cond):
            X_cond = X_cond.toarray()
        X_cond = np.asarray(X_cond, dtype=np.float64)

        # --- Compute effect sizes ---
        ref_mean = X_ref.mean(axis=0)
        cond_mean = X_cond.mean(axis=0)

        if method == "mean_shift":
            effect_sizes = cond_mean - ref_mean
        elif method == "fold_change":
            # Add pseudocount to avoid log(0)
            pseudo = 1e-9
            effect_sizes = np.log2((cond_mean + pseudo) / (ref_mean + pseudo))
        elif method == "cohen_d":
            ref_std = X_ref.std(axis=0, ddof=1)
            cond_std = X_cond.std(axis=0, ddof=1)
            n_ref = X_ref.shape[0]
            n_cond = X_cond.shape[0]
            # Pooled standard deviation
            pooled_std = np.sqrt(
                ((n_ref - 1) * ref_std**2 + (n_cond - 1) * cond_std**2) / (n_ref + n_cond - 2)
            )
            pooled_std = np.where(pooled_std == 0, 1e-9, pooled_std)
            effect_sizes = (cond_mean - ref_mean) / pooled_std

        # --- Compute p-values (Welch's t-test per gene) ---
        pvalues = np.zeros(len(gene_names))
        for gene_idx in range(len(gene_names)):
            _, pval = ttest_ind(
                X_cond[:, gene_idx],
                X_ref[:, gene_idx],
                equal_var=False,
            )
            pvalues[gene_idx] = pval if not np.isnan(pval) else 1.0

        # --- FDR correction ---
        _, fdr_values, _, _ = multipletests(pvalues, method="fdr_bh")

        # --- Build per-condition result ---
        cond_df = pd.DataFrame(
            {
                "gene": gene_names,
                "condition": cond,
                "effect_size": effect_sizes,
                "pvalue": pvalues,
                "fdr": fdr_values,
            }
        )
        all_results.append(cond_df)
        uns_dict[cond] = cond_df

    # --- Store in adata.uns ---
    adata.uns["perturbation_signature"] = uns_dict

    # --- Combine all results ---
    result_df = pd.concat(all_results, ignore_index=True)
    return result_df
