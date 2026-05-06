"""Cell type proportion computation across conditions.

Provides singlet.cell_type_proportions() — computes the fraction of each
cell type per sample or condition, useful for differential composition analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad
    import pandas as pd


def cell_type_proportions(
    adata: ad.AnnData,
    groupby: str,
    *,
    condition_key: str | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Compute cell type proportions per condition/sample.

    Calculates the fraction (or raw count) of each cell type within each
    condition group. Useful for identifying shifts in cell composition
    between experimental conditions.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    groupby : str
        Key in ``adata.obs`` for cell type labels (e.g., 'cell_type', 'leiden').
    condition_key : str, optional
        Key in ``adata.obs`` for the condition/sample grouping. If None,
        computes proportions across the entire dataset (single row).
    normalize : bool, default True
        If True, return proportions (fractions summing to 1 per condition).
        If False, return raw cell counts.

    Returns
    -------
    pandas.DataFrame
        Wide-format DataFrame with conditions as rows and cell types as columns.
        Also stored in ``adata.uns['cell_type_proportions']``.

    Examples
    --------
    >>> import singlet
    >>> # Proportions per sample
    >>> props = singlet.cell_type_proportions(adata, 'cell_type', condition_key='sample')
    >>> props
    cell_type   B cell  Monocyte  T cell
    sample
    sample_1      0.2       0.3     0.5
    sample_2      0.1       0.4     0.5

    >>> # Raw counts (no normalization)
    >>> counts = singlet.cell_type_proportions(adata, 'leiden', normalize=False)
    """
    import pandas as pd

    if not hasattr(adata, "obs"):
        raise TypeError(
            f"cell_type_proportions() requires an AnnData object, got {type(adata).__name__}"
        )

    if groupby not in adata.obs.columns:
        raise KeyError(
            f"groupby key '{groupby}' not found in adata.obs. "
            f"Available keys: {list(adata.obs.columns)}"
        )

    if condition_key is not None and condition_key not in adata.obs.columns:
        raise KeyError(
            f"condition_key '{condition_key}' not found in adata.obs. "
            f"Available keys: {list(adata.obs.columns)}"
        )

    if condition_key is not None:
        # Cross-tabulation: conditions × cell types
        ct = pd.crosstab(
            adata.obs[condition_key],
            adata.obs[groupby],
        )
    else:
        # Single row: all cells
        counts_series = adata.obs[groupby].value_counts()
        ct = pd.DataFrame(
            [counts_series.values],
            columns=counts_series.index,
            index=["all"],
        )
        ct.index.name = "condition"

    # Sort columns alphabetically for consistency
    ct = ct.reindex(sorted(ct.columns), axis=1)

    if normalize:
        # Normalize each row to sum to 1
        row_sums = ct.sum(axis=1)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        result = ct.div(row_sums, axis=0)
    else:
        result = ct

    # Store in adata.uns
    adata.uns["cell_type_proportions"] = result

    return result
