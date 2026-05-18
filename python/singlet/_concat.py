# SPDX-License-Identifier: MIT
"""Concatenation of multiple AnnData objects.

Provides singlet.concatenate() — merges multiple AnnData objects into one,
handling different gene sets via outer/inner join.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import anndata


def concatenate(
    adatas: list,
    *,
    join: str = "inner",
    batch_key: str = "batch",
    batch_categories: Optional[list[str]] = None,
    index_unique: str = "-",
) -> "anndata.AnnData":
    """Concatenate multiple AnnData objects.

    Merges cells from multiple datasets, handling gene intersection/union
    and adding batch labels.

    Parameters
    ----------
    adatas : list of AnnData
        AnnData objects to concatenate.
    join : str, default "inner"
        How to handle genes: "inner" keeps only shared genes,
        "outer" keeps all genes (missing filled with zeros).
    batch_key : str, default "batch"
        Column name in obs for batch labels.
    batch_categories : list[str] or None, default None
        Labels for each batch. If None, uses "0", "1", "2", ...
    index_unique : str, default "-"
        Separator for making obs_names unique. Each batch's cell
        names are suffixed with f"{index_unique}{batch_idx}".
        Set to None to keep original names (may cause duplicates).

    Returns
    -------
    anndata.AnnData
        Concatenated AnnData object.

    Examples
    --------
    >>> import singlet
    >>> adata1 = singlet.load("GSM1234567")
    >>> adata2 = singlet.load("GSM1234568")
    >>> combined = singlet.concatenate([adata1, adata2])
    >>> combined.obs["batch"].value_counts()
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    if not adatas:
        raise ValueError("adatas list must not be empty.")

    if len(adatas) == 1:
        import anndata as ad

        result = adatas[0].copy()
        if batch_key:
            cat = batch_categories[0] if batch_categories else "0"
            result.obs[batch_key] = pd.Categorical([cat] * result.n_obs)
        return result

    # Validate all are AnnData
    for i, a in enumerate(adatas):
        if not hasattr(a, "X") or not hasattr(a, "var_names"):
            raise TypeError(f"Element {i} is not an AnnData object (got {type(a).__name__})")

    # Determine batch categories
    if batch_categories is None:
        batch_categories = [str(i) for i in range(len(adatas))]
    elif len(batch_categories) != len(adatas):
        raise ValueError(
            f"batch_categories length ({len(batch_categories)}) "
            f"must match adatas length ({len(adatas)})"
        )

    # Determine gene set
    all_var_names = [set(a.var_names) for a in adatas]

    if join == "inner":
        common_genes = set.intersection(*all_var_names)
        if not common_genes:
            raise ValueError("No shared genes found across datasets (inner join).")
        # Preserve order from first AnnData
        gene_order = [g for g in adatas[0].var_names if g in common_genes]
    elif join == "outer":
        all_genes = set.union(*all_var_names)
        # Order: first dataset order, then remaining genes alphabetically
        gene_order = list(adatas[0].var_names)
        remaining = sorted(all_genes - set(gene_order))
        gene_order.extend(remaining)
    else:
        raise ValueError(f"join must be 'inner' or 'outer', got '{join}'")

    n_genes = len(gene_order)
    gene_to_idx = {g: i for i, g in enumerate(gene_order)}

    # Build concatenated matrix
    X_blocks = []
    obs_list = []
    obs_names_list = []

    for batch_idx, (adata, cat) in enumerate(zip(adatas, batch_categories)):
        n_cells = adata.n_obs

        # Map genes
        if join == "inner":
            # Subset to common genes (in correct order)
            col_indices = [list(adata.var_names).index(g) for g in gene_order]
            if sp.issparse(adata.X):
                X_batch = adata.X[:, col_indices]
            else:
                X_batch = adata.X[:, col_indices]
        else:
            # Outer join: create full-width matrix, fill matched columns
            if sp.issparse(adata.X):
                X_batch = sp.lil_matrix((n_cells, n_genes), dtype=np.float32)
                for j, gene in enumerate(adata.var_names):
                    if gene in gene_to_idx:
                        col = gene_to_idx[gene]
                        X_batch[:, col] = adata.X[:, j]
                X_batch = X_batch.tocsr()
            else:
                X_batch = np.zeros((n_cells, n_genes), dtype=np.float32)
                for j, gene in enumerate(adata.var_names):
                    if gene in gene_to_idx:
                        col = gene_to_idx[gene]
                        X_batch[:, col] = adata.X[:, j]

        X_blocks.append(X_batch)

        # Obs metadata
        obs_df = adata.obs.copy()
        obs_df[batch_key] = cat

        # Make names unique
        if index_unique is not None:
            names = [f"{name}{index_unique}{batch_idx}" for name in adata.obs_names]
        else:
            names = list(adata.obs_names)

        obs_names_list.extend(names)
        obs_list.append(obs_df)

    # Stack matrices
    if all(sp.issparse(x) for x in X_blocks):
        X_combined = sp.vstack(X_blocks, format="csr")
    else:
        X_combined = np.vstack([x.toarray() if sp.issparse(x) else x for x in X_blocks])

    # Combine obs
    obs_combined = pd.concat(obs_list, ignore_index=True)
    obs_combined.index = obs_names_list
    obs_combined[batch_key] = pd.Categorical(obs_combined[batch_key], categories=batch_categories)

    # Create result AnnData
    import anndata as ad

    result = ad.AnnData(X=X_combined)
    result.obs = obs_combined
    result.var_names = gene_order

    return result
