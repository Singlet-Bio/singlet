"""Gene module scoring for AnnData objects.

Computes per-cell enrichment scores for user-defined gene modules using a
control-gene background strategy. For each module, control genes are drawn
from the same expression bin as the module genes, and the score is the
difference in mean expression between the module and control genes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def gene_module_score(
    adata,
    gene_modules: dict,
    *,
    ctrl_size: int = 50,
    n_bins: int = 25,
    random_state: int = 0,
) -> "pd.DataFrame":
    """Score cells for enrichment of gene modules.

    For each module, computes a per-cell score defined as the mean expression
    of the module genes minus the mean expression of a set of control genes
    drawn from the same expression bins. This controls for differences in
    average expression level.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with genes in columns (adata.var_names) and
        cells in rows. Expression values are taken from adata.X.
    gene_modules : dict
        Mapping of module name (str) to gene list (list of str). Each gene
        should correspond to an entry in adata.var_names.
    ctrl_size : int, default 50
        Number of control genes sampled per expression bin for each module
        gene.
    n_bins : int, default 25
        Number of expression-level bins used to stratify genes.
    random_state : int, default 0
        Random seed for reproducible control gene sampling.

    Returns
    -------
    pandas.DataFrame
        DataFrame with cells as rows (indexed by adata.obs_names) and one
        column per module containing the enrichment score.

    Raises
    ------
    TypeError
        If ``adata`` is not an AnnData object or ``gene_modules`` is not a
        dict.
    ValueError
        If ``gene_modules`` is empty, or if a module has no valid genes
        remaining after filtering to genes present in adata.var_names.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> modules = {"cell_cycle": ["CDK1", "MKI67", "TOP2A"]}
    >>> scores = singlet.gene_module_score(adata, modules)
    >>> scores.shape[1]
    1
    """
    import warnings

    import numpy as np
    import pandas as pd
    from scipy import sparse

    # --- Input validation ---
    if not hasattr(adata, "obs") or not hasattr(adata, "var_names"):
        raise TypeError(
            f"gene_module_score() requires an AnnData object, got {type(adata).__name__}"
        )

    if not isinstance(gene_modules, dict):
        raise TypeError(f"gene_modules must be a dict, got {type(gene_modules).__name__}")

    if len(gene_modules) == 0:
        raise ValueError("gene_modules must not be empty")

    rng = np.random.default_rng(random_state)

    # --- Compute mean expression per gene for binning ---
    X = adata.X
    if sparse.issparse(X):
        mean_expr = np.asarray(X.mean(axis=0)).ravel()
    else:
        mean_expr = np.mean(X, axis=0).ravel()

    var_names = np.array(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(var_names)}

    # Bin genes by mean expression
    bin_indices = pd.cut(mean_expr, bins=n_bins, labels=False)
    # pd.cut can return NaN for genes exactly at the boundary; assign to last bin
    bin_indices = np.where(np.isnan(bin_indices), n_bins - 1, bin_indices).astype(int)

    # Pre-compute bin membership: bin -> list of gene indices
    bin_members = {}
    for bin_id in range(n_bins):
        bin_members[bin_id] = np.where(bin_indices == bin_id)[0]

    # --- Score each module ---
    scores = {}

    for module_name, gene_list in gene_modules.items():
        # Filter to genes present in adata
        valid_indices = []
        missing_genes = []
        for gene in gene_list:
            if gene in gene_to_idx:
                valid_indices.append(gene_to_idx[gene])
            else:
                missing_genes.append(gene)

        if missing_genes:
            warnings.warn(
                f"Module '{module_name}': {len(missing_genes)} gene(s) not found in "
                f"adata.var_names and will be skipped: {missing_genes[:5]}"
                + ("..." if len(missing_genes) > 5 else ""),
                stacklevel=2,
            )

        if len(valid_indices) == 0:
            raise ValueError(
                f"Module '{module_name}' has no valid genes after filtering to adata.var_names"
            )

        valid_indices = np.array(valid_indices)
        module_gene_set = set(valid_indices)

        # Get module gene expression (cells × module_genes)
        if sparse.issparse(X):
            module_expr = np.asarray(X[:, valid_indices].toarray())
        else:
            module_expr = np.asarray(X[:, valid_indices])

        # Sample control genes from matching expression bins
        ctrl_indices = []
        for gene_idx in valid_indices:
            gene_bin = bin_indices[gene_idx]
            candidates = bin_members[gene_bin]
            # Exclude module genes from candidates
            candidates = candidates[~np.isin(candidates, list(module_gene_set))]

            if len(candidates) == 0:
                # Fall back to all non-module genes in nearby bins
                candidates = np.concatenate(
                    [
                        bin_members.get(gene_bin + offset, np.array([], dtype=int))
                        for offset in [-1, 0, 1]
                    ]
                )
                candidates = candidates[~np.isin(candidates, list(module_gene_set))]

            n_sample = min(ctrl_size, len(candidates))
            if n_sample > 0:
                sampled = rng.choice(candidates, size=n_sample, replace=False)
                ctrl_indices.append(sampled)

        if ctrl_indices:
            ctrl_indices_all = np.unique(np.concatenate(ctrl_indices))
        else:
            ctrl_indices_all = np.array([], dtype=int)

        # Get control gene expression
        if len(ctrl_indices_all) > 0:
            if sparse.issparse(X):
                ctrl_expr = np.asarray(X[:, ctrl_indices_all].toarray())
            else:
                ctrl_expr = np.asarray(X[:, ctrl_indices_all])
            ctrl_mean = ctrl_expr.mean(axis=1)
        else:
            ctrl_mean = np.zeros(X.shape[0])

        # Score = mean(module genes) - mean(control genes)
        module_mean = module_expr.mean(axis=1)
        score = module_mean - ctrl_mean

        scores[module_name] = score
        adata.obs[f"score_{module_name}"] = score

    # Store metadata
    adata.uns["gene_module_score"] = {
        "params": {
            "ctrl_size": ctrl_size,
            "n_bins": n_bins,
            "random_state": random_state,
        },
        "modules": list(gene_modules.keys()),
    }

    result = pd.DataFrame(scores, index=adata.obs_names)
    return result
