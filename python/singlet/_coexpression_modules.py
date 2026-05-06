"""WGCNA-style coexpression module detection.

Provides singlet.coexpression_modules() — identify modules of co-expressed
genes using soft-thresholded correlation and hierarchical clustering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def coexpression_modules(
    adata: "AnnData",
    *,
    n_modules: int = 10,
    method: str = "wgcna_lite",
    min_module_size: int = 30,
    power: int = 6,
    n_top_genes: int = 2000,
    layer: str | None = None,
    random_state: int = 0,
) -> dict:
    """Identify coexpression modules using WGCNA-style analysis.

    Computes a gene-gene correlation matrix, applies soft-thresholding,
    optionally computes topological overlap, and uses hierarchical
    clustering to define gene modules. Module eigengenes (first PC of
    each module) are computed to summarize module activity per cell.

    Parameters
    ----------
    adata
        Annotated data matrix (log-normalized recommended).
    n_modules
        Target number of modules for tree cutting.
    method
        Module detection method:
        - 'wgcna_lite': soft-power → TOM → hierarchical clustering
        - 'correlation_cluster': abs correlation → hierarchical clustering
    min_module_size
        Minimum number of genes per module. Smaller modules are merged
        into the unassigned module (label 0).
    power
        Soft-thresholding power for adjacency (WGCNA beta parameter).
    n_top_genes
        Number of top variable genes to use for module detection.
    layer
        Expression layer. None uses .X.
    random_state
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'modules': dict mapping module_id (int) → list of gene names
        - 'eigengenes': pd.DataFrame (n_obs × n_modules) of module eigengenes
        - 'n_modules': number of modules detected (excluding unassigned)

    Also stores:
        - adata.var['coexpression_module']: module assignment per gene
          (0 = unassigned, 1..k for modules)
        - adata.obsm['module_eigengenes']: module eigengene matrix

    Raises
    ------
    ValueError
        If method is not recognized or n_top_genes < min_module_size.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> result = singlet.coexpression_modules(adata, n_modules=8)
    >>> print(f"Found {result['n_modules']} modules")
    """
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import cut_tree, linkage
    from scipy.sparse import issparse
    from scipy.spatial.distance import squareform

    valid_methods = ("wgcna_lite", "correlation_cluster")
    if method not in valid_methods:
        msg = f"method must be one of {valid_methods}, got '{method}'"
        raise ValueError(msg)

    if n_top_genes < min_module_size:
        msg = f"n_top_genes ({n_top_genes}) must be >= min_module_size ({min_module_size})"
        raise ValueError(msg)

    # Get expression matrix
    X = adata.layers[layer] if layer is not None else adata.X
    if issparse(X):
        X = np.asarray(X.todense())
    else:
        X = np.asarray(X, dtype=np.float64)

    # Select top variable genes
    n_genes = X.shape[1]
    n_use = min(n_top_genes, n_genes)
    gene_var = np.var(X, axis=0)
    top_idx = np.argsort(gene_var)[::-1][:n_use]
    top_idx = np.sort(top_idx)  # Keep original ordering
    gene_names = np.array(adata.var_names)[top_idx]
    X_sub = X[:, top_idx]

    # Compute correlation matrix
    # Center each gene
    X_centered = X_sub - X_sub.mean(axis=0, keepdims=True)
    norms = np.sqrt((X_centered**2).sum(axis=0, keepdims=True))
    norms[norms == 0] = 1.0
    X_normed = X_centered / norms
    cor_mat = X_normed.T @ X_normed  # n_genes x n_genes

    # Clip to [-1, 1]
    np.clip(cor_mat, -1.0, 1.0, out=cor_mat)

    if method == "wgcna_lite":
        # Soft-thresholding: adjacency = |cor|^power
        adjacency = np.abs(cor_mat) ** power

        # Topological Overlap Matrix (TOM)
        # TOM_ij = (sum_u(a_iu * a_uj) + a_ij) / (min(k_i, k_j) + 1 - a_ij)
        # where k_i = sum_u(a_iu)
        connectivity = adjacency.sum(axis=0) - np.diag(adjacency)
        numerator = adjacency @ adjacency + adjacency
        np.fill_diagonal(numerator, 0)

        # Compute denominator
        ki = connectivity[:, np.newaxis]
        kj = connectivity[np.newaxis, :]
        min_k = np.minimum(ki, kj)
        denominator = min_k + 1.0 - adjacency
        denominator[denominator == 0] = 1.0

        tom = numerator / denominator
        np.fill_diagonal(tom, 1.0)
        np.clip(tom, 0.0, 1.0, out=tom)

        # Distance = 1 - TOM
        dist_mat = 1.0 - tom
    else:
        # correlation_cluster: distance = 1 - |cor|
        dist_mat = 1.0 - np.abs(cor_mat)

    # Ensure diagonal is 0 and matrix is symmetric
    np.fill_diagonal(dist_mat, 0.0)
    dist_mat = (dist_mat + dist_mat.T) / 2.0
    np.clip(dist_mat, 0.0, None, out=dist_mat)

    # Hierarchical clustering
    dist_condensed = squareform(dist_mat, checks=False)
    linkage_mat = linkage(dist_condensed, method="average")

    # Cut tree to get target n_modules
    labels = cut_tree(linkage_mat, n_clusters=n_modules).ravel()

    # Relabel from 1..k, merging small modules to 0
    unique_labels = np.unique(labels)
    module_map = {}
    module_id = 1
    for lab in unique_labels:
        count = (labels == lab).sum()
        if count >= min_module_size:
            module_map[lab] = module_id
            module_id += 1
        else:
            module_map[lab] = 0  # unassigned

    final_labels = np.array([module_map[lab] for lab in labels])

    # Build module gene lists
    modules = {}
    for mid in sorted(set(final_labels)):
        mask = final_labels == mid
        modules[int(mid)] = list(gene_names[mask])

    # Compute module eigengenes (first PC of each module)
    n_real_modules = len([m for m in modules if m != 0])
    eigengene_matrix = np.zeros((adata.n_obs, n_real_modules))
    eigengene_names = []

    for idx, mid in enumerate(sorted(m for m in modules if m != 0)):
        module_genes = modules[mid]
        gene_idx_in_sub = [i for i, gn in enumerate(gene_names) if gn in module_genes]
        X_mod = X_sub[:, gene_idx_in_sub]

        # Center and compute first PC via SVD
        X_mod_c = X_mod - X_mod.mean(axis=0, keepdims=True)
        # Use truncated SVD for efficiency
        try:
            u, s, _vt = np.linalg.svd(X_mod_c, full_matrices=False)
            eigengene_matrix[:, idx] = u[:, 0] * s[0]
        except np.linalg.LinAlgError:
            eigengene_matrix[:, idx] = 0.0
        eigengene_names.append(f"ME{mid}")

    eigengenes_df = pd.DataFrame(
        eigengene_matrix,
        index=adata.obs_names,
        columns=eigengene_names,
    )

    # Store in adata
    # Map all genes to module labels (unassigned genes not in top set get 0)
    all_module_labels = np.zeros(adata.n_vars, dtype=np.int32)
    for i, idx in enumerate(top_idx):
        all_module_labels[idx] = final_labels[i]
    adata.var["coexpression_module"] = all_module_labels
    adata.obsm["module_eigengenes"] = eigengene_matrix

    result = {
        "modules": {k: v for k, v in modules.items() if k != 0},
        "eigengenes": eigengenes_df,
        "n_modules": n_real_modules,
    }

    # Also store unassigned if any
    if 0 in modules:
        result["unassigned"] = modules[0]

    adata.uns["coexpression_modules"] = {
        "modules": result["modules"],
        "n_modules": n_real_modules,
        "params": {
            "method": method,
            "power": power,
            "n_top_genes": n_top_genes,
            "min_module_size": min_module_size,
        },
    }

    return result
