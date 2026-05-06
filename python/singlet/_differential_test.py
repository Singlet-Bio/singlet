"""Distribution-level differential testing between cell groups.

Tests whether the overall distribution of cells differs between groups
in embedding space, using non-parametric statistical tests.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


def differential_test(
    adata: "AnnData",
    groupby: str,
    *,
    groups: list[str] | None = None,
    method: str = "ks",
    use_rep: str = "X_pca",
) -> "pd.DataFrame":
    """Test whether cell distributions differ between groups in embedding space.

    Performs distribution-level (not gene-by-gene) differential testing between
    cell groups using non-parametric statistics on embedding coordinates.

    Parameters
    ----------
    adata
        Annotated data matrix with embedding in ``adata.obsm[use_rep]``.
    groupby
        Column in ``adata.obs`` defining groups to compare.
    groups
        Specific groups to compare. If ``None``, all pairwise comparisons are
        performed.
    method
        Statistical test to use:
        - ``'ks'``: Kolmogorov-Smirnov 2-sample test (averaged across PCs)
        - ``'energy'``: Energy distance test
        - ``'mmd'``: Maximum Mean Discrepancy with RBF kernel
    use_rep
        Key in ``adata.obsm`` for the embedding to use.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: group1, group2, statistic, pvalue.
        Also stored in ``adata.uns['differential_test']``.

    Raises
    ------
    KeyError
        If ``groupby`` is not in ``adata.obs`` or ``use_rep`` not in
        ``adata.obsm``.
    ValueError
        If ``method`` is not one of 'ks', 'energy', 'mmd', or if any
        specified group is not found.
    """
    import pandas as pd

    if groupby not in adata.obs.columns:
        msg = f"Column '{groupby}' not found in adata.obs"
        raise KeyError(msg)
    if use_rep not in adata.obsm:
        msg = f"Representation '{use_rep}' not found in adata.obsm"
        raise KeyError(msg)

    valid_methods = {"ks", "energy", "mmd"}
    if method not in valid_methods:
        msg = f"Method must be one of {valid_methods}, got '{method}'"
        raise ValueError(msg)

    labels = adata.obs[groupby]
    embedding = adata.obsm[use_rep]

    if groups is None:
        groups = sorted(labels.unique().tolist())
    else:
        available = set(labels.unique())
        for grp in groups:
            if grp not in available:
                msg = f"Group '{grp}' not found in adata.obs['{groupby}']"
                raise ValueError(msg)

    results = []
    for grp1, grp2 in combinations(groups, 2):
        mask1 = labels.values == grp1
        mask2 = labels.values == grp2
        emb1 = embedding[mask1]
        emb2 = embedding[mask2]

        if method == "ks":
            stat, pval = _ks_test(emb1, emb2)
        elif method == "energy":
            stat, pval = _energy_test(emb1, emb2)
        else:  # mmd
            stat, pval = _mmd_test(emb1, emb2)

        results.append({"group1": grp1, "group2": grp2, "statistic": stat, "pvalue": pval})

    result_df = pd.DataFrame(results)
    adata.uns["differential_test"] = result_df

    return result_df


def _ks_test(emb1: np.ndarray, emb2: np.ndarray) -> tuple[float, float]:
    """Average KS statistic and combined p-value across dimensions."""
    from scipy import stats

    n_dims = emb1.shape[1]
    ks_stats = []
    pvalues = []

    for dim in range(n_dims):
        stat, pval = stats.ks_2samp(emb1[:, dim], emb2[:, dim])
        ks_stats.append(stat)
        pvalues.append(pval)

    # Average statistic, Fisher's method for combining p-values
    avg_stat = float(np.mean(ks_stats))
    # Fisher's method: -2 * sum(log(p)) ~ chi2(2k)
    pvalues_arr = np.array(pvalues)
    # Clip p-values to avoid log(0)
    pvalues_arr = np.clip(pvalues_arr, 1e-300, 1.0)
    chi2_stat = -2.0 * np.sum(np.log(pvalues_arr))
    combined_pval = float(stats.chi2.sf(chi2_stat, 2 * n_dims))

    return avg_stat, combined_pval


def _energy_test(emb1: np.ndarray, emb2: np.ndarray) -> tuple[float, float]:
    """Energy distance between two samples with permutation test."""
    rng = np.random.default_rng(42)

    energy_stat = _compute_energy_distance(emb1, emb2)

    # Permutation test for p-value
    n1 = len(emb1)
    combined = np.vstack([emb1, emb2])
    n_perm = 199
    perm_stats = np.empty(n_perm)

    for idx in range(n_perm):
        perm = rng.permutation(len(combined))
        perm_emb1 = combined[perm[:n1]]
        perm_emb2 = combined[perm[n1:]]
        perm_stats[idx] = _compute_energy_distance(perm_emb1, perm_emb2)

    pval = float((np.sum(perm_stats >= energy_stat) + 1) / (n_perm + 1))

    return float(energy_stat), pval


def _compute_energy_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute energy distance between two point clouds."""
    from scipy.spatial.distance import cdist

    # Energy distance = 2*E[|X-Y|] - E[|X-X'|] - E[|Y-Y'|]
    cross = cdist(emb1, emb2, metric="euclidean")
    within1 = cdist(emb1, emb1, metric="euclidean")
    within2 = cdist(emb2, emb2, metric="euclidean")

    energy = 2.0 * np.mean(cross) - np.mean(within1) - np.mean(within2)
    return max(energy, 0.0)


def _mmd_test(emb1: np.ndarray, emb2: np.ndarray) -> tuple[float, float]:
    """Maximum Mean Discrepancy with RBF kernel and permutation test."""
    rng = np.random.default_rng(42)

    # Bandwidth selection: median heuristic
    from scipy.spatial.distance import cdist

    combined = np.vstack([emb1, emb2])
    # Subsample for bandwidth estimation if large
    if len(combined) > 500:
        sub_idx = rng.choice(len(combined), 500, replace=False)
        sub = combined[sub_idx]
    else:
        sub = combined

    dists = cdist(sub, sub, metric="sqeuclidean")
    median_dist = np.median(dists[dists > 0])
    bandwidth = median_dist if median_dist > 0 else 1.0

    mmd_stat = _compute_mmd(emb1, emb2, bandwidth)

    # Permutation test
    n1 = len(emb1)
    n_perm = 199
    perm_stats = np.empty(n_perm)

    for idx in range(n_perm):
        perm = rng.permutation(len(combined))
        perm_emb1 = combined[perm[:n1]]
        perm_emb2 = combined[perm[n1:]]
        perm_stats[idx] = _compute_mmd(perm_emb1, perm_emb2, bandwidth)

    pval = float((np.sum(perm_stats >= mmd_stat) + 1) / (n_perm + 1))

    return float(mmd_stat), pval


def _compute_mmd(emb1: np.ndarray, emb2: np.ndarray, bandwidth: float) -> float:
    """Compute MMD^2 with RBF kernel."""
    from scipy.spatial.distance import cdist

    k_xx = np.exp(-cdist(emb1, emb1, metric="sqeuclidean") / bandwidth)
    k_yy = np.exp(-cdist(emb2, emb2, metric="sqeuclidean") / bandwidth)
    k_xy = np.exp(-cdist(emb1, emb2, metric="sqeuclidean") / bandwidth)

    mmd_sq = np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy)
    return max(mmd_sq, 0.0)
