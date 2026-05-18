# SPDX-License-Identifier: MIT
"""Batch effect evaluation metrics.

Provides singlet.batch_evaluation() — compute multiple batch mixing metrics
(kBET acceptance rate, LISI, batch ASW) to assess integration quality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def batch_evaluation(
    adata: AnnData,
    batch_key: str,
    *,
    label_key: str | None = None,
    use_rep: str = "X_pca",
    n_neighbors: int = 50,
) -> dict:
    """Compute batch effect evaluation metrics.

    Assesses how well batches are mixed in the embedding space using
    multiple complementary metrics.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with a dimensionality reduction in obsm.
    batch_key : str
        Key in ``adata.obs`` containing batch labels.
    label_key : str or None, default None
        Key in ``adata.obs`` containing biological labels (e.g., cell type).
        If provided, also computes label LISI for bio-conservation assessment.
    use_rep : str, default 'X_pca'
        Key in ``adata.obsm`` for the embedding coordinates.
    n_neighbors : int, default 50
        Number of nearest neighbors for kBET and LISI computation.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'kbet_acceptance_rate': float in [0, 1]. Higher = better mixing.
        - 'batch_lisi': float. Higher = better mixing (max = n_batches).
        - 'label_lisi': float or None. Lower = better bio-conservation.
        - 'batch_asw': float in [-1, 1]. Closer to 0 = better mixing.
        Also stored in ``adata.uns['batch_evaluation']``.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.pca(adata)
    >>> result = singlet.batch_evaluation(adata, batch_key='batch')
    >>> result['kbet_acceptance_rate']
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    # Validate inputs
    if batch_key not in adata.obs.columns:
        msg = f"batch_key {batch_key!r} not found in adata.obs"
        raise KeyError(msg)

    if label_key is not None and label_key not in adata.obs.columns:
        msg = f"label_key {label_key!r} not found in adata.obs"
        raise KeyError(msg)

    if use_rep not in adata.obsm:
        msg = f"Representation {use_rep!r} not found in adata.obsm. Run singlet.pca(adata) first."
        raise KeyError(msg)

    coords = np.asarray(adata.obsm[use_rep])
    n_cells = coords.shape[0]
    k_actual = min(n_neighbors, n_cells - 1)

    if k_actual < 1:
        result = {
            "kbet_acceptance_rate": 1.0,
            "batch_lisi": 1.0,
            "label_lisi": None,
            "batch_asw": 0.0,
        }
        adata.uns["batch_evaluation"] = result
        return result

    # Build kNN
    nn = NearestNeighbors(n_neighbors=k_actual, algorithm="auto")
    nn.fit(coords)
    indices = nn.kneighbors(coords, return_distance=False)

    batch_labels = adata.obs[batch_key].values
    unique_batches = np.unique(batch_labels)

    # --- kBET acceptance rate ---
    # For each cell, test if batch composition in neighborhood matches global
    # using a chi-squared test. Acceptance = fraction of cells that pass.
    global_batch_freq = np.array([(batch_labels == b).sum() / n_cells for b in unique_batches])

    accepted = 0
    for cell_idx in range(n_cells):
        neighbor_batches = batch_labels[indices[cell_idx]]
        observed = np.array(
            [(neighbor_batches == b).sum() for b in unique_batches], dtype=np.float64
        )
        expected = global_batch_freq * k_actual

        # Chi-squared statistic
        # Only include bins with expected > 0
        valid = expected > 0
        if valid.sum() < 2:
            accepted += 1
            continue

        chi2_stat = np.sum((observed[valid] - expected[valid]) ** 2 / expected[valid])
        df = valid.sum() - 1

        # p-value from chi-squared distribution (simplified)
        # Use survival function approximation
        pval = _chi2_pvalue(chi2_stat, df)
        if pval > 0.05:
            accepted += 1

    kbet_acceptance = accepted / n_cells

    # --- LISI (Local Inverse Simpson Index) ---
    batch_lisi = _compute_lisi(indices, batch_labels, unique_batches)

    label_lisi = None
    if label_key is not None:
        label_labels = adata.obs[label_key].values
        unique_labels = np.unique(label_labels)
        label_lisi = _compute_lisi(indices, label_labels, unique_labels)

    # --- Batch ASW (Average Silhouette Width by batch) ---
    batch_asw = _compute_batch_asw(coords, batch_labels, unique_batches)

    result = {
        "kbet_acceptance_rate": float(kbet_acceptance),
        "batch_lisi": float(batch_lisi),
        "label_lisi": float(label_lisi) if label_lisi is not None else None,
        "batch_asw": float(batch_asw),
    }

    adata.uns["batch_evaluation"] = result
    return result


def _compute_lisi(
    indices,
    labels,
    unique_labels,
) -> float:
    """Compute mean Local Inverse Simpson Index."""
    import numpy as np

    n_cells = indices.shape[0]
    lisi_values = np.zeros(n_cells, dtype=np.float64)

    for cell_idx in range(n_cells):
        neighbor_labels = labels[indices[cell_idx]]
        # Frequency of each label in neighborhood
        freqs = np.array(
            [(neighbor_labels == lab).sum() for lab in unique_labels],
            dtype=np.float64,
        )
        total = freqs.sum()
        if total == 0:
            lisi_values[cell_idx] = 1.0
            continue
        probs = freqs / total
        # Simpson index = sum(p^2), LISI = 1/Simpson
        simpson = np.sum(probs**2)
        lisi_values[cell_idx] = 1.0 / simpson if simpson > 0 else 1.0

    return float(np.mean(lisi_values))


def _compute_batch_asw(coords, batch_labels, unique_batches) -> float:
    """Compute average silhouette width for batch labels.

    Returns value closer to 0 when batches are well-mixed.
    """
    import numpy as np

    n_cells = coords.shape[0]
    n_batches = len(unique_batches)

    if n_batches < 2 or n_cells < 2:
        return 0.0

    # Use sklearn silhouette if available, with sampling for large datasets
    n_sample = min(n_cells, 5000)
    rng = np.random.default_rng(42)
    if n_sample < n_cells:
        sample_idx = rng.choice(n_cells, n_sample, replace=False)
        coords_s = coords[sample_idx]
        labels_s = batch_labels[sample_idx]
    else:
        coords_s = coords
        labels_s = batch_labels

    # Check we still have at least 2 unique labels in sample
    if len(np.unique(labels_s)) < 2:
        return 0.0

    from sklearn.metrics import silhouette_score

    asw = silhouette_score(coords_s, labels_s, metric="euclidean")
    return float(asw)


def _chi2_pvalue(chi2_stat: float, df: int) -> float:
    """Approximate chi-squared p-value using regularized gamma function."""
    import numpy as np

    if df <= 0:
        return 1.0
    if chi2_stat <= 0:
        return 1.0

    # Use scipy if available, else approximate
    try:
        from scipy.stats import chi2

        return float(chi2.sf(chi2_stat, df))
    except ImportError:
        # Rough approximation: normal approximation for large df
        z_val = np.sqrt(2 * chi2_stat) - np.sqrt(2 * df - 1)
        # Standard normal CDF approximation
        pval = 0.5 * (1.0 - np.tanh(z_val * 0.7))
        return float(np.clip(pval, 0.0, 1.0))
