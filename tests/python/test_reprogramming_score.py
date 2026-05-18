# SPDX-License-Identifier: MIT
"""Tests for singlet.reprogramming_score."""

from __future__ import annotations

import numpy as np
import pytest

import singlet


@pytest.fixture()
def adata_with_types():
    """Create a small AnnData with cell types and PCA embedding."""
    import anndata as ad

    rng = np.random.default_rng(42)
    n_cells = 150
    n_genes = 100
    n_pcs = 20

    X = rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32)

    # Create 3 cell type clusters in PCA space
    pca = np.zeros((n_cells, n_pcs))
    # Source cluster: centered at (-5, 0, 0, ...)
    pca[:50, 0] = rng.normal(-5, 0.5, 50)
    pca[:50, 1] = rng.normal(0, 0.5, 50)
    # Target cluster: centered at (5, 0, 0, ...)
    pca[50:100, 0] = rng.normal(5, 0.5, 50)
    pca[50:100, 1] = rng.normal(0, 0.5, 50)
    # Intermediate cluster: centered at (0, 0, 0, ...)
    pca[100:150, 0] = rng.normal(0, 0.5, 50)
    pca[100:150, 1] = rng.normal(0, 0.5, 50)

    cell_types = (
        ["source"] * 50 + ["target"] * 50 + ["intermediate"] * 50
    )

    adata = ad.AnnData(X=X)
    adata.obs["cell_type"] = cell_types
    adata.obsm["X_pca"] = pca

    return adata


def test_basic_reprogramming_score(adata_with_types):
    """Test basic functionality returns expected structure."""
    result = singlet.reprogramming_score(
        adata_with_types, source_type="source", target_type="target"
    )

    assert result is adata_with_types
    assert "reprogramming_score" in adata_with_types.obs.columns
    scores = adata_with_types.obs["reprogramming_score"].values
    assert scores.shape == (150,)
    assert np.all(scores >= 0)
    assert np.all(scores <= 1)


def test_source_cells_have_low_score(adata_with_types):
    """Source cells should have low reprogramming score."""
    singlet.reprogramming_score(
        adata_with_types, source_type="source", target_type="target"
    )
    scores = adata_with_types.obs["reprogramming_score"].values
    source_mean = np.mean(scores[:50])
    target_mean = np.mean(scores[50:100])
    assert source_mean < target_mean


def test_target_cells_have_high_score(adata_with_types):
    """Target cells should have high reprogramming score."""
    singlet.reprogramming_score(
        adata_with_types, source_type="source", target_type="target"
    )
    scores = adata_with_types.obs["reprogramming_score"].values
    target_mean = np.mean(scores[50:100])
    assert target_mean > 0.7


def test_intermediate_cells_have_middle_score(adata_with_types):
    """Intermediate cells should score between source and target."""
    singlet.reprogramming_score(
        adata_with_types, source_type="source", target_type="target"
    )
    scores = adata_with_types.obs["reprogramming_score"].values
    source_mean = np.mean(scores[:50])
    intermediate_mean = np.mean(scores[100:150])
    target_mean = np.mean(scores[50:100])
    assert source_mean < intermediate_mean < target_mean


def test_custom_type_key(adata_with_types):
    """Test using a custom type key column."""
    adata_with_types.obs["lineage"] = adata_with_types.obs["cell_type"]
    singlet.reprogramming_score(
        adata_with_types,
        source_type="source",
        target_type="target",
        type_key="lineage",
    )
    assert "reprogramming_score" in adata_with_types.obs.columns


def test_custom_use_rep(adata_with_types):
    """Test using a custom embedding key."""
    adata_with_types.obsm["X_custom"] = adata_with_types.obsm["X_pca"][:, :5]
    singlet.reprogramming_score(
        adata_with_types,
        source_type="source",
        target_type="target",
        use_rep="X_custom",
    )
    assert "reprogramming_score" in adata_with_types.obs.columns


def test_missing_type_key_raises(adata_with_types):
    """Should raise KeyError if type_key column doesn't exist."""
    with pytest.raises(KeyError, match="not found in adata.obs"):
        singlet.reprogramming_score(
            adata_with_types,
            source_type="source",
            target_type="target",
            type_key="nonexistent",
        )


def test_missing_use_rep_raises(adata_with_types):
    """Should raise KeyError if use_rep doesn't exist."""
    with pytest.raises(KeyError, match="not found in adata.obsm"):
        singlet.reprogramming_score(
            adata_with_types,
            source_type="source",
            target_type="target",
            use_rep="X_nonexistent",
        )


def test_invalid_source_type_raises(adata_with_types):
    """Should raise ValueError if source type not in data."""
    with pytest.raises(ValueError, match="Source type"):
        singlet.reprogramming_score(
            adata_with_types, source_type="neuron", target_type="target"
        )


def test_invalid_target_type_raises(adata_with_types):
    """Should raise ValueError if target type not in data."""
    with pytest.raises(ValueError, match="Target type"):
        singlet.reprogramming_score(
            adata_with_types, source_type="source", target_type="neuron"
        )


def test_score_symmetry(adata_with_types):
    """Swapping source/target should invert scores approximately."""
    singlet.reprogramming_score(
        adata_with_types, source_type="source", target_type="target"
    )
    forward_scores = adata_with_types.obs["reprogramming_score"].values.copy()

    singlet.reprogramming_score(
        adata_with_types, source_type="target", target_type="source"
    )
    reverse_scores = adata_with_types.obs["reprogramming_score"].values

    # Forward + reverse should approximately sum to 1
    np.testing.assert_allclose(forward_scores + reverse_scores, 1.0, atol=1e-10)
