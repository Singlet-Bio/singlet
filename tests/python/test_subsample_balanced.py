# SPDX-License-Identifier: MIT
"""Tests for singlet.subsample_balanced()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=200, m=100, seed=42, n_groups=4):
    """Create test AnnData with unequal group sizes."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(2, size=(n, m)).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs_names = [f"cell_{i}" for i in range(n)]
    # Create unequal groups: sizes 80, 60, 40, 20
    sizes = [80, 60, 40, 20]
    labels = []
    for idx, size in enumerate(sizes[:n_groups]):
        labels.extend([f"group_{idx}"] * size)
    # Fill remaining if needed
    while len(labels) < n:
        labels.append(f"group_{n_groups - 1}")
    adata.obs["cluster"] = labels[:n]
    adata.obs["cluster"] = adata.obs["cluster"].astype("category")
    # Add a second grouping key for stratified tests
    adata.obs["batch"] = ["batch_0" if i < n // 2 else "batch_1" for i in range(n)]
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    return adata


def test_subsample_balanced_basic():
    """Basic balanced subsampling by n_per_group."""
    adata = _make_adata()
    result = singlet.subsample_balanced(adata, groupby="cluster", n_per_group=20)
    assert result is not adata  # copy
    # Each group should have at most 20 cells
    counts = result.obs["cluster"].value_counts()
    assert counts.max() <= 20


def test_subsample_balanced_respects_n():
    """Each group should have exactly n_per_group if large enough."""
    adata = _make_adata()
    result = singlet.subsample_balanced(adata, groupby="cluster", n_per_group=15)
    counts = result.obs["cluster"].value_counts()
    # All groups have >= 20 cells, so all should have exactly 15
    assert all(c == 15 for c in counts.values)


def test_subsample_balanced_no_upsample():
    """Groups smaller than n_per_group keep all cells."""
    adata = _make_adata()  # group_3 has 20 cells
    result = singlet.subsample_balanced(adata, groupby="cluster", n_per_group=50)
    counts = result.obs["cluster"].value_counts()
    # group_3 has only 20, should keep all 20
    assert counts["group_3"] == 20
    # group_0 has 80, should have 50
    assert counts["group_0"] == 50


def test_subsample_balanced_frac():
    """Fractional subsampling."""
    adata = _make_adata()
    result = singlet.subsample_balanced(adata, groupby="cluster", frac=0.5)
    # group_0: 80 -> 40, group_1: 60 -> 30, group_2: 40 -> 20, group_3: 20 -> 10
    counts = result.obs["cluster"].value_counts()
    assert counts["group_0"] == 40
    assert counts["group_1"] == 30
    assert counts["group_2"] == 20
    assert counts["group_3"] == 10


def test_subsample_balanced_frac_one():
    """frac=1.0 should keep all cells."""
    adata = _make_adata()
    result = singlet.subsample_balanced(adata, groupby="cluster", frac=1.0)
    assert result.shape[0] == adata.shape[0]


def test_subsample_balanced_reproducible():
    """Same random_state gives same result."""
    adata = _make_adata()
    r1 = singlet.subsample_balanced(adata, groupby="cluster", n_per_group=10, random_state=42)
    r2 = singlet.subsample_balanced(adata, groupby="cluster", n_per_group=10, random_state=42)
    assert list(r1.obs_names) == list(r2.obs_names)


def test_subsample_balanced_different_seeds():
    """Different random_state gives different result."""
    adata = _make_adata()
    r1 = singlet.subsample_balanced(adata, groupby="cluster", n_per_group=10, random_state=0)
    r2 = singlet.subsample_balanced(adata, groupby="cluster", n_per_group=10, random_state=99)
    # Likely different (not guaranteed but highly probable)
    assert list(r1.obs_names) != list(r2.obs_names)


def test_subsample_balanced_multi_key():
    """Stratified subsampling by multiple keys."""
    adata = _make_adata()
    result = singlet.subsample_balanced(
        adata, groupby=["cluster", "batch"], n_per_group=5
    )
    # Should have subsampled from each (cluster, batch) combination
    assert result.shape[0] <= adata.shape[0]
    assert result.shape[0] > 0


def test_subsample_balanced_inplace():
    """copy=False should modify in place."""
    adata = _make_adata()
    original_n = adata.shape[0]
    result = singlet.subsample_balanced(
        adata, groupby="cluster", n_per_group=10, copy=False
    )
    assert result is None
    assert adata.shape[0] < original_n


def test_subsample_balanced_invalid_both():
    """Should raise when both n_per_group and frac specified."""
    adata = _make_adata()
    with pytest.raises(ValueError, match="Specify either"):
        singlet.subsample_balanced(adata, groupby="cluster", n_per_group=10, frac=0.5)


def test_subsample_balanced_invalid_neither():
    """Should raise when neither n_per_group nor frac specified."""
    adata = _make_adata()
    with pytest.raises(ValueError, match="Must specify"):
        singlet.subsample_balanced(adata, groupby="cluster")


def test_subsample_balanced_invalid_frac():
    """Should raise for invalid frac values."""
    adata = _make_adata()
    with pytest.raises(ValueError, match="frac must be"):
        singlet.subsample_balanced(adata, groupby="cluster", frac=0.0)
    with pytest.raises(ValueError, match="frac must be"):
        singlet.subsample_balanced(adata, groupby="cluster", frac=1.5)


def test_subsample_balanced_invalid_groupby():
    """Should raise KeyError for missing groupby key."""
    adata = _make_adata()
    with pytest.raises(KeyError, match="nonexistent"):
        singlet.subsample_balanced(adata, groupby="nonexistent", n_per_group=10)


def test_subsample_balanced_invalid_n():
    """Should raise for n_per_group < 1."""
    adata = _make_adata()
    with pytest.raises(ValueError, match="n_per_group must be"):
        singlet.subsample_balanced(adata, groupby="cluster", n_per_group=0)


def test_subsample_balanced_preserves_data():
    """Subsampled cells should have correct expression values."""
    adata = _make_adata()
    result = singlet.subsample_balanced(adata, groupby="cluster", n_per_group=10)
    # Check that values match original
    for cell_name in result.obs_names[:5]:
        orig_idx = list(adata.obs_names).index(cell_name)
        np.testing.assert_array_equal(
            np.asarray(result[cell_name].X), np.asarray(adata[orig_idx].X)
        )


def test_subsample_balanced_small_dataset():
    """Works with very small datasets."""
    rng = np.random.default_rng(10)
    X = rng.poisson(2, size=(10, 5)).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = [f"g{i}" for i in range(5)]
    adata.obs_names = [f"c{i}" for i in range(10)]
    adata.obs["group"] = ["A"] * 5 + ["B"] * 5
    adata.obs["group"] = adata.obs["group"].astype("category")

    result = singlet.subsample_balanced(adata, groupby="group", n_per_group=3)
    counts = result.obs["group"].value_counts()
    assert counts["A"] == 3
    assert counts["B"] == 3
