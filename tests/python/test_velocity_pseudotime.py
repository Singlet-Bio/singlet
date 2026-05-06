"""Tests for singlet.velocity_pseudotime()."""

import numpy as np
import pytest
import singlet
from singlet._velocity_pseudotime import velocity_pseudotime


def _make_adata(n_cells=100, n_pcs=20):
    """Create AnnData with PCA embedding that has a clear trajectory."""
    import anndata as ad

    rng = np.random.default_rng(123)

    # Create a trajectory: cells lie along a 1D curve with some noise
    t = np.linspace(0, 1, n_cells)
    X_pca = np.zeros((n_cells, n_pcs), dtype=np.float64)
    # Primary trajectory direction in PC1
    X_pca[:, 0] = t * 10 + rng.normal(0, 0.3, n_cells)
    # Some noise in other PCs
    for pc in range(1, n_pcs):
        X_pca[:, pc] = rng.normal(0, 0.5, n_cells)

    # Create minimal AnnData
    X = rng.poisson(3, size=(n_cells, 50)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(50)]
    adata.obsm["X_pca"] = X_pca

    return adata, t


class TestVelocityPseudotime:
    def test_basic_output(self):
        """Should add velocity_pseudotime to obs."""
        adata, _ = _make_adata()
        result = velocity_pseudotime(adata)
        assert result is adata
        assert "velocity_pseudotime" in adata.obs.columns

    def test_normalized_range(self):
        """Pseudotime should be in [0, 1]."""
        adata, _ = _make_adata()
        velocity_pseudotime(adata)
        pt = adata.obs["velocity_pseudotime"].values
        assert pt.min() >= 0.0
        assert pt.max() <= 1.0

    def test_correlates_with_trajectory(self):
        """Pseudotime should correlate with ground truth trajectory."""
        adata, true_t = _make_adata(n_cells=150)
        velocity_pseudotime(adata, n_neighbors=20)
        pt = adata.obs["velocity_pseudotime"].values

        # Either positive or negative correlation is fine (direction may flip)
        corr = np.abs(np.corrcoef(true_t, pt)[0, 1])
        assert corr > 0.5, f"Expected |correlation| > 0.5, got {corr:.3f}"

    def test_root_key(self):
        """Should use root_key to select root cell."""
        adata, _ = _make_adata()
        # Mark last cell as root
        adata.obs["is_root"] = 0.0
        adata.obs.iloc[-1, adata.obs.columns.get_loc("is_root")] = 1.0
        velocity_pseudotime(adata, root_key="is_root")
        pt = adata.obs["velocity_pseudotime"].values
        # Root cell should have pseudotime 0
        assert pt[adata.n_obs - 1] == 0.0

    def test_custom_use_rep(self):
        """Should work with custom embedding key."""
        adata, _ = _make_adata()
        adata.obsm["X_custom"] = adata.obsm["X_pca"][:, :5]
        velocity_pseudotime(adata, use_rep="X_custom")
        assert "velocity_pseudotime" in adata.obs.columns

    def test_custom_n_neighbors(self):
        """Should work with different n_neighbors values."""
        adata, _ = _make_adata()
        velocity_pseudotime(adata, n_neighbors=10)
        assert "velocity_pseudotime" in adata.obs.columns

    def test_small_dataset(self):
        """Should work with very small dataset."""
        adata, _ = _make_adata(n_cells=20, n_pcs=5)
        velocity_pseudotime(adata, n_neighbors=5)
        pt = adata.obs["velocity_pseudotime"].values
        assert len(pt) == 20
        assert np.all(np.isfinite(pt))

    def test_missing_embedding_raises(self):
        """Should raise KeyError if embedding not found."""
        adata, _ = _make_adata()
        with pytest.raises(KeyError, match="X_missing"):
            velocity_pseudotime(adata, use_rep="X_missing")

    def test_missing_root_key_raises(self):
        """Should raise KeyError if root_key not in obs."""
        adata, _ = _make_adata()
        with pytest.raises(KeyError, match="nonexistent"):
            velocity_pseudotime(adata, root_key="nonexistent")

    def test_type_error(self):
        """Should raise TypeError for non-AnnData."""
        with pytest.raises(TypeError, match="velocity_pseudotime"):
            velocity_pseudotime("not_adata")

    def test_no_nans(self):
        """Output pseudotime should not contain NaN."""
        adata, _ = _make_adata()
        velocity_pseudotime(adata)
        pt = adata.obs["velocity_pseudotime"].values
        assert np.all(np.isfinite(pt))

    def test_public_api(self):
        """Function should be accessible from singlet namespace."""
        assert hasattr(singlet, "velocity_pseudotime")
        assert callable(singlet.velocity_pseudotime)

    def test_n_neighbors_clamped(self):
        """n_neighbors larger than n_cells should not crash."""
        adata, _ = _make_adata(n_cells=15, n_pcs=5)
        velocity_pseudotime(adata, n_neighbors=100)
        assert "velocity_pseudotime" in adata.obs.columns
