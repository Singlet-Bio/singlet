"""Tests for singlet.palantir_pseudotime."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import singlet


@pytest.fixture
def adata_with_pca():
    """Create test AnnData with PCA embedding."""
    rng = np.random.default_rng(42)
    n_cells = 150
    n_genes = 100

    # Simulate a trajectory (cells along a path)
    t_vals = np.linspace(0, 1, n_cells)
    X = rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32)
    # Add a gradient along the trajectory
    for gene_idx in range(10):
        X[:, gene_idx] += (t_vals * 10).astype(np.float32)

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = AnnData(X=X, obs=obs, var=var)

    # Add PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=20, random_state=42)
    adata.obsm["X_pca"] = pca.fit_transform(X)

    return adata


@pytest.fixture
def adata_named_cells():
    """Create test AnnData with named cells."""
    rng = np.random.default_rng(7)
    n_cells = 100
    n_genes = 50

    X = rng.poisson(3, size=(n_cells, n_genes)).astype(np.float32)
    obs_names = [f"barcode_{i:04d}" for i in range(n_cells)]
    obs = pd.DataFrame(index=obs_names)
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = AnnData(X=X, obs=obs, var=var)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=15, random_state=0)
    adata.obsm["X_pca"] = pca.fit_transform(X)
    return adata


class TestPalantirPseudotime:
    def test_basic(self, adata_with_pca):
        """Test basic functionality with integer root."""
        result = singlet.palantir_pseudotime(adata_with_pca, root_cell=0)
        assert result is adata_with_pca
        assert "palantir_pseudotime" in adata_with_pca.obs.columns
        assert "palantir_waypoint_distances" in adata_with_pca.obsm

    def test_pseudotime_range(self, adata_with_pca):
        """Pseudotime should be in [0, 1]."""
        singlet.palantir_pseudotime(adata_with_pca, root_cell=0)
        pt = adata_with_pca.obs["palantir_pseudotime"].values
        assert pt.min() >= 0.0
        assert pt.max() <= 1.0

    def test_root_has_zero_pseudotime(self, adata_with_pca):
        """Root cell should have pseudotime 0."""
        singlet.palantir_pseudotime(adata_with_pca, root_cell=0)
        pt = adata_with_pca.obs["palantir_pseudotime"].values
        assert pt[0] == 0.0

    def test_string_root_cell(self, adata_named_cells):
        """Test with string root cell name."""
        root_name = "barcode_0000"
        singlet.palantir_pseudotime(adata_named_cells, root_cell=root_name)
        pt = adata_named_cells.obs["palantir_pseudotime"].values
        assert pt[0] == 0.0

    def test_waypoint_distances_shape(self, adata_with_pca):
        """Waypoint distances should have correct shape."""
        n_wp = 50
        singlet.palantir_pseudotime(adata_with_pca, root_cell=0, n_waypoints=n_wp)
        wp_dists = adata_with_pca.obsm["palantir_waypoint_distances"]
        assert wp_dists.shape == (adata_with_pca.n_obs, n_wp)

    def test_params_stored(self, adata_with_pca):
        """Parameters should be stored in uns."""
        singlet.palantir_pseudotime(adata_with_pca, root_cell=5, n_neighbors=20)
        params = adata_with_pca.uns["palantir_params"]
        assert params["root_cell"] == 5
        assert params["n_neighbors"] == 20

    def test_copy_mode(self, adata_with_pca):
        """Test copy mode doesn't modify original."""
        result = singlet.palantir_pseudotime(
            adata_with_pca, root_cell=0, copy=True
        )
        assert result is not adata_with_pca
        assert "palantir_pseudotime" in result.obs.columns
        assert "palantir_pseudotime" not in adata_with_pca.obs.columns

    def test_invalid_root_index(self, adata_with_pca):
        """Test error on out-of-bounds root index."""
        with pytest.raises(IndexError, match="out of bounds"):
            singlet.palantir_pseudotime(adata_with_pca, root_cell=9999)

    def test_invalid_root_name(self, adata_named_cells):
        """Test error on unknown root name."""
        with pytest.raises(KeyError, match="not found"):
            singlet.palantir_pseudotime(adata_named_cells, root_cell="nonexistent")

    def test_missing_representation(self, adata_with_pca):
        """Test error on missing representation."""
        with pytest.raises(KeyError, match="not found"):
            singlet.palantir_pseudotime(
                adata_with_pca, root_cell=0, use_rep="X_nonexistent"
            )

    def test_custom_n_components(self, adata_with_pca):
        """Test with different n_components."""
        singlet.palantir_pseudotime(adata_with_pca, root_cell=0, n_components=5)
        assert "palantir_pseudotime" in adata_with_pca.obs.columns

    def test_small_n_waypoints(self, adata_with_pca):
        """Test with small number of waypoints."""
        singlet.palantir_pseudotime(adata_with_pca, root_cell=0, n_waypoints=10)
        wp = adata_with_pca.obsm["palantir_waypoint_distances"]
        assert wp.shape[1] == 10

    def test_n_waypoints_exceeds_cells(self, adata_with_pca):
        """Waypoints capped at n_cells."""
        singlet.palantir_pseudotime(adata_with_pca, root_cell=0, n_waypoints=99999)
        wp = adata_with_pca.obsm["palantir_waypoint_distances"]
        assert wp.shape[1] == adata_with_pca.n_obs

    def test_different_root_gives_different_pt(self, adata_with_pca):
        """Different root cells should give different pseudotime."""
        adata1 = adata_with_pca.copy()
        adata2 = adata_with_pca.copy()
        singlet.palantir_pseudotime(adata1, root_cell=0)
        singlet.palantir_pseudotime(adata2, root_cell=50)
        pt1 = adata1.obs["palantir_pseudotime"].values
        pt2 = adata2.obs["palantir_pseudotime"].values
        # Should not be identical
        assert not np.allclose(pt1, pt2)

    def test_no_nan_in_pseudotime(self, adata_with_pca):
        """Pseudotime should have no NaN values."""
        singlet.palantir_pseudotime(adata_with_pca, root_cell=0)
        pt = adata_with_pca.obs["palantir_pseudotime"].values
        assert not np.any(np.isnan(pt))
