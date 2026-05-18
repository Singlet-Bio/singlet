# SPDX-License-Identifier: MIT
"""Tests for singlet.optimal_transport()."""

import numpy as np
import pytest

import singlet


@pytest.fixture
def adata_with_timepoints():
    """Create test AnnData with time point labels."""
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(42)
    n_cells = 120
    n_genes = 200

    # Create two time points with slightly different distributions
    X = sp.csr_matrix(rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32))
    adata = ad.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    # Assign time points
    time_labels = ["day0"] * 60 + ["day3"] * 60
    adata.obs["timepoint"] = time_labels

    # Add PCA representation (mock)
    pca_rep = rng.standard_normal((n_cells, 20))
    # Make day3 cells slightly shifted to simulate trajectory
    pca_rep[60:, 0] += 2.0
    adata.obsm["X_pca"] = pca_rep

    return adata


class TestOptimalTransport:
    """Tests for optimal_transport function."""

    def test_basic_coupling(self, adata_with_timepoints):
        """Test basic OT coupling computation."""
        adata = adata_with_timepoints
        coupling = singlet.optimal_transport(
            adata, time_key="timepoint", source_time="day0", target_time="day3"
        )

        # Should return ndarray
        assert isinstance(coupling, np.ndarray)
        # Shape should be (n_source, n_target)
        assert coupling.shape == (60, 60)
        # Coupling should be non-negative
        assert np.all(coupling >= 0)
        # Rows should approximately sum to 1/n_source (uniform marginal)
        row_sums = coupling.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0 / 60, rtol=0.1)

    def test_stored_in_uns(self, adata_with_timepoints):
        """Test coupling stored in adata.uns."""
        adata = adata_with_timepoints
        coupling = singlet.optimal_transport(
            adata, time_key="timepoint", source_time="day0", target_time="day3"
        )

        key = "ot_coupling_day0_day3"
        assert key in adata.uns
        np.testing.assert_array_equal(adata.uns[key], coupling)

    def test_epsilon_effect(self, adata_with_timepoints):
        """Test that epsilon controls sparsity of coupling."""
        adata = adata_with_timepoints

        # Small epsilon → sparser coupling
        coupling_sparse = singlet.optimal_transport(
            adata,
            time_key="timepoint",
            source_time="day0",
            target_time="day3",
            epsilon=0.01,
        )

        # Large epsilon → more uniform coupling
        coupling_dense = singlet.optimal_transport(
            adata,
            time_key="timepoint",
            source_time="day0",
            target_time="day3",
            epsilon=1.0,
        )

        # Dense coupling should have smaller max entry (more spread out)
        assert coupling_dense.max() < coupling_sparse.max()

    def test_max_iter(self, adata_with_timepoints):
        """Test max_iter parameter."""
        adata = adata_with_timepoints
        coupling = singlet.optimal_transport(
            adata,
            time_key="timepoint",
            source_time="day0",
            target_time="day3",
            max_iter=5,
        )
        # Should still produce valid output
        assert coupling.shape == (60, 60)
        assert np.all(coupling >= 0)

    def test_numeric_time_values(self):
        """Test with numeric time point values."""
        import anndata as ad
        import scipy.sparse as sp

        rng = np.random.default_rng(123)
        n_cells = 80
        X = sp.csr_matrix(rng.poisson(1, size=(n_cells, 100)).astype(np.float32))
        adata = ad.AnnData(X=X)
        adata.obs["day"] = [0] * 40 + [3] * 40
        adata.obsm["X_pca"] = rng.standard_normal((n_cells, 10))

        coupling = singlet.optimal_transport(
            adata, time_key="day", source_time=0, target_time=3
        )
        assert coupling.shape == (40, 40)
        assert "ot_coupling_0_3" in adata.uns

    def test_invalid_time_key(self, adata_with_timepoints):
        """Test error on missing time_key."""
        with pytest.raises(KeyError, match="not found in adata.obs"):
            singlet.optimal_transport(
                adata_with_timepoints,
                time_key="nonexistent",
                source_time="day0",
                target_time="day3",
            )

    def test_invalid_time_value(self, adata_with_timepoints):
        """Test error when no cells match time value."""
        with pytest.raises(ValueError, match="No cells found"):
            singlet.optimal_transport(
                adata_with_timepoints,
                time_key="timepoint",
                source_time="day99",
                target_time="day3",
            )

    def test_missing_representation(self, adata_with_timepoints):
        """Test error on missing representation."""
        with pytest.raises(KeyError, match="not found in adata.obsm"):
            singlet.optimal_transport(
                adata_with_timepoints,
                time_key="timepoint",
                source_time="day0",
                target_time="day3",
                use_rep="X_nonexistent",
            )

    def test_invalid_epsilon(self, adata_with_timepoints):
        """Test error on non-positive epsilon."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            singlet.optimal_transport(
                adata_with_timepoints,
                time_key="timepoint",
                source_time="day0",
                target_time="day3",
                epsilon=-0.1,
            )

    def test_invalid_max_iter(self, adata_with_timepoints):
        """Test error on invalid max_iter."""
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            singlet.optimal_transport(
                adata_with_timepoints,
                time_key="timepoint",
                source_time="day0",
                target_time="day3",
                max_iter=0,
            )

    def test_asymmetric_groups(self):
        """Test with different sized source and target groups."""
        import anndata as ad
        import scipy.sparse as sp

        rng = np.random.default_rng(99)
        n_cells = 100
        X = sp.csr_matrix(rng.poisson(1, size=(n_cells, 50)).astype(np.float32))
        adata = ad.AnnData(X=X)
        adata.obs["time"] = ["early"] * 30 + ["late"] * 70
        adata.obsm["X_pca"] = rng.standard_normal((n_cells, 10))

        coupling = singlet.optimal_transport(
            adata, time_key="time", source_time="early", target_time="late"
        )
        assert coupling.shape == (30, 70)

    def test_column_sums(self, adata_with_timepoints):
        """Test that column sums equal target marginal."""
        adata = adata_with_timepoints
        coupling = singlet.optimal_transport(
            adata, time_key="timepoint", source_time="day0", target_time="day3"
        )
        # Columns should approximately sum to 1/n_target
        col_sums = coupling.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0 / 60, rtol=0.1)
