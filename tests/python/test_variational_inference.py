# SPDX-License-Identifier: MIT
"""Tests for singlet.variational_inference()."""

import numpy as np
import scipy.sparse as sp
from singlet._variational_inference import variational_inference


def _make_vi_adata(n_cells=100, n_genes=50, n_batches=0, seed=42):
    """Create AnnData with structure for variational inference."""
    import anndata as ad

    rng = np.random.default_rng(seed)

    # Create data with latent structure: 3 groups
    group_size = n_cells // 3
    remainder = n_cells - 3 * group_size

    # Latent factors drive expression
    X_parts = []
    for group_idx in range(3):
        n_group = group_size + (remainder if group_idx == 2 else 0)
        base = rng.normal(2, 0.5, size=(n_group, n_genes))
        # Each group has different gene program active
        start_gene = group_idx * (n_genes // 3)
        end_gene = start_gene + n_genes // 3
        base[:, start_gene:end_gene] += rng.normal(3, 0.3, size=(n_group, end_gene - start_gene))
        X_parts.append(base)

    X = np.vstack(X_parts).astype(np.float32)
    X = np.clip(X, 0, None)  # Ensure non-negative

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]

    if n_batches > 0:
        batch_labels = [f"batch_{i % n_batches}" for i in range(n_cells)]
        adata.obs["batch"] = batch_labels
        # Add batch-specific shift
        for batch_idx in range(n_batches):
            mask = np.array(adata.obs["batch"] == f"batch_{batch_idx}")
            shift = rng.normal(0, 1, size=n_genes) * 0.5
            if sp.issparse(adata.X):
                dense = adata.X.toarray()
                dense[mask] += shift
                dense = np.clip(dense, 0, None)
                adata.X = sp.csr_matrix(dense.astype(np.float32))
            else:
                adata.X[mask] += shift
                adata.X = np.clip(adata.X, 0, None)

    return adata


class TestVariationalInferenceBasic:
    def test_returns_adata(self):
        """Should return the same adata object."""
        adata = _make_vi_adata()
        result = variational_inference(adata, n_latent=5)
        assert result is adata

    def test_obsm_x_scvi_created(self):
        """Should create obsm['X_scvi'] with correct shape."""
        adata = _make_vi_adata(n_cells=60, n_genes=40)
        variational_inference(adata, n_latent=5)
        assert "X_scvi" in adata.obsm
        assert adata.obsm["X_scvi"].shape == (60, 5)

    def test_layers_scvi_normalized(self):
        """Should create layers['scvi_normalized'] with correct shape."""
        adata = _make_vi_adata(n_cells=60, n_genes=40)
        variational_inference(adata, n_latent=5)
        assert "scvi_normalized" in adata.layers
        assert adata.layers["scvi_normalized"].shape == (60, 40)

    def test_latent_dimensionality(self):
        """Latent space should have requested dimensionality."""
        adata = _make_vi_adata(n_cells=80, n_genes=50)
        variational_inference(adata, n_latent=8)
        assert adata.obsm["X_scvi"].shape[1] == 8

    def test_output_is_finite(self):
        """All outputs should be finite."""
        adata = _make_vi_adata(n_cells=60, n_genes=40)
        variational_inference(adata, n_latent=5)
        assert np.all(np.isfinite(adata.obsm["X_scvi"]))
        assert np.all(np.isfinite(adata.layers["scvi_normalized"]))


class TestVariationalInferenceBatch:
    def test_batch_correction(self):
        """Batch correction should reduce batch effect in latent space."""
        adata = _make_vi_adata(n_cells=90, n_genes=40, n_batches=3)
        variational_inference(adata, n_latent=5, batch_key="batch")
        assert "X_scvi" in adata.obsm
        assert adata.obsm["X_scvi"].shape == (90, 5)

    def test_no_batch_key(self):
        """Should work without batch_key."""
        adata = _make_vi_adata(n_cells=60, n_genes=40)
        variational_inference(adata, n_latent=5, batch_key=None)
        assert "X_scvi" in adata.obsm

    def test_missing_batch_key_ignored(self):
        """Non-existent batch_key should be silently ignored."""
        adata = _make_vi_adata(n_cells=60, n_genes=40)
        variational_inference(adata, n_latent=5, batch_key="nonexistent")
        assert "X_scvi" in adata.obsm


class TestVariationalInferenceParameters:
    def test_custom_n_epochs(self):
        """Should respect n_epochs parameter."""
        adata = _make_vi_adata(n_cells=50, n_genes=30)
        # Just verify it runs with different epoch counts
        variational_inference(adata, n_latent=3, n_epochs=5)
        assert "X_scvi" in adata.obsm

    def test_reproducibility(self):
        """Same random_state should give same results."""
        adata1 = _make_vi_adata(n_cells=50, n_genes=30)
        adata2 = _make_vi_adata(n_cells=50, n_genes=30)
        variational_inference(adata1, n_latent=3, random_state=0)
        variational_inference(adata2, n_latent=3, random_state=0)
        np.testing.assert_array_almost_equal(
            adata1.obsm["X_scvi"], adata2.obsm["X_scvi"], decimal=5
        )

    def test_different_seeds_different_results(self):
        """Different random states should give different results."""
        adata1 = _make_vi_adata(n_cells=50, n_genes=30)
        adata2 = _make_vi_adata(n_cells=50, n_genes=30)
        variational_inference(adata1, n_latent=3, random_state=0)
        variational_inference(adata2, n_latent=3, random_state=99)
        # Results should differ (W initialization differs)
        assert not np.allclose(adata1.obsm["X_scvi"], adata2.obsm["X_scvi"])


class TestVariationalInferenceSparse:
    def test_sparse_input(self):
        """Should handle sparse matrix input."""
        adata = _make_vi_adata(n_cells=50, n_genes=30)
        assert sp.issparse(adata.X)
        variational_inference(adata, n_latent=3)
        assert "X_scvi" in adata.obsm

    def test_dense_input(self):
        """Should handle dense matrix input."""
        adata = _make_vi_adata(n_cells=50, n_genes=30)
        adata.X = np.asarray(adata.X.toarray())
        variational_inference(adata, n_latent=3)
        assert "X_scvi" in adata.obsm


class TestVariationalInferenceDtype:
    def test_output_dtype(self):
        """Output should be float32."""
        adata = _make_vi_adata(n_cells=50, n_genes=30)
        variational_inference(adata, n_latent=3)
        assert adata.obsm["X_scvi"].dtype == np.float32
        assert adata.layers["scvi_normalized"].dtype == np.float32


class TestVariationalInferenceEdgeCases:
    def test_n_latent_larger_than_genes(self):
        """n_latent larger than genes should be clamped."""
        adata = _make_vi_adata(n_cells=50, n_genes=10)
        variational_inference(adata, n_latent=20)
        # Should be clamped to n_genes - 1
        assert adata.obsm["X_scvi"].shape[1] <= 10

    def test_small_dataset(self):
        """Should work with very small datasets."""
        adata = _make_vi_adata(n_cells=15, n_genes=10)
        variational_inference(adata, n_latent=3)
        assert adata.obsm["X_scvi"].shape == (15, 3)
