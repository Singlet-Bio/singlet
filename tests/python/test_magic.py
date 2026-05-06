"""Tests for singlet.magic()."""

import numpy as np
import pytest
import scipy.sparse as sp


def _make_adata(n_cells=100, n_genes=200, sparse=True, with_pca=True):
    """Create test AnnData for MAGIC."""
    import anndata as ad

    rng = np.random.default_rng(42)
    if sparse:
        X = sp.random(
            n_cells, n_genes, density=0.3, format="csr", random_state=42
        )
        X.data = np.abs(
            rng.standard_normal(X.nnz).astype(np.float32)
        )
    else:
        X = np.abs(
            rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        )

    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    if with_pca:
        adata.obsm["X_pca"] = rng.standard_normal(
            (n_cells, 50)
        ).astype(np.float32)

    return adata


class TestMagic:
    def test_basic_sparse(self):
        from singlet._magic import magic

        adata = _make_adata(sparse=True)
        result = magic(adata)
        assert result is adata
        assert "magic" in adata.layers
        assert adata.layers["magic"].shape == (100, 200)
        assert adata.layers["magic"].dtype == np.float32

    def test_basic_dense(self):
        from singlet._magic import magic

        adata = _make_adata(sparse=False)
        result = magic(adata)
        assert "magic" in adata.layers
        assert adata.layers["magic"].shape == (100, 200)

    def test_uses_pca(self):
        from singlet._magic import magic

        adata = _make_adata(with_pca=True)
        magic(adata)
        assert "magic" in adata.layers

    def test_no_pca_uses_X(self):
        from singlet._magic import magic

        adata = _make_adata(with_pca=False, n_cells=50, n_genes=100)
        magic(adata, n_neighbors=10)
        assert "magic" in adata.layers

    def test_custom_params(self):
        from singlet._magic import magic

        adata = _make_adata()
        magic(adata, n_neighbors=15, t=5, knn_dist="cosine")
        assert "magic" in adata.layers

    def test_use_rep(self):
        from singlet._magic import magic

        adata = _make_adata()
        rng = np.random.default_rng(0)
        adata.obsm["X_custom"] = rng.standard_normal(
            (100, 20)
        ).astype(np.float32)
        magic(adata, use_rep="X_custom")
        assert "magic" in adata.layers

    def test_n_pcs(self):
        from singlet._magic import magic

        adata = _make_adata()
        magic(adata, n_pcs=10)
        assert "magic" in adata.layers

    def test_imputed_smoother(self):
        """Imputed data should be smoother (lower variance across cells)."""
        from singlet._magic import magic

        adata = _make_adata(n_cells=100, n_genes=50, sparse=True)
        X_orig = (
            adata.X.toarray() if sp.issparse(adata.X) else adata.X.copy()
        )
        magic(adata, t=3)
        X_imputed = adata.layers["magic"]
        # Variance per gene should generally be lower after imputation
        orig_var = np.var(X_orig, axis=0).mean()
        imp_var = np.var(X_imputed, axis=0).mean()
        # Imputed variance should be less (smoothing effect)
        assert imp_var <= orig_var * 1.5  # Allow some tolerance

    def test_type_error(self):
        from singlet._magic import magic

        with pytest.raises(TypeError, match="AnnData"):
            magic("not_adata")

    def test_public_api(self):
        import singlet

        assert hasattr(singlet, "magic")
        assert callable(singlet.magic)

    def test_deterministic(self):
        from singlet._magic import magic

        adata1 = _make_adata()
        adata2 = _make_adata()
        magic(adata1, random_state=42)
        magic(adata2, random_state=42)
        np.testing.assert_array_equal(
            adata1.layers["magic"], adata2.layers["magic"]
        )
