"""Tests for singlet.plot_embedding()."""

import numpy as np
import pytest
import scipy.sparse as sp


def _make_adata(n_cells=100, n_genes=50, with_umap=True, with_obs=True):
    """Create test AnnData for embedding plot."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(42)
    X = sp.random(
        n_cells, n_genes, density=0.3, format="csr", random_state=42
    )
    X.data = np.abs(rng.standard_normal(X.nnz).astype(np.float32))

    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    if with_umap:
        adata.obsm["X_umap"] = rng.standard_normal(
            (n_cells, 2)
        ).astype(np.float32)

    if with_obs:
        adata.obs["cluster"] = pd.Categorical(
            rng.choice(["A", "B", "C"], n_cells)
        )
        adata.obs["n_counts"] = rng.uniform(1000, 5000, n_cells)

    return adata


class TestPlotEmbedding:
    def test_basic_no_color(self):
        import matplotlib

        matplotlib.use("Agg")
        from singlet._plot_embedding import plot_embedding

        adata = _make_adata()
        fig, axes = plot_embedding(adata)
        assert fig is not None

    def test_single_categorical(self):
        import matplotlib

        matplotlib.use("Agg")
        from singlet._plot_embedding import plot_embedding

        adata = _make_adata()
        fig, axes = plot_embedding(adata, color="cluster")
        assert fig is not None

    def test_single_continuous(self):
        import matplotlib

        matplotlib.use("Agg")
        from singlet._plot_embedding import plot_embedding

        adata = _make_adata()
        fig, axes = plot_embedding(adata, color="n_counts")
        assert fig is not None

    def test_gene_expression(self):
        import matplotlib

        matplotlib.use("Agg")
        from singlet._plot_embedding import plot_embedding

        adata = _make_adata()
        fig, axes = plot_embedding(adata, color="GENE0")
        assert fig is not None

    def test_multiple_colors(self):
        import matplotlib

        matplotlib.use("Agg")
        from singlet._plot_embedding import plot_embedding

        adata = _make_adata()
        fig, axes = plot_embedding(
            adata, color=["cluster", "n_counts", "GENE0"]
        )
        assert fig is not None

    def test_custom_basis(self):
        import matplotlib

        matplotlib.use("Agg")
        from singlet._plot_embedding import plot_embedding

        adata = _make_adata()
        rng = np.random.default_rng(0)
        adata.obsm["X_tsne"] = rng.standard_normal(
            (100, 2)
        ).astype(np.float32)
        fig, axes = plot_embedding(
            adata, basis="X_tsne", color="cluster"
        )
        assert fig is not None

    def test_missing_basis_raises(self):
        import matplotlib

        matplotlib.use("Agg")
        from singlet._plot_embedding import plot_embedding

        adata = _make_adata(with_umap=False)
        with pytest.raises(ValueError, match="X_umap"):
            plot_embedding(adata)

    def test_invalid_color_raises(self):
        import matplotlib

        matplotlib.use("Agg")
        from singlet._plot_embedding import plot_embedding

        adata = _make_adata()
        with pytest.raises(ValueError, match="not_a_key"):
            plot_embedding(adata, color="not_a_key")

    def test_frameon_false(self):
        import matplotlib

        matplotlib.use("Agg")
        from singlet._plot_embedding import plot_embedding

        adata = _make_adata()
        fig, axes = plot_embedding(adata, frameon=False)
        assert fig is not None

    def test_ncols(self):
        import matplotlib

        matplotlib.use("Agg")
        from singlet._plot_embedding import plot_embedding

        adata = _make_adata()
        fig, axes = plot_embedding(
            adata,
            color=[
                "cluster", "n_counts", "GENE0",
                "GENE1", "GENE2",
            ],
            ncols=2,
        )
        assert fig is not None

    def test_public_api(self):
        import singlet

        assert hasattr(singlet, "plot_embedding")
        assert callable(singlet.plot_embedding)

    def test_type_error(self):
        import matplotlib

        matplotlib.use("Agg")
        from singlet._plot_embedding import plot_embedding

        with pytest.raises((TypeError, ValueError, AttributeError)):
            plot_embedding("not_adata")
