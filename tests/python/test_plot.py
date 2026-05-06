"""Tests for singlet.plot_umap() and singlet.plot_violin()."""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._plot import plot_dotplot, plot_umap, plot_violin


def _make_plot_adata(n_cells=100, n_genes=50):
    """Create AnnData with UMAP coordinates and clusters for plotting."""
    import anndata as ad

    rng = np.random.default_rng(42)
    X = sp.random(n_cells, n_genes, density=0.3, format="csr", random_state=42)
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]

    # UMAP coordinates
    adata.obsm["X_umap"] = rng.standard_normal((n_cells, 2)).astype(np.float32)

    # Categorical obs
    labels = [f"cluster_{i % 5}" for i in range(n_cells)]
    adata.obs["leiden"] = pd.Categorical(labels)

    # Continuous obs
    adata.obs["n_genes"] = rng.integers(100, 5000, size=n_cells)
    adata.obs["total_counts"] = rng.integers(500, 20000, size=n_cells).astype(float)

    return adata


class TestPlotUmap:
    def test_basic_no_color(self):
        """Should plot without color (gray points)."""
        adata = _make_plot_adata()
        fig = plot_umap(adata, show=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_categorical_color(self):
        """Should plot with categorical obs column."""
        adata = _make_plot_adata()
        fig = plot_umap(adata, color="leiden", show=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_continuous_color(self):
        """Should plot with continuous obs column."""
        adata = _make_plot_adata()
        fig = plot_umap(adata, color="n_genes", show=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_gene_color(self):
        """Should plot colored by gene expression."""
        adata = _make_plot_adata()
        fig = plot_umap(adata, color="GENE0", show=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_gene_from_layer(self):
        """Should use specified layer for gene expression."""
        adata = _make_plot_adata()
        adata.layers["raw"] = adata.X.copy()
        fig = plot_umap(adata, color="GENE0", layer="raw", show=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_title(self):
        """Should use custom title."""
        adata = _make_plot_adata()
        fig = plot_umap(adata, color="leiden", title="My Title", show=False)
        ax = fig.axes[0]
        assert ax.get_title() == "My Title"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_save(self, tmp_path):
        """Should save figure to file."""
        adata = _make_plot_adata()
        path = str(tmp_path / "test_umap.png")
        plot_umap(adata, color="leiden", save=path, show=False)
        import os

        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_missing_umap_raises(self):
        """Should raise KeyError when X_umap is missing."""
        import anndata as ad

        adata = ad.AnnData(X=sp.random(50, 30, format="csr"))
        with pytest.raises(KeyError, match="X_umap"):
            plot_umap(adata, show=False)

    def test_missing_color_raises(self):
        """Should raise KeyError for invalid color key."""
        adata = _make_plot_adata()
        with pytest.raises(KeyError, match="nonexistent"):
            plot_umap(adata, color="nonexistent", show=False)

    def test_type_error(self):
        """Should raise TypeError for non-AnnData."""
        with pytest.raises(TypeError, match="plot_umap"):
            plot_umap("not_adata", show=False)

    def test_custom_ax(self):
        """Should plot on provided axes."""
        import matplotlib.pyplot as plt

        adata = _make_plot_adata()
        fig, ax = plt.subplots()
        result = plot_umap(adata, color="leiden", ax=ax, show=False)
        assert result is not None
        plt.close(fig)

    def test_public_api(self):
        assert hasattr(singlet, "plot_umap")
        assert callable(singlet.plot_umap)


class TestPlotViolin:
    def test_single_key(self):
        """Should plot single obs column."""
        adata = _make_plot_adata()
        fig = plot_violin(adata, "n_genes", show=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_multiple_keys(self):
        """Should plot multiple keys as subplots."""
        adata = _make_plot_adata()
        fig = plot_violin(adata, ["n_genes", "total_counts"], show=False)
        assert fig is not None
        assert len(fig.axes) == 2
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_groupby(self):
        """Should group violins by obs column."""
        adata = _make_plot_adata()
        fig = plot_violin(adata, "n_genes", groupby="leiden", show=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_gene_key(self):
        """Should plot gene expression as violin."""
        adata = _make_plot_adata()
        fig = plot_violin(adata, "GENE0", show=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_log_transform(self):
        """Should apply log1p when log=True."""
        adata = _make_plot_adata()
        fig = plot_violin(adata, "total_counts", log=True, show=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_save(self, tmp_path):
        """Should save figure to file."""
        adata = _make_plot_adata()
        path = str(tmp_path / "test_violin.png")
        plot_violin(adata, "n_genes", save=path, show=False)
        import os

        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_missing_key_raises(self):
        """Should raise KeyError for missing key."""
        adata = _make_plot_adata()
        with pytest.raises(KeyError, match="nonexistent"):
            plot_violin(adata, "nonexistent", show=False)

    def test_missing_groupby_raises(self):
        """Should raise KeyError for missing groupby."""
        adata = _make_plot_adata()
        with pytest.raises(KeyError, match="bad_col"):
            plot_violin(adata, "n_genes", groupby="bad_col", show=False)

    def test_type_error(self):
        """Should raise TypeError for non-AnnData."""
        with pytest.raises(TypeError, match="plot_violin"):
            plot_violin("not_adata", "n_genes", show=False)

    def test_public_api(self):
        assert hasattr(singlet, "plot_violin")
        assert callable(singlet.plot_violin)


class TestPlotDotplot:
    def test_basic(self):
        """Should plot dot plot for given genes."""
        adata = _make_plot_adata()
        fig = plot_dotplot(adata, ["GENE0", "GENE1", "GENE2"], groupby="leiden", show=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_dict_var_names(self):
        """Should accept dict of gene groups."""
        adata = _make_plot_adata()
        var_dict = {"group_a": ["GENE0", "GENE1"], "group_b": ["GENE2", "GENE3"]}
        fig = plot_dotplot(adata, var_dict, groupby="leiden", show=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_missing_genes_filtered(self):
        """Should work when some genes missing (filters to available)."""
        adata = _make_plot_adata()
        fig = plot_dotplot(adata, ["GENE0", "NOTEXIST"], groupby="leiden", show=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_all_missing_raises(self):
        """Should raise ValueError when no genes found."""
        adata = _make_plot_adata()
        with pytest.raises(ValueError, match="None of the specified"):
            plot_dotplot(adata, ["FAKE1", "FAKE2"], groupby="leiden", show=False)

    def test_missing_groupby_raises(self):
        """Should raise KeyError for missing groupby."""
        adata = _make_plot_adata()
        with pytest.raises(KeyError, match="bad_col"):
            plot_dotplot(adata, ["GENE0"], groupby="bad_col", show=False)

    def test_type_error(self):
        """Should raise TypeError for non-AnnData."""
        with pytest.raises(TypeError, match="plot_dotplot"):
            plot_dotplot("not_adata", ["GENE0"], groupby="x", show=False)

    def test_save(self, tmp_path):
        """Should save to file."""
        adata = _make_plot_adata()
        path = str(tmp_path / "dotplot.png")
        plot_dotplot(adata, ["GENE0", "GENE1"], groupby="leiden", save=path, show=False)
        import os

        assert os.path.exists(path)

    def test_public_api(self):
        assert hasattr(singlet, "plot_dotplot")
        assert callable(singlet.plot_dotplot)
