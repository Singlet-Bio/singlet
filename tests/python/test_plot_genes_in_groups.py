# SPDX-License-Identifier: MIT
"""Tests for singlet.plot_genes_in_groups()."""

import numpy as np
import pytest
import singlet
from singlet._plot_genes_in_groups import plot_genes_in_groups


def _make_adata(n_cells=120, n_genes=80):
    """Create AnnData with groups and gene expression."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(99)

    X = rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32)

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Add categorical grouping
    groups = np.array(["A", "B", "C"] * (n_cells // 3) + ["A"] * (n_cells % 3))
    adata.obs["cluster"] = pd.Categorical(groups[:n_cells])
    adata.obs["batch"] = pd.Categorical(
        rng.choice(["batch1", "batch2"], size=n_cells)
    )

    return adata


class TestPlotGenesInGroups:
    def test_basic_output(self):
        """Should return (fig, axes) tuple."""
        import matplotlib.figure

        adata = _make_adata()
        fig, axes = plot_genes_in_groups(adata, ["gene_0", "gene_1"], groupby="cluster")
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(axes) == 2

    def test_single_gene(self):
        """Should work with single gene."""
        adata = _make_adata()
        fig, axes = plot_genes_in_groups(adata, ["gene_5"], groupby="cluster")
        assert len(axes) == 1

    def test_many_genes(self):
        """Should work with many genes."""
        adata = _make_adata()
        genes = [f"gene_{i}" for i in range(5)]
        fig, axes = plot_genes_in_groups(adata, genes, groupby="cluster")
        assert len(axes) == 5

    def test_different_groupby(self):
        """Should work with different groupby columns."""
        adata = _make_adata()
        fig, axes = plot_genes_in_groups(adata, ["gene_0"], groupby="batch")
        assert len(axes) == 1

    def test_custom_figsize(self):
        """Should accept custom figsize."""
        adata = _make_adata()
        fig, axes = plot_genes_in_groups(
            adata, ["gene_0"], groupby="cluster", figsize=(10, 4)
        )
        w, h = fig.get_size_inches()
        assert abs(w - 10.0) < 0.1
        assert abs(h - 4.0) < 0.1

    def test_use_raw(self):
        """Should use raw when specified."""
        adata = _make_adata()
        adata.raw = adata.copy()
        fig, axes = plot_genes_in_groups(
            adata, ["gene_0"], groupby="cluster", use_raw=True
        )
        assert len(axes) == 1

    def test_use_raw_none_raises(self):
        """Should raise if use_raw=True but raw is None."""
        adata = _make_adata()
        with pytest.raises(ValueError, match="raw is None"):
            plot_genes_in_groups(adata, ["gene_0"], groupby="cluster", use_raw=True)

    def test_missing_gene_raises(self):
        """Should raise KeyError for missing gene."""
        adata = _make_adata()
        with pytest.raises(KeyError, match="NONEXISTENT"):
            plot_genes_in_groups(adata, ["NONEXISTENT"], groupby="cluster")

    def test_missing_groupby_raises(self):
        """Should raise KeyError for missing groupby."""
        adata = _make_adata()
        with pytest.raises(KeyError, match="missing_col"):
            plot_genes_in_groups(adata, ["gene_0"], groupby="missing_col")

    def test_empty_var_names_raises(self):
        """Should raise ValueError for empty var_names."""
        adata = _make_adata()
        with pytest.raises(ValueError, match="must not be empty"):
            plot_genes_in_groups(adata, [], groupby="cluster")

    def test_type_error(self):
        """Should raise TypeError for non-AnnData."""
        with pytest.raises(TypeError, match="plot_genes_in_groups"):
            plot_genes_in_groups("not_adata", ["gene"], groupby="x")

    def test_sparse_input(self):
        """Should handle sparse expression matrix."""
        import scipy.sparse as sp

        adata = _make_adata()
        adata.X = sp.csr_matrix(adata.X)
        fig, axes = plot_genes_in_groups(adata, ["gene_0", "gene_1"], groupby="cluster")
        assert len(axes) == 2

    def test_public_api(self):
        """Function should be accessible from singlet namespace."""
        assert hasattr(singlet, "plot_genes_in_groups")
        assert callable(singlet.plot_genes_in_groups)

    def test_axes_have_titles(self):
        """Each subplot should have a title matching gene name."""
        adata = _make_adata()
        genes = ["gene_0", "gene_3"]
        fig, axes = plot_genes_in_groups(adata, genes, groupby="cluster")
        for ax, gene in zip(axes, genes):
            assert ax.get_title() == gene
