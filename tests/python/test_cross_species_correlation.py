# SPDX-License-Identifier: MIT
"""Tests for singlet.cross_species_correlation()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from singlet._cross_species_correlation import cross_species_correlation


def _make_adata(n_cells=80, n_genes=100, prefix="GENE", seed=42):
    """Create AnnData with clustered structure."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    X = rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32)

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"{prefix}{i}" for i in range(n_genes)]

    # Assign clusters
    labels = np.array(["A"] * (n_cells // 2) + ["B"] * (n_cells - n_cells // 2))
    adata.obs["cluster"] = pd.Categorical(labels)

    # Make cluster A have high expression of first 10 genes
    X_dense = X.copy()
    X_dense[: n_cells // 2, :10] += 10.0
    X_dense[n_cells // 2 :, 10:20] += 10.0
    adata.X = sp.csr_matrix(X_dense)

    return adata


class TestCrossSpeciesCorrelation:
    def test_basic_shared_genes(self):
        """Should compute correlation with shared gene names."""
        adata1 = _make_adata(n_cells=60, n_genes=50, prefix="GENE", seed=42)
        adata2 = _make_adata(n_cells=60, n_genes=50, prefix="GENE", seed=99)

        result = cross_species_correlation(adata1, adata2, groupby1="cluster", groupby2="cluster")

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)  # 2 clusters x 2 clusters
        assert list(result.index) == ["A", "B"]
        assert list(result.columns) == ["A", "B"]

    def test_self_correlation_high(self):
        """Same dataset should have high self-correlation."""
        adata = _make_adata(n_cells=60, n_genes=50)

        result = cross_species_correlation(adata, adata, groupby1="cluster", groupby2="cluster")

        # Diagonal should be 1.0 (same centroids)
        assert result.loc["A", "A"] > 0.99
        assert result.loc["B", "B"] > 0.99

    def test_gene_mapping(self):
        """Should correctly map gene names between species."""
        adata1 = _make_adata(n_cells=60, n_genes=50, prefix="HUMAN_")
        adata2 = _make_adata(n_cells=60, n_genes=50, prefix="MOUSE_")

        # Create mapping from human to mouse
        mapping = {f"HUMAN_{i}": f"MOUSE_{i}" for i in range(50)}

        result = cross_species_correlation(
            adata1, adata2, gene_mapping=mapping, groupby1="cluster", groupby2="cluster"
        )

        assert result.shape == (2, 2)
        # Since same data structure, corresponding clusters should correlate
        assert result.loc["A", "A"] > result.loc["A", "B"]

    def test_partial_gene_mapping(self):
        """Should work with partial gene mapping (subset of genes)."""
        adata1 = _make_adata(n_cells=40, n_genes=50, prefix="H_")
        adata2 = _make_adata(n_cells=40, n_genes=50, prefix="M_")

        # Only map first 25 genes
        mapping = {f"H_{i}": f"M_{i}" for i in range(25)}

        result = cross_species_correlation(
            adata1, adata2, gene_mapping=mapping, groupby1="cluster", groupby2="cluster"
        )

        assert result.shape == (2, 2)

    def test_no_shared_genes_raises(self):
        """Should raise if no genes are shared."""
        adata1 = _make_adata(n_cells=20, n_genes=20, prefix="AAA_")
        adata2 = _make_adata(n_cells=20, n_genes=20, prefix="BBB_")

        with pytest.raises(ValueError, match="No shared genes found"):
            cross_species_correlation(adata1, adata2)

    def test_invalid_method_raises(self):
        """Should raise for invalid method."""
        adata1 = _make_adata(n_cells=20, n_genes=20)
        adata2 = _make_adata(n_cells=20, n_genes=20)

        with pytest.raises(ValueError, match="method must be one of"):
            cross_species_correlation(adata1, adata2, method="invalid")

    def test_spearman_method(self):
        """Spearman correlation should work."""
        adata1 = _make_adata(n_cells=40, n_genes=30)
        adata2 = _make_adata(n_cells=40, n_genes=30)

        result = cross_species_correlation(
            adata1, adata2, method="spearman", groupby1="cluster", groupby2="cluster"
        )

        assert result.shape == (2, 2)
        # Values should be valid correlations
        assert result.values.min() >= -1.0
        assert result.values.max() <= 1.0

    def test_without_groupby(self):
        """Without groupby, should correlate cell-by-cell."""
        adata1 = _make_adata(n_cells=10, n_genes=30)
        adata2 = _make_adata(n_cells=15, n_genes=30)

        result = cross_species_correlation(adata1, adata2)

        assert result.shape == (10, 15)
        assert list(result.index) == list(adata1.obs_names)
        assert list(result.columns) == list(adata2.obs_names)

    def test_one_groupby(self):
        """Should work with groupby on only one dataset."""
        adata1 = _make_adata(n_cells=40, n_genes=30)
        adata2 = _make_adata(n_cells=30, n_genes=30)

        result = cross_species_correlation(adata1, adata2, groupby1="cluster")

        assert result.shape[0] == 2  # 2 clusters from adata1
        assert result.shape[1] == 30  # 30 cells from adata2

    def test_dense_matrix(self):
        """Should work with dense matrices."""
        import anndata as ad

        rng = np.random.default_rng(42)
        X1 = rng.poisson(3, size=(30, 40)).astype(np.float32)
        X2 = rng.poisson(3, size=(30, 40)).astype(np.float32)

        adata1 = ad.AnnData(X=X1)
        adata2 = ad.AnnData(X=X2)
        adata1.var_names = [f"G{i}" for i in range(40)]
        adata2.var_names = [f"G{i}" for i in range(40)]
        adata1.obs["grp"] = pd.Categorical(["X"] * 15 + ["Y"] * 15)
        adata2.obs["grp"] = pd.Categorical(["X"] * 15 + ["Y"] * 15)

        result = cross_species_correlation(adata1, adata2, groupby1="grp", groupby2="grp")

        assert result.shape == (2, 2)

    def test_correlation_values_valid(self):
        """All correlation values should be in [-1, 1]."""
        adata1 = _make_adata(n_cells=50, n_genes=40)
        adata2 = _make_adata(n_cells=50, n_genes=40, seed=99)

        result = cross_species_correlation(adata1, adata2, groupby1="cluster", groupby2="cluster")

        assert np.all(result.values >= -1.0 - 1e-10)
        assert np.all(result.values <= 1.0 + 1e-10)

    def test_singlet_import(self):
        """Should be importable from singlet namespace."""
        import singlet

        assert hasattr(singlet, "cross_species_correlation")
        assert callable(singlet.cross_species_correlation)
