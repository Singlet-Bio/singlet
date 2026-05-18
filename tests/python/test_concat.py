# SPDX-License-Identifier: MIT
"""Tests for singlet.concatenate()."""

import pytest
import scipy.sparse as sp
import singlet
from singlet._concat import concatenate


def _make_adata(n_cells=50, n_genes=30, gene_prefix="GENE", seed=42):
    """Create test AnnData."""
    import anndata as ad

    X = sp.random(n_cells, n_genes, density=0.3, format="csr", random_state=seed)
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"{gene_prefix}{i}" for i in range(n_genes)]
    adata.obs["sample_id"] = [f"s{seed}"] * n_cells
    return adata


class TestConcatenate:
    def test_basic_inner_join(self):
        """Should concatenate with inner join (shared genes)."""
        a1 = _make_adata(n_cells=30, n_genes=20, seed=1)
        a2 = _make_adata(n_cells=40, n_genes=20, seed=2)
        result = concatenate([a1, a2])
        assert result.n_obs == 70
        assert result.n_vars == 20
        assert "batch" in result.obs.columns

    def test_batch_labels(self):
        """Should assign correct batch labels."""
        a1 = _make_adata(n_cells=20, seed=1)
        a2 = _make_adata(n_cells=30, seed=2)
        result = concatenate([a1, a2])
        assert (result.obs["batch"] == "0").sum() == 20
        assert (result.obs["batch"] == "1").sum() == 30

    def test_custom_batch_categories(self):
        """Should use custom batch category names."""
        a1 = _make_adata(n_cells=20, seed=1)
        a2 = _make_adata(n_cells=20, seed=2)
        result = concatenate([a1, a2], batch_categories=["sample_A", "sample_B"])
        assert "sample_A" in result.obs["batch"].values
        assert "sample_B" in result.obs["batch"].values

    def test_custom_batch_key(self):
        """Should use custom batch column name."""
        a1 = _make_adata(n_cells=20, seed=1)
        a2 = _make_adata(n_cells=20, seed=2)
        result = concatenate([a1, a2], batch_key="dataset")
        assert "dataset" in result.obs.columns

    def test_inner_join_different_genes(self):
        """Inner join should keep only shared genes."""
        a1 = _make_adata(n_cells=20, n_genes=30, gene_prefix="GENE", seed=1)
        import anndata as ad

        # a2 has genes GENE10-GENE39 (overlap: GENE10-GENE29)
        X2 = sp.random(25, 30, density=0.3, format="csr", random_state=2)
        a2 = ad.AnnData(X=X2)
        a2.obs_names = [f"c2_{i}" for i in range(25)]
        a2.var_names = [f"GENE{i}" for i in range(10, 40)]

        result = concatenate([a1, a2], join="inner")
        # Shared: GENE10-GENE29 = 20 genes
        assert result.n_vars == 20
        assert result.n_obs == 45

    def test_outer_join(self):
        """Outer join should keep all genes (fill missing with 0)."""
        a1 = _make_adata(n_cells=20, n_genes=10, gene_prefix="A", seed=1)
        import anndata as ad

        X2 = sp.random(15, 8, density=0.3, format="csr", random_state=2)
        a2 = ad.AnnData(X=X2)
        a2.obs_names = [f"c2_{i}" for i in range(15)]
        a2.var_names = [f"B{i}" for i in range(8)]

        result = concatenate([a1, a2], join="outer")
        # 10 A genes + 8 B genes = 18 total
        assert result.n_vars == 18
        assert result.n_obs == 35

    def test_single_adata(self):
        """Should work with single AnnData (returns copy)."""
        a1 = _make_adata(n_cells=30, seed=1)
        result = concatenate([a1])
        assert result.n_obs == 30
        assert "batch" in result.obs.columns

    def test_three_datasets(self):
        """Should work with 3+ datasets."""
        adatas = [_make_adata(n_cells=20, seed=i) for i in range(3)]
        result = concatenate(adatas)
        assert result.n_obs == 60
        assert len(result.obs["batch"].unique()) == 3

    def test_obs_names_unique(self):
        """Obs names should be made unique with suffix."""
        a1 = _make_adata(n_cells=10, seed=1)
        a2 = _make_adata(n_cells=10, seed=2)
        result = concatenate([a1, a2])
        assert len(set(result.obs_names)) == 20  # all unique

    def test_obs_metadata_preserved(self):
        """Should preserve obs columns from input datasets."""
        a1 = _make_adata(n_cells=20, seed=1)
        a2 = _make_adata(n_cells=20, seed=2)
        result = concatenate([a1, a2])
        assert "sample_id" in result.obs.columns

    def test_sparse_output(self):
        """Concatenating sparse inputs should produce sparse output."""
        a1 = _make_adata(n_cells=20, seed=1)
        a2 = _make_adata(n_cells=20, seed=2)
        result = concatenate([a1, a2])
        assert sp.issparse(result.X)

    def test_empty_list_raises(self):
        """Should raise ValueError for empty list."""
        with pytest.raises(ValueError, match="must not be empty"):
            concatenate([])

    def test_type_error(self):
        """Should raise TypeError for non-AnnData elements."""
        with pytest.raises(TypeError, match="not an AnnData"):
            concatenate(["not_adata", "also_not"])

    def test_invalid_join_raises(self):
        """Should raise ValueError for invalid join type."""
        a1 = _make_adata(seed=1)
        a2 = _make_adata(seed=2)
        with pytest.raises(ValueError, match="join must be"):
            concatenate([a1, a2], join="left")

    def test_public_api(self):
        assert hasattr(singlet, "concatenate")
        assert callable(singlet.concatenate)
