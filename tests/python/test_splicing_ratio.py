"""Tests for singlet.splicing_ratio()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._splicing_ratio import splicing_ratio


def _make_splicing_adata(n_cells=100, n_genes=50):
    """Create test AnnData with spliced and unspliced layers."""
    import anndata as ad

    rng = np.random.default_rng(42)
    X = sp.csr_matrix(rng.poisson(5, (n_cells, n_genes)).astype(np.float32))

    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE{idx}" for idx in range(n_genes)]
    adata.obs_names = [f"cell_{idx}" for idx in range(n_cells)]

    # Spliced: higher counts (mature mRNA dominates)
    spliced = rng.poisson(8, (n_cells, n_genes)).astype(np.float32)
    # Unspliced: lower counts
    unspliced = rng.poisson(3, (n_cells, n_genes)).astype(np.float32)

    adata.layers["spliced"] = sp.csr_matrix(spliced)
    adata.layers["unspliced"] = sp.csr_matrix(unspliced)

    return adata


class TestSplicingRatio:
    def test_basic(self):
        """Basic splicing ratio computation."""
        adata = _make_splicing_adata()
        result = splicing_ratio(adata)

        assert result is adata  # returns same object
        assert "splicing_ratio" in adata.layers
        assert "mean_splicing_ratio" in adata.obs.columns

    def test_ratio_in_range(self):
        """Splicing ratio values should be in [0, 1] or NaN."""
        adata = _make_splicing_adata()
        splicing_ratio(adata)

        ratio = adata.layers["splicing_ratio"]
        valid = ~np.isnan(ratio)
        assert (ratio[valid] >= 0).all()
        assert (ratio[valid] <= 1).all()

    def test_ratio_computation(self):
        """Verify ratio = spliced / (spliced + unspliced)."""
        import anndata as ad

        # Simple case: dense, known values
        spliced = np.array([[10, 0, 5], [8, 2, 0]], dtype=np.float32)
        unspliced = np.array([[5, 0, 5], [2, 8, 0]], dtype=np.float32)

        adata = ad.AnnData(X=spliced + unspliced)
        adata.var_names = ["A", "B", "C"]
        adata.obs_names = ["c1", "c2"]
        adata.layers["spliced"] = spliced
        adata.layers["unspliced"] = unspliced

        splicing_ratio(adata, min_counts=0)

        ratio = adata.layers["splicing_ratio"]
        # Gene A, cell 1: 10/(10+5) = 0.667
        assert abs(ratio[0, 0] - 10 / 15) < 1e-5
        # Gene C, cell 1: 5/(5+5) = 0.5
        assert abs(ratio[0, 2] - 0.5) < 1e-5
        # Gene A, cell 2: 8/(8+2) = 0.8
        assert abs(ratio[1, 0] - 0.8) < 1e-5

    def test_min_counts_masking(self):
        """Entries below min_counts should be NaN."""
        import anndata as ad

        spliced = np.array([[3, 100], [1, 50]], dtype=np.float32)
        unspliced = np.array([[2, 100], [1, 50]], dtype=np.float32)

        adata = ad.AnnData(X=spliced + unspliced)
        adata.var_names = ["low", "high"]
        adata.obs_names = ["c1", "c2"]
        adata.layers["spliced"] = spliced
        adata.layers["unspliced"] = unspliced

        splicing_ratio(adata, min_counts=10)

        ratio = adata.layers["splicing_ratio"]
        # Gene 'low': total=5 < 10 → NaN
        assert np.isnan(ratio[0, 0])
        assert np.isnan(ratio[1, 0])
        # Gene 'high': total=200 >= 10 → valid
        assert not np.isnan(ratio[0, 1])
        assert not np.isnan(ratio[1, 1])

    def test_mean_splicing_ratio(self):
        """Mean splicing ratio should be average across non-NaN genes."""
        import anndata as ad

        spliced = np.array([[8, 6], [10, 0]], dtype=np.float32)
        unspliced = np.array([[2, 4], [0, 0]], dtype=np.float32)

        adata = ad.AnnData(X=spliced + unspliced)
        adata.var_names = ["A", "B"]
        adata.obs_names = ["c1", "c2"]
        adata.layers["spliced"] = spliced
        adata.layers["unspliced"] = unspliced

        splicing_ratio(adata, min_counts=5)

        # Cell 1: gene A = 8/10=0.8, gene B = 6/10=0.6 → mean=0.7
        assert abs(adata.obs["mean_splicing_ratio"].iloc[0] - 0.7) < 1e-5
        # Cell 2: gene A = 10/10=1.0, gene B = 0/0=NaN (total=0 < 5) → mean=1.0
        assert abs(adata.obs["mean_splicing_ratio"].iloc[1] - 1.0) < 1e-5

    def test_sparse_input(self):
        """Should work with sparse layer matrices."""
        adata = _make_splicing_adata()
        # Confirm layers are sparse
        assert sp.issparse(adata.layers["spliced"])
        assert sp.issparse(adata.layers["unspliced"])

        splicing_ratio(adata)
        assert "splicing_ratio" in adata.layers

    def test_dense_input(self):
        """Should work with dense layer matrices."""
        adata = _make_splicing_adata()
        adata.layers["spliced"] = adata.layers["spliced"].toarray()
        adata.layers["unspliced"] = adata.layers["unspliced"].toarray()

        splicing_ratio(adata)
        assert "splicing_ratio" in adata.layers

    def test_custom_layer_names(self):
        """Should work with custom layer names."""
        adata = _make_splicing_adata()
        adata.layers["my_spliced"] = adata.layers["spliced"]
        adata.layers["my_unspliced"] = adata.layers["unspliced"]

        splicing_ratio(adata, spliced_layer="my_spliced", unspliced_layer="my_unspliced")
        assert "splicing_ratio" in adata.layers

    def test_missing_spliced_layer_raises(self):
        """Should raise KeyError when spliced layer missing."""
        adata = _make_splicing_adata()
        del adata.layers["spliced"]
        with pytest.raises(KeyError, match="spliced"):
            splicing_ratio(adata)

    def test_missing_unspliced_layer_raises(self):
        """Should raise KeyError when unspliced layer missing."""
        adata = _make_splicing_adata()
        del adata.layers["unspliced"]
        with pytest.raises(KeyError, match="unspliced"):
            splicing_ratio(adata)

    def test_negative_min_counts_raises(self):
        """Should raise ValueError for negative min_counts."""
        adata = _make_splicing_adata()
        with pytest.raises(ValueError, match="min_counts"):
            splicing_ratio(adata, min_counts=-1)

    def test_type_error(self):
        """Should raise TypeError on non-AnnData input."""
        with pytest.raises(TypeError, match="splicing_ratio"):
            splicing_ratio("not_adata")

    def test_min_counts_zero(self):
        """min_counts=0 should compute ratio for all non-zero totals."""
        import anndata as ad

        spliced = np.array([[1, 0], [0, 1]], dtype=np.float32)
        unspliced = np.array([[1, 0], [0, 1]], dtype=np.float32)

        adata = ad.AnnData(X=spliced + unspliced)
        adata.var_names = ["A", "B"]
        adata.obs_names = ["c1", "c2"]
        adata.layers["spliced"] = spliced
        adata.layers["unspliced"] = unspliced

        splicing_ratio(adata, min_counts=0)

        ratio = adata.layers["splicing_ratio"]
        # (1,0): 1/2=0.5
        assert abs(ratio[0, 0] - 0.5) < 1e-5
        # (0,0): 0/0 → division by zero gives NaN
        assert np.isnan(ratio[0, 1]) or abs(ratio[0, 1]) < 1e-10

    def test_all_zero_total(self):
        """Cells with all-zero counts should get NaN mean ratio."""
        import anndata as ad

        spliced = np.array([[0, 0], [5, 5]], dtype=np.float32)
        unspliced = np.array([[0, 0], [5, 5]], dtype=np.float32)

        adata = ad.AnnData(X=spliced + unspliced)
        adata.var_names = ["A", "B"]
        adata.obs_names = ["c1", "c2"]
        adata.layers["spliced"] = spliced
        adata.layers["unspliced"] = unspliced

        splicing_ratio(adata, min_counts=1)

        # Cell 1 has all zeros → all NaN → mean is NaN
        assert np.isnan(adata.obs["mean_splicing_ratio"].iloc[0])
        # Cell 2 has total=10 per gene → valid
        assert not np.isnan(adata.obs["mean_splicing_ratio"].iloc[1])

    def test_output_dtype(self):
        """Splicing ratio layer should be float32."""
        adata = _make_splicing_adata()
        splicing_ratio(adata)
        assert adata.layers["splicing_ratio"].dtype == np.float32

    def test_public_api(self):
        """Should be accessible via singlet.splicing_ratio."""
        assert hasattr(singlet, "splicing_ratio")
        assert callable(singlet.splicing_ratio)
