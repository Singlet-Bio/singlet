# SPDX-License-Identifier: MIT
"""Tests for singlet.calculate_qc_metrics()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._qc import calculate_qc_metrics


def _make_adata(n_cells=100, n_genes=50, with_mito=True):
    """Create AnnData with optional mitochondrial genes."""
    import anndata as ad

    rng = np.random.default_rng(42)
    X = rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32)

    gene_names = [f"GENE{i}" for i in range(n_genes)]
    if with_mito:
        # Replace last 5 genes with MT- prefix
        for i in range(5):
            gene_names[n_genes - 5 + i] = f"MT-{gene_names[n_genes - 5 + i]}"

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = gene_names
    return adata


class TestCalculateQcMetrics:
    def test_basic_inplace(self):
        """Should store metrics in adata.obs and adata.var."""
        adata = _make_adata()
        ret = calculate_qc_metrics(adata)
        assert ret is None
        assert "n_genes_by_counts" in adata.obs.columns
        assert "total_counts" in adata.obs.columns
        assert "n_cells_by_counts" in adata.var.columns

    def test_not_inplace(self):
        """inplace=False should return DataFrames."""
        adata = _make_adata()
        result = calculate_qc_metrics(adata, inplace=False)
        assert isinstance(result, tuple)
        obs_df, var_df = result
        assert isinstance(obs_df, pd.DataFrame)
        assert isinstance(var_df, pd.DataFrame)
        assert "n_genes_by_counts" not in adata.obs.columns

    def test_total_counts_correct(self):
        """total_counts should equal row sums."""
        adata = _make_adata()
        calculate_qc_metrics(adata)
        expected = np.asarray(adata.X.sum(axis=1)).ravel()
        np.testing.assert_allclose(adata.obs["total_counts"].values, expected)

    def test_n_genes_correct(self):
        """n_genes_by_counts should equal number of non-zero genes per cell."""
        adata = _make_adata()
        calculate_qc_metrics(adata)
        expected = np.asarray((adata.X > 0).sum(axis=1)).ravel()
        np.testing.assert_array_equal(adata.obs["n_genes_by_counts"].values, expected)

    def test_mito_autodetect(self):
        """Should auto-detect MT- genes and compute pct_counts_mt."""
        adata = _make_adata(with_mito=True)
        calculate_qc_metrics(adata)
        assert "pct_counts_mt" in adata.obs.columns
        assert "total_counts_mt" in adata.obs.columns
        # MT percent should be between 0 and 100
        pct = adata.obs["pct_counts_mt"].values
        assert np.all(pct >= 0) and np.all(pct <= 100)

    def test_no_mito(self):
        """Without MT- genes, should not add mt columns."""
        adata = _make_adata(with_mito=False)
        calculate_qc_metrics(adata)
        assert "pct_counts_mt" not in adata.obs.columns

    def test_custom_qc_vars(self):
        """Should compute metrics for custom gene sets."""
        adata = _make_adata(with_mito=False)
        adata.var["ribo"] = [i < 10 for i in range(50)]
        calculate_qc_metrics(adata, qc_vars=["ribo"])
        assert "pct_counts_ribo" in adata.obs.columns
        assert "total_counts_ribo" in adata.obs.columns

    def test_percent_top(self):
        """Should compute percent in top N genes."""
        adata = _make_adata()
        calculate_qc_metrics(adata, percent_top=[10, 20])
        assert "pct_counts_in_top_10_genes" in adata.obs.columns
        assert "pct_counts_in_top_20_genes" in adata.obs.columns

    def test_gene_metrics(self):
        """Should compute per-gene metrics."""
        adata = _make_adata()
        calculate_qc_metrics(adata)
        assert "mean_counts" in adata.var.columns
        assert "pct_dropout_by_counts" in adata.var.columns
        assert "n_cells_by_counts" in adata.var.columns

    def test_dense_input(self):
        """Should work with dense matrix."""
        adata = _make_adata()
        adata.X = adata.X.toarray()
        calculate_qc_metrics(adata)
        assert "total_counts" in adata.obs.columns

    def test_type_error(self):
        """Should raise TypeError for non-AnnData."""
        with pytest.raises(TypeError, match="calculate_qc_metrics"):
            calculate_qc_metrics("not_adata")

    def test_public_api(self):
        assert hasattr(singlet, "calculate_qc_metrics")
        assert callable(singlet.calculate_qc_metrics)
