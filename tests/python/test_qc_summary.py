# SPDX-License-Identifier: MIT
"""Tests for singlet.qc_summary()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._qc_summary import qc_summary


def _make_qc_adata(n_cells=100, n_genes=200):
    """Create test AnnData with QC-relevant gene names."""
    import anndata as ad

    rng = np.random.default_rng(42)

    X = rng.poisson(5, (n_cells, n_genes)).astype(np.float32)

    adata = ad.AnnData(X=sp.csr_matrix(X))

    # Create gene names with mito and ribo genes
    gene_names = []
    for idx in range(n_genes):
        if idx < 5:
            gene_names.append(f"MT-GENE{idx}")
        elif idx < 15:
            gene_names.append(f"RPS{idx}")
        elif idx < 25:
            gene_names.append(f"RPL{idx}")
        else:
            gene_names.append(f"GENE{idx}")

    adata.var_names = gene_names
    adata.obs_names = [f"cell_{idx}" for idx in range(n_cells)]

    return adata


class TestQCSummary:
    def test_basic(self):
        """Basic QC summary computation."""
        adata = _make_qc_adata()
        result = qc_summary(adata)

        assert isinstance(result, pd.DataFrame)
        assert "median" in result.columns
        assert "mean" in result.columns
        assert "std" in result.columns
        assert "q25" in result.columns
        assert "q75" in result.columns

    def test_stores_obs_columns(self):
        """QC metrics stored in adata.obs."""
        adata = _make_qc_adata()
        qc_summary(adata)

        assert "qc_n_genes" in adata.obs.columns
        assert "qc_total_counts" in adata.obs.columns
        assert "qc_pct_mito" in adata.obs.columns
        assert "qc_pct_ribo" in adata.obs.columns
        assert "qc_complexity" in adata.obs.columns

    def test_metrics_in_summary(self):
        """Summary includes expected metrics."""
        adata = _make_qc_adata()
        result = qc_summary(adata)

        expected_metrics = {"n_genes", "total_counts", "pct_mito", "pct_ribo", "complexity"}
        assert expected_metrics.issubset(set(result.index))

    def test_mito_pct_positive(self):
        """Mitochondrial percentage should be positive when MT genes exist."""
        adata = _make_qc_adata()
        qc_summary(adata)
        assert (adata.obs["qc_pct_mito"] >= 0).all()
        # Should have some mito since we have MT- genes
        assert adata.obs["qc_pct_mito"].mean() > 0

    def test_ribo_pct_positive(self):
        """Ribosomal percentage should be positive when ribo genes exist."""
        adata = _make_qc_adata()
        qc_summary(adata)
        assert (adata.obs["qc_pct_ribo"] >= 0).all()
        assert adata.obs["qc_pct_ribo"].mean() > 0

    def test_complexity_range(self):
        """Complexity should be between 0 and ~1.5."""
        adata = _make_qc_adata()
        qc_summary(adata)
        complexity = adata.obs["qc_complexity"]
        assert (complexity >= 0).all()
        assert (complexity <= 2).all()

    def test_groupby(self):
        """Groupby produces per-group statistics."""
        adata = _make_qc_adata()
        adata.obs["cluster"] = np.random.default_rng(42).choice(
            ["A", "B", "C"], size=adata.n_obs
        )
        result = qc_summary(adata, groupby="cluster")

        assert "group" in result.columns
        assert set(result["group"].unique()) == {"A", "B", "C"}

    def test_groupby_missing_key_raises(self):
        """Missing groupby key raises KeyError."""
        adata = _make_qc_adata()
        with pytest.raises(KeyError, match="not_a_key"):
            qc_summary(adata, groupby="not_a_key")

    def test_custom_mito_prefix(self):
        """Custom mitochondrial prefix works."""
        import anndata as ad

        rng = np.random.default_rng(42)
        n_cells, n_genes = 50, 100
        X = rng.poisson(5, (n_cells, n_genes)).astype(np.float32)
        adata = ad.AnnData(X=sp.csr_matrix(X))
        gene_names = [f"mt-gene{idx}" if idx < 5 else f"GENE{idx}" for idx in range(n_genes)]
        adata.var_names = gene_names
        adata.obs_names = [f"cell_{idx}" for idx in range(n_cells)]

        qc_summary(adata, mito_prefix="mt-")
        assert adata.obs["qc_pct_mito"].mean() > 0

    def test_custom_ribo_prefix(self):
        """Custom ribosomal prefix works."""
        import anndata as ad

        rng = np.random.default_rng(42)
        n_cells, n_genes = 50, 100
        X = rng.poisson(5, (n_cells, n_genes)).astype(np.float32)
        adata = ad.AnnData(X=sp.csr_matrix(X))
        gene_names = [f"MRPS{idx}" if idx < 10 else f"GENE{idx}" for idx in range(n_genes)]
        adata.var_names = gene_names
        adata.obs_names = [f"cell_{idx}" for idx in range(n_cells)]

        qc_summary(adata, ribo_prefix="MRPS|MRPL")
        assert adata.obs["qc_pct_ribo"].mean() > 0

    def test_no_mito_genes(self):
        """Works when no mitochondrial genes exist."""
        import anndata as ad

        rng = np.random.default_rng(42)
        n_cells, n_genes = 50, 50
        X = rng.poisson(5, (n_cells, n_genes)).astype(np.float32)
        adata = ad.AnnData(X=sp.csr_matrix(X))
        adata.var_names = [f"GENE{idx}" for idx in range(n_genes)]
        adata.obs_names = [f"cell_{idx}" for idx in range(n_cells)]

        qc_summary(adata)
        assert (adata.obs["qc_pct_mito"] == 0).all()

    def test_doublet_score_included(self):
        """Doublet score included when available."""
        adata = _make_qc_adata()
        adata.obs["doublet_score"] = np.random.default_rng(42).uniform(0, 1, adata.n_obs)
        result = qc_summary(adata)
        assert "doublet_score" in result.index

    def test_dense_matrix(self):
        """Works with dense X matrix."""
        adata = _make_qc_adata()
        adata.X = np.asarray(adata.X.todense())
        result = qc_summary(adata)
        assert isinstance(result, pd.DataFrame)

    def test_summary_values_reasonable(self):
        """Summary statistics should be numerically reasonable."""
        adata = _make_qc_adata()
        result = qc_summary(adata)

        # Total counts median should be positive
        total_row = result.loc["total_counts"]
        assert total_row["median"] > 0
        assert total_row["mean"] > 0
        # Q25 <= median <= Q75
        assert total_row["q25"] <= total_row["median"] <= total_row["q75"]

    def test_n_genes_is_integer(self):
        """n_genes in obs should be integer."""
        adata = _make_qc_adata()
        qc_summary(adata)
        assert adata.obs["qc_n_genes"].dtype in (np.int64, np.int32, int)

    def test_singlet_api(self):
        """Function accessible via singlet namespace."""
        adata = _make_qc_adata()
        result = singlet.qc_summary(adata)
        assert isinstance(result, pd.DataFrame)
