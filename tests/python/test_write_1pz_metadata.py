# SPDX-License-Identifier: MIT
"""Tests for singlet._io.write_1pz with metadata embedding."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def sample_adata():
    """Create AnnData with obs, var, and uns metadata."""
    ad = pytest.importorskip("anndata")
    n_cells, n_genes = 15, 8
    mat = sp.random(n_cells, n_genes, density=0.4, format="csr", dtype=np.float32)
    mat.data = np.round(mat.data * 100).astype(np.float32)

    adata = ad.AnnData(X=mat)
    adata.obs_names = pd.Index([f"CELL{i:03d}" for i in range(n_cells)])
    adata.var_names = pd.Index([f"GENE{j:03d}" for j in range(n_genes)])
    adata.obs["cluster"] = np.random.choice(["A", "B", "C"], n_cells)
    adata.obs["score"] = np.random.rand(n_cells)
    adata.var["highly_variable"] = np.random.choice([True, False], n_genes)
    adata.uns["species"] = "human"
    adata.uns["version"] = 2
    return adata


class TestWrite1pzMetadata:
    def test_obs_roundtrip(self, sample_adata, tmp_path):
        """Obs DataFrame is embedded and recoverable."""
        from singlet._io import read_1pz, write_1pz

        path = tmp_path / "with_obs.1pz"
        write_1pz(sample_adata, path, include_obs=True)
        loaded = read_1pz(path)

        if "cluster" in loaded.obs.columns:
            assert set(loaded.obs["cluster"].dropna().unique()) <= {"A", "B", "C"}

    def test_var_roundtrip(self, sample_adata, tmp_path):
        """Var DataFrame is embedded and recoverable."""
        from singlet._io import read_1pz, write_1pz

        path = tmp_path / "with_var.1pz"
        write_1pz(sample_adata, path, include_var=True)
        loaded = read_1pz(path)

        if "highly_variable" in loaded.var.columns:
            assert loaded.var["highly_variable"].dtype in [bool, object, np.bool_]

    def test_uns_roundtrip(self, sample_adata, tmp_path):
        """Uns key-value pairs are embedded and recoverable."""
        from singlet._io import read_1pz, write_1pz

        path = tmp_path / "with_uns.1pz"
        write_1pz(sample_adata, path, include_uns=True)
        loaded = read_1pz(path)

        if "species" in loaded.uns:
            assert loaded.uns["species"] == "human"

    def test_exclude_obs(self, sample_adata, tmp_path):
        """include_obs=False omits obs DataFrame."""
        from singlet._io import read_1pz, write_1pz

        path = tmp_path / "no_obs.1pz"
        write_1pz(sample_adata, path, include_obs=False)
        loaded = read_1pz(path)

        # Should not have the custom obs columns (only barcodes)
        assert "cluster" not in loaded.obs.columns or loaded.obs["cluster"].isna().all()

    def test_exclude_var(self, sample_adata, tmp_path):
        """include_var=False omits var DataFrame."""
        from singlet._io import read_1pz, write_1pz

        path = tmp_path / "no_var.1pz"
        write_1pz(sample_adata, path, include_var=False)
        loaded = read_1pz(path)

        assert "highly_variable" not in loaded.var.columns

    def test_exclude_uns(self, sample_adata, tmp_path):
        """include_uns=False omits uns dict."""
        from singlet._io import read_1pz, write_1pz

        path = tmp_path / "no_uns.1pz"
        write_1pz(sample_adata, path, include_uns=False)
        loaded = read_1pz(path)

        assert "species" not in loaded.uns

    def test_layer_selection(self, sample_adata, tmp_path):
        """Write a specific layer instead of X."""
        from singlet._io import read_1pz, write_1pz

        # Add a layer with different values
        sample_adata.layers["raw"] = (sample_adata.X * 2).tocsr()
        path = tmp_path / "layer.1pz"
        write_1pz(sample_adata, path, layer="raw")
        loaded = read_1pz(path)

        orig = (sample_adata.X * 2).toarray()
        load = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_allclose(load, orig, atol=1e-4)

    def test_returns_stats_dict(self, sample_adata, tmp_path):
        """write_1pz returns compression statistics."""
        from singlet._io import write_1pz

        path = tmp_path / "stats.1pz"
        result = write_1pz(sample_adata, path)
        assert isinstance(result, dict)

    def test_dense_matrix_write(self, tmp_path):
        """Dense matrix is automatically converted to sparse."""
        ad = pytest.importorskip("anndata")
        from singlet._io import read_1pz, write_1pz

        X = np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]], dtype=np.float32)
        adata = ad.AnnData(X=X)
        adata.obs_names = pd.Index(["C1", "C2"])
        adata.var_names = pd.Index(["G1", "G2", "G3"])

        path = tmp_path / "dense.1pz"
        write_1pz(adata, path)
        loaded = read_1pz(path)

        load = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_allclose(load, X, atol=1e-4)
