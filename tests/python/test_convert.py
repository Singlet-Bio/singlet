"""Unit tests for singlet.convert — format conversion utilities."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from singlet.convert import (
    from_h5ad,
    from_mtx,
    to_csc,
    to_h5ad,
    to_mtx,
)


@pytest.fixture
def sample_adata():
    """Create a small AnnData for testing."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 100, 50
    X = sp.random(n_cells, n_genes, density=0.1, format="csr", random_state=42).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs_names = pd.Index([f"cell_{i}" for i in range(n_cells)])
    adata.var_names = pd.Index([f"gene_{i}" for i in range(n_genes)])
    adata.obs["batch"] = rng.choice(["A", "B"], size=n_cells)
    return adata


class TestToH5ad:
    def test_roundtrip(self, sample_adata, tmp_path):
        path = tmp_path / "test.h5ad"
        to_h5ad(sample_adata, path)
        assert path.exists()
        assert path.stat().st_size > 0

        loaded = from_h5ad(path)
        assert loaded.shape == sample_adata.shape
        assert list(loaded.obs_names) == list(sample_adata.obs_names)
        assert list(loaded.var_names) == list(sample_adata.var_names)

    def test_preserves_obs(self, sample_adata, tmp_path):
        path = tmp_path / "test.h5ad"
        to_h5ad(sample_adata, path)
        loaded = from_h5ad(path)
        assert "batch" in loaded.obs.columns
        assert list(loaded.obs["batch"]) == list(sample_adata.obs["batch"])

    def test_compression(self, sample_adata, tmp_path):
        path_gz = tmp_path / "test_gz.h5ad"
        to_h5ad(sample_adata, path_gz, compression="gzip")
        assert path_gz.exists()
        # Just verify it's a valid h5ad (compression is best-effort for small data)
        loaded = from_h5ad(path_gz)
        assert loaded.shape == sample_adata.shape


class TestToCsc:
    def test_returns_csc(self, sample_adata):
        mat = to_csc(sample_adata)
        assert sp.issparse(mat)
        assert mat.format == "csc"
        assert mat.shape == sample_adata.shape

    def test_from_dense(self):
        X = np.array([[1, 0, 3], [0, 2, 0]], dtype=np.float32)
        adata = ad.AnnData(X=X)
        mat = to_csc(adata)
        assert sp.issparse(mat)
        assert mat.format == "csc"
        assert mat.shape == (2, 3)

    def test_layer(self, sample_adata):
        sample_adata.layers["raw"] = sample_adata.X.copy()
        mat = to_csc(sample_adata, layer="raw")
        assert mat.shape == sample_adata.shape


class TestToMtx:
    def test_creates_files(self, sample_adata, tmp_path):
        out_dir = tmp_path / "mtx_out"
        to_mtx(sample_adata, out_dir)
        assert (out_dir / "matrix.mtx.gz").exists()
        assert (out_dir / "barcodes.tsv.gz").exists()
        assert (out_dir / "features.tsv.gz").exists()

    def test_roundtrip(self, sample_adata, tmp_path):
        out_dir = tmp_path / "mtx_out"
        to_mtx(sample_adata, out_dir)
        loaded = from_mtx(out_dir)
        assert loaded.shape == sample_adata.shape
        # Values should be preserved (within float precision)
        orig_dense = sample_adata.X.toarray() if sp.issparse(sample_adata.X) else sample_adata.X
        load_dense = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_array_almost_equal(orig_dense, load_dense, decimal=5)


class TestFromMtx:
    def test_reads_barcodes(self, sample_adata, tmp_path):
        out_dir = tmp_path / "mtx_out"
        to_mtx(sample_adata, out_dir)
        loaded = from_mtx(out_dir)
        assert list(loaded.obs_names) == list(sample_adata.obs_names)

    def test_reads_features(self, sample_adata, tmp_path):
        out_dir = tmp_path / "mtx_out"
        to_mtx(sample_adata, out_dir)
        loaded = from_mtx(out_dir)
        assert list(loaded.var_names) == list(sample_adata.var_names)

    def test_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            from_mtx(tmp_path / "nonexistent")


class TestFromH5ad:
    def test_backed_mode(self, sample_adata, tmp_path):
        path = tmp_path / "backed.h5ad"
        to_h5ad(sample_adata, path)
        loaded = from_h5ad(path, backed="r")
        assert loaded.shape == sample_adata.shape
        loaded.file.close()
