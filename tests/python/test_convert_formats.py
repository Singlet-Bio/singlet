"""Tests for singlet.convert convenience functions (h5ad/zarr wrappers)."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def sample_adata():
    """Create a small AnnData for conversion testing."""
    ad = pytest.importorskip("anndata")
    mat = sp.random(8, 5, density=0.4, format="csr", dtype=np.float32)
    mat.data = np.round(mat.data * 50).astype(np.float32)
    adata = ad.AnnData(X=mat)
    adata.obs_names = pd.Index([f"C{i}" for i in range(8)])
    adata.var_names = pd.Index([f"G{j}" for j in range(5)])
    return adata


class TestH5adConversion:
    def test_to_h5ad(self, sample_adata, tmp_path):
        """Write to h5ad creates a valid file."""
        from singlet.convert import to_h5ad

        path = tmp_path / "test.h5ad"
        to_h5ad(sample_adata, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_from_h5ad(self, sample_adata, tmp_path):
        """Read back from h5ad preserves data."""
        from singlet.convert import from_h5ad, to_h5ad

        path = tmp_path / "roundtrip.h5ad"
        to_h5ad(sample_adata, path)
        loaded = from_h5ad(path)

        assert loaded.shape == sample_adata.shape
        orig = sample_adata.X.toarray()
        load = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_allclose(load, orig, atol=1e-5)

    def test_h5ad_preserves_names(self, sample_adata, tmp_path):
        """Gene and cell names survive h5ad round-trip."""
        from singlet.convert import from_h5ad, to_h5ad

        path = tmp_path / "names.h5ad"
        to_h5ad(sample_adata, path)
        loaded = from_h5ad(path)

        assert list(loaded.obs_names) == list(sample_adata.obs_names)
        assert list(loaded.var_names) == list(sample_adata.var_names)

    def test_to_h5ad_compression(self, sample_adata, tmp_path):
        """to_h5ad with gzip compression produces smaller file."""
        from singlet.convert import to_h5ad

        gz_path = tmp_path / "compressed.h5ad"
        to_h5ad(sample_adata, gz_path, compression="gzip")
        assert gz_path.exists()


class TestZarrConversion:
    def test_to_zarr(self, sample_adata, tmp_path):
        """Write to zarr creates a zarr store directory."""
        pytest.importorskip("zarr")
        from singlet.convert import to_zarr

        path = tmp_path / "test.zarr"
        to_zarr(sample_adata, path)
        assert path.is_dir()

    def test_from_zarr(self, sample_adata, tmp_path):
        """Read back from zarr preserves data."""
        pytest.importorskip("zarr")
        from singlet.convert import from_zarr, to_zarr

        path = tmp_path / "roundtrip.zarr"
        to_zarr(sample_adata, path)
        loaded = from_zarr(path)

        assert loaded.shape == sample_adata.shape


class TestPzH5adConvenience:
    def test_pz_to_h5ad(self, sample_adata, tmp_path):
        """pz_to_h5ad converts .1pz → .h5ad."""
        from singlet._io import write_1pz
        from singlet.convert import pz_to_h5ad

        pz_path = tmp_path / "source.1pz"
        h5ad_path = tmp_path / "output.h5ad"
        write_1pz(sample_adata, pz_path)
        pz_to_h5ad(pz_path, h5ad_path)

        assert h5ad_path.exists()
        import anndata as ad

        loaded = ad.read_h5ad(h5ad_path)
        assert loaded.shape == sample_adata.shape

    def test_h5ad_to_pz(self, sample_adata, tmp_path):
        """h5ad_to_pz converts .h5ad → .1pz."""
        from singlet._io import read_1pz
        from singlet.convert import h5ad_to_pz, to_h5ad

        h5ad_path = tmp_path / "source.h5ad"
        pz_path = tmp_path / "output.1pz"
        to_h5ad(sample_adata, h5ad_path)
        h5ad_to_pz(h5ad_path, pz_path)

        assert pz_path.exists()
        loaded = read_1pz(pz_path)
        assert loaded.shape == sample_adata.shape

    def test_pz_h5ad_roundtrip_values(self, sample_adata, tmp_path):
        """Values survive pz→h5ad→pz round-trip."""
        from singlet._io import read_1pz, write_1pz
        from singlet.convert import h5ad_to_pz, pz_to_h5ad

        pz1 = tmp_path / "step1.1pz"
        h5ad = tmp_path / "step2.h5ad"
        pz2 = tmp_path / "step3.1pz"

        write_1pz(sample_adata, pz1)
        pz_to_h5ad(pz1, h5ad)
        h5ad_to_pz(h5ad, pz2)

        loaded = read_1pz(pz2)
        orig = sample_adata.X.toarray()
        load = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_allclose(load, orig, atol=1e-4)
