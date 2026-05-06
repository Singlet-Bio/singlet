"""Tests for singlet._io.read_matrix and .spz format round-trip."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def sample_adata():
    """Create AnnData with integer count data."""
    ad = pytest.importorskip("anndata")
    n_cells, n_genes = 12, 6
    mat = sp.random(n_cells, n_genes, density=0.5, format="csr", dtype=np.float64)
    mat.data = np.round(mat.data * 50).astype(np.float64)

    adata = ad.AnnData(X=mat)
    adata.obs_names = pd.Index([f"BC{i:04d}" for i in range(n_cells)])
    adata.var_names = pd.Index([f"GENE{j}" for j in range(n_genes)])
    return adata


class TestSpzRoundTrip:
    def test_basic_roundtrip(self, sample_adata, tmp_path):
        """write_spz → read_spz preserves shape and values."""
        from singlet._io import read_spz, write_spz

        path = tmp_path / "test.spz"
        write_spz(sample_adata, path)
        loaded = read_spz(path)

        assert loaded.shape == sample_adata.shape
        orig = sample_adata.X.toarray()
        load = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_allclose(load, orig, atol=1e-4)

    def test_preserves_gene_names(self, sample_adata, tmp_path):
        """Gene names survive spz round-trip."""
        from singlet._io import read_spz, write_spz

        path = tmp_path / "genes.spz"
        write_spz(sample_adata, path)
        loaded = read_spz(path)

        assert list(loaded.var_names) == list(sample_adata.var_names)

    def test_preserves_barcodes(self, sample_adata, tmp_path):
        """Cell barcodes survive spz round-trip."""
        from singlet._io import read_spz, write_spz

        path = tmp_path / "barcodes.spz"
        write_spz(sample_adata, path)
        loaded = read_spz(path)

        assert list(loaded.obs_names) == list(sample_adata.obs_names)

    def test_spz_info(self, sample_adata, tmp_path):
        """spz_info returns metadata dict."""
        from singlet._io import spz_info, write_spz

        path = tmp_path / "info.spz"
        write_spz(sample_adata, path)
        info = spz_info(path)

        assert isinstance(info, dict)
        assert info.get("rows", 0) > 0


class TestReadMatrix:
    def test_auto_detects_1pz(self, tmp_path):
        """read_matrix detects and reads .1pz files."""
        ad = pytest.importorskip("anndata")
        from singlet._io import read_matrix, write_1pz

        mat = sp.random(5, 3, density=0.5, format="csr", dtype=np.float32)
        mat.data = np.round(mat.data * 10).astype(np.float32)
        adata = ad.AnnData(X=mat)
        adata.obs_names = pd.Index([f"C{i}" for i in range(5)])
        adata.var_names = pd.Index([f"G{j}" for j in range(3)])

        path = tmp_path / "auto.1pz"
        write_1pz(adata, path)
        loaded = read_matrix(path)

        assert loaded.shape == (5, 3)

    def test_auto_detects_spz(self, sample_adata, tmp_path):
        """read_matrix detects and reads .spz files."""
        from singlet._io import read_matrix, write_spz

        path = tmp_path / "auto.spz"
        write_spz(sample_adata, path)
        loaded = read_matrix(path)

        assert loaded.shape == sample_adata.shape

    def test_invalid_format_raises(self, tmp_path):
        """read_matrix raises on invalid file."""
        from singlet._io import read_matrix

        path = tmp_path / "bad.bin"
        path.write_bytes(b"\x00" * 16)
        with pytest.raises(ValueError, match="Unknown file format"):
            read_matrix(path)


def test_write_spz_invalid_precision(tmp_path):
    """write_spz raises ValueError for invalid precision."""
    import anndata as ad
    import numpy as np
    import pytest
    import scipy.sparse as sp
    from singlet._io import write_spz

    adata = ad.AnnData(X=sp.csr_matrix(np.ones((3, 5))))
    with pytest.raises(ValueError, match="Invalid precision"):
        write_spz(adata, tmp_path / "test.spz", precision="invalid")
