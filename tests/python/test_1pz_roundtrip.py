# SPDX-License-Identifier: MIT
"""Integration tests for singlet._io read_1pz/write_1pz round-trip."""

import numpy as np
import pytest
import scipy.sparse as sp


class TestOnePzRoundTrip:
    """Test write → read cycle for .1pz format."""

    def _make_adata(self, n_genes=100, n_cells=50, density=0.3):
        ad = pytest.importorskip("anndata")
        import pandas as pd

        # AnnData is cells × genes; use integer counts like real scRNA-seq
        mat = sp.random(n_cells, n_genes, density=density, format="csr", dtype=np.float32)
        mat.data = np.round(mat.data * 100).astype(np.float32)
        adata = ad.AnnData(X=mat)
        adata.obs_names = pd.Index([f"CELL{i:04d}" for i in range(n_cells)])
        adata.var_names = pd.Index([f"GENE{j:04d}" for j in range(n_genes)])
        return adata

    def test_basic_roundtrip(self, tmp_path):
        """Write and read back preserves shape and values."""
        from singlet._io import read_1pz, write_1pz

        adata = self._make_adata()
        path = tmp_path / "test.1pz"
        write_1pz(adata, path)
        loaded = read_1pz(path)

        assert loaded.shape == adata.shape
        orig = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        load = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_allclose(load, orig, atol=1e-5)

    def test_preserves_gene_names(self, tmp_path):
        """Gene names survive round-trip."""
        from singlet._io import read_1pz, write_1pz

        adata = self._make_adata(n_genes=10, n_cells=5)
        path = tmp_path / "genes.1pz"
        write_1pz(adata, path)
        loaded = read_1pz(path)

        assert list(loaded.var_names) == list(adata.var_names)

    def test_preserves_cell_barcodes(self, tmp_path):
        """Cell barcodes survive round-trip."""
        from singlet._io import read_1pz, write_1pz

        adata = self._make_adata(n_genes=10, n_cells=5)
        path = tmp_path / "cells.1pz"
        write_1pz(adata, path)
        loaded = read_1pz(path)

        assert list(loaded.obs_names) == list(adata.obs_names)

    def test_empty_matrix(self, tmp_path):
        """Writing an all-zeros matrix works."""
        ad = pytest.importorskip("anndata")
        import pandas as pd
        from singlet._io import read_1pz, write_1pz

        mat = sp.csr_matrix((5, 3), dtype=np.float32)
        adata = ad.AnnData(X=mat)
        adata.obs_names = pd.Index([f"C{i}" for i in range(5)])
        adata.var_names = pd.Index([f"G{j}" for j in range(3)])

        path = tmp_path / "empty.1pz"
        write_1pz(adata, path)
        loaded = read_1pz(path)

        assert loaded.shape == (5, 3)
        load_dense = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        assert load_dense.sum() == 0

    def test_integer_values(self, tmp_path):
        """Integer count matrices work."""
        ad = pytest.importorskip("anndata")
        import pandas as pd
        from singlet._io import read_1pz, write_1pz

        # cells=10, genes=20
        mat = sp.random(10, 20, density=0.4, format="csr", dtype=np.float32)
        mat.data = np.round(mat.data * 100).astype(np.float32)
        adata = ad.AnnData(X=mat)
        adata.obs_names = pd.Index([f"C{i}" for i in range(10)])
        adata.var_names = pd.Index([f"G{j}" for j in range(20)])

        path = tmp_path / "int.1pz"
        write_1pz(adata, path)
        loaded = read_1pz(path)

        orig = adata.X.toarray()
        load = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_allclose(load, orig, atol=1e-4)

    def test_info_1pz(self, tmp_path):
        """info_1pz returns shape metadata."""
        from singlet._io import info_1pz, write_1pz

        adata = self._make_adata(n_genes=30, n_cells=20)
        path = tmp_path / "info.1pz"
        write_1pz(adata, path)
        info = info_1pz(path)

        assert isinstance(info, dict)
        # Should have dimension info
        assert info.get("n_genes", info.get("nrow", 0)) > 0 or info.get("m", 0) > 0


class TestDetectFormat:
    """Test format detection from magic bytes."""

    def test_1pz_detected(self, tmp_path):
        """Files with 1PZ magic byte are detected."""
        from singlet._io import _detect_format, write_1pz

        ad = pytest.importorskip("anndata")
        import pandas as pd

        # cells=3, genes=5
        adata = ad.AnnData(X=sp.random(3, 5, density=0.5, format="csr", dtype=np.float32))
        adata.obs_names = pd.Index([f"C{i}" for i in range(3)])
        adata.var_names = pd.Index([f"G{j}" for j in range(5)])

        path = tmp_path / "det.1pz"
        write_1pz(adata, path)
        assert _detect_format(path) == "1pz"

    def test_too_small_raises(self, tmp_path):
        """Files smaller than 8 bytes raise ValueError."""
        from singlet._io import _detect_format

        path = tmp_path / "tiny.1pz"
        path.write_bytes(b"\x00\x01\x02")
        with pytest.raises(ValueError, match="too small"):
            _detect_format(path)

    def test_unknown_magic_raises(self, tmp_path):
        """Unknown magic bytes raise ValueError."""
        from singlet._io import _detect_format

        path = tmp_path / "bad.bin"
        path.write_bytes(b"\x00" * 16)
        with pytest.raises(ValueError, match="Unknown file format"):
            _detect_format(path)
