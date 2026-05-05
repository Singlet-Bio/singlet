"""Tests for singlet._io internal helpers and col_range feature."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


class TestResultToAnndata:
    """Test _result_to_anndata internal helper."""

    def test_basic_conversion(self):
        """Converts CSC components to AnnData (transposed)."""
        from singlet._io import _result_to_anndata

        # Create a 3 genes × 4 cells matrix in CSC format
        mat = sp.random(3, 4, density=0.5, format="csc", dtype=np.float64)
        result = {
            "data": mat.data,
            "indices": mat.indices,
            "indptr": mat.indptr,
            "shape": list(mat.shape),
        }
        adata = _result_to_anndata(result)
        # Transposed: should be 4 cells × 3 genes
        assert adata.shape == (4, 3)

    def test_with_dimnames(self):
        """Attaches row/col names."""
        from singlet._io import _result_to_anndata

        mat = sp.csc_matrix(np.array([[1, 0], [0, 2], [3, 0]], dtype=np.float64))
        result = {
            "data": mat.data,
            "indices": mat.indices,
            "indptr": mat.indptr,
            "shape": list(mat.shape),
            "rownames": ["GENE_A", "GENE_B", "GENE_C"],
            "colnames": ["CELL_1", "CELL_2"],
        }
        adata = _result_to_anndata(result)
        assert list(adata.var_names) == ["GENE_A", "GENE_B", "GENE_C"]
        assert list(adata.obs_names) == ["CELL_1", "CELL_2"]

    def test_non_monotonic_indptr_repair(self):
        """Non-monotonic indptr is repaired via maximum.accumulate."""
        from singlet._io import _result_to_anndata

        # Simulate a bug where indptr decreases
        data = np.array([1.0, 2.0, 3.0])
        indices = np.array([0, 1, 0])
        # indptr should be [0, 1, 2, 3] but has a "dip"
        indptr = np.array([0, 1, 1, 3])  # non-monotonic at position 2
        result = {
            "data": data,
            "indices": indices,
            "indptr": indptr,
            "shape": [2, 3],
        }
        # Should not raise
        adata = _result_to_anndata(result)
        assert adata.shape == (3, 2)

    def test_empty_matrix(self):
        """Handles empty sparse matrix."""
        from singlet._io import _result_to_anndata

        result = {
            "data": np.array([], dtype=np.float64),
            "indices": np.array([], dtype=np.int32),
            "indptr": np.array([0, 0, 0, 0], dtype=np.int32),
            "shape": [5, 3],
        }
        adata = _result_to_anndata(result)
        assert adata.shape == (3, 5)
        assert adata.X.nnz == 0


class TestReadSpzColRange:
    """Test col_range parameter for chunked reads."""

    def test_col_range_subset(self, tmp_path):
        """Reading with col_range returns subset of cells."""
        ad = pytest.importorskip("anndata")
        from singlet._io import read_spz, write_spz

        mat = sp.random(10, 20, density=0.4, format="csr", dtype=np.float64)
        mat.data = np.round(mat.data * 50).astype(np.float64)
        adata = ad.AnnData(X=mat)
        adata.obs_names = pd.Index([f"C{i:02d}" for i in range(10)])
        adata.var_names = pd.Index([f"G{j:02d}" for j in range(20)])

        path = tmp_path / "full.spz"
        write_spz(adata, path)

        # Read columns 2-5 (cells in the on-disk genes×cells format)
        subset = read_spz(path, col_range=(2, 5))
        # col_range applies to the on-disk matrix columns (cells)
        assert subset.shape[0] == 3  # 3 cells (5-2)
        assert subset.shape[1] == 20  # all genes

    def test_col_range_full(self, tmp_path):
        """col_range=(0, n_cells) returns all data."""
        ad = pytest.importorskip("anndata")
        from singlet._io import read_spz, write_spz

        mat = sp.random(5, 8, density=0.5, format="csr", dtype=np.float64)
        mat.data = np.round(mat.data * 30).astype(np.float64)
        adata = ad.AnnData(X=mat)
        adata.obs_names = pd.Index([f"C{i}" for i in range(5)])
        adata.var_names = pd.Index([f"G{j}" for j in range(8)])

        path = tmp_path / "full2.spz"
        write_spz(adata, path)

        full = read_spz(path)
        subset = read_spz(path, col_range=(0, 5))
        assert subset.shape == full.shape

        full_dense = full.X.toarray() if sp.issparse(full.X) else full.X
        sub_dense = subset.X.toarray() if sp.issparse(subset.X) else subset.X
        np.testing.assert_allclose(sub_dense, full_dense, atol=1e-4)


class TestImportSparsepress:
    """Test _import_sparsepress legacy import helper."""

    def test_import_sparsepress_not_installed(self, monkeypatch):
        """Raises ImportError when sparsepress is not available."""
        import builtins
        import sys

        from singlet._io import _import_sparsepress

        # Remove sparsepress from sys.modules if present
        monkeypatch.delitem(sys.modules, "sparsepress", raising=False)

        original_import = builtins.__import__

        def fail_import(name, *args, **kwargs):
            if name == "sparsepress":
                raise ImportError("no sparsepress")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_import)
        with pytest.raises(ImportError, match="legacy sparsepress_v2 format"):
            _import_sparsepress()

    def test_import_sparsepress_installed(self, monkeypatch):
        """Returns module when sparsepress is importable."""
        import sys
        from types import ModuleType

        from singlet._io import _import_sparsepress

        mock_sp = ModuleType("sparsepress")
        mock_sp.sp_read = lambda *a: None
        monkeypatch.setitem(sys.modules, "sparsepress", mock_sp)

        result = _import_sparsepress()
        assert result is mock_sp


class TestWriteSpzIntegerPath:
    """Test write_spz with integer-typed data."""

    def test_write_read_integer_matrix(self, tmp_path):
        """Integer AnnData roundtrips via write_spz/read_spz."""
        import anndata as ad
        from singlet._io import read_spz, write_spz

        # Create integer-typed matrix
        mat = sp.csr_matrix(np.array([[1, 0, 3], [0, 5, 0], [2, 0, 4]], dtype=np.int32))
        adata = ad.AnnData(X=mat)
        adata.obs_names = pd.Index(["c0", "c1", "c2"])
        adata.var_names = pd.Index(["g0", "g1", "g2"])

        path = tmp_path / "int.spz"
        write_spz(adata, path)
        loaded = read_spz(path)

        expected = mat.toarray()
        actual = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_array_equal(actual, expected)


class TestWrite1pzUnsFiltering:
    """Test write_1pz with non-scalar uns values."""

    def test_non_scalar_uns_excluded(self, tmp_path):
        """Non-scalar uns values (arrays, dicts) are excluded from metadata."""
        import anndata as ad
        from singlet._io import read_1pz, write_1pz

        mat = sp.csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.float32))
        adata = ad.AnnData(X=mat)
        adata.uns["array_val"] = np.array([1, 2, 3])
        adata.uns["dict_val"] = {"nested": True}
        adata.uns["list_val"] = [1, 2, 3]
        # No scalar values → uns_dict should be None internally

        path = tmp_path / "no_uns.1pz"
        write_1pz(adata, path)
        loaded = read_1pz(path)
        assert loaded.shape == (2, 2)
        # No uns should survive (all non-scalar)
        assert len(loaded.uns) == 0

    def test_mixed_uns_keeps_scalars(self, tmp_path):
        """Scalar uns values survive roundtrip, non-scalars are excluded."""
        import anndata as ad
        from singlet._io import read_1pz, write_1pz

        mat = sp.csr_matrix(np.array([[5, 6], [7, 8]], dtype=np.float32))
        adata = ad.AnnData(X=mat)
        adata.uns["version"] = "1.0"
        adata.uns["n_cells"] = 42
        adata.uns["big_array"] = np.zeros(1000)

        path = tmp_path / "mixed_uns.1pz"
        write_1pz(adata, path)
        loaded = read_1pz(path)
        assert "version" in loaded.uns
        assert loaded.uns["version"] == "1.0"


class TestLegacyV2Paths:
    """Test legacy sparsepress_v2 read paths (lines 84, 175, 253-285)."""

    def test_read_spz_legacy_dispatches(self):
        """read_spz dispatches to _read_spz_legacy for v2 files."""
        from unittest.mock import MagicMock, patch

        mock_adata = MagicMock()
        with (
            patch("singlet._io._detect_format_version", return_value=2),
            patch("singlet._io._read_spz_legacy", return_value=mock_adata) as mock_legacy,
        ):
            from singlet._io import read_spz

            result = read_spz("/fake/path.spz")
            mock_legacy.assert_called_once_with("/fake/path.spz", col_range=None)
            assert result is mock_adata

    def test_spz_info_legacy_dispatches(self):
        """spz_info dispatches to _spz_info_legacy for v2 files."""
        from unittest.mock import patch

        with (
            patch("singlet._io._detect_format_version", return_value=2),
            patch("singlet._io._spz_info_legacy", return_value={"rows": 10}) as mock_info,
        ):
            from singlet._io import spz_info

            result = spz_info("/fake/path.spz")
            mock_info.assert_called_once_with("/fake/path.spz")
            assert result == {"rows": 10}

    def test_read_spz_legacy_full(self):
        """_read_spz_legacy calls sparsepress and returns AnnData."""
        from unittest.mock import MagicMock, patch

        mock_sp = MagicMock()
        mock_sp.sp_read.return_value = {
            "data": np.array([1.0, 2.0, 3.0]),
            "indices": np.array([0, 1, 0]),
            "indptr": np.array([0, 2, 3]),
            "shape": [2, 2],
        }

        with patch("singlet._io._import_sparsepress", return_value=mock_sp):
            from singlet._io import _read_spz_legacy

            adata = _read_spz_legacy("/fake.spz")
            assert adata.shape == (2, 2)  # transposed: 2 cells × 2 genes

    def test_read_spz_legacy_with_col_range(self):
        """_read_spz_legacy respects col_range subsetting."""
        from unittest.mock import MagicMock, patch

        mock_sp = MagicMock()
        # Full matrix: 3 genes × 4 cells CSC
        mat = sp.random(3, 4, density=0.6, format="csc", dtype=np.float64)
        mock_sp.sp_read.return_value = {
            "data": mat.data,
            "indices": mat.indices,
            "indptr": mat.indptr,
            "shape": list(mat.shape),
        }

        with patch("singlet._io._import_sparsepress", return_value=mock_sp):
            from singlet._io import _read_spz_legacy

            adata = _read_spz_legacy("/fake.spz", col_range=(1, 3))
            # Subsetted to cols 1-3: 3 genes × 2 cells → transposed: 2 cells × 3 genes
            assert adata.shape == (2, 3)

    def test_spz_info_legacy_impl(self):
        """_spz_info_legacy delegates to sparsepress.sp_info."""
        from unittest.mock import MagicMock, patch

        mock_sp = MagicMock()
        mock_sp.sp_info.return_value = {"rows": 5, "cols": 10, "nnz": 50}

        with patch("singlet._io._import_sparsepress", return_value=mock_sp):
            from singlet._io import _spz_info_legacy

            result = _spz_info_legacy("/fake.spz")
            assert result == {"rows": 5, "cols": 10, "nnz": 50}
            mock_sp.sp_info.assert_called_once_with("/fake.spz")
