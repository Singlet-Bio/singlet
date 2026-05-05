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
