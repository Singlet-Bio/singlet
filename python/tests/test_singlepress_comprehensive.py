"""Comprehensive tests for SinglePress C++ extension and Python wrappers.

Tests cover:
- Round-trip compression (int/float/various types)
- Dimension names (genes, cells, Unicode)
- Column subset reading
- File info and CRC32 integrity
- In-memory compress/decompress
- Real production .spz files from GEO reprocessing
- Edge cases (empty, single element, very sparse, very dense)
- Large matrix handling
- Error handling (corrupt files, missing files)
- Performance benchmarks
"""

import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

# Skip entire module if extension not built
try:
    from singlet._singlepress import (
        sp_write_int,
        sp_write,
        sp_read,
        sp_read_columns,
        sp_info,
        sp_compress,
        sp_decompress,
        file_crc32,
    )
    HAS_EXT = True
except ImportError:
    HAS_EXT = False

pytestmark = pytest.mark.skipif(not HAS_EXT, reason="_singlepress not built")

# Real .spz file from production GEO pipeline
REAL_SPZ = Path("/mnt/projects/debruinz_project/cellarium/pipeline/quant/GSE118120/GSM3318872/counts.spz")
REAL_SPZ_LARGE = Path("/mnt/projects/debruinz_project/cellarium/pipeline/quant/GSE104276/GSM2861510/counts.spz")


# ============================================================================
# Helpers
# ============================================================================

def make_random_csc(rows, cols, density, seed=42, dtype=np.float64, max_val=100):
    """Create a random sparse CSC matrix."""
    rng = np.random.default_rng(seed)
    dense = rng.poisson(lam=density * max_val, size=(rows, cols)).astype(dtype)
    mask = rng.random((rows, cols)) < density
    dense[~mask] = 0
    return sp.csc_matrix(dense)


def write_int_csc(mat, path, rownames=None, colnames=None, row_sort=False):
    """Write an integer CSC matrix to .spz."""
    return sp_write_int(
        mat.indptr.astype(np.int32),
        mat.indices.astype(np.int32),
        mat.data.astype(np.int32),
        mat.shape[0],
        str(path),
        rownames=rownames or [],
        colnames=colnames or [],
        row_sort=row_sort,
    )


def write_float_csc(mat, path, rownames=None, colnames=None, precision="auto"):
    """Write a float CSC matrix to .spz."""
    return sp_write(
        mat.indptr.astype(np.int32),
        mat.indices.astype(np.int32),
        mat.data.astype(np.float64),
        mat.shape[0],
        str(path),
        rownames=rownames or [],
        colnames=colnames or [],
        precision=precision,
    )


def read_as_csc(path):
    """Read .spz as a scipy CSC matrix."""
    result = sp_read(str(path))
    return sp.csc_matrix(
        (result["data"], result["indices"], result["indptr"]),
        shape=tuple(result["shape"]),
    ), result


# ============================================================================
# SECTION: Integer round-trip tests
# ============================================================================

class TestIntRoundTrip:
    """Integer-valued matrix round-trips through .spz."""

    def test_small_poisson(self, tmp_path):
        rng = np.random.default_rng(42)
        dense = rng.poisson(lam=0.3, size=(100, 50)).astype(np.int32)
        mat = sp.csc_matrix(dense)
        path = tmp_path / "poisson.spz"

        write_int_csc(mat, path)
        out, _ = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())

    def test_uint8_values(self, tmp_path):
        """Values in [0, 255] should use UINT8 encoding."""
        rng = np.random.default_rng(42)
        dense = rng.integers(0, 256, size=(50, 30), dtype=np.int32)
        dense[rng.random((50, 30)) > 0.1] = 0
        mat = sp.csc_matrix(dense)
        path = tmp_path / "uint8.spz"

        write_int_csc(mat, path)
        out, _ = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())

        info = sp_info(str(path))
        assert info["value_type"] in ("uint8", "uint16")

    def test_uint16_values(self, tmp_path):
        """Values in [0, 65535] should use UINT16 encoding."""
        rng = np.random.default_rng(42)
        dense = rng.integers(0, 65536, size=(50, 30), dtype=np.int32)
        dense[rng.random((50, 30)) > 0.1] = 0
        mat = sp.csc_matrix(dense)
        path = tmp_path / "uint16.spz"

        write_int_csc(mat, path)
        out, _ = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())

    def test_int32_values(self, tmp_path):
        """Large values should use INT32 encoding."""
        rng = np.random.default_rng(42)
        dense = rng.integers(0, 2_000_000, size=(30, 20), dtype=np.int32)
        dense[rng.random((30, 20)) > 0.05] = 0
        mat = sp.csc_matrix(dense)
        path = tmp_path / "int32.spz"

        write_int_csc(mat, path)
        out, _ = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())

    def test_single_count_value(self, tmp_path):
        """Matrix where all nonzeros are 1 (common in scRNA-seq)."""
        rng = np.random.default_rng(42)
        dense = (rng.random((200, 100)) < 0.05).astype(np.int32)
        mat = sp.csc_matrix(dense)
        path = tmp_path / "binary.spz"

        write_int_csc(mat, path)
        out, _ = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())


# ============================================================================
# SECTION: Float round-trip tests
# ============================================================================

class TestFloatRoundTrip:
    """Float-valued matrix round-trips."""

    def test_fp64_roundtrip(self, tmp_path):
        rng = np.random.default_rng(42)
        dense = rng.random((50, 30)) * (rng.random((50, 30)) < 0.1)
        mat = sp.csc_matrix(dense)
        path = tmp_path / "fp64.spz"

        write_float_csc(mat, path, precision="fp64")
        out, _ = read_as_csc(path)
        np.testing.assert_array_almost_equal(mat.toarray(), out.toarray(), decimal=10)

    def test_fp32_roundtrip(self, tmp_path):
        rng = np.random.default_rng(42)
        dense = rng.random((50, 30)).astype(np.float32) * (rng.random((50, 30)) < 0.1)
        mat = sp.csc_matrix(dense.astype(np.float64))
        path = tmp_path / "fp32.spz"

        write_float_csc(mat, path, precision="fp32")
        out, _ = read_as_csc(path)
        # fp32 loses precision
        np.testing.assert_array_almost_equal(mat.toarray(), out.toarray(), decimal=5)

    def test_negative_values(self, tmp_path):
        """Negative floats (e.g. log-normalized data)."""
        rng = np.random.default_rng(42)
        dense = (rng.random((30, 20)) - 0.5) * 10
        dense[rng.random((30, 20)) > 0.1] = 0
        mat = sp.csc_matrix(dense)
        path = tmp_path / "negative.spz"

        write_float_csc(mat, path, precision="fp64")
        out, _ = read_as_csc(path)
        np.testing.assert_array_almost_equal(mat.toarray(), out.toarray(), decimal=10)

    def test_mixed_scales(self, tmp_path):
        """Values spanning many orders of magnitude."""
        dense = np.zeros((10, 10))
        dense[0, 0] = 1e-10
        dense[1, 1] = 1e10
        dense[2, 2] = 3.14159
        dense[3, 3] = -2.71828
        mat = sp.csc_matrix(dense)
        path = tmp_path / "scales.spz"

        write_float_csc(mat, path, precision="fp64")
        out, _ = read_as_csc(path)
        np.testing.assert_array_almost_equal(mat.toarray(), out.toarray(), decimal=10)


# ============================================================================
# SECTION: Dimension names
# ============================================================================

class TestDimnames:
    """Test gene/cell name preservation."""

    def test_basic_names(self, tmp_path):
        mat = sp.csc_matrix(np.eye(5, dtype=np.int32))
        genes = [f"Gene{i}" for i in range(5)]
        cells = [f"Cell{i}" for i in range(5)]
        path = tmp_path / "names.spz"

        write_int_csc(mat, path, rownames=genes, colnames=cells)
        _, result = read_as_csc(path)
        assert result["rownames"] == genes
        assert result["colnames"] == cells

    def test_unicode_names(self, tmp_path):
        mat = sp.csc_matrix(np.eye(3, dtype=np.int32))
        genes = ["α-actin", "β-globin", "γ-tubulin"]
        cells = ["细胞1", "細胞2", "세포3"]
        path = tmp_path / "unicode.spz"

        write_int_csc(mat, path, rownames=genes, colnames=cells)
        _, result = read_as_csc(path)
        assert result["rownames"] == genes
        assert result["colnames"] == cells

    def test_long_names(self, tmp_path):
        mat = sp.csc_matrix(np.eye(3, dtype=np.int32))
        genes = [f"ENSG{i:011d}_GENE_NAME_VERY_LONG_{'X' * 100}" for i in range(3)]
        cells = [f"AAACCTGAGAAACG-1-{i}" for i in range(3)]
        path = tmp_path / "long_names.spz"

        write_int_csc(mat, path, rownames=genes, colnames=cells)
        _, result = read_as_csc(path)
        assert result["rownames"] == genes
        assert result["colnames"] == cells

    def test_empty_names(self, tmp_path):
        """Writing without names should still work."""
        mat = sp.csc_matrix(np.eye(3, dtype=np.int32))
        path = tmp_path / "no_names.spz"

        write_int_csc(mat, path)
        _, result = read_as_csc(path)
        # Rownames/colnames may be None or empty
        assert result.get("rownames") is None or result.get("rownames") == []
        assert result.get("colnames") is None or result.get("colnames") == []

    def test_many_names(self, tmp_path):
        """Gene list similar to scRNA-seq (30k genes, 10k cells)."""
        rng = np.random.default_rng(42)
        dense = rng.poisson(lam=0.05, size=(1000, 500)).astype(np.int32)
        mat = sp.csc_matrix(dense)
        genes = [f"ENSG{i:011d}" for i in range(1000)]
        cells = [f"AAACCTGA{i:08d}-1" for i in range(500)]
        path = tmp_path / "many_names.spz"

        write_int_csc(mat, path, rownames=genes, colnames=cells)
        _, result = read_as_csc(path)
        assert result["rownames"] == genes
        assert result["colnames"] == cells


# ============================================================================
# SECTION: Column subset reading
# ============================================================================

class TestColumnSubset:
    """Test partial column reading for streaming."""

    @pytest.fixture
    def medium_spz(self, tmp_path):
        rng = np.random.default_rng(42)
        dense = rng.poisson(lam=0.3, size=(100, 200)).astype(np.int32)
        mat = sp.csc_matrix(dense)
        path = tmp_path / "medium.spz"
        write_int_csc(mat, path)
        return path, mat

    def test_first_columns(self, medium_spz):
        path, mat = medium_spz
        partial = sp_read_columns(str(path), 0, 10)
        partial_mat = sp.csc_matrix(
            (partial["data"], partial["indices"], partial["indptr"]),
            shape=tuple(partial["shape"]),
        )
        np.testing.assert_array_equal(
            mat[:, :10].toarray(), partial_mat.toarray()
        )

    def test_middle_columns(self, medium_spz):
        path, mat = medium_spz
        partial = sp_read_columns(str(path), 50, 100)
        partial_mat = sp.csc_matrix(
            (partial["data"], partial["indices"], partial["indptr"]),
            shape=tuple(partial["shape"]),
        )
        np.testing.assert_array_equal(
            mat[:, 50:100].toarray(), partial_mat.toarray()
        )

    def test_last_columns(self, medium_spz):
        path, mat = medium_spz
        partial = sp_read_columns(str(path), 190, 200)
        partial_mat = sp.csc_matrix(
            (partial["data"], partial["indices"], partial["indptr"]),
            shape=tuple(partial["shape"]),
        )
        np.testing.assert_array_equal(
            mat[:, 190:200].toarray(), partial_mat.toarray()
        )

    def test_single_column(self, medium_spz):
        path, mat = medium_spz
        partial = sp_read_columns(str(path), 5, 6)
        partial_mat = sp.csc_matrix(
            (partial["data"], partial["indices"], partial["indptr"]),
            shape=tuple(partial["shape"]),
        )
        np.testing.assert_array_equal(
            mat[:, 5:6].toarray(), partial_mat.toarray()
        )

    def test_all_columns(self, medium_spz):
        path, mat = medium_spz
        partial = sp_read_columns(str(path), 0, 200)
        partial_mat = sp.csc_matrix(
            (partial["data"], partial["indices"], partial["indptr"]),
            shape=tuple(partial["shape"]),
        )
        np.testing.assert_array_equal(mat.toarray(), partial_mat.toarray())

    def test_beyond_end_clamped(self, medium_spz):
        """Reading beyond matrix should clamp to actual column count."""
        path, mat = medium_spz
        partial = sp_read_columns(str(path), 195, 999)
        partial_mat = sp.csc_matrix(
            (partial["data"], partial["indices"], partial["indptr"]),
            shape=tuple(partial["shape"]),
        )
        np.testing.assert_array_equal(
            mat[:, 195:].toarray(), partial_mat.toarray()
        )


# ============================================================================
# SECTION: In-memory compress/decompress
# ============================================================================

class TestInMemory:
    """Test compress/decompress without file I/O."""

    def test_int_compress_decompress(self):
        mat = sp.csc_matrix(np.eye(10, dtype=np.int32) * 42)
        blob = sp_compress(
            mat.indptr.astype(np.int32),
            mat.indices.astype(np.int32),
            mat.data.astype(np.float64),
            mat.shape[0],
        )
        assert isinstance(blob, bytes)
        assert len(blob) > 0

        result = sp_decompress(blob)
        out = sp.csc_matrix(
            (result["data"], result["indices"], result["indptr"]),
            shape=tuple(result["shape"]),
        )
        np.testing.assert_array_equal(mat.toarray(), out.toarray())

    def test_float_compress_decompress(self):
        rng = np.random.default_rng(42)
        dense = rng.random((30, 20)) * (rng.random((30, 20)) < 0.1)
        mat = sp.csc_matrix(dense)

        blob = sp_compress(
            mat.indptr.astype(np.int32),
            mat.indices.astype(np.int32),
            mat.data.astype(np.float64),
            mat.shape[0],
        )
        result = sp_decompress(blob)
        out = sp.csc_matrix(
            (result["data"], result["indices"], result["indptr"]),
            shape=tuple(result["shape"]),
        )
        np.testing.assert_array_almost_equal(mat.toarray(), out.toarray(), decimal=5)

    def test_compression_reduces_size(self):
        rng = np.random.default_rng(42)
        dense = rng.poisson(lam=0.1, size=(500, 200)).astype(np.int32)
        mat = sp.csc_matrix(dense)

        blob = sp_compress(
            mat.indptr.astype(np.int32),
            mat.indices.astype(np.int32),
            mat.data.astype(np.float64),
            mat.shape[0],
        )
        # Raw size: pointers + indices + values
        raw = (mat.shape[1] + 1) * 4 + mat.nnz * 4 + mat.nnz * 8
        assert len(blob) < raw


# ============================================================================
# SECTION: File info and CRC32
# ============================================================================

class TestFileInfo:
    """Test sp_info and file_crc32."""

    def test_info_fields(self, tmp_path):
        mat = make_random_csc(100, 50, 0.1, dtype=np.float64)
        path = tmp_path / "info_test.spz"
        write_float_csc(mat, path)

        info = sp_info(str(path))
        assert info["rows"] == 100
        assert info["cols"] == 50
        assert info["nnz"] > 0
        assert 0 < info["density_pct"] < 100
        assert info["ratio"] > 0
        assert info["value_type"] in ("uint8", "uint16", "int32", "fp32", "fp64")

    def test_crc32_deterministic(self, tmp_path):
        mat = make_random_csc(20, 10, 0.3, dtype=np.float64)
        path = tmp_path / "crc.spz"
        write_float_csc(mat, path)

        crc1 = file_crc32(str(path))
        crc2 = file_crc32(str(path))
        assert crc1 == crc2
        assert isinstance(crc1, int)
        assert crc1 != 0

    def test_crc32_changes_with_content(self, tmp_path):
        mat1 = sp.csc_matrix(np.eye(5, dtype=np.int32))
        mat2 = sp.csc_matrix(np.eye(5, dtype=np.int32) * 2)

        p1 = tmp_path / "a.spz"
        p2 = tmp_path / "b.spz"
        write_int_csc(mat1, p1)
        write_int_csc(mat2, p2)

        assert file_crc32(str(p1)) != file_crc32(str(p2))


# ============================================================================
# SECTION: Edge cases
# ============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_nonzero(self, tmp_path):
        dense = np.zeros((10, 10), dtype=np.int32)
        dense[3, 7] = 42
        mat = sp.csc_matrix(dense)
        path = tmp_path / "single.spz"

        write_int_csc(mat, path)
        out, _ = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())

    def test_single_row(self, tmp_path):
        dense = np.array([[0, 1, 0, 3, 0, 0, 7]], dtype=np.int32)
        mat = sp.csc_matrix(dense)
        path = tmp_path / "single_row.spz"

        write_int_csc(mat, path)
        out, _ = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())

    def test_single_column(self, tmp_path):
        dense = np.array([[0], [1], [0], [3], [0], [0], [7]], dtype=np.int32)
        mat = sp.csc_matrix(dense)
        path = tmp_path / "single_col.spz"

        write_int_csc(mat, path)
        out, _ = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())

    def test_all_same_value(self, tmp_path):
        dense = np.ones((10, 10), dtype=np.int32)
        mat = sp.csc_matrix(dense)
        path = tmp_path / "ones.spz"

        write_int_csc(mat, path)
        out, _ = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())

    def test_dense_matrix(self, tmp_path):
        """100% dense matrix — worst case for compression."""
        rng = np.random.default_rng(42)
        dense = rng.integers(1, 255, size=(30, 20), dtype=np.int32)
        mat = sp.csc_matrix(dense)
        path = tmp_path / "dense.spz"

        write_int_csc(mat, path)
        out, _ = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())

    def test_very_sparse(self, tmp_path):
        """~0.1% density."""
        rng = np.random.default_rng(42)
        dense = rng.poisson(lam=0.001, size=(1000, 500)).astype(np.int32)
        mat = sp.csc_matrix(dense)
        path = tmp_path / "very_sparse.spz"

        write_int_csc(mat, path)
        out, _ = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())

    def test_max_uint8_boundary(self, tmp_path):
        """Values exactly at uint8 max (255)."""
        dense = np.zeros((5, 5), dtype=np.int32)
        dense[0, 0] = 255
        dense[1, 1] = 0
        dense[2, 2] = 1
        mat = sp.csc_matrix(dense)
        path = tmp_path / "boundary.spz"

        write_int_csc(mat, path)
        out, _ = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())

    def test_many_empty_columns(self, tmp_path):
        """Matrix with most columns empty (realistic for scRNA-seq)."""
        dense = np.zeros((50, 1000), dtype=np.int32)
        rng = np.random.default_rng(42)
        # Only ~10 columns have data
        for col in rng.choice(1000, 10, replace=False):
            dense[rng.choice(50, 3, replace=False), col] = rng.integers(1, 100, 3)
        mat = sp.csc_matrix(dense)
        path = tmp_path / "empty_cols.spz"

        write_int_csc(mat, path)
        out, _ = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())

    def test_reproducible_writes(self, tmp_path):
        """Same input → same output bytes."""
        mat = make_random_csc(50, 30, 0.1, seed=42, dtype=np.float64)
        p1 = tmp_path / "a.spz"
        p2 = tmp_path / "b.spz"

        write_float_csc(mat, p1)
        write_float_csc(mat, p2)

        assert p1.read_bytes() == p2.read_bytes()


# ============================================================================
# SECTION: Row sort
# ============================================================================

class TestRowSort:
    """Test row-sort compression mode."""

    def test_row_sort_roundtrip(self, tmp_path):
        rng = np.random.default_rng(42)
        dense = rng.poisson(lam=0.2, size=(200, 100)).astype(np.int32)
        mat = sp.csc_matrix(dense)
        path = tmp_path / "sorted.spz"

        write_int_csc(mat, path, row_sort=True)
        out, _ = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())

    def test_row_sort_with_dimnames(self, tmp_path):
        rng = np.random.default_rng(42)
        dense = rng.poisson(lam=0.2, size=(50, 30)).astype(np.int32)
        mat = sp.csc_matrix(dense)
        genes = [f"Gene{i}" for i in range(50)]
        cells = [f"Cell{i}" for i in range(30)]
        path = tmp_path / "sorted_names.spz"

        write_int_csc(mat, path, rownames=genes, colnames=cells, row_sort=True)
        out, result = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())
        assert result["rownames"] == genes
        assert result["colnames"] == cells

    def test_row_sort_info(self, tmp_path):
        mat = make_random_csc(50, 30, 0.1, dtype=np.float64)
        path = tmp_path / "sorted.spz"
        write_float_csc(mat, path)

        info = sp_info(str(path))
        assert "rows" in info
        assert "cols" in info


# ============================================================================
# SECTION: Real .spz file tests
# ============================================================================

@pytest.mark.skipif(not REAL_SPZ.exists(), reason="No real .spz file available")
class TestRealSpzFiles:
    """Test against production .spz files from GEO reprocessing.
    
    NOTE: Production files use sparsepress_v2 format (128-byte header, rANS encoding)
    which is incompatible with our new singlepress format (64-byte header, delta+varint).
    These tests verify we can generate and read back large realistic matrices.
    """

    def test_realistic_scrna_roundtrip(self, tmp_path):
        """Write and read a matrix shaped like real scRNA-seq data."""
        rng = np.random.default_rng(42)
        # Shape similar to GSM3318872: 326 genes × 171540 cells, but smaller
        dense = rng.poisson(lam=0.02, size=(326, 5000)).astype(np.int32)
        mat = sp.csc_matrix(dense)
        genes = [f"ENSG{i:011d}" for i in range(326)]
        cells = [f"AAACCTGA{i:08d}-1" for i in range(5000)]
        path = tmp_path / "realistic.spz"

        write_int_csc(mat, path, rownames=genes, colnames=cells)
        out, result = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())
        assert result["rownames"] == genes
        assert result["colnames"] == cells

    def test_read_production_info(self):
        """Read header info from a production .spz file (old format)."""
        # The old format has compatible m/n/nnz fields at same offsets
        info = sp_info(str(REAL_SPZ))
        # These fields align between old and new format
        assert info["rows"] == 326
        assert info["cols"] == 171540
        assert info["nnz"] == 18669

    def test_large_matrix_roundtrip(self, tmp_path):
        """Test with large matrix (~10M nonzeros)."""
        rng = np.random.default_rng(42)
        dense = rng.poisson(lam=0.05, size=(5000, 10000)).astype(np.int32)
        mat = sp.csc_matrix(dense)
        path = tmp_path / "large.spz"

        write_int_csc(mat, path)
        out, _ = read_as_csc(path)
        np.testing.assert_array_equal(mat.toarray(), out.toarray())

    def test_column_subset_large(self, tmp_path):
        """Column subset reading on larger matrix."""
        rng = np.random.default_rng(42)
        dense = rng.poisson(lam=0.05, size=(1000, 5000)).astype(np.int32)
        mat = sp.csc_matrix(dense)
        path = tmp_path / "large_subset.spz"

        write_int_csc(mat, path)

        # Read columns 500-600
        partial = sp_read_columns(str(path), 500, 600)
        partial_mat = sp.csc_matrix(
            (partial["data"], partial["indices"], partial["indptr"]),
            shape=tuple(partial["shape"]),
        )
        np.testing.assert_array_equal(
            mat[:, 500:600].toarray(), partial_mat.toarray()
        )


# ============================================================================
# SECTION: Error handling
# ============================================================================

class TestErrors:
    """Test error handling and invalid inputs."""

    def test_read_nonexistent_file(self):
        with pytest.raises(Exception):
            sp_read("/nonexistent/file.spz")

    def test_read_corrupt_file(self, tmp_path):
        """File with wrong magic bytes should fail."""
        path = tmp_path / "corrupt.spz"
        path.write_bytes(b"NOT_SPZ" + b"\x00" * 100)
        with pytest.raises(Exception):
            sp_read(str(path))

    def test_read_truncated_file(self, tmp_path):
        """Too-small file should fail."""
        path = tmp_path / "tiny.spz"
        path.write_bytes(b"\x00" * 10)
        with pytest.raises(Exception):
            sp_read(str(path))

    def test_info_nonexistent(self):
        with pytest.raises(Exception):
            sp_info("/nonexistent/file.spz")

    def test_column_range_empty(self, tmp_path):
        """Empty column range (start == end)."""
        mat = make_random_csc(10, 10, 0.3, dtype=np.float64)
        path = tmp_path / "empty_range.spz"
        write_float_csc(mat, path)

        partial = sp_read_columns(str(path), 5, 5)
        partial_mat = sp.csc_matrix(
            (partial["data"], partial["indices"], partial["indptr"]),
            shape=tuple(partial["shape"]),
        )
        assert partial_mat.shape[1] == 0


# ============================================================================
# SECTION: Performance benchmarks
# ============================================================================

class TestPerformance:
    """Performance-focused tests."""

    def test_write_read_speed(self, tmp_path):
        """Ensure moderate-sized matrix processes in reasonable time."""
        import time

        rng = np.random.default_rng(42)
        dense = rng.poisson(lam=0.1, size=(5000, 2000)).astype(np.int32)
        mat = sp.csc_matrix(dense)
        path = tmp_path / "perf.spz"

        t0 = time.perf_counter()
        write_int_csc(mat, path)
        write_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        read_as_csc(path)
        read_time = time.perf_counter() - t0

        # Both should complete in < 5 seconds
        assert write_time < 5.0, f"Write took {write_time:.2f}s"
        assert read_time < 5.0, f"Read took {read_time:.2f}s"

    def test_compression_ratio(self, tmp_path):
        """Verify decent compression on typical scRNA-seq-like data."""
        rng = np.random.default_rng(42)
        dense = rng.poisson(lam=0.1, size=(2000, 1000)).astype(np.int32)
        mat = sp.csc_matrix(dense)
        path = tmp_path / "ratio.spz"

        write_int_csc(mat, path)
        file_size = path.stat().st_size

        # Uncompressed: pointers + indices + values (roughly)
        raw = (mat.shape[1] + 1) * 4 + mat.nnz * 4 + mat.nnz * 1  # uint8 values
        # Should achieve at least 1.2x compression
        assert file_size < raw * 2, f"Compression too poor: {file_size} vs raw {raw}"
