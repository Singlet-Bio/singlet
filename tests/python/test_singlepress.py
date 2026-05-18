# SPDX-License-Identifier: MIT
"""Tests for SinglePress C++ extension (compression round-trips)."""

import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("singlet._singlepress", reason="_singlepress not built"),
    reason="_singlepress extension not built",
)


@pytest.fixture
def tmp_spz(tmp_path):
    """Create a temporary SPZ file for tests that need a pre-written file."""
    from singlet._singlepress import sp_write_int

    rng = np.random.default_rng(42)
    mat = sp.csc_matrix(rng.poisson(1, size=(5, 10)).astype(np.int32))
    path = tmp_path / "fixture.spz"
    sp_write_int(
        mat.indptr.astype(np.int32),
        mat.indices.astype(np.int32),
        mat.data.astype(np.int32),
        mat.shape[0],
        str(path),
    )
    return path


class TestRoundTrip:
    """Test compress → decompress preserves data."""

    def test_int_roundtrip(self, tmp_path):
        from singlet._singlepress import sp_read, sp_write_int

        rng = np.random.default_rng(42)
        dense = rng.poisson(lam=0.3, size=(100, 50)).astype(np.int32)
        mat = sp.csc_matrix(dense)

        path = str(tmp_path / "test.spz")
        sp_write_int(
            mat.indptr.astype(np.int32),
            mat.indices.astype(np.int32),
            mat.data.astype(np.int32),
            mat.shape[0],
            path,
        )

        result = sp_read(path)
        out = sp.csc_matrix(
            (result["data"], result["indices"], result["indptr"]),
            shape=tuple(result["shape"]),
        )

        np.testing.assert_array_equal(mat.toarray(), out.toarray())

    def test_float_roundtrip(self, tmp_path):
        from singlet._singlepress import sp_read, sp_write

        rng = np.random.default_rng(42)
        dense = rng.random((50, 30)) * (rng.random((50, 30)) < 0.1)
        mat = sp.csc_matrix(dense)

        path = str(tmp_path / "test_float.spz")
        sp_write(
            mat.indptr.astype(np.int32),
            mat.indices.astype(np.int32),
            mat.data.astype(np.float64),
            mat.shape[0],
            path,
        )

        result = sp_read(path)
        out = sp.csc_matrix(
            (result["data"], result["indices"], result["indptr"]),
            shape=tuple(result["shape"]),
        )

        np.testing.assert_array_almost_equal(mat.toarray(), out.toarray(), decimal=5)

    def test_dimnames_preserved(self, tmp_path):
        from singlet._singlepress import sp_read, sp_write_int

        rng = np.random.default_rng(42)
        mat = sp.csc_matrix(rng.poisson(1, size=(10, 5)).astype(np.int32))
        genes = [f"Gene{i}" for i in range(10)]
        cells = [f"Cell{i}" for i in range(5)]

        path = str(tmp_path / "names.spz")
        sp_write_int(
            mat.indptr.astype(np.int32),
            mat.indices.astype(np.int32),
            mat.data.astype(np.int32),
            mat.shape[0],
            path,
            rownames=genes,
            colnames=cells,
        )

        result = sp_read(path)
        assert result["rownames"] == genes
        assert result["colnames"] == cells

    def test_in_memory_roundtrip(self):
        from singlet._singlepress import sp_compress, sp_decompress

        rng = np.random.default_rng(42)
        mat = sp.csc_matrix(rng.poisson(1, size=(20, 15)).astype(np.float64))

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

        np.testing.assert_array_equal(mat.toarray(), out.toarray())


class TestInfo:
    """Test sp_info reads header correctly."""

    def test_info(self, tmp_spz):
        from singlet._singlepress import sp_info

        info = sp_info(str(tmp_spz))
        assert info["rows"] == 5
        assert info["cols"] == 10
        assert info["nnz"] > 0
        assert info["ratio"] > 0

    def test_file_crc32(self, tmp_spz):
        from singlet._singlepress import file_crc32

        crc = file_crc32(str(tmp_spz))
        assert isinstance(crc, int)
        assert crc > 0


class TestColumnRead:
    """Test partial column reading."""

    def test_column_subset(self, tmp_spz):
        from singlet._singlepress import sp_read, sp_read_columns

        full = sp_read(str(tmp_spz))
        full_mat = sp.csc_matrix(
            (full["data"], full["indices"], full["indptr"]),
            shape=tuple(full["shape"]),
        )

        # Read columns 2-5
        partial = sp_read_columns(str(tmp_spz), 2, 5)
        partial_mat = sp.csc_matrix(
            (partial["data"], partial["indices"], partial["indptr"]),
            shape=tuple(partial["shape"]),
        )

        np.testing.assert_array_equal(
            full_mat[:, 2:5].toarray(),
            partial_mat.toarray(),
        )
