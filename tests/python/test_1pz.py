"""Comprehensive tests for .1pz format.

Tests:
  1. Round-trip (write → read), exact value recovery
  2. CRC32 integrity (validate, corrupt detection)
  3. Metadata (rownames, colnames round-trip)
  4. Column sums (auto-computed, match expected)
  5. Column-range reads (partial decode correctness)
  6. Native int32 read path
  7. Log-normalization (lognorm function)
  8. OnePZFile lazy handle
  9. Transpose storage + row-range reads
"""

import numpy as np
import pytest
import scipy.sparse as ss
import singlepress


@pytest.fixture
def sample_matrix():
    """Create a reproducible sparse integer matrix."""
    rng = np.random.RandomState(42)
    m, n = 500, 200
    density = 0.05
    nnz = int(m * n * density)
    rows = rng.randint(0, m, nnz)
    cols = rng.randint(0, n, nnz)
    vals = rng.randint(1, 100, nnz).astype(np.int32)
    mat = ss.csc_matrix((vals, (rows, cols)), shape=(m, n))
    mat.sum_duplicates()
    return mat


@pytest.fixture
def gene_names():
    return [f"Gene_{i}" for i in range(500)]


@pytest.fixture
def cell_barcodes():
    return [f"CELL_{i:04d}" for i in range(200)]


class TestRoundTrip:
    def test_write_read_roundtrip(self, sample_matrix, tmp_path):
        path = str(tmp_path / "test.1pz")
        stats = singlepress.write_1pz(path, sample_matrix)
        assert stats["m"] == 500
        assert stats["n"] == 200
        assert stats["has_colsums"] is True

        mat2 = singlepress.read_1pz(path)
        assert mat2.shape == sample_matrix.shape
        assert mat2.nnz == sample_matrix.nnz
        diff = (sample_matrix - mat2).data
        assert np.all(diff == 0), "Values don't match after round-trip"

    def test_int32_roundtrip(self, sample_matrix, tmp_path):
        path = str(tmp_path / "test_int.1pz")
        singlepress.write_1pz(path, sample_matrix)
        mat2 = singlepress.read_1pz_int(path)
        assert mat2.dtype == np.int32
        assert mat2.shape == sample_matrix.shape
        diff = sample_matrix.astype(np.float64) - mat2.astype(np.float64)
        assert np.all(diff.data == 0)

    def test_empty_matrix(self, tmp_path):
        mat = ss.csc_matrix((100, 50), dtype=np.float64)
        path = str(tmp_path / "empty.1pz")
        singlepress.write_1pz(path, mat)
        mat2 = singlepress.read_1pz(path)
        assert mat2.shape == (100, 50)
        assert mat2.nnz == 0


class TestCRC:
    def test_validate_valid_file(self, sample_matrix, tmp_path):
        path = str(tmp_path / "valid.1pz")
        singlepress.write_1pz(path, sample_matrix)
        result = singlepress.validate_1pz(path)
        assert result["valid"] is True
        assert result["file_crc_ok"] is True
        assert result["footer_ok"] is True

    def test_detect_corruption(self, sample_matrix, tmp_path):
        path = str(tmp_path / "corrupt.1pz")
        singlepress.write_1pz(path, sample_matrix)
        # Corrupt a byte in the middle of the file
        with open(path, "r+b") as f:
            f.seek(200)
            f.write(b"\xff")
        result = singlepress.validate_1pz(path)
        assert result["valid"] is False or result["file_crc_ok"] is False


class TestMetadata:
    def test_rownames_colnames(self, sample_matrix, gene_names, cell_barcodes, tmp_path):
        path = str(tmp_path / "meta.1pz")
        singlepress.write_1pz(
            path,
            sample_matrix,
            rownames=gene_names,
            colnames=cell_barcodes,
        )
        info = singlepress.info_1pz(path)
        assert info["has_metadata"] is True

        mat2 = singlepress.read_1pz(path)
        assert hasattr(mat2, "rownames")
        assert mat2.rownames == gene_names
        assert hasattr(mat2, "colnames")
        assert mat2.colnames == cell_barcodes

    def test_no_metadata(self, sample_matrix, tmp_path):
        path = str(tmp_path / "nometa.1pz")
        singlepress.write_1pz(path, sample_matrix)
        info = singlepress.info_1pz(path)
        # No rownames/colnames passed => has_metadata should be False
        assert info["has_metadata"] is False


class TestColsums:
    def test_colsums_match(self, sample_matrix, tmp_path):
        path = str(tmp_path / "cs.1pz")
        singlepress.write_1pz(path, sample_matrix)

        # Read colsums from file
        cs = singlepress.colsums_1pz(path)
        assert len(cs) == sample_matrix.shape[1]

        # Compare with manually computed column sums
        expected = np.array(sample_matrix.sum(axis=0)).ravel()
        np.testing.assert_array_equal(cs, expected.astype(np.uint64))

    def test_colsums_from_read(self, sample_matrix, tmp_path):
        path = str(tmp_path / "cs2.1pz")
        singlepress.write_1pz(path, sample_matrix)
        mat2 = singlepress.read_1pz(path)
        assert hasattr(mat2, "colsums")
        expected = np.array(sample_matrix.sum(axis=0)).ravel()
        np.testing.assert_array_equal(mat2.colsums, expected.astype(np.uint64))


class TestColumnRange:
    def test_column_range_read(self, sample_matrix, tmp_path):
        path = str(tmp_path / "range.1pz")
        singlepress.write_1pz(path, sample_matrix)

        sub = singlepress.read_1pz_columns(path, 10, 50)
        expected = sample_matrix[:, 10:50]
        assert sub.shape == expected.shape
        diff = (expected - sub).data
        assert np.all(diff == 0)

    def test_column_range_full(self, sample_matrix, tmp_path):
        path = str(tmp_path / "range_full.1pz")
        singlepress.write_1pz(path, sample_matrix)
        sub = singlepress.read_1pz_columns(path, 0, sample_matrix.shape[1])
        diff = (sample_matrix - sub).data
        assert np.all(diff == 0)


class TestLognorm:
    def test_lognorm_scalar(self):
        result = singlepress.lognorm(5, np.array([100]), scale=10000)
        expected = np.log1p(5 * 10000 / 100)
        np.testing.assert_almost_equal(result, expected)

    def test_lognorm_sparse(self, sample_matrix, tmp_path):
        path = str(tmp_path / "ln.1pz")
        singlepress.write_1pz(path, sample_matrix)
        cs = singlepress.colsums_1pz(path)

        norm = singlepress.lognorm(sample_matrix.astype(np.float64), cs, scale=10000)
        assert norm.shape == sample_matrix.shape
        # Check a few values manually
        for j in range(min(5, sample_matrix.shape[1])):
            s, e = sample_matrix.indptr[j], sample_matrix.indptr[j + 1]
            if s < e:
                expected = np.log1p(sample_matrix.data[s:e].astype(np.float64) * 10000 / cs[j])
                np.testing.assert_allclose(norm.data[s:e], expected, rtol=1e-12)


class TestOnePZFile:
    def test_lazy_access(self, sample_matrix, gene_names, cell_barcodes, tmp_path):
        path = str(tmp_path / "lazy.1pz")
        singlepress.write_1pz(
            path,
            sample_matrix,
            rownames=gene_names,
            colnames=cell_barcodes,
        )

        pz = singlepress.open_1pz(path)
        assert pz.shape == (500, 200)
        assert pz.nnz == sample_matrix.nnz
        assert pz.has_colsums is True
        assert pz.has_metadata is True
        assert len(pz.colsums) == 200
        assert pz.rownames == gene_names
        assert pz.colnames == cell_barcodes

    def test_partial_read(self, sample_matrix, tmp_path):
        path = str(tmp_path / "partial.1pz")
        singlepress.write_1pz(path, sample_matrix)

        pz = singlepress.open_1pz(path)
        sub = pz.read(cols=(0, 10))
        expected = sample_matrix[:, :10]
        assert sub.shape == expected.shape
        diff = (expected - sub).data
        assert np.all(diff == 0)

    def test_read_normalized(self, sample_matrix, tmp_path):
        path = str(tmp_path / "norm.1pz")
        singlepress.write_1pz(path, sample_matrix)

        pz = singlepress.open_1pz(path)
        norm = pz.read_normalized(scale=10000)
        assert norm.shape == sample_matrix.shape
        # All normalized values should be non-negative
        assert np.all(norm.data >= 0)

    def test_repr(self, sample_matrix, tmp_path):
        path = str(tmp_path / "repr.1pz")
        singlepress.write_1pz(path, sample_matrix)
        pz = singlepress.open_1pz(path)
        r = repr(pz)
        assert "OnePZFile" in r
        assert "500" in r


class TestTranspose:
    def test_transpose_storage(self, sample_matrix, tmp_path):
        path = str(tmp_path / "trans.1pz")
        stats = singlepress.write_1pz(path, sample_matrix, store_transpose=True)
        assert stats["has_transpose"] is True

        info = singlepress.info_1pz(path)
        assert info["has_transpose"] is True

        # File should be larger than without transpose
        path2 = str(tmp_path / "notrans.1pz")
        stats2 = singlepress.write_1pz(path2, sample_matrix, store_transpose=False)
        assert stats["compressed_bytes"] > stats2["compressed_bytes"]

    def test_row_range_read(self, sample_matrix, tmp_path):
        path = str(tmp_path / "rowrange.1pz")
        singlepress.write_1pz(path, sample_matrix, store_transpose=True)

        pz = singlepress.open_1pz(path)
        sub = pz.read(rows=(10, 50))
        # sub should have 40 rows and 200 columns
        assert sub.shape[0] == 40 or sub.shape[1] == 40  # depending on orientation


class TestInfo:
    def test_info_fields(self, sample_matrix, tmp_path):
        path = str(tmp_path / "info.1pz")
        singlepress.write_1pz(path, sample_matrix)
        info = singlepress.info_1pz(path)
        assert info["version"] == 1
        assert info["m"] == 500
        assert info["n"] == 200
        assert info["codec"] == "vocsc+zstd"
        assert "chunk_cols" in info
        assert "has_colsums" in info


class TestObsVarUns:
    """Tests for embedded obs/var DataFrames and uns key-value metadata."""

    @pytest.fixture
    def obs_df(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "cell_type": [f"type_{i % 5}" for i in range(200)],
                "n_genes": np.random.RandomState(42).randint(100, 5000, 200),
                "score": np.random.RandomState(42).random(200).astype(np.float32),
            },
            index=[f"CELL_{i:04d}" for i in range(200)],
        )

    @pytest.fixture
    def var_df(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "gene_symbol": [f"SYM_{i}" for i in range(500)],
                "highly_variable": [i % 10 == 0 for i in range(500)],
            },
            index=[f"Gene_{i}" for i in range(500)],
        )

    def test_obs_var_roundtrip(self, sample_matrix, tmp_path, obs_df, var_df):
        import pandas as pd

        path = str(tmp_path / "obsvar.1pz")
        uns = {"organism": "human", "version": "1.0"}
        singlepress.write_1pz(
            path,
            sample_matrix,
            rownames=[f"Gene_{i}" for i in range(500)],
            colnames=[f"CELL_{i:04d}" for i in range(200)],
            obs=obs_df,
            var=var_df,
            uns=uns,
        )

        info = singlepress.info_1pz(path)
        assert info["has_obs_var"] is True

        mat2 = singlepress.read_1pz(path)
        assert hasattr(mat2, "obs") and mat2.obs is not None
        assert hasattr(mat2, "var") and mat2.var is not None
        assert hasattr(mat2, "uns") and mat2.uns is not None

        pd.testing.assert_frame_equal(mat2.obs, obs_df)
        pd.testing.assert_frame_equal(mat2.var, var_df)
        assert mat2.uns == uns

    def test_obs_only(self, sample_matrix, tmp_path, obs_df):
        path = str(tmp_path / "obs_only.1pz")
        singlepress.write_1pz(path, sample_matrix, obs=obs_df)
        mat2 = singlepress.read_1pz(path)
        assert mat2.obs is not None
        assert mat2.var is None
        assert mat2.uns == {}

    def test_var_only(self, sample_matrix, tmp_path, var_df):
        path = str(tmp_path / "var_only.1pz")
        singlepress.write_1pz(path, sample_matrix, var=var_df)
        mat2 = singlepress.read_1pz(path)
        assert mat2.obs is None
        assert mat2.var is not None

    def test_uns_only(self, sample_matrix, tmp_path):
        path = str(tmp_path / "uns_only.1pz")
        uns = {"organism": "mouse", "tissue": "brain"}
        singlepress.write_1pz(path, sample_matrix, uns=uns)
        mat2 = singlepress.read_1pz(path)
        assert mat2.uns == uns
        assert mat2.obs is None

    def test_no_obs_var_backward_compat(self, sample_matrix, tmp_path):
        """Files without obs/var should still work and report no obs/var."""
        path = str(tmp_path / "noobs.1pz")
        singlepress.write_1pz(path, sample_matrix)
        info = singlepress.info_1pz(path)
        assert info["has_obs_var"] is False

        mat2 = singlepress.read_1pz(path)
        assert mat2.obs is None
        assert mat2.var is None
        assert mat2.uns == {}

    def test_onepzfile_obs_var(self, sample_matrix, tmp_path, obs_df, var_df):
        import pandas as pd

        path = str(tmp_path / "lazy_obsvar.1pz")
        uns = {"key": "value"}
        singlepress.write_1pz(
            path,
            sample_matrix,
            obs=obs_df,
            var=var_df,
            uns=uns,
        )
        pz = singlepress.open_1pz(path)
        assert pz.has_obs_var is True
        pd.testing.assert_frame_equal(pz.obs, obs_df)
        pd.testing.assert_frame_equal(pz.var, var_df)
        assert pz.uns == uns

    def test_int_read_obs_var(self, sample_matrix, tmp_path, obs_df):
        """read_1pz_int should also attach obs/var/uns."""
        path = str(tmp_path / "int_obs.1pz")
        singlepress.write_1pz(path, sample_matrix, obs=obs_df)
        mat2 = singlepress.read_1pz_int(path)
        assert mat2.dtype == np.int32
        assert mat2.obs is not None
        assert len(mat2.obs) == 200

    def test_values_preserved_with_obs_var(self, sample_matrix, tmp_path, obs_df, var_df):
        """Matrix values should be unaffected by obs/var metadata."""
        path = str(tmp_path / "vals.1pz")
        singlepress.write_1pz(path, sample_matrix, obs=obs_df, var=var_df)
        mat2 = singlepress.read_1pz(path)
        diff = (sample_matrix - mat2).data
        assert np.all(diff == 0)


class TestOnTheFlyNormalize:
    """Test on-the-fly log-normalization: log1p(x * scale / colsum)."""

    def _expected_lognorm(self, mat, scale=10000.0):
        """Compute expected log-normalized matrix from raw counts."""
        mat = mat.astype(np.float64).copy()
        colsums = np.asarray(mat.sum(axis=0)).ravel()
        result = mat.copy()
        factors = scale / colsums
        for j in range(mat.shape[1]):
            s, e = mat.indptr[j], mat.indptr[j + 1]
            if s < e:
                result.data[s:e] = np.log1p(mat.data[s:e] * factors[j])
        return result

    def test_read_1pz_normalize(self, sample_matrix, tmp_path):
        """read_1pz(normalize=True) returns log-normalized data."""
        path = str(tmp_path / "norm.1pz")
        singlepress.write_1pz(path, sample_matrix)
        mat = singlepress.read_1pz(path, normalize=True, scale=10000)
        expected = self._expected_lognorm(sample_matrix)
        np.testing.assert_allclose(mat.toarray(), expected.toarray(), rtol=1e-12)

    def test_read_1pz_normalize_custom_scale(self, sample_matrix, tmp_path):
        """read_1pz(normalize=True, scale=1e6) uses custom scaling."""
        path = str(tmp_path / "norm_cpm.1pz")
        singlepress.write_1pz(path, sample_matrix)
        mat = singlepress.read_1pz(path, normalize=True, scale=1e6)
        expected = self._expected_lognorm(sample_matrix, scale=1e6)
        np.testing.assert_allclose(mat.toarray(), expected.toarray(), rtol=1e-12)

    def test_open_normalize(self, sample_matrix, tmp_path):
        """open_1pz(normalize=True) makes all reads normalized."""
        path = str(tmp_path / "opn.1pz")
        singlepress.write_1pz(path, sample_matrix)
        pz = singlepress.open_1pz(path, normalize=True)
        mat = pz.read()
        expected = self._expected_lognorm(sample_matrix)
        np.testing.assert_allclose(mat.toarray(), expected.toarray(), rtol=1e-12)

    def test_normalized_method(self, sample_matrix, tmp_path):
        """pz.normalized() returns a handle that normalizes."""
        path = str(tmp_path / "meth.1pz")
        singlepress.write_1pz(path, sample_matrix)
        pz = singlepress.open_1pz(path)
        npz = pz.normalized()
        assert npz.normalize is True
        mat = npz.read()
        expected = self._expected_lognorm(sample_matrix)
        np.testing.assert_allclose(mat.toarray(), expected.toarray(), rtol=1e-12)

    def test_getitem_normalize(self, sample_matrix, gene_names, cell_barcodes, tmp_path):
        """pz[rows, cols] respects normalize flag."""
        path = str(tmp_path / "idx.1pz")
        singlepress.write_1pz(path, sample_matrix, rownames=gene_names, colnames=cell_barcodes)
        pz = singlepress.open_1pz(path, normalize=True)
        sub = pz[:, 0:50]
        expected = self._expected_lognorm(sample_matrix)[:, 0:50]
        np.testing.assert_allclose(sub.toarray(), expected.toarray(), rtol=1e-12)

    def test_read_normalize_override(self, sample_matrix, tmp_path):
        """pz.read(normalize=True) overrides default False."""
        path = str(tmp_path / "override.1pz")
        singlepress.write_1pz(path, sample_matrix)
        pz = singlepress.open_1pz(path)  # normalize=False by default
        mat = pz.read(normalize=True)
        expected = self._expected_lognorm(sample_matrix)
        np.testing.assert_allclose(mat.toarray(), expected.toarray(), rtol=1e-12)

    def test_partial_col_normalize(self, sample_matrix, tmp_path):
        """Partial column reads are normalized with correct colsums slice."""
        path = str(tmp_path / "partial.1pz")
        singlepress.write_1pz(path, sample_matrix)
        pz = singlepress.open_1pz(path, normalize=True)
        sub = pz.read(cols=(10, 30))
        expected = self._expected_lognorm(sample_matrix)[:, 10:30]
        np.testing.assert_allclose(sub.toarray(), expected.toarray(), rtol=1e-12)

    def test_stats_bypass_normalize(self, sample_matrix, tmp_path):
        """nnz_per_col/row and rowsums use raw counts even when normalized."""
        path = str(tmp_path / "stats.1pz")
        singlepress.write_1pz(path, sample_matrix)
        pz = singlepress.open_1pz(path, normalize=True)
        npc = pz.nnz_per_col()
        assert npc.sum() == sample_matrix.nnz
        rs = pz.rowsums()
        expected_rs = np.asarray(sample_matrix.astype(np.float64).sum(axis=1)).ravel()
        np.testing.assert_allclose(rs, expected_rs)

    def test_str_shows_normalize(self, sample_matrix, tmp_path):
        """str(pz) indicates normalization mode."""
        path = str(tmp_path / "show.1pz")
        singlepress.write_1pz(path, sample_matrix)
        pz = singlepress.open_1pz(path, normalize=True)
        s = str(pz)
        assert "Normalize" in s
        assert "10000" in s


class TestLargeMetadata:
    """Regression test for metadata decompression buffer overflow.

    Gene/barcode names with repetitive prefixes compress >10:1 with ZSTD,
    which previously caused 'Destination buffer is too small' errors.
    """

    def test_highly_compressible_metadata_roundtrip(self, tmp_path):
        """Metadata that compresses >10:1 should round-trip correctly."""
        rng = np.random.RandomState(42)
        m, n = 60000, 20000
        nnz = int(m * n * 0.001)
        rows = rng.randint(0, m, nnz)
        cols = rng.randint(0, n, nnz)
        vals = rng.randint(1, 50, nnz).astype(np.int32)
        mat = ss.csc_matrix((vals, (rows, cols)), shape=(m, n))
        mat.sum_duplicates()

        # Highly repetitive names — compress well beyond 10:1
        genes = [f"ENSMUSG00000{i:06d}" for i in range(m)]
        barcodes = [f"AAACCTGAGAAACCAT-{i:06d}" for i in range(n)]

        path = str(tmp_path / "large_meta.1pz")
        singlepress.write_1pz(path, mat, rownames=genes, colnames=barcodes)

        recovered = singlepress.read_1pz(path)
        assert recovered.shape == (m, n)
        assert recovered.rownames == genes
        assert recovered.colnames == barcodes
        diff = (mat != recovered).nnz
        assert diff == 0
