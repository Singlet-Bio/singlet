# SPDX-License-Identifier: MIT
"""Tests for the in-tree .1pz (TP1Z) codec via singlet._pz.

Tests cover:
  1. Round-trip (write → read), exact value recovery
  2. Metadata (rownames, colnames, user_kv round-trip)
  3. info_1pz header fields
  4. Large / highly-compressible metadata (regression for buffer overflow)

Features of the external ``singlepress`` package that are NOT part of the
in-tree codec (obs/var DataFrames, colsums, validate, open_1pz, normalize,
read_1pz_int) are not tested here — those are external-package contracts.
"""

import numpy as np
import pytest
import scipy.sparse as ss


@pytest.fixture
def sample_matrix():
    """Create a reproducible sparse integer matrix (genes × cells)."""
    rng = np.random.RandomState(42)
    m, n = 500, 200
    density = 0.05
    nnz = int(m * n * density)
    rows = rng.randint(0, m, nnz)
    cols = rng.randint(0, n, nnz)
    vals = rng.randint(1, 100, nnz).astype(np.uint32)
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
        from singlet._pz import read_1pz, write_1pz

        path = str(tmp_path / "test.1pz")
        ok = write_1pz(
            path,
            sample_matrix.indptr.astype(np.int32),
            sample_matrix.indices.astype(np.int32),
            sample_matrix.data.astype(np.uint32),
            sample_matrix.shape[0],
            sample_matrix.shape[1],
        )
        assert ok is True

        r = read_1pz(path)
        assert r["m"] == 500
        assert r["n"] == 200
        assert r["nnz"] == sample_matrix.nnz

        recovered = ss.csc_matrix(
            (r["data"], r["indices"], r["indptr"]),
            shape=(r["m"], r["n"]),
        )
        diff = (sample_matrix.astype(np.int64) - recovered.astype(np.int64)).data
        assert np.all(diff == 0), "Values don't match after round-trip"

    def test_empty_matrix(self, tmp_path):
        from singlet._pz import read_1pz, write_1pz

        path = str(tmp_path / "empty.1pz")
        ok = write_1pz(
            path,
            np.zeros(51, dtype=np.int32),  # indptr of length n+1 = 51
            np.array([], dtype=np.int32),
            np.array([], dtype=np.uint32),
            100,  # m
            50,   # n
        )
        assert ok is True

        r = read_1pz(path)
        assert r["m"] == 100
        assert r["n"] == 50
        assert r["nnz"] == 0

    def test_uint8_data(self, tmp_path):
        from singlet._pz import read_1pz, write_1pz

        rng = np.random.RandomState(7)
        mat = ss.random(50, 30, density=0.3, format="csc", random_state=rng)
        data = (mat.data * 200).astype(np.uint8)
        mat.data = data

        path = str(tmp_path / "u8.1pz")
        write_1pz(
            path,
            mat.indptr.astype(np.int32),
            mat.indices.astype(np.int32),
            data,
            50, 30,
        )
        r = read_1pz(path)
        assert r["m"] == 50 and r["n"] == 30
        recovered = ss.csc_matrix((r["data"], r["indices"], r["indptr"]), shape=(50, 30))
        np.testing.assert_array_equal(
            recovered.toarray().astype(np.int64),
            mat.toarray().astype(np.int64),
        )


class TestMetadata:
    def test_rownames_colnames(self, sample_matrix, gene_names, cell_barcodes, tmp_path):
        from singlet._pz import read_1pz, write_1pz

        path = str(tmp_path / "meta.1pz")
        write_1pz(
            path,
            sample_matrix.indptr.astype(np.int32),
            sample_matrix.indices.astype(np.int32),
            sample_matrix.data.astype(np.uint32),
            sample_matrix.shape[0],
            sample_matrix.shape[1],
            rownames=gene_names,
            colnames=cell_barcodes,
        )

        r = read_1pz(path)
        assert r["rownames"] == gene_names
        assert r["colnames"] == cell_barcodes

    def test_user_kv(self, sample_matrix, tmp_path):
        from singlet._pz import read_1pz, write_1pz

        path = str(tmp_path / "kv.1pz")
        kv = {"organism": "human", "tissue": "brain"}
        write_1pz(
            path,
            sample_matrix.indptr.astype(np.int32),
            sample_matrix.indices.astype(np.int32),
            sample_matrix.data.astype(np.uint32),
            sample_matrix.shape[0],
            sample_matrix.shape[1],
            user_meta=kv,
        )

        r = read_1pz(path)
        assert r["user_kv"]["organism"] == "human"
        assert r["user_kv"]["tissue"] == "brain"

    def test_no_metadata(self, sample_matrix, tmp_path):
        from singlet._pz import read_1pz, write_1pz

        path = str(tmp_path / "nometa.1pz")
        write_1pz(
            path,
            sample_matrix.indptr.astype(np.int32),
            sample_matrix.indices.astype(np.int32),
            sample_matrix.data.astype(np.uint32),
            sample_matrix.shape[0],
            sample_matrix.shape[1],
        )
        r = read_1pz(path)
        assert r["rownames"] == []
        assert r["colnames"] == []
        assert r["user_kv"] == {}


class TestInfo:
    def test_info_fields(self, sample_matrix, tmp_path):
        from singlet._pz import info_1pz, write_1pz

        path = str(tmp_path / "info.1pz")
        write_1pz(
            path,
            sample_matrix.indptr.astype(np.int32),
            sample_matrix.indices.astype(np.int32),
            sample_matrix.data.astype(np.uint32),
            sample_matrix.shape[0],
            sample_matrix.shape[1],
        )
        info = info_1pz(path)
        assert info["magic"] == "TP1Z"
        assert info["version"] == 1
        assert info["m"] == 500
        assert info["n"] == 200
        assert info["nnz"] == sample_matrix.nnz
        assert "chunk_cols" in info
        assert "num_chunks" in info


class TestHighLevelIO:
    """Tests for singlet._io high-level AnnData round-trip."""

    def _make_adata(self, n_genes=100, n_cells=50, density=0.3):
        ad = pytest.importorskip("anndata")
        import pandas as pd

        mat = ss.random(n_cells, n_genes, density=density, format="csr", dtype=np.float32)
        mat.data = np.round(mat.data * 100).astype(np.float32)
        adata = ad.AnnData(X=mat)
        adata.obs_names = pd.Index([f"CELL{i:04d}" for i in range(n_cells)])
        adata.var_names = pd.Index([f"GENE{j:04d}" for j in range(n_genes)])
        return adata

    def test_basic_roundtrip(self, tmp_path):
        from singlet._io import read_1pz, write_1pz

        adata = self._make_adata()
        path = tmp_path / "test.1pz"
        write_1pz(adata, path)
        loaded = read_1pz(path)

        assert loaded.shape == adata.shape
        orig = adata.X.toarray() if ss.issparse(adata.X) else adata.X
        load = loaded.X.toarray() if ss.issparse(loaded.X) else loaded.X
        np.testing.assert_allclose(load, orig, atol=1.0)

    def test_preserves_names(self, tmp_path):
        from singlet._io import read_1pz, write_1pz

        adata = self._make_adata(n_genes=10, n_cells=5)
        path = tmp_path / "names.1pz"
        write_1pz(adata, path)
        loaded = read_1pz(path)

        assert list(loaded.var_names) == list(adata.var_names)
        assert list(loaded.obs_names) == list(adata.obs_names)

    def test_uns_kv_roundtrip(self, tmp_path):
        from singlet._io import read_1pz, write_1pz

        adata = self._make_adata(n_genes=10, n_cells=5)
        adata.uns["organism"] = "mouse"
        adata.uns["version"] = "1.0"
        path = tmp_path / "uns.1pz"
        write_1pz(adata, path)
        loaded = read_1pz(path)

        assert loaded.uns.get("organism") == "mouse"
        assert loaded.uns.get("version") == "1.0"

    def test_total_counts_computed(self, tmp_path):
        from singlet._io import read_1pz, write_1pz

        adata = self._make_adata(n_genes=20, n_cells=10)
        path = tmp_path / "tc.1pz"
        write_1pz(adata, path)
        loaded = read_1pz(path)

        assert "total_counts" in loaded.obs.columns
        expected = np.array(adata.X.sum(axis=1)).ravel()
        np.testing.assert_allclose(loaded.obs["total_counts"].values, expected, atol=1.0)

    def test_store_transpose_raises(self, tmp_path):
        from singlet._io import write_1pz

        adata = self._make_adata(n_genes=5, n_cells=3)
        path = tmp_path / "transpose.1pz"
        with pytest.raises(NotImplementedError, match="store_transpose"):
            write_1pz(adata, path, store_transpose=True)

    def test_info_1pz(self, tmp_path):
        from singlet._io import info_1pz, write_1pz

        adata = self._make_adata(n_genes=30, n_cells=20)
        path = tmp_path / "info.1pz"
        write_1pz(adata, path)
        info = info_1pz(path)

        assert isinstance(info, dict)
        assert info.get("m", 0) > 0
        assert info.get("magic") == "TP1Z"


class TestLargeMetadata:
    """Regression test for metadata decompression buffer overflow.

    Gene/barcode names with repetitive prefixes compress >10:1 with ZSTD,
    which previously caused 'Destination buffer is too small' errors.
    """

    def test_highly_compressible_metadata_roundtrip(self, tmp_path):
        from singlet._pz import read_1pz, write_1pz

        rng = np.random.RandomState(42)
        m, n = 10000, 5000
        nnz = int(m * n * 0.001)
        rows = rng.randint(0, m, nnz)
        cols = rng.randint(0, n, nnz)
        vals = rng.randint(1, 50, nnz).astype(np.uint32)
        mat = ss.csc_matrix((vals, (rows, cols)), shape=(m, n))
        mat.sum_duplicates()

        # Highly repetitive names — compress well beyond 10:1
        genes = [f"ENSMUSG00000{i:06d}" for i in range(m)]
        barcodes = [f"AAACCTGAGAAACCAT-{i:06d}" for i in range(n)]

        path = str(tmp_path / "large_meta.1pz")
        write_1pz(
            path,
            mat.indptr.astype(np.int32),
            mat.indices.astype(np.int32),
            mat.data.astype(np.uint32),
            m, n,
            rownames=genes,
            colnames=barcodes,
        )

        r = read_1pz(path)
        assert r["m"] == m and r["n"] == n
        assert r["rownames"] == genes
        assert r["colnames"] == barcodes
        recovered = ss.csc_matrix((r["data"], r["indices"], r["indptr"]), shape=(m, n))
        diff = (mat.astype(np.int64) - recovered.astype(np.int64)).nnz
        assert diff == 0
