# SPDX-License-Identifier: MIT
"""Property-based tests for .1pz and .1pz codec round-trips.

Uses hypothesis to generate random sparse matrices and verify that
write → read preserves shape, nnz, and exact values.
"""

import os
import tempfile

import anndata as ad
import numpy as np
import scipy.sparse as sp
import singlet
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# ─── Strategies ─────────────────────────────────────────────────────────────


@st.composite
def sparse_adata(draw, max_rows=200, max_cols=100, max_val=65535):
    """Generate a random sparse AnnData with uint16-compatible values."""
    n_rows = draw(st.integers(min_value=1, max_value=max_rows))
    n_cols = draw(st.integers(min_value=1, max_value=max_cols))
    density = draw(st.floats(min_value=0.0, max_value=0.5))

    mat = sp.random(n_rows, n_cols, density=density, format="csc", dtype=np.float64)
    mat.data = np.round(mat.data * max_val).astype(np.uint16).astype(np.float64)
    mat.eliminate_zeros()

    adata = ad.AnnData(X=mat.tocsc())
    adata.obs_names = [f"cell_{i}" for i in range(n_rows)]
    adata.var_names = [f"gene_{i}" for i in range(n_cols)]
    return adata


# ─── .1pz property tests ────────────────────────────────────────────────────


class TestOnePZProperties:
    """Property-based tests for .1pz format codec."""

    @given(adata=sparse_adata())
    @settings(max_examples=50, deadline=30000)
    def test_shape_preserved(self, adata):
        """write_1pz → read_1pz preserves shape."""
        with tempfile.NamedTemporaryFile(suffix=".1pz", delete=False) as f:
            path = f.name
        try:
            singlet.write_1pz(adata, path)
            loaded = singlet.read_1pz(path)
            assert loaded.shape == adata.shape
        finally:
            os.unlink(path)

    @given(adata=sparse_adata())
    @settings(max_examples=50, deadline=30000)
    def test_nnz_preserved(self, adata):
        """write_1pz → read_1pz preserves nnz."""
        with tempfile.NamedTemporaryFile(suffix=".1pz", delete=False) as f:
            path = f.name
        try:
            singlet.write_1pz(adata, path)
            loaded = singlet.read_1pz(path)
            assert sp.issparse(loaded.X)
            assert loaded.X.nnz == adata.X.nnz
        finally:
            os.unlink(path)

    @given(adata=sparse_adata())
    @settings(max_examples=50, deadline=30000)
    def test_values_exact(self, adata):
        """write_1pz → read_1pz preserves exact uint16 values."""
        with tempfile.NamedTemporaryFile(suffix=".1pz", delete=False) as f:
            path = f.name
        try:
            singlet.write_1pz(adata, path)
            loaded = singlet.read_1pz(path)
            diff = abs(adata.X - loaded.X).sum()
            assert diff == 0.0, f"Value mismatch: total diff = {diff}"
        finally:
            os.unlink(path)

    @given(adata=sparse_adata())
    @settings(max_examples=30, deadline=30000)
    def test_obs_var_names_preserved(self, adata):
        """write_1pz → read_1pz preserves obs/var names."""
        with tempfile.NamedTemporaryFile(suffix=".1pz", delete=False) as f:
            path = f.name
        try:
            singlet.write_1pz(adata, path)
            loaded = singlet.read_1pz(path)
            assert list(loaded.obs_names) == list(adata.obs_names)
            assert list(loaded.var_names) == list(adata.var_names)
        finally:
            os.unlink(path)

    @given(adata=sparse_adata(max_rows=50, max_cols=20))
    @settings(max_examples=20, deadline=30000)
    def test_transpose_option(self, adata):
        """write_1pz with store_transpose=True still round-trips."""
        with tempfile.NamedTemporaryFile(suffix=".1pz", delete=False) as f:
            path = f.name
        try:
            singlet.write_1pz(adata, path, store_transpose=True)
            loaded = singlet.read_1pz(path)
            assert loaded.shape == adata.shape
            diff = abs(adata.X - loaded.X).sum()
            assert diff == 0.0
        finally:
            os.unlink(path)


# ─── .1pz property tests ────────────────────────────────────────────────────


class TestSPZProperties:
    """Property-based tests for .1pz format codec."""

    @given(adata=sparse_adata(max_rows=100, max_cols=50))
    @settings(max_examples=50, deadline=30000)
    def test_shape_preserved(self, adata):
        """write_1pz → read_1pz preserves shape."""
        with tempfile.NamedTemporaryFile(suffix=".1pz", delete=False) as f:
            path = f.name
        try:
            singlet.write_1pz(adata, path)
            loaded = singlet.read_1pz(path)
            assert loaded.shape == adata.shape
        finally:
            os.unlink(path)

    @given(adata=sparse_adata(max_rows=100, max_cols=50))
    @settings(max_examples=50, deadline=30000)
    def test_values_exact(self, adata):
        """write_1pz → read_1pz preserves exact values."""
        with tempfile.NamedTemporaryFile(suffix=".1pz", delete=False) as f:
            path = f.name
        try:
            singlet.write_1pz(adata, path)
            loaded = singlet.read_1pz(path)
            diff = abs(adata.X - loaded.X).sum()
            assert diff == 0.0, f"Value mismatch: total diff = {diff}"
        finally:
            os.unlink(path)

    @given(adata=sparse_adata(max_rows=100, max_cols=50))
    @settings(max_examples=30, deadline=30000)
    def test_col_range_subset(self, adata):
        """read_1pz round-trips shape."""
        assume(adata.n_obs >= 2)
        with tempfile.NamedTemporaryFile(suffix=".1pz", delete=False) as f:
            path = f.name
        try:
            singlet.write_1pz(adata, path)
            loaded = singlet.read_1pz(path)
            assert loaded.shape == adata.shape
        finally:
            os.unlink(path)


# ─── Deterministic edge cases ────────────────────────────────────────────────


class TestCodecEdgeCases:
    """Deterministic edge-case tests for codec limits."""

    def test_empty_matrix_1pz(self, tmp_path):
        """Zero-nnz matrix round-trips through .1pz."""
        mat = sp.csc_matrix((10, 5), dtype=np.uint16)
        adata = ad.AnnData(X=mat)
        adata.obs_names = [f"c{i}" for i in range(10)]
        adata.var_names = [f"g{i}" for i in range(5)]
        path = str(tmp_path / "empty.1pz")
        singlet.write_1pz(adata, path)
        loaded = singlet.read_1pz(path)
        assert loaded.shape == (10, 5)
        assert loaded.X.nnz == 0

    def test_empty_matrix_1pz(self, tmp_path):
        """Zero-nnz matrix round-trips through .1pz."""
        mat = sp.csc_matrix((10, 5), dtype=np.uint16)
        adata = ad.AnnData(X=mat)
        adata.obs_names = [f"c{i}" for i in range(10)]
        adata.var_names = [f"g{i}" for i in range(5)]
        path = str(tmp_path / "empty.1pz")
        singlet.write_1pz(adata, path)
        loaded = singlet.read_1pz(path)
        assert loaded.shape == (10, 5)
        assert loaded.X.nnz == 0

    def test_single_element_1pz(self, tmp_path):
        """1x1 matrix with 1 nonzero round-trips."""
        mat = sp.csc_matrix(np.array([[42]], dtype=np.uint16))
        adata = ad.AnnData(X=mat)
        adata.obs_names = ["cell_0"]
        adata.var_names = ["gene_0"]
        path = str(tmp_path / "one.1pz")
        singlet.write_1pz(adata, path)
        loaded = singlet.read_1pz(path)
        assert loaded.X[0, 0] == 42

    def test_max_uint16_value(self, tmp_path):
        """Maximum uint16 value (65535) preserved in .1pz."""
        mat = sp.csc_matrix(np.array([[65535]], dtype=np.uint16))
        adata = ad.AnnData(X=mat)
        adata.obs_names = ["cell_0"]
        adata.var_names = ["gene_0"]
        path = str(tmp_path / "max.1pz")
        singlet.write_1pz(adata, path)
        loaded = singlet.read_1pz(path)
        assert loaded.X[0, 0] == 65535

    def test_dense_column_1pz(self, tmp_path):
        """Fully dense column round-trips."""
        mat = sp.csc_matrix(np.arange(1, 101, dtype=np.uint16).reshape(100, 1))
        adata = ad.AnnData(X=mat)
        adata.obs_names = [f"c{i}" for i in range(100)]
        adata.var_names = ["gene_0"]
        path = str(tmp_path / "dense_col.1pz")
        singlet.write_1pz(adata, path)
        loaded = singlet.read_1pz(path)
        diff = abs(adata.X - loaded.X).sum()
        assert diff == 0.0

    def test_wide_matrix_1pz(self, tmp_path):
        """Matrix wider than tall (more genes than cells) round-trips."""
        mat = sp.random(5, 500, density=0.1, format="csc", dtype=np.float64)
        mat.data = np.round(mat.data * 1000).astype(np.uint16).astype(np.float64)
        mat.eliminate_zeros()
        adata = ad.AnnData(X=mat)
        adata.obs_names = [f"c{i}" for i in range(5)]
        adata.var_names = [f"g{i}" for i in range(500)]
        path = str(tmp_path / "wide.1pz")
        singlet.write_1pz(adata, path)
        loaded = singlet.read_1pz(path)
        assert loaded.shape == (5, 500)
        diff = abs(adata.X - loaded.X).sum()
        assert diff == 0.0
