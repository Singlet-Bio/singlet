# SPDX-License-Identifier: MIT
"""Unit tests for format conversion utilities.

Covers both import paths:
  - ``singlet.convert``     (top-level re-export)
  - ``singlet.io.convert``  (canonical module)

Conversions tested: h5ad, zarr, TileDB-SOMA, MTX, CSC, and the
.1pz/.spz <-> h5ad convenience wrappers.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse as sp
from singlet.convert import (
    from_h5ad,
    from_mtx,
    to_csc,
    to_h5ad,
    to_mtx,
)


@pytest.fixture
def sample_adata():
    """Create a small float-valued AnnData for testing."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 100, 50
    X = sp.random(n_cells, n_genes, density=0.1, format="csr", random_state=42).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs_names = pd.Index([f"cell_{i}" for i in range(n_cells)])
    adata.var_names = pd.Index([f"gene_{i}" for i in range(n_genes)])
    adata.obs["batch"] = rng.choice(["A", "B"], size=n_cells)
    return adata


@pytest.fixture
def int_adata():
    """Integer-valued AnnData for lossless round-trips."""
    mat = sp.random(8, 5, density=0.4, format="csr", dtype=np.float32)
    mat.data = np.round(mat.data * 50).astype(np.float32)
    adata = ad.AnnData(X=mat)
    adata.obs_names = pd.Index([f"C{i}" for i in range(8)])
    adata.var_names = pd.Index([f"G{j}" for j in range(5)])
    return adata


# ---------------------------------------------------------------------------
# to_h5ad / from_h5ad
# ---------------------------------------------------------------------------


class TestToH5ad:
    def test_roundtrip(self, sample_adata, tmp_path):
        path = tmp_path / "test.h5ad"
        to_h5ad(sample_adata, path)
        assert path.exists()
        assert path.stat().st_size > 0

        loaded = from_h5ad(path)
        assert loaded.shape == sample_adata.shape
        assert list(loaded.obs_names) == list(sample_adata.obs_names)
        assert list(loaded.var_names) == list(sample_adata.var_names)

    def test_preserves_obs(self, sample_adata, tmp_path):
        path = tmp_path / "test.h5ad"
        to_h5ad(sample_adata, path)
        loaded = from_h5ad(path)
        assert "batch" in loaded.obs.columns
        assert list(loaded.obs["batch"]) == list(sample_adata.obs["batch"])

    def test_compression(self, sample_adata, tmp_path):
        path_gz = tmp_path / "test_gz.h5ad"
        to_h5ad(sample_adata, path_gz, compression="gzip")
        assert path_gz.exists()
        # Just verify it's a valid h5ad (compression is best-effort for small data)
        loaded = from_h5ad(path_gz)
        assert loaded.shape == sample_adata.shape


class TestFromH5ad:
    def test_backed_mode(self, sample_adata, tmp_path):
        path = tmp_path / "backed.h5ad"
        to_h5ad(sample_adata, path)
        loaded = from_h5ad(path, backed="r")
        assert loaded.shape == sample_adata.shape
        loaded.file.close()


class TestH5adConversion:
    """h5ad round-trips on integer-valued data (value preservation)."""

    def test_to_h5ad(self, int_adata, tmp_path):
        """Write to h5ad creates a valid file."""
        path = tmp_path / "test.h5ad"
        to_h5ad(int_adata, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_from_h5ad(self, int_adata, tmp_path):
        """Read back from h5ad preserves data."""
        path = tmp_path / "roundtrip.h5ad"
        to_h5ad(int_adata, path)
        loaded = from_h5ad(path)

        assert loaded.shape == int_adata.shape
        orig = int_adata.X.toarray()
        load = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_allclose(load, orig, atol=1e-5)

    def test_h5ad_preserves_names(self, int_adata, tmp_path):
        """Gene and cell names survive h5ad round-trip."""
        path = tmp_path / "names.h5ad"
        to_h5ad(int_adata, path)
        loaded = from_h5ad(path)

        assert list(loaded.obs_names) == list(int_adata.obs_names)
        assert list(loaded.var_names) == list(int_adata.var_names)

    def test_to_h5ad_compression(self, int_adata, tmp_path):
        """to_h5ad with gzip compression produces a valid file."""
        gz_path = tmp_path / "compressed.h5ad"
        to_h5ad(int_adata, gz_path, compression="gzip")
        assert gz_path.exists()


# ---------------------------------------------------------------------------
# to_csc
# ---------------------------------------------------------------------------


class TestToCsc:
    def test_returns_csc(self, sample_adata):
        mat = to_csc(sample_adata)
        assert sp.issparse(mat)
        assert mat.format == "csc"
        assert mat.shape == sample_adata.shape

    def test_from_dense(self):
        X = np.array([[1, 0, 3], [0, 2, 0]], dtype=np.float32)
        adata = ad.AnnData(X=X)
        mat = to_csc(adata)
        assert sp.issparse(mat)
        assert mat.format == "csc"
        assert mat.shape == (2, 3)

    def test_layer(self, sample_adata):
        sample_adata.layers["raw"] = sample_adata.X.copy()
        mat = to_csc(sample_adata, layer="raw")
        assert mat.shape == sample_adata.shape

    def test_sparse_input(self):
        """to_csc via singlet.io.convert on sparse input."""
        from singlet.io.convert import to_csc as io_to_csc

        adata = ad.AnnData(X=sp.random(10, 5, density=0.3, format="csr"))
        result = io_to_csc(adata)
        assert sp.issparse(result)
        assert result.format == "csc"
        assert result.shape == (10, 5)

    def test_dense_input(self):
        """to_csc via singlet.io.convert on dense input."""
        from singlet.io.convert import to_csc as io_to_csc

        adata = ad.AnnData(X=sp.random(10, 5, density=0.3, format="csr").toarray())
        result = io_to_csc(adata)
        assert sp.issparse(result)
        assert result.format == "csc"

    def test_layer_selection(self):
        """to_csc via singlet.io.convert honors the layer argument."""
        from singlet.io.convert import to_csc as io_to_csc

        X = sp.random(4, 3, density=0.5, format="csr")
        layer = sp.random(4, 3, density=0.8, format="csr")
        adata = ad.AnnData(X=X, layers={"raw": layer})
        result = io_to_csc(adata, layer="raw")
        # Check it used the layer, not X
        assert result.nnz == layer.tocsc().nnz


# ---------------------------------------------------------------------------
# to_mtx / from_mtx
# ---------------------------------------------------------------------------


class TestToMtx:
    def test_creates_files(self, sample_adata, tmp_path):
        out_dir = tmp_path / "mtx_out"
        to_mtx(sample_adata, out_dir)
        assert (out_dir / "matrix.mtx.gz").exists()
        assert (out_dir / "barcodes.tsv.gz").exists()
        assert (out_dir / "features.tsv.gz").exists()

    def test_roundtrip(self, sample_adata, tmp_path):
        out_dir = tmp_path / "mtx_out"
        to_mtx(sample_adata, out_dir)
        loaded = from_mtx(out_dir)
        assert loaded.shape == sample_adata.shape
        # Values should be preserved (within float precision)
        orig_dense = sample_adata.X.toarray() if sp.issparse(sample_adata.X) else sample_adata.X
        load_dense = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_array_almost_equal(orig_dense, load_dense, decimal=5)


class TestMtxRoundTrip:
    """to_mtx/from_mtx round-trips via singlet.io.convert."""

    def _make_adata(self, n_cells=5, n_genes=3):
        mat = sp.random(n_cells, n_genes, density=0.5, format="csr")
        adata = ad.AnnData(X=mat)
        adata.obs_names = pd.Index([f"CELL{i}" for i in range(n_cells)])
        adata.var_names = pd.Index([f"GENE{j}" for j in range(n_genes)])
        return adata

    def test_roundtrip(self, tmp_path):
        from singlet.io.convert import from_mtx as io_from_mtx
        from singlet.io.convert import to_mtx as io_to_mtx

        adata = self._make_adata()
        io_to_mtx(adata, tmp_path)

        # Check files were created
        assert (tmp_path / "matrix.mtx.gz").exists()
        assert (tmp_path / "barcodes.tsv.gz").exists()
        assert (tmp_path / "features.tsv.gz").exists()

        # Read back
        loaded = io_from_mtx(tmp_path)
        assert loaded.shape == adata.shape
        assert list(loaded.obs_names) == list(adata.obs_names)
        assert list(loaded.var_names) == list(adata.var_names)

        # Values match
        orig_dense = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        load_dense = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_allclose(orig_dense, load_dense, atol=1e-6)

    def test_to_mtx_creates_directory(self, tmp_path):
        from singlet.io.convert import to_mtx as io_to_mtx

        adata = self._make_adata()
        new_dir = tmp_path / "sub" / "dir"
        io_to_mtx(adata, new_dir)
        assert (new_dir / "matrix.mtx.gz").exists()

    def test_to_mtx_with_layer(self, tmp_path):
        from singlet.io.convert import from_mtx as io_from_mtx
        from singlet.io.convert import to_mtx as io_to_mtx

        X = sp.random(3, 2, density=0.3, format="csr")
        layer = sp.random(3, 2, density=1.0, format="csr")
        adata = ad.AnnData(X=X, layers={"counts": layer})
        adata.obs_names = pd.Index(["A", "B", "C"])
        adata.var_names = pd.Index(["G1", "G2"])

        io_to_mtx(adata, tmp_path, layer="counts")
        loaded = io_from_mtx(tmp_path)

        load_dense = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        layer_dense = layer.toarray()
        np.testing.assert_allclose(load_dense, layer_dense, atol=1e-6)

    def test_to_mtx_dense_input(self, tmp_path):
        from singlet.io.convert import from_mtx as io_from_mtx
        from singlet.io.convert import to_mtx as io_to_mtx

        X = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
        adata = ad.AnnData(X=X)
        adata.obs_names = pd.Index(["c1", "c2", "c3"])
        adata.var_names = pd.Index(["g1", "g2"])

        io_to_mtx(adata, tmp_path)
        loaded = io_from_mtx(tmp_path)
        load_dense = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_allclose(load_dense, X, atol=1e-6)


class TestFromMtx:
    def test_reads_barcodes(self, sample_adata, tmp_path):
        out_dir = tmp_path / "mtx_out"
        to_mtx(sample_adata, out_dir)
        loaded = from_mtx(out_dir)
        assert list(loaded.obs_names) == list(sample_adata.obs_names)

    def test_reads_features(self, sample_adata, tmp_path):
        out_dir = tmp_path / "mtx_out"
        to_mtx(sample_adata, out_dir)
        loaded = from_mtx(out_dir)
        assert list(loaded.var_names) == list(sample_adata.var_names)

    def test_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            from_mtx(tmp_path / "nonexistent")

    def test_missing_matrix_raises(self, tmp_path):
        from singlet.io.convert import from_mtx as io_from_mtx

        with pytest.raises(FileNotFoundError, match="No matrix.mtx"):
            io_from_mtx(tmp_path)

    def test_uncompressed_matrix(self, tmp_path):
        """Reads uncompressed matrix.mtx if .gz doesn't exist."""
        from singlet.io.convert import from_mtx as io_from_mtx

        # MTX format is genes×cells (3 genes, 2 cells) → from_mtx transposes → (2, 3)
        mat = sp.random(3, 2, density=0.5, format="coo")
        # Write uncompressed
        scipy.io.mmwrite(tmp_path / "matrix.mtx", mat)
        # Write barcodes (2 cells)
        with open(tmp_path / "barcodes.tsv", "w") as f:
            f.write("BC1\nBC2\n")
        # Write genes (alternative name, 3 genes)
        with open(tmp_path / "genes.tsv", "w") as f:
            f.write("G1\tGene1\nG2\tGene2\nG3\tGene3\n")

        loaded = io_from_mtx(tmp_path)
        assert loaded.shape == (2, 3)
        assert "gene_name" in loaded.var.columns

    def test_features_with_gene_name(self, tmp_path):
        """Features file with two columns populates var['gene_name']."""
        from singlet.io.convert import from_mtx as io_from_mtx

        mat = sp.random(2, 2, density=1.0, format="coo")
        scipy.io.mmwrite(tmp_path / "matrix.mtx", mat)

        with open(tmp_path / "features.tsv", "w") as f:
            f.write("ENSG001\tTP53\tGene Expression\nENSG002\tBRCA1\tGene Expression\n")

        loaded = io_from_mtx(tmp_path)
        assert list(loaded.var_names) == ["ENSG001", "ENSG002"]
        assert list(loaded.var["gene_name"]) == ["TP53", "BRCA1"]


class TestMtxViaSingletConvert:
    """Test to_mtx/from_mtx via singlet.convert (not singlet.io.convert)."""

    def test_to_mtx_dense_via_convert(self, tmp_path):
        """to_mtx handles dense matrix input by converting to sparse."""
        X = np.array([[1, 0, 3], [0, 5, 0], [2, 0, 4]], dtype=np.float32)
        adata = ad.AnnData(X=X)  # dense input
        adata.var_names = pd.Index(["A", "B", "C"])
        adata.obs_names = pd.Index(["c1", "c2", "c3"])

        outdir = tmp_path / "mtx_out"
        to_mtx(adata, outdir)
        assert (outdir / "matrix.mtx.gz").exists()

        loaded = from_mtx(outdir)
        load_dense = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_allclose(load_dense, X, atol=1e-6)


# ---------------------------------------------------------------------------
# zarr conversion
# ---------------------------------------------------------------------------


class TestZarrConversion:
    def test_to_zarr(self, int_adata, tmp_path):
        """Write to zarr creates a zarr store directory."""
        pytest.importorskip("zarr")
        from singlet.convert import to_zarr

        path = tmp_path / "test.zarr"
        to_zarr(int_adata, path)
        assert path.is_dir()

    def test_from_zarr(self, int_adata, tmp_path):
        """Read back from zarr preserves data."""
        pytest.importorskip("zarr")
        from singlet.convert import from_zarr, to_zarr

        path = tmp_path / "roundtrip.zarr"
        to_zarr(int_adata, path)
        loaded = from_zarr(path)

        assert loaded.shape == int_adata.shape


# ---------------------------------------------------------------------------
# .1pz / .spz <-> h5ad convenience wrappers
# ---------------------------------------------------------------------------


class TestPzH5adConvenience:
    """pz_to_h5ad / h5ad_to_pz via singlet.convert."""

    def test_pz_to_h5ad(self, int_adata, tmp_path):
        """pz_to_h5ad converts .1pz → .h5ad."""
        from singlet._io import write_1pz
        from singlet.convert import pz_to_h5ad

        pz_path = tmp_path / "source.1pz"
        h5ad_path = tmp_path / "output.h5ad"
        write_1pz(int_adata, pz_path)
        pz_to_h5ad(pz_path, h5ad_path)

        assert h5ad_path.exists()
        loaded = ad.read_h5ad(h5ad_path)
        assert loaded.shape == int_adata.shape

    def test_h5ad_to_pz(self, int_adata, tmp_path):
        """h5ad_to_pz converts .h5ad → .1pz."""
        from singlet._io import read_1pz
        from singlet.convert import h5ad_to_pz, to_h5ad

        h5ad_path = tmp_path / "source.h5ad"
        pz_path = tmp_path / "output.1pz"
        to_h5ad(int_adata, h5ad_path)
        h5ad_to_pz(h5ad_path, pz_path)

        assert pz_path.exists()
        loaded = read_1pz(pz_path)
        assert loaded.shape == int_adata.shape

    def test_pz_h5ad_roundtrip_values(self, int_adata, tmp_path):
        """Values survive pz→h5ad→pz round-trip."""
        from singlet._io import read_1pz, write_1pz
        from singlet.convert import h5ad_to_pz, pz_to_h5ad

        pz1 = tmp_path / "step1.1pz"
        h5ad = tmp_path / "step2.h5ad"
        pz2 = tmp_path / "step3.1pz"

        write_1pz(int_adata, pz1)
        pz_to_h5ad(pz1, h5ad)
        h5ad_to_pz(h5ad, pz2)

        loaded = read_1pz(pz2)
        orig = int_adata.X.toarray()
        load = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_allclose(load, orig, atol=1e-4)


class TestPzH5adConvenience:
    """pz_to_h5ad, h5ad_to_pz via singlet.io.convert."""

    def test_pz_to_h5ad_via_io(self, int_adata, tmp_path):
        """Convert .1pz → .h5ad via singlet.io.convert."""
        from singlet._io import write_1pz
        from singlet.io.convert import pz_to_h5ad

        pz_path = tmp_path / "data.1pz"
        write_1pz(int_adata, pz_path)

        h5ad_path = tmp_path / "out.h5ad"
        pz_to_h5ad(pz_path, h5ad_path)
        assert h5ad_path.exists()

    def test_h5ad_to_pz_via_io(self, int_adata, tmp_path):
        """Convert .h5ad → .1pz via singlet.io.convert."""
        from singlet.io.convert import h5ad_to_pz, to_h5ad

        h5ad_path = tmp_path / "data.h5ad"
        to_h5ad(int_adata, h5ad_path)

        pz_path = tmp_path / "out.1pz"
        h5ad_to_pz(h5ad_path, pz_path)
        assert pz_path.exists()

    def test_from_h5ad_via_io(self, int_adata, tmp_path):
        """from_h5ad imported from singlet.io.convert reads correctly."""
        from singlet.io.convert import from_h5ad as io_from_h5ad
        from singlet.io.convert import to_h5ad as io_to_h5ad

        path = tmp_path / "test.h5ad"
        io_to_h5ad(int_adata, path)
        loaded = io_from_h5ad(path)
        assert loaded.shape == int_adata.shape


# ---------------------------------------------------------------------------
# TileDB-SOMA conversion
# ---------------------------------------------------------------------------


class TestTileDBConversion:
    """TileDB-SOMA roundtrip tests."""

    def test_to_tiledb_creates_store(self, int_adata, tmp_path):
        """to_tiledb creates a TileDB-SOMA experiment."""
        tiledbsoma = pytest.importorskip("tiledbsoma")  # noqa: F841
        from singlet.convert import to_tiledb

        uri = str(tmp_path / "test.tiledb")
        to_tiledb(int_adata, uri)
        assert (tmp_path / "test.tiledb").exists()

    def test_roundtrip_tiledb(self, int_adata, tmp_path):
        """Write and read back via TileDB-SOMA preserves shape."""
        tiledbsoma = pytest.importorskip("tiledbsoma")  # noqa: F841
        from singlet.convert import from_tiledb, to_tiledb

        uri = str(tmp_path / "rt.tiledb")
        to_tiledb(int_adata, uri)
        loaded = from_tiledb(uri)
        assert loaded.shape == int_adata.shape

    def test_from_tiledb_top_level(self, int_adata, tmp_path):
        """singlet.from_tiledb works at package level."""
        tiledbsoma = pytest.importorskip("tiledbsoma")  # noqa: F841
        import singlet

        uri = str(tmp_path / "pkg.tiledb")
        singlet.to_tiledb(int_adata, uri)
        loaded = singlet.from_tiledb(uri)
        assert loaded.shape == int_adata.shape
