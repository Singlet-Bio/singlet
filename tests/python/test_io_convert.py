"""Tests for singlet.io.convert (format conversion utilities)."""

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp
from singlet.io.convert import from_mtx, to_csc, to_mtx

# ---------------------------------------------------------------------------
# to_csc
# ---------------------------------------------------------------------------


class TestToCsc:
    def _make_adata(self, dense=False):
        """Create a minimal AnnData-like object."""
        ad = pytest.importorskip("anndata")
        mat = sp.random(10, 5, density=0.3, format="csr")
        if dense:
            mat = mat.toarray()
        return ad.AnnData(X=mat)

    def test_sparse_input(self):
        adata = self._make_adata(dense=False)
        result = to_csc(adata)
        assert sp.issparse(result)
        assert result.format == "csc"
        assert result.shape == (10, 5)

    def test_dense_input(self):
        adata = self._make_adata(dense=True)
        result = to_csc(adata)
        assert sp.issparse(result)
        assert result.format == "csc"

    def test_layer_selection(self):
        ad = pytest.importorskip("anndata")
        X = sp.random(4, 3, density=0.5, format="csr")
        layer = sp.random(4, 3, density=0.8, format="csr")
        adata = ad.AnnData(X=X, layers={"raw": layer})
        result = to_csc(adata, layer="raw")
        # Check it used the layer, not X
        assert result.nnz == layer.tocsc().nnz


# ---------------------------------------------------------------------------
# to_mtx round-trip with from_mtx
# ---------------------------------------------------------------------------


class TestMtxRoundTrip:
    def _make_adata(self, n_cells=5, n_genes=3):
        ad = pytest.importorskip("anndata")
        import pandas as pd

        mat = sp.random(n_cells, n_genes, density=0.5, format="csr")
        adata = ad.AnnData(X=mat)
        adata.obs_names = pd.Index([f"CELL{i}" for i in range(n_cells)])
        adata.var_names = pd.Index([f"GENE{j}" for j in range(n_genes)])
        return adata

    def test_roundtrip(self, tmp_path):
        adata = self._make_adata()
        to_mtx(adata, tmp_path)

        # Check files were created
        assert (tmp_path / "matrix.mtx.gz").exists()
        assert (tmp_path / "barcodes.tsv.gz").exists()
        assert (tmp_path / "features.tsv.gz").exists()

        # Read back
        loaded = from_mtx(tmp_path)
        assert loaded.shape == adata.shape
        assert list(loaded.obs_names) == list(adata.obs_names)
        assert list(loaded.var_names) == list(adata.var_names)

        # Values match
        orig_dense = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        load_dense = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_allclose(orig_dense, load_dense, atol=1e-6)

    def test_to_mtx_creates_directory(self, tmp_path):
        adata = self._make_adata()
        new_dir = tmp_path / "sub" / "dir"
        to_mtx(adata, new_dir)
        assert (new_dir / "matrix.mtx.gz").exists()

    def test_to_mtx_with_layer(self, tmp_path):
        ad = pytest.importorskip("anndata")
        import pandas as pd

        X = sp.random(3, 2, density=0.3, format="csr")
        layer = sp.random(3, 2, density=1.0, format="csr")
        adata = ad.AnnData(X=X, layers={"counts": layer})
        adata.obs_names = pd.Index(["A", "B", "C"])
        adata.var_names = pd.Index(["G1", "G2"])

        to_mtx(adata, tmp_path, layer="counts")
        loaded = from_mtx(tmp_path)

        load_dense = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        layer_dense = layer.toarray()
        np.testing.assert_allclose(load_dense, layer_dense, atol=1e-6)

    def test_to_mtx_dense_input(self, tmp_path):
        ad = pytest.importorskip("anndata")
        import pandas as pd

        X = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
        adata = ad.AnnData(X=X)
        adata.obs_names = pd.Index(["c1", "c2", "c3"])
        adata.var_names = pd.Index(["g1", "g2"])

        to_mtx(adata, tmp_path)
        loaded = from_mtx(tmp_path)
        load_dense = loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X
        np.testing.assert_allclose(load_dense, X, atol=1e-6)


# ---------------------------------------------------------------------------
# from_mtx edge cases
# ---------------------------------------------------------------------------


class TestFromMtx:
    def test_missing_matrix_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No matrix.mtx"):
            from_mtx(tmp_path)

    def test_uncompressed_matrix(self, tmp_path):
        """Reads uncompressed matrix.mtx if .gz doesn't exist."""
        pytest.importorskip("anndata")

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

        loaded = from_mtx(tmp_path)
        assert loaded.shape == (2, 3)
        assert "gene_name" in loaded.var.columns

    def test_features_with_gene_name(self, tmp_path):
        """Features file with two columns populates var['gene_name']."""
        pytest.importorskip("anndata")

        mat = sp.random(2, 2, density=1.0, format="coo")
        scipy.io.mmwrite(tmp_path / "matrix.mtx", mat)

        with open(tmp_path / "features.tsv", "w") as f:
            f.write("ENSG001\tTP53\tGene Expression\nENSG002\tBRCA1\tGene Expression\n")

        loaded = from_mtx(tmp_path)
        assert list(loaded.var_names) == ["ENSG001", "ENSG002"]
        assert list(loaded.var["gene_name"]) == ["TP53", "BRCA1"]


class TestSpzH5adConvenience:
    """Test spz_to_h5ad, h5ad_to_spz, pz_to_h5ad, h5ad_to_pz via singlet.io."""

    @pytest.fixture
    def int_adata(self):
        """Integer-valued AnnData for lossless roundtrips."""
        import anndata as ad
        import pandas as pd
        import scipy.sparse as sp

        mat = sp.random(4, 6, density=0.5, format="csr", dtype=np.float32)
        mat.data = np.round(mat.data * 50).astype(np.float32)
        adata = ad.AnnData(X=mat)
        adata.obs_names = pd.Index([f"C{i}" for i in range(4)])
        adata.var_names = pd.Index([f"G{j}" for j in range(6)])
        return adata

    def test_spz_to_h5ad(self, int_adata, tmp_path):
        """Convert .spz → .h5ad via singlet.io.convert."""
        from singlet._io import write_spz
        from singlet.io.convert import spz_to_h5ad

        spz_path = tmp_path / "data.spz"
        write_spz(int_adata, spz_path)

        h5ad_path = tmp_path / "out.h5ad"
        spz_to_h5ad(spz_path, h5ad_path)
        assert h5ad_path.exists()

    def test_h5ad_to_spz(self, int_adata, tmp_path):
        """Convert .h5ad → .spz via singlet.io.convert."""
        from singlet.io.convert import h5ad_to_spz, to_h5ad

        h5ad_path = tmp_path / "data.h5ad"
        to_h5ad(int_adata, h5ad_path)

        spz_path = tmp_path / "out.spz"
        h5ad_to_spz(h5ad_path, spz_path)
        assert spz_path.exists()

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
        from singlet.io.convert import from_h5ad, to_h5ad

        path = tmp_path / "test.h5ad"
        to_h5ad(int_adata, path)
        loaded = from_h5ad(path)
        assert loaded.shape == int_adata.shape
