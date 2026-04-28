"""Tests for .spz I/O (read/write AnnData)."""

import numpy as np
import pytest


class TestSpzIO:
    """Test read_spz / write_spz round-trips through AnnData."""

    def test_anndata_roundtrip(self, tmp_path, small_anndata):
        """Write AnnData → .spz → read back, verify contents match."""
        try:
            from singlet._io import read_spz, write_spz
        except ImportError:
            pytest.skip("_singlepress extension not built")

        path = tmp_path / "roundtrip.spz"
        write_spz(small_anndata, path)

        adata2 = read_spz(path)

        # Shape should match
        assert adata2.shape == small_anndata.shape

        # Values should be close
        np.testing.assert_array_almost_equal(
            small_anndata.X.toarray(), adata2.X.toarray(), decimal=5
        )

        # Gene names preserved
        assert list(adata2.var_names) == list(small_anndata.var_names)

        # Cell names preserved
        assert list(adata2.obs_names) == list(small_anndata.obs_names)

    def test_spz_info(self, tmp_spz):
        from singlet._io import spz_info

        info = spz_info(tmp_spz)
        assert "rows" in info
        assert "cols" in info
        assert info["rows"] == 5
        assert info["cols"] == 10


class TestConversions:
    """Test format conversion functions."""

    def test_to_h5ad(self, tmp_path, small_anndata):
        from singlet.convert import to_h5ad, from_h5ad

        path = tmp_path / "test.h5ad"
        to_h5ad(small_anndata, path)

        adata2 = from_h5ad(path)
        assert adata2.shape == small_anndata.shape
        np.testing.assert_array_almost_equal(
            small_anndata.X.toarray(), adata2.X.toarray(), decimal=5
        )

    def test_to_csc(self, small_anndata):
        from singlet.convert import to_csc
        import scipy.sparse as sp

        csc = to_csc(small_anndata)
        assert sp.issparse(csc)
        assert csc.format == "csc"
        assert csc.shape == small_anndata.shape

    def test_to_mtx(self, tmp_path, small_anndata):
        from singlet.convert import to_mtx, from_mtx

        mtx_dir = tmp_path / "mtx_output"
        to_mtx(small_anndata, mtx_dir)

        assert (mtx_dir / "matrix.mtx.gz").exists()
        assert (mtx_dir / "barcodes.tsv.gz").exists()
        assert (mtx_dir / "features.tsv.gz").exists()

        adata2 = from_mtx(mtx_dir)
        assert adata2.shape == small_anndata.shape


class TestPreprocessing:
    """Test preprocessing utilities (no external tools needed)."""

    def test_species_lookup(self):
        from singlet.preprocessing import get_taxon_id, get_species_info

        assert get_taxon_id("human") == 9606
        assert get_taxon_id("Homo sapiens") == 9606

        info = get_species_info(9606)
        assert info["name"] == "human"
        assert "assembly" in info

    def test_species_list(self):
        from singlet.preprocessing import list_supported_species

        species = list_supported_species()
        assert len(species) > 20
        names = [s["name"] for s in species]
        assert "human" in names
        assert "mouse" in names

    def test_unknown_species(self):
        from singlet.preprocessing import get_taxon_id

        with pytest.raises(KeyError):
            get_taxon_id("unicorn")
