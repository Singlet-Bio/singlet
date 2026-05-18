# SPDX-License-Identifier: MIT
"""Comprehensive tests for singlet Python API (I/O, convert, torch, preprocessing).

Tests cover:
- Format conversions (h5ad, MTX)
- PyTorch integration (CSR, COO, Dataset, DataLoader)
- Preprocessing utilities (species lookup, protocol detection)
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def adata_int():
    """AnnData with integer counts (typical scRNA-seq)."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(42)
    dense = rng.poisson(lam=0.5, size=(50, 200)).astype(np.float64)
    X = sp.csr_matrix(dense)

    adata = ad.AnnData(X=X)
    adata.var_names = pd.Index([f"Gene{i}" for i in range(200)])
    adata.obs_names = pd.Index([f"Cell{i}" for i in range(50)])
    return adata


@pytest.fixture
def adata_float():
    """AnnData with float values (e.g. normalized)."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(42)
    dense = rng.random((30, 100)) * (rng.random((30, 100)) < 0.1)
    X = sp.csr_matrix(dense)

    adata = ad.AnnData(X=X)
    adata.var_names = pd.Index([f"Gene{i}" for i in range(100)])
    adata.obs_names = pd.Index([f"Cell{i}" for i in range(30)])
    return adata


@pytest.fixture
def adata_large():
    """Larger AnnData (1000 cells × 5000 genes)."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(42)
    dense = rng.poisson(lam=0.05, size=(1000, 5000)).astype(np.float64)
    X = sp.csr_matrix(dense)

    adata = ad.AnnData(X=X)
    adata.var_names = pd.Index([f"ENSG{i:011d}" for i in range(5000)])
    adata.obs_names = pd.Index([f"AAACCTGA{i:08d}-1" for i in range(1000)])
    return adata


# ============================================================================
# SECTION: Format conversions
# ============================================================================


class TestConversions:
    """Test convert module."""

    def test_to_from_h5ad(self, tmp_path, adata_int):
        from singlet.convert import from_h5ad, to_h5ad

        path = tmp_path / "test.h5ad"
        to_h5ad(adata_int, path)
        assert path.exists()

        adata2 = from_h5ad(path)
        assert adata2.shape == adata_int.shape
        np.testing.assert_array_almost_equal(adata_int.X.toarray(), adata2.X.toarray(), decimal=5)

    def test_to_csc(self, adata_int):
        from singlet.convert import to_csc

        csc = to_csc(adata_int)
        assert sp.issparse(csc)
        assert csc.format == "csc"
        assert csc.shape == adata_int.shape

    def test_to_from_mtx(self, tmp_path, adata_int):
        from singlet.convert import from_mtx, to_mtx

        mtx_dir = tmp_path / "mtx"
        to_mtx(adata_int, mtx_dir)

        assert (mtx_dir / "matrix.mtx.gz").exists()
        assert (mtx_dir / "barcodes.tsv.gz").exists()
        assert (mtx_dir / "features.tsv.gz").exists()

        adata2 = from_mtx(mtx_dir)
        assert adata2.shape == adata_int.shape



# ============================================================================
# SECTION: PyTorch integration
# ============================================================================

try:
    import torch as _torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTorchIntegration:
    """Test PyTorch sparse tensor creation."""

    def test_from_anndata_csr(self, adata_int):
        from singlet.torch import from_anndata

        t = from_anndata(adata_int, dtype="float32")
        assert t.is_sparse_csr
        assert t.shape == adata_int.shape

    def test_from_anndata_float64(self, adata_int):
        from singlet.torch import from_anndata

        t = from_anndata(adata_int, dtype="float64")
        assert t.dtype == _torch.float64

    def test_from_anndata_values(self, adata_int):
        """Verify tensor values match AnnData."""
        from singlet.torch import from_anndata

        t = from_anndata(adata_int, dtype="float64")
        dense_torch = t.to_dense().numpy()
        dense_orig = adata_int.X.toarray()
        np.testing.assert_array_almost_equal(dense_torch, dense_orig, decimal=5)

    def test_spz_dataset_length(self, adata_int):
        from singlet.torch import SpzDataset

        ds = SpzDataset(adata_int)
        assert len(ds) == adata_int.n_obs

    def test_spz_dataset_item(self, adata_int):
        from singlet.torch import SpzDataset

        ds = SpzDataset(adata_int)
        cell = ds[0]
        assert cell.shape == (adata_int.n_vars,)
        assert cell.dtype == _torch.float32

    def test_spz_dataset_sparse(self, adata_int):
        from singlet.torch import SpzDataset

        ds = SpzDataset(adata_int, sparse=True)
        cell = ds[0]
        assert cell.is_sparse

    def test_dataloader_batches(self, adata_int):
        from singlet.torch import DataLoader

        loader = DataLoader(adata_int, batch_size=10, shuffle=False)
        total = 0
        for batch in loader:
            assert batch.shape[1] == adata_int.n_vars
            total += batch.shape[0]
        assert total == adata_int.n_obs

    def test_dataloader_all_data(self, adata_int):
        """Verify DataLoader yields all the data."""
        from singlet.torch import DataLoader

        loader = DataLoader(adata_int, batch_size=50, shuffle=False)
        batches = list(loader)
        assert len(batches) == 1
        result = batches[0].numpy()
        np.testing.assert_array_almost_equal(result, adata_int.X.toarray(), decimal=5)

    def test_gene_subsetting(self, adata_int):
        from singlet.torch import SpzDataset

        genes = list(adata_int.var_names[:10])
        ds = SpzDataset(adata_int, genes=genes)
        assert ds.n_genes == 10

        cell = ds[0]
        assert cell.shape == (10,)


# ============================================================================
# SECTION: Preprocessing utilities
# ============================================================================


class TestPreprocessing:
    """Test preprocessing utilities."""

    def test_species_human(self):
        from singlet.preprocessing import get_species_info, get_taxon_id

        assert get_taxon_id("human") == 9606
        assert get_taxon_id("Homo sapiens") == 9606

        info = get_species_info(9606)
        assert info["name"] == "human"
        assert "assembly" in info

    def test_species_mouse(self):
        from singlet.preprocessing import get_taxon_id

        assert get_taxon_id("mouse") == 10090
        assert get_taxon_id("Mus musculus") == 10090

    def test_species_list(self):
        from singlet.preprocessing import list_supported_species

        species = list_supported_species()
        assert len(species) > 20
        names = {s["name"] for s in species}
        assert "human" in names
        assert "mouse" in names

    def test_unknown_species(self):
        from singlet.preprocessing import get_taxon_id

        with pytest.raises(KeyError):
            get_taxon_id("unicorn")

    def test_protocol_detection_heuristic(self):
        """Test the protocol inference from read lengths."""
        from singlet.preprocessing._detect import _infer_protocol

        # 10x v3: R1=28bp (bc+UMI), R2=91bp (cDNA)
        result = _infer_protocol(28, 91)
        assert result.protocol == "10xv3"
        assert result.mode == "droplet"
        assert result.confidence == "high"

        # 10x v2: R1=26bp, R2=98bp
        result = _infer_protocol(26, 98)
        assert result.protocol == "10xv2"
        assert result.mode == "droplet"

        # Catalog hint overrides
        result = _infer_protocol(50, 50, catalog_hint="10xv3 chemistry")
        assert result.protocol == "10xv3"
        assert result.confidence == "high"

        # Ambiguous: both long
        result = _infer_protocol(100, 100)
        assert result.confidence == "low"

    def test_protocol_detection_dataclass(self):
        from singlet.preprocessing import ProtocolDetection

        pd = ProtocolDetection(
            protocol="10xv3",
            mode="droplet",
            confidence="high",
            reason="test",
            r1_len=28,
            r2_len=91,
            chemistry="10xv3",
        )
        assert pd.protocol == "10xv3"
        assert pd.chemistry == "10xv3"


class TestCatalogNewFunctions:
    """Tests for summary(), samples(), and top_series()."""

    def test_summary_returns_string(self):
        import singlet

        result = singlet.summary()
        assert isinstance(result, str)
        assert "singlet atlas" in result
        assert "samples" in result

    def test_top_series_returns_dataframe(self):
        import singlet

        result = singlet.top_series(n=5)
        assert hasattr(result, "columns")
        assert "gse_id" in result.columns
        assert "total_cells" in result.columns
        assert len(result) <= 5
        # Should be sorted descending by total_cells
        if len(result) > 1:
            assert result["total_cells"].iloc[0] >= result["total_cells"].iloc[1]

    def test_top_series_min_samples(self):
        import singlet

        result = singlet.top_series(min_samples=5)
        assert all(result["n_samples"] >= 5)

    def test_samples_returns_dataframe(self):
        import singlet

        result = singlet.samples()
        assert hasattr(result, "columns")
        assert "gse_id" in result.columns
