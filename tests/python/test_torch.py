# SPDX-License-Identifier: MIT
"""Tests for PyTorch integration."""

import pytest

torch = pytest.importorskip("torch")


class TestTorchIntegration:
    """Test PyTorch sparse tensor creation and DataLoaders."""

    def test_from_anndata(self, small_anndata):
        from singlet.torch import from_anndata

        tensor = from_anndata(small_anndata, dtype="float32")
        assert tensor.is_sparse_csr
        assert tensor.shape == small_anndata.shape

    def test_from_anndata_float64(self, small_anndata):
        from singlet.torch import from_anndata

        tensor = from_anndata(small_anndata, dtype="float64")
        assert tensor.dtype == torch.float64

    def test_spz_dataset(self, small_anndata):
        from singlet.torch import SpzDataset

        ds = SpzDataset(small_anndata)
        assert len(ds) == small_anndata.n_obs
        assert ds.n_genes == small_anndata.n_vars

        cell = ds[0]
        assert cell.shape == (small_anndata.n_vars,)
        assert cell.dtype == torch.float32

    def test_spz_dataset_sparse(self, small_anndata):
        from singlet.torch import SpzDataset

        ds = SpzDataset(small_anndata, sparse=True)
        cell = ds[0]
        assert cell.is_sparse

    def test_dataloader(self, small_anndata):
        from singlet.torch import DataLoader

        loader = DataLoader(small_anndata, batch_size=3, shuffle=False)
        batches = list(loader)
        total = sum(b.shape[0] for b in batches)
        assert total == small_anndata.n_obs

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_from_anndata_cuda(self, small_anndata):
        from singlet.torch import from_anndata

        tensor = from_anndata(small_anndata, device="cuda")
        assert tensor.device.type == "cuda"
