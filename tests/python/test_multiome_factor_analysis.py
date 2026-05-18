# SPDX-License-Identifier: MIT
"""Tests for singlet.multiome_factor_analysis."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import singlet


@pytest.fixture()
def adata_multiome():
    """Create AnnData with two modalities in obsm."""
    import anndata as ad

    rng = np.random.default_rng(77)
    n_cells = 100
    n_genes = 200

    X = rng.poisson(3, size=(n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=X)

    # RNA modality: 100 features
    adata.obsm["X_rna"] = rng.normal(0, 1, size=(n_cells, 100)).astype(
        np.float64
    )
    # ATAC modality: 80 features
    adata.obsm["X_atac"] = rng.normal(0, 1, size=(n_cells, 80)).astype(
        np.float64
    )

    return adata


@pytest.fixture()
def adata_layers():
    """Create AnnData with modalities in layers."""
    import anndata as ad

    rng = np.random.default_rng(88)
    n_cells = 60
    n_genes = 50

    X = rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=X)

    # Store modalities as layers
    adata.layers["spliced"] = rng.poisson(
        3, size=(n_cells, n_genes)
    ).astype(np.float64)
    adata.layers["unspliced"] = rng.poisson(
        1, size=(n_cells, n_genes)
    ).astype(np.float64)

    return adata


def test_svd_basic(adata_multiome):
    """Test SVD method returns expected structure."""
    result = singlet.multiome_factor_analysis(
        adata_multiome, modality_keys=["X_rna", "X_atac"], n_factors=10
    )

    assert result is adata_multiome
    assert "X_mofa" in adata_multiome.obsm
    assert adata_multiome.obsm["X_mofa"].shape == (100, 10)
    assert "mofa_loadings" in adata_multiome.uns
    assert "X_rna" in adata_multiome.uns["mofa_loadings"]
    assert "X_atac" in adata_multiome.uns["mofa_loadings"]
    assert adata_multiome.uns["mofa_loadings"]["X_rna"].shape == (10, 100)
    assert adata_multiome.uns["mofa_loadings"]["X_atac"].shape == (10, 80)


def test_nmf_basic(adata_multiome):
    """Test NMF method returns expected structure."""
    # Make data non-negative for NMF
    adata_multiome.obsm["X_rna"] = np.abs(adata_multiome.obsm["X_rna"])
    adata_multiome.obsm["X_atac"] = np.abs(adata_multiome.obsm["X_atac"])

    result = singlet.multiome_factor_analysis(
        adata_multiome,
        modality_keys=["X_rna", "X_atac"],
        n_factors=5,
        method="nmf",
    )

    assert "X_mofa" in adata_multiome.obsm
    assert adata_multiome.obsm["X_mofa"].shape == (100, 5)
    # NMF factors should be non-negative
    assert np.all(adata_multiome.obsm["X_mofa"] >= 0)


def test_layers_as_modalities(adata_layers):
    """Test using layers as modality keys."""
    result = singlet.multiome_factor_analysis(
        adata_layers, modality_keys=["spliced", "unspliced"], n_factors=5
    )

    assert "X_mofa" in adata_layers.obsm
    assert adata_layers.obsm["X_mofa"].shape == (60, 5)
    assert adata_layers.uns["mofa_loadings"]["spliced"].shape == (5, 50)
    assert adata_layers.uns["mofa_loadings"]["unspliced"].shape == (5, 50)


def test_params_stored(adata_multiome):
    """Test that parameters are stored in uns."""
    singlet.multiome_factor_analysis(
        adata_multiome,
        modality_keys=["X_rna", "X_atac"],
        n_factors=8,
        method="svd",
        random_state=42,
    )

    params = adata_multiome.uns["mofa_params"]
    assert params["modality_keys"] == ["X_rna", "X_atac"]
    assert params["n_factors"] == 8
    assert params["method"] == "svd"
    assert params["random_state"] == 42


def test_sparse_input():
    """Test that sparse modality matrices are handled."""
    import anndata as ad

    rng = np.random.default_rng(55)
    n_cells = 80
    n_genes = 100

    X = rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=X)

    # Sparse modality
    dense = rng.normal(0, 1, size=(n_cells, 60))
    adata.obsm["X_mod1"] = sp.csr_matrix(dense)
    adata.obsm["X_mod2"] = rng.normal(0, 1, size=(n_cells, 40))

    singlet.multiome_factor_analysis(
        adata, modality_keys=["X_mod1", "X_mod2"], n_factors=5
    )

    assert adata.obsm["X_mofa"].shape == (80, 5)


def test_reproducibility(adata_multiome):
    """Test that same random_state gives same results."""
    import anndata as ad

    adata_copy = ad.AnnData(
        X=adata_multiome.X.copy(),
        obsm={
            "X_rna": adata_multiome.obsm["X_rna"].copy(),
            "X_atac": adata_multiome.obsm["X_atac"].copy(),
        },
    )

    singlet.multiome_factor_analysis(
        adata_multiome,
        modality_keys=["X_rna", "X_atac"],
        n_factors=5,
        random_state=0,
    )
    singlet.multiome_factor_analysis(
        adata_copy,
        modality_keys=["X_rna", "X_atac"],
        n_factors=5,
        random_state=0,
    )

    np.testing.assert_array_almost_equal(
        adata_multiome.obsm["X_mofa"], adata_copy.obsm["X_mofa"]
    )


def test_missing_key_raises(adata_multiome):
    """Should raise KeyError if modality key not found."""
    with pytest.raises(KeyError, match="not found"):
        singlet.multiome_factor_analysis(
            adata_multiome, modality_keys=["X_rna", "X_nonexistent"]
        )


def test_invalid_method_raises(adata_multiome):
    """Should raise ValueError for invalid method."""
    with pytest.raises(ValueError, match="Method must be"):
        singlet.multiome_factor_analysis(
            adata_multiome,
            modality_keys=["X_rna", "X_atac"],
            method="pca",
        )


def test_too_many_factors_raises(adata_multiome):
    """Should raise ValueError if n_factors too large."""
    with pytest.raises(ValueError, match="n_factors"):
        singlet.multiome_factor_analysis(
            adata_multiome,
            modality_keys=["X_rna", "X_atac"],
            n_factors=500,
        )


def test_three_modalities():
    """Test with three modalities."""
    import anndata as ad

    rng = np.random.default_rng(33)
    n_cells = 50

    X = rng.poisson(2, size=(n_cells, 30)).astype(np.float32)
    adata = ad.AnnData(X=X)

    adata.obsm["X_rna"] = rng.normal(0, 1, size=(n_cells, 40))
    adata.obsm["X_atac"] = rng.normal(0, 1, size=(n_cells, 30))
    adata.obsm["X_protein"] = rng.normal(0, 1, size=(n_cells, 20))

    singlet.multiome_factor_analysis(
        adata,
        modality_keys=["X_rna", "X_atac", "X_protein"],
        n_factors=10,
    )

    assert adata.obsm["X_mofa"].shape == (50, 10)
    assert adata.uns["mofa_loadings"]["X_rna"].shape == (10, 40)
    assert adata.uns["mofa_loadings"]["X_atac"].shape == (10, 30)
    assert adata.uns["mofa_loadings"]["X_protein"].shape == (10, 20)
