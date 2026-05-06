"""Tests for singlet.score_cell_cycle()."""

import numpy as np
import singlet
from anndata import AnnData


def _make_adata_with_cc_genes(n=80, seed=42):
    """Create AnnData with some cell cycle gene names."""
    rng = np.random.default_rng(seed)
    # Include some known S and G2M genes
    cc_genes = ["MCM5", "PCNA", "CDK1", "TOP2A", "MKI67"]
    other_genes = [f"gene_{i}" for i in range(45)]
    all_genes = cc_genes + other_genes
    m = len(all_genes)

    X = rng.poisson(5, size=(n, m)).astype(np.float32)
    # Make CC genes higher in some cells (simulate cycling)
    X[:20, :5] += 10  # First 20 cells are cycling

    adata = AnnData(X=np.log1p(X))
    adata.var_names = all_genes
    return adata


def test_cell_cycle_basic():
    adata = _make_adata_with_cc_genes()
    singlet.score_cell_cycle(adata)
    assert "S_score" in adata.obs.columns
    assert "G2M_score" in adata.obs.columns
    assert "phase" in adata.obs.columns


def test_cell_cycle_phases():
    adata = _make_adata_with_cc_genes()
    singlet.score_cell_cycle(adata)
    phases = adata.obs["phase"].unique()
    for phase in phases:
        assert phase in ["G1", "S", "G2M"]


def test_cell_cycle_copy():
    adata = _make_adata_with_cc_genes()
    result = singlet.score_cell_cycle(adata, copy=True)
    assert result is not None
    assert "phase" not in adata.obs.columns
    assert "phase" in result.obs.columns


def test_cell_cycle_custom_genes():
    rng = np.random.default_rng(42)
    adata = AnnData(X=np.log1p(rng.poisson(5, size=(50, 20)).astype(np.float32)))
    adata.var_names = [f"g{i}" for i in range(20)]
    singlet.score_cell_cycle(
        adata,
        s_genes=["g0", "g1", "g2"],
        g2m_genes=["g5", "g6", "g7"],
    )
    assert "phase" in adata.obs.columns


def test_cell_cycle_no_cc_genes():
    """When no CC genes are present, all should be G1."""
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((30, 10)).astype(np.float32))
    adata.var_names = [f"random_{i}" for i in range(10)]
    singlet.score_cell_cycle(adata)
    assert (adata.obs["phase"] == "G1").all()


def test_cell_cycle_scores_finite():
    adata = _make_adata_with_cc_genes()
    singlet.score_cell_cycle(adata)
    assert np.all(np.isfinite(adata.obs["S_score"].values))
    assert np.all(np.isfinite(adata.obs["G2M_score"].values))
