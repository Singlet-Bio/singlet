"""Tests for singlet.pseudobulk()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=90, m=30, seed=42):
    rng = np.random.default_rng(seed)
    adata = AnnData(X=rng.poisson(5, size=(n, m)).astype(np.float32))
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs["group"] = [f"g{i % 3}" for i in range(n)]
    adata.obs["condition"] = [f"cond{i % 2}" for i in range(n)]
    return adata


def test_pseudobulk_sum():
    adata = _make_adata()
    result = singlet.pseudobulk(adata, "group", agg="sum")
    assert result.n_obs == 3
    assert result.n_vars == 30


def test_pseudobulk_mean():
    adata = _make_adata()
    result = singlet.pseudobulk(adata, "group", agg="mean")
    assert result.n_obs == 3


def test_pseudobulk_median():
    adata = _make_adata()
    result = singlet.pseudobulk(adata, "group", agg="median")
    assert result.n_obs == 3


def test_pseudobulk_n_cells():
    adata = _make_adata()
    result = singlet.pseudobulk(adata, "group")
    assert "n_cells" in result.obs.columns
    assert result.obs["n_cells"].sum() == adata.n_obs


def test_pseudobulk_sum_correct():
    adata = _make_adata()
    result = singlet.pseudobulk(adata, "group", agg="sum")
    # Check g0 manually
    mask = adata.obs["group"] == "g0"
    expected = np.asarray(adata.X[mask.values]).sum(axis=0)
    # Find g0 row in result
    g0_idx = list(result.obs_names).index("g0")
    assert np.allclose(result.X[g0_idx], expected, atol=1e-5)


def test_pseudobulk_mean_correct():
    adata = _make_adata()
    result = singlet.pseudobulk(adata, "group", agg="mean")
    mask = adata.obs["group"] == "g1"
    expected = np.asarray(adata.X[mask.values]).mean(axis=0)
    g1_idx = list(result.obs_names).index("g1")
    assert np.allclose(result.X[g1_idx], expected, atol=1e-5)


def test_pseudobulk_multi_key():
    adata = _make_adata()
    result = singlet.pseudobulk(adata, ["group", "condition"])
    # 3 groups × 2 conditions = 6 combinations
    assert result.n_obs == 6
    assert "group" in result.obs.columns
    assert "condition" in result.obs.columns
    assert "n_cells" in result.obs.columns


def test_pseudobulk_multi_key_n_cells():
    adata = _make_adata()
    result = singlet.pseudobulk(adata, ["group", "condition"])
    assert result.obs["n_cells"].sum() == adata.n_obs


def test_pseudobulk_layer():
    adata = _make_adata()
    adata.layers["raw"] = adata.X.copy() * 2
    result_x = singlet.pseudobulk(adata, "group", agg="mean")
    result_layer = singlet.pseudobulk(adata, "group", layer="raw", agg="mean")
    assert np.allclose(result_layer.X, result_x.X * 2, atol=1e-4)


def test_pseudobulk_invalid_agg_raises():
    adata = _make_adata()
    with pytest.raises(ValueError, match="agg"):
        singlet.pseudobulk(adata, "group", agg="invalid")


def test_pseudobulk_missing_key_raises():
    adata = _make_adata()
    with pytest.raises(KeyError, match="nonexistent"):
        singlet.pseudobulk(adata, "nonexistent")


def test_pseudobulk_missing_layer_raises():
    adata = _make_adata()
    with pytest.raises(KeyError, match="missing_layer"):
        singlet.pseudobulk(adata, "group", layer="missing_layer")


def test_pseudobulk_preserves_var():
    adata = _make_adata()
    adata.var["feature_type"] = "Gene Expression"
    result = singlet.pseudobulk(adata, "group")
    assert "feature_type" in result.var.columns
    assert (result.var["feature_type"] == "Gene Expression").all()


def test_pseudobulk_sparse_input():
    from scipy.sparse import csr_matrix

    adata = _make_adata()
    adata.X = csr_matrix(adata.X)
    result = singlet.pseudobulk(adata, "group", agg="sum")
    assert result.n_obs == 3
    # Check correctness
    mask = adata.obs["group"] == "g0"
    expected = np.asarray(adata.X[mask.values].todense()).sum(axis=0)
    g0_idx = list(result.obs_names).index("g0")
    assert np.allclose(result.X[g0_idx], expected, atol=1e-5)


def test_pseudobulk_single_group():
    adata = _make_adata()
    adata.obs["group"] = "all"
    result = singlet.pseudobulk(adata, "group", agg="mean")
    assert result.n_obs == 1
    assert result.obs["n_cells"].iloc[0] == adata.n_obs


def test_pseudobulk_obs_names():
    adata = _make_adata()
    result = singlet.pseudobulk(adata, "group")
    assert set(result.obs_names) == {"g0", "g1", "g2"}
