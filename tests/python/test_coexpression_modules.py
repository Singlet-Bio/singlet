"""Tests for singlet.coexpression_modules()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata(n_obs=100, n_vars=200, seed=42):
    """Create test AnnData with correlated gene modules."""
    rng = np.random.default_rng(seed)

    # Ensure at least 120 genes for the 3 modules
    actual_vars = max(n_vars, 120)
    X = rng.standard_normal((n_obs, actual_vars)).astype(np.float32)

    # Module 1: genes 0-39 correlated via shared factor
    factor1 = rng.standard_normal(n_obs)
    for i in range(40):
        X[:, i] += factor1 * 3.0

    # Module 2: genes 40-79 correlated via different factor
    factor2 = rng.standard_normal(n_obs)
    for i in range(40, 80):
        X[:, i] += factor2 * 3.0

    # Module 3: genes 80-119 correlated
    factor3 = rng.standard_normal(n_obs)
    for i in range(80, 120):
        X[:, i] += factor3 * 3.0

    # Trim to requested n_vars if smaller was requested but we expanded
    X = X[:, :actual_vars]

    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(actual_vars)]
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    return adata


def test_coexpression_modules_basic():
    """Basic invocation returns expected structure."""
    adata = _make_adata()
    result = singlet.coexpression_modules(adata, n_modules=5, min_module_size=10)

    assert "modules" in result
    assert "eigengenes" in result
    assert "n_modules" in result
    assert result["n_modules"] >= 1
    assert "coexpression_module" in adata.var.columns
    assert "module_eigengenes" in adata.obsm


def test_coexpression_modules_var_labels():
    """Module labels are stored correctly in adata.var."""
    adata = _make_adata()
    singlet.coexpression_modules(adata, n_modules=5, min_module_size=10)

    labels = adata.var["coexpression_module"].values
    assert labels.dtype in (np.int32, np.int64)
    # At least some genes assigned
    assert (labels > 0).sum() > 0


def test_coexpression_modules_eigengenes_shape():
    """Module eigengenes have correct dimensions."""
    adata = _make_adata()
    result = singlet.coexpression_modules(adata, n_modules=4, min_module_size=10)

    eigengenes = adata.obsm["module_eigengenes"]
    assert eigengenes.shape[0] == adata.n_obs
    assert eigengenes.shape[1] == result["n_modules"]


def test_coexpression_modules_wgcna_method():
    """wgcna_lite method works correctly."""
    adata = _make_adata()
    result = singlet.coexpression_modules(
        adata, n_modules=3, method="wgcna_lite", min_module_size=10, power=6
    )
    assert result["n_modules"] >= 1


def test_coexpression_modules_correlation_method():
    """correlation_cluster method works correctly."""
    adata = _make_adata()
    result = singlet.coexpression_modules(
        adata, n_modules=3, method="correlation_cluster", min_module_size=10
    )
    assert result["n_modules"] >= 1


def test_coexpression_modules_min_module_size():
    """Small modules are merged to unassigned (0)."""
    adata = _make_adata(n_obs=80, n_vars=200)
    result = singlet.coexpression_modules(
        adata, n_modules=10, min_module_size=50, n_top_genes=200
    )
    # With high min_module_size, all surviving modules meet threshold
    for mod_id, genes in result["modules"].items():
        assert len(genes) >= 50


def test_coexpression_modules_n_top_genes():
    """n_top_genes parameter limits genes used."""
    adata = _make_adata(n_obs=60, n_vars=300)
    singlet.coexpression_modules(adata, n_modules=3, n_top_genes=100, min_module_size=10)

    # Only top 100 genes should have non-zero labels
    labels = adata.var["coexpression_module"].values
    assert (labels > 0).sum() <= 100


def test_coexpression_modules_invalid_method():
    """Invalid method raises ValueError."""
    adata = _make_adata()
    with pytest.raises(ValueError, match="method must be one of"):
        singlet.coexpression_modules(adata, method="invalid")


def test_coexpression_modules_invalid_n_top():
    """n_top_genes < min_module_size raises ValueError."""
    adata = _make_adata()
    with pytest.raises(ValueError, match="n_top_genes"):
        singlet.coexpression_modules(adata, n_top_genes=5, min_module_size=30)


def test_coexpression_modules_eigengenes_finite():
    """Eigengenes should be finite values."""
    adata = _make_adata()
    singlet.coexpression_modules(adata, n_modules=3, min_module_size=10)
    assert np.all(np.isfinite(adata.obsm["module_eigengenes"]))


def test_coexpression_modules_uns_stored():
    """Results are stored in adata.uns."""
    adata = _make_adata()
    singlet.coexpression_modules(adata, n_modules=4, min_module_size=10)
    assert "coexpression_modules" in adata.uns
    assert "params" in adata.uns["coexpression_modules"]


def test_coexpression_modules_sparse_input():
    """Works with sparse input matrix."""
    from scipy.sparse import csr_matrix

    adata = _make_adata(n_obs=60, n_vars=150)
    adata.X = csr_matrix(adata.X)
    result = singlet.coexpression_modules(adata, n_modules=3, min_module_size=10)
    assert result["n_modules"] >= 1
