"""Tests for singlet.subsample()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from singlet._subsample import subsample


def _make_adata(n_cells=500, n_genes=200):
    import anndata as ad

    rng = np.random.default_rng(42)
    X = sp.random(n_cells, n_genes, density=0.2, format="csr", random_state=42)
    X.data = rng.integers(1, 50, size=X.nnz).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.obs["batch"] = ["A"] * (n_cells // 2) + ["B"] * (n_cells - n_cells // 2)
    return adata


class TestSubsample:
    def test_n_obs(self):
        adata = _make_adata()
        result = subsample(adata, n_obs=100)
        assert result.shape[0] == 100
        assert result.shape[1] == 200

    def test_fraction(self):
        adata = _make_adata()
        result = subsample(adata, fraction=0.2)
        assert result.shape[0] == 100

    def test_returns_copy(self):
        adata = _make_adata()
        result = subsample(adata, n_obs=50)
        assert result is not adata
        assert adata.shape[0] == 500  # original unchanged

    def test_inplace(self):
        adata = _make_adata()
        ret = subsample(adata, n_obs=50, copy=False)
        assert ret is None
        assert adata.shape[0] == 50

    def test_preserves_obs(self):
        adata = _make_adata()
        result = subsample(adata, n_obs=100)
        assert "batch" in result.obs.columns

    def test_reproducible(self):
        adata = _make_adata()
        r1 = subsample(adata, n_obs=50, random_state=42)
        r2 = subsample(adata, n_obs=50, random_state=42)
        assert list(r1.obs_names) == list(r2.obs_names)

    def test_different_seeds(self):
        adata = _make_adata()
        r1 = subsample(adata, n_obs=50, random_state=0)
        r2 = subsample(adata, n_obs=50, random_state=1)
        assert list(r1.obs_names) != list(r2.obs_names)

    def test_n_obs_exceeds_total(self):
        adata = _make_adata(n_cells=50)
        result = subsample(adata, n_obs=100)
        assert result.shape[0] == 50  # clamped to total

    def test_fraction_validation(self):
        adata = _make_adata()
        with pytest.raises(ValueError, match="fraction"):
            subsample(adata, fraction=0)
        with pytest.raises(ValueError, match="fraction"):
            subsample(adata, fraction=1.5)

    def test_both_params_raises(self):
        adata = _make_adata()
        with pytest.raises(ValueError, match="either"):
            subsample(adata, n_obs=10, fraction=0.1)

    def test_neither_param_raises(self):
        adata = _make_adata()
        with pytest.raises(ValueError, match="Must specify"):
            subsample(adata)

    def test_type_error(self):
        with pytest.raises(TypeError, match="subsample"):
            subsample("not_adata", n_obs=10)

    def test_public_api(self):
        assert hasattr(singlet, "subsample")
        assert callable(singlet.subsample)
