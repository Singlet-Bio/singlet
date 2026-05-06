"""Tests for singlet.fate_probabilities()."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._fate_probabilities import fate_probabilities


def _make_trajectory_adata(n_cells=100, n_genes=50):
    """Create test AnnData with pseudotime and PCA for fate probability testing."""
    import anndata as ad

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    # Create pseudotime: linear from 0 to 1
    adata.obs["dpt_pseudotime"] = np.linspace(0, 1, n_cells)

    # Create PCA embedding (2D for simplicity)
    adata.obsm["X_pca"] = rng.standard_normal((n_cells, 20)).astype(np.float32)

    return adata


def _make_branching_adata():
    """Create AnnData with branching trajectory (2 fates)."""
    import anndata as ad

    rng = np.random.default_rng(42)
    n_cells = 80
    n_genes = 30

    X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    # Branch 1: cells 0-39 (low pseudotime → fate A at end)
    # Branch 2: cells 40-79 (low pseudotime → fate B at end)
    pseudotime = np.zeros(n_cells)
    pseudotime[:40] = np.linspace(0, 1, 40)
    pseudotime[40:] = np.linspace(0, 1, 40)
    adata.obs["dpt_pseudotime"] = pseudotime

    # PCA: separate the two branches
    pca_embed = np.zeros((n_cells, 10), dtype=np.float32)
    pca_embed[:40, 0] = np.linspace(0, 5, 40)
    pca_embed[:40, 1] = 1.0
    pca_embed[40:, 0] = np.linspace(0, 5, 40)
    pca_embed[40:, 1] = -1.0
    adata.obsm["X_pca"] = pca_embed

    return adata


class TestFateProbabilities:
    def test_basic_dict_input(self):
        adata = _make_trajectory_adata()
        # Terminal states: first 5 and last 5 cells
        terminal = {
            "fate_A": list(range(90, 95)),
            "fate_B": list(range(95, 100)),
        }
        result = fate_probabilities(adata, terminal)
        assert result is adata
        assert "fate_probabilities" in adata.obsm
        fp = adata.obsm["fate_probabilities"]
        assert isinstance(fp, pd.DataFrame)
        assert fp.shape == (100, 2)
        assert list(fp.columns) == ["fate_A", "fate_B"]

    def test_probabilities_sum_to_one(self):
        adata = _make_trajectory_adata()
        terminal = {
            "fate_A": list(range(90, 95)),
            "fate_B": list(range(95, 100)),
        }
        fate_probabilities(adata, terminal)
        fp = adata.obsm["fate_probabilities"]
        row_sums = fp.sum(axis=1).values
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_terminal_cells_have_prob_one(self):
        adata = _make_trajectory_adata()
        terminal = {
            "fate_A": [90, 91, 92],
            "fate_B": [97, 98, 99],
        }
        fate_probabilities(adata, terminal)
        fp = adata.obsm["fate_probabilities"]
        # Terminal cells of fate_A should have prob 1.0 for fate_A
        assert fp.iloc[90]["fate_A"] == pytest.approx(1.0, abs=0.01)
        assert fp.iloc[91]["fate_A"] == pytest.approx(1.0, abs=0.01)
        # Terminal cells of fate_B should have prob 1.0 for fate_B
        assert fp.iloc[97]["fate_B"] == pytest.approx(1.0, abs=0.01)
        assert fp.iloc[99]["fate_B"] == pytest.approx(1.0, abs=0.01)

    def test_string_terminal_states(self):
        adata = _make_trajectory_adata()
        # Add obs column with terminal labels
        labels = [None] * 100
        for idx in range(90, 95):
            labels[idx] = "fate_A"
        for idx in range(95, 100):
            labels[idx] = "fate_B"
        adata.obs["terminal"] = pd.Categorical(
            labels, categories=["fate_A", "fate_B"]
        )
        fate_probabilities(adata, "terminal")
        fp = adata.obsm["fate_probabilities"]
        assert fp.shape == (100, 2)
        assert "fate_A" in fp.columns
        assert "fate_B" in fp.columns

    def test_uns_metadata(self):
        adata = _make_trajectory_adata()
        terminal = {"fate_A": [95, 96, 97], "fate_B": [98, 99]}
        fate_probabilities(adata, terminal, n_neighbors=10)
        assert "fate_probabilities" in adata.uns
        params = adata.uns["fate_probabilities"]["params"]
        assert params["n_neighbors"] == 10
        assert params["pseudotime_key"] == "dpt_pseudotime"
        assert adata.uns["fate_probabilities"]["fate_names"] == ["fate_A", "fate_B"]

    def test_missing_pseudotime_raises(self):
        import anndata as ad

        adata = ad.AnnData(X=np.ones((10, 5)))
        with pytest.raises(KeyError, match="Pseudotime key"):
            fate_probabilities(adata, {"A": [8, 9]})

    def test_invalid_terminal_states_type(self):
        adata = _make_trajectory_adata()
        with pytest.raises(TypeError, match="terminal_states must be"):
            fate_probabilities(adata, [1, 2, 3])

    def test_missing_obs_key_raises(self):
        adata = _make_trajectory_adata()
        with pytest.raises(KeyError, match="not found in adata.obs"):
            fate_probabilities(adata, "nonexistent_key")

    def test_type_error_non_adata(self):
        with pytest.raises(TypeError, match="requires an AnnData object"):
            fate_probabilities(np.zeros((10, 5)), {"A": [1]})

    def test_values_bounded_zero_one(self):
        adata = _make_trajectory_adata()
        terminal = {
            "fate_A": list(range(90, 95)),
            "fate_B": list(range(95, 100)),
        }
        fate_probabilities(adata, terminal)
        fp = adata.obsm["fate_probabilities"]
        assert (fp.values >= 0).all()
        assert (fp.values <= 1.0 + 1e-6).all()

    def test_custom_pseudotime_key(self):
        adata = _make_trajectory_adata()
        adata.obs["my_pseudotime"] = adata.obs["dpt_pseudotime"]
        terminal = {"A": [95, 96, 97, 98, 99]}
        fate_probabilities(adata, terminal, pseudotime_key="my_pseudotime")
        assert "fate_probabilities" in adata.obsm

    def test_single_fate(self):
        """Single terminal state: most cells should reach it with high probability."""
        adata = _make_trajectory_adata()
        # Use cells at the end of pseudotime as the single terminal
        terminal = {"only_fate": list(range(95, 100))}
        fate_probabilities(adata, terminal)
        fp = adata.obsm["fate_probabilities"]
        assert fp.shape == (100, 1)
        # Terminal cells should have probability 1
        for idx in range(95, 100):
            assert fp["only_fate"].iloc[idx] == pytest.approx(1.0, abs=0.01)
        # All values should be between 0 and 1
        assert (fp.values >= 0).all()
        assert (fp.values <= 1.0 + 1e-6).all()

    def test_registered_in_singlet(self):
        assert hasattr(singlet, "fate_probabilities")
        assert "fate_probabilities" in singlet.__all__
