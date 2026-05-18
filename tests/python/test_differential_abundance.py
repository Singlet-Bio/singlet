# SPDX-License-Identifier: MIT
"""Tests for singlet.differential_abundance() differential abundance testing."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._differential_abundance import differential_abundance


def _make_da_adata(n_cells=200, n_genes=100, n_clusters=4):
    """Create AnnData with cluster and condition annotations for DA testing.

    Deliberately unbalances condition assignment in some clusters to
    produce real differential abundance signal.
    """
    import anndata as ad

    rng = np.random.default_rng(42)

    X = sp.random(n_cells, n_genes, density=0.3, format="csr", random_state=42)
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Assign clusters evenly
    cluster_labels = [f"cluster_{i % n_clusters}" for i in range(n_cells)]
    adata.obs["cluster"] = pd.Categorical(cluster_labels)

    # Assign conditions: unbalanced in cluster_0 and cluster_1 for signal
    conditions = []
    for i in range(n_cells):
        c = i % n_clusters
        if c == 0:
            # 80% condition A
            conditions.append("A" if rng.random() < 0.8 else "B")
        elif c == 1:
            # 20% condition A
            conditions.append("A" if rng.random() < 0.2 else "B")
        else:
            # balanced
            conditions.append("A" if rng.random() < 0.5 else "B")
    adata.obs["condition"] = pd.Categorical(conditions)

    # PCA embedding for milo
    adata.obsm["X_pca"] = rng.standard_normal((n_cells, 20)).astype(np.float32)

    return adata


class TestDifferentialAbundanceSimple:
    def test_simple_method_basic(self):
        """Simple method returns DataFrame with expected columns."""
        adata = _make_da_adata()
        result = differential_abundance(adata, "cluster", "condition", method="simple")
        assert list(result.columns) == ["group", "n_cells", "logFC", "pvalue", "padj"]

    def test_simple_returns_dataframe(self):
        """Return type is pd.DataFrame."""
        adata = _make_da_adata()
        result = differential_abundance(adata, "cluster", "condition", method="simple")
        assert isinstance(result, pd.DataFrame)

    def test_simple_groups_match(self):
        """Groups in result match groups in adata."""
        adata = _make_da_adata(n_clusters=5)
        result = differential_abundance(adata, "cluster", "condition", method="simple")
        expected_groups = set(adata.obs["cluster"].unique())
        result_groups = set(result["group"].values)
        assert result_groups == expected_groups

    def test_simple_n_cells_sum(self):
        """Sum of n_cells across groups equals total cells."""
        adata = _make_da_adata()
        result = differential_abundance(adata, "cluster", "condition", method="simple")
        assert result["n_cells"].sum() == adata.n_obs


class TestDifferentialAbundanceMilo:
    def test_milo_method_basic(self):
        """Milo method returns DataFrame with expected columns."""
        adata = _make_da_adata()
        result = differential_abundance(adata, "cluster", "condition", method="milo")
        assert list(result.columns) == [
            "neighborhood_index",
            "n_cells",
            "logFC",
            "pvalue",
            "padj",
            "group",
        ]

    def test_milo_returns_dataframe(self):
        """Return type is pd.DataFrame."""
        adata = _make_da_adata()
        result = differential_abundance(adata, "cluster", "condition", method="milo")
        assert isinstance(result, pd.DataFrame)

    def test_milo_neighborhood_indices_valid(self):
        """Neighborhood indices should be valid cell indices."""
        adata = _make_da_adata()
        result = differential_abundance(adata, "cluster", "condition", method="milo")
        assert all(0 <= idx < adata.n_obs for idx in result["neighborhood_index"])

    def test_milo_groups_subset(self):
        """Groups in milo result should be a subset of groupby values."""
        adata = _make_da_adata()
        result = differential_abundance(adata, "cluster", "condition", method="milo")
        expected_groups = set(adata.obs["cluster"].unique())
        result_groups = set(result["group"].values)
        assert result_groups.issubset(expected_groups)


class TestDifferentialAbundanceStatistics:
    def test_pvalues_valid(self):
        """All p-values should be between 0 and 1."""
        adata = _make_da_adata()
        for method in ("simple", "milo"):
            result = differential_abundance(
                adata, "cluster", "condition", method=method
            )
            assert (result["pvalue"] >= 0).all()
            assert (result["pvalue"] <= 1).all()

    def test_padj_valid(self):
        """Adjusted p-values between 0 and 1, and >= raw p-values."""
        adata = _make_da_adata()
        for method in ("simple", "milo"):
            result = differential_abundance(
                adata, "cluster", "condition", method=method
            )
            assert (result["padj"] >= 0).all()
            assert (result["padj"] <= 1).all()
            assert (result["padj"] >= result["pvalue"] - 1e-10).all()

    def test_logfc_finite(self):
        """logFC values should all be finite."""
        adata = _make_da_adata()
        for method in ("simple", "milo"):
            result = differential_abundance(
                adata, "cluster", "condition", method=method
            )
            assert np.all(np.isfinite(result["logFC"].values))


class TestDifferentialAbundanceStorage:
    def test_stores_uns(self):
        """Result stored in adata.uns['differential_abundance']."""
        adata = _make_da_adata()
        result = differential_abundance(adata, "cluster", "condition", method="simple")
        assert "differential_abundance" in adata.uns
        assert "params" in adata.uns["differential_abundance"]
        assert "results" in adata.uns["differential_abundance"]
        pd.testing.assert_frame_equal(
            adata.uns["differential_abundance"]["results"], result
        )

    def test_stores_params(self):
        """Stored params match inputs."""
        adata = _make_da_adata()
        differential_abundance(adata, "cluster", "condition", method="milo", n_neighbors=15)
        params = adata.uns["differential_abundance"]["params"]
        assert params["groupby"] == "cluster"
        assert params["condition_key"] == "condition"
        assert params["method"] == "milo"
        assert params["n_neighbors"] == 15


class TestDifferentialAbundanceErrors:
    def test_type_error(self):
        """Non-AnnData raises TypeError."""
        with pytest.raises(TypeError, match="AnnData"):
            differential_abundance("not_adata", "cluster", "condition")

    def test_missing_groupby(self):
        """Missing groupby column raises KeyError."""
        adata = _make_da_adata()
        with pytest.raises(KeyError, match="nonexistent"):
            differential_abundance(adata, "nonexistent", "condition")

    def test_missing_condition_key(self):
        """Missing condition_key column raises KeyError."""
        adata = _make_da_adata()
        with pytest.raises(KeyError, match="nonexistent"):
            differential_abundance(adata, "cluster", "nonexistent")

    def test_invalid_method(self):
        """Invalid method raises ValueError."""
        adata = _make_da_adata()
        with pytest.raises(ValueError, match="method"):
            differential_abundance(adata, "cluster", "condition", method="invalid")

    def test_single_condition(self):
        """Fewer than 2 conditions raises ValueError."""
        adata = _make_da_adata()
        adata.obs["condition"] = "A"
        with pytest.raises(ValueError, match="at least 2"):
            differential_abundance(adata, "cluster", "condition")


class TestDifferentialAbundanceAPI:
    def test_public_api(self):
        """singlet.differential_abundance is callable."""
        assert hasattr(singlet, "differential_abundance")
        assert callable(singlet.differential_abundance)

    def test_reproducible(self):
        """Same random_state gives same results for milo."""
        adata1 = _make_da_adata()
        adata2 = _make_da_adata()
        r1 = differential_abundance(adata1, "cluster", "condition", method="milo", random_state=0)
        r2 = differential_abundance(adata2, "cluster", "condition", method="milo", random_state=0)
        pd.testing.assert_frame_equal(r1, r2)
