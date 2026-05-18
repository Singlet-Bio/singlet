# SPDX-License-Identifier: MIT
"""Tests for singlet.cell_communication()."""

import numpy as np
import pandas as pd
import pytest
import singlet
from anndata import AnnData
from scipy import sparse


def _make_adata(n_cells=120, n_genes=100, seed=42):
    """Create test AnnData with cell type annotations."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(3, (n_cells, n_genes)).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    # Assign 3 cell types
    adata.obs["leiden"] = pd.Categorical(
        [f"cluster_{i % 3}" for i in range(n_cells)]
    )
    return adata


def _make_lr_pairs():
    """Create ligand-receptor pairs using genes that exist in test data."""
    return [
        ("GENE0", "GENE1"),
        ("GENE2", "GENE3"),
        ("GENE4", "GENE5"),
        ("GENE10", "GENE20"),
    ]


class TestCellCommunication:
    def test_basic_permutation(self):
        adata = _make_adata()
        pairs = _make_lr_pairs()
        result = singlet.cell_communication(
            adata, pairs, groupby="leiden", method="permutation", n_perms=50
        )
        assert isinstance(result, pd.DataFrame)
        assert "ligand" in result.columns
        assert "receptor" in result.columns
        assert "sender" in result.columns
        assert "receiver" in result.columns
        assert "score" in result.columns
        assert "pvalue" in result.columns
        assert "fdr" in result.columns

    def test_basic_expression(self):
        adata = _make_adata()
        pairs = _make_lr_pairs()
        result = singlet.cell_communication(
            adata, pairs, groupby="leiden", method="expression"
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "score" in result.columns

    def test_stores_in_uns(self):
        adata = _make_adata()
        pairs = _make_lr_pairs()
        singlet.cell_communication(
            adata, pairs, groupby="leiden", method="permutation", n_perms=20
        )
        assert "cell_communication" in adata.uns
        stored = adata.uns["cell_communication"]
        assert isinstance(stored, pd.DataFrame)

    def test_dataframe_input(self):
        adata = _make_adata()
        pairs_df = pd.DataFrame(
            {
                "ligand": ["GENE0", "GENE2", "GENE4"],
                "receptor": ["GENE1", "GENE3", "GENE5"],
            }
        )
        result = singlet.cell_communication(
            adata, pairs_df, groupby="leiden", method="permutation", n_perms=20
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_expected_row_count(self):
        """Should have rows for each LR pair × sender × receiver combination."""
        adata = _make_adata()
        pairs = [("GENE0", "GENE1"), ("GENE2", "GENE3")]
        result = singlet.cell_communication(
            adata, pairs, groupby="leiden", method="expression"
        )
        # 3 cell types → 9 sender-receiver pairs × 2 LR pairs = 18 rows
        assert len(result) == 18

    def test_scores_nonnegative(self):
        adata = _make_adata()
        pairs = _make_lr_pairs()
        result = singlet.cell_communication(
            adata, pairs, groupby="leiden", method="expression"
        )
        assert (result["score"] >= 0).all()

    def test_pvalues_bounded(self):
        adata = _make_adata()
        pairs = _make_lr_pairs()
        result = singlet.cell_communication(
            adata, pairs, groupby="leiden", method="permutation", n_perms=50
        )
        assert (result["pvalue"] >= 0).all()
        assert (result["pvalue"] <= 1).all()
        assert (result["fdr"] >= 0).all()
        assert (result["fdr"] <= 1).all()

    def test_missing_genes_skipped(self):
        adata = _make_adata()
        pairs = [
            ("GENE0", "GENE1"),  # valid
            ("NONEXIST1", "NONEXIST2"),  # invalid — should be skipped
        ]
        result = singlet.cell_communication(
            adata, pairs, groupby="leiden", method="expression"
        )
        # Only valid pair should appear
        assert all(result["ligand"] == "GENE0")
        assert all(result["receptor"] == "GENE1")

    def test_sparse_input(self):
        rng = np.random.default_rng(42)
        n_cells, n_genes = 100, 80
        X = rng.poisson(2, (n_cells, n_genes)).astype(np.float32)
        adata = AnnData(X=sparse.csr_matrix(X))
        adata.var_names = [f"GENE{i}" for i in range(n_genes)]
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        adata.obs["leiden"] = pd.Categorical(
            [f"ct_{i % 2}" for i in range(n_cells)]
        )
        pairs = [("GENE0", "GENE1")]
        result = singlet.cell_communication(
            adata, pairs, groupby="leiden", method="permutation", n_perms=20
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_custom_groupby(self):
        adata = _make_adata()
        adata.obs["cell_type"] = pd.Categorical(
            [f"type_{i % 4}" for i in range(adata.n_obs)]
        )
        pairs = [("GENE0", "GENE1")]
        result = singlet.cell_communication(
            adata, pairs, groupby="cell_type", method="expression"
        )
        # 4 cell types → 16 sender-receiver pairs × 1 LR pair
        assert len(result) == 16

    def test_fdr_less_equal_one(self):
        adata = _make_adata()
        pairs = _make_lr_pairs()
        result = singlet.cell_communication(
            adata, pairs, groupby="leiden", method="permutation", n_perms=100
        )
        assert (result["fdr"] <= 1.0 + 1e-10).all()

    def test_high_expression_pair_has_high_score(self):
        """L-R pair with high expression should score higher."""
        rng = np.random.default_rng(42)
        n_cells = 120
        X = np.ones((n_cells, 50), dtype=np.float32)
        # Boost GENE0 in cluster_0 (sender) and GENE1 in cluster_1 (receiver)
        adata = AnnData(X=X)
        adata.var_names = [f"GENE{i}" for i in range(50)]
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        adata.obs["leiden"] = pd.Categorical(
            [f"cluster_{i % 3}" for i in range(n_cells)]
        )
        # Boost ligand in cluster_0
        mask_0 = adata.obs["leiden"] == "cluster_0"
        adata.X[mask_0.values, 0] = 100.0
        # Boost receptor in cluster_1
        mask_1 = adata.obs["leiden"] == "cluster_1"
        adata.X[mask_1.values, 1] = 100.0

        pairs = [("GENE0", "GENE1")]
        result = singlet.cell_communication(
            adata, pairs, groupby="leiden", method="expression"
        )
        # Score for sender=cluster_0, receiver=cluster_1 should be highest
        top_row = result.loc[result["score"].idxmax()]
        assert top_row["sender"] == "cluster_0"
        assert top_row["receiver"] == "cluster_1"
