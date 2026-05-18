# SPDX-License-Identifier: MIT
"""Tests for singlet.filter_rank_genes_groups()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_de_adata(n=90, m=60, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m)).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs["group"] = [f"g{i % 3}" for i in range(n)]
    for i in range(n):
        g = i % 3
        X[i, g * 10 : (g + 1) * 10] += 5.0
    adata.X = X
    singlet.rank_genes_groups(adata, groupby="group")
    return adata


def test_filter_basic():
    adata = _make_de_adata()
    singlet.filter_rank_genes_groups(adata)
    assert "rank_genes_groups_filtered" in adata.uns


def test_filter_key_added():
    adata = _make_de_adata()
    singlet.filter_rank_genes_groups(adata, key_added="my_filtered")
    assert "my_filtered" in adata.uns


def test_filter_max_pval():
    adata = _make_de_adata()
    singlet.filter_rank_genes_groups(adata, max_pval=0.01)
    filtered = adata.uns["rank_genes_groups_filtered"]
    # Some genes should be filtered out
    assert "names" in filtered


def test_filter_min_fold_change():
    adata = _make_de_adata()
    singlet.filter_rank_genes_groups(adata, min_fold_change=2.0)
    filtered = adata.uns["rank_genes_groups_filtered"]
    assert "names" in filtered


def test_filter_removes_genes():
    adata = _make_de_adata()
    # Very strict filter should remove some genes
    singlet.filter_rank_genes_groups(adata, min_fold_change=10.0)
    filtered = adata.uns["rank_genes_groups_filtered"]
    groups = list(filtered["names"].keys())
    # Check that some genes are empty string (filtered out)
    has_empty = any(
        filtered["names"][g][i] == "" for g in groups for i in range(len(filtered["names"][g]))
    )
    assert has_empty


def test_filter_no_de_raises():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((30, 20)).astype(np.float32))
    with pytest.raises(KeyError):
        singlet.filter_rank_genes_groups(adata)


def test_filter_preserves_structure():
    adata = _make_de_adata()
    singlet.filter_rank_genes_groups(adata)
    filtered = adata.uns["rank_genes_groups_filtered"]
    original = adata.uns["rank_genes_groups"]
    # Same groups
    assert list(filtered["names"].keys()) == list(original["names"].keys())
