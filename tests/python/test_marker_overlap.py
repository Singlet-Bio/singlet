# SPDX-License-Identifier: MIT
"""Tests for singlet.marker_gene_overlap()."""

import numpy as np
import pandas as pd
import pytest
import singlet
from anndata import AnnData


def _make_de_adata(n=90, m=100, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m)).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs["group"] = [f"g{i % 3}" for i in range(n)]
    # Add differential signal
    for i in range(n):
        g = i % 3
        X[i, g * 10 : (g + 1) * 10] += 5.0
    adata.X = X
    singlet.rank_genes_groups(adata, groupby="group")
    return adata


def test_overlap_basic():
    adata = _make_de_adata()
    ref = {"type_A": ["gene_0", "gene_1", "gene_2"], "type_B": ["gene_10", "gene_11"]}
    result = singlet.marker_gene_overlap(adata, ref)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 3  # 3 groups
    assert result.shape[1] == 2  # 2 reference types


def test_overlap_count_values():
    adata = _make_de_adata()
    ref = {"type_A": [f"gene_{i}" for i in range(10)]}
    result = singlet.marker_gene_overlap(adata, ref, method="overlap_count")
    # g0 should overlap with genes 0-9
    assert result.loc["g0", "type_A"] > 0


def test_overlap_jaccard():
    adata = _make_de_adata()
    ref = {"type_A": [f"gene_{i}" for i in range(10)]}
    result = singlet.marker_gene_overlap(adata, ref, method="jaccard")
    assert (result.values >= 0).all()
    assert (result.values <= 1).all()


def test_overlap_coef():
    adata = _make_de_adata()
    ref = {"type_A": [f"gene_{i}" for i in range(10)]}
    result = singlet.marker_gene_overlap(adata, ref, method="overlap_coef")
    assert (result.values >= 0).all()
    assert (result.values <= 1).all()


def test_overlap_n_genes():
    adata = _make_de_adata()
    ref = {"type_A": [f"gene_{i}" for i in range(10)]}
    r1 = singlet.marker_gene_overlap(adata, ref, n_genes=5)
    r2 = singlet.marker_gene_overlap(adata, ref, n_genes=50)
    # More genes should give >= overlap
    assert r2.values.sum() >= r1.values.sum()


def test_overlap_no_de_raises():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((30, 20)).astype(np.float32))
    with pytest.raises(KeyError):
        singlet.marker_gene_overlap(adata, {"a": ["gene_0"]})


def test_overlap_normalize_reference():
    adata = _make_de_adata()
    ref = {"type_A": [f"gene_{i}" for i in range(10)]}
    result = singlet.marker_gene_overlap(adata, ref, normalize="reference")
    assert (result.values >= 0).all()


def test_overlap_invalid_method_raises():
    adata = _make_de_adata()
    ref = {"type_A": ["gene_0"]}
    with pytest.raises(ValueError):
        singlet.marker_gene_overlap(adata, ref, method="invalid")
