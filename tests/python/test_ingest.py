# SPDX-License-Identifier: MIT
"""Tests for singlet.ingest()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_ref_query(n_ref=100, n_query=50, n_genes=200, seed=42):
    rng = np.random.default_rng(seed)
    genes = [f"gene_{i}" for i in range(n_genes)]

    # Reference
    X_ref = rng.standard_normal((n_ref, n_genes)).astype(np.float32)
    ref = AnnData(X=X_ref)
    ref.var_names = genes
    ref.obs["celltype"] = [f"type_{i % 3}" for i in range(n_ref)]
    singlet.pca(ref)
    singlet.neighbors(ref)
    singlet.umap(ref)

    # Query (same genes)
    X_query = rng.standard_normal((n_query, n_genes)).astype(np.float32)
    query = AnnData(X=X_query)
    query.var_names = genes

    return ref, query


def test_ingest_basic():
    ref, query = _make_ref_query()
    singlet.ingest(query, ref, obs_to_transfer="celltype")
    assert "celltype" in query.obs.columns
    assert len(query.obs["celltype"]) == query.n_obs


def test_ingest_embedding_transferred():
    ref, query = _make_ref_query()
    singlet.ingest(query, ref, obs_to_transfer="celltype")
    assert "X_umap" in query.obsm
    assert query.obsm["X_umap"].shape == (query.n_obs, 2)


def test_ingest_labels_are_valid():
    ref, query = _make_ref_query()
    singlet.ingest(query, ref, obs_to_transfer="celltype")
    valid_labels = set(ref.obs["celltype"].unique())
    for label in query.obs["celltype"]:
        assert label in valid_labels


def test_ingest_copy():
    ref, query = _make_ref_query()
    result = singlet.ingest(query, ref, obs_to_transfer="celltype", copy=True)
    assert result is not None
    assert "celltype" not in query.obs.columns
    assert "celltype" in result.obs.columns


def test_ingest_multiple_obs():
    ref, query = _make_ref_query()
    ref.obs["batch"] = [f"b{i % 2}" for i in range(ref.n_obs)]
    singlet.ingest(query, ref, obs_to_transfer=["celltype", "batch"])
    assert "celltype" in query.obs.columns
    assert "batch" in query.obs.columns


def test_ingest_n_pcs():
    ref, query = _make_ref_query()
    singlet.ingest(query, ref, obs_to_transfer="celltype", n_pcs=10)
    assert "celltype" in query.obs.columns


def test_ingest_no_pca_raises():
    rng = np.random.default_rng(42)
    ref = AnnData(X=rng.standard_normal((50, 100)).astype(np.float32))
    query = AnnData(X=rng.standard_normal((20, 100)).astype(np.float32))
    ref.var_names = [f"g{i}" for i in range(100)]
    query.var_names = [f"g{i}" for i in range(100)]
    with pytest.raises(KeyError):
        singlet.ingest(query, ref, obs_to_transfer="celltype")


def test_ingest_n_neighbors():
    ref, query = _make_ref_query()
    singlet.ingest(query, ref, obs_to_transfer="celltype", n_neighbors=5)
    assert "celltype" in query.obs.columns


def test_ingest_embedding_finite():
    ref, query = _make_ref_query()
    singlet.ingest(query, ref, obs_to_transfer="celltype")
    assert np.all(np.isfinite(query.obsm["X_umap"]))
