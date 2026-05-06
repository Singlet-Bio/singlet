"""Tests for singlet.paga() and singlet.plot_paga()."""

import numpy as np
import pytest
import singlet
from anndata import AnnData


def _make_adata(n=120, m=200, seed=42):
    rng = np.random.default_rng(seed)
    adata = AnnData(X=rng.standard_normal((n, m)).astype(np.float32))
    singlet.pca(adata)
    singlet.neighbors(adata)
    singlet.leiden(adata)
    return adata


def test_paga_basic():
    adata = _make_adata()
    singlet.paga(adata)
    assert "paga" in adata.uns
    assert "connectivities" in adata.uns["paga"]
    assert "connectivities_tree" in adata.uns["paga"]


def test_paga_connectivity_shape():
    adata = _make_adata()
    singlet.paga(adata)
    n_clusters = len(adata.obs["leiden"].cat.categories)
    conn = adata.uns["paga"]["connectivities"]
    assert conn.shape == (n_clusters, n_clusters)


def test_paga_connectivity_symmetric():
    adata = _make_adata()
    singlet.paga(adata)
    from scipy.sparse import issparse

    conn = adata.uns["paga"]["connectivities"]
    if issparse(conn):
        conn = conn.toarray()
    assert np.allclose(conn, conn.T, atol=1e-10)


def test_paga_connectivity_nonneg():
    adata = _make_adata()
    singlet.paga(adata)
    from scipy.sparse import issparse

    conn = adata.uns["paga"]["connectivities"]
    if issparse(conn):
        assert (conn.data >= 0).all()
    else:
        assert (conn >= 0).all()


def test_paga_custom_groups():
    adata = _make_adata()
    adata.obs["custom"] = [f"c{i % 4}" for i in range(adata.n_obs)]
    singlet.paga(adata, groups="custom")
    conn = adata.uns["paga"]["connectivities"]
    assert conn.shape == (4, 4)


def test_paga_copy():
    adata = _make_adata()
    result = singlet.paga(adata, copy=True)
    assert result is not None
    assert "paga" not in adata.uns
    assert "paga" in result.uns


def test_paga_no_neighbors_raises():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 100)).astype(np.float32))
    adata.obs["group"] = ["a"] * 25 + ["b"] * 25
    with pytest.raises(KeyError):
        singlet.paga(adata, groups="group")


def test_paga_no_groups_raises():
    adata = _make_adata()
    with pytest.raises(KeyError):
        singlet.paga(adata, groups="nonexistent")


def test_paga_threshold():
    adata = _make_adata()
    singlet.paga(adata, threshold=0.5)
    from scipy.sparse import issparse

    conn = adata.uns["paga"]["connectivities"]
    if issparse(conn):
        # All non-zero values should be >= 0.5
        assert (conn.data >= 0.5).all() or conn.nnz == 0
    else:
        assert ((conn == 0) | (conn >= 0.5)).all()


# plot_paga tests


def test_plot_paga_basic():
    adata = _make_adata()
    singlet.paga(adata)
    result = singlet.plot_paga(adata)
    assert result is None  # show=True


def test_plot_paga_show_false():
    adata = _make_adata()
    singlet.paga(adata)
    ax = singlet.plot_paga(adata, show=False)
    assert ax is not None


def test_plot_paga_circle_layout():
    adata = _make_adata()
    singlet.paga(adata)
    ax = singlet.plot_paga(adata, layout="circle", show=False)
    assert ax is not None


def test_plot_paga_save(tmp_path):
    adata = _make_adata()
    singlet.paga(adata)
    path = str(tmp_path / "paga.png")
    singlet.plot_paga(adata, save=path)
    import os

    assert os.path.exists(path)


def test_plot_paga_no_paga_raises():
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.standard_normal((50, 100)).astype(np.float32))
    with pytest.raises(KeyError):
        singlet.plot_paga(adata)
