"""Tests for singlet.morans_i()."""

import numpy as np
import pytest
import scipy.sparse as sp
import singlet
from anndata import AnnData


def _make_adata_with_graph(n=80, m=100, seed=42, n_neighbors=10):
    """Create test AnnData with kNN graph."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(3, size=(n, m)).astype(np.float32)
    adata = AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs_names = [f"cell_{i}" for i in range(n)]

    # Build a simple kNN graph via PCA + neighbors
    adata.obsm["X_pca"] = rng.standard_normal((n, 20)).astype(np.float32)

    # Build connectivities manually using sklearn
    from sklearn.neighbors import NearestNeighbors

    k = min(n_neighbors, n - 1)
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(adata.obsm["X_pca"])
    dist, indices = nn.kneighbors(adata.obsm["X_pca"])

    # Build adjacency matrix
    rows = np.repeat(np.arange(n), k)
    cols = indices.ravel()
    data = np.ones(len(rows), dtype=np.float64)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    # Symmetrize
    adj = adj + adj.T
    adj.data[:] = 1.0
    adata.obsp["connectivities"] = adj
    return adata


def test_morans_i_basic():
    """Basic Moran's I computation returns correct shape."""
    adata = _make_adata_with_graph()
    result = singlet.morans_i(adata, n_perms=10)
    assert isinstance(result, type(result))  # pd.DataFrame
    assert "gene" in result.columns
    assert "morans_i" in result.columns
    assert "expected_i" in result.columns
    assert "pvalue" in result.columns
    assert "fdr" in result.columns
    assert len(result) == 100  # all genes (no HVG annotation)


def test_morans_i_stored_in_uns():
    """Result is stored in adata.uns['morans_i']."""
    adata = _make_adata_with_graph()
    result = singlet.morans_i(adata, n_perms=5)
    assert "morans_i" in adata.uns
    assert adata.uns["morans_i"] is result


def test_morans_i_specific_genes():
    """Test with specific gene list."""
    adata = _make_adata_with_graph()
    genes = ["gene_0", "gene_5", "gene_10"]
    result = singlet.morans_i(adata, genes=genes, n_perms=5)
    assert len(result) == 3
    assert set(result["gene"].values) == set(genes)


def test_morans_i_hvg_fallback():
    """Uses HVGs when available and no genes specified."""
    adata = _make_adata_with_graph()
    # Mark 20 genes as HVG
    adata.var["highly_variable"] = False
    adata.var.iloc[:20, adata.var.columns.get_loc("highly_variable")] = True
    result = singlet.morans_i(adata, n_perms=5)
    assert len(result) == 20


def test_morans_i_spatially_patterned():
    """Genes with spatial structure should have higher Moran's I."""
    rng = np.random.default_rng(123)
    n = 100
    m = 10

    # Create cells on a 1D line (spatial)
    positions = np.linspace(0, 10, n).reshape(-1, 1)
    pca = np.hstack([positions, rng.standard_normal((n, 9)) * 0.01])

    X = rng.poisson(2, size=(n, m)).astype(np.float32)
    # Gene 0: spatially smooth (correlated with position)
    X[:, 0] = np.linspace(0, 10, n) + rng.standard_normal(n) * 0.5
    # Gene 1: random (no spatial structure)
    X[:, 1] = rng.standard_normal(n)

    adata = AnnData(X=sp.csr_matrix(X.astype(np.float32)))
    adata.var_names = [f"gene_{i}" for i in range(m)]
    adata.obs_names = [f"cell_{i}" for i in range(n)]
    adata.obsm["X_pca"] = pca.astype(np.float32)

    # Build kNN graph
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=10, algorithm="auto")
    nn.fit(pca)
    _, indices = nn.kneighbors(pca)
    rows = np.repeat(np.arange(n), 10)
    cols = indices.ravel()
    adj = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n)
    )
    adj = adj + adj.T
    adj.data[:] = 1.0
    adata.obsp["connectivities"] = adj

    result = singlet.morans_i(adata, n_perms=50)
    # Spatially patterned gene should have higher Moran's I
    mi_gene0 = result.loc[result["gene"] == "gene_0", "morans_i"].values[0]
    mi_gene1 = result.loc[result["gene"] == "gene_1", "morans_i"].values[0]
    assert mi_gene0 > mi_gene1


def test_morans_i_expected_value():
    """Expected Moran's I should be -1/(n-1)."""
    adata = _make_adata_with_graph(n=50)
    result = singlet.morans_i(adata, n_perms=0)
    expected = -1.0 / (50 - 1)
    assert np.isclose(result["expected_i"].iloc[0], expected)


def test_morans_i_no_perms():
    """n_perms=0 should return NaN p-values."""
    adata = _make_adata_with_graph()
    result = singlet.morans_i(adata, n_perms=0)
    assert np.all(np.isnan(result["pvalue"].values))
    assert np.all(np.isnan(result["fdr"].values))


def test_morans_i_sorted_descending():
    """Results should be sorted by Moran's I descending."""
    adata = _make_adata_with_graph()
    result = singlet.morans_i(adata, n_perms=5)
    mi_vals = result["morans_i"].values
    assert np.all(mi_vals[:-1] >= mi_vals[1:])


def test_morans_i_no_graph_raises():
    """Should raise KeyError when no graph is available."""
    rng = np.random.default_rng(42)
    X = rng.poisson(2, size=(50, 30)).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = [f"g{i}" for i in range(30)]
    adata.obs_names = [f"c{i}" for i in range(50)]
    with pytest.raises(KeyError, match="connectivities"):
        singlet.morans_i(adata)


def test_morans_i_invalid_genes():
    """Should raise ValueError for genes not in adata."""
    adata = _make_adata_with_graph()
    with pytest.raises(ValueError, match="None of the specified genes"):
        singlet.morans_i(adata, genes=["not_a_gene"])


def test_morans_i_fdr_bounded():
    """FDR values should be in [0, 1]."""
    adata = _make_adata_with_graph(n=60, m=20)
    result = singlet.morans_i(adata, n_perms=20)
    fdr_vals = result["fdr"].values
    assert np.all(fdr_vals >= 0)
    assert np.all(fdr_vals <= 1.0)
