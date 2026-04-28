"""Shared test fixtures."""

import pytest
import numpy as np
import scipy.sparse as sp


@pytest.fixture
def small_csc():
    """A small CSC sparse matrix for testing (5 genes × 10 cells)."""
    rng = np.random.default_rng(42)
    dense = rng.poisson(lam=0.5, size=(5, 10)).astype(np.float64)
    return sp.csc_matrix(dense)


@pytest.fixture
def small_anndata(small_csc):
    """A small AnnData object (10 cells × 5 genes)."""
    import anndata as ad
    import pandas as pd

    # AnnData is cells × genes, small_csc is genes × cells
    adata = ad.AnnData(X=small_csc.T.tocsr())
    adata.var_names = pd.Index([f"Gene{i}" for i in range(5)])
    adata.obs_names = pd.Index([f"Cell{i}" for i in range(10)])
    return adata


@pytest.fixture
def tmp_spz(tmp_path, small_csc):
    """Write a temporary .spz file and return its path."""
    path = tmp_path / "test.spz"

    try:
        from singlet._singlepress import sp_write_int

        mat = small_csc
        sp_write_int(
            mat.indptr.astype(np.int32),
            mat.indices.astype(np.int32),
            mat.data.astype(np.int32),
            mat.shape[0],
            str(path),
            rownames=[f"Gene{i}" for i in range(mat.shape[0])],
            colnames=[f"Cell{i}" for i in range(mat.shape[1])],
        )
        return path
    except ImportError:
        pytest.skip("_singlepress extension not built")
