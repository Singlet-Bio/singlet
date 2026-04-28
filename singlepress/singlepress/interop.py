"""
singlepress.interop — Ecosystem interoperability for .1pz files.

Conversions between .1pz and:
  - AnnData (scanpy ecosystem)
  - MuData (muon/multimodal)
  - pandas DataFrame (dense)
  - scipy sparse formats (COO, CSR, CSC, LIL)
  - HDF5 / .h5ad (AnnData on-disk format)
  - 10x Genomics .h5 (CellRanger format)

Usage:
    import singlepress.interop as spi

    # AnnData
    adata = spi.to_anndata("counts.1pz")
    spi.from_anndata(adata, "counts.1pz")

    # HDF5 / h5ad
    spi.to_h5ad("counts.1pz", "counts.h5ad")
    spi.from_h5ad("counts.h5ad", "counts.1pz")

    # 10x CellRanger .h5
    spi.from_10x_h5("filtered_feature_bc_matrix.h5", "counts.1pz")

    # Standard formats
    csr = spi.to_csr("counts.1pz")
    coo = spi.to_coo("counts.1pz")
    df = spi.to_dataframe("counts.1pz")
"""

from __future__ import annotations

import os
import numpy as np
import scipy.sparse as ss
from typing import Optional, Union

import singlepress


# ============================================================================
# AnnData interoperability
# ============================================================================

def to_anndata(
    path: str,
    *,
    num_threads: int = 8,
    backed: bool = False,
):
    """Read a .1pz file into an AnnData object.

    Parameters
    ----------
    path : str
        Path to .1pz file.
    num_threads : int
        Decompression threads.
    backed : bool
        If True, return a lazy-backed AnnData that reads on demand.
        Not yet implemented — raises NotImplementedError.

    Returns
    -------
    anndata.AnnData
        AnnData object with:
        - .X: sparse CSC matrix
        - .obs: cell metadata DataFrame (if present)
        - .var: gene metadata DataFrame (if present)
        - .uns: study metadata dict (if present)
        - .obs_names: cell barcodes (if present)
        - .var_names: gene names (if present)
    """
    import anndata as ad
    import pandas as pd

    if backed:
        raise NotImplementedError(
            "Backed mode not yet supported. Use to_anndata() without backed=True."
        )

    mat = singlepress.read_1pz(path, num_threads=num_threads)

    obs = getattr(mat, "obs", None)
    var = getattr(mat, "var", None)
    uns = getattr(mat, "uns", None) or {}
    rownames = getattr(mat, "rownames", None)
    colnames = getattr(mat, "colnames", None)

    # AnnData expects obs x var (cells x genes), .1pz stores genes x cells
    # Transpose: .1pz CSC (genes×cells) -> AnnData CSR (cells×genes)
    X = mat.T.tocsr()

    # Build obs DataFrame
    if obs is not None:
        obs_df = obs.copy()
    else:
        obs_df = pd.DataFrame(index=range(X.shape[0]))

    if colnames is not None:
        obs_df.index = pd.Index(colnames)

    # Build var DataFrame
    if var is not None:
        var_df = var.copy()
    else:
        var_df = pd.DataFrame(index=range(X.shape[1]))

    if rownames is not None:
        var_df.index = pd.Index(rownames)

    # Colsums as obs column
    colsums = getattr(mat, "colsums", None)
    if colsums is not None:
        obs_df["total_counts"] = colsums.astype(np.float64)

    adata = ad.AnnData(
        X=X,
        obs=obs_df,
        var=var_df,
        uns=dict(uns),
    )

    return adata


def from_anndata(
    adata,
    path: str,
    *,
    store_transpose: bool = False,
    num_threads: int = 8,
    level: int = 3,
    chunk_cols: int = 1024,
) -> dict:
    """Write an AnnData object to a .1pz file.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object. X should be a sparse or dense matrix (cells × genes).
    path : str
        Output .1pz file path.
    store_transpose : bool
        Store transpose for fast gene-range reads.
    num_threads : int
        Compression threads.
    level : int
        Zstd compression level.
    chunk_cols : int
        Columns per chunk.

    Returns
    -------
    dict
        Compression statistics.
    """
    import pandas as pd

    # AnnData is cells × genes; .1pz stores genes × cells (CSC)
    X = adata.X
    if ss.issparse(X):
        mat = X.T.tocsc()
    else:
        mat = ss.csc_matrix(np.asarray(X).T)

    rownames = list(adata.var_names) if adata.var_names is not None else None
    colnames = list(adata.obs_names) if adata.obs_names is not None else None

    # Extract obs/var DataFrames (exclude index, which becomes colnames/rownames)
    obs = adata.obs if len(adata.obs.columns) > 0 else None
    var = adata.var if len(adata.var.columns) > 0 else None

    # Extract string-keyed uns entries
    uns = {}
    if adata.uns:
        for k, v in adata.uns.items():
            if isinstance(v, str):
                uns[k] = v
            elif isinstance(v, (int, float)):
                uns[k] = str(v)
            # Skip complex objects (arrays, dicts, etc.)

    return singlepress.write_1pz(
        path, mat,
        rownames=rownames,
        colnames=colnames,
        obs=obs,
        var=var,
        uns=uns if uns else None,
        store_transpose=store_transpose,
        num_threads=num_threads,
        level=level,
        chunk_cols=chunk_cols,
    )


# ============================================================================
# HDF5 / .h5ad interoperability
# ============================================================================

def to_h5ad(
    pz_path: str,
    h5ad_path: str,
    *,
    num_threads: int = 8,
    compression: str = "gzip",
):
    """Convert a .1pz file to .h5ad (AnnData on-disk format).

    Parameters
    ----------
    pz_path : str
        Input .1pz file.
    h5ad_path : str
        Output .h5ad file.
    num_threads : int
        Decompression threads for reading .1pz.
    compression : str
        HDF5 compression for .h5ad. Default "gzip".
    """
    adata = to_anndata(pz_path, num_threads=num_threads)
    adata.write_h5ad(h5ad_path, compression=compression)


def from_h5ad(
    h5ad_path: str,
    pz_path: str,
    *,
    store_transpose: bool = False,
    num_threads: int = 8,
    level: int = 3,
    chunk_cols: int = 1024,
) -> dict:
    """Convert an .h5ad file to .1pz format.

    Parameters
    ----------
    h5ad_path : str
        Input .h5ad file.
    pz_path : str
        Output .1pz file.

    Returns
    -------
    dict
        Compression statistics.
    """
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    return from_anndata(
        adata, pz_path,
        store_transpose=store_transpose,
        num_threads=num_threads,
        level=level,
        chunk_cols=chunk_cols,
    )


def from_10x_h5(
    h5_path: str,
    pz_path: str,
    *,
    store_transpose: bool = False,
    num_threads: int = 8,
    level: int = 3,
    chunk_cols: int = 1024,
) -> dict:
    """Convert a 10x Genomics CellRanger .h5 file to .1pz format.

    Reads the standard 10x HDF5 format (filtered_feature_bc_matrix.h5)
    and writes a .1pz file with gene names and cell barcodes.

    Parameters
    ----------
    h5_path : str
        Input 10x .h5 file (CellRanger output).
    pz_path : str
        Output .1pz file.

    Returns
    -------
    dict
        Compression statistics.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        # 10x format: matrix/{barcodes, data, indices, indptr, shape, features/{name, id}}
        if "matrix" in f:
            grp = f["matrix"]
            shape = tuple(grp["shape"][:])
            indptr = np.asarray(grp["indptr"][:], dtype=np.int32)
            indices = np.asarray(grp["indices"][:], dtype=np.int32)
            data = np.asarray(grp["data"][:], dtype=np.float64)
            mat = ss.csc_matrix((data, indices, indptr), shape=shape)

            barcodes = [b.decode() for b in grp["barcodes"][:]]
            if "features" in grp:
                gene_names = [n.decode() for n in grp["features"]["name"][:]]
            elif "genes" in grp:
                gene_names = [n.decode() for n in grp["genes"][:]]
            else:
                gene_names = None
        else:
            raise ValueError(f"Unrecognized 10x HDF5 format in {h5_path}")

    return singlepress.write_1pz(
        pz_path, mat,
        rownames=gene_names,
        colnames=barcodes,
        store_transpose=store_transpose,
        num_threads=num_threads,
        level=level,
        chunk_cols=chunk_cols,
    )


# ============================================================================
# Standard sparse format conversions
# ============================================================================

def to_csr(path: str, *, num_threads: int = 8) -> ss.csr_matrix:
    """Read .1pz to scipy CSR matrix."""
    return singlepress.read_1pz(path, num_threads=num_threads).tocsr()


def to_coo(path: str, *, num_threads: int = 8) -> ss.coo_matrix:
    """Read .1pz to scipy COO matrix."""
    return singlepress.read_1pz(path, num_threads=num_threads).tocoo()


def to_csc(path: str, *, num_threads: int = 8) -> ss.csc_matrix:
    """Read .1pz to scipy CSC matrix (native format)."""
    return singlepress.read_1pz(path, num_threads=num_threads)


def to_dense(path: str, *, num_threads: int = 8) -> np.ndarray:
    """Read .1pz to dense numpy array. Warning: may use large memory."""
    return singlepress.read_1pz(path, num_threads=num_threads).toarray()


def to_dataframe(
    path: str,
    *,
    num_threads: int = 8,
    use_names: bool = True,
) -> "pd.DataFrame":
    """Read .1pz to pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to .1pz file.
    use_names : bool
        If True, use stored rownames/colnames as index/columns.

    Returns
    -------
    pandas.DataFrame
        Dense DataFrame. Warning: may use large memory for big matrices.
    """
    import pandas as pd

    mat = singlepress.read_1pz(path, num_threads=num_threads)
    dense = mat.toarray()

    rownames = getattr(mat, "rownames", None) if use_names else None
    colnames = getattr(mat, "colnames", None) if use_names else None

    return pd.DataFrame(
        dense,
        index=rownames,
        columns=colnames,
    )


def from_csr(
    mat: ss.csr_matrix,
    path: str,
    *,
    rownames: Optional[list[str]] = None,
    colnames: Optional[list[str]] = None,
    **kwargs,
) -> dict:
    """Write a CSR matrix to .1pz."""
    return singlepress.write_1pz(path, mat.tocsc(), rownames=rownames, colnames=colnames, **kwargs)


def from_coo(
    mat: ss.coo_matrix,
    path: str,
    *,
    rownames: Optional[list[str]] = None,
    colnames: Optional[list[str]] = None,
    **kwargs,
) -> dict:
    """Write a COO matrix to .1pz."""
    return singlepress.write_1pz(path, mat.tocsc(), rownames=rownames, colnames=colnames, **kwargs)


def from_dense(
    array: np.ndarray,
    path: str,
    *,
    rownames: Optional[list[str]] = None,
    colnames: Optional[list[str]] = None,
    **kwargs,
) -> dict:
    """Write a dense numpy array to .1pz."""
    mat = ss.csc_matrix(array)
    return singlepress.write_1pz(path, mat, rownames=rownames, colnames=colnames, **kwargs)


def from_dataframe(
    df: "pd.DataFrame",
    path: str,
    **kwargs,
) -> dict:
    """Write a pandas DataFrame to .1pz.

    Uses DataFrame index as rownames and columns as colnames.
    """
    rownames = [str(x) for x in df.index]
    colnames = [str(x) for x in df.columns]
    mat = ss.csc_matrix(df.values)
    return singlepress.write_1pz(path, mat, rownames=rownames, colnames=colnames, **kwargs)


# ============================================================================
# MatrixMarket (.mtx) interoperability
# ============================================================================

def from_mtx(
    mtx_dir: str,
    pz_path: str,
    *,
    store_transpose: bool = False,
    num_threads: int = 8,
    level: int = 3,
    chunk_cols: int = 1024,
) -> dict:
    """Read a 10x-style MatrixMarket directory into a .1pz file.

    Expects the standard 10x output layout::

        mtx_dir/
            matrix.mtx.gz    (or matrix.mtx)
            features.tsv.gz  (or genes.tsv.gz / genes.tsv)
            barcodes.tsv.gz  (or barcodes.tsv)

    Parameters
    ----------
    mtx_dir : str
        Path to the directory containing matrix.mtx[.gz], features/genes,
        and barcodes files.
    pz_path : str
        Output .1pz file path.
    store_transpose, num_threads, level, chunk_cols
        Passed to write_1pz.

    Returns
    -------
    dict with compression stats from write_1pz.
    """
    from scipy.io import mmread

    # Find the matrix file
    mtx_file = None
    for name in ("matrix.mtx.gz", "matrix.mtx"):
        candidate = os.path.join(mtx_dir, name)
        if os.path.exists(candidate):
            mtx_file = candidate
            break
    if mtx_file is None:
        raise FileNotFoundError(f"No matrix.mtx[.gz] found in {mtx_dir}")

    mat = mmread(mtx_file)
    if not ss.issparse(mat):
        mat = ss.csc_matrix(mat)
    elif not ss.isspmatrix_csc(mat):
        mat = mat.tocsc()

    # Read gene names
    gene_names = None
    for name in ("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"):
        candidate = os.path.join(mtx_dir, name)
        if os.path.exists(candidate):
            import pandas as pd
            df = pd.read_csv(candidate, sep="\t", header=None)
            gene_names = df.iloc[:, 1].astype(str).tolist() if df.shape[1] > 1 else df.iloc[:, 0].astype(str).tolist()
            break

    # Read barcodes
    barcodes = None
    for name in ("barcodes.tsv.gz", "barcodes.tsv"):
        candidate = os.path.join(mtx_dir, name)
        if os.path.exists(candidate):
            import pandas as pd
            df = pd.read_csv(candidate, sep="\t", header=None)
            barcodes = df.iloc[:, 0].astype(str).tolist()
            break

    return singlepress.write_1pz(
        pz_path, mat,
        rownames=gene_names,
        colnames=barcodes,
        store_transpose=store_transpose,
        num_threads=num_threads,
        level=level,
        chunk_cols=chunk_cols,
    )


def to_mtx(
    pz_path: str,
    output_dir: str,
    *,
    num_threads: int = 8,
) -> str:
    """Write a .1pz file to 10x-style MatrixMarket directory.

    Creates::

        output_dir/
            matrix.mtx.gz
            features.tsv.gz
            barcodes.tsv.gz

    Parameters
    ----------
    pz_path : str
        Input .1pz file path.
    output_dir : str
        Output directory (created if needed).
    num_threads : int
        OpenMP threads for reading.

    Returns
    -------
    str : output_dir path.
    """
    import gzip
    from scipy.io import mmwrite

    os.makedirs(output_dir, exist_ok=True)
    pz = singlepress.open_1pz(pz_path, num_threads=num_threads)
    mat = pz.read()

    # Write matrix.mtx.gz
    mtx_path = os.path.join(output_dir, "matrix.mtx")
    mmwrite(mtx_path, mat)
    # Compress
    with open(mtx_path, "rb") as f_in:
        with gzip.open(mtx_path + ".gz", "wb") as f_out:
            f_out.write(f_in.read())
    os.remove(mtx_path)

    # Write features.tsv.gz
    rownames = pz.rownames
    if rownames:
        features_path = os.path.join(output_dir, "features.tsv.gz")
        with gzip.open(features_path, "wt") as f:
            for name in rownames:
                f.write(f"{name}\t{name}\tGene Expression\n")

    # Write barcodes.tsv.gz
    colnames = pz.colnames
    if colnames:
        barcodes_path = os.path.join(output_dir, "barcodes.tsv.gz")
        with gzip.open(barcodes_path, "wt") as f:
            for name in colnames:
                f.write(f"{name}\n")

    return output_dir


# ============================================================================
# CSV / TSV interoperability
# ============================================================================

def from_csv(
    csv_path: str,
    pz_path: str,
    *,
    sep: str = ",",
    has_header: bool = True,
    has_index: bool = True,
    store_transpose: bool = False,
    num_threads: int = 8,
    level: int = 3,
    chunk_cols: int = 1024,
) -> dict:
    """Read a dense CSV/TSV file into a .1pz file.

    Expects genes as rows and cells as columns (features × cells).

    Parameters
    ----------
    csv_path : str
        Input CSV/TSV file path.
    pz_path : str
        Output .1pz file path.
    sep : str
        Column separator. Default ',' for CSV; use '\\t' for TSV.
    has_header : bool
        If True, first row contains column names (cell barcodes).
    has_index : bool
        If True, first column contains row names (gene names).
    store_transpose, num_threads, level, chunk_cols
        Passed to write_1pz.

    Returns
    -------
    dict with compression stats from write_1pz.
    """
    import pandas as pd

    df = pd.read_csv(
        csv_path,
        sep=sep,
        header=0 if has_header else None,
        index_col=0 if has_index else None,
    )

    rownames = [str(x) for x in df.index] if has_index else None
    colnames = [str(x) for x in df.columns] if has_header else None

    mat = ss.csc_matrix(df.values)

    return singlepress.write_1pz(
        pz_path, mat,
        rownames=rownames,
        colnames=colnames,
        store_transpose=store_transpose,
        num_threads=num_threads,
        level=level,
        chunk_cols=chunk_cols,
    )


def to_csv(
    pz_path: str,
    csv_path: str,
    *,
    sep: str = ",",
    num_threads: int = 8,
) -> str:
    """Write a .1pz file to CSV/TSV.

    Output has genes as rows and cells as columns (features × cells).
    Warning: dense output may be very large.

    Parameters
    ----------
    pz_path : str
        Input .1pz file path.
    csv_path : str
        Output CSV/TSV file path.
    sep : str
        Column separator. Default ',' for CSV; use '\\t' for TSV.
    num_threads : int
        OpenMP threads for reading.

    Returns
    -------
    str : csv_path.
    """
    import pandas as pd

    pz = singlepress.open_1pz(pz_path, num_threads=num_threads)
    mat = pz.read().toarray()

    df = pd.DataFrame(
        mat,
        index=pz.rownames,
        columns=pz.colnames,
    )
    df.to_csv(csv_path, sep=sep)
    return csv_path


# ============================================================================
# Loom (.loom) interoperability
# ============================================================================

def from_loom(
    loom_path: str,
    pz_path: str,
    *,
    layer: Optional[str] = None,
    store_transpose: bool = False,
    num_threads: int = 8,
    level: int = 3,
    chunk_cols: int = 1024,
) -> dict:
    """Read a .loom file (HDF5 variant) into a .1pz file.

    Loom files store matrices as genes × cells (same as .1pz).

    Parameters
    ----------
    loom_path : str
        Input .loom file path.
    pz_path : str
        Output .1pz file path.
    layer : str, optional
        Layer name in the loom file. Default reads the main matrix.
    store_transpose, num_threads, level, chunk_cols
        Passed to write_1pz.

    Returns
    -------
    dict with compression stats from write_1pz.
    """
    import h5py

    with h5py.File(loom_path, "r") as f:
        if layer is not None:
            dense = f["layers"][layer][:]
        else:
            dense = f["matrix"][:]

        # Loom stores genes × cells
        mat = ss.csc_matrix(dense)

        gene_names = None
        if "ra" in f and "Gene" in f["ra"]:
            gene_names = [x.decode() if isinstance(x, bytes) else str(x) for x in f["ra"]["Gene"][:]]
        elif "row_attrs" in f and "Gene" in f["row_attrs"]:
            gene_names = [x.decode() if isinstance(x, bytes) else str(x) for x in f["row_attrs"]["Gene"][:]]

        barcodes = None
        if "ca" in f and "CellID" in f["ca"]:
            barcodes = [x.decode() if isinstance(x, bytes) else str(x) for x in f["ca"]["CellID"][:]]
        elif "col_attrs" in f and "CellID" in f["col_attrs"]:
            barcodes = [x.decode() if isinstance(x, bytes) else str(x) for x in f["col_attrs"]["CellID"][:]]

    return singlepress.write_1pz(
        pz_path, mat,
        rownames=gene_names,
        colnames=barcodes,
        store_transpose=store_transpose,
        num_threads=num_threads,
        level=level,
        chunk_cols=chunk_cols,
    )
