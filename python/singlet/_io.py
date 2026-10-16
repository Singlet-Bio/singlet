# SPDX-License-Identifier: MIT
""".1pz file I/O — single-matrix AnnData round-trip.

This module is the thin AnnData adapter on top of the in-tree TP1Z codec at
`include/singlet/pileup/pz_reader.h` and `include/singlet/pileup/pz_writer.h`
(binding: `singlet._pz`). It handles single-block ``.1pz`` files (magic ``TP1Z``,
version 1), the only layout this package reads or writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import anndata


_TP1_MAGIC = b"\x54\x50\x31\x5a"  # "TP1Z" — legacy single-block .1pz


def _detect_format(path: str | Path) -> str:
    """Detect file format from header magic bytes.

    Returns ``"1pz"``. Raises :class:`ValueError` for anything else.
    """
    with open(path, "rb") as f:
        header = f.read(8)
    if len(header) < 8:
        raise ValueError(f"File too small to be a valid matrix file: {path}")
    if header[:4] == _TP1_MAGIC:
        return "1pz"
    raise ValueError(f"Unknown file format (bad magic): {path}")


# ============================================================================
# Internal helper: read .1pz via native codec, return CSC + metadata
# ============================================================================


def _read_pz_native(path: str):
    """Read a .1pz file using the in-tree codec.

    Returns ``(csc_matrix, rownames, colnames, user_kv)``.
    The matrix has shape ``(m, n)`` = ``(genes, cells)`` — caller transposes
    to cells × genes for AnnData.
    """
    import scipy.sparse as sp

    from singlet._pz import read_1pz as _native_read

    r = _native_read(path)
    mat = sp.csc_matrix(
        (r["data"], r["indices"], r["indptr"]),
        shape=(r["m"], r["n"]),
    )
    return mat, r["rownames"], r["colnames"], r["user_kv"]


# ============================================================================
# .1pz format (legacy single-block via in-tree codec)
# ============================================================================


def read_1pz(path: str | Path) -> "anndata.AnnData":
    """Read a single-block ``.1pz`` file into AnnData.

    Returns
    -------
    anndata.AnnData
        Sparse count matrix (CSC) with cells in ``obs`` and genes in
        ``var``. (``.1pz`` stores genes × cells; this transposes.)
        ``rownames`` → ``var_names`` (genes), ``colnames`` → ``obs_names``
        (cells), ``user_kv`` → ``uns``. ``total_counts`` is computed from
        the matrix.
    """
    import anndata as ad
    import numpy as np

    if path is None:
        raise TypeError("read_1pz() requires a file path, got None")
    if not str(path).strip():
        raise ValueError("read_1pz() requires a non-empty file path")
    path_str = str(Path(path).expanduser())
    if not Path(path_str).exists():
        raise FileNotFoundError(f"File not found: {path_str}")

    mat, rownames, colnames, user_kv = _read_pz_native(path_str)

    # mat is genes × cells (m × n); transpose to cells × genes for AnnData
    adata = ad.AnnData(X=mat.T)

    if rownames:
        import pandas as pd

        adata.var_names = pd.Index(rownames)
    if colnames:
        import pandas as pd

        adata.obs_names = pd.Index(colnames)

    # Compute per-cell total counts from the matrix (column sums of genes×cells)
    adata.obs["total_counts"] = np.array(mat.sum(axis=0)).ravel().astype(np.float64)

    if user_kv:
        adata.uns.update(user_kv)

    return adata


def write_1pz(
    adata,
    path: str | Path,
    *,
    layer: Optional[str] = None,
    store_transpose: bool = False,
    include_obs: bool = True,
    include_var: bool = True,
    include_uns: bool = True,
) -> dict:
    """Write AnnData to a single-block ``.1pz`` file.

    Note
    ----
    ``store_transpose=True`` is not supported by the in-tree codec; passing
    it raises :class:`NotImplementedError`.  ``include_obs`` and
    ``include_var`` are accepted for API compatibility but the native codec
    only persists flat ``user_kv``; obs/var DataFrames are not stored.
    If ``include_uns=True`` (default), scalar-valued ``adata.uns`` entries
    are serialised as ``user_kv`` strings.
    """
    import numpy as np
    import scipy.sparse as sp

    from singlet._pz import write_1pz as _native_write

    if store_transpose:
        raise NotImplementedError(
            "store_transpose=True is not supported by the in-tree TP1Z codec."
        )

    if adata is None:
        raise TypeError("write_1pz() requires an AnnData object, got None")
    if not hasattr(adata, "X") or not hasattr(adata, "obs"):
        raise TypeError(f"write_1pz() requires an AnnData object, got {type(adata).__name__}")

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    mat = adata.layers[layer] if layer else adata.X
    if not sp.issparse(mat):
        mat = sp.csc_matrix(mat)
    # AnnData is cells × genes → genes × cells on disk
    mat = mat.T.tocsc()
    # The reader refuses a column that repeats a row, so merge duplicates and sort before writing.
    mat.sum_duplicates()
    mat.sort_indices()

    # Convert data to an integer dtype acceptable by the native codec.
    # Count matrices are often stored as float32 with integer values.
    data = mat.data
    if np.issubdtype(data.dtype, np.floating):
        data = np.round(data).astype(np.int64)
    elif not np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.int64)

    # Choose smallest unsigned width that fits all values
    max_val = int(data.max()) if len(data) > 0 else 0
    if max_val <= 255:
        data = data.astype(np.uint8)
    elif max_val <= 65535:
        data = data.astype(np.uint16)
    else:
        data = data.astype(np.uint32)

    indptr = mat.indptr.astype(np.int32)
    indices = mat.indices.astype(np.int32)
    m, n = mat.shape  # genes × cells

    rownames = list(adata.var_names) if len(adata.var_names) > 0 else None
    colnames = list(adata.obs_names) if len(adata.obs_names) > 0 else None

    user_meta = None
    if include_uns and adata.uns:
        user_meta = {
            k: str(v) for k, v in adata.uns.items() if isinstance(v, (str, int, float, bool))
        }
        if not user_meta:
            user_meta = None

    _native_write(
        str(path),
        indptr,
        indices,
        data,
        m,
        n,
        rownames=rownames,
        colnames=colnames,
        user_meta=user_meta,
    )

    nnz = int(mat.nnz)
    return {"path": str(path), "m": m, "n": n, "nnz": nnz}


def info_1pz(path: str | Path) -> dict:
    """Read a single-block ``.1pz`` header without decompressing payloads."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    from singlet._pz import info_1pz as _info

    return _info(str(path))


def read_matrix(path: str | Path, **kwargs):
    """Return an AnnData from a ``.1pz`` file (checks the magic bytes, then calls :func:`read_1pz`)."""
    _detect_format(path)
    return read_1pz(path)


def read_kraken2(gse_dir: str | Path):
    """Read a ``kraken2.1pz`` microbiome matrix from a GSE directory.

    Returns an AnnData of cells × taxa.
    """
    import anndata as ad
    import pandas as pd

    gse_dir = Path(gse_dir)
    k2_path = gse_dir / "kraken2.1pz"
    if not k2_path.exists():
        raise FileNotFoundError(f"No kraken2.1pz in {gse_dir}")

    mat, rownames, _colnames, user_kv = _read_pz_native(str(k2_path))

    # mat is taxa × cells; transpose to cells × taxa
    adata = ad.AnnData(X=mat.T)
    if rownames:
        adata.var_names = pd.Index(rownames)

    feat_path = gse_dir / "kraken2_features.parquet"
    if feat_path.exists():
        feat_df = pd.read_parquet(feat_path)
        feat_df.index = adata.var_names
        for col in feat_df.columns:
            adata.var[col] = feat_df[col].values

    if user_kv:
        adata.uns.update(user_kv)

    return adata
