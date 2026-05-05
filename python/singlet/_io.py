""".spz / .1pz file I/O — wraps the singlepress C++ extensions.

Supports three formats:
  .1pz — New VOCSC + byte-split + zstd-3 format (preferred)
  .spz (v1) — singlepress format (64-byte header, delta+varint)
  .spz (v2) — legacy sparsepress_v2 format (128-byte header, rANS encoding)

Format is auto-detected from magic bytes:
  TP1Z (0x5A315054) → .1pz
  SPRZ (0x5A525053) → .spz v1
  SPZ2 (0x53505A32) → .spz v2
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

# .1pz magic bytes (little-endian)
_TP1_MAGIC = b"\x54\x50\x31\x5a"  # 0x5A315054 = "TP1Z"


def _detect_format(path: str | Path) -> str:
    """Detect file format from header magic bytes.

    Returns
    -------
    str
        '1pz' for .1pz format, 'spz1' for singlepress, 'spz2' for legacy.
    """
    with open(path, "rb") as f:
        header = f.read(8)
    if len(header) < 8:
        raise ValueError(f"File too small to be a valid matrix file: {path}")
    magic = header[:4]
    if magic == _TP1_MAGIC:
        return "1pz"
    if magic in (b"SPRZ", b"SPZ2"):
        hdr_size_field = struct.unpack_from("<H", header, 6)[0]
        if hdr_size_field == 128:
            return "spz2"
        return "spz1"
    raise ValueError(f"Unknown file format (bad magic): {path}")


def _detect_format_version(path: str | Path) -> int:
    """Detect .spz format version from file header (legacy compat).

    Returns 1 for new singlepress format, 2 for legacy sparsepress_v2.
    """
    fmt = _detect_format(path)
    if fmt == "spz2":
        return 2
    if fmt == "spz1":
        return 1
    raise ValueError(f"Not a .spz file: {path}")


def read_spz(path: str | Path, *, col_range: tuple[int, int] | None = None) -> "anndata.AnnData":
    """Read a .spz file into AnnData.

    Parameters
    ----------
    path : str or Path
        Path to a .spz file.
    col_range : tuple of (start, end), optional
        Read only columns [start, end) for streaming/chunked reads.

    Returns
    -------
    anndata.AnnData
        Sparse count matrix (CSR) with gene names in var and
        cell barcodes in obs. (.spz stores genes×cells, we transpose.)
    """

    version = _detect_format_version(path)

    if version == 2:
        return _read_spz_legacy(path, col_range=col_range)

    from singlet._singlepress import sp_read, sp_read_columns

    if col_range is not None:
        result = sp_read_columns(str(path), col_range[0], col_range[1])
    else:
        result = sp_read(str(path))

    return _result_to_anndata(result)


def write_spz(
    adata,
    path: str | Path,
    *,
    layer: Optional[str] = None,
    row_sort: bool = False,
    precision: str = "auto",
) -> dict:
    """Write AnnData to .spz format.

    Parameters
    ----------
    adata : anndata.AnnData
        The data to write. Must have a sparse or dense X matrix.
    path : str or Path
        Output file path.
    layer : str, optional
        Write this layer instead of X.
    row_sort : bool
        Sort genes by nnz before compression (better ratio).
    precision : str
        Value type: "auto", "uint8", "uint16", "int32", "fp32", "fp64".

    Returns
    -------
    dict
        Compression statistics (raw_bytes, compressed_bytes, ratio, etc.).
    """
    import numpy as np
    import scipy.sparse as sp

    from singlet._singlepress import sp_write, sp_write_int

    mat = adata.layers[layer] if layer else adata.X
    if not sp.issparse(mat):
        mat = sp.csc_matrix(mat)

    # AnnData is cells × genes → transpose to genes × cells for .spz
    mat = mat.T.tocsc()

    rownames = list(adata.var_names)
    colnames = list(adata.obs_names)

    # Use integer path when values are integral
    if np.issubdtype(mat.dtype, np.integer):
        return sp_write_int(
            mat.indptr.astype(np.int32),
            mat.indices.astype(np.int32),
            mat.data.astype(np.int32),
            mat.shape[0],
            str(path),
            row_sort=row_sort,
            rownames=rownames,
            colnames=colnames,
        )
    else:
        return sp_write(
            mat.indptr.astype(np.int32),
            mat.indices.astype(np.int32),
            mat.data.astype(np.float64),
            mat.shape[0],
            str(path),
            row_sort=row_sort,
            precision=precision,
            rownames=rownames,
            colnames=colnames,
        )


def spz_info(path: str | Path) -> dict:
    """Read .spz header without decompressing.

    Returns
    -------
    dict
        File metadata: rows, cols, nnz, compression ratio, value_type, etc.
    """
    version = _detect_format_version(path)
    if version == 2:
        return _spz_info_legacy(path)
    from singlet._singlepress import sp_info

    return sp_info(str(path))


# ============================================================================
# Legacy sparsepress_v2 support
# ============================================================================


def _import_sparsepress() -> "module":
    """Import the sparsepress package for legacy v2 format support."""
    try:
        import sparsepress

        return sparsepress
    except ImportError:
        pass

    # Try relative to workspace root (development layout)
    import sys
    from pathlib import Path

    workspace = Path(__file__).resolve().parent.parent.parent
    sp_dir = workspace / "sparsepress"
    if sp_dir.is_dir() and str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
        try:
            import sparsepress

            return sparsepress
        except ImportError:
            pass

    raise ImportError(
        "This .spz file uses the legacy sparsepress_v2 format (v2). "
        "Install the 'sparsepress' package to read it, or convert it "
        "to the new singlepress format."
    )


def _result_to_anndata(result: dict) -> "anndata.AnnData":
    """Convert a raw sp_read result dict to AnnData."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    data = np.asarray(result["data"])
    indices = np.asarray(result["indices"])
    indptr = np.asarray(result["indptr"])
    shape = tuple(result["shape"])

    # Repair non-monotonic indptr (legacy sparsepress reader bug at chunk
    # boundaries can cause decreases of 1 in the column pointer array).
    indptr = np.maximum.accumulate(indptr)

    mat = sp.csc_matrix((data, indices, indptr), shape=shape)

    # Transpose to cells × genes for AnnData
    adata = ad.AnnData(X=mat.T)

    if "rownames" in result and result["rownames"] is not None:
        import pandas as pd

        adata.var_names = pd.Index(result["rownames"])
    if "colnames" in result and result["colnames"] is not None:
        import pandas as pd

        adata.obs_names = pd.Index(result["colnames"])

    return adata


def _read_spz_legacy(path: str | Path, *, col_range: tuple[int, int] | None = None) -> "anndata.AnnData":
    """Read a legacy sparsepress_v2 file via the sparsepress package."""
    sp_mod = _import_sparsepress()

    if col_range is not None:
        result = sp_mod.sp_read(str(path))
        # Apply column subsetting manually
        import scipy.sparse as sp_lib

        shape = tuple(result["shape"])
        mat = sp_lib.csc_matrix(
            (result["data"], result["indices"], result["indptr"]),
            shape=shape,
        )
        mat = mat[:, col_range[0] : col_range[1]]
        result = {
            "data": mat.data,
            "indices": mat.indices,
            "indptr": mat.indptr,
            "shape": list(mat.shape),
        }
        # Preserve dimnames if present
        if "rownames" in result:
            result["rownames"] = result.get("rownames")

    else:
        result = sp_mod.sp_read(str(path))

    return _result_to_anndata(result)


def _spz_info_legacy(path: str | Path) -> dict:
    """Read header info from a legacy sparsepress_v2 file."""
    sp_mod = _import_sparsepress()
    return sp_mod.sp_info(str(path))


# ============================================================================
# .1pz format support (VOCSC + byte-split + zstd-3)
# ============================================================================


def read_1pz(path: str | Path) -> "anndata.AnnData":
    """Read a .1pz file into AnnData.

    Parameters
    ----------
    path : str or Path
        Path to a .1pz file.

    Returns
    -------
    anndata.AnnData
        Sparse count matrix (CSR) with genes in var and cells in obs.
        (.1pz stores genes×cells, we transpose.)
        If the file contains metadata, gene names populate var_names
        and cell barcodes populate obs_names.
        Column sums are stored in obs["total_counts"] if available.
        Embedded obs/var DataFrames are merged into adata.obs/adata.var.
        Embedded uns dict is stored in adata.uns.
    """
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    try:
        import singlepress

        mat = singlepress.read_1pz(str(path), num_threads=8)
    except (ImportError, AttributeError):
        from singlepress._pz_codec import pz_read

        result = pz_read(str(path), 8)
        m, n = result["m"], result["n"]
        mat = sp.csc_matrix(
            (result["values"], result["indices"], result["indptr"]),
            shape=(m, n),
        )

    # Transpose to cells × genes for AnnData
    adata = ad.AnnData(X=mat.T)

    # Attach metadata if present
    if hasattr(mat, "rownames") and mat.rownames:
        import pandas as pd

        adata.var_names = pd.Index(mat.rownames)
    if hasattr(mat, "colnames") and mat.colnames:
        import pandas as pd

        adata.obs_names = pd.Index(mat.colnames)
    if hasattr(mat, "colsums") and mat.colsums is not None:
        adata.obs["total_counts"] = mat.colsums.astype(np.float64)

    # Merge embedded obs DataFrame
    if hasattr(mat, "obs") and mat.obs is not None:
        import pandas as pd

        obs_df = mat.obs
        obs_df.index = adata.obs_names
        for col in obs_df.columns:
            adata.obs[col] = obs_df[col].values

    # Merge embedded var DataFrame
    if hasattr(mat, "var") and mat.var is not None:
        import pandas as pd

        var_df = mat.var
        var_df.index = adata.var_names
        for col in var_df.columns:
            adata.var[col] = var_df[col].values

    # Store unstructured metadata
    if hasattr(mat, "uns") and mat.uns:
        adata.uns.update(mat.uns)

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
    """Write AnnData to .1pz format (VOCSC + zstd-3).

    Parameters
    ----------
    adata : anndata.AnnData
        The data to write.
    path : str or Path
        Output file path.
    layer : str, optional
        Write this layer instead of X.
    store_transpose : bool
        Store transpose for efficient row-range reads.
    include_obs : bool
        Embed adata.obs DataFrame in the .1pz file. Default True.
    include_var : bool
        Embed adata.var DataFrame in the .1pz file. Default True.
    include_uns : bool
        Embed string key-value pairs from adata.uns. Default True.

    Returns
    -------
    dict
        Compression statistics.
    """
    import scipy.sparse as sp
    import singlepress

    mat = adata.layers[layer] if layer else adata.X
    if not sp.issparse(mat):
        mat = sp.csc_matrix(mat)

    # AnnData is cells × genes → transpose to genes × cells
    mat = mat.T.tocsc()

    # Extract names for metadata
    rownames = list(adata.var_names) if len(adata.var_names) > 0 else None
    colnames = list(adata.obs_names) if len(adata.obs_names) > 0 else None

    # Extract obs/var DataFrames
    obs_df = adata.obs if include_obs and len(adata.obs.columns) > 0 else None
    var_df = adata.var if include_var and len(adata.var.columns) > 0 else None

    # Extract string key-value pairs from uns
    uns_dict = None
    if include_uns and adata.uns:
        uns_dict = {
            k: str(v) for k, v in adata.uns.items() if isinstance(v, (str, int, float, bool))
        }
        if not uns_dict:
            uns_dict = None

    return singlepress.write_1pz(
        str(path),
        mat,
        rownames=rownames,
        colnames=colnames,
        obs=obs_df,
        var=var_df,
        uns=uns_dict,
        store_transpose=store_transpose,
    )


def info_1pz(path: str | Path) -> dict:
    """Read .1pz file header without decompressing."""
    import singlepress

    return singlepress.info_1pz(str(path))


def read_matrix(path: str | Path, **kwargs):
    """Auto-detect format and read a .spz or .1pz file into AnnData.

    Parameters
    ----------
    path : str or Path
        Path to a .spz or .1pz file.

    Returns
    -------
    anndata.AnnData
    """
    fmt = _detect_format(path)
    if fmt == "1pz":
        return read_1pz(path)
    else:
        return read_spz(path, **kwargs)


def read_kraken2(
    gse_dir: str | Path,
):
    """Read a kraken2.1pz microbiome matrix from a GSE directory.

    Parameters
    ----------
    gse_dir : str or Path
        Path to a GSE directory containing ``kraken2.1pz`` and
        ``kraken2_features.parquet``.

    Returns
    -------
    anndata.AnnData
        Sparse count matrix (cells × taxa) with taxon IDs in var_names.
        If ``kraken2_features.parquet`` exists, taxon metadata is in var.
    """
    import anndata as ad
    import pandas as pd

    gse_dir = Path(gse_dir)
    k2_path = gse_dir / "kraken2.1pz"
    if not k2_path.exists():
        raise FileNotFoundError(f"No kraken2.1pz in {gse_dir}")

    try:
        import singlepress

        mat = singlepress.read_1pz(str(k2_path), num_threads=8)
    except (ImportError, AttributeError):
        import scipy.sparse as sp
        from singlepress._pz_codec import pz_read

        result = pz_read(str(k2_path), 8)
        mat = sp.csc_matrix(
            (result["values"], result["indices"], result["indptr"]),
            shape=(result["m"], result["n"]),
        )

    # Transpose: on-disk is taxa × cells → cells × taxa for AnnData
    adata = ad.AnnData(X=mat.T)

    # Taxon names from rownames
    if hasattr(mat, "rownames") and mat.rownames:
        adata.var_names = pd.Index(mat.rownames)

    # Load taxon features if available
    feat_path = gse_dir / "kraken2_features.parquet"
    if feat_path.exists():
        feat_df = pd.read_parquet(feat_path)
        feat_df.index = adata.var_names
        for col in feat_df.columns:
            adata.var[col] = feat_df[col].values

    # Store uns metadata
    if hasattr(mat, "uns") and mat.uns:
        adata.uns.update(mat.uns)

    return adata
