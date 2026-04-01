""".spz file I/O — wraps the _singlepress C++ extension.

Supports both new singlepress format (v1, 64-byte header, delta+varint)
and legacy sparsepress_v2 format (v2, 128-byte header, rANS encoding).
Legacy format is auto-detected and requires the `sparsepress` package.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional


def _detect_format_version(path: str | Path) -> int:
    """Detect .spz format version from file header.

    Returns 1 for new singlepress format, 2 for legacy sparsepress_v2.

    Discriminator: both formats have version=2 at byte 4, but:
    - Legacy: uint16 header_size=128 at bytes 6-7
    - New: uint8 row_sorted (0|1) at byte 6, uint8 reserved=0 at byte 7
    """
    with open(path, "rb") as f:
        header = f.read(8)
    if len(header) < 8:
        raise ValueError(f"File too small to be a valid .spz: {path}")
    magic = header[:4]
    if magic not in (b"SPRZ", b"SPZ2"):
        raise ValueError(f"Not a .spz file (bad magic): {path}")
    # Check header_size field at bytes 6-7 (uint16 LE for old format)
    hdr_size_field = struct.unpack_from("<H", header, 6)[0]
    if hdr_size_field == 128:
        return 2  # legacy sparsepress_v2
    return 1  # new singlepress


def read_spz(path: str | Path, *, col_range: tuple[int, int] | None = None):
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
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

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

def _import_sparsepress():
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


def _result_to_anndata(result):
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


def _read_spz_legacy(path: str | Path, *, col_range=None):
    """Read a legacy sparsepress_v2 file via the sparsepress package."""
    sp_mod = _import_sparsepress()

    if col_range is not None:
        result = sp_mod.sp_read(str(path))
        # Apply column subsetting manually
        import numpy as np
        import scipy.sparse as sp_lib

        shape = tuple(result["shape"])
        mat = sp_lib.csc_matrix(
            (result["data"], result["indices"], result["indptr"]),
            shape=shape,
        )
        mat = mat[:, col_range[0]:col_range[1]]
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



