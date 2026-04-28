"""
singlepress — Python interface for the .1pz sparse matrix format.

Format:
  .1pz  — VOCSC + byte-split + zstd-3 (13x compression, 4000+ MB/s decode)
           CRC32 integrity, metadata, column sums, partial reads, transpose
           AnnData-like: embedded obs (cell metadata), var (gene metadata), uns (study metadata)

Usage:
    import singlepress as sp
    import scipy.sparse as ss
    import pandas as pd

    mat = ss.random(10000, 5000, density=0.05, format='csc', dtype='float64')

    # Write/read .1pz
    sp.write_1pz("matrix.1pz", mat, rownames=["g1","g2",...], colnames=["c1",...])
    mat2 = sp.read_1pz("matrix.1pz")

    # Write with AnnData-like metadata
    obs = pd.DataFrame({"cell_type": ["T-cell", "B-cell", ...], "gsm_id": [...]})
    var = pd.DataFrame({"gene_name": [...]})
    sp.write_1pz("matrix.1pz", mat,
                 obs=obs, var=var,
                 uns={"gse_id": "GSE128553", "organism": "Mus musculus"})

    # Lazy/partial access
    pz = sp.open_1pz("matrix.1pz")
    pz.shape, pz.colsums, pz.rownames
    pz.obs        # pandas DataFrame (cell metadata)
    pz.var        # pandas DataFrame (gene metadata)
    pz.uns        # dict (study metadata)
    sub = pz.read(cols=(0, 100))
    norm = pz.read_normalized(scale=10000)

    # Log-normalization from stored column sums
    norm_mat = sp.lognorm(mat2, pz.colsums, scale=10000)
"""

from __future__ import annotations

import os
import numpy as np
import scipy.sparse as ss
from typing import Optional, Union

__all__ = [
    "write_1pz",
    "read_1pz",
    "read_1pz_int",
    "read_1pz_columns",
    "info_1pz",
    "validate_1pz",
    "colsums_1pz",
    "open_1pz",
    "cbind_1pz",
    "rbind_1pz",
    "subset_1pz",
    "sample_1pz",
    "lognorm",
    "OnePZFile",
    # Submodules (lazy-loaded)
    "interop",
    "torch",
]


# ============================================================================
# .1pz format — VOCSC + byte-split + zstd-3, CRC32, metadata, colsums
# ============================================================================

def _human_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}" if nbytes != int(nbytes) else f"{int(nbytes)} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"

def _df_to_columns(df: "pd.DataFrame") -> tuple[list[str], list[tuple]]:
    """Convert a pandas DataFrame to (index, column_specs) for C++ serialization.

    Returns
    -------
    index : list[str]
        Index values as strings.
    col_specs : list[tuple(name, data)]
        Each tuple is (column_name, data) where data is:
        - list[str] for string/object columns
        - numpy array for numeric columns (int32/int64/float32/float64)
        - numpy uint8 array for bool columns
        - dict(levels=list[str], codes=np.int32) for categorical columns
    """
    import pandas as pd

    index = [str(x) for x in df.index]
    col_specs = []
    for col_name in df.columns:
        series = df[col_name]
        if hasattr(series, "cat"):
            # Categorical / factor
            cat = series.cat
            col_specs.append((str(col_name), {
                "levels": [str(x) for x in cat.categories],
                "codes": np.ascontiguousarray(cat.codes.values, dtype=np.int32),
            }))
        elif series.dtype == object or series.dtype.name == "string":
            col_specs.append((str(col_name), [str(x) for x in series]))
        elif np.issubdtype(series.dtype, np.bool_):
            col_specs.append((str(col_name),
                np.ascontiguousarray(series.values.astype(np.uint8))))
        elif series.dtype == np.int32:
            col_specs.append((str(col_name),
                np.ascontiguousarray(series.values, dtype=np.int32)))
        elif np.issubdtype(series.dtype, np.integer):
            col_specs.append((str(col_name),
                np.ascontiguousarray(series.values, dtype=np.int64)))
        elif series.dtype == np.float32:
            col_specs.append((str(col_name),
                np.ascontiguousarray(series.values, dtype=np.float32)))
        elif np.issubdtype(series.dtype, np.floating):
            col_specs.append((str(col_name),
                np.ascontiguousarray(series.values, dtype=np.float64)))
        else:
            # Fallback: convert to string
            col_specs.append((str(col_name), [str(x) for x in series]))
    return index, col_specs


def _columns_to_df(data: dict) -> "pd.DataFrame":
    """Reconstruct a pandas DataFrame from C++ deserialized column data.

    Parameters
    ----------
    data : dict
        Keys are column names, values are numpy arrays or string lists.
        Special key "__index__" holds the DataFrame index.
        Categorical columns are dicts with "levels" and "codes" keys.
    """
    import pandas as pd

    index = data.pop("__index__", None)
    columns = {}
    for name, values in data.items():
        if isinstance(values, dict) and "levels" in values and "codes" in values:
            # Categorical
            columns[name] = pd.Categorical.from_codes(
                values["codes"], categories=values["levels"])
        elif isinstance(values, list):
            columns[name] = values
        else:
            columns[name] = values
    df = pd.DataFrame(columns)
    if index is not None and len(index) > 0:
        df.index = pd.Index(index)
    return df

def write_1pz(
    path: str,
    matrix: ss.spmatrix,
    *,
    rownames: Optional[list[str]] = None,
    colnames: Optional[list[str]] = None,
    obs: Optional["pd.DataFrame"] = None,
    var: Optional["pd.DataFrame"] = None,
    uns: Optional[dict[str, str]] = None,
    store_transpose: bool = False,
    num_threads: int = 8,
    level: int = 3,
    chunk_cols: int = 1024,
    mode: str = "default",
) -> dict:
    """Write a scipy sparse matrix to a .1pz file.

    Parameters
    ----------
    path : str
        Output file path (should end in .1pz).
    matrix : scipy.sparse.spmatrix
        Sparse matrix (will be converted to CSC if needed).
        Values must be non-negative integers.
    rownames : list[str] or None
        Row labels (e.g., gene names). Stored in file metadata.
    colnames : list[str] or None
        Column labels (e.g., cell barcodes). Stored in file metadata.
    obs : pandas.DataFrame or None
        Cell-level metadata (one row per column of the matrix).
        Analogous to AnnData.obs. Serialized in native columnar format.
    var : pandas.DataFrame or None
        Feature-level metadata (one row per row of the matrix).
        Analogous to AnnData.var. Serialized in native columnar format.
    uns : dict[str, str] or None
        Unstructured string key-value metadata (e.g., study accession, organism).
        Analogous to AnnData.uns. All values must be strings.
    store_transpose : bool
        Also store the transposed matrix for efficient row-range reads.
        Roughly doubles file size but enables fast gene-range access.
    num_threads : int
        OpenMP threads for parallel encoding. Default 8.
    level : int
        Zstd compression level (used when mode="default"). Default 3.
    chunk_cols : int
        Columns per chunk. Default 1024.
    mode : str
        Compression mode controlling the speed/ratio trade-off:

        - ``"fast"`` — LZ4 (3x faster reads, ~6% larger files)
        - ``"default"`` — zstd level 3 (balanced)
        - ``"small"`` — zstd level 16 (best compression, slower writes)

        When *mode* is set, it overrides *level* with the optimal preset.
        To use a custom zstd level, leave mode as ``"default"`` and set *level*.

    Returns
    -------
    dict with compression stats including has_metadata, has_colsums,
    has_transpose, has_obs, has_var, has_uns.
    """
    from ._pz_codec import pz_write, pz_write_int, pz_write_i64, pz_write_int_i64

    _MODE_PRESETS = {
        "fast":    (1, 1),   # codec_id=LZ4,  level=1
        "default": (0, 3),   # codec_id=ZSTD, level=3
        "small":   (0, 16),  # codec_id=ZSTD, level=16
    }
    if mode not in _MODE_PRESETS:
        raise ValueError(
            f"Unknown mode {mode!r}; choose from {list(_MODE_PRESETS)}")
    codec_id, preset_level = _MODE_PRESETS[mode]
    if mode != "default":
        level = preset_level

    if not ss.issparse(matrix):
        matrix = ss.csc_matrix(matrix)
    elif not ss.isspmatrix_csc(matrix):
        matrix = matrix.tocsc()
    matrix.sort_indices()

    nrows, ncols = matrix.shape
    nnz = matrix.nnz
    use_i64 = nnz > np.iinfo(np.int32).max
    idx_dtype = np.int64 if use_i64 else np.int32
    indptr = np.ascontiguousarray(matrix.indptr, dtype=idx_dtype)
    indices = np.ascontiguousarray(matrix.indices, dtype=idx_dtype)

    rn = list(rownames) if rownames is not None else []
    cn = list(colnames) if colnames is not None else []

    # Convert DataFrames to column specs for C++ native serialization
    obs_index, obs_cols = _df_to_columns(obs) if obs is not None else ([], [])
    var_index, var_cols = _df_to_columns(var) if var is not None else ([], [])
    kv_pairs = list(uns.items()) if uns else []

    if np.issubdtype(matrix.data.dtype, np.integer):
        data = np.ascontiguousarray(matrix.data, dtype=np.int32)
        write_fn = pz_write_int_i64 if use_i64 else pz_write_int
        return write_fn(indptr, indices, data, nrows, path,
                        num_threads, level, chunk_cols, rn, cn,
                        store_transpose, kv_pairs,
                        obs_index, obs_cols, var_index, var_cols,
                        codec_id)
    else:
        data = np.ascontiguousarray(matrix.data, dtype=np.float64)
        write_fn = pz_write_i64 if use_i64 else pz_write
        return write_fn(indptr, indices, data, nrows, path,
                        num_threads, level, chunk_cols, rn, cn,
                        store_transpose, kv_pairs,
                        obs_index, obs_cols, var_index, var_cols,
                        codec_id)


def read_1pz(
    path: str,
    *,
    num_threads: int = 8,
    normalize: bool = False,
    scale: float = 10000.0,
) -> ss.csc_matrix:
    """Read a .1pz file and return a scipy.sparse.csc_matrix.

    Metadata (rownames, colnames, colsums, obs, var, uns) are attached
    as attributes on the returned matrix if present in the file.

    Parameters
    ----------
    path : str
        Path to a .1pz file.
    num_threads : int
        Number of decompression threads. Default 8.
    normalize : bool
        If True, apply on-the-fly log-normalization: ``log1p(x * scale / colsum)``.
        Equivalent to Seurat's ``LogNormalize``. Requires stored column sums.
    scale : float
        Scaling factor for normalization. Default 10000 (standard for scRNA-seq).

    Returns
    -------
    scipy.sparse.csc_matrix
        The decompressed sparse matrix with optional .rownames, .colnames,
        .colsums, .obs, .var, .uns attributes.
    """
    from ._pz_codec import pz_read

    result = pz_read(path, num_threads)
    m, n, nnz = result["m"], result["n"], result["nnz"]
    mat = ss.csc_matrix(
        (result["values"], result["indices"], result["indptr"]),
        shape=(m, n),
    )
    if "rownames" in result:
        mat.rownames = list(result["rownames"])
    if "colnames" in result:
        mat.colnames = list(result["colnames"])
    if "colsums" in result:
        mat.colsums = np.asarray(result["colsums"])
    mat.uns = dict(result["uns"]) if "uns" in result else {}
    mat.obs = _columns_to_df(dict(result["obs"])) if "obs" in result else None
    mat.var = _columns_to_df(dict(result["var"])) if "var" in result else None

    if normalize:
        cs = getattr(mat, "colsums", None)
        if cs is None:
            raise ValueError("File has no stored colsums; cannot normalize")
        mat = lognorm(mat, cs, scale=scale)

    return mat


def read_1pz_int(
    path: str,
    *,
    num_threads: int = 8,
) -> ss.csc_matrix:
    """Read a .1pz file with native int32 values (avoids float conversion).

    Returns
    -------
    scipy.sparse.csc_matrix with int32 data dtype and optional metadata attributes.
    """
    from ._pz_codec import pz_read_int

    result = pz_read_int(path, num_threads)
    m, n = result["m"], result["n"]
    mat = ss.csc_matrix(
        (result["values"], result["indices"], result["indptr"]),
        shape=(m, n),
    )
    if "rownames" in result:
        mat.rownames = list(result["rownames"])
    if "colnames" in result:
        mat.colnames = list(result["colnames"])
    if "colsums" in result:
        mat.colsums = np.asarray(result["colsums"])
    mat.uns = dict(result["uns"]) if "uns" in result else {}
    mat.obs = _columns_to_df(dict(result["obs"])) if "obs" in result else None
    mat.var = _columns_to_df(dict(result["var"])) if "var" in result else None
    return mat


def read_1pz_columns(
    path: str,
    col_start: int,
    col_end: int,
    *,
    num_threads: int = 8,
) -> ss.csc_matrix:
    """Read a column range [col_start, col_end) from a .1pz file.

    Only decompresses the chunks covering the requested columns.

    Returns
    -------
    scipy.sparse.csc_matrix
        Submatrix with (col_end - col_start) columns.
    """
    from ._pz_codec import pz_read_columns

    result = pz_read_columns(path, col_start, col_end, num_threads)
    m, n = result["m"], result["n"]
    return ss.csc_matrix(
        (result["values"], result["indices"], result["indptr"]),
        shape=(m, n),
    )


def info_1pz(path: str) -> dict:
    """Read .1pz file header without decompression.

    Returns dict with version, m, n, nnz, has_metadata, has_colsums,
    has_transpose, has_obs_var, codec, etc.
    """
    from ._pz_codec import pz_info
    return pz_info(path)


def validate_1pz(path: str) -> dict:
    """Validate a .1pz file's CRC32 integrity.

    Returns dict with valid (bool), file_crc_ok, footer_ok, error.
    """
    from ._pz_codec import pz_validate
    return pz_validate(path)


def colsums_1pz(path: str) -> np.ndarray:
    """Read just the column sums from a .1pz file (fast, no full decompression).

    Returns
    -------
    numpy.ndarray of uint64, length n (number of columns).
    """
    from ._pz_codec import pz_colsums
    return np.asarray(pz_colsums(path))


def lognorm(
    x: Union[ss.spmatrix, np.ndarray, float],
    colsums: np.ndarray,
    *,
    scale: float = 10000.0,
) -> Union[ss.csc_matrix, np.ndarray, float]:
    """Apply log-normalization: log1p(x * scale / colsum).

    This is the standard scRNA-seq normalization. Column sums can be
    obtained from colsums_1pz() or from the .colsums attribute of
    matrices returned by read_1pz().

    Parameters
    ----------
    x : sparse matrix, array, or scalar
        Raw count data. For sparse matrices, must be CSC format.
    colsums : array of shape (n_cols,)
        Column sums (total UMI per cell).
    scale : float
        Scaling factor. Default 10000 (standard for scRNA-seq).

    Returns
    -------
    Same type as x, with log-normalization applied.
    """
    if isinstance(x, (int, float)):
        # Scalar element normalization
        cs = float(colsums) if np.ndim(colsums) == 0 else float(colsums.ravel()[0])
        return float(np.log1p(x * scale / cs))

    if ss.issparse(x):
        if not ss.isspmatrix_csc(x):
            x = x.tocsc()
        result = x.copy()
        colsums = np.asarray(colsums, dtype=np.float64)
        factors = scale / colsums
        # Vectorized: scale each column's nnz by its factor
        for j in range(x.shape[1]):
            s, e = x.indptr[j], x.indptr[j+1]
            if s < e:
                result.data[s:e] = np.log1p(x.data[s:e] * factors[j])
        return result

    # Dense array
    x = np.asarray(x, dtype=np.float64)
    colsums = np.asarray(colsums, dtype=np.float64)
    return np.log1p(x * (scale / colsums))


class OnePZFile:
    """Lazy handle to a .1pz file for metadata access and partial reads.

    Mirrors AnnData's API: .obs (cell metadata), .var (gene metadata),
    .uns (unstructured study metadata), plus .rownames, .colnames, .colsums.

    Usage::

        pz = singlepress.open_1pz("counts.1pz")
        pz.shape          # (m, n)
        pz.rownames       # gene names (or None)
        mat = pz.read()   # full read (raw counts)
        mat = pz.read(normalize=True)  # on-the-fly log-normalization
        sub = pz.read(cols=(0, 100))   # partial column read

        # Open in normalized mode — all reads return log-normalized data
        pz = singlepress.open_1pz("counts.1pz", normalize=True)
        pz[:, 0:100]      # returns log-normalized submatrix
        pz.to_anndata()    # AnnData with log-normalized counts
    """

    def __init__(self, path: str, num_threads: int = 8,
                 normalize: bool = False, scale: float = 10000.0):
        self.path = str(path)
        self.num_threads = num_threads
        self.normalize = normalize
        self.scale = scale
        self._info = None
        self._colsums = None
        self._meta = None

    def _get_info(self):
        if self._info is None:
            self._info = info_1pz(self.path)
        return self._info

    @property
    def shape(self) -> tuple[int, int]:
        info = self._get_info()
        return (info["m"], info["n"])

    @property
    def nnz(self) -> int:
        return self._get_info()["nnz"]

    @property
    def has_metadata(self) -> bool:
        return self._get_info().get("has_metadata", False)

    @property
    def has_colsums(self) -> bool:
        return self._get_info().get("has_colsums", False)

    @property
    def has_transpose(self) -> bool:
        return self._get_info().get("has_transpose", False)

    @property
    def has_obs_var(self) -> bool:
        return self._get_info().get("has_obs_var", False)

    @property
    def colsums(self) -> Optional[np.ndarray]:
        if self._colsums is None and self.has_colsums:
            self._colsums = colsums_1pz(self.path)
        return self._colsums

    @property
    def rownames(self) -> Optional[list[str]]:
        self._load_meta()
        return self._meta.get("rownames")

    @property
    def colnames(self) -> Optional[list[str]]:
        self._load_meta()
        return self._meta.get("colnames")

    @property
    def obs(self) -> Optional["pd.DataFrame"]:
        """Cell-level metadata DataFrame (one row per column)."""
        self._load_meta()
        return self._meta.get("obs")

    @property
    def var(self) -> Optional["pd.DataFrame"]:
        """Feature-level metadata DataFrame (one row per row)."""
        self._load_meta()
        return self._meta.get("var")

    @property
    def uns(self) -> Optional[dict[str, str]]:
        """Unstructured study-level metadata (string key-value pairs)."""
        self._load_meta()
        return self._meta.get("uns")

    def _load_meta(self):
        if self._meta is not None:
            return
        if not self.has_metadata:
            self._meta = {}
            return
        from ._pz_codec import pz_read
        result = pz_read(self.path, 1)  # single thread, full read
        self._meta = {}
        if "rownames" in result:
            self._meta["rownames"] = list(result["rownames"])
        if "colnames" in result:
            self._meta["colnames"] = list(result["colnames"])
        if "uns" in result:
            self._meta["uns"] = dict(result["uns"])
        if "obs" in result:
            self._meta["obs"] = _columns_to_df(dict(result["obs"]))
        if "var" in result:
            self._meta["var"] = _columns_to_df(dict(result["var"]))

    def _apply_normalize(self, mat, col_offset=None, normalize=None, scale=None):
        """Apply log-normalization if enabled. Internal helper.

        Parameters
        ----------
        mat : csc_matrix
            Raw count matrix to normalize.
        col_offset : tuple (start, end), optional
            If the matrix is a column slice, the original column range.
        normalize : bool or None
            Override self.normalize if not None.
        scale : float or None
            Override self.scale if not None.
        """
        do_norm = normalize if normalize is not None else self.normalize
        if not do_norm:
            return mat
        sc = scale if scale is not None else self.scale
        cs = self.colsums
        if cs is None:
            raise ValueError("File has no stored colsums; cannot normalize")
        if col_offset is not None:
            cs = cs[col_offset[0]:col_offset[1]]
        return lognorm(mat, cs, scale=sc)

    def read(
        self,
        *,
        cols: Optional[tuple[int, int]] = None,
        rows: Optional[tuple[int, int]] = None,
        dtype: str = "float64",
        normalize: Optional[bool] = None,
        scale: Optional[float] = None,
    ) -> ss.csc_matrix:
        """Read the matrix (full or partial).

        Parameters
        ----------
        cols : tuple (start, end), optional
            Read only columns [start, end). Efficient: only relevant chunks decoded.
        rows : tuple (start, end), optional
            Read only rows [start, end). Requires stored transpose.
        dtype : "float64" or "int32"
            Value dtype. Use "int32" for raw counts to save memory.
        normalize : bool, optional
            If True, apply on-the-fly log-normalization: ``log1p(x * scale / colsum)``.
            Defaults to the handle's ``normalize`` setting (set at construction or
            via :meth:`normalized`).
        scale : float, optional
            Scaling factor for normalization. Defaults to the handle's ``scale``.
        """
        if rows is not None:
            from ._pz_codec import pz_read_rows
            result = pz_read_rows(self.path, rows[0], rows[1], self.num_threads)
            m, n = result["m"], result["n"]
            mat = ss.csc_matrix(
                (result["values"], result["indices"], result["indptr"]),
                shape=(m, n),
            )
            # This is CSC of X^T[:, row_start:row_end], transpose to get rows
            return mat.T.tocsc()

        if cols is not None:
            if dtype == "int32":
                from ._pz_codec import pz_read_columns_int
                result = pz_read_columns_int(self.path, cols[0], cols[1], self.num_threads)
            else:
                from ._pz_codec import pz_read_columns
                result = pz_read_columns(self.path, cols[0], cols[1], self.num_threads)
            m, n = result["m"], result["n"]
            mat = ss.csc_matrix(
                (result["values"], result["indices"], result["indptr"]),
                shape=(m, n),
            )
            return self._apply_normalize(mat, col_offset=cols, normalize=normalize, scale=scale)

        if dtype == "int32":
            mat = read_1pz_int(self.path, num_threads=self.num_threads)
        else:
            mat = read_1pz(self.path, num_threads=self.num_threads)
        return self._apply_normalize(mat, normalize=normalize, scale=scale)

    def read_normalized(self, *, scale: float = 10000.0) -> ss.csc_matrix:
        """Read and apply log-normalization: log1p(x * scale / colsum).

        Uses stored column sums for efficient normalization without
        needing a separate pass over the data.
        """
        return self.read(normalize=True, scale=scale)

    def normalized(self, scale: float = 10000.0) -> "OnePZFile":
        """Return a new handle that normalizes on every read.

        Equivalent to opening the file with ``normalize=True``.
        All subsequent reads, indexing, and conversions on the returned
        handle will automatically apply ``log1p(x * scale / colsum)``.

        Parameters
        ----------
        scale : float
            Scaling factor. Default 10000 (standard for scRNA-seq).

        Returns
        -------
        OnePZFile
            A copy of this handle with normalization enabled.

        Examples
        --------
        >>> pz = singlepress.open_1pz("counts.1pz")
        >>> npz = pz.normalized()
        >>> npz[:, 0:100]        # returns log-normalized data
        >>> npz.to_anndata()     # AnnData with log-normalized counts
        """
        return OnePZFile(self.path, num_threads=self.num_threads,
                         normalize=True, scale=scale)

    def __repr__(self):
        info = self._get_info()
        m, n = info["m"], info["n"]
        parts = [f"OnePZFile({self.path!r}, shape=({m}, {n}), nnz={info['nnz']}"]
        if info.get("has_obs_var"):
            parts.append(", obs/var=True")
        parts.append(f", v1.0)")
        return "".join(parts)

    def __str__(self):
        info = self._get_info()
        m, n, nnz = info["m"], info["n"], info["nnz"]
        density = nnz / (m * n) * 100 if m * n > 0 else 0.0
        try:
            size = os.path.getsize(self.path)
        except OSError:
            size = 0

        lines = [
            f"OnePZFile: {os.path.basename(self.path)}",
            f"  Shape:   {m:,} features x {n:,} cells",
            f"  NNZ:     {nnz:,} ({density:.1f}% dense)",
            f"  Size:    {_human_size(size)}",
            f"  Format:  .1pz v1.0",
        ]

        flags = []
        if info.get("has_metadata"):
            flags.append("names")
        if info.get("has_colsums"):
            flags.append("colsums")
        if info.get("has_transpose"):
            flags.append("transpose")
        if info.get("has_obs_var"):
            flags.append("obs/var")
        if flags:
            lines.append(f"  Stored:  {', '.join(flags)}")

        if self.normalize:
            lines.append(f"  Normalize: log1p(x * {self.scale:.0f} / colsum)")

        rn = self.rownames
        if rn:
            preview = ", ".join(rn[:3])
            suffix = f", ... ({len(rn):,} total)" if len(rn) > 3 else ""
            lines.append(f"  Rows:    {preview}{suffix}")

        cn = self.colnames
        if cn:
            preview = ", ".join(cn[:3])
            suffix = f", ... ({len(cn):,} total)" if len(cn) > 3 else ""
            lines.append(f"  Cols:    {preview}{suffix}")

        obs = self.obs
        if obs is not None:
            lines.append(f"  obs:     {obs.shape[0]:,} cells x {obs.shape[1]} fields ({', '.join(obs.columns[:4])}{'...' if len(obs.columns) > 4 else ''})")

        var = self.var
        if var is not None:
            lines.append(f"  var:     {var.shape[0]:,} features x {var.shape[1]} fields ({', '.join(var.columns[:4])}{'...' if len(var.columns) > 4 else ''})")

        uns = self.uns
        if uns:
            lines.append(f"  uns:     {len(uns)} entries ({', '.join(list(uns.keys())[:4])}{'...' if len(uns) > 4 else ''})")

        return "\n".join(lines)

    # -- Conversion methods --------------------------------------------------

    def to_csc(self) -> ss.csc_matrix:
        """Read as scipy CSC matrix (native format). Respects normalize setting."""
        return self.read()

    def to_csr(self) -> ss.csr_matrix:
        """Read as scipy CSR matrix. Respects normalize setting."""
        return self.read().tocsr()

    def to_coo(self) -> ss.coo_matrix:
        """Read as scipy COO matrix. Respects normalize setting."""
        return self.read().tocoo()

    def to_dense(self) -> np.ndarray:
        """Read as dense numpy array. Respects normalize setting."""
        return self.read().toarray()

    def to_dataframe(self) -> "pd.DataFrame":
        """Read as pandas DataFrame. Respects normalize setting."""
        if self.normalize:
            import pandas as pd
            mat = self.read()
            rn = self.rownames
            cn = self.colnames
            return pd.DataFrame(mat.toarray(),
                                index=rn if rn else None,
                                columns=cn if cn else None)
        from singlepress.interop import to_dataframe
        return to_dataframe(self.path, num_threads=self.num_threads)

    def to_anndata(self):
        """Read as AnnData object (cells x genes). Respects normalize setting."""
        if self.normalize:
            import anndata
            mat = self.read()
            rn = self.rownames
            cn = self.colnames
            obs_df = self.obs
            var_df = self.var
            adata = anndata.AnnData(
                X=mat.T.tocsr(),
                obs=obs_df if obs_df is not None else None,
                var=var_df if var_df is not None else None,
            )
            if cn is not None:
                adata.obs_names = cn
            if rn is not None:
                adata.var_names = rn
            return adata
        from singlepress.interop import to_anndata
        return to_anndata(self.path, num_threads=self.num_threads)

    def __array__(self, dtype=None):
        """Support ``numpy.asarray(pz)``."""
        arr = self.to_dense()
        return arr if dtype is None else arr.astype(dtype)

    # -- AnnData-style aliases -----------------------------------------------

    @property
    def n_obs(self) -> int:
        """Number of observations (cells/columns). AnnData convention."""
        return self.shape[1]

    @property
    def n_vars(self) -> int:
        """Number of variables (features/rows). AnnData convention."""
        return self.shape[0]

    @property
    def obs_names(self) -> Optional[list[str]]:
        """Cell barcodes (alias for colnames). AnnData convention."""
        return self.colnames

    @property
    def var_names(self) -> Optional[list[str]]:
        """Gene names (alias for rownames). AnnData convention."""
        return self.rownames

    @property
    def T(self) -> ss.csc_matrix:
        """Transpose of the matrix. Returns materialized CSC (cells×genes)."""
        return self.read().T.tocsc()

    # -- Indexing: pz[rows, cols] --------------------------------------------

    def __getitem__(self, key):
        """Subset the matrix by row/column indices, names, or boolean masks.

        Returns a scipy.sparse.csc_matrix (materialized).

        Supported indexing patterns::

            pz[row_slice, col_slice]           # integer ranges
            pz[row_list, :]                    # list of int indices
            pz[["CD3D", "CD4"], :]             # gene names (requires rownames)
            pz[:, bool_array]                  # boolean cell mask
            pz["CD3D"]                         # single gene → 1×n sparse row
            pz[:, 0:100]                       # first 100 cells

        Parameters
        ----------
        key : tuple, str, list, slice
            Indexing expression.

        Returns
        -------
        scipy.sparse.csc_matrix
        """
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("OnePZFile indexing requires exactly 2 dimensions [rows, cols]")
            row_idx, col_idx = key
        elif isinstance(key, str):
            # Single gene name
            row_idx, col_idx = key, slice(None)
        elif isinstance(key, (list, np.ndarray)):
            row_idx, col_idx = key, slice(None)
        else:
            row_idx, col_idx = key, slice(None)

        m, n = self.shape

        # Resolve column index
        col_indices = self._resolve_axis_index(col_idx, n, self.colnames, "col")

        # Resolve row index
        row_indices = self._resolve_axis_index(row_idx, m, self.rownames, "row")

        # Fast path: contiguous column range, all rows
        if row_indices is None and isinstance(col_indices, tuple):
            return self.read(cols=col_indices)

        # Fast path: contiguous column range with row subset
        if isinstance(col_indices, tuple):
            mat = self.read(cols=col_indices)
        elif col_indices is None:
            mat = self.read()
        else:
            # Arbitrary column selection — read full, then subset
            mat = self.read()
            if isinstance(col_indices, tuple):
                mat = mat[:, col_indices[0]:col_indices[1]]
            else:
                mat = mat[:, col_indices]

        if row_indices is not None:
            if isinstance(row_indices, tuple):
                mat = mat[row_indices[0]:row_indices[1], :]
            else:
                mat = mat[row_indices, :]

        return mat

    def _resolve_axis_index(self, idx, size, names, axis):
        """Resolve an index to None (all), tuple (start, end), or array.

        Returns
        -------
        None : select all
        tuple[int, int] : contiguous range [start, end)
        np.ndarray : integer index array
        """
        if idx is None or (isinstance(idx, slice) and idx == slice(None)):
            return None

        if isinstance(idx, slice):
            start, stop, step = idx.indices(size)
            if step == 1:
                return (start, stop)
            return np.arange(start, stop, step)

        if isinstance(idx, str):
            if names is None:
                raise IndexError(f"Cannot index by name — no {axis}names stored in file")
            try:
                i = names.index(idx)
            except ValueError:
                raise IndexError(f"{axis}name {idx!r} not found")
            return np.array([i])

        if isinstance(idx, (list, tuple)):
            if len(idx) == 0:
                return np.array([], dtype=np.intp)
            if isinstance(idx[0], str):
                if names is None:
                    raise IndexError(f"Cannot index by name — no {axis}names stored in file")
                name_to_idx = {n: i for i, n in enumerate(names)}
                out = []
                for name in idx:
                    if name not in name_to_idx:
                        raise IndexError(f"{axis}name {name!r} not found")
                    out.append(name_to_idx[name])
                return np.array(out)
            return np.asarray(idx)

        if isinstance(idx, np.ndarray):
            if idx.dtype == bool:
                if len(idx) != size:
                    raise IndexError(
                        f"Boolean index length {len(idx)} != {axis} size {size}")
                return np.where(idx)[0]
            return idx

        if isinstance(idx, (int, np.integer)):
            return np.array([int(idx)])

        raise IndexError(f"Unsupported {axis} index type: {type(idx)}")

    # -- Summary statistics --------------------------------------------------

    def nnz_per_col(self) -> np.ndarray:
        """Number of non-zero entries per column (cell). Array of length n_obs."""
        mat = self.read(dtype="int32", normalize=False)
        return np.diff(mat.indptr)

    def nnz_per_row(self) -> np.ndarray:
        """Number of non-zero entries per row (feature). Array of length n_vars."""
        mat = self.read(dtype="int32", normalize=False)
        result = np.zeros(self.shape[0], dtype=np.int64)
        np.add.at(result, mat.indices, 1)
        return result

    def rowsums(self) -> np.ndarray:
        """Row sums of the raw matrix. Array of length n_vars."""
        mat = self.read(normalize=False)
        return np.asarray(mat.sum(axis=1)).ravel()

    def describe(self) -> dict:
        """Summary statistics without full decompression where possible."""
        info = self._get_info()
        m, n, nnz = info["m"], info["n"], info["nnz"]
        try:
            file_size = os.path.getsize(self.path)
        except OSError:
            file_size = 0
        return {
            "path": self.path,
            "shape": (m, n),
            "nnz": nnz,
            "density": nnz / (m * n) if m * n > 0 else 0.0,
            "file_size": file_size,
            "version": info["version"],
            "has_metadata": info.get("has_metadata", False),
            "has_colsums": info.get("has_colsums", False),
            "has_transpose": info.get("has_transpose", False),
            "has_obs_var": info.get("has_obs_var", False),
        }

    # -- File operations -----------------------------------------------------

    def copy(self, dest: str) -> "OnePZFile":
        """Copy the .1pz file to a new path. Returns a new OnePZFile handle."""
        import shutil
        shutil.copy2(self.path, dest)
        return OnePZFile(dest, num_threads=self.num_threads)

    def head(self, n: int = 10) -> ss.csc_matrix:
        """Read the first ``n`` cells (columns)."""
        return self.read(cols=(0, min(n, self.shape[1])))

    def tail(self, n: int = 10) -> ss.csc_matrix:
        """Read the last ``n`` cells (columns)."""
        ncols = self.shape[1]
        return self.read(cols=(max(0, ncols - n), ncols))


def open_1pz(path: str, num_threads: int = 8,
             normalize: bool = False, scale: float = 10000.0) -> OnePZFile:
    """Open a .1pz file for lazy access.

    Parameters
    ----------
    path : str
        Path to a .1pz file.
    num_threads : int
        Number of decompression threads. Default 8.
    normalize : bool
        If True, all reads automatically apply log-normalization:
        ``log1p(x * scale / colsum)``. Equivalent to Seurat's ``LogNormalize``.
    scale : float
        Scaling factor for normalization. Default 10000.

    Returns
    -------
    OnePZFile
        Lazy file handle for metadata inspection and partial reads.
    """
    return OnePZFile(path, num_threads=num_threads,
                     normalize=normalize, scale=scale)


# ============================================================================
# cbind — horizontal concatenation of .1pz files
# ============================================================================

def cbind_1pz(
    *paths: str,
    output: str,
    num_threads: int = 8,
    level: int = 3,
    chunk_cols: int = 1024,
    verify_rownames: bool = True,
) -> dict:
    """Horizontally concatenate (column-bind) multiple .1pz files.

    Reads all input files and writes a single .1pz file with columns
    (cells) concatenated in order. Row counts must match; rownames are
    validated for consistency when present.

    Parameters
    ----------
    *paths : str
        Two or more .1pz file paths.
    output : str
        Output .1pz file path.
    num_threads : int
        Threads for read/write. Default 8.
    level : int
        Zstd compression level. Default 3.
    chunk_cols : int
        Columns per chunk in output. Default 1024.
    verify_rownames : bool
        Raise ValueError if rownames differ across files. Default True.

    Returns
    -------
    dict
        Compression statistics from write.
    """
    import pandas as pd

    if len(paths) < 2:
        raise ValueError("cbind_1pz requires at least 2 input files")

    mats = []
    all_colnames = []
    all_obs = []
    rownames = None
    var_df = None
    all_uns = {}

    for p in paths:
        mat = read_1pz(p, num_threads=num_threads)
        rn = getattr(mat, "rownames", None)
        cn = getattr(mat, "colnames", None)
        obs = getattr(mat, "obs", None)
        var = getattr(mat, "var", None)
        uns = getattr(mat, "uns", None)

        # Validate row compatibility
        if mats and mat.shape[0] != mats[0].shape[0]:
            raise ValueError(
                f"Row count mismatch: {p} has {mat.shape[0]} rows, "
                f"expected {mats[0].shape[0]}")

        if verify_rownames and rn is not None:
            if rownames is None:
                rownames = rn
            elif rn != rownames:
                raise ValueError(f"Rownames mismatch in {p}")
        elif rn is not None and rownames is None:
            rownames = rn

        if cn is not None:
            all_colnames.extend(cn)
        if obs is not None:
            all_obs.append(obs)
        if var is not None and var_df is None:
            var_df = var
        if uns:
            all_uns.update(uns)

        mats.append(mat)

    combined = ss.hstack(mats, format="csc")
    colnames = all_colnames if all_colnames else None
    obs = pd.concat(all_obs, ignore_index=True) if all_obs else None

    return write_1pz(
        output, combined,
        rownames=rownames,
        colnames=colnames,
        obs=obs,
        var=var_df,
        uns=all_uns if all_uns else None,
        num_threads=num_threads,
        level=level,
        chunk_cols=chunk_cols,
    )


# ============================================================================
# rbind — vertical concatenation of .1pz files
# ============================================================================

def rbind_1pz(
    *paths: str,
    output: str,
    num_threads: int = 8,
    level: int = 3,
    chunk_cols: int = 1024,
    verify_colnames: bool = True,
) -> dict:
    """Vertically concatenate (row-bind) multiple .1pz files.

    Reads all input files and writes a single .1pz file with rows
    (features/genes) concatenated in order. Column counts must match;
    colnames are validated for consistency when present.

    Parameters
    ----------
    *paths : str
        Input .1pz file paths (at least 2).
    output : str
        Output .1pz file path.
    num_threads : int
        OpenMP threads for encoding. Default 8.
    level : int
        Zstd compression level. Default 3.
    chunk_cols : int
        Columns per chunk. Default 1024.
    verify_colnames : bool
        When True, raise on colname mismatch between inputs.

    Returns
    -------
    dict with compression stats from write_1pz.
    """
    import pandas as pd

    if len(paths) < 2:
        raise ValueError("rbind_1pz requires at least 2 input files")

    mats = []
    colnames = None
    all_rownames: list[str] = []
    var_frames: list["pd.DataFrame"] = []
    obs_df = None
    all_uns: dict[str, str] = {}

    for p in paths:
        pz = open_1pz(p, num_threads=num_threads)
        mat = pz.read()
        info_dict = pz._get_info()

        cn = pz.colnames
        rn = pz.rownames
        obs = pz.obs
        var = pz.var
        uns = pz.uns

        # Validate column count
        if colnames is not None and mat.shape[1] != len(colnames) if colnames else False:
            raise ValueError(
                f"Column count mismatch: expected {len(colnames)}, got {mat.shape[1]} in {p}")
        if mats and mat.shape[1] != mats[0].shape[1]:
            raise ValueError(
                f"Column count mismatch: expected {mats[0].shape[1]}, got {mat.shape[1]} in {p}")

        # Validate/collect colnames
        if verify_colnames and cn is not None:
            if colnames is None:
                colnames = cn
            elif cn != colnames:
                raise ValueError(f"Colnames mismatch in {p}")
        elif cn is not None and colnames is None:
            colnames = cn

        if rn is not None:
            all_rownames.extend(rn)
        if var is not None:
            var_frames.append(var)
        if obs is not None and obs_df is None:
            obs_df = obs
        if uns:
            all_uns.update(uns)

        mats.append(mat)

    combined = ss.vstack(mats, format="csc")
    rownames = all_rownames if all_rownames else None
    var_df = pd.concat(var_frames, ignore_index=True) if var_frames else None

    return write_1pz(
        output, combined,
        rownames=rownames,
        colnames=colnames,
        obs=obs_df,
        var=var_df,
        uns=all_uns if all_uns else None,
        num_threads=num_threads,
        level=level,
        chunk_cols=chunk_cols,
    )


# ============================================================================
# subset_1pz — write a filtered subset to a new .1pz
# ============================================================================

def subset_1pz(
    path: str,
    output: str,
    *,
    obs_indices: Optional[Union[list[int], np.ndarray, slice]] = None,
    var_indices: Optional[Union[list[int], np.ndarray, slice]] = None,
    obs_names: Optional[list[str]] = None,
    var_names: Optional[list[str]] = None,
    obs_mask: Optional[np.ndarray] = None,
    var_mask: Optional[np.ndarray] = None,
    num_threads: int = 8,
    level: int = 3,
    chunk_cols: int = 1024,
) -> dict:
    """Write a filtered subset of a .1pz file to a new .1pz file.

    Exactly one of (obs_indices, obs_names, obs_mask) should be given for
    cell selection, and one of (var_indices, var_names, var_mask) for
    feature selection. If both axes are omitted, the file is copied.

    Parameters
    ----------
    path : str
        Input .1pz file path.
    output : str
        Output .1pz file path.
    obs_indices : array-like or slice, optional
        Integer indices of cells to keep.
    var_indices : array-like or slice, optional
        Integer indices of features to keep.
    obs_names : list[str], optional
        Cell barcodes to keep (requires stored colnames).
    var_names : list[str], optional
        Gene names to keep (requires stored rownames).
    obs_mask : np.ndarray[bool], optional
        Boolean mask for cells.
    var_mask : np.ndarray[bool], optional
        Boolean mask for features.
    num_threads, level, chunk_cols : int
        Encoding parameters.

    Returns
    -------
    dict with compression stats from write_1pz.
    """
    import pandas as pd

    pz = open_1pz(path, num_threads=num_threads)
    m, n = pz.shape

    # Resolve column (cell) selection
    col_sel = None
    if obs_mask is not None:
        col_sel = np.where(obs_mask)[0]
    elif obs_names is not None:
        cn = pz.colnames
        if cn is None:
            raise ValueError("Cannot subset by obs_names — no colnames stored in file")
        name_map = {name: i for i, name in enumerate(cn)}
        col_sel = np.array([name_map[name] for name in obs_names
                            if name in name_map])
    elif obs_indices is not None:
        if isinstance(obs_indices, slice):
            col_sel = np.arange(*obs_indices.indices(n))
        else:
            col_sel = np.asarray(obs_indices)

    # Resolve row (feature) selection
    row_sel = None
    if var_mask is not None:
        row_sel = np.where(var_mask)[0]
    elif var_names is not None:
        rn = pz.rownames
        if rn is None:
            raise ValueError("Cannot subset by var_names — no rownames stored in file")
        name_map = {name: i for i, name in enumerate(rn)}
        row_sel = np.array([name_map[name] for name in var_names
                            if name in name_map])
    elif var_indices is not None:
        if isinstance(var_indices, slice):
            row_sel = np.arange(*var_indices.indices(m))
        else:
            row_sel = np.asarray(var_indices)

    # Read and subset the matrix
    mat = pz.read()
    if col_sel is not None:
        mat = mat[:, col_sel]
    if row_sel is not None:
        mat = mat[row_sel, :]

    # Subset metadata
    rownames = pz.rownames
    if rownames and row_sel is not None:
        rownames = [rownames[i] for i in row_sel]

    colnames = pz.colnames
    if colnames and col_sel is not None:
        colnames = [colnames[i] for i in col_sel]

    obs_df = pz.obs
    if obs_df is not None and col_sel is not None:
        obs_df = obs_df.iloc[col_sel].reset_index(drop=True)

    var_df = pz.var
    if var_df is not None and row_sel is not None:
        var_df = var_df.iloc[row_sel].reset_index(drop=True)

    return write_1pz(
        output, mat,
        rownames=rownames,
        colnames=colnames,
        obs=obs_df,
        var=var_df,
        uns=pz.uns,
        num_threads=num_threads,
        level=level,
        chunk_cols=chunk_cols,
    )


# ============================================================================
# sample_1pz — random downsample cells to a new .1pz
# ============================================================================

def sample_1pz(
    path: str,
    output: str,
    *,
    n: Optional[int] = None,
    fraction: Optional[float] = None,
    seed: int = 42,
    num_threads: int = 8,
    level: int = 3,
    chunk_cols: int = 1024,
) -> dict:
    """Write a random cell sample from a .1pz file to a new .1pz file.

    Exactly one of ``n`` (absolute count) or ``fraction`` (proportion)
    must be given.

    Parameters
    ----------
    path : str
        Input .1pz file path.
    output : str
        Output .1pz file path.
    n : int, optional
        Number of cells to sample.
    fraction : float, optional
        Fraction of cells to sample (0 < fraction ≤ 1).
    seed : int
        Random seed for reproducibility. Default 42.
    num_threads, level, chunk_cols : int
        Encoding parameters.

    Returns
    -------
    dict with compression stats from write_1pz.
    """
    if (n is None) == (fraction is None):
        raise ValueError("Exactly one of n or fraction must be specified")

    pz = open_1pz(path, num_threads=num_threads)
    total = pz.shape[1]

    if fraction is not None:
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        n = max(1, int(round(total * fraction)))

    n = min(n, total)

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total, size=n, replace=False))

    return subset_1pz(
        path, output,
        obs_indices=indices,
        num_threads=num_threads,
        level=level,
        chunk_cols=chunk_cols,
    )
