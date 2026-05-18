# SPDX-License-Identifier: MIT
""".1pz file I/O — single-matrix AnnData round-trip.

For multi-block canonical v2 sample outputs, see :mod:`singlet.pz_v2`
and :class:`singlet.io.SingletSample`.

This module is the thin AnnData adapter on top of the singlepress C++
codec. It handles only legacy single-block ``.1pz`` files (magic
``TP1Z``); the canonical v2 format (magic ``1PZ02``) is read via
:func:`singlet.pz_v2.read_pz_v2`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import anndata


_TP1_MAGIC = b"\x54\x50\x31\x5a"  # "TP1Z" — legacy single-block .1pz
_PZ_V2_MAGIC_PREFIX = b"1PZ02"     # canonical v2 multi-block .1pz


def _detect_format(path: str | Path) -> str:
    """Detect file format from header magic bytes.

    Returns ``"1pz"`` (legacy single-block) or ``"1pz_v2"`` (canonical
    multi-block). Raises :class:`ValueError` for anything else.
    """
    with open(path, "rb") as f:
        header = f.read(8)
    if len(header) < 8:
        raise ValueError(f"File too small to be a valid matrix file: {path}")
    if header[:5] == _PZ_V2_MAGIC_PREFIX:
        return "1pz_v2"
    if header[:4] == _TP1_MAGIC:
        return "1pz"
    raise ValueError(f"Unknown file format (bad magic): {path}")


# ============================================================================
# .1pz format (legacy single-block via singlepress)
# ============================================================================


def read_1pz(path: str | Path) -> "anndata.AnnData":
    """Read a single-block ``.1pz`` file into AnnData.

    For canonical v2 multi-block sample outputs use
    :class:`singlet.io.SingletSample`.

    Returns
    -------
    anndata.AnnData
        Sparse count matrix (CSR) with cells in ``obs`` and genes in
        ``var``. (``.1pz`` stores genes × cells; this transposes.)
        Stored ``rownames``, ``colnames``, ``colsums``, embedded
        obs/var DataFrames, and ``uns`` are merged into the returned
        AnnData.
    """
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    if path is None:
        raise TypeError("read_1pz() requires a file path, got None")
    if not str(path).strip():
        raise ValueError("read_1pz() requires a non-empty file path")
    path_str = str(Path(path).expanduser())
    if not Path(path_str).exists():
        raise FileNotFoundError(f"File not found: {path_str}")

    try:
        import singlepress

        mat = singlepress.read_1pz(path_str, num_threads=8)
    except (ImportError, AttributeError):
        from singlepress._pz_codec import pz_read

        result = pz_read(path_str, 8)
        m, n = result["m"], result["n"]
        mat = sp.csc_matrix(
            (result["values"], result["indices"], result["indptr"]),
            shape=(m, n),
        )

    adata = ad.AnnData(X=mat.T)

    if hasattr(mat, "rownames") and mat.rownames:
        import pandas as pd

        adata.var_names = pd.Index(mat.rownames)
    if hasattr(mat, "colnames") and mat.colnames:
        import pandas as pd

        adata.obs_names = pd.Index(mat.colnames)
    if hasattr(mat, "colsums") and mat.colsums is not None:
        adata.obs["total_counts"] = mat.colsums.astype(np.float64)

    if hasattr(mat, "obs") and mat.obs is not None:
        obs_df = mat.obs
        obs_df.index = adata.obs_names
        for col in obs_df.columns:
            adata.obs[col] = obs_df[col].values

    if hasattr(mat, "var") and mat.var is not None:
        var_df = mat.var
        var_df.index = adata.var_names
        for col in var_df.columns:
            adata.var[col] = var_df[col].values

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
    """Write AnnData to a single-block ``.1pz`` file.

    For canonical multi-block sample outputs, use
    :func:`singlet.pz_v2.write_pz_v2`.
    """
    import scipy.sparse as sp
    import singlepress

    if adata is None:
        raise TypeError("write_1pz() requires an AnnData object, got None")
    if not hasattr(adata, "X") or not hasattr(adata, "obs"):
        raise TypeError(
            f"write_1pz() requires an AnnData object, got {type(adata).__name__}"
        )

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    mat = adata.layers[layer] if layer else adata.X
    if not sp.issparse(mat):
        mat = sp.csc_matrix(mat)
    mat = mat.T.tocsc()  # AnnData is cells × genes → genes × cells on disk

    rownames = list(adata.var_names) if len(adata.var_names) > 0 else None
    colnames = list(adata.obs_names) if len(adata.obs_names) > 0 else None
    obs_df = adata.obs if include_obs and len(adata.obs.columns) > 0 else None
    var_df = adata.var if include_var and len(adata.var.columns) > 0 else None

    uns_dict = None
    if include_uns and adata.uns:
        uns_dict = {
            k: str(v)
            for k, v in adata.uns.items()
            if isinstance(v, (str, int, float, bool))
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
    """Read a single-block ``.1pz`` header without decompressing payloads."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    import singlepress

    return singlepress.info_1pz(str(path))


def read_matrix(path: str | Path, **kwargs):
    """Auto-detect ``.1pz`` format variant and return an AnnData.

    Routes single-block ``.1pz`` files through :func:`read_1pz`. For
    canonical multi-block files (magic ``1PZ02``) callers should use
    :class:`singlet.io.SingletSample`; this function raises a
    :class:`ValueError` so it cannot be silently misused.
    """
    fmt = _detect_format(path)
    if fmt == "1pz":
        return read_1pz(path)
    raise ValueError(
        f"{path} is a canonical multi-block .1pz file (magic 1PZ02); "
        "open it with singlet.io.SingletSample or singlet.pz_v2.read_pz_v2"
    )


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

    adata = ad.AnnData(X=mat.T)
    if hasattr(mat, "rownames") and mat.rownames:
        adata.var_names = pd.Index(mat.rownames)

    feat_path = gse_dir / "kraken2_features.parquet"
    if feat_path.exists():
        feat_df = pd.read_parquet(feat_path)
        feat_df.index = adata.var_names
        for col in feat_df.columns:
            adata.var[col] = feat_df[col].values

    if hasattr(mat, "uns") and mat.uns:
        adata.uns.update(mat.uns)

    return adata
