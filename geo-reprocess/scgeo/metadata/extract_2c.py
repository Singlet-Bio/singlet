"""Metadata extraction from additional supplementary file formats (Stage 2c).

Supports:
  - **XLSX / XLS** — Excel files with cell-level annotations
  - **H5Seurat** — HDF5-based Seurat objects (meta.data group)
  - **HDF5** — Generic HDF5 with obs-like groups
  - **H5MU** — MuData format (.obs group)

All extractors return a :class:`pandas.DataFrame` with a ``barcode``
column and every author-provided annotation column.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Recognises a 10x-style barcode
_BARCODE_RE = re.compile(r"[ACGTN]{12,18}(-\d+)?$")

# Column names likely to contain cell barcodes
_BARCODE_COL_NAMES = frozenset({
    "barcode", "barcodes", "cell_barcode", "cell_barcodes",
    "cell_id", "cellid", "cell_name", "cellname",
    "cells", "cb", "index", "cell", "cellbarcode",
    "cell.id", "cell.name", "cell.barcode",
    "sampleid", "sample_id", "sample",
    "observation", "obs_names", "cellnames",
    "unnamed:_0", "unnamed: 0",
})


def extract_metadata_from_xlsx(path: Path) -> pd.DataFrame:
    """Extract cell-level metadata from an Excel file.

    Reads the first sheet (or all sheets if the first is small) and
    detects barcode columns.  Returns all columns.

    Args:
        path: Path to .xlsx or .xls file (optionally .gz compressed).

    Returns:
        DataFrame with a ``barcode`` column and all annotation columns.
        Empty DataFrame on failure.
    """
    try:
        # Handle .gz compression
        actual_path = path
        if str(path).endswith(".gz"):
            import gzip
            import tempfile
            suffix = ".xlsx" if ".xlsx" in str(path).lower() else ".xls"
            tmp = Path(tempfile.mktemp(suffix=suffix))
            with gzip.open(str(path), "rb") as fin, open(str(tmp), "wb") as fout:
                fout.write(fin.read())
            actual_path = tmp
        else:
            tmp = None

        try:
            # Try reading with openpyxl first (xlsx), then xlrd (xls)
            try:
                sheets = pd.read_excel(str(actual_path), sheet_name=None,
                                       engine="openpyxl", nrows=0)
            except Exception:
                sheets = pd.read_excel(str(actual_path), sheet_name=None,
                                       nrows=0)

            # Find the sheet with the most columns (likely the metadata sheet)
            best_sheet = None
            best_ncols = 0
            for name, preview in sheets.items():
                if len(preview.columns) > best_ncols:
                    best_ncols = len(preview.columns)
                    best_sheet = name

            if best_sheet is None:
                logger.warning("XLSX %s: no sheets found", path.name)
                return pd.DataFrame()

            # Read the full sheet
            try:
                df = pd.read_excel(str(actual_path), sheet_name=best_sheet,
                                   engine="openpyxl")
            except Exception:
                df = pd.read_excel(str(actual_path), sheet_name=best_sheet)

            if df.empty or len(df) < 2:
                logger.info("XLSX %s: empty or too few rows", path.name)
                return pd.DataFrame()

            # Skip if too many columns (likely expression matrix)
            if len(df.columns) > 500:
                logger.info("XLSX %s: skipping expression matrix (%d cols)",
                            path.name, len(df.columns))
                return pd.DataFrame()

            # Find barcode column
            bc_col = _find_barcode_col(df)
            if bc_col:
                df["barcode"] = df[bc_col].astype(str)
                if bc_col != "barcode":
                    df = df.drop(columns=[bc_col])
            else:
                # Try index as barcode
                sample_idx = str(df.index[0])
                if _BARCODE_RE.search(sample_idx):
                    df["barcode"] = df.index.astype(str)
                    df = df.reset_index(drop=True)
                elif len(df) > 50:
                    # Use first column if it has unique cell-like values
                    first_col = df.columns[0]
                    if df[first_col].nunique() > 0.8 * len(df):
                        df["barcode"] = df[first_col].astype(str)
                        if first_col != "barcode":
                            df = df.drop(columns=[first_col])
                    else:
                        logger.warning("XLSX %s: no barcode column found", path.name)
                        return pd.DataFrame()
                else:
                    logger.warning("XLSX %s: no barcode column found", path.name)
                    return pd.DataFrame()

            df = df.reset_index(drop=True)
            logger.info("XLSX %s (sheet=%s): %d cells × %d columns",
                        path.name, best_sheet, len(df), len(df.columns) - 1)
            return df

        finally:
            if tmp is not None and tmp.exists():
                tmp.unlink()

    except ImportError as e:
        logger.warning("Cannot read XLSX %s: %s (install openpyxl)", path.name, e)
        return pd.DataFrame()
    except Exception as e:
        logger.warning("Failed to read XLSX %s: %s", path.name, e)
        return pd.DataFrame()


def extract_metadata_from_h5seurat(path: Path) -> pd.DataFrame:
    """Extract cell-level metadata from an H5Seurat file.

    H5Seurat files store Seurat objects in HDF5 format. The metadata
    lives in ``meta.data/`` as a group with datasets for each column.
    Factor variables are stored as ``levels`` + ``values`` arrays.

    Args:
        path: Path to .h5Seurat file.

    Returns:
        DataFrame with a ``barcode`` column and all metadata columns.
        Empty DataFrame on failure.
    """
    try:
        import h5py
        import numpy as np

        with h5py.File(str(path), "r") as f:
            # Find cell names/barcodes
            barcodes = None
            for bc_path in ("cell.names", "meta.data/_index"):
                if bc_path in f:
                    raw = f[bc_path][:]
                    barcodes = [b.decode() if isinstance(b, bytes) else str(b)
                                for b in raw]
                    break

            if barcodes is None:
                logger.warning("H5Seurat %s: no cell names found", path.name)
                return pd.DataFrame()

            data = {"barcode": barcodes}

            # Read meta.data group
            if "meta.data" not in f:
                logger.warning("H5Seurat %s: no meta.data group", path.name)
                return pd.DataFrame(data)

            meta = f["meta.data"]
            for key in meta.keys():
                if key == "_index":
                    continue
                try:
                    item = meta[key]
                    if isinstance(item, h5py.Group):
                        # Factor variable: levels + values
                        if "levels" in item and "values" in item:
                            levels = [v.decode() if isinstance(v, bytes) else str(v)
                                      for v in item["levels"][:]]
                            codes = item["values"][:]
                            vals = [levels[int(c)] if 0 <= int(c) < len(levels) else None
                                    for c in codes]
                            data[key] = vals
                    elif isinstance(item, h5py.Dataset):
                        vals = item[:]
                        if vals.dtype.kind in ("S", "O"):
                            vals = [v.decode() if isinstance(v, bytes) else str(v)
                                    for v in vals]
                        else:
                            vals = list(vals)
                        data[key] = vals
                except Exception as e:
                    logger.debug("H5Seurat %s: skip column %s: %s", path.name, key, e)

            # Also read active.ident if available
            if "active.ident" in f:
                try:
                    ai = f["active.ident"]
                    if isinstance(ai, h5py.Group) and "levels" in ai and "values" in ai:
                        levels = [v.decode() if isinstance(v, bytes) else str(v)
                                  for v in ai["levels"][:]]
                        codes = ai["values"][:]
                        data["active_ident"] = [
                            levels[int(c)] if 0 <= int(c) < len(levels) else None
                            for c in codes
                        ]
                except Exception:
                    pass

            df = pd.DataFrame(data)
            logger.info("H5Seurat %s: %d cells × %d columns",
                        path.name, len(df), len(df.columns) - 1)
            return df

    except ImportError:
        logger.warning("h5py not installed; cannot read H5Seurat %s", path.name)
        return pd.DataFrame()
    except Exception as e:
        logger.warning("Failed to read H5Seurat %s: %s", path.name, e)
        return pd.DataFrame()


def extract_metadata_from_hdf5(path: Path) -> pd.DataFrame:
    """Extract cell-level metadata from a generic HDF5 file.

    Inspects the HDF5 structure for known metadata patterns:
    - ``obs/`` group (AnnData-style)
    - ``meta.data/`` group (H5Seurat-style)
    - ``col_attrs/`` group (Loom-style)

    Args:
        path: Path to .hdf5 or .h5 file.

    Returns:
        DataFrame with a ``barcode`` column and metadata columns.
        Empty DataFrame if no metadata structure is found.
    """
    try:
        import h5py

        with h5py.File(str(path), "r") as f:
            # Check for anndata-style obs
            if "obs" in f:
                from scgeo.metadata.extract import _extract_h5ad_via_h5py
                return _extract_h5ad_via_h5py(path)

            # Check for H5Seurat-style meta.data
            if "meta.data" in f:
                return extract_metadata_from_h5seurat(path)

            # Check for loom-style col_attrs
            if "col_attrs" in f:
                return _extract_loom_style_h5(f, path.name)

            # Check for CellRanger-style matrix/barcodes
            for group_name in ("matrix", f.keys()):
                if isinstance(group_name, str) and group_name in f:
                    grp = f[group_name]
                    if isinstance(grp, h5py.Group) and "barcodes" in grp:
                        barcodes = grp["barcodes"][:]
                        barcodes = [b.decode() if isinstance(b, bytes) else str(b)
                                    for b in barcodes]
                        # 10x h5 only has expression data, no metadata columns
                        logger.info("HDF5 %s: CellRanger format, barcodes only (%d cells)",
                                    path.name, len(barcodes))
                        return pd.DataFrame()

            logger.info("HDF5 %s: no recognized metadata structure. Keys: %s",
                        path.name, list(f.keys())[:10])
            return pd.DataFrame()

    except Exception as e:
        logger.warning("Failed to read HDF5 %s: %s", path.name, e)
        return pd.DataFrame()


def _extract_loom_style_h5(f, fname: str) -> pd.DataFrame:
    """Extract metadata from loom-style col_attrs in an HDF5 file."""
    import h5py

    col_attrs = f["col_attrs"]
    data = {}
    for key in col_attrs.keys():
        try:
            ds = col_attrs[key]
            if isinstance(ds, h5py.Dataset) and len(ds.shape) == 1:
                vals = ds[:]
                if vals.dtype.kind in ("S", "O"):
                    vals = [v.decode() if isinstance(v, bytes) else str(v) for v in vals]
                else:
                    vals = list(vals)
                data[key] = vals
        except Exception:
            pass

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Find barcode column
    bc_col = _find_barcode_col(df)
    if bc_col:
        df["barcode"] = df[bc_col].astype(str)
        if bc_col != "barcode":
            df = df.drop(columns=[bc_col])
    elif "CellID" in df.columns:
        df["barcode"] = df["CellID"].astype(str)
    else:
        df["barcode"] = [str(i) for i in range(len(df))]

    logger.info("HDF5 (loom-style) %s: %d cells × %d columns",
                fname, len(df), len(df.columns) - 1)
    return df


def extract_metadata_from_h5mu(path: Path) -> pd.DataFrame:
    """Extract cell-level metadata from a MuData (.h5mu) file.

    MuData files store multimodal data in HDF5 format with an ``obs``
    group similar to AnnData.

    Args:
        path: Path to .h5mu file.

    Returns:
        DataFrame with a ``barcode`` column and all metadata columns.
        Empty DataFrame on failure.
    """
    try:
        import h5py
        from scgeo.metadata.extract import _extract_h5ad_via_h5py

        with h5py.File(str(path), "r") as f:
            if "obs" in f:
                return _extract_h5ad_via_h5py(path)
            # Check modality-level obs (e.g., mod/rna/obs)
            if "mod" in f:
                for mod_name in f["mod"].keys():
                    mod = f["mod"][mod_name]
                    if isinstance(mod, h5py.Group) and "obs" in mod:
                        logger.info("H5MU %s: reading obs from mod/%s",
                                    path.name, mod_name)
                        return _extract_h5ad_via_h5py(path)

        logger.info("H5MU %s: no obs group found", path.name)
        return pd.DataFrame()

    except Exception as e:
        logger.warning("Failed to read H5MU %s: %s", path.name, e)
        return pd.DataFrame()


def _find_barcode_col(df: pd.DataFrame) -> Optional[str]:
    """Find the column most likely to contain cell barcodes."""
    # Check by name
    for col in df.columns:
        if col.lower().replace(" ", "_").replace("-", "_") in _BARCODE_COL_NAMES:
            return col
    # Check by content (10x barcode pattern)
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(5)
        if all(_BARCODE_RE.search(v) for v in sample if v):
            return col
    return None
