"""Metadata extraction from Python-readable supplementary formats.

Supports:
  - **h5ad** (AnnData) — extracts ``.obs`` DataFrame
  - **CSV / TSV / TXT** — reads tabular files with barcode detection
  - **Loom** — extracts column attributes as metadata

All extractors return a :class:`pandas.DataFrame` with a ``barcode``
column and every author-provided annotation column.  No columns are
filtered — the full author metadata is preserved.
"""

import gzip
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Recognises a 10x-style barcode (12–18 ACGTN bases, optional -N suffix)
_BARCODE_RE = re.compile(r"[ACGTN]{12,18}(-\d+)?$")

# Broader pattern: any plausible cell identifier (includes plate-well IDs,
# sample_barcode combos, SmartSeq-style IDs, etc.)
_CELL_ID_RE = re.compile(
    r"^("                                           # start
    r"[ACGTN]{12,18}(-\d+)?"                        # 10x barcode
    r"|[A-Za-z0-9]+_[ACGTN]{12,18}(-\d+)?"          # prefix_barcode
    r"|[A-H]\d{1,2}"                                  # plate well (A1, H12)
    r"|[A-Za-z0-9._-]+_[A-H]\d{1,2}"                  # sample_well
    r"|[A-Za-z][A-Za-z0-9._-]{4,}"                     # generic cell ID (>=5 chars)
    r")$"
)

# Column names that are likely barcode identifiers
_BARCODE_COL_NAMES = frozenset({
    "barcode", "barcodes", "cell_barcode", "cell_barcodes",
    "cell_id", "cellid", "cell_name", "cellname",
    "cells", "cb", "index", "cell", "cellbarcode",
    "cell.id", "cell.name", "cell.barcode",
    "sampleid", "sample_id", "sample",
    "observation", "obs_names", "cellnames",
    "unnamed:_0", "unnamed: 0",  # pandas default for unnamed index
})

# Column names that indicate cell-level annotation (metadata, not expression)
_ANNOTATION_COL_NAMES = re.compile(
    r"(cell.?type|cluster|cell.?id|barcode|sample|donor|patient"
    r"|tissue|organ|disease|condition|genotype|treatment|batch"
    r"|louvain|leiden|seurat.?cluster|annotation|label|ident"
    r"|age|sex|gender|ethnicity|origin|lineage|subtype"
    r"|umap|tsne|pca|n_genes|n_counts|percent.?mt"
    r"|library|replicate|timepoint|stage|group)",
    re.IGNORECASE,
)

# Gene-name patterns that indicate an expression matrix (columns are genes)
_GENE_COL_RE = re.compile(
    r"^(ENS[A-Z]{3,4}\d{11}|[A-Z][a-z]{2,}\d*|[A-Z]{2,}\d*|" # ENSG..., Gapdh, TP53
    r"MT-|Rpl\d|Rps\d|LOC\d|LINC\d)"  # mitochondrial, ribosomal
)


def extract_metadata_from_h5ad(path: Path) -> pd.DataFrame:
    """Extract all cell-level metadata from an h5ad file.

    Reads the ``.obs`` DataFrame, which contains every annotation the
    author stored (cell type, cluster, donor, UMAP coords, QC metrics,
    etc.).  The cell index becomes the ``barcode`` column.

    Args:
        path: Path to the ``.h5ad`` file.

    Returns:
        DataFrame with a ``barcode`` column and all ``.obs`` columns.
        Empty DataFrame on failure.
    """
    try:
        import anndata as ad
        adata = ad.read_h5ad(str(path), backed="r")
        obs = adata.obs.copy()
        obs["barcode"] = obs.index.astype(str)
        obs = obs.reset_index(drop=True)
        logger.info("h5ad %s: %d cells × %d columns", path.name, len(obs), len(obs.columns) - 1)
        return obs
    except ImportError:
        # Fallback: use h5py directly
        return _extract_h5ad_via_h5py(path)
    except Exception as e:
        logger.warning("Failed to read h5ad %s: %s", path.name, e)
        return pd.DataFrame()


def _extract_h5ad_via_h5py(path: Path) -> pd.DataFrame:
    """Fallback h5ad reader using h5py when anndata is not installed."""
    try:
        import h5py
        import numpy as np

        with h5py.File(str(path), "r") as f:
            if "obs" not in f:
                logger.warning("h5ad %s has no 'obs' group", path.name)
                return pd.DataFrame()

            obs = f["obs"]
            # Find index key
            idx_key = obs.attrs.get("_index", "_index")
            if isinstance(idx_key, bytes):
                idx_key = idx_key.decode()
            if idx_key not in obs:
                # Try common alternatives
                for alt in ["CellID", "cell_id", "index", "_index"]:
                    if alt in obs:
                        idx_key = alt
                        break

            if idx_key not in obs:
                logger.warning("h5ad %s: cannot find index in obs", path.name)
                return pd.DataFrame()

            # Read barcodes
            raw_barcodes = obs[idx_key][:]
            barcodes = [
                b.decode() if isinstance(b, bytes) else str(b) for b in raw_barcodes
            ]

            # Read all other datasets in obs/
            data = {"barcode": barcodes}
            for key in obs.keys():
                if key == idx_key or key == "__categories":
                    continue
                try:
                    ds = obs[key]
                    if hasattr(ds, "shape"):
                        vals = ds[:]
                        # Decode bytes
                        if vals.dtype.kind == "S" or vals.dtype.kind == "O":
                            vals = [
                                v.decode() if isinstance(v, bytes) else v
                                for v in vals
                            ]
                        # Handle categorical encoded as integer codes + categories
                        if "__categories" in obs and key in obs["__categories"]:
                            cats = obs["__categories"][key][:]
                            cats = [c.decode() if isinstance(c, bytes) else c for c in cats]
                            vals = [cats[int(v)] if 0 <= int(v) < len(cats) else None for v in vals]
                        data[key] = vals
                except Exception:
                    pass

            df = pd.DataFrame(data)
            logger.info(
                "h5ad (h5py) %s: %d cells × %d columns",
                path.name, len(df), len(df.columns) - 1,
            )
            return df

    except Exception as e:
        logger.warning("Failed to read h5ad via h5py %s: %s", path.name, e)
        return pd.DataFrame()


def extract_metadata_from_tabular(path: Path) -> pd.DataFrame:
    """Extract all metadata from a tabular (CSV/TSV/TXT) file.

    Detects whether barcodes are in the row index or in a named column.
    Supports gzipped files.  Returns **all** columns — no filtering.

    Args:
        path: Path to CSV, TSV, or TXT file (optionally gzipped).

    Returns:
        DataFrame with a ``barcode`` column and all author columns.
        Empty DataFrame on failure.
    """
    try:
        fn = path.name.lower()
        compression = "gzip" if fn.endswith(".gz") else None
        fn_base = fn.rstrip(".gz")

        if fn_base.endswith(".tsv") or fn_base.endswith(".txt"):
            sep = "\t"
        else:
            sep = ","

        # Peek at the first few rows to detect structure
        preview = pd.read_csv(str(path), sep=sep, compression=compression,
                              index_col=0, nrows=5)

        # Skip expression matrices: >500 columns OR >50 columns that look
        # like gene names and no annotation-like columns
        if len(preview.columns) > 500:
            logger.info(
                "Tabular %s: skipping expression matrix (%d columns)",
                path.name, len(preview.columns),
            )
            return pd.DataFrame()
        if len(preview.columns) > 50:
            ann_hits = sum(1 for c in preview.columns if _ANNOTATION_COL_NAMES.search(str(c)))
            gene_hits = sum(1 for c in preview.columns if _GENE_COL_RE.match(str(c)))
            if gene_hits > len(preview.columns) * 0.5 and ann_hits < 3:
                logger.info(
                    "Tabular %s: skipping likely expression matrix "
                    "(%d cols, %d gene-like, %d annotation-like)",
                    path.name, len(preview.columns), gene_hits, ann_hits,
                )
                return pd.DataFrame()

        # Check if index looks like barcodes (10x or other cell IDs)
        sample_idx = str(preview.index[0])
        idx_is_barcode = (
            _BARCODE_RE.search(sample_idx)
            or _CELL_ID_RE.match(sample_idx)
        )
        if idx_is_barcode:
            # Barcodes are in the index
            df = pd.read_csv(str(path), sep=sep, compression=compression, index_col=0)
            df["barcode"] = df.index.astype(str)
            df = df.reset_index(drop=True)
            logger.info(
                "Tabular %s: %d cells × %d columns (index=barcode)",
                path.name, len(df), len(df.columns) - 1,
            )
            return df

        # Check if any column is a barcode column (by name)
        preview_flat = pd.read_csv(str(path), sep=sep, compression=compression, nrows=5)
        bc_col = _find_barcode_column(preview_flat)
        if bc_col:
            df = pd.read_csv(str(path), sep=sep, compression=compression)
            df["barcode"] = df[bc_col].astype(str)
            if bc_col != "barcode":
                df = df.drop(columns=[bc_col])
            logger.info(
                "Tabular %s: %d cells × %d columns (col=%s)",
                path.name, len(df), len(df.columns) - 1, bc_col,
            )
            return df

        # Check if *any* column values look like 10x barcodes
        for col in preview_flat.columns:
            sample_val = str(preview_flat[col].iloc[0])
            if _BARCODE_RE.search(sample_val):
                df = pd.read_csv(str(path), sep=sep, compression=compression)
                df["barcode"] = df[col].astype(str)
                if col != "barcode":
                    df = df.drop(columns=[col])
                logger.info(
                    "Tabular %s: %d cells × %d columns (detected col=%s)",
                    path.name, len(df), len(df.columns) - 1, col,
                )
                return df

        # Check if the 'Unnamed: 0' or first column has cell-like IDs
        first_col = preview_flat.columns[0]
        first_vals = preview_flat[first_col].astype(str).tolist()
        if all(_CELL_ID_RE.match(v) for v in first_vals if v and v != "nan"):
            df = pd.read_csv(str(path), sep=sep, compression=compression)
            df["barcode"] = df[first_col].astype(str)
            if first_col != "barcode":
                df = df.drop(columns=[first_col])
            logger.info(
                "Tabular %s: %d cells × %d columns (first-col=%s)",
                path.name, len(df), len(df.columns) - 1, first_col,
            )
            return df

        # If nrows > 100 and < 5000 columns, assume it's metadata even without
        # recognisable barcodes — use the index as an identifier
        full_rows = pd.read_csv(
            str(path), sep=sep, compression=compression, index_col=0, nrows=0,
        )
        # Re-read to count rows cheaply
        try:
            row_estimate = sum(1 for _ in open(str(path), "rb")) - 1 if compression is None else len(preview) + 1
        except Exception:
            row_estimate = 0

        if row_estimate > 100 and len(full_rows.columns) < 500:
            df = pd.read_csv(str(path), sep=sep, compression=compression, index_col=0)
            if len(df) > 100:
                df["barcode"] = df.index.astype(str)
                df = df.reset_index(drop=True)
                logger.info(
                    "Tabular %s: %d cells × %d columns (index-as-id, no barcode pattern)",
                    path.name, len(df), len(df.columns) - 1,
                )
                return df

        logger.warning(
            "No barcodes found in %s; columns: %s",
            path.name, list(preview_flat.columns)[:10],
        )
        return pd.DataFrame()

    except Exception as e:
        logger.warning("Failed to read tabular %s: %s", path.name, e)
        return pd.DataFrame()


def extract_metadata_from_loom(path: Path) -> pd.DataFrame:
    """Extract all cell-level metadata from a Loom file.

    Reads column attributes (cell-level metadata) from the loom
    file's ``col_attrs``.

    Args:
        path: Path to the ``.loom`` file.

    Returns:
        DataFrame with a ``barcode`` column and all column attributes.
        Empty DataFrame on failure.
    """
    try:
        import loompy

        with loompy.connect(str(path), mode="r") as ds:
            data = {}
            for key in ds.ca.keys():
                data[key] = ds.ca[key]

            df = pd.DataFrame(data)

            # Find barcode column
            bc_col = _find_barcode_column(df)
            if bc_col:
                df["barcode"] = df[bc_col].astype(str)
                if bc_col != "barcode":
                    df = df.drop(columns=[bc_col])
            elif "CellID" in df.columns:
                df["barcode"] = df["CellID"].astype(str)
            else:
                # Use index as barcode
                df["barcode"] = [str(i) for i in range(len(df))]
                logger.warning("Loom %s: no barcode column found, using integer index", path.name)

            logger.info("Loom %s: %d cells × %d columns", path.name, len(df), len(df.columns) - 1)
            return df

    except ImportError:
        logger.warning("loompy not installed; cannot read %s", path.name)
        return pd.DataFrame()
    except Exception as e:
        logger.warning("Failed to read loom %s: %s", path.name, e)
        return pd.DataFrame()


def _find_barcode_column(df: pd.DataFrame) -> Optional[str]:
    """Find the column most likely to contain cell barcodes."""
    for col in df.columns:
        if col.lower().replace(" ", "_").replace("-", "_") in _BARCODE_COL_NAMES:
            return col
    return None


def peek_tabular_content(path: Path) -> dict:
    """Peek at a tabular file to classify its content without full read.

    Returns a dict with:
      - ``is_metadata``: True if columns look like annotations
      - ``is_expression``: True if columns look like gene names
      - ``n_cols``: number of columns
      - ``n_rows_est``: estimated number of rows (from preview)
      - ``annotation_cols``: list of annotation-like column names
      - ``has_barcode``: True if a barcode column/index was detected

    This is cheaper than a full extraction and helps prioritize which
    files to download and try first.
    """
    result = {
        "is_metadata": False, "is_expression": False,
        "n_cols": 0, "n_rows_est": 0,
        "annotation_cols": [], "has_barcode": False,
    }
    try:
        fn = path.name.lower()
        compression = "gzip" if fn.endswith(".gz") else None
        fn_base = fn.rstrip(".gz")
        sep = "\t" if fn_base.endswith((".tsv", ".txt")) else ","

        preview = pd.read_csv(
            str(path), sep=sep, compression=compression, nrows=10,
        )
        result["n_cols"] = len(preview.columns)
        result["n_rows_est"] = len(preview)

        # Check for barcode column
        bc_col = _find_barcode_column(preview)
        if bc_col:
            result["has_barcode"] = True
        else:
            # Check first value of each column for barcode pattern
            for col in preview.columns:
                val = str(preview[col].iloc[0])
                if _BARCODE_RE.search(val):
                    result["has_barcode"] = True
                    break

        # Classify columns
        ann_cols = []
        gene_count = 0
        for col in preview.columns:
            cs = str(col)
            if _ANNOTATION_COL_NAMES.search(cs):
                ann_cols.append(cs)
            if _GENE_COL_RE.match(cs):
                gene_count += 1

        result["annotation_cols"] = ann_cols

        # Decision logic
        n = len(preview.columns)
        if n > 50 and gene_count > n * 0.5 and len(ann_cols) < 3:
            result["is_expression"] = True
        elif len(ann_cols) >= 2 or result["has_barcode"]:
            result["is_metadata"] = True
        elif n <= 50:
            result["is_metadata"] = True  # Small files are likely metadata

    except Exception as e:
        logger.debug("peek_tabular_content %s: %s", path.name, e)

    return result


# ───────────────────────────────────────── tar archive extraction ──

# File extensions inside tar archives that are worth extracting for metadata
_TAR_EXTRACTABLE_EXT = {
    ".h5ad", ".csv", ".csv.gz", ".tsv", ".tsv.gz",
    ".txt", ".txt.gz", ".tab", ".tab.gz", ".loom",
}

# Files inside tar that are definitely NOT metadata
_TAR_SKIP_RE = re.compile(
    r"(matrix\.mtx|genes\.tsv|barcodes\.tsv|features\.tsv"
    r"|\.bam|\.bai|\.bed|\.bw|\.bigwig|\.gtf|\.gff|\.h5$"
    r"|\.fasta|\.fa$|\.fastq|\.sam$|\.cram$)",
    re.IGNORECASE,
)


def extract_metadata_from_tar(
    path: Path,
    max_members: int = 50,
    max_extract_mb: int = 500,
) -> list:
    """Extract metadata-candidate files from a tar archive.

    Scans tar members for files that look like they could contain
    cell-level metadata (h5ad, csv, tsv, txt, loom).  Skips known
    count-matrix files (matrix.mtx, barcodes.tsv, .h5, etc.).

    Args:
        path: Path to the ``.tar`` or ``.tar.gz`` file.
        max_members: Maximum number of members to consider.
        max_extract_mb: Skip individual members larger than this (MB).

    Returns:
        List of ``(member_filename, format_key, DataFrame)`` tuples
        for each successfully extracted metadata file inside the tar.
        Returns an empty list on failure.
    """
    import tarfile
    import tempfile

    results = []
    fn = path.name.lower()
    mode = "r:gz" if fn.endswith(".tar.gz") or fn.endswith(".tgz") else "r:"

    try:
        with tarfile.open(str(path), mode) as tf:
            # Security: filter out path traversal attempts
            members = []
            for m in tf:
                if m.name.startswith("/") or ".." in m.name:
                    continue
                if not m.isfile():
                    continue
                members.append(m)
                if len(members) >= max_members:
                    break

            # Classify members and pick metadata candidates
            candidates = []
            for m in members:
                member_fn = m.name.split("/")[-1].lower()

                # Skip known non-metadata
                if _TAR_SKIP_RE.search(member_fn):
                    continue

                # Skip too-large members
                if m.size > max_extract_mb * 1024 * 1024:
                    continue

                # Check if extension is extractable
                has_ext = False
                for ext in _TAR_EXTRACTABLE_EXT:
                    if member_fn.endswith(ext):
                        has_ext = True
                        break
                if not has_ext:
                    continue

                # Classify by extension
                if member_fn.endswith(".h5ad"):
                    fmt = "h5ad"
                elif member_fn.endswith(".loom"):
                    fmt = "loom"
                else:
                    fmt = "tabular"

                is_meta_like = bool(re.search(
                    r"(meta|annot|obs|cell.?type|cluster|label|barcode|pheno|coldata)",
                    member_fn, re.IGNORECASE,
                ))
                candidates.append((m, fmt, is_meta_like))

            # Sort: metadata-like first
            candidates.sort(key=lambda x: not x[2])

            logger.info(
                "Tar %s: %d members, %d metadata candidates",
                path.name, len(members), len(candidates),
            )

            # Extract and process candidates (cap at 10 successful)
            extractors = {
                "h5ad": extract_metadata_from_h5ad,
                "tabular": extract_metadata_from_tabular,
                "loom": extract_metadata_from_loom,
            }

            with tempfile.TemporaryDirectory(prefix="tar_meta_") as tmpdir:
                tmpdir = Path(tmpdir)
                for m, fmt, _ in candidates:
                    member_fn = m.name.split("/")[-1]
                    dest = tmpdir / member_fn

                    try:
                        # Extract individual member safely
                        with tf.extractfile(m) as src, open(dest, "wb") as dst:
                            while True:
                                chunk = src.read(65536)
                                if not chunk:
                                    break
                                dst.write(chunk)
                    except Exception as e:
                        logger.debug("Tar extract failed for %s: %s", member_fn, e)
                        continue

                    extractor = extractors.get(fmt)
                    if extractor is None:
                        continue

                    try:
                        obs = extractor(dest)
                        if not obs.empty and "barcode" in obs.columns and len(obs) > 0:
                            results.append((member_fn, fmt, obs))
                            logger.info(
                                "Tar %s: extracted %d cells from %s (%s)",
                                path.name, len(obs), member_fn, fmt,
                            )
                    except Exception as e:
                        logger.debug("Tar extract parse failed %s: %s", member_fn, e)
                        continue

                    if len(results) >= 10:
                        break

    except Exception as e:
        logger.warning("Failed to read tar %s: %s", path.name, e)

    return results


def extract_metadata_from_expression(path: Path) -> pd.DataFrame:
    """Salvage annotation columns from an expression matrix file.

    When a tabular file is primarily an expression matrix (gene names as
    columns), it may still have a few annotation columns (cell_type,
    cluster, etc.) alongside the expression data.  This function reads
    only those annotation columns plus the barcode/index, discarding the
    gene expression columns entirely.

    Args:
        path: Path to a tabular file that was identified as an expression
            matrix by ``extract_metadata_from_tabular``.

    Returns:
        DataFrame with ``barcode`` and any annotation columns found.
        Empty DataFrame if no annotation columns exist.
    """
    try:
        fn = path.name.lower()
        compression = "gzip" if fn.endswith(".gz") else None
        fn_base = fn.rstrip(".gz")
        sep = "\t" if fn_base.endswith((".tsv", ".txt")) else ","

        # Read just the header + a few rows
        preview = pd.read_csv(
            str(path), sep=sep, compression=compression,
            index_col=0, nrows=5,
        )

        if len(preview.columns) < 10:
            return pd.DataFrame()  # Too few columns to be an expression matrix

        # Find annotation columns
        ann_cols = [
            c for c in preview.columns if _ANNOTATION_COL_NAMES.search(str(c))
        ]
        if not ann_cols:
            return pd.DataFrame()

        # Check if index looks like barcodes
        sample_idx = str(preview.index[0])
        idx_is_barcode = (
            _BARCODE_RE.search(sample_idx)
            or _CELL_ID_RE.match(sample_idx)
        )

        if not idx_is_barcode:
            # Try to find a barcode column among the annotation columns
            preview_flat = pd.read_csv(
                str(path), sep=sep, compression=compression, nrows=5,
            )
            bc_col = _find_barcode_column(preview_flat)
            if not bc_col:
                return pd.DataFrame()
            # Read only barcode + annotation columns
            usecols = [bc_col] + [c for c in ann_cols if c != bc_col]
            df = pd.read_csv(
                str(path), sep=sep, compression=compression, usecols=usecols,
            )
            df["barcode"] = df[bc_col].astype(str)
            if bc_col != "barcode":
                df = df.drop(columns=[bc_col])
        else:
            # Read only index + annotation columns
            df = pd.read_csv(
                str(path), sep=sep, compression=compression,
                index_col=0, usecols=[0] + [
                    i + 1 for i, c in enumerate(preview.columns) if c in ann_cols
                ],
            )
            df["barcode"] = df.index.astype(str)
            df = df.reset_index(drop=True)

        if "barcode" in df.columns and len(df) > 0:
            meta_cols = [c for c in df.columns if c != "barcode"]
            logger.info(
                "Expression-salvage %s: %d cells × %d annotation columns (%s)",
                path.name, len(df), len(meta_cols),
                ", ".join(meta_cols[:5]),
            )
            return df

        return pd.DataFrame()

    except Exception as e:
        logger.debug("Expression salvage failed %s: %s", path.name, e)
        return pd.DataFrame()
