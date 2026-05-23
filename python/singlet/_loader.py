# SPDX-License-Identifier: MIT
"""Load datasets from local catalog, Zenodo, or AWS.

Priority order:
  1. Local file path (.1pz, .spz, .h5ad, .zarr)
  2. Local catalog directory (set via SINGLET_CATALOG_DIR or singlet.set_catalog_dir())
  3. Zenodo download (free, cached at ~/.singlet/data/)
  4. AWS streaming (token-priced)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import anndata

_ZENODO_BASE = "https://zenodo.org/records/XXXXXX/files"
_AWS_BASE = "https://data.singlet.bio/v1"


def _cache_dir() -> Path:
    d = Path.home() / ".singlet" / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_gse_path(accession: str) -> Optional[Path]:
    """Resolve a GSE accession to a local .1pz path via the catalog."""
    from singlet._catalog import _get_catalog_dir, _load_catalog

    cat_dir = _get_catalog_dir()
    if cat_dir is None:
        return None

    cat = _load_catalog()
    rows = cat[cat["gse_id"] == accession]
    if rows.empty:
        return None

    row = rows.iloc[0]
    rel_path = row.get("path", "")
    if not rel_path:
        return None

    # path is relative like "pipeline/quant/GSE264667"
    # cat_dir is the catalog root (e.g. /mnt/projects/.../cellarium)
    base = cat_dir.parent  # go up from catalog/ to cellarium/
    gse_dir = base / rel_path

    counts_path = gse_dir / "counts.1pz"
    if counts_path.exists():
        return counts_path

    return None


def download(
    accession: str,
    output_dir: Optional[str | Path] = None,
    force: bool = False,
    source: str = "zenodo",
) -> Path:
    """Download a dataset file.

    Parameters
    ----------
    accession : str
        GEO series accession (e.g. "GSE264667").
    output_dir : path, optional
        Where to save. Defaults to ``~/.singlet/data/``.
    force : bool
        Re-download even if cached.
    source : str
        ``"zenodo"`` (free) or ``"aws"`` (token-priced).

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    import requests

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    if source not in ("zenodo", "aws"):
        raise ValueError(f"source must be 'zenodo' or 'aws', got {source!r}")

    dest_dir = Path(output_dir) if output_dir else _cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{accession}.1pz"

    if dest.exists() and not force:
        return dest

    if source == "aws":
        from singlet._auth import _get_headers

        url = f"{_AWS_BASE}/{accession}"
        headers = _get_headers()
    else:
        url = f"{_ZENODO_BASE}/{accession}.1pz"
        headers = {}

    resp = requests.get(url, stream=True, timeout=60, headers=headers)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        if resp.status_code == 404:
            raise FileNotFoundError(
                f"Dataset '{accession}' not found on {source}. "
                f"Check the accession or use singlet.catalog() to browse available datasets."
            ) from None
        raise RuntimeError(f"Failed to download '{accession}' from {source}: {e}") from e

    # Write to temp file first, then atomic rename (interrupted downloads don't corrupt cache)
    tmp_dest = dest.with_suffix(".1pz.part")
    total = int(resp.headers.get("content-length", 0))
    try:
        if tqdm is not None:
            with (
                open(tmp_dest, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True, desc=accession) as pbar,
            ):
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    pbar.update(len(chunk))
        else:
            with open(tmp_dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
        tmp_dest.rename(dest)
    except BaseException:
        tmp_dest.unlink(missing_ok=True)
        raise

    return dest


def load(
    source: str | Path,
    *,
    genes: Optional[Sequence[str]] = None,
    obs_filter: Optional[dict] = None,
    backend: str = "zenodo",
) -> anndata.AnnData:
    """Load a dataset as AnnData.

    This is the primary entry point. Pass a GEO accession to load from
    local catalog (instant), Zenodo (free download), or AWS (token-priced),
    or a local file path to read directly.

    Parameters
    ----------
    source : str or Path
        GEO accession (e.g. "GSE264667") or path to a local file.
        Supports ``.1pz``, ``.spz``, ``.h5ad``, and ``.zarr``.
    genes : list of str, optional
        Subset to these gene names (column slice).
    obs_filter : dict, optional
        Filter cells by obs columns, e.g. ``{"organism": "Homo sapiens"}``.
    backend : str
        ``"zenodo"`` (default, free) or ``"aws"`` (fast streaming).

    Returns
    -------
    anndata.AnnData
        Count matrix with obs metadata, var gene annotations, and uns study info.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> adata = singlet.load("/path/to/counts.1pz")
    >>> adata = singlet.load("GSE264667", genes=["TP53", "BRCA1"])
    """
    from singlet._io import read_1pz, read_matrix

    if source is None:
        raise TypeError("load() requires a file path or GEO accession, got None")
    if not str(source).strip():
        raise ValueError("load() requires a non-empty file path or GEO accession")

    path = Path(source).expanduser()

    # Detect if source looks like a file path (has directory separators or known extension)
    _is_path = (
        "/" in str(source)
        or "\\" in str(source)
        or path.suffix.lower()
        in (
            ".1pz",
            ".spz",
            ".h5ad",
            ".zarr",
        )
    )

    if path.exists():
        if path.is_dir():
            raise IsADirectoryError(
                f"'{path}' is a directory. Use singlet.load_dir() for pipeline output directories, "
                f"or provide a file path (.1pz, .spz, .h5ad)."
            )
        suffix = path.suffix.lower()
        if suffix == ".h5ad":
            import anndata as ad

            adata = ad.read_h5ad(path)
        elif suffix == ".zarr":
            import anndata as ad

            adata = ad.read_zarr(path)
        elif suffix in (".1pz", ".spz"):
            adata = read_matrix(path)
        else:
            adata = read_matrix(path)
    elif _is_path:
        raise FileNotFoundError(f"File not found: {path}")
    else:
        # Treat as GSE accession
        local_path = _resolve_gse_path(str(source))
        if local_path is not None:
            adata = read_1pz(local_path)
        else:
            local = download(str(source), source=backend)
            adata = read_1pz(local)

    # Gene subset
    if genes is not None:
        gene_mask = adata.var_names.isin(genes)
        n_found = gene_mask.sum()
        if n_found == 0:
            import warnings

            warnings.warn(
                f"None of the {len(genes)} requested genes were found in the dataset. "
                f"First few available: {list(adata.var_names[:5])}",
                UserWarning,
                stacklevel=2,
            )
        elif n_found < len(genes):
            import warnings

            missing = sorted(set(genes) - set(adata.var_names))
            warnings.warn(
                f"{len(missing)} of {len(genes)} requested genes not found: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
                UserWarning,
                stacklevel=2,
            )
        adata = adata[:, gene_mask].copy()

    # Obs filter
    if obs_filter is not None:
        mask = True
        for col, val in obs_filter.items():
            if col not in adata.obs.columns:
                raise KeyError(
                    f"obs_filter column {col!r} not found. Available: {list(adata.obs.columns)}"
                )
            if isinstance(val, list):
                mask = mask & adata.obs[col].isin(val)
            else:
                mask = mask & (adata.obs[col] == val)
        adata = adata[mask].copy()

    return adata


def load_sample(
    gsm_id: str,
    *,
    genes: Optional[Sequence[str]] = None,
) -> anndata.AnnData:
    """Load a single GSM sample using column-range reads.

    Requires a local catalog directory with sample_index.parquet.

    Parameters
    ----------
    gsm_id : str
        GEO sample accession (e.g. "GSM3308814").
    genes : list of str, optional
        Subset to these gene names.

    Returns
    -------
    anndata.AnnData
        Count matrix for just this sample.
    """
    import anndata as ad
    import scipy.sparse as sp

    from singlet._catalog import _get_catalog_dir, _load_sample_index
    from singlet._pz import read_1pz as _native_read

    idx = _load_sample_index()
    rows = idx[idx["gsm_id"] == gsm_id]
    if rows.empty:
        raise KeyError(f"Sample {gsm_id!r} not found in sample index")

    row = rows.iloc[0]
    cat_dir = _get_catalog_dir()
    if cat_dir is None:
        raise RuntimeError(
            "load_sample requires a local catalog. "
            "Set SINGLET_CATALOG_DIR or call singlet.set_catalog_dir()"
        )

    base = cat_dir.parent
    gse_path = row["gse_id"]
    subdir = row.get("species_subdir", "")
    if subdir:
        counts_path = base / "pipeline" / "quant" / gse_path / subdir / "counts.1pz"
    else:
        counts_path = base / "pipeline" / "quant" / gse_path / "counts.1pz"

    col_start = int(row["col_offset"])
    col_end = col_start + int(row["col_count"])

    # TODO: native read_1pz_columns slice — currently we read all then slice
    r = _native_read(str(counts_path))
    full_mat = sp.csc_matrix(
        (r["data"], r["indices"], r["indptr"]),
        shape=(r["m"], r["n"]),
    )
    mat = full_mat[:, col_start:col_end]

    adata = ad.AnnData(X=mat.T)

    if r["rownames"]:
        import pandas as pd

        adata.var_names = pd.Index(r["rownames"])

    adata.obs["gsm_id"] = gsm_id
    adata.obs["gse_id"] = row["gse_id"]
    adata.obs["organism"] = row["organism"]

    if genes is not None:
        gene_mask = adata.var_names.isin(genes)
        adata = adata[:, gene_mask].copy()

    return adata


def load_dir(
    path: str | Path,
    *,
    layer: str = "gene_counts",
    with_qc: bool = True,
    with_doublets: bool = True,
) -> anndata.AnnData:
    """Load a singlet pipeline output directory as AnnData.

    Reads the count matrix (.1pz), attaches gene names from
    gene_expression.tsv, cell barcodes from auto_barcodes.tsv,
    and optionally merges QC metrics and doublet scores.

    Parameters
    ----------
    path : str or Path
        Path to a singlet sample output directory (containing
        gene_counts.1pz, auto_barcodes.tsv, gene_expression.tsv).
    layer : str
        Which .1pz to load: "gene_counts", "exon_counts", "intron_counts",
        or "gene_counts_em". Default "gene_counts".
    with_qc : bool
        Merge cell_qc_metrics.tsv into obs if available.
    with_doublets : bool
        Merge doublet_scores.tsv into obs if available.

    Returns
    -------
    anndata.AnnData
        Sparse count matrix (cells × genes) with metadata.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load_dir("/path/to/quant/GSM3573650")
    >>> adata
    AnnData object with n_obs × n_vars = 75420 × 38606
    """
    import pandas as pd

    from singlet._io import read_1pz

    path = Path(path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Not a directory: {path}")

    # Read count matrix
    pz_file = path / f"{layer}.1pz"
    if not pz_file.exists():
        raise FileNotFoundError(f"Missing {pz_file}")
    adata = read_1pz(pz_file)

    # Attach gene names
    gene_file = path / "gene_expression.tsv"
    if gene_file.exists():
        genes = pd.read_csv(gene_file, sep="\t", usecols=["gene_id", "gene_name"])  # type: ignore[arg-type]
        if len(genes) == adata.n_vars:
            adata.var_names = pd.Index(genes["gene_name"].values)
            adata.var["gene_id"] = genes["gene_id"].values

    # Attach barcodes
    bc_file = path / "auto_barcodes.tsv"
    if bc_file.exists():
        barcodes = pd.read_csv(bc_file, header=None)[0].values
        if len(barcodes) == adata.n_obs:
            adata.obs_names = pd.Index(barcodes)

    # Merge QC metrics
    if with_qc:
        qc_file = path / "cell_qc_metrics.tsv"
        if qc_file.exists():
            qc = pd.read_csv(qc_file, sep="\t", index_col="barcode")
            overlap = adata.obs_names.intersection(qc.index)
            if len(overlap) > 0:
                for col in qc.columns:
                    adata.obs[col] = qc[col].reindex(adata.obs_names).values

    # Merge doublet scores
    if with_doublets:
        dub_file = path / "doublet_scores.tsv"
        if dub_file.exists():
            dub = pd.read_csv(dub_file, sep="\t", index_col="barcode")
            overlap = adata.obs_names.intersection(dub.index)
            if len(overlap) > 0:
                for col in dub.columns:
                    adata.obs[col] = dub[col].reindex(adata.obs_names).values

    # Merge cell cycle scores
    cc_file = path / "cell_cycle_scores.tsv"
    if cc_file.exists():
        cc = pd.read_csv(cc_file, sep="\t", index_col="barcode")
        overlap = adata.obs_names.intersection(cc.index)
        if len(overlap) > 0:
            for col in cc.columns:
                adata.obs[col] = cc[col].reindex(adata.obs_names).values

    # Store ancestry call if available
    ancestry_file = path / "ancestry_call.json"
    if ancestry_file.exists():
        import json as _json

        with open(ancestry_file) as f:
            adata.uns["ancestry"] = _json.load(f)

    # Store sex call if available
    sex_file = path / "sex_call.json"
    if sex_file.exists():
        import json as _json

        with open(sex_file) as f:
            adata.uns["sex_call"] = _json.load(f)

    # Store summary metrics if available
    summary_file = path / "summary.json"
    if summary_file.exists():
        import json as _json

        with open(summary_file) as f:
            adata.uns["summary"] = _json.load(f)

    # Store saturation curve if available
    sat_file = path / "saturation_curve.tsv"
    if sat_file.exists():
        adata.uns["saturation_curve"] = pd.read_csv(sat_file, sep="\t")

    # Store source path
    adata.uns["singlet_dir"] = str(path)

    return adata
