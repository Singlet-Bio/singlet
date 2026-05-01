"""Browse the SingletDB catalog.

Loads catalog_v1.parquet and sample_index.parquet from either:
  1. A local catalog directory (set via SINGLET_CATALOG_DIR or singlet.set_catalog_dir())
  2. Zenodo download (cached at ~/.singlet/cache/)

Catalog v1.0 schema:
  catalog_v1.parquet:  gse_id, organism, n_samples, n_cells, n_genes, reference,
                       protocol, has_kraken2, has_author_meta, license, path, catalog_version
  sample_index.parquet: gsm_id, gse_id, organism, n_cells, pipeline_version,
                        species_subdir, col_offset, col_count
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

_CATALOG_URL = "https://raw.githubusercontent.com/Singlet-Bio/singlet/main/python/singlet/data/catalog_v1.parquet"
_SAMPLE_INDEX_URL = "https://raw.githubusercontent.com/Singlet-Bio/singlet/main/python/singlet/data/sample_index.parquet"

_CATALOG_CACHE: Optional[pd.DataFrame] = None
_SAMPLE_INDEX_CACHE: Optional[pd.DataFrame] = None
_CATALOG_DIR: Optional[Path] = None


def set_catalog_dir(path: str | Path) -> None:
    """Set the local catalog directory containing catalog_v1.parquet and sample_index.parquet."""
    global _CATALOG_DIR, _CATALOG_CACHE, _SAMPLE_INDEX_CACHE
    _CATALOG_DIR = Path(path)
    _CATALOG_CACHE = None
    _SAMPLE_INDEX_CACHE = None


def _get_catalog_dir() -> Optional[Path]:
    if _CATALOG_DIR is not None:
        return _CATALOG_DIR
    env = os.environ.get("SINGLET_CATALOG_DIR")
    if env:
        return Path(env)
    return None


def _download_parquet(url: str, cache_path: Path) -> pd.DataFrame:
    import requests
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return pd.read_parquet(cache_path)


def _load_catalog() -> pd.DataFrame:
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE

    cat_dir = _get_catalog_dir()
    if cat_dir is not None:
        path = cat_dir / "catalog_v1.parquet"
        if path.exists():
            _CATALOG_CACHE = pd.read_parquet(path)
            return _CATALOG_CACHE

    cache_path = Path.home() / ".singlet" / "cache" / "catalog_v1.parquet"
    if cache_path.exists():
        _CATALOG_CACHE = pd.read_parquet(cache_path)
        return _CATALOG_CACHE

    try:
        _CATALOG_CACHE = _download_parquet(_CATALOG_URL, cache_path)
    except Exception:
        bundled = Path(__file__).parent / "data" / "catalog_v1.parquet"
        if bundled.exists():
            _CATALOG_CACHE = pd.read_parquet(bundled)
        else:
            raise RuntimeError(
                "Could not load catalog. Set SINGLET_CATALOG_DIR or check internet."
            )
    return _CATALOG_CACHE


def _load_sample_index() -> pd.DataFrame:
    global _SAMPLE_INDEX_CACHE
    if _SAMPLE_INDEX_CACHE is not None:
        return _SAMPLE_INDEX_CACHE

    cat_dir = _get_catalog_dir()
    if cat_dir is not None:
        path = cat_dir / "sample_index.parquet"
        if path.exists():
            _SAMPLE_INDEX_CACHE = pd.read_parquet(path)
            return _SAMPLE_INDEX_CACHE

    cache_path = Path.home() / ".singlet" / "cache" / "sample_index.parquet"
    if cache_path.exists():
        _SAMPLE_INDEX_CACHE = pd.read_parquet(cache_path)
        return _SAMPLE_INDEX_CACHE

    try:
        _SAMPLE_INDEX_CACHE = _download_parquet(_SAMPLE_INDEX_URL, cache_path)
    except Exception:
        bundled = Path(__file__).parent / "data" / "sample_index.parquet"
        if bundled.exists():
            _SAMPLE_INDEX_CACHE = pd.read_parquet(bundled)
        else:
            raise RuntimeError(
                "Could not load sample index. Set SINGLET_CATALOG_DIR or check internet."
            )
    return _SAMPLE_INDEX_CACHE


def refresh() -> None:
    """Re-download the latest catalog and sample index from GitHub.

    Clears the in-memory cache and overwrites the local cache files.
    """
    global _CATALOG_CACHE, _SAMPLE_INDEX_CACHE
    _CATALOG_CACHE = None
    _SAMPLE_INDEX_CACHE = None
    cache_dir = Path.home() / ".singlet" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _download_parquet(_CATALOG_URL, cache_dir / "catalog_v1.parquet")
    _download_parquet(_SAMPLE_INDEX_URL, cache_dir / "sample_index.parquet")
    # Reload
    _load_catalog()
    _load_sample_index()
    print(f"Updated: {len(_SAMPLE_INDEX_CACHE)} samples, {len(_CATALOG_CACHE)} series")


def catalog(search: Optional[str] = None) -> pd.DataFrame:
    """Return the full dataset catalog as a DataFrame.

    Parameters
    ----------
    search : str, optional
        Filter catalog rows where any text column matches this substring.

    Returns
    -------
    pd.DataFrame
        One row per GSE. Columns: gse_id, organism, n_samples, n_cells,
        n_genes, reference, protocol, has_kraken2, license, path, etc.
    """
    df = _load_catalog()
    if search is not None:
        text_cols = df.select_dtypes(include="object").columns
        mask = df[text_cols].apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        return df[mask].reset_index(drop=True)
    return df


def sample_index(gse_id: Optional[str] = None) -> pd.DataFrame:
    """Return the sample-level index with column offsets.

    Parameters
    ----------
    gse_id : str, optional
        Filter to samples within this GSE.

    Returns
    -------
    pd.DataFrame
        One row per GSM. Columns: gsm_id, gse_id, organism, n_cells,
        pipeline_version, species_subdir, col_offset, col_count.
    """
    df = _load_sample_index()
    if gse_id is not None:
        df = df[df["gse_id"] == gse_id]
    return df.reset_index(drop=True)


def info(accession: str) -> dict:
    """Return metadata for a single dataset.

    Parameters
    ----------
    accession : str
        GEO series accession (e.g. "GSE136831").

    Returns
    -------
    dict
        Dataset metadata including organism, n_cells, protocol, etc.
    """
    df = _load_catalog()
    rows = df[df["gse_id"] == accession]
    if rows.empty:
        raise KeyError(f"Accession {accession!r} not found in catalog")
    return rows.iloc[0].to_dict()


def species() -> list:
    """Return all species with processed data."""
    df = _load_catalog()
    col = "organisms" if "organisms" in df.columns else "organism"
    all_species = set()
    for org in df[col].dropna():
        for s in str(org).replace(";", ",").split(","):
            s = s.strip()
            if s and s != "unknown":
                all_species.add(s)
    return sorted(all_species)


def tissues() -> pd.DataFrame:
    """Return tissue/source breakdown across SUCCESS samples.

    Returns a DataFrame with columns: source, count, sorted by count descending.
    """
    df = _load_sample_index()
    df = df[df["status"] == "SUCCESS"]
    if "source" not in df.columns:
        return pd.DataFrame(columns=["source", "count"])
    counts = df["source"].dropna().value_counts().reset_index()
    counts.columns = ["source", "count"]
    return counts


def protocols() -> pd.DataFrame:
    """Return protocol breakdown across SUCCESS samples.

    Returns a DataFrame with columns: protocol, count, sorted by count descending.
    """
    df = _load_sample_index()
    df = df[df["status"] == "SUCCESS"]
    if "protocol" not in df.columns:
        return pd.DataFrame(columns=["protocol", "count"])
    valid = df["protocol"].dropna()
    valid = valid[valid.str.strip() != ""]
    counts = valid.value_counts().reset_index()
    counts.columns = ["protocol", "count"]
    return counts


def datasets(
    organism: Optional[str] = None,
    protocol: Optional[str] = None,
    min_cells: Optional[int] = None,
    has_kraken2: Optional[bool] = None,
) -> pd.DataFrame:
    """Filter catalog by organism, protocol, cell count, or kraken2 availability.

    Parameters
    ----------
    organism : str, optional
        Filter by organism (substring match, e.g. "Homo sapiens").
    protocol : str, optional
        Filter by protocol (e.g. "10xv3").
    min_cells : int, optional
        Minimum cell count.
    has_kraken2 : bool, optional
        Filter by kraken2 availability.
    """
    df = _load_catalog()
    if organism is not None:
        col = "organisms" if "organisms" in df.columns else "organism"
        df = df[df[col].str.contains(organism, case=False, na=False)]
    if protocol is not None:
        col = "protocols" if "protocols" in df.columns else "protocol"
        df = df[df[col].str.contains(protocol, case=False, na=False)]
    if min_cells is not None:
        col = "total_cells" if "total_cells" in df.columns else "n_cells"
        df = df[df[col] >= min_cells]
    if has_kraken2 is not None and "has_kraken2" in df.columns:
        df = df[df["has_kraken2"] == has_kraken2]
    return df.reset_index(drop=True)


def summary() -> str:
    """Print a one-line summary of the atlas and return it as a string.

    Example output:
        singlet atlas: 1,814 samples (687 SUCCESS) • 928 series • 8 species • 2.3M cells
    """
    df = _load_sample_index()
    total = len(df)
    success = df[df["status"] == "SUCCESS"] if "status" in df.columns else df
    cells_col = "cells_called" if "cells_called" in df.columns else "n_cells"
    total_cells = int(success[cells_col].sum()) if cells_col in success.columns else 0
    n_success = len(success)
    n_series = df["gse_id"].nunique() if "gse_id" in df.columns else 0
    n_species = len(species())

    def _fmt(n: int) -> str:
        if n >= 1e6:
            return f"{n / 1e6:.1f}M"
        if n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return str(n)

    msg = (
        f"singlet atlas: {total:,} samples ({n_success:,} SUCCESS) • "
        f"{n_series:,} series • {n_species} species • {_fmt(total_cells)} cells"
    )
    print(msg)
    return msg


def samples(
    gse_id: Optional[str] = None,
    organism: Optional[str] = None,
    status: Optional[str] = None,
    tissue: Optional[str] = None,
    min_cells: Optional[int] = None,
    quality_tier: Optional[str] = None,
    search: Optional[str] = None,
) -> pd.DataFrame:
    """Query the sample index with optional filters.

    Parameters
    ----------
    gse_id : str, optional
        Filter by GEO series accession.
    organism : str, optional
        Filter by organism (substring match).
    status : str, optional
        Filter by pipeline status (e.g. "SUCCESS").
    tissue : str, optional
        Filter by tissue/source (substring match, e.g. "brain", "lung", "PBMC").
    min_cells : int, optional
        Minimum cell count.
    quality_tier : str, optional
        Filter by quality tier: "gold" (MR>=70%, genes>=500, cells>=500),
        "silver" (MR>=50%, genes>=200, cells>=100), or "bronze" (all SUCCESS).
    search : str, optional
        Text search across title and other text columns (substring match).

    Returns
    -------
    pd.DataFrame
        Filtered sample index.
    """
    df = _load_sample_index()
    if search is not None:
        text_cols = df.select_dtypes(include="object").columns
        mask = df[text_cols].apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        df = df[mask]
    if gse_id is not None:
        df = df[df["gse_id"] == gse_id]
    if organism is not None:
        df = df[df["organism"].str.contains(organism, case=False, na=False)]
    if status is not None:
        df = df[df["status"] == status]
    if tissue is not None and "source" in df.columns:
        df = df[df["source"].str.contains(tissue, case=False, na=False)]
    if min_cells is not None:
        col = "cells_called" if "cells_called" in df.columns else "n_cells"
        df = df[df[col] >= min_cells]
    if quality_tier is not None:
        df = df[df["status"] == "SUCCESS"]
        cells_col = "cells_called" if "cells_called" in df.columns else "n_cells"
        has_genes = "median_genes" in df.columns
        if quality_tier == "gold":
            df = df[(df["mapping_rate"] >= 0.7) & (df[cells_col] >= 500)]
            if has_genes:
                df = df[df["median_genes"] >= 500]
        elif quality_tier == "silver":
            df = df[(df["mapping_rate"] >= 0.5) & (df[cells_col] >= 100)]
            if has_genes:
                df = df[df["median_genes"] >= 200]
    return df.reset_index(drop=True)


def top_series(
    n: int = 10,
    min_samples: int = 3,
    organism: Optional[str] = None,
) -> pd.DataFrame:
    """Return top GEO series ranked by total cell count.

    Parameters
    ----------
    n : int
        Number of series to return (default 10).
    min_samples : int
        Minimum number of SUCCESS samples in series (default 3).
    organism : str, optional
        Filter by organism (substring match).

    Returns
    -------
    pd.DataFrame
        Columns: gse_id, n_samples, total_cells, avg_mapping_rate, avg_median_genes, organism
    """
    df = _load_sample_index()
    df = df[df["status"] == "SUCCESS"]
    if organism is not None:
        df = df[df["organism"].str.contains(organism, case=False, na=False)]

    cells_col = "cells_called" if "cells_called" in df.columns else "n_cells"

    agg_dict = {
        "n_samples": (cells_col, "count"),
        "total_cells": (cells_col, "sum"),
        "avg_mapping_rate": ("mapping_rate", "mean"),
        "organism": ("organism", "first"),
    }
    if "median_genes" in df.columns:
        agg_dict["avg_median_genes"] = ("median_genes", "mean")

    grouped = df.groupby("gse_id").agg(**agg_dict).reset_index()

    grouped = grouped[grouped["n_samples"] >= min_samples]
    grouped = grouped.sort_values("total_cells", ascending=False).head(n)
    return grouped.reset_index(drop=True)
