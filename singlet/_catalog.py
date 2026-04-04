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

_CATALOG_URL = "https://zenodo.org/records/XXXXXX/files/catalog_v1.parquet"
_SAMPLE_INDEX_URL = "https://zenodo.org/records/XXXXXX/files/sample_index.parquet"

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
        raise RuntimeError(
            "Could not load sample index. Set SINGLET_CATALOG_DIR or check internet."
        )
    return _SAMPLE_INDEX_CACHE


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
    all_species = set()
    for org in df["organism"].dropna():
        for s in org.split("|"):
            all_species.add(s.strip())
    return sorted(all_species)


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
        df = df[df["organism"].str.contains(organism, case=False, na=False)]
    if protocol is not None:
        df = df[df["protocol"].str.contains(protocol, case=False, na=False)]
    if min_cells is not None:
        df = df[df["n_cells"] >= min_cells]
    if has_kraken2 is not None:
        df = df[df["has_kraken2"] == has_kraken2]
    return df.reset_index(drop=True)
