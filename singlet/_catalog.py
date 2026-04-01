"""Browse the SingletDB catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

_CATALOG_URL = "https://zenodo.org/records/XXXXXX/files/singletdb_catalog.parquet"
_CATALOG_CACHE: Optional[pd.DataFrame] = None


def _load_catalog() -> pd.DataFrame:
    """Load the catalog, using a cached copy if available."""
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE

    cache_dir = Path.home() / ".singlet" / "cache"
    cache_path = cache_dir / "catalog.parquet"

    if cache_path.exists():
        _CATALOG_CACHE = pd.read_parquet(cache_path)
        return _CATALOG_CACHE

    try:
        import requests

        cache_dir.mkdir(parents=True, exist_ok=True)
        resp = requests.get(_CATALOG_URL, timeout=60)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        _CATALOG_CACHE = pd.read_parquet(cache_path)
    except Exception:
        bundled = Path(__file__).parent / "data" / "catalog.parquet"
        if bundled.exists():
            _CATALOG_CACHE = pd.read_parquet(bundled)
        else:
            raise RuntimeError(
                "Could not load catalog. Check your internet connection."
            )

    return _CATALOG_CACHE


def catalog(search: Optional[str] = None) -> pd.DataFrame:
    """Return the full dataset catalog as a DataFrame.

    Parameters
    ----------
    search : str, optional
        Filter catalog rows where any text column matches this substring.

    Returns
    -------
    pd.DataFrame
        One row per GEO series. Columns include accession, species, tissue,
        cell_count, sample_count, modality, pub_date, title, etc.
    """
    df = _load_catalog()
    if search is not None:
        text_cols = df.select_dtypes(include="object").columns
        mask = df[text_cols].apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        return df[mask].reset_index(drop=True)
    return df


def info(accession: str) -> dict:
    """Return metadata for a single dataset.

    Parameters
    ----------
    accession : str
        GEO series accession (e.g. "GSE136831").

    Returns
    -------
    dict
        Dataset metadata including species, tissue, cell_count, modality, etc.
    """
    df = _load_catalog()
    rows = df[df["accession"] == accession]
    if rows.empty:
        raise KeyError(f"Accession {accession!r} not found in catalog")
    return rows.iloc[0].to_dict()


def species() -> list[str]:
    """Return all species with processed data."""
    df = _load_catalog()
    return sorted(df["species"].dropna().unique().tolist())


def tissues(species: Optional[str] = None) -> list[str]:
    """Return all tissues, optionally filtered by species."""
    df = _load_catalog()
    if species is not None:
        df = df[df["species"].str.lower() == species.lower()]
    return sorted(df["tissue"].dropna().unique().tolist())


def datasets(
    species: Optional[str] = None,
    tissue: Optional[str] = None,
    modality: Optional[str] = None,
) -> pd.DataFrame:
    """Filter catalog by species, tissue, and/or modality."""
    df = _load_catalog()
    if species is not None:
        df = df[df["species"].str.lower() == species.lower()]
    if tissue is not None:
        df = df[df["tissue"].str.lower() == tissue.lower()]
    if modality is not None:
        df = df[df["modality"].str.lower() == modality.lower()]
    return df.reset_index(drop=True)
