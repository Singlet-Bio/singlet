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
    """Return tissue breakdown across SUCCESS samples.

    Returns a DataFrame with columns: tissue, count, sorted by count descending.
    Uses normalized tissue annotations extracted from GEO characteristics.
    """
    df = _load_sample_index()
    df = df[df["status"] == "SUCCESS"]
    if "tissue" in df.columns:
        counts = df["tissue"].dropna().value_counts().reset_index()
        counts.columns = ["tissue", "count"]
        return counts
    if "source" not in df.columns:
        return pd.DataFrame(columns=["tissue", "count"])
    counts = df["source"].dropna().value_counts().reset_index()
    counts.columns = ["tissue", "count"]
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


def failure_categories() -> pd.DataFrame:
    """Return failure category breakdown for non-SUCCESS samples.

    Analyzes why samples failed processing. Categories include:
    - align_low_map: mapping rate below threshold
    - download_fail: SRA download failed
    - pipeline_crash: unexpected pipeline error
    - cells_below_threshold: too few cells called
    - align_oom: out-of-memory during alignment
    - data_incomplete: missing input data

    Returns a DataFrame with columns: category, count, pct, sorted by count descending.
    """
    df = _load_sample_index()
    failed = df[df["status"] != "SUCCESS"]
    total_failed = len(failed)
    if total_failed == 0:
        return pd.DataFrame(columns=["category", "count", "pct"])

    # Infer failure category from status + metrics
    categories = []
    for _, row in failed.iterrows():
        mr = row.get("mapping_rate") or 0
        cells = row.get("cells_called") or 0
        status = row.get("status", "")

        if mr == 0 and cells == 0:
            if status == "HARD_FAIL":
                categories.append("download_fail")
            else:
                categories.append("pipeline_crash")
        elif mr > 0 and mr < 0.1:
            categories.append("align_low_map")
        elif mr >= 0.1 and cells < 50:
            categories.append("cells_below_threshold")
        elif mr >= 0.1 and cells >= 50:
            categories.append("align_low_map")
        else:
            categories.append("unknown")

    from collections import Counter
    counts = Counter(categories)
    result = pd.DataFrame([
        {"category": cat, "count": c, "pct": round(c / total_failed * 100, 1)}
        for cat, c in counts.most_common()
    ])
    return result


def cell_types() -> pd.DataFrame:
    """Return cell type distribution across SUCCESS samples.

    Parses the 'source' field for cell type annotations from GEO.
    Normalizes common variants (e.g., 'PBMCs' → 'PBMC', 'K562' → 'cell line').
    Excludes tissue-only annotations (brain, blood, lung, etc.).

    Returns a DataFrame with columns: cell_type, count, sorted by count descending.
    """
    df = _load_sample_index()
    success = df[df["status"] == "SUCCESS"]

    # These are tissues, NOT cell types — skip them
    TISSUE_WORDS = {
        "blood", "brain", "lung", "liver", "kidney", "skin", "heart", "spleen",
        "pancreas", "colon", "intestine", "stomach", "ovary", "testis", "breast",
        "muscle", "bone marrow", "lymph node", "thymus", "prostate", "placenta",
        "retina", "adipose", "bladder", "thyroid", "embryo", "tumor", "tonsil",
        "dorsal root ganglion", "leptomeninges", "csf", "esophagus", "villi",
        "terminal ileum", "cornea", "uterus", "endometrium", "human endometrium",
        "bronchoalveolar lavage fluid", "melanoma", "cerebrospinal fluid cells",
        "hippocampus", "jejunum", "ileum", "vulva", "oropharynx", "adnexa",
        "frontal cortex", "carotid artery", "cardiac left atria", "pancreatic islet",
    }

    # Cell type normalization
    NORMALIZE = {
        "pbmcs": "PBMC", "pbmc": "PBMC", "peripheral blood mononuclear cells": "PBMC",
        "periperal blood mononuclear cells": "PBMC",
        "k562": "cell line (K562)", "k562 cells": "cell line (K562)",
        "jurkat": "cell line (Jurkat)", "jurkat cells": "cell line (Jurkat)",
        "hek293": "cell line (HEK293)", "hela": "cell line (HeLa)",
        "mcf7": "cell line (MCF7)", "a549": "cell line (A549)",
        "mda-mb-231 cells": "cell line (MDA-MB-231)",
        "mdamb231 cell line": "cell line (MDA-MB-231)",
        "thp-1 cell": "cell line (THP-1)", "thp-1 cells": "cell line (THP-1)",
        "cell line": "cell line", "hesc": "stem cell (hESC)",
        "hek 293t": "cell line (HEK293)",
        "pluripotent stem cells": "stem cell (iPSC)", "ipsc": "stem cell (iPSC)",
        "ips cells": "stem cell (iPSC)", "hpsc": "stem cell (hPSC)",
        "immune cells": "immune cells", "immune cell": "immune cells",
        "t cells": "T cells", "t cell": "T cells", "cd4 t cells": "CD4+ T cells",
        "cd8 t cells": "CD8+ T cells", "cd4+ t cells": "CD4+ T cells",
        "cd8+ t cells": "CD8+ T cells",
        "car t cells": "CAR-T cells", "manufactured car t cells": "CAR-T cells",
        "car-t cells": "CAR-T cells",
        "b cells": "B cells", "b cell": "B cells", "memory b cell": "memory B cells",
        "nk cells": "NK cells", "nk cell": "NK cells",
        "sorted memory b and t lymphocytes": "lymphocytes",
        "plasma cells": "plasma cells", "monocytes": "monocytes",
        "macrophages": "macrophages", "dendritic cells": "dendritic cells",
        "fibroblasts": "fibroblasts", "fibroblast": "fibroblasts",
        "epithelial cells": "epithelial cells", "endothelial cells": "endothelial cells",
        "neurons": "neurons", "astrocytes": "astrocytes",
        "organoid": "organoid", "organoids": "organoid",
        "peripheral blood": "PBMC",
    }

    KEYWORDS = [
        ("pbmc", "PBMC"), ("peripheral blood mononuclear", "PBMC"),
        ("t cell", "T cells"), ("cd4", "CD4+ T cells"), ("cd8", "CD8+ T cells"),
        ("car-t", "CAR-T cells"), ("car t", "CAR-T cells"),
        ("b cell", "B cells"), ("memory b", "memory B cells"),
        ("nk cell", "NK cells"), ("natural killer", "NK cells"),
        ("monocyte", "monocytes"), ("macrophage", "macrophages"),
        ("dendritic", "dendritic cells"), ("neutrophil", "neutrophils"),
        ("stem cell", "stem cells"), ("ipsc", "stem cell (iPSC)"),
        ("ips ", "stem cell (iPSC)"), ("hesc", "stem cell (hESC)"),
        ("pluripotent", "stem cells"),
        ("fibroblast", "fibroblasts"), ("epithelial", "epithelial cells"),
        ("endothelial", "endothelial cells"), ("neuron", "neurons"),
        ("astrocyte", "astrocytes"), ("microglia", "microglia"),
        ("organoid", "organoid"),
    ]

    from collections import Counter
    cell_type_counts = Counter()

    for _, row in success.iterrows():
        src = str(row.get("source") or "").lower().strip()
        if not src:
            continue

        # Skip if it's clearly just a tissue
        if src in TISSUE_WORDS:
            continue

        # Try exact match
        if src in NORMALIZE:
            cell_type_counts[NORMALIZE[src]] += 1
            continue

        # Try keyword match
        matched = False
        for keyword, ct in KEYWORDS:
            if keyword in src:
                cell_type_counts[ct] += 1
                matched = True
                break

        if not matched:
            # Skip if it contains only tissue keywords
            is_tissue = any(tw in src for tw in TISSUE_WORDS)
            if not is_tissue and len(src) < 40 and not src.startswith("gsm"):
                # Skip entries that look like experiment descriptions or anatomy
                skip_words = ("seq", "rna-", "protocol", "library", "experiment",
                              "sample", "cortex", "artery", "junction", "eminence",
                              "lobe", "ventricle")
                if not any(w in src for w in skip_words):
                    cell_type_counts[src] += 1

    result = pd.DataFrame([
        {"cell_type": ct, "count": c}
        for ct, c in cell_type_counts.most_common()
        if c >= 2  # Filter noise (singletons are too noisy)
    ])
    return result


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
    """Return a one-line summary of the atlas.

    Example output:
        singlet atlas: 2,712 samples (1,139 SUCCESS) • 572 series • 17 species • 29 protocols • 3.3M cells
    """
    df = _load_sample_index()
    total = len(df)
    success = df[df["status"] == "SUCCESS"] if "status" in df.columns else df
    cells_col = "cells_called" if "cells_called" in df.columns else "n_cells"
    total_cells = int(success[cells_col].sum()) if cells_col in success.columns else 0
    n_success = len(success)
    n_series = df["gse_id"].nunique() if "gse_id" in df.columns else 0

    # Count species from sample_index (more complete than catalog)
    if "organism" in df.columns:
        n_species = df["organism"].dropna().loc[lambda s: (s != "unknown") & (s.str.strip() != "")].nunique()
    else:
        n_species = len(species())

    # Count protocols (SUCCESS only)
    if "protocol" in success.columns:
        valid_protocols = success["protocol"].dropna().loc[lambda s: (s.str.strip() != "") & (s != "unknown")]
        n_protocols = valid_protocols.nunique()
    else:
        n_protocols = 0

    # Count tissues (SUCCESS only)
    if "tissue" in success.columns:
        n_tissues = success["tissue"].dropna().nunique()
    else:
        n_tissues = 0

    # Count cell types (SUCCESS only)
    if "cell_type" in success.columns:
        n_cell_types = success["cell_type"].dropna().nunique()
    else:
        n_cell_types = 0

    def _fmt(n: int) -> str:
        if n >= 1e6:
            return f"{n / 1e6:.1f}M"
        if n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return str(n)

    msg = (
        f"singlet atlas: {total:,} samples ({n_success:,} SUCCESS) • "
        f"{n_series:,} series • {n_species} species • {n_protocols} protocols • "
        f"{n_tissues} tissues • {n_cell_types} cell types • {_fmt(total_cells)} cells"
    )
    return msg


def samples(
    gse_id: Optional[str] = None,
    organism: Optional[str] = None,
    status: Optional[str] = None,
    tissue: Optional[str] = None,
    cell_type: Optional[str] = None,
    protocol: Optional[str] = None,
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
    cell_type : str, optional
        Filter by cell type (substring match, e.g. "PBMC", "T cells", "stem cell").
    protocol : str, optional
        Filter by protocol (e.g. "10xv3", "dropseq", "celseq2").
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
    if tissue is not None:
        if "tissue" in df.columns:
            df = df[df["tissue"].str.contains(tissue, case=False, na=False)]
        elif "source" in df.columns:
            df = df[df["source"].str.contains(tissue, case=False, na=False)]
    if cell_type is not None and "cell_type" in df.columns:
        df = df[df["cell_type"].str.contains(cell_type, case=False, na=False)]
    if protocol is not None and "protocol" in df.columns:
        df = df[df["protocol"].str.contains(protocol, case=False, na=False)]
    if min_cells is not None:
        col = "cells_called" if "cells_called" in df.columns else "n_cells"
        df = df[df[col] >= min_cells]
    if quality_tier is not None:
        df = df[df["status"] == "SUCCESS"]
        cells_col = "cells_called" if "cells_called" in df.columns else "n_cells"
        has_genes = "median_genes" in df.columns
        gold_mask = (df["mapping_rate"] >= 0.7) & (df[cells_col] >= 500)
        if has_genes:
            gold_mask = gold_mask & (df["median_genes"] >= 500)
        silver_mask = (df["mapping_rate"] >= 0.5) & (df[cells_col] >= 100)
        if has_genes:
            silver_mask = silver_mask & (df["median_genes"] >= 200)
        silver_mask = silver_mask & ~gold_mask

        if quality_tier == "gold":
            df = df[gold_mask]
        elif quality_tier == "silver":
            df = df[silver_mask]
        elif quality_tier == "bronze":
            df = df[~gold_mask & ~silver_mask]
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


def quality_tiers() -> pd.DataFrame:
    """Classify SUCCESS samples into quality tiers.

    Tiers:
      - gold: mapping_rate >= 0.7, median_genes >= 500, cells_called >= 500
      - silver: mapping_rate >= 0.5, median_genes >= 200, cells_called >= 100
      - bronze: all other SUCCESS samples

    Returns
    -------
    pd.DataFrame
        Columns: tier, count, pct, avg_mapping_rate, avg_median_genes, avg_cells
    """
    df = _load_sample_index()
    success = df[df["status"] == "SUCCESS"].copy()
    cells_col = "cells_called" if "cells_called" in success.columns else "n_cells"

    gold_mask = (
        (success["mapping_rate"] >= 0.7) &
        (success["median_genes"] >= 500) &
        (success[cells_col] >= 500)
    )
    silver_mask = (
        ~gold_mask &
        (success["mapping_rate"] >= 0.5) &
        (success["median_genes"] >= 200) &
        (success[cells_col] >= 100)
    )
    bronze_mask = ~gold_mask & ~silver_mask

    tiers = []
    for name, mask in [("gold", gold_mask), ("silver", silver_mask), ("bronze", bronze_mask)]:
        subset = success[mask]
        n = len(subset)
        tiers.append({
            "tier": name,
            "count": n,
            "pct": round(n / len(success) * 100, 1) if len(success) > 0 else 0,
            "avg_mapping_rate": round(subset["mapping_rate"].mean(), 4) if n > 0 else None,
            "avg_median_genes": round(subset["median_genes"].mean(), 1) if n > 0 and "median_genes" in subset.columns else None,
            "avg_cells": round(subset[cells_col].mean(), 0) if n > 0 else None,
        })

    return pd.DataFrame(tiers)
