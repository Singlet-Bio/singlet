"""SOFT file metadata extraction for GSE/GSM-level annotations.

Reads the pre-parsed SOFT catalog (stage3_soft_parsed.parquet) and builds
per-cell observation DataFrames from sample-level characteristics.
"""

import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pyarrow.parquet as pq

from scgeo.config import get_config

logger = logging.getLogger(__name__)


def load_soft_metadata(
    gse_id: str,
    catalog_path: Optional[Path] = None,
) -> Dict:
    """Load SOFT-parsed metadata for a GSE from the catalog parquet.

    Args:
        gse_id: GEO series accession (e.g. ``"GSE281311"``).
        catalog_path: Path to ``stage3_soft_parsed.parquet``.
            Defaults to ``{catalog_dir}/stage3_soft_parsed.parquet``.

    Returns:
        Dictionary with keys:

        - ``gse_id``: series accession
        - ``overall_design``: experiment description
        - ``data_processing``: methods text
        - ``samples``: ``{gsm_id: {key: value, ...}}`` from
          :func:`parse_characteristics`
        - ``supplementary_files``: ``{gsm_id: raw_string}``
    """
    if catalog_path is None:
        config = get_config()
        catalog_path = config.paths.catalog_dir / "stage3_soft_parsed.parquet"

    t = pq.read_table(
        str(catalog_path),
        columns=[
            "gse_id", "gsm_id", "sample_characteristics", "sample_source",
            "overall_design", "sample_data_processing", "supplementary_files",
        ],
        filters=[("gse_id", "=", gse_id)],
    )
    df = t.to_pandas()

    if df.empty:
        logger.warning("No SOFT data found for %s", gse_id)
        return {
            "gse_id": gse_id,
            "overall_design": "",
            "data_processing": "",
            "samples": {},
            "supplementary_files": {},
        }

    return {
        "gse_id": gse_id,
        "overall_design": df["overall_design"].iloc[0] or "",
        "data_processing": df["sample_data_processing"].iloc[0] or "",
        "samples": {
            row["gsm_id"]: parse_characteristics(row["sample_characteristics"])
            for _, row in df.iterrows()
        },
        "supplementary_files": {
            row["gsm_id"]: row["supplementary_files"] or ""
            for _, row in df.iterrows()
        },
    }


def parse_characteristics(chars_str: Optional[str]) -> OrderedDict:
    """Parse SOFT characteristics string into an ordered dictionary.

    GEO SOFT stores sample characteristics as semicolon-separated
    ``key: value`` pairs, e.g.::

        tissue: lung ;; cell type: epithelial ;; genotype: wildtype

    Args:
        chars_str: Raw characteristics string from SOFT.

    Returns:
        ``OrderedDict`` of ``{key: value}`` pairs.  Keys are lowercased
        and stripped.  Returns empty dict if *chars_str* is falsy.
    """
    if not chars_str:
        return OrderedDict()
    result = OrderedDict()
    for kv in chars_str.split(" ;; "):
        if ":" in kv:
            key, val = kv.split(":", 1)
            result[key.strip().lower()] = val.strip()
    return result


def build_gsm_level_obs(
    gse_id: str,
    gsm_id: str,
    soft_meta: Dict,
    dataset_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Build per-cell observation DataFrame from GSM-level SOFT metadata.

    Every cell in the GSM receives the same sample-level annotations
    (tissue, treatment, genotype, etc.).  GSE-level text fields
    (``overall_design``, ``data_processing``) are stored once per GSM.

    Args:
        gse_id: GEO series accession.
        gsm_id: GEO sample accession.
        soft_meta: Output of :func:`load_soft_metadata`.
        dataset_dir: Root of the cellarium dataset tree.
            Defaults to ``{project_base}/dataset``.

    Returns:
        DataFrame indexed by cell barcode with one column per
        characteristic key, plus ``gse_overall_design`` and
        ``gse_data_processing``.  Empty DataFrame if the sample
        has no ``cells.parquet``.
    """
    if dataset_dir is None:
        config = get_config()
        dataset_dir = config.paths.project_base / "dataset"

    cells_path = dataset_dir / gse_id / gsm_id / "cells.parquet"
    if not cells_path.exists():
        logger.debug("No cells.parquet for %s/%s", gse_id, gsm_id)
        return pd.DataFrame()

    cells = pq.read_table(str(cells_path), columns=["barcode"]).to_pandas()
    barcodes = cells["barcode"].tolist()

    chars = soft_meta["samples"].get(gsm_id, {})
    obs_data = {"barcode": barcodes}

    for key, val in chars.items():
        col_name = re.sub(r"[^a-z0-9_]", "_", key)
        obs_data[col_name] = val  # broadcast scalar to all cells

    obs_data["gse_overall_design"] = soft_meta["overall_design"]
    obs_data["gse_data_processing"] = soft_meta["data_processing"]

    obs_df = pd.DataFrame(obs_data)
    obs_df.index = obs_df["barcode"]
    obs_df.index.name = None
    return obs_df
