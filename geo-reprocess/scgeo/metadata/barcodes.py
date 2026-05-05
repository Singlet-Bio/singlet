"""Barcode normalization and matching between author metadata and processed cells.

Handles the various barcode formats found in GEO supplementary files:
  - ``ACGTACGTACGTACGT`` (bare 16-bp 10x barcode)
  - ``ACGTACGTACGTACGT-1`` (10x with gem-well suffix)
  - ``SampleName_ACGTACGTACGTACGT-1`` (sample-prefixed)
  - ``SampleName_ACGTACGTACGTACGT`` (sample-prefixed, no suffix)
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scgeo.config import get_config

logger = logging.getLogger(__name__)

# Recognises a 10x-style barcode (12–18 ACGTN bases, optionally with -N suffix)
_BARCODE_RE = re.compile(r"^[ACGTN]{12,18}(-\d+)?$")


def normalize_barcode(bc: str) -> str:
    """Normalize a cell barcode to its canonical sequence form.

    Strips:
      - Trailing gem-well suffix (``-1``, ``-2``, etc.)
      - Leading sample prefix (``SampleName_``)

    Args:
        bc: Raw barcode string.

    Returns:
        Bare nucleotide sequence (uppercase).

    Examples:
        >>> normalize_barcode("Vehicle1_AAACCCAAGCATGAAT-1")
        'AAACCCAAGCATGAAT'
        >>> normalize_barcode("AAACCCAAGCATGAAT")
        'AAACCCAAGCATGAAT'
    """
    bc = str(bc).strip()
    # Strip trailing -N or _N suffix (gem-well indicator)
    bc = re.sub(r"[-_]\d+$", "", bc)
    # If the whole thing is a barcode, return it
    if _BARCODE_RE.match(bc):
        return bc
    # Try splitting on underscore — check from the right
    parts = bc.split("_")
    for i in range(len(parts)):
        candidate = "_".join(parts[i:])
        candidate = re.sub(r"-\d+$", "", candidate)
        if _BARCODE_RE.match(candidate):
            return candidate
    # Fallback: return as-is (non-10x barcodes, e.g. Smart-seq cell IDs)
    return bc


def match_barcodes(
    author_barcodes: List[str],
    our_barcodes: List[str],
) -> Dict[str, str]:
    """Match author barcodes to our processed barcodes.

    Tries multiple strategies in order of specificity and returns the
    mapping from the strategy with the highest match count:

    1. **Direct match** — exact string equality
    2. **Normalized match** — :func:`normalize_barcode` on both sides
    3. **Suffix-stripped match** — remove ``-N`` suffix from author side
    4. **Case-insensitive match** — lowercased comparison
    5. **Substring match** — author barcode contained in our barcode
       or vice versa (catches ``SampleX_BARCODE`` vs ``BARCODE`` cases)
    6. **Nucleotide extraction** — extract any 12-18bp nucleotide
       sequence from both sides and match

    Args:
        author_barcodes: Barcodes from the author's metadata file.
        our_barcodes: Barcodes from ``cells.parquet``.

    Returns:
        ``{author_barcode: our_barcode}`` for every matched pair.
    """
    our_set = set(our_barcodes)
    our_norm = {normalize_barcode(bc): bc for bc in our_barcodes}

    strategies: List[Tuple[str, Dict[str, str]]] = []

    # Strategy 1: Direct
    direct = {abc: abc for abc in author_barcodes if abc in our_set}
    strategies.append(("direct", direct))

    # Strategy 2: Normalize both sides
    normalized = {}
    for abc in author_barcodes:
        norm = normalize_barcode(abc)
        if norm in our_norm:
            normalized[abc] = our_norm[norm]
    strategies.append(("normalized", normalized))

    # Strategy 3: Strip -N or _N from author
    suffix_stripped = {}
    for abc in author_barcodes:
        stripped = re.sub(r"[-_]\d+$", "", abc)
        if stripped in our_set:
            suffix_stripped[abc] = stripped
    strategies.append(("suffix-stripped", suffix_stripped))

    # Strategy 4: Case-insensitive match
    our_lower_map = {}
    for obc in our_barcodes:
        our_lower_map.setdefault(obc.lower(), obc)
    case_insensitive = {}
    for abc in author_barcodes:
        lower = abc.lower()
        if lower in our_lower_map:
            case_insensitive[abc] = our_lower_map[lower]
    strategies.append(("case-insensitive", case_insensitive))

    # Strategy 5: Substring match (author bc ends with our bc or vice versa)
    # Only attempt if other strategies yielded few matches and sets are < 200k
    best_so_far = max(len(m) for _, m in strategies)
    if best_so_far < len(our_barcodes) * 0.1 and len(our_barcodes) < 200_000 and len(author_barcodes) < 200_000:
        our_suffix_map: Dict[str, str] = {}
        for obc in our_barcodes:
            # Index by last 16+ chars (nucleotide portion)
            m = _BARCODE_RE.match(re.sub(r"[-_]\d+$", "", obc))
            if m:
                our_suffix_map[m.group(0)] = obc

        substring_match = {}
        for abc in author_barcodes:
            stripped = re.sub(r"[-_]\d+$", "", abc)
            # Check if stripping any prefix gives us a matching barcode
            m = _BARCODE_RE.search(stripped)
            if m and m.group(0) in our_suffix_map:
                substring_match[abc] = our_suffix_map[m.group(0)]
        strategies.append(("substring", substring_match))

    # Strategy 6: Extract any nucleotide sequence from both sides
    # Catches complex prefixes like "sample1.ACGT..." or "patient_A_ACGT..."
    best_so_far = max(len(m) for _, m in strategies)
    if best_so_far < len(our_barcodes) * 0.1 and len(our_barcodes) < 200_000 and len(author_barcodes) < 200_000:
        nuc_re = re.compile(r"[ACGTN]{12,18}")
        our_nuc_map: Dict[str, str] = {}
        for obc in our_barcodes:
            m = nuc_re.search(obc)
            if m:
                our_nuc_map.setdefault(m.group(0), obc)

        nuc_match = {}
        for abc in author_barcodes:
            m = nuc_re.search(abc.upper())
            if m and m.group(0) in our_nuc_map:
                nuc_match[abc] = our_nuc_map[m.group(0)]
        strategies.append(("nucleotide-extract", nuc_match))

    # Return strategy with most matches
    best_name, best_map = max(strategies, key=lambda x: len(x[1]))
    logger.debug(
        "Barcode matching: %s -> %s",
        ", ".join(f"{name}={len(m)}" for name, m in strategies),
        best_name,
    )
    return best_map


def align_author_metadata(
    author_obs: pd.DataFrame,
    gse_id: str,
    gsm_id: str,
    dataset_dir=None,
) -> Tuple[pd.DataFrame, Dict]:
    """Align author cell-level metadata to our cell barcodes.

    Reads our ``cells.parquet`` for the given GSM and matches author
    barcodes.  Unmatched cells receive ``NaN`` for all annotation
    columns.

    The function returns **all** author-provided columns — not just
    cell types.  QC-related columns that duplicate our own pipeline
    metrics are not filtered here; the caller decides what to keep.

    Args:
        author_obs: DataFrame with a ``barcode`` column (or barcode
            index) and arbitrary annotation columns.
        gse_id: GEO series accession.
        gsm_id: GEO sample accession.
        dataset_dir: Root of the cellarium dataset tree.

    Returns:
        Tuple of:

        - DataFrame indexed by our barcodes with author columns
          (``NaN`` where no match).
        - Stats dict: ``n_author``, ``n_ours``, ``n_matched``,
          ``match_rate``.
    """
    if dataset_dir is None:
        config = get_config()
        dataset_dir = config.paths.project_base / "dataset"

    cells_path = dataset_dir / gse_id / gsm_id / "cells.parquet"
    if not cells_path.exists():
        return pd.DataFrame(), {"n_author": 0, "n_ours": 0, "n_matched": 0, "match_rate": 0.0}

    cells = pq.read_table(str(cells_path), columns=["barcode"]).to_pandas()
    our_barcodes = cells["barcode"].tolist()
    n_ours = len(our_barcodes)

    if author_obs.empty:
        empty = pd.DataFrame(index=our_barcodes)
        return empty, {"n_author": 0, "n_ours": n_ours, "n_matched": 0, "match_rate": 0.0}

    # Ensure barcode column
    if "barcode" not in author_obs.columns:
        if author_obs.index.dtype == object:
            author_obs = author_obs.copy()
            author_obs["barcode"] = author_obs.index
        else:
            return pd.DataFrame(index=our_barcodes), {
                "n_author": len(author_obs), "n_ours": n_ours,
                "n_matched": 0, "match_rate": 0.0,
            }

    author_barcodes = author_obs["barcode"].tolist()
    n_author = len(author_barcodes)

    # Match
    bc_mapping = match_barcodes(author_barcodes, our_barcodes)
    n_matched = len(bc_mapping)

    # Build reverse mapping: our_bc -> author_bc
    reverse_map = {v: k for k, v in bc_mapping.items()}

    # Annotation columns (everything except barcode)
    ann_cols = [c for c in author_obs.columns if c != "barcode"]

    # Index author_obs by barcode for fast lookup
    author_indexed = author_obs.set_index("barcode", drop=False)

    rows = []
    for our_bc in our_barcodes:
        if our_bc in reverse_map:
            author_bc = reverse_map[our_bc]
            if author_bc in author_indexed.index:
                row = author_indexed.loc[author_bc]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                rows.append(row[ann_cols].to_dict())
            else:
                rows.append({c: np.nan for c in ann_cols})
        else:
            rows.append({c: np.nan for c in ann_cols})

    result = pd.DataFrame(rows, index=our_barcodes)
    result.index.name = None

    stats = {
        "n_author": n_author,
        "n_ours": n_ours,
        "n_matched": n_matched,
        "match_rate": n_matched / n_ours if n_ours > 0 else 0.0,
    }
    logger.info(
        "%s/%s barcode match: %d/%d (%.1f%%)",
        gse_id, gsm_id, n_matched, n_ours, stats["match_rate"] * 100,
    )
    return result, stats
