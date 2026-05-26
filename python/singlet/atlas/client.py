# SPDX-License-Identifier: MIT
"""Cloudflare-native published-atlas client (BLOCKER #8 skeleton)."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

# Production endpoints (see MVP_ROADMAP.md BLOCKER #7/#8).
_DEFAULT_CDN = "https://r2.singlet.bio"
_DEFAULT_API = "https://api.singlet.bio"

_INDEX_KEY = "catalog/human_10x_atlas.parquet"


def _cdn_base() -> str:
    return os.environ.get("SINGLET_CDN_BASE", _DEFAULT_CDN).rstrip("/")


def _api_base() -> str:
    return os.environ.get("SINGLET_API_BASE", _DEFAULT_API).rstrip("/")


def index() -> "pd.DataFrame":
    """Fetch ``human_10x_atlas.parquet`` from R2 and return a DataFrame.

    Notes
    -----
    Skeleton — implementation pending BLOCKER #8 sprint.
    The intended implementation uses PyArrow over HTTP range so we don't
    pull the full Parquet for column-projected queries.
    """
    raise NotImplementedError(
        "singlet.atlas.index is a BLOCKER #8 skeleton; "
        "implementation pending Cloudflare Worker + R2 catalog Parquet landing"
    )


def sample(gsm_id: str) -> dict[str, Any]:
    """Fetch the per-sample summary for ``gsm_id`` via the Cloudflare Worker.

    Returns
    -------
    dict
        Contents of the sample's ``summary.json`` on R2 (edge-cached).

    Notes
    -----
    Skeleton — implementation pending BLOCKER #8 sprint.
    """
    raise NotImplementedError(
        "singlet.atlas.sample is a BLOCKER #8 skeleton; "
        "implementation pending Cloudflare Worker landing"
    )


def search(**filters: Any) -> dict[str, Any]:
    """Search the atlas catalog via the Cloudflare Worker.

    Examples
    --------
    >>> search(tissue="lung", protocol="10xv3")
    >>> search(disease="COVID-19", min_cells=10000)

    Notes
    -----
    Skeleton — implementation pending BLOCKER #8 sprint.
    """
    raise NotImplementedError(
        "singlet.atlas.search is a BLOCKER #8 skeleton; "
        "implementation pending Cloudflare Worker landing"
    )
