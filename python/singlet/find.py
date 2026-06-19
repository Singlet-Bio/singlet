# SPDX-License-Identifier: MIT
"""Natural-language search over the Singlet atlas.

:func:`find` turns a plain-English description (e.g. ``"human lung fibroblasts"``)
into a list of matching GEO accessions by calling the hosted search endpoint.
:func:`find_load` is a convenience that loads the matches directly into one
AnnData.

The endpoint is ``GET {API_BASE}/nl-search`` where ``API_BASE`` is
``$SINGLET_API_BASE`` (default ``https://singlet.bio/api``). It returns JSON with
a top-level ``accessions`` array.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata

__all__ = ["find", "find_load"]

_API_BASE_DEFAULT = "https://singlet.bio/api"
_USER_AGENT = "singlet-find/1"


class SingletSearchError(RuntimeError):
    """Raised when the natural-language search endpoint cannot be reached."""


def _api_base() -> str:
    """Base URL for the public REST API. Override with ``$SINGLET_API_BASE``."""
    return os.environ.get("SINGLET_API_BASE", _API_BASE_DEFAULT).rstrip("/")


def find(query: str, *, level: str = "gsm", limit: int = 50) -> list[str]:
    """Find datasets by natural-language description.

    Sends *query* to the hosted search endpoint and returns the matching GEO
    accessions, most relevant first.

    Parameters
    ----------
    query : str
        Plain-English description, e.g. ``"exhausted T cells in melanoma"`` or
        ``"human lung 10x"``.
    level : str
        Accession granularity to return: ``"gsm"`` (samples, default) or
        ``"gse"`` (series).
    limit : int
        Maximum number of accessions to return.

    Returns
    -------
    list of str
        Matching accession strings (e.g. ``["GSM5238385", ...]``). Empty if
        nothing matched. Feed the result straight into :func:`singlet.load`.

    Raises
    ------
    SingletSearchError
        If the search endpoint cannot be reached or returns an error.

    Examples
    --------
    >>> import singlet
    >>> accs = singlet.find("human lung fibroblasts")            # doctest: +SKIP
    >>> adata = singlet.load(accs)                               # doctest: +SKIP
    """
    if query is None or not str(query).strip():
        raise ValueError("find() requires a non-empty query string")

    params = {"q": str(query), "level": level, "limit": int(limit)}
    url = f"{_api_base()}/nl-search?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        raise SingletSearchError(
            f"Search request failed: {url} → HTTP {e.code}"
        ) from e
    except urllib.error.URLError as e:
        raise SingletSearchError(
            f"Could not reach the Singlet search endpoint ({url}): {e.reason}"
        ) from e
    except (ValueError, json.JSONDecodeError) as e:
        raise SingletSearchError(
            f"Search endpoint returned an invalid response from {url}"
        ) from e

    accessions = payload.get("accessions")
    if accessions is None:
        # Tolerate alternate shapes but be explicit if neither is present.
        accessions = payload.get("results") or []
    return [str(a) for a in accessions][: int(limit)]


def find_load(
    query: str,
    *,
    level: str = "gsm",
    limit: int = 50,
    **load_kwargs,
) -> "anndata.AnnData":
    """Find datasets by natural-language description and load them as AnnData.

    Convenience wrapper equivalent to ``singlet.load(singlet.find(query, ...))``.

    Parameters
    ----------
    query : str
        Plain-English description (see :func:`find`).
    level : str
        ``"gsm"`` (default) or ``"gse"``.
    limit : int
        Maximum number of accessions to load and concatenate.
    **load_kwargs
        Forwarded to :func:`singlet.load` (e.g. ``genes=``, ``obs_filter=``).

    Returns
    -------
    anndata.AnnData
        The matching datasets concatenated into one AnnData.

    Raises
    ------
    SingletSearchError
        If the search endpoint cannot be reached.
    LookupError
        If the query matched no datasets.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.find_load("human pancreas islet cells")   # doctest: +SKIP
    """
    accessions = find(query, level=level, limit=limit)
    if not accessions:
        raise LookupError(f"No datasets matched query {query!r}")

    from singlet._loader import load

    return load(accessions, **load_kwargs)
