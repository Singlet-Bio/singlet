"""Structured per-sample facts from the singlet.bio study API, as obs columns.

``GET {API_BASE}/gse/<GSE>`` returns, for every sample, ``facts`` taken from the sample's own GEO record (never inferred):

* ``tissue_type``: CELLxGENE schema ``tissue_type`` (``tissue``, ``organoid``, ``primary cell culture``, ``cell line``);
* ``cell_line``: Cellosaurus lines, ``[{"id": "CVCL_0004", "name": "K-562"}]``;
* ``tumor_sample_type``: GDC/TCGA-style ``tumor``, ``primary tumor``, ``metastatic``, ``recurrent tumor``, ``adjacent normal``;
* ``disease``: MONDO terms, or PATO:0000461 ``normal`` for healthy donors, ``[{"id": ..., "label": ...}]``;
* ``donor_status`` (``healthy`` / ``diseased``) and ``sample_kind`` (``primary``, ``cell_line``, ``stem_cell``, ``organoid``,
  ``xenograft``).

:func:`facts_obs` turns one sample's facts into the columns of :data:`FACT_OBS_COLUMNS`, repeated on each of its cells. A field
the record does not state is missing (NaN in the categorical), not an empty string. Several lines or diseases are joined with
``"; "``, ids in the same order as the names. The facts are fetched once per study and process; with ``$SINGLET_OFFLINE`` set,
for a study read from a local directory, or when the API cannot be reached, there are no facts and no columns are added.
"""

from __future__ import annotations

import json
import threading
from typing import Dict, Optional

import numpy as np

#: obs columns built from the facts, in order (ontology pairs follow the CELLxGENE naming)
FACT_OBS_COLUMNS = (
    "tissue_type",
    "cell_line",
    "cell_line_cellosaurus_id",
    "tumor_sample_type",
    "disease",
    "disease_ontology_term_id",
    "donor_status",
    "sample_kind",
)

_SCALARS = ("tissue_type", "tumor_sample_type", "donor_status", "sample_kind")
_memo: Dict[str, Dict[str, dict]] = {}
_lock = threading.Lock()


def _join(items, key: str) -> Optional[str]:
    vals = [str(x.get(key) or "") for x in items if isinstance(x, dict)]
    vals = [v for v in vals if v]
    return "; ".join(vals) if vals else None


def flatten(facts: Optional[dict]) -> Dict[str, Optional[str]]:
    """One sample's ``facts`` object -> {column: value or None} for :data:`FACT_OBS_COLUMNS`."""
    f = facts if isinstance(facts, dict) else {}
    out: Dict[str, Optional[str]] = {c: None for c in FACT_OBS_COLUMNS}
    for k in _SCALARS:
        v = f.get(k)
        out[k] = str(v) if v not in (None, "") else None
    lines = f.get("cell_line") if isinstance(f.get("cell_line"), list) else []
    out["cell_line"] = _join(lines, "name") or _join(lines, "id")
    out["cell_line_cellosaurus_id"] = _join(lines, "id")
    dis = f.get("disease") if isinstance(f.get("disease"), list) else []
    out["disease"] = _join(dis, "label") or _join(dis, "id")
    out["disease_ontology_term_id"] = _join(dis, "id")
    return out


def parse_study_payload(payload) -> Dict[str, dict]:
    """``/api/gse/<GSE>`` JSON -> {GSM: facts} for the samples that carry any."""
    out: Dict[str, dict] = {}
    if not isinstance(payload, dict):
        return out
    for s in payload.get("samples") or []:
        if (
            isinstance(s, dict)
            and s.get("gsm_id")
            and isinstance(s.get("facts"), dict)
            and s["facts"]
        ):
            out[str(s["gsm_id"]).upper()] = s["facts"]
    return out


def fetch_sample_facts(gse: str, *, timeout: float = 15.0) -> Dict[str, dict]:
    """{GSM: facts} for a study from the live API, memoised per process; {} when offline or unreachable (never raises)."""
    import urllib.error
    import urllib.parse
    import urllib.request

    from singlet.find import _api_base, _offline, _request_headers

    gse = str(gse).strip().upper()
    with _lock:
        if gse in _memo:
            return _memo[gse]
    if _offline():
        return {}
    url = f"{_api_base()}/gse/{urllib.parse.quote(gse)}"
    try:
        req = urllib.request.Request(url, headers=_request_headers())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            facts = parse_study_payload(json.load(resp))
    except (urllib.error.URLError, OSError, ValueError):
        return {}  # not memoised: a later call may reach the API
    with _lock:
        _memo[gse] = facts
    return facts


def clear_memo() -> None:
    """Forget the facts fetched so far (tests, or to pick up a catalog update in a long session)."""
    with _lock:
        _memo.clear()


def study_facts(study) -> Dict[str, dict]:
    """The facts for a :class:`~singlet.s1.client.Study`, fetched only when it is read over HTTP(S) from a non-loopback host (a
    local directory or a local test server has no study API behind it)."""
    from urllib.parse import urlparse

    base = str(getattr(getattr(study, "transport", None), "base", "") or "")
    if not base.startswith(("http://", "https://")):
        return {}
    host = (urlparse(base).hostname or "").lower()
    if host in ("localhost", "::1") or host.startswith("127."):
        return {}
    return fetch_sample_facts(study.gse)


def facts_obs(facts: Optional[dict], n: int) -> Dict[str, np.ndarray]:
    """The fact columns for ``n`` cells of one sample (object arrays; None is missing)."""
    flat = flatten(facts)
    return {c: np.full(n, flat[c], dtype=object) for c in FACT_OBS_COLUMNS}
