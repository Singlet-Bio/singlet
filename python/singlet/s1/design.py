# SPDX-License-Identifier: MIT
"""singlet.s1.design -- readers for the design catalog sidecar of a study.

::

    d = singlet.design("GSE190760")               # one GET of s1/d/GSE190760.design.json (ETag revalidated)
    d.factors, d.levels, d.units                  # pandas DataFrames
    d.contrasts                                   # list of dicts
    d.tier, d.provenance
    singlet.contrasts(["GSE190760", "GSE190487"], factor_type="time", min_grade="B")
    singlet.sample_status("GSE190760")            # one row per unit: status and where it stands

A sidecar is a small JSON object next to the study manifest (``s1/d/<GSE>.design.json``, schema ``singlet-design-catalog-0.1``,
docs/CONTRASTS_CATALOG_SCHEMA.md). It exists before any counts do, so these readers work for studies that are scheduled, held
or still running. The data host is the one of :func:`singlet.s1.client.open_study`: ``base=`` or ``$SINGLET_DATA_BASE`` (the Beta
host is ``https://beta-data.singlet.bio``; a directory with the server layout also works). Sidecars are cached beside the
manifests (``<cache>/s1/d``) and revalidated with the cached ETag, like manifests.

Tiers.  ``confirmed`` is checked by a person against the paper. ``candidate`` is model-curated and unverified: read the contrast
evidence before relying on it. ``unresolved`` means the design could not be settled from the study's records. Every object returned
here carries the tier, and printing one repeats the caveat (:data:`CAVEAT`).
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from . import transport as T
from .client import S1Error, _transport_for, cache_root
from .transport import NotFound, Transport, TransportError

__all__ = [
    "Design",
    "ContrastList",
    "DesignNotFound",
    "design",
    "contrasts",
    "sample_status",
    "CAVEAT",
    "GRADES",
    "TIERS",
]

SCHEMA_PREFIX = "singlet-design-catalog-"
SCHEMA_MAJOR_MAX = 0  # readers accept singlet-design-catalog-0.x
TIERS = ("confirmed", "candidate", "unresolved")
GRADES = ("A", "B", "C", "D")  # best first
CAVEAT = (
    "tier 'candidate' means model-curated and unverified: the factors, levels and contrasts were read from the study's "
    "records by a language model and no person has checked them against the paper; check each contrast's evidence before use"
)


class DesignNotFound(S1Error):
    """The study has no design sidecar on the data host (not curated yet, or not in this channel's catalog)."""


def _caveat_for(tier: Optional[str]) -> str:
    if tier == "confirmed":
        return "tier 'confirmed': checked by a person against the paper"
    if tier == "unresolved":
        return "tier 'unresolved': the design could not be settled from the study's records; treat every field as incomplete. " + CAVEAT
    return CAVEAT


def _normalize(gse: str) -> str:
    from singlet._integrity import normalize_gse

    return normalize_gse(gse)


def _key(gse: str) -> str:
    return f"s1/d/{gse}.design.json"


def _cache_paths(gse: str):
    d = cache_root() / "d"
    return d / f"{gse}.design.json", d / f"{gse}.etag"


def _check(doc: Any, gse: str) -> dict:
    if not isinstance(doc, dict) or not str(doc.get("schema", "")).startswith(SCHEMA_PREFIX):
        raise S1Error(f"{gse}: not a singlet design catalog sidecar")
    try:
        major = int(str(doc["schema"])[len(SCHEMA_PREFIX) :].split(".")[0])
    except ValueError:
        raise S1Error(f"{gse}: unreadable sidecar schema {doc.get('schema')!r}") from None
    if major > SCHEMA_MAJOR_MAX:
        raise S1Error(f"{gse}: sidecar schema {doc['schema']} is newer than this client supports; upgrade singlet")
    if doc.get("study") not in (None, gse):
        raise S1Error(f"{gse}: the sidecar at this key describes {doc.get('study')}")
    return doc


def _fetch(gse: str, base: Optional[str], transport: Optional[Transport]) -> tuple:
    """(document, source) of the sidecar of ``gse``; the source says where it came from. Raises :class:`DesignNotFound`."""
    t = transport or _transport_for(base)
    key = _key(gse)

    def missing(e: Exception):
        return DesignNotFound(
            f"{gse} has no design sidecar at {t.url(key)}: it is not in this catalog yet. Check the data host "
            f"(base= or SINGLET_DATA_BASE; Beta is https://beta-data.singlet.bio) and the accession"
        )

    if t.is_local():
        try:
            raw = bytes(t.get(key, None, kind="design").body)
        except NotFound as e:
            raise missing(e) from e
        return _check(json.loads(raw.decode("utf-8")), gse), "local"
    cp, ep = _cache_paths(gse)
    etag = None
    if cp.exists() and ep.exists():
        try:
            etag = ep.read_text().strip() or None
        except OSError:
            etag = None
    try:
        r = t.get(key, None, if_none_match=etag, kind="design")
    except NotFound as e:
        raise missing(e) from e
    except TransportError as e:
        if cp.exists():
            warnings.warn(f"could not revalidate the design of {gse} ({e}); using the cached sidecar", RuntimeWarning, stacklevel=3)
            return _check(json.loads(cp.read_text(encoding="utf-8")), gse), "cache-offline"
        raise
    if r.status == 304 and cp.exists():
        return _check(json.loads(cp.read_text(encoding="utf-8")), gse), "cache-304"
    raw = bytes(r.body)
    doc = _check(json.loads(raw.decode("utf-8")), gse)
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(f".part{os.getpid()}")
        tmp.write_bytes(raw)
        os.replace(tmp, cp)
        if r.etag:
            ep.write_text(r.etag)
        else:
            ep.unlink(missing_ok=True)
    except OSError:  # a read-only cache is not an error
        pass
    return doc, "network"


# ================================================================================================================= tables
def _frames(doc: dict):
    import pandas as pd

    tier = doc.get("tier")
    factors, levels, units = [], [], []
    for f in doc.get("factors") or []:
        factors.append(
            {
                "factor_id": f.get("id"),
                "name": f.get("name"),
                "type": f.get("type"),
                "unit": f.get("unit"),
                "source": f.get("source"),
                "confidence": f.get("confidence"),
                "evidence": f.get("evidence"),
                "n_levels": len(f.get("levels") or []),
            }
        )
        for lv in f.get("levels") or []:
            levels.append(
                {
                    "factor_id": f.get("id"),
                    "level_id": lv.get("id"),
                    "label": lv.get("label"),
                    "standard": lv.get("standard"),
                    "role": lv.get("role"),
                    "ontology": list(lv.get("ontology") or []),
                }
            )
    fids = [f["factor_id"] for f in factors]
    for u in doc.get("units") or []:
        pool = u.get("pool") or {}
        row: Dict[str, Any] = {
            "unit_id": u.get("unit_id"),
            "gsm": u.get("gsm"),
            "library": u.get("library"),
            "biosample_key": u.get("biosample_key"),
            "donor": u.get("donor"),
            "pool_type": pool.get("type"),
            "pool_n_members": pool.get("n_members"),
            "pool_mapping": pool.get("mapping"),
            "status": u.get("status"),
            "in_campaign_sample_list": u.get("in_campaign_sample_list"),
        }
        lev = u.get("levels") or {}
        for fid in fids:
            row[fid] = lev.get(fid)
        for k, v in (u.get("covariates") or {}).items():
            row[f"cov_{k}"] = v
        units.append(row)
    fcols = ["factor_id", "name", "type", "unit", "source", "confidence", "evidence", "n_levels"]
    lcols = ["factor_id", "level_id", "label", "standard", "role", "ontology"]
    out = (
        pd.DataFrame(factors, columns=fcols),
        pd.DataFrame(levels, columns=lcols),
        pd.DataFrame(units),
    )
    for df in out:
        df.attrs["tier"] = tier
        df.attrs["caveat"] = _caveat_for(tier)
    return out


def _status_frame(doc: dict):
    import pandas as pd

    tier = doc.get("tier")
    rows = []
    for u in doc.get("units") or []:
        rows.append(
            {
                "study": doc.get("study"),
                "unit_id": u.get("unit_id"),
                "gsm": u.get("gsm"),
                "library": u.get("library"),
                "status": u.get("status"),
                "pool_type": (u.get("pool") or {}).get("type"),
                "in_campaign_sample_list": u.get("in_campaign_sample_list"),
                "tier": tier,
            }
        )
    cols = ["study", "unit_id", "gsm", "library", "status", "pool_type", "in_campaign_sample_list", "tier"]
    df = pd.DataFrame(rows, columns=cols)
    df.attrs["tier"] = tier
    df.attrs["caveat"] = _caveat_for(tier)
    return df


# ================================================================================================================= objects
@dataclass
class Design:
    """The design of one study, read from its sidecar.

    ``factors``, ``levels`` and ``units`` are DataFrames (``units`` has one column per factor id holding the unit's level id),
    ``contrasts`` is a list of dicts as stored (each has ``grade``, ``status``, ``evidence`` and the counts behind it), ``tier``
    and ``provenance`` come from the sidecar. ``doc`` is the full sidecar. Printing it repeats the tier caveat."""

    gse: str
    tier: Optional[str]
    summary: Optional[str]
    design_class: Optional[str]
    flags: List[Any]
    factors: Any
    levels: Any
    units: Any
    contrasts: List[Dict[str, Any]]
    provenance: Dict[str, Any]
    doc: Dict[str, Any] = field(repr=False, default_factory=dict)
    source: str = ""

    @property
    def caveat(self) -> str:
        return _caveat_for(self.tier)

    def sample_status(self):
        """The status table of this study's units (see :func:`sample_status`)."""
        return _status_frame(self.doc)

    def __repr__(self) -> str:
        return (
            f"Design({self.gse}, tier={self.tier!r}, class={self.design_class!r}, {len(self.factors)} factors, "
            f"{len(self.units)} units, {len(self.contrasts)} contrasts)\n"
            f"  {self.summary or ''}\n  note: {self.caveat}"
        )


class ContrastList(list):
    """A list of contrast dicts, each tagged with its ``study`` and ``tier``. Printing it repeats the tier caveat;
    :meth:`to_frame` flattens the main fields into a DataFrame."""

    @property
    def tiers(self) -> List[str]:
        return sorted({c.get("tier") for c in self if c.get("tier")})

    @property
    def caveat(self) -> str:
        return CAVEAT if (not self.tiers or "candidate" in self.tiers) else _caveat_for(self.tiers[0])

    def to_frame(self):
        import pandas as pd

        cols = [
            "study", "id", "factor", "factor_type", "test", "reference", "grade", "status", "tier",
            "replication_unit", "pairing", "n_test_units", "n_reference_units",
        ]  # fmt: skip
        rows = []
        for c in self:
            n = c.get("n") or {}
            rows.append(
                {
                    "study": c.get("study"), "id": c.get("id"), "factor": c.get("factor"),
                    "factor_type": c.get("factor_type"), "test": c.get("test"), "reference": c.get("reference"),
                    "grade": c.get("grade"), "status": c.get("status"), "tier": c.get("tier"),
                    "replication_unit": c.get("replication_unit"), "pairing": c.get("pairing"),
                    "n_test_units": n.get("test_units"), "n_reference_units": n.get("reference_units"),
                }
            )  # fmt: skip
        df = pd.DataFrame(rows, columns=cols)
        df.attrs["caveat"] = self.caveat
        return df

    def __repr__(self) -> str:
        return f"{list.__repr__(self)}\n  {len(self)} contrast(s); note: {self.caveat}"


# ================================================================================================================= readers
def design(
    gse: str,
    base: Optional[str] = None,
    *,
    transport: Optional[Transport] = None,
) -> Design:
    """Read the design sidecar of study ``gse`` (``s1/d/<GSE>.design.json``) and return a :class:`Design`.

    ``base`` is the data host (default ``$SINGLET_DATA_BASE`` or ``https://data.singlet.bio``; use
    ``https://beta-data.singlet.bio`` for Beta, or a directory with the server layout). Raises :class:`DesignNotFound` when the
    study has no sidecar. The tier is ``candidate`` until a person has checked the design: see :data:`CAVEAT`."""
    gse = _normalize(gse)
    doc, source = _fetch(gse, base, transport)
    factors, levels, units = _frames(doc)
    return Design(
        gse=gse,
        tier=doc.get("tier"),
        summary=doc.get("summary"),
        design_class=doc.get("design_class"),
        flags=list(doc.get("flags") or []),
        factors=factors,
        levels=levels,
        units=units,
        contrasts=[dict(c) for c in (doc.get("contrasts") or [])],
        provenance=dict(doc.get("provenance") or {}),
        doc=doc,
        source=source,
    )


def _as_list(v: Union[None, str, Iterable[str]]) -> Optional[List[str]]:
    if v is None:
        return None
    if isinstance(v, str):
        return [v]
    return list(v)


def contrasts(
    studies: Union[str, Sequence[str]],
    factor_type: Union[None, str, Sequence[str]] = None,
    min_grade: Optional[str] = None,
    tier: Union[None, str, Sequence[str]] = None,
    base: Optional[str] = None,
    *,
    missing: str = "warn",
    transport: Optional[Transport] = None,
) -> ContrastList:
    """Contrasts of the studies you name, filtered client-side from their sidecars (one GET per study; no catalog endpoint).

    ``factor_type``: one type or several (``"time"``, ``"disease"``, ``"treatment"``, ``"tissue_site"``, ...). ``min_grade``: ``"A"``
    to ``"D"``, keeps contrasts at that grade or better (``"B"`` keeps A and B). ``tier``: one tier or several
    (``"confirmed"``, ``"candidate"``, ``"unresolved"``) of the study sidecar; the default keeps every tier and every
    contrast carries its own, so check it. ``missing``: what to do about a study without a sidecar, ``"warn"`` (default),
    ``"skip"`` or ``"error"``.

    Each returned dict is the stored contrast plus ``study`` and ``tier``."""
    if missing not in ("warn", "skip", "error"):
        raise ValueError("missing must be 'warn', 'skip' or 'error'")
    if min_grade is not None:
        min_grade = str(min_grade).strip().upper()
        if min_grade not in GRADES:
            raise ValueError(f"min_grade must be one of {GRADES}, got {min_grade!r}")
    ftypes = _as_list(factor_type)
    tiers = _as_list(tier)
    if tiers is not None:
        bad = [t for t in tiers if t not in TIERS]
        if bad:
            raise ValueError(f"unknown tier {bad}; tiers are {TIERS}")
    ids = [studies] if isinstance(studies, str) else list(studies)
    out = ContrastList()
    for s in ids:
        gse = _normalize(s)
        try:
            doc, _ = _fetch(gse, base, transport)
        except DesignNotFound as e:
            if missing == "error":
                raise
            if missing == "warn":
                warnings.warn(str(e), RuntimeWarning, stacklevel=2)
            continue
        st = doc.get("tier")
        if tiers is not None and st not in tiers:
            continue
        for c in doc.get("contrasts") or []:
            if ftypes is not None and c.get("factor_type") not in ftypes:
                continue
            if min_grade is not None:
                g = c.get("grade")
                if g not in GRADES or GRADES.index(g) > GRADES.index(min_grade):
                    continue
            d = dict(c)
            d["study"] = gse
            d["tier"] = st
            out.append(d)
    return out


def sample_status(
    gse: str,
    base: Optional[str] = None,
    *,
    transport: Optional[Transport] = None,
):
    """Processing status of each unit of study ``gse`` from its sidecar, as a DataFrame (``study``, ``unit_id``, ``gsm``,
    ``library``, ``status``, ``pool_type``, ``in_campaign_sample_list``, ``tier``). ``status`` is ``scheduled``, ``held``,
    ``running``, ``counts_available`` or ``needs_debug``. It is the status when the sidecar was written, so a study that
    has finished since can read ``scheduled``; ``open_study`` is the check for what is available."""
    return design(gse, base, transport=transport).sample_status()
