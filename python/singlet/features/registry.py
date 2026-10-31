# SPDX-License-Identifier: MIT
"""Feature registry (catalog table ``cell_feature``, singlet-website ``schema/035_cell_tier.sql``) and per-sample summaries
(``sample_cell_summary``), as JSON rows for ``POST /api/ingest/cell_feature`` and ``POST /api/ingest/sample_cell_summary``.

Feature ids are ``<set>/<name>`` with set and name matching ``[a-z][a-z0-9_]{0,47}`` (the cell index's rule). The summaries count
cells on the cell index's u8 code (``floor(x * 254 + 0.5)``), so they equal what the index's own D1 export gives for the same file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import __version__
from .genesets import DATA, ORGANISMS, collection
from .rank import MAX_RANK

SCALE8 = 254
K_NULL = 16
MIN_GENES_DETECTED = 100
CATEGORY_NA = 65535


@dataclass
class Feature:
    name: str
    feature_set: str
    kind: str = "score"
    label: str = ""
    description: str = ""
    method: str = ""
    method_version: str = ""
    source_ref: str = ""
    licence: str = ""
    organisms: Tuple[str, ...] = ()
    ontology_id: str = ""
    categories: Tuple[str, ...] = ()
    thresholds: Tuple[float, ...] = (0.5, 0.8)
    comparable: str = "atlas"
    storage: str = "shard"

    @property
    def id(self) -> str:
        return f"{self.feature_set}/{self.name}"

    def cell_feature_row(self) -> dict:
        return {
            "feature_id": self.id,
            "feature_set": self.feature_set,
            "kind": self.kind,
            "label": self.label or self.name,
            "description": self.description,
            "method": self.method,
            "method_version": self.method_version,
            "source_ref": self.source_ref,
            "licence": self.licence,
            "organisms": json.dumps(list(self.organisms)),
            "ontology_id": self.ontology_id or None,
            "categories": json.dumps(list(self.categories)) if self.kind == "category" else None,
            "thresholds": json.dumps(list(self.thresholds)) if self.kind == "score" else "[]",
            "comparable": self.comparable,
            "storage": self.storage,
            "status": "active",
        }


@lru_cache(maxsize=None)
def _file_digest(name: str) -> str:
    return hashlib.sha256((DATA / name).read_bytes()).hexdigest()[:8]


def method_version(feature_set: str) -> str:
    """Everything that changes the values of a set: package version, rank cap, null construction (abundance references by
    digest), gene-set release and its file digest. (``doublet``: :func:`singlet.features.doublet.method_version`.)"""
    if feature_set == "doublet":
        from . import doublet as DBL

        return DBL.method_version()
    c = collection(feature_set)
    null = f"null=abundance50-k{K_NULL}(human:{_file_digest('abundance_human.parquet')},mouse:{_file_digest('abundance_mouse.parquet')})"
    return (
        f"singlet-features={__version__};rank-auc-z/max_rank={MAX_RANK},z_scale={Z_SCALE};{null};min_genes={MIN_GENES_DETECTED};"
        f"{c['collection']}={c['release']}({_file_digest(feature_set + '.json')})"
    )


SCORE_METHOD = "rank-auc-z"
# score = clip(z / Z_SCALE, 0, 1): thresholds 0.3 and 0.5 are z >= 3 and z >= 5 against the cell's own null sets
Z_SCALE = 10.0
SCORE_THRESHOLDS = (0.3, 0.5)
SCORE_DESC = (
    "z of the set's rank AUC (UCell-like, among the cell's top {max_rank} genes) against {k} abundance-matched random gene "
    "sets scored in the same cell, divided by {zs:g} and clipped to [0, 1]: 0 = no higher than random genes of the same "
    "abundance, 0.3 = z 3, 0.5 = z 5."
)


def _source_ref(c: dict, d: dict) -> str:
    ids = ", ".join(f"{o} {v}" for o, v in d.get("source_id", {}).items())
    base = (
        f"MSigDB {c['release']} {d['source_name']}"
        if c["collection"] == "hallmark"
        else c["attribution"]
    )
    return f"{base} ({ids})" if ids else base


@lru_cache(maxsize=None)
def features(feature_set: str) -> Tuple[Feature, ...]:
    """Registered features of a set, in sidecar column order."""
    if feature_set == "doublet":
        from . import doublet as DBL

        return tuple(DBL.features())
    c = collection(feature_set)
    mv = method_version(feature_set)
    orgs = tuple(
        ORGANISMS[o] for o in ORGANISMS if any(d["genes"].get(o) for d in c["sets"].values())
    )
    desc = SCORE_DESC.format(max_rank=MAX_RANK, k=K_NULL, zs=Z_SCALE)
    out: List[Feature] = []
    if feature_set == "hallmark":
        for name, d in c["sets"].items():
            out.append(
                Feature(
                    name=name,
                    feature_set=feature_set,
                    label="Hallmark " + name.replace("_", " "),
                    description=desc,
                    method=SCORE_METHOD,
                    method_version=mv,
                    source_ref=_source_ref(c, d),
                    licence=c["licence"],
                    organisms=orgs,
                    thresholds=SCORE_THRESHOLDS,
                )
            )
    elif feature_set == "cell_cycle":
        common = dict(
            feature_set=feature_set, method_version=mv, licence=c["licence"], organisms=orgs
        )
        out.append(
            Feature(
                name="phase",
                kind="category",
                label="Cell-cycle phase",
                description=(
                    "S or G2M when that phase's score exceeds its abundance-matched random sets in the same cell by z >= "
                    f"{PHASE_Z} (the larger z wins); G1 otherwise (not cycling: G0 or G1)."
                ),
                method="rank-auc-null-z",
                source_ref=c["attribution"] + ", S and G2/M gene lists",
                categories=PHASES,
                ontology_id="GO:0007049",
                **common,
            )
        )
        for name, label, src in (
            ("s_score", "S-phase score", "s_phase"),
            ("g2m_score", "G2/M-phase score", "g2m_phase"),
        ):
            out.append(
                Feature(
                    name=name,
                    label=label,
                    description=desc,
                    method=SCORE_METHOD,
                    source_ref=f"{c['attribution']}, {src.replace('_', ' ')} genes",
                    thresholds=SCORE_THRESHOLDS,
                    **common,
                )
            )
    else:
        raise KeyError(feature_set)
    return tuple(out)


PHASES = ("G1", "S", "G2M")
PHASE_Z = 3.0
FEATURE_SETS = ("hallmark", "cell_cycle")
# sets that need more than the counts (the engine's doublet score, the cell type network, GPU or many cores): run on request
# with ``features_run.py run --sets doublet``, never part of the default
EXTRA_SETS = ("doublet",)
ALL_SETS = FEATURE_SETS + EXTRA_SETS


def registry_rows(sets: Sequence[str] = FEATURE_SETS) -> List[dict]:
    return [f.cell_feature_row() for s in sets for f in features(s)]


# --------------------------------------------------------------------------------------------------------------- summaries
def _q8(x: np.ndarray) -> np.ndarray:
    return np.floor(np.clip(x, 0, 1) * SCALE8 + 0.5).astype(np.int64)


def _order_stats(x: np.ndarray):
    xs = np.sort(x)
    n = len(xs)
    return (
        round(float(xs[int(np.floor(0.5 * (n - 1)))]), 6),
        round(float(xs[int(np.floor(0.9 * (n - 1)))]), 6),
        round(float(xs.mean()), 6),
    )


def summary_rows(
    gsm: str, gse: str, values: Dict[str, np.ndarray], feats: Sequence[Feature], version: str
) -> List[dict]:
    """``sample_cell_summary`` rows of one sample and set: per score, cells with a value, cells at or above the two thresholds
    and order statistics (on the u8 code the index stores); per category value, ``item = <feature id>=<category>`` with its cell
    count."""
    rows: List[dict] = []
    base = {"gsm_id": gsm, "gse_id": gse, "subject": "feature", "version": version}
    for f in feats:
        v = values.get(f.name)
        if v is None:
            continue
        if f.kind == "category":
            v = np.asarray(v, np.int64)
            for ci, cname in enumerate(f.categories):
                k = int((v == ci).sum())
                if k:
                    rows.append(
                        {
                            **base,
                            "item": f"{f.id}={cname}",
                            "n_cells": k,
                            "n_hi1": None,
                            "n_hi2": None,
                            "q50": None,
                            "q90": None,
                            "mean": None,
                        }
                    )
            continue
        x = np.asarray(v, np.float64)
        x = x[np.isfinite(x)]
        if not len(x):
            continue
        code = _q8(x)
        thr = (list(f.thresholds) + [None, None])[:2]
        his = [
            None if t is None else int((code >= int(np.floor(t * SCALE8 + 0.5))).sum()) for t in thr
        ]
        q50, q90, mean = _order_stats(code / SCALE8)
        rows.append(
            {
                **base,
                "item": f.id,
                "n_cells": int(len(x)),
                "n_hi1": his[0],
                "n_hi2": his[1],
                "q50": q50,
                "q90": q90,
                "mean": mean,
            }
        )
    return rows
