# SPDX-License-Identifier: MIT
"""CRISPR guides as store members of the GEX sample (``docs/format-store.md`` section 14), laid out like ``adt_counts.1pz``.

* ``crispr_counts.1pz`` (kind 1, pz2): guides x the sample's columns, u32 UMIs; rows ``Inline`` (guide ids, the order
  of ``crispr_guides.json``), columns ``Cells`` (column ``j`` is ``col`` ``j`` of the cells table, called or not: the
  uncalled columns are the ambient background the assignment test uses).
* ``crispr_guides.json`` (kind 3, stored): ``{"schema": "singlet-crispr-guides/1", "method", "geometry", "counting",
  "guides": [...], "library": {...}}``: per row ``row``, ``id``, ``name``, ``sequence``, ``target_gene``,
  ``target_gene_id``, ``kind`` and ``ambient_umis``; the protospacer geometry measured from the reads; the library QC.

Per-cell guide calls (Singlet's ambient-calibrated caller) are cell metadata: :func:`cell_labels` builds the feature
sidecar ``<GSM>.features.crispr.parquet`` (``col``, ``n_guides``, ``guide_ids``, ``target``, ``class``).
"""

from __future__ import annotations

import json
from typing import Dict, Sequence

import numpy as np

from singlet.adt.store import _csc_member

SCHEMA = "singlet-crispr-guides/1"
METHOD_VERSION = (
    "singlet-crispr/1 (protospacer by anchor or offset; cellranger-fb-logic counting; "
    "ambient-calibrated Poisson assignment alpha 1e-3/G, >= 3 UMIs, >= 5% of the cell)"
)
MEMBERS = ("crispr_counts.1pz", "crispr_guides.json")


def build_members(
    M,
    library,
    gex_barcodes: Sequence[str],
    ambient: np.ndarray,
    geometry: str,
    counting: Dict,
    qc: Dict,
) -> Dict[str, tuple]:
    ids = list(library.ids)
    if len(set(ids)) != len(ids):
        raise ValueError("guide ids must be unique")
    body, dg = _csc_member(M, ids, list(gex_barcodes))
    guides = [
        {
            "row": i,
            "id": ids[i],
            "name": library.names[i],
            "sequence": library.sequences[i],
            "target_gene": library.target_gene[i],
            "target_gene_id": library.target_gene_id[i],
            "kind": library.kind[i],
            "ambient_umis": int(ambient[i]),
        }
        for i in range(len(ids))
    ]
    lib = {
        k: qc[k]
        for k in (
            "verdict",
            "failed",
            "warned",
            "frac_valid_barcode",
            "frac_guide",
            "frac_cells_assigned",
            "multiple_rate",
            "cell_empty_ratio",
            "guides_detected",
        )
        if k in qc
    }
    doc = {
        "schema": SCHEMA,
        "method": METHOD_VERSION,
        "geometry": geometry,
        "counting": counting,
        "source": library.source,
        "guides": guides,
        "library": lib,
    }
    return {
        "crispr_counts.1pz": (
            "pz2",
            body,
            dg,
            {"modality": "crispr", "panel": "crispr_guides.json"},
        ),
        "crispr_guides.json": (
            "stored",
            json.dumps(doc, sort_keys=True, separators=(",", ":"), default=float).encode(),
            None,
            {"content": "application/json"},
        ),
    }


def cell_labels(labels, cols: np.ndarray):
    """``labels``: :func:`singlet.crispr.assign.summarize` of the called cells; ``cols``: their column numbers."""
    out = labels[["n_guides", "guide_ids", "target", "class"]].copy()
    out.insert(0, "col", np.asarray(cols, dtype=np.int64))
    return out.sort_values("col").reset_index(drop=True)


def build_sidecar(gsm: str, members: Dict[str, tuple]) -> bytes:
    from singlet.store.container import S2Writer

    w = S2Writer(gsm)
    for name in MEMBERS:
        if name not in members:
            continue
        kind, body, dg, meta = members[name]
        if kind == "pz2":
            w.add_matrix(name, body, dg, meta)
        else:
            w.add_stored(name, body, meta, flags=0)
    return w.build()


def read_members(buf: bytes):
    from singlet.store.codec import decode_any
    from singlet.store.container import S2Reader

    r = S2Reader(buf)
    out = {}
    if "crispr_counts.1pz" in r.members:
        csc = decode_any(r.read_member("crispr_counts.1pz"))
        out["crispr"] = (csc.to_scipy(), list(csc.rownames or []))
    if "crispr_guides.json" in r.members:
        out["guides"] = json.loads(r.legacy_blob("crispr_guides.json"))
    return out
