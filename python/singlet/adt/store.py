# SPDX-License-Identifier: MIT
"""ADT/HTO as v2 members (``docs/format-v2.md`` section 13, proposal in ``docs/adt-modality.md``).

Members of one sample:

* ``adt_counts.1pz`` (kind 1, pz2): antibodies and isotype controls x the GEX sample's columns, u32 UMIs; rows ``Inline``
  (the antibody names of ``adt_panel.json``, in order), columns ``Cells`` (the barcodes of the sample's ``cells`` table:
  column ``j`` is ``col`` ``j``). Columns with no ADT molecule are empty columns, so every column of the cells table has
  one, called or not (the uncalled ones are the empty droplets DSB needs).
* ``hto_counts.1pz`` (kind 1, pz2): hashtags, same layout. Present only when the library has hashtags.
* ``adt_panel.json`` (kind 3, stored): ``{"schema": "singlet-adt-panel/1", "features": [...], "library": {...}}``, one
  entry per matrix row (``matrix``, ``row``, ``name``, ``sequence``, ``kind``, ``barcode_id``, ``ensembl``,
  ``catalog_specificity``, ``clone``, ``source``) plus the counting facts (chemistry, R2 offset, translation, method
  version), the ambient profile and the library QC verdict.

The same members go either into the GEX sample object at migration (preferred while the campaign has not run) or into a
sidecar object ``v2/x/<GSM>/adt.<d16>.s2`` referenced by the study manifest (``samples[i].extra_objects.adt``), so readers
find them the same way.
"""

from __future__ import annotations

import json
from typing import Dict, List, Sequence

import numpy as np

PANEL_SCHEMA = "singlet-adt-panel/1"
METHOD_VERSION = (
    "singlet-adt-count/1 (cellranger-fb-logic; cb 1mm posterior 0.975; umi 1hd cellranger)"
)
MEMBERS = ("adt_counts.1pz", "hto_counts.1pz", "adt_panel.json")


def _csc_member(M, rownames: List[str], colnames: List[str]):
    """pz2 bytes and digest of a features x columns matrix with inline rows and columns from the cells table."""
    from singlet.v2.codec import EncodeOptions, get_codec
    from singlet.v2.matrix import Csc

    M = M.tocsc()
    M.sort_indices()
    csc = Csc(
        m=M.shape[0],
        n=M.shape[1],
        indptr=M.indptr.astype(np.int64),
        indices=M.indices.astype(np.uint32),
        data=M.data.astype(np.uint32),
        rownames=list(rownames),
        colnames=list(colnames),
    )
    why = csc.check_canonical()
    if why:
        raise ValueError("not canonical: %s" % why)
    body = get_codec().encode(csc, EncodeOptions(row_names="inline", col_names="cells"))
    return body, csc.digest()


def build_members(
    M, panel, gex_barcodes: Sequence[str], library: Dict, ambient: np.ndarray, counting: Dict
) -> Dict[str, tuple]:
    """-> {name: (kind, body bytes, digest16 or None, meta)}; kind 'pz2' or 'stored'."""
    kinds = np.asarray(panel["kind"])
    ab = np.flatnonzero(kinds != "hashtag")
    ht = np.flatnonzero(kinds == "hashtag")
    names = list(panel["name"])
    if len(set(names)) != len(names):
        raise ValueError("panel names must be unique")
    out = {}
    feats = []
    for mname, rows in (("adt_counts.1pz", ab), ("hto_counts.1pz", ht)):
        if len(rows) == 0:
            continue
        body, dg = _csc_member(M[rows], [names[i] for i in rows], list(gex_barcodes))
        out[mname] = (
            "pz2",
            body,
            dg,
            {"modality": "adt" if mname == "adt_counts.1pz" else "hto", "panel": "adt_panel.json"},
        )
        for r, i in enumerate(rows):
            p = panel.iloc[i]
            feats.append(
                {
                    "matrix": mname,
                    "row": r,
                    "name": names[i],
                    "sequence": p["sequence"],
                    "kind": p["kind"],
                    **{
                        k: (str(p[k]) if k in panel.columns and isinstance(p[k], str) else "")
                        for k in ("barcode_id", "ensembl", "catalog_specificity", "clone", "source")
                    },
                    "ambient_umis": int(ambient[i]),
                }
            )
    lib = {
        k: library.get(k)
        for k in (
            "verdict",
            "failed",
            "warned",
            "n_called",
            "median_umi_cells",
            "cell_empty_ratio",
            "frac_valid_barcode",
            "frac_feature",
            "saturation",
            "rna_concordance",
            "hto_singlet_share",
        )
        if k in library
    }
    doc = {
        "schema": PANEL_SCHEMA,
        "method": METHOD_VERSION,
        "counting": counting,
        "features": feats,
        "library": lib,
    }
    out["adt_panel.json"] = (
        "stored",
        json.dumps(doc, sort_keys=True, separators=(",", ":"), default=float).encode(),
        None,
        {"content": "application/json"},
    )
    return out


def build_sidecar(gsm: str, members: Dict[str, tuple]) -> bytes:
    """A small ``.s2`` object holding only the ADT members (for samples whose GEX object is already written)."""
    from singlet.v2.container import S2Writer

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


def read_sidecar(buf: bytes):
    """Decode an object holding the ADT members: {'adt': (scipy csc, row names), 'hto': ..., 'panel': dict}."""
    from singlet.v2.codec import decode_any
    from singlet.v2.container import S2Reader

    r = S2Reader(buf)
    out = {}
    for name, key in (("adt_counts.1pz", "adt"), ("hto_counts.1pz", "hto")):
        if name in r.members:
            csc = decode_any(r.read_member(name))
            out[key] = (csc.to_scipy(), list(csc.rownames or []))
    if "adt_panel.json" in r.members:
        out["panel"] = json.loads(r.legacy_blob("adt_panel.json"))
    return out
