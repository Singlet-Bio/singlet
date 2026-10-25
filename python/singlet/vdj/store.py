# SPDX-License-Identifier: MIT
"""V(D)J as v2 members of the GEX sample (``docs/format-v2.md`` section 14), the ADT layout's rules.

Members:

* ``vdj_contigs.parquet`` (kind 3, stored): one row per assembled contig with junction support (kept or not), the
  columns of :data:`singlet.vdj.assemble.CONTIG_COLUMNS` plus ``col``: the column of the sample's ``cells`` table, -1
  for barcodes that are not columns (their contigs are kept: ambient receptor transcripts are evidence too). Contigs
  are raw data of the sample, like counts, so they live in the sample's immutable object.
* ``vdj_summary.json`` (kind 3, stored): ``{"schema": "singlet-vdj/1", "method", "reference", "library", "counting"}``:
  method version, the segment reference's provenance and digest, the libraries used, read statistics and the library
  QC verdict.

Per-cell labels (chains, pairing, clonotype) are cell metadata: :func:`cell_labels` builds the feature sidecar
``<GSM>.features.vdj.parquet`` (``docs/CELL-ARCHITECTURE.md`` section 3), keyed by ``col``.

The members go into the GEX sample object at migration, or into a sidecar object ``v2/x/<GSM>/vdj.<d16>.s2`` referenced as
``samples[i].extra_objects.vdj`` (the ADT route, ``docs/format-v2.md`` section 13).
"""

from __future__ import annotations

import io
import json
from typing import Dict, Sequence

import numpy as np

SCHEMA = "singlet-vdj/1"
METHOD_VERSION = "singlet-vdj/1 (15-mer segment index; 19-mer UMI-weighted junction graph; cb 1mm posterior 0.975; umi 1hd cellranger)"
MEMBERS = ("vdj_contigs.parquet", "vdj_summary.json")
LABEL_COLUMNS = ["col", "receptor", "paired", "clonotype_id", "clonotype_size", "excess_chains",
                 "TRA_cdr3", "TRB_cdr3", "TRG_cdr3", "TRD_cdr3", "IGH_cdr3", "IGK_cdr3", "IGL_cdr3",
                 "TRA_v", "TRB_v", "IGH_v", "IGK_v", "IGL_v", "IGH_c"]


def _parquet(df) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="zstd")
    return buf.getvalue()


def with_columns(df, gex_barcodes: Sequence[str]):
    pos = {b: i for i, b in enumerate(gex_barcodes)}
    pos.update({b.split("-")[0] + "-1": i for i, b in enumerate(gex_barcodes)})
    out = df.copy()
    out.insert(0, "col", [pos.get(b, -1) for b in out["barcode"]])
    return out


def build_members(contigs, gex_barcodes: Sequence[str], library: Dict, counting: Dict, reference: Dict) -> Dict[str, tuple]:
    """-> {name: ('stored', body bytes, None, meta)}."""
    c = with_columns(contigs[contigs["junction_umis"] >= 1], gex_barcodes)
    c = c.sort_values(["col", "barcode", "chain", "umis"], ascending=[True, True, True, False]).reset_index(drop=True)
    for k in ("productive", "full_length", "high_confidence", "kept"):
        c[k] = c[k].astype(bool)
    for k in ("umis", "junction_umis", "reads"):
        c[k] = c[k].astype(np.int32)
    doc = {"schema": SCHEMA, "method": METHOD_VERSION, "reference": reference, "counting": counting,
           "library": {k: v for k, v in library.items() if not isinstance(v, (dict,)) or k == "rules"}}
    return {
        "vdj_contigs.parquet": ("stored", _parquet(c), None, {"content": "application/vnd.apache.parquet", "modality": "vdj"}),
        "vdj_summary.json": ("stored", json.dumps(doc, sort_keys=True, separators=(",", ":"), default=float).encode(),
                             None, {"content": "application/json"}),
    }


def cell_labels(cells, gex_barcodes: Sequence[str]):
    """Per-cell feature sidecar frame (columns :data:`LABEL_COLUMNS`) for the cells that are columns."""
    c = with_columns(cells, gex_barcodes)
    c = c[c["col"] >= 0]
    for k in LABEL_COLUMNS:
        if k not in c:
            c[k] = None
    return c[LABEL_COLUMNS].sort_values("col").reset_index(drop=True)


def build_sidecar(gsm: str, members: Dict[str, tuple]) -> bytes:
    from singlet.v2.container import S2Writer

    w = S2Writer(gsm)
    for name in MEMBERS:
        if name in members:
            kind, body, dg, meta = members[name]
            w.add_stored(name, body, meta, flags=0)
    return w.build()


def read_members(buf: bytes):
    """Decode an object holding the V(D)J members: {'contigs': DataFrame, 'summary': dict}."""
    import pyarrow.parquet as pq

    from singlet.v2.container import S2Reader

    r = S2Reader(buf)
    out = {}
    if "vdj_contigs.parquet" in r.members:
        out["contigs"] = pq.read_table(io.BytesIO(r.legacy_blob("vdj_contigs.parquet"))).to_pandas()
    if "vdj_summary.json" in r.members:
        out["summary"] = json.loads(r.legacy_blob("vdj_summary.json"))
    return out
