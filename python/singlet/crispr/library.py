# SPDX-License-Identifier: MIT
"""One CRISPR guide library of the atlas -> guide counts, per-cell calls, QC verdict and the v2 members of its GEX sample.

    python -m singlet.crispr.library --r1 R1.fastq.gz [...] --r2 R2.fastq.gz [...] --guides feature_reference.csv
        --gex-columns cols.txt --gex-called called.txt --whitelist WL [--translation T] --gsm GSM... --out DIR

Writes ``qc.json`` and, when the library passes QC, ``<GSM>.crispr.s2`` (``crispr_counts.1pz`` + ``crispr_guides.json``)
and ``<GSM>.features.crispr.parquet``. Nothing is uploaded. A library without a guide list cannot be named and is
reported, not counted (``guides`` is required).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Optional, Sequence

import numpy as np

from . import assign as AS
from . import count as CC
from . import guides as G
from . import qc as Q
from . import store as ST


def process_library(r1: Sequence[str], r2: Sequence[str], *, library: G.GuideLibrary, gex_columns: Sequence[str],
                    gex_called: Sequence[str], whitelist_codes: np.ndarray, umi_len: int, translation=None,
                    geometry: Optional[G.Geometry] = None, sample_r2: Sequence[bytes] = (), shards: int = 0,
                    tmpdir: Optional[str] = None) -> Dict:
    L = max(len(s) for s in library.sequences)
    if geometry is None:
        det = G.detect_geometry(list(sample_r2), L, library.sequences)
        geometry = det["geometry"]
        if library.pattern:
            pg = G.parse_pattern(library.pattern, L)
            geometry = G.Geometry(L, pg.anchor, pg.offset, geometry.reverse if geometry.anchor == pg.anchor else False)
    lc = CC.count_library(r1, r2, library=library, geometry=geometry, cb_len=16, umi_len=umi_len,
                          whitelist_codes=whitelist_codes, gex_barcodes=gex_columns, translation=translation,
                          shards=shards, tmpdir=tmpdir)
    called = set(gex_called)
    cmask = np.array([c in called for c in gex_columns])
    unc = np.flatnonzero(~cmask)
    b_empty = (AS.ambient_load(np.asarray(lc.matrix[:, unc].sum(axis=0)).ravel()) if len(unc) >= 1000
               else float(np.median(lc.outside_totals)) if len(lc.outside_totals) else 0.0)
    cols = np.flatnonzero(cmask)
    calls = AS.call_ambient(lc.matrix[:, cols], AS.ambient_profile(lc.ambient), b_empty)
    labels = AS.summarize(calls, library.ids, library.target_gene)
    qc = Q.library_qc(lc, cmask, labels)
    counting = {"stats": lc.stats, "umi_len": umi_len, "ambient_load": b_empty}
    members = (ST.build_members(lc.matrix, library, gex_columns, lc.ambient, geometry.describe(), counting, qc)
               if qc["verdict"] != "fail" else None)
    return {"counts": lc, "labels": ST.cell_labels(labels, cols), "qc": qc, "members": members}


def _lines(path):
    with open(path) as f:
        return [x.strip().split(",")[0] for x in f if x.strip()]


def main(argv=None):
    import dnaio

    from singlet.adt import count as AC
    from singlet.adt import seq as S

    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", nargs="+", required=True)
    ap.add_argument("--r2", nargs="+", required=True)
    ap.add_argument("--guides", required=True, help="Cell Ranger feature reference CSV or a guide table TSV")
    ap.add_argument("--gex-columns", required=True)
    ap.add_argument("--gex-called", required=True)
    ap.add_argument("--whitelist", required=True)
    ap.add_argument("--translation", default=None)
    ap.add_argument("--umi-len", type=int, default=None)
    ap.add_argument("--gsm", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.guides.endswith(".csv"):
        lib = G.from_feature_reference(a.guides)
    else:
        with open(a.guides) as f:
            lib = G.from_table(csv.DictReader(f, delimiter="\t"), os.path.basename(a.guides))
    sample = []
    with dnaio.open(a.r1[0], file2=a.r2[0]) as f:
        for i, (x, y) in enumerate(f):
            sample.append((x.sequence.encode(), y.sequence.encode()))
            if i >= 50_000:
                break
    umi_len = a.umi_len or (12 if len(sample[0][0]) >= 28 else 10)
    cols = _lines(a.gex_columns)
    wl = AC.load_barcode_list(a.whitelist)
    codes, bad = S.encode_strings(wl, 16)
    trans = None
    if a.translation:
        gex_codes, _ = S.encode_strings([c.split("-")[0] for c in cols], 16)
        cb = S.encode_strings([x[:16] for x, _ in sample], 16)[0]
        trans, _ = AC.best_translation(cb, np.sort(gex_codes), a.translation)
    out = process_library(a.r1, a.r2, library=lib, gex_columns=cols, gex_called=_lines(a.gex_called),
                          whitelist_codes=np.sort(codes[bad == 0]), umi_len=umi_len, translation=trans,
                          sample_r2=[y for _, y in sample])
    with open(os.path.join(a.out, "qc.json"), "w") as f:
        json.dump(out["qc"], f, indent=1, default=float)
    if out["members"] is not None:
        with open(os.path.join(a.out, f"{a.gsm}.crispr.s2"), "wb") as f:
            f.write(ST.build_sidecar(a.gsm, out["members"]))
        out["labels"].to_parquet(os.path.join(a.out, f"{a.gsm}.features.crispr.parquet"), index=False)
    print(json.dumps({k: out["qc"][k] for k in ("verdict", "failed", "warned", "frac_cells_assigned") if k in out["qc"]},
                     default=float))


if __name__ == "__main__":
    main()
