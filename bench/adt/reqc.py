# SPDX-License-Identifier: MIT
"""Recompute QC for processed libraries from their saved counts (after a rule change; no reads are touched).

    python -P bench/adt/reqc.py --data $WORK/data/adt [--gsm GSM ...]

Rewrites ``qc.json``, ``concordance.tsv`` and ``cells.parquet`` of every library with ``status: ok`` and updates the
verdict in ``status.json``.
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd


def main():
    from singlet.adt import count as C
    from singlet.adt import gex as G
    from singlet.adt import qc as Q

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gsm", nargs="*")
    a = ap.parse_args()
    for st_path in sorted(glob.glob(os.path.join(a.data, "GSE*", "GSM*", "status.json"))):
        d = os.path.dirname(st_path)
        st = json.load(open(st_path))
        if st.get("status") != "ok" or (a.gsm and st["gsm"] not in a.gsm):
            continue
        old = json.load(open(os.path.join(d, "qc.json")))
        M, feats, ambient, _, meta = C.load_npz(os.path.join(d, "counts.npz"))
        pan = pd.read_csv(os.path.join(d, "panel.tsv"), sep="\t", dtype=str, keep_default_na=False)
        gx = G.read_legacy(st["gse"], meta["gex_gsm"])
        disc = st.get("discovery")
        res = Q.library_qc(M, pan, gx.called, old["stats"], ambient, gx.exon, gx.genes, disc)
        lib = res["library"]
        lib["barcode_overlap_head"] = st.get("barcode_overlap_head")
        keep = {k: old[k] for k in ("gsm", "gse", "gex_gsm", "lib_class", "stats")}
        with open(os.path.join(d, "qc.json"), "w") as f:
            json.dump({**keep, **lib}, f, indent=1, default=float)
        if res["concordance"] is not None:
            res["concordance"].to_csv(os.path.join(d, "concordance.tsv"), sep="\t", index=False)
        Q.cell_table(M, pan, res["cells"], res["hto"]).to_parquet(os.path.join(d, "cells.parquet"))
        st["verdict"], st["failed"] = lib["verdict"], lib["failed"]
        with open(st_path, "w") as f:
            json.dump(st, f, indent=1, default=str)
        print(st["gsm"], old.get("verdict"), "->", lib["verdict"], lib["failed"], lib.get("rna_concordance"),
              lib.get("rna_pairs"), flush=True)


if __name__ == "__main__":
    sys.exit(main())
