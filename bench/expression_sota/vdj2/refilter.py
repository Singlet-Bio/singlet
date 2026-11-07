# SPDX-License-Identifier: MIT
"""Re-apply the contig filters (``singlet.vdj.assemble.filter_contigs``) to a finished run with other filter settings,
without re-assembling (contigs.csv holds every contig, kept or not).

    python -P refilter.py IN_DIR OUT_DIR [--param ambient_rescue=cell ...]

With no --param the output must equal the input's ``kept`` column (checked and printed: a test that the refactored filter
reproduces the assembler's).
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd


def tf(x):
    return str(x).strip().lower() in ("true", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--param", action="append", default=[])
    a = ap.parse_args()
    from singlet.vdj import assemble as A
    from singlet.vdj import clonotype as CL

    run = json.load(open(os.path.join(a.inp, "run.json")))
    params = A.Params()
    for k, v in run["params"].items():
        if hasattr(params, k):
            t = type(getattr(params, k))
            setattr(params, k, t(v) if t is not bool else tf(v))
    for kv in a.param:
        k, v = kv.split("=", 1)
        t = type(getattr(params, k))
        setattr(params, k, v.lower() in ("1", "true", "yes") if t is bool else t(v))
    d = pd.read_csv(os.path.join(a.inp, "contigs.csv"), keep_default_na=False, na_values=[""])
    for c in ("kept", "productive", "high_confidence", "full_length"):
        d[c] = d[c].map(tf)
    for c in ("v_gene", "d_gene", "j_gene", "c_gene"):
        d[c] = d[c].fillna("")
    d["bcix"] = d["barcode"]
    f = A.filter_contigs(d.drop(columns=["kept"]), params)
    f["contig_id"] = f["barcode"] + "_contig_" + (f.groupby("bcix").cumcount() + 1).astype(str)
    old = dict(zip(d.barcode + "|" + d.chain + "|" + d.cdr3_nt.astype(str), d.kept))
    changed = int(sum(k != old.get(c) for c, k in zip(f.barcode + "|" + f.chain + "|" + f.cdr3_nt.astype(str), f.kept)))
    os.makedirs(a.out, exist_ok=True)
    f[A.CONTIG_COLUMNS].to_csv(os.path.join(a.out, "contigs.csv"), index=False)
    cells = CL.assign_clonotypes(CL.cell_table(f[A.CONTIG_COLUMNS]))
    cells.to_csv(os.path.join(a.out, "cells.csv"), index=False)
    run["params"] = vars(params)
    run["refiltered_from"] = os.path.abspath(a.inp)
    run["kept_changed"] = changed
    with open(os.path.join(a.out, "run.json"), "w") as fh:
        json.dump(run, fh, indent=1, default=str)
    print(a.out, "contigs", len(f), "kept", int(f.kept.sum()), "was", int(d.kept.sum()), "changed", changed)


if __name__ == "__main__":
    main()
