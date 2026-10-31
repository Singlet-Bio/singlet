# SPDX-License-Identifier: MIT
"""Development-library diagnostics only (never run on held-out libraries).

    python -P diag_dev.py RUNS DATA LIB:CHAINS [...]

1. For reference top contigs per (cell, chain): matched / method absent / method different junction, by reference V gene.
2. Method contigs on GEX cells of the wrong lineage: UMIs, chain, whether the reference has the same junction.
"""
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vdj"))
import compare_vdj as CV  # noqa: E402

runs, data = sys.argv[1:3]
for item in sys.argv[3:]:
    lib, chains = item.split(":")
    keep = CV.TCR if chains == "TCR" else CV.BCR
    ref = CV.load_cellranger(os.path.join(data, lib, "filtered_contig_annotations.csv"))
    ref = ref[ref.chain.isin(keep)]
    r = os.path.join(runs, lib)
    gex = set(l.strip() for l in open(os.path.join(r, "gex_cells.txt")))
    lab = dict(pd.read_csv(os.path.join(r, "gex_lineage.csv"))[["barcode", "lineage"]].values)
    allc = pd.read_csv(os.path.join(r, os.environ.get("SDIR", "singlet"), "contigs.csv"))
    m = CV.load_singlet(os.path.join(r, os.environ.get("SDIR", "singlet"), "contigs.csv"), os.path.join(r, os.environ.get("SDIR", "singlet"), "cells.csv"))
    m = m[m.chain.isin(keep) & m.barcode.isin(gex)]
    print("=====", lib, "ref contigs", len(ref), "method contigs on GEX cells", len(m))
    rt = ref.sort_values("umis", ascending=False).drop_duplicates(["barcode", "chain"])
    mt = m.sort_values("umis", ascending=False).drop_duplicates(["barcode", "chain"])
    mm = {(b, c): (n, v, j, u) for b, c, n, v, j, u in zip(mt.barcode, mt.chain, mt.cdr3_nt, mt.v, mt.j, mt.umis)}
    mall = m.groupby(["barcode", "chain"])["cdr3_nt"].agg(set).to_dict()
    cat = Counter()
    byv = {}
    jbad = Counter()
    for b, c, n, v, j in zip(rt.barcode, rt.chain, rt.cdr3_nt, rt.v, rt.j):
        got = mm.get((b, c))
        if got is None:
            k = "absent"
        elif got[0] == n:
            k = "match"
            if got[2] != j:
                jbad[(j, got[2])] += 1
        elif n in mall.get((b, c), ()):
            k = "ref junction present but not top"
        else:
            k = "different"
        cat[(c, k)] += 1
        byv.setdefault((c, v), Counter())[k] += 1
    print(sorted(cat.items()))
    bad = sorted(((sum(x for k, x in cnt.items() if k != "match"), key, dict(cnt)) for key, cnt in byv.items()), reverse=True)[:12]
    for t in bad:
        print("  worst V:", t)
    print("  J disagreements (ref, method):", jbad.most_common(8))
    # wrong lineage
    ll = m.barcode.map(lab)
    wrong = m[((m.chain.isin(CV.TCR)) & (ll == "B")) | ((m.chain.isin(CV.BCR)) & (ll == "T"))]
    rk = set(zip(ref.barcode, ref.chain, ref.cdr3_nt))
    print("  wrong-lineage method contigs:", len(wrong), "umis", sorted(wrong.umis.tolist()))
    print("   also in reference:", sum((b, c, n) in rk for b, c, n in zip(wrong.barcode, wrong.chain, wrong.cdr3_nt)))
    right = m[~m.index.isin(wrong.index)]
    print("   all-contig umis quantiles", right.umis.quantile([.01, .05, .25, .5]).tolist())
    cols = [c for c in allc.columns if c not in ("sequence",)]
    print("  contig columns:", cols)
