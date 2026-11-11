#!/usr/bin/env python3
"""Counts table of score_d52.py -> per (pool, tool, stratum): sites, concordance, non-reference concordance, false-alt rate (truth 0/0
called 0/1 or 1/1), pooled over the pool's donors; plus an overall row per tool. Markdown on stdout. Counts only."""
import sys
import pandas as pd

t = pd.read_csv(sys.argv[1], sep="\t")
c = [f"t{a}_c{b}" for a in range(3) for b in range(3)]


def row(g):
    s = g[c].sum()
    n = s.sum()
    conc = sum(s[f"t{a}_c{a}"] for a in range(3)) / max(n, 1)
    nr = s[["t1_c0", "t1_c1", "t1_c2", "t2_c0", "t2_c1", "t2_c2"]].sum()
    nrc = (s["t1_c1"] + s["t2_c2"]) / max(nr, 1)
    r0 = s[["t0_c0", "t0_c1", "t0_c2"]].sum()
    fa = (s["t0_c1"] + s["t0_c2"]) / max(r0, 1)
    return pd.Series({"donors": g["donor"].nunique(), "sites": int(n), "concordance": round(conc, 4),
                      "nonref_conc": round(nrc, 4), "false_alt": round(fa, 4)})


by = t.groupby(["pool", "tool", "stratum"]).apply(row).reset_index()
allp = t.groupby(["tool", "stratum"]).apply(row).reset_index().assign(pool="ALL")
out = pd.concat([by, allp])[["pool", "tool", "stratum", "donors", "sites", "concordance", "nonref_conc", "false_alt"]]
print(out.to_markdown(index=False))
