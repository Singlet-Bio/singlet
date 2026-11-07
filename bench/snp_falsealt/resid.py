"""PRIVATE (Clipper). Residual false alts after R12, by substitution/dp/ad. Counts only. Usage: resid.py OUT pkl..."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/bench-public/code/public/genotype/lib")
import genotypes as G
d = pd.concat([pd.read_pickle(f) for f in sys.argv[2:]]); d = d[(d.dp >= 10)]
d["c12"] = G.genotype_from_counts(d.ad.values, d.dp.values, 0.01, 0.9)
h = d[(d.t == 0) & (d.c12 >= 0)]
o = open(sys.argv[1], "w")
def P_(*a): print(*a, file=o)
old = d[d.t == 0]
P_("pooled Kang stim+ctrl+b1 a/b/c, hom-ref truth, dp>=10: sites", len(old), "old false-alt", round((old.c >= 1).mean(), 4),
   "| R12 called", len(h), "false-alt among called", round((h.c12 >= 1).mean(), 4), "residual n", int((h.c12 >= 1).sum()))
fa = h[h.c12 >= 1]
P_("residual by ad:", fa.ad.value_counts().sort_index().to_dict())
P_("residual by dp bin:", {f"{lo}-{hi}": int(((fa.dp >= lo) & (fa.dp < hi)).sum()) for lo, hi in [(10, 15), (15, 25), (25, 50), (50, 100), (100, 10**9)]})
P_("sub: n_homref, resid_rate, share_of_resid  (A>G/T>C = editing-like)")
for s, g in h.groupby("sub"):
    P_(s, len(g), round((g.c12 >= 1).mean(), 4), round((g.c12 >= 1).sum() / max(len(fa), 1), 3))
ed = h["sub"].isin(["A>G", "T>C"])
P_("editing-like A>G/T>C: rate", round((h[ed].c12 >= 1).mean(), 4), " other:", round((h[~ed].c12 >= 1).mean(), 4))
# where the old call was false-alt: how many removed by R12 outright (c12==0 or uncalled)
fo = old[old.c >= 1]
P_("old false alts", len(fo), "-> R12 0/0:", int((fo.c12 == 0).sum()), " uncalled:", int((fo.c12 < 0).sum()), " still alt:", int((fo.c12 >= 1).sum()))
