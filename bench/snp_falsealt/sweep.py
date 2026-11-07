"""PRIVATE (Clipper). Rule sweep on strat1 pickles (counts only). Usage: sweep.py OUT.txt pkl [pkl...]"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/bench-public/code/public/genotype/lib")
import genotypes as G
o = open(sys.argv[1], "w")
def P_(*a): print(*a, file=o, flush=True)
for f in sys.argv[2:]:
    d = pd.read_pickle(f); d = d[d.dp >= 10]
    h = d[d.t == 0]; al = d[d.t >= 1]; het = d[d.t == 1]
    P_(f"== {f.split('/')[-1]}  hom-ref {len(h)}  true-alt {len(al)}")
    P_("rule | false-alt (of hom-ref, uncalled counted as not-alt) | true-alt kept as alt | het kept as het | overall conc (called) | called frac")

    def rep(name, cfun):
        ch = cfun(h); ca = cfun(al); ce = cfun(het); cd = cfun(d)
        P_(f"{name:30s} {(ch>=1).mean():.4f}  {(ca>=1).mean():.4f}  {(ce==1).mean():.4f}  {((cd==d.t.values)[cd>=0]).mean():.4f}  {(cd>=0).mean():.3f}")
    rep("current VB-AF 0.1/0.9", lambda x: x.c.values)
    for e in (0.005, 0.01, 0.02):
        for post in (0.9, 0.99):
            rep(f"binom e={e} post={post}", lambda x: G.genotype_from_counts(x.ad.values, x.dp.values, e, post))
    for e in (0.01,):
        for post in (0.5,):
            rep(f"binom e={e} argmax", lambda x: G.genotype_from_counts(x.ad.values, x.dp.values, e, post))
    for cut in (0.15, 0.2, 0.25):
        rep(f"VB-AF hom-ref cut {cut}", lambda x: np.where(x.af.values < cut, 0, x.c.values))
    for m in (2, 3):
        rep(f"VB gt + alt needs ad>={m}", lambda x: np.where(x.ad.values < m, 0, x.c.values))
    rep("raw frac cut 0.1 (ad/dp<0.1->0)", lambda x: np.where(x.ad.values / x.dp.values.clip(1) < 0.1, 0, x.c.values))
