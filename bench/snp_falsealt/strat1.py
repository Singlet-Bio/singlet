"""PRIVATE (Clipper). Stratify Singlet donor-VCF false alts on the Kang stim lane. Writes counts only.
Usage: strat1.py RUNS TRUTH_DIR VARIANT_RUN_DIR OUT.txt POOL"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
LIB = "/mnt/projects/debruinz_project/debruinz/singlet/bench-public/code/public/genotype/lib"
sys.path.insert(0, LIB); sys.path.insert(0, LIB + "/../../capabilities")
import snp_concordance as S, genotypes as G, pools as P, demux_eval as DE

runs, tdir, var, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
pool = sys.argv[5]; rd = P.rundir(var, pool)
T = P.load_truth(tdir, pool)
tg = S.truth_genotypes(runs, tdir, pool)
lab = DE.read_singlet(rd / "singlet_auto")  # same labels as scorer
mapping = S.match_by_cells(T, lab["label"])
rows = []
for f in sorted((rd / "singlet_auto/private/donor").glob("donor?.vcf")):
    cl = f.stem
    if cl not in mapping: continue
    keys, gt, dp = G.read_donor_vcf(f)
    # raw AD / AF from the vcf
    ad, af = [], []
    with open(f) as fh:
        for l in fh:
            if l[0] == "#": continue
            x = l.split("\t"); p = x[9].split(":")
            af.append(float(p[1])); ad.append(int(p[2].split(",")[1]))
    ad = np.array(ad); af = np.array(af); gt = gt[0]; dp = dp[0]
    idx = {k: i for i, k in enumerate(keys)}
    tkeys, tgt = tg[mapping[cl]]
    sel = np.array([idx.get(k, -1) for k in tkeys]); h = sel >= 0
    d = pd.DataFrame(dict(key=np.array(tkeys)[h], t=tgt[h], c=gt[sel[h]], dp=dp[sel[h]], ad=ad[sel[h]], af=af[sel[h]]))
    d["donor"] = cl; rows.append(d[(d.t >= 0) & (d.c >= 0)])
d = pd.concat(rows); d["sub"] = d.key.str.split(":").str[2] + ">" + d.key.str.split(":").str[3]
d["chrom"] = d.key.str.split(":").str[0]
d["frac"] = d.ad / d.dp.clip(lower=1)
# alternative caller on the same aggregated counts: fixed binomial model (truth's)
d["c_bin"] = G.genotype_from_counts(d.ad.values, d.dp.values)
o = open(out, "w")
def P_(*a): print(*a, file=o, flush=True)
h10 = d[(d.dp >= 10) & (d.t == 0)]
P_("hom-ref truth sites, tool dp>=10:", len(h10), "false-alt(c>=1) =", round((h10.c >= 1).mean(), 4))
fa = h10[h10.c >= 1]
P_("false alts:", len(fa), " het:", (fa.c == 1).sum(), " hom-alt:", (fa.c == 2).sum())
P_("false alts with ad==0 (no alt read at all):", (fa.ad == 0).sum(), "ad==1:", (fa.ad == 1).sum(), "ad==2:", (fa.ad == 2).sum(), "ad>=3:", (fa.ad >= 3).sum())
P_("binomial-model caller on same AD/DP (confident only): false-alt among called =", round((h10.c_bin[h10.c_bin >= 0] >= 1).mean(), 4), " called frac", round((h10.c_bin >= 0).mean(), 4))
P_("\n-- by dp bin (hom-ref truth, dp>=10): n, false-alt, mean af_vb of false alts, frac with ad==0")
for lo, hi in [(10, 15), (15, 25), (25, 50), (50, 100), (100, 10**9)]:
    s = h10[(h10.dp >= lo) & (h10.dp < hi)]
    if len(s): P_(f"dp[{lo},{hi})", len(s), round((s.c >= 1).mean(), 4), round((s[s.c >= 1].af).mean(), 3) if (s.c >= 1).any() else "-", round((s[s.c >= 1].ad == 0).mean(), 3) if (s.c >= 1).any() else "-")
P_("\n-- by alt reads ad (hom-ref truth, dp>=10): n, false-alt")
for a in [0, 1, 2, 3, 4, 5]:
    s = h10[h10.ad == a] if a < 5 else h10[h10.ad >= 5]
    P_("ad", a, len(s), round((s.c >= 1).mean(), 4) if len(s) else "-")
P_("\n-- by alt fraction bin: n, false-alt")
for lo, hi in [(0, 1e-9), (1e-9, .02), (.02, .05), (.05, .1), (.1, .2), (.2, 1.01)]:
    s = h10[(h10.frac >= lo) & (h10.frac < hi)]
    P_(f"frac[{lo:g},{hi:g})", len(s), round((s.c >= 1).mean(), 4) if len(s) else "-")
P_("\n-- by substitution (hom-ref, dp>=10): n, false-alt, share of all false alts")
tot = len(fa)
for s_, g_ in h10.groupby("sub"):
    P_(s_, len(g_), round((g_.c >= 1).mean(), 4), round((g_.c >= 1).sum() / tot, 3))
P_("\n-- by chrom group: n, false-alt")
for c_, g_ in h10.groupby("chrom"):
    P_(c_, len(g_), round((g_.c >= 1).mean(), 4))
# also sensitivity of the VB-AF cut itself
P_("\n-- AF cut sweep, hom-ref dp>=10: cut, false-alt ; true alt (t>=1, dp>=10) retained")
al = d[(d.dp >= 10) & (d.t >= 1)]
for cut in [.1, .15, .2, .25, .3, .4]:
    P_(cut, round((h10.af >= cut).mean(), 4), round((al.af >= cut).mean(), 4))
P_("\n-- ad-based rule: call alt only if ad>=m, hom-ref dp>=10 false-alt ; true alt kept")
for m in [1, 2, 3, 4]:
    P_(m, round(((h10.c >= 1) & (h10.ad >= m)).mean(), 4), round(((al.c >= 1) & (al.ad >= m)).sum() / max((al.c >= 1).sum(), 1), 4))
d.drop(columns=["key"]).to_pickle(out + ".pkl")  # private, no site ids
