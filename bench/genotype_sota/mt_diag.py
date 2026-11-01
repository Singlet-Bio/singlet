"""Dev-lane (kang2018_ctrl only) diagnosis of Singlet's per-donor mt variant calls against the replicate-lane truth haplotypes.
Counts only: for true positives, false positives (called, covered, not in truth) and false negatives (truth, not called), by donor-level
features of the call: pseudobulk depth, alt fraction, strand balance, cells with alt, D-loop, homopolymer context, and whether the
mgatk+vireo haplotype has the call. No site, allele or haplotype is written.
    python mt_diag.py RUNS KANG_TRUTH_DIR OUT.tsv   (RUNS = $WORK/bench-public/runs)
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/bench-public/code_gsota/public/genotype/lib")
sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/bench-public/code_gsota/public/capabilities")
import mt_eval as M, pools as P, demux_eval as DE

POOL = "kang2018_ctrl"
runs, truth_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rd = P.rundir(runs, POOL); ref = M.read_ref(rd / "singlet_auto"); T = P.load_truth(truth_dir, POOL)

def vcf(path):
    rec = {}
    for l in open(path):
        if l[0] == "#": continue
        f = l.rstrip("\n").split("\t"); info = dict(x.split("=") for x in f[7].split(";") if "=" in x)
        sb = [int(x) for x in info.get("SB", "0,0").split(",")]
        rec[(int(f[1]) - 1, M.BASES.index(f[4].strip()[0]))] = dict(dp=int(info["DP"]), af=float(info["AF"]), sbal=min(sb) / max(sum(sb), 1), nc=int(info.get("NC", 0)))
    return rec

def homo(p):  # longest run of the reference base at p
    a = b = p
    while a > 0 and ref[a - 1] == ref[p]: a -= 1
    while b < len(ref) - 1 and ref[b + 1] == ref[p]: b += 1
    return b - a + 1

mg = M.load_mgatk(rd, POOL); reps = {r: M.load_mgatk(P.rundir(runs, r), r) for r in P.REPLICATES[POOL]}
truth_h = {}
for rep, cm in reps.items():
    Tr = P.load_truth(truth_dir, rep)
    for d, g in Tr[Tr["multiplets"] == "singlet"].groupby("ind"):
        if d != "<NA>" and len(g) >= DE.MIN_TRUTH_DONOR_CELLS: truth_h[d] = M.haplotype(cm.pseudobulk(g.index.tolist()), ref)
lab = DE.read_singlet(rd / "singlet_auto")["label"]; mp = M.cluster_map(T, lab)
labv = DE.read_vireo(rd / "vireo" / "told_s1")["label"]; mpv = M.cluster_map(T, labv)
inv_v = {td: cl for cl, td in mpv.items()}
rows = []
for cl, td in mp.items():
    if td not in truth_h: continue
    Hh, cov = truth_h[td]; S = vcf(rd / "singlet_auto" / "private" / "mt" / f"{cl}_mt_variants.vcf")
    mgv = M.haplotype(mg.pseudobulk(labv[labv == inv_v[td]].index.tolist()), ref)[0] if td in inv_v else set()
    for k, v in S.items():
        if k[0] not in cov: continue
        cls = "TP" if k in Hh else "FP"
        rows.append(dict(cls=cls, **v, dloop=int(k[0] < 576 or k[0] >= 16023), homo=homo(k[0]), mgatk=int(k in mgv)))
    for k in Hh - set(S):
        rows.append(dict(cls="FN", dp=np.nan, af=np.nan, sbal=np.nan, nc=np.nan, dloop=int(k[0] < 576 or k[0] >= 16023), homo=homo(k[0]), mgatk=int(k in mgv)))
d = pd.DataFrame(rows)
d["dp_bin"] = pd.cut(d.dp, [0, 30, 60, 120, 250, 1e9]); d["af_bin"] = pd.cut(d.af, [0, .5, .8, .9, .95, 1.01]); d["sb_bin"] = pd.cut(d.sbal, [-.01, .05, .15, .5])
d["nc_bin"] = pd.cut(d.nc, [-1, 5, 15, 40, 1e9]); d["homo_bin"] = pd.cut(d.homo, [0, 1, 3, 100], labels=["1", "2-3", "4+"])
res = []
for col in ["dp_bin", "af_bin", "sb_bin", "nc_bin", "dloop", "homo_bin", "mgatk"]:
    t = d.groupby([col, "cls"], observed=True).size().unstack(fill_value=0)
    for c in ("TP", "FP", "FN"):
        if c not in t: t[c] = 0
    t["feature"] = col; t = t.reset_index().rename(columns={col: "bin"}); res.append(t[["feature", "bin", "TP", "FP", "FN"]])
r = pd.concat(res); r["precision"] = r.TP / (r.TP + r.FP); r.to_csv(out, sep="\t", index=False)
print(d.cls.value_counts().to_dict()); print(r.to_string())
