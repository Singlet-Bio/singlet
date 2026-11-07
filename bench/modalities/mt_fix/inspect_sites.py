"""Per-donor allele fractions of the caller's sites, from the independent pysam pileup + truth donors.
Usage: inspect_sites.py TRUTH REF_PILEUP.npz HARNESS_OUT OUT.tsv
Joins private/mt_sites.tsv (all evaluated candidates + class + fail bits) with per-truth-donor pseudobulk AF of the alt
allele, so heteroplasmic calls can be checked for donor specificity (true heteroplasmy is a property of a donor)."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_truth, norm_bc

truth, npz, hdir, out = sys.argv[1:5]
T = load_truth(truth)
z = np.load(npz, allow_pickle=True)
cells = [norm_bc(c) for c in z["cells"]]
ref = pd.DataFrame(z["arr"], columns=["ci", "pos0", "A", "C", "G", "T"])
ref["bc"] = [cells[i] for i in ref.ci]
ref["ind"] = ref.bc.map(T.ind)
ref = ref[ref.bc.map(T.multiplets) == "singlet"].dropna(subset=["ind"])
S = pd.read_csv(os.path.join(hdir, "private", "mt_sites.tsv"), sep="\t")
S["pos0"] = S.pos - 1
donors = sorted(ref.ind.unique())
pb = {d: g.groupby("pos0")[list("ACGT")].sum() for d, g in ref.groupby("ind")}
rows = []
for _, s in S.iterrows():
    af, dp = [], []
    for d in donors:
        t = pb[d]
        if s.pos0 in t.index:
            r = t.loc[s.pos0]
            tot = r.sum()
            dp.append(int(tot)); af.append(r[s.alt] / tot if tot else np.nan)
        else:
            dp.append(0); af.append(np.nan)
    a = np.array(af, float)
    d_ = dict(s)
    d_.update({"af_" + d: v for d, v in zip(donors, a)})
    d_["max_af"] = np.nanmax(a) if np.isfinite(a).any() else np.nan
    d_["min_af"] = np.nanmin(a) if np.isfinite(a).any() else np.nan
    d_["n_donors_af_ge_0.9"] = int((a >= 0.9).sum())
    d_["n_donors_af_ge_0.02"] = int((a >= 0.02).sum())
    rows.append(d_)
pd.DataFrame(rows).to_csv(out, sep="\t", index=False, float_format="%.4g")
print("wrote", out, len(rows))
