# SPDX-License-Identifier: MIT
"""PREREG sections 2-4, 7: every caller on the independent sets, once.

    python -P eval_independent.py --data DIR --i1-truth vireo/donor_ids.tsv --i1-cells singlet/cell_calls.tsv \
        --frozen frozen.json --out results/independent.json [--only I1|I2]

I1  Stoeckius 2018 PBMC, 8 donors: deposited HTO table x cells called by Singlet, truth = vireo genotype donors.
I2  Stoeckius 2018 mixed cell lines (12 tags, three per line): deposited HTO table x deposited RNA cells, truth from RNA only
    (cluster -> cell line by marker genes; cross-line doublets by a scrublet-style simulated-doublet kNN score).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hto_callers as H  # noqa: E402

N_BOOT = 1000


# ------------------------------------------------------------------------------------------------------------- I1
def load_i1(a):
    T = pd.read_csv(a.i1_truth, sep="\t")
    cells = pd.read_csv(a.i1_cells, sep="\t")
    flag = cells.columns[1]
    called = cells[cells[flag].astype(str).str.upper() == "TRUE"].iloc[:, 0].astype(str).str.replace(r"-1$", "", regex=True)
    H_ = pd.read_csv(os.path.join(a.data, "geo", "GSM2895283_Hashtag-HTO-count.csv.gz"), index_col=0)
    tagrows = [r for r in H_.index if r.startswith("Batch")]
    H_ = H_.loc[tagrows]
    H_.columns = [str(c).replace("-1", "") for c in H_.columns]
    X = H_.reindex(columns=list(called)).fillna(0)
    t = T.iloc[:, :2].copy()
    t.columns = ["cell", "donor"]
    t["cell"] = t["cell"].astype(str).str.replace(r"-1$", "", regex=True)
    t = t.drop_duplicates("cell").set_index("cell")["donor"].reindex(called)
    truth = np.where(t.isna(), "unassigned", t.values).astype(object)
    truth = np.where(truth == "doublet", "Doublet", truth)
    return X.to_numpy(float), list(X.index), truth, {"called": len(called), "in_hto_table": int(called.isin(H_.columns).sum())}


def tag_map(call, truth, ntags):
    """Hungarian one-to-one tag -> donor on vireo singlets (the top tag of each cell, caller-free: argmax of the counts)."""
    donors = sorted({d for d in truth if d not in ("Doublet", "unassigned", "Negative")})
    return donors


def hungarian(X, truth):
    donors = sorted({d for d in truth if d not in ("Doublet", "unassigned", "Negative")})
    sing = np.isin(truth, donors)
    top = X[:, sing].argmax(axis=0)
    M = np.zeros((X.shape[0], len(donors)))
    for i, d in zip(top, truth[sing]):
        M[i, donors.index(d)] += 1
    r, c = linear_sum_assignment(-M)
    return {int(i): donors[j] for i, j in zip(r, c)}, M


# ------------------------------------------------------------------------------------------------------------- I2
MARKERS = {  # fixed before the run; used only to name the four RNA clusters
    "K562": ["HBG1", "HBG2", "HBZ", "HBE1", "GYPA", "GATA1", "HBA1"],
    "THP1": ["CD14", "CSF1R", "CD68", "LYZ", "S100A8", "S100A9", "FCER1G", "IL1B"],
    "KG1": ["CD34", "KIT", "MPO", "AZU1", "ELANE", "PRTN3", "CD33", "MS4A3"],
    "HEK": ["KRT8", "KRT18", "NEFM", "NEFL", "CDH2", "TAGLN", "CRABP2", "LIN28B"],
}


def load_i2(a):
    rna = pd.read_csv(os.path.join(a.data, "geo", "GSM3501446_MixCellLines-RNA.umi.txt.gz"), sep="\t", index_col=0)
    hto = pd.read_csv(os.path.join(a.data, "geo", "GSM3501447_MixCellLines-HTO-counts.csv.gz"), index_col=0)
    tags = [c for c in hto.columns if c.split("_")[0] in ("HEK", "THP1", "K562", "KG1")]
    cells = [c for c in rna.columns if c in hto.index]
    R = rna[cells]
    keep = R.sum(0) >= 500
    R = R.loc[:, keep.values]
    cells = list(R.columns)
    Xh = hto.loc[cells, tags].to_numpy(float).T
    line_of_tag = {i: t.split("_")[0] for i, t in enumerate(tags)}
    return R, tags, Xh, line_of_tag


def rna_truth(R, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.neighbors import NearestNeighbors

    C = R.to_numpy(float)
    keep = (C > 0).sum(1) >= 10
    G = np.array(R.index)[keep]
    C = C[keep]

    def norm(M):
        return np.log1p(M / M.sum(0, keepdims=True) * 1e4)

    L = norm(C)
    var = L.var(1) / (L.mean(1) + 1e-6)
    hv = np.argsort(-var)[:2000]
    Z = L[hv].T
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    svd = TruncatedSVD(20, random_state=seed).fit((Z - mu) / sd)
    P = svd.transform((Z - mu) / sd)
    km = KMeans(4, n_init=20, random_state=seed).fit(P)
    lab = km.labels_
    gi = {g: i for i, g in enumerate(G)}
    score = np.zeros((4, 4))
    names = list(MARKERS)
    for c in range(4):
        for m, genes in enumerate(names):
            idx = [gi[g] for g in MARKERS[genes] if g in gi]
            score[c, m] = (L[idx][:, lab == c].mean() - L[idx].mean()) / (L[idx].std() + 1e-6) if idx else 0
    r, cc = linear_sum_assignment(-score)
    marker_line = {int(i): names[j] for i, j in zip(r, cc)}  # QC only (Amendment 2): the marker naming swapped THP1 / KG1
    # scrublet-style cross-line doublets: simulate doublets from cells of different clusters, kNN in PCA space
    rng = np.random.default_rng(seed)
    n = C.shape[1]
    i, j = rng.integers(0, n, 2 * n), rng.integers(0, n, 2 * n)
    ok = lab[i] != lab[j]
    i, j = i[ok][:n], j[ok][:n]
    S = norm(C[:, i] + C[:, j])
    Ps = svd.transform((S[hv].T - mu) / sd)
    allP = np.vstack([P, Ps])
    isim = np.r_[np.zeros(n), np.ones(len(i))]
    nn = NearestNeighbors(n_neighbors=31).fit(allP)
    _, idx = nn.kneighbors(P)
    frac = isim[idx[:, 1:]].mean(1)
    dbl = frac > 0.5
    return lab, dbl, {"marker_cluster_line_QC_only": marker_line, "marker_score": score.round(2).tolist(), "n_doublet": int(dbl.sum()), "n": int(n)}


def name_clusters(lab, dbl, Xh, lot):
    """Amendment 2: cluster -> cell line by the one-to-one assignment (Hungarian) maximising the number of non-doublet cells whose
    strongest tag belongs to that line; the same kind of caller-free label matching as tag -> donor on I1."""
    fam = np.array([lot[i] for i in Xh.argmax(0)], dtype=object)
    lines = sorted(set(lot.values()))
    M = np.zeros((4, len(lines)))
    for c in range(4):
        for j, ln in enumerate(lines):
            M[c, j] = ((lab == c) & ~dbl & (fam == ln)).sum()
    r, cc = linear_sum_assignment(-M)
    cl = {int(i): lines[j] for i, j in zip(r, cc)}
    line = np.array([cl[l] for l in lab], dtype=object)
    return np.where(dbl, "Doublet", line).astype(object), cl, M.tolist(), lines


# --------------------------------------------------------------------------------------------------------- scoring
def evaluate(X, truth, t2t, callers, label):
    out = {}
    calls = {}
    for name, fn in callers.items():
        c = np.asarray(fn(X))
        calls[name] = c
        m, per = H.macro_f1(c, truth, t2t)
        ci_m, ci_s = H.bootstrap(c, truth, t2t, n_boot=N_BOOT)
        out[name] = {"macro_f1": m, "macro_f1_ci95": ci_m, "F_S": per["S"], "F_D": per["D"], "F_N": per["N"],
                     "dbl_called_singlet": H.genetic_doublets_called_singlet(c, truth), "dbl_called_singlet_ci95": ci_s,
                     "called_S_D_N": [int((c >= 0).sum()), int((c == H.DBL).sum()), int((c == H.NEG).sum())]}
        print(label, name, round(m, 4), ci_m, flush=True)
    return out, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--i1-truth")
    ap.add_argument("--i1-cells")
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--truth-only", action="store_true", help="build and report the truth only (no caller is run)")
    a = ap.parse_args()
    fz = json.load(open(a.frozen))
    callers = dict(H.CALLERS)
    if fz.get("singlet_caller") == "bffc":
        p = fz["params"]
        callers["singlet_frozen_bffc"] = lambda X, p=p: H.bff_cluster(X, alpha=p["alpha"], beta=p["beta"], tn_rule=p["tn_rule"], t2_rule=p["t2_rule"])[0]
    res = {"frozen": fz}
    competitors = ["hasheddrops", "gmm", "htodemux", "bff_raw", "bff_cluster"]
    singlet_names = [n for n in callers if n == "current" or n.startswith("singlet_frozen")]
    sets = []
    if a.only in (None, "I1"):
        X, tags, truth, info = load_i1(a)
        keep = truth != "unassigned"
        t2t, M = hungarian(X, truth)
        Xk, tk = X[:, keep], truth[keep]
        sets.append(("I1", Xk, tk, t2t, {**info, "n_scored": int(keep.sum()), "n_unassigned_excluded": int((~keep).sum()),
                                         "truth_counts": pd.Series(tk).value_counts().to_dict(), "tag_to_donor": t2t,
                                         "contingency_top_tag_by_donor": M.tolist()}))
    if a.only in (None, "I2"):
        R, tags, Xh, lot = load_i2(a)
        lab, dbl, info = rna_truth(R)
        truth, cl, M, lines = name_clusters(lab, dbl, Xh, lot)
        info.update(cluster_line=cl, cluster_by_top_tag_line=M, lines=lines)
        top_line = np.array([lot[i] for i in Xh.argmax(0)], dtype=object)
        info["concordance_rna_line_vs_top_tag_line"] = float((truth[truth != "Doublet"] == top_line[truth != "Doublet"]).mean())
        if a.truth_only:
            print(json.dumps(info, indent=1, default=str))
            return
        # tag -> line is known from the tag names; a singlet call is correct if its line equals the RNA line
        sets.append(("I2", Xh, truth, lot, {**info, "truth_counts": pd.Series(truth).value_counts().to_dict(), "tags": tags}))
    for label, X, truth, t2t, info in sets:
        out, calls = evaluate(X, truth, t2t, callers, label)
        best = max(competitors, key=lambda n: out[n]["macro_f1"])
        res[label] = {"info": info, "callers": out, "best_competitor": best}
        for s in singlet_names:
            d, ci = H.paired_bootstrap_diff(calls[s], calls[best], truth, t2t, n_boot=N_BOOT)
            res[label][f"{s}_minus_best_competitor"] = {"diff": out[s]["macro_f1"] - out[best]["macro_f1"], "paired_boot_mean": d,
                                                         "ci95": ci, "pass_within_0.01": bool(out[s]["macro_f1"] >= out[best]["macro_f1"] - 0.01)}
            print(label, s, "minus", best, res[label][f"{s}_minus_best_competitor"], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=lambda o: None)


if __name__ == "__main__":
    main()
