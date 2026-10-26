# SPDX-License-Identifier: MIT
"""Genotype truth for a donor-multiplexed lane (Kang 2018).

    genotruth.py DS CELLSNP_DIR AUTHORS_TABLE STIM OUT_DIR

1. Universe and caller columns: common.gather_callers(DS).
2. Donor genotypes from the authors' demuxlet singlets (their donor ids come from SNP-array genotypes): half of the singlets
   (random, seed 1) are pooled per donor and genotyped at every cellsnp site (alt fraction model {0.05, 0.5, 0.95}); a site is
   kept when every donor has >= 10 UMIs, every call has a log-likelihood margin >= 4 and the donors are not all alike. The other
   half of the singlets with >= 1,500 UMIs are the calibration positives.
3. Ambient: barcodes with 10-49 UMIs that no caller calls, split at random: half gives the donor weights of the ambient mixture
   (ML on the pooled counts), half the calibration negatives.
4. Per barcode: LLR = max over (single donor | donor pair) x contamination c of the log-likelihood minus that of the ambient mixture;
   informative depth = UMIs at kept sites. Calibration and classification: common.calibrate / common.classify.
5. Intact-cell flags (mito / intronic share against the positives) for the strict variant.

Writes OUT_DIR/DS.truth.tsv.gz and OUT_DIR/DS.truth.json.
"""

from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def load_cellsnp(d):
    samples = [C.strip(x) for x in C.lines(os.path.join(d, "cellSNP.samples.tsv"))]
    f = lambda t: os.path.join(d, f"cellSNP.tag.{t}.mtx")  # noqa: E731
    AD = C.read_mtx(f("AD")).T.tocsr()
    DP = C.read_mtx(f("DP")).T.tocsr()
    AD, DP = C.align(AD, DP)
    return np.array(samples), AD, DP


def genotype(A, D, e=0.05):
    """A, D: donors x sites pooled counts -> dosage {0, 0.5, 1}, margin."""
    ps = np.array([e, 0.5, 1 - e])
    ll = np.stack([A * np.log(p) + (D - A) * np.log(1 - p) for p in ps])  # 3 x donors x sites
    o = np.sort(ll, axis=0)
    return np.array([0.0, 0.5, 1.0])[ll.argmax(0)], o[-1] - o[-2]


def main(ds, csdir, authors, stim, out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(1)
    t, missing = C.gather_callers(ds)
    bcs, AD, DP = load_cellsnp(csdir)
    pos_of = {b: i for i, b in enumerate(bcs)}
    a = pd.read_csv(authors, sep="\t", index_col=0)
    a = a[a["stim"] == stim]
    a.index = a.index.map(C.strip)
    sing = a[(a["multiplets"] == "singlet") & a.index.isin(pos_of)]
    half = rng.random(len(sing)) < 0.5
    geno_cells, cal_cells = sing[half], sing[~half]
    donors = sorted(sing["ind"].unique())
    # 2. genotypes
    A = np.zeros((len(donors), AD.shape[1]))
    Dd = np.zeros_like(A)
    for k, d in enumerate(donors):
        ix = [pos_of[b] for b in geno_cells.index[geno_cells["ind"] == d]]
        A[k] = np.asarray(AD[ix].sum(0)).ravel()
        Dd[k] = np.asarray(DP[ix].sum(0)).ravel()
    G, margin = genotype(A, Dd)
    keep = (Dd >= 10).all(0) & (margin >= 4).all(0) & (G.min(0) != G.max(0))
    G = G[:, keep]
    AD, DP = C.align(AD[:, keep], DP[:, keep])
    S = G.shape[1]
    eps = 0.01
    P = np.where(G == 0, eps, np.where(G == 1, 1 - eps, 0.5))  # donors x sites
    print(f"[geno] donors {donors}; sites kept {S} of {len(keep)}; genotype cells {len(geno_cells)}, calibration cells {len(cal_cells)}")
    # 3. ambient weights
    tt = t.set_index("barcode")
    callers = [c for c in ("called_v2", "knee", "edcr", "cellbender", "cellranger", "demuxlet_table") if c in tt]
    anyc = tt[callers].max(axis=1)
    emp = tt[(tt["umi_total"] >= 10) & (tt["umi_total"] < 50) & (anyc == 0)].index
    emp = np.array([b for b in emp if b in pos_of])
    e_half = rng.random(len(emp)) < 0.5
    e1 = [pos_of[b] for b in emp[e_half]]
    e2 = [pos_of[b] for b in emp[~e_half]]
    a1 = np.asarray(AD[e1].sum(0)).ravel()
    d1 = np.asarray(DP[e1].sum(0)).ravel()

    def nll(z):
        w = np.exp(z - z.max())
        w /= w.sum()
        f = np.clip(w @ P, 1e-6, 1 - 1e-6)
        return -(a1 * np.log(f) + (d1 - a1) * np.log1p(-f)).sum()
    z = minimize(nll, np.zeros(len(donors)), method="L-BFGS-B").x
    w = np.exp(z - z.max())
    w /= w.sum()
    amb = w @ P
    cell_frac = sing["ind"].value_counts(normalize=True).reindex(donors).to_numpy()
    print("[geno] ambient donor weights", np.round(w, 3).tolist(), "cell donor shares", np.round(cell_frac, 3).tolist())
    # 4. model matrix
    cols, desc = [amb], [("ambient", -1, -1, 1.0)]
    for c in C.C_GRID:
        for k in range(len(donors)):
            cols.append((1 - c) * P[k] + c * amb)
            desc.append(("single", k, -1, c))
        for k1, k2 in combinations(range(len(donors)), 2):
            cols.append((1 - c) * 0.5 * (P[k1] + P[k2]) + c * amb)
            desc.append(("doublet", k1, k2, c))
    F = np.stack(cols, axis=1)  # sites x models

    def llr_fn(ad, dp, full=False):
        L = C.ll_matrix(ad, dp, F)
        best = L[:, 1:].argmax(1) + 1
        llr = L[np.arange(len(L)), best] - L[:, 0]
        return (llr, best) if full else llr
    pos_umis = [C.umis_of(AD, DP, pos_of[b]) for b in cal_cells.index[(tt.reindex(cal_cells.index)["umi_total"] >= 1500).to_numpy()]]
    neg_umis = [C.umis_of(AD, DP, i) for i in e2]
    cal, n_min = C.calibrate(llr_fn, pos_umis, neg_umis, S)
    print(cal.to_string())
    print("[geno] minimum informative depth N =", n_min)
    # per barcode
    idx = np.array([pos_of.get(b, -1) for b in t["barcode"]])
    has = idx >= 0
    llr = np.full(len(t), np.nan)
    best = np.full(len(t), -1)
    l_, b_ = llr_fn(AD[idx[has]], DP[idx[has]], full=True)
    llr[has], best[has] = l_, b_
    depth = np.zeros(len(t))
    depth[has] = np.asarray(DP[idx[has]].sum(1)).ravel()
    t["snp_umis"] = depth
    t["llr"] = llr
    t["model"] = [desc[j][0] if j >= 0 else "" for j in best]
    t["donor"] = [(donors[desc[j][1]] if j > 0 else "") for j in best]
    t["contam"] = [desc[j][3] if j >= 0 else np.nan for j in best]
    t["truth"] = C.classify(np.nan_to_num(llr, nan=0.0), depth, cal, n_min)
    t.loc[~has, "truth"] = "unknown"
    # 5. intact-cell flags against the calibration positives
    ref = tt.reindex(cal_cells.index)
    ref = ref[ref["umi_total"] >= 1500]
    intr = lambda x: 1 - x["umi_exon"] / x["umi_total"]  # noqa: E731
    ea = tt.reindex(emp[e_half])
    amb_intr = 1 - ea["umi_exon"].sum() / ea["umi_total"].sum()
    st, hm, li, q = C.intact_state(t["umi_total"], t["pct_mt"] / 100, intr(t), ref["pct_mt"].to_numpy() / 100,
                                   intr(ref).to_numpy(), amb_intr)
    t["intact_state"], t["high_mito"], t["low_intronic"] = st, hm.astype(np.int8), li.astype(np.int8)
    t["truth_strict"] = C.strict_truth(t["truth"].to_numpy(), st)
    t["calib_role"] = ""
    t.loc[t["barcode"].isin(geno_cells.index), "calib_role"] = "genotype"
    t.loc[t["barcode"].isin(ref.index), "calib_role"] = "positive"
    t.loc[t["barcode"].isin(emp[e_half]), "calib_role"] = "ambient_fit"
    t.loc[t["barcode"].isin(emp[~e_half]), "calib_role"] = "negative"
    t.to_csv(os.path.join(out, f"{ds}.truth.tsv.gz"), sep="\t", index=False)
    summ = dict(dataset=ds, kind="genotype", donors=[str(d) for d in donors], n_sites=int(S), ambient_weights=w.tolist(),
                cell_donor_shares=cell_frac.tolist(), n_genotype_cells=int(len(geno_cells)), n_positives=len(pos_umis),
                n_negatives_pool=len(neg_umis), n_min=n_min, calibration=cal.to_dict("records"), intact_reference=q,
                missing_from_universe=missing, truth_counts=t["truth"].value_counts().to_dict(),
                truth_strict_counts=t["truth_strict"].value_counts().to_dict())
    C.dump(summ, os.path.join(out, f"{ds}.truth.json"))
    print(summ["truth_counts"], summ["truth_strict_counts"])


if __name__ == "__main__":
    main(*sys.argv[1:6])
