# SPDX-License-Identifier: MIT
"""Barnyard truth (human + mouse cells in one lane).

    barnyard.py DS OUT_DIR

Species counts per barcode from the Singlet run's raw matrix (ENSG = human, ENSMUSG = mouse). The informative "site" is the
species of each UMI. Models: pure human / pure mouse source (cross-species rate EPS) with ambient contamination c in C_GRID,
against the ambient split (human share of pooled barcodes with 10-49 UMIs that no caller calls, half of them; the other half are
the calibration negatives). Calibration positives: species-pure barcodes (share >= 0.9 of one species) with >= 5,000 UMIs that every
caller calls, thinned. A barcode that is not species-pure but whose human share differs from the ambient one (binomial p < 1e-6,
>= 200 UMIs) is a heterotypic doublet and counts as a cell. Strict variant: intact-cell flags (mito / intronic share against the
positives). Writes OUT_DIR/DS.truth.tsv.gz and OUT_DIR/DS.truth.json.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

EPS = 0.005


def species_counts(run):
    d = os.path.join(run, "raw_feature_bc_matrix")
    feats = pd.read_csv(os.path.join(d, "features.tsv.gz"), sep="\t", header=None)
    bcs = [C.strip(b) for b in C.lines(os.path.join(d, "barcodes.tsv.gz"))]
    M = C.read_mtx(os.path.join(d, "matrix.mtx.gz")).tocsc()  # genes x barcodes
    mouse = feats[0].str.startswith("ENSMUSG").to_numpy()
    human = feats[0].str.startswith("ENSG").to_numpy()
    nh = np.asarray(M[human].sum(0)).ravel()
    nm = np.asarray(M[mouse].sum(0)).ravel()
    return pd.DataFrame({"barcode": bcs, "n_human": nh, "n_mouse": nm})


def main(ds, out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(1)
    t, missing = C.gather_callers(ds)
    sc = species_counts(C.DATASETS[ds]["run"])
    t = t.merge(sc, on="barcode", how="left").fillna({"n_human": 0, "n_mouse": 0})
    n = (t["n_human"] + t["n_mouse"]).to_numpy().astype(int)
    h = t["n_human"].to_numpy().astype(int)
    callers = [c for c in ("called_v2", "knee", "edcr", "cellbender", "cellranger") if c in t]
    anyc = t[callers].max(axis=1).to_numpy()
    allc = t[callers].min(axis=1).to_numpy()
    emp = np.nonzero((t["umi_total"].to_numpy() >= 10) & (t["umi_total"].to_numpy() < 50) & (anyc == 0) & (n > 0))[0]
    e_half = rng.random(len(emp)) < 0.5
    e1, e2 = emp[e_half], emp[~e_half]
    h_amb = h[e1].sum() / n[e1].sum()
    frac = np.where(n > 0, h / np.maximum(n, 1), np.nan)
    pos = np.nonzero((t["umi_total"].to_numpy() >= 5000) & (allc == 1) & ((frac >= 0.9) | (frac <= 0.1)))[0]
    print(f"[barnyard] {ds}: ambient human share {h_amb:.3f} from {len(e1)} barcodes; positives {len(pos)}; negatives pool {len(e2)}")
    cols = [np.array([h_amb])]
    for c in C.C_GRID:
        cols.append(np.array([(1 - c) * (1 - EPS) + c * h_amb]))
        cols.append(np.array([(1 - c) * EPS + c * h_amb]))
    F = np.stack(cols, axis=1)  # 1 x models

    def llr_fn(ad, dp, full=False):
        L = C.ll_matrix(ad, dp, F)
        best = L[:, 1:].argmax(1) + 1
        llr = L[np.arange(len(L)), best] - L[:, 0]
        return (llr, best) if full else llr

    def umis(i, cap=400):
        k = min(n[i], cap)  # subsample huge cells for speed (calibration only needs <= 300)
        if n[i] > cap:
            alt = (rng.random(k) < h[i] / n[i]).astype(np.int8)
        else:
            alt = np.r_[np.ones(h[i], np.int8), np.zeros(n[i] - h[i], np.int8)]
        return np.zeros(k, dtype=np.int64), alt
    cal, n_min = C.calibrate(llr_fn, [umis(i) for i in pos], [umis(i) for i in e2], 1)
    print(cal.to_string())
    print("[barnyard] minimum informative depth N =", n_min)
    import scipy.sparse as sp
    AD = sp.csr_matrix(h.reshape(-1, 1).astype(float))
    DP = sp.csr_matrix(n.reshape(-1, 1).astype(float))
    llr, best = llr_fn(AD, DP, full=True)
    truth = C.classify(llr, n, cal, n_min)
    p_amb = np.minimum(1, 2 * np.minimum(binom.cdf(h, n, h_amb), binom.sf(h - 1, n, h_amb)))
    doublet = (truth != "cell") & (n >= 200) & (p_amb < 1e-6) & (frac > 0.1) & (frac < 0.9)
    truth = np.where(doublet, "cell", truth)
    t["species_umis"], t["human_share"], t["llr"] = n, frac, llr
    t["model"] = np.where(doublet, "doublet", np.where(best % 2 == 1, "human", "mouse"))
    t["truth"] = truth
    ref = t.iloc[pos]
    intr = lambda x: 1 - x["umi_exon"] / x["umi_total"]  # noqa: E731
    ea = t.iloc[e1]
    amb_intr = 1 - ea["umi_exon"].sum() / ea["umi_total"].sum()
    st, hm, li, q = C.intact_state(t["umi_total"], t["pct_mt"] / 100, intr(t), ref["pct_mt"].to_numpy() / 100,
                                   intr(ref).to_numpy(), amb_intr)
    t["intact_state"], t["high_mito"], t["low_intronic"] = st, hm.astype(np.int8), li.astype(np.int8)
    t["truth_strict"] = C.strict_truth(t["truth"].to_numpy(), st)
    t["calib_role"] = ""
    t.loc[t.index[pos], "calib_role"] = "positive"
    t.loc[t.index[e1], "calib_role"] = "ambient_fit"
    t.loc[t.index[e2], "calib_role"] = "negative"
    t.to_csv(os.path.join(out, f"{ds}.truth.tsv.gz"), sep="\t", index=False)
    summ = dict(dataset=ds, kind="barnyard", ambient_human_share=float(h_amb), n_positives=len(pos), n_negatives_pool=len(e2),
                n_min=n_min, calibration=cal.to_dict("records"), intact_reference=q, missing_from_universe=missing,
                truth_counts=pd.Series(truth).value_counts().to_dict(), n_doublet=int(doublet.sum()),
                truth_strict_counts=t["truth_strict"].value_counts().to_dict())
    C.dump(summ, os.path.join(out, f"{ds}.truth.json"))
    print(summ["truth_counts"], summ["truth_strict_counts"])


if __name__ == "__main__":
    main(*sys.argv[1:3])
