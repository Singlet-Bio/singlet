# SPDX-License-Identifier: MIT
"""Candidate called_v2 changes, scored on the independent truths.

    variants.py calls DS OUT_DIR        -> OUT_DIR/DS.variants.tsv.gz (barcode x variant, 0/1)
    variants.py score  TRUTH_DIR OUT_JSON DS...

Calls use python/singlet/v2/cellcall.py 2.1 (``$CELLCALL_DIR``) on the Singlet run's stored exon + intron counts, loaded with
bench/called_v2_truth/common.py (copied as kcommon.py). Variants:
  base          2.1 defaults (plateau lower, bounded knee retain, tail p, test floor 50, FDR 0.001)
  fdr_0.01 / fdr_1e-4   BH threshold
  test_min_100  test floor 100 (BH over the barcodes above it)
  exon_only     EmptyDrops on exon-only gene counts
  mito_20       base, minus added barcodes (tested, not in the knee call) with >= 20% mitochondrial UMIs
  mito_q99      base, minus added barcodes whose mito share is significantly (binomial p < 0.01) above the 99th percentile of the
                knee-called cells
Scoring: the per-barcode posterior P(cell) from score.py (mixture over the independent truth; posteriors fixed, only the calls
change), lenient and intact; bootstrap CIs and the paired difference to base.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402


def load_exon_only(run, K):
    feats = K._lines(os.path.join(run, "exon_counts_features.tsv.gz"))
    M = K.read_mtx(os.path.join(run, "exon_counts.mtx.gz"))
    keys = [f.split("_", 1)[0] for f in feats]
    u = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    rows = np.array([u[k] for k in keys])
    A = sp.csr_matrix((np.ones(len(keys), np.int64), (rows, np.arange(len(keys)))), shape=(len(u), len(keys)))
    return (A @ M).tocsc()


def calls(ds, out):
    import kcommon as K
    CC = K.CC
    from scipy.stats import binom
    run = C.DATASETS[ds]["run"]
    R = K.load_run(run)
    X, called, tot = R["X"], R["called"], R["tot"]
    cv2, pv, q, prm = CC.call_detail(X, called)
    print(f"[var] {ds} base: {int(cv2.sum())} (knee {int(called.sum())}, lower {prm['lower']}, retain {prm['retain_total']})")
    V = {"base": cv2}
    tested = np.isfinite(pv)
    for f in (0.01, 1e-4):
        qq = np.full(len(pv), np.nan)
        qq[tested] = CC.bh(pv[tested])
        V[f"fdr_{f:g}"] = called | (np.nan_to_num(qq, nan=1.0) <= f)
    t100 = tested & (tot > 100)
    qq = np.full(len(pv), np.nan)
    qq[t100] = CC.bh(pv[t100])
    V["test_min_100"] = called | (np.nan_to_num(qq, nan=1.0) <= 0.001)
    Xe = load_exon_only(run, K)
    ce, _, _, pe = CC.call_detail(Xe, called)
    V["exon_only"] = ce
    print(f"[var] {ds} exon_only: {int(ce.sum())} (lower {pe['lower']})")
    added = cv2 & ~called
    V["mito_20"] = cv2 & ~(added & (R["pct_mt"] >= 20))
    km = R["pct_mt"][called] / 100
    q99 = np.quantile(km, 0.99)
    k = np.round(R["pct_mt"] / 100 * tot).astype(int)
    V["mito_q99"] = cv2 & ~(added & (binom.sf(k - 1, tot, q99) < 0.01))
    df = pd.DataFrame({"barcode": [C.strip(b) for b in R["barcodes"]], **{k: v.astype(np.int8) for k, v in V.items()}})
    df = df[df.drop(columns="barcode").max(axis=1) == 1]
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"{ds}.variants.tsv.gz"), sep="\t", index=False)
    print({k: int(v.sum()) for k, v in V.items()})


def score(tdir, out_json, *dss, nb=500, seed=9):
    res = {}
    for ds in dss:
        t = pd.read_csv(os.path.join(tdir, f"{ds}.truth.tsv.gz"), sep="\t", dtype={"barcode": str}, low_memory=False)
        p = pd.read_csv(os.path.join(tdir, f"{ds}.posterior.tsv.gz"), sep="\t", dtype={"barcode": str})
        v = pd.read_csv(os.path.join(tdir, f"{ds}.variants.tsv.gz"), sep="\t", dtype={"barcode": str})
        u = p.merge(t[["barcode", "intact_state", "umi_total", "called_v2"]], on="barcode").merge(v, on="barcode", how="left").fillna(0)
        names = [c for c in v.columns if c != "barcode"]
        post = u["post"].to_numpy()
        nd = (u["intact_state"] != "damaged").to_numpy()
        A = {n: u[n].to_numpy().astype(bool) for n in names + ["called_v2"]}
        rng = np.random.default_rng(seed)

        def met(ix):
            o = {}
            for var, w in (("lenient", post[ix]), ("intact", (post * nd)[ix])):
                tot = w.sum()
                for n, a in A.items():
                    tp = w[a[ix]].sum()
                    pr, rc = tp / max(a[ix].sum(), 1), tp / max(tot, 1e-9)
                    o[(var, n)] = (pr, rc, 2 * pr * rc / max(pr + rc, 1e-12))
            return o
        full = met(np.arange(len(u)))
        boots = [met(rng.integers(0, len(u), len(u))) for _ in range(nb)]
        R = {"lenient": {}, "intact": {}}
        for (var, n), (pr, rc, f) in full.items():
            b = np.array([x[(var, n)] for x in boots])
            bb = np.array([x[(var, "base")] for x in boots])
            ci = lambda z: [float(np.quantile(z, 0.025)), float(np.quantile(z, 0.975))]  # noqa: E731
            R[var][n] = {"n_called": int(A[n].sum()), "precision": float(pr), "recall": float(rc), "f1": float(f),
                         "f1_ci": ci(b[:, 2]), "f1_minus_base_ci": ci(b[:, 2] - bb[:, 2])}
            print(f"{ds:12s} {var:8s} {n:13s} n={A[n].sum():6d} P={pr:.4f} R={rc:.4f} F1={f:.4f} dF1 vs base {np.round(ci(b[:, 2] - bb[:, 2]), 4)}")
        res[ds] = R
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "calls":
        calls(sys.argv[2], sys.argv[3])
    else:
        score(*sys.argv[2:])
