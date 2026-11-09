# SPDX-License-Identifier: MIT
"""Doublet-score sweep on development libraries: binary CSC prep, dbl_run per variant, AUPRC/AUROC against truth.

  dbl_sweep.py prep   WORKROOT DATADIR              WORKROOT/<ds>/cells.mtx|cells.csv|genes.csv -> DATADIR/<ds>.{hdr,indptr,indices,data,barcodes.tsv}
  dbl_sweep.py score  DATADIR RUNDIR LABEL OUT.json TRUTHDIR WORKROOT   score RUNDIR/LABEL/<ds>.tsv for every dataset

Development libraries only: the four Kang lanes (demuxlet multiplet label) and the barnyards (a cell is a doublet when its
minor species holds >= 20% of its UMIs). scDblFinder and Scrublet scores come from WORKROOT/<ds>/tools/{scdblfinder,scrublet}.csv
(same cells, defaults). Differences carry a paired bootstrap 95% interval (cells resampled together, 1000 draws, seed 7).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io

KANG = {"kang2018_ctrl": ("GSE96583_batch2.total.tsne.df.tsv.gz", "stim", "ctrl"),
        "kang2018_b1_a": ("GSE96583_batch1.total.tsne.df.tsv.gz", "batch", "A"),
        "kang2018_b1_b": ("GSE96583_batch1.total.tsne.df.tsv.gz", "batch", "B"),
        "kang2018_b1_c": ("GSE96583_batch1.total.tsne.df.tsv.gz", "batch", "C")}
BARN = ["hgmm1k_v3", "hgmm1k_v2"]


def prep(work, out):
    import scipy.sparse as sp
    os.makedirs(out, exist_ok=True)
    for ds in list(KANG) + BARN:
        m = scipy.io.mmread(f"{work}/{ds}/cells.mtx").tocsc()
        m.data = np.minimum(m.data, 65535).astype(np.uint16)
        m.indices = m.indices.astype(np.int32)
        m.indptr = m.indptr.astype(np.int32)
        np.array([m.shape[0], m.shape[1], m.nnz], dtype=np.uint32).tofile(f"{out}/{ds}.hdr")
        m.indptr.tofile(f"{out}/{ds}.indptr")
        m.indices.tofile(f"{out}/{ds}.indices")
        m.data.tofile(f"{out}/{ds}.data")
        pd.read_csv(f"{work}/{ds}/cells.csv").barcode.astype(str).to_csv(f"{out}/{ds}.barcodes.tsv", index=False, header=False)
        print(ds, m.shape, flush=True)


def auprc(s, y):
    o = np.argsort(-s, kind="stable")
    y = y[o]
    tp = np.cumsum(y)
    return float((tp / np.arange(1, len(y) + 1) * y).sum() / max(y.sum(), 1))


def auroc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def strip(x):
    return x.astype(str).str.replace("-1$", "", regex=True)


def truth(ds, tdir, work):
    if ds in KANG:
        f, col, val = KANG[ds]
        t = pd.read_csv(f"{tdir}/{f}", sep="\t", index_col=0)
        t = t[t[col].astype(str) == val]
        t.index = strip(pd.Series(t.index)).values
        lab = t["multiplets"].astype(str)
        return lab[lab.isin(["singlet", "doublet"])].eq("doublet").astype(int)
    cells = pd.read_csv(f"{work}/{ds}/cells.csv")
    genes = pd.read_csv(f"{work}/{ds}/genes.csv")
    m = scipy.io.mmread(f"{work}/{ds}/cells.mtx").tocsc()
    mouse = (genes.species.values == "mouse")
    mo = np.asarray(m[mouse].sum(0)).ravel()
    tot = np.asarray(m.sum(0)).ravel()
    minor = np.minimum(mo, tot - mo) / np.maximum(tot, 1)
    return pd.Series((minor >= 0.2).astype(int), index=strip(cells.barcode).values)


def score(data, run, label, out, tdir, work):
    rng = np.random.default_rng(7)
    res = {}
    for ds in list(KANG) + BARN:
        p = f"{run}/{label}/{ds}.tsv"
        if not os.path.exists(p):
            res[ds] = {"error": "no run"}
            continue
        bcs = strip(pd.read_csv(f"{data}/{ds}.barcodes.tsv", header=None)[0]).values
        r = pd.read_csv(p, sep="\t")
        sc = {"singlet": pd.Series(r.score.values, index=bcs)}
        call = pd.Series(r.is_doublet.values.astype(bool), index=bcs)
        for t in ("scdblfinder", "scrublet"):
            q = f"{work}/{ds}/tools/{t}.csv"
            if os.path.exists(q):
                x = pd.read_csv(q)
                sc[t] = pd.Series(x.score.values, index=strip(x.barcode).values)
        y = truth(ds, tdir, work)
        common = sorted(set(y.index) & set(sc["singlet"].index) & set(sc.get("scdblfinder", sc["singlet"]).index))
        yy = y.reindex(common).values
        out_ds = {"n": len(common), "n_doublets": int(yy.sum())}
        S = {k: v.reindex(common).fillna(-1).values.astype(float) for k, v in sc.items()}
        for k, v in S.items():
            out_ds[k] = {"auprc": round(auprc(v, yy), 4), "auroc": round(auroc(v, yy), 4)}
        cl = call.reindex(common).fillna(False).values
        out_ds["call"] = {"n": int(cl.sum()), "recall": round(float(cl[yy == 1].mean()), 4) if yy.sum() else None,
                          "precision": round(float(yy[cl].mean()), 4) if cl.sum() else None}
        if "scdblfinder" in S and yy.sum() >= 30:
            d = []
            for _ in range(1000):
                i = rng.integers(0, len(yy), len(yy))
                d.append(auprc(S["singlet"][i], yy[i]) - auprc(S["scdblfinder"][i], yy[i]))
            lo, hi = np.percentile(d, [2.5, 97.5])
            diff = out_ds["singlet"]["auprc"] - out_ds["scdblfinder"]["auprc"]
            tol = float((hi - lo) / 2)
            out_ds["vs_scdblfinder"] = {"diff": round(diff, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "tol": round(tol, 4),
                                        "pass": bool(diff >= -tol)}
        res[ds] = out_ds
    json.dump(res, open(out, "w"), indent=1)
    kang = [res[d] for d in KANG if "singlet" in res.get(d, {})]
    line = " ".join(f"{d[-4:]}:{res[d]['singlet']['auprc']:.3f}" for d in KANG if "singlet" in res.get(d, {}))
    npass = sum(1 for k in kang if k.get("vs_scdblfinder", {}).get("pass"))
    mean = np.mean([k["singlet"]["auprc"] for k in kang]) if kang else float("nan")
    barn = " ".join(f"{d}:{res[d]['singlet']['auroc']:.3f}/{res[d].get('scdblfinder', {}).get('auroc', float('nan')):.3f}" for d in BARN if "singlet" in res.get(d, {}))
    print(f"{label}\tkang_auprc {line}\tmean {mean:.4f}\tpass {npass}/4\tbarn_auroc(singlet/sdf) {barn}", flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "prep":
        prep(*sys.argv[2:4])
    else:
        score(*sys.argv[2:8])
