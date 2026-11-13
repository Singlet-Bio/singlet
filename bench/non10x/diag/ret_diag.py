# SPDX-License-Identifier: MIT
# D-152 diagnostic (run on Clipper in debug-cpu; see the PR body of ve/non10x-routes for what it measured).
import sys, os, gzip, numpy as np, pandas as pd
R = sys.argv[1]; T = sys.argv[2]
A = pd.read_csv(T, sep=r"\s+", index_col=0, engine="python")
aset = set(A.columns.astype(str))
au = A.sum(axis=0).sort_values(ascending=False)
print("authors cells", len(aset), "umi median", float(au.median()), "q10", float(au.quantile(.1)), "q25", float(au.quantile(.25)))
q = pd.read_csv(os.path.join(R, "cell_qc_metrics.tsv"), sep="\t", usecols=["barcode", "n_umi", "n_genes"])
cc = pd.read_csv(os.path.join(R, "cell_calls.tsv"), sep="\t")
q["bc"] = q.barcode.astype(str).str.split("-").str[0]; cc["bc"] = cc.barcode.astype(str).str.split("-").str[0]
q = q.merge(cc[["bc", "is_cell", "called_knee", "called_v2", "called_lowrna"]], on="bc")
q["called"] = q.is_cell.astype(str).str.upper() == "TRUE"
q["auth"] = q.bc.isin(aset)
print("barcodes", len(q), "called", int(q.called.sum()), "knee", int((q.called_knee == 1).sum()), "called_v2", int((q.called_v2 == 1).sum()), "auth with any umi", int(q.auth.sum()))
c = q[q.called]
print("called in authors", int(c.auth.sum()), "precision", round(float(c.auth.mean()), 3), "recall", round(c.auth.sum() / len(aset), 3), "called umi median", float(c.n_umi.median()))
print("authors' cells: our umi quantiles", q[q.auth].n_umi.quantile([.05, .1, .25, .5, .75]).tolist())
for t in (60, 100, 150, 200, 250, 300, 400, 500):
    s = q[q.n_umi >= t]
    print("n_umi>=%3d: barcodes %6d  in authors %5d  precision %.3f recall %.3f" % (t, len(s), int(s.auth.sum()), float(s.auth.mean()), s.auth.sum() / len(aset)))
