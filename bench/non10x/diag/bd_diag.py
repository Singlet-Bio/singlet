# SPDX-License-Identifier: MIT
# D-152 diagnostic (run on Clipper in debug-cpu; see the PR body of ve/non10x-routes for what it measured).
import sys, numpy as np, pandas as pd, gzip, json, os
R = sys.argv[1]; T = "/mnt/projects/debruinz_project/debruinz/singlet/data/non10x/truth/GSM4764685_SampleTag_0%d_SC106_DBEC_MolsPerCell.csv.gz"
rng = np.random.default_rng(1)
mats = []
for i in (1, 2, 3, 4):
    d = pd.read_csv(T % i, index_col=0, comment="#")
    d = d.select_dtypes("number")
    tot = d.sum(1)
    print("tag", i, "cells", d.shape[0], "genes", d.shape[1], "total molecules %.3e" % tot.sum(), "median umi", tot.median(), "median genes", float((d.values > 0).sum(1).__array__().astype(float).tolist() and np.median((d.values > 0).sum(1))))
    mats.append(d)
allc = pd.concat(mats, axis=0, sort=False).fillna(0)
A = allc.values.astype(np.int64)
umi_a = A.sum(1); gen_a = (A > 0).sum(1)
print("authors union cells", A.shape[0], "total molecules %.3e" % umi_a.sum(), "median umi", np.median(umi_a), "median genes", np.median(gen_a))
q = pd.read_csv(os.path.join(R, "cell_qc_metrics.tsv"), sep="\t", usecols=["barcode", "n_umi", "n_genes", "n_reads"])
cc = pd.read_csv(os.path.join(R, "cell_calls.tsv"), sep="\t", usecols=["barcode", "is_cell"])
q = q.merge(cc, on="barcode")
called = q[q.is_cell.astype(str).str.upper() == "TRUE"]
print("ours: reads total in file", q.n_reads.sum(), "called", len(called), "called umi median", called.n_umi.median(), "genes median", called.n_genes.median(), "total umi in cells %.3e" % called.n_umi.sum(), "total umi all %.3e" % q.n_umi.sum())
for k in (1000, 2000, 4000, 8000, 10275):
    top = q.sort_values("n_umi", ascending=False).head(k)
    print("ours top %d by umi: median umi %.0f median genes %.0f | authors top %d: median umi %.0f median genes %.0f" % (k, top.n_umi.median(), top.n_genes.median(), k, np.median(np.sort(umi_a)[::-1][:k]), np.median(gen_a[np.argsort(-umi_a)[:k]])))
print("ours called with n_umi<300:", int((called.n_umi < 300).sum()), "<500:", int((called.n_umi < 500).sum()), " authors cells with umi<300:", int((umi_a < 300).sum()), "<500", int((umi_a < 500).sum()))
# binomial thinning of the authors' matrix to our depth (cell-by-cell total UMI scaled by global factor f)
for target in (float(called.n_umi.median()),):
    f = target / np.median(umi_a)
    B = rng.binomial(A, f)
    print("thin authors by f=%.4f (matches our median umi %.0f): median umi %.0f median genes %.0f (ours %.0f)" % (f, target, np.median(B.sum(1)), np.median((B > 0).sum(1)), called.n_genes.median()))
    # rank-matched: thin so that total molecules equal ours in the top-N cells
