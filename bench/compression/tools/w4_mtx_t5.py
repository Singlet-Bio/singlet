#!/usr/bin/env python
"""w4_mtx test 5: is ambient_profile.tsv a function of the .1pz counts (sum over non-cell barcodes + Jeffreys 0.5 pseudocount)?"""
import os, sys, numpy as np
sys.path.insert(0, '../../python')
from singlet._pz import read_1pz
HERE = os.path.dirname(os.path.abspath(__file__)); BC = os.path.dirname(HERE); CORP = os.path.join(BC, "corpus_w4_mtx")
for d in sys.argv[1:]:
    p = os.path.join(CORP, d)
    L = open(os.path.join(p, "ambient_profile.tsv")).read().split("\n")[1:-1]
    names = [l.split("\t")[0] for l in L]; sval = [l.split("\t")[1] for l in L]
    v = np.array([float(x) for x in sval]); N = len(v); u = v.min()
    print(f"[{d}] ambient rows {N}, min {u!r}, sum {v.sum():.7f}, 1/u={1/u:.2f}")
    k = np.rint((v / u - 1) / 2)    # implied count if value=(2c+1)*u
    print("   fraction of values that are odd multiples of min (+-1%):", np.mean(np.abs((v / u - 1) / 2 - k) < 0.02 * np.maximum(1, k)))
    ex = read_1pz(os.path.join(p, "exon_counts.1pz"))
    assert ex["rownames"] == names, "row order differs"
    cells = [l.split("\t")[0] for l in open(os.path.join(p, "cell_calls.tsv")).read().split("\n")[1:-1]]
    cs = set(cells); iscell = np.array([c in cs for c in ex["colnames"]])
    ip = ex["indptr"].astype(np.int64); cn = np.repeat(np.arange(ex["n"]), np.diff(ip)); ix = ex["indices"].astype(np.int64); dv = ex["data"].astype(np.int64)
    print(f"   exon 1pz: {ex['m']} rows x {ex['n']} cols, cells among cols={iscell.sum()}, called cells={len(cells)}")
    for tag, sel in (("non-cell columns", ~iscell[cn]), ("all columns", np.ones(len(cn), bool)), ("cell columns", iscell[cn])):
        c = np.bincount(ix[sel], weights=dv[sel], minlength=ex["m"]).astype(np.int64); T = int(c.sum())
        for alpha in (0.5, 1.0, 0.0):
            pred = (c + alpha) / (T + alpha * N)
            rel = np.abs(pred - v) / v
            sig6 = np.array([float("%.6g" % x) for x in pred])
            print(f"   {tag:17s} alpha={alpha}: T={T} max rel err {rel.max():.3g}, exact 6-sig-digit match {np.mean(sig6 == v):.6f} ({int((sig6==v).sum())}/{N})")
    # implied counts vs non-cell column sums
    sel = ~iscell[cn]; c = np.bincount(ix[sel], weights=dv[sel], minlength=ex["m"]).astype(np.int64)
    print("   implied c_i from file == non-cell column sums:", int((k == c).sum()), "of", N, "; implied T =", int(k.sum()), "vs non-cell total", int(c.sum()), "vs all total", int(dv.sum()))
