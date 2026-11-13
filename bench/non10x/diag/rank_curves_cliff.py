# SPDX-License-Identifier: MIT
# D-152 diagnostic (run on Clipper in debug-cpu; see the PR body of ve/non10x-routes for what it measured).
import sys, os, numpy as np, pandas as pd
J = "/mnt/projects/debruinz_project/debruinz/singlet/jobs/non10x/"
libs = {"barnyard30M": J + "out-val/dropseq_barnyard/res", "barnyardFull": J + "out-full/dropseq_barnyard_full/res", "retina30M": J + "out-val/dropseq_retina/res"}
T = "/mnt/projects/debruinz_project/debruinz/singlet/data/non10x/truth/"
def nb(b):
    for i in range(len(b)):
        for c in "ACGT":
            if c != b[i]: yield b[:i] + c + b[i+1:]
def truth(name):
    rd = lambda p: pd.read_csv(p, sep=r"\s+", index_col=0, engine="python")
    if name.startswith("barnyard"):
        H = rd(T + "GSM1544798_SpeciesMix_ThousandSTAMPs_HUMAN.digital_expression.txt.gz"); M = rd(T + "GSM1544798_SpeciesMix_ThousandSTAMPs_MOUSE.digital_expression.txt.gz")
        tot = H.sum(axis=0).add(M.sum(axis=0), fill_value=0)
        return set(tot.sort_values(ascending=False).head(1020).index.astype(str))
    A = rd(T + "GSM1626793_P14Retina_1.digital_expression.txt.gz")
    return set(A.columns.astype(str))
pts = [10, 30, 100, 300, 600, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 15000, 20000, 40000]
for name, R in libs.items():
    q = pd.read_csv(os.path.join(R, "cell_qc_metrics.tsv"), sep="\t", usecols=["barcode", "n_umi"])
    q["bc"] = q.barcode.astype(str).str.split("-").str[0]
    q = q[q.n_umi > 0].sort_values("n_umi", ascending=False).reset_index(drop=True)
    bcs = q.bc.tolist(); u = q.n_umi.values.astype(float); idx = {b: i for i, b in enumerate(bcs)}
    parent = np.arange(len(u))
    for i in range(min(len(u), 400000)):
        best = -1
        for x in nb(bcs[i]):
            j = idx.get(x)
            if j is not None and u[j] >= 3 * u[i] and (best < 0 or u[j] > u[best]): best = j
        if best >= 0: parent[i] = best
    for i in range(len(u)):
        p = i
        while parent[p] != p: p = parent[p]
        parent[i] = p
    mt = np.zeros(len(u)); np.add.at(mt, parent, u)
    roots = parent == np.arange(len(u))
    s = np.sort(mt[roots])[::-1]
    A = truth(name); aset_idx = np.array([bcs[i] in A for i in range(len(u))])
    print("==", name, "barcodes", len(u), "roots", int(roots.sum()), "authors present", int(aset_idx.sum()))
    mt_r = mt[roots]; fl = aset_idx[roots]
    order = np.argsort(-mt_r); ss = mt_r[order]; ff = fl[order]
    n = len(ss)
    # largest gap: max over k of ss[k]/ss[1.5k]
    ks = np.arange(100, min(n // 3, 100000))
    ratio = ss[ks] / np.maximum(1.0, ss[np.minimum(n - 1, (1.5 * ks).astype(int))])
    kb = int(ks[np.argmax(ratio)]); rb = float(ratio.max())
    thr = float(np.sqrt(ss[kb] * ss[min(n - 1, int(1.5 * kb))]))
    ncell = int((ss >= thr).sum())
    prec = float(ff[:ncell].mean()); rec = float(ff[:ncell].sum() / max(1, len(A)))
    print("gap rule: k*=%d ratio=%.2f thr=%.0f cells=%d precision=%.3f recall=%.3f (authors %d)" % (kb, rb, thr, ncell, prec, rec, len(A)))
    for k in (100, 300, 500, 800, 1000, 1200, 2000, 5000):
        if k < n // 2: print("   ratio s[k]/s[1.5k] at k=%d: %.2f" % (k, ss[k] / ss[int(1.5 * k)]))
