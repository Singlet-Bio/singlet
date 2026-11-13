# SPDX-License-Identifier: MIT
# D-152 diagnostic (run on Clipper in debug-cpu; see the PR body of ve/non10x-routes for what it measured).
import sys, os, gzip, numpy as np, pandas as pd, scipy.io
R = sys.argv[1]
T = "/mnt/projects/debruinz_project/debruinz/singlet/data/non10x/truth/GSM1544798_SpeciesMix_ThousandSTAMPs_%s.digital_expression.txt.gz"
rd = lambda p: pd.read_csv(p, sep=r"\s+", index_col=0, engine="python")
H, M = rd(T % "HUMAN"), rd(T % "MOUSE")
ah, am = H.sum(0), M.sum(0); tot = ah.add(am, fill_value=0)
auth = list(tot.sort_values(ascending=False).head(1020).index.astype(str)); aset = set(auth)
q = pd.read_csv(os.path.join(R, "cell_qc_metrics.tsv"), sep="\t", usecols=["barcode", "n_umi", "n_genes"])
q["bc"] = q.barcode.astype(str).str.split("-").str[0]
cc = pd.read_csv(os.path.join(R, "cell_calls.tsv"), sep="\t")
cc["bc"] = cc.barcode.astype(str).str.split("-").str[0]
q = q.merge(cc[["bc", "is_cell", "called_knee", "umi_exon"]], on="bc")
q["called"] = q.is_cell.astype(str).str.upper() == "TRUE"
q = q.sort_values("n_umi", ascending=False).reset_index(drop=True)
print("barcodes", len(q), "called", int(q.called.sum()), "knee-called", int((q.called_knee == 1).sum()))
for k in (100, 500, 1000, 1020, 1500, 2000, 3000, 4443, 6000, 10000, 50000):
    if k < len(q): print("rank", k, "umi", int(q.n_umi[k]))
print("authors' 1020: min umi in authors", float(tot.reindex(auth).min()), "ours umi on those: median", float(q[q.bc.isin(aset)].n_umi.median()))
# HD1 neighbours
def nb(b):
    for i in range(len(b)):
        for c in "ACGT":
            if c != b[i]: yield b[:i] + c + b[i+1:]
top = q.head(6000)
rank = {b: i for i, b in enumerate(q.bc)}
umi = dict(zip(q.bc, q.n_umi))
cat = []
for b, c in zip(top.bc, top.called):
    par = None
    for x in nb(b):
        if x in umi and umi[x] > umi[b]:
            if par is None or umi[x] > umi[par]: par = x
    cat.append(par)
top = top.assign(parent=cat)
top["hd1_child"] = top.parent.notna()
top["ratio"] = top.apply(lambda r: r.n_umi / umi[r.parent] if isinstance(r.parent, str) else np.nan, axis=1)
cl = top[top.called]
print("called:", len(cl), "in authors", int(cl.bc.isin(aset).sum()), "hd1-child of a larger barcode", int(cl.hd1_child.sum()), "of which parent in authors", int(cl[cl.hd1_child].parent.isin(aset).sum()))
nonauth = cl[~cl.bc.isin(aset)]
print("non-author called:", len(nonauth), "hd1 child:", int(nonauth.hd1_child.sum()), "umi median", float(nonauth.n_umi.median()), "umi q10/q90", float(nonauth.n_umi.quantile(.1)), float(nonauth.n_umi.quantile(.9)))
print("non-author non-hd1-child:", int((~nonauth.hd1_child).sum()), "umi median", float(nonauth[~nonauth.hd1_child].n_umi.median()))
print("child/parent umi ratio among non-author hd1-children: quantiles", nonauth[nonauth.hd1_child].ratio.quantile([.1,.5,.9]).round(4).tolist())
# species fractions
Mx = scipy.io.mmread(os.path.join(R, "raw_feature_bc_matrix", "matrix.mtx.gz")).tocsc()
feats = [l.split("\t")[0] for l in gzip.open(os.path.join(R, "raw_feature_bc_matrix", "features.tsv.gz"), "rt")]
bcs = [l.strip().split("-")[0] for l in gzip.open(os.path.join(R, "raw_feature_bc_matrix", "barcodes.tsv.gz"), "rt")]
mouse = np.array([f.startswith("ENSMUSG") for f in feats])
if Mx.shape[0] != len(feats): Mx = Mx.T.tocsc()
idx = {b: i for i, b in enumerate(bcs)}
def frac(bl):
    cols = [idx[b] for b in bl if b in idx]
    X = Mx[:, cols]
    h = np.asarray(X[~mouse].sum(0)).ravel(); m = np.asarray(X[mouse].sum(0)).ravel(); t = h + m
    ok = t > 0
    mn = np.minimum(h, m)[ok] / t[ok]
    return float((mn > 0.1).mean()), int(ok.sum())
for name, sub in (("authors", cl[cl.bc.isin(aset)]), ("hd1-child", nonauth[nonauth.hd1_child]), ("non-author non-child", nonauth[~nonauth.hd1_child]),
                  ("non-author umi<1000", nonauth[nonauth.n_umi < 1000]), ("non-author umi>=1000", nonauth[nonauth.n_umi >= 1000])):
    if len(sub): print("mixed rate", name, frac(list(sub.bc)), "n", len(sub))
# knee on rank curve: where would a log-log steepest-drop knee be
u = q.n_umi.values.astype(float); lr = np.log10(np.arange(1, len(u) + 1)); lu = np.log10(np.maximum(u, 1))
for k in (500, 800, 1000, 1500, 2500):
    print("drop between rank", k, "and", 2 * k, ":", round(float(u[k] / max(1, u[2 * k])), 2))
