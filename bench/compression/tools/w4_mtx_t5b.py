import os, sys, numpy as np
sys.path.insert(0, '../../python')
from singlet._pz import read_1pz
HERE = os.path.dirname(os.path.abspath(__file__)); BC = os.path.dirname(HERE); CORP = os.path.join(BC, "corpus_w4_mtx")
d = sys.argv[1]; p = os.path.join(CORP, d)
L = open(os.path.join(p, "ambient_profile.tsv")).read().split("\n")[1:-1]
v = np.array([float(l.split("\t")[1]) for l in L]); N = len(v); u = v.min()
k = np.rint((v / u - 1) / 2).astype(np.int64)
ex = read_1pz(os.path.join(p, "exon_counts.1pz")); it = read_1pz(os.path.join(p, "intron_counts.1pz"))
cs = set(l.split("\t")[0] for l in open(os.path.join(p, "cell_calls.tsv")).read().split("\n")[1:-1])
iscell = np.array([c in cs for c in ex["colnames"]])
ip = ex["indptr"].astype(np.int64); cn = np.repeat(np.arange(ex["n"]), np.diff(ip)); ix = ex["indices"].astype(np.int64); dv = ex["data"].astype(np.int64)
ipi = it["indptr"].astype(np.int64)
coltot_e = np.add.reduceat(dv, ip[:-1]) if ex["n"] else None
coltot_i = np.array([it["data"][ipi[j]:ipi[j+1]].sum() for j in range(it["n"])], dtype=np.int64)
coltot_e = np.array([ex["data"][ip[j]:ip[j+1]].sum() for j in range(ex["n"])], dtype=np.int64)
print("n implied T", k.sum(), "N", N, " cols", ex["n"], " cells", iscell.sum())
print("non-cell column UMI totals (exon only): min/median/max", coltot_e[~iscell].min(), np.median(coltot_e[~iscell]), coltot_e[~iscell].max(), "; cells min", coltot_e[iscell].min())
print("exon+intron per column tot: non-cell min/median/max", (coltot_e+coltot_i)[~iscell].min(), np.median((coltot_e+coltot_i)[~iscell]), (coltot_e+coltot_i)[~iscell].max())
best = []
for name, tot in (("exon", coltot_e), ("exon+intron", coltot_e + coltot_i)):
    for L_ in [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 10**9]:
        sel_cols = (~iscell) & (tot <= L_)
        sel = sel_cols[cn]
        c = np.bincount(ix[sel], weights=dv[sel], minlength=ex["m"]).astype(np.int64)
        best.append((int((c == k).sum()), name, L_, int(c.sum()), int(sel_cols.sum())))
for b in sorted(best, reverse=True)[:6]: print("match rows %d of %d  tot=%s <= %s  T=%d ncols=%d" % (b[0], N, b[1], b[2], b[3], b[4]))
# differences k - c for the plain non-cell set
sel = (~iscell)[cn]; c = np.bincount(ix[sel], weights=dv[sel], minlength=ex["m"]).astype(np.int64)
diff = k - c
print("k - c(non-cell): min/max", diff.min(), diff.max(), "n>0", int((diff>0).sum()), "n<0", int((diff<0).sum()), "sum diff", int(diff.sum()))
print("hist of diff:", dict(zip(*[x.tolist() for x in np.unique(np.clip(diff,-3,5), return_counts=True)])))
# are implied counts equal to counts of ALL non-cell raw mtx barcodes? that needs gene level - skip
# does k correlate with mtx gene-level sums over non-cell barcodes? done in later step
