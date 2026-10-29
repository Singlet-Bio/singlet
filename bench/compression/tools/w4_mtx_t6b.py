import os, sys, numpy as np
sys.path.insert(0, '../../python')
from singlet._pz import read_1pz
HERE = os.path.dirname(os.path.abspath(__file__)); BC = os.path.dirname(HERE); CORP = os.path.join(BC, "corpus_w4_mtx")
def ge(d):
    G = [l.split("\t") for l in open(os.path.join(CORP, d, "gene_expression.tsv")).read().split("\n")[1:-1]]
    return [g[0] for g in G], np.array([float(g[2]) for g in G]), np.array([float(g[3]) for g in G]), np.array([float(g[4]) for g in G]), np.array([float(g[5]) for g in G])
a = ge("GSE309702.GSM9277757"); b = ge("GSE171161.GSM5220137")
print("effective_length identical across the two GRCm39 samples:", a[0] == b[0] and bool((a[1] == b[1]).all()), "; gene ids identical:", a[0] == b[0])
gid, el, cnt, tpm, fpkm = a
nz = cnt > 0
print("FPKM/TPM ratio over nonzero genes: min/max", (fpkm[nz]/tpm[nz]).min(), (fpkm[nz]/tpm[nz]).max(), " count nonzero genes", nz.sum(), "of", len(cnt))
print("count integer-valued:", bool((cnt == np.rint(cnt)).all()))
# against exon 1pz gene sums, among genes
d = "GSE309702.GSM9277757"; p = os.path.join(CORP, d)
r = read_1pz(os.path.join(p, "exon_counts.1pz")); rs = np.bincount(r["indices"].astype(np.int64), weights=r["data"].astype(np.float64), minlength=r["m"])
gs = {}
for n, v in zip(r["rownames"], rs): gs[n.split("_")[0]] = gs.get(n.split("_")[0], 0) + v
s = np.array([gs.get(g, 0) for g in gid])
print("gene_expression count vs exon1pz gene sums: equal on genes with either>0:", int(((s == cnt) & ((s > 0) | (cnt > 0))).sum()), "of", int(((s > 0) | (cnt > 0)).sum()), "; sum count", cnt.sum(), "sum exon", s.sum())
ri = read_1pz(os.path.join(p, "intron_counts.1pz")); rsi = np.bincount(ri["indices"].astype(np.int64), weights=ri["data"].astype(np.float64), minlength=ri["m"])
gi = {}
for n, v in zip(ri["rownames"], rsi): gi[n.split("_")[0]] = gi.get(n.split("_")[0], 0) + v
si = np.array([gi.get(g, 0) for g in gid])
print("count vs exon+intron:", int(((s + si == cnt) & ((s + si > 0) | (cnt > 0))).sum()), "equal; corr(exon+intron,count)", np.corrcoef(s + si, cnt)[0, 1], " corr(exon,count)", np.corrcoef(s, cnt)[0, 1])
# cell_qc vs 1pz columns
Q = [l.split("\t") for l in open(os.path.join(p, "cell_qc_metrics.tsv")).read().split("\n")[1:-1]]
qb = [q[0] for q in Q]; nu = np.array([int(q[1]) for q in Q]); ng = np.array([int(q[2]) for q in Q])
print("cell_qc rows", len(Q), "== exon 1pz columns", len(r["colnames"]), "; same barcodes & order as exon 1pz columns:", qb == r["colnames"])
ip = r["indptr"].astype(np.int64); ce = np.array([r["data"][ip[j]:ip[j+1]].sum() for j in range(r["n"])])
ipi = ri["indptr"].astype(np.int64); ci = np.array([ri["data"][ipi[j]:ipi[j+1]].sum() for j in range(ri["n"])])
print("n_umi == exon col sum:", int((ce == nu).sum()), "; == exon+intron col sum:", int((ce + ci == nu).sum()), "of", len(nu), "; corr", np.corrcoef(ce, nu)[0, 1])
print("n_genes == exon col nnz:", int((np.diff(ip) == ng).sum()), "of", len(ng))
print("sample rows:", Q[:2], "\ncols per sample doublet etc.")
