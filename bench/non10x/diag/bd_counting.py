# SPDX-License-Identifier: MIT
# D-152 diagnostic: what counting choices move the BD WTA median genes per cell? usage: bd_counting.py RES_DIR
import sys, os, numpy as np, pandas as pd, scipy.io as sio, scipy.sparse as sp, gzip
R = sys.argv[1]
def load(name):
    m = sio.mmread(os.path.join(R, name + ".mtx.gz")).tocsc()
    b = pd.read_csv(os.path.join(R, name + "_barcodes.tsv.gz"), header=None)[0].astype(str).values
    f = pd.read_csv(os.path.join(R, name + "_features.tsv.gz"), header=None, sep="\t")[0].astype(str).values
    return m, b, f
cc = pd.read_csv(os.path.join(R, "cell_calls.tsv"), sep="\t", usecols=["barcode", "is_cell"])
cells = set(cc.barcode[cc.is_cell.astype(str).str.upper() == "TRUE"].astype(str))
E, be, fe = load("exon_counts"); I, bi, fi = load("intron_counts")
print("matrix shapes exon", E.shape, "intron", I.shape, "called cells", len(cells))
sel = np.array([b in cells for b in be]); E = E[:, sel]; be = be[sel]
idx = {b: i for i, b in enumerate(bi)}; I = I[:, [idx[b] for b in be if b in idx]]
def gname(x): return np.array([s.split("_")[0] for s in x])
ge = gname(fe); gi = gname(fi)
def collapse(M, g):
    u, inv = np.unique(g, return_inverse=True)
    return sp.csr_matrix((np.ones(len(g)), (inv, np.arange(len(g)))), shape=(len(u), len(g))) @ M, u
Ge, ue = collapse(E, ge); Gi, ui = collapse(I, gi)
def stats(label, M):
    M = sp.csc_matrix(M); umi = np.asarray(M.sum(0)).ravel(); gn = np.diff(M.indptr)
    print("%-34s cells %d  median UMIs %.0f  median genes %.0f  genes>=1 over cells %d" % (label, M.shape[1], np.median(umi), np.median(gn), int((np.asarray(M.sum(1)).ravel() > 0).sum())))
stats("exon only (feature level)", E)
stats("exon only (gene level)", Ge)
al = pd.Index(ue).union(pd.Index(ui)); 
def reindex(M, u):
    pos = al.get_indexer(u); R_ = sp.csr_matrix((np.ones(len(u)), (pos, np.arange(len(u)))), shape=(len(al), len(u))); return R_ @ M
S = reindex(Ge, ue) + reindex(Gi, ui)[:, :Ge.shape[1]] if Gi.shape[1] == Ge.shape[1] else reindex(Ge, ue)
stats("exon + intron (gene level)", S)
umi = np.asarray(sp.csc_matrix(Ge).sum(0)).ravel(); order = np.argsort(-umi)
for k in (1000, 3000):
    stats("exon, top %d cells by UMI" % k, sp.csc_matrix(Ge)[:, order[:k]])
