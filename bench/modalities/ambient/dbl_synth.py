"""Synthetic heterotypic doublets: add N_SYN columns that are the sum of one human and one mouse called cell, to
(a) Singlet's exon matrix (input of detect_doublets in the binary) and (b) the gene-level matrix (Scrublet, scDblFinder).
Truth = the synthetic columns plus natural cells with minor-species fraction >= 0.2."""
import gzip, os, sys, json
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

N_SYN = 44  # 5% of 880
D = f"{WORK}/dbl"; os.makedirs(D, exist_ok=True)
meta = pd.read_csv(f"{WORK}/cells.csv")
rng = np.random.default_rng(7)
hum = np.where(meta.species == "human")[0]; mou = np.where(meta.species == "mouse")[0]
pairs = np.c_[rng.choice(hum, N_SYN, replace=False), rng.choice(mou, N_SYN, replace=False)]

def augment(X):  # genes x cells csc
    S = sp.hstack([X[:, a] + X[:, b] for a, b in pairs]).tocsc()
    return sp.hstack([X, S]).tocsc()

G = scipy.io.mmread(f"{WORK}/cells.mtx").tocsc()
scipy.io.mmwrite(f"{D}/gene_synth.mtx", augment(G).tocoo())

with gzip.open(f"{SINGLET}/exon_counts.mtx.gz", "rt") as f: E = scipy.io.mmread(f).tocsc()
ebc = pd.read_csv(f"{SINGLET}/exon_counts_barcodes.tsv.gz", sep="\t", header=None)[0].astype(str).str.replace("-1", "", regex=False).values
ci = pd.Series(np.arange(len(ebc)), index=ebc).reindex(meta.barcode).astype(int).values
E = augment(E[:, ci]).tocsc()
E.data = np.minimum(E.data, 65535)
E.indices = E.indices.astype(np.int32); E.indptr = E.indptr.astype(np.int32)
np.array([E.shape[0], E.shape[1], E.nnz], dtype=np.uint32).tofile(f"{D}/exon.hdr")
E.indptr.tofile(f"{D}/exon.indptr"); E.indices.tofile(f"{D}/exon.indices"); E.data.astype(np.uint16).tofile(f"{D}/exon.data")
truth = np.r_[(meta.minor_frac.values >= 0.2), np.ones(N_SYN, bool)]
pd.DataFrame({"col": np.arange(len(truth)), "truth": truth, "synthetic": np.r_[np.zeros(len(meta), bool), np.ones(N_SYN, bool)]}).to_csv(f"{D}/truth.csv", index=False)
print("exon matrix", E.shape, "nnz", E.nnz, " truth positives", int(truth.sum()))
