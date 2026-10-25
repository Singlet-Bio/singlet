"""Atlas cell-type centroids over the cell type network's 6,000-gene vocabulary: for every Cell Ontology term with at least
MIN cells in the CELLxGENE training corpus ($WORK/data/celltype_nn: corpus_counts.npz cells x vocab, corpus_obs.parquet),
the mean of up to CAP cells' depth-normalized profiles, sampled across datasets.  Output $LQC/dbl/atlas_centroids.npz.
"""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

D = "/mnt/projects/debruinz_project/debruinz/singlet/data/celltype_nn"
OUT = os.environ.get("LQC", "/mnt/projects/debruinz_project/debruinz/singlet/latent_qc") + "/dbl/atlas_centroids.npz"
MIN, CAP = 50, 400

obs = pd.read_parquet(f"{D}/corpus_obs.parquet")
print(obs.shape, list(obs.columns))
print(obs.head(3).T)
col = next(c for c in obs.columns if obs[c].astype(str).str.startswith("CL:").mean() > 0.9)
ds = next((c for c in obs.columns if "dataset" in c.lower()), None)
print("term column", col, "dataset column", ds)
rng = np.random.default_rng(0)
obs = obs.reset_index(drop=True)
obs["_r"] = rng.random(len(obs))
counts = obs[col].value_counts()
terms = counts.index[counts >= MIN].tolist()
# round-robin over datasets: rank cells within (term, dataset), take the lowest ranks first
key = [col, ds] if ds else [col]
obs["_rank"] = obs.groupby(key)["_r"].rank(method="first")
sel = obs[obs[col].isin(terms)].sort_values([col, "_rank", "_r"]).groupby(col).head(CAP)
X = sp.load_npz(f"{D}/corpus_counts.npz").tocsr()
print("corpus", X.shape, X.nnz)
idx = sel.index.values
Xs = X[idx]
del X
n = np.asarray(Xs.sum(1)).ravel()
Xs = sp.diags(1 / np.maximum(n, 1)) @ Xs
lab = sel[col].values
M = np.zeros((len(terms), Xs.shape[1]), np.float32)
nn = np.zeros(len(terms), np.int64)
ti = {t: i for i, t in enumerate(terms)}
G = sp.csr_matrix((np.ones(len(lab)), ([ti[t] for t in lab], np.arange(len(lab)))), shape=(len(terms), len(lab)))
M = np.asarray((G @ Xs).todense(), np.float32)
nn = np.asarray(G.sum(1)).ravel()
M /= nn[:, None]
name_col = next((c for c in obs.columns if c.startswith("cell_type") and c != col and obs[c].dtype == object), None)
names = sel.groupby(col)[name_col].first().reindex(terms).astype(str).values if name_col else np.array(terms)
np.savez(OUT, terms=np.array(terms), names=names, M=M, n=nn, full_counts=counts.reindex(terms).values)
print("centroids", M.shape, "->", OUT)
print(json.dumps(dict(zip(terms[:10], names[:10].tolist()))))
