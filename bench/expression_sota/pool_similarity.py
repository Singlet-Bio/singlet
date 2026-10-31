# SPDX-License-Identifier: MIT
"""How close is the ambient pool's gene profile to the pseudobulk of the called cells? (nuclei libraries: the pool is nuclear debris)
  python pool_similarity.py OUT.json NAME=RUN_DIR [NAME=RUN_DIR ...]
Per run: ambient_profile.tsv (gene fraction of the 5..100-UMI pool) vs the column sum of filtered_feature_bc_matrix (called cells).
Reports cosine on raw fractions, cosine on sqrt fractions, Jensen-Shannon distance, and the intronic fraction of the cells and of the pool."""
import gzip, json, sys, numpy as np, pandas as pd, scipy.io as sio
out = sys.argv[1]; res = {}
for arg in sys.argv[2:]:
    name, run = arg.split("=", 1)
    a = pd.read_csv(f"{run}/ambient_profile.tsv", sep="\t"); a["g"] = a.feature.str.split("_").str[0]; a = a.groupby("g").ambient_fraction.sum()
    f = pd.read_csv(f"{run}/filtered_feature_bc_matrix/features.tsv.gz", sep="\t", header=None)
    M = sio.mmread(gzip.open(f"{run}/filtered_feature_bc_matrix/matrix.mtx.gz", "rb")).tocsr()
    pb = pd.Series(np.asarray(M.sum(1)).ravel(), index=f[0].astype(str).values); pb = pb.groupby(level=0).sum()
    idx = a.index.union(pb.index); x = a.reindex(idx).fillna(0).values; y = pb.reindex(idx).fillna(0).values
    x = x / x.sum(); y = y / y.sum()
    cos = float(x @ y / np.sqrt((x @ x) * (y @ y))); sx, sy = np.sqrt(x), np.sqrt(y); cos_s = float(sx @ sy / np.sqrt((sx @ sx) * (sy @ sy)))
    m = (x + y) / 2; kl = lambda p, q: float(np.sum(p[p > 0] * np.log(p[p > 0] / q[p > 0])))
    js = float(np.sqrt(0.5 * kl(x, m) + 0.5 * kl(y, m)))
    res[name] = {"cosine": cos, "cosine_sqrt": cos_s, "js_distance": js, "n_genes": int(len(idx))}
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in res[name].items()}, flush=True)
json.dump(res, open(out, "w"), indent=1)
