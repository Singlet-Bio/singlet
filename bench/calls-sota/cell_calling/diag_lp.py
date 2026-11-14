# SPDX-License-Identifier: MIT
"""Post-hoc diagnostic: observed DM log-probability against the simulated null for pure cells with p near 1 (hgmm5k_v3). python -P diag_lp.py CELLCALL.py RUN_DIR TRUTH"""
import gzip, importlib.util, sys
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sp
ccp, run, tr = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("cc", ccp); cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
sys.argv = ["x", ccp, "/dev/null"]
def one(n):
    names = [l.strip().split("\t")[0].split("_", 1)[0] for l in gzip.open(f"{run}/{n}_counts_features.tsv.gz", "rt")]
    with gzip.open(f"{run}/{n}_counts.mtx.gz", "rb") as f: M = sp.csc_matrix(scipy.io.mmread(f))
    g = sorted(set(names)); gi = {x: i for i, x in enumerate(g)}
    return g, sp.csc_matrix(sp.csr_matrix((np.ones(len(names)), ([gi[x] for x in names], np.arange(len(names)))), shape=(len(g), len(names))) @ M)
g1, E = one("exon"); g2, I = one("intron"); gl = sorted(set(g1) | set(g2)); pos = {x: i for i, x in enumerate(gl)}
lift = lambda g, M: sp.csr_matrix((np.ones(len(g)), ([pos[x] for x in g], np.arange(len(g)))), shape=(len(gl), len(g))) @ M
X = sp.csc_matrix(lift(g1, E) + lift(g2, I)).astype(np.int64)
bcs = np.array([l.strip().split("\t")[0].split("-")[0] for l in gzip.open(f"{run}/exon_counts_barcodes.tsv.gz", "rt")])
tot = np.asarray(X.sum(axis=0)).ravel()
cl = pd.read_csv(f"{run}/cell_calls.tsv", sep="\t", dtype={"barcode": str}); cl["barcode"] = cl.barcode.str.split("-").str[0]
called = cl.set_index("barcode")["called"].reindex(bcs).fillna(0).astype(int).to_numpy().astype(bool)
lower = 100
amb = (tot > 0) & (tot <= lower) & ~called
asum = np.asarray(X[:, amb].sum(axis=1)).ravel(); prop = cc.good_turing_proportions(asum); alpha = cc.dm_alpha_mle(X[:, amb], prop)
print("alpha", alpha, "pool", int(amb.sum()), "max prop", prop.max(), "mass of top 50 genes", np.sort(prop)[::-1][:50].sum())
t = pd.read_csv(tr, sep="\t", dtype={"barcode": str}, low_memory=False)
bad = t[(t.edcr == 1) & (t.called_v2 == 0) & (t.umi_total >= 5000) & (t.umi_total < 9000)].barcode.head(6).tolist()
idx = [int(np.where(bcs == b)[0][0]) for b in bad if (bcs == b).any()]
ut = np.array(sorted(set(int(tot[i]) for i in idx)))
prob, alias = cc.alias_table(prop); sims = cc._mc_kernel()(prob, alias, prop, float(alpha), ut.astype(np.int64), 2000, 20261023)
lp = cc.dm_logprob(X[:, idx], prop, alpha)
for k, i in enumerate(idx):
    c = np.searchsorted(ut, tot[i]); s = sims[:, c]
    print(bcs[i], "tot", tot[i], "obs lp", round(lp[k], 1), "sim mean", round(s.mean(), 1), "sd", round(s.std(), 1), "sim max", round(s.max(), 1), "z", round((lp[k] - s.mean()) / s.std(), 1))
# the same for the DM log-prob of a simulated draw computed by dm_logprob (consistency of path formula and dm_logprob)
rng = np.random.default_rng(1); n = int(tot[idx[0]]); cnt = rng.multinomial(n, prop)
Y = sp.csc_matrix(cnt.reshape(-1, 1)); print("multinomial draw dm_logprob", round(float(cc.dm_logprob(Y, prop, alpha)[0]), 1), "path sims mean at same total", round(float(sims[:, np.searchsorted(ut, tot[idx[0]])].mean()), 1))
