# SPDX-License-Identifier: MIT
"""Post-hoc diagnostic (labelled, not part of the gate): variants of the Singlet call on hgmm5k_v3 stored counts, scored by the registered posterior F1.
python -P diag_hgmm5k_variants.py CELLCALL.py RUN_DIR TRUTH_DIR"""
import importlib.util, gzip, sys
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sp
sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/jobs/cellcall_truth")
import score as SC
ccp, run, td = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("cellcall", ccp); cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
def rd(n):
    with gzip.open(f"{run}/{n}_counts.mtx.gz", "rb") as f: return sp.csc_matrix(scipy.io.mmread(f))
bcs = [l.strip().split("\t")[0] for l in gzip.open(f"{run}/exon_counts_barcodes.tsv.gz", "rt")]
def agg(n):
    names = [l.strip().split("	")[0].split("_", 1)[0] for l in gzip.open(f"{run}/{n}_counts_features.tsv.gz", "rt")]
    M = rd(n); g = sorted(set(names)); gi = {x: i for i, x in enumerate(g)}
    A = sp.csr_matrix((np.ones(len(names)), ([gi[x] for x in names], np.arange(len(names)))), shape=(len(g), len(names)))
    return g, sp.csc_matrix(A @ M)
g1, E = agg("exon"); g2, I = agg("intron")
gl = sorted(set(g1) | set(g2)); pos = {x: i for i, x in enumerate(gl)}
def lift(g, M):
    L = sp.csr_matrix((np.ones(len(g)), ([pos[x] for x in g], np.arange(len(g)))), shape=(len(gl), len(g)))
    return L @ M
X = sp.csc_matrix(lift(g1, E) + lift(g2, I)).astype(np.int64)
print("matrix", X.shape, flush=True)
cl = pd.read_csv(f"{run}/cell_calls.tsv", sep="\t", dtype={"barcode": str})
called = cl.set_index("barcode")["called"].reindex(bcs).fillna(0).astype(int).to_numpy().astype(bool)
t = pd.read_csv(f"{td}/hgmm5k_v3.truth.tsv.gz", sep="\t", dtype={"barcode": str}, low_memory=False)
_, p = SC.mixture_score(t, nb=2)
u = t.merge(p, on="barcode")
w = u.post.to_numpy(); idx = {b: i for i, b in enumerate(u.barcode)}
def f1(callset):
    a = np.array([callset.get(b, False) for b in u.barcode]); tp = w[a].sum(); P = tp / max(a.sum(), 1); R = tp / w.sum(); return round(P, 4), round(R, 4), round(2 * P * R / (P + R), 4), int(a.sum())
print("edcr", f1(dict(zip(t.barcode, t.edcr == 1))), "cellbender", f1(dict(zip(t.barcode, t.cellbender == 1))), flush=True)
for name, over in (("lower5", {"lower_rule": 5}), ("lower10", {"lower_rule": 10}), ("lower20", {"lower_rule": 20}), ("lower50", {"lower_rule": 50}), ("lower200", {"lower_rule": 200})):
    c, pv, q, par = cc.call_detail(X, called, None, **over)
    print(name, f1(dict(zip(bcs, c))), "lower", par.get("lower"), "retain", par.get("retain_total"), "n_tested", par.get("n_tested"), flush=True)
