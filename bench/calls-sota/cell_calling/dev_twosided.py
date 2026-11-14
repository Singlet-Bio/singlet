# SPDX-License-Identifier: MIT
"""Post-hoc candidate scored on the development libraries: Singlet call (v2.1 default) against the rank-based ambient pool, both scored by the registered
posterior F1 and matched-count recall, next to the stored EmptyDrops_CR and CellBender calls.
python -P dev_rank.py CELLCALL_V22.py OUT.json"""
import gzip, importlib.util, json, sys
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sp
sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/jobs/cellcall_truth")
import score as SC
W = "/mnt/projects/debruinz_project/debruinz/singlet"
spec = importlib.util.spec_from_file_location("cc", sys.argv[1]); cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
RUNS = {"kang_b1A": f"{W}/bench-public/runs/kang2018_b1_a/singlet", "kang_b1B": f"{W}/bench-public/runs/kang2018_b1_b/singlet",
        "kang_b1C": f"{W}/bench-public/runs/kang2018_b1_c/singlet", "kang_b2ctrl": f"{W}/pipeline-benchmark/runs/kang2018_ctrl/singlet-public",
        "hgmm1k_v2": f"{W}/bench-public/runs/hgmm1k_v2/singlet", "hgmm1k_v3": f"{W}/pipeline-benchmark/runs/hgmm1k_v3/singlet-public",
        "hgmm5k_v3": f"{W}/bench-public/runs/hgmm5k_v3/singlet"}
def load(run):
    def one(n):
        names = [l.strip().split("\t")[0].split("_", 1)[0] for l in gzip.open(f"{run}/{n}_counts_features.tsv.gz", "rt")]
        with gzip.open(f"{run}/{n}_counts.mtx.gz", "rb") as f: M = sp.csc_matrix(scipy.io.mmread(f))
        g = sorted(set(names)); gi = {x: i for i, x in enumerate(g)}
        return g, sp.csc_matrix(sp.csr_matrix((np.ones(len(names)), ([gi[x] for x in names], np.arange(len(names)))), shape=(len(g), len(names))) @ M)
    g1, E = one("exon"); g2, I = one("intron"); gl = sorted(set(g1) | set(g2)); pos = {x: i for i, x in enumerate(gl)}
    lift = lambda g, M: sp.csr_matrix((np.ones(len(g)), ([pos[x] for x in g], np.arange(len(g)))), shape=(len(gl), len(g))) @ M
    bcs = [l.strip().split("\t")[0].split("-")[0] for l in gzip.open(f"{run}/exon_counts_barcodes.tsv.gz", "rt")]
    return sp.csc_matrix(lift(g1, E) + lift(g2, I)).astype(np.int64), bcs
R = {}
for ds, run in RUNS.items():
    X, bcs = load(run)
    cl = pd.read_csv(f"{run}/cell_calls.tsv", sep="\t", dtype={"barcode": str}); cl["barcode"] = cl.barcode.str.split("-").str[0]
    called = cl.set_index("barcode")["called"].reindex(bcs).fillna(0).astype(int).to_numpy().astype(bool)
    t = pd.read_csv(f"/mnt/projects/debruinz_project/debruinz/singlet/state/cellcall_truth/truth/{ds}.truth.tsv.gz", sep="\t", dtype={"barcode": str}, low_memory=False)
    t["barcode"] = t.barcode.str.split("-").str[0]
    _, p = SC.mixture_score(t, nb=2); p["barcode"] = p.barcode.str.split("-").str[0]
    u = t.merge(p, on="barcode"); w = u.post.to_numpy(); tot = u.umi_total.to_numpy().astype(float); N = int(round(w.sum()))
    def sc(a):
        a = np.asarray(a, bool); tp = w[a].sum(); P = tp / max(a.sum(), 1); Rr = tp / w.sum()
        o = np.lexsort((-tot, ~a)); s = np.zeros(len(a), bool); s[o[:N]] = True
        return dict(n=int(a.sum()), P=round(P, 4), R=round(Rr, 4), F1=round(2 * P * Rr / max(P + Rr, 1e-12), 4), recallN=round(w[s].sum() / w.sum(), 4))
    res = {"edcr": sc(u.edcr == 1)}
    if "cellbender" in u and u.cellbender.notna().any(): res["cellbender"] = sc(u.cellbender == 1)
    for name, over in (("default", {}), ("two_sided", {"two_sided": True}), ("rank_two_sided", {"ambient": "rank", "two_sided": True})):
        c, pv, q, par = cc.call_detail(X, called, None, **over)
        m = dict(zip(bcs, c)); res[name] = sc([m.get(b, False) for b in u.barcode]); res[name]["lower"] = par.get("lower"); res[name]["n_ambient"] = par.get("n_ambient")
    R[ds] = res; print(ds, json.dumps(res), flush=True)
json.dump(R, open(sys.argv[2], "w"), indent=1, default=float)
