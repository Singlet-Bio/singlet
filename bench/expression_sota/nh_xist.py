# SPDX-License-Identifier: MIT
"""Ambient truth in the Gaublomme 2019 mouse nucleus-hashing library from sex: Xist is expressed only in female nuclei, so Xist UMI in
male-hash nuclei came from ambient RNA. python nh_xist.py RUN_DIR HASHTAG_COUNTS OUT.json
rho_true(male nucleus) = Xist UMI / n_umi / a_X, a_X = Xist share of all UMI in the ambient pool (uncalled barcodes, 1-100 UMI).
Compared with Singlet's per-cell rho_total and rho (ambient_contamination.tsv) and the removed fraction."""
import gzip, json, sys, numpy as np, pandas as pd, scipy.io as sio
run, hf, out = sys.argv[1:4]
bare = lambda b: b.split("-")[0]
def load(prefix):
    M = sio.mmread(gzip.open(f"{run}/{prefix}.mtx.gz", "rb")).tocsr()
    f = [l.rstrip("\n").split("\t")[0] for l in gzip.open(f"{run}/{prefix}_features.tsv.gz", "rt")]
    b = [bare(l.split("\t")[0].strip()) for l in gzip.open(f"{run}/{prefix}_barcodes.tsv.gz", "rt")]
    return M, f, b
Me, fe, be = load("exon_counts"); Mi, fi, bi = load("intron_counts")
xe = np.array(["Xist" in x or "ENSMUSG00000086503" in x for x in fe]); xi = np.array(["Xist" in x or "ENSMUSG00000086503" in x for x in fi])
print("Xist rows exon/intron", xe.sum(), xi.sum())
D = pd.DataFrame({"exon_tot": np.asarray(Me.sum(0)).ravel(), "xist_e": np.asarray(Me[xe].sum(0)).ravel()}, index=be)
D2 = pd.DataFrame({"int_tot": np.asarray(Mi.sum(0)).ravel(), "xist_i": np.asarray(Mi[xi].sum(0)).ravel()}, index=bi)
D = D.join(D2).fillna(0); D["tot"] = D.exon_tot + D.int_tot; D["xist"] = D.xist_e + D.xist_i
cc = pd.read_csv(f"{run}/cell_calls.tsv", sep="\t", dtype={"barcode": str}); cc.index = cc.barcode.map(bare)
D["v2"] = cc.called_v2.reindex(D.index).fillna(0).astype(int); D["knee"] = (cc["called_knee"] if "called_knee" in cc else cc["called"]).reindex(D.index).fillna(0).astype(int)
pool = D[(D.v2 == 0) & (D.tot >= 1) & (D.tot <= 100)]
aX = pool.xist.sum() / pool.tot.sum()
h = pd.read_csv(hf, sep="\t"); h.index = h.barcode.map(bare); T = h.drop(columns="barcode")
D["tag"] = T.idxmax(1).reindex(D.index); D["hmax"] = T.max(1).reindex(D.index).fillna(0)
cells = D[(D.knee == 1) & (D.hmax >= 66)].copy(); cells["xf"] = cells.xist / cells.tot
g = cells.groupby("tag").agg(n=("xf", "size"), xist_frac_median=("xf", "median"), xist_frac_mean=("xf", "mean"))
print(g.to_string())
# female nuclei: median Xist share ~0.24% of UMI; male ~0 (the cut 5e-4 is in the gap)
fem = g.index[g.xist_frac_median > 5e-4].tolist(); mal = [t for t in g.index if t not in fem and g.loc[t, "n"] >= 50]
m = cells[cells.tag.isin(mal)].copy(); m["rho_true"] = m.xf / aX
A = pd.read_csv(f"{run}/ambient_contamination.tsv", sep="\t"); A.index = A.barcode.map(bare); m = m.join(A[["rho", "rho_total", "n_umi", "n_ambient_umi_removed"]])
m["removed"] = m.n_ambient_umi_removed / m.n_umi
res = {"aX_pool_share": float(aX), "female_tags": fem, "male_tags": mal, "n_male_nuclei": int(len(m)), "rho_true_mean": float(m.rho_true.mean()),
       "rho_true_umi_weighted": float(m.xist.sum() / m.tot.sum() / aX)}
for k in ("rho", "rho_total", "removed"):
    res[k] = {"mean": float(m[k].mean()), "ratio": float(m[k].mean() / m.rho_true.mean()), "umi_weighted": float((m[k] * m.tot).sum() / m.tot.sum())}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
