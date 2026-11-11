# SPDX-License-Identifier: MIT
"""Sex-independent ambient check for mouse brain nuclei: genes exclusive to one cell class are ambient (upper bound; low native expression exists)
in nuclei of the other class. rho_ub(class A cells) = (marker-B UMI share of A cells) / (marker-B UMI share of the pool), pool = uncalled barcodes with 1-100 UMI.
  python nuc_markers.py RUN_DIR OUT.json
Classes by marker UMI share among called_v2 nuclei: neuronal (Snap25 Syt1 Rbfox3 Slc17a7 Gad1 Gad2 Stmn2 Camk2a), oligodendrocyte (Plp1 Mbp Mobp Mog Mag Cldn11),
astrocyte (Aqp4 Gfap Slc1a2 Aldh1l1), microglia (Csf1r Cx3cr1 P2ry12 Tmem119). A nucleus is assigned to the class whose marker share is highest if >= 4x the next and >= 0.2% of its UMI."""
import gzip, json, sys, numpy as np, pandas as pd, scipy.io as sio
run, out = sys.argv[1:3]; bare = lambda b: b.split("-")[0]
CLS = {"neuron": "Snap25 Syt1 Rbfox3 Slc17a7 Gad1 Gad2 Stmn2 Camk2a", "oligo": "Plp1 Mbp Mobp Mog Mag Cldn11", "astro": "Aqp4 Gfap Slc1a2 Aldh1l1", "microglia": "Csf1r Cx3cr1 P2ry12 Tmem119"}
CLS = {k: set(v.split()) for k, v in CLS.items()}
def load(prefix):
    M = sio.mmread(gzip.open(f"{run}/{prefix}.mtx.gz", "rb")).tocsr()
    f = [l.rstrip("\n").split("\t")[0] for l in gzip.open(f"{run}/{prefix}_features.tsv.gz", "rt")]
    b = [bare(l.split("\t")[0].strip()) for l in gzip.open(f"{run}/{prefix}_barcodes.tsv.gz", "rt")]
    return M, f, b
Me, fe, be = load("exon_counts"); Mi, fi, bi = load("intron_counts")
sym = lambda x: x.split("_")[1] if "_" in x else x
def rows(f, genes): return np.array([sym(x).split("_")[0] in genes for x in f])
D = pd.DataFrame(index=be)
tot_e = np.asarray(Me.sum(0)).ravel(); D["tot"] = pd.Series(tot_e, index=be)
Ti = pd.Series(np.asarray(Mi.sum(0)).ravel(), index=bi); D["tot"] = D.tot + Ti.reindex(D.index).fillna(0)
for k, g in CLS.items():
    ce = np.asarray(Me[rows(fe, g)].sum(0)).ravel(); ci = pd.Series(np.asarray(Mi[rows(fi, g)].sum(0)).ravel(), index=bi)
    D[k] = ce + ci.reindex(D.index).fillna(0).values
cc = pd.read_csv(f"{run}/cell_calls.tsv", sep="\t", dtype={"barcode": str}); cc.index = cc.barcode.map(bare)
D["v2"] = cc.called_v2.reindex(D.index).fillna(0).astype(int)
pool = D[(D.v2 == 0) & (D.tot >= 1) & (D.tot <= 100)]
pshare = {k: pool[k].sum() / pool.tot.sum() for k in CLS}
cells = D[D.v2 == 1].copy(); cells = cells[cells.tot > 0]
S = pd.DataFrame({k: cells[k] / cells.tot for k in CLS})
top = S.idxmax(1); srt = np.sort(S.values, axis=1); ok = (srt[:, -1] >= 0.002) & (srt[:, -1] >= 4 * srt[:, -2].clip(1e-9))
cells["cls"] = np.where(ok, top, "unassigned")
A = pd.read_csv(f"{run}/ambient_contamination.tsv", sep="\t"); A.index = A.barcode.map(bare); cells = cells.join(A[["rho", "rho_total"]])
res = {"pool_marker_share": {k: float(v) for k, v in pshare.items()}, "n_cells": int(len(cells)), "classes": {}}
for c in CLS:
    sub = cells[cells.cls == c]
    if len(sub) < 30: continue
    ests = {}
    for other in CLS:
        if other == c: continue
        ub = (sub[other].sum() / sub.tot.sum()) / pshare[other] if pshare[other] > 0 else None
        ests[other] = {"upper_bound_rho": None if ub is None else float(ub), "marker_umi": int(sub[other].sum())}
    res["classes"][c] = {"n": int(len(sub)), "per_other_class": ests, "singlet_rho_total_mean": float(sub.rho_total.mean()) if sub.rho_total.notna().any() else None, "singlet_rho_mean": float(sub.rho.mean()) if sub.rho.notna().any() else None}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
