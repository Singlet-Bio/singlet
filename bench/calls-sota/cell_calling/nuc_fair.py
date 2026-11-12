# SPDX-License-Identifier: MIT
"""Fair nuclei test on stored-format input with real hashtag truth (post hoc, labelled; development library, seen before).

Gaublomme 2019 mouse nuclei hashing, 8 hashtags, Singlet run with the stored universe (barcodes with >= 5 reads). Truth class H: cell = total hashtag UMIs >= 100
(above the empty-droplet 99th percentile of 66 recorded in expression-sota/nuclei/nucleus_hash.json), empty = total hashtag UMIs <= 66, between = unknown.
Decided barcodes with >= 50 RNA UMIs. Callers: knee (stored STARsolo-style flag), production called_v2 (engine, with its nuclei rule), python v2.1 default,
python rank-pool candidate, DropletUtils emptyDrops, CellBender (if present).

  python -P nuc_fair.py prep  RUN_DIR HASHTAGS.tsv CELLCALL_V22.py OUT_DIR
  python -P nuc_fair.py score OUT_DIR CELLBENDER_CELL_BARCODES.csv|none OUT.json
"""
import gzip, importlib.util, json, os, sys
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sp
mode = sys.argv[1]
if mode == "prep":
    run, hf, ccp, out = sys.argv[2:6]; os.makedirs(out, exist_ok=True)
    spec = importlib.util.spec_from_file_location("cc", ccp); cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
    def one(n):
        names = [l.strip().split("\t")[0].split("_", 1)[0] for l in gzip.open(f"{run}/{n}_counts_features.tsv.gz", "rt")]
        with gzip.open(f"{run}/{n}_counts.mtx.gz", "rb") as f: M = sp.csc_matrix(scipy.io.mmread(f))
        g = sorted(set(names)); gi = {x: i for i, x in enumerate(g)}
        return g, sp.csc_matrix(sp.csr_matrix((np.ones(len(names)), ([gi[x] for x in names], np.arange(len(names)))), shape=(len(g), len(names))) @ M)
    g1, E = one("exon"); g2, I = one("intron"); gl = sorted(set(g1) | set(g2)); pos = {x: i for i, x in enumerate(gl)}
    lift = lambda g, M: sp.csr_matrix((np.ones(len(g)), ([pos[x] for x in g], np.arange(len(g)))), shape=(len(gl), len(g))) @ M
    X = sp.csc_matrix(lift(g1, E) + lift(g2, I)).astype(np.int64)
    bcs = [l.strip().split("\t")[0].split("-")[0] for l in gzip.open(f"{run}/exon_counts_barcodes.tsv.gz", "rt")]
    tot = np.asarray(X.sum(axis=0)).ravel().astype(np.int64)
    cl = pd.read_csv(f"{run}/cell_calls.tsv", sep="\t", dtype={"barcode": str}).set_index("barcode"); cl.index = cl.index.str.split("-").str[0]
    knee = cl["called_knee"].reindex(bcs).fillna(0).astype(int).to_numpy().astype(bool)
    prod = cl["called_v2"].reindex(bcs).fillna(0).astype(int).to_numpy()
    c21, _, _, p21 = cc.call_detail(X, knee, None)
    crk, _, _, prk = cc.call_detail(X, knee, None, ambient="rank")
    h = pd.read_csv(hf, sep="\t", dtype={"barcode": str}).set_index("barcode"); h.index = h.index.str.split("-").str[0]
    htot = h.sum(axis=1).reindex(bcs).fillna(0).to_numpy()
    df = pd.DataFrame({"barcode": bcs, "umi_total": tot, "htot": htot, "knee": knee.astype(int), "prod_called_v2": prod, "v21": c21.astype(int), "v21_rank_pool": crk.astype(int)})
    df["truth"] = np.where(htot >= 100, "cell", np.where(htot <= 66, "empty", "unknown"))
    df.to_csv(f"{out}/table_nocomp.tsv.gz", sep="\t", index=False)
    json.dump({"v21": p21, "rank": prk}, open(f"{out}/params.json", "w"), indent=1, default=float)
    print("n", len(df), {k: int(df[k].sum()) for k in ("knee", "prod_called_v2", "v21", "v21_rank_pool")}, df.truth.value_counts().to_dict(), p21.get("lower"), prk.get("lower"))
    Gc = X.tocoo(); Gc.row.astype(np.int32).tofile(f"{out}/i.bin"); Gc.col.astype(np.int32).tofile(f"{out}/j.bin"); Gc.data.astype(np.int32).tofile(f"{out}/x.bin")
    open(f"{out}/dims.txt", "w").write(f"{X.shape[0]} {X.shape[1]} {Gc.nnz}\n")
    d = f"{out}/cb_in"; os.makedirs(d, exist_ok=True)
    with gzip.open(f"{d}/matrix.mtx.gz", "wb") as f: scipy.io.mmwrite(f, X, field="integer")
    with gzip.open(f"{d}/barcodes.tsv.gz", "wt") as f: f.write("\n".join(bcs) + "\n")
    with gzip.open(f"{d}/features.tsv.gz", "wt") as f: f.write("\n".join(f"{g}\t{g}\tGene Expression" for g in gl) + "\n")
    s = np.sort(tot)[::-1]; e = int((s >= np.quantile(s[:3000], 0.99) / 10.0).sum()); open(f"{d}/params.txt", "w").write(f"{e} {max(3 * e, 10000)}\n")
else:
    out, cbf, outj = sys.argv[2:5]
    t = pd.read_csv(f"{out}/table.tsv.gz", sep="\t", dtype={"barcode": str})
    if cbf != "none" and os.path.exists(cbf):
        cb = {l.strip().split(",")[0].split("-")[0] for l in open(cbf) if l.strip() and not l.startswith("barcode")}
        t["cellbender"] = t.barcode.isin(cb).astype(int)
    d = t[t.truth.isin(["cell", "empty"]) & (t.umi_total >= 50)].reset_index(drop=True)
    T = (d.truth == "cell").to_numpy(); tot = d.umi_total.to_numpy().astype(float); N = int(T.sum())
    res = {"n_decided": int(len(d)), "n_truth_cells": N, "tools": {}}
    for k in ("knee", "prod_called_v2", "v21", "v21_rank_pool", "edcr", "cellbender"):
        if k not in d: continue
        a = d[k].to_numpy().astype(bool); tp = int((a & T).sum()); P = tp / max(a.sum(), 1); R = tp / N
        o = np.lexsort((-tot, ~a)); s = np.zeros(len(a), bool); s[o[:N]] = True
        res["tools"][k] = dict(n_called_all=int(t[k].sum()), n_called_decided=int(a.sum()), precision=round(P, 4), recall=round(R, 4), f1=round(2 * P * R / max(P + R, 1e-12), 4), recall_at_N=round(float((s & T).sum() / N), 4))
    json.dump(res, open(outj, "w"), indent=1); print(json.dumps(res, indent=1))
