# SPDX-License-Identifier: MIT
"""D-150 non-10x validation scorer (bars: PASS_BARS.md).

    python score.py <kind> <out_dir> <truth> [<truth2>]

kinds: barnyard (truth = author HUMAN file, truth2 = author MOUSE file; digital expression, genes x barcodes),
       dge (one digital expression file, genes x barcodes), umifm (Baron csv, cells x genes), bdcsv (BD DBEC MolsPerCell csv).
Prints one line: SCORE {json}.
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import spearmanr

kind, out = sys.argv[1], sys.argv[2]
res = {"kind": kind}
summ = json.load(open(os.path.join(out, "summary.json")))
res.update(status=summ.get("status"), protocol=summ.get("protocol_name"), warnings=summ.get("warnings"),
           mapping=summ.get("mapping_rate"), reads=summ.get("total_reads"))
cc = pd.read_csv(os.path.join(out, "cell_calls.tsv"), sep="\t")
cells = cc[cc.is_cell.astype(str).str.upper() == "TRUE"].barcode.astype(str).str.split("-").str[0].tolist()
res["cells_called"] = len(cells)
q = pd.read_csv(os.path.join(out, "cell_qc_metrics.tsv"), sep="\t", usecols=["barcode", "n_umi", "n_genes"])
q["barcode"] = q.barcode.astype(str).str.split("-").str[0]
cq = q.set_index("barcode").loc[lambda d: d.index.isin(cells)]
res["median_umi"] = float(cq.n_umi.median()) if len(cq) else 0.0
res["median_genes"] = float(cq.n_genes.median()) if len(cq) else 0.0


def read_dge(p):
    return pd.read_csv(p, sep=r"\s+", index_col=0, engine="python")


def load_mtx(stem="exon_counts"):
    M = scipy.io.mmread(os.path.join(out, stem + ".mtx.gz")).tocsr()
    ids = [ln.split("\t")[0] for ln in gzip.open(os.path.join(out, stem + "_features.tsv.gz"), "rt")]
    bc = [ln.strip().split("-")[0] for ln in gzip.open(os.path.join(out, stem + "_barcodes.tsv.gz"), "rt")]
    return M, ids, bc


if kind == "barnyard":
    H, Mo = read_dge(sys.argv[3]), read_dge(sys.argv[4])
    ah, am = H.sum(0), Mo.sum(0)
    tot = ah.add(am, fill_value=0)
    auth = tot.sort_values(ascending=False).head(1020)
    ah, am = ah.reindex(auth.index).fillna(0), am.reindex(auth.index).fillna(0)
    afrac = np.maximum(ah, am) / (ah + am)
    amixed = float((np.minimum(ah, am) / (ah + am) > 0.10).mean())
    M, ids, bc = load_mtx()
    mouse = np.array([i.startswith("ENSMUSG") for i in ids])
    if M.shape[0] != len(ids):
        M = M.T.tocsr()
    bidx = {b: i for i, b in enumerate(bc)}
    called = [b for b in cells if b in bidx]
    X = M[:, [bidx[b] for b in called]]
    hum = np.asarray(X[~mouse].sum(0)).ravel()
    mus = np.asarray(X[mouse].sum(0)).ravel()
    t = hum + mus
    ok = t > 0
    major = np.maximum(hum, mus)[ok] / t[ok]
    minor = np.minimum(hum, mus)[ok] / t[ok]
    aset = set(auth.index.astype(str))
    inter = [b for b in called if b in aset]
    res.update(n_author=1020, recall=len(inter) / 1020, precision=len(inter) / max(1, len(called)),
               median_major=float(np.median(major)), share_major_ge_090=float((major >= 0.9).mean()),
               mixed_rate=float((minor > 0.10).mean()), author_mixed_rate=amixed,
               author_median_major=float(np.median(afrac)), n_called_with_umi=int(ok.sum()))
    # supplementary (not a bar): mixed-cell rate restricted to the authors' cell set
    pos = {b: k for k, b in enumerate(called)}
    ii = [pos[b] for b in inter if ok[pos[b]]]
    if ii:
        mi = np.minimum(hum, mus)[ii] / t[ii]
        res["supp_mixed_rate_author_cells"] = float((mi > 0.10).mean())
    if len(inter) > 10:
        ours = pd.Series(t, index=called).reindex(inter)
        res["umi_spearman"] = float(spearmanr(ours.values, auth.reindex(inter).values)[0])
    c = res["cells_called"] / 1020
    res["bars"] = {"cells_0.7_1.3": 0.7 <= c <= 1.3, "recall>=0.7": res["recall"] >= 0.7,
                   "precision>=0.8": res["precision"] >= 0.8, "median_major>=0.95": res["median_major"] >= 0.95,
                   "share_pure>=0.9": res["share_major_ge_090"] >= 0.9,
                   "mixed_within_3pp": abs(res["mixed_rate"] - amixed) <= 0.03,
                   "spearman>=0.9": res.get("umi_spearman", 0) >= 0.9, "status_success": summ.get("status") == "success"}
elif kind == "dgebc":
    # author matrices whose columns carry the cell barcode (Mereu 2020 Drop-seq "Dropseq_N701_ACGT..." in two files, human and mouse, same columns;
    # Hughes 2020 Seq-Well, comma separated, columns = barcodes). argv[3] = "prefix=<column prefix>;sep=<ws|comma>", argv[4:] = files, summed.
    # Bars as the ordinary libraries (cells 0.6-1.6x, genes 0.4-1.25x, status success) plus barcode recall >= 0.70 and precision >= 0.80.
    opts = dict(kv.split("=", 1) for kv in sys.argv[3].split(";") if "=" in kv)
    prefix, sep = opts.get("prefix", ""), opts.get("sep", "ws")
    tabs = []
    for pth in sys.argv[4:]:
        d = pd.read_csv(pth, sep=("," if sep == "comma" else r"\s+"), index_col=0, engine="python" if sep != "comma" else "c")
        d = d[[c for c in d.columns if str(c).startswith(prefix)]] if prefix else d
        d.columns = [str(c)[len(prefix):].split("-")[0] for c in d.columns]
        tabs.append(d)
    T = tabs[0]
    for d in tabs[1:]:
        T = pd.concat([T, d], axis=0, sort=False)   # different species: rows are different genes, columns the same barcodes
    T = T.fillna(0)
    T = T.groupby(level=0).sum() if T.index.has_duplicates else T
    ac = T.shape[1]
    aset = set(T.columns.astype(str))
    ag = float(np.median((T.values > 0).sum(0)))
    au = float(np.median(T.values.sum(0)))
    comp = str.maketrans("ACGT", "TGCA")
    rc = lambda b: b.translate(comp)[::-1]
    direct = len(set(cells) & aset)
    flipped = len({rc(b) for b in cells} & aset)
    orient = "direct" if direct >= flipped else "revcomp"
    inter = max(direct, flipped)
    res.update(author_cells=int(ac), author_median_genes=ag, author_median_umi=au, recall=inter / max(1, ac), precision=inter / max(1, len(cells)),
               barcode_orientation=orient, recall_direct=direct / max(1, ac), recall_revcomp=flipped / max(1, ac))
    r = res["cells_called"] / max(1, ac)
    g = res["median_genes"] / max(1.0, ag)
    res.update(cell_ratio=r, genes_ratio=g, umi_ratio=res["median_umi"] / max(1.0, au))
    res["bars"] = {"cells_ratio": 0.6 <= r <= 1.6, "genes_ratio_0.4_1.25": 0.4 <= g <= 1.25, "recall>=0.7": res["recall"] >= 0.7,
                   "precision>=0.8": res["precision"] >= 0.8, "status_success": summ.get("status") == "success"}
    if opts.get("bars") == "ordinary":   # barcodes of the deposit are not comparable (inDrop: padded canonical barcodes): ordinary-library bars only
        res["bars"].pop("recall>=0.7")
        res["bars"].pop("precision>=0.8")
elif kind == "mtx":
    # a Matrix Market author deposit (genes x cells or cells x genes) with its genes/features and barcodes files; ordinary-library bars
    paths = sys.argv[3:]
    mp = [p for p in paths if ".mtx" in os.path.basename(p)][0]
    n_lines = lambda p: sum(1 for _ in (gzip.open(p, "rt") if p.endswith(".gz") else open(p)))
    gp = [p for p in paths if "genes" in os.path.basename(p) or "features" in os.path.basename(p)][0]
    bp = [p for p in paths if "barcodes" in os.path.basename(p)][0]
    ng, nb = n_lines(gp), n_lines(bp)
    Mt = scipy.io.mmread(mp).tocsc()
    X = Mt if Mt.shape == (ng, nb) else Mt.T.tocsc()
    ac = X.shape[1]
    ag = float(np.median(np.diff(X.indptr)))
    au = float(np.median(np.asarray(X.sum(0)).ravel()))
    res.update(author_cells=int(ac), author_median_genes=ag, author_median_umi=au)
    r = res["cells_called"] / max(1, ac)
    g = res["median_genes"] / max(1.0, ag)
    res.update(cell_ratio=r, genes_ratio=g, umi_ratio=res["median_umi"] / max(1.0, au))
    res["bars"] = {"cells_ratio": 0.6 <= r <= 1.6, "genes_ratio_0.4_1.25": 0.4 <= g <= 1.25, "status_success": summ.get("status") == "success"}
else:
    if kind == "dge":
        T = read_dge(sys.argv[3])
        ac = T.shape[1]
        ag = float(np.median((T > 0).sum(0)))
        au = float(np.median(T.sum(0)))
    else:
        if kind == "bdcsv":   # union of the per-tag matrices of one cartridge (genes may differ per file: align on columns)
            T = pd.concat([pd.read_csv(p, index_col=0, comment="#") for p in sys.argv[3:]], axis=0, sort=False).fillna(0)
        else:
            T = pd.read_csv(sys.argv[3], index_col=0)
        T = T.select_dtypes("number").drop(columns=["assigned_cluster"], errors="ignore")
        ac = T.shape[0]
        ag = float(np.median((T.values > 0).sum(1)))
        au = float(np.median(T.values.sum(1)))
    res.update(author_cells=int(ac), author_median_genes=ag, author_median_umi=au)
    lo, hi = (0.8, 6.0) if kind == "bdcsv" else (0.6, 1.6)
    r = res["cells_called"] / max(1, ac)
    g = res["median_genes"] / max(1.0, ag)
    res.update(cell_ratio=r, genes_ratio=g, umi_ratio=res["median_umi"] / max(1.0, au))
    res["bars"] = {"cells_ratio": lo <= r <= hi, "genes_ratio_0.4_1.25": 0.4 <= g <= 1.25,
                   "status_success": summ.get("status") == "success"}
res["PASS"] = all(res["bars"].values())
print("SCORE " + json.dumps(res))
