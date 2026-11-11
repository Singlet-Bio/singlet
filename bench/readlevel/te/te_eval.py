# SPDX-License-Identifier: MIT
"""TE subfamily counts of the engine against scTE and SoloTE on real data with an independent truth.

* ``bulk`` (A375, GSE212945): the same library sequenced on nanopore gives subfamily abundance without short-read multimapping
  (te_longread_truth.py). Each counter's pseudobulk over shared cells is compared to it: Spearman and Pearson of log1p(CPM) over all
  subfamilies the truth sees, and over the young subfamilies where multimapping decides the answer.
* ``twoc`` (mouse ESC, 2C-like cells): cells whose Zscan4 genes are on (STARsolo gene counts, independent of all three counters) are the
  positives. Score = MERVL-int + MT2_Mm molecules over all TE molecules of the cell; AUROC per counter over shared cells.

    python te_eval.py bulk --engine OUT --scte scte.h5ad --solote SoloTE_output_dir --truth truth.tsv --cells barcodes.tsv --out a375.json
    python te_eval.py twoc --engine OUT --scte scte.h5ad --solote DIR --solo-gene Solo.out/Gene/filtered --out mesc.json
Env PZ_SO: path of a built singlet/_pz*.so (reads the engine's .1pz).
"""

from __future__ import annotations

import argparse
import gzip
import importlib.machinery
import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

YOUNG = re.compile(r"^(L1HS|L1PA[2-8]A?|AluY[a-z0-9]*|SVA_[A-F]|HERVK-int|LTR5_Hs|L1Md_T|L1Md_A|L1Md_Gf|IAPEz-int|IAPLTR1_Mm|MMERVK10C-int)$")


def opener(p):
    p = str(p)
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def load_pz():
    so = os.environ["PZ_SO"]
    loader = importlib.machinery.ExtensionFileLoader("singlet._pz", so)
    spec = importlib.util.spec_from_file_location("singlet._pz", so, loader=loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m


def bare(b: str) -> str:
    return b.split("-")[0]


def collapse(names, M):
    """Sum rows of features x cells matrix M by the names' subfamily."""
    subs = sorted(set(names))
    idx = {s: i for i, s in enumerate(subs)}
    rows = np.array([idx[n] for n in names])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=(len(subs), len(rows)))
    return subs, sp.csr_matrix(A @ M)


def engine_te(out: Path):
    pz = load_pz()
    r = pz.read_1pz(str(out / "te_counts.1pz"))
    M = sp.csc_matrix((r["data"], r["indices"], r["indptr"]), shape=(r["m"], r["n"]))
    names = [n.split(":")[0] for n in r["rownames"]]  # "sub:family:class"
    subs, X = collapse(names, M)
    called = set()
    with open(out / "cell_calls.tsv") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ic = hdr.index("is_cell") if "is_cell" in hdr else None
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if ic is None or p[ic].upper() in ("TRUE", "1", "T"):
                called.add(bare(p[0]))
    return subs, X, [bare(b) for b in r["colnames"]], called


def scte_te(path: Path, te_subs: set):
    import anndata as ad

    a = ad.read_h5ad(path)  # cells x features (genes and TE subfamilies)
    keep = [i for i, v in enumerate(a.var_names) if v in te_subs]
    X = sp.csr_matrix(a.X[:, keep]).T.tocsr()
    return [a.var_names[i] for i in keep], X, [bare(b) for b in a.obs_names]


def solote_sub(name: str):
    """Subfamily of a SoloTE feature, or None for a gene. SoloTE writes loci as 'SoloTE|chr|start|end|sub:family:class|strand'-like
    strings and subfamily-level (multi-mapped) features as 'SoloTE|sub:family:class' variants; the token with two ':' is the TE."""
    if "|" not in name and ":" not in name:
        return None
    for tok in re.split(r"[|]", name):
        if tok.count(":") >= 1 and not tok.startswith("chr"):
            return tok.split(":")[0]
    return None


def solote_te(d: Path):
    mtx = next(d.rglob("matrix.mtx*"))
    base = mtx.parent
    feats = [ln.rstrip("\n").split("\t") for ln in opener(next(base.glob("features.tsv*")))]
    bcs = [bare(ln.strip()) for ln in opener(next(base.glob("barcodes.tsv*")))]
    import scipy.io

    M = scipy.io.mmread(opener(mtx) if str(mtx).endswith(".gz") else str(mtx)).tocsr()
    name = [f[1] if len(f) > 1 else f[0] for f in feats]
    subs = [solote_sub(n) for n in name]
    keep = [i for i, s in enumerate(subs) if s]
    s2, X = collapse([subs[i] for i in keep], M[keep])
    return s2, X, bcs, name[:3] + [name[i] for i in keep[:3]]


def align(subs, X, bcs, cells):
    col = {b: i for i, b in enumerate(bcs)}
    sel = [col[c] for c in cells if c in col]
    return dict(zip(subs, np.asarray(X[:, sel].sum(axis=1)).ravel())), X[:, sel]


def lcpm(d: dict, keys):
    v = np.array([d.get(k, 0.0) for k in keys], float)
    return np.log1p(v / max(v.sum(), 1) * 1e6)


def bulk(a):
    truth = {}
    for ln in open(a.truth):
        if ln.startswith("subfamily"):
            continue
        s, n = ln.split("\t")
        truth[s] = float(n)
    es, ex, eb, called = engine_te(Path(a.engine))
    tools = {"engine": (es, ex, eb)}
    tools["scTE"] = scte_te(Path(a.scte), set(es) | set(truth))
    s2, x2, b2, sample = solote_te(Path(a.solote))
    tools["SoloTE"] = (s2, x2, b2)
    cells = set(called)
    for s, x, b in tools.values():
        cells &= set(b)
    if a.cells:
        cells &= {bare(ln.strip()) for ln in opener(a.cells)}
    cells = sorted(cells)
    keys = sorted(k for k, v in truth.items() if v >= a.min_truth)
    young = [k for k in keys if YOUNG.match(k)]
    t_all, t_young = lcpm(truth, keys), lcpm(truth, young)
    res = {"cells_shared": len(cells), "subfamilies": len(keys), "young": young, "solote_feature_sample": sample, "tools": {}}
    for k, (s, x, b) in tools.items():
        pb, _ = align(s, x, b, cells)
        v = lcpm(pb, keys)
        vy = lcpm(pb, young)
        res["tools"][k] = {
            "te_molecules": float(sum(pb.values())),
            "spearman": round(spearmanr(t_all, v)[0], 4),
            "pearson_log": round(pearsonr(t_all, v)[0], 4),
            "young_spearman": round(spearmanr(t_young, vy)[0], 4) if len(young) > 3 else None,
            "young_pearson_log": round(pearsonr(t_young, vy)[0], 4) if len(young) > 3 else None,
            "subfamilies_detected": int(sum(1 for kk in keys if pb.get(kk, 0) > 0)),
        }
    return res


def auroc(score, label):
    from scipy.stats import rankdata

    r = rankdata(score)
    n1 = label.sum()
    n0 = len(label) - n1
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def twoc(a):
    import scipy.io

    gd = Path(a.solo_gene)
    feats = [ln.rstrip("\n").split("\t") for ln in opener(next(gd.glob("features.tsv*")))]
    gb = [bare(ln.strip()) for ln in opener(next(gd.glob("barcodes.tsv*")))]
    mtx = next(gd.glob("matrix.mtx*"))
    G = scipy.io.mmread(opener(mtx) if str(mtx).endswith(".gz") else str(mtx)).tocsr()
    zrows = [i for i, f in enumerate(feats) if len(f) > 1 and f[1].startswith("Zscan4")]
    z = np.asarray(G[zrows].sum(axis=0)).ravel()
    zof = dict(zip(gb, z))
    es, ex, eb, called = engine_te(Path(a.engine))
    tools = {"engine": (es, ex, eb), "scTE": scte_te(Path(a.scte), set(es))}
    s2, x2, b2, sample = solote_te(Path(a.solote))
    tools["SoloTE"] = (s2, x2, b2)
    cells = set(called) & set(gb)
    for s, x, b in tools.values():
        cells &= set(b)
    cells = sorted(cells)
    label = np.array([zof[c] >= a.zscan4_min for c in cells])
    res = {"cells_shared": len(cells), "positives": int(label.sum()), "zscan4_min": a.zscan4_min, "solote_feature_sample": sample, "tools": {}}
    for k, (s, x, b) in tools.items():
        _, Xs = align(s, x, b, cells)
        idx = {n: i for i, n in enumerate(s)}
        m = [idx[n] for n in ("MERVL-int", "MT2_Mm") if n in idx]
        num = np.asarray(Xs[m].sum(axis=0)).ravel() if m else np.zeros(len(cells))
        den = np.asarray(Xs.sum(axis=0)).ravel()
        score = num / np.maximum(den, 1)
        res["tools"][k] = {"auroc_mervl_fraction": round(auroc(score, label), 4), "auroc_mervl_raw": round(auroc(num, label), 4),
                           "mean_mervl_pos": round(float(num[label].mean()), 3) if label.any() else None,
                           "mean_mervl_neg": round(float(num[~label].mean()), 3)}
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("bulk", "twoc"):
        p = sub.add_parser(n)
        p.add_argument("--engine", required=True)
        p.add_argument("--scte", required=True)
        p.add_argument("--solote", required=True)
        p.add_argument("--out", required=True)
    sub.choices["bulk"].add_argument("--truth", required=True)
    sub.choices["bulk"].add_argument("--cells")
    sub.choices["bulk"].add_argument("--min-truth", type=float, default=5)
    sub.choices["twoc"].add_argument("--solo-gene", required=True)
    sub.choices["twoc"].add_argument("--zscan4-min", type=float, default=3)
    a = ap.parse_args()
    res = bulk(a) if a.cmd == "bulk" else twoc(a)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
