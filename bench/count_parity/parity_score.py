"""Singlet versus STARsolo on every counting axis with real truth, paired bootstrap CIs, and a gate verdict.

Axes (lower is better unless marked):
  barnyard.X       wrong-species UMI fraction in pure cells, Singlet X (raw_feature_bc_matrix) vs STARsolo GeneFull
  barnyard.exon    same, Singlet exon layer (exon_counts collapsed by gene) vs STARsolo Gene
  barnyard.intron  same, Singlet intron layer vs STARsolo Velocyto unspliced
  chrY             non-PAR chrY UMIs per million human UMIs in pure human cells (HEK293T is female), X vs GeneFull
  cite (higher)    mean Spearman RNA (log1p CP10k) vs protein (CLR) over the shared marker pairs, X vs GeneFull
Truths and cell sets are those of bench/exact_vs_pseudo/score.py (published Cell Ranger cells; purity from Cell Ranger's
species fraction). Every CI is a paired bootstrap over cells: the same resampled cells for both tools, 95% percentile.

Gate: an axis fails when Singlet is worse than STARsolo beyond its CI (the whole CI of Singlet minus STARsolo lies on the
worse side of zero). Exit status 1 if any axis fails, so a CI job can call this directly.

  python parity_score.py --singlet DIR --starsolo DIR --cr CR.h5 --gtf GENES.gtf --kind barnyard|cite --out OUT.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import numpy as np
import scipy.io
import scipy.sparse as sp
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exact_vs_pseudo"))
import loaders as L  # noqa: E402

C_ADT_GENE = {"cd3": "CD3E", "cd4": "CD4", "cd8a": "CD8A", "cd14": "CD14", "cd16": "FCGR3A", "cd56": "NCAM1", "cd19": "CD19",
              "cd25": "IL2RA", "pd-1": "PDCD1", "cd127": "IL7R", "tigit": "TIGIT", "cd15": "FUT4"}
NBOOT = int(os.environ.get("PARITY_NBOOT", "2000"))


def collapse_by_gene(d, stem):
    """Singlet per-feature layer (exon / intron features 'ENSG..._NAME_...') summed into genes -> Mat (cells x genes)."""
    M = scipy.io.mmread(os.path.join(d, stem + ".mtx.gz")).tocsr()
    with gzip.open(os.path.join(d, stem + "_features.tsv.gz"), "rt") as f:
        ids = [ln.split("\t")[0].split("_")[0] for ln in f]
    uniq, inv = np.unique(ids, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))
    X = (G @ M).T.tocsr().astype(np.float64)
    with gzip.open(os.path.join(d, stem + "_barcodes.tsv.gz"), "rt") as f:
        bcs = np.array([ln.strip().split("-")[0] for ln in f])
    return L.Mat(bcs, uniq, X)


def align(m, bcs):
    idx = {b: i for i, b in enumerate(m.barcodes)}
    rows = np.array([idx.get(b, -1) for b in bcs])
    X = m.X[np.where(rows < 0, 0, rows)]
    return sp.csr_matrix(sp.diags((rows >= 0).astype(float)) @ X)


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def ratio_axis(num_a, den_a, num_b, den_b, rng, scale=1.0):
    """Paired bootstrap of sum(num)/sum(den) for tools a (Singlet) and b (STARsolo) over the same cells."""
    k = len(num_a)
    ra = scale * num_a.sum() / max(den_a.sum(), 1); rb = scale * num_b.sum() / max(den_b.sum(), 1)
    d = np.empty(NBOOT); ba = np.empty(NBOOT); bb = np.empty(NBOOT)
    for i in range(NBOOT):
        ix = rng.integers(0, k, k)
        ba[i] = scale * num_a[ix].sum() / max(den_a[ix].sum(), 1); bb[i] = scale * num_b[ix].sum() / max(den_b[ix].sum(), 1)
        d[i] = ba[i] - bb[i]
    return {"singlet": float(ra), "starsolo": float(rb), "singlet_ci": ci(ba), "starsolo_ci": ci(bb),
            "diff": float(ra - rb), "diff_ci": ci(d), "num_singlet": float(num_a.sum()), "num_starsolo": float(num_b.sum())}


def pure_cells(cr):
    human = np.array([g.startswith("ENSG") for g in cr.genes])
    tot = np.asarray(cr.X.sum(1)).ravel()
    fh = np.asarray(cr.X[:, human].sum(1)).ravel() / np.maximum(tot, 1)
    ok = tot >= 500
    return ok & (fh >= 0.9), ok & (fh <= 0.1)


def wrong_species(m, bcs, pure_h, pure_m):
    X = align(m, bcs)
    ish = np.array([g.startswith("ENSG") for g in m.genes])
    h = np.asarray(X[:, ish].sum(1)).ravel(); mo = np.asarray(X[:, ~ish].sum(1)).ravel()
    sel = pure_h | pure_m
    wrong = np.where(pure_h, mo, np.where(pure_m, h, 0.0))
    return wrong[sel], (h + mo)[sel]


def chry(m, bcs, pure_h, ycols):
    X = align(m, bcs)[pure_h]
    isy = np.array([g in ycols for g in m.genes]); isH = np.array([g.startswith("ENSG") for g in m.genes])
    return np.asarray(X[:, isy].sum(1)).ravel(), np.asarray(X[:, isH].sum(1)).ravel(), dict(zip(m.genes[isy], X[:, isy].sum(0).A1))


def spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def cite_axis(ms, mo, cr, h5, rng):
    ab = L.load_cr_h5(h5, "Antibody Capture")
    P = ab.X.toarray(); a = np.log1p(P); Pclr = a - a.mean(1, keepdims=True)
    sym2id = dict(zip(cr.names, cr.genes))
    chosen = [(p, C_ADT_GENE[p.split("_")[0].lower()], j) for j, p in enumerate(ab.genes) if p.split("_")[0].lower() in C_ADT_GENE]

    def vals(m):
        X = align(m, cr.barcodes); lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
        col = {g: i for i, g in enumerate(m.genes)}; out = {}
        for p, g, j in chosen:
            gid = sym2id.get(g)
            if gid in col:
                out[(p, g)] = (np.log1p(1e4 * X[:, col[gid]].toarray().ravel() / lib), Pclr[:, j])
        return out
    va, vb = vals(ms), vals(mo)
    keys = [k for k in va if k in vb and not np.isnan(spearman(*va[k])) and not np.isnan(spearman(*vb[k]))]
    n = len(cr.barcodes)
    mean = lambda v, ix: np.nanmean([spearman(v[k][0][ix], v[k][1][ix]) for k in keys])  # noqa: E731
    full = np.arange(n)
    ra, rb = mean(va, full), mean(vb, full)
    nb = max(NBOOT // 4, 200); d = np.empty(nb); ba = np.empty(nb); bb = np.empty(nb)
    for i in range(nb):
        ix = rng.integers(0, n, n); ba[i] = mean(va, ix); bb[i] = mean(vb, ix); d[i] = ba[i] - bb[i]
    return {"singlet": float(ra), "starsolo": float(rb), "singlet_ci": ci(ba), "starsolo_ci": ci(bb), "diff": float(ra - rb),
            "diff_ci": ci(d), "pairs": ["~".join(k) for k in keys], "higher_is_better": True,
            "per_pair": {"~".join(k): [spearman(*va[k]), spearman(*vb[k])] for k in keys}}


def verdict(ax):
    lo, hi = ax["diff_ci"]
    if ax.get("higher_is_better"):
        worse, better = hi < 0, ax["diff"] >= 0
    else:
        worse, better = lo > 0, ax["diff"] <= 0
    return "FAIL" if worse else ("PASS_AT_OR_BETTER" if better else "PASS_WITHIN_CI")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--singlet", required=True); ap.add_argument("--starsolo", required=True)
    ap.add_argument("--cr", required=True); ap.add_argument("--gtf", required=True)
    ap.add_argument("--kind", choices=["barnyard", "cite"], required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261106)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cr = L.load_cr_h5(a.cr)
    res = {"singlet": a.singlet, "starsolo": a.starsolo, "kind": a.kind, "n_boot": NBOOT, "axes": {}}
    if a.kind == "barnyard":
        ph, pm = pure_cells(cr)
        res["n_pure_human"], res["n_pure_mouse"] = int(ph.sum()), int(pm.sum())
        layers = [("barnyard.X", lambda: L.load_singlet(a.singlet), lambda: L.load_starsolo(a.starsolo, "GeneFull")),
                  ("barnyard.exon", lambda: collapse_by_gene(a.singlet, "exon_counts"), lambda: L.load_starsolo(a.starsolo, "Gene")),
                  ("barnyard.intron", lambda: collapse_by_gene(a.singlet, "intron_counts"), lambda: L.load_starsolo_velocyto(a.starsolo, "unspliced"))]
        for name, ls, lo in layers:
            ms, mo = ls(), lo()
            wa, ta = wrong_species(ms, cr.barcodes, ph, pm); wb, tb = wrong_species(mo, cr.barcodes, ph, pm)
            res["axes"][name] = ratio_axis(wa, ta, wb, tb, rng)
            if name == "barnyard.X":
                chrom = L.gene_chrom(a.gtf)
                ycols = {g for g, (c, n, t) in chrom.items() if c.endswith("chrY") and g.startswith("ENSG") and "_PAR_Y" not in g}
                ya, ha, ga = chry(ms, cr.barcodes, ph, ycols); yb, hb, gb = chry(mo, cr.barcodes, ph, ycols)
                res["axes"]["chrY"] = ratio_axis(ya, ha, yb, hb, rng, 1e6)
                top = sorted(set(ga) | set(gb), key=lambda g: -(ga.get(g, 0) - gb.get(g, 0)))[:10]
                res["axes"]["chrY"]["top_excess_genes"] = [(chrom.get(g, ("", g, ""))[1] or g, float(ga.get(g, 0)), float(gb.get(g, 0))) for g in top]
    else:
        res["axes"]["cite"] = cite_axis(L.load_singlet(a.singlet), L.load_starsolo(a.starsolo, "GeneFull"), cr, a.cr, rng)
    for ax in res["axes"].values():
        ax["verdict"] = verdict(ax)
    res["gate"] = "FAIL" if any(ax["verdict"] == "FAIL" for ax in res["axes"].values()) else "PASS"
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for k, ax in res["axes"].items():
        print(f"{k:16s} singlet {ax['singlet']:.6g}  starsolo {ax['starsolo']:.6g}  diff {ax['diff']:+.4g} "
              f"[{ax['diff_ci'][0]:+.4g}, {ax['diff_ci'][1]:+.4g}]  {ax['verdict']}")
    print("GATE", res["gate"])
    sys.exit(1 if res["gate"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
