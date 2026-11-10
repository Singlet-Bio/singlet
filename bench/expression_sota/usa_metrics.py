# SPDX-License-Identifier: MIT
"""USA (spliced / unspliced / ambiguous) benchmark of a Singlet run against STARsolo Velocyto and alevin-fry USA.

    python usa_metrics.py PB DATASET SINGLET_RUN_DIR OUT.json [--label NAME]

PB = pipeline-benchmark root on Clipper (competitor runs in PB/runs/DATASET/{starsolo-public-cr,alevinfry}).
Reuses the extractors of bench/public-counts (extract/runio.py). Cells = the barcodes STARsolo (EmptyDrops_CR) calls.

Metrics (pre-registered in docs/research-beta/sota-expression.md):
  primary  pooled per-gene unspliced r (Pearson on log1p of the summed UMI), singlet~starsolo, singlet~alevin-fry,
           starsolo~alevin-fry (the competitors' mutual agreement is the bar).
  co-primary (truth-like, no competitor needed)
           unspliced share U/(S+A+U) per tool (nuclei libraries are intron rich),
           cross-species intron leak on barnyard libraries: fraction of the intron (U) UMIs in a pure-species cell that map
           to the other species' genes, next to the same fraction for exon (S+A) UMIs.
  also     per-cell median r of the unspliced vector, S / U / A shares.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "extract"))
import runio as R  # noqa: E402


def pooled(m: R.GeneMat, idx: np.ndarray) -> np.ndarray:
    return np.asarray(m.X[idx].sum(axis=0)).ravel().astype(float)


def r_log1p(a: np.ndarray, b: np.ndarray) -> float:
    keep = (a > 0) | (b > 0)
    if keep.sum() < 3:
        return float("nan")
    return float(np.corrcoef(np.log1p(a[keep]), np.log1p(b[keep]))[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    keep = (a > 0) | (b > 0)
    return float(spearmanr(a[keep], b[keep])[0])


def cell_r(A, B, idx) -> float:
    ra = []
    sa, sb = A.X[idx].tocsr(), B.X[idx].tocsr()
    for i in range(len(idx)):
        a = sa[i].toarray().ravel()
        b = sb[i].toarray().ravel()
        k = (a > 0) | (b > 0)
        if k.sum() >= 20 and np.std(np.log1p(a[k])) > 0 and np.std(np.log1p(b[k])) > 0:
            ra.append(np.corrcoef(np.log1p(a[k]), np.log1p(b[k]))[0, 1])
    return float(np.median(ra)) if ra else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pb"), ap.add_argument("ds"), ap.add_argument("singlet"), ap.add_argument("out")
    ap.add_argument("--label", default="singlet")
    a = ap.parse_args()
    runs = Path(a.pb) / "runs" / a.ds
    star = runs / "starsolo-public-cr"
    af_dir = runs / ("alevinfry-fw" if a.ds == "pbmc5p_v1" else "alevinfry")
    cells = R.starsolo_cells(star)
    s = R.load_singlet(Path(a.singlet), cells)
    universe = sorted({R.canon_gene(g) for g in R.singlet_known_genes(Path(a.singlet))})
    E, U, A = (s[k].pad(universe) for k in ("E", "U", "A"))
    S = R.GeneMat(universe, E.X - A.X, cells, E.present)
    S.X.data = np.maximum(S.X.data, 0)
    S.X.eliminate_zeros()
    vel = {k: R.load_starsolo(star, "Velocyto", cells, layer=k) for k in ("spliced", "unspliced", "ambiguous")}
    af = R.load_alevinfry(af_dir, cells)
    L = {"singlet": {"S": S, "U": U, "A": A},
         "starsolo": {"S": vel["spliced"], "U": vel["unspliced"], "A": vel["ambiguous"]},
         "alevinfry": {"S": af["S"], "U": af["U"], "A": af["A"]}}
    present = np.ones(len(cells), bool)
    for t in L:
        for k in L[t]:
            present &= L[t][k].present
    idx = np.nonzero(present)[0]
    # common gene axis
    G = sorted(set.intersection(*[set(L[t][k].genes) for t in L for k in "SUA"]))
    Lg = {t: {k: L[t][k].pad(G) for k in "SUA"} for t in L}
    out: dict = {"dataset": a.ds, "label": a.label, "singlet_run": str(a.singlet), "n_cells": int(len(idx)), "n_genes": len(G)}
    SA = {t: R.GeneMat(G, Lg[t]["S"].X + Lg[t]["A"].X, cells, present) for t in Lg}
    for fam, mats in (("unspliced", {t: Lg[t]["U"] for t in Lg}), ("spliced_ambiguous", SA)):
        pv = {t: pooled(m, idx) for t, m in mats.items()}
        pr = {}
        for x, y in itertools.combinations(pv, 2):
            pr[f"{x}~{y}"] = {"pearson_log1p": r_log1p(pv[x], pv[y]), "spearman": spearman(pv[x], pv[y])}
        out[f"{fam}_pooled_gene_r"] = pr
    sub = idx[np.linspace(0, len(idx) - 1, min(len(idx), 400)).astype(int)]
    out["unspliced_cell_median_r"] = {f"{x}~{y}": cell_r(Lg[x]["U"], Lg[y]["U"], sub)
                                      for x, y in itertools.combinations(Lg, 2)}
    sh = {}
    for t in Lg:
        tot = {k: float(Lg[t][k].X[idx].sum()) for k in "SUA"}
        n = sum(tot.values())
        sh[t] = {k: v / n for k, v in tot.items()} | {"umi": n}
    out["layer_shares"] = sh
    out["unspliced_fraction"] = {t: sh[t]["U"] for t in sh}
    # barnyard leak
    if a.ds.startswith("hgmm"):
        isH = np.array([g.startswith("ENSG") for g in G])
        leak = {}
        for t in Lg:
            Xs = {k: Lg[t][k].X[idx] for k in "SUA"}
            Xe, Xu = Xs["S"] + Xs["A"], Xs["U"]
            he, mu_e = np.asarray(Xe[:, isH].sum(axis=1)).ravel(), np.asarray(Xe[:, ~isH].sum(axis=1)).ravel()
            hi, mu_i = np.asarray(Xu[:, isH].sum(axis=1)).ravel(), np.asarray(Xu[:, ~isH].sum(axis=1)).ravel()
            fh = he / np.maximum(he + mu_e, 1)
            pure_h, pure_m = fh >= 0.95, fh <= 0.05
            wrong_i = hi[pure_m].sum() + mu_i[pure_h].sum()
            all_i = (hi + mu_i)[pure_h | pure_m].sum()
            wrong_e = he[pure_m].sum() + mu_e[pure_h].sum()
            all_e = (he + mu_e)[pure_h | pure_m].sum()
            leak[t] = {"intron_leak": float(wrong_i / max(all_i, 1)), "exon_leak": float(wrong_e / max(all_e, 1)),
                       "pure_human": int(pure_h.sum()), "pure_mouse": int(pure_m.sum())}
        out["cross_species_leak"] = leak
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
