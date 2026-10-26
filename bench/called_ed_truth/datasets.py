# SPDX-License-Identifier: MIT
"""called_ed against ground truth on the pipeline-benchmark / modality-bench datasets (docs/called-ed-truth.md).

For one Singlet output folder: rebuild the exon+intron gene matrix, reproduce the production ``called_ed`` count, compute
retain/FDR variants from one Monte Carlo run, score every variant against the truth set (Cell Ranger filtered barcodes, the
demuxlet table for Kang 2018, CellBender), and classify the barcodes a variant calls beyond the truth set:

* ambient-like: the barcode's counts are better explained by the ambient profile than by any truth-cell cluster profile
  (multinomial log-likelihood ratio <= 0), or, in the barnyard, its human fraction is consistent with the ambient one;
* doublet: Singlet donor demux calls it a doublet (Kang), or in the barnyard it is mixed-species but unlike ambient;
* damaged: mitochondrial share >= 20%;
* likely cell: the rest.

Usage: python -P datasets.py NAME RUN_DIR --truth KIND:PATH [--truth KIND:PATH ...] --out OUT_DIR
KIND = cellranger (filtered h5 or barcodes.tsv[.gz]), kang (tsne.df.tsv.gz, ctrl rows), cellbender (cell_barcodes.csv).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402


def strip(b):
    return b.split("-")[0]


def truth_set(kind, path):
    if kind == "cellranger":
        if path.endswith(".h5"):
            import h5py

            with h5py.File(path, "r") as h:
                grp = h["matrix"] if "matrix" in h else h[list(h.keys())[0]]
                return {strip(b.decode()) for b in grp["barcodes"][:]}
        with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as fh:
            return {strip(ln.strip()) for ln in fh if ln.strip()}
    if kind == "kang":
        df = pd.read_csv(path, sep="\t")
        df = df[df["stim"] == "ctrl"]
        return {strip(b) for b in df.index.astype(str)}
    if kind == "cellbender":
        with open(path) as fh:
            return {strip(ln.strip().split(",")[0]) for ln in fh if ln.strip() and not ln.startswith("barcode")}
    raise ValueError(kind)


def kang_table(path):
    df = pd.read_csv(path, sep="\t")
    df = df[df["stim"] == "ctrl"].copy()
    df.index = [strip(b) for b in df.index.astype(str)]
    return df


def q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {str(k): float(np.quantile(x, k)) for k in qs} if len(x) else None


def cluster_profiles(X, cells, k_max=12, seed=0):
    """Summed-count profiles of k-means clusters of the truth cells (log CP10k of the 2,000 most expressed genes, 30 SVD dims)."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD

    Xc = X[:, cells].T.tocsr().astype(np.float64)
    tot = np.asarray(Xc.sum(axis=1)).ravel()
    top = np.argsort(-np.asarray(Xc.sum(axis=0)).ravel())[:2000]
    Z = Xc[:, top].multiply(1e4 / np.maximum(tot, 1)[:, None]).tocsr()
    Z.data = np.log1p(Z.data)
    k = int(max(2, min(k_max, cells.sum() // 50)))
    E = TruncatedSVD(n_components=min(30, Z.shape[1] - 1), random_state=seed).fit_transform(Z)
    lab = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(E)
    P = []
    for c in range(k):
        s = np.asarray(Xc[lab == c].sum(axis=0)).ravel()
        P.append(s / s.sum())
    return np.array(P), lab


def llr_cell_vs_ambient(X, idx, P, amb, eps=1e-7):
    """max over cluster profiles of the multinomial log-likelihood minus that under the ambient profile, per barcode."""
    Y = X[:, idx].T.tocsr()
    la = np.log(amb + eps)
    lc = np.log(0.99 * P + 0.01 * amb[None, :] + eps)  # clusters smoothed toward ambient (cells carry ambient RNA too)
    ll_amb = Y @ la
    ll_c = (Y @ lc.T)
    return np.asarray(ll_c.max(axis=1)).ravel() - np.asarray(ll_amb).ravel()


def calibrate_llr(X, cells, P, amb, target_totals, rng, n=600):
    """Share with LLR > 0 among (a) truth cells binomially downsampled to the target totals, (b) multinomial draws from the
    ambient profile at the same totals. (a) estimates sensitivity at that depth, (b) the false-positive rate on pure ambient."""
    import scipy.sparse as sp

    ci = np.nonzero(cells)[0]
    pick = rng.choice(ci, size=min(n, len(ci)), replace=False)
    tt = rng.choice(target_totals, size=len(pick), replace=True)
    cols, sims = [], []
    for j, t in zip(pick, tt):
        col = X[:, j].toarray().ravel()
        tot = col.sum()
        if tot <= t:
            cols.append(col)
        else:  # draw t molecules without replacement
            idx = np.repeat(np.arange(len(col)), col)
            keep = rng.choice(idx, size=int(t), replace=False)
            cols.append(np.bincount(keep, minlength=len(col)))
        sims.append(rng.multinomial(int(t), amb))
    D = sp.csc_matrix(np.array(cols).T)
    S = sp.csc_matrix(np.array(sims).T)
    a = llr_cell_vs_ambient(D, np.arange(D.shape[1]), P, amb)
    b = llr_cell_vs_ambient(S, np.arange(S.shape[1]), P, amb)
    return {"downsampled_cells_cell_like": float((a > 0).mean()), "simulated_ambient_cell_like": float((b > 0).mean()),
            "target_total_median": float(np.median(tt))}


def species(R, ambient):
    from scipy.stats import binom

    g = R["genes"]
    human = np.array([x.startswith("ENSG") for x in g])
    mouse = np.array([x.startswith("ENSMUSG") for x in g])
    if human.sum() < 1000 or mouse.sum() < 1000:
        return None
    X = R["X"]
    hu = np.asarray(X[human].sum(axis=0)).ravel()
    mo = np.asarray(X[mouse].sum(axis=0)).ravel()
    n = hu + mo
    hf = hu / np.maximum(n, 1)
    h_amb = hu[ambient].sum() / max(1, n[ambient].sum())
    pb = np.minimum(1, 2 * np.minimum(binom.cdf(hu, n, h_amb), binom.sf(hu - 1, n, h_amb)))
    unlike_amb = pb < 0.001
    pure = unlike_amb & (np.maximum(hf, 1 - hf) >= 0.9)
    return {"hf": hf, "pure": pure, "mixed_unlike_ambient": unlike_amb & ~pure, "like_ambient": ~unlike_amb, "h_amb": float(h_amb)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("run")
    ap.add_argument("--truth", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--donor", default=None, help="Singlet donor_assignments.tsv (Kang)")
    ap.add_argument("--niters", type=int, default=10000)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(7)
    os.makedirs(a.out, exist_ok=True)
    R = K.load_run(a.run)
    X, tot, called, bcs = R["X"], R["tot"], R["called"], R["barcodes"]
    lower = K.CC.plateau_lower(tot, called)
    pv, amb = K.raw_pvalues(X, called, lower, niters=a.niters)
    specs, kinfo = K.variant_specs(tot, called, lower)
    S = {"name": a.name, "run": a.run, "n_barcodes": int(len(bcs)), "n_called": int(called.sum()),
         "prod_called_ed_in_file": int(R["prod_v2"].sum()), "lower": int(lower), **kinfo,
         "alpha": amb["alpha"], "n_ambient": amb["n_ambient"], "n_tested": int(np.isfinite(pv).sum()),
         "n_tested_gt50": int((np.isfinite(pv) & (tot > max(lower, 50))).sum())}
    calls = {}
    for name, (fdr, ret, tmin, tail) in specs.items():
        cv, _ = K.finish(tot, called, amb["pv_tail"] if tail else pv, lower, fdr, ret, tmin)
        calls[name] = cv
    if lower < 100:  # the DropletUtils default ambient bound, for comparison
        pv100, amb100 = K.raw_pvalues(X, called, 100, niters=a.niters)
        calls["lower=100, bounded knee + tail p, fdr .001"], _ = K.finish(tot, called, amb100["pv_tail"], 100, 0.001,
                                                                           K.knee_bounded(tot, 100), 50)
    S["prod_reproduced"] = int(calls["prod (knee, fdr .001)"].sum())
    S["prod_reproduced_identical"] = bool((calls["prod (knee, fdr .001)"] == R["prod_v2"]).all())

    truths = {}
    for spec in a.truth:
        kind, path = spec.split(":", 1)
        T = truth_set(kind, path)
        truths[kind] = np.array([b in T for b in bcs])
        S.setdefault("truth", {})[kind] = {"n_truth": len(T), "n_truth_in_barcodes": int(truths[kind].sum()),
                                           "truth_umi": q(tot[truths[kind]])}
    main_truth = next(iter(truths)) if truths else None
    T = truths.get(main_truth)

    # ---- per-barcode evidence
    ambient_mask = (tot > 0) & (tot <= lower) & ~called
    sp_ = species(R, ambient_mask)
    cand = (tot > lower) & (np.logical_or.reduce(list(calls.values())) | (T if T is not None else False))
    ci = np.nonzero(cand)[0]
    cells_for_profiles = T if T is not None else called
    P, _ = cluster_profiles(X, cells_for_profiles)
    llr = np.full(len(bcs), np.nan)
    llr[ci] = llr_cell_vs_ambient(X, ci, P, amb["prop"])
    donor = None
    if a.donor:
        dd = pd.read_csv(a.donor, sep="\t", dtype={"cell": str}).set_index("cell")
        donor = dd.reindex(bcs)
    kang = None
    for spec in a.truth:
        if spec.startswith("kang:"):
            kang = kang_table(spec.split(":", 1)[1]).reindex(bcs)

    def classify(mask):
        idx = mask
        cat = np.full(len(bcs), "", dtype=object)
        amb_like = llr <= 0
        if sp_ is not None:
            amb_like = amb_like | sp_["like_ambient"]
        dbl = np.zeros(len(bcs), bool)
        if donor is not None:
            dbl |= (donor["donor_id"] == "doublet").to_numpy()
        if sp_ is not None:
            dbl |= sp_["mixed_unlike_ambient"]
        dmg = R["pct_mt"] >= 20
        for nm, m in (("ambient_like", amb_like), ("doublet", dbl), ("damaged", dmg)):
            cat[idx & (cat == "") & m] = nm
        cat[idx & (cat == "")] = "likely_cell"
        out = {c: int((cat[idx] == c).sum()) for c in ("likely_cell", "ambient_like", "doublet", "damaged")}
        return out, cat

    def describe(mask):
        d = {"n": int(mask.sum()), "umi": q(tot[mask]), "genes": q(R["ngenes"][mask]), "pct_mt": q(R["pct_mt"][mask]),
             "intron_frac": q(R["intron_frac"][mask]), "llr": q(llr[mask]),
             "llr_pos_frac": float((llr[mask] > 0).mean()) if mask.any() else None}
        if sp_ is not None and mask.any():
            d["species"] = {"pure": int(sp_["pure"][mask].sum()), "mixed_unlike_ambient": int(sp_["mixed_unlike_ambient"][mask].sum()),
                            "like_ambient": int(sp_["like_ambient"][mask].sum()), "hf": q(sp_["hf"][mask])}
        if donor is not None and mask.any():
            ds = donor["donor_id"][mask].fillna("NA")
            st = np.where(ds.isin(["doublet", "unassigned", "NA"]), ds, "singlet")
            d["singlet_demux"] = {k: int((st == k).sum()) for k in ("singlet", "doublet", "unassigned", "NA")}
            pm = donor["prob_max"][mask].astype(float)
            d["singlet_demux"]["singlet_prob_ge_0.9"] = int(((st == "singlet") & (pm.to_numpy() >= 0.9)).sum())
        return d

    S["variants"] = {}
    for name, cv in calls.items():
        v = {"n": int(cv.sum()), "added": int((cv & ~called).sum())}
        for kind, Tk in truths.items():
            tp = int((cv & Tk).sum())
            v[kind] = {"recall": tp / max(1, int(Tk.sum())), "precision": tp / max(1, int(cv.sum())),
                       "jaccard": tp / max(1, int((cv | Tk).sum())), "extra": int((cv & ~Tk).sum()),
                       "missed": int((~cv & Tk).sum())}
        if T is not None:
            ex = cv & ~T
            v["extra_classes"], _ = classify(ex)
            v["extra_describe"] = describe(ex)
        S["variants"][name] = v

    if T is not None:
        S["groups"] = {
            "truth_cells": describe(T),
            "truth_cells_not_called_stored": describe(T & ~called),
            "stored_called_not_truth": describe(called & ~T),
            "truth_missed_by_prod": describe(T & ~calls["prod (knee, fdr .001)"]),
        }
        lo = cand & ~called & ~T
        S["groups"]["uncalled_nontruth_tested"] = describe(lo & np.isfinite(pv))
        ex = calls["prod (knee, fdr .001)"] & ~T
        if ex.any():
            S["llr_calibration_at_extra_depth"] = calibrate_llr(X, T, P, amb["prop"], tot[ex], rng)
        if kang is not None:
            mult = kang["multiplets"].to_numpy()
            S["groups"]["kang_truth_singlet"] = describe(T & (mult == "singlet"))
            S["groups"]["kang_truth_doublet"] = describe(T & (mult == "doublet"))
            S["groups"]["kang_truth_ambs"] = describe(T & (mult == "ambs"))
            # truth cells at the extras' depth: how well does demux assign cells with that few UMIs?
            if ex.any():
                hi = np.quantile(tot[ex], 0.9)
                S["groups"]["kang_truth_at_extra_depth"] = describe(T & (tot <= hi))
            if donor is not None:
                # concordance of Singlet's donor with the demuxlet donor on truth singlets
                sing = T & (mult == "singlet") & ~donor["donor_id"].isin(["doublet", "unassigned"]).to_numpy() & donor["donor_id"].notna().to_numpy()
                ct = pd.crosstab(donor["donor_id"][sing], kang["ind"][sing])
                mp = ct.idxmax(axis=1).to_dict()
                agree = np.array([mp.get(d) == i for d, i in zip(donor["donor_id"][sing], kang["ind"][sing])])
                S["kang_donor_map"] = {str(k): int(v) for k, v in mp.items()}
                S["kang_truth_singlet_donor_agreement"] = float(agree.mean())

    # per-barcode table (candidates only), kept on Clipper
    tab = pd.DataFrame({"barcode": bcs[ci], "total": tot[ci], "n_genes": R["ngenes"][ci], "pct_mt": R["pct_mt"][ci],
                        "intron_frac": R["intron_frac"][ci], "called": called[ci], "pvalue": pv[ci],
                        "pvalue_tail": amb["pv_tail"][ci], "llr": llr[ci]})
    for kind, Tk in truths.items():
        tab[f"truth_{kind}"] = Tk[ci]
    for name, cv in calls.items():
        tab["v:" + name] = cv[ci]
    if sp_ is not None:
        tab["human_frac"] = sp_["hf"][ci]
    if donor is not None:
        tab["donor_id"] = donor["donor_id"].to_numpy()[ci]
        tab["donor_prob_max"] = donor["prob_max"].to_numpy()[ci]
    if T is not None:
        for nm in ("prod (knee, fdr .001)", "bounded knee + tail p, fdr .001", "bounded knee, fdr .01"):
            _, cat = classify(calls[nm] & ~T)
            tab["class_extra:" + nm] = cat[ci]
    tab.to_csv(os.path.join(a.out, f"{a.name}.barcodes.tsv.gz"), sep="\t", index=False)
    S["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, f"{a.name}.json"), "w") as fh:
        json.dump(S, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: S[k] for k in ("name", "n_called", "prod_called_ed_in_file", "prod_reproduced",
                                        "prod_reproduced_identical", "lower", "knee_current", "knee_bounded", "bounds")}))
    for name, v in S["variants"].items():
        print(f"  {name:28s} n={v['n']:6d} " + " ".join(
            f"{k}: R={v[k]['recall']:.3f} P={v[k]['precision']:.3f} extra={v[k]['extra']}" for k in truths) +
            (f" classes={v.get('extra_classes')}" if "extra_classes" in v else ""))


if __name__ == "__main__":
    main()
