# SPDX-License-Identifier: MIT
"""Validate ``called_ed`` variants on one fetched sample against the stored call, the authors' Cell Ranger call, hashtags
(GSE108313), species purity (barnyard references) and the empty-droplet study's profile test.

For each count matrix (``sum`` = exon + intron, ``exon``, ``max``) and ambient rule (``lower``), EmptyDrops Monte Carlo p-values
are computed once for every barcode above ``lower`` (one Polya-urn path per draw, so the p-value of a barcode does not depend on
which other barcodes are tested); the variants (test floor, FDR, retain, union with the stored call or not) are then cheap
post-processing of those p-values. Writes ``<out>/<GSE>/<GSM>.v2.json`` and the raw p-values ``<GSM>.pv.npz``.

Usage: ``python -P validate.py DATA/GSE --out RESULTS/GSE --truth-cache CACHE [--edrop-results EDROP/GSE]``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback

import numba as nb
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "empty_droplets"))
import callers as C  # noqa: E402
from analyze import gene_sets, load, q  # noqa: E402

import call_ed as V  # noqa: E402
import truth as T  # noqa: E402

MATRICES = ("sum", "exon", "max")
LOWERS = (100, "adaptive", "plateau", 20, 10)
LOWERS_SECONDARY = (100, "plateau")  # for the exon and max matrices
TEST_MIN = (0, 50, 100, 200)
FDRS = (0.001, 0.01)
RETAIN = ("knee", None)
MODES = ("union", "pure")  # union: stored call + significant uncalled; pure: EmptyDrops alone (called barcodes tested too)


def double_counting(E: sp.csc_matrix, I: sp.csc_matrix, ambient: np.ndarray, called: np.ndarray) -> dict:
    """How often one molecule sits in both the exon and the intron matrix.

    In ambient barcodes (independent molecules from one ambient pool) the expected number of (exon UMI, intron UMI) pairs on
    the same gene is E_b * I_b * sum_g pe_g * pi_g; overlaps above that are molecules counted twice. Reported as the share of
    intron UMIs that duplicate an exon UMI (pooled over the ambient barcodes), the same from the 1 exon + 1 intron barcodes
    alone, and for the called cells the upper bound sum_g min(exon_g, intron_g) / total."""
    Ea, Ia = E[:, ambient], I[:, ambient]
    pe = np.asarray(Ea.sum(axis=1)).ravel().astype(float)
    pi = np.asarray(Ia.sum(axis=1)).ravel().astype(float)
    if pe.sum() == 0 or pi.sum() == 0:
        return {"error": "empty ambient exon or intron"}
    pe /= pe.sum()
    pi /= pi.sum()
    s = float((pe * pi).sum())
    et = np.asarray(Ea.sum(axis=0)).ravel().astype(float)
    it = np.asarray(Ia.sum(axis=0)).ravel().astype(float)
    mn = np.asarray(Ea.minimum(Ia).sum(axis=0)).ravel().astype(float)
    exp = (et * it * s)
    one = (et == 1) & (it == 1)
    out = {
        "same_gene_prob_independent": s,
        "ambient_intron_dup_share": float((mn.sum() - exp.sum()) / max(1.0, it.sum())),
        "n_1e1i": int(one.sum()),
        "frac_1e1i_same_gene": float(mn[one].mean()) if one.any() else None,
    }
    if one.any():
        r = mn[one].mean()
        out["frac_1e1i_one_molecule"] = float((r - s) / (1 - s))
    Ec, Ic = E[:, called], I[:, called]
    tc = np.asarray(Ec.sum(axis=0)).ravel() + np.asarray(Ic.sum(axis=0)).ravel()
    mc = np.asarray(Ec.minimum(Ic).sum(axis=0)).ravel()
    out["called_min_share_upper_bound"] = q(mc / np.maximum(tc, 1), (0.05, 0.25, 0.5, 0.75, 0.95))
    return out


def metrics(cv2, called, tot, truth) -> dict:
    added = cv2 & ~called
    m = {"n": int(cv2.sum()), "added": int(added.sum()), "dropped": int((called & ~cv2).sum()),
         "added_umi_med": float(np.median(tot[added])) if added.any() else None,
         "jaccard_stored": float((cv2 & called).sum() / max(1, (cv2 | called).sum()))}
    a = truth.get("author")
    if a is not None:
        tp = int((cv2 & a).sum())
        m["author"] = {"tp": tp, "fp": int((cv2 & ~a).sum()), "fn": int((~cv2 & a).sum()),
                       "recall": tp / max(1, int(a.sum())), "precision": tp / max(1, int(cv2.sum())),
                       "jaccard": tp / max(1, int((cv2 | a).sum()))}
    h = truth.get("hto")
    if h is not None:
        meas = h["measured"]
        m["hto"] = {"singlet_called": int((cv2 & h["singlet"]).sum()), "positive_called": int((cv2 & h["positive"]).sum()),
                    "negative_called": int((cv2 & h["negative"]).sum()),
                    "added_singlet": int((added & h["singlet"]).sum()), "added_negative": int((added & h["negative"]).sum()),
                    "added_measured": int((added & meas).sum()),
                    "singlet_recall_ge20": float(cv2[h["singlet"] & (tot >= 20)].mean()) if (h["singlet"] & (tot >= 20)).any() else None}
    sp_ = truth.get("species")
    if sp_ is not None:
        ad = added & (tot >= 30)
        m["species"] = {"added_ge30": int(ad.sum()), "added_pure": int((ad & sp_["pure"]).sum()),
                        "added_pure_frac": float(sp_["pure"][ad].mean()) if ad.any() else None}
    pt = truth.get("profile_sig")
    if pt is not None:
        m["profile"] = {"added_sig_frac": float(pt[added].mean()) if added.any() else None}
    return m


def species_truth(d, X, tot, ambient):
    _, _, human, mouse = gene_sets(d["genes"], d["symbols"], d["chroms"])
    if human.sum() <= 1000 or mouse.sum() <= 1000:
        return None
    from scipy.stats import binom

    hu = np.asarray(X[human].sum(axis=0)).ravel()
    mo = np.asarray(X[mouse].sum(axis=0)).ravel()
    hf = hu / np.maximum(hu + mo, 1)
    h_amb = hu[ambient].sum() / max(1, (hu + mo)[ambient].sum())
    k, nn = hu.astype(np.int64), (hu + mo).astype(np.int64)
    pb = np.minimum(1, 2 * np.minimum(binom.cdf(k, nn, h_amb), binom.sf(k - 1, nn, h_amb)))
    return {"pure": (pb < 0.001) & (np.maximum(hf, 1 - hf) >= 0.9), "ambient_human_frac": float(h_amb)}


def validate_sample(path, out_dir, truth_cache, edrop_dir=None, niters=10000):
    t0 = time.time()
    d = load(path)
    meta = d["meta"]
    gse, gsm = meta["gse"], meta["gsm"]
    called = d["called"]
    E, I = d["exon"], d["intron"]
    bcs = d["barcodes"]
    n = len(bcs)
    Xs = V.gene_matrix(E, I, "sum")
    tot_sum = np.asarray(Xs.sum(axis=0)).ravel().astype(np.int64)
    protocol = (meta.get("gsm_meta") or {}).get("protocol") or (meta.get("summary") or {}).get("protocol_id")
    R = {"gse": gse, "gsm": gsm, "protocol": protocol, "family": V.protocol_family(protocol), "n_barcodes": n,
         "n_called": int(called.sum()), "umi_called": q(tot_sum[called]) if called.any() else None}

    # ---- truth
    truth = {}
    auth, src = T.authors(gse, gsm, os.path.join(truth_cache, "authors"))
    R["author_source"] = src
    if auth is not None:
        a = np.array([b in auth for b in bcs.astype(str)])
        truth["author"] = a
        R["author"] = {"n_author": len(auth), "n_author_stored": int(a.sum())}
        R["stored_call"] = metrics(called, called, tot_sum, {"author": a})["author"]
    if (gse, gsm) == T.HTO_SAMPLE:
        h = T.hashtags(bcs, tot_sum, os.path.join(truth_cache, "hto"))
        truth["hto"] = h
        R["hto"] = {"threshold": h["threshold"], "n_measured": int(h["measured"].sum()), "n_singlet": int(h["singlet"].sum()),
                    "n_singlet_ge20": int((h["singlet"] & (tot_sum >= 20)).sum()),
                    "stored_call": metrics(called, called, tot_sum, {"hto": h})["hto"]}
    lower_ad = V.ambient_lower(tot_sum, called, "adaptive")
    amb_ad = (tot_sum > 0) & (tot_sum <= lower_ad) & ~called
    spt = species_truth(d, Xs, tot_sum, amb_ad)
    if spt is not None:
        truth["species"] = spt
        R["species_ambient_human_frac"] = spt["ambient_human_frac"]
    if edrop_dir:
        bcp = os.path.join(edrop_dir, f"{gsm}.bc.npz")
        if os.path.exists(bcp):
            B = np.load(bcp)
            if "fdr_cell" in B.files and len(B["fdr_cell"]) == n:
                truth["profile_sig"] = (~called) & (np.nan_to_num(B["fdr_cell"], nan=1.0) < 0.01)

    # ---- double counting (ambient pool of the adaptive rule)
    R["double_counting"] = double_counting(E, I, amb_ad, called)

    # ---- comparators
    comp = {"stored": metrics(called, called, tot_sum, truth)}
    om, thr = C.ordmag(tot_sum)
    comp["ordmag_stored_counts"] = {**metrics(om, called, tot_sum, truth), "threshold": thr}
    try:
        cr = C.cr3_like(Xs, niters=niters, seed=11)
        comp["cr3_like"] = {**metrics(cr["called"], called, tot_sum, truth), "ambient_rule": cr["ambient_rule"]}
        comp["cr3_like_union"] = metrics(cr["called"] | called, called, tot_sum, truth)
    except Exception as e:
        comp["cr3_like"] = {"error": repr(e)}
    R["comparators"] = comp

    # ---- EmptyDrops runs and variants
    runs, pv_store = [], {}
    for mname in MATRICES:
        X = Xs if mname == "sum" else V.gene_matrix(E, I, mname)
        tot = tot_sum if mname == "sum" else np.asarray(X.sum(axis=0)).ravel().astype(np.int64)
        for lrule in (LOWERS if mname == "sum" else LOWERS_SECONDARY):
            lower = V.ambient_lower(tot, called, lrule)
            amb = (tot > 0) & (tot <= lower) & ~called
            test = tot > lower
            t1 = time.time()
            ed = C.emptydrops(X, ambient_mask=amb, test_mask=test, niters=niters, seed=V.DEFAULTS["seed"])
            run = {"matrix": mname, "lower_rule": lrule, "lower": lower, "n_ambient": int(amb.sum()),
                   "ambient_umi": int(tot[amb].sum()), "n_tested_max": int(test.sum())}
            if "error" in ed:
                run["error"] = ed["error"]
                runs.append(run)
                continue
            knee, _ = C.barcode_ranks(tot, lower=lower)
            run.update({"alpha": float(ed["alpha"]), "knee": float(knee), "seconds": round(time.time() - t1, 1)})
            pv = ed["pvalue"]
            pv_store[f"{mname}_{lrule}"] = pv.astype(np.float32)
            variants = []
            for mode in MODES:
                tmask = test & (~called if mode == "union" else np.ones(n, bool))
                for tmin in TEST_MIN:
                    for ret in RETAIN:
                        rt = float(knee) if ret == "knee" and np.isfinite(knee) and knee > lower else None
                        for fdr in FDRS:
                            cv2, _ = V.finish(tot, called, pv, tmask, fdr, rt, mode == "union", tmin)
                            variants.append({"mode": mode, "test_min": tmin, "retain": ret, "fdr": fdr,
                                             **metrics(cv2, called, tot_sum, truth)})
            run["variants"] = variants
            runs.append(run)
            print(f"  {gsm} {mname} lower={lrule}->{lower} alpha={run['alpha']:.0f} t={run['seconds']}s", flush=True)
    R["runs"] = runs

    # ---- the prototype's defaults, end to end
    res = V.call_ed(E, I, called, protocol)
    R["prototype"] = {"params": res["params"], **metrics(res["called_ed"], called, tot_sum, truth)}
    R["seconds"] = round(time.time() - t0, 1)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{gsm}.v2.json"), "w") as fh:
        json.dump(R, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    np.savez_compressed(os.path.join(out_dir, f"{gsm}.pv.npz"), barcodes=bcs, total_sum=tot_sum, called=called,
                        called_ed=res["called_ed"], fdr_v2=res["fdr"].astype(np.float32), **pv_store)
    return R


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--truth-cache", required=True)
    ap.add_argument("--edrop-results", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--niters", type=int, default=10000)
    ap.add_argument("--gsm", nargs="*")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args(argv)
    nb.set_num_threads(max(1, a.threads))
    files = sorted(glob.glob(os.path.join(a.data_dir, "GSM*.npz")))
    if a.gsm:
        files = [f for f in files if os.path.basename(f)[:-4] in set(a.gsm)]
    for f in files:
        g = os.path.basename(f)[:-4]
        if not a.redo and os.path.exists(os.path.join(a.out, f"{g}.v2.json")):
            print(g, "done already")
            continue
        try:
            R = validate_sample(f, a.out, a.truth_cache, a.edrop_results, a.niters)
            p = R["prototype"]
            print(f"{g} ok called={R['n_called']} v2={p['n']} (+{p['added']} -{p['dropped']}) "
                  f"author={json.dumps(p.get('author'))} t={R['seconds']}s", flush=True)
        except Exception:
            print(g, "FAILED", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
