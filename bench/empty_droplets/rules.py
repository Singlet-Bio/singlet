# SPDX-License-Identifier: MIT
"""Candidate keep rules for v2: what each one saves and what it loses.

For every analysed sample (fetch.py npz + analyze.py bc.npz):

* storage: matrix entries (feature-level non-zeros of exon, intron, junction and V(D)J members) and estimated bytes kept under
  each rule (bytes = member bytes x share of that member's non-zeros kept: an approximation, the codec's cost per entry is not
  uniform),
* biology: profile-test cell-like uncalled barcodes, barnyard species-pure barcodes and EmptyDrops (adaptive) calls that a rule
  drops,
* re-calling: EmptyDrops run on the full stored matrix vs on the reduced one, where the dropped barcodes survive only as a summed
  ambient profile (their per-barcode totals stay in the cells table). The reduced run uses the full ambient sum (exactly
  recoverable from the stored aggregate) and estimates alpha from the ambient barcodes that remain.

Usage: ``python -P rules.py DATA RESULTS --out rules.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import callers as C  # noqa: E402
from analyze import load  # noqa: E402


def rules_for(tot, called, q05):
    floor_rel = lambda k: max(1.0, k * q05)
    return {
        "drop_all_uncalled": called.copy(),
        "keep_umi>=10": called | (tot >= 10),
        "keep_umi>=20": called | (tot >= 20),
        "keep_umi>=50": called | (tot >= 50),
        "keep_umi>=100": called | (tot >= 100),
        "keep_umi>=min(20,0.25q05)": called | (tot >= min(20.0, floor_rel(0.25))),
        "keep_umi>=min(10,0.1q05)": called | (tot >= min(10.0, floor_rel(0.1))),
    }


def ed_full_vs_reduced(X, tot, keep, lower, niters, seed=5):
    """EmptyDrops calls (FDR 0.001) on the full matrix vs the reduced one (dropped barcodes only as a summed profile)."""
    knee = C.barcode_ranks(tot, lower=lower)[0]
    retain = knee if np.isfinite(knee) and knee > lower else None
    full = C.emptydrops(X, lower=lower, niters=niters, retain=retain, seed=seed)
    if "error" in full:
        return {"error": full["error"]}
    amb_all = (tot <= lower) & (tot > 0)
    amb_kept = amb_all & keep
    # reduced: same ambient profile (sum over all ambient barcodes, recoverable from the stored aggregate), alpha from kept ones
    test = (tot > lower) & keep
    Xs = X.tocsc()
    asum = np.asarray(Xs[:, amb_all].sum(axis=1)).ravel().astype(np.int64)
    prop = C.good_turing_proportions(asum)
    alpha_red = C.dm_alpha_mle(Xs[:, amb_kept], prop) if amb_kept.sum() >= 50 else full["alpha"]
    ti = np.nonzero(test)[0]
    lp = C.dm_logprob(Xs[:, ti], prop, alpha_red)
    tt = tot[ti]
    ut = np.unique(tt[tt <= 50000])
    pv = np.full(len(ti), 1.0 / (niters + 1))
    if len(ut):
        prob, alias = C.alias_table(prop)
        sims = C._mc_paths(prob, alias, prop, float(alpha_red), ut.astype(np.int64), int(niters), int(seed))
        sims.sort(axis=0)
        m = tt <= 50000
        col = np.searchsorted(ut, tt[m])
        n_le = np.array([np.searchsorted(sims[:, c], o, side="right") for c, o in zip(col, lp[m])])
        pv[m] = (n_le + 1) / (niters + 1)
    if retain is not None:
        pv[tt >= retain] = 0.0
    called_red = np.zeros(len(tot), bool)
    called_red[ti] = C.bh(pv) <= 0.001
    called_full = np.nan_to_num(full["fdr"], nan=1.0) <= 0.001
    return {"lower": lower, "alpha_full": full["alpha"], "alpha_reduced": alpha_red, "n_full": int(called_full.sum()),
            "n_reduced": int(called_red.sum()), "both": int((called_full & called_red).sum()),
            "full_only": int((called_full & ~called_red).sum()), "reduced_only": int((called_red & ~called_full).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--niters", type=int, default=3000)
    a = ap.parse_args()
    out = {}
    for bcf in sorted(glob.glob(os.path.join(a.results, "GSE*", "GSM*.bc.npz"))):
        gse = os.path.basename(os.path.dirname(bcf))
        gsm = os.path.basename(bcf)[: -len(".bc.npz")]
        dpath = os.path.join(a.data, gse, f"{gsm}.npz")
        if not os.path.exists(dpath):
            continue
        d = load(dpath)
        z = np.load(bcf)
        called = z["called"].astype(bool)
        if called.sum() < 200:
            continue
        X = (d["exon"] + d["intron"]).tocsc()
        tot = np.asarray(X.sum(axis=0)).ravel().astype(np.int64)
        if np.median(tot[called]) < 100:
            continue
        unc = ~called
        q05 = float(np.quantile(tot[called], 0.05))
        sig = unc & (np.nan_to_num(z["fdr_cell"], nan=1) < 0.01) if "fdr_cell" in z.files else np.zeros(len(tot), bool)
        pure = None
        if "species_p" in z.files:
            hf = z["human_frac"]
            if np.quantile(hf[called], 0.05) < 0.2 and np.quantile(hf[called], 0.95) > 0.8:  # a real two-species mixture
                pure = unc & (z["species_p"] < 0.001) & (np.maximum(hf, 1 - hf) >= 0.9)
        edc = None
        for k in ("emptydrops_adaptive_called", "emptydrops_lower100_called"):
            if k in z.files:
                edc = unc & z[k].astype(bool)
                break
        meta = d["meta"]
        nnz = {k: d[f"{k}_nnz_features"] for k in ("exon", "intron")}
        for k in ("sj", "vdj"):
            if d[f"{k}_nnz"] is not None:
                nnz[k] = d[f"{k}_nnz"]
        mbytes = {k: meta.get(f"{k}_bytes", 0) or 0 for k in nnz}
        R = {"q05_called": q05, "n_barcodes": int(len(tot)), "member_bytes": mbytes, "rules": {}}
        for name, keep in rules_for(tot, called, q05).items():
            e = {"barcodes_kept": int(keep.sum())}
            kept_bytes = 0.0
            for k, v in nnz.items():
                share = float(v[keep].sum() / max(1, v.sum()))
                e[f"entries_kept_{k}"] = share
                kept_bytes += share * mbytes[k]
            e["bytes_kept_est"] = kept_bytes
            e["bytes_total"] = float(sum(mbytes.values()))
            e["cell_like_lost"] = int((sig & ~keep).sum())
            e["cell_like_total"] = int(sig.sum())
            if pure is not None:
                e["species_pure_lost"] = int((pure & ~keep).sum())
                e["species_pure_total"] = int(pure.sum())
            if edc is not None:
                e["emptydrops_extra_lost"] = int((edc & ~keep).sum())
                e["emptydrops_extra_total"] = int(edc.sum())
            R["rules"][name] = e
        keep = rules_for(tot, called, q05)["keep_umi>=20"]
        try:
            R["recall_emptydrops_keep20"] = ed_full_vs_reduced(X, tot, keep, int(np.clip(0.5 * q05, 25, 100)), a.niters)
        except Exception as ex:
            R["recall_emptydrops_keep20"] = {"error": repr(ex)}
        out[f"{gse}/{gsm}"] = R
        print(gse, gsm, json.dumps(R["recall_emptydrops_keep20"]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
