# SPDX-License-Identifier: MIT
"""Re-score the called_v2 validation (docs/called-v2.md, 65 samples / 42 usable) with retain variants.

Reuses the raw Monte Carlo p-values that bench/called_v2/validate.py stored (``<GSM>.pv.npz``, key ``sum_plateau``: exon+intron,
plateau ``lower``; p-values do not depend on which barcodes are tested) and the same truth (authors' Cell Ranger calls,
hashtags, species purity). Only retain / FDR / test floor change, so every variant shares the original Monte Carlo draws.

  MODE score:     python -P revalidate.py score RESULTS/GSE --data DATA/GSE --truth-cache CACHE --out OUT/GSE
  MODE aggregate: python -P revalidate.py aggregate OUT --out REPORT

Needs bench/called_v2 and bench/empty_droplets (branch claude/festive-vaughan-4a4a8f) on ``$CV2_BENCH``.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
B = os.environ.get("CV2_BENCH", "")
sys.path.insert(0, os.path.join(B, "called_v2"))
sys.path.insert(0, os.path.join(B, "empty_droplets"))

import common as K  # noqa: E402

EXCLUDE_GSE = {"GSE145370": "TCR libraries", "GSE233574": "Parse sub-libraries (broken stored call)"}


def usable(R):
    uc = R.get("umi_called") or {}
    return R["gse"] not in EXCLUDE_GSE and R["n_called"] >= 200 and (uc.get("0.5") or 0) >= 100


def score(res_dir, data_dir, cache, out_dir):
    import truth as T
    import validate as VA
    import call_v2 as V
    from analyze import load

    os.makedirs(out_dir, exist_ok=True)
    for jf in sorted(glob.glob(os.path.join(res_dir, "GSM*.v2.json"))):
        R = json.load(open(jf))
        gse, gsm = R["gse"], R["gsm"]
        pz = np.load(os.path.join(res_dir, f"{gsm}.pv.npz"), allow_pickle=False)
        if "sum_plateau" not in pz.files:
            print(gsm, "no sum_plateau p-values")
            continue
        tot = pz["total_sum"].astype(np.int64)
        called = pz["called"].astype(bool)
        pv = pz["sum_plateau"].astype(np.float64)
        run = next((r for r in R["runs"] if r["matrix"] == "sum" and str(r["lower_rule"]) == "plateau"), None)
        lower = int(run["lower"])
        fam = V.protocol_family(R.get("protocol"))
        truth = {}
        auth, _ = T.authors(gse, gsm, os.path.join(cache, "authors"))
        bcs = pz["barcodes"].astype(str)
        if auth is not None:
            truth["author"] = np.array([b in auth for b in bcs])
        if (gse, gsm) == T.HTO_SAMPLE:
            truth["hto"] = T.hashtags(bcs, tot, os.path.join(cache, "hto"))
        d = load(os.path.join(data_dir, f"{gsm}.npz"))
        Xs = V.gene_matrix(d["exon"], d["intron"], "sum")
        assert (np.asarray(Xs.sum(axis=0)).ravel() == tot).all(), "totals differ from the stored run"
        fam_s = fam[0] if isinstance(fam, (list, tuple)) else fam
        if fam_s != "combinatorial" and ((tot > 0) & (tot <= lower) & ~called).sum() >= 10:
            pv_new, amb = K.raw_pvalues(Xs, called, lower)
            chk = np.isfinite(pv_new) & np.isfinite(pv)
            pv_check = {"n": int(chk.sum()), "identical": int((pv_new[chk] == pv[chk]).sum())}
            pv, pv_tail = pv_new, amb["pv_tail"]
        else:
            pv_check, pv_tail = None, pv
        if gse == "GSE163793":
            lower_ad = V.ambient_lower(tot, called, "adaptive")
            amb_ad = (tot > 0) & (tot <= lower_ad) & ~called
            spt = VA.species_truth(d, Xs, tot, amb_ad)
            if spt is not None:
                truth["species"] = spt
        specs, kinfo = K.variant_specs(tot, called, lower)
        out = {"gse": gse, "gsm": gsm, "usable": usable(R), "family": fam, "n_called": R["n_called"], "lower": lower,
               "pv_check": pv_check,
               "knee_recorded": run.get("knee"), **kinfo, "variants": {}}
        base = None
        for name, (fdr, ret, tmin, tail) in specs.items():
            if fam in ("combinatorial",) or (isinstance(fam, (list, tuple)) and fam[0] == "combinatorial"):
                cv = called.copy()
            else:
                cv, _ = K.finish(tot, called, pv_tail if tail else pv, lower, fdr, ret, tmin)
            m = VA.metrics(cv, called, tot, truth)
            if base is None:
                base = cv
            else:
                added0 = base & ~called
                m["kept_of_prod_added"] = int((cv & added0).sum())
                if "author" in truth:
                    m["kept_of_prod_added_author"] = int((cv & added0 & truth["author"]).sum())
                    m["prod_added_author"] = int((added0 & truth["author"]).sum())
                if "hto" in truth:
                    m["kept_of_prod_added_hto_singlet"] = int((cv & added0 & truth["hto"]["singlet"]).sum())
            m["auto_retained"] = int(((~called) & (tot > max(lower, tmin)) & (tot >= ret)).sum()) if ret is not None and np.isfinite(ret) and ret > lower else 0
            out["variants"][name] = m
        if pv_check is not None and lower < 100:
            pv100, amb100 = K.raw_pvalues(Xs, called, 100)
            cv, _ = K.finish(tot, called, amb100["pv_tail"], 100, 0.001, K.knee_bounded(tot, 100), 50)
            m = VA.metrics(cv, called, tot, truth)
            added0 = base & ~called
            m["kept_of_prod_added"] = int((cv & added0).sum())
            if "author" in truth:
                m["kept_of_prod_added_author"] = int((cv & added0 & truth["author"]).sum())
                m["prod_added_author"] = int((added0 & truth["author"]).sum())
            if "hto" in truth:
                m["kept_of_prod_added_hto_singlet"] = int((cv & added0 & truth["hto"]["singlet"]).sum())
            m["auto_retained"] = 0
            out["variants"]["lower=100, bounded knee + tail p, fdr .001"] = m
        else:
            out["variants"]["lower=100, bounded knee + tail p, fdr .001"] = out["variants"]["bounded knee + tail p, fdr .001"]
        with open(os.path.join(out_dir, f"{gsm}.rv.json"), "w") as fh:
            json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        v = out["variants"]
        print(f"{gse}/{gsm} pv_check={pv_check} lower={lower} knee={kinfo['knee_current']:.0f} (rec {run.get('knee')}) bounded={kinfo['knee_bounded']:.0f} "
              f"umin={kinfo['called_min_umi']:.0f} " + " ".join(f"[{k}: n={x['n']} auto={x['auto_retained']}]" for k, x in v.items()),
              flush=True)


def aggregate(out_root, report):
    os.makedirs(report, exist_ok=True)
    Rs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out_root, "GSE*", "GSM*.rv.json")))]
    U = [R for R in Rs if R["usable"]]
    names = list(max(U, key=lambda R: len(R["variants"]))["variants"])
    auth = [R for R in U if R["variants"][names[0]].get("author") and R["gse"] != "GSE213957"]
    S = {"n_samples": len(Rs), "n_usable": len(U), "n_author_samples": len(auth), "variants": {}}
    for nm in names:
        by = defaultdict(lambda: defaultdict(list))
        for R in auth:
            a = R["variants"][nm]["author"]
            for f in ("jaccard", "recall", "precision"):
                by[f][R["gse"]].append(a[f])
        row = {f: float(np.mean([np.mean(v) for v in by[f].values()])) for f in ("jaccard", "recall", "precision")}
        row["per_study_jaccard"] = {g: float(np.mean(v)) for g, v in by["jaccard"].items()}
        row["added_total_usable"] = int(sum(R["variants"][nm]["added"] for R in U))
        row["added_median_usable"] = float(np.median([R["variants"][nm]["added"] for R in U]))
        row["auto_retained_total_usable"] = int(sum(R["variants"][nm]["auto_retained"] for R in U))
        row["samples_with_auto_retained"] = int(sum(R["variants"][nm]["auto_retained"] > 0 for R in U))
        if nm != names[0]:
            ka = sum(R["variants"][nm].get("kept_of_prod_added_author", 0) for R in auth)
            pa = sum(R["variants"][nm].get("prod_added_author", 0) for R in auth)
            row["prod_recovered_author_cells_kept"] = [ka, pa]
        h = [R for R in U if R["variants"][nm].get("hto")]
        if h:
            row["hto"] = {k: h[0]["variants"][nm]["hto"][k] for k in ("singlet_called", "added_singlet", "added_negative", "singlet_recall_ge20")}
            row["hto"]["n"] = h[0]["variants"][nm]["n"]
            if nm != names[0]:
                row["hto"]["kept_of_prod_added_singlet"] = h[0]["variants"][nm].get("kept_of_prod_added_hto_singlet")
        spc = [R for R in U if R["variants"][nm].get("species")]
        row["species"] = {R["gsm"]: R["variants"][nm]["species"] for R in spc}
        S["variants"][nm] = row
    S["knees"] = [{k: R[k] for k in ("gse", "gsm", "usable", "lower", "knee_current", "knee_recorded", "knee_bounded", "bounds",
                                     "knee_bounded_lower100", "called_min_umi", "n_called")} | {
        "auto_retained_prod": R["variants"][names[0]]["auto_retained"], "added_prod": R["variants"][names[0]]["added"],
        "added_bounded": R["variants"]["bounded knee, fdr .001"]["added"], "added_none": R["variants"]["retain none, fdr .001"]["added"]}
        for R in Rs]
    with open(os.path.join(report, "revalidate.json"), "w") as fh:
        json.dump(S, fh, indent=1)
    for nm, r in S["variants"].items():
        print(f"{nm:28s} J={r['jaccard']:.3f} R={r['recall']:.3f} P={r['precision']:.3f} added_total={r['added_total_usable']} "
              f"auto={r['auto_retained_total_usable']} in {r['samples_with_auto_retained']} samples "
              f"kept={r.get('prod_recovered_author_cells_kept')} hto={r.get('hto')}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["score", "aggregate"])
    ap.add_argument("path")
    ap.add_argument("--data")
    ap.add_argument("--truth-cache")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "score":
        score(a.path, a.data, a.truth_cache, a.out)
    else:
        aggregate(a.path, a.out)
