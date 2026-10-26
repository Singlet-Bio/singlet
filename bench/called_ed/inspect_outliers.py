# SPDX-License-Identifier: MIT
"""Follow-up on the outliers of the validation run: who are the barcodes ``called_ed`` adds where agreement is low?

For each named sample: added barcodes split by UMI bin, with authors' membership, profile-test significance (empty-droplet
study), intronic and mitochondrial fractions and, in mixed-species samples, the human fraction. Hashtag-negative additions in
GSE108313 are compared with the hashtag singlets added. Prints JSON lines.

Usage: ``python -P inspect_outliers.py --data DATA --results RESULTS --edrop EDROP --truth-cache CACHE``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "empty_droplets"))
from analyze import gene_sets, load  # noqa: E402

import truth as T  # noqa: E402

SAMPLES = [("GSE157827", "GSM4775563"), ("GSE157827", "GSM4775562"), ("GSE179165", "GSM5410927"), ("GSE230559", "GSM7226734"),
           ("GSE163793", "GSM4987146"), ("GSE163793", "GSM4987148"), ("GSE163793", "GSM4987142"), ("GSE163793", "GSM4987478"),
           ("GSE266546", "GSM8249613"), ("GSE266546", "GSM8249621"), ("GSE108313", "GSM2895282"), ("GSE246624", "GSM7873657")]
BINS = [0, 50, 100, 200, 300, 500, 1000, 10**9]


def med(v):
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--edrop", required=True)
    ap.add_argument("--truth-cache", required=True)
    a = ap.parse_args()
    for gse, gsm in SAMPLES:
        d = load(os.path.join(a.data, gse, f"{gsm}.npz"))
        P = np.load(os.path.join(a.results, gse, f"{gsm}.pv.npz"))
        called, cv2, tot = P["called"], P["called_ed"], P["total_sum"]
        E, I = d["exon"], d["intron"]
        X = (E + I).tocsc()
        itot = np.asarray(I.sum(axis=0)).ravel()
        mt, _, human, mouse = gene_sets(d["genes"], d["symbols"], d["chroms"])
        mtf = np.asarray(X[mt].sum(axis=0)).ravel() / np.maximum(tot, 1)
        intr = itot / np.maximum(tot, 1)
        hf = None
        if human.sum() > 1000 and mouse.sum() > 1000:
            hu = np.asarray(X[human].sum(axis=0)).ravel()
            mo = np.asarray(X[mouse].sum(axis=0)).ravel()
            hf = hu / np.maximum(hu + mo, 1)
        B = np.load(os.path.join(a.edrop, gse, f"{gsm}.bc.npz"))
        sig = np.nan_to_num(B["fdr_cell"], nan=1.0) < 0.01 if "fdr_cell" in B.files else np.zeros(len(tot), bool)
        auth, _ = T.authors(gse, gsm, os.path.join(a.truth_cache, "authors"))
        au = np.array([b in auth for b in d["barcodes"]]) if auth is not None else None
        added = cv2 & ~called
        amb = (tot > 0) & (tot <= 10) & ~called
        out = {"sample": f"{gse}/{gsm}", "called": int(called.sum()), "added": int(added.sum()),
               "called_intronic_med": med(intr[called]), "ambient_intronic_med": med(intr[amb]),
               "called_mt_med": med(mtf[called]), "ambient_mt_med": med(mtf[amb]), "bins": []}
        if hf is not None:
            out["ambient_human_frac"] = float(np.average(hf[amb], weights=tot[amb]))
        if au is not None:
            fn = au & ~cv2
            out["author_not_v2"] = int(fn.sum())
            out["author_not_v2_umi_q"] = [float(x) for x in np.quantile(tot[fn], [0.1, 0.5, 0.9])] if fn.any() else None
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = added & (tot >= lo) & (tot < hi)
            if not m.any():
                continue
            row = {"lo": lo, "n": int(m.sum()), "profile_sig": float(sig[m].mean()), "intronic_med": med(intr[m]),
                   "mt_med": med(mtf[m])}
            if au is not None:
                row["author_frac"] = float(au[m].mean())
            if hf is not None:
                h = hf[m]
                row["human_frac_hist"] = np.histogram(h, bins=[0, 0.1, 0.3, 0.7, 0.9, 1.0001])[0].tolist()
            out["bins"].append(row)
        if (gse, gsm) == T.HTO_SAMPLE:
            h = T.hashtags(d["barcodes"], tot, os.path.join(a.truth_cache, "hto"))
            for nm, m in (("added_singlet", added & h["singlet"]), ("added_negative", added & h["negative"]),
                          ("called_negative", called & h["negative"]), ("called_singlet", called & h["singlet"])):
                out[nm] = {"n": int(m.sum()), "umi_med": med(tot[m].astype(float)), "intronic_med": med(intr[m]),
                           "mt_med": med(mtf[m]), "profile_sig": float(sig[m].mean()) if m.any() else None,
                           "hto_total_med": med(h["hto_total"][m])}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
