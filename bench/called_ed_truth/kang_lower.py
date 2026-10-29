# SPDX-License-Identifier: MIT
"""Kang 2018: does a higher ambient bound (``lower``) remove the ambient-like extras that remain after the knee fix?

For each ``lower``, bounded knee + tail p-values at FDR 0.001; the extras (called, not in the demuxlet table) are scored with
the genotype check of kang_genotype.py (vireo with known donor genotypes): share of confident single-donor calls, against
synthetic ambient and depth-matched truth cells (kang_genotype.json) at the same depth.

Usage: python -P kang_lower.py RUN_DIR TRUTH_TSV GENO_DIR OUT_JSON
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402
from datasets import truth_set  # noqa: E402


def main(run, truth_tsv, geno, out):
    R = K.load_run(run)
    X, tot, called, bcs = R["X"], R["tot"], R["called"], R["barcodes"]
    T = truth_set("kang", truth_tsv)
    Tm = np.array([b in T for b in bcs])
    v = pd.read_csv(os.path.join(geno, "vireo", "donor_ids.tsv"), sep="\t")
    v["barcode"] = v["cell"].str.split("-").str[0]
    v = v.set_index("barcode")
    conf = ((~v["donor_id"].isin(["doublet", "unassigned"])) & (v["prob_max"] >= 0.9))
    G = json.load(open(os.path.join(geno, "kang_genotype.json")))
    a0, c0 = G["synthetic_ambient"]["confident_singlet_frac"], G["truth_downsampled"]["confident_singlet_frac"]
    plateau = K.CC.plateau_lower(tot, called)
    res = []
    for lower in sorted({plateau, 10, 20, 30, 50, 100}):
        pv, amb = K.raw_pvalues(X, called, lower)
        kb = K.knee_bounded(tot, lower)
        for tail in (False, True):
            cv, _ = K.finish(tot, called, amb["pv_tail"] if tail else pv, lower, 0.001, kb, 50)
            ex = cv & ~Tm
            eb = bcs[ex]
            cov = [b for b in eb if b in conf.index]
            r = {"lower": int(lower), "tail": tail, "alpha": amb["alpha"], "knee_bounded": kb, "n": int(cv.sum()),
                 "extra": int(ex.sum()), "extra_umi_median": float(np.median(tot[ex])) if ex.any() else None,
                 "extra_genotyped": len(cov),
                 "extra_conf_singlet_frac": float(conf.reindex(cov).mean()) if cov else None}
            # mixture estimate of the real-cell share among genotyped extras (rates from kang_genotype.json)
            if cov:
                r["est_real_frac"] = (r["extra_conf_singlet_frac"] - a0) / (c0 - a0)
            res.append(r)
            print(r, flush=True)
    json.dump({"plateau_lower": int(plateau), "results": res}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:5])
