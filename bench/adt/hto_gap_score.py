# SPDX-License-Identifier: MIT
"""Before and after scores of the hashtag caller against the best published caller, with paired bootstrap intervals.

    python -P bench/adt/hto_gap_score.py --zenodo DIR --exe hto_demux_cli --artifacts ART --selection selection.json --out out.json

ART holds the gate artifacts (hto_<ds>_{truth,singlet,published}.parquet; "singlet" is the gate-4 call, the "before"). The "after"
calls come from the C++ header through ``exe`` with the held-out setting of ``selection.json`` for each dataset, and are checked
call for call against the numpy reference of hto_lodo.py. Metrics are the gate's: F (mean over hashtags of tp / (tp + (fp + fn) / 2),
doublets and negatives called as a tag count as fp) and the share of genetic doublets called singlets. The best competitor is chosen
per metric among the published callers; the tolerance is the half-width of the 95% paired bootstrap interval (cells resampled
together, 500 draws, seed 7) of Singlet minus that caller; pass = within it on the right side.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hto_lodo as L  # noqa: E402

PUB = {"howitt2023_bal1": "BAL1", "howitt2023_bal2": "BAL2", "howitt2023_bal3": "BAL3", "howitt2023_cl": "CL"}
COMP = ["hashedDrops", "hashedDrops_default", "hashsolo", "HTODemux", "GMMDemux", "deMULTIplex", "BFF_raw", "BFF_cluster", "demuxmix"]


def metrics(calls, truth_tag, is_dbl, tags):
    f = []
    for tag in tags:
        tp = np.sum((calls == tag) & (truth_tag == tag))
        fp = np.sum((calls == tag) & (truth_tag != tag))
        fn = np.sum((calls != tag) & (truth_tag == tag))
        f.append(tp / (tp + 0.5 * (fp + fn)) if tp + fp + fn else np.nan)
    return float(np.nanmean(f)), float((is_dbl & np.isin(calls, tags)).sum() / max(is_dbl.sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    for k in ("zenodo", "exe", "artifacts", "selection", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    sel = {h["held_out"]: h for h in json.load(open(a.selection))["held_out"]}
    data = L.load_howitt(a.zenodo)
    rng = np.random.default_rng(7)
    out = {}
    for pub, ds in PUB.items():
        caps, hmap = data[ds]
        params = sel[ds]["params"]
        tr = pd.read_parquet(f"{a.artifacts}/hto_{pub}_truth.parquet").set_index("barcode")
        before = pd.read_parquet(f"{a.artifacts}/hto_{pub}_singlet.parquet").set_index("barcode")["call"]
        pb = pd.read_parquet(f"{a.artifacts}/hto_{pub}_published.parquet").set_index("barcode")
        parts, mism = [], 0
        for cap, h, _ in caps:
            src = (os.path.join(a.zenodo, "BAL_data", f"batch{ds[-1]}_{cap}_hto_counts.csv") if ds != "CL"
                   else os.path.join(a.zenodo, "cell_line_data", f"lmo_counts_capture{cap[-1]}.csv"))
            o = os.path.join(tempfile.mkdtemp(), "o.tsv")
            subprocess.run([a.exe, src, o, "--confident-min", str(params["confident_min"]), "--doublet-nmads", str(params["doublet_nmads"]),
                            "--doublet-min", str(params["doublet_min"]), "--veto", str(params["veto"] if params["veto"] is not None else 2.0)], check=True)
            c = pd.read_csv(o, sep="\t", dtype=str).set_index("barcode")["call"]
            ref = L.to_names(L.call_hashtags(h.to_numpy(float), params), list(h.index))
            mism += int((c.reindex(h.columns).to_numpy(object) != ref).sum())
            parts.append(c)
        after = pd.concat(parts)
        idx = tr.index
        inv = {v: k for k, v in hmap.items()}
        truth_tag = tr.genetic_donor.map(inv).to_numpy(object)
        is_dbl = (tr.genetic_donor == "Doublet").to_numpy()
        tags = list(hmap)
        calls = {"before": before.reindex(idx).astype(str).to_numpy(object), "after": after.reindex(idx).astype(str).to_numpy(object)}
        for c in COMP:
            if c in pb:
                calls[c] = pb[c].reindex(idx).astype(str).to_numpy(object)
        base = {k: metrics(v, truth_tag, is_dbl, tags) for k, v in calls.items()}
        draws = [rng.integers(0, len(idx), len(idx)) for _ in range(500)]
        res = {"cells": len(idx), "params": params, "cpp_vs_numpy_mismatches": mism,
               "scores": {k: {"f": round(v[0], 4), "dbl_singlet": round(v[1], 4)} for k, v in base.items()}}
        for tag_, metric_i, better in (("f", 0, 1), ("dbl_singlet", 1, -1)):
            comp = {c: base[c][metric_i] for c in COMP if c in base}
            best = max(comp, key=lambda c: better * comp[c])
            for who in ("before", "after"):
                d = []
                for ix in draws:
                    m1 = metrics(calls[who][ix], truth_tag[ix], is_dbl[ix], tags)[metric_i]
                    m2 = metrics(calls[best][ix], truth_tag[ix], is_dbl[ix], tags)[metric_i]
                    d.append(m1 - m2)
                lo, hi = np.percentile(d, [2.5, 97.5])
                tol = float((hi - lo) / 2)
                diff = base[who][metric_i] - comp[best]
                res.setdefault(who, {})[tag_] = {"best_competitor": best, "best_value": round(comp[best], 4), "singlet": round(base[who][metric_i], 4),
                                                 "diff": round(diff, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "tol": round(tol, 4),
                                                 "pass": bool(better * diff >= -tol)}
        out[pub] = res
        print(pub, json.dumps({k: res[k] for k in ("before", "after")}), "mismatch", mism, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    n = {w: sum(v[w][m]["pass"] for v in out.values() for m in ("f", "dbl_singlet")) for w in ("before", "after")}
    print("rows passed of 8:", n)


if __name__ == "__main__":
    main()
