# SPDX-License-Identifier: MIT
"""Hashtag demultiplexing on the public benchmark of Howitt et al. 2023 (NAR Genom Bioinform 5:lqad086; Zenodo 8304003).

    python -P bench/adt/hto_zenodo.py --zenodo $WORK/data/adt_bench/zenodo_8304003 --published PUBDIR --out OUT

Data: BAL (24 paediatric donors, 3 batches x 8 TotalSeq-A hashtags, 2 captures each) and a lung cancer cell-line mix (3 lines,
MULTI-seq lipid tags, 3 captures), with vireo genotype calls per cell (``genetic_donor``: donor / ``Doublet`` / ``Negative``)
as the truth. The paper's protocol is followed exactly (``analysis/BAL_analysis.Rmd`` in Oshlack/hashtag-demux-paper):

* each method runs on each capture separately, on the called cells only (no empty droplets are provided);
* donor *i* of a batch is hashtag *i* (``BAL A`` = ``BAL 01``, ``BAL I`` = ``BAL 09``, ``BAL Q`` = ``BAL 17``; cell lines
  ``CL 01..03`` in donor order);
* F-score per hashtag over all cells of the batch, ``tp / (tp + (fp + fn) / 2)`` with ``fp`` including cells whose truth is a
  doublet or negative, averaged over the hashtags (their ``calculate_HTO_fscore``);
* the second metric: the share of genetic doublets that a method calls a singlet.

``--published`` is the CSV export of the paper's own calls (``export_rds_calls.R``); re-scoring them with this code must give
the paper's Table 1, which validates the metric before Singlet is compared.

Writes ``fscores.tsv`` (dataset x method: mean F, per-hashtag F, doublets called singlet, seconds), ``calls/`` and
``summary.json``.
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hto_methods as HM  # noqa: E402

PAPER_TABLE1 = {  # mean F-score, Table 1 of Howitt et al. 2023 (BAL batches 1-3, cell lines)
    "hashedDrops (optimised)": {"BAL1": 0.930, "BAL2": 0.847, "BAL3": 0.785, "CL": 0.864},
    "hashedDrops (default)": {"BAL1": 0.907, "BAL2": 0.613, "BAL3": 0.573, "CL": 0.663},
    "demuxmix": {"BAL1": 0.907, "BAL2": 0.833, "BAL3": 0.775, "CL": 0.743},
    "HTODemux": {"BAL1": 0.892, "BAL2": 0.794, "BAL3": 0.742, "CL": 0.737},
    "GMM-Demux": {"BAL1": 0.905, "BAL2": 0.770, "BAL3": 0.720, "CL": 0.823},
    "HashSolo": {"BAL1": 0.920, "BAL2": 0.783, "BAL3": 0.737, "CL": 0.806},
    "deMULTIplex": {"BAL1": 0.907, "BAL2": 0.791, "BAL3": 0.624, "CL": 0.828},
    "BFFraw": {"BAL1": 0.879, "BAL2": 0.539, "BAL3": 0.466, "CL": 0.791},
    "BFFcluster": {"BAL1": 0.935, "BAL2": 0.119, "BAL3": 0.188, "CL": 0.874},
}


def read_counts(path):
    d = pd.read_csv(path, index_col=0)
    d.index = [str(i) for i in d.index]
    return d


def datasets(z):
    """-> [(dataset, [(capture, hto counts DataFrame, truth Series)], hashtag->donor map)]"""
    out = []
    for b in (1, 2, 3):
        caps = []
        for c in (1, 2):
            h = read_counts(os.path.join(z, "BAL_data", f"batch{b}_c{c}_hto_counts.csv"))
            dn = pd.read_csv(os.path.join(z, "BAL_data", f"batch{b}_c{c}_donors.csv"))
            truth = dn.set_index("Barcode")["genetic_donor"].reindex(h.columns)
            caps.append((f"c{c}", h, truth))
        tags = list(caps[0][1].index)
        first = ord("A") + 8 * (b - 1)
        hmap = {t: f"BAL {chr(first + i)}" for i, t in enumerate(tags)}
        out.append((f"BAL{b}", caps, hmap))
    cl = []
    for c in (1, 2, 3):
        f = os.path.join(z, "cell_line_data", f"lmo_counts_capture{c}.csv")
        dfile = os.path.join(z, "cell_line_data", f"lmo_donors_capture{c}.csv")
        if not (os.path.exists(f) and os.path.exists(dfile)):
            continue
        h = read_counts(f)
        dn = pd.read_csv(dfile)
        bc = "Barcode" if "Barcode" in dn else dn.columns[0]
        dcol = "genetic_donor" if "genetic_donor" in dn else dn.columns[-1]
        cl.append((f"c{c}", h, dn.set_index(bc)[dcol].reindex(h.columns)))
    if cl:
        donors = sorted({d for _, _, t in cl for d in t.dropna().unique() if d not in ("Doublet", "Negative")})
        tags = list(cl[0][1].index)
        out.append(("CL", cl, {t: d for t, d in zip(tags, donors)}))
    return out


def fscore(calls: pd.Series, truth: pd.Series, hmap: dict):
    """Howitt et al. ``calculate_HTO_fscore``: per hashtag, truth = donor mapped to the hashtag."""
    truth_tag = truth.map({v: k for k, v in hmap.items()})  # donor -> hashtag (Doublet/Negative -> NaN)
    per = {}
    for tag in hmap:
        tp = int(((calls == tag) & (truth_tag == tag)).sum())
        fp = int(((calls == tag) & (truth_tag != tag)).sum())
        fn = int(((calls != tag) & (truth_tag == tag)).sum())
        per[tag] = tp / (tp + 0.5 * (fp + fn)) if tp + fp + fn else float("nan")
    dbl = truth.eq("Doublet")
    sing_call = calls.isin(list(hmap))
    return {"f_mean": float(np.nanmean(list(per.values()))), "f_per_hashtag": per,
            "doublets_called_singlet": float((dbl & sing_call).sum() / max(int(dbl.sum()), 1)),
            "genetic_doublets": int(dbl.sum()), "cells": int(len(calls))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo", required=True)
    ap.add_argument("--published", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="singlet,singlet_cm05,htodemux,multiseqdemux,hashsolo")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "calls"), exist_ok=True)
    rows = []
    for name, caps, hmap in datasets(a.zenodo):
        truth = pd.concat([t for _, _, t in caps])
        for m in a.methods.split(","):
            fn = HM.METHODS.get(m.replace("_cm05", ""))
            parts, secs = [], 0.0
            try:
                for cap, h, _ in caps:
                    wd = os.path.join(a.out, "work", name, cap, m)
                    os.makedirs(wd, exist_ok=True)
                    kw = {"confident_min": 0.5} if m == "singlet_cm05" else {}
                    res = fn(h, None, workdir=wd, **kw)
                    parts.append(res["call"])
                    secs += float(res.attrs.get("seconds", np.nan))
                calls = pd.concat(parts)
                calls.to_csv(os.path.join(a.out, "calls", f"{name}_{m}.csv"))
                r = fscore(calls, truth.reindex(calls.index), hmap)
                rows.append({"dataset": name, "method": m, "source": "this run", "seconds": secs, **r})
            except Exception as e:
                rows.append({"dataset": name, "method": m, "source": "this run", "error": f"{type(e).__name__}: {e}"[:300]})
            print(name, m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()
                            if k != "f_per_hashtag"}, flush=True)
        # the paper's own calls, re-scored with this code (metric validation)
        if a.published:
            pub = os.path.join(a.published, ("lmo" if name == "CL" else f"batch{name[-1]}") + "_published_calls.csv")
            if os.path.exists(pub):
                P = pd.read_csv(pub)
                P["key"] = P["cell"].astype(str)
                P = P.set_index("key")
                for col in [c for c in P.columns if c.endswith("_calls")]:
                    calls = P[col].astype(str)
                    tr = P["genetic_donor"].astype(str) if "genetic_donor" in P else truth.reindex(P.index)
                    r = fscore(calls, tr, hmap)
                    rows.append({"dataset": name, "method": col.replace("_calls", ""), "source": "published calls", **r})
                    print(name, "published", col, round(r["f_mean"], 4), flush=True)
    df = pd.DataFrame(rows)
    df.assign(f_per_hashtag=df.get("f_per_hashtag", pd.Series(dtype=object)).map(json.dumps)).to_csv(
        os.path.join(a.out, "fscores.tsv"), sep="\t", index=False)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"paper_table1": PAPER_TABLE1,
                   "results": df.drop(columns=["f_per_hashtag"], errors="ignore").to_dict("records")}, f, indent=1,
                  default=str)


if __name__ == "__main__":
    sys.exit(main())
