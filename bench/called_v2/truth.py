# SPDX-License-Identifier: MIT
"""Ground truth for ``called_v2``, cached per sample under ``<cache>/``.

* ``authors(gse, gsm)``: the barcodes of the authors' own filtered matrix in GEO (Cell Ranger's call in every case used here),
  found by ``bench/empty_droplets/groundtruth.author_barcodes``; cached as ``<GSM>.authors.txt.gz`` (or ``.none.json``).
* ``hashtags(barcodes, total)``: GSE108313 GSM2895282 only -- per stored barcode, hashtag counts from the authors' HTO table
  (GSM2895283): ``hto_total``, ``hto_top_frac``, ``measured``; positive / singlet as in the empty-droplet study (>= 5x the median
  hashtag count of barcodes with < 10 RNA UMIs; singlet = one hashtag >= 60%).
"""

from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "empty_droplets"))
import groundtruth as G  # noqa: E402

HTO_SAMPLE = ("GSE108313", "GSM2895282")


def authors(gse: str, gsm: str, cache: str):
    """(set of barcodes, source) or (None, why)."""
    os.makedirs(cache, exist_ok=True)
    pos = os.path.join(cache, f"{gsm}.authors.txt.gz")
    neg = os.path.join(cache, f"{gsm}.authors.none.json")
    if os.path.exists(pos):
        with gzip.open(pos, "rt") as fh:
            lines = fh.read().split("\n")
        return set(l for l in lines[1:] if l), lines[0]
    if os.path.exists(neg):
        with open(neg) as fh:
            return None, json.load(fh)["why"]
    try:
        s, src = G.author_barcodes(gse, gsm)
    except Exception as e:  # network or parsing: do not cache, report
        return None, f"error: {e!r}"
    if s is None:
        with open(neg, "w") as fh:
            json.dump({"why": src}, fh)
        return None, src
    with gzip.open(pos, "wt") as fh:
        fh.write(src + "\n" + "\n".join(sorted(s)) + "\n")
    return s, src


def hashtags(barcodes: np.ndarray, total: np.ndarray, cache: str) -> dict:
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "GSM2895283_Hashtag-HTO-count.csv.gz")
    url = G.geo_dir("GSM2895283", "samples") + "GSM2895283_Hashtag-HTO-count.csv.gz"
    if not os.path.exists(path):
        data = G.get(url)
        with open(path + ".tmp", "wb") as fh:
            fh.write(data)
        os.replace(path + ".tmp", path)
    with gzip.open(path, "rt") as fh:
        lines = fh.read().splitlines()
    head = lines[0].split(",")
    rows = {}
    for l in lines[1:]:
        p = l.split(",")
        rows[p[0].strip('"')] = np.array([float(x) for x in p[1:]])
    cols = [G.norm_bc(h.strip('"')) for h in head[1:]]
    tags = [r for r in rows if not r.lower().startswith(("no_match", "ambiguous", "total", "bad_struct", "unmapped"))]
    H = np.stack([rows[t] for t in tags])
    hsum = H.sum(axis=0)
    top = H.max(axis=0) / np.maximum(hsum, 1)
    idx = {b: i for i, b in enumerate(cols) if b}
    j = np.array([idx.get(b, -1) for b in barcodes.astype(str)])
    have = j >= 0
    hs = np.where(have, hsum[np.maximum(j, 0)], np.nan)
    tf = np.where(have, top[np.maximum(j, 0)], np.nan)
    emp = have & (total < 10)
    thr = float(5 * np.median(hs[emp]))
    pos = have & (hs >= thr)
    return {"measured": have, "hto_total": hs, "hto_top_frac": tf, "positive": pos, "singlet": pos & (tf >= 0.6),
            "negative": have & (hs < thr), "threshold": thr, "source": url}
