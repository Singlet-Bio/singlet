# SPDX-License-Identifier: MIT
"""Independent ground truth from the authors' own GEO deposits, compared with Singlet's call and the profile test.

* Author cell calls: the barcodes of the authors' filtered matrices (Cell Ranger ``filtered_feature_bc_matrix`` or equivalent)
  found in the sample's GEO supplementary files (or the series files for GSE256089), or the authors' cell metadata (GSE213957).
* Cell hashing (GSE108313 PBMC): hashtag (HTO) read counts of 65,000 barcodes, GSM2895283, for the RNA library GSM2895282.
  A droplet holding a stained cell carries one dominant hashtag at high count; an empty droplet only free antibody.

Usage: ``python -P groundtruth.py --data DATA --results RESULTS GSE...`` (runs inside a Clipper job; GEO is public).
Writes ``<RESULTS>/<GSE>/<GSM>.author.json``.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import bin_table, q  # noqa: E402

UA = "singlet-emptydrop-study/1.0 (+https://singlet.bio)"
GEO = "https://ftp.ncbi.nlm.nih.gov/geo"
BC_RE = re.compile(r"[ACGTN]{12,16}")


def get(url, retries=5):
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=300) as r:
                return r.read()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(3 * (i + 1))


def listing(url):
    try:
        html = get(url).decode("utf-8", "replace")
    except Exception:
        return []
    return sorted(set(re.findall(r'href="([^"?/][^"]*)"', html)))


def geo_dir(acc, kind):
    stub = acc[:-3] + "nnn"
    return f"{GEO}/{kind}/{stub}/{acc}/suppl/"


def norm_bc(s):
    m = BC_RE.search(s)
    return m.group(0) if m else None


def barcodes_from_bytes(name, data):
    """Barcodes from a barcodes.tsv(.gz), or from the barcodes member of a tar(.gz) of a filtered matrix."""
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        out = []
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            for m in tf.getmembers():
                b = os.path.basename(m.name)
                if b.startswith("barcodes.tsv") and ("filtered" in m.name or "raw" not in m.name):
                    raw = tf.extractfile(m).read()
                    if b.endswith(".gz"):
                        raw = gzip.decompress(raw)
                    out = raw.decode().split()
                    break
        return out
    raw = gzip.decompress(data) if name.endswith(".gz") else data
    return raw.decode().split()


def author_barcodes(gse, gsm):
    """(set of normalised barcodes, source description) or (None, why)."""
    if gse == "GSE213957":
        if gsm != "GSM6596913":
            return None, "author metadata only for the species-mixing GEX library GSM6596913"
        url = geo_dir(gse, "series") + "GSE213957_SpeciesMixing_CaRPool-seq.metadata.tsv.gz"
        rows = gzip.decompress(get(url)).decode().splitlines()
        bcs = {norm_bc(r.split("\t")[0]) for r in rows[1:]} - {None}
        return bcs, url
    if gse == "GSE145370":
        # the atlas holds the TCR libraries (GSM4317423..36, "S1xx_TCR Sample k"); 10x 5' GEX and V(D)J libraries of one GEM
        # well share cell barcodes, so the authors' GEX call (GSM4317409..22, "S1xx_mRNA Sample k") is the cell set
        gex = f"GSM{int(gsm[3:]) - 14}"
        for f in listing(geo_dir(gex, "samples")):
            if "filtered_feature_bc_matrix" in f:
                url = geo_dir(gex, "samples") + f
                return {norm_bc(b) for b in barcodes_from_bytes(f, get(url))} - {None}, url + " (GEX library of the same GEM well)"
        return None, f"no filtered matrix for {gex}"
    if gse == "GSE256089":
        # GEX GSMs GSM8085472, 74, ... pair with lane files BatchA_Lane1, Lane3, ... (HTO GSMs are the odd ones)
        k = int(gsm[3:]) - 8085472
        if k % 2:
            return None, "HTO library GSM (its RNA call is meaningless)"
        lanes = [f for f in listing(geo_dir(gse, "series")) if f.endswith("_barcodes.tsv.gz")]
        lanes = sorted(lanes, key=lambda f: (f.split("_")[1], int(re.search(r"Lane(\d+)", f).group(1))))
        if k // 2 >= len(lanes):
            return None, "no lane file for this GSM"
        url = geo_dir(gse, "series") + lanes[k // 2]
        return {norm_bc(b) for b in barcodes_from_bytes(url, get(url))} - {None}, url + " (pairing by GSM order, check)"
    files = listing(geo_dir(gsm, "samples"))
    cand = [f for f in files if re.search(r"barcodes\.tsv(\.gz)?$", f) and "raw" not in f.lower()]
    tars = [f for f in files if re.search(r"\.(tar\.gz|tgz|tar)$", f, re.I)]
    cand += sorted(tars, key=lambda f: "filtered" not in f.lower())  # filtered-named tars first; any tar is opened and searched
    for f in cand:
        url = geo_dir(gsm, "samples") + f
        try:
            bcs = barcodes_from_bytes(f, get(url))
        except Exception:
            continue
        s = {norm_bc(b) for b in bcs} - {None}
        if s:
            if len(s) > 200000:
                return None, f"{f}: {len(s)} barcodes (a raw list, not a call)"
            return s, url
    return None, f"no filtered barcodes among {len(files)} supplementary files"


def compare(gse, gsm, data, results):
    z = np.load(os.path.join(data, gse, f"{gsm}.npz"), allow_pickle=False)
    bcs = z["barcodes"].astype(str)
    called = z["called"].astype(bool)
    an = os.path.join(results, gse, f"{gsm}.bc.npz")
    A = np.load(an) if os.path.exists(an) else None
    tot = A["total"] if A is not None else None
    out = {"gse": gse, "gsm": gsm}
    auth, src = author_barcodes(gse, gsm)
    out["author_source"] = src
    if auth is not None:
        a = np.array([b in auth for b in bcs])
        unc = ~called
        out["author"] = {
            "n_author": len(auth), "n_author_in_stored": int(a.sum()), "n_author_not_stored": int(len(auth) - a.sum()),
            "n_ours": int(called.sum()), "both": int((a & called).sum()), "author_not_ours": int((a & unc).sum()),
            "ours_not_author": int((called & ~a).sum()),
            "jaccard": float((a & called).sum() / max(1, (a | called).sum())),
        }
        if A is not None:
            sig = unc & (np.nan_to_num(A["fdr_cell"], nan=1) < 0.01) if "fdr_cell" in A.files else np.zeros(len(bcs), bool)
            out["author"].update({
                "author_not_ours_umi": q(tot[a & unc]), "ours_not_author_umi": q(tot[called & ~a]),
                "author_not_ours_profile_sig": int((a & sig).sum()),
                "profile_sig_that_author_called": float(a[sig].mean()) if sig.any() else None,
                "uncalled_by_bin": bin_table({"frac_author_called": (a[unc].astype(float), "mean"),
                                              "frac_profile_sig": (sig[unc].astype(float), "mean")}, tot[unc]),
                "called_by_bin": bin_table({"frac_author_called": (a[called].astype(float), "mean")}, tot[called]),
            })
    if gse == "GSE108313" and gsm == "GSM2895282":
        out["hashing"] = hashing(bcs, called, A)
    os.makedirs(os.path.join(results, gse), exist_ok=True)
    with open(os.path.join(results, gse, f"{gsm}.author.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def hashing(bcs, called, A):
    url = geo_dir("GSM2895283", "samples") + "GSM2895283_Hashtag-HTO-count.csv.gz"
    lines = gzip.decompress(get(url)).decode().splitlines()
    head = lines[0].split(",")
    rows = {}
    for l in lines[1:]:
        p = l.split(",")
        rows[p[0].strip('"')] = np.array([float(x) for x in p[1:]])
    cols = [norm_bc(h.strip('"')) for h in head[1:]]
    tags = [r for r in rows if not r.lower().startswith(("no_match", "ambiguous", "total", "bad_struct", "unmapped"))]
    H = np.stack([rows[t] for t in tags])  # tags x barcodes
    hsum = H.sum(axis=0)
    top = H.max(axis=0) / np.maximum(hsum, 1)
    idx = {b: i for i, b in enumerate(cols) if b}
    j = np.array([idx.get(b, -1) for b in bcs])
    have = j >= 0
    hs = np.where(have, hsum[np.maximum(j, 0)], np.nan)
    tf = np.where(have, top[np.maximum(j, 0)], np.nan)
    # Reference for empty droplets: barcodes with < 10 RNA UMIs (free antibody only). HTO-positive = hashtag count >= 5x their
    # median; singlet = HTO-positive with one hashtag carrying >= 60% of the counts (a stained cell from one donor batch).
    tot0 = A["total"] if A is not None else np.zeros(len(bcs))
    emp = have & (tot0 < 10)
    thr = float(5 * np.median(hs[emp])) if emp.any() else float("nan")
    pos = hs >= thr
    singlet = pos & (tf >= 0.6)
    unc = ~called
    tot = A["total"] if A is not None else np.zeros(len(bcs))
    sig = unc & (np.nan_to_num(A["fdr_cell"], nan=1) < 0.01) if A is not None and "fdr_cell" in A.files else np.zeros(len(bcs), bool)
    return {
        "source": url, "tags": tags, "n_tabulated": len(cols), "stored_in_table": int(have.sum()), "n_stored": int(len(bcs)),
        "hto_threshold": thr, "empty_reference_median": float(np.median(hs[emp])) if emp.any() else None,
        "hto_pos_all_tabulated": int((hsum >= thr).sum()), "called_hto_singlet": float(np.nanmean(singlet[called & have])),
        "called_hto_pos": float(np.nanmean(pos[called & have])), "uncalled_hto_pos": float(np.nanmean(pos[unc & have])),
        "uncalled_hto_singlet": float(np.nanmean(singlet[unc & have])),
        "n_uncalled_hto_pos": int((pos & unc & have).sum()), "n_uncalled_hto_singlet": int((singlet & unc & have).sum()),
        "profile_sig_hto_pos": float(np.nanmean(pos[sig & have])) if (sig & have).any() else None,
        "profile_notsig_hto_pos": float(np.nanmean(pos[unc & ~sig & have])) if (unc & ~sig & have).any() else None,
        "uncalled_by_bin": bin_table({"frac_hto_pos": (np.where(have, pos, np.nan)[unc].astype(float), "mean"),
                                      "frac_hto_singlet": (np.where(have, singlet, np.nan)[unc].astype(float), "mean"),
                                      "median_hto_count": (hs[unc], "median"),
                                      "frac_profile_sig": (sig[unc].astype(float), "mean")}, tot[unc]),
        "called_by_bin": bin_table({"frac_hto_pos": (np.where(have, pos, np.nan)[called].astype(float), "mean"),
                                    "frac_hto_singlet": (np.where(have, singlet, np.nan)[called].astype(float), "mean"),
                                    "median_hto_count": (hs[called], "median")}, tot[called]),
        "hto_count_quantiles": {"called": q(hs[called & have]), "uncalled": q(hs[unc & have])},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gse", nargs="+")
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    for gse in a.gse:
        d = os.path.join(a.data, gse)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npz"):
                continue
            gsm = f[:-4]
            try:
                o = compare(gse, gsm, a.data, a.results)
                print(gse, gsm, o.get("author_source"), json.dumps(o.get("author", {}))[:300], flush=True)
            except Exception as e:
                print(gse, gsm, "FAILED", repr(e), flush=True)


if __name__ == "__main__":
    main()
