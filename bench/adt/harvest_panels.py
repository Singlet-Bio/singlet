# SPDX-License-Identifier: MIT
"""Harvest the authors' feature references from GEO supplementary files (runs on Clipper; small text files only).

    python -P bench/adt/harvest_panels.py --inventory adt_hto_inventory.tsv --out $WORK/state/adt/panels_geo

For every candidate file named in the inventory's ``panel_files`` (sample- or series-level supplementary files whose
names suggest a feature reference): skip it when larger than ``MAX_BYTES`` or not text, else download it and look for a
column of nucleotide barcodes (ACGT, 8-25 nt, at least 2 rows) and a name column. Each parsed file becomes a Cell Ranger
style ``<out>/files/<file>.csv``; ``index.tsv`` lists every candidate and what happened to it; ``geo_catalog.tsv`` is the
union of parsed sequences with the authors' names (most common spelling first, every spelling in ``aliases``) and the
studies that use them. This second catalog tier names a sequence only as some study's authors named it.
"""

import argparse
import collections
import gzip
import io
import os
import re
import sys
import time
import urllib.request

import pandas as pd

MAX_BYTES = 5 << 20
TEXT = re.compile(r"\.(csv|tsv|txt|tab)(\.gz)?$", re.I)
SEQ = re.compile(r"^[ACGTacgt]{8,25}$")


def url_for(name):
    acc = name.split("_")[0]
    if acc.startswith("GSM"):
        return f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{acc[:-3]}nnn/{acc}/suppl/{name}"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{acc[:-3]}nnn/{acc}/suppl/{name}"


def fetch(url):
    req = urllib.request.Request(url, method="HEAD")
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                size = int(r.headers.get("Content-Length") or 0)
            if size > MAX_BYTES:
                return None, size
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            return data, len(data)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b"", -404
            time.sleep(2 ** i)
        except Exception:
            time.sleep(2 ** i)
    return b"", -1


def parse(data, name):
    if name.endswith(".gz"):
        data = gzip.decompress(data)
    text = data.decode("utf-8", "replace")
    best = None
    for sep in (",", "\t", None):
        for header in (0, None):
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep, header=header, dtype=str, engine="python")
            except Exception:
                continue
            for c in df.columns:
                v = df[c].dropna().astype(str).str.strip()
                ok = v.str.fullmatch(SEQ.pattern)
                if ok.sum() >= 2 and ok.mean() >= 0.8:
                    names = None
                    prio = {"name": 0, "antibody": 1, "adt": 2, "feature": 3, "tag": 4, "id": 5}
                    for c2 in sorted(df.columns, key=lambda x: prio.get(str(x).lower(), 9)):
                        if c2 == c:
                            continue
                        w = df[c2].dropna().astype(str)
                        if w.nunique() <= 1 or w.str.fullmatch(r"(?i)R[12]|antibody capture|custom|5P.*|\^.*").mean() > 0.5:
                            continue  # read / feature_type / pattern columns
                        if len(w) and not w.str.fullmatch(SEQ.pattern).any() and w.str.match(r"^[A-Za-z]").mean() > 0.5:
                            names = c2
                            break
                    if names is None:
                        continue
                    sub = df[ok.reindex(df.index, fill_value=False)]
                    pat = None
                    for c3 in df.columns:
                        if str(c3).lower() == "pattern":
                            pat = sub[c3]
                    out = pd.DataFrame({"name": sub[names].astype(str).str.strip(),
                                        "sequence": sub[c].str.upper().str.strip(),
                                        "pattern": pat if pat is not None else ""})
                    if best is None or len(out) > len(best):
                        best = out
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    inv = pd.read_csv(a.inventory, sep="\t", dtype=str)
    os.makedirs(os.path.join(a.out, "files"), exist_ok=True)
    cand = collections.OrderedDict()
    for r in inv.itertuples():
        for f in str(r.panel_files).split(";") if isinstance(r.panel_files, str) else []:
            if f:
                cand.setdefault(f, set()).add(r.gse)
    rows, cat = [], collections.defaultdict(list)
    for i, (f, gses) in enumerate(cand.items()):
        rec = {"file": f, "gses": ",".join(sorted(gses)), "status": "", "bytes": 0, "n_features": 0, "lengths": ""}
        if not TEXT.search(f):
            rec["status"] = "not text"
            rows.append(rec)
            continue
        cached = os.path.join(a.out, "raw", f)
        if os.path.exists(cached):
            data = open(cached, "rb").read()
            size = len(data)
        else:
            data, size = fetch(url_for(f))
            if data:
                os.makedirs(os.path.dirname(cached), exist_ok=True)
                with open(cached, "wb") as fh:
                    fh.write(data)
        rec["bytes"] = size
        if data is None:
            rec["status"] = "too large"
        elif size < 0:
            rec["status"] = f"download failed ({size})"
        else:
            try:
                p = parse(data, f)
            except Exception as e:
                p, rec["status"] = None, f"parse error {type(e).__name__}"
            if p is None or len(p) == 0:
                rec["status"] = rec["status"] or "no barcode column"
            else:
                rec["status"] = "parsed"
                rec["n_features"] = len(p)
                rec["lengths"] = ",".join(map(str, sorted(p.sequence.str.len().unique())))
                p.assign(id=p.name, read="R2", feature_type="Antibody Capture")[
                    ["id", "name", "read", "pattern", "sequence", "feature_type"]].to_csv(
                    os.path.join(a.out, "files", re.sub(r"\.(gz)$", "", f) + ".csv"), index=False)
                for q, n in zip(p.sequence, p.name):
                    if isinstance(n, str) and n and n.lower() != "nan":
                        cat[q].append((n, f.split("_")[0], ",".join(sorted(gses))))
        rows.append(rec)
        if i % 50 == 0:
            print(f"{i}/{len(cand)}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "index.tsv"), sep="\t", index=False)
    out = []
    for q, lst in sorted(cat.items()):
        names = collections.Counter(n for n, _, _ in lst)
        out.append({"sequence": q, "name": names.most_common(1)[0][0], "aliases": "|".join(sorted(map(str, names))),
                    "n_files": len(lst), "studies": ",".join(sorted({g for _, _, g in lst}))})
    pd.DataFrame(out).to_csv(os.path.join(a.out, "geo_catalog.tsv"), sep="\t", index=False)
    print(pd.DataFrame(rows).status.value_counts().to_dict(), len(out), "sequences")


if __name__ == "__main__":
    sys.exit(main())
