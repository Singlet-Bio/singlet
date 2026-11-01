"""Pick a stratified set of Research Alpha samples whose GEO sample record carries the authors' filtered matrix.

    python -I select_samples.py CENSUS_DIR POLICY_JSON OUT_TSV [--per-stratum 10] [--pool 160] [--seed 7]

Inputs (read-only, from the Alpha census):
  CENSUS_DIR/d1/{sample_qc,gsm,bundle_index}.jsonl   D1 dumps (singlet-catalog, read-only SELECTs)
  CENSUS_DIR/studies_action.tsv                       per-study action and `downloadable_now`
  POLICY_JSON                                         alpha-policy.json: withheld_studies, excluded_samples

A sample is "in Alpha" when its study bundle is not withheld, the sample is not excluded (strand / non-RNA library /
empty matrix), the study is downloadable now, and the sample is usable with QC tier 0 (the public site's basis).

Strata: species (human GRCh38 / mouse GRCm39) x chemistry (10x 3' v2, 3' v3, 5', nuclei) x size (Singlet called cells
< 3,000 / >= 3,000). Chemistry comes from Singlet's protocol label plus the GEO curation's assay family; nuclei and 5'
are confirmed from the GEO sample text. One sample per study per stratum.

A sample qualifies when its GEO *sample* supplementary folder holds a Cell Ranger-style filtered matrix: a 10x .h5
without "raw" in the name, or a matrix.mtx + barcodes.tsv + features/genes.tsv triplet without "raw".
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import random
import re
import time
import urllib.request
from collections import defaultdict

UA = {"User-Agent": "singlet-alpha-bench/1.0 (+https://singlet.bio)"}
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/samples"
NUC_RE = re.compile(r"single[- ]?nucle|snrna|sn-rna|nuclei|nucleus isolation|isolated nucle|nuclear suspension|"
                    r"single nuclear|snseq|sn-seq|dronc", re.I)
P5_RE = re.compile(r"5['’ ]?\s*(prime|')|5'|five prime|single cell 5|sc5p|5' v[123]|5’|vdj", re.I)


def get(url, tries=4, timeout=60):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (i + 1))
        except Exception:  # noqa: BLE001
            time.sleep(2 * (i + 1))
    return None


def geo_listing(gsm):
    html = get(f"{GEO}/{gsm[:-3]}nnn/{gsm}/suppl/")
    if html is None:
        return []
    out = []
    for m in re.finditer(r'href="([^"?/][^"]*)"[^\n]*?\s(\d[\d.]*[KMG]?)\s*$', html, re.M):
        out.append((m.group(1), m.group(2)))
    if not out:
        out = [(f, "") for f in re.findall(r'href="([^"?/][^"]*)"', html)]
    return out


def size_bytes(s):
    if not s:
        return None
    mult = {"K": 1e3, "M": 1e6, "G": 1e9}.get(s[-1], 1)
    try:
        return int(float(s.rstrip("KMG")) * mult)
    except ValueError:
        return None


def matrix_kind(files):
    """'h5' / 'mtx' / None, and the files used."""
    names = [f for f, _ in files]
    sizes = dict(files)
    h5 = [f for f in names if f.lower().endswith(".h5") and "raw" not in f.lower() and "molecule" not in f.lower()]
    if h5:
        f = sorted(h5, key=lambda x: ("filtered" not in x.lower(), x))[0]
        return "h5", [f], size_bytes(sizes.get(f))
    trip = {}
    for f in names:
        lf = f.lower()
        if "raw" in lf:
            continue
        for key in ("matrix", "barcodes", "features", "genes"):
            if re.search(key + r"\.(mtx|tsv)(\.gz)?$", lf):
                trip["features" if key == "genes" else key] = f
    if {"matrix", "barcodes", "features"} <= set(trip):
        return "mtx", [trip["matrix"], trip["barcodes"], trip["features"]], size_bytes(sizes.get(trip["matrix"]))
    return None, [], None


def geo_text(gsm):
    t = get(f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm}&targ=self&form=text&view=brief") or ""
    keep = [l for l in t.splitlines() if l.startswith(("!Sample_title", "!Sample_source", "!Sample_characteristics",
                                                         "!Sample_extract_protocol", "!Sample_description",
                                                         "!Sample_library", "!Sample_molecule"))]
    return "\n".join(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("census")
    ap.add_argument("policy")
    ap.add_argument("out")
    ap.add_argument("--per-stratum", type=int, default=10)
    ap.add_argument("--pool", type=int, default=160, help="candidates screened per (species, chemistry label)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    L = lambda f: [json.loads(l) for l in open(f"{a.census}/d1/{f}.jsonl", encoding="utf-8")]  # noqa: E731
    qc = {r["gsm_id"]: r for r in L("sample_qc")}
    gsm = {r["gsm_id"]: r for r in L("gsm")}
    pol = json.load(open(a.policy, encoding="utf-8"))
    wh, ex = set(pol["withheld_studies"]), set(pol["excluded_samples"])
    act = {r["gse"]: r for r in csv.DictReader(open(f"{a.census}/studies_action.tsv", encoding="utf-8"), delimiter="\t")}
    pub = []
    for line in open(f"{a.census}/d1/bundle_index.jsonl", encoding="utf-8"):
        r = json.loads(line)
        g = r["gse_id"]
        if g in wh or act.get(g, {}).get("downloadable_now") != "1":
            continue
        for s in sorted({e["p"].split("/")[1] for e in json.loads(r["entries"]) if e["p"].startswith("samples/")}):
            q = qc.get(s)
            if s in ex or not q or q["usable"] != 1 or (q["qc_tier"] or 0) != 0:
                continue
            pub.append((g, s))
    print("alpha public samples", len(pub))
    sp = {"GRCh38-2024-A": "human", "GRCm39-2024-A": "mouse"}
    pools = defaultdict(list)
    for g, s in pub:
        q, m = qc[s], gsm.get(s, {})
        if q["reference_build"] not in sp:
            continue
        fam = m.get("assay_family")
        if fam == "10x 5'" or q["protocol"] == "10xv3-5p":
            lab = "5p"
        elif fam == "10x 3'" and q["protocol"] in ("10xv2", "10xv3"):
            lab = q["protocol"]
        else:
            continue
        pools[(sp[q["reference_build"]], lab)].append((g, s))
    rng = random.Random(a.seed)
    cands = []
    for k, v in sorted(pools.items()):
        rng.shuffle(v)
        seen, pick = set(), []
        for g, s in v:
            if g in seen:
                continue
            seen.add(g)
            pick.append((k, g, s))
            if len(pick) >= (a.pool * (2 if k[1] != "5p" else 1)):
                break
        print(k, "pool", len(v), "screen", len(pick))
        cands += pick

    def screen(c):
        (spc, lab), g, s = c
        files = geo_listing(s)
        kind, used, nbytes = matrix_kind(files)
        if kind is None or (nbytes and nbytes > 1.5e9):
            return None
        txt = geo_text(s)
        return dict(species=spc, label=lab, gse=g, gsm=s, author_kind=kind, author_files=";".join(used),
                    author_bytes=nbytes or "", nuclei=int(bool(NUC_RE.search(txt))),
                    text_5p=int(bool(P5_RE.search(txt))), protocol=qc[s]["protocol"],
                    assay_family=gsm.get(s, {}).get("assay_family"), n_cells_singlet=qc[s]["n_cells_called"],
                    reference_build=qc[s]["reference_build"], n_files_geo=len(files))

    ok = []
    with cf.ThreadPoolExecutor(6) as pool:
        for i, r in enumerate(pool.map(screen, cands)):
            if r:
                ok.append(r)
            if i % 100 == 0:
                print("screened", i, "with matrix", len(ok), flush=True)
    print("with filtered matrix", len(ok), "of", len(cands))
    for r in ok:
        if r["nuclei"]:
            r["chem"] = "nuclei"
        elif r["label"] == "5p":
            r["chem"] = "5p"
        else:
            r["chem"] = {"10xv2": "3p_v2", "10xv3": "3p_v3"}[r["label"]]
        r["size"] = "large" if (r["n_cells_singlet"] or 0) >= 3000 else "small"
        r["stratum"] = f"{r['species']}|{r['chem']}|{r['size']}"
    by = defaultdict(list)
    for r in ok:
        by[r["stratum"]].append(r)
    final, used_gse = [], set()
    for st in sorted(by):
        n = 0
        for r in by[st]:
            if r["gse"] in used_gse:
                continue
            final.append(r)
            used_gse.add(r["gse"])
            n += 1
            if n >= a.per_stratum:
                break
        print(st, "available", len(by[st]), "picked", n)
    cols = ["stratum", "species", "chem", "size", "gse", "gsm", "protocol", "assay_family", "reference_build",
            "n_cells_singlet", "nuclei", "text_5p", "author_kind", "author_files", "author_bytes"]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(final)
    with open(a.out.replace(".tsv", "_screened.tsv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols + ["label"], delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(ok)
    print("final", len(final))


if __name__ == "__main__":
    main()
