#!/usr/bin/env python3
"""Link each GEX sample to its companion libraries (ADT/HTO/CRISPR/VDJ/ATAC) by GSE + GEO title stem.

Titles come from NCBI E-utilities (db=gds), never from GEO web pages. Rate: 3 requests/s (no API key). Responses are
cached under --cache so a rerun costs nothing. Ambiguous or unmatched libraries are written out, not guessed.

  companion_libs.py --samples samples.parquet --cache DIR --out DIR
Writes companions.tsv (gex_gsm, lib_type, lib_gsm, gse, stem, lib_runs, lib_reads) and ambiguous.tsv (lib_gsm, lib_type, gse, reason, candidates).
"""
import argparse, json, os, re, sys, time, urllib.parse, urllib.request

EU = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
LIB_TYPES = ("adt", "hto", "crispr", "vdj", "atac", "multiome", "spatial")
_last = [0.0]


def get(endpoint, **params):
    wait = 0.34 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    url = EU + endpoint + "?" + urllib.parse.urlencode(dict(params, retmode="json", tool="singlet-companion"))
    for attempt in range(5):
        try:
            _last[0] = time.time()
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # 429/5xx/timeouts: back off
            time.sleep(2 ** attempt)
            err = e
    raise RuntimeError("eutils failed: %s %s" % (url, err))


def gse_titles(gse, cache):
    path = os.path.join(cache, gse + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    r = get("esearch.fcgi", db="gds", term="%s[ACCN] AND gsm[ETYP]" % gse, retmax=100000)["esearchresult"]
    ids, out = r.get("idlist", []), {}
    # the GSE record itself is not a GSM; esearch on the series accession returns its samples via the gds link
    if not ids:
        r = get("esearch.fcgi", db="gds", term="%s[GSE] AND gsm[ETYP]" % gse, retmax=100000)["esearchresult"]
        ids = r.get("idlist", [])
    for i in range(0, len(ids), 200):
        res = get("esummary.fcgi", db="gds", id=",".join(ids[i:i + 200]))["result"]
        for u in res.get("uids", []):
            e = res[u]
            out[e["accession"]] = e.get("title", "")
    json.dump(out, open(path, "w"))
    return out


_W = ("gene[ _-]?expression", "gexl?", "rna", "cite(?:[ _-]?seq)?", "adt(?:[ _-]?seq)?", "hto", "hashtags?",
      "antibody(?: capture)?", "fbc", "feature", "feature[ _-]?barcod\w*", "crispr", "grna", "sgrna", "guides?",
      r"v\(d\)j", "vdj", "tcr", "bcr", "sctcr", "scbcr", "tel", "bel", "atac", "multiome", "scrna(?:[ _-]?seq)?",
      "seq", "fastqs?", "library", r"sequencing run \d+")
TOKEN = re.compile(r"(?<![a-z0-9])(?:" + "|".join(_W) + r")(?![a-z0-9])|\[[^\]]*\]|\(.*?fastqs?\)", re.I)


def stem(title):
    t = TOKEN.sub(" ", title or "")
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    s = pd.read_parquet(a.samples)
    t = s.title_lib_type.fillna("").astype(str)
    s["kind"] = [t_ if (ta == "scrna" and t_ in ("adt", "hto", "crispr", "vdj")) else (ta if ta in LIB_TYPES else
                 "gex" if ta == "scrna" else "") for ta, t_ in zip(s.target_assay.fillna("").astype(str), t)]
    gses = sorted(set(s[s.kind.isin(LIB_TYPES)].gse_id) & set(s[s.kind == "gex"].gse_id))
    print("studies to fetch:", len(gses), flush=True)
    pairs, amb = [], []
    for n, gse in enumerate(gses):
        titles = gse_titles(gse, a.cache)
        d = s[s.gse_id == gse]
        gex = d[d.kind == "gex"]
        by_stem = {}
        for g in gex.gsm_id:
            by_stem.setdefault(stem(titles.get(g)), []).append(g)
        for _, r in d[d.kind.isin(LIB_TYPES)].iterrows():
            st = stem(titles.get(r.gsm_id))
            c = by_stem.get(st, [])
            if st and len(c) == 1:
                pairs.append((c[0], r.kind, r.gsm_id, gse, st, str(r.srr_accessions or ""), int(r.reads) if r.reads == r.reads else 0))
            else:
                amb.append((r.gsm_id, r.kind, gse, "no_title" if not titles.get(r.gsm_id) else
                            ("no_stem_match" if not c else "multiple_gex"), ",".join(c[:8])))
        if n % 25 == 0:
            print(n, "/", len(gses), flush=True)
    for name, rows, cols in (("companions.tsv", pairs, "gex_gsm lib_type lib_gsm gse stem lib_runs lib_reads"),
                             ("ambiguous.tsv", amb, "lib_gsm lib_type gse reason candidates")):
        with open(os.path.join(a.out, name), "w") as f:
            f.write("\t".join(cols.split()) + "\n")
            for r in rows:
                f.write("\t".join(map(str, r)) + "\n")
    cp = pd.DataFrame(pairs, columns=["gex_gsm", "lib_type", "lib_gsm", "gse", "stem", "lib_runs", "lib_reads"])
    print("paired lib GSMs by type:\n", cp.lib_type.value_counts().to_string())
    print("distinct GEX samples with a companion, by type:\n", cp.groupby("lib_type").gex_gsm.nunique().to_string())
    ca = pd.DataFrame(amb, columns=["lib_gsm", "lib_type", "gse", "reason", "cand"])
    print("ambiguous:\n", ca.groupby(["lib_type", "reason"]).size().to_string())
    done = set(s[s.processing_status == "done"].gsm_id)
    print("GEX samples already done with a companion:", cp[cp.gex_gsm.isin(done)].groupby("lib_type").gex_gsm.nunique().to_dict())


if __name__ == "__main__":
    sys.exit(main())
