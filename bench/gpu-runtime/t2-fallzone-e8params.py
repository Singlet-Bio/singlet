#!/usr/bin/env python3
"""t2-fallzone (follow-up): per GPU run of the named campaign directory, the inputs of the engine's "E8 does not run" test that its own logs and
provenance show: the STAR parameter block of the item's out/star_Log.out (effective parameters, first block that names them) and the campaign's
pipeline_args / engine_protocol / assay. Reads campaign_provenance.json (fixed fields) and out/star_Log.out (regex hits). Nothing else.
usage: t2-fallzone-e8params.py CAMPAIGN_DIR CENSUS_TSV  -> TSV on stdout (CENSUS_TSV: output of t2-fallzone-census.py, for the list of keys)"""
import json, os, re, sys

PAR = ("outSAMtype", "soloType", "soloCBlen", "soloUMIlen", "soloStrand", "soloBarcodeMate", "soloFeatures", "quantMode", "soloCBposition",
       "soloUMIposition", "chimSegmentMin", "outFilterType", "waspOutputMode", "outSJtype")
LV = re.compile(r"E19: level (\d) reached")
NU = re.compile(r"E10 \(level 5\) not used: ([^\n]*)")


def params(path):
    d = {}
    try:
        with open(path, errors="replace") as f:
            txt = f.read(3 << 20)
    except OSError:
        return d
    blk = txt.split("##### Final user input:")[-1] if "##### Final user input:" in txt else txt
    for ln in blk.splitlines():
        t = ln.split(None, 1)
        if len(t) == 2 and t[0] in PAR and t[0] not in d:
            d[t[0]] = " ".join(t[1].replace("~RE-DEFINED", "").split())[:60]
    m = LV.search(txt); d["level"] = m.group(1) if m else ""
    m = NU.search(txt); d["not_used"] = m.group(1)[:50] if m else ""
    rf = re.search(r"^readFilesIn\s+(.*)$", blk, re.M)
    d["nFiles"] = str(len(rf.group(1).split())) if rf else ""
    return d


cdir, census = sys.argv[1], sys.argv[2]
rows = [l.rstrip("\n").split("\t") for l in open(census)][1:]
cols = list(PAR) + ["nFiles", "level", "not_used"]
print("\t".join(["key", "device", "pairs_M", "assay", "engine_protocol", "chimeric_flag", "hla_args"] + cols))
for r in rows:
    if len(r) < 19 or r[3] in ("", "bigmem") or r[18] == "":
        continue
    sh, key = r[1], r[2]
    d = os.path.join(cdir, "samples", sh, key)
    try:
        c = json.load(open(os.path.join(d, "campaign_provenance.json")))
    except (OSError, ValueError):
        c = {}
    pa = c.get("pipeline_args") or []
    p = params(os.path.join(d, "out", "star_Log.out"))
    print("\t".join([key, r[3], "%.1f" % (float(r[5] or 0) / 1e6), str(c.get("assay")), str(c.get("engine_protocol")), "yes" if "--chimeric" in pa else "no",
                     "yes" if any(a.startswith("--hla") for a in pa) else "no"] + [p.get(k, "") for k in cols]))
