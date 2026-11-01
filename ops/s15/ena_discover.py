#!/usr/bin/env python3
"""S15: read counts per GSM from ENA for the pilot picks, plus deep/shallow candidates (excluding the pre-pilot ids). Output small TSVs only."""
import sys, json, urllib.parse, urllib.request, collections, csv
excl = set(open(sys.argv[1]).read().split()); picks = sys.argv[2].split(","); out = sys.argv[3]
F = "run_accession,sample_alias,study_alias,experiment_title,read_count,library_layout,instrument_model,library_strategy"
def ena(q, lim=0):
    u = "https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&format=tsv&limit=%d&fields=%s&query=%s" % (lim, F, urllib.parse.quote(q))
    return list(csv.DictReader(urllib.request.urlopen(u, timeout=600).read().decode().splitlines(), delimiter="\t"))
def agg(rows):
    d = collections.OrderedDict()
    for r in rows:
        g = r["sample_alias"]
        if not g.startswith("GSM"): continue
        x = d.setdefault(g, {"gsm": g, "gse": r["study_alias"], "title": r["experiment_title"][:90], "runs": 0, "reads": 0, "inst": r["instrument_model"]})
        x["runs"] += 1; x["reads"] += int(r["read_count"] or 0)
    return d
# picks
rows = []
for i in range(0, len(picks), 10):
    rows += ena(" OR ".join('sample_alias="%s"' % p for p in picks[i:i+10]))
pk = agg(rows)
with open(out + ".picks.tsv", "w") as f:
    for g in picks: x = pk.get(g); f.write("\t".join(map(str, [g, x and x["gse"], x and x["runs"], x and x["reads"], x and x["title"], x and x["inst"]])) + "\n")
# discovery: human single-cell RNA-seq
base = 'library_source="TRANSCRIPTOMIC SINGLE CELL" AND tax_id=9606 AND library_strategy="RNA-Seq" AND instrument_platform="ILLUMINA"'
allr = ena(base)
d = agg(allr)
cand = [x for g, x in d.items() if g not in excl and x["gse"] not in excl]
with open(out + ".deep.tsv", "w") as f:
    for x in sorted(cand, key=lambda x: -x["reads"])[:60]: f.write("\t".join(map(str, [x["gsm"], x["gse"], x["runs"], x["reads"], x["title"], x["inst"]])) + "\n")
print("runs", len(allr), "gsm", len(d), "cand", len(cand), "deep>600M", sum(x["reads"] > 600e6 for x in cand))
