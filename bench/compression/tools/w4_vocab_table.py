#!/usr/bin/env python
"""w4_vocab: fill the size comparison table in notes/w4_vocab.md from results/w4_vocab_baseline_<build>.json and w4_vocab_sizes.tsv."""
import csv, json, os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
sizes = {}
rows = list(csv.reader(open(os.path.join(R, "w4_vocab_sizes.tsv")), delimiter="\t"))
builds = rows[0][1:]
for r in rows[1:]:
    for b, v in zip(builds, r[1:]): sizes.setdefault(b, {})[r[0]] = v
TRACK_A = {"GRCh38-2024-A": 3569124, "GRCm39-2024-A": 4470944}
hdr = ["build", "naive: exon+intron names (NUL joined)", "+ feature_vocab.json raw", "zstd-19 (exon, intron separately)", "xz-9e (exon, intron separately)",
       "feature_vocab.json: deflate-9 / zstd-19 / xz-9e", "Track A vocabulary file (xz)", "structured pack (all three)", "pack / (xz names + deflated vocab json)"]
out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
for b in builds:
    p = os.path.join(R, "w4_vocab_baseline_%s.json" % b)
    s = sizes[b]
    naive = int(s["exon_raw"]) + int(s["intron_raw"])
    if os.path.exists(p):
        x = json.load(open(p))
        xz = x["names_xz9e_each"]; z = x["names_zstd19_each"]
        vj = "%s / %s / %s" % (format(x["vocab_json_deflate9"], ","), format(x["vocab_json_zstd19"], ","), format(x["vocab_json_xz9e"], ","))
        ratio = "%.2fx smaller" % ((xz + x["vocab_json_deflate9"]) / int(s["pack_bytes"])) if int(s["pack_bytes"]) > 1000 else "composite"
        zs, xs = format(z, ","), format(xz, ",")
    else:
        zs = xs = vj = ratio = "not finished"
    ta = format(TRACK_A[b], ",") if b in TRACK_A else "-"
    out.append("| %s | %s | %s | %s | %s | %s | %s | **%s** | %s |" % (b, format(naive, ","), format(naive + int(s["vocab_json_raw"]), ","), zs, xs, vj, ta,
                                                                 format(int(s["pack_bytes"]), ","), ratio))
tab = "\n".join(out)
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notes", "w4_vocab.md")
t = open(p).read()
if "<!--BASELINES-->" in t:
    t = t.replace("<!--BASELINES-->", tab)
else:
    a = t.index("| build | naive:"); b = t.index("\n\n", a)
    t = t[:a] + tab + t[b:]
open(p, "w").write(t)
print(tab)
