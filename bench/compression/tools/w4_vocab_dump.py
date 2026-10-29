#!/usr/bin/env python
"""w4_vocab: write NUL-terminated reference row-name files (exon / intron) of the fetched samples for the C++ roundtrip test.
    python -P tools/w4_vocab_dump.py OUT_DIR      -> OUT_DIR/<build>.exon.nul, <build>.intron.nul
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w4_vocab_lib as L
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
seen = set()
for r in csv.DictReader(open(os.path.join(L.DATA, "index.tsv")), delimiter="\t"):
    if r["build"] in seen: continue
    if r["kind"] != "exon": continue
    seen.add(r["build"])
    for kind in ("exon", "intron"):
        rows, _, _ = L.load_meta(os.path.join(L.DATA, "%s.%s.%s.meta" % (r["gse"], r["gsm"], kind)))
        with open(os.path.join(out, "%s.%s.nul" % (r["build"], kind)), "wb") as f:
            f.write(b"".join(x.encode() + b"\0" for x in rows))
