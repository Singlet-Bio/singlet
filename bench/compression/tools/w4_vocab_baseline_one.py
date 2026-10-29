#!/usr/bin/env python
"""w4_vocab: baseline sizes (naive, zstd-19, xz-9e) of the plain row names + feature_vocab.json for ONE build.
    python -P tools/w4_vocab_baseline_one.py <build>   -> results/w4_vocab_baseline_<build>.json
(run the builds in parallel processes; equals `w4_vocab_build.py --baselines` row for row)"""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w4_vocab_lib as L, w4_vocab_build as B
build = sys.argv[1]
r0 = next(r for r in csv.DictReader(open(os.path.join(L.DATA, "index.tsv")), delimiter="\t") if r["build"] == build)
gse, gsm = r0["gse"], r0["gsm"]
ex, _, _ = L.load_meta(os.path.join(L.DATA, "%s.%s.exon.meta" % (gse, gsm)))
it, _, _ = L.load_meta(os.path.join(L.DATA, "%s.%s.intron.meta" % (gse, gsm)))
vraw = open(os.path.join(L.DATA, "%s.feature_vocab.json" % gse), "rb").read()
r = B.sizes_of(ex, it, vraw)
r["build"] = build
json.dump(r, open(os.path.join(L.ROOT, "results", "w4_vocab_baseline_%s.json" % build), "w"), indent=1)
print(build, r)
