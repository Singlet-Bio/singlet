# SPDX-License-Identifier: MIT
"""Compare two engine runs (base, new) of the same libraries: file identity per library, and doublet scores against truth.

    engine_compare.py G DATA WORK TRUTH     G/runs/{base,new}/<ds>/ ; DATA, WORK, TRUTH as dbl_sweep.py
"""
import hashlib
import importlib.util
import json
import os
import sys

import pandas as pd

G, DATA, WORK, TRUTH = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("dbl_sweep", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dbl_sweep.py"))
sw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sw)
SKIP = {"run.log", "time.txt", "command.txt", "exit.txt", "stream_path.txt", "node.txt", "provenance.json", "summary.json",
        "metrics_summary.csv", "read_stats.tsv", "star_Log.out", "star_Log.final.out", "star_Log.progress.out", "star_Log.std.out", "binary_sha256.txt", "pileup_stats.json"}


def md5(p):
    if p.endswith("ambient_contamination.json"):   # carries a wall-time field
        import re
        return hashlib.md5(re.sub(rb'"seconds": [0-9.e+-]+', b"", open(p, "rb").read())).hexdigest()
    h = hashlib.md5()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


out = {"identity": {}}
for ds in sorted(os.listdir(f"{G}/runs/base")):
    b, n = f"{G}/runs/base/{ds}", f"{G}/runs/new/{ds}"
    if not os.path.isdir(n):
        continue
    fb = {f for f in os.listdir(b) if os.path.isfile(f"{b}/{f}")} - SKIP
    fn = {f for f in os.listdir(n) if os.path.isfile(f"{n}/{f}")} - SKIP
    same = [f for f in sorted(fb & fn) if md5(f"{b}/{f}") == md5(f"{n}/{f}")]
    diff = sorted((fb & fn) - set(same))
    out["identity"][ds] = {"identical": len(same), "different": diff, "only_base": sorted(fb - fn), "only_new": sorted(fn - fb),
                           "exit": [open(f"{b}/exit.txt").read().strip(), open(f"{n}/exit.txt").read().strip()]}
    print(ds, len(same), "identical; differ:", diff, flush=True)

for label in ("base", "new"):
    for ds in list(sw.KANG) + sw.BARN:
        p = f"{G}/runs/{label}/{ds}/doublet_scores.tsv"
        if not os.path.exists(p):
            continue
        bcs = sw.strip(pd.read_csv(f"{DATA}/{ds}.barcodes.tsv", header=None)[0]).values
        d = pd.read_csv(p, sep="\t")
        d["barcode"] = sw.strip(d.barcode)
        d = d.drop_duplicates("barcode").set_index("barcode").reindex(bcs)
        os.makedirs(f"{G}/scored/{label}", exist_ok=True)
        pd.DataFrame({"score": d.doublet_score.fillna(-1).values, "is_doublet": d.is_doublet.astype(str).isin(["True", "1", "TRUE"]).values}) \
            .to_csv(f"{G}/scored/{label}/{ds}.tsv", sep="\t", index=False)
    sw.score(DATA, f"{G}/scored", label, f"{G}/scored/{label}.json", TRUTH, WORK)
json.dump(out, open(f"{G}/compare.json", "w"), indent=1)
