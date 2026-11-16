# SPDX-License-Identifier: MIT
"""ARI of donor_assignments.tsv on Kang batch-1 lane A with the registered scorer (bench-public demux_eval).
usage: score_kang_ari.py LIB_DIR TRUTH_DIR OUT_JSON PATH:LABEL [PATH:LABEL ...]"""
import sys, json
from pathlib import Path
import pandas as pd
lib, truth_dir, out, *items = sys.argv[1:]
sys.path.insert(0, lib); sys.path.insert(0, str(Path(lib).parent.parent / "capabilities"))
import pools as P, demux_eval as DE
pool = "kang2018_b1_a"
T = P.load_truth(truth_dir, pool); T = T[T["ind"] != "<NA>"]
res = {}
for it in items:
    f, tag = it.rsplit(":", 1)
    d = pd.read_csv(f, sep="\t", dtype=str)
    d = d[d["cell_status"] == "cell"] if "cell_status" in d else d
    d = d.set_index("cell")["donor_id"]
    r = DE.score(T, d); res[tag] = r
    print(tag, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()})
json.dump(res, open(out, "w"), indent=1)
