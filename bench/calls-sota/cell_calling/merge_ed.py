# SPDX-License-Identifier: MIT
"""Join the DropletUtils::emptyDrops calls (emptydrops.tsv) to the call table of a held-out library. python -P merge_ed.py OUT_DIR"""
import sys

import pandas as pd

O = sys.argv[1]
t = pd.read_csv(f"{O}/table_nocomp.tsv.gz", sep="\t", dtype={"barcode": str})
e = pd.read_csv(f"{O}/emptydrops.tsv", sep="\t")
assert len(e) == len(t) and (e["total"].values == t["umi_total"].values).all(), "row mismatch"
t["edcr"] = e["edcr"].values
t.to_csv(f"{O}/table.tsv.gz", sep="\t", index=False)
print("table", len(t))
