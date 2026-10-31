# SPDX-License-Identifier: MIT
"""Development-library diagnostic: reference vs method junction where the top junction differs.
    python -P diag_diff.py RUNS DATA LIB CHAIN [V-prefix]
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vdj"))
import compare_vdj as CV  # noqa: E402

runs, data, lib, chain = sys.argv[1:5]
vp = sys.argv[5] if len(sys.argv) > 5 else ""
ref = CV.load_cellranger(os.path.join(data, lib, "filtered_contig_annotations.csv"))
ref = ref[ref.chain == chain].sort_values("umis", ascending=False).drop_duplicates(["barcode", "chain"])
m = CV.load_singlet(os.path.join(runs, lib, os.environ.get("SDIR", "singlet"), "contigs.csv"))
m = m[m.chain == chain].sort_values("umis", ascending=False).drop_duplicates(["barcode", "chain"])
j = ref.merge(m, on=["barcode", "chain"], suffixes=("_r", "_m"))
d = j[(j.cdr3_nt_r != j.cdr3_nt_m) & j.v_r.str.startswith(vp)]
print(len(d), "different of", len(j))
for r in d.head(25).itertuples():
    print(r.v_r, r.v_m, r.j_r, r.j_m, int(r.umis_r), int(r.umis_m))
    print("  R", r.cdr3_nt_r)
    print("  M", r.cdr3_nt_m)
