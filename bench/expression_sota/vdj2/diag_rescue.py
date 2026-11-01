# SPDX-License-Identifier: MIT
"""Development only: the contigs a filter variant keeps that the base run does not, against the reference cell.

    python -P diag_rescue.py RUNS DATA LIB:TCR|BCR BASE_DIR VARIANT_DIR
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vdj"))
import compare_vdj as CV  # noqa: E402

runs, data, spec, bdir, vdir = sys.argv[1:6]
lib, ch = spec.split(":")
keep = CV.TCR if ch == "TCR" else CV.BCR
ref = CV.load_cellranger(os.path.join(data, lib, "filtered_contig_annotations.csv"))
ref = ref[ref.chain.isin(keep)]
cells = CV.cellranger_cells(os.path.join(data, lib, "filtered_contig_annotations.csv"))
rk = set(zip(ref.barcode, ref.chain, ref.cdr3_nt))
rchains = ref.groupby("barcode").apply(lambda g: sorted(zip(g.chain, g.umis.astype(int), g.cdr3_nt.str[-12:])))
b = CV.load_singlet(os.path.join(runs, lib, bdir, "contigs.csv"))
v = CV.load_singlet(os.path.join(runs, lib, vdir, "contigs.csv"))
bk = set(zip(b.barcode, b.chain, b.cdr3_nt))
new = v[[k not in bk for k in zip(v.barcode, v.chain, v.cdr3_nt)]]
gp = os.path.join(runs, lib, "gex_cells.txt")
gex = set(l.strip() for l in open(gp)) if os.path.exists(gp) else None
for r in new.itertuples():
    k = (r.barcode, r.chain, r.cdr3_nt)
    print(r.barcode, r.chain, r.umis, "in_ref" if k in rk else "not_in_ref", "ref_cell" if r.barcode in cells else "not_ref_cell",
          "gex" if gex is None or r.barcode in gex else "not_gex", rchains.get(r.barcode, "-"))
print("new", len(new), "in ref", sum((x in rk) for x in zip(new.barcode, new.chain, new.cdr3_nt)))
