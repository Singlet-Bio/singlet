# SPDX-License-Identifier: MIT
"""Development only: Singlet's wrong-lineage contigs (TCR on GEX-B barcodes, BCR on GEX-T barcodes) and what lets them through.

    python -P diag_wl.py RUNS DATA SDIR LIB:TCR|BCR [...]

Per wrong contig: UMIs, reads, junction UMIs, the junction's UMIs in its strongest barcode (jmax: ambient source?), whether the
barcode is a Cell Ranger V(D)J cell and whether Cell Ranger reports the same contig, the barcode's other kept chains and the
contig's V / C genes. Then the same per-contig features for all kept contigs on lineage-consistent barcodes, for comparison.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vdj"))
import compare_vdj as CV  # noqa: E402


def tf(x):
    return str(x).strip().lower() in ("true", "1")


runs, data, sdir = sys.argv[1:4]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
allw = []
for spec in sys.argv[4:]:
    lib, ch = spec.split(":")
    keep = CV.TCR if ch == "TCR" else CV.BCR
    fca = os.path.join(data, lib, "filtered_contig_annotations.csv")
    ref = CV.load_cellranger(fca)
    rk = set(zip(ref.barcode, ref.chain, ref.cdr3_nt))
    rcells = CV.cellranger_cells(fca)
    r = os.path.join(runs, lib)
    gex = set(l.strip() for l in open(os.path.join(r, "gex_cells.txt")))
    lab = dict(pd.read_csv(os.path.join(r, "gex_lineage.csv"))[["barcode", "lineage"]].values)
    d = pd.read_csv(os.path.join(r, sdir, "contigs.csv"))
    for c in ("kept", "productive"):
        d[c] = d[c].map(tf)
    d = d[d.chain.isin(keep)]
    d["cdr3_nt"] = d["cdr3_nt"].astype(str)
    d["jmax"] = d.groupby(["chain", "cdr3_nt"])["umis"].transform("max")
    d["n_bc"] = d.groupby(["chain", "cdr3_nt"])["barcode"].transform("nunique")
    k = d[d.kept & d.productive & d.barcode.isin(gex)].copy()
    k["lineage"] = k.barcode.map(lab)
    k["wrong"] = (k.chain.isin(CV.TCR) & (k.lineage == "B")) | (k.chain.isin(CV.BCR) & (k.lineage == "T"))
    k["cr_cell"] = k.barcode.isin(rcells)
    k["cr_contig"] = [x in rk for x in zip(k.barcode, k.chain, k.cdr3_nt)]
    k["bc_umis"] = k.groupby("barcode")["umis"].transform("sum")
    k["bc_chains"] = k.groupby("barcode")["chain"].transform(lambda s: "+".join(sorted(s)))
    k["rpu"] = k.reads / k.umis.clip(lower=1)
    k["lib"] = lib
    w = k[k.wrong]
    ok = k[~k.wrong & k.lineage.isin(["T", "B"])]
    print(f"===== {lib}: wrong {len(w)} of {int(k.lineage.isin(['T', 'B']).sum())} lineage-known kept contigs")
    print(w[["barcode", "chain", "umis", "junction_umis", "reads", "jmax", "n_bc", "cr_cell", "cr_contig", "bc_chains", "bc_umis",
             "v_gene", "c_gene"]].to_string(index=False))
    q = [.05, .25, .5]
    print("  wrong   umis q", w.umis.quantile(q).tolist(), "rpu q", w.rpu.quantile(q).round(0).tolist(), "jmax/umis med",
          float((w.jmax / w.umis).median()) if len(w) else None)
    print("  correct umis q", ok.umis.quantile(q).tolist(), "rpu q", ok.rpu.quantile(q).round(0).tolist(), "jmax/umis med",
          float((ok.jmax / ok.umis).median()))
    print("  correct contigs with umis <= 2:", int((ok.umis <= 2).sum()), "of", len(ok), "; of them CR-confirmed",
          int(((ok.umis <= 2) & ok.cr_contig).sum()))
    allw.append(w)
w = pd.concat(allw)
print("ALL wrong", len(w), "cr_contig", int(w.cr_contig.sum()), "cr_cell", int(w.cr_cell.sum()), "umis<=2", int((w.umis <= 2).sum()),
      "single-chain barcodes", int((~w.bc_chains.str.contains(r"\+")).sum()))
