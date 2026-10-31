# SPDX-License-Identifier: MIT
"""Re-score the pin's (d9834b8a) called_v2 and knee call on the bench/cellcall_truth tables (Kang b2 ctrl genotype, hgmm1k_v3 species).
Replaces the called_v2 and knee columns of the stored truth table with the pin's cell_calls.tsv and runs the cellcall_truth scorer."""
import os, sys, pandas as pd
J = "/mnt/projects/debruinz_project/debruinz/singlet/jobs/cellcall_truth"
S = "/mnt/projects/debruinz_project/debruinz/singlet/state/cellcall_truth/truth"
PB = "/mnt/projects/debruinz_project/debruinz/singlet/pipeline-benchmark/runs"
O = "/mnt/projects/debruinz_project/debruinz/singlet/expression-sota/cellcall"
os.makedirs(f"{O}/truth", exist_ok=True)
sys.path.insert(0, J)
import score
for ds, run in (("kang_b2ctrl", "kang2018_ctrl"), ("hgmm1k_v3", "hgmm1k_v3")):
    t = pd.read_csv(f"{S}/{ds}.truth.tsv.gz", sep="\t", dtype={"barcode": str})
    c = pd.read_csv(f"{PB}/{run}/singlet-public/cell_calls.tsv", sep="\t", dtype={"barcode": str}).set_index("barcode")
    c.index = c.index.str.split("-").str[0]
    b = t["barcode"].str.split("-").str[0]
    t["called_v2"] = c["called_v2"].reindex(b).fillna(0).astype(int).to_numpy()
    t["knee"] = c["called"].reindex(b).fillna(0).astype(int).to_numpy()
    t.to_csv(f"{O}/truth/{ds}.truth.tsv.gz", sep="\t", index=False)
score.main(f"{O}/truth", f"{O}/score_pin.json", "kang_b2ctrl", "hgmm1k_v3")
