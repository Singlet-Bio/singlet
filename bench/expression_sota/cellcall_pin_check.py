# SPDX-License-Identifier: MIT
"""Is the pin's cell_calls.tsv the same as the call columns embedded in the cellcall_truth tables? Prints the disagreement."""
import sys, pandas as pd
S = "/mnt/projects/debruinz_project/debruinz/singlet/state/cellcall_truth/truth"
PB = "/mnt/projects/debruinz_project/debruinz/singlet/pipeline-benchmark/runs"
for ds, run in (("kang_b2ctrl", "kang2018_ctrl"), ("hgmm1k_v3", "hgmm1k_v3")):
    t = pd.read_csv(f"{S}/{ds}.truth.tsv.gz", sep="\t", dtype={"barcode": str})
    c = pd.read_csv(f"{PB}/{run}/singlet-public/cell_calls.tsv", sep="\t", dtype={"barcode": str})
    print(ds, "truth cols:", list(t.columns))
    t["b"], c["b"] = t["barcode"].str.split("-").str[0], c["barcode"].str.split("-").str[0]
    m = t.merge(c[["b", "called", "called_v2"]], on="b", how="left", suffixes=("", "_pin"))
    for col in ("called_v2", "singlet", "called"):
        if col in t.columns:
            pin = m["called_v2_pin"] if col != "called" else m["called_pin"]
            print(f"  {col}: truth-table {int(t[col].sum())} pin {int(pin.fillna(0).sum())} differ {int((m[col].astype(int) != pin.fillna(0).astype(int)).sum())} of {len(m)}")
