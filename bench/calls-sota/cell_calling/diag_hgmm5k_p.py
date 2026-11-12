# SPDX-License-Identifier: MIT
"""Post-hoc diagnostic: p-values of the barcodes EmptyDrops_CR calls and Singlet does not (hgmm5k_v3). python -P diag_hgmm5k_p.py TRUTH CV2_TSV"""
import sys
import numpy as np, pandas as pd
t = pd.read_csv(sys.argv[1], sep="\t", dtype={"barcode": str}, low_memory=False)
c = pd.read_csv(sys.argv[2], sep="\t", dtype={"barcode": str})[["barcode", "pvalue", "fdr"]]
u = t.merge(c, on="barcode")
m = u[(u.edcr == 1) & (u.called_v2 == 0)]
print("n", len(m), "tested (p not NaN)", int(m.pvalue.notna().sum()))
print("pvalue quantiles", m.pvalue.quantile([0, .1, .5, .9, 1]).to_dict())
print("fdr quantiles", m.fdr.quantile([0, .1, .5, .9, 1]).to_dict())
print("p == 1e-4 floor share", float((m.pvalue < 1.0001e-4).mean()))
a = u[(u.called_v2 == 0) & (u.umi_total >= 500) & (u.umi_total < 5000)]
print("all uncalled 500-5000 UMI:", len(a), "tested", int(a.pvalue.notna().sum()), "fdr<=0.001", int((a.fdr <= 0.001).sum()), "fdr<=0.01", int((a.fdr <= 0.01).sum()), "fdr<=0.05", int((a.fdr <= 0.05).sum()))
tested = u[u.pvalue.notna()]
print("tested", len(tested), "p at floor", int((tested.pvalue < 1.0001e-4).sum()), "p<1e-3", int((tested.pvalue < 1e-3).sum()))
print(m[["umi_total", "pvalue", "fdr", "n_human", "n_mouse", "human_share", "truth"]].head(12).to_string())
