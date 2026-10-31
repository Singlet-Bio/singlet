# SPDX-License-Identifier: MIT
"""Median intronic fraction (cell_qc_metrics.tsv intronic_pct / 100) of the called_v2 cells of each run. python intronic_median.py NAME=RUN ..."""
import sys, pandas as pd
for arg in sys.argv[1:]:
    n, r = arg.split("=", 1)
    q = pd.read_csv(f"{r}/cell_qc_metrics.tsv", sep="\t", usecols=["barcode", "intronic_pct", "n_umi"])
    print(n, "n", len(q), "median intronic fraction %.3f" % (q.intronic_pct.median() / 100), "UMI-weighted %.3f" % ((q.intronic_pct * q.n_umi).sum() / q.n_umi.sum() / 100), flush=True)
