# SPDX-License-Identifier: MIT
"""Per-barcode read-level features of ambient-pool barcodes (uncalled, 1-100 UMI) against called cells, from cell_qc_metrics.tsv.
  python soup_features.py OUT.json NAME=RUN_DIR ...
Effect size = AUC (Mann-Whitney) of cells vs pool barcodes: 0.5 = no difference. Features are per-read fractions (n_reads) or the file's own rates."""
import json, sys, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
out = sys.argv[1]; res = {}
bare = lambda b: b.split("-")[0]
for arg in sys.argv[2:]:
    name, run = arg.split("=", 1)
    q = pd.read_csv(f"{run}/cell_qc_metrics.tsv", sep="\t", dtype={"barcode": str}); q["b"] = q.barcode.map(bare)
    cc = pd.read_csv(f"{run}/cell_calls.tsv", sep="\t", dtype={"barcode": str}); v2 = set(cc.barcode[cc.called_v2 == 1].map(bare))
    q["cell"] = q.b.isin(v2)
    pool = q[(~q.cell) & (q.n_umi >= 1) & (q.n_umi <= 100)]; cells = q[q.cell & (q.n_umi >= 500)]
    f = pd.DataFrame(index=q.index)
    nr = q.n_reads.clip(lower=1)
    for c in ("exonic_reads", "intronic_reads", "antisense_reads", "mito_reads", "intergenic_reads", "unmapped_reads", "multimapped_reads", "star_multimapped_reads"):
        if c in q: f[c + "_frac"] = q[c] / nr
    for c in ("pct_mt", "pct_ribo", "intronic_pct", "duplication_rate", "saturation", "mismatch_rate"):
        if c in q: f[c] = q[c]
    f["reads_per_umi"] = q.n_reads / q.n_umi.clip(lower=1)
    f["log_genes_per_umi"] = np.log1p(q.n_genes) - np.log1p(q.n_umi)
    r = {"n_pool": int(len(pool)), "n_cells": int(len(cells)), "features": {}}
    for c in f.columns:
        a, b = f.loc[cells.index, c].dropna(), f.loc[pool.index, c].dropna()
        if len(a) < 30 or len(b) < 30 or a.nunique() < 2 and b.nunique() < 2: continue
        auc = float(mannwhitneyu(a, b).statistic / (len(a) * len(b)))
        r["features"][c] = {"cells_median": float(a.median()), "pool_median": float(b.median()), "auc_cells_vs_pool": auc}
    res[name] = r
    print(name, r["n_pool"], r["n_cells"], {k: (round(v["cells_median"], 3), round(v["pool_median"], 3), round(v["auc_cells_vs_pool"], 2)) for k, v in r["features"].items()}, flush=True)
json.dump(res, open(out, "w"), indent=1)
