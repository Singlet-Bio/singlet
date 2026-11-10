"""Aggregate counts-only snp tables: python summarize_snps.py TSV [TSV...] -> concordance / false-alt / true-alt kept per pool and tool."""
import os, sys, pandas as pd, numpy as np
CM = [f"t{t}_c{u}" for t in range(3) for u in range(3)]
for f in sys.argv[1:]:
    d = pd.read_csv(f, sep="\t"); d = d[d.stratum == os.environ.get("STRATUM", "dp10")]
    print("==", f)
    rows = []
    for (pool, tool), g in list(d.groupby(["pool", "tool"])) + [(("ALL", t), x) for t, x in d.groupby("tool")]:
        c = g[CM].sum().values.reshape(3, 3).astype(float)
        called = c[:, 1:].sum() + c[:, 0].sum()
        nr = c[1:].sum()
        rows.append(dict(pool=pool, tool=tool, n=int(c.sum()), conc=c.trace() / c.sum(), false_alt=c[0, 1:].sum() / c[0].sum(),
                         true_alt_kept=(c[1, 1] + c[2, 2]) / nr))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
