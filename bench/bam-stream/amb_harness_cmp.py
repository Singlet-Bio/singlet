"""Per-cell rho agreement of amb_harness outputs: stream vs sorted and stream vs 1% read drop, per ensemble size."""
import sys
import numpy as np, pandas as pd
o, sets, ens = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
def load(s, r, m):
    return pd.read_csv(f"{o}/{s}_{r}_e{m}.tsv", sep="\t").set_index("barcode")
for s in sets:
    for m in ens:
        a = load(s, "runs_r14", m)
        for r, lab in (("runs_r14s", "sorted"), ("runs_r14d", "drop1")):
            b = load(s, r, m)
            i = a.index.intersection(b.index)
            x, y = a.loc[i, "rho"], b.loc[i, "rho"]
            d = (x - y).abs()
            print(f"{s} ensemble={m:>3} stream_vs_{lab:<6} cells={len(i)} pearson={np.corrcoef(x, y)[0, 1]:.4f} "
                  f"absdiff_q50={d.quantile(.5):.4f} q90={d.quantile(.9):.4f} q99={d.quantile(.99):.4f} mean_rho={x.mean():.4f}/{y.mean():.4f}")
