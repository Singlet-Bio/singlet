"""Collect all per-library CSVs into results/ and print markdown tables.  Usage: aggregate.py [OUTDIR_FOR_RESULTS]
Run in a Slurm job (reads $LQC/ambient_truth/out/*)."""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlib  # noqa: E402

res = sys.argv[1] if len(sys.argv) > 1 else f"{atlib.OUT}/summary"
os.makedirs(res, exist_ok=True)


def cat(pattern):
    fs = sorted(glob.glob(f"{atlib.OUT}/*/{pattern}"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


est = pd.concat([cat("est_*.csv"), cat("ext_*.csv")], ignore_index=True)
cor = pd.concat([cat("corr_*.csv"), cat("extcorr_*.csv")], ignore_index=True)
est.to_csv(f"{res}/estimates_all.csv", index=False)
cor.to_csv(f"{res}/correction_all.csv", index=False)
TUNE = "hgmm1k_v3_singlet"
tune = est[(est.lib == TUNE) & (est.tag == "tune")]
tune.to_csv(f"{res}/tune_grid.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)
if len(tune):
    t = tune.pivot_table(index="method", columns="view", values=["spearman", "ratio"]).round(3)
    print("## tuning grid (hgmm1k_v3_singlet)\n", t.to_string())
