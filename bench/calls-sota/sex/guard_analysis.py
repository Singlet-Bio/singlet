#!/usr/bin/env python3
"""Runtime of the sex call, and the evidence-adequacy statistic behind the held-out miss (post-hoc, not a gate number).

Part 1: time evidence extraction (XIST and chrY columns of a gene x cell matrix) and the call on the lcl_pool library.
Part 2: per unit, the fraction of cells with any sex-informative UMI (XIST or chrY). Development units first; the held-out
units are printed after, labelled post-hoc.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import gzip

# sexmod.py in this directory on Clipper is a copy of python/singlet/celllabels/sex.py (origin/main 4227560a)
W = Path("/mnt/projects/debruinz_project/debruinz/singlet")
OUT = W / "bench/calls-sota/sex"
spec = importlib.util.spec_from_file_location("sexmod", OUT / "sexmod.py")
SX = importlib.util.module_from_spec(spec)
sys.modules["sexmod"] = SX
spec.loader.exec_module(SX)

f = W / "bpub/runs/lcl_gse230239/singlet/filtered_feature_bc_matrix"
M = scipy.io.mmread(gzip.open(f / "matrix.mtx.gz")).tocsr()
names = pd.read_csv(f / "features.tsv.gz", header=None, sep="\t")[1].astype(str).tolist()
X = M.T.tocsr() if M.shape[0] == len(names) else M
t0 = time.perf_counter()
r = SX.call_sex(X, names, species="human")
dt = time.perf_counter() - t0
print(f"timing: evidence extraction + call on {X.shape[0]:,} cells x {X.shape[1]:,} genes, one core: {dt:.2f} s "
      f"= {dt / 60 / X.shape[0] * 1e4:.5f} core-minutes per 10,000 cells", flush=True)


def inf_table(df, unit, label):
    g = df.groupby(unit).agg(truth=("truth", "first"), n=("x", "size"),
                             inf_frac=("inf", "mean"), y_frac=("y1", "mean"), x_frac=("x1", "mean"))
    g = g[g.n >= 200]
    print(f"\n{label}: units with >= 200 cells = {len(g)}")
    for tr in ("female", "male"):
        s = g[g.truth == tr].inf_frac
        if len(s):
            print(f"  {tr}: informative-cell fraction min {s.min():.3f}, 5% {s.quantile(.05):.3f}, median {s.median():.3f}, "
                  f"max {s.max():.3f}; units below 0.25: {(s < .25).sum()}, below 0.15: {(s < .15).sum()}")
    return g


def prep(df):
    df = df.copy()
    df["inf"] = (df.x + df.y) > 0
    df["x1"] = df.x > 0
    df["y1"] = df.y > 0
    return df


# development: tier 1 and tier 2
dev = []
for nm in ("lcl_pool", "osorio"):
    e = pd.read_parquet(OUT / "evidence" / f"{nm}.parquet")
    e["x"] = e.x.astype(float)
    e["y"] = e.y.astype(float)
    dev.append(e[["unit", "truth", "x", "y"]].assign(unit=lambda d: nm + ":" + d.unit.astype(str)))
c = pd.read_parquet(W / "bpub/artifacts/chromosomal-sex/sex_cells.parquet")
s = pd.read_csv(W / "bpub/artifacts/chromosomal-sex/sex_samples.tsv", sep="\t")
t2 = pd.DataFrame({"unit": c.gsm, "truth": c.gsm.map(dict(zip(s.gsm, s.truth))), "x": c.xist.astype(float), "y": c.y_singlet.astype(float)})
g1 = inf_table(prep(pd.concat(dev)), "unit", "development tier 1")
g2 = inf_table(prep(t2), "unit", "development tier 2 (GEO metadata)")
print("\ntier 2 units with the lowest informative fraction:")
print(g2.sort_values("inf_frac").head(12).to_string())
ho = pd.read_parquet(OUT / "evidence/schihost.parquet")
ho["x"] = ho.x.astype(float)
ho["y"] = ho.y.astype(float)
g3 = inf_table(prep(ho), "unit", "held-out, POST-HOC (seen after the gate was scored)")
print(g3.sort_values("inf_frac").head(4).to_string())
