"""Score external-tool outputs (production harness, DecontX) for LIB VIEW TRUTHVIEW.  Usage: score_ext.py LIB VIEW TRUTHVIEW"""
import copy
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlib  # noqa: E402

lib, view, tview = sys.argv[1:4]
ds = atlib.load(lib, view)
tds = ds if tview == view else atlib.load(lib, tview)
sc = copy.copy(tds); sc.view = f"{view}->{tview}"
d = f"{atlib.LQC}/ambient_truth/work/{lib}_{view}"
out = f"{atlib.OUT}/{lib}"
os.makedirs(out, exist_ok=True)
rows, crows = [], []
for th in (1, 16):
    p = f"{d}/prod_t{th}"
    if not os.path.exists(f"{p}/rho.f64"):
        continue
    rho, Cc = atlib.load_harness_out(ds, p)
    el, kb = open(f"{p}/time.txt").read().split()[:2]
    r = atlib.est_row(sc, f"production_t{th}", rho)
    r.update(seconds=float(el), maxrss_gb=float(kb) / 1e6, n_cells=len(rho), n_barcodes=ds.X.shape[1])
    rows.append(r)
    np.save(f"{out}/rho_{view}-{tview}_production_t{th}.npy", rho)
    if view == tview and th == 1:
        rm, rh = atlib.removed_by_species(ds, Cc)
        crows.append(atlib.correction_row(ds, "production", rho, rm, rh))
for v in ("default", "background"):
    p = f"{d}/decontx_{v}.csv"
    if not os.path.exists(p):
        continue
    t = pd.read_csv(p)
    r = atlib.est_row(sc, f"decontx_{v}", t.rho.values)
    r.update(seconds=float(t.seconds[0]), maxrss_gb=np.nan, n_cells=len(t), n_barcodes=ds.X.shape[1])
    rows.append(r)
    np.save(f"{out}/rho_{view}-{tview}_decontx_{v}.npy", t.rho.values)
    if view == tview:
        crows.append(atlib.correction_row(ds, f"decontx_{v}", t.rho.values, t.rem_m.values, t.rem_h.values))
for df, nm in ((pd.DataFrame(rows), f"ext_{view}-{tview}.csv"), (pd.DataFrame(crows), f"extcorr_{view}.csv")):
    if len(df):
        q = f"{out}/{nm}"
        df.to_csv(q, mode="a", header=not os.path.exists(q), index=False)
        print(df.round(3).to_string())
