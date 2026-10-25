"""Run ambient estimators on one library / gene view and score them against truth from a (possibly different) view.
Usage: run_method.py LIB VIEW TRUTHVIEW SPEC [SPEC ...] [--tag T]
SPEC: knn:k=30:tau=10:kappa=0[:latent=pca|vae] | cluster:tau=10:kappa=0[:latent=..] | marker:m=100 | marker:m=all
Rows (score + wall seconds + peak RSS GB) are appended to $OUT/LIB/est_VIEW-TRUTHVIEW.csv; rho vectors saved as npy.
VIEW = full -> scored against full-gene truth (and the correction table is written); A/B = gene-split (estimate on one gene half,
truth from the other half, so estimator and truth noise are independent)."""
import copy
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlib  # noqa: E402
import methods  # noqa: E402


def parse(spec):
    name, *kv = spec.split(":")
    return name, dict(x.split("=") for x in kv)


def main():
    args = sys.argv[1:]
    tag = ""
    if "--tag" in args:
        i = args.index("--tag"); tag = args[i + 1]; del args[i:i + 2]
    lib, view, tview, specs = args[0], args[1], args[2], args[3:]
    with atlib.Timer() as tl:
        ds = atlib.load(lib, view)
        tds = ds if tview == view else atlib.load(lib, tview)
    sc = copy.copy(tds)
    sc.view = f"{view}->{tview}"
    out = f"{atlib.OUT}/{lib}"
    os.makedirs(out, exist_ok=True)
    rows, crows = [], []
    for spec in specs:
        name, p = parse(spec)
        latent = p.get("latent", "pca")
        with atlib.Timer() as t:
            rm = rh = None
            if name in ("knn", "cluster"):
                Z = None
                if latent == "vae":
                    Z = np.load(f"{atlib.DATA}/{lib}/vae_{view}.npy")
                rho, rm, rh = methods.latent_knn(ds, latent=latent, mode=name, k=int(p.get("k", 30)), tau=float(p.get("tau", 10)),
                                                 iters=int(p.get("iters", 3)), kappa=float(p.get("kappa", 0)), Z=Z)
            elif name == "marker":
                rho = methods.marker_oracle(ds, None if p["m"] == "all" else int(p["m"]))
            else:
                raise SystemExit("unknown method " + name)
        r = atlib.est_row(sc, spec, rho)
        r.update(seconds=t.sec, maxrss_gb=t.maxrss_gb, load_seconds=tl.sec, n_cells=len(rho), n_barcodes=ds.X.shape[1], tag=tag)
        rows.append(r)
        np.save(f"{out}/rho_{view}-{tview}_{spec.replace(':', '_').replace('=', '')}.npy", rho)
        if rm is not None and view == tview:
            crows.append(atlib.correction_row(ds, spec, rho, rm, rh))
        print(f"{lib} {view}->{tview} {spec}: spearman {r['spearman']:.3f} pearson {r['pearson']:.3f} ratio {r['ratio']:.2f} "
              f"({t.sec:.1f}s, {t.maxrss_gb:.1f} GB)", flush=True)
    for df, nm in ((pd.DataFrame(rows), f"est_{view}-{tview}.csv"), (pd.DataFrame(crows), f"corr_{view}.csv")):
        if len(df):
            p_ = f"{out}/{nm}"
            df.to_csv(p_, mode="a", header=not os.path.exists(p_), index=False)


if __name__ == "__main__":
    main()
