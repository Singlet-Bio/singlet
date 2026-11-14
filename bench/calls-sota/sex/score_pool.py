#!/usr/bin/env python3
"""Rescore with the pool check (gate addendum). sexmod.py = call at gate time, sexmod_pool.py = call with the pool check.
Full count matrices are used so the check can read total UMIs and escape genes. POST-HOC for held-out libraries.
usage: score_pool.py name [name...]   (lcl_pool, osorio, schihost, schihost_pr8)
"""
import gzip, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sps

W = Path("/mnt/projects/debruinz_project/debruinz/singlet"); OUT = W / "bench/calls-sota/sex"


def load(fn, tag):
    sp = importlib.util.spec_from_file_location(tag, OUT / fn); m = importlib.util.module_from_spec(sp); sys.modules[tag] = m; sp.loader.exec_module(m); return m


OLD, NEW = load("sexmod.py", "sexold"), load("sexmod_pool.py", "sexnew")
for name in sys.argv[1:]:
    d = OUT / "work/mtx" / name
    X = scipy.io.mmread(gzip.open(d / "matrix.mtx.gz")).T.tocsr()
    genes = (d / "genes.tsv").read_text().split("\n")[: X.shape[1]]
    ev = pd.read_parquet(OUT / "evidence" / f"{name}.parquet")
    grp = ev["demux"].fillna("").astype(str).tolist()
    cx = pd.read_csv(OUT / "work/cellxy" / f"{name}.tsv", sep="\t", keep_default_na=False, na_values=["NA"]) if (OUT / f"work/cellxy/{name}.tsv").exists() else None
    res = {"old": OLD.call_sex(X, genes, species="human", groups=grp), "new": NEW.call_sex(X, genes, species="human", groups=grp)}
    ev["old"], ev["new"] = res["old"].label, res["new"].label
    if cx is not None:
        ev = ev.merge(cx[["col", "pred_qc"]], on="col", how="left"); ev["cellxy"] = ev.pred_qc.str.lower().where(ev.pred_qc.isin(["Female", "Male"]))
    print(f"\n== {name} ({len(ev):,} cells) ==")
    for m in ["old", "new"] + (["cellxy"] if cx is not None else []):
        rows = []
        for u, g in ev.groupby("unit"):
            if len(g) < 200: continue
            c = g[m].to_numpy(); nf, nm = (c == "female").sum(), (c == "male").sum()
            dc = "female" if nf > nm else ("male" if nm > nf else None)
            rows.append((u, (c == g.truth.to_numpy()).mean(), np.isin(c, ["female", "male"]).mean(), dc == g.truth.iloc[0]))
        r = pd.DataFrame(rows, columns=["u", "acc", "cov", "ok"])
        print(f"{m:7s} units {len(r)}  per-cell acc {r.acc.mean():.4f}  call rate {r['cov'].mean():.4f}  donor acc {r.ok.mean():.4f} ({r.ok.sum()}/{len(r)})")
    ch = {k: v for k, v in res["new"].groups.items() if v.get("pool_check")}
    print("pool_check fired for:", {k: v["call"] for k, v in ch.items()})
    bad = ev.groupby("unit").apply(lambda g: (g.new == g.truth).mean()).sort_values().head(3)
    print("lowest per-cell accuracy units (new):", bad.round(3).to_dict())
