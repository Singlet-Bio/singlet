#!/usr/bin/env python3
"""Score the sex call against the gate in bench/calls-sota/gates/sex.md.

    python score.py --sets lcl_pool,osorio,geo --tag dev      (development)
    python score.py --sets schihost --tag heldout             (held-out, run once)

Inputs (all under bench/calls-sota/sex/): evidence/<name>.parquet (tier 1), work/cellxy/<name>.tsv (cellXY calls), and for the
GEO-metadata tier the stored per-cell evidence of the earlier public benchmark.
Methods: singlet_dep (donor groups where demultiplexing exists), singlet_ung (whole library one group), rule2 (two-gene rule on
XIST and the non-PAR chrY genes Singlet uses), cellxy (classifySex defaults).
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

W = Path("/mnt/projects/debruinz_project/debruinz/singlet")
OUT = W / "bench/calls-sota/sex"
MIN_CELLS = 200
METHODS = ["singlet_dep", "singlet_ung", "rule2", "cellxy"]


def load_sex(path=None):
    spec = importlib.util.spec_from_file_location("sexmod", path or OUT / "sexmod.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["sexmod"] = m
    spec.loader.exec_module(m)
    return m


def singlet_calls(SX, x, y, groups):
    X = sps.csr_matrix(np.column_stack([x, y]).astype(np.float64))
    r = SX.call_sex(X, ["XIST", "RPS4Y1"], species="human", groups=groups)
    return np.asarray(r.label, dtype=object)


def rule2(x, y):
    return np.where((x > 0) & (y == 0), "female", np.where((y > 0) & (x == 0), "male", None)).astype(object)


def add_methods(SX, df, timing):
    """df has x, y, demux ('' = none), sample; adds singlet_dep, singlet_ung, rule2 (cellxy is added by the caller)."""
    dep, ung = np.empty(len(df), dtype=object), np.empty(len(df), dtype=object)
    xs, ys, ds = df["x"].to_numpy(), df["y"].to_numpy(), df["demux"].to_numpy()
    for _, idx in df.groupby("sample", sort=False).indices.items():
        d = ds[idx]
        grp = None if not (d != "").any() else list(d)
        t = time.perf_counter()
        dep[idx] = singlet_calls(SX, xs[idx], ys[idx], grp)
        timing.append((len(idx), time.perf_counter() - t))
        ung[idx] = singlet_calls(SX, xs[idx], ys[idx], None)
    df["singlet_dep"], df["singlet_ung"] = dep, ung
    df["rule2"] = rule2(xs, ys)
    return df


def unit_table(df, m):
    rows = []
    for (ds, unit), g in df.groupby(["dataset", "unit"], sort=False):
        t = g["truth"].to_numpy()
        c = g[m].to_numpy()
        called = np.isin(c, ["female", "male"])
        nf, nm = int((c == "female").sum()), int((c == "male").sum())
        dc = "female" if nf > nm else ("male" if nm > nf else None)
        rows.append(dict(dataset=ds, unit=unit, n=len(g), truth=t[0], acc=float((c == t).mean()), cov=float(called.mean()),
                         prec=float((c[called] == t[called]).mean()) if called.any() else np.nan,
                         donor_call=dc, donor_ok=bool(dc == t[0]), donor_abstain=dc is None))
    return pd.DataFrame(rows)


def boot_diff(a, b, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 2:
        return (float("nan"), float("nan"))
    s = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def summarize(df, label):
    res, tabs = {}, {}
    lines = [f"### {label}", "",
             "| method | units (n>=200) | per-cell acc (macro) | call rate | precision | donor acc | donor abstain | pooled-cell acc |",
             "|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        u = unit_table(df, m)
        tabs[m] = u
        big = u[u.n >= MIN_CELLS]
        r = dict(units=int(len(big)), acc=float(big.acc.mean()), cov=float(big["cov"].mean()), prec=float(big.prec.mean()),
                 donor_acc=float(big.donor_ok.mean()), donor_abstain=float(big.donor_abstain.mean()),
                 pooled_acc=float((df[m].to_numpy() == df["truth"].to_numpy()).mean()))
        res[m] = r
        lines.append(f"| {m} | {r['units']} | {r['acc']:.4f} | {r['cov']:.4f} | {r['prec']:.4f} | {r['donor_acc']:.4f} | "
                     f"{r['donor_abstain']:.4f} | {r['pooled_acc']:.4f} |")
    for mine in ("singlet_dep", "singlet_ung"):
        base = tabs[mine][tabs[mine].n >= MIN_CELLS].set_index(["dataset", "unit"])
        for comp in ("rule2", "cellxy"):
            c = tabs[comp][tabs[comp].n >= MIN_CELLS].set_index(["dataset", "unit"])
            j = base.join(c, lsuffix="_s", rsuffix="_c", how="inner")
            lo, hi = boot_diff(j.acc_s, j.acc_c)
            res[f"diff_{mine}_minus_{comp}"] = dict(mean=float((j.acc_s - j.acc_c).mean()), ci=[lo, hi])
            lines.append(f"\n{mine} minus {comp}, per-cell accuracy over units: {float((j.acc_s - j.acc_c).mean()):+.4f} "
                         f"(95% bootstrap CI {lo:+.4f} to {hi:+.4f}).")
    lines.append("")
    return res, tabs, "\n".join(lines)


def load_tier2(SX, timing):
    c = pd.read_parquet(W / "bpub/artifacts/chromosomal-sex/sex_cells.parquet")
    s = pd.read_csv(W / "bpub/artifacts/chromosomal-sex/sex_samples.tsv", sep="\t")
    truth = dict(zip(s.gsm, s.truth))
    df = pd.DataFrame({"dataset": "geo_metadata", "tier": 2, "sample": c.gsm, "unit": c.gsm, "truth": c.gsm.map(truth),
                       "x": c.xist.to_numpy(float), "y": c.y_singlet.to_numpy(float), "demux": c.donor.fillna("").astype(str),
                       "stored": c.singlet_label, "cellxy": c.cellxy_qc.where(c.cellxy_qc.isin(["female", "male"]))})
    df = df[df.truth.isin(["female", "male"])].reset_index(drop=True)
    df = add_methods(SX, df, timing)
    print("tier2 stored vs recomputed (deployed) label agreement:",
          float((df.stored.astype(str) == df.singlet_dep.astype(str)).mean()), flush=True)
    return df


def load_tier1(SX, name, timing):
    e = pd.read_parquet(OUT / "evidence" / f"{name}.parquet")
    e["sample"] = name
    cx = pd.read_csv(OUT / "work/cellxy" / f"{name}.tsv", sep="\t", keep_default_na=False, na_values=["NA"])
    e = e.merge(cx[["col", "pred_qc"]], on="col", how="left")
    e["cellxy"] = e["pred_qc"].str.lower().where(e["pred_qc"].isin(["Female", "Male"]))
    e["demux"] = e["demux"].fillna("").astype(str)
    e["x"] = e["x"].astype(float)
    e["y"] = e["y"].astype(float)
    e["dataset"] = name
    return add_methods(SX, e, timing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sexmod", default=None)
    a = ap.parse_args()
    SX = load_sex(a.sexmod)
    timing, out_md, out_js, frames = [], [], {}, []
    names = a.sets.split(",")
    for name in names:
        df = load_tier2(SX, timing) if name == "geo" else load_tier1(SX, name, timing)
        frames.append(df)
        res, tabs, md = summarize(df, f"{name} (tier {int(df.tier.iloc[0])}, {len(df):,} cells)")
        out_md.append(md)
        out_js[name] = res
        allu = pd.concat([tabs[m].assign(method=m) for m in METHODS])
        allu.to_csv(OUT / f"units_{a.tag}_{name}.tsv", sep="\t", index=False)
        if name != "geo":
            out_md.append("```\n" + allu.to_string(index=False) + "\n```")
    t1 = [f for f in frames if int(f.tier.iloc[0]) == 1]
    if len(t1) > 1:
        res, _, md = summarize(pd.concat(t1, ignore_index=True), "tier 1 combined")
        out_md.insert(0, md)
        out_js["tier1_combined"] = res
    n = sum(k for k, _ in timing)
    s = sum(t for _, t in timing)
    out_js["runtime"] = dict(cells=int(n), seconds=float(s), core_minutes_per_10k_cells=float(s / 60 / n * 1e4))
    out_md.append(f"Runtime of the Singlet sex call (deployed mode, one core, evidence already extracted): {s:.1f} s for {n:,} "
                  f"cells = {s / 60 / n * 1e4:.4f} core-minutes per 10,000 cells.")
    (OUT / f"score_{a.tag}.md").write_text("\n\n".join(out_md) + "\n")
    (OUT / f"score_{a.tag}.json").write_text(json.dumps(out_js, indent=1))
    print("\n\n".join(out_md))


if __name__ == "__main__":
    main()
