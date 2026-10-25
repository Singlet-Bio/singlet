# SPDX-License-Identifier: MIT
"""Compare guide calls of every caller with a reference and measure target knockdown in the expression.

    python -P compare_crispr.py --run OUT --reference protospacer_calls_per_cell.csv --ref-format cellranger
        --gex filtered_feature_bc_matrix.h5 --guides feature_reference.csv --out metrics.json
    python -P compare_crispr.py --run OUT --reference K562_essential_raw_singlecell_01.h5ad --ref-format replogle
        --gemgroup 4 --guides guides.tsv --out metrics.json

Cells: the reference's cells that are also called GEX cells of the run (``assign_*.csv``). Per caller:

* agreement with the reference (:func:`singlet.crispr.assign.agreement`): single-guide recall / precision, assigned
  and multiple-guide rates; with ``--level target`` (Replogle: the reference names the target gene of a guide pair)
  labels are compared as target-gene sets;
* knockdown, which does not depend on any caller's labels: for each targeting guide (or target), 1 - mean CP10k of
  its target in the caller's single-guide cells / mean in cells whose only guide is non-targeting; the median over
  guides with >= 5 cells, and the share of guides whose target is significantly lower (one-sided Mann-Whitney, BH
  0.05);
* cells only one caller assigns: the knockdown in cells Singlet calls and the reference does not (and the reverse)
  says which caller's extra calls are real.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

CALLERS = ("singlet", "cellranger_gmm", "poisson_gaussian")


def ref_cellranger(path):
    d = pd.read_csv(path)
    ids = d["feature_call"].astype(str)
    n = d["num_features"].astype(int)
    return pd.DataFrame({"barcode": d["cell_barcode"].astype(str), "guide_ids": ids, "n_guides": n,
                         "class": np.where(n == 1, "single", "multiple")})


def ref_replogle(path, gemgroup):
    import anndata as ad

    a = ad.read_h5ad(path, backed="r")
    o = a.obs
    gg = o["gem_group"].astype(int) if "gem_group" in o else pd.Series([int(i.split("-")[1]) for i in o.index], o.index)
    o = o[gg == int(gemgroup)]
    bc = [i.split("-")[0] + "-1" for i in o.index]
    gene = o["gene"].astype(str).to_numpy()
    cls = np.where(np.isin(gene, ["non-targeting"]), "single", "single")
    out = pd.DataFrame({"barcode": bc, "guide_ids": o.get("sgID_AB", o["gene"]).astype(str).to_numpy(), "target": gene,
                        "n_guides": 1, "class": cls, "_obs_name": o.index})
    return out, a


def _cp10k_targets(X, genes, targets):
    tot = np.asarray(X.sum(axis=1)).ravel()
    out = {}
    gi = {g: i for i, g in enumerate(genes)}
    for t in set(targets):
        if t in gi:
            col = X[:, gi[t]]
            v = np.asarray(col.todense() if hasattr(col, "todense") else col).ravel().astype(float)
            out[t] = v / np.maximum(tot, 1) * 1e4
    return out


def knockdown_stats(lab: pd.DataFrame, expr: dict, nt_target="non-targeting", key="target", min_cells=5):
    from scipy.stats import mannwhitneyu

    single = lab["class"].to_numpy() == "single"
    tg = lab[key].to_numpy()
    ctrl = single & (lab["kind"].to_numpy() == "non-targeting")
    rows = []
    groups = lab.loc[single & ~ctrl, "guide_ids"].unique()
    for g in groups:
        sel = single & (lab["guide_ids"].to_numpy() == g)
        t = tg[sel][0]
        if t not in expr or sel.sum() < min_cells or ctrl.sum() < min_cells:
            continue
        e = expr[t]
        cm = e[ctrl].mean()
        if cm <= 0:
            continue
        p = mannwhitneyu(e[sel], e[ctrl], alternative="less").pvalue
        rows.append({"guide": g, "target": t, "cells": int(sel.sum()), "knockdown": 1 - e[sel].mean() / cm, "p": p})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"guides_scored": 0}
    from scipy.stats import false_discovery_control

    df["q"] = false_discovery_control(df["p"].to_numpy())
    return df, {"guides_scored": int(len(df)), "median_knockdown": float(df["knockdown"].median()),
                "frac_guides_significant": float((df["q"] < 0.05).mean()), "cells_scored": int(df["cells"].sum())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--ref-format", choices=["cellranger", "replogle"], required=True)
    ap.add_argument("--gemgroup", default=None)
    ap.add_argument("--gex", default=None, help="10x h5 (cellranger format) for expression")
    ap.add_argument("--guides", required=True, help="feature reference CSV or guide table TSV")
    ap.add_argument("--level", choices=["guide", "target"], default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from singlet.crispr import assign as AS
    from singlet.crispr import guides as G

    if a.guides.endswith(".csv"):
        lib = G.from_feature_reference(a.guides)
    else:
        import csv

        with open(a.guides) as f:
            lib = G.from_table(csv.DictReader(f, delimiter="\t"), "table")
    gt = dict(zip(lib.ids, lib.target_gene))
    gk = dict(zip(lib.ids, lib.kind))
    if a.ref_format == "cellranger":
        ref = ref_cellranger(a.reference)
        ref["target"] = ref["guide_ids"].map(lambda s: "|".join(sorted({gt.get(x, x) for x in s.split("|")})))
        adata = None
        level = a.level or "guide"
    else:
        ref, adata = ref_replogle(a.reference, a.gemgroup)
        level = a.level or "target"
    labs = {}
    for c in CALLERS:
        p = os.path.join(a.run, f"assign_{c}.csv")
        if os.path.exists(p):
            labs[c] = pd.read_csv(p, keep_default_na=False)
    base = labs["singlet"]
    cells = sorted(set(ref["barcode"]) & set(base["barcode"]))
    res = {"reference": os.path.basename(a.reference), "level": level, "cells_compared": len(cells),
           "reference_cells": int(len(ref)), "run_called_cells": int(len(base))}
    R = ref.set_index("barcode").loc[cells].reset_index()
    if level == "target":
        R = R.assign(guide_ids=R["target"])
    # expression for knockdown (caller-independent)
    expr, kd_cells = None, None
    if a.gex:
        import scanpy as sc

        g = sc.read_10x_h5(a.gex)
        g.var_names_make_unique()
        g = g[[c for c in base["barcode"] if c in set(g.obs_names)]]
        expr = _cp10k_targets(g.X, list(g.var_names), [t for t in lib.target_gene if t])
        kd_cells = list(g.obs_names)
    elif adata is not None:
        obs_of = dict(zip(ref["barcode"], ref["_obs_name"]))
        kd_cells = [c for c in base["barcode"] if c in obs_of]
        sub = adata[[obs_of[c] for c in kd_cells]].to_memory()
        genes = list(sub.var["gene_name"]) if "gene_name" in sub.var else list(sub.var_names)
        expr = _cp10k_targets(sub.X, genes, [t for t in set(lib.target_gene) if t])
    for c, lab in labs.items():
        L = lab.set_index("barcode")
        Lc = L.loc[cells].reset_index()
        if level == "target":
            Lc = Lc.assign(guide_ids=Lc["target"])
        m = AS.agreement(Lc, R)
        m["assigned_all_called"] = float((L["class"] != "unassigned").mean())
        m["multiple_all_called"] = float((L["class"] == "multiple").mean())
        if expr is not None:
            K = L.loc[kd_cells].reset_index()
            K["kind"] = K["guide_ids"].map(lambda s: gk.get(s, "multiple") if "|" not in s else "multiple")
            K["target"] = K["guide_ids"].map(lambda s: gt.get(s, ""))
            exprK = {t: v for t, v in expr.items()}
            df, kd = knockdown_stats(K, exprK)
            m.update({f"kd_{k}": v for k, v in kd.items()})
            df.to_csv(os.path.join(a.run, f"knockdown_{c}.csv"), index=False)
        res[c] = m
    # discordant cells: Singlet single vs reference unassigned / other caller
    if expr is not None:
        S = labs["singlet"].set_index("barcode").loc[kd_cells]
        for other in [k for k in labs if k != "singlet"]:
            O = labs[other].set_index("barcode").loc[kd_cells]
            for name, a_, b_ in (("singlet_only", S, O), (f"{other}_only", O, S)):
                sel = (a_["class"] == "single") & (b_["class"] == "unassigned")
                K = a_[sel].reset_index()
                if len(K) < 5:
                    res.setdefault("discordant", {})[f"{name}_vs_{other}"] = {"cells": int(len(K))}
                    continue
                Kall = a_.reset_index().copy()
                Kall.loc[~(sel.to_numpy() | ((a_["class"] == "single") & (a_["guide_ids"].map(lambda s: gk.get(s)) == "non-targeting")).to_numpy()), "class"] = "x"
                Kall["kind"] = Kall["guide_ids"].map(lambda s: gk.get(s, "multiple"))
                Kall["target"] = Kall["guide_ids"].map(lambda s: gt.get(s, ""))
                _, kd = knockdown_stats(Kall, expr, min_cells=3)
                res.setdefault("discordant", {})[f"{name}_vs_{other}"] = {"cells": int(sel.sum()), **kd}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
