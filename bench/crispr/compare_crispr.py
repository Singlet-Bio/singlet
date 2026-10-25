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

CALLERS = ("singlet", "cellranger_gmm", "poisson_gaussian")  # assign_<caller>.csv of the run


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
        if t not in expr or sel.sum() < min_cells:
            continue
        e = expr[t]
        # control: non-targeting cells; a library without named non-targeting guides (Replogle 2022: they have no
        # coordinates) uses the single-guide cells of every other target, the usual control of an essential-gene screen
        c_ = ctrl if ctrl.sum() >= 20 else (single & (tg != t) & (tg != ""))
        cm = e[c_].mean()
        if cm <= 0:
            continue
        p = mannwhitneyu(e[sel], e[c_], alternative="less").pvalue
        rows.append({"guide": g, "target": t, "cells": int(sel.sum()), "knockdown": 1 - e[sel].mean() / cm, "p": p})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"guides_scored": 0}
    from scipy.stats import false_discovery_control

    df["q"] = false_discovery_control(df["p"].to_numpy())
    return df, {"guides_scored": int(len(df)), "median_knockdown": float(df["knockdown"].median()),
                "frac_guides_significant": float((df["q"] < 0.05).mean()), "cells_scored": int(df["cells"].sum())}


def counting_concordance(cr_h5, counts, barcodes):
    """Our guide UMIs vs Cell Ranger's on the same cells and guides."""
    import h5py
    from scipy import sparse

    with h5py.File(cr_h5, "r") as f:
        m = f["matrix"]
        ft = np.array([x.decode() for x in m["features"]["feature_type"][:]])
        ids = np.array([x.decode() for x in m["features"]["id"][:]])
        bc = np.array([x.decode() for x in m["barcodes"][:]])
        A = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:]))
    rows = np.flatnonzero(ft == "CRISPR Guide Capture")
    X, feats = counts
    fi = {f: i for i, f in enumerate(feats)}
    bi = {b: i for i, b in enumerate(bc)}
    keep_r = [r for r in rows if ids[r] in fi]
    keep_c = [i for i, b in enumerate(barcodes) if b in bi]
    C = A[keep_r][:, [bi[barcodes[i]] for i in keep_c]].toarray().astype(float)
    O = X[[fi[ids[r]] for r in keep_r]][:, keep_c].toarray().astype(float)
    nz = (C > 0) | (O > 0)
    return {"guides": len(keep_r), "cells": len(keep_c), "umi_ratio": float(O.sum() / max(1, C.sum())),
            "pearson_log1p": float(np.corrcoef(np.log1p(O[nz]), np.log1p(C[nz]))[0, 1]) if nz.sum() > 2 else float("nan"),
            "entries_identical": float((O[nz] == C[nz]).mean()) if nz.any() else float("nan"),
            "per_cell_total_ratio_median": float(np.median(O.sum(0)[C.sum(0) > 0] / C.sum(0)[C.sum(0) > 0]))}


def multiplet_check(S, P, expr, gt, gk, cells):
    """Cells Singlet calls single (guide g) and Cell Ranger multiple (g + others): the expression of g's target and of
    the other guides' targets, each relative to non-targeting cells (1 = no knockdown). Real multiplets knock down
    both; an ambient extra guide leaves its target untouched."""
    S = S.set_index("barcode").loc[cells]
    P = P.set_index("barcode").loc[cells]
    ctrl = ((S["class"] == "single") & (S["guide_ids"].map(lambda x: gk.get(x)) == "non-targeting")).to_numpy()
    ref = {t: v[ctrl].mean() for t, v in expr.items() if ctrl.sum() >= 5 and v[ctrl].mean() > 0}
    pos = {c: i for i, c in enumerate(cells)}
    kept, extra, clean = [], [], []
    for c, sg, sc, pg, pc in zip(S.index, S["guide_ids"], S["class"], P["guide_ids"], P["class"]):
        if sc != "single" or gk.get(sg) != "targeting":
            continue
        t = gt.get(sg)
        if t not in ref:
            continue
        i = pos[c]
        if pc == "single" and pg == sg:
            clean.append(expr[t][i] / ref[t])
        elif pc == "multiple" and sg in pg.split("|"):
            kept.append(expr[t][i] / ref[t])
            for h in pg.split("|"):
                th = gt.get(h)
                if h != sg and gk.get(h) == "targeting" and th in ref and th != t:
                    extra.append(expr[th][i] / ref[th])
    f = lambda v: {"n": len(v), "mean_rel_expr": float(np.mean(v)) if v else float("nan"),
                   "median_rel_expr": float(np.median(v)) if v else float("nan")}
    return {"singlet_single_ref_single": f(clean), "singlet_guide_in_ref_multiple": f(kept),
            "ref_extra_guides_in_those_cells": f(extra)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--ref-format", choices=["cellranger", "replogle"], required=True)
    ap.add_argument("--gemgroup", default=None)
    ap.add_argument("--gex", default=None, help="10x h5 (cellranger format) for expression")
    ap.add_argument("--guides", required=True, help="feature reference CSV or guide table TSV")
    ap.add_argument("--level", choices=["guide", "target"], default=None)
    ap.add_argument("--cr-thresholds", default=None, help="Cell Ranger protospacer_umi_thresholds.json: its rule on our counts")
    ap.add_argument("--cr-h5", default=None, help="Cell Ranger matrix with the CRISPR features: counting concordance")
    ap.add_argument("--columns", default=None, help="columns of counts.npz (one barcode per line)")
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
    if a.ref_format == "cellranger":
        # the published Cell Ranger calls as a caller of their own (cells it does not list are unassigned)
        P = ref.set_index("barcode")
        pub = base[["barcode"]].copy()
        pub["guide_ids"] = pub["barcode"].map(P["guide_ids"]).fillna("")
        pub["n_guides"] = pub["barcode"].map(P["n_guides"]).fillna(0).astype(int)
        pub["class"] = np.where(pub["n_guides"] == 0, "unassigned", np.where(pub["n_guides"] == 1, "single", "multiple"))
        pub["target"] = pub["guide_ids"].map(lambda s: "|".join(sorted({gt.get(x, x) for x in s.split("|")})) if s else "")
        labs["cellranger_published"] = pub
    counts, counting_res = None, None
    if a.columns and os.path.exists(os.path.join(a.run, "counts.npz")):
        from singlet.adt import count as AC

        M, feats, _, _, _ = AC.load_npz(os.path.join(a.run, "counts.npz"))
        cols = [ln.strip() for ln in open(a.columns) if ln.strip()]
        ci = {c: i for i, c in enumerate(cols)}
        sel = [ci[c] for c in base["barcode"]]
        counts = (M[:, sel].tocsc(), feats)
    if a.cr_thresholds and counts is not None:
        thr = json.load(open(a.cr_thresholds))
        X, feats = counts
        t = np.array([thr.get(f, np.inf) for f in feats], dtype=float)
        from scipy import sparse

        Xc = sparse.csr_matrix(X)
        called = sparse.csr_matrix(Xc.multiply(sparse.csr_matrix(Xc >= t[:, None])) > 0)
        lab = AS.summarize(called, feats, [gt.get(f, "") for f in feats])
        lab.insert(0, "barcode", base["barcode"].to_numpy())
        labs["cellranger_thresholds_on_singlet_counts"] = lab
    if a.cr_h5 and counts is not None:
        counting_res = counting_concordance(a.cr_h5, counts, list(base["barcode"]))
    cells = sorted(set(ref["barcode"]) & set(base["barcode"]))
    res = {"reference": os.path.basename(a.reference), "level": level, "cells_compared": len(cells),
           "reference_cells": int(len(ref)), "run_called_cells": int(len(base))}
    if counting_res is not None:
        res["counting_vs_cellranger"] = counting_res
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
            # a dual-guide vector puts two guides of one gene in a cell: compare target sets, single = one target
            nt_ = Lc["target"].map(lambda x: len([y for y in str(x).split("|") if y]))
            Lc = Lc.assign(guide_ids=Lc["target"], **{"class": np.where(nt_ == 0, "unassigned",
                                                                         np.where(nt_ == 1, "single", "multiple"))})
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
    if expr is not None and "cellranger_published" in labs:
        res["multiplet_check"] = multiplet_check(labs["singlet"], labs["cellranger_published"], expr, gt, gk, kd_cells)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
