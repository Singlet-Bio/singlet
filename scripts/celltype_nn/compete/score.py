#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Score every method of the cell type comparison the same way, from per-cell call files.

Input: ``<compete>/calls/<method>/<GSE>/<GSM>.{parquet,csv}`` (barcode, pred_cl, score[, prob, pruned]) and
``timings.jsonl``, plus the queries' truth (``query/<GSE>/<GSM>.cells.csv``). A method that abstains writes a null ``pred_cl``.

Per method:

* accuracy at Cell Ontology depth 1-4 on the guide's tree and the exact / coarse / finer / wrong / unassigned fractions
  (``singlet.celllabels.evaluate``; each study pools its samples, then the mean over studies), and CL-level consistency;
* calibration, for methods whose score is a probability: ECE of that probability against "the truth lies in the predicted
  term's subtree" (predictions finer than the truth cannot be checked and are left out), 10 equal-width bins, pooled cells;
* abstention: wrong-call rate when only the most confident fraction of cells is kept (coverage 1.0, 0.8, 0.5) and the area
  under that risk-coverage curve (AURC; lower is better); for SingleR also its own pruning;
* cross-tissue generalisation: the same metrics split by whether the study's tissue is well represented in the training
  reference (at least ``--seen-cells`` corpus cells in the tissue group) or not;
* runtime: CPU-seconds (wall x threads) per million query cells; training time is reported separately.

  python scripts/celltype_nn/compete/score.py --compete $D/compete --data $D --out results.json [--methods a,b]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "python"))

COVERAGES = (1.0, 0.8, 0.5)


def _read_calls(p: Path):
    import pandas as pd

    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, dtype={"pred_cl": object, "pruned": object})
    df["pred_cl"] = df["pred_cl"].where(df["pred_cl"].astype(str).str.startswith("CL:"), None)
    if "pruned" in df:
        df["pruned"] = df["pruned"].where(df["pruned"].astype(str).str.startswith("CL:"), None)
    return df


def _subtree_correct(onto, pred, truth):
    out = np.full(len(pred), np.nan)
    for i, (p, t) in enumerate(zip(pred, truth)):
        if not isinstance(p, str):
            continue
        if p in onto.ancestors(t):
            out[i] = 1.0
        elif t not in onto.ancestors(p):
            out[i] = 0.0
    return out


def _ece(prob, correct):
    m = ~np.isnan(correct) & ~np.isnan(prob)
    p, c = prob[m], correct[m]
    if len(p) == 0:
        return None, []
    edges = np.linspace(0, 1, 11)
    e, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if s.any():
            rows.append(dict(lo=float(lo), hi=float(hi), n=int(s.sum()), mean_prob=float(p[s].mean()), accuracy=float(c[s].mean())))
            e += s.mean() * abs(p[s].mean() - c[s].mean())
    return float(e), rows


def _risk_coverage(score, wrong):
    """wrong-call rate among the most confident cells at each coverage, and the area under the curve."""
    o = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")
    w = wrong[o].astype(float)
    cum = np.cumsum(w) / np.arange(1, len(w) + 1)
    at = {f"{c:g}": float(cum[max(int(round(c * len(w))) - 1, 0)]) for c in COVERAGES}
    return at, float(cum.mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compete", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="corpus dir (tissue representation of the reference)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--methods", default="")
    ap.add_argument("--seen-cells", type=int, default=20000)
    a = ap.parse_args(argv)

    import pandas as pd
    from singlet.celllabels import evaluate as EV
    from singlet.celllabels.guide import Guide, ontology, tissue_groups
    from singlet.celltype_nn.ontology import is_consistent

    onto = ontology()
    samples = pd.read_csv(a.compete / "query" / "samples.csv")
    corpus = pd.read_parquet(a.data / "corpus_obs.parquet", columns=["tissue_general", "split"])
    grp_count = defaultdict(int)
    for t, n in corpus[corpus["split"] == "train"]["tissue_general"].value_counts().items():
        for g in tissue_groups(str(t)):
            grp_count[g] += int(n)
    study_seen = {}
    for g, d in samples.groupby("gse"):
        groups = tissue_groups(str(d["cxg_tissue"].iloc[0])) or tissue_groups(str(d["tissue_text"].iloc[0]))
        study_seen[g] = dict(groups=sorted(groups), reference_cells=int(max([grp_count[x] for x in groups], default=0)))
        study_seen[g]["seen"] = study_seen[g]["reference_cells"] >= a.seen_cells
    methods = [m for m in (a.methods.split(",") if a.methods else sorted(p.name for p in (a.compete / "calls").iterdir() if p.is_dir())) if m]
    guides = {sp: Guide([], species=sp) for sp in ("human", "mouse")}
    tms = {sp: EV.TruthMapper(g) for sp, g in guides.items()}
    out = dict(methods={}, studies=study_seen, seen_cells_threshold=a.seen_cells)
    for m in methods:
        per_study = defaultdict(list)
        pooled_prob, pooled_corr, pooled_score, pooled_wrong, pooled_cl = [], [], [], [], []
        pruned_wrong, pruned_cov, missing = [], [], 0
        for r in samples.itertuples():
            f = None
            for ext in (".parquet", ".csv"):
                p = a.compete / "calls" / m / r.gse / f"{r.gsm}{ext}"
                if p.exists():
                    f = p
            if f is None:
                missing += 1
                continue
            c = _read_calls(f)
            q = pd.read_csv(a.compete / "query" / r.gse / f"{r.gsm}.cells.csv", dtype={"truth": str}, keep_default_na=False)
            c = q[["barcode", "truth"]].merge(c, on="barcode", how="left")
            c = c[c["truth"] != ""]
            truth = c["truth"].to_numpy(str)
            pred = c["pred_cl"].to_numpy(object)
            score = pd.to_numeric(c["score"], errors="coerce").to_numpy(float)
            g, tm = guides[r.species], tms[r.species]
            node = np.array([tm(x) if isinstance(x, str) else 0 for x in pred], np.int32)
            rep = EV.report(g, truth, node)
            rep["n_cells"] = int(len(c))
            per_study[r.gse].append(rep)
            cat = np.array([is_consistent(onto, p, t) if isinstance(p, str) else "unassigned" for p, t in zip(pred, truth)])
            pooled_cl.append(cat)
            pooled_wrong.append(cat == "wrong")
            pooled_score.append(score)
            if "prob" in c and pd.to_numeric(c["prob"], errors="coerce").notna().any():
                pooled_prob.append(pd.to_numeric(c["prob"], errors="coerce").to_numpy(float))
                pooled_corr.append(_subtree_correct(onto, pred, truth))
            if "pruned" in c:
                pr = c["pruned"].to_numpy(object)
                keep = np.array([isinstance(x, str) for x in pr])
                pruned_cov.append(keep)
                pruned_wrong.append((cat == "wrong") & keep)
        if not per_study:
            continue
        res = dict(missing_samples=missing)
        res["per_study"] = {}
        for gse, reps in per_study.items():
            n = sum(x["n_testable"] for x in reps)
            row = {k: float(np.average([x["fractions"][k] for x in reps], weights=[x["n_testable"] for x in reps])) for k in
                   ("exact", "coarse", "finer", "wrong", "unassigned")}
            for d in (1, 2, 3, 4):
                xs = [(x["by_depth"][d]["accuracy"], x["by_depth"][d]["n"]) for x in reps if d in x["by_depth"]]
                row[f"d{d}"] = float(np.average([v for v, _ in xs], weights=[k for _, k in xs])) if xs else None
            row["n"] = n
            row["seen_tissue"] = study_seen[gse]["seen"]
            res["per_study"][gse] = row

        def mean_over(rows):
            o = {}
            for k in ("d1", "d2", "d3", "d4", "exact", "coarse", "finer", "wrong", "unassigned"):
                v = [r[k] for r in rows if r[k] is not None]
                o[k] = float(np.mean(v)) if v else None
            o["consistent"] = None if o["exact"] is None else o["exact"] + o["coarse"] + o["finer"]
            o["n_studies"] = len(rows)
            return o

        res["mean_over_studies"] = mean_over(list(res["per_study"].values()))
        res["seen_tissue"] = mean_over([r for r in res["per_study"].values() if r["seen_tissue"]])
        res["unseen_tissue"] = mean_over([r for r in res["per_study"].values() if not r["seen_tissue"]])
        cat = np.concatenate(pooled_cl)
        res["cl_level_pooled"] = {k: float(np.mean(cat == k)) for k in ("exact", "coarse", "finer", "wrong", "unassigned")}
        res["coverage"] = float(np.mean(cat != "unassigned"))
        at, aurc = _risk_coverage(np.concatenate(pooled_score), np.concatenate(pooled_wrong))
        res["risk_at_coverage"], res["aurc"] = at, aurc
        if pooled_prob:
            e, bins = _ece(np.concatenate(pooled_prob), np.concatenate(pooled_corr))
            res["ece"], res["reliability"] = e, bins
        if pruned_cov:
            kc = np.concatenate(pruned_cov)
            res["own_abstention"] = dict(coverage=float(kc.mean()), wrong_among_kept=float(np.concatenate(pruned_wrong).sum() / max(kc.sum(), 1)))
        tf = a.compete / "calls" / m / "timings.jsonl"
        if tf.exists():
            T = [json.loads(x) for x in tf.read_text().splitlines() if x.strip()]
            last = {(t["gse"], t["gsm"]): t for t in T}
            cells = sum(t["n_cells"] for t in last.values())
            res["runtime"] = dict(cpu_seconds_per_million_cells=sum(t["seconds"] * t.get("threads", 1) for t in last.values()) / max(cells, 1) * 1e6,
                                  wall_seconds_per_million_cells=sum(t["seconds"] for t in last.values()) / max(cells, 1) * 1e6,
                                  devices=sorted({t.get("device", "cpu") for t in last.values()}), cells=cells)
        for name in ("train.json",):
            for d in (a.compete / "models" / m,):
                if (d / name).exists():
                    res["training"] = json.loads((d / name).read_text())
        out["methods"][m] = res
        mo = res["mean_over_studies"]
        print(f"{m:16s} d1 {mo['d1']:.3f} d2 {mo['d2']:.3f} d3 {mo['d3']:.3f} d4 {mo['d4'] or 0:.3f} wrong {mo['wrong']:.3f} "
              f"ece {res.get('ece') if res.get('ece') is None else round(res['ece'], 3)} aurc {aurc:.3f} "
              f"cpu-s/M {res.get('runtime', {}).get('cpu_seconds_per_million_cells', float('nan')):.0f} missing {missing}", flush=True)
    a.out.write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
