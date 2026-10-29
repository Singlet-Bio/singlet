# SPDX-License-Identifier: MIT
"""Scoring of per-cell cell type calls against Cell Ontology truth, shared by the comparison harness
(``scripts/celltype_nn/compete/score.py``) and the public benchmark (``bench/public/capabilities/cell_type.py``).

Input: one table of calls (``tool, gse, gsm, barcode, pred_cl, score, prob[, pruned]``; ``pred_cl`` null = no call) and one of
truth (``gse, gsm, barcode, species, truth``). Per tool and study:

* accuracy at depth 1-4 of the guide's tree and the exact / coarse / finer / wrong / unassigned fractions
  (:mod:`singlet.celllabels.evaluate`; a study pools its samples' cells);
* CL-level categories, for abstention (risk-coverage) and calibration;
* calibration of ``prob`` against "the truth lies in the predicted term's subtree" (finer predictions cannot be checked).

Aggregates are means over studies, with bootstrap confidence intervals over studies.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

DEPTHS = (1, 2, 3, 4)
FRACTIONS = ("exact", "coarse", "finer", "wrong", "unassigned")
COVERAGES = (1.0, 0.8, 0.5)


def subtree_correct(onto, pred, truth) -> np.ndarray:
    out = np.full(len(pred), np.nan)
    for i, (p, t) in enumerate(zip(pred, truth)):
        if not isinstance(p, str):
            continue
        if p in onto.ancestors(t):
            out[i] = 1.0
        elif t not in onto.ancestors(p):
            out[i] = 0.0
    return out


def ece(prob: np.ndarray, correct: np.ndarray):
    m = ~np.isnan(correct) & ~np.isnan(prob)
    p, c = prob[m], correct[m]
    if len(p) == 0:
        return None, []
    edges = np.linspace(0, 1, 11)
    e, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if s.any():
            rows.append(
                dict(
                    lo=float(lo),
                    hi=float(hi),
                    n=int(s.sum()),
                    mean_prob=float(p[s].mean()),
                    accuracy=float(c[s].mean()),
                )
            )
            e += s.mean() * abs(p[s].mean() - c[s].mean())
    return float(e), rows


def risk_coverage(score: np.ndarray, wrong: np.ndarray):
    """Wrong-call rate among the most confident cells at each coverage, and the area under the risk-coverage curve."""
    o = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")
    w = wrong[o].astype(float)
    cum = np.cumsum(w) / np.arange(1, len(w) + 1)
    at = {f"{c:g}": float(cum[max(int(round(c * len(w))) - 1, 0)]) for c in COVERAGES}
    return at, float(cum.mean())


def score_tool(calls, truth) -> dict:
    """Per-study metrics and pooled per-cell vectors of one tool. ``calls`` and ``truth`` are DataFrames (see module doc)."""
    import pandas as pd

    from singlet.celllabels import evaluate as EV
    from singlet.celllabels.guide import Guide, ontology

    from .ontology import is_consistent

    onto = ontology()
    guides = {sp: Guide([], species=sp) for sp in ("human", "mouse")}
    tms = {sp: EV.TruthMapper(g) for sp, g in guides.items()}
    df = truth.merge(calls, on=["gse", "gsm", "barcode"], how="left")
    df["pred_cl"] = df["pred_cl"].where(df["pred_cl"].astype(str).str.startswith("CL:"), None)
    per_study, cat_all, score_all, prob_all, corr_all, pruned_kept, gse_all = (
        {},
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for gse, d in df.groupby("gse", sort=True):
        reps = []
        for (gsm, sp), s in d.groupby(["gsm", "species"], sort=False):
            node = np.array(
                [tms[sp](x) if isinstance(x, str) else 0 for x in s["pred_cl"]], np.int32
            )
            rep = EV.report(guides[sp], s["truth"].to_numpy(str), node)
            reps.append(rep)
        row = {
            k: float(
                np.average(
                    [x["fractions"][k] for x in reps], weights=[x["n_testable"] for x in reps]
                )
            )
            for k in FRACTIONS
        }
        for dep in DEPTHS:
            xs = [
                (x["by_depth"][dep]["accuracy"], x["by_depth"][dep]["n"])
                for x in reps
                if dep in x["by_depth"]
            ]
            row[f"d{dep}"] = (
                float(np.average([v for v, _ in xs], weights=[k for _, k in xs])) if xs else None
            )
        row["consistent"] = row["exact"] + row["coarse"] + row["finer"]
        row["n"] = int(sum(x["n_testable"] for x in reps))
        pred = d["pred_cl"].to_numpy(object)
        tr = d["truth"].to_numpy(str)
        cat = np.array(
            [
                is_consistent(onto, p, t) if isinstance(p, str) else "unassigned"
                for p, t in zip(pred, tr)
            ]
        )
        row["cl_wrong"] = float(np.mean(cat == "wrong"))
        row["cl_consistent"] = float(np.mean(np.isin(cat, ["exact", "coarse", "finer"])))
        per_study[gse] = row
        cat_all.append(cat)
        gse_all.append(np.full(len(cat), gse, object))
        score_all.append(pd.to_numeric(d["score"], errors="coerce").to_numpy(float))
        if "prob" in d and pd.to_numeric(d["prob"], errors="coerce").notna().any():
            prob_all.append(pd.to_numeric(d["prob"], errors="coerce").to_numpy(float))
            corr_all.append(subtree_correct(onto, pred, tr))
        if "pruned" in d and d["pruned"].notna().any():
            pruned_kept.append(
                np.array([isinstance(x, str) and x.startswith("CL:") for x in d["pruned"]])
            )
    out = dict(per_study=per_study)
    cat = np.concatenate(cat_all)
    out["coverage"] = float(np.mean(cat != "unassigned"))
    out["risk_at_coverage"], out["aurc"] = risk_coverage(np.concatenate(score_all), cat == "wrong")
    if prob_all:
        out["ece"], out["reliability"] = ece(np.concatenate(prob_all), np.concatenate(corr_all))
    if pruned_kept:
        k = np.concatenate(pruned_kept)
        out["own_abstention"] = dict(
            coverage=float(k.mean()),
            wrong_among_kept=float(((cat == "wrong") & k).sum() / max(k.sum(), 1)),
        )
    return out


def mean_over_studies(
    per_study: Dict[str, dict], key: str, studies: Optional[Iterable[str]] = None
) -> Optional[float]:
    v = [
        r[key]
        for g, r in per_study.items()
        if (studies is None or g in studies) and r.get(key) is not None
    ]
    return float(np.mean(v)) if v else None


def bootstrap_studies(
    per_study_by_tool: Dict[str, Dict[str, dict]],
    key: str,
    ref: str,
    n_boot: int = 2000,
    seed: int = 0,
):
    """95% CIs of each tool's mean over studies, and of each tool's difference to ``ref``, resampling studies."""
    tools = list(per_study_by_tool)
    studies = sorted(
        set.intersection(
            *[
                set(g for g, r in per_study_by_tool[t].items() if r.get(key) is not None)
                for t in tools
            ]
        )
    )
    if not studies:
        return {}, {}
    M = np.array([[per_study_by_tool[t][g][key] for g in studies] for t in tools], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(studies), size=(n_boot, len(studies)))
    boot = M[:, idx].mean(2)  # tools x n_boot
    ci = {
        t: [float(np.quantile(boot[i], 0.025)), float(np.quantile(boot[i], 0.975))]
        for i, t in enumerate(tools)
    }
    r = tools.index(ref)
    diff = {
        t: [
            float(np.quantile(boot[i] - boot[r], 0.025)),
            float(np.quantile(boot[i] - boot[r], 0.975)),
        ]
        for i, t in enumerate(tools)
    }
    return ci, diff
