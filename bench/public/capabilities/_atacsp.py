"""Metric helpers shared by the atac, multiome and spatial capability modules (owner: atac-multiome-spatial).

Inputs are the artifacts written by bench/atac_spatial/make_artifacts.py: per-barcode / per-spot / per-gene tables, peak
BEDs and small JSON summaries. Every function here is deterministic and cheap (laptop, seconds).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def read_tsv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, sep="\t", index_col=0)


def read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def spearman(x, y) -> float | None:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[m], y[m])[0]) if m.sum() >= 10 else None


def pearson_log(x, y) -> float | None:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return float(pearsonr(np.log1p(x[m]), np.log1p(y[m]))[0]) if m.sum() >= 10 else None


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def read_bed(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
    return d[~d.chrom.isin(["chrM", "MT"])].reset_index(drop=True)


def overlapped_fraction(A: pd.DataFrame, B: pd.DataFrame) -> float:
    """Fraction of intervals of A that overlap at least one interval of B (1 bp)."""
    hit = 0
    for c, ga in A.groupby("chrom"):
        gb = B[B.chrom == c].sort_values("start")
        if gb.empty:
            continue
        bs = gb.start.to_numpy(); bmax = np.maximum.accumulate(gb.end.to_numpy())
        k = np.searchsorted(bs, ga.end.to_numpy())
        ok = (k > 0) & (bmax[np.maximum(k - 1, 0)] > ga.start.to_numpy())
        hit += int(ok.sum())
    return hit / max(1, len(A))


def peak_scores(P: pd.DataFrame, R: pd.DataFrame) -> dict:
    """Recall (reference peaks hit), precision (own peaks hitting the reference), F1, counts, widths."""
    rec = overlapped_fraction(R, P); prec = overlapped_fraction(P, R)
    return {"recall": rec, "precision": prec, "f1": 2 * rec * prec / max(1e-12, rec + prec), "n": int(len(P)),
            "median_width": float((P.end - P.start).median())}


def archr_default_cells(fragments, tsse) -> np.ndarray:
    """ArchR's default cell filter (createArrowFiles: minFrags 1000, minTSS 4)."""
    f = np.nan_to_num(np.asarray(fragments, float)); t = np.nan_to_num(np.asarray(tsse, float))
    return (f >= 1000) & (t >= 4)


def otsu_threshold(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, float)); n = len(x)
    pre = np.concatenate([[0], np.cumsum(x)])
    best, thr = -1.0, float(x[n // 2])
    for i in range(1, n):
        if x[i] == x[i - 1]:
            continue
        w0 = i / n; w1 = 1 - w0; m0 = pre[i] / i; m1 = (pre[n] - pre[i]) / (n - i)
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best:
            best, thr = v, 0.5 * (x[i - 1] + x[i])
    return thr


def cohen_kappa(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, int); y_pred = np.asarray(y_pred, int)
    po = float(np.mean(y_true == y_pred))
    p1 = np.mean(y_true) * np.mean(y_pred) + (1 - np.mean(y_true)) * (1 - np.mean(y_pred))
    return (po - p1) / max(1e-12, 1 - p1)


def bootstrap_metric(fn, n: int, n_boot: int = 1000, seed: int = 0):
    """95% CI of fn(idx) over bootstrap resamples of n rows."""
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals else None


def metric(id_, name, unit, category, dataset, direction, singlet, competitors, rule, notes=None, ci=None):
    m = {"id": id_, "name": name, "unit": unit, "category": category, "dataset": dataset, "direction": direction,
         "singlet": {"value": None if singlet is None else round(float(singlet), 4)},
         "competitors": [{"tool": t, "value": None if v is None else round(float(v), 4)} for t, v in competitors],
         "rule": rule}
    if ci:
        m["singlet"]["ci"] = [round(ci[0], 4), round(ci[1], 4)]
    if notes:
        m["notes"] = notes
    if not m["competitors"]:
        del m["competitors"]
    return m


def diff_tolerance(f_singlet, f_comp, n: int, n_boot: int = 500, seed: int = 0, unit: str = "cells"):
    """Match-or-beat tolerance from measured noise: the half-width of the bootstrap 95% CI of the difference
    f_singlet(idx) - f_comp(idx), resampling the same n units (cells, spots, genes or chromosomes) for both tools."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        a, b = f_singlet(i), f_comp(i)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            d.append(a - b)
    if len(d) < 50:
        return None
    lo, hi = np.quantile(d, [0.025, 0.975])
    half = float((hi - lo) / 2)
    return {"type": "match_or_beat", "against": "best", "tolerance": {"abs": round(half, 4)},
            "rationale": f"half-width of the bootstrap 95% CI of the Singlet - competitor difference ({n_boot} resamples "
                         f"of the {n} {unit}): differences smaller than this are not distinguishable on this dataset"}


def peak_f1_by_chrom(P: pd.DataFrame, R: pd.DataFrame, chroms) -> float:
    """Peak F1 restricted to a (bootstrap) multiset of chromosomes."""
    Ps = pd.concat([P[P.chrom == c] for c in chroms]); Rs = pd.concat([R[R.chrom == c] for c in chroms])
    return peak_scores(Ps.reset_index(drop=True), Rs.reset_index(drop=True))["f1"]
