# SPDX-License-Identifier: MIT
"""Family comparison engine shared by gex-counts and spliced-unspliced.

A *family* is a set of tools that count the same quantity (for example Singlet exon_counts, STARsolo Gene and the published
Cell Ranger matrix all count exonic UMIs). Singlet is one member; the others are its competitors. For one dataset the engine reads
the per-cell table written by ``extract/counts.py`` and measures, for every pair of tools, three agreements:

  ratio   |ln(median over cells of UMI_a / UMI_b)|       (0 = same totals; a distance, lower is better)
  corr    median over cells of the Pearson r of log1p UMI over genes
  rho     Spearman rho of the pseudobulk gene counts

The question the gate asks is whether Singlet sits inside the cloud of existing tools: its mean distance to the competitors must not
exceed the distance the competitors have from each other (their mutual spread), within the half-width of the paired bootstrap 95% CI
of the difference. With two competitors Y and Z that reads  mean(d(S,Y), d(S,Z)) <= d(Y,Z)  (and the same for r and rho).
Averaging over the competitors, rather than requiring Singlet to be at least as close to each one separately, keeps the gate from
failing two equally good tools by coin-flip: if Singlet were an exchangeable third tool it would fail the one-sided per-reference
test a third of the time. Each per-reference value is in the metric's notes and in the figures, so the worst case is visible.

When a dataset has only one competitor in a family (for example the published Cell Ranger matrix is not comparable on the 60 M-read 5'
subsample) there is no spread to measure: the gate is then a threshold taken from the other datasets (the largest competitor spread for
the ratio, the lowest for r and rho).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import _pubcommon as C


@dataclass
class Spec:
    fam: str                      # column prefix in the tables
    singlet: str                  # tool key of Singlet's variant
    label: dict[str, str]         # tool key -> display label
    declared: dict[str, str]      # tool key -> declared contract tool name
    quantity: str                 # e.g. "exonic UMI"
    id_prefix: str                # metric id suffix, e.g. "exonic"


@dataclass
class Pair:
    value: float
    boot: np.ndarray              # bootstrap samples (same resamples for every pair of a family and statistic)


@dataclass
class FamStat:
    ds: str
    spec: Spec
    members: list[str]
    n_cells: int
    n_genes: int
    ratio: dict = field(default_factory=dict)    # (a, b) -> Pair  |ln median ratio|
    raw_ratio: dict = field(default_factory=dict)  # (a, b) -> median ratio a / b
    corr: dict = field(default_factory=dict)
    rho: dict = field(default_factory=dict)

    @property
    def comps(self) -> list[str]:
        return [t for t in self.members if t != self.spec.singlet]


def _col(df: pd.DataFrame, fam: str, a: str, b: str):
    if f"{fam}:r:{a}~{b}" in df.columns:
        return df[f"{fam}:r:{a}~{b}"].to_numpy(float)
    if f"{fam}:r:{b}~{a}" in df.columns:
        return df[f"{fam}:r:{b}~{a}"].to_numpy(float)
    return None


def members(df: pd.DataFrame, spec: Spec) -> list[str]:
    pre = f"{spec.fam}:n:"
    return [c[len(pre):] for c in df.columns if c.startswith(pre) and df[c].notna().any()]


def collect(ds: str, spec: Spec, cells: pd.DataFrame, genes: pd.DataFrame, dsidx: int) -> FamStat | None:
    have = members(cells, spec)
    if spec.singlet not in have or len(have) < 2:
        return None
    n = {t: cells[f"{spec.fam}:n:{t}"].to_numpy(float) for t in have}
    pairs = [(spec.singlet, c) for c in have if c != spec.singlet] + list(itertools.combinations([t for t in have if t != spec.singlet], 2))
    # ---- UMI ratio on cells with UMI in every member
    ok = np.ones(len(cells), bool)
    for t in have:
        ok &= np.isfinite(n[t]) & (n[t] > 0)
    st = FamStat(ds, spec, have, int(ok.sum()), 0)
    idx = C.boot_indices(int(ok.sum()), offset=dsidx * 101 + 1)
    for a, b in pairs:
        r = n[a][ok] / n[b][ok]
        med = float(np.median(r))
        st.raw_ratio[(a, b)] = med
        st.ratio[(a, b)] = Pair(abs(np.log(med)), np.abs(np.log(np.median(r[idx], axis=1))))
    # ---- per-cell correlation
    cr = {(a, b): _col(cells, spec.fam, a, b) for a, b in pairs}
    valid = np.ones(len(cells), bool)
    for c in cr.values():
        valid &= np.isfinite(c)
    if valid.sum() >= 20:
        idx = C.boot_indices(int(valid.sum()), offset=dsidx * 101 + 2)
        for p, c in cr.items():
            v = c[valid]
            st.corr[p] = Pair(float(np.median(v)), np.median(v[idx], axis=1))
    # ---- pseudobulk rho
    gcols = {t: genes[f"{spec.fam}:{t}"].to_numpy(float) for t in have if f"{spec.fam}:{t}" in genes.columns}
    if len(gcols) == len(have):
        expressed = np.zeros(len(genes), bool)
        for g in gcols.values():
            expressed |= g > 0
        st.n_genes = int(expressed.sum())
        ranks = {t: pd.Series(g[expressed]).rank().to_numpy() for t, g in gcols.items()}
        idx = C.boot_indices(st.n_genes, offset=dsidx * 101 + 3, B=200)
        for a, b in pairs:
            rho = C.spearman(gcols[a][expressed], gcols[b][expressed])
            st.rho[(a, b)] = Pair(rho, np.array([C.pearson(ranks[a][i], ranks[b][i]) for i in idx]))
    return st


def _key(d: dict, a: str, b: str):
    return d.get((a, b)) or d.get((b, a))


def _agg(d: dict, a: str, others: list[str]) -> tuple[float, np.ndarray]:
    ps = [_key(d, a, o) for o in others]
    return float(np.mean([p.value for p in ps])), np.mean([p.boot for p in ps], axis=0)


STATS = (
    # attribute, id stem, display name, unit, direction, category
    ("ratio", "umi_ratio_distance", "Per-cell UMI total, distance of the median ratio from 1", "|ln ratio|", "lower", "agreement"),
    ("corr", "gene_corr", "Per-cell gene correlation", "r", "higher", "agreement"),
    ("rho", "pseudobulk_rho", "Pseudobulk gene Spearman rho", "rho", "higher", "agreement"),
)


def build_metrics(stats: list[FamStat]) -> list[dict]:
    """Contract metrics from the collected statistics. Two passes: a dataset with one competitor borrows its threshold from the others."""
    spread: dict = {}
    for s in stats:
        if len(s.comps) < 2:
            continue
        for attr, *_ in STATS:
            d = getattr(s, attr)
            if not d:
                continue
            vals = [_key(d, a, b).value for a, b in itertools.combinations(s.comps, 2)]
            spread.setdefault((s.spec.fam, attr), []).append(float(np.mean(vals)))
    metrics = []
    for s in stats:
        sp = s.spec
        L, D = sp.label, sp.declared
        comps = s.comps
        for attr, stem, name, unit, direction, cat in STATS:
            d = getattr(s, attr)
            if not d or any(_key(d, sp.singlet, c) is None for c in comps):
                continue
            val, boot = _agg(d, sp.singlet, comps)
            ci = C.ci95(boot)
            per_ref = "; ".join(
                (f"{L[c]}: ratio {s.raw_ratio[(sp.singlet, c)]:.3f}"
                 if attr == "ratio" else f"{L[c]}: {_key(d, sp.singlet, c).value:.4f}") for c in comps)
            basis = {"ratio": f"cells with UMI in every tool ({s.n_cells})", "corr": "cells common to the family",
                     "rho": f"{s.n_genes} genes with any UMI"}[attr]
            notes = f"Singlet vs each competitor (S/competitor for ratios): {per_ref}. Basis: {basis}."
            if len(comps) >= 2:
                cvals = []
                for c in comps:
                    others = [o for o in comps if o != c]
                    v, b = _agg(d, c, others)
                    cvals.append({"tool": D[c], "value": v, "ci": C.ci95(b)})
                ref_val, ref_boot = float(np.mean([_key(d, a, b).value for a, b in itertools.combinations(comps, 2)])), \
                    np.mean([_key(d, a, b).boot for a, b in itertools.combinations(comps, 2)], axis=0)
                diff = (boot - ref_boot) if direction == "lower" else (ref_boot - boot)
                hw = C.half_width(C.ci95(diff))
                rule = C.match_rule(hw, (f"gate = Singlet's mean {unit} distance to {', '.join(L[c] for c in comps)} against the distance the competitors "
                                         f"have from each other ({ref_val:.4f}); tolerance = half-width of the paired bootstrap 95% CI of the difference: {hw:.4f}"))
                metrics.append(C.metric(f"{stem}.{sp.id_prefix}", f"{name}, {sp.quantity}", unit, cat, s.ds, direction, val, ci, cvals, rule,
                                        notes=notes + f" Competitor value = its mean distance to the other competitors."))
            else:
                vals = spread.get((sp.fam, attr), [])
                if not vals:
                    continue
                thr = max(vals) if direction == "lower" else min(vals)
                rule = {"type": "threshold", "value": C.rnd(thr),
                        "rationale": (f"only one competitor ({L[comps[0]]}) on this dataset, so there is no competitor spread to measure; threshold = the "
                                      f"{'largest' if direction == 'lower' else 'lowest'} competitor-vs-competitor value on the other datasets ({thr:.4f})")}
                metrics.append(C.metric(f"{stem}.{sp.id_prefix}", f"{name}, {sp.quantity}", unit, cat, s.ds, direction, val, ci, [], rule, notes=notes))
    return metrics
