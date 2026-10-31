# SPDX-License-Identifier: MIT
"""Score one sample: every registered feature set, as sidecar tables and summary rows.

Input is a :class:`singlet.celllabels.io.SampleInput` (or any object with ``gsm``, ``gse``, ``X`` called cells x genes raw counts,
``gene_names``, ``col``, ``species`` and optionally ``gene_ids``). Cells with fewer than ``MIN_GENES_DETECTED`` detected genes and
samples of other organisms get NaN / no category.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sps

from . import genesets as GS
from . import rank as R
from . import registry as REG


@dataclass
class SetResult:
    feature_set: str
    values: Dict[str, np.ndarray]
    features: Sequence[REG.Feature]
    method_version: str
    coverage: Dict[str, float] = field(default_factory=dict)
    dropped: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, np.ndarray] = field(
        default_factory=dict
    )  # diagnostics, never written to a sidecar


def _empty(fs: str, n: int) -> SetResult:
    feats = REG.features(fs)
    vals = {
        f.name: (
            np.full(n, REG.CATEGORY_NA, np.int64)
            if f.kind == "category"
            else np.full(n, np.nan, np.float32)
        )
        for f in feats
    }
    return SetResult(fs, vals, feats, REG.method_version(fs))


def _doublet_result(sample, timings: Optional[dict]) -> SetResult:
    from . import doublet as DBL

    t0 = time.perf_counter()
    values, extra = DBL.score_sample(sample, timings=timings)
    res = SetResult("doublet", values, REG.features("doublet"), REG.method_version("doublet"))
    if not extra:
        res.dropped = {
            "score_v2": f"no score: other organism, fewer than {DBL.MIN_CELLS} cells, or no engine doublet score"
        }
    if timings is not None:
        timings["doublet_s"] = time.perf_counter() - t0
    return res


def score_sample(
    sample,
    sets: Sequence[str] = REG.FEATURE_SETS,
    *,
    timings: Optional[dict] = None,
) -> Dict[str, SetResult]:
    """Every requested set; ``doublet`` also needs ``sample.prod_doublet_score`` (the engine's score per row of ``X``)."""
    out: Dict[str, SetResult] = {}
    if "doublet" in sets:
        out["doublet"] = _doublet_result(sample, timings)
        sets = tuple(s for s in sets if s != "doublet")
    if sets:
        out.update(_score_gene_sets(sample, sets, timings))
    return out


def _score_gene_sets(sample, sets: Sequence[str], timings: Optional[dict]) -> Dict[str, SetResult]:
    X = sps.csr_matrix(sample.X, dtype=np.float32)
    n = X.shape[0]
    org = sample.species if sample.species in GS.ORGANISMS else None
    if org is None or n == 0:
        return {fs: _empty(fs, n) for fs in sets}
    t0 = time.perf_counter()
    genes = GS.GeneIndex(sample.gene_names, getattr(sample, "gene_ids", None))
    S = R.rank_signal(X, R.MAX_RANK)
    low = np.diff(X.indptr) < REG.MIN_GENES_DETECTED
    if timings is not None:
        timings["rank_s"] = time.perf_counter() - t0
    out: Dict[str, SetResult] = {}
    for fs in sets:
        t1 = time.perf_counter()
        comp = GS.compile_collection(fs, org, genes, k_null=REG.K_NULL)
        auc, nm, sd = R.score_sets(X, comp.index, null_sets=comp.nulls, S=S)
        z = R.zscore(auc, nm, sd)
        score = np.clip(z / REG.Z_SCALE, 0.0, 1.0).astype(np.float32)
        score[low] = np.nan
        score[:, ~comp.ok] = np.nan
        col_of = {name: j for j, name in enumerate(comp.names)}
        res = _empty(fs, n)
        res.coverage = {k: float(c) for k, c in zip(comp.names, comp.coverage)}
        res.dropped = dict(comp.dropped)
        if fs == "hallmark":
            for f in res.features:
                res.values[f.name] = score[:, col_of[f.name]]
            res.extra = {"auc": auc, "null_mean": nm, "z": z}
        elif fs == "cell_cycle":
            js, jg = col_of["s_phase"], col_of["g2m_phase"]
            zs, zg = z[:, js], z[:, jg]
            phase = np.where(np.maximum(zs, zg) >= REG.PHASE_Z, np.where(zg > zs, 2, 1), 0).astype(
                np.int64
            )
            if not (comp.ok[js] and comp.ok[jg]):
                phase[:] = REG.CATEGORY_NA
            phase[low] = REG.CATEGORY_NA
            res.values["phase"] = phase
            res.values["s_score"] = score[:, js]
            res.values["g2m_score"] = score[:, jg]
            res.extra = {"z_s": zs, "z_g2m": zg, "auc": auc}
        out[fs] = res
        if timings is not None:
            timings[f"{fs}_s"] = time.perf_counter() - t1
    return out


def summaries(sample, results: Dict[str, SetResult]) -> List[dict]:
    rows: List[dict] = []
    for r in results.values():
        rows += REG.summary_rows(sample.gsm, sample.gse, r.values, r.features, r.method_version)
    return rows
