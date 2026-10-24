# SPDX-License-Identifier: MIT
"""Ontology-aware scoring of cell type calls against author labels (Cell Ontology ids).

Truth and predictions are both mapped to guide nodes through CL ancestry: a truth id goes to the deepest node whose CL id is the
truth id or one of its ancestors (an exact CL match wins); when no node is an ancestor, to the lowest common guide ancestor of
the nodes that descend from it. Each call is then

    exact       predicted node == truth node
    coarse      predicted node is a strict ancestor of the truth node (right, but stopped early)
    finer       predicted node is a strict descendant of the truth node (consistent with a coarser author label)
    wrong       anything else
    unassigned  no call

and accuracy at guide depth ``d`` compares the depth-``d`` ancestors of truth and prediction over the cells whose truth reaches
depth ``d`` (stopping early counts as wrong at the deeper levels). Depth 1 is the compartment (immune, epithelial, ...), 2 the
lineage, 3 the cell type, 4 the subtype. Same rules as the singlet-agents benchmark, so the numbers are comparable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .guide import Guide, Ontology, ontology


def _node_cl_sets(guide: Guide) -> List[set]:
    return [set([n.get("cl")] + list(n.get("also_cl") or [])) - {None, ""} for n in guide.nodes]


class TruthMapper:
    def __init__(self, guide: Guide, onto: Optional[Ontology] = None):
        self.g, self.o = guide, onto or ontology()
        self.sets = _node_cl_sets(guide)
        self.cache: Dict[str, int] = {}

    def __call__(self, cl_id: str) -> int:
        hit = self.cache.get(cl_id)
        if hit is not None:
            return hit
        g, anc = self.g, self.o.ancestors(cl_id)
        owner: Dict[str, int] = {}
        for j in range(1, len(g.nodes)):
            for c in self.sets[j] & anc:
                if c not in owner or g.depth[j] < g.depth[owner[c]]:
                    owner[c] = j
        if cl_id in owner:
            j = owner[cl_id]
        elif owner:
            j = max(owner.values(), key=lambda k: (g.depth[k], -k))
        else:
            desc = [
                k
                for k in range(1, len(g.nodes))
                if any(cl_id in self.o.ancestors(c) for c in self.sets[k])
            ]
            j = _lca(g, desc) if desc else 0
        self.cache[cl_id] = int(j)
        return int(j)


def _lca(g: Guide, nodes: Sequence[int]) -> int:
    paths = [[0] + g.path(j) for j in nodes]
    out = 0
    for level in zip(*paths):
        if all(x == level[0] for x in level):
            out = level[0]
        else:
            break
    return out


def classify(g: Guide, truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    cat = np.empty(len(pred), dtype=object)
    for i, (t, p) in enumerate(zip(truth, pred)):
        if t == 0:
            cat[i] = "untestable"
        elif p == t:
            cat[i] = "exact"
        elif p == 0:
            cat[i] = "unassigned"
        elif p in g.path(int(t)):
            cat[i] = "coarse"
        elif t in g.path(int(p)):
            cat[i] = "finer"
        else:
            cat[i] = "wrong"
    return cat


def report(
    g: Guide,
    truth_cl: Sequence[str],
    pred_nodes: np.ndarray,
    score: Optional[np.ndarray] = None,
    max_depth: int = 4,
) -> dict:
    """Metrics for one method on one sample/study. ``pred_nodes`` index ``g``; ``score`` (optional) gives calibration bins."""
    tm = TruthMapper(g)
    tn = np.array([tm(t) for t in truth_cl], np.int32)
    pn = np.asarray(pred_nodes, np.int32)
    cat = classify(g, tn, pn)
    ok = cat != "untestable"
    frac = {
        k: float(np.mean(cat[ok] == k)) for k in ("exact", "coarse", "finer", "wrong", "unassigned")
    }
    by_depth = {}
    for d in range(1, max_depth + 1):
        td, pd_ = g.ancestor_at_depth(tn[ok], d), g.ancestor_at_depth(pn[ok], d)
        keep = g.depth[td] == d
        if keep.sum() < 20:
            continue
        by_depth[d] = dict(
            n=int(keep.sum()),
            accuracy=float(np.mean(td[keep] == pd_[keep])),
            n_classes=int(len(np.unique(td[keep]))),
        )
    out = dict(
        n_cells=int(len(pn)),
        n_testable=int(ok.sum()),
        fractions=frac,
        consistent=frac["exact"] + frac["coarse"] + frac["finer"],
        by_depth=by_depth,
        mean_pred_depth=float(np.mean(g.depth[pn])),
        mean_truth_depth=float(np.mean(g.depth[tn[ok]])) if ok.any() else None,
    )
    if score is not None:
        s = np.asarray(score, np.float64)[ok]
        right = np.isin(cat[ok], ["exact", "coarse", "finer"])
        bins = []
        for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)):
            m = (s >= lo) & (s < hi)
            if m.sum():
                bins.append(
                    dict(lo=lo, hi=min(hi, 1.0), n=int(m.sum()), consistent=float(right[m].mean()))
                )
        out["calibration"] = bins
    wrong = np.flatnonzero(cat == "wrong")
    pairs: Dict[str, int] = {}
    for t, p in zip(tn[wrong], pn[wrong]):
        k = f"{g.ids[t]}>{g.ids[p]}"
        pairs[k] = pairs.get(k, 0) + 1
    out["top_confusions"] = dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:15])
    return out
