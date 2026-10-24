# SPDX-License-Identifier: MIT
"""The model's output space: Cell Ontology terms, their ``is_a`` closure, the credit a prediction earns and the roll-up.

The network outputs a softmax over a fixed list of CL terms (:class:`TermSpace`), internal terms included: mass on 'T cell'
means "a T cell, subtype unresolved". The probability of any term ``a`` (in the space or an ancestor of one) is the mass on
every output term that is ``a`` or a descendant of it, so probabilities roll up the DAG without double counting.

Credit of an output term ``v`` against a target term ``t`` (what the loss and the evaluation share):

    1                     v is t or a descendant of t (finer than the author is never penalised)
    IC(v) / IC(t)         v is a strict ancestor of t (coarser: partial credit, less the more general it is)
    0                     anything else (another branch)

``IC`` is structural information content over the output space: ``log |V| - log |{v in V: a is v or an ancestor of v}|``, so the
root has 0 and a leaf of the space has the maximum. The training loss is ``-log(sum_v p_v * credit(v, t))``: the expected
credit, which is maximised by moving mass to the deepest term the evidence supports.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from singlet.celllabels.guide import Ontology, ontology


class TermSpace:
    def __init__(self, terms: Sequence[str], onto: Optional[Ontology] = None):
        self.o = onto or ontology()
        self.terms: List[str] = list(terms)
        self.index: Dict[str, int] = {t: i for i, t in enumerate(self.terms)}
        anc = [self.o.ancestors(t) for t in self.terms]
        self.closure: List[str] = sorted(set().union(*anc)) if anc else []
        self.cindex: Dict[str, int] = {t: i for i, t in enumerate(self.closure)}
        # R[v, a] = 1 when closure term a is v or an ancestor of v: probabilities over the closure are p @ R
        R = np.zeros((len(self.terms), len(self.closure)), np.float32)
        for i, A in enumerate(anc):
            R[i, [self.cindex[a] for a in A]] = 1.0
        self.R = R
        n_desc = R.sum(0)
        self.ic_closure = (np.log(len(self.terms)) - np.log(np.maximum(n_desc, 1))).astype(np.float32)
        self.ic = self.ic_closure[[self.cindex[t] for t in self.terms]]
        self.depth_closure = np.array([len(self.o.ancestors(t)) for t in self.closure], np.float32)
        self._credit: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.terms)

    def name(self, t: str) -> str:
        return self.o.name(t)

    def ancestors(self, t: str) -> Set[str]:
        return self.o.ancestors(t)

    # --------------------------------------------------------------------------------------------------- credit and loss
    def credit(self, t: str) -> np.ndarray:
        """Credit of every output term against target ``t`` (any CL id)."""
        hit = self._credit.get(t)
        if hit is not None:
            return hit
        at = self.o.ancestors(t)
        ic_t = self.ic_of(t)
        c = np.zeros(len(self.terms), np.float32)
        for i, v in enumerate(self.terms):
            if t in self.o.ancestors(v):
                c[i] = 1.0
            elif v in at:
                c[i] = min(1.0, float(self.ic[i]) / ic_t) if ic_t > 0 else 0.0
        self._credit[t] = c
        return c

    def ic_of(self, t: str) -> float:
        """IC of any term: from the closure when it is there, else from the output terms below it (a leaf if none)."""
        j = self.cindex.get(t)
        if j is not None:
            return float(self.ic_closure[j])
        n = sum(1 for v in self.terms if t in self.o.ancestors(v))
        return float(np.log(len(self.terms)) - np.log(max(n, 1)))

    def credit_matrix(self, targets: Sequence[str]) -> np.ndarray:
        """Unique targets x output terms."""
        return np.stack([self.credit(t) for t in targets]) if len(targets) else np.zeros((0, len(self)), np.float32)

    # ------------------------------------------------------------------------------------------------ roll-up and calls
    def rollup(self, p: np.ndarray) -> np.ndarray:
        """Cells x closure terms: probability that the cell is each closure term or one of its descendants."""
        return np.asarray(p, np.float32) @ self.R

    def call(self, p: np.ndarray, threshold: float = 0.5):
        """The deepest closure term (highest IC) whose rolled-up probability is at least ``threshold``, per cell.

        Returns (closure index array, its rolled-up probability). Probabilities only grow towards the root, so the root always
        qualifies; ties in IC go to the larger probability, then to the deeper term (terms with the same output descendants,
        such as 'alpha-beta T cell' and 'mature alpha-beta T cell', differ only in depth)."""
        P = self.rollup(p)
        ok = P >= threshold
        key = np.where(ok, self.ic_closure[None, :] + 1e-3 * P + 1e-6 * self.depth_closure[None, :], -np.inf)
        j = key.argmax(1)
        return j, P[np.arange(len(j)), j]

    def map_to_space(self, t: str) -> Optional[str]:
        """``t`` if it is an output term, else its most specific ancestor that is one (None if none is)."""
        if t in self.index:
            return t
        anc = [a for a in self.o.ancestors(t) if a in self.index]
        if not anc:
            return None
        return max(anc, key=lambda a: (float(self.ic[self.index[a]]), a))


def build_space(counts: Dict[str, int], min_cells: int, onto: Optional[Ontology] = None) -> tuple:
    """Output terms from target counts: every term with at least ``min_cells`` cells; a rarer term is folded into its most
    specific ancestor (added to the space when no ancestor qualifies). Returns (terms, {original term: space term})."""
    o = onto or ontology()
    known = {t: n for t, n in counts.items() if t in o.terms}
    keep = {t for t, n in known.items() if n >= min_cells}
    fold: Dict[str, str] = {t: t for t in keep}

    def depth(a: str) -> int:
        return len(o.ancestors(a))

    for t in sorted(set(known) - keep, key=lambda x: -depth(x)):
        anc = (o.ancestors(t) - {t}) & keep
        if anc:
            fold[t] = max(anc, key=lambda a: (depth(a), a))
        else:
            cand = o.ancestors(t) - {t}
            if not cand:
                continue
            a = max(cand, key=lambda x: (depth(x), x))
            keep.add(a)
            fold[t] = a
            fold[a] = a
    terms = sorted(keep)
    return terms, fold


def is_consistent(onto: Ontology, pred: str, truth: str) -> str:
    """'exact' | 'coarse' (pred is an ancestor of truth) | 'finer' (pred is a descendant) | 'wrong'."""
    if pred == truth:
        return "exact"
    if pred in onto.ancestors(truth):
        return "coarse"
    if truth in onto.ancestors(pred):
        return "finer"
    return "wrong"


def lca_set(onto: Ontology, terms: Iterable[str]) -> Set[str]:
    """Most specific common ancestors of ``terms`` (several when the DAG branches)."""
    terms = list(terms)
    if not terms:
        return set()
    common = set(onto.ancestors(terms[0]))
    for t in terms[1:]:
        common &= onto.ancestors(t)
    return {a for a in common if not any(a != b and a in onto.ancestors(b) for b in common)}
