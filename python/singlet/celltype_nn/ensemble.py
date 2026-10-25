# SPDX-License-Identifier: MIT
"""The annotator ensemble whose calls the network learns to reconstitute (weak labels next to the author labels).

Members (choice and licences: ``docs/celltype_nn.md`` section 2):

* ``guide``: Singlet's marker hierarchy (:mod:`singlet.celllabels`, MIT), called per sample with the sample's tissue text.
* ``celltypist``: CellTypist 1.7 (MIT; Dominguez Conde et al. 2022) run as published, per cell (no majority voting), with
  ``Immune_All_Low`` on every human cell plus the model matched to the sample's tissue. Its labels are mapped to CL ids by a
  crosswalk (:func:`celltypist_crosswalk`).
* ``singler``: a correlation annotator rebuilt from the published description of SingleR (Aran et al. 2019; the R package is
  GPL-3 and none of its code is used): reference profiles are pseudobulks per (dataset, CL term) of the training corpus;
  marker genes per pair of labels; a cell scores each label by the 0.8 quantile of its Spearman correlations with that label's
  profiles over the marker genes, then one round of fine-tuning re-scores its top candidates on the markers among them.

Every member returns a :class:`MemberCalls` (CL id per cell, None = no call, and a confidence in [0, 1]).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sps

from singlet.celllabels.guide import Ontology, ontology, tissue_groups


@dataclass
class MemberCalls:
    cl: np.ndarray  # object array of CL ids or None
    score: np.ndarray  # float32
    raw: Optional[np.ndarray] = None  # the member's own label (CellTypist), for the crosswalk


# --------------------------------------------------------------------------------------------------------------- guide
def run_guide(X, symbols: Sequence[str], species: str, tissue: Optional[str], disease: Optional[str] = None) -> MemberCalls:
    from singlet.celllabels.celltype import annotate_cell_types

    r = annotate_cell_types(X, list(symbols), species=species, tissue=tissue, disease=disease, method="guide")
    s = np.nan_to_num(r.path_score if r.path_score is not None else r.score, nan=0.0).astype(np.float32)
    return MemberCalls(r.cl, s)


# ---------------------------------------------------------------------------------------------------------- celltypist
# tissue group (singlet.celllabels.guide.tissue_groups) -> CellTypist models, per species
CELLTYPIST_MODELS = {
    "human": {
        "*": ["Immune_All_Low.pkl"],
        "lung": ["Human_Lung_Atlas.pkl"],
        "gut": ["Cells_Intestinal_Tract.pkl"],
        "stomach": ["Cells_Intestinal_Tract.pkl"],
        "esophagus": ["Cells_Intestinal_Tract.pkl"],
        "skin": ["Adult_Human_Skin.pkl"],
        "heart": ["Healthy_Adult_Heart.pkl"],
        "liver": ["Healthy_Human_Liver.pkl"],
        "pancreas": ["Adult_Human_PancreaticIslet.pkl"],
        "breast": ["Cells_Adult_Breast.pkl"],
        "brain": ["Adult_Human_MTG.pkl"],
        "female_reproductive": ["Human_Endometrium_Atlas.pkl"],
        "lymphoid_organ": ["Cells_Human_Tonsil.pkl"],
        "placenta": ["Human_Placenta_Decidua.pkl"],
        "eye": ["Fetal_Human_Retina.pkl"],
    },
    "mouse": {
        "*": [],
        "gut": ["Adult_Mouse_Gut.pkl"],
        "brain": ["Mouse_Whole_Brain.pkl"],
        "liver": ["Healthy_Mouse_Liver.pkl"],
    },
}


def celltypist_models_for(species: str, tissue: Optional[str]) -> List[str]:
    table = CELLTYPIST_MODELS.get(species, {})
    out = list(table.get("*", []))
    for g in sorted(tissue_groups(tissue)):
        for m in table.get(g, []):
            if m not in out:
                out.append(m)
    return out


def run_celltypist(X, symbols: Sequence[str], species: str, tissue: Optional[str], model_dir: Path) -> Dict[str, MemberCalls]:
    """Raw CellTypist labels per model (``cl`` left None; map with a crosswalk)."""
    import anndata as ad
    import celltypist
    import pandas as pd

    from singlet.celllabels.celltype import lognorm

    out: Dict[str, MemberCalls] = {}
    names = pd.Index([str(s) for s in symbols])
    keep = ~names.duplicated()
    A = ad.AnnData(X=lognorm(sps.csr_matrix(X))[:, np.flatnonzero(keep)], var=pd.DataFrame(index=names[keep]))
    for m in celltypist_models_for(species, tissue):
        p = Path(model_dir) / m
        if not p.exists():
            continue
        model = celltypist.models.Model.load(str(p))
        if len(set(model.features) & set(A.var_names)) < 100:
            continue
        pred = celltypist.annotate(A, model=model, majority_voting=False)
        lab = pred.predicted_labels["predicted_labels"].astype(str).to_numpy()
        prob = pred.probability_matrix.max(1).to_numpy(np.float32)
        out[m] = MemberCalls(np.array([None] * len(lab), dtype=object), prob, raw=lab.astype(object))
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9+]+", " ", s.lower()).strip()


def celltypist_crosswalk(
    model_labels: Sequence[str], author_cl: Sequence[str], onto: Optional[Ontology] = None, min_cover: float = 0.6, min_cells: int = 30
) -> Dict[str, str]:
    """CellTypist label -> CL id, from cells of the training split that carry both.

    An exact name match to a CL term wins. Otherwise the label maps to the most specific CL term that is the author term (or an
    ancestor of it) for at least ``min_cover`` of the label's cells. Labels seen on fewer than ``min_cells`` cells are not
    mapped."""
    o = onto or ontology()
    byname = {_norm(v.get("name", "")): k for k, v in o.terms.items()}
    labs = np.asarray(model_labels, dtype=object)
    auth = np.asarray(author_cl, dtype=object)
    out: Dict[str, str] = {}
    for lab in sorted(set(labs.tolist())):
        hit = byname.get(_norm(str(lab)))
        if hit:
            out[lab] = hit
            continue
        cells = auth[labs == lab]
        if len(cells) < min_cells:
            continue
        cover: Dict[str, int] = {}
        for t in cells:
            for a in o.ancestors(t):
                cover[a] = cover.get(a, 0) + 1
        ok = [a for a, n in cover.items() if n >= min_cover * len(cells)]
        if ok:
            out[lab] = max(ok, key=lambda a: (len(o.ancestors(a)), cover[a]))
    return out


def apply_crosswalk(calls: MemberCalls, xw: Dict[str, str]) -> MemberCalls:
    cl = np.array([xw.get(x) for x in calls.raw], dtype=object)
    return MemberCalls(cl, calls.score, calls.raw)


def pool_celltypist(
    calls: Dict[str, MemberCalls], crosswalks: Dict[str, Dict[str, str]], onto: Optional[Ontology] = None, n: Optional[int] = None
) -> MemberCalls:
    """One CellTypist call per cell from several models: the mapped calls, keeping a more specific one (a descendant of the
    current call) when its model's confidence is at least 0.5. The immune model comes first, tissue models refine it."""
    o = onto or ontology()
    if calls:
        n = len(next(iter(calls.values())).score)
    elif n is None:
        raise ValueError("pool_celltypist: no model ran; pass n (cells) to get an all-None call")
    cl = np.full(n, None, object)
    sc = np.zeros(n, np.float32)
    for m in sorted(calls, key=lambda x: (x != "Immune_All_Low.pkl", x)):
        mapped = apply_crosswalk(calls[m], crosswalks.get(m, {}))
        for i in np.flatnonzero([c is not None for c in mapped.cl]):
            new = mapped.cl[i]
            if cl[i] is None or (cl[i] in o.ancestors(new) and mapped.score[i] >= 0.5):
                cl[i], sc[i] = new, mapped.score[i]
    return MemberCalls(cl, sc)


def vote(calls: Sequence[Sequence[Optional[str]]], onto: Optional[Ontology] = None) -> np.ndarray:
    """Consensus of several members per cell: the most specific common ancestor of any two calls that agree (one is the
    other or an ancestor of it, or they share a specific ancestor); a lone call stands; no call when nothing is called."""
    o = onto or ontology()
    out = []
    for row in zip(*calls):
        c = [x for x in row if isinstance(x, str)]
        best, bd = None, -1
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                common = o.ancestors(c[i]) & o.ancestors(c[j])
                for a in common:
                    d = len(o.ancestors(a))
                    if d > bd:
                        best, bd = a, d
        out.append(best if best is not None else (c[0] if len(c) == 1 else None))
    return np.array(out, dtype=object)


# ------------------------------------------------------------------------------------------------- SingleR-style member
def _rank_rows(D: np.ndarray) -> np.ndarray:
    """Average ranks per row (ties averaged), centred and scaled so that a dot product of two rows is Spearman's rho."""
    from scipy.stats import rankdata

    R = rankdata(D, axis=1).astype(np.float32)
    R -= R.mean(1, keepdims=True)
    n = np.sqrt((R * R).sum(1, keepdims=True))
    return R / np.maximum(n, 1e-12)


def _de_n(n_labels: int) -> int:
    # SingleR's default number of markers per label pair: 500 * (2/3)^log2(n_labels)
    return max(5, int(round(500 * (2 / 3) ** np.log2(max(n_labels, 2)))))


class SingleRRef:
    """Reference for the SingleR-style member: pseudobulk profiles (log1p CP10k means) per (group, label)."""

    def __init__(self, profiles: np.ndarray, labels: np.ndarray, groups: np.ndarray, genes: Sequence[str]):
        self.P = np.asarray(profiles, np.float32)  # samples x genes
        self.labels = np.asarray(labels, dtype=object)
        self.groups = np.asarray(groups, dtype=object)
        self.genes = list(genes)
        self.ulabels = np.array(sorted(set(self.labels.tolist())), dtype=object)
        self.lab_idx = {l: i for i, l in enumerate(self.ulabels)}
        self.sample_lab = np.array([self.lab_idx[l] for l in self.labels])
        L = len(self.ulabels)
        med = np.stack([np.median(self.P[self.sample_lab == i], 0) for i in range(L)])
        k = _de_n(L)
        self.k = k
        # pairwise markers: M[a, b] = the k genes most up in a against b
        self.pair = np.zeros((L, L, k), np.int32)
        for a in range(L):
            diff = med[a][None, :] - med
            idx = np.argpartition(-diff, k, axis=1)[:, :k]
            self.pair[a] = idx
        self.union = np.unique(self.pair.reshape(-1))

    @classmethod
    def build(cls, X_lognorm: sps.csr_matrix, labels: Sequence[str], groups: Sequence[str], genes: Sequence[str], min_cells: int = 10):
        """Pseudobulk means of ``X_lognorm`` (cells x genes) per (group, label) with at least ``min_cells`` cells."""
        import pandas as pd

        key = pd.Series([f"{g}\t{l}" for g, l in zip(groups, labels)])
        codes, uniq = pd.factorize(key)
        n = np.bincount(codes)
        G = sps.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
        sums = np.asarray((G @ X_lognorm).todense(), np.float32)
        keep = n >= min_cells
        prof = sums[keep] / n[keep, None]
        gl = [u.split("\t") for u in np.asarray(uniq)[keep]]
        return cls(prof, np.array([x[1] for x in gl], object), np.array([x[0] for x in gl], object), genes)

    def score(
        self, X_lognorm: sps.csr_matrix, exclude_group: Optional[Sequence[str]] = None, quantile: float = 0.8, fine_tune: bool = True,
        top: int = 5, chunk: int = 4096,
    ) -> MemberCalls:
        """Label per cell. ``exclude_group`` (per cell) leaves the cell's own dataset out of the reference (cross-fitting)."""
        n = X_lognorm.shape[0]
        U = self.union
        RU = _rank_rows(self.P[:, U])
        L = len(self.ulabels)
        best = np.empty(n, dtype=object)
        conf = np.zeros(n, np.float32)
        excl = None if exclude_group is None else np.asarray(exclude_group, dtype=object)
        for s in range(0, n, chunk):
            Q = np.asarray(X_lognorm[s : s + chunk][:, U].todense(), np.float32)
            rho = _rank_rows(Q) @ RU.T  # cells x samples
            if excl is not None:
                rho[excl[s : s + chunk, None] == self.groups[None, :]] = np.nan
            sc = np.full((len(Q), L), -1.0, np.float32)
            for i in range(L):
                m = self.sample_lab == i
                with np.errstate(all="ignore"):
                    v = np.nanquantile(rho[:, m], quantile, axis=1) if m.sum() > 1 else rho[:, m][:, 0]
                sc[:, i] = np.nan_to_num(v, nan=-1.0)
            Qfull = None
            for r in range(len(Q)):
                cand = np.argsort(-sc[r])[:top]
                if fine_tune and len(cand) > 1 and sc[r, cand[0]] - sc[r, cand[1]] < 0.05:
                    if Qfull is None:
                        Qfull = np.asarray(X_lognorm[s : s + chunk].todense(), np.float32)
                    cand = self._fine_tune(Qfull[r], cand, sc[r], excl[s + r] if excl is not None else None, quantile)
                best[s + r] = self.ulabels[cand[0]]
                # confidence: margin of the winner over the runner-up on the coarse scores, squashed to [0, 1]
                d = sc[r, cand[0]] - (sc[r, cand[1]] if len(cand) > 1 else -1.0)
                conf[s + r] = float(np.clip(d / 0.1, 0, 1))
        return MemberCalls(best, conf)

    def _fine_tune(self, q: np.ndarray, cand: np.ndarray, sc: np.ndarray, excl, quantile: float) -> np.ndarray:
        cand = list(cand)
        while len(cand) > 1:
            genes = np.unique(np.concatenate([self.pair[a, b] for a in cand for b in cand if a != b]))
            m = np.isin(self.sample_lab, cand)
            if excl is not None:
                m &= self.groups != excl
            if m.sum() == 0:
                break
            rq = _rank_rows(q[genes][None, :])
            rr = _rank_rows(self.P[m][:, genes])
            rho = (rq @ rr.T)[0]
            lab = self.sample_lab[m]
            s = np.array([np.quantile(rho[lab == a], quantile) if (lab == a).any() else -1 for a in cand])
            keep = [a for a, v in zip(cand, s) if v >= s.max() - 0.05]
            if len(keep) == len(cand):
                cand = [cand[int(np.argmax(s))]] + [a for a in cand if a != cand[int(np.argmax(s))]]
                break
            cand = sorted(keep, key=lambda a: -s[cand.index(a)])
        return np.array(cand)


# ------------------------------------------------------------------------------------------------------------ consensus
def refinement(author: str, calls: Sequence[Optional[str]], onto: Optional[Ontology] = None, min_votes: int = 2) -> Optional[str]:
    """The ensemble's refinement of an author term: the most specific common ancestor of the member calls that are strict
    descendants of ``author``, when at least ``min_votes`` members make such a call and that ancestor is still below the author's
    term. None otherwise (the author term stays the only target)."""
    o = onto or ontology()
    below = [c for c in calls if c and c != author and author in o.ancestors(c)]
    if len(below) < min_votes:
        return None
    common = set(o.ancestors(below[0]))
    for c in below[1:]:
        common &= o.ancestors(c)
    common = {a for a in common if a != author and author in o.ancestors(a)}
    if not common:
        return None
    return max(common, key=lambda a: (len(o.ancestors(a)), a))
