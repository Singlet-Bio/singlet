# SPDX-License-Identifier: MIT
"""Chromosomal sex per cell from expression, pooled per donor.

Evidence per cell: UMIs of XIST (``x``), summed UMIs of non-PAR chrY genes (``y``) and the cell's total UMIs (``n``). Two Poisson
components with rates per UMI: female cells express XIST at ``a_f`` and chrY at a leak rate ``e_y``; male cells express chrY at
``b_m`` and XIST at a leak rate ``e_x``. The four rates are fitted per sample by EM, shrunk towards atlas defaults (so a sample of
one sex keeps sensible rates for the other), with a male fraction ``pi`` per group. Groups are the donors of the sample when
genotype demultiplexing found more than one (``donor/donor_assignments.tsv``), otherwise the whole sample.

A cell's posterior is ``sigmoid(LLR + logit(pi_group))`` with the per-cell log-likelihood ratio clipped to +-``LLR_CAP`` (counts
are overdispersed, so no single cell is allowed to be certain on its own). In a single-sex group every cell, including the many
with no XIST or chrY UMI at all, inherits the group's sex unless its own evidence is strongly contrary (a doublet with another
donor, a mislabelled barcode). Labels are ``female`` (PATO:0000383), ``male`` (PATO:0000384) or ``ambiguous`` (posterior between
``1 - MIN_POST`` and ``MIN_POST``, or strong XIST and chrY in the same cell). ``score`` is the posterior of the label given.

Loss of chrY in ageing blood cells does not flip a male donor's cells: such cells have neither XIST nor chrY, so the group prior
decides them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import scipy.sparse as sps

PATO = {"female": "PATO:0000383", "male": "PATO:0000384"}

# Non-PAR chrY genes broadly expressed outside the testis. Mouse: Xist is the X evidence too.
CHRY = {
    "human": ("RPS4Y1", "DDX3Y", "UTY", "KDM5D", "EIF1AY", "ZFY", "USP9Y", "TXLNGY", "NLGN4Y", "PRKY"),
    "mouse": ("Ddx3y", "Eif2s3y", "Uty", "Kdm5d"),
}
XIST = {"human": ("XIST",), "mouse": ("Xist",)}

# Atlas defaults (rates per UMI, gene-level exon + intron counts) and their prior weight in UMIs.
DEFAULTS = {"a_f": 2e-3, "e_y": 2e-6, "b_m": 6e-4, "e_x": 4e-6}
PRIOR_UMIS = 2e5
LLR_CAP = 6.0
MIN_POST = 0.9
EM_ITERS = 30


@dataclass
class SexResult:
    label: np.ndarray  # per cell: 'female' | 'male' | 'ambiguous'
    score: np.ndarray  # posterior of the label (float32)
    p_male: np.ndarray  # per-cell posterior P(male)
    groups: Dict[str, dict] = field(default_factory=dict)  # per group: call, pi_male, n_cells, n_informative
    rates: Dict[str, float] = field(default_factory=dict)


def _genes_index(gene_names: Sequence[str], wanted: Sequence[str]) -> np.ndarray:
    up = {w.upper() for w in wanted}
    return np.array([j for j, g in enumerate(gene_names) if str(g).upper() in up], np.int64)


def sex_evidence(X, gene_names: Sequence[str], species: str = "human"):
    """(x, y, n) per cell: XIST UMIs, chrY UMIs, total UMIs."""
    X = sps.csr_matrix(X)
    n = np.asarray(X.sum(1)).ravel().astype(np.float64)
    xi = _genes_index(gene_names, XIST[species])
    yi = _genes_index(gene_names, CHRY[species])
    x = np.asarray(X[:, xi].sum(1)).ravel().astype(np.float64) if len(xi) else np.zeros(X.shape[0])
    y = np.asarray(X[:, yi].sum(1)).ravel().astype(np.float64) if len(yi) else np.zeros(X.shape[0])
    return x, y, n


def _llr(x, y, n, r):
    a_f, e_y, b_m, e_x = r["a_f"], r["e_y"], r["b_m"], r["e_x"]
    v = y * np.log(b_m / e_y) - n * (b_m - e_y) + x * np.log(e_x / a_f) - n * (e_x - a_f)
    return np.clip(v, -LLR_CAP, LLR_CAP)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def call_sex(
    X,
    gene_names: Sequence[str],
    *,
    species: str = "human",
    groups: Optional[Sequence[Optional[str]]] = None,
) -> SexResult:
    """Per-cell sex. ``groups``: donor label per cell (None / '' for cells not assigned to a donor, which then use the pooled
    prior of the sample); None = one group for the whole sample."""
    if species not in CHRY:
        raise ValueError(f"no sex genes for species {species!r}")
    x, y, n = sex_evidence(X, gene_names, species)
    m = len(n)
    g = np.array(["sample"] * m, dtype=object) if groups is None else np.array([s if s else "" for s in groups], dtype=object)
    labels = sorted({s for s in g if s})
    gi = {s: i for i, s in enumerate(labels)}
    gidx = np.array([gi.get(s, -1) for s in g], np.int64)
    rates = dict(DEFAULTS)
    pi = np.full(len(labels), 0.5)
    pi_all = 0.5
    def group_prior():
        return np.where(gidx >= 0, pi[np.maximum(gidx, 0)], pi_all) if len(labels) else np.full(m, pi_all)

    for _ in range(EM_ITERS):
        prior = group_prior()
        p = _sigmoid(_llr(x, y, n, rates) + np.log(prior / (1 - prior)))
        w0 = PRIOR_UMIS
        f, mm = 1 - p, p
        rates["a_f"] = (f @ x + w0 * DEFAULTS["a_f"]) / (f @ n + w0)
        rates["e_y"] = (f @ y + w0 * DEFAULTS["e_y"]) / (f @ n + w0)
        rates["b_m"] = (mm @ y + w0 * DEFAULTS["b_m"]) / (mm @ n + w0)
        rates["e_x"] = (mm @ x + w0 * DEFAULTS["e_x"]) / (mm @ n + w0)
        # identifiability: the signal rate stays well above the leak rate of the same gene set
        rates["e_x"] = min(rates["e_x"], rates["a_f"] / 20)
        rates["e_y"] = min(rates["e_y"], rates["b_m"] / 20)
        for s, i in gi.items():
            sel = gidx == i
            pi[i] = np.clip((p[sel].sum() + 0.5) / (sel.sum() + 1), 1e-3, 1 - 1e-3)
        pi_all = float(np.clip((p.sum() + 0.5) / (m + 1), 1e-3, 1 - 1e-3))
    llr = _llr(x, y, n, rates)
    prior = group_prior()
    p = _sigmoid(llr + np.log(prior / (1 - prior)))
    label = np.where(p >= MIN_POST, "male", np.where(p <= 1 - MIN_POST, "female", "ambiguous")).astype(object)
    both = (x >= 3) & (y >= 3) & (x >= 0.3 * rates["a_f"] * n) & (y >= 0.3 * rates["b_m"] * n)
    label[both] = "ambiguous"
    score = np.where(label == "male", p, np.where(label == "female", 1 - p, np.maximum(p, 1 - p))).astype(np.float32)
    score[both] = 0.5
    out_groups: Dict[str, dict] = {}
    informative = (x > 0) | (y > 0)
    for s, i in gi.items():
        sel = gidx == i
        piv = float(pi[i])
        call = "male" if piv >= MIN_POST else ("female" if piv <= 1 - MIN_POST else "ambiguous")
        out_groups[s] = dict(call=call, pi_male=piv, n_cells=int(sel.sum()), n_informative=int((informative & sel).sum()))
    return SexResult(label=label, score=score, p_male=p.astype(np.float32), groups=out_groups, rates={k: float(v) for k, v in rates.items()})
