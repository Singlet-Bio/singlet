# SPDX-License-Identifier: MIT
"""Chromosomal sex per cell from expression, pooled per donor.

Evidence per cell: UMIs of XIST (``x``) and summed UMIs of non-PAR chrY genes (``y``). Only these sex-informative UMIs are used,
and only through their split: in a female cell almost all of them are XIST (fraction ``Q_FEMALE``), in a male cell almost none
(``Q_MALE``, the floor set by ambient RNA and mapping noise). The absolute rates are not used, because they vary several-fold
between protocols, cells and nuclei and sequencing depths; and the absence of XIST is not taken as evidence of a Y (shallow cells
detect neither). A cell's log-likelihood ratio male:female is then ``(y - x) * log(Q_FEMALE / Q_MALE)``, clipped to +-``LLR_CAP``.

Cells are pooled per group: the donors of the sample when genotype demultiplexing found more than one
(``donor/donor_assignments.tsv``), otherwise the whole sample. The group's male fraction ``pi`` is fitted by EM over its
informative cells (Jeffreys prior), and each cell's posterior is ``sigmoid(LLR + logit(pi))``. In a single-sex group every cell,
including the many with no XIST or chrY UMI at all, inherits the group's sex unless its own evidence is strongly contrary (a
doublet with another donor, a mislabelled barcode); in a mixed group (pooled donors without demultiplexing) uninformative cells
stay ambiguous.

Labels: ``female`` (PATO:0000383), ``male`` (PATO:0000384), ``ambiguous`` (posterior between ``1 - MIN_POST`` and ``MIN_POST``,
or substantial XIST and chrY in the same cell). ``score`` is the posterior of the label given (0.5 for 'both present').
Loss of chrY in ageing blood cells does not flip a male donor's cells: such cells have neither XIST nor chrY, so the group
decides them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import scipy.sparse as sps

PATO = {"female": "PATO:0000383", "male": "PATO:0000384"}

# Non-PAR chrY genes broadly expressed outside the testis. XIST is the X evidence in both species.
CHRY = {
    "human": (
        "RPS4Y1",
        "DDX3Y",
        "UTY",
        "KDM5D",
        "EIF1AY",
        "ZFY",
        "USP9Y",
        "TXLNGY",
        "NLGN4Y",
        "PRKY",
    ),
    "mouse": ("Ddx3y", "Eif2s3y", "Uty", "Kdm5d"),
}
XIST = {"human": ("XIST",), "mouse": ("Xist",)}

Q_FEMALE = 0.97  # fraction of sex-informative UMIs that are XIST in a female cell
Q_MALE = 0.03  # ... in a male cell
LLR_CAP = 6.0
MIN_POST = 0.9
EM_ITERS = 50


@dataclass
class SexResult:
    label: np.ndarray  # per cell: 'female' | 'male' | 'ambiguous'
    score: np.ndarray  # posterior of the label (float32)
    p_male: np.ndarray  # per-cell posterior P(male)
    groups: Dict[str, dict] = field(
        default_factory=dict
    )  # per group: call, pi_male, n_cells, n_informative, xist_umis, chry_umis


def _genes_index(gene_names: Sequence[str], wanted: Sequence[str]) -> np.ndarray:
    up = {w.upper() for w in wanted}
    return np.array([j for j, g in enumerate(gene_names) if str(g).upper() in up], np.int64)


def sex_evidence(X, gene_names: Sequence[str], species: str = "human"):
    """(x, y) per cell: XIST UMIs and chrY UMIs."""
    X = sps.csr_matrix(X)
    xi = _genes_index(gene_names, XIST[species])
    yi = _genes_index(gene_names, CHRY[species])
    x = np.asarray(X[:, xi].sum(1)).ravel().astype(np.float64) if len(xi) else np.zeros(X.shape[0])
    y = np.asarray(X[:, yi].sum(1)).ravel().astype(np.float64) if len(yi) else np.zeros(X.shape[0])
    return x, y


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _fit_pi(llr: np.ndarray) -> float:
    """Male fraction of a group from its informative cells' LLRs (EM, Jeffreys prior)."""
    if len(llr) == 0:
        return 0.5
    pi = 0.5
    for _ in range(EM_ITERS):
        p = _sigmoid(llr + np.log(pi / (1 - pi)))
        new = (p.sum() + 0.5) / (len(llr) + 1.0)
        new = float(np.clip(new, 1e-4, 1 - 1e-4))
        if abs(new - pi) < 1e-7:
            pi = new
            break
        pi = new
    return pi


def call_sex(
    X,
    gene_names: Sequence[str],
    *,
    species: str = "human",
    groups: Optional[Sequence[Optional[str]]] = None,
) -> SexResult:
    """Per-cell sex. ``groups``: donor label per cell ('' or None for cells not assigned to a donor, which use the pooled
    fraction of the whole sample); None = the whole sample is one group."""
    if species not in CHRY:
        raise ValueError(f"no sex genes for species {species!r}")
    x, y = sex_evidence(X, gene_names, species)
    m = len(x)
    llr = np.clip((y - x) * np.log(Q_FEMALE / Q_MALE), -LLR_CAP, LLR_CAP)
    inf = (x + y) > 0
    g = (
        np.array(["sample"] * m, dtype=object)
        if groups is None
        else np.array([s or "" for s in groups], dtype=object)
    )
    pi_all = _fit_pi(llr[inf])
    prior = np.full(m, pi_all)
    out_groups: Dict[str, dict] = {}
    for s in sorted({v for v in g if v}):
        sel = g == s
        pi = _fit_pi(llr[sel & inf])
        prior[sel] = pi
        call = "male" if pi >= MIN_POST else ("female" if pi <= 1 - MIN_POST else "ambiguous")
        out_groups[s] = dict(
            call=call,
            pi_male=pi,
            n_cells=int(sel.sum()),
            n_informative=int((sel & inf).sum()),
            xist_umis=int(x[sel].sum()),
            chry_umis=int(y[sel].sum()),
        )
    p = _sigmoid(llr + np.log(prior / (1 - prior)))
    label = np.where(
        p >= MIN_POST, "male", np.where(p <= 1 - MIN_POST, "female", "ambiguous")
    ).astype(object)
    both = (x >= 2) & (y >= 2) & (np.minimum(x, y) >= 0.25 * np.maximum(x, y))
    label[both] = "ambiguous"
    score = np.where(
        label == "male", p, np.where(label == "female", 1 - p, np.maximum(p, 1 - p))
    ).astype(np.float32)
    score[both] = 0.5
    return SexResult(label=label, score=score, p_male=p.astype(np.float32), groups=out_groups)
