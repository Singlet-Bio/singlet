# SPDX-License-Identifier: MIT
"""singlet.multimap: the retrieval-time EM and Uniform distribution of the multi-mapping record against a line-by-line port of
STARsolo's per-cell loop (SoloFeature_collapseUMIall.cpp, the multi-gene part of collapseUMIperCB)."""

import numpy as np
import pytest
import scipy.sparse as sp
from singlet.multimap import class_genes, distribute


def star_cell(gEu_full, umis, method):
    """STARsolo for one cell: umis = list of gene tuples (one per molecule); returns {gene: multimapper count}."""
    genes = sorted({g for u in umis for g in u})
    idx = {g: i for i, g in enumerate(genes)}
    ug = [[idx[g] for g in u] for u in umis]
    uni = np.zeros(len(genes))
    for u in ug:
        for g in u:
            uni[g] += 1.0 / len(u)
    if method == "Uniform":
        return {genes[i]: uni[i] for i in range(len(genes))}
    gEu = np.array([gEu_full.get(g, 0.0) for g in genes])
    old = uni + gEu
    it = 0
    while True:
        it += 1
        new = gEu.copy()
        old = np.where(old < 0.01, 0.0, old)
        for u in ug:
            norm = 1.0 / sum(old[g] for g in u)
            for g in u:
                new[g] += old[g] * norm
        if np.max(np.abs(new - old)) < 0.01 or it > 100:
            break
        old = new
    return {genes[i]: new[i] - gEu[i] for i in range(len(genes))}


@pytest.mark.parametrize("method", ["EM", "Uniform"])
def test_matches_star_loop(method):
    rng = np.random.default_rng(7)
    n_cells, n_genes = 40, 30
    unique = sp.random(
        n_cells, n_genes, density=0.3, random_state=3, data_rvs=lambda n: rng.integers(1, 50, n)
    ).tocsr()
    pairs = [(a, b) for a in range(n_genes) for b in range(a + 1, n_genes) if rng.random() < 0.03]
    classes = pairs + [(g,) for g in range(0, n_genes, 7)]
    rec = sp.random(
        n_cells,
        len(classes),
        density=0.15,
        random_state=5,
        data_rvs=lambda n: rng.integers(1, 6, n),
    ).tocsr()
    got = distribute(unique, rec, classes, method).toarray()
    for c in range(n_cells):
        umis = []
        row = rec.getrow(c)
        for k, n in zip(row.indices, row.data):
            umis += [classes[k]] * int(n)
        want = np.zeros(n_genes)
        if umis:
            gEu = {g: float(unique[c, g]) for g in range(n_genes)}
            for g, v in star_cell(gEu, umis, method).items():
                want[g] = v
        np.testing.assert_allclose(got[c], want, rtol=1e-9, atol=1e-9)
        assert got[c].sum() == pytest.approx(
            row.data.sum()
        )  # every molecule is distributed in full


def test_em_follows_unique_prior():
    unique = sp.csr_matrix(np.array([[90.0, 10.0, 0.0]]))
    got = distribute(unique, sp.csr_matrix(np.array([[10.0]])), [(0, 1)], "EM").toarray()[0]
    assert got[0] > 8.9 and got[1] < 1.1 and got[2] == 0
    assert got.sum() == pytest.approx(10.0)


def test_class_genes():
    row = {"ENSG1": 0, "ENSG2": 1}
    assert class_genes(["ENSG2_B|ENSG1_A", "ENSG1_A", "ENSG3_C|ENSG1_A"], row) == [(0, 1), (0,), ()]
