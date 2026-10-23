# SPDX-License-Identifier: MIT
"""singlet.annotate models on synthetic data: guide compilation, gating, sex pooling per donor, ancestry recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps

from singlet.annotate import ancestry as ANC
from singlet.annotate import celltype as CT
from singlet.annotate import evaluate as EV
from singlet.annotate import guide as G
from singlet.annotate import sex as SX


def test_guide_compiles_for_both_species():
    for sp in ("human", "mouse"):
        g = G.Guide(["CD3E", "Cd3e", "MS4A1", "Ms4a1"], species=sp)
        assert g.ids[0] == "root" and g.cl[0] == "CL:0000000"
        assert len(g.nodes) > 200
        assert (g.depth[1:] >= 1).all()
        assert all(c.startswith("CL:") for c in g.cl)


def test_tissue_context_restricts_nodes():
    full = G.Guide(["CD3E"], species="human")
    blood = G.Guide(["CD3E"], species="human", tissue="peripheral blood")
    assert len(blood.nodes) < len(full.nodes)
    assert "t_cell" in blood.index
    assert G.tissue_groups("adrenal gland") == {"adrenal"}  # 'renal' only at a word start
    assert G.tissue_groups("temporal cortex") == {"brain"}


def test_every_guide_cl_id_is_in_the_ontology():
    onto = G.ontology()
    for n in G.load_nodes():
        assert n["cl"] in onto.terms, n["id"]


def _two_lineages(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    markers = sorted({m for nd in G.load_nodes() for m in (nd["markers"].get("pos") or {})})
    genes = markers + ["XIST", "RPS4Y1", "DDX3Y"] + [f"G{i}" for i in range(1500)]
    gi = {g: i for i, g in enumerate(genes)}
    X = rng.poisson(0.1, (n, len(genes))).astype(np.float32)
    h = n // 2
    for g in ("PTPRC", "CD3E", "CD3D", "CD2", "TRAC", "IL7R", "CD8A", "CD8B"):
        X[:h, gi[g]] += rng.poisson(6, h)
    for g in ("PTPRC", "CD19", "MS4A1", "CD79A", "CD79B", "BANK1"):
        X[h:, gi[g]] += rng.poisson(6, n - h)
    # real lineages differ in hundreds of genes, which is what the neighbour graph sees
    X[:h, -1500:-1300] += rng.poisson(1.0, (h, 200))
    X[h:, -1300:-1100] += rng.poisson(1.0, (n - h, 200))
    return X, genes, h


@pytest.mark.parametrize("method", ["guide", "selftrain"])
def test_cell_types_separate_t_and_b(method):
    X, genes, h = _two_lineages()
    r = CT.annotate_cell_types(X, genes, species="human", tissue="blood", method=method)
    t = r.guide.index["t_cell"]
    b = r.guide.index["b_cell"]
    in_t = np.array([t in r.guide.path(int(j)) for j in r.node])
    in_b = np.array([b in r.guide.path(int(j)) for j in r.node])
    # T-like calls include innate-like T nodes (MAIT) that sit beside t_cell in the guide; B and T must not mix
    assert (~in_b[:h]).mean() > 0.95 and in_b[h:].mean() > 0.9 and in_t[:h].mean() > 0.8 and (~in_t[h:]).mean() > 0.95
    assert np.isfinite(r.score[r.node > 0]).all() and ((r.score[r.node > 0] >= 0) & (r.score[r.node > 0] <= 1)).all()
    rep = EV.report(r.guide, ["CL:0000084"] * h + ["CL:0000236"] * (len(r.node) - h), r.node, r.score)
    assert rep["by_depth"][2]["accuracy"] > 0.9


def test_empty_cells_get_no_call():
    X, genes, _ = _two_lineages(200)
    X[:10] = 0
    r = CT.annotate_cell_types(X, genes, species="human", method="guide")
    assert (r.node[:10] == 0).all() and np.isnan(r.score[:10]).all()
    assert all(c is None for c in r.cl[:10])


def _sex_counts(n_f, n_m, seed=0, depth=3000):
    rng = np.random.default_rng(seed)
    genes = ["XIST", "RPS4Y1", "DDX3Y", "UTY", "A", "B"]
    n = n_f + n_m
    lib = rng.integers(depth // 2, depth * 2, n)
    X = np.zeros((n, len(genes)), np.float32)
    X[:, 4] = lib
    X[:n_f, 0] = rng.poisson(lib[:n_f] * 2e-3)
    X[n_f:, 1] = rng.poisson(lib[n_f:] * 4e-4)
    X[n_f:, 2] = rng.poisson(lib[n_f:] * 1e-4)
    return sps.csr_matrix(X), genes


def test_sex_single_donor_sample_inherits_group_call():
    X, genes = _sex_counts(500, 0, depth=800)  # shallow: many cells with zero XIST
    r = SX.call_sex(X, genes)
    assert r.groups["sample"]["call"] == "female"
    assert (r.label == "female").mean() > 0.97


def test_sex_pooled_per_donor():
    X, genes = _sex_counts(300, 300)
    donors = ["donor0"] * 300 + ["donor1"] * 300
    r = SX.call_sex(X, genes, groups=donors)
    assert r.groups["donor0"]["call"] == "female" and r.groups["donor1"]["call"] == "male"
    assert (r.label[:300] == "female").mean() > 0.97 and (r.label[300:] == "male").mean() > 0.97
    # without donor labels, a mixed sample is left to each cell's own evidence
    r2 = SX.call_sex(X, genes)
    assert r2.groups["sample"]["call"] == "ambiguous"
    assert (r2.label[:300] != "male").mean() > 0.9 and (r2.label[300:] != "female").mean() > 0.9


def _af(m, rng):
    F = rng.beta(0.6, 0.6, (m, 5)) * 0.9 + 0.05
    return F


@pytest.mark.parametrize("pop", list(ANC.POPS))
def test_ancestry_recovers_population(pop):
    rng = np.random.default_rng(ANC.POPS.index(pop))
    m = 4000
    F = _af(m, rng)
    g = rng.binomial(2, F[:, ANC.POPS.index(pop)])
    dp = rng.integers(1, 12, m)
    ad = rng.binomial(dp, np.clip(g / 2, 0.01, 0.99))
    c = ANC.infer_ancestry(ad, dp, F, n_boot=5)
    assert c.group == pop and c.prob is not None and c.prob > 0.8
    assert c.hancestro == ANC.HANCESTRO[pop][0]
    assert c.stability == 1.0


def test_ancestry_admixed_and_low_data():
    rng = np.random.default_rng(7)
    m = 6000
    F = _af(m, rng)
    f = 0.5 * F[:, ANC.POPS.index("AFR")] + 0.5 * F[:, ANC.POPS.index("EUR")]
    g = rng.binomial(2, f)
    dp = rng.integers(2, 10, m)
    ad = rng.binomial(dp, np.clip(g / 2, 0.01, 0.99))
    c = ANC.infer_ancestry(ad, dp, F, n_boot=0)
    assert c.group == "admixed" and c.hancestro == "HANCESTRO:0306"
    assert 0.3 < c.q["AFR"] < 0.7 and 0.3 < c.q["EUR"] < 0.7
    few = ANC.infer_ancestry(ad[:50], dp[:50], F[:50])
    assert few.group is None and few.hancestro is None


def test_align_sites_flips_swapped_alleles():
    af = pd.DataFrame({"chrom": ["1", "1"], "pos": [100, 200], "ref": ["A", "C"], "alt": ["G", "T"],
                       **{p: [0.1, 0.2] for p in ANC.POPS}})
    sites = pd.DataFrame({"chrom": ["chr1", "chr1"], "pos": [100, 200], "ref": ["A", "T"], "alt": ["G", "C"], "ad": [3, 1], "dp": [5, 4]})
    ad, dp, F = ANC.align_sites(sites, af)
    assert sorted(zip(ad.tolist(), dp.tolist())) == [(3, 4), (3, 5)]
    assert F.shape == (2, 5)
