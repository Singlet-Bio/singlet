# SPDX-License-Identifier: MIT
"""singlet.crispr: protospacer geometry, counting through the ADT core, and guide assignment on simulated cells."""

import numpy as np
import pytest
from scipy import sparse
from singlet.adt import seq as S
from singlet.crispr import assign as AS
from singlet.crispr import count as CC
from singlet.crispr import guides as G

RNG = np.random.default_rng(3)


def rand(n):
    return "".join(RNG.choice(list("ACGT"), n))


def test_parse_pattern():
    g = G.parse_pattern("(BC)GTTTAAGAGCTAAGCTGGAAACAGCATAGCAAGTT", 20)
    assert g.anchor == "GTTTAAGAGCTA" and g.offset is None
    assert G.parse_pattern("^NNNNNNNNNN(BC)", 20).offset == 10
    assert G.parse_pattern("5PNNNNNNNNNN(BC)", 20).offset == 10
    with pytest.raises(ValueError):
        G.parse_pattern("(BC)(BC)", 20)


def test_extract_and_detect():
    guides = [rand(20) for _ in range(10)]
    sc = G.SCAFFOLDS["cr_default"]
    reads = [(rand(RNG.integers(5, 15)) + guides[i % 10] + sc + rand(30)).encode()[:90] for i in range(500)]
    det = G.detect_geometry(reads, 20, guides)
    g = det["geometry"]
    assert g.anchor == sc[:12] and not g.reverse
    assert det["guide_rate"] > 0.99
    assert G.extract(reads[3], g).decode() == guides[3]
    rcr = [G.rc(r) for r in reads]
    assert G.detect_geometry(rcr, 20, guides)["geometry"].reverse


def test_count_library_variable_lengths():
    guides = [rand(20) for _ in range(6)] + [rand(19) for _ in range(2)]
    lib = G.from_table([{"id": f"g{i}", "sequence": s, "target_gene": f"T{i}"} for i, s in enumerate(guides)], "t")
    cells = S.decode(RNG.integers(0, 4 ** 16, 30, dtype=np.uint64), 16)
    truth = {c: i % len(guides) for i, c in enumerate(cells)}
    sc = G.SCAFFOLDS["cr_default"]
    pairs = []
    for c, gi in truth.items():
        for _ in range(int(RNG.integers(3, 8))):
            umi = rand(12)
            for _ in range(int(RNG.integers(1, 4))):
                pairs.append(((c + umi).encode(), (rand(int(RNG.integers(3, 12))) + guides[gi] + sc).encode()))
    wl, _ = S.encode_strings(cells, 16)
    lc = CC.count_library([], [], library=lib, geometry=G.Geometry(20, anchor=sc[:12]), cb_len=16, umi_len=12,
                          whitelist_codes=np.sort(wl), gex_barcodes=cells, pairs_factory=lambda: iter(pairs))
    M = lc.matrix.toarray()
    assert M.argmax(axis=0).tolist() == [truth[c] for c in cells]
    assert (M.sum(axis=0) > 0).all()


def simulate_cells(n=3000, G_=40, moi_multi=0.08, amb_frac=0.04, seed=0):
    rng = np.random.default_rng(seed)
    abund = rng.lognormal(0, 1.0, G_)
    abund /= abund.sum()
    truth = np.zeros((G_, n), bool)
    X = np.zeros((G_, n), np.int64)
    for j in range(n):
        u = rng.random()
        k = 0 if u < 0.15 else (2 if u < 0.15 + moi_multi else 1)
        gs = rng.choice(G_, k, replace=False, p=abund) if k else []
        tot = int(rng.lognormal(3.5, 0.8))
        for g in gs:
            truth[g, j] = True
            X[g, j] += rng.poisson(tot / max(1, k))
        X[:, j] += rng.poisson(amb_frac * tot * abund * G_ / 4 + 0.02)
    amb = rng.poisson(5000 * abund)
    # empty droplets: ambient only
    empties = rng.poisson(amb_frac * 30 * abund.sum() * G_ / 4 + 0.02 * G_, 5000)
    return sparse.csc_matrix(X), truth, amb, empties


def _f1(C, truth):
    C = C.toarray()
    tp = (C & truth).sum()
    return 2 * tp / (C.sum() + truth.sum())


def test_callers_recover_truth():
    X, truth, amb, emp = simulate_cells()
    p = AS.ambient_profile(amb)
    f_amb = _f1(AS.call_ambient(X, p, AS.ambient_load(emp)), truth)
    f_gmm = _f1(AS.call_gmm(X), truth)
    f_pg = _f1(AS.call_poisson_gaussian(X), truth)
    assert f_amb > 0.95, (f_amb, f_gmm, f_pg)
    assert f_amb >= max(f_gmm, f_pg) - 0.01, (f_amb, f_gmm, f_pg)
    lab = AS.summarize(AS.call_ambient(X, p, AS.ambient_load(emp)), [f"g{i}" for i in range(X.shape[0])])
    assert set(lab["class"]) <= {"unassigned", "single", "multiple"}


def test_two_guide_library_keeps_doublets():
    """A375-like: two guides, ~8% doublets carrying both. The tested guide must not count in its own background."""
    X, truth, amb, emp = simulate_cells(n=3000, G_=2, moi_multi=0.08, amb_frac=0.02, seed=4)
    called = AS.call_ambient(X, AS.ambient_profile(amb), AS.ambient_load(emp)).toarray()
    both_true = truth.all(axis=0)
    assert both_true.sum() > 100
    assert (called.all(axis=0) & both_true).sum() >= 0.8 * both_true.sum()
    assert _f1(sparse.csc_matrix(called), truth) > 0.95
