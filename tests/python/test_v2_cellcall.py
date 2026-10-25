# SPDX-License-Identifier: MIT
"""called_v2 (singlet.v2.cellcall): gene aggregation, ambient bound, knee, and the EmptyDrops call on a simulated sample with
known low-RNA cells. The Monte Carlo tests need numba (the ``migrate`` extra) and are skipped without it."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from singlet.v2 import cellcall as CC  # noqa: E402
from singlet.v2.matrix import Csc  # noqa: E402


def csc_of(M: np.ndarray, rownames, colnames) -> Csc:
    S = sp.csc_matrix(M)
    S.sort_indices()
    return Csc(
        m=S.shape[0],
        n=S.shape[1],
        indptr=S.indptr.astype(np.int64),
        indices=S.indices.astype(np.uint32),
        data=S.data.astype(np.uint32),
        rownames=list(rownames),
        colnames=list(colnames),
    )


def simulate(seed=0, n_amb=20000, n_big=800, n_low=400, G=1500):
    """Ambient droplets (1-40 UMIs), large called cells and uncalled low-RNA cells (150-600 UMIs) of three types."""
    rng = np.random.default_rng(seed)
    amb = rng.dirichlet(np.full(G, 0.3))
    types = [rng.dirichlet(np.full(G, 0.1)) for _ in range(3)]
    cols, called, low = [], [], []
    for _ in range(n_amb):
        cols.append(rng.multinomial(rng.integers(1, 40), amb))
        called.append(False)
        low.append(False)
    for i in range(n_big):
        cols.append(rng.multinomial(rng.integers(3000, 20000), 0.9 * types[i % 3] + 0.1 * amb))
        called.append(True)
        low.append(False)
    for i in range(n_low):
        cols.append(rng.multinomial(rng.integers(150, 600), 0.8 * types[i % 3] + 0.2 * amb))
        called.append(False)
        low.append(True)
    M = np.array(cols).T.astype(np.int64)
    E = np.floor(M * 0.7).astype(np.int64)
    return E, M - E, np.array(called), np.array(low)


def test_protocol_family():
    assert CC.protocol_family("10xv3") == ("droplet", True)
    assert CC.protocol_family("10x-3p-v2") == ("droplet", True)
    assert CC.protocol_family("microwell") == ("droplet", True)
    assert CC.protocol_family("parse") == ("combinatorial", False)
    assert CC.protocol_family("SPLiT-seq") == ("combinatorial", False)
    assert CC.protocol_family("indrop") == ("indrop", False)
    assert CC.protocol_family("Drop-seq") == ("dropseq", False)


def test_gene_counts_aggregates_features_and_aligns_columns():
    ex = csc_of(
        np.array([[1, 0, 2], [0, 3, 0], [4, 0, 0]]),
        ["G1_A___chr1:1-2", "G1_A___chr1:5-9", "G2_B___chr2:1-3"],
        ["AAA", "CCC", "GGG"],
    )
    # the intron matrix lists the barcodes in another order and lacks one
    it = csc_of(
        np.array([[5, 0], [0, 7]]), ["G1_A_intron_chr1:3-4", "G3_C_intron_chr3:1-9"], ["GGG", "AAA"]
    )
    X = CC.gene_counts(ex, it, ["AAA", "CCC", "GGG"]).toarray()
    # genes sorted: G1, G2, G3; AAA: G1 1, G2 4, G3 7; CCC: G1 3; GGG: G1 2 + 5
    assert X.tolist() == [[1, 3, 7], [4, 0, 0], [7, 0, 0]]


def test_plateau_lower_sits_on_the_ambient_plateau():
    rng = np.random.default_rng(1)
    tot = np.concatenate([rng.integers(1, 12, 30000), rng.integers(2000, 9000, 1000)])
    called = tot >= 2000
    lo = CC.plateau_lower(tot, called)
    assert 5 <= lo <= 40
    # no ambient barcodes at all: falls back to half the called 5th percentile, clipped
    assert CC.plateau_lower(np.full(500, 5000), np.ones(500, bool)) == 100


def test_knee_between_ambient_and_cells():
    rng = np.random.default_rng(2)
    tot = np.concatenate([rng.integers(1, 30, 20000), rng.integers(3000, 10000, 2000)])
    k = CC.knee(tot, 30)
    assert 30 < k <= 10000


def test_bh():
    q = CC.bh(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(
        q, [0.04, 0.16 / 3, 0.16 / 3, 0.5]
    )  # sorted 0.01, 0.03, 0.04: 0.04, 0.06 -> 0.0533, 0.0533


def test_combinatorial_protocol_keeps_the_stored_call():
    E, In, called, _ = simulate(n_amb=200, n_big=20, n_low=10, G=100)
    cv2, fdr, params = CC.call(sp.csc_matrix(E + In), called, "parse")
    assert (cv2 == called).all() and np.isnan(fdr).all() and params["method"] == "stored-call"


@pytest.fixture(scope="module")
def sim_call():
    pytest.importorskip("numba")
    E, In, called, low = simulate()
    X = sp.csc_matrix(E + In)
    return X, called, low, CC.call(X, called, "10xv3", niters=2000)


def test_call_recovers_low_rna_cells_and_rejects_ambient(sim_call):
    X, called, low, (cv2, fdr, params) = sim_call
    assert cv2[called].all()  # never drops a stored cell
    assert cv2[low].mean() > 0.95
    amb = ~called & ~low
    assert cv2[amb].mean() < 0.001
    assert (
        params["validated"]
        and params["n_called_v2"] == int(cv2.sum())
        and params["n_added"] == int((cv2 & ~called).sum())
    )
    assert params["alpha"] > 0 and params["n_ambient"] > 1000 and params["lower"] >= 5
    # fdr: NaN exactly where not tested (stored cells, ambient pool, below the floor); flagged ones at <= 0.001
    tot = np.asarray(X.sum(axis=0)).ravel()
    tested = ~called & (tot > max(params["lower"], params["test_min"]))
    assert np.isnan(fdr[~tested]).all() and np.isfinite(fdr[tested]).all()
    assert (fdr[cv2 & ~called] <= 0.001).all()


def test_call_is_deterministic(sim_call):
    X, called, _, (cv2, fdr, params) = sim_call
    cv2b, fdrb, paramsb = CC.call(X, called, "10xv3", niters=2000)
    assert (cv2 == cv2b).all() and np.array_equal(fdr, fdrb, equal_nan=True) and params == paramsb


def test_tiny_sample_records_an_error_and_keeps_the_call():
    pytest.importorskip("numba")
    E, In, called, _ = simulate(n_amb=5, n_big=30, n_low=5, G=50)
    cv2, fdr, params = CC.call(sp.csc_matrix(E + In), called, "10xv3", niters=200)
    assert (cv2 == called).all() and "error" in params


# ----------------------------------------------------------------------------------------------------------- golden vectors
# Shared with the C++ port (include/singlet/pileup/called_v2.h); layout in tests/data/called_v2_golden/README.md.
GOLDEN = ROOT / "tests" / "data" / "called_v2_golden"
GOLDEN_CASES = (
    sorted(p.name for p in GOLDEN.iterdir() if (p / "case.json").is_file()) if GOLDEN.is_dir() else []
)


def read_golden(name):
    import json

    d = GOLDEN / name
    case = json.loads((d / "case.json").read_text())
    with open(d / "counts.mtx") as fh:
        line = fh.readline()
        assert line.startswith("%%MatrixMarket matrix coordinate integer general")
        while line.startswith("%"):
            line = fh.readline()
        m, n, nnz = (int(v) for v in line.split())
        E = np.loadtxt(fh, dtype=np.int64, ndmin=2) if nnz else np.zeros((0, 3), np.int64)
    assert E.shape == (nnz, 3)
    X = sp.csc_matrix((E[:, 2], (E[:, 0] - 1, E[:, 1] - 1)), shape=(m, n))
    called = np.loadtxt(d / "called.txt", dtype=np.int64, ndmin=1).astype(bool)
    assert len(called) == n == case["n_barcodes"] and m == case["n_genes"]
    exp = np.genfromtxt(
        d / "expected.tsv", delimiter="\t", names=True, dtype=None, encoding=None, ndmin=1
    )
    return case, X, called, exp


def test_golden_vectors_exist():
    synth = {"synth_knee_below_plateau", "synth_tail_extrapolation", "synth_mixed"}
    assert synth <= set(GOLDEN_CASES)
    assert len(set(GOLDEN_CASES) - synth) >= 1  # real subsampled samples


@pytest.mark.parametrize("name", GOLDEN_CASES)
def test_golden_vector(name):
    pytest.importorskip("numba")
    case, X, called, exp = read_golden(name)
    cv2, pv, q, params = CC.call_detail(X, called, case["protocol"], **case["options"])
    e, tol = case["expected"], case["tolerance"]
    niters = params["niters"]
    assert params["lower"] == e["lower"]
    assert (params["retain_total"] is None) == (e["retain_total"] is None)
    if e["retain_total"] is not None:
        assert params["retain_total"] == pytest.approx(e["retain_total"], rel=tol["retain_total_rel"])
    assert params["alpha"] == pytest.approx(e["alpha"], rel=tol["alpha_rel"])
    for k in ("n_ambient", "ambient_umi", "n_tested", "n_tail", "n_called_v2", "n_added"):
        assert params[k] == e[k], k
    # tested barcodes: exactly the listed columns
    tested = np.nonzero(np.isfinite(q))[0]
    cols = exp["col"].astype(np.int64)
    assert np.array_equal(tested, cols)
    tot = np.asarray(X.sum(axis=0)).ravel()
    assert np.array_equal(tot[cols], exp["total"])
    # called set: exact
    expect_cv2 = called.copy()
    expect_cv2[cols[exp["called_v2"].astype(bool)]] = True
    assert np.array_equal(cv2, expect_cv2)
    # p-values: Monte Carlo values (n_le + 1) / (niters + 1) exactly, the tail extrapolation to a relative tolerance
    ep, gp = exp["pvalue"], pv[cols]
    mc = ep >= 1.0 / (niters + 1)
    assert np.array_equal(np.round(gp[mc] * (niters + 1)), np.round(ep[mc] * (niters + 1)))
    assert np.allclose(gp[~mc], ep[~mc], rtol=tol["pvalue_tail_rel"], atol=0)
    assert np.allclose(q[cols], exp["fdr"], rtol=tol["fdr_rel"], atol=0)
    # call() agrees (FDR as float32, NaN = not tested)
    cv2b, fdrb, _ = CC.call(X, called, case["protocol"], **case["options"])
    assert np.array_equal(cv2b, cv2) and np.array_equal(fdrb, q.astype(np.float32), equal_nan=True)


def test_golden_knee_case_shows_the_unbounded_knee_bug():
    case, X, called, _ = read_golden("synth_knee_below_plateau")
    tot = np.asarray(X.sum(axis=0)).ravel()
    lower = CC.plateau_lower(tot, called)
    # the 2.0 knee (spline over all totals > lower) sat just above lower: every tested barcode would be retained untested
    assert case["knee_unbounded"] < 2 * lower
    assert CC.knee(tot, lower) > 50 * lower
    tested = ~called & (tot > max(lower, 50))
    assert (tot[tested] >= case["knee_unbounded"]).all()


def test_golden_tail_case_needs_the_extrapolation():
    case, _, _, exp = read_golden("synth_tail_extrapolation")
    e = case["expected"]
    # fewer barcodes at the Monte Carlo floor than BH needs at FDR 0.001 (k / n_tested >= 1 / (0.001 * 10001)) ...
    assert e["n_tail"] * 0.001 * 10001 < e["n_tested"]
    # ... yet the tail p-values let BH call them
    added = exp["called_v2"].astype(bool)
    assert e["n_added"] > 0 and (exp["pvalue"][added] < 1 / 10001).all()


def test_knee_uses_the_curve_bounds():
    rng = np.random.default_rng(5)
    # ambient at 1-3, a debris band at 8-14 (its drop-off is the sharpest bend above lower = 7), cells at 1,000-16,000
    tot = np.concatenate(
        [
            rng.integers(1, 4, 20000),
            rng.integers(8, 15, 8000),
            np.round(10 ** rng.uniform(3, 4.2, 600)).astype(int),
        ]
    )
    b = CC.curve_bounds(tot, 7)
    u, _ = CC._rank_curve(tot, 7)
    assert u[b[0]] > u[b[1]] >= 8
    assert CC.knee(tot, 7) > 500
