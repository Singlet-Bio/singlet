# SPDX-License-Identifier: MIT
"""Write the called_ed golden vectors (README.md in this directory describes the layout).

  python -P generate.py synthetic OUT_DIR
  python -P generate.py real NAME RUN_DIR OUT_DIR [--rate R] [--genes G] [--seed S]

``synthetic`` builds the three simulated cases; ``real`` subsamples the stored counts of a Singlet output folder (exon + intron
matrices and ``cell_calls.tsv``, read by bench/called_ed_truth/common.py ``load_run``). The expected values are those of
``singlet.store.cellcall.call_detail`` with default parameters, from the checkout this file lives in.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from singlet.store import cellcall as CC  # noqa: E402

FORMAT = "singlet-called-ed-golden/1"


# ----------------------------------------------------------------------------------------------------------- writing
def knee_unbounded(totals, lower, exclude_from=50):
    """The knee of cellcall 2.0 (spline over every total > ``lower``), kept here only to document the bug each case shows."""
    u, rr = CC._rank_curve(np.asarray(totals), lower)
    if len(u) <= exclude_from + 5:
        return float("nan")
    try:
        return CC._spline_knee(np.log10(rr)[exclude_from:], np.log10(u)[exclude_from:])
    except Exception:
        return float("nan")


def _num(x):
    """JSON-safe scalar: NaN becomes null."""
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _g(x) -> str:
    return repr(float(x))  # shortest round-trip decimal; "nan" for NaN


def write_case(out, name, X, called, description, source, protocol="10xv3", genes=None):
    X = sp.csc_matrix(X).astype(np.int64)
    X.eliminate_zeros()
    X.sort_indices()
    called = np.asarray(called, bool)
    cv2, pv, q, params = CC.call_detail(X, called, protocol)
    tot = np.asarray(X.sum(axis=0)).ravel().astype(np.int64)
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    G, n = X.shape
    with open(os.path.join(d, "counts.mtx"), "w", newline="\n") as fh:
        fh.write("%%MatrixMarket matrix coordinate integer general\n")
        fh.write(f"% {name}: genes x barcodes, gene-level counts (exon + intron); entries sorted by column, then row\n")
        fh.write(f"{G} {n} {X.nnz}\n")
        col = np.repeat(np.arange(n), np.diff(X.indptr))
        fh.write("".join(f"{r + 1} {c + 1} {v}\n" for r, c, v in zip(X.indices, col, X.data)))
    with open(os.path.join(d, "called.txt"), "w", newline="\n") as fh:
        fh.write("".join("1\n" if c else "0\n" for c in called))
    tested = np.nonzero(np.isfinite(q))[0]
    with open(os.path.join(d, "expected.tsv"), "w", newline="\n") as fh:
        fh.write("col\ttotal\tpvalue\tfdr\tcalled_ed\n")
        for j in tested:
            fh.write(f"{j}\t{tot[j]}\t{_g(pv[j])}\t{_g(q[j])}\t{int(cv2[j])}\n")
    lower = params.get("lower")
    keys = ("method", "version", "lower", "retain_total", "alpha", "n_ambient", "ambient_umi", "n_tested", "n_tail",
            "n_called_ed", "n_added", "error")
    case = {
        "format": FORMAT,
        "name": name,
        "description": description,
        "source": source,
        "protocol": protocol,
        "options": {},
        "n_genes": G,
        "n_barcodes": n,
        "n_called": int(called.sum()),
        "expected": {k: _num(params[k]) for k in keys if k in params},
        "knee_unbounded": _num(knee_unbounded(tot, lower)) if lower is not None else None,
        "tolerance": {
            "called_ed": "exact",
            "lower": "exact",
            "pvalue_monte_carlo": "exact (pvalue >= 1/(niters+1): (n_le + 1) / (niters + 1))",
            "pvalue_tail_log_rel": 1e-6,
            "fdr_log_rel": 1e-6,
            "alpha_rel": 1e-6,
            "retain_total_rel": 1e-9,
        },
    }
    if genes is not None:
        case["genes"] = list(genes)
    with open(os.path.join(d, "case.json"), "w", newline="\n") as fh:
        json.dump(case, fh, indent=1, allow_nan=False)
        fh.write("\n")
    e = case["expected"]
    print(f"{name}: {G} genes x {n} barcodes, nnz {X.nnz}, called {int(called.sum())}, lower {e.get('lower')}, "
          f"knee {e.get('retain_total')} (unbounded {case['knee_unbounded']}), tested {e.get('n_tested')}, "
          f"tail {e.get('n_tail')}, called_ed {e.get('n_called_ed')}", flush=True)
    return case


# ----------------------------------------------------------------------------------------------------------- synthetic
def _profiles(rng, G, k):
    amb = rng.dirichlet(np.full(G, 0.3))
    types = [rng.dirichlet(np.full(G, 0.1)) for _ in range(k)]
    return amb, types


def _draw(rng, totals, p):
    return np.array([rng.multinomial(int(t), p) for t in totals]).T


AMBIENT_DM_ALPHA = 300.0


def _draw_ambient(rng, totals, p, a=AMBIENT_DM_ALPHA):
    """Dirichlet-multinomial droplets (concentration ``a``, as in real ambient pools: fitted alphas 44-4,200 in the benchmark
    samples). Pure multinomial ambient would put the alpha MLE at infinity, where the flat likelihood makes the fitted value
    depend on rounding."""
    out = []
    for t in totals:
        g = rng.standard_gamma(a * p)
        out.append(rng.multinomial(int(t), g / g.sum()))
    return np.array(out).T


def synth_knee_below_plateau(rng, G=80):
    """A Kang-like rank curve: ambient at 1-3 UMIs (plateau lower 7), a dense debris band at 8-14 UMIs whose drop-off is the
    sharpest bend among the totals > lower, an ambient shoulder of 15-~150 UMIs, called cells at 1,000-16,000 UMIs and low-RNA
    cells at 200-1,000. The unbounded knee lands at 8, so every tested barcode (the ambient shoulder included) was called."""
    amb, types = _profiles(rng, G, 3)
    parts, called = [], []
    t_amb = rng.integers(1, 4, 4000)
    t_deb = rng.integers(8, 15, 1600)
    t_sh = np.round(15 * np.exp(rng.exponential(1.0, 160))).astype(int)
    t_cell = np.round(10 ** rng.uniform(3, 4.2, 120)).astype(int)
    t_low = np.round(10 ** rng.uniform(2.3, 3, 30)).astype(int)
    # strong overdispersion (alpha 30): with 1-3 UMIs per ambient droplet a weaker one leaves the alpha MLE nearly flat
    parts += [_draw_ambient(rng, t, amb, 30.0) for t in (t_amb, t_deb, t_sh)]
    called += [False] * (len(t_amb) + len(t_deb) + len(t_sh))
    parts.append(np.hstack([_draw(rng, t_cell[i::3], 0.9 * types[i] + 0.1 * amb) for i in range(3)]))
    called += [True] * len(t_cell)
    parts.append(np.hstack([_draw(rng, t_low[i::3], 0.8 * types[i] + 0.2 * amb) for i in range(3)]))
    called += [False] * len(t_low)
    return np.hstack(parts), np.array(called)


def synth_tail(rng, G=40):
    """Ambient at 1-15 UMIs, many ambient-like tested barcodes (800 at 51-120 UMIs) and few uncalled low-RNA cells (30 at
    150-400): the cells sit below every simulated path, but fewer than n_tested / 10 barcodes are at the Monte Carlo floor,
    so BH passes none of them at FDR 0.001 without the tail extrapolation."""
    amb, types = _profiles(rng, G, 3)
    parts, called = [], []
    t_amb = rng.integers(1, 16, 3000)
    t_tst = rng.integers(51, 121, 800)
    t_cell = np.round(10 ** rng.uniform(3.4, 4, 60)).astype(int)
    t_low = rng.integers(150, 401, 30)
    parts += [_draw_ambient(rng, t_amb, amb), _draw_ambient(rng, t_tst, amb)]
    called += [False] * (len(t_amb) + len(t_tst))
    parts.append(np.hstack([_draw(rng, t_cell[i::3], 0.9 * types[i] + 0.1 * amb) for i in range(3)]))
    called += [True] * len(t_cell)
    parts.append(np.hstack([_draw(rng, t_low[i::3], 0.7 * types[i] + 0.3 * amb) for i in range(3)]))
    called += [False] * len(t_low)
    return np.hstack(parts), np.array(called)


def synth_mixed(rng, G=100):
    """The unit-test simulation at small scale: ambient (1-19 UMIs), called cells, uncalled low-RNA cells (150-600 UMIs) of
    three types, plus ambient-like barcodes at 51-200 UMIs; knee and test both active."""
    amb, types = _profiles(rng, G, 3)
    parts, called = [], []
    t_amb = rng.integers(1, 20, 3000)
    t_mid = rng.integers(51, 201, 150)
    t_cell = rng.integers(3000, 20000, 60)
    t_low = rng.integers(150, 601, 60)
    parts += [_draw_ambient(rng, t_amb, amb), _draw_ambient(rng, t_mid, amb)]
    called += [False] * (len(t_amb) + len(t_mid))
    parts.append(np.hstack([_draw(rng, t_cell[i::3], 0.9 * types[i] + 0.1 * amb) for i in range(3)]))
    called += [True] * len(t_cell)
    parts.append(np.hstack([_draw(rng, t_low[i::3], 0.8 * types[i] + 0.2 * amb) for i in range(3)]))
    called += [False] * len(t_low)
    return np.hstack(parts), np.array(called)


def synthetic(out):
    for name, fn, seed in (
        ("synth_knee_below_plateau", synth_knee_below_plateau, 3),
        ("synth_tail_extrapolation", synth_tail, 2),
        ("synth_mixed", synth_mixed, 3),
    ):
        rng = np.random.default_rng(seed)
        X, called = fn(rng)
        write_case(out, name, X, called, " ".join(fn.__doc__.split()),
                   f"tests/data/called_ed_golden/generate.py {fn.__name__}, numpy default_rng({seed})")


# ----------------------------------------------------------------------------------------------------------- real
def real(name, run_dir, out, rate, n_genes, seed):
    """Subsample a stored sample: a uniform ``rate`` share of the nonzero barcodes (keeps the shape of the log-log rank curve),
    the ``n_genes`` genes with the most counts over the kept uncalled barcodes plus one row ``OTHER`` holding every other gene
    (keeps totals). Called barcodes enter the call only through their totals, so their counts are stored as one entry in the
    ``OTHER`` row."""
    sys.path.insert(0, os.path.join(ROOT, "bench", "called_ed_truth"))
    os.environ.setdefault("CELLCALL_DIR", os.path.join(ROOT, "python", "singlet", "store"))
    import common as K

    R = K.load_run(run_dir)
    X, called = R["X"].tocsc(), R["called"]
    tot = np.asarray(X.sum(axis=0)).ravel()
    rng = np.random.default_rng(seed)
    nz = np.nonzero(tot > 0)[0]
    keep = np.sort(nz[rng.random(len(nz)) < rate])
    Xk, ck = X[:, keep].tocsc(), called[keep]
    gsum = np.asarray(Xk[:, ~ck].sum(axis=1)).ravel()
    top = np.sort(np.argsort(-gsum, kind="stable")[:n_genes])
    rest = np.setdiff1d(np.arange(X.shape[0]), top)
    U = sp.vstack([Xk[top], sp.csr_matrix(np.asarray(Xk[rest].sum(axis=0)))]).tocsc()
    tk = np.asarray(Xk.sum(axis=0)).ravel()
    C = sp.csc_matrix((tk[ck], (np.full(int(ck.sum()), len(top)), np.nonzero(ck)[0])), shape=U.shape)
    U = U.tolil()
    U[:, np.nonzero(ck)[0]] = 0
    U = (U.tocsc() + C).tocsc()
    assert (np.asarray(U.sum(axis=0)).ravel() == tk).all()
    genes = [str(g) for g in R["genes"][top]] + ["OTHER"]
    desc = (f"{name}: {rate:g} of the {len(nz)} nonzero barcodes of the stored exon + intron counts (seed {seed}), the "
            f"{n_genes} genes with most counts in the kept uncalled barcodes + OTHER; called barcodes as their total in OTHER")
    write_case(out, name, U, ck, desc, os.path.basename(os.path.normpath(run_dir)) + " (Clipper, " + run_dir + ")",
               genes=genes)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("out")
    r = sub.add_parser("real")
    r.add_argument("name")
    r.add_argument("run_dir")
    r.add_argument("out")
    r.add_argument("--rate", type=float, default=0.2)
    r.add_argument("--genes", type=int, default=300)
    r.add_argument("--seed", type=int, default=20261025)
    a = ap.parse_args()
    if a.mode == "synthetic":
        synthetic(a.out)
    else:
        real(a.name, a.run_dir, a.out, a.rate, a.genes, a.seed)


if __name__ == "__main__":
    main()
