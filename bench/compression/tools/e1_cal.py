#!/usr/bin/env python
"""E1 extension: a stronger *sequential* (own-half) predictor than the global-alpha Dirichlet.

    python -P tools/e1_cal.py run <sample> <member> [--res interval|gene]      (after e1_split.py run for the same file)
    python -P tools/e1_cal.py report                                          -> results/e1_cal_report.md

Why: a Dirichlet / Dirichlet-multinomial with ONE alpha treats every feature as equally "bursty". Sparse genomic
counts are not: housekeeping features are close to Poisson, rare / cell-type-specific features are very bursty.
L3cal learns the A -> B map from OTHER cells instead of assuming it:  for each feature f of the query cell, with
u = N_A q_f (expected A count under L1) and a = observed A count,

        E[b_f] = N_B q_f rho(bin(u), class(a / u))

where rho is the empirical ratio  sum b / sum (N_B q)  over (cell, feature) pairs of training cells that fall in the same
(expression-level bin, enrichment-class) table cell (class 0 = a_f = 0, else floor(log2(a/u))), shrunk toward the pooled
(over u) ratio of its class.  Training uses the A/B halves of pop cells in the OTHER 2-fold half than the scored cell
(never the scored cell itself); q is the same leave-one-out L1 used everywhere.  The predictor is normalised to sum to 1
over all features (exactly, via a sorted-q histogram) and scored like L1-L3 (multiset + sparse).

CONFOUND CONTROL.  The table also repairs the *marginal* calibration of L1 (e.g. features never seen in the other cells
carry far more mass than the additive floor delta gives them).  That is a better static model, not state information.
So we also fit the a-free version  L1cal:  E[b_f] = N_B q_f rho_m(bin(u))  and attribute
    L1cal - L1            -> static recalibration
    L2cal - L1cal         -> neighbours   (L2cal = (1-w) L1cal + w neighbour mean, 16 priors)
    L3cal - L1cal         -> the cell's own half (sequential)
    mix   - L1cal         -> both: (1-lam) L3cal + lam L2cal, lam in {.25,.5,.75}
"""
import os, sys, time, math, argparse, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1_split as E

UEDGES = 10.0 ** np.arange(-7.0, 2.01, 0.5)          # 19 edges -> 18 bins of u = N_A q
NU = len(UEDGES) - 1
RLO, RHI = -4, 16                                     # floor(log2(a/u)) clipped to [RLO, RHI]
NCLS = 1 + (RHI - RLO + 1)                            # class 0 = a == 0
KAPPA = 3.0
LAMS = [0.25, 0.5, 0.75]


def ubin(u):
    return np.clip(np.searchsorted(UEDGES, u, side="right") - 1, 0, NU - 1)


def rcls(a, u):
    """class index for a > 0 (1..NCLS-1)"""
    r = np.floor(np.log2(a / u)).astype(np.int64)
    return 1 + np.clip(r, RLO, RHI) - RLO


class Cal:
    def __init__(self, S):
        self.S = S
        qt = (S.P + S.d1) / (S.T + S.F * S.d1)            # global smoothed profile (off-support approximation of q1)
        self.qt = qt
        o = np.argsort(qt)
        self.qs = qt[o]
        self.cs = np.concatenate(([0.0], np.cumsum(self.qs)))

    def qdense(self, NA):
        """mass of the global profile per u-bin, u = N_A q"""
        idx = np.searchsorted(self.qs, UEDGES / NA, side="left")
        idx[0], idx[-1] = 0, len(self.qs)
        return self.cs[idx[1:]] - self.cs[idx[:-1]]

    def cell_q1(self, p):
        S = self.S
        s, e = S.pptr[p], S.pptr[p + 1]
        fi = S.pidx[s:e]
        x = S.pval[s:e]
        N = S.Npop[p]
        q0b = (S.P[fi] - x + S.d1) / (S.T - N + S.F * S.d1)
        bk = S.bucket[p]
        q1 = (S.Pb[bk][fi] - x + S.tau * q0b) / (S.Tb[bk] - N + S.tau)
        return fi, x, q1, N

    def train(self, cells):
        S = self.S
        numer = np.zeros((NU, NCLS))
        denom = np.zeros((NU, NCLS))
        for p in cells:
            fi, x, q1, N = self.cell_q1(p)
            s, e = S.pptr[p], S.pptr[p + 1]
            a = S.Afull[s:e]
            b = x - a
            NA = a.sum()
            NB = N - NA
            if NA <= 0 or NB <= 0:
                continue
            Qd = self.qdense(NA)
            pos = a > 0
            # a > 0 features: exact q
            u = NA * q1[pos]
            bi = ubin(u)
            ci = rcls(a[pos], u)
            np.add.at(numer, (bi, ci), b[pos])
            np.add.at(denom, (bi, ci), NB * q1[pos])
            # a == 0 features: support ones exact, the rest from the dense histogram minus the whole support
            z0 = ~pos
            bz = ubin(NA * q1[z0])
            np.add.at(numer[:, 0], bz, b[z0])
            np.add.at(denom[:, 0], bz, NB * q1[z0])
            qtp = self.qt[fi]
            sub = np.bincount(ubin(NA * qtp), weights=qtp, minlength=NU)
            denom[:, 0] += NB * np.maximum(Qd - sub, 0.0)
        tn, td = numer.sum(0), denom.sum(0)
        marg = np.where(td > 0, tn / np.maximum(td, 1e-300), 1.0)
        # classes with little pooled evidence (< 20 expected UMIs) borrow the pooled ratio of the next richer-evidence lower class
        cn = cd = 0.0
        grp = []
        for c in range(NCLS - 1, 0, -1):
            cn, cd = cn + tn[c], cd + td[c]
            grp.append(c)
            if cd >= 20.0 or c == 1:
                r = cn / cd if cd > 0 else 1.0
                for g in grp:
                    marg[g] = r
                cn = cd = 0.0
                grp = []
        rho = np.maximum((numer + KAPPA * marg[None, :]) / (denom + KAPPA), 1e-3)
        mall = numer.sum() / denom.sum()
        rho_m = (numer.sum(1) + KAPPA * mall) / (denom.sum(1) + KAPPA)
        return rho, rho_m, numer, denom

    def predict_marg(self, rho_m, fi, q1, NA):
        """L1cal: a-free recalibration by expression level only (normalised over the whole universe)."""
        Qd = self.qdense(NA)
        qtf = self.qt[fi]
        rho_f = rho_m[ubin(NA * q1)]
        Z = (rho_m * Qd).sum() + (q1 * rho_f - qtf * rho_m[ubin(NA * qtf)]).sum()
        return q1 * rho_f / Z

    def predict(self, rho, fi, a, q1, NA):
        """L3cal at all support features (normalised over the whole universe)."""
        Qd = self.qdense(NA)
        qtf = self.qt[fi]
        pos = a > 0
        cls = np.zeros(len(fi), dtype=np.int64)
        u = NA * q1
        bi = ubin(u)
        cls[pos] = rcls(a[pos], u[pos])
        rho_f = rho[bi, cls]
        rho0_t = rho[ubin(NA * qtf), 0]
        Z = (rho[:, 0] * Qd).sum() + (q1 * rho_f - qtf * rho0_t).sum()
        return q1 * rho_f / Z


def run(sample, member, res, cap, seed, log):
    d = E.load_matrix(sample, member)
    gmap = E.gene_map(sample, member) if res == "gene" else None
    ptr, ind, val, M = E.to_resolution(d, gmap)
    is_cell = d["is_cell"]
    n = len(ptr) - 1
    col = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
    Nall = np.bincount(col, weights=val.astype(np.float64), minlength=n)
    del col
    cols = np.flatnonzero(is_cell & (Nall >= E.N_MIN_CELL))
    main = E.load_percell(os.path.join(E.RES, f"e1_percell_{sample}__{member}__{res}__cells.npz"))
    S = E.Subset("cells", ptr, ind, val, M, cols, E.N_MIN_CELL, cap, seed, log, use_nbr=True)
    assert np.array_equal(S.sidx, main["pop"]), "scored cells differ from the main run (different cap/seed?)"
    S.d0, S.d1, S.tau = float(main["delta0"][0]), float(main["delta1"][0]), float(main["tau"][0])
    S.neighbours()
    C = Cal(S)
    rng = np.random.default_rng(seed + 99)
    fold = rng.integers(0, 2, S.npop)
    t0 = time.time()
    rho, rho_m, stats = [None, None], [None, None], []
    for f in (0, 1):
        r, rm, nu, de = C.train(np.flatnonzero(fold == f))
        rho[f], rho_m[f] = r, rm
        stats.append((nu, de))
    log(f"calibration tables trained on {int((fold == 0).sum())}/{int((fold == 1).sum())} cells ({time.time() - t0:.0f}s)")
    ns = len(S.sidx)
    npr = len(E.KS) * len(E.WS)
    nanf = lambda *sh: np.full(sh, np.nan)
    out = dict(ms_cal=nanf(ns), pres_cal=nanf(ns), cnt_cal=nanf(ns),
               ms_l1cal=nanf(ns), pres_l1cal=nanf(ns), cnt_l1cal=nanf(ns),
               ms_l2cal=nanf(ns, npr), pres_l2cal=nanf(ns, npr), cnt_l2cal=nanf(ns, npr),
               ms_mix=nanf(ns, len(LAMS), npr), pres_mix=nanf(ns, len(LAMS), npr), cnt_mix=nanf(ns, len(LAMS), npr),
               pop=S.sidx.astype(np.int64))
    lf = S.lf
    for ii, p in enumerate(S.sidx):
        fi, x, q1, N = C.cell_q1(p)
        s, e = S.pptr[p], S.pptr[p + 1]
        a = S.Afull[s:e]
        b = x - a
        NA, NB = a.sum(), N - a.sum()
        if NA <= 0 or NB <= 0:
            continue
        bm = b > 0
        Bv = b[bm]
        lfB = lf[Bv.astype(np.int64)]
        lfB_sum, lfNB = lfB.sum(), lf[int(NB)]
        fo = 1 - fold[p]
        qcB = C.predict(rho[fo], fi, a, q1, NA)[bm]
        out["ms_cal"][ii], out["pres_cal"][ii], out["cnt_cal"][ii] = E.score_all(qcB, Bv, NB, lfB, lfB_sum, lfNB)
        q1c = C.predict_marg(rho_m[fo], fi, q1, NA)
        out["ms_l1cal"][ii], out["pres_l1cal"][ii], out["cnt_l1cal"][ii] = E.score_all(q1c[bm], Bv, NB, lfB, lfB_sum, lfNB)
        pn = S.pn[:, S.pn_off[ii]:S.pn_off[ii + 1]].astype(np.float64)
        q2 = np.array([(1.0 - w) * q1c + w * pn[ki] for ki in range(len(E.KS)) for w in E.WS])[:, bm]
        out["ms_l2cal"][ii], out["pres_l2cal"][ii], out["cnt_l2cal"][ii] = E.score_all(q2, Bv, NB, lfB, lfB_sum, lfNB)
        for li, lam in enumerate(LAMS):
            qm = (1.0 - lam) * qcB[None, :] + lam * q2
            out["ms_mix"][ii, li], out["pres_mix"][ii, li], out["cnt_mix"][ii, li] = E.score_all(qm, Bv, NB, lfB, lfB_sum, lfNB)
        if ii % 1000 == 0:
            log(f"[cal] {ii}/{ns} ({time.time() - t0:.0f}s)")
    out["rho"] = np.stack(rho)
    out["rho_m"] = np.stack(rho_m)
    out["numer"] = np.stack([s_[0] for s_ in stats])
    out["denom"] = np.stack([s_[1] for s_ in stats])
    np.savez_compressed(os.path.join(E.RES, f"e1_cal_{sample}__{member}__{res}.npz"), **out)
    log(f"wrote e1_cal_{sample}__{member}__{res}.npz ({time.time() - t0:.0f}s)")


def report():
    import e1_report as R
    lines = ["# E1 extension: calibrated own-half predictor (L3cal), recalibrated baseline (L1cal) and mixtures\n",
             "Bits per held-out nonzero (b/nnzB); hyper-parameters cross-fitted over cells. Gains are vs the main run's L1 unless a "
             "column says vs L1cal. L1cal = a-free recalibration by expression level (static model repair); L2cal = neighbours on L1cal; "
             "L3cal = learned A->B map; mix = (1-lam) L3cal + lam L2cal (see tools/e1_cal.py).\n",
             "| file | cells | L1 | L1cal gain | L2 gain | L2cal gain | L3 (global alpha) gain | L3cal gain | mix gain | own-half = L3cal-L1cal | nbr = L2cal-L1cal | mix-L1cal | mix-L1cal bits/cell | mix-L1cal b/nnz_cell |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    paths = sorted(glob.glob(os.path.join(E.RES, "e1_cal_*__interval.npz"))) + sorted(glob.glob(os.path.join(E.RES, "e1_cal_*__gene.npz")))
    for p in paths:
        base = os.path.basename(p)[len("e1_cal_"):-4]
        sample, member, res = base.split("__")
        mp = os.path.join(E.RES, f"e1_percell_{sample}__{member}__{res}__cells.npz")
        fd = R.FileData(mp)
        c = E.load_percell(p)
        v = ~np.isnan(np.load(mp)["N"])
        sel = fd.select()
        z = fd.z
        nB, nC = z["nnzB"].sum(), z["nnz"].sum()
        L1 = sel["L1"]["ms"].sum()
        l1c = c["ms_l1cal"][v]
        l3c = c["ms_cal"][v]
        l2c = R.pick(c["ms_l2cal"][v], R.xfit(c["ms_l2cal"][v], fd.par))
        M = c["ms_mix"][v].reshape(fd.n, -1)
        cand = np.hstack([l3c[:, None], l2c[:, None], M])
        mix = R.pick(cand, R.xfit(cand, fd.par))
        g = lambda a: (L1 - a.sum()) / nB
        d = lambda a: (l1c.sum() - a.sum())
        lines.append(f"| {sample}/{member} {res} | {fd.n} | {L1 / nB:.3f} | {g(l1c):+.3f} | {g(sel['L2']['ms']):+.3f} | {g(l2c):+.3f} | "
                     f"{g(sel['L3seq']['ms']):+.3f} | {g(l3c):+.3f} | {g(mix):+.3f} | {d(l3c) / nB:+.3f} | {d(l2c) / nB:+.3f} | {d(mix) / nB:+.3f} | "
                     f"{d(mix) / fd.n:+.1f} | {d(mix) / nC:+.3f} |")
    lines.append("")
    for p in paths:
        base = os.path.basename(p)[len("e1_cal_"):-4]
        c = E.load_percell(p)
        rho = c["rho"].mean(0)
        rm = c["rho_m"].mean(0)
        lines.append(f"{base}: marginal rho_m by u=N_A q bin: " + ", ".join(f"{UEDGES[i]:.0e}:{rm[i]:.2f}" for i in range(0, NU, 2)))
        lines.append(f"{base}: rho(a=0) by u bin: " + ", ".join(f"{UEDGES[i]:.0e}:{rho[i, 0]:.2f}" for i in range(0, NU, 2)))
    with open(os.path.join(E.RES, "e1_cal_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("sample")
    r.add_argument("member")
    r.add_argument("--res", default="interval")
    r.add_argument("--cap", type=int, default=5000)
    r.add_argument("--seed", type=int, default=20261021)
    sp.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.sample, a.member, a.res, a.cap, a.seed, lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
    else:
        report()
