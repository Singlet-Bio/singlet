#!/usr/bin/env python
"""E1: how much of a cell's UMIs are predictable from the rest of the cell? (split predictability ladder)

    python -P tools/e1_split.py run  <sample> <member> [--res interval|gene] [--cap 5000] [--subsets cells,empties]
    python -P tools/e1_split.py agg  [--res interval|gene]        # per-cell npz -> results/e1_split*.tsv, e1_hpi*.tsv

Protocol (see plans/E1_split_predictability.md).  For every called cell with N >= 50 UMIs (empties: N >= 20, control):
  * split its UMIs into halves A / B per unit of count (A_f ~ Binomial(x_f, 1/2), B = x - A; fixed seed);
  * score B, as a multiset of UMIs and as a sparse presence+count vector, under predictors q that never see B:
      L0  population per-feature frequency, leave-one-out (this cell's full counts removed)
      L1  L0 x depth bucket (16 log-spaced buckets of the cell's full N): bucket profile shrunk to L0 (strength tau)
      L2  (1-w) L1 + w * mean profile of the k nearest neighbours; neighbours are found with the query's A half only
          against the OTHER cells' FULL profiles (log1p, top 2000 features, L2-normalised, cosine)
      L3  Dirichlet posterior (alpha * prior + counts_A) / (alpha + N_A), prior in {L1, every L2}   (plug-in q)
      L3dm same, scored with the exact Dirichlet-multinomial predictive of the multiset B given A (extra)
      L4  (alpha * L1 + counts_A + counts_B) / (alpha + N): in-sample oracle (biased low, B is in the model)
      F1 / F1dm  full-cell cost under static L1 / under the Dirichlet-multinomial with L1 centre (extras: converts
                 the half-cell numbers to bits per nnz of the whole cell and gives the within-cell burstiness gain)
  * smoothing constants (delta, tau) are tuned on the A halves of 400 cells; alpha, k, w are swept and all reported.
Sparse-vector score = independent Poisson per feature with rate N_B q_f, split into presence bits (Bernoulli
1-exp(-lam), absent cost lam log2 e) and count bits (zero-truncated Poisson); identity: sparse = multiset +
[N_B log2 e - N_B log2 N_B + log2 N_B!]  (i.e. + the cost of coding N_B under Poisson(N_B)); checked at run time.

Corpus files are untrusted data: only parsed as numbers here (run with python -P).
"""
import os, sys, time, math, argparse, csv, glob
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
E1CACHE = os.path.join(RES, "e1_cache")
TDCACHE = os.path.join(RES, ".track_d_cache")

LN2 = math.log(2.0)
LOG2E = 1.0 / LN2

ALPHAS = np.array([3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0])
KS = [10, 30, 100, 300]
WS = [0.15, 0.3, 0.5, 0.7]
DELTA_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
TAU_GRID = [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0, 300000.0, 1000000.0, 10000000.0]
NB_BUCKETS = 16
NTOP = 2000
N_MIN_CELL = 50
N_MIN_EMPTY = 20
N_TUNE = 400
N_HPOST = 300

# ---------------------------------------------------------------------------------------------
# numerics

try:
    from scipy.special import gammaln as _scipy_gammaln
    import scipy.sparse as _sp
except Exception:  # scipy is optional: numpy fallbacks below
    _scipy_gammaln = None
    _sp = None


def gammaln_np(x):
    """ln Gamma(x) for x > 0, numpy only (recurrence to x >= 10 then Stirling series); ~1e-10 accurate."""
    x = np.asarray(x, dtype=np.float64)
    y = x.copy()
    acc = np.zeros_like(x)
    for _ in range(12):
        m = y < 10.0
        if not m.any():
            break
        acc[m] += np.log(y[m])
        y[m] += 1.0
    s = (y - 0.5) * np.log(y) - y + 0.9189385332046727 + 1.0 / (12 * y) - 1.0 / (360 * y ** 3) + 1.0 / (1260 * y ** 5)
    return s - acc


def gammaln(x):
    if _scipy_gammaln is not None:
        return _scipy_gammaln(x)
    return gammaln_np(x)


def log2fact_table(maxn):
    return np.concatenate(([0.0], np.cumsum(np.log2(np.arange(1, int(maxn) + 1, dtype=np.float64)))))


def chao_shen_bits(x):
    """Chao-Shen coverage-adjusted entropy (bits) of a count vector x (positive integers)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.sum()
    f1 = float((x == 1).sum())
    if f1 >= n:
        f1 = n - 1.0
    cov = 1.0 - f1 / n
    pa = cov * x / n
    denom = -np.expm1(n * np.log1p(-pa))
    return float(-(pa * np.log2(pa) / denom).sum())


def plugin_bits(x):
    x = np.asarray(x, dtype=np.float64)
    p = x / x.sum()
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------------------------------------
# loading

def _cell_flags(sample, colnames):
    calls = {}
    with open(os.path.join(ROOT, "corpus", sample, "cell_calls.tsv")) as f:
        rd = csv.DictReader(f, delimiter="\t")
        for row in rd:
            calls[row["barcode"]] = row["is_cell"] == "TRUE"
    return np.array([calls.get(b, calls.get(b.split("-")[0], False)) for b in colnames], dtype=bool)


def load_matrix(sample, member):
    """CSC (columns = barcodes). Uses Track D's read-only cache when present, else the compiled reader."""
    cp = os.path.join(TDCACHE, f"{sample}__{member}.npz")
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=False)
        return dict(m=int(z["m"]), n=int(z["n"]), indptr=z["indptr"].astype(np.int64),
                    indices=z["indices"], data=z["data"], is_cell=z["is_cell"])
    sys.path.insert(0, os.path.join(ROOT, "..", "..", "python"))
    from singlet._pz import read_1pz
    r = read_1pz(os.path.join(ROOT, "corpus", sample, f"{member}_counts.1pz"))
    return dict(m=int(r["m"]), n=int(r["n"]), indptr=np.asarray(r["indptr"], dtype=np.int64),
                indices=np.asarray(r["indices"]), data=np.asarray(r["data"]),
                is_cell=_cell_flags(sample, list(r["colnames"])))


def gene_map(sample, member):
    """row -> gene index (rows are ENSG..._SYMBOL_chr:a-b or ..._intron_...; gene = text before the first '_')."""
    os.makedirs(E1CACHE, exist_ok=True)
    p = os.path.join(E1CACHE, f"{sample}__{member}__genemap.npy")
    if os.path.exists(p):
        return np.load(p)
    sys.path.insert(0, os.path.join(ROOT, "..", "..", "python"))
    from singlet._pz import read_1pz
    r = read_1pz(os.path.join(ROOT, "corpus", sample, f"{member}_counts.1pz"))
    genes = np.array([s.split("_", 1)[0] for s in r["rownames"]])
    _, inv = np.unique(genes, return_inverse=True)
    inv = inv.astype(np.int32)
    np.save(p, inv)
    return inv


def to_resolution(d, gmap):
    """Return (indptr, indices, data, M) at interval resolution (gmap None) or gene resolution."""
    ip, ind, val = d["indptr"], d["indices"], d["data"]
    if gmap is None:
        return ip, ind.astype(np.int64), val.astype(np.int64), d["m"]
    n = len(ip) - 1
    G = int(gmap.max()) + 1
    col = np.repeat(np.arange(n, dtype=np.int64), np.diff(ip))
    key = col * G + gmap[ind].astype(np.int64)
    order = np.argsort(key, kind="stable")
    key = key[order]
    v = val[order].astype(np.int64)
    new = np.ones(len(key), dtype=bool)
    new[1:] = key[1:] != key[:-1]
    st = np.flatnonzero(new)
    v2 = np.add.reduceat(v, st)
    k2 = key[st]
    col2 = k2 // G
    g2 = k2 % G
    ip2 = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(col2, minlength=n), out=ip2[1:])
    return ip2, g2, v2, G


def gather_cols(ptr, cols):
    """positions of all entries of the given columns, concatenated, plus the new column pointer."""
    starts = ptr[cols]
    lens = ptr[cols + 1] - starts
    newp = np.zeros(len(cols) + 1, dtype=np.int64)
    np.cumsum(lens, out=newp[1:])
    tot = int(newp[-1])
    gp = np.arange(tot, dtype=np.int64) + np.repeat(starts - newp[:-1], lens)
    return gp, newp


# ---------------------------------------------------------------------------------------------
# scoring primitives (all inputs are restricted to the B support)

def score_all(qB, Bv, NB, lfB, lfB_sum, lfNB):
    """qB [..., nbs] = predicted probabilities at the support of B; returns (ms, pres, cnt), each [...].
    ms   : multiset bits of B under multinomial(N_B, q):  -sum B log2 q - log2 N_B! + sum log2 B!
    pres : sparse-vector presence bits, Bernoulli(1-exp(-lam)) per feature, lam = N_B q (absent mass lam log2 e)
    cnt  : sparse-vector count bits, zero-truncated Poisson(lam) given presence"""
    lq = np.log2(qB)
    ms = -(lq * Bv).sum(-1) - lfNB + lfB_sum
    lam = NB * qB
    em = -np.expm1(-lam)
    lem = np.log2(em)
    pres = -lem.sum(-1) + (NB - lam.sum(-1)) * LOG2E
    cnt = (-(Bv * (lq + math.log2(NB))) + lam * LOG2E + lfB + lem).sum(-1)
    return ms, pres, cnt


# ---------------------------------------------------------------------------------------------
# per-subset engine

def sample_rngs(seed):
    return np.random.default_rng(seed), np.random.default_rng(seed + 1000003)


class Subset:
    """Population (a set of columns), its profile tables and the scored sample."""

    def __init__(self, name, ptr, ind, val, M, cols, nmin, cap, seed, log, use_nbr, afrac=0.5):
        self.name = name
        self.log = log
        self.use_nbr = use_nbr
        t0 = time.time()
        gp, pptr = gather_cols(ptr, cols)
        raw = ind[gp]
        pval = val[gp].astype(np.float64)
        npop = len(cols)
        cellid = np.repeat(np.arange(npop, dtype=np.int32), np.diff(pptr))
        present = np.bincount(raw, minlength=M) > 0
        uniq = np.flatnonzero(present)
        F = len(uniq)
        fmap = np.full(M, -1, dtype=np.int32)
        fmap[uniq] = np.arange(F, dtype=np.int32)
        pidx = fmap[raw]
        P = np.bincount(pidx, weights=pval, minlength=F)
        Npop = np.bincount(cellid, weights=pval, minlength=npop)
        nnzpop = np.diff(pptr)
        edges = np.logspace(math.log10(nmin), math.log10(Npop.max() + 1.0), NB_BUCKETS + 1)
        bucket = np.clip(np.searchsorted(edges, Npop, side="right") - 1, 0, NB_BUCKETS - 1).astype(np.int32)
        Pb = np.bincount(bucket[cellid].astype(np.int64) * F + pidx, weights=pval,
                         minlength=NB_BUCKETS * F).reshape(NB_BUCKETS, F)
        self.__dict__.update(dict(pptr=pptr, pidx=pidx, pval=pval, cellid=cellid, npop=npop, F=F, P=P, T=float(P.sum()),
                                  Npop=Npop, nnzpop=nnzpop, edges=edges, bucket=bucket, Pb=Pb, Tb=Pb.sum(1),
                                  cols=cols))
        rs, rb = sample_rngs(seed)
        order = rs.permutation(npop)
        self.sidx = order[:min(cap, npop)]
        self.Afull = rb.binomial(pval.astype(np.int64), afrac).astype(np.float64)
        self.lf = log2fact_table(max(Npop.max(), 2) + 2)
        log(f"[{name}] pop={npop} F={F} nnz={len(pidx)} T={self.T:.0f} scored={len(self.sidx)} "
            f"N: median={np.median(Npop):.0f} max={Npop.max():.0f} ({time.time() - t0:.1f}s)")

    # --- smoothing constants tuned on A halves
    def tune(self):
        P, T, F, Pb, Tb = self.P, self.T, self.F, self.Pb, self.Tb
        tune_cells = self.sidx[:N_TUNE]
        d0 = np.array(DELTA_GRID)
        c0 = np.zeros(len(d0))
        tauG = np.array(TAU_GRID)
        c1 = np.zeros((len(d0), len(tauG)))
        nA = 0.0
        for p in tune_cells:
            s, e = self.pptr[p], self.pptr[p + 1]
            fi = self.pidx[s:e]
            x = self.pval[s:e]
            a = self.Afull[s:e]
            N = self.Npop[p]
            am = a > 0
            aB = a[am]
            nA += aB.sum()
            Pf = P[fi][am]
            xa = x[am]
            q0 = (Pf[None, :] - xa[None, :] + d0[:, None]) / (T - N + F * d0)[:, None]
            c0 -= (aB * np.log2(q0)).sum(1)
            bk = self.bucket[p]
            pbf = Pb[bk][fi][am]
            q1 = (pbf[None, None, :] - xa[None, None, :] + tauG[None, :, None] * q0[:, None, :]) / \
                 (Tb[bk] - N + tauG)[None, :, None]
            c1 -= (aB * np.log2(q1)).sum(-1)
        i0 = int(np.argmin(c0))
        j1 = np.unravel_index(int(np.argmin(c1)), c1.shape)
        self.d0 = float(d0[i0])
        self.d1 = float(d0[j1[0]])
        self.tau = float(tauG[j1[1]])
        self.log(f"[{self.name}] tuned on A halves of {len(tune_cells)} cells: delta0={self.d0} (L0 {c0[i0] / nA:.4f} b/UMI) "
                 f"delta1={self.d1} tau={self.tau} (L1 {c1[j1] / nA:.4f} b/UMI)")

    # --- neighbour search (query = A half of scored cell, pool = other cells' full counts)
    def neighbours(self):
        P = self.P
        top = np.argsort(-P, kind="stable")[:NTOP]
        topidx = np.full(self.F, -1, dtype=np.int32)
        topidx[top] = np.arange(len(top), dtype=np.int32)
        nt = len(top)
        R = np.zeros((self.npop, nt), dtype=np.float32)
        ti = topidx[self.pidx]
        sel = ti >= 0
        R[self.cellid[sel], ti[sel]] = np.log1p(self.pval[sel]).astype(np.float32)
        nr = np.sqrt((R * R).sum(1))
        nz = nr > 0
        R[nz] /= nr[nz, None]
        ns = len(self.sidx)
        K = min(max(KS), self.npop - 1)
        nbr = np.zeros((ns, K), dtype=np.int32)
        valid = np.zeros(ns, dtype=bool)
        t0 = time.time()
        BATCH = 256
        for b0 in range(0, ns, BATCH):
            ids = self.sidx[b0:b0 + BATCH]
            Q = np.zeros((len(ids), nt), dtype=np.float32)
            for i, p in enumerate(ids):
                s, e = self.pptr[p], self.pptr[p + 1]
                t = topidx[self.pidx[s:e]]
                a = self.Afull[s:e]
                m = (t >= 0) & (a > 0)
                Q[i, t[m]] = np.log1p(a[m]).astype(np.float32)
            qn = np.sqrt((Q * Q).sum(1))
            qz = qn > 0
            Q[qz] /= qn[qz, None]
            S = Q @ R.T
            S[np.arange(len(ids)), ids] = -np.inf  # never the query cell itself
            part = np.argpartition(S, -K, axis=1)[:, -K:]
            ps = np.take_along_axis(S, part, 1)
            o = np.argsort(-ps, axis=1, kind="stable")
            nbr[b0:b0 + len(ids)] = np.take_along_axis(part, o, 1)
            valid[b0:b0 + len(ids)] = qz
            if (b0 // BATCH) % 4 == 0:
                self.log(f"[{self.name}] neighbours {b0 + len(ids)}/{ns} ({time.time() - t0:.0f}s)")
        del R
        self.nbr, self.nbr_valid, self.K = nbr, valid, K
        if _sp is not None:
            self.precompute_pn()
        self.log(f"[{self.name}] neighbour search done ({time.time() - t0:.0f}s); {int((~valid).sum())} queries with empty A-vector")

    def precompute_pn(self):
        """scipy path: mean neighbour profile (k in KS) at each scored cell's own features, via sparse matmul."""
        t0 = time.time()
        Pn = _sp.csr_matrix(((self.pval / self.Npop[self.cellid]).astype(np.float32), self.pidx, self.pptr),
                            shape=(self.npop, self.F))
        ns = len(self.sidx)
        lens = self.nnzpop[self.sidx]
        off = np.zeros(ns + 1, dtype=np.int64)
        np.cumsum(lens, out=off[1:])
        pn = np.zeros((len(KS), int(off[-1])), dtype=np.float32)
        buf = np.zeros(self.F, dtype=np.float32)
        BATCH = 256
        for b0 in range(0, ns, BATCH):
            b1 = min(ns, b0 + BATCH)
            nb = b1 - b0
            rows, cols, dat = [], [], []
            for ki, k in enumerate(KS):
                kk = min(k, self.K)
                rows.append(np.repeat(np.arange(nb, dtype=np.int64) + ki * nb, kk))
                cols.append(self.nbr[b0:b1, :kk].ravel())
                dat.append(np.full(nb * kk, 1.0 / kk, dtype=np.float32))
            W = _sp.csr_matrix((np.concatenate(dat), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(len(KS) * nb, self.npop))
            Rm = (W @ Pn).tocsr()
            for i in range(nb):
                p = self.sidx[b0 + i]
                fi = self.pidx[self.pptr[p]:self.pptr[p + 1]]
                for ki in range(len(KS)):
                    r = ki * nb + i
                    s_, e_ = Rm.indptr[r], Rm.indptr[r + 1]
                    idx = Rm.indices[s_:e_]
                    buf[idx] = Rm.data[s_:e_]
                    pn[ki, off[b0 + i]:off[b0 + i + 1]] = buf[fi]
                    buf[idx] = 0.0
        self.pn, self.pn_off = pn, off
        self.log(f"[{self.name}] neighbour mean profiles ({time.time() - t0:.0f}s)")

    def nbr_profiles(self, ii, fi, pos):
        """mean profile of the k nearest neighbours at the cell's features, k in KS -> [len(KS), n_c]"""
        if getattr(self, "pn", None) is not None:
            return self.pn[:, self.pn_off[ii]:self.pn_off[ii + 1]].astype(np.float64)
        J = self.nbr[ii]
        starts = self.pptr[J]
        lens = self.pptr[J + 1] - starts
        first = np.cumsum(lens) - lens
        tot = int(lens.sum())
        gp = np.arange(tot, dtype=np.int64) + np.repeat(starts - first, lens)
        pp = pos[self.pidx[gp]]
        w = self.pval[gp] / np.repeat(self.Npop[J], lens)
        r = np.repeat(np.arange(len(J), dtype=np.int32), lens)
        sel = pp >= 0
        pp, w, r = pp[sel], w[sel], r[sel]
        out = np.zeros((len(KS), len(fi)))
        for ki, k in enumerate(KS):
            kk = min(k, len(J))
            m = r < kk
            out[ki] = np.bincount(pp[m], weights=w[m], minlength=len(fi)) / kk
        return out

    # --- main scoring loop
    def score(self):
        ns = len(self.sidx)
        npri = 1 + len(KS) * len(WS) if self.use_nbr else 1
        na = len(ALPHAS)
        z = lambda *shape: np.full(shape, np.nan, dtype=np.float64)
        o = dict(
            N=z(ns), NA=z(ns), NB=z(ns), nnz=z(ns), nnzB=z(ns), bucket=z(ns), pop=self.sidx.astype(np.int64),
            ms_L0=z(ns), pres_L0=z(ns), cnt_L0=z(ns), ms_L1=z(ns), pres_L1=z(ns), cnt_L1=z(ns),
            ms_L3=z(ns, na, npri), pres_L3=z(ns, na, npri), cnt_L3=z(ns, na, npri), ms_L3dm=z(ns, na, npri),
            F1=z(ns), F1dm=z(ns, na),
            Hplug=z(ns), Hcs=z(ns), lfN=z(ns), sumlfx=z(ns), lfNB=z(ns), sumlfB=z(ns),
        )
        if self.use_nbr:
            o.update(ms_L2=z(ns, npri - 1), pres_L2=z(ns, npri - 1), cnt_L2=z(ns, npri - 1),
                     ms_L4=z(ns, na), pres_L4=z(ns, na), cnt_L4=z(ns, na))
        pos = np.full(self.F, -1, dtype=np.int32)
        lf = self.lf
        P, T, F, Pb, Tb = self.P, self.T, self.F, self.Pb, self.Tb
        d0, d1, tau = self.d0, self.d1, self.tau
        t0 = time.time()
        checked = 0
        for ii, p in enumerate(self.sidx):
            s, e = self.pptr[p], self.pptr[p + 1]
            fi = self.pidx[s:e]
            x = self.pval[s:e]
            a = self.Afull[s:e]
            b = x - a
            N = self.Npop[p]
            NA = a.sum()
            NB = N - NA
            if NA <= 0 or NB <= 0:
                continue
            bm = b > 0
            Bv = b[bm]
            lfB = lf[Bv.astype(np.int64)]
            lfB_sum = lfB.sum()
            lfNB = lf[int(NB)]
            xB, aB = x[bm], a[bm]
            o["N"][ii], o["NA"][ii], o["NB"][ii] = N, NA, NB
            o["nnz"][ii], o["nnzB"][ii], o["bucket"][ii] = len(x), bm.sum(), self.bucket[p]
            # H(pi) estimators on the full cell
            xi = x.astype(np.int64)
            o["Hplug"][ii] = plugin_bits(x)
            o["Hcs"][ii] = chao_shen_bits(x)
            o["lfN"][ii] = lf[int(N)]
            o["sumlfx"][ii] = lf[xi].sum()
            o["lfNB"][ii], o["sumlfB"][ii] = lfNB, lfB_sum
            # L0 / L1
            Pf = P[fi]
            q0 = (Pf - x + d0) / (T - N + F * d0)
            q0b = (Pf - x + d1) / (T - N + F * d1)
            bk = self.bucket[p]
            q1 = (Pb[bk][fi] - x + tau * q0b) / (Tb[bk] - N + tau)
            q0B = q0[bm]
            o["ms_L0"][ii], o["pres_L0"][ii], o["cnt_L0"][ii] = score_all(q0B, Bv, NB, lfB, lfB_sum, lfNB)
            # priors
            pri = [q1]
            if self.use_nbr:
                if self.nbr_valid[ii]:
                    pos[fi] = np.arange(len(fi), dtype=np.int32)
                    pn = self.nbr_profiles(ii, fi, pos)
                    pos[fi] = -1
                    for ki in range(len(KS)):
                        for w in WS:
                            pri.append((1.0 - w) * q1 + w * pn[ki])
                else:
                    for _ in range(len(KS) * len(WS)):
                        pri.append(q1)
            Q = np.array(pri)                    # [npri, n_c]
            QB = Q[:, bm]                        # [npri, nbs]
            o["ms_L1"][ii], o["pres_L1"][ii], o["cnt_L1"][ii] = score_all(QB[0], Bv, NB, lfB, lfB_sum, lfNB)
            if self.use_nbr:
                o["ms_L2"][ii], o["pres_L2"][ii], o["cnt_L2"][ii] = score_all(QB[1:], Bv, NB, lfB, lfB_sum, lfNB)
            # L3: Dirichlet posterior mean, multinomial scoring
            al = ALPHAS[:, None, None]
            q3 = (al * QB[None] + aB) / (ALPHAS + NA)[:, None, None]
            o["ms_L3"][ii], o["pres_L3"][ii], o["cnt_L3"][ii] = score_all(q3, Bv, NB, lfB, lfB_sum, lfNB)
            # L3dm: exact Dirichlet-multinomial predictive of the multiset B given A
            aq = al * QB[None]
            t = gammaln(aq + xB) - gammaln(aq + aB)
            l2s = ((gammaln(ALPHAS + NA) - gammaln(ALPHAS + N))[:, None] + t.sum(-1)) * LOG2E
            o["ms_L3dm"][ii] = -l2s - lfNB + lfB_sum
            # L4: in-sample oracle (alpha * L1 + counts_A + counts_B)
            if self.use_nbr:
                q4 = (ALPHAS[:, None] * QB[0][None] + xB) / (ALPHAS + N)[:, None]
                o["ms_L4"][ii], o["pres_L4"][ii], o["cnt_L4"][ii] = score_all(q4, Bv, NB, lfB, lfB_sum, lfNB)
            # full-cell reference costs under L1 (static) and the Dirichlet-multinomial centred on L1
            lfx = o["sumlfx"][ii]
            o["F1"][ii] = -(np.log2(q1) * x).sum() - o["lfN"][ii] + lfx
            aq1 = ALPHAS[:, None] * q1[None]
            l2s1 = ((gammaln(ALPHAS) - gammaln(ALPHAS + N)) + (gammaln(aq1 + x) - gammaln(aq1)).sum(-1)) * LOG2E
            o["F1dm"][ii] = -l2s1 - o["lfN"][ii] + lfx
            # run-time identity check: sparse = multiset + N_B log2 e - N_B log2 N_B + log2 N_B!
            if checked < 20:
                lhs = o["pres_L1"][ii] + o["cnt_L1"][ii] - o["ms_L1"][ii]
                rhs = NB * LOG2E - NB * math.log2(NB) + lfNB
                assert abs(lhs - rhs) < 1e-6 * max(1.0, abs(rhs)) + 1e-6, (lhs, rhs)
                checked += 1
            if ii % 500 == 0:
                self.log(f"[{self.name}] scored {ii}/{ns} ({time.time() - t0:.0f}s)")
        self.out = o
        self.log(f"[{self.name}] scoring done ({time.time() - t0:.0f}s)")

    def hpost(self, alpha_star):
        """entropy of the L3 posterior mean (alpha* L1 + x)/(alpha* + N) of the FULL cell over the whole universe,
        and of the L1 profile itself, for the first N_HPOST scored cells."""
        P, T, F, Pb, Tb = self.P, self.T, self.F, self.Pb, self.Tb
        ns = min(N_HPOST, len(self.sidx))
        hp = np.full(ns, np.nan)
        h1 = np.full(ns, np.nan)
        for ii in range(ns):
            p = self.sidx[ii]
            s, e = self.pptr[p], self.pptr[p + 1]
            fi = self.pidx[s:e]
            xd = np.zeros(F)
            xd[fi] = self.pval[s:e]
            N = self.Npop[p]
            bk = self.bucket[p]
            q0d = (P + self.d1 - xd) / (T - N + F * self.d1)
            q1d = (Pb[bk] - xd + self.tau * q0d) / (Tb[bk] - N + self.tau)
            post = (alpha_star * q1d + xd) / (alpha_star + N)
            hp[ii] = -(post * np.log2(post)).sum()
            h1[ii] = -(q1d * np.log2(q1d)).sum()
        self.out["Hpost"] = hp
        self.out["H1"] = h1


def prior_labels(use_nbr):
    lab = ["L1"]
    if use_nbr:
        lab += [f"k={k},w={w}" for k in KS for w in WS]
    return lab


# ---------------------------------------------------------------------------------------------
# run one file

def run_file(sample, member, res, cap, seed, subsets, log, afrac=0.5):
    t0 = time.time()
    d = load_matrix(sample, member)
    gmap = gene_map(sample, member) if res == "gene" else None
    ptr, ind, val, M = to_resolution(d, gmap)
    is_cell = d["is_cell"]
    n = len(ptr) - 1
    col = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
    Nall = np.bincount(col, weights=val.astype(np.float64), minlength=n)
    del col
    log(f"{sample}/{member} res={res}: n={n} nnz={len(ind)} M={M} called={int(is_cell.sum())} "
        f"called N>=50: {int((is_cell & (Nall >= N_MIN_CELL)).sum())} empties N>=20: {int((~is_cell & (Nall >= N_MIN_EMPTY)).sum())} "
        f"(load {time.time() - t0:.0f}s)")
    results = {}
    for sub in subsets:
        if sub == "cells":
            cols = np.flatnonzero(is_cell & (Nall >= N_MIN_CELL))
            S = Subset("cells", ptr, ind, val, M, cols, N_MIN_CELL, cap, seed, log, use_nbr=True, afrac=afrac)
        elif sub == "empties":
            cols = np.flatnonzero(~is_cell & (Nall >= N_MIN_EMPTY))
            if len(cols) < 200:
                log(f"empties: only {len(cols)} droplets with N>=20, skipped")
                continue
            S = Subset("empties", ptr, ind, val, M, cols, N_MIN_EMPTY, cap, seed, log, use_nbr=False, afrac=afrac)
        else:
            raise SystemExit("unknown subset " + sub)
        S.tune()
        if S.use_nbr:
            S.neighbours()
        S.score()
        # alpha* = best plug-in L3 alpha with the L1 prior (pooled B multiset bits), for the posterior-entropy estimate
        o = S.out
        v = ~np.isnan(o["N"])
        if sub == "cells":
            ai = int(np.argmin(np.nansum(o["ms_L3"][v][:, :, 0], 0)))
            S.hpost(float(ALPHAS[ai]))
            o["Hpost_alpha"] = np.array([ALPHAS[ai]])
        o["edges"] = S.edges
        o["delta0"], o["delta1"], o["tau"] = np.array([S.d0]), np.array([S.d1]), np.array([S.tau])
        o["F"], o["npop"] = np.array([S.F]), np.array([S.npop])
        results[sub] = o
        outp = (os.path.join(RES, f"e1_percell_{sample}__{member}__{res}__{sub}.npz") if afrac == 0.5
                else os.path.join(RES, f"e1_afrac_{sample}__{member}__{res}__f{int(round(afrac * 100))}.npz"))
        np.savez_compressed(outp, **{k: v for k, v in o.items()})
        log(f"wrote {outp}")
        del S
    log(f"{sample}/{member} res={res} total {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------------------------
# series enumeration + aggregation (shared with e1_report.py)

def load_percell(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def series(z, use_nbr=None):
    """Yield (level, params, basis, ms[ncell], pres[ncell]|None, cnt[ncell]|None). basis 'B' = held-out half, 'full' = whole cell."""
    if use_nbr is None:
        use_nbr = "ms_L2" in z
    labs = prior_labels(use_nbr)
    yield "L0", "", "B", z["ms_L0"], z["pres_L0"], z["cnt_L0"]
    yield "L1", "", "B", z["ms_L1"], z["pres_L1"], z["cnt_L1"]
    if use_nbr:
        for j, lab in enumerate(labs[1:]):
            yield "L2", lab, "B", z["ms_L2"][:, j], z["pres_L2"][:, j], z["cnt_L2"][:, j]
    for ai, al in enumerate(ALPHAS):
        for j, lab in enumerate(labs):
            yield "L3", f"prior={lab},alpha={al:g}", "B", z["ms_L3"][:, ai, j], z["pres_L3"][:, ai, j], z["cnt_L3"][:, ai, j]
    for ai, al in enumerate(ALPHAS):
        for j, lab in enumerate(labs):
            yield "L3dm", f"prior={lab},alpha={al:g}", "B", z["ms_L3dm"][:, ai, j], None, None
    if use_nbr:
        for ai, al in enumerate(ALPHAS):
            yield "L4", f"alpha={al:g}", "B", z["ms_L4"][:, ai], z["pres_L4"][:, ai], z["cnt_L4"][:, ai]
    yield "F1", "", "full", z["F1"], None, None
    for ai, al in enumerate(ALPHAS):
        yield "F1dm", f"alpha={al:g}", "full", z["F1dm"][:, ai], None, None


TSV_COLS = ["file", "subset", "res", "bucket", "N_lo", "N_hi", "level", "params", "basis", "n_cells", "umis", "nnz",
            "ms_bits_per_umi", "ms_bits_per_nnz", "sp_bits_per_umi", "sp_bits_per_nnz", "sp_pres_bits_per_nnz",
            "sp_cnt_bits_per_nnz"]


def agg_rows(fname, sub, res, z):
    v = ~np.isnan(z["N"])
    bucket = z["bucket"]
    edges = z["edges"]
    rows = []
    groups = [("all", v, edges[0], edges[-1])]
    for g in range(NB_BUCKETS):
        m = v & (bucket == g)
        if m.sum() >= 5:
            groups.append((str(g), m, edges[g], edges[g + 1]))
    for level, params, basis, ms, pres, cnt in series(z):
        for gname, m, lo, hi in groups:
            if basis == "B":
                um, nz = z["NB"][m].sum(), z["nnzB"][m].sum()
            else:
                um, nz = z["N"][m].sum(), z["nnz"][m].sum()
            msb = np.nansum(ms[m])
            row = [fname, sub, res, gname, f"{lo:.1f}", f"{hi:.1f}", level, params, basis, int(m.sum()), int(um), int(nz),
                   f"{msb / um:.5f}", f"{msb / nz:.5f}"]
            if pres is not None:
                sp = np.nansum(pres[m]) + np.nansum(cnt[m])
                row += [f"{sp / um:.5f}", f"{sp / nz:.5f}", f"{np.nansum(pres[m]) / nz:.5f}", f"{np.nansum(cnt[m]) / nz:.5f}"]
            else:
                row += ["", "", "", ""]
            rows.append(row)
    return rows


def hpi_row(fname, res, z):
    v = ~np.isnan(z["N"])
    N, nnz = z["N"][v], z["nnz"][v]
    qs = lambda a: [f"{np.nanpercentile(a, q):.3f}" for q in (25, 50, 75)]
    hplug, hcs = z["Hplug"][v], z["Hcs"][v]
    hm_cs = N * hcs - z["lfN"][v] + z["sumlfx"][v]            # implied H(Multinomial(N, pi)), Chao-Shen H
    hm_pl = N * hplug - z["lfN"][v] + z["sumlfx"][v]
    row = [fname, res, int(v.sum()), f"{np.median(N):.0f}", f"{np.median(nnz):.0f}"]
    row += qs(hplug) + qs(hcs) + qs(hm_cs / nnz) + qs(hm_pl / nnz)
    if "Hpost" in z:
        hp = z["Hpost"]
        k = len(hp)
        hm_po = N[:k] * hp - z["lfN"][v][:k] + z["sumlfx"][v][:k]
        row += qs(hp) + qs(z["H1"]) + qs(hm_po / nnz[:k])
        row += [f"{float(z['Hpost_alpha'][0]):g}"]
    else:
        row += [""] * 10
    return row


HPI_COLS = ["file", "res", "n_cells", "median_N", "median_nnz",
            "Hplug_q25", "Hplug_q50", "Hplug_q75", "Hcs_q25", "Hcs_q50", "Hcs_q75",
            "HMult_cs_per_nnz_q25", "HMult_cs_per_nnz_q50", "HMult_cs_per_nnz_q75",
            "HMult_plug_per_nnz_q25", "HMult_plug_per_nnz_q50", "HMult_plug_per_nnz_q75",
            "Hpost_q25", "Hpost_q50", "Hpost_q75", "H1prior_q25", "H1prior_q50", "H1prior_q75",
            "HMult_post_per_nnz_q25", "HMult_post_per_nnz_q50", "HMult_post_per_nnz_q75", "Hpost_alpha"]


def do_agg(res):
    tsv = os.path.join(RES, "e1_split.tsv" if res == "interval" else f"e1_split_{res}.tsv")
    hpi = os.path.join(RES, "e1_hpi.tsv" if res == "interval" else f"e1_hpi_{res}.tsv")
    fl = sorted(glob.glob(os.path.join(RES, f"e1_percell_*__{res}__*.npz")))
    with open(tsv, "w", newline="") as f, open(hpi, "w", newline="") as g:
        w = csv.writer(f, delimiter="\t")
        wh = csv.writer(g, delimiter="\t")
        w.writerow(TSV_COLS)
        wh.writerow(HPI_COLS)
        for p in fl:
            base = os.path.basename(p)[len("e1_percell_"):-4]
            sample, member, r, sub = base.split("__")
            fname = f"{sample}/{member}"
            z = load_percell(p)
            for row in agg_rows(fname, sub, res, z):
                w.writerow(row)
            if sub == "cells":
                wh.writerow(hpi_row(fname, res, z))
    print("wrote", tsv, hpi, len(fl), "files")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("sample")
    r.add_argument("member", choices=["exon", "intron"])
    r.add_argument("--res", default="interval", choices=["interval", "gene"])
    r.add_argument("--cap", type=int, default=5000)
    r.add_argument("--seed", type=int, default=20261021)
    r.add_argument("--subsets", default="cells,empties")
    r.add_argument("--afrac", type=float, default=0.5, help="probability that a UMI goes to the context half A (default 0.5)")
    a = sp.add_parser("agg")
    a.add_argument("--res", default="interval", choices=["interval", "gene"])
    args = ap.parse_args()
    if args.cmd == "run":
        def log(msg):
            print(time.strftime("%H:%M:%S"), msg, flush=True)
        run_file(args.sample, args.member, args.res, args.cap, args.seed, args.subsets.split(","), log, afrac=args.afrac)
    else:
        do_agg(args.res)


if __name__ == "__main__":
    main()
