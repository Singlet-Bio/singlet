"""E2 model prototype: ideal code length (bits) of cells/empties under a block-adaptive rank-1 Poisson presence model.
   hazard h_fj = A_f * C_b(j) (natural units), p = 1-exp(-h).  A_f from running presence counts z_f, C_b from running bucket nnz.
   python -P tools/e2_proto.py sample member [options]"""
import sys, argparse, time
sys.path.insert(0, 'tools')
from e2_common import *

LN2 = np.log(2.0)

def bucket_of(N, bpo):
    """log-spaced depth bucket, bpo buckets per octave, N>=0 -> bucket index (N=0 -> 0)."""
    N = np.maximum(N, 1)
    return np.floor(np.log2(N) * bpo + 1e-9).astype(np.int64)

class Model:
    def __init__(self, m, nbk, bpo, a_pc, gamma_mode):
        self.m, self.nbk, self.bpo = m, nbk, bpo
        self.z = np.zeros(m, np.int64)         # presence counts per feature
        self.nh = np.zeros(nbk)                # columns per bucket
        self.T = np.zeros(nbk)                 # nnz per bucket
        self.a_pc = a_pc
        self.A = None; self.C = None
        self.gamma_mode = gamma_mode
        self.xg = np.linspace(-30, 4, 2400)    # grid over log A
        self.yg = np.linspace(-12, 12, 1200)   # grid over log C

    def centres(self):
        return 2.0 ** ((np.arange(self.nbk) + 0.5) / self.bpo)

    def fit(self, rounds=3):
        m = self.m
        ncol = self.nh.sum()
        if ncol == 0:
            # flat prior: hazard per feature 1/m_flat, scale = N
            self.A = np.full(m, 1.0 / 30000.0)
            self.C = self.centres().copy()
            return
        Nc = self.centres()
        have = self.nh > 0
        # initial C: proportional to N
        C = Nc.copy()
        A = self.A if self.A is not None else np.full(m, 1.0 / 30000.0)
        zsel = self.z
        for r in range(rounds):
            # A step: Phi(x) = sum_b nh_b (1 - exp(-e^x C_b)), monotone in x
            ex = np.exp(self.xg)
            Phi = (self.nh[None, :] * (-np.expm1(-ex[:, None] * C[None, :]))).sum(1)
            tgt = zsel + self.a_pc
            xf = np.interp(tgt, Phi, self.xg)
            A = np.exp(xf)
            # C step: Psi(y) = sum_f (1 - exp(-A_f e^y)); use histogram of log A
            hb = np.linspace(-30, 4, 700)
            hist, _ = np.histogram(np.log(A), bins=np.append(hb, 4.1))
            hc = np.exp(hb)
            ey = np.exp(self.yg)
            Psi = (hist[None, :] * (-np.expm1(-ey[:, None] * hc[None, :]))).sum(1)
            mean_nnz = np.where(have, self.T / np.maximum(self.nh, 1), 0.0)
            yb = np.interp(mean_nnz, Psi, self.yg)
            Cn = np.exp(yb)
            # smooth/extrapolate for buckets without enough data: weighted log-linear fit of log C vs log N
            w = np.minimum(self.nh, 50.0)
            if (w > 0).sum() >= 2:
                xs = np.log(Nc); ys = np.log(Cn)
                W = w
                sw = W.sum(); mx = (W * xs).sum() / sw; my = (W * ys).sum() / sw
                sxx = (W * (xs - mx) ** 2).sum(); sxy = (W * (xs - mx) * (ys - my)).sum()
                slope = sxy / sxx if sxx > 0 else 1.0
                slope = min(max(slope, 0.3), 1.5)
                fit = np.exp(my + slope * (xs - mx))
            else:
                fit = Nc * (Cn[have][0] / Nc[have][0])
            lam = self.nh / (self.nh + 20.0)    # trust in the bucket's own estimate
            C = np.exp(lam * np.log(Cn) + (1 - lam) * np.log(fit))
        self.A, self.C = A, C

def run(d, cols, sched, bpo, a_pc, counts=True):
    m = d['m']
    nbk = int(np.log2(max(d['N'].max(), 2)) * bpo) + 2
    mod = Model(m, nbk, bpo, a_pc, None)
    colid, idx, val = columns(d, cols)
    kk = d['k'][cols]; NN = d['N'][cols]
    ip = np.concatenate([[0], np.cumsum(kk)])
    bk = bucket_of(NN, bpo)
    tot_pres = 0.0; tot_cnt = 0.0
    nblocks = 0
    pos = 0
    n = len(cols)
    ctab = None
    # count context tables: ctx = quantised log2 hazard (half-octave) 0..47 ; symbols min(v-1,15)
    NCX, NS = 64, 17
    cnt_tab = np.zeros((NCX, NS))
    block_log = []
    t0 = time.time()
    while pos < n:
        sz = sched(pos)
        e = min(n, pos + sz)
        mod.fit()
        lo, hi = ip[pos], ip[e]
        ci = colid[lo:hi] - pos
        f = idx[lo:hi]
        v = val[lo:hi]
        bj = bk[pos:e]
        Cj = mod.C[bj]                       # per column
        h = mod.A[f] * Cj[ci]                # per nnz
        # present cost: -log2(1-exp(-h)); absent: total hazard minus present hazards
        pres = -np.log2(-np.expm1(-h))
        Stot = mod.A.sum()
        absent = (Cj * Stot).sum() / LN2 - (h.sum()) / LN2
        bp = pres.sum() + absent
        tot_pres += bp
        bc = 0.0
        if counts:
            cx = np.clip(np.floor(np.log2(h) * 2).astype(np.int64) + 40, 0, NCX - 1)
            s = np.minimum(v - 1, 16)
            tab = cnt_tab + 0.4 * np.array([1.0 / (i + 1) ** 2 for i in range(NS)])[None, :] + 0.02
            P = tab[cx, s] / tab[cx].sum(1)
            bc = -np.log2(P).sum() + np.sum(np.where(v > 16, 2 * np.log2(np.maximum(v - 15, 1)) + 1, 0))
            np.add.at(cnt_tab, (cx, s), 1.0)
        tot_cnt += bc
        block_log.append((pos, e - pos, hi - lo, bp / max(hi - lo, 1), bc / max(hi - lo, 1)))
        # update state
        np.add.at(mod.z, f, 1)
        np.add.at(mod.nh, bj, 1.0)
        np.add.at(mod.T, bj, np.diff(ip[pos:e + 1]).astype(float))
        pos = e
        nblocks += 1
    nnz = ip[-1]
    return tot_pres / nnz, tot_cnt / nnz, nblocks, nnz, block_log

def make_sched(B, n0, rho):
    def sched(pos):
        return int(max(n0, min(B, int(rho * pos) - pos if pos > 0 else n0)))
    return sched

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('sample'); ap.add_argument('member')
    ap.add_argument('--B', type=int, default=1024)
    ap.add_argument('--n0', type=int, default=64)
    ap.add_argument('--rho', type=float, default=2.0)
    ap.add_argument('--bpo', type=int, default=4)
    ap.add_argument('--a', type=float, default=0.3)
    ap.add_argument('--cls', default='cells')
    ap.add_argument('--order', default='natural')
    a = ap.parse_args()
    d = load(a.sample, a.member)
    sel = d['is_cell'] if a.cls == 'cells' else ~d['is_cell']
    cols = np.flatnonzero(sel)
    if a.order == 'random':
        rng = np.random.default_rng(1)
        cols = cols[rng.permutation(len(cols))]
    t = time.time()
    pres, cnt, nb, nnz, log = run(d, cols, make_sched(a.B, a.n0, a.rho), a.bpo, a.a)
    print(f"{a.sample} {a.member} {a.cls} order={a.order} B={a.B} n0={a.n0} rho={a.rho} bpo={a.bpo} a={a.a}: cols={len(cols)} nnz={nnz} blocks={nb}  presence {pres:.3f}  counts {cnt:.3f}  total {pres+cnt:.3f} bits/nnz   ({time.time()-t:.1f}s)")
    for r in log[:8]: print('   blk', r)
