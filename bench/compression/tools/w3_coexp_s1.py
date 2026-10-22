"""w3_coexp stage 1: how much does within-cell co-expression add on top of a per-gene x depth model?

Gene level, called cells, presence only.  Run with `python -P` (corpus data are parsed, never executed).

Model (all on the top-G "target" genes; hubs and the tail keep the baseline, so the gain is exactly the saving on targets):
  baseline : P[b, g] = per-gene x depth-bucket Bernoulli (16 log buckets of interval nnz), class-pooled (c_r2-like, class =
             quantised file presence count) shrunk toward the gene's own per-bucket rate with weight kappa (best kappa on the
             evaluated half is used, i.e. the baseline is slightly favoured).
  cond.    : logit p[c, g] = logit P[b_c, g] + delta_g + s_g * sum_k q_gk * f_ck ,  f_ck = h_ck - Pk[b_c]   (hub presence,
             centred by the hub's own baseline rate in the bucket) ; q int8, s_g per-target scale ; delta_g = in-file calibration
             (moment matching on the fitting half; a decoder can learn it online).
Regimes:
  i  : weights trained on OTHER studies (leave-one-study-out pool), baseline + delta from the opposite half of the same file.
  ii : weights (and hubs) trained on one half of this file, evaluated on the other half; cross-fitted (A->B, B->A) so every
       called cell is scored out of sample; transmitted weights counted as side info.
  in : in-sample reference (everything fitted and scored on the same cells) -- an upper bound, labelled.
"""
import sys, os, time, json, csv, argparse
import numpy as np
import scipy.sparse as sp
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
torch.set_num_threads(int(os.environ.get('W3_THREADS', '3')))

NB = 16
EDGES = np.exp(np.linspace(np.log(30.0), np.log(8000.0), NB - 1))   # 15 edges -> 16 log buckets of interval nnz
BUCKET_LO = np.concatenate([[0], EDGES])


def bkt(d):
    return np.searchsorted(EDGES, d, side='right').astype(np.int64)


def read_tsv(p):
    with open(p, newline='') as f:
        r = csv.reader(f, delimiter='\t')
        hdr = next(r)
        return hdr, [row for row in r]


class Study:
    def __init__(self, cid, base):
        z = np.load(f'results/w3_coexp_cache/{cid}.npz')
        self.cid = cid; self.base = base
        self.study = str(z['study'])
        self.gids = z['gids']; self.gsym = z['gsym']
        ptr = z['c_ptr']; idx = z['c_idx']; nc = len(ptr) - 1
        self.n = nc; ng = len(self.gids)
        self.X = sp.csr_matrix((np.ones(len(idx), np.uint8), idx, ptr), shape=(nc, ng))
        self.C = sp.csr_matrix((z['c_cnt'].astype(np.int32), idx, ptr), shape=(nc, ng))
        self.d = z['c_d'].astype(np.int64); self.b = bkt(self.d)
        self.N = z['c_N']
        self.nnz_gene = np.diff(ptr)

    def rate(self):
        return np.asarray(self.X.sum(0)).ravel() / max(self.n, 1)


def load_all():
    S = {}
    for tsv, base in (('corpus.tsv', 'corpus'), ('corpus_train.tsv', 'corpus_train')):
        hdr, rr = read_tsv(tsv)
        for x in rr:
            p = f'results/w3_coexp_cache/{x[0]}.npz'
            if os.path.exists(p):
                try:
                    S[x[0]] = Study(x[0], base)
                except Exception as e:          # file still being written by the prep job
                    print(f'  skip {x[0]}: {e}', flush=True)
    return S


def organism(s):
    return 'mouse' if len(s.gids) > 60000 else 'human'


# ---------------------------------------------------------------------------------------------- baseline
def counts_by_bucket(Xu, b):
    """Xu: n x U dense uint8 -> A[NB, U] presence counts per bucket, Mb[NB]"""
    n, U = Xu.shape
    A = np.zeros((NB, U), np.float64)
    for bb in range(NB):
        sel = np.nonzero(b == bb)[0]
        if len(sel):
            A[bb] = Xu[sel].sum(0, dtype=np.int64)
    return A, np.bincount(b, minlength=NB).astype(np.float64)


def fit_base(A, Mb, kappa):
    """class-pooled per-bucket rate, shrunk toward by gene's own rate with weight kappa -> P[NB, U] (probabilities)"""
    k = A.sum(0)
    cls = np.where(k <= 8, k, 8 + np.round(4 * np.log2(np.maximum(k, 1) / 8))).astype(np.int64)
    T = np.zeros((NB, cls.max() + 1))
    for c in np.unique(cls):
        g = cls == c
        T[:, c] = A[:, g].sum(1) / (np.maximum(Mb, 1) * g.sum())
    Tg = T[:, cls]
    P = np.zeros((NB, A.shape[1]))
    for bb in range(NB):
        if Mb[bb] > 0:
            P[bb] = (A[bb] + kappa * Tg[bb]) / (Mb[bb] + kappa) if (Mb[bb] + kappa) > 0 else Tg[bb]
    ne = np.nonzero(Mb > 0)[0]
    for bb in range(NB):
        if Mb[bb] == 0:
            P[bb] = P[ne[np.argmin(np.abs(ne - bb))]]
    return np.clip(P, 1e-5, 1 - 1e-5)


def base_bits(P, AE, ME):
    """bits per bucket for cells E with per-bucket presence counts AE[NB,U] and sizes ME[NB]"""
    lp = np.log2(P); lq = np.log2(1 - P)
    return -((AE * lp).sum(1) + ((ME[:, None] - AE) * lq).sum(1))


KAPPAS = (4.0, 16.0, 64.0, 256.0, 1024.0, 1e9)   # 1e9 = pure class pooled (c_r2 like)


def best_base(AF, MF, AE, ME):
    best = None
    for kp in KAPPAS:
        P = fit_base(AF, MF, kp)
        bb = base_bits(P, AE, ME)
        if best is None or bb.sum() < best[0]:
            best = (bb.sum(), kp, P, bb)
    return best  # total bits, kappa, P, per-bucket bits


# ---------------------------------------------------------------------------------------------- hubs
def std_resid(Xu, b, P):
    Pb = P[b]
    return ((Xu.astype(np.float32) - Pb.astype(np.float32)) / np.sqrt(Pb * (1 - Pb)).astype(np.float32))


def greedy_hubs(Z, K, floor=0.05):
    """forward selection of K columns of Z (n x U standardised residuals) maximising the explained variance of all columns
    (pivoted Cholesky of the correlation matrix).  Returns ordered column indices."""
    Zt = torch.from_numpy(Z) if isinstance(Z, np.ndarray) else Z
    n = Zt.shape[0]
    C = (Zt.T @ Zt) / n
    sel = []
    mask = torch.zeros(C.shape[0], dtype=torch.bool)
    for _ in range(K):
        d = torch.diagonal(C).clamp_min(0)
        score = (C ** 2).sum(0) / (d + floor)
        score[mask] = -1
        k = int(torch.argmax(score))
        sel.append(k); mask[k] = True
        c = C[:, k].clone()
        C -= torch.outer(c, c) / (c[k] + floor)
    return sel


def var_hubs(P_list, Mb_list, K):
    """highest depth-averaged presence variance sum_b M_b P(1-P)"""
    v = 0
    for P, Mb in zip(P_list, Mb_list):
        v = v + (Mb[:, None] * P * (1 - P)).sum(0) / Mb.sum()
    return list(np.argsort(-v)[:K])


# ---------------------------------------------------------------------------------------------- features
DEPTH_SCALE = 0.5 / (np.log(8000.0 / 30.0) / 14 / np.sqrt(12.0))    # within-bucket log-depth, scaled to std 0.5


def ld_means(ld, b):
    """per-bucket mean of log interval-nnz (fallback: geometric bucket centre)"""
    m = np.zeros(NB)
    ed = np.concatenate([[20.0], EDGES, [12000.0]])
    for bb in range(NB):
        sel = b == bb
        m[bb] = ld[sel].mean() if sel.any() else 0.5 * (np.log(ed[bb]) + np.log(ed[bb + 1]))
    return m


def hub_feats(Xu, b, P, hub, ld, ldm, feat):
    """features f[n, 1+K]: column 0 = within-bucket log depth (scaled), columns 1.. = hub presence centred by the
    baseline rate of the hub in the cell's depth bucket"""
    dcol = ((ld - ldm[b]) * DEPTH_SCALE).astype(np.float32)[:, None]
    if feat == 'pres':
        H = Xu[:, hub].astype(np.float32) - P[b][:, hub].astype(np.float32)
        return np.concatenate([dcol, H], axis=1)
    raise ValueError(feat)


# ---------------------------------------------------------------------------------------------- training
def fit_W(studs, K, G, lam, iters=80, log=None):
    """studs: list of dict(F[n,K] f32 tensor, Y[n,G] uint8 tensor, b long tensor, O[NB,G] f32 tensor offsets).
    returns W[G,K] (float32 tensor)"""
    S = len(studs)
    W = torch.zeros(G, K, requires_grad=True)
    al = torch.zeros(S, G, requires_grad=True)
    opt = torch.optim.LBFGS([W, al], lr=1.0, max_iter=iters, history_size=15, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-6, tolerance_change=1e-9)
    ntot = sum(s['F'].shape[0] for s in studs)
    state = {'it': 0}

    def closure():
        opt.zero_grad()
        loss = 0.0
        for si, s in enumerate(studs):
            z = s['O'][s['b']] + al[si] + s['F'] @ W.T
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(z, s['Y'].float(), reduction='sum')
        loss = loss + lam * (W ** 2).sum() + 1e-3 * (al ** 2).sum()
        loss = loss / ntot
        loss.backward()
        state['it'] += 1
        return loss

    t0 = time.time()
    last = opt.step(closure)
    if log:
        log(f'    fit_W K={K} G={G} lam={lam} n={ntot} evals={state["it"]} loss/cell={float(last):.4f} {time.time()-t0:.0f}s')
    return W.detach()


def quantise(W):
    """W[G, 1+K]: column 0 (depth slope) stays float32 (a codec would use finer depth buckets instead); hub columns int8
    with a per-target scale.  returns (q int8 [G,K], s float32 [G], W_hat [G,1+K])"""
    Wh = W[:, 1:]
    if Wh.shape[1] == 0:
        return torch.zeros(W.shape[0], 0, dtype=torch.int8), torch.ones(W.shape[0]), W.clone()
    mx = Wh.abs().amax(1).clamp_min(1e-8)
    s = mx / 127.0
    q = torch.round(Wh / s[:, None]).clamp(-127, 127).to(torch.int8)
    return q, s, torch.cat([W[:, :1], q.float() * s[:, None]], dim=1)


def q_entropy_bits(q):
    if q.numel() == 0:
        return 0.0
    v = q.numpy().ravel().astype(np.int64) + 127
    c = np.bincount(v, minlength=255).astype(np.float64)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum() * q.numel())


# ---------------------------------------------------------------------------------------------- calibration + evaluation
def calibrate(F, Yb, b, O, What, P_t, Mb, mode='g', iters=6):
    """delta[NB, G] (or [1,G]) such that sum_c sigmoid(O[b]+W f + delta) matches the baseline's expected count."""
    n, G = Yb.shape
    z0 = O[b] + F @ What.T
    if mode == 'none':
        return torch.zeros(1, G)
    if mode == 'g':
        target = (torch.from_numpy(Mb).float()[:, None] * P_t).sum(0)    # expected count under baseline over all cells
        d = torch.zeros(G)
        for _ in range(iters):
            p = torch.sigmoid(z0 + d)
            f = p.sum(0) - target
            fp = (p * (1 - p)).sum(0).clamp_min(1e-6)
            d = (d - f / fp).clamp(-4, 4)
        return d[None, :]
    if mode == 'gb':
        d = torch.zeros(NB, G)
        for bb in range(NB):
            sel = (b == bb)
            if sel.sum() == 0:
                continue
            zb = z0[sel]
            target = float(Mb[bb]) * P_t[bb]
            db = torch.zeros(G)
            for _ in range(iters):
                p = torch.sigmoid(zb + db)
                f = p.sum(0) - target
                fp = (p * (1 - p)).sum(0).clamp_min(1e-6)
                db = (db - f / fp).clamp(-4, 4)
            d[bb] = db
        # empty buckets: copy nearest
        ne = [bb for bb in range(NB) if (b == bb).any()]
        for bb in range(NB):
            if bb not in ne:
                d[bb] = d[ne[int(np.argmin(np.abs(np.array(ne) - bb)))]]
        return d
    raise ValueError(mode)


def eval_bits(F, Y, b, O, What, delta):
    """per-bucket bits of target block Y (n x G uint8) under the conditional model"""
    n = F.shape[0]
    out = np.zeros(NB)
    ch = 2048
    for i in range(0, n, ch):
        sl = slice(i, min(n, i + ch))
        z = O[b[sl]] + F[sl] @ What.T
        z = z + (delta[b[sl]] if delta.shape[0] == NB else delta)
        nll = torch.nn.functional.softplus(z) - Y[sl].float() * z
        rows = nll.sum(1).numpy().astype(np.float64) / np.log(2.0)
        np.add.at(out, b[sl].numpy(), rows)
    return out


# ---------------------------------------------------------------------------------------------- pipeline
def dense(S, rows, U):
    sub = S.X[rows][:, U]
    return sub.toarray()


def rank_universe(rates, Usize):
    return np.argsort(-rates, kind='stable')[:Usize]


def pool_prepare(pool, Usize, cap, rng, log):
    """universe from mean per-study rate; per-study dense blocks, in-sample baselines (kappa=16)"""
    rate = np.mean([s.rate() for s in pool], axis=0)
    U = rank_universe(rate, Usize)
    P_list = []
    items = []
    for s in pool:
        rows = np.arange(s.n)
        if s.n > cap:
            rows = np.sort(rng.choice(s.n, cap, replace=False))
        Xu = dense(s, rows, U)
        b = s.b[rows]
        ld = np.log(s.d[rows].astype(np.float64))
        A, Mb = counts_by_bucket(Xu, b)
        P = fit_base(A, Mb, 16.0)
        items.append(dict(Xu=Xu, b=b, P=P, Mb=Mb, cid=s.cid, ld=ld, ldm=ld_means(ld, b)))
    return U, items


def make_study_tensors(it, hub, tgt, feat):
    F = torch.from_numpy(hub_feats(it['Xu'], it['b'], it['P'], hub, it['ld'], it['ldm'], feat))
    Y = torch.from_numpy(it['Xu'][:, tgt])
    O = torch.from_numpy(np.log(it['P'][:, tgt]) - np.log1p(-it['P'][:, tgt])).float()
    return dict(F=F, Y=Y, b=torch.from_numpy(it['b']), O=O)


def select_targets(U_n, hub, G):
    hubset = set(int(h) for h in hub)
    tgt = [j for j in range(U_n) if j not in hubset][:G]
    return np.array(tgt)


def logit_np(P):
    return np.log(P) - np.log1p(-P)



def append_rows(path, rows):
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter='\t')
        if new:
            w.writeheader()
        w.writerows(rows)


class Emitter:
    def __init__(self, T, args, log):
        self.T = T; self.args = args; self.log = log
        self.nnz_int_all = int(T.d.sum()); self.nnz_gene_all = int(T.nnz_gene.sum())
        self.nnz_b = np.bincount(T.b, weights=T.d, minlength=NB)
        self.cells_b = np.bincount(T.b, minlength=NB)

    def emit(self, regime, hubsel, feat, K, Gs, kappa, lam, dmode, base_b, cond_b, side_raw, side_ent, tgt_share, note=''):
        T = self.T; tag = T.cid
        gain_b = base_b - cond_b
        gain = float(gain_b.sum())
        ni, ng = self.nnz_int_all, self.nnz_gene_all
        row = dict(file=tag, regime=regime, hubsel=hubsel, feat=feat, K=K, G=Gs, kappa=kappa, lam=lam, dmode=dmode,
                   cells=T.n, nnz_int=ni, nnz_gene=ng, tgt_share=round(tgt_share, 4),
                   base_bits=round(float(base_b.sum()), 1), cond_bits=round(float(cond_b.sum()), 1), gain_bits=round(gain, 1),
                   gain_bpn_int=round(gain / ni, 5), gain_bpn_gene=round(gain / ng, 5),
                   base_bpn_tgt=round(float(base_b.sum()) / max(tgt_share * ng, 1), 4),
                   side_bits_raw=int(side_raw), side_bits_ent=int(side_ent),
                   net_bpn_int_raw=round((gain - side_raw) / ni, 5), net_bpn_int_ent=round((gain - side_ent) / ni, 5), note=note)
        show = ('regime', 'hubsel', 'feat', 'K', 'G', 'kappa', 'lam', 'dmode', 'gain_bits', 'gain_bpn_int', 'gain_bpn_gene',
                'net_bpn_int_raw', 'net_bpn_int_ent', 'tgt_share', 'note')
        self.log('  ' + ' '.join(f'{k}={v}' for k, v in row.items() if k in show))
        brs = []
        for bb in range(NB):
            brs.append(dict(file=tag, regime=regime, hubsel=hubsel, feat=feat, K=K, G=Gs, lam=lam, dmode=dmode, note=note, bucket=bb,
                            nnz_lo=round(float(BUCKET_LO[bb]), 1), cells=int(self.cells_b[bb]), nnz_int=int(self.nnz_b[bb]),
                            base_bits=round(float(base_b[bb]), 1), cond_bits=round(float(cond_b[bb]), 1),
                            gain_bits=round(float(gain_b[bb]), 1), gain_bpn=round(float(gain_b[bb]) / max(self.nnz_b[bb], 1), 5)))
        append_rows(self.args.out, [row]); append_rows(self.args.out_bucket, brs)


def fold_state(T, Fh, Eh, U, G0):
    """dense blocks + baseline (kappa chosen on the first G0 genes of U) for fitting half Fh / scored half Eh"""
    XF = dense(T, Fh, U); XE = dense(T, Eh, U)
    bF = T.b[Fh]; bE = T.b[Eh]
    ldF = np.log(T.d[Fh].astype(np.float64)); ldE = np.log(T.d[Eh].astype(np.float64))
    AF, MF = counts_by_bucket(XF, bF); AE, ME = counts_by_bucket(XE, bE)
    _, kp, _, _ = best_base(AF[:, :G0], MF, AE[:, :G0], ME)
    P = fit_base(AF, MF, kp)
    return dict(XF=XF, XE=XE, bF=bF, bE=bE, AF=AF, MF=MF, AE=AE, ME=ME, kp=kp, P=P, ldF=ldF, ldE=ldE, ldm=ld_means(ldF, bF))


def eval_config(st, hub, tgt, W, quant, feat, dmode):
    """returns (base_bits[NB], cond_bits[NB], tgt_nnz) for the scored half"""
    P = st['P']
    What = quantise(W)[2] if quant else W
    base_b = base_bits(P[:, tgt], st['AE'][:, tgt], st['ME'])
    O = torch.from_numpy(logit_np(P[:, tgt])).float()
    FF = torch.from_numpy(hub_feats(st['XF'], st['bF'], P, hub, st['ldF'], st['ldm'], feat))
    FE = torch.from_numpy(hub_feats(st['XE'], st['bE'], P, hub, st['ldE'], st['ldm'], feat))
    delta = calibrate(FF, None, torch.from_numpy(st['bF']), O, What, torch.from_numpy(P[:, tgt]).float(), st['MF'], dmode)
    cond_b = eval_bits(FE, torch.from_numpy(st['XE'][:, tgt]), torch.from_numpy(st['bE']), O, What, delta)
    return base_b, cond_b, float(st['XE'][:, tgt].sum())


WCACHE = {}


def run_file(T, S, args, log, which=('i', 'ii')):
    rng = np.random.default_rng(12345)
    org = organism(T)
    tag = T.cid
    others = [s for s in S.values() if organism(s) == org and s.study != T.study]
    perm = rng.permutation(T.n)
    halves = [np.sort(perm[: T.n // 2]), np.sort(perm[T.n // 2:])]
    em = Emitter(T, args, log)
    Ks = sorted(args.K); Kmax = max(Ks); Usize = args.U
    Gs_list = args.G
    ng_all = em.nnz_gene_all
    t00 = time.time()

    # --------------------------------------------------------------- regime i: weights from other studies
    if 'i' in which and others:
        U, items = pool_prepare(others, Usize, args.cap, rng, log)
        log(f'[{tag}] regime i pool={[s.cid for s in others]} U={len(U)} ({time.time()-t00:.0f}s)')
        hub_sets = {}
        for hs in args.hubsel:
            if hs == 'greedy':
                Zs = []
                for it in items:
                    nsub = min(len(it['b']), 3000)
                    sel = rng.choice(len(it['b']), nsub, replace=False)
                    Zs.append(std_resid(it['Xu'][sel], it['b'][sel], it['P']))
                hub_sets[hs] = greedy_hubs(np.concatenate(Zs), Kmax)
                del Zs
            else:
                hub_sets[hs] = var_hubs([it['P'] for it in items], [it['Mb'] for it in items], Kmax)
        log(f'[{tag}] hubs chosen ({time.time()-t00:.0f}s)')
        folds = [fold_state(T, halves[fi], halves[1 - fi], U, 2048) for fi in range(2)]
        for hs in args.hubsel:
            for K in Ks:
                hub = np.array(hub_sets[hs][:K], dtype=np.int64)
                append_rows(args.out_hubs, [dict(kind='hub_list', file=tag, regime='i', hubsel=hs, K=K,
                                                 genes=';'.join(f'{T.gids[U[h]]}|{T.gsym[U[h]]}' for h in hub),
                                                 note='pool=' + ','.join(s.cid for s in others))])
                for g_ in Gs_list:
                    tgt = select_targets(len(U), hub, g_)
                    for feat in args.feat:
                        for lam in args.lam:
                            studs = [make_study_tensors(it, hub, tgt, feat) for it in items]
                            ck = (frozenset(s.cid for s in others), hs, K, len(tgt), feat, lam)
                            if ck in WCACHE:
                                W = WCACHE[ck]
                            else:
                                W = fit_W(studs, K + 1, len(tgt), lam, iters=args.iters, log=log)
                                WCACHE[ck] = W
                            del studs
                            if args.save_w and K in args.save_w:
                                q, sc, _ = quantise(W)
                                np.savez_compressed(f'results/w3_coexp_cache/W_{tag}_{hs}_K{K}_G{len(tgt)}_{feat}_lam{lam}.npz',
                                                    q=q.numpy(), s=sc.numpy(), hub=U[hub], tgt=U[tgt], hub_ids=T.gids[U[hub]],
                                                    hub_sym=T.gsym[U[hub]], tgt_ids=T.gids[U[tgt]])
                            for dmode in args.dmode:
                                for quant in ((True, False) if args.also_float else (True,)):
                                    tb = np.zeros(NB); tc = np.zeros(NB); tn = 0.0; kap = []
                                    for st in folds:
                                        b_, c_, n_ = eval_config(st, hub, tgt, W, quant, feat, dmode)
                                        tb += b_; tc += c_; tn += n_; kap.append(st['kp'])
                                    em.emit('i', hs, feat, K, len(tgt), float(np.mean(kap)), lam, dmode, tb, tc, 0, 0, tn / ng_all,
                                            note='' if quant else 'float_weights')
        del items, folds

    # --------------------------------------------------------------- regime ii: weights from this file's other half (cross-fitted)
    if 'ii' in which:
        acc = {}
        for fi in range(2):
            Fh, Eh = halves[fi], halves[1 - fi]
            rF = np.asarray(T.X[Fh].sum(0)).ravel() / len(Fh)
            U = rank_universe(rF, Usize)
            st = fold_state(T, Fh, Eh, U, 2048)
            P16 = fit_base(st['AF'], st['MF'], 16.0)
            for hs in args.hubsel:
                if hs == 'greedy':
                    Z = std_resid(st['XF'], st['bF'], P16)
                    hubs_all = greedy_hubs(Z, Kmax)
                    del Z
                else:
                    hubs_all = var_hubs([P16], [st['MF']], Kmax)
                for K in Ks:
                    hub = np.array(hubs_all[:K], dtype=np.int64)
                    if fi == 0:
                        append_rows(args.out_hubs, [dict(kind='hub_list', file=tag, regime='ii', hubsel=hs, K=K,
                                                         genes=';'.join(f'{T.gids[U[h]]}|{T.gsym[U[h]]}' for h in hub),
                                                         note='fold0 (fit on half A)')])
                    for g_ in Gs_list:
                        tgt = select_targets(len(U), hub, g_)
                        for feat in args.feat:
                            for lam in args.lam_ii:
                                it = dict(Xu=st['XF'], b=st['bF'], P=P16, ld=st['ldF'], ldm=st['ldm'])
                                W = fit_W([make_study_tensors(it, hub, tgt, feat)], K + 1, len(tgt), lam, iters=args.iters, log=None)
                                q, sc, _ = quantise(W)
                                for dmode in args.dmode[:1]:
                                    b_, c_, n_ = eval_config(st, hub, tgt, W, True, feat, dmode)
                                    key = (hs, K, len(tgt), feat, lam, dmode)
                                    a = acc.setdefault(key, dict(tb=np.zeros(NB), tc=np.zeros(NB), tn=0.0, kap=[], side_raw=0, side_ent=0))
                                    a['tb'] += b_; a['tc'] += c_; a['tn'] += n_; a['kap'].append(st['kp'])
                                    if fi == 0:
                                        idb = int(np.ceil(np.log2(Usize)))
                                        a['side_raw'] = len(tgt) * K * 8 + len(tgt) * 32 + K * idb      # int8 weights + fp16 scale + fp16 depth slope
                                        a['side_ent'] = q_entropy_bits(q) + len(tgt) * 32 + K * idb
            log(f'[{tag}] regime ii fold {fi} done ({time.time()-t00:.0f}s)')
            del st
        for key, a in acc.items():
            hs, K, Gt, feat, lam, dmode = key
            em.emit('ii', hs, feat, K, Gt, float(np.mean(a['kap'])), lam, dmode, a['tb'], a['tc'], a['side_raw'], a['side_ent'],
                    a['tn'] / ng_all)

    # --------------------------------------------------------------- in-sample reference (labelled upper bound)
    if 'in' in which:
        allr = np.arange(T.n)
        rF = np.asarray(T.X.sum(0)).ravel() / T.n
        U = rank_universe(rF, Usize)
        st = fold_state(T, allr, allr, U, 2048)
        P16 = fit_base(st['AF'], st['MF'], 16.0)
        for hs in args.hubsel:
            if hs == 'greedy':
                Z = std_resid(st['XF'], st['bF'], P16)
                hubs_all = greedy_hubs(Z, Kmax)
                del Z
            else:
                hubs_all = var_hubs([P16], [st['MF']], Kmax)
            for K in Ks:
                hub = np.array(hubs_all[:K], dtype=np.int64)
                for g_ in Gs_list:
                    tgt = select_targets(len(U), hub, g_)
                    for feat in args.feat:
                        for lam in args.lam_ii:
                            W = fit_W([make_study_tensors(dict(Xu=st['XF'], b=st['bF'], P=P16, ld=st['ldF'], ldm=st['ldm']), hub, tgt, feat)],
                                      K + 1, len(tgt), lam, iters=args.iters, log=None)
                            b_, c_, n_ = eval_config(st, hub, tgt, W, True, feat, args.dmode[0])
                            em.emit('in', hs, feat, K, len(tgt), st['kp'], lam, args.dmode[0], b_, c_, 0, 0, n_ / ng_all,
                                    note='in-sample upper bound')
    log(f'[{tag}] done in {time.time()-t00:.0f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', nargs='+', required=True)
    ap.add_argument('--K', type=int, nargs='+', default=[64])
    ap.add_argument('--G', type=int, nargs='+', default=[2048])
    ap.add_argument('--U', type=int, default=4096)
    ap.add_argument('--hubsel', nargs='+', default=['greedy'])
    ap.add_argument('--feat', nargs='+', default=['pres'])
    ap.add_argument('--lam', type=float, nargs='+', default=[30.0])
    ap.add_argument('--lam_ii', type=float, nargs='+', default=[30.0])
    ap.add_argument('--dmode', nargs='+', default=['g'])
    ap.add_argument('--cap', type=int, default=6000)
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--regimes', default='i,ii')
    ap.add_argument('--also_float', action='store_true')
    ap.add_argument('--save_w', type=int, nargs='*', default=[])
    ap.add_argument('--out', default='results/w3_coexp_s1.tsv')
    ap.add_argument('--out_bucket', default='results/w3_coexp_s1_bucket.tsv')
    ap.add_argument('--out_hubs', default='results/w3_coexp_s1_hublists.tsv')
    args = ap.parse_args()
    S = load_all()
    log = lambda s: print(s, flush=True)
    for t in args.test:
        log(f'== {t}  cells={S[t].n}  threads={torch.get_num_threads()}')
        run_file(S[t], S, args, log, which=tuple(args.regimes.split(',')))


if __name__ == '__main__':
    main()
