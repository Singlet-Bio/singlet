"""Idea 3, stage A measurements: gene/interval factorisation.  Usage: python -P tools/idea3_measure_a.py [sample ...]
Appends one JSON line per sample x kind to results/idea3_a_measure.jsonl."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idea3_common import *
from idea3_prep import *
from scipy.special import gammaln

LN2 = np.log(2.0)
ALPHAS = [0.05, 0.25, 1.0]
KAPPAS = [2.0, 8.0, 32.0]
MSHR = [5.0, 30.0]

def lg2(a):
    return gammaln(a) / LN2

def pool_for(sid, kind):
    sp_ = SPECIES[sid]; st = STUDY[sid]
    return [s for s in STUDY if SPECIES[s] == sp_ and STUDY[s] != st and os.path.exists(cache_path(s, kind))]

def k_cost_bits(k):
    """per-cell nnz coded as zigzag delta vs previous cell: static order-0 entropy of the bit-length symbol + raw offset bits."""
    k = k.astype(np.int64)
    d = np.diff(k, prepend=0)
    z = np.where(d >= 0, 2 * d, -2 * d - 1)
    sb = np.zeros(len(z), np.int64)
    nz = z > 0
    sb[nz] = np.floor(np.log2(z[nz])).astype(np.int64) + 1
    h = np.bincount(sb)
    h = h[h > 0]
    ent = -(h * np.log2(h / h.sum())).sum()
    raw = np.maximum(sb - 1, 0).sum()
    return float(ent + raw)

def xent_pres(n_bf, N_b, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float((-(n_bf * np.log2(p) + (N_b[:, None] - n_bf) * np.log2(1 - p))).sum())

def class_of(rate):
    c = np.full(len(rate), 63, np.int64)
    ok = rate > 0
    c[ok] = np.clip(np.floor(3 * (-np.log2(rate[ok]))), 0, 63).astype(np.int64)
    return c

def count_bits(feat, x, col, b_col, cls):
    """counts-given-presence: contexts (cls>>2 [16], bucket//4 [6]); symbol = min(x,16)-1 plus escape raw bits (static empirical entropy)."""
    ctx = (cls[feat] >> 2) * 6 + (b_col[col] // 4)
    sym = np.minimum(x, 16) - 1
    h = np.bincount(ctx * 16 + sym, minlength=96 * 16).reshape(96, 16).astype(np.float64)
    tot = h.sum(1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        e = np.where(h > 0, h * np.log2(tot / h), 0).sum()
    esc = x >= 16
    escbits = (2 * np.floor(np.log2(x[esc] - 15)) + 1).sum()
    return float(e + escbits)

def level_costs(col, feat, x, k, F, pres_own, N_own, pool_pres, pool_N, tag, pool_ids=(), kind=None, pkey=None, nkey=None):
    out = {}
    b_col = bucket(k)
    ntot = N_own.sum()
    out[tag + '_kbits'] = k_cost_bits(k)
    n_bf = pres_own.toarray().astype(np.float64)
    rate_own = n_bf.sum(0) / max(ntot, 1)
    Nown = N_own.astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        p_in = np.where(Nown[:, None] > 0, n_bf / np.maximum(Nown[:, None], 1), 0)
    out[tag + '_pres_in_bf'] = xent_pres(n_bf, Nown, p_in)
    cls = class_of(rate_own)
    ncls = 64
    nc = np.bincount(cls, minlength=ncls).astype(np.float64)
    n_cb = np.zeros((ncls, B))
    for b in range(B):
        n_cb[:, b] = np.bincount(cls, weights=n_bf[b], minlength=ncls)
    p_cb = n_cb / np.maximum(nc[:, None] * Nown[None, :], 1)
    out[tag + '_pres_in_cls'] = xent_pres(n_bf, Nown, p_cb[cls].T)
    h = np.bincount(cls, minlength=ncls); h = h[h > 0]
    out[tag + '_cls_side'] = float(-(h * np.log2(h / h.sum())).sum() + ncls * B * 12)
    out[tag + '_cnt_in_cls'] = count_bits(feat, x, col, b_col, cls)
    if pool_pres is not None:
        Np = pool_N.astype(np.float64)
        rate_p = pool_pres.sum(0) / max(Np.sum(), 1)
        clsp = class_of(rate_p)
        ncp = np.bincount(clsp, minlength=ncls).astype(np.float64)
        n_cbp = np.zeros((ncls, B))
        for b in range(B):
            n_cbp[:, b] = np.bincount(clsp, weights=pool_pres[b], minlength=ncls)
        p0 = n_cbp / np.maximum(ncp[:, None] * np.maximum(Np[None, :], 1), 1)
        nb = [bb for bb in range(B) if Np[bb] > 0]
        for b in range(B):
            if Np[b] == 0 and nb:
                j = min(nb, key=lambda t: abs(t - b)); p0[:, b] = p0[:, j]
        for mm in MSHR:
            p = (pool_pres + mm * p0[clsp].T) / (Np[:, None] + mm)
            out[tag + f'_pres_oos_m{int(mm)}'] = xent_pres(n_bf, Nown, p)
        out[tag + '_pres_oos_cls'] = xent_pres(n_bf, Nown, p0[clsp].T)
        out[tag + '_cnt_oos_cls'] = count_bits(feat, x, col, b_col, clsp)
        out[tag + '_model_entries'] = int(F * B)
        # best single pool member (oracle choice, log2(pool) bits of side info): class-only table
        best = None
        for sid2 in pool_ids:
            d2 = load(sid2, kind)
            pp = d2[pkey].toarray().astype(np.float64); NN = d2[nkey].astype(np.float64)
            rate2 = pp.sum(0) / max(NN.sum(), 1); c2 = class_of(rate2)
            nc2 = np.bincount(c2, minlength=ncls).astype(np.float64)
            ncb2 = np.zeros((ncls, B))
            for b in range(B):
                ncb2[:, b] = np.bincount(c2, weights=pp[b], minlength=ncls)
            q0 = ncb2 / np.maximum(nc2[:, None] * np.maximum(NN[None, :], 1), 1)
            nb2 = [bb for bb in range(B) if NN[bb] > 0]
            for b in range(B):
                if NN[b] == 0 and nb2:
                    j = min(nb2, key=lambda t: abs(t - b)); q0[:, b] = q0[:, j]
            xe = xent_pres(n_bf, Nown, q0[c2].T)
            if best is None or xe < best[0]:
                best = (xe, sid2, c2)
        if best is not None:
            out[tag + '_pres_oos_cls_best'] = best[0]
            out[tag + '_pres_oos_cls_best_id'] = best[1]
            out[tag + '_cnt_oos_cls_best'] = count_bits(feat, x, col, b_col, best[2])
    return out

def analyse(sid, kind, S, pool_ids):
    M = S.E if kind == 'exon' else S.I
    gor = S.egene if kind == 'exon' else S.igene
    first = S.efirst if kind == 'exon' else S.ifirst
    G = len(first) - 1
    m, n = M.shape
    col, row, x = entries(M)
    gcol, gg, gc, st = gene_groups(col, row, x, gor)
    U = int(x.sum()); nnzI = len(x); nnzG = len(gc)
    ng = np.diff(first)
    own = load(sid, kind)
    r = dict(sample=sid, kind=kind, m=m, G=G, n=n, nnz_int=nnzI, nnz_gene=nnzG, U=U,
             ivl_per_gene_mean=float(ng.mean()), ivl_per_gene_med=float(np.median(ng)), ivl_per_gene_p90=float(np.percentile(ng, 90)), ivl_per_gene_max=int(ng.max()),
             umi_weighted_ivl_per_gene=float((own['Tg'] * ng).sum() / U), frac_umi_single_ivl_genes=float(own['Tg'][ng == 1].sum() / U),
             frac_umi_c1=float(gc[gc == 1].sum() / U), nnz_int_per_nnz_gene=nnzI / nnzG, umi_per_nnz_int=U / nnzI, umi_per_nnz_gene=U / nnzG)
    T_i = own['T_i'].astype(np.float64); Tg_of_i = own['Tg'][gor].astype(np.float64)
    ok = T_i > 0
    H_in = float((T_i[ok] * np.log2(Tg_of_i[ok] / T_i[ok])).sum())
    r['H_I_given_G_in_bits_per_umi'] = H_in / U
    cg_entry = np.repeat(gc, np.diff(np.append(st, nnzI)))
    beta = np.where(cg_entry == 1, 0, np.where(cg_entry == 2, 1, np.where(cg_entry <= 4, 2, 3)))
    Tb = np.bincount(beta * m + row, weights=x, minlength=4 * m).reshape(4, m)
    Tbg = np.zeros((4, G))
    for q in range(4):
        Tbg[q] = np.bincount(gor, weights=Tb[q], minlength=G)
    Hb = 0.0
    for q, nm in enumerate(['c1', 'c2', 'c3_4', 'c5p']):
        okq = Tb[q] > 0
        tq = Tb[q].sum()
        hq = float((Tb[q][okq] * np.log2(Tbg[q][gor][okq] / Tb[q][okq])).sum())
        Hb += hq
        r['umi_frac_' + nm] = float(tq / U)
        r['H_I_given_G_in_' + nm] = hq / max(tq, 1)
    r['H_I_given_G_cbucket_in_bits_per_umi'] = Hb / U
    coeff = float(lg2(gc + 1).sum() - lg2(x + 1).sum())
    r['multiset_saving_bits'] = coeff
    r['split_in_iid_bits'] = H_in
    r['split_in_multiset_bits'] = H_in - coeff
    Tg_own = own['Tg'].astype(np.float64)
    def dm_bits(alpha_i, A):
        gain = lg2(alpha_i[ok] + T_i[ok]) - lg2(alpha_i[ok])
        per_g = np.bincount(gor[ok], weights=gain, minlength=G)
        return float(-(lg2(A) - lg2(A + Tg_own) + per_g).sum() - coeff)
    def iid_bits(p):
        return float((x * -np.log2(p[row])).sum())
    # length prior (zero-shot: depends only on interval coordinates, which the decoder has via the names)
    L = np.array([int(t.split('-')[1]) - int(t.split('-')[0].split(':')[1]) + 1 for t in (rn.rsplit('_', 1)[-1] for rn in (S.erows if kind == 'exon' else S.irows))], np.float64)
    L = np.maximum(L, 1.0)
    p_len = L / np.bincount(gor, weights=L, minlength=G)[gor]
    r['split_len_iid_bits'] = iid_bits(p_len); r['split_len_multiset_bits'] = r['split_len_iid_bits'] - coeff
    for kap in KAPPAS:
        r[f'split_adapt_lenprior_k{kap}_multiset_bits'] = dm_bits(kap * p_len, np.full(G, kap))
    if pool_ids:
        Tn = sum(load(s, kind)['T_i'] for s in pool_ids).astype(np.float64)
        Tng = np.bincount(gor, weights=Tn, minlength=G)
        best = None
        for a in ALPHAS:
            p = (Tn + a) / (Tng[gor] + a * ng[gor])
            xc = iid_bits(p)
            r[f'split_oos_a{a}_iid_bits'] = xc
            r[f'split_oos_a{a}_multiset_bits'] = xc - coeff
            if best is None or xc < best[1]:
                best = (a, xc)
        r['split_oos_best_alpha'] = best[0]
        # best single pool member as static / prior
        bs = None
        for s2 in pool_ids:
            T2 = load(s2, kind)['T_i'].astype(np.float64)
            T2g = np.bincount(gor, weights=T2, minlength=G)
            p2 = (T2 + 0.25) / (T2g[gor] + 0.25 * ng[gor])
            xc = iid_bits(p2)
            if bs is None or xc < bs[1]:
                bs = (s2, xc, p2)
        r['split_oosbest_id'] = bs[0]
        r['split_oosbest_iid_bits'] = bs[1]; r['split_oosbest_multiset_bits'] = bs[1] - coeff
        for kap in KAPPAS:
            r[f'split_adapt_oosbestprior_k{kap}_multiset_bits'] = dm_bits(kap * bs[2], np.full(G, kap))
    for a in [0.05, 0.25, 0.5]:
        alpha_i = np.full(m, a); A = a * ng.astype(np.float64)
        gain = lg2(alpha_i[ok] + T_i[ok]) - lg2(alpha_i[ok])
        per_g = np.bincount(gor[ok], weights=gain, minlength=G)
        seq = -(lg2(A) - lg2(A + own['Tg']) + per_g).sum()
        r[f'split_adapt_flat_a{a}_multiset_bits'] = float(seq - coeff)
    pool_I = pool_G = None; NI_p = NG_p = None
    if pool_ids:
        pls = [load(s, kind) for s in pool_ids]
        pool_I = sum(p['presI'].toarray().astype(np.float64) for p in pls); NI_p = sum(p['NI'] for p in pls)
        pool_G = sum(p['presG'].toarray().astype(np.float64) for p in pls); NG_p = sum(p['NG'] for p in pls)
        del pls
    r.update(level_costs(col, row, x, own['kI'], m, own['presI'], own['NI'], pool_I, NI_p, 'int', pool_ids, kind, 'presI', 'NI'))
    r.update(level_costs(gcol, gg, gc, own['kG'], G, own['presG'], own['NG'], pool_G, NG_p, 'gene', pool_ids, kind, 'presG', 'NG'))
    r['pool'] = ','.join(pool_ids)
    return r

if __name__ == '__main__':
    sids = sys.argv[1:] or TEST
    for sid in sids:
        S = Sample(sid)
        for kind in ('exon', 'intron'):
            t = time.time()
            rr = analyse(sid, kind, S, pool_for(sid, kind))
            print(sid, kind, 'done', round(time.time() - t, 1), flush=True)
            with open(os.path.join(ROOT, 'results', 'idea3_a_measure.jsonl'), 'a') as f:
                f.write(json.dumps(rr, default=float) + '\n')
