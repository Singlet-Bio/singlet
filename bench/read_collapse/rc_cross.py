#!/usr/bin/env python3
"""Cross-sample analyses: exact-cache hit rates (Q1), top-K prefilter and k-mer classifier (Q3), pool growth and memory (Q4).
Every held-out number uses only the other samples as training or pool."""
import sys, os, json, time, argparse
import numpy as np

U = np.uint64
W_MULTI = 30.0
T0 = time.time()
INV = np.iinfo(np.uint64).max


def log(*a):
    print('[rc_cross %6.0fs]' % (time.time() - T0), *a, file=sys.stderr, flush=True)


def cost_of(bits):
    return 1.0 + (W_MULTI - 1.0) * ((bits & 1) > 0)


CLASSES = [('all', lambda b: b >= 0), ('simple', lambda b: b == 0), ('multi_NH>1', lambda b: (b & 1) > 0),
           ('softclipped', lambda b: (b & 2) > 0), ('spliced', lambda b: (b & 4) > 0),
           ('heavy_proxy_any', lambda b: (b & 7) > 0), ('unmapped', lambda b: (b & 8) > 0)]


def hitstats(d, hit):
    cnt = d['cnt'].astype(np.int64)
    cost = cost_of(d['bits'])
    out = {}
    for name, f in CLASSES:
        m = f(d['bits'])
        if not m.any():
            continue
        out[name] = dict(distinct=int(m.sum()), reads_hit=float(cnt[m & hit].sum() / cnt[m].sum()),
                         distinct_hit=float((m & hit).sum() / m.sum()), cost_hit=float(cost[m & hit].sum() / cost[m].sum()))
    return out


def inset(sorted_arr, q):
    if len(sorted_arr) == 0:
        return np.zeros(len(q), bool)
    i = np.searchsorted(sorted_arr, q)
    i[i == len(sorted_arr)] = 0
    return sorted_arr[i] == q


def kmer_matrix(X, k):
    n, L = X.shape
    nw = max(L - k + 1, 0)
    out = np.full((n, nw), INV, U)
    code = np.zeros(n, U)
    cs = np.cumsum(X == 4, axis=1, dtype=np.int16)
    mask = U((1 << (2 * k)) - 1)
    for j in range(L):
        code = ((code << U(2)) | (X[:, j] & 3).astype(U)) & mask
        if j >= k - 1:
            nb = cs[:, j] - (cs[:, j - k] if j >= k else 0)
            out[:, j - k + 1] = np.where(nb == 0, code, INV)
    return out


def to_codes(seqS):
    L = seqS.dtype.itemsize
    a = np.frombuffer(seqS.tobytes(), dtype=np.uint8).reshape(len(seqS), L)
    lut = np.full(256, 4, np.uint8)
    lut[[65, 67, 71, 84]] = [0, 1, 2, 3]
    lut[[97, 99, 103, 116]] = [0, 1, 2, 3]
    return lut[a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)     # tsv: name chem tissue genome fq
    ap.add_argument('--data', required=True)        # dir with <sample>/distinct.npz
    ap.add_argument('--out', required=True)
    ap.add_argument('--kmer', type=int, default=25)
    ap.add_argument('--train-cap', type=int, default=300000)
    ap.add_argument('--test-cap', type=int, default=1500000)
    ap.add_argument('--orders', type=int, default=8)
    A = ap.parse_args()
    meta = {}
    for ln in open(A.samples):
        f = ln.rstrip('\n').split('\t')
        if len(f) >= 4 and not ln.startswith('#'):
            meta[f[0]] = dict(chem=f[1], tissue=f[2], genome=f[3])
    S = [s for s in meta if os.path.exists(os.path.join(A.data, s, 'distinct.npz'))]
    log('samples', S)
    D = {}
    for s in S:
        z = np.load(os.path.join(A.data, s, 'distinct.npz'), allow_pickle=False)
        D[s] = {k: z[k] for k in z.files}
    ns = len(S)
    R = dict(samples=S, meta={s: meta[s] for s in S}, sizes={s: dict(reads=int(D[s]['cnt'].sum()), distinct=int(len(D[s]['ukey']))) for s in S})
    if ns < 2:
        json.dump(R, open(A.out, 'w'), indent=1)
        return
    # ---- union table with presence bits
    ak = np.concatenate([D[s]['ukey'] for s in S])
    sid = np.concatenate([np.full(len(D[s]['ukey']), i, np.uint16) for i, s in enumerate(S)])
    order = np.argsort(ak, kind='stable')
    sk = ak[order]
    ss = (np.uint32(1) << sid[order].astype(np.uint32)).astype(np.uint32)
    first = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    pres = np.bitwise_or.reduceat(ss, first)
    ukall = sk[first]
    firstsid = sid[order][first]
    # signature of the first sample holding each key (cross-sample exactness check)
    sigall = np.concatenate([D[s]['sig'] for s in S])[order][first]
    del ak, sid, order, sk, ss
    NU = len(ukall)
    log('union entries', NU)
    POS = {s: np.searchsorted(ukall, D[s]['ukey']) for s in S}
    PRES = {s: pres[POS[s]] for s in S}
    bit = {s: np.uint32(1 << i) for i, s in enumerate(S)}
    R['union_entries_all_samples'] = int(NU)

    def cat(H, o):
        a, b = meta[H], meta[o]
        sc, st = a['chem'] == b['chem'], a['tissue'] == b['tissue']
        return 'A_same_chem_same_tissue' if sc and st else 'B_same_chem_diff_tissue' if sc else 'C_diff_chem_same_tissue' if st else 'D_diff_chem_diff_tissue'

    # ---- Q1 cross-sample by category
    q1 = []
    for H in S:
        others = [s for s in S if s != H]
        pools = {}
        for o in others:
            pools.setdefault(cat(H, o), []).append(o)
        pools['ALL_others'] = others
        for pname, ps in pools.items():
            pm = np.uint32(sum(int(bit[o]) for o in ps))
            hit = (PRES[H] & pm) != 0
            r = dict(heldout=H, pool=pname, pool_samples=ps, stats=hitstats(D[H], hit))
            # exactness: where the first holder of the key is another sample with the same genome, is the record equal?
            fs = firstsid[POS[H]]
            n = e = 0
            for j, o in enumerate(S):
                if o == H or o not in ps or meta[o]['genome'] != meta[H]['genome']:
                    continue
                mm = hit & (fs == j) & (D[H]['bits'] < 8)
                n += int(mm.sum())
                e += int((sigall[POS[H]][mm] == D[H]['sig'][mm]).sum())
            r['record_check'] = dict(hits_checked=n, same_record=e)
            q1.append(r)
        log('q1', H)
    R['q1_cross'] = q1

    # ---- Q4 pool growth (random orders) and memory per entry
    rng = np.random.default_rng(7)
    q4 = []
    for H in S:
        others = [s for s in S if s != H]
        sizes = sorted(set([1, 2, 3, 5, len(others)]) & set(range(1, len(others) + 1)))
        for sz in sizes:
            reps = 1 if sz == len(others) else A.orders
            acc = []
            for _ in range(reps):
                ps = list(rng.permutation(others)[:sz])
                pm = np.uint32(sum(int(bit[o]) for o in ps))
                hit = (PRES[H] & pm) != 0
                st = hitstats(D[H], hit)
                entries = int(np.count_nonzero(pres & pm))
                acc.append((st, entries))
            keys = ['all', 'multi_NH>1', 'simple']
            q4.append(dict(heldout=H, pool_size=sz, orders=reps, pool_entries=float(np.mean([e for _, e in acc])),
                           **{k: {m: float(np.mean([a[0][k][m] for a in acc if k in a[0]])) for m in ('reads_hit', 'distinct_hit', 'cost_hit')}
                              for k in keys if all(k in a[0] for a in acc)}))
        log('q4', H)
    R['q4_pool_growth'] = q4
    nops = np.concatenate([D[s]['nops'][D[s]['bits'] < 8] for s in S]).astype(np.float64)
    nh = np.maximum(np.concatenate([D[s]['nh'][D[s]['bits'] < 8] for s in S]).astype(np.float64), 1)
    rec_bytes = float(((9 + 2 * nops) * nh).mean())
    R['memory'] = dict(mean_record_bytes=rec_bytes, mean_nh=float(nh.mean()), mean_cigar_ops=float(nops.mean()))
    log('memory', R['memory'])

    # ---- Q3a top-K exact table
    totalf = np.zeros(NU)
    FR = {}
    for s in S:
        f = D[s]['cnt'] / D[s]['cnt'].sum()
        FR[s] = f
        totalf += np.bincount(POS[s], weights=f, minlength=NU)
    gname_ids = {}
    for s in S:
        for nm in D[s]['names']:
            gname_ids.setdefault(str(nm), len(gname_ids))
    gl = list(gname_ids.keys())
    def gcodes(s):
        m = np.array([gname_ids[str(x)] for x in D[s]['names']], np.int32)
        return np.where(D[s]['gene'] >= 0, m[np.maximum(D[s]['gene'], 0)], -1).astype(np.int32)
    gcode = np.concatenate([gcodes(s) for s in S])
    sidc = np.concatenate([np.full(len(D[s]['ukey']), i, np.int32) for i, s in enumerate(S)])
    gclsc = np.concatenate([D[s]['gcls'] for s in S])
    posc = np.concatenate([POS[s] for s in S])
    # gene of each union key from the first sample that holds it (by sample order)
    ordr = np.lexsort((sidc, posc))
    fpos = np.flatnonzero(np.r_[True, posc[ordr][1:] != posc[ordr][:-1]])
    gname_u = gcode[ordr][fpos]
    gcls_u = gclsc[ordr][fpos]
    del gcode, sidc, gclsc, posc, ordr, fpos
    Ks = [10 ** 4, 10 ** 5, 10 ** 6, 3 * 10 ** 6, 10 ** 7]
    q3 = []
    for H in S:
        trainsets = {'all_others': [s for s in S if s != H]}
        a_ = [s for s in S if s != H and cat(H, s).startswith('A_')]
        if a_:
            trainsets['same_chem_tissue'] = a_
        for tname, tr in trainsets.items():
            if tname == 'all_others':
                agg = totalf.copy()
                agg[POS[H]] -= FR[H]
            else:
                agg = np.zeros(NU)
                for s in tr:
                    agg += np.bincount(POS[s], weights=FR[s], minlength=NU)
            agg[agg < 1e-15] = -1.0
            kmax = int(min(max(Ks), (agg > 0).sum()))
            top = np.argpartition(-agg, kmax - 1)[:kmax]
            top = top[np.argsort(-agg[top], kind='stable')]
            rank = np.full(NU, np.iinfo(np.uint32).max, np.uint32)
            rank[top] = np.arange(kmax, dtype=np.uint32)
            rk = rank[POS[H]]
            rec = dict(heldout=H, training=tname, train_samples=tr, K={})
            for K in Ks:
                hit = rk < min(K, kmax)
                rec['K'][str(K)] = dict(entries=int(min(K, kmax)), **hitstats(D[H], hit))
            t1 = top[:10000]
            share = np.bincount(np.maximum(gname_u[t1], 0), weights=np.where(gname_u[t1] >= 0, agg[t1], 0), minlength=len(gl))
            tot = share.sum()
            tg = np.argsort(-share)[:15]
            rec['top1e4_genes'] = [dict(gene=gl[g], mass_share=float(share[g] / tot)) for g in tg if share[g] > 0]
            rec['top1e4_class_mass'] = {n: float(agg[t1][gcls_u[t1] == c].sum() / agg[t1].sum()) for c, n in ((1, 'mito'), (2, 'ribo'), (3, 'MALAT1'), (0, 'other'))}
            cnt = D[H]['cnt'].astype(np.int64)
            rec['heldout_true_class_read_share'] = {n: float(cnt[D[H]['gcls'] == c].sum() / cnt.sum()) for c, n in ((1, 'mito'), (2, 'ribo'), (3, 'MALAT1'), (0, 'other_or_unmapped'))}
            q3.append(rec)
            del agg, rank, top
        log('q3 topk', H)
    R['q3_topk'] = q3
    del totalf, pres, ukall, sigall, POS, PRES

    # ---- Q3b k-mer classifier for mito / ribo / MALAT1 origin
    k = A.kmer
    SUB = {}
    for s in S:
        p = os.path.join(A.data, s, 'seqsub.npz')
        if os.path.exists(p):
            z = np.load(p)
            SUB[s] = dict(seq=z['seq'], gcls=z['gcls'])
    cls_names = {1: 'mito', 2: 'ribo', 3: 'MALAT1'}
    q3b = []
    rng = np.random.default_rng(11)
    for H in S:
        if H not in SUB:
            continue
        train = [s for s in S if s != H and s in SUB]
        pos_k = {c: [] for c in cls_names}
        neg_k = []
        for s in train:
            g = SUB[s]['gcls']
            for c in list(cls_names) + [0]:
                idx = np.flatnonzero(g == c)
                if len(idx) == 0:
                    continue
                idx = rng.choice(idx, min(len(idx), A.train_cap), replace=False)
                for i in range(0, len(idx), 100000):
                    km = kmer_matrix(to_codes(SUB[s]['seq'][idx[i:i + 100000]]), k).ravel()
                    km = km[km != INV]
                    (neg_k if c == 0 else pos_k[c]).append(km)
        neg = np.unique(np.concatenate(neg_k)) if neg_k else np.zeros(0, U)
        pos = {}
        for c in cls_names:
            if pos_k[c]:
                u, n_ = np.unique(np.concatenate(pos_k[c]), return_counts=True)
                pos[c] = u[n_ >= 3]
            else:
                pos[c] = np.zeros(0, U)
        model = {}
        for c in cls_names:
            m_ = pos[c]
            m_ = m_[~inset(neg, m_)]
            for c2 in cls_names:
                if c2 != c:
                    m_ = m_[~inset(pos[c2], m_)]
            model[c] = m_
        del neg, pos, pos_k, neg_k
        g = SUB[H]['gcls']
        idx = np.arange(len(g))
        if len(idx) > A.test_cap:
            idx = np.sort(rng.choice(idx, A.test_cap, replace=False))
        hits = np.zeros((len(idx), 3), np.int32)
        for i in range(0, len(idx), 200000):
            km = kmer_matrix(to_codes(SUB[H]['seq'][idx[i:i + 200000]]), k)
            sh = km.shape
            flat = km.ravel()
            for j, c in enumerate(cls_names):
                hits[i:i + 200000, j] = inset(model[c], flat).reshape(sh).sum(1)
        truth = g[idx]
        rec = dict(heldout=H, train_samples=train, test_reads=int(len(idx)), model_kmers={cls_names[c]: int(len(model[c])) for c in cls_names},
                   truth_share={cls_names[c]: float((truth == c).mean()) for c in cls_names}, by_threshold={})
        for t in (1, 3, 6):
            best = hits.argmax(1)
            pred = np.where(hits.max(1) >= t, best + 1, 0)
            rr = {}
            for c in cls_names:
                tp = int(((pred == c) & (truth == c)).sum())
                pc = int((pred == c).sum())
                tc = int((truth == c).sum())
                rr[cls_names[c]] = dict(precision=tp / pc if pc else None, recall=tp / tc if tc else None, routed_share=pc / len(idx), true_share=tc / len(idx))
            rec['by_threshold'][str(t)] = rr
        q3b.append(rec)
        log('q3 kmer', H)
    R['q3_kmer'] = q3b
    json.dump(R, open(A.out, 'w'), indent=1)
    log('done')


if __name__ == '__main__':
    main()
