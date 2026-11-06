#!/usr/bin/env python3
"""Per-sample extraction and within-sample analyses for the read-collapse study.

stdin: tab text from `samtools view -F 0x900 BAM | awk` with columns
       flag chr pos cigar nh as cb ur seq
Reads the tagged BAM rows (mapped reads) plus an optional file of unmapped
cDNA sequences (one per line). Writes into --out:
  distinct.npz   one row per distinct read sequence (exact-cache table)
  seqsub.npz     read-level subset with sequences (for Hamming and k-mer work)
  within.json    Q1 (within sample) and Q2 (grouping) results
"""
import sys, os, re, json, time, argparse
import numpy as np
import pandas as pd
from pandas.util import hash_array

U = np.uint64
COLS = ['flag', 'chr', 'pos', 'cigar', 'nh', 'as_', 'cb', 'ur', 'seq']
RCT = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
W_MULTI = 30.0          # mapping cost of an NH>1 distinct read relative to a simple one (heavy lane: 2.2% of reads, ~40% of kernel time)
VERIFY = 0.10           # cost of verifying or patching one member, relative to mapping it
T0 = time.time()


def log(*a):
    print('[rc_extract %6.0fs]' % (time.time() - T0), *a, file=sys.stderr, flush=True)


def mix(a, b):
    with np.errstate(over='ignore'):
        return a ^ (b.astype(U) + U(0x9E3779B97F4A7C15) + (a << U(6)) + (a >> U(2)))


def load_gtf(path):
    g = pd.read_csv(path, sep='\t', comment='#', header=None, usecols=[0, 2, 3, 4, 8],
                    names=['chr', 'ft', 's', 'e', 'attr'], dtype={'chr': str, 'ft': str, 'attr': str})
    g = g[g.ft == 'gene'].copy()
    nm = g.attr.str.extract(r'gene_name "([^"]+)"')[0]
    gi = g.attr.str.extract(r'gene_id "([^"]+)"')[0]
    g['name'] = nm.fillna(gi)
    chrs = {c: i for i, c in enumerate(sorted(g.chr.unique()))}
    g['cid'] = g.chr.map(chrs).astype(np.int64)
    g = g.sort_values(['cid', 's']).reset_index(drop=True)
    cls = np.zeros(len(g), np.uint8)
    ribo = g.name.str.match(r'^(?:RPL|RPS)\d+[A-Z]{0,2}$|^RPLP\d$', case=False).values
    mal = (g.name.str.lower() == 'malat1').values
    mt = g.chr.str.contains(r'(?:^|_)(?:chrM|MT)$', regex=True).values | g.name.str.match(r'^mt-', case=False).values
    cls[mal] = 3
    cls[ribo] = 2
    cls[mt] = 1
    return dict(chrs=chrs, cid=g.cid.values, skey=(g.cid.values << 32) | g.s.values.astype(np.int64),
                end=g.e.values.astype(np.int64), names=g.name.values.astype('U'), cls=cls)


def gene_of(G, chrname, pos):
    cid = chrname.map(G['chrs']).fillna(-1).astype(np.int64).values
    p = pos.astype(np.int64)
    c = np.searchsorted(G['skey'], (cid << 32) | p, 'right') - 1
    out = np.full(len(p), -1, np.int32)
    for step in range(3):
        i = c - step
        ii = np.where(i >= 0, i, 0)
        ok = (out < 0) & (i >= 0) & (cid >= 0) & (G['cid'][ii] == cid) & (G['end'][ii] >= p)
        out[ok] = ii[ok]
    return out


def cost_of(bits):
    return 1.0 + (W_MULTI - 1.0) * ((bits & 1) > 0)


# ---------------------------------------------------------------- reading
class Acc:
    names = ['key', 'cu', 'sig', 'rsig', 'bits', 'gene', 'pos', 'chrc', 'as_', 'nh', 'nops']

    def __init__(self):
        self.d = {k: [] for k in self.names}
        self.n = 0
        self.sub = {k: [] for k in ['seq', 'ridx']}

    def add(self, arrs, seqs, sel):
        for k in self.names:
            self.d[k].append(arrs[k])
        idx = np.flatnonzero(sel)
        self.sub['seq'].extend(seqs[i] for i in idx)
        self.sub['ridx'].append(idx + self.n)
        self.n += len(arrs['key'])


def hash_str(a):
    return hash_array(np.asarray(a, dtype=object))


def process_bam_chunk(c, G):
    rev = (c.flag.values & 16) > 0
    key = hash_str(c.seq.values)
    key = np.where(rev, key ^ U(0xA5A5A5A5A5A5A5A5), key)
    ch = hash_str(c.cigar.values)
    hchr = hash_str(c.chr.values)
    pos = c.pos.values.astype(np.int64)
    nh = c.nh.values.astype(np.int64)
    as_ = c.as_.values.astype(np.int64)
    s = mix(mix(mix(ch, hchr), pos.astype(U)), rev.astype(U))
    s = mix(s, nh.astype(U))
    rsig = s
    sig = mix(s, (as_ + 1000).astype(U))
    cg = c.cigar
    bits = ((nh > 1) * 1 + cg.str.contains('S', regex=False).values * 2 + cg.str.contains('N', regex=False).values * 4).astype(np.uint8)
    nops = cg.str.count(r'[MIDNSHP=X]').values.astype(np.uint8)
    cbv = c.cb.values != ''
    cu = np.where(cbv, hash_str((c.cb + c.ur).values), U(0))
    gene = gene_of(G, c.chr, c.pos.values)
    arrs = dict(key=key, cu=cu, sig=sig, rsig=rsig, bits=bits, gene=gene, pos=pos.astype(np.int32),
                chrc=(hchr & U(0x7FFFFFFF)).astype(np.int32), as_=as_.astype(np.int16),
                nh=np.minimum(nh, 255).astype(np.uint8), nops=nops)
    seqs = [(x.translate(RCT)[::-1] if r else x) for x, r in zip(c.seq.values, rev)]
    return arrs, seqs


def process_unmapped_chunk(seqs):
    n = len(seqs)
    key = hash_str(seqs)
    z = np.zeros(n, U)
    arrs = dict(key=key, cu=z, sig=np.full(n, 0x5555, U), rsig=np.full(n, 0x5555, U), bits=np.full(n, 8, np.uint8),
                gene=np.full(n, -1, np.int32), pos=np.zeros(n, np.int32), chrc=np.zeros(n, np.int32),
                as_=np.zeros(n, np.int16), nh=np.zeros(n, np.uint8), nops=np.zeros(n, np.uint8))
    return arrs, list(seqs)


# ---------------------------------------------------------------- analyses
CLASSES = [('simple', lambda b: b == 0), ('multi_NH>1', lambda b: (b & 1) > 0), ('softclipped', lambda b: (b & 2) > 0),
           ('spliced', lambda b: (b & 4) > 0), ('heavy_proxy_any', lambda b: (b & 7) > 0), ('unmapped', lambda b: (b & 8) > 0)]


def q1_within(ukey, cnt, bits, cls_gene):
    N = int(cnt.sum())
    D = len(ukey)
    out = dict(reads=N, distinct=D, distinct_frac=D / N)
    edges = [1, 2, 3, 5, 9, 17, 65, 257, 1025, 10 ** 12]
    lab = ['1', '2', '3-4', '5-8', '9-16', '17-64', '65-256', '257-1024', '>1024']
    ab = []
    for i in range(len(lab)):
        m = (cnt >= edges[i]) & (cnt < edges[i + 1])
        ab.append(dict(bin=lab[i], distinct=int(m.sum()), distinct_share=float(m.sum() / D), reads=int(cnt[m].sum()), read_share=float(cnt[m].sum() / N)))
    out['abundance'] = ab
    cl = []
    for name, f in CLASSES:
        m = f(bits)
        r = int(cnt[m].sum())
        cl.append(dict(cls=name, distinct=int(m.sum()), reads=r, read_share=r / N, distinct_share=float(m.sum() / D),
                       distinct_per_read=float(m.sum() / max(r, 1)),
                       cost_share=float(cost_of(bits)[m].sum() / cost_of(bits).sum())))
    out['classes'] = cl
    srt = np.sort(cnt)[::-1]
    cs = np.cumsum(srt, dtype=np.float64)
    out['topk_within'] = {str(k): float(cs[min(k, D) - 1] / N) for k in (10, 100, 1000, 10 ** 4, 10 ** 5, 10 ** 6)}
    return out


def q2_cbumi(a, ukey, cnt):
    key, cu, sig, rsig, bits, pos, chrc = (a[k] for k in ['key', 'cu', 'sig', 'rsig', 'bits', 'pos', 'chrc'])
    vi = np.flatnonzero(cu != 0)
    _, ui = np.unique(mix(cu[vi], key[vi]), return_index=True)
    rows = vi[ui]
    kc = cnt[np.searchsorted(ukey, key[rows])].astype(np.int64)
    o = np.lexsort((-kc, cu[rows]))
    rows = rows[o]
    c_ = cu[rows]
    st = np.r_[True, c_[1:] != c_[:-1]]
    rep = rows[np.flatnonzero(st)[np.cumsum(st) - 1]]
    mem = rows[~st]
    rp = rep[~st]
    P = len(rows)
    strict = sig[mem] == sig[rp]
    relax = rsig[mem] == rsig[rp]
    near = (chrc[mem] == chrc[rp]) & (np.abs(pos[mem].astype(np.int64) - pos[rp]) <= 50)
    cost_all = cost_of(bits[rows]).sum()
    cm = cost_of(bits[mem])

    def net(ok):
        return float(((1 - VERIFY) * cm[ok].sum() - VERIFY * cm[~ok].sum()) / cost_all)
    res = dict(valid_read_distinct_pairs=int(P), groups=int(st.sum()), groups_with_2plus=int(len(np.unique(cu[mem]))),
               members=int(len(mem)), member_share=float(len(mem) / max(P, 1)),
               agree_strict=float(strict.mean()) if len(mem) else None, agree_relaxed_no_score=float(relax.mean()) if len(mem) else None,
               agree_locus_50bp=float(near.mean()) if len(mem) else None,
               gross_saving_strict=float(cm[strict].sum() / cost_all), gross_saving_relaxed=float(cm[relax].sum() / cost_all),
               net_saving_strict_verify10=net(strict), net_saving_relaxed_verify10=net(relax))
    return res


def to_codes(seqS):
    L = seqS.dtype.itemsize
    a = np.frombuffer(seqS.tobytes(), dtype=np.uint8).reshape(len(seqS), L)
    lut = np.full(256, 4, np.uint8)
    lut[[65, 67, 71, 84]] = [0, 1, 2, 3]
    lut[[97, 99, 103, 116]] = [0, 1, 2, 3]
    return lut[a]


def buckets(X, kk=15, pre=25):
    n, L = X.shape
    valid_p = (X[:, :pre] != 4).all(1) if L >= pre else np.zeros(n, bool)
    pk = np.zeros(n, U)
    for j in range(min(pre, L)):
        pk = (pk << U(2)) | (X[:, j] & 3).astype(U)
    mm = np.full(n, np.iinfo(np.uint64).max, U)
    code = np.zeros(n, U)
    cs = np.cumsum(X == 4, axis=1, dtype=np.int16)
    mask = U((1 << (2 * kk)) - 1)
    ok = np.zeros(n, bool)
    for j in range(L):
        code = ((code << U(2)) | (X[:, j] & 3).astype(U)) & mask
        if j >= kk - 1:
            nb = cs[:, j] - (cs[:, j - kk] if j >= kk else 0)
            with np.errstate(over='ignore'):
                h = code * U(0x9E3779B97F4A7C15)
            good = nb == 0
            mm = np.where(good & (h < mm), h, mm)
            ok |= good
    return dict(prefix25=(pk, valid_p), minimizer15=(mm, ok))


def q2_hamming(sub, p_frac):
    key = sub['key']
    uk, ui, ucnt = np.unique(key, return_index=True, return_counts=True)
    seq = sub['seq'][ui]
    X = to_codes(seq)
    sig, rsig, as_, bits = sub['sig'][ui], sub['rsig'][ui], sub['as_'][ui], sub['bits'][ui]
    n = len(uk)
    cost = cost_of(bits)
    res = dict(subset_read_fraction=p_frac, distinct_in_subset=int(n), schemes={})
    for sname, (bk, valid) in buckets(X).items():
        idx = np.flatnonzero(valid)
        o = np.lexsort((-ucnt[idx], bk[idx]))
        idx = idx[o]
        b = bk[idx]
        st = np.r_[True, b[1:] != b[:-1]]
        rep = idx[np.flatnonzero(st)[np.cumsum(st) - 1]]
        mem, rp = idx[~st], rep[~st]
        hd = np.empty(len(mem), np.uint8)
        for i in range(0, len(mem), 400000):
            sl = slice(i, i + 400000)
            hd[sl] = np.minimum((X[mem[sl]] != X[rp[sl]]).sum(1), 255)
        hist = {str(d): int((hd == d).sum()) for d in range(0, 7)}
        hist['7+'] = int((hd >= 7).sum())
        sres = dict(bucket_members=int(len(mem)), buckets=int(st.sum()), hd_hist=hist, by_d={})
        call = cost.sum()
        for d in (1, 2):
            cov = hd <= d
            m, r = mem[cov], rp[cov]
            nc = int(cov.sum())
            if nc == 0:
                sres['by_d'][str(d)] = dict(covered=0)
                continue
            strict = sig[m] == sig[r]
            relax = rsig[m] == rsig[r]
            dl = as_[m].astype(np.int32) - as_[r]
            cm = cost[m]
            multi = (bits[m] & 1) > 0

            def net(ok):
                return float(((1 - VERIFY) * cm[ok].sum() - VERIFY * cm[~ok].sum()) / call)
            sres['by_d'][str(d)] = dict(
                covered=nc, covered_share_of_distinct=float(nc / n),
                agree_strict=float(strict.mean()), agree_relaxed_no_score=float(relax.mean()),
                agree_relaxed_simple=float(relax[~multi].mean()) if (~multi).any() else None,
                agree_relaxed_multi=float(relax[multi].mean()) if multi.any() else None,
                delta_as_zero_of_relaxed=float((dl[relax] == 0).mean()) if relax.any() else None,
                delta_as_minus2hd_of_relaxed=float((dl[relax] == -2 * hd[cov][relax].astype(np.int32)).mean()) if relax.any() else None,
                gross_saving_strict=float(cm[strict].sum() / call), gross_saving_relaxed=float(cm[relax].sum() / call),
                net_saving_strict_verify10=net(strict), net_saving_relaxed_verify10=net(relax))
        res['schemes'][sname] = sres
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', required=True)
    ap.add_argument('--gtf', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--unmapped', default='')
    ap.add_argument('--input-reads', type=int, default=0)
    ap.add_argument('--seq-cap', type=float, default=6e6)
    ap.add_argument('--rows', default='')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--chunk', type=int, default=3000000)
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    rng = np.random.default_rng(A.seed)
    G = load_gtf(A.gtf)
    log('gtf genes', len(G['names']))
    r0, r1 = (int(x) if x else None for x in A.rows.split(':')) if A.rows else (None, None)
    p = 1.0 if A.input_reads <= 0 else min(1.0, A.seq_cap / A.input_reads)
    acc = Acc()
    row = 0
    dt = dict(flag=np.int64, pos=np.int64, nh=np.int64, as_=np.int64, chr=str, cigar=str, cb=str, ur=str, seq=str)
    for c in pd.read_csv(sys.stdin, sep='\t', header=None, names=COLS, dtype=dt, na_filter=False, quoting=3, chunksize=A.chunk):
        lo = row
        row += len(c)
        if r0 is not None:
            c = c.iloc[max(0, r0 - lo): (r1 - lo) if r1 is not None else None]
            if len(c) == 0:
                continue
        arrs, seqs = process_bam_chunk(c, G)
        acc.add(arrs, seqs, rng.random(len(c)) < p)
        log('bam rows', acc.n)
    n_bam = acc.n
    if A.unmapped and os.path.exists(A.unmapped):
        for c in pd.read_csv(A.unmapped, header=None, names=['seq'], dtype=str, na_filter=False, quoting=3, chunksize=A.chunk):
            arrs, seqs = process_unmapped_chunk(c.seq.values)
            acc.add(arrs, seqs, rng.random(len(c)) < p)
        log('unmapped rows', acc.n - n_bam)
    a = {k: np.concatenate(v) for k, v in acc.d.items()}
    ridx = np.concatenate(acc.sub['ridx']) if acc.sub['ridx'] else np.zeros(0, np.int64)
    seqS = np.array(acc.sub['seq'], dtype='S')
    N = len(a['key'])
    log('reads', N, 'in bam', n_bam, 'subset', len(ridx))
    order = np.argsort(a['key'], kind='stable')
    sk = a['key'][order]
    st = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    cnt = np.diff(np.r_[st, len(sk)]).astype(np.uint32)
    ukey = sk[st]
    fi = order[st]
    del order, sk
    bits_d, gene_d = a['bits'][fi], a['gene'][fi]
    gcls = np.where(gene_d >= 0, G['cls'][np.maximum(gene_d, 0)], 0).astype(np.uint8)
    np.savez(os.path.join(A.out, 'distinct.npz'), ukey=ukey, cnt=cnt, bits=bits_d, gene=gene_d, gcls=gcls, sig=a['sig'][fi],
             nops=a['nops'][fi], nh=a['nh'][fi], names=G['names'], gene_cls=G['cls'])
    W = dict(sample=A.sample, input_reads=A.input_reads, reads_in_tables=int(N), reads_from_bam=int(n_bam), reads_unmapped_fastq=int(N - n_bam))
    W['q1_within'] = q1_within(ukey, cnt.astype(np.int64), bits_d, gcls)
    # exactness of sequence -> record inside the sample
    m = a['bits'] < 8
    pr = mix(a['key'][m], a['sig'][m])
    _, u1 = np.unique(pr, return_index=True)
    _, kc = np.unique(a['key'][m][u1], return_counts=True)
    W['same_sequence_different_record_keys'] = int((kc > 1).sum())
    W['mapped_distinct_keys'] = int(len(kc))
    W['mean_alignment_record_ops'] = float(a['nops'][fi][bits_d < 8].mean()) if (bits_d < 8).any() else None
    W['mean_nh'] = float(np.maximum(a['nh'][fi][bits_d < 8], 1).mean()) if (bits_d < 8).any() else None
    log('q1 done')
    W['q2_cbumi'] = q2_cbumi(a, ukey, cnt)
    log('cbumi done')
    sub = {k: a[k][ridx] for k in ['key', 'sig', 'rsig', 'bits', 'as_', 'gene', 'nh']}
    sub['seq'] = seqS
    sgc = np.where(sub['gene'] >= 0, G['cls'][np.maximum(sub['gene'], 0)], 0).astype(np.uint8)
    np.savez(os.path.join(A.out, 'seqsub.npz'), seq=seqS, key=sub['key'], bits=sub['bits'], gcls=sgc, gene=sub['gene'])
    W['mean_read_len'] = float(np.mean([len(x) for x in seqS[:200000]])) if len(seqS) else None
    del a
    W['q2_hamming'] = q2_hamming(sub, p) if len(ridx) else None
    log('hamming done')
    json.dump(W, open(os.path.join(A.out, 'within.json'), 'w'), indent=1)
    log('done')


if __name__ == '__main__':
    main()
