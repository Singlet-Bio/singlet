"""Idea 1 (instance 5300): cross-entropy of an Idea 1 model file on a corpus file, independent of the C++ coder.
   python -P tools/idea1_ce5300.py MODEL.bin SAMPLE MEMBER
presence = sum over columns of  sum_f step_fb  +  sum over nnz of [-log2(1-2^-step) - step]  (step = absent cost, bits)
counts   = -log2 of the quantised count frequency (+ escape tail), side = bucket histogram entropy (12-bit freqs)."""
import sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from idea1_data import load, colsums

def read_model(fn):
    b = open(fn, 'rb').read()
    assert b[:8] == b'IDEA1M02'
    m, nb, ncls, cnb, typ = struct.unpack_from('<5I', b, 8)
    o = 28
    edges = np.frombuffer(b, '<u4', 18, o).astype(np.int64); o += 72 + 8
    if typ == 0:
        st = np.frombuffer(b, '<u4', 256, o).astype(np.float64) / 2**24; o += 1024
        codes = np.frombuffer(b, np.uint8, nb * m, o).reshape(nb, m); o += nb * m
        step = st[codes]
    else:
        A = np.frombuffer(b, '<u4', 256, o).astype(np.float64) / 2**28; o += 1024
        Cb = np.frombuffer(b, '<u4', nb, o).astype(np.float64) / 2**31; o += 4 * nb
        codes = np.frombuffer(b, np.uint8, m, o); o += m
        step = Cb[:, None] * A[codes][None, :]
    cls = np.frombuffer(b, np.uint8, m, o); o += m
    cf = np.frombuffer(b, '<u2', ncls * cnb * 16, o).reshape(ncls, cnb, 16); o += ncls * cnb * 32
    assert o == len(b), (o, len(b))
    return dict(m=m, nb=nb, ncls=ncls, cnb=cnb, type=typ, edges=edges[:nb + 1], step=step, cls=cls, cf=cf, bytes=len(b))

def ce(fn, sid, member, nnz_extra=None):
    M = read_model(fn); d = load(sid, member); nnz = len(d['indices'])
    assert M['m'] == d['m']
    nb = M['nb']
    D = colsums(d); k = np.diff(d['indptr'])
    edges = M['edges'].copy(); edges[-1] = 2**40
    b = (np.searchsorted(edges, D, side='right') - 1) if nb > 1 else np.zeros(len(D), np.int64)
    Nb = np.bincount(b, minlength=nb)
    stp = M['step']
    pres_abs = (Nb * stp.sum(1)).sum()
    pres = pres_abs
    bn = np.repeat(b, k)
    s = stp[bn, d['indices']]
    pres += (-np.log2(-np.expm1(-s * np.log(2))) - s).sum()
    P = M['cf'].astype(np.float64) / 2**14
    cb = bn if M['cnb'] > 1 else np.zeros_like(bn)
    c = np.minimum(d['data'], 16).astype(np.int64) - 1
    cnt = -np.log2(P[M['cls'][d['indices']], cb, c]).sum()
    big = d['data'][d['data'] >= 16].astype(np.int64) - 15
    if len(big): cnt += (np.floor(np.log2(big)) * 2 + 5).sum()   # 5 bits + nbv raw bits (approx; nbv>16 split is the same count)
    side = 0.0
    if nb > 1:
        f = np.maximum(1, (Nb * 4096) // len(b)).astype(np.int64)
        f[np.argmax(Nb)] += 4096 - f.sum()
        side = -(Nb * np.log2(f / 4096.0)).sum()
    return dict(nnz=nnz, absent_part=pres_abs / nnz, present_part=(pres - pres_abs) / nnz, presence=pres / nnz, counts=cnt / nnz, side=side / nnz, total=(pres + cnt + side) / nnz, model_bytes=M['bytes'])

if __name__ == '__main__':
    r = ce(sys.argv[1], sys.argv[2], sys.argv[3])
    print(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items()))
