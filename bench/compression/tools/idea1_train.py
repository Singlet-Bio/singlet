"""Idea 1 trainer: builds the quantised presence + count model for one (organism, member, holdout, variant).
   python -P tools/idea1_train.py --org human --member exon --hold GSE142444 --var s3
Variants
   s1   order-0 per-feature presence (no depth), global count distribution            (type 0, nb=1)
   s2   per-feature x 16 depth buckets (dense 8-bit table), global count distribution (type 0, nb=16)
   s3   as s2 + count model by (feature-rate class x depth bucket)                    (type 0, nb=16)
   s3r  rank-1 factored presence  mu_fb = A[a_f] * C_b  + same count model as s3     (type 1, nb=16)
The presence table of s2/s3/s3r is the same cloglog fit  P(absent) = exp(-exp(a_f + c_b)),
fitted out of sample on the training studies (dense = 8-bit logit of p_fb; r1 = 8-bit a_f + 16 C_b).
Models are written to results/idea1_models/{org}_{member}_{var}_{hold}.bin   (hold = GSEid | ins)
"""
import sys, os, struct, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from idea1_data import load, colsums, ALL, TEST, TRAIN, genes, CACHE, ROOT
import idea1_lib as L
from idea1_fit import fit_cloglog

EDGES = np.array([0, 1, 2, 3, 5, 9, 16, 30, 55, 100, 180, 320, 560, 1000, 1800, 3200, 10**9], dtype=np.int64)
LO, HI = -16.0, 6.5                 # logit range of the dense 8-bit code
DL = (HI - LO) / 255.0
AL, AH = -16.0, 2.2                 # range of a_f (log mu at the reference bucket) for the rank-1 code
DA = (AH - AL) / 255.0
FR = 24
CMAX = 16
CB = 14
LOG2E = 1.0 / np.log(2.0)

def step_lut():
    q = np.arange(256)
    p = 1.0 / (1.0 + np.exp(-(LO + q * DL)))
    st = np.round((2.0 ** FR) * (-np.log1p(-p) * LOG2E))
    return np.maximum(st, 1).astype(np.uint32), p

def a_lut():
    q = np.arange(256)
    return np.round((2.0 ** 28) * np.exp(AL + q * DA) * LOG2E).astype(np.uint32)

def quantise(p):
    p = np.clip(p, 1e-9, 0.9997)
    lg = np.log(p) - np.log1p(-p)
    return np.clip(np.round((lg - LO) / DL), 0, 255).astype(np.uint8)

def dequant(q):
    return step_lut()[1][q]

FNV_OFF, FNV_PR, M64 = 1469598103934665603, 1099511628211, (1 << 64) - 1
def rn_hash(org, member):
    cp = os.path.join(CACHE, f'rnhash_{org}_{member}.txt')
    if os.path.exists(cp):
        return int(open(cp).read())
    sys.path.insert(0, os.path.join(ROOT, '..', '..', 'python'))
    from singlet._pz import read_1pz
    ref = 'pbmc_v2_typical' if org == 'human' else 'brain_sn_v3_deep'
    r = read_1pz(os.path.join(ROOT, 'corpus', ref, f'{member}_counts.1pz'))
    h = FNV_OFF
    for s in r['rownames']:
        for byte in s.encode():
            h = ((h ^ byte) * FNV_PR) & M64
        h = (h * FNV_PR) & M64          # the NUL separator (h ^= 0)
    open(cp, 'w').write(str(h))
    return h

def samples_for(org, hold):
    if hold.startswith('only_'):          # single-sample model (candidate for per-file model selection)
        return [hold[5:]]
    out = []
    for sid, (g, o) in ALL.items():
        if o != org: continue
        if hold != 'ins' and g == hold: continue
        out.append(sid)
    return out

def counts_cached(sid, member):
    cp = os.path.join(CACHE, f'cnt16_{sid}_{member}.npz')
    if os.path.exists(cp):
        z = np.load(cp)
        return z['n'].astype(np.float64), z['N']
    d = load(sid, member)
    t, N = L.counts_table(d, EDGES)
    np.savez_compressed(cp, n=t.astype(np.int32), N=N)
    return t, N

def pooled_counts(samples, member):
    n = None
    for sid in samples:
        t, N = counts_cached(sid, member)
        if n is None: n, NN = t, N.astype(np.float64)
        else: n += t; NN += N
    return n, NN

def count_hist(samples, member, cls, ncls, cnb):
    H = np.zeros((ncls, cnb, CMAX))
    for sid in samples:
        d = load(sid, member)
        D = colsums(d); b = L.bucket_of(D, EDGES); k = np.diff(d['indptr'])
        bn = np.repeat(b, k) if cnb > 1 else np.zeros(len(d['indices']), int)
        c = np.minimum(d['data'], CMAX).astype(np.int64) - 1
        key = (cls[d['indices']] * cnb + bn) * CMAX + c
        H += np.bincount(key, minlength=ncls * cnb * CMAX).reshape(ncls, cnb, CMAX)
    return H

def quant_freq(P):
    """rows of probabilities -> integer freqs >=1 summing to 2^CB"""
    tot = 1 << CB
    f = np.maximum(1, np.floor(P * tot)).astype(np.int64)
    for r in range(f.shape[0]):
        diff = tot - f[r].sum()
        j = int(np.argmax(f[r]))
        f[r, j] += diff
        assert f[r, j] >= 1 and f[r].sum() == tot
    return f.astype(np.uint16)

def gene_prior(tot, g):
    ng = g.max() + 1
    sz = np.bincount(g, minlength=ng).astype(np.float64)
    G = np.bincount(g, weights=tot, minlength=ng)
    return (G / sz)[g]

def get_fit(org, member, hold, n, N, g):
    cp = os.path.join(CACHE, f'fit_{org}_{member}_{hold}.npz')
    if os.path.exists(cp):
        z = np.load(cp)
        return z['a'], z['c']
    if hold.startswith('only_'):
        c1 = os.path.join(CACHE, f'fit1_{hold[5:]}_{member}.npz')      # written by idea1_select.py (same recipe, 8 iterations)
        if os.path.exists(c1):
            z = np.load(c1)
            return z['a'], z['c']
    tot = n.sum(1)
    gm = gene_prior(tot, g)
    eps = np.where(tot > 0, 0.0, 0.2 * gm / (1 + gm / 50.0))     # pseudo-hits for features unseen in training
    a, c = fit_cloglog(n, N, eps=eps)
    np.savez(cp, a=a, c=c)
    return a, c

def presence_p(a, c):
    mu = np.exp(np.clip(a[:, None] + c[None, :], -40, 8))
    return -np.expm1(-mu)

def header(m, nb, ncls, cnb, typ, edges, rnh):
    out = bytearray(b'IDEA1M02')
    out += struct.pack('<5I', m, nb, ncls, cnb, typ)
    e = np.zeros(18, np.uint32); e[:nb + 1] = np.minimum(edges, 2**32 - 1)
    out += e.astype('<u4').tobytes()
    out += struct.pack('<Q', rnh)
    return out

def build(org, member, hold, var):
    t0 = time.time()
    samples = samples_for(org, hold)
    n16, N16 = pooled_counts(samples, member)
    g = genes(org, member)
    m = n16.shape[0]
    tot = n16.sum(1)
    rnh = rn_hash(org, member)
    # ---- count model
    if var in ('s3', 's3r'):
        ncls, cnb = 16, 16
        cls = np.clip(np.floor(np.log2(tot + 1) / 1.4), 0, 15).astype(np.uint8)
    else:
        ncls, cnb = 1, 1
        cls = np.zeros(m, np.uint8)
    H = count_hist(samples, member, cls.astype(np.int64), ncls, cnb)
    marg = H.sum(1, keepdims=True) + 0.5
    marg = marg / marg.sum(2, keepdims=True)
    P = (H + 40.0 * marg) / (H.sum(2, keepdims=True) + 40.0)
    cf = quant_freq(P.reshape(-1, CMAX)).reshape(ncls * cnb, CMAX)
    tail = bytes(cls.tobytes()) + cf.astype('<u2').tobytes()
    # ---- presence
    if var == 's1':
        Nt = N16.sum()
        gm = gene_prior(tot, g)
        eps = np.where(tot > 0, 0.0, 0.3 * gm / (1 + gm / 20.0))
        p = np.clip((tot + eps + 1e-3) / (Nt + 1.0), 1e-8, 0.9997)
        out = header(m, 1, ncls, cnb, 0, np.array([0, 2**32 - 1]), rnh)
        out += step_lut()[0].astype('<u4').tobytes() + quantise(p).tobytes() + tail
        return bytes(out), dict(samples=samples, secs=time.time() - t0)
    a, c = get_fit(org, member, hold, n16, N16, g)
    if var in ('s2', 's3'):
        p = np.clip(presence_p(a, c), 1e-9, 0.9997)
        codes = quantise(p)                                   # [m, 16]
        out = header(m, 16, ncls, cnb, 0, EDGES, rnh)
        out += step_lut()[0].astype('<u4').tobytes() + np.ascontiguousarray(codes.T).tobytes() + tail
        return bytes(out), dict(samples=samples, secs=time.time() - t0)
    if var == 's3r':
        cn = c - c.max()                                       # reference bucket = deepest-rate bucket
        aa = a + c.max()
        aq = np.clip(np.round((aa - AL) / DA), 0, 255).astype(np.uint8)
        Cb = np.maximum(1, np.round((2.0 ** 31) * np.exp(np.maximum(cn, -17.0)))).astype(np.uint32)
        out = header(m, 16, ncls, cnb, 1, EDGES, rnh)
        out += a_lut().astype('<u4').tobytes() + Cb.astype('<u4').tobytes() + aq.tobytes() + tail
        return bytes(out), dict(samples=samples, secs=time.time() - t0)
    raise ValueError(var)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--org', required=True); ap.add_argument('--member', required=True)
    ap.add_argument('--hold', required=True); ap.add_argument('--var', required=True)
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'results', 'idea1_models'))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    blob, info = build(a.org, a.member, a.hold, a.var)
    fn = os.path.join(a.outdir, f'{a.org}_{a.member}_{a.var}_{a.hold}.bin')
    open(fn, 'wb').write(blob)
    print(fn, len(blob), info)
