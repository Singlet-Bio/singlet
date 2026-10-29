"""Build and export Idea 4 "pp" (Poisson-process variant) model files, loaded by kernels/idea4_pp.h.

Model = gene profile pi0 (log-coded u16), interval split q (log-coded u8), optional topic ratios
(uint8, quadratic companding), count tables for called cells / ambient columns, theta prior
histograms, optional rownames blob. Training uses only the samples passed in (out-of-sample is the
caller's responsibility).
"""
import os, sys, struct
import numpy as np, scipy.sparse as sp
import zstandard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import idea4_lib as L
import idea4_nmf as NM

NCB, NSYM, CSB, NLEV = 40, 16, 14, 16
LEVEL_TQ = np.array([0, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256], dtype=np.float64)


class Ref:
    """Reference-build structure (rows grouped per gene)."""
    def __init__(self, rownames):
        self.rownames = rownames
        self.go, self.st, self.ids, self.ln = L.gene_structure(rownames)
        self.m = len(rownames)
        self.G = len(self.ids)


def static_params(ref, cnt_i, alpha=1000.0, pc_q=4.0):
    """pi0 (gene) and q (interval) from summed interval counts; returns quantised codes and dequantised floats."""
    st, go, ln = ref.st, ref.go, ref.ln
    cnt_g = np.add.reduceat(cnt_i, st[:-1])
    lg = np.add.reduceat(ln, st[:-1])
    a_g = alpha * lg / lg.sum()
    pi0 = (cnt_g + a_g)
    pi0 = pi0 / pi0.sum()
    code0 = np.clip(np.round(-np.log(pi0) * 2048.0), 0, 65535).astype(np.uint16)
    pi0d = np.exp(-code0.astype(np.float64) / 2048.0)
    qv = cnt_i + pc_q * ln / lg[go]
    qmax = np.maximum.reduceat(qv, st[:-1])
    qcode = np.clip(np.round(-16.0 * np.log2(qv / qmax[go])), 0, 255).astype(np.uint8)
    w = np.exp2(-qcode.astype(np.float64) / 16.0)
    wt = np.add.reduceat(w, st[:-1])
    qd = w / wt[go]
    return dict(code0=code0, pi0d=pi0d, qcode=qcode, qd=qd)


def cbucket(lam):
    lam = np.maximum(lam, 1e-300)
    e = np.floor(np.log2(lam))
    half = (lam >= np.exp2(e) * 1.5).astype(np.int64)
    b = 2 * (e.astype(np.int64) + 10) + half
    return np.clip(b, 0, NCB - 1)


def make_ctab(counts_bs):
    """counts_bs: (NCB, NSYM) histogram -> cumulative uint16 table (scale 2^CSB)"""
    cum = np.zeros((NCB, NSYM + 1), dtype=np.uint16)
    for b in range(NCB):
        h = counts_bs[b].astype(np.float64) + 0.25
        h = h + 0.05 * np.exp(-0.7 * np.arange(NSYM))
        f = h / h.sum() * ((1 << CSB) - NSYM)
        f = 1 + np.floor(f).astype(np.int64)
        f[np.argmax(f)] += (1 << CSB) - f.sum()
        cum[b, 1:] = np.cumsum(f)
    return cum


def hist_counts(bucket_idx, vals):
    s = np.minimum(vals, NSYM) - 1
    return np.bincount(bucket_idx.astype(np.int64) * NSYM + s.astype(np.int64), minlength=NCB * NSYM).reshape(NCB, NSYM).astype(np.float64)


def dequant_phi(Rcode, tscale, pi0d):
    """Rcode (K,G) uint8 -> normalised float Phi (G,K) exactly like the C++ model."""
    u = tscale[:, None].astype(np.float64) * (Rcode.astype(np.float64) ** 2) / 65536.0
    v = pi0d[None, :] * u
    v /= v.sum(1, keepdims=True)
    return np.ascontiguousarray(v.T.astype(np.float32))


def quantise_phi(Phi, pi0d, rmax_cap=3000.0):
    G, K = Phi.shape
    r = Phi.astype(np.float64) / pi0d[:, None]          # ratios
    Rcode = np.zeros((K, G), dtype=np.uint8)
    tscale = np.zeros(K, dtype=np.uint32)
    for k in range(K):
        rk = r[:, k]
        rmax = min(rk.max(), rmax_cap)
        c = np.round(np.sqrt(np.minimum(rk / rmax, 1.0)) * 255.0)
        c = np.clip(c, 1, 255)
        Rcode[k] = c.astype(np.uint8)
        tscale[k] = np.uint32(max(1, round(rmax * 65536.0 / 65025.0)))
    return Rcode, tscale


def theta_codes(Theta):
    """Mirror of Fitter::quantise without refinement. Theta (n,K) -> level codes (n,K)."""
    mx = Theta.max(1, keepdims=True)
    v = Theta / np.maximum(mx, 1e-30) * 256.0
    c = np.where(v >= 1.41421356, np.clip(np.round(2.0 * np.log2(np.maximum(v, 1e-9) / 2.0)).astype(np.int64) + 1, 1, 15), 0)
    return c.astype(np.int64)


def map_gene_nnz_to_interval(Xi, Xg, go):
    """for each nnz of the interval-level csr Xi, the position of its (row, gene) entry in Xg (both sorted)."""
    n, m = Xi.shape
    G = Xg.shape[1]
    ri = np.repeat(np.arange(n, dtype=np.int64), np.diff(Xi.indptr))
    key_i = ri * G + go[Xi.indices]
    rg = np.repeat(np.arange(n, dtype=np.int64), np.diff(Xg.indptr))
    key_g = rg * G + Xg.indices.astype(np.int64)
    return np.searchsorted(key_g, key_i)


def write_model(path, ref, par, K=0, Rcode=None, tscale=None, thprior=None, ctab=None, rownames_blob=True):
    G, m = ref.G, ref.m
    flags = 1 if rownames_blob else 0
    out = bytearray()
    out += struct.pack('<8I', 0x314D3449, 1, m, G, K, flags, 0, 0)
    out += ref.st.astype('<u4').tobytes()
    out += par['code0'].astype('<u2').tobytes()
    out += par['qcode'].astype('u1').tobytes()
    if K:
        out += Rcode.astype('u1').tobytes()
        out += tscale.astype('<u4').tobytes()
        out += thprior.astype('<u2').tobytes()
    out += ctab[0].astype('<u2').tobytes()
    out += ctab[1].astype('<u2').tobytes()
    if rownames_blob:
        raw = '\n'.join(ref.rownames).encode()
        z = zstandard.ZstdCompressor(level=19).compress(raw)
        out += struct.pack('<I', len(z)) + z
    with open(path, 'wb') as f:
        f.write(out)
    return len(out)


# ---------------------------------------------------------------------------------------------
# Per-sample training pieces (cached): summed called-cell counts, a capped called-cell sample, an empties sample
# ---------------------------------------------------------------------------------------------
CACHE = os.path.join(L.ROOT, 'corpus_idea4', 'cache')


def get_piece(sid, cap=2500, amb_cols=20000):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f'{sid}_c{cap}_a{amb_cols}.npz')
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        m = int(z['m'])
        Xc = sp.csr_matrix((z['c_data'], z['c_indices'], z['c_indptr']), shape=(int(z['c_n']), m))
        Xe = sp.csr_matrix((z['e_data'], z['e_indices'], z['e_indptr']), shape=(int(z['e_n']), m))
        return dict(sid=sid, m=m, cnt_i=z['cnt_i'], Xc=Xc, Xe=Xe, called_total=int(z['called_total']))
    S = L.load_sample(sid)
    sel = np.where(S.called)[0]
    Xall = L.csr_of(S, sel).tocsr()
    cnt_i = np.asarray(Xall.sum(0)).ravel().astype(np.float64)
    idx = np.arange(len(sel))
    if len(idx) > cap:
        idx = np.sort(np.random.default_rng(1).choice(idx, cap, replace=False))
    Xc = Xall[idx].tocsr(); Xc.sort_indices()
    en = np.where(~S.called & (S.depth >= 1))[0]
    if len(en) > amb_cols:
        en = np.sort(np.random.default_rng(2).choice(en, amb_cols, replace=False))
    Xe = L.csr_of(S, en).tocsr(); Xe.sort_indices()
    np.savez(p, m=S.m, cnt_i=cnt_i, c_data=Xc.data, c_indices=Xc.indices, c_indptr=Xc.indptr, c_n=Xc.shape[0],
             e_data=Xe.data, e_indices=Xe.indices, e_indptr=Xe.indptr, e_n=Xe.shape[0], called_total=len(sel))
    return dict(sid=sid, m=S.m, cnt_i=cnt_i, Xc=Xc, Xe=Xe, called_total=len(sel))


_REF = {}


def get_ref(org):
    """Reference structure for the organism, cached on disk (rownames list)."""
    if org in _REF:
        return _REF[org]
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f'ref_{org}.txt')
    if os.path.exists(p):
        rn = open(p).read().split('\n')
    else:
        sid = 'h_d' if org == 'human' else 'm_brain_c'
        S = L.load_sample(sid)
        rn = S.rownames
        open(p, 'w').write('\n'.join(rn))
    _REF[org] = Ref(rn)
    return _REF[org]


def read_model(path):
    """Inverse of write_model (needed to graft human topics onto the mouse reference)."""
    b = open(path, 'rb').read()
    hdr = struct.unpack('<8I', b[:32]); p = 32
    m, G, K, flags = hdr[2], hdr[3], hdr[4], hdr[5]
    def take(dt, n):
        nonlocal p
        a = np.frombuffer(b, dtype=dt, count=n, offset=p); p += a.nbytes; return a
    d = dict(m=m, G=G, K=K, flags=flags)
    d['gstart'] = take('<u4', G + 1)
    d['code0'] = take('<u2', G)
    d['qcode'] = take('u1', m)
    if K:
        d['Rcode'] = take('u1', K * G).reshape(K, G)
        d['tscale'] = take('<u4', K)
        d['thprior'] = take('<u2', K * NLEV).reshape(K, NLEV)
    d['ctab0'] = take('<u2', NCB * (NSYM + 1)).reshape(NCB, NSYM + 1)
    d['ctab1'] = take('<u2', NCB * (NSYM + 1)).reshape(NCB, NSYM + 1)
    return d
