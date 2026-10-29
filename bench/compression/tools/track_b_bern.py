# Held-out comparison: per-(rank,depth-bucket) Bernoulli model for head rows vs contexted gap model (pos x k), bits per head nnz.
import sys, os
sys.path.insert(0, '../../python'); sys.path.insert(0, 'tools')
import numpy as np
from singlet._pz import read_1pz

def lb2(x):
    x = np.asarray(x, dtype=np.int64)
    l = np.floor(np.log2(np.maximum(x, 1))).astype(np.int64)
    out = np.where(x < 4, x, 2 * l + ((x >> np.maximum(l - 1, 0)) & 1))
    return out

def gsym(g):
    g = np.asarray(g, dtype=np.int64)
    b = np.floor(np.log2(np.maximum(g, 1))).astype(np.int64)
    s = np.where(g < 8, g, 8 + (b - 3) * 4 + ((g >> np.maximum(b - 2, 0)) & 3))
    ex = np.where(g < 8, 0, b - 2)
    return s, ex

def run(path, R=4096):
    r = read_1pz(path)
    ip = np.asarray(r['indptr'], dtype=np.int64); ix = np.asarray(r['indices']).astype(np.int64)
    n, m = r['n'], r['m']
    cc = np.diff(ip)
    col = np.repeat(np.arange(n), cc)
    rn = np.bincount(ix, minlength=m)
    order = np.lexsort((np.arange(m), -rn))
    rank = np.empty(m, dtype=np.int64); rank[order] = np.arange(m)
    rk = rank[ix]
    o = np.lexsort((rk, col)); rk = rk[o]; col = col[o]
    kb = np.minimum(np.floor(np.log2(cc + 1)).astype(np.int64), 11)
    # gap symbols
    first = np.ones(len(rk), bool); first[1:] = col[1:] != col[:-1]
    prev = np.where(first, -1, np.roll(rk, 1))
    g = rk - prev - 1
    s, ex = gsym(g)
    pb = np.minimum(lb2(prev + 1), 39)
    ctx = pb * 12 + kb[col]
    train = (col % 2 == 0)
    # gap model: held-out cost
    nsym = 80
    H = np.zeros((40 * 12, nsym)) + 0.0
    np.add.at(H, (ctx[train], s[train]), 1)
    P = (H + 0.05) / (H + 0.05).sum(1, keepdims=True)
    test = ~train
    bits = -np.log2(P[ctx[test], s[test]]) + ex[test]
    head = rk[test] < R
    print(f"{os.path.basename(os.path.dirname(path))}: nnz test {test.sum()}, head frac {head.mean():.2f}; gap-ctx heldout bits/nnz all {bits.mean():.3f}, head elems {bits[head].mean():.3f}, tail elems {bits[~head].mean():.3f}")
    # Bernoulli on head rows
    hm = np.zeros((n, R), dtype=np.uint8)
    sel = rk < R
    hm[col[sel], rk[sel]] = 1
    tr_cols = np.nonzero(np.arange(n) % 2 == 0)[0]; te_cols = np.nonzero(np.arange(n) % 2 == 1)[0]
    total = 0.0
    for b in range(12):
        trb = tr_cols[kb[tr_cols] == b]; teb = te_cols[kb[te_cols] == b]
        if len(teb) == 0: continue
        c = hm[trb].sum(0).astype(np.float64) if len(trb) else np.zeros(R)
        p = (c + 0.05) / (len(trb) + 0.1)
        # smooth across ranks? no
        x = hm[teb].astype(np.float64)
        total += -(x * np.log2(p) + (1 - x) * np.log2(1 - p)).sum()
    nh = hm[te_cols].sum()
    print(f"   Bernoulli(rank,k) heldout head cost {total/nh:.3f} bits per head nnz (includes absences) vs gap {bits[head].mean():.3f}")

if __name__ == '__main__':
    for p in sys.argv[1:]:
        run(p)
