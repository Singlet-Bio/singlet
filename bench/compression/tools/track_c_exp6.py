"""Value-model contexts (in-sample empirical conditional entropy, bits/nnz)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *
from track_c_exp4 import rbin

def cond_ent(ctx, sym, nsym):
    key = ctx.astype(np.int64) * nsym + sym
    C = np.bincount(key).astype(float)
    C = np.pad(C, (0, (-len(C)) % nsym)).reshape(-1, nsym)
    n = C.sum(1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -np.where(C > 0, C * np.log2(C / n), 0)
    return h.sum()

def run(test, mem, nb=32):
    d = get(f'{test}__{mem}'); m, n = d['m'], d['n']; nnz = len(d['indices'])
    idx, col, v = d['indices'], d['col'], d['data'].astype(np.int64)
    b = bucketize(d, ('nz', nb, 0, 14)); k = np.bincount(idx, minlength=m); ce = rbin(k, n)
    S = 9
    vs = np.minimum(v, S) - 1                      # symbols 0..8 (8 = 9+)
    esc = np.where(v > S, 2 * np.log2(v - S + 1) + 1, 0).sum()   # crude Elias-gamma for tail
    out = {}
    out['v0'] = cond_ent(np.zeros(nnz, dtype=np.int64), vs, S)
    out['class'] = cond_ent(ce[idx] // 2, vs, S)
    out['class,b'] = cond_ent((ce[idx] // 3) * 8 + b[col] // 4, vs, S)
    # previous value in column in rank order (rank by k desc) : approximate by feature order sorted by k
    rank = np.empty(m, dtype=np.int64); rank[np.argsort(-k, kind='stable')] = np.arange(m)
    # reorder entries within each column by rank
    o = np.lexsort((rank[idx], col)); vo = v[o]; co = col[o]
    prev = np.concatenate([[0], vo[:-1]]); prev[np.concatenate([[True], co[1:] != co[:-1]])] = 0
    pc = np.minimum(prev, 3)
    ceo = ce[idx][o]; bo = b[col][o]
    out['class,b,prev'] = cond_ent(((ceo // 3) * 8 + bo // 4) * 4 + pc, np.minimum(vo, S) - 1, S)
    # running mean of earlier values in col (quantized)
    cs = np.cumsum(vo) - vo; start = np.flatnonzero(np.concatenate([[True], co[1:] != co[:-1]]))
    grp = np.repeat(np.arange(len(start)), np.diff(np.concatenate([start, [nnz]])))
    cnt = np.arange(nnz) - start[grp]; base = cs[start][grp]
    mean_prev = np.where(cnt > 0, (cs - base) / np.maximum(cnt, 1), 1.0)
    mq = np.clip(np.round((mean_prev - 1) * 6), 0, 7).astype(np.int64)
    out['class,b,prev,runmean'] = cond_ent((((ceo // 3) * 8 + bo // 4) * 4 + pc) * 8 + mq, np.minimum(vo, S) - 1, S)
    print(f"{test:22s} {mem}: ones={np.mean(v==1):.3f} esc={esc/nnz:.3f} | " + ' | '.join(f"{k_}: {x/nnz:.3f}" for k_, x in out.items()), flush=True)

if __name__ == '__main__':
    mem = sys.argv[1]
    for t in (sys.argv[2:] or TESTS): run(t, mem)
