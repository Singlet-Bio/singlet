"""Side-info (feature class) context study: in-sample empirical conditional entropy given various contexts (bits/nnz)."""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *
from track_c_train import cabs, cx_of
from track_c_exp6 import cond_ent
def run(test, mem, hold):
    import struct
    org = ORG[test]; g, ng = gene_index(org, mem)
    f = open(os.path.join(os.path.dirname(__file__), '..', 'results', f'track_c_model_{org}_{mem}_{hold}.bin'), 'rb')
    magic, ver, m, nh, nb1, nc1, nc2, ncx, nst, ncg1, ncg2, nbg = struct.unpack('<IIIQ8I', f.read(52))
    pl = np.frombuffer(f.read(m), np.uint8).astype(int); f.read(m); gf = np.frombuffer(f.read(m), np.uint8)
    d = get(f'{test}__{mem}'); k = np.bincount(d['indices'], minlength=m); c = cabs(k); nnz = len(d['indices'])
    sh = int(round(2 * math.log2(nnz))); cx = cx_of(pl, sh)
    used = (c > 0); cs = np.cumsum(used); gstart = np.flatnonzero(gf); gid = np.cumsum(gf) - 1
    cs0 = np.concatenate([[0], cs]); base = cs0[gstart][gid]; prev_cnt = cs0[:-1] - base
    stt = np.where(gf == 1, 0, np.where(prev_cnt > 0, 2, 1))
    # class of last used exon in gene
    idx = np.where(used, np.arange(m), -1); last = np.maximum.accumulate(idx)
    lastprev = np.concatenate([[-1], last[:-1]])
    ok = (lastprev >= gstart[gid]) & (lastprev >= 0) & (gf == 0)
    lc = np.where(ok, 1 + np.minimum(c[np.maximum(lastprev, 0)] >> 2, 30), 0)
    # gene-level evidence from profile: other exons' plevel (max within gene of seen plevel)
    out = {}
    out['cx'] = cond_ent(cx, c, 128)
    out['cx,st'] = cond_ent(cx * 3 + stt, c, 128)
    out['cx,lastclass'] = cond_ent(cx * 32 + lc, c, 128)
    out['cx//2,st,lastclass'] = cond_ent(((cx // 2) * 3 + stt) * 32 + lc, c, 128)
    # next-exon-in-gene lookahead not allowed; but mixed: cx and prev_cnt capped
    out['cx,min(prev_cnt,3),lc>>2'] = cond_ent(((cx * 4 + np.minimum(prev_cnt, 3)) * 9 + (lc >> 2)), c, 128)
    print(f"{test} {mem} (side info bits/nnz, in-sample tables, ctx counts small): " + ' | '.join(f'{k_}: {v/nnz:.3f}' for k_, v in out.items()), flush=True)
for t in sys.argv[3:]: run(t, sys.argv[1], sys.argv[2])
