#!/usr/bin/env python
"""Train Track C model files (integer tables for the C++ decoder).

    python -P tools/track_c_train.py ORG MEM HOLDOUT        # ORG human|mouse, MEM exon|intron, HOLDOUT study id or 'none'
Writes results/track_c_model_<ORG>_<MEM>_<HOLDOUT>.bin (and results/track_c_samples.tsv once).
Pool = every cached file of that organism/member whose study != HOLDOUT (so the held-out study is never seen).
"""
import sys, os, struct, math, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *
from track_c_exp3 import pava_dec

NB = 32; NB1 = 33
BTH = [1, 2, 2, 3, 4, 5, 7, 9, 12, 16, 21, 29, 39, 52, 70, 95, 128, 174, 235, 318, 431, 584, 790, 1070, 1449, 1962, 2656, 3597, 4871, 6597, 8934, 12099]
NC2 = 121; NCX = 105; NST = 3; NLEV = 7; NBG = 9
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

def bucket_of(nz):
    th = np.array(BTH)
    b = (nz[:, None] >= th[None, :]).sum(1) if len(nz) < 200000 else np.searchsorted(np.sort(th), nz, side='right')
    return b.astype(np.int32)      # 0 for nnz==0

def bucket_fast(nz):
    return np.searchsorted(np.array(BTH), nz, side='right').astype(np.int32)

def fnv_names(names):
    h = 1469598103934665603
    for s in names:
        for byte in s.encode():
            h = ((h ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def cabs(k):
    c = np.zeros(len(k), dtype=np.int32)
    sm = (k >= 1) & (k <= 8); c[sm] = k[sm]
    bg = k > 8; c[bg] = 8 + np.floor(4 * np.log2(k[bg] / 8.0) + 0.5).astype(np.int32)
    return np.minimum(c, 120)

def rbin_c(k, n):
    c = np.zeros(len(k), dtype=np.int32)
    nz = k > 0
    c[nz] = 1 + np.clip(np.floor(-np.log2(k[nz] / n) / 0.25 + 0.5), 0, 119).astype(np.int32)
    return c

def code16(p):
    p = np.maximum(p, 2.0 ** -31)
    return np.clip(np.round(-np.log2(p) * 2048), 0, 65535).astype(np.uint16)

def pava_rows(A, M, lo, hi):
    """monotone (non-increasing in class index) smoothing of A/M within classes lo..hi-1 for each bucket row"""
    P = np.zeros(A.shape)
    for b in range(A.shape[0]):
        y = A[b, lo:hi] / np.maximum(M[b, lo:hi], 1e-9); w = M[b, lo:hi] + 1e-6
        P[b, lo:hi] = pava_dec(y, w)
    return P

def xfer_tables(pool, mem):
    """feature-independent tables (rate-class presence table P2, value priors V2) from a pool of files"""
    A2 = np.zeros((NB1, NC2)); M2 = np.zeros((NB1, NC2)); CV2 = np.zeros((31, NBG, NLEV + 1))
    for k in pool:
        d = get(f'{k}__{mem}'); b = bucket_fast(d['nz']); kf = np.bincount(d['indices'], minlength=d['m'])
        c = rbin_c(kf, d['n']); mc = np.bincount(c, minlength=NC2).astype(float)
        A2 += np.bincount(b[d['col']].astype(np.int64) * NC2 + c[d['indices']], minlength=NB1 * NC2).reshape(NB1, NC2)
        M2 += np.bincount(b, minlength=NB1).astype(float)[:, None] * mc[None, :]
        v = d['data'].astype(np.int64); bg = b[d['col']] >> 2; ctx = (c[d['indices']] >> 2).astype(np.int64) * NBG + bg
        for l in range(NLEV + 1):
            CV2.reshape(-1, NLEV + 1)[:, l] += np.bincount(ctx[v >= l + 1], minlength=31 * NBG)
    M2[:, 0] = 1
    P2 = np.zeros((NB1, NC2)); P2[:, 1:] = pava_rows(A2, M2, 1, NC2)[:, 1:]; P2[:, 0] = 1e-9
    P2 = np.clip(P2, 1e-9, 0.9995)
    C = CV2.reshape(-1, NLEV + 1); glob = C.sum(0); gr = (glob[1:] + 1) / (glob[:-1] + 2)
    V2 = np.clip(np.round((C[:, 1:] + 8 * gr[None, :]) / (C[:, :-1] + 8) * 65536), 64, 65536 - 64).astype(np.uint16)
    return P2, V2

def main_xfer(org, mem):
    """cross-species transfer model: only the feature-independent tables, learned on the OTHER organism (ver=2: no profile, adaptive CPT)"""
    src = 'human' if org == 'mouse' else 'mouse'
    pool = [k for k in STUDY if ORG[k] == src]
    P2, V2 = xfer_tables(pool, mem)
    g, ng = gene_index(org, mem)
    rows = open(os.path.join(SCR, f"{'pbmc_v2_typical' if org=='human' else 'brain_sn_v3_deep'}__{mem}.rows.txt")).read().split(chr(10))
    m = len(rows)
    hf = os.path.join(RES, f'.names_hash_{org}_{mem}.json')
    if os.path.exists(hf): nh = json.load(open(hf))['hash']
    else: nh = fnv_names(rows); json.dump({'hash': nh}, open(hf, 'w'))
    gfirst = np.zeros(m, dtype=np.uint8)
    out = os.path.join(RES, f'track_c_model_{org}_{mem}_x{src}.bin')
    nc1 = 1
    with open(out, 'wb') as f:
        f.write(struct.pack('<II', 0x314D4354, 2)); f.write(struct.pack('<IQ', m, nh))
        f.write(struct.pack('<IIIIIIII', NB1, nc1, NC2, NCX, NST, 1, 31, NBG))
        f.write(np.zeros(m, np.uint8).tobytes()); f.write(np.zeros(m, np.uint8).tobytes()); f.write(gfirst.tobytes())
        f.write(code16(np.full((NB1, nc1), 1e-3)).tobytes()); f.write(code16(P2).tobytes()); f.write(np.zeros((NCX, NST, 127), np.uint16).tobytes())
        f.write(np.full((1 * NBG * NLEV,), 16000, np.uint16).tobytes()); f.write(V2.tobytes())
    print(f'wrote {out} ({os.path.getsize(out)} bytes) from {len(pool)} {src} files', flush=True)

def main(org, mem, holdout):
    if holdout.startswith('x'): return main_xfer(org, mem)
    g, ng = gene_index(org, mem)
    if holdout == 'pbmconly': pool = [k for k in TRAIN_STUDY if ORG[k] == org]
    else: pool = [k for k in STUDY if ORG[k] == org and (holdout == 'none' or STUDY[k] != holdout)]
    names = {k: f'{k}__{mem}' for k in pool}
    first = get(names[pool[0]]); m = first['m']
    rows = open(os.path.join(SCR, f"{'pbmc_v2_typical' if org=='human' else 'brain_sn_v3_deep'}__{mem}.rows.txt")).read().split('\n')
    assert len(rows) == m
    K = {}; NNZ = {}; NN = {}; BK = {}
    for k in pool:
        d = get(names[k]); K[k] = np.bincount(d['indices'], minlength=m); NNZ[k] = len(d['indices']); NN[k] = d['n']; BK[k] = bucket_fast(d['nz'])
    studies = sorted(set(STUDY[k] for k in pool))
    def prof(keys): return sum(K[k] / NNZ[k] for k in keys) / len(keys)
    gcount = np.bincount(g, minlength=ng)
    gfirst = np.zeros(m, dtype=np.uint8); gfirst[0] = 1; gfirst[1:] = (g[1:] != g[:-1])
    def plevel_of(s):
        gs = np.bincount(g, weights=s, minlength=ng) / np.maximum(gcount, 1)
        pl = np.zeros(m, dtype=np.int32)
        seen = s > 0
        lev = np.clip(np.round(-2 * np.log2(np.where(seen, s, 1))), 0, 214).astype(np.int32)
        pl[seen] = 41 + lev[seen]
        gm = gs[g]; un = (~seen) & (gm > 0)
        gl = np.clip(np.round(-np.log2(np.where(un, gm, 1))), 0, 39).astype(np.int32)
        pl[un] = 1 + gl[un]
        return pl
    s_all = prof(pool); plevel = plevel_of(s_all)
    # ---------------- rung-1 profile classes + table
    cls1, nc1 = make_classes(s_all, g, ng)
    A1 = np.zeros((NB1, nc1)); M1 = np.zeros((NB1, nc1))
    for st in studies:
        inn = [k for k in pool if STUDY[k] != st]; out = [k for k in pool if STUDY[k] == st]
        if not inn: continue
        c_in, _ = make_classes(prof(inn), g, ng); mc = np.bincount(c_in, minlength=nc1).astype(float)
        for k in out:
            d = get(names[k]); b = BK[k]
            key = b[d['col']].astype(np.int64) * nc1 + c_in[d['indices']]
            A1 += np.bincount(key, minlength=NB1 * nc1).reshape(NB1, nc1)
            M1 += np.bincount(b, minlength=NB1).astype(float)[:, None] * mc[None, :]
    if M1.sum() == 0:     # single-study pool: fall back to in-sample calibration
        mc = np.bincount(cls1, minlength=nc1).astype(float)
        for k in pool:
            d = get(names[k]); b = BK[k]
            A1 += np.bincount(b[d['col']].astype(np.int64) * nc1 + cls1[d['indices']], minlength=NB1 * nc1).reshape(NB1, nc1)
            M1 += np.bincount(b, minlength=NB1).astype(float)[:, None] * mc[None, :]
    nlev_seen = 96
    P1 = np.zeros((NB1, nc1))
    P1[:, :nlev_seen] = pava_rows(A1, M1, 0, nlev_seen)[:, :nlev_seen]
    P1[:, nlev_seen:] = pava_rows(A1, M1, nlev_seen, nc1)[:, nlev_seen:]
    P1 = np.clip(P1, 1e-9, 0.9995)
    # ---------------- rung-1b file-class table + value priors
    A2 = np.zeros((NB1, NC2)); M2 = np.zeros((NB1, NC2))
    CV2 = np.zeros((31, NBG, NLEV + 1))        # counts v>=l+1  (index l, 0..7)
    CV1 = np.zeros(((nc1 + 3) // 4, NBG, NLEV + 1))
    for k in pool:
        d = get(names[k]); b = BK[k]
        c = rbin_c(K[k], NN[k]); mc = np.bincount(c, minlength=NC2).astype(float)
        key = b[d['col']].astype(np.int64) * NC2 + c[d['indices']]
        A2 += np.bincount(key, minlength=NB1 * NC2).reshape(NB1, NC2)
        M2 += np.bincount(b, minlength=NB1).astype(float)[:, None] * mc[None, :]
        v = d['data'].astype(np.int64); bg = b[d['col']] >> 2
        for cc, CV, ncg in ((c[d['indices']] >> 2, CV2, 31), (cls1[d['indices']] >> 2, CV1, CV1.shape[0])):
            ctx = cc.astype(np.int64) * NBG + bg
            for l in range(NLEV + 1):
                sel = v >= l + 1
                CV.reshape(-1, NLEV + 1)[:, l] += np.bincount(ctx[sel], minlength=ncg * NBG)
    M2[:, 0] = 1
    P2 = np.zeros((NB1, NC2)); P2[:, 1:] = pava_rows(A2, M2, 1, NC2)[:, 1:]
    P2[:, 0] = 1e-9
    P2 = np.clip(P2, 1e-9, 0.9995)
    def vtab(CV):
        C = CV.reshape(-1, NLEV + 1)
        glob = C.sum(0); gr = (glob[1:] + 1) / (glob[:-1] + 2)       # P(v>l+1 | v>=l+1)
        num = C[:, 1:]; den = C[:, :-1]
        p = (num + 8 * gr[None, :]) / (den + 8)
        return np.clip(np.round(p * 65536), 64, 65536 - 64).astype(np.uint16)
    V2 = vtab(CV2); V1 = vtab(CV1)
    # ---------------- CPT for class side info
    Cnt = np.zeros((NCX, NST, 128))
    for st in studies:
        inn = [k for k in pool if STUDY[k] != st]; out = [k for k in pool if STUDY[k] == st]
        if not inn: continue
        pl_in = plevel_of(prof(inn))
        for k in out:
            sh = int(round(2 * math.log2(max(1.0, NNZ[k]))))
            c = cabs(K[k])
            cx = cx_of(pl_in, sh)
            used = (c > 0).astype(np.int64); cs = np.cumsum(used)
            gstart = np.flatnonzero(gfirst)                      # start index of each gene
            gid = np.cumsum(gfirst) - 1
            base = np.concatenate([[0], cs])[gstart][gid]         # used count before gene start
            prev_used = (np.concatenate([[0], cs])[:-1] - base) > 0   # used in earlier exons of this gene
            stt = np.where(gfirst == 1, 0, np.where(prev_used, 2, 1))
            np.add.at(Cnt, (cx, stt, c), 1)
    glob = Cnt.sum((0, 1)) + 0.5; glob /= glob.sum()
    Ccx = Cnt.sum(1)
    Pcx = (Ccx + 8 * glob[None, :]) / (Ccx.sum(1, keepdims=True) + 8)
    Pcs = (Cnt + 8 * Pcx[:, None, :]) / (Cnt.sum(2, keepdims=True) + 8)
    cpt = np.zeros((NCX, NST, 127), dtype=np.uint16)
    for cx in range(NCX):
        for s_ in range(NST):
            pr = Pcs[cx, s_]
            # node probabilities over a 128-leaf tree; node index 1..127, children 2i, 2i+1
            sums = np.zeros(256); sums[128:] = pr
            for i in range(127, 0, -1): sums[i] = sums[2 * i] + sums[2 * i + 1]
            for i in range(1, 128):
                p1 = sums[2 * i + 1] / max(sums[i], 1e-30)
                cpt[cx, s_, i - 1] = int(min(max(round(p1 * 65536), 4), 65536 - 4))
    # ---------------- write
    os.makedirs(RES, exist_ok=True)
    hf = os.path.join(RES, f'.names_hash_{org}_{mem}.json')
    if os.path.exists(hf): nh = json.load(open(hf))['hash']
    else:
        nh = fnv_names(rows); json.dump({'hash': nh}, open(hf, 'w'))
    out = os.path.join(RES, f'track_c_model_{org}_{mem}_{holdout}.bin')
    with open(out, 'wb') as f:
        f.write(struct.pack('<II', 0x314D4354, 1)); f.write(struct.pack('<IQ', m, nh))
        f.write(struct.pack('<IIIIIIII', NB1, nc1, NC2, NCX, NST, CV1.shape[0], 31, NBG))
        f.write(plevel.astype(np.uint8).tobytes()); f.write(cls1.astype(np.uint8).tobytes()); f.write(gfirst.tobytes())
        f.write(code16(P1).tobytes()); f.write(code16(P2).tobytes()); f.write(cpt.tobytes())
        f.write(V1.tobytes()); f.write(V2.tobytes())
    print(f'wrote {out} ({os.path.getsize(out)} bytes), pool={pool}, studies={studies}', flush=True)

def cx_of(pl, sh):
    """mirror of the decoder's context function"""
    cx = np.zeros(len(pl), dtype=np.int64)
    gl = (pl >= 1) & (pl <= 40)
    cx[gl] = 1 + np.clip(pl[gl] - 1 - (sh >> 1) + 18, 0, 39)
    sn = pl >= 41
    cx[sn] = 41 + np.clip(pl[sn] - 41 - sh + 36, 0, 63)
    return cx

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
