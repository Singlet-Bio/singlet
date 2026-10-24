"""Idea 3, stage B measurements: how much does the exon matrix of the same sample tell about the intron matrix?
Gene level, per cell.  2-fold cross-validated (cells split by parity) so the context-count bias is removed.
Usage: python -P tools/idea3_measure_b.py [sample ...]   -> results/idea3_b_measure.jsonl"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idea3_common import *
from idea3_prep import bucket, entries, gene_groups, B

def class_of(rate):
    c = np.full(len(rate), 63, np.int64)
    ok = rate > 0
    c[ok] = np.clip(np.floor(3 * (-np.log2(rate[ok]))), 0, 63).astype(np.int64)
    return c

def ebucket(e):
    return np.where(e == 0, 0, np.where(e == 1, 1, np.where(e <= 3, 2, 3)))

def cell_cls(k):
    """coarse cell depth class for the exon side: 0 if kE==0 else 1+bucket//4 (1..6)"""
    b = bucket(k)
    return np.where(b < 0, 0, 1 + b // 4)

def cost(N, Y, other_N, other_Y):
    """N,Y,other_N,other_Y: arrays over contexts. probabilities from other fold (KT), evaluated on this fold. bits"""
    p = (other_Y + 0.4) / (other_N + 0.8)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float((-(Y * np.log2(p) + (N - Y) * np.log2(1 - p))).sum())

def analyse(sid):
    S = Sample(sid)
    n = S.E.shape[1]
    colE, rowE, xE = entries(S.E); gcE, ggE, gvE, _ = gene_groups(colE, rowE, xE, S.egene)
    colI, rowI, xI = entries(S.I); gcI, ggI, gvI, _ = gene_groups(colI, rowI, xI, S.igene)
    Ge, Gi = len(S.gids_e), len(S.gids_i)
    kE = np.bincount(gcE, minlength=n); kI = np.bincount(gcI, minlength=n)
    ggI_e = S.i2e[ggI]
    keyE = gcE.astype(np.int64) * Ge + ggE
    keyI = gcI.astype(np.int64) * Ge + ggI_e
    pos = np.searchsorted(keyE, keyI)
    pos = np.minimum(pos, len(keyE) - 1)
    hit = keyE[pos] == keyI
    eI = np.where(hit, gvE[pos], 0)                       # exon gene count at each intron nonzero
    # restrict to cells with kI > 0
    bI = bucket(kI); cE = cell_cls(kE)
    r = dict(sample=sid, n=n, nnz_gene_I=len(gvI), nnz_gene_E=len(gvE), nnz_int_I=int(S.I.nnz), nnz_int_E=int(S.E.nnz), U_I=int(xI.sum()), U_E=int(xE.sum()))
    # gene class (in-sample intron presence rate)
    ncell_k = int((kI > 0).sum())
    rate = np.bincount(ggI, minlength=Gi) / max(ncell_k, 1)
    cls = class_of(rate)
    ngcls = np.bincount(cls, minlength=64).astype(np.float64)
    # --- full-resolution context counts, per parity fold
    # context index: ((cls*B + b)*7 + cE)*4 + e
    # exon-side gene class (coarse, 8 levels) of the same gene: presence rate of the exon gene
    rateE = np.bincount(ggE, minlength=Ge) / max(int((kE > 0).sum()), 1)
    clsE8 = (class_of(rateE) >> 3)[S.i2e]          # per intron gene
    NCTX = 64 * B * 7 * 4 * 8
    def ctx(clsv, bv, cv, ev, kv):
        return (((clsv * B + bv) * 7 + cv) * 4 + ev) * 8 + kv
    ok_cell = bI >= 0
    Nf = np.zeros((2, NCTX)); Yf = np.zeros((2, NCTX))
    # cells per (parity, b, cE)
    for par in (0, 1):
        cm = ok_cell & ((np.arange(n) % 2) == par)
        cc = np.zeros((B, 7))
        np.add.at(cc, (bI[cm], cE[cm]), 1)
        # base N for e=0 : cells(b,cE) * genes(cls)
        base = np.zeros(NCTX).reshape(64, B, 7, 4, 8)
        # genes per (cls, clsE8)
        ngk = np.zeros((64, 8))
        np.add.at(ngk, (cls, clsE8), 1)
        base[:, :, :, 0, :] = ngk[:, None, None, :] * cc[None, :, :, None]
        # exon nonzero pairs (c, g) with g in intron genes (g of exon index -> intron gene idx)
        e2i = -np.ones(Ge, np.int64); e2i[S.i2e] = np.arange(Gi)
        gi_of = e2i[ggE]
        sel = (gi_of >= 0) & cm[gcE]
        gcv = gcE[sel]; gv = gvE[sel]; gi_ = gi_of[sel]
        ee = ebucket(gv)
        ix = ctx(cls[gi_], bI[gcv], cE[gcv], ee, clsE8[gi_])
        cntE = np.bincount(ix, minlength=NCTX).astype(np.float64)
        tg = cntE.reshape(64, B, 7, 4, 8)
        flat = base.copy()
        flat[:, :, :, 0, :] -= tg[:, :, :, 1:, :].sum(3)
        flat[:, :, :, 1:, :] = tg[:, :, :, 1:, :]
        Nf[par] = flat.reshape(-1)
        # intron nonzero pairs
        selI = cm[gcI]
        ixI = ctx(cls[ggI[selI]], bI[gcI[selI]], cE[gcI[selI]], ebucket(eI[selI]), clsE8[ggI[selI]])
        Yf[par] = np.bincount(ixI, minlength=NCTX)
    def marg(keep):
        """sum contexts over dims not in keep. keep subset of 'cbEe' (cls,b,cE,e). returns (N,Y) arrays shape (2, K)"""
        shape = (2, 64, B, 7, 4, 8)
        Nn = Nf.reshape(shape); Yy = Yf.reshape(shape)
        axes = tuple(i + 1 for i, ch in enumerate('cbEek') if ch not in keep)
        return Nn.sum(axes).reshape(2, -1), Yy.sum(axes).reshape(2, -1)
    res = {}
    for name, keep in [('A_cls_b', 'cb'), ('B_cls_b_e', 'cbe'), ('C_cls_b_cE', 'cbE'), ('D_cls_b_cE_e', 'cbEe'), ('A0_cls', 'c'), ('A1_b_only', 'b'), ('E_cls_e', 'ce'), ('F_cls_k_b', 'ckb'), ('G_cls_k_b_e', 'ckbe')]:
        N, Y = marg(keep)
        bits = cost(N[0], Y[0], N[1], Y[1]) + cost(N[1], Y[1], N[0], Y[0])
        res[name] = bits
        r['pres_bits_' + name] = bits
    # raw MI between presence of E and I (no conditioning), over the pair universe (cells kI>0, intron genes)
    N, Y = marg('e')
    Nall = N.sum(0); Yall = Y.sum(0)
    tot = Nall.sum(); py = Yall.sum() / tot
    mi = 0.0
    for e in range(4):
        pe = Nall[e] / tot
        for yv, py_ in ((1, Yall[e] / max(Nall[e], 1)), (0, 1 - Yall[e] / max(Nall[e], 1))):
            pj = pe * py_
            pm = py if yv else 1 - py
            if pj > 0 and pm > 0:
                mi += pj * np.log2(py_ / pm)
    r['MI_raw_bits_total'] = float(mi * tot)
    r['P_I_given_e'] = [float(Yall[e] / max(Nall[e], 1)) for e in range(4)]
    r['P_e'] = [float(Nall[e] / tot) for e in range(4)]
    # counts of intron given presence: contexts (cls>>2, b//4 [6]) vs + e-bucket, CV by parity
    par = (gcI % 2)
    sym = np.minimum(gvI, 16) - 1
    ebk = ebucket(eI)
    cb = (cls[ggI] >> 2) * 6 + np.maximum(bI[gcI], 0) // 4
    esc = gvI >= 16
    escbits = float((2 * np.floor(np.log2(np.maximum(gvI[esc] - 15, 1))) + 1).sum())
    def cnt_cost(ctxv, nctx):
        tot = 0.0
        H = np.zeros((2, nctx, 16))
        for p in (0, 1):
            m_ = par == p
            H[p] = np.bincount(ctxv[m_] * 16 + sym[m_], minlength=nctx * 16).reshape(nctx, 16)
        for p in (0, 1):
            m_ = par == p
            tr = H[1 - p] + 0.3
            pr = tr / tr.sum(1, keepdims=True)
            tot += float(-np.log2(pr[ctxv[m_], sym[m_]]).sum())
        return tot
    r['cnt_bits_A'] = cnt_cost(cb, 96)
    r['cnt_bits_B_e'] = cnt_cost(cb * 4 + ebk, 96 * 4)
    r['cnt_bits_C_cE'] = cnt_cost(cb * 7 + cE[gcI], 96 * 7)
    r['cnt_bits_D_cE_e'] = cnt_cost((cb * 7 + cE[gcI]) * 4 + ebk, 96 * 28)
    r['cnt_escape_bits'] = escbits
    # cell-level k coupling: bits to code kI as zigzag delta in file order (static order-0) vs conditioned on bucket of kE
    def zz_bits(k):
        d = np.diff(k.astype(np.int64), prepend=0); z = np.where(d >= 0, 2 * d, -2 * d - 1)
        sb = np.zeros(len(z), np.int64); nz = z > 0; sb[nz] = np.floor(np.log2(z[nz])).astype(np.int64) + 1
        h = np.bincount(sb); h = h[h > 0]
        return float(-(h * np.log2(h / h.sum())).sum() + np.maximum(sb - 1, 0).sum())
    r['k_bits_I_alone'] = zz_bits(kI)
    # k_I predicted from k_E: code the residual vs round(a*kE) where a = median ratio
    ratio = float(np.median(kI[(kE > 20) & (kI > 0)] / kE[(kE > 20) & (kI > 0)])) if ((kE > 20) & (kI > 0)).any() else 0.0
    pred = np.round(ratio * kE).astype(np.int64)
    d = kI.astype(np.int64) - pred
    z = np.where(d >= 0, 2 * d, -2 * d - 1)
    sb = np.zeros(len(z), np.int64); nz = z > 0; sb[nz] = np.floor(np.log2(z[nz])).astype(np.int64) + 1
    h = np.bincount(sb); h = h[h > 0]
    r['k_bits_I_given_E_ratio'] = float(-(h * np.log2(h / h.sum())).sum() + np.maximum(sb - 1, 0).sum())
    r['kI_over_kE_median'] = ratio
    r['corr_log_kE_kI'] = float(np.corrcoef(np.log1p(kE), np.log1p(kI))[0, 1])
    # fraction of intron gene-nonzeros whose gene also has exon counts
    r['frac_intron_nz_with_exon'] = float((eI > 0).mean())
    r['frac_intron_nz_with_exon_umi'] = float(gvI[eI > 0].sum() / gvI.sum())
    return r

if __name__ == '__main__':
    sids = sys.argv[1:] or TEST
    for sid in sids:
        t = time.time()
        rr = analyse(sid)
        print(sid, 'done', round(time.time() - t, 1), flush=True)
        with open(os.path.join(ROOT, 'results', 'idea3_b_measure.jsonl'), 'a') as f:
            f.write(json.dumps(rr, default=float) + '\n')
