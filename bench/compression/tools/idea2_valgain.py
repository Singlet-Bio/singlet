"""How much could per-feature (or per-gene-sibling) information help the count values?  5-fold CV, pbmc-style files.
M1: P(c | rate bucket of log(k_j u_f / U))   (what Track D calls val_ratectx_cv)
M2: P(c | f) smoothed toward M1 (per-feature table, top-T features)
usage: python -u -P tools/idea2_valgain.py sample member [T]"""
import sys, numpy as np
sys.path.insert(0, '../../python'); sys.path.insert(0, 'tools')
from idea2_clustergain import load

def main():
    sample, member = sys.argv[1], sys.argv[2]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    m, n, ip, ix, d, isc = load(sample, member)
    cells = np.nonzero(isc)[0]; nc = len(cells)
    colid = np.repeat(np.arange(n), np.diff(ip))
    msk = isc[colid]
    f = ix[msk]; c = d[msk].astype(np.int64); col = colid[msk]
    kj = np.zeros(n); np.add.at(kj, col, c)
    u = np.bincount(f, weights=c, minlength=m); U = u.sum()
    nnz = len(f)
    cap = 10
    cc = np.minimum(c, cap) - 1
    lam = np.log(kj[col] * u[f] / U + 1e-12)
    rb = np.clip(((lam + 12) * 3).astype(int), 0, 59)
    cell_idx = np.searchsorted(cells, col)
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, nc)[cell_idx]
    z = np.bincount(f, minlength=m); order = np.argsort(-z, kind='stable'); rank = np.full(m, -1); rank[order[:T]] = np.arange(T)
    fr = rank[f]
    loss1 = np.zeros(nnz); loss2 = np.zeros(nnz)
    for k_ in range(5):
        tr = fold != k_; te = ~tr
        tab1 = np.zeros((60, cap)) + 0.05
        np.add.at(tab1, (rb[tr], cc[tr]), 1)
        p1 = tab1 / tab1.sum(1, keepdims=True)
        loss1[te] = -np.log2(p1[rb[te], cc[te]])
        sel = tr & (fr >= 0)
        tabf = np.zeros((T, cap)); np.add.at(tabf, (fr[sel], cc[sel]), 1)
        nf = tabf.sum(1)
        best = None
        for a in (3, 10, 30, 100):
            idx = te & (fr >= 0)
            pf = (tabf[fr[idx]] + a * p1[rb[idx]]) / (nf[fr[idx]][:, None] + a)
            l = -np.log2(pf[np.arange(idx.sum()), cc[idx]])
            if best is None or l.sum() < best[0]:
                best = (l.sum(), a, l, idx)
        l2 = loss1.copy()
        loss2[te] = loss1[te]
        loss2[best[3]] = best[2]
    topm = fr >= 0
    print(f'{sample} {member}: nnz {nnz} (top-{T}: {topm.sum()})')
    print(f'  M1 rate-bucket      : {loss1.mean():.4f} bits/nnz overall, {loss1[topm].mean():.4f} on top-{T}')
    print(f'  M2 +per-feature     : {loss2.mean():.4f} bits/nnz overall, {loss2[topm].mean():.4f} on top-{T}')
    # sibling info: sum of other entries of the same gene? (not computed here)

if __name__ == '__main__':
    main()
