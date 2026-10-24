"""Track D numerical helpers: writer-symbol streams, entropies, ambient multinomial model.
All routines work on the CSC arrays returned by track_d_common.load(). Corpus data are untrusted
input: they are only parsed as numbers here."""
import math
import numpy as np
from scipy.special import gammaln

LN2 = math.log(2.0)

def log2_fact(k):
    return gammaln(np.asarray(k, dtype=np.float64) + 1.0) / LN2

def log2_binom(n, k):
    n = np.asarray(n, dtype=np.float64); k = np.asarray(k, dtype=np.float64)
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)) / LN2

def h0_counts(c):
    """Order-0 empirical entropy in bits (total, not per symbol) from a count vector."""
    c = np.asarray(c, dtype=np.float64)
    c = c[c > 0]
    n = c.sum()
    return float(-(c * np.log2(c / n)).sum()) if n > 0 else 0.0

def h0_symbols(x, with_mdl=False):
    """Total order-0 entropy (bits) of an integer symbol array; optionally + MDL parameter cost
    0.5*log2(n)*(distinct-1)."""
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    if x.min() >= 0 and x.max() < 5_000_000:
        c = np.bincount(x)
    else:
        _, c = np.unique(x, return_counts=True)
    h = h0_counts(c)
    if with_mdl:
        h += 0.5 * math.log2(max(x.size, 2)) * (int((c > 0).sum()) - 1)
    return h

def columns_of(d):
    ip = d['indptr']; n = int(d['n'])
    return np.repeat(np.arange(n, dtype=np.int64), np.diff(ip))

def writer_streams(d, col=None):
    """Reproduce the symbol streams the production writer feeds zstd (VOCSC, frequency-sorted rows).
    Returns dict of arrays aligned to the writer's stream order:
      gap (nnz), gap_val (value group of each gap), gap_col (column of each gap),
      g_col, g_val, g_cnt (one per (column, value) group), plus per-column ng."""
    m = int(d['m']); n = int(d['n'])
    ind = d['indices']; val = d['data']
    if col is None:
        col = columns_of(d)
    rnnz = np.bincount(ind, minlength=m)
    order = np.lexsort((np.arange(m), -rnnz))          # rows by (nnz desc, id asc), matches counting sort
    inv = np.empty(m, dtype=np.int64); inv[order] = np.arange(m)
    prow = inv[ind]
    o = np.lexsort((prow, val, col))
    col_s = col[o]; val_s = val[o]; prow_s = prow[o]
    new_grp = np.ones(len(o), dtype=bool)
    new_grp[1:] = (col_s[1:] != col_s[:-1]) | (val_s[1:] != val_s[:-1])
    prev = np.zeros(len(o), dtype=np.int64)
    prev[1:] = prow_s[:-1] + 1
    prev[new_grp] = 0
    gap = prow_s - prev
    gi = np.flatnonzero(new_grp)
    g_col = col_s[gi]; g_val = val_s[gi]
    g_cnt = np.diff(np.append(gi, len(o)))
    ng = np.bincount(g_col, minlength=n)
    return dict(gap=gap, gap_val=val_s, gap_col=col_s, g_col=g_col, g_val=g_val, g_cnt=g_cnt, ng=ng)

def depth_bucket(x, nb=16):
    """Log-spaced bucket index in [0, nb) for positive integers (1 -> 0, saturating)."""
    b = np.floor(np.log2(np.maximum(x, 1)) * (nb / 14.0)).astype(np.int64)  # 2^14 = 16k top
    return np.clip(b, 0, nb - 1)


# ---------------------------------------------------------------------------------------------
# column subsets
def subcsc(d, colmask, col=None):
    """Return dict(n, indptr, indices, data, k, nnzc, orig_cols) for the selected columns (original order)."""
    if col is None:
        col = columns_of(d)
    colmask = np.asarray(colmask, dtype=bool)
    sel = colmask[col]
    idx = np.flatnonzero(colmask)
    cc = np.diff(d['indptr'])[idx]
    ip = np.zeros(len(idx) + 1, dtype=np.int64); ip[1:] = np.cumsum(cc)
    ind = d['indices'][sel]; dat = d['data'][sel]
    ncol = np.repeat(np.arange(len(idx)), cc)
    k = np.bincount(ncol, weights=dat, minlength=len(idx)).astype(np.int64)
    return dict(n=len(idx), indptr=ip, indices=ind, data=dat, k=k, nnzc=cc, col=ncol, orig=idx, m=int(d['m']))


def kt_like_bits_binary(n1, n, a_prior_p, alpha):
    """Sum over cells of -log2 BetaBinomial(n1 | n, alpha*p, alpha*(1-p)) (closed-form cold-start cost)."""
    a = alpha * a_prior_p; b = alpha * (1 - a_prior_p)
    lg = gammaln
    ll = (lg(n + 1) - lg(n1 + 1) - lg(n - n1 + 1)
          + lg(n1 + a) + lg(n - n1 + b) - lg(n + a + b) - lg(a) - lg(b) + lg(a + b))
    # beta-binomial pmf includes binomial coefficient (counts-only); sequence prob divides by C(n,n1)
    ll_seq = ll - (lg(n + 1) - lg(n1 + 1) - lg(n - n1 + 1))
    return float(-(ll_seq).sum() / LN2)


# ---------------------------------------------------------------------------------------------
# held-out presence / value models
VMAX = 12   # value alphabet 1..VMAX-1, VMAX+ = tail (Elias-gamma on excess)

def _val_sym(v):
    return np.minimum(v, VMAX) - 1   # 0..VMAX-1

def _tail_bits(v):
    ex = np.maximum(v - VMAX + 1, 1)
    return np.where(v >= VMAX, 2 * np.floor(np.log2(ex)) + 1, 0.0)

def value_cost_entries(train, test, ctx_fn):
    """Per-nonzero bits to code the values of `test` with a table learned on `train` (add-0.5 smoothing,
    backed off to the global table)."""
    ctr = ctx_fn(train); cte = ctx_fn(test)
    nctx = int(max(ctr.max(), cte.max())) + 1
    S = VMAX
    tab = np.zeros((nctx, S))
    np.add.at(tab, (ctr, _val_sym(train['data'])), 1.0)
    tab += 0.5
    glob = tab.sum(0); glob = glob / glob.sum()
    tab = tab + 2.0 * glob[None, :]
    P = tab / tab.sum(1, keepdims=True)
    return -np.log2(P[cte, _val_sym(test['data'])]) + _tail_bits(test['data'])

def value_cost(train, test, ctx_fn):
    return float(value_cost_entries(train, test, ctx_fn).sum())

def rate_ctx_factory(train):
    """Context = log2 of expected rate k_j * q_f, with q_f the pooled training profile."""
    m = train['m']
    C = np.bincount(train['indices'], weights=train['data'], minlength=m).astype(np.float64)
    q = (C + 0.05) / (C.sum() + 0.05 * m)
    def fn(s):
        k = s['k'][s['col']].astype(np.float64)
        r = np.log2(k * q[s['indices']])
        return np.clip(np.floor(r).astype(np.int64) + 22, 0, 30)
    return fn

def presence_cost_feature(train, test, alphas=(0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000), all_alphas=False):
    """Per-feature Bernoulli presence (no depth). Returns (best_bits, best_alpha), or {alpha: bits}."""
    m = train['m']; Ntr = train['n']; Nte = test['n']
    n1 = np.bincount(train['indices'], minlength=m).astype(np.float64)
    g = train['indices'].size / (Ntr * m)
    t1 = np.bincount(test['indices'], minlength=m).astype(np.float64)
    best = (1e300, None)
    allr = {}
    for a in alphas:
        p = (n1 + a * g) / (Ntr + a)
        # sum over test cells of -log2 p for present, -log2(1-p) for absent
        bits = -(t1 * np.log2(p)).sum() - ((Nte - t1) * np.log2(1 - p)).sum()
        allr[a] = float(bits)
        if bits < best[0]:
            best = (float(bits), a)
    return allr if all_alphas else best

def presence_cost_feature_depth(train, test, nb=16, alphas=(0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 1e4, 1e5), all_alphas=False, alpha1=1.0):
    """Per-feature x depth-bucket Bernoulli presence with backoff to the feature rate scaled by depth.
    Cells with no train/test hits and a tiny prior mean are summed analytically (first order), which
    keeps the cost O(active cells) instead of O(m * nb) per smoothing value."""
    m = train['m']
    btr = depth_bucket(train['nnzc'], nb); bte = depth_bucket(test['nnzc'], nb)
    Ntr = train['n']
    n1 = np.bincount(train['indices'], minlength=m).astype(np.float64)
    g = train['indices'].size / (Ntr * m)
    p1 = (n1 + alpha1 * g) / (Ntr + alpha1)
    Nb_tr = np.bincount(btr, minlength=nb).astype(np.float64)
    Nb_te = np.bincount(bte, minlength=nb).astype(np.float64)
    mean_nnz = train['nnzc'].mean()
    s = np.ones(nb)
    for b in range(nb):
        sel = btr == b
        s[b] = train['nnzc'][sel].mean() / mean_nnz if sel.any() else (2.0 ** ((b + 0.5) * 14.0 / nb)) / mean_nnz
    ktr = train['indices'] * nb + btr[train['col']]
    kte = test['indices'] * nb + bte[test['col']]
    utr, ctr = np.unique(ktr, return_counts=True)
    ute, cte = np.unique(kte, return_counts=True)
    mu_full = 1 - (1 - p1[:, None]) ** s[None, :]     # m x nb
    N1 = np.zeros(m * nb); N1[utr] = ctr
    T1 = np.zeros(m * nb); T1[ute] = cte
    muf = mu_full.ravel()
    act = (N1 > 0) | (T1 > 0) | (muf > 1e-3)
    ai = np.flatnonzero(act)
    bi = ai % nb
    N1a, T1a, mua = N1[ai], T1[ai], muf[ai]
    Nbt, Nbe = Nb_tr[bi], Nb_te[bi]
    # bucket-wise total prior mass of the inactive cells
    M_in = np.bincount(np.arange(m * nb)[~act] % nb, weights=muf[~act], minlength=nb)
    best = (1e300, None)
    allr = {}
    for a in alphas:
        p = np.clip((N1a + a * mua) / (Nbt + a), 1e-12, 1 - 1e-9)
        bits = -(T1a * np.log2(p)).sum() - ((Nbe - T1a) * np.log2(1 - p)).sum()
        # inactive cells: T1=0, N1=0 -> p = a mu/(Nb+a); -log2(1-p) ~ p/ln2
        bits += float((Nb_te * (a * M_in / (Nb_tr + a))).sum() / LN2)
        allr[a] = float(bits)
        if bits < best[0]:
            best = (float(bits), a)
    return allr if all_alphas else best

def plugin_feature_bits(s, nb=None):
    """In-sample plug-in (empirical-entropy) presence cost: sum_f N h(p_f); with nb: per (f,bucket)."""
    m = s['m']
    def h(n1, n):
        n1 = np.asarray(n1, dtype=np.float64)
        p = n1 / n
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -(n1 * np.log2(p) + (n - n1) * np.log2(1 - p))
        return np.nansum(np.where((n1 > 0) & (n1 < n), t, 0.0))
    if nb is None:
        n1 = np.bincount(s['indices'], minlength=m)
        return float(h(n1, s['n']))
    b = depth_bucket(s['nnzc'], nb)
    tot = 0.0
    for bb in range(nb):
        sel_cols = b == bb
        if not sel_cols.any():
            continue
        sel = sel_cols[s['col']]
        n1 = np.bincount(s['indices'][sel], minlength=m)
        tot += float(h(n1, int(sel_cols.sum())))
    return tot


# ---------------------------------------------------------------------------------------------
# Ambient (empty droplet) multinomial model
def ambient_inputs(d, col=None):
    """Split a matrix into empty-droplet sub-CSC (columns with k>=1 that are not called cells) and
    the pooled profile of the called cells (used only as a prior)."""
    if col is None:
        col = columns_of(d)
    ic = d['is_cell']
    cc = np.diff(d['indptr'])
    emp = subcsc(d, (~ic) & (cc > 0), col)
    cel = subcsc(d, ic, col)
    m = int(d['m'])
    Ccells = np.bincount(cel['indices'], weights=cel['data'], minlength=m).astype(np.float64)
    zero_cols = int(((~ic) & (cc == 0)).sum())
    return emp, Ccells, zero_cols

def prior_profile(Ccells, m, delta):
    return (Ccells + delta) / (Ccells.sum() + delta * m)

def droplet_cost_bits(f, x, ecol, k, logp_entry):
    """Per-droplet -log2 Mult(x | k, p) given log2 p at each nonzero entry."""
    lf = log2_fact(k)
    sx = np.bincount(ecol, weights=log2_fact(x), minlength=len(k))
    sp = np.bincount(ecol, weights=x * logp_entry, minlength=len(k))
    return -(lf - sx + sp)

def loo_logp(f, x, ecol, k, T, N, r, gamma):
    num = T[f] - x + gamma * r[f]
    den = (N - k[ecol]) + gamma
    return np.log2(num / den)

def plugin_logp(f, T, N, r, gamma):
    return np.log2((T[f] + gamma * r[f]) / (N + gamma))

def synth_droplets(rng, p, k):
    """Draw independent Mult(k_j, p) droplets. Returns f, x, ecol (CSC order, features ascending)."""
    cdf = np.cumsum(p); cdf /= cdf[-1]
    tot = int(k.sum())
    u = rng.random(tot)
    ff = np.searchsorted(cdf, u).astype(np.int64)
    ff = np.minimum(ff, len(p) - 1)
    ecol = np.repeat(np.arange(len(k), dtype=np.int64), k)
    key = ecol * len(p) + ff
    uk, cnt = np.unique(key, return_counts=True)
    return uk % len(p), cnt.astype(np.int64), uk // len(p)


def presence_percell_feature(train, test, alpha):
    m = train['m']; Ntr = train['n']
    n1 = np.bincount(train['indices'], minlength=m).astype(np.float64)
    g = train['indices'].size / (Ntr * m)
    p = (n1 + alpha * g) / (Ntr + alpha)
    base = -np.log2(1 - p).sum()
    ent = np.log2(1 - p[test['indices']]) - np.log2(p[test['indices']])
    return base + np.bincount(test['col'], weights=ent, minlength=test['n'])

def presence_percell_featdepth(train, test, alpha, alpha1, nb=16):
    m = train['m']
    btr = depth_bucket(train['nnzc'], nb); bte = depth_bucket(test['nnzc'], nb)
    Ntr = train['n']
    n1 = np.bincount(train['indices'], minlength=m).astype(np.float64)
    g = train['indices'].size / (Ntr * m)
    p1 = (n1 + alpha1 * g) / (Ntr + alpha1)
    Nb_tr = np.bincount(btr, minlength=nb).astype(np.float64)
    mean_nnz = train['nnzc'].mean()
    s = np.ones(nb)
    for b in range(nb):
        sel = btr == b
        s[b] = train['nnzc'][sel].mean() / mean_nnz if sel.any() else (2.0 ** ((b + 0.5) * 14.0 / nb)) / mean_nnz
    ktr = train['indices'] * nb + btr[train['col']]
    N1 = np.bincount(ktr, minlength=m * nb).astype(np.float64).reshape(m, nb)
    mu = 1 - (1 - p1[:, None]) ** s[None, :]
    P = np.clip((N1 + alpha * mu) / (Nb_tr[None, :] + alpha), 1e-12, 1 - 1e-9)
    base_b = -np.log2(1 - P).sum(0)                      # per bucket
    pe = P[test['indices'], bte[test['col']]]
    ent = np.log2(1 - pe) - np.log2(pe)
    return base_b[bte] + np.bincount(test['col'], weights=ent, minlength=test['n'])
