"""Singlet junction counts / PSI vs STARsolo SJ (+ STAR SJ.out.tab) on one sample.
Run inside a Slurm job:  python sj_compare.py SINGLET_DIR SOLO_DIR GTF OUT_JSON
SINGLET_DIR: Singlet output prefix (sj_counts*, splice_psi*, star_SJ.out.tab)
SOLO_DIR:    STARsolo prefix (Solo.out/SJ/raw, Solo.out/Gene/filtered, SJ.out.tab)
"""
import sys, gzip, json, collections
import numpy as np, scipy.sparse as sp
from scipy.io import mmread
from scipy.stats import spearmanr, pearsonr

SD, OD, GTF, OUT = sys.argv[1:5]


def opn(p):
    with open(p, 'rb') as f:
        m = f.read(2)
    return gzip.open(p, 'rb') if m == b'\x1f\x8b' else open(p, 'rb')


def lines(p):
    with opn(p) as f:
        return [l.decode().rstrip('\n') for l in f]


def mtx(p):
    with opn(p) as f:
        hdr = f.readline().decode()
    with opn(p) as f:
        return sp.csr_matrix(mmread(f)), hdr.strip()


def parse(n):
    c, r = n.rsplit(':', 1)
    a, b = r.split('-')
    return c, int(a), int(b)


R = {}
# ---------- load ----------
s_feat = lines(f'{SD}/sj_counts_features.tsv.gz')
s_bc = lines(f'{SD}/sj_counts_barcodes.tsv.gz')
S, S_hdr = mtx(f'{SD}/sj_counts.mtx.gz')            # junction x barcode
P, P_hdr = mtx(f'{SD}/splice_psi.mtx')
R['singlet_shape'] = list(S.shape)
R['singlet_hdr'] = S_hdr
R['psi_hdr'] = P_hdr
raw_keys = [parse(n) for n in s_feat]
s_keys = list(raw_keys)

sj_raw = f'{OD}/Solo.out/SJ/raw'
o_keys = []
for l in lines(f'{sj_raw}/features.tsv'):
    f = l.split('\t')
    o_keys.append((f[0], int(f[1]), int(f[2])))
o_bc = lines(f'{sj_raw}/barcodes.tsv')
O, _ = mtx(f'{sj_raw}/matrix.mtx')
cells = lines(f'{OD}/Solo.out/Gene/filtered/barcodes.tsv')
R['solo_shape'] = list(O.shape)
R['n_cells_called'] = len(cells)


def sjout(p):
    d = {}
    for l in lines(p):
        f = l.split('\t')
        d[(f[0], int(f[1]), int(f[2]))] = (int(f[6]), int(f[7]), int(f[5]), int(f[3]), int(f[4]))
    return d


sjo_s = sjout(f'{SD}/star_SJ.out.tab')
sjo_o = sjout(f'{OD}/SJ.out.tab')
R['SJ.out.tab_rows'] = {'singlet_star': len(sjo_s), 'starsolo': len(sjo_o)}
R['SJ.out.tab_identical_between_runs'] = (sjo_s == sjo_o)

# ---------- coordinate convention ----------
best = None
oset = set(sjo_o)
for ds in (-1, 0, 1):
    for de in (-1, 0, 1):
        n = sum(((c, a + ds, b + de) in oset) for c, a, b in s_keys)
        if best is None or n > best[0]:
            best = (n, ds, de)
R['coord_shift_singlet_to_STAR'] = {'ds': best[1], 'de': best[2], 'matched': best[0]}
ds, de = best[1], best[2]
s_keys = [(c, a + ds, b + de) for c, a, b in s_keys]

# ---------- common junction universe & cell columns ----------
uni = {}
for k in list(s_keys) + o_keys:
    uni.setdefault(k, len(uni))
nj = len(uni)


def remap(M, keys):
    idx = np.array([uni[k] for k in keys])
    M = M.tocoo()
    return sp.csr_matrix((M.data, (idx[M.row], M.col)), shape=(nj, M.shape[1]))


S_u = remap(S, s_keys)
O_u = remap(O, o_keys)
S_bc = {b: i for i, b in enumerate(s_bc)}
O_bc = {b: i for i, b in enumerate(o_bc)}
R['called_cells_in_singlet_sj'] = sum(c in S_bc for c in cells)
cs = [c for c in cells if c in S_bc and c in O_bc]
Sc = S_u[:, [S_bc[c] for c in cs]].tocsr()
Oc = O_u[:, [O_bc[c] for c in cs]].tocsr()

# ---------- A. per-junction ----------
S_all = np.asarray(S_u.sum(1)).ravel()
O_all = np.asarray(O_u.sum(1)).ravel()
S_cell = np.asarray(Sc.sum(1)).ravel()
O_cell = np.asarray(Oc.sum(1)).ravel()
inv = [None] * nj
for k, i in uni.items():
    inv[i] = k
J_o = np.array([sjo_o.get(k, (0, 0, 0, 0, 0))[0] for k in inv])
R['totals'] = {'singlet_all_bc': int(S_all.sum()), 'starsolo_UMI_all_bc': int(O_all.sum()),
               'singlet_called': int(S_cell.sum()), 'starsolo_UMI_called': int(O_cell.sum()),
               'SJ.out.tab_unique_reads': int(J_o.sum()), 'junctions_singlet_nz': int((S_all > 0).sum()),
               'junctions_starsolo_nz': int((O_all > 0).sum()), 'junctions_SJ.out.tab': int((J_o > 0).sum())}


def jac(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def corr(x, y, m):
    x, y = np.log10(1 + x[m]), np.log10(1 + y[m])
    return {'n': int(m.sum()), 'pearson_log10': float(pearsonr(x, y)[0]), 'spearman': float(spearmanr(x, y)[0])}


R['junction_set'] = {}
for N in (1, 2, 3, 5, 10, 30):
    R['junction_set'][f'N>={N}'] = {
        'n_singlet': int((S_all >= N).sum()), 'n_starsolo_UMI': int((O_all >= N).sum()), 'n_SJout_reads': int((J_o >= N).sum()),
        'jaccard_singlet_vs_starsolo': jac(S_all >= N, O_all >= N),
        'jaccard_singlet_vs_SJout': jac(S_all >= N, J_o >= N),
        'jaccard_starsolo_vs_SJout': jac(O_all >= N, J_o >= N)}
R['junction_count_corr'] = {
    'singlet_vs_starsolo_UMI_all': corr(S_all, O_all, (S_all > 0) | (O_all > 0)),
    'singlet_vs_SJout_reads': corr(S_all, J_o, (S_all > 0) | (J_o > 0)),
    'starsolo_UMI_vs_SJout_reads': corr(O_all, J_o, (O_all > 0) | (J_o > 0)),
    'singlet_vs_starsolo_called_cells': corr(S_cell, O_cell, (S_cell > 0) | (O_cell > 0)),
    'singlet_vs_starsolo_both_N>=10': corr(S_all, O_all, (S_all >= 10) & (O_all >= 10))}
mm10 = (S_all >= 10) & (O_all >= 10)
R['singlet_over_starsolo_ratio_median(N>=10)'] = float(np.median(S_all[mm10] / O_all[mm10]))
R['singlet_over_SJout_ratio_median(N>=10)'] = float(np.median(S_all[(S_all >= 10) & (J_o >= 10)] / J_o[(S_all >= 10) & (J_o >= 10)]))
R['singlet_junctions_only_in_singlet(N>=3)'] = int(((S_all >= 3) & (O_all == 0)).sum())
R['starsolo_junctions_only_in_starsolo(N>=3)'] = int(((O_all >= 3) & (S_all == 0)).sum())

# ---------- B. per-cell ----------
cn_s = np.asarray(Sc.sum(0)).ravel()
cn_o = np.asarray(Oc.sum(0)).ravel()
nz_s = np.asarray((Sc > 0).sum(0)).ravel()
nz_o = np.asarray((Oc > 0).sum(0)).ravel()
Sb = (Sc > 0).astype(np.int8)
Ob = (Oc > 0).astype(np.int8)
inter = np.asarray(Sb.multiply(Ob).sum(0)).ravel()
union = nz_s + nz_o - inter
cell_j = inter / np.maximum(union, 1)
Sx, Ox = Sc.tocoo(), Oc.tocoo()
d = collections.defaultdict(lambda: [0, 0])
for r, c, v in zip(Sx.row, Sx.col, Sx.data):
    d[(r, c)][0] = v
for r, c, v in zip(Ox.row, Ox.col, Ox.data):
    d[(r, c)][1] = v
arr = np.array(list(d.values()), dtype=float)
bn = (arr[:, 0] > 0) & (arr[:, 1] > 0)
R['per_cell'] = {
    'cells': len(cs),
    'total_corr_across_cells': {'pearson': float(pearsonr(cn_s, cn_o)[0]), 'spearman': float(spearmanr(cn_s, cn_o)[0])},
    'median_sj_per_cell': {'singlet': float(np.median(cn_s)), 'starsolo_UMI': float(np.median(cn_o))},
    'median_distinct_junctions_per_cell': {'singlet': float(np.median(nz_s)), 'starsolo': float(np.median(nz_o))},
    'median_per_cell_jaccard_of_junction_sets': float(np.median(cell_j)),
    'entries_union': int(len(arr)), 'entries_both_nonzero': int(bn.sum()),
    'entry_pearson_union': float(pearsonr(arr[:, 0], arr[:, 1])[0]),
    'entry_exact_equal_frac_where_both_nz': float((arr[bn, 0] == arr[bn, 1]).mean()),
    'entry_pearson_both_nz': float(pearsonr(arr[bn, 0], arr[bn, 1])[0])}

# ---------- C. cassette-exon PSI ----------
tx = collections.defaultdict(list)
for l in open(GTF):
    if l.startswith('#'):
        continue
    f = l.rstrip('\n').split('\t')
    if f[2] != 'exon':
        continue
    t = f[8].split('transcript_id "')[1].split('"')[0]
    tx[t].append((f[0], int(f[3]), int(f[4])))
jset, trip = set(), set()
for t, ex in tx.items():
    ex.sort(key=lambda x: x[1])
    for a, b in zip(ex[:-1], ex[1:]):
        if b[1] > a[2] + 1:
            jset.add((a[0], a[2] + 1, b[1] - 1))
for t, ex in tx.items():
    for a, b, c in zip(ex[:-2], ex[1:-1], ex[2:]):
        j3 = (a[0], a[2] + 1, c[1] - 1)
        j1 = (a[0], a[2] + 1, b[1] - 1)
        j2 = (a[0], b[2] + 1, c[1] - 1)
        if j3 in jset and j1 in jset and j2 in jset and j3 != j1 and (b[2] - b[1]) < 500:
            trip.add((j1, j2, j3))
trip = sorted(trip)
R['cassette_events_annotated'] = len(trip)
T = np.array([[uni.get(j, -1) for j in t] for t in trip])
T = T[(T >= 0).all(1)]
R['cassette_events_with_all_3_junctions_seen_by_any_tool'] = int(len(T))


def psi_pseudobulk(vec):
    x = vec[T]
    return (x[:, 0] + x[:, 1]) / 2.0, x[:, 2]


def psi_cmp(a, b, minc):
    ia, sa = a
    ib, sb = b
    m = (ia + sa >= minc) & (ib + sb >= minc)
    if m.sum() < 3:
        return {'n': int(m.sum())}
    pa = ia[m] / (ia[m] + sa[m])
    pb = ib[m] / (ib[m] + sb[m])
    return {'n': int(m.sum()), 'pearson': float(pearsonr(pa, pb)[0]), 'spearman': float(spearmanr(pa, pb)[0]),
            'MAE': float(np.abs(pa - pb).mean()), 'frac_abs_diff_lt_0.05': float((np.abs(pa - pb) < 0.05).mean())}


S_ps, O_ps, J_ps = (psi_pseudobulk(v.astype(float)) for v in (S_cell, O_cell, J_o))
R['cassette_pseudobulk_called_cells'] = {f'min_cov>={m}': {'singlet_vs_starsolo': psi_cmp(S_ps, O_ps, m)} for m in (10, 20, 50)}
S_all_ps, O_all_ps = psi_pseudobulk(S_all.astype(float)), psi_pseudobulk(O_all.astype(float))
R['cassette_pseudobulk_all_barcodes'] = {f'min_cov>={m}': {
    'singlet_vs_starsolo': psi_cmp(S_all_ps, O_all_ps, m),
    'singlet_vs_SJout_reads': psi_cmp(S_all_ps, J_ps, m),
    'starsolo_vs_SJout_reads': psi_cmp(O_all_ps, J_ps, m)} for m in (20, 50)}


def psi_cell(M):
    X = M.tocsr()
    return (X[T[:, 0]].toarray() + X[T[:, 1]].toarray()) / 2.0, X[T[:, 2]].toarray()


iS, sS = psi_cell(Sc)
iO, sO = psi_cell(Oc)
pc = {}
for minc in (2, 5, 10):
    m = ((iS + sS) >= minc) & ((iO + sO) >= minc)
    pa = iS[m] / (iS[m] + sS[m])
    pb = iO[m] / (iO[m] + sO[m])
    pc[f'min_cov>={minc}'] = {'n_cell_event_pairs': int(m.sum()),
                              'pearson': float(pearsonr(pa, pb)[0]) if m.sum() > 3 else None,
                              'MAE': float(np.abs(pa - pb).mean()) if m.sum() else None,
                              'frac_equal_within_0.05': float((np.abs(pa - pb) < 0.05).mean()) if m.sum() else None}
R['cassette_per_cell'] = pc

# ---------- D. Singlet emitted PSI: 0/1 bug status & consistency ----------
Pc = P.tocoo()
v = Pc.data
R['singlet_psi_entries'] = {
    'shape': list(P.shape), 'nnz': int(len(v)), 'dtype': str(v.dtype),
    'frac_exactly_0': float((v == 0).mean()), 'frac_exactly_1': float((v == 1).mean()),
    'frac_strictly_between_0_1': float(((v > 0) & (v < 1)).mean()),
    'n_distinct_values': int(len(np.unique(np.round(v, 4)))), 'max': float(v.max()), 'min': float(v.min())}
don, acc = collections.defaultdict(list), collections.defaultdict(list)
for i, (c, a, b) in enumerate(raw_keys):
    don[f'{c}:{a}-D'].append(i)
    acc[f'{c}:{b}-A'].append(i)
evs = sorted([(k, j) for k, j in list(don.items()) + list(acc.items()) if len(j) >= 2], key=lambda x: x[0])
jev = np.full(len(raw_keys), -1)
for e, (k, js) in enumerate(evs):
    for j in js:
        if jev[j] == -1:
            jev[j] = e
R['psi_events'] = len(evs)
Sc_ = S.tocoo()
m = jev[Sc_.row] >= 0
ev_of = jev[Sc_.row][m]
key = ev_of.astype(np.int64) * S.shape[1] + Sc_.col[m]
uk, inv_k = np.unique(key, return_inverse=True)
ssum = np.bincount(inv_k, weights=Sc_.data[m].astype(float))
rec = sp.csr_matrix((Sc_.data[m] / ssum[inv_k], (Sc_.row[m], Sc_.col[m])), shape=S.shape)
diff = rec - P.tocsr()
R['psi_recompute_from_singlet_counts'] = {'nnz_recomputed': int(rec.nnz), 'nnz_emitted': int(P.nnz),
                                          'max_abs_diff': float(abs(diff).max()) if diff.nnz else 0.0}
# PSI from STARsolo counts with the same event definition vs Singlet emitted PSI (called cells)
Oc_ = O_u[:, [O_bc[c] for c in cs]].tocoo()
s_idx = {k: i for i, k in enumerate(s_keys)}
u2s = np.array([s_idx.get(inv[u], -1) for u in range(nj)])
rj = u2s[Oc_.row]
mm = rj >= 0
rj, cc, vv = rj[mm], Oc_.col[mm], Oc_.data[mm]
evo = jev[rj]
m2 = evo >= 0
ko = evo[m2].astype(np.int64) * len(cs) + cc[m2]
uko, ik = np.unique(ko, return_inverse=True)
so = np.bincount(ik, weights=vv[m2].astype(float))
psi_o = sp.csr_matrix((vv[m2] / so[ik], (rj[m2], cc[m2])), shape=(S.shape[0], len(cs)))
psi_s = P.tocsc()[:, [S_bc[c] for c in cs]].tocsr()
A, Bd = psi_s.tocoo(), psi_o.tocoo()
da = {(r, c): x for r, c, x in zip(A.row, A.col, A.data)}
db = {(r, c): x for r, c, x in zip(Bd.row, Bd.col, Bd.data)}
common = [k for k in da if k in db]
a_ = np.array([da[k] for k in common])
b_ = np.array([db[k] for k in common])
R['psi_singlet_vs_starsolo_same_definition_called_cells'] = {
    'entries_singlet': len(da), 'entries_starsolo': len(db), 'entries_common': len(common),
    'pearson': float(pearsonr(a_, b_)[0]), 'MAE': float(np.abs(a_ - b_).mean()),
    'frac_equal_within_0.05': float((np.abs(a_ - b_) < 0.05).mean())}
json.dump(R, open(OUT, 'w'), indent=1)
print(json.dumps(R, indent=1))
