"""w4_reader: stage-by-stage perf_counter decomposition of the production to_anndata() path for one sample / the bundle.

This re-implements, line for line, SingletBundle._load_gsm_gene_counts + _gene_level_part (layers=True) with a timer
around every step, and checks the result equals the production output. Production code is imported read-only.
Usage: python -P tools/w4_reader_stages.py [bundle] [--gsm GSM|all] [--out results/w4_reader_stages.tsv]
"""
import sys, os, time, tempfile, warnings, argparse, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..', 'python')))
warnings.filterwarnings('ignore')
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('bundle', nargs='?', default=os.path.join(HERE, '..', 'corpus', 'bundles', 'GSE145370.singlet'))
ap.add_argument('--gsm', default='all')
ap.add_argument('--out', default=os.path.join(HERE, '..', 'results', 'w4_reader_stages.tsv'))
ap.add_argument('--no-check', action='store_true')
a = ap.parse_args()
import pandas as pd, anndata as ad
from scipy.sparse import coo_matrix, csc_matrix, vstack
from singlet.bundle import SingletBundle, _pz_dims_from_bytes, _called_from_calls
import singlet._pz as _pz


class Tm:
    def __init__(s):
        s.d = collections.OrderedDict()

    def __call__(s, name):
        return _Ctx(s, name)

    def add(s, name, dt):
        s.d[name] = s.d.get(name, 0.0) + dt


class _Ctx:
    def __init__(s, tm, name):
        s.tm, s.name = tm, name

    def __enter__(s):
        s.t = time.perf_counter()

    def __exit__(s, *e):
        s.tm.add(s.name, time.perf_counter() - s.t)


def decode(raw, tm, tag):
    t = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix='.1pz', delete=False) as tf:
        tf.write(raw)
        p = tf.name
    tm.add('tmpfile write+unlink', time.perf_counter() - t)
    try:
        t = time.perf_counter()
        r = _pz.read_1pz(p)
        tm.add('native read_1pz (decode+binding copies+names->str)', time.perf_counter() - t)
    finally:
        t = time.perf_counter()
        os.unlink(p)
        tm.add('tmpfile write+unlink', time.perf_counter() - t)
    with tm('scipy csc_matrix construct'):
        mat = csc_matrix((r['data'], r['indices'], r['indptr']), shape=(r['m'], r['n']))
    return mat, r['rownames'], r['colnames']


def one_sample(b, gsm, tm):
    """Replicates _load_gsm_gene_counts + 2x _gene_level_part. Returns (X csc cells x genes, barcodes, layers)."""
    with tm('vocab: gene_order list + dict (per sample)'):
        gene_order = [g['gene_id'] for g in b.feature_vocab['genes']]
        gene_id_to_row = {g: i for i, g in enumerate(gene_order)}
    n_genes = len(gene_order)
    raws = {}
    for kind in ('exon', 'intron'):
        with tm('zip extract (read STORED member + crc32)'):
            raws[kind] = b._extract_pz(gsm, f'{kind}_counts.1pz')
    with tm('cell_calls.tsv: zip read + pd.read_csv + called list'):
        calls = b._read_cell_calls(gsm)
        called = _called_from_calls(calls)
    mats = {}
    for kind in ('exon', 'intron'):
        mats[kind] = decode(raws[kind], tm, kind)
    all_bcs = [str(x) for x in mats['exon'][2]]
    with tm('barcode dict + column index'):
        bc_to_col = {bc: i for i, bc in enumerate(all_bcs)}
        matched = [bc for bc in called if bc in bc_to_col]
    n_cells = len(matched)
    gene = {}
    for kind in ('exon', 'intron'):
        mat, rownames, pz_all_bcs = mats[kind]
        with tm('agg: column subset mat[:, cell_idx]'):
            _bc_to_col = {bc: i for i, bc in enumerate(pz_all_bcs)}
            _cell_idx = np.array([_bc_to_col[bc] for bc in matched if bc in _bc_to_col], dtype=np.int32)
            mat_cells = mat[:, _cell_idx]
        with tm('agg: feature name -> gene (python split/dict loops)'):
            gids = [r_.split('_')[0] for r_ in rownames]
            feat_to_gene = np.array([gene_id_to_row.get(g, -1) for g in gids], dtype=np.int32)
        with tm('agg: indicator @ matrix, astype int32'):
            valid = feat_to_gene >= 0
            rows_v = feat_to_gene[valid]
            indicator = coo_matrix((np.ones(valid.sum(), dtype=np.float32), (rows_v, np.arange(valid.sum()))),
                                   shape=(n_genes, valid.sum())).tocsr()
            gene[kind] = indicator.dot(mat_cells[valid, :]).astype(np.int32)
    with tm('X = exon_gene + intron_gene; .T.tocsc()'):
        gene_mat = (gene['exon'] + gene['intron']).astype(np.int32)
        X = gene_mat.T.tocsc()
    layers = {}
    for kind in ('exon', 'intron'):
        fname = f'{kind}_counts.1pz'
        with tm('layers: list_files + header check'):
            set(b.list_files(gsm))
            with b._zip().open(f'samples/{gsm}/{fname}') as fh:
                _pz_dims_from_bytes(fh.read(96))
        with tm('zip extract (read STORED member + crc32)'):
            raw = b._extract_pz(gsm, fname)
        mat, rownames, colnames = decode(raw, tm, kind)
        with tm('layers: AnnData(mat.T.tocsr()) + pd.Index(names) + is_unique'):
            adata = ad.AnnData(X=mat.T.tocsr())
            adata.var_names = pd.Index(rownames)
            adata.obs_names = pd.Index(colnames)
        with tm('vocab: gene_order list + dict (per sample)'):
            gene_order2 = [g['gene_id'] for g in b.feature_vocab['genes']]
            gene_row = {g: i for i, g in enumerate(gene_order2)}
        with tm('layers: feature name -> gene (python split/dict loops)'):
            feat_gene = np.array([gene_row.get(str(f).split('_')[0], -1) for f in adata.var_names], dtype=np.int64)
        with tm('layers: projector, X[idx] @ projector, astype, tocsr'):
            valid = feat_gene >= 0
            projector = coo_matrix((np.ones(int(valid.sum()), dtype=np.float32), (np.flatnonzero(valid), feat_gene[valid])),
                                   shape=(adata.n_vars, len(gene_order2))).tocsr()
            bc_row = {bc: i for i, bc in enumerate(adata.obs_names)}
            idx = np.array([bc_row[bc] for bc in matched if bc in bc_row], dtype=np.int64)
            layers[kind] = (adata.X[idx, :] @ projector).astype(np.int32).tocsr()
    return X, matched, layers


b = SingletBundle.open(a.bundle)
gsms = b.gsm_ids if a.gsm == 'all' else [a.gsm]
b._load_gsm_gene_counts(gsms[0])  # warm caches
tm_all = Tm()
Xs = []
for g in gsms:
    tm = Tm()
    t0 = time.perf_counter()
    X, bcs, layers = one_sample(b, g, tm)
    dt = time.perf_counter() - t0
    ok = 'unchecked'
    if not a.no_check:
        prodX, prodbcs = b._load_gsm_gene_counts(g)
        ok = (prodbcs == bcs) and (abs(prodX - X).sum() == 0)
        for kind in ('exon', 'intron'):
            ref = b._gene_level_part(g, kind, bcs)
            ok = ok and (abs(ref - layers[kind]).sum() == 0)
    for k, v in tm.d.items():
        tm_all.add(k, v)
    tm_all.add('TOTAL', dt)
    print(f'{g}: cells={X.shape[0]} nnz={X.nnz} total {dt:.2f}s equal_to_production={ok}', flush=True)
    Xs.append(X)
t = time.perf_counter()
Xall = vstack(Xs, format='csr')
tm_all.add('bundle: vstack CSR (not in TOTAL)', time.perf_counter() - t)
total = tm_all.d['TOTAL']
print()
print(f'{"stage":72s} {"seconds":>9s} {"share":>7s}')
for k, v in tm_all.d.items():
    if k == 'TOTAL':
        continue
    print(f'{k:72s} {v:9.3f} {100 * v / total:6.1f}%')
print(f'{"TOTAL (sum over samples, excl. vstack/obs/AnnData)":72s} {total:9.3f}')
with open(a.out, 'w') as f:
    f.write('stage\tseconds\tshare\n')
    for k, v in tm_all.d.items():
        f.write(f'{k}\t{v:.4f}\t{v / total:.4f}\n')
