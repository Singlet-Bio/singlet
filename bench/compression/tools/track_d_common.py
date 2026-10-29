"""Shared loaders for Track D tools. Corpus files are untrusted data: only parsed, never executed."""
import glob, os, sys, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'python'))
from singlet._pz import read_1pz

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
CACHE = os.environ.get('TD_CACHE', os.path.join(ROOT, 'results', '.track_d_cache'))

def corpus_files():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, 'corpus', '*', '*_counts.1pz'))):
        sample = os.path.basename(os.path.dirname(p))
        member = os.path.basename(p).replace('_counts.1pz', '')
        out.append((sample, member, p))
    return out

def load(sample, member, path=None):
    """Returns dict with m,n,indptr,indices,data (numpy), colnames (list), is_cell (bool array, True = called)."""
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, f'{sample}__{member}.npz')
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=False)
        return {k: z[k] for k in z.files}
    path = path or os.path.join(ROOT, 'corpus', sample, f'{member}_counts.1pz')
    r = read_1pz(path)
    colnames = list(r['colnames'])
    calls = {}
    with open(os.path.join(ROOT, 'corpus', sample, 'cell_calls.tsv')) as f:
        rd = csv.DictReader(f, delimiter='\t')
        for row in rd:
            calls[row['barcode']] = row['is_cell'] == 'TRUE'
    # barcode matching: colnames may carry a suffix such as -1
    def key(b): return b.split('-')[0]
    is_cell = np.array([calls.get(b, calls.get(key(b), False)) for b in colnames], dtype=bool)
    d = dict(m=np.int64(r['m']), n=np.int64(r['n']), indptr=np.asarray(r['indptr'], dtype=np.int64),
             indices=np.asarray(r['indices'], dtype=np.int64), data=np.asarray(r['data'], dtype=np.int64),
             is_cell=is_cell)
    np.savez(cp, **d)
    return d
