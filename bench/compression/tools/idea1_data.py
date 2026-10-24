"""Idea 1 data loader: reads .1pz via the compiled reader and caches CSC arrays as .npz in corpus_idea1/cache."""
import sys, os, glob, hashlib
sys.path.insert(0, '../../python')
import numpy as np
from singlet._pz import read_1pz

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, 'corpus_idea1', 'cache')

# study of each sample id (leave-one-STUDY-out); organism
TEST = {
 'pbmc_v2_typical': ('GSE142444', 'human'), 'pbmc_v2_shallow': ('GSE142444', 'human'),
 'pbmc_v3_lowgenes': ('GSE244263', 'human'), 'tcell_5p': ('GSE152522', 'human'),
 'tumor_3p': ('GSE145370', 'human'),
 'brain_sn_v3_deep': ('GSE285936', 'mouse'), 'brain_sn_v3_intronic': ('GSE285936', 'mouse'),
}
TRAIN = {
 'h_pbmc_a': ('GSE165080', 'human'), 'h_b': ('GSE285335', 'human'), 'h_c': ('GSE244515', 'human'),
 'm_brain_a': ('GSE249382', 'mouse'), 'm_brain_c': ('GSE245201', 'mouse'), 'm_brain_d': ('GSE254360', 'mouse'),
}
ALL = {**TEST, **TRAIN}

def colsums(d):
    c = np.concatenate([[0], np.cumsum(d['data'].astype(np.int64))])
    return c[d['indptr'][1:]] - c[d['indptr'][:-1]]

def path_of(sid, member):
    d = 'corpus' if sid in TEST else 'corpus_idea1'
    return os.path.join(ROOT, d, sid, f'{member}_counts.1pz')

def load(sid, member):
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, f'{sid}_{member}.npz')
    if os.path.exists(cp):
        z = np.load(cp)
        return dict(m=int(z['m']), n=int(z['n']), indptr=z['indptr'], indices=z['indices'], data=z['data'], rnhash=str(z['rnhash']))
    r = read_1pz(path_of(sid, member))
    ip = np.asarray(r['indptr'], dtype=np.int64)
    ix = np.asarray(r['indices'], dtype=np.int32)
    dt = np.asarray(r['data']).astype(np.uint32)
    h = hashlib.sha1('\n'.join(r['rownames']).encode()).hexdigest()
    np.savez(cp, m=r['m'], n=r['n'], indptr=ip, indices=ix, data=dt, rnhash=h)
    return dict(m=int(r['m']), n=int(r['n']), indptr=ip, indices=ix, data=dt, rnhash=h)

if __name__ == '__main__':
    for sid in ALL:
        for mem in ('exon', 'intron'):
            d = load(sid, mem)
            cc = np.diff(d['indptr'])
            D = colsums(d)
            print(sid, mem, d['m'], d['n'], len(d['indices']), d['rnhash'][:8], 'ones=%.3f' % (d['data'] == 1).mean(),
                  'D med=%d max=%d' % (np.median(D), D.max()), 'k med=%d' % np.median(cc), flush=True)

def genes(sid_or_org, member):
    """gene index per row (int32) + number of intervals per gene; cached per (org, member)"""
    org = sid_or_org if sid_or_org in ('human', 'mouse') else ALL[sid_or_org][1]
    cp = os.path.join(CACHE, f'genes_{org}_{member}.npy')
    if os.path.exists(cp):
        return np.load(cp)
    ref = 'pbmc_v2_typical' if org == 'human' else 'brain_sn_v3_deep'
    r = read_1pz(path_of(ref, member))
    g = np.empty(len(r['rownames']), dtype=np.int32)
    ids = {}
    for i, nm in enumerate(r['rownames']):
        k = nm.split('_', 1)[0]
        g[i] = ids.setdefault(k, len(ids))
    np.save(cp, g)
    return g
