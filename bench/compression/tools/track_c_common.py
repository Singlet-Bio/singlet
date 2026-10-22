"""Shared helpers for Track C python experiments."""
import os, numpy as np, glob
SCR = os.environ.get('TC_CACHE', 'C:/Users/zacha/AppData/Local/Temp/claude/C--Users-zacha-Projects-singlet-bio/8e8b9acd-9e9b-409a-a4ac-d2afe710d803/scratchpad/cache')
def load(name):
    """name like 'pbmc_v2_typical__exon' -> dict(indptr, indices, data, m, n)"""
    z = np.load(os.path.join(SCR, name + '.npz'))
    return dict(indptr=z['indptr'], indices=z['indices'], data=z['data'], m=int(z['m']), n=int(z['n']))
def colids(indptr):
    n = len(indptr) - 1
    return np.repeat(np.arange(n), np.diff(indptr))

TEST_STUDY = {'pbmc_v2_typical':'GSE142444','pbmc_v2_shallow':'GSE142444','pbmc_v3_lowgenes':'GSE244263','tcell_5p':'GSE152522',
              'tumor_3p':'GSE145370','brain_sn_v3_deep':'GSE285936','brain_sn_v3_intronic':'GSE285936'}
TRAIN_STUDY = {'h_pbmc_a':'GSE165080','h_pbmc_b':'GSE165080','h_b':'GSE285335','h_c':'GSE244515','h_d':'GSE268210',
               'm_brain_a':'GSE249382','m_brain_b':'GSE249382','m_brain_c':'GSE245201','m_brain_d':'GSE254360'}
STUDY = {**TEST_STUDY, **TRAIN_STUDY}
ORG = {k: ('mouse' if k.startswith('brain') or k.startswith('m_') else 'human') for k in STUDY}
TESTS = list(TEST_STUDY)

def col_stats(d):
    ip = d['indptr']; nz = np.diff(ip)
    cs = np.concatenate([[0], np.cumsum(d['data'].astype(np.int64))])
    tot = cs[ip[1:]] - cs[ip[:-1]]
    return nz, tot

def log2_bucket(x, nb, lo, hi):
    """bucket index of x (>=1) over log2 scale: 0..nb-1 mapping log2 x in [lo,hi]"""
    t = (np.log2(np.maximum(x, 1)) - lo) / (hi - lo) * nb
    return np.clip(np.floor(t), 0, nb - 1).astype(np.int32)
