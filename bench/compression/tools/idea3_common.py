"""Idea 3 shared helpers: load a sample (exon+intron), gene structure, caching."""
import sys, os, glob, hashlib, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'python'))
from singlet._pz import read_1pz
import numpy as np
import scipy.sparse as sp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CACHE = os.path.join(ROOT, 'corpus_idea3', 'cache')
os.makedirs(CACHE, exist_ok=True)

TEST = ['pbmc_v2_typical', 'pbmc_v2_shallow', 'pbmc_v3_lowgenes', 'brain_sn_v3_deep', 'brain_sn_v3_intronic', 'tcell_5p', 'tumor_3p']
TRAIN_H = ['h_pbmc_a', 'h_pbmc_b', 'h_b', 'h_c', 'h_d']
TRAIN_M = ['m_brain_a', 'm_brain_b', 'm_brain_c', 'm_brain_d']
STUDY = {'pbmc_v2_typical': 'GSE142444', 'pbmc_v2_shallow': 'GSE142444', 'pbmc_v3_lowgenes': 'GSE244263',
         'brain_sn_v3_deep': 'GSE285936', 'brain_sn_v3_intronic': 'GSE285936', 'tcell_5p': 'GSE152522',
         'tumor_3p': 'GSE145370', 'h_pbmc_a': 'GSE165080', 'h_pbmc_b': 'GSE165080', 'h_b': 'GSE285335',
         'h_c': 'GSE244515', 'h_d': 'GSE268210', 'm_brain_a': 'GSE249382', 'm_brain_b': 'GSE249382',
         'm_brain_c': 'GSE245201', 'm_brain_d': 'GSE254360'}
SPECIES = {k: ('mouse' if k.startswith('m_') or k.startswith('brain') else 'human') for k in STUDY}

def sample_dir(sid):
    for base in ('corpus', 'corpus_train', 'corpus_idea3'):
        d = os.path.join(ROOT, base, sid)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(sid)

def load_matrix(path):
    r = read_1pz(path)
    M = sp.csc_matrix((r['data'].astype(np.int64), r['indices'], r['indptr']), shape=(r['m'], r['n']))
    return M, r['rownames'], r['colnames']

def gene_structure(rownames):
    """rows are contiguous by gene. returns gene_of_row (int32), gene_ids(list), first_row(len G+1)."""
    gid = [r.split('_', 1)[0] for r in rownames]
    gene_of_row = np.empty(len(gid), np.int32)
    ids = []
    prev = None
    for i, g in enumerate(gid):
        if g != prev:
            ids.append(g)
            prev = g
        gene_of_row[i] = len(ids) - 1
    first = np.zeros(len(ids) + 1, np.int64)
    np.add.at(first, gene_of_row + 1, 1)
    first = np.cumsum(first)
    return gene_of_row, ids, first

def interval_coord(name):
    return name.rsplit('_', 1)[-1]

class Sample:
    """exon + intron CSC matrices (interval level) of one sample, plus gene structure."""
    def __init__(self, sid):
        self.sid = sid
        d = sample_dir(sid)
        self.E, self.erows, self.cols = load_matrix(os.path.join(d, 'exon_counts.1pz'))
        self.I, self.irows, icols = load_matrix(os.path.join(d, 'intron_counts.1pz'))
        assert icols == self.cols
        self.egene, self.gids_e, self.efirst = gene_structure(self.erows)
        self.igene, self.gids_i, self.ifirst = gene_structure(self.irows)
        # map intron gene index -> exon gene index
        pos = {g: i for i, g in enumerate(self.gids_e)}
        self.i2e = np.array([pos[g] for g in self.gids_i], np.int64)

def rowhash(rownames):
    h = hashlib.sha1()
    for r in rownames:
        h.update(r.encode()); h.update(b'\0')
    return h.hexdigest()[:12]

def gene_sum(M, gene_of_row, G):
    """interval-level CSC -> gene-level CSC (sum of intervals)."""
    # row aggregation matrix
    A = sp.csr_matrix((np.ones(len(gene_of_row), np.int64), (gene_of_row, np.arange(len(gene_of_row)))), shape=(G, len(gene_of_row)))
    return (A @ M).tocsc()
