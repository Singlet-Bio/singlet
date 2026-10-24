"""Idea 4 shared helpers: load .1pz samples, gene grouping, called/empty split, Poisson-process cost.

Run with `python -P` from bench/compression. Data under corpus/ and corpus_idea4/ is untrusted input;
we only read it through the compiled reader.
"""
import os, sys, json, re
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'python'))
from singlet._pz import read_1pz  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

TEST = {  # id -> (dir, study, organism)
    'pbmc_v2_typical': ('corpus/pbmc_v2_typical', 'GSE142444', 'human'),
    'pbmc_v2_shallow': ('corpus/pbmc_v2_shallow', 'GSE142444', 'human'),
    'pbmc_v3_lowgenes': ('corpus/pbmc_v3_lowgenes', 'GSE244263', 'human'),
    'tcell_5p': ('corpus/tcell_5p', 'GSE152522', 'human'),
    'tumor_3p': ('corpus/tumor_3p', 'GSE145370', 'human'),
    'brain_sn_v3_deep': ('corpus/brain_sn_v3_deep', 'GSE285936', 'mouse'),
    'brain_sn_v3_intronic': ('corpus/brain_sn_v3_intronic', 'GSE285936', 'mouse'),
}
TRAIN = {
    'h_pbmc_a': ('corpus_idea4/h_pbmc_a', 'GSE165080', 'human'),
    'h_pbmc_b': ('corpus_idea4/h_pbmc_b', 'GSE165080', 'human'),
    'h_b': ('corpus_idea4/h_b', 'GSE285335', 'human'),
    'h_c': ('corpus_idea4/h_c', 'GSE244515', 'human'),
    'h_d': ('corpus_idea4/h_d', 'GSE268210', 'human'),
    'm_brain_a': ('corpus_idea4/m_brain_a', 'GSE249382', 'mouse'),
    'm_brain_b': ('corpus_idea4/m_brain_b', 'GSE249382', 'mouse'),
    'm_brain_c': ('corpus_idea4/m_brain_c', 'GSE245201', 'mouse'),
    'm_brain_d': ('corpus_idea4/m_brain_d', 'GSE254360', 'mouse'),
}
ALL = {**TEST, **TRAIN}


class Sample:
    pass


def load_sample(sid):
    d, study, org = ALL[sid]
    S = Sample()
    S.id, S.study, S.org = sid, study, org
    r = read_1pz(os.path.join(ROOT, d, 'exon_counts.1pz'))
    S.m, S.n = r['m'], r['n']
    S.indptr = np.asarray(r['indptr']).astype(np.int64)
    S.indices = np.asarray(r['indices']).astype(np.int32)
    S.data = np.asarray(r['data']).astype(np.int32)
    S.rownames = r['rownames']
    S.colnames = r['colnames']
    calls = {}
    with open(os.path.join(ROOT, d, 'cell_calls.tsv')) as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            calls[p[0]] = (p[1] == 'TRUE')
    S.called = np.array([calls.get(b, False) for b in S.colnames], dtype=bool)
    cs = np.concatenate([[0], np.cumsum(S.data, dtype=np.int64)])
    S.depth = (cs[S.indptr[1:]] - cs[S.indptr[:-1]]).astype(np.int64)
    S.nz = np.diff(S.indptr)
    return S


def csr_of(S, cols=None):
    """cells x intervals CSR (int32 data) for the selected columns."""
    X = sp.csr_matrix((S.data, S.indices, S.indptr), shape=(S.n, S.m))
    if cols is not None:
        X = X[cols]
    return X


def gene_structure(rownames):
    """Group intervals by gene (rows are contiguous per gene). Returns gene_of (m,), starts (G+1,),
    gene_ids, interval length (m,)"""
    m = len(rownames)
    gene_of = np.empty(m, dtype=np.int32)
    starts = []
    ids = []
    prev = None
    ln = np.empty(m, dtype=np.float64)
    pat = re.compile(r':(\d+)-(\d+)$')
    for i, nm in enumerate(rownames):
        g = nm.split('_', 1)[0]
        if g != prev:
            starts.append(i)
            ids.append(g)
            prev = g
        gene_of[i] = len(ids) - 1
        mo = pat.search(nm)
        ln[i] = (int(mo.group(2)) - int(mo.group(1)) + 1) if mo else 100.0
    starts.append(m)
    return gene_of, np.array(starts, dtype=np.int64), ids, ln


def gene_matrix(gene_of, G, X):
    """Aggregate an interval-level CSR (cells x m) to gene level (cells x G)."""
    A = sp.csr_matrix((np.ones(len(gene_of), dtype=np.float32), (np.arange(len(gene_of)), gene_of)), shape=(len(gene_of), G))
    return (X @ A).tocsr()


LOG2E = 1.4426950408889634
