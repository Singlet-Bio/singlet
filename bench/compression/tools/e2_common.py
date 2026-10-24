"""Shared loader for E2 tools. Reads the Track D npz cache (read-only); corpus data is only parsed, never executed."""
import os, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
CACHE = os.path.join(ROOT, 'results', '.track_d_cache')

SAMPLES = ['brain_sn_v3_deep', 'brain_sn_v3_intronic', 'pbmc_v2_shallow', 'pbmc_v2_typical',
           'pbmc_v3_lowgenes', 'tcell_5p', 'tumor_3p']

def load(sample, member):
    z = np.load(os.path.join(CACHE, f'{sample}__{member}.npz'), allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d['m'] = int(d['m']); d['n'] = int(d['n'])
    d['indptr'] = d['indptr'].astype(np.int64)
    d['indices'] = d['indices'].astype(np.int32)
    d['data'] = d['data'].astype(np.int32)
    d['is_cell'] = d['is_cell'].astype(bool)
    n = d['n']
    ip = d['indptr']
    d['k'] = np.diff(ip)
    cs = np.concatenate([[0], np.cumsum(d['data'], dtype=np.int64)])
    d['N'] = cs[ip[1:]] - cs[ip[:-1]]
    return d

def columns(d, cols):
    """Concatenate the CSC slices of the listed columns -> (colid_per_nnz (index into cols), indices, data)."""
    ip = d['indptr']
    lens = ip[cols + 1] - ip[cols]
    tot = int(lens.sum())
    starts = np.repeat(ip[cols] - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
    pos = np.arange(tot, dtype=np.int64) + starts
    colid = np.repeat(np.arange(len(cols)), lens)
    return colid, d['indices'][pos], d['data'][pos]
