import scipy.sparse as sp, scipy.io, numpy as np
from scipy.stats import pearsonr
import sys, os, gzip
sys.path.insert(0, '/mnt/home/debruinz/Singlet-AI/singlify/python')
import importlib.util as _ilu, os as _os
_so = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'singlify', '_pz_io.cpython-39-x86_64-linux-gnu.so')
_spec = _ilu.spec_from_file_location('_pz_io', _so)
_pz_io = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pz_io)

SINGLIFY_DIR = '/mnt/projects/debruinz_project/singlify_validation/e2e_panelA_human_af14abd'
GOLD_SOLO = '/mnt/projects/debruinz_project/singlify_validation/starsolo/SRR32855204_matched/Solo.out'

def load_1pz(path):
    d = _pz_io.read_1pz(path)
    mat = sp.csc_matrix((d['data'], d['indices'], d['indptr']), shape=tuple(d['shape']))
    return mat, list(d['rownames']), list(d['colnames'])

def read_lines(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        return [l.strip() for l in f]

print('Loading singlify exon_counts.1pz...')
sg_mat, sg_genes, sg_cells = load_1pz(f'{SINGLIFY_DIR}/exon_counts.1pz')
sg_cells_clean = [c.split('-')[0] for c in sg_cells]
print(f'singlify: {sg_mat.shape[0]}g x {sg_mat.shape[1]}c')

gene_dir = f'{GOLD_SOLO}/Gene/filtered'
gold_mat = scipy.io.mmread(f'{gene_dir}/matrix.mtx').tocsc()
gold_cells = [b.split('-')[0] for b in read_lines(f'{gene_dir}/barcodes.tsv')]
print(f'gold: {gold_mat.shape[0]}g x {gold_mat.shape[1]}c')

sg_bc2idx = {c: i for i, c in enumerate(sg_cells_clean)}
gd_bc2idx = {c: i for i, c in enumerate(gold_cells)}
common = sorted(set(sg_cells_clean) & set(gold_cells))
print(f'Common barcodes: {len(common)}')

sg_idx = [sg_bc2idx[c] for c in common]
gd_idx = [gd_bc2idx[c] for c in common]

sg_g = np.array(sg_mat[:, sg_idx].sum(axis=1)).flatten()
gd_g = np.array(gold_mat[:, gd_idx].sum(axis=1)).flatten()
mask = (sg_g > 0) | (gd_g > 0)
r_gene, _ = pearsonr(sg_g[mask], gd_g[mask])
print(f'Gene Pearson r: {r_gene:.6f}  n_genes={mask.sum()}')

sg_c = np.array(sg_mat[:, sg_idx].sum(axis=0)).flatten()
gd_c = np.array(gold_mat[:, gd_idx].sum(axis=0)).flatten()
r_cell, _ = pearsonr(sg_c, gd_c)
print(f'Cell Pearson r: {r_cell:.6f}  n_common={len(common)}')

sg_set = set(sg_cells_clean)
gd_set = set(gold_cells)
jaccard = len(sg_set & gd_set) / len(sg_set | gd_set)
print(f'Cell Jaccard: {jaccard:.4f}  sg={len(sg_set)} gd={len(gd_set)} shared={len(sg_set&gd_set)}')
print(f'Gold recall: {len(sg_set & gd_set)/len(gd_set):.4f}')

def load_sj(path):
    s = set()
    with open(path) as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 4:
                s.add((p[0], p[1], p[2], p[3]))
    return s

sg_sj = load_sj(f'{SINGLIFY_DIR}/star_SJ.out.tab')
gold_sj_path = '/mnt/projects/debruinz_project/singlify_validation/starsolo/SRR32855204_matched/SJ.out.tab'
if os.path.exists(gold_sj_path):
    gd_sj = load_sj(gold_sj_path)
    sj_j = len(sg_sj & gd_sj) / len(sg_sj | gd_sj)
    print(f'SJ Jaccard: {sj_j:.4f}  sg={len(sg_sj)} gd={len(gd_sj)} shared={len(sg_sj&gd_sj)}')
else:
    print(f'SJ gold not found at {gold_sj_path}')

import json
with open(f'{SINGLIFY_DIR}/summary.json') as f:
    summary = json.load(f)
print(f'Protocol detected: {summary.get("protocol","?")}')
print(f'Cells: {summary.get("cells","?")}  Mapping rate: {summary.get("mapping_rate","?")}')
snp_ad = os.path.exists(f'{SINGLIFY_DIR}/snp_ad.1pz')
snp_dp = os.path.exists(f'{SINGLIFY_DIR}/snp_dp.1pz')
print(f'snp_ad.1pz: {snp_ad}  snp_dp.1pz: {snp_dp}')
