#!/usr/bin/env python
"""Track D, task 4: the orchestrator oracle and an out-of-sample chooser.

Reads results/track_d_oracle_chunks.tsv (written by bin/track_d_oracle.exe: one row per 1024-column chunk with
cheap features and the byte count of each coder) and reports, per file and overall:
  best single method (whole file), per-chunk oracle, oracle gap,
  and how much of that gap is captured by classifiers trained on OTHER studies (leave-one-study-out)
  on cheap chunk features, with the bits needed to signal the choice.
Writes results/track_d_oracle.tsv (per-file summary) and prints the tables.

    python -P tools/track_d_oracle.py
"""
import csv, os, sys, math, warnings
import numpy as np
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from track_d_common import ROOT
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

SRC = os.path.join(ROOT, 'results', 'track_d_oracle_chunks.tsv')
MOD = os.path.join(ROOT, 'results', 'track_d_oracle_models.tsv')
OUT = os.path.join(ROOT, 'results', 'track_d_oracle.tsv')
STUDY = {'pbmc_v2_typical': 'GSE142444', 'pbmc_v2_shallow': 'GSE142444', 'pbmc_v3_lowgenes': 'GSE244263',
         'brain_sn_v3_deep': 'GSE285936', 'brain_sn_v3_intronic': 'GSE285936', 'tcell_5p': 'GSE152522', 'tumor_3p': 'GSE145370'}
FEATS = ['nnz', 'ncols_nz', 'mean_col_nnz', 'q10_col_nnz', 'q50_col_nnz', 'q90_col_nnz', 'max_col_nnz', 'frac_ones', 'mean_val',
         'frac_ge3', 'mean_gap', 'frac_gap0', 'frac_top64', 'meta_per_nnz', 'pos']
LOGF = {'nnz', 'ncols_nz', 'mean_col_nnz', 'q10_col_nnz', 'q50_col_nnz', 'q90_col_nnz', 'max_col_nnz', 'mean_gap'}


MNAMES = ['M2', 'mult1', 'mult16', 'amb']


def load(with_models=False):
    rows = list(csv.DictReader(open(SRC), delimiter='\t'))
    models = {}
    mnames = MNAMES if with_models else []
    if with_models:
        for r in csv.DictReader(open(MOD), delimiter='\t'):
            models[(r['sample'], r['member'], int(r['chunk']))] = r
    methods = [k for k in rows[0] if k.startswith('sz_')]
    # drop methods that were not run (all zero)
    methods = [k for k in methods if any(float(r[k]) > 0 for r in rows)]
    R = []
    for r in rows:
        d = dict(sample=r['sample'], member=r['member'], chunk=int(r['chunk']), nchunks=int(r['nchunks']))
        f = []
        for k in FEATS:
            if k == 'meta_per_nnz':
                v = float(r['meta_bytes']) / max(float(r['nnz']), 1)
            elif k == 'pos':
                v = int(r['chunk']) / max(int(r['nchunks']) - 1, 1)
            else:
                v = float(r[k])
            f.append(math.log1p(v) if k in LOGF else v)
        d['x'] = np.array(f)
        d['sz'] = np.array([float(r[m]) + (4 if m in ('sz_z1', 'sz_z3', 'sz_z9', 'sz_z19', 'sz_rawz19') else 0) for m in methods])
        if with_models:
            mr = models[(r['sample'], r['member'], int(r['chunk']))]
            side = float(mr['side_bits'])
            d['sz'] = np.concatenate([d['sz'], [(float(mr[k]) + side) / 8.0 for k in mnames]])
        R.append(d)
    return R, methods + ['sz_' + k for k in mnames]


def main():
    with_models = '--models' in sys.argv
    R, methods = load(with_models)
    global OUT
    if with_models:
        OUT = OUT.replace('.tsv', '_with_models.tsv')
    K = len(methods)
    files = sorted({(r['sample'], r['member']) for r in R})
    X = np.array([r['x'] for r in R]); S = np.array([r['sz'] for r in R])
    fid = np.array([files.index((r['sample'], r['member'])) for r in R])
    stu = np.array([STUDY[r['sample']] for r in R])
    best_idx = S.argmin(1)
    print('methods:', [m[3:] for m in methods])
    out_rows = []
    pred = {name: np.zeros(len(R), dtype=int) for name in ('logreg', 'tree', 'ridge')}
    for study in sorted(set(stu)):
        te = stu == study; tr = ~te
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        # classification of the best method
        ytr = best_idx[tr]
        if len(set(ytr)) > 1:
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
            pred['logreg'][te] = lr.predict(Xte)
            dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=0).fit(Xtr, ytr)
            pred['tree'][te] = dt.predict(Xte)
        else:
            pred['logreg'][te] = ytr[0]; pred['tree'][te] = ytr[0]
        # cost-sensitive: ridge on log size ratio vs the first method, choose argmin
        ratio = np.log(S[tr] / S[tr][:, [0]])
        # include log nnz offset implicitly: ratios are scale free
        rg = Ridge(alpha=3.0).fit(Xtr, ratio)
        pred['ridge'][te] = rg.predict(Xte).argmin(1)
    hdr = ['sample', 'member', 'nchunks', 'bytes_z3(today)', 'best_single', 'bytes_best_single', 'bytes_oracle', 'oracle_gap_pct',
           'oracle_gap_pct_incl_signal', 'sig_bits_per_chunk_entropy', 'logreg_captured_pct', 'tree_captured_pct', 'ridge_captured_pct',
           'ridge_vs_best_single_pct', 'ridge_vs_z3_pct']
    out_rows.append(hdr)
    tot = dict(z3=0, single=0, oracle=0, oracle_sig=0, ridge=0, logreg=0, tree=0)
    z3i = methods.index('sz_z3')
    for i, (s, m) in enumerate(files):
        sel = fid == i
        Sf = S[sel]
        tots = Sf.sum(0)
        bs = tots.argmin()
        single = tots[bs]
        oracle = Sf.min(1).sum()
        # signalling cost: empirical entropy of the oracle choice (bits per chunk), applied to every chunk
        cnt = np.bincount(best_idx[sel], minlength=K).astype(float)
        p = cnt[cnt > 0] / cnt.sum()
        H = float(-(p * np.log2(p)).sum())
        sig = H * sel.sum() / 8.0
        gap = single - oracle
        row = [s, m, int(sel.sum()), int(tots[z3i]), methods[bs][3:], int(single), int(oracle), 100 * gap / single, 100 * (single - oracle - sig) / single, H]
        for name in ('logreg', 'tree', 'ridge'):
            got = Sf[np.arange(sel.sum()), pred[name][sel]].sum()
            row.append(100 * (single - got) / gap if gap > 0 else float('nan'))
            if name == 'ridge':
                got_r = got
        row += [100 * (single - got_r) / single, 100 * (tots[z3i] - got_r) / tots[z3i]]
        out_rows.append(row)
        tot['z3'] += tots[z3i]; tot['single'] += single; tot['oracle'] += oracle; tot['oracle_sig'] += oracle + sig
        for name in ('logreg', 'tree', 'ridge'):
            tot[name] += Sf[np.arange(sel.sum()), pred[name][sel]].sum()
    with open(OUT, 'w') as f:
        for r in out_rows:
            f.write('\t'.join(f'{v:.4g}' if isinstance(v, float) else str(v) for v in r) + '\n')
    for r in out_rows:
        print('  '.join(f'{v:>10.3f}' if isinstance(v, float) else f'{str(v):>10s}' for v in r))
    print('TOTAL over files: today(z3) %d  sum of per-file best single %d  oracle %d (%.2f%% below single)  oracle+signal %d' %
          (tot['z3'], tot['single'], tot['oracle'], 100 * (tot['single'] - tot['oracle']) / tot['single'], tot['oracle_sig']))
    for name in ('logreg', 'tree', 'ridge'):
        print(f'  {name}: {tot[name]:.0f} bytes; captures {100 * (tot["single"] - tot[name]) / (tot["single"] - tot["oracle"]):.1f}% of the oracle gap')
    # method win counts
    print('oracle wins per method:', {methods[k][3:]: int((best_idx == k).sum()) for k in range(K)})
    print('whole-file totals per method (all files):', {methods[k][3:]: int(S[:, k].sum()) for k in range(K)})


if __name__ == '__main__':
    main()
