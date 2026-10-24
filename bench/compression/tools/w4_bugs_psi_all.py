"""w4_bugs item 1+3 over the 14-sample bundle: run bin/w4_bugs_psi.exe (HEAD headers) and bin/w4_bugs_psi_8e.exe (headers at
8e1803b = git_sha in every shipped summary.json) on every sj_counts.1pz, compare with the shipped splice_psi.1pz."""
import os, subprocess, sys, hashlib, json, glob
import numpy as np
sys.path.insert(0, '../../python')
from singlet._pz import read_1pz
root = 'corpus_w4_bugs/GSE145370/samples/'
rows = []
tot = dict(n=0, trunc_bad=0, rint_bad=0, one=0, half=0, shipbytes=0)
for gsm in sorted(os.listdir(root)):
    D = root + gsm + '/'
    outs = {}
    for tag, exe in (('head', 'bin/w4_bugs_psi.exe'), ('e8', 'bin/w4_bugs_psi_8e.exe')):
        pre = f'corpus_w4_bugs/psi_{tag}_{gsm}'
        if not os.path.exists(pre + '.rint.1pz'):
            subprocess.run([os.path.abspath(exe), D + 'sj_counts.1pz', pre], check=True, capture_output=True)
        outs[tag] = pre
    ship = open(D + 'splice_psi.1pz', 'rb').read()
    sd = np.asarray(read_1pz(D + 'splice_psi.1pz')['data'])
    f32 = np.fromfile(outs['head'] + '.psi_f32.bin', dtype=np.float32)
    tv = np.asarray(read_1pz(outs['head'] + '.intree.1pz')['data'])
    r = dict(gsm=gsm, nnz=len(sd), n_one=int((sd == 1).sum()), frac_psi_eq_1=float((f32 == 1).mean()),
             n_half=int((f32 == 0.5).sum()), n_between=int(((f32 > .5) & (f32 < 1)).sum()),
             intree_vs_ship_mismatch=int((tv != sd).sum()),
             rint_bytes_equal_head=open(outs['head'] + '.rint.1pz', 'rb').read() == ship,
             rint_bytes_equal_8e=open(outs['e8'] + '.rint.1pz', 'rb').read() == ship,
             trunc_bytes_equal_head=open(outs['head'] + '.intree.1pz', 'rb').read() == ship,
             distinct_psi=len(np.unique(f32)))
    r['mismatch_pct'] = 100 * r['intree_vs_ship_mismatch'] / r['nnz']
    rows.append(r)
    print(json.dumps(r))
cols = list(rows[0].keys())
with open('results/w4_bugs_psi_all.tsv', 'w') as f:
    f.write('\t'.join(cols) + '\n')
    for r in rows: f.write('\t'.join(str(r[c]) for c in cols) + '\n')
print('TOTAL nnz', sum(r['nnz'] for r in rows), 'mismatch', sum(r['intree_vs_ship_mismatch'] for r in rows),
      'rint head bytes equal', sum(r['rint_bytes_equal_head'] for r in rows), '/14;  rint 8e bytes equal', sum(r['rint_bytes_equal_8e'] for r in rows),
      '/14; trunc bytes equal', sum(r['trunc_bytes_equal_head'] for r in rows))
