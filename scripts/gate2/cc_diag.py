"""cell-calling diagnosis on the gate-2 artifact tables: where do Singlet's extra and missing calls sit (UMI total bins)?"""
import gzip, sys
import numpy as np, pandas as pd
A = '/mnt/projects/debruinz_project/debruinz/singlet/release-gate/gate2/artifacts/cell-calling'
for ds in sys.argv[1:]:
    df = pd.read_csv(f'{A}/{ds}.barcodes.tsv.gz', sep='\t', index_col=0)
    print('==', ds, len(df), 'cols', list(df.columns))
    tcol = 'demuxlet' if ds == 'kang2018_ctrl' else 'cellranger'
    T = df[tcol].astype(bool).to_numpy()
    for c in ('singlet', 'starsolo_edcr', 'cellbender'):
        if c in df.columns:
            a = df[c].astype(bool).to_numpy()
            print(f'  {c}: n={a.sum()} tp={(a&T).sum()} fp={(a&~T).sum()} fn={(~a&T).sum()}')
    tot = next((c for c in df.columns if 'umi' in c.lower() or 'total' in c.lower()), None)
    if tot:
        t = df[tot].to_numpy(); bins = [0, 50, 100, 200, 300, 500, 1000, 2000, 5000, 1e9]
        print('  total col', tot)
        print('  bin        truth  singlet  edcr  s_only  s_fp(not truth)  t_missed')
        s = df['singlet'].astype(bool).to_numpy(); e = df['starsolo_edcr'].astype(bool).to_numpy() if 'starsolo_edcr' in df.columns else s
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (t > lo) & (t <= hi)
            print(f'  {lo:>6}-{hi:<8.0f} {int((m&T).sum()):6d} {int((m&s).sum()):7d} {int((m&e).sum()):6d} {int((m&s&~e).sum()):7d} {int((m&s&~T).sum()):10d} {int((m&T&~s).sum()):8d}')
