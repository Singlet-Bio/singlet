"""Before/after of the cell-calling change on the gate datasets: cv2_stored output (is_cell = 2.2, called_v2 = 2.1) against the truth
columns of the gate-2 artifact tables. usage: cc_eval.py OUTDIR ds1 ds2 ... ; reads OUTDIR/<ds>.tsv"""
import sys
import numpy as np, pandas as pd
A = '/mnt/projects/debruinz_project/debruinz/singlet/release-gate/gate2/artifacts/cell-calling'
out = sys.argv[1]
print('dataset      caller   n_cells precision recall jaccard   | edcr: precision recall jaccard')
for ds in sys.argv[2:]:
    art = pd.read_csv(f'{A}/{ds}.barcodes.tsv.gz', sep=chr(9), index_col=0)
    tcol = 'demuxlet' if ds == 'kang2018_ctrl' else 'cellranger'
    truth = set(art.index[art[tcol].astype(bool)])
    edcr = set(art.index[art['starsolo_edcr'].astype(bool)])
    r = pd.read_csv(f'{out}/{ds}.tsv', sep=chr(9), index_col=0)
    def prf(cells):
        tp = len(cells & truth); return tp / max(len(cells), 1), tp / max(len(truth), 1), tp / max(len(cells | truth), 1)
    pe, re_, je = prf(edcr)
    for name, col in (('2.1 old', 'called_v2'), ('2.2 new', 'is_cell')):
        cells = set(r.index[r[col].astype(int) == 1]); p, rc, j = prf(cells)
        print(f'{ds:14s} {name:8s} {len(cells):6d}  {p:.4f}  {rc:.4f}  {j:.4f}   | {pe:.4f} {re_:.4f} {je:.4f}')
    low = set(r.index[r['called_lowrna'].astype(int) == 1]); print(f'{ds:14s} lowrna   {len(low):6d}')
