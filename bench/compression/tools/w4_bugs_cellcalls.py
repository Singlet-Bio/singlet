"""w4_bugs item 2: what cell_calls.tsv really contains, on all 21 inspected samples (7 corpus + 14 bundle)."""
import sys, os, json, glob, collections
import numpy as np
sys.path.insert(0, '../../python')
from singlet._pz import read_1pz, info_1pz
items = []
for d in sorted(glob.glob('corpus/*/')):
    if os.path.exists(d + 'cell_calls.tsv'):
        items.append((os.path.basename(os.path.normpath(d)), d))
for g in sorted(os.listdir('corpus_w4_bugs/GSE145370/samples')):
    items.append((g, f'corpus_w4_bugs/GSE145370/samples/{g}/'))
rows = []
for name, d in items:
    lines = open(d + 'cell_calls.tsv').read().split('\n')
    hdr = lines[0].split('\t'); body = [l.split('\t') for l in lines[1:] if l]
    col = {h: [r[i] for r in body] for i, h in enumerate(hdr)}
    pv = col.get('empty_drops_pvalue'); meth = collections.Counter(col['call_method'])
    isc = collections.Counter(col['is_cell'])
    summ = json.load(open(d + 'summary.json')) if os.path.exists(d + 'summary.json') else {}
    ex = d + 'exon_counts.1pz'
    ncol = info_1pz(ex)['n'] if os.path.exists(ex) else None
    r = dict(sample=name, header='|'.join(hdr), n_rows=len(body), methods=dict(meth), is_cell=dict(isc),
             pvalue_values=dict(collections.Counter(pv)), n_cells_called_summary=summ.get('n_cells_called'),
             estimated_cells=summ.get('estimated_cells'), exon_columns=ncol,
             rows_sorted_bytewise=[r[0] for r in body] == sorted(r[0] for r in body))
    # are all called barcodes columns of the exon matrix? are they the first columns?
    if ncol:
        cn = read_1pz(ex)['colnames']; pos = {b: i for i, b in enumerate(cn)}
        called = [r[0] for r in body]
        idx = np.array([pos.get(b, -1) for b in called])
        r['called_missing_from_matrix'] = int((idx < 0).sum())
        r['called_pos_max'] = int(idx.max()); r['called_pos_median'] = float(np.median(idx))
    rows.append(r); print(json.dumps(r))
with open('results/w4_bugs_cellcalls.json', 'w') as f: json.dump(rows, f, indent=1)
print('methods over all:', collections.Counter(m for r in rows for m in r['methods']),
      '| p-values over all:', collections.Counter(p for r in rows for p in r['pvalue_values']),
      '| is_cell over all:', collections.Counter(p for r in rows for p in r['is_cell']),
      '| headers:', set(r['header'] for r in rows))
print('rows == n_cells_called:', sum(r['n_rows'] == r['n_cells_called_summary'] for r in rows), 'of', sum(r['n_cells_called_summary'] is not None for r in rows))
