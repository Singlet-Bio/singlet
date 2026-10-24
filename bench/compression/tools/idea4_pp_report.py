"""Summarise results/idea4_pp_run.tsv (harness output of bin/idea4_pp_bench.exe) into markdown tables.

python -P tools/idea4_pp_report.py [results/idea4_pp_run.tsv]
Rows: one per (sample, fold, k, adapt).  Columns: called-cell bits/nnz (payload incl. side info), empties bits/nnz,
total payload bits/nnz, side bytes/cell, model MB, decode ms/cell of the matvec, matrix_ok.
"""
import csv, sys, collections, re

path = sys.argv[1] if len(sys.argv) > 1 else 'results/idea4_pp_run.tsv'
rows = list(csv.DictReader(open(path), delimiter='\t'))
R = collections.OrderedDict()
for r in rows:
    m = re.match(r'idea4_(static|nmf)_(\w+?)_k(\d+)_a(\d)$', r['codec'])
    if not m:
        continue
    fold, k, a = m.group(2), int(m.group(3)), int(m.group(4))
    R[(r['sample'], fold, k, a)] = r


def f(r, c, d=3):
    try:
        return ('%.' + str(d) + 'f') % float(r[c])
    except Exception:
        return 'n/a'


samples = sorted(set(k[0] for k in R))
print('| sample | fold | k | adapt | cells b/nnz | empties b/nnz | payload b/nnz | side B/cell | model MB | matvec ms/cell | dec Mnnz/s | ok |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for key, r in sorted(R.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
    print('| %s | %s | %d | %d | %s | %s | %s | %s | %.2f | %s | %s | %s |' % (
        key[0], key[1], key[2], key[3], f(r, 'bits_nnz_cells'), f(r, 'bits_nnz_amb'), f(r, 'payload_bits_per_nnz'),
        f(r, 'side_bytes_per_cell', 2), float(r['model_bytes']) / 1048576, f(r, 'matvec_ms_per_cell'), f(r, 'dec_Mnnz_s', 2), r['matrix_ok']))
