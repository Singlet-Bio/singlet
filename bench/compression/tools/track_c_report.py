#!/usr/bin/env python
"""Render the Track C CV table (markdown) from results/track_c_cv.tsv + results/track_c_baseline_chunks.tsv."""
import csv, sys, collections
cv = list(csv.DictReader(open('results/track_c_cv.tsv'), delimiter='\t'))
base = {(r['sample'], r['member']): r for r in csv.DictReader(open('results/track_c_baseline_chunks.tsv'), delimiter='\t')}
codecs = ['c_r1', 'c_r1b', 'c_r2', 'c_r2p', 'c_r2_nomodel', 'c_r1_insample', 'c_r2_insample']
tab = collections.OrderedDict()
for r in cv:
    tab.setdefault((r['sample'], r['member']), {})[r['codec']] = r
print('| sample | member | held-out study | nnz | today chunks b/nnz | ' + ' | '.join(codecs) + ' | c_r2 vs today | c_r2 xent | c_r2 total b/nnz | c_r2 dec Mnnz/s | ok |')
print('|' + '---|' * (9 + len(codecs)))
for (s, m), d in tab.items():
    b = float(base[(s, m)]['chunk_bits_per_nnz'])
    cells = []
    for c in codecs:
        cells.append(f"{float(d[c]['payload_bits_per_nnz']):.2f}" if c in d else '-')
    r2 = d.get('c_r2')
    if not r2: continue
    p = float(r2['payload_bits_per_nnz'])
    ok = all(x['matrix_ok'] == '1' for x in d.values())
    print(f"| {s} | {m.replace('_counts.1pz','')} | {r2['model_tag'].split('_')[-1]} | {int(r2['nnz'])/1e6:.2f}M | {b:.2f} | " + ' | '.join(cells) +
          f" | {100*(p/b-1):+.1f}% | {float(r2['xent_bits_nnz']):.3f} | {float(r2['total_bits_per_nnz']):.2f} | {float(r2['dec_Mnnz_s']):.2f} | {'OK' if ok else 'FAIL'} |")
