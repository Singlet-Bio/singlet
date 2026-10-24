#!/usr/bin/env python
"""Track D, task 3: recursion table.

Joins results/track_d_second.tsv (second-pass sizes written by bin/track_d_second.exe for every codec dump in
results/track_d_dump/) with the nnz of each matrix and writes results/track_d_recursion.tsv:

  sample member codec first_bytes first_bits_per_nnz method second_bytes ratio second_bits_per_nnz

and prints the per-codec summary (median / min / max ratio over the corpus, per method) used in notes/track_d.md.

    python -P tools/track_d_recursion.py
"""
import csv, os, sys, statistics, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from track_d_common import ROOT

NNZ = {}
for line in csv.DictReader(open(os.path.join(ROOT, 'results', 'track_d_entropy.tsv')), delimiter='\t'):
    if line['quantity'] == 'comb_presence_m' and line['subset'] == 'all':
        NNZ[(line['sample'], line['member'])] = int(line['nnz'])


def main():
    rows = []
    for r in csv.DictReader(open(os.path.join(ROOT, 'results', 'track_d_second.tsv')), delimiter='\t'):
        base = r['file'][:-4] if r['file'].endswith('.bin') else r['file']
        parts = base.split('__')
        if len(parts) < 3:
            continue
        sample, member, codec = parts[0], parts[1], '__'.join(parts[2:])
        nnz = NNZ.get((sample, member))
        if not nnz:
            continue
        fb = int(r['in_bytes']); sb = int(r['out_bytes'])
        rows.append((sample, member, codec, fb, fb * 8 / nnz, r['method'], sb, sb / fb, sb * 8 / nnz))
    rows.sort()
    out = os.path.join(ROOT, 'results', 'track_d_recursion.tsv')
    with open(out, 'w') as f:
        f.write('sample\tmember\tcodec\tfirst_bytes\tfirst_bits_per_nnz\tmethod\tsecond_bytes\tratio\tsecond_bits_per_nnz\n')
        for r in rows:
            f.write('\t'.join(f'{v:.5g}' if isinstance(v, float) else str(v) for v in r) + '\n')
    # summary: per codec x method, aggregate over files
    agg = collections.defaultdict(list)
    first = collections.defaultdict(lambda: [0, 0])
    for sample, member, codec, fb, fbpn, method, sb, ratio, sbpn in rows:
        agg[(codec, method)].append((fb, sb, ratio))
    methods = sorted({k[1] for k in agg})
    codecs = sorted({k[0] for k in agg})
    print(f'{"codec":28s}{"files":>6s}' + ''.join(f'{m:>16s}' for m in methods))
    for c in codecs:
        cells = []
        n = 0
        for m in methods:
            v = agg.get((c, m), [])
            n = max(n, len(v))
            if not v:
                cells.append(f'{"-":>16s}')
            else:
                tot_in = sum(x[0] for x in v); tot_out = sum(x[1] for x in v)
                rs = [x[2] for x in v]
                cells.append(f'{tot_out / tot_in:8.3f}({min(rs):.2f}-{max(rs):.2f})'.rjust(16))
        print(f'{c:28s}{n:6d}' + ''.join(cells))


if __name__ == '__main__':
    main()
