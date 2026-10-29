# Frontier data file for Track B: (payload bits/nnz, decode Mnnz/s) per codec, averaged over the corpus files each codec
# was measured on, plus the index-only survey rows. Writes results/track_b_frontier.tsv and prints markdown + ASCII scatter.
#   python -P tools/track_b_frontier.py results/track_b_codecs.tsv [results/track_b_variants.tsv ...] --index results/track_b_index.tsv
import sys
import csv
import math
import statistics as st
from collections import defaultdict


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter='\t'))


def fnum(r, k, default=float('nan')):
    try:
        v = r.get(k, '')
        return float(v) if v not in ('', None) else default
    except ValueError:
        return default


def main():
    args = sys.argv[1:]
    index = None
    if '--index' in args:
        i = args.index('--index')
        index = args[i + 1]
        del args[i:i + 2]
    out = []
    by = defaultdict(list)
    for p in args:
        for r in rows(p):
            by[r['codec']].append(r)
    for c, x in sorted(by.items()):
        files = {(r['sample'], r['member']) for r in x}
        # exclude failed rows
        good = [r for r in x if r['matrix_ok'] == '1']
        bits = st.mean(fnum(r, 'payload_bits_nnz') for r in good)
        tot = st.mean(fnum(r, 'total_bits_nnz') for r in good)
        canon = st.median(fnum(r, 'payload_Mnnz_s_canon') for r in good)
        nat = st.median(fnum(r, 'payload_Mnnz_s_native') for r in good)
        ec = st.median([fnum(r, 'est_Mnnz_s_canon_core3GHz') for r in good if not math.isnan(fnum(r, 'est_Mnnz_s_canon_core3GHz'))] or [float('nan')])
        en = st.median([fnum(r, 'est_Mnnz_s_native_core3GHz') for r in good if not math.isnan(fnum(r, 'est_Mnnz_s_native_core3GHz'))] or [float('nan')])
        c3 = st.median(fnum(r, 'Mnnz_s_canon_3GHz') for r in good)
        n3 = st.median(fnum(r, 'Mnnz_s_native_3GHz') for r in good)
        out.append(dict(kind='codec', name=c, files=len(files), ok=len(good) == len(x), payload_bits=bits, total_bits=tot,
                        canon_wall=canon, native_wall=nat, canon_thread_tsc=c3, native_thread_tsc=n3, canon_core3=ec, native_core3=en))
    if index:
        g = defaultdict(list)
        for r in rows(index):
            g[(r['space'], r['coder'], r['stage'])].append(r)
        for (sp, cd, stg), x in sorted(g.items()):
            if sp != 'rank':
                continue
            out.append(dict(kind='index-only', name=f'{cd}+{stg}', files=len(x), ok=all(r['ok'] == '1' for r in x),
                            payload_bits=st.mean(float(r['idx_bits_nnz']) for r in x), total_bits=float('nan'),
                            canon_wall=float('nan'), native_wall=st.median(float(r['dec_Mnnz_s']) for r in x),
                            canon_thread_tsc=float('nan'), native_thread_tsc=float('nan'), canon_core3=float('nan'),
                            native_core3=float('nan')))   # index survey calibration (taken after the run) is too noisy to normalise; wall clock only
    cols = ['kind', 'name', 'files', 'ok', 'payload_bits', 'total_bits', 'canon_wall', 'native_wall', 'canon_thread_tsc', 'native_thread_tsc', 'canon_core3', 'native_core3']
    with open('results/track_b_frontier.tsv', 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in out:
            f.write('\t'.join(f'{r[c]:.3f}' if isinstance(r[c], float) else str(r[c]) for c in cols) + '\n')
    print('| kind | name | files | ok | payload bits/nnz | canon Mnnz/s (wall) | native Mnnz/s (wall) | canon @core3GHz | native @core3GHz |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in out:
        print(f"| {r['kind']} | {r['name']} | {r['files']} | {r['ok']} | {r['payload_bits']:.2f} | {r['canon_wall']:.1f} | {r['native_wall']:.1f} | {r['canon_core3']:.0f} | {r['native_core3']:.0f} |")


if __name__ == '__main__':
    main()
