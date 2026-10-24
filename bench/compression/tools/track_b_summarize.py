# Summaries of Track B result TSVs as markdown tables.
#   python -P tools/track_b_summarize.py index results/track_b_index.tsv
#   python -P tools/track_b_summarize.py codecs results/track_b_codecs.tsv
import sys
import csv
import statistics as st
from collections import defaultdict


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter='\t'))


def med(x):
    return st.median(x) if x else float('nan')


def index_summary(path):
    R = rows(path)
    g = defaultdict(list)
    files = set()
    for r in R:
        if r['ok'] != '1':
            print('!! failed row', r)
        g[(r['space'], r['coder'], r['stage'])].append(r)
        files.add(r['sample'])
    print(f'{len(files)} files; mean idx bits/nnz (mean over files of bits/nnz), median decode Mnnz/s and cycles/nnz\n')
    print('| space | coder | raw bits | +zstd3 bits | +zstd19 bits | raw Mnnz/s | raw cyc/nnz | zstd3 Mnnz/s | zstd19 Mnnz/s |')
    print('|---|---|---|---|---|---|---|---|---|')
    keys = sorted({(s, c) for (s, c, _) in g})
    for s, c in sorted(keys, key=lambda k: (k[0] != 'rank', k[1])):
        def b(stage):
            x = g.get((s, c, stage), [])
            return st.mean(float(r['idx_bits_nnz']) for r in x) if x else float('nan')

        def sp(stage):
            x = g.get((s, c, stage), [])
            return med([float(r['dec_Mnnz_s']) for r in x])

        def cy(stage):
            x = g.get((s, c, stage), [])
            return med([float(r['cyc_per_nnz']) for r in x])

        print(f'| {s} | {c} | {b("raw"):.2f} | {b("zstd3"):.2f} | {b("zstd19"):.2f} | {sp("raw"):.0f} | {cy("raw"):.1f} | {sp("zstd3"):.0f} | {sp("zstd19"):.0f} |')


def codec_summary(path):
    R = rows(path)
    by = defaultdict(list)
    for r in R:
        by[r['codec']].append(r)
    print('| codec | files | all ok | mean payload bits/nnz | median payload Mnnz/s canon | median native | median est. core-3GHz canon | native |')
    print('|---|---|---|---|---|---|---|---|')
    for c, x in by.items():
        ok = all(r['matrix_ok'] == '1' for r in x)
        pb = st.mean(float(r['payload_bits_nnz']) for r in x)
        cn = med([float(r['payload_Mnnz_s_canon']) for r in x])
        na = med([float(r['payload_Mnnz_s_native']) for r in x])
        ec = med([float(r.get('est_Mnnz_s_canon_core3GHz', 0) or 0) for r in x])
        en = med([float(r.get('est_Mnnz_s_native_core3GHz', 0) or 0) for r in x])
        print(f'| {c} | {len(x)} | {ok} | {pb:.2f} | {cn:.1f} | {na:.1f} | {ec:.0f} | {en:.0f} |')


if __name__ == '__main__':
    {'index': index_summary, 'codecs': codec_summary}[sys.argv[1]](sys.argv[2])
