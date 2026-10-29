# Compare Track B codec payloads with today's production .1pz payload (chunks + pointer block + chunk table), per file.
#   python -P tools/track_b_vs_today.py results/track_b_codecs.tsv [codec ...]
import sys
import csv
import glob
import os
import struct

HDR = "<IHBBIIQBBHIIIIIQIIQII"


def today(path):
    sz = os.path.getsize(path)
    with open(path, 'rb') as f:
        hdr = f.read(96)
    (magic, ver, vt, flags, m, n, nnz, pw, lvl, _pad, nch, perm_z, ptr_z, chunk_cols, ff, meta_off, meta_z, colsums_z, t_off, t_z, t_ch) = struct.unpack(HDR, hdr[:80])
    chunks_z = sz - 96 - perm_z - ptr_z - meta_z - colsums_z - 4 * nch - 16
    return dict(nnz=nnz, payload=chunks_z + ptr_z + 4 * nch, chunks=chunks_z, perm=perm_z, meta=meta_z, total=sz)


def main():
    tsv = sys.argv[1]
    want = sys.argv[2:] or ['b_rans16']
    T = {}
    for p in glob.glob('corpus/*/*_counts.1pz'):
        T[(os.path.basename(os.path.dirname(p)), os.path.basename(p))] = today(p)
    rows = list(csv.DictReader(open(tsv), delimiter='\t'))
    print('| file | nnz | today payload b/nnz | ' + ' | '.join(f'{w} payload b/nnz (vs today)' for w in want) + ' |')
    print('|---|---|---|' + '---|' * len(want))
    sums = {w: [0.0, 0.0] for w in want}
    for key in sorted(T):
        t = T[key]
        cells = []
        for w in want:
            r = [x for x in rows if (x['sample'], x['member']) == key and x['codec'] == w]
            if not r:
                cells.append('-')
                continue
            b = float(r[0]['payload_bits_nnz'])
            cells.append(f"{b:.2f} ({100 * (b * t['nnz'] / 8 / t['payload'] - 1):+.0f}%)")
            sums[w][0] += b * t['nnz'] / 8
            sums[w][1] += t['payload']
        print(f"| {key[0]}/{key[1][:6]} | {t['nnz']} | {t['payload'] * 8.0 / t['nnz']:.2f} | " + ' | '.join(cells) + ' |')
    print('| all (byte-weighted) | | | ' + ' | '.join(f'{100 * (sums[w][0] / sums[w][1] - 1):+.1f}%' if sums[w][1] else '-' for w in want) + ' |')


if __name__ == '__main__':
    main()
