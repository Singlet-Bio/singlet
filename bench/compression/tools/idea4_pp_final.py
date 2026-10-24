"""Final summary tables for notes/idea4_pp.md from results/idea4_pp_run.tsv (+ Track C comparison).

python -P tools/idea4_pp_final.py > results/idea4_pp_final.md
"""
import csv, re, sys, collections

rows = []
NORD = {}
for path, mode in (('results/idea4_pp_run.tsv', 'k0only'), ('results/idea4_pp_run2a.tsv', 'all'), ('results/idea4_pp_run2b.tsv', 'all')):
    try:
        fh = open(path)
    except OSError:
        continue
    for r in csv.DictReader(fh, delimiter='	'):
        m = re.match(r'idea4_(static|nmf)_(\w+?)_k(\d+)_a(\d)$', r['codec'])
        if not (m and r.get('model_bytes') is not None):
            continue
        r['fold'], r['k'], r['a'] = m.group(2), int(m.group(3)), int(m.group(4))
        if mode == 'k0only' and r['k'] != 0:
            NORD[(r['sample'], r['fold'], r['k'], r['a'])] = r      # first-version encoder (no rate-distortion theta choice)
            continue
        rows.append(r)
R = {}
for r in rows:
    R[(r['sample'], r['fold'], r['k'], r['a'])] = r          # last one wins

LOSO = collections.OrderedDict([
    ('pbmc_v2_typical', 'F142444'), ('pbmc_v2_shallow', 'F142444'), ('pbmc_v3_lowgenes', 'F244263'),
    ('tcell_5p', 'F152522'), ('tumor_3p', 'F145370'), ('brain_sn_v3_deep', 'M285936'), ('brain_sn_v3_intronic', 'M285936')])


def g(s, fold, k, a, col):
    r = R.get((s, fold, k, a))
    if r is None:
        return None
    try:
        return float(r[col])
    except Exception:
        return None


def fmt(v, d=3):
    return 'n/a' if v is None else ('%.' + str(d) + 'f') % v


# Track C c_r2 LOSO payload bits/nnz
tc = {}
try:
    for r in csv.DictReader(open('results/track_c_cv.tsv'), delimiter='\t'):
        if r['codec'] == 'c_r2' and r['member'] == 'exon_counts.1pz' and 'none' not in r['model_tag'] and 'ins' not in r['model_tag']:
            tc[r['sample']] = float(r['payload_bits_per_nnz'])
except Exception:
    pass

print('## T1. H4 test: called cells, bits per interval-nnz (payload incl. depth-scale and theta side information), static shipped base')
print('| sample | cells | nnz/cell | static K=0 | K=8 | K=16 | K=32 | best gain vs static | side B/cell (best K) | H4 (>= 1 bit)? |')
print('|---|---|---|---|---|---|---|---|---|---|')
for s, f in LOSO.items():
    st = g(s, f, 0, 0, 'bits_nnz_cells')
    if st is None:
        continue
    ks = {k: g(s, f, k, 0, 'bits_nnz_cells') for k in (8, 16, 32)}
    avail = {k: v for k, v in ks.items() if v is not None}
    best = min(avail, key=avail.get) if avail else None
    cells = g(s, f, 0, 0, 'cells'); nc = g(s, f, 0, 0, 'nnz_cells')
    gain = (st - avail[best]) if best else None
    side = g(s, f, best, 0, 'side_bytes_per_cell') if best else None
    print('| %s | %d | %.0f | %s | %s | %s | %s | %s | %s | %s |' % (
        s, cells or 0, (nc / cells) if cells else 0, fmt(st), fmt(ks[8]), fmt(ks[16]), fmt(ks[32]),
        ('%+.3f (K=%d)' % (gain, best)) if gain is not None else 'n/a', fmt(side, 2), 'no' if (gain is not None and gain < 1) else ('YES' if gain is not None else 'n/a')))
print()
print('## T2. Same with the decoder-adaptive base profile (gene profile + interval split re-estimated from decoded cells)')
print('| sample | static K=0 | K=16 | K=32 | gain of K=16 | gain of K=32 | static shipped (T1) | adaptive-base gain over shipped static |')
print('|---|---|---|---|---|---|---|---|')
for s, f in LOSO.items():
    a0 = g(s, f, 0, 1, 'bits_nnz_cells')
    if a0 is None:
        continue
    k16 = g(s, f, 16, 1, 'bits_nnz_cells'); k32 = g(s, f, 32, 1, 'bits_nnz_cells'); sh = g(s, f, 0, 0, 'bits_nnz_cells')
    print('| %s | %s | %s | %s | %s | %s | %s | %s |' % (s, fmt(a0), fmt(k16), fmt(k32),
          fmt(a0 - k16 if k16 else None), fmt(a0 - k32 if k32 else None), fmt(sh), fmt(sh - a0 if sh else None)))
print()
print('## T3. Whole file (all columns), payload bits per nnz = rANS bytes * 8 / nnz (names excluded), vs Track C c_r2 (LOSO)')
print('| sample | static shipped | best K shipped | static adaptive-base | best K + adaptive-base | Track C c_r2 | empties b/nnz (K=0) |')
print('|---|---|---|---|---|---|---|')
for s, f in LOSO.items():
    p00 = g(s, f, 0, 0, 'payload_bits_per_nnz')
    if p00 is None:
        continue
    pk = [g(s, f, k, 0, 'payload_bits_per_nnz') for k in (8, 16, 32)]; pk = [v for v in pk if v is not None]
    p01 = g(s, f, 0, 1, 'payload_bits_per_nnz')
    pka = [g(s, f, k, 1, 'payload_bits_per_nnz') for k in (16, 32)]; pka = [v for v in pka if v is not None]
    print('| %s | %s | %s | %s | %s | %s | %s |' % (s, fmt(p00), fmt(min(pk) if pk else None), fmt(p01), fmt(min(pka) if pka else None), fmt(tc.get(s)), fmt(g(s, f, 0, 0, 'bits_nnz_amb'))))
print()
print('## T4. Decode cost and model size (CPU time, machine shared with ~7 agents; min of reps)')
print('| sample | K | adapt | model MB (+names blob) | matvec/build ms per called cell | decode Mnnz/s (CPU) | enc ms | matrix_ok |')
print('|---|---|---|---|---|---|---|---|')
for (s, f, k, a), r in sorted(R.items()):
    if f in LOSO.values() and LOSO.get(s) == f:
        print('| %s | %d | %d | %.2f | %s | %s | %s | %s |' % (s, k, a, float(r['model_bytes']) / 1048576, fmt(float(r['matvec_ms_per_cell']), 3), fmt(float(r['dec_Mnnz_s']), 2), fmt(float(r['enc_ms']), 0), r['matrix_ok']))
print()
print('## T5. In-sample upper bound (model also trained on the test sample) and cross-domain')
print('| sample | fold | K | cells b/nnz | side B/cell |')
print('|---|---|---|---|---|')
for (s, f, k, a), r in sorted(R.items()):
    if f in ('IShuman', 'ISmouse', 'XH', 'XM'):
        print('| %s | %s | %d | %s | %s |' % (s, f, k, fmt(float(r['bits_nnz_cells'])), fmt(float(r['side_bytes_per_cell']), 2)))
