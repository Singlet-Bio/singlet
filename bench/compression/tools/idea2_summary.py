"""Idea 2 final tables.

usage: python -P tools/idea2_summary.py [static_tag] [adaptive_tag]
Reads results/idea2_m_breakdown.tsv (one row per bench run, tagged), results/idea2_m_final2.log (decode CPU traces),
Track D references, and prints markdown tables.
"""
import csv, re, sys, collections

def load(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))

def fl(x):
    try: return float(x)
    except Exception: return float('nan')

def main(argv):
    stag = argv[0] if argv else 'static_final'
    atag = argv[1] if len(argv) > 1 else 'k8'
    rows = load('results/idea2_m_breakdown.tsv')
    td = collections.defaultdict(dict)
    for r in load('results/track_d_entropy.tsv'):
        td[(r['sample'], r['member'], r['subset'])][r['quantity']] = fl(r['bits_per_nnz'])
    zs = {}
    for r in load('results/track_d_split.tsv'):
        zs[(r['sample'], r['member'], r['subset'], r['level'])] = (fl(r['chunk_bytes']), fl(r['nnz']))
    # decode CPU traces
    traces = {}
    cur = {}
    try:
        for line in open('results/idea2_m_final2.log'):
            m = re.match(r'\s*\[i2\] dec (headers|z coded|fit|model init|cells|empties) @cpu (\d+) ms', line)
            if m: cur[m.group(1)] = float(m.group(2)); continue
            m = re.match(r'== corpus/(\w+)/(\w+)_counts', line)
            if m: key = (m.group(1), m.group(2)); traces[key] = dict(cur); cur = {}; continue
            m = re.match(r'\s*\[idea2\] decode CPU (\d+) ms', line)
            if m and key in traces: traces[key]['total'] = float(m.group(1))
    except FileNotFoundError:
        pass

    def pick(tag):
        d = {}
        for r in rows:
            if r['tag'] == tag:
                d[(r['sample'], r['member'].split('_')[0])] = r
        return d
    S, A = pick(stag), pick(atag)

    def cells_bits(r):
        return (fl(r['hk_cells_bits']) + fl(r['flag_bits']) + fl(r['label_bits']) + fl(r['nnz_excess_bits']) + fl(r['side_z_bits'])
                + fl(r['cell_top_bits']) + fl(r['cell_tail_bits']) + fl(r['cell_val_bits']) + fl(r['perm_bits']))

    def empt_data(r): return fl(r['empt_set_bits']) + fl(r['empt_val_bits']) + fl(r['empt_lab_bits'])

    print('### Called cells (payload bits/nnz, side information included)\n')
    print('| file | nnz | static (this) | of which z_f | zstd-3 chunks | Track D static CV (no side info) | adaptive K=8 | delta | top/tail/val |')
    print('|---|---|---|---|---|---|---|---|---|')
    for key in sorted(S):
        r = S[key]; nc = fl(r['nnz_cells'])
        c = cells_bits(r) / nc
        z3 = zs.get(key + ('cells', '3'), (float('nan'), 1))
        m2 = td.get(key + ('cells',), {}).get('M2_featdepth_cv_total', float('nan'))
        a = cells_bits(A[key]) / nc if key in A else float('nan')
        print('| %s %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %+.3f | %.2f/%.2f/%.2f |' % (
            key[0], key[1], nc, c, fl(r['side_z_bits']) / nc, z3[0] * 8 / z3[1], m2, a, a - c,
            fl(r['cell_top_bits']) / nc, fl(r['cell_tail_bits']) / nc, fl(r['cell_val_bits']) / nc))

    print('\n### Empty droplets (bits/nnz of empties)\n')
    print('| file | nnz | data | + k_j | zstd-3 | MC floor (p*) | MLE oracle | LOO | data/LOO | data/MC |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for key in sorted(S):
        r = S[key]; ne = fl(r['nnz_empt'])
        d = empt_data(r) / ne; dk = (empt_data(r) + fl(r['hk_empt_bits'])) / ne
        z3 = zs.get(key + ('empties', '3'), (float('nan'), 1))
        t = td.get(key + ('empties',), {})
        print('| %s %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %+.1f%% | %+.1f%% |' % (
            key[0], key[1], ne, d, dk, z3[0] * 8 / z3[1], t.get('amb_entropy_mc_bits', float('nan')),
            t.get('amb_entropy_mc_mle_bits', float('nan')), t.get('amb_loo_bits', float('nan')),
            100 * (d / t.get('amb_loo_bits', float('nan')) - 1), 100 * (d / t.get('amb_entropy_mc_bits', float('nan')) - 1)))

    print('\n### Side information and decode speed (single thread, CPU time on a saturated shared machine)\n')
    print('| file | split+k_j bytes | nnz-excess B | z_f B | payload B | cells Mnnz/s | empties Mnnz/s | whole file Mnnz/s |')
    print('|---|---|---|---|---|---|---|---|')
    for key in sorted(S):
        r = S[key]
        tr = traces.get(key, {})
        nc, ne = fl(r['nnz_cells']), fl(r['nnz_empt'])
        cs = (tr.get('cells', float('nan')) - tr.get('model init', float('nan')))
        es = (tr.get('empties', float('nan')) - tr.get('cells', float('nan')))
        tot = tr.get('total', float('nan'))
        print('| %s %s | %.0f | %.0f | %.0f | %.0f | %.2f | %.2f | %.2f |' % (
            key[0], key[1], (fl(r['hdr_k_bits']) + fl(r['flag_bits']) + fl(r['label_bits'])) / 8, fl(r['nnz_excess_bits']) / 8,
            fl(r['side_z_bits']) / 8, fl(r['payload_bytes']), nc / cs / 1e3, ne / es / 1e3, (nc + ne) / tot / 1e3))

if __name__ == '__main__':
    main(sys.argv[1:])
