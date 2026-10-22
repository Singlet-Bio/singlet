#!/usr/bin/env python
"""Render the Track D result TSVs as markdown tables (stdout) for notes/track_d.md.

    python -P tools/track_d_tables.py [entropy|ambient|gof|cells|recursion|floor|all]
"""
import csv, os, sys, collections, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from track_d_common import ROOT

ORDER = ['pbmc_v2_typical', 'pbmc_v2_shallow', 'pbmc_v3_lowgenes', 'brain_sn_v3_deep', 'brain_sn_v3_intronic', 'tcell_5p', 'tumor_3p']


def rd(name):
    p = os.path.join(ROOT, 'results', name)
    return list(csv.DictReader(open(p), delimiter='\t')) if os.path.exists(p) else []


def files():
    return [(s, m) for s in ORDER for m in ('exon', 'intron')]


def md(hdr, rows):
    print('| ' + ' | '.join(hdr) + ' |')
    print('|' + '|'.join('---' for _ in hdr) + '|')
    for r in rows:
        print('| ' + ' | '.join(str(x) for x in r) + ' |')
    print()


def f2(x, nd=2):
    return '' if x is None or (isinstance(x, float) and x != x) else f'{x:.{nd}f}'


def entropy():
    E = collections.defaultdict(dict)
    for r in rd('track_d_entropy.tsv'):
        E[(r['sample'], r['member'], r['subset'])][r['quantity']] = (float(r['bits']), int(r['nnz']))
    S = {}
    for r in rd('track_d_split.tsv'):
        S[(r['sample'], r['member'], r['subset'], r['level'])] = int(r['chunk_bytes'])
    print('### Order-0 and combinatorial reference points, all columns (bits/nnz)\n')
    hdr = ['file', 'nnz', 'today z3', 'z19', 'comb C(m_eff,nnz_j)', 'H0 writer streams', 'H0 +MDL', 'pres feat (plug-in)', 'pres feat x depth (plug-in)', 'M1 2-fold CV', 'M2 2-fold CV']
    rows = []
    for s, m in files():
        q = E.get((s, m, 'all'))
        if not q:
            continue
        nnz = q['comb_presence_m'][1]
        g = lambda k: f2(q[k][0] / nnz) if k in q else ''
        rows.append([f'{s} {m}', nnz, f2(S[(s, m, 'all', '3')] * 8 / nnz), f2(S[(s, m, 'all', '19')] * 8 / nnz),
                     f2((q['comb_presence_meff'][0] + q['comb_composition'][0]) / nnz), g('w_H0_total'), g('w_H0_total_mdl'),
                     g('pres_feat_plugin'), g('pres_featdepth_plugin'), g('M1_feat_cv_total'), g('M2_featdepth_cv_total')])
    md(hdr, rows)
    print('### Today\'s chunk payload split by column class (re-encoded with the production writer, zstd-3)\n')
    hdr = ['file', 'cells nnz', 'cells bytes', 'cells bits/nnz', 'empties nnz', 'empties bytes', 'empties bits/nnz', 'empties share of payload']
    rows = []
    for s, m in files():
        if (s, m, 'cells', '3') not in S:
            continue
        qc = E[(s, m, 'cells')]['comb_presence_m'][1]; qe = E[(s, m, 'empties')]['comb_presence_m'][1]
        cb, eb = S[(s, m, 'cells', '3')], S[(s, m, 'empties', '3')]
        rows.append([f'{s} {m}', qc, cb, f2(cb * 8 / qc), qe, eb, f2(eb * 8 / qe), f'{100 * eb / (cb + eb):.0f}%'])
    md(hdr, rows)


def ambient():
    E = collections.defaultdict(dict)
    for r in rd('track_d_entropy.tsv'):
        E[(r['sample'], r['member'], r['subset'])][r['quantity']] = (float(r['bits']), int(r['nnz']), r['note'])
    S = {}
    for r in rd('track_d_split.tsv'):
        S[(r['sample'], r['member'], r['subset'], r['level'])] = int(r['chunk_bytes'])
    A = collections.defaultdict(dict)
    for r in rd('track_d_asymptote.tsv'):
        A[(r['sample'], r['member'])][r['model']] = float(r['L_at_max'])
    print('### Empty droplets: multinomial ambient floor versus today (bits/nnz of the empties)\n')
    hdr = ['file', 'empties (nnz)', 'today z3', 'H0 writer', 'comb', 'Mult plug-in (in-sample p)', 'Mult LOO (held-out p)', 'E[H(Mult)] MC', 'held-out mixture C=1 / 4 / 16', 'k_j side (bits/nnz)']
    rows = []
    for s, m in files():
        q = E.get((s, m, 'empties'))
        if not q or 'amb_loo_bits' not in q:
            continue
        nnz = q['amb_loo_bits'][1]
        g = lambda k: f2(q[k][0] / nnz)
        a = A.get((s, m), {})
        mix = ' / '.join(f2(a[k]) for k in ('E_mult1', 'E_mult4', 'E_mult16') if k in a)
        rows.append([f'{s} {m}', nnz, f2(S[(s, m, 'empties', '3')] * 8 / nnz), g('w_H0_total'), f2((q['comb_presence_meff'][0] + q['comb_composition'][0]) / nnz),
                     g('amb_plugin_unsmoothed_bits'), g('amb_loo_bits'), g('amb_entropy_mc_mle_bits'), mix, g('amb_side_H0_kj')])
    md(hdr, rows)


def gof():
    G = collections.defaultdict(dict)
    for r in rd('track_d_gof.tsv'):
        G[(r['sample'], r['member'])][(r['test'], r['bucket'])] = (float(r['real']), float(r['synth_mean']), float(r['synth_sd']), float(r['z']), int(r['n_droplets']))
    print('### Goodness of fit of empty droplets to one pooled multinomial (cross-fitted, features with >= 10 ambient UMIs in the fitting half; synthetic = parametric bootstrap, 5 draws)\n')
    hdr = ['file', 'droplets', 'UMI coverage of W', 'Pearson dispersion real / synth (z)', 'distinct features per UMI real / synth (z)', 'repeat features per UMI real / synth (z)', 'LOO code length real / synth bits per droplet (z)']
    rows = []
    for s, m in files():
        g = G.get((s, m))
        if not g:
            continue
        def cell(t, b='all'):
            v = g.get((t, b))
            return '' if v is None else f'{v[0]:.3f} / {v[1]:.3f} ({v[3]:+.0f})'
        cov = g.get(('W_umi_coverage', 'all'))
        rows.append([f'{s} {m}', g[('W_cost_bits_per_droplet', 'all')][4], f2(cov[0] if cov else float('nan')), cell('W_pearson_dispersion'), cell('W_distinct_per_umi'), cell('W_repeat_per_umi'), cell('W_cost_bits_per_droplet')])
    md(hdr, rows)
    print('### Same tests by droplet size k (Pearson dispersion real / synth): where the heterogeneity lives\n')
    buckets = ['k1-1', 'k2-3', 'k4-7', 'k8-15', 'k16-31', 'k32-63', 'k64-127', 'k128-255', 'k256-inf']
    hdr = ['file'] + buckets
    rows = []
    for s, m in files():
        g = G.get((s, m))
        if not g:
            continue
        row = [f'{s} {m}']
        for b in buckets:
            v = g.get(('W_pearson_dispersion', b))
            row.append('' if v is None else f'{v[0]:.2f} (n={v[4]})')
        rows.append(row)
    md(hdr, rows)


def cells():
    C = collections.defaultdict(dict)
    meta = {}
    for r in rd('track_d_curve.tsv'):
        if r['model'].startswith('E_'):
            continue
        C[(r['sample'], r['member'])][(r['model'], int(r['n_train_cells']))] = float(r['bits_per_nnz'])
        meta[(r['sample'], r['member'])] = (int(r['n_test_cells']), int(r['nnz_test']))
    A = collections.defaultdict(dict)
    for r in rd('track_d_asymptote.tsv'):
        A[(r['sample'], r['member'])][r['model']] = r
    S = {}
    for r in rd('track_d_split.tsv'):
        S[(r['sample'], r['member'], r['subset'], r['level'])] = (int(r['chunk_bytes']), int(r['nnz']))
    print('### Called cells: held-out cross-entropy (bits/nnz of the test cells) at the largest training size, and extrapolated asymptotes\n')
    hdr = ['file', 'test cells / nnz', 'today z3 (cells)', 'comb', 'M1 feat', 'M2 feat x depth', 'mult1 (pooled profile)', 'dm1 (Dirichlet-mult)', 'mult4', 'mult16', 'mult64', 'best', 'Linf (b=0.5)', 'Linf free [95% boot]', 'b free']
    rows = []
    for s, m in files():
        c = C.get((s, m))
        if not c:
            continue
        nmax = max(n for (_, n) in c)
        def at(model):
            ks = [n for (mm, n) in c if mm == model and n > 0]
            return f2(c[(model, max(ks))]) if ks else ''
        a = A.get((s, m), {})
        best = min(a, key=lambda k: float(a[k]['L_at_max'])) if a else ''
        ab = a.get(best)
        rows.append([f'{s} {m}', f'{meta[(s, m)][0]} / {meta[(s, m)][1]}', f2(S[(s, m, 'cells', '3')][0] * 8 / S[(s, m, 'cells', '3')][1]),
                     f2(c.get(('comb', 0))), at('M1'), at('M2'), at('mult1'), at('dm1'), at('mult4'), at('mult16'), at('mult64'),
                     best, f2(float(ab['Linf_b05'])) if ab else '', f'{float(ab["Linf_free"]):.2f} [{float(ab["Linf_lo95"]):.2f}, {float(ab["Linf_hi95"]):.2f}]' if ab else '', f2(float(ab['b_free'])) if ab else ''])
    md(hdr, rows)
    print('### Diminishing returns: mult1 cross-entropy vs number of training cells (bits/nnz)\n')
    sizes_hdr = ['1/32', '1/16', '1/8', '1/4', '1/2', 'all']
    hdr = ['file'] + [f'pool x {x}' for x in sizes_hdr]
    rows = []
    for s, m in files():
        c = C.get((s, m))
        if not c:
            continue
        ks = sorted(n for (mm, n) in c if mm == 'mult1' and n > 0)
        row = [f'{s} {m}'] + [''] * (6 - len(ks)) + [f'{c[("mult1", n)]:.3f} (n={n})' for n in ks]
        rows.append(row)
    md(hdr, rows)


def recursion():
    rows_ = rd('track_d_recursion.tsv')
    if not rows_:
        print('(no recursion rows yet)\n'); return
    agg = collections.defaultdict(list)
    for r in rows_:
        agg[(r['codec'], r['method'])].append((int(r['first_bytes']), int(r['second_bytes']), float(r['ratio'])))
    methods = ['zstd22', 'xz', 'brotli', 'bz2', 'cm']
    codecs = sorted({k[0] for k in agg})
    print('### Second-pass ratio (bytes out / bytes in) of every codec output; aggregate over files, then min-max over files\n')
    hdr = ['codec', 'files'] + methods
    rows = []
    for c in codecs:
        row = [c, max(len(agg.get((c, m), [])) for m in methods)]
        for m in methods:
            v = agg.get((c, m))
            if not v:
                row.append('')
            else:
                ti = sum(x[0] for x in v); to = sum(x[1] for x in v); rs = [x[2] for x in v]
                row.append(f'{to / ti:.3f} ({min(rs):.3f}-{max(rs):.3f})')
        rows.append(row)
    md(hdr, rows)


def floor():
    rows_ = rd('track_d_floor.tsv')
    if not rows_:
        print('(no floor rows yet)\n'); return
    print('### Floor estimate per file (bytes) versus today\n')
    hdr = ['file', 'file bytes', 'names+perm+ptr+colsums', 'payload today', 'cells today bpn -> floor warm / central / low', 'empties today bpn -> floor LOO / plug-in', 'k_j side bytes', 'barcode identity bytes', 'payload floor (central)', 'payload excess', 'file floor central [lo, hi]', 'today / floor', 'irreducible % of file']
    rows = []
    for r in rows_:
        rows.append([f"{r['sample']} {r['member']}", r['file_bytes'], int(r['names_meta_z']) + int(r['perm_z']) + int(r['ptr_z']) + int(r['colsums_z']), r['chunk_bytes'],
                     f"{float(r['cells_today_bpn']):.2f} -> {float(r['cells_floor_warm_bpn']):.2f} / {float(r['cells_floor_central_bpn']):.2f} / {float(r['cells_floor_lo_bpn']):.2f}",
                     f"{float(r['empties_today_bpn']):.2f} -> {float(r['empties_floor_loo_bpn']):.2f} / {float(r['empties_floor_plugin_bpn']):.2f}",
                     f"{float(r['side_kj_cells_bytes']) + float(r['side_kj_empties_bytes']):.0f}", f"{float(r['barcode_identity_bytes']):.0f}",
                     f"{float(r['payload_floor_central_bytes']):.0f}", f"{float(r['payload_excess_pct']):.0f}%",
                     f"{float(r['floor_total_central_bytes']):.0f} [{float(r['floor_total_lo_bytes']):.0f}, {float(r['floor_total_hi_bytes']):.0f}]",
                     f"{float(r['today_over_floor_x']):.1f}x", f"{float(r['irreducible_pct_of_file']):.0f}%"])
    md(hdr, rows)


def oracle():
    for fn, title in (('track_d_oracle.tsv', 'entropy coders of one transform (9 coders)'),
                      ('track_d_oracle_with_models.tsv', 'coders + 4 ideal model code lengths (M2, mult1, mult16, amb; 2-fold CV)')):
        rows_ = rd(fn)
        if not rows_:
            continue
        print(f'### Orchestrator oracle, per 1024-column chunk, {title}' + chr(10))
        hdr = ['file', 'chunks', 'today z3 (bytes)', 'best single (whole file)', 'per-chunk oracle', 'oracle gap', 'signal bits/chunk', 'ridge chooser: share of gap captured / vs best single']
        rows = []
        tz = ts = to = 0
        for r in rows_:
            z3 = int(float(r['bytes_z3(today)'])); sb = int(float(r['bytes_best_single'])); orc = int(float(r['bytes_oracle']))
            tz += z3; ts += sb; to += orc
            rows.append([f"{r['sample']} {r['member']}", r['nchunks'], f'{z3:,}', f"{r['best_single']} {sb:,}", f'{orc:,}', f"{float(r['oracle_gap_pct']):.3f}%",
                         f"{float(r['sig_bits_per_chunk_entropy']):.2f}", f"{float(r['ridge_captured_pct']):.0f}% / {float(r['ridge_vs_best_single_pct']):+.2f}%"])
        rows.append(['**all 14**', '', f'{tz:,}', f'{ts:,} ({100 * (ts - tz) / tz:+.1f}% vs today)', f'{to:,}', f'{100 * (ts - to) / ts:.2f}%', '', ''])
        md(hdr, rows)


def main():
    what = sys.argv[1:] or ['all']
    for w in what:
        if w in ('entropy', 'all'): entropy()
        if w in ('ambient', 'all'): ambient()
        if w in ('gof', 'all'): gof()
        if w in ('cells', 'all'): cells()
        if w in ('recursion', 'all'): recursion()
        if w in ('floor', 'all'): floor()
        if w in ('oracle', 'all'): oracle()


if __name__ == '__main__':
    main()
