#!/usr/bin/env python
"""Track D synthesis: per-file floor estimate versus today's bytes.  Writes results/track_d_floor.tsv.

Pieces (all bytes):
  today            file size, chunk payload (VOCSC+zstd-3), chunk payload split cells / empties (re-encode, track_d_split.tsv)
  floor_empties    ambient multinomial leave-one-droplet-out code length + order-0 entropy of the per-droplet totals k_j
                   (incl. all-zero columns); plug-in (in-sample) value is the optimistic end; held-out mixture
                   (E_mult*) is reported if available (relaxes exchangeability)
  floor_cells      held-out cross-entropy of the best model at the largest training size (warm; upper end),
                   power-law extrapolations (b=0.5 central, free-b lower end) and the order-0 entropy of k_j
  barcodes         n_cols * log2(whitelist size): identity of the barcodes in column order (irreducible unless the
                   column order is canonical)
  removable        everything else: names (vocabulary id), row permutation, colsums, nnz pointers, payload excess
"""
import csv, math, os, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from track_d_common import ROOT
from pz_breakdown import breakdown
from track_d_common import load
from track_d_lib import h0_symbols, columns_of
import numpy as np

WL = {'pbmc_v2_typical': 737280, 'pbmc_v2_shallow': 737280, 'pbmc_v3_lowgenes': 3686400, 'brain_sn_v3_deep': 3686400,
      'brain_sn_v3_intronic': 3686400, 'tcell_5p': 737280, 'tumor_3p': 737280}


def rd(name):
    return list(csv.DictReader(open(os.path.join(ROOT, 'results', name)), delimiter='\t'))


def main():
    ent = collections.defaultdict(dict)
    for r in rd('track_d_entropy.tsv'):
        ent[(r['sample'], r['member'], r['subset'])][r['quantity']] = (float(r['bits']), int(r['nnz']))
    split = {}
    for r in rd('track_d_split.tsv'):
        split[(r['sample'], r['member'], r['subset'], r['level'])] = (int(r['chunk_bytes']), int(r['nnz']), int(r['ncols']))
    asym = collections.defaultdict(dict)
    for r in rd('track_d_asymptote.tsv'):
        asym[(r['sample'], r['member'])][r['model']] = {k: float(v) for k, v in r.items() if k not in ('sample', 'member', 'model', 'note')}
    nsizes = collections.defaultdict(set)
    for r in rd('track_d_curve.tsv'):
        nsizes[(r['sample'], r['member'], r['model'])].add(int(r['n_train_cells']))
    out = []
    hdr = ['sample', 'member', 'file_bytes', 'names_meta_z', 'perm_z', 'ptr_z', 'colsums_z', 'chunk_bytes', 'cells_chunk_bytes', 'empties_chunk_bytes',
           'cells_nnz', 'empties_nnz', 'cells_today_bpn', 'empties_today_bpn',
           'cells_floor_warm_bpn', 'cells_floor_central_bpn', 'cells_floor_lo_bpn', 'cells_best_model',
           'empties_floor_loo_bpn', 'empties_floor_plugin_bpn', 'empties_floor_heldout_mix_bpn',
           'side_kj_cells_bytes', 'side_kj_empties_bytes', 'barcode_identity_bytes',
           'floor_total_central_bytes', 'floor_total_lo_bytes', 'floor_total_hi_bytes',
           'payload_floor_central_bytes', 'payload_excess_pct', 'today_over_floor_x', 'irreducible_pct_of_file']
    out.append(hdr)
    for sample in WL:
        for member in ('exon', 'intron'):
            key = (sample, member)
            path = os.path.join(ROOT, 'corpus', sample, f'{member}_counts.1pz')
            if not os.path.exists(path) or key not in asym:
                continue
            b = breakdown(path)
            cb, cn, nc_cells = split[(sample, member, 'cells', '3')]
            eb, en, nc_emp = split[(sample, member, 'empties', '3')]
            # cells floor: best model across the ladder
            allm = asym[key]
            models = {k: v for k, v in allm.items() if not k.startswith('E_')}
            warm = min(v['L_at_max'] for v in models.values())
            bm = min(models, key=lambda k: models[k]['L_at_max'])
            central = min(min(v['Linf_b05'], v['L_at_max']) for v in models.values())
            # free-exponent lower end only from models whose curve has >= 5 training sizes (mult64 often has 3-4)
            lo = min(min(v['Linf_lo95'], v['Linf_free']) for k, v in models.items()
                     if v['Linf_free'] == v['Linf_free'] and len(nsizes[(sample, member, k)]) >= 5)
            lo = min(lo, central)
            e = ent[(sample, member, 'empties')]
            nz_e = e['amb_loo_bits'][1]
            loo = e['amb_loo_bits'][0]
            plug = e['amb_plugin_bits'][0]
            held = min([v['L_at_max'] for k, v in allm.items() if k.startswith('E_')] or [float('nan')])
            ce = ent[(sample, member, 'cells')]
            side_c = ce['side_H0_kj'][0] / 8
            d = load(sample, member)
            colx = columns_of(d)
            kcol = np.bincount(colx, weights=d['data'], minlength=int(d['n'])).astype(np.int64)
            side_e = h0_symbols(kcol[~d['is_cell']]) / 8
            bc = b['n'] * math.log2(WL[sample]) / 8
            f_c = lambda bpn: bpn * cn / 8
            floor_c = {'c': f_c(central), 'lo': f_c(lo), 'hi': f_c(warm)}
            cand = [loo, plug] + ([held * nz_e] if held == held else [])
            floor_e = {'c': min([loo] + ([held * nz_e] if held == held else [])) / 8, 'lo': min(cand) / 8, 'hi': loo / 8}
            pay = {k: floor_c[k] + floor_e[k] + side_c + side_e for k in floor_c}
            tot = {k: pay[k] + bc for k in pay}
            row = [sample, member, b['bytes'], b['meta_z'], b['perm_z'], b['ptr_z'], b['colsums_z'], b['chunks_z'], cb, eb, cn, en,
                   cb * 8 / cn, eb * 8 / en, warm, central, lo, bm, loo / nz_e, plug / nz_e, held,
                   side_c, side_e, bc, tot['c'], tot['lo'], tot['hi'], pay['c'], 100 * (b['chunks_z'] / pay['c'] - 1),
                   b['bytes'] / tot['c'], 100 * tot['c'] / b['bytes']]
            out.append(row)
    with open(os.path.join(ROOT, 'results', 'track_d_floor.tsv'), 'w') as f:
        for r in out:
            f.write('\t'.join(f'{v:.5g}' if isinstance(v, float) else str(v) for v in r) + '\n')
    for r in out:
        print('  '.join(f'{v:>9.3f}' if isinstance(v, float) else f'{str(v):>9s}' for v in r))


if __name__ == '__main__':
    main()
