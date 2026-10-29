"""Idea 2 report: join results/idea2_m_breakdown.tsv (my runs) with Track D references.

usage: python -P tools/idea2_report.py [tag-filter ...]
Prints, per (sample, member, tag): bits/nnz for called cells (payload incl. side info) and empties, next to
  * zstd-3 chunk bytes for the same column subsets (results/track_d_split.tsv)
  * Track D static per-feature x depth CV estimate (M2_featdepth_cv_total, no parameter cost) for cells
  * Track D multinomial references for empties: amb_entropy_mc (smoothed p*), amb_entropy_mc_mle (oracle MLE),
    amb_loo (leave-one-droplet-out), plus order-0 cost of k_j.
"""
import csv, sys, collections

def load(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))

def main(argv):
    rows = load('results/idea2_m_breakdown.tsv')
    td = collections.defaultdict(dict)
    for r in load('results/track_d_entropy.tsv'):
        td[(r['sample'], r['member'], r['subset'])][r['quantity']] = float(r['bits_per_nnz'])
    zs = {}
    for r in load('results/track_d_split.tsv'):
        zs[(r['sample'], r['member'], r['subset'], r['level'])] = (float(r['chunk_bytes']), float(r['nnz']))
    tags = set(argv)
    print('sample\tmember\tcodec\ttag\tcells_b/nnz\t(top,tail,val,side_z,hdr)\tcells_zstd3\tcells_trackD_M2cv\tempt_b/nnz(data)\tempt_b/nnz(+k)\tempt_zstd3\tMC_floor\tMC_mle\tLOO')
    for r in rows:
        if tags and r['tag'] not in tags:
            continue
        member = r['member'].split('_')[0]
        nc, ne = float(r['nnz_cells']), float(r['nnz_empt'])
        def b(k): return float(r[k + '_bits'])
        cells_bits = b('hdr_k') * 0 + float(r['hk_cells_bits']) + b('flag') + b('label') + b('nnz_excess') + b('side_z') + b('cell_top') + b('cell_tail') + b('cell_val') + b('perm')
        empt_data = b('empt_set') + b('empt_val') + b('empt_lab')
        empt_all = empt_data + float(r['hk_empt_bits'])
        key = (r['sample'], member)
        z3c = zs.get(key + ('cells', '3'), (0, 1)); z3e = zs.get(key + ('empties', '3'), (0, 1))
        tdc = td.get(key + ('cells',), {}); tde = td.get(key + ('empties',), {})
        print('\t'.join(str(x) for x in [
            r['sample'], member, r['codec'], r['tag'],
            '%.3f' % (cells_bits / nc),
            '%.2f/%.2f/%.2f/%.2f/%.2f' % (b('cell_top') / nc, b('cell_tail') / nc, b('cell_val') / nc, b('side_z') / nc, (float(r['hk_cells_bits']) + b('nnz_excess') + b('flag') + b('label') + b('perm')) / nc),
            '%.3f' % (z3c[0] * 8 / z3c[1]), '%.3f' % tdc.get('M2_featdepth_cv_total', float('nan')),
            '%.3f' % (empt_data / max(ne, 1)), '%.3f' % (empt_all / max(ne, 1)), '%.3f' % (z3e[0] * 8 / z3e[1]),
            '%.3f' % tde.get('amb_entropy_mc_bits', float('nan')), '%.3f' % tde.get('amb_entropy_mc_mle_bits', float('nan')),
            '%.3f' % tde.get('amb_loo_bits', float('nan'))]))

if __name__ == '__main__':
    main(sys.argv[1:])
