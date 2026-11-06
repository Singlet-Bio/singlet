#!/usr/bin/env python3
"""Collect within.json per sample and cross.json into summary.md and summary.json."""
import sys, os, json, glob
import numpy as np

res = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else res
samples = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(res, '*', 'within.json')))
WI = {s: json.load(open(os.path.join(res, s, 'within.json'))) for s in samples}
CR = json.load(open(os.path.join(res, 'cross.json'))) if os.path.exists(os.path.join(res, 'cross.json')) else {}
L = []


def pct(x, d=1):
    return '-' if x is None else ('%.' + str(d) + 'f%%') % (100 * x)


def table(head, rows):
    L.append('| ' + ' | '.join(head) + ' |')
    L.append('|' + '---|' * len(head))
    for r in rows:
        L.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    L.append('')


L.append('# Read collapse and cache study: overnight summary\n')
L.append('Costs use a weight of 30 for a distinct read with NH>1 against 1 for others (heavy lane: 2.2% of distinct reads, about 40% of kernel time). '
         'Verification or patching of a group member costs 10% of mapping it. Reads come from the CPU engine tagged BAM (mapped reads) plus the unmapped FASTQ. '
         'Reads in neither file (for example no valid barcode) are absent. Sequence-level work (Hamming, k-mers) uses a read subset of at most 6 M reads per sample.\n')
if CR:
    table(['sample', 'chem', 'tissue', 'reads in tables', 'distinct'], [[s, CR['meta'][s]['chem'], CR['meta'][s]['tissue'], '%.1fM' % (v['reads'] / 1e6), '%.1fM' % (v['distinct'] / 1e6)] for s, v in CR['sizes'].items()])

L.append('## Q1a. Exact duplicates within a sample\n')
table(['sample', 'reads', 'distinct', 'distinct frac', 'top 1e4 seqs cover', 'top 1e6 seqs cover', 'same seq, different record'],
      [[s, '%.1fM' % (w['q1_within']['reads'] / 1e6), '%.1fM' % (w['q1_within']['distinct'] / 1e6), pct(w['q1_within']['distinct_frac']),
        pct(w['q1_within']['topk_within']['10000']), pct(w['q1_within']['topk_within']['1000000']),
        '%d of %d' % (w['same_sequence_different_record_keys'], w['mapped_distinct_keys'])] for s, w in WI.items()])
L.append('Abundance distribution (share of distinct sequences / share of reads), pbmc1k_v3 if present, else first sample:\n')
ref = 'pbmc1k_v3' if 'pbmc1k_v3' in WI else samples[0]
table(['multiplicity', 'distinct share', 'read share'], [[b['bin'], pct(b['distinct_share']), pct(b['read_share'])] for b in WI[ref]['q1_within']['abundance']])
L.append('Per read class in %s (class is that of the first read of the sequence):\n' % ref)
table(['class', 'read share', 'distinct per read', 'cost share'], [[c['cls'], pct(c['read_share']), '%.3f' % c['distinct_per_read'], pct(c['cost_share'])] for c in WI[ref]['q1_within']['classes']])

summ = {}
if CR:
    L.append('## Q1b. Exact duplicates across samples (held-out sample against a pool of other samples)\n')
    L.append('Hit rate of distinct sequences, reads and cost-weighted distinct sequences. Pool A is same chemistry and tissue, B same chemistry other tissue, C other chemistry same tissue, D neither.\n')
    rows = []
    for r in CR['q1_cross']:
        st = r['stats']['all']
        rows.append([r['heldout'], r['pool'], len(r['pool_samples']), pct(st['reads_hit']), pct(st['distinct_hit']), pct(st['cost_hit']),
                     pct(r['stats'].get('multi_NH>1', {}).get('distinct_hit')), pct(r['stats'].get('simple', {}).get('distinct_hit')),
                     pct(r['stats'].get('softclipped', {}).get('distinct_hit')), pct(r['stats'].get('spliced', {}).get('distinct_hit')),
                     '%d/%d' % (r['record_check']['same_record'], r['record_check']['hits_checked'])])
    table(['held-out', 'pool', 'n', 'reads hit', 'distinct hit', 'cost hit', 'NH>1 hit', 'simple hit', 'clipped hit', 'spliced hit', 'record equal'], rows)
    allr = [r for r in CR['q1_cross'] if r['pool'] == 'ALL_others']
    summ['exact_cache_all_pool_mean_cost_hit'] = float(np.mean([r['stats']['all']['cost_hit'] for r in allr]))
    summ['exact_cache_all_pool_mean_distinct_hit'] = float(np.mean([r['stats']['all']['distinct_hit'] for r in allr]))

    L.append('## Q4. Cache economics\n')
    mb = CR['memory']
    rb = mb['mean_record_bytes']
    L.append('Mean alignment record %.1f bytes (chr, position, strand, score, CIGAR ops, all NH alignments). Entry = key + record + length byte, hash table load factor 0.75.\n' % rb)
    table(['key', 'bytes per entry', 'MB per million entries'],
          [[n, '%.1f' % ((kb + rb + 1) / 0.75), '%.0f' % ((kb + rb + 1) / 0.75 * 1e6 / 1048576)] for n, kb in (('64-bit hash', 8), ('128-bit hash', 16), ('packed 2-bit sequence (91 bases)', 24))])
    L.append('Hit rate against pool size (mean over random pool orders; hit of distinct sequences, reads, cost):\n')
    rows = []
    for r in CR['q4_pool_growth']:
        rows.append([r['heldout'], r['pool_size'], '%.1fM' % (r['pool_entries'] / 1e6), pct(r['all']['reads_hit']), pct(r['all']['distinct_hit']), pct(r['all']['cost_hit']), pct(r.get('multi_NH>1', {}).get('distinct_hit'))])
    table(['held-out', 'pool samples', 'pool entries', 'reads hit', 'distinct hit', 'cost hit', 'NH>1 distinct hit'], rows)

    L.append('## Q3a. Top-K exact sequence table (training: other samples only)\n')
    rows = []
    for r in CR['q3_topk']:
        for K, v in r['K'].items():
            rows.append([r['heldout'], r['training'], K, pct(v['all']['reads_hit']), pct(v['all']['distinct_hit']), pct(v['all']['cost_hit']), pct(v.get('multi_NH>1', {}).get('reads_hit'))])
    table(['held-out', 'training', 'K', 'reads covered', 'distinct covered', 'cost covered', 'NH>1 reads covered'], rows)
    k6 = [r['K']['1000000']['all']['cost_hit'] for r in CR['q3_topk'] if r['training'] == 'all_others' and '1000000' in r['K']]
    summ['topk_1e6_mean_cost_hit'] = float(np.mean(k6)) if k6 else None
    L.append('Dominant genes among the top 1e4 training sequences, and mito, ribosomal and MALAT1 shares:\n')
    rows = []
    for r in CR['q3_topk']:
        if r['training'] != 'all_others':
            continue
        rows.append([r['heldout'], ', '.join('%s %s' % (g['gene'], pct(g['mass_share'], 0)) for g in r['top1e4_genes'][:8]),
                     ' '.join('%s %s' % (k, pct(v, 0)) for k, v in r['top1e4_class_mass'].items()),
                     ' '.join('%s %s' % (k, pct(v, 1)) for k, v in r['heldout_true_class_read_share'].items())])
    table(['held-out', 'top genes (share of top-1e4 mass)', 'top-1e4 class mass', 'held-out true read share'], rows)

    L.append('## Q3b. k-mer classifier (k=25) for mito, ribosomal and MALAT1 origin (output changing; truth is the genome alignment)\n')
    rows = []
    for r in CR['q3_kmer']:
        for t in ('1', '3'):
            for c, v in r['by_threshold'][t].items():
                rows.append([r['heldout'], t, c, pct(v['true_share'], 2), pct(v['routed_share'], 2), pct(v['precision']), pct(v['recall'])])
    table(['held-out', 'min k-mers', 'class', 'true read share', 'routed read share', 'precision', 'recall'], rows)

L.append('## Q2. Tolerance and heuristic grouping (within sample)\n')
L.append('Agreement is measured among covered members against the group representative (most abundant sequence). Strict means chr, position, strand, CIGAR, NH and score. '
         'Relaxed leaves out the score. Net saving counts verification at 10%, a failed verification costs the full mapping again on top. Saving is a share of total mapping cost.\n')
rows = []
for s, w in WI.items():
    h = w.get('q2_hamming')
    if not h:
        continue
    for sch, sv in h['schemes'].items():
        for d, v in sv['by_d'].items():
            if v.get('covered', 0) == 0:
                continue
            rows.append([s, sch, d, pct(v['covered_share_of_distinct']), pct(v['agree_strict']), pct(v['agree_relaxed_no_score']), pct(v['agree_relaxed_simple']), pct(v['agree_relaxed_multi']),
                         pct(v['net_saving_strict_verify10']), pct(v['net_saving_relaxed_verify10'])])
table(['sample', 'bucket', 'HD<=', 'covered of distinct', 'agree strict', 'agree relaxed', 'relaxed simple', 'relaxed NH>1', 'net saving strict', 'net saving relaxed'], rows)
L.append('Same cell barcode and UMI:\n')
rows = []
for s, w in WI.items():
    c = w['q2_cbumi']
    rows.append([s, '%.1fM' % (c['valid_read_distinct_pairs'] / 1e6), pct(c['member_share']), pct(c['agree_strict']), pct(c['agree_relaxed_no_score']), pct(c['agree_locus_50bp']),
                 pct(c['gross_saving_relaxed']), pct(c['net_saving_relaxed_verify10'])])
table(['sample', 'distinct (molecule, seq) pairs', 'non-representative members', 'agree strict', 'agree relaxed', 'same locus 50 bp', 'gross saving relaxed', 'net saving relaxed'], rows)

# ---- ranking
cands = []
if CR:
    cands.append(('Exact cross-sample cache, pool of all other samples', summ['exact_cache_all_pool_mean_cost_hit'], 'exact'))
    if summ.get('topk_1e6_mean_cost_hit') is not None:
        cands.append(('Top-1e6 exact sequence table from other samples', summ['topk_1e6_mean_cost_hit'], 'exact'))
for name, key in (('Hamming grouping, best scheme with HD<=1 (verified)', ('1',)), ('Hamming grouping, best scheme with HD<=2 (verified)', ('2',))):
    vals = []
    for s, w in WI.items():
        h = w.get('q2_hamming')
        if h:
            vals.append(max([sv['by_d'].get(key[0], {}).get('net_saving_relaxed_verify10', 0) or 0 for sv in h['schemes'].values()] + [0]))
    if vals:
        cands.append((name + ', relaxed key with score patch', float(np.mean(vals)), 'exact only if the patch is verified'))
cu = [w['q2_cbumi']['net_saving_relaxed_verify10'] for w in WI.values()]
cands.append(('Same barcode and UMI grouping (verified), relaxed key', float(np.mean(cu)), 'exact only if the patch is verified'))
cands.sort(key=lambda x: -x[1])
L.append('## Ranking by mean share of mapping cost saved while keeping output exact\n')
table(['rank', 'approach', 'mean cost saved', 'output'], [[i + 1, n, pct(v), o] for i, (n, v, o) in enumerate(cands)])
L.append('## Recommendation\n')
if cands:
    L.append('Start with %s. It saves %s of cost on average and needs no change to the aligner. ' % (cands[0][0].lower(), pct(cands[0][1])) +
             'A k-mer routing of mito and ribosomal reads changes output and is not recommended for the byte-identical aligner. '
             'Use its precision as a guide for which sequences a cache should hold first.\n')
json.dump(dict(per_sample=WI, cross=CR, ranking=[dict(approach=n, mean_cost_saved=v, output=o) for n, v, o in cands]), open(os.path.join(out, 'summary.json'), 'w'), indent=1)
open(os.path.join(out, 'summary.md'), 'w').write('\n'.join(L))
print('wrote', out)
