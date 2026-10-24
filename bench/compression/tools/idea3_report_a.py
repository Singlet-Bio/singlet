"""Summarise results/idea3_a_measure.jsonl into tables (printed)."""
import json, sys, os
import pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
rows = [json.loads(l) for l in open(os.path.join(ROOT, 'results', 'idea3_a_measure.jsonl'))]
d = {}
for r in rows: d[(r['sample'], r['kind'])] = r   # last wins
acc = pd.read_csv(os.path.join(ROOT, 'results', 'track_a_account_files.tsv'), sep='\t')
acc['sample_'] = acc['sample']; acc['kind'] = acc['member'].str.replace('_counts.1pz', '', regex=False)
refbits = {}
for _, a in acc.iterrows():
    refbits[(a['sample'], a['kind'])] = (a.chunk_payload + a.ptr_z + a.colsums_z + a.ctable + a.chunk_hdr) * 8
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50)
# 1. structure
t1 = []
for (s, k), r in d.items():
    t1.append(dict(sample=s, kind=k, G=r['G'], ivl_gene_mean=r['ivl_per_gene_mean'], med=r['ivl_per_gene_med'], p90=r['ivl_per_gene_p90'], max=r['ivl_per_gene_max'],
                   umi_w=r['umi_weighted_ivl_per_gene'], single=r['frac_umi_single_ivl_genes'], nnzI=r['nnz_int'], nnzG=r['nnz_gene'], U=r['U'], umi_per_nnzI=r['umi_per_nnz_int'], c1=r['frac_umi_c1']))
print(pd.DataFrame(t1).round(3).to_string(index=False))
# 2. split entropies (bits per UMI)
t2 = []
for (s, k), r in d.items():
    U = r['U']; nI = r['nnz_int']; nG = r['nnz_gene']
    row = dict(sample=s, kind=k, H_in=r['H_I_given_G_in_bits_per_umi'], H_in_cb=r['H_I_given_G_cbucket_in_bits_per_umi'],
               Hc1=r['H_I_given_G_in_c1'], Hc2=r['H_I_given_G_in_c2'], Hc34=r['H_I_given_G_in_c3_4'], Hc5p=r['H_I_given_G_in_c5p'])
    row['ms_in/U'] = r['split_in_multiset_bits'] / U
    row['len/U'] = r['split_len_multiset_bits'] / U
    for key, nm in [('split_oos_a0.25_multiset_bits', 'oos_pool/U'), ('split_oosbest_multiset_bits', 'oos_best1/U'),
                    ('split_adapt_flat_a0.05_multiset_bits', 'adapt_flat.05/U'), ('split_adapt_flat_a0.25_multiset_bits', 'adapt_flat.25/U'),
                    ('split_adapt_lenprior_k8.0_multiset_bits', 'adapt_len8/U'), ('split_adapt_oosbestprior_k8.0_multiset_bits', 'adapt_oosbest8/U')]:
        row[nm] = r.get(key, float('nan')) / U
    row['adapt_flat.05/nnzI'] = r['split_adapt_flat_a0.05_multiset_bits'] / nI
    row['adapt_flat.05/nnzG'] = r['split_adapt_flat_a0.05_multiset_bits'] / nG
    t2.append(row)
print(pd.DataFrame(t2).round(3).to_string(index=False))
# 3. gene vs interval totals (bits per nnz_int)
t3 = []
for (s, k), r in d.items():
    nI = r['nnz_int']
    adapt = min(r['split_adapt_flat_a0.05_multiset_bits'], r['split_adapt_flat_a0.25_multiset_bits'])
    g_self = r['gene_kbits'] + r['gene_pres_in_cls'] + r['gene_cls_side'] + r['gene_cnt_in_cls']
    i_self = r['int_kbits'] + r['int_pres_in_cls'] + r['int_cls_side'] + r['int_cnt_in_cls']
    g_oos = r['gene_kbits'] + r.get('gene_pres_oos_cls_best', float('nan')) + r.get('gene_cnt_oos_cls_best', float('nan'))
    i_oos = r['int_kbits'] + r.get('int_pres_oos_cls_best', float('nan')) + r.get('int_cnt_oos_cls_best', float('nan'))
    t3.append(dict(sample=s, kind=k, ref=refbits.get((s, k), float('nan')) / nI,
                   int_self=i_self / nI, gene_self=g_self / nI, split_adapt=adapt / nI, gene_plus_split_self=(g_self + adapt) / nI,
                   gene_frac_of_total=g_self / (g_self + adapt), split_frac=adapt / (g_self + adapt),
                   int_oos_best=i_oos / nI, gene_oos_best=g_oos / nI, int_floor_bf=(r['int_kbits'] + r['int_pres_in_bf'] + r['int_cnt_in_cls']) / nI,
                   gene_floor_bf=(r['gene_kbits'] + r['gene_pres_in_bf'] + r['gene_cnt_in_cls'] + r['split_in_multiset_bits']) / nI))
print(pd.DataFrame(t3).round(2).to_string(index=False))
