"""Summarise results/idea3_b_measure.jsonl (stage B measurements)."""
import json, os
import pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
rows = {}
for l in open(os.path.join(ROOT, 'results', 'idea3_b_measure.jsonl')):
    r = json.loads(l); rows[r['sample']] = r
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50)
out = []
for s, r in rows.items():
    nG = r['nnz_gene_I']; nI = r['nnz_int_I']
    pres_gain = r['pres_bits_A_cls_b'] - r['pres_bits_B_cls_b_e']
    cnt_gain = r['cnt_bits_A'] - r['cnt_bits_B_e']
    out.append(dict(sample=s, nnzG_I=nG, frac_I_with_exon=r['frac_intron_nz_with_exon'], P_I_e0=r['P_I_given_e'][0], P_I_e1=r['P_I_given_e'][1],
                    P_I_e2_3=r['P_I_given_e'][2], P_I_e4p=r['P_I_given_e'][3],
                    pres_A=r['pres_bits_A_cls_b'] / nG, pres_B=r['pres_bits_B_cls_b_e'] / nG,
                    MIraw=r['MI_raw_bits_total'] / nG,
                    pres_gain_per_nzG=pres_gain / nG, pres_gain_per_nzI=pres_gain / nI, cnt_gain_per_nzG=cnt_gain / nG,
                    total_gain_per_nzG=(pres_gain + cnt_gain) / nG,
                    pres_gain_pct=100 * pres_gain / r['pres_bits_A_cls_b'],
                    cellD_vs_C=(r['pres_bits_C_cls_b_cE'] - r['pres_bits_D_cls_b_cE_e']) / nG,
                    k_I_alone=r['k_bits_I_alone'] / nG, k_I_given_E=r['k_bits_I_given_E_ratio'] / nG, corr=r['corr_log_kE_kI']))
print(pd.DataFrame(out).round(3).to_string(index=False))
