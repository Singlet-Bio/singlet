"""Tables for the idea3 codecs from results/idea3_run*.tsv / .log and results/idea3_joint.tsv -> results/idea3_codec_numbers.md"""
import os, re, sys
import pandas as pd

NL = chr(10)
BS = chr(92)

def to_md(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return NL.join(lines)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R = lambda f: os.path.join(ROOT, 'results', f)
acc = pd.read_csv(R('track_a_account_files.tsv'), sep='\t')
ref = {}
for _, a in acc.iterrows():
    ref[(a['sample'], a['member'])] = (a.chunk_payload + a.ptr_z + a.colsums_z + a.ctable + a.chunk_hdr) * 8.0

files = [f for f in ('idea3_run.tsv', 'idea3_run_oos.tsv', 'idea3_run_extra.tsv') if os.path.exists(R(f))]
d = pd.concat([pd.read_csv(R(f), sep='\t') for f in files], ignore_index=True)

# CPU times are only in the logs for the first set of runs
cpu = {}
pat = re.compile(r'\s+(\S+)\s+\d+ B .*enc\s+(\d+) ms \(cpu\s+(\d+)\)\s+dec\s+(\d+) ms \(cpu\s+(\d+), ([0-9.]+) Mnnz/s cpu\)')
for lf in ('idea3_run.log', 'idea3_run_oos.log', 'idea3_run_extra.log'):
    if not os.path.exists(R(lf)):
        continue
    cur = None
    for line in open(R(lf)):
        if line.startswith('== '):
            parts = line.split()[1].replace(BS, '/').split('/')
            cur = (parts[-2], parts[-1])
        else:
            m = pat.match(line)
            if m and cur:
                cpu[(cur[0], cur[1], m.group(1))] = (float(m.group(3)), float(m.group(5)), float(m.group(6)))
nan = float('nan')
d['enc_cpu_ms'] = [cpu.get((s, m, c), (nan, nan, nan))[0] for s, m, c in zip(d['sample'], d['member'], d['codec'])]
d['dec_cpu_ms_min'] = [cpu.get((s, m, c), (nan, nan, nan))[1] for s, m, c in zip(d['sample'], d['member'], d['codec'])]
d['dec_cpu_Mnnz_s'] = [cpu.get((s, m, c), (nan, nan, nan))[2] for s, m, c in zip(d['sample'], d['member'], d['codec'])]
d['ref_bpn'] = [ref.get((s, m), nan) / n for s, m, n in zip(d['sample'], d['member'], d['nnz'])]
d['vs_ref_%'] = 100 * (d['payload_bits_per_nnz'] / d['ref_bpn'] - 1)
d['K'] = d['k_bits'] / d['nnz']; d['P'] = d['pres_bits'] / d['nnz']; d['C'] = d['cnt_bits'] / d['nnz']; d['S'] = d['split_bits'] / d['nnz']
d['side'] = d['side_bytes'] * 8 / d['nnz']
d['model_MB'] = d['model_bytes'] / 1e6
cols = ['sample', 'member', 'codec', 'nnz', 'ref_bpn', 'payload_bits_per_nnz', 'vs_ref_%', 'K', 'P', 'C', 'S', 'side', 'dec_cpu_Mnnz_s', 'enc_cpu_ms', 'dec_cpu_ms_min', 'matrix_ok', 'model_MB']
md = ['# idea3 codec numbers (payload only; names/colnames excluded for every codec; ref = zstd-3 VOCSC chunk payload + ptr + colsums)' + NL]
md.append(to_md(d[cols].round(3)))
if os.path.exists(R('idea3_joint.tsv')):
    j = pd.read_csv(R('idea3_joint.tsv'), sep='\t')
    md.append(NL + NL + '# joint exon + intron' + NL)
    rows = []
    for s, g in j.groupby('sample'):
        g = g.set_index('codec')
        try:
            e = g.loc['idea3_joint_exon_self']; i0 = g.loc['idea3_joint_intron_indep']; i1 = g.loc['idea3_joint_intron_cond']; i2 = g.loc['idea3_joint_intron_nocond']
        except KeyError:
            continue
        nI = i1['nnz']
        rows.append(dict(sample=s, exon_payload_B=e['payload_bytes'], intron_indep_B=i0['payload_bytes'], intron_nocond_B=i2['payload_bytes'], intron_cond_B=i1['payload_bytes'],
                         saved_B=i0['payload_bytes'] - i1['payload_bytes'], saved_bits_per_intron_nnz=(i0['payload_bytes'] - i1['payload_bytes']) * 8 / nI,
                         saved_pct_of_intron=100 * (i0['payload_bytes'] - i1['payload_bytes']) / i0['payload_bytes'],
                         saved_pct_of_joint=100 * (i0['payload_bytes'] - i1['payload_bytes']) / (e['payload_bytes'] + i0['payload_bytes']),
                         both_ok=int(e['matrix_ok'] and i0['matrix_ok'] and i1['matrix_ok'] and i2['matrix_ok'])))
    rows_j_list = rows
    md.append(to_md(pd.DataFrame(rows).round(3)))
open(R('idea3_codec_numbers.md'), 'w').write(NL.join(md) + NL)
print(NL.join(md))

# ---- compact tables for the notes -> results/idea3_codec_snippet.md
snip = []
self_ = d[d.codec == 'idea3_gene_then_interval'].set_index(['sample', 'member'])
dire = d[d.codec == 'idea3_interval_direct'].set_index(['sample', 'member'])
order = ['pbmc_v2_typical', 'pbmc_v2_shallow', 'pbmc_v3_lowgenes', 'tcell_5p', 'tumor_3p', 'brain_sn_v3_deep', 'brain_sn_v3_intronic']
rows = []
for s in order:
    for m in ('exon_counts.1pz', 'intron_counts.1pz'):
        if (s, m) not in self_.index:
            continue
        a = self_.loc[(s, m)]; b = dire.loc[(s, m)] if (s, m) in dire.index else None
        rows.append({'sample': s, 'matrix': m.split('_')[0], 'nnz': int(a['nnz']), 'today b/nnz': round(a['ref_bpn'], 2), 'gene->interval b/nnz': round(a['payload_bits_per_nnz'], 2),
                     'vs today %': round(a['vs_ref_%'], 1), 'k': round(a['K'], 2), 'which gene': round(a['P'], 2), 'how many': round(a['C'], 2), 'which interval': round(a['S'], 2), 'side': round(a['side'], 2),
                     'interval-direct b/nnz': round(b['payload_bits_per_nnz'], 2) if b is not None else '', 'dec Mnnz/s (cpu)': round(a['dec_cpu_Mnnz_s'], 2)})
snip.append('### C1. Payload bits per interval nonzero (self-described tables, no shipped model; payload only)' + NL)
snip.append(to_md(pd.DataFrame(rows)))
oos_s = d[d.codec == 'idea3_gene_then_interval_oos'].set_index(['sample', 'member'])
oos_a = d[d.codec == 'idea3_gti_oos_adapt'].set_index(['sample', 'member'])
rows = []
for (s, m), a in oos_s.iterrows():
    if (s, m) not in self_.index:
        continue
    b = oos_a.loc[(s, m)] if (s, m) in oos_a.index else None
    rows.append({'sample': s, 'matrix': m.split('_')[0], 'today b/nnz': round(a['ref_bpn'], 2), 'self-described': round(self_.loc[(s, m)]['payload_bits_per_nnz'], 2),
                 'oos static split': round(a['payload_bits_per_nnz'], 2), 'oos + adaptive split': round(b['payload_bits_per_nnz'], 2) if b is not None else '',
                 'oos static vs today %': round(a['vs_ref_%'], 1), 'model MB': round(a['model_MB'], 2)})
snip.append(NL + NL + '### C2. Out-of-sample (leave-one-study-out pooled tables) vs self-described, bits per interval nonzero' + NL)
snip.append(to_md(pd.DataFrame(rows)))
if os.path.exists(R('idea3_joint.tsv')):
    snip.append(NL + NL + '### C3. Intron coded conditionally on the decoded exon matrix (payload bytes; saved = independent - conditional)' + NL)
    snip.append(to_md(pd.DataFrame(rows_j) if False else pd.DataFrame(
        [dict(sample=r['sample'], exon_B=int(r['exon_payload_B']), intron_indep_B=int(r['intron_indep_B']), intron_cond_B=int(r['intron_cond_B']), saved_B=int(r['saved_B']),
              saved_bits_per_intron_nnz=r['saved_bits_per_intron_nnz'], saved_pct_of_intron=r['saved_pct_of_intron'], saved_pct_of_joint=r['saved_pct_of_joint'], both_ok=r['both_ok'])
         for r in sorted(rows_j_list, key=lambda r: order.index(r['sample']))]).round(3)))
open(R('idea3_codec_snippet.md'), 'w').write(NL.join(snip) + NL)
