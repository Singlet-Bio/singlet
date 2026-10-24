"""Assemble the markdown tables of notes/idea1.md from the result TSVs.  python -P tools/idea1_tables.py > results/idea1_tables.md"""
import csv, os, sys, collections
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
def rd(fn):
    p = os.path.join(R, fn)
    if not os.path.exists(p): return []
    return list(csv.DictReader(open(p), delimiter='\t'))
ORDER = ['pbmc_v2_typical', 'pbmc_v2_shallow', 'pbmc_v3_lowgenes', 'tcell_5p', 'tumor_3p', 'brain_sn_v3_deep', 'brain_sn_v3_intronic']
base = {}
for r in rd('track_a_account_files.tsv'):
    mem = r['member'].replace('_counts.1pz', '')
    base[(r['sample'], mem)] = (float(r['chunk_payload']) + float(r['ptr_z'])) * 8 / float(r['nnz'])
bits = collections.defaultdict(dict)
meta = {}
for fn in ('idea1_run_bits.tsv', 'idea1_run_bits_mouse.tsv', 'idea1_run_s4.tsv'):
    for r in rd(fn):
        mem = r['member'].replace('_counts.1pz', '')
        bits[(r['sample'], mem)][r['codec']] = r
        meta[(r['sample'], mem)] = (int(r['nnz']), int(r['n']))
ver = {}
for fn in ('idea1_verify_human.tsv', 'idea1_verify_mouse.tsv'):
    for r in rd(fn):
        ver[(r['sample'], r['member'], r['codec'])] = r
fl = {(r['sample'], r['member']): r for r in rd('idea1_floatce.tsv')}

def f(x, d=2):
    return '-' if x is None else f'{x:.{d}f}'
def pb(key, codec):
    r = bits[key].get(codec)
    return float(r['payload_bits_per_nnz']) if r else None

print('### Table 1. Payload bits/nnz (names excluded), out of sample unless marked ins\n')
print('| file | nnz (M) | VOCSC zstd-3 chunks | s1 order-0 | s2 f x depth | s3 + counts | s3r rank-1 | s3 ins | s3r ins | model CE (py, s3) | coder - CE | oracle f x d (file, presence only) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for s in ORDER:
    for mem in ('exon', 'intron'):
        k = (s, mem)
        if k not in bits: continue
        v = ver.get((s, mem, 'idea1_s3'))
        o = fl.get(k)
        print(f"| {s} {mem} | {meta[k][0]/1e6:.2f} | {f(base.get(k))} | {f(pb(k,'idea1_s1'))} | {f(pb(k,'idea1_s2'))} | {f(pb(k,'idea1_s3'))} | {f(pb(k,'idea1_s3r'))} | {f(pb(k,'idea1_s3_ins'))} | {f(pb(k,'idea1_s3r_ins'))} | {f(float(v['py_total']) if v else None, 3)} | {f(float(v['cpp_payload'])-float(v['py_total']) if v else None, 3)} | {f(float(o['file_oracle_fxd']) if o else None)} |")

print('\n### Table 2. Presence split and quantisation (bits/nnz, out of sample, s3 / s3r models)\n')
print('| file | float cloglog presence | dense 8-bit presence | rank-1 presence | counts (s3) | bucket side info | coder escapes+rounding vs CE (s3) |')
print('|---|---|---|---|---|---|---|')
for s in ORDER:
    for mem in ('exon', 'intron'):
        o = fl.get((s, mem)); v = ver.get((s, mem, 'idea1_s3'))
        if not o or not v: continue
        print(f"| {s} {mem} | {float(o['float_presence']):.3f} | {float(o['dense_presence']):.3f} | {float(o['rank1_presence']):.3f} | {float(v['py_counts']):.3f} | {float(v['py_side']):.3f} | {float(v['diff_pct']):+.2f}% |")

print('\n### Table 3. Decode speed of the stage-4 decoder (single thread, 4 lanes, min of 3)\n')
print('| file | nnz (M) | s4 dense: wall Mnnz/s | s4 dense: core-cyc/nnz (norm) | s4 dense: Mnnz/s @3 GHz norm | s4r rank-1: wall Mnnz/s | s4r: core-cyc/nnz (norm) | s4r: Mnnz/s @3 GHz norm | payload bits/nnz s4r |')
print('|---|---|---|---|---|---|---|---|---|')
def norm(r):
    pc, cal = float(r['pay_cyc_per_nnz']), float(r['cal_cyc_per_iter'])
    return pc / (cal / 4.0)
for s in ORDER:
    for mem in ('exon', 'intron'):
        k = (s, mem)
        a, b = bits[k].get('idea1_s4'), bits[k].get('idea1_s4r')
        if not a or not b or 'pay_cyc_per_nnz' not in a: continue
        print(f"| {s} {mem} | {meta[k][0]/1e6:.2f} | {float(a['payload_Mnnz_s']):.2f} | {norm(a):.0f} | {3000/norm(a):.1f} | {float(b['payload_Mnnz_s']):.2f} | {norm(b):.0f} | {3000/norm(b):.1f} | {float(b['payload_bits_per_nnz']):.3f} |")

print('\n### Table 4. H1 per file (H1: within 0.2 bits/nnz of the per-feature x depth cross-entropy at >= 20 Mnnz/s; out-of-sample models)\n')
print('| file | CE of shipped dense model (s3, py) | stage-4 dense coder | delta | CE rank-1 (s3r, py) | stage-4 rank-1 coder | delta | size clause | dense Mnnz/s (3 GHz-norm) | rank-1 Mnnz/s (3 GHz-norm) | speed clause |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for s in ORDER:
    for mem in ('exon', 'intron'):
        k = (s, mem)
        a, b = bits[k].get('idea1_s4'), bits[k].get('idea1_s4r')
        v3, v3r = ver.get((s, mem, 'idea1_s3')), ver.get((s, mem, 'idea1_s3r'))
        if not (a and b and v3 and v3r) or 'pay_cyc_per_nnz' not in a: continue
        da = float(a['payload_bits_per_nnz']) - float(v3['py_total'])
        db = float(b['payload_bits_per_nnz']) - float(v3r['py_total'])
        na, nb_ = 3000 / norm(a), 3000 / norm(b)
        print(f"| {s} {mem} | {float(v3['py_total']):.3f} | {float(a['payload_bits_per_nnz']):.3f} | {da:+.3f} | {float(v3r['py_total']):.3f} | {float(b['payload_bits_per_nnz']):.3f} | {db:+.3f} | {'met' if max(da, db) <= 0.2 else 'NOT met'} | {na:.1f} | {nb_:.1f} | {'met' if max(na, nb_) >= 20 else 'NOT met'} |")

ch = {(r['sample'], r['member']): r for r in rd('idea1_choice.tsv')}
s5 = collections.defaultdict(dict)
for r in rd('idea1_run_s5.tsv'):
    s5[(r['sample'], r['member'].replace('_counts.1pz', ''))][r['codec']] = r
if s5:
    print('\n### Table 5. Per-file model selection from a library of single-sample rank-1 models (all from OTHER studies), out of sample\n')
    print('| file | VOCSC zstd-3 chunks | s3r pooled (leave-study-out) | s5r selected | chosen model | gain vs pooled | vs VOCSC | stage-4 speed (3 GHz-norm Mnnz/s) |')
    print('|---|---|---|---|---|---|---|---|')
    for s in ORDER:
        for mem in ('exon', 'intron'):
            k = (s, mem)
            r5 = s5.get(k, {}).get('idea1_s5r')
            if not r5: continue
            p3 = pb(k, 'idea1_s3r')
            x = float(r5['payload_bits_per_nnz'])
            sp = (3000 / norm(r5)) if 'pay_cyc_per_nnz' in r5 and float(r5['pay_cyc_per_nnz']) > 0 else float('nan')
            print(f"| {s} {mem} | {f(base.get(k))} | {f(p3)} | {x:.2f} | {ch[k]['hold'] if k in ch else '-'} | {x - p3:+.2f} | {x - base[k]:+.2f} | {sp:.1f} |")

print('\n### Table 0 (summary). Payload bits/nnz (out of sample), model size, coder-vs-CE gap and speed per file\n')
print('| file | VOCSC zstd-3 | s1 order-0 | s2 + depth | s3 + counts | s5r per-file model | coder - CE (s4/s4r, worst) | s4r Mnnz/s @3 GHz | s4 dense Mnnz/s |')
print('|---|---|---|---|---|---|---|---|---|')
for s in ORDER:
    for mem in ('exon', 'intron'):
        k = (s, mem)
        a, b = bits[k].get('idea1_s4'), bits[k].get('idea1_s4r')
        v3, v3r = ver.get((s, mem, 'idea1_s3')), ver.get((s, mem, 'idea1_s3r'))
        r5 = s5.get(k, {}).get('idea1_s5r')
        if not (a and b and v3 and v3r) or 'pay_cyc_per_nnz' not in a: continue
        da = float(a['payload_bits_per_nnz']) - float(v3['py_total']); db = float(b['payload_bits_per_nnz']) - float(v3r['py_total'])
        print(f"| {s} {mem} | {f(base.get(k))} | {f(pb(k,'idea1_s1'))} | {f(pb(k,'idea1_s2'))} | {f(pb(k,'idea1_s3'))} | {f(float(r5['payload_bits_per_nnz']) if r5 else None)} | {max(da, db):+.3f} | {3000/norm(b):.1f} | {3000/norm(a):.1f} |")
