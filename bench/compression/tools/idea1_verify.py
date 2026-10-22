"""Compare the C++ coder's payload bits/nnz (harness TSV) with the Python cross-entropy of the same model file.
   python -P tools/idea1_verify.py OUT.tsv IN1.tsv [IN2.tsv ...]"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import idea1_ce as C
from idea1_data import ROOT

def model_path(row):
    om = {310797: 'human_exon', 272191: 'human_intron', 386028: 'mouse_exon', 307730: 'mouse_intron'}[int(row['m'])]
    c = row['codec'].replace('idea1_', '')
    ins = c.endswith('_ins'); c = c.replace('_ins', '')
    var = {'s4': 's3', 's4r': 's3r'}.get(c, c)
    hold = 'ins' if ins else row['hold']
    return os.path.join(ROOT, 'results', 'idea1_models', f'{om}_{var}_{hold}.bin')

out = sys.argv[1]
seen = set()
rows_out = []
for fn in sys.argv[2:]:
    for r in csv.DictReader(open(fn), delimiter='\t'):
        key = (r['sample'], r['member'], r['codec'])
        if key in seen: continue
        seen.add(key)
        try:
            mp = model_path(r)
            res = C.ce(mp, r['sample'], r['member'].replace('_counts.1pz', ''))
        except Exception as e:
            print('skip', key, e); continue
        cpp = float(r['payload_bits_per_nnz'])
        rows_out.append(dict(sample=r['sample'], member=r['member'].replace('_counts.1pz', ''), codec=r['codec'], hold=r['hold'],
                             nnz=r['nnz'], cpp_payload=cpp, py_total=res['total'], diff_pct=100.0 * (cpp - res['total']) / res['total'],
                             py_presence=res['presence'], py_counts=res['counts'], py_side=res['side'], model_bytes=res['model_bytes'],
                             matrix_ok=r['matrix_ok']))
        print('%-18s %-7s %-15s cpp %8.3f  py %8.3f  diff %+6.2f%%' % (r['sample'], r['member'][:6], r['codec'], cpp, res['total'], rows_out[-1]['diff_pct']), flush=True)
with open(out, 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()), delimiter='\t')
    w.writeheader(); w.writerows(rows_out)
