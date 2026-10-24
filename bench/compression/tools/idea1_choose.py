"""Per-file model selection (what an encoder would do): evaluate the cross-entropy of every candidate model of the library on the
file and pick the best.  Library for test file F (study S) = rank-1 single-sample models of all samples of the same organism
that are NOT from study S, plus the pooled leave-S-out model.  The choice costs ~3 bits per file (index into <= 9 models).
   python -P tools/idea1_choose.py   -> results/idea1_choice.tsv (sample, member, hold-tag) and results/idea1_choice_detail.tsv"""
import sys, os, csv, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import idea1_ce as C
from idea1_data import ALL, TEST, ROOT
M = os.path.join(ROOT, 'results', 'idea1_models')
only = set(sys.argv[1:])            # optional filters like human:exon mouse:intron
CH = os.path.join(ROOT, 'results', 'idea1_choice.tsv')
DT = os.path.join(ROOT, 'results', 'idea1_choice_detail.tsv')
old = {}
if os.path.exists(CH):
    for ln in open(CH).read().splitlines()[1:]:
        a, b, c = ln.split('	'); old[(a, b)] = c
olddet = {}
if os.path.exists(DT):
    for ln in open(DT).read().splitlines()[1:]:
        t = ln.split('	'); olddet[(t[0], t[1])] = t
rows, detail = [], []
for sid, (gse, org) in TEST.items():
    for member in ('exon', 'intron'):
        if only and f'{org}:{member}' not in only: continue
        cands = {}
        for fn in glob.glob(os.path.join(M, f'{org}_{member}_s3r_only_*.bin')):
            other = os.path.basename(fn).split('_only_')[1][:-4]
            if ALL[other][0] == gse: continue
            cands['only_' + other] = fn
        cands[gse] = os.path.join(M, f'{org}_{member}_s3r_{gse}.bin')
        res = {}
        for tag, fn in cands.items():
            try:
                res[tag] = C.ce(fn, sid, member)['total']
            except Exception as e:
                print('skip', sid, member, tag, e)
        best = min(res, key=res.get)
        rows.append((sid, member, best))
        detail.append((sid, member, best, res[best], res.get(gse), len(res)))
        print('%-22s %-6s best %-22s %.3f  pooled %.3f   %s' % (sid, member, best, res[best], res.get(gse, float('nan')),
              ' '.join('%s=%.2f' % (k.replace('only_', ''), v) for k, v in sorted(res.items(), key=lambda kv: kv[1])[:4])), flush=True)
with open(os.path.join(ROOT, 'results', 'idea1_choice.tsv'), 'w') as f:
    f.write('sample\tmember\thold\n')
    for r in rows: f.write('\t'.join(r) + '\n')
with open(os.path.join(ROOT, 'results', 'idea1_choice_detail.tsv'), 'w') as f:
    f.write('sample\tmember\tbest\tbest_bits\tpooled_bits\tn_candidates\n')
    for r in detail: f.write('\t'.join(str(x) for x in r) + '\n')
