"""Train all Idea 1 models needed by the benchmark. Usage: python -P tools/idea1_train_all.py [var,var] [org_member,..] [hold,...]"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import idea1_train as T
vars_ = (sys.argv[1] if len(sys.argv) > 1 else 's1,s2,s3,s3r').split(',')
combos = [('human', 'exon'), ('human', 'intron'), ('mouse', 'exon'), ('mouse', 'intron')]
if len(sys.argv) > 2 and sys.argv[2] != 'all':
    want = set(sys.argv[2].split(','))
    combos = [c for c in combos if f'{c[0]}_{c[1]}' in want]
holds = {'human': ['GSE142444', 'GSE244263', 'GSE152522', 'GSE145370', 'ins'], 'mouse': ['GSE285936', 'ins']}
if len(sys.argv) > 3 and sys.argv[3] != 'all':
    hs = set(sys.argv[3].split(','))
    holds = {k: [h for h in v if h in hs] for k, v in holds.items()}
outdir = os.path.join(T.ROOT, 'results', 'idea1_models')
os.makedirs(outdir, exist_ok=True)
for org, member in combos:
    for hold in holds[org]:
        for var in vars_:
            fn = os.path.join(outdir, f'{org}_{member}_{var}_{hold}.bin')
            if os.path.exists(fn):
                continue
            t0 = time.time()
            blob, info = T.build(org, member, hold, var)
            open(fn, 'wb').write(blob)
            print(fn, len(blob), 'train %.0fs' % (time.time() - t0), flush=True)
