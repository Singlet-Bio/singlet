"""Train the single-sample candidate models of the per-file model library (rank-1 + count tables, variant s3r):
   results/idea1_models/{org}_{member}_s3r_only_{sample}.bin     python -P tools/idea1_train_only.py [human|mouse] [exon|intron]"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import idea1_train as T
from idea1_data import ALL
orgs = [sys.argv[1]] if len(sys.argv) > 1 else ['human', 'mouse']
members = [sys.argv[2]] if len(sys.argv) > 2 else ['exon', 'intron']
outdir = os.path.join(T.ROOT, 'results', 'idea1_models')
for org in orgs:
    for member in members:
        for sid, (g, o) in ALL.items():
            if o != org: continue
            fn = os.path.join(outdir, f'{org}_{member}_s3r_only_{sid}.bin')
            if os.path.exists(fn): continue
            t0 = time.time()
            blob, info = T.build(org, member, 'only_' + sid, 's3r')
            open(fn, 'wb').write(blob)
            print(fn, len(blob), 'train %.0fs' % (time.time() - t0), flush=True)
