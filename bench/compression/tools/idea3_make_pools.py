"""Write results/idea3_pools.tsv: for every corpus / train file, the leave-one-study-out training pool (same species,
other studies, same matrix kind).  Paths are relative to bench/compression (the bench cwd)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idea3_common import *

rows = []
allids = TEST + TRAIN_H + TRAIN_M
for sid in allids:
    for kind in ('exon', 'intron'):
        d = sample_dir(sid)
        rel = os.path.relpath(os.path.join(d, f'{kind}_counts.1pz'), ROOT).replace('\\', '/')
        pool = []
        for s2 in allids:
            if SPECIES[s2] == SPECIES[sid] and STUDY[s2] != STUDY[sid]:
                pool.append(os.path.relpath(os.path.join(sample_dir(s2), f'{kind}_counts.1pz'), ROOT).replace('\\', '/'))
        rows.append(rel + '\t' + ','.join(pool))
with open(os.path.join(ROOT, 'results', 'idea3_pools.tsv'), 'w') as f:
    f.write('\n'.join(rows) + '\n')
print(len(rows), 'pools written')
