#!/usr/bin/env python
"""Write results/track_c_samples.tsv: colname-hash -> study id for every cached (test + train) file, so codecs can pick the leave-one-study-out model."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_common import *
from track_c_train import fnv_names
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'track_c_samples.tsv')
rows = {}
for k, st in STUDY.items():
    for mem in ('exon', 'intron'):
        p = os.path.join(SCR, f'{k}__{mem}.cols.txt')
        if os.path.exists(p):
            rows[fnv_names(open(p).read().split('\n'))] = st
with open(out, 'w') as f:
    for h, st in rows.items(): f.write(f'{h} {st}\n')
print(len(rows), 'fingerprints')
