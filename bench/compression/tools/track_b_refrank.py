# Does the rank order need to be file-specific? Code each human exon (or intron) file with a rank order derived from the OTHER
# files' pooled row counts (leave-one-out; same reference build, so rows align) and compare the static pos x depth gap cost
# with the own-order cost. The own-order cost does not include the row-count side info that has to be shipped (see
# results/track_b_codecs2.tsv side_bytes); the reference order needs none (it is a shared model).
#   python -P tools/track_b_refrank.py exon|intron
import sys, os, glob
sys.path.insert(0, '../../python'); sys.path.insert(0, 'tools')
import numpy as np
from track_b_models import Mat, gap_bits, gcls, ent

kind = sys.argv[1] if len(sys.argv) > 1 else 'exon'
files = sorted(f for f in glob.glob(f'corpus/*/{kind}_counts.1pz') if 'brain' not in f)
mats = {}
for f in files:
    mats[f] = Mat(f)
    print('loaded', f, mats[f].nnz, flush=True)
# side info per file (bytes) from the codec run
import csv
side = {}
for r in csv.DictReader(open('results/track_b_codecs2.tsv'), delimiter='\t'):
    if r['codec'] == 'b_rans16':
        side[(r['sample'], r['member'])] = float(r['side_bytes'])
print('| file | nnz | own-order gap bits/nnz | side bits/nnz | own + side | leave-one-out reference order | delta vs own+side | reference order by pooled counts (others) |')
print('|---|---|---|---|---|---|---|---|')
for f, M in mats.items():
    others = [mats[g] for g in files if g != f]
    pooled = sum(o.rn / o.nnz for o in others) * M.nnz   # normalise each study to equal weight, scale to this file
    own, _ = gap_bits(M, M.rank_by(M.rn))
    ref, _ = gap_bits(M, M.rank_by(pooled))
    s = side[(os.path.basename(os.path.dirname(f)), os.path.basename(f))] * 8 / M.nnz
    print(f"| {os.path.basename(os.path.dirname(f))}/{kind} | {M.nnz} | {own:.3f} | {s:.3f} | {own + s:.3f} | {ref:.3f} | {ref - own - s:+.3f} | pooled {len(others)} studies |", flush=True)
