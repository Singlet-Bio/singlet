#!/usr/bin/env python3
import sys, re
sys.path.insert(0, '/mnt/home/debruinz/Singlet-AI/singlepress')
import singlepress as sp
pz = sp.read_1pz_int('/mnt/projects/debruinz_project/singlify_validation/singlify_out/SRR32855204_n6/exon_counts.1pz')
rn = list(pz.rownames)
chroms = {}
for r in rn:
    m = re.search(r'_([^_:]+):\d', r)
    if m:
        c = m.group(1)
        chroms[c] = chroms.get(c, 0) + 1
print('Chromosomes (top 20):')
for c in sorted(chroms, key=lambda x: chroms[x], reverse=True)[:20]:
    print(f'  {c}: {chroms[c]}')
xist = [r for r in rn if re.search(r'_XIST_', r)]
ddx = [r for r in rn if re.search(r'_DDX3Y_', r)]
uty = [r for r in rn if re.search(r'_UTY_', r)]
print(f'XIST features: {len(xist)} DDX3Y: {len(ddx)} UTY: {len(uty)}')
if xist: print('  XIST[0]:', xist[0])
if ddx: print('  DDX3Y[0]:', ddx[0])
